//! Regression tests for issue #238: `compute_extra_output_columns` must
//! thread each observation's **occasion** kappa into `pk_param_fn` and into
//! `DerivedContext.eta`, instead of silently using a BSV-only eta vector
//! (kappas → 0). Both the individual-parameter map (driving `[derived]`
//! expressions and `[output]` columns) and `ctx.eta` are checked.

use super::*;
use crate::types::{
    BloqMethod, CompiledModel, DerivedContext, DerivedExprSpec, DerivedKind, DoseEvent, ErrorModel,
    ErrorSpec, GradientMethod, IndivParamPartials, ModelParameters, OmegaMatrix, PkModel, PkParams,
    Population, ScalingSpec, SigmaVector, Subject, PK_IDX_LAGTIME,
};
use nalgebra::DVector;
use std::collections::HashMap;

/// 1-cpt IV model with one BSV eta (`ETA_CL`) and one IOV kappa
/// (`KAPPA_CL`). CL = 10 · exp(κ); the kappa is read from `eta[1]` with a
/// `.get(1)` guard, so the *broken* (BSV-only) path would read κ=0 → CL=10
/// for every observation, while the fix yields the per-occasion CL.
fn minimal_iov_model(derived_exprs: Vec<DerivedExprSpec>) -> CompiledModel {
    CompiledModel {
        frem_config: None,
        residual_error_eta: None,
        analytical_init: Vec::new(),
        analytic_readout: None,
        ruv_magnitude: None,
        absorption_ode_equivalent: None,
        mixture: None,
        name: "test_iov_kappa".into(),
        pk_model: PkModel::OneCptIv,
        error_model: ErrorModel::Additive,
        error_spec: ErrorSpec::Single(ErrorModel::Additive),
        residual_correlations: Vec::new(),
        pk_param_fn: Box::new(
            |_theta: &[f64], eta: &[f64], _cov: &HashMap<String, f64>, _t: f64| {
                let kappa = eta.get(1).copied().unwrap_or(0.0);
                let mut p = PkParams::default();
                p.values[0] = 10.0 * kappa.exp(); // CL slot
                p
            },
        ),
        n_theta: 0,
        n_eta: 1,
        n_epsilon: 1,
        n_kappa: 1,
        kappa_names: vec!["KAPPA_CL".into()],
        theta_names: Vec::new(),
        eta_names: vec!["ETA_CL".into()],
        indiv_param_names: vec!["CL".into()],
        indiv_param_partials: IndivParamPartials::empty(),
        default_params: ModelParameters {
            theta: Vec::new(),
            theta_names: Vec::new(),
            theta_lower: Vec::new(),
            theta_upper: Vec::new(),
            theta_fixed: Vec::new(),
            omega: OmegaMatrix::from_diagonal(&[1.0], vec!["ETA_CL".into()]),
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: Some(OmegaMatrix::from_diagonal(&[1.0], vec!["KAPPA_CL".into()])),
            kappa_fixed: vec![false],
        },
        omega_init_as_sd: vec![false],
        sigma_init_as_sd: vec![false],
        kappa_init_as_sd: vec![false],
        mu_refs: HashMap::new(),
        kappa_mu_refs: HashMap::new(),
        tv_fn: Some(Box::new(|_t, _c| vec![])),
        pk_indices: vec![0],
        eta_map: Vec::new(),
        pk_idx_f64: Vec::new(),
        sel_flat: Vec::new(),
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
        derived_exprs,
        output_columns: Vec::new(),
        #[cfg(feature = "survival")]
        endpoints: std::collections::HashMap::new(),
    }
}

/// Subject with two occasions: obs 0,1 on occasion 1; obs 2,3 on occasion 2.
fn two_occasion_subject() -> Subject {
    Subject {
        fremtype: Vec::new(),
        id: "S1".into(),
        doses: Vec::new(),
        obs_times: vec![0.0, 1.0, 2.0, 3.0],
        obs_raw_times: vec![0.0, 1.0, 2.0, 3.0],
        observations: vec![1.0; 4],
        obs_cmts: vec![1; 4],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        obs_records: vec![],
    }
}

fn sr_iov(n_obs: usize) -> SubjectResult {
    SubjectResult {
        id: "S1".into(),
        eta: DVector::from_vec(vec![0.0]), // BSV η = 0 so CL is driven purely by κ
        ipred: vec![1.0; n_obs],
        pred: vec![1.0; n_obs],
        iwres: vec![0.0; n_obs],
        cwres: vec![0.0; n_obs],
        npde: vec![],
        npd: vec![],
        ofv_contribution: 0.0,
        cens: vec![0; n_obs],
        n_obs,
        extra_columns: Vec::new(),
        per_obs_tad: Vec::new(),
        compartment_states: Vec::new(),
        #[cfg(feature = "survival")]
        discrete_rows: Vec::new(),
    }
}

/// Both the indiv-param map (CL, via `pk_param_fn`) and `ctx.eta` must carry
/// the per-observation occasion kappa. With κ₁ = ln 2 (occasion 1) and
/// κ₂ = ln 3 (occasion 2), CL = 10·exp(κ) = [20, 20, 30, 30] and the kappa
/// exposed through `ctx.eta[1]` = [ln2, ln2, ln3, ln3]. The pre-fix code
/// produced CL = 10 for every row (κ silently 0).
#[test]
fn derived_and_indiv_use_per_occasion_kappa() {
    let ln2 = 2.0_f64.ln();
    let ln3 = 3.0_f64.ln();

    let derived_exprs = vec![
        // CL_OUT exercises the pk_param_fn call that builds per_obs_indiv.
        DerivedExprSpec {
            name: "CL_OUT".into(),
            kind: DerivedKind::PerRow {
                eval: Box::new(|ctx: &DerivedContext| {
                    ctx.indiv_params.get("CL").copied().unwrap_or(f64::NAN)
                }),
            },
            uses_compartments: false,
        },
        // K_OUT exercises DerivedContext.eta threading (eta[1] = occasion κ).
        DerivedExprSpec {
            name: "K_OUT".into(),
            kind: DerivedKind::PerRow {
                eval: Box::new(|ctx: &DerivedContext| ctx.eta.get(1).copied().unwrap_or(-1.0)),
            },
            uses_compartments: false,
        },
    ];

    let model = minimal_iov_model(derived_exprs);
    let population = Population {
        subjects: vec![two_occasion_subject()],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    // ebe_kappas[subject][occasion]; occasion order matches split_obs_by_occasion
    // (first-seen): occasion 1 → index 0 (κ=ln2), occasion 2 → index 1 (κ=ln3).
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![
        DVector::from_vec(vec![ln2]),
        DVector::from_vec(vec![ln3]),
    ]];
    let mut subjects_results = vec![sr_iov(4)];

    compute_extra_output_columns(&model, &population, &[], &kappas, &mut subjects_results);

    let cols = &subjects_results[0].extra_columns;
    let cl = &cols.iter().find(|(n, _)| n == "CL_OUT").unwrap().1;
    let kout = &cols.iter().find(|(n, _)| n == "K_OUT").unwrap().1;

    let expected_cl = [20.0, 20.0, 30.0, 30.0];
    let expected_k = [ln2, ln2, ln3, ln3];
    for (j, (&got, &exp)) in cl.iter().zip(expected_cl.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-9,
            "CL_OUT[{j}] (occasion {}): got {got}, expected {exp}",
            if j < 2 { 1 } else { 2 }
        );
    }
    for (j, (&got, &exp)) in kout.iter().zip(expected_k.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-12,
            "ctx.eta[1] at obs {j}: got {got}, expected occasion κ {exp}"
        );
    }
}

/// Defensive path: when a subject's kappa vector is missing (e.g. fewer
/// occasion entries than occasions seen), the kappa slots fall back to 0
/// rather than panicking — CL collapses to the κ=0 value (10).
#[test]
fn missing_kappa_falls_back_to_zero() {
    let derived_exprs = vec![DerivedExprSpec {
        name: "CL_OUT".into(),
        kind: DerivedKind::PerRow {
            eval: Box::new(|ctx: &DerivedContext| {
                ctx.indiv_params.get("CL").copied().unwrap_or(f64::NAN)
            }),
        },
        uses_compartments: false,
    }];
    let model = minimal_iov_model(derived_exprs);
    let population = Population {
        subjects: vec![two_occasion_subject()],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    // Empty kappas for the subject → every occasion lookup misses → κ=0.
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]];
    let mut subjects_results = vec![sr_iov(4)];

    compute_extra_output_columns(&model, &population, &[], &kappas, &mut subjects_results);

    let cl = &subjects_results[0].extra_columns[0].1;
    for (j, &v) in cl.iter().enumerate() {
        assert!(
            (v - 10.0).abs() < 1e-9,
            "CL_OUT[{j}] with missing kappa should fall back to κ=0 → 10, got {v}"
        );
    }
}

/// Caller-level regression for the per-dose-occasion absorption lag
/// (follow-up to #238). `compute_extra_output_columns` must build TAD using
/// each *dose's* occasion lag, not the observation's. The model puts IOV on
/// the lag (`lag = 1.0 + κ`); a subject is dosed BID across two occasions:
///   morning dose @0  (occasion 1, κ=0.0 → lag 1.0)
///   evening dose @12 (occasion 2, κ=0.5 → lag 1.5)
/// with observations @2 (occ 1) and @13 (occ 2). At obs @13 the evening dose
/// arrives at 13.5 — not yet absorbed — so TAD counts from the morning dose's
/// arrival at 1.0 → 12.0. Applying the obs-occasion lag (1.5) to every dose,
/// as before this follow-up, would give 11.5.
#[test]
fn tad_uses_per_dose_occasion_lag() {
    let mut model = minimal_iov_model(vec![]);
    // Declare an absorption lag (ALAG → PK_IDX_LAGTIME) so `model.has_lagtime()`
    // holds; compute_extra_output_columns only builds per-dose lags for models
    // that declare a lag.
    model.indiv_param_names = vec!["CL".into(), "ALAG".into()];
    model.pk_indices = vec![0, PK_IDX_LAGTIME];
    // Drive the absorption lag (slot PK_IDX_LAGTIME) from the occasion kappa.
    model.pk_param_fn = Box::new(
        |_theta: &[f64], eta: &[f64], _cov: &HashMap<String, f64>, _t: f64| {
            let kappa = eta.get(1).copied().unwrap_or(0.0);
            let mut p = PkParams::default();
            p.values[PK_IDX_LAGTIME] = 1.0 + kappa;
            p
        },
    );

    let subject = Subject {
        fremtype: Vec::new(),
        id: "S1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![2.0, 13.0],
        obs_raw_times: vec![2.0, 13.0],
        observations: vec![1.0, 1.0],
        obs_cmts: vec![1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0, 0],
        occasions: vec![1, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    // ebe_kappas in split_obs_by_occasion order: occ 1 → idx 0 (κ=0.0),
    // occ 2 → idx 1 (κ=0.5).
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![
        DVector::from_vec(vec![0.0]),
        DVector::from_vec(vec![0.5]),
    ]];
    let mut subjects_results = vec![sr_iov(2)];

    compute_extra_output_columns(&model, &population, &[], &kappas, &mut subjects_results);

    let tad = &subjects_results[0].per_obs_tad;
    assert!(
        (tad[0] - 1.0).abs() < 1e-9,
        "obs@2 TAD: morning dose arrives @1.0 → 1.0, got {}",
        tad[0]
    );
    assert!(
        (tad[1] - 12.0).abs() < 1e-9,
        "obs@13 TAD must be 12.0 (evening dose uses its own occ-2 lag 1.5 → arrives \
             @13.5, excluded; counts from morning @1.0). The pre-follow-up obs-occasion \
             lag would give 11.5. Got {}",
        tad[1]
    );
}

/// Regression for the per-compartment TAD lag (issue #369). On an ODE model
/// declaring `ALAGn`, the TAD column must anchor each dose on *its own*
/// compartment's lag — resolved through `dose_attr_map`, the same single
/// source of truth the prediction paths use — not the bare `PK_IDX_LAGTIME`
/// slot, which a model declaring only `ALAG2` leaves at 0. A dose into
/// compartment 2 at t=0 with `ALAG2 = 2` effectively arrives at t=2, so an
/// observation at t=3 has TAD = 1.0. The pre-fix code read the bare lag (0)
/// and reported TAD = 3.0.
#[test]
fn tad_uses_per_compartment_alag_on_ode_models() {
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVLAG2(2.0, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  ALAG2 = TVLAG2

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -CL/V * depot
  d/dt(central) =  CL/V * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP)
"#;
    let model = crate::parser::model_parser::parse_full_model(src)
        .expect("parse ok")
        .model;
    assert!(model.has_lagtime(), "ALAG2 must enable has_lagtime()");

    // Dose into compartment 2 (central) at t=0; observe cmt 2 at t=3.
    let subject = Subject {
        fremtype: Vec::new(),
        id: "S1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0)],
        obs_times: vec![3.0],
        obs_raw_times: vec![3.0],
        observations: vec![1.0],
        obs_cmts: vec![2],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0],
        occasions: vec![1],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let theta = model.default_params.theta.clone();
    let kappas: Vec<Vec<DVector<f64>>> = Vec::new(); // no IOV
    let mut subjects_results = vec![sr_iov(1)];

    compute_extra_output_columns(&model, &population, &theta, &kappas, &mut subjects_results);

    let tad = &subjects_results[0].per_obs_tad;
    // 3 − (dose@0 + ALAG2=2) = 1.0; the pre-fix bare-lag path gives 3.0.
    assert!((tad[0] - 1.0).abs() < 1e-9, "tad: {}", tad[0]);
}

/// Regression for issue #369: a negative compartment-indexed `ALAGn` must
/// raise `W_NEGATIVE_LAGTIME`. The bare-slot check alone never sees the
/// `ALAG2` spare slot, so a bad per-route lag would otherwise slip through.
#[test]
fn negative_alag_emits_negative_lagtime_warning() {
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVLAG2(-1.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  ALAG2 = TVLAG2

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -CL/V * depot
  d/dt(central) =  CL/V * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP)
"#;
    let model = crate::parser::model_parser::parse_full_model(src)
        .expect("parse ok")
        .model;

    let subject = Subject {
        fremtype: Vec::new(),
        id: "S1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0)],
        obs_times: vec![3.0],
        obs_raw_times: vec![3.0],
        observations: vec![1.0],
        obs_cmts: vec![2],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0],
        occasions: vec![1],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };

    let diags = check_model_data_warnings(&model, &population, &model.default_params);
    let neg = diags
        .iter()
        .find(|d| d.code == "W_NEGATIVE_LAGTIME")
        .expect("a negative ALAG2 must raise W_NEGATIVE_LAGTIME");
    assert!(
        neg.message.contains("ALAG2") && neg.message.contains("compartment-2"),
        "warning must name the offending compartment-indexed lag, got: {}",
        neg.message
    );
}

/// A negative **per-route** lag (`fn(..., lag=L)`) must raise `W_NEGATIVE_LAGTIME`
/// too. A per-route lag lives on the forcing's `lag_slot`, NOT the dose-attribute
/// lag machinery, so `model.has_lagtime()` is false and the compartment-lag scan
/// never sees it — its own gate (any forcing with a `lag_slot`) is what fires here.
#[test]
fn negative_per_route_lag_emits_negative_lagtime_warning() {
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVLAG(-1.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  LAG = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
"#;
    let model = crate::parser::model_parser::parse_full_model(src)
        .expect("parse ok")
        .model;
    let subject = Subject {
        fremtype: Vec::new(),
        id: "S1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![3.0],
        obs_raw_times: vec![3.0],
        observations: vec![1.0],
        obs_cmts: vec![1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0],
        occasions: vec![1],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let diags = check_model_data_warnings(&model, &population, &model.default_params);
    let neg = diags
        .iter()
        .find(|d| d.code == "W_NEGATIVE_LAGTIME")
        .expect("a negative per-route lag must raise W_NEGATIVE_LAGTIME");
    assert!(
        neg.message.contains("Per-route") && neg.message.contains("FirstOrder"),
        "warning must name the per-route lag and its route kind, got: {}",
        neg.message
    );
}

/// The per-compartment negative-lag scan is ODE-only. An analytical model can
/// still report `has_lagtime()` (lag bound via `pk_indices`), so
/// `check_model_data_warnings` must take the `ode_spec == None` path: the bare
/// negative lag still warns, but no compartment-indexed `ALAGn` is emitted.
#[test]
fn negative_lag_scan_skips_analytical_models() {
    let mut model = minimal_iov_model(vec![]);
    // Analytical (ode_spec stays None); has_lagtime() via pk_indices carrying
    // PK_IDX_LAGTIME; bare lag evaluates negative.
    model.indiv_param_names = vec!["CL".into(), "LAGTIME".into()];
    model.pk_indices = vec![0, PK_IDX_LAGTIME];
    model.pk_param_fn = Box::new(
        |_t: &[f64], _e: &[f64], _c: &HashMap<String, f64>, _time: f64| {
            let mut p = PkParams::default();
            p.values[PK_IDX_LAGTIME] = -1.0;
            p
        },
    );
    assert!(model.has_lagtime() && model.ode_spec.is_none());

    let subject = Subject {
        fremtype: Vec::new(),
        id: "S1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0],
        obs_raw_times: vec![1.0],
        observations: vec![1.0],
        obs_cmts: vec![1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0],
        occasions: vec![1],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };

    let diags = check_model_data_warnings(&model, &population, &model.default_params);
    assert!(
        diags.iter().any(|d| d.code == "W_NEGATIVE_LAGTIME"),
        "bare negative lag must still warn on an analytical model"
    );
    assert!(
        !diags.iter().any(|d| d.message.contains("ALAG")),
        "no compartment-indexed ALAGn warning for an analytical (non-ODE) model"
    );
}

// ── #830: block_sigma L2 pairing warnings ───────────────────────────────

/// A two-endpoint block_sigma model correlating cmt-1 (slot 0) with cmt-2
/// (slot 1) via ρ.
fn block_sigma_two_endpoint_model() -> CompiledModel {
    let mut m = minimal_iov_model(vec![]);
    m.error_spec = ErrorSpec::PerCmt(HashMap::from([
        (
            1,
            crate::types::EndpointError {
                error_model: ErrorModel::Proportional,
                sigma_idx: vec![0],
            },
        ),
        (
            2,
            crate::types::EndpointError {
                error_model: ErrorModel::Proportional,
                sigma_idx: vec![1],
            },
        ),
    ]));
    m.error_model = ErrorModel::Proportional;
    m.n_epsilon = 2;
    m.default_params.sigma = SigmaVector {
        values: vec![0.3, 0.4],
        names: vec!["S1".into(), "S2".into()],
    };
    m.default_params.sigma_fixed = vec![false, false];
    m.residual_correlations = vec![crate::types::ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 1,
        rho: 0.5,
    }];
    m
}

fn obs_only_subject(cmts: Vec<usize>, times: Vec<f64>, l2: Vec<i64>) -> Subject {
    let n = cmts.len();
    Subject {
        fremtype: Vec::new(),
        id: "S1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: times.clone(),
        obs_raw_times: times,
        observations: vec![1.0; n],
        obs_cmts: cmts,
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: l2,
        dose_occasions: vec![1],
        obs_records: vec![],
    }
}

fn population_of(subject: Subject) -> Population {
    Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// Four co-temporal rows (total1, unbound1, total2, unbound2) with a
/// block_sigma correlation and NO L2 column: a total row can pair with either
/// unbound row, so the greedy row-order pairing is ambiguous and must warn.
#[test]
fn block_sigma_ambiguous_pairing_without_l2_warns() {
    let model = block_sigma_two_endpoint_model();
    let pop = population_of(obs_only_subject(
        vec![1, 2, 1, 2],
        vec![1.0, 1.0, 1.0, 1.0],
        Vec::new(),
    ));
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    assert!(
        diags.iter().any(|d| d.code == "W_BLOCK_SIGMA_L2_ORDER"),
        "ambiguous co-temporal pairing with no L2 must warn"
    );
}

/// The same four rows grouped by an explicit L2 id are unambiguous — no warning.
#[test]
fn block_sigma_l2_grouped_pairing_does_not_warn() {
    let model = block_sigma_two_endpoint_model();
    let pop = population_of(obs_only_subject(
        vec![1, 2, 1, 2],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![1, 1, 2, 2],
    ));
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    assert!(
        !diags.iter().any(|d| d.code == "W_BLOCK_SIGMA_L2_ORDER"),
        "explicit L2 grouping is order-independent — must not warn"
    );
}

/// A clean two-row draw (one total + one unbound) has only one possible
/// pairing even without L2 — no warning.
#[test]
fn block_sigma_two_row_layout_does_not_warn() {
    let model = block_sigma_two_endpoint_model();
    let pop = population_of(obs_only_subject(vec![1, 2], vec![1.0, 1.0], Vec::new()));
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    assert!(
        !diags.iter().any(|d| d.code == "W_BLOCK_SIGMA_L2_ORDER"),
        "an unambiguous two-row draw must not warn"
    );
}

/// An L2 column present while the model declares no block_sigma correlation
/// means L2 is inert (and a covariate of that name was silently reserved) —
/// warn W_L2_UNUSED.
#[test]
fn l2_column_without_block_sigma_warns_unused() {
    let model = minimal_iov_model(vec![]); // no residual_correlations
    let pop = population_of(obs_only_subject(vec![1], vec![1.0], vec![0]));
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    assert!(
        diags.iter().any(|d| d.code == "W_L2_UNUSED"),
        "an L2 column unused by any block_sigma model must warn"
    );
}

/// No L2 column and no block_sigma: nothing to warn about.
#[test]
fn no_l2_column_no_block_sigma_is_quiet() {
    let model = minimal_iov_model(vec![]);
    let pop = population_of(obs_only_subject(vec![1], vec![1.0], Vec::new()));
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    assert!(!diags.iter().any(|d| d.code == "W_L2_UNUSED"));
    assert!(!diags.iter().any(|d| d.code == "W_BLOCK_SIGMA_L2_ORDER"));
}

/// A block_sigma model whose data DOES carry an L2 column must not raise
/// W_L2_UNUSED — the column is consumed.
#[test]
fn block_sigma_with_l2_is_not_flagged_unused() {
    let model = block_sigma_two_endpoint_model();
    let pop = population_of(obs_only_subject(vec![1, 2], vec![1.0, 1.0], vec![1, 1]));
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    assert!(!diags.iter().any(|d| d.code == "W_L2_UNUSED"));
}

/// An SS=1 dose combined with an [initial_conditions] baseline double-counts
/// the starting amount (the SS closed form already pre-equilibrates an
/// infinite dose history). `check_model_data_warnings` must surface it
/// (W_STEADY_STATE_INIT) rather than silently superpose the two (#521 review).
#[test]
fn ss_dose_with_analytical_init_warns() {
    let src = r#"
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[initial_conditions]
  init(central) = 30 * V

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = crate::parser::model_parser::parse_full_model(src)
        .expect("parse ok")
        .model;
    assert_eq!(model.analytical_init.len(), 1);

    // Subject with a steady-state dose (ss=true, ii=24).
    let subject = Subject {
        fremtype: Vec::new(),
        id: "S1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)],
        obs_times: vec![1.0, 6.0],
        obs_raw_times: vec![1.0, 6.0],
        observations: vec![1.0, 1.0],
        obs_cmts: vec![1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0, 0],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };

    let diags = check_model_data_warnings(&model, &population, &model.default_params);
    assert!(
        diags.iter().any(|d| d.code == "W_STEADY_STATE_INIT"),
        "SS dose + [initial_conditions] must raise W_STEADY_STATE_INIT; got: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

/// No SS dose → no W_STEADY_STATE_INIT, even with an init baseline present.
#[test]
fn non_ss_dose_with_analytical_init_does_not_warn() {
    let src = r#"
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[initial_conditions]
  init(central) = 30 * V

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = crate::parser::model_parser::parse_full_model(src)
        .expect("parse ok")
        .model;
    let subject = Subject {
        fremtype: Vec::new(),
        id: "S1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0],
        obs_raw_times: vec![1.0],
        observations: vec![1.0],
        obs_cmts: vec![1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let diags = check_model_data_warnings(&model, &population, &model.default_params);
    assert!(
        !diags.iter().any(|d| d.code == "W_STEADY_STATE_INIT"),
        "no SS dose → no W_STEADY_STATE_INIT"
    );
}

/// Caller-level regression: the per-dose lag uses each dose's *covariate*
/// snapshot (`dose_cov`), not the observation's — so a lag depending on a
/// time-varying covariate is shifted by conditions at dosing, matching
/// `predict_iov`. Lag = `LAGCOV`: the dose sees `LAGCOV=1` (lag 1.0), the
/// observation sees `LAGCOV=5` (lag 5.0). With a dose @0 and obs @3, the dose
/// covariate gives arrival @1.0 → TAD 2.0; the obs covariate would push arrival
/// to @5.0 (after the obs) → TAD NaN. Asserting TAD = 2.0 proves `dose_cov` is
/// used. (No IOV here — this isolates the covariate basis, not kappa.)
#[test]
fn tad_lag_uses_dose_covariate_not_obs() {
    let mut model = minimal_iov_model(vec![]);
    model.indiv_param_names = vec!["CL".into(), "ALAG".into()];
    model.pk_indices = vec![0, PK_IDX_LAGTIME];
    // Lag = LAGCOV covariate (time-varying); independent of eta/kappa.
    model.pk_param_fn = Box::new(
        |_theta: &[f64], _eta: &[f64], cov: &HashMap<String, f64>, _t: f64| {
            let mut p = PkParams::default();
            p.values[PK_IDX_LAGTIME] = cov.get("LAGCOV").copied().unwrap_or(0.0);
            p
        },
    );

    let cov_dose = HashMap::from([("LAGCOV".to_string(), 1.0)]);
    let cov_obs = HashMap::from([("LAGCOV".to_string(), 5.0)]);
    let subject = Subject {
        fremtype: Vec::new(),
        id: "S1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![3.0],
        obs_raw_times: vec![3.0],
        observations: vec![1.0],
        obs_cmts: vec![1],
        covariates: HashMap::new(),
        dose_covariates: vec![cov_dose],
        obs_covariates: vec![cov_obs],
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0],
        occasions: vec![1],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![subject],
        covariate_names: vec!["LAGCOV".into()],
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![DVector::from_vec(vec![0.0])]];
    let mut subjects_results = vec![sr_iov(1)];

    compute_extra_output_columns(&model, &population, &[], &kappas, &mut subjects_results);

    let tad = &subjects_results[0].per_obs_tad;
    assert!(
        (tad[0] - 2.0).abs() < 1e-9,
        "TAD must use the dose covariate (LAGCOV=1 → lag 1.0 → arrival @1.0 → TAD 2.0); \
             the obs covariate (LAGCOV=5) would push arrival to @5.0 (after the obs) → NaN. \
             Got {}",
        tad[0]
    );
}

/// #847: `check_model_data_warnings` flags a combined error model whose
/// **additive** SD initial estimate is negligible relative to the data scale
/// (median |DV|) on that endpoint — the pre-fit guard against the additive-
/// collapse local minimum. Exercises the endpoint-enumeration + observation-
/// gathering plumbing (the median/threshold decision is unit-tested separately
/// in `additive_init_scale_tests`).
fn combined_endpoint_model() -> CompiledModel {
    let mut m = minimal_iov_model(vec![]);
    m.error_spec = ErrorSpec::PerCmt(HashMap::from([(
        1,
        crate::types::EndpointError {
            error_model: ErrorModel::Combined,
            sigma_idx: vec![0, 1], // [proportional, additive]
        },
    )]));
    m.error_model = ErrorModel::Combined;
    m.n_epsilon = 2;
    m.default_params.sigma = SigmaVector {
        values: vec![0.2, 0.5], // additive SD 0.5 ≪ 1% of median |DV| (2.0)
        names: vec!["PROP".into(), "ADD".into()],
    };
    m.default_params.sigma_fixed = vec![false, false];
    m
}

fn subject_with_obs(cmt: usize, dvs: Vec<f64>) -> Subject {
    let n = dvs.len();
    Subject {
        fremtype: Vec::new(),
        id: "S1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: (0..n).map(|i| (i + 1) as f64).collect(),
        obs_raw_times: (0..n).map(|i| (i + 1) as f64).collect(),
        observations: dvs,
        obs_cmts: vec![cmt; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        obs_records: vec![],
    }
}

#[test]
fn additive_init_scale_warns_end_to_end() {
    let model = combined_endpoint_model();
    // median |DV| = 200 → 1% threshold = 2.0; additive SD init 0.5 < 2.0 → warn.
    let pop = population_of(subject_with_obs(1, vec![100.0, 200.0, 300.0]));
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    let d = diags
        .iter()
        .find(|d| d.code == "W_ADDITIVE_INIT_SCALE")
        .expect("negligible additive init on a combined endpoint must warn");
    assert!(d.message.contains("CMT=1"), "message: {}", d.message);
    assert!(d.message.contains("ADD"), "message: {}", d.message);
}

#[test]
fn additive_init_scale_silent_when_well_scaled() {
    let mut model = combined_endpoint_model();
    // Bump additive SD to 40 (far above 1% of 200) → no warning.
    model.default_params.sigma.values[1] = 40.0;
    let pop = population_of(subject_with_obs(1, vec![100.0, 200.0, 300.0]));
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    assert!(
        !diags.iter().any(|d| d.code == "W_ADDITIVE_INIT_SCALE"),
        "a data-scaled additive init must not warn"
    );
}

/// #847 (Copilot review): `ErrorSpec::Selected` dispatch keys are covariate
/// **branch indices**, not CMTs — the warning must label them `endpoint={k}`,
/// not `CMT={k}`. A single-branch selector routes every observation to key 0.
#[test]
fn additive_init_scale_selected_labels_endpoint_not_cmt() {
    let mut model = combined_endpoint_model();
    model.error_spec = ErrorSpec::Selected {
        selector: crate::types::ErrorSelector {
            eval: Box::new(|_cov| 0), // one branch → key 0 for every obs
            branch_labels: vec!["all".into()],
        },
        endpoints: HashMap::from([(
            0,
            crate::types::EndpointError {
                error_model: ErrorModel::Combined,
                sigma_idx: vec![0, 1],
            },
        )]),
    };
    let pop = population_of(subject_with_obs(1, vec![100.0, 200.0, 300.0]));
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    let d = diags
        .iter()
        .find(|d| d.code == "W_ADDITIVE_INIT_SCALE")
        .expect("negligible additive init on a Selected combined endpoint must warn");
    assert!(
        d.message.contains("endpoint=0"),
        "Selected key must be labelled endpoint=, got: {}",
        d.message
    );
    assert!(
        !d.message.contains("CMT="),
        "Selected key must not be mislabelled CMT=, got: {}",
        d.message
    );
}
