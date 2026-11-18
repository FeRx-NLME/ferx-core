use super::*;
use crate::types::{
    CompiledModel, DoseEvent, ErrorModel, ErrorSpec, ModelParameters, OmegaMatrix, PkParams,
    Population, SigmaVector, Subject,
};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Regression: on a subject with time-varying covariates the sdtab IPRED
/// (`SubjectResult.ipred`) must come from the **TV-aware** prediction path
/// — `compute_predictions_with_tv` — so it agrees with the IPRED used by
/// the FOCEI marginal during the fit.
///
/// Before this fix, `compute_subject_results` called `model_preds` with a
/// single per-subject `pk_params` derived from `subject.covariates`. For
/// TV-covariate subjects that silently used the wrong covariate snapshot
/// for every observation, producing sdtab IPREDs that drifted to ~0 after
/// the first dose and inflated IWRES into 30+. The OFV was fine because
/// the FOCEI marginal already routed through `compute_predictions_with_tv`
/// — only the post-fit diagnostic path was broken.
///
/// This test constructs a minimal 1-cpt IV bolus model where `pk_param_fn`
/// reads `WT` to scale CL, and a subject whose `obs_covariates` carry
/// `WT = [70, 140, 210]` (vs `subject.covariates["WT"] = 70`). The TV
/// path gives a strictly different concentration profile from the no-TV
/// path, so the assertion `sdtab IPRED == compute_predictions_with_tv`
/// fails *loudly* if the dispatch ever regresses to `model_preds`.
#[test]
fn test_sdtab_ipred_honours_tv_covariates() {
    // ── Minimal CompiledModel: 1-cpt IV bolus, CL scaled by per-event WT ──
    let omega = OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]);
    let default_params = ModelParameters {
        sigma: SigmaVector {
            values: vec![0.1],
            names: vec!["PROP_ERR".into()],
        },
        ..crate::types::test_helpers::bare_params(omega)
    };
    let model = CompiledModel {
        name: "tv_cov_sdtab_regression".into(),
        // CL = TVCL · exp(η_CL) · (WT/70) — reads WT from the covariate map
        // that `compute_predictions_with_tv` substitutes per-event from
        // `obs_covariates` / `dose_covariates`. With WT changing per obs
        // the TV path produces a profile that the (broken) no-TV path
        // can't match.
        pk_param_fn: Box::new(
            |theta: &[f64], eta: &[f64], cov: &HashMap<String, f64>, _t: f64| {
                let mut p = PkParams::default();
                let wt = cov.get("WT").copied().unwrap_or(70.0);
                p.values[0] = theta[0] * eta[0].exp() * (wt / 70.0);
                p.values[1] = theta[1];
                p
            },
        ),
        referenced_covariates: vec!["WT".into()],
        ..crate::types::test_helpers::bare_model(default_params.clone())
    };

    // Subject with TV WT: subject.covariates["WT"] = 70 (the no-TV snapshot)
    // but obs_covariates have WT = [70, 140, 210] at the three observation
    // times. dose_covariates set to the same WT=70 the dose-time snapshot
    // would carry.
    let mut subj_cov = HashMap::new();
    subj_cov.insert("WT".to_string(), 70.0);
    let mut obs_covs: Vec<HashMap<String, f64>> = Vec::new();
    for wt in [70.0_f64, 140.0, 210.0] {
        let mut m = HashMap::new();
        m.insert("WT".to_string(), wt);
        obs_covs.push(m);
    }
    let mut dose_covs: Vec<HashMap<String, f64>> = Vec::new();
    let mut m = HashMap::new();
    m.insert("WT".to_string(), 70.0);
    dose_covs.push(m);
    let subject = Subject {
        id: "TVS".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 2.0, 3.0],
        observations: vec![10.0, 5.0, 2.5],
        // placeholders; values don't matter for the IPRED check
        obs_cmts: vec![1, 1, 1],
        covariates: subj_cov,
        dose_covariates: dose_covs,
        obs_covariates: obs_covs,
        cens: vec![0, 0, 0],
        occasions: vec![1, 1, 1],
        dose_occasions: vec![1],
        ..Default::default()
    };
    // Sanity: this subject must be classified TV — that's the regime the
    // bug lived in.
    assert!(
        subject.has_tv_covariates(),
        "test setup wrong: subject must have TV covariates"
    );

    let population = Population {
        subjects: vec![subject.clone()],
        covariate_names: vec!["WT".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    // Fixed EBE at η = 0; H matrix is irrelevant for the IPRED check but
    // must be the right shape for CWRES not to panic.
    let eta_hats = vec![DVector::from_vec(vec![0.0])];
    let h_matrices = vec![DMatrix::from_element(3, 1, 0.5)];
    let kappas: Vec<Vec<DVector<f64>>> = vec![Vec::new()];

    // Reference IPRED: the TV-aware dispatcher the FOCEI marginal uses.
    let ipred_reference = crate::pk::compute_predictions_with_tv(
        &model,
        &subject,
        &default_params.theta,
        eta_hats[0].as_slice(),
    );
    // Sanity: the TV path must NOT collapse to the no-TV path here. If
    // both paths produced the same IPRED, this regression test would
    // trivially pass even if the dispatch in `compute_subject_results`
    // regressed to `model_preds` — so we verify the TV vs no-TV gap is
    // visible before relying on the equality assertion below.
    let pk_no_tv = (model.pk_param_fn)(
        &default_params.theta,
        eta_hats[0].as_slice(),
        &subject.covariates,
        0.0,
    );
    let ipred_no_tv = model_preds(
        &model,
        &subject,
        &pk_no_tv,
        &default_params.theta,
        eta_hats[0].as_slice(),
    );
    let gap: f64 = ipred_reference
        .iter()
        .zip(ipred_no_tv.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        gap > 1e-3,
        "test setup wrong: TV and no-TV IPRED paths must differ noticeably \
             for this regression test to mean anything; got gap = {gap}, \
             ipred_tv = {ipred_reference:?}, ipred_no_tv = {ipred_no_tv:?}"
    );

    // The actual regression assertion: `compute_subject_results` IPRED
    // must equal the TV-aware reference. If the dispatch ever falls back
    // to `model_preds` it will be `ipred_no_tv` instead — failure here.
    let results = compute_subject_results(
        &model,
        &population,
        &default_params,
        &eta_hats,
        &h_matrices,
        &kappas,
        true,
        None,
        false,
    )
    .0;
    assert_eq!(results.len(), 1);
    let sdtab_ipred = &results[0].ipred;
    assert_eq!(sdtab_ipred.len(), 3);
    for (j, (&got, &expected)) in sdtab_ipred.iter().zip(ipred_reference.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-12,
            "sdtab IPRED at obs {j} = {got}, expected (TV-aware) {expected} \
                 — `compute_subject_results` must route IPRED through \
                 `compute_predictions_with_tv` for TV-covariate subjects"
        );
    }

    // Regression for #456: the sdtab population PRED column (f at η = 0) must
    // ALSO route through the TV-aware predictor, so it honours the per-event
    // covariate snapshot and agrees with the public `predict()` output. Before
    // the fix it used `model_preds`, which reads the single `subject.covariates`
    // snapshot for every observation and so silently ignored the TV breakpoints.
    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    let pred_reference =
        crate::pk::compute_predictions_with_tv(&model, &subject, &default_params.theta, &zero_eta);
    let pk_pred_no_tv =
        (model.pk_param_fn)(&default_params.theta, &zero_eta, &subject.covariates, 0.0);
    let pred_no_tv = model_preds(
        &model,
        &subject,
        &pk_pred_no_tv,
        &default_params.theta,
        &zero_eta,
    );
    let pred_gap: f64 = pred_reference
        .iter()
        .zip(pred_no_tv.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        pred_gap > 1e-3,
        "test setup wrong: TV and no-TV PRED paths must differ noticeably; \
             got gap = {pred_gap}"
    );
    let sdtab_pred = &results[0].pred;
    assert_eq!(sdtab_pred.len(), 3);
    for (j, (&got, &expected)) in sdtab_pred.iter().zip(pred_reference.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-12,
            "sdtab PRED at obs {j} = {got}, expected (TV-aware) {expected} \
                 — `compute_subject_results` must route PRED through \
                 `compute_predictions_with_tv` for TV-covariate subjects"
        );
    }
}

#[test]
fn test_sdtab_iwres_uses_block_sigma_correlation() {
    use crate::parser::model_parser::parse_full_model;

    let src = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.0] FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;
    let model = parse_full_model(src).expect("model parses").model;
    assert_eq!(model.residual_correlations.len(), 1);

    let mut subject = Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0],
        observations: vec![0.0],
        obs_cmts: vec![1],
        cens: vec![0],
        occasions: vec![1],
        dose_occasions: vec![1],
        ..Default::default()
    };
    let eta = DVector::from_vec(vec![0.0]);
    let ipred = crate::pk::compute_predictions_with_tv(
        &model,
        &subject,
        &model.default_params.theta,
        eta.as_slice(),
    );
    subject.observations[0] = ipred[0] + 2.0;

    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let eta_hats = vec![eta];
    let h_matrices = vec![DMatrix::zeros(1, 1)];
    let kappas: Vec<Vec<DVector<f64>>> = vec![Vec::new()];

    let results = compute_subject_results(
        &model,
        &population,
        &model.default_params,
        &eta_hats,
        &h_matrices,
        &kappas,
        false,
        None,
        false,
    )
    .0;

    let f = ipred[0];
    let expected_v = (f * 0.2).powi(2) + 1.0 + 2.0 * f * 0.5 * 0.2;
    let expected_iwres = 2.0 / expected_v.sqrt();
    assert!(
        (results[0].iwres[0] - expected_iwres).abs() < 1e-12,
        "sdtab IWRES must include block_sigma cross term; got {}, expected {}",
        results[0].iwres[0],
        expected_iwres
    );
}

/// Regression for #506: simulation must honour time-varying covariate
/// snapshots on observation rows. The pre-fix simulation path evaluated
/// `pk_param_fn` once with `subject.covariates`, so changing `WT` on obs rows
/// had no effect on emitted IPRED/DV unless the baseline dose row changed.
#[test]
fn test_simulate_honours_tv_covariates() {
    let default_params = ModelParameters {
        omega_fixed: Vec::new(),
        sigma: SigmaVector {
            values: vec![0.0],
            names: vec!["PROP_ERR".into()],
        },
        ..crate::types::test_helpers::bare_params(OmegaMatrix::from_diagonal(&[], vec![]))
    };
    let model = CompiledModel {
        name: "tv_cov_sim_regression".into(),
        error_spec: ErrorSpec::Single(ErrorModel::Proportional),
        pk_param_fn: Box::new(
            |theta: &[f64], _eta: &[f64], cov: &HashMap<String, f64>, _t: f64| {
                let mut p = PkParams::default();
                let wt = cov.get("WT").copied().unwrap_or(70.0);
                p.values[0] = theta[0] * (wt / 70.0);
                p.values[1] = theta[1];
                p
            },
        ),
        n_eta: 0,
        eta_names: Vec::new(),
        omega_init_as_sd: Vec::new(),
        eta_map: Vec::new(),
        sel_flat: Vec::new(),
        referenced_covariates: vec!["WT".into()],
        ..crate::types::test_helpers::bare_model(default_params.clone())
    };

    let mut baseline_cov = HashMap::new();
    baseline_cov.insert("WT".to_string(), 70.0);

    let dose_covariates = vec![HashMap::from([("WT".to_string(), 70.0)])];
    let obs_covariates = vec![
        HashMap::from([("WT".to_string(), 70.0)]),
        HashMap::from([("WT".to_string(), 140.0)]),
        HashMap::from([("WT".to_string(), 210.0)]),
    ];
    let subject = Subject {
        id: "SIMTV".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 2.0, 3.0],
        observations: vec![0.0, 0.0, 0.0],
        obs_cmts: vec![1, 1, 1],
        covariates: baseline_cov,
        dose_covariates,
        obs_covariates,
        cens: vec![0, 0, 0],
        occasions: vec![1, 1, 1],
        dose_occasions: vec![1],
        ..Default::default()
    };
    assert!(subject.has_tv_covariates());

    let reference =
        crate::pk::compute_predictions_with_tv(&model, &subject, &default_params.theta, &[]);
    let pk_no_tv = (model.pk_param_fn)(&default_params.theta, &[], &subject.covariates, 0.0);
    let no_tv = model_preds(&model, &subject, &pk_no_tv, &default_params.theta, &[]);
    let gap: f64 = reference
        .iter()
        .zip(no_tv.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        gap > 1e-3,
        "test setup wrong: TV and baseline-only simulation paths must differ; \
             got gap={gap}, tv={reference:?}, no_tv={no_tv:?}"
    );

    let population = Population {
        subjects: vec![subject],
        covariate_names: vec!["WT".into()],
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let rows = simulate_with_seed(&model, &population, &default_params, 1, 506).unwrap();
    assert_eq!(rows.len(), reference.len());
    for (j, (row, &expected)) in rows.iter().zip(reference.iter()).enumerate() {
        assert!(
            (row.ipred - expected).abs() < 1e-12,
            "simulation IPRED at obs {j} = {}, expected TV-aware {expected}",
            row.ipred
        );
    }
}
