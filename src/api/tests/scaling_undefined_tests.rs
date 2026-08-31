//! Issue #1028 — an undefined identifier in `[scaling]` must not reach the
//! predictor as a silent zero.
//!
//! The parser classifies any identifier it cannot bind to a theta, an eta, an
//! individual parameter, or an ODE state as a **covariate**. A covariate missing
//! from the data resolves to the covariate map's `0.0` default, so a typo'd (or
//! simply undeclared) name in `[scaling]` used to collapse the whole structural
//! prediction — the model validated, fit, converged, and reported plausible
//! parameters for a model nobody wrote.
//!
//! Two things had to hold for that to be caught, and neither did:
//!
//! 1. the name has to be *registered* as a required data column — the Form C `y`
//!    half did this, the Form B `obs_scale` half did not (see
//!    `parser::model_parser::tests::scaling_obs_scale_registers_covariate_references`);
//!    and
//! 2. some entry point has to *check* the registry against the data. `fit()` does
//!    (via `check_model_data`) and `simulate()` does (via `check_covariates`), but
//!    `predict()` ran no data check at all — which is exactly the call the issue's
//!    reprex used.
//!
//! These tests pin (2), with the reprex's own model.
use super::*;
use crate::parser::model_parser::parse_model_string;
use std::collections::HashMap;

/// The issue's reprex: a clock-only ODE whose Form C readout multiplies by a name
/// nothing defines.
const UNDEFINED_READOUT: &str = r#"
[parameters]
  theta TVA(1.0, 0.1, 10.0)
  omega ETA_A ~ 0.01
  sigma ADD_ERR ~ 1.0 (sd)
[individual_parameters]
  A = TVA * exp(ETA_A)
[structural_model]
  ode(states=[clock])
[odes]
  d/dt(clock) = 1
[scaling]
  y[CMT=1] = A * TOTALLY_UNDEFINED_NAME
[error_model]
  DV ~ additive(ADD_ERR)
[fit_options]
  method = focei
  gradient = fd
  covariance = false
"#;

/// One dose-free subject observed on a small time grid, with `covariate_names`
/// under the caller's control so the same population shape can be run with and
/// without the column the model wants.
fn population(covariate_names: &[&str], covariates: HashMap<String, f64>) -> Population {
    let times = vec![0.0, 10.0, 20.0];
    let n = times.len();
    Population {
        subjects: vec![Subject {
            id: "1".into(),
            doses: Vec::new(),
            obs_times: times,
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates,
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
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
            obs_records: vec![],
        }],
        covariate_names: covariate_names.iter().map(|s| s.to_string()).collect(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// `predict()` on the reprex must fail loudly rather than return an all-zero
/// prediction. `fit()` already reported `E_MISSING_COVARIATE` here; `predict()` ran
/// no data check, so it silently served `PRED = 0` at every row.
#[test]
#[should_panic(expected = "TOTALLY_UNDEFINED_NAME")]
fn predict_rejects_an_undefined_scaling_identifier() {
    let model = parse_model_string(UNDEFINED_READOUT).expect("parse");
    let pop = population(&[], HashMap::new());
    let _ = predict(&model, &pop, &model.default_params);
}

/// Positive control: the guard keys on the *data*, not on the name. The identical
/// model against data that carries the column is a legitimate covariate-scaled
/// readout, predicts without panicking, and — since the readout is `A · COV` with
/// `A > 0` and `COV = 2` — is non-zero everywhere. This is what distinguishes the
/// fix from "reject anything unbound at parse time", which would have broken every
/// covariate reference in `[scaling]`.
#[test]
fn predict_accepts_a_scaling_identifier_the_data_carries() {
    let model =
        parse_model_string(&UNDEFINED_READOUT.replace("TOTALLY_UNDEFINED_NAME", "SOME_COVARIATE"))
            .expect("parse");
    let pop = population(
        &["SOME_COVARIATE"],
        HashMap::from([("SOME_COVARIATE".to_string(), 2.0)]),
    );
    let preds = predict(&model, &pop, &model.default_params);
    assert_eq!(preds.len(), 3, "one row per observation");
    assert!(
        preds.iter().all(|p| p.pred > 0.0),
        "a covariate the data carries must scale the readout, got {:?}",
        preds.iter().map(|p| p.pred).collect::<Vec<_>>()
    );
}

/// The check keys on what the *predictor* can resolve, which is wider than
/// `Population::covariate_names`. A CSV-read population always lists every
/// covariate column there, but an in-memory `Population` may populate each
/// `Subject::covariates` map and leave the name list empty — a construction
/// `predict()` has always served correctly, because it resolves covariates from
/// the subject map (`Subject::obs_cov`), not the list. Adding the check to
/// `predict()` must not turn that into a panic.
#[test]
fn predict_accepts_a_covariate_carried_only_by_the_subject_maps() {
    let model =
        parse_model_string(&UNDEFINED_READOUT.replace("TOTALLY_UNDEFINED_NAME", "SOME_COVARIATE"))
            .expect("parse");
    let pop = population(&[], HashMap::from([("SOME_COVARIATE".to_string(), 2.0)]));
    assert!(
        pop.covariate_names.is_empty(),
        "the point of this case is an unpopulated name list"
    );
    let preds = predict(&model, &pop, &model.default_params);
    assert_eq!(preds.len(), 3, "one row per observation");
    assert!(
        preds.iter().all(|p| p.pred > 0.0),
        "a covariate every subject map carries must still scale the readout, got {:?}",
        preds.iter().map(|p| p.pred).collect::<Vec<_>>()
    );
}
