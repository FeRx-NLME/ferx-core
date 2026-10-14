//! A data check that names an observation's time names the `TIME` as written (#1959).
//!
//! The fixture is the record the reader moves: an observation listed before a co-timed
//! `MDV=1` / `EVID=2` row sits one ULP below its written `TIME = 2` on the engine clock
//! (#1810), so a message reading that clock says `TIME 1.9999999999999998`, a time the
//! data does not hold. Each test asserts the written time and refuses the moved one,
//! so a message that reads `obs_times` dies on either assertion.

use super::validation::{check_kappa_weights, check_model_data, check_theta_gather_indices};
use crate::parser::model_parser::{parse_full_model, parse_model_string};
use crate::types::{DoseEvent, Population, Subject};
use std::collections::HashMap;

const MOVED: &str = "1.9999999999999998";

/// One subject with observations written at 1, 2 and 3, the second moved a ULP below
/// 2 as the reader moves it; `values` are each observation's `column`.
fn moved_subject(column: &str, values: [f64; 3]) -> Population {
    let snap = |v: f64| HashMap::from([(column.to_string(), v)]);
    let subject = Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 2.0f64.next_down(), 3.0],
        obs_raw_times: vec![1.0, 2.0, 3.0],
        observations: vec![9.0; 3],
        obs_cmts: vec![1; 3],
        cens: vec![0; 3],
        covariates: snap(values[0]),
        dose_covariates: vec![snap(values[0])],
        obs_covariates: values.iter().map(|&v| snap(v)).collect(),
        occasions: vec![1; 3],
        dose_occasions: vec![1],
        ..Default::default()
    };
    assert_eq!(
        format!("{}", subject.obs_times[1]),
        MOVED,
        "the fixture moves it"
    );
    Population {
        subjects: vec![subject],
        covariate_names: vec![column.to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn names_the_written_time(message: &str, at: &str) {
    assert!(message.contains(at), "{message}");
    assert!(!message.contains(MOVED), "{message}");
}

/// `E_RUV_MAGNITUDE_NONPOSITIVE`: a zero `weight = WPSE` on the moved observation.
/// The magnitude is evaluated at the written time (`ruv_obs_mult`), and now reported
/// at it.
///
/// Mutation — the message's `time` read off `obs_times`: it names `MOVED`.
#[test]
fn the_residual_magnitude_check_names_the_written_time() {
    let model = parse_model_string(
        "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  omega ETA_CL ~ 0.09\n  \
         sigma ADD_ERR ~ 1.0 (variance) FIX\n[individual_parameters]\n  CL = TVCL * \
         exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, \
         v=V)\n[error_model]\n  DV ~ additive(ADD_ERR) weight = WPSE\n[covariates]\n  \
         WPSE continuous\n",
    )
    .expect("parse");
    let msg = check_model_data(&model, &moved_subject("WPSE", [0.5, 0.0, 2.0]))
        .into_iter()
        .find(|d| d.code == "E_RUV_MAGNITUDE_NONPOSITIVE")
        .expect("a zero weight is an error")
        .message;
    names_the_written_time(&msg, "at subject `1`, TIME 2 —");
}

/// `E_THETA_GATHER_INDEX_RANGE`: a 0-based index on the moved observation.
///
/// Mutation — the observation rows' time read off `obs_times`: it names `MOVED`.
#[test]
fn the_gather_index_check_names_the_written_time() {
    let model = parse_full_model(
        "[parameters]\n  theta TVCL(2.0, 0.001, 10.0)\n  theta PLACEBO[4](0.5, -10.0, 10.0)\n  \
         theta TVV(10.0, 0.1, 500.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.02\n\
         [individual_parameters]\n  CL = TVCL + PLACEBO[PLA_IDX]\n  V  = TVV\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  \
         DV ~ proportional(PROP_ERR)\n",
    )
    .unwrap()
    .model;
    let diags = check_theta_gather_indices(&model, &moved_subject("PLA_IDX", [1.0, 0.0, 2.0]));
    assert_eq!(diags.len(), 1, "{diags:?}");
    assert_eq!(diags[0].code, "E_THETA_GATHER_INDEX_RANGE");
    names_the_written_time(&diags[0].message, "on subject `1` at TIME 2.");
}

/// `E_KAPPA_WEIGHT_NONPOSITIVE`: a zero `weight = NARM` on the moved observation. The
/// weight is evaluated where the individual parameters it is desugared into are, at the
/// written time since #1969, and reported at it.
///
/// Mutation — the message's `time` read off `obs_times`: it names `MOVED`.
#[test]
fn the_kappa_weight_check_names_the_written_time() {
    let model = parse_model_string(
        "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  omega ETA_CL ~ 0.09\n  \
         kappa KAPPA_CL ~ 0.09 weight = NARM\n  sigma PROP_ERR ~ 0.04\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V  = TVV\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  \
         DV ~ proportional(PROP_ERR)\n[covariates]\n  NARM continuous\n\
         [fit_options]\n  iov_column = OCC\n",
    )
    .expect("parse");
    let diags = check_kappa_weights(&model, &moved_subject("NARM", [100.0, 0.0, 100.0]));
    assert_eq!(diags.len(), 1, "{diags:?}");
    assert_eq!(diags[0].code, "E_KAPPA_WEIGHT_NONPOSITIVE");
    names_the_written_time(&diags[0].message, "at subject `1`, TIME 2 —");
}

/// #1969: a kappa `weight` that reads `TIME` is evaluated at the written time, as the
/// individual parameters it is desugared into read it. `weight = 2 - TIME` is exactly 0 at
/// the observation written at 2, so the check fires there; on the engine clock it read
/// `2 − 2⁻ = 4.4e-16 > 0`, and the check passed a weight the fit then divides by zero.
///
/// Mutation — the check's evaluation time read off `obs_times`: no diagnostic.
#[test]
fn the_kappa_weight_check_evaluates_a_time_reading_weight_at_the_written_time() {
    let model = parse_model_string(
        "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  omega ETA_CL ~ 0.09\n  \
         kappa KAPPA_CL ~ 0.09 weight = 2 - TIME\n  sigma PROP_ERR ~ 0.04\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V  = TVV\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  \
         DV ~ proportional(PROP_ERR)\n[fit_options]\n  iov_column = OCC\n",
    )
    .expect("parse");
    let pop = moved_subject("X", [1.0, 1.0, 1.0]);
    let s = &pop.subjects[0];
    assert!(
        2.0 - s.obs_times[1] > 0.0 && 2.0 - s.obs_raw_times[1] == 0.0,
        "the weight must straddle zero between the two clocks, or this tests nothing"
    );
    let diags = check_kappa_weights(&model, &pop);
    assert_eq!(diags.len(), 1, "{diags:?}");
    assert_eq!(diags[0].code, "E_KAPPA_WEIGHT_NONPOSITIVE");
    names_the_written_time(&diags[0].message, "evaluates to 0 at subject `1`, TIME 2 —");
}
