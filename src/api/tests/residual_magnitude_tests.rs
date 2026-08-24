//! `E_RUV_MAGNITUDE_NONPOSITIVE` (#484 / #1029): a per-observation residual
//! magnitude — or the multiplier a `weight = <expr>` modifier compiles to —
//! that evaluates to zero, a negative number, or a non-finite value at the
//! initial estimates is a hard error, not a fit that silently lets one row own
//! the objective.

use super::check_model_data;
use crate::parser::model_parser::parse_model_string;
use crate::types::{Population, Subject};
use std::collections::HashMap;

fn weighted_model() -> crate::types::CompiledModel {
    parse_model_string(
        "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  omega ETA_CL ~ 0.09\n  \
         sigma ADD_ERR ~ 1.0 (variance) FIX\n[individual_parameters]\n  CL = TVCL * \
         exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, \
         v=V)\n[error_model]\n  DV ~ additive(ADD_ERR) weight = WPSE\n[covariates]\n  \
         WPSE continuous\n",
    )
    .expect("parse")
}

fn population_with_weights(weights: &[f64]) -> Population {
    let snap = |w: f64| -> HashMap<String, f64> { [("WPSE".to_string(), w)].into_iter().collect() };
    let n = weights.len();
    let subject = Subject {
        id: "1".to_string(),
        obs_times: (0..n).map(|j| j as f64 + 1.0).collect(),
        observations: vec![9.0; n],
        obs_cmts: vec![1; n],
        cens: vec![0; n],
        covariates: snap(weights[0]),
        obs_covariates: weights.iter().map(|&w| snap(w)).collect(),
        ..Default::default()
    };
    Population {
        subjects: vec![subject],
        covariate_names: vec!["WPSE".to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn magnitude_error(pop: &Population) -> Option<String> {
    check_model_data(&weighted_model(), pop)
        .into_iter()
        .find(|d| d.code == "E_RUV_MAGNITUDE_NONPOSITIVE")
        .map(|d| d.message)
}

#[test]
fn positive_weights_pass() {
    assert_eq!(
        magnitude_error(&population_with_weights(&[0.5, 1.5, 2.0])),
        None
    );
}

#[test]
fn a_zero_weight_is_rejected() {
    // The concrete MBMA failure: one trial arm reports no standard error, the
    // covariate evaluates to 0, and that row is handed infinite precision.
    let msg = magnitude_error(&population_with_weights(&[0.5, 0.0, 2.0]))
        .expect("a zero weight must be an error");
    assert!(msg.contains("sigma slot 0"), "got: {msg}");
    assert!(msg.contains("TIME 2"), "got: {msg}");
}

#[test]
fn a_negative_weight_is_rejected() {
    // A negative multiplier squares away to the same variance as its absolute
    // value, so it never means what it says.
    assert!(magnitude_error(&population_with_weights(&[0.5, -1.0])).is_some());
}

#[test]
fn a_non_finite_weight_is_rejected() {
    assert!(magnitude_error(&population_with_weights(&[f64::NAN, 1.0])).is_some());
}

#[test]
fn a_model_with_no_magnitude_is_not_checked() {
    // The check must be a no-op for the overwhelmingly common bare-sigma model,
    // whose subjects carry no weight covariate at all.
    let plain = parse_model_string(
        "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  omega ETA_CL ~ 0.09\n  \
         sigma ADD_ERR ~ 1.0\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = \
         TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ \
         additive(ADD_ERR)\n",
    )
    .expect("parse");
    let pop = population_with_weights(&[0.0, 0.0]);
    assert!(check_model_data(&plain, &pop)
        .iter()
        .all(|d| d.code != "E_RUV_MAGNITUDE_NONPOSITIVE"));
}
