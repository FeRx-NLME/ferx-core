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

/// A `TAD`-reading magnitude on a dataset with a **pre-dose** sample (review
/// of #1273): `Subject::time_after_dose` once returned `NaN` there, this check
/// saw a non-finite multiplier and `fit()` refused a valid model with the
/// covariate-cell message. The pre-dose row is Pharmpy's dose group `0` and
/// reads a finite offset now.
#[test]
fn a_tad_magnitude_passes_with_a_pre_dose_sample() {
    use crate::types::DoseEvent;
    let model = parse_model_string(
        "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  theta RUV_TV(0.5, 0.01, \
         10.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, \
         v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR * (if (TAD < 2.0) RUV_TV else \
         1.0))\n",
    )
    .expect("parse");
    assert!(model.ruv_magnitude.as_ref().unwrap().uses_tad);
    let subject = Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![-0.5, 1.0, 4.0],
        observations: vec![9.0; 3],
        obs_cmts: vec![1; 3],
        cens: vec![0; 3],
        ..Default::default()
    };
    let pop = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let diags = check_model_data(&model, &pop);
    assert!(
        diags
            .iter()
            .all(|d| d.code != "E_RUV_MAGNITUDE_NONPOSITIVE"),
        "{diags:?}"
    );
    // The pre-dose row is in the early group (TAD = 0 < 2), like Pharmpy's
    // group 0 — not, as a NaN would have it, in the late one.
    let mult = model
        .ruv_obs_mult(&pop.subjects[0], &model.default_params.theta)
        .expect("active magnitude");
    let rows: Vec<f64> = mult.iter().map(|r| r[0]).collect();
    assert_eq!(rows, [0.5, 0.5, 1.0]);
}

// ── simulate() parity (#1083) ────────────────────────────────────────────────
//
// `fit()` has rejected a non-positive magnitude since #1029, but every
// `simulate()` entry point ran its own hand-picked subset of the model-vs-data
// checks and this one was in none of them. The failure is silent in the worst
// way: a `weight = <expr>` modifier multiplies the *additive* loading, so a zero
// weight removes that arm's residual noise entirely and the simulated
// observation comes back equal to its own IPRED — finite, plottable, and wrong.

use crate::api::{simulate_with_options_diag, simulate_with_seed};
use crate::types::SimOutcome;
use crate::SimulateOptions;

/// The Gaussian value on a simulated row, or a panic for a non-Gaussian one.
fn continuous(r: &crate::api::SimulationResult) -> f64 {
    match r.outcome {
        SimOutcome::Continuous { value } => value,
        _ => panic!("expected a Gaussian row"),
    }
}

fn sim_opts() -> SimulateOptions {
    SimulateOptions {
        seed: Some(42),
        match_method: None,
        horizon: None,
    }
}

#[test]
fn simulate_rejects_a_zero_weight_the_way_fit_does() {
    let model = weighted_model();
    let pop = population_with_weights(&[0.5, 0.0, 2.0]);
    let err = simulate_with_options_diag(&model, &pop, &model.default_params, 1, &sim_opts())
        .expect_err("a zero weight must not simulate");
    assert!(
        err.contains("sigma slot 0") && err.contains("TIME 2"),
        "got: {err}"
    );
}

#[test]
fn simulate_still_runs_with_positive_weights() {
    // The guard must not cost the working case: the same fixture with every
    // weight positive simulates one row per observation, as before.
    let model = weighted_model();
    let pop = population_with_weights(&[0.5, 1.5, 2.0]);
    let out = simulate_with_options_diag(&model, &pop, &model.default_params, 1, &sim_opts())
        .expect("positive weights simulate");
    assert_eq!(out.results.len(), 3);
}

#[test]
#[should_panic(expected = "sigma slot 0")]
fn the_vec_returning_entry_point_panics_on_a_zero_weight() {
    // `simulate` / `simulate_with_seed` return a bare `Vec` and cannot signal, so
    // the contract is enforced as a panic at the shared chokepoint — the same
    // split `validate_iov_simulatable` already uses. Failing loud beats emitting
    // rows whose variability has been quietly removed.
    let model = weighted_model();
    let pop = population_with_weights(&[0.5, 0.0, 2.0]);
    let _ = simulate_with_seed(&model, &pop, &model.default_params, 1, 42);
}

#[test]
fn a_weight_of_one_is_bit_identical_to_no_weight() {
    // The degenerate oracle for the whole mechanism: `weight = W` desugars to a
    // magnitude multiplying the additive loading, so `W ≡ 1` must reproduce the
    // unweighted model's draws exactly — same seed, same epsilons, same rows. A
    // scaling applied on the wrong side, or applied twice, shows up here and
    // nowhere else, because the two models are otherwise identical.
    let weighted = weighted_model();
    let plain = parse_model_string(
        "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  omega ETA_CL ~ 0.09\n  \
         sigma ADD_ERR ~ 1.0 (variance) FIX\n[individual_parameters]\n  CL = TVCL * \
         exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, \
         v=V)\n[error_model]\n  DV ~ additive(ADD_ERR)\n",
    )
    .expect("parse");
    let pop = population_with_weights(&[1.0, 1.0, 1.0]);

    let a = simulate_with_options_diag(&weighted, &pop, &weighted.default_params, 1, &sim_opts())
        .expect("weighted simulates");
    let b = simulate_with_options_diag(&plain, &pop, &plain.default_params, 1, &sim_opts())
        .expect("unweighted simulates");
    assert_eq!(a.results.len(), b.results.len());
    for (x, y) in a.results.iter().zip(b.results.iter()) {
        assert_eq!(
            continuous(x).to_bits(),
            continuous(y).to_bits(),
            "weight = 1 must be bit-identical to no weight at TIME {}",
            x.time
        );
    }
}

#[test]
fn a_larger_weight_widens_the_simulated_residual() {
    // The half the degenerate oracle can't see: the weight must actually reach
    // the draw, and in the right direction. Same seed and same single-observation
    // subject, so the standard normal is shared and only the loading differs —
    // the deviation from IPRED then scales with the weight.
    let model = weighted_model();
    let dev = |w: f64| -> f64 {
        let pop = population_with_weights(&[w]);
        let out = simulate_with_options_diag(&model, &pop, &model.default_params, 1, &sim_opts())
            .expect("simulates");
        let r = &out.results[0];
        (continuous(r) - r.ipred).abs()
    };
    let (small, large) = (dev(0.5), dev(2.0));
    assert!(
        large > small * 3.0,
        "a 4x weight must widen the residual ~4x: {small} then {large}"
    );
}

/// #1182: a `power(...)` exponent at or below zero at the initial estimates is
/// rejected on the same code, with a message that says "exponent" rather than
/// "magnitude for sigma slot" — the row's upper half is the exponent.
#[test]
fn a_non_positive_power_exponent_is_rejected_as_an_exponent() {
    let model = parse_model_string(
        "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  theta RUV_POW(0.0, -1.0, 10.0)\n  \
         omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * \
         exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, \
         v=V)\n[error_model]\n  DV ~ power(PROP_ERR, RUV_POW)\n",
    )
    .expect("parse");
    let pop = population_with_weights(&[0.5, 1.5]);
    let msg = check_model_data(&model, &pop)
        .into_iter()
        .find(|d| d.code == "E_RUV_MAGNITUDE_NONPOSITIVE")
        .map(|d| d.message)
        .expect("a zero exponent must be an error");
    assert!(
        msg.contains("power exponent for sigma slot 0"),
        "got: {msg}"
    );
    assert!(!msg.contains("magnitude for sigma slot"), "got: {msg}");
    // A positive exponent passes.
    let ok = parse_model_string(
        "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  theta RUV_POW(1.3, 0.01, 10.0)\n  \
         omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * \
         exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, \
         v=V)\n[error_model]\n  DV ~ power(PROP_ERR, RUV_POW)\n",
    )
    .expect("parse");
    assert!(check_model_data(&ok, &pop)
        .into_iter()
        .all(|d| d.code != "E_RUV_MAGNITUDE_NONPOSITIVE"));
}
