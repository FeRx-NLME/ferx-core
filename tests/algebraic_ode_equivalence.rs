//! Equivalence between a compartment-free (`$PRED`-equivalent) `[structural_model]`
//! (issue #811) and the dummy-ODE workaround it replaces.
//!
//! Before #811 the only way to fit a response-versus-time model with no dosing —
//! the shape of every model-based meta-analysis structural model — was to declare
//! a compartment nobody uses, drive it with `d/dt(clock) = 1`, and put the real
//! equation in a `[scaling] y = <expr>` readout. The compartment-free form deletes
//! that state. Since the workaround is what users write today (see the MBMA
//! umbrella issue #1032), agreement with it is both the correctness oracle and the
//! migration guarantee: the same model written both ways must predict the same
//! numbers.
//!
//! The model is the Emax time-course from the naproxen/osteoarthritis case study
//! shape — `y = E0 − EMAX·t/(ET50 + t)` — with between-subject variability on the
//! baseline, so the readout exercises an η-dependent individual parameter and the
//! `TIME` built-in together.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::Population;
use ferx_core::{fit, predict, FitOptions};

mod common;

/// The dummy `clock` state is integrated by RK45, so the twin can only match the
/// closed-form evaluation to solver tolerance (defaults: abstol 1e-6, reltol 1e-4).
/// The readout itself is evaluated identically on both sides — the only difference
/// is that one of them integrated a state it never reads.
const ATOL: f64 = 1e-6;
const RTOL: f64 = 1e-6;

const THETAS_AND_INDIV: &str = r"
[parameters]
  theta TVE0(10.0, 0.1, 100.0)
  theta TVEMAX(6.0, 0.1, 100.0)
  theta TVET50(2.0, 0.01, 100.0)

  omega ETA_E0 ~ 0.09

  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  E0   = TVE0 * exp(ETA_E0)
  EMAX = TVEMAX
  ET50 = TVET50
";

/// The compartment-free form: the equation *is* the structural model.
fn algebraic_src() -> String {
    format!(
        "{THETAS_AND_INDIV}
[structural_model]
  EFF = EMAX * TIME / (ET50 + TIME)
  y   = E0 - EFF

[error_model]
  DV ~ proportional(PROP_ERR)
"
    )
}

/// The workaround: one inert state exists only so the model has a compartment.
fn dummy_ode_src() -> String {
    format!(
        "{THETAS_AND_INDIV}
[structural_model]
  ode(states=[clock])

[odes]
  d/dt(clock) = 1

[scaling]
  EFF = EMAX * TIME / (ET50 + TIME)
  y   = E0 - EFF

[error_model]
  DV ~ proportional(PROP_ERR)
"
    )
}

/// Two subjects, no doses at all — the defining feature of this model class.
/// Observation times span the steep early rise and the plateau.
fn population() -> Population {
    let obs_times = vec![0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
    let n = obs_times.len();
    let subjects = ["1", "2"]
        .iter()
        .map(|id| common::subject(id, vec![], obs_times.clone(), vec![0.0; n], vec![1; n]))
        .collect();
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects,
    }
}

#[test]
fn algebraic_structural_model_matches_the_dummy_ode_twin() {
    let alg = parse_full_model(&algebraic_src())
        .expect("compartment-free model parses")
        .model;
    let ode = parse_full_model(&dummy_ode_src())
        .expect("dummy-ODE twin parses")
        .model;

    // The two must actually be the two different engines, or this proves nothing.
    assert!(alg.is_algebraic(), "compartment-free model");
    assert!(
        alg.ode_spec.is_none(),
        "no ODE spec on the compartment-free model"
    );
    assert!(!ode.is_algebraic(), "the twin is an ODE model");
    assert!(ode.ode_spec.is_some(), "the twin integrates a state");

    let pop = population();
    let pa = predict(&alg, &pop, &alg.default_params);
    let po = predict(&ode, &pop, &ode.default_params);
    assert_eq!(pa.len(), po.len());
    assert!(!pa.is_empty());

    for (x, y) in pa.iter().zip(po.iter()) {
        let tol = ATOL + RTOL * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "t={:.3}: compartment-free PRED {:.9} vs dummy-ODE PRED {:.9} \
             (|diff| {:.2e} > tol {:.2e})",
            x.time,
            x.pred,
            y.pred,
            (x.pred - y.pred).abs(),
            tol
        );
    }
}

/// The predictions are the equation, not an artefact of agreeing with a twin that
/// is wrong the same way: check them against the closed form evaluated by hand at
/// the population estimate (η = 0, so `E0 = TVE0`).
#[test]
fn algebraic_structural_model_predicts_the_written_equation() {
    let alg = parse_full_model(&algebraic_src())
        .expect("compartment-free model parses")
        .model;
    let pop = population();
    let preds = predict(&alg, &pop, &alg.default_params);
    assert!(!preds.is_empty());

    let (e0, emax, et50) = (10.0_f64, 6.0_f64, 2.0_f64);
    for p in &preds {
        let expected = e0 - emax * p.time / (et50 + p.time);
        assert!(
            (p.pred - expected).abs() < 1e-9,
            "t={:.3}: PRED {:.9}, hand-computed {:.9}",
            p.time,
            p.pred,
            expected
        );
    }
    // Guard against a degenerate all-equal series passing the loop above.
    assert!(
        (preds[0].pred - preds[preds.len() - 1].pred).abs() > 1.0,
        "the time course must actually vary across the observation window"
    );
}

/// A compartment-free model reaches the estimation path — the model/data
/// validators, the inner EBE loop and the outer optimizer all have to tolerate a
/// model with no doses, no compartments and a placeholder `pk_model`. A handful
/// of outer iterations on data simulated from the model itself; this asserts the
/// path runs and scores, not that it converges (Tier 2).
#[test]
fn algebraic_structural_model_fits() {
    let model = parse_full_model(&algebraic_src())
        .expect("compartment-free model parses")
        .model;

    // Observations *from* the model at the population estimate, so the objective
    // is well-behaved and a wrong prediction path shows up as a non-finite OFV.
    let mut pop = population();
    let (e0, emax, et50) = (10.0_f64, 6.0_f64, 2.0_f64);
    for s in &mut pop.subjects {
        s.observations = s
            .obs_times
            .iter()
            .map(|t| e0 - emax * t / (et50 + t))
            .collect();
    }

    let opts = FitOptions {
        outer_maxiter: 3,
        run_covariance_step: false,
        verbose: false,
        ..Default::default()
    };

    let result = fit(&model, &pop, &model.default_params, &opts)
        .expect("a compartment-free model must be fittable");
    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
}
