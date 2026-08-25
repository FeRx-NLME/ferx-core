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
use ferx_core::{fit, pk, predict, FitOptions};

mod common;

/// # What this twin proves, and what it does not
///
/// It proves the **values** agree. It is not evidence about the new sensitivity
/// kernel: the two sides reach the same number by structurally different routes —
/// the twin's readout reads a *state* (`clock`, carried through the ODE jet),
/// while the compartment-free readout reads the `TIME` built-in (`Op::PushTime`,
/// whose derivative is identically zero). A green twin says nothing about
/// `∂y/∂θ` or `∂y/∂η`; `Dual2`-vs-FD parity (`src/sens/algebraic_tests.rs`) is the
/// only oracle for that half.
///
/// The tolerance is deliberately tight. The twin integrates, so *some* slack is
/// needed — but the quantity being compared is an expression evaluation on both
/// sides, not an integration result, so anything loose enough to absorb a real
/// disagreement (a dropped term, a parameter read at the wrong occasion) would
/// make the test worthless. 1e-10 relative is comfortably above the observed
/// agreement and far below any difference that could matter.
const ATOL: f64 = 1e-10;
const RTOL: f64 = 1e-10;

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

/// The same pair with inter-occasion variability on the maximum effect — the
/// between-treatment-arm variance component of an MBMA model (#1031). κ reaches
/// the prediction only through the individual parameters the equation reads, so
/// this is the case where a BSV-only convention would silently drop it.
const IOV_THETAS_AND_INDIV: &str = r"
[parameters]
  theta TVE0(10.0, 0.1, 100.0)
  theta TVEMAX(6.0, 0.1, 100.0)
  theta TVET50(2.0, 0.01, 100.0)

  omega ETA_E0 ~ 0.09
  kappa KAPPA_EMAX ~ 0.04

  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  E0   = TVE0 * exp(ETA_E0)
  EMAX = TVEMAX * exp(KAPPA_EMAX)
  ET50 = TVET50
";

const IOV_TAIL: &str = r"
[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  iov_column = OCC
";

fn iov_algebraic_src() -> String {
    format!(
        "{IOV_THETAS_AND_INDIV}
[structural_model]
  EFF = EMAX * TIME / (ET50 + TIME)
  y   = E0 - EFF
{IOV_TAIL}"
    )
}

fn iov_dummy_ode_src() -> String {
    format!(
        "{IOV_THETAS_AND_INDIV}
[structural_model]
  ode(states=[clock])

[odes]
  d/dt(clock) = 1

[scaling]
  EFF = EMAX * TIME / (ET50 + TIME)
  y   = E0 - EFF
{IOV_TAIL}"
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

/// Inter-occasion variability on a compartment-free model must agree with the
/// dummy-ODE twin, which reaches the same prediction through a state it integrates.
///
/// This is the sharpest check on the per-occasion readout: κ enters only through
/// the individual parameters the equation reads, so a BSV-only evaluation would
/// hold κ at zero and still produce a perfectly plausible — and wrong — fit.
#[test]
fn algebraic_iov_matches_the_dummy_ode_twin() {
    let alg = parse_full_model(&iov_algebraic_src())
        .expect("compartment-free IOV model parses")
        .model;
    let ode = parse_full_model(&iov_dummy_ode_src())
        .expect("dummy-ODE IOV twin parses")
        .model;
    assert!(alg.is_algebraic() && alg.n_kappa == 1);
    assert!(ode.ode_spec.is_some() && ode.n_kappa == 1);

    // Two occasions per subject — the treatment arms of a study.
    let mut pop = population();
    let (e0, emax, et50) = (10.0_f64, 6.0_f64, 2.0_f64);
    for s in &mut pop.subjects {
        s.occasions = s
            .obs_times
            .iter()
            .enumerate()
            .map(|(i, _)| if i < s.obs_times.len() / 2 { 1 } else { 2 })
            .collect();
        s.observations = s
            .obs_times
            .iter()
            .map(|t| e0 - emax * t / (et50 + t))
            .collect();
    }

    // Compare the *predictions* at a fixed parameter point rather than two fits:
    // an optimizer comparison would confound a readout difference with the
    // different paths two gradient routes take from the same start.
    let theta = alg.default_params.theta.clone();
    let eta_bsv = vec![0.15];
    let kappas = vec![vec![0.30], vec![-0.25]];
    let subject = &pop.subjects[0];

    let a = pk::predict_iov(&alg, subject, &theta, &eta_bsv, &kappas);
    let d = pk::predict_iov(&ode, subject, &theta, &eta_bsv, &kappas);
    assert_eq!(a.len(), d.len());
    for (j, (x, y)) in a.iter().zip(&d).enumerate() {
        assert!(
            (x - y).abs() <= ATOL + RTOL * x.abs(),
            "obs {j} (t={:.3}): compartment-free {x:.9} vs dummy-ODE {y:.9}",
            subject.obs_times[j]
        );
    }

    // The two occasions carry different κ, so the prediction must actually differ
    // between them — otherwise both engines could be dropping κ and this test
    // would pass by agreeing on the wrong answer. Compare the same elapsed time in
    // each half (index 0 of occasion 1 vs index 0 of occasion 2 share no time, so
    // use the κ = 0 baseline as the reference instead).
    let flat = pk::predict_iov(&alg, subject, &theta, &eta_bsv, &[vec![0.0], vec![0.0]]);
    let moved = a
        .iter()
        .zip(&flat)
        .enumerate()
        // t = 0 has no Emax term, so κ on EMAX cannot move it.
        .filter(|(j, _)| subject.obs_times[*j] > 0.0)
        .filter(|(_, (x, f))| (*x - *f).abs() > 1e-9)
        .count();
    assert!(
        moved > 0,
        "kappa must move the prediction — otherwise the equivalence above is vacuous"
    );
}
