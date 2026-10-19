//! An adaptive simulation runs on a fit's point estimates when it is handed them
//! (#1594, PR-A).
//!
//! The documented route is `fitted_params_from_result(&fit, &model)` →
//! `simulate_adaptive_from_spec(.., &params, ..)`. `model.default_params` are the
//! *initial* estimates from `[parameters]`, not a fit's, so an entry point that read
//! them instead of its `params` argument would quietly simulate the wrong model.
//!
//! The oracle is the static engine: with a controller that holds at every decision
//! and the dose schedule carried as a base regimen, the adaptive run is a plain
//! simulation of that regimen, so its IPRED must equal `simulate_with_options` at the
//! same parameters and seed, bit for bit. One subject, one replicate: the static path
//! draws ε between subjects on the η stream, the adaptive path does not, so only the
//! first (subject, replicate) shares its η draw across the two engines.
//!
//! It runs once per readout form. Form C (`y = central / V`) divides inside the readout
//! expression. Form A (`obs_scale = V`) divides afterwards, on a path of its own that
//! reads `params.theta` again (#2129, #2142): a run that used the fitted θ for the model
//! and the initial θ for the scale is visible only there.
//!
//! Tier 2: the fit is two outer iterations, which is enough to move θ, Ω and σ off
//! their initial values; nothing here needs it to converge.

mod common;

use ferx_core::estimation::uncertainty_samples::fitted_params_from_result;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{DoseEvent, Population, SimOutcome};
use ferx_core::{
    fit, simulate_adaptive_from_spec, simulate_with_options, AdaptiveSimulateOptions, FitOptions,
    SimulateOptions,
};

/// How the model turns `central` (an amount) into a concentration.
#[derive(Clone, Copy)]
enum Readout {
    /// Form A: `obs_scale = V`, divided after the readout.
    ObsScale,
    /// Form C: `y = central / V`, divided inside the readout expression.
    Expression,
}

// 1-cpt IV ODE with BSV on CL, in the given readout form. The monitored signal is
// `central / V` in both, so the controller sees the same concentration.
//
// The `[adaptive_dosing]` block holds at every decision (`signal >= 0` always matches,
// first match wins), so every dose comes from the subject's base regimen. The
// decisions sit on the dose times, so the reactive driver and the static engine
// segment the horizon identically.
fn model_src(readout: Readout) -> String {
    let (structural, scaling) = match readout {
        Readout::ObsScale => ("ode(obs_cmt=central, states=[central])", "obs_scale = V"),
        Readout::Expression => ("ode(states=[central])", "y = central / V"),
    };
    format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  {structural}
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  {scaling}
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central / V
  at = [0, 24]
  start_dose = 100
  route = bolus(cmt=1)
  dose_bounds = [0, 1000]
  when signal >= 0 : hold
"#
    )
}

const OBS_TIMES: [f64; 7] = [1.0, 4.0, 8.0, 23.0, 25.0, 30.0, 47.0];

fn regimen() -> Vec<DoseEvent> {
    [0.0, 24.0]
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect()
}

fn population(n: usize) -> Population {
    let k = OBS_TIMES.len();
    let subjects = (1..=n)
        .map(|i| {
            common::subject(
                &i.to_string(),
                regimen(),
                OBS_TIMES.to_vec(),
                vec![0.0; k],
                vec![1; k],
            )
        })
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
fn adaptive_run_on_fitted_params_matches_static_simulate_obs_scale_readout() {
    check_fitted_params_reach_the_adaptive_run(Readout::ObsScale);
}

#[test]
fn adaptive_run_on_fitted_params_matches_static_simulate_expression_readout() {
    check_fitted_params_reach_the_adaptive_run(Readout::Expression);
}

fn check_fitted_params_reach_the_adaptive_run(readout: Readout) {
    let parsed = parse_full_model(&model_src(readout)).expect("model parses");
    let model = &parsed.model;
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] present");

    // Data from a truth away from the initial estimates, so two outer iterations
    // move θ off them.
    let mut truth = model.default_params.clone();
    truth.theta = vec![3.0, 35.0];
    let mut data = population(8);
    let sim = simulate_with_options(
        model,
        &data,
        &truth,
        1,
        &SimulateOptions {
            seed: Some(11),
            ..Default::default()
        },
    )
    .expect("data simulation runs");
    let mut rows = sim.iter();
    for s in &mut data.subjects {
        for dv in &mut s.observations {
            let row = rows.next().expect("one row per observation");
            *dv = match row.outcome {
                SimOutcome::Continuous { value } => value,
                _ => unreachable!("Gaussian endpoint"),
            };
        }
    }

    let fit_opts = FitOptions {
        outer_maxiter: 2,
        run_covariance_step: false,
        verbose: false,
        ..Default::default()
    };
    let fit_result = fit(model, &data, &model.default_params, &fit_opts).expect("quick fit runs");
    let params = fitted_params_from_result(&fit_result, model).expect("fitted params");

    // Straddle: the fixture can only tell `params` from `default_params` if they differ.
    for (i, (f, d)) in params
        .theta
        .iter()
        .zip(&model.default_params.theta)
        .enumerate()
    {
        assert_ne!(
            f.to_bits(),
            d.to_bits(),
            "fitted θ[{i}] = {f} equals its initial estimate, so this test could not see an \
             entry point that ignores its params"
        );
    }

    let seed = 7;
    let target = population(1);
    // `AdaptiveSimulateOptions` is `#[non_exhaustive]`: build from `default()`.
    let mut adaptive_opts = AdaptiveSimulateOptions::default();
    adaptive_opts.seed = Some(seed);

    let adaptive = simulate_adaptive_from_spec(model, &target, &params, 1, spec, &adaptive_opts)
        .expect("adaptive run on fitted params");
    assert!(
        adaptive.ledger.is_empty(),
        "the controller holds throughout"
    );

    let reference = simulate_with_options(
        model,
        &target,
        &params,
        1,
        &SimulateOptions {
            seed: Some(seed),
            ..Default::default()
        },
    )
    .expect("static simulate on fitted params");

    assert_eq!(adaptive.trajectories.len(), OBS_TIMES.len());
    assert_eq!(adaptive.trajectories.len(), reference.len());
    for (a, r) in adaptive.trajectories.iter().zip(&reference) {
        assert_eq!((&a.id, a.sim, a.cmt), (&r.id, r.sim, r.cmt));
        assert_eq!(a.time.to_bits(), r.time.to_bits());
        assert_eq!(
            a.ipred.to_bits(),
            r.ipred.to_bits(),
            "t = {}: adaptive IPRED {} != simulate IPRED {} at the same fitted params and seed",
            a.time,
            a.ipred,
            r.ipred
        );
    }
}
