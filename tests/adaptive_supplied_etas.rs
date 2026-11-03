//! Supplied per-subject η from a fit on the adaptive path (#1594, PR-B), test B1.
//!
//! A fit's EBEs η̂ fed back through `SuppliedEtas::from_fit` must reproduce the fit's
//! own IPRED: with a controller that holds at every decision and the fit's doses as
//! the base regimen, the adaptive run is the fitted model's individual prediction
//! for each subject. That ties the supplied-η path to `fit()`'s IPRED, which is the
//! NONMEM-anchored quantity.
//!
//! The simulated population is the fitted one **reversed**, so a lookup by position
//! instead of by ID picks another subject's η̂ and fails. Every subject has its own
//! η̂ (asserted), and the regimen has two doses, so the second lands on residual drug.
//!
//! Tier 2: two outer iterations; the oracle needs η̂ to be the fit's, not optimal.

mod common;

use ferx_core::estimation::uncertainty_samples::fitted_params_from_result;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{DoseEvent, Population, SimOutcome};
use ferx_core::{
    fit, simulate_adaptive_from_spec_with_inputs, simulate_with_options, AdaptiveInputs,
    AdaptiveSimulateOptions, FitOptions, SimulateOptions, SuppliedEtas,
};

// 1-cpt IV ODE, BSV on CL. The always-hold block makes every dose a base dose;
// decisions sit on the dose times, so both engines segment the horizon alike.
const MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central / V
  at = [0, 24]
  start_dose = 100
  route = bolus(cmt=1)
  dose_bounds = [0, 1000]
  when signal >= 0 : hold
"#;

const OBS_TIMES: [f64; 6] = [2.0, 8.0, 20.0, 26.0, 32.0, 44.0];

fn population(ids: &[&str]) -> Population {
    let k = OBS_TIMES.len();
    let doses: Vec<DoseEvent> = [0.0, 24.0]
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: ids
            .iter()
            .map(|id| {
                common::subject(
                    id,
                    doses.clone(),
                    OBS_TIMES.to_vec(),
                    vec![0.0; k],
                    vec![1; k],
                )
            })
            .collect(),
    }
}

const IDS: [&str; 6] = ["1", "2", "3", "4", "5", "6"];

/// A fit of `MODEL` on data simulated from it.
fn fitted() -> (ferx_core::ParsedModel, ferx_core::FitResult) {
    let parsed = parse_full_model(MODEL).expect("model parses");
    let model = &parsed.model;
    let mut data = population(&IDS);
    let sim = simulate_with_options(
        model,
        &data,
        &model.default_params,
        1,
        &SimulateOptions {
            seed: Some(21),
            ..Default::default()
        },
    )
    .expect("data simulation");
    let mut rows = sim.iter();
    for s in &mut data.subjects {
        for dv in &mut s.observations {
            *dv = match rows.next().expect("row").outcome {
                SimOutcome::Continuous { value } => value,
                _ => unreachable!("Gaussian endpoint"),
            };
        }
    }
    let opts = FitOptions {
        outer_maxiter: 2,
        run_covariance_step: false,
        verbose: false,
        ..Default::default()
    };
    let result = fit(model, &data, &model.default_params, &opts).expect("quick fit");
    (parsed, result)
}

#[test]
fn supplied_ebes_reproduce_the_fits_own_ipred() {
    let (parsed, fit_result) = fitted();
    let model = &parsed.model;
    let spec = parsed.adaptive_dosing.as_ref().expect("block");
    let params = fitted_params_from_result(&fit_result, model).expect("fitted params");
    let etas = SuppliedEtas::from_fit(&fit_result, model).expect("from_fit");

    // Non-degenerate: every subject's η̂ differs from every other's, so a mix-up of
    // subjects cannot pass.
    let ebes: Vec<u64> = fit_result
        .subjects
        .iter()
        .map(|s| s.eta[0].to_bits())
        .collect();
    for i in 0..ebes.len() {
        for j in i + 1..ebes.len() {
            assert_ne!(ebes[i], ebes[j], "subjects {i} and {j} share η̂");
        }
    }

    let reversed: Vec<&str> = IDS.iter().rev().copied().collect();
    let target = population(&reversed);
    let mut opts = AdaptiveSimulateOptions::default();
    opts.seed = Some(3);
    let inputs = AdaptiveInputs::new(&params).with_etas(&etas);
    let res = simulate_adaptive_from_spec_with_inputs(model, &target, &inputs, 1, spec, &opts)
        .expect("adaptive run on supplied EBEs");
    assert!(res.ledger.is_empty(), "the controller holds throughout");
    assert!(res.metrics.iter().all(|m| m.eta_supplied));

    // Bit-identical, measured (macOS, 2026-10-19): the two engines integrate the same
    // segments at the same η̂, so there is no solver gap to leave room for.
    let mut compared = 0;
    for s in &fit_result.subjects {
        let rows: Vec<(f64, f64)> = res
            .trajectories
            .iter()
            .filter(|r| r.id == s.id)
            .map(|r| (r.time, r.ipred))
            .collect();
        assert_eq!(rows.len(), s.ipred.len(), "subject {}", s.id);
        for ((t, got), want) in rows.iter().zip(&s.ipred) {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "subject {} at t = {t}: adaptive IPRED {got} != the fit's {want}",
                s.id
            );
            compared += 1;
        }
    }
    assert_eq!(compared, IDS.len() * OBS_TIMES.len());
}

/// `from_fit` refuses a fit of another model: its η are named differently.
#[test]
fn from_fit_refuses_a_fit_of_another_model() {
    let (_, fit_result) = fitted();
    let other = parse_full_model(&MODEL.replace("ETA_CL", "ETA_K"))
        .expect("renamed model parses")
        .model;
    let err = SuppliedEtas::from_fit(&fit_result, &other).expect_err("refused");
    assert_eq!(
        err.diagnostic().map(|d| d.code.as_str()),
        Some("E_SUPPLIED_ETAS")
    );
    assert_eq!(
        err.to_string(),
        "the fit's η are named [ETA_CL] but the model's are [ETA_K]; the fit is of another \
         model"
    );
}
