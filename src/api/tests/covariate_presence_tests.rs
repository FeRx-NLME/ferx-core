//! Tier-1 tests for #2181: a covariate is present when **every subject's map** carries
//! it, never because `Population::covariate_names` lists it.
//!
//! Five in-memory populations (the issue's table), through every entry point that runs
//! the check: `check_covariates`, `predict_diag`, `simulate_with_seed`, `fit` (2 outer
//! iterations) and, for the `[adaptive_dosing]` `observe` covariate,
//! `simulate_adaptive_from_spec`.
//!
//! | Population | Every entry point |
//! |---|---|
//! | no `WTX` anywhere (control) | `E_MISSING_COVARIATE` |
//! | `WTX` listed, on no subject | `E_MISSING_COVARIATE`, the control's text |
//! | `WTX` listed, on subject 1 only | `E_MISSING_COVARIATE`, the control's text |
//! | `WTX` unlisted, on subject 1 only (control) | `E_MISSING_COVARIATE` |
//! | `WTX` unlisted, on every subject | `Ok`, finite |
//!
//! Mutations: restoring the `covariate_names` arm of `population_carries` turns the two
//! listed cells `Ok` (every prediction 0 for an uncarried subject); restoring the
//! adaptive check's `covariate_names`-only predicate does that for its listed cells *and*
//! refuses the last cell; listing `covariate_names` unfiltered in the message makes the
//! listed cells name `WTX` as both missing and available.

use super::*;
use crate::parser::model_parser::{parse_full_model, parse_model_string};
use crate::types::{DoseEvent, Population, Subject};
use std::collections::HashMap;

/// `CL = TVCL * WTX * exp(ETA_CL)`: an uncarried `WTX` read as 0 gives CL = 0.
const ONE_CPT_IV_WTX: &str = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  \
    theta TVV(40.0, 1.0, 500.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.02\n\
    [individual_parameters]\n  CL = TVCL * WTX * exp(ETA_CL)\n  V  = TVV\n\
    [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n";

/// The model itself reads no covariate, so only the `observe` check can see `WTX`.
const ODE_ADAPTIVE: &str = "[parameters]\n  theta TVCL(4.0, 0.5, 20.0)\n  \
    theta TVV(50.0, 5.0, 200.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.10 (sd)\n\
    [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  \
    ode(states=[central])\n[odes]\n  d/dt(central) = -(CL / V) * central\n[scaling]\n  \
    y = central / V\n[error_model]\n  DV ~ proportional(PROP)\n";

const MISSING: &str = "Model references covariate(s) not found in data (case-sensitive): WTX. \
    Available covariate columns: (none).";

const OBSERVE_MISSING: &str = "simulate_adaptive_from_spec: [adaptive_dosing] `observe` \
    references covariate(s) not found in data (case-sensitive): WTX. Available covariate \
    columns: (none).";

fn subject(id: &str, wtx: Option<f64>) -> Subject {
    let mut s = Subject {
        id: id.to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 4.0, 8.0],
        observations: vec![2.0, 1.5, 1.0],
        obs_cmts: vec![1; 3],
        cens: vec![0; 3],
        ..Default::default()
    };
    if let Some(w) = wtx {
        s.covariates.insert("WTX".to_string(), w);
    }
    s
}

/// One cell of the table: its population, and whether the check must accept it.
struct Cell {
    label: &'static str,
    population: Population,
    accepted: bool,
}

fn cells() -> Vec<Cell> {
    let pop = |listed: bool, wtx: [Option<f64>; 2]| Population {
        subjects: vec![subject("1", wtx[0]), subject("2", wtx[1])],
        covariate_names: if listed {
            vec!["WTX".to_string()]
        } else {
            Vec::new()
        },
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let cell = |label, population, accepted| Cell {
        label,
        population,
        accepted,
    };
    vec![
        cell("no WTX anywhere", pop(false, [None, None]), false),
        cell("WTX listed, on no subject", pop(true, [None, None]), false),
        cell(
            "WTX listed, on subject 1 only",
            pop(true, [Some(1.0), None]),
            false,
        ),
        cell(
            "WTX unlisted, on subject 1 only",
            pop(false, [Some(1.0), None]),
            false,
        ),
        cell(
            "WTX unlisted, on every subject",
            pop(false, [Some(1.0), Some(1.2)]),
            true,
        ),
    ]
}

/// The refusal an entry point must return on a refused cell: `E_MISSING_COVARIATE` with
/// the control's text, never an `Ok` whose predictions read `WTX` as 0.
fn assert_refused<T>(entry: &str, cell: &Cell, got: Result<T, crate::EngineError>, text: &str) {
    let Err(e) = got else {
        panic!(
            "{entry}, {}: accepted, so an uncarried WTX is read as 0",
            cell.label
        );
    };
    assert_eq!(
        e.code(),
        Some("E_MISSING_COVARIATE"),
        "{entry}, {}: {e}",
        cell.label
    );
    assert!(
        e.to_string().ends_with(text),
        "{entry}, {}: text\n  got:  {e}\n  want: …{text}",
        cell.label
    );
}

fn assert_finite(entry: &str, cell: &Cell, values: impl IntoIterator<Item = f64>) {
    let values: Vec<f64> = values.into_iter().collect();
    assert!(!values.is_empty(), "{entry}, {}: no rows", cell.label);
    // Finite *and* non-zero: a WTX read as 0 gives CL = 0, which is finite.
    assert!(
        values.iter().all(|v| v.is_finite() && *v != 0.0),
        "{entry}, {}: {values:?}",
        cell.label
    );
}

#[test]
fn check_covariates_decides_presence_from_the_subjects() {
    let model = parse_model_string(ONE_CPT_IV_WTX).expect("parse");
    for cell in cells() {
        let diags = check_covariates(&model, &cell.population);
        if cell.accepted {
            assert!(diags.is_empty(), "{}: {diags:?}", cell.label);
        } else {
            assert_eq!(diags.len(), 1, "{}: {diags:?}", cell.label);
            assert_eq!(diags[0].code, "E_MISSING_COVARIATE", "{}", cell.label);
            assert_eq!(diags[0].message, MISSING, "{}", cell.label);
        }
    }
}

/// The subject-static map decides presence, not the per-observation snapshots: a
/// dose, reset or PK-only row without a snapshot of its own, and an adaptive decision
/// before the first record, fall back to the static map, so a name only the
/// observation snapshots carry is read as 0 there (review of #2181). Both checks
/// refuse it; the same snapshots beside a static value pass.
#[test]
fn per_observation_snapshots_alone_do_not_carry_a_covariate() {
    let model = parse_model_string(ONE_CPT_IV_WTX).expect("parse");
    let mut pop = cells().pop().expect("cells").population;
    for s in &mut pop.subjects {
        let w = s.covariates["WTX"];
        s.obs_covariates = vec![HashMap::from([("WTX".to_string(), w)]); 3];
    }
    assert!(
        check_covariates(&model, &pop).is_empty(),
        "control: static + snapshots"
    );

    for s in &mut pop.subjects {
        s.covariates.clear();
    }
    let diags = check_covariates(&model, &pop);
    assert_eq!(diags.len(), 1, "snapshots only: {diags:?}");
    assert_eq!(diags[0].message, MISSING);
    let cell = Cell {
        label: "WTX on the observation snapshots only",
        population: pop,
        accepted: false,
    };
    let ode = parse_model_string(ODE_ADAPTIVE).expect("parse");
    let got = simulate_adaptive_from_spec(
        &ode,
        &cell.population,
        &ode.default_params,
        1,
        &observe_wtx_spec(),
        &adaptive_options(),
    );
    assert_refused("simulate_adaptive_from_spec", &cell, got, OBSERVE_MISSING);
}

#[test]
fn predict_and_simulate_refuse_an_uncarried_covariate() {
    let model = parse_model_string(ONE_CPT_IV_WTX).expect("parse");
    let params = &model.default_params;
    for cell in cells() {
        let predicted = predict_diag(&model, &cell.population, params);
        let simulated = simulate_with_seed(&model, &cell.population, params, 1, 7);
        if cell.accepted {
            let p = predicted.expect("predict_diag");
            assert_finite("predict_diag", &cell, p.results.iter().map(|r| r.pred));
            let s = simulated.expect("simulate_with_seed");
            assert_finite("simulate_with_seed", &cell, s.iter().map(|r| r.ipred));
        } else {
            assert_refused("predict_diag", &cell, predicted, MISSING);
            assert_refused("simulate_with_seed", &cell, simulated, MISSING);
        }
    }
}

#[test]
fn fit_refuses_an_uncarried_covariate() {
    let model = parse_model_string(ONE_CPT_IV_WTX).expect("parse");
    let options = FitOptions {
        outer_maxiter: 2,
        run_covariance_step: false,
        ..Default::default()
    };
    for cell in cells() {
        let got = fit(&model, &cell.population, &model.default_params, &options);
        if cell.accepted {
            let f = got.expect("fit");
            assert!(f.ofv.is_finite(), "{}: OFV {}", cell.label, f.ofv);
            // A WTX read as 0 repels every subject it reaches: OFV ~1e12 (#2181).
            assert!(f.ofv.abs() < 1e6, "{}: OFV {}", cell.label, f.ofv);
        } else {
            assert_refused("fit", &cell, got, MISSING);
        }
    }
}

/// The TDM example's spec, observing `central / WTX` with no assay error.
fn observe_wtx_spec() -> crate::sim::adaptive::AdaptiveDosingSpec {
    let tdm = std::fs::read_to_string("examples/adaptive_tdm_titration.ferx").expect("example");
    let mut spec = parse_full_model(&tdm)
        .expect("parse")
        .adaptive_dosing
        .expect("has [adaptive_dosing]");
    spec.with_assay_error = false;
    spec.assay_cmt = None;
    spec.observe = Some("central / WTX".to_string());
    spec.observe_declared_covariates = vec!["WTX".to_string()];
    spec
}

fn adaptive_options() -> AdaptiveSimulateOptions {
    AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default()
    }
}

/// The `observe` covariate check routes through the same predicate, so it refuses the
/// listed cells and accepts a covariate every subject carries but nobody listed.
#[test]
fn adaptive_observe_covariate_decides_presence_from_the_subjects() {
    let model = parse_model_string(ODE_ADAPTIVE).expect("parse");
    let (spec, options) = (observe_wtx_spec(), adaptive_options());
    for cell in cells() {
        let got = simulate_adaptive_from_spec(
            &model,
            &cell.population,
            &model.default_params,
            1,
            &spec,
            &options,
        );
        if cell.accepted {
            let r = got.expect("simulate_adaptive_from_spec");
            assert!(!r.decisions.is_empty(), "{}: no decisions", cell.label);
            assert_finite(
                "simulate_adaptive_from_spec",
                &cell,
                r.trajectories.iter().map(|t| t.ipred),
            );
        } else {
            assert_refused("simulate_adaptive_from_spec", &cell, got, OBSERVE_MISSING);
        }
    }
}
