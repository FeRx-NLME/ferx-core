//! Tier-1 tests for the `[odes]` RHS-route note (#2681) on its two consumers: `fit()`'s
//! warnings and `ferx check`. The message itself (one test per refusal, each sentence) is
//! pinned in `parser/rhs_regcode_tests.rs`; these pin that both consumers carry it, once,
//! as an informational entry, and that a compiled RHS carries none.
use super::validation::check_ode_rhs_route;
use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::types::{WarningCode, WarningSeverity};
use std::collections::HashMap;

/// A one-state elimination whose RHS is `rhs`.
fn model(rhs: &str) -> CompiledModel {
    parse_model_string(&format!(
        "
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = {rhs}
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
"
    ))
    .expect("parse")
}

const COMPILED: &str = "-(CL / V) * central";
// `tanh` keeps the interpreter (an inline `if` no longer does, #2680).
const INTERPRETED: &str = "-(CL / V) * tanh(central)";

fn subject(id: &str) -> Subject {
    let obs_times = vec![0.5, 2.0, 8.0, 24.0];
    let observations = obs_times.iter().map(|t| 10.0 / (1.0 + t)).collect();
    Subject {
        obs_rgroup: Vec::new(),
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![1, 1, 1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cmt_toggles: Vec::new(),
        cens: vec![0; 4],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
        simulated_mixnum: None,
    }
}

fn population() -> Population {
    Population {
        subjects: vec![subject("1"), subject("2")],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn route_entries(r: &FitResult) -> Vec<&crate::types::WarningEntry> {
    r.warnings_structured
        .iter()
        .filter(|e| e.category == WarningCode::OdeRhsRoute)
        .collect()
}

/// Both sides of the gate in one test: an evaluation-only fit of the interpreted RHS
/// reports the route once, as `Info`, with the construct in the message and `details`;
/// the same fit of the compiled twin reports nothing.
#[test]
fn a_fit_reports_an_interpreted_rhs_once_and_a_compiled_one_not_at_all() {
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..Default::default()
    }
    .quiet();
    let pop = population();

    let m = model(INTERPRETED);
    let r = fit(&m, &pop, &m.default_params, &opts).expect("fit");
    let entries = route_entries(&r);
    assert_eq!(entries.len(), 1, "{:?}", r.warnings);
    let e = entries[0];
    assert_eq!(e.severity, WarningSeverity::Info);
    assert!(
        e.message.starts_with("W_ODE_RHS_INTERPRETED: "),
        "{}",
        e.message
    );
    assert!(e.message.contains("`tanh`"), "{}", e.message);
    assert!(e.message.contains("`d/dt(central)`"), "{}", e.message);
    assert_eq!(
        r.warnings.iter().filter(|w| **w == e.message).count(),
        1,
        "{:?}",
        r.warnings
    );
    let d = e.details.as_ref().expect("details");
    assert_eq!(d["route"], "interpreter");
    assert_eq!(d["site"], "`d/dt(central)`");

    let m = model(COMPILED);
    let r = fit(&m, &pop, &m.default_params, &opts).expect("fit");
    assert!(route_entries(&r).is_empty(), "{:?}", r.warnings);
    assert!(
        !r.warnings.iter().any(|w| w.starts_with("W_ODE_RHS_")),
        "{:?}",
        r.warnings
    );
}

/// `ferx check`'s row is the fit's message, `Info` (so it is not counted as a warning),
/// on block `odes`; a compiled RHS gets no row.
#[test]
fn check_reports_the_same_note_as_an_info_row() {
    let m = model(INTERPRETED);
    let d = check_ode_rhs_route(&m).expect("an interpreted RHS is reported");
    let (msg, _) = postfit::ode_rhs_route_warning(&m).expect("and so is the fit's");
    assert_eq!(d.severity, crate::diagnostics::Severity::Info);
    assert_eq!(d.code, "W_ODE_RHS_INTERPRETED");
    assert_eq!(d.block.as_deref(), Some("odes"));
    assert_eq!(d.message, msg);
    assert!(check_ode_rhs_route(&model(COMPILED)).is_none());
    // A closed-form model has no RHS to report.
    let closed = parse_model_string(
        "
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
",
    )
    .expect("parse");
    assert!(check_ode_rhs_route(&closed).is_none());
    assert!(postfit::ode_rhs_route_warning(&closed).is_none());
}

/// `sqrt` compiles (#2844): a fit and `ferx check` of a `sqrt` RHS report no route note, while
/// a `tanh` RHS, which still keeps the interpreter, reports it — both sides of the gate on
/// the same model shape, so a gate that lets everything through, or nothing, fails here.
#[test]
fn a_sqrt_rhs_is_compiled_and_a_tanh_one_is_still_interpreted() {
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..Default::default()
    }
    .quiet();
    let pop = population();

    let sqrt = model("-(CL / V) * sqrt(central)");
    let r = fit(&sqrt, &pop, &sqrt.default_params, &opts).expect("fit");
    assert!(route_entries(&r).is_empty(), "{:?}", r.warnings);
    assert!(
        !r.warnings.iter().any(|w| w.starts_with("W_ODE_RHS_")),
        "{:?}",
        r.warnings
    );
    assert!(check_ode_rhs_route(&sqrt).is_none());

    let tanh = model(INTERPRETED);
    let r = fit(&tanh, &pop, &tanh.default_params, &opts).expect("fit");
    let entries = route_entries(&r);
    assert_eq!(entries.len(), 1, "{:?}", r.warnings);
    assert!(
        entries[0].message.contains("`tanh`"),
        "{}",
        entries[0].message
    );
    assert!(
        !entries[0].message.contains("`sqrt`"),
        "{}",
        entries[0].message
    );
    assert!(check_ode_rhs_route(&tanh).is_some());
}

/// The model-file entry point reaches the row: `validate_model_file` on a file with an
/// interpreted RHS lists it, under its code.
#[test]
fn validate_model_file_lists_the_row() {
    let dir = std::env::temp_dir().join(format!("ferx_rhs_route_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.ferx");
    std::fs::write(
        &path,
        "
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * tanh(central)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
",
    )
    .unwrap();
    let report = validate_model_file(path.to_str().unwrap(), None);
    let _ = std::fs::remove_dir_all(&dir);
    let row = report
        .diagnostics
        .iter()
        .find(|d| d.code == "W_ODE_RHS_INTERPRETED")
        .unwrap_or_else(|| panic!("{:?}", report.diagnostics));
    assert!(row.message.contains("`tanh`"), "{}", row.message);
}
