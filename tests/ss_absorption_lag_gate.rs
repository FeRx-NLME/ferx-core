//! `E_ABSORPTION_SS_LAG` scoping (#719 gap 1, review follow-up on PR #834): the ODE-path SS
//! rejection in `check_absorption_dosing` must gate on whether the *SS-dosed* compartment
//! itself carries a lag (`CompiledModel::has_lagtime_on_cmt`), not on `model.has_lagtime()`
//! (model-wide) — a lag declared on a different absorption/dosing pathway must not block an
//! SS dose into an unlagged compartment. Both directions were previously untested: no test
//! exercised `E_ABSORPTION_SS_LAG` at all, and none pinned the compartment-scoped fix.

mod common;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{check_model_data, DoseEvent, Population};

// `first_order(ka=KA)` absorption straight into `central` (CMT=1, no lag on that route).
// `periph` (CMT=2) is an inert second compartment that exists only to carry its own
// compartment-indexed `ALAG2` — a dosing route the SS subject below never actually uses.
const TWO_ROUTE_MODEL: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  theta TVLAG2(1.0, 0.0, 10.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  KA    = TVKA
  ALAG2 = TVLAG2

[structural_model]
  ode(obs_cmt=central, states=[central, periph])

[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central
  d/dt(periph)  = 0

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn ss_pop(cmt: usize) -> Population {
    let mut ss_dose = DoseEvent::new(0.0, 100.0, cmt, 0.0, true, 12.0);
    ss_dose.ii = 12.0;
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![ss_dose],
            vec![1.0, 4.0, 8.0],
            vec![0.0; 3],
            vec![1; 3],
        )],
    }
}

#[test]
fn ss_into_lagged_absorption_compartment_is_rejected() {
    // `ALAG1` puts the lag on the *same* compartment (CMT=1) the SS dose targets.
    let src = TWO_ROUTE_MODEL.replace("ALAG2 = TVLAG2", "ALAG1 = TVLAG2");
    let model = parse_full_model(&src).expect("model parses").model;
    let pop = ss_pop(1);
    let diags = check_model_data(&model, &pop);
    assert!(
        diags.iter().any(|d| d.code == "E_ABSORPTION_SS_LAG"),
        "SS into a lagged absorption compartment must raise E_ABSORPTION_SS_LAG, got: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

#[test]
fn ss_into_absorption_not_blocked_by_lag_on_other_compartment() {
    // `ALAG2` (as declared in `TWO_ROUTE_MODEL`) lags CMT=2, a route this subject never
    // doses. The SS dose targets the unlagged CMT=1 absorption compartment and must be
    // accepted — `model.has_lagtime()` is true model-wide, but `has_lagtime_on_cmt(1)`
    // must be false.
    let model = parse_full_model(TWO_ROUTE_MODEL)
        .expect("model parses")
        .model;
    assert!(
        model.has_lagtime(),
        "sanity: model must declare a lag somewhere (ALAG2)"
    );
    let pop = ss_pop(1);
    let diags = check_model_data(&model, &pop);
    assert!(
        !diags.iter().any(|d| d.code == "E_ABSORPTION_SS_LAG"),
        "a lag on a different compartment (ALAG2) must not block an SS dose into the \
         unlagged CMT=1 absorption compartment, got: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}
