//! Validation of `SS=1` doses into a built-in absorption compartment (#719, #2786).
//!
//! Until #2786 an SS dose into an absorption compartment carrying a lag was refused
//! (`E_ABSORPTION_SS_LAG`, scoped to the SS-dosed compartment, #834) and so was one into a
//! `zero_order()` window (`E_ABSORPTION_SS_ZERO_ORDER`). Both are served now, so the gate keeps
//! only what the lowering cannot represent: a zero-order window mixed with another absorption
//! function on the same compartment (`E_ABSORPTION_SS_ZERO_ORDER_MIXED`) and a zero-order
//! compartment carrying two lags (`E_ABSORPTION_SS_ZERO_ORDER_LAG`).

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

/// No `E_ABSORPTION_SS*` code, retired or current, in `diags`.
fn no_ss_absorption_refusal(diags: &[ferx_core::Diagnostic]) -> bool {
    !diags.iter().any(|d| d.code.starts_with("E_ABSORPTION_SS_"))
}

#[test]
fn ss_into_lagged_absorption_compartment_is_accepted() {
    // `ALAG1` puts the lag on the *same* compartment (CMT=1) the SS dose targets: the case
    // `E_ABSORPTION_SS_LAG` used to refuse.
    let src = TWO_ROUTE_MODEL.replace("ALAG2 = TVLAG2", "ALAG1 = TVLAG2");
    let model = parse_full_model(&src).expect("model parses").model;
    assert!(model.has_lagtime_on_cmt(1), "sanity: CMT 1 is lagged");
    let diags = check_model_data(&model, &ss_pop(1));
    assert!(
        no_ss_absorption_refusal(&diags),
        "SS into a lagged absorption compartment is served, got: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

/// A per-route `lag=` on a density kernel: the other spelling `E_ABSORPTION_SS_LAG` refused.
#[test]
fn ss_into_per_route_lagged_absorption_compartment_is_accepted() {
    let src = TWO_ROUTE_MODEL.replace("ALAG2 = TVLAG2", "LAG = TVLAG2");
    let src = src.replace("first_order(ka=KA)", "first_order(ka=KA, lag=LAG)");
    let model = parse_full_model(&src).expect("model parses").model;
    let diags = check_model_data(&model, &ss_pop(1));
    assert!(
        no_ss_absorption_refusal(&diags),
        "SS into a per-route-lagged kernel is served, got: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

#[test]
fn ss_into_absorption_not_blocked_by_lag_on_other_compartment() {
    // `ALAG2` (as declared in `TWO_ROUTE_MODEL`) lags CMT=2, a route this subject never
    // doses. The SS dose targets the unlagged CMT=1 absorption compartment and must be
    // accepted — `model.has_lagtime()` is true model-wide, but `has_lagtime_on_cmt(1)`
    // must be false.
    let model = parse_full_model(TWO_ROUTE_MODEL).expect("model parses").model;
    assert!(model.has_lagtime(), "sanity: model must declare a lag somewhere (ALAG2)");
    let pop = ss_pop(1);
    let diags = check_model_data(&model, &pop);
    assert!(
        no_ss_absorption_refusal(&diags),
        "a lag on a different compartment (ALAG2) must not block an SS dose into the \
         unlagged CMT=1 absorption compartment, got: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

// ── what the lowering cannot represent keeps a clear, specific error ──

const ZERO_ORDER_ONLY: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVDUR(2.0, 0.1, 20.0)
  theta TVLAG(1.0, 0.0, 10.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR
  LAG = TVLAG
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn codes(diags: &[ferx_core::Diagnostic]) -> Vec<&str> {
    diags.iter().map(|d| d.code.as_str()).collect()
}

/// A lone zero-order window passes the gate whether its lag is per-route or on the compartment.
#[test]
fn ss_into_a_lone_zero_order_window_is_accepted_with_either_lag_spelling() {
    let alag = ZERO_ORDER_ONLY
        .replace("  LAG = TVLAG\n", "  ALAG1 = TVLAG\n")
        .replace("zero_order(dur=DUR, lag=LAG)", "zero_order(dur=DUR)");
    for (label, src) in [("route lag", ZERO_ORDER_ONLY), ("ALAG1", alag.as_str())] {
        let model = parse_full_model(src).expect("parses").model;
        let diags = check_model_data(&model, &ss_pop(1));
        assert!(no_ss_absorption_refusal(&diags), "{label}: {:?}", codes(&diags));
    }
}

/// `zero_order()` mixed with a density on one compartment: refused, naming the mixed model —
/// and not the lag, which this model does not have.
#[test]
fn ss_into_a_mixed_zero_order_compartment_names_the_mixed_model() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVDUR(3.0, 0.05, 24.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  FZO  = TVFZO
  FZO1 = 1 - TVFZO
  KA   = TVKA
  DUR  = TVDUR
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = parse_full_model(src).expect("parses").model;
    let diags = check_model_data(&model, &ss_pop(1));
    let d = diags
        .iter()
        .find(|d| d.code == "E_ABSORPTION_SS_ZERO_ORDER_MIXED")
        .unwrap_or_else(|| panic!("mixed SS must be refused, got {:?}", codes(&diags)));
    assert!(d.message.contains("mixed"), "{}", d.message);
    assert!(!d.message.contains("per-route"), "{}", d.message);
    assert!(!codes(&diags).contains(&"E_ABSORPTION_SS_ZERO_ORDER_LAG"), "{:?}", codes(&diags));
}

/// A lone zero-order compartment with a per-route `lag=` AND its own `ALAG1`: the two lags add
/// and the lowered infusion carries one, so it is refused naming the two lags.
#[test]
fn ss_into_a_zero_order_compartment_with_two_lags_names_the_lags() {
    let src = ZERO_ORDER_ONLY.replace("  LAG = TVLAG\n", "  LAG = TVLAG\n  ALAG1 = TVLAG\n");
    let model = parse_full_model(&src).expect("parses").model;
    let diags = check_model_data(&model, &ss_pop(1));
    let d = diags
        .iter()
        .find(|d| d.code == "E_ABSORPTION_SS_ZERO_ORDER_LAG")
        .unwrap_or_else(|| panic!("two lags must be refused, got {:?}", codes(&diags)));
    assert!(d.message.contains("per-route") && d.message.contains("ALAG"), "{}", d.message);
    assert!(!d.message.contains("mixed"), "{}", d.message);
    assert!(!codes(&diags).contains(&"E_ABSORPTION_SS_ZERO_ORDER_MIXED"), "{:?}", codes(&diags));
    // The same model without the SS dose is fine: the refusal is the SS combination's.
    let mut plain = ss_pop(1);
    plain.subjects[0].doses[0].ss = false;
    let diags = check_model_data(&model, &plain);
    assert!(no_ss_absorption_refusal(&diags), "{:?}", codes(&diags));
}
