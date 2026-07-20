//! Tier-2 integration tests for #375 — an infusion into a compartment the
//! analytical closed forms cannot deliver into must be **rejected up front**,
//! not `panic!`ed on from inside the event-driven prediction walk.
//!
//! The reported failure was reachable from ordinary NONMEM-format data: a
//! positive `RATE` into `CMT=3` of a `two_cpt_oral` model, on a subject that
//! also carries a time-varying covariate (which is what routes the subject to
//! the event-driven walk in the first place). Nothing validated a *fixed*
//! positive `RATE` against the model's topology — the datareader has no model,
//! and the parse-time infusable check only fires for a declared `D{cmt}`/
//! `R{cmt}` — so the walk's routing `match` was the first thing to see it and
//! it aborted the process.
//!
//! These tests pin the public-boundary contract: `fit()` returns `Err`,
//! `predict()`/`simulate()` panic with the actionable diagnostic (the existing
//! convention for entry points that run no data-check), and the routable
//! compartments still predict.
//!
//! All return immediately (a `check_model_data` pass, a `predict()` at fixed
//! parameters, or a `fit()` that errors before iterating), so they need no
//! `slow-tests` gate.

use ferx_core::api::check_model_data;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{
    fit, predict, read_nonmem_csv, simulate, CompiledModel, FitOptions, Population, Severity,
};
use std::io::Write;
use tempfile::NamedTempFile;

/// Two-compartment oral model with a time-varying covariate on CL, so every
/// subject is routed to the event-driven walk (`has_tv_covariates`) — the path
/// that used to panic. `WT` is the TV covariate.
const TWO_CPT_ORAL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVQ(3.0, 0.1, 50.0)
  theta TVV2(80.0, 5.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * (WT / 70) * exp(ETA_CL)
  V  = TVV
  Q  = TVQ
  V2 = TVV2
  KA = TVKA

[structural_model]
  pk two_cpt_oral(cl=CL, v=V, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP)
"#;

fn write_csv(contents: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().expect("temp file");
    write!(f, "{contents}").expect("write csv");
    f.flush().expect("flush");
    f
}

fn model_of(src: &str) -> CompiledModel {
    parse_full_model(src).expect("model parses").model
}

fn pop_of(csv: &str) -> Population {
    let f = write_csv(csv);
    read_nonmem_csv(f.path(), None, None).expect("dataset loads")
}

const OBS_ROWS: &str = "1,1,5.0,0,.,2,.,0,70\n1,4,4.0,0,.,2,.,0,72\n1,8,3.0,0,.,2,.,0,74\n";

/// `RATE>0` into CMT=3 — the oral **peripheral**, which no closed form can
/// infuse into. This is the #375 repro.
fn peripheral_infusion_csv() -> String {
    format!("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,WT\n1,0,.,1,100,3,20,1,70\n{OBS_ROWS}")
}

/// The same dataset dosing into CMT=2 (central) — routable, must still work.
fn central_infusion_csv() -> String {
    format!("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,WT\n1,0,.,1,100,2,20,1,70\n{OBS_ROWS}")
}

/// …and into CMT=1 (the oral depot), routable since #400.
fn depot_infusion_csv() -> String {
    format!("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,WT\n1,0,.,1,100,1,20,1,70\n{OBS_ROWS}")
}

// ── fit(): a recoverable error, not a crash ──

#[test]
fn fit_rejects_an_infusion_into_an_oral_peripheral() {
    let model = model_of(TWO_CPT_ORAL);
    let pop = pop_of(&peripheral_infusion_csv());
    let err = fit(&model, &pop, &model.default_params, &FitOptions::default())
        .expect_err("an unroutable infusion must be an Err, not a panic");
    assert!(
        err.contains("compartment 3") && err.contains("two_cpt_oral"),
        "error should name the offending compartment and model: {err}"
    );
    assert!(
        err.contains("[1, 2]"),
        "error should name the infusable compartments: {err}"
    );
}

#[test]
fn check_model_data_reports_the_unroutable_infusion() {
    let model = model_of(TWO_CPT_ORAL);
    let pop = pop_of(&peripheral_infusion_csv());
    let diags = check_model_data(&model, &pop);
    let d = diags
        .iter()
        .find(|d| d.code == "E_DOSE_CMT_NOT_INFUSABLE")
        .unwrap_or_else(|| panic!("expected E_DOSE_CMT_NOT_INFUSABLE, got {diags:?}"));
    assert_eq!(d.severity, Severity::Error);
    // The subject/time context the deep-in-the-walk panic could never carry.
    assert!(d.message.contains("subject 1"), "{}", d.message);
    assert!(d.message.contains("time 0"), "{}", d.message);
}

// ── predict()/simulate(): the existing loud-not-silent convention ──

#[test]
#[should_panic(expected = "a dose into a compartment the model cannot deliver into")]
fn predict_panics_before_reaching_the_event_driven_walk() {
    // Previously this panicked from `propagate_with_bounds`'s routing `match`
    // with no subject/time context; now the entry-point guard intercepts it.
    let model = model_of(TWO_CPT_ORAL);
    let pop = pop_of(&peripheral_infusion_csv());
    let _ = predict(&model, &pop, &model.default_params);
}

#[test]
#[should_panic(expected = "a dose into a compartment the model cannot deliver into")]
fn simulate_panics_before_reaching_the_event_driven_walk() {
    let model = model_of(TWO_CPT_ORAL);
    let pop = pop_of(&peripheral_infusion_csv());
    let _ = simulate(&model, &pop, &model.default_params, 1);
}

// ── positive controls: the routable compartments still predict ──

#[test]
fn central_and_depot_infusions_still_predict_on_the_event_driven_walk() {
    let model = model_of(TWO_CPT_ORAL);
    for (label, csv) in [
        ("central", central_infusion_csv()),
        ("depot", depot_infusion_csv()),
    ] {
        let pop = pop_of(&csv);
        assert!(
            check_model_data(&model, &pop).iter().all(|d| !d.is_error()),
            "{label} infusion must stay accepted"
        );
        let preds = predict(&model, &pop, &model.default_params);
        assert_eq!(preds.len(), 3, "{label}: one row per observation");
        assert!(
            preds.iter().all(|p| p.pred.is_finite() && p.pred > 0.0),
            "{label}: predictions should be finite and positive, got {:?}",
            preds.iter().map(|p| p.pred).collect::<Vec<_>>()
        );
    }
}

/// A bolus into the same peripheral compartment is *not* rejected — the walk
/// adds the amount to the state directly. Guards against the check being
/// over-broad and breaking a legitimate dosing pattern.
#[test]
fn a_bolus_into_the_peripheral_is_still_accepted() {
    let model = model_of(TWO_CPT_ORAL);
    let pop = pop_of(&format!(
        "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,WT\n1,0,.,1,100,3,0,1,70\n{OBS_ROWS}"
    ));
    assert!(
        check_model_data(&model, &pop).iter().all(|d| !d.is_error()),
        "a peripheral bolus must stay accepted"
    );
    let preds = predict(&model, &pop, &model.default_params);
    assert!(preds.iter().all(|p| p.pred.is_finite()));
}
