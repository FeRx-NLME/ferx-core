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

// ── pure-TTE subjects: the PK model is a placeholder, do not validate against it ──

/// A pure-TTE model still needs a `[structural_model]` line, and the idiom is a
/// **dummy** `one_cpt_iv` with throwaway CL/V (see `tests/tte_smoke.rs`; the
/// parser supplies exactly that when the block is absent). That placeholder has
/// one compartment, so validating a TTE dataset's dose records against it would
/// reject a working model over a dose the PK engine never reads — the subject
/// has no Gaussian observation, so the analytical predictor is never invoked.
#[cfg(feature = "survival")]
const TTE_ONLY: &str = r#"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 5.0)
  theta DUMMY_CL(1.0, 0.01, 100.0)
  theta DUMMY_V(10.0, 0.1, 1000.0)
  omega ETA_LAMBDA ~ 0.0
  sigma SIGMA_DV ~ 0.1

[individual_parameters]
  LAMBDA = TVLAMBDA * exp(ETA_LAMBDA)
  CL     = DUMMY_CL
  V      = DUMMY_V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model]
  cmt    = 2
  family = exponential
  scale  = LAMBDA
"#;

#[cfg(feature = "survival")]
#[test]
fn pure_tte_subject_is_not_validated_against_the_placeholder_pk_model() {
    let model = model_of(TTE_ONLY);
    assert_eq!(
        format!("{:?}", model.pk_model),
        "OneCptIv",
        "fixture must exercise the 1-compartment placeholder"
    );
    // A dose row into CMT=2 — out of range for `one_cpt_iv`, but this subject's
    // only observation is the TTE event at CMT=2, so no PK prediction happens.
    let pop = pop_of("ID,TIME,DV,EVID,AMT,CMT,MDV\n1,0,.,1,100,2,1\n1,5,1,0,.,2,0\n");
    let diags = check_model_data(&model, &pop);
    assert!(
        diags.iter().all(|d| !d.is_error()),
        "a pure-TTE subject must not be rejected over its dose compartment: {diags:?}"
    );
}

/// …but a subject that *does* carry a Gaussian observation is still checked,
/// even on a model that also has a TTE endpoint. Guards against the exemption
/// being too broad and silently disabling the fix for joint PK-TTE models.
#[cfg(feature = "survival")]
#[test]
fn a_gaussian_observation_on_a_tte_model_is_still_validated() {
    let model = model_of(TTE_ONLY);
    // Same model; this subject has a Gaussian PK observation at CMT=1 as well as
    // the TTE record at CMT=2, so the analytical predictor does run for it.
    let pop =
        pop_of("ID,TIME,DV,EVID,AMT,CMT,MDV\n1,0,.,1,100,2,1\n1,3,4.2,0,.,1,0\n1,5,1,0,.,2,0\n");
    let diags = check_model_data(&model, &pop);
    assert!(
        diags.iter().any(|d| d.code == "E_DOSE_CMT_OUT_OF_RANGE"),
        "a subject with Gaussian observations must still be validated: {diags:?}"
    );
}

/// The exemption above must be **population**-level, not per-subject. In a joint
/// PK-TTE dataset a subject can legitimately carry dose records and a TTE record
/// but no PK sample (an early dropout, or a TTE-only arm) — and the PK path
/// still runs for it, because the event-driven walk is driven by the event
/// schedule and the FOCE/FOCEI sensitivity provider runs per subject regardless.
/// A per-subject exemption let exactly this subject carry an unroutable dose
/// straight back into the panics, which is strictly worse than the bug #375
/// reports. Subject 1 supplies the PK samples; subject 2 is the dangerous one.
#[cfg(feature = "survival")]
#[test]
fn a_tte_only_subject_of_a_pk_tte_dataset_is_still_validated() {
    let model = model_of(TTE_ONLY);
    let pop = pop_of(
        "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
         1,0,.,1,100,1,0,1\n\
         1,3,4.2,0,.,1,.,0\n\
         1,9,0,0,.,2,.,0\n\
         2,0,.,1,100,2,20,1\n\
         2,5,1,0,.,2,.,0\n",
    );
    let diags = check_model_data(&model, &pop);
    assert!(
        diags.iter().any(|d| d.code == "E_DOSE_CMT_OUT_OF_RANGE"),
        "the TTE-only subject's out-of-range dose must still be caught: {diags:?}"
    );
}

/// A bolus into the same peripheral compartment is *not* rejected — the walk
/// adds the amount to the state directly. Guards against the check being
/// over-broad and breaking a legitimate dosing pattern.
///
/// The assertion is deliberately weak (finite, not a specific value): which
/// analytical path serves this subject decides the *answer*, because the
/// superposition path ignores `dose.cmt` and would compute this as a depot dose.
/// This test pins "not rejected", NOT "correct" — the value disagreement is a
/// separate, pre-existing defect tracked outside this PR, and asserting a number
/// here would bless whichever path happens to run.
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
