//! Tier-1 tests for the `W_DEGENERATE_CL_V` fit warning (#2104).
//!
//! Both closed-form engines serve a record whose `CL` or `V` is non-positive or infinite a
//! concentration of 0, a prediction NONMEM would refuse. The fit completes and scores it, so
//! nothing else in the output says it happened. These pin who the warning names, which
//! records it counts (each evaluated under its own covariate snapshot and occasion κ), every
//! sentence of its text, and the models it stays silent on.
use super::*;
use crate::types::WarningCode;
use nalgebra::DVector;
use std::collections::HashMap;

const LINEAR_AGE_SRC: &str = "
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(80.0, 5.0, 500.0)
  theta AGE_SLOPE(-0.01, -1.0, 1.0)
  omega ETA_V ~ 0.09
  sigma PROP ~ 0.01

[individual_parameters]
  CL = TVCL * (1 + AGE_SLOPE * (AGE - 40))
  V = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";

const LINEAR_AGE_ODE_SRC: &str = "
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(80.0, 5.0, 500.0)
  theta AGE_SLOPE(-0.01, -1.0, 1.0)
  omega ETA_V ~ 0.09
  sigma PROP ~ 0.01

[individual_parameters]
  CL = TVCL * (1 + AGE_SLOPE * (AGE - 40))
  V = TVV * exp(ETA_V)

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)
";

const IOV_SRC: &str = "
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(80.0, 5.0, 500.0)
  omega ETA_V ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma PROP ~ 0.01

[individual_parameters]
  CL = TVCL * (1 + KAPPA_CL)
  V = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  iov_column = OCC
";

fn parse(src: &str) -> CompiledModel {
    crate::parser::model_parser::parse_full_model(src)
        .expect("fixture model parses")
        .model
}

fn age(a: f64) -> HashMap<String, f64> {
    HashMap::from([("AGE".to_string(), a)])
}

/// One IV bolus at TIME 0 and observations at 1, 2, 3, every record at `ages[k]`
/// (dose first, then the three observations).
fn subject(id: &str, ages: [f64; 4]) -> Subject {
    let tv = ages.iter().any(|&a| a != ages[0]);
    Subject {
        obs_rgroup: Vec::new(),
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 2.0, 3.0],
        obs_raw_times: Vec::new(),
        observations: vec![1.0, 1.0, 1.0],
        obs_cmts: vec![1, 1, 1],
        covariates: age(ages[0]),
        dose_covariates: if tv { vec![age(ages[0])] } else { Vec::new() },
        obs_covariates: if tv {
            ages[1..].iter().map(|&a| age(a)).collect()
        } else {
            Vec::new()
        },
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0, 0, 0],
        occasions: vec![1, 1, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

fn population(subjects: Vec<Subject>) -> Population {
    Population {
        subjects,
        covariate_names: vec!["AGE".into()],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn etas(n: usize, v: f64) -> Vec<DVector<f64>> {
    vec![DVector::from_element(1, v); n]
}

/// AGE 200 gives `CL = 5 · (1 − 0.01 · 160) = −3` on every record of S2; S1 at AGE 40 is
/// fine. The warning names S2 alone and counts all four of its records. Each assertion is
/// one sentence of the message: deleting that sentence reddens it.
#[test]
fn a_negative_clearance_names_the_subject_and_counts_its_records() {
    let model = parse(LINEAR_AGE_SRC);
    let pop = population(vec![subject("S1", [40.0; 4]), subject("S2", [200.0; 4])]);
    let (msg, entry) = degenerate_cl_v_warning(
        &model,
        &pop,
        &model.default_params.theta,
        &etas(2, 0.0),
        &[],
        None,
    )
    .expect("a negative CL must warn");
    assert_eq!(entry.category, WarningCode::DegenerateClearanceOrVolume);
    assert_eq!(entry.message, msg);
    // Sentence 1: who, how many records, where first.
    assert!(
        msg.starts_with("W_DEGENERATE_CL_V: 1 subject(s) [S2]"),
        "{msg}"
    );
    assert!(
        !msg.contains("S1"),
        "the in-range subject is not named: {msg}"
    );
    assert!(msg.contains("on 4 record(s)"), "{msg}");
    assert!(msg.contains("first at TIME 0 in subject S2"), "{msg}");
    // Sentence 2, gated on non-positive / infinite records: what the engines served.
    assert!(
        msg.contains("On 4 of them CL or V is non-positive or infinite"),
        "{msg}"
    );
    assert!(msg.contains("serve a concentration of 0"), "{msg}");
    // Its NaN twin is gated on NaN records, of which there are none here.
    assert!(
        !msg.contains("is NaN"),
        "no NaN record, no NaN sentence: {msg}"
    );
    // Sentence 3: why NONMEM differs.
    assert!(
        msg.contains("PREDPP rejects a non-positive PK parameter"),
        "{msg}"
    );
    // Sentence 4: the advice.
    assert!(msg.contains("exponential or power model"), "{msg}");
    // The emitted text classifies to its own code, not the generic "degenerate" →
    // optimizer-health arm its wording would otherwise fall into.
    assert_eq!(
        crate::types::classify_warning(&msg).category,
        WarningCode::DegenerateClearanceOrVolume
    );
    let details = entry.details.expect("native entry carries details");
    assert_eq!(details["n_records"], 4);
    assert_eq!(details["n_zero_records"], 4);
    assert_eq!(details["n_nan_records"], 0);
    assert_eq!(details["subjects"][0]["id"], "S2");
    assert_eq!(
        details["subjects"][0]["times"],
        serde_json::json!([0.0, 1.0, 2.0, 3.0])
    );
}

/// A time-varying covariate puts the negative CL on one record only (TIME 2): that record
/// alone is counted, so the per-record snapshot is what is read, not the t = 0 baseline.
#[test]
fn a_time_varying_covariate_counts_only_the_offending_record() {
    let model = parse(LINEAR_AGE_SRC);
    let pop = population(vec![subject("S1", [40.0, 40.0, 200.0, 40.0])]);
    let (msg, entry) = degenerate_cl_v_warning(
        &model,
        &pop,
        &model.default_params.theta,
        &etas(1, 0.0),
        &[],
        None,
    )
    .expect("one negative record must warn");
    assert!(msg.contains("on 1 record(s)"), "{msg}");
    assert!(msg.contains("first at TIME 2 in subject S1"), "{msg}");
    assert_eq!(
        entry.details.unwrap()["subjects"][0]["times"],
        serde_json::json!([2.0])
    );
}

/// `V = TVV · exp(ETA_V)` overflows to `+inf` at an extreme η̂: infinite is degenerate too.
#[test]
fn an_infinite_volume_warns() {
    let model = parse(LINEAR_AGE_SRC);
    let pop = population(vec![subject("S1", [40.0; 4])]);
    let (msg, _) = degenerate_cl_v_warning(
        &model,
        &pop,
        &model.default_params.theta,
        &etas(1, 800.0),
        &[],
        None,
    )
    .expect("an infinite V must warn");
    assert!(
        msg.contains("[S1]") && msg.contains("on 4 record(s)"),
        "{msg}"
    );
}

/// On an IOV model each record reads its own occasion's κ: `KAPPA_CL = −2` on occasion 2
/// gives `CL = −5` at the TIME 3 observation alone.
#[test]
fn an_iov_model_reads_each_records_own_occasion_kappa() {
    let model = parse(IOV_SRC);
    assert_eq!(model.n_kappa, 1);
    let pop = population(vec![subject("S1", [40.0; 4])]);
    let kappas = vec![vec![
        DVector::from_element(1, 0.0),
        DVector::from_element(1, -2.0),
    ]];
    let (msg, _) = degenerate_cl_v_warning(
        &model,
        &pop,
        &model.default_params.theta,
        &etas(1, 0.0),
        &kappas,
        None,
    )
    .expect("a negative occasion-2 CL must warn");
    assert!(msg.contains("on 1 record(s)"), "{msg}");
    assert!(msg.contains("first at TIME 3 in subject S1"), "{msg}");
}

/// Silent when every record is in range, and on an ODE model, whose engine the closed-form
/// rule does not govern (the message's claim about what was served would be false there).
#[test]
fn silent_when_in_range_or_not_a_closed_form() {
    let model = parse(LINEAR_AGE_SRC);
    let fine = population(vec![subject("S1", [40.0; 4]), subject("S2", [60.0; 4])]);
    assert!(degenerate_cl_v_warning(
        &model,
        &fine,
        &model.default_params.theta,
        &etas(2, 0.0),
        &[],
        None
    )
    .is_none());

    let ode = parse(LINEAR_AGE_ODE_SRC);
    let bad = population(vec![subject("S2", [200.0; 4])]);
    assert!(degenerate_cl_v_warning(
        &ode,
        &bad,
        &ode.default_params.theta,
        &etas(1, 0.0),
        &[],
        None
    )
    .is_none());
}

/// A `NaN` CL — here from a covariate value the expression cannot use — is reported too,
/// though the engines' degenerate rule deliberately excludes `NaN`: such a record is served
/// `NaN`, not 0, and the message says so. Both sentence gates are asserted in one test with
/// a mixed population, so forcing either branch reddens it: S1 is `NaN` on its TIME 1
/// record only, S2 negative on every record.
#[test]
fn a_nan_clearance_warns_with_its_own_outcome() {
    let model = parse(LINEAR_AGE_SRC);
    let nan_only = population(vec![subject("S1", [40.0, f64::NAN, 40.0, 40.0])]);
    let (msg, entry) = degenerate_cl_v_warning(
        &model,
        &nan_only,
        &model.default_params.theta,
        &etas(1, 0.0),
        &[],
        None,
    )
    .expect("a NaN CL must warn");
    assert!(
        msg.contains("non-positive or non-finite CL or V on 1 record(s)"),
        "{msg}"
    );
    assert!(msg.contains("first at TIME 1 in subject S1"), "{msg}");
    assert!(msg.contains("On 1 of them CL or V is NaN"), "{msg}");
    assert!(msg.contains("repelled rather than scored"), "{msg}");
    assert!(
        !msg.contains("serve a concentration of 0"),
        "a NaN record is not served 0: {msg}"
    );
    let details = entry.details.unwrap();
    assert_eq!(details["n_nan_records"], 1);
    assert_eq!(
        details["subjects"][0]["nan_times"],
        serde_json::json!([1.0])
    );

    let mixed = population(vec![
        subject("S1", [40.0, f64::NAN, 40.0, 40.0]),
        subject("S2", [200.0; 4]),
    ]);
    let (msg, entry) = degenerate_cl_v_warning(
        &model,
        &mixed,
        &model.default_params.theta,
        &etas(2, 0.0),
        &[],
        None,
    )
    .expect("a mixed population must warn");
    assert!(msg.contains("2 subject(s) [S1, S2]"), "{msg}");
    assert!(msg.contains("on 5 record(s)"), "{msg}");
    assert!(
        msg.contains("On 4 of them CL or V is non-positive or infinite"),
        "{msg}"
    );
    assert!(msg.contains("On 1 of them CL or V is NaN"), "{msg}");
    let details = entry.details.unwrap();
    assert_eq!(
        (
            details["n_zero_records"].as_u64(),
            details["n_nan_records"].as_u64()
        ),
        (Some(4), Some(1))
    );
}
