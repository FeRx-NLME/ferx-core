//! #1909: an `ADDL`-implied dose takes its lag from its parent row and everything else
//! from the record that governs it.

use super::*;
use std::io::Write as _;

const MODEL: &str = r#"
[parameters]
  theta TVCL(0.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(1.0, 0.01, 100.0)
  omega ETA_CL ~ 0.01
  sigma PROP ~ 0.1 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  KA    = TVKA
  F1    = 1 / X
  ALAG1 = 0.25 * X
  ALAG2 = 0.1 * X
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = -KA*depot
  d/dt(central) = KA*depot - CL/V*central
[scaling]
  y = central / V
[covariates]
  X continuous
[error_model]
  DV ~ proportional(PROP)
"#;

/// A train into each compartment, each with one implied dose that the observation at 18
/// (X = 4) governs; both dose rows have X = 1.
fn fixture() -> (CompiledModel, Subject) {
    let (model, mut pop) = fixture_population();
    (model, pop.subjects.remove(0))
}

fn fixture_population() -> (CompiledModel, crate::types::Population) {
    let model = crate::parser::model_parser::parse_full_model(MODEL)
        .expect("fixture parses")
        .model;
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(
        b"ID,TIME,DV,EVID,MDV,AMT,CMT,ADDL,II,X\n1,0,.,1,1,100,1,1,12,1\n\
          1,1,.,1,1,50,2,1,12,1\n1,18,5,0,0,.,2,.,.,4\n",
    )
    .unwrap();
    let pop = crate::io::datareader::read_nonmem_csv(f.path(), None, None).unwrap();
    (model, pop)
}

#[test]
fn an_implied_dose_takes_only_its_own_compartments_lag_from_its_parent() {
    let (model, subject) = fixture();
    let times: Vec<f64> = subject.doses.iter().map(|d| d.time).collect();
    assert_eq!(times, vec![0.0, 1.0, 12.0, 13.0]);
    assert_eq!(
        (0..4)
            .map(|k| subject.implied_dose_parent(k))
            .collect::<Vec<_>>(),
        vec![None, None, Some(0), Some(1)]
    );
    let map = model.active_dose_attr_map();
    let (lag1, lag2) = (map.lag_slot(1), map.lag_slot(2));
    let f1 = map.indexed_slot(crate::types::DoseAttr::F, 1).expect("F1");
    assert!(lag1 != lag2 && lag1 != f1);

    let theta = model.default_params.theta.clone();
    let eta = vec![0.0; model.n_eta];
    let mut dose = Vec::new();
    compute_dose_pk_params_into(&model, &subject, &theta, &eta, &mut dose);
    let v = |k: usize, slot: usize| dose[k].values[slot];
    // The dose at 12 (into 1): ALAG1 from the parent (X = 1), F1 from the observation
    // (X = 4). Its ALAG2 is not its compartment's, so it stays the governing record's.
    assert_eq!((v(2, lag1), v(2, f1), v(2, lag2)), (0.25, 0.25, 0.4));
    // The dose at 13 (into 2): ALAG2 from its parent; ALAG1 is not its compartment's.
    assert_eq!((v(3, lag2), v(3, lag1)), (0.1, 1.0));
    // The dose rows keep their own.
    assert_eq!((v(0, lag1), v(0, f1), v(1, lag2)), (0.25, 1.0, 0.1));

    // The generic copy over seed vectors is the same copy.
    let mut seeds: Vec<Vec<f64>> = (0..subject.doses.len())
        .map(|k| {
            (model.pk_param_fn)(
                &theta,
                &eta,
                subject.dose_cov(k),
                subject.dose_param_time(k),
            )
            .values
            .to_vec()
        })
        .collect();
    assert_ne!(
        seeds[2][lag1], dose[2].values[lag1],
        "the copy is not a no-op here"
    );
    inherit_parent_dose_attrs_g(&model, &subject, &mut seeds);
    for (k, s) in seeds.iter().enumerate() {
        assert_eq!(s.as_slice(), &dose[k].values[..s.len()], "dose {k}");
    }
}

#[test]
fn the_tad_column_shifts_an_implied_dose_by_its_parents_lag() {
    // The last arrival before 18 is the dose at 13 into compartment 2: ALAG2 = 0.1 from its
    // parent (X = 1), so TAD = 18 − 13.1. Under the governing record's lag (X = 4) it would
    // arrive at 13.4, and TAD would read 4.6.
    let (model, pop) = fixture_population();
    let mut results = vec![crate::types::test_helpers::dummy_subject(
        "1",
        model.n_eta,
        1,
    )];
    crate::api::compute_extra_output_columns(
        &model,
        &pop,
        &model.default_params.theta,
        &[],
        &mut results,
        None,
    );
    let want = 18.0 - (13.0 + 0.1);
    assert!(
        results[0].per_obs_tad.len() == 1 && (results[0].per_obs_tad[0] - want).abs() < 1e-12,
        "TAD {:?}, want [{want}]",
        results[0].per_obs_tad
    );
}

#[test]
fn the_adaptive_event_builder_takes_an_implied_doses_lag_from_its_parent() {
    // `compute_event_pk_params_iov` is the adaptive driver's per-event builder; with no
    // decision window open every record reads the baseline η.
    let (model, subject) = fixture();
    let theta = &model.default_params.theta;
    let eta = vec![0.0; model.n_eta];
    let mut want = Vec::new();
    compute_dose_pk_params_into(&model, &subject, theta, &eta, &mut want);
    let got = compute_event_pk_params_iov(&model, &subject, theta, &eta, &[], &[]);
    let lag1 = model.active_dose_attr_map().lag_slot(1);
    assert_eq!(got.dose[2].values[lag1], 0.25, "the parent row's ALAG1");
    for (k, (g, w)) in got.dose.iter().zip(&want).enumerate() {
        assert_eq!(g.values, w.values, "dose {k}");
    }
}

#[test]
fn a_parent_that_is_not_an_earlier_dose_record_is_ignored() {
    // Only a dose list edited after the read can hold one; the copy must not reach it.
    let (model, mut subject) = fixture();
    let set = |s: &mut Subject, parent: usize| {
        s.doses[2].implied.as_mut().unwrap().parent = parent;
    };
    set(&mut subject, 3); // a later dose
    assert_eq!(subject.implied_dose_parent(2), None);
    set(&mut subject, 2); // itself
    assert_eq!(subject.implied_dose_parent(2), None);
    subject.doses[3].implied.as_mut().unwrap().parent = 2; // an implied dose
    set(&mut subject, 0);
    assert_eq!(subject.implied_dose_parent(3), None);
    assert_eq!(subject.implied_dose_parent(2), Some(0));
    assert_eq!(
        parent_dose_attr_copies(&model, &subject).collect::<Vec<_>>(),
        vec![(2, 0, model.active_dose_attr_map().lag_slot(1))]
    );
}
