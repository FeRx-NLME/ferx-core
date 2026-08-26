//! `[simulation] covariate NAME = ...` — the design covariates of an invented
//! arm (#1083).
//!
//! `fit()` always has a data row to read `weight = NARM` / `weight = WPSE` from:
//! the arm exists, and its size and reported standard error are columns.
//! `simulate()` frequently does not — the point of simulating a trial is to
//! invent arms that are not in the data. These pin the convention: the values are
//! part of the design and must be stated, and a design that omits one is refused
//! by name rather than defaulted.

use super::run::simulation_design_covariates;
use crate::parser::model_parser::parse_model_string;
use crate::types::SimulationSpec;

const WEIGHTED_RUV: &str = "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  \
     omega ETA_CL ~ 0.09\n  sigma ADD_ERR ~ 1.0\n[individual_parameters]\n  \
     CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  \
     pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ additive(ADD_ERR) weight = WPSE\n\
     [covariates]\n  WPSE continuous\n";

const UNWEIGHTED: &str = "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  \
     omega ETA_CL ~ 0.09\n  sigma ADD_ERR ~ 1.0\n[individual_parameters]\n  \
     CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  \
     pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ additive(ADD_ERR)\n";

fn spec(n_subjects: usize, covariates: Vec<(String, Vec<f64>)>) -> SimulationSpec {
    SimulationSpec {
        n_subjects,
        dose_amt: 100.0,
        dose_cmt: 1,
        obs_times: vec![1.0, 2.0],
        seed: 7,
        horizon: None,
        covariates,
    }
}

#[test]
fn each_subject_gets_its_own_arm_value() {
    let model = parse_model_string(WEIGHTED_RUV).expect("parse");
    let s = spec(3, vec![("WPSE".to_string(), vec![0.5, 1.0, 2.0])]);
    let (names, per_subject) = simulation_design_covariates(&s, &model).expect("resolves");
    assert_eq!(names, vec!["WPSE".to_string()]);
    assert_eq!(per_subject.len(), 3);
    for (i, want) in [0.5, 1.0, 2.0].iter().enumerate() {
        assert_eq!(
            per_subject[i].get("WPSE"),
            Some(want),
            "subject {i} must carry its own arm's value, not the first arm's"
        );
    }
}

#[test]
fn a_design_that_omits_a_referenced_weight_is_refused_by_name() {
    // The failure this whole feature exists to remove. Note what the message may
    // *not* be: "not found in data" names no fix on a path that has no data file.
    let model = parse_model_string(WEIGHTED_RUV).expect("parse");
    let err = simulation_design_covariates(&spec(3, vec![]), &model)
        .expect_err("an unresolved weight must not be defaulted");
    assert!(err.starts_with("[simulation]:"), "got: {err}");
    assert!(err.contains("`WPSE`"), "got: {err}");
    assert!(
        err.contains("covariate WPSE = <value>"),
        "the message must name the fix: {err}"
    );
}

#[test]
fn a_model_that_references_nothing_needs_no_design_covariates() {
    // The overwhelmingly common case must stay a no-op — every pre-#1083
    // `[simulation]` block declares no covariates at all.
    let model = parse_model_string(UNWEIGHTED).expect("parse");
    let (names, per_subject) = simulation_design_covariates(&spec(2, vec![]), &model)
        .expect("an unweighted model needs no design covariates");
    assert!(names.is_empty());
    assert_eq!(per_subject.len(), 2);
    assert!(per_subject.iter().all(|m| m.is_empty()));
}

#[test]
fn an_extra_design_covariate_the_model_ignores_is_allowed() {
    // Stating a covariate the model happens not to reference is not an error:
    // it lands on the simulated subjects and shows up in the written dataset,
    // which is how a design carries an arm label the model does not use.
    let model = parse_model_string(UNWEIGHTED).expect("parse");
    let s = spec(2, vec![("NARM".to_string(), vec![400.0, 25.0])]);
    let (names, per_subject) = simulation_design_covariates(&s, &model).expect("resolves");
    assert_eq!(names, vec!["NARM".to_string()]);
    assert_eq!(per_subject[1].get("NARM"), Some(&25.0));
}
