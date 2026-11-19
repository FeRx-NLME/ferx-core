//! A fit lowers an `SS=1` dose into a `zero_order()` window once, where it binds the data,
//! not by a clone of the subject in every walk (#2786 review).
// --- the imports of the original parent module (relocated test, #3041) ---
use ferx_model::types::DoseEvent;
use ferx_model::types::Subject;
use std::borrow::Cow;

use ferx_model::dosing::*;
use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;

const DATA: &str = "ID,TIME,EVID,MDV,DV,AMT,RATE,CMT,II,SS
1,0,1,1,0,4321.5,0,1,12,1
1,2,0,0,80.0,0,0,1,0,0
1,6,0,0,55.0,0,0,1,0,0
1,10,0,0,40.0,0,0,1,0,0
2,0,1,1,0,4321.5,0,1,12,1
2,2,0,0,90.0,0,0,1,0,0
2,6,0,0,60.0,0,0,1,0,0
2,10,0,0,45.0,0,0,1,0,0
";

fn fixture(amt: &str) -> (crate::types::CompiledModel, crate::types::Population) {
    let model = crate::parser::model_parser::parse_model_file(Path::new(
        "examples/zero_order_absorption.ferx",
    ))
    .expect("model");
    let dir = tempfile::tempdir().expect("tmp");
    let csv = dir.path().join("d.csv");
    std::fs::write(&csv, DATA.replace("4321.5", amt)).expect("write");
    let pop = crate::read_nonmem_csv(&csv, None, None).expect("data");
    (model, pop)
}

/// Mutation: return `Cow::Borrowed` from `lower_population_zero_order_ss` -> the bound
/// population is not lowered and every walk clones its subject.
#[test]
fn a_bound_population_is_lowered_once_and_its_walks_borrow() {
    let (model, pop) = fixture("4321.25");
    let map = model.active_dose_attr_map();
    assert!(
        matches!(lower_zero_order_ss(&pop.subjects[0], map), Cow::Owned(_)),
        "fixture: the data as read is rewritten by every walk"
    );
    let bound = lower_population_zero_order_ss(&pop, map);
    assert!(matches!(bound, Cow::Owned(_)));
    for s in &bound.subjects {
        assert!(matches!(lower_zero_order_ss(s, map), Cow::Borrowed(_)));
        assert!(is_lowered_zero_order_ss(map, &s.doses[0]));
    }
    assert!(matches!(lower_population_zero_order_ss(&bound, map), Cow::Borrowed(_)));
    // The data as read is untouched: the lowering is the fit's own.
    assert!(!pop.subjects[0].doses[0].is_infusion());
}

/// A whole fit lowers each affected subject exactly once. Mutation: drop the binding in
/// `fit_inner` -> every walk lowers (thousands of lowerings, not 2).
#[test]
fn a_fit_lowers_each_subject_once() {
    let (model, pop) = fixture("4321.5");
    let o = crate::types::FitOptions {
        method: crate::types::EstimationMethod::FoceI,
        outer_maxiter: 2,
        threads: Some(1),
        verbose: false,
        ..crate::types::FitOptions::default()
    };
    let before = MARKER_LOWERINGS.load(Relaxed);
    crate::api::fit(&model, &pop, &model.default_params, &o).expect("fit");
    let n = MARKER_LOWERINGS.load(Relaxed) - before;
    assert_eq!(n, pop.subjects.len(), "lowerings per fit");
}
