use std::collections::BTreeMap;

use super::*;
use crate::ruvsearch::{RuvFeature, RuvsearchOptions, StepRow};
use crate::search::test_support::{converged_fit, model_text};

/// One ruvsearch row, with the fields a pre-screen row carries.
#[allow(clippy::too_many_arguments)]
fn step_row(
    candidate: &str,
    feature: Option<RuvFeature>,
    screened: bool,
    ofv: Option<f64>,
    parent_ofv: f64,
    cwres_dofv: Option<f64>,
    selected: bool,
) -> StepRow {
    StepRow {
        iteration: 1,
        candidate: candidate.into(),
        feature,
        screened,
        parent_ofv,
        ofv,
        lrt: None,
        cwres_dofv,
        note: None,
        selected,
        converged: Some(true),
        passed: true,
        failures: vec![],
        seconds: 1.0,
    }
}

fn ruvsearch_result(rows: Vec<StepRow>) -> crate::ruvsearch::RuvsearchResult {
    crate::ruvsearch::RuvsearchResult {
        options: RuvsearchOptions::default(),
        input_model: model_text("[parameters]\ntheta CL = 1\n"),
        input_ofv: 100.0,
        base_id: "base".into(),
        base_ofv: 100.0,
        rows,
        final_id: "power-1".into(),
        final_model: model_text("[parameters]\ntheta CL = 2\n"),
        final_ofv: 90.0,
        final_fit: Some(converged_fit(90.0)),
        features: vec![RuvFeature::Power],
        models: BTreeMap::new(),
        notes: vec![],
        cancelled: false,
    }
}

/// A CWRES pre-screen candidate is fitted to the parent's conditional weighted
/// residuals, so its number is **not** a data OFV.
///
/// It used to land in the pipeline's `ofv` and `d_ofv` columns under the label
/// `ofv`, beside the data OFVs of every other step — a comparison of unlike
/// quantities that reads as ordinary data. The screened row now reports its
/// number under `cwres_ofv` in `value`, with the data columns empty; the
/// unscreened row beside it is unchanged, which is the half of the assertion
/// that keeps this from passing on a blanket blanking.
#[test]
fn a_cwres_prescreen_row_is_not_reported_as_a_data_ofv() {
    let out = from_ruvsearch(
        3,
        "ruvsearch",
        ruvsearch_result(vec![
            step_row(
                "power-cwres",
                Some(RuvFeature::Power),
                true,
                Some(-12.5),
                -4.0,
                Some(8.5),
                false,
            ),
            step_row(
                "power-1",
                Some(RuvFeature::Power),
                false,
                Some(90.0),
                100.0,
                None,
                true,
            ),
        ]),
    );
    let screened = &out.rows[0];
    assert_eq!(screened.criterion, "cwres_ofv");
    assert_eq!(screened.value, Some(-12.5));
    assert_eq!(
        screened.ofv, None,
        "a CWRES fit has no data OFV to put in the OFV column"
    );
    assert_eq!(screened.d_ofv, None);
    // `cwres_dofv` is base − row; the pipeline's `d` is row − base, so the
    // sign is flipped and the convention "negative is better" holds.
    assert_eq!(screened.d_value, Some(-8.5));

    let fitted = &out.rows[1];
    assert_eq!(fitted.criterion, "ofv");
    assert_eq!(fitted.value, Some(90.0));
    assert_eq!(fitted.ofv, Some(90.0));
    assert_eq!(fitted.d_ofv, Some(-10.0));
}

/// `selected` on a screened row means the pre-screen chose that feature for
/// the data refit — the opposite of being screened out, which is what every
/// screened row used to be labelled.
#[test]
fn the_prescreen_winner_is_not_labelled_screened_out() {
    let out = from_ruvsearch(
        3,
        "ruvsearch",
        ruvsearch_result(vec![
            step_row(
                "power-cwres",
                Some(RuvFeature::Power),
                true,
                Some(-12.5),
                -4.0,
                Some(8.5),
                true,
            ),
            step_row(
                "combined-cwres",
                Some(RuvFeature::Combined),
                true,
                Some(-5.0),
                -4.0,
                Some(1.0),
                false,
            ),
        ]),
    );
    assert_eq!(
        out.rows[0].note.as_deref(),
        Some("chosen by the CWRES pre-screen for the data refit")
    );
    assert_eq!(out.rows[1].note.as_deref(), Some("screened out on CWRES"));
    // And the summary no longer prints "SELECTED (screened out on CWRES)".
    let rendered = crate::amd::report::render_summary(&crate::amd::AmdResult {
        options: crate::amd::AmdOptions::default(),
        input_model: model_text("[parameters]\ntheta CL = 1\n"),
        input_fit: Some(converged_fit(100.0)),
        steps: vec![crate::amd::StepOutcome {
            index: 3,
            step: Step::Residual,
            rerun: false,
            dir: "03-ruvsearch".into(),
            skipped: None,
            failed: None,
            criterion: "ofv",
            value_before: Some(100.0),
            value_after: Some(90.0),
            ofv_before: Some(100.0),
            ofv_after: Some(90.0),
            selected: vec!["power".into()],
            candidates: 2,
            notes: vec![],
            seconds: 1.0,
        }],
        rows: out.rows,
        final_model: model_text("[parameters]\ntheta CL = 2\n"),
        final_fit: Some(converged_fit(90.0)),
        notes: vec![],
        cancelled: false,
    });
    assert!(
        !rendered.contains("SELECTED (screened out on CWRES)"),
        "{rendered}"
    );
    assert!(
        rendered.contains("SELECTED (chosen by the CWRES pre-screen for the data refit)"),
        "{rendered}"
    );
}
