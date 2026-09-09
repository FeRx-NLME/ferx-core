use super::super::{AmdOptions, AmdResult, CandidateRow, Retries, Step, StepOutcome, Strategy};
use super::*;
use crate::search::test_support::{converged_fit, model_text};

fn row(step: usize, tool: &str, id: &str, ofv: f64) -> CandidateRow {
    CandidateRow {
        step,
        tool: tool.into(),
        id: id.into(),
        parent: Some("base".into()),
        description: "one peripheral".into(),
        criterion: "bic_mixed",
        value: Some(ofv + 5.0),
        d_value: Some(-2.0),
        ofv: Some(ofv),
        d_ofv: Some(-7.0),
        rank: Some(1),
        converged: Some(true),
        passed: true,
        failures: vec![],
        error: None,
        note: None,
        seconds: 12.5,
        selected: true,
    }
}

/// A pipeline that ran one step and skipped one, with a candidate that failed
/// the gate — the three shapes the report has to render.
fn result() -> AmdResult {
    let mut failed = row(1, "modelsearch", "cand2", 995.0);
    failed.selected = false;
    failed.passed = false;
    failed.rank = None;
    failed.failures = vec!["stalled at the initial estimates (#751)".into()];
    let mut broken = row(1, "modelsearch", "cand3", 0.0);
    broken.selected = false;
    broken.passed = false;
    broken.ofv = None;
    broken.value = None;
    broken.d_ofv = None;
    broken.rank = None;
    broken.error = Some("does not compile: unknown parameter `Q`".into());
    AmdResult {
        options: AmdOptions {
            strategy: Strategy::Default,
            retries: Retries::Final,
            skip: vec![],
        },
        input_model: model_text("[parameters]\ntheta CL = 1\n"),
        input_fit: Some(converged_fit(1010.0)),
        steps: vec![
            StepOutcome {
                index: 1,
                step: Step::Structural,
                rerun: false,
                dir: "01-modelsearch".into(),
                skipped: None,
                failed: None,
                criterion: "bic_mixed",
                value_before: Some(1015.0),
                value_after: Some(995.0),
                ofv_before: Some(1010.0),
                ofv_after: Some(990.0),
                selected: vec!["FO, 1 peripheral".into()],
                candidates: 3,
                notes: vec!["the input model ranks better on one layer".into()],
                seconds: 42.0,
            },
            StepOutcome {
                index: 2,
                step: Step::Covariates,
                rerun: false,
                dir: "02-covsearch".into(),
                skipped: Some("the search space has no COVARIATE statement".into()),
                failed: None,
                criterion: "",
                value_before: None,
                value_after: None,
                ofv_before: None,
                ofv_after: None,
                selected: vec![],
                candidates: 0,
                notes: vec![],
                seconds: 0.0,
            },
        ],
        rows: vec![row(1, "modelsearch", "cand1", 990.0), failed, broken],
        final_model: model_text("[parameters]\ntheta CL = 2\n"),
        final_fit: Some(converged_fit(990.0)),
        notes: vec!["the retries pass on `retries` did not improve the fit".into()],
        cancelled: false,
    }
}

/// The three files land, with the declared headers and one record per row.
#[test]
fn write_report_writes_the_two_tables_and_the_final_model() {
    let dir = tempfile::tempdir().unwrap();
    let result = result();
    write_report(dir.path(), &result).unwrap();

    let steps = std::fs::read_to_string(steps_path(dir.path())).unwrap();
    let lines: Vec<&str> = steps.lines().collect();
    assert_eq!(lines[0], STEP_COLUMNS.join(","));
    assert_eq!(lines.len(), 1 + result.steps.len());
    assert!(lines[1].starts_with("structural,modelsearch,false,01-modelsearch,ran,"));
    assert!(
        lines[2].contains("skipped") && lines[2].contains("no COVARIATE statement"),
        "{}",
        lines[2]
    );

    let candidates = std::fs::read_to_string(candidates_path(dir.path())).unwrap();
    let lines: Vec<&str> = candidates.lines().collect();
    assert_eq!(lines[0], CANDIDATE_COLUMNS.join(","));
    assert_eq!(lines.len(), 1 + result.rows.len());
    assert!(lines[2].contains("stalled at the initial estimates"));
    assert!(lines[3].contains("does not compile"));

    assert_eq!(
        std::fs::read_to_string(final_model_path(dir.path())).unwrap(),
        result.final_model.render()
    );
}

/// Every record has exactly as many fields as its header.
///
/// A column added to `STEP_COLUMNS` without a value beside it — or a value
/// without a column — shifts every field after it, which reads as plausible
/// data rather than as an error. `csv` refuses an inconsistent record only
/// when it is written *after* the header on the same writer, which is what
/// this re-reads to confirm.
#[test]
fn every_record_has_as_many_fields_as_its_header() {
    let dir = tempfile::tempdir().unwrap();
    write_report(dir.path(), &result()).unwrap();
    for (path, columns) in [
        (steps_path(dir.path()), STEP_COLUMNS.len()),
        (candidates_path(dir.path()), CANDIDATE_COLUMNS.len()),
    ] {
        let mut reader = csv::Reader::from_path(&path).unwrap();
        assert_eq!(reader.headers().unwrap().len(), columns);
        for record in reader.records() {
            assert_eq!(
                record.unwrap().len(),
                columns,
                "{} has a record of the wrong width",
                path.display()
            );
        }
    }
}

/// The summary carries what a reader needs to audit the run: the strategy, the
/// step table, the skip *with its reason*, every candidate including the ones
/// that failed and why, and the final model's estimates with their standard
/// errors.
#[test]
fn the_summary_shows_every_step_every_candidate_and_the_estimates() {
    let out = render_summary(&result());
    assert!(out.contains("default strategy, retries final"), "{out}");
    assert!(out.contains("Start model: OFV 1010.000"));
    // The step table, with the skip explained rather than omitted.
    assert!(out.contains("structural"));
    assert!(
        out.contains("skipped: the search space has no COVARIATE statement"),
        "{out}"
    );
    // Every candidate, with the reason a failed one was not selected.
    assert!(out.contains("cand1") && out.contains("SELECTED"));
    assert!(
        out.contains("excluded: stalled at the initial estimates (#751)"),
        "{out}"
    );
    assert!(out.contains("failed: does not compile"), "{out}");
    // Notes, from the pipeline and from the steps.
    assert!(out.contains("did not improve the fit"));
    assert!(out.contains("the input model ranks better on one layer"));
    // The final model and what the pipeline was worth.
    assert!(out.contains("Final model: OFV 990.000 (-20.000 against the start model)"));
    // The estimates table, which is what carries the standard errors.
    assert!(
        out.contains(&ferx_core::io::output::parameter_table(
            result().final_fit.as_ref().unwrap()
        )),
        "the estimates table is missing"
    );
}

/// A row the step carried forward *and* the gate rejected says both.
///
/// A search that has to return something can hand back a model that failed the
/// gate — the input it started from, when no candidate passed. Printing
/// `SELECTED` and stopping there tells the reader the opposite of what the
/// verdict says.
#[test]
fn a_selected_row_that_failed_the_gate_still_says_so() {
    let mut result = result();
    result.rows[0].passed = false;
    result.rows[0].failures = vec!["condition number 1e9 exceeds 1e3".into()];
    let out = render_summary(&result);
    assert!(
        out.contains("SELECTED (failed the gate: condition number 1e9 exceeds 1e3)"),
        "{out}"
    );
    // With a note as well, both are kept.
    result.rows[0].note = Some("p = 0.0010 (df 1)".into());
    let out = render_summary(&result);
    assert!(
        out.contains(
            "SELECTED (p = 0.0010 (df 1); failed the gate: condition number 1e9 exceeds 1e3)"
        ),
        "{out}"
    );
}

/// A step that ran and failed is `failed` in the table with its message, told
/// apart from a step that never ran at all.
#[test]
fn a_failed_step_is_told_apart_from_a_skipped_one() {
    let mut result = result();
    result.steps[1].skipped = None;
    result.steps[1].failed = Some("`WT` is not a covariate of the dataset".into());
    let dir = tempfile::tempdir().unwrap();
    write_report(dir.path(), &result).unwrap();
    let steps = std::fs::read_to_string(steps_path(dir.path())).unwrap();
    let row = steps.lines().nth(2).unwrap();
    assert!(row.contains(",failed,"), "{row}");
    assert!(row.contains("not a covariate"), "{row}");
    let out = render_summary(&result);
    assert!(
        out.contains("failed: `WT` is not a covariate of the dataset"),
        "{out}"
    );
}

/// A cancelled run says so, so a short table is not read as a finished
/// pipeline.
#[test]
fn a_cancelled_run_says_so() {
    let mut result = result();
    result.cancelled = true;
    assert!(render_summary(&result).contains("cancelled"));
}
