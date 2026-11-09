//! `round_to` in `[adaptive_dosing]` (#944 PR3), through the public entry points: the
//! per-subject grid refusals V7 and V8 (bounds that are expressions are checked per
//! subject, and per replicate for an oracle subject), and an absorbed step in the
//! ledger and metrics.
//!
//! Adaptive-dosing exception (AGENTS.md): no NONMEM. The oracles are hand-computed grid
//! values (`s = 250`, so every grid dose is an exact integer); the end-to-end reference
//! is the `round_to = 250` arm of `tests/reference/vanco_mgkg_mrgsolve/`.
//!
//! Each test names the regression it exists to catch.

use super::*;
use crate::diagnostics::EngineError;
use crate::parser::model_parser::parse_full_model;
use crate::types::ParsedModel;
use std::io::Write;

/// 1-cpt bolus, CL and V scaled by weight, η on CL (variance 0.09, so a drawn CL is
/// visibly not the typical one).
fn model(block: &str) -> ParsedModel {
    parse_full_model(&format!(
        "[parameters]
  theta TVCL(6.0, 0.1, 50.0)
  theta TVV(0.7, 0.05, 5.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 1.0
[individual_parameters]
  CL = TVCL * (WT / 70) * exp(ETA_CL)
  V  = TVV * WT
[structural_model]
  ode(states=[CENT])
[odes]
  d/dt(CENT) = -(CL / V) * CENT
[scaling]
  y = CENT / V
[error_model]
  DV ~ additive(ADD)
[adaptive_dosing]
{block}"
    ))
    .expect("model parses")
}

/// A block with `round_to = 250`, the given start dose and bounds, decisions at 0, 24
/// and 48, and the one rule `rule`.
fn block(start: &str, bounds: &str, extra: &str, rule: &str) -> String {
    format!(
        "  observe = CENT / V
  at = [0, 24, 48]
  start_dose = {start}
  route = bolus(cmt=1)
  dose_bounds = {bounds}
  round_to = 250
{extra}  when {rule}"
    )
}

fn pop(subjects: &[(&str, f64)]) -> Population {
    let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,MDV,WT\n");
    for (id, wt) in subjects {
        for t in [0.0, 24.0, 48.0] {
            s.push_str(&format!("{id},{t},0,0,0,1,0,{wt}\n"));
        }
    }
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    f.write_all(s.as_bytes()).expect("write csv");
    crate::read_nonmem_csv(f.path(), None, None).expect("data reads")
}

fn run(p: &ParsedModel, population: &Population) -> Result<AdaptiveSimulationResult, EngineError> {
    let spec = p.adaptive_dosing.as_ref().expect("block present");
    let opts = AdaptiveSimulateOptions {
        seed: Some(17),
        ..Default::default()
    };
    simulate_adaptive_from_spec(
        &p.model,
        population,
        &p.model.default_params,
        1,
        spec,
        &opts,
    )
}

/// V7 per subject: `[10 * WT, 12 * WT]` is `[200, 240]` at WT 20 (no multiple of 250)
/// and `[1000, 1200]` at WT 100. The run is refused before any subject runs, naming
/// subject 1 and not subject 2, which sits first in the population so a check of the
/// first subject only would miss subject 1. The message is asserted whole, so deleting
/// either of its sentences (the refusal, or the reads) reddens this test.
///
/// Mutations: V7 checked on the first subject only, or only at parse time; the wrong
/// subject named; the evaluated bounds dropped from the text.
#[test]
fn bounds_holding_no_multiple_are_refused_per_subject() {
    let p = model(&block(
        "10 * WT",
        "[10 * WT, 12 * WT]",
        "",
        "signal > 1000000 : decrease 25%",
    ));
    let e = run(&p, &pop(&[("2", 100.0), ("1", 20.0)])).expect_err("subject 1 has no grid");
    assert_eq!(e.code(), Some("E_ADAPTIVE_DOSE_VALUE"), "{e}");
    assert_eq!(
        e.message(),
        "[adaptive_dosing] subject 1: no multiple of round_to = 250 lies within dose_bounds \
         = [10 * WT, 12 * WT], which evaluate to [200, 240]. The dose expressions read \
         WT = 20, at the first decision (t = 0)."
    );
    assert!(!e.message().contains("subject 2"), "{e}");
    run(&p, &pop(&[("2", 100.0)])).expect("subject 2 alone has a grid");
}

/// V8 per subject: `WT - 20` is a low of 0 at WT 20 and 50 at WT 70 (which aligns to
/// 250). With a percentage step the WT-20 subject is refused, naming it and not the
/// other; the message is asserted whole, so each clause — the evaluated bounds, the
/// cause, the consequence, the advice — is killed by deleting it.
///
/// Mutations: V8 per run instead of per subject; V8 on the raw low (the WT-70 subject,
/// low 50 < 250, refused too); the advice dropped.
#[test]
fn a_low_bound_aligning_to_zero_is_refused_per_subject() {
    let p = model(&block(
        "500",
        "[WT - 20, 4000]",
        "",
        "signal < 1000000 : increase 25%",
    ));
    let e = run(&p, &pop(&[("2", 70.0), ("1", 20.0)])).expect_err("subject 1: low 0");
    assert_eq!(e.code(), Some("E_ADAPTIVE_DOSE_VALUE"), "{e}");
    assert_eq!(
        e.message(),
        "[adaptive_dosing] subject 1: with round_to = 250, dose_bounds = [WT - 20, 4000] \
         evaluate to [0, 4000]: a low of 0 lets a dose round to 0, and a running dose of 0 \
         cannot rise again under percentage steps; set the lower bound above 0 (any value \
         up to 250 raises it to 250). The dose expressions read WT = 20, at the first \
         decision (t = 0)."
    );
    assert!(!e.message().contains("subject 2"), "{e}");
    run(&p, &pop(&[("2", 70.0)])).expect("low 50 aligns to 250");
}

/// V7 and V8 on an oracle subject: the bounds read CL, so they are evaluated per
/// replicate from the drawn CL, and the refusal names the replicate. `[CL, CL + 1]` is
/// about `[6, 7]` (no multiple of 250); `[0 * CL, 4000]` has a low of 0.
///
/// Mutations: the grid checks skipped on the oracle path (`oracle_dose_plan`); the
/// replicate dropped from the oracle message.
#[test]
fn an_oracle_subject_grid_refusal_names_the_replicate() {
    let oracle = "  dose_knowledge = oracle\n";
    let p = model(&block(
        "CL",
        "[CL, CL + 1]",
        oracle,
        "signal > 1000000 : decrease 25%",
    ));
    let e = run(&p, &pop(&[("1", 70.0)])).expect_err("no multiple of 250 near CL");
    assert_eq!(e.code(), Some("E_ADAPTIVE_DOSE_VALUE"), "{e}");
    let head = "[adaptive_dosing] subject 1, replicate 1: no multiple of round_to = 250 lies \
                within dose_bounds = [CL, CL + 1], which evaluate to [";
    assert!(e.message().starts_with(head), "{e}");

    let p = model(&block(
        "500",
        "[0 * CL, 4000]",
        oracle,
        "signal < 1000000 : increase 25%",
    ));
    let e = run(&p, &pop(&[("1", 70.0)])).expect_err("low 0");
    assert_eq!(e.code(), Some("E_ADAPTIVE_DOSE_VALUE"), "{e}");
    let head = "[adaptive_dosing] subject 1, replicate 1: with round_to = 250, dose_bounds = \
                [0 * CL, 4000] evaluate to [0, 4000]: a low of 0 lets a dose round to 0";
    assert!(e.message().starts_with(head), "{e}");
}

/// An absorbed step through the whole engine: 500 +10% is 550, which rounds back to
/// 500, at every decision. Each ledger row names the rule that fired and keeps
/// `amt = 500` — the record a step clamped at a bound produces — and the metrics count
/// no increase, since they count realized changes. No new audit field.
///
/// Mutations: the absorbed step logged as a plain re-issue (`rule_fired = "bolus"`);
/// the unrounded 550 reaching the ledger.
#[test]
fn an_absorbed_step_logs_the_rule_and_counts_no_increase() {
    let p = model(&block(
        "500",
        "[250, 4000]",
        "",
        "signal < 1000000 : increase 10%",
    ));
    let r = run(&p, &pop(&[("1", 70.0)])).expect("runs");
    assert_eq!(r.ledger.len(), 3, "{:?}", r.ledger);
    for e in &r.ledger {
        assert_eq!(e.amt.to_bits(), 500.0f64.to_bits(), "{e:?}");
        assert_eq!(e.rule_fired, "signal < 1000000 : increase 10%", "{e:?}");
    }
    assert_eq!(r.metrics[0].n_increases, 0, "{:?}", r.metrics[0]);
    assert_eq!(r.metrics[0].n_decreases, 0, "{:?}", r.metrics[0]);
}
