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

use super::adaptive::{absorbed_step_warning, AbsorbedPlan};
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
    pop_at(subjects, &[0.0, 24.0, 48.0])
}

/// Subjects `(id, WT)` with an observation record at each of `times`.
fn pop_at(subjects: &[(&str, f64)], times: &[f64]) -> Population {
    let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,MDV,WT\n");
    for (id, wt) in subjects {
        for t in times {
            s.push_str(&format!("{id},{t},0,0,0,1,0,{wt}\n"));
        }
    }
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    f.write_all(s.as_bytes()).expect("write csv");
    crate::read_nonmem_csv(f.path(), None, None).expect("data reads")
}

fn run(p: &ParsedModel, population: &Population) -> Result<AdaptiveSimulationResult, EngineError> {
    run_n(p, population, 1)
}

fn run_n(
    p: &ParsedModel,
    population: &Population,
    n_sim: usize,
) -> Result<AdaptiveSimulationResult, EngineError> {
    let spec = p.adaptive_dosing.as_ref().expect("block present");
    let opts = AdaptiveSimulateOptions {
        seed: Some(17),
        ..Default::default()
    };
    simulate_adaptive_from_spec(
        &p.model,
        population,
        &p.model.default_params,
        n_sim,
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

// ── #2810: `W_ADAPTIVE_STEP_ABSORBED` ───────────────────────────────────────────────

/// The run's `W_ADAPTIVE_STEP_ABSORBED` lines.
fn absorbed_warnings(r: &AdaptiveSimulationResult) -> Vec<&str> {
    r.warnings
        .iter()
        .filter(|w| w.starts_with("W_ADAPTIVE_STEP_ABSORBED"))
        .map(String::as_str)
        .collect()
}

/// The shipped example, a 20 kg and a 40 kg subject. At 20 kg the start dose
/// `15 * WT = 300` is placed at 250 and the cap `30 * WT = 600` aligns to 500;
/// `increase 25%` gives 312.5, which rounds back to 250. The warning names that subject
/// and dose, and the ledger shows what it says: the increase rule fired and every dose
/// is 250. At 40 kg the start dose is 500, which the same rule raises (625 is a tie, up
/// to 750), so that subject is not named, and its ledger does rise.
///
/// The message is asserted whole: deleting any sentence, the count, the dose or the
/// lowest raisable dose reddens this test. The absence of `'40'` is asserted separately.
///
/// Mutations: the check on the first subject only, or on the last; the check run on
/// the unplaced start dose (300 * 1.25 = 375 rounds to 500, so no warning); a warning
/// for every subject with `round_to`.
#[test]
fn the_shipped_rounded_example_warns_for_the_subject_the_grid_holds_at_250() {
    let p = crate::parser::model_parser::parse_full_model_file(std::path::Path::new(
        "examples/adaptive_vanco_mgkg_rounded.ferx",
    ))
    .expect("the shipped example parses");
    let population = pop_at(&[("40", 40.0), ("20", 20.0)], &[0.0, 72.0, 156.0]);
    let r = run(&p, &population).expect("runs");

    assert_eq!(
        absorbed_warnings(&r),
        [
            "W_ADAPTIVE_STEP_ABSORBED: with round_to = 250, `increase 25%` (the smallest \
          increase step) cannot raise the lowest dose of 1 of 2 subject(s): the step rounds \
          back to the same dose, so each time that rule fires there the dose stays where it \
          is. From the start dose on: '20' at 250. The lowest dose this step can raise is \
          500. To let these doses rise, use a smaller round_to or a larger increase step; a \
          dose_bounds low of at least 500 keeps every dose out of this range."
        ]
    );
    assert!(!absorbed_warnings(&r)[0].contains("'40'"));

    let of = |id: &str| -> Vec<&crate::sim::adaptive::DoseLedgerEntry> {
        r.ledger.iter().filter(|e| e.subject == id).collect()
    };
    let light = of("20");
    assert_eq!(light.len(), 13, "one dose per decision: {light:?}");
    for e in &light {
        assert_eq!(e.amt.to_bits(), 250.0f64.to_bits(), "{e:?}");
    }
    let fired = light
        .iter()
        .filter(|e| e.rule_fired.ends_with("increase 25%"))
        .count();
    assert!(
        fired >= 2,
        "the increase rule fired {fired} time(s): {light:?}"
    );
    // The subject that is not named does climb, under the same rule.
    let heavy = of("40");
    assert_eq!(heavy[0].amt.to_bits(), 500.0f64.to_bits(), "{:?}", heavy[0]);
    assert!(heavy.iter().any(|e| e.amt > 500.0), "{heavy:?}");
}

/// Both sides of the threshold through the whole engine: from 500, `increase 25%` is
/// exactly half a step (625, a tie, up to 750) and the run does not warn, and its dose
/// rises; `increase 24.99%` is just under and the run warns, and its dose stays.
///
/// Mutations: ties down in the check only (the 25% run warns while its ledger rises);
/// a warning whenever `round_to` is set.
#[test]
fn the_warning_straddles_half_a_step_and_agrees_with_the_ledger() {
    let one = pop(&[("1", 70.0)]);
    let run_with = |pct: &str| {
        let rule = format!("signal < 1000000 : increase {pct}%");
        run(&model(&block("500", "[250, 4000]", "", &rule)), &one).expect("runs")
    };
    let at = run_with("25");
    assert_eq!(absorbed_warnings(&at), [""; 0], "{:?}", at.warnings);
    assert_eq!(
        at.ledger[0].amt.to_bits(),
        750.0f64.to_bits(),
        "{:?}",
        at.ledger
    );

    let below = run_with("24.99");
    assert_eq!(absorbed_warnings(&below).len(), 1, "{:?}", below.warnings);
    assert!(
        absorbed_warnings(&below)[0].contains("`increase 24.99%`")
            && absorbed_warnings(&below)[0].contains("1 of 1 subject(s)")
            && absorbed_warnings(&below)[0].contains("From the start dose on: '1' at 500."),
        "{:?}",
        below.warnings
    );
    for e in &below.ledger {
        assert_eq!(e.amt.to_bits(), 500.0f64.to_bits(), "{e:?}");
    }
}

/// Per subject, with bounds and start dose that are expressions, and both cases in one
/// line. `start_dose = 10 * WT`, `dose_bounds = [5 * WT, 4000]`, `increase 10%` (raises
/// a dose from 1250 up) and `decrease 25%`:
///
/// - `a`, 100 kg: starts at 1000, which is absorbed.
/// - `b`, 150 kg: starts at 1500, which rises; the decreases bring it to its low bound
///   750 (1500, 1250, 1000, 750), which is absorbed.
/// - `c`, 300 kg: starts at 3000; the decreases stop at its low bound 1500, which rises.
///   Not named.
///
/// Asserted whole, so each sentence and each group is pinned; `c` is asserted absent.
///
/// Mutations: one plan checked for every subject (the block's first subject's); the two
/// groups merged or swapped; `c` counted in "of 3".
#[test]
fn each_subject_is_checked_on_its_own_plan_and_the_two_cases_are_named_apart() {
    let p = model(&block(
        "10 * WT",
        "[5 * WT, 4000]",
        "  when signal > 1000000 : decrease 25%\n",
        "signal < 0 : increase 10%",
    ));
    let r = run(&p, &pop(&[("c", 300.0), ("a", 100.0), ("b", 150.0)])).expect("runs");
    assert_eq!(
        absorbed_warnings(&r),
        [
            "W_ADAPTIVE_STEP_ABSORBED: with round_to = 250, `increase 10%` (the smallest \
          increase step) cannot raise the lowest dose of 2 of 3 subject(s): the step rounds \
          back to the same dose, so each time that rule fires there the dose stays where it \
          is. From the start dose on: 'a' at 1000. Once the decrease rules have brought the \
          dose down: 'b' at 750. The lowest dose this step can raise is 1250. To let these \
          doses rise, use a smaller round_to or a larger increase step; a dose_bounds low \
          of at least 1250 keeps every dose out of this range."
        ]
    );
    assert!(!absorbed_warnings(&r)[0].contains("'c'"));
}

/// An oracle subject's plan is evaluated in each replicate, after its draw, so the
/// check runs there too and the line names the replicates: how many of the run's, and
/// the first. Three replicates, each with a start dose of 500 under `increase 10%`.
///
/// Mutations: oracle plans not checked (no line); one entry per replicate (the subject
/// counted three times, "3 of 1"); the last replicate named as the first.
#[test]
fn an_oracle_subject_is_named_with_its_replicates() {
    let p = model(&block(
        "500 + 0 * CL",
        "[250, 4000]",
        "  dose_knowledge = oracle\n",
        "signal < 0 : increase 10%",
    ));
    let r = run_n(&p, &pop(&[("1", 70.0)]), 3).expect("runs");
    assert_eq!(absorbed_warnings(&r).len(), 1, "{:?}", r.warnings);
    let w = absorbed_warnings(&r)[0];
    assert!(w.contains("of 1 of 1 subject(s)"), "{w}");
    assert!(
        w.contains("From the start dose on: '1' at 500 (in 3 of 3 replicate(s), first in 1)."),
        "{w}"
    );
}

/// More subjects than are named: the first ten in population order, then a count. The
/// plans are handed over in reverse, as replicates of a parallel run can arrive.
///
/// Mutations: no cap; the count of all subjects in place of the rest; arrival order.
#[test]
fn subjects_past_the_tenth_are_counted_not_named() {
    let p = model(&block(
        "500",
        "[250, 4000]",
        "",
        "signal < 0 : increase 10%",
    ));
    let spec = p.adaptive_dosing.as_ref().expect("block present");
    let ids: Vec<String> = (1..=12).map(|i| format!("s{i}")).collect();
    let subjects: Vec<(&str, f64)> = ids.iter().map(|id| (id.as_str(), 70.0)).collect();
    let population = pop(&subjects);
    let plans = (0..12)
        .rev()
        .map(|subject| AbsorbedPlan {
            subject,
            replicate: None,
            found: crate::sim::adaptive::AbsorbedIncrease::FromStart(500.0),
        })
        .collect();
    let w = absorbed_step_warning(spec, &population, 1, plans).expect("a line");
    assert!(w.contains("of 12 of 12 subject(s)"), "{w}");
    assert!(
        w.contains(
            "From the start dose on: 's1' at 500, 's2' at 500, 's3' at 500, 's4' at 500, \
             's5' at 500, 's6' at 500, 's7' at 500, 's8' at 500, 's9' at 500, 's10' at 500 \
             and 2 more. The lowest"
        ),
        "{w}"
    );
    assert_eq!(
        absorbed_step_warning(spec, &population, 1, Vec::new()),
        None
    );
}
