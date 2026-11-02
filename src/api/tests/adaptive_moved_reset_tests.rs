//! `W_ADAPTIVE_DECISION_BEFORE_MOVED_RESET` (#2559): a decision the reader's move of an
//! in-order EVID=3/4 reset (#1609) put before that reset on the engine clock.
//!
//! Every fixture is read through `read_nonmem_csv`, because the move is the reader's: an
//! in-memory subject carries no written times, so its two clocks agree by construction.
//! Each test asserts its own premise (where the reader put the reset) before the warning,
//! so a reader change that stops moving the reset fails as a premise, not as a silent
//! warning. The infusion and `ADDL` geometries are the #1609 comment's (#2556 review, row 3).
use super::*;
use crate::parser::model_parser::{parse_full_model, parse_model_string};
use crate::sim::adaptive::{ControllerCtx, DoseAction};

const CODE: &str = "W_ADAPTIVE_DECISION_BEFORE_MOVED_RESET";

const MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

const HEADER: &str = "ID,TIME,DV,EVID,AMT,CMT,MDV,RATE,II,ADDL\n";

/// 600 into central at `rate` from 0, EVID=3 at 18. At rate 20 the infusion runs to 30, so
/// the reader moves the reset to 31; at rate 40 it ends at 15 and the reset stays at 18.
fn infusion_rows(id: &str, rate: f64) -> String {
    format!(
        "{id},0,0,1,600,1,1,{rate},0,0
{id},6,0,0,0,1,0,0,0,0
{id},12,0,0,0,1,0,0,0,0
{id},18,0,3,0,1,1,0,0,0
{id},20,0,0,0,1,0,0,0,0
{id},30,0,0,0,1,0,0,0,0
{id},42,0,0,0,1,0,0,0,0
"
    )
}

/// 100 q12h with `ADDL=3` (doses at 0, 12, 24, 36), EVID=3 at 18: the implied doses run
/// the watermark to 36, so the reader moves the reset to 37.
const ADDL_CSV: &str = "1,0,0,1,100,1,1,0,12,3
1,6,0,0,0,1,0,0,0,0
1,18,0,3,0,1,1,0,0,0
1,20,0,0,0,1,0,0,0,0
1,42,0,0,0,1,0,0,0,0
";

/// A bolus and an observation at 18, then EVID=3 at 18: the reader's `≤` watermark test
/// moves a reset written at the same `TIME` as the record before it, to 19.
const COTIMED_CSV: &str = "1,0,0,1,100,1,1,0,0,0
1,18,0,0,0,1,0,0,0,0
1,18,0,3,0,1,1,0,0,0
1,24,0,0,0,1,0,0,0,0
";

/// The infusion subject, then a second reset written at 40. The first move's offset (13)
/// carries into it, so it runs at 53 without being moved itself.
const TWO_RESETS_CSV: &str = "1,0,0,1,600,1,1,20,0,0
1,12,0,0,0,1,0,0,0,0
1,18,0,3,0,1,1,0,0,0
1,24,0,0,0,1,0,0,0,0
1,40,0,3,0,1,1,0,0,0
1,42,0,1,100,1,1,0,0,0
1,48,0,0,0,1,0,0,0,0
";

/// A restarting period: `TIME` goes back to 0 at the reset, so the reader moves the reset
/// to 25 (past the last record at 24) and the second period's rows with it.
const RESTART_CSV: &str = "1,0,0,1,100,1,1,0,0,0
1,6,0,0,0,1,0,0,0,0
1,24,0,0,0,1,0,0,0,0
1,0,0,3,0,1,1,0,0,0
1,0,0,1,100,1,1,0,0,0
1,6,0,0,0,1,0,0,0,0
";

fn read(rows: &str) -> Population {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut f, format!("{HEADER}{rows}").as_bytes()).unwrap();
    crate::io::datareader::read_nonmem_csv(f.path(), None, None).unwrap()
}

fn model() -> CompiledModel {
    parse_model_string(MODEL).expect("parse ODE model")
}

/// This run's `W_ADAPTIVE_DECISION_BEFORE_MOVED_RESET` lines, from a holding controller.
fn warnings_for(pop: &Population, decisions: &[f64]) -> Vec<String> {
    let m = model();
    let opts = AdaptiveSimulateOptions {
        seed: Some(11),
        decision_times: decisions.to_vec(),
        ..Default::default()
    };
    let make = || |_c: &ControllerCtx| vec![DoseAction::Hold];
    let res = simulate_adaptive(&m, pop, &m.default_params, 1, make, &opts)
        .expect("a holding controller runs");
    res.warnings
        .into_iter()
        .filter(|w| w.starts_with(CODE))
        .collect()
}

/// The premise of every moved-reset fixture: where the reader put each reset, and the
/// written `TIME` it kept for it.
fn assert_resets(pop: &Population, engine: &[f64], written: &[f64], label: &str) {
    let s = &pop.subjects[0];
    assert_eq!(s.reset_times, engine, "{label}: engine reset times");
    let got: Vec<f64> = s.reset_times.iter().map(|&t| s.param_time_at(t)).collect();
    assert_eq!(got, written, "{label}: written reset times");
}

#[test]
fn straddle_moved_reset_warns_and_unmoved_twin_is_silent() {
    // The twins differ only in RATE: 20 runs the infusion across the reset, 40 ends it
    // before. Same decisions, same written reset.
    let moved = read(&infusion_rows("1", 20.0));
    let unmoved = read(&infusion_rows("1", 40.0));
    assert_resets(&moved, &[31.0], &[18.0], "moved twin");
    assert_resets(&unmoved, &[18.0], &[18.0], "unmoved twin");
    let decisions = [0.0, 12.0, 24.0, 36.0];
    let w_moved = warnings_for(&moved, &decisions);
    let w_unmoved = warnings_for(&unmoved, &decisions);
    assert_eq!(w_moved.len(), 1, "moved twin warns once: {w_moved:?}");
    assert!(w_moved[0].contains("subject 1:"), "{}", w_moved[0]);
    assert!(
        w_unmoved.is_empty(),
        "an unmoved reset must not warn: {w_unmoved:?}"
    );
}

#[test]
fn infusion_case_names_decision_and_both_reset_times() {
    // The full text, written out rather than rebuilt from the format string, so deleting
    // or rewording any sentence of the warning fails here.
    let pop = read(&infusion_rows("1", 20.0));
    assert_resets(&pop, &[31.0], &[18.0], "infusion");
    let w = warnings_for(&pop, &[0.0, 12.0, 24.0, 36.0]);
    assert_eq!(
        w,
        vec![
            "W_ADAPTIVE_DECISION_BEFORE_MOVED_RESET: subject 1: decision time(s) 24 come \
             after the reset written at TIME 18 but before it on the engine clock, where it \
             runs at 31. The data reader moved this reset, and every row after it, later by \
             13 (#1609). `decision_times` and the `[adaptive_dosing]` block's `at` are read \
             on the engine clock, so a dose issued at these times lands before the reset, \
             and the reset removes it. To decide after the reset, add 13 to each decision \
             time written between it and the next reset."
                .to_string()
        ]
    );
}

#[test]
fn addl_case_flags_both_decisions_before_the_moved_reset() {
    // The implied doses at 24 and 36 are not records: counting them would make the reset
    // look like a restart (written 18 < 36) and silence the warning.
    let pop = read(ADDL_CSV);
    assert_resets(&pop, &[37.0], &[18.0], "ADDL");
    let w = warnings_for(&pop, &[0.0, 12.0, 24.0, 36.0]);
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(
        w[0].contains("decision time(s) 24, 36 come after the reset written at TIME 18")
            && w[0].contains("where it runs at 37.")
            && w[0].contains("later by 19 (#1609)")
            && w[0].contains("add 19 to each")
            && w[0].contains("are read on the engine clock"),
        "{}",
        w[0]
    );
}

#[test]
fn decision_after_the_moved_reset_is_silent() {
    // 40 is after the reset on both clocks (18 written, 31 engine); 12 is before it on both.
    let pop = read(&infusion_rows("1", 20.0));
    assert_resets(&pop, &[31.0], &[18.0], "infusion");
    let w = warnings_for(&pop, &[12.0, 40.0]);
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn decision_at_the_written_reset_warns_and_at_the_engine_reset_does_not() {
    // The driver applies a reset before the decision hook at the same break, so a decision
    // at a reset's instant is after it: at 18 it is after the written reset and before the
    // engine one; at 31 it is after both.
    let pop = read(&infusion_rows("1", 20.0));
    assert_resets(&pop, &[31.0], &[18.0], "infusion");
    let w = warnings_for(&pop, &[18.0, 31.0]);
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(
        w[0].contains("decision time(s) 18 come after"),
        "18 flagged, 31 not: {}",
        w[0]
    );
}

#[test]
fn reset_cotimed_with_the_previous_record_warns() {
    let pop = read(COTIMED_CSV);
    assert_resets(&pop, &[19.0], &[18.0], "co-timed");
    let w = warnings_for(&pop, &[0.0, 18.0, 24.0]);
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(
        w[0].contains("decision time(s) 18 come after the reset written at TIME 18")
            && w[0].contains("where it runs at 19.")
            && w[0].contains("later by 1 (#1609)"),
        "{}",
        w[0]
    );
}

#[test]
fn every_moved_reset_is_checked_not_only_the_first() {
    let pop = read(TWO_RESETS_CSV);
    assert_resets(&pop, &[31.0, 53.0], &[18.0, 40.0], "two resets");
    let w = warnings_for(&pop, &[0.0, 24.0, 45.0]);
    assert_eq!(w.len(), 2, "one line per reset: {w:?}");
    assert!(
        w[0].contains("decision time(s) 24 come after the reset written at TIME 18")
            && w[0].contains("runs at 31."),
        "{}",
        w[0]
    );
    assert!(
        w[1].contains("decision time(s) 45 come after the reset written at TIME 40")
            && w[1].contains("runs at 53."),
        "{}",
        w[1]
    );
}

#[test]
fn restarting_period_is_silent() {
    // The reset is moved (0 → 25), and decision 6 lies after it on the written clock and
    // before it on the engine clock, so the side test alone would flag it. It is a
    // restart, though: `TIME` 6 exists in both periods.
    let pop = read(RESTART_CSV);
    assert_resets(&pop, &[25.0], &[0.0], "restart");
    let d = 6.0;
    assert!(d >= 0.0 && d < 25.0, "premise: the side test straddles");
    let w = warnings_for(&pop, &[0.0, d, 31.0]);
    assert!(w.is_empty(), "a restarting period must not warn: {w:?}");
}

/// Period 1 at 0–24, a restart at `TIME` 0 (engine 25), then 600 @ 20 from written 0
/// (engine 25 to 55) and an in-order EVID=3 written at 30, which the infusion moves to 56.
/// Written 30 is above every period-1 `TIME`, so it passes the in-order test, but after a
/// restart the user writes decisions on the engine clock: 40 is period-2 written 15, before
/// that reset as meant.
const RESET_AFTER_RESTART_CSV: &str = "1,0,0,1,100,1,1,0,0,0
1,6,0,0,0,1,0,0,0,0
1,24,0,0,0,1,0,0,0,0
1,0,0,3,0,1,1,0,0,0
1,0,0,1,600,1,1,20,0,0
1,6,0,0,0,1,0,0,0,0
1,30,0,3,0,1,1,0,0,0
1,36,0,0,0,1,0,0,0,0
";

#[test]
fn a_reset_after_a_restart_is_not_checked() {
    // #2595 review r1, row 1: this warned "decision time(s) 40 come after the reset written
    // at TIME 30 … add 26". 40 is before that reset as meant, and 25 of the 26 is the
    // restart's offset, not the reset's own move.
    let pop = read(RESET_AFTER_RESTART_CSV);
    assert_resets(&pop, &[25.0, 56.0], &[0.0, 30.0], "reset after a restart");
    let d = 40.0;
    assert!(
        d >= 30.0 && d < 56.0,
        "premise: the side test straddles the second reset"
    );
    let w = warnings_for(&pop, &[25.0, d]);
    assert!(
        w.is_empty(),
        "no reset after a restart is on the decision clock: {w:?}"
    );
}

#[test]
fn from_spec_with_an_every_grid_warns() {
    let src = format!(
        "{MODEL}
[adaptive_dosing]
  observe = central
  at = every 12 from 0 to 48
  start_dose = 0
  route = bolus(cmt=1)
  dose_bounds = [0, 0]
  when signal < -1000000000 : increase 25%
"
    );
    let parsed = parse_full_model(&src).expect("parse model with [adaptive_dosing]");
    let (m, spec) = (
        &parsed.model,
        parsed.adaptive_dosing.as_ref().expect("block parsed"),
    );
    let pop = read(&infusion_rows("1", 20.0));
    assert_resets(&pop, &[31.0], &[18.0], "infusion");
    let mut opts = AdaptiveSimulateOptions::default();
    opts.seed = Some(11);
    let res =
        simulate_adaptive_from_spec(m, &pop, &m.default_params, 1, spec, &opts).expect("spec run");
    let w: Vec<&String> = res
        .warnings
        .iter()
        .filter(|w| w.starts_with(CODE))
        .collect();
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(w[0].contains("decision time(s) 24 come after"), "{}", w[0]);
}

#[test]
fn more_than_ten_lines_are_counted_in_one_closing_line() {
    let rows: String = (1..=11)
        .map(|i| infusion_rows(&i.to_string(), 20.0))
        .collect();
    let pop = read(&rows);
    let w = warnings_for(&pop, &[24.0]);
    assert_eq!(w.len(), 11, "{w:?}");
    assert!(w[9].contains("subject 10:"), "{}", w[9]);
    assert_eq!(
        w[10],
        "W_ADAPTIVE_DECISION_BEFORE_MOVED_RESET: and 1 more reset(s) like these, not listed."
    );
}
