//! A monitor's readout reads the `TIME` as written, not the engine clock, after a reset
//! that restarts `TIME` (#2756).
//!
//! The reader shifts a restarted period onto the monotonic engine clock (#1969). Every
//! static engine and the driver's own trajectory rows read the written time
//! (`Subject::readout_time`), and so does the decision's PK. The monitor read engine
//! `t_start`, so a `TIME`-reading readout handed the controller a value off by the reset
//! shift, which no output row reports.
//!
//! Each fixture is read through `read_nonmem_csv`, because the shift is the reader's: an
//! in-memory subject carries no written times, so its two clocks agree by construction.
//! Each test asserts the premise (where the reader put the period) before the readings.
//! Adaptive-dosing exception (AGENTS.md): no NONMEM. The oracles are the driver's own
//! trajectory row at the same point and exact closed forms of a linear readout.
use super::adaptive_auc_tv_tests::{assert_distinct, assert_window_aucs, fixed_dose_spec, AUC_TOL};
use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::sim::adaptive::{ControllerCtx, DoseAction, MonitorSpec, ObserveMode};

/// A readout that carries `1000 · TIME` on top of the concentration, so a wrong clock moves
/// the value by `1000 ×` the reset shift.
const MODEL_TIME_READOUT: &str = r#"
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
  y = central / V + 1000.0 * TIME
[error_model]
  DV ~ proportional(PROP)
"#;

const HEADER: &str = "ID,TIME,DV,EVID,AMT,CMT,MDV\n";

/// A dose at 0 and records at 6 and 24, then `EVID=3` at a written 0 that restarts `TIME`,
/// a dose there and a record at a written 6. The reader moves the reset to engine 25 (past
/// the last record at 24), so the second period runs 25 ahead of its written `TIME`.
const RESTART_CSV: &str = "1,0,0,1,100,1,1
1,6,0,0,0,1,0
1,24,0,0,0,1,0
1,0,0,3,0,1,1
1,0,0,1,100,1,1
1,6,0,0,0,1,0
";

const SHIFT: f64 = 25.0;

fn read(rows: &str) -> Population {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut f, format!("{HEADER}{rows}").as_bytes()).unwrap();
    crate::io::datareader::read_nonmem_csv(f.path(), None, None).unwrap()
}

/// The premise: the reader put the reset at engine 25 and the last record at engine 31,
/// written 6, so the two clocks differ by `SHIFT` in the second period.
fn assert_restart_premise(pop: &Population) {
    let s = &pop.subjects[0];
    assert_eq!(s.reset_times, [SHIFT], "engine reset time");
    assert_eq!(s.obs_times, [6.0, 24.0, 31.0], "engine observation times");
    assert_eq!(
        s.obs_raw_times,
        [6.0, 24.0, 6.0],
        "written observation times"
    );
}

/// The decision hook (`[scaling]` readout arm). Decisions at 6 (first period, both clocks
/// agree) and 31 (second period, written 6) each sit on a record, so the monitor's reading
/// must equal that record's trajectory row to the bit: the same state, PK, η, covariate
/// row and now the same `TIME`. The first decision is the straddle's control side; the
/// second differed by `1000 × 25` on `main` (31001.2259… against 6001.2259…, measured on
/// #2756), and the straddle is asserted so it cannot silently become a tautology.
///
/// Mutation: pass engine `t_start` at the decision hook again. The t = 31 assertion dies,
/// and the AUC test stays green (measured).
#[test]
fn monitor_readout_reads_the_written_time_after_a_restart() {
    let m = parse_model_string(MODEL_TIME_READOUT).expect("parse");
    let pop = read(RESTART_CSV);
    assert_restart_premise(&pop);
    let opts = AdaptiveSimulateOptions {
        seed: Some(11),
        decision_times: vec![6.0, 31.0],
        monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Ipred)],
        ..Default::default()
    };
    let make = || |_c: &ControllerCtx| vec![DoseAction::Hold];
    let res = simulate_adaptive(&m, &pop, &m.default_params, 1, make, &opts).expect("runs");
    assert_eq!(res.decisions.len(), 2);
    assert_eq!(res.trajectories.len(), 3);

    // Decision k sits on record `on_row[k]`; the straddle: the clocks agree at the first
    // and differ by the shift at the second.
    let s = &pop.subjects[0];
    let on_row = [0usize, 2];
    assert_eq!(s.obs_times[on_row[0]], s.obs_raw_times[on_row[0]]);
    assert_eq!(s.obs_times[on_row[1]] - s.obs_raw_times[on_row[1]], SHIFT);

    for (k, d) in res.decisions.iter().enumerate() {
        let got = d.observed_signals[0].value;
        let row = &res.trajectories[on_row[k]];
        assert!(got.is_finite(), "decision {k}: monitor read {got}");
        assert_eq!(
            got.to_bits(),
            row.ipred.to_bits(),
            "decision {k} at engine t={}: the monitor read {got}, its record's trajectory row \
             (TIME as written = {}) reads {}",
            d.time,
            row.time,
            row.ipred
        );
    }
}

/// The window AUC (`observe` arm). `observe = 1000 · TIME` makes each window AUC the exact
/// integral of `1000 · written(t)` over the engine window, whatever the doses. Decisions at
/// 0, 10, 25 and 37 put the restart on a window edge:
///
/// - `[0, 10]`: 1000 · 10² / 2 = 50 000;
/// - `[10, 25]`: 1000 · (25² − 10²) / 2 = 262 500. Its last node is the pre-event value at
///   the reset, a left limit, which must stay in the first period (written 25), not read
///   the reset row's restarted 0;
/// - `[25, 37]`: written 0 … 12, so 1000 · 12² / 2 = 72 000. On the engine clock it would
///   read 25 … 37, 1000 · (37² − 25²) / 2 = 372 000.
///
/// Mutations, each measured to redden this test alone:
/// - the AUC nodes on engine time: the `[25, 37]` window dies ("window 2"). `assert_distinct`
///   shows the engine-clock closed form is far outside the band;
/// - the pre-event node read through `param_time_at(t_end)`: the `[10, 25]` window dies
///   ("window 1"), because its last panel reads 0 for 25 000.
#[test]
fn window_auc_integrates_the_written_time_after_a_restart() {
    let m = parse_model_string(MODEL_TIME_READOUT).expect("parse");
    let pop = read(RESTART_CSV);
    assert_restart_premise(&pop);
    let decisions = [0.0, 10.0, SHIFT, 37.0];
    let spec = fixed_dose_spec(Some("1000.0 * TIME"), &decisions, 100.0);
    let half_sq = |a: f64, b: f64| 1000.0 * (b * b - a * a) / 2.0;
    let want = [
        half_sq(0.0, 10.0),
        half_sq(10.0, 25.0),
        half_sq(0.0, 37.0 - SHIFT),
    ];
    let engine_clock = [half_sq(0.0, 10.0), half_sq(10.0, 25.0), half_sq(25.0, 37.0)];
    assert_distinct(
        &want[2..],
        &engine_clock[2..],
        AUC_TOL,
        "the restarted window, written vs engine clock",
    );
    assert_window_aucs(
        &m,
        &pop,
        &spec,
        1,
        &want,
        AUC_TOL,
        "observe = 1000·TIME across a restart",
    );
}

/// The same readout with the restart **strictly inside** a window, and the next window
/// opening at a break that is no data record (review r1, row 2). Decisions at 0, 20, 28
/// and 40:
///
/// - `[0, 20]`: 1000 · 20² / 2 = 200 000;
/// - `[20, 28]` holds the reset at 25. It integrates written 20 … 25, then written 0 … 3:
///   112 500 + 4 500 = 117 000 (engine clock: 192 000);
/// - `[28, 40]` opens at decision 28, a break with no row in the reset-period table, so the
///   first segment's offset comes from `param_time_at`'s period arithmetic, not from a row
///   lookup. Written 3 … 15: 108 000 (engine clock: 408 000).
///
/// Mutations, measured. `assert_window_aucs` stops at the first window out of band, so each
/// row names that window:
/// - drop the segment offset (nodes after `t_start` read engine `t`): `[20, 28]` dies here,
///   and `[25, 37]` in the edge-placed test;
/// - the AUC nodes on engine time: `[20, 28]` dies;
/// - every node through `param_time_at(t)` (no left limit): `[20, 28]` dies, because its
///   pre-event node at the reset reads the restarted 0.
#[test]
fn window_auc_integrates_the_written_time_with_the_restart_inside_a_window() {
    let m = parse_model_string(MODEL_TIME_READOUT).expect("parse");
    let pop = read(RESTART_CSV);
    assert_restart_premise(&pop);
    let decisions = [0.0, 20.0, 28.0, 40.0];
    // Premise: the restart lies strictly inside the second window, and the third window
    // opens at a time that is no data record.
    assert!(decisions[1] < SHIFT && SHIFT < decisions[2]);
    let s = &pop.subjects[0];
    assert!(!s.obs_times.contains(&decisions[2]) && !s.reset_times.contains(&decisions[2]));
    let spec = fixed_dose_spec(Some("1000.0 * TIME"), &decisions, 100.0);
    let half_sq = |a: f64, b: f64| 1000.0 * (b * b - a * a) / 2.0;
    let want = [
        half_sq(0.0, 20.0),
        half_sq(20.0, SHIFT) + half_sq(0.0, 28.0 - SHIFT),
        half_sq(28.0 - SHIFT, 40.0 - SHIFT),
    ];
    let engine_clock = [half_sq(0.0, 20.0), half_sq(20.0, 28.0), half_sq(28.0, 40.0)];
    assert_distinct(
        &want[1..],
        &engine_clock[1..],
        AUC_TOL,
        "the windows after the restart, written vs engine clock",
    );
    assert_window_aucs(
        &m,
        &pop,
        &spec,
        1,
        &want,
        AUC_TOL,
        "observe = 1000·TIME, restart inside a window",
    );
}

/// The `observe` arm at decisions (review r1, row 3), including one **exactly on** the
/// reset. `observe = 1000 · TIME` reads no state, so each reading is `1000 ×` the written
/// TIME, exactly: 0 at decision 0, 0 at the reset (applied before the decision hook, so
/// the restarted value), and 6000 at engine 31 (written 6). On the engine clock the last
/// two would read 25 000 and 31 000. Decision 0 is the straddle's control, where both
/// clocks agree.
///
/// Mutation: pass engine `t_start` at the decision hook. The decisions at 25 and 31 die
/// here, as does `monitor_readout_reads_the_written_time_after_a_restart`.
#[test]
fn observe_reads_the_written_time_at_decisions_including_one_on_the_reset() {
    let m = parse_model_string(MODEL_TIME_READOUT).expect("parse");
    let pop = read(RESTART_CSV);
    assert_restart_premise(&pop);
    let decisions = [0.0, SHIFT, 31.0];
    let spec = fixed_dose_spec(Some("1000.0 * TIME"), &decisions, 100.0);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default()
    };
    let res = simulate_adaptive_from_spec(&m, &pop, &m.default_params, 1, &spec, &opts)
        .expect("spec run");
    let got: Vec<f64> = res
        .decisions
        .iter()
        .map(|d| d.observed_signals[0].value)
        .collect();
    let want = [0.0, 0.0, 6000.0];
    let engine_clock = [0.0, 25_000.0, 31_000.0];
    assert_eq!(
        want[0], engine_clock[0],
        "the control decision: both clocks agree"
    );
    assert!(want[1..]
        .iter()
        .zip(&engine_clock[1..])
        .all(|(w, e)| w != e));
    assert_eq!(
        got, want,
        "observe = 1000·TIME at decisions 0 / 25 (the reset) / 31"
    );
}
