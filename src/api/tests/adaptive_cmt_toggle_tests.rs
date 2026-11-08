//! Adaptive dosing on a subject with compartment on/off records (`EVID=2`, `CMT=-N` / `+N`;
//! #2753, lifting #2737's refusal).
//!
//! Adaptive-dosing exception (AGENTS.md): no NONMEM. The oracles are
//!
//! - the **degenerate oracle**: a controller that re-emits a fixed regimen, scored against
//!   `predict()` on base ∪ ledger. `predict()` sends a toggled subject to
//!   `event_driven_walk`, the engine #2737 anchored against NONMEM
//!   (`nonmem_anchor/cmt_toggle.ctl`), so the anchor carries over. The driver integrates with
//!   `integrate_segment`, a different walk, so the two agree to solver noise, not to the bit
//!   (as the reset oracles in `adaptive_sim_tests.rs`); [`ORACLE_TOL`] is measured;
//! - the **frozen-schedule replay**, on by default in every run here, whose `Ok` is part of
//!   each assertion: on the time-varying / IOV cells it is the replay
//!   `adaptive_frozen_replay_tv`, so those cells are its side of the mutation check;
//! - **closed forms** for the window AUC and for what a monitor reads.
//!
//! The driver-level closed-form tests (and the per-side mutation split) live in
//! `ode/predictions_tests.rs` (`cmt_toggle_*`, `adaptive_driver_applies_cmt_toggles_*`).
use super::adaptive_auc_tv_tests::{
    assert_distinct, assert_window_aucs, exact_window_aucs, fixed_dose_spec, read_pop, Seg, AUC_TOL,
};
use super::*;
use crate::diagnostics::EngineError;
use crate::parser::model_parser::parse_model_string;
use crate::sim::adaptive::{ControllerCtx, DoseAction, MonitorSpec, ObserveMode};

/// Relative band of the degenerate oracle (driver on `integrate_segment` vs `predict()` on
/// `event_driven_walk`). Measured on macOS arm64: 3.9e-7 (the oracle fixture), 7.0e-7
/// (reset × base × toggle), and 0 under the time-varying covariate, where both walks break
/// at every record and so walk the same segments. 5e-6 is 7× the worst. A dropped or
/// misplaced on/off row moves a prediction by tens of percent (the untoggled twin: > 10%).
const ORACLE_TOL: f64 = 5e-6;

/// Oral 1-cpt `[odes]` model with a `urine` state fed by elimination (the bladder case of
/// `docs/data-format.qmd`). KA = 0.3, so `depot` holds 40.7 at an off row at 3 after a 100
/// dose at 0. Elimination saturates (`central / (1 + central / 200)`) so the system is not
/// linear: a linear one is solved by an exact propagator that makes the driver and
/// `predict()` agree bit for bit, and would not measure the oracle's band. η is declared
/// but unread: the run is deterministic.
const GUT: &str = r#"
[parameters]
  theta TVKA(0.3, 0.01, 10.0)
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  KA = TVKA
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[depot, central, urine])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL / V) * central / (1 + central / 200)
  d/dt(urine) = (CL / V) * central / (1 + central / 200)
[scaling]
  y[CMT=1] = depot
  y[CMT=2] = central
  y[CMT=3] = urine
[error_model]
  DV ~ proportional(PROP)
"#;

/// One CSV row: `(TIME, EVID, AMT, CMT, RATE)`. `CMT` is signed: `-N` on an `EVID=2` row
/// turns compartment `N` off.
type Row = (f64, u8, f64, i32, f64);

fn csv(rows: &[Row]) -> String {
    let mut s = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV,RATE\n");
    for &(t, evid, amt, cmt, rate) in rows {
        let mdv = u8::from(evid != 0);
        s.push_str(&format!("1,{t},0,{evid},{amt},{cmt},{mdv},{rate}\n"));
    }
    s
}

fn obs(t: f64, cmt: i32) -> Row {
    (t, 0, 0.0, cmt, 0.0)
}
fn off(t: f64, cmt: i32) -> Row {
    (t, 2, 0.0, -cmt, 0.0)
}
fn on(t: f64, cmt: i32) -> Row {
    (t, 2, 0.0, cmt, 0.0)
}
fn bolus(t: f64, amt: f64, cmt: i32) -> Row {
    (t, 1, amt, cmt, 0.0)
}

fn run(
    model: &CompiledModel,
    rows: &[Row],
    decisions: &[f64],
    monitors: Vec<MonitorSpec>,
    controller: impl Fn(&ControllerCtx) -> Vec<DoseAction> + Clone + Send + Sync,
) -> Result<AdaptiveSimulationResult, EngineError> {
    let pop = read_pop(&csv(rows), &[]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: decisions.to_vec(),
        monitors,
        ..Default::default() // verify = true
    };
    let make = move || controller.clone();
    simulate_adaptive(model, &pop, &model.default_params, 1, make, &opts)
}

fn depot_bolus(_: &ControllerCtx) -> Vec<DoseAction> {
    vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
}

/// `predict()` on `rows` plus the realized ledger as dose rows.
fn static_preds(model: &CompiledModel, rows: &[Row], res: &AdaptiveSimulationResult) -> Vec<f64> {
    let mut all = rows.to_vec();
    all.extend(
        res.ledger
            .iter()
            .map(|e| (e.time, 1, e.amt, e.cmt as i32, e.rate)),
    );
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    let pop = read_pop(&csv(&all), &[]);
    predict(model, &pop, &model.default_params)
        .expect("predict the static twin")
        .iter()
        .map(|p| p.pred)
        .collect()
}

/// Worst relative gap, every value asserted finite first (a `NaN` must not hide in the fold).
fn worst_rel(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter().chain(want).all(|x| x.is_finite()),
        "{got:?} / {want:?}"
    );
    got.iter()
        .zip(want)
        .map(|(g, w)| ((g - w) / w).abs())
        .fold(0.0, f64::max) // nan-ok: every operand asserted finite above
}

fn ipreds(res: &AdaptiveSimulationResult) -> Vec<f64> {
    res.trajectories.iter().map(|t| t.ipred).collect()
}

/// The base subject of the oracle: `depot` off at 3 with 40.7 in it, back on by the
/// controller's dose at 6, off again at 8 with 54.9 in it, an explicit on row at 9; central
/// observed throughout (`depot` is never observed while off, so `E_OBS_CMT_OFF` does not
/// apply).
fn oracle_rows() -> Vec<Row> {
    vec![
        obs(1.0, 2),
        off(3.0, 1),
        obs(4.0, 2),
        obs(7.0, 2),
        off(8.0, 1),
        obs(8.5, 2),
        on(9.0, 1),
        obs(10.0, 2),
        obs(12.5, 2),
    ]
}

/// Degenerate oracle (#2753): a fixed-dose controller on a toggled subject reproduces
/// `predict()` on base ∪ ledger, which carries #2737's NONMEM anchor. Multi-dose; the dose
/// at 6 lands in `depot` while it is off, and turns it on. The untoggled twin is asserted
/// far away, so the rows are live.
#[test]
fn adaptive_cmt_toggle_matches_static_predict() {
    let m = parse_model_string(GUT).expect("parse");
    let rows = oracle_rows();
    let res = run(&m, &rows, &[0.0, 6.0, 12.0], vec![], depot_bolus)
        .expect("runs and passes the frozen replay");
    assert_eq!(res.ledger.len(), 3);
    let want = static_preds(&m, &rows, &res);
    let worst = worst_rel(&ipreds(&res), &want);
    eprintln!("degenerate oracle: worst rel {worst:e}");
    assert!(worst < ORACLE_TOL, "worst rel {worst:e}");

    let untoggled: Vec<Row> = rows.iter().copied().filter(|r| r.1 != 2).collect();
    let free = static_preds(&m, &untoggled, &res);
    assert!(
        worst_rel(&free, &want) > 0.1,
        "the on/off rows must move the answer"
    );
}

/// A decision exactly at an off row (#2753, decision 1 of the plan): the row applies first,
/// so a monitor on `depot` reads 0 and the dose decided there is a record after the row — it
/// lands and turns `depot` back on. The straddle pair puts the decision one ULP either side:
/// before the row the monitor reads the amount and the dose is wiped by the row (so `central`
/// at 4 equals a run that never dosed there); after it the run equals the tie.
#[test]
fn adaptive_decision_at_an_off_row_reads_after_it_and_its_dose_lands() {
    let m = parse_model_string(GUT).expect("parse");
    let rows = vec![bolus(0.0, 100.0, 1), obs(1.0, 2), off(3.0, 1), obs(4.0, 2)];
    let mon = || vec![MonitorSpec::new("D", 1, ObserveMode::Ipred)];
    let cell = |t: f64| {
        run(&m, &rows, &[t], mon(), depot_bolus).unwrap_or_else(|e| panic!("cell {t}: {e}"))
    };
    let read = |r: &AdaptiveSimulationResult| r.decisions[0].observed_signals[0].value;
    let c4 = |r: &AdaptiveSimulationResult| r.trajectories[1].ipred;

    let before = cell(3.0f64.next_down());
    let tie = cell(3.0);
    let after = cell(3.0f64.next_up());
    let held = run(&m, &rows, &[3.0], mon(), |_: &ControllerCtx| {
        vec![DoseAction::Hold]
    })
    .unwrap();

    assert!(read(&before) > 40.0, "before the row: {}", read(&before));
    assert_eq!(
        read(&tie),
        0.0,
        "at the row the monitor reads the off state"
    );
    assert_eq!(read(&after), 0.0);
    // Before: the 1-ULP-early dose lands and the row wipes it, so central at 4 is the held
    // run's to round-off. At the tie it lands after the row, as one ULP later does.
    assert!(
        ((c4(&before) - c4(&held)) / c4(&held)).abs() < 1e-9,
        "{} vs held {}",
        c4(&before),
        c4(&held)
    );
    assert!(
        ((c4(&tie) - c4(&after)) / c4(&after)).abs() < 1e-6,
        "tie {} vs after {}",
        c4(&tie),
        c4(&after)
    );
    assert!(
        c4(&tie) > 1.1 * c4(&held),
        "the dose at the tie lands: {} vs held {}",
        c4(&tie),
        c4(&held)
    );
}

/// A monitor on a compartment while it is off reads 0, though mass keeps flowing at it
/// (#2753, decision 2): `urine` is fed by elimination, so only the hold at zero keeps it
/// there. Before the off row it reads a positive amount. A `Dv` monitor there under the
/// proportional error model is not refused (the plan expected the RUV-magnitude refusal;
/// measured, there is none): its variance at 0 is the residual-error floor, so it reads
/// noise of σ = 1e-6 around 0.
#[test]
fn adaptive_monitor_on_an_off_compartment_reads_zero() {
    let m = parse_model_string(GUT).expect("parse");
    let rows = vec![bolus(0.0, 100.0, 1), obs(1.0, 2), off(3.0, 3), obs(8.0, 2)];
    let hold = |_: &ControllerCtx| vec![DoseAction::Hold];
    let res = run(
        &m,
        &rows,
        &[2.0, 6.0],
        vec![MonitorSpec::new("U", 3, ObserveMode::Ipred)],
        hold,
    )
    .expect("runs");
    let u = |i: usize| res.decisions[i].observed_signals[0].value;
    assert!(u(0) > 1.0, "urine before the off row: {}", u(0));
    assert_eq!(
        u(1),
        0.0,
        "urine while off, with central still eliminating into it"
    );

    let dv = run(
        &m,
        &rows,
        &[2.0, 6.0],
        vec![MonitorSpec::new("U", 3, ObserveMode::Dv)],
        hold,
    )
    .expect("a Dv monitor on an off compartment runs");
    let d = |i: usize| dv.decisions[i].observed_signals[0].value;
    assert_ne!(d(0), u(0), "before the row the assay draw is live");
    // While off the latent is 0, the proportional variance 0, floored to `MIN_VARIANCE`
    // (1e-12): the reading is that floor's noise, σ = 1e-6 (measured −8.6e-7 at this seed).
    assert!(
        d(1).abs() < 6e-6,
        "while off: only the variance floor's noise, got {}",
        d(1)
    );
}

/// A reset turns every compartment back on (#2753; the static walk's rule, measured in
/// `nonmem_anchor/cmt_toggle.ctl`). `urine` is off from 3; the EVID=3 reset at 5 turns it
/// on, so a monitor at 8 reads what elimination put there since the base dose at 5.5, and
/// the run matches `predict()` on the urine observation at 10 too.
#[test]
fn adaptive_cmt_toggle_with_reset_and_base_regimen() {
    let m = parse_model_string(GUT).expect("parse");
    let rows = vec![
        bolus(0.0, 100.0, 1),
        obs(1.0, 2),
        off(3.0, 3),
        obs(4.0, 2),
        (5.0, 3, 0.0, 1, 0.0),
        bolus(5.5, 100.0, 1),
        obs(7.0, 2),
        obs(10.0, 3),
    ];
    let res = run(
        &m,
        &rows,
        &[2.0, 8.0],
        vec![MonitorSpec::new("U", 3, ObserveMode::Ipred)],
        depot_bolus,
    )
    .expect("runs and passes the frozen replay");
    assert!(
        res.decisions[1].observed_signals[0].value > 1.0,
        "urine after the reset is on again"
    );
    let want = static_preds(&m, &rows, &res);
    let worst = worst_rel(&ipreds(&res), &want);
    eprintln!("reset × base × toggle: worst rel {worst:e}");
    assert!(worst < ORACLE_TOL, "worst rel {worst:e}");
}

/// Time-varying covariate × on/off rows: the driver runs on its per-event-PK path and the
/// verifier on `adaptive_frozen_replay_tv`, whose `Ok` is that replay's side of the mutation
/// check (#2753). CL steps at the EVID=2 rows, which carry their own covariate row.
#[test]
fn adaptive_cmt_toggle_under_a_time_varying_covariate() {
    let model = r#"
[parameters]
  theta TVKA(0.3, 0.01, 10.0)
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  KA = TVKA
  CL = TVCL * CRCL / 100.0
  V  = TVV
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL / V) * central / (1 + central / 200)
[scaling]
  y[CMT=1] = depot
  y[CMT=2] = central
[error_model]
  DV ~ proportional(PROP)
"#;
    let m = parse_model_string(model).expect("parse");
    let rows: [(f64, u8, i32, f64); 7] = [
        (1.0, 0, 2, 100.0),
        (3.0, 2, -1, 60.0),
        (4.0, 0, 2, 60.0),
        (7.0, 0, 2, 40.0),
        (8.0, 2, -1, 80.0),
        (9.0, 2, 1, 80.0),
        (10.0, 0, 2, 120.0),
    ];
    let to_csv = |doses: &[(f64, f64)]| {
        let mut all: Vec<(f64, u8, f64, i32, f64)> = rows
            .iter()
            .map(|&(t, e, c, cr)| (t, e, 0.0, c, cr))
            .collect();
        // A dose row is a record, and the segment ending at it runs under its row (#1073).
        // The adaptive run has no row there, so its segment runs under the next record's:
        // give each dose row that record's `CRCL` (100 at 0, 40 at 6) so both segment alike.
        all.extend(
            doses
                .iter()
                .map(|&(t, a)| (t, 1, a, 1, if t < 1.0 { 100.0 } else { 40.0 })),
        );
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut s = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV,CRCL\n");
        for (t, e, a, c, cr) in all {
            s.push_str(&format!("1,{t},0,{e},{a},{c},{},{cr}\n", u8::from(e != 0)));
        }
        s
    };
    let pop = read_pop(&to_csv(&[]), &["CRCL"]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: vec![0.0, 6.0],
        ..Default::default()
    };
    let res = simulate_adaptive(&m, &pop, &m.default_params, 1, || depot_bolus, &opts)
        .expect("runs and passes adaptive_frozen_replay_tv");
    let realized: Vec<(f64, f64)> = res.ledger.iter().map(|e| (e.time, e.amt)).collect();
    let want: Vec<f64> = predict(
        &m,
        &read_pop(&to_csv(&realized), &["CRCL"]),
        &m.default_params,
    )
    .unwrap()
    .iter()
    .map(|p| p.pred)
    .collect();
    let worst = worst_rel(&ipreds(&res), &want);
    eprintln!("tv covariate × toggle: worst rel {worst:e}");
    assert!(worst < ORACLE_TOL, "worst rel {worst:e}");
}

/// The window AUC (`auc_target`, #943) over a window that holds an off interval (#2753). A
/// 1-cpt bolus model with `central` off from 10 to 14: window [0, 12) integrates the decay
/// to 10 and nothing after; the decision at 12 doses into the off compartment, which turns it
/// on, so window [12, 24) is a plain decay from that dose (the on row at 14 restarts nothing).
/// Closed form: an off row with nothing flowing in is a reset of that compartment alone, so
/// [`exact_window_aucs`] with a "reset" at 10 is exact. The toggle-blind twin is distinct.
#[test]
fn adaptive_auc_window_spanning_an_off_interval() {
    let model = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
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
    let m = parse_model_string(model).expect("parse");
    let pop = read_pop(
        &csv(&[obs(2.0, 1), off(10.0, 1), on(14.0, 1), obs(30.0, 1)]),
        &[],
    );
    let decisions = [0.0, 12.0, 24.0];
    let spec = fixed_dose_spec(Some("central"), &decisions, 100.0);
    let k = 0.1;
    let seg = |t0: f64, t1: f64| Seg {
        t0,
        t1,
        k,
        factor: 1.0,
    };
    let segs = [seg(0.0, 10.0), seg(10.0, 12.0), seg(12.0, 24.0)];
    let boluses = [(0.0, 100.0), (12.0, 100.0)];
    let want = exact_window_aucs(&segs, &boluses, &[10.0], &decisions);
    let blind = exact_window_aucs(&segs, &boluses, &[], &decisions);
    assert_distinct(&want, &blind, AUC_TOL, "off interval vs toggle ignored");
    assert_window_aucs(
        &m,
        &pop,
        &spec,
        7,
        &want,
        AUC_TOL,
        "window over an off interval",
    );
}

/// IOV × on/off rows: per-decision κ on the driver and on `adaptive_frozen_replay_tv`; the
/// replay's `Ok` is the assertion (#2753).
#[test]
fn adaptive_cmt_toggle_under_iov_passes_the_replay() {
    let model = r#"
[parameters]
  theta TVKA(0.3, 0.01, 10.0)
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  kappa KAPPA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  KA = TVKA
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL / V) * central
[scaling]
  y[CMT=1] = depot
  y[CMT=2] = central
[error_model]
  DV ~ proportional(PROP)
"#;
    let m = parse_model_string(model).expect("parse");
    assert_eq!(m.n_kappa, 1);
    let res = run(&m, &oracle_rows(), &[0.0, 6.0, 12.0], vec![], depot_bolus)
        .expect("runs and passes adaptive_frozen_replay_tv");
    assert_eq!(res.ledger.len(), 3);
    assert!(ipreds(&res).iter().all(|x| x.is_finite() && *x > 0.0));
}
