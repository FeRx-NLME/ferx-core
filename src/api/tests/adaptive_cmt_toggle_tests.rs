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
/// lands and turns `depot` back on. That holds for a row at the decision's engine time; the
/// last cells pin the file-order exception (a row listed after a co-timed dose). The straddle pair puts the decision one ULP either side:
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

    // File order (review r1 #1). A row listed after a dose at the same TIME is placed one ULP
    // above it by the reader (`order_cmt_toggle_ties`), and a decision reads before the doses
    // at its time (#933). So a decision at that written TIME comes before the row: the monitor
    // reads the amount, and a dose into the switched compartment is wiped by the row, which is
    // what a dose row written ahead of the off row does in `predict()`. Listed before the
    // dose, the row is placed one ULP below it and the decision comes after it. The two cells
    // differ only in where the off row is listed.
    let listed = |off_first: bool| {
        let mut rows = vec![bolus(0.0, 100.0, 1), obs(1.0, 2)];
        if off_first {
            rows.extend([off(3.0, 1), bolus(3.0, 50.0, 2)]);
        } else {
            rows.extend([bolus(3.0, 50.0, 2), off(3.0, 1)]);
        }
        rows.push(obs(4.0, 2));
        rows
    };
    let dosed = |off_first: bool| run(&m, &listed(off_first), &[3.0], mon(), depot_bolus).unwrap();
    let hold = |off_first: bool| {
        run(
            &m,
            &listed(off_first),
            &[3.0],
            mon(),
            |_: &ControllerCtx| vec![DoseAction::Hold],
        )
        .unwrap()
    };
    let (after_dose, before_dose) = (dosed(false), dosed(true));
    assert!(
        read(&after_dose) > 40.0,
        "row listed after a co-timed dose: the decision reads before it, got {}",
        read(&after_dose)
    );
    assert_eq!(
        c4(&after_dose),
        c4(&hold(false)),
        "and its dose is wiped by the row"
    );
    assert_eq!(
        read(&before_dose),
        0.0,
        "row listed before the co-timed dose: the decision reads after it"
    );
    assert!(
        c4(&before_dose) > 1.1 * c4(&hold(true)),
        "and its dose lands: {} vs held {}",
        c4(&before_dose),
        c4(&hold(true))
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

/// Time-varying covariate × reset × on/off rows (review r1 #2): a reset turns every
/// compartment back on on the per-event path too, in the driver and in
/// `adaptive_frozen_replay_tv`. `urine` (fed by elimination) is off from 3; the `EVID=3` row
/// at 5 turns it on, so the urine observation at 10 reads what flowed in since. Against
/// `predict()`, and the replay's `Ok`: a replay that keeps `urine` off past the reset reads 0
/// there and is refused.
#[test]
fn adaptive_cmt_toggle_reset_under_a_time_varying_covariate() {
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
    let m = parse_model_string(model).expect("parse");
    // (TIME, EVID, CMT, CRCL)
    let rows: [(f64, u8, i32, f64); 6] = [
        (1.0, 0, 2, 100.0),
        (3.0, 2, -3, 60.0),
        (4.0, 0, 2, 60.0),
        (5.0, 3, 1, 80.0),
        (7.0, 0, 2, 40.0),
        (10.0, 0, 3, 120.0),
    ];
    let to_csv = |doses: &[(f64, f64, f64)]| {
        let mut all: Vec<(f64, u8, f64, i32, f64)> = rows
            .iter()
            .map(|&(t, e, c, cr)| (t, e, 0.0, c, cr))
            .collect();
        // Each dose row carries the next record's CRCL (see the covariate test above).
        all.extend(doses.iter().map(|&(t, a, cr)| (t, 1, a, 1, cr)));
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
    let next_crcl = |t: f64| if t < 1.0 { 100.0 } else { 40.0 };
    let realized: Vec<(f64, f64, f64)> = res
        .ledger
        .iter()
        .map(|e| (e.time, e.amt, next_crcl(e.time)))
        .collect();
    let want: Vec<f64> = predict(
        &m,
        &read_pop(&to_csv(&realized), &["CRCL"]),
        &m.default_params,
    )
    .unwrap()
    .iter()
    .map(|p| p.pred)
    .collect();
    let got = ipreds(&res);
    assert!(
        got[3] > 1.0,
        "urine at 10 is on again after the reset: {}",
        got[3]
    );
    let worst = worst_rel(&got, &want);
    eprintln!("tv covariate × reset × toggle: worst rel {worst:e}");
    assert!(worst < ORACLE_TOL, "worst rel {worst:e}");
}

/// A dose the reset killed does not turn its compartment on when it would have arrived
/// (review r1 #3; `ResetGate::live`, as `event_driven_walk` skips a dead dose). A lagged dose
/// into `urine` is recorded at 1 (ALAG3 = 4, so it would arrive at 5); the `EVID=3` row at 2
/// kills it; `urine` is switched off at 3. Nothing turns it back on, so a monitor on `urine`
/// at 6 reads 0 although elimination keeps flowing at it. Turning it on at the dead arrival
/// would let urine fill from 5. `central` matches `predict()` throughout.
#[test]
fn adaptive_dead_lagged_dose_does_not_turn_its_compartment_on() {
    let m = parse_model_string(&GUT.replace("  V  = TVV\n", "  V  = TVV\n  ALAG3 = 4.0\n"))
        .expect("parse");
    let rows = vec![
        bolus(0.0, 100.0, 1),
        bolus(1.0, 10.0, 3),
        (2.0, 3, 0.0, 1, 0.0),
        bolus(2.5, 100.0, 1),
        off(3.0, 3),
        obs(4.0, 2),
        obs(8.0, 2),
    ];
    let res = run(
        &m,
        &rows,
        &[6.0],
        vec![MonitorSpec::new("U", 3, ObserveMode::Ipred)],
        |_: &ControllerCtx| vec![DoseAction::Hold],
    )
    .expect("runs and passes the frozen replay");
    assert_eq!(
        res.decisions[0].observed_signals[0].value, 0.0,
        "urine stays off: the dead arrival at 5 turns nothing on"
    );
    // A wider band than `ORACLE_TOL`, measured on macOS arm64: the driver breaks at the dead
    // arrival (5) and the decision (6), `event_driven_walk` at the records, and on this lagged,
    // saturating geometry that alone gives 1.35e-5 at t = 8 on the same subject with the off
    // row deleted (toggle-free, so not this feature). With the row: 2.0e-5. 1e-4 is 5× that;
    // a turned-on `urine` is caught by the monitor assertion above, not by this one.
    let want = static_preds(&m, &rows, &res);
    let got = ipreds(&res);
    let worst = worst_rel(&got, &want);
    assert!(worst < 1e-4, "worst rel {worst:e}: {got:?} vs {want:?}");
}

/// Oral 1-cpt `[odes]` model with CL on a time-varying `CRCL`, so a run takes the per-event
/// path and its verifier is `adaptive_frozen_replay_tv`.
const GUT_CRCL: &str = r#"
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

/// A held run on `pop` (one decision at 7, no dose), with the verifier on, and `predict()` on
/// the same subject. The decision sits on an observation's time, so the driver breaks where
/// `predict()` does and the two walk the same segments.
fn held_run_and_predict(
    m: &CompiledModel,
    pop: &Population,
) -> (Result<AdaptiveSimulationResult, EngineError>, Vec<f64>) {
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: vec![7.0],
        ..Default::default() // verify = true
    };
    let hold = |_: &ControllerCtx| vec![DoseAction::Hold];
    let res = simulate_adaptive(m, pop, &m.default_params, 1, || hold, &opts);
    let want = predict(m, pop, &m.default_params)
        .expect("predict the static twin")
        .iter()
        .map(|p| p.pred)
        .collect();
    (res, want)
}

/// An off row inside an `ADDL` train stops the rest of the train (#2732: an implied dose
/// takes its cutoff from its parent's record), in the driver **and** in
/// `adaptive_frozen_replay_tv` (#2811). The train is 100 into `depot` at 0, 4, 8 and 12;
/// `depot` goes off at 6, so the doses at 8 and 12 never land and nothing turns `depot` back
/// on. Before #2811 the replay's own bolus loop had no `ResetGate::live` test: it applied
/// both doses (129.0 at t = 9 against the driver's 103.3), and the verifier refused the run.
///
/// The replay's `Ok` is its side of the assertion; `predict()` (`event_driven_walk`, which
/// #2737 anchored against NONMEM) is the driver's. The twin without the off row is far away,
/// so the cut is live.
#[test]
fn adaptive_off_row_inside_an_addl_train_passes_the_replay() {
    let m = parse_model_string(GUT_CRCL).expect("parse");
    let data = |off_row: bool| {
        let mut s = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV,CRCL,ADDL,II\n");
        s.push_str("1,0,0,1,100,1,1,100,3,4\n1,1,0,0,0,2,0,100,0,0\n");
        if off_row {
            s.push_str("1,6,0,2,0,-1,1,60,0,0\n");
        }
        s.push_str("1,7,0,0,0,2,0,60,0,0\n1,9,0,0,0,2,0,40,0,0\n");
        s.push_str("1,13,0,0,0,2,0,120,0,0\n1,16,0,0,0,2,0,120,0,0\n");
        read_pop(&s, &["CRCL"])
    };
    let pop = data(true);
    assert!(pop.subjects[0].has_tv_covariates());
    assert_eq!(pop.subjects[0].doses.len(), 4, "the train is expanded");

    let (res, want) = held_run_and_predict(&m, &pop);
    let got = ipreds(&res.expect("runs and passes adaptive_frozen_replay_tv"));
    let worst = worst_rel(&got, &want);
    eprintln!("tv covariate × ADDL train cut by an off row: worst rel {worst:e}");
    assert!(
        worst < ORACLE_TOL,
        "worst rel {worst:e}: {got:?} vs {want:?}"
    );

    let (uncut, _) = held_run_and_predict(&m, &data(false));
    let uncut = ipreds(&uncut.expect("the uncut train runs"));
    // Observations at 9, 13 and 16 follow the doses the off row cancels.
    for j in 2..5 {
        assert!(
            uncut[j] > 1.2 * got[j],
            "obs {j}: the cancelled doses are missing ({} vs uncut {})",
            got[j],
            uncut[j]
        );
    }
}

/// An off row at exactly a base dose's time stops that dose (the engines' order,
/// `Dose < PkOnly`; `CoTimedOff::Stops`), in the driver and in `adaptive_frozen_replay_tv`
/// (#2811). Only a subject built in memory holds such a tie: the reader puts the row a ULP
/// after the dose, where the dose lands and the row then zeroes it, which gives the same
/// trajectory by another route and is asserted first. Before #2811 the replay applied the
/// tied dose and left `depot` on (77.0 at t = 4 against the driver's 51.5), and the verifier
/// refused the run. The twin without the off row is far away, so the row is live.
#[test]
fn adaptive_off_row_tied_with_a_base_dose_passes_the_replay() {
    let m = parse_model_string(GUT_CRCL).expect("parse");
    let data = |off_row: bool| {
        let mut s = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV,CRCL\n");
        s.push_str("1,0,0,1,100,1,1,100\n1,1,0,0,0,2,0,100\n1,3,0,1,100,1,1,60\n");
        if off_row {
            s.push_str("1,3,0,2,0,-1,1,60\n");
        }
        s.push_str("1,4,0,0,0,2,0,60\n1,7,0,0,0,2,0,40\n");
        read_pop(&s, &["CRCL"])
    };
    let mut pop = data(true);

    // As read: the row sits one ULP after the dose.
    let (res, want) = held_run_and_predict(&m, &pop);
    let as_read = ipreds(&res.expect("as read: runs and passes the replay"));
    let worst = worst_rel(&as_read, &want);
    assert!(worst < ORACLE_TOL, "as read: worst rel {worst:e}");

    // Tied: the row stops the dose at its arrival.
    let s = &mut pop.subjects[0];
    let row = s.cmt_toggles[0].pk_only;
    assert_eq!(s.pk_only_times[row], s.doses[1].time.next_up());
    s.pk_only_times[row] = s.doses[1].time;
    let (res, want) = held_run_and_predict(&m, &pop);
    let tied = ipreds(&res.expect("tied: runs and passes adaptive_frozen_replay_tv"));
    let worst = worst_rel(&tied, &want);
    eprintln!("tv covariate × off row tied with a base dose: worst rel {worst:e}");
    assert!(
        worst < ORACLE_TOL,
        "tied: worst rel {worst:e}: {tied:?} vs {want:?}"
    );

    let (kept, _) = held_run_and_predict(&m, &data(false));
    let kept = ipreds(&kept.expect("the twin without the off row runs"));
    // The dose at 3 has 1 h of absorption behind it at t = 4, and 4 h at t = 7.
    for j in 1..3 {
        assert!(
            kept[j] > 1.2 * tied[j],
            "obs {j}: the stopped dose is missing ({} vs kept {})",
            tied[j],
            kept[j]
        );
    }
}
