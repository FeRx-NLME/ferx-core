//! NONMEM cross-check for #1588: **an `SS=1` record resets co-timed doses by row order,
//! on every state engine.**
//!
//! NONMEM resets the system at an `SS=1` record in (time, row order): a dose row that
//! comes *before* a co-timed `SS=1` row is wiped, one that comes *after* it superposes,
//! and a lagged dose still pending at the record is cancelled. Before #1588 the ODE
//! static walker (and the adaptive driver sharing its two passes) applied every co-timed
//! bolus *after* the reset whatever its row, and once the `SS=1` dose carries a lag every
//! state engine did the same — the record-time seed is its own event and sorts before
//! every co-timed arrival (`DoseRecord < Dose`). The fix gates each arrival on
//! `ResetGate::live` (then `SsResetGate`), #1576's record-keyed predicate.
//!
//! The reference is NONMEM 7.6.0 (`nm3`, `anchor` build), `ADVAN2 TRANS2`, `MAXEVAL=0`,
//! `FORMAT=s1PE23.16`, `CL = 2, V = 20, KA = 0.15`, `II = 12`, `AMT = 100`:
//! `nonmem_anchor/ss_reset_tie.{csv,ctl}` (`ALAG1 = 0`) and `ss_reset_tie_lag.ctl`
//! (`ALAG1 = 2`, depot doses only). On the wiped cells NONMEM equals the surviving `SS=1`
//! regimen's closed form (an independent Python pulse-train sum) to 8.8e-16. IDs:
//!
//! | ID | rows at t = 10 (row order) | NONMEM |
//! |---|---|---|
//! | 1 | bolus central, then `SS=1` depot | wiped |
//! | 2 | `SS=1`, then bolus central | superposes |
//! | 3 | bolus depot, then `SS=1` | wiped (lag 2: the pending lagged bolus is cancelled) |
//! | 4 | `SS=1`, then bolus depot | superposes |
//! | 5 | `SS=1`; bolus central at 11 | superposes |
//! | 6 | `SS=1` alone | — |
//! | 7 | depot bolus at 0, then as ID 1 | wiped |
//! | 8 | depot bolus at 0, then as ID 2 | superposes |
//! | 9 | `SS=1` 100 at 10; `SS=1` 200 at 11 | the second resets the first |
//! | 10 | depot bolus at 9, `SS=1` at 10 | lag 2: the bolus (arrives 11) is cancelled |
//! | 11 | `SS=1` 100, then `SS=1` 200, co-timed | the second wins |
//! | 12 | depot bolus at 8, `SS=1` at 10 | lag 2: the bolus arrives **exactly at** the record, and is wiped |
//! | 13 | `SS=1` 100 central (unlagged), then `SS=1` 200 depot, co-timed | the second wins |
//!
//! ID 12 pins the edge of "reached" (`ResetGate` adds `EVENT_MATCH_TOL` to the break):
//! an arrival that lands on the record's own time. ID 13 is the one shape where only the
//! arrival re-equilibration gate decides the answer at lag 2: the wiped `SS=1` dose is
//! unlagged, the live one is seeded at its record, so nothing later overwrites a wrong
//! re-equilibration (PR #1601 review finding 1).
//! **IDs 2, 5, 8 at lag 2 are the #1275 rows**: a dose lands between the lagged `SS=1`
//! record and its arrival (t = 12). The dense ODE walks (static, with-states, adaptive) used
//! to re-equilibrate at that arrival and wipe it (−51 … −54 %); they now flow the record's
//! seed to the arrival (`dosing::ss_equilibrates_at_arrival`) and are asserted on every ID.
//!
//! **Analytic at lag 2 is asserted on the depot-only IDs** (3, 4, 6, 9, 10, 11, 12): the
//! analytic `lagtime=` slot lags every dose record, so it cannot express NONMEM's
//! depot-only `ALAG1` on a central dose.
//!
//! The event-driven variant of each engine is reached by prefixing every subject with an
//! `EVID=3` row at t = 0, generated here from the committed csv (a reset before anything
//! happens changes nothing, and routes the subject to the event-driven walk).
//!
//! The oracle is the committed NONMEM table itself, read at run time. Evaluation only —
//! not gated.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::pk::compute_predictions_with_states;
use ferx_core::sens::ode_provider::ode_subject_sensitivities;
use ferx_core::sim::adaptive::{ControllerCtx, DoseAction};
use ferx_core::{
    predict, read_nonmem_csv, simulate_adaptive, AdaptiveSimulateOptions, CompiledModel, Population,
};

const DATA: &str = "nonmem_anchor/ss_reset_tie.csv";
const TABLE_LAG0: &str = "nonmem_anchor/results/ss_reset_tie.sdtab";
const TABLE_LAG2: &str = "nonmem_anchor/results/ss_reset_tie_lag.sdtab";

/// IDs with a central (CMT 2) dose, which the analytic single `lagtime` slot would lag.
const CENTRAL_DOSE: &[&str] = &["1", "2", "5", "7", "8", "13"];

fn ode_depot(lag: f64) -> CompiledModel {
    let src = format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  theta TVLAG({lag}, 0.0, 10.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  ALAG1 = TVLAG
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA*depot
  d/dt(central) = KA*depot - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-11
"#
    );
    parse_full_model(&src).expect("ODE model parses").model
}

fn analytic(lag: f64) -> CompiledModel {
    let src = format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  theta TVLAG({lag}, 0.0, 10.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  LAG = TVLAG
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAG)
[error_model]
  DV ~ proportional(PROP_ERR)
"#
    );
    parse_full_model(&src).expect("analytic model parses").model
}

fn load(path: &std::path::Path) -> Population {
    read_nonmem_csv(path, None, None).expect("dataset loads")
}

/// The committed dataset, as is: every subject takes the static walk.
fn static_pop() -> Population {
    load(std::path::Path::new(DATA))
}

/// The committed dataset with an `EVID=3` row at t = 0 ahead of each subject, which routes
/// it to the event-driven walk and changes nothing else. Written to a per-call temp file.
fn event_driven_pop() -> Population {
    let text = std::fs::read_to_string(DATA).expect("csv");
    let mut out = String::new();
    let mut last_id = String::new();
    for (i, line) in text.lines().enumerate() {
        if i > 0 {
            let id = line.split(',').next().expect("ID column").to_string();
            if id != last_id {
                out.push_str(&format!("{id},0,0,3,0,1,1,0,0,0\n"));
                last_id = id;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    // Unique per call, not just per process: the tests run on concurrent threads, and a
    // shared path let one test's `remove_file` race another's read.
    static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "ferx_1588_ss_reset_tie_evid3_{}_{}.csv",
        std::process::id(),
        CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::write(&path, out).expect("write temp csv");
    let pop = load(&path);
    let _ = std::fs::remove_file(&path);
    assert!(
        pop.subjects.iter().all(|s| s.reset_times == vec![0.0]),
        "every subject must carry the leading reset, or it is not on the event-driven walk"
    );
    pop
}

/// `(ID, TIME, PRED)` for every observation record of a NONMEM `$TABLE ... NOAPPEND` file
/// whose columns are `ID TIME EVID PRED`.
fn nonmem_obs(path: &str) -> Vec<(String, f64, f64)> {
    let text = std::fs::read_to_string(path).expect("NONMEM table");
    text.lines()
        .skip(2)
        .map(|l| {
            let c: Vec<f64> = l
                .split_whitespace()
                .map(|x| x.parse().expect("number"))
                .collect();
            (c[0], c[1], c[2], c[3])
        })
        .filter(|&(_, _, evid, _)| evid == 0.0)
        .map(|(id, t, _, p)| (format!("{}", id as i64), t, p))
        .collect()
}

/// Assert the engine's `(ID, TIME, value)` rows reproduce `table` to `bound` (relative)
/// outside `skip_ids`, listing every miss. Each value is asserted finite before it is
/// folded (`f64::max` discards `NaN`), and a skip list naming an ID the data lacks fails.
fn assert_matches(label: &str, got: &[(String, f64, f64)], table: &str, bound: f64, skip: &[&str]) {
    let want = nonmem_obs(table);
    assert_eq!(got.len(), want.len(), "{label}: observation count");
    let mut worst = 0.0_f64;
    let mut bad = Vec::new();
    let mut skipped = std::collections::BTreeSet::new();
    for ((gid, gt, g), (id, t, w)) in got.iter().zip(&want) {
        assert!(
            gid == id && (gt - t).abs() < 1e-12,
            "{label}: row order ({gid}, {gt}) vs NONMEM ({id}, {t})"
        );
        if skip.contains(&id.as_str()) {
            skipped.insert(id.clone());
            continue;
        }
        assert!(g.is_finite(), "{label}: ID {id} t={t} non-finite ({g})");
        let rel = (g - w).abs() / w.abs();
        if !(rel < bound) {
            bad.push(format!(
                "ID {id} t={t}: ferx {g} vs NONMEM {w} (rel {rel:+.3e})"
            ));
        }
        worst = worst.max(rel);
    }
    assert_eq!(
        skipped.len(),
        skip.len(),
        "{label}: a skipped ID is not in the data"
    );
    println!(
        "#1588 {label}: worst rel vs NONMEM {worst:.3e} (bound {bound:.0e}, skipped {skip:?})"
    );
    assert!(
        bad.is_empty(),
        "{label}: {} rows off:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

fn via_predict(model: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    predict(model, pop, &model.default_params)
        .expect("predict")
        .into_iter()
        .map(|p| (p.id, p.time, p.pred))
        .collect()
}

/// `compute_predictions_with_states`: `ode_predictions_with_states` on a static subject,
/// `ode_predictions_event_driven_with_states` on a reset one.
fn via_with_states(model: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let eta = vec![0.0; model.n_eta];
    let mut out = Vec::new();
    for s in &pop.subjects {
        let (ipred, _) =
            compute_predictions_with_states(model, s, &model.default_params.theta, &eta);
        for (t, v) in s.obs_times.iter().zip(ipred) {
            out.push((s.id.clone(), *t, v));
        }
    }
    out
}

/// The dual ODE walk's value (`integrate_tvcov_g`). Asserts every subject stays on the
/// analytic provider: an FD-routed subject would make this leg FD-of-f64 and blind to the
/// dual walk.
fn via_dual(model: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let eta = vec![0.0; model.n_eta];
    let mut out = Vec::new();
    for s in &pop.subjects {
        let sens = ode_subject_sensitivities(model, s, &model.default_params.theta, &eta)
            .unwrap_or_else(|| panic!("ID {}: routed to FD, the dual walk is not exercised", s.id));
        for (t, o) in s.obs_times.iter().zip(sens.obs) {
            out.push((s.id.clone(), *t, o.f));
        }
    }
    out
}

/// `simulate_adaptive` with a controller that never doses: the base regimen through the
/// adaptive driver, frozen-replay verifier on. One decision after the tie.
/// It runs at η = 0 exactly, as the NONMEM `PRED` it is judged against does: `omega ~ 0.0`
/// is PD-regularised to Ω = 1e-8 (`OmegaMatrix`), so the default parameters would draw η
/// with sd 1e-4 and put the driver ~3e-4 off the table (#1603). η threading is pinned
/// elsewhere (`adaptive_iov_matches_predict_iov_with_reconstructed_kappa`).
fn via_adaptive(model: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let mut opts = AdaptiveSimulateOptions::default();
    opts.seed = Some(1);
    opts.decision_times = vec![25.0];
    let mut params = model.default_params.clone();
    params.omega.chol.fill(0.0);
    params.omega.matrix.fill(0.0);
    let res = simulate_adaptive(
        model,
        pop,
        &params,
        1,
        || |_: &ControllerCtx| -> Vec<DoseAction> { Vec::new() },
        &opts,
    )
    .expect("adaptive sim runs and passes the frozen-replay verifier");
    assert!(res.ledger.is_empty(), "the no-op controller dosed");
    res.trajectories
        .into_iter()
        .map(|r| (r.id, r.time, r.ipred))
        .collect()
}

// ODE f64 and dual: measured worst 2.2e-12 (probe at fc1d1404 on untouched IDs; see the
// printed worst below). Analytic: 1.6e-15. Adaptive at η = 0: 1.95e-12 (lag 0) / 2.20e-12
// (lag 2) — the 3.1e-4 it showed before was its η draw from the regularised `omega ~ 0.0`
// (#1603), not the driver. The smallest defect it guards is +9.8 %.
const ODE_BOUND: f64 = 1e-10;
const ANALYTIC_BOUND: f64 = 1e-13;

// ---- ODE static walker (`ode_predictions`): S1 + S2 ----

#[test]
fn ode_static_walker_resets_a_co_timed_dose_by_row_order_lag0() {
    let got = via_predict(&ode_depot(0.0), &static_pop());
    assert_matches("ODE static, lag 0", &got, TABLE_LAG0, ODE_BOUND, &[]);
}

#[test]
fn ode_static_walker_resets_a_co_timed_dose_by_row_order_lag2() {
    let got = via_predict(&ode_depot(2.0), &static_pop());
    assert_matches("ODE static, lag 2", &got, TABLE_LAG2, ODE_BOUND, &[]);
}

// ---- ODE event-driven walker (`ode_predictions_event_driven`): S5 ----

#[test]
fn ode_event_driven_walker_resets_a_co_timed_dose_by_row_order_lag0() {
    let got = via_predict(&ode_depot(0.0), &event_driven_pop());
    assert_matches("ODE event-driven, lag 0", &got, TABLE_LAG0, ODE_BOUND, &[]);
}

#[test]
fn ode_event_driven_walker_resets_a_co_timed_dose_by_row_order_lag2() {
    let got = via_predict(&ode_depot(2.0), &event_driven_pop());
    assert_matches("ODE event-driven, lag 2", &got, TABLE_LAG2, ODE_BOUND, &[]);
}

// ---- ODE single-pass states (`ode_predictions_with_states`): S3; event-driven: S5 ----

#[test]
fn ode_with_states_resets_a_co_timed_dose_by_row_order() {
    for (lag, table) in [(0.0, TABLE_LAG0), (2.0, TABLE_LAG2)] {
        let m = ode_depot(lag);
        let got = via_with_states(&m, &static_pop());
        assert_matches(
            &format!("ODE with_states, lag {lag}"),
            &got,
            table,
            ODE_BOUND,
            &[],
        );
        let got = via_with_states(&m, &event_driven_pop());
        assert_matches(
            &format!("ODE event-driven with_states, lag {lag}"),
            &got,
            table,
            ODE_BOUND,
            &[],
        );
    }
}

// ---- adaptive driver (shares S1 + S2 over its `shadow` list) ----

#[test]
fn adaptive_driver_resets_a_co_timed_dose_by_row_order() {
    for (lag, table) in [(0.0, TABLE_LAG0), (2.0, TABLE_LAG2)] {
        let got = via_adaptive(&ode_depot(lag), &static_pop());
        assert_matches(&format!("adaptive, lag {lag}"), &got, table, ODE_BOUND, &[]);
    }
}

// ---- dual ODE walk (`integrate_tvcov_g`): S6 ----

#[test]
fn dual_ode_walk_value_resets_a_co_timed_dose_by_row_order() {
    for (lag, table) in [(0.0, TABLE_LAG0), (2.0, TABLE_LAG2)] {
        let m = ode_depot(lag);
        let got = via_dual(&m, &static_pop());
        assert_matches(
            &format!("dual ODE (static subjects), lag {lag}"),
            &got,
            table,
            ODE_BOUND,
            &[],
        );
        let got = via_dual(&m, &event_driven_pop());
        assert_matches(
            &format!("dual ODE (EVID=3 subjects), lag {lag}"),
            &got,
            table,
            ODE_BOUND,
            &[],
        );
    }
}

// ---- analytic: superposition (#1576's cutoff, the control) and event-driven (S7) ----

#[test]
fn analytic_superposition_resets_a_co_timed_dose_by_row_order() {
    let got = via_predict(&analytic(0.0), &static_pop());
    assert_matches(
        "analytic superposition, lag 0",
        &got,
        TABLE_LAG0,
        ANALYTIC_BOUND,
        &[],
    );
    let got = via_predict(&analytic(2.0), &static_pop());
    assert_matches(
        "analytic superposition, lag 2",
        &got,
        TABLE_LAG2,
        ANALYTIC_BOUND,
        CENTRAL_DOSE,
    );
}

#[test]
fn analytic_event_driven_walk_resets_a_co_timed_dose_by_row_order() {
    let got = via_predict(&analytic(0.0), &event_driven_pop());
    assert_matches(
        "analytic event-driven, lag 0",
        &got,
        TABLE_LAG0,
        ANALYTIC_BOUND,
        &[],
    );
    let got = via_predict(&analytic(2.0), &event_driven_pop());
    assert_matches(
        "analytic event-driven, lag 2",
        &got,
        TABLE_LAG2,
        ANALYTIC_BOUND,
        CENTRAL_DOSE,
    );
}
