//! NONMEM cross-check for #1587: **an EVID=3/4 reset cancels a dose whose record precedes
//! it, however late its lagged arrival — on every engine.**
//!
//! NONMEM resets the system at the reset **record** and cancels every dose recorded before
//! it, including a lagged bolus whose arrival (`record + ALAG`) and a lagged infusion whose
//! window lie after the reset. Before #1587 every ferx engine keyed its reset test on the
//! arrival and kept such a dose (+270 % on the issue's case). The fix keys it on the record
//! (`dosing::evid_reset_live`, read through `ResetGate` at every arrival site).
//!
//! The reference is NONMEM 7.6.0 (`nm3`, `anchor` build), `MAXEVAL=0`, all `$THETA` `FIX`,
//! `FORMAT=s1PE23.16`, `CL = 2, V = 20, KA = 0.15`, `AMT = 100` into the depot, `ALAG1`
//! from the `LAGC` data column: `nonmem_anchor/evid_reset_lag.{csv,ctl}` (`ADVAN2 TRANS2`).
//! `evid_reset_lag_advan13.ctl` (`ADVAN13 TOL=12`) agrees with it to 2.8e-11 on every ID but
//! 23 — a lag ≥ `II` steady state with no reset, where NONMEM's `ADVAN13` SS routine departs
//! from its own `ADVAN2` by up to 0.20 (#1121's territory; the cancelled twin, ID 22, is 0 on
//! both) — and the
//! surviving cells (IDs 1, 3, 5, 6, 16) equal an independent Python Bateman sum, outside
//! both engines, to 4.2e-15. IDs (lag 2 unless noted):
//!
//! | ID | rows | NONMEM |
//! |---|---|---|
//! | 1 | dose 9, EVID=3 10, dose 10.001 | only the 10.001 dose (the issue's case) |
//! | 2 / 3 | dose 9, EVID=3 10 / EVID=3 8.5, dose 9 | **0** / live — same arrival, record straddles |
//! | 4 | dose 9, EVID=3 12 (arrival before the reset) | live, then 0 |
//! | 5 / 6 | dose 9, EVID=4 10 / EVID=4 8.5, dose 9 | first dose cancelled / both live |
//! | 7 / 8 | dose row then EVID=3 row, both at 10 / reversed | wiped / live |
//! | 9 / 10 | as 7 / 8 at lag 0 | wiped / live |
//! | 11 | 2 h infusion at 9, EVID=3 10 | 0 — but not a discriminating cell: its unlagged window ends at 11, so the reader shifts the reset to 12 (`RESET_SEGMENT_GAP`), past the lagged arrival at 11, and the old arrival rule gave 0 too. IDs 17 and 18 carry this cell |
//! | 12 | unlagged 4 h infusion at 9, EVID=3 10 | 0 after the reset |
//! | 13 / 14 | lag 11, `SS=1` at 10 (II 12), EVID=3 15 / no reset | **0** / live |
//! | 15 | dose row then EVID=4 row, both at 10 | only the EVID=4 dose |
//! | 16 | as 1, second dose IV into central | only the IV dose |
//! | 17 | 0.25 h infusion at 9, EVID=3 10 (no reader shift) | **0** |
//! | 18 / 19 | lag 5, 2 h infusion at 9, EVID=3 10 / no reset | **0** / live |
//! | 20 / 21 | dose 8 arriving exactly at an EVID=3 / EVID=4 at 10 | **0** / only the EVID=4 dose |
//! | 22 / 23 | lag 13 ≥ II 12, `SS=1` at 10, EVID=3 15 / no reset | **0** / live (re-equilibrates at its arrival) |
//!
//! The co-timed rows (7–10, 15) reach the engines through the reader's reset shift
//! (`RESET_SEGMENT_GAP`): a dose row *before* a co-timed reset row ends up recorded before
//! the shifted reset, so the record rule sees the row order.
//!
//! The analytic engines skip ID 16 (its single `lagtime` slot would lag the central dose).
//! Evaluation only — not gated.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::pk::compute_predictions_with_states;
use ferx_core::sens::ode_provider::ode_subject_sensitivities;
use ferx_core::sens::provider::subject_sensitivities;
use ferx_core::sim::adaptive::{ControllerCtx, DoseAction};
use ferx_core::{
    predict, read_nonmem_csv, simulate_adaptive, AdaptiveSimulateOptions, CompiledModel, Population,
};

const DATA: &str = "nonmem_anchor/evid_reset_lag.csv";
const TABLE: &str = "nonmem_anchor/results/evid_reset_lag.sdtab";
const TABLE_ADVAN13: &str = "nonmem_anchor/results/evid_reset_lag_advan13.sdtab";

/// IDs judged against the `ADVAN13` table instead of `ADVAN2`. ID 23 is an `SS=1` dose whose
/// lag is at least `II`, with no reset (the live control of ID 22): there NONMEM's two ADVANs
/// disagree by up to 0.20, and ferx's `ALAG >= II` convention is `ADVAN13`'s (#1604, out of
/// scope here). ID 22, the cancelled twin under test, reads 0 in both tables.
const ADVAN13_REFERENCE: &[&str] = &["23"];

/// The analytic `lagtime=` slot lags every dose record, so it cannot express NONMEM's
/// depot-only `ALAG1` on ID 16's central dose.
const CENTRAL_DOSE: &[&str] = &["16"];

/// The analytic dual's skip list: `CENTRAL_DOSE`, plus ID 23 — an `SS=1` dose with
/// `ALAG > II` on the static closed-form superposition, which wraps the pre-arrival tail where
/// the value path clamps it (#1353, open, out of scope). It reads `ADVAN2`'s 4.5306 there
/// while ferx's own value path reads 4.4741. ID 22, its cancelled twin, is asserted.
const ANALYTIC_DUAL_SKIP: &[&str] = &["16", "23"];

/// Pairs that differ in one input and straddle the gate: (with the dose cancelled, with it
/// live). The live member carries one more dose than the cancelled one.
const STRADDLES: &[(&str, &str)] = &[
    ("2", "3"),   // reset after vs before the record, same arrival
    ("5", "6"),   // the same for EVID=4
    ("7", "8"),   // dose row before vs after a co-timed EVID=3 row
    ("13", "14"), // lagged SS dose with vs without the reset
    ("18", "19"), // lagged infusion with vs without the reset
    ("22", "23"), // SS dose with lag ≥ II with vs without the reset
];

const MODEL_TAIL: &str = r#"
[error_model]
  DV ~ additive(ADD_ERR)
"#;

fn ode() -> CompiledModel {
    let src = format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  omega ETA_CL ~ 0.0
  sigma ADD_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  ALAG1 = LAGC
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA*depot
  d/dt(central) = KA*depot - CL/V*central
[scaling]
  y = central / V
{MODEL_TAIL}
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-11
"#
    );
    parse_full_model(&src).expect("ODE model parses").model
}

fn analytic() -> CompiledModel {
    let src = format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  omega ETA_CL ~ 0.0
  sigma ADD_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  LAG = LAGC
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAG)
{MODEL_TAIL}
"#
    );
    parse_full_model(&src).expect("analytic model parses").model
}

fn pop() -> Population {
    read_nonmem_csv(std::path::Path::new(DATA), None, None).expect("dataset loads")
}

/// `(ID, TIME, PRED)` for every observation of the committed NONMEM table.
/// The reference: the `ADVAN2` table, with the `ADVAN13` value on `ADVAN13_REFERENCE` IDs.
fn nonmem_obs() -> Vec<(String, f64, f64)> {
    let (a2, a13) = (read_table(TABLE), read_table(TABLE_ADVAN13));
    assert_eq!(
        a2.len(),
        a13.len(),
        "the two NONMEM tables must cover the same rows"
    );
    a2.into_iter()
        .zip(a13)
        .map(|(r2, r13)| {
            assert!(r2.0 == r13.0 && r2.1 == r13.1, "table rows out of step");
            if ADVAN13_REFERENCE.contains(&r2.0.as_str()) {
                r13
            } else {
                r2
            }
        })
        .collect()
}

fn read_table(path: &str) -> Vec<(String, f64, f64)> {
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

/// Assert the engine's `(ID, TIME, value)` rows reproduce NONMEM within `abs + rel·|want|`
/// (the cancelled cells are exactly `0`, where a relative error alone is undefined), outside
/// `skip`. Every value is asserted finite before it is folded (`f64::max` discards `NaN`);
/// every miss is listed; a skipped ID absent from the data fails.
fn assert_matches(label: &str, got: &[(String, f64, f64)], bound: f64, skip: &[&str]) {
    let want = nonmem_obs();
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
        let err = (g - w).abs() / (1.0 + w.abs());
        // An `ADVAN13` reference is itself an ODE solve (`TOL=12`, measured 2.4e-12 from the
        // closed form here), so it cannot judge the analytic engines to `ANALYTIC_BOUND`.
        let bound = if ADVAN13_REFERENCE.contains(&id.as_str()) {
            bound.max(ODE_BOUND)
        } else {
            bound
        };
        if !(err < bound) {
            bad.push(format!(
                "ID {id} t={t}: ferx {g} vs NONMEM {w} (err {err:.3e})"
            ));
        }
        worst = worst.max(err);
    }
    assert_eq!(
        skipped.len(),
        skip.len(),
        "{label}: a skipped ID is not in the data"
    );
    println!(
        "#1587 {label}: worst err vs NONMEM {worst:.3e} (bound {bound:.0e}, skipped {skip:?})"
    );
    assert!(
        bad.is_empty(),
        "{label}: {} rows off (a dose recorded before an EVID=3/4 reset must be cancelled):\n{}",
        bad.len(),
        bad.join("\n")
    );
}

fn via_predict(model: &CompiledModel) -> Vec<(String, f64, f64)> {
    predict(model, &pop(), &model.default_params)
        .expect("predict")
        .into_iter()
        .map(|p| (p.id, p.time, p.pred))
        .collect()
}

/// Engine-side values carry the reader's shifted clock (`RESET_SEGMENT_GAP`), so they are
/// paired with the raw times `predict` reports, row for row.
fn raw_times() -> Vec<f64> {
    via_predict(&ode()).into_iter().map(|r| r.1).collect()
}

fn with_raw(rows: Vec<(String, f64)>) -> Vec<(String, f64, f64)> {
    let raw = raw_times();
    assert_eq!(rows.len(), raw.len(), "row count vs predict");
    rows.into_iter()
        .zip(raw)
        .map(|((id, v), t)| (id, t, v))
        .collect()
}

/// `compute_predictions_with_states`: `ode_predictions_event_driven_with_states` on a reset
/// subject, `ode_predictions_with_states` on the two without one (IDs 14, 19).
fn via_with_states(model: &CompiledModel) -> Vec<(String, f64, f64)> {
    let eta = vec![0.0; model.n_eta];
    let mut out = Vec::new();
    for s in &pop().subjects {
        let (ipred, _) =
            compute_predictions_with_states(model, s, &model.default_params.theta, &eta);
        out.extend(ipred.into_iter().map(|v| (s.id.clone(), v)));
    }
    with_raw(out)
}

/// A dual walk's value (`o.f`). Asserts every subject stays on the analytic provider: an
/// FD-routed subject would make this leg FD-of-f64 and blind to the dual walk.
fn via_dual(
    model: &CompiledModel,
    sens: impl Fn(&ferx_core::types::Subject) -> Option<ferx_core::sens::provider::SubjectSens>,
) -> Vec<(String, f64, f64)> {
    let mut out = Vec::new();
    for s in &pop().subjects {
        let r = sens(s)
            .unwrap_or_else(|| panic!("ID {}: routed to FD, the dual walk is not exercised", s.id));
        out.extend(r.obs.into_iter().map(|o| (s.id.clone(), o.f)));
    }
    let _ = model;
    with_raw(out)
}

/// `simulate_adaptive` with a controller that never doses: the base regimen through the
/// adaptive driver, frozen-replay verifier on.
/// It runs at η = 0 exactly, as the NONMEM `PRED` it is judged against does: `omega ~ 0.0`
/// is PD-regularised to Ω = 1e-8 (`OmegaMatrix`), so the default parameters would draw η
/// with sd 1e-4 and put the driver ~3e-4 off the table (#1603). η threading is pinned
/// elsewhere (`adaptive_iov_matches_predict_iov_with_reconstructed_kappa`).
fn via_adaptive(model: &CompiledModel) -> Vec<(String, f64, f64)> {
    let mut opts = AdaptiveSimulateOptions::default();
    opts.seed = Some(1);
    opts.decision_times = vec![25.0];
    let mut params = model.default_params.clone();
    params.omega.chol.fill(0.0);
    params.omega.matrix.fill(0.0);
    let res = simulate_adaptive(
        model,
        &pop(),
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

// Measured at the fix (err = |ferx − NONMEM| / (1 + |NONMEM|)): ODE predict / with-states /
// dual 3.7e-12 (27× under `ODE_BOUND`); analytic value and dual 2.1e-15 against `ADVAN2`
// (48×), and 2.4e-12 on ID 23 against `ADVAN13`, which is judged to `ODE_BOUND`; adaptive
// at η = 0 3.7e-12, equal to ODE predict to four figures (the 2.7e-4 it showed before was its
// η draw from the regularised `omega ~ 0.0`, #1603). The smallest defect guarded is ID 13's
// kept pulse, 0.66 absolute.
const ODE_BOUND: f64 = 1e-10;
const ANALYTIC_BOUND: f64 = 1e-13;

/// The fixture straddles the gate: each pair differs in one input, and NONMEM reads the
/// dose cancelled in the first member and live in the second, so the second reads higher by
/// a whole dose's contribution at every observation from t = 16 (every lagged window has
/// opened by then). An engine matching the table on both members therefore sees the gate
/// from both sides.
#[test]
fn the_nonmem_table_straddles_the_record_gate() {
    let want = nonmem_obs();
    let late = |id: &str| -> Vec<f64> {
        want.iter()
            .filter(|(i, t, _)| i == id && *t >= 16.0)
            .map(|r| r.2)
            .collect()
    };
    for &(dead, live) in STRADDLES {
        let (d, l) = (late(dead), late(live));
        assert!(!d.is_empty() && d.len() == l.len(), "IDs {dead}/{live}");
        assert!(
            d.iter().zip(&l).all(|(&a, &b)| b - a > 0.1),
            "ID {live} must carry the dose ID {dead} cancels: {l:?} vs {d:?}"
        );
    }
}

#[test]
fn ode_predict_cancels_a_dose_recorded_before_the_reset() {
    assert_matches("ODE predict", &via_predict(&ode()), ODE_BOUND, &[]);
}

#[test]
fn ode_with_states_cancels_a_dose_recorded_before_the_reset() {
    assert_matches("ODE with-states", &via_with_states(&ode()), ODE_BOUND, &[]);
}

#[test]
fn ode_dual_walk_value_cancels_a_dose_recorded_before_the_reset() {
    let m = ode();
    let eta = vec![0.0; m.n_eta];
    let got = via_dual(&m, |s| {
        ode_subject_sensitivities(&m, s, &m.default_params.theta, &eta)
    });
    assert_matches("ODE dual", &got, ODE_BOUND, &[]);
}

#[test]
fn adaptive_driver_cancels_a_dose_recorded_before_the_reset() {
    assert_matches("adaptive", &via_adaptive(&ode()), ODE_BOUND, &[]);
}

#[test]
fn analytic_event_walk_cancels_a_dose_recorded_before_the_reset() {
    assert_matches(
        "analytic",
        &via_predict(&analytic()),
        ANALYTIC_BOUND,
        CENTRAL_DOSE,
    );
}

#[test]
fn analytic_dual_cancels_a_dose_recorded_before_the_reset() {
    let m = analytic();
    let eta = vec![0.0; m.n_eta];
    let got = via_dual(&m, |s| {
        subject_sensitivities(&m, s, &m.default_params.theta, &eta)
    });
    assert_matches("analytic dual", &got, ANALYTIC_BOUND, ANALYTIC_DUAL_SKIP);
}
