//! NONMEM cross-check for #1586: **an `SS=1` record stops every infusion and zero-order
//! window whose dose record precedes it — on every engine.**
//!
//! NONMEM resets the system at an `SS=1` record: the state becomes that record's own steady
//! state, and every input whose record precedes it in (time, row order) is gone — a window
//! running across the record, an infusion row co-timed in an earlier row, a lagged window
//! still pending, the #1121 residual of an earlier `SS=1` infusion, a zero-order window.
//! Before #1586 the state engines kept every such window running (+14 % to +175 %), and on
//! analytic models the dual walk did too while the value (superposition, #1589) did not, so
//! FOCE/FOCEI differentiated a different dosing history than they predicted. The fix reads
//! window membership through the one `ResetGate` every arrival already reads.
//!
//! The reference is NONMEM 7.6.0 (`nm3`, `anchor` build), `MAXEVAL=0`, all `$THETA` `FIX`,
//! `FORMAT=s1PE23.16`: `nonmem_anchor/ss_infusion_stop.{csv,ctl}` (`ADVAN2 TRANS2`), one
//! compartment with first-order absorption, `CL = 2, V = 20, KA = 0.15`, the `SS=1` record
//! q12h `AMT = 100` into the depot at t = 10 unless noted, `ALAG2` from the `LAG2` column.
//! `ADVAN2` equals a pure-Python closed form of the rule (exact two-state propagation,
//! 800-cycle `SS` run-in, outside both engines) to 4.4e-15 on all 36 IDs, and
//! `ss_infusion_stop_advan13.ctl` (`ADVAN13 TOL=12`) agrees with it to 3.2e-11.
//!
//! | ID | rows | NONMEM |
//! |---|---|---|
//! | 1 | `SS=1` alone | control |
//! | 2 / 3 | infusion 0–20 into the depot / central, running across | stopped at 10 |
//! | 4 / 5 | central infusion row before / after a co-timed `SS=1` row | **never runs** / runs |
//! | 6 | depot infusion row before a co-timed `SS=1` row | never runs |
//! | 7 / 8 | infusion ends at 8 / starts at 12 | controls |
//! | 9 | `SS=1` at 10, central infusion 15–35, `SS=1` at 22 | stopped at 22 |
//! | 10 | central infusion 0–20, `SS=1` infusion (`RATE = 20`) into central at 10 | earlier stopped, own runs |
//! | 12 | `LAG2 = 4`, central infusion recorded at 8 (window 12–22) | **cancelled** (record keyed) |
//! | 13 | `LAG2 = 4`, central infusion recorded at 4 (window 8–18) | stopped at 10 |
//! | 14 / 15 | `LAG2 = 4`, lagged `SS=1` central infusion at 10, `SS=1` depot bolus at 11 / none | #1121 residual stopped at 11 / runs |
//! | 20 | `zero_order` window 5–13 into the depot, `SS=1` bolus into central at 10 | stopped |
//! | 21 | zero-order window 0–8 | control |
//! | 22 / 23 | zero-order row before / after a co-timed `SS=1` row | **never runs** / runs |
//!
//! IDs + 100 are the same cells behind a leading EVID=3 at t = 0, which routes the ODE
//! `predict` and with-states legs to the event-driven walkers; IDs < 100 run the dense ones.
//! The zero-order IDs (20–23) are NONMEM `RATE = -2` with `D1 = 8` into the depot, and ferx
//! `zero_order(dur = 8)` from `nonmem_anchor/ss_infusion_stop_zo.csv`. Evaluation only — not
//! gated.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::pk::compute_predictions_with_states;
use ferx_core::sens::ode_provider::ode_subject_sensitivities;
use ferx_core::sens::provider::subject_sensitivities;
use ferx_core::sim::adaptive::{ControllerCtx, DoseAction};
use ferx_core::types::Subject;
use ferx_core::{
    predict, read_nonmem_csv, simulate_adaptive, AdaptiveSimulateOptions, CompiledModel, Population,
};

const DATA: &str = "nonmem_anchor/ss_infusion_stop.csv";
const DATA_ZO: &str = "nonmem_anchor/ss_infusion_stop_zo.csv";
const TABLE: &str = "nonmem_anchor/results/ss_infusion_stop.sdtab";

/// The infusion cells at lag 0: every engine, the analytic ones included.
const A0: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// The lagged cells: `ALAG2` on central only, which the analytic single `lagtime` slot cannot
/// express (it would lag the depot `SS=1` dose too).
const A4: &[u32] = &[12, 13, 14, 15];
/// The zero-order cells.
const Z: &[u32] = &[20, 21, 22, 23];

/// Pairs that differ in one input and straddle the gate: (window stopped, window live).
const STRADDLES: &[(&str, &str)] = &[
    ("4", "5"),   // central infusion row before vs after a co-timed `SS=1` row
    ("14", "15"), // #1121 residual with vs without a later `SS=1` record
    ("22", "23"), // zero-order row before vs after a co-timed `SS=1` row
];

/// `ids` and their event-driven twins (+ 100).
fn with_twins(ids: &[u32]) -> Vec<u32> {
    ids.iter()
        .copied()
        .chain(ids.iter().map(|i| i + 100))
        .collect()
}

const PARAMS: &str = r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  omega ETA_CL ~ 0.0
  sigma ADD_ERR ~ 0.1 (sd)
"#;

const TAIL: &str = r#"
[error_model]
  DV ~ additive(ADD_ERR)
"#;

const TOL: &str = r#"
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-11
"#;

/// The ODE model. `scaled = false` drops `[scaling]` for the EKF leg, whose entry point reads
/// the raw `central` amount (the leg divides by `V` itself).
fn ode_model(zero_order: bool, scaled: bool) -> CompiledModel {
    let (ind, depot) = if zero_order {
        ("DUR = 8.0", "zero_order(dur=DUR) - KA*depot")
    } else {
        ("ALAG2 = LAG2", "-KA*depot")
    };
    let scaling = if scaled {
        "[scaling]\n  y = central / V"
    } else {
        ""
    };
    let src = format!(
        r#"{PARAMS}
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  {ind}
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = {depot}
  d/dt(central) = KA*depot - CL/V*central
{scaling}
{TAIL}
{TOL}
"#
    );
    parse_full_model(&src).expect("ODE model parses").model
}

fn analytic() -> CompiledModel {
    let src = format!(
        r#"{PARAMS}
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
{TAIL}
"#
    );
    parse_full_model(&src).expect("analytic model parses").model
}

/// The subjects `ids`, from the zero-order dataset when they are zero-order cells.
fn pop(ids: &[u32]) -> Population {
    let zo = ids.iter().any(|i| Z.contains(&(i % 100)));
    assert!(
        ids.iter().all(|i| Z.contains(&(i % 100)) == zo),
        "one dataset per fixture"
    );
    let path = if zo { DATA_ZO } else { DATA };
    let mut p = read_nonmem_csv(std::path::Path::new(path), None, None).expect("dataset loads");
    p.subjects
        .retain(|s| ids.iter().any(|i| s.id == i.to_string()));
    assert_eq!(p.subjects.len(), ids.len(), "every fixture ID is in {path}");
    p
}

/// `(ID, TIME, PRED)` for every observation of the committed `ADVAN2` table, in table order.
fn nonmem_obs() -> Vec<(String, f64, f64)> {
    let text = std::fs::read_to_string(TABLE).expect("NONMEM table");
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

/// Assert the engine's `(ID, TIME, value)` rows reproduce NONMEM on `ids` within
/// `bound · (1 + |want|)`. Every value is asserted finite before it is folded (`f64::max`
/// discards `NaN`); every miss is listed with its engine and ID; an ID the engine did not
/// report fails.
fn assert_matches(label: &str, got: &[(String, f64, f64)], ids: &[u32], bound: f64) {
    let ids: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    let want: Vec<_> = nonmem_obs()
        .into_iter()
        .filter(|r| ids.contains(&r.0))
        .collect();
    assert_eq!(got.len(), want.len(), "{label}: observation count");
    let mut worst = 0.0_f64;
    let mut bad = Vec::new();
    for ((gid, gt, g), (id, t, w)) in got.iter().zip(&want) {
        assert!(
            gid == id && (gt - t).abs() < 1e-12,
            "{label}: row order ({gid}, {gt}) vs NONMEM ({id}, {t})"
        );
        assert!(g.is_finite(), "{label}: ID {id} t={t} non-finite ({g})");
        let err = (g - w).abs() / (1.0 + w.abs());
        if !(err < bound) {
            bad.push(format!(
                "{label} ID {id} t={t}: ferx {g} vs NONMEM {w} (err {err:.3e})"
            ));
        }
        worst = worst.max(err);
    }
    println!("#1586 {label}: worst err vs NONMEM {worst:.3e} (bound {bound:.0e})");
    assert!(
        bad.is_empty(),
        "{label}: {} rows off (an `SS=1` record must stop every window recorded before it):\n{}",
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

/// Engine-side values carry the reader's shifted clock on the EVID=3 twins, so they are
/// paired with the raw times `predict` reports, row for row.
fn with_raw(
    model: &CompiledModel,
    pop: &Population,
    rows: Vec<(String, f64)>,
) -> Vec<(String, f64, f64)> {
    let raw = via_predict(model, pop);
    assert_eq!(rows.len(), raw.len(), "row count vs predict");
    rows.into_iter()
        .zip(raw)
        .map(|((id, v), (rid, t, _))| {
            assert_eq!(id, rid, "row order vs predict");
            (id, t, v)
        })
        .collect()
}

/// `compute_predictions_with_states`: `ode_predictions_with_states` on IDs < 100,
/// `ode_predictions_event_driven_with_states` (through `apply_segment_boundary`) on the
/// EVID=3 twins.
fn via_with_states(model: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let eta = vec![0.0; model.n_eta];
    let mut out = Vec::new();
    for s in &pop.subjects {
        let (ipred, _) =
            compute_predictions_with_states(model, s, &model.default_params.theta, &eta);
        out.extend(ipred.into_iter().map(|v| (s.id.clone(), v)));
    }
    with_raw(model, pop, out)
}

/// A dual walk's value (`o.f`). Every subject must stay on the analytic provider: an
/// FD-routed subject would make this leg FD-of-f64 and blind to the dual walk (T3).
fn via_dual(
    model: &CompiledModel,
    pop: &Population,
    sens: impl Fn(&Subject) -> Option<ferx_core::sens::provider::SubjectSens>,
) -> Vec<(String, f64, f64)> {
    let mut out = Vec::new();
    for s in &pop.subjects {
        let r = sens(s)
            .unwrap_or_else(|| panic!("ID {}: routed to FD, the dual walk is not exercised", s.id));
        out.extend(r.obs.into_iter().map(|o| (s.id.clone(), o.f)));
    }
    with_raw(model, pop, out)
}

fn via_ode_dual(model: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let eta = vec![0.0; model.n_eta];
    via_dual(model, pop, |s| {
        ode_subject_sensitivities(model, s, &model.default_params.theta, &eta)
    })
}

fn via_analytic_dual(model: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let eta = vec![0.0; model.n_eta];
    via_dual(model, pop, |s| {
        subject_sensitivities(model, s, &model.default_params.theta, &eta)
    })
}

/// `simulate_adaptive` with a controller that never doses: the base regimen through the
/// adaptive driver, frozen-replay verifier on.
fn via_adaptive(model: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let mut opts = AdaptiveSimulateOptions::default();
    opts.seed = Some(1);
    opts.decision_times = vec![50.0];
    let res = simulate_adaptive(
        model,
        pop,
        &model.default_params,
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

/// The EKF propagator with zero diffusion: its mean is the deterministic ODE solution. It
/// reads the raw `central` amount, so the leg runs the unscaled twin and divides by `V`.
fn via_ekf(pop: &Population) -> Vec<(String, f64, f64)> {
    let m = ode_model(false, false);
    let eta = vec![0.0; m.n_eta];
    let ode = m.ode_spec.as_ref().expect("ODE spec");
    let mut out = Vec::new();
    for s in &pop.subjects {
        let pk = (m.pk_param_fn)(&m.default_params.theta, &eta, &s.covariates, 0.0);
        let (amounts, _) = ferx_core::ode::predictions::ode_predictions_ekf_with_diffusion(
            ode,
            &pk.values,
            s,
            &[0.0, 0.0],
            |_| 1.0,
        );
        out.extend(amounts.into_iter().map(|a| (s.id.clone(), a / 20.0)));
    }
    with_raw(&ode_model(false, true), pop, out)
}

// Measured at the fix (err = |ferx − NONMEM| / (1 + |NONMEM|)): ODE predict / with-states
// 2.6e-12, ODE dual 2.5e-12, EKF 2.3e-12 (38× under `ODE_BOUND`); analytic value 8.6e-16 and
// dual 9.8e-16 (100× under `ANALYTIC_BOUND`); adaptive 3.1e-4 (the driver's own floor, #1603;
// 3×). The defects guarded are +14 % (ID 2) to +175 % (ID 22) by t = 15 / 21, i.e. err of
// order 0.1, two orders above the loosest bound.
const ODE_BOUND: f64 = 1e-10;
const ANALYTIC_BOUND: f64 = 1e-13;
const ADAPTIVE_BOUND: f64 = 1e-3;

/// The fixture straddles the gate: each pair differs in one input, NONMEM stops the window
/// in the first member and runs it in the second, and from t = 12 (after both records) the
/// two differ at every observation by more than 1 % and somewhere by more than 10 %. (Not
/// 10 % pointwise: 14 decays from its record while 15's residual rises, so they cross —
/// measured, the closest row is 3.8 % apart, at t = 40.) So an engine matching the table on
/// both members sees the gate from both sides, and a regenerated table that collapsed a pair
/// would fail here rather than silently turn the twins into a tautology.
#[test]
fn the_nonmem_table_straddles_the_ss_record_gate() {
    let want = nonmem_obs();
    let late = |id: &str| -> Vec<f64> {
        want.iter()
            .filter(|(i, t, _)| i == id && *t >= 12.0)
            .map(|r| r.2)
            .collect()
    };
    for &(stopped, live) in STRADDLES {
        let (s, l) = (late(stopped), late(live));
        assert!(!s.is_empty() && s.len() == l.len(), "IDs {stopped}/{live}");
        let rel: Vec<f64> = s
            .iter()
            .zip(&l)
            .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()))
            .collect();
        assert!(
            rel.iter().all(|&r| r > 0.01) && rel.iter().any(|&r| r > 0.1),
            "IDs {stopped}/{live} must differ after the record (> 1 % everywhere, > 10 % \
             somewhere): relative gaps {rel:?}"
        );
    }
}

#[test]
fn ode_predict_stops_a_window_recorded_before_the_ss_record() {
    for (zo, ids) in [(false, [A0, A4].concat()), (true, Z.to_vec())] {
        let ids = with_twins(&ids);
        let m = ode_model(zo, true);
        assert_matches(
            &format!("ODE predict (zero_order {zo})"),
            &via_predict(&m, &pop(&ids)),
            &ids,
            ODE_BOUND,
        );
    }
}

#[test]
fn ode_with_states_stops_a_window_recorded_before_the_ss_record() {
    for (zo, ids) in [(false, [A0, A4].concat()), (true, Z.to_vec())] {
        let ids = with_twins(&ids);
        let m = ode_model(zo, true);
        let p = pop(&ids);
        assert_matches(
            &format!("ODE with-states (zero_order {zo})"),
            &via_with_states(&m, &p),
            &ids,
            ODE_BOUND,
        );
    }
}

#[test]
fn adaptive_driver_stops_a_window_recorded_before_the_ss_record() {
    for (zo, ids) in [(false, [A0, A4].concat()), (true, Z.to_vec())] {
        let ids = with_twins(&ids);
        let m = ode_model(zo, true);
        assert_matches(
            &format!("adaptive (zero_order {zo})"),
            &via_adaptive(&m, &pop(&ids)),
            &ids,
            ADAPTIVE_BOUND,
        );
    }
}

/// The EKF has no lag, no reset and no input-rate forcing support, so only the lag-0
/// infusion cells without the leading EVID=3 are meaningful on it.
#[test]
fn ekf_stops_an_infusion_recorded_before_the_ss_record() {
    assert_matches("EKF", &via_ekf(&pop(A0)), A0, ODE_BOUND);
}

#[test]
fn ode_dual_walk_stops_a_window_recorded_before_the_ss_record() {
    for (zo, ids) in [(false, [A0, A4].concat()), (true, Z.to_vec())] {
        let ids = with_twins(&ids);
        let m = ode_model(zo, true);
        assert_matches(
            &format!("ODE dual (zero_order {zo})"),
            &via_ode_dual(&m, &pop(&ids)),
            &ids,
            ODE_BOUND,
        );
    }
}

#[test]
fn analytic_predict_stops_an_infusion_recorded_before_the_ss_record() {
    let ids = with_twins(A0);
    assert_matches(
        "analytic predict",
        &via_predict(&analytic(), &pop(&ids)),
        &ids,
        ANALYTIC_BOUND,
    );
}

#[test]
fn analytic_dual_stops_an_infusion_recorded_before_the_ss_record() {
    let ids = with_twins(A0);
    assert_matches(
        "analytic dual",
        &via_analytic_dual(&analytic(), &pop(&ids)),
        &ids,
        ANALYTIC_BOUND,
    );
}

/// T3, the value/gradient seam: on an analytic model a central infusion routes the value to
/// superposition and the dual to the event walk. Both must describe the same dosing history,
/// per ID, without either leg falling back to FD (`via_dual` panics on `None`). IDs 3, 4, 9
/// and 10 are the cells that were exact in value and 43–93 % off in the dual before #1586.
#[test]
fn analytic_dual_value_equals_the_prediction_on_every_id() {
    let ids = with_twins(A0);
    let (m, p) = (analytic(), pop(&ids));
    let dual = via_analytic_dual(&m, &p);
    let value = via_predict(&m, &p);
    let mut worst = 0.0_f64;
    for ((id, t, d), (_, _, v)) in dual.iter().zip(&value) {
        assert!(d.is_finite() && v.is_finite(), "ID {id} t={t}: {d} / {v}");
        let err = (d - v).abs() / (1.0 + v.abs());
        assert!(
            err < ANALYTIC_BOUND,
            "ID {id} t={t}: dual {d} vs value {v} (err {err:.3e})"
        );
        worst = worst.max(err);
    }
    println!("#1586 analytic dual vs value: worst {worst:.3e}");
}
