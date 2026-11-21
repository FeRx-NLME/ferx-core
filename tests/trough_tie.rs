//! #2124 / #2123: an observation listed **before** a co-timed dose or `EVID=2` row reads the
//! state before that record, on every ODE engine and at every instant — as NONMEM does.
//!
//! The reader places such an observation one ULP earlier (#614 / #1810), and a dose matches
//! a break symmetrically within 1e-12. So whenever the observation's time is a break of its
//! own, the dose listed after it was applied there and the observation read post-dose:
//!
//! * at the subject's **first** record time, at any time (IDs 1, 8);
//! * at its **last** observation, which is always a break (`t_last`), at any time (IDs 10, 12);
//! * on the adaptive driver's **time-varying** path, where every observation is a break, at
//!   a time ≤ 8, where one ULP is below the 1e-15 break dedup and the dose's break merged into
//!   the observation's (ID 14; above 8 the two breaks stay apart, #2532). The same merge made
//!   the break-alignment guard refuse an observation before a co-timed `EVID=2` row outright
//!   (#2123; ID 7).
//!
//! At any other instant the observation was recorded at the end of the segment before the
//! dose, so it already read pre-dose and must still (IDs 3, 5, 6). Measured on `65e9d6f8`
//! (`main` before this fix): ID 1 read 16 on `predict()`, the dual walk and the state pass;
//! ID 8 16, ID 10 10.781, ID 12 4.165 on `predict()` and the dual walk; ID 14 12.725 on the
//! adaptive driver; ID 7 refused — each the post-dose value, NONMEM's being the pre-dose one.
//!
//! Model `nonmem_anchor/init_seed_tie_fit.ferx`: one compartment, CL = 5, V = 50
//! (k = 0.1), `init(central) = 10·WT`, so the concentration is `WT/5 = 14` at the start
//! and every 100-unit dose adds 2. WT reaches the prediction only through that seed, and
//! every subject's first record has WT = 70. With nothing else in the model each value is a
//! closed form, so the oracle is exact rather than a second engine: a ferx engine that
//! disagrees with it is wrong.
//!
//! | ID | records | what it pins |
//! |---|---|---|
//! | 1 | obs, dose at 0 | the first-instant trough reads the seed, 14 |
//! | 2 | dose, obs at 0 | control: dose first reads post-dose, 16 |
//! | 3 | obs, dose at 4 | a trough with residual drug, `16e^{-0.4}`, inside the merge |
//! | 4 | dose, obs at 4 | control for ID 3: post-dose |
//! | 5 | obs, dose at 8.2 | one ULP is 1.8e-15 here, so the breaks never merged |
//! | 6 | obs, dose at 24 | a day-later trough, `16e^{-2.4}` |
//! | 7 | obs, `EVID=2` at 4 | #2123 at a later instant (the row changes WT, which is inert) |
//! | 8 | first record at 10: obs, dose | the first instant **above 8**: the seed, 14 |
//! | 9 | first record at 10: dose, obs | control for ID 8: post-dose, 16 |
//! | 10 | obs, dose at 6, the last records | a last-observation trough, `16e^{-0.6}` |
//! | 11 | dose, obs at 6, the last records | control for ID 10: post-dose |
//! | 12 | obs, dose at 20, the last records | a last-observation trough above 8, `16e^{-2}` |
//! | 13 | dose, obs at 20, the last records | control for ID 12: post-dose |
//! | 14 | obs, dose at 4, `EVID=2` at 5 | the time-varying path inside the merge, `16e^{-0.4}` |
//! | 15 | dose, obs at 4, `EVID=2` at 5 | control for ID 14: post-dose |
//!
//! IDs 1 / 2, 3 / 4, 8 / 9, 10 / 11, 12 / 13 and 14 / 15 each differ only in the file order
//! of the tied pair, so they straddle the rule: an engine ignoring the order passes at most
//! one of each.
//!
//! **Two oracles, one dataset.** `nonmem_anchor/trough_tie.ctl` runs this exact file
//! (`nonmem_anchor/trough_tie.csv`) through NONMEM 7.6.0 under the same model, and
//! [`nonmem_reads_every_cell_as_the_closed_form`] checks its `IPRED` against the closed forms
//! row by row. The engine legs are judged against the closed forms, so each is anchored to
//! NONMEM through them, and a fixture row added without re-running NONMEM fails that test
//! on the row count.
//!
//! Tier 2: one `predict()`, one dual walk, one state pass and one no-op adaptive run per
//! subject, and a read of the committed NONMEM table.

use ferx_core::sens::provider::subject_eta_grad;
use ferx_core::sim::adaptive::{ControllerCtx, DoseAction};
use ferx_core::{
    parse_full_model_file, predict, read_nonmem_csv, simulate_adaptive, AdaptiveSimulateOptions,
    CompiledModel, Population,
};
use std::path::Path;

const K: f64 = 0.1;
const SEED: f64 = 14.0;
const DOSE: f64 = 2.0;

/// The concentration at `t` with the seed at 0 and a dose at each of `doses` already applied.
fn conc(t: f64, doses: &[f64]) -> f64 {
    SEED * (-K * t).exp() + doses.iter().map(|td| DOSE * (-K * (t - td)).exp()).sum::<f64>()
}

/// `(ID, TIME, closed form)` for every observation of `nonmem_anchor/trough_tie.csv`, in file order.
fn expected() -> Vec<(&'static str, f64, f64)> {
    vec![
        ("1", 0.0, conc(0.0, &[])),
        ("1", 2.0, conc(2.0, &[0.0])),
        ("2", 0.0, conc(0.0, &[0.0])),
        ("2", 2.0, conc(2.0, &[0.0])),
        ("3", 2.0, conc(2.0, &[0.0])),
        ("3", 4.0, conc(4.0, &[0.0])),
        ("3", 6.0, conc(6.0, &[0.0, 4.0])),
        ("4", 2.0, conc(2.0, &[0.0])),
        ("4", 4.0, conc(4.0, &[0.0, 4.0])),
        ("4", 6.0, conc(6.0, &[0.0, 4.0])),
        ("5", 8.2, conc(8.2, &[0.0])),
        ("5", 10.0, conc(10.0, &[0.0, 8.2])),
        ("6", 24.0, conc(24.0, &[0.0])),
        ("6", 26.0, conc(26.0, &[0.0, 24.0])),
        ("7", 4.0, conc(4.0, &[0.0])),
        ("7", 6.0, conc(6.0, &[0.0])),
        // IDs 8 / 9 start at TIME 10, where the seed is applied: their clock is TIME − 10.
        ("8", 10.0, conc(0.0, &[])),
        ("8", 12.0, conc(2.0, &[0.0])),
        ("9", 10.0, conc(0.0, &[0.0])),
        ("9", 12.0, conc(2.0, &[0.0])),
        ("10", 6.0, conc(6.0, &[0.0])),
        ("11", 6.0, conc(6.0, &[0.0, 6.0])),
        ("12", 20.0, conc(20.0, &[0.0])),
        ("13", 20.0, conc(20.0, &[0.0, 20.0])),
        ("14", 4.0, conc(4.0, &[0.0])),
        ("14", 6.0, conc(6.0, &[0.0, 4.0])),
        ("15", 4.0, conc(4.0, &[0.0, 4.0])),
        ("15", 6.0, conc(6.0, &[0.0, 4.0])),
    ]
}

fn setup() -> (CompiledModel, Population) {
    let m = parse_full_model_file(Path::new("nonmem_anchor/init_seed_tie_fit.ferx"))
        .expect("model")
        .model;
    let pop = read_nonmem_csv(Path::new("nonmem_anchor/trough_tie.csv"), None, None).expect("data");
    (m, pop)
}

/// Compare an engine's rows with the closed forms. Every value is checked finite first and
/// against the bound individually, so the message names the row. Returns the worst error.
fn assert_closed_form(leg: &str, got: &[(String, f64, f64)]) -> f64 {
    let want = expected();
    assert_eq!(got.len(), want.len(), "{leg}: observation count");
    let mut worst = 0.0_f64;
    for ((id, t, v), (wid, wt, w)) in got.iter().zip(&want) {
        assert!(
            id == wid && (t - wt).abs() < 1e-9,
            "{leg}: row ({id}, {t}) is expected ({wid}, {wt})"
        );
        assert!(v.is_finite(), "{leg} ID {id} TIME {t}: {v}");
        let err = (v - w).abs();
        assert!(
            err < BOUND,
            "{leg} ID {id} TIME {t}: ferx {v} vs closed form {w} (|error| {err:.3e})"
        );
        worst = worst.max(err);
    }
    println!("{leg}: worst |ferx − closed form| {worst:.3e} over {} observations", want.len());
    worst
}

// Measured worst |ferx − closed form| on macOS arm64 over the 28 observations: 3.6e-15 on
// predict, the dual walk and the state pass, 5.3e-15 on the adaptive driver (a linear
// one-state system, so the solve is near exact). Not yet measured on Linux, hence the wide
// headroom (1,900×). The smallest defect it guards is the dose, 2.
const BOUND: f64 = 1e-11;

fn via_predict(m: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    predict(m, pop, &m.default_params)
        .expect("predict")
        .into_iter()
        .map(|p| (p.id, p.time, p.pred))
        .collect()
}

/// Label one value per observation with `predict()`'s `(ID, TIME)`.
fn label(
    m: &CompiledModel, pop: &Population, values: Vec<(String, f64)>,
) -> Vec<(String, f64, f64)> {
    let labels = via_predict(m, pop);
    assert_eq!(values.len(), labels.len(), "row count against predict");
    values
        .into_iter()
        .zip(labels)
        .map(|((id, f), (pid, t, _))| {
            assert_eq!(id, pid, "rows are in one subject order");
            (id, t, f)
        })
        .collect()
}

/// `(ID, TIME, IPRED)` of every observation (`EVID = 0`) row of
/// `nonmem_anchor/results/{stem}.tab`, in file order, by column name.
fn nonmem_ipred(stem: &str) -> Vec<(String, f64, f64)> {
    let path = format!("nonmem_anchor/results/{stem}.tab");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text.lines().skip_while(|l| !l.trim_start().starts_with("ID"));
    let header: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
    let col = |c: &str| header.iter().position(|h| *h == c).expect(c);
    let (id, time, evid, ipred) = (col("ID"), col("TIME"), col("EVID"), col("IPRED"));
    lines
        .map(|l| {
            l.split_whitespace()
                .map(|v| v.parse::<f64>().unwrap_or_else(|e| panic!("{path}: {v}: {e}")))
                .collect::<Vec<f64>>()
        })
        .filter(|r| r[evid] == 0.0)
        .map(|r| (format!("{}", r[id] as i64), r[time], r[ipred]))
        .collect()
}

// Measured worst |NONMEM − closed form|: 2.566e-8 (ADVAN13 TOL=9; the table is committed, so
// this is the same on every platform), as on `init_seed_tie`'s anchor. ~4× headroom; the
// smallest defect it guards is the dose, 2, so a NONMEM run reading any observation-first
// cell post-dose fails by 2e7× the bound.
const NONMEM_BOUND: f64 = 1e-7;

/// NONMEM 7.6.0 on this dataset (`nonmem_anchor/trough_tie.ctl`) reads every cell as the
/// closed form: each observation listed before a co-timed dose or `EVID=2` row pre-dose, at
/// the first instant, the last observation and on the time-varying subjects alike. This is
/// what makes the closed forms NONMEM's convention rather than ours, and it fails on the row
/// count when a subject is added to the fixture without re-running NONMEM.
#[test]
fn nonmem_reads_every_cell_as_the_closed_form() {
    let got = nonmem_ipred("trough_tie");
    let want = expected();
    assert_eq!(got.len(), want.len(), "NONMEM table rows vs fixture observations");
    let mut worst = 0.0_f64;
    for ((id, t, v), (wid, wt, w)) in got.iter().zip(&want) {
        assert!(
            id == wid && (t - wt).abs() < 1e-9,
            "NONMEM row ({id}, {t}) is expected ({wid}, {wt})"
        );
        assert!(v.is_finite(), "NONMEM ID {id} TIME {t}: {v}");
        let err = (v - w).abs();
        assert!(
            err < NONMEM_BOUND,
            "NONMEM ID {id} TIME {t}: {v} vs closed form {w} (|error| {err:.3e})"
        );
        worst = worst.max(err);
    }
    println!("nonmem: worst |NONMEM − closed form| {worst:.3e} over {} observations", want.len());
}

#[test]
fn predict_reads_a_trough_before_its_co_timed_record() {
    let (m, pop) = setup();
    assert_closed_form("predict", &via_predict(&m, &pop));
}

/// The adjoint's recorder: the dual walk `fit()`'s gradients differentiate. It had the same
/// merge as the value path, so `fit()` scored a first-instant trough
/// post-dose.
#[test]
fn dual_walk_reads_a_trough_before_its_co_timed_record() {
    let (m, pop) = setup();
    let eta = vec![0.0; m.n_eta];
    let mut values = Vec::new();
    for s in &pop.subjects {
        let obs = subject_eta_grad(&m, s, &m.default_params.theta, &eta)
            .unwrap_or_else(|| panic!("ID {}: routed to FD, the dual walk is not exercised", s.id));
        assert_eq!(obs.len(), s.obs_times.len(), "ID {}: dual rows", s.id);
        values.extend(obs.iter().map(|o| (s.id.clone(), o.f)));
    }
    assert_closed_form("dual", &label(&m, &pop, values));
}

/// The state pass behind the sdtab compartment columns and `[derived]`, read from the
/// states (`central / V`), not from its `ipred`.
#[test]
fn state_pass_reads_a_trough_before_its_co_timed_record() {
    let (m, pop) = setup();
    let theta = &m.default_params.theta;
    let eta = vec![0.0; m.n_eta];
    let v = theta[m.theta_names.iter().position(|n| n == "TVV").expect("TVV")];
    let mut values = Vec::new();
    for s in &pop.subjects {
        let (_, states) = ferx_core::pk::compute_predictions_with_states(&m, s, theta, &eta);
        assert_eq!(states.len(), s.obs_times.len(), "ID {}: state rows", s.id);
        values.extend(states.iter().map(|u| (s.id.clone(), u[0] / v)));
    }
    assert_closed_form("states", &label(&m, &pop, values));
}

/// `simulate_adaptive` with a controller that never doses, the frozen-replay verifier on,
/// one subject at a time. `WT` is declared time-varying on IDs 7, 14 and 15 (their `EVID=2`
/// row changes it), which puts them on the driver's time-varying path, where #2123's guard
/// refused ID 7 and ID 14's trough read post-dose. The verifier alone cannot see a post-dose trough — the replay reads through
/// the same walk — so the leg is judged against the closed forms.
#[test]
fn adaptive_driver_reads_a_trough_before_its_co_timed_record() {
    let (m, pop) = setup();
    assert_closed_form("adaptive", &via_adaptive(&m, &pop));
}

/// `simulate_adaptive` per subject with a no-op controller and the verifier on.
fn via_adaptive(m: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let mut got = Vec::new();
    for s in &pop.subjects {
        let one = Population { subjects: vec![s.clone()], ..pop.clone() };
        let mut opts = AdaptiveSimulateOptions::default();
        opts.seed = Some(1);
        // Inside the subject's span: a decision before its first record (IDs 8 / 9 start at
        // 10) makes the frozen replay start integrating at the decision and refuse the run
        // whatever the record order — a separate defect, #2125.
        let first = s
            .doses
            .iter()
            .map(|d| d.time)
            .chain(s.obs_times.iter().copied())
            .fold(f64::INFINITY, f64::min); // nan-ok: the reader refuses a non-finite TIME
        opts.decision_times = vec![first + 1.0];
        opts.verify = true;
        let res = simulate_adaptive(
            m,
            &one,
            &m.default_params,
            1,
            || |_: &ControllerCtx| -> Vec<DoseAction> { Vec::new() },
            &opts,
        )
        .unwrap_or_else(|e| panic!("ID {}: adaptive run refused: {e}", s.id));
        assert!(res.ledger.is_empty(), "ID {}: the no-op controller dosed", s.id);
        got.extend(res.trajectories.into_iter().map(|r| (r.id, r.time, r.ipred)));
    }
    got
}

/// #2534: the same tie when a *derived* break lands one ULP below the dose, on the trough's
/// bits. `5.1 + 2.96` is `8.06` one ULP down, so an infusion of 2.96 at `RATE` 1 from 5.1
/// (IDs 1 / 2) ends there and a dose at 5.1 under `ALAG1 = 2.96` (IDs 3 / 4) arrives there.
/// The bolus at 8.06 fired at that break, so the trough listed before it read post-dose. The
/// infusion case was fixed by #2485. On `8391287e` the lag case read 144.664 on `predict()`
/// and the state pass, and `fit()` scored 827.394 at `maxiter = 0` (NONMEM 821.994), while
/// the dual walk read 44.664: the objective and its gradient described two functions. Model
/// `nonmem_anchor/derived_break_tie_fit.ferx` (K = 0.1, KA = 1, IV 100 at 0), anchored by
/// `nonmem_anchor/derived_break_tie.ctl`.
#[test]
fn a_trough_on_a_derived_break_reads_before_its_co_timed_dose_2534() {
    let parsed = parse_full_model_file(Path::new(TIE_MODEL)).expect("model");
    let m = &parsed.model;
    let pop = read_nonmem_csv(Path::new("nonmem_anchor/derived_break_tie.csv"), None, None)
        .expect("data");
    let (t0, t1) = (8.06, 10.0);
    let inf = 10.0 * (1.0 - e(2.96)); // the infusion's amount left at its end
    let base = 100.0 * (e(t1) + e(t1 - t0));
    let want = [
        100.0 * e(t0) + inf,
        base + inf * e(t1 - t0),
        100.0 * e(t0) + inf + 100.0,
        base + inf * e(t1 - t0),
        100.0 * e(t0),
        base + lagged_depot(),
        100.0 * e(t0) + 100.0,
        base + lagged_depot(),
    ];
    let (dual, states) = dual_and_states(m, &pop);
    let legs = [
        (
            "predict",
            predict(m, &pop, &m.default_params).expect("predict").iter().map(|p| p.pred).collect(),
        ),
        ("dual", dual),
        ("states", states),
        ("adaptive", via_adaptive(m, &pop).into_iter().map(|r| r.2).collect()),
        ("nonmem", nonmem_ipred("derived_break_tie").into_iter().map(|r| r.2).collect()),
    ];
    // Measured worst |leg − closed form| on macOS arm64: 1.7e-13 on the engines,
    // 9.5e-8 on NONMEM's table (`TOL=9`), so ~10× headroom. The smallest defect is 100.
    assert_legs(&legs, &want, 1e-6);
    let ext = std::fs::read_to_string("nonmem_anchor/results/derived_break_tie.ext").expect("ext");
    let mut lines = ext.lines().skip_while(|l| !l.trim_start().starts_with("ITERATION"));
    let head: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
    let obj = head.iter().position(|h| *h == "OBJ").expect("OBJ column");
    let objv: f64 = lines
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .find(|r| r[0].parse::<f64>() == Ok(-1e9))
        .expect("#OBJV row")[obj]
        .parse()
        .expect("OBJ");
    let ofv = ferx_core::fit(m, &pop, &m.default_params, &parsed.fit_options).expect("fit").ofv;
    // Measured |ferx − NONMEM #OBJV| on macOS arm64: 3.1e-8. The defect was 5.4.
    let d = (ofv - objv).abs();
    println!("fit: OFV {ofv:.8}, |Δ #OBJV| {d:.3e}");
    assert!(d < 1e-5, "fit OFV {ofv} vs NONMEM {objv}");
}

/// #2534 on the walks the anchor does not reach (#3045 review r1's mutation sweep): ID 3's
/// lag tie with an inert reset after the data, which sends the state pass down the dense
/// walk (`apply_segment_boundary`), and with the co-timed dose an `SS=1` depot dose, whose
/// lag seeds it at its record (`ss_seeded_at_record`), without and with the reset. An
/// `SS=1` dose resets the system, so each trough is the IV's alone, `100·e^{−0.806}`; every
/// row is also judged against the dual walk, which orders events by exact time. The adaptive
/// driver re-seeds an `SS=1` base dose before its decision hook (`reseed_prescheduled_states_at`).
#[test]
fn a_lag_tie_reads_pre_dose_on_the_reset_and_steady_state_walks_2534() {
    let m = parse_full_model_file(Path::new(TIE_MODEL)).expect("model").model;
    let head = "ID,TIME,DV,MDV,EVID,AMT,RATE,CMT,SS,II\n";
    let tie = |id: u32, ss: &str, reset: bool| {
        let mut r = format!(
            "{id},0,0,1,1,100,0,2,0,0\n{id},5.1,0,1,1,100,0,1,0,0\n{id},8.06,1,0,0,0,0,2,0,0\n\
             {id},8.06,0,1,1,100,0,{ss}\n{id},10,1,0,0,0,0,2,0,0\n"
        );
        if reset {
            r.push_str(&format!("{id},20,0,1,3,0,0,2,0,0\n"));
        }
        r
    };
    let csv = [head.into(), tie(5, "2,0,0", true), tie(6, "1,1,24", false), tie(7, "1,1,24", true)];
    let path = std::env::temp_dir().join(format!("ferx_2534_walks_{}.csv", std::process::id()));
    std::fs::write(&path, csv.concat()).expect("write");
    let pop = read_nonmem_csv(&path, None, None).expect("data");
    let _ = std::fs::remove_file(&path);
    let (dual, states) = dual_and_states(&m, &pop);
    let trough = 100.0 * e(8.06);
    for (k, want) in [(0, trough), (1, 100.0 * (e(10.0) + e(10.0 - 8.06)) + lagged_depot())] {
        assert!((dual[k] - want).abs() < 1e-6, "dual ID 5 row {k}: {} vs {want}", dual[k]);
    }
    assert!((dual[2] - trough).abs() < 1e-6 && (dual[4] - trough).abs() < 1e-6, "{dual:?}");
    let pred =
        predict(&m, &pop, &m.default_params).expect("predict").iter().map(|p| p.pred).collect();
    let adaptive = via_adaptive(&m, &pop).into_iter().map(|r| r.2).collect();
    assert_legs(&[("predict", pred), ("states", states), ("adaptive", adaptive)], &dual, 1e-6);
}

const TIE_MODEL: &str = "nonmem_anchor/derived_break_tie_fit.ferx";

fn e(t: f64) -> f64 {
    (-K * t).exp()
}

/// The lag-tie fixture's depot dose of 100 at 5.1, arrived at 8.06, read at 10 (KA = 1).
fn lagged_depot() -> f64 {
    100.0 / 0.9 * (e(10.0 - 8.06) - (8.06 - 10.0_f64).exp())
}

/// The dual walk's and the state pass's central amount per observation, in subject order.
fn dual_and_states(m: &CompiledModel, pop: &Population) -> (Vec<f64>, Vec<f64>) {
    let (th, eta) = (&m.default_params.theta, vec![0.0; m.n_eta]);
    let (mut dual, mut states) = (Vec::new(), Vec::new());
    for s in &pop.subjects {
        let d = subject_eta_grad(m, s, th, &eta).unwrap_or_else(|| panic!("ID {}: no dual", s.id));
        dual.extend(d.iter().map(|o| o.f));
        let (_, st) = ferx_core::pk::compute_predictions_with_states(m, s, th, &eta);
        states.extend(st.iter().map(|u| u[1]));
    }
    (dual, states)
}

/// Every leg's errors against `want` first, then one assert naming each leg off by `bound`
/// or more, so a failure shows which engines split. `all`, not a `max` fold, keeps a NaN.
fn assert_legs(legs: &[(&str, Vec<f64>)], want: &[f64], bound: f64) {
    let bad: Vec<String> = legs
        .iter()
        .filter_map(|(leg, got)| {
            let errs: Vec<f64> = got.iter().zip(want).map(|(g, w)| (g - w).abs()).collect();
            println!("{leg}: |error| {errs:?}");
            (got.len() != want.len() || !errs.iter().all(|e| *e < bound))
                .then(|| format!("{leg}: {got:?}"))
        })
        .collect();
    assert!(bad.is_empty(), "want {want:?}\n{}", bad.join("\n"));
}

/// The two walks that still match a dose by the band alone (`integrate_g`, the EKF; see
/// `event_on_break`) on the infusion-end tie, IDs 1 / 2 of the anchor: a no-lag twin goes to
/// the static dual walk, and an SDE twin to the EKF, whose state mean is the ODE's. Both
/// read the trough pre-dose. Measured worst error 5.7e-14 (dual) and 1.5e-11 (EKF).
#[test]
fn the_band_only_walks_read_an_infusion_end_tie_pre_dose_2534() {
    let ode = std::fs::read_to_string(TIE_MODEL)
        .expect("model")
        .replace("  theta TVLAG(2.96, 0.0, 1e15)\n", "")
        .replace("  ALAG1 = TVLAG\n", "");
    let sde = ode.replace("[error_model]", "[diffusion]\n  central ~ 1e-6\n[error_model]");
    let mut pop = read_nonmem_csv(Path::new("nonmem_anchor/derived_break_tie.csv"), None, None)
        .expect("data");
    pop.subjects.truncate(2);
    let inf = 10.0 * (1.0 - e(2.96));
    let at10 = 100.0 * (e(10.0) + e(10.0 - 8.06)) + inf * e(10.0 - 8.06);
    let want = [100.0 * e(8.06) + inf, at10, 100.0 * e(8.06) + inf + 100.0, at10];
    for src in [&ode, &sde] {
        let m = ferx_core::parse_model_string(src).expect("twin");
        assert!(!m.has_lagtime());
        let pred = predict(&m, &pop, &m.default_params).expect("predict");
        let (dual, _) = dual_and_states(&m, &pop);
        assert_legs(
            &[("predict", pred.iter().map(|p| p.pred).collect()), ("dual", dual)],
            &want,
            1e-9,
        );
    }
}
