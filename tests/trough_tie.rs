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
    SEED * (-K * t).exp()
        + doses
            .iter()
            .map(|td| DOSE * (-K * (t - td)).exp())
            .sum::<f64>()
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
    println!(
        "{leg}: worst |ferx − closed form| {worst:.3e} over {} observations",
        want.len()
    );
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
    m: &CompiledModel,
    pop: &Population,
    values: Vec<(String, f64)>,
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
/// `nonmem_anchor/results/trough_tie.tab`, in file order.
fn nonmem_ipred() -> Vec<(String, f64, f64)> {
    let path = "nonmem_anchor/results/trough_tie.tab";
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text
        .lines()
        .skip_while(|l| !l.trim_start().starts_with("ID"));
    let header: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
    let col = |c: &str| header.iter().position(|h| *h == c).expect(c);
    let (id, time, evid, ipred) = (col("ID"), col("TIME"), col("EVID"), col("IPRED"));
    lines
        .map(|l| {
            l.split_whitespace()
                .map(|v| {
                    v.parse::<f64>()
                        .unwrap_or_else(|e| panic!("{path}: {v}: {e}"))
                })
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
    let got = nonmem_ipred();
    let want = expected();
    assert_eq!(
        got.len(),
        want.len(),
        "NONMEM table rows vs fixture observations"
    );
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
    println!(
        "nonmem: worst |NONMEM − closed form| {worst:.3e} over {} observations",
        want.len()
    );
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
    let mut got = Vec::new();
    for s in &pop.subjects {
        let one = Population {
            subjects: vec![s.clone()],
            ..pop.clone()
        };
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
            .fold(f64::INFINITY, f64::min);
        opts.decision_times = vec![first + 1.0];
        opts.verify = true;
        let res = simulate_adaptive(
            &m,
            &one,
            &m.default_params,
            1,
            || |_: &ControllerCtx| -> Vec<DoseAction> { Vec::new() },
            &opts,
        )
        .unwrap_or_else(|e| panic!("ID {}: adaptive run refused: {e}", s.id));
        assert!(
            res.ledger.is_empty(),
            "ID {}: the no-op controller dosed",
            s.id
        );
        got.extend(
            res.trajectories
                .into_iter()
                .map(|r| (r.id, r.time, r.ipred)),
        );
    }
    assert_closed_form("adaptive", &got);
}
