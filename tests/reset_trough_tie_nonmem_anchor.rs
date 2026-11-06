//! NONMEM anchor for #2728: an observation listed between a co-timed `EVID=3` reset and a
//! dose is scored on the reset state.
//!
//! The reader's #614 pass moves an observation listed before a co-timed dose one ULP below
//! the tie, so the engines reach it before the dose. A reset at the tie is always its first
//! record (the reader shifts a reset past every record before it), so the observation was
//! listed after the reset; moved alone, it sat below the reset too and read the state from
//! before it. The reset now moves with it.
//!
//! `nonmem_anchor/reset_trough_tie.ctl` (`ADVAN6 TOL=9`, every `$THETA` `FIX`,
//! `METHOD=1 INTER MAXEVAL=0 POSTHOC`): one state, `A_0(1)` re-seeded at each reset, the
//! seed doubling at `TIME ≥ 30`, so a reset whose `$PK` read its moved engine time
//! (one ULP below 30) instead of its written `TIME` would seed half the level.
//!
//! * ID 1: `EVID=3`, obs, dose at 30 — the #2728 case; the obs reads the reset state (10).
//! * ID 2: obs, `EVID=3`, dose — the control; the obs reads the pre-reset state (0.7468).
//! * ID 3: `EVID=3`, dose, obs — post-dose (20). ID 4: `EVID=3`, obs, no dose.
//! * ID 5: `EVID=3`, obs, obs, dose — two troughs at the tie.
//! * ID 6: a second period restarting `TIME` at 0, then ID 1's tie at its written 30, so
//!   the moved reset's written time is read through a non-zero period offset.
//!
//! On the merge-base `c32f1be9`, `predict()` gave PRED 0.7468 (NONMEM 10) at ID 1's tie,
//! and on IDs 4–6's, and the objective was far off NONMEM's.
//!
//! Tier 2: one `predict()`, one dual walk per subject and one `fit()` at `maxiter = 0`.

use ferx_core::sens::provider::subject_eta_grad;
use ferx_core::{
    fit, parse_full_model_file, predict, read_nonmem_csv, CompiledModel, FitOptions, Population,
};
use std::path::Path;

const NAME: &str = "reset_trough_tie";

/// `(ID, TIME, column)` of every observation (EVID = 0) row in the NONMEM `.tab`.
fn nonmem_column(column: &str) -> Vec<(String, f64, f64)> {
    let path = format!("nonmem_anchor/results/{NAME}.tab");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text
        .lines()
        .skip_while(|l| !l.trim_start().starts_with("ID"));
    let header: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
    let col = |c: &str| header.iter().position(|h| *h == c).expect(c);
    let (id, time, evid, value) = (col("ID"), col("TIME"), col("EVID"), col(column));
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
        .map(|r| (format!("{}", r[id] as i64), r[time], r[value]))
        .collect()
}

/// The final `#OBJV` in the NONMEM `.ext` (the `-1000000000` row).
fn nonmem_objv() -> f64 {
    let path = format!("nonmem_anchor/results/{NAME}.ext");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let row = text
        .lines()
        .find(|l| l.split_whitespace().next() == Some("-1000000000"))
        .expect("final-estimate row");
    row.split_whitespace().last().unwrap().parse().unwrap()
}

fn setup() -> (CompiledModel, FitOptions, Population) {
    let parsed = parse_full_model_file(Path::new(&format!("nonmem_anchor/{NAME}_fit.ferx")))
        .unwrap_or_else(|e| panic!("{NAME}: {e}"));
    let pop = read_nonmem_csv(Path::new(&format!("nonmem_anchor/{NAME}.csv")), None, None)
        .unwrap_or_else(|e| panic!("{NAME}: {e}"));
    (parsed.model, parsed.fit_options, pop)
}

/// Compare `got` with NONMEM's rows one by one; returns the worst absolute error. Each
/// value is checked finite before the fold (`f64::max` would drop a NaN).
fn assert_matches(leg: &str, got: &[(String, f64, f64)], bound: f64) -> f64 {
    let nm = nonmem_column("PRED");
    assert_eq!(got.len(), nm.len(), "{leg}: observation count");
    let mut worst = 0.0_f64;
    for ((id, t, v), (nid, nt, want)) in got.iter().zip(&nm) {
        assert!(
            id == nid && (t - nt).abs() < 1e-9,
            "{leg}: ferx row ({id}, {t}) is NONMEM's ({nid}, {nt})"
        );
        assert!(v.is_finite(), "{leg} ID {id} TIME {t}: {v}");
        let err = (v - want).abs();
        assert!(err.is_finite(), "{leg} ID {id} TIME {t}: NONMEM {want}");
        assert!(
            err < bound,
            "{leg} ID {id} TIME {t}: ferx {v} vs NONMEM {want} (|error| {err:.3e})"
        );
        worst = worst.max(err);
    }
    println!(
        "{leg}: worst |ferx − NONMEM| {worst:.3e} over {} observations",
        nm.len()
    );
    worst
}

fn via_predict(m: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    predict(m, pop, &m.default_params)
        .expect("predict")
        .into_iter()
        .map(|p| (p.id, p.time, p.pred))
        .collect()
}

// Measured worst |ferx − NONMEM PRED| on Linux x86_64: 8.3e-11 on `predict()` and on the
// dual walk, so ~1e4× headroom. The smallest defect it guards: the reset's `$PK` reading
// its moved engine time (seed 5, not 10) at ID 1's tie, 5.0 away; the tie reading the
// pre-reset state is 9.25 away.
const PRED_BOUND: f64 = 1e-6;

/// `predict()` at η = 0 against NONMEM's `PRED`, including the control (ID 2) whose obs
/// reads the pre-reset state, so a fix that moved every co-timed obs past the reset fails.
#[test]
fn predict_scores_an_obs_between_a_reset_and_a_dose_on_the_reset_state() {
    let (m, _, pop) = setup();
    let got = via_predict(&m, &pop);
    assert_matches("predict", &got, PRED_BOUND);
    // The straddle itself, on NONMEM's own table: ID 1 (after the reset) and ID 2 (before
    // it) differ at the tie, so this fixture can tell the two orders apart.
    let nm = nonmem_column("PRED");
    let at_tie = |id: &str| {
        nm.iter()
            .find(|r| r.0 == id && r.1 == 30.0)
            .map(|r| r.2)
            .unwrap()
    };
    assert!(
        (at_tie("1") - at_tie("2")).abs() > 1.0,
        "the pair no longer straddles the reset"
    );
}

/// The dual ODE walk (the adjoint's recorder) at η = 0, rows labelled with `predict()`'s.
#[test]
fn dual_walk_scores_an_obs_between_a_reset_and_a_dose_on_the_reset_state() {
    let (m, _, pop) = setup();
    let eta = vec![0.0; m.n_eta];
    let labels = via_predict(&m, &pop);
    let mut got = Vec::new();
    for s in &pop.subjects {
        let obs = subject_eta_grad(&m, s, &m.default_params.theta, &eta)
            .unwrap_or_else(|| panic!("ID {}: routed to FD, the dual walk is not exercised", s.id));
        assert_eq!(obs.len(), s.obs_times.len(), "ID {}: dual rows", s.id);
        got.extend(obs.iter().map(|o| (s.id.clone(), o.f)));
    }
    assert_eq!(got.len(), labels.len(), "dual vs predict row count");
    let got: Vec<_> = got
        .into_iter()
        .zip(labels)
        .map(|((id, f), (pid, t, _))| {
            assert_eq!(id, pid, "dual and predict rows are in one subject order");
            (id, t, f)
        })
        .collect();
    assert_matches("dual", &got, PRED_BOUND);
}

// Measured |ferx − NONMEM #OBJV| on Linux x86_64: 4.0e-9, so 2500× headroom. With the
// reset left in place the OFV was 18315 (NONMEM −72.46); with it moved but reading its
// moved time, 19255.
const OFV_BOUND: f64 = 1e-5;

/// `fit()` at `maxiter = 0` against NONMEM's `#OBJV`, η live (ID 1's EBE is pulled by
/// the tie when it reads the wrong state).
#[test]
fn fit_objective_with_an_obs_between_a_reset_and_a_dose_matches_nonmem() {
    let (m, opts, pop) = setup();
    let r = fit(&m, &pop, &m.default_params, &opts).expect("fit");
    let want = nonmem_objv();
    assert!(r.ofv.is_finite(), "OFV {}", r.ofv);
    let err = (r.ofv - want).abs();
    println!("ferx OFV {:.10}, NONMEM {want:.10}, |Δ| {err:.3e}", r.ofv);
    assert!(
        err < OFV_BOUND,
        "ferx OFV {} vs NONMEM #OBJV {want} (|Δ| {err:.3e})",
        r.ofv
    );
    assert!(
        !r.warnings
            .iter()
            .any(|w| w.contains("finite-difference inner gradients")),
        "a subject routed to FD, so the dual path was not exercised: {:?}",
        r.warnings
    );
}
