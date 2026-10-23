//! NONMEM anchors for #1809 and #1810: the records that are neither a dose nor a scored
//! observation (`EVID=2`, and `EVID=0` with `MDV=1`).
//!
//! * `evid2_clock` (#1809): such a row at `TIME 0` starts the subject's clock, as NONMEM's
//!   first record of any type does. An `[odes]` model whose state moves on its own
//!   (`PLB·exp(−TF·T)` in the right-hand side) must integrate from 0, not from the first
//!   observation. IDs 1 and 3 have constant covariates, the subjects the reader used to drop
//!   the row on; IDs 2 and 4 change `EXPO` at the first observation; ID 5 adds a dose.
//! * `tied_evid2`, `tied_evid2_advan1` (#1810): #1810's seven co-timed cases on an `[odes]`
//!   model and on `ADVAN1`. The interval ending at a tie runs on the first record's `X` in
//!   file order.
//!
//! The control streams are the drafts NONMEM 7.6.0 ran (`nonmem:7.6.0-anchor`, `METHOD=0
//! MAXEVAL=0`), identical to each `.lst`'s echo; @TeunP ran them on PR #1875 and posted the
//! `results/` files and the `_fit.ferx` twins. The comparison is `PRED` (η = 0) against the
//! `.tab`, printed to 11 significant figures, so the `$ESTIMATION` method does not enter.
//! On the merge-base `9a12c915`, TeunP measured the worst `PRED` error as 3.90, 4.00 and 2.47,
//! so each anchor is red without the fixes.
//!
//! * `addl_tie` (#1909): an `ADDL`-implied dose is not an event record in NONMEM. The advance
//!   across it runs under the next record at or after it, so the implied dose takes that
//!   record's covariates, not its parent row's. Fourteen cases on `ADVAN1` with
//!   `CL = TVCL·X`. IDs 1–9: an `EVID=2` or `MDV=1` row, an observation, or nothing at the
//!   implied dose's time; a constant-`X` control; explicit doses tied with an `EVID=2` row
//!   in either order; an `ADDL` infusion; and a train whose next record is two doses away.
//!   IDs 10–14 give the record at the implied dose's time a different `X` from the record
//!   after it, so each rule of the selection is visible in `PRED`: a co-timed record
//!   governs (10); co-timed records resolve by file order, in both orders (11, 12); and
//!   an explicit dose governs, after the implied one (13) or tied with it (14). Each case
//!   is multi-dose, so each implied dose lands on residual drug. The analytic and ODE twins
//!   run against the same `.tab`. NONMEM 7.5.1, `METHOD=0 MAXEVAL=0`, `.tab` at 17
//!   significant figures. The control stream equals the `.lst`'s echo up to layout. A
//!   closed form outside both engines, a sum of exponentials over the records,
//!   reproduces the `.tab` to rounding under this rule. Under the parent-row rule it
//!   reproduces ferx before the fix: a worst error of 2.34 (30 %, ID 9 at `TIME 30`).
//! * `addl_f`, `addl_lag`, `addl_ftime` (#1909): which of an implied dose's attributes
//!   follow the governing record. `ADVAN2` on one dataset of four subjects, X stepping
//!   1 → 2 → 4: the implied dose at 12 falls between records (ID 1), ties with an `EVID=2`
//!   row (2) or with an observation (3), and ID 4 holds X constant. `F1 = 1/X` and
//!   `F1 = 0.5 from TIME 13` follow the governing record; `ALAG1 = 0.25·X` stays the parent
//!   row's. Under the parent-row rule ferx was off by 11.8 on `addl_f` and 4.05 on
//!   `addl_ftime`. Under a governing-record lag it is off by 1.67 on `addl_lag` (13 %, ID 1
//!   at `TIME 14`). Analytic and ODE twins, NONMEM 7.5.1, `.tab` at 17 significant figures.
//! * `addl_lagt`, `addl_dx`, `addl_dt` (#1909): the `TIME` each attribute sees.
//!   `ALAG1 = 0.25 + 0.05·TIME` on the same data reads the parent row's `TIME`; reading
//!   the governing `TIME` instead is 3.78 off. On `addl_inf`, an implied `RATE=-2` dose
//!   takes `D1 = X` and `D1 = 0.5 + 0.1·TIME` from the governing record, where the
//!   parent-row rule is 6.50 and 6.97 off. Its `RATE=-1` subjects are left out (see the
//!   test).
//!
//! Tier 2: one `predict()` per anchor.

use ferx_core::{parse_model_file, predict, read_nonmem_csv};
use std::path::Path;

/// `(ID, TIME, PRED)` of every scored observation in `nonmem_anchor/results/<name>.tab`.
fn nonmem_preds(name: &str) -> Vec<(String, f64, f64)> {
    let path = format!("nonmem_anchor/results/{name}.tab");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text
        .lines()
        .skip_while(|l| !l.trim_start().starts_with("ID"));
    let header: Vec<&str> = lines
        .next()
        .expect(".tab header")
        .split_whitespace()
        .collect();
    let col = |name: &str| header.iter().position(|h| *h == name);
    let (id, time, evid, pred) = (
        col("ID").unwrap(),
        col("TIME").unwrap(),
        col("EVID").unwrap(),
        col("PRED").unwrap(),
    );
    let mdv = col("MDV");
    lines
        .map(|l| {
            l.split_whitespace()
                .map(|v| {
                    v.parse::<f64>()
                        .unwrap_or_else(|e| panic!("{path}: {v}: {e}"))
                })
                .collect::<Vec<f64>>()
        })
        .filter(|r| r[evid] == 0.0 && mdv.is_none_or(|m| r[m] == 0.0))
        .map(|r| (format!("{}", r[id] as u32), r[time], r[pred]))
        .collect()
}

/// Runs the `_fit.ferx` twin's `predict()` on `csv` and returns the worst `|PRED − NONMEM|`
/// against `results/<anchor>.tab`; `twin` names the `nonmem_anchor/<twin>_fit.ferx` model
/// (the anchor's own name but for a second engine's twin of one `.ctl`). `pk_only` is each
/// subject's expected pk-only record times: the rows must reach the engine, or the anchor
/// would pass on a reader that drops them only where `PRED` cannot see it.
fn worst_pred_error(anchor: &str, twin: &str, csv: &str, pk_only: &[(&str, &[f64])]) -> f64 {
    worst_pred_error_on(anchor, twin, csv, pk_only, None)
}

/// [`worst_pred_error`] restricted to the subjects in `only` when it is `Some`.
fn worst_pred_error_on(
    anchor: &str,
    twin: &str,
    csv: &str,
    pk_only: &[(&str, &[f64])],
    only: Option<&[&str]>,
) -> f64 {
    let keep = |id: &str| only.is_none_or(|ids| ids.contains(&id));
    let model_path = format!("nonmem_anchor/{twin}_fit.ferx");
    let model = parse_model_file(Path::new(&model_path)).unwrap_or_else(|e| panic!("{e}"));
    let csv = format!("nonmem_anchor/{csv}.csv");
    let pop = read_nonmem_csv(Path::new(&csv), None, None).unwrap_or_else(|e| panic!("{e}"));
    for &(id, want) in pk_only {
        let s = pop.subjects.iter().find(|s| s.id == id).expect("subject");
        let got = &s.pk_only_times;
        assert!(
            got.len() == want.len() && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-9),
            "{anchor} subject {id}: pk-only records at {got:?}, the file has them at {want:?}"
        );
    }
    let mut ferx = predict(&model, &pop, &model.default_params).expect("predict");
    ferx.retain(|r| keep(&r.id));
    let mut nm = nonmem_preds(anchor);
    nm.retain(|(id, _, _)| keep(id));
    assert!(!nm.is_empty(), "{anchor}: no observations kept");
    assert_eq!(ferx.len(), nm.len(), "{anchor}: observation count");
    let mut worst = 0.0_f64;
    for (r, (id, t, want)) in ferx.iter().zip(&nm) {
        assert!(
            r.id == *id && (r.time - t).abs() < 1e-9,
            "{anchor}: ferx row ({}, {}) is NONMEM's ({id}, {t})",
            r.id,
            r.time
        );
        // Finite first: `f64::max` would discard a NaN.
        assert!(
            r.pred.is_finite(),
            "{anchor} ID {id} TIME {t}: PRED {}",
            r.pred
        );
        let err = (r.pred - want).abs();
        assert!(
            err < PRED_TOL,
            "{anchor} ID {id} TIME {t}: ferx PRED {} vs NONMEM {want} (|error| {err:.3e})",
            r.pred
        );
        worst = worst.max(err);
    }
    println!(
        "{anchor}: worst |PRED − NONMEM| {worst:.3e} over {} observations",
        nm.len()
    );
    worst
}

#[test]
fn a_leading_evid2_or_mdv1_record_starts_the_clock_matches_nonmem() {
    worst_pred_error(
        "evid2_clock",
        "evid2_clock",
        "evid2_clock",
        &[
            ("1", &[0.0]),
            ("2", &[0.0]),
            ("3", &[0.0]),
            ("4", &[0.0]),
            ("5", &[0.0]),
        ],
    );
}

#[test]
fn co_timed_records_follow_file_order_matches_nonmem() {
    // Case 3's row is 1e-6 after the tie, so it is not moved; cases 1 and 4 move the
    // observation one ULP below 4, and case 6 the EVID=2 row.
    let at4: &[f64] = &[4.0];
    let pk_only: &[(&str, &[f64])] = &[
        ("1", at4),
        ("2", at4),
        ("3", &[4.000001]),
        ("4", at4),
        ("5", at4),
        ("6", at4),
        ("7", &[]),
    ];
    worst_pred_error("tied_evid2", "tied_evid2", "tied_evid2", pk_only);
    worst_pred_error(
        "tied_evid2_advan1",
        "tied_evid2_advan1",
        "tied_evid2",
        pk_only,
    );
}

#[test]
fn an_addl_implied_dose_runs_under_the_record_that_ends_its_interval_matches_nonmem() {
    // ID 6 lists its EVID=2 row before the co-timed dose, so #1810 moves it one ULP below 12.
    let at12: &[f64] = &[12.0];
    let pk_only: &[(&str, &[f64])] = &[
        ("1", at12),
        ("2", at12),
        ("3", &[]),
        ("4", &[]),
        ("5", &[]),
        ("6", at12),
        ("7", at12),
        ("8", at12),
        ("9", &[24.0]),
        ("10", at12),
        ("11", at12),
        ("12", at12),
        ("13", &[]),
        ("14", &[]),
    ];
    worst_pred_error("addl_tie", "addl_tie", "addl_tie", pk_only);
    worst_pred_error("addl_tie", "addl_tie_ode", "addl_tie", pk_only);
}

#[test]
fn an_addl_implied_dose_takes_f_and_time_from_the_governing_record_and_lag_from_its_parent_matches_nonmem(
) {
    let pk_only: &[(&str, &[f64])] = &[("1", &[]), ("2", &[12.0]), ("3", &[]), ("4", &[])];
    for anchor in ["addl_f", "addl_lag", "addl_ftime", "addl_lagt"] {
        worst_pred_error(anchor, anchor, "addl_dose", pk_only);
        worst_pred_error(anchor, &format!("{anchor}_ode"), "addl_dose", pk_only);
    }
}

#[test]
fn an_addl_implied_infusion_takes_its_modelled_duration_from_the_governing_record_matches_nonmem() {
    // IDs 3, 4 and 6 code RATE=-1. NONMEM re-reads R1 on every advance, which ferx does
    // not do for any modelled-rate infusion, implied or not. That defect predates #1909
    // and is tracked on its own, so only the D1 subjects are compared here.
    let at12_24: &[f64] = &[12.0, 24.0];
    let pk_only: &[(&str, &[f64])] = &[("1", &[]), ("2", at12_24), ("5", &[])];
    let d1: &[&str] = &["1", "2", "5"];
    for anchor in ["addl_dx", "addl_dt"] {
        worst_pred_error_on(anchor, anchor, "addl_inf", pk_only, Some(d1));
        worst_pred_error_on(
            anchor,
            &format!("{anchor}_ode"),
            "addl_inf",
            pk_only,
            Some(d1),
        );
    }
}

// Measured on Linux x86_64 (the reference platform) at 2f826b90: worst |PRED − NONMEM| 2.79e-9
// (evid2_clock; 4.7e-11 on tied_evid2_advan1, 0 on tied_evid2), so 36× headroom. The `.tab`
// prints 11 significant figures, about 1e-9 at these magnitudes. The #1909 anchors print 17
// and stay under 1.1e-14 (Linux x86_64 at 5998edf2; 3.2e-14 on addl_dt, macOS arm64). The
// smallest regression they catch moves a PRED by 8.3e-2: the governing record taken strictly
// after the dose, on addl_dt.
const PRED_TOL: f64 = 1e-7;
