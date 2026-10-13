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

/// Runs the `_fit.ferx` twin's `predict()` on `csv` and returns the worst `|PRED − NONMEM|`.
/// `pk_only` is each subject's expected pk-only record times: the rows must reach the engine,
/// or the anchor would pass on a reader that drops them only where `PRED` cannot see it.
fn worst_pred_error(anchor: &str, csv: &str, pk_only: &[(&str, &[f64])]) -> f64 {
    let model_path = format!("nonmem_anchor/{anchor}_fit.ferx");
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
    let ferx = predict(&model, &pop, &model.default_params).expect("predict");
    let nm = nonmem_preds(anchor);
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
    worst_pred_error("tied_evid2", "tied_evid2", pk_only);
    worst_pred_error("tied_evid2_advan1", "tied_evid2", pk_only);
}

// Measured on Linux x86_64 (the reference platform) at 2f826b90: worst |PRED − NONMEM| 2.79e-9
// (evid2_clock; 4.7e-11 on tied_evid2_advan1, 0 on tied_evid2), so 36× headroom. The `.tab`
// prints 11 significant figures, about 1e-9 at these magnitudes. Each regression the anchors
// exist for moves a PRED by 2.5 or more.
const PRED_TOL: f64 = 1e-7;
