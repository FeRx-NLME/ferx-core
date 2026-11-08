//! Tier 2: `stats::km::kaplan_meier` against R's `survival::survfit` (#1886).
//!
//! NONMEM has no Kaplan–Meier estimator, so survfit is the external reference.
//! `tests/fixtures/km_survfit/make_reference.R` draws 120 subjects in two strata
//! with whole-unit times, so ties are frequent: 20 of the 36 reference rows carry
//! more than one event, and several carry an event and a censoring at the same time,
//! which is exactly where the tie convention decides the answer.

use ferx_core::stats::km::{kaplan_meier, KmCurve};
use std::collections::BTreeMap;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/km_survfit");

/// The measured worst error of `surv` / `cumhaz` against survfit is printed by the
/// test. KM is a finite product and Nelson–Aalen a finite sum over the same rows, so
/// the two engines differ only in rounding. Measured on macOS arm64 at the commit
/// that added this test: worst |Δsurv| = 1.110e-16 (1 ULP below 1), worst
/// |Δcumhaz| = 0. The bound is 9× the larger. Mutation-checked: counting censorings
/// before events at a tied time, moving one `n.censor` to the next row of the fixture,
/// and perturbing one `surv` by 1e-14 each fail this test.
const TOL: f64 = 1e-15;

struct Row {
    time: f64,
    n_risk: usize,
    n_event: usize,
    n_censor: usize,
    surv: f64,
    cumhaz: f64,
}

fn read_csv(name: &str) -> Vec<Vec<String>> {
    let text = std::fs::read_to_string(format!("{DIR}/{name}")).unwrap();
    text.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
        .collect()
}

/// The data, grouped by the exact stratum label.
fn data_by_stratum() -> BTreeMap<String, (Vec<f64>, Vec<bool>)> {
    let mut by: BTreeMap<String, (Vec<f64>, Vec<bool>)> = BTreeMap::new();
    for r in read_csv("data.csv") {
        let e = by.entry(r[1].clone()).or_default();
        e.0.push(r[2].parse().unwrap());
        e.1.push(r[3] == "1");
    }
    by
}

fn reference_by_stratum() -> BTreeMap<String, Vec<Row>> {
    let mut by: BTreeMap<String, Vec<Row>> = BTreeMap::new();
    for r in read_csv("reference.csv") {
        by.entry(r[0].clone()).or_default().push(Row {
            time: r[1].parse().unwrap(),
            n_risk: r[2].parse().unwrap(),
            n_event: r[3].parse().unwrap(),
            n_censor: r[4].parse().unwrap(),
            surv: r[5].parse().unwrap(),
            cumhaz: r[6].parse().unwrap(),
        });
    }
    by
}

/// Compare one curve against its survfit rows; returns the worst |Δsurv| and |Δcumhaz|.
fn compare(label: &str, got: &KmCurve, want: &[Row]) -> (f64, f64) {
    assert_eq!(got.time.len(), want.len(), "stratum {label}: row count");
    let (mut ws, mut wh) = (0.0_f64, 0.0_f64);
    for (j, w) in want.iter().enumerate() {
        let at = format!("stratum {label}, t = {}", w.time);
        assert_eq!(got.time[j], w.time, "{at}: time");
        assert_eq!(got.n_risk[j], w.n_risk, "{at}: n.risk");
        assert_eq!(got.n_event[j], w.n_event, "{at}: n.event");
        assert_eq!(got.n_censor[j], w.n_censor, "{at}: n.censor");
        let (ds, dh) = (
            (got.surv[j] - w.surv).abs(),
            (got.cumhaz[j] - w.cumhaz).abs(),
        );
        assert!(ds.is_finite() && dh.is_finite(), "{at}: non-finite");
        assert!(
            ds <= TOL,
            "{at}: surv {} vs survfit {}",
            got.surv[j],
            w.surv
        );
        assert!(
            dh <= TOL,
            "{at}: cumhaz {} vs survfit {}",
            got.cumhaz[j],
            w.cumhaz
        );
        ws = ws.max(ds);
        wh = wh.max(dh);
    }
    (ws, wh)
}

/// T8 + T9. Every row of every stratum equals survfit's: the integer columns
/// exactly, `surv` and `cumhaz` within the measured `TOL`. The strata are grouped by
/// exact label, and both strata must be present, so merging them (or grouping by a
/// prefix) changes every row count.
#[test]
fn kaplan_meier_matches_survfit_per_stratum() {
    let data = data_by_stratum();
    let reference = reference_by_stratum();
    assert_eq!(
        data.keys().collect::<Vec<_>>(),
        ["A", "B"],
        "two strata in the data"
    );
    assert_eq!(
        reference.keys().collect::<Vec<_>>(),
        ["A", "B"],
        "two strata in the reference"
    );
    let (mut ws, mut wh) = (0.0_f64, 0.0_f64);
    for (label, (times, events)) in &data {
        let curve = kaplan_meier(times, events).unwrap();
        let (s, h) = compare(label, &curve, &reference[label]);
        assert!(s.is_finite() && h.is_finite());
        ws = ws.max(s);
        wh = wh.max(h);
    }
    // The fixture must exercise ties, or it cannot see the convention.
    let tied = reference
        .values()
        .flatten()
        .filter(|r| r.n_event > 1)
        .count();
    let mixed = reference
        .values()
        .flatten()
        .filter(|r| r.n_event > 0 && r.n_censor > 0)
        .count();
    assert!(
        tied >= 10 && mixed >= 3,
        "fixture lost its ties: {tied} tied, {mixed} mixed"
    );
    eprintln!("km vs survfit: worst |Δsurv| = {ws:.3e}, worst |Δcumhaz| = {wh:.3e}");
}
