//! Tier 2: `stats::cif::aalen_johansen` against R's `survival::survfit` multi-state
//! fit, the Aalen–Johansen estimator (#1887).
//!
//! NONMEM has no nonparametric cumulative-incidence estimator, so survfit is the
//! external reference. `tests/fixtures/cif_survfit/make_reference.R` draws 160
//! subjects in two strata with two competing causes, on whole-unit times, so ties are
//! frequent: 21 of the 32 reference rows carry more than one event and 11 carry an
//! event and a censoring at the same time, which is where the tie convention decides
//! the answer. The strata swap the two causes' rates, so a cause-index mix-up moves
//! every row.

use ferx_core::stats::cif::{aalen_johansen, CifCurve};
use std::collections::BTreeMap;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/cif_survfit");

/// The measured worst error of `surv`, `cif1` and `cif2` against survfit is printed
/// by the test. Both engines compute the same finite sums and products over the same
/// rows, so they differ only in rounding. Measured on macOS arm64 at the commit that
/// added this test: worst |Δ| = 2.220e-16 (plan M1, Python model: 5.6e-16). The bound
/// keeps 4.5× headroom. Mutation-checked: form A with `S(t)` in place of `S(t⁻)`, each cause taking the whole
/// all-cause drop, `1 − KM_k`, a swapped cause index, and perturbing one fixture `cif1`
/// by 1e-14 each fail this test.
const TOL: f64 = 1e-15;

struct Row {
    time: f64,
    n_risk: usize,
    n_event: [usize; 2],
    n_censor: usize,
    surv: f64,
    cif: [f64; 2],
}

fn read_csv(name: &str) -> Vec<Vec<String>> {
    let text = std::fs::read_to_string(format!("{DIR}/{name}")).unwrap();
    text.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
        .collect()
}

/// The data, grouped by the exact stratum label: `(times, causes)`.
fn data_by_stratum() -> BTreeMap<String, (Vec<f64>, Vec<usize>)> {
    let mut by: BTreeMap<String, (Vec<f64>, Vec<usize>)> = BTreeMap::new();
    for r in read_csv("data.csv") {
        let e = by.entry(r[1].clone()).or_default();
        e.0.push(r[2].parse().unwrap());
        e.1.push(r[3].parse().unwrap());
    }
    by
}

fn reference_by_stratum() -> BTreeMap<String, Vec<Row>> {
    let mut by: BTreeMap<String, Vec<Row>> = BTreeMap::new();
    for r in read_csv("reference.csv") {
        let p = |i: usize| r[i].parse::<f64>().unwrap();
        let u = |i: usize| r[i].parse::<usize>().unwrap();
        by.entry(r[0].clone()).or_default().push(Row {
            time: p(1),
            n_risk: u(2),
            n_event: [u(3), u(4)],
            n_censor: u(5),
            surv: p(6),
            cif: [p(7), p(8)],
        });
    }
    by
}

/// Compare one curve against its survfit rows; returns the worst |Δ| over `surv`
/// and both CIFs.
fn compare(label: &str, got: &CifCurve, want: &[Row]) -> f64 {
    let km = &got.all_cause;
    assert_eq!(km.time.len(), want.len(), "stratum {label}: row count");
    assert_eq!(got.n_causes(), 2, "stratum {label}: causes");
    let mut worst = 0.0_f64;
    for (j, w) in want.iter().enumerate() {
        let at = format!("stratum {label}, t = {}", w.time);
        assert_eq!(km.time[j], w.time, "{at}: time");
        assert_eq!(km.n_risk[j], w.n_risk, "{at}: n.risk");
        assert_eq!(km.n_censor[j], w.n_censor, "{at}: n.censor");
        for k in 0..2 {
            assert_eq!(got.n_event[k][j], w.n_event[k], "{at}: n.event of cause {}", k + 1);
        }
        let mut gaps = vec![("surv", km.surv[j], w.surv)];
        gaps.push(("cif1", got.cif[0][j], w.cif[0]));
        gaps.push(("cif2", got.cif[1][j], w.cif[1]));
        for (what, g, r) in gaps {
            let d = (g - r).abs();
            assert!(d.is_finite(), "{at}: {what} non-finite ({g} vs {r})");
            assert!(d <= TOL, "{at}: {what} {g} vs survfit {r} (|Δ| = {d:e})");
            worst = worst.max(d);
        }
    }
    worst
}

/// T5. Every row of every stratum equals survfit's: the integer columns exactly,
/// `surv`, `cif1` and `cif2` within the measured `TOL`. Both strata must be present,
/// so merging them changes every row count.
#[test]
fn aalen_johansen_matches_survfit_multistate_per_stratum() {
    let data = data_by_stratum();
    let reference = reference_by_stratum();
    assert_eq!(data.keys().collect::<Vec<_>>(), ["A", "B"], "two strata in the data");
    assert_eq!(reference.keys().collect::<Vec<_>>(), ["A", "B"], "two strata in the reference");
    let mut worst = 0.0_f64;
    for (label, (times, causes)) in &data {
        let curve = aalen_johansen(times, causes, 2).unwrap();
        let w = compare(label, &curve, &reference[label]);
        assert!(w.is_finite());
        worst = worst.max(w);
    }
    // The fixture must exercise ties and both causes, or it cannot see the convention.
    let rows = || reference.values().flatten();
    let tied = rows().filter(|r| r.n_event[0] + r.n_event[1] > 1).count();
    let both = rows().filter(|r| r.n_event[0] > 0 && r.n_event[1] > 0).count();
    let mixed = rows().filter(|r| r.n_event[0] + r.n_event[1] > 0 && r.n_censor > 0).count();
    assert!(
        tied >= 10 && both >= 5 && mixed >= 9,
        "fixture lost its ties: {tied} tied, {both} with both causes, {mixed} mixed"
    );
    eprintln!("aalen_johansen vs survfit: worst |Δ| = {worst:.3e}");
}
