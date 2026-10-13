//! NONMEM anchor for the occasion an EVID=3/4 reset row re-seeds `init(...)` under when the
//! occasion is **derived** by `iov_occasion = dose | time(...)`.
//!
//! `apply_iov_occasion_rule` rewrote `Subject::occasions` and `dose_occasions` but never
//! `reset_occasions`. With no `iov_column` the label stayed empty and the reset fell back to
//! the record before it, so period 2's re-seed ran under period 1's κ. Over an `iov_column`
//! it kept the dataset's label, which the rule's 0-based renumbering leaves naming no κ
//! group, so the re-seed ran at κ = 0 and the subject routed to FD.
//!
//! # The control streams
//!
//! A two-period crossover, six subjects: a 100 mg bolus at t = 0, and period 2 opens at
//! t = 12 with a reset and a second 100 mg. `KAPPA_B` is in `BASE`, which reaches the
//! prediction only through `A_0(1)`/`init(central)` (at t = 0 and at the re-seed), and in
//! `CL`. TVBASE matches the dose, so the seed is half the post-reset signal, and the reset
//! discards a live residue. `ADVAN1 TRANS2`, `MAXEVAL=0 METHOD=1 INTER`: NONMEM's closed
//! form against ferx's `[odes]` integration, so the two sides share no solver.
//!
//! | arm | dataset | the reset |
//! |---|---|---|
//! | A | `iov_rule_reset.csv`       | one EVID=4 row (reset + dose) |
//! | B | `iov_rule_reset_split.csv` | an EVID=3 row, then an EVID=1 dose row at the same time |
//!
//! NONMEM derives the occasion from `TIME` in `$PK` (`IF (TIME.GE.12) OCCT = 2`) and
//! ignores the data's `OCC`, so it is the reference for the derived rule. The data's `OCC`
//! carries exactly those labels, so the same arms anchor the column fit, the κ-label route
//! that `reset_init_snapshot_nonmem_anchor`'s arm J anchors only through a covariate.
//!
//! # The ferx cells
//!
//! One model (`iov_rule_reset.ferx`, which reads `OCC` through `iov_column`) and five
//! cells per arm: the column fit; `dose` and `time(12)` over the column's labels; and
//! `dose` and `time(12)` with the labels cleared, which is what the reader returns without
//! `iov_column`. Before the fix the derived cells landed on the two wrong states the
//! in-repo `test_iov_occasion_rule_labels_reset_rows_in_fit` measures — far outside the
//! tolerances below.
//!
//! Raw NONMEM output is committed under `nonmem_anchor/results/iov_rule_reset_{A,B}.*`.
//!
//! Tier 2: `maxiter = 0`, one objective evaluation per cell, no convergence loop.

use std::collections::HashMap;
use std::path::PathBuf;

use ferx_core::{fit, prepare_run, FitResult, IovOccasionRule};

fn anchor(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("nonmem_anchor");
    p.push(name);
    p.to_string_lossy().into_owned()
}

fn whitespace_rows(name: &str) -> Vec<Vec<String>> {
    let path = anchor(&format!("results/{name}"));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .map(|l| l.split_whitespace().map(str::to_string).collect())
        .collect()
}

/// `#OBJV` at the evaluated vector: the `OBJ` column of the `.ext`'s `-1000000000` row.
fn nm_ofv(arm: &str) -> f64 {
    let rows = whitespace_rows(&format!("iov_rule_reset_{arm}.ext"));
    let header = rows
        .iter()
        .find(|r| r.first().map(String::as_str) == Some("ITERATION"))
        .expect(".ext header");
    let obj = header.iter().position(|h| h == "OBJ").expect("OBJ column");
    let last = rows
        .iter()
        .find(|r| r.first().map(String::as_str) == Some("-1000000000"))
        .expect(".ext final row");
    last[obj].parse().expect("OBJ value")
}

/// Per subject: NONMEM's POSTHOC `ETA(1)` (ETA_CL), `ETA(2)`, `ETA(3)` (KAPPA_B, occasions 1, 2).
fn nm_phi(arm: &str) -> HashMap<String, [f64; 3]> {
    let rows = whitespace_rows(&format!("iov_rule_reset_{arm}.phi"));
    let header = rows
        .iter()
        .find(|r| r.first().map(String::as_str) == Some("SUBJECT_NO"))
        .expect(".phi header");
    let at = |label: &str| header.iter().position(|h| h == label).expect(label);
    let id_col = at("ID");
    let eta_cols = [at("ETA(1)"), at("ETA(2)"), at("ETA(3)")];
    rows.iter()
        .filter(|r| r.len() == header.len() && r[0] != "SUBJECT_NO")
        .map(|r| {
            let id: f64 = r[id_col].parse().expect("ID");
            (
                format!("{}", id as i64),
                eta_cols.map(|c| r[c].parse().expect("ETA")),
            )
        })
        .collect()
}

/// One ferx cell: `rule` over the column's labels, or with them cleared when `labelled` is
/// false.
fn ferx_fit(data: &str, rule: IovOccasionRule, labelled: bool) -> FitResult {
    let mut run = prepare_run(&anchor("iov_rule_reset.ferx"), Some(&anchor(data)))
        .expect("the anchor model and dataset load");
    if !labelled {
        for s in &mut run.population.subjects {
            s.occasions.clear();
            s.dose_occasions.clear();
            s.reset_occasions.clear();
        }
    }
    let mut opts = run.parsed.fit_options.clone();
    opts.iov_occasion = rule;
    fit(&run.parsed.model, &run.population, &run.init_params, &opts).expect("the cell evaluates")
}

/// `|ΔOFV|` against NONMEM, every cell. Measured on Linux x86_64: 5.747e-8 in all ten cells,
/// so 17× headroom. Without the fix the derived cells sit at 11.00 (over `iov_column`) and
/// 28.05 (no column), seven orders of magnitude out.
const OFV_TOL: f64 = 1e-6;
/// `|Δη̂|` and `|Δκ̂|` against NONMEM's `.phi`, every subject. Measured on Linux x86_64:
/// 4.523e-7 in all ten cells. That is the `.phi`'s own rounding: it prints six significant
/// figures, so ±5e-7 at these magnitudes. Hence 11× headroom. Without the fix: 0.512 and 0.362.
const EBE_TOL: f64 = 5e-6;

fn assert_arm_matches_nonmem(arm: &str, data: &str) {
    let want_ofv = nm_ofv(arm);
    let want_ebe = nm_phi(arm);
    assert_eq!(want_ebe.len(), 6, "arm {arm}: six subjects in the .phi");
    let cells = [
        ("iov_column", IovOccasionRule::Column, true),
        ("dose over iov_column", IovOccasionRule::PerDose, true),
        (
            "time(12) over iov_column",
            IovOccasionRule::TimeWindows(vec![12.0]),
            true,
        ),
        ("dose, no column", IovOccasionRule::PerDose, false),
        (
            "time(12), no column",
            IovOccasionRule::TimeWindows(vec![12.0]),
            false,
        ),
    ];
    for (cell, rule, labelled) in cells {
        let r = ferx_fit(data, rule, labelled);
        assert!(r.ofv.is_finite(), "arm {arm}, {cell}: OFV {}", r.ofv);
        let d_ofv = (r.ofv - want_ofv).abs();
        let mut worst_ebe = 0.0f64;
        for (s, kappas) in r.subjects.iter().zip(&r.ebe_kappas) {
            let want = want_ebe
                .get(&s.id)
                .unwrap_or_else(|| panic!("arm {arm}: no .phi row for subject {}", s.id));
            assert_eq!(
                kappas.len(),
                2,
                "arm {arm}, {cell}: subject {} occasions",
                s.id
            );
            let got = [s.eta[0], kappas[0][0], kappas[1][0]];
            for (g, w) in got.iter().zip(want) {
                assert!(g.is_finite(), "arm {arm}, {cell}: subject {} EBE {g}", s.id);
                worst_ebe = worst_ebe.max((g - w).abs());
            }
        }
        eprintln!(
            "arm {arm}, {cell}: ferx {:.10} NONMEM {want_ofv:.10} |dOFV| {d_ofv:.3e} \
             worst |dEBE| {worst_ebe:.3e}",
            r.ofv
        );
        assert!(
            d_ofv < OFV_TOL,
            "arm {arm}, {cell}: ferx OFV {} vs NONMEM {want_ofv} (|d| {d_ofv:.3e})",
            r.ofv
        );
        assert!(
            worst_ebe < EBE_TOL,
            "arm {arm}, {cell}: worst EBE gap {worst_ebe:.3e} against NONMEM's .phi"
        );
    }
}

#[test]
fn an_evid4_reset_row_reseeds_under_its_derived_occasion() {
    assert_arm_matches_nonmem("A", "iov_rule_reset.csv");
}

#[test]
fn an_evid3_reset_row_reseeds_under_its_derived_occasion() {
    // The arm the defect's κ = 0 cliff needs: the EVID=3 row carries no dose, so a stale
    // label on it is shared by no dose or observation and names no κ group.
    assert_arm_matches_nonmem("B", "iov_rule_reset_split.csv");
}
