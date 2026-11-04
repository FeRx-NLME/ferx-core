//! mrgsolve cross-check for **per-subject dose expressions** (#944, epic #391): one
//! `[adaptive_dosing]` block whose `start_dose` and `dose_bounds` are expressions over
//! the WT covariate, evaluated once per subject at the first decision.
//!
//! NONMEM has no native feedback dosing, so the external anchor for this feature family is
//! **mrgsolve**. The kit lives in `tests/reference/vanco_mgkg_mrgsolve/`:
//!
//!   - `vanco_mgkg_mrgsolve.R` runs a 1-cpt IV vancomycin model (CL = 6·WT/70 L/h,
//!     V = 0.7·WT L, 1-h infusions) over a q12 grid in an R loop that mirrors the
//!     controller: start at 15·WT mg, clamp every dose to `[0, min(30·WT, 4000)]`, and
//!     step ±25% on the second consecutive breach of a 15–25 mg/L trough band.
//!   - `expected.md` is the frozen per-decision log: trough, the rule that acted, the dose
//!     issued, and each trough's distance to the nearer threshold.
//!
//! The ferx side is `examples/adaptive_vanco_mgkg.ferx`, run at a supplied η = 0
//! (`common::typical_value_etas`): the model's `omega ~ 1e-10` would otherwise draw an η
//! of order 1e-5 on CL, a one-sided trough drift against the typical-value mrgsolve run.
//!
//! Non-degeneracy: two weights, so the two subjects' ledgers differ from the first dose
//! on; the `min` binds for the 150 kg subject only (4000 < 30·150) while the 50 kg subject
//! is capped at 30·WT, so the ledgers part again at t = 60 — a plan evaluated once per run
//! or with one side of the `min` cannot match both. Multi-dose with drug on board at every
//! later dose. Every trough after t = 0 is ≥ 1.37 mg/L from a threshold, so a
//! cross-solver difference cannot flip a decision.

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{read_nonmem_csv, AdaptiveSimulationResult, Population};
use std::path::Path;

mod common;

const KIT: &str = "tests/reference/vanco_mgkg_mrgsolve";
const EXAMPLE: &str = "examples/adaptive_vanco_mgkg.ferx";

/// One row of `expected.md`: `(time, trough, fired, amt)`.
type RefRow = (f64, f64, String, f64);

fn reference() -> Vec<Vec<RefRow>> {
    let md = std::fs::read_to_string(format!("{KIT}/expected.md")).expect("expected.md");
    let mut subjects: Vec<Vec<RefRow>> = Vec::new();
    for line in md.lines() {
        if line.starts_with("## Subject") {
            subjects.push(Vec::new());
            continue;
        }
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        // `| decision | time | trough | fired | amt | margin |`
        if cells.len() >= 7 && cells[1].parse::<usize>().is_ok() {
            subjects.last_mut().expect("a subject header first").push((
                cells[2].parse().expect("time"),
                cells[3].parse().expect("trough"),
                cells[4].to_string(),
                cells[5].parse().expect("amt"),
            ));
        }
    }
    subjects
}

fn kit_population() -> Population {
    read_nonmem_csv(
        Path::new(&format!("{KIT}/vanco_mgkg_subjects.csv")),
        None,
        None,
    )
    .expect("subjects load")
}

fn run() -> AdaptiveSimulationResult {
    let parsed = parse_full_model_file(Path::new(EXAMPLE)).expect("example parses");
    let spec = parsed.adaptive_dosing.as_ref().expect("block attached");
    common::simulate_from_spec_at_typical_value(&parsed.model, &kit_population(), spec)
}

/// Cross-solver trough band (mg/L), ferx (RK45 at the model's default tolerances) against
/// mrgsolve (LSODA at 1e-10) at η = 0. The worst gap over all 26 decisions is printed by
/// the test: 1.435e-9 mg/L on macOS arm64, so 1e-7 is ~70× headroom, and `expected.md`'s 9
/// decimals floor the comparison at 5e-10. A dosing defect worth naming moves a trough
/// by far more (one subject dosed at the other's weight is a factor of 3), and the
/// smallest decision margin is 1.37 mg/L.
const TROUGH_TOL: f64 = 1e-7;

#[test]
fn vanco_mgkg_matches_mrgsolve() {
    let res = run();
    let reference = reference();
    assert_eq!(reference.len(), 2, "two subjects in the kit");
    let mut worst: f64 = 0.0;
    for (id, want) in ["1", "2"].iter().zip(&reference) {
        let got: Vec<_> = res.decisions.iter().filter(|d| d.subject == *id).collect();
        assert_eq!(got.len(), want.len(), "subject {id}: one decision per row");
        let ledger: Vec<_> = res.ledger.iter().filter(|e| e.subject == *id).collect();
        assert_eq!(
            ledger.len(),
            want.len(),
            "subject {id}: a dose at every decision"
        );
        for ((d, e), (t, trough, fired, amt)) in got.iter().zip(&ledger).zip(want) {
            assert_eq!((d.time, e.time), (*t, *t), "subject {id}");
            let sig = d.observed_signals[0].value;
            assert!(sig.is_finite(), "subject {id} t={t}: trough {sig}");
            let gap = (sig - trough).abs();
            worst = worst.max(gap);
            assert!(
                gap < TROUGH_TOL,
                "subject {id} t={t}: ferx trough {sig:.9} vs mrgsolve {trough:.9}"
            );
            // Every amount is an exact binary fraction (15·WT, ×1.25, or a cap), so the
            // engines agree on the bits.
            assert_eq!(e.amt, *amt, "subject {id} t={t}: dose");
            assert_eq!(
                e.rule_fired.contains("increase"),
                fired == "increase",
                "subject {id} t={t}: rule_fired {:?} vs {fired:?}",
                e.rule_fired
            );
        }
    }
    eprintln!("worst ferx-vs-mrgsolve trough gap: {worst:.3e} mg/L");
}

/// The scenario the anchor relies on, read off the frozen reference so a regenerated kit
/// that lost a case fails here by name: the start doses are 15·WT, subject 1 is capped at
/// 30·WT, subject 2 at 4000 (the other side of the `min`), and the two are dosed
/// differently from the first decision on.
#[test]
fn vanco_mgkg_kit_is_non_degenerate() {
    let r = reference();
    let (s1, s2) = (&r[0], &r[1]);
    assert_eq!(
        (s1[0].3, s2[0].3),
        (15.0 * 50.0, 15.0 * 150.0),
        "start doses"
    );
    let cap = |s: &Vec<RefRow>| s.iter().map(|row| row.3).fold(0.0, f64::max);
    assert_eq!(cap(s1), 30.0 * 50.0, "subject 1 capped at 30 mg/kg");
    assert_eq!(cap(s2), 4000.0, "subject 2 capped at 4000 mg");
    assert!(4000.0 < 30.0 * 150.0, "the min binds for subject 2 only");
    // An increase absorbed at the cap: the rule fires and the dose stays.
    assert!(s1
        .windows(2)
        .any(|w| w[1].2 == "increase" && w[1].3 == w[0].3));
}
