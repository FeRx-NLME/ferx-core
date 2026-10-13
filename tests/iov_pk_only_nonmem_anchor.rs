//! NONMEM anchors for #1876: on IOV data, a pk-only row (`EVID=2`, or `EVID=0` with `MDV=1`)
//! runs the individual-parameter program under **its own `OCC`**, as NONMEM runs `$PK` on every
//! record. Under the end-of-interval rule that row's κ governs the interval ending at it.
//!
//! Model and data: `warfarin_iov` at its initial estimates, `ADVAN2 TRANS2`, `MAXEVAL=0`,
//! NONMEM 7.6.0 (`nonmem:7.6.0-anchor`), run by @TeunP on PR #1875. The files are theirs,
//! verbatim, except that each control stream's `$DATA data.csv` names its committed CSV, and
//! the `focei_*` streams other than `focei_E` are generated from `foce_*` by their stated rule
//! (`METHOD=1 INTER`); the rule reproduces the posted `focei_E` byte for byte.
//!
//! * **A** is `data/warfarin_iov.csv` unchanged.
//! * **E** adds an `EVID=2` row at TIME 30 / OCC 1 and at TIME 150 / OCC 2 on every subject.
//!   Each row's occasion is its neighbours', so only κ = 0 at the row is wrong here.
//! * **X** adds one `EVID=2` row at TIME 110 / **OCC 2** on every subject, between the OCC-1
//!   observation at 96 and the OCC-1 trough at 120. Its occasion is not the record before it,
//!   so X is the case that tells the row's own label from a neighbour lookup.
//! * **XM** is X with the row written as `EVID=0, MDV=1`. TeunP ran it and reports it equals
//!   X "to every digit" for both methods, but posted no `.ext`; the test therefore compares
//!   XM against X's.
//!
//! The `_k0` streams add `IF(EVID.EQ.2) KAPPA = 0` (ferx before #1876); `X_prev` adds
//! `IF(EVID.EQ.2) KAPPA = ETA(4)`, i.e. the occasion of the record before the row — what a
//! reader that lost `pk_only_occasions` would fall back to. Each is asserted as a straddle,
//! so the failure names the convention ferx fell into, and so the fixture cannot quietly go
//! degenerate. `X_prev` equals A to 4e-11, so only the X anchor sees that fallback.
//!
//! Tier 2: one evaluation per (engine, method, dataset), no convergence loop.

use ferx_core::run_model_with_data;
use ferx_core::types::{FitResult, Population};

/// The final-estimate row's `OBJ` of `nonmem_anchor/results/<name>.ext`.
fn nonmem_obj(name: &str) -> f64 {
    let path = format!("nonmem_anchor/results/{name}.ext");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let rows: Vec<Vec<&str>> = text
        .lines()
        .map(|l| l.split_whitespace().collect())
        .collect();
    let header = rows
        .iter()
        .find(|r| r.first() == Some(&"ITERATION"))
        .expect(".ext header");
    let last = rows
        .iter()
        .find(|r| r.first() == Some(&"-1000000000"))
        .expect(".ext final-estimate row");
    let i = header.iter().position(|h| *h == "OBJ").expect(".ext OBJ");
    let obj: f64 = last[i].parse().expect(".ext OBJ value");
    assert!(obj.is_finite(), "{path}: OBJ {obj}");
    obj
}

fn ferx(engine: &str, method: &str, data: &str) -> (FitResult, Population) {
    let model = format!("nonmem_anchor/iov_pk_only_{engine}_{method}.ferx");
    let csv = format!("nonmem_anchor/iov_pk_only_ferx_{data}.csv");
    run_model_with_data(&model, Some(&csv)).unwrap_or_else(|e| panic!("{model} on {csv}: {e}"))
}

/// The pk-only rows each dataset must carry, per subject: `(time, OCC)`.
fn expected_pk_only(data: &str) -> Vec<(f64, u32)> {
    match data {
        "A" => vec![],
        "E" => vec![(30.0, 1), (150.0, 2)],
        "X" | "XM" => vec![(110.0, 2)],
        _ => unreachable!(),
    }
}

#[test]
fn pk_only_row_runs_under_its_own_occasion_matches_nonmem() {
    let mut failures = Vec::new();
    for method in ["foce", "focei"] {
        let nm = |run: &str| nonmem_obj(&format!("iov_pk_only_{method}_{run}"));
        let (a, e, e_k0) = (nm("A"), nm("E"), nm("E_k0"));
        let (x, x_k0, x_prev) = (nm("X"), nm("X_k0"), nm("X_prev"));
        // The fixture's own straddles: each wrong convention sits far outside the bound.
        for (what, gap) in [
            ("E − E_k0", e - e_k0),
            ("X − X_k0", x - x_k0),
            ("X − X_prev", x - x_prev),
        ] {
            assert!(
                gap.abs() > 1e3 * OFV_TOL,
                "{method}: NONMEM {what} = {gap:.3e} is not well above the bound {OFV_TOL:e}; \
                 the anchor no longer separates the conventions"
            );
        }
        let mut by_engine: Vec<Vec<f64>> = Vec::new();
        for engine in ["an", "ode"] {
            let mut ofvs = Vec::new();
            for (data, want) in [("A", a), ("E", e), ("X", x), ("XM", x)] {
                let (fit, pop) = ferx(engine, method, data);
                let tag = format!("{engine}_{method} on {data}");
                // The rows must reach the engine, labelled: a reader that drops them, or
                // drops their labels, would otherwise pass E (or A) by construction.
                for s in &pop.subjects {
                    let got: Vec<(f64, u32)> = s
                        .pk_only_times
                        .iter()
                        .copied()
                        .zip(s.pk_only_occasions.iter().copied())
                        .collect();
                    assert_eq!(
                        s.pk_only_occasions.len(),
                        s.pk_only_times.len(),
                        "{tag} subject {}: pk_only_occasions is not parallel to pk_only_times",
                        s.id
                    );
                    assert_eq!(
                        got,
                        expected_pk_only(data),
                        "{tag} subject {}: pk-only (time, OCC)",
                        s.id
                    );
                }
                // The dual inner gradient ran, so the analytic provider is what was anchored.
                if let Some(w) = fit
                    .warnings
                    .iter()
                    .find(|w| w.contains("finite-difference inner gradients"))
                {
                    failures.push(format!("{tag}: {w}"));
                }
                let ofv = fit.ofv;
                assert!(ofv.is_finite(), "{tag}: ferx OFV {ofv}");
                let gap = (ofv - want).abs();
                println!("#1876 {tag}: ferx OFV {ofv:.10} vs NONMEM {want:.10} (|gap| {gap:.3e})");
                if !(gap < OFV_TOL) {
                    // Name a wrong convention only when the OFV equals it to the bound. A
                    // change on one side only (the engine's value or the provider's
                    // gradient) moves the EBEs and lands *near* a convention, and on X the
                    // two sit 0.02 apart, so "nearest" would blame the wrong one: measured,
                    // reverting only the value to κ = 0 lands closer to X_prev than X_k0.
                    let wrong: &[(&str, f64)] = match data {
                        "E" => &[("E_k0: the pk-only row ran under κ = 0 (#1876)", e_k0)],
                        "X" | "XM" => &[
                            ("X_k0: the pk-only row ran under κ = 0 (#1876)", x_k0),
                            (
                                "X_prev: the row took the previous record's occasion, so \
                                 its own label (pk_only_occasions) was not used",
                                x_prev,
                            ),
                        ],
                        _ => &[],
                    };
                    let why = wrong
                        .iter()
                        .map(|&(what, v)| (what, (ofv - v).abs()))
                        .find(|&(_, d)| d < OFV_TOL)
                        .map_or_else(
                            || {
                                " — equal to no recorded wrong convention; a change to one \
                                 side only (value or gradient) lands off them, and the \
                                 Tier-1 parity tests name the side"
                                    .to_string()
                            },
                            |(what, d)| format!(" — it equals {what} (|Δ| {d:.3e})"),
                        );
                    failures.push(format!(
                        "{tag}: ferx OFV {ofv:.8} vs NONMEM {want:.8} (|gap| {gap:.3e}){why}"
                    ));
                }
                ofvs.push(ofv);
            }
            by_engine.push(ofvs);
        }
        // Two engines, so not one bit pattern: identical OFVs would mean one model file ran
        // the other's engine, and this test would anchor only one of them.
        let differ = by_engine[0]
            .iter()
            .zip(&by_engine[1])
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count();
        assert!(
            differ > 0,
            "{method}: the analytical and [odes] OFVs are bit-identical — one model file ran \
             the other's engine"
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// Measured on Linux x86_64 (the reference platform) at 2f826b90: worst |OFV − NONMEM| 5.03e-8
// (an_focei on X), so 20× headroom. The nearest wrong convention is 0.788 away (X_k0, FOCEI).
// When ferx is reverted to a wrong convention on both sides, its OFV equals that convention's
// NONMEM OBJ to within 1.85e-7 (macOS, mutations B2/B3), which is inside the bound, so the
// diagnosis names it.
const OFV_TOL: f64 = 1e-6;
