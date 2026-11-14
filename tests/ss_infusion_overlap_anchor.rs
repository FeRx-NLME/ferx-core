//! `SS=1` infusions whose duration exceeds the dosing interval (overlapping pulses).
//!
//! At steady state an infusion of length `T_inf = m·II + r` (`0 < r ≤ II`) has `m + 1` pulses
//! running on `[0, r)` of every cycle and `m` on `[r, II)`, and at the record `m` previous
//! pulses are still infusing — they stop at `T_inf − II`, `T_inf − 2·II`, … after it. Every
//! engine used to drop the whole steady state for such a dose except the analytic
//! superposition: the closed-form event walk, its recorded dual, and the ODE engine and its
//! recorded dual returned a zero state at the record.
//!
//! Reference: `nonmem_anchor/ss_infusion_overlap_ref_{1cpt,2cpt}.csv`, an exact pulse-train
//! superposition outside ferx (`ss_infusion_overlap_ref.py`: modal matrix exponential, every
//! virtual pulse summed until it underflows, exact piecewise propagation after the record).
//! The NONMEM control streams `ss_infusion_overlap_{1cpt,2cpt}.ctl` are committed for the run;
//! NONMEM was not available where this was written.
//!
//! Dataset (`ss_infusion_overlap_gen.py`), `II = 8`, `RATE = 50`: `T_inf = 1.5·II` (ID 1),
//! `2.5·II` (IDs 2, 4, 5), exactly `2·II` (ID 3, the `r = II` edge); `ALAG = 3` (IDs 4, 6,
//! the record-seeded state); `WT` stepping at `t = 10` (IDs 5, 6 — the closed-form *event
//! walk*; the flat IDs take the superposition). Every subject gets a non-SS infusion at
//! `t = 24` landing on residual drug.

use std::path::PathBuf;

fn anchor(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("nonmem_anchor");
    p.push(name);
    p
}

fn reference(m: &str) -> std::collections::HashMap<(String, u64), f64> {
    let text = std::fs::read_to_string(anchor(&format!("ss_infusion_overlap_ref_{m}.csv")))
        .expect("the reference table is committed");
    text.lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let t: f64 = f[1].parse().unwrap();
            ((f[0].to_string(), t.to_bits()), f[2].parse().unwrap())
        })
        .collect()
}

/// Worst relative error of `predict()` against the reference, per ID group.
fn worst(model: &str, m: &str, ids: &[&str]) -> f64 {
    let parsed = ferx_core::parse_full_model_file(&anchor(model)).expect("model parses");
    let (population, _) = ferx_core::read_nonmem_csv_with_covariates(
        &anchor("ss_infusion_overlap.csv"),
        parsed.covariate_decls.as_deref().unwrap_or(&[]),
        &[],
        None,
    )
    .expect("dataset loads");
    let want = reference(m);
    let preds = ferx_core::predict(&parsed.model, &population, &parsed.model.default_params)
        .expect("predict");
    let mut worst = 0.0_f64;
    let mut n = 0;
    for p in preds.iter().filter(|p| ids.contains(&p.id.as_str())) {
        let w = want[&(p.id.clone(), p.time.to_bits())];
        let rel = (p.pred - w).abs() / w.abs();
        assert!(
            rel.is_finite(),
            "{model} ID {} t={}: {} vs {w}",
            p.id,
            p.time,
            p.pred
        );
        worst = worst.max(rel);
        n += 1;
    }
    assert!(n > 0, "{model}: no row of {ids:?} compared");
    eprintln!("{model} {ids:?}: worst relative error {worst:.3e} over {n} rows");
    worst
}

const FLAT: &[&str] = &["1", "2", "3", "4"];
const STEPPED: &[&str] = &["5", "6"];
const ALL: &[&str] = &["1", "2", "3", "4", "5", "6"];

/// Closed form: superposition (flat IDs) and the event walk (`WT`-stepped IDs).
#[test]
fn overlapping_ss_infusion_matches_the_reference_on_the_closed_forms() {
    for m in ["1cpt", "2cpt"] {
        let model = format!("ss_infusion_overlap_{m}_fit.ferx");
        // Measured on Linux: superposition 5.1e-16 / 3.6e-15 (1-/2-cpt), event walk
        // 5.0e-16 / 1.6e-15; before #2786 the event walk read 9.8e-1 (an empty state).
        assert!(worst(&model, m, FLAT) < 1e-13, "{model} superposition");
        assert!(worst(&model, m, STEPPED) < 1e-13, "{model} event walk");
    }
}

/// The ODE engine, every ID.
#[test]
fn overlapping_ss_infusion_matches_the_reference_on_the_ode_engine() {
    for m in ["1cpt", "2cpt"] {
        let model = format!("ss_infusion_overlap_{m}_ode_fit.ferx");
        // Measured on Linux at `ode_reltol = ode_abstol = 1e-12`: 5.8e-13 / 3.0e-13 (1-/2-cpt);
        // before #2786, 9.8e-1 (an empty state).
        assert!(worst(&model, m, ALL) < 1e-11, "{model}");
    }
}
