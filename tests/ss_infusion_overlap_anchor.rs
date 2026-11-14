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
//! NONMEM: `ss_infusion_overlap_{1cpt,2cpt}.ctl` (`ADVAN1 TRANS2` / `ADVAN3 TRANS4`,
//! `MAXEVAL=0`), run with NONMEM 7.6.0 (beta 1, native Windows gfortran build) at `4339dde5`;
//! tables in `nonmem_anchor/results/ss_infusion_overlap_{1cpt,2cpt}.tab`. NONMEM's PRED
//! agrees with the reference to 6.1e-16 (1-cpt) and 1.2e-14 (2-cpt, ADVAN3's own round-off,
//! uniform over IDs), so it places the still-running SS pulses, the lag on an SS infusion and
//! the record-time `WT` exactly as the reference does. ferx is compared with both.
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

type Table = std::collections::HashMap<(String, u64), f64>;

/// NONMEM's `PRED` column from `results/ss_infusion_overlap_{m}.tab`, keyed like [`reference`].
fn nonmem(m: &str) -> Table {
    let text = std::fs::read_to_string(anchor(&format!("results/ss_infusion_overlap_{m}.tab")))
        .expect("the NONMEM table is committed");
    let mut lines = text.lines().skip(1);
    let head: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
    let col = |name: &str| head.iter().position(|h| *h == name).expect("column");
    let (id, time, pred) = (col("ID"), col("TIME"), col("PRED"));
    lines
        .map(|l| {
            let f: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
            (((f[id] as u64).to_string(), f[time].to_bits()), f[pred])
        })
        .collect()
}

/// Worst relative error of `predict()` against the reference, per ID group.
fn worst(model: &str, m: &str, ids: &[&str]) -> f64 {
    worst_against(model, &reference(m), ids)
}

/// Worst relative error of `predict()` against `want`, per ID group.
fn worst_against(model: &str, want: &Table, ids: &[&str]) -> f64 {
    let parsed = ferx_core::parse_full_model_file(&anchor(model)).expect("model parses");
    let (population, _) = ferx_core::read_nonmem_csv_with_covariates(
        &anchor("ss_infusion_overlap.csv"),
        parsed.covariate_decls.as_deref().unwrap_or(&[]),
        &[],
        None,
    )
    .expect("dataset loads");
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
/// Closed forms vs NONMEM: the measured worst is 8.3e-15 (2-cpt), so 1e-13 is ~12x headroom.
const BOUND_CLOSED: f64 = 1e-13;

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

/// NONMEM's own PRED, every engine. The reference above is what NONMEM was checked against;
/// this pins ferx to NONMEM directly, so a change to the reference script cannot move both.
#[test]
fn overlapping_ss_infusion_matches_nonmem() {
    for m in ["1cpt", "2cpt"] {
        let want = nonmem(m);
        let closed = format!("ss_infusion_overlap_{m}_fit.ferx");
        let ode = format!("ss_infusion_overlap_{m}_ode_fit.ferx");
        // Measured on Linux x86_64: superposition 6.1e-16 / 8.3e-15 (1-/2-cpt), event walk
        // 0 / 6.7e-15, ODE 5.8e-13 / 3.1e-13 — the 2-cpt floor is ADVAN3's own round-off.
        let w_flat = worst_against(&closed, &want, FLAT);
        let w_step = worst_against(&closed, &want, STEPPED);
        let w_ode = worst_against(&ode, &want, ALL);
        assert!(
            w_flat < BOUND_CLOSED,
            "{closed} superposition vs NONMEM: {w_flat:.3e}"
        );
        assert!(
            w_step < BOUND_CLOSED,
            "{closed} event walk vs NONMEM: {w_step:.3e}"
        );
        assert!(w_ode < 1e-11, "{ode} vs NONMEM: {w_ode:.3e}");
    }
}
