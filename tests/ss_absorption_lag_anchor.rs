//! `SS=1` doses into an **absorption input** — a `zero_order()` window or a first-order
//! kernel / depot — with and without a lag (#2786).
//!
//! Two combinations used to be refused (`E_ABSORPTION_SS_ZERO_ORDER`, `E_ABSORPTION_SS_LAG`):
//! an `SS=1` dose into a `zero_order()` window, and an `SS=1` dose into an absorption-fed
//! compartment carrying a compartment `lagtime`/`ALAG` or a per-route `lag=`.
//!
//! Reference: `nonmem_anchor/ss_absorption_lag_ref_{zo,fo}.csv`, an exact pulse-train
//! superposition outside ferx (`ss_absorption_lag_ref.py`: modal matrix exponential, every
//! virtual pulse summed until it underflows, exact piecewise propagation after the record; the
//! record's parameters, `WT` included, govern the steady state, each later interval runs under
//! the record that ends it). NONMEM: `ss_zero_order_lag.ctl` (`ADVAN1`, `RATE=-2`, `D1 = DR`,
//! `ALAG1 = LG`) and `ss_first_order_lag.ctl` (`ADVAN2`, `ALAG1 = LG` on the depot), both
//! `MAXEVAL=0`, NONMEM 7.6.0; tables in `nonmem_anchor/results/ss_{zero,first}_order_lag.tab`.
//!
//! Dataset (`ss_absorption_lag_gen.py`), `II = 8`: IDs 1-9 zero-order (duration below, at and
//! above `II`; lag 0, below `II`, lag + duration past `II`, `= II`, `> II`; a `WT` step on IDs
//! 8-9), IDs 11-17 first-order (bolus at lag 0 / 2 / 5 / `= II` / `> II`, a `WT` step, and an
//! infusion into the depot longer than `II`); every subject takes a non-SS dose at `t = 24`
//! onto residual drug.
//!
//! # `lag > II`: ferx clamps, NONMEM's analytic ADVANs wrap
//!
//! ferx puts a lagged SS dose's train on the record when `lag >= II` (the clamp of
//! `dosing::ss_seed_phase`, `docs/model-file/lagtime.qmd`, verified against NONMEM 7.6.0 `ADVAN13`
//! in `ss_lag_ge_ii`). NONMEM's analytic `ADVAN1` (ID 7) and `ADVAN2` (ID 15) do something else:
//! from `t = LG - II` on they follow the **unclamped** periodic train (arrivals `t0 + LG - j*II`),
//! agreeing with `ss_absorption_lag_ref_*_wrap.csv` to rounding, and before it their PRED is not
//! a concentration (ID 15: `-335` at `t = 0`; ID 7: `34.7` where the train says `47.5`). So IDs
//! 7 and 15 are compared with the clamped reference, and with NONMEM only through the wrapped
//! one — pinning that the divergence is exactly the convention and nothing else — plus a
//! straddle assertion so the exclusion cannot silently widen. See also
//! `nonmem_anchor/ss_lagtime_edge*`, which records the other NONMEM divergence for `lag >= II`.

use std::collections::HashMap;
use std::path::PathBuf;

fn anchor(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("nonmem_anchor");
    p.push(name);
    p
}

type Table = HashMap<(String, u64), f64>;

/// A `ID,TIME,PRED` reference table, keyed like [`nonmem`].
fn reference(file: &str) -> Table {
    let text = std::fs::read_to_string(anchor(file)).expect("the reference table is committed");
    text.lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let t: f64 = f[1].parse().unwrap();
            ((f[0].to_string(), t.to_bits()), f[2].parse().unwrap())
        })
        .collect()
}

/// NONMEM's `PRED` column from `results/<file>` (header on line 2, whitespace separated),
/// keyed by `(ID, TIME)`. The dose rows are in the table too; only observation times are looked up.
fn nonmem(file: &str) -> Table {
    let text = std::fs::read_to_string(anchor(&format!("results/{file}")))
        .expect("the NONMEM table is committed");
    let mut lines = text.lines().skip(1);
    let head: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
    let col = |name: &str| head.iter().position(|h| *h == name).expect("column");
    let (id, time, pred) = (col("ID"), col("TIME"), col("PRED"));
    lines
        .filter(|l| !l.trim().is_empty() && !l.contains("TABLE"))
        .map(|l| {
            let f: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
            (((f[id] as u64).to_string(), f[time].to_bits()), f[pred])
        })
        .collect()
}

/// `predict()` of `model` over the shared dataset: `(ID, TIME, PRED)` per observation.
///
/// The dataset codes the zero-order stream's doses `RATE = -2` (NONMEM's `D1`); a ferx model
/// whose `zero_order(dur=DUR)` carries the window takes them as plain boluses. The first-order
/// stream observes `CMT 2` (the central of `ADVAN2`); the kernel twin has the central only,
/// as its `CMT 1`.
fn preds(
    model: &str,
    ids: &[&str],
    zero_order_bolus: bool,
    observe_cmt1: bool,
) -> Vec<(String, f64, f64)> {
    let parsed = ferx_core::parse_full_model_file(&anchor(model)).expect("model parses");
    let (mut population, _) = ferx_core::read_nonmem_csv_with_covariates(
        &anchor("ss_absorption_lag.csv"),
        parsed.covariate_decls.as_deref().unwrap_or(&[]),
        &[],
        None,
    )
    .expect("dataset loads");
    // Each twin models one stream: the other's subjects would read its columns as zeros.
    population.subjects.retain(|s| ids.contains(&s.id.as_str()));
    assert_eq!(
        population.subjects.len(),
        ids.len(),
        "{model}: a subject is missing"
    );
    for s in &mut population.subjects {
        if zero_order_bolus {
            for d in &mut s.doses {
                if !matches!(d.rate_mode, ferx_core::RateMode::Fixed) {
                    d.rate_mode = ferx_core::RateMode::Fixed;
                    d.infusion_def = ferx_core::InfusionDef::RateDefined;
                    d.rate = 0.0;
                    d.duration = 0.0;
                }
            }
        }
        if observe_cmt1 {
            s.obs_cmts.iter_mut().for_each(|c| *c = 1);
        }
    }
    let out = ferx_core::predict(&parsed.model, &population, &parsed.model.default_params)
        .expect("predict");
    out.iter().map(|p| (p.id.clone(), p.time, p.pred)).collect()
}

/// Worst relative error of `got` against `want` over the rows of `ids` that pass `keep`;
/// panics on a missing or non-finite row, and on an empty comparison.
fn worst(
    label: &str,
    got: &[(String, f64, f64)],
    want: &Table,
    ids: &[&str],
    keep: impl Fn(&str, f64) -> bool,
) -> f64 {
    let mut worst = 0.0_f64;
    let mut n = 0;
    for (id, t, p) in got
        .iter()
        .filter(|(id, t, _)| ids.contains(&id.as_str()) && keep(id, *t))
    {
        let w = want[&(id.clone(), t.to_bits())];
        let rel = (p - w).abs() / w.abs();
        assert!(rel.is_finite(), "{label} ID {id} t={t}: {p} vs {w}");
        worst = worst.max(rel);
        n += 1;
    }
    assert!(n > 0, "{label}: no row of {ids:?} compared");
    eprintln!("{label} {ids:?}: worst relative error {worst:.3e} over {n} rows");
    worst
}

const ZO_ALL: &[&str] = &["1", "2", "3", "4", "5", "6", "7", "8", "9"];
const ZO_NONMEM: &[&str] = &["1", "2", "3", "4", "5", "6", "8", "9"];
const FO_BOLUS: &[&str] = &["11", "12", "13", "14", "15", "16"];
const FO_ALL: &[&str] = &["11", "12", "13", "14", "15", "16", "17"];
const FO_NONMEM: &[&str] = &["11", "12", "13", "14", "16", "17"];
/// Every row.
fn all(_: &str, _: f64) -> bool {
    true
}

/// The zero-order window, both spellings (`zero_order(dur, lag=)` and `ALAG1`), against the
/// exact reference — every ID, the `lag > II` one (7) included (clamp convention).
#[test]
fn zero_order_ss_lag_matches_the_reference_in_both_spellings() {
    let want = reference("ss_absorption_lag_ref_zo.csv");
    for (model, bound) in [
        ("ss_zero_order_lag_fit.ferx", 1e-11),
        ("ss_zero_order_lag_alag_fit.ferx", 1e-11),
    ] {
        let got = preds(model, ZO_ALL, true, false);
        let w = worst(model, &got, &want, ZO_ALL, all);
        assert!(w < bound, "{model}: {w:.3e}");
    }
}

/// The zero-order window against NONMEM's own PRED (IDs 1-9 but 7, see the module docs).
#[test]
fn zero_order_ss_lag_matches_nonmem() {
    let want = nonmem("ss_zero_order_lag.tab");
    for model in [
        "ss_zero_order_lag_fit.ferx",
        "ss_zero_order_lag_alag_fit.ferx",
    ] {
        let got = preds(model, ZO_ALL, true, false);
        let w = worst(model, &got, &want, ZO_NONMEM, all);
        // Measured on Linux at ode_reltol = 1e-12: 6.0e-13 (both spellings), ~17x headroom.
        assert!(w < 1e-11, "{model} vs NONMEM: {w:.3e}");
    }
}

/// ID 7 (`LG = 9 > II = 8`): NONMEM follows the unclamped train from `t = LG - II = 1`, ferx the
/// clamp. Pins the divergence as exactly the convention, with the straddle on both sides.
#[test]
fn zero_order_ss_lag_beyond_ii_is_the_documented_convention_split() {
    let nm = nonmem("ss_zero_order_lag.tab");
    let wrap = reference("ss_absorption_lag_ref_zo_wrap.csv");
    // NONMEM = the wrapped train from t = 1 (measured 4.7e-16).
    let nm_rows: Vec<(String, f64, f64)> = nm
        .iter()
        .filter(|((id, tb), _)| id == "7" && f64::from_bits(*tb) >= 1.0)
        .filter(|(k, _)| wrap.contains_key(k))
        .map(|((id, tb), p)| (id.clone(), f64::from_bits(*tb), *p))
        .collect();
    let w = worst("NONMEM vs wrapped train", &nm_rows, &wrap, &["7"], all);
    assert!(w < 1e-13, "NONMEM ID 7 vs the wrapped train: {w:.3e}");
    // ferx (clamp) is a different train: measured 0.30 from t = 1 on, and NONMEM's own pre-t=1
    // PRED (34.66 at t = 0.5) is neither train's (wrapped 47.48, clamped 55.15).
    let got = preds("ss_zero_order_lag_fit.ferx", ZO_ALL, true, false);
    let split = worst("ferx vs NONMEM", &got, &nm, &["7"], |_, t| t >= 1.0);
    assert!(
        split > 0.05,
        "ID 7 no longer differs from NONMEM: {split:.3e}"
    );
}

/// The first-order kernel with a per-route lag (`first_order(ka, lag=)`), the bolus IDs.
#[test]
fn first_order_kernel_ss_lag_matches_the_reference() {
    let want = reference("ss_absorption_lag_ref_fo.csv");
    let model = "ss_first_order_lag_fit.ferx";
    let got = preds(model, FO_BOLUS, false, true);
    // Measured on Linux: 6.8e-11 (the kernel twin; its periodic sum truncates at 1e-10 relative).
    let w = worst(model, &got, &want, FO_BOLUS, all);
    assert!(w < 1e-9, "{model}: {w:.3e}");
}

/// The explicit depot with `ALAG1` on it, every ID including the SS infusion longer than `II`.
#[test]
fn first_order_depot_ss_lag_matches_the_reference() {
    let want = reference("ss_absorption_lag_ref_fo.csv");
    let model = "ss_first_order_lag_alag_fit.ferx";
    let got = preds(model, FO_ALL, false, false);
    // Measured on Linux: 3.0e-13.
    let w = worst(model, &got, &want, FO_ALL, all);
    assert!(w < 1e-10, "{model}: {w:.3e}");
}

/// Both first-order spellings against NONMEM's PRED (IDs 11-17 but 15, see the module docs;
/// the kernel twin has no infusion ID).
#[test]
fn first_order_ss_lag_matches_nonmem() {
    let want = nonmem("ss_first_order_lag.tab");
    let kernel = preds("ss_first_order_lag_fit.ferx", FO_BOLUS, false, true);
    let w = worst(
        "kernel vs NONMEM",
        &kernel,
        &want,
        &["11", "12", "13", "14", "16"],
        all,
    );
    // Measured 6.8e-11 (kernel) and 3.0e-13 (depot); NONMEM vs the reference 4e-16.
    assert!(w < 1e-9, "kernel vs NONMEM: {w:.3e}");
    let depot = preds("ss_first_order_lag_alag_fit.ferx", FO_ALL, false, false);
    let w = worst("depot vs NONMEM", &depot, &want, FO_NONMEM, all);
    assert!(w < 1e-10, "depot vs NONMEM: {w:.3e}");
}

/// ID 15 (`LG = 11 > II = 8`): NONMEM's PRED is negative on `0 <= t < LG - II = 3` and, from
/// there on, the unclamped train; ferx follows the clamp throughout and stays physical.
#[test]
fn first_order_ss_lag_beyond_ii_is_the_documented_convention_split() {
    let nm = nonmem("ss_first_order_lag.tab");
    let wrap = reference("ss_absorption_lag_ref_fo_wrap.csv");
    let nm_rows: Vec<(String, f64, f64)> = nm
        .iter()
        .filter(|((id, tb), _)| id == "15" && f64::from_bits(*tb) >= 3.0)
        .filter(|(k, _)| wrap.contains_key(k))
        .map(|((id, tb), p)| (id.clone(), f64::from_bits(*tb), *p))
        .collect();
    let w = worst("NONMEM vs wrapped train", &nm_rows, &wrap, &["15"], all);
    assert!(w < 1e-13, "NONMEM ID 15 vs the wrapped train: {w:.3e}");
    // The straddle: NONMEM is negative before t = 3 (measured -200.9 at t = 0.5), ferx is not.
    let early = |id: &str, t: f64| id == "15" && t < 3.0;
    let nm_early: Vec<f64> = nm
        .iter()
        .filter(|((id, tb), _)| {
            id == "15" && early(id, f64::from_bits(*tb)) && f64::from_bits(*tb) > 0.0
        })
        .map(|(_, p)| *p)
        .collect();
    assert!(
        !nm_early.is_empty() && nm_early.iter().all(|p| *p < 0.0),
        "NONMEM's ID 15 PRED before t = LG - II is no longer negative: {nm_early:?}"
    );
    for model in [
        "ss_first_order_lag_fit.ferx",
        "ss_first_order_lag_alag_fit.ferx",
    ] {
        let kernel = !model.contains("alag");
        let got = preds(model, if kernel { FO_BOLUS } else { FO_ALL }, false, kernel);
        assert!(
            got.iter()
                .filter(|(id, t, _)| early(id, *t))
                .all(|(_, _, p)| *p > 0.0),
            "{model}: ferx's ID 15 PRED before t = LG - II must be a concentration"
        );
        // From t = 3 on the two trains differ (measured 0.34 at t = 7.9...), so the divergence
        // is confined to the convention and not to the early window.
        let split = worst(model, &got, &nm, &["15"], |_, t| t >= 3.0);
        assert!(split > 0.05, "{model}: ID 15 no longer differs from NONMEM");
    }
}
