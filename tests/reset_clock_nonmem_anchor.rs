//! NONMEM measurement for #1978: what `TIME` reads inside `[odes]` and in a non-Gaussian
//! record's predictor after an `EVID=3/4` row restarts `TIME`.
//!
//! The reader shifts a reset period that restarts `TIME` past the previous one
//! (`RESET_SEGMENT_GAP`, `src/io/datareader.rs`), so the engine clock of period 2 runs at
//! `t + offset` while the record was written at `t`. Three surfaces read that engine clock:
//! the `[odes]` right-hand side (the integrator's `t`), a `[binary_model]` linear predictor
//! (`for_each_binary_lp`, `src/categorical/mod.rs`), and an `[event_model] hazard`
//! (integrated in ODE time, and read at an exact event). #1978 asked what NONMEM reads there
//! before changing anything. This file holds the answer.
//!
//! # The measurement
//!
//! Three fixtures, `nonmem_anchor/reset_clock_{ode,cat,tte}.ctl`, each on two-period data
//! whose period 2 restarts `TIME` at 0 on a reset row (`make_reset_clock_fixtures.py`). Each
//! is fitted by NONMEM; then, at NONMEM's optimum and with `MAXEVAL=0`
//! (`make_reset_clock_arms.py`):
//!
//! * `shifted`: the same model on the data moved exactly as ferx's reader moves it. This is
//!   the objective NONMEM reports *if the clock keeps running*.
//! * `control`: the time term zeroed. Nothing reads the clock, so ferx and NONMEM must agree
//!   whatever clock either uses; a gap anywhere else is then the clock and nothing else.
//!
//! The TTE arms run at η = 0 (see `make_reset_clock_arms.py` for the unrelated Laplace-level
//! gap that motivates it), so the written data has an evaluation arm of its own, `eval`.
//!
//! NONMEM 7.5.1 (`nmfe75`, `pmx` image `872eb1f3b96d`), 2026-10-14. ADVAN6 for the ODE
//! fixtures: in that image ADVAN13 aborts with `LSODA CODE -3` whenever an ETA is live.
//! ADVAN14 gives the same objectives to <= 2e-5.
//!
//! # The answer
//!
//! **NONMEM reads `TIME` as written on all three surfaces.** Directly, from its tables:
//! `$DES`'s `T` restarts at the reset row (a compartment integrating `T` holds `TIME²/2` in
//! both periods); the binary log-odds and the TTE hazard in `$ERROR` read the written
//! `TIME`; and the TTE cumulative hazard is the hazard integrated on the restarted clock.
//! ferx on main scores each written dataset at NONMEM's *`shifted`* objective, so the engine
//! clock is a defect, not a documented difference:
//!
//! | fixture | NONMEM written | ferx main | NONMEM shifted | ΔOFV (ferx − NONMEM) |
//! |---|---|---|---|---|
//! | `ode` (`[odes]` reads `TIME`) | −1442.1814208 | 6205.4986098 | 6205.4986098 | 7647.68 |
//! | `cat` (`[binary_model]` reads `TIME`) | 779.5259695 | 914.1576717 | 914.1576675 | 134.63 |
//! | `tte` (`hazard` reads `TIME`, η = 0) | 4032.7939587 | 4271.4439361 | 4271.4439368 | 238.65 |
//!
//! ferx on Linux aarch64; macOS arm64 differs only in the cat cell's 8th significant figure
//! (finite-difference Laplace). With the time term removed (`control`), ferx and NONMEM
//! agree to 4.7e-8 (ode), 6.6e-5 (cat) and 4.5e-7 (tte).
//!
//! The `*_reads_the_engine_clock` tests pin that measurement on main: they fail once the
//! clock is fixed, and are then replaced by the `*_matches_nonmem_written` tests, `#[ignore]`d
//! here, which are the fix's acceptance criterion.
//!
//! Tier 2: `maxiter = 0`, one objective evaluation per cell, no convergence loop.

use std::path::PathBuf;

use ferx_core::{fit, prepare_run};

fn anchor(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("nonmem_anchor");
    p.push(name);
    p.to_string_lossy().into_owned()
}

/// Header + numeric rows of a NONMEM output file under `nonmem_anchor/results/`, skipping
/// the `TABLE NO.` line.
fn table(name: &str) -> (Vec<String>, Vec<Vec<f64>>) {
    let text = std::fs::read_to_string(anchor(&format!("results/{name}")))
        .unwrap_or_else(|e| panic!("results/{name}: {e}"));
    let mut lines = text.lines().filter(|l| !l.starts_with("TABLE NO."));
    let header: Vec<String> = lines
        .next()
        .expect("header")
        .split_whitespace()
        .map(str::to_string)
        .collect();
    let rows = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|x| x.parse().expect("numeric cell"))
                .collect()
        })
        .collect();
    (header, rows)
}

fn col(header: &[String], name: &str) -> usize {
    header
        .iter()
        .position(|h| h == name)
        .unwrap_or_else(|| panic!("column {name} in {header:?}"))
}

/// The objective at NONMEM's final row (`-1000000000`) of `results/<stem>.ext`.
fn nm_ofv(stem: &str) -> f64 {
    let (h, rows) = table(&format!("{stem}.ext"));
    let obj = col(&h, "OBJ");
    rows.iter()
        .find(|r| r[0] == -1_000_000_000.0)
        .map(|r| r[obj])
        .unwrap_or_else(|| panic!("{stem}.ext: no final row"))
}

/// ferx's objective for `nonmem_anchor/<model>.ferx` on `nonmem_anchor/<data>.csv`, evaluated
/// once (`maxiter = 0` in every model file).
fn ferx_ofv(model: &str, data: &str) -> f64 {
    let run = prepare_run(
        &anchor(&format!("{model}.ferx")),
        Some(&anchor(&format!("{data}.csv"))),
    )
    .unwrap_or_else(|e| panic!("{model}: the anchor model and dataset load: {e}"));
    let r = fit(
        &run.parsed.model,
        &run.population,
        &run.init_params,
        &run.parsed.fit_options,
    )
    .unwrap_or_else(|e| panic!("{model}: the evaluation runs: {e}"));
    assert!(r.ofv.is_finite(), "{model}: OFV {}", r.ofv);
    r.ofv
}

/// The ferx-vs-NONMEM bound per fixture. Measured on main, worst over each fixture's
/// control and running-clock cells, Linux aarch64 (`tools/linux-test.sh`) / macOS arm64:
/// ode 4.9e-8 / 4.4e-8, cat 6.6e-5 / 7.8e-5 (the binary endpoint's finite-difference
/// Laplace), tte 6.2e-7 / 6.2e-7. Each bound sits ≥ 10× above both. The clock gaps they
/// separate are 134 to 7648.
const OFV_TOL_ODE: f64 = 5e-7;
const OFV_TOL_CAT: f64 = 1e-3;
const OFV_TOL_TTE: f64 = 5e-6;

/// The clock gap the fixtures were built to produce: any reading of `TIME` that is not the
/// written one moves the objective by far more than this.
const CLOCK_GAP_MIN: f64 = 100.0;

fn report(label: &str, ferx: f64, nm: f64) -> f64 {
    let d = (ferx - nm).abs();
    eprintln!("{label}: ferx {ferx:.10} NONMEM {nm:.10} |dOFV| {d:.3e}");
    d
}

// ---------------------------------------------------------------------------------------
// What NONMEM reads, from its own tables. No ferx involved.
// ---------------------------------------------------------------------------------------

/// `$DES`'s `T` restarts with `TIME` at an `EVID=4` row. Compartment 2 of
/// `reset_clock_ode.ctl` integrates `DADT(2) = T` and is zeroed by the reset, so it holds
/// `TIME²/2` if `T` is the written `TIME`, and `((t + 73)² − 73²)/2` if it keeps running past
/// period 1 (73 = ferx's offset). The `shifted` arm, where the data's `TIME` itself runs on,
/// holds the latter: the probe distinguishes the two.
#[test]
fn nonmem_des_t_restarts_at_the_reset_row() {
    let (h, rows) = table("reset_clock_ode.tab");
    let (id, time, evid, probe) = (
        col(&h, "ID"),
        col(&h, "TIME"),
        col(&h, "EVID"),
        col(&h, "PROBE"),
    );
    let (mut cur, mut in_p2, mut period2_obs) = (f64::NAN, false, 0);
    for r in &rows {
        if r[id] != cur {
            (cur, in_p2) = (r[id], false);
        }
        in_p2 |= r[evid] == 4.0;
        let want = r[time] * r[time] / 2.0;
        assert!(
            (r[probe] - want).abs() <= 1e-8 * want.max(1.0),
            "TIME {} PROBE {} want {want}",
            r[time],
            r[probe]
        );
        if in_p2 && r[evid] == 0.0 {
            period2_obs += 1;
        }
    }
    assert_eq!(period2_obs, 20 * 11, "period-2 observations checked");

    // the running-clock arm, for contrast: the integral of T since the reset row's TIME
    let (h, rows) = table("reset_clock_ode_shifted.tab");
    let (id, time, evid, probe) = (
        col(&h, "ID"),
        col(&h, "TIME"),
        col(&h, "EVID"),
        col(&h, "PROBE"),
    );
    let (mut cur, mut reset) = (f64::NAN, 0.0);
    for r in &rows {
        if r[id] != cur {
            (cur, reset) = (r[id], 0.0);
        }
        if r[evid] == 4.0 {
            reset = r[time];
        }
        let want = (r[time] * r[time] - reset * reset) / 2.0;
        assert!(
            (r[probe] - want).abs() <= 1e-8 * want.max(1.0),
            "shifted: TIME {} PROBE {} want {want}",
            r[time],
            r[probe]
        );
    }
}

/// `$ERROR` reads the written `TIME` at a binary record and at a TTE record after the
/// restart: the tabled log-odds and hazard are the model evaluated at the written `TIME`.
#[test]
fn nonmem_error_reads_the_written_time() {
    // reset_clock_cat.ctl: LP = THETA(1) + THETA(2)*TIME + ETA(1)
    let (th0, tht) = (-1.0125653067216978, 0.028349473864571983);
    let (h, rows) = table("reset_clock_cat.tab");
    let (time, evid, lp, eta) = (
        col(&h, "TIME"),
        col(&h, "EVID"),
        col(&h, "LP"),
        col(&h, "ETA1"),
    );
    let mut n = 0;
    for r in rows.iter().filter(|r| r[evid] == 0.0) {
        let want = th0 + tht * r[time] + r[eta];
        assert!(
            (r[lp] - want).abs() <= 1e-12,
            "cat: TIME {} LP {}",
            r[time],
            r[lp]
        );
        n += 1;
    }
    assert_eq!(n, 50 * 12, "cat: binary records checked");

    // reset_clock_tte_eval.ctl: HAZ = H0*EXP(GAM*TIME)*EXP(BETA*CONC), every TTE record in
    // period 2
    let (h0, beta, gam) = (
        0.019846376447344233,
        0.14887686935395772,
        0.041821685439565881,
    );
    let (h, rows) = table("reset_clock_tte_eval.tab");
    let (time, cmt, evid, ipred, haz) = (
        col(&h, "TIME"),
        col(&h, "CMT"),
        col(&h, "EVID"),
        col(&h, "IPRED"),
        col(&h, "HAZ"),
    );
    let mut n = 0;
    for r in rows.iter().filter(|r| r[cmt] == 2.0 && r[evid] == 0.0) {
        let want = h0 * (gam * r[time]).exp() * (beta * r[ipred]).exp();
        assert!(
            (r[haz] / want - 1.0).abs() <= 1e-12,
            "tte: TIME {} HAZ {} want {want}",
            r[time],
            r[haz]
        );
        n += 1;
    }
    assert_eq!(n, 40, "tte: one TTE record per subject");
}

// ---------------------------------------------------------------------------------------
// [odes] reading TIME
// ---------------------------------------------------------------------------------------

#[test]
fn ode_control_matches_nonmem() {
    let d = report(
        "ode/control",
        ferx_ofv("reset_clock_ode_control", "reset_clock_ode"),
        nm_ofv("reset_clock_ode_control"),
    );
    assert!(d <= OFV_TOL_ODE, "ode/control |dOFV| {d:.3e}");
}

/// #1978 as measured: on main, `[odes]` reads the engine clock, so ferx scores the written
/// data at NONMEM's running-clock objective. Fails once the clock is fixed; replace it with
/// [`ode_matches_nonmem_written`].
#[test]
fn ode_reads_the_engine_clock() {
    let ferx = ferx_ofv("reset_clock_ode", "reset_clock_ode");
    let d = report(
        "ode/written-vs-shifted",
        ferx,
        nm_ofv("reset_clock_ode_shifted"),
    );
    assert!(
        d <= OFV_TOL_ODE,
        "ode: ferx is off NONMEM's shifted arm by {d:.3e}"
    );
    let gap = report("ode/written", ferx, nm_ofv("reset_clock_ode"));
    assert!(gap >= CLOCK_GAP_MIN, "ode: clock gap {gap:.3e}");
}

#[test]
#[ignore = "#1978: [odes] TIME reads the engine clock on reset data; the fix's acceptance test"]
fn ode_matches_nonmem_written() {
    let d = report(
        "ode/written",
        ferx_ofv("reset_clock_ode", "reset_clock_ode"),
        nm_ofv("reset_clock_ode"),
    );
    assert!(d <= OFV_TOL_ODE, "ode |dOFV| {d:.3e}");
}

// ---------------------------------------------------------------------------------------
// [binary_model] and [event_model] reading TIME
// ---------------------------------------------------------------------------------------

#[cfg(feature = "survival")]
#[test]
fn cat_control_matches_nonmem() {
    let d = report(
        "cat/control",
        ferx_ofv("reset_clock_cat_control", "reset_clock_cat"),
        nm_ofv("reset_clock_cat_control"),
    );
    assert!(d <= OFV_TOL_CAT, "cat/control |dOFV| {d:.3e}");
}

/// #1978 as measured for a `[binary_model]` linear predictor. See [`ode_reads_the_engine_clock`].
#[cfg(feature = "survival")]
#[test]
fn cat_reads_the_engine_clock() {
    let ferx = ferx_ofv("reset_clock_cat", "reset_clock_cat");
    let d = report(
        "cat/written-vs-shifted",
        ferx,
        nm_ofv("reset_clock_cat_shifted"),
    );
    assert!(
        d <= OFV_TOL_CAT,
        "cat: ferx is off NONMEM's shifted arm by {d:.3e}"
    );
    let gap = report("cat/written", ferx, nm_ofv("reset_clock_cat"));
    assert!(gap >= CLOCK_GAP_MIN, "cat: clock gap {gap:.3e}");
}

#[cfg(feature = "survival")]
#[test]
#[ignore = "#1978: a [binary_model] TIME reads the engine clock on reset data; the fix's acceptance test"]
fn cat_matches_nonmem_written() {
    let d = report(
        "cat/written",
        ferx_ofv("reset_clock_cat", "reset_clock_cat"),
        nm_ofv("reset_clock_cat"),
    );
    assert!(d <= OFV_TOL_CAT, "cat |dOFV| {d:.3e}");
}

#[cfg(feature = "survival")]
#[test]
fn tte_control_matches_nonmem() {
    let d = report(
        "tte/control",
        ferx_ofv("reset_clock_tte_control", "reset_clock_tte"),
        nm_ofv("reset_clock_tte_control"),
    );
    assert!(d <= OFV_TOL_TTE, "tte/control |dOFV| {d:.3e}");
}

/// #1978 as measured for an `[event_model] hazard`. See [`ode_reads_the_engine_clock`].
#[cfg(feature = "survival")]
#[test]
fn tte_reads_the_engine_clock() {
    let ferx = ferx_ofv("reset_clock_tte_eval", "reset_clock_tte");
    let d = report(
        "tte/written-vs-shifted",
        ferx,
        nm_ofv("reset_clock_tte_shifted"),
    );
    assert!(
        d <= OFV_TOL_TTE,
        "tte: ferx is off NONMEM's shifted arm by {d:.3e}"
    );
    let gap = report("tte/written", ferx, nm_ofv("reset_clock_tte_eval"));
    assert!(gap >= CLOCK_GAP_MIN, "tte: clock gap {gap:.3e}");
}

#[cfg(feature = "survival")]
#[test]
#[ignore = "#1978: an [event_model] hazard's TIME reads the engine clock on reset data; the fix's acceptance test"]
fn tte_matches_nonmem_written() {
    let d = report(
        "tte/written",
        ferx_ofv("reset_clock_tte_eval", "reset_clock_tte"),
        nm_ofv("reset_clock_tte_eval"),
    );
    assert!(d <= OFV_TOL_TTE, "tte |dOFV| {d:.3e}");
}
