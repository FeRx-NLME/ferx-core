//! NONMEM anchor for a **lagged zero-order absorption route** reaching the *states*
//! engines — issue [#1171].
//!
//! ferx: `d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central`, `y = central/V`.
//! NONMEM (`nonmem_anchor/lagged_zo.ctl`, 7.6.0, `ADVAN13 TOL=9`, `MAXEVAL=0 POSTHOC`,
//! every `$THETA` `FIX`): a `$DES` carrying elimination only, with the input supplied
//! as NONMEM's own **duration-modeled dose** — `RATE=-2` in the data plus `D1 = DUR`
//! — shifted by `ALAG1 = LAG`.
//!
//! Two design points this anchor turns on, both of which the obvious alternative
//! gets wrong:
//!
//! * **`D1`/`ALAG1`, not `mixed_zero_first.ctl`'s `PODO`/`F1=0` `$DES` trick.** `PODO`
//!   tracks only the *most recent* dose, and this fixture needs the second window to
//!   open while drug from the first is still present. NONMEM's native duration-modeled
//!   dose superposes correctly.
//! * **The target is `PRED`/`IPRED`, not the OFV.** ferx's objective path
//!   (`compute_predictions_with_tv` → `ode_predictions`) was already correct here, so
//!   an OFV-only anchor passes vacuously in *both* directions — it could not have
//!   caught #1171, where the two *states* engines dropped the zero-order window and
//!   returned `0.0` at every observation.
//!
//! The setup was validated before being trusted: NONMEM's `PRED` at `t = 2.3` is
//! `7.6884E-01` against `0.768837` computed by hand from the closed form
//! (`nonmem_anchor/simulate_lagged_zo.py`). `#OBJV = -264.65083533221622`.
//!
//! **Only the `lag_cmt = 0` case is expressible.** `ALAGn` is per-compartment, so a
//! per-route lag *composed* with a compartment lag needs two compartments and stops
//! being the same object — the same limitation `tests/dose_form_lag_nonmem_anchor.rs`
//! records.
//!
//! # Tiering
//!
//! The `PRED`-table checks are **ungated**: they are single ODE evaluations at
//! `η = 0` with no convergence loop, they are what carries this diff's Codecov patch
//! coverage, and they compare like with like (`PRED` against ferx at zero η, never
//! against `IPRED`). The `POSTHOC` sdtab check runs a `fit`, so it carries the
//! `slow-tests` gate the other NONMEM anchor suites use.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::pk::{compute_predictions_with_states, compute_predictions_with_tv};
use ferx_core::types::CompiledModel;
use ferx_core::{read_nonmem_csv, Population};
use std::collections::HashMap;
use std::path::Path;

fn anchor(name: &str) -> String {
    format!("nonmem_anchor/{name}")
}

/// The committed ferx twin of `lagged_zo.ctl` — same θ, all fixed, `maxiter = 0`,
/// `ode_reltol/abstol = 1e-9` (NONMEM `TOL=9`).
fn anchor_model() -> (CompiledModel, ferx_core::FitOptions) {
    let src = std::fs::read_to_string(anchor("lagged_zo_fit.ferx"))
        .expect("the ferx twin of the anchor is committed");
    let parsed = parse_full_model(&src).expect("the anchor model must parse");
    (parsed.model, parsed.fit_options)
}

/// The ferx-keyed twin of `lagged_zo_nm.csv` (no `RATE` column — ferx reads the
/// duration off the model's `zero_order(dur=…)`).
fn anchor_population() -> Population {
    read_nonmem_csv(Path::new(&anchor("lagged_zo.csv")), None, None)
        .expect("the anchor dataset is committed")
}

/// `(ID, TIME) → column` from `results/lagged_zo.tab`.
///
/// The table has a row per *record*, dose rows (`MDV=1`) included, so callers look
/// up by key rather than zipping positionally. `PRED` is the η = 0 oracle; `IPRED`
/// is NONMEM's `POSTHOC` value at its own η̂.
fn nonmem_column(column: &str) -> HashMap<(String, u64), f64> {
    let text = std::fs::read_to_string(anchor("results/lagged_zo.tab"))
        .expect("the NONMEM table is committed");
    let mut lines = text.lines();
    lines.next().expect("TABLE NO. banner");
    let header: Vec<&str> = lines
        .next()
        .expect("column header")
        .split_whitespace()
        .collect();
    let id_col = header.iter().position(|c| *c == "ID").expect("ID column");
    let time_col = header
        .iter()
        .position(|c| *c == "TIME")
        .expect("TIME column");
    let col = header
        .iter()
        .position(|c| c == &column)
        .unwrap_or_else(|| panic!("{column} column"));

    let mut out = HashMap::new();
    for line in lines {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() <= col.max(time_col).max(id_col) {
            continue;
        }
        let id: f64 = f[id_col].parse().expect("numeric ID");
        let time: f64 = f[time_col].parse().expect("numeric TIME");
        let value: f64 = f[col].parse().expect("numeric value");
        // IDs are written `1.0000E+00`; ferx keeps the raw CSV string `1`.
        out.insert((format!("{}", id as i64), time.to_bits()), value);
    }
    out
}

/// NONMEM writes `$TABLE` at five significant digits, so ~1e-5 relative is the
/// table's own resolution and 1e-4 is the tightest defensible bound. Measured
/// agreement is 2.8e-5, i.e. print precision. The defect this guards against
/// returns `0.0`, a relative error of exactly 1.
const TABLE_TOL: f64 = 1e-4;

/// Compare one ferx per-observation series against a NONMEM `$TABLE` column,
/// asserting non-degeneracy on both sides first: an all-zero reference, or an
/// all-zero ferx series, is exactly the failure mode under test and must not be
/// allowed to pass as agreement.
fn assert_matches_table(
    got: &[(String, f64, f64)],
    table: &HashMap<(String, u64), f64>,
    tol: f64,
    what: &str,
) {
    assert!(
        got.len() >= 100,
        "{what}: expected the full 12 × 10 record set, got {}",
        got.len()
    );
    let peak = got.iter().fold(0.0f64, |m, r| m.max(r.2.abs()));
    assert!(
        peak > 1e-3,
        "{what}: ferx returned an all-zero series (peak {peak:.3e}) — that is the \
         #1171 signature, not agreement"
    );
    let mut worst = 0.0f64;
    for (id, time, value) in got {
        let want = table
            .get(&(id.clone(), time.to_bits()))
            .unwrap_or_else(|| panic!("{what}: NONMEM has no row for ID {id} at t={time}"));
        assert!(
            want.abs() > 1e-3,
            "{what}: the NONMEM reference at ID {id} t={time} is {want}, a degenerate \
             comparison"
        );
        let rel = (value - want).abs() / want.abs();
        worst = worst.max(rel);
        assert!(
            rel <= tol,
            "{what}: ID {id} t={time} — ferx {value} vs NONMEM {want} (rel {rel:.3e} > {tol:.1e})"
        );
    }
    assert!(
        worst > 0.0,
        "{what}: bit-identical to NONMEM is implausible"
    );
}

/// Evaluate `f` at **η = 0** for every subject, returning `(id, time, value)` rows
/// aligned with the dataset's observation records.
fn ferx_rows_at_zero_eta(
    model: &CompiledModel,
    pop: &Population,
    f: impl Fn(&CompiledModel, &ferx_core::Subject, &[f64], &[f64]) -> Vec<f64>,
) -> Vec<(String, f64, f64)> {
    let theta = &model.default_params.theta;
    let zero_eta = vec![0.0; model.default_params.omega.dim()];
    let mut out = Vec::new();
    for subject in &pop.subjects {
        let values = f(model, subject, theta, &zero_eta);
        assert_eq!(values.len(), subject.obs_times.len());
        for (t, v) in subject.obs_times.iter().zip(values) {
            out.push((subject.id.clone(), *t, v));
        }
    }
    out
}

/// **The objective's engine.** Correct before #1171 and after it — recorded here so
/// the two checks below are read as *the states engines catching up to this one*,
/// not as a tolerance being relaxed to fit.
#[test]
fn objective_path_matches_the_nonmem_pred_column() {
    let (model, _) = anchor_model();
    let pop = anchor_population();
    let rows = ferx_rows_at_zero_eta(&model, &pop, compute_predictions_with_tv);
    assert_matches_table(
        &rows,
        &nonmem_column("PRED"),
        TABLE_TOL,
        "compute_predictions_with_tv vs NONMEM PRED",
    );
}

/// **The sdtab IPRED engine** (`ode_predictions_with_states`). Without #1171's fix
/// this returns `0.0` at every one of the 120 observations — a pure lagged
/// `zero_order` model produced an all-zero sdtab IPRED column.
#[test]
fn states_path_ipred_matches_the_nonmem_pred_column() {
    let (model, _) = anchor_model();
    let pop = anchor_population();
    let rows = ferx_rows_at_zero_eta(&model, &pop, |m, s, th, e| {
        compute_predictions_with_states(m, s, th, e).0
    });
    assert_matches_table(
        &rows,
        &nonmem_column("PRED"),
        TABLE_TOL,
        "compute_predictions_with_states IPRED vs NONMEM PRED",
    );
}

/// **The compartment-state column.** `states[j][central] / V` is the same
/// concentration NONMEM's `$ERROR IPRED = A(1)/V` reports, so the anchor pins the
/// amounts as well as the readout. Also zero before the fix.
#[test]
fn states_path_compartment_amounts_match_the_nonmem_pred_column() {
    let (model, _) = anchor_model();
    let pop = anchor_population();
    // V is fixed (θ₂) and η_V = 0 here, so the scale is the typical value.
    let v = model.default_params.theta[1];
    let rows = ferx_rows_at_zero_eta(&model, &pop, |m, s, th, e| {
        compute_predictions_with_states(m, s, th, e)
            .1
            .iter()
            .map(|u| u[0] / v)
            .collect()
    });
    assert_matches_table(
        &rows,
        &nonmem_column("PRED"),
        TABLE_TOL,
        "compute_predictions_with_states compartment amounts / V vs NONMEM PRED",
    );
}

/// **The dense engine** (`ode_dense_solve_states` over `build_segment_break_times`) —
/// the second defective builder, and the one feeding the joint PK-TTE hazard, the
/// Markov endpoint NLL, the adaptive AUC signal, `[derived]` grid integrals and
/// `simulate()`'s event-time search. Anchoring it here anchors all of them: none of
/// those five surfaces has a NONMEM equivalent, but every one of them reads this
/// trajectory.
#[test]
fn dense_solve_states_matches_the_nonmem_pred_column() {
    let (model, _) = anchor_model();
    let pop = anchor_population();
    let ode = model
        .ode_spec
        .as_ref()
        .expect("the anchor model is an [odes] model");
    let v = model.default_params.theta[1];
    let rows = ferx_rows_at_zero_eta(&model, &pop, |m, s, th, e| {
        let pk = ferx_core::pk::compute_event_pk_params(m, s, th, e).obs[0];
        ferx_core::ode::ode_dense_solve_states(ode, &pk.values, th, e, s, &s.obs_times)
            .iter()
            .map(|u| u[0] / v)
            .collect()
    });
    assert_matches_table(
        &rows,
        &nonmem_column("PRED"),
        TABLE_TOL,
        "ode_dense_solve_states / V vs NONMEM PRED",
    );
}

/// The anchor must **discriminate**. NONMEM's own `PRED` is far enough from the
/// drug-free `0.0` that a dropped zero-order window fails by a relative 1 at every
/// record; this pins that separation so the anchor cannot quietly stop testing
/// anything (the failure mode a tolerance widening would hide).
#[test]
fn the_anchor_discriminates_a_dropped_zero_order_window() {
    let table = nonmem_column("PRED");
    let obs: Vec<f64> = table.values().copied().filter(|v| v.abs() > 0.0).collect();
    assert!(
        obs.len() >= 100,
        "the PRED column must carry the 120 observation records, found {}",
        obs.len()
    );
    let smallest = obs.iter().fold(f64::INFINITY, |m, v| m.min(v.abs()));
    assert!(
        smallest > 1e3 * TABLE_TOL,
        "the smallest NONMEM PRED is {smallest}; a dropped window (ferx → 0.0) must \
         fail this anchor by orders of magnitude, not marginally"
    );
}

/// **sdtab level, through `fit`.** `MAXEVAL=0 POSTHOC` on both sides: no outer
/// steps, only the inner EBE search, so the comparison is ferx's η̂ and IPRED
/// against NONMEM's. Slow-gated because it runs a fit; the η = 0 checks above are
/// what run on a PR.
///
/// The tolerance is looser than `TABLE_TOL` because the two inner optimizers stop
/// at slightly different η̂ — the point of this test is the *sdtab column* being
/// populated and near NONMEM's, which the η = 0 tests above pin exactly.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored lagged zero-order sdtab: opt in with --features slow-tests"
)]
fn sdtab_ipred_matches_nonmem_posthoc() {
    let (model, mut opts) = anchor_model();
    let pop = anchor_population();
    opts.verbose = false;
    opts.run_covariance_step = false;
    let result =
        ferx_core::fit(&model, &pop, &model.default_params, &opts).expect("anchor fit must run");

    let table = nonmem_column("IPRED");
    let mut rows = Vec::new();
    for (sr, subject) in result.subjects.iter().zip(&pop.subjects) {
        assert_eq!(sr.ipred.len(), subject.obs_times.len());
        assert!(
            !sr.compartment_states.is_empty(),
            "subject {} has no sdtab compartment states",
            sr.id
        );
        for (t, &ip) in subject.obs_times.iter().zip(&sr.ipred) {
            rows.push((sr.id.clone(), *t, ip));
        }
    }
    // 3 % — NONMEM's and ferx's inner optimizers stop at slightly different η̂ on a
    // 12-subject, 10-observation-per-subject dataset under a 15 % proportional error
    // model. Still 30× under the defect (a relative 1 at every record).
    assert_matches_table(&rows, &table, 3e-2, "sdtab IPRED vs NONMEM POSTHOC IPRED");
}
