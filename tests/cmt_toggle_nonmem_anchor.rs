//! NONMEM anchor for #2732: compartment on/off records (`EVID=2`, `CMT=-N`).
//!
//! `nonmem_anchor/cmt_toggle.{ctl,csv}` is one subject per measured case, run on NONMEM
//! 7.5.1 (`ADVAN6 TOL=12`, `METHOD=1 INTER MAXEVAL=0`), with the table in
//! `nonmem_anchor/results/`. `cmt_toggle_gen.py` documents each ID. Between them they
//! fix these rules:
//!
//! - Off zeroes the amount and holds it there. Inflow is lost, and a term reading the
//!   state sees zero (ID 3, on a back-transfer, so an unmasked state would refill
//!   central).
//! - On restarts from zero (ID 1, 2) and does not re-apply `A_0` (ID 24).
//! - A dose turns its compartment on at its arrival, not its record (ID 4, 8, 11, 12).
//! - An off row ends dose input into that compartment for good: a running infusion
//!   (ID 6, 7), a lagged dose in flight (ID 10), the rest of an `ADDL` train (ID 21).
//!   Input elsewhere carries on (ID 9, 22).
//! - A reset turns every compartment on (ID 15, 16).
//! - Co-timed records take effect in file order (ID 5, 17, 18, 19, 20).
//! - The row is a record, so its covariates govern the interval ending at it (ID 23).
//!
//! Multi-dose by construction: ID 2 toggles with residual drug on both sides of every
//! row, and a second dose lands between windows, so neither side of an event is zero.
//!
//! The oracle is the committed table, read at run time. Evaluation only, so not gated.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::pk::compute_predictions_with_states;
use ferx_core::{CompiledModel, Population};
use std::path::{Path, PathBuf};

/// `METHOD=1 INTER MAXEVAL=0` objective, from `results/cmt_toggle.ext`.
const NM_OFV: f64 = 18898.345332748067;
/// Measured on macOS arm64: worst 5.4e-15 on the linear route and in the states, 6.2e-13 on
/// `rodas5p`, 1.5e-11 on `rk45` at `1e-10` tolerances. Bound set from the last, with about
/// 70× headroom for the solvers' platform variation.
const BOUND: f64 = 1e-9;
/// Measured on macOS arm64: Δ 1.8e-5 on 18898, the two engines' EBE tolerances. About
/// 50× headroom. A wrong amount at σ = 0.1 moves this by whole OFV units.
const OFV_BOUND: f64 = 1e-3;

fn anchor(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("nonmem_anchor")
        .join(name)
}

/// The anchor model with `options` appended to `[fit_options]`.
fn model(options: &str) -> CompiledModel {
    let source = std::fs::read_to_string(anchor("cmt_toggle.ferx")).unwrap();
    parse_full_model(&format!("{source}{options}\n"))
        .expect("the anchor model parses")
        .model
}

fn population() -> Population {
    ferx_core::read_nonmem_csv(&anchor("cmt_toggle.csv"), None, None)
        .expect("the anchor dataset loads")
}

/// `(ID, TIME, CMT, PRED)` of every observation in the NONMEM table, whose columns are
/// `ID TIME EVID CMT A1 A2 IPRED PRED`.
fn nonmem_obs() -> Vec<(String, f64, usize, f64)> {
    let text = std::fs::read_to_string(anchor("results/cmt_toggle.tab")).expect("NONMEM table");
    text.lines()
        .skip(2)
        .map(|l| {
            l.split_whitespace()
                .map(|x| x.parse::<f64>().expect("number"))
                .collect::<Vec<f64>>()
        })
        .filter(|c| c[2] == 0.0)
        .map(|c| (format!("{}", c[0] as i64), c[1], c[3] as usize, c[7]))
        .collect()
}

/// Assert `got`, one prediction per NONMEM observation in file order, within `bound` of
/// NONMEM's `PRED`, as `|Δ| / (1 + |PRED|)` because several cases read an exact zero.
fn assert_preds(label: &str, got: &[(String, f64, f64)], bound: f64) {
    let want = nonmem_obs();
    assert_eq!(got.len(), want.len(), "{label}: observation count");
    let mut worst = 0.0_f64;
    let mut bad = Vec::new();
    for ((gid, gt, g), (id, t, _, w)) in got.iter().zip(&want) {
        assert!(
            gid == id && (gt - t).abs() < 1e-12,
            "{label}: row order ({gid}, {gt}) vs NONMEM ({id}, {t})"
        );
        assert!(g.is_finite(), "{label}: ID {id} t={t} non-finite ({g})");
        let rel = (g - w).abs() / (1.0 + w.abs());
        assert!(rel.is_finite(), "{label}: ID {id} t={t} rel {rel}");
        if !(rel < bound) {
            bad.push(format!(
                "ID {id} t={t}: ferx {g} vs NONMEM {w} (rel {rel:.3e})"
            ));
        }
        worst = worst.max(rel);
    }
    println!("#2732 {label}: worst rel vs NONMEM {worst:.3e} (bound {bound:.0e})");
    assert!(bad.is_empty(), "{label}:\n{}", bad.join("\n"));
}

fn via_predict(m: &CompiledModel, pop: &Population) -> Vec<(String, f64, f64)> {
    let preds = ferx_core::predict(m, pop, &m.default_params).expect("predict");
    let ids = pop
        .subjects
        .iter()
        .flat_map(|s| s.obs_times.iter().map(|_| s.id.clone()));
    ids.zip(preds.iter())
        .map(|(id, p)| (id, p.time, p.pred))
        .collect()
}

#[test]
fn predictions_match_nonmem_on_the_linear_route() {
    let m = model("");
    assert_preds("linear", &via_predict(&m, &population()), BOUND);
}

#[test]
fn predictions_match_nonmem_on_a_stepping_solver() {
    let m = model("  ode_method = rk45\n  ode_reltol = 1e-10\n  ode_abstol = 1e-10");
    assert_preds("rk45", &via_predict(&m, &population()), BOUND);
}

#[test]
fn predictions_match_nonmem_on_a_stiff_solver() {
    let m = model("  ode_method = rodas5p\n  ode_reltol = 1e-10\n  ode_abstol = 1e-10");
    assert_preds("rodas5p", &via_predict(&m, &population()), BOUND);
}

/// The compartment amounts come from the walk that made the prediction, so each
/// observation's own compartment reads what NONMEM's `PRED` says.
#[test]
fn compartment_states_match_nonmem() {
    let m = model("");
    let pop = population();
    let want = nonmem_obs();
    let eta = vec![0.0; m.n_eta];
    let mut got = Vec::new();
    for s in &pop.subjects {
        let (_, states) = compute_predictions_with_states(&m, s, &m.default_params.theta, &eta);
        assert_eq!(states.len(), s.obs_times.len(), "subject {}", s.id);
        for (j, row) in states.iter().enumerate() {
            let w = &want[got.len()];
            got.push((s.id.clone(), s.obs_raw_times[j], row[w.2 - 1]));
        }
    }
    assert_preds("states", &got, BOUND);
}

#[test]
fn the_objective_matches_nonmem() {
    let (result, _) = ferx_core::run_model_with_data(
        anchor("cmt_toggle.ferx").to_str().unwrap(),
        Some(anchor("cmt_toggle.csv").to_str().unwrap()),
    )
    .expect("the anchor evaluates");
    let delta = (result.ofv - NM_OFV).abs();
    println!(
        "#2732 OFV: ferx {:.6} vs NONMEM {NM_OFV:.6} (Δ {delta:.3e})",
        result.ofv
    );
    assert!(result.ofv.is_finite(), "OFV {}", result.ofv);
    assert!(
        delta < OFV_BOUND,
        "OFV {} vs {NM_OFV} (Δ {delta:.3e})",
        result.ofv
    );
}
