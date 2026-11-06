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
//! - A reset turns every compartment on (ID 15, 16), and takes effect in file order against
//!   an off row at its own `TIME`: it undoes one before it (ID 29), not one after it (ID 30).
//! - Co-timed records take effect in file order (ID 5, 17, 18, 19, 20). IDs 25 to 28 are
//!   the ties the #614 and #1810 passes leave wrong, so only `order_cmt_toggle_ties`
//!   orders them: an infusion before an off row (25), an observation between a dose and
//!   an off row (26), and an off row between an observation and a dose (27, 28).
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
const NM_OFV: f64 = 19834.949763454737;
/// Measured on macOS arm64: worst 5.4e-15 on the linear route and in the states, 6.2e-13 on
/// `rodas5p`, 1.5e-11 on `rk45` at `1e-10` tolerances. Bound set from the last, with about
/// 70× headroom for the solvers' platform variation.
const BOUND: f64 = 1e-9;
/// Measured on macOS arm64: Δ 1.8e-5 on 19445, the two engines' EBE tolerances. About
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

/// `compute_predictions_ode` is public and takes one flat parameter vector, so it has its
/// own route to the walk. ID 1 reads the same whichever way it gets there.
#[test]
fn the_flat_parameter_entry_point_honours_the_row() {
    let m = model("");
    let pop = population();
    let s = &pop.subjects[0];
    let theta = &m.default_params.theta;
    let eta = vec![0.0; m.n_eta];
    let pk = (m.pk_param_fn)(theta, &eta, &s.covariates, 0.0);
    let ode = m.ode_spec.as_ref().expect("an [odes] model");
    let got = ferx_core::pk::compute_predictions_ode(ode, s, &pk.values, theta, &eta);
    let want = nonmem_obs();
    assert_eq!(got.len(), 3);
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w.3).abs() / (1.0 + w.3.abs()) < BOUND, "{g} vs {w:?}");
    }
}

/// A subject built in memory sets `cmt_toggles` and the pk-only times and nothing else, and
/// must predict what the same records read from a file predict: the dose cutoffs are not
/// the reader's to supply. Each case is a dose the off row must stop, with the compartment
/// back on afterwards, which is where a missing cutoff shows. ID 6 is an infusion stopped at
/// 3, ID 10 a lagged dose in flight, and ID 25 an infusion at the off row's own `TIME`, where
/// the written times hold the tie the reader would have split.
#[test]
fn a_subject_built_in_memory_predicts_what_the_reader_predicts() {
    use ferx_core::types::DoseEvent;
    let m = model("");
    let pop = population();
    let theta = &m.default_params.theta;
    let eta = vec![0.0; m.n_eta];
    for id in ["6", "10", "25"] {
        let read = pop.subjects.iter().find(|s| s.id == id).unwrap();
        let mut built = read.clone();
        built.doses = read
            .doses
            .iter()
            .map(|d| DoseEvent::new(d.time, d.amt, d.cmt_1based(), d.rate, false, 0.0))
            .collect();
        built.pk_only_times = read.pk_only_raw_times.clone();
        let want = ferx_core::pk::compute_predictions_with_tv(&m, read, theta, &eta);
        let got = ferx_core::pk::compute_predictions_with_tv(&m, &built, theta, &eta);
        assert_eq!(got.len(), want.len(), "ID {id}");
        for (g, w) in got.iter().zip(&want) {
            assert!(
                g.is_finite() && (g - w).abs() < 1e-9,
                "ID {id}: {got:?} vs {want:?}"
            );
        }
        let (_, states) = compute_predictions_with_states(&m, &built, theta, &eta);
        let (_, read_states) = compute_predictions_with_states(&m, read, theta, &eta);
        for (g, w) in states.iter().flatten().zip(read_states.iter().flatten()) {
            assert!(g.is_finite() && (g - w).abs() < 1e-9, "ID {id}: {states:?}");
        }
    }
}

/// What a row does to a dose lives in the row: clear the rows from a subject the reader
/// built and the infusion of ID 6 runs to its end. This pins the stored dose, which every
/// walk reads, against one built fresh.
#[test]
fn clearing_the_rows_clears_what_they_did_to_the_doses() {
    use ferx_core::types::DoseEvent;
    let m = model("");
    let pop = population();
    let theta = &m.default_params.theta;
    let eta = vec![0.0; m.n_eta];
    let mut cleared = pop.subjects.iter().find(|s| s.id == "6").unwrap().clone();
    cleared.cmt_toggles.clear();
    let mut plain = cleared.clone();
    plain.doses = cleared
        .doses
        .iter()
        .map(|d| DoseEvent::new(d.time, d.amt, d.cmt_1based(), d.rate, false, 0.0))
        .collect();
    // A later dose breaks the dense walk's segment past the off row, so a cutoff left on the
    // stored dose would stop the infusion there too.
    for s in [&mut cleared, &mut plain] {
        s.doses.push(DoseEvent::new(4.0, 1.0, 1, 0.0, false, 0.0));
    }
    let got = ferx_core::pk::compute_predictions_with_tv(&m, &cleared, theta, &eta);
    let want = ferx_core::pk::compute_predictions_with_tv(&m, &plain, theta, &eta);
    assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
    assert_eq!(got, want);
}
