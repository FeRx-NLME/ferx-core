//! #2505: `SimulationOutput::parameters` reports the η, individual parameters and
//! `[derived]` values each simulated subject was drawn with.
//!
//! NONMEM's `$SIMULATION` + `$TABLE … FIRSTONLY` is not comparable draw for draw on a fixed
//! seed, so the oracles are the two the issue names, both outside the table's own code:
//! (a) the returned η, folded into θ and handed to the public `predict()` at η = 0,
//! reproduces the returned IPRED; (b) on a linear one-compartment oral model at steady
//! state, the returned `F·dose/CL` equals the trapezoid AUC_τ of the returned IPRED.

use super::*;
use crate::parser::model_parser::parse_model_string;
use std::collections::HashMap;
use std::io::Write;

/// One-compartment oral, log-normal η on CL, V and KA. `extra` is appended verbatim (a
/// `[derived]` block); `cl` is CL's expression.
fn source(cl: &str, extra: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(1.2, 0.005, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.16
  sigma PROP ~ 0.1 (sd)
[individual_parameters]
  CL = {cl}
  V  = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP)
{extra}
"#
    )
}

fn model(cl: &str, extra: &str) -> CompiledModel {
    let src = source(cl, extra);
    parse_model_string(&src).unwrap_or_else(|e| panic!("{e}\n{src}"))
}

fn read(csv: &str) -> Population {
    let mut f = tempfile::Builder::new()
        .prefix("ferx_2505_")
        .suffix(".csv")
        .tempfile()
        .expect("create #2505 fixture csv");
    f.write_all(csv.as_bytes())
        .expect("write #2505 fixture csv");
    f.flush().expect("flush #2505 fixture csv");
    crate::read_nonmem_csv(f.path(), None, None).expect("read #2505 fixture csv")
}

/// `n` subjects, 100 mg at t = 0 and observations at `times`; `wt(id, t)` is the WT column.
fn population(n: usize, times: &[f64], wt: impl Fn(usize, f64) -> f64) -> Population {
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,MDV,WT\n");
    for id in 1..=n {
        csv.push_str(&format!("{id},0,0,100,1,1,1,{}\n", wt(id, 0.0)));
        for &t in times {
            csv.push_str(&format!("{id},{t},1.0,0,0,1,0,{}\n", wt(id, t)));
        }
    }
    read(&csv)
}

const TIMES: [f64; 6] = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0];

fn opts(seed: u64) -> SimulateOptions {
    SimulateOptions {
        seed: Some(seed),
        ..SimulateOptions::default()
    }
}

fn column(table: &SimulatedParameters, names: &[String], name: &str) -> usize {
    names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("no column {name} in {names:?} ({table:?})"))
}

/// One table row per `(draw, sim, id)`, in the order the rows produced them, and every
/// key the rows carry has exactly one table row.
fn assert_keys_match(out: &SimulationOutput) {
    let mut row_keys: Vec<(usize, usize, String)> = Vec::new();
    for r in &out.results {
        let key = (r.draw, r.sim, r.id.clone());
        if row_keys.last() != Some(&key) {
            row_keys.push(key);
        }
    }
    let table_keys: Vec<(usize, usize, String)> = out
        .parameters
        .rows
        .iter()
        .map(|p| (p.draw, p.sim, p.id.clone()))
        .collect();
    assert_eq!(table_keys, row_keys);
}

/// (a) Round trip: fold each row's η into θ and `predict()` that subject at η = 0. The
/// prediction must reproduce the IPRED the simulation emitted for the same
/// `(draw, sim, id)` — so the η is the one the rows were simulated with, of the right
/// subject and replicate — and the individual CL / V / KA must be `θ·exp(η)`.
#[test]
fn returned_eta_reproduces_the_simulated_ipred_through_predict() {
    let m = model("TVCL * exp(ETA_CL)", "");
    let pop = population(4, &TIMES, |_, _| 70.0);
    let out = simulate_with_options_diag(&m, &pop, &m.default_params, 3, &opts(2505))
        .expect("simulation runs");
    let table = &out.parameters;
    assert_eq!(table.eta_names, ["ETA_CL", "ETA_V", "ETA_KA"]);
    assert_eq!(table.individual_names, ["CL", "V", "KA"]);
    assert!(table.derived_names.is_empty());
    assert!(table.varying.is_empty(), "{:?}", table.varying);
    assert_eq!(table.rows.len(), 3 * 4);
    assert_keys_match(&out);

    let theta = &m.default_params.theta;
    let (cl, v, ka) = (
        column(table, &table.individual_names, "CL"),
        column(table, &table.individual_names, "V"),
        column(table, &table.individual_names, "KA"),
    );
    let mut worst = 0.0_f64;
    for p in &table.rows {
        assert!(p.kappas.is_empty() && p.mixnum.is_none());
        // The η must not be zero, or this round trip says nothing about which η it is.
        assert!(p.eta.iter().all(|e| *e != 0.0), "{:?}", p.eta);
        let folded: Vec<f64> = (0..3).map(|k| theta[k] * p.eta[k].exp()).collect();
        assert_eq!(p.individual[cl], folded[0]);
        assert_eq!(p.individual[v], folded[1]);
        assert_eq!(p.individual[ka], folded[2]);

        let mut params = m.default_params.clone();
        params.theta = folded;
        let subject_pop = Population {
            subjects: pop
                .subjects
                .iter()
                .filter(|s| s.id == p.id)
                .cloned()
                .collect(),
            ..pop.clone()
        };
        let predicted = predict(&m, &subject_pop, &params).expect("predict runs");
        let simulated: Vec<&SimulationResult> = out
            .results
            .iter()
            .filter(|r| r.draw == p.draw && r.sim == p.sim && r.id == p.id)
            .collect();
        assert_eq!(predicted.len(), simulated.len());
        for (want, got) in predicted.iter().zip(&simulated) {
            assert_eq!(want.time, got.time);
            assert!(want.pred.is_finite() && got.ipred.is_finite());
            worst = worst.max(((want.pred - got.ipred) / got.ipred).abs());
        }
    }
    // Measured 0 (macOS arm64): the folded θ gives the same CL/V/KA bit for bit, and both
    // paths run the same closed form. A wrong η is off by exp(η), i.e. percent-level.
    assert!(worst <= 1e-14, "worst relative IPRED gap {worst:e}");
}

/// (b) At steady state on a linear one-compartment oral model, AUC over one dosing
/// interval is exactly `F·dose/CL`. The trapezoid of the returned IPRED on a 0.01 h grid
/// over τ = 12 h must equal it computed from the returned CL (F = 1).
#[test]
fn returned_cl_gives_the_steady_state_auc_tau() {
    let m = model("TVCL * exp(ETA_CL)", "");
    let tau = 12.0;
    let dose = 100.0;
    let grid: Vec<f64> = (0..=1200).map(|i| i as f64 * 0.01).collect();
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,MDV,SS,II\n");
    for id in 1..=5 {
        csv.push_str(&format!("{id},0,0,{dose},1,1,1,1,{tau}\n"));
        for t in &grid {
            csv.push_str(&format!("{id},{t},1.0,0,0,1,0,0,0\n"));
        }
    }
    let pop = read(&csv);
    let out = simulate_with_options_diag(&m, &pop, &m.default_params, 2, &opts(7))
        .expect("simulation runs");
    let table = &out.parameters;
    let cl = column(table, &table.individual_names, "CL");
    assert_eq!(table.rows.len(), 2 * 5);
    let mut worst = 0.0_f64;
    for p in &table.rows {
        let points: Vec<(f64, f64)> = out
            .results
            .iter()
            .filter(|r| r.draw == p.draw && r.sim == p.sim && r.id == p.id)
            .map(|r| (r.time, r.ipred))
            .collect();
        assert_eq!(points.len(), grid.len());
        let auc = super::output_columns::trapezoid(&points);
        let want = dose / p.individual[cl];
        assert!(auc.is_finite() && want.is_finite());
        worst = worst.max(((auc - want) / want).abs());
    }
    // Measured worst 1.99e-6 (macOS arm64): the trapezoid's own O(dt²) error on the 0.01 h
    // grid. 1e-5 gives 5× headroom. A wrong CL (another subject's, or the typical value)
    // is off by its η, sd 0.3, i.e. tens of percent.
    assert!(worst < 1e-5, "worst relative AUC_tau gap {worst:e}");
}

/// A `[derived]` block is evaluated with the simulated draw: `KE = CL / V` from the drawn
/// parameters, `CMAX = max(IPRED)` from the simulated IPRED, and `DVMAX = max(DV)` from
/// the **simulated** values — the template's DV is 1.0 everywhere, so reading it instead
/// would report 1.0.
#[test]
fn derived_columns_are_evaluated_on_the_simulated_draw() {
    let m = model(
        "TVCL * exp(ETA_CL)",
        "[derived]\n  KE = CL / V\n  CMAX = max(IPRED)\n  DVMAX = max(DV)\n",
    );
    let pop = population(3, &TIMES, |_, _| 70.0);
    let out = simulate_with_options_diag(&m, &pop, &m.default_params, 2, &opts(11))
        .expect("simulation runs");
    let table = &out.parameters;
    let (cl, v) = (
        column(table, &table.individual_names, "CL"),
        column(table, &table.individual_names, "V"),
    );
    let ke = column(table, &table.derived_names, "KE");
    let cmax = column(table, &table.derived_names, "CMAX");
    let dvmax = column(table, &table.derived_names, "DVMAX");
    assert!(table.varying.is_empty(), "{:?}", table.varying);
    for p in &table.rows {
        let rows: Vec<&SimulationResult> = out
            .results
            .iter()
            .filter(|r| r.draw == p.draw && r.sim == p.sim && r.id == p.id)
            .collect();
        let max_ipred = rows.iter().map(|r| r.ipred).fold(f64::MIN, f64::max);
        let max_dv = rows
            .iter()
            .map(|r| r.outcome.continuous_value())
            .fold(f64::MIN, f64::max);
        assert!(max_ipred.is_finite() && max_dv.is_finite());
        assert_eq!(p.derived[ke], p.individual[cl] / p.individual[v]);
        assert_eq!(p.derived[cmax], max_ipred);
        assert_eq!(p.derived[dvmax], max_dv);
        assert_ne!(p.derived[dvmax], 1.0, "DVMAX read the template's DV");
    }
}

/// A covariate that changes within a subject makes CL differ across its rows. The table
/// reports the first observation row's CL and names CL in `varying`; V, which reads no
/// covariate, is not named. The same model on a constant WT names nothing — so the flag
/// tracks the data, not the expression.
#[test]
fn a_time_varying_covariate_is_reported_at_the_first_row_and_named_varying() {
    let m = model("TVCL * (WT / 70) * exp(ETA_CL)", "");
    let theta = &m.default_params.theta;

    // WT 70 at the dose and first observation, 90 from t = 4 on.
    let tv = population(2, &TIMES, |_, t| if t < 4.0 { 70.0 } else { 90.0 });
    let out = simulate_with_options_diag(&m, &tv, &m.default_params, 1, &opts(3))
        .expect("simulation runs");
    let table = &out.parameters;
    let cl = column(table, &table.individual_names, "CL");
    assert_eq!(table.varying, ["CL"]);
    for p in &table.rows {
        assert_eq!(p.individual[cl], theta[0] * p.eta[0].exp());
    }

    let flat = population(2, &TIMES, |_, _| 70.0);
    let out = simulate_with_options_diag(&m, &flat, &m.default_params, 1, &opts(3))
        .expect("simulation runs");
    assert!(out.parameters.varying.is_empty());
}

/// On a `[mixture]` model each row reports the class the subject's rows report, and the
/// individual CL is that class's. `logit(1) = 0` makes both classes likely, and the test
/// requires that both occur, so a table stuck on one class cannot pass.
#[test]
fn a_mixture_row_reports_its_class_and_that_class_s_parameters() {
    let m = parse_model_string(
        "[parameters]\n  theta TVCL(1.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
         theta MIXL(0.0, -2000.0, 2000.0, FIX)\n  omega ETA_CL ~ 0.09\n  sigma EPS ~ 0.04\n\
         [mixture]\n  nsub = 2\n  logit(1) = MIXL\n[individual_parameters]\n  \
         CL = if (MIXNUM == 1) TVCL * exp(ETA_CL) else 4 * TVCL * exp(ETA_CL)\n  V  = TVV\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  \
         DV ~ proportional(EPS)\n",
    )
    .expect("mixture fixture parses");
    let pop = population(8, &TIMES, |_, _| 70.0);
    let out = simulate_with_options_diag(&m, &pop, &m.default_params, 3, &opts(2163))
        .expect("simulation runs");
    assert_keys_match(&out);
    let table = &out.parameters;
    let cl = column(table, &table.individual_names, "CL");
    let class_of: HashMap<(usize, usize, String), Option<usize>> = out
        .results
        .iter()
        .map(|r| ((r.draw, r.sim, r.id.clone()), r.mixnum))
        .collect();
    let mut seen = [false; 2];
    for p in &table.rows {
        let k = p.mixnum.expect("a mixture row carries its class");
        assert_eq!(Some(k), class_of[&(p.draw, p.sim, p.id.clone())]);
        seen[k - 1] = true;
        let scale = if k == 1 { 1.0 } else { 4.0 };
        assert_eq!(
            p.individual[cl],
            scale * m.default_params.theta[0] * p.eta[0].exp()
        );
    }
    assert_eq!(seen, [true, true], "both classes must occur");
}

/// `simulate_with_uncertainty_diag` reports one row per `(draw, sim, id)`, each under its
/// own draw's θ: the returned CL / V / KA reproduce that row's IPRED through the
/// one-compartment oral closed form, which a table carrying the point estimate's θ (or
/// another draw's) would not.
#[test]
fn uncertainty_rows_carry_each_draws_parameters() {
    let m = model("TVCL * exp(ETA_CL)", "");
    let pop = population(3, &TIMES, |_, _| 70.0);
    let fit = super::simulate_with_uncertainty_tests::synthetic_fit(&m.default_params);
    let opts = SimulateUncertaintyOptions {
        n_uncertainty_draws: 3,
        n_sim_per_draw: 2,
        seed: Some(9),
        ..SimulateUncertaintyOptions::default()
    };
    let out = simulate_with_uncertainty_diag(&m, &pop, &fit, &opts).expect("simulation runs");
    let table = &out.parameters;
    assert_eq!(table.rows.len(), 3 * 2 * 3);
    assert_keys_match(&out);
    let ix = |n: &str| column(table, &table.individual_names, n);
    let (cl, v, ka) = (ix("CL"), ix("V"), ix("KA"));
    let mut worst = 0.0_f64;
    for p in &table.rows {
        let (cl, v, ka) = (p.individual[cl], p.individual[v], p.individual[ka]);
        let k = cl / v;
        for r in out
            .results
            .iter()
            .filter(|r| r.draw == p.draw && r.sim == p.sim && r.id == p.id)
        {
            let want = 100.0 * ka / (v * (ka - k)) * ((-k * r.time).exp() - (-ka * r.time).exp());
            assert!(r.ipred.is_finite() && want.is_finite());
            worst = worst.max(((r.ipred - want) / want).abs());
        }
    }
    // Measured worst 0 (macOS arm64); two spellings of one closed form, so allow ULP-level
    // drift. Another draw's θ moves CL by its uncertainty draw, far above this.
    assert!(worst < 1e-13, "worst relative IPRED gap {worst:e}");

    // The row-only twin returns the same rows without building the table.
    let rows = simulate_with_uncertainty(&m, &pop, &fit, &opts).expect("simulation runs");
    assert_eq!(rows.len(), out.results.len());
}
