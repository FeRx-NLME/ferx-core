//! One-shot generator + ferx side of the **transit + multi-dose + lagtime + F** NONMEM anchor
//! (#719 close-out: "lag/F/covariate are also anchored vs NONMEM, incl. multiple dose").
//! Run manually:
//!   cargo test --test gen_transit_multidose_lag_f_anchor --no-default-features --features ci,slow-tests -- --nocapture
//!
//! Writes two CSVs and prints the ferx FOCEI fit so the NONMEM control
//! (`nonmem_anchor/transit_multidose_lag_f.ctl`) can be run on the matched data and the OFV /
//! estimates compared:
//!   * `nonmem_anchor/transit_multidose_lag_f.csv` — dose CMT 1, **obs CMT 5** (explicit chain's
//!     central).
//!   * `data/transit_multidose_lag_f.csv`          — dose CMT 1, **obs CMT 1** (ferx's single
//!     disposition compartment). Same DV; only the obs-CMT label differs.
//!
//! ## What this anchors
//!
//! ferx's `one_cpt_transit(..., lagtime=LAG, f=FB)` under:
//!   * an **estimated absorption lagtime** (truth 0.5 h) — every dose's transit input is delayed;
//!   * a **fixed bioavailability** F = 0.7 — F is structurally non-identifiable on single-route
//!     oral data (only CL/F, V/F are), so it is FIXed in both engines; the anchor verifies ferx
//!     *applies* F (scales the absorbed amount) identically to NONMEM, not that it estimates it;
//!   * **multiple overlapping doses** — 100 mg q6h × 5 (see the cov-anchor generator for why these
//!     overlap), so the twin's cross-dose superposition is exercised;
//!   * a **TV covariate** — a fixed allometric `CL = TVCL·(WT/70)^0.75` that composes on the twin
//!     path (present, not estimated — the estimated-covariate case is the cov anchor).
//!
//! Because `lagtime=`/`f=` are mapped, ferx routes each subject to its `transit()` ODE twin (#735)
//! — this is the twin-path counterpart to the analytic cov anchor. The NONMEM side is the same
//! explicit integer-N (n=3 → 4 transit compartments) Erlang chain, now with native `ALAG1`/`F1` on
//! the dose compartment, so lag/F/multi-dose are all NONMEM-native (no hand-coded `$DES` shift —
//! which choked LSODA in the first_order_alag experiment; a native `ALAG1` restarts the integrator
//! cleanly at the lagged dose).

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, Population};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions};
use std::fmt::Write as _;

mod common;

const DOSE_TIMES: [f64; 5] = [0.0, 6.0, 12.0, 18.0, 24.0]; // 100 mg q6h × 5 (overlapping)
const TADS: [f64; 5] = [0.75, 1.5, 3.0, 4.5, 5.75]; // obs within each interval, all past the 0.5 h lag
const N_SUBJECTS: usize = 24;
const DOSE: f64 = 100.0;

fn weight_for(id: usize) -> f64 {
    50.0 + 2.0 * ((id - 1) % 24) as f64
}

fn model() -> ferx_core::types::CompiledModel {
    parse_model_string(
        r"
[parameters]
  theta TVCL(9.0, 0.1, 100.0)
  theta TVV(60.0, 1.0, 500.0)
  theta TVMTT(3.0, 0.05, 24.0)
  theta TVN(3.0, FIX)
  theta TVLAG(0.5, 0.001, 3.0)
  theta FBIO(0.7, FIX)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * (WT/70.0)^0.75 * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  MTT = TVMTT
  NTR = TVN
  LAG = TVLAG
  FB  = FBIO
[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT, lagtime=LAG, f=FB)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-9
",
    )
    .unwrap()
}

fn build_subject(id: usize) -> ferx_core::types::Subject {
    let mut doses = Vec::new();
    let mut obs_times = Vec::new();
    for &start in &DOSE_TIMES {
        doses.push(DoseEvent::new(start, DOSE, 1, 0.0, false, 0.0));
        for &tad in &TADS {
            obs_times.push(start + tad);
        }
    }
    let n = obs_times.len();
    let mut s = common::subject(&id.to_string(), doses, obs_times, vec![0.0; n], vec![1; n]);
    s.covariates.insert("WT".to_string(), weight_for(id));
    s
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "manual anchor generator: opt in with --features slow-tests"
)]
fn generate_transit_multidose_lag_f_anchor() {
    let model = model();
    let design = Population {
        subjects: (1..=N_SUBJECTS).map(build_subject).collect(),
        covariate_names: vec!["WT".to_string()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    // Simulate one replicate (BSV on CL & V + proportional residual) from the lag+F twin model.
    let sims = simulate_with_seed(&model, &design, &model.default_params, 1, 719_735);
    let mut pop = design.clone();
    for subj in pop.subjects.iter_mut() {
        subj.observations = sims
            .iter()
            .filter(|r| r.id == subj.id)
            .map(|r| r.outcome.continuous_value())
            .collect();
    }

    let mut csv_nm = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV,WT\n");
    let mut csv_ferx = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV,WT\n");
    for subj in &pop.subjects {
        let wt = subj.covariates["WT"];
        let mut oi = 0usize;
        for &start in &DOSE_TIMES {
            let _ = writeln!(csv_nm, "{},{},.,1,{},1,1,{}", subj.id, start, DOSE, wt);
            let _ = writeln!(csv_ferx, "{},{},.,1,{},1,1,{}", subj.id, start, DOSE, wt);
            for &tad in &TADS {
                let dv = subj.observations[oi];
                oi += 1;
                let t = start + tad;
                let _ = writeln!(csv_nm, "{},{},{:.6},0,.,5,0,{}", subj.id, t, dv, wt);
                let _ = writeln!(csv_ferx, "{},{},{:.6},0,.,1,0,{}", subj.id, t, dv, wt);
            }
        }
    }
    std::fs::create_dir_all("nonmem_anchor").unwrap();
    std::fs::write("nonmem_anchor/transit_multidose_lag_f.csv", csv_nm).unwrap();
    std::fs::create_dir_all("data").unwrap();
    std::fs::write("data/transit_multidose_lag_f.csv", csv_ferx).unwrap();
    println!(
        "WROTE nonmem_anchor/transit_multidose_lag_f.csv + data/transit_multidose_lag_f.csv ({} subj, {} doses each)",
        pop.subjects.len(),
        DOSE_TIMES.len()
    );

    // ferx FOCEI fit on the same data (transit + lagtime + F + multi-dose → routed to ODE twin).
    // The RK45 twin at 1e-9 is expensive, so this is a *minimal* sanity fit (a few outer steps) —
    // the committed validation is `transit_multidose_lag_f_nonmem_anchor.rs`, which evaluates the
    // objective at NONMEM's optimum (fast, deterministic) rather than fitting the slow twin to
    // convergence.
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.methods = vec![];
    opts.outer_maxiter = 3; // slow twin — minimal sanity only (real check = the committed anchor)
    opts.run_covariance_step = false;
    opts.ode_reltol = 1e-9;
    opts.ode_abstol = 1e-9;
    let r = fit(&model, &pop, &model.default_params, &opts).expect("transit+lag+F+multidose fit");
    println!("FERX_OFV {:.4}", r.ofv);
    for (n, v) in r.theta_names.iter().zip(r.theta.iter()) {
        println!("FERX_THETA {n} = {v:.5}");
    }
    for (i, n) in r.eta_names.iter().enumerate() {
        println!("FERX_OMEGA {n} = {:.5}", r.omega[(i, i)]);
    }
    for (n, v) in r.sigma_names.iter().zip(r.sigma.iter()) {
        println!("FERX_SIGMA {n} = {:.6} (sd) var = {:.6}", v, v * v);
    }
}
