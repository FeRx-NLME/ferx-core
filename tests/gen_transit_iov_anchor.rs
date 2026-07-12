//! One-shot generator + ferx side of the **transit + IOV** NONMEM anchor (#719).
//! Run manually:
//!   cargo test --test gen_transit_iov_anchor --no-default-features --features ci,slow-tests -- --nocapture
//!
//! Writes `nonmem_anchor/transit_iov.csv` and prints the ferx FOCEI fit so the NONMEM `.ctl`
//! (`nonmem_anchor/transit_iov.ctl`, a hand-coded `$DES` Savic transit density into a single
//! central compartment + IOV on CL) can be run on the same data and the OFV / estimates
//! compared.
//!
//! Design rationale: **one dose per occasion, three occasions 24 h apart, with near-complete
//! washout between them** (ke = CL/V ≈ 0.3/h ⇒ >10 half-lives per interval). This matters for
//! the anchor: ferx superposes the transit input of *every* dose, while a NONMEM `$DES` that
//! captures only the most-recent dose (`PODO`/`TDOS`) sees a single dose per occasion — with
//! washout the two coincide exactly, so any OFV gap is real signal, not a multi-dose bookkeeping
//! artifact. `one_cpt_transit` is transit-density-directly-into-central (its `effective_for`
//! twin is `d/dt(central) = transit(n, mtt) - (CL/V)·central`), so the NONMEM model is a single
//! compartment with `F1 = 0` and the density delivered as `R_in`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, Population};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions};
use std::fmt::Write as _;

mod common;

// Occasions 48 h apart give near-total washout between them (>10 half-lives at ke ≈ 0.3/h),
// and the post-dose observation window stays in the informative range (tad ≤ 3 h) — no
// washout-tail observations, which under proportional error would carry vanishing variance and
// amplify tiny ferx-vs-NONMEM EBE differences into a spurious OFV gap.
const OCC_STARTS: [f64; 3] = [0.0, 48.0, 96.0];
const TADS: [f64; 6] = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0];
const N_SUBJECTS: usize = 24;
const DOSE: f64 = 100.0;

fn model() -> ferx_core::types::CompiledModel {
    parse_model_string(
        r"
[parameters]
  theta TVCL(9.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0, 0.1, 30.0)
  omega ETA_V  ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * exp(KAPPA_CL)
  V   = TVV  * exp(ETA_V)
  MTT = TVMTT
  NTR = TVN
[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  iov_column = OCC
  ode_reltol = 1e-9
  ode_abstol = 1e-9
",
    )
    .unwrap()
}

/// One subject: a dose at each occasion start (OCC 1/2/3) and seven post-dose observations
/// per occasion. Dose and observations both land on CMT 1 (central) — `one_cpt_transit` has a
/// single disposition compartment fed by the transit density.
fn build_subject(id: usize) -> ferx_core::types::Subject {
    let mut doses = Vec::new();
    let mut dose_occasions = Vec::new();
    let mut obs_times = Vec::new();
    let mut occasions = Vec::new();
    for (o, &start) in OCC_STARTS.iter().enumerate() {
        doses.push(DoseEvent::new(start, DOSE, 1, 0.0, false, 0.0));
        dose_occasions.push((o + 1) as u32);
        for &tad in &TADS {
            obs_times.push(start + tad);
            occasions.push((o + 1) as u32);
        }
    }
    let n = obs_times.len();
    let mut s = common::subject(&id.to_string(), doses, obs_times, vec![0.0; n], vec![1; n]);
    s.occasions = occasions;
    s.dose_occasions = dose_occasions;
    s
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "manual anchor generator: opt in with --features slow-tests"
)]
fn generate_transit_iov_anchor() {
    let model = model();
    let design = Population {
        subjects: (1..=N_SUBJECTS).map(build_subject).collect(),
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    // Simulate one replicate (BSV + per-occasion IOV kappa + proportional residual).
    let sims = simulate_with_seed(&model, &design, &model.default_params, 1, 719_719);
    let mut pop = design.clone();
    for subj in pop.subjects.iter_mut() {
        subj.observations = sims
            .iter()
            .filter(|r| r.id == subj.id)
            .map(|r| r.outcome.continuous_value())
            .collect();
    }

    // NONMEM-format CSV, records time-sorted per subject (dose then its occasion's obs).
    let mut csv = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV,OCC\n");
    for subj in &pop.subjects {
        let mut oi = 0usize;
        for (o, &start) in OCC_STARTS.iter().enumerate() {
            let occ = o + 1;
            let _ = writeln!(csv, "{},{},.,1,{},1,1,{}", subj.id, start, DOSE, occ);
            for &tad in &TADS {
                let dv = subj.observations[oi];
                oi += 1;
                let _ = writeln!(csv, "{},{},{:.6},0,.,1,0,{}", subj.id, start + tad, dv, occ);
            }
        }
    }
    std::fs::create_dir_all("nonmem_anchor").unwrap();
    std::fs::write("nonmem_anchor/transit_iov.csv", csv).unwrap();
    println!(
        "WROTE nonmem_anchor/transit_iov.csv ({} subjects, {} occasions each)",
        pop.subjects.len(),
        OCC_STARTS.len()
    );

    // ferx FOCEI fit on the same data (analytic one_cpt_transit + IOV → routed to the transit()
    // ODE twin per #719).
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.methods = vec![];
    opts.outer_maxiter = 20; // bounded cross-check only; the committed anchor is transit_iov_nonmem_anchor.rs
    opts.run_covariance_step = false;
    let r = fit(&model, &pop, &model.default_params, &opts).expect("transit+IOV fit");
    println!("FERX_OFV {:.4}", r.ofv);
    for (n, v) in r.theta_names.iter().zip(r.theta.iter()) {
        println!("FERX_THETA {n} = {v:.5}");
    }
    for (i, n) in r.eta_names.iter().enumerate() {
        println!("FERX_OMEGA {n} = {:.5}", r.omega[(i, i)]);
    }
    if let Some(omega_iov) = &r.omega_iov {
        for (i, n) in r.kappa_names.iter().enumerate() {
            println!("FERX_OMEGA_IOV {n} = {:.5}", omega_iov[(i, i)]);
        }
    }
    for (n, v) in r.sigma_names.iter().zip(r.sigma.iter()) {
        println!("FERX_SIGMA {n} = {:.6} (sd) var = {:.6}", v, v * v);
    }
}
