//! One-shot generator + ferx side of the IIV-on-RUV NONMEM anchor (#409).
//! Run manually:
//!   cargo test --test gen_iiv_anchor --no-default-features --features ci,slow-tests -- --nocapture
//! Writes `nonmem_anchor/iiv_on_ruv.csv` and prints the ferx FOCEI fit so the
//! NONMEM `.ctl` can be run on the same data and the OFV / ETA_RUV variance
//! compared.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::Population;
use ferx_core::{
    fit, read_nonmem_csv, simulate_with_seed, EstimationMethod, FitOptions, SimulationResult,
};
use std::fmt::Write as _;
use std::path::Path;

fn model() -> ferx_core::types::CompiledModel {
    parse_model_string(
        r"
[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.30
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method = focei
",
    )
    .unwrap()
}

fn replicate(base: &Population, copies: usize) -> Population {
    let mut pop = base.clone();
    pop.subjects.clear();
    let mut next = 1usize;
    for _ in 0..copies {
        for s in &base.subjects {
            let mut c = s.clone();
            c.id = next.to_string();
            next += 1;
            pop.subjects.push(c);
        }
    }
    pop
}

fn inject(pop: &mut Population, sims: &[SimulationResult]) {
    for subj in pop.subjects.iter_mut() {
        let v: Vec<f64> = sims
            .iter()
            .filter(|r| r.id == subj.id)
            .map(|r| r.outcome.continuous_value())
            .collect();
        subj.observations = v;
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "manual anchor generator: opt in with --features slow-tests"
)]
fn generate_iiv_on_ruv_anchor() {
    let model = model();
    let design = replicate(
        &read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).unwrap(),
        5,
    );
    let sims = simulate_with_seed(&model, &design, &model.default_params, 1, 409409);
    let mut pop = design.clone();
    inject(&mut pop, &sims);

    // Write the NONMEM-format CSV (ID,TIME,DV,EVID,AMT,CMT,RATE,MDV).
    let mut csv = String::from("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n");
    for subj in &pop.subjects {
        for d in &subj.doses {
            let _ = writeln!(
                csv,
                "{},{},.,1,{},{},{},1",
                subj.id, d.time, d.amt, d.cmt, d.rate as i64
            );
        }
        for (&t, &dv) in subj.obs_times.iter().zip(subj.observations.iter()) {
            // NONMEM ADVAN2 convention: depot = CMT 1, central = CMT 2. ferx's
            // `one_cpt_oral` observes its own CMT 1 (central), but the in-memory
            // `pop` the ferx fit uses keeps that; only the on-disk CSV (for the
            // NONMEM run) labels observations CMT 2.
            let _ = writeln!(csv, "{},{},{:.6},0,.,2,0,0", subj.id, t, dv);
        }
    }
    std::fs::create_dir_all("nonmem_anchor").unwrap();
    std::fs::write("nonmem_anchor/iiv_on_ruv.csv", csv).unwrap();
    println!(
        "WROTE nonmem_anchor/iiv_on_ruv.csv ({} subjects)",
        pop.subjects.len()
    );

    // ferx FOCEI fit on the same data.
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.methods = vec![];
    opts.interaction = true;
    opts.run_covariance_step = false;
    let r = fit(&model, &pop, &model.default_params, &opts).unwrap();
    let ridx = r.eta_names.iter().position(|n| n == "ETA_RUV").unwrap();
    println!("FERX_OFV {:.4}", r.ofv);
    for (n, v) in r.theta_names.iter().zip(r.theta.iter()) {
        println!("FERX_THETA {n} = {v:.5}");
    }
    for (i, n) in r.eta_names.iter().enumerate() {
        println!("FERX_OMEGA {n} = {:.5}", r.omega[(i, i)]);
    }
    println!("FERX_ETA_RUV_VAR {:.5}", r.omega[(ridx, ridx)]);
    for (n, v) in r.sigma_names.iter().zip(r.sigma.iter()) {
        println!("FERX_SIGMA {n} = {v:.6} (sd)  var = {:.6}", v * v);
    }
}
