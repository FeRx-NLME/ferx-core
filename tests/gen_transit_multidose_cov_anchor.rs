//! One-shot generator + ferx side of the **transit + multi-dose + TV-covariate** NONMEM anchor
//! (#719 close-out: "lag/F/covariate are also anchored vs NONMEM, incl. multiple dose").
//! Run manually:
//!   cargo test --test gen_transit_multidose_cov_anchor --no-default-features --features ci,slow-tests -- --nocapture
//!
//! Writes two CSVs and prints the ferx FOCEI fit so the NONMEM control
//! (`nonmem_anchor/transit_multidose_cov.ctl`) can be run on the matched data and the OFV /
//! estimates compared:
//!   * `nonmem_anchor/transit_multidose_cov.csv` — dose CMT 1, **obs CMT 5** (the explicit
//!     integer-N transit chain's central compartment; see below).
//!   * `data/transit_multidose_cov.csv`          — dose CMT 1, **obs CMT 1** (ferx's single
//!     disposition compartment). Same DV values, only the obs-CMT label differs (the igd-anchor
//!     re-key pattern).
//!
//! ## What this anchors
//!
//! ferx's **analytic** `one_cpt_transit(cl, v, n, mtt)` closed form under:
//!   * a **continuous TV covariate** on CL — `CL = TVCL·(WT/70)^dWTCL·exp(η)`, `dWTCL` estimated;
//!   * **multiple overlapping doses** — 100 mg q6h × 5. With `MTT = 3 h` (density tail to ≈ 8 h)
//!     and `ke = CL/V ≈ 0.15/h` (t½ ≈ 4.6 h, accumulation ratio ≈ 1.7 at q6h), each dose's
//!     absorption tail AND elimination overlap the next — so the closed-form **superposition**
//!     across doses is genuinely exercised, not a sequence of washed-out single doses.
//! No lagtime/F, so ferx stays on the analytic closed-form path (the ODE-twin path with lag/F is
//! anchored separately by `gen_transit_multidose_lag_f_anchor.rs`).
//!
//! ## Why an explicit integer-N chain on the NONMEM side
//!
//! For integer `n`, the Savic density `R_in(t) = D·ktr·(ktr·t)^n·e^{-ktr·t}/n!`, `ktr=(n+1)/MTT`,
//! is *exactly* the input rate into central from an Erlang chain of `n+1` first-order transit
//! compartments (dose in the first, flux from the `(n+1)`-th into central, every transfer rate
//! `ktr`). For `n = 3` that is 4 transit compartments + central. NONMEM integrates that chain with
//! its native multi-dose bookkeeping, so **superposition across the five doses is automatic** — no
//! hand-coded `$DES` sum over `PODO`/`TDOS` (which sees only the most-recent dose). ferx's
//! single-compartment `one_cpt_transit` with `n = 3` FIXed produces the identical central profile,
//! so the two are likelihood-identical up to the obs-CMT re-key.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, Population};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions};
use std::fmt::Write as _;

mod common;

const DOSE_TIMES: [f64; 5] = [0.0, 6.0, 12.0, 18.0, 24.0]; // 100 mg q6h × 5 (overlapping)
const TADS: [f64; 5] = [0.5, 1.0, 2.0, 4.0, 5.5]; // obs within each 6 h interval
const N_SUBJECTS: usize = 24;
const DOSE: f64 = 100.0;

/// Per-subject weight, a deterministic **42–134 kg** spread (no RNG — covariates are design,
/// not simulated). The wide range matters: the WT-on-CL power `dWTCL = 0.75` must be *identifiable*
/// against IIV. Over 42–134 kg the WT term `0.75·log(WT/70)` spans ≈ ±0.44 in log-CL (SD ≈ 0.25
/// across the uniform spread), which must dominate the CL IIV (here SD 0.10, ω² = 0.01) for the
/// covariate to be recovered rather than absorbed into η — a narrow range or large IIV leaves
/// `dWTCL` swamped and both engines drift it toward 0 (a weak anchor that never exercises a
/// non-trivial covariate at the optimum). The engine-agreement claim holds regardless; this design
/// choice is what makes the anchor *also* recover the simulated covariate.
fn weight_for(id: usize) -> f64 {
    42.0 + 4.0 * ((id - 1) % 24) as f64
}

fn model() -> ferx_core::types::CompiledModel {
    parse_model_string(
        r"
[parameters]
  theta TVCL(9.0, 0.1, 100.0)
  theta TVV(60.0, 1.0, 500.0)
  theta TVMTT(3.0, 0.05, 24.0)
  theta TVN(3.0, FIX)
  theta DWTCL(0.75, 0.0, 2.0)
  omega ETA_CL ~ 0.01
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * (WT/70.0)^DWTCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  MTT = TVMTT
  NTR = TVN
[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)
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

/// One subject: a 100 mg dose at each of five q6h times (all CMT 1 — the transit input), five
/// post-dose observations per interval. Weight is attached as a time-constant covariate.
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
fn generate_transit_multidose_cov_anchor() {
    let model = model();
    let design = Population {
        subjects: (1..=N_SUBJECTS).map(build_subject).collect(),
        covariate_names: vec!["WT".to_string()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    // Simulate one replicate (BSV on CL & V + proportional residual) from the analytic model.
    let sims = simulate_with_seed(&model, &design, &model.default_params, 1, 719_611);
    let mut pop = design.clone();
    for subj in pop.subjects.iter_mut() {
        subj.observations = sims
            .iter()
            .filter(|r| r.id == subj.id)
            .map(|r| r.outcome.continuous_value())
            .collect();
    }

    // Two NONMEM-format CSVs, records time-sorted per subject (each dose then its interval's obs).
    // NONMEM sees obs on CMT 5 (explicit chain's central); ferx sees obs on CMT 1 (its single
    // disposition compartment). Identical DV; only the obs-CMT label differs.
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
    std::fs::write("nonmem_anchor/transit_multidose_cov.csv", csv_nm).unwrap();
    std::fs::create_dir_all("data").unwrap();
    std::fs::write("data/transit_multidose_cov.csv", csv_ferx).unwrap();
    println!(
        "WROTE nonmem_anchor/transit_multidose_cov.csv + data/transit_multidose_cov.csv ({} subj, {} doses each)",
        pop.subjects.len(),
        DOSE_TIMES.len()
    );

    // ferx FOCEI fit on the same data (analytic one_cpt_transit + WT covariate + multi-dose).
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.methods = vec![];
    opts.outer_maxiter = 30; // bounded cross-check; the committed anchor is the objective-at-optimum test
    opts.run_covariance_step = false;
    opts.ode_reltol = 1e-9;
    opts.ode_abstol = 1e-9;
    let r = fit(&model, &pop, &model.default_params, &opts).expect("transit+cov+multidose fit");
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
