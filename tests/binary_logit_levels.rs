//! A θ level block read by a `[binary_model]` predictor (#2562), through the public `fit()`:
//! the NONMEM anchor A-B1. Every fit is an evaluation (`outer_maxiter = 0`), so the file is
//! Tier 2.
//!
//! **A-B1.** `nonmem_anchor/profile_logit_b1_laplace.ctl` codes B1 (2400 Bernoulli rows,
//! `logit p = TVL + PLACEBO[study, visit] + η + EMAX·TRT`, ID = study) with every level a
//! `THETA`, each study's last visit dependent as ferx lays it out, under
//! `METHOD=COND LAPLACE LIKE`. ferx is handed NONMEM's whole optimum — the 32 θ and Ω — and
//! evaluates the objective once. Each record is scored at its own level (the visit) and its
//! own arm (`TRT` changes within each study), so a record read at its subject's baseline
//! moves the objective by hundreds.
//!
//! ferx's FOCE and FOCEI on a binary row are the Laplace form (the exact discrete
//! η-Hessian, #2437; the `reset_clock_cat` anchor), so all three methods are held to the
//! one NONMEM objective.

#![cfg(feature = "survival")]

use ferx_core::types::{EstimationMethod, FitOptions, OmegaMatrix};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The final row of `nonmem_anchor/results/profile_logit_b1_laplace.ext`: θ in order, Ω,
/// and the objective.
fn nonmem_optimum() -> (Vec<f64>, f64, f64) {
    let text =
        std::fs::read_to_string(root().join("nonmem_anchor/results/profile_logit_b1_laplace.ext"))
            .unwrap();
    let mut names: Vec<String> = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.first() {
            Some(&"ITERATION") => names = fields.iter().map(|s| s.to_string()).collect(),
            Some(&"-1000000000") => {
                let get = |name: &str| -> f64 {
                    let i = names.iter().position(|n| n == name).unwrap();
                    fields[i].parse().unwrap()
                };
                let theta = (1..=32).map(|k| get(&format!("THETA{k}"))).collect();
                return (theta, get("OMEGA(1,1)"), get("OBJ"));
            }
            _ => {}
        }
    }
    panic!("profile_logit_b1_laplace.ext: no final row");
}

/// |ΔOFV| bound. Measured 1.42e-9 under all three methods, macOS arm64 and Linux aarch64
/// alike — the gap of an independent Python Laplace at the same point (1.46e-9), so it is
/// NONMEM's printed precision, not ferx's. The bound sits 35× above it. A record read at
/// its subject's baseline level or arm moves the objective by far more.
const A_B1_OFV: f64 = 5e-8;

fn evaluate(method: EstimationMethod, interaction: bool) -> (f64, f64, Vec<String>) {
    let p = ferx_core::prepare_run(
        root()
            .join("nonmem_anchor/profile_logit_b1.ferx")
            .to_str()
            .unwrap(),
        Some(
            root()
                .join("nonmem_anchor/profile_logit_b1.csv")
                .to_str()
                .unwrap(),
        ),
    )
    .unwrap_or_else(|e| panic!("B1 loads: {e}"));
    assert_eq!(
        p.parsed.model.theta_names.len(),
        32,
        "TVL, EMAX and 30 free levels, NONMEM's THETA order"
    );
    assert_eq!(p.parsed.model.theta_names[2], "PLACEBO[STUDY=1,TIME=0]");
    assert_eq!(p.parsed.model.theta_names[31], "PLACEBO[STUDY=10,TIME=8]");
    let (theta, omega, obj) = nonmem_optimum();
    let mut params = p.init_params.clone();
    params.theta = theta;
    params.omega = OmegaMatrix::from_diagonal(&[omega], vec!["ETA_S".into()]);
    let options = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: false,
        inner_tol: 1e-8,
        method,
        interaction,
        ..p.parsed.fit_options.clone()
    };
    let f = ferx_core::fit(&p.parsed.model, &p.population, &params, &options)
        .unwrap_or_else(|e| panic!("{method:?}: {e}"));
    (f.ofv, obj, f.warnings)
}

/// A-B1 under `method`: the objective at NONMEM's optimum, on the exact discrete
/// η-Hessian and gradient (a finite-difference warning would mean the predictor's
/// per-record read left the dual path).
fn assert_reproduces(method: EstimationMethod, interaction: bool) {
    let (ofv, obj, warnings) = evaluate(method, interaction);
    let d = ofv - obj;
    eprintln!("A-B1 {method:?}: ferx {ofv:.10} NONMEM {obj:.10} ΔOFV {d:.3e}");
    assert!(ofv.is_finite(), "{method:?}: OFV {ofv}");
    assert!(d.abs() <= A_B1_OFV, "{method:?}: |ΔOFV| {:.3e}", d.abs());
    assert!(
        !warnings
            .iter()
            .any(|w| w.contains("finite difference") || w.contains("finite-difference")),
        "{method:?}: {warnings:?}"
    );
}

#[test]
fn binary_levels_reproduce_nonmem_laplace() {
    assert_reproduces(EstimationMethod::Laplace, true);
}

#[test]
fn binary_levels_reproduce_nonmem_under_focei() {
    assert_reproduces(EstimationMethod::FoceI, true);
}

#[test]
fn binary_levels_reproduce_nonmem_under_foce() {
    assert_reproduces(EstimationMethod::Foce, false);
}
