//! Tier 2 for the DCM export (#1331): the public `predict()` / `simulate()`
//! entry points on a fitted DCM and on its plain-equation export agree record
//! for record, bit for bit.
//!
//! The unit tests in `src/nn/export_tests.rs` compare `pk_param_fn` at a grid
//! of covariates; this runs the whole pipeline a user runs — data read from a
//! CSV, dosing, the structural model, η drawn by the simulator — on a dataset
//! built to keep every path of the network live:
//!
//! - subject 1 has a **time-varying** network input (`WT` changes between
//!   doses), so the per-event evaluation is exercised, not just the
//!   subject-static one (#1300);
//! - subject 2 **misses** a network input (`CRCL` blank), which the network
//!   reads as `NaN`; the export must give `NaN` there too, not a finite value;
//! - subject 3 is a plain complete subject.

#![cfg(feature = "nn")]

use ferx_core::nn::export::{export_dcm_as_analytic, DcmExportOptions};
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{CompiledModel, ModelParameters};
use ferx_core::{predict, simulate_with_seed};

const MODEL: &str = include_str!("data/dcm_export/model.ferx");
const THETA: &str = include_str!("data/dcm_export/theta.csv");

const DATA: &str = "\
ID,TIME,DV,AMT,EVID,MDV,CMT,WT,CRCL
1,0,.,100,1,1,1,55,90
1,1,1.0,0,0,0,1,55,90
1,4,1.0,0,0,0,1,55,90
1,12,.,100,1,1,1,95,90
1,13,1.0,0,0,0,1,95,90
1,20,1.0,0,0,0,1,95,90
2,0,.,100,1,1,1,70,.
2,2,1.0,0,0,0,1,70,.
2,8,1.0,0,0,0,1,70,.
3,0,.,100,1,1,1,82,120
3,2,1.0,0,0,0,1,82,120
3,8,1.0,0,0,0,1,82,120
";

/// The DCM at the fixture's weights, and its export.
fn models() -> (
    CompiledModel,
    ModelParameters,
    CompiledModel,
    ModelParameters,
) {
    let dcm = parse_model_string(MODEL).expect("DCM parses under nn");
    let mut dcm_params = dcm.default_params.clone();
    let (names, theta): (Vec<String>, Vec<f64>) = THETA
        .lines()
        .skip(1)
        .map(|l| {
            let (n, v) = l.split_once(',').unwrap();
            (n.to_string(), v.parse::<f64>().unwrap())
        })
        .unzip();
    for (n, v) in names.iter().zip(&theta) {
        let i = dcm_params.theta_names.iter().position(|t| t == n).unwrap();
        dcm_params.theta[i] = *v;
    }
    let text = export_dcm_as_analytic(
        MODEL,
        &dcm_params.theta_names,
        &dcm_params.theta,
        &DcmExportOptions::default(),
    )
    .unwrap();
    let plain = parse_model_string(&text).unwrap();
    // The export rewrites only the network; every other θ keeps the model file's
    // value, so carry the DCM's (`TVV`) across by name for the comparison.
    let mut plain_params = plain.default_params.clone();
    for (i, n) in plain_params.theta_names.clone().iter().enumerate() {
        if let Some(j) = dcm_params.theta_names.iter().position(|t| t == n) {
            plain_params.theta[i] = dcm_params.theta[j];
        }
    }
    (dcm, dcm_params, plain, plain_params)
}

fn population() -> ferx_core::types::Population {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.csv");
    std::fs::write(&path, DATA).unwrap();
    ferx_core::io::datareader::read_nonmem_csv(&path, Some(&["WT", "CRCL"]), None).unwrap()
}

/// Bitwise equality that treats two `NaN`s as equal (the missing-input subject).
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[test]
fn predict_on_the_export_equals_predict_on_the_dcm() {
    let (dcm, dcm_params, plain, plain_params) = models();
    let pop = population();
    let a = predict(&dcm, &pop, &dcm_params).unwrap();
    let b = predict(&plain, &pop, &plain_params).unwrap();
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), 8);
    let mut n_nan = 0;
    for (x, y) in a.iter().zip(&b) {
        assert_eq!((&x.id, x.time), (&y.id, y.time));
        assert!(
            same(x.pred, y.pred),
            "id {} t {}: {} vs {}",
            x.id,
            x.time,
            x.pred,
            y.pred
        );
        if x.id == "2" {
            assert!(x.pred.is_nan(), "a missing network input is NaN in the DCM");
            n_nan += 1;
        } else {
            assert!(x.pred.is_finite() && x.pred > 0.0);
        }
    }
    assert_eq!(n_nan, 2);
    // The time-varying input moves the prediction: subject 1's two dosing
    // intervals differ in WT, so the per-event network evaluation is live.
    let s1: Vec<f64> = a.iter().filter(|r| r.id == "1").map(|r| r.pred).collect();
    let dcm_static = {
        // Subject 1 at WT = 55 throughout (only subject 1 carries 95).
        let static_data = DATA.replace(",95,90", ",55,90");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.csv");
        std::fs::write(&path, static_data).unwrap();
        let pop =
            ferx_core::io::datareader::read_nonmem_csv(&path, Some(&["WT", "CRCL"]), None).unwrap();
        predict(&dcm, &pop, &dcm_params).unwrap()
    };
    let s1_static: Vec<f64> = dcm_static
        .iter()
        .filter(|r| r.id == "1")
        .map(|r| r.pred)
        .collect();
    assert_ne!(
        s1[3], s1_static[3],
        "WT must change subject 1's second interval"
    );
}

/// At η drawn by the simulator (the same seed, the same Ω, so the same draws),
/// the individual predictions agree record for record.
#[test]
fn simulated_ipred_on_the_export_equals_the_dcm() {
    let (dcm, dcm_params, plain, plain_params) = models();
    let pop = population();
    let a = simulate_with_seed(&dcm, &pop, &dcm_params, 3, 1331).unwrap();
    let b = simulate_with_seed(&plain, &pop, &plain_params, 3, 1331).unwrap();
    assert_eq!(a.len(), b.len());
    assert!(!a.is_empty());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!((&x.id, x.sim, x.time), (&y.id, y.sim, y.time));
        assert!(
            same(x.ipred, y.ipred),
            "id {} t {}: {} vs {}",
            x.id,
            x.time,
            x.ipred,
            y.ipred
        );
    }
}
