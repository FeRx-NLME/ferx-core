//! NONMEM anchor for IMPMAP estimation of an inter-occasion variability model
//! (#2301): warfarin with IOV on CL, fitted by `method = impmap` in ferx and by
//! `$EST METHOD=IMPMAP` in NONMEM 7.6.0 from the same initial estimates.
//!
//! ## Reference
//!
//! `nonmem_anchor/warfarin_iov_impmap.ctl` on `data/warfarin_iov.csv` (identical
//! to `tests/nonmem/warfarin_iov.csv`), `NITER=200 ISAMPLE=300 SEED=12345
//! PRINT=1`; outputs in `nonmem_anchor/results/warfarin_iov_impmap.{ext,lst}`.
//! The same container reproduces the committed 7.5.1 FOCEI reference
//! (`tests/nonmem/warfarin_iov.ctl`, OFV 308.830) to the printed digits.
//!
//! ferx reports the mean of its last `impmap_averaging = 50` iterates; NONMEM
//! reports its last iterate, which carries the full single-iteration
//! Monte-Carlo noise (its final ω²_KA is 0.0289, its last-50 mean 0.0265). So
//! the reference here is NONMEM's **last-50 mean**, iterations 151–200 of the
//! `.ext`, and each bound is a multiple of NONMEM's own last-50 standard
//! deviation — the scale on which two MCEM runs can be told apart at all.
//!
//! Measured (Linux aarch64 via `tools/linux-test.sh`, and macOS arm64 — the same
//! in every printed digit), ferx vs that mean, in NONMEM SDs:
//! TVCL +0.15, TVV +0.05, TVKA +0.20, ω²_CL −0.53, ω²_V +0.00, ω²_KA +0.02,
//! **Ω_IOV +0.06**, σ² +0.06, IS −2LL −0.83. The worst is 0.83 of a bound of
//! `Z_BOUND = 3`.
//!
//! Non-degeneracy: Ω_IOV starts at 0.01 and lands at 0.0354, 34 NONMEM SDs
//! away, so an Ω_IOV M-step that never moves it fails by an order of magnitude.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::types::KappaTreatment;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, FitResult};
use std::path::Path;

/// `(name, NONMEM last-50 mean, NONMEM last-50 SD)` from
/// `nonmem_anchor/results/warfarin_iov_impmap.ext`, iterations 151–200.
const NM_TVCL: (&str, f64, f64) = ("TVCL", 0.170_412_48, 0.006_132_94);
const NM_TVV: (&str, f64, f64) = ("TVV", 8.616_881_6, 0.079_097);
const NM_TVKA: (&str, f64, f64) = ("TVKA", 1.191_451_2, 0.014_476_7);
const NM_OM_CL: (&str, f64, f64) = ("omega ETA_CL", 0.042_146_216, 0.002_479_49);
const NM_OM_V: (&str, f64, f64) = ("omega ETA_V", 0.010_553_698, 0.000_245_52);
const NM_OM_KA: (&str, f64, f64) = ("omega ETA_KA", 0.026_538_932, 0.001_236_15);
const NM_OM_IOV: (&str, f64, f64) = ("omega_iov KAPPA_CL", 0.035_404_076, 0.000_748_445);
const NM_SIGMA2: (&str, f64, f64) = ("sigma² PROP_ERR", 0.035_304_198, 0.000_161_983);
const NM_OBJ: (&str, f64, f64) = ("IS -2LL", 310.316_360_7, 0.806_158);

/// Bound, in NONMEM last-50 SDs. The measured worst is 0.83 (see the module
/// docs), so this leaves 3.6× headroom.
const Z_BOUND: f64 = 3.0;

/// Ω_IOV's initial estimate in both control streams.
const OMEGA_IOV_START: f64 = 0.01;

fn fit_impmap() -> FitResult {
    let model = parse_model_file(Path::new("examples/warfarin_iov.ferx"))
        .expect("warfarin_iov model must parse");
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data must load");
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Impmap;
    opts.impmap_iterations = 200;
    opts.impmap_samples = 300;
    opts.impmap_seed = Some(12345);
    opts.run_covariance_step = false;
    opts.verbose = false;
    fit(&model, &pop, &model.default_params, &opts).expect("IMPMAP on an IOV model must fit")
}

fn assert_within(got: f64, (name, mean, sd): (&str, f64, f64)) {
    assert!(
        got.is_finite(),
        "{name}: ferx estimate is not finite ({got})"
    );
    let z = (got - mean) / sd;
    eprintln!("{name:20} ferx {got:.6}  NONMEM mean50 {mean:.6}  z {z:+.2}");
    assert!(
        z.abs() <= Z_BOUND,
        "{name}: ferx {got:.6} is {z:+.2} NONMEM SDs from NONMEM's IMPMAP last-50 mean \
         {mean:.6} (SD {sd:.3e}); bound ±{Z_BOUND}"
    );
}

fn theta(r: &FitResult, name: &str) -> f64 {
    let i = r.theta_names.iter().position(|n| n == name).unwrap();
    r.theta[i]
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored IMPMAP with IOV (#2301): opt in with --features slow-tests"
)]
fn impmap_iov_matches_nonmem_impmap() {
    let r = fit_impmap();

    assert_within(theta(&r, "TVCL"), NM_TVCL);
    assert_within(theta(&r, "TVV"), NM_TVV);
    assert_within(theta(&r, "TVKA"), NM_TVKA);
    assert_within(r.omega[(0, 0)], NM_OM_CL);
    assert_within(r.omega[(1, 1)], NM_OM_V);
    assert_within(r.omega[(2, 2)], NM_OM_KA);
    assert_within(r.sigma[0] * r.sigma[0], NM_SIGMA2);

    let iov = r
        .omega_iov
        .as_ref()
        .expect("an IMPMAP IOV fit must report Ω_IOV");
    let om_iov = iov[(0, 0)];
    assert_within(om_iov, NM_OM_IOV);
    assert!(
        om_iov - OMEGA_IOV_START > 10.0 * NM_OM_IOV.2,
        "Ω_IOV must leave its start {OMEGA_IOV_START}: got {om_iov}"
    );

    let is = r
        .importance_sampling
        .as_ref()
        .expect("the final IMPMAP stage evaluates the IS objective");
    assert!(matches!(is.kappa_treatment, KappaTreatment::Marginalized));
    assert_within(is.minus2_log_likelihood, NM_OBJ);
}
