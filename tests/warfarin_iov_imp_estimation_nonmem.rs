//! NONMEM anchor for IMP estimation of an inter-occasion variability model
//! (#2301, #2322 review finding 4): warfarin with IOV on CL, fitted by
//! `method = imp` in ferx and by `$EST METHOD=IMP` in NONMEM 7.6.0 from the same
//! initial estimates. The IMPMAP twin is `tests/warfarin_iov_impmap_nonmem.rs`;
//! this one reaches `imp`'s own IOV code, the sample-moment proposal recentre
//! (the stacked `S − m mᵀ` capped by the block-diagonal joint prior).
//! `tests/warfarin_iov_imp_nonmem.rs` is a different check: it scores the IS
//! objective at fixed parameters (`imp_eval_only`) and estimates nothing.
//!
//! ## Reference
//!
//! `nonmem_anchor/warfarin_iov_imp.ctl` on `data/warfarin_iov.csv`,
//! `NITER=200 ISAMPLE=300 PRINT=1`, run at five seeds: `SEED=12345` (outputs in
//! `nonmem_anchor/results/warfarin_iov_imp.{ext,lst}`) and 1001–1004
//! (`results/warfarin_iov_imp_seed100{1..4}.ext`). ferx reports the mean of its
//! last `imp_averaging = 50` iterates, so each NONMEM run is reduced to its
//! last-50 mean (iterations 151–200) and the reference is the mean of the five.
//!
//! **Why five seeds.** IMP re-centres each proposal on the previous
//! iteration's sample moments, so its iterates are strongly autocorrelated: at
//! `SEED=12345` σ²'s last-50 SD is 6.6e-5, yet its 25-iteration block means
//! wander 0.03517 → 0.03592 over iterations 51–200, and the five seeds' last-50
//! means spread with SD 2.4e-4. Scored against that one run's within-run SD,
//! ferx's σ² read as −7.3 SDs; against the five-seed mean it is −0.78. The scale
//! for each parameter is therefore the larger of NONMEM's mean within-run
//! last-50 SD (per-iteration noise) and the SD of the five last-50 means
//! (autocorrelated drift) — measured, both columns in the table below.
//!
//! Measured (Linux aarch64 via `tools/linux-test.sh` and macOS arm64, the same
//! in every printed digit), ferx vs the five-seed mean, in that scale:
//! TVCL +0.44, TVV −0.46, TVKA +0.10, ω²_CL −0.33, ω²_V −0.25, ω²_KA +0.26,
//! **Ω_IOV −0.12**, σ² −0.78, IS −2LL −0.74. The worst is 0.78 of a bound of
//! `Z_BOUND = 3`. ferx itself moves far less across seeds (12345, 1001–1004:
//! TVV within 8e-4, σ² within 7e-6), so the bound is NONMEM's spread.
//!
//! Non-degeneracy: Ω_IOV starts at 0.01 and NONMEM lands at 0.0355, 38 scale
//! units away, so an Ω_IOV M-step that never moves it fails by an order of
//! magnitude.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::types::KappaTreatment;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, FitResult};
use std::path::Path;

/// `(name, mean of the five NONMEM last-50 means, scale)`; the scale is
/// `max(mean within-run last-50 SD, SD of the five last-50 means)`.
///
/// | parameter | within | across |
/// |---|---|---|
/// | TVCL | 3.29e-3 | 1.42e-3 |
/// | TVV | 4.33e-2 | 8.0e-3 |
/// | TVKA | 8.81e-3 | 1.12e-3 |
/// | ω²_CL | 1.39e-3 | 2.35e-4 |
/// | ω²_V | 2.62e-4 | 6.4e-5 |
/// | ω²_KA | 1.31e-3 | 5.93e-4 |
/// | Ω_IOV | 6.64e-4 | 2.17e-4 |
/// | σ² | 1.36e-4 | **2.38e-4** |
/// | IS −2LL | 0.414 | 0.038 |
const NM_TVCL: (&str, f64, f64) = ("TVCL", 0.169_883_64, 0.003_292_58);
const NM_TVV: (&str, f64, f64) = ("TVV", 8.641_931_2, 0.043_344_4);
const NM_TVKA: (&str, f64, f64) = ("TVKA", 1.194_642_1, 0.008_805_97);
const NM_OM_CL: (&str, f64, f64) = ("omega ETA_CL", 0.041_280_905, 0.001_389_05);
const NM_OM_V: (&str, f64, f64) = ("omega ETA_V", 0.010_579_656, 0.000_262_189);
const NM_OM_KA: (&str, f64, f64) = ("omega ETA_KA", 0.026_601_621, 0.001_311_42);
const NM_OM_IOV: (&str, f64, f64) = ("omega_iov KAPPA_CL", 0.035_538_41, 0.000_664_259);
const NM_SIGMA2: (&str, f64, f64) = ("sigma² PROP_ERR", 0.035_494_471, 0.000_238_097);
const NM_OBJ: (&str, f64, f64) = ("IS -2LL", 310.178_424, 0.413_716);

/// Bound, in scale units. The measured worst is 0.78 (see the module docs), so
/// this leaves 3.8× headroom.
const Z_BOUND: f64 = 3.0;

/// Ω_IOV's initial estimate in both control streams.
const OMEGA_IOV_START: f64 = 0.01;

fn fit_imp() -> FitResult {
    let model = parse_model_file(Path::new("examples/warfarin_iov.ferx"))
        .expect("warfarin_iov model must parse");
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data must load");
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Imp;
    opts.imp_iterations = 200;
    opts.imp_samples = 300;
    opts.imp_seed = Some(12345);
    opts.run_covariance_step = false;
    opts.verbose = false;
    fit(&model, &pop, &model.default_params, &opts).expect("IMP on an IOV model must fit")
}

fn assert_within(got: f64, (name, mean, sd): (&str, f64, f64)) {
    assert!(
        got.is_finite(),
        "{name}: ferx estimate is not finite ({got})"
    );
    let z = (got - mean) / sd;
    eprintln!("{name:20} ferx {got:.6}  NONMEM 5-seed mean {mean:.6}  z {z:+.2}");
    assert!(
        z.abs() <= Z_BOUND,
        "{name}: ferx {got:.6} is {z:+.2} scale units from NONMEM's five-seed IMP mean \
         {mean:.6} (scale {sd:.3e}); bound ±{Z_BOUND}"
    );
}

fn theta(r: &FitResult, name: &str) -> f64 {
    let i = r.theta_names.iter().position(|n| n == name).unwrap();
    r.theta[i]
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored IMP with IOV (#2301): opt in with --features slow-tests"
)]
fn imp_iov_matches_nonmem_imp() {
    let r = fit_imp();

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
        .expect("an IMP IOV fit must report Ω_IOV");
    let om_iov = iov[(0, 0)];
    assert_within(om_iov, NM_OM_IOV);
    assert!(
        om_iov - OMEGA_IOV_START > 10.0 * NM_OM_IOV.2,
        "Ω_IOV must leave its start {OMEGA_IOV_START}: got {om_iov}"
    );

    let is = r
        .importance_sampling
        .as_ref()
        .expect("the final IMP stage evaluates the IS objective");
    assert!(matches!(is.kappa_treatment, KappaTreatment::Marginalized));
    assert_within(is.minus2_log_likelihood, NM_OBJ);
}
