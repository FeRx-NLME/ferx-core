//! NONMEM 7.5.1 IMP cross-check for the warfarin IOV joint-sampling path.
//!
//! Validates that ferx's IS -2LL evaluated at NONMEM's FOCEI MLE is close
//! to NONMEM's own IMP reference on the same model/data, confirming that
//! joint (η, κ) sampling produces a marginal likelihood comparable to
//! NONMEM `$EST METHOD=IMP LAPLACIAN=1`.
//!
//! ## Reference values
//!
//! All from `tests/nonmem/warfarin_iov_imp.ctl` and `tests/nonmem/warfarin_iov.ctl`
//! run on `data/warfarin_iov.csv` with NONMEM 7.5.1:
//!
//! | Engine  | Method | OFV (without constant) |
//! |---------|--------|-------------------------|
//! | NONMEM  | FOCEI  | 308.83                  |
//! | NONMEM  | IMP    | 310.18                  |
//! | ferx    | FOCEI  | 307.84                  |
//! | ferx    | IMP    | 309.00  (MC SE 0.014)   |
//!
//! NONMEM's FOCEI and IMP MLEs are close but not identical. The fixed-param
//! test below evaluates the IS -2LL at NONMEM's **FOCEI** MLE. Since the two
//! MLEs are nearby on the likelihood surface, we expect the IMP -2LL at
//! NONMEM's FOCEI MLE to fall between NONMEM's FOCEI (308.83) and IMP (310.18)
//! plus a small cross-engine margin — empirically around [307, 313].
//!
//! The free-fit test runs ferx's full SAEM → FOCEI → IMP chain and validates
//! against the NONMEM IMP reference of 310.18.
//!
//! ## Control streams
//!
//! - `tests/nonmem/warfarin_iov.ctl`     — FOCEI reference
//! - `tests/nonmem/warfarin_iov_imp.ctl` — IMP reference (METHOD=IMP INTER)

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::KappaTreatment;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

// NONMEM 7.5.1 FOCEI MLE — same constants as warfarin_iov_nonmem.rs.
const NM_TVCL: f64 = 0.172776;
const NM_TVV: f64 = 8.62821;
const NM_TVKA: f64 = 1.17856;
const NM_OMEGA_CL: f64 = 0.0399349;
const NM_OMEGA_V: f64 = 0.0107782;
const NM_OMEGA_KA: f64 = 0.0254197;
const NM_OMEGA_IOV: f64 = 0.0357084;
const NM_SIGMA_PROP_SD: f64 = 0.188116; // sqrt(0.0353877)

// NONMEM 7.5.1 reference OFVs (without constant).
const NM_IMP_OFV: f64 = 310.18;

/// Evaluate ferx IS -2LL at NONMEM's FOCEI MLE (all params FIXed).
///
/// Fixes θ/Ω/σ at NONMEM's FOCEI MLE, runs FOCEI to converge EBEs at that
/// point, then runs IMP. Since all params are FIXed the outer optimizer takes
/// no steps — this is a pure objective evaluation at a known point.
///
/// At a given parameter point, IS-2LL and the FOCEI Laplace OFV should agree
/// closely (both estimate the same marginal likelihood). The Laplace
/// approximation error for this 10-subject IOV model is expected to be small,
/// so we assert IS-2LL ≈ FOCEI OFV ± 3 units. We also record the value
/// relative to NONMEM's FOCEI OFV (308.83) for cross-engine documentation.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn imp_at_nonmem_focei_mle_is_close_to_focei_at_same_point() {
    let fixed = format!(
        r"
[parameters]
  theta TVCL({NM_TVCL}, FIX)
  theta TVV({NM_TVV}, FIX)
  theta TVKA({NM_TVKA}, FIX)
  omega ETA_CL ~ {NM_OMEGA_CL} FIX
  omega ETA_V  ~ {NM_OMEGA_V} FIX
  omega ETA_KA ~ {NM_OMEGA_KA} FIX
  kappa KAPPA_CL ~ {NM_OMEGA_IOV} FIX
  sigma PROP_ERR ~ {prop} (sd) FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  iov_column = OCC
",
        prop = NM_SIGMA_PROP_SD,
    );

    let model = parse_model_string(&fixed).expect("fixed-param IOV model must parse");
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data must load");

    let mut opts = FitOptions::default();
    opts.methods = vec![EstimationMethod::FoceI, EstimationMethod::Imp];
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.is_samples = 3000;
    opts.is_seed = Some(2026);
    opts.is_eval_only = true; // IS IOV scoring path; estimating IMP refuses IOV

    let result = fit(&model, &pop, &model.default_params, &opts)
        .expect("fixed-param FOCEI → IMP must succeed");

    let imp = result
        .importance_sampling
        .as_ref()
        .expect("IMP stage must populate importance_sampling");

    assert!(
        imp.minus2_log_likelihood.is_finite(),
        "IS -2LL must be finite, got {}",
        imp.minus2_log_likelihood
    );
    assert!(
        matches!(imp.kappa_treatment, KappaTreatment::Marginalized),
        "kappa_treatment must be Marginalized for IOV model"
    );

    // IS-2LL and the FOCEI Laplace OFV at the same parameter point should
    // agree within the Laplace approximation error for this dataset. FOCEI OFV
    // at NONMEM's MLE ≈ 308.2 (ferx) — see warfarin_iov_nonmem.rs.
    let laplace_gap = (imp.minus2_log_likelihood - result.ofv).abs();
    assert!(
        laplace_gap < 3.0,
        "IS-2LL ({:.4}) vs FOCEI OFV ({:.4}) at NONMEM's MLE: gap {:.4} exceeds \
         expected Laplace approximation error of ±3 units for this IOV dataset",
        imp.minus2_log_likelihood,
        result.ofv,
        laplace_gap
    );

    // Cross-engine documentation: IS-2LL at NONMEM's FOCEI MLE vs references.
    // Not a hard assertion — just ensures the value is in a plausible range.
    // NONMEM FOCEI: 308.83 | NONMEM IMP: 310.18 | ferx FOCEI @NM-MLE: ~308.2
    assert!(
        (305.0..315.0).contains(&imp.minus2_log_likelihood),
        "IS-2LL {:.4} is outside the plausible [305, 315] range for warfarin IOV \
         at NONMEM's FOCEI MLE — check for a regression in the joint-sampling path",
        imp.minus2_log_likelihood
    );
}

/// Full SAEM → FOCEI → IMP chain: ferx finds its own MLE and reports an IS
/// -2LL that matches NONMEM's IMP reference within a cross-engine margin.
///
/// NONMEM 7.5.1 IMP at NONMEM's IMP MLE: 310.18.
/// ferx IMP at ferx's FOCEI MLE:          309.00 (MC SE 0.014).
/// The 1.18-unit gap is within the expected cross-engine spread for IOV
/// (compare the FOCEI gap: ferx 307.84 vs NONMEM 308.83 = 1.0 unit).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn imp_free_fit_matches_nonmem_reference() {
    let model = parse_model_file(Path::new("examples/warfarin_iov.ferx"))
        .expect("warfarin_iov model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data must load");

    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.methods = vec![
        EstimationMethod::Saem,
        EstimationMethod::FoceI,
        EstimationMethod::Imp,
    ];
    opts.is_samples = 3000;
    opts.is_seed = Some(2026);
    opts.is_eval_only = true; // IS IOV scoring path; estimating IMP refuses IOV

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("warfarin_iov SAEM → FOCEI → IMP must succeed");

    let imp = result
        .importance_sampling
        .as_ref()
        .expect("IMP stage must populate importance_sampling");

    assert!(
        matches!(imp.kappa_treatment, KappaTreatment::Marginalized),
        "kappa_treatment must be Marginalized for IOV model"
    );
    assert!(
        imp.mc_standard_error < 0.1,
        "MC SE should be small at K=3000, got {:.4}",
        imp.mc_standard_error
    );

    // ferx IMP reference: 309.00.  NONMEM IMP reference: 310.18.
    // Tolerance ±2 absorbs MC noise and platform-dependent FOCEI polish.
    let gap = (imp.minus2_log_likelihood - NM_IMP_OFV).abs();
    assert!(
        gap < 2.0,
        "ferx IMP -2LL = {:.4}; NONMEM IMP ref = {:.2}; \
         gap {:.4} exceeds ±2 cross-engine tolerance",
        imp.minus2_log_likelihood,
        NM_IMP_OFV,
        gap
    );
}
