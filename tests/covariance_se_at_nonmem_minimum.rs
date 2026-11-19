//! Regression guard for issue #432: ferx's FOCEI covariance step reproduces
//! NONMEM `$COVARIANCE MATRIX=R` **at NONMEM's own minimum**.
//!
//! Background: on the weakly-identified warfarin block-omega model, ferx's
//! reported `SE(ω²KA)` (at ferx's optimum) differs ~25% from NONMEM's — *not*
//! because the covariance computation is wrong, but because the ω²KA direction
//! is extremely flat and the two engines settle on optima ~0.1% apart (NONMEM's
//! run flagged "PROBLEMS OCCURRED WITH THE MINIMIZATION"; ferx descends a further
//! 2.8 OFV to the true minimum — verified genuine via the IMP marginal). The SE
//! is hyper-sensitive to that sub-percent location difference.
//!
//! This test removes the location confound: it **pins the parameters to NONMEM's
//! exact FOCEI estimates** (`outer_maxiter = 0`, no re-optimization) and asserts
//! that ferx's covariance SEs — including the flat ω²KA — match NONMEM's
//! `MATRIX=R` values *tightly*. At the identical point the two agree to <1%,
//! which is the proof that the covariance/Hessian is correct (see #432). The
//! `BLOCK(2)` on (CL,V) + diagonal KA layout means `se_omega` is the full
//! column-major lower triangle, so diagonals are read via `omega_se_at` (#226).
//!
//! NONMEM 7.5.1 reference: `tests/nonmem/` block-omega FOCEI run, `$EST METHOD=1
//! INTERACTION`, `$COV MATRIX=R`; estimates and SEs from the `.ext` rows at
//! ITERATION = -1000000000 / -1000000001.

use ferx_core::types::omega_se_at;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, Optimizer};
use std::path::Path;

/// Model initialized **at NONMEM's FOCEI minimum** (block-omega warfarin). THETA(4)
/// is the proportional SD (`W = THETA4·IPRED`, `$SIGMA 1 FIX`), matching ferx's
/// `sigma PROP_ERR (sd)`.
const NM_MIN_SRC: &str = r"
[parameters]
  theta TVCL(0.132696, 0.001, 10.0)
  theta TVV(7.73752, 0.1, 500.0)
  theta TVKA(0.810794, 0.01, 50.0)
  block_omega (ETA_CL, ETA_V) = [0.0285938, 0.00190087, 0.00959304]
  omega ETA_KA ~ 0.335846
  sigma PROP_ERR ~ 0.010565 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method        = focei
  mu_referencing = true
";

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored covariance cross-check (#432): opt in with --features slow-tests"
)]
fn focei_covariance_se_matches_nonmem_at_its_own_minimum() {
    let model = ferx_core::parser::model_parser::parse_model_string(NM_MIN_SRC)
        .expect("NONMEM-minimum block-omega model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin_block_omega.csv"), None, None)
        .expect("block-omega data loads");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true;
    opts.optimizer = Optimizer::Lbfgs;
    // PIN to NONMEM's minimum — no re-optimization — so the covariance is
    // evaluated at the exact point NONMEM reported (the crux of #432).
    opts.outer_maxiter = 0;
    opts.run_covariance_step = true;
    opts.inner_tol = 1e-8; // sharp inner mode so the SEs are fully converged
    opts.inner_maxiter = 500;
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("pinned cov fit runs");

    // Parameters must not have moved (outer_maxiter = 0).
    assert!(
        (result.theta[2] - 0.810794).abs() < 1e-3 && (result.omega[(2, 2)] - 0.335846).abs() < 1e-3,
        "params should stay pinned at NONMEM's minimum (TVKA {}, omega_KA {})",
        result.theta[2],
        result.omega[(2, 2)]
    );

    assert!(result.covariance_matrix.is_some(), "covariance step must succeed");
    let se_theta = result.se_theta.as_ref().expect("theta SEs present");
    let se_sigma = result.se_sigma.as_ref().expect("sigma SEs present");
    let omega_se =
        |i: usize| omega_se_at(&result.se_omega, 3, i, i).expect("omega diag SE present");

    // NONMEM 7.5.1 FOCEI MATRIX=R SE row (.ext, ITER=-1000000001) at this minimum.
    // At the identical point ferx reproduces these to <1%; a 6% band absorbs any
    // platform/profile FD-step difference while still proving the SE is correct
    // (the flat ω²KA would be ~25% off if evaluated at ferx's own optimum).
    let band = 0.06;
    let checks: [(&str, f64, f64); 7] = [
        ("TVCL", se_theta[0], 7.09762e-3),
        ("TVV", se_theta[1], 2.40059e-1),
        ("TVKA", se_theta[2], 1.48631e-1),
        ("PROP_ERR", se_sigma[0], 8.35394e-4),
        ("omega_CL", omega_se(0), 1.27953e-2),
        ("omega_V", omega_se(1), 4.30584e-3),
        ("omega_KA", omega_se(2), 1.50360e-1),
    ];
    for (name, ferx_se, nm) in checks {
        let rel = (ferx_se - nm).abs() / nm;
        assert!(
            ferx_se.is_finite() && rel < band,
            "SE({name}) = {ferx_se:.6} vs NONMEM {nm:.6} — rel diff {:.1}% exceeds {:.0}% \
             (covariance must match NONMEM at the SAME point; #432)",
            rel * 100.0,
            band * 100.0
        );
    }
}
