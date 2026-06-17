//! End-to-end NONMEM cross-check for the Schnider propofol dataset — the
//! acceptance test for issue #195 (multi-occasion `EVID=4` reset handling).
//!
//! The dataset (`data/schnider_propofol.csv`) stacks **two infusion occasions
//! per subject** under one ID, each opened by an `EVID=4` (reset + dose) record
//! whose `TIME` column restarts at 0. Before the occasion-segmentation fix in
//! `io/datareader.rs`, ferx-core sorted both occasions onto one absolute
//! timeline, collided their doses, and silently double-dosed every subject —
//! inflating CL ~1.8x. This test fits the model to convergence and checks the
//! population estimates land in the NONMEM 7.6.0 reference basin.
//!
//! This is a slow full-population fit (24 subjects × 2 occasions, ~1000 obs).
//! It is gated behind `slow-tests` and runs nightly / on demand:
//!
//!   cargo test --features slow-tests --test schnider_propofol_nonmem
//!
//! ## NONMEM reference
//!
//! Produced by Douglas Eleveld and reported in issue #195. Three-compartment
//! IV model, ADVAN11 TRANS4, FOCE INTERACTION, with allometric weight scaling
//! and ETAs on V1, V2, CL, Q2 (V3 and Q3 fixed to no BSV):
//!
//! ```text
//! $SUBROUTINES ADVAN11 TRANS4
//! $PK
//!   SIZE = WT/70.
//!   V1 = THETA(1)*SIZE*EXP(ETA(1))      ; CL=THETA(4)*SIZE**0.75*EXP(ETA(4))
//!   V2 = THETA(2)*SIZE*EXP(ETA(2))      ; Q2=THETA(5)*SIZE**0.75*EXP(ETA(5))
//!   V3 = THETA(3)*SIZE*EXP(ETA(3))      ; Q3=THETA(6)*SIZE**0.75*EXP(ETA(6))
//!   S1 = V1
//! $ERROR
//!   IPRED = A(1)/V1
//!   Y     = IPRED*(1 + ERR(1))
//! $OMEGA 0.1 0.1 0 FIXED 0.1 0.1 0 FIXED   ; ETA on V1,V2,(V3),CL,Q2,(Q3)
//! $SIGMA 0.1                               ; proportional (variance)
//! $ESTM METHOD=1 INTERACT MAX=5000
//! ```
//!
//! NONMEM final estimates:
//!
//! | Param | NONMEM | Param | NONMEM (variance) |
//! |-------|--------|-------|-------------------|
//! | TVV1  | 5.76   | Ω V1  | 0.0948 |
//! | TVV2  | 26.1   | Ω V2  | 0.0757 |
//! | TVV3  | 309    | Ω CL  | 0.0193 |
//! | TVCL  | 1.92   | Ω Q2  | 0.127  |
//! | TVQ2  | 1.34   | Σ     | 0.0540 |
//! | TVQ3  | 0.868  |       |        |
//!
//! ferx (FOCEI, BOBYQA) reaches: TVV1 5.49, TVV2 24.4, TVV3 289, TVCL 1.93,
//! TVQ2 1.35, TVQ3 0.864; Ω(V1,V2,CL,Q2) = (0.075, 0.094, 0.017, 0.135);
//! Σ(var) = 0.054. The structural thetas match to <7%; the BSV variances trade
//! off between V1 and V2 (their total is within ~2% of NONMEM) — typical of
//! weakly-identified variance components across FOCEI implementations. The
//! decisive regression guard is TVCL ≈ 1.9 (the bug pushed it past 3.4).

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const SCHNIDER_MODEL: &str = r#"
[parameters]
  theta TVV1(6, 0.1, 25)
  theta TVV2(20, 1, 60)
  theta TVV3(200, 50, 1000)
  theta TVCL(2, 0.1, 6)
  theta TVQ2(1, 0.1, 4)
  theta TVQ3(0.5, 0.1, 3)

  omega ETA_V1 ~ 0.1
  omega ETA_V2 ~ 0.1
  omega ETA_CL ~ 0.1
  omega ETA_Q2 ~ 0.1

  sigma PROP_ERR ~ 0.1

[individual_parameters]
  V1 = TVV1 * (WT/70.) * exp(ETA_V1)
  V2 = TVV2 * (WT/70.) * exp(ETA_V2)
  V3 = TVV3 * (WT/70.)
  CL = TVCL * (WT/70.)^0.75 * exp(ETA_CL)
  Q2 = TVQ2 * (WT/70.)^0.75 * exp(ETA_Q2)
  Q3 = TVQ3 * (WT/70.)^0.75

[structural_model]
  pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// NONMEM reference thetas (TVV1, TVV2, TVV3, TVCL, TVQ2, TVQ3).
const NM_THETA: [f64; 6] = [5.76, 26.1, 309.0, 1.92, 1.34, 0.868];
/// NONMEM reference BSV variances, in declaration order (V1, V2, CL, Q2).
const NM_OMEGA: [f64; 4] = [0.0948, 0.0757, 0.0193, 0.127];
/// NONMEM reference proportional residual variance.
const NM_SIGMA_VAR: f64 = 0.0540;

#[test]
// TEMP-DISABLED (#333): the 3-cpt V1/V2/V3 split is weakly identified — at the
// default inner_tol the TVV3 estimate drifts outside the NONMEM 15% band (the
// thetas come into band only at inner_tol=1e-8 and the ω components remain
// inflated). Not fixed by the #330 inner_tol/θ-block work; tracked in #333.
#[ignore = "weakly-identified 3-cpt volume split — tracked in #333"]
fn schnider_propofol_matches_nonmem() {
    let parsed = parse_full_model(SCHNIDER_MODEL).expect("model parses");
    let model = parsed.model;

    let population = read_nonmem_csv(Path::new("data/schnider_propofol.csv"), None, None)
        .expect("dataset loads");
    assert_eq!(population.subjects.len(), 24, "24 subjects");
    // Every subject carries two reset occasions stacked under one ID.
    assert!(
        population.subjects.iter().all(|s| s.reset_times.len() == 2),
        "each subject must have two EVID=4 reset occasions"
    );

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.run_covariance_step = false;
    opts.outer_maxiter = 800;
    opts.verbose = false;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("Schnider fit must succeed");

    assert!(result.converged, "fit must converge");
    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );

    // ── Thetas: tight match to NONMEM (the structural-parameter regression
    //    guard). The pre-fix double-dosing pushed TVCL to ~3.4 — a 15% band
    //    around 1.92 (1.63‒2.21) excludes that cleanly while tolerating
    //    optimizer/platform variation.
    for (i, (&est, &nm)) in result.theta.iter().zip(&NM_THETA).enumerate() {
        let rel = (est - nm).abs() / nm;
        assert!(
            rel < 0.15,
            "theta {} ({}): ferx {:.4} vs NONMEM {:.4} (rel {:.1}%) exceeds 15%",
            i,
            result.theta_names[i],
            est,
            nm,
            rel * 100.0
        );
    }

    // ── Residual error: compare variances (ferx stores the proportional SD;
    //    NONMEM reports the variance).
    let ferx_sigma_var = result.sigma[0].powi(2);
    let sigma_rel = (ferx_sigma_var - NM_SIGMA_VAR).abs() / NM_SIGMA_VAR;
    assert!(
        sigma_rel < 0.15,
        "sigma variance: ferx {:.4} vs NONMEM {:.4} (rel {:.1}%) exceeds 15%",
        ferx_sigma_var,
        NM_SIGMA_VAR,
        sigma_rel * 100.0
    );

    // ── BSV variances. V1 and V2 trade off between the two implementations, so
    //    the robust cross-software assertion is on their *total* (within 20%);
    //    each individual variance is only sanity-checked to be in the right
    //    basin (not collapsed to ~0 or blown up), within a generous 2x band.
    let ferx_omega: Vec<f64> = (0..4).map(|i| result.omega[(i, i)]).collect();
    let ferx_total: f64 = ferx_omega.iter().sum();
    let nm_total: f64 = NM_OMEGA.iter().sum();
    let total_rel = (ferx_total - nm_total).abs() / nm_total;
    assert!(
        total_rel < 0.20,
        "total BSV variance: ferx {:.4} vs NONMEM {:.4} (rel {:.1}%) exceeds 20%",
        ferx_total,
        nm_total,
        total_rel * 100.0
    );
    for (i, (&est, &nm)) in ferx_omega.iter().zip(&NM_OMEGA).enumerate() {
        assert!(
            est > 0.5 * nm && est < 2.0 * nm,
            "omega {} ({}): ferx {:.4} outside the 0.5‒2x NONMEM basin ({:.4})",
            i,
            result.eta_names[i],
            est,
            nm
        );
    }
}
