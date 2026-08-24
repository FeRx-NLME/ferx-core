//! NONMEM 7.6.0 cross-check for sample-size-weighted IOV (#1031).
//!
//! NONMEM has no `weight =` on an `$OMEGA`, so the equivalent model writes the
//! scaling into `$PK` by hand — `CL = THETA(1)*EXP(ETA(1) + KAPPA/SQRT(NARM))`.
//! That *is* what ferx's `kappa KAPPA_CL ~ 0.01 weight = NARM` desugars into, so
//! the anchor is a direct one: same data, same 1-cpt oral structure, same FOCE
//! objective, one declaration versus three lines of hand arithmetic.
//!
//! ## Data
//!
//! `tests/nonmem/kappa_weight_iov.csv` — `data/warfarin_iov.csv` plus an arm-size
//! column `NARM`: 400 on occasion 1, 25 on occasion 2. The 16× spread in N is a
//! 4× spread in κ's effective SD, so a scaling applied wrongly (`/N` instead of
//! `/√N`) or not at all cannot pass as a rounding difference.
//!
//! ## Reference values (`tests/nonmem/kappa_weight_iov.{ctl,lst,ext}`)
//!
//! Converged FOCE fits from the same starting values:
//!
//! | Quantity            | NONMEM 7.6.0 | ferx     |
//! |---------------------|--------------|----------|
//! | OFV (no constant)   | 207.455      | 205.683  |
//! | TVCL                | 0.31727      | 0.32464  |
//! | TVV                 | 8.4018       | 8.4445   |
//! | TVKA                | 2.6180       | 2.4467   |
//! | ω²(CL)              | 0.59721      | 0.61281  |
//! | ω²(V)               | 0.012400     | 0.012779 |
//! | ω²(KA)              | 1.0571       | 0.89725  |
//! | γ² (Ω_IOV)          | 2.0013       | 1.8402   |
//! | σ² (prop)           | 0.037588     | 0.040884 |
//!
//! ferx reaches a *lower* objective from the same start, on a model whose IOV
//! direction is weakly identified (κ enters divided by √400 and √25).
//!
//! ## Is the residual gap the weighting?
//!
//! No — it is the documented cross-engine FOCE/IOV difference, which exists
//! without any weight and is *smaller* under weighting where κ is small.
//! Both engines evaluated at identical fixed parameters:
//!
//! | Parameters          | Model                | NONMEM  | ferx    | Δ    |
//! |---------------------|----------------------|---------|---------|------|
//! | inits, γ² = 0.01    | weighted `κ/√NARM`   | 466.779 | 466.720 | 0.06 |
//! | inits, γ² = 0.01    | plain `κ`            | 332.265 | 331.620 | 0.65 |
//! | NM MLE, γ² = 2.00   | weighted `κ/√NARM`   | 207.455 | 205.943 | 1.51 |
//! | NM MLE, γ² = 2.00   | plain `κ`            | 238.889 | 238.524 | 0.37 |
//!
//! The gap tracks how strongly κ enters the marginal, not whether it is
//! weighted. The *declaration itself* is exact: `weight = NARM` and the
//! hand-written `KAPPA_CL / sqrt(NARM)` score bit-identically in ferx under
//! every estimator (`tests/weighted_kappa_iov.rs`), and the analytic `∂f/∂κ`
//! matches finite differences of `predict_iov` (`sens::provider`).

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

// NONMEM 7.6.0 FOCE reference (tests/nonmem/kappa_weight_iov.ext).
const NM_OFV: f64 = 207.455;
const NM_TVCL: f64 = 0.317269;
const NM_TVV: f64 = 8.40182;
const NM_TVKA: f64 = 2.61803;
const NM_OMEGA_IOV: f64 = 2.00128;

fn anchor_fit() -> ferx_core::types::FitResult {
    let model = parse_model_file(Path::new("tests/nonmem/kappa_weight_iov.ferx"))
        .expect("weighted-kappa anchor model must parse");
    assert!(
        model.has_weighted_kappa(),
        "the anchor model must carry `weight = NARM`"
    );
    let pop = read_nonmem_csv(
        Path::new("tests/nonmem/kappa_weight_iov.csv"),
        None,
        Some("OCC"),
    )
    .expect("anchor data must load");
    let opts = FitOptions {
        method: EstimationMethod::Foce,
        iov_column: Some("OCC".to_string()),
        run_covariance_step: false,
        ..Default::default()
    };
    fit(&model, &pop, &model.default_params, &opts).expect("anchor fit")
}

/// ferx's weighted-kappa fit must land on NONMEM's hand-written twin: the same
/// objective to within the cross-engine FOCE/IOV gap documented above (never
/// worse than NONMEM's), and the same estimates including the *unweighted* γ².
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn weighted_kappa_matches_the_nonmem_hand_written_twin() {
    let res = anchor_fit();
    assert!(res.converged, "the anchor fit must converge");
    // ferx must not score *worse* than NONMEM, and must stay within the
    // cross-engine gap measured at matched parameters (1.51) plus margin.
    assert!(
        res.ofv <= NM_OFV + 1e-6 && res.ofv >= NM_OFV - 5.0,
        "ferx OFV {} is not within the cross-engine band around NONMEM's {NM_OFV}",
        res.ofv
    );
    // Structural parameters within 10% — a weakly-identified IOV surface, and
    // the two engines stop at slightly different points on it.
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
    assert!(rel(res.theta[0], NM_TVCL) < 0.10, "TVCL {:?}", res.theta);
    assert!(rel(res.theta[1], NM_TVV) < 0.10, "TVV {:?}", res.theta);
    assert!(rel(res.theta[2], NM_TVKA) < 0.10, "TVKA {:?}", res.theta);
    // The reported Ω_IOV is the *unweighted* γ² — the quantity NONMEM's
    // `$OMEGA BLOCK(1)` holds. A `/NARM` instead of `/sqrt(NARM)` scaling would
    // land here as a factor of ~20², not a 10% difference.
    let gamma2 = res.omega_iov.as_ref().expect("IOV matrix present")[(0, 0)];
    assert!(
        rel(gamma2, NM_OMEGA_IOV) < 0.15,
        "gamma^2 {gamma2} vs NONMEM {NM_OMEGA_IOV}"
    );
}

/// The fit reports the weight and the median arm size next to that γ², so the
/// effective between-arm SD (`γ/√N`) is readable without a calculator.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn the_anchor_fit_reports_the_arm_size() {
    let res = anchor_fit();
    assert_eq!(res.kappa_weights, vec![Some("NARM".to_string())]);
    let n = res.kappa_weight_typical[0].expect("a typical arm size");
    assert!(n == 400.0 || n == 25.0, "unexpected median arm size {n}");
}
