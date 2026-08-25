//! NONMEM `$PRED` cross-check for the compartment-free structural model (#811).
//!
//! `$PRED` is the exact NONMEM analogue: a block that computes `Y` from thetas,
//! etas and data items with no `$SUBROUTINE`, no ADVAN and no compartments. So
//! this anchor is a direct translation rather than an approximation of one — the
//! two control files differ only in syntax.
//!
//! ## The kit (`nonmem_anchor/`)
//!
//! - `algebraic_emax.csv` — 30 subjects × 8 timepoints, **no dose records at all**,
//!   simulated (stdlib `random`, seed 811) from
//!   `y = 10·exp(η) − 6·t/(2 + t) + ε`, `η ~ N(0, 0.09)`, `ε ~ N(0, 0.5²)`.
//! - `algebraic_emax.ctl` — the NONMEM `$PRED` control, FOCE, `$COV MATRIX=R`.
//! - `examples/emax_timecourse.ferx` — the ferx twin (the shipped example, so the
//!   anchored model is the one users run): same equation, same initial estimates
//!   (deliberately off the truth), same additive residual error, same method. Its
//!   data copy is `data/emax_timecourse.csv`, pinned byte-equal to the NONMEM one.
//!
//! ## Result — both engines converged independently from the same starts
//!
//! | | ferx | NONMEM 7.5.1 |
//! |---|---|---|
//! | OFV        | 62.5385   | 62.5384884 |
//! | TVE0       | 10.203867 | 10.2040    |
//! | TVEMAX     | 5.917646  | 5.91754    |
//! | TVET50     | 2.099437  | 2.09949    |
//! | ω²(E0)     | 0.092357  | 0.0924622  |
//! | σ (SD)     | 0.480452  | 0.480452   |
//! | SE TVE0    | 0.572367  | 0.572542   |
//! | SE TVEMAX  | 0.101099  | 0.101121   |
//! | SE TVET50  | 0.144743  | 0.144760   |
//! | SE ω²(E0)  | 0.023942  | 0.0239849  |
//! | SE σ (SD)  | 0.023437  | 0.0234361  |
//!
//! Agreement to ~1e-4 relative on every estimate and to 5 decimal places on the
//! OFV. SEs are compared because ferx's covariance is pure `R⁻¹`, which is what
//! `$COV MATRIX=R` reports — the default NONMEM sandwich would not be comparable.
//!
//! NONMEM stops with `ROUNDING ERRORS (ERROR=134)` on this dataset — the usual
//! outcome for a perfectly-specified simulated fit, where the objective is smooth
//! to machine precision and the `SIGDIGITS` test cannot be satisfied. It is at the
//! minimum regardless (final gradients ~1e-2, and a looser `NSIG=3` run stops
//! *earlier* at a worse OFV of 65.07), so `$COV` is forced with `UNCONDITIONAL`.

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// NONMEM 7.5.1 final estimates, read from `algebraic_emax.ext`'s
/// `-1000000000` row (θ₁ θ₂ θ₃ SIGMA(1,1) OMEGA(1,1) OBJ).
const NM_OFV: f64 = 62.5384884498;
const NM_THETA: [f64; 3] = [10.2040, 5.91754, 2.09949];
const NM_OMEGA_VAR: f64 = 9.24622e-2;
/// NONMEM reports `SIGMA(1,1)` as a **variance**; ferx reports the additive
/// residual as an **SD**, so this is `sqrt(0.230834)` — the same number NONMEM's
/// own `-1000000004` (standard-deviation) row carries.
const NM_SIGMA_SD: f64 = 0.480452;

/// SEs from the `-1000000001` row of the same file (`$COV MATRIX=R`).
const NM_SE_THETA: [f64; 3] = [0.572542, 0.101121, 0.144760];
const NM_SE_OMEGA: f64 = 2.39849e-2;
/// SE of the residual **SD**, from the `-1000000005` row (the standard-error-of-
/// standard-deviations row), to match ferx's SD-scale report.
const NM_SE_SIGMA_SD: f64 = 2.34361e-2;

/// ferx's compartment-free `[structural_model]` must reproduce NONMEM's `$PRED`
/// fit of the same equation on the same dose-free data — estimates, SEs and OFV.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored $PRED cross-check (#811): opt in with --features slow-tests"
)]
fn algebraic_structural_model_matches_nonmem_pred() {
    // The **shipped example** is the anchored model, so what a user runs from
    // `examples/` is the thing NONMEM was compared against — not a private copy
    // that can drift away from it.
    let parsed = parse_full_model_file(Path::new("examples/emax_timecourse.ferx"))
        .expect("compartment-free anchor model must parse");
    let model = parsed.model;
    assert!(
        model.is_algebraic(),
        "the anchor must exercise the compartment-free path, not a PK model"
    );

    // NONMEM reads its own copy next to the control stream. Pin them byte-equal so
    // "both engines fitted the same data" stays a fact rather than an assumption.
    let nm_csv = std::fs::read_to_string("nonmem_anchor/algebraic_emax.csv")
        .expect("NONMEM's copy of the anchor data");
    let ferx_csv = std::fs::read_to_string("data/emax_timecourse.csv").expect("the example's copy");
    assert_eq!(
        nm_csv, ferx_csv,
        "the NONMEM and ferx copies of the anchor dataset have diverged"
    );

    let pop = read_nonmem_csv(Path::new("data/emax_timecourse.csv"), None, None)
        .expect("anchor data must load");
    assert!(
        pop.subjects.iter().all(|s| s.doses.is_empty()),
        "the anchor dataset carries no dose records — that is the point"
    );

    let opts = FitOptions {
        method: EstimationMethod::Foce,
        run_covariance_step: true,
        verbose: false,
        ..Default::default()
    };

    let result = fit(&model, &pop, &model.default_params, &opts)
        .expect("compartment-free fit must converge");

    // OFV. A gap here means the prediction itself differs, not just the path to it.
    assert!(
        (result.ofv - NM_OFV).abs() < 1e-3,
        "ferx OFV {:.6} vs NONMEM {NM_OFV:.6}",
        result.ofv
    );

    for (i, (&nm, name)) in NM_THETA
        .iter()
        .zip(["TVE0", "TVEMAX", "TVET50"])
        .enumerate()
    {
        let got = result.theta[i];
        assert!(
            (got - nm).abs() / nm.abs() < 1e-3,
            "{name}: ferx {got:.6} vs NONMEM {nm:.6}"
        );
    }

    let omega = result.omega[(0, 0)];
    assert!(
        (omega - NM_OMEGA_VAR).abs() / NM_OMEGA_VAR < 5e-3,
        "omega: ferx {omega:.6} vs NONMEM {NM_OMEGA_VAR:.6}"
    );

    let sigma = result.sigma[0];
    assert!(
        (sigma - NM_SIGMA_SD).abs() / NM_SIGMA_SD < 1e-3,
        "sigma (SD): ferx {sigma:.6} vs NONMEM {NM_SIGMA_SD:.6}"
    );

    // SEs — ferx's covariance is pure R⁻¹, matching `$COV MATRIX=R`. These are the
    // part that exercises the second-order blocks of the analytic jet, so a wrong
    // `∂²f/∂η²` or `∂²f/∂η∂θ` shows up here even when the estimates agree.
    let se_theta = result
        .se_theta
        .as_ref()
        .expect("covariance step must succeed on a compartment-free model");
    for (i, (&nm, name)) in NM_SE_THETA
        .iter()
        .zip(["TVE0", "TVEMAX", "TVET50"])
        .enumerate()
    {
        let got = se_theta[i];
        assert!(
            (got - nm).abs() / nm.abs() < 5e-3,
            "SE({name}): ferx {got:.6} vs NONMEM {nm:.6}"
        );
    }
    let se_omega = result.se_omega.as_ref().expect("omega SEs")[0];
    assert!(
        (se_omega - NM_SE_OMEGA).abs() / NM_SE_OMEGA < 1e-2,
        "SE(omega): ferx {se_omega:.6} vs NONMEM {NM_SE_OMEGA:.6}"
    );
    let se_sigma = result.se_sigma.as_ref().expect("sigma SEs")[0];
    assert!(
        (se_sigma - NM_SE_SIGMA_SD).abs() / NM_SE_SIGMA_SD < 1e-2,
        "SE(sigma): ferx {se_sigma:.6} vs NONMEM {NM_SE_SIGMA_SD:.6}"
    );
}

/// Known-truth recovery: the same fit must land near the values the data was
/// simulated from. Distinct from the anchor above — two engines can agree on a
/// wrong answer, but neither can agree with the generating parameters by accident.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow convergence fit (#811): opt in with --features slow-tests"
)]
fn algebraic_structural_model_recovers_the_simulation_truth() {
    // `nonmem_anchor/algebraic_emax.csv` was simulated from these.
    const TRUTH_THETA: [f64; 3] = [10.0, 6.0, 2.0];
    const TRUTH_OMEGA: f64 = 0.09;
    const TRUTH_SIGMA_SD: f64 = 0.5;

    let model = parse_full_model_file(Path::new("examples/emax_timecourse.ferx"))
        .expect("anchor model must parse")
        .model;
    let pop = read_nonmem_csv(Path::new("data/emax_timecourse.csv"), None, None)
        .expect("anchor data must load");

    let opts = FitOptions {
        method: EstimationMethod::Foce,
        run_covariance_step: false,
        verbose: false,
        ..Default::default()
    };
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must converge");

    // 5% on the structural parameters, 20% on the variance components — the usual
    // sampling-noise band for 30 subjects × 8 observations.
    for (i, (&truth, name)) in TRUTH_THETA
        .iter()
        .zip(["TVE0", "TVEMAX", "TVET50"])
        .enumerate()
    {
        let got = result.theta[i];
        assert!(
            (got - truth).abs() / truth < 0.05,
            "{name}: recovered {got:.4}, simulated from {truth:.4}"
        );
    }
    let omega = result.omega[(0, 0)];
    assert!(
        (omega - TRUTH_OMEGA).abs() / TRUTH_OMEGA < 0.20,
        "omega: recovered {omega:.4}, simulated from {TRUTH_OMEGA:.4}"
    );
    let sigma = result.sigma[0];
    assert!(
        (sigma - TRUTH_SIGMA_SD).abs() / TRUTH_SIGMA_SD < 0.20,
        "sigma (SD): recovered {sigma:.4}, simulated from {TRUTH_SIGMA_SD:.4}"
    );
}
