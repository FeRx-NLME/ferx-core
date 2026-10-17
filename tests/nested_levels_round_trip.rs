//! Round trip for nested levels (#1082 PR4, Tier 3): data simulated by ferx at the
//! truth of the design note's exact3 anchor are refitted, and every variance component
//! comes back.
//!
//! The design is `nonmem_anchor/nested_levels_3.csv` (12 sponsors × 2 studies × 2 arms ×
//! 5 visits, NARM in {25, …, 400}) repeated [`REPEATS`] times under new sponsor ids, so
//! 240 sponsors, 480 studies and 960 arms. The truth is the generator's
//! (`gen.py`: E0 10, EMAX 20, ED50 30, ET50 3; ω²_sponsor 0.09, ω²_study 0.04, γ²_arm 4
//! with the arm κ scaled by 1/√NARM, σ² 0.25). The refit starts from the anchor's
//! NONMEM estimates on the 12-sponsor data (ω²_study 0.090, γ² 3.75), not from the
//! truth.
//!
//! This is the end-to-end test of the simulate side: a study κ drawn per arm instead of
//! per study inflates the arm variance and deflates the study variance the refit sees.
//! Measured (macOS arm64, release) with `simulate()` drawing one κ per cell: ω²_study
//! 0.011607 (−71%, band 13.5%) and γ²_arm 7.566337 (+89%, band 10.2%), so the test fails
//! on both, by more than 5× the band.

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions, Population, SimOutcome};

const MODEL: &str = include_str!("../nonmem_anchor/nested_levels_3.ferx");
const REPEATS: usize = 20;

/// The anchor model with its initial estimates replaced by the generator's truth.
fn truth_model() -> String {
    MODEL
        .replace(
            "theta TVE0(10.1818, 0.0, 1000.0)",
            "theta TVE0(10.0, 0.0, 1000.0)",
        )
        .replace(
            "theta TVEMAX(22.0140, 0.0, 1000.0)",
            "theta TVEMAX(20.0, 0.0, 1000.0)",
        )
        .replace(
            "theta ED50(33.4048, 0.1, 10000.0)",
            "theta ED50(30.0, 0.1, 10000.0)",
        )
        .replace(
            "theta ET50(3.20150, 0.01, 1000.0)",
            "theta ET50(3.0, 0.01, 1000.0)",
        )
        .replace("omega ETA_SP ~ 0.0699388", "omega ETA_SP ~ 0.09")
        .replace(
            "kappa K_ST ~ 0.0904362 at STUDY",
            "kappa K_ST ~ 0.04 at STUDY",
        )
        .replace("kappa K_ARM ~ 3.75478 at ARM", "kappa K_ARM ~ 4.0 at ARM")
        .replace("sigma ADD ~ 0.255164", "sigma ADD ~ 0.25")
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn nested_variance_components_are_recovered_from_simulated_data() {
    let truth_src = truth_model();
    assert_ne!(truth_src, MODEL);
    let truth = parse_full_model(&truth_src).expect("truth model parses");
    let start = parse_full_model(MODEL).expect("anchor model parses");
    assert_eq!(
        truth
            .model
            .default_params
            .omega_iov
            .as_ref()
            .unwrap()
            .matrix[(1, 1)],
        4.0
    );

    let (design, _) = read_population_for(
        &truth.model,
        &truth.covariate_decls,
        "nonmem_anchor/nested_levels_3.csv",
        None,
        None,
        None,
        &truth.column_map,
    )
    .expect("design loads");
    let mut subjects = Vec::new();
    for r in 0..REPEATS {
        for s in &design.subjects {
            let mut s = s.clone();
            s.id = format!("{}_{r}", s.id);
            subjects.push(s);
        }
    }
    let mut population = Population {
        subjects,
        ..design.clone()
    };

    let rows = simulate_with_seed(
        &truth.model,
        &population,
        &truth.model.default_params,
        1,
        1082,
    )
    .expect("simulates");
    for s in &mut population.subjects {
        s.observations = rows
            .iter()
            .filter(|r| r.id == s.id)
            .map(|r| match r.outcome {
                SimOutcome::Continuous { value } => value,
                _ => panic!("Gaussian rows"),
            })
            .collect();
        assert_eq!(s.observations.len(), 20);
    }

    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        run_covariance_step: false,
        ..start.fit_options.clone()
    };
    let r = fit(
        &start.model,
        &population,
        &start.model.default_params,
        &opts,
    )
    .expect("refits");
    let iov = r.omega_iov.as_ref().unwrap();
    let got = [
        ("TVE0", r.theta[0], 10.0, BAND_THETA),
        ("TVEMAX", r.theta[1], 20.0, BAND_THETA),
        ("ED50", r.theta[2], 30.0, BAND_THETA),
        ("ET50", r.theta[3], 3.0, BAND_THETA),
        ("ω²_sponsor", r.omega[(0, 0)], 0.09, BAND_SPONSOR),
        ("ω²_study (K_ST at STUDY)", iov[(0, 0)], 0.04, BAND_STUDY),
        ("γ²_arm (K_ARM at ARM)", iov[(1, 1)], 4.0, BAND_ARM),
        // `FitResult::sigma` is the SD (`sigma ADD ~ 0.25` declares the variance).
        ("σ²", r.sigma[0].powi(2), 0.25, BAND_SIGMA),
    ];
    eprintln!("converged {}  OFV {:.6}", r.converged, r.ofv);
    for (what, v, want, _) in got {
        eprintln!(
            "{what}: {v:.6} (truth {want}, relative error {:+.4})",
            (v - want) / want
        );
    }
    assert!(r.converged, "the refit converged");
    for (what, v, want, band) in got {
        assert!(v.is_finite(), "{what}: {v}");
        let rel = ((v - want) / want).abs();
        assert!(
            rel < band,
            "{what}: {v} vs truth {want}, |relative error| {rel:.4} ≥ {band}"
        );
    }
}

// Bands on |relative error| against the truth, from the Linux run (`tools/linux-test.sh`,
// Linux/aarch64, `--profile ci-test`, seed 1082; converged, OFV 1776.974216). The
// realised errors are the sampling error of one simulated dataset, which the seed
// fixes; each band is 1.5× its worst realised error. macOS arm64 gives the same digits.
//
// | quantity | refit (Linux) | truth | relative error |
// |---|---|---|---|
// | TVE0 / TVEMAX / ED50 / ET50 | 9.99351 / 19.7437 / 28.4871 / 3.00055 | 10 / 20 / 30 / 3 | −0.06% / −1.28% / −5.04% / +0.02% |
// | ω²_sponsor | 0.083858 | 0.09 | −6.82% |
// | ω²_study | 0.036409 | 0.04 | −8.98% |
// | γ²_arm | 4.271548 | 4 | +6.79% |
// | σ² (0.491382²) | 0.241456 | 0.25 | −3.42% |

/// Worst θ: ED50, 5.04%.
const BAND_THETA: f64 = 0.076;
/// 6.82%.
const BAND_SPONSOR: f64 = 0.102;
/// 8.98%.
const BAND_STUDY: f64 = 0.135;
/// 6.79%.
const BAND_ARM: f64 = 0.102;
/// 3.42%.
const BAND_SIGMA: f64 = 0.051;
