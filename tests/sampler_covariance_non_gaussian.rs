//! Tier-3 (#1781): standard errors after `method = saem`, `imp` and `impmap` on a model with a
//! non-Gaussian endpoint, against a reference computed **outside ferx**.
//!
//! Fixture: `examples/tte_exponential.ferx` on `data/tte_exponential.csv` (30 subjects, one
//! exponential event time each, administrative censoring at 30 h, ω² on log λ).
//!
//! Reference: the exact marginal likelihood, by brute force — composite Simpson over η/ω on
//! [−12, 12] with 4001 points per subject, Newton in (log λ, log ω²) to |step| < 1e-9, then at
//! that minimum `R = ½·H` from a central-difference Hessian of the OFV and `S = Σᵢ gᵢgᵢᵀ` from
//! central-difference per-subject scores, delta-mapped to λ and ω². Pure Python, no ferx code:
//! λ̂ = 0.033776, ω̂² = 0.071468, OFV 166.5267, and
//!
//! | estimator | SE(λ) | SE(ω²) |
//! |---|---|---|
//! | `R⁻¹` | 0.008182 | 0.6851 |
//! | `S⁻¹` | 0.008133 | 0.8946 |
//! | `R⁻¹SR⁻¹` | 0.008234 | 0.5272 |
//!
//! ferx's own 9-node `method = laplace` fit lands on the same point and reproduces the `R⁻¹`
//! and `S⁻¹` rows to 4 and 6 significant figures (0.008178 / 0.6843, 0.008133 / 0.8946).
//!
//! NONMEM 7.6 on the same data, for scale: `METHOD=IMP LAPLACE` reaches λ = 0.03370,
//! ω² = 0.0905 with SE(λ) = 0.00840 / 0.00847 / 0.00835 under `MATRIX=R` / `S` / default
//! (RSR); `METHOD=COND LAPLACE` (= FOCEI) reaches 0.03323 / 0.2847 with SE(λ) 0.00916 (R).
//! NONMEM's own `METHOD=SAEM` collapses ω² to 1.5e-15 on this data, so it is not a reference.
//!
//! Before the fix the FOCEI objective's curvature was inverted at the samplers' estimates,
//! where it is indefinite: SE(λ) = 4.2–9.8 under `r` and 1.3e4–4.1e4 under `rsr`.
//!
//! This is the broad full-fit check. It compares each sampler's covariance at its *own* estimate
//! with the oracle at the *MLE*, so its bands carry estimator location as well as covariance
//! error. The precise check is the same-point anchor,
//! `covariance::tests::the_sampler_points_match_the_exact_marginal_covariance`, which evaluates
//! ferx and the oracle at one point (worst realised error 2.0e-3, bound 5e-3).
//!
//! Each estimator is banded against its own reference row. Bands are measured (release CLI,
//! Linux x86_64, 2026-10-13) and printed by every run: the worst realised relative error is
//! 5.2% on SE(λ) and 12.8% on SE(ω²), both SAEM, whose estimate is not the reference minimum
//! (exact OFV 166.8789 against 166.5267; IMP and IMPMAP are within 6e-4 of it and within 0.5%
//! / 12.4% of the reference). The bands are 10% and 20%. A regression to the floored FOCEI
//! `R` misses them by three orders of magnitude, and a silent change of estimator is caught
//! by the published-estimator assertion.

#![cfg(all(feature = "survival", feature = "slow-tests"))]

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{CovarianceMethod, EstimationMethod};
use ferx_core::{fit, FitOptions};

/// The exact marginal's `(SE(λ), SE(ω²))` under each estimator (see the module docs).
fn reference(cm: CovarianceMethod) -> (f64, f64) {
    match cm {
        CovarianceMethod::Hessian => (0.008182, 0.6851),
        CovarianceMethod::CrossProduct => (0.008133, 0.8946),
        CovarianceMethod::Sandwich => (0.008234, 0.5272),
    }
}
const BAND_LAMBDA: f64 = 0.10;
const BAND_OMEGA2: f64 = 0.20;

#[test]
fn sampler_standard_errors_on_a_tte_model_match_the_exact_marginal() {
    let src = std::fs::read_to_string("examples/tte_exponential.ferx").expect("model file");
    let model = parse_model_string(&src).expect("parse");
    let (pop, _) = read_population_for(
        &model,
        &None,
        "data/tte_exponential.csv",
        None,
        None,
        None,
        &[],
    )
    .expect("data");

    let mut failures = Vec::new();
    for method in [
        EstimationMethod::Saem,
        EstimationMethod::Imp,
        EstimationMethod::Impmap,
    ] {
        for cm in [
            CovarianceMethod::Hessian,
            CovarianceMethod::Sandwich,
            CovarianceMethod::CrossProduct,
        ] {
            let opts = FitOptions {
                method,
                covariance_method: cm,
                covariance_method_set: true,
                run_covariance_step: true,
                ..FitOptions::default()
            };
            let r = fit(&model, &pop, &model.default_params, &opts).expect("fit");
            let label = format!("{method:?} / {}", cm.label());
            assert_eq!(
                r.covariance_method,
                Some(cm),
                "{label}: the published estimator"
            );
            assert_eq!(
                r.warnings
                    .iter()
                    .filter(|w| w.starts_with("W_COV_QUADRATURE_OBJECTIVE"))
                    .count(),
                1,
                "{label}: {:?}",
                r.warnings
            );
            let se_l = r.se_theta.as_ref().expect("se_theta")[0];
            let se_o = r.se_omega.as_ref().expect("se_omega")[0];
            assert!(
                se_l.is_finite() && se_o.is_finite(),
                "{label}: {se_l} {se_o}"
            );
            let (ref_l, ref_o) = reference(cm);
            let (el, eo) = ((se_l - ref_l).abs() / ref_l, (se_o - ref_o).abs() / ref_o);
            eprintln!(
                "{label:>20}: λ = {:.6} SE {se_l:.6} ({:+.1}%), ω² = {:.6} SE {se_o:.4} ({:+.1}%)",
                r.theta[0],
                100.0 * (se_l / ref_l - 1.0),
                r.omega[(0, 0)],
                100.0 * (se_o / ref_o - 1.0),
            );
            if el > BAND_LAMBDA || eo > BAND_OMEGA2 {
                failures.push(format!(
                    "{label}: SE(λ) {se_l} ({el:.3}), SE(ω²) {se_o} ({eo:.3})"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The FOCEI fit on the same data is untouched by #1781 — no route note, and its SEs stay at
/// the values the base commit reports (release CLI, Linux x86_64): SE(λ) 0.008871 under `r`.
#[test]
fn a_focei_fit_on_the_same_model_keeps_its_own_objective() {
    let src = std::fs::read_to_string("examples/tte_exponential.ferx").expect("model file");
    let model = parse_model_string(&src).expect("parse");
    let (pop, _) = read_population_for(
        &model,
        &None,
        "data/tte_exponential.csv",
        None,
        None,
        None,
        &[],
    )
    .expect("data");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        run_covariance_step: true,
        ..FitOptions::default()
    };
    let r = fit(&model, &pop, &model.default_params, &opts).expect("fit");
    assert!(
        !r.warnings.iter().any(|w| w.contains("W_COV_QUADRATURE")),
        "{:?}",
        r.warnings
    );
    let se_l = r.se_theta.as_ref().expect("se_theta")[0];
    eprintln!("FOCEI SE(λ) = {se_l:.6}");
    assert!((se_l - 0.008871).abs() < 1e-4, "FOCEI SE(λ) = {se_l}");
}
