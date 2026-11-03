//! #2179: SIR's proposal conditioning (`condition_free_proposal`) floored the eigenvalues of the
//! packed covariance at `max(1e-10·λ_max, 1e-12)`, where an **identity-packed** θ (a negative
//! lower bound) carries its own units. With the warfarin volume written as `TVV_S = k·TVV`, its
//! variance scales as `k²`: at `k = 1e5` it became `λ_max` and the floor swallowed every other
//! direction (6 "carry no uncertainty", ESS 67), at `k = 1e6` every interval collapsed to a point
//! (ESS 1), at `k = 1e8` every draw was rejected, and at `k = 3e-6` `TVV_S` itself fell under the
//! absolute `1e-12` and was reported as not identified.
//!
//! The covariance step is already unit-free here (#2145, `cov_floor_identity_theta_units.rs`), so
//! each `k` hands SIR the same proposal up to rounding; with the same seed the intervals and ESS
//! must then equal the `k = 1` run's, `TVV_S`'s read as `/ k`, with no conditioning warning.
//! Evaluate-once fits (`outer_maxiter = 0`) and a short SIR: Tier 2, not gated.
use ferx_core::types::{CovarianceMethod, EstimationMethod, FitOptions, FitResult};
use ferx_core::{fit, read_nonmem_csv};
use std::path::Path;

const TVV: f64 = 7.737706;
const NAMES: [&str; 3] = ["TVCL", "TVV_S/k", "TVKA"];

fn model_src(k: f64) -> String {
    let tvv_s = TVV * k;
    format!(
        "[parameters]
  theta TVCL(0.132695, 0.001, 10.0)
  theta TVV_S({tvv_s:e}, {lo:e}, {hi:e})
  theta TVKA(0.810796, 0.01, 50.0)
  omega ETA_CL ~ 0.028590
  omega ETA_V  ~ 0.009592
  omega ETA_KA ~ 0.335871
  sigma PROP_ERR ~ 0.010565 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV_S / {k:e} * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
",
        lo = -100.0 * tvv_s,
        hi = 100.0 * tvv_s,
    )
}

/// The θ intervals in [`NAMES`] order (`TVV_S`'s divided by `k`) and the ESS, after asserting
/// the proposal needed no unit-driven repair.
fn sir_run(k: f64) -> (Vec<(f64, f64)>, f64) {
    let model = ferx_core::parser::model_parser::parse_model_string(&model_src(k)).unwrap();
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).unwrap();
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        outer_maxiter: 0,
        run_covariance_step: true,
        covariance_method: CovarianceMethod::Hessian,
        covariance_method_set: true,
        sir: true,
        sir_samples: 400,
        sir_resamples: 100,
        sir_seed: Some(7),
        verbose: false,
        ..FitOptions::default()
    };
    let r: FitResult = fit(&model, &pop, &model.default_params, &opts).unwrap();
    assert!(
        !r.warnings.iter().any(|w| w.contains("carry no uncertainty")
            || w.contains("carries no uncertainty")
            || w.contains("invalid weights")),
        "k = {k}: SIR's proposal floor fired; warnings: {:?}",
        r.warnings
    );
    let ci = r
        .sir_ci_theta
        .clone()
        .unwrap_or_else(|| panic!("k = {k}: no SIR CIs; warnings: {:?}", r.warnings));
    let ci = ci
        .iter()
        .enumerate()
        .map(|(i, &(lo, hi))| if i == 1 { (lo / k, hi / k) } else { (lo, hi) })
        .collect();
    (ci, r.sir_ess.expect("SIR ESS"))
}

#[test]
fn sir_intervals_do_not_move_with_an_identity_thetas_units() {
    let (reference, ess_ref) = sir_run(1.0);
    for (i, (lo, hi)) in reference.iter().enumerate() {
        assert!(
            lo.is_finite() && hi.is_finite() && hi > lo,
            "k = 1: CI({}) = [{lo}, {hi}]",
            NAMES[i]
        );
    }
    assert!(
        ess_ref.is_finite() && ess_ref > 10.0,
        "k = 1: ESS {ess_ref}"
    );
    for k in [1e5, 1e6, 3e-6] {
        let (ci, ess) = sir_run(k);
        assert!(ess.is_finite(), "k = {k}: ESS {ess}");
        // Every bound checked for finiteness before the fold, which `f64::max` would let a `NaN`
        // through.
        let mut worst = ((ess - ess_ref) / ess_ref).abs();
        eprintln!("k = {k:e}: ESS {ess:.6} vs k = 1 {ess_ref:.6}");
        for (i, ((lo, hi), (rlo, rhi))) in ci.iter().zip(&reference).enumerate() {
            assert!(
                lo.is_finite() && hi.is_finite(),
                "k = {k}: CI({}) = [{lo}, {hi}]",
                NAMES[i]
            );
            let rel = ((lo - rlo).abs() / rlo.abs()).max((hi - rhi).abs() / rhi.abs());
            eprintln!(
                "k = {k:e}: CI({}) = [{lo:.6}, {hi:.6}] vs k = 1 [{rlo:.6}, {rhi:.6}], rel {rel:.2e}",
                NAMES[i]
            );
            worst = worst.max(rel);
        }
        eprintln!("k = {k:e}: worst rel {worst:.3e}");
        // Measured (macOS arm64): worst 1.5e-10 (k = 1e5), 9.2e-11 (k = 1e6), 2.2e-11
        // (k = 3e-6), the covariance step's own rounding across k (#2145); bound 1e-8, 68×
        // headroom. With the equilibration switched off this run fails at k = 1e5 already: 6
        // directions "carry no uncertainty" and ESS 27.4 against 172.1.
        assert!(worst < 1e-8, "k = {k}: worst rel {worst}");
    }
}
