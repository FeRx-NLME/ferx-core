//! A large identity-packed θ is estimated, not left at its initial value (#2146).
//!
//! `TVV_S` has a negative lower bound, so it is packed on the identity scale, and
//! `V = TVV_S / 1e5` puts its initial value at 1e6. Before #2146 any identity-packed
//! θ switched the NLopt scale vector to all ones, and every optimizer stopped with
//! `TVV_S` within 0.1 % of 1e6 and `converged = true`, at OFV −265.42 against the
//! −286.0042 the same model reaches with `V = TVV_S` started at 10. Dividing that
//! θ by its initial magnitude is what moves it; this file pins it end to end on
//! the default `auto` (L-BFGS) and on the derivative-free BOBYQA, which `auto`
//! leaves unscaled.
//!
//! The control arm, `parameter_scaling = none`, keeps the all-ones vector and must
//! still stall: it is what shows the fixture has the defect at all.
//!
//! Gate: full fits to convergence, skipped in the default PR job.
//!
//!   cargo test --features slow-tests --test identity_theta_scaling

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, FitOptions, FitResult, Optimizer, ParameterScaling};
use std::path::Path;

const MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV_S(1000000, -50000000, 50000000)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV_S / 1e5 * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  maxiter    = 300
  covariance = false
"#;

/// The minimum, reached by L-BFGS with `TVV_S` started at 1e6 under the fix.
const OFV_MIN: f64 = -286.0042;

/// Where every optimizer stopped before the fix.
const OFV_STALLED: f64 = -265.42;

fn fit_with(optimizer: Optimizer, parameter_scaling: ParameterScaling) -> FitResult {
    let parsed = parse_full_model(MODEL).expect("model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    let options = FitOptions {
        optimizer,
        parameter_scaling,
        verbose: false,
        ..parsed.fit_options
    };
    let r = fit(
        &parsed.model,
        &population,
        &parsed.model.default_params,
        &options,
    )
    .expect("fit must succeed");
    // Printed for the band measurements (`-- --nocapture`).
    eprintln!(
        "{optimizer:?} / {parameter_scaling:?}: OFV {:.10}, TVV_S {:.6}, {} iterations",
        r.ofv,
        tvv_s(&r),
        r.n_iterations
    );
    r
}

fn tvv_s(result: &FitResult) -> f64 {
    let i = result
        .theta_names
        .iter()
        .position(|n| n == "TVV_S")
        .expect("TVV_S is a theta");
    result.theta[i]
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_large_identity_theta_reaches_the_minimum_under_auto() {
    let r = fit_with(Optimizer::Auto, ParameterScaling::Auto);
    assert!(r.ofv.is_finite(), "OFV must be finite, got {}", r.ofv);
    // PLACEHOLDER band: macOS arm64 gives −286.0042 (33 iterations) and
    // TVV_S 773770.6; to be set from a Linux x86_64 run.
    assert!(
        (r.ofv - OFV_MIN).abs() < 1e-3,
        "auto: OFV {} is not the minimum {OFV_MIN} (stalled fits end at {OFV_STALLED})",
        r.ofv
    );
    assert!(
        (tvv_s(&r) - 773_770.6).abs() < 100.0,
        "auto: TVV_S {} did not reach ~773770.6 from its initial 1e6",
        tvv_s(&r)
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_large_identity_theta_reaches_the_minimum_under_bobyqa() {
    let r = fit_with(Optimizer::Bobyqa, ParameterScaling::Auto);
    assert!(r.ofv.is_finite(), "OFV must be finite, got {}", r.ofv);
    // PLACEHOLDER band: macOS arm64 gives −285.9834 (173 iterations) and
    // TVV_S 774411.0; to be set from a Linux x86_64 run.
    assert!(
        r.ofv < OFV_MIN + 0.05 && r.ofv > OFV_MIN - 1e-3,
        "bobyqa: OFV {} is not within 0.05 of the minimum {OFV_MIN} \
         (stalled fits end at {OFV_STALLED})",
        r.ofv
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn an_explicit_none_still_leaves_the_large_identity_theta_stalled() {
    let r = fit_with(Optimizer::Auto, ParameterScaling::None);
    assert!(r.ofv.is_finite(), "OFV must be finite, got {}", r.ofv);
    // PLACEHOLDER band: macOS arm64 gives −265.4195 and TVV_S 999999.99999.
    assert!(
        (r.ofv - OFV_STALLED).abs() < 0.05,
        "none: OFV {} should stay at the stall {OFV_STALLED}; if it reaches the \
         minimum the fixture no longer shows the defect the other two tests fix",
        r.ofv
    );
    assert!(
        (tvv_s(&r) - 1e6).abs() < 1e3,
        "none: TVV_S {} should stay within 0.1 % of its initial 1e6",
        tvv_s(&r)
    );
}
