//! A large-magnitude identity-packed θ is estimated, not left at its initial value (#2146).
//!
//! `TVV_S` has a negative lower bound, so it is packed on the identity scale, and
//! `V = TVV_S / 1e5` puts its initial value at 1e6: the same structural start as the
//! plain warfarin fit (`V = TVV`, θ₀ = 10), whose minimum is OFV −286.0042 at
//! `TVV_S = 773770.6`. The NLopt scale vector used to give every coordinate 1.0 once
//! any θ was identity-packed, so the optimizer saw this one in raw units, with a
//! packed gradient ~1/|θ₀| of the unit-magnitude one: L-BFGS stopped at −265.4195 with
//! `TVV_S` unmoved and BOBYQA at −265.4166, each with `converged = true`. Under `auto`
//! and `abs` an identity θ with `|θ₀| > 10` is now divided by `|θ₀|`
//! (`nlopt_scale_vector`).
//!
//! The `parameter_scaling = none` arm keeps the all-ones vector and must still stop
//! short: it shows the fixture has the defect.
//!
//! Gate: full fits to convergence, skipped in the default PR job.
//!
//!   cargo test --features slow-tests --test identity_theta_large

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, FitOptions, FitResult, Optimizer, ParameterScaling};
use std::path::Path;

const MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV_S(1000000.0, -50000000.0, 50000000.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV_S / 100000.0 * exp(ETA_V)
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

/// The minimum of the warfarin fixture, reached from θ₀ = 10 on the plain model.
const WARFARIN_MIN: f64 = -286.0042;
/// The gradient optimizers' band. L-BFGS (`auto`, 33 iterations), L-BFGS under an
/// explicit `abs`, and SLSQP (105) each print −286.0042 (macOS arm64); the stalled
/// arms sit ~20.6 units away.
const TOL_GRADIENT: f64 = 1e-2;
/// BOBYQA's band. It printed −286.0042 here (macOS arm64, 335 iterations), but the
/// #2146 plan measured it 0.02 short at θ₀ = 1e6 on an earlier tree, so the band
/// leaves room for that.
const TOL_BOBYQA: f64 = 5e-2;

fn warfarin(optimizer: Optimizer, scaling: ParameterScaling) -> FitResult {
    let parsed = parse_full_model(MODEL).expect("model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("data loads");
    let options = FitOptions {
        optimizer,
        parameter_scaling: scaling,
        run_covariance_step: false,
        verbose: false,
        ..parsed.fit_options.clone()
    };
    fit(&parsed.model, &pop, &parsed.model.default_params, &options).expect("fit runs")
}

fn assert_at_min(r: &FitResult, tol: f64, label: &str) {
    eprintln!("{label}: OFV {:.6}, TVV_S {}", r.ofv, r.theta[1]);
    assert!(r.ofv.is_finite(), "{label}: OFV {}", r.ofv);
    assert!(
        (r.ofv - WARFARIN_MIN).abs() < tol,
        "{label}: OFV {:.4}, want {WARFARIN_MIN} ± {tol}",
        r.ofv
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn gradient_optimizers_reach_the_minimum_from_a_large_identity_theta() {
    for (opt, scaling) in [
        (Optimizer::Auto, ParameterScaling::Auto),
        (Optimizer::NloptLbfgs, ParameterScaling::Abs),
        (Optimizer::Slsqp, ParameterScaling::Auto),
    ] {
        let r = warfarin(opt, scaling);
        assert_at_min(&r, TOL_GRADIENT, &format!("{opt:?} / {scaling:?}"));
        let tvv_s = r.theta[1];
        assert!(
            (tvv_s - 773770.6).abs() < 100.0,
            "{opt:?} / {scaling:?}: TVV_S {tvv_s}, want 773770.6"
        );
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn bobyqa_reaches_the_minimum_from_a_large_identity_theta() {
    let r = warfarin(Optimizer::Bobyqa, ParameterScaling::Auto);
    assert_at_min(&r, TOL_BOBYQA, "bobyqa");
}

/// The control arm: with no scaling the fixture still stops short, so the tests above
/// are about the scale, not about a fixture any optimizer solves.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn an_explicit_none_still_stops_short_on_the_fixture() {
    for opt in [Optimizer::Auto, Optimizer::Bobyqa] {
        let r = warfarin(opt, ParameterScaling::None);
        assert!(r.ofv.is_finite(), "{opt:?}: OFV {}", r.ofv);
        assert!(
            r.ofv > WARFARIN_MIN + 10.0,
            "{opt:?}, parameter_scaling = none reached {:.4}: the fixture no longer shows \
             the defect",
            r.ofv
        );
    }
}
