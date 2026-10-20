//! A near-zero identity-packed θ is estimated, not stopped short of its minimum (#2159).
//!
//! `TVV_S` has a negative lower bound, so it is packed on the identity scale, and
//! `V = TVV_S / 1e-6` puts its initial value at 1e-5: the same structural start as
//! the plain warfarin fit (`V = TVV_S`, θ₀ = 10), whose minimum is OFV −286.0042 at
//! `TVV_S = 7.7377`. On the unit NLopt scale every identity θ gets, BOBYQA's first
//! probes put the volume negative and it stopped at −205.9732, SLSQP at −198.9491
//! and L-BFGS at −269.4658 (macOS arm64, `badfdd5e`), each with `converged = true`.
//! The flat-θ pre-flight now probes such a θ at half its value and, when that moves
//! the objective, divides it by its initial magnitude (`identity_theta_scales`).
//!
//! The control is `two_cpt_oral_cov` with both covariate exponents started at 0.001:
//! O(1) slopes the objective barely sees at that size, which the rule must leave on
//! the unit scale. Dividing them by 0.001 instead took BOBYQA to −1185.41 against
//! −1199.3264 (the issue's measurement). The `parameter_scaling = none` arm keeps the
//! all-ones vector and must still stop short: it shows the fixture has the defect.
//!
//! Gate: full fits to convergence, skipped in the default PR job.
//!
//!   cargo test --features slow-tests --test identity_theta_near_zero

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, FitOptions, FitResult, Optimizer, ParameterScaling};
use std::path::Path;

const MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV_S(1e-05, -0.0005, 0.0005)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV_S / 1e-6 * exp(ETA_V)
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

/// The minimum of the warfarin fixture, reached from θ₀ = 10 on the plain model by
/// every optimizer and, with the fix, from θ₀ = 1e-5 by all three (−286.0042 each,
/// macOS arm64).
const WARFARIN_MIN: f64 = -286.0042;
/// The `two_cpt_oral_cov` minimum with both exponents free, reached by L-BFGS,
/// SLSQP and BOBYQA from 0.001 with and without the fix.
const TWO_CPT_MIN: f64 = -1199.3264;
/// The issue's bound. Every arm that reaches the minimum prints the same four
/// decimals; the stalled arms sit 17–87 units away.
const TOL: f64 = 0.05;

fn run(src: &str, data: &str, optimizer: Optimizer, scaling: ParameterScaling) -> FitResult {
    let parsed = parse_full_model(src).expect("model parses");
    let pop = read_nonmem_csv(Path::new(data), None, None).expect("data loads");
    let options = FitOptions {
        optimizer,
        parameter_scaling: scaling,
        run_covariance_step: false,
        verbose: false,
        ..parsed.fit_options.clone()
    };
    fit(&parsed.model, &pop, &parsed.model.default_params, &options).expect("fit runs")
}

fn warfarin(optimizer: Optimizer, scaling: ParameterScaling) -> FitResult {
    run(MODEL, "data/warfarin.csv", optimizer, scaling)
}

fn assert_at_min(r: &FitResult, want: f64, label: &str) {
    assert!(r.ofv.is_finite(), "{label}: OFV {}", r.ofv);
    assert!(
        (r.ofv - want).abs() < TOL,
        "{label}: OFV {:.4}, want {want} ± {TOL}",
        r.ofv
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn bobyqa_reaches_the_minimum_from_a_near_zero_identity_theta() {
    let r = warfarin(Optimizer::Bobyqa, ParameterScaling::Auto);
    assert_at_min(&r, WARFARIN_MIN, "bobyqa");
    let tvv_s = r.theta[1];
    assert!(
        (tvv_s - 7.7377e-6).abs() < 1e-8,
        "bobyqa: TVV_S {tvv_s:e}, want 7.7377e-6"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn gradient_optimizers_reach_the_minimum_from_a_near_zero_identity_theta() {
    for opt in [Optimizer::Auto, Optimizer::Slsqp] {
        let r = warfarin(opt, ParameterScaling::Auto);
        assert_at_min(&r, WARFARIN_MIN, &format!("{opt:?}"));
    }
}

/// The control arm: with the probe off the fixture still stops short, so the two
/// tests above are about the scale, not about a fixture any optimizer solves.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn an_explicit_none_still_stops_short_on_the_fixture() {
    let r = warfarin(Optimizer::Bobyqa, ParameterScaling::None);
    assert!(
        r.ofv > WARFARIN_MIN + 10.0,
        "parameter_scaling = none reached {:.4}: the fixture no longer shows the defect",
        r.ofv
    );
}

/// The rule must not move a near-zero O(1) slope off the unit scale: both covariate
/// exponents of `two_cpt_oral_cov` started at 0.001 still reach −1199.3264.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_near_zero_slope_keeps_reaching_the_minimum() {
    let src = std::fs::read_to_string("examples/two_cpt_oral_cov.ferx").expect("example reads");
    let src = src
        .replace(
            "theta THETA_WT(0.6, 0.01, 5.0)",
            "theta THETA_WT(0.001, -5.0, 5.0)",
        )
        .replace(
            "theta THETA_CRCL(0.3, 0.01, 5.0)",
            "theta THETA_CRCL(0.001, -5.0, 5.0)",
        );
    assert!(src.contains("THETA_WT(0.001, -5.0") && src.contains("THETA_CRCL(0.001, -5.0"));
    for opt in [Optimizer::Bobyqa, Optimizer::Auto, Optimizer::Slsqp] {
        let r = run(
            &src,
            "data/two_cpt_oral_cov.csv",
            opt,
            ParameterScaling::Auto,
        );
        assert_at_min(&r, TWO_CPT_MIN, &format!("two_cpt_oral_cov {opt:?}"));
    }
}
