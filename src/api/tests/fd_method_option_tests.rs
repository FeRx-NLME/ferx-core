//! `check_model_options` gate on the adaptive finite-difference options (#1314).

use super::*;
use crate::types::test_helpers::analytical_model;
use crate::types::{GradientMethod, InnerFdMethod, OuterFdMethod};

fn fd_diags(opts: &FitOptions) -> Vec<Diagnostic> {
    let model = analytical_model(GradientMethod::Auto);
    check_model_options(&model, opts)
        .into_iter()
        .filter(|d| d.code == "E_OUTER_FD_NOISE_REQUIRED" || d.code == "E_INNER_FD_NOISE_REQUIRED")
        .collect()
}

/// The default is `Fixed` on both axes, and `Fixed` needs no noise estimate.
#[test]
fn fixed_intervals_need_no_noise_estimate() {
    assert!(fd_diags(&FitOptions::default()).is_empty());
}

/// The gate must reject every value that makes the search unusable, not only
/// `None`. `noise` is a divisor: at `0.0` every ratio is `inf`, no interval is
/// ever accepted, and every coordinate silently falls back — a whole fit run
/// under a setting the user believes is active. `.ferx` files cannot express
/// these (the parser's `parse_pos_finite` rejects them first), but a Rust or
/// ferx-r caller assigning the field directly can, and this is the gate that
/// stops it.
#[test]
fn adaptive_outer_intervals_reject_a_non_positive_noise_estimate() {
    for noise in [
        None,
        Some(0.0),
        Some(-1e-6),
        Some(f64::NAN),
        Some(f64::INFINITY),
    ] {
        for method in [OuterFdMethod::Shi, OuterFdMethod::Gill] {
            let opts = FitOptions {
                outer_fd_method: method,
                outer_fd_noise_abs: noise,
                ..Default::default()
            };
            let diags = fd_diags(&opts);
            assert_eq!(
                diags.len(),
                1,
                "{method:?} with outer_fd_noise_abs = {noise:?} must be rejected"
            );
            assert_eq!(diags[0].code, "E_OUTER_FD_NOISE_REQUIRED");
            // The message promises "positive"; the check must actually enforce it.
            assert!(
                diags[0].message.contains("positive"),
                "{}",
                diags[0].message
            );
        }
    }
}

#[test]
fn adaptive_outer_intervals_accept_a_positive_noise_estimate() {
    for method in [OuterFdMethod::Shi, OuterFdMethod::Gill] {
        let opts = FitOptions {
            outer_fd_method: method,
            outer_fd_noise_abs: Some(1e-6),
            ..Default::default()
        };
        assert!(fd_diags(&opts).is_empty(), "{method:?} must be accepted");
    }
}

/// Both inner bounds are required, and each is checked for positivity
/// independently — a fixture that only ever varies one of them cannot tell a
/// two-sided gate from a one-sided one.
#[test]
fn adaptive_inner_intervals_reject_a_non_positive_noise_estimate() {
    let bad = [None, Some(0.0), Some(-1.0), Some(f64::NAN)];
    for noise in bad {
        for (objective, prediction) in [(noise, Some(1e-9)), (Some(1e-7), noise)] {
            let opts = FitOptions {
                inner_fd_method: InnerFdMethod::Shi,
                inner_fd_objective_noise_abs: objective,
                inner_fd_prediction_noise_abs: prediction,
                ..Default::default()
            };
            let diags = fd_diags(&opts);
            assert_eq!(
                diags.len(),
                1,
                "objective = {objective:?}, prediction = {prediction:?} must be rejected"
            );
            assert_eq!(diags[0].code, "E_INNER_FD_NOISE_REQUIRED");
            assert!(
                diags[0].message.contains("positive"),
                "{}",
                diags[0].message
            );
        }
    }
}

#[test]
fn adaptive_inner_intervals_accept_positive_noise_estimates() {
    let opts = FitOptions {
        inner_fd_method: InnerFdMethod::Shi,
        inner_fd_objective_noise_abs: Some(1e-7),
        inner_fd_prediction_noise_abs: Some(1e-9),
        ..Default::default()
    };
    assert!(fd_diags(&opts).is_empty());
}
