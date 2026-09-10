//! Adaptive outer finite-difference intervals in `central_diff_packed` (#1314).

use super::*;

fn packed_bounds(lower: f64, upper: f64, n: usize) -> PackedBounds {
    PackedBounds {
        lower: vec![lower; n],
        upper: vec![upper; n],
    }
}

/// The default `Fixed` policy must be bit-identical to the pre-#1314 stencil,
/// including the `reject_value` wall a guard-rejected probe contributes. This is
/// the assertion that would have caught the default-path change the adaptive
/// work introduced: a rejected probe used to push the optimizer away with a
/// `~1e24` derivative, and briefly returned `0.0` — "stationary" — instead.
#[test]
fn fixed_intervals_keep_the_guard_wall_on_a_rejected_probe() {
    let x = [1.0, 2.0];
    let fixed = [false, false];
    let bounds = packed_bounds(-10.0, 10.0, 2);
    // Coordinate 0 is scorable on both arms; coordinate 1 is rejected on its
    // upper arm only, which is the one-sided rejection that has to repel.
    let eval = |v: &[f64]| -> Option<f64> {
        if v[1] > 2.0 {
            None
        } else {
            Some(v.iter().map(|a| a * a).sum())
        }
    };
    let g = central_diff_packed(
        &x,
        &fixed,
        &bounds,
        OuterFdMethod::Fixed,
        None,
        Some(GUARD_REJECT_OFV),
        eval,
    );
    assert!((g[0] - 2.0).abs() < 1e-5, "smooth coordinate: {}", g[0]);
    assert!(
        g[1] > 1e20,
        "a guard-rejected upper arm must repel, not read as stationary (got {})",
        g[1]
    );

    // With `reject_value = None` the coordinate is dropped instead — the
    // per-subject convention. Both spellings are exercised here, so collapsing
    // them into one is a red test.
    let dropped = central_diff_packed(&x, &fixed, &bounds, OuterFdMethod::Fixed, None, None, eval);
    assert_eq!(dropped[1], 0.0);
    assert!((dropped[0] - 2.0).abs() < 1e-5);
}

/// The adaptive branch must reach the same derivative as the fixed one on a
/// smooth objective.
#[test]
fn adaptive_intervals_agree_with_the_fixed_stencil_on_a_smooth_objective() {
    let x = [1.0, 2.0];
    let fixed = [false, false];
    let bounds = packed_bounds(-10.0, 10.0, 2);
    let eval = |v: &[f64]| -> Option<f64> { Some(v[0].powi(3) + v[1].powi(3)) };
    let want = [3.0, 12.0];

    for method in [OuterFdMethod::Shi, OuterFdMethod::Gill] {
        let g = central_diff_packed(&x, &fixed, &bounds, method, Some(1e-8), None, eval);
        for k in 0..2 {
            assert!(g[k].is_finite(), "{method:?} coordinate {k} non-finite");
            assert!(
                (g[k] - want[k]).abs() < 1e-3,
                "{method:?} coordinate {k}: {} vs {}",
                g[k],
                want[k]
            );
        }
    }
}

/// No adaptive interval fits — every probe is guard-rejected at every width —
/// so the coordinate must fall back to the fixed stencil rather than record a
/// zero.
///
/// The regression this catches: the first version `continue`d on `None`,
/// leaving `grad[k] = 0.0`. A gradient of zeros is not "unknown" to a gradient
/// optimizer, it is a stationary point, and the fit stops at its start values
/// and reports standard errors for them. Mutating the fallback back to
/// `continue` must redden this.
#[test]
fn adaptive_intervals_fall_back_to_the_fixed_stencil_rather_than_zero() {
    let x = [1.0];
    let fixed = [false];
    let bounds = packed_bounds(-10.0, 10.0, 1);
    // Only the two points of the fixed stencil are scorable — `h = eps·(1+|x|)`
    // with the module's `eps = 1e-4`, so exactly `1 ± 2e-4`. Every adaptive
    // probe, at every width either search tries, lands elsewhere and comes back
    // `None`.  An "everything within ±2e-4 is scorable" region would not do:
    // Gill simply shrinks into it and returns a perfectly good interval of its
    // own (measured: 2.999755), so the fixture would test nothing.
    let h = 1e-4 * (1.0 + 1.0f64.abs());
    let (xp, xm) = (1.0f64 + h, 1.0f64 - h);
    let eval = |v: &[f64]| -> Option<f64> { (v[0] == xp || v[0] == xm).then(|| v[0].powi(3)) };

    for method in [OuterFdMethod::Shi, OuterFdMethod::Gill] {
        let g = central_diff_packed(&x, &fixed, &bounds, method, Some(1e-8), None, eval);
        // The fixed stencil of `x³` is exactly `3x² + h²`.
        assert!(
            (g[0] - (3.0 + h * h)).abs() < 1e-12,
            "{method:?}: expected the fixed-stencil fallback, got {}",
            g[0]
        );
    }
}

/// An adaptive method with no usable noise bound is the same fallback, and the
/// same non-zero requirement. `check_model_options` rejects this configuration
/// up front, but `central_diff_packed` is reachable from callers that never ran
/// it.
#[test]
fn adaptive_intervals_without_a_noise_bound_use_the_fixed_stencil() {
    let x = [2.0];
    let fixed = [false];
    let bounds = packed_bounds(-10.0, 10.0, 1);
    let eval = |v: &[f64]| -> Option<f64> { Some(v[0].powi(3)) };
    let reference =
        central_diff_packed(&x, &fixed, &bounds, OuterFdMethod::Fixed, None, None, eval);
    for noise in [None, Some(0.0)] {
        let g = central_diff_packed(&x, &fixed, &bounds, OuterFdMethod::Shi, noise, None, eval);
        assert_eq!(g, reference, "noise = {noise:?} must fall back exactly");
        assert!(g[0] > 11.0, "and must not be a fabricated zero: {}", g[0]);
    }
}

/// Fixed coordinates are never probed, under any policy.
#[test]
fn adaptive_intervals_skip_fixed_coordinates() {
    let x = [1.0, 2.0];
    let fixed = [true, false];
    let bounds = packed_bounds(-10.0, 10.0, 2);
    let eval = |v: &[f64]| -> Option<f64> {
        assert_eq!(v[0], 1.0, "a fixed coordinate must never be perturbed");
        Some(v[0].powi(3) + v[1].powi(3))
    };
    let g = central_diff_packed(
        &x,
        &fixed,
        &bounds,
        OuterFdMethod::Shi,
        Some(1e-8),
        None,
        eval,
    );
    assert_eq!(g[0], 0.0);
    assert!((g[1] - 12.0).abs() < 1e-3);
}

/// The per-subject noise share is the OFV-scale bound divided by `2·N`, because
/// the OFV is `2·Σᵢ nllᵢ`. Handing the unscaled bound to a per-subject search
/// makes every ratio read as noise and grows `h` to the box cap.
#[test]
fn per_subject_noise_share_divides_by_two_n() {
    assert_eq!(per_subject_noise_abs(Some(1e-4), 50), Some(1e-6));
    assert_eq!(per_subject_noise_abs(None, 50), None);
    // Degenerate population: the divisor must not be zero.
    assert_eq!(per_subject_noise_abs(Some(1e-4), 0), Some(5e-5));
}
