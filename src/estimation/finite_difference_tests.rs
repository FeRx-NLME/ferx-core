use super::*;
use std::cell::Cell;

const BOUNDS: AxisBounds = AxisBounds {
    lower: -10.0,
    upper: 10.0,
};

const WIDE: AxisBounds = AxisBounds {
    lower: f64::NEG_INFINITY,
    upper: f64::INFINITY,
};

/// Deterministic pseudo-noise of amplitude `amp`, keyed on the probe point: the
/// same `x` always returns the same perturbation, so the evaluator stays a
/// *function* and the test is reproducible.  A smooth ripple (`sin(1e6·x)`)
/// would not do — its second difference is a genuine curvature the search could
/// legitimately resolve, which is the opposite of what these tests need.
fn ripple(x: f64, amp: f64) -> f64 {
    let bits = x.to_bits().wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let unit = ((bits >> 11) as f64) / ((1u64 << 53) as f64);
    amp * (2.0 * unit - 1.0)
}

/// Counts evaluations so a test can tell "accepted an interval" from "spent the
/// whole budget and returned the last thing it tried".
fn counted<F: Fn(f64) -> Option<f64>>(
    f: F,
) -> (impl Fn(f64) -> Option<f64>, std::rc::Rc<Cell<usize>>) {
    let n = std::rc::Rc::new(Cell::new(0));
    let seen = n.clone();
    (
        move |x: f64| {
            seen.set(seen.get() + 1);
            f(x)
        },
        n,
    )
}

#[test]
fn fixed_method_selects_no_adaptive_interval() {
    // `Fixed` must not silently route to an adaptive search: its stencil lives
    // in the caller, and `None` is the signal to use it.
    let d = adaptive_first_derivative(OuterFdMethod::Fixed, 2.0, 0.01, BOUNDS, 1e-8, |x| {
        Some(x * x * x)
    });
    assert!(d.is_none());
}

#[test]
fn shi_recovers_cubic_derivative() {
    let (eval, evals) = counted(|x: f64| Some(x * x * x));
    let d = adaptive_first_derivative(OuterFdMethod::Shi, 2.0, 0.01, BOUNDS, 1e-8, eval).unwrap();
    assert!(d.accepted, "ratio test must accept, not exhaust the budget");
    // Measured: 1.5625e-6, which is exactly the accepted interval's own
    // truncation `h² = 0.00125²` — the search is the only error source here.
    assert!((d.derivative[0] - 12.0).abs() < 3e-6);
    // 4 probes per trial interval, so exhausting the budget is 4·MAX_SEARCH_ITERS
    // evaluations — which is exactly what the pre-fix factor-4 window did here
    // (60 evaluations, no acceptance). Bounding the count rather than pinning it
    // keeps this green for any correctly-sized window, not just this one.
    assert!(
        evals.get() < 4 * MAX_SEARCH_ITERS,
        "search exhausted its budget ({} evaluations)",
        evals.get()
    );
}

/// The acceptance window's *width* is the property, so assert the property and
/// not one particular pair of endpoints — `[1.5, 12.0]` would be as correct as
/// `[1.0, 8.0]`, and a test that pins `RATIO_LOW` would redden on it.
///
/// `Δ³f(h) ≈ 8h³f'''`, so halving `h` divides the ratio by 8 and the reachable
/// ratios are `{r₀·8ᵏ}`. A closed window spanning a factor of at least 8 always
/// contains one of them; anything narrower can sit between two consecutive
/// members. The pre-fix `[1.5, 6.0]` spanned 4, and on `f = x³, x = 2` it was
/// straddled by `r(0.00125) = 1.1719` and `r(0.0025) = 9.375`.
#[test]
fn shi_acceptance_window_spans_at_least_one_halving() {
    assert!(
        RATIO_HIGH / RATIO_LOW >= 8.0,
        "window [{RATIO_LOW}, {RATIO_HIGH}] spans {}, below the factor of 8 the \
         search steps by; it can be straddled and never accept",
        RATIO_HIGH / RATIO_LOW
    );
    // And on the fixture that straddled the old one, the search now accepts.
    let d = shi_first_derivative(2.0, 0.01, BOUNDS, 1e-8, |x| Some(x * x * x)).unwrap();
    assert!(d.accepted, "accepted at h = {}", d.h);
    let ratio = 6e8 * d.h.powi(3);
    assert!(
        (RATIO_LOW..=RATIO_HIGH).contains(&ratio),
        "accepted ratio {ratio} outside the window it was accepted by"
    );
}

/// Gill's second difference moves by a factor of 4 per halving, so its window
/// carries the same reachability requirement at that factor.
#[test]
fn gill_acceptance_window_spans_at_least_one_halving() {
    assert!(
        GILL_RATIO_HIGH / GILL_RATIO_LOW >= 4.0,
        "window [{GILL_RATIO_LOW}, {GILL_RATIO_HIGH}] spans {}, below the factor \
         of 4 the search steps by",
        GILL_RATIO_HIGH / GILL_RATIO_LOW
    );
}

/// Every start must reach the window, for any curvature: that is what a closed
/// factor-8 window buys.  Ratios reachable from a given start are `{r₀·8ᵏ}`, so
/// this sweep walks `r₀` across a full period of that lattice and more.
#[test]
fn shi_window_is_reachable_from_every_start() {
    for scale_exp in -6i32..=6 {
        let scale = 10f64.powi(scale_exp);
        for &h0 in &[1e-1, 1e-2, 1e-3, 1e-4] {
            let d = shi_first_derivative(2.0, h0, BOUNDS, 1e-8, |x| Some(scale * x * x * x))
                .unwrap_or_else(|| panic!("no interval for scale {scale}, h0 {h0}"));
            assert!(
                d.accepted,
                "scale {scale}, h0 {h0}: search exhausted instead of accepting"
            );
            let got = d.derivative[0];
            // A central difference of `s·x³` is `s·(3x² + h²)` exactly, so the
            // accepted interval's own truncation is the reference — not a
            // hand-picked tolerance that would silently absorb a wrong `h`.
            let want = scale * (12.0 + d.h * d.h);
            assert!(
                got.is_finite(),
                "scale {scale}, h0 {h0}: non-finite derivative"
            );
            assert!(
                (got - want).abs() <= 1e-9 * want.abs(),
                "scale {scale}, h0 {h0}: {got} vs {want} at h = {}",
                d.h
            );
        }
    }
}

/// A guard-rejected probe must shrink the interval, not abandon the coordinate.
/// The caller's fallback for `None` is a *fixed* stencil, and the caller before
/// this fix recorded a fabricated `0.0`; either way the search itself has room
/// to move inward and should use it.
#[test]
fn shi_shrinks_past_a_rejected_probe() {
    // Only |x - 2| <= 0.004 is scorable, so the initial h = 0.01 has both ±3h
    // arms (±0.03) rejected, as do the first two halvings.
    let d = shi_first_derivative(2.0, 0.01, BOUNDS, 1e-8, |x| {
        ((x - 2.0f64).abs() <= 0.004).then(|| x * x * x)
    })
    .expect("search must find a usable interval inside the scorable region");
    assert!(d.accepted);
    assert!(d.h * 3.0 <= 0.004, "accepted stencil must stay scorable");
    assert!((d.derivative[0] - 12.0).abs() < 1e-3);
}

/// No usable stencil at all → `None`, so the caller falls back to its own
/// fixed difference. Recording a zero derivative here reads to the optimizer as
/// "this coordinate is stationary" and can end a fit at its start values.
#[test]
fn shi_reports_no_interval_when_every_probe_is_rejected() {
    let d = shi_first_derivative(2.0, 0.01, BOUNDS, 1e-8, |_| None);
    assert!(d.is_none());
}

#[test]
fn shi_reports_no_interval_at_a_closed_bound() {
    let at_bound = AxisBounds {
        lower: -1.0,
        upper: 2.0,
    };
    assert!(shi_first_derivative(2.0, 0.1, at_bound, 1e-8, |x| Some(x * x * x)).is_none());
}

#[test]
fn shi_stays_inside_its_bounds() {
    let bounds = AxisBounds {
        lower: -1.0,
        upper: 1.0,
    };
    let d = shi_first_derivative(0.99, 0.1, bounds, 1e-8, |x| {
        assert!(
            (-1.0..=1.0).contains(&x),
            "probed {x}, outside the supplied bounds"
        );
        Some(x * x)
    });
    assert!(d.is_some());
}

/// A near-quadratic coordinate has `f''' = 0`, so its ratio is 0 however wide
/// `h` grows and the ratio test *never* endorses an interval. The estimate is
/// still the best available and is returned — but flagged `accepted = false`,
/// which is precisely the distinction the first version of this module could
/// not express.
#[test]
fn shi_does_not_claim_acceptance_when_the_bounds_stopped_the_search() {
    let d = shi_first_derivative(2.0, 0.01, BOUNDS, 1e-8, |x| Some(x * x)).unwrap();
    assert!(
        !d.accepted,
        "a bound-limited interval must not report as ratio-accepted"
    );
    assert!((d.derivative[0] - 4.0).abs() < 1e-9);
    // Growth stopped at the widest stencil the box admits: cap = min(12, 8)/3.
    assert!(d.h <= BOUNDS.central_cap(2.0) + 1e-12);
}

/// `noise = 0` makes `r = |Δ³f| / 0` infinite for every interval, so no search
/// is meaningful. `Some(0.0)` reaching here from the Rust API is the failure
/// `check_model_options` exists to stop; this is the second line of defence.
#[test]
fn shi_rejects_a_non_positive_noise_bound() {
    assert!(shi_first_derivative(2.0, 0.01, BOUNDS, 0.0, |x| Some(x * x * x)).is_none());
    assert!(shi_first_derivative(2.0, 0.01, BOUNDS, -1.0, |x| Some(x * x * x)).is_none());
    assert!(shi_first_derivative(2.0, 0.01, BOUNDS, f64::NAN, |x| Some(x * x * x)).is_none());
}

#[test]
fn vector_shi_recovers_component_derivatives() {
    let d =
        shi_central_vector_derivative(2.0, 0.01, BOUNDS, 1e-8, |x| Some(vec![x * x, x * x * x]))
            .unwrap();
    assert!(d.accepted);
    assert!((d.derivative[0] - 4.0).abs() < 1e-3);
    assert!((d.derivative[1] - 12.0).abs() < 1e-3);
}

/// The scalar and vector entry points are one implementation; a divergence in
/// the representability guard or the non-finite policy is what having two
/// copies used to permit. Bit-equality, not a tolerance.
#[test]
fn scalar_and_vector_shi_are_the_same_search() {
    let scalar = shi_first_derivative(2.0, 0.01, BOUNDS, 1e-8, |x| Some(x * x * x)).unwrap();
    let vector =
        shi_central_vector_derivative(2.0, 0.01, BOUNDS, 1e-8, |x| Some(vec![x * x * x])).unwrap();
    assert_eq!(scalar.h, vector.h);
    assert_eq!(scalar.accepted, vector.accepted);
    assert_eq!(scalar.derivative, vector.derivative);
}

#[test]
fn gill_uses_a_one_sided_derivative() {
    let d = adaptive_first_derivative(OuterFdMethod::Gill, 1.0, 0.01, BOUNDS, 1e-8, |x| {
        Some(x * x)
    })
    .unwrap();
    assert!(d.accepted);
    // Measured: 1.41421e-4. The one-sided truncation term is ½·h·f'' = h at the
    // balanced h = 2·√(noise/|f''|) = 1.41421e-4, so the bound below is that
    // value with ~2x headroom.
    let err = (d.derivative[0] - 2.0).abs();
    assert!(err < 3e-4, "realised one-sided error {err}");
}

/// A coordinate pinned at its upper bound has no forward room at all. A
/// forward-only rule returns "no interval" there and the caller falls back to a
/// fixed stencil that would probe outside the box; picking the roomier side
/// differentiates it properly.
#[test]
fn gill_differentiates_a_coordinate_at_its_upper_bound() {
    let d = adaptive_first_derivative(OuterFdMethod::Gill, 10.0, 0.01, BOUNDS, 1e-8, |x| {
        assert!(x <= 10.0, "probed {x}, above the upper bound");
        Some(x * x)
    })
    .unwrap();
    // Measured: 1.41421e-4 — the same balanced interval as the forward case, so
    // the backward direction costs nothing in accuracy.
    let err = (d.derivative[0] - 20.0).abs();
    assert!(err < 3e-4, "realised one-sided error {err}");
}

/// The curvature that sets Gill's interval is searched, not sampled once. With
/// noise of amplitude 1e-8 the second difference at h₀ = 1e-6 is `2·h₀² = 2e-12`
/// of signal under up to 4e-8 of noise — unresolvable, so a single-shot
/// curvature is a measurement of the noise. The search must widen until the
/// second difference clears the noise bound before believing it.
#[test]
fn gill_searches_for_a_resolvable_curvature() {
    let f = |x: f64| x * x + ripple(x, 1e-8);
    let h0 = 1e-6;

    // Straddle: assert the start really is noise-dominated, so this fixture
    // cannot quietly become a test of an already-resolvable interval.
    let single_shot =
        (f(1.0 + 2.0 * h0) - 2.0 * f(1.0 + h0) + f(1.0)).abs() / (GILL_NOISE_WEIGHT * 1e-8);
    assert!(
        single_shot < GILL_RATIO_LOW,
        "fixture no longer noise-dominated at h0: ratio {single_shot}"
    );

    let d = adaptive_first_derivative(OuterFdMethod::Gill, 1.0, h0, WIDE, 1e-8, |x| Some(f(x)))
        .unwrap();
    assert!(
        d.accepted,
        "curvature must be resolved, not inferred from noise"
    );
    assert!(
        d.h > 1e-5,
        "search must widen well past the noise-dominated start, got h = {}",
        d.h
    );
    // Measured: 4.7926e-5 on the noisy quadratic.
    let err = (d.derivative[0] - 2.0).abs();
    assert!(err < 2e-4, "realised error {err} on the noisy quadratic");
}

#[test]
fn gill_rejects_a_non_positive_noise_bound() {
    assert!(
        adaptive_first_derivative(OuterFdMethod::Gill, 1.0, 0.01, BOUNDS, 0.0, |x| Some(x * x))
            .is_none()
    );
}
