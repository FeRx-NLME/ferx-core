//! Unit tests for the stochastic optimizer.

use super::*;

/// Adam's first step is `lr` in magnitude essentially regardless of gradient
/// scale — bias correction makes `m̂/√v̂ = g/|g|` at `t = 1`, so the step is
/// `lr·|g|/(|g| + eps)`. This scale invariance is what makes one learning rate
/// workable across NN weights and log-σ simultaneously, so it is worth pinning.
///
/// The invariance is exact only for `|g| ≫ eps`; the `eps` floor is what damps
/// a coordinate whose gradient has genuinely vanished, and the test asserts that
/// crossover explicitly rather than papering over it with a loose tolerance.
#[test]
fn first_step_is_lr_sized_regardless_of_gradient_magnitude() {
    let cfg = AdamConfig {
        lr: 0.05,
        ..Default::default()
    };
    for &g in &[1e-6_f64, 1.0, 1e6] {
        let mut st = AdamState::new(1);
        let mut x = [0.0];
        st.step(&mut x, &[g], &cfg);
        let want = -cfg.lr * g.abs() / (g.abs() + cfg.eps);
        assert!(
            (x[0] - want).abs() < 1e-15,
            "grad {g}: expected step {want}, got {}",
            x[0]
        );
        // Well above the eps floor, the step is lr to within 1%.
        assert!(
            (x[0].abs() / cfg.lr - 1.0).abs() < 0.01,
            "grad {g}: step {} is not lr-sized",
            x[0]
        );
    }

    // At the eps floor the damping is real and must not be silently absorbed:
    // a gradient equal to eps takes a half-sized step.
    let mut st = AdamState::new(1);
    let mut x = [0.0];
    st.step(&mut x, &[cfg.eps], &cfg);
    assert!(
        (x[0] + cfg.lr / 2.0).abs() < 1e-15,
        "at |g| = eps the step should be lr/2, got {}",
        x[0]
    );
}

/// Hand-computed three-step trace on a constant gradient, checked against the
/// Kingma & Ba update written out longhand.
#[test]
fn three_steps_match_hand_computed_trace() {
    let cfg = AdamConfig {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    };
    let g = 2.0_f64;

    let mut st = AdamState::new(1);
    let mut x = [1.0_f64];

    // Reference: run the update by hand.
    let (mut m, mut v) = (0.0_f64, 0.0_f64);
    let mut want = 1.0_f64;
    for t in 1..=3 {
        m = cfg.beta1 * m + (1.0 - cfg.beta1) * g;
        v = cfg.beta2 * v + (1.0 - cfg.beta2) * g * g;
        let m_hat = m / (1.0 - cfg.beta1.powi(t));
        let v_hat = v / (1.0 - cfg.beta2.powi(t));
        want -= cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);

        st.step(&mut x, &[g], &cfg);
        assert!(
            (x[0] - want).abs() < 1e-12,
            "step {t}: got {}, want {want}",
            x[0]
        );
    }
    assert_eq!(st.steps(), 3);
}

/// A constant positive gradient must walk the parameter monotonically downhill.
#[test]
fn descends_a_constant_gradient() {
    let cfg = AdamConfig::default();
    let mut st = AdamState::new(1);
    let mut x = [10.0_f64];
    let mut prev = x[0];
    for _ in 0..50 {
        st.step(&mut x, &[1.0], &cfg);
        assert!(x[0] < prev, "expected monotone descent");
        prev = x[0];
    }
}

/// Adam on a quadratic must approach its minimum.
#[test]
fn converges_on_a_quadratic() {
    // f(x) = ½(x − 3)², ∇f = x − 3.
    let cfg = AdamConfig {
        lr: 0.1,
        ..Default::default()
    };
    let mut st = AdamState::new(1);
    let mut x = [0.0_f64];
    for _ in 0..2000 {
        let g = x[0] - 3.0;
        st.step(&mut x, &[g], &cfg);
    }
    assert!((x[0] - 3.0).abs() < 1e-3, "converged to {}", x[0]);
}

/// A non-finite gradient entry must not poison the moment estimates. Without the
/// guard, one overflowing subject leaves `v = NaN` and every later step for that
/// coordinate produces `NaN` — the fit is silently dead from then on.
#[test]
fn non_finite_gradient_does_not_poison_state() {
    let cfg = AdamConfig::default();
    let mut st = AdamState::new(2);
    let mut x = [0.0_f64, 0.0];

    st.step(&mut x, &[f64::NAN, 1.0], &cfg);
    assert!(x[0].is_finite(), "NaN gradient produced NaN parameter");
    assert!(x[1] < 0.0, "finite coordinate should still descend");

    // And the poisoned coordinate must still respond to a later good gradient.
    st.step(&mut x, &[1.0, 1.0], &cfg);
    assert!(x[0].is_finite() && x[0] < 0.0, "coordinate 0 stayed dead");

    st.step(&mut x, &[f64::INFINITY, 1.0], &cfg);
    assert!(x[0].is_finite(), "infinite gradient produced non-finite x");
}

/// The averager reports the arithmetic mean of what it was given, and `None`
/// before anything is accumulated (so the caller falls back to the last iterate
/// rather than dividing by zero).
#[test]
fn polyak_averager_means_accumulated_iterates() {
    let mut avg = PolyakAverager::new(2);
    assert!(avg.mean().is_none());
    assert_eq!(avg.count(), 0);

    avg.accumulate(&[1.0, 10.0]);
    avg.accumulate(&[2.0, 20.0]);
    avg.accumulate(&[6.0, 30.0]);

    let m = avg.mean().expect("three iterates accumulated");
    assert!((m[0] - 3.0).abs() < 1e-12);
    assert!((m[1] - 20.0).abs() < 1e-12);
    assert_eq!(avg.count(), 3);
}

/// The averaging window: an explicit `avg_last` wins; otherwise the final
/// quarter. Always leaves at least one iteration to average.
#[test]
fn averaging_start_picks_the_tail_window() {
    // Default: final 25%.
    assert_eq!(averaging_start(2000, None), 1500);
    assert_eq!(averaging_start(100, None), 75);

    // Explicit window.
    assert_eq!(averaging_start(2000, Some(500)), 1500);
    assert_eq!(averaging_start(100, Some(10)), 90);

    // A window larger than the run averages the whole run rather than underflowing.
    assert_eq!(averaging_start(10, Some(1000)), 0);

    // Degenerate runs still average at least the single iteration they have.
    assert_eq!(averaging_start(1, None), 0);
    assert_eq!(averaging_start(3, Some(0)), 2);
    assert_eq!(averaging_start(0, None), 0);
}
