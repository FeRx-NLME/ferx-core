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
        // Off: this trace is pinned to the bare Kingma & Ba update.
        grad_clip: 0.0,
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

// ---------------------------------------------------------------------------
// Gradient clipping (#1097)
// ---------------------------------------------------------------------------

/// The clip is a pure rescale: it caps the length and leaves the direction alone.
#[test]
fn grad_clip_scale_caps_norm_and_preserves_direction() {
    // 3-4-5 triangle: ‖g‖ = 5.
    let g = [3.0, 4.0];

    // Inside the ball, and exactly on it, are both untouched.
    assert_eq!(grad_clip_scale(&g, 10.0), 1.0);
    assert_eq!(grad_clip_scale(&g, 5.0), 1.0);

    // Outside: scaled to exactly the clip length.
    let s = grad_clip_scale(&g, 1.0);
    let clipped = [g[0] * s, g[1] * s];
    let norm = (clipped[0] * clipped[0] + clipped[1] * clipped[1]).sqrt();
    assert!((norm - 1.0).abs() < 1e-12, "norm {norm} should be the clip");

    // Direction preserved: the ratio between components is unchanged. This is the
    // property a per-coordinate `clamp` would destroy, which is why the clip is on
    // the norm.
    assert!((clipped[0] / clipped[1] - g[0] / g[1]).abs() < 1e-12);
}

/// `0` (and any non-positive value) disables clipping, however large the gradient.
#[test]
fn grad_clip_scale_disabled_is_identity() {
    let g = [1e14, -3e13];
    assert_eq!(grad_clip_scale(&g, 0.0), 1.0);
    assert_eq!(grad_clip_scale(&g, -1.0), 1.0);
}

/// A non-finite coordinate must not drive the norm to infinity and thereby scale
/// every *other* coordinate to zero. `step` already treats such entries as zero;
/// the norm has to agree, or one overflowing subject silences the whole vector.
#[test]
fn grad_clip_scale_ignores_non_finite_coordinates() {
    let g = [3.0, f64::NAN, 4.0, f64::INFINITY];
    // Norm is taken over the finite entries only: ‖(3,4)‖ = 5.
    let s = grad_clip_scale(&g, 1.0);
    assert!((s - 0.2).abs() < 1e-12, "expected 1/5, got {s}");

    // An all-non-finite gradient has norm 0, so nothing is rescaled.
    assert_eq!(grad_clip_scale(&[f64::NAN, f64::INFINITY], 1.0), 1.0);
    // A genuinely zero gradient likewise.
    assert_eq!(grad_clip_scale(&[0.0, 0.0], 1.0), 1.0);
}

/// The defect the clip exists for: one catastrophic gradient inflates `v` so far
/// that every later step is numerically zero, and `v` only decays at `beta2` per
/// iteration. Unclipped, a `1e14` spike freezes the optimizer; clipped, the same
/// spike leaves the following steps at their ordinary size.
///
/// Measured over the *steady state* rather than from the spike onwards, because `m`
/// is poisoned as well. Both moments are inflated at first, their ratio is O(1), and
/// the optimizer still moves. The freeze sets in only once `m` has decayed to the
/// scale of the ordinary gradients while `v` has not — and the two decay at very
/// different rates: `m` sheds the eleven orders of magnitude between `1e14` and `1e3`
/// in `ln(1e11)/ln(1/0.9) ≈ 240` iterations, whereas `v` needs
/// `ln(1e22)/ln(1/0.999) ≈ 5e4` to shed twenty-two. That asymmetry is the bug, so the
/// measurement window opens after `m` has settled and well before `v` has.
#[test]
fn grad_clip_prevents_second_moment_poisoning() {
    // A spike, then ordinary gradients of the size a healthy fit produces.
    let spike = 1e14;
    let ordinary = 1e3;
    const WARMUP: usize = 300;
    const MEASURED: usize = 100;

    let run = |clip: f64| -> f64 {
        let cfg = AdamConfig {
            lr: 0.02,
            grad_clip: clip,
            ..Default::default()
        };
        let mut st = AdamState::new(1);
        let mut x = [0.0f64];
        st.step(&mut x, &[spike], &cfg);
        for _ in 0..WARMUP {
            st.step(&mut x, &[ordinary], &cfg);
        }
        let settled = x[0];
        // Still far short of the ~28 000 iterations an unclipped `v` needs to decay
        // back to the scale of these gradients.
        for _ in 0..MEASURED {
            st.step(&mut x, &[ordinary], &cfg);
        }
        (x[0] - settled).abs()
    };

    let frozen = run(0.0);
    let clipped = run(1e4);

    // Unclipped the optimizer is inert: `√v̂ ≈ 2.7e12` against gradients of `1e3` gives
    // steps of ~7e-12, so 100 iterations move it by ~7e-10.
    assert!(
        frozen < 1e-6,
        "expected the unclipped optimizer to be frozen, moved {frozen}"
    );
    // Clipped it takes ordinary Adam steps — approaching `lr` per iteration once the
    // gradient sign is consistent, so ~2 over this window.
    assert!(
        clipped > 0.1,
        "expected the clipped optimizer to move freely, moved {clipped}"
    );
    // The point of the fix, stated as a ratio: orders of magnitude, not a nudge.
    assert!(
        clipped / frozen > 1e6,
        "clipped {clipped} vs frozen {frozen}"
    );
}

/// Clipping is asymptotically a no-op: Adam is scale-invariant in steady state, so a
/// clip that binds uniformly on every iteration leaves the trajectory alone. This is
/// why the threshold is not a sensitive tuning parameter, and why models that already
/// converged are unaffected by turning it on.
#[test]
fn grad_clip_is_a_no_op_when_it_binds_uniformly() {
    let grads = [[2.0, -1.0], [3.0, -2.0], [1.0, -0.5], [2.5, -1.5]];

    let run = |clip: f64| -> [f64; 2] {
        let cfg = AdamConfig {
            lr: 0.02,
            grad_clip: clip,
            ..Default::default()
        };
        let mut st = AdamState::new(2);
        let mut x = [0.0f64; 2];
        for g in grads.iter() {
            st.step(&mut x, g, &cfg);
        }
        x
    };

    // A clip small enough to bind on every iteration rescales each gradient, and Adam's
    // `m̂/√v̂` is invariant to a per-iteration rescale only in the limit — over a handful
    // of steps the bias corrections leave a small difference, so this asserts closeness
    // rather than equality.
    let unclipped = run(0.0);
    let clipped = run(0.5);
    for k in 0..2 {
        assert!(
            (unclipped[k] - clipped[k]).abs() < 5e-3,
            "coord {k}: {} vs {}",
            unclipped[k],
            clipped[k]
        );
    }
}
