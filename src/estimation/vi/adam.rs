//! Adam, plus the Polyak parameter averaging a stochastic optimizer needs to
//! report a point estimate.
//!
//! Adam (Kingma & Ba 2015) is the optimizer variational inference runs on: the
//! ELBO gradient is a Monte-Carlo estimate, so the derivative-free and
//! quasi-Newton optimizers the rest of `ferx-core` uses (BOBYQA, SLSQP, L-BFGS)
//! do not apply — they assume a deterministic objective.
//!
//! Nothing here is VI-specific; it is a plain optimizer over `f64` slices.
//!
//! # Why averaging is not optional
//!
//! The last iterate of a stochastic optimizer is a *draw*, not an estimate: it
//! carries the full gradient noise of the final step. Reporting it would make
//! two runs of the same fit disagree by more than the estimator's actual
//! accuracy. [`PolyakAverager`] accumulates the tail of the trajectory and
//! reports its mean, which is what Janssen et al. (2024) do (they average the
//! final 500 of 2000 epochs) and what the classical Polyak–Ruppert result
//! recommends.

/// Adam hyperparameters. Defaults are the canonical values from the paper,
/// except `lr`, which is set by the caller from `vi_lr`.
#[derive(Debug, Clone, Copy)]
pub struct AdamConfig {
    /// Step size. Janssen et al. use 0.1 (dropping to 0.01 when unstable).
    pub lr: f64,
    /// First-moment decay.
    pub beta1: f64,
    /// Second-moment decay.
    pub beta2: f64,
    /// Denominator floor, guarding division by a zero second moment.
    pub eps: f64,
    /// Global-L2 gradient clip. `0.0` disables clipping.
    ///
    /// When `‖g‖₂` exceeds this, the whole vector is rescaled to length
    /// `grad_clip`. The *direction* is untouched — only the magnitude is capped,
    /// so clipping never redirects the step, it only refuses to take a violent
    /// one. See [`AdamState::step`] for why that matters here.
    pub grad_clip: f64,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            grad_clip: 0.0,
        }
    }
}

/// Factor that rescales `grad` to length `clip`, or `1.0` when no clipping applies.
///
/// Returns `1.0` for a disabled clip (`clip <= 0`), for a gradient already inside the
/// ball, and for a zero or non-finite norm — in every one of those cases the caller
/// should use the gradient exactly as given.
///
/// Split out from [`AdamState::step`] so the geometry (direction preserved, length
/// capped) is testable without an optimizer around it.
pub fn grad_clip_scale(grad: &[f64], clip: f64) -> f64 {
    // Disabled by a non-positive threshold; also short-circuits a NaN or infinite one,
    // for which "never clip" is the only sensible reading.
    if !clip.is_finite() || clip <= 0.0 {
        return 1.0;
    }
    let norm = grad
        .iter()
        .map(|g| if g.is_finite() { g * g } else { 0.0 })
        .sum::<f64>()
        .sqrt();
    if norm.is_finite() && norm > clip {
        clip / norm
    } else {
        1.0
    }
}

/// Per-coordinate Adam moments. One of these per independently-stepped
/// parameter block (the packed population vector, and one per subject's `φ`).
#[derive(Debug, Clone)]
pub struct AdamState {
    /// Exponential moving average of the gradient.
    m: Vec<f64>,
    /// Exponential moving average of the squared gradient.
    v: Vec<f64>,
    /// Step counter, for bias correction. Shared across all coordinates.
    t: u64,
}

impl AdamState {
    pub fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }

    /// Number of coordinates this state tracks.
    pub fn dim(&self) -> usize {
        self.m.len()
    }

    /// Steps taken so far.
    pub fn steps(&self) -> u64 {
        self.t
    }

    /// One Adam step, updating `x` in place by **descending** `grad`.
    ///
    /// `grad` must be the gradient of the quantity being *minimized*. VI
    /// maximizes the ELBO, so its caller passes `−∇ELBO`.
    ///
    /// Non-finite gradient entries are treated as zero rather than being allowed
    /// to poison `m`/`v` permanently: a single overflow in one subject's
    /// likelihood would otherwise leave `v = NaN` for the rest of the fit, and
    /// every subsequent step for that coordinate would produce `NaN`. Skipping
    /// the update leaves the coordinate where it was, which is recoverable.
    /// Callers that need to know this happened should check their own gradient.
    ///
    /// # Why the gradient is clipped
    ///
    /// A finite but enormous gradient does not need to be `NaN` to disable the
    /// optimizer, because `v` is an average of *squares* with a long memory. At the
    /// default `beta2 = 0.999`, one gradient of `1e14` leaves
    /// `v ≈ 0.001·(1e14)² = 1e25`, so `√v̂ ≈ 1e12`. Once the fit recovers and ordinary
    /// gradients are `~1e3`, the step is `lr·1e3/1e12 ≈ 1e-9·lr` — zero for practical
    /// purposes — and `v` sheds those twelve orders of magnitude only at `0.999` per
    /// iteration, i.e. over ~28 000 steps. The optimizer is frozen, and because a frozen
    /// trace is a flat trace, the early-stopping rule then certifies it as converged.
    ///
    /// VI reaches gradients like that routinely from ordinary starting values: under a
    /// proportional error model a prediction near zero makes `(y−f)²/(σf)²` explode. On
    /// warfarin from the `ferx-testdata` initial estimates, iteration 0 evaluates at
    /// `−2·ELBO = 1.5e14` and the fit then sits at `σ = exp(5)` — its runaway bound —
    /// reporting estimates ~1594 objective units short of FOCEI (#1097).
    ///
    /// Clipping bounds `‖g‖₂`, hence bounds `v` by `grad_clip²`, so the ratio between a
    /// catastrophic gradient and an ordinary one can never grow large enough to stall the
    /// step. It is close to a no-op once a fit is healthy: Adam is scale-invariant in
    /// steady state (scaling every gradient by `k` leaves `m̂/√v̂` unchanged), so a clip
    /// that binds uniformly does not move the trajectory. That is why the threshold is not
    /// sensitive — on warfarin every value from `1` to `1e5` recovers the FOCEI estimates
    /// and only `0` fails — and why two models that already converged are unchanged to
    /// seven significant figures with clipping on.
    ///
    /// Clipping the *norm* rather than each coordinate is deliberate: a per-coordinate
    /// `clamp` shortens the largest components relative to the rest and so points the step
    /// somewhere the gradient does not.
    pub fn step(&mut self, x: &mut [f64], grad: &[f64], cfg: &AdamConfig) {
        debug_assert_eq!(x.len(), self.m.len());
        debug_assert_eq!(grad.len(), self.m.len());

        self.t += 1;
        // Bias-correction denominators. `beta^t` underflows to 0 for large t,
        // which is the correct limit: the correction factor tends to 1.
        let bc1 = 1.0 - cfg.beta1.powi(self.t as i32);
        let bc2 = 1.0 - cfg.beta2.powi(self.t as i32);

        // Uniform rescale factor, applied inline below so no copy of `grad` is made.
        // Non-finite entries are excluded from the norm for the same reason `step`
        // treats them as zero: one overflowing subject must not silence every other
        // coordinate by driving the norm to infinity.
        let scale = grad_clip_scale(grad, cfg.grad_clip);

        for i in 0..x.len() {
            let g = if grad[i].is_finite() {
                grad[i] * scale
            } else {
                0.0
            };
            self.m[i] = cfg.beta1 * self.m[i] + (1.0 - cfg.beta1) * g;
            self.v[i] = cfg.beta2 * self.v[i] + (1.0 - cfg.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            x[i] -= cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
        }
    }
}

/// Running mean of the tail of an optimization trajectory (Polyak–Ruppert).
///
/// The caller decides *when* to start accumulating (see [`averaging_start`]);
/// this only holds the running sum.
#[derive(Debug, Clone)]
pub struct PolyakAverager {
    sum: Vec<f64>,
    count: usize,
}

impl PolyakAverager {
    pub fn new(n: usize) -> Self {
        Self {
            sum: vec![0.0; n],
            count: 0,
        }
    }

    /// Fold one iterate into the running mean.
    pub fn accumulate(&mut self, x: &[f64]) {
        debug_assert_eq!(x.len(), self.sum.len());
        for (s, &xi) in self.sum.iter_mut().zip(x.iter()) {
            *s += xi;
        }
        self.count += 1;
    }

    /// How many iterates have been folded in.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The averaged iterate, or `None` if nothing was accumulated (so the
    /// caller falls back to the last iterate rather than dividing by zero).
    pub fn mean(&self) -> Option<Vec<f64>> {
        if self.count == 0 {
            return None;
        }
        let n = self.count as f64;
        Some(self.sum.iter().map(|s| s / n).collect())
    }
}

/// Default averaging window: the final quarter of the run.
pub const DEFAULT_AVG_FRACTION: f64 = 0.25;

/// First iteration index (0-based) whose iterate should be averaged.
///
/// `avg_last = Some(k)` averages the final `k` iterations; `None` averages the
/// final [`DEFAULT_AVG_FRACTION`] of them. The result is always in
/// `0..n_iters`, and always leaves at least one iteration to average, so a
/// 1-iteration run still reports that iteration rather than nothing.
pub fn averaging_start(n_iters: usize, avg_last: Option<usize>) -> usize {
    if n_iters == 0 {
        return 0;
    }
    let window = match avg_last {
        Some(k) => k.max(1),
        None => (((n_iters as f64) * DEFAULT_AVG_FRACTION).round() as usize).max(1),
    };
    n_iters.saturating_sub(window)
}

#[cfg(test)]
#[path = "adam_tests.rs"]
mod tests;
