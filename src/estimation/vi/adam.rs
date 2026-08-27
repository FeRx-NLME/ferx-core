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
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
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
    pub fn step(&mut self, x: &mut [f64], grad: &[f64], cfg: &AdamConfig) {
        debug_assert_eq!(x.len(), self.m.len());
        debug_assert_eq!(grad.len(), self.m.len());

        self.t += 1;
        // Bias-correction denominators. `beta^t` underflows to 0 for large t,
        // which is the correct limit: the correction factor tends to 1.
        let bc1 = 1.0 - cfg.beta1.powi(self.t as i32);
        let bc2 = 1.0 - cfg.beta2.powi(self.t as i32);

        for i in 0..x.len() {
            let g = if grad[i].is_finite() { grad[i] } else { 0.0 };
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
