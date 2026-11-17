//! Automatic end of SAEM's exploration phase.
//!
//! The exploration phase runs the stochastic approximation at full step
//! (γ = 1) so the iterates can travel to the neighbourhood of the maximiser;
//! the convergence phase then averages with γ_k = 1/(k − K1)^a and freezes
//! whatever the iterates had reached. Ending the first phase too early freezes
//! a *still-moving* parameter: weakly identified Ω entries decay slowly under
//! the exploration step cap (`OMEGA_SA_MAX_STEP`) and the averaging then pins
//! them well above their optimum (pk2_big: ω²_Q 0.019 against ≈ 0.003).
//!
//! # The stop rule
//!
//! Monolix ends the phase by a convergence test, between a minimum of 150 and a
//! maximum of 500 iterations; NONMEM's SAEM burn-in stops (`CTYPE`, `CITER`,
//! `CALPHA`) when a linear regression of each monitored quantity on the
//! iteration number over a trailing window has a slope that is not
//! significantly non-zero. This module is that test with the one correction the
//! statistics demand: SAEM trajectories are **autocorrelated** (θ is the
//! M-step of a Markov chain; Ω is further smoothed by the exploration step cap
//! `γ_Ω ≤ OMEGA_SA_MAX_STEP = 0.1`, an exponential filter with a time constant
//! of 10 iterations), and an ordinary least-squares standard error treats the
//! iterates as independent, reading every wobble as a trend (measured on
//! stationary SAEM traces: 20-40 % of windows flagged for log Ω at a nominal
//! 5 %, so the rule would essentially never stop).
//!
//! The standard remedy is **batch means** (Glynn & Iglehart 1990; Geweke 1992
//! for MCMC diagnostics): the trailing window is cut into [`N_BATCHES`]
//! batches of [`BATCH_LEN`] iterations, each at least twice the Ω filter's time
//! constant so the batch means are close to independent, and the least-squares
//! slope of the batch means on the batch index is tested with a Student t
//! (`N_BATCHES − 2` degrees of freedom, two-sided 5 %). A coordinate "drifts"
//! when that test rejects. Exploration stops at the first iteration ≥ the
//! minimum at which **no** tracked coordinate drifts. The level is
//! deliberately not multiplicity-corrected: a Bonferroni correction would make
//! the rule stop *sooner*, the wrong direction for a rule whose failure is
//! stopping early; the maximum bounds the cost of the opposite error.
//!
//! Tracked coordinates (scale-free in the test): log Ω_jj, the correlations
//! Ω_jk/√(Ω_jjΩ_kk), log σ, and θ. A series that is exactly constant (fixed
//! parameters, structural zeros) cannot drift.

use nalgebra::DMatrix;

/// Batches per test window.
pub(crate) const N_BATCHES: usize = 5;
/// Iterations per batch: twice the time constant `1/γ_Ω = 10` of the capped Ω
/// stochastic-approximation filter (`OMEGA_SA_MAX_STEP`).
pub(crate) const BATCH_LEN: usize = 20;
/// Trailing window (iterations) of the trend test.
pub(crate) const TREND_WINDOW: usize = N_BATCHES * BATCH_LEN;
/// Two-sided 5 % Student-t quantile, `N_BATCHES − 2 = 3` degrees of freedom.
pub(crate) const TREND_T: f64 = 3.182_446_305_284_263;

/// `true` when the least-squares slope of the batch means of `y` (the last
/// [`TREND_WINDOW`] values) is significantly non-zero, see the module docs. A
/// non-finite series counts as drifting; an exactly constant one does not.
pub(crate) fn trend_detected(y: &[f64]) -> bool {
    if y.len() < TREND_WINDOW {
        return false;
    }
    let y = &y[y.len() - TREND_WINDOW..];
    if y.iter().any(|v| !v.is_finite()) {
        return true;
    }
    let m: Vec<f64> = y
        .chunks(BATCH_LEN)
        .map(|c| c.iter().sum::<f64>() / c.len() as f64)
        .collect();
    let b = m.len() as f64;
    let xbar = (b - 1.0) / 2.0;
    let mbar = m.iter().sum::<f64>() / b;
    let sxx: f64 = (0..m.len()).map(|i| (i as f64 - xbar).powi(2)).sum();
    let slope = m
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64 - xbar) * (v - mbar))
        .sum::<f64>()
        / sxx;
    let intercept = mbar - slope * xbar;
    let ss: f64 = m
        .iter()
        .enumerate()
        .map(|(i, v)| (v - intercept - slope * i as f64).powi(2))
        .sum();
    // Numerically constant: the fitted change across the window is at rounding level of the
    // series scale. (Not "zero residual": a noiseless exact ramp has zero residual and is a trend.)
    let scale = y.iter().fold(0.0_f64, |a, v| a.max(v.abs())).max(1e-300);
    if slope.abs() * (b - 1.0) <= 1e-12 * scale {
        return false;
    }
    let se = (ss / (b - 2.0) / sxx).sqrt();
    slope.abs() > TREND_T * se
}

/// `(automatic, minimum, maximum)` exploration length for a run.
///
/// Precedence: an explicit `n_exploration` is a fixed length — automatic stop
/// off, `min = max = n`, and `n_exploration_min` / `n_exploration_max` are
/// ignored. Otherwise the stop is automatic between `n_exploration_min` and
/// `n_exploration_max` (a maximum below the minimum is raised to it).
pub(crate) fn exploration_bounds(options: &crate::types::FitOptions) -> (bool, usize, usize) {
    match options.saem_n_exploration {
        Some(n) => (false, n, n),
        None => (
            true,
            options.saem_exploration_min,
            options
                .saem_exploration_max
                .max(options.saem_exploration_min),
        ),
    }
}

/// Per-iteration record of the tracked coordinates and the stop decision.
#[derive(Default)]
pub(crate) struct ExplorationMonitor {
    /// `series[c][k]`: coordinate `c` after exploration iteration `k + 1`.
    series: Vec<Vec<f64>>,
}

impl ExplorationMonitor {
    pub(crate) fn push(&mut self, coords: &[f64]) {
        if self.series.is_empty() {
            self.series = vec![Vec::new(); coords.len()];
        }
        for (s, c) in self.series.iter_mut().zip(coords) {
            s.push(*c);
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.series.first().map_or(0, Vec::len)
    }

    /// No tracked coordinate drifts over the trailing window. `false` until a
    /// full window exists.
    pub(crate) fn converged(&self) -> bool {
        if self.len() < TREND_WINDOW {
            return false;
        }
        self.series
            .iter()
            .all(|s| !trend_detected(&s[s.len() - TREND_WINDOW..]))
    }

    /// Whether exploration may end after iteration `k` (1-based): the minimum
    /// has been run and no tracked coordinate drifts.
    pub(crate) fn may_stop_at(&self, k: usize, k_min: usize) -> bool {
        k >= k_min && self.converged()
    }

    /// Indices of the coordinates that still drift (diagnostics / tests).
    pub(crate) fn drifting(&self) -> Vec<usize> {
        if self.len() < TREND_WINDOW {
            return Vec::new();
        }
        (0..self.series.len())
            .filter(|&c| {
                let s = &self.series[c];
                trend_detected(&s[s.len() - TREND_WINDOW..])
            })
            .collect()
    }
}

/// The tracked coordinates of one iterate: θ, log Ω_jj, correlations, log σ
/// (and the same for Ω_IOV when present).
pub(crate) fn exploration_coords(
    theta: &[f64],
    omega: &DMatrix<f64>,
    omega_iov: Option<&DMatrix<f64>>,
    sigma: &[f64],
) -> Vec<f64> {
    let mut out: Vec<f64> = theta.to_vec();
    for m in std::iter::once(omega).chain(omega_iov) {
        let n = m.nrows();
        for j in 0..n {
            out.push(m[(j, j)].max(1e-300).ln());
        }
        for j in 0..n {
            for k in 0..j {
                let d = (m[(j, j)] * m[(k, k)]).max(1e-300).sqrt();
                out.push(m[(j, k)] / d);
            }
        }
    }
    out.extend(sigma.iter().map(|s| s.abs().max(1e-300).ln()));
    out
}

#[cfg(test)]
#[path = "saem_exploration_tests.rs"]
mod tests;
