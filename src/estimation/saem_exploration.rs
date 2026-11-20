//! Automatic end of SAEM's exploration phase.
//!
//! The exploration phase runs the stochastic approximation at full step
//! (γ = 1) so the iterates can travel to the neighbourhood of the maximiser;
//! the convergence phase then averages with γ_k = 1/(k − K1)^a and freezes
//! whatever the iterates had reached. Ending the first phase too early freezes
//! a *still-moving* parameter: weakly identified Ω entries decay slowly and the
//! averaging then pins them above their optimum (pk2_big: ω²_Q 0.019 against
//! ≈ 0.003).
//!
//! # The stop rule: NONMEM's termination test, with its interval chosen per coordinate
//!
//! NONMEM's SAEM termination test (`$EST … CTYPE=3 CITER=10 CINTERVAL=… CALPHA=…`,
//! Users Guide §I.46) takes, for each monitored quantity, the latest `CITER = 10`
//! iterates spaced `CINTERVAL` apart, regresses them on the iteration number, and
//! ends burn-in when no slope is significant. The guide's point is the spacing:
//! consecutive SAEM iterates are highly correlated, so the points must be spread
//! over a long stretch (`CINTERVAL` 10–100; `CINTERVAL=0` lets NONMEM pick it).
//! An earlier version of this module used a 100-iteration window of five 20-iteration
//! batch means; on traced runs the integrated autocorrelation time of the iterates is
//! ~35 iterations at the median and 90–170 at the 90th percentile, so that window
//! flagged 12–23 % of stationary coordinates (nominal 5 %) and stopped mostly by chance.
//!
//! Here each coordinate gets its own interval: its integrated autocorrelation time
//! τ̂ (Geyer, on the linearly detrended series after the first [`TAU_SKIP`]
//! iterations), clamped to [[`SPACING_MIN`], [`SPACING_MAX`]] — inside NONMEM's
//! recommended 10–100. With spacing ≈ τ the ten points are close to independent, and
//! the OLS slope is tested with a Student t (`CITER − 2 = 8` df) at two-sided 5 %
//! (`CALPHA = 0.05`, NONMEM's `AUTO=1` value). A coordinate whose series is shorter
//! than `CITER × spacing` is *not yet testable* and blocks the stop like a drifting
//! one. The test runs every [`CHECK_EVERY`] iterations; exploration ends at the first
//! check ≥ the minimum where every coordinate is testable and none drifts, never
//! later than the maximum (default 4000, NONMEM's `AUTO=1` `NBURN`).
//!
//! Calibration on simulated AR(1) noise (2000 iterations): false-trend rate 6.8 % /
//! 6.5 % / 11.3 % at lag-1 correlation 0 / 0.9 / 0.97 (nominal 5 %); a drift of 15
//! marginal sds over 600 iterations at 0.9 is caught 83 % of the time. On traced SAEM
//! runs of a pk2_big stand-in the 100-iteration batch-means rule it replaces stopped
//! at 273–442 with ω²_Q still 40–90 % above its plateau; this rule stops at its plateau.
//!
//! Tracked coordinates (NONMEM's `CTYPE=3` set, scale-free in the test): the
//! complete-data −log L (Σᵢ −log p(yᵢ, ηᵢ; θ, Ω, σ), Monolix's "convergence
//! indicator", NONMEM's objective), θ, log Ω_jj, the correlations Ω_jk/√(Ω_jjΩ_kk)
//! (and the same for Ω_IOV), log σ. A series that is exactly constant (fixed
//! parameters, structural zeros) cannot drift.

use nalgebra::DMatrix;

/// Points per regression (NONMEM `CITER`).
pub(crate) const CITER: usize = 10;
/// Bounds on a coordinate's point spacing (NONMEM `CINTERVAL`), in iterations.
pub(crate) const SPACING_MIN: usize = 10;
/// See [`SPACING_MIN`]. The upper bound keeps the longest window (`CITER ×
/// SPACING_MAX` = 500 iterations) inside a practical maximum.
pub(crate) const SPACING_MAX: usize = 50;
/// Shortest series on which any coordinate can be tested.
pub(crate) const MIN_TEST_WINDOW: usize = CITER * SPACING_MIN;
/// Leading iterations left out of the autocorrelation estimate (the Ω burn-in).
pub(crate) const TAU_SKIP: usize = 20;
/// The stop is evaluated every this many iterations.
pub(crate) const CHECK_EVERY: usize = 10;
/// Two-sided 5 % Student-t quantile, `CITER − 2 = 8` degrees of freedom (NONMEM
/// `CALPHA=0.05`, its `AUTO=1` value).
pub(crate) const TREND_T: f64 = 2.306_004_135_204_166;

/// Point spacing for series `y`: its integrated autocorrelation time on the
/// linearly detrended values after [`TAU_SKIP`], clamped to
/// [[`SPACING_MIN`], [`SPACING_MAX`]].
pub(crate) fn spacing(y: &[f64]) -> usize {
    let y = &y[TAU_SKIP.min(y.len())..];
    let n = y.len();
    if n < 4 {
        return SPACING_MAX;
    }
    let xbar = (n - 1) as f64 / 2.0;
    let ybar = y.iter().sum::<f64>() / n as f64;
    let sxx: f64 = (0..n).map(|i| (i as f64 - xbar).powi(2)).sum();
    let slope =
        y.iter().enumerate().map(|(i, v)| (i as f64 - xbar) * (v - ybar)).sum::<f64>() / sxx;
    let resid: Vec<f64> =
        y.iter().enumerate().map(|(i, v)| v - ybar - slope * (i as f64 - xbar)).collect();
    let tau = crate::stats::convergence::integrated_autocorr_time(&resid, SPACING_MAX as f64);
    (tau.ceil() as usize).clamp(SPACING_MIN, SPACING_MAX)
}

/// Termination test of one coordinate's series `y` (all exploration iterates so
/// far): `Some(true)` when the slope of its latest [`CITER`] points, spaced by
/// [`spacing`], is significant at 5 %; `Some(false)` when it is not, or the
/// points are numerically constant; `None` when the series is still shorter than
/// `CITER × spacing` (not yet testable). A non-finite point counts as drifting.
pub(crate) fn trend_test(y: &[f64]) -> Option<bool> {
    let c = spacing(y);
    let n = y.len();
    if n < CITER * c {
        return None;
    }
    let pts: Vec<f64> = (0..CITER).rev().map(|j| y[n - 1 - j * c]).collect();
    if pts.iter().any(|v| !v.is_finite()) {
        return Some(true);
    }
    let b = CITER as f64;
    let xbar = (b - 1.0) / 2.0;
    let mbar = pts.iter().sum::<f64>() / b;
    let sxx: f64 = (0..CITER).map(|i| (i as f64 - xbar).powi(2)).sum();
    let slope =
        pts.iter().enumerate().map(|(i, v)| (i as f64 - xbar) * (v - mbar)).sum::<f64>() / sxx;
    let ss: f64 =
        pts.iter().enumerate().map(|(i, v)| (v - mbar - slope * (i as f64 - xbar)).powi(2)).sum();
    // Numerically constant: the fitted change across the points is at rounding level of the
    // series scale. (Not "zero residual": a noiseless exact ramp has zero residual and is a trend.)
    let scale = pts.iter().fold(0.0_f64, |a, v| a.max(v.abs())).max(1e-300);
    if slope.abs() * (b - 1.0) <= 1e-12 * scale {
        return Some(false);
    }
    let se = (ss / (b - 2.0) / sxx).sqrt();
    Some(slope.abs() > TREND_T * se)
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
            options.saem_exploration_max.max(options.saem_exploration_min),
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

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.series.first().map_or(0, Vec::len)
    }

    /// Every tracked coordinate is testable and none drifts.
    pub(crate) fn converged(&self) -> bool {
        !self.series.is_empty() && self.series.iter().all(|s| trend_test(s) == Some(false))
    }

    /// Whether exploration may end after iteration `k` (1-based): the minimum has
    /// been run, `k` is a check iteration ([`CHECK_EVERY`]), and the monitor has
    /// converged.
    pub(crate) fn may_stop_at(&self, k: usize, k_min: usize) -> bool {
        k >= k_min && k.is_multiple_of(CHECK_EVERY) && self.converged()
    }

    /// Indices of the coordinates not yet shown stationary: drifting, or still
    /// too short to test at their spacing (diagnostics / warning).
    pub(crate) fn drifting(&self) -> Vec<usize> {
        (0..self.series.len()).filter(|&c| trend_test(&self.series[c]) != Some(false)).collect()
    }
}

/// The tracked coordinates of one iterate: the complete-data −log L, θ,
/// log Ω_jj, correlations (and the same for Ω_IOV when present), log σ.
/// [`exploration_coord_names`] names them in the same order.
pub(crate) fn exploration_coords(
    complete_nll: f64, theta: &[f64], omega: &DMatrix<f64>, omega_iov: Option<&DMatrix<f64>>,
    sigma: &[f64],
) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(1 + theta.len() + sigma.len());
    out.push(complete_nll);
    out.extend_from_slice(theta);
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

/// Names of the coordinates [`exploration_coords`] returns, in its order, for
/// the maximum-reached warning.
pub(crate) fn exploration_coord_names(
    theta: &[String], eta: &[String], kappa: Option<&[String]>, sigma: &[String],
) -> Vec<String> {
    let mut out = vec!["complete-data -logL".to_string()];
    out.extend(theta.iter().cloned());
    for (label, names) in std::iter::once(("OMEGA", eta)).chain(kappa.map(|k| ("OMEGA_IOV", k))) {
        for n in names {
            out.push(format!("{label}({n})"));
        }
        for j in 0..names.len() {
            for k in 0..j {
                out.push(format!("corr {label}({},{})", names[k], names[j]));
            }
        }
    }
    out.extend(sigma.iter().map(|n| format!("SIGMA({n})")));
    out
}

/// Comma-separated names of the drifting coordinates, at most `max_listed`,
/// then "and N more". An index without a name is printed as `#i`.
pub(crate) fn drifting_summary(drifting: &[usize], names: &[String], max_listed: usize) -> String {
    if drifting.is_empty() {
        return "none".to_string();
    }
    let mut parts: Vec<String> = drifting
        .iter()
        .take(max_listed)
        .map(|&i| names.get(i).cloned().unwrap_or_else(|| format!("#{i}")))
        .collect();
    if drifting.len() > max_listed {
        parts.push(format!("and {} more", drifting.len() - max_listed));
    }
    parts.join(", ")
}

#[cfg(test)]
#[path = "saem_exploration_tests.rs"]
mod tests;
