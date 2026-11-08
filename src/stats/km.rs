//! Kaplan–Meier and Nelson–Aalen estimates, and the pointwise band a
//! Kaplan–Meier visual predictive check draws from replicate curves (#1886).
//!
//! Pure `f64` math with no feature gate: the TTE VPC driver
//! (`api::vpc_tte`, behind `survival`) feeds it, and so can any caller holding
//! `(time, event)` pairs.
//!
//! **Tie convention.** The one R's `survival::survfit` uses, so a curve here
//! equals survfit's row for row. One row per distinct time, event or censoring.
//! At a time `t` the risk set is every subject with time `≥ t`. Events at `t`
//! are counted before censorings at `t`, so the subjects censored at `t` are
//! still in the risk set of the events at `t`. Then
//! `S(t) = Π (1 − d_j / n_j)` and `H(t) = Σ d_j / n_j` over the rows up to `t`.
//!
//! **Quantile type.** The band uses
//! [`quantile_sorted`], R's default
//! type 7, the type `vpc::vpc_tte` uses. No second quantile rule exists here.

use crate::stats::convergence::quantile_sorted;

/// A Kaplan–Meier curve with its Nelson–Aalen cumulative hazard, one row per
/// distinct time in the data (survfit's layout).
///
/// The vectors are parallel. `time` is strictly increasing. `n_risk[j]` is the
/// number of subjects with time `≥ time[j]`. `n_event[j]` and `n_censor[j]`
/// count the events and censorings at exactly `time[j]`. `surv[j]` and
/// `cumhaz[j]` are the estimates just after `time[j]`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KmCurve {
    pub time: Vec<f64>,
    pub n_risk: Vec<usize>,
    pub n_event: Vec<usize>,
    pub n_censor: Vec<usize>,
    pub surv: Vec<f64>,
    pub cumhaz: Vec<f64>,
}

/// Why [`kaplan_meier`] or [`vpc_band`] refused its input.
#[derive(Debug, Clone, PartialEq)]
pub enum KmInputError {
    /// `times` and `events` have different lengths.
    LengthMismatch { times: usize, events: usize },
    /// A time that is not finite, or is negative.
    BadTime { index: usize, value: f64 },
    /// Band probabilities that are not strictly inside (0, 1) and strictly increasing.
    BadProbs { probs: [f64; 3] },
}

impl std::fmt::Display for KmInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KmInputError::LengthMismatch { times, events } => write!(
                f,
                "kaplan_meier: {times} times but {events} event flags; give one flag per time"
            ),
            KmInputError::BadTime { index, value } => write!(
                f,
                "kaplan_meier: time {value} at index {index} is not a finite, non-negative number"
            ),
            KmInputError::BadProbs { probs } => write!(
                f,
                "band probabilities {probs:?} must be strictly inside (0, 1) and strictly increasing"
            ),
        }
    }
}

impl std::error::Error for KmInputError {}

/// The Kaplan–Meier curve and Nelson–Aalen cumulative hazard of `(times, events)`,
/// where `events[i]` is `true` for an observed event and `false` for a right
/// censoring. See the module doc for the tie convention.
///
/// An empty input gives an empty curve, on which [`KmCurve::at`] is `None` everywhere.
///
/// # Errors
///
/// [`KmInputError::LengthMismatch`] when the two slices differ in length, and
/// [`KmInputError::BadTime`] for a time that is not finite or is negative.
pub fn kaplan_meier(times: &[f64], events: &[bool]) -> Result<KmCurve, KmInputError> {
    if times.len() != events.len() {
        return Err(KmInputError::LengthMismatch {
            times: times.len(),
            events: events.len(),
        });
    }
    if let Some((index, &value)) = times
        .iter()
        .enumerate()
        .find(|(_, t)| !(t.is_finite() && **t >= 0.0))
    {
        return Err(KmInputError::BadTime { index, value });
    }
    let mut order: Vec<usize> = (0..times.len()).collect();
    order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));

    let mut curve = KmCurve::default();
    let mut at_risk = times.len();
    let (mut s, mut h) = (1.0_f64, 0.0_f64);
    let mut k = 0;
    while k < order.len() {
        let t = times[order[k]];
        let (mut d, mut c) = (0usize, 0usize);
        while k < order.len() && times[order[k]] == t {
            if events[order[k]] {
                d += 1;
            } else {
                c += 1;
            }
            k += 1;
        }
        // Events before censorings: the `c` censored at `t` are still in `at_risk`.
        let n = at_risk;
        if d > 0 {
            s *= 1.0 - d as f64 / n as f64;
            h += d as f64 / n as f64;
        }
        curve.time.push(t);
        curve.n_risk.push(n);
        curve.n_event.push(d);
        curve.n_censor.push(c);
        curve.surv.push(s);
        curve.cumhaz.push(h);
        at_risk -= d + c;
    }
    Ok(curve)
}

impl KmCurve {
    /// `S(t)`: `1` before the first time, the step value at or after it, and past
    /// the last time `Some(0.0)` when the curve has reached zero and `None` when it
    /// has not, since the estimate is undefined once nobody is under follow-up.
    /// `None` for an empty curve and for a `t` that is not a number.
    pub fn at(&self, t: f64) -> Option<f64> {
        let (&last_t, &last_s) = (self.time.last()?, self.surv.last()?);
        if t.is_nan() {
            return None;
        }
        if t > last_t {
            return (last_s == 0.0).then_some(0.0);
        }
        // Number of rows with time ≤ t.
        let j = self.time.partition_point(|&x| x <= t);
        Some(if j == 0 { 1.0 } else { self.surv[j - 1] })
    }
}

/// A pointwise band over replicate curves, one entry per grid time.
///
/// `n_used[k]` is how many curves are defined at `grid[k]` ([`KmCurve::at`] is
/// `Some`). `lo / med / hi[k]` are the type-7 quantiles of those values at the
/// three probabilities, or `None` when `n_used[k]` is below
/// [`min_curves_for`] of the probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct VpcBand {
    pub lo: Vec<Option<f64>>,
    pub med: Vec<Option<f64>>,
    pub hi: Vec<Option<f64>>,
    pub n_used: Vec<usize>,
}

/// Whether `probs` are strictly inside (0, 1) and strictly increasing.
pub fn probs_valid(probs: [f64; 3]) -> bool {
    probs.iter().all(|p| *p > 0.0 && *p < 1.0) && probs[0] < probs[1] && probs[1] < probs[2]
}

/// The fewest curves a band point needs before its outer quantiles are read:
/// `⌈1 / min(p, 1 − p)⌉ − 1` over the three probabilities, 19 for 5 % / 95 %.
/// With fewer, the outer quantile is the sample minimum or maximum under any
/// rule, not an estimate of the tail. Assumes [`probs_valid`].
pub fn min_curves_for(probs: [f64; 3]) -> usize {
    // Increasing probabilities: the smallest tail is the first one's or the last one's.
    let tail = if probs[0] < 1.0 - probs[2] {
        probs[0]
    } else {
        1.0 - probs[2]
    };
    // `1 / 0.05` is 20 up to rounding; the epsilon keeps the ceiling from stepping to 21.
    ((1.0 / tail) - 1e-9).ceil() as usize - 1
}

/// The pointwise `probs` band of `curves` on `grid`. At each grid time only the
/// curves still defined there contribute (a replicate past its last follow-up
/// is left out, not counted as 0 or 1), and a point with fewer than
/// [`min_curves_for`] of them is `None`. No `NaN` reaches a quantile.
///
/// # Errors
///
/// [`KmInputError::BadProbs`] unless [`probs_valid`].
pub fn vpc_band(
    curves: &[KmCurve],
    grid: &[f64],
    probs: [f64; 3],
) -> Result<VpcBand, KmInputError> {
    if !probs_valid(probs) {
        return Err(KmInputError::BadProbs { probs });
    }
    let need = min_curves_for(probs);
    let mut band = VpcBand {
        lo: Vec::with_capacity(grid.len()),
        med: Vec::with_capacity(grid.len()),
        hi: Vec::with_capacity(grid.len()),
        n_used: Vec::with_capacity(grid.len()),
    };
    let mut vals: Vec<f64> = Vec::with_capacity(curves.len());
    for &t in grid {
        vals.clear();
        vals.extend(curves.iter().filter_map(|c| c.at(t)));
        debug_assert!(vals.iter().all(|v| v.is_finite()));
        vals.sort_by(f64::total_cmp);
        band.n_used.push(vals.len());
        let q = |p: f64| (vals.len() >= need).then(|| quantile_sorted(&vals, p));
        band.lo.push(q(probs[0]));
        band.med.push(q(probs[1]));
        band.hi.push(q(probs[2]));
    }
    Ok(band)
}

#[cfg(test)]
#[path = "km_tests.rs"]
mod tests;
