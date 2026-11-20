//! The Aalen–Johansen cumulative incidence of each cause under competing risks
//! (#1887).
//!
//! Pure `f64` math with no feature gate, like [`crate::stats::km`]: a competing-risks
//! VPC feeds it, and so can any caller holding `(time, cause)` pairs.
//!
//! **Estimator.** With `S` the all-cause Kaplan–Meier curve, the cumulative incidence
//! of cause `k` is
//!
//! `F_k(t) = Σ_{t_j ≤ t} (S(t_j⁻) − S(t_j)) · d_kj / d_j`,
//!
//! where `d_kj` is the number of cause-`k` events at `t_j` and `d_j` the number of
//! events of any cause there. The all-cause drop at `t_j` is split among the causes in
//! proportion to their event counts: the same convention the parametric CIF of a
//! fitted model uses, which splits each step by `dh_k / dh_tot`. Algebraically it
//! equals the textbook `S(t_j⁻) · d_kj / n_j`; in floating point it is the tighter of
//! the two on both identities below, so it is the one used.
//!
//! **One tie convention, one counting pass.** `S` is [`kaplan_meier`] on the
//! any-cause event flags, so the rows, the risk sets and the tie convention (events
//! before censorings at a tied time) are survfit's, exactly as for a Kaplan–Meier
//! curve. The per-cause counts are read off that curve's rows; no second counting
//! pass exists.
//!
//! **Identities.** `Σ_k F_k(t) + S(t) = 1`, and with a single cause `F = 1 − S`. Both
//! hold to rounding, not bit for bit.
//!
//! **Not `1 − KM_k`.** One minus the Kaplan–Meier curve of cause `k` with the other
//! causes treated as censorings estimates the incidence in a world without the other
//! causes, and overstates `F_k` whenever they are present.

use crate::stats::km::{kaplan_meier, KmCurve};

/// The Aalen–Johansen cumulative incidence curves of `n_causes` competing causes,
/// on the rows of their all-cause Kaplan–Meier curve.
///
/// `n_event[k][j]` is the number of cause-`k + 1` events at `all_cause.time[j]`, and
/// `cif[k][j]` is `F_{k+1}` just after that time. Every inner vector is parallel to
/// `all_cause.time`, and `Σ_k n_event[k][j] = all_cause.n_event[j]`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CifCurve {
    pub all_cause: KmCurve,
    pub n_event: Vec<Vec<usize>>,
    pub cif: Vec<Vec<f64>>,
}

/// Why [`aalen_johansen`] refused its input.
#[derive(Debug, Clone, PartialEq)]
pub enum CifInputError {
    /// `times` and `causes` have different lengths.
    LengthMismatch { times: usize, causes: usize },
    /// A time that is not finite, or is negative.
    BadTime { index: usize, value: f64 },
    /// A cause label above `n_causes`.
    BadCause { index: usize, value: usize, n_causes: usize },
}

impl std::fmt::Display for CifInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CifInputError::LengthMismatch { times, causes } => write!(
                f,
                "aalen_johansen: {times} times but {causes} cause labels; give one label per time"
            ),
            CifInputError::BadTime { index, value } => write!(
                f,
                "aalen_johansen: time {value} at index {index} is not a finite, non-negative number"
            ),
            CifInputError::BadCause { index, value, n_causes } => write!(
                f,
                "aalen_johansen: cause {value} at index {index} is not in 0..={n_causes}; use 0 \
                 for a censoring and 1..={n_causes} for the causes"
            ),
        }
    }
}

impl std::error::Error for CifInputError {}

/// The Aalen–Johansen cumulative incidence of each cause of `(times, causes)`, where
/// `causes[i]` is `0` for a right censoring and `k ∈ 1..=n_causes` for an event of
/// cause `k`. See the module doc for the estimator and the tie convention.
///
/// An empty input gives an empty curve with `n_causes` empty cause rows, on which
/// [`CifCurve::at`] is `None` everywhere.
///
/// # Errors
///
/// [`CifInputError::LengthMismatch`] when the two slices differ in length,
/// [`CifInputError::BadTime`] for a time that is not finite or is negative, and
/// [`CifInputError::BadCause`] for a label above `n_causes`.
pub fn aalen_johansen(
    times: &[f64], causes: &[usize], n_causes: usize,
) -> Result<CifCurve, CifInputError> {
    if times.len() != causes.len() {
        return Err(CifInputError::LengthMismatch { times: times.len(), causes: causes.len() });
    }
    if let Some((index, &value)) =
        times.iter().enumerate().find(|(_, t)| !(t.is_finite() && **t >= 0.0))
    {
        return Err(CifInputError::BadTime { index, value });
    }
    if let Some((index, &value)) = causes.iter().enumerate().find(|(_, &c)| c > n_causes) {
        return Err(CifInputError::BadCause { index, value, n_causes });
    }
    let any: Vec<bool> = causes.iter().map(|&c| c > 0).collect();
    // Lengths and times were checked above, so `kaplan_meier` cannot refuse them.
    let all_cause = kaplan_meier(times, &any).expect("inputs validated above");
    let rows = all_cause.time.len();
    let mut n_event = vec![vec![0usize; rows]; n_causes];
    for (&t, &c) in times.iter().zip(causes) {
        if c > 0 {
            // Every input time is a row of the curve built from these same times.
            let j = all_cause.time.partition_point(|&x| x < t);
            n_event[c - 1][j] += 1;
        }
    }
    let mut cif = vec![vec![0.0f64; rows]; n_causes];
    let mut s_prev = 1.0_f64;
    for j in 0..rows {
        let (d, s) = (all_cause.n_event[j], all_cause.surv[j]);
        for k in 0..n_causes {
            let before = if j == 0 { 0.0 } else { cif[k][j - 1] };
            let dk = n_event[k][j];
            cif[k][j] = if dk > 0 { before + (s_prev - s) * dk as f64 / d as f64 } else { before };
        }
        s_prev = s;
    }
    Ok(CifCurve { all_cause, n_event, cif })
}

impl CifCurve {
    /// The number of causes.
    pub fn n_causes(&self) -> usize {
        self.cif.len()
    }

    /// `F_{k+1}(t)` for the zero-based cause index `k`: `0` before the first time, the
    /// step value at or after it, and past the last time the final value when the
    /// all-cause curve has reached zero and `None` when it has not.
    ///
    /// Definedness is the all-cause curve's: this is `None` exactly where
    /// [`KmCurve::at`] of `all_cause` is, so a band over CIF curves keeps the same
    /// replicates at a grid time as a band over their all-cause curves. `None` for a
    /// `k` that is not a cause, for an empty curve and for a `t` that is not a number.
    pub fn at(&self, k: usize, t: f64) -> Option<f64> {
        let f = self.cif.get(k)?;
        self.all_cause.at(t)?;
        // Number of rows with time ≤ t (every row, past the last time).
        let j = self.all_cause.time.partition_point(|&x| x <= t);
        Some(if j == 0 { 0.0 } else { f[j - 1] })
    }
}

#[cfg(test)]
#[path = "cif_tests.rs"]
mod tests;
