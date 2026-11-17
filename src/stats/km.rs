//! Kaplan–Meier and Nelson–Aalen estimates, the mean cumulative function of
//! repeated events, the Kaplan–Meier mean covariate, and the pointwise band a visual
//! predictive check draws from replicate curves (#1886).
//!
//! Pure `f64` math with no feature gate: the TTE VPC drivers (`api::vpc_tte`,
//! `api::vpc_rtte` and `api::vpc_kmmc`, behind `survival`) feed it, and so can any
//! caller holding `(time, event)` pairs or event times with follow-up ends.
//!
//! **Tie convention.** The one R's `survival::survfit` uses, so a curve here
//! equals survfit's row for row. One row per distinct time, event or censoring.
//! At a time `t` the risk set is every subject with time `≥ t`. Events at `t`
//! are counted before censorings at `t`, so the subjects censored at `t` are
//! still in the risk set of the events at `t`. Then
//! `S(t) = Π (1 − d_j / n_j)` and `H(t) = Σ d_j / n_j` over the rows up to `t`.
//!
//! **One counting pass.** [`kaplan_meier`], [`kaplan_meier_mean_covariate`] and
//! [`mean_cumulative`] count their rows in the same private pass. They differ in one
//! thing: whether a subject leaves the risk set at its event (Kaplan–Meier and its
//! mean covariate) or stays in it until its follow-up ends (the mean cumulative
//! function, where a subject can have any number of events).
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

/// Why [`kaplan_meier`], [`kaplan_meier_mean_covariate`], [`vpc_band`],
/// [`vpc_band_mcf`] or [`vpc_band_kmmc`] refused its input.
#[derive(Debug, Clone, PartialEq)]
pub enum KmInputError {
    /// `times` and `events` have different lengths.
    LengthMismatch { times: usize, events: usize },
    /// A time that is not finite, or is negative.
    BadTime { index: usize, value: f64 },
    /// Band probabilities that are not strictly inside (0, 1) and strictly increasing.
    BadProbs { probs: [f64; 3] },
    /// [`kaplan_meier_mean_covariate`]: `times` and `values` have different lengths.
    ValueLengthMismatch { times: usize, values: usize },
    /// [`kaplan_meier_mean_covariate`]: a covariate value that is not finite.
    BadValue { index: usize, value: f64 },
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
            KmInputError::ValueLengthMismatch { times, values } => write!(
                f,
                "kaplan_meier_mean_covariate: {times} times but {values} covariate values; give \
                 one value per time"
            ),
            KmInputError::BadValue { index, value } => write!(
                f,
                "kaplan_meier_mean_covariate: covariate value {value} at index {index} is not a \
                 finite number"
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
    let marks: Vec<(f64, bool)> = times.iter().copied().zip(events.iter().copied()).collect();
    let mut curve = KmCurve::default();
    let (mut s, mut h) = (1.0_f64, 0.0_f64);
    for row in count_pass(&marks, true).0 {
        if row.n_event > 0 {
            s *= 1.0 - row.n_event as f64 / row.n_risk as f64;
            h += row.n_event as f64 / row.n_risk as f64;
        }
        curve.time.push(row.time);
        curve.n_risk.push(row.n_risk);
        curve.n_event.push(row.n_event);
        curve.n_censor.push(row.n_leave);
        curve.surv.push(s);
        curve.cumhaz.push(h);
    }
    Ok(curve)
}

/// A Kaplan–Meier mean covariate (KMMC) curve: at each row of the Kaplan–Meier
/// curve of the same data, the mean of a per-subject value over the subjects at
/// risk there (Hooker & Karlsson, PAGE 21 (2012) Abstr 2564).
///
/// The vectors are parallel, and `time`, `n_risk`, `n_event` and `n_censor` equal
/// [`kaplan_meier`]'s on the same `(times, events)` row for row, since both come
/// from one counting pass. `mean[j]` is the mean of the values of the `n_risk[j]`
/// subjects with time `≥ time[j]`: a subject censored at a tied time is still in
/// the set (the module's tie convention). For a 0/1 value it is the share coded 1.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KmmcCurve {
    pub time: Vec<f64>,
    pub n_risk: Vec<usize>,
    pub n_event: Vec<usize>,
    pub n_censor: Vec<usize>,
    pub mean: Vec<f64>,
}

/// The Kaplan–Meier mean covariate curve of `(times, events)` with one covariate
/// value per subject in `values`: the mean of `values` over each Kaplan–Meier risk
/// set `{i : times[i] ≥ t}`. See [`KmmcCurve`].
///
/// An empty input gives an empty curve, on which [`KmmcCurve::at`] is `None`
/// everywhere.
///
/// # Errors
///
/// As [`kaplan_meier`], plus [`KmInputError::ValueLengthMismatch`] when `values`
/// is not as long as `times`, and [`KmInputError::BadValue`] for a value that is
/// not finite (a missing covariate is `NaN` and would make every mean `NaN`).
pub fn kaplan_meier_mean_covariate(
    times: &[f64],
    events: &[bool],
    values: &[f64],
) -> Result<KmmcCurve, KmInputError> {
    if times.len() != events.len() {
        return Err(KmInputError::LengthMismatch {
            times: times.len(),
            events: events.len(),
        });
    }
    if times.len() != values.len() {
        return Err(KmInputError::ValueLengthMismatch {
            times: times.len(),
            values: values.len(),
        });
    }
    if let Some((index, &value)) = times
        .iter()
        .enumerate()
        .find(|(_, t)| !(t.is_finite() && **t >= 0.0))
    {
        return Err(KmInputError::BadTime { index, value });
    }
    if let Some((index, &value)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(KmInputError::BadValue { index, value });
    }
    let marks: Vec<(f64, bool)> = times.iter().copied().zip(events.iter().copied()).collect();
    let (rows, order) = count_pass(&marks, true);
    // suffix[k] = the sum of the values at sorted positions k.., so the risk set of a
    // row starting at `start` sums to suffix[start].
    let mut suffix = vec![0.0_f64; order.len() + 1];
    for k in (0..order.len()).rev() {
        suffix[k] = suffix[k + 1] + values[order[k]];
    }
    let mut curve = KmmcCurve::default();
    for row in rows {
        debug_assert_eq!(row.n_risk, order.len() - row.start);
        curve.time.push(row.time);
        curve.n_risk.push(row.n_risk);
        curve.n_event.push(row.n_event);
        curve.n_censor.push(row.n_leave);
        curve.mean.push(suffix[row.start] / row.n_risk as f64);
    }
    Ok(curve)
}

impl KmmcCurve {
    /// The mean covariate over the subjects at risk **at** `t`: the value of the first
    /// row with time `≥ t`, so the curve is left-continuous (it takes a row's value
    /// up to and including that row's time, and the next row's just after it).
    /// Before the first row it is the mean over everyone.
    ///
    /// `None` past the last time (nobody is at risk), where that row's `n_risk` is
    /// below `min_at_risk` (too few subjects for a mean worth plotting), for an
    /// empty curve, and for a `t` that is not a number.
    pub fn at(&self, t: f64, min_at_risk: usize) -> Option<f64> {
        if t.is_nan() {
            return None;
        }
        // Number of rows with time < t, i.e. the index of the first row with time ≥ t.
        let j = self.time.partition_point(|&x| x < t);
        if j >= self.time.len() || self.n_risk[j] < min_at_risk {
            return None;
        }
        Some(self.mean[j])
    }
}

/// One row of [`count_pass`]: a distinct time, the risk set at it, and the events
/// and the departures at exactly that time. `start` is the position of the row's
/// first mark in the pass's sorted order, so under `event_leaves` the risk set is
/// the marks at sorted positions `start..`.
struct CountRow {
    time: f64,
    n_risk: usize,
    n_event: usize,
    n_leave: usize,
    start: usize,
}

/// The counting pass [`kaplan_meier`] and [`mean_cumulative`] share, under the
/// module's tie convention. `marks` are `(time, is_event)`. A mark that is not an
/// event is a departure from the risk set: a censoring, or the end of a subject's
/// follow-up. With `event_leaves` an event is a departure too, so every subject
/// carries one mark (Kaplan–Meier). Without it a subject stays at risk after an
/// event and carries one departure and any number of events (the mean cumulative
/// function), and the risk set starts as the number of departures.
///
/// One row per distinct time, in increasing order. Events at a time are counted
/// before the departures at it, so `n_risk` still holds the subjects leaving there.
/// The second value is the sorted order itself (indices into `marks`), which
/// [`kaplan_meier_mean_covariate`] reads the risk sets from.
fn count_pass(marks: &[(f64, bool)], event_leaves: bool) -> (Vec<CountRow>, Vec<usize>) {
    let mut order: Vec<usize> = (0..marks.len()).collect();
    order.sort_by(|&a, &b| marks[a].0.total_cmp(&marks[b].0));

    let mut at_risk = if event_leaves {
        marks.len()
    } else {
        marks.iter().filter(|m| !m.1).count()
    };
    let mut rows = Vec::new();
    let mut k = 0;
    while k < order.len() {
        let t = marks[order[k]].0;
        let start = k;
        let (mut d, mut c) = (0usize, 0usize);
        while k < order.len() && marks[order[k]].0 == t {
            if marks[order[k]].1 {
                d += 1;
            } else {
                c += 1;
            }
            k += 1;
        }
        // Events before departures: the `c` leaving at `t` are still in `at_risk`.
        rows.push(CountRow {
            time: t,
            n_risk: at_risk,
            n_event: d,
            n_leave: c,
            start,
        });
        at_risk -= if event_leaves { d + c } else { c };
    }
    (rows, order)
}

/// The mean cumulative function (MCF) of repeated events with its risk set, one row
/// per distinct time in the data: an event time or the end of a subject's follow-up.
///
/// The vectors are parallel. `time` is strictly increasing. `n_risk[j]` is the
/// number of subjects whose follow-up ends at or after `time[j]`, whatever number of
/// events they have had. `n_event[j]` counts all events at exactly `time[j]`, over
/// all subjects. `mcf[j]` is the Nelson–Aalen estimate `Σ n_event / n_risk` of the
/// expected number of events per subject just after `time[j]`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct McfCurve {
    pub time: Vec<f64>,
    pub n_risk: Vec<usize>,
    pub n_event: Vec<usize>,
    pub mcf: Vec<f64>,
}

/// Why [`mean_cumulative`] refused its input.
#[derive(Debug, Clone, PartialEq)]
pub enum McfInputError {
    /// An event time that is not finite, or is negative.
    BadEventTime { index: usize, value: f64 },
    /// A follow-up end that is not finite, or is negative.
    BadFollowUpEnd { index: usize, value: f64 },
    /// An event later than every follow-up end: nobody is at risk for it.
    EventAfterFollowUp { time: f64 },
}

impl std::fmt::Display for McfInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McfInputError::BadEventTime { index, value } => write!(
                f,
                "mean_cumulative: event time {value} at index {index} is not a finite, \
                 non-negative number"
            ),
            McfInputError::BadFollowUpEnd { index, value } => write!(
                f,
                "mean_cumulative: follow-up end {value} at index {index} is not a finite, \
                 non-negative number"
            ),
            McfInputError::EventAfterFollowUp { time } => write!(
                f,
                "mean_cumulative: the event at time {time} is later than every follow-up end, \
                 so no subject is at risk for it"
            ),
        }
    }
}

impl std::error::Error for McfInputError {}

/// The mean cumulative function of repeated events: `event_times` are all events of
/// all subjects, `follow_up_ends` one time per subject, the end of its follow-up. A
/// subject with no event still gives its follow-up end, since it is at risk until
/// then. A subject stays in the risk set after an event, and an event at a subject's
/// own follow-up end is counted with that subject still at risk (the module's tie
/// convention). Two events at one time, in one subject or in two, both count.
///
/// No follow-up end and no event give an empty curve, on which [`McfCurve::at`] is
/// `None` everywhere.
///
/// # Errors
///
/// [`McfInputError::BadEventTime`] and [`McfInputError::BadFollowUpEnd`] for a time
/// that is not finite or is negative, and [`McfInputError::EventAfterFollowUp`] for
/// an event later than every follow-up end, whose risk set is empty.
pub fn mean_cumulative(
    event_times: &[f64],
    follow_up_ends: &[f64],
) -> Result<McfCurve, McfInputError> {
    let bad = |v: &[f64]| {
        v.iter()
            .copied()
            .enumerate()
            .find(|(_, t)| !(t.is_finite() && *t >= 0.0))
    };
    if let Some((index, value)) = bad(event_times) {
        return Err(McfInputError::BadEventTime { index, value });
    }
    if let Some((index, value)) = bad(follow_up_ends) {
        return Err(McfInputError::BadFollowUpEnd { index, value });
    }
    let marks: Vec<(f64, bool)> = event_times
        .iter()
        .map(|&t| (t, true))
        .chain(follow_up_ends.iter().map(|&t| (t, false)))
        .collect();
    let mut curve = McfCurve::default();
    let mut m = 0.0_f64;
    for row in count_pass(&marks, false).0 {
        if row.n_event > 0 {
            if row.n_risk == 0 {
                return Err(McfInputError::EventAfterFollowUp { time: row.time });
            }
            m += row.n_event as f64 / row.n_risk as f64;
        }
        curve.time.push(row.time);
        curve.n_risk.push(row.n_risk);
        curve.n_event.push(row.n_event);
        curve.mcf.push(m);
    }
    Ok(curve)
}

impl McfCurve {
    /// `MCF(t)`: `0` before the first time, the step value at or after it, and `None`
    /// past the last time, which is the last follow-up end: the estimate is undefined
    /// once nobody is under follow-up. `None` for an empty curve and for a `t` that is
    /// not a number.
    pub fn at(&self, t: f64) -> Option<f64> {
        let &last_t = self.time.last()?;
        if t.is_nan() || t > last_t {
            return None;
        }
        // Number of rows with time ≤ t.
        let j = self.time.partition_point(|&x| x <= t);
        Some(if j == 0 { 0.0 } else { self.mcf[j - 1] })
    }
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
/// `n_used[k]` is how many curves are defined at `grid[k]` ([`KmCurve::at`] or
/// [`McfCurve::at`] is `Some`). `lo / med / hi[k]` are the type-7 quantiles of those values at the
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
/// With fewer than about `1 / p` curves a `p`-tail holds less than one curve, so
/// the outer quantile is an interpolation between the two most extreme values
/// rather than an estimate of the tail. Assumes [`probs_valid`].
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
    band_over(curves.len(), |j, t| curves[j].at(t), grid, probs)
}

/// [`vpc_band`] over mean cumulative functions: the pointwise `probs` band of
/// [`McfCurve::at`] on `grid`, with the same rule for a curve past its last
/// follow-up end and the same floor.
///
/// # Errors
///
/// [`KmInputError::BadProbs`] unless [`probs_valid`].
pub fn vpc_band_mcf(
    curves: &[McfCurve],
    grid: &[f64],
    probs: [f64; 3],
) -> Result<VpcBand, KmInputError> {
    band_over(curves.len(), |j, t| curves[j].at(t), grid, probs)
}

/// [`vpc_band`] over Kaplan–Meier mean covariate curves: the pointwise `probs` band
/// of [`KmmcCurve::at`]`(t, min_at_risk)` on `grid`. A curve past its last time, or
/// with fewer than `min_at_risk` subjects at risk at `t`, is left out at `t`, so the
/// at-risk floor applies to each replicate on its own, as it does to the observed
/// curve. The same [`min_curves_for`] floor then applies to the curves that remain.
///
/// # Errors
///
/// [`KmInputError::BadProbs`] unless [`probs_valid`].
pub fn vpc_band_kmmc(
    curves: &[KmmcCurve],
    grid: &[f64],
    probs: [f64; 3],
    min_at_risk: usize,
) -> Result<VpcBand, KmInputError> {
    band_over(
        curves.len(),
        |j, t| curves[j].at(t, min_at_risk),
        grid,
        probs,
    )
}

/// The band of `n_curves` curves, `at(j, t)` being curve `j` at time `t`, or `None`
/// where that curve is undefined.
fn band_over(
    n_curves: usize,
    at: impl Fn(usize, f64) -> Option<f64>,
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
    let mut vals: Vec<f64> = Vec::with_capacity(n_curves);
    for &t in grid {
        vals.clear();
        vals.extend((0..n_curves).filter_map(|j| at(j, t)));
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
