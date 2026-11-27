//! Population summary of an adaptive run, with parameter-uncertainty bands and a pooled
//! prediction interval (#946 PR2, the S5 slice of epic #391).
//!
//! [`summarize_adaptive`] is post-processing only: it borrows a finished
//! [`AdaptiveSimulationResult`] and reads its `metrics` and `decisions` rows. It cannot
//! change a trajectory, a ledger entry or a decision.
//!
//! **One kernel, two producers.** Each metrics row yields a value per scalar metric, and
//! each decision yields its analytes' latent signals keyed by `decision_idx`. Both are fed
//! as `(draw, metric, value)` triples to one kernel, which forms the per-draw statistics,
//! the band across draws and the pooled prediction interval. A run that carries no value
//! for a metric (no `target_window`, too few decisions for an AUC window, a decision after
//! a `Stop`) contributes nothing and is counted in `n_missing`: missing values are never
//! imputed.
//!
//! **Pooling.** A draw's statistics are over every run tagged with that draw (`S × N`
//! today, `S × N × R` once assay replicates land), each run weighted equally.
//!
//! **Quantiles.** Type 7 (R's default), the rule of the VPC bands
//! ([`crate::stats::km::vpc_band`]). The `p`-quantile is read only from
//! `⌈1 / min(p, 1 − p)⌉ − 1` values (19 for 5 % / 95 %, 1 for the median); below that
//! it is `None` rather than an interpolation between the two most extreme values.

use super::adaptive_decomposition::jackknife;
use super::AdaptiveSimulationResult;
use crate::diagnostics::EngineError;
use crate::sim::adaptive::AdaptiveSubjectMetrics;
use crate::stats::convergence::quantile_sorted;
use crate::stats::km::{gated_quantiles, min_n_for_prob, probs_valid};
use std::collections::BTreeMap;

/// A quantity summarised by [`summarize_adaptive`]: one field of
/// [`AdaptiveSubjectMetrics`], or the latent signal of one analyte at one decision.
///
/// `time_to_discontinuation` is not one: it is censored (`None` for a run that never
/// stopped), so its mean would describe only the runs that stopped. `Discontinued`
/// covers it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum AdaptiveSummaryMetric {
    /// [`AdaptiveSubjectMetrics::cumulative_dose`].
    CumulativeDose,
    /// [`AdaptiveSubjectMetrics::n_doses`].
    NDoses,
    /// [`AdaptiveSubjectMetrics::n_increases`].
    NIncreases,
    /// [`AdaptiveSubjectMetrics::n_decreases`].
    NDecreases,
    /// [`AdaptiveSubjectMetrics::n_holds`].
    NHolds,
    /// [`AdaptiveSubjectMetrics::n_not_due`].
    NNotDue,
    /// [`AdaptiveSubjectMetrics::discontinued`] as 1 or 0, so its `Mean` is the fraction
    /// discontinued.
    Discontinued,
    /// [`AdaptiveSubjectMetrics::signal_min`].
    SignalMin,
    /// [`AdaptiveSubjectMetrics::signal_max`].
    SignalMax,
    /// [`AdaptiveSubjectMetrics::signal_mean`].
    SignalMean,
    /// [`AdaptiveSubjectMetrics::pct_readings_in_window`].
    PctReadingsInWindow,
    /// [`AdaptiveSubjectMetrics::auc_target_attainment`].
    AucTargetAttainment,
    /// The latent signal ([`crate::sim::adaptive::ObservedSignal::latent`]) of `analyte` at
    /// the decision with this 0-based `decision_idx`.
    Signal {
        /// The monitor's name ([`crate::sim::adaptive::ObservedSignal::name`]).
        analyte: String,
        /// The decision's index in the schedule.
        decision_idx: usize,
    },
}

/// The scalar metrics, in output order: every one gets rows, present or not.
const SCALAR_METRICS: [AdaptiveSummaryMetric; 12] = [
    AdaptiveSummaryMetric::CumulativeDose,
    AdaptiveSummaryMetric::NDoses,
    AdaptiveSummaryMetric::NIncreases,
    AdaptiveSummaryMetric::NDecreases,
    AdaptiveSummaryMetric::NHolds,
    AdaptiveSummaryMetric::NNotDue,
    AdaptiveSummaryMetric::Discontinued,
    AdaptiveSummaryMetric::SignalMin,
    AdaptiveSummaryMetric::SignalMax,
    AdaptiveSummaryMetric::SignalMean,
    AdaptiveSummaryMetric::PctReadingsInWindow,
    AdaptiveSummaryMetric::AucTargetAttainment,
];

impl AdaptiveSummaryMetric {
    /// This metric's value in one run's metrics row; `None` where the run carries none.
    /// The only mapping from a metric to its field. `Signal` is not a field of the row.
    pub(crate) fn value(&self, m: &AdaptiveSubjectMetrics) -> Option<f64> {
        use AdaptiveSummaryMetric::*;
        match self {
            CumulativeDose => Some(m.cumulative_dose),
            NDoses => Some(m.n_doses as f64),
            NIncreases => Some(m.n_increases as f64),
            NDecreases => Some(m.n_decreases as f64),
            NHolds => Some(m.n_holds as f64),
            NNotDue => Some(m.n_not_due as f64),
            Discontinued => Some(if m.discontinued { 1.0 } else { 0.0 }),
            SignalMin => m.signal_min,
            SignalMax => m.signal_max,
            SignalMean => m.signal_mean,
            PctReadingsInWindow => m.pct_readings_in_window,
            AucTargetAttainment => m.auc_target_attainment,
            Signal { .. } => None,
        }
    }
}

/// A per-draw statistic of one metric over the draw's runs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AdaptiveSummaryStatistic {
    /// The mean over the runs that carry the metric.
    Mean,
    /// The type-7 median over the runs that carry the metric.
    Median,
    /// The fraction of the runs carrying the metric whose value is `≥` the threshold,
    /// e.g. `PctReadingsInWindow ≥ 0.8`. Requested through
    /// [`AdaptiveSummaryOptions::thresholds`].
    FractionAtLeast(f64),
}

/// What [`summarize_adaptive`] computes besides the mean and median of every metric.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveSummaryOptions {
    /// The three probabilities of every band and prediction interval: strictly inside
    /// (0, 1) and strictly increasing. Default `[0.05, 0.5, 0.95]`.
    pub probs: [f64; 3],
    /// `(metric, a)`: add a `FractionAtLeast(a)` row for `metric`, e.g.
    /// `(PctReadingsInWindow, 0.8)`. Empty by default: a threshold has no meaning without
    /// the target the run was scored against. A threshold on a metric no run carries,
    /// including a `Signal` at an analyte or `decision_idx` no decision recorded, gives rows
    /// with `value = None`, not an error. A repeated `(metric, a)` gives one set of rows.
    pub thresholds: Vec<(AdaptiveSummaryMetric, f64)>,
}

impl Default for AdaptiveSummaryOptions {
    fn default() -> Self {
        Self { probs: [0.05, 0.5, 0.95], thresholds: Vec::new() }
    }
}

/// Three type-7 quantiles at `probs`. A quantile is `None` while fewer values than it
/// needs are present (see the module docs), and `NaN` if any value was not finite.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveQuantiles {
    /// The probabilities, as in [`AdaptiveSummaryOptions::probs`].
    pub probs: [f64; 3],
    /// The quantile at each probability.
    pub values: [Option<f64>; 3],
}

/// One statistic of one metric over one draw's runs.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveSummaryRow {
    /// The parameter draw (1 for a run without uncertainty).
    pub draw: usize,
    /// The quantity summarised.
    pub metric: AdaptiveSummaryMetric,
    /// The statistic.
    pub statistic: AdaptiveSummaryStatistic,
    /// The statistic over the `n_runs − n_missing` runs carrying the metric; `None` when
    /// no run carries it. `NaN` if any carried value was not finite.
    pub value: Option<f64>,
    /// Runs in the draw.
    pub n_runs: usize,
    /// Runs in the draw that carry no value for the metric (skipped, never imputed).
    pub n_missing: usize,
    /// The Monte Carlo standard error of `value` (#3215): the delete-one jackknife over the
    /// draw's sims that carry the metric, a sim being the independent replicate (each
    /// subject's η and mixture class are drawn once per sim and shared by its `n_reps`
    /// replicates, so runs within one sim are correlated). With `S` such sims, read it with
    /// a t on `S − 1` degrees of freedom. `None` for `Median` (the delete-one jackknife is
    /// inconsistent for a quantile) and with fewer than two such sims (so also when `value`
    /// is `None`); otherwise `NaN` when `value` is.
    pub mc_se: Option<f64>,
}

/// Whether a band could be formed.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AdaptiveBandStatus {
    /// The run has one parameter draw (no uncertainty, or `n_draws = 1`): there is no
    /// spread across draws to describe. Never reported as a zero-width band. It describes the
    /// run, not the metric: in a run with several draws a metric present in only one of them
    /// gets `Quantiles`, whose levels follow the per-level count rule on that one value.
    NotDrawn,
    /// The quantiles across draws of the per-draw statistic.
    Quantiles(AdaptiveQuantiles),
}

/// The parameter-uncertainty band of one per-draw statistic: its quantiles **across
/// draws**. It describes how much the statistic (say, the fraction attaining target)
/// moves with the parameters, not how much patients vary: that is
/// [`AdaptivePredictionInterval`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveSummaryBand {
    /// The quantity summarised.
    pub metric: AdaptiveSummaryMetric,
    /// The per-draw statistic the band is over.
    pub statistic: AdaptiveSummaryStatistic,
    /// Draws whose statistic has a value: the band's sample size.
    pub n_draws_present: usize,
    /// Draws in which no run carries the metric (skipped, never imputed). With
    /// `n_draws_present` it sums to [`AdaptivePopulationSummary::n_draws`].
    pub n_draws_missing: usize,
    /// The band, or why there is none.
    pub status: AdaptiveBandStatus,
}

/// The prediction interval of one metric: the quantiles of its individual values
/// **pooled over every run** of every draw. It describes where one simulated patient
/// falls, parameter uncertainty included; it is not a band on a population statistic.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptivePredictionInterval {
    /// The quantity summarised.
    pub metric: AdaptiveSummaryMetric,
    /// The pooled quantiles.
    pub quantiles: AdaptiveQuantiles,
    /// Runs pooled.
    pub n_runs: usize,
    /// Runs carrying no value for the metric (skipped, never imputed).
    pub n_missing: usize,
}

/// Output of [`summarize_adaptive`]. Bands and prediction intervals are kept apart, in
/// their own types.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptivePopulationSummary {
    /// Distinct draws in the result.
    pub n_draws: usize,
    /// One row per `(metric, statistic, draw)`, so any other quantile across draws can be
    /// computed from them.
    pub rows: Vec<AdaptiveSummaryRow>,
    /// One band per `(metric, statistic)`.
    pub bands: Vec<AdaptiveSummaryBand>,
    /// One prediction interval per metric.
    pub prediction_intervals: Vec<AdaptivePredictionInterval>,
}

/// Summarise an adaptive run's population outcomes (#946): per-draw statistics of every
/// metric, their bands across parameter draws, and pooled prediction intervals. See the
/// module docs for pooling, missing values and the quantile rule.
///
/// Every metric of [`AdaptiveSummaryMetric`] gets a `Mean` and a `Median` row per draw,
/// and a `FractionAtLeast` row per requested threshold; every analyte the decisions
/// recorded gets the same at each `decision_idx`. A supplied-η subject
/// ([`super::AdaptiveInputs::with_etas`]) is summarised like any other run. The result is
/// only borrowed (`&`), so summarising cannot change a trajectory, a ledger row or a
/// decision.
///
/// # Errors
///
/// `E_ADAPTIVE_OPTIONS` when `opts.probs` are not strictly inside (0, 1) and strictly
/// increasing, or a threshold is not finite.
pub fn summarize_adaptive(
    result: &AdaptiveSimulationResult, opts: &AdaptiveSummaryOptions,
) -> Result<AdaptivePopulationSummary, EngineError> {
    check_summary_options(opts, "summarize_adaptive")?;
    Ok(summarize_values(&runs_per_draw(result), run_values(result), opts))
}

/// Refuse summary options a summary cannot use, naming the entry `who`.
pub(super) fn check_summary_options(
    opts: &AdaptiveSummaryOptions, who: &str,
) -> Result<(), EngineError> {
    if !probs_valid(opts.probs) {
        return Err(crate::entry_refusals::adaptive_options(format!(
            "{who}: probabilities {:?} must be strictly inside (0, 1) and strictly increasing",
            opts.probs
        )));
    }
    if let Some((m, a)) = opts.thresholds.iter().find(|(_, a)| !a.is_finite()) {
        return Err(crate::entry_refusals::adaptive_options(format!(
            "{who}: the threshold for {m:?} is {a}; it must be finite"
        )));
    }
    Ok(())
}

/// Runs per draw: one per metrics row.
pub(super) fn runs_per_draw(result: &AdaptiveSimulationResult) -> BTreeMap<usize, usize> {
    let mut runs: BTreeMap<usize, usize> = BTreeMap::new();
    for row in &result.metrics {
        *runs.entry(row.draw).or_default() += 1;
    }
    runs
}

/// The summary kernel: per-draw statistics, bands and prediction intervals of `values`
/// (`(run, metric, value)`, at most one per run and metric) over `runs` runs per draw.
/// [`summarize_adaptive`] feeds it one arm's values; the contrast feeds it the paired
/// differences of two arms.
pub(super) fn summarize_values<'a>(
    runs: &BTreeMap<usize, usize>,
    values: impl Iterator<Item = (RunKey<'a>, AdaptiveSummaryMetric, f64)>,
    opts: &AdaptiveSummaryOptions,
) -> AdaptivePopulationSummary {
    // The values present per (metric, draw), each with its run's sim.
    let mut by_metric: BTreeMap<AdaptiveSummaryMetric, BTreeMap<usize, Vec<(usize, f64)>>> =
        SCALAR_METRICS.iter().map(|m| (m.clone(), BTreeMap::new())).collect();
    for (run, metric, v) in values {
        by_metric.entry(metric).or_default().entry(run.draw).or_default().push((run.sim, v));
    }
    let mut values = by_metric;

    for (m, _) in &opts.thresholds {
        values.entry(m.clone()).or_default();
    }
    let mut thresholds: Vec<(&AdaptiveSummaryMetric, f64)> = Vec::new();
    for (m, a) in &opts.thresholds {
        if !thresholds.iter().any(|(tm, ta)| *tm == m && ta.to_bits() == a.to_bits()) {
            thresholds.push((m, *a));
        }
    }

    let probs = opts.probs;
    let need = probs.map(min_n_for_prob);
    let quantiles = |vals: &mut Vec<f64>| AdaptiveQuantiles {
        probs,
        values: gated_quantiles(vals, probs, need),
    };
    let n_draws = runs.len();
    let total_runs: usize = runs.values().sum();
    let mut out = AdaptivePopulationSummary {
        n_draws,
        rows: Vec::new(),
        bands: Vec::new(),
        prediction_intervals: Vec::new(),
    };
    for (metric, by_draw) in &values {
        let mut stats = vec![AdaptiveSummaryStatistic::Mean, AdaptiveSummaryStatistic::Median];
        stats.extend(
            thresholds
                .iter()
                .filter(|(m, _)| *m == metric)
                .map(|&(_, a)| AdaptiveSummaryStatistic::FractionAtLeast(a)),
        );
        for &statistic in &stats {
            let mut across: Vec<f64> = Vec::with_capacity(n_draws);
            for (&draw, &n_runs) in runs {
                let cell = by_draw.get(&draw).map_or(&[][..], Vec::as_slice);
                let vals: Vec<f64> = cell.iter().map(|&(_, v)| v).collect();
                let value = draw_statistic(&vals, statistic);
                across.extend(value);
                out.rows.push(AdaptiveSummaryRow {
                    draw,
                    metric: metric.clone(),
                    statistic,
                    value,
                    n_runs,
                    n_missing: missing(n_runs, vals.len()),
                    mc_se: mc_se(cell, statistic),
                });
            }
            let n_draws_present = across.len();
            out.bands.push(AdaptiveSummaryBand {
                metric: metric.clone(),
                statistic,
                n_draws_present,
                n_draws_missing: n_draws - n_draws_present,
                status: if n_draws > 1 {
                    AdaptiveBandStatus::Quantiles(quantiles(&mut across))
                } else {
                    AdaptiveBandStatus::NotDrawn
                },
            });
        }
        let mut pooled: Vec<f64> = by_draw.values().flatten().map(|&(_, v)| v).collect();
        let n_present = pooled.len();
        out.prediction_intervals.push(AdaptivePredictionInterval {
            metric: metric.clone(),
            quantiles: quantiles(&mut pooled),
            n_runs: total_runs,
            n_missing: missing(total_runs, n_present),
        });
    }
    out
}

/// The run a value came from: `(subject, draw, sim, rep)`, the key of a metrics row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct RunKey<'a> {
    pub(super) subject: &'a str,
    pub(super) draw: usize,
    pub(super) sim: usize,
    pub(super) rep: usize,
}

/// Every value a run carries, with its run and metric: the scalar metrics of each metrics
/// row (those it carries), then each analyte's latent signal at each decision. The one
/// producer behind [`summarize_adaptive`] and [`super::decompose_adaptive`], so both read
/// the same values and count the same values as missing.
pub(super) fn run_values(
    result: &AdaptiveSimulationResult,
) -> impl Iterator<Item = (RunKey<'_>, AdaptiveSummaryMetric, f64)> {
    let scalars = result.metrics.iter().flat_map(|row| {
        let run = RunKey { subject: &row.subject, draw: row.draw, sim: row.sim, rep: row.rep };
        SCALAR_METRICS.iter().filter_map(move |m| m.value(row).map(|v| (run, m.clone(), v)))
    });
    let signals = result.decisions.iter().flat_map(|d| {
        let run = RunKey { subject: &d.subject, draw: d.draw, sim: d.sim, rep: d.rep };
        d.observed_signals.iter().map(move |s| {
            let metric = AdaptiveSummaryMetric::Signal {
                analyte: s.name.clone(),
                decision_idx: d.decision_idx,
            };
            (run, metric, s.latent)
        })
    });
    scalars.chain(signals)
}

/// Runs that carry no value. At most one value per run is an engine invariant (one
/// metrics row per run, and the driver refuses two monitors with one name), so a larger
/// count is a defect: it panics in release too, rather than wrapping into a huge count.
fn missing(n_runs: usize, present: usize) -> usize {
    n_runs.checked_sub(present).expect("more values than runs: one analyte recorded twice")
}

/// The delete-one jackknife standard error of a `Mean` or `FractionAtLeast` over the sims
/// of `cell` (`(sim, value)`, one draw's values), each sim one unit: a smooth statistic is
/// a ratio `T / N` of per-sim sums, so a replicate leaving sim `u` out is
/// `(T − t_u) / (N − n_u)`, with the statistic's own per-run [`score`]. Only sims carrying
/// a value are units. `None` for `Median` and below two units; otherwise `NaN` if any value
/// is not finite, like the statistic itself (`≥` alone would score a `NaN` as 0).
fn mc_se(cell: &[(usize, f64)], statistic: AdaptiveSummaryStatistic) -> Option<f64> {
    let mut per_sim: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for &(sim, v) in cell {
        let unit = per_sim.entry(sim).or_default();
        unit.0 += score(v, statistic)?;
        unit.1 += 1.0;
    }
    let units: Vec<(f64, f64)> = per_sim.into_values().collect();
    if units.len() < 2 {
        return None;
    }
    if cell.iter().any(|(_, v)| !v.is_finite()) {
        return Some(f64::NAN);
    }
    let (t, n) = units.iter().fold((0.0, 0.0), |(t, n), u| (t + u.0, n + u.1));
    let ratio = |drop: Option<usize>| {
        vec![Some(drop.map_or(t / n, |u| (t - units[u].0) / (n - units[u].1)))]
    };
    jackknife(units.len(), ratio)[0]
}

/// One run's score under a smooth statistic, whose value is the mean score: the value
/// itself for `Mean`, the indicator `v ≥ a` for `FractionAtLeast(a)`. `None` for `Median`,
/// which is not a mean. The one definition behind [`draw_statistic`] and [`mc_se`].
fn score(v: f64, statistic: AdaptiveSummaryStatistic) -> Option<f64> {
    match statistic {
        AdaptiveSummaryStatistic::Mean => Some(v),
        AdaptiveSummaryStatistic::FractionAtLeast(a) => Some(if v >= a { 1.0 } else { 0.0 }),
        AdaptiveSummaryStatistic::Median => None,
    }
}

/// One statistic over the values one draw's runs carry; `None` if there are none. A
/// non-finite value makes it `NaN` rather than being dropped (`≥` and `max` would discard
/// a `NaN`; the sum and the quantile do not).
fn draw_statistic(vals: &[f64], statistic: AdaptiveSummaryStatistic) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    if vals.iter().any(|v| !v.is_finite()) {
        return Some(f64::NAN);
    }
    let n = vals.len() as f64;
    Some(match statistic {
        AdaptiveSummaryStatistic::Median => {
            let mut v = vals.to_vec();
            v.sort_by(f64::total_cmp);
            quantile_sorted(&v, 0.5)
        }
        _ => vals.iter().map(|&v| score(v, statistic)).sum::<Option<f64>>()? / n,
    })
}
