//! Visual predictive check for a repeated-events (RTTE) endpoint (#1886): the mean
//! cumulative function and Kaplan–Meier curves by event order.
//!
//! [`vpc_rtte`] simulates `n_sim` replicates of the observed design with one
//! [`simulate_with_options`](super::simulate_with_options) call and returns, per
//! stratum, three families of observed curves beside the pointwise band of the same
//! curves over the replicates. It never calls `fit()`. The estimators and their tie
//! convention are [`crate::stats::km`]'s.
//!
//! **Follow-up.** A subject's follow-up end `cᵢ` is the time of its last record on the
//! endpoint, an event or a censoring. A censoring that is not the last record does not
//! end follow-up, which is how the likelihood reads it.
//!
//! **The three risk sets.**
//!
//! - *Mean cumulative function.* A subject is at risk while it is under follow-up,
//!   whatever number of events it has had: `Y(t) = #{i : cᵢ ≥ t}` and
//!   `MCF(t) = Σ_{s ≤ t} dN(s) / Y(s)`, the expected number of events per subject by `t`.
//! - *Absolute time, order `k`.* Every subject gives the time of its `k`-th event, or
//!   `cᵢ` censored when it has fewer than `k`. The Kaplan–Meier curve of those is the
//!   probability of fewer than `k` events by `t`.
//! - *Gap time, order `k ≥ 2`.* Only the subjects with at least `k − 1` events. Each
//!   gives the time from its `(k − 1)`-th event to its `k`-th, or to `cᵢ` censored. Gap
//!   order 1 would be absolute order 1 again, so it is not returned.
//!
//! Two events at one time in one subject both count in the MCF, make the absolute
//! curves of two consecutive orders step at the same time, and give a gap of 0. An
//! event at `cᵢ` is an event with the subject still at risk, and the next order's gap
//! is then censored at 0.
//!
//! **What a gap-time band means.** It is the distribution, over replicates, of the same
//! statistic computed the same way: the Kaplan–Meier curve of the `k`-th gap among the
//! replicate's subjects that reached `k − 1` events. Those are different subjects in
//! different replicates, and for `k ≥ 2` they are selected by having had `k − 1` events
//! with follow-up left that depends on the earlier gaps. So the curve is a predictive
//! check, the observed curve and the replicates carrying the same selection, and not an
//! estimate of the marginal gap-time distribution. A replicate in which no subject
//! reached `k − 1` events has no curve and is left out of the band.
//!
//! **Censoring of the replicates.** The simulator follows every subject to the horizon
//! `H` (the caller's, or the latest recorded time on the endpoint). A band built from
//! that has more subjects at risk late than the data had, and is too narrow there.
//! Under [`VpcCensoring::Observed`] each replicate subject is therefore cut at its own
//! observed follow-up end: its events after `cᵢ` are dropped and its follow-up ends at
//! `min(cᵢ, H)`. Unlike a single event, a recurrent subject's follow-up end is always
//! observed, so no censoring time is drawn. [`VpcCensoring::Horizon`] follows every
//! replicate subject to `H`.
//!
//! **The data on `[0, H]` only.** The observed streams are cut at `H` the same way,
//! once, before anything is computed from them: a subject's events after `H` are
//! dropped, an event exactly at `H` is kept, and its follow-up ends at `min(cᵢ, H)`. The
//! result is the one for the data cut at `H` first, every curve, the default
//! `max_order` and `W_VPC_RTTE_ENDS_ON_EVENT` included (#2908). Without the cut, a
//! gap-time risk set and the gap a subject is censored at would read follow-up the
//! replicates do not have. With `sim.horizon` unset `H` is the latest follow-up end and
//! the cut does nothing. [`vpc_tte`](super::vpc_tte) does not cut its observed records:
//! an event after `H` cut to a censoring at `H` would enter its reverse Kaplan–Meier
//! censoring distribution and move the band on `[0, H]`.

use super::simulate::simulate_with_options_impl;
use super::vpc_tte::{
    build_strata, check_band_options, checked_record, duplicate_id, no_record, refuse, resolve_cmt,
    stratum_grid, stratum_value, Entry, VpcCensoring, VpcGrid, VpcStratify,
};
use super::{SimulateOptions, SimulationResult};
use crate::diagnostics::EngineError;
use crate::stats::km::{kaplan_meier, mean_cumulative, vpc_band, vpc_band_mcf, KmCurve, McfCurve};
use crate::types::{CompiledModel, ModelParameters, ObsRecord, Population, SimOutcome};
use std::collections::HashMap;

const WHO: Entry = Entry::Rtte;

/// The default `max_order` is the largest `k` for which at least this many observed
/// subjects have `k` or more events.
const MIN_SUBJECTS_PER_ORDER: usize = 10;

/// Options for [`vpc_rtte`].
#[derive(Debug, Clone)]
pub struct VpcRtteOptions {
    /// Number of simulated replicates. At least `min_curves_for(probs) + 1`
    /// (20 at the default probabilities).
    pub n_sim: usize,
    /// Band probabilities, strictly inside (0, 1) and increasing. Default
    /// `[0.05, 0.5, 0.95]`.
    pub probs: [f64; 3],
    pub grid: VpcGrid,
    /// How long a replicate subject is followed (see the module doc).
    pub censoring: VpcCensoring,
    pub stratify: VpcStratify,
    /// The repeated-events endpoint to check. Required when the model has more than
    /// one TTE endpoint.
    pub cmt: Option<usize>,
    /// The highest event order `K`: absolute-time curves for orders `1..=K` and
    /// gap-time curves for `2..=K`. `None` takes the largest `k` for which at least 10
    /// observed subjects, counted across all strata, have `k` or more events, and at
    /// least 1. One `K` applies to every stratum. `Some(0)` is refused.
    pub max_order: Option<usize>,
    /// Passed to the simulator. `sim.seed` seeds the replicates; `sim.horizon` is `H`,
    /// and when `None` `H` is the latest time recorded on `cmt`. The observed data are
    /// read on `[0, H]` only (see the module doc).
    pub sim: SimulateOptions,
}

impl Default for VpcRtteOptions {
    fn default() -> Self {
        VpcRtteOptions {
            n_sim: 200,
            probs: [0.05, 0.5, 0.95],
            grid: VpcGrid::Auto,
            censoring: VpcCensoring::Observed,
            stratify: VpcStratify::None,
            cmt: None,
            max_order: None,
            sim: SimulateOptions::default(),
        }
    }
}

/// One Kaplan–Meier curve by event order with its band. `grid`, `obs_on_grid`, `lo`,
/// `med`, `hi` and `n_used` are parallel. A band entry is `None` where fewer than
/// `min_curves_for(probs)` replicate curves are defined.
///
/// When no observed subject is at risk for the order (a gap-time order nobody reached
/// in this stratum) `observed` is empty and every band entry is `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct RtteOrderBand {
    /// The event order `k`, from 1.
    pub order: usize,
    /// The observed subjects in the curve: all of the stratum's for absolute time,
    /// those with at least `k − 1` events for gap time.
    pub n_subjects: usize,
    pub observed: KmCurve,
    pub grid: Vec<f64>,
    /// The observed curve on the grid (`None` past its last time unless it reached 0).
    pub obs_on_grid: Vec<Option<f64>>,
    pub lo: Vec<Option<f64>>,
    pub med: Vec<Option<f64>>,
    pub hi: Vec<Option<f64>>,
    /// How many replicate curves are defined at each grid time.
    pub n_used: Vec<usize>,
}

/// The mean cumulative function with its band, laid out as [`RtteOrderBand`].
#[derive(Debug, Clone, PartialEq)]
pub struct RtteMcfBand {
    pub observed: McfCurve,
    pub grid: Vec<f64>,
    /// The observed curve on the grid (`None` past the last follow-up end).
    pub obs_on_grid: Vec<Option<f64>>,
    pub lo: Vec<Option<f64>>,
    pub med: Vec<Option<f64>>,
    pub hi: Vec<Option<f64>>,
    /// How many replicate curves are defined at each grid time.
    pub n_used: Vec<usize>,
}

/// One stratum of an [`RtteVpc`].
#[derive(Debug, Clone, PartialEq)]
pub struct RtteVpcStratum {
    /// `all`, `NAME=value` or `dose=value`.
    pub label: String,
    /// The stratum's covariate or dose value; `None` for [`VpcStratify::None`].
    pub value: Option<f64>,
    pub n_subjects: usize,
    pub mcf: RtteMcfBand,
    /// Absolute-time curves, orders `1..=max_order`.
    pub absolute: Vec<RtteOrderBand>,
    /// Gap-time curves, orders `2..=max_order`.
    pub gap: Vec<RtteOrderBand>,
}

/// The result of [`vpc_rtte`].
#[derive(Debug, Clone, PartialEq)]
pub struct RtteVpc {
    pub cmt: usize,
    /// `H`, the horizon the replicates were simulated to.
    pub horizon: f64,
    pub n_sim: usize,
    pub probs: [f64; 3],
    pub censoring: VpcCensoring,
    /// The highest event order returned, as given or as chosen from the data.
    pub max_order: usize,
    pub strata: Vec<RtteVpcStratum>,
    /// The warning of this check about the data's follow-up on `[0, H]`
    /// (`W_VPC_RTTE_ENDS_ON_EVENT`), then the simulator's (a degenerate recurrent
    /// hazard, …).
    pub warnings: Vec<String>,
}

/// One subject's observed stream on the checked endpoint.
struct Stream {
    id: String,
    /// Its event times, in time order.
    events: Vec<f64>,
    /// Its follow-up end: the time of its last record, event or censoring.
    end: f64,
    /// Whether that last record is an event.
    ends_on_event: bool,
    stratum: usize,
}

/// Which time an order's curve runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Clock {
    Absolute,
    Gap,
}

/// A subject's `(time, is an event)` on the order-`order` curve of `clock`, from its
/// event times (in time order, none after `end`) and its follow-up end; `None` when it
/// is not in that curve's risk set. The observed data and every replicate go through
/// this one function.
///
/// Absolute time: the `order`-th event, else `end` censored. Gap time: only a subject
/// with `order − 1` events, measured from the last of those (from 0 for order 1).
fn order_mark(events: &[f64], end: f64, order: usize, clock: Clock) -> Option<(f64, bool)> {
    let origin = match clock {
        Clock::Absolute => 0.0,
        Clock::Gap if order == 1 => 0.0,
        Clock::Gap => *events.get(order - 2)?,
    };
    Some(match events.get(order - 1) {
        Some(t) => (t - origin, true),
        None => (end - origin, false),
    })
}

/// The Kaplan–Meier curve of order `order` on `clock` over `streams` (event times and
/// follow-up end per subject), and the number of subjects in it.
fn order_curve<'a>(
    streams: impl Iterator<Item = (&'a [f64], f64)>,
    order: usize,
    clock: Clock,
) -> (KmCurve, usize) {
    let (times, events): (Vec<f64>, Vec<bool>) = streams
        .filter_map(|(ev, end)| order_mark(ev, end, order, clock))
        .unzip();
    let curve = kaplan_meier(&times, &events)
        .expect("event times and follow-up ends are finite, non-negative and ordered");
    (curve, times.len())
}

/// The mean cumulative function over `streams`.
fn mcf_curve<'a>(streams: impl Iterator<Item = (&'a [f64], f64)>) -> McfCurve {
    let (mut events, mut ends) = (Vec::new(), Vec::new());
    for (ev, end) in streams {
        events.extend_from_slice(ev);
        ends.push(end);
    }
    mean_cumulative(&events, &ends)
        .expect("event times are finite, non-negative and not after their follow-up end")
}

/// A stream (event times in time order) followed to `end`, cut at `horizon`: the events
/// up to `min(end, horizon)`, an event exactly there kept, and that time as the new
/// follow-up end. The observed streams and the replicates under
/// [`VpcCensoring::Observed`] both go through it.
fn cut_at(events: &[f64], end: f64, horizon: f64) -> (&[f64], f64) {
    let end = end.min(horizon);
    (&events[..events.partition_point(|&t| t <= end)], end)
}

/// The part of a replicate subject's simulated events (in time order, all before
/// `horizon`) that the VPC keeps, and where its follow-up ends, given its observed
/// follow-up end.
fn replicate_follow_up(
    events: &[f64],
    observed_end: f64,
    horizon: f64,
    censoring: VpcCensoring,
) -> (&[f64], f64) {
    match censoring {
        VpcCensoring::Observed => cut_at(events, observed_end, horizon),
        VpcCensoring::Horizon => (events, horizon),
    }
}

/// The observed streams on `cmt`, and the strata's values (sorted) and labels.
#[allow(clippy::type_complexity)]
fn observed_streams(
    population: &Population,
    cmt: usize,
    stratify: &VpcStratify,
) -> Result<(Vec<Stream>, Vec<Option<f64>>, Vec<String>), EngineError> {
    let mut picked: Vec<(String, Vec<f64>, f64, bool)> = Vec::new();
    let mut per_subject: Vec<Option<f64>> = Vec::new();
    let mut seen: HashMap<&str, ()> = HashMap::new();
    for s in &population.subjects {
        if seen.insert(s.id.as_str(), ()).is_some() {
            return Err(duplicate_id(WHO, &s.id));
        }
        let mut events = Vec::new();
        let mut end: Option<f64> = None;
        let mut ends_on_event = false;
        for r in &s.obs_records {
            let ObsRecord::Event {
                time,
                event_type,
                entry_time,
                cmt: rc,
            } = r
            else {
                continue;
            };
            if *rc != cmt {
                continue;
            }
            let (t, is_event) = checked_record(WHO, &s.id, cmt, *time, *entry_time, event_type)?;
            if let Some(prev) = end.filter(|prev| t < *prev) {
                return Err(refuse(format!(
                    "vpc_rtte: subject {} has records on cmt {cmt} out of time order (time {t} \
                     follows {prev}). Sort each subject's records by time.",
                    s.id
                )));
            }
            if is_event {
                events.push(t);
            }
            end = Some(t);
            ends_on_event = is_event;
        }
        let Some(end) = end else {
            continue;
        };
        per_subject.push(stratum_value(WHO, s, stratify)?);
        picked.push((s.id.clone(), events, end, ends_on_event));
    }
    if picked.is_empty() {
        return Err(no_record(WHO, cmt));
    }
    let (values, labels, index) = build_strata(WHO, &per_subject, stratify)?;
    let streams = picked
        .into_iter()
        .zip(index)
        .map(|((id, events, end, ends_on_event), stratum)| Stream {
            id,
            events,
            end,
            ends_on_event,
            stratum,
        })
        .collect();
    Ok((streams, values, labels))
}

/// The simulated event times on `cmt` as `[replicate][subject]`, each in the simulator's
/// (time) order. The simulator regenerates one stream per subject per replicate,
/// however many records the subject was observed with, and closes it with a censoring
/// row at the horizon, which carries no event and is not read.
fn replicate_events(
    results: &[SimulationResult],
    cmt: usize,
    streams: &[Stream],
    n_sim: usize,
) -> Vec<Vec<Vec<f64>>> {
    let by_id: HashMap<&str, usize> = streams
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id.as_str(), i))
        .collect();
    let mut reps = vec![vec![Vec::new(); streams.len()]; n_sim];
    for row in results {
        let SimOutcome::Event {
            time,
            observed: true,
        } = row.outcome
        else {
            continue;
        };
        if row.cmt != cmt {
            continue;
        }
        if let Some(&i) = by_id.get(row.id.as_str()) {
            reps[row.sim - 1][i].push(time);
        }
    }
    reps
}

/// The default `max_order`: the largest `k` for which at least
/// [`MIN_SUBJECTS_PER_ORDER`] observed subjects have `k` or more events, and at least 1.
fn default_max_order(streams: &[Stream]) -> usize {
    let with = |k: usize| streams.iter().filter(|s| s.events.len() >= k).count();
    let mut k = 1;
    while with(k + 1) >= MIN_SUBJECTS_PER_ORDER {
        k += 1;
    }
    k
}

/// What the data's follow-up does to the comparison, for [`RtteVpc::warnings`], read
/// on the streams already cut at `H`.
///
/// `W_VPC_RTTE_ENDS_ON_EVENT`: subjects whose last record is an event. Their follow-up
/// ends at that event, so the observed subject has an event exactly at its follow-up
/// end, where a simulated subject has one with probability 0. A subject followed past
/// `H` is cut to a follow-up ending at `H` with no event there, so it is not counted,
/// whatever its last record was.
fn follow_up_warnings(streams: &[Stream], cmt: usize) -> Vec<String> {
    let total = streams.len();
    let mut warnings = Vec::new();
    let on_event = streams.iter().filter(|s| s.ends_on_event).count();
    if on_event > 0 {
        warnings.push(format!(
            "W_VPC_RTTE_ENDS_ON_EVENT: {on_event} of {total} subjects on cmt {cmt} have an \
             event as their last record, so their follow-up ends at that event. Such a subject \
             has an event exactly at its follow-up end, which a simulated subject almost never \
             has. If follow-up went on after the last event, add a censoring record at its end."
        ));
    }
    warnings
}

/// A visual predictive check for one repeated-events (`type = rtte`) endpoint.
///
/// Simulates `opts.n_sim` replicates of `population` under `params` with one simulator
/// call. Per stratum it returns the observed mean cumulative function, the observed
/// Kaplan–Meier curves by event order on absolute time (orders `1..=K`) and on gap time
/// (orders `2..=K`), and for each the pointwise `opts.probs` band of the same curve
/// over the replicates. The module doc gives the risk sets, the reading of a gap-time
/// band and the censoring of the replicates; [`crate::stats::km`] the tie convention
/// and the quantile type (survfit's and R's type 7).
///
/// # Errors
///
/// `E_VPC_TTE_INPUT` for an input a curve would be wrong on: no TTE endpoint, a `cmt`
/// that is not one, several TTE endpoints and no `cmt`, a single-event endpoint (which
/// [`vpc_tte`](super::vpc_tte) draws), left truncation, an interval-censored record, a
/// time that is not finite and non-negative, a subject's records out of time order, a
/// duplicated subject id, no record on the endpoint, `max_order = Some(0)`, bad
/// `probs`, too few `n_sim`, a bad grid time, a stratification covariate a subject
/// lacks or with more than [`VPC_TTE_MAX_STRATA`](super::VPC_TTE_MAX_STRATA) values.
/// Otherwise any `Err` the simulator returns.
pub fn vpc_rtte(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    opts: &VpcRtteOptions,
) -> Result<RtteVpc, EngineError> {
    let probs = opts.probs;
    check_band_options(WHO, probs, opts.n_sim, &opts.grid)?;
    if opts.max_order == Some(0) {
        return Err(refuse(
            "vpc_rtte: `max_order = 0` asks for no event order. Set it to at least 1, or leave \
             it unset to take it from the data."
                .to_string(),
        ));
    }
    let cmt = resolve_cmt(WHO, model, opts.cmt)?;
    let (mut streams, values, labels) = observed_streams(population, cmt, &opts.stratify)?;
    let horizon = opts.sim.horizon.unwrap_or_else(|| {
        streams
            .iter()
            .map(|s| s.end)
            .fold(f64::NEG_INFINITY, f64::max)
    });
    // The data on [0, H] only (module doc), before anything is read from them: the
    // result is the one for the data cut at H first. A subject followed past H ends at
    // H on no event.
    for s in &mut streams {
        let (kept, end) = cut_at(&s.events, s.end, horizon);
        let kept = kept.len();
        s.events.truncate(kept);
        s.ends_on_event &= s.end <= horizon;
        s.end = end;
    }
    let max_order = opts
        .max_order
        .unwrap_or_else(|| default_max_order(&streams));

    let mut warnings = follow_up_warnings(&streams, cmt);

    let mut sim_opts = opts.sim.clone();
    sim_opts.horizon = Some(horizon);
    let out = simulate_with_options_impl(model, population, params, opts.n_sim, &sim_opts, false)?;
    warnings.extend(out.warnings);
    let reps = replicate_events(&out.results, cmt, &streams, opts.n_sim);

    // `check_band_options` validated `probs`, the one thing a band refuses.
    const PROBS_OK: &str = "the band probabilities were validated above";
    let mut strata = Vec::with_capacity(values.len());
    for (k, (value, label)) in values.into_iter().zip(labels).enumerate() {
        let members: Vec<usize> = (0..streams.len())
            .filter(|&i| streams[i].stratum == k)
            .collect();
        let observed = || {
            members
                .iter()
                .map(|&i| (streams[i].events.as_slice(), streams[i].end))
        };
        // One replicate's streams in this stratum, cut to the follow-up the VPC keeps.
        let cut: Vec<Vec<(&[f64], f64)>> = reps
            .iter()
            .map(|rep| {
                members
                    .iter()
                    .map(|&i| replicate_follow_up(&rep[i], streams[i].end, horizon, opts.censoring))
                    .collect()
            })
            .collect();

        let obs_mcf = mcf_curve(observed());
        let rep_mcf: Vec<McfCurve> = cut.iter().map(|r| mcf_curve(r.iter().copied())).collect();
        let grid = stratum_grid(
            &opts.grid,
            horizon,
            &obs_mcf.time,
            rep_mcf.iter().map(|c| c.time.as_slice()),
        );
        let band = vpc_band_mcf(&rep_mcf, &grid, probs).expect(PROBS_OK);
        let mcf = RtteMcfBand {
            obs_on_grid: grid.iter().map(|&t| obs_mcf.at(t)).collect(),
            observed: obs_mcf,
            grid,
            lo: band.lo,
            med: band.med,
            hi: band.hi,
            n_used: band.n_used,
        };

        let order_band = |order: usize, clock: Clock| -> RtteOrderBand {
            let (obs, n_subjects) = order_curve(observed(), order, clock);
            // Nobody observed at risk for this order: no curve to check, so no band.
            let curves: Vec<KmCurve> = if n_subjects == 0 {
                Vec::new()
            } else {
                cut.iter()
                    .map(|r| order_curve(r.iter().copied(), order, clock).0)
                    .collect()
            };
            let grid = stratum_grid(
                &opts.grid,
                horizon,
                &obs.time,
                curves.iter().map(|c| c.time.as_slice()),
            );
            let band = vpc_band(&curves, &grid, probs).expect(PROBS_OK);
            RtteOrderBand {
                order,
                n_subjects,
                obs_on_grid: grid.iter().map(|&t| obs.at(t)).collect(),
                observed: obs,
                grid,
                lo: band.lo,
                med: band.med,
                hi: band.hi,
                n_used: band.n_used,
            }
        };
        let absolute = (1..=max_order)
            .map(|order| order_band(order, Clock::Absolute))
            .collect();
        let gap = (2..=max_order)
            .map(|order| order_band(order, Clock::Gap))
            .collect();
        strata.push(RtteVpcStratum {
            label,
            value,
            n_subjects: members.len(),
            mcf,
            absolute,
            gap,
        });
    }
    Ok(RtteVpc {
        cmt,
        horizon,
        n_sim: opts.n_sim,
        probs,
        censoring: opts.censoring,
        max_order,
        strata,
        warnings,
    })
}

#[cfg(test)]
#[path = "tests/vpc_rtte_tests.rs"]
mod tests;
