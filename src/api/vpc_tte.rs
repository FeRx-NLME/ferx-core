//! Kaplan–Meier visual predictive check for a single-event TTE endpoint (#1886).
//!
//! [`vpc_tte`] simulates `n_sim` replicates of the observed design with one
//! [`simulate_with_options`](super::simulate_with_options) call, computes each
//! replicate's Kaplan–Meier curve with [`crate::stats::km`], and returns the
//! observed curve beside the pointwise band of the replicates, per stratum. It never
//! calls `fit()`.
//!
//! **Censoring.** The simulator either censors a record at its own time (a
//! right-censored record) or draws it unbounded (an event record), and a set
//! horizon replaces every record's window. So the replicates are drawn with
//! horizon `H` (the caller's, or the latest follow-up time in the data) and then,
//! under [`VpcCensoring::Observed`], given a censoring time `cᵢ` each: a draw
//! `t > cᵢ` becomes `(cᵢ, censored)`. A subject observed censored keeps its own time.
//! A subject observed with an event has no observed censoring time, only the fact
//! that it exceeds its event time `tᵢ`, so it draws one from the censoring
//! distribution `G` conditional on `c > tᵢ`, where `G` is the reverse Kaplan–Meier
//! curve of the observed censorings **in its own stratum** (`H` when `G` has no mass
//! past `tᵢ`). The replicate `cᵢ` is then marginally distributed as its stratum's `G`,
//! which is what makes the replicates exchangeable with the data when censoring is
//! independent of the event time within a stratum. A `G` pooled across strata would
//! give one arm's event subjects another arm's dropout (#2807 review r1). Keeping `H` for event subjects instead (#1886 plan D4) gives the
//! replicates too little censoring and a band too narrow: measured coverage 0.855
//! for a nominal 0.90 at 3H/4 with 45 % censored, against 0.91 with the draw.
//! [`VpcCensoring::Horizon`] skips the step and censors every replicate at `H`.

use super::simulate::simulate_with_options_impl;
use super::{SimulateOptions, SimulationResult};
use crate::diagnostics::EngineError;
use crate::stats::km::{kaplan_meier, min_curves_for, probs_valid, vpc_band, KmCurve};
use crate::types::{
    CompiledModel, EndpointLikelihood, EventType, ModelParameters, ObsRecord, Population,
    SimOutcome, TteRecurrence,
};
use std::collections::HashMap;

/// The most strata [`VpcStratify::Covariate`] accepts before it asks for binning.
pub const VPC_TTE_MAX_STRATA: usize = 10;

/// The number of evenly spaced points [`VpcGrid::Auto`] adds on `[0, H]`.
const AUTO_GRID_POINTS: usize = 101;

/// The times the band is evaluated at.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum VpcGrid {
    /// `{0}`, the stratum's observed event and censoring times, and 101 evenly
    /// spaced points on `[0, H]`. The observed times make the overlay exact; the
    /// even points keep the band from skipping steps between sparse observed times.
    #[default]
    Auto,
    /// Exactly these times (sorted, duplicates dropped).
    Times(Vec<f64>),
    /// `{0}`, the observed times and every step time of every replicate curve in
    /// the stratum, up to `H`: exact step ribbons, and a large grid.
    AllSteps,
}

/// How the replicates are censored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VpcCensoring {
    /// Censor each subject observed censored at its own time; draw an event subject's
    /// censoring time from the reverse Kaplan–Meier censoring distribution given that it
    /// exceeds the event time (see the module doc).
    #[default]
    Observed,
    /// Censor every replicate at `H` only.
    Horizon,
}

/// How subjects are split into strata. A subject's stratum is the same in every
/// replicate, since the replicates reuse the observed population.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum VpcStratify {
    /// One stratum, `all`.
    #[default]
    None,
    /// By the exact value of this covariate at the subject's first record. More than
    /// [`VPC_TTE_MAX_STRATA`] distinct values are refused.
    Covariate(String),
    /// By the amount of the subject's first dose (0 for a subject with no dose).
    Dose,
}

/// Options for [`vpc_tte`].
#[derive(Debug, Clone)]
pub struct VpcTteOptions {
    /// Number of simulated replicates. At least `min_curves_for(probs) + 1`
    /// (20 at the default probabilities).
    pub n_sim: usize,
    /// Band probabilities, strictly inside (0, 1) and increasing. Default
    /// `[0.05, 0.5, 0.95]`.
    pub probs: [f64; 3],
    pub grid: VpcGrid,
    pub censoring: VpcCensoring,
    pub stratify: VpcStratify,
    /// The TTE endpoint to check. Required when the model has more than one.
    pub cmt: Option<usize>,
    /// Passed to the simulator. `sim.seed` seeds the replicates and the censoring draws;
    /// `sim.horizon` is `H`, and when `None` `H` is the latest time recorded on `cmt`.
    pub sim: SimulateOptions,
}

impl Default for VpcTteOptions {
    fn default() -> Self {
        VpcTteOptions {
            n_sim: 200,
            probs: [0.05, 0.5, 0.95],
            grid: VpcGrid::Auto,
            censoring: VpcCensoring::Observed,
            stratify: VpcStratify::None,
            cmt: None,
            sim: SimulateOptions::default(),
        }
    }
}

/// One stratum of a [`TteVpc`]. `grid`, `obs_on_grid`, `lo`, `med`, `hi` and
/// `n_used` are parallel. A band entry is `None` where fewer than
/// `min_curves_for(probs)` replicates are still under follow-up.
#[derive(Debug, Clone, PartialEq)]
pub struct TteVpcStratum {
    /// `all`, `NAME=value` or `dose=value`.
    pub label: String,
    /// The stratum's covariate or dose value; `None` for [`VpcStratify::None`].
    pub value: Option<f64>,
    pub n_subjects: usize,
    /// The observed Kaplan–Meier curve with its Nelson–Aalen cumulative hazard.
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

/// The result of [`vpc_tte`].
#[derive(Debug, Clone, PartialEq)]
pub struct TteVpc {
    pub cmt: usize,
    /// `H`, the horizon the replicates were simulated to.
    pub horizon: f64,
    pub n_sim: usize,
    pub probs: [f64; 3],
    pub censoring: VpcCensoring,
    pub strata: Vec<TteVpcStratum>,
    /// The simulator's warnings (a degenerate hazard draw, a failed ODE solve, …).
    pub warnings: Vec<String>,
}

/// A replicate's draw `(t, observed)` censored at the subject's censoring time `c`: a
/// draw past `c` becomes `(c, censored)`, anything else is kept.
fn post_censor(t: f64, observed: bool, c: f64) -> (f64, bool) {
    if t > c {
        (c, false)
    } else {
        (t, observed)
    }
}

/// The censoring distribution `G` of `rows`: their reverse Kaplan–Meier curve, a subject
/// observed censored counting as a censoring "event" and every other subject as censored
/// for it. Built per stratum by the caller.
fn censoring_curve<'a>(rows: impl Iterator<Item = &'a ObsRow>) -> KmCurve {
    let (t, c): (Vec<f64>, Vec<bool>) = rows.map(|r| (r.time, r.censor_at.is_some())).unzip();
    kaplan_meier(&t, &c)
        .expect("`observed_rows` refuses a time that is not finite and non-negative")
}

/// A censoring time drawn from `g` conditional on `c > t`, by inversion with `u ∈ [0, 1)`:
/// the first time `x` with `G(x) / G(t) ≤ u`, or `horizon` when there is none (no
/// censoring mass left past `t`). `G` is non-increasing and `u < 1`, so that ratio is
/// reached only past `t`, and only at a step where `G` drops, a censoring time.
fn draw_censoring(g: &KmCurve, t: f64, u: f64, horizon: f64) -> f64 {
    let g_t = match g.at(t) {
        Some(v) if v > 0.0 => v,
        _ => return horizon,
    };
    g.time
        .iter()
        .zip(&g.surv)
        .find(|(_, s)| **s / g_t <= u)
        .map_or(horizon, |(x, _)| *x)
}

/// A uniform `[0, 1)` from `(seed, sim, subject)`, independent of row order (SplitMix64).
fn censoring_uniform(seed: u64, sim: usize, subject: usize) -> f64 {
    let mut z = seed
        ^ (sim as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (subject as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// One subject's observed record on the checked endpoint.
struct ObsRow {
    id: String,
    time: f64,
    event: bool,
    /// Its own censoring time when every TTE record it has is right-censored.
    censor_at: Option<f64>,
    stratum: usize,
}

fn refuse(message: String) -> EngineError {
    super::entry_refusals::vpc_tte_input(message)
}

/// The endpoint `opts.cmt` names, or the model's only TTE endpoint.
fn resolve_cmt(model: &CompiledModel, cmt: Option<usize>) -> Result<usize, EngineError> {
    let tte = model.tte_cmts();
    let list = tte
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let c = match (cmt, tte.as_slice()) {
        (_, []) => {
            return Err(refuse(
                "vpc_tte: the model declares no TTE endpoint ([event_model]).".to_string(),
            ))
        }
        (Some(c), _) if !tte.contains(&c) => {
            return Err(refuse(format!(
                "vpc_tte: `cmt = {c}` is not a TTE endpoint. The model's TTE endpoints are on \
                 cmt {list}."
            )))
        }
        (Some(c), _) => c,
        (None, [only]) => *only,
        (None, _) => {
            return Err(refuse(format!(
                "vpc_tte: the model has TTE endpoints on cmt {list}; set `cmt` to the one to \
                 check. Under competing risks 1 − KM of one cause is not its cumulative \
                 incidence: the curve is that cause's survival with the other causes as \
                 censoring."
            )))
        }
    };
    if matches!(
        model.endpoints.get(&c),
        Some(EndpointLikelihood::Tte {
            recurrence: TteRecurrence::Repeated { .. },
            ..
        })
    ) {
        return Err(refuse(format!(
            "vpc_tte: cmt {c} is a repeated-events (RTTE) endpoint. A Kaplan–Meier VPC of \
             repeated events is not supported yet."
        )));
    }
    Ok(c)
}

/// Each subject's stratum value, or `None` for [`VpcStratify::None`].
fn stratum_value(
    subject: &crate::types::Subject,
    stratify: &VpcStratify,
) -> Result<Option<f64>, EngineError> {
    Ok(match stratify {
        VpcStratify::None => None,
        VpcStratify::Dose => Some(subject.doses.first().map_or(0.0, |d| d.amt)),
        VpcStratify::Covariate(name) => match subject.covariates.get(name) {
            Some(v) => Some(*v),
            None => {
                return Err(refuse(format!(
                    "vpc_tte: subject {} has no covariate `{name}` to stratify on.",
                    subject.id
                )))
            }
        },
    })
}

/// The observed rows on `cmt`, and the strata's values (sorted) and labels.
#[allow(clippy::type_complexity)]
fn observed_rows(
    population: &Population,
    cmt: usize,
    stratify: &VpcStratify,
) -> Result<(Vec<ObsRow>, Vec<Option<f64>>, Vec<String>), EngineError> {
    let mut picked: Vec<(String, f64, bool, Option<f64>, Option<f64>)> = Vec::new();
    let mut seen: HashMap<&str, ()> = HashMap::new();
    for s in &population.subjects {
        if seen.insert(s.id.as_str(), ()).is_some() {
            return Err(refuse(format!(
                "vpc_tte: subject id {} appears more than once in the population. Simulated \
                 rows are matched to subjects by id.",
                s.id
            )));
        }
        let mut on_cmt = Vec::new();
        let mut all_censored = true;
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
            all_censored &= matches!(event_type, EventType::RightCensored);
            if *rc != cmt {
                continue;
            }
            if !(time.is_finite() && *time >= 0.0) {
                return Err(refuse(format!(
                    "vpc_tte: subject {} has time {time} on cmt {cmt}, which is not a finite, \
                     non-negative number.",
                    s.id
                )));
            }
            if *entry_time > 0.0 {
                return Err(refuse(format!(
                    "vpc_tte: subject {} has entry time {entry_time} on cmt {cmt}. Left \
                     truncation (delayed entry) is not supported yet.",
                    s.id
                )));
            }
            let event = match event_type {
                EventType::Exact => true,
                EventType::RightCensored => false,
                EventType::IntervalCensored { .. } => {
                    return Err(refuse(format!(
                        "vpc_tte: subject {} has an interval-censored record on cmt {cmt}. A \
                         Kaplan–Meier curve needs exact or right-censored times.",
                        s.id
                    )))
                }
            };
            on_cmt.push((*time, event));
        }
        match on_cmt.as_slice() {
            [] => {}
            [(time, event)] => picked.push((
                s.id.clone(),
                *time,
                *event,
                all_censored.then_some(*time),
                stratum_value(s, stratify)?,
            )),
            more => {
                return Err(refuse(format!(
                    "vpc_tte: subject {} has {} records on cmt {cmt}. A single-event \
                     Kaplan–Meier curve takes one per subject.",
                    s.id,
                    more.len()
                )))
            }
        }
    }
    if picked.is_empty() {
        return Err(refuse(format!(
            "vpc_tte: no subject has a record on cmt {cmt}."
        )));
    }
    let mut values: Vec<Option<f64>> = picked.iter().map(|p| p.4).collect();
    values.sort_by(|a, b| a.unwrap_or(0.0).total_cmp(&b.unwrap_or(0.0)));
    values.dedup_by(|a, b| a.map(f64::to_bits) == b.map(f64::to_bits));
    if let VpcStratify::Covariate(name) = stratify {
        if values.len() > VPC_TTE_MAX_STRATA {
            return Err(refuse(format!(
                "vpc_tte: covariate `{name}` has {} distinct values, more than the \
                 {VPC_TTE_MAX_STRATA} strata a VPC can show. Bin it into a categorical covariate \
                 and stratify on that.",
                values.len()
            )));
        }
    }
    let labels = values
        .iter()
        .map(|v| match (stratify, v) {
            (VpcStratify::Covariate(name), Some(v)) => format!("{name}={v}"),
            (VpcStratify::Dose, Some(v)) => format!("dose={v}"),
            _ => "all".to_string(),
        })
        .collect();
    let rows = picked
        .into_iter()
        .map(|(id, time, event, censor_at, v)| {
            let stratum = values
                .iter()
                .position(|x| x.map(f64::to_bits) == v.map(f64::to_bits))
                .expect("every value is in the deduplicated list");
            ObsRow {
                id,
                time,
                event,
                censor_at,
                stratum,
            }
        })
        .collect();
    Ok((rows, values, labels))
}

/// The grid of one stratum.
fn stratum_grid(grid: &VpcGrid, horizon: f64, observed: &KmCurve, reps: &[KmCurve]) -> Vec<f64> {
    let mut g: Vec<f64> = match grid {
        VpcGrid::Times(t) => t.clone(),
        VpcGrid::Auto | VpcGrid::AllSteps => {
            let mut g = vec![0.0];
            g.extend(observed.time.iter().copied().filter(|&t| t <= horizon));
            if *grid == VpcGrid::Auto {
                let n = AUTO_GRID_POINTS - 1;
                g.extend((0..=n).map(|k| horizon * k as f64 / n as f64));
            } else {
                for c in reps {
                    g.extend(c.time.iter().copied().filter(|&t| t <= horizon));
                }
            }
            g
        }
    };
    g.sort_by(f64::total_cmp);
    g.dedup();
    g
}

/// A Kaplan–Meier VPC for one single-event TTE endpoint.
///
/// Simulates `opts.n_sim` replicates of `population` under `params` with one
/// simulator call, then builds the observed Kaplan–Meier curve and the pointwise
/// `opts.probs` band of the replicate curves on the grid, per stratum. The module
/// doc explains the censoring; [`crate::stats::km`] the tie convention and the
/// quantile type (survfit's and R's type 7).
///
/// # Errors
///
/// `E_VPC_TTE_INPUT` for an input the curve would be wrong on: no TTE endpoint, a
/// `cmt` that is not one, several TTE endpoints and no `cmt`, a repeated-events
/// endpoint, left truncation, an interval-censored record, more than one record per
/// subject on the endpoint, a duplicated subject id, no record on it at all, bad
/// `probs`, too few `n_sim`, a grid time that is not finite and non-negative, a
/// stratification covariate a subject lacks or with more than
/// [`VPC_TTE_MAX_STRATA`] values. Otherwise any `Err` the simulator returns.
pub fn vpc_tte(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    opts: &VpcTteOptions,
) -> Result<TteVpc, EngineError> {
    let probs = opts.probs;
    if !probs_valid(probs) {
        return Err(refuse(format!(
            "vpc_tte: band probabilities {probs:?} must be strictly inside (0, 1) and strictly \
             increasing."
        )));
    }
    let need = min_curves_for(probs) + 1;
    if opts.n_sim < need {
        return Err(refuse(format!(
            "vpc_tte: `n_sim = {}` is too few. The band at probabilities {probs:?} needs at \
             least {need} replicates.",
            opts.n_sim
        )));
    }
    if let VpcGrid::Times(t) = &opts.grid {
        if let Some(bad) = t.iter().find(|t| !(t.is_finite() && **t >= 0.0)) {
            return Err(refuse(format!(
                "vpc_tte: grid time {bad} is not a finite, non-negative number."
            )));
        }
    }
    let cmt = resolve_cmt(model, opts.cmt)?;
    let (rows, values, labels) = observed_rows(population, cmt, &opts.stratify)?;
    let horizon = opts.sim.horizon.unwrap_or_else(|| {
        rows.iter()
            .map(|r| r.time)
            .fold(f64::NEG_INFINITY, f64::max)
    });

    let mut sim_opts = opts.sim.clone();
    sim_opts.horizon = Some(horizon);
    let out = simulate_with_options_impl(model, population, params, opts.n_sim, &sim_opts, false)?;

    let by_id: HashMap<&str, (usize, &ObsRow)> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.id.as_str(), (i, r)))
        .collect();
    let n_strata = values.len();
    let g: Vec<KmCurve> = (0..n_strata)
        .map(|k| censoring_curve(rows.iter().filter(|r| r.stratum == k)))
        .collect();
    let seed = opts.sim.seed.unwrap_or_else(rand::random);
    // reps[sim][stratum] = (times, events)
    let mut reps: Vec<Vec<(Vec<f64>, Vec<bool>)>> =
        vec![vec![(Vec::new(), Vec::new()); n_strata]; opts.n_sim];
    for SimulationResult {
        sim,
        id,
        cmt: rc,
        outcome,
        ..
    } in &out.results
    {
        let SimOutcome::Event { time, observed } = outcome else {
            continue;
        };
        if *rc != cmt {
            continue;
        }
        let Some(&(i, row)) = by_id.get(id.as_str()) else {
            continue;
        };
        let (t, e) = match opts.censoring {
            VpcCensoring::Observed => {
                let c = row.censor_at.unwrap_or_else(|| {
                    draw_censoring(
                        &g[row.stratum],
                        row.time,
                        censoring_uniform(seed, *sim, i),
                        horizon,
                    )
                });
                post_censor(*time, *observed, c)
            }
            VpcCensoring::Horizon => (*time, *observed),
        };
        let slot = &mut reps[sim - 1][row.stratum];
        slot.0.push(t);
        slot.1.push(e);
    }

    let mut strata = Vec::with_capacity(n_strata);
    for (k, (value, label)) in values.into_iter().zip(labels).enumerate() {
        let (ot, oe): (Vec<f64>, Vec<bool>) = rows
            .iter()
            .filter(|r| r.stratum == k)
            .map(|r| (r.time, r.event))
            .unzip();
        let observed = kaplan_meier(&ot, &oe).map_err(|e| refuse(format!("vpc_tte: {e}")))?;
        let curves = reps
            .iter()
            .map(|r| kaplan_meier(&r[k].0, &r[k].1))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| refuse(format!("vpc_tte: a replicate {e}")))?;
        let grid = stratum_grid(&opts.grid, horizon, &observed, &curves);
        let band = vpc_band(&curves, &grid, probs).map_err(|e| refuse(format!("vpc_tte: {e}")))?;
        strata.push(TteVpcStratum {
            label,
            value,
            n_subjects: ot.len(),
            obs_on_grid: grid.iter().map(|&t| observed.at(t)).collect(),
            observed,
            grid,
            lo: band.lo,
            med: band.med,
            hi: band.hi,
            n_used: band.n_used,
        });
    }
    Ok(TteVpc {
        cmt,
        horizon,
        n_sim: opts.n_sim,
        probs,
        censoring: opts.censoring,
        strata,
        warnings: out.warnings,
    })
}

#[cfg(test)]
#[path = "tests/vpc_tte_tests.rs"]
mod tests;
