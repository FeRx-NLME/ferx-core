//! Cumulative-incidence visual predictive check for competing risks (#1887).
//!
//! [`vpc_cif`] simulates `n_sim` replicates of the observed design with one
//! [`simulate_with_options`](super::simulate_with_options) call and returns, per stratum,
//! the observed Aalen–Johansen cumulative incidence of **every** cause and the observed
//! all-cause Kaplan–Meier curve, each beside the pointwise band of the replicates. It
//! never calls `fit()`. The estimator is [`crate::stats::cif`].
//!
//! **Why not one Kaplan–Meier VPC per cause.** `1 − KM` of one cause with the other
//! causes as censorings is not that cause's cumulative incidence: it estimates the
//! incidence in a world without the other causes, and overstates it. And the all-cause
//! curve cannot see a model that swaps the causes' rates while keeping their sum: the
//! cumulative incidences can.
//!
//! **The causes are every TTE endpoint of the model.** Each must be single-event. Every
//! subject with any cause record must have exactly one on every cause, all at the same
//! time, with at most one of them an event: the simulator competes only the causes a
//! subject has records on, so a subject with fewer would be simulated under fewer causes
//! than the data imply. A model with one TTE endpoint is accepted; its cumulative
//! incidence is `1 − KM`, and the result equals [`vpc_tte`](super::vpc_tte)'s turned
//! upside down.
//!
//! **Censoring** is [`vpc_tte`](super::vpc_tte)'s, on the all-cause time: under
//! [`VpcCensoring::Observed`] a subject observed censored for every cause keeps its own
//! time, and a subject with an event draws a censoring time from its stratum's reverse
//! Kaplan–Meier curve given that it exceeds the event time. A replicate's first event past
//! its censoring time becomes a censoring there. The draws are keyed by subject exactly
//! as `vpc_tte`'s are, so with one cause the two give the same replicates. The data are
//! read on `[0, H]` only, by the same cut. Like `vpc_tte`, this assumes censoring is
//! independent of the event within a stratum.
//!
//! **Pointwise bands do not add up.** At every time the observed `F_1 + … + F_K + S = 1`,
//! but the band medians (or limits) of the causes and of `S` need not sum to 1.

use super::simulate::simulate_with_options_impl;
use super::vpc_tte::{
    build_strata, check_band_options, checked_record, duplicate_id, horizon_and_cut, is_repeated,
    no_tte_endpoint, refuse, stratum_grid, stratum_value, Entry, ObsRow, ReplicateCensor,
    TteCensorRule, VpcCensoring, VpcGrid, VpcStratify,
};
use super::{SimulateOptions, SimulationResult};
use crate::diagnostics::EngineError;
use crate::stats::cif::{aalen_johansen, vpc_band_cif, CifCurve};
use crate::stats::km::{band_over, VpcBand};
use crate::types::{CompiledModel, ModelParameters, ObsRecord, Population, SimOutcome};
use std::collections::HashMap;

const WHO: Entry = Entry::Cif;

/// Options for [`vpc_cif`].
#[derive(Debug, Clone)]
pub struct VpcCifOptions {
    /// Number of simulated replicates. At least `min_curves_for(probs) + 1`
    /// (20 at the default probabilities).
    pub n_sim: usize,
    /// Band probabilities, strictly inside (0, 1) and increasing. Default
    /// `[0.05, 0.5, 0.95]`.
    pub probs: [f64; 3],
    /// The grid; [`VpcGrid::Auto`] reads the observed all-cause curve's times.
    pub grid: VpcGrid,
    pub censoring: VpcCensoring,
    pub stratify: VpcStratify,
    /// Passed to the simulator. `sim.seed` seeds the replicates and the censoring draws;
    /// `sim.horizon` is `H`, and when `None` `H` is the latest time recorded on a cause.
    pub sim: SimulateOptions,
}

impl Default for VpcCifOptions {
    fn default() -> Self {
        VpcCifOptions {
            n_sim: 200,
            probs: [0.05, 0.5, 0.95],
            grid: VpcGrid::Auto,
            censoring: VpcCensoring::Observed,
            stratify: VpcStratify::None,
            sim: SimulateOptions::default(),
        }
    }
}

/// An observed curve on the grid beside the pointwise band of the replicates. All five
/// vectors are parallel to the stratum's `grid`. A band entry is `None` where fewer than
/// `min_curves_for(probs)` replicates are still under follow-up; `obs_on_grid` is `None`
/// past the observed curve's last time.
#[derive(Debug, Clone, PartialEq)]
pub struct CifVpcBand {
    pub obs_on_grid: Vec<Option<f64>>,
    pub lo: Vec<Option<f64>>,
    pub med: Vec<Option<f64>>,
    pub hi: Vec<Option<f64>>,
    /// How many replicate curves are defined at each grid time.
    pub n_used: Vec<usize>,
}

/// The cumulative incidence of one cause in one stratum.
#[derive(Debug, Clone, PartialEq)]
pub struct CifCauseBand {
    /// The cause's TTE endpoint.
    pub cmt: usize,
    pub band: CifVpcBand,
}

/// One stratum of a [`CifVpc`].
#[derive(Debug, Clone, PartialEq)]
pub struct CifVpcStratum {
    /// `all`, `NAME=value` or `dose=value`.
    pub label: String,
    /// The stratum's covariate or dose value; `None` for [`VpcStratify::None`].
    pub value: Option<f64>,
    pub n_subjects: usize,
    /// The observed cumulative incidences, cause `k` being `CifVpc::cmts[k]`, on the rows
    /// of the observed all-cause Kaplan–Meier curve.
    pub observed: CifCurve,
    pub grid: Vec<f64>,
    /// The all-cause survival `S`: the observed Kaplan–Meier curve and its band.
    pub survival: CifVpcBand,
    /// One cumulative-incidence band per cause, in the order of `CifVpc::cmts`.
    pub causes: Vec<CifCauseBand>,
}

/// The result of [`vpc_cif`].
#[derive(Debug, Clone, PartialEq)]
pub struct CifVpc {
    /// The causes' TTE endpoints, ascending.
    pub cmts: Vec<usize>,
    /// `H`, the horizon the replicates were simulated to.
    pub horizon: f64,
    pub n_sim: usize,
    pub probs: [f64; 3],
    pub censoring: VpcCensoring,
    pub strata: Vec<CifVpcStratum>,
    /// The simulator's warnings (a degenerate hazard draw, a failed ODE solve, …).
    pub warnings: Vec<String>,
}

/// The model's TTE endpoints, every one single-event.
fn resolve_causes(model: &CompiledModel) -> Result<Vec<usize>, EngineError> {
    let cmts = model.tte_cmts();
    if cmts.is_empty() {
        return Err(no_tte_endpoint(WHO));
    }
    if let Some(c) = cmts.iter().find(|&&c| is_repeated(model, c)) {
        return Err(refuse(format!(
            "{WHO}: cmt {c} is a repeated-events (RTTE) endpoint. A cumulative incidence takes \
             single-event causes; `vpc_rtte` draws the Kaplan–Meier and \
             mean-cumulative-function VPC of a repeated-events endpoint."
        )));
    }
    Ok(cmts)
}

/// The observed rows (one per subject with a cause record, in population order) with each
/// row's cause (`0` censored, `k + 1` for `cmts[k]`), and the strata's values and labels.
#[allow(clippy::type_complexity)]
fn observed_cause_rows(
    population: &Population, cmts: &[usize], stratify: &VpcStratify,
) -> Result<(Vec<ObsRow>, Vec<usize>, Vec<Option<f64>>, Vec<String>), EngineError> {
    let list = cmts.iter().map(usize::to_string).collect::<Vec<_>>().join(", ");
    let mut picked: Vec<(String, f64, usize, Option<f64>)> = Vec::new();
    let mut seen: HashMap<&str, ()> = HashMap::new();
    for s in &population.subjects {
        if seen.insert(s.id.as_str(), ()).is_some() {
            return Err(duplicate_id(WHO, &s.id));
        }
        // Per cause: the (time, is an event) of each of its records.
        let mut per_cause: Vec<Vec<(f64, bool)>> = vec![Vec::new(); cmts.len()];
        for r in &s.obs_records {
            let ObsRecord::Event { time, event_type, entry_time, cmt } = r else {
                continue;
            };
            let Some(k) = cmts.iter().position(|c| c == cmt) else {
                continue;
            };
            per_cause[k].push(checked_record(WHO, &s.id, *cmt, *time, *entry_time, event_type)?);
        }
        if per_cause.iter().all(Vec::is_empty) {
            continue;
        }
        for (k, recs) in per_cause.iter().enumerate() {
            match recs.len() {
                1 => {}
                0 => {
                    return Err(refuse(format!(
                        "{WHO}: subject {} has no record on cmt {}. The simulator competes only \
                         the causes a subject has records on, so a subject needs one record on \
                         every cause (cmt {list}).",
                        s.id, cmts[k]
                    )))
                }
                n => {
                    return Err(refuse(format!(
                        "{WHO}: subject {} has {n} records on cmt {}. A cause takes one record \
                         per subject.",
                        s.id, cmts[k]
                    )))
                }
            }
        }
        let recs: Vec<(f64, bool)> = per_cause.into_iter().map(|r| r[0]).collect();
        let t0 = recs[0].0;
        if let Some(k) = recs.iter().position(|r| r.0.to_bits() != t0.to_bits()) {
            return Err(refuse(format!(
                "{WHO}: subject {} has time {t0} on cmt {} but {} on cmt {}. Every cause is \
                 followed to the same time, the first event or the censoring.",
                s.id, cmts[0], recs[k].0, cmts[k]
            )));
        }
        let events: Vec<usize> = (0..recs.len()).filter(|&k| recs[k].1).collect();
        if let [a, b, ..] = events.as_slice() {
            return Err(refuse(format!(
                "{WHO}: subject {} has an event on cmt {} and on cmt {}. Under competing risks \
                 a subject has one cause, the first.",
                s.id, cmts[*a], cmts[*b]
            )));
        }
        let cause = events.first().map_or(0, |k| k + 1);
        picked.push((s.id.clone(), t0, cause, stratum_value(WHO, s, stratify)?));
    }
    if picked.is_empty() {
        return Err(refuse(format!("{WHO}: no subject has a record on cmt {list}.")));
    }
    let per_subject: Vec<Option<f64>> = picked.iter().map(|p| p.3).collect();
    let (values, labels, index) = build_strata(WHO, &per_subject, stratify)?;
    let causes = picked.iter().map(|p| p.2).collect();
    let rows = picked
        .into_iter()
        .zip(index)
        .map(|((id, time, cause, _), stratum)| ObsRow {
            id,
            time,
            event: cause > 0,
            // Every record censored: the subject's own censoring time (`vpc_tte`'s rule).
            censor_at: (cause == 0).then_some(time),
            stratum,
        })
        .collect();
    Ok((rows, causes, values, labels))
}

/// One stratum of one replicate: each subject's post-censored all-cause time and cause.
#[derive(Debug, Clone, Default)]
struct CauseStratum {
    times: Vec<f64>,
    causes: Vec<usize>,
}

/// Simulate the replicates to `horizon` and collect, per replicate and stratum, each
/// subject's first event time and its cause, censored under `opts.censoring`. The
/// simulator pushes every cause of a subject at the winning time and marks the winner
/// observed, so the records of one `(sim, subject)` give `(t, cause)`. The censoring
/// uniforms are keyed by `(sim.seed, sim, row index)`, as [`vpc_tte`](super::vpc_tte)'s.
#[allow(clippy::type_complexity)]
fn simulate_cause_replicates(
    model: &CompiledModel, population: &Population, params: &ModelParameters, opts: &VpcCifOptions,
    cmts: &[usize], rows: &[ObsRow], n_strata: usize, horizon: f64,
) -> Result<(Vec<Vec<CauseStratum>>, Vec<String>), EngineError> {
    let mut sim_opts = opts.sim.clone();
    sim_opts.horizon = Some(horizon);
    let out = simulate_with_options_impl(model, population, params, opts.n_sim, &sim_opts, false)?;

    let by_id: HashMap<&str, usize> =
        rows.iter().enumerate().map(|(i, r)| (r.id.as_str(), i)).collect();
    // drawn[sim - 1][row] = (time, cause) of that replicate subject.
    let mut drawn: Vec<Vec<Option<(f64, usize)>>> = vec![vec![None; rows.len()]; opts.n_sim];
    for SimulationResult { sim, id, cmt, outcome, .. } in &out.results {
        let SimOutcome::Event { time, observed } = outcome else {
            continue;
        };
        let (Some(k), Some(&i)) = (cmts.iter().position(|c| c == cmt), by_id.get(id.as_str()))
        else {
            continue;
        };
        let slot = drawn[sim - 1][i].get_or_insert((*time, 0));
        if *observed {
            *slot = (*time, k + 1);
        }
    }

    let censor = ReplicateCensor::new(
        opts.censoring,
        TteCensorRule::KeepObserved,
        opts.sim.seed,
        rows,
        n_strata,
        horizon,
    );
    let mut reps: Vec<Vec<CauseStratum>> =
        vec![vec![CauseStratum::default(); n_strata]; opts.n_sim];
    for (s, per_row) in drawn.iter().enumerate() {
        for (i, d) in per_row.iter().enumerate() {
            let Some((time, cause)) = *d else {
                continue;
            };
            let row = &rows[i];
            let (t, e) = censor.apply(s + 1, i, row, time, cause > 0);
            let slot = &mut reps[s][row.stratum];
            slot.times.push(t);
            slot.causes.push(if e { cause } else { 0 });
        }
    }
    Ok((reps, out.warnings))
}

fn on_grid(band: VpcBand, obs_on_grid: Vec<Option<f64>>) -> CifVpcBand {
    CifVpcBand { obs_on_grid, lo: band.lo, med: band.med, hi: band.hi, n_used: band.n_used }
}

/// A cumulative-incidence VPC for competing risks: every TTE endpoint of the model is a
/// cause.
///
/// Simulates `opts.n_sim` replicates of `population` under `params` with one simulator
/// call, then builds per stratum the observed Aalen–Johansen cumulative incidence of each
/// cause and the observed all-cause Kaplan–Meier curve, each with the pointwise
/// `opts.probs` band of the replicate curves on the grid. The module doc explains the
/// rules; [`crate::stats::cif`] the estimator.
///
/// # Errors
///
/// `E_VPC_TTE_INPUT` for an input the curves would be wrong on: no TTE endpoint, a
/// repeated-events endpoint, a subject with records on some causes but not all, more
/// than one record on a cause, cause records at different times, events on two causes,
/// left truncation, an interval-censored record, a duplicated subject id, no record on
/// any cause, bad `probs`, too few `n_sim`, a grid time that is not finite and
/// non-negative, a stratification covariate a subject lacks or with more than
/// [`VPC_TTE_MAX_STRATA`](super::VPC_TTE_MAX_STRATA) values. Otherwise any `Err` the
/// simulator returns.
pub fn vpc_cif(
    model: &CompiledModel, population: &Population, params: &ModelParameters, opts: &VpcCifOptions,
) -> Result<CifVpc, EngineError> {
    let probs = opts.probs;
    check_band_options(WHO, probs, opts.n_sim, &opts.grid)?;
    let cmts = resolve_causes(model)?;
    let (mut rows, causes, values, labels) =
        observed_cause_rows(population, &cmts, &opts.stratify)?;
    // The data on [0, H] only: a row cut at H is a censoring there, so its cause is read
    // through `r.event` below.
    let horizon = horizon_and_cut(&mut rows, opts.sim.horizon);

    let n_strata = values.len();
    let (reps, warnings) = simulate_cause_replicates(
        model, population, params, opts, &cmts, &rows, n_strata, horizon,
    )?;

    let k_causes = cmts.len();
    let err = |e: &dyn std::fmt::Display| refuse(format!("{WHO}: {e}"));
    let mut strata = Vec::with_capacity(n_strata);
    for (s, (value, label)) in values.into_iter().zip(labels).enumerate() {
        let (ot, oc): (Vec<f64>, Vec<usize>) = rows
            .iter()
            .zip(&causes)
            .filter(|(r, _)| r.stratum == s)
            .map(|(r, &c)| (r.time, if r.event { c } else { 0 }))
            .unzip();
        let observed = aalen_johansen(&ot, &oc, k_causes).map_err(|e| err(&e))?;
        let curves = reps
            .iter()
            .map(|r| aalen_johansen(&r[s].times, &r[s].causes, k_causes))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| err(&e))?;
        let grid = stratum_grid(
            &opts.grid,
            horizon,
            &observed.all_cause.time,
            curves.iter().map(|c| c.all_cause.time.as_slice()),
        );
        let surv = band_over(curves.len(), |j, t| curves[j].all_cause.at(t), &grid, probs)
            .map_err(|e| err(&e))?;
        let survival = on_grid(surv, grid.iter().map(|&t| observed.all_cause.at(t)).collect());
        let mut cause_bands = Vec::with_capacity(k_causes);
        for (k, &cmt) in cmts.iter().enumerate() {
            let band = vpc_band_cif(&curves, k, &grid, probs).map_err(|e| err(&e))?;
            let obs = grid.iter().map(|&t| observed.at(k, t)).collect();
            cause_bands.push(CifCauseBand { cmt, band: on_grid(band, obs) });
        }
        strata.push(CifVpcStratum {
            label,
            value,
            n_subjects: ot.len(),
            observed,
            grid,
            survival,
            causes: cause_bands,
        });
    }
    Ok(CifVpc {
        cmts,
        horizon,
        n_sim: opts.n_sim,
        probs,
        censoring: opts.censoring,
        strata,
        warnings,
    })
}

#[cfg(test)]
#[path = "tests/vpc_cif_tests.rs"]
mod tests;
