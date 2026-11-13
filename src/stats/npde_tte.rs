//! npde for single-event time-to-event records (#1889).
//!
//! A TTE record is scored against the model's **population** predictive distribution
//! of the event time. With `F̂(y)` the fraction of `K` simulated replicates of the
//! subject that have an event strictly before `y`:
//!
//! - an observed event at `t` has `pd = F̂(t)`;
//! - a time right-censored at `c` has `pd` drawn uniformly on `[F̂(c), 1]`: all the
//!   data say is that the event lies past `c`, so the score is imputed from the part
//!   of the predictive distribution past `c`.
//!
//! `npde = Φ⁻¹(pd)` after the `[1/(2K), 1 − 1/(2K)]` clamp the Gaussian scores use.
//! `pd = P(T_sim < t)`, so an event earlier than the model predicts has a negative
//! npde, the direction a low concentration has.
//!
//! **No simulated time is ranked.** Only the count of replicates with an event before
//! `y` enters, so a replicate censored at the simulation horizon, a cure-fraction draw
//! and a failed solve are all "no event before `y`" and can never put `+∞` or `NaN`
//! into the empirical CDF.
//!
//! **Censoring is imputed on `pd`, not on a time.** The predictive distribution of
//! `T | T > c` is marginal over η: drawing η from its prior and then a time past `c`
//! given that η would skip the weighting of η by `S(c | η)`. Imputing `pd` on
//! `[F̂(c), 1]` needs only `F̂(c)` and is exact for every hazard kind.
//!
//! One event per subject leaves nothing to decorrelate, so NPDE and NPD are the same
//! number here and one `npde` is reported. On a joint PK-TTE subject the Gaussian rows
//! are scored by [`super::npde::compute_npde_npd`] and never see these draws.
//!
//! The reference draws come from [`crate::api::simulate_with_options_diag`] with the
//! horizon at the latest scored time (nothing later is ever needed), in batches of
//! [`BATCH`] replicates whose rows are reduced to integer counts before the next batch
//! runs.

use super::npde::{clamp_prob, effective_seed};
use crate::api::{simulate_with_options_diag, SimulateOptions};
use crate::diagnostics::EngineError;
use crate::sim::adaptive::combine_seed;
use crate::stats::special::normal_inv_cdf;
use crate::types::{
    CompiledModel, EndpointLikelihood, EventType, IovOccasionRule, ModelParameters, ObsRecord,
    Population, SimOutcome, TteRecurrence,
};
use rand::{RngExt, SeedableRng};
use std::collections::HashMap;

/// Replicates simulated per simulator call. A constant, so the scores for a seed do
/// not depend on anything the caller sets but the seed and `nsim`.
pub(crate) const BATCH: usize = 100;

/// Purpose tag of the per-batch simulation seeds.
const BATCH_SALT: u64 = 0x1889_BA7C_0000_0001;
/// Purpose tag of the per-subject imputation uniforms, a stream disjoint from the
/// simulation's so one subject's score does not move when another's record changes.
const IMPUTE_SALT: u64 = 0x1889_1A9B_0000_0002;

/// The npde of one single-event TTE record.
#[derive(Debug, Clone, PartialEq)]
pub struct TteNpde {
    /// CMT of the TTE endpoint.
    pub cmt: usize,
    /// Data TIME of the record as written.
    pub time: f64,
    /// The prediction discrepancy after the `[1/(2K), 1 − 1/(2K)]` clamp: the
    /// fraction of replicates with an event before the record's time for an observed
    /// event, a uniform draw between that fraction and 1 for a right-censored time.
    pub pd: f64,
    /// `Φ⁻¹(pd)`. NPDE and NPD are the same number for a single event.
    pub npde: f64,
    /// True when the record is right-censored, so `pd` was imputed.
    pub imputed: bool,
}

/// The clamped prediction discrepancy of one record: `n_before` of `k` replicates
/// had an event strictly before its time; `u ∈ [0, 1)` is read only when `censored`.
pub(crate) fn tte_pd(n_before: usize, k: usize, censored: bool, u: f64) -> f64 {
    let f = n_before as f64 / k as f64;
    let pd = if censored { f + u * (1.0 - f) } else { f };
    clamp_prob(pd, k)
}

/// The uniform a censored record of subject `index` is imputed with under `seed`.
fn impute_uniform(seed: u64, index: usize) -> f64 {
    rand::rngs::StdRng::seed_from_u64(combine_seed(seed ^ IMPUTE_SALT, index as u64))
        .random::<f64>()
}

/// The simulation seed of batch `batch` under `seed`.
fn batch_seed(seed: u64, batch: usize) -> u64 {
    combine_seed(seed ^ BATCH_SALT, batch as u64)
}

fn refuse(message: String) -> EngineError {
    crate::api::npde_tte_input(message)
}

/// The model's one single-event TTE endpoint.
fn scored_cmt(model: &CompiledModel) -> Result<usize, EngineError> {
    let tte = model.tte_cmts();
    let cmt = match tte.as_slice() {
        [] => {
            return Err(refuse(
                "compute_tte_npde: the model declares no TTE endpoint ([event_model]).".to_string(),
            ))
        }
        [only] => *only,
        several => {
            let list = several
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(refuse(format!(
                "compute_tte_npde: the model has TTE endpoints on cmt {list} (competing risks). \
                 npde under competing risks is not supported yet."
            )));
        }
    };
    if matches!(
        model.endpoints.get(&cmt),
        Some(EndpointLikelihood::Tte {
            recurrence: TteRecurrence::Repeated { .. },
            ..
        })
    ) {
        return Err(refuse(format!(
            "compute_tte_npde: cmt {cmt} is a repeated events (RTTE) endpoint. npde for \
             repeated events is not supported yet."
        )));
    }
    Ok(cmt)
}

/// One subject's scored record: its engine time, its written time and whether it is
/// right-censored. `None` for a subject with no record on the endpoint.
pub(crate) type Scored = Option<(f64, f64, bool)>;

/// Each subject's record on `cmt`, in population order.
pub(crate) fn scored_records(
    population: &Population,
    cmt: usize,
) -> Result<Vec<Scored>, EngineError> {
    let mut seen: HashMap<&str, ()> = HashMap::new();
    let mut out = Vec::with_capacity(population.subjects.len());
    for s in &population.subjects {
        if seen.insert(s.id.as_str(), ()).is_some() {
            return Err(refuse(format!(
                "compute_tte_npde: subject id {} appears more than once in the population. \
                 Simulated draws are matched to subjects by id.",
                s.id
            )));
        }
        let mut on_cmt = Vec::new();
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
            if !(time.is_finite() && *time >= 0.0) {
                return Err(refuse(format!(
                    "compute_tte_npde: subject {} has time {time} on cmt {cmt}, which is not a \
                     finite, non-negative number.",
                    s.id
                )));
            }
            if *entry_time > 0.0 {
                return Err(refuse(format!(
                    "compute_tte_npde: subject {} has entry time {entry_time} on cmt {cmt}: left \
                     truncation (delayed entry) is not supported yet.",
                    s.id
                )));
            }
            let censored = match event_type {
                EventType::Exact => false,
                EventType::RightCensored => true,
                EventType::IntervalCensored { .. } => {
                    return Err(refuse(format!(
                        "compute_tte_npde: subject {} has an interval-censored record on cmt \
                         {cmt}; npde needs an exact or a right-censored time.",
                        s.id
                    )))
                }
            };
            on_cmt.push((*time, s.param_time_at(*time), censored));
        }
        match on_cmt.as_slice() {
            [] => out.push(None),
            [one] => out.push(Some(*one)),
            more => {
                return Err(refuse(format!(
                    "compute_tte_npde: subject {} has {} records on cmt {cmt}. A single-event \
                     endpoint takes one per subject.",
                    s.id,
                    more.len()
                )))
            }
        }
    }
    if out.iter().all(Option::is_none) {
        return Err(refuse(format!(
            "compute_tte_npde: no subject has a record on cmt {cmt}."
        )));
    }
    Ok(out)
}

/// The simulator's warnings over every batch, one string per warning code: the first
/// message of the code, and how many the reference draws raised when more than one.
/// A per-subject simulation warning (`W_TTE_DEGENERATE_HAZARD`, `W_TTE_SOLVE_FAILED`)
/// would otherwise repeat once per batch.
pub(crate) fn fold_warnings(all: Vec<String>, nsim: usize) -> Vec<String> {
    let mut order: Vec<String> = Vec::new();
    let mut first: HashMap<String, (String, usize)> = HashMap::new();
    for w in all {
        let code = w.split(':').next().unwrap_or_default().to_string();
        match first.get_mut(&code) {
            Some(entry) => entry.1 += 1,
            None => {
                order.push(code.clone());
                first.insert(code, (w, 1));
            }
        }
    }
    order
        .into_iter()
        .map(|code| {
            let (w, n) = first
                .remove(&code)
                .expect("every code in `order` was inserted");
            if n == 1 {
                w
            } else {
                format!(
                    "{w} [{n} `{code}` warnings were raised over the {nsim} npde reference \
                     replicates; the first is shown.]"
                )
            }
        })
        .collect()
}

/// Whether a simulated replicate `(time, observed)` had its event strictly before `y`.
/// A replicate with no event (`observed = false`: censored at the horizon, a
/// degenerate hazard draw, a failed solve) is never before anything, whatever time it
/// is reported at.
pub(crate) fn event_before(time: f64, observed: bool, y: f64) -> bool {
    observed && time < y
}

/// Per subject, how many of the `nsim` reference replicates had an event before its
/// scored time, and how many replicates were drawn for it (`nsim` for a scored
/// subject, 0 for one with no record), with every batch's simulator warnings.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reference_counts(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    nsim: usize,
    seed: u64,
    iov_occasion: Option<&IovOccasionRule>,
    cmt: usize,
    records: &[Scored],
) -> Result<(Vec<usize>, Vec<usize>, Vec<String>), EngineError> {
    // The latest scored time: no later draw is ever compared with anything.
    let horizon = records
        .iter()
        .flatten()
        .map(|r| r.0)
        .fold(0.0_f64, |a, b| if b > a { b } else { a });
    let by_id: HashMap<&str, usize> = population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id.as_str(), i))
        .collect();
    let mut n_before = vec![0usize; records.len()];
    let mut n_drawn = vec![0usize; records.len()];
    let mut warnings = Vec::new();
    let mut done = 0;
    let mut batch = 0;
    while done < nsim {
        let n = BATCH.min(nsim - done);
        let opts = SimulateOptions {
            seed: Some(batch_seed(seed, batch)),
            horizon: Some(horizon),
            iov_occasion: iov_occasion.cloned(),
            ..Default::default()
        };
        let out = simulate_with_options_diag(model, population, params, n, &opts)?;
        for row in &out.results {
            let SimOutcome::Event { time, observed } = row.outcome else {
                continue;
            };
            if row.cmt != cmt {
                continue;
            }
            let i = by_id[row.id.as_str()];
            let Some((y, ..)) = records[i] else {
                continue;
            };
            n_drawn[i] += 1;
            if event_before(time, observed, y) {
                n_before[i] += 1;
            }
        }
        warnings.extend(out.warnings);
        done += n;
        batch += 1;
    }
    Ok((n_before, n_drawn, warnings))
}

/// npde of every single-event TTE record of `population` under `params` (#1889).
///
/// Returns one inner `Vec` per subject in population order, holding one [`TteNpde`]
/// per event record of the subject (so one, or none for a subject with no record on
/// the endpoint, such as a PK-only subject of a joint model), and the simulator's
/// warnings folded to one string per warning code. The module doc gives the rule.
///
/// `nsim` is the number of reference replicates `K`; `seed` resolves as
/// [`effective_seed`](super::npde::effective_seed) does, so `None` is reproducible.
/// `iov_occasion` is passed to the simulator (see
/// [`SimulateOptions::iov_occasion`]). A fit with `[fit_options] npde_nsim > 0` stores
/// these scores on [`crate::types::TteObsDiagnostic::npde`]; calling this with the
/// fitted parameters and the fit's recorded `npde_seed` reproduces them bit for bit.
///
/// A replicate whose hazard draw is degenerate or whose event-time solve failed is in
/// the denominator and counts as "no event", as the simulator reports it.
///
/// # Errors
///
/// `E_NPDE_TTE_INPUT` for an input the score would be wrong on: `nsim = 0`, no TTE
/// endpoint, more than one (competing risks), a repeated-events endpoint, left
/// truncation, an interval-censored record, more than one record per subject on the
/// endpoint, a time that is not finite and non-negative, a duplicated subject id, or no
/// record on the endpoint at all. Otherwise any `Err` the simulator returns.
pub fn compute_tte_npde(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    nsim: usize,
    seed: Option<u64>,
    iov_occasion: Option<&IovOccasionRule>,
) -> Result<(Vec<Vec<TteNpde>>, Vec<String>), EngineError> {
    if nsim == 0 {
        return Err(refuse(
            "compute_tte_npde: `nsim` must be at least 1.".to_string(),
        ));
    }
    let cmt = scored_cmt(model)?;
    let records = scored_records(population, cmt)?;
    let seed = effective_seed(seed);
    let (n_before, _, warnings) = reference_counts(
        model,
        population,
        params,
        nsim,
        seed,
        iov_occasion,
        cmt,
        &records,
    )?;

    let scores = records
        .iter()
        .enumerate()
        .map(|(i, r)| match r {
            None => Vec::new(),
            Some((_, written, censored)) => {
                let u = if *censored {
                    impute_uniform(seed, i)
                } else {
                    0.0
                };
                let pd = tte_pd(n_before[i], nsim, *censored, u);
                vec![TteNpde {
                    cmt,
                    time: *written,
                    pd,
                    npde: normal_inv_cdf(pd),
                    imputed: *censored,
                }]
            }
        })
        .collect();
    Ok((scores, fold_warnings(warnings, nsim)))
}

#[cfg(test)]
#[path = "npde_tte_tests.rs"]
mod tests;
