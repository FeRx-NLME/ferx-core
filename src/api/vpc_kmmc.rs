//! Kaplan–Meier mean covariate (KMMC) visual predictive check for a single-event TTE
//! endpoint (#1886, Hooker & Karlsson, PAGE 21 (2012) Abstr 2564).
//!
//! At each step of the Kaplan–Meier curve, [`vpc_kmmc`] takes the mean of one baseline
//! covariate over the subjects still at risk, in the data and in each of `n_sim`
//! replicates of the observed design, and returns the observed curve beside the
//! pointwise band of the replicates, per stratum. A covariate that drives the hazard
//! makes the running mean drift as high-hazard subjects leave the risk set; a model
//! that lacks the effect drifts too little, and the observed curve leaves the band. It
//! never calls `fit()`.
//!
//! **Every replicate keeps the data's covariates.** The replicates reuse the observed
//! population, so the curves differ only in **who stays at risk**. At `t = 0` everyone
//! is at risk everywhere, the observed value and the whole band are the population
//! mean, and the band has width 0; the signal is the gap that opens over time.
//!
//! **Censoring differs from [`vpc_tte`](super::vpc_tte)'s.** Under
//! [`VpcCensoring::Observed`] every replicate subject draws its censoring time from its
//! own stratum's reverse Kaplan–Meier censoring distribution `G`, unconditionally. The
//! KM-VPC instead lets a subject observed censored keep its own time. For a Kaplan–Meier
//! curve both are fine, but for the mean covariate the kept times over-cover: censored
//! subjects are selected for a low hazard (`T > c`), so keeping their exact `cᵢ`
//! removes the same covariate values from the risk set at the same times in every
//! replicate as in the data, and the band hugs the observed curve. Measured on 200
//! datasets × 200 replicates with a 0.891 nominal: 0.950 / 0.955 coverage at H/2 and
//! 3H/4 with kept times, 0.89 / 0.925 with the unconditional draw (#1886 PR3 plan M5).
//! The draw assumes what `G` already assumes: censoring is independent of the
//! covariate within a stratum, so stratify on what drives dropout.

use super::vpc_tte::{
    check_band_options, observed_rows, refuse, resolve_cmt, simulate_replicates, stratum_grid,
    Entry, TteCensorRule, VpcCensoring, VpcStratify, VpcTteOptions,
};
use crate::diagnostics::EngineError;
use crate::stats::km::{kaplan_meier_mean_covariate, vpc_band_kmmc, KmmcCurve};
use crate::types::{CompiledModel, ModelParameters, Population, Subject};
use std::collections::HashMap;

const WHO: Entry = Entry::Kmmc;

/// Options for [`vpc_kmmc`]. Build with [`VpcKmmcOptions::new`] and set fields after.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VpcKmmcOptions {
    /// The Kaplan–Meier VPC options: replicates, band probabilities, grid, censoring
    /// (see the module doc for how [`VpcCensoring::Observed`] differs here),
    /// stratification, endpoint and simulator options.
    pub vpc: VpcTteOptions,
    /// The covariate whose mean is taken: each subject's baseline value
    /// (`Subject::covariates`, its first non-missing value).
    pub covariate: String,
    /// `None`: the arithmetic mean (for a 0/1 covariate, the share coded 1).
    /// `Some(v)`: the share of at-risk subjects whose value equals `v` exactly, for a
    /// multi-level code where a mean of codes means nothing.
    pub level: Option<f64>,
    /// A curve is undefined where fewer subjects than this are at risk; applied to
    /// the observed curve and to each replicate on its own. Default 10.
    pub min_at_risk: usize,
}

impl VpcKmmcOptions {
    /// The defaults for `covariate`: [`VpcTteOptions::default`], the mean
    /// (`level: None`) and `min_at_risk: 10`.
    pub fn new(covariate: impl Into<String>) -> Self {
        VpcKmmcOptions {
            vpc: VpcTteOptions::default(),
            covariate: covariate.into(),
            level: None,
            min_at_risk: 10,
        }
    }
}

impl Default for VpcKmmcOptions {
    /// [`VpcKmmcOptions::new`] with no covariate named: [`vpc_kmmc`] refuses it until
    /// `covariate` is set. The struct is `#[non_exhaustive]`, so outside this crate it is
    /// built from `new` or `default()` and its fields are then assigned.
    fn default() -> Self {
        VpcKmmcOptions::new(String::new())
    }
}

/// One stratum of a [`KmmcVpc`]. `grid`, `obs_on_grid`, `lo`, `med`, `hi` and
/// `n_used` are parallel. A band entry is `None` where fewer than
/// `min_curves_for(probs)` replicates are defined (at least `min_at_risk` subjects at
/// risk, and not past their last time).
#[derive(Debug, Clone, PartialEq)]
pub struct KmmcVpcStratum {
    /// `all`, `NAME=value` or `dose=value`.
    pub label: String,
    /// The stratum's covariate or dose value; `None` for [`VpcStratify::None`].
    pub value: Option<f64>,
    pub n_subjects: usize,
    /// The observed KMMC curve, one row per row of the stratum's Kaplan–Meier curve.
    pub observed: KmmcCurve,
    pub grid: Vec<f64>,
    /// The observed curve on the grid, `KmmcCurve::at(t, min_at_risk)`.
    pub obs_on_grid: Vec<Option<f64>>,
    pub lo: Vec<Option<f64>>,
    pub med: Vec<Option<f64>>,
    pub hi: Vec<Option<f64>>,
    /// How many replicate curves are defined at each grid time.
    pub n_used: Vec<usize>,
}

/// The result of [`vpc_kmmc`].
#[derive(Debug, Clone, PartialEq)]
pub struct KmmcVpc {
    pub cmt: usize,
    pub covariate: String,
    pub level: Option<f64>,
    /// `H`, the horizon the replicates were simulated to.
    pub horizon: f64,
    pub n_sim: usize,
    pub probs: [f64; 3],
    pub min_at_risk: usize,
    pub censoring: VpcCensoring,
    pub strata: Vec<KmmcVpcStratum>,
    /// `W_VPC_KMMC_TIME_VARYING_COVARIATE` when the covariate changes within a subject,
    /// then the simulator's warnings.
    pub warnings: Vec<String>,
}

/// Each observed row's covariate value, with the refusals for a subject that lacks
/// the covariate and for a missing (non-finite) baseline.
fn row_covariates(
    rows_ids: &[&str],
    by_id: &HashMap<&str, &Subject>,
    name: &str,
) -> Result<Vec<f64>, EngineError> {
    let mut values = Vec::with_capacity(rows_ids.len());
    let mut missing: Vec<&str> = Vec::new();
    for &id in rows_ids {
        match by_id[id].covariates.get(name) {
            None => {
                return Err(refuse(format!(
                    "{WHO}: subject {id} has no covariate `{name}` to take the mean of."
                )))
            }
            Some(v) if !v.is_finite() => missing.push(id),
            Some(v) => values.push(*v),
        }
    }
    if let Some(first) = missing.first() {
        return Err(refuse(format!(
            "{WHO}: covariate `{name}` is missing (every value blank or not finite) in {} \
             subjects, first {first}. The mean over a risk set needs a value for every \
             subject in it.",
            missing.len()
        )));
    }
    Ok(values)
}

/// The value the curve averages: the covariate itself, or 1/0 for being at `level`.
fn curve_values(cov: &[f64], name: &str, level: Option<f64>) -> Result<Vec<f64>, EngineError> {
    let Some(lv) = level else {
        return Ok(cov.to_vec());
    };
    if !cov.iter().any(|v| v.to_bits() == lv.to_bits()) {
        return Err(refuse(format!(
            "{WHO}: no subject on the endpoint has covariate `{name}` = {lv}, the `level` \
             asked for."
        )));
    }
    Ok(cov
        .iter()
        .map(|v| {
            if v.to_bits() == lv.to_bits() {
                1.0
            } else {
                0.0
            }
        })
        .collect())
}

/// `W_VPC_KMMC_TIME_VARYING_COVARIATE` when `name` changes within any of `subjects`.
fn time_varying_warning<'a>(
    subjects: impl Iterator<Item = &'a Subject>,
    name: &str,
) -> Option<String> {
    let n = subjects
        .filter(|s| s.time_varying_covariate_names().iter().any(|c| c == name))
        .count();
    (n > 0).then(|| {
        format!(
            "W_VPC_KMMC_TIME_VARYING_COVARIATE: covariate `{name}` changes over time in {n} \
             subjects. The Kaplan–Meier mean covariate uses each subject's first value, not \
             its value at the risk time."
        )
    })
}

/// A Kaplan–Meier mean covariate VPC for one single-event TTE endpoint.
///
/// Simulates `opts.vpc.n_sim` replicates of `population` under `params` with one
/// simulator call, then builds, per stratum, the observed KMMC curve of
/// `opts.covariate` and the pointwise `opts.vpc.probs` band of the replicate curves on
/// the grid. The module doc explains the replicate censoring;
/// [`crate::stats::km::kaplan_meier_mean_covariate`] the risk set and ties.
///
/// # Errors
///
/// `E_VPC_TTE_INPUT` for every input [`vpc_tte`](super::vpc_tte) refuses, and also: an
/// empty `covariate`, a repeated-events endpoint (KMMC by event order is not supported yet), a subject on
/// the endpoint that lacks the covariate or whose baseline is missing, a `level` no
/// such subject has, and stratifying on the covariate itself. Otherwise any `Err` the
/// simulator returns.
pub fn vpc_kmmc(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    opts: &VpcKmmcOptions,
) -> Result<KmmcVpc, EngineError> {
    let vpc = &opts.vpc;
    let name = opts.covariate.as_str();
    let probs = vpc.probs;
    check_band_options(WHO, probs, vpc.n_sim, &vpc.grid)?;
    if name.is_empty() {
        return Err(refuse(format!(
            "{WHO}: `covariate` is empty. Name the covariate whose mean to take, e.g. \
             `VpcKmmcOptions::new(\"WT\")`."
        )));
    }
    if vpc.stratify == VpcStratify::Covariate(opts.covariate.clone()) {
        return Err(refuse(format!(
            "{WHO}: stratifying on `{name}`, the covariate whose mean is taken, makes the \
             mean constant within each stratum. Stratify on another covariate, or not at \
             all."
        )));
    }
    let cmt = resolve_cmt(WHO, model, vpc.cmt)?;
    let (rows, values, labels) = observed_rows(WHO, population, cmt, &vpc.stratify)?;

    let by_id: HashMap<&str, &Subject> = population
        .subjects
        .iter()
        .map(|s| (s.id.as_str(), s))
        .collect();
    let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    let cov = row_covariates(&ids, &by_id, name)?;
    let vals = curve_values(&cov, name, opts.level)?;
    let mut warnings: Vec<String> = time_varying_warning(ids.iter().map(|id| by_id[id]), name)
        .into_iter()
        .collect();

    let horizon = vpc.sim.horizon.unwrap_or_else(|| {
        rows.iter()
            .map(|r| r.time)
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let n_strata = values.len();
    let (reps, sim_warnings) = simulate_replicates(
        model,
        population,
        params,
        vpc,
        cmt,
        &rows,
        n_strata,
        horizon,
        TteCensorRule::DrawAll,
    )?;
    warnings.extend(sim_warnings);

    let mut strata = Vec::with_capacity(n_strata);
    for (k, (value, label)) in values.into_iter().zip(labels).enumerate() {
        let mut ot = Vec::new();
        let mut oe = Vec::new();
        let mut ov = Vec::new();
        for (i, r) in rows.iter().enumerate().filter(|(_, r)| r.stratum == k) {
            ot.push(r.time);
            oe.push(r.event);
            ov.push(vals[i]);
        }
        let observed = kaplan_meier_mean_covariate(&ot, &oe, &ov)
            .map_err(|e| refuse(format!("{WHO}: {e}")))?;
        let curves = reps
            .iter()
            .map(|r| {
                let s = &r[k];
                let v: Vec<f64> = s.rows.iter().map(|&i| vals[i]).collect();
                kaplan_meier_mean_covariate(&s.times, &s.events, &v)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| refuse(format!("{WHO}: a replicate {e}")))?;
        let grid = stratum_grid(
            &vpc.grid,
            horizon,
            &observed.time,
            curves.iter().map(|c| c.time.as_slice()),
        );
        let band = vpc_band_kmmc(&curves, &grid, probs, opts.min_at_risk)
            .map_err(|e| refuse(format!("{WHO}: {e}")))?;
        strata.push(KmmcVpcStratum {
            label,
            value,
            n_subjects: ot.len(),
            obs_on_grid: grid
                .iter()
                .map(|&t| observed.at(t, opts.min_at_risk))
                .collect(),
            observed,
            grid,
            lo: band.lo,
            med: band.med,
            hi: band.hi,
            n_used: band.n_used,
        });
    }
    Ok(KmmcVpc {
        cmt,
        covariate: opts.covariate.clone(),
        level: opts.level,
        horizon,
        n_sim: vpc.n_sim,
        probs,
        min_at_risk: opts.min_at_risk,
        censoring: vpc.censoring,
        strata,
        warnings,
    })
}

#[cfg(test)]
#[path = "tests/vpc_kmmc_tests.rs"]
mod tests;
