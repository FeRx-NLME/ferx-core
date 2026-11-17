//! Count endpoints — Poisson and negative binomial (`[count_model]`, #2877).
//!
//! A count DV `k ∈ {0, 1, 2, …}` is observed via [`ObsRecord::Count`] and scored per record
//! against a law with mean `λ(θ, η, covariates, TIME)`:
//!
//! ```text
//! Poisson:   −log P(k) = λ − k·log λ + ln Γ(k + 1)
//! NegBin:    −log P(k) = −[ln Γ(k + r) − ln Γ(r) − ln Γ(k + 1)
//!                          + r·log(r/(r+λ)) + k·log(λ/(r+λ))]      (var = λ + λ²/r)
//! ```
//!
//! The constant `ln Γ(k + 1)` is kept: NONMEM's `F_FLAG = 1` objective `−2·log PMF` carries
//! it, and leaving it out would shift the OFV by `2·Σ ln k!`.
//!
//! The mean is written either as `lambda = …` (the mean itself, which must evaluate to a
//! finite positive number) or `log_lambda = …` (its log, any finite number); the size `r` of
//! a negative binomial is a second expression. Everything a record needs is evaluated in one
//! place, [`for_each_count_param`], which the likelihood, the sampler, the predictor and the
//! diagnostics all walk, as the binary endpoint's `for_each_binary_lp` is shared. The record
//! kernel [`count_record_nll_g`] is written once over any [`PkNum`]: the `f64` value path and
//! the discrete term's compiled plan (the exact inner / outer derivatives) run the same code.

use crate::parser::model_parser::with_model_time;
use crate::sens::num::{BranchCmp, PkNum};
use crate::stats::likelihood::SUBJECT_SENTINEL_NLL;
use crate::stats::special::ln_gamma;
use crate::types::{
    CompiledModel, CountFamily, CountMeanScale, CountParamFn, DiscreteObsDiagnostic,
    EndpointLikelihood, EndpointPredictionResult, ObsRecord, Prediction, Subject,
};
use std::collections::HashMap;

/// Largest count whose negative-binomial `ln Γ(k + r) − ln Γ(r)` is summed term by term in
/// [`count_record_nll_g`] (`k` logs per record); a larger count takes the `ln Γ` difference.
const NEGBIN_SUM_MAX_K: u32 = 256;

/// One count record's negative log-likelihood, over any [`PkNum`] — `None` when the law is
/// undefined at this point: a mean (or `log λ`) that is not finite, a `lambda =` mean that is
/// not positive, a `log λ` whose exponential overflows, or (negative binomial) a size that is
/// not finite and positive. The value path turns `None` into the `1e20` sentinel; the plan
/// path declines the point (the caller falls back to FD there).
///
/// `size` is read only for [`CountFamily::NegBin`].
pub(crate) fn count_record_nll_g<T: PkNum>(
    family: CountFamily,
    scale: CountMeanScale,
    mean: T,
    size: T,
    k: u32,
) -> Option<T> {
    let inf = f64::INFINITY;
    if !mean.br_abs_lt(inf) {
        return None;
    }
    let (mu, log_mu) = match scale {
        CountMeanScale::Log => {
            let mu = mean.exp();
            if !mu.br_abs_lt(inf) {
                return None;
            }
            (mu, mean)
        }
        CountMeanScale::Linear => {
            if !mean.br_cmp_c(BranchCmp::Gt, 0.0) {
                return None;
            }
            (mean, mean.ln())
        }
    };
    let kf = f64::from(k);
    let ln_k_fact = T::from_f64(ln_gamma(kf + 1.0));
    let k_t = T::from_f64(kf);
    match family {
        CountFamily::Poisson => Some(mu - k_t * log_mu + ln_k_fact),
        CountFamily::NegBin => {
            if !(size.br_abs_lt(inf) && size.br_cmp_c(BranchCmp::Gt, 0.0)) {
                return None;
            }
            let r = size;
            // `ln Γ(k + r) − ln Γ(r)`: for the counts a record carries, the exact finite sum
            // `Σ_{j<k} ln(r + j)`, which does not cancel. The `ln Γ` difference loses
            // `|ln Γ(r)|·1e-13` to cancellation — 2e-3 at `r = 1e9`, where the law is a
            // Poisson to 1e-8 — so it is kept only past `NEGBIN_SUM_MAX_K`.
            let lg = if k <= NEGBIN_SUM_MAX_K {
                (0..k).fold(T::from_f64(0.0), |acc, j| {
                    acc + (r + T::from_f64(f64::from(j))).ln()
                })
            } else {
                (r + k_t).ln_gamma() - r.ln_gamma()
            };
            // `r·log(r/(r+λ)) = −r·log1p(λ/r)`, which keeps its digits as `r` grows.
            let log_p = lg - ln_k_fact - r_log1p_ratio(mu, r) + k_t * (log_mu - (r + mu).ln());
            Some(T::from_f64(0.0) - log_p)
        }
    }
}

/// Below this `x = λ/r` the NB's `r·log1p(x)` is taken from its series rather than `ln(1 + x)`.
const LOG1P_SERIES_MAX: f64 = 1e-3;

/// `r·log1p(λ/r)` for a negative binomial, generic over the number type. `PkNum` has no
/// `log1p`, and `r·ln(1 + λ/r)` loses `λ/r` entirely once it falls below machine epsilon
/// (`λ = 1, r = 1e17` scored 0 instead of 1). For `x = λ/r < 1e-3` it is
/// `λ·(1 − x/2 + x²/3 − x³/4 + x⁴/5 − x⁵/6)`, whose dropped `x⁶/7` term is below 1.5e-19
/// relative; each derivative the dual carries is that of a polynomial, so it is exact too.
/// Above it, `λ·ln(u)/(u − 1)` with `u = 1 + x`, which is `log1p` to a few ulps.
fn r_log1p_ratio<T: PkNum>(mu: T, r: T) -> T {
    let x = mu / r;
    if x.br_cmp_c(BranchCmp::Lt, LOG1P_SERIES_MAX) {
        let c = |v: f64| T::from_f64(v);
        let poly = c(1.0)
            - x * (c(0.5) - x * (c(1.0 / 3.0) - x * (c(0.25) - x * (c(0.2) - x * c(1.0 / 6.0)))));
        mu * poly
    } else {
        // `r·ln(u)` with `u = 1 + x` carries the rounding of `u`, `eps/x` relative; the
        // ratio `ln(u)/(u − 1)` cancels it (Goldberg), and `r·x = λ`. `u − 1 ≥ 1e-3` here.
        let u = T::from_f64(1.0) + x;
        mu * u.ln() / (u - T::from_f64(1.0))
    }
}

/// `(mean λ, size)` on the natural scale from a record's raw `[mean, size]`: `None` when the
/// law is undefined there (the same points [`count_record_nll_g`] declines). `size` is `None`
/// for a Poisson endpoint.
fn natural_params(
    family: CountFamily,
    scale: CountMeanScale,
    raw: [f64; 2],
) -> Option<(f64, Option<f64>)> {
    let lambda = match scale {
        CountMeanScale::Linear => raw[0],
        CountMeanScale::Log => raw[0].exp(),
    };
    let ok = |x: f64| x.is_finite() && (x > 0.0 || (scale == CountMeanScale::Log && x >= 0.0));
    if !raw[0].is_finite() || !ok(lambda) {
        return None;
    }
    match family {
        CountFamily::Poisson => Some((lambda, None)),
        CountFamily::NegBin => {
            let r = raw[1];
            (r.is_finite() && r > 0.0).then_some((lambda, Some(r)))
        }
    }
}

/// The count endpoints of `model`, ascending CMT (`endpoints` is a `BTreeMap`), as
/// `(cmt, family, scale, param_fn)`. The fixed order is what keeps a multi-endpoint OFV sum
/// and the simulation RNG stream reproducible.
fn count_endpoints(
    model: &CompiledModel,
) -> impl Iterator<Item = (usize, CountFamily, CountMeanScale, &CountParamFn)> {
    model.endpoints.iter().filter_map(|(cmt, ep)| match ep {
        EndpointLikelihood::Count {
            family,
            scale,
            param_fn,
            ..
        } => Some((*cmt, *family, *scale, param_fn)),
        _ => None,
    })
}

/// One count record as seen by [`for_each_count_param`].
struct CountRecordEval<'a> {
    /// Internal (occasion-shifted) clock, for any further evaluation on this record.
    model_time: f64,
    /// The covariate map the parameters were evaluated with.
    covariates: &'a HashMap<String, f64>,
    /// The user clock (input TIME), for every reported row.
    raw_time: f64,
    count: u32,
    /// `[mean, size]` on the endpoint's scale (see [`CountParamFn`]).
    raw: [f64; 2],
}

/// Evaluate the endpoint's parameters once per count record on `cmt`. **Single owner** of the
/// record-selection, time-scoping and covariate-snapshot rule for the count family: the
/// likelihood, the sampler, the predictor and the diagnostics all walk records through here,
/// so they cannot disagree about which rows belong to the endpoint or what `λ` a row carries.
fn for_each_count_param<F>(
    param_fn: &CountParamFn,
    cmt: usize,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    mut f: F,
) where
    F: FnMut(CountRecordEval<'_>),
{
    let snaps = subject.record_covariate_snapshots();
    for (idx, r) in subject.obs_records.iter().enumerate() {
        let ObsRecord::Count {
            time,
            raw_time,
            count,
            cmt: c,
        } = r
        else {
            continue;
        };
        if *c != cmt {
            continue;
        }
        let covariates = subject.record_cov(snaps, idx);
        let raw = with_model_time(*time, || param_fn(theta, eta, covariates));
        f(CountRecordEval {
            model_time: *time,
            covariates,
            raw_time: *raw_time,
            count: *count,
            raw,
        });
    }
}

/// Negative log-likelihood of one subject's records on one count endpoint. A record whose law
/// is undefined (see [`count_record_nll_g`]) adds the `1e20` sentinel, as an overflowed binary
/// predictor does, so the optimizer is repelled rather than handed a `NaN`.
fn count_data_term(
    family: CountFamily,
    scale: CountMeanScale,
    param_fn: &CountParamFn,
    cmt: usize,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    let mut nll = 0.0;
    for_each_count_param(param_fn, cmt, subject, theta, eta, |r| {
        nll += count_record_nll_g(family, scale, r.raw[0], r.raw[1], r.count)
            .filter(|v| v.is_finite())
            .unwrap_or(SUBJECT_SENTINEL_NLL);
    });
    nll
}

/// The count half of [`super::discrete_subject_nll`]: every count endpoint's data term,
/// summed in ascending CMT order. `0.0` without a count endpoint.
pub(crate) fn count_subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    let mut nll = 0.0;
    for (cmt, family, scale, param_fn) in count_endpoints(model) {
        nll += count_data_term(family, scale, param_fn, cmt, subject, theta, eta);
    }
    nll
}

/// The count endpoints' share of [`super::DiscreteTerm`]: the same sum as
/// [`count_subject_nll`] on any VM number type, through each endpoint's `param_program` and
/// the same record kernel. `None` — the caller falls back to FD for the point — when an
/// endpoint has no program or a record's law is undefined.
pub(crate) fn count_term_eval<T: crate::sens::num::vm::VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    th: &[T],
    et: &[T],
) -> Option<T> {
    let mut nll = T::from_f64(0.0);
    let snaps = subject.record_covariate_snapshots();
    for (cmt, ep) in &model.endpoints {
        let EndpointLikelihood::Count {
            family,
            scale,
            param_program,
            ..
        } = ep
        else {
            continue;
        };
        let prog = param_program.as_ref()?;
        for (idx, r) in subject.obs_records.iter().enumerate() {
            let ObsRecord::Count {
                time,
                count,
                cmt: c,
                ..
            } = r
            else {
                continue;
            };
            if c != cmt {
                continue;
            }
            let cov = subject.record_cov(snaps, idx);
            let outs = with_model_time(*time, || prog.eval_g(th, et, cov))?;
            let mut outs = outs.into_iter();
            let mean = outs.next()?;
            let size = outs.next().unwrap_or_else(|| T::from_f64(0.0));
            nll = nll + count_record_nll_g(*family, *scale, mean, size, *count)?;
        }
    }
    Some(nll)
}

/// Whether every count endpoint carries a dual-evaluable program (vacuously `true` with
/// none) — the count share of `all_discrete_endpoints_analytic`.
pub(crate) fn all_count_endpoints_analytic(model: &CompiledModel) -> bool {
    model.endpoints.values().all(|ep| match ep {
        EndpointLikelihood::Count { param_program, .. } => param_program.is_some(),
        _ => true,
    })
}

/// One draw from the endpoint's law at `(lambda, size)`: Poisson, or the negative binomial as
/// its gamma–Poisson mixture (`λ' ~ Gamma(shape r, scale λ/r)`, then `Poisson(λ')`). `None`
/// when the draw does not fit a `u32` count.
fn draw_count<R: rand::Rng>(lambda: f64, size: Option<f64>, rng: &mut R) -> Option<u32> {
    use rand_distr::Distribution;
    if lambda <= 0.0 {
        // The zero-mean limit (a `log_lambda` that underflows): zero with probability one,
        // for either family. Checked first, since `Gamma::new(r, 0)` refuses a zero scale.
        return Some(0);
    }
    let rate = match size {
        None => lambda,
        Some(r) => rand_distr::Gamma::new(r, lambda / r).ok()?.sample(rng),
    };
    if rate <= 0.0 {
        // A zero mean draws zero with probability one (`Poisson::new` refuses it).
        return Some(0);
    }
    let k: f64 = rand_distr::Poisson::new(rate).ok()?.sample(rng);
    (k.is_finite() && k >= 0.0 && k <= f64::from(u32::MAX)).then_some(k as u32)
}

/// Draw a simulated count for every record of every count endpoint, one
/// [`SimOutcome::Count`](crate::types::SimOutcome::Count) row per record — the count analogue
/// of [`super::simulate_binary`], on the same RNG stream, right after it. `ipred` carries the
/// record's mean `λ`. A record whose law is undefined (a non-positive or non-finite mean or
/// size) or whose draw overflows a `u32` is skipped and counted in one warning per subject
/// and endpoint, never emitted as an arbitrary value.
#[allow(clippy::too_many_arguments)]
pub fn simulate_count<R: rand::Rng>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    draw: usize,
    sim: usize,
    rng: &mut R,
    results: &mut Vec<crate::api::SimulationResult>,
    warnings: &mut Vec<String>,
) {
    if subject.obs_records.is_empty() {
        return;
    }
    for (cmt, family, scale, param_fn) in count_endpoints(model) {
        let mut n_skipped = 0usize;
        for_each_count_param(param_fn, cmt, subject, theta, eta, |r| {
            let Some((lambda, size)) = natural_params(family, scale, r.raw) else {
                n_skipped += 1;
                return;
            };
            let Some(count) = draw_count(lambda, size, rng) else {
                n_skipped += 1;
                return;
            };
            results.push(crate::api::SimulationResult {
                draw,
                sim,
                id: subject.id.clone(),
                time: r.raw_time,
                cmt,
                ipred: lambda,
                outcome: crate::types::SimOutcome::Count { count },
                mixnum: None,
            });
        });
        if n_skipped > 0 {
            warnings.push(format!(
                "[count_model] cmt = {cmt}: subject {} has {n_skipped} record(s) with no count \
                 law to draw from, so they carry no simulated outcome. The mean must be finite \
                 and positive (and the negative-binomial size `r` finite and positive) at every \
                 record. Check the `lambda` / `log_lambda` and `r` expressions and the \
                 subject's covariates.",
                subject.id
            ));
        }
    }
}

/// `Prediction::Rate` for every count record, at the supplied `eta` (the caller's EBE, or
/// zeros for a population prediction). A record whose law is undefined predicts
/// `lambda = NaN`, which is what it is.
pub fn predict_count(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    results: &mut Vec<EndpointPredictionResult>,
) {
    if subject.obs_records.is_empty() {
        return;
    }
    for (cmt, family, scale, param_fn) in count_endpoints(model) {
        for_each_count_param(param_fn, cmt, subject, theta, eta, |r| {
            let (lambda, size) = natural_params(family, scale, r.raw).unwrap_or((
                f64::NAN,
                (family == CountFamily::NegBin).then_some(f64::NAN),
            ));
            results.push(EndpointPredictionResult {
                id: subject.id.clone(),
                cmt,
                time: r.raw_time,
                observed: Some(f64::from(r.count)),
                prediction: Prediction::Rate { lambda, size },
            });
        });
    }
}

/// Pearson residual of a count `k` against its law: `(k − λ)/√Var`, `Var = λ` (Poisson) or
/// `λ + λ²/r` (negative binomial). `NaN` where the variance is not positive and finite.
pub fn count_pearson_residual(k: f64, lambda: f64, size: Option<f64>) -> f64 {
    let var = match size {
        None => lambda,
        Some(r) => lambda + lambda * lambda / r,
    };
    if !(var > 0.0 && var.is_finite()) {
        return f64::NAN;
    }
    (k - lambda) / var.sqrt()
}

/// Post-fit per-record diagnostics for one subject's count endpoints: `PRED` the mean at
/// `η = 0`, `IPRED` the mean at the subject's EBE, `IWRES` the Pearson residual at the EBE
/// ([`count_pearson_residual`]). Empty without count records.
pub fn count_diagnostics(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Vec<DiscreteObsDiagnostic> {
    let mut rows = Vec::new();
    if subject.obs_records.is_empty() {
        return rows;
    }
    let zero_eta = vec![0.0_f64; eta.len()];
    for (cmt, family, scale, param_fn) in count_endpoints(model) {
        for_each_count_param(param_fn, cmt, subject, theta, eta, |r| {
            let ind = natural_params(family, scale, r.raw);
            let pop = with_model_time(r.model_time, || {
                natural_params(family, scale, param_fn(theta, &zero_eta, r.covariates))
            });
            let k = f64::from(r.count);
            let (ipred, iwres) = ind.map_or((f64::NAN, f64::NAN), |(l, s)| {
                (l, count_pearson_residual(k, l, s))
            });
            rows.push(DiscreteObsDiagnostic {
                cmt,
                time: r.raw_time,
                dv: k,
                pred: pop.map_or(f64::NAN, |(l, _)| l),
                ipred,
                iwres,
            });
        });
    }
    rows
}

#[cfg(test)]
#[path = "count_tests.rs"]
mod tests;
