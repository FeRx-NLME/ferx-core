//! Turning a set of replicate fits into bias, standard errors and confidence
//! intervals — the contents of PsN's `bootstrap_results.csv`.

use super::{BootstrapOptions, ReplicateResult};

/// Summary statistics for one estimated parameter.
#[derive(Debug, Clone)]
pub struct ParameterSummary {
    pub name: String,
    /// The base model's estimate on the original dataset.
    pub original: Option<f64>,
    pub mean: f64,
    /// `mean − original`: the bootstrap estimate of bias.
    pub bias: Option<f64>,
    /// Standard deviation of the bootstrap distribution — *this* is the
    /// bootstrap standard error, not any per-replicate covariance-step SE.
    pub standard_error: f64,
    pub median: f64,
    /// Percentile interval of the bootstrap distribution. `None` when there are
    /// too few successful samples to resolve the requested tail (see
    /// [`percentile`]).
    pub ci_percentile: Option<(f64, f64)>,
    /// `original ± z·SE`, i.e. the interval you get by *assuming* normality with
    /// the bootstrap standard deviation. Reported alongside the percentile
    /// interval precisely so the two can be compared: a large disagreement is
    /// the signal that the normal approximation — and hence the `R⁻¹` standard
    /// errors — is not to be trusted for that parameter.
    pub ci_standard_error: Option<(f64, f64)>,
}

/// The whole `bootstrap_results` table.
#[derive(Debug, Clone)]
pub struct BootstrapSummary {
    pub parameters: Vec<ParameterSummary>,
    /// Replicates that produced an estimate at all (the fit returned `Ok`).
    pub n_completed: usize,
    /// Replicates that survived the skip filters and fed the statistics.
    pub n_included: usize,
    /// Why the excluded ones were excluded, as `(reason, count)`.
    pub excluded_by: Vec<(String, usize)>,
    pub confidence_level: f64,
    /// Means of the per-replicate diagnostics, PsN's `diagnostic.means`.
    pub diagnostic_means: Vec<(String, f64)>,
}

/// PsN's percentile estimator: the weighted average at `x_((n+1)p)`.
///
/// Verbatim from the bootstrap user guide:
///
/// ```text
/// n = number of observations + 1
/// p = percentile / 100
/// i = integer part of n*p
/// f = decimal part of n*p
/// percentile = (1-f)*x_i + f*x_(i+1)
/// ```
///
/// Returns `None` when the sample is too small to resolve the tail — when
/// `(n+1)p < 1` there is no `x_i` below the requested quantile and any answer
/// would be an extrapolation past the smallest observation. That one condition
/// reproduces every threshold the guide tabulates: **19** samples for 5–95%,
/// **39** for 2.5–97.5%, **199** for 0.5–99.5%, **1999** for 0.05–99.95%.
///
/// `sorted` must be ascending.
pub fn percentile(sorted: &[f64], pct: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let p = pct / 100.0;
    let n = sorted.len() as f64 + 1.0;
    let np = n * p;
    // Both tails: the upper one runs out of data at `(n+1)(1-p) < 1` by the same
    // argument, which is why PsN quotes a single sample count per interval.
    if np < 1.0 || np > sorted.len() as f64 {
        return None;
    }
    let i = np.floor();
    let f = np - i;
    let idx = i as usize; // 1-based in the formula
    let lo = sorted[idx - 1];
    // `i == n-1` lands exactly on the last element, with `f == 0`.
    let hi = if idx < sorted.len() {
        sorted[idx]
    } else {
        sorted[idx - 1]
    };
    Some((1.0 - f) * lo + f * hi)
}

/// Standard normal quantile — Acklam's rational approximation, relative error
/// below ~1.15e-9 over the whole range.
///
/// Deliberately *not* a bisection on the CDF built from
/// [`ferx_core::stats::special::erf`], which was the first thing tried here: that
/// converges exactly, but to the inverse of an `erf` whose own approximation
/// error is ~1.5e-7, landing `z₀.₉₇₅` at 1.9599628 instead of 1.9599640. Inverting
/// an approximation gives you the approximation's accuracy, not the bisection's.
/// The error is irrelevant to a confidence half-width either way; the point is
/// that a number printed as a normal quantile should be one.
// Acklam's published coefficients, kept at their published precision so they can
// be checked against the source rather than against whatever `f64` happens to
// round them to.
#[allow(clippy::excessive_precision)]
pub fn normal_quantile(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "normal_quantile: p must be in (0,1)");
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.02425;

    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    if p < P_LOW {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - P_LOW {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        // The central branch is odd in `q = p − 0.5`, so symmetric tails come
        // out exactly symmetric rather than to within the approximation.
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

/// Whether a replicate is excluded, and why — `None` if it is included.
///
/// Applied here, at summary time, rather than when the fit finishes: that is
/// what lets `--summarize` recompute the whole table from a stored
/// `raw_results.csv` under different criteria without refitting anything, which
/// is the PsN behaviour and the reason the raw file carries the diagnostics
/// rather than only the survivors.
pub fn exclusion_reason(r: &ReplicateResult, options: &BootstrapOptions) -> Option<&'static str> {
    if r.error.is_some() {
        return Some("fit failed");
    }
    if options.skip_minimization_terminated && !r.converged {
        return Some("minimization terminated");
    }
    if options.skip_estimate_near_boundary && r.estimate_near_boundary {
        return Some("estimate near boundary");
    }
    if options.skip_covariance_step_terminated && !r.covariance_step_successful {
        return Some("covariance step terminated");
    }
    if options.skip_with_covstep_warnings && r.covariance_step_warnings {
        return Some("covariance step warnings");
    }
    None
}

/// Compute the summary table from every replicate result.
///
/// `original` is the base-model fit (replicate 0); it supplies the reference for
/// bias and for the standard-error interval, and is never itself part of the
/// bootstrap distribution.
pub fn summarize(
    names: &[String],
    original: Option<&ReplicateResult>,
    replicates: &[ReplicateResult],
    options: &BootstrapOptions,
) -> BootstrapSummary {
    let n_completed = replicates.iter().filter(|r| r.error.is_none()).count();

    let mut excluded_counts: Vec<(String, usize)> = Vec::new();
    let mut included: Vec<&ReplicateResult> = Vec::new();
    for r in replicates {
        match exclusion_reason(r, options) {
            None => included.push(r),
            Some(reason) => match excluded_counts.iter_mut().find(|(k, _)| k == reason) {
                Some((_, n)) => *n += 1,
                None => excluded_counts.push((reason.to_string(), 1)),
            },
        }
    }

    let alpha = (100.0 - options.confidence_level) / 2.0;
    let z = normal_quantile(1.0 - alpha / 100.0);

    let mut parameters = Vec::with_capacity(names.len());
    for (j, name) in names.iter().enumerate() {
        let values: Vec<f64> = included
            .iter()
            .filter_map(|r| r.estimates.get(j).copied())
            .filter(|v| v.is_finite())
            .collect();
        let n = values.len();
        let mean = if n == 0 {
            f64::NAN
        } else {
            values.iter().sum::<f64>() / n as f64
        };
        // Sample standard deviation (n−1). The bootstrap distribution is a
        // sample from the sampling distribution, so the unbiased divisor is the
        // right one; with n=1 there is no spread to report.
        let sd = if n > 1 {
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt()
        } else {
            f64::NAN
        };
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite values sort"));
        let median = percentile(&sorted, 50.0).unwrap_or(f64::NAN);
        let ci_percentile = match (
            percentile(&sorted, alpha),
            percentile(&sorted, 100.0 - alpha),
        ) {
            (Some(lo), Some(hi)) => Some((lo, hi)),
            _ => None,
        };
        let orig = original.and_then(|o| o.estimates.get(j).copied());
        parameters.push(ParameterSummary {
            name: name.clone(),
            original: orig,
            mean,
            bias: orig.map(|o| mean - o),
            standard_error: sd,
            median,
            ci_percentile,
            ci_standard_error: orig
                .filter(|_| sd.is_finite())
                .map(|o| (o - z * sd, o + z * sd)),
        });
    }

    let diagnostic_means = diagnostic_means(replicates);

    BootstrapSummary {
        parameters,
        n_completed,
        n_included: included.len(),
        excluded_by: excluded_counts,
        confidence_level: options.confidence_level,
        diagnostic_means,
    }
}

/// PsN's `diagnostic.means`: the mean of each per-replicate diagnostic, over
/// **every** completed replicate rather than only the included ones — the point
/// of the block is to say how the run as a whole behaved, including the part the
/// filters removed.
fn diagnostic_means(replicates: &[ReplicateResult]) -> Vec<(String, f64)> {
    let completed: Vec<&ReplicateResult> =
        replicates.iter().filter(|r| r.error.is_none()).collect();
    if completed.is_empty() {
        return Vec::new();
    }
    let n = completed.len() as f64;
    let frac = |f: &dyn Fn(&ReplicateResult) -> bool| -> f64 {
        completed.iter().filter(|r| f(r)).count() as f64 / n
    };
    vec![
        (
            "minimization_successful".to_string(),
            frac(&|r| r.converged),
        ),
        (
            "estimate_near_boundary".to_string(),
            frac(&|r| r.estimate_near_boundary),
        ),
        (
            "covariance_step_successful".to_string(),
            frac(&|r| r.covariance_step_successful),
        ),
        (
            "covariance_step_warnings".to_string(),
            frac(&|r| r.covariance_step_warnings),
        ),
        (
            "ofv".to_string(),
            completed.iter().map(|r| r.ofv).sum::<f64>() / n,
        ),
        (
            "subproblem_est_time".to_string(),
            completed.iter().map(|r| r.seconds).sum::<f64>() / n,
        ),
    ]
}

#[cfg(test)]
#[path = "summary_tests.rs"]
mod tests;
