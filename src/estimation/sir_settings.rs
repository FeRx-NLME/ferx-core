//! SIR's settings and its percentile interval. Below the estimators so `types` and the level
//! binding can name them (#3041).
use crate::cold_sort::ColdSort;
use crate::types::*;

/// Every [`FitOptions`] setting that changes a SIR result, as one SIR run used
/// it (#1758). Carried on [`FitResult::sir_settings`] so a later
/// [`run_sir`](crate::run_sir) on that fit re-scores under the same settings by
/// default, and so a reader can tell which scale, degrees of freedom and seed
/// produced the reported intervals.
///
/// Recorded inside `run_sir_core` from the options it scored with, never by a
/// caller, so the record cannot disagree with the draws.
///
/// `method` and `interaction` are not here: they already ride on the
/// [`FitResult`] (`method`, `interaction`), and which of them SIR scores is
/// [`run_sir`](crate::run_sir)'s business (#1710).
///
/// `Default` is `SirSettings::from_options(&FitOptions::default())`, the
/// settings of a SIR run that changed nothing.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SirSettings {
    /// `sir_samples`: proposal draws.
    pub samples: usize,
    /// `sir_resamples`: draws kept by the resampling step.
    pub resamples: usize,
    /// The seed the draws used, resolved: `sir_seed`, or the built-in default
    /// when that was unset. So the run is reproducible from this field alone.
    pub seed: u64,
    /// `sir_df`: Student-t proposal degrees of freedom.
    pub df: f64,
    /// `sir_scale`: the scale SIR's flat prior is flat on (#1723).
    pub scale: SirScale,
    /// `sir_keep_samples`: whether the resampled vectors were retained.
    pub keep_samples: bool,
    /// The inner-loop and ODE settings each draw's EBE re-solve and objective ran
    /// under (#426). The same record a fit carries on
    /// [`FitResult::scoring_settings`](crate::FitResult::scoring_settings), taken here
    /// from the options SIR was handed. `inner_restarts` is recorded for completeness:
    /// the draws are warm-started and run no inner multi-start.
    pub scoring: crate::ScoringSettings,
}

impl SirSettings {
    /// The settings a SIR run under `options` uses: each field copied, the seed
    /// resolved to the built-in default when `options.sir_seed` is `None`.
    pub fn from_options(options: &FitOptions) -> Self {
        Self {
            samples: options.sir_samples,
            resamples: options.sir_resamples,
            seed: options.sir_seed.unwrap_or(DEFAULT_SIR_SEED),
            df: options.sir_df,
            scale: options.sir_scale,
            keep_samples: options.sir_keep_samples,
            scoring: crate::ScoringSettings::from_options(options),
        }
    }

    /// The first field outside the `[fit_options]` domain, if any (#1770): `df`, then the
    /// scoring half. Fields are named as the record names them (`df`, not `sir_df`).
    pub(crate) fn check_domain(
        &self,
    ) -> Result<(), crate::estimation::population_check::OutOfDomain> {
        crate::estimation::population_check::sir_df_in_domain("df", self.df)?;
        self.scoring.check_domain()
    }
}

/// Compute 2.5th and 97.5th percentiles from a sample.
pub(crate) fn percentile_ci(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut sorted = values.to_vec();
    sorted.cold_sort_by(&|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let lo_idx = ((n as f64) * 0.025).floor() as usize;
    let hi_idx = ((n as f64) * 0.975).ceil() as usize;
    let lo = sorted[lo_idx.min(n - 1)];
    let hi = sorted[hi_idx.min(n - 1)];
    (lo, hi)
}

/// The SIR seed when `sir_seed` is unset.
pub(crate) const DEFAULT_SIR_SEED: u64 = 12345;
