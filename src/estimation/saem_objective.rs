//! The objective a SAEM fit reports by default: NONMEM's `SAEMOBJ` (#2973, `saem_ofv = saem`).
//!
//! NONMEM prints a `SAEMOBJ` at every `PRINT` iteration of `METHOD=SAEM` and reports its last
//! value as the step's objective. It is **not** a likelihood: it is a Monte-Carlo readout of the
//! joint density of the data and the random effects, and NONMEM says so by following SAEM with
//! `METHOD=IMP EONLY=1` for the marginal one (`saem_ofv = imp`).
//!
//! # What it is
//!
//! Established by a closed-form linear-Gaussian experiment on NONMEM 7.6.0 (where `E[−2 log p]`
//! and the marginal are known exactly), and checked on the SAEM runs in
//! `nonmem_anchor/results/`:
//!
//! ```text
//!   SAEMOBJ = Σ_i  −2 · log( mean_k  p(y_i, η_ik | θ, Ω, σ) )        (no 2π constants)
//! ```
//!
//! `p` is the joint density of subject `i`'s data and random effects, with `log|Ω|` and
//! `log σ²` but neither the `N_obs·log 2π` nor the `NIND·NETA·log 2π` constant, so
//! `−log p` is [`crate::stats::likelihood::individual_nll`]. The mean sits **inside the log**:
//! during burn-in it runs over the chain states of the current iteration; in the accumulation
//! (convergence) phase over the draws **and the iterations**, which is the value printed
//! last. In the limit of many draws the mean-in-the-log lowers `E[−2 log p]` by
//! `n_η·(1 − ln 2) = 0.307` per subject per random effect, the offset the committed anchors
//! show against the posterior expectation.
//!
//! ferx keeps, per subject, a running log-sum-exp of `−nll_i` over the convergence phase, one
//! draw per iteration: the chain state after the E-step, scored under the parameters the
//! iteration ends with. With step `γ_k = 1/(k − K₁)` the stochastic-approximation average *is*
//! this plain mean. The accumulator reads the NLL cache the iteration refreshes anyway, so the
//! objective is free.

use crate::types::CompiledModel;

/// Why a SAEM objective is not defined for this model: the caller reports the
/// importance-sampled value and says so in a warning.
///
/// Only a mixture: its complete-data density can include the sampled class (`p(y, η, c)`, with
/// the mixing probability) or be the class-marginal `p(y, η)`, NONMEM's `$MIX` SAEMOBJ
/// convention is not anchored by any committed run, and picking one unverified would be a
/// guess. IOV models are covered: the cache holds `−log p(y, η, κ)` (the occasion effects'
/// prior included), averaged inside the log like any other draw (not anchored either), an
/// IOV-only model (`n_η = 0`, `n_κ > 0`) included. With no random effects at all (no η and no
/// κ) SAEMOBJ is the data `−2 log L` at the final estimates, so the last draw's value is
/// reported, not an average over iterations at moving parameters.
pub(crate) fn unsupported(model: &CompiledModel) -> Option<&'static str> {
    if model.mixture.is_some() {
        Some(
            "the SAEM objective of a mixture (complete data with or without the sampled class) \
             has no NONMEM anchor",
        )
    } else {
        None
    }
}

/// Per-subject running `log mean_t exp(−nll_t)`, numerically stable (log-sum-exp with a running
/// maximum; a raw exponential is never formed).
pub(crate) struct SaemObjAccumulator {
    enabled: bool,
    /// No η and no κ: the value is the last draw's (the data −2 log L), not a mean.
    last_only: bool,
    max: Vec<f64>,
    /// `Σ_t exp(−nll_t − max)`.
    sum: Vec<f64>,
    n: u64,
}

impl SaemObjAccumulator {
    pub(crate) fn new(n_subjects: usize, enabled: bool, last_only: bool) -> Self {
        let n = if enabled { n_subjects } else { 0 };
        Self { enabled, last_only, max: vec![f64::NEG_INFINITY; n], sum: vec![0.0; n], n: 0 }
    }

    /// One draw per subject: `nll[i]` is `−log p(y_i, η_i)` (no constants), the joint NLL.
    pub(crate) fn update(&mut self, nll: &[f64]) {
        if !self.enabled {
            return;
        }
        if self.last_only {
            self.max = nll.iter().map(|v| -v).collect();
            self.sum = vec![1.0; nll.len()];
            self.n = 1;
            return;
        }
        for (i, &v) in nll.iter().enumerate() {
            let x = -v;
            if x > self.max[i] {
                // Rescale the running sum to the new maximum (exp(-inf) = 0 on first use).
                self.sum[i] = self.sum[i] * (self.max[i] - x).exp() + 1.0;
                self.max[i] = x;
            } else {
                self.sum[i] += (x - self.max[i]).exp();
            }
        }
        self.n += 1;
    }

    /// `−2 Σ_i log mean`, `None` before the first draw or when a subject's draws diverged.
    pub(crate) fn value(&self) -> Option<f64> {
        if !self.enabled || self.n == 0 {
            return None;
        }
        let ln_n = (self.n as f64).ln();
        let mut total = 0.0;
        for (m, s) in self.max.iter().zip(&self.sum) {
            let l = m + s.ln() - ln_n;
            if !l.is_finite() || *m < -crate::stats::likelihood::SUBJECT_SENTINEL_NLL * 0.5 {
                return None;
            }
            total += l;
        }
        Some(-2.0 * total)
    }
}

#[cfg(test)]
#[path = "saem_objective_tests.rs"]
mod tests;
