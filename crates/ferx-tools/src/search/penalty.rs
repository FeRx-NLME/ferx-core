//! pyDarwin-style penalized fitness (#1185, P6 of the #1175 epic).
//!
//! pyDarwin ranks a candidate on one scalar: its OFV plus a charge for every
//! estimated parameter and for every way the fit went wrong, so a model that
//! buys ten OFV points with two extra parameters and a failed covariance step
//! loses to one that did not. The inputs are exactly the ones
//! [`check_strictness`](ferx_core::check_strictness) already reads — the
//! free-parameter tally, `converged`, the covariance status, the largest
//! parameter correlation, the condition number — so this is a scoring
//! function over an existing verdict, not new numerics.
//!
//! Two things differ from a strictness *gate*. A gate excludes; a penalty
//! ranks, so a non-converged model that is still far better than anything
//! converged stays visible, charged. And a penalty applies to the runner's
//! every candidate whatever tool submitted it — `[rank] type = "penalized"`
//! is a [`Criterion`](super::Criterion) like `bic`, usable by the stepwise
//! tools as well as by the global search the issue asked for.
//!
//! The defaults are pyDarwin's, verbatim:
//!
//! | penalty | default |
//! |---|---|
//! | per estimated θ | 10 |
//! | per estimated Ω element | 10 |
//! | per estimated σ element | 10 |
//! | failure to converge | 100 |
//! | covariance step failed (or not run) | 100 |
//! | any \|correlation\| > `max_correlation` (0.95) | 100 |
//! | condition number > `max_condition_number` (1000) | 100 |
//! | per non-influential gene | 0.00001 |
//! | crash value (no fit at all) | 99 999 999 |
//!
//! and one that is ferx's, because ferx has a strictness *gate* where
//! pyDarwin has only penalties:
//!
//! | penalty | default |
//! |---|---|
//! | a fit that failed the strictness gate | 100 |
//!
//! The non-influential, crash and gate charges are not properties of a fit
//! alone, so [`Penalties::score`] does not apply them: the search that knows
//! a gene changed nothing in the rendered model, that a candidate produced
//! no fit, or that the gate refused it, applies them itself — see
//! `globalsearch`.

use ferx_core::{max_abs_correlation, CovarianceStatus, FitResult};
use serde::{Deserialize, Serialize};

/// The penalty schedule. Every field has pyDarwin's default, so a
/// `[rank.penalties]` table states only what it changes.
///
/// `Copy`, because it travels inside [`Criterion`](super::Criterion), which
/// the runner copies into every result and the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Penalties {
    /// Per estimated (free) θ.
    pub theta: f64,
    /// Per estimated Ω element — every free entry of the packed Ω and IOV
    /// κ triangles, so a 2×2 block costs three.
    pub omega: f64,
    /// Per estimated σ element.
    pub sigma: f64,
    /// A fit with `converged == false`.
    pub convergence: f64,
    /// A fit whose covariance step failed, was not requested, or delivered
    /// no matrix. Charged uniformly when no candidate runs one, in which
    /// case it changes no ranking; turn `covariance = true` on in the base
    /// model's `[fit_options]` for the correlation and condition-number
    /// charges to have anything to read.
    pub covariance: f64,
    /// Any parameter correlation whose absolute value exceeds
    /// [`max_correlation`](Self::max_correlation).
    pub correlation: f64,
    pub max_correlation: f64,
    /// A condition number above [`max_condition_number`](Self::max_condition_number),
    /// or one that is `NaN`.
    pub condition_number: f64,
    pub max_condition_number: f64,
    /// Per gene of a genome that changed nothing in the rendered model — a
    /// covariate on a parameter the structural choice removed. A tie-break
    /// towards the simpler genotype among identical phenotypes; the runner
    /// already fits identical phenotypes once.
    pub non_influential: f64,
    /// The fitness of a candidate that produced no fit: it does not compile,
    /// or `fit()` returned an error. Large enough to lose to any fitted
    /// model, finite so a genetic algorithm can still rank it.
    pub crash: f64,
    /// A fit the `[strictness]` gate refused — charged by a global search on
    /// top of the criterion, whatever the criterion, so that a model the
    /// search can never select still steers it but does not win it. Under a
    /// `penalized` criterion this comes on top of the per-failure charges
    /// above, which is deliberate: the gate is a stronger statement than
    /// any one of them. `0` makes an ineligible fit rank on its criterion
    /// alone.
    pub gate: f64,
}

impl Default for Penalties {
    fn default() -> Self {
        Self {
            theta: 10.0,
            omega: 10.0,
            sigma: 10.0,
            convergence: 100.0,
            covariance: 100.0,
            correlation: 100.0,
            max_correlation: 0.95,
            condition_number: 100.0,
            max_condition_number: 1000.0,
            non_influential: 0.00001,
            crash: 99_999_999.0,
            gate: 100.0,
        }
    }
}

impl Penalties {
    /// Every field finite and non-negative, and the two thresholds positive.
    pub fn validate(&self) -> Result<(), String> {
        let charges = [
            ("theta", self.theta),
            ("omega", self.omega),
            ("sigma", self.sigma),
            ("convergence", self.convergence),
            ("covariance", self.covariance),
            ("correlation", self.correlation),
            ("condition_number", self.condition_number),
            ("non_influential", self.non_influential),
            ("crash", self.crash),
            ("gate", self.gate),
        ];
        for (name, v) in charges {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!(
                    "[rank.penalties] {name} = {v}: a penalty must be a finite, non-negative \
                     number"
                ));
            }
        }
        if !(self.max_correlation.is_finite() && self.max_correlation > 0.0) {
            return Err(format!(
                "[rank.penalties] max_correlation = {}: must be a positive, finite threshold",
                self.max_correlation
            ));
        }
        if !(self.max_condition_number.is_finite() && self.max_condition_number > 0.0) {
            return Err(format!(
                "[rank.penalties] max_condition_number = {}: must be a positive, finite threshold",
                self.max_condition_number
            ));
        }
        Ok(())
    }

    /// The charged terms of one fit, each named, in schedule order — what
    /// [`score`](Self::score) sums onto the OFV. A term that does not apply
    /// is absent rather than zero, so a report can list what a candidate
    /// was charged for.
    pub fn terms(&self, result: &FitResult) -> Vec<(&'static str, f64)> {
        let mut out = Vec::new();
        let (n_theta, n_omega, n_sigma) = parameter_counts(result);
        if n_theta > 0 {
            out.push(("theta", self.theta * n_theta as f64));
        }
        if n_omega > 0 {
            out.push(("omega", self.omega * n_omega as f64));
        }
        if n_sigma > 0 {
            out.push(("sigma", self.sigma * n_sigma as f64));
        }
        if !result.converged {
            out.push(("convergence", self.convergence));
        }
        let has_covariance = matches!(result.covariance_status, CovarianceStatus::Computed)
            && result.covariance_matrix.is_some();
        if !has_covariance {
            out.push(("covariance", self.covariance));
        }
        if let Some(r) = max_abs_correlation(result) {
            if r > self.max_correlation {
                out.push(("correlation", self.correlation));
            }
        }
        if let Some(cn) = result.cov_condition_number {
            if cn.is_nan() || cn > self.max_condition_number {
                out.push(("condition_number", self.condition_number));
            }
        }
        out
    }

    /// `OFV + Σ terms`: the penalized fitness of one fit. Lower is better.
    pub fn score(&self, result: &FitResult) -> f64 {
        result.ofv + self.terms(result).iter().map(|(_, v)| v).sum::<f64>()
    }

    /// The charge for `n` non-influential genes.
    pub fn non_influential_charge(&self, n: usize) -> f64 {
        self.non_influential * n as f64
    }
}

/// `(θ, Ω, σ)` free-parameter counts, from the fit's Delattre tally. A
/// result whose tally does not add up to `n_parameters` — a bundle saved
/// before the tally existed — is charged every free parameter at the θ
/// rate, which is the one count it does carry.
fn parameter_counts(result: &FitResult) -> (usize, usize, usize) {
    let inp = &result.bic_inputs;
    if inp.n_free() == result.n_parameters {
        (
            inp.theta_random + inp.theta_fixed,
            inp.omega + inp.kappa,
            inp.sigma,
        )
    } else {
        (result.n_parameters, 0, 0)
    }
}

#[cfg(test)]
#[path = "penalty_tests.rs"]
mod tests;
