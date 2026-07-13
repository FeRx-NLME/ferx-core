//! Categorical / count observation likelihoods (Phase 4, Track C — #760).
//!
//! Slice 1a ships the **Binary / Bernoulli** (logistic) endpoint: `state ∈ {0,1}`
//! observed via [`ObsRecord::DiscreteState`], scored against `p = inv_logit(lp)`
//! where the linear predictor `lp` (log-odds) is the user's `[binary_model] logit`
//! expression over θ/η/covariates. The per-subject data term is the negative
//! Bernoulli log-likelihood
//!
//! ```text
//! NLL = −Σ_r [ y_r·log p_r + (1 − y_r)·log(1 − p_r) ]
//!     =  Σ_r [ softplus(lp_r) − y_r·lp_r ]        (numerically stable form)
//! ```
//!
//! evaluated **per record** under the model-time guard so a `TIME` term in the
//! predictor is honoured (baseline covariates are read from `subject.covariates`).
//! Ordinal / Poisson / negative-binomial are later slices of this module.
//!
//! The whole module is gated behind `survival` (the non-Gaussian endpoint feature —
//! see `Cargo.toml`); [`LinkFn`]/[`LinearPredictorFn`] live under the same gate.

use crate::parser::model_parser::with_model_time;
use crate::types::{
    CompiledModel, EndpointLikelihood, LinearPredictorFn, LinkFn, ObsRecord, Subject,
};
use std::collections::HashMap;

/// `log(1 + eˣ)`, computed without overflow: `max(x, 0) + log1p(e^{−|x|})`.
#[inline]
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Negative Bernoulli log-likelihood for one subject's binary records on `cmt`.
/// Scans `records` (typically the subject's full `obs_records`) and scores only the
/// [`ObsRecord::DiscreteState`] rows on `cmt` — filtering by CMT here (rather than
/// trusting the caller) keeps a second binary CMT's rows from being double-counted and
/// lets the caller pass `&subject.obs_records` with zero allocation. States are assumed
/// validated to `{0,1}` by [`validate_binary_states`] at fit setup; a `debug_assert`
/// backstops that, and a release-safe clamp keeps a mis-validated `state ≥ 2` from
/// injecting a wild `y` into the sum.
///
/// The linear predictor is evaluated **once per record** inside [`with_model_time`]
/// so a `TIME` term resolves to that record's time; a predictor without `TIME` is
/// constant across records (the guard is a cheap thread-local set/restore).
///
/// Returns the raw NLL (positive). Callers apply the same `2·` OFV-scale factor they
/// apply to `tte_endpoint_nll`.
pub(crate) fn binary_data_term(
    link: LinkFn,
    lp_fn: &LinearPredictorFn,
    cmt: usize,
    records: &[ObsRecord],
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
) -> f64 {
    let mut nll = 0.0;
    for r in records {
        if let ObsRecord::DiscreteState {
            time,
            state,
            cmt: c,
        } = r
        {
            if *c != cmt {
                continue; // a DiscreteState row for a different endpoint's CMT
            }
            debug_assert!(*state <= 1, "binary state must be 0/1 (validated pre-fit)");
            let lp = with_model_time(*time, || lp_fn(theta, eta, covariates));
            if !lp.is_finite() {
                // An overflowed predictor (extreme θ / covariate) makes
                // `softplus(±∞) − y·(±∞)` a NaN/∞ that the FOCEI outer objective
                // (`foce_subject_nll_interaction_with_tte`) does not guard at its final
                // return — repel the optimizer with the survival module's `1e20`
                // sentinel instead (fail-loud-but-alive), as `tte_data_term` does for
                // its own degenerate cases.
                nll += 1e20;
                continue;
            }
            let y = (*state as f64).min(1.0);
            nll += match link {
                // p = inv_logit(lp): −[y·log p + (1−y)·log(1−p)] = softplus(lp) − y·lp.
                LinkFn::Logit => softplus(lp) - y * lp,
            };
        }
    }
    nll
}

/// Sum the per-subject NLL of every **discrete** (non-Gaussian, non-TTE) endpoint — the
/// binary / Bernoulli term today; ordinal / Poisson / negative-binomial are future arms of
/// the same match. Returns `0.0` when the model has no discrete endpoint or the subject has
/// no records for one. Callers apply the site's OFV-scale factor (`2·` on the halved
/// FOCEI/IOV `data_ll`, `1·` on the raw SAEM / non-interaction paths) — the same factor they
/// apply to the TTE term.
///
/// A new discrete endpoint family adds **one arm here** and needs no new likelihood
/// dispatch-site edit: every FOCEI / IOV / SAEM path, the FD-Hessian closure, and IMP reach
/// it through the single `accumulate_non_gaussian_nll` seam. It also doubles as the FD-Hessian
/// closure at the FOCEI interaction site: evaluated at a perturbed η it re-scores every
/// discrete record, so `data_term_hessian_fd` picks up the curvature w.r.t. η.
pub(crate) fn discrete_subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    if subject.obs_records.is_empty() {
        return 0.0;
    }
    let mut nll = 0.0;
    for (cmt, endpoint) in &model.endpoints {
        // Binary today; a new discrete family (ordinal / Poisson / negative-binomial …) adds
        // its branch here — reached through this one function, so no likelihood dispatch site
        // changes. (Gaussian is scored by the residual path, TTE by the hazard dispatch.)
        if let EndpointLikelihood::Binary { link, lp_fn, .. } = endpoint {
            // `binary_data_term` filters `obs_records` by `cmt` itself — no per-call
            // allocation, even inside the FD-Hessian closure.
            nll += binary_data_term(
                *link,
                lp_fn,
                *cmt,
                &subject.obs_records,
                theta,
                eta,
                &subject.covariates,
            );
        }
    }
    nll
}

/// Fail-loud check that every binary record's observed state is `0` or `1`. Run once
/// at fit setup: the datareader accepts any non-negative integer DV on a discrete CMT
/// (it can't know binary-vs-ordinal), so a `Binary` endpoint must reject `state ≥ 2`
/// itself rather than silently fold it into the Bernoulli term. `records` may be a
/// subject's full `obs_records`; only `DiscreteState` rows on `cmt` are checked.
pub(crate) fn validate_binary_states(cmt: usize, records: &[ObsRecord]) -> Result<(), String> {
    for r in records {
        if let ObsRecord::DiscreteState { state, cmt: c, .. } = r {
            if *c == cmt && *state > 1 {
                return Err(format!(
                    "[binary_model] cmt = {cmt}: observed DV must be 0 or 1 (Bernoulli), got \
                     {state}. For an ordered response with more than two categories use an \
                     ordinal endpoint (not yet supported)."
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn const_lp(value: f64) -> LinearPredictorFn {
        // Predictor that ignores θ/η/cov and returns a fixed log-odds.
        Box::new(move |_t: &[f64], _e: &[f64], _c: &HashMap<String, f64>| value)
    }

    fn rec(time: f64, state: usize) -> ObsRecord {
        ObsRecord::DiscreteState {
            time,
            state,
            cmt: 3,
        }
    }

    /// −log Bernoulli against a hand computation for a mixed 0/1 subject.
    #[test]
    fn binary_data_term_matches_hand_computation() {
        let cov = HashMap::new();
        // lp = 1.0 ⇒ p = 1/(1+e⁻¹) ≈ 0.731059.
        let lp = 1.0_f64;
        let p = 1.0 / (1.0 + (-lp).exp());
        let recs = [rec(0.0, 1), rec(5.0, 0), rec(9.0, 1)];
        // Two events (y=1) and one non-event (y=0).
        let expect = -(2.0 * p.ln() + (1.0 - p).ln());
        let got = binary_data_term(LinkFn::Logit, &const_lp(lp), 3, &recs, &[], &[], &cov);
        assert!((got - expect).abs() < 1e-12, "got {got}, expect {expect}");
    }

    /// p = 0.5 at lp = 0 ⇒ every record contributes ln 2.
    #[test]
    fn binary_data_term_half_probability() {
        let cov = HashMap::new();
        let recs = [rec(0.0, 0), rec(1.0, 1)];
        let got = binary_data_term(LinkFn::Logit, &const_lp(0.0), 3, &recs, &[], &[], &cov);
        assert!((got - 2.0 * std::f64::consts::LN_2).abs() < 1e-12);
    }

    /// Stable at extreme log-odds where the naive `(1+eˣ).ln()` would overflow, and
    /// still equal to −log p to full precision.
    #[test]
    fn binary_data_term_stable_at_extremes() {
        let cov = HashMap::new();
        // lp = +40, y = 1 ⇒ p ≈ 1 ⇒ −log p ≈ 0.
        let nll = |lp: f64, state: usize| {
            binary_data_term(
                LinkFn::Logit,
                &const_lp(lp),
                3,
                &[rec(0.0, state)],
                &[],
                &[],
                &cov,
            )
        };
        // lp = +40, y = 1 ⇒ p ≈ 1 ⇒ −log p ≈ 0.
        assert!(nll(40.0, 1).is_finite() && nll(40.0, 1) < 1e-16);
        // lp = −40, y = 1 ⇒ p ≈ 0 ⇒ −log p ≈ 40.
        assert!((nll(-40.0, 1) - 40.0).abs() < 1e-9);
        // lp = 800 (naive e^lp overflows to ∞): softplus stays finite ≈ lp for y = 0.
        assert!((nll(800.0, 0) - 800.0).abs() < 1e-9);
        assert!((1.0_f64 + 800.0_f64.exp()).ln().is_infinite()); // the naive form does overflow
                                                                 // Non-finite lp (overflowed predictor) → the 1e20 sentinel, never NaN/∞.
        assert_eq!(nll(f64::INFINITY, 1), 1e20);
        assert_eq!(nll(f64::NEG_INFINITY, 0), 1e20);
    }

    /// `validate_binary_states` rejects a non-Bernoulli code with a message that names
    /// the CMT and the offending value.
    #[test]
    fn validate_rejects_state_two() {
        let recs = [rec(0.0, 0), rec(1.0, 2)];
        let err = validate_binary_states(3, &recs).unwrap_err();
        assert!(err.contains("cmt = 3"), "msg: {err}");
        assert!(err.contains("got 2"), "msg: {err}");
        assert!(err.contains("0 or 1"), "msg: {err}");
        // A clean 0/1 subject passes.
        assert!(validate_binary_states(3, &[rec(0.0, 0), rec(1.0, 1)]).is_ok());
    }

    /// `softplus` agrees with the naive form in the safe range.
    #[test]
    fn softplus_matches_naive_in_range() {
        for x in [-5.0_f64, -1.0, 0.0, 0.5, 3.0, 20.0] {
            let naive = (1.0 + x.exp()).ln();
            assert!((softplus(x) - naive).abs() < 1e-12, "x={x}");
        }
    }
}
