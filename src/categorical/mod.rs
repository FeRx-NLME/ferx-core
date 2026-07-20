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
    CompiledModel, EndpointLikelihood, EndpointPredictionResult, LinearPredictorFn, LinkFn,
    ObsRecord, Prediction, Subject,
};
use rand::RngExt;
use std::collections::HashMap;

/// `log(1 + eˣ)`, computed without overflow: `max(x, 0) + log1p(e^{−|x|})`.
#[inline]
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// `1 / (1 + e^{−x})`, computed without overflow by branching on the sign so the
/// exponential always has a non-positive argument. Maps `±∞` to `1`/`0` and
/// propagates NaN (callers guard that case explicitly).
#[inline]
fn inv_logit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Evaluate the linear predictor once per binary record on `cmt`, invoking
/// `f(time, observed_state, lp)` for each.
///
/// **Single owner of the record-selection + time-scoping rule.** The likelihood
/// ([`binary_data_term`]), the simulation sampler ([`simulate_binary`]) and the
/// predictor ([`predict_binary`]) all route through here, so they cannot disagree
/// about *which* rows belong to the endpoint, *what time* a `TIME` term resolves
/// to, or *what `lp`* a record carries. A drift between the sampler's `lp` and the
/// likelihood's `lp` is precisely the silent bug the SSE round-trip exists to catch
/// (§8.8.8) — sharing the evaluator makes that class of drift unrepresentable
/// rather than merely tested-for.
///
/// Filtering by CMT here (rather than trusting the caller) keeps a second binary
/// CMT's rows from being double-counted and lets callers pass a subject's full
/// `obs_records` with zero allocation.
fn for_each_binary_lp<F>(
    lp_fn: &LinearPredictorFn,
    cmt: usize,
    records: &[ObsRecord],
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    mut f: F,
) where
    F: FnMut(f64, usize, f64),
{
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
            // The predictor is evaluated **once per record** inside `with_model_time`
            // so a `TIME` term resolves to that record's time; a predictor without
            // `TIME` is constant across records (the guard is a cheap thread-local
            // set/restore).
            let lp = with_model_time(*time, || lp_fn(theta, eta, covariates));
            f(*time, *state, lp);
        }
    }
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
    for_each_binary_lp(
        lp_fn,
        cmt,
        records,
        theta,
        eta,
        covariates,
        |_t, state, lp| {
            debug_assert!(state <= 1, "binary state must be 0/1 (validated pre-fit)");
            if !lp.is_finite() {
                // An overflowed predictor (extreme θ / covariate) makes
                // `softplus(±∞) − y·(±∞)` a NaN/∞ that the FOCEI outer objective
                // (`foce_subject_nll_interaction_with_tte`) does not guard at its final
                // return — repel the optimizer with the survival module's `1e20`
                // sentinel instead (fail-loud-but-alive), as `tte_data_term` does for
                // its own degenerate cases.
                nll += 1e20;
                return;
            }
            let y = (state as f64).min(1.0);
            nll += match link {
                // p = inv_logit(lp): −[y·log p + (1−y)·log(1−p)] = softplus(lp) − y·lp.
                LinkFn::Logit => softplus(lp) - y * lp,
            };
        },
    );
    nll
}

/// `P(Y = 1)` for one record's linear predictor under `link`.
#[inline]
fn binary_prob(link: LinkFn, lp: f64) -> f64 {
    match link {
        LinkFn::Logit => inv_logit(lp),
    }
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

/// Draw a simulated outcome for every binary record of every `Binary` endpoint, pushing
/// one [`SimOutcome::Category`] row per record (Slice 1b, §8.8.2).
///
/// Binary is the *easy* row of the per-endpoint simulation table: it fits the **fixed
/// observation grid**, so — unlike TTE/RTTE/CTMM — there is no horizon, no event
/// location and no path generator. The record times come from the input dataset
/// exactly as the Gaussian grid does; only the generative step differs
/// (`p = link⁻¹(lp@η)`, then one Bernoulli draw) .
///
/// Called from `emit_subject_rows` alongside [`crate::survival::simulate_tte`], on the
/// same RNG stream and after the Gaussian rows, so a mixed PK + binary model emits both.
/// Before this producer existed, a binary endpoint contributed **zero** rows to
/// `simulate()` with no error — the discrete records live in `obs_records`, which the
/// Gaussian emitter (driven by `obs_times`) never visits.
///
/// `ipred` on the emitted row carries `p`, matching the Gaussian convention that `ipred`
/// is the individual prediction the outcome was drawn around.
#[allow(clippy::too_many_arguments)]
pub fn simulate_binary<R: rand::Rng>(
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
    for (cmt, endpoint) in &model.endpoints {
        let EndpointLikelihood::Binary { link, lp_fn, .. } = endpoint else {
            continue;
        };
        let mut n_nonfinite = 0usize;
        for_each_binary_lp(
            lp_fn,
            *cmt,
            &subject.obs_records,
            theta,
            eta,
            &subject.covariates,
            |time, _state, lp| {
                // A NaN predictor has no Bernoulli law to draw from. Unlike the fit
                // path there is no optimizer to steer away from it (§8.8.3's "simulation
                // has no optimizer to steer"), so count it and warn rather than emit a
                // silently-arbitrary 0/1. `±∞` is *not* degenerate — `inv_logit` maps it
                // to a valid p = 1/0 — so only NaN is rejected.
                if lp.is_nan() {
                    n_nonfinite += 1;
                    return;
                }
                let p = binary_prob(*link, lp);
                // One uniform per record on the shared stream, drawn from the same
                // `Open01` distribution the TTE samplers use. `u ∈ (0, 1)` with a strict
                // `u < p` makes the degenerate ends exact: p = 0 never fires, p = 1
                // always does.
                let u: f64 = rng.sample(rand::distr::Open01);
                let state = usize::from(u < p);
                results.push(crate::api::SimulationResult {
                    draw,
                    sim,
                    id: subject.id.clone(),
                    time,
                    cmt: *cmt,
                    ipred: p,
                    outcome: crate::types::SimOutcome::Category { state },
                });
            },
        );
        if n_nonfinite > 0 {
            warnings.push(format!(
                "[binary_model] cmt = {cmt}: subject {} has {n_nonfinite} record(s) whose linear \
                 predictor evaluated to NaN — those rows were skipped and carry no simulated \
                 outcome. Check the `logit` expression and the subject's covariates.",
                subject.id
            ));
        }
    }
}

/// Category probabilities `[1 − p, p]` for every binary record, at the supplied `eta`
/// (the caller's EBE, or zeros for a population prediction).
///
/// The returned `probs` are indexed so `probs[k] == P(Y = k)` — i.e. index *is* the
/// observed 0/1 DV code, which is what makes [`Prediction::prob`] usable directly
/// against a data value.
pub fn predict_binary(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    results: &mut Vec<EndpointPredictionResult>,
) {
    if subject.obs_records.is_empty() {
        return;
    }
    for (cmt, endpoint) in &model.endpoints {
        let EndpointLikelihood::Binary { link, lp_fn, .. } = endpoint else {
            continue;
        };
        for_each_binary_lp(
            lp_fn,
            *cmt,
            &subject.obs_records,
            theta,
            eta,
            &subject.covariates,
            |time, state, lp| {
                let p = binary_prob(*link, lp);
                results.push(EndpointPredictionResult {
                    id: subject.id.clone(),
                    cmt: *cmt,
                    time,
                    observed: Some(state as f64),
                    prediction: Prediction::CatProbs {
                        probs: vec![1.0 - p, p],
                    },
                });
            },
        );
    }
}

/// Standardized (Pearson) residual for a binary record: `(y − p) / √(p(1−p))`
/// (§8.8.5). Returns NaN at a degenerate `p ∈ {0, 1}`, where the residual is
/// undefined — callers surface it rather than silently substituting a finite value.
///
/// Pairs with [`predict_binary`], whose rows carry both the observed DV and `p`, so a
/// caller (the R wrapper, a diagnostics table) can form the residual without
/// re-deriving the probability. The core `{model}-sdtab.csv` writer is still
/// Gaussian-shaped (IPRED/IWRES/CWRES) and does not yet emit a discrete-endpoint row;
/// giving every non-Gaussian endpoint its own diagnostics block is a separate slice.
pub fn binary_pearson_residual(y: f64, p: f64) -> f64 {
    let v = p * (1.0 - p);
    if v <= 0.0 {
        return f64::NAN;
    }
    (y - p) / v.sqrt()
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

    // ---- Slice 1b: sampler / predictor ------------------------------------------------

    /// `inv_logit` matches the naive form in range, saturates at `±∞` without
    /// producing NaN, and never leaves `[0, 1]` at extremes where the naive
    /// `1/(1+e^{−x})` would divide by an overflowed denominator.
    #[test]
    fn inv_logit_is_stable_and_bounded() {
        for x in [-5.0_f64, -1.0, 0.0, 0.5, 3.0, 20.0] {
            let naive = 1.0 / (1.0 + (-x).exp());
            assert!((inv_logit(x) - naive).abs() < 1e-12, "x={x}");
        }
        // Extremes: exactly saturated, still in range, no NaN.
        assert_eq!(inv_logit(f64::INFINITY), 1.0);
        assert_eq!(inv_logit(f64::NEG_INFINITY), 0.0);
        for x in [-800.0_f64, -40.0, 40.0, 800.0] {
            let p = inv_logit(x);
            assert!((0.0..=1.0).contains(&p), "x={x} gave p={p}");
        }
        // Symmetry: p(−x) == 1 − p(x).
        for x in [0.3_f64, 2.0, 11.0] {
            assert!(
                (inv_logit(-x) - (1.0 - inv_logit(x))).abs() < 1e-15,
                "x={x}"
            );
        }
        assert!(inv_logit(f64::NAN).is_nan());
    }

    /// **The drift guard.** The sampler's `p` and the likelihood's NLL must describe the
    /// same Bernoulli law: `−log P(y)` recomputed from the sampler's `p` has to equal
    /// `binary_data_term` for that record. A future link or a refactor that changes one
    /// side without the other fails here rather than silently producing an SSE that
    /// cannot recover its own generating θ (§8.8.8).
    #[test]
    fn sampler_probability_agrees_with_likelihood() {
        let cov = HashMap::new();
        for lp in [-6.0_f64, -1.3, 0.0, 0.7, 4.2] {
            let p = binary_prob(LinkFn::Logit, lp);
            for state in [0usize, 1] {
                let nll_from_p = if state == 1 { -p.ln() } else { -(1.0 - p).ln() };
                let nll_from_term = binary_data_term(
                    LinkFn::Logit,
                    &const_lp(lp),
                    3,
                    &[rec(0.0, state)],
                    &[],
                    &[],
                    &cov,
                );
                assert!(
                    (nll_from_p - nll_from_term).abs() < 1e-12,
                    "lp={lp} y={state}: sampler p implies {nll_from_p}, likelihood says {nll_from_term}"
                );
            }
        }
    }

    /// The Bernoulli draw is exact at the degenerate ends and unbiased in between.
    /// `Open01` never yields 0 or 1, so `u < p` makes `p = 0` impossible and `p = 1`
    /// certain — the property that keeps a saturated predictor from flipping states.
    #[test]
    fn bernoulli_draw_is_exact_at_the_ends_and_calibrated_between() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(20260720);
        let draw = |rng: &mut rand::rngs::StdRng, p: f64| -> usize {
            let u: f64 = rng.sample(rand::distr::Open01);
            usize::from(u < p)
        };
        // Degenerate ends are exact over many draws, not merely likely.
        for _ in 0..1000 {
            assert_eq!(draw(&mut rng, 0.0), 0);
            assert_eq!(draw(&mut rng, 1.0), 1);
        }
        // Calibration: the realised frequency tracks p. 20k draws ⇒ SE ≈ 0.0035 at
        // p = 0.5, so a 0.02 band is ~6 SE — tight enough to catch an inverted
        // comparison or an off-by-one link, loose enough never to flake.
        for p in [0.25_f64, 0.5, 0.8] {
            let n = 20_000;
            let ones = (0..n).filter(|_| draw(&mut rng, p) == 1).count();
            let freq = ones as f64 / n as f64;
            assert!((freq - p).abs() < 0.02, "p={p} gave frequency {freq}");
        }
    }

    /// Pearson residual: sign follows `y − p`, magnitude is standardized, and a
    /// degenerate `p` yields NaN rather than a plausible-looking finite number.
    #[test]
    fn pearson_residual_standardizes_and_flags_degenerate_p() {
        // p = 0.5 ⇒ √(p(1−p)) = 0.5 ⇒ residual = ±1.
        assert!((binary_pearson_residual(1.0, 0.5) - 1.0).abs() < 1e-12);
        assert!((binary_pearson_residual(0.0, 0.5) + 1.0).abs() < 1e-12);
        // A surprising observation is a large positive residual.
        let r = binary_pearson_residual(1.0, 0.01);
        assert!(r > 9.0, "expected a large residual, got {r}");
        // Degenerate p: undefined, surfaced as NaN (not 0, not ±∞).
        assert!(binary_pearson_residual(1.0, 0.0).is_nan());
        assert!(binary_pearson_residual(0.0, 1.0).is_nan());
    }

    /// `Prediction::prob` reads a category out by index and refuses a continuous
    /// prediction instead of panicking or coercing.
    #[test]
    fn prediction_prob_accessor() {
        let p = Prediction::CatProbs {
            probs: vec![0.3, 0.7],
        };
        assert_eq!(p.prob(0), Some(0.3));
        assert_eq!(p.prob(1), Some(0.7));
        assert_eq!(
            p.prob(2),
            None,
            "out-of-range index must be None, not a panic"
        );
        assert_eq!(Prediction::Continuous { pred: 1.0 }.prob(0), None);
    }
}
