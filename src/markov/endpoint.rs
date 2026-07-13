//! CTMM endpoint wiring (Phase 5, Track D — #759).
//!
//! Bridges the model layer to the pure numerical [`ctmm_data_term`] kernel in
//! [`crate::markov`]. Its job is everything the leaf deliberately does **not**
//! know about (§8.7 of `plans/tte-survival-markov.md`):
//!
//! - **build the generator** `Q(θ, η, cov)` from the endpoint's
//!   [`GeneratorFn`](crate::types::GeneratorFn) (the `[markov_model] transition`
//!   intensities, diagonal filled row-sum-zero), once per subject
//!   (time-homogeneous — a drug-driven `Q(t)` is Phase 6);
//! - **collect the observations** — map each subject
//!   [`ObsRecord::DiscreteState`] row on the endpoint's CMT to a
//!   [`StateObs`], translating its raw DV code to the 0-based generator index via
//!   the endpoint's `state_codes` table; and
//! - **sum the per-subject NLL** across every CTMM endpoint, returning the raw
//!   (positive) `−Σ log P(Δt)[s,s′]`. Callers apply the same `2·` OFV-scale
//!   factor they apply to the TTE / binary terms.
//!
//! The whole module is gated behind `markov` (which implies `survival`).

use crate::markov::{ctmm_data_term, StateObs};
use crate::types::{CompiledModel, EndpointLikelihood, GeneratorFn, ObsRecord, Subject};
use std::collections::HashMap;

/// Large finite objective returned for a subject whose CTMM data term cannot be
/// evaluated cleanly — mirrors the `survival` module's `1e20` convention (repel
/// the optimizer, never `−inf`/`NaN`/panic mid-fit). Two triggers, **both of
/// which are ruled out before the fit starts** by [`validate_ctmm_states`] and
/// the shared TV-covariate guard, so this is a defensive backstop, not an
/// expected path:
///
/// - a [`crate::markov::MarkovError`] from [`ctmm_data_term`] (structural/data
///   problem — validated pre-fit), and
/// - an observed DV code not present in the endpoint's `state_codes`
///   (validated pre-fit by [`validate_ctmm_states`]).
///
/// [`ctmm_data_term`]'s own parameter-driven degeneracy (an observed transition
/// whose probability underflowed) already returns its `1e20` sentinel *inside*
/// an `Ok`, so that live optimization path flows through the normal sum below.
const SUBJECT_SENTINEL_NLL: f64 = 1e20;

/// Collect one subject's [`ObsRecord::DiscreteState`] rows on `cmt` into
/// [`StateObs`] in record order, translating each raw DV code to its 0-based
/// generator index through `state_codes`.
///
/// Returns `Err(bad_code)` if a row carries a DV code absent from `state_codes`
/// — a data/model mismatch [`validate_ctmm_states`] rejects before the fit, so
/// the caller treats it as the [`SUBJECT_SENTINEL_NLL`] backstop rather than a
/// per-iteration failure.
fn collect_state_obs(
    cmt: usize,
    state_codes: &[usize],
    records: &[ObsRecord],
) -> Result<Vec<StateObs>, usize> {
    let mut obs = Vec::new();
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
            // Map the raw DV code → generator index. `state_codes` is tiny
            // (S states); a linear scan is cheaper than a HashMap here.
            let idx = state_codes.iter().position(|&code| code == *state);
            match idx {
                Some(i) => obs.push(StateObs {
                    time: *time,
                    state: i,
                }),
                None => return Err(*state),
            }
        }
    }
    Ok(obs)
}

/// Negative CTMM log-likelihood for one subject on a single `Ctmm` endpoint:
/// build `Q(θ,η,cov)`, gather its state observations, and score the transition
/// chain with [`ctmm_data_term`]. Returns the raw NLL (positive); the caller
/// applies the OFV-scale factor.
///
/// Evaluated fresh at each `eta` — including the perturbed `eta` of the FOCEI
/// FD-Hessian closure — so the generator's η-curvature is captured without any
/// separate gradient plumbing (the FD path differentiates through
/// `generator_fn`).
#[allow(clippy::too_many_arguments)]
fn ctmm_endpoint_nll(
    cmt: usize,
    n_states: usize,
    state_codes: &[usize],
    generator_fn: &GeneratorFn,
    records: &[ObsRecord],
    covariates: &HashMap<String, f64>,
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    let obs = match collect_state_obs(cmt, state_codes, records) {
        Ok(o) => o,
        Err(_bad_code) => return SUBJECT_SENTINEL_NLL, // validated pre-fit; defensive
    };
    // Fewer than two states carry no transition — no contribution (also handled
    // inside `ctmm_data_term`, but skipping the generator build is cheaper).
    if obs.len() < 2 {
        return 0.0;
    }
    let q = generator_fn(theta, eta, covariates);
    debug_assert_eq!(
        q.nrows(),
        n_states,
        "generator_fn must return an n_states×n_states matrix"
    );
    match ctmm_data_term(&q, &obs) {
        Ok(nll) => nll,
        // Structural/data MarkovError — ruled out by validate_ctmm_states + the
        // datareader's sorted/finite-time guarantees before the fit. Repel
        // rather than panic in the hot loop.
        Err(_) => SUBJECT_SENTINEL_NLL,
    }
}

/// Sum the CTMM-endpoint NLL over every `Ctmm` CMT for one subject, mirroring the
/// per-endpoint TTE / binary dispatch. Returns `0.0` when the model has no CTMM
/// endpoint or the subject has no discrete-state records for it.
///
/// This doubles as the FOCEI FD-Hessian closure at the interaction site:
/// evaluated at a perturbed η it rebuilds `Q(η)` and re-scores every transition,
/// so `data_term_hessian_fd` picks up the CTMM curvature w.r.t. η.
pub fn ctmm_subject_nll(
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
        if let EndpointLikelihood::Ctmm {
            n_states,
            state_codes,
            generator_fn,
            ..
        } = endpoint
        {
            nll += ctmm_endpoint_nll(
                *cmt,
                *n_states,
                state_codes,
                generator_fn,
                &subject.obs_records,
                &subject.covariates,
                theta,
                eta,
            );
        }
    }
    nll
}

/// Fail-loud check that every CTMM record's observed DV code is one of the
/// endpoint's declared `state_codes`. Run once at fit setup: the datareader
/// accepts any non-negative integer DV on a discrete CMT (it cannot know which
/// codes a given endpoint declared), so the `Ctmm` endpoint must reject an
/// undeclared code itself rather than let [`ctmm_subject_nll`] silently fall back
/// to the sentinel. `records` may be a subject's full `obs_records`; only
/// `DiscreteState` rows on `cmt` are checked.
pub fn validate_ctmm_states(
    cmt: usize,
    state_codes: &[usize],
    records: &[ObsRecord],
) -> Result<(), String> {
    for r in records {
        if let ObsRecord::DiscreteState { state, cmt: c, .. } = r {
            if *c == cmt && !state_codes.contains(state) {
                let mut declared = state_codes.to_vec();
                declared.sort_unstable();
                return Err(format!(
                    "[markov_model] cmt = {cmt}: observed DV {state} is not one of the declared \
                     states {declared:?}. Every DV value on a CTMM CMT must match a `states` code."
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GeneratorFn;
    use nalgebra::DMatrix;

    /// A constant 2-state generator `Q = [[-a, a], [b, -b]]`, ignoring θ/η/cov.
    fn const_gen(a: f64, b: f64) -> GeneratorFn {
        Box::new(move |_t: &[f64], _e: &[f64], _c: &HashMap<String, f64>| {
            DMatrix::from_row_slice(2, 2, &[-a, a, b, -b])
        })
    }

    fn disc(time: f64, state: usize, cmt: usize) -> ObsRecord {
        ObsRecord::DiscreteState { time, state, cmt }
    }

    /// NLL over a record slice with no covariates and no θ/η.
    fn nll(state_codes: &[usize], gen: &GeneratorFn, recs: &[ObsRecord]) -> f64 {
        ctmm_endpoint_nll(5, 2, state_codes, gen, recs, &HashMap::new(), &[], &[])
    }

    /// Closed-form 2-state occupancy: P(t)[0,0] for Q=[[-a,a],[b,-b]] is
    /// (b + a·e^{-(a+b)t})/(a+b). The endpoint NLL of a single 0→0 gap must equal
    /// −log of that.
    #[test]
    fn ctmm_endpoint_single_gap_matches_closed_form() {
        let (a, b, dt) = (0.7_f64, 0.3, 2.0);
        let lam = a + b;
        let p00 = (b + a * (-lam * dt).exp()) / lam;
        let got = nll(
            &[0, 1],
            &const_gen(a, b),
            &[disc(0.0, 0, 5), disc(dt, 0, 5)],
        );
        assert!((got - (-p00.ln())).abs() < 1e-12, "got {got}");
    }

    /// DV codes need not be 0-based/contiguous: states coded 1/2 map to generator
    /// indices 0/1 through `state_codes`, giving the identical likelihood.
    #[test]
    fn ctmm_endpoint_maps_noncontiguous_codes() {
        let (a, b, dt) = (0.5_f64, 0.5, 1.0);
        let lam = a + b;
        // 1 → 2 transition ⇒ generator index 0 → 1 ⇒ P[0,1] = a(1-e)/lam.
        let p01 = a * (1.0 - (-lam * dt).exp()) / lam;
        let got = nll(
            &[1, 2],
            &const_gen(a, b),
            &[disc(0.0, 1, 5), disc(dt, 2, 5)],
        );
        assert!((got - (-p01.ln())).abs() < 1e-12, "got {got}");
    }

    /// A subject with a single observation carries no transition ⇒ 0 contribution.
    #[test]
    fn ctmm_endpoint_single_obs_is_zero() {
        assert_eq!(nll(&[0, 1], &const_gen(0.4, 0.6), &[disc(0.0, 0, 5)]), 0.0);
    }

    /// An undeclared DV code hits the defensive sentinel in the hot path (and is
    /// what `validate_ctmm_states` rejects up front).
    #[test]
    fn ctmm_endpoint_unknown_code_sentinels() {
        let got = nll(
            &[0, 1],
            &const_gen(0.4, 0.6),
            &[disc(0.0, 0, 5), disc(1.0, 9, 5)],
        );
        assert_eq!(got, SUBJECT_SENTINEL_NLL);
    }

    /// `validate_ctmm_states` names the CMT, the offending DV, and the declared set.
    #[test]
    fn validate_rejects_undeclared_code() {
        let recs = [disc(0.0, 0, 5), disc(1.0, 2, 5)];
        let err = validate_ctmm_states(5, &[0, 1], &recs).unwrap_err();
        assert!(err.contains("cmt = 5"), "msg: {err}");
        assert!(err.contains("DV 2"), "msg: {err}");
        assert!(err.contains("[0, 1]"), "msg: {err}");
        // A clean subject passes; rows on another CMT are ignored.
        assert!(validate_ctmm_states(5, &[0, 1], &[disc(0.0, 0, 5), disc(1.0, 7, 6)]).is_ok());
    }

    /// Rows on a second endpoint's CMT are not folded into this endpoint's chain.
    #[test]
    fn collect_state_obs_filters_by_cmt() {
        let recs = [disc(0.0, 0, 5), disc(1.0, 3, 6), disc(2.0, 1, 5)];
        let obs = collect_state_obs(5, &[0, 1], &recs).unwrap();
        assert_eq!(obs.len(), 2);
        assert_eq!(obs[0].state, 0);
        assert_eq!(obs[1].state, 1);
        assert_eq!(obs[1].time, 2.0);
    }
}
