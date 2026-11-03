//! Discrete-time Markov model (DTMM) endpoint — `[markov_model] type = dtmm` (Phase 4b,
//! #2473).
//!
//! The probability of the state observed at scored record `k` depends on the state
//! observed at record `k − 1`. Per subject and endpoint CMT, with `s_0, s_1, …` the
//! states of its [`ObsRecord::DiscreteState`] rows in record order,
//!
//! ```text
//!   NLL = −Σ_{k ≥ 1} log P_k[s_{k−1}, s_k]
//! ```
//!
//! where `P_k` is the transition-probability matrix **at record `k`**: the declared
//! off-diagonal probabilities `P_jk` (the `transition j -> k = <expr>` right-hand sides)
//! evaluated at that record, and the stay probability `P_jj = 1 − Σ_{k≠j} P_jk` filled in.
//! "At that record" means its `TIME` as written and its covariates (the reader's per-row
//! snapshot, `Subject::record_covariate_snapshots`) — what a `$PRED` reads from the current record.
//! A probability that reads an ODE model state (a drug-driven DTMM) is refused at parse. The previous record contributes
//! only its *state*. This is the likelihood NONMEM scores with a `PREVIOUS`-DV `$PRED` under
//! `$EST … LAPLACE LIKE`.
//!
//! Conventions, each a deliberate answer to an open question of #2473:
//!
//! - **The first record is conditioned on.** It contributes nothing (as NONMEM's
//!   `PREVIOUS`-DV models do: the first record of a subject has no previous state). There
//!   is no `init` / stationary distribution scoring.
//! - **A step is a record, not a time unit.** `P_k` does not depend on `Δt`; the chain
//!   links consecutive scored records whatever their spacing, exactly as a `PREVIOUS`-DV
//!   `$PRED` does. A model that needs `Δt` can read `TIME`.
//! - **A missing state does not break the chain.** A row with a missing DV is not a
//!   scored record (the reader drops it as `MDV = 1`), so the next scored record links to
//!   the last *observed* state — again the `PREVIOUS`-DV behaviour.
//! - **Invalid probabilities are penalized, not refused.** `P_k` must be a stochastic
//!   matrix — *every* row, not only the one the step reads: a declared probability outside
//!   `[0, 1]` (or non-finite) or a row whose off-diagonals sum past 1 anywhere in `P_k`
//!   depends on the parameters, so it repels the optimizer with [`SUBJECT_SENTINEL_NLL`]. An observed transition that was never
//!   declared is a model/data mismatch no parameter value can fix, and is refused before
//!   the fit by [`validate_dtmm_records`].
//!
//! The whole module is gated behind `markov`.

use crate::parser::model_parser::with_model_time;
use crate::stats::likelihood::SUBJECT_SENTINEL_NLL;
use crate::types::{CompiledModel, EndpointLikelihood, ObsRecord, Subject};

/// Round-off allowance on the probability bounds: a declared probability may sit this far
/// outside `[0, 1]`, and a row's off-diagonals may sum this far past 1, before the matrix is
/// treated as invalid. Matches the CTMM's off-diagonal tolerance.
const PROB_TOL: f64 = 1e-12;

/// One scored record of a DTMM chain.
#[derive(Clone, Copy)]
struct ChainRecord {
    /// Index into `Subject::obs_records` — and into an aligned covariate snapshot.
    idx: usize,
    /// The TIME the input carried (not the reader's occasion-shifted internal clock): what a probability's `TIME` reads, and what messages name.
    raw_time: f64,
    /// 0-based state index.
    state: usize,
}

/// Every `DiscreteState` row on `cmt`, in record order. `Err(code)` on a DV code outside
/// `state_codes` (refused before the fit by [`validate_dtmm_records`]).
fn collect_states(
    cmt: usize,
    state_codes: &[usize],
    records: &[ObsRecord],
) -> Result<Vec<ChainRecord>, usize> {
    let mut out = Vec::new();
    for (idx, r) in records.iter().enumerate() {
        if let ObsRecord::DiscreteState {
            raw_time,
            state,
            cmt: c,
            ..
        } = r
        {
            if *c != cmt {
                continue;
            }
            match state_codes.iter().position(|&code| code == *state) {
                Some(i) => out.push(ChainRecord {
                    idx,
                    raw_time: *raw_time,
                    state: i,
                }),
                None => return Err(*state),
            }
        }
    }
    Ok(out)
}

/// Whether the declared probabilities `probs` (parallel to `transitions`) define a
/// stochastic `n_states × n_states` matrix: every one finite and in `[0, 1]`, and every
/// row's off-diagonals summing to at most 1 (so every stay probability is in `[0, 1]`).
fn is_stochastic(transitions: &[(usize, usize)], probs: &[f64], n_states: usize) -> bool {
    let mut row_sum = vec![0.0_f64; n_states];
    for (&(j, _), &p) in transitions.iter().zip(probs) {
        if !p.is_finite() || !(-PROB_TOL..=1.0 + PROB_TOL).contains(&p) {
            return false;
        }
        row_sum[j] += p;
    }
    row_sum.iter().all(|&s| s <= 1.0 + PROB_TOL)
}

/// The probability of the observed step `from → to` in a matrix already checked by
/// [`is_stochastic`]: the declared off-diagonal, or the filled-in stay probability. `None`
/// when it is `≤ 0` — an observed step the parameters make impossible.
fn step_probability(
    transitions: &[(usize, usize)],
    probs: &[f64],
    from: usize,
    to: usize,
) -> Option<f64> {
    let mut off_sum = 0.0;
    let mut p_to = 0.0;
    for (&(j, k), &p) in transitions.iter().zip(probs) {
        if j == from {
            off_sum += p;
            if k == to {
                p_to = p;
            }
        }
    }
    let p = if from == to { 1.0 - off_sum } else { p_to };
    (p > 0.0).then_some(p.min(1.0))
}

/// Negative DTMM log-likelihood for one subject on one `Dtmm` endpoint. Raw (positive);
/// the caller applies the OFV-scale factor.
#[allow(clippy::too_many_arguments)]
fn dtmm_endpoint_nll(
    subject: &Subject,
    cmt: usize,
    n_states: usize,
    state_codes: &[usize],
    transitions: &[(usize, usize)],
    prob_fn: &crate::types::TransitionProbFn,
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    let Ok(chain) = collect_states(cmt, state_codes, &subject.obs_records) else {
        return SUBJECT_SENTINEL_NLL; // validated pre-fit; defensive
    };
    if chain.len() < 2 {
        return 0.0;
    }
    let snaps = subject.record_covariate_snapshots();
    let mut nll = 0.0;
    for k in 1..chain.len() {
        let (prev, rec) = (chain[k - 1], chain[k]);
        // Everything but the previous *state* comes from the current record.
        let cov = subject.record_cov(snaps, rec.idx);
        // `TIME` is the record's TIME as written, as a `$PRED` reads it — not the engine
        // clock, which the reader stacks across reset occasions.
        let probs = with_model_time(rec.raw_time, || prob_fn(theta, eta, cov));
        if !is_stochastic(transitions, &probs, n_states) {
            return SUBJECT_SENTINEL_NLL;
        }
        match step_probability(transitions, &probs, prev.state, rec.state) {
            Some(p) => nll -= p.ln(),
            None => return SUBJECT_SENTINEL_NLL,
        }
    }
    nll
}

/// The `Dtmm` endpoints of `model`, in ascending CMT order, so a multi-endpoint sum is
/// reproducible across processes (`model.endpoints` is a `HashMap`).
fn dtmm_endpoints(model: &CompiledModel) -> impl Iterator<Item = (usize, &EndpointLikelihood)> {
    model
        .dtmm_cmts()
        .into_iter()
        .filter_map(move |cmt| model.endpoints.get(&cmt).map(|ep| (cmt, ep)))
}

/// Sum the DTMM-endpoint NLL over every `Dtmm` CMT for one subject. `0.0` when the model
/// has no DTMM endpoint or the subject no records for one.
///
/// Reached through `crate::categorical::discrete_subject_nll`, the single discrete-endpoint
/// seam, so every FOCEI / IOV / SAEM / IMP path and the FD-Hessian closure score it with no
/// dispatch edit of its own.
pub(crate) fn dtmm_subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    if subject.obs_records.is_empty() {
        return 0.0;
    }
    let mut nll = 0.0;
    for (cmt, endpoint) in dtmm_endpoints(model) {
        if let EndpointLikelihood::Dtmm {
            n_states,
            state_codes,
            transitions,
            prob_fn,
            ..
        } = endpoint
        {
            nll += dtmm_endpoint_nll(
                subject,
                cmt,
                *n_states,
                state_codes,
                transitions,
                prob_fn,
                theta,
                eta,
            );
        }
    }
    nll
}

/// Every `Dtmm` endpoint's data term on the subject as a function of `(θ, η)`, on any number
/// type: each record's probabilities replayed through the endpoint's program (at the record's
/// covariates and written `TIME`) and scored as `−log P_obs`, where `P_obs` is the declared
/// off-diagonal or the stay probability `1 − Σ_{k≠j} P_jk`. The `f64` instance is
/// [`dtmm_subject_nll`]; a first-order η-jet is the inner gradient, a second-order one the
/// Laplace η-Hessian, and tape leaves the outer θ-gradient, all through
/// [`crate::categorical::DiscreteTerm`].
///
/// `None` — the caller finite-differences — when an endpoint has no dual program, or in the
/// penalized regime (a non-stochastic matrix or a zero-probability step), where the value is
/// the flat sentinel and has no derivative. Validity is judged on the values alone, exactly as
/// the value path judges it.
pub(crate) struct DtmmTerm<'a> {
    pub(crate) model: &'a CompiledModel,
    pub(crate) subject: &'a Subject,
}

impl crate::survival::SubjectTerm for DtmmTerm<'_> {
    fn eval<T: crate::sens::num::vm::VmNum>(&self, th: &[T], et: &[T]) -> Option<T> {
        let (model, subject) = (self.model, self.subject);
        let mut nll = T::from_f64(0.0);
        if subject.obs_records.is_empty() {
            return Some(nll);
        }
        let snaps = subject.record_covariate_snapshots();
        for (cmt, endpoint) in dtmm_endpoints(model) {
            let EndpointLikelihood::Dtmm {
                n_states,
                state_codes,
                transitions,
                prob_program,
                ..
            } = endpoint
            else {
                continue;
            };
            let prog = prob_program.as_ref()?;
            let chain = collect_states(cmt, state_codes, &subject.obs_records).ok()?;
            for k in 1..chain.len() {
                let (from, rec) = (chain[k - 1].state, chain[k]);
                let to = rec.state;
                let cov = subject.record_cov(snaps, rec.idx);
                let probs = with_model_time(rec.raw_time, || prog.eval_g(th, et, cov))?;
                if probs.len() != transitions.len() {
                    return None;
                }
                let vals: Vec<f64> = probs.iter().map(|p| p.val()).collect();
                if !is_stochastic(transitions, &vals, *n_states) {
                    return None;
                }
                step_probability(transitions, &vals, from, to)?;
                // `P_obs`: the observed off-diagonal, or one minus the row's off-diagonals.
                let mut p_obs = T::from_f64(if from == to { 1.0 } else { 0.0 });
                for (&(j, k2), &p) in transitions.iter().zip(&probs) {
                    if j != from {
                        continue;
                    }
                    if from == to {
                        p_obs = p_obs - p;
                    } else if k2 == to {
                        p_obs = p_obs + p;
                    }
                }
                nll = nll - p_obs.ln();
            }
        }
        Some(nll)
    }
}

/// Exact `∂/∂η` of [`dtmm_subject_nll`] — [`DtmmTerm`] at fixed θ on a first-order η-jet —
/// at the same 1× scale as the value. `None` as for [`DtmmTerm`].
#[cfg(test)]
pub(crate) fn dtmm_subject_eta_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>)> {
    let f = crate::survival::FixedTheta {
        theta,
        term: &DtmmTerm { model, subject },
    };
    if eta.is_empty() {
        use crate::sens::eta_jets::EtaFunction;
        return Some((f.eval::<f64>(&[])?, Vec::new()));
    }
    crate::survival::eta_gradient_by_bucket(eta, &f)
}

/// Whether every `Dtmm` endpoint of `model` carries a dual-evaluable probability program,
/// i.e. [`dtmm_subject_eta_grad`] can serve it. Vacuously `true` with none.
pub(crate) fn all_dtmm_endpoints_analytic(model: &CompiledModel) -> bool {
    model.endpoints.values().all(|ep| match ep {
        EndpointLikelihood::Dtmm { prob_program, .. } => prob_program.is_some(),
        _ => true,
    })
}

/// Fail-loud check of one subject's DTMM records on `cmt`, run once at fit setup:
///
/// - every observed DV is a declared state code;
/// - observation times are non-decreasing in record order (the reader keeps observation
///   rows in file order, and the chain is the record order); and
/// - every observed change of state `j → k` is a declared `transition` — an undeclared
///   step has probability 0 for **every** parameter value, so the subject could only ever
///   score the sentinel, and no optimizer can fix that.
pub fn validate_dtmm_records(
    cmt: usize,
    state_codes: &[usize],
    transitions: &[(usize, usize)],
    records: &[ObsRecord],
) -> Result<(), String> {
    crate::markov::endpoint::validate_ctmm_states(cmt, state_codes, records)?;
    crate::markov::endpoint::validate_ctmm_times(cmt, records)?;
    let states = collect_states(cmt, state_codes, records)
        .expect("codes were validated just above, so every row maps to a state");
    for w in states.windows(2) {
        let (from, to, time) = (w[0].state, w[1].state, w[1].raw_time);
        if from != to && !transitions.contains(&(from, to)) {
            return Err(format!(
                "[markov_model] cmt = {cmt} (type = dtmm): the data step from state {} to state \
                 {} (at TIME {time}) is not a declared transition, so it has probability 0 \
                 whatever the parameters. Declare `transition` lines for every change of state \
                 the data contain.",
                state_codes[from], state_codes[to]
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "dtmm_tests.rs"]
mod tests;
