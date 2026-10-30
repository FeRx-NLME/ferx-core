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
//! evaluated at that record — so a `TIME` term reads the record's own time — and the stay
//! probability `P_jj = 1 − Σ_{k≠j} P_jk` filled in. This is the likelihood NONMEM scores
//! with a `PREVIOUS`-DV `$PRED` under `$EST … LAPLACE LIKE`.
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
//! - **Invalid probabilities are penalized, not refused.** A row whose declared
//!   probabilities leave `[0, 1]` or sum past 1 depends on the parameters, so it repels the
//!   optimizer with [`SUBJECT_SENTINEL_NLL`]. An observed transition that was never
//!   declared is a model/data mismatch no parameter value can fix, and is refused before
//!   the fit by [`validate_dtmm_records`].
//!
//! The whole module is gated behind `markov`.

use crate::parser::model_parser::with_model_time;
use crate::stats::likelihood::SUBJECT_SENTINEL_NLL;
use crate::types::{CompiledModel, EndpointLikelihood, ObsRecord, Subject};

/// Round-off allowance on the probability bounds: a declared probability may sit this far
/// outside `[0, 1]`, and a row's off-diagonals may sum this far past 1, before the row is
/// treated as invalid. Matches the CTMM's off-diagonal tolerance.
const PROB_TOL: f64 = 1e-12;

/// The state index of every `DiscreteState` row on `cmt`, in record order, paired with the
/// row's internal model time. `Err(code)` on a DV code outside `state_codes` (refused before
/// the fit by [`validate_dtmm_records`]).
fn collect_states(
    cmt: usize,
    state_codes: &[usize],
    records: &[ObsRecord],
) -> Result<Vec<(f64, usize)>, usize> {
    let mut out = Vec::new();
    for r in records {
        if let ObsRecord::DiscreteState {
            time,
            state,
            cmt: c,
            ..
        } = r
        {
            if *c != cmt {
                continue;
            }
            match state_codes.iter().position(|&code| code == *state) {
                Some(i) => out.push((*time, i)),
                None => return Err(*state),
            }
        }
    }
    Ok(out)
}

/// The probability of the observed step `from → to` given the declared probabilities
/// `probs` (parallel to `transitions`), or `None` when the `from` row is invalid — a
/// non-finite declared probability, one outside `[0, 1]`, or off-diagonals summing past 1
/// — or the observed step has probability `≤ 0`.
///
/// Only the `from` row is checked: it is the only row the step reads, which is also all a
/// `PREVIOUS`-DV `$PRED` evaluates.
fn step_probability(
    transitions: &[(usize, usize)],
    probs: &[f64],
    from: usize,
    to: usize,
) -> Option<f64> {
    let mut off_sum = 0.0;
    let mut p_to = 0.0;
    for (&(j, k), &p) in transitions.iter().zip(probs) {
        if j != from {
            continue;
        }
        if !p.is_finite() || !(-PROB_TOL..=1.0 + PROB_TOL).contains(&p) {
            return None;
        }
        off_sum += p;
        if k == to {
            p_to = p;
        }
    }
    if off_sum > 1.0 + PROB_TOL {
        return None;
    }
    let p = if from == to { 1.0 - off_sum } else { p_to };
    (p > 0.0).then_some(p)
}

/// Negative DTMM log-likelihood for one subject on one `Dtmm` endpoint. Raw (positive);
/// the caller applies the OFV-scale factor.
#[allow(clippy::too_many_arguments)]
fn dtmm_endpoint_nll(
    cmt: usize,
    state_codes: &[usize],
    transitions: &[(usize, usize)],
    prob_fn: &crate::types::TransitionProbFn,
    records: &[ObsRecord],
    covariates: &std::collections::HashMap<String, f64>,
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    let Ok(states) = collect_states(cmt, state_codes, records) else {
        return SUBJECT_SENTINEL_NLL; // validated pre-fit; defensive
    };
    let mut nll = 0.0;
    for w in states.windows(2) {
        let ((_, from), (time, to)) = (w[0], w[1]);
        // The probabilities of the *current* record: a `TIME` term reads its time.
        let probs = with_model_time(time, || prob_fn(theta, eta, covariates));
        match step_probability(transitions, &probs, from, to) {
            Some(p) => nll -= p.min(1.0).ln(),
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
            state_codes,
            transitions,
            prob_fn,
            ..
        } = endpoint
        {
            nll += dtmm_endpoint_nll(
                cmt,
                state_codes,
                transitions,
                prob_fn,
                &subject.obs_records,
                &subject.covariates,
                theta,
                eta,
            );
        }
    }
    nll
}

/// Exact `∂/∂η` of [`dtmm_subject_nll`]: each record's probabilities are replayed over
/// `Dual1` (`EndpointExprProgram::eval_duals`, θ and η seeded) and chained through
///
/// ```text
///   ∂/∂η [−log P_jk] = −(∂P_jk/∂η) / P_jk                       (j ≠ k)
///   ∂/∂η [−log P_jj] = (Σ_{k≠j} ∂P_jk/∂η) / P_jj,  P_jj = 1 − Σ_{k≠j} P_jk
/// ```
///
/// Returns `(value, ∂/∂η)` at the same 1× scale as the value. `None` — the caller falls
/// back to FD — when an endpoint has no dual program, or on the penalized regime (an
/// invalid row or a zero-probability step), where the value is the flat sentinel.
pub(crate) fn dtmm_subject_eta_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>)> {
    let n_eta = model.n_eta;
    let mut nll = 0.0;
    let mut grad = vec![0.0_f64; n_eta];
    if subject.obs_records.is_empty() {
        return Some((nll, grad));
    }
    for (cmt, endpoint) in dtmm_endpoints(model) {
        let EndpointLikelihood::Dtmm {
            state_codes,
            transitions,
            prob_program,
            ..
        } = endpoint
        else {
            continue;
        };
        let prog = prob_program.as_ref()?;
        let n_theta = prog.n_theta_axis();
        // An axis layout that disagrees with the model's η count would zero-fill a
        // gradient reported as analytic. Decline instead.
        if prog.n_axes() != n_theta + n_eta {
            return None;
        }
        let states = collect_states(cmt, state_codes, &subject.obs_records).ok()?;
        for w in states.windows(2) {
            let ((_, from), (time, to)) = (w[0], w[1]);
            let outs = with_model_time(time, || prog.eval_duals(theta, eta, &subject.covariates))?;
            if outs.len() != transitions.len() {
                return None;
            }
            let probs: Vec<f64> = outs.iter().map(|(p, _)| *p).collect();
            let p = step_probability(transitions, &probs, from, to)?;
            nll -= p.min(1.0).ln();
            // `∂P_obs/∂η`: the observed off-diagonal's jet, or minus the row's jets for a
            // stay. `−log` then contributes `−∂P_obs/∂η / P_obs`.
            for (&(j, k), (_, dp)) in transitions.iter().zip(&outs) {
                if j != from || (from != to && k != to) {
                    continue;
                }
                let sign = if from == to { 1.0 } else { -1.0 };
                for (acc, d) in grad.iter_mut().zip(&dp[n_theta..]) {
                    if !d.is_finite() {
                        return None;
                    }
                    *acc += sign * d / p;
                }
            }
        }
    }
    Some((nll, grad))
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
        let ((_, from), (time, to)) = (w[0], w[1]);
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
