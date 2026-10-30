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
//! "At that record" means its `TIME`, its covariates (the reader's per-row snapshot,
//! [`record_covariate_snapshots`]) and, for a drug-driven model, the ODE state at its time —
//! everything a `$PRED` reads from the current record. The previous record contributes
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
use std::collections::HashMap;

/// Round-off allowance on the probability bounds: a declared probability may sit this far
/// outside `[0, 1]`, and a row's off-diagonals may sum this far past 1, before the matrix is
/// treated as invalid. Matches the CTMM's off-diagonal tolerance.
const PROB_TOL: f64 = 1e-12;

/// One scored record of a DTMM chain.
#[derive(Clone, Copy)]
struct ChainRecord {
    /// Index into `Subject::obs_records` — and into an aligned covariate snapshot.
    idx: usize,
    /// Internal clock — occasion-shifted, and one ULP below a same-TIME dose the row precedes:
    /// what the ODE state is read at.
    time: f64,
    /// The TIME the input carried: what a probability's `TIME` reads, and what messages name.
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
            time,
            raw_time,
            state,
            cmt: c,
        } = r
        {
            if *c != cmt {
                continue;
            }
            match state_codes.iter().position(|&code| code == *state) {
                Some(i) => out.push(ChainRecord {
                    idx,
                    time: *time,
                    raw_time: *raw_time,
                    state: i,
                }),
                None => return Err(*state),
            }
        }
    }
    Ok(out)
}

/// The reader's per-row covariate snapshot of `subject`, **aligned with
/// `subject.obs_records`**, or `None` when there is none to align.
///
/// `Subject::endpoint_covariates` holds one LOCF snapshot per data row routed to a
/// non-Gaussian endpoint, in reading order, taken after the row's own cells are read — so
/// the snapshot of a state row carries that row's covariate values. It is empty when no
/// covariate varies within the subject (the baseline map is then exact). A discrete-state or
/// count row yields exactly one record and one snapshot, so the two lists are parallel —
/// unless the subject also carries time-to-event rows, which can yield zero or two records
/// per row. Those subjects, and an in-memory subject whose snapshot length does not match,
/// return `None`; `crate::api::check_survival_tv_covariates` refuses such a subject when a
/// probability covariate varies, so `None` only ever stands for "the baseline is exact".
pub(crate) fn record_covariate_snapshots(subject: &Subject) -> Option<&[HashMap<String, f64>]> {
    let snaps = &subject.endpoint_covariates;
    let aligned = !snaps.is_empty()
        && snaps.len() == subject.obs_records.len()
        && !subject
            .obs_records
            .iter()
            .any(|r| matches!(r, ObsRecord::Event { .. }));
    aligned.then_some(snaps.as_slice())
}

/// The covariate map record `rec` is evaluated with: its snapshot when aligned, else the
/// subject's baseline map.
fn record_cov<'a>(
    subject: &'a Subject,
    snaps: Option<&'a [HashMap<String, f64>]>,
    rec: &ChainRecord,
) -> &'a HashMap<String, f64> {
    snaps.map_or(&subject.covariates, |s| &s[rec.idx])
}

/// The ODE state vector at every record of `chain`, for a drug-driven DTMM: one `[odes]`
/// solve for the subject at `(θ, η)`, read at the records' times. The disposition is the
/// `$PK` snapshot at the baseline covariates and every dose quantity is read at its own dose
/// record — exactly the drug-driven CTMM's solve, with the same fit-setup guards against a
/// `TIME`- or covariate-dependent disposition it would freeze. `None` when the model has no
/// ODE system or the solve did not reach a record.
fn chain_ode_states(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    chain: &[ChainRecord],
) -> Option<Vec<Vec<f64>>> {
    let ode = model.ode_spec.as_ref()?;
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let mut pk_at_dose = Vec::new();
    crate::pk::compute_dose_pk_params_into(model, subject, theta, eta, &mut pk_at_dose);
    let times: Vec<f64> = chain.iter().map(|r| r.time).collect();
    let states = crate::ode::predictions::ode_dense_solve_states_reading(
        ode,
        &pk.values,
        crate::ode::predictions::DoseReads::PerDose(&pk_at_dose),
        subject,
        &times,
    );
    let ok = states.len() == times.len() && states.iter().flatten().all(|x| x.is_finite());
    ok.then_some(states)
}

/// The covariates record `rec`'s probabilities read: `cov`, with each referenced ODE state
/// overlaid under its name when the model is drug-driven (`state` is that record's state
/// vector). A state parses as a covariate leaf, so this is how the expression sees it.
fn with_states<'a>(
    cov: &'a HashMap<String, f64>,
    prob_states: &[(String, usize)],
    state: Option<&[f64]>,
) -> std::borrow::Cow<'a, HashMap<String, f64>> {
    match state {
        Some(st) if !prob_states.is_empty() => {
            let mut c = cov.clone();
            for (name, i) in prob_states {
                c.insert(name.clone(), st.get(*i).copied().unwrap_or(f64::NAN));
            }
            std::borrow::Cow::Owned(c)
        }
        _ => std::borrow::Cow::Borrowed(cov),
    }
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
    model: &CompiledModel,
    subject: &Subject,
    cmt: usize,
    n_states: usize,
    state_codes: &[usize],
    transitions: &[(usize, usize)],
    prob_fn: &crate::types::TransitionProbFn,
    prob_states: &[(String, usize)],
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    let Ok(chain) = collect_states(cmt, state_codes, &subject.obs_records) else {
        return SUBJECT_SENTINEL_NLL; // validated pre-fit; defensive
    };
    if chain.len() < 2 {
        return 0.0;
    }
    let ode_states = if prob_states.is_empty() {
        None
    } else {
        match chain_ode_states(model, subject, theta, eta, &chain) {
            Some(s) => Some(s),
            None => return SUBJECT_SENTINEL_NLL,
        }
    };
    let snaps = record_covariate_snapshots(subject);
    let mut nll = 0.0;
    for k in 1..chain.len() {
        let (prev, rec) = (chain[k - 1], chain[k]);
        // Everything but the previous *state* comes from the current record.
        let cov = with_states(
            record_cov(subject, snaps, &rec),
            prob_states,
            ode_states.as_ref().map(|s| s[k].as_slice()),
        );
        // `TIME` is the record's TIME as written, as a `$PRED` reads it — not the engine
        // clock, which the reader moves for record order (one ULP) and stacks across reset
        // occasions. The ODE state above was read on the engine clock.
        let probs = with_model_time(rec.raw_time, || prob_fn(theta, eta, &cov));
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
            prob_states,
            ..
        } = endpoint
        {
            nll += dtmm_endpoint_nll(
                model,
                subject,
                cmt,
                *n_states,
                state_codes,
                transitions,
                prob_fn,
                prob_states,
                theta,
                eta,
            );
        }
    }
    nll
}

/// Exact `∂/∂η` of [`dtmm_subject_nll`]: each record's probabilities are replayed over
/// `Dual1` (`EndpointExprProgram::eval_duals`, θ and η seeded, at the record's covariates
/// and time) and chained through
///
/// ```text
///   ∂/∂η [−log P_jk] = −(∂P_jk/∂η) / P_jk                       (j ≠ k)
///   ∂/∂η [−log P_jj] = (Σ_{k≠j} ∂P_jk/∂η) / P_jj,  P_jj = 1 − Σ_{k≠j} P_jk
/// ```
///
/// Returns `(value, ∂/∂η)` at the same 1× scale as the value. `None` — the caller falls
/// back to FD — when an endpoint has no dual program (including every drug-driven DTMM,
/// whose probabilities also depend on η through the ODE trajectory), or on the penalized
/// regime (a non-stochastic matrix or a zero-probability step), where the value is the flat
/// sentinel.
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
    let snaps = record_covariate_snapshots(subject);
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
        let n_theta = prog.n_theta_axis();
        // An axis layout that disagrees with the model's η count would zero-fill a
        // gradient reported as analytic. Decline instead.
        if prog.n_axes() != n_theta + n_eta {
            return None;
        }
        let chain = collect_states(cmt, state_codes, &subject.obs_records).ok()?;
        for k in 1..chain.len() {
            let (from, rec) = (chain[k - 1].state, chain[k]);
            let to = rec.state;
            let cov = record_cov(subject, snaps, &rec);
            let outs = with_model_time(rec.raw_time, || prog.eval_duals(theta, eta, cov))?;
            if outs.len() != transitions.len() {
                return None;
            }
            let probs: Vec<f64> = outs.iter().map(|(p, _)| *p).collect();
            if !is_stochastic(transitions, &probs, *n_states) {
                return None;
            }
            let p = step_probability(transitions, &probs, from, to)?;
            nll -= p.ln();
            // `∂P_obs/∂η`: the observed off-diagonal's jet, or minus the row's jets for a
            // stay. `−log` then contributes `−∂P_obs/∂η / P_obs`.
            for (&(j, k2), (_, dp)) in transitions.iter().zip(&outs) {
                if j != from || (from != to && k2 != to) {
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
