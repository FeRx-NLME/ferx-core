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
use crate::stats::likelihood::SUBJECT_SENTINEL_NLL;
use crate::types::{
    CompiledModel, DiscreteObsDiagnostic, EndpointLikelihood, EndpointPredictionResult,
    LinearPredictorFn, LinkFn, ObsRecord, Prediction, Subject,
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
///
/// The same function as the `[covariate_nn]` sigmoid, and the same code: it delegates to
/// `nn::activation::sigmoid`, which is compiled in every build (#1331), so the two cannot drift.
#[inline]
fn inv_logit(x: f64) -> f64 {
    crate::nn::activation::sigmoid(x)
}

/// One binary record's negative log-likelihood at linear predictor `lp`, over any
/// [`PkNum`](crate::sens::num::PkNum): `softplus(lp) − y·lp` under the logit link. The
/// value path ([`binary_data_term`]) is the `f64` instance; [`discrete_subject_eta_hessian`]
/// runs it on η-jets (#2437). `softplus` enters through its value and first four
/// derivatives (`σ`, `σ(1−σ)`, `σ(1−σ)(1−2σ)`, `σ(1−σ)(1−6σ+6σ²)`): the gradient's
/// third-order jets and the covariance step's nested duals read past the second (#2482),
/// which a second-order lift would leave zero. The `f64` instance is the stable form above,
/// bit for bit.
///
/// Under the probit link (#2603) the record term is `−log Φ(s·lp)` with `s = 2y − 1`, by
/// `Φ(−x) = 1 − Φ(x)`: one function of `x = s·lp` for both outcomes, lifted from
/// [`neg_log_normal_cdf_jet`], whose value and derivatives never form `Φ` in the lower tail
/// (below `x = −2`), where it would underflow.
#[inline]
fn binary_record_nll_g<T: crate::sens::num::PkNum>(link: LinkFn, lp: T, y: f64) -> T {
    match link {
        LinkFn::Probit => {
            let s = if y > 0.5 { 1.0 } else { -1.0 };
            let [v, d1, d2, d3, d4] = neg_log_normal_cdf_jet(s * lp.val());
            // d^k/dlp^k g(s·lp) = s^k g^(k)(s·lp), and s^2 = 1.
            lp.lift4(v, s * d1, d2, s * d3, d4)
        }
        LinkFn::Logit => {
            let x = lp.val();
            let p = inv_logit(x);
            let q = p * (1.0 - p);
            lp.lift4(
                softplus(x),
                p,
                q,
                q * (1.0 - 2.0 * p),
                q * (1.0 - 6.0 * p + 6.0 * p * p),
            ) - T::from_f64(y) * lp
        }
    }
}

/// Below this `x`, [`neg_log_normal_cdf_jet`] takes the lower-tail continued fraction.
const PROBIT_CF_BELOW: f64 = -2.0;

/// `g(x) = −log Φ(x)` and its first four derivatives, the probit record term of
/// [`binary_record_nll_g`] (#2603). Accurate to `f64` rounding in both tails, which the
/// direct recursion in the inverse Mills ratio `m = φ/Φ` is not.
///
/// **From `x = −2` up**, `Φ` is at least 0.02, so `m = φ(x)/Φ(x)` is formed directly and the
/// derivatives follow from `m′ = −m(x + m)` (no cancellation there, since `x + m > 0` is not
/// small):
///
/// ```text
/// g′ = −m,   g″ = −m′,   g‴ = −m″,   g⁗ = −m‴
/// m″ = −m′(x + m) − m(1 + m′)
/// m‴ = −m″(x + m) − 2m′(1 + m′) − m·m″
/// ```
///
/// The value is `−ln Φ(x)` for `x ≤ 0` and `−ln_1p(−Φ(−x))` above, so the upper tail keeps
/// its digits (`Φ(30)` is `1` in `f64`, but `−log Φ(30) = 4.9e-198`).
///
/// **Below `x = −2`** that recursion cancels: with `t = −x`, `m ≈ t + 1/t`, so `x + m ≈ 1/t`
/// and `g‴ ≈ −2/t³` come out of differences of `O(t)` numbers, all digits lost by
/// `t ≈ 40`. There `g = t²/2 + ½ln 2π + L(t)`, `L = −ln R` with `R = Q/φ` the Mills ratio,
/// and `L′(t) = m − t = a` is the Laplace continued fraction
/// `a = 1/(t + 2/(t + 3/(t + …)))`, a sum of positive terms. It is evaluated on a third-order
/// Taylor jet in `t`, which gives `a`, `a′`, `a″`, `a‴` without a single subtraction, and
/// `d/dx = −d/dt` maps them over: `g′ = −(t + a)`, `g″ = 1 + a′`, `g‴ = −a″`, `g⁗ = a‴`,
/// value `t²/2 + ½ln 2π + ln(t + a)`. Nothing forms `Φ`, so nothing underflows, however far
/// out `x` is.
pub(crate) fn neg_log_normal_cdf_jet(x: f64) -> [f64; 5] {
    use crate::stats::special::normal_cdf;
    if x < PROBIT_CF_BELOW {
        use crate::sens::taylor::Taylor;
        let t = -x;
        let a = crate::stats::special::mills_excess(Taylor::<1, 3, 4>::seed(t, 0));
        // `c[k]` is the k-th derivative over k!.
        let (a0, a1, a2, a3) = (a.c[0], a.c[1], 2.0 * a.c[2], 6.0 * a.c[3]);
        let value = 0.5 * t * t + 0.5 * std::f64::consts::TAU.ln() + (t + a0).ln();
        return [value, -(t + a0), 1.0 + a1, -a2, a3];
    }
    let phi = (-0.5 * x * x).exp() / std::f64::consts::TAU.sqrt();
    let cdf = normal_cdf(x);
    let value = if x > 0.0 {
        -(-normal_cdf(-x)).ln_1p()
    } else {
        -cdf.ln()
    };
    let m = phi / cdf;
    if m == 0.0 {
        // `φ` has underflowed (from `x ≈ 38.6`). Every derivative is `m` times a polynomial
        // in `x` and `m`, so its limit is `0`; at `x = +∞` the recursion would form `0·∞`
        // (#2772).
        return [value, 0.0, 0.0, 0.0, 0.0];
    }
    let a = x + m;
    let m1 = -m * a;
    let m2 = -m1 * a - m * (1.0 + m1);
    let m3 = -m2 * a - 2.0 * m1 * (1.0 + m1) - m * m2;
    [value, -m, -m1, -m2, -m3]
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
/// They share `lp`, not the mapping from `lp` to an outcome: [`binary_data_term`] repels
/// the optimizer with the `1e20` sentinel for **any** non-finite `lp`, whereas
/// [`simulate_binary`] treats `±∞` as a legitimate degenerate `p = 1/0` and draws from it
/// (only NaN is skipped). So a saturated predictor is sampled but scored as a repel — the
/// one place the two deliberately differ.
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
    snaps: Option<&[HashMap<String, f64>]>,
    baseline: &HashMap<String, f64>,
    mut f: F,
) where
    F: FnMut(BinaryRecordLp<'_>),
{
    for (idx, r) in records.iter().enumerate() {
        if let ObsRecord::DiscreteState {
            time,
            raw_time,
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
            // set/restore). The guard gets the **internal** clock (what the engine and
            // the likelihood use); `raw_time` is carried alongside purely for reporting.
            // Each record reads its own covariate snapshot when the subject carries one
            // aligned with its records (#2562): a time-varying covariate, or a θ level
            // index that changes between visits. Otherwise the baseline map is exact.
            let covariates = snaps.and_then(|s| s.get(idx)).unwrap_or(baseline);
            let lp = with_model_time(*time, || lp_fn(theta, eta, covariates));
            f(BinaryRecordLp {
                model_time: *time,
                covariates,
                raw_time: *raw_time,
                state: *state,
                lp,
            });
        }
    }
}

/// One binary record as seen by [`for_each_binary_lp`].
///
/// `raw_time` is the **user clock** — the TIME the input CSV carried — so every
/// reported row (sdtab, `simulate()`, `predict_categorical`) joins back to the input
/// and to covtab. It differs from the engine's internal clock whenever a subject has
/// reset-delimited occasions, where the internal timeline is shifted to stay monotonic.
/// The predictor itself is always evaluated on the internal clock; only reporting uses
/// this field.
struct BinaryRecordLp<'a> {
    /// Internal (occasion-shifted) clock — what `with_model_time` must be given for any
    /// *further* predictor evaluation on this record, so it agrees with `lp` above.
    model_time: f64,
    /// The covariate map `lp` was evaluated with, for any further predictor evaluation on
    /// this record.
    covariates: &'a HashMap<String, f64>,
    raw_time: f64,
    state: usize,
    lp: f64,
}

/// The binary endpoints of `model`, in ascending CMT order.
///
/// `model.endpoints` is a `BTreeMap` (#2531), so it already iterates in ascending CMT
/// order; this helper exists to filter to the binary family. RNG consumption order,
/// emitted row order, and therefore `simulate_with_seed`'s reproducibility all depend on
/// visiting endpoints in that fixed order.
fn binary_endpoints(model: &CompiledModel) -> impl Iterator<Item = (usize, &EndpointLikelihood)> {
    model
        .binary_cmts()
        .into_iter()
        .filter_map(move |cmt| model.endpoints.get(&cmt).map(|ep| (cmt, ep)))
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
    snaps: Option<&[HashMap<String, f64>]>,
    baseline: &HashMap<String, f64>,
) -> f64 {
    let mut nll = 0.0;
    for_each_binary_lp(lp_fn, cmt, records, theta, eta, snaps, baseline, |r| {
        let (state, lp) = (r.state, r.lp);
        debug_assert!(state <= 1, "binary state must be 0/1 (validated pre-fit)");
        if !lp.is_finite() {
            // An overflowed predictor (extreme θ / covariate) makes
            // `softplus(±∞) − y·(±∞)` a NaN/∞ that the FOCEI outer objective
            // (`foce_subject_nll_interaction_with_tte`) does not guard at its final
            // return — repel the optimizer with the survival module's `1e20`
            // sentinel instead (fail-loud-but-alive), as `tte_data_term` does for
            // its own degenerate cases.
            nll += SUBJECT_SENTINEL_NLL;
            return;
        }
        let y = (state as f64).min(1.0);
        // p = inv_logit(lp): −[y·log p + (1−y)·log(1−p)] = softplus(lp) − y·lp.
        nll += binary_record_nll_g(link, lp, y);
    });
    nll
}

/// `P(Y = 1)` for one record's linear predictor under `link`.
#[inline]
fn binary_prob(link: LinkFn, lp: f64) -> f64 {
    match link {
        LinkFn::Logit => inv_logit(lp),
        LinkFn::Probit => crate::stats::special::pnorm(lp),
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
/// closure at the FOCEI interaction site, for the subjects [`discrete_subject_eta_hessian`]
/// declines (#2437): evaluated at a perturbed η it re-scores every discrete record, so
/// `data_term_hessian_fd` picks up the curvature w.r.t. η.
pub(crate) fn discrete_subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    if subject.obs_records.is_empty() {
        return 0.0;
    }
    // The discrete-time Markov endpoint (#2473) is the second family on this seam.
    #[cfg(feature = "markov")]
    let dtmm = crate::markov::dtmm::dtmm_subject_nll(model, subject, theta, eta);
    #[cfg(not(feature = "markov"))]
    let dtmm = 0.0;
    dtmm + binary_subject_nll(model, subject, theta, eta)
}

/// The binary / Bernoulli half of [`discrete_subject_nll`].
pub(crate) fn binary_subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    // Hot path: `discrete_subject_nll` runs per subject per FOCEI inner iteration and per
    // FD-Hessian stencil point, so it must not allocate. Find the binary endpoints by
    // scanning the map directly — no `Vec`, no sort. With a single binary endpoint (every
    // real model today) score it straight through; only a genuinely multi-endpoint model
    // pays for `binary_endpoints`' sorted walk, whose determinism keeps the summed OFV
    // reproducible across processes. (An earlier version wrote `let cmts = binary_cmts()`
    // here, which allocated + sorted on every call — the exact cost this avoids.)
    let mut binaries = model.endpoints.iter().filter_map(|(cmt, ep)| match ep {
        EndpointLikelihood::Binary { link, lp_fn, .. } => Some((*cmt, link, lp_fn)),
        _ => None,
    });
    match (binaries.next(), binaries.next()) {
        (None, _) => return 0.0, // no binary endpoint
        (Some((cmt, link, lp_fn)), None) => {
            // Exactly one — no ordering question, so skip the sorted walk entirely.
            return binary_data_term(
                *link,
                lp_fn,
                cmt,
                &subject.obs_records,
                theta,
                eta,
                subject.record_covariate_snapshots(),
                &subject.covariates,
            );
        }
        _ => {} // two or more — fall through to the deterministic walk below
    }

    let mut nll = 0.0;
    for (cmt, endpoint) in binary_endpoints(model) {
        // Binary today; a new discrete family (ordinal / Poisson / negative-binomial …) adds
        // its branch here — reached through this one function, so no likelihood dispatch site
        // changes. (Gaussian is scored by the residual path, TTE by the hazard dispatch.)
        if let EndpointLikelihood::Binary { link, lp_fn, .. } = endpoint {
            // `binary_data_term` filters `obs_records` by `cmt` itself, so no records
            // are copied. The sum below is over f64s, so its order must be fixed for the
            // OFV to be bit-reproducible on a multi-endpoint model: `binary_endpoints`
            // walks ascending CMT, the same order `model.endpoints` iterates (#2531).
            nll += binary_data_term(
                *link,
                lp_fn,
                cmt,
                &subject.obs_records,
                theta,
                eta,
                subject.record_covariate_snapshots(),
                &subject.covariates,
            );
        }
    }
    nll
}

/// Exact `∂/∂η` of [`discrete_subject_nll`] — the analytic counterpart of
/// finite-differencing the whole Bernoulli data term (#2024 phase 1).
///
/// Each binary endpoint's linear predictor is replayed over a first-order dual in η alone
/// (`EndpointExprProgram::eval_g`, θ constant), and the logit data term — differentiated
/// on the same dual through a lifted softplus — has the score:
///
/// ```text
///   ∂/∂η [softplus(lp) − y·lp] = (p − y)·∂lp/∂η,    p = inv_logit(lp)
/// ```
///
/// Returns `(value, ∂/∂η)` with the gradient at the **same 1× scale as the value** — the
/// caller applies the objective's factor, as for
/// [`ctmm_subject_eta_grad`](crate::markov::endpoint::ctmm_subject_eta_grad). The value is
/// the same sum [`discrete_subject_nll`] returns.
///
/// `None` — caller falls back to FD for this point — when *any* binary endpoint on the
/// subject cannot be served exactly: no dual program (an unbindable identifier), or a
/// non-finite predictor on some record, where the value is the flat `1e20` repellent with
/// no meaningful derivative.
///
/// Only the **η** block is returned (the inner EBE loop is the consumer).
pub(crate) fn discrete_subject_eta_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>)> {
    if subject.obs_records.is_empty() {
        return Some((0.0, vec![0.0; model.n_eta]));
    }
    let term = DiscreteTerm { model, subject };
    let nll = crate::survival::FixedTheta { theta, term: &term };
    if eta.is_empty() {
        use crate::sens::eta_jets::EtaFunction;
        return Some((nll.eval::<f64>(&[])?, Vec::new()));
    }
    // η-only: the program reads θ as constants, so the dual carries one lane per random
    // effect and no width cap applies.
    crate::survival::eta_gradient_by_bucket(eta, &nll)
}

/// Every discrete endpoint's data term on the subject as a function of `(θ, η)`, on any
/// number type: each binary endpoint's linear predictor per record through
/// `softplus(lp) − y·lp`, plus the DTMM's `−log P_obs` per step
/// ([`crate::markov::dtmm::DtmmTerm`], #2473) — the same sum [`discrete_subject_nll`] scores.
pub(crate) struct DiscreteTerm<'a> {
    pub(crate) model: &'a CompiledModel,
    pub(crate) subject: &'a Subject,
}

impl crate::survival::SubjectTerm for DiscreteTerm<'_> {
    fn eval<T: crate::sens::num::vm::VmNum>(&self, th: &[T], et: &[T]) -> Option<T> {
        let mut nll = T::from_f64(0.0);
        for (cmt, endpoint) in binary_endpoints(self.model) {
            let EndpointLikelihood::Binary {
                link, lp_program, ..
            } = endpoint
            else {
                continue;
            };
            let prog = lp_program.as_ref()?;
            let snaps = self.subject.record_covariate_snapshots();
            for (idx, r) in self.subject.obs_records.iter().enumerate() {
                let ObsRecord::DiscreteState {
                    time,
                    state,
                    cmt: c,
                    ..
                } = r
                else {
                    continue;
                };
                if *c != cmt {
                    continue;
                }
                // The value path's map (`for_each_binary_lp`), so the two cannot drift.
                let cov = self.subject.record_cov(snaps, idx);
                let lp = with_model_time(*time, || prog.eval_g(th, et, cov))?
                    .into_iter()
                    .next()?;
                if !lp.val().is_finite() {
                    return None;
                }
                // The same per-record kernel the value path runs (#2437).
                nll = nll + binary_record_nll_g(*link, lp, (*state as f64).min(1.0));
            }
        }
        #[cfg(feature = "markov")]
        if self.model.has_dtmm() {
            nll = nll
                + crate::markov::dtmm::DtmmTerm {
                    model: self.model,
                    subject: self.subject,
                }
                .eval(th, et)?;
        }
        Some(nll)
    }
}

/// Exact `(value, ∂²/∂η²)` of [`discrete_subject_nll`] for the FOCEI Laplace term (#2437):
/// [`DiscreteTerm`] at fixed θ on η-jets, at the same 1× scale as the value. `None` — the
/// caller finite-differences the Hessian — when a binary endpoint has no dual program, a
/// predictor is non-finite, or η is past the widest dispatch bucket.
pub(crate) fn discrete_subject_eta_hessian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, nalgebra::DMatrix<f64>)> {
    crate::sens::eta_jets::eta_hessian(
        &crate::survival::FixedTheta {
            theta,
            term: &DiscreteTerm { model, subject },
        },
        eta,
    )
}

/// Whether [`discrete_subject_eta_grad`] can serve every binary endpoint of `model`: each
/// carries a dual-evaluable linear-predictor program. The model-level counterpart of the
/// per-subject routing check. Vacuously `true` with no binary endpoint.
pub(crate) fn all_binary_endpoints_analytic(model: &CompiledModel) -> bool {
    model.endpoints.values().all(|ep| match ep {
        EndpointLikelihood::Binary { lp_program, .. } => lp_program.is_some(),
        _ => true,
    })
}

/// Whether [`discrete_subject_eta_grad`] can serve every discrete endpoint of `model` —
/// binary and DTMM alike (#2473). Vacuously `true` with none.
pub(crate) fn all_discrete_endpoints_analytic(model: &CompiledModel) -> bool {
    #[cfg(feature = "markov")]
    if !crate::markov::dtmm::all_dtmm_endpoints_analytic(model) {
        return false;
    }
    all_binary_endpoints_analytic(model)
}

/// Draw a simulated outcome for every binary record of every `Binary` endpoint, pushing
/// one [`SimOutcome::Category`](crate::types::SimOutcome::Category) row per record (Slice 1b, §8.8.2).
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
    for (cmt, endpoint) in binary_endpoints(model) {
        let EndpointLikelihood::Binary { link, lp_fn, .. } = endpoint else {
            continue;
        };
        let mut n_nonfinite = 0usize;
        for_each_binary_lp(
            lp_fn,
            cmt,
            &subject.obs_records,
            theta,
            eta,
            subject.record_covariate_snapshots(),
            &subject.covariates,
            |r| {
                let lp = r.lp;
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
                    // Raw data TIME, matching the Gaussian rows and sdtab.
                    time: r.raw_time,
                    cmt,
                    ipred: p,
                    outcome: crate::types::SimOutcome::Category { state },
                    mixnum: None, // stamped by the caller that holds the class (#2372)
                });
            },
        );
        if n_nonfinite > 0 {
            warnings.push(format!(
                "[binary_model] cmt = {cmt}: subject {} has {n_nonfinite} record(s) whose linear \
                 predictor evaluated to NaN — those rows were skipped and carry no simulated \
                 outcome. Check the linear predictor (`logit =` / `probit =`) and the subject's \
                 covariates.",
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
    for (cmt, endpoint) in binary_endpoints(model) {
        let EndpointLikelihood::Binary { link, lp_fn, .. } = endpoint else {
            continue;
        };
        for_each_binary_lp(
            lp_fn,
            cmt,
            &subject.obs_records,
            theta,
            eta,
            subject.record_covariate_snapshots(),
            &subject.covariates,
            |r| {
                let p = binary_prob(*link, r.lp);
                results.push(EndpointPredictionResult {
                    id: subject.id.clone(),
                    cmt,
                    time: r.raw_time,
                    // Clamped like every other consumer: `validate_binary_states` runs
                    // at fit setup only, so an out-of-range DV can reach this public
                    // entry point and must not escape as `observed: Some(2.0)` against
                    // a two-element `probs` (breaking the "index is the DV code" contract).
                    observed: Some((r.state as f64).min(1.0)),
                    prediction: Prediction::CatProbs {
                        probs: vec![1.0 - p, p],
                    },
                });
            },
        );
    }
}

/// Standardized (Pearson) residual for a binary record: `(y − p) / √(p·q)`, where
/// `q = 1 − p` is supplied **independently** rather than recomputed (§8.8.5).
///
/// Taking `q` as its own argument is the whole point. `inv_logit` saturates to exactly
/// `1.0` once `lp ≳ 36.7`, so a caller computing `1.0 - p` there gets exactly `0` to
/// catastrophic cancellation and the residual collapses to NaN — blanking precisely the
/// most extreme, most diagnostic observations, and only on the *positive* tail (the
/// negative tail keeps full resolution because `1 − p ≈ 1` there). Evaluating `q` as
/// `inv_logit(−lp)` instead preserves it down to ~1e-308, so the two tails behave
/// symmetrically. The likelihood has no such cliff — `softplus` resolves `lp = 40` to
/// 4.2e-18 — so recomputing `q` here would make the diagnostics and the OFV disagree
/// about which records are informative.
///
/// Returns NaN only when the variance genuinely underflows to zero, where the residual
/// is undefined; callers surface that as an empty sdtab cell rather than substituting a
/// finite value.
pub fn binary_pearson_residual(y: f64, p: f64, q: f64) -> f64 {
    let v = p * q;
    if !(v > 0.0) {
        return f64::NAN;
    }
    (y - p) / v.sqrt()
}

/// Post-fit per-record diagnostics for one subject's binary endpoints (§8.8.5): the
/// population prediction at `η = 0`, the individual prediction at the subject's EBE `η`,
/// and the standardized residual between the observed 0/1 and the individual `p`.
///
/// Returns an empty vec for a subject with no binary records, so a Gaussian-only or
/// TTE-only fit carries nothing and its output is unchanged.
pub fn binary_diagnostics(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Vec<DiscreteObsDiagnostic> {
    let mut rows = Vec::new();
    if subject.obs_records.is_empty() {
        return rows;
    }
    let zero_eta = vec![0.0_f64; eta.len()];
    for (cmt, endpoint) in binary_endpoints(model) {
        let EndpointLikelihood::Binary { link, lp_fn, .. } = endpoint else {
            continue;
        };
        // One pass, computing both predictions per record. An earlier version walked the
        // records twice (once at η = 0 for PRED, once at the EBE for IPRED) and paired
        // them by index; that pairing was only incidentally correct, and the fallback it
        // needed was dead code that would have masked a divergence rather than failing
        // loudly. Evaluating both here makes the alignment structural.
        for_each_binary_lp(
            lp_fn,
            cmt,
            &subject.obs_records,
            theta,
            eta,
            subject.record_covariate_snapshots(),
            &subject.covariates,
            |r| {
                let p = binary_prob(*link, r.lp);
                // `q` from `inv_logit(−lp)`, never `1 − p` — see `binary_pearson_residual`.
                let q = binary_prob(*link, -r.lp);
                // PRED is the typical-subject probability (η = 0), mirroring the Gaussian
                // `pred`/`ipred` split; IPRED conditions on this subject's EBE.
                let pred = with_model_time(r.model_time, || {
                    binary_prob(*link, lp_fn(theta, &zero_eta, r.covariates))
                });
                let y = (r.state as f64).min(1.0);
                rows.push(DiscreteObsDiagnostic {
                    cmt,
                    time: r.raw_time,
                    dv: y,
                    pred,
                    ipred: p,
                    iwres: binary_pearson_residual(y, p, q),
                });
            },
        );
    }
    rows
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
            raw_time: time,
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
        let got = binary_data_term(LinkFn::Logit, &const_lp(lp), 3, &recs, &[], &[], None, &cov);
        assert!((got - expect).abs() < 1e-12, "got {got}, expect {expect}");
    }

    /// p = 0.5 at lp = 0 ⇒ every record contributes ln 2.
    #[test]
    fn binary_data_term_half_probability() {
        let cov = HashMap::new();
        let recs = [rec(0.0, 0), rec(1.0, 1)];
        let got = binary_data_term(
            LinkFn::Logit,
            &const_lp(0.0),
            3,
            &recs,
            &[],
            &[],
            None,
            &cov,
        );
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
                None,
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
        assert_eq!(nll(f64::INFINITY, 1), SUBJECT_SENTINEL_NLL);
        assert_eq!(nll(f64::NEG_INFINITY, 0), SUBJECT_SENTINEL_NLL);
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
                    None,
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
    ///
    /// This validates the `Open01` + `u < p` primitive in isolation (a local closure);
    /// that the *production* [`simulate_binary`] is actually wired to it — so an inverted
    /// comparison is caught on the fast PR job, not only by the slow SSE — is pinned by
    /// `production_binary_draw_frequency_tracks_p` in `tests/categorical_smoke.rs`.
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
    /// genuinely degenerate variance yields NaN rather than a plausible-looking number.
    #[test]
    fn pearson_residual_standardizes_and_flags_degenerate_p() {
        let res = |y: f64, p: f64| binary_pearson_residual(y, p, 1.0 - p);
        // p = 0.5 ⇒ √(p·q) = 0.5 ⇒ residual = ±1.
        assert!((res(1.0, 0.5) - 1.0).abs() < 1e-12);
        assert!((res(0.0, 0.5) + 1.0).abs() < 1e-12);
        // A surprising observation is a large positive residual.
        let r = res(1.0, 0.01);
        assert!(r > 9.0, "expected a large residual, got {r}");
        // Degenerate variance: undefined, surfaced as NaN (not 0, not ±∞).
        assert!(binary_pearson_residual(1.0, 0.0, 1.0).is_nan());
        assert!(binary_pearson_residual(0.0, 1.0, 0.0).is_nan());
        // NaN inputs must not sneak through the `!(v > 0.0)` guard as a finite number.
        assert!(binary_pearson_residual(1.0, f64::NAN, f64::NAN).is_nan());
    }

    /// **The tail-symmetry guard.** `q` is passed independently precisely so the
    /// residual does not collapse on the positive tail, where `1 − p` cancels to exactly
    /// zero once `inv_logit` saturates. Both tails must stay finite and mirror each other.
    #[test]
    fn pearson_residual_survives_the_saturating_tail() {
        // `1 - inv_logit(lp)` is exactly 0 here — the cancellation that used to blank
        // the most extreme (most diagnostic) observations.
        assert_eq!(1.0 - inv_logit(40.0), 0.0);
        let from_lp = |y: f64, lp: f64| {
            binary_pearson_residual(
                y,
                binary_prob(LinkFn::Logit, lp),
                binary_prob(LinkFn::Logit, -lp),
            )
        };
        for lp in [36.0_f64, 36.8, 40.0, 50.0] {
            let pos = from_lp(0.0, lp); // surprising: y=0 at p≈1
            let neg = from_lp(1.0, -lp); // its mirror image: y=1 at p≈0
            assert!(pos.is_finite(), "lp={lp}: positive tail collapsed to {pos}");
            assert!(neg.is_finite(), "lp={lp}: negative tail collapsed to {neg}");
            // Bernoulli symmetry: the two are exact negatives of each other.
            assert!(
                (pos + neg).abs() < 1e-6 * neg.abs().max(1.0),
                "lp={lp}: tails asymmetric ({pos} vs {neg})"
            );
            assert!(pos < -1e7, "lp={lp}: expected a large residual, got {pos}");
        }
        // And it must not quantize: a larger lp is a strictly larger residual.
        assert!(from_lp(0.0, 40.0).abs() > from_lp(0.0, 36.0).abs());
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

#[cfg(test)]
pub(crate) mod exact_eta_hessian_tests {
    use super::*;

    /// A model string against an example dataset, through the production loader.
    pub(crate) fn load(src: &str, data: &str) -> crate::api::PreparedRun {
        let root = env!("CARGO_MANIFEST_DIR");
        let f = tempfile::Builder::new().suffix(".ferx").tempfile().unwrap();
        std::fs::write(f.path(), src).unwrap();
        crate::api::prepare_run(f.path().to_str().unwrap(), Some(&format!("{root}/{data}")))
            .unwrap()
    }

    /// The binary η-Hessian is exact. With `lp = TH0 + ETA_0 + THX·X + THT·TIME·exp(ETA_1)`
    /// it is `Σ p(1−p)∇lp∇lpᵀ + (p − y)∇²lp`, `∇lp = (1, THT·TIME·e^{η₁})`, `∇²lp` nonzero
    /// only at `(1,1)` (`= ∇lp₁`): built here by hand and matched to rounding; and the
    /// production route takes it (the stencil agrees to its own accuracy).
    #[test]
    fn the_binary_eta_hessian_is_exact() {
        let prep = load(
            r"
[parameters]
  theta TH0(0.2, -10.0, 10.0)
  theta THX(0.4, -10.0, 10.0)
  theta THT(0.05, -10.0, 10.0)
  omega ETA_0 ~ 0.3
  omega ETA_1 ~ 0.2
[covariates]
  X continuous
[binary_model]
  cmt   = 3
  logit = TH0 + ETA_0 + THX * X + THT * TIME * exp(ETA_1)
",
            "data/binary_logistic.csv",
        );
        let model = &prep.parsed.model;
        let theta = &model.default_params.theta;
        let eta = [0.11, -0.07];
        let mut worst_closed = 0.0f64;
        let mut n = 0;
        for s in prep.population.subjects.iter().take(6) {
            if s.obs_records.is_empty() {
                continue;
            }
            let (_, h) = discrete_subject_eta_hessian(model, s, theta, &eta).expect("in scope");
            // By hand.
            let mut want = nalgebra::DMatrix::<f64>::zeros(2, 2);
            for r in &s.obs_records {
                let ObsRecord::DiscreteState { time, state, .. } = r else {
                    continue;
                };
                let x = s.covariates.get("X").copied().unwrap_or(0.0);
                let tt = theta[2] * time * eta[1].exp();
                let lp = theta[0] + eta[0] + theta[1] * x + tt;
                let p = inv_logit(lp);
                let y = (*state as f64).min(1.0);
                let g = [1.0, tt];
                for k in 0..2 {
                    for l in 0..2 {
                        want[(k, l)] += p * (1.0 - p) * g[k] * g[l];
                    }
                }
                want[(1, 1)] += (p - y) * tt;
            }
            let scale = want.iter().fold(1e-300f64, |a, &v| a.max(v.abs()));
            for (a, b) in h.iter().zip(want.iter()) {
                worst_closed = worst_closed.max((a - b).abs() / scale);
            }
            n += 1;
        }
        assert!(n > 2, "too few subjects ({n})");
        eprintln!("binary η-Hessian: vs closed form {worst_closed:.2e}");
        // Measured 1.4e-17 against the closed form (`non_gaussian_eta_hessian_tests` pins the
        // same Hessian against a Richardson stencil of the value path).
        assert!(worst_closed < 1e-12, "{worst_closed:.2e}");
    }
}

/// The predictor reads each record's own covariate snapshot (#2562).
#[cfg(test)]
mod per_record_covariate_tests {
    use super::*;
    use crate::parser::model_parser::parse_model_string;

    const XS: [f64; 3] = [0.0, 1.0, 2.5];
    const STATES: [usize; 3] = [0, 1, 1];
    const THETA: [f64; 2] = [-0.4, 0.8];
    const ETA: [f64; 1] = [0.3];

    fn model() -> CompiledModel {
        parse_model_string(
            r"
[parameters]
  theta TH0(-0.4, -10.0, 10.0)
  theta THX(0.8, -10.0, 10.0)
  omega ETA_0 ~ 0.3
[binary_model]
  cmt   = 3
  logit = TH0 + THX * X + ETA_0
",
        )
        .unwrap()
    }

    /// Three binary records whose `X` is 0, 1, 2.5 in their own snapshots, against a
    /// baseline `X` of 0: the first record's value, so a baseline read is wrong on two.
    fn subject() -> Subject {
        let x = |v: f64| HashMap::from([("X".to_string(), v)]);
        Subject {
            id: "1".into(),
            covariates: x(XS[0]),
            obs_records: (0..3)
                .map(|r| ObsRecord::DiscreteState {
                    time: r as f64,
                    raw_time: r as f64,
                    state: STATES[r],
                    cmt: 3,
                })
                .collect(),
            endpoint_covariates: XS.iter().map(|&v| x(v)).collect(),
            ..Default::default()
        }
    }

    fn lp(r: usize, eta: f64) -> f64 {
        THETA[0] + THETA[1] * XS[r] + eta
    }

    fn hand_nll(eta: f64) -> f64 {
        (0..3)
            .map(|r| softplus(lp(r, eta)) - STATES[r] as f64 * lp(r, eta))
            .sum()
    }

    /// The value path and the dual path (inner η-gradient, η-Hessian, the adjoint's
    /// jets) both score each record at its own `X`, asserted separately so a failure
    /// names the side; the η-gradient is `Σ (p − y)`.
    ///
    /// Mutations: either path reading `subject.covariates` again dies on its own
    /// assertion (the baseline-read NLL is 0.95 away).
    #[test]
    fn value_and_dual_paths_read_each_records_snapshot() {
        let (m, s) = (model(), subject());
        let want = hand_nll(ETA[0]);
        let baseline: f64 = (0..3)
            .map(|r| {
                let l = THETA[0] + ETA[0];
                softplus(l) - STATES[r] as f64 * l
            })
            .sum();
        assert!(
            (want - baseline).abs() > 0.5,
            "fixture is live: {want} vs {baseline}"
        );
        let value = binary_subject_nll(&m, &s, &THETA, &ETA);
        assert!(
            (value - want).abs() < 1e-12,
            "value path: {value} vs {want}"
        );
        let (dual, grad) = discrete_subject_eta_grad(&m, &s, &THETA, &ETA).expect("in scope");
        assert!((dual - want).abs() < 1e-12, "dual path: {dual} vs {want}");
        let g: f64 = (0..3)
            .map(|r| inv_logit(lp(r, ETA[0])) - STATES[r] as f64)
            .sum();
        assert!(
            (grad[0] - g).abs() < 1e-12,
            "dual gradient: {} vs {g}",
            grad[0]
        );
    }

    /// `predict_binary`, `binary_diagnostics` (PRED at η = 0 and IPRED) and
    /// `simulate_binary` report each record's probability at its own `X`.
    ///
    /// Mutation: any one of them reading the baseline dies on its own assertion.
    #[test]
    fn predictions_diagnostics_and_draws_read_each_records_snapshot() {
        let (m, s) = (model(), subject());
        let want = |r: usize, eta: f64| inv_logit(lp(r, eta));
        let close = |a: f64, b: f64| (a - b).abs() < 1e-14;

        let mut preds = Vec::new();
        predict_binary(&m, &s, &THETA, &ETA, &mut preds);
        for (r, p) in preds.iter().enumerate() {
            let Prediction::CatProbs { probs } = &p.prediction else {
                panic!("binary prediction");
            };
            assert!(
                close(probs[1], want(r, ETA[0])),
                "predict_binary record {r}"
            );
        }

        let diags = binary_diagnostics(&m, &s, &THETA, &ETA);
        for (r, d) in diags.iter().enumerate() {
            assert!(
                close(d.ipred, want(r, ETA[0])),
                "diagnostics IPRED record {r}"
            );
            assert!(close(d.pred, want(r, 0.0)), "diagnostics PRED record {r}");
        }

        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(2562);
        let (mut sims, mut warnings) = (Vec::new(), Vec::new());
        simulate_binary(
            &m,
            &s,
            &THETA,
            &ETA,
            0,
            0,
            &mut rng,
            &mut sims,
            &mut warnings,
        );
        assert_eq!(sims.len(), 3);
        for (r, x) in sims.iter().enumerate() {
            assert!(
                close(x.ipred, want(r, ETA[0])),
                "simulate_binary record {r}"
            );
        }
    }
}

#[cfg(test)]
#[path = "probit_tests.rs"]
mod probit_tests;
