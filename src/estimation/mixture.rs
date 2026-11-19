//! Mixture-model FOCE objective (#977 Phase 3).
//!
//! A `$MIXTURE` model's subject marginal is a covariate-weighted mixture over
//! `K` class-conditional likelihoods:
//!
//! ```text
//! p_ik = softmax_k( g_k(θ, covariates_i) )       Σ_k p_ik = 1
//! L_i  = Σ_k p_ik · L_ik(θ, Ω_k, Σ_k)
//! OFV  = −2 Σ_i log Σ_k p_ik exp(−nll_ik)         ← log-sum-exp for stability
//! ```
//!
//! `nll_ik` is the ordinary per-subject FOCE negative log-likelihood
//! ([`crate::stats::likelihood::foce_subject_nll`]) evaluated with class-`k` parameters and the class-`k`
//! empirical Bayes estimate. Because the class-specific typical values are
//! selected by the reserved `MIXNUM` index inside `[individual_parameters]`, and
//! `MIXNUM` resolves from a thread-local, each class's inner solve + NLL is run
//! under a `MixtureClassGuard`. The per-class loop is **serial** on purpose: a
//! thread-local set on the outer thread would not reach the rayon workers the
//! standard inner loop uses, so the mixture path drives `find_ebe` itself.
//!
//! Cost is ≈ K× the inner loop (K EBE searches per subject per outer iteration).
//! Parallelising across (subject × class) is a later optimisation (#977 risks).

use nalgebra::{DMatrix, DVector};

use crate::estimation::covariance::ExactDecline;
use crate::estimation::inner_optimizer::{
    build_schedule_cache, find_ebe_cached, find_ebe_for_stage_with_prev, InnerLoopStats, PrevEtas,
};
use crate::estimation::parameterization::{
    fd_theta_step_floors, pack_params, pack_with_bounds, packed_segments, theta_packs_log,
    unpack_params, PackedBounds,
};
use crate::parser::model_parser::{
    eval_mixing_log_probs, mixing_logp_derivs, mixing_logp_grad, MixtureClassGuard,
};
use crate::stats::likelihood::{foce_subject_nll_cov, foce_subject_nll_iov_cov};
use crate::types::{CompiledModel, FitOptions, MixtureParams, ModelParameters, Population};

/// `∂ ln p_ik / ∂ θ_j` for a set of subjects at one θ, recorded once per distinct value of the
/// covariates the mixing expressions read — not once per subject. θ is fixed for the cache's
/// life (one outer gradient, one score context), and the derivative depends on the subject only
/// through those covariates, so subjects that share them share one recording and one run; a
/// covariate-free mixing map records once for the whole population.
pub(crate) struct MixingGradCache {
    covariates: Vec<String>,
    floor: Vec<f64>,
    rows: std::sync::Mutex<
        std::collections::HashMap<Vec<u64>, std::sync::Arc<std::sync::OnceLock<Vec<Vec<f64>>>>>,
    >,
    #[cfg(test)]
    computed: std::sync::atomic::AtomicUsize,
}

impl MixingGradCache {
    pub(crate) fn new(model: &CompiledModel) -> Self {
        Self {
            covariates: model
                .mixture
                .as_ref()
                .map(|m| m.logit_covariates.clone())
                .unwrap_or_default(),
            floor: fd_theta_step_floors(model),
            rows: Default::default(),
            #[cfg(test)]
            computed: Default::default(),
        }
    }

    /// The subject's `K × n_theta` rows ([`mixing_logp_grad`]), bit-equal to calling it directly.
    pub(crate) fn rows(
        &self,
        model: &CompiledModel,
        spec: &crate::types::MixtureSpec,
        theta: &[f64],
        covariates: &std::collections::HashMap<String, f64>,
    ) -> Vec<Vec<f64>> {
        // A missing column and a stored value are different inputs; the flag keeps them apart.
        let key: Vec<u64> = self
            .covariates
            .iter()
            .flat_map(|n| match covariates.get(n) {
                Some(v) => [1, v.to_bits()],
                None => [0, 0],
            })
            .collect();
        // One slot per key: workers racing on a new key wait on one recording, and the map's
        // lock is not held while it runs, so distinct keys record in parallel.
        let slot = self.rows.lock().unwrap().entry(key).or_default().clone();
        slot.get_or_init(|| {
            #[cfg(test)]
            self.computed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            mixing_logp_grad(model, spec, theta, covariates, &self.floor)
        })
        .clone()
    }

    /// Recordings made so far.
    #[cfg(test)]
    pub(crate) fn computed(&self) -> usize {
        self.computed.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
thread_local! {
    /// The recordings `mixture_subject_gradients` made in its last call on this thread.
    pub(crate) static MIXING_RECORDINGS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Result of one mixture-objective evaluation.
pub struct MixtureEval {
    /// Population objective, `−2 Σ_i log Σ_k p_ik exp(−nll_ik)`.
    pub ofv: f64,
    /// MIXEST-class EBE per subject (for warm-start carry / postfit).
    pub mixest_etas: Vec<DVector<f64>>,
    /// MIXEST-class inner Hessian per subject.
    pub mixest_h_mats: Vec<DMatrix<f64>>,
    /// MIXEST-class per-occasion κ̂ per subject (#985). Empty inner `Vec` for a
    /// non-IOV mixture; `mixest_kappas[i]` is the winning class's per-occasion
    /// κ EBEs, so postfit (IPRED/IWRES/CWRES, per-subject OFV, κ shrinkage, and
    /// the `.fitrx` `ebe_kappas` export) sees the IOV the fit actually used
    /// rather than κ = 0.
    pub mixest_kappas: Vec<Vec<DVector<f64>>>,
    /// Posterior class membership `PMIX_ik` per subject (length `K`).
    pub pmix: Vec<Vec<f64>>,
    /// `MIXEST_i` = argmax-posterior class per subject (0-based).
    pub mixest: Vec<usize>,
    /// Per-class EBE cache `[class][subject]` for warm-starting the next eval.
    pub etas_by_class: Vec<Vec<DVector<f64>>>,
    /// Inner-loop stats aggregated over each subject's winning class.
    pub ebe_stats: InnerLoopStats,
    /// Subjects with a contributing class whose FD Jacobian ran at the fit's tolerance
    /// because its own tighter solve could not finish (#2701).
    pub(crate) n_fd_h_redone: usize,
    /// Per-class objective `nll_ik` per subject (`[subject][class]`), the scores `pmix` was
    /// built from. Kept in log space for the merged-class check (#2907): a `PMIX` that
    /// underflowed to 0 has lost the comparison this preserves.
    pub(crate) class_nll: Vec<Vec<f64>>,
}

/// A class-`k` (0-based) view of `params`: base theta / FIX flags, with the
/// class's Omega/Sigma swapped in.
pub fn class_params(params: &ModelParameters, k: usize) -> ModelParameters {
    let mix = params
        .mixture
        .as_ref()
        .expect("class_params called on non-mixture ModelParameters");
    let mut p = params.clone();
    p.omega = mix.omega[k].clone();
    p.sigma = mix.sigma[k].clone();
    p
}

/// Largest per-subject log Bayes factor between two classes below which the post-fit check
/// calls them merged (#2907). See `merged_classes_from` for the measurement it was set from.
pub(crate) const MERGED_CLASS_LOG_BF: f64 = 1.0;

/// Posterior class probability that no subject reaches in a class the post-fit check calls
/// empty (#2907).
pub(crate) const EMPTY_CLASS_PMIX: f64 = 1e-2;

/// One finding of the post-fit merged-class check (#2907). Classes are 1-based, as `MIXNUM`
/// spells them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ClassCollapse {
    /// Every subject's posterior probability of `class` is below `EMPTY_CLASS_PMIX`;
    /// `max_pmix` is the largest.
    Empty { class: usize, max_pmix: f64 },
    /// Both classes carry posterior mass, and no subject's data favours one over the other
    /// by more than `MERGED_CLASS_LOG_BF`; `max_log_bf` is the largest such factor.
    Merged { a: usize, b: usize, max_log_bf: f64 },
}

/// The post-fit merged-class check (#2907) at the final evaluation: `nll[i][k]` is subject
/// `i`'s class objective `nll_ik` and `pmix[i][k]` its posterior `PMIX_ik`, both in subject
/// order (`MixtureEval::class_nll` and `MixtureEval::pmix`).
///
/// A pair is judged on the objective, not on its parameters: classes that are equal but
/// written two different ways have different parameters and the same likelihood. Subject
/// `i`'s log Bayes factor between classes `a` and `b` is `|nll_ia − nll_ib|`, the amount its
/// data moved its class odds away from the prior odds (`PMIX_ik ∝ p_ik · exp(−nll_ik)`). When
/// no subject moves them by more than `MERGED_CLASS_LOG_BF`, the data cannot tell the two
/// classes apart.
///
/// The factor is read off `nll` and not reconstructed from `PMIX`: in a fit with three or more
/// classes, a subject dominated by a third class can have both `PMIX_ia` and `PMIX_ib`
/// underflow to 0 although `nll_ia = nll_ib`, and `0 / 0` would call the pair separated. A
/// factor that is not finite (a class objective that is `NaN` or infinite) separates the pair,
/// so a failed score never produces the warning.
///
/// A class whose posterior is below `EMPTY_CLASS_PMIX` in every subject holds nobody, and is
/// reported as empty; a pair with an empty class is not also reported as merged, since the
/// factor of a class with no mass says nothing about the fit. A non-finite posterior makes its
/// class non-empty, for the same reason.
///
/// Pairwise, so with three classes two merged ones beside a third are reported.
pub(crate) fn merged_classes_from(nll: &[Vec<f64>], pmix: &[Vec<f64>]) -> Vec<ClassCollapse> {
    let k = pmix.first().map_or(0, Vec::len);
    if k < 2 || pmix.len() != nll.len() || nll.iter().any(|n| n.len() != k) {
        return Vec::new();
    }
    // `None` when some subject's posterior of the class is not finite: such a class is never
    // called empty (`f64::max` would drop the `NaN` and leave a maximum of 0).
    let max_pmix: Vec<Option<f64>> = (0..k)
        .map(|c| {
            pmix.iter().try_fold(0.0f64, |m, p| {
                let v = p[c];
                v.is_finite().then(|| m.max(v))
            })
        })
        .collect();
    let empty: Vec<bool> = max_pmix
        .iter()
        .map(|m| m.is_some_and(|m| m < EMPTY_CLASS_PMIX))
        .collect();
    let mut out: Vec<ClassCollapse> = (0..k)
        .filter(|&c| empty[c])
        .map(|c| ClassCollapse::Empty {
            class: c + 1,
            max_pmix: max_pmix[c].unwrap_or(0.0),
        })
        .collect();
    for a in 0..k {
        for b in a + 1..k {
            if empty[a] || empty[b] {
                continue;
            }
            let mut max_log_bf = 0.0f64;
            for n in nll {
                let bf = (n[a] - n[b]).abs();
                // NaN (∞ − ∞) and ∞ both separate the pair; `f64::max` would drop NaN.
                max_log_bf = if bf.is_finite() {
                    max_log_bf.max(bf)
                } else {
                    f64::INFINITY
                };
                if max_log_bf >= MERGED_CLASS_LOG_BF {
                    break;
                }
            }
            if max_log_bf < MERGED_CLASS_LOG_BF {
                out.push(ClassCollapse::Merged {
                    a: a + 1,
                    b: b + 1,
                    max_log_bf,
                });
            }
        }
    }
    out
}

/// The `W_MIXTURE_MERGED_CLASSES` warning for `findings`, or `None` when there are none.
pub(crate) fn merged_classes_warning(findings: &[ClassCollapse]) -> Option<String> {
    if findings.is_empty() {
        return None;
    }
    let parts: Vec<String> = findings
        .iter()
        .map(|f| match f {
            ClassCollapse::Empty { class, max_pmix } => format!(
                "class {class} is empty (no subject's posterior probability of it exceeds \
                 {max_pmix:.2e})"
            ),
            ClassCollapse::Merged { a, b, max_log_bf } => format!(
                "classes {a} and {b} have merged (no subject's data favours either by more \
                 than {max_log_bf:.3} log-likelihood units)"
            ),
        })
        .collect();
    Some(format!(
        "W_MIXTURE_MERGED_CLASSES: at the final estimates {}. The fit has fewer distinct \
         classes than `nsub`, which is often a local optimum rather than the best mixture: \
         refit with class starts further apart, or with `optimizer = bobyqa`, and compare the \
         objective (#2907).",
        parts.join("; ")
    ))
}

/// Combine one subject's per-class scores into its mixture contribution.
///
/// Given `ln p_ik` (`logp`) and `nll_ik` (`nll`), returns
/// `(contribution, PMIX, MIXEST)` where the contribution is
/// `−2 log Σ_k p_ik exp(−nll_ik)` (via log-sum-exp), `PMIX_ik ∝
/// p_ik exp(−nll_ik)`, and `MIXEST` is the argmax-posterior class (0-based).
pub fn combine_subject(logp: &[f64], nll: &[f64]) -> (f64, Vec<f64>, usize) {
    let k = logp.len();
    // terms_k = ln p_ik − nll_ik  (== ln [p_ik · exp(−nll_ik)])
    let terms: Vec<f64> = (0..k).map(|c| logp[c] - nll[c]).collect();
    let max = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = max + terms.iter().map(|t| (t - max).exp()).sum::<f64>().ln();
    let pmix: Vec<f64> = terms.iter().map(|t| (t - lse).exp()).collect();
    // First-wins argmax (stable label-order tie-break: prefer the lower class).
    let mut mixest = 0;
    for c in 1..k {
        if pmix[c] > pmix[mixest] {
            mixest = c;
        }
    }
    (2.0 * (-lse), pmix, mixest)
}

/// Posterior weight below which a class cannot meaningfully move a subject's
/// mixture marginal, so its inner-solve status is not tallied (#984 review).
const PMIX_CONTRIB_TOL: f64 = 1e-6;

/// True if any class flagged `true` in `flags` carries non-negligible posterior
/// weight `probs[k]` (> [`PMIX_CONTRIB_TOL`]). Used to score EBE
/// non-convergence / fallback / hard-reject over every *contributing* class of a
/// subject, not just its winning class — a non-converged non-winning class still
/// enters the log-sum-exp OFV with weight `PMIX_ik` and must be surfaced.
fn any_contributing(flags: &[bool], probs: &[f64]) -> bool {
    flags
        .iter()
        .zip(probs)
        .any(|(&f, &p)| f && p > PMIX_CONTRIB_TOL)
}

/// Evaluate the mixture FOCE objective at `params` (#977 Phase 3).
///
/// `warm` optionally carries the previous iteration's per-class EBEs
/// (`[class][subject]`) to warm-start each class's inner solve. They are taken as modes a
/// solve returned, so an exactly-zero one is warm like any other and does not rerun the
/// cold inner multi-start; pass `None` for a cold start.
pub fn mixture_ofv(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    warm: Option<&[Vec<DVector<f64>>]>,
) -> MixtureEval {
    mixture_ofv_impl(model, population, params, options, warm, false).0
}

/// Every class's inner modes at one mixture evaluation, `[class][subject]`: the
/// per-class κ̂ that `MixtureEval` keeps only for the MIXEST class (its
/// [`MixtureEval::etas_by_class`] already holds every class). The covariance step's
/// per-subject mixture score (#1956) differentiates every class's `nll_ik` at its own
/// mode, so it needs them all.
pub(crate) struct MixtureClassModes {
    /// Every class's EBE `η̂` per subject, `[class][subject]` ([`MixtureEval::etas_by_class`]):
    /// what a re-solve warm-starts from ([`mixture_candidate_hessian`]).
    pub(crate) etas: Vec<Vec<DVector<f64>>>,
    /// Per-occasion κ̂ per class per subject (empty inner `Vec` without IOV).
    pub(crate) kappas: Vec<Vec<Vec<DVector<f64>>>>,
}

/// [`mixture_ofv`], also returning every class's modes ([`MixtureClassModes`]).
pub(crate) fn mixture_ofv_with_modes(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    warm: Option<&[Vec<DVector<f64>>]>,
) -> (MixtureEval, MixtureClassModes) {
    mixture_ofv_impl(model, population, params, options, warm, true)
}

/// [`mixture_ofv_with_modes`]; `snapshot_etas` copies every class's `η̂` into the modes
/// (`MixtureClassModes::etas`), which only the callers that warm-start from them read.
fn mixture_ofv_impl(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    warm: Option<&[Vec<DVector<f64>>]>,
    snapshot_etas: bool,
) -> (MixtureEval, MixtureClassModes) {
    let spec = model
        .mixture
        .as_ref()
        .expect("mixture_ofv on a non-mixture model");
    let mp = params
        .mixture
        .as_ref()
        .expect("mixture_ofv with no MixtureParams");
    let k = mp.omega.len();
    let n = population.subjects.len();
    let interaction = options.interaction;

    // Per-class inner solves. `nll[class][subject]`, plus per-class EBE cache and
    // convergence flags.
    let mut nll: Vec<Vec<f64>> = vec![vec![0.0; n]; k];
    let mut etas_by_class: Vec<Vec<DVector<f64>>> = Vec::with_capacity(k);
    let mut hmats_by_class: Vec<Vec<DMatrix<f64>>> = Vec::with_capacity(k);
    // Per-occasion κ̂ per class per subject (#985). Empty inner `Vec` for a
    // non-IOV class; kept so the MIXEST class's κ can be carried into postfit.
    let mut kappas_by_class: Vec<Vec<Vec<DVector<f64>>>> = Vec::with_capacity(k);
    let mut converged: Vec<Vec<bool>> = vec![vec![true; n]; k];
    let mut fallback: Vec<Vec<bool>> = vec![vec![false; n]; k];
    let mut nm_polished: Vec<Vec<bool>> = vec![vec![false; n]; k];
    let mut hard_reject: Vec<Vec<bool>> = vec![vec![false; n]; k];
    let mut fd_h_redone: Vec<Vec<bool>> = vec![vec![false; n]; k];

    // Laplace / AGQ (#3025): each class is scored with the quadrature subject objective at
    // its own mode, the single population's `agq_subject_term` on the class parameters, and
    // solved under the single population's quadrature inner policy (Hessian seed, terminal
    // exact Hessian). FOCE/FOCEI keep the historical solve and score untouched.
    let quad = options.agq_nodes().map(|n_nodes| {
        let policy =
            crate::estimation::outer_optimizer::agq_inner_solve_policy(model, options, false)
                .with_prev_etas(PrevEtas::Solved);
        (
            n_nodes,
            options.hessian_anchor(),
            policy,
            build_schedule_cache(model, population),
        )
    });

    for cls in 0..k {
        // MIXNUM = cls+1 for the whole class solve (serial → thread-local safe).
        let _guard = MixtureClassGuard::enter(cls + 1);
        let cp = class_params(params, cls);
        let mut etas_c = Vec::with_capacity(n);
        let mut hmats_c = Vec::with_capacity(n);
        let mut kappas_c = Vec::with_capacity(n);
        for (i, subject) in population.subjects.iter().enumerate() {
            let warm_i = warm
                .and_then(|w| w.get(cls))
                .and_then(|wc| wc.get(i))
                .map(|e| e.as_slice());
            let mut ebe = if let Some((_, _, policy, schedules)) = &quad {
                find_ebe_cached(
                    model,
                    subject,
                    &cp,
                    options.inner_maxiter,
                    options.inner_tol,
                    warm_i,
                    None,
                    options.inner_restarts,
                    schedules[i].as_ref(),
                    *policy,
                )
            } else {
                find_ebe_for_stage_with_prev(
                    model,
                    subject,
                    &cp,
                    options.inner_maxiter,
                    options.inner_tol,
                    warm_i,
                    None,
                    options.inner_restarts,
                    options,
                    // An earlier pass's class modes, an exactly-zero one included.
                    PrevEtas::Solved,
                )
            };
            fd_h_redone[cls][i] = ebe.fd_h_redone;
            // Inter-occasion variability (#985): `find_ebe` already routes to the
            // IOV inner solve when `n_kappa > 0` (each class carries the shared base
            // `omega_iov`, since `class_params` only swaps Ω/Σ), so `ebe.kappas` are
            // the per-occasion κ̂ for this class. The outer marginal must then use the
            // IOV FOCE nll, which augments the marginal with the κ prior + κ columns.
            // We key the branch on `cp.omega_iov`, the exact field the IOV nll unwraps
            // — the same predicate `mixture_gradient` gates on — so the two paths can
            // never disagree on scope, and no `expect`/unwrap can fire on a
            // present-`n_kappa`-but-absent-`omega_iov` model. It self-reduces to
            // `foce_subject_nll` when `omega_iov` is absent, keeping a non-IOV mixture
            // on the byte-identical legacy call.
            nll[cls][i] = if let Some((n_nodes, anchor, _, schedules)) = &quad {
                let work = (ebe.terminal_hessian.take(), ebe.nll);
                crate::estimation::agq::agq_subject_term_with_work(
                    model,
                    subject,
                    &cp,
                    ebe.eta.as_slice(),
                    &ebe.kappas,
                    *n_nodes,
                    *anchor,
                    schedules[i].as_ref(),
                    Some(&work),
                )
            } else if let Some(omega_iov) = cp.omega_iov.as_ref() {
                foce_subject_nll_iov_cov(
                    model,
                    subject,
                    &cp.theta,
                    &ebe.eta,
                    &ebe.h_matrix,
                    &cp.omega,
                    &cp.sigma.values,
                    model.residual_cov_at(&cp),
                    interaction,
                    &ebe.kappas,
                    omega_iov,
                )
            } else {
                foce_subject_nll_cov(
                    model,
                    subject,
                    &cp.theta,
                    &ebe.eta,
                    &ebe.h_matrix,
                    &cp.omega,
                    &cp.sigma.values,
                    model.residual_cov_at(&cp),
                    interaction,
                )
            };
            converged[cls][i] = ebe.converged;
            fallback[cls][i] = ebe.used_fallback;
            nm_polished[cls][i] = ebe.nm_polished;
            hard_reject[cls][i] = ebe.hard_reject;
            etas_c.push(ebe.eta);
            hmats_c.push(ebe.h_matrix);
            kappas_c.push(ebe.kappas);
        }
        etas_by_class.push(etas_c);
        hmats_by_class.push(hmats_c);
        kappas_by_class.push(kappas_c);
    }

    // Combine per subject via log-sum-exp; pick MIXEST for warm-start / postfit.
    let mut ofv = 0.0;
    let mut pmix = Vec::with_capacity(n);
    let mut class_nll = Vec::with_capacity(n);
    let mut mixest = Vec::with_capacity(n);
    let mut mixest_etas = Vec::with_capacity(n);
    let mut mixest_h_mats = Vec::with_capacity(n);
    let mut mixest_kappas = Vec::with_capacity(n);
    let (mut n_unconverged, mut n_fallback, mut n_start_rejected) = (0usize, 0usize, 0usize);
    let mut n_nm_polished = 0usize;
    let mut n_fd_h_redone = 0usize;

    for (i, subject) in population.subjects.iter().enumerate() {
        let logp = eval_mixing_log_probs(spec, &params.theta, &subject.covariates);
        let nll_i: Vec<f64> = (0..k).map(|cls| nll[cls][i]).collect();
        let (contrib, probs, best) = combine_subject(&logp, &nll_i);
        class_nll.push(nll_i);
        ofv += contrib;
        // Convergence is scored over *every* class that contributes meaningfully to
        // this subject's marginal, not just the winning (`best`) class: each class's
        // `nll_ik` enters the log-sum-exp with weight `PMIX_ik`, so a non-converged
        // non-winning class still contaminates the OFV. Scoring only `best` would let
        // that pass the outer EBE-convergence guard silently. A class with negligible
        // posterior weight cannot move the marginal, so it is not counted. Each
        // subject contributes at most one to each tally.
        let unconverged_i: Vec<bool> = (0..k).map(|cls| !converged[cls][i]).collect();
        let fallback_i: Vec<bool> = (0..k).map(|cls| fallback[cls][i]).collect();
        let nm_polished_i: Vec<bool> = (0..k).map(|cls| nm_polished[cls][i]).collect();
        let reject_i: Vec<bool> = (0..k).map(|cls| hard_reject[cls][i]).collect();
        if any_contributing(&unconverged_i, &probs) {
            n_unconverged += 1;
        }
        if any_contributing(&fallback_i, &probs) {
            n_fallback += 1;
        }
        if any_contributing(&nm_polished_i, &probs) {
            n_nm_polished += 1;
        }
        if any_contributing(&reject_i, &probs) {
            n_start_rejected += 1;
        }
        let fd_h_redone_i: Vec<bool> = (0..k).map(|cls| fd_h_redone[cls][i]).collect();
        if any_contributing(&fd_h_redone_i, &probs) {
            n_fd_h_redone += 1;
        }
        mixest_etas.push(etas_by_class[best][i].clone());
        mixest_h_mats.push(hmats_by_class[best][i].clone());
        mixest_kappas.push(kappas_by_class[best][i].clone());
        pmix.push(probs);
        mixest.push(best);
    }

    let etas_snapshot = if snapshot_etas {
        etas_by_class.clone()
    } else {
        Vec::new()
    };
    let eval = MixtureEval {
        ofv,
        mixest_etas,
        mixest_h_mats,
        mixest_kappas,
        pmix,
        mixest,
        etas_by_class,
        ebe_stats: InnerLoopStats {
            n_unconverged,
            n_fallback,
            n_nm_polished,
            n_start_rejected,
        },
        n_fd_h_redone,
        class_nll,
    };
    let modes = MixtureClassModes {
        etas: etas_snapshot,
        kappas: kappas_by_class,
    };
    (eval, modes)
}

/// Analytic outer gradient of the mixture OFV in **real packed space** (#977
/// Phase 4): `∂OFV/∂x = 2 Σ_i Σ_k w_ik (∂nll_ik/∂x − ∂ln p_ik/∂x)`, where
/// `w_ik = PMIX_ik` (from `eval`), `∂nll_ik/∂x` is the per-class per-subject
/// analytic FOCE/FOCEI packed gradient, and `∂ln p_ik/∂θ` is the mixing
/// gradient (theta slots only). The un-scaled gradient the objective closure
/// then multiplies by `scale[k]` for optimizer space.
///
/// Under inter-occasion variability (κ) `eval` does not carry every class's κ̂ (it keeps the
/// MIXEST class's), so the per-class modes are re-solved here, warm from `eval`'s own EBEs
/// (`mixture_ofv_with_modes`), and the gradient is taken on that evaluation: an IOV
/// mixture is served exactly like any other. The fit itself holds the modes already and
/// calls `mixture_gradient_with_modes` directly. `None` only when
/// `mixture_gradient_with_modes` declines.
pub fn mixture_gradient(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    eval: &MixtureEval,
) -> Option<Vec<f64>> {
    if params.omega_iov.is_some() {
        let (own, modes) = mixture_ofv_with_modes(
            model,
            population,
            params,
            options,
            Some(&eval.etas_by_class),
        );
        return mixture_gradient_with_modes(model, population, params, options, &own, Some(&modes));
    }
    mixture_gradient_with_modes(model, population, params, options, eval, None)
}

/// [`mixture_gradient`] with every class's modes from the same evaluation
/// ([`mixture_ofv_with_modes`]): under IOV each class's subject gradient is taken at its
/// stacked `[η̂, κ̂]`, so an IOV mixture is exact too. The base Ω may be a block: a class
/// override can only name a diagonal base Ω's coordinate (a parse-time rule), so a block
/// Ω's coordinates route to themselves. A subject gradient the adjoint declines (a failed
/// plan, #2786) is `None` for the whole ([`mixture_gradient_salvaged`] salvages it per
/// term). Under IOV without `modes` there is no κ̂ to differentiate at, so it is `None`.
pub(crate) fn mixture_gradient_with_modes(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    eval: &MixtureEval,
    modes: Option<&MixtureClassModes>,
) -> Option<Vec<f64>> {
    let rows =
        mixture_subject_gradients(model, population, params, options, eval, modes, None, |f| {
            f()
        })?;
    let mut grad = vec![0.0f64; packed_segments(params).total()];
    for row in &rows {
        for (g, v) in grad.iter_mut().zip(row) {
            *g += 2.0 * v;
        }
    }
    Some(grad)
}

/// [`mixture_gradient_with_modes`] that **salvages** a declined (subject × class) term
/// instead of returning `None` for the whole population (#2906), as #1529 does for a
/// single population: every term the adjoint serves keeps its exact gradient, and only a
/// declined `∂nll_ik/∂x` is replaced by that one subject's reconverged central
/// difference under class `k` ([`class_subject_reconverged_gradient`]) — `2·n_free`
/// solves of one subject in one class, where the whole-objective
/// [`mixture_gradient_fd`] re-solves every subject in every class per probe. A term
/// whose posterior weight is exactly zero carries no gradient and is not differenced. Without
/// `modes` (IOV only) the evaluation itself is re-solved, so the gradient is taken at one
/// solve's joint modes and `eval` is not read.
///
/// Each subject that took a salvage is recorded on `declines` (structural when no
/// recorder serves the subject's data shape under that class, as
/// `outer_optimizer::is_structural_decline` asks), so the fit's #1154 warning reports
/// it. Total: the mixing map's own derivative is exact, or its central difference after a
/// plan failure ([`crate::parser::model_parser::mixing_logp_grad`]), so no model reaches a
/// whole-objective difference. Under IOV `modes` must be supplied (the fit's outer loop
/// always holds them, `mixture_ofv_with_modes`).
pub(crate) fn mixture_gradient_salvaged(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    eval: &MixtureEval,
    modes: Option<&MixtureClassModes>,
    declines: &crate::estimation::outer_optimizer::OuterFdDeclineLog,
) -> Vec<f64> {
    let salvage = |d: DeclinedTerm<'_>| {
        let subject = &population.subjects[d.subject];
        declines.record(d.subject);
        declines.record_reconverged(d.subject);
        if d.cp.omega_iov.is_none() {
            let _guard = MixtureClassGuard::enter(d.cls + 1);
            if crate::estimation::outer_optimizer::is_structural_decline(
                model,
                subject,
                d.cp,
                d.eta.as_slice(),
            ) {
                declines.record_structural(d.subject);
            }
        }
        Some(class_subject_reconverged_gradient(
            model, subject, params, d.cls, d.eta, options,
        ))
    };
    // An IOV gradient reads every class's κ̂. The caller's gate for producing them
    // (`needs_class_modes`) is this one; a caller that did not have them gets the evaluation
    // re-solved here, η̂ and κ̂ together from one solve: a class's stacked `[η̂, κ̂]` is a joint
    // stationary point only when both come from the same solve, and the caller's `eval` may
    // have been warm-started to a different η̂, so its η̂ is not paired with a fresh κ̂.
    let resolved;
    let (eval, modes) = match modes {
        None if needs_class_modes(params) => {
            resolved = mixture_ofv_with_modes(model, population, params, options, None);
            (&resolved.0, Some(&resolved.1))
        }
        m => (eval, m),
    };
    let rows = mixture_subject_gradients(
        model,
        population,
        params,
        options,
        eval,
        modes,
        Some(&salvage),
        |f| f(),
    )
    .expect("a mixture model's params carry the mixture (the caller holds a `MixtureEval`)");
    let mut grad = vec![0.0f64; packed_segments(params).total()];
    for row in &rows {
        for (g, v) in grad.iter_mut().zip(row) {
            *g += 2.0 * v;
        }
    }
    grad
}

/// Whether the mixture gradient needs every class's modes ([`MixtureClassModes`]) — an IOV
/// mixture's κ̂, which the eval keeps for the MIXEST class only. The one condition the
/// producer (the outer loop) and the consumer ([`mixture_subject_gradients`]) both read.
pub(crate) fn needs_class_modes(params: &ModelParameters) -> bool {
    params.omega_iov.is_some()
}

/// One (subject × class) term the adjoint declined: what its salvage needs.
pub(crate) struct DeclinedTerm<'a> {
    /// The subject's index in the population.
    pub(crate) subject: usize,
    /// The class, 0-based.
    pub(crate) cls: usize,
    /// The class parameters, [`class_params`].
    pub(crate) cp: &'a ModelParameters,
    /// The subject's class-`cls` mode, stacked `[η̂, κ̂]` under IOV.
    pub(crate) eta: &'a DVector<f64>,
}

/// A salvage of a declined term: `d(nll_ik)/dx_k` over the class layout, or `None` to give up.
pub(crate) type SalvageTerm<'a> = dyn Fn(DeclinedTerm<'_>) -> Option<Vec<f64>> + Sync + 'a;

/// Each subject's mixture gradient `∂nllᵢ/∂x = Σ_k w_ik (∂nll_ik/∂x − ∂ln p_ik/∂x)` over the
/// mixture's packed layout — `w_ik = PMIX_ik` (from `eval`), `∂nll_ik/∂x` the class's exact
/// FOCE/FOCEI subject gradient at its own mode under `MIXNUM = k`, routed by
/// [`route_class_gradient`], and `∂ln p_ik/∂θ` the mixing map's. [`mixture_gradient_with_modes`]
/// sums them; the BHHH Newton decrement reads them per subject. `salvage`: with a log, a class
/// gradient the adjoint declines takes the subject's reconverged central difference
/// ([`class_subject_reconverged_gradient`]) and the subject is recorded on the log; without
/// one the decline is `None` for the whole. `each` runs each subject's row on its worker
/// (a caller's per-subject check wraps it there).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mixture_subject_gradients(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    eval: &MixtureEval,
    modes: Option<&MixtureClassModes>,
    salvage: Option<&SalvageTerm<'_>>,
    each: impl Fn(&dyn Fn() -> Option<Vec<f64>>) -> Option<Vec<f64>> + Sync,
) -> Option<Vec<Vec<f64>>> {
    use crate::sens::ode_provider::adjoint::PackedPoint;
    let spec = model.mixture.as_ref()?;
    let mp = params.mixture.as_ref()?;
    let iov = needs_class_modes(params);
    if iov && modes.is_none() {
        return None;
    }
    let dtheta_dx = theta_chain_factors(params);
    // The objective the fit scores: FOCE, FOCEI, or the quadrature (#3025).
    let objective = crate::estimation::covariance::cov_objective(options);
    let mixing = MixingGradCache::new(model);
    // Class k's packed layout [θ, Ω(base slots), σ, …], unpacked once per class.
    let classes: Vec<(ModelParameters, Vec<f64>)> = (0..mp.omega.len())
        .map(|cls| {
            let cp = class_params(params, cls);
            let x_k = pack_params(&cp);
            (cp, x_k)
        })
        .collect();
    let points: Vec<Option<PackedPoint<'_>>> = classes
        .iter()
        .map(|(cp, x_k)| PackedPoint::new(cp, x_k))
        .collect();
    let n = packed_segments(params).total();
    let rows = crate::par::map(population.subjects.len(), |i| {
        let subject = &population.subjects[i];
        each(&|| {
            let dlnp = mixing.rows(model, spec, &params.theta, &subject.covariates);
            let mut g = vec![0.0f64; n];
            for (cls, (cp, _x_k)) in classes.iter().enumerate() {
                let w = eval.pmix[i][cls];
                if w == 0.0 {
                    continue;
                }
                // `MIXNUM` resolves to the class on this (worker) thread.
                let _class = MixtureClassGuard::enter(cls + 1);
                let mut misshaped = false;
                let eta = match modes.filter(|_| iov) {
                    Some(m) => match crate::estimation::covariance::stacked_mode(
                        model,
                        subject,
                        &eval.etas_by_class[cls][i],
                        m.kappas[cls].get(i),
                    ) {
                        Some(mode) => DVector::from_column_slice(&mode),
                        // A κ̂ count or width that does not match the occasions (a class
                        // solve that failed or was hard-rejected): the value scores this
                        // term at `class_nll`, the sentinel for such a mode. A repelled
                        // term contributes zero, as a non-mixture subject does; one with a
                        // real objective takes the reconverged difference from κ̂ = 0, which
                        // re-solves its own mode (the exact route would differentiate at the
                        // κ̂ = 0 start, not at a mode). Only with a salvage: without one the
                        // decline is `None` for the whole.
                        None if salvage.is_none() => return None,
                        None if !crate::stats::likelihood::is_usable_subject_nll(
                            eval.class_nll[i][cls],
                        ) =>
                        {
                            continue
                        }
                        None => {
                            misshaped = true;
                            let cells =
                                crate::stats::stacked_layout::group_count(subject) * model.n_kappa;
                            let mut stacked = eval.etas_by_class[cls][i].as_slice().to_vec();
                            stacked.resize(stacked.len() + cells, 0.0);
                            DVector::from_vec(stacked)
                        }
                    },
                    None => eval.etas_by_class[cls][i].clone(),
                };
                let exact = points[cls].as_ref().filter(|_| !misshaped).and_then(|pt| {
                    crate::estimation::sens_outer_gradient::subject_objective_gradient_at(
                        model,
                        subject,
                        pt,
                        eta.as_slice(),
                        objective,
                    )
                });
                let g_c = match exact {
                    Some(g_c) => g_c,
                    None if salvage.is_some() => salvage?(DeclinedTerm {
                        subject: i,
                        cls,
                        cp,
                        eta: &eta,
                    })?,
                    None => return None,
                };
                route_class_gradient(&mut g, &g_c, w, cls, params, mp);
                subtract_mixing_gradient(&mut g, &dlnp[cls], w, &dtheta_dx);
            }
            Some(g)
        })
    });
    #[cfg(test)]
    MIXING_RECORDINGS.with(|c| c.set(mixing.computed()));
    rows.into_iter().collect()
}

/// `d(nll_ik)/dx_k` over the class-`cls` layout `pack_params(&class_params(params, cls))`
/// for one subject, by central differences that **re-solve that subject's class-`cls` mode**
/// (warm-started at `eta`) at every probe — so the EBE response is included, as it is in the
/// adjoint's gradient it stands in for (#2906). Runs under `MIXNUM = cls + 1` on the
/// calling thread, the convention [`mixture_ofv`]'s serial class solves use.
///
/// The template the difference walks differs from [`class_params`] in its FIX flags only:
/// a base Ω/Σ coordinate the class overrides takes the override's flag (a free override of
/// a FIX base is free here, and is routed onto the override by [`class_route`]), and the
/// class layout's inert override tail is pinned, so no probe is spent on a coordinate
/// `class_params` has already swapped out.
fn class_subject_reconverged_gradient(
    model: &CompiledModel,
    subject: &crate::types::Subject,
    params: &ModelParameters,
    cls: usize,
    eta: &DVector<f64>,
    options: &FitOptions,
) -> Vec<f64> {
    let mut cp = class_params(params, cls);
    if let Some(mix) = cp.mixture.as_mut() {
        for (a, &(c, e)) in mix.omega_override_addr.iter().enumerate() {
            if c == cls {
                if let Some(f) = cp.omega_fixed.get_mut(e) {
                    *f = mix.omega_override_fixed.get(a).copied().unwrap_or(false);
                }
            }
        }
        for (b, &(c, t)) in mix.sigma_override_addr.iter().enumerate() {
            if c == cls {
                if let Some(f) = cp.sigma_fixed.get_mut(t) {
                    *f = mix.sigma_override_fixed.get(b).copied().unwrap_or(false);
                }
            }
        }
        mix.omega_override_fixed.iter_mut().for_each(|f| *f = true);
        mix.sigma_override_fixed.iter_mut().for_each(|f| *f = true);
    }
    let x_k = pack_params(&cp);
    let bounds = crate::estimation::parameterization::compute_bounds(&cp);
    let _guard = MixtureClassGuard::enter(cls + 1);
    if options.agq_nodes().is_some() {
        // The quadrature objective's own reconverged difference, as a single population's
        // salvage takes it; a subject repelled at `x` contributes zero, as the FOCE one does.
        let context = crate::estimation::agq::SubjectScoreContext::new(
            model, &cp, &x_k, options, &bounds, true,
        );
        return context
            .reconverged_score(subject, eta.as_slice())
            .unwrap_or_else(|| vec![0.0; x_k.len()]);
    }
    crate::estimation::outer_optimizer::subject_reconverged_fd_gradient(
        &x_k, &cp, model, subject, eta, &bounds, options,
    )
}

/// θ → packed chain factor `dθ_j/dx_j`: `θ_j` for a log-packed θ, 1 for an identity one.
fn theta_chain_factors(params: &ModelParameters) -> Vec<f64> {
    (0..params.theta.len())
        .map(|j| {
            if theta_packs_log(params.theta_lower[j]) {
                params.theta[j]
            } else {
                1.0
            }
        })
        .collect()
}

/// Add `scale ·` one class's packed gradient `g_c` into the mixture's packed gradient
/// `grad` — the one routing both [`mixture_gradient`] and the covariance step's
/// per-subject score (#1956) use.
///
/// `g_c` is over the class-`cls` layout `pack_params(&class_params(params, cls))`, whose
/// base Ω/Σ coordinates hold *that class's* values. θ, Ω_IOV (κ) and residual-correlation
/// coordinates are shared by every class and route to themselves. A base Ω/Σ coordinate
/// routes to the class's override coordinate when the class overrides it (`omega(k)` /
/// `sigma(k)`), else to the shared base coordinate. Overrides require a diagonal base Ω
/// (parse-time rule), so a base Ω coordinate's index is its eta's index whenever an
/// override can name it. The class layout's own override tail carries no dependence
/// (`class_params` has already swapped the class's values into the base slots) and is
/// not read.
fn route_class_gradient(
    grad: &mut [f64],
    g_c: &[f64],
    scale: f64,
    cls: usize,
    params: &ModelParameters,
    mp: &MixtureParams,
) {
    for (s, dest) in class_route(cls, params, mp).into_iter().enumerate() {
        if let Some(d) = dest {
            grad[d] += scale * g_c[s];
        }
    }
}

/// [`route_class_gradient`]'s map: for each coordinate of the class-`cls` layout, the
/// mixture coordinate it lands on, `None` for the class layout's inert override tail.
fn class_route(cls: usize, params: &ModelParameters, mp: &MixtureParams) -> Vec<Option<usize>> {
    let seg = packed_segments(params);
    let mut dest: Vec<Option<usize>> = vec![None; seg.total()];
    for (j, d) in dest.iter_mut().enumerate().take(seg.theta) {
        *d = Some(j);
    }
    for e in 0..seg.omega {
        let s = seg.omega_start() + e;
        dest[s] = Some(
            mp.omega_override_addr
                .iter()
                .position(|&(c, ee)| c == cls && ee == e)
                .map(|a| seg.mixture_omega_start() + a)
                .unwrap_or(s),
        );
    }
    for t in 0..seg.sigma {
        let s = seg.sigma_start() + t;
        dest[s] = Some(
            mp.sigma_override_addr
                .iter()
                .position(|&(c, tt)| c == cls && tt == t)
                .map(|b| seg.mixture_sigma_start() + b)
                .unwrap_or(s),
        );
    }
    for s in (seg.iov_start()..seg.mixture_omega_start()).chain(seg.rho_start()..seg.total()) {
        dest[s] = Some(s);
    }
    dest
}

/// The exact covariance R-matrix of the mixture objective, `∂²OFV/∂x²` over the mixture's
/// packed layout at `x_hat`, or `None` (the caller keeps the stencil) when any class is
/// outside the adjoint's scope, a class mode does not re-converge, or the mixing map declines.
///
/// One cold `mixture_ofv` at the covariance inner tolerance finds the class modes, and
/// [`mixture_candidate_hessian`] — the one implementation the trust region shares — takes the
/// derivatives at `x_hat` along every axis, warm from them.
pub(crate) fn mixture_cov_hessian(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x_hat: &[f64],
    options: &FitOptions,
) -> Option<DMatrix<f64>> {
    let params = unpack_params(x_hat, template);
    let n = packed_segments(&params).total();
    if x_hat.len() != n || params.mixture.is_none() {
        return None;
    }
    let score_options = cov_score_options(model, options);
    let (_, modes) = mixture_ofv_with_modes(model, population, &params, &score_options, None);
    let axes: Vec<usize> = (0..n).collect();
    mixture_candidate_hessian(model, population, template, x_hat, &modes, options, &axes)
        .ok()
        .map(|(_, h)| h)
}

/// The covariance step's options: the inner tolerance every exact mixture derivative
/// re-solves its modes at.
fn cov_score_options(model: &CompiledModel, options: &FitOptions) -> FitOptions {
    FitOptions {
        inner_tol: options.effective_cov_inner_tol(model.uses_closed_form_ltbs_inner()),
        ..options.clone()
    }
}

/// The exact gradient and Hessian of the mixture objective `OFV = −2 Σᵢ log Σₖ pᵢₖ e^{−F_ik}`
/// along `axes` (mixture packed coordinates, in `axes` order), at `x`, both from one pass at
/// the class modes `warm` re-solved **warm** at the covariance inner tolerance — the
/// derivatives the trust region models each iterate with, and (at `x̂`, every axis) the
/// covariance step's R-matrix. The same objective as [`mixture_ofv`]: the same class
/// probabilities, class parameters and inner-tolerance handling, the class objective being
/// whichever the fit scores (FOCE, FOCEI, Laplace, AGQ).
///
/// [`ExactDecline::OutOfScope`]: `analytic_cov_hessian = false`, `gradient = fd`, a model no
/// recorder serves, or a subject a recorder declines; the caller takes the finite-difference
/// Hessian of the same objective. [`ExactDecline::NotResolved`]: in scope but not at this
/// point — a class mode that did not re-converge (the start is not re-searched from cold, so a
/// class cannot jump basins), a non-finite term, or a cancelled run.
pub(crate) fn mixture_candidate_hessian(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    warm: &MixtureClassModes,
    options: &FitOptions,
    axes: &[usize],
) -> Result<(Vec<f64>, DMatrix<f64>), ExactDecline> {
    let params = unpack_params(x, template);
    let n_subjects = population.subjects.len();
    let k_classes = params.mixture.as_ref().map(|m| m.omega.len());
    if x.len() != packed_segments(&params).total()
        || k_classes != Some(warm.etas.len())
        || warm.etas.iter().any(|c| c.len() != n_subjects)
    {
        return Err(ExactDecline::OutOfScope);
    }
    // A warm re-solve: the class modes are tracked, not searched for (no multi-start).
    let score_options = FitOptions {
        inner_restarts: 0,
        ..cov_score_options(model, options)
    };
    let (eval, modes) =
        mixture_ofv_with_modes(model, population, &params, &score_options, Some(&warm.etas));
    if crate::cancel::is_cancelled(&options.cancel) || eval.ebe_stats.n_unconverged > 0 {
        return Err(ExactDecline::NotResolved);
    }
    mixture_derivatives_at_modes(model, population, &params, &eval, &modes, options, axes)
}

/// [`mixture_candidate_hessian`]'s assembly at class modes already solved (`eval`, `modes`).
///
/// Per subject, with `u_k = ln p_k − F_k` (`F_k` the class objective at its own mode) and
/// `w_k = PMIX_k`, the marginal is `−LSE_k u_k`, so
///
/// ```text
/// ∇(−log Lᵢ)  = −ḡ,
/// ∇²(−log Lᵢ) = Σ_k w_k (∇²F_k − ∇² ln p_k) − Σ_k w_k ∇u_k ∇u_kᵀ + ḡ ḡᵀ,   ḡ = Σ_k w_k ∇u_k.
/// ```
///
/// `∇F_k`, `∇²F_k` are the adjoint's exact profile gradient and Hessian on the class layout
/// (class modes, `MIXNUM = k`), routed onto the mixture layout by [`class_route`]; `∇ ln p_k`,
/// `∇² ln p_k` are the mixing programs' exact θ-derivatives, chained through θ's packing. The
/// OFV is `2 Σᵢ`. Along a proper subset of the axes each class's Hessian is the adjoint's block
/// on just the class coordinates that route onto an axis (the full Hessian's sub-block to the
/// bit); along all of them it is the full Hessian.
fn mixture_derivatives_at_modes(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eval: &MixtureEval,
    modes: &MixtureClassModes,
    options: &FitOptions,
    axes: &[usize],
) -> Result<(Vec<f64>, DMatrix<f64>), ExactDecline> {
    use crate::sens::ode_provider::adjoint::{
        subject_adjoint_cov_block_at, subject_adjoint_cov_hessian_at, subject_adjoint_gradient_at,
        PackedPoint,
    };
    if !options.analytic_cov_hessian
        || !crate::sens::ode_provider::adjoint::recorder_in_scope(model)
        || crate::types::GradientMethod::forced_fd(model)
    {
        return Err(ExactDecline::OutOfScope);
    }
    let spec = model.mixture.as_ref().ok_or(ExactDecline::OutOfScope)?;
    let mp = params.mixture.as_ref().ok_or(ExactDecline::OutOfScope)?;
    let objective = crate::estimation::covariance::cov_objective(options);
    let n = packed_segments(params).total();
    // The mixture coordinate -> its position on the axes.
    let mut pos: Vec<Option<usize>> = vec![None; n];
    for (p, &a) in axes.iter().enumerate() {
        *pos.get_mut(a).ok_or(ExactDecline::OutOfScope)? = Some(p);
    }
    let full = axes.len() == n && pos.iter().all(Option::is_some);
    let m = axes.len();
    let classes: Vec<(ModelParameters, Vec<f64>, Vec<Option<usize>>, Vec<usize>)> =
        (0..mp.omega.len())
            .map(|cls| {
                let cp = class_params(params, cls);
                let xc = pack_params(&cp);
                let route = class_route(cls, params, mp);
                // The class coordinates that land on an axis.
                let cax: Vec<usize> = (0..route.len())
                    .filter(|&s| route[s].is_some_and(|d| pos[d].is_some()))
                    .collect();
                (cp, xc, route, cax)
            })
            .collect();
    let chain = theta_chain_factors(params);
    let log_packed: Vec<bool> = params
        .theta_lower
        .iter()
        .map(|&l| theta_packs_log(l))
        .collect();
    let n_theta = params.theta.len();
    type Term = Result<(DVector<f64>, DMatrix<f64>), ExactDecline>;
    let per_subject: Vec<Term> = crate::par::map(population.subjects.len(), |i| -> Term {
        let subject = &population.subjects[i];
        let (dlnp, d2lnp) =
            mixing_logp_derivs(model, spec, &params.theta, &subject.covariates, true)
                .ok_or(ExactDecline::OutOfScope)?;
        let mut h = DMatrix::<f64>::zeros(m, m);
        let mut gbar = DVector::<f64>::zeros(m);
        for (cls, (cp, xc, route, cax)) in classes.iter().enumerate() {
            let w = eval.pmix[i][cls];
            if w == 0.0 {
                continue;
            }
            let _guard = MixtureClassGuard::enter(cls + 1);
            let pt = PackedPoint::new(cp, xc).ok_or(ExactDecline::OutOfScope)?;
            let b = crate::estimation::covariance::stacked_mode(
                model,
                subject,
                &eval.etas_by_class[cls][i],
                modes.kappas[cls].get(i),
            )
            .ok_or(ExactDecline::NotResolved)?;
            let g_c = subject_adjoint_gradient_at(model, subject, &pt, &b, objective)
                .ok_or(ExactDecline::OutOfScope)?;
            let h_c = if full {
                subject_adjoint_cov_hessian_at(model, subject, &pt, &b, objective)
            } else {
                subject_adjoint_cov_block_at(model, subject, &pt, &b, objective, cax)
            }
            .ok_or(ExactDecline::OutOfScope)?;
            if g_c.len() != route.len() || h_c.nrows() != if full { route.len() } else { cax.len() }
            {
                return Err(ExactDecline::OutOfScope);
            }
            // ∇u_k = ∇ln p_k − ∇F_k on the axes; the class Hessian scattered onto them.
            let mut du = DVector::<f64>::zeros(m);
            for (ia, &sa) in cax.iter().enumerate() {
                let Some(pa) = route[sa].and_then(|d| pos[d]) else {
                    continue;
                };
                du[pa] -= g_c[sa];
                for (ic, &sc) in cax.iter().enumerate() {
                    let Some(pc) = route[sc].and_then(|d| pos[d]) else {
                        continue;
                    };
                    let v = if full { h_c[(sa, sc)] } else { h_c[(ia, ic)] };
                    h[(pa, pc)] += w * v;
                }
            }
            for j in 0..n_theta {
                let Some(pj) = pos[j] else { continue };
                du[pj] += dlnp[cls][j] * chain[j];
                for l in 0..n_theta {
                    let Some(pl) = pos[l] else { continue };
                    let mut d = d2lnp[cls][j * n_theta + l] * chain[j] * chain[l];
                    if j == l && log_packed[j] {
                        // d(θ_j)/dx_j = θ_j on a log-packed coordinate.
                        d += dlnp[cls][j] * chain[j];
                    }
                    h[(pj, pl)] -= w * d;
                }
            }
            h -= w * &du * du.transpose();
            gbar += w * &du;
        }
        h += &gbar * gbar.transpose();
        if h.iter().all(|v| v.is_finite()) && gbar.iter().all(|v| v.is_finite()) {
            Ok((gbar, h))
        } else {
            Err(ExactDecline::NotResolved)
        }
    });
    if crate::cancel::is_cancelled(&options.cancel) {
        return Err(ExactDecline::NotResolved);
    }
    let mut g = vec![0.0f64; m];
    let mut acc = DMatrix::<f64>::zeros(m, m);
    for r in per_subject {
        let (gbar, h) = r?;
        for (gk, v) in g.iter_mut().zip(gbar.iter()) {
            *gk -= 2.0 * v;
        }
        acc += h;
    }
    Ok((g, 2.0 * acc))
}

/// Subtract `scale · ∂ln p_ik/∂x` (θ coordinates only) from `grad`, given the class's
/// `∂ln p_ik/∂θ` row `dlnp_c` and the θ chain factors.
fn subtract_mixing_gradient(grad: &mut [f64], dlnp_c: &[f64], scale: f64, dtheta_dx: &[f64]) {
    for j in 0..dtheta_dx.len() {
        grad[j] -= scale * dlnp_c[j] * dtheta_dx[j];
    }
}

/// One class's view for the per-subject score: the class template, its packed point
/// and its FD box.
struct ClassScoreView {
    template: ModelParameters,
    x: Vec<f64>,
    bounds: PackedBounds,
}

/// Per-subject scores of the **mixture** objective for the covariance step's
/// score cross-product `S = Σᵢ gᵢgᵢᵀ` (#1956), built once at `x̂`.
///
/// The subject's marginal is `−log Lᵢ = −log Σₖ pᵢₖ e^{−nllᵢₖ}`, so its score is
///
/// ```text
/// gᵢ = Σₖ wᵢₖ (∂nllᵢₖ/∂x − ∂ln pᵢₖ/∂x),   wᵢₖ = PMIXᵢₖ
/// ```
///
/// on the −logL scale (S's scale; [`mixture_gradient`] is the same sum ×2 over
/// subjects). Each `dnllᵢₖ/dx` is the single-population total-derivative score
/// (`covariance::subject_total_score`, the class mode re-estimated at every `x` — NONMEM's
/// `MATRIX=S` convention, the one every other S uses since #2974) evaluated under
/// `MIXNUM = k`, with class `k`'s own η̂/κ̂ and Ω/Σ, then routed onto the mixture
/// layout by `route_class_gradient` — the routing the outer gradient uses. The modes
/// come from one cold `mixture_ofv` at `x̂` and the covariance inner tolerance, the
/// same evaluation `compute_covariance` takes its base OFV from.
pub(crate) struct MixtureScoreContext {
    params: ModelParameters,
    eval: MixtureEval,
    modes: MixtureClassModes,
    classes: Vec<ClassScoreView>,
    dtheta_dx: Vec<f64>,
    /// The covariance step's options: the inner tolerance the modes were solved at, and the one
    /// the reconverged fallback's probes re-solve at (#2261).
    score_options: FitOptions,
    /// The mixing derivatives at `params.theta`, shared by the subjects' scores.
    mixing: MixingGradCache,
}

impl MixtureScoreContext {
    pub(crate) fn new(
        model: &CompiledModel,
        population: &Population,
        template: &ModelParameters,
        x_hat: &[f64],
        options: &FitOptions,
    ) -> Self {
        let params = unpack_params(x_hat, template);
        let score_options = FitOptions {
            inner_tol: options.effective_cov_inner_tol(model.uses_closed_form_ltbs_inner()),
            ..options.clone()
        };
        let (eval, modes) =
            mixture_ofv_with_modes(model, population, &params, &score_options, None);
        let mp = params
            .mixture
            .as_ref()
            .expect("MixtureScoreContext on non-mixture parameters");
        let classes = (0..mp.omega.len())
            .map(|cls| {
                let mut cp = class_params(&params, cls);
                // The class layout's base Ω/Σ coordinates hold this class's values, so they
                // are held exactly when this class's value is: an override carries its own
                // FIX flag. The layout's override tail has no dependence at all (the class
                // values are already in the base slots), so hold it and the FD fallback
                // skips it.
                for (a, &(c, e)) in mp.omega_override_addr.iter().enumerate() {
                    if c == cls {
                        if let Some(f) = cp.omega_fixed.get_mut(e) {
                            *f = mp.omega_override_fixed[a];
                        }
                    }
                }
                for (b, &(c, t)) in mp.sigma_override_addr.iter().enumerate() {
                    if c == cls {
                        if let Some(f) = cp.sigma_fixed.get_mut(t) {
                            *f = mp.sigma_override_fixed[b];
                        }
                    }
                }
                if let Some(cm) = cp.mixture.as_mut() {
                    cm.omega_override_fixed.fill(true);
                    cm.sigma_override_fixed.fill(true);
                }
                let start = pack_with_bounds(&cp);
                ClassScoreView {
                    template: cp,
                    x: start.packed,
                    bounds: start.bounds,
                }
            })
            .collect();
        let dtheta_dx = theta_chain_factors(&params);
        Self {
            params,
            eval,
            modes,
            classes,
            dtheta_dx,
            score_options,
            mixing: MixingGradCache::new(model),
        }
    }

    /// Each class's scorer, built once for the population (not per subject × class): its packed
    /// point, and under Laplace / AGQ the single population's quadrature score context on the
    /// class template (#3025).
    pub(crate) fn class_scorers<'a>(&'a self, model: &'a CompiledModel) -> Vec<ClassScorer<'a>> {
        let quadrature = self.score_options.agq_nodes().is_some();
        self.classes
            .iter()
            .map(|v| {
                if quadrature {
                    ClassScorer {
                        point: None,
                        quadrature: Some(
                            crate::estimation::agq::SubjectScoreContext::new(
                                model,
                                &v.template,
                                &v.x,
                                &self.score_options,
                                &v.bounds,
                                crate::estimation::outer_optimizer::reconverge_this_eval(
                                    &self.score_options,
                                    0,
                                ),
                            )
                            .with_fd_inner_tol(self.score_options.inner_tol),
                        ),
                    }
                } else {
                    ClassScorer {
                        point: crate::sens::ode_provider::adjoint::PackedPoint::new(
                            &v.template,
                            &v.x,
                        ),
                        quadrature: None,
                    }
                }
            })
            .collect()
    }

    /// Subject `i`'s score `∂(−log Lᵢ)/∂x` over the full mixture packed layout; `scorers` is
    /// [`Self::class_scorers`]. `None` when a quadrature class has no finite score at `x̂`.
    pub(crate) fn score(
        &self,
        model: &CompiledModel,
        population: &Population,
        i: usize,
        scorers: &[ClassScorer<'_>],
    ) -> Option<Vec<f64>> {
        let spec = model
            .mixture
            .as_ref()
            .expect("mixture score on a non-mixture model");
        let mp = self.params.mixture.as_ref().expect("mixture parameters");
        let covariates = &population.subjects[i].covariates;
        let dlnp = self
            .mixing
            .rows(model, spec, &self.params.theta, covariates);
        let mut g = vec![0.0f64; packed_segments(&self.params).total()];
        for (cls, view) in self.classes.iter().enumerate() {
            let w = self.eval.pmix[i][cls];
            if w == 0.0 {
                continue;
            }
            // MIXNUM = cls+1 on this (worker) thread for the class's predictions.
            let _guard = MixtureClassGuard::enter(cls + 1);
            let g_c = if let Some(context) = &scorers[cls].quadrature {
                // The quadrature class's own score, the single population's (`Ψ = F − wᵀ∇_η l`,
                // else its reconverged difference), at the class mode.
                context.score(
                    &population.subjects[i],
                    self.eval.etas_by_class[cls][i].as_slice(),
                    &self.modes.kappas[cls][i],
                )?
            } else {
                // Asymmetry (#1973): the class kernel here is the single-population S's,
                // not `mixture_gradient`'s `subject_gradient_at`; only the routing
                // is shared, so a kernel change moves one of S and the outer gradient.
                crate::estimation::covariance::subject_total_score(
                    &view.x,
                    &view.template,
                    model,
                    population,
                    i,
                    &self.eval.etas_by_class[cls][i],
                    &self.modes.kappas[cls][i],
                    scorers[cls].point.as_ref(),
                    &view.bounds,
                    &self.score_options,
                )
            };
            route_class_gradient(&mut g, &g_c, w, cls, &self.params, mp);
            subtract_mixing_gradient(&mut g, &dlnp[cls], w, &self.dtheta_dx);
        }
        Some(g)
    }
}

/// One class's scorer for [`MixtureScoreContext::score`]: its packed point (FOCE / FOCEI) or the
/// quadrature score context on the class template (Laplace / AGQ).
pub(crate) struct ClassScorer<'a> {
    point: Option<crate::sens::ode_provider::adjoint::PackedPoint<'a>>,
    quadrature: Option<crate::estimation::agq::SubjectScoreContext<'a>>,
}

/// Central-FD outer gradient of the mixture OFV in real packed space — the test oracle
/// for the exact gradient ([`mixture_gradient_salvaged`]; no model reaches it in a fit,
/// #3004). Re-runs `mixture_ofv` at `x ± hᵢ` per coordinate (`2·n` cold evals).
///
/// `hᵢ = 1e-5·floorᵢ`, `floor` from `fd_step_floors`: a flat `1e-5` on every coordinate
/// but a small identity-packed θ, whose packed coordinate is the θ itself, so a flat step
/// is absolute in its own units — 1.2 % of a θ of 8e-4. There it is `1e-5·|θ₀|` (#2187).
///
/// Each probe is clamped to `init_params`' packed box — the box the outer optimizer
/// searches — and the difference divided by the clamped span, as `central_diff_packed_gated`
/// does, so a coordinate on a bound is differenced one-sidedly inside it rather than
/// evaluated at a point the optimizer can never reach. A FIX coordinate's box is pinned
/// to a point, so its span is zero and its gradient stays 0.
#[cfg(test)]
pub(crate) fn mixture_gradient_fd(
    model: &CompiledModel,
    population: &Population,
    x: &[f64],
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Vec<f64> {
    let floor = crate::estimation::parameterization::fd_step_floors(init_params, model);
    let bounds = crate::estimation::parameterization::compute_bounds(init_params);
    let n = x.len();
    let mut g = vec![0.0f64; n];
    for i in 0..n {
        let h = 1e-5 * floor[i];
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] = (x[i] + h).min(bounds.upper[i]);
        xm[i] = (x[i] - h).max(bounds.lower[i]);
        // Collapsed probes only — see `central_diff_packed_gated`: an absolute cutoff on the span
        // would zero a θ declared below ~1e-11 (#2187 review).
        if xp[i] <= xm[i] {
            continue;
        }
        let denom = xp[i] - xm[i];
        let pp = crate::estimation::parameterization::unpack_params(&xp, init_params);
        let pm = crate::estimation::parameterization::unpack_params(&xm, init_params);
        let fp = mixture_ofv(model, population, &pp, options, None).ofv;
        let fm = mixture_ofv(model, population, &pm, options, None).ofv;
        g[i] = (fp - fm) / denom;
    }
    g
}

#[cfg(test)]
mod tests {
    use super::{any_contributing, combine_subject};

    mod merged_classes {
        use super::super::{
            combine_subject, merged_classes_from, merged_classes_warning, ClassCollapse,
            EMPTY_CLASS_PMIX, MERGED_CLASS_LOG_BF,
        };

        /// `(nll, pmix)` per subject from per-class probabilities and nll's, with the
        /// posteriors built through the objective's own `combine_subject`, so the check reads
        /// them the way the fit builds them.
        fn posteriors(p: &[f64], nll: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
            let logp: Vec<f64> = p.iter().map(|v| v.ln()).collect();
            let pmix = nll.iter().map(|n| combine_subject(&logp, n).1).collect();
            (nll.to_vec(), pmix)
        }

        #[test]
        fn separated_classes_draw_no_finding() {
            // One subject decisively per class: factors of 6 nats each way.
            let (nl, pm) = posteriors(&[0.5, 0.5], &[vec![10.0, 16.0], vec![16.0, 10.0]]);
            assert!(merged_classes_from(&nl, &pm).is_empty());
            assert_eq!(merged_classes_warning(&[]), None);
        }

        #[test]
        fn a_pair_under_the_factor_is_merged_and_one_subject_over_it_separates() {
            // Every subject's data moves its odds by 0.4 of the threshold, under unequal
            // mixing (0.3 / 0.7): the prior does not enter the factor.
            let d = 0.4 * MERGED_CLASS_LOG_BF;
            let near = vec![vec![5.0, 5.0 + d], vec![7.0 + d, 7.0], vec![3.0, 3.0]];
            let (nl, pm) = posteriors(&[0.3, 0.7], &near);
            let got = merged_classes_from(&nl, &pm);
            match got.as_slice() {
                [ClassCollapse::Merged {
                    a: 1,
                    b: 2,
                    max_log_bf,
                }] => {
                    assert!((max_log_bf - d).abs() < 1e-9, "factor {max_log_bf} vs {d}");
                }
                other => panic!("want classes 1 and 2 merged, got {other:?}"),
            }
            // The threshold straddled: one subject at 1.5× the factor separates the pair.
            let mut far = near.clone();
            far.push(vec![4.0, 4.0 + 1.5 * MERGED_CLASS_LOG_BF]);
            let (nl, pm) = posteriors(&[0.3, 0.7], &far);
            assert!(merged_classes_from(&nl, &pm).is_empty());
        }

        #[test]
        fn three_classes_name_only_the_merged_pair() {
            // Classes 1 and 3 identical, class 2 far from both.
            let nll = vec![vec![5.0, 15.0, 5.0], vec![9.0, 2.0, 9.0]];
            let (nl, pm) = posteriors(&[0.2, 0.5, 0.3], &nll);
            let got = merged_classes_from(&nl, &pm);
            assert!(
                matches!(got.as_slice(), [ClassCollapse::Merged { a: 1, b: 3, .. }]),
                "{got:?}"
            );
            let w = merged_classes_warning(&got).unwrap();
            assert!(w.starts_with("W_MIXTURE_MERGED_CLASSES: "), "{w}");
            assert!(w.contains("classes 1 and 3 have merged"), "{w}");
            assert!(
                !w.contains("classes 1 and 2") && !w.contains("classes 2 and 3"),
                "{w}"
            );
            assert!(!w.contains("empty"), "{w}");
        }

        #[test]
        fn an_empty_class_is_reported_and_not_also_as_merged() {
            // Class 2's mixing probability is 1e-6, and its likelihood no better than class
            // 1's, so its posterior stays below the empty threshold in every subject. Equal
            // nll's would also make the pair "merged"; the empty finding replaces that.
            let p2 = 1e-6;
            let nll = vec![vec![5.0, 5.0], vec![8.0, 8.5]];
            let (nl, pm) = posteriors(&[1.0 - p2, p2], &nll);
            assert!(pm.iter().all(|p| p[1] < EMPTY_CLASS_PMIX));
            let got = merged_classes_from(&nl, &pm);
            assert!(
                matches!(got.as_slice(), [ClassCollapse::Empty { class: 2, .. }]),
                "{got:?}"
            );
            let w = merged_classes_warning(&got).unwrap();
            assert!(w.contains("class 2 is empty"), "{w}");
            assert!(!w.contains("merged ("), "{w}");
            // The straddle: one subject whose data favours class 2 strongly enough to give it
            // posterior mass makes the class non-empty, and the pair separated.
            let mut nll = nll;
            nll.push(vec![30.0, 5.0]);
            let (nl, pm) = posteriors(&[1.0 - p2, p2], &nll);
            assert!(merged_classes_from(&nl, &pm).is_empty());
        }

        #[test]
        fn a_subject_where_both_posteriors_underflow_still_counts_the_pair_merged() {
            // Review round 1: three classes, classes 1 and 2 identical. Subject 1 is dominated
            // by class 3 by ~1000 nats, so its PMIX_1 and PMIX_2 underflow to exactly 0, and a
            // factor rebuilt from PMIX would be 0 / 0. Read from the nll's it is 0.
            let nll = vec![
                vec![1005.0, 1005.0, 2.0],
                vec![3.0, 3.0, 20.0],
                vec![4.0, 4.0, 30.0],
            ];
            let (nl, pm) = posteriors(&[0.3, 0.3, 0.4], &nll);
            assert!(
                pm[0][0] == 0.0 && pm[0][1] == 0.0,
                "fixture must underflow: {pm:?}"
            );
            let got = merged_classes_from(&nl, &pm);
            assert!(
                matches!(got.as_slice(), [ClassCollapse::Merged { a: 1, b: 2, .. }]),
                "{got:?}"
            );
        }

        #[test]
        fn a_non_finite_class_objective_separates_the_pair() {
            // An infinite or NaN nll (a failed score) never produces the warning.
            let pm = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
            for bad in [f64::INFINITY, f64::NAN] {
                let nll = vec![vec![bad, bad], vec![2.0, 2.0]];
                assert!(merged_classes_from(&nll, &pm).is_empty(), "nll {bad}");
            }
        }

        #[test]
        fn a_non_finite_posterior_does_not_make_its_class_empty() {
            // Review round 1: class 2's posterior is tiny everywhere except one NaN, which a
            // NaN-dropping max would fold to a maximum below the threshold. Classes are kept
            // separated by their nll's, so the only possible finding is the empty one.
            let nll = vec![vec![1.0, 10.0], vec![1.0, 10.0]];
            let tiny = vec![vec![1.0, 1e-6], vec![1.0, 1e-6]];
            assert!(matches!(
                merged_classes_from(&nll, &tiny).as_slice(),
                [ClassCollapse::Empty { class: 2, .. }]
            ));
            let nan = vec![vec![1.0, 1e-6], vec![f64::NAN, f64::NAN]];
            assert!(merged_classes_from(&nll, &nan).is_empty());
        }
    }

    #[test]
    fn any_contributing_scores_nonwinning_classes() {
        // #984 regression: a non-winning class that failed to converge (class 2 here,
        // flag = true) but still carries non-negligible posterior weight must be
        // surfaced, even though class 1 wins.
        let flags = [false, true]; // class 2 failed
        let winner_heavy = [0.98, 0.02]; // class 1 wins, class 2 still contributes
        assert!(
            any_contributing(&flags, &winner_heavy),
            "non-winning contributing class must count"
        );
        // A failed class with negligible weight cannot move the marginal → not counted.
        let winner_only = [1.0 - 1e-9, 1e-9];
        assert!(!any_contributing(&flags, &winner_only));
        // No class flagged → nothing counted.
        assert!(!any_contributing(&[false, false], &winner_heavy));
    }

    #[test]
    fn combine_two_equal_classes() {
        // Equal mixing (ln 0.5 each) and equal nll → contribution collapses to the
        // single-class 2·nll, and posterior is 50/50.
        let ln_half = 0.5_f64.ln();
        let (contrib, pmix, mixest) = combine_subject(&[ln_half, ln_half], &[3.0, 3.0]);
        // L_i = 0.5 e^-3 + 0.5 e^-3 = e^-3 → -2 log L = 6.
        assert!((contrib - 6.0).abs() < 1e-12, "contrib = {contrib}");
        assert!((pmix[0] - 0.5).abs() < 1e-12 && (pmix[1] - 0.5).abs() < 1e-12);
        assert_eq!(mixest, 0); // tie → first
    }

    #[test]
    fn combine_prefers_lower_nll_class() {
        // Class 2 fits much better (lower nll) → posterior mass and MIXEST there.
        let ln_half = 0.5_f64.ln();
        let (_c, pmix, mixest) = combine_subject(&[ln_half, ln_half], &[10.0, 1.0]);
        assert_eq!(mixest, 1);
        assert!(pmix[1] > 0.99, "class-2 posterior {}", pmix[1]);
    }

    #[test]
    fn combine_matches_hand_computed() {
        // p = [0.3, 0.7], nll = [2.0, 4.0].
        // L = 0.3 e^-2 + 0.7 e^-4 = 0.0405999.. + 0.0128198.. = 0.0534197..
        let logp = [0.3_f64.ln(), 0.7_f64.ln()];
        let (contrib, pmix, _m) = combine_subject(&logp, &[2.0, 4.0]);
        let l = 0.3 * (-2.0_f64).exp() + 0.7 * (-4.0_f64).exp();
        assert!((contrib - (-2.0 * l.ln())).abs() < 1e-10);
        let p0 = 0.3 * (-2.0_f64).exp() / l;
        assert!((pmix[0] - p0).abs() < 1e-10);
    }

    /// The mixing map's exact θ-gradient and Hessian on both forms, at a θ far below 1
    /// (#2147: the difference quotients this replaced stepped it across zero). With
    /// `g = sqrt(TINY)`, `g′ = 1/(2g)`, `g″ = −g′/(2·TINY)`: the logit form's
    /// `ln p₁ = g − ln(1 + eᵍ)`, `ln p₂ = −ln(1 + eᵍ)` and the p form's `ln g`, `ln(1 − g)`,
    /// differentiated by hand; `TVCL` reads neither, so its row and column are zero.
    #[test]
    fn the_mixing_map_derivatives_are_exact_at_a_tiny_theta() {
        use crate::parser::model_parser::mixing_logp_derivs;
        use std::collections::HashMap;

        let tiny = 7.7e-8_f64;
        let g = tiny.sqrt();
        let (dg, d2g) = (0.5 / g, -0.25 / (g * tiny));
        let none = HashMap::new();
        for logit in [true, false] {
            let mixing = if logit {
                "logit(1) = sqrt(TINY)"
            } else {
                "p(1) = sqrt(TINY)"
            };
            let src = format!(
                "[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TINY({tiny:e}, -1.0, 1.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.1
[mixture]
  nsub = 2
  {mixing}
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
[structural_model]
  pk one_cpt_iv(cl=CL, v=10.0)
[error_model]
  DV ~ proportional(EPS)
"
            );
            let model = crate::parser::model_parser::parse_model_string(&src).unwrap();
            let spec = model.mixture.as_ref().unwrap();
            let theta = &model.default_params.theta;
            let floor = crate::estimation::parameterization::fd_theta_step_floors(&model);
            assert_eq!(floor, vec![1.0, tiny], "{mixing}: floors");
            let (grad, hess) = mixing_logp_derivs(&model, spec, theta, &none, true).expect("exact");
            let (d, d2) = if logit {
                let p1 = 1.0 / (1.0 + (-g).exp());
                let q = -p1 * (1.0 - p1) * dg * dg;
                (
                    [(1.0 - p1) * dg, -p1 * dg],
                    [q + (1.0 - p1) * d2g, q - p1 * d2g],
                )
            } else {
                let (a, b) = (dg / g, dg / (1.0 - g));
                ([a, -b], [d2g / g - a * a, -d2g / (1.0 - g) - b * b])
            };
            for c in 0..2 {
                // Measured at rounding (the closed form and the recording associate
                // differently).
                let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
                assert!(
                    rel(grad[c][1], d[c]) < 1e-12,
                    "{mixing} class {}: grad",
                    c + 1
                );
                assert!(
                    rel(hess[c][3], d2[c]) < 1e-10,
                    "{mixing} class {}: hess",
                    c + 1
                );
                assert_eq!([grad[c][0], hess[c][0], hess[c][1], hess[c][2]], [0.0; 4]);
            }
        }
    }

    // ── End-to-end fit() smoke / convergence tests ──

    /// Two-class bimodal-CL 1-cpt IV dataset: `n_per` subjects per group, group A
    /// (low WT) with `cl_a`, group B (high WT) with `cl_b`. Deterministic mild
    /// proportional perturbation so sigma is identifiable.
    fn bimodal_csv(n_per: usize, cl_a: f64, cl_b: f64) -> String {
        let v = 10.0_f64;
        let dose = 100.0_f64;
        let times = [0.5_f64, 1.0, 2.0, 4.0, 8.0];
        let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,WT\n");
        let mut sid = 0;
        for (grp, &cl) in [cl_a, cl_b].iter().enumerate() {
            let wt = if grp == 0 { 60.0 } else { 90.0 };
            for _ in 0..n_per {
                sid += 1;
                s.push_str(&format!("{sid},0,0,{dose},1,1,{wt}\n"));
                for (ti, &t) in times.iter().enumerate() {
                    let c = (dose / v) * (-(cl / v) * t).exp();
                    // Deterministic ~±3% ripple keyed by (subject,time).
                    let ripple = 1.0 + 0.03 * (((sid + ti) as f64) * 1.3).sin();
                    let dv = c * ripple;
                    s.push_str(&format!("{sid},{t},{dv:.5},0,0,1,{wt}\n"));
                }
            }
        }
        s
    }

    const MIX_FIT_MODEL: &str = r"
[parameters]
  theta TVCL1(1.2, 0.01, 100.0)
  theta TVCL2(2.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  theta BWT(0.0, -5.0, 5.0)
  omega ETA_CL ~ 0.05
  sigma EPS ~ 0.01

[mixture]
  nsub = 2
  logit(1) = MIXL + BWT*(WT - 75)

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    fn read_pop(csv: &str) -> crate::types::Population {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        crate::io::datareader::read_nonmem_csv(f.path(), Some(&["WT"]), None).unwrap()
    }

    // ── IOV + mixture (#985) ──

    /// Two-class bimodal-CL 1-cpt IV dataset with **two dosing occasions** per
    /// subject (`OCC` column), so an IOV-on-CL mixture has per-occasion κ to
    /// estimate. Occasion 1: dose at t=0, obs 0.5/1/2/4 h; occasion 2: dose at
    /// t=24, obs 24.5/25/26/28 h. Deterministic ±3 % ripple keeps σ identifiable.
    fn bimodal_iov_csv(n_per: usize, cl_a: f64, cl_b: f64) -> String {
        let v = 10.0_f64;
        let dose = 100.0_f64;
        let occ_start = [0.0_f64, 24.0];
        let rel_times = [0.5_f64, 1.0, 2.0, 4.0];
        let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,OCC,WT\n");
        let mut sid = 0;
        for (grp, &cl) in [cl_a, cl_b].iter().enumerate() {
            let wt = if grp == 0 { 60.0 } else { 90.0 };
            for _ in 0..n_per {
                sid += 1;
                for (occ, &t0) in occ_start.iter().enumerate() {
                    let occ_idx = occ + 1;
                    s.push_str(&format!("{sid},{t0},0,{dose},1,1,{occ_idx},{wt}\n"));
                    for (ti, &dt) in rel_times.iter().enumerate() {
                        let c = (dose / v) * (-(cl / v) * dt).exp();
                        let ripple = 1.0 + 0.03 * (((sid + ti + occ) as f64) * 1.3).sin();
                        let dv = c * ripple;
                        let t = t0 + dt;
                        s.push_str(&format!("{sid},{t},{dv:.5},0,0,1,{occ_idx},{wt}\n"));
                    }
                }
            }
        }
        s
    }

    fn read_pop_iov(csv: &str) -> crate::types::Population {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        crate::io::datareader::read_nonmem_csv(f.path(), Some(&["WT"]), Some("OCC")).unwrap()
    }

    /// MIXNUM-branched CL with a per-occasion κ on CL (`+ KAPPA_CL`), constant
    /// `p(1)` mixing, diagonal IOV Ω. The IOV segment is packed between σ and the
    /// (here empty) per-class override tail, so this exercises the #985 coexistence.
    const MIX_IOV_MODEL: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta P1(0.5, 0.001, 0.999)
  omega ETA_CL ~ 0.05
  kappa KAPPA_CL ~ 0.02
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  p(1) = P1

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL + KAPPA_CL) else TVCL2 * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    #[test]
    fn iov_mixture_fit_runs_and_emits_posteriors() {
        // #985: IOV + mixture, previously rejected at fit(). The per-class inner
        // solve routes to the IOV EBE (find_ebe → find_ebe_iov) and the marginal
        // uses foce_subject_nll_iov. fit() returns Ok with a finite OFV and every
        // subject carries a K=2 posterior, exactly like a non-IOV mixture fit.
        let model = crate::parser::model_parser::parse_model_string(MIX_IOV_MODEL).unwrap();
        assert_eq!(model.n_kappa, 1, "one IOV kappa on CL");
        let pop = read_pop_iov(&bimodal_iov_csv(4, 1.0, 3.0));
        let opts = crate::types::FitOptions {
            outer_maxiter: 3,
            ..crate::types::FitOptions::default()
        };
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("IOV + mixture fit should return Ok");
        assert!(res.ofv.is_finite(), "OFV = {}", res.ofv);
        for sr in &res.subjects {
            let pmix = sr
                .pmix
                .as_ref()
                .expect("PMIX populated for an IOV mixture fit");
            assert_eq!(pmix.len(), 2);
            assert!((pmix.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(sr.mixest.is_some());
        }
    }

    #[test]
    fn iov_mixture_postfit_carries_kappa_ebes() {
        // #985 regression (PR #986 review): the final mixture branch used to return `Vec::new()` for
        // the per-subject κ EBEs (MixtureEval carried none), so postfit ran with κ = 0
        // — sdtab IPRED/IWRES/CWRES, per-subject OFV, κ shrinkage, and the `.fitrx`
        // `ebe_kappas` export all silently ignored the IOV the fit actually used. The
        // MIXEST-class κ̂ must now flow through to `FitResult.ebe_kappas` (one entry
        // per subject, correctly dimensioned) and to a populated κ shrinkage vector.
        let model = crate::parser::model_parser::parse_model_string(MIX_IOV_MODEL).unwrap();
        let pop = read_pop_iov(&bimodal_iov_csv(4, 1.0, 3.0));
        let opts = crate::types::FitOptions {
            outer_maxiter: 3,
            ..crate::types::FitOptions::default()
        };
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("IOV + mixture fit should return Ok");

        // One κ set per subject, each with two occasions (OCC 1 and 2), each a
        // length-1 κ vector (single KAPPA_CL). Empty ⇒ the postfit-drops-κ bug.
        assert_eq!(
            res.ebe_kappas.len(),
            pop.subjects.len(),
            "ebe_kappas must have one entry per subject"
        );
        for ks in &res.ebe_kappas {
            assert_eq!(ks.len(), 2, "two occasions per subject");
            assert!(
                ks.iter().all(|k| k.len() == 1),
                "one κ (KAPPA_CL) per occasion"
            );
        }
        // κ is genuinely estimated (not a zero placeholder): at least one occasion of
        // one subject moved off the prior mode.
        assert!(
            res.ebe_kappas
                .iter()
                .flatten()
                .any(|k| k.iter().any(|&v| v.abs() > 1e-6)),
            "at least one κ̂ should be non-zero"
        );
        // κ shrinkage is reported (one per kappa parameter) — empty before the fix,
        // since fit.rs gates it on `ebe_kappas` being non-empty.
        assert_eq!(
            res.shrinkage_kappa.len(),
            1,
            "pooled κ shrinkage populated for the IOV mixture"
        );
    }

    /// #1932: the evaluation-only branch (`outer_maxiter = 0`) carries the MIXEST-class κ̂
    /// into postfit, as #986 made the converged branch do. The twin is the converged branch
    /// at the same point: every parameter `FIX` and `outer_maxiter = 5`. The pair straddles
    /// `evaluate_at_initial_params`, so dropping κ̂ from either branch reddens it: the
    /// branch that drops it fails its own "populated" check, and the diagnostics part.
    #[test]
    fn iov_mixture_eval_only_carries_kappa_ebes() {
        let src: String = MIX_IOV_MODEL
            .lines()
            .map(|l| {
                let t = l.trim_start();
                if ["theta ", "omega ", "kappa ", "sigma "]
                    .iter()
                    .any(|p| t.starts_with(p))
                {
                    format!("{l} FIX\n")
                } else {
                    format!("{l}\n")
                }
            })
            .collect();
        let model = crate::parser::model_parser::parse_model_string(&src).unwrap();
        let pop = read_pop_iov(&bimodal_iov_csv(4, 1.0, 3.0));
        let run = |outer_maxiter: usize| {
            let opts = crate::types::FitOptions {
                outer_maxiter,
                ..crate::types::FitOptions::default()
            };
            crate::api::fit(&model, &pop, &model.default_params, &opts)
                .expect("all-FIX IOV + mixture fit should return Ok")
        };
        let (eval, conv) = (run(0), run(5));
        // The same point on both branches. The marginal integrates κ itself, so the OFV
        // agrees with or without the fix; what the fix moves is everything postfit
        // computes from the per-subject κ̂.
        assert_eq!(
            eval.ofv.to_bits(),
            conv.ofv.to_bits(),
            "eval-only {} vs converged {}",
            eval.ofv,
            conv.ofv
        );
        for (name, res) in [("eval-only", &eval), ("converged", &conv)] {
            assert_eq!(
                res.ebe_kappas.len(),
                pop.subjects.len(),
                "{name}: one κ̂ set per subject"
            );
            assert!(
                res.ebe_kappas
                    .iter()
                    .flatten()
                    .any(|k| k.iter().any(|&v| v.abs() > 1e-6)),
                "{name}: κ̂ is estimated, not a zero placeholder"
            );
            assert!(
                res.shrinkage_kappa.len() == 1 && res.shrinkage_kappa[0].is_finite(),
                "{name}: κ shrinkage {:?}",
                res.shrinkage_kappa
            );
        }
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&eval.shrinkage_kappa),
            bits(&conv.shrinkage_kappa),
            "κ shrinkage: eval-only {:?} vs converged {:?}",
            eval.shrinkage_kappa,
            conv.shrinkage_kappa
        );
        for (e, c) in eval.subjects.iter().zip(&conv.subjects) {
            assert_eq!(bits(&e.ipred), bits(&c.ipred), "IPRED, subject {}", e.id);
            assert_eq!(bits(&e.iwres), bits(&c.iwres), "IWRES, subject {}", e.id);
            assert_eq!(bits(&e.cwres), bits(&c.cwres), "CWRES, subject {}", e.id);
            assert_eq!(
                e.ofv_contribution.to_bits(),
                c.ofv_contribution.to_bits(),
                "per-subject OFV, subject {}",
                e.id
            );
        }
        for (ek, ck) in eval.ebe_kappas.iter().zip(&conv.ebe_kappas) {
            for (a, b) in ek.iter().zip(ck) {
                assert_eq!(bits(a.as_slice()), bits(b.as_slice()), "κ̂");
            }
        }
    }

    #[test]
    fn iov_mixture_ofv_differs_from_no_iov_marginal() {
        // The IOV marginal must actually enter the objective: at the same params, a
        // κ-augmented mixture OFV (foce_subject_nll_iov) differs from what the plain
        // per-class marginal would give. We compare mixture_ofv on the IOV model to
        // the same data/model with the kappa stripped (n_kappa = 0), which routes
        // through the legacy foce_subject_nll — they must not coincide.
        let iov = crate::parser::model_parser::parse_model_string(MIX_IOV_MODEL).unwrap();
        let no_iov_src = MIX_IOV_MODEL
            .replace("  kappa KAPPA_CL ~ 0.02\n", "")
            .replace(" + KAPPA_CL", "");
        let no_iov = crate::parser::model_parser::parse_model_string(&no_iov_src).unwrap();
        let csv = bimodal_iov_csv(4, 1.0, 3.0);
        let pop_iov = read_pop_iov(&csv);
        let pop_no = read_pop_iov(&csv);
        let opts = crate::types::FitOptions::default();
        let o_iov = super::mixture_ofv(&iov, &pop_iov, &iov.default_params, &opts, None).ofv;
        let o_no = super::mixture_ofv(&no_iov, &pop_no, &no_iov.default_params, &opts, None).ofv;
        assert!(o_iov.is_finite() && o_no.is_finite());
        assert!(
            (o_iov - o_no).abs() > 1e-6,
            "IOV marginal must change the OFV: iov {o_iov} vs no-iov {o_no}"
        );
    }

    /// #1722 review finding 1: each class's EBE solve follows the stage's residual-variance
    /// convention — under a non-interaction stage with a proportional error, bit-identical to
    /// the frozen-variance solve of that class (under its `MIXNUM` guard), under interaction
    /// to the conditional one; the two differ on this fixture.
    #[test]
    fn mixture_class_solves_follow_the_stage_convention() {
        use crate::estimation::inner_optimizer::{find_ebe_with_variance, EbeVariance};
        let model = crate::parser::model_parser::parse_model_string(
            &MIX_FIT_MODEL.replace("sigma EPS ~ 0.01", "sigma EPS ~ 0.09"),
        )
        .unwrap();
        let pop = read_pop(&bimodal_csv(3, 1.0, 3.0));
        let params = &model.default_params;
        let mut sep = 0.0_f64;
        for interaction in [false, true] {
            let opts = crate::types::FitOptions {
                method: if interaction {
                    crate::types::EstimationMethod::FoceI
                } else {
                    crate::types::EstimationMethod::Foce
                },
                interaction,
                ..Default::default()
            };
            let eval = super::mixture_ofv(&model, &pop, params, &opts, None);
            for (cls, etas) in eval.etas_by_class.iter().enumerate() {
                let _guard = crate::parser::model_parser::MixtureClassGuard::enter(cls + 1);
                let cp = super::class_params(params, cls);
                for (s, got) in pop.subjects.iter().zip(etas) {
                    let solve = |v| {
                        find_ebe_with_variance(
                            &model,
                            s,
                            &cp,
                            opts.inner_maxiter,
                            opts.inner_tol,
                            None,
                            None,
                            opts.inner_restarts,
                            v,
                        )
                        .eta
                    };
                    let (frozen, cond) = (
                        solve(EbeVariance::Population),
                        solve(EbeVariance::Conditional),
                    );
                    let want = if interaction { &cond } else { &frozen };
                    for (a, b) in got.iter().zip(want.iter()) {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "interaction {interaction}, class {cls}"
                        );
                    }
                    sep = sep.max((&frozen - &cond).amax());
                }
            }
        }
        assert!(
            sep > 1e-3,
            "the conventions must separate on this fixture ({sep:e})"
        );
    }

    /// #2588 (M12): the mixture evaluation's `n_nm_polished` counts the subjects with a
    /// contributing class whose EBE the certificate moved under `inner_optimizer =
    /// nelder_mead`. The oracle re-solves each class on the calling thread exactly as
    /// [`mixture_class_solves_follow_the_stage_convention`] does (bit-identical η there),
    /// and scores the flags over the evaluation's own posterior weights. Swept over
    /// `inner_tol`, so some cells are not saturated: the test requires a subject whose
    /// winning class (`mixest`) was **not** moved while a contributing other class was, the
    /// subject that "count the winning class only" would miss (#2598 review r1, row 3).
    /// Under the default optimizer the count is 0.
    #[test]
    fn mixture_counts_the_subjects_a_contributing_class_moved() {
        use crate::estimation::inner_optimizer::{
            arm_inner_settings, find_ebe_with_variance, EbeVariance, InnerSettings,
        };
        use crate::types::InnerOptimizer;
        let model = crate::parser::model_parser::parse_model_string(
            &MIX_FIT_MODEL.replace("sigma EPS ~ 0.01", "sigma EPS ~ 0.09"),
        )
        .unwrap();
        let pop = read_pop(&bimodal_csv(3, 1.0, 3.0));
        let params = &model.default_params;
        let (mut non_winning, mut cells) = (0, Vec::new());
        for tol in [1e-5, 1e-3, 1e-2, 1e-1] {
            let eval = |mode| {
                let opts = crate::types::FitOptions {
                    method: crate::types::EstimationMethod::Foce,
                    interaction: false,
                    inner_optimizer: mode,
                    inner_tol: tol,
                    ..Default::default()
                };
                crate::api::with_fit_scope(&opts, || {
                    super::mixture_ofv(&model, &pop, params, &opts, None)
                })
                .expect("the scope builds")
            };
            let nm = eval(InnerOptimizer::NelderMead);
            let opts = crate::types::FitOptions::default();
            let k = nm.etas_by_class.len();
            let moved: Vec<Vec<bool>> = (0..k)
                .map(|cls| {
                    let _class = crate::parser::model_parser::MixtureClassGuard::enter(cls + 1);
                    let _armed = arm_inner_settings(InnerSettings {
                        mode: InnerOptimizer::NelderMead,
                        warm: false,
                    });
                    let cp = super::class_params(params, cls);
                    pop.subjects
                        .iter()
                        .map(|s| {
                            find_ebe_with_variance(
                                &model,
                                s,
                                &cp,
                                opts.inner_maxiter,
                                tol,
                                None,
                                None,
                                opts.inner_restarts,
                                EbeVariance::Population,
                            )
                            .nm_polished
                        })
                        .collect()
                })
                .collect();
            let mut expected = 0;
            for i in 0..pop.subjects.len() {
                let flags: Vec<bool> = (0..k).map(|cls| moved[cls][i]).collect();
                if any_contributing(&flags, &nm.pmix[i]) {
                    expected += 1;
                    non_winning += usize::from(!flags[nm.mixest[i]]);
                }
            }
            assert_eq!(nm.ebe_stats.n_nm_polished, expected, "inner_tol = {tol}");
            assert_eq!(eval(InnerOptimizer::Auto).ebe_stats.n_nm_polished, 0);
            cells.push((tol, expected));
        }
        eprintln!(
            "mixture under nelder_mead, (inner_tol, moved of {}): {cells:?}; {non_winning} \
             counted through a non-winning class",
            pop.subjects.len()
        );
        assert!(cells.iter().any(|&(_, e)| e > 0), "fixture: no class moved");
        assert!(
            non_winning > 0,
            "no subject was counted through a contributing non-winning class"
        );
    }

    #[test]
    fn iov_mixture_gradient_is_served() {
        // The eval-only entry point re-solves every class's κ̂ and serves an IOV mixture
        // exactly (it was `None`, routing to whole-objective FD, #985). A wrong-length or
        // mis-slotted κ gradient would fail the length and FD-parity checks below.
        let model = crate::parser::model_parser::parse_model_string(MIX_IOV_MODEL).unwrap();
        let pop = read_pop_iov(&bimodal_iov_csv(3, 1.0, 3.0));
        let opts = crate::types::FitOptions {
            inner_tol: 1e-10,
            ..crate::types::FitOptions::default()
        };
        let eval = super::mixture_ofv(&model, &pop, &model.default_params, &opts, None);
        let g = super::mixture_gradient(&model, &pop, &model.default_params, &opts, &eval)
            .expect("an IOV mixture is served, not routed to FD");
        use crate::estimation::parameterization::{pack_params, packed_len};
        assert_eq!(g.len(), packed_len(&model.default_params));
        let x = pack_params(&model.default_params);
        let fd = super::mixture_gradient_fd(&model, &pop, &x, &model.default_params, &opts);
        assert_eq!(fd.len(), g.len());
        let mut worst = 0.0f64;
        for (e, f) in g.iter().zip(&fd) {
            assert!(e.is_finite() && f.is_finite());
            worst = worst.max((e - f).abs() / (1.0 + f.abs()));
        }
        eprintln!("IOV mixture_gradient vs reconverged FD: worst {worst:.2e}");
        assert!(worst < 1e-7, "{worst:.2e}");
    }

    /// #3004 review: an IOV mixture handed no class modes used to panic
    /// [`super::mixture_gradient_salvaged`] (`.expect` on a `None` the two gates produced
    /// separately: the outer loop's `n_kappa > 0`, the gradient's `omega_iov.is_some()`). The
    /// modes are now re-solved from the same inputs, so the answer is the one with the modes
    /// supplied, bit for bit. Mutation: drop the `None if needs_class_modes` arm -> panic.
    #[test]
    fn iov_mixture_gradient_without_class_modes_resolves_them() {
        use crate::estimation::outer_optimizer::OuterFdDeclineLog;
        let model = crate::parser::model_parser::parse_model_string(MIX_IOV_MODEL).unwrap();
        let pop = read_pop_iov(&bimodal_iov_csv(3, 1.0, 3.0));
        let opts = crate::types::FitOptions {
            inner_tol: 1e-10,
            ..crate::types::FitOptions::default()
        };
        let params = &model.default_params;
        assert!(super::needs_class_modes(params), "premise: IOV");
        let (eval, modes) = super::mixture_ofv_with_modes(&model, &pop, params, &opts, None);
        let log = OuterFdDeclineLog::new(pop.subjects.len());
        let with = super::mixture_gradient_salvaged(
            &model,
            &pop,
            params,
            &opts,
            &eval,
            Some(&modes),
            &log,
        );
        let without =
            super::mixture_gradient_salvaged(&model, &pop, params, &opts, &eval, None, &log);
        assert!(with.iter().all(|g| g.is_finite()));
        assert_eq!(with.len(), without.len());
        for (a, b) in with.iter().zip(&without) {
            assert_eq!(a.to_bits(), b.to_bits(), "{a} vs {b}");
        }
    }

    /// #3009 review 3: an IOV mixture handed no class modes re-solves the evaluation, η̂ and κ̂
    /// from one solve. The caller's `eval` here comes from a loose, warm-started solve whose η̂
    /// differs from the cold one (asserted: the premise), so pairing its η̂ with a fresh κ̂ is
    /// not a joint mode. The answer must be the gradient of the cold evaluation with its own
    /// modes, bit for bit, and must not depend on the `eval` passed.
    /// Mutation: keep the caller's `eval` and re-solve only the modes -> the bits differ.
    #[test]
    fn iov_mixture_gradient_without_class_modes_pairs_one_solves_eta_and_kappa() {
        use crate::estimation::outer_optimizer::OuterFdDeclineLog;
        let model = crate::parser::model_parser::parse_model_string(MIX_IOV_MODEL).unwrap();
        let pop = read_pop_iov(&bimodal_iov_csv(3, 1.0, 3.0));
        let opts = crate::types::FitOptions {
            inner_tol: 1e-10,
            ..crate::types::FitOptions::default()
        };
        let loose = crate::types::FitOptions {
            inner_tol: 1e-2,
            inner_maxiter: 3,
            ..crate::types::FitOptions::default()
        };
        let params = &model.default_params;
        let (cold, modes) = super::mixture_ofv_with_modes(&model, &pop, params, &opts, None);
        let k = cold.etas_by_class.len();
        let warm: Vec<Vec<nalgebra::DVector<f64>>> = (0..k)
            .map(|_| {
                (0..pop.subjects.len())
                    .map(|_| nalgebra::DVector::from_element(model.n_eta, 0.7))
                    .collect()
            })
            .collect();
        let (warmed, _) = super::mixture_ofv_with_modes(&model, &pop, params, &loose, Some(&warm));
        let mut apart = 0.0f64;
        for (a, b) in cold
            .etas_by_class
            .iter()
            .flatten()
            .zip(warmed.etas_by_class.iter().flatten())
        {
            for (u, v) in a.iter().zip(b.iter()) {
                assert!(u.is_finite() && v.is_finite());
                apart = apart.max((u - v).abs()); // nan-ok: operands asserted finite above
            }
        }
        eprintln!("MEASURE warm vs cold class eta apart {apart:.3e}");
        assert!(
            apart > 1e-3,
            "premise: the warm eval's eta is off the cold one: {apart:e}"
        );
        let log = OuterFdDeclineLog::new(pop.subjects.len());
        let want = super::mixture_gradient_salvaged(
            &model,
            &pop,
            params,
            &opts,
            &cold,
            Some(&modes),
            &log,
        );
        let got =
            super::mixture_gradient_salvaged(&model, &pop, params, &opts, &warmed, None, &log);
        assert_eq!(want.len(), got.len());
        for (a, b) in want.iter().zip(&got) {
            assert!(a.is_finite() && b.is_finite());
            assert_eq!(a.to_bits(), b.to_bits(), "{a} vs {b}");
        }
    }

    /// #3009 review 4: a class mode whose κ̂ count does not match the subject's occasions (a
    /// failed or hard-rejected class solve) used to panic [`super::mixture_gradient_salvaged`]
    /// (`.expect` on the whole row). The value path scores that term at `class_nll`:
    ///
    /// - with a real objective, the term takes the reconverged difference from a κ̂ = 0 start,
    ///   which re-solves its own mode and is recorded on the log — the same gradient as the
    ///   clean modes' (measured below; differentiating at the κ̂ = 0 start itself is off by
    ///   2.6e-2);
    /// - at the sentinel (repelled), it contributes zero — bit for bit the gradient with that
    ///   term's weight zeroed — and nothing is recorded.
    ///
    /// The second half is a straddle on the first: zeroing a term whose value is real moves the
    /// gradient (asserted), so a zero fill that ignored `class_nll` would redden the first half.
    #[test]
    fn iov_mixture_gradient_survives_a_misshaped_class_mode() {
        use crate::estimation::outer_optimizer::OuterFdDeclineLog;
        let model = crate::parser::model_parser::parse_model_string(MIX_IOV_MODEL).unwrap();
        let pop = read_pop_iov(&bimodal_iov_csv(3, 1.0, 3.0));
        let opts = crate::types::FitOptions {
            inner_tol: 1e-10,
            ..crate::types::FitOptions::default()
        };
        let params = &model.default_params;
        let (eval, modes) = super::mixture_ofv_with_modes(&model, &pop, params, &opts, None);
        let n = pop.subjects.len();
        let clean_log = OuterFdDeclineLog::new(n);
        let clean = super::mixture_gradient_salvaged(
            &model,
            &pop,
            params,
            &opts,
            &eval,
            Some(&modes),
            &clean_log,
        );
        assert!(
            clean_log.declined_indices().is_empty(),
            "control: nothing declines"
        );

        // The class with the larger weight for subject 0: its term is live.
        let cls = if eval.pmix[0][0] >= eval.pmix[0][1] {
            0
        } else {
            1
        };
        assert!(eval.pmix[0][cls] > 0.1, "fixture: a live term");
        let mut bad = super::MixtureClassModes {
            etas: modes.etas.clone(),
            kappas: modes.kappas.clone(),
        };
        assert!(bad.kappas[cls][0].len() >= 2, "fixture: several occasions");
        bad.kappas[cls][0].pop();

        // Real objective: differenced at the re-solved mode.
        let log = OuterFdDeclineLog::new(n);
        let g =
            super::mixture_gradient_salvaged(&model, &pop, params, &opts, &eval, Some(&bad), &log);
        assert_eq!(
            log.declined_indices(),
            vec![0],
            "the mis-shaped term is salvaged"
        );
        let mut worst = 0.0f64;
        for (a, b) in g.iter().zip(&clean) {
            assert!(a.is_finite() && b.is_finite());
            worst = worst.max((a - b).abs() / (1.0 + b.abs()));
        }
        eprintln!("MEASURE mis-shaped class mode, salvaged vs clean: worst rel {worst:.3e}");
        assert!(worst < 1e-4, "{worst:e}");

        // Repelled: the term is zero, exactly as if its weight were.
        let (mut repelled, _) = super::mixture_ofv_with_modes(&model, &pop, params, &opts, None);
        repelled.class_nll[0][cls] = crate::stats::likelihood::SUBJECT_SENTINEL_NLL;
        let log = OuterFdDeclineLog::new(n);
        let g0 = super::mixture_gradient_salvaged(
            &model,
            &pop,
            params,
            &opts,
            &repelled,
            Some(&bad),
            &log,
        );
        assert!(
            log.declined_indices().is_empty(),
            "a repelled term takes no salvage"
        );
        let (mut unweighted, _) = super::mixture_ofv_with_modes(&model, &pop, params, &opts, None);
        unweighted.pmix[0][cls] = 0.0;
        let want = super::mixture_gradient_salvaged(
            &model,
            &pop,
            params,
            &opts,
            &unweighted,
            Some(&modes),
            &OuterFdDeclineLog::new(n),
        );
        assert_eq!(g0, want, "a repelled term contributes exactly zero");
        let live = g0
            .iter()
            .zip(&clean)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max); // nan-ok: every operand asserted finite above
        assert!(
            live > 1e3 * worst,
            "straddle: the zeroed term is live, {live:e} vs {worst:e}"
        );
    }

    #[test]
    fn iov_mixture_covariance_step_runs() {
        // #985: the covariance step (FD-of-mixture-OFV Hessian) runs for an IOV
        // mixture — the packed layout already interleaves the κ segment, and
        // cov_ofv → mixture_ofv now carries IOV. The free thetas get finite positive
        // packed-scale variance; the covariance is packed-square over the full layout.
        use crate::estimation::covariance::{compute_covariance, CovarianceStepResult};
        use crate::estimation::parameterization::{pack_params, packed_len};
        let model = crate::parser::model_parser::parse_model_string(MIX_IOV_MODEL).unwrap();
        let pop = read_pop_iov(&bimodal_iov_csv(8, 1.0, 3.0));
        let opts = crate::types::FitOptions::default();
        let params = &model.default_params;
        let x = pack_params(params);
        let eval = super::mixture_ofv(&model, &pop, params, &opts, None);
        let matrix = match compute_covariance(
            &x,
            params,
            &model,
            &pop,
            &eval.mixest_etas,
            &eval.mixest_h_mats,
            // Pass the real MIXEST-class per-occasion κ̂ (not an empty slice) so the
            // test exercises the full IOV packed layout end-to-end and stays correct
            // if the covariance method later consumes κ (Copilot review, PR #986).
            &eval.mixest_kappas,
            &opts,
        ) {
            CovarianceStepResult::Success(out) => out.matrix,
            CovarianceStepResult::Unusable(msg) => panic!("covariance unusable: {msg}"),
            CovarianceStepResult::FailedNonPd { reason, .. } => {
                panic!("covariance non-PD: {reason}")
            }
        };
        assert_eq!(
            matrix.nrows(),
            packed_len(params),
            "covariance is packed-square"
        );
        for i in 0..4 {
            let v = matrix[(i, i)];
            assert!(v.is_finite() && v > 0.0, "theta[{i}] variance = {v}");
        }
    }

    #[test]
    fn mixture_fit_runs_and_returns_finite_ofv() {
        // Tier-2: fit() returns after a few outer iterations with a finite OFV.
        let model = crate::parser::model_parser::parse_model_string(MIX_FIT_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(3, 1.0, 3.0));
        let mut opts = crate::types::FitOptions::default();
        opts.outer_maxiter = 3;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("mixture fit should return Ok");
        assert!(res.ofv.is_finite(), "OFV = {}", res.ofv);

        // Phase 5: per-subject posteriors are threaded onto every SubjectResult.
        for sr in &res.subjects {
            let pmix = sr.pmix.as_ref().expect("PMIX populated for a mixture fit");
            assert_eq!(pmix.len(), 2, "K = 2 posterior weights per subject");
            let sum: f64 = pmix.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "PMIX sums to 1, got {sum}");
            assert!(pmix.iter().all(|&p| (0.0..=1.0).contains(&p)));
            // MIXEST is the 1-based argmax-posterior class.
            let mixest = sr.mixest.expect("MIXEST populated for a mixture fit");
            assert!(mixest == 1 || mixest == 2, "MIXEST in 1..=K, got {mixest}");
            let argmax = if pmix[0] >= pmix[1] { 1 } else { 2 };
            assert_eq!(mixest, argmax, "MIXEST = argmax_k PMIX_ik (1-based)");
        }
    }

    #[test]
    fn mixture_eval_only_emits_posteriors() {
        // Regression (#980): an eval-only run (outer_maxiter = 0, NONMEM
        // MAXEVAL=0 semantics) must still emit PMIX/MIXEST per subject, exactly
        // like a converged mixture fit. The eval-only path previously hard-coded
        // `mixture_posteriors: None`, so the sdtab columns silently vanished.
        let model = crate::parser::model_parser::parse_model_string(MIX_FIT_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(3, 1.0, 3.0));
        let opts = crate::types::FitOptions {
            outer_maxiter: 0,
            ..crate::types::FitOptions::default()
        };
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("eval-only mixture fit should return Ok");
        assert!(res.ofv.is_finite(), "OFV = {}", res.ofv);
        for sr in &res.subjects {
            let pmix = sr
                .pmix
                .as_ref()
                .expect("PMIX populated for an eval-only mixture fit");
            assert_eq!(pmix.len(), 2, "K = 2 posterior weights per subject");
            assert!((pmix.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(
                sr.mixest.is_some(),
                "MIXEST populated for eval-only mixture"
            );
        }
    }

    /// Constant-mixing MIXNUM model with FIXED Ω/Σ, mirroring the NONMEM anchor
    /// (`tests/nonmem/mixture_iv.ctl`): TVCL1/TVCL2 branch, `p(1)` direct-form
    /// mixing, no per-class overrides. The four free thetas (TVCL1, TVCL2, TVV,
    /// P1) are the covariance step's targets.
    const MIX_COV_MODEL: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta P1(0.5, 0.001, 0.999)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  p(1) = P1

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    #[test]
    fn mixture_covariance_step_runs_and_sizes() {
        // #983 Phase 6: the covariance step builds its FD Hessian on the K-fold
        // mixture OFV (`compute_covariance` branches on `template.mixture`), so a
        // mixture fit now yields a covariance matrix + SEs — it previously skipped
        // the step entirely. Evaluated near the data-generating optimum so the
        // 4-theta Hessian is positive-definite; FIXED Ω/Σ rows stay zero.
        use crate::estimation::covariance::{compute_covariance, CovarianceStepResult};
        use crate::estimation::parameterization::pack_params;
        let model = crate::parser::model_parser::parse_model_string(MIX_COV_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(8, 1.0, 3.0));
        let opts = crate::types::FitOptions::default();
        let params = &model.default_params;
        let x = pack_params(params);
        // MIXEST-class EBEs/H fill the warm-start slots the mixture branch ignores.
        let eval = super::mixture_ofv(&model, &pop, params, &opts, None);
        let matrix = match compute_covariance(
            &x,
            params,
            &model,
            &pop,
            &eval.mixest_etas,
            &eval.mixest_h_mats,
            &[],
            &opts,
        ) {
            CovarianceStepResult::Success(out) => out.matrix,
            CovarianceStepResult::Unusable(msg) => panic!("covariance unusable: {msg}"),
            CovarianceStepResult::FailedNonPd { reason, .. } => {
                panic!("covariance non-PD: {reason}")
            }
        };
        assert_eq!(matrix.nrows(), x.len(), "covariance is packed-square");
        // Free thetas (idx 0..4) get finite positive packed-scale variance.
        for i in 0..4 {
            let v = matrix[(i, i)];
            assert!(v.is_finite() && v > 0.0, "theta[{i}] variance = {v}");
        }
        // FIXED Ω (idx 4) and Σ (idx 5) contribute no information → zeroed rows.
        assert_eq!(matrix[(4, 4)], 0.0, "FIXED omega row zeroed");
        assert_eq!(matrix[(5, 5)], 0.0, "FIXED sigma row zeroed");
    }

    #[test]
    fn mixture_covariance_uses_cov_inner_tol_not_fit_tol() {
        // #984 regression: the mixture covariance step must reconverge its per-class
        // EBEs at `cov_inner_tol`, not the fit's `inner_tol`. Otherwise a loose fit +
        // an explicit tight `cov_inner_tol` (set for trustworthy SEs) would silently
        // build the Hessian on the loose, contaminated EBEs. We build the covariance
        // twice: a reference at a tight fit tol, and a probe at an absurdly loose fit
        // tol but the same explicit tight `cov_inner_tol`. With the fix they must
        // match; before it, the probe would use the loose tol and diverge.
        use crate::estimation::covariance::{compute_covariance, CovarianceStepResult};
        use crate::estimation::parameterization::pack_params;
        let model = crate::parser::model_parser::parse_model_string(MIX_COV_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(8, 1.0, 3.0));
        let params = &model.default_params;
        let x = pack_params(params);

        let run = |inner_tol: f64, cov_inner_tol: Option<f64>| {
            let opts = crate::types::FitOptions {
                inner_tol,
                cov_inner_tol,
                ..crate::types::FitOptions::default()
            };
            let eval = super::mixture_ofv(&model, &pop, params, &opts, None);
            match compute_covariance(
                &x,
                params,
                &model,
                &pop,
                &eval.mixest_etas,
                &eval.mixest_h_mats,
                &[],
                &opts,
            ) {
                CovarianceStepResult::Success(out) => out.matrix,
                CovarianceStepResult::Unusable(msg) => panic!("covariance unusable: {msg}"),
                CovarianceStepResult::FailedNonPd { reason, .. } => {
                    panic!("covariance non-PD: {reason}")
                }
            }
        };

        let reference = run(1e-8, None); // cov tol defaults to the tight fit tol
        let probe = run(1.0, Some(1e-8)); // loose fit, tight explicit cov tol
        for i in 0..4 {
            let (r, p) = (reference[(i, i)], probe[(i, i)]);
            let denom = r.abs().max(p.abs()).max(1e-9);
            assert!(
                (r - p).abs() / denom < 1e-6,
                "theta[{i}] variance depends on the fit inner_tol: ref {r} vs probe {p}"
            );
        }
    }

    /// Build a constant-mixing MIXNUM model with the given mixing line and the
    /// `MIXL`/`P1` mixing-coefficient parameter declaration, so a `p(k)` model and
    /// its softmax-equivalent `logit(k)` model share every other line.
    fn mix_form_model(mix_param: &str, mixing_line: &str) -> String {
        format!(
            r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  {mix_param}
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  {mixing_line}

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
        )
    }

    #[test]
    fn mixture_pform_matches_equivalent_logit() {
        // #983 item 4: the `p(k)` mixing form is evaluated (probabilities used
        // directly, then renormalised), not silently dropped. A constant
        // `p(1) = 0.3` must give the identical objective to the softmax-equivalent
        // `logit(1) = ln(0.3/0.7)` (reference class 2 has logit 0), since both
        // resolve to p = [0.3, 0.7] and every other line is shared.
        let pform = mix_form_model("theta P1(0.3, 0.001, 0.999)", "p(1) = P1");
        let logit_c = (0.3f64 / 0.7).ln(); // ≈ -0.8473
        let lform = mix_form_model(
            &format!("theta MIXL({logit_c}, -10.0, 10.0)"),
            "logit(1) = MIXL",
        );
        let mp = crate::parser::model_parser::parse_model_string(&pform).unwrap();
        let ml = crate::parser::model_parser::parse_model_string(&lform).unwrap();
        let pop = read_pop(&bimodal_csv(6, 1.0, 3.0));
        let opts = crate::types::FitOptions::default();

        let op = super::mixture_ofv(&mp, &pop, &mp.default_params, &opts, None);
        let ol = super::mixture_ofv(&ml, &pop, &ml.default_params, &opts, None);
        assert!(
            (op.ofv - ol.ofv).abs() < 1e-6,
            "p-form OFV {} vs equivalent logit {}",
            op.ofv,
            ol.ofv
        );
        // Posterior weights match subject-by-subject too.
        for (pp, pl) in op.pmix.iter().zip(&ol.pmix) {
            for (a, b) in pp.iter().zip(pl) {
                assert!((a - b).abs() < 1e-9, "PMIX {a} vs {b}");
            }
        }
    }

    #[test]
    fn mixture_gradient_serves_the_pform() {
        // #983 item 4: the `p(k)` form used to have no analytic mixing-probability
        // gradient, so `mixture_gradient` returned None and the caller differenced the
        // whole mixture objective. The mixing map is now differentiated exactly for both
        // forms (`mixing_logp_grad_exact`), so the p form takes the analytic gradient —
        // pinned against that central difference of the objective, and on the mixing θ.
        use crate::estimation::parameterization::pack_params;
        let m = mix_form_model("theta P1(0.4, 0.001, 0.999)", "p(1) = P1");
        let model = crate::parser::model_parser::parse_model_string(&m).unwrap();
        let pop = read_pop(&bimodal_csv(4, 1.0, 3.0));
        let opts = crate::types::FitOptions::default();
        let params = &model.default_params;
        let eval = super::mixture_ofv(&model, &pop, params, &opts, None);
        let ana = super::mixture_gradient(&model, &pop, params, &opts, &eval)
            .expect("the p(k) form is in analytic scope");
        let fd = super::mixture_gradient_fd(&model, &pop, &pack_params(params), params, &opts);
        let i_p1 = model
            .theta_names
            .iter()
            .position(|n| n == "P1")
            .expect("P1");
        assert!(fd[i_p1].abs() > 1e-2, "∂OFV/∂P1 should be clearly nonzero");
        // A FIX coordinate's quotient collapses to 0 while the analytic gradient reports
        // the true derivative; the optimizer pins those coordinates, so only free ones are
        // compared.
        let fixed = crate::estimation::parameterization::packed_fixed_mask(params);
        let mut worst = 0.0f64;
        for (i, (a, f)) in ana.iter().zip(&fd).enumerate() {
            if fixed[i] {
                continue;
            }
            let rel = (a - f).abs() / f.abs().max(a.abs()).max(1e-3);
            worst = worst.max(rel);
            // Measured 3.0e-5: the objective quotient's truncation. 30× headroom.
            assert!(
                rel < 1e-3,
                "coord {i}: analytic {a:.6} vs FD {f:.6} (rel {rel:.2e})"
            );
        }
        eprintln!("p-form mixture gradient vs the objective's central difference: {worst:.2e}");
    }

    // Variance-mixture (classes differ only by their Ω/Σ overrides, no `MIXNUM`
    // branch) so the per-subject analytic (adjoint) gradient is in scope — the
    // parity test then exercises the real analytic gradient (softmax mixing
    // slots, base + per-class-override Ω/σ slots). A `MIXNUM`-conditional model
    // is out of analytic scope and takes the FD route (see `mixture_gradient`).
    const MIX_PARITY_MODEL: &str = r"
[parameters]
  theta TVCL(1.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.2, -10.0, 10.0)
  theta BWT(0.05, -5.0, 5.0)
  omega ETA_CL ~ 0.06
  sigma EPS ~ 0.02

[mixture]
  nsub = 2
  logit(1) = MIXL + BWT*(WT - 75)
  omega(2) ETA_CL ~ 0.15
  sigma(2) EPS ~ 0.03

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    /// The exact gradient of `mixture_ofv` at tight inner tolerance, with every class's modes.
    fn exact_mixture_grad(
        model: &crate::types::CompiledModel,
        pop: &crate::types::Population,
        params: &crate::types::ModelParameters,
        opts: &crate::types::FitOptions,
    ) -> Vec<f64> {
        let (eval, modes) = super::mixture_ofv_with_modes(model, pop, params, opts, None);
        super::mixture_gradient_with_modes(model, pop, params, opts, &eval, Some(&modes))
            .expect("exact mixture gradient in scope")
    }

    /// The mixture fixtures the exact derivatives are pinned on: logit mixing on a
    /// covariate with per-class Ω/σ overrides, the `p` form, and an IOV mixture.
    fn mixture_cases() -> Vec<(
        &'static str,
        crate::types::CompiledModel,
        crate::types::Population,
    )> {
        let parse = |src: &str| crate::parser::model_parser::parse_model_string(src).unwrap();
        vec![
            (
                "logit+overrides",
                parse(MIX_PARITY_MODEL),
                read_pop(&bimodal_csv(4, 1.0, 3.0)),
            ),
            (
                "p-form",
                parse(MIX_COV_MODEL),
                read_pop(&bimodal_csv(4, 1.0, 3.0)),
            ),
            (
                "iov",
                parse(MIX_IOV_MODEL),
                read_pop_iov(&bimodal_iov_csv(4, 1.0, 3.0)),
            ),
        ]
    }

    /// An IOV mixture's outer gradient is exact (each class's subject gradient at its stacked
    /// `[η̂, κ̂]`), against the central difference of the reconverged mixture objective
    /// (`mixture_gradient_fd`); the eval-only entry point re-solves the per-class κ̂ and agrees.
    #[test]
    fn an_iov_mixture_gradient_is_exact() {
        use crate::estimation::parameterization::pack_params;
        let model = crate::parser::model_parser::parse_model_string(MIX_IOV_MODEL).unwrap();
        let pop = read_pop_iov(&bimodal_iov_csv(4, 1.0, 3.0));
        let params = &model.default_params;
        let mut worst = 0.0f64;
        for interaction in [true, false] {
            let opts = crate::types::FitOptions {
                interaction,
                inner_tol: 1e-10,
                ..crate::types::FitOptions::default()
            };
            let exact = exact_mixture_grad(&model, &pop, params, &opts);
            let x = pack_params(params);
            let fd = super::mixture_gradient_fd(&model, &pop, &x, params, &opts);
            assert_eq!(exact.len(), fd.len());
            for (k, (e, f)) in exact.iter().zip(&fd).enumerate() {
                assert!(e.is_finite() && f.is_finite(), "[{k}]");
                worst = worst.max((e - f).abs() / (1.0 + f.abs()));
            }
            let eval = super::mixture_ofv(&model, &pop, params, &opts, None);
            let direct = super::mixture_gradient(&model, &pop, params, &opts, &eval)
                .expect("the eval-only entry re-solves the modes and serves IOV");
            for (d, e) in direct.iter().zip(&exact) {
                assert!((d - e).abs() <= 1e-6 * (1.0 + e.abs()), "{d} vs {e}");
            }
        }
        eprintln!("IOV mixture gradient vs reconverged FD: worst {worst:.2e}");
        assert!(worst < 1e-4, "{worst:.2e}");
    }

    /// The four class objectives a mixture is fit under: FOCE, FOCEI, Laplace, FOCEI with
    /// `n_agq = 3` (#3025), at a tight inner tolerance.
    fn objective_cases() -> Vec<(&'static str, crate::types::FitOptions)> {
        use crate::types::{EstimationMethod, FitOptions};
        let base = FitOptions {
            inner_tol: 1e-10,
            ..FitOptions::default()
        };
        vec![
            (
                "foce",
                FitOptions {
                    method: EstimationMethod::Foce,
                    interaction: false,
                    ..base.clone()
                },
            ),
            (
                "focei",
                FitOptions {
                    method: EstimationMethod::FoceI,
                    interaction: true,
                    ..base.clone()
                },
            ),
            (
                "laplace",
                FitOptions {
                    method: EstimationMethod::Laplace,
                    interaction: true,
                    ..base.clone()
                },
            ),
            (
                "agq3",
                FitOptions {
                    method: EstimationMethod::FoceI,
                    interaction: true,
                    n_agq: 3,
                    ..base
                },
            ),
        ]
    }

    /// The exact mixture R-matrix (`mixture_cov_hessian`) is the derivative of the exact
    /// mixture gradient: against its Richardson central difference with every class's mode
    /// re-solved at each point, on every mixture fixture and every class objective (FOCE,
    /// FOCEI, Laplace, AGQ). A dropped outer-product term, a mis-routed override block or a
    /// missing θ-packing second derivative of `ln p` misses by the size of that term.
    #[test]
    fn the_mixture_cov_hessian_is_the_derivative_of_the_exact_gradient() {
        use crate::estimation::parameterization::{pack_params, unpack_params};
        let mut worst = 0.0f64;
        for (tag, model, pop) in mixture_cases() {
            let params = &model.default_params;
            let x = pack_params(params);
            let n = x.len();
            for (kind, opts) in objective_cases() {
                let h = super::mixture_cov_hessian(&model, &pop, params, &x, &opts)
                    .unwrap_or_else(|| panic!("{tag} {kind}: in scope"));
                assert_eq!((h.nrows(), h.ncols()), (n, n));
                let mut fd = nalgebra::DMatrix::<f64>::zeros(n, n);
                for i in 0..n {
                    let grad_at = |d: f64| {
                        let mut xp = x.clone();
                        xp[i] += d;
                        exact_mixture_grad(&model, &pop, &unpack_params(&xp, params), &opts)
                    };
                    let step = 1e-3 * (1.0 + x[i].abs());
                    let central = |st: f64| {
                        let (gp, gm) = (grad_at(st), grad_at(-st));
                        (0..n)
                            .map(|j| (gp[j] - gm[j]) / (2.0 * st))
                            .collect::<Vec<_>>()
                    };
                    let (d1, d2) = (central(step), central(0.5 * step));
                    for j in 0..n {
                        fd[(j, i)] = (4.0 * d2[j] - d1[j]) / 3.0;
                    }
                }
                let mut gap = 0.0f64;
                for (a, b) in h.iter().zip(fd.iter()) {
                    assert!(a.is_finite() && b.is_finite(), "{tag} {kind}");
                    gap = gap.max((a - b).abs() / (1.0 + b.abs()));
                }
                assert!(
                    h.iter().map(|v| v.abs()).fold(0.0, f64::max) > 1.0,
                    "{tag} {kind}: a vanishing Hessian tests nothing"
                );
                eprintln!("{tag} {kind}: mixture R vs FD of exact gradient {gap:.2e}");
                worst = worst.max(gap);
            }
        }
        assert!(worst < 1e-5, "{worst:.2e}");
    }

    /// Both classes carry weight: some subject sits strictly between the classes, for each.
    fn assert_non_degenerate_classes(eval: &super::MixtureEval, tag: &str) {
        for cls in 0..eval.pmix[0].len() {
            assert!(
                eval.pmix.iter().any(|p| p[cls] > 0.02 && p[cls] < 0.98),
                "{tag}: class {cls} never carries a mixed posterior"
            );
        }
    }

    /// Laplace / AGQ mixture outer gradient (#3025) against the central difference of the
    /// reconverged mixture objective, on the logit-on-covariate fixture (both classes carry
    /// weight; mixing depends on WT), the p-form and the IOV mixture. Under the quadrature the
    /// objective is `agq_subject_term` per class, so a gradient left on the FOCE/FOCEI
    /// objective (the gradient-objective mutation) misses by the size of the Laplace
    /// correction.
    #[test]
    fn quadrature_mixture_gradient_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        let mut worst = 0.0f64;
        for (tag, model, pop) in mixture_cases() {
            let params = &model.default_params;
            let x = pack_params(params);
            for (kind, opts) in objective_cases() {
                if kind == "foce" || kind == "focei" {
                    continue;
                }
                let (eval, modes) =
                    super::mixture_ofv_with_modes(&model, &pop, params, &opts, None);
                if tag == "logit+overrides" {
                    assert_non_degenerate_classes(&eval, tag);
                }
                let exact = super::mixture_gradient_with_modes(
                    &model,
                    &pop,
                    params,
                    &opts,
                    &eval,
                    Some(&modes),
                )
                .unwrap_or_else(|| panic!("{tag} {kind}: exact gradient in scope"));
                let fd = super::mixture_gradient_fd(&model, &pop, &x, params, &opts);
                // A FIX coordinate's FD is pinned to 0 (its box is a point); the exact gradient
                // is differentiated there too and zeroed by the outer loop.
                let fixed = crate::estimation::parameterization::packed_fixed_mask(params);
                let mut gap = 0.0f64;
                for (k, (e, f)) in exact
                    .iter()
                    .zip(&fd)
                    .enumerate()
                    .filter(|(k, _)| !fixed[*k])
                {
                    assert!(e.is_finite() && f.is_finite(), "{tag} {kind} [{k}]");
                    gap = gap.max((e - f).abs() / (1.0 + f.abs()));
                }
                eprintln!("{tag} {kind}: mixture gradient vs reconverged FD {gap:.2e}");
                worst = worst.max(gap); // nan-ok: gap is a fold of asserted-finite operands
            }
        }
        assert!(worst < 1e-5, "{worst:.2e}");
    }

    /// The quadrature objective is the single population's: a mixture of two **identical**
    /// classes at `p = ½` (so `p₁ + p₂ = 1` exactly) is the single-population Laplace / AGQ
    /// fit of the same model — in OFV and in gradient (`∂/∂TVCL` splits over the two classes,
    /// the mixing coordinate's gradient is 0). Measured gaps are printed; the modes come from
    /// different inner policies, so they agree to the inner tolerance, not to the bit.
    #[test]
    fn identical_classes_equal_the_single_population_quadrature() {
        use crate::estimation::parameterization::pack_params;
        let mix_src = MIX_COV_MODEL
            .replace("theta TVCL2(3.0", "theta TVCL2(1.0")
            .replace("ETA_CL ~ 0.09 FIX", "ETA_CL ~ 0.09")
            .replace("sigma EPS ~ 0.04 FIX", "sigma EPS ~ 0.04");
        let single_src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let parse = |src: &str| crate::parser::model_parser::parse_model_string(src).unwrap();
        let (mix, single) = (parse(&mix_src), parse(single_src));
        let pop = read_pop(&bimodal_csv(4, 1.0, 3.0));
        let mut worst = (0.0f64, 0.0f64);
        for (kind, opts) in objective_cases() {
            if kind == "foce" || kind == "focei" {
                continue;
            }
            let n_nodes = opts.agq_nodes().unwrap();
            let anchor = opts.hessian_anchor();
            // Mixture.
            let mp = &mix.default_params;
            let (eval, modes) = super::mixture_ofv_with_modes(&mix, &pop, mp, &opts, None);
            let g_mix =
                super::mixture_gradient_with_modes(&mix, &pop, mp, &opts, &eval, Some(&modes))
                    .expect("mixture gradient");
            // Single population at the same parameters.
            let sp = &single.default_params;
            let etas: Vec<_> = pop
                .subjects
                .iter()
                .map(|s| {
                    crate::estimation::inner_optimizer::find_ebe(
                        &single,
                        s,
                        sp,
                        200,
                        opts.inner_tol,
                        None,
                        None,
                        0,
                    )
                    .eta
                })
                .collect();
            let kappas = vec![Vec::new(); pop.subjects.len()];
            let ofv_single = 2.0
                * crate::estimation::agq::agq_population_nll(
                    &single, &pop, sp, &etas, &kappas, n_nodes, anchor,
                );
            let g_single = crate::estimation::agq::agq_population_gradient(
                &single,
                &pop,
                sp,
                sp,
                &pack_params(sp),
                &etas,
                &kappas,
                n_nodes,
                anchor,
            )
            .expect("single-population gradient");
            let d_ofv = (eval.ofv - ofv_single).abs() / ofv_single.abs();
            // Mixture packed layout: [TVCL1, TVCL2, TVV, P1, Ω, σ]; single: [TVCL, TVV, Ω, σ].
            let d_g = [
                (g_mix[0] + g_mix[1] - g_single[0]).abs(),
                (g_mix[2] - g_single[1]).abs(),
                g_mix[3].abs(),
                (g_mix[4] - g_single[2]).abs(),
                (g_mix[5] - g_single[3]).abs(),
            ]
            .into_iter()
            .fold(0.0f64, f64::max); // nan-ok: every operand is asserted finite below
            assert!(g_mix.iter().chain(&g_single).all(|v| v.is_finite()) && d_g.is_finite());
            eprintln!("{kind}: identical classes vs single population: OFV {d_ofv:.2e}, gradient {d_g:.2e}");
            worst = (worst.0.max(d_ofv), worst.1.max(d_g));
        }
        assert!(worst.0 < 1e-9 && worst.1 < 1e-6, "{worst:?}");
    }

    /// `mixture_candidate_hessian` at a point that is not the optimum, warm-started from the
    /// modes at the start, on a proper subset of free axes: its gradient and Hessian are the
    /// full cold gradient / R-matrix at that point restricted to the axes (the Hessian is the
    /// adjoint's block, the full matrix's sub-block), and the gradient matches the reconverged
    /// FD of the mixture objective. On every parsed fixture and every class objective it never
    /// declines as out of scope (#3025).
    #[test]
    fn candidate_hessian_on_axes_matches_the_full_derivatives() {
        use crate::estimation::parameterization::{pack_params, packed_fixed_mask, unpack_params};
        let mut worst = (0.0f64, 0.0f64);
        for (tag, model, pop) in mixture_cases() {
            let params0 = &model.default_params;
            let x0 = pack_params(params0);
            let fixed = packed_fixed_mask(params0);
            let free: Vec<usize> = (0..x0.len()).filter(|&i| !fixed[i]).collect();
            let axes: Vec<usize> = free.iter().copied().step_by(2).collect();
            assert!(
                axes.len() >= 2 && axes.len() < free.len(),
                "{tag}: a proper subset"
            );
            let x1: Vec<f64> = x0
                .iter()
                .enumerate()
                .map(|(i, v)| v + 0.04 * (1.0 + v.abs()) * if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            for (kind, opts) in objective_cases() {
                let (_, warm) = super::mixture_ofv_with_modes(&model, &pop, params0, &opts, None);
                let (g, h) = super::mixture_candidate_hessian(
                    &model, &pop, params0, &x1, &warm, &opts, &axes,
                )
                .unwrap_or_else(|e| panic!("{tag} {kind}: {e:?}"));
                let p1 = unpack_params(&x1, params0);
                let full_g = exact_mixture_grad(&model, &pop, &p1, &opts);
                let full_h = super::mixture_cov_hessian(&model, &pop, params0, &x1, &opts)
                    .unwrap_or_else(|| panic!("{tag} {kind}: full R"));
                let fd = super::mixture_gradient_fd(&model, &pop, &x1, params0, &opts);
                let (mut dh, mut dg, mut dfd) = (0.0f64, 0.0f64, 0.0f64);
                for (a, &ia) in axes.iter().enumerate() {
                    assert!(
                        g[a].is_finite() && fd[ia].is_finite() && full_g[ia].is_finite(),
                        "{tag} {kind}"
                    );
                    dg = dg.max((g[a] - full_g[ia]).abs() / (1.0 + full_g[ia].abs()));
                    dfd = dfd.max((g[a] - fd[ia]).abs() / (1.0 + fd[ia].abs()));
                    for (b, &ib) in axes.iter().enumerate() {
                        assert!(h[(a, b)].is_finite() && full_h[(ia, ib)].is_finite());
                        dh = dh.max(
                            (h[(a, b)] - full_h[(ia, ib)]).abs() / (1.0 + full_h[(ia, ib)].abs()),
                        );
                    }
                }
                eprintln!("{tag} {kind}: axes vs full: grad {dg:.2e}, hess {dh:.2e}; grad vs FD {dfd:.2e}");
                assert!(dg < 1e-6 && dh < 1e-6, "{tag} {kind}: {dg:.2e} {dh:.2e}");
                assert!(dfd < 1e-4, "{tag} {kind}: gradient vs FD {dfd:.2e}");
                worst = (worst.0.max(dg.max(dh)), worst.1.max(dfd));
            }
        }
        eprintln!(
            "candidate hessian: worst full gap {:.2e}, worst FD gap {:.2e}",
            worst.0, worst.1
        );
    }

    /// Each decline kind of `mixture_candidate_hessian`, and only through its own cause.
    /// `OutOfScope`: the user's opt-outs (`analytic_cov_hessian = false`, `gradient = fd`) and a
    /// warm start that does not match the population. `NotResolved`: a warm class mode that
    /// cannot re-converge — far from the mode, with a one-iteration inner budget — which the
    /// control (the same call from the true modes) resolves.
    #[test]
    fn candidate_hessian_declines_by_kind() {
        use crate::estimation::covariance::ExactDecline;
        use crate::estimation::parameterization::{pack_params, packed_fixed_mask};
        use crate::types::GradientMethod;
        let model = crate::parser::model_parser::parse_model_string(MIX_PARITY_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(4, 1.0, 3.0));
        let params = &model.default_params;
        let x = pack_params(params);
        let fixed = packed_fixed_mask(params);
        let axes: Vec<usize> = (0..x.len()).filter(|&i| !fixed[i]).collect();
        for (kind, opts) in objective_cases() {
            let (_, warm) = super::mixture_ofv_with_modes(&model, &pop, params, &opts, None);
            let call = |model: &crate::types::CompiledModel,
                        warm: &super::MixtureClassModes,
                        o: &crate::types::FitOptions| {
                super::mixture_candidate_hessian(model, &pop, params, &x, warm, o, &axes)
            };
            assert!(
                call(&model, &warm, &opts).is_ok(),
                "{kind}: control resolves"
            );

            let off = crate::types::FitOptions {
                analytic_cov_hessian: false,
                ..opts.clone()
            };
            assert_eq!(
                call(&model, &warm, &off).err(),
                Some(ExactDecline::OutOfScope),
                "{kind}"
            );
            let mut fd_model =
                crate::parser::model_parser::parse_model_string(MIX_PARITY_MODEL).unwrap();
            fd_model.gradient_method = GradientMethod::Fd;
            assert_eq!(
                call(&fd_model, &warm, &opts).err(),
                Some(ExactDecline::OutOfScope),
                "{kind}: gradient = fd"
            );
            let mut short = super::MixtureClassModes {
                etas: warm.etas.clone(),
                kappas: warm.kappas.clone(),
            };
            short.etas[0].pop();
            assert_eq!(
                call(&model, &short, &opts).err(),
                Some(ExactDecline::OutOfScope),
                "{kind}: warm start of the wrong shape"
            );

            let lost = super::MixtureClassModes {
                etas: warm
                    .etas
                    .iter()
                    .map(|c| {
                        c.iter()
                            .map(|e| nalgebra::DVector::from_element(e.len(), 9.0))
                            .collect()
                    })
                    .collect(),
                kappas: warm.kappas.clone(),
            };
            let starved = crate::types::FitOptions {
                inner_maxiter: 1,
                ..opts.clone()
            };
            assert_eq!(
                call(&model, &lost, &starved).err(),
                Some(ExactDecline::NotResolved),
                "{kind}: class modes that do not re-converge"
            );
        }
    }

    /// Routing (#3025): a Laplace mixture fit takes the exact gradient and the exact R-matrix —
    /// no finite-difference fallback warning, no whole-population stencil, the covariance step
    /// predicted analytic — and `E_MIXTURE_METHOD` no longer refuses it.
    #[test]
    fn a_laplace_mixture_fit_takes_the_exact_routes() {
        use crate::estimation::covariance::{predicted_r_route, PredictedRRoute};
        let model = crate::parser::model_parser::parse_model_string(MIX_PARITY_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(5, 1.0, 3.0));
        for (kind, n_agq) in [("laplace", 1usize), ("agq3", 3)] {
            let opts = crate::types::FitOptions {
                method: if n_agq == 1 {
                    crate::types::EstimationMethod::Laplace
                } else {
                    crate::types::EstimationMethod::FoceI
                },
                interaction: true,
                n_agq,
                outer_maxiter: 3,
                run_covariance_step: true,
                ..crate::types::FitOptions::default()
            };
            assert_eq!(
                predicted_r_route(&model, &opts),
                PredictedRRoute::Analytic,
                "{kind}"
            );
            let fit = crate::api::fit(&model, &pop, &model.default_params, &opts)
                .unwrap_or_else(|e| panic!("{kind}: {e}"));
            assert!(fit.ofv.is_finite(), "{kind}");
            let fd_warning = fit
                .warnings
                .iter()
                .find(|w| w.contains("finite-difference") || w.contains("finite difference"));
            assert!(fd_warning.is_none(), "{kind}: {fd_warning:?}");
        }
    }
    /// #2187: `mixture_gradient_fd` steps a small identity-packed θ by `1e-5·|θ₀|`, not a
    /// flat `1e-5` in its own units. `MIX_PARITY_MODEL` with `V = TVV_S / k`, `TVV_S`
    /// identity-packed at `10·k`: `k·∂OFV/∂TVV_S` at `k = 1e-4` is `∂OFV/∂TVV_S` at `k = 1`,
    /// whose floor is 1 and so keeps the flat step. Measured (macOS arm64): 4.1e-9 relative;
    /// the flat step, 1 % of `TVV_S`, gives 2.8e-3. The bound sits between the two.
    #[test]
    fn mixture_gradient_fd_steps_a_small_identity_theta_relatively() {
        use crate::estimation::parameterization::pack_params;
        let pop = read_pop(&bimodal_csv(4, 1.0, 3.0));
        let opts = crate::types::FitOptions::default();
        let scaled = |k: f64| -> f64 {
            let tvv = 10.0 * k;
            let src = MIX_PARITY_MODEL
                .replace(
                    "theta TVV(10.0, 0.1, 1000.0)",
                    &format!(
                        "theta TVV_S({tvv:e}, {:e}, {:e})",
                        -100.0 * tvv,
                        100.0 * tvv
                    ),
                )
                .replace("V  = TVV\n", &format!("V  = TVV_S / {k:e}\n"));
            assert!(src.contains("TVV_S /"), "fixture: V is rescaled");
            let model = crate::parser::model_parser::parse_model_string(&src).unwrap();
            let x = pack_params(&model.default_params);
            assert_eq!(x[1], tvv, "fixture: TVV_S is identity-packed");
            let fd = super::mixture_gradient_fd(&model, &pop, &x, &model.default_params, &opts);
            k * fd[1]
        };
        let reference = scaled(1.0);
        assert!(
            reference.is_finite() && reference.abs() > 1e-3,
            "fixture: live gradient ({reference})"
        );
        // `1e-14` puts the probe span (`2e-5·|θ₀|` = 2e-18) under the absolute `1e-16`
        // cutoff the stencil once skipped on, which returned an exact 0 (#2187 review).
        for k in [1e-4, 1e-14] {
            let small = scaled(k);
            assert!(small.is_finite(), "k = {k:e}");
            let rel = (small - reference).abs() / reference.abs();
            eprintln!("MEASURE #2187 mixture k={k:e}: {reference:e} vs {small:e}, rel {rel:e}");
            assert!(rel < 1e-6, "k = {k:e}: rel {rel:e}");
        }
    }

    /// #2187 review: `mixture_gradient_fd` clamps each probe to the packed box. `SH` is
    /// identity-packed on `[-1, 0.9]` and `V` carries `(c - SH)^2.5`, `c = 0.900001`, which is
    /// undefined `1e-6` past the upper bound and flat up to it, so the slope in `V` is the
    /// `0.3·SH` term. At `x` on that bound the unclamped `+h` probe (`h = 1e-5·|θ₀| = 5e-6`)
    /// lands past `c`; the clamped one differences backward inside the box, over the span
    /// `h`, and must match a backward difference at a tenth of the step. Measured (macOS
    /// arm64): 1.3e-6 relative. Unclamped, the `+h` OFV is the `1.6e21` guard and the FD
    /// `1.6e26` — finite, so it is the bound, not `is_finite`, that catches it; clamped but
    /// still divided by `2h`, it is half the slope, 0.5 relative.
    #[test]
    fn mixture_gradient_fd_clamps_a_probe_on_a_bound_to_the_box() {
        use crate::estimation::parameterization::{compute_bounds, pack_params, unpack_params};
        let src = MIX_PARITY_MODEL
            .replace(
                "  theta BWT(0.05, -5.0, 5.0)\n",
                "  theta BWT(0.05, -5.0, 5.0)\n  theta SH(0.5, -1.0, 0.9)\n",
            )
            .replace(
                "V  = TVV\n",
                "V  = TVV * (1 + 0.3 * SH + (0.900001 - SH) * (0.900001 - SH) * sqrt(0.900001 - SH))\n",
            );
        assert!(src.contains("theta SH(") && src.contains("sqrt(0.900001 - SH)"));
        let model = crate::parser::model_parser::parse_model_string(&src).unwrap();
        let params = &model.default_params;
        let k = model.theta_names.iter().position(|n| n == "SH").unwrap();
        let bounds = compute_bounds(params);
        let mut x = pack_params(params);
        assert_eq!(x[k], 0.5, "fixture: SH is identity-packed");
        x[k] = bounds.upper[k];
        assert_eq!(x[k], 0.9, "fixture: SH sits on its declared upper bound");

        let pop = read_pop(&bimodal_csv(4, 1.0, 3.0));
        let opts = crate::types::FitOptions::default();
        let fd = super::mixture_gradient_fd(&model, &pop, &x, params, &opts);

        let ofv_at = |v: f64| {
            let mut xv = x.clone();
            xv[k] = v;
            super::mixture_ofv(&model, &pop, &unpack_params(&xv, params), &opts, None).ofv
        };
        let h = 5e-7; // a tenth of `mixture_gradient_fd`'s step, `1e-5·|θ₀|`
        let reference = (ofv_at(x[k]) - ofv_at(x[k] - h)) / h;
        assert!(
            reference.is_finite() && reference.abs() > 1e-2,
            "fixture: live slope ({reference})"
        );
        assert!(fd[k].is_finite(), "FD at the bound is {}", fd[k]);
        let rel = (fd[k] - reference).abs() / reference.abs();
        eprintln!(
            "MEASURE #2187 mixture bound: {} vs {reference:e}, rel {rel:e}",
            fd[k]
        );
        assert!(rel < 1e-3, "rel {rel:e}");
    }

    #[test]
    fn mixture_gradient_matches_fd() {
        // Analytic posterior-weighted outer gradient vs central FD of the mixture
        // OFV, over the full packed vector (θ incl. mixing logits, base Ω/σ, and
        // per-class Ω/σ overrides). Exercises the softmax mixing gradient and the
        // override-slot mapping. FOCE FD carries some inner-solve noise, so the
        // tolerance is per-coordinate relative with an absolute floor.
        use crate::estimation::parameterization::pack_params;
        let model = crate::parser::model_parser::parse_model_string(MIX_PARITY_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(5, 1.0, 3.0));
        let opts = crate::types::FitOptions::default();
        let params = &model.default_params;

        let eval = super::mixture_ofv(&model, &pop, params, &opts, None);
        let ana = super::mixture_gradient(&model, &pop, params, &opts, &eval)
            .expect("analytic gradient in scope (logit form, diagonal base Ω)");
        let x = pack_params(params);
        let fd = super::mixture_gradient_fd(&model, &pop, &x, params, &opts);

        assert_eq!(ana.len(), fd.len());
        let mut worst = 0.0f64;
        for (i, (a, f)) in ana.iter().zip(&fd).enumerate() {
            let denom = f.abs().max(a.abs()).max(1e-3);
            let rel = (a - f).abs() / denom;
            worst = worst.max(rel);
            assert!(
                rel < 2e-2,
                "coord {i}: analytic {a:.6} vs FD {f:.6} (rel {rel:.2e})"
            );
        }
        assert!(worst.is_finite());
    }

    #[test]
    fn mixture_gradient_mixnum_branch_matches_fd() {
        // Regression (#980): a MIXNUM-branched typical value (CL = if MIXNUM==1
        // TVCL1… else TVCL2…) *is* in analytic scope (the conditional differentiates
        // in closed form), so `mixture_gradient` returns Some and drives a gradient
        // optimizer. The per-class analytic sensitivity runs under a rayon
        // `par_iter`; the `MIXTURE_CLASS` thread-local must be set *inside* each
        // worker, else class 2 is computed on the TVCL1 branch — a silently wrong,
        // class-swapped gradient (∂/∂TVCL2 ≈ 0). Central FD of the mixture OFV knows
        // nothing of the thread-local, so parity here pins the fix.
        use crate::estimation::parameterization::pack_params;
        let model = crate::parser::model_parser::parse_model_string(MIX_FIT_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(5, 1.0, 3.0));
        let opts = crate::types::FitOptions::default();
        let params = &model.default_params;

        let eval = super::mixture_ofv(&model, &pop, params, &opts, None);
        let ana = super::mixture_gradient(&model, &pop, params, &opts, &eval)
            .expect("MIXNUM-conditional typical value is in analytic scope");
        let x = pack_params(params);
        let fd = super::mixture_gradient_fd(&model, &pop, &x, params, &opts);

        assert_eq!(ana.len(), fd.len());
        // Locate the TVCL1 / TVCL2 theta slots: the class-swap bug zeroes the
        // gradient for the class that isn't the thread-local default (class 1).
        let i_tvcl2 = model
            .theta_names
            .iter()
            .position(|n| n == "TVCL2")
            .expect("TVCL2 theta");
        assert!(
            fd[i_tvcl2].abs() > 1e-2,
            "FD ∂OFV/∂TVCL2 should be clearly nonzero, got {}",
            fd[i_tvcl2]
        );
        for (i, (a, f)) in ana.iter().zip(&fd).enumerate() {
            let denom = f.abs().max(a.abs()).max(1e-3);
            let rel = (a - f).abs() / denom;
            assert!(
                rel < 2e-2,
                "coord {i}: analytic {a:.6} vs FD {f:.6} (rel {rel:.2e})"
            );
        }
    }

    #[test]
    fn mixture_fit_slsqp_uses_analytic_gradient() {
        // A user-chosen NLopt gradient optimizer is honoured for mixtures (Phase 4)
        // and drives the analytic outer gradient. Variance-mixture model → in
        // analytic scope. Just assert it returns Ok with a finite OFV after a few
        // iterations (the gradient path is exercised on every eval).
        let model = crate::parser::model_parser::parse_model_string(MIX_PARITY_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(4, 1.0, 3.0));
        let opts = crate::types::FitOptions {
            optimizer: crate::types::Optimizer::Slsqp,
            outer_maxiter: 4,
            ..crate::types::FitOptions::default()
        };
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("SLSQP mixture fit should return Ok");
        assert!(res.ofv.is_finite(), "OFV = {}", res.ofv);
    }

    #[test]
    fn mixture_downgraded_optimizer_warns() {
        // #984 regression: a mixture cannot be driven by the built-in BFGS /
        // trust-region / Gauss-Newton optimizers (none carry the mixture objective),
        // so an *explicit* such choice is run under BOBYQA — but must say so in
        // FitResult.warnings rather than dropping the choice invisibly.
        let model = crate::parser::model_parser::parse_model_string(MIX_FIT_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(3, 1.0, 3.0));
        let opts = crate::types::FitOptions {
            optimizer: crate::types::Optimizer::Bfgs,
            outer_maxiter: 2,
            ..crate::types::FitOptions::default()
        };
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("mixture fit with a downgraded optimizer should still return Ok");
        assert!(
            res.warnings
                .iter()
                .any(|w| w.contains("BOBYQA") && w.contains("`bfgs`")),
            "expected an optimizer-downgrade warning, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn mixture_default_optimizer_does_not_warn() {
        // The default `auto` optimizer downgrades to BOBYQA by design and must stay
        // silent — only an explicitly-chosen unsupported optimizer warns.
        let model = crate::parser::model_parser::parse_model_string(MIX_FIT_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(3, 1.0, 3.0));
        let opts = crate::types::FitOptions {
            outer_maxiter: 2,
            ..crate::types::FitOptions::default()
        };
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).unwrap();
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("does not carry the mixture objective")),
            "auto optimizer must not emit a downgrade warning, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn mixture_explicit_bobyqa_does_not_warn() {
        // #2852: an explicit `optimizer = bobyqa` is the optimizer the mixture path
        // runs, so the fit must not claim the choice "was replaced by BOBYQA".
        let model = crate::parser::model_parser::parse_model_string(MIX_FIT_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(3, 1.0, 3.0));
        let opts = crate::types::FitOptions {
            optimizer: crate::types::Optimizer::Bobyqa,
            outer_maxiter: 2,
            ..crate::types::FitOptions::default()
        };
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("explicit-bobyqa mixture fit should return Ok");
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("does not carry the mixture objective")),
            "explicit bobyqa runs as requested and must not warn, got {:?}",
            res.warnings
        );
    }

    /// What `resolve_outer_optimizer` owes one requested optimizer on a mixture model.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum OnMixture {
        /// Runs as requested, no warning.
        Honoured,
        /// `auto`: runs BOBYQA, silently.
        SilentBobyqa,
        /// Runs BOBYQA and says so, naming the requested optimizer as the user spells it
        /// in a model file (`Optimizer::label`), written out here so the test does not
        /// borrow the spelling from the code under test.
        Replaced(&'static str),
    }

    /// The warning's whole input space on the optimizer axis (#2852). Exhaustive on
    /// purpose, with no wildcard: a new `Optimizer` variant does not compile until it
    /// has a row here, and then belongs in `ALL_OPTIMIZERS` below.
    fn on_mixture(requested: crate::types::Optimizer) -> OnMixture {
        use crate::types::Optimizer;
        match requested {
            Optimizer::Auto => OnMixture::SilentBobyqa,
            Optimizer::Bobyqa | Optimizer::Slsqp | Optimizer::NloptLbfgs | Optimizer::Mma => {
                OnMixture::Honoured
            }
            Optimizer::Bfgs => OnMixture::Replaced("bfgs"),
            Optimizer::Lbfgs => OnMixture::Replaced("lbfgs"),
            Optimizer::TrustRegion => OnMixture::Replaced("trust_region"),
        }
    }

    #[test]
    fn mixture_replacement_warning_covers_every_optimizer() {
        use crate::estimation::outer_optimizer::resolve_outer_optimizer;
        use crate::types::Optimizer;
        const ALL_OPTIMIZERS: [Optimizer; 8] = [
            Optimizer::Auto,
            Optimizer::Bfgs,
            Optimizer::Lbfgs,
            Optimizer::Slsqp,
            Optimizer::NloptLbfgs,
            Optimizer::Mma,
            Optimizer::Bobyqa,
            Optimizer::TrustRegion,
        ];
        let model = crate::parser::model_parser::parse_model_string(MIX_FIT_MODEL).unwrap();
        for requested in ALL_OPTIMIZERS {
            // The analytic flag is read only by the non-mixture `auto` rule; neither
            // value may change what a mixture runs or whether a warning is owed.
            for analytic in [true, false] {
                let (resolved, warning) =
                    resolve_outer_optimizer(requested, &model, true, analytic);
                match on_mixture(requested) {
                    OnMixture::Honoured => {
                        assert_eq!(resolved, requested, "{requested:?} runs as requested");
                        assert_eq!(warning, None, "{requested:?} is honoured: no warning");
                    }
                    OnMixture::SilentBobyqa => {
                        assert_eq!(resolved, Optimizer::Bobyqa);
                        assert_eq!(warning, None, "auto downgrades silently");
                    }
                    OnMixture::Replaced(name) => {
                        assert_eq!(resolved, Optimizer::Bobyqa, "{requested:?} runs BOBYQA");
                        let w = warning
                            .unwrap_or_else(|| panic!("{requested:?} on a mixture must warn"));
                        assert!(
                            w.contains("does not carry the mixture objective")
                                && w.contains("was replaced by BOBYQA"),
                            "{requested:?}: {w}"
                        );
                        // The advice half: what the user can pick instead.
                        assert!(
                            w.starts_with(
                                "Mixture models are optimized with BOBYQA or an NLopt gradient \
                                 method (SLSQP / L-BFGS / MMA);"
                            ),
                            "{requested:?}: message must say what a mixture can run: {w}"
                        );
                        // The discriminator and its negatives: the message names the
                        // optimizer that was asked for, in option spelling; never the
                        // one that ran ("the requested `bobyqa` optimizer ... was
                        // replaced by BOBYQA"); and never a Rust `Debug` name such as
                        // `TrustRegion`, which no model file contains.
                        assert!(
                            w.contains(&format!("the requested `{name}` optimizer")),
                            "{requested:?}: message must name {name}: {w}"
                        );
                        assert!(!w.contains("`bobyqa`"), "{requested:?}: {w}");
                        assert!(
                            !w.contains(&format!("{requested:?}")) && !w.contains("Bobyqa"),
                            "{requested:?}: no Debug spelling in a user-facing message: {w}"
                        );
                    }
                }

                // Without a mixture nothing is ever replaced: no warning for any
                // optimizer, and an explicit choice is returned unchanged.
                let (resolved, warning) =
                    resolve_outer_optimizer(requested, &model, false, analytic);
                assert_eq!(warning, None, "{requested:?} without a mixture: no warning");
                if requested == Optimizer::Auto {
                    assert_ne!(
                        resolved,
                        Optimizer::Auto,
                        "auto resolves to a concrete optimizer"
                    );
                } else {
                    assert_eq!(resolved, requested, "{requested:?} without a mixture");
                }
            }
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn mixture_fit_recovers_bimodal_cl() {
        // Tier-3: full convergence recovers the two class CLs (≈1 and ≈3) and the
        // class typical values stay separated (label-order convention: class 1 < 2).
        let model = crate::parser::model_parser::parse_model_string(MIX_FIT_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(15, 1.0, 3.0));
        let opts = crate::types::FitOptions::default();
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("mixture fit should converge");
        // theta order: TVCL1, TVCL2, TVV, MIXL, BWT.
        let tvcl1 = res.theta[0];
        let tvcl2 = res.theta[1];
        let tvv = res.theta[2];
        assert!(res.ofv.is_finite());
        assert!((tvcl1 - 1.0).abs() < 0.4, "TVCL1 ≈ 1 expected, got {tvcl1}");
        assert!((tvcl2 - 3.0).abs() < 0.6, "TVCL2 ≈ 3 expected, got {tvcl2}");
        assert!((tvv - 10.0).abs() < 2.0, "TVV ≈ 10 expected, got {tvv}");
        assert!(tvcl2 > tvcl1, "classes must stay separated");
    }

    // ── Per-subject mixture score for the covariance S (#1956) ──

    /// Subject `i`'s mixture marginal `−log Σₖ pᵢₖ e^{−nllᵢₖ}` at packed `x`, every class's
    /// mode re-estimated at `x` (warm-started from the context's) — the function whose total
    /// derivative the score is since #2974, NONMEM's `MATRIX=S` convention. Built from the
    /// objective's pieces directly (`unpack_params`'s override routing, `class_params`, the
    /// inner search, the per-class FOCE nll `mixture_ofv` takes, the mixing map), not from
    /// anything `MixtureScoreContext::score` calls for its routing, mixing gradient or class
    /// weights.
    fn subject_marginal_reestimated(
        model: &crate::types::CompiledModel,
        pop: &crate::types::Population,
        template: &crate::types::ModelParameters,
        x: &[f64],
        i: usize,
        ctx: &super::MixtureScoreContext,
        interaction: bool,
    ) -> f64 {
        let p = crate::estimation::parameterization::unpack_params(x, template);
        let subject = &pop.subjects[i];
        let logp = super::eval_mixing_log_probs(
            model.mixture.as_ref().unwrap(),
            &p.theta,
            &subject.covariates,
        );
        let opts = crate::types::FitOptions {
            interaction,
            ..Default::default()
        };
        let k = p.mixture.as_ref().unwrap().omega.len();
        let nll: Vec<f64> = (0..k)
            .map(|c| {
                let _guard = super::MixtureClassGuard::enter(c + 1);
                let cp = super::class_params(&p, c);
                let warm = crate::estimation::outer_optimizer::joint_warm_start(
                    &ctx.eval.etas_by_class[c][i],
                    &ctx.modes.kappas[c][i],
                );
                let mu_k = crate::estimation::parameterization::compute_mu_k(
                    model,
                    &cp.theta,
                    opts.mu_referencing,
                );
                let ebe = crate::estimation::inner_optimizer::find_ebe_for_stage(
                    model,
                    subject,
                    &cp,
                    1000,
                    1e-12,
                    Some(warm.as_slice()),
                    Some(&mu_k),
                    0,
                    &opts,
                );
                match cp.omega_iov.as_ref() {
                    Some(iov) => super::foce_subject_nll_iov_cov(
                        model,
                        subject,
                        &cp.theta,
                        &ebe.eta,
                        &ebe.h_matrix,
                        &cp.omega,
                        &cp.sigma.values,
                        model.residual_cov_at(&cp),
                        interaction,
                        &ebe.kappas,
                        iov,
                    ),
                    None => super::foce_subject_nll_cov(
                        model,
                        subject,
                        &cp.theta,
                        &ebe.eta,
                        &ebe.h_matrix,
                        &cp.omega,
                        &cp.sigma.values,
                        model.residual_cov_at(&cp),
                        interaction,
                    ),
                }
            })
            .collect();
        super::combine_subject(&logp, &nll).0 / 2.0
    }

    /// Every subject's mixture score against a central FD of its marginal, on every
    /// free packed coordinate, to relative error `|g − fd| / max(|fd|, 1) < tol`. Each
    /// caller's `tol` is its measured worst error (macOS arm64) with ≥ 10× headroom.
    fn assert_score_matches_fd(
        src: &str,
        pop: &crate::types::Population,
        interaction: bool,
        tol: f64,
        label: &str,
    ) {
        use crate::estimation::parameterization::{pack_params, packed_fixed_mask};
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        let template = &model.default_params;
        let opts = crate::types::FitOptions {
            interaction,
            ..Default::default()
        };
        let x = pack_params(template);
        let ctx = super::MixtureScoreContext::new(&model, pop, template, &x, &opts);
        let fixed = packed_fixed_mask(template);
        let f = |xv: &[f64], i: usize| {
            subject_marginal_reestimated(&model, pop, template, xv, i, &ctx, interaction)
        };
        let points = ctx.class_scorers(&model);
        let mut worst = 0.0f64;
        for i in 0..pop.subjects.len() {
            let g = ctx
                .score(&model, pop, i, &points)
                .expect("a finite FOCE/FOCEI class score");
            assert_eq!(g.len(), x.len(), "{label}: score spans the packed layout");
            for j in (0..x.len()).filter(|&j| !fixed[j]) {
                let h = 1e-5 * (1.0 + x[j].abs());
                let (mut xp, mut xm) = (x.clone(), x.clone());
                xp[j] += h;
                xm[j] -= h;
                let fd = (f(&xp, i) - f(&xm, i)) / (2.0 * h);
                // Fold only finite values: `f64::max` would swallow a NaN score.
                assert!(
                    g[j].is_finite() && fd.is_finite(),
                    "{label}: subject {i} coord {j}: score {} / FD {fd}",
                    g[j]
                );
                let err = (g[j] - fd).abs() / fd.abs().max(1.0);
                assert!(
                    err < tol,
                    "{label}: subject {i} coord {j}: score {:.8e} vs FD {fd:.8e} (rel {err:.2e})",
                    g[j]
                );
                worst = worst.max(err);
            }
        }
        eprintln!("{label}: worst score-vs-FD rel error {worst:.3e} (bound {tol:.0e})");
    }

    #[test]
    fn mixture_score_matches_fd_logit_mixnum_branch_focei() {
        // Logit mixing with a covariate, a MIXNUM-branched CL, free Ω/Σ, FOCEI: the
        // closed-form (Laplace) per-class score, the class guard on the worker thread
        // (∂/∂TVCL2 is zero under the default MIXNUM = 1), and the softmax term.
        // Measured worst 2.870e-7.
        let pop = read_pop(&bimodal_csv(4, 1.0, 3.0));
        assert_score_matches_fd(MIX_FIT_MODEL, &pop, true, 1e-5, "logit/MIXNUM FOCEI");
    }

    #[test]
    fn mixture_score_matches_fd_logit_mixnum_branch_foce() {
        // Same fixture under FOCE: the Sheiner–Beal per-class score (its θ axis is a
        // forward FD of the predictions, hence the looser floor). Measured worst 9.221e-5.
        let pop = read_pop(&bimodal_csv(4, 1.0, 3.0));
        assert_score_matches_fd(MIX_FIT_MODEL, &pop, false, 1e-3, "logit/MIXNUM FOCE");
    }

    #[test]
    fn mixture_score_matches_fd_pform_iov() {
        // `p(1)` mixing with IOV:
        // the per-class score takes the FD path over `foce_subject_nll_iov` at the
        // class's own κ̂, and the κ coordinate routes to itself.
        // Measured worst 8.701e-6.
        let pop = read_pop_iov(&bimodal_iov_csv(3, 1.0, 3.0));
        assert_score_matches_fd(MIX_IOV_MODEL, &pop, true, 1e-4, "p-form/IOV FOCEI");
    }

    #[test]
    fn mixture_score_matches_fd_variance_overrides() {
        // Classes differing by `omega(2)` / `sigma(2)` overrides: class 2's base-slot
        // score must land on the override coordinates, class 1's on the base ones.
        // Measured worst 1.074e-7.
        let pop = read_pop(&bimodal_csv(4, 1.0, 3.0));
        assert_score_matches_fd(MIX_PARITY_MODEL, &pop, true, 1e-5, "overrides FOCEI");
    }

    /// FIX base Ω/Σ with *free* class-2 overrides, plus IOV so every class score takes
    /// the FD path — whose fixed mask is the class view's, so class 2's base slots must
    /// be free there although the base declaration is FIX.
    const MIX_FIXED_BASE_FREE_OVERRIDE_MODEL: &str = r"
[parameters]
  theta TVCL(1.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta P1(0.4, 0.001, 0.999)
  omega ETA_CL ~ 0.06 FIX
  kappa KAPPA_CL ~ 0.02
  sigma EPS ~ 0.02 FIX

[mixture]
  nsub = 2
  p(1) = P1
  omega(2) ETA_CL ~ 0.15
  sigma(2) EPS ~ 0.03

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    #[test]
    fn mixture_score_matches_fd_fixed_base_free_override() {
        // Measured worst 9.349e-6.
        let pop = read_pop_iov(&bimodal_iov_csv(3, 1.0, 3.0));
        assert_score_matches_fd(
            MIX_FIXED_BASE_FREE_OVERRIDE_MODEL,
            &pop,
            true,
            1e-4,
            "FIX base / free override, IOV",
        );
    }

    /// A free residual correlation (`block_sigma`, #847): ρ is packed last, after the
    /// mixture override tail, and is shared by every class.
    const MIX_RHO_MODEL: &str = r"
[parameters]
  theta TVCL1(1.2, 0.01, 100.0)
  theta TVCL2(2.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.05
  block_sigma (PROP_ERR, ADD_ERR) = [0.01, 0.02, 0.25]

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
";

    #[test]
    fn mixture_score_matches_fd_residual_correlation() {
        // Measured worst 4.286e-6.
        let model = crate::parser::model_parser::parse_model_string(MIX_RHO_MODEL).unwrap();
        assert!(
            !model.default_params.residual_correlations.is_empty(),
            "fixture precondition: a live residual correlation"
        );
        let pop = read_pop(&bimodal_csv(4, 1.0, 3.0));
        assert_score_matches_fd(MIX_RHO_MODEL, &pop, true, 1e-4, "block_sigma ρ FOCEI");
    }

    #[test]
    fn mixture_gradient_covers_residual_correlation() {
        // The outer gradient spans the whole packed layout, ρ included: it used to be
        // sized `[θ, Ω, σ, overrides]`, one short of `pack_params` on a `block_sigma`
        // mixture, and the outer loop reads `grad_raw[k]` for every packed `k`.
        // Measured worst 8.024e-8 (FOCE).
        use crate::estimation::parameterization::pack_params;
        let model = crate::parser::model_parser::parse_model_string(MIX_RHO_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(4, 1.0, 3.0));
        let opts = crate::types::FitOptions::default();
        let params = &model.default_params;
        let eval = super::mixture_ofv(&model, &pop, params, &opts, None);
        let ana = super::mixture_gradient(&model, &pop, params, &opts, &eval)
            .expect("analytic gradient in scope (logit form, diagonal base Ω)");
        let x = pack_params(params);
        assert_eq!(ana.len(), x.len(), "gradient spans the packed layout");
        let fd = super::mixture_gradient_fd(&model, &pop, &x, params, &opts);
        let mut worst = 0.0f64;
        for (i, (a, f)) in ana.iter().zip(&fd).enumerate() {
            assert!(a.is_finite() && f.is_finite(), "coord {i}: {a} / {f}");
            let rel = (a - f).abs() / f.abs().max(1.0);
            assert!(
                rel < 1e-6,
                "coord {i}: analytic {a:.8e} vs FD {f:.8e} (rel {rel:.2e})"
            );
            worst = worst.max(rel);
        }
        eprintln!("ρ mixture gradient: worst rel error {worst:.3e}");
    }

    #[test]
    fn mixture_score_modes_use_cov_inner_tol_not_fit_tol() {
        // The S path's per-class modes are reconverged at `cov_inner_tol`, like the R
        // path's (`mixture_covariance_uses_cov_inner_tol_not_fit_tol`): an absurdly
        // loose fit tol with a tight explicit cov tol must give the same S as a tight
        // fit tol. Measured worst 0 (bit-identical, macOS arm64); modes at the fit tol
        // instead give S[0,0] 0.1357 vs 0.5454.
        use crate::estimation::covariance::assemble_score_cross_product;
        use crate::estimation::parameterization::{pack_params, pack_with_bounds};
        let model = crate::parser::model_parser::parse_model_string(MIX_COV_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(8, 1.0, 3.0));
        let params = &model.default_params;
        let x = pack_params(params);
        let start = pack_with_bounds(params);
        let free: Vec<usize> = (0..x.len()).filter(|&j| !start.fixed[j]).collect();
        let run = |inner_tol: f64, cov_inner_tol: Option<f64>| {
            let opts = crate::types::FitOptions {
                inner_tol,
                cov_inner_tol,
                interaction: true,
                ..crate::types::FitOptions::default()
            };
            let eval = super::mixture_ofv(&model, &pop, params, &opts, None);
            assemble_score_cross_product(
                &x,
                params,
                &model,
                &pop,
                &eval.mixest_etas,
                &eval.mixest_kappas,
                &start.bounds,
                &opts,
                &free,
            )
            .expect("finite S")
        };
        let reference = run(1e-8, None);
        let probe = run(1.0, Some(1e-8));
        let mut worst = 0.0f64;
        for j in 0..free.len() {
            for k in 0..free.len() {
                let (r, p) = (reference[(j, k)], probe[(j, k)]);
                assert!(r.is_finite() && p.is_finite(), "S[{j},{k}]: {r} / {p}");
                let rel = (r - p).abs() / r.abs().max(p.abs()).max(1e-9);
                assert!(
                    rel < 1e-6,
                    "S[{j},{k}] depends on the fit inner_tol: ref {r} vs probe {p}"
                );
                worst = worst.max(rel);
            }
        }
        eprintln!("S at cov_inner_tol vs tight fit tol: worst rel {worst:.3e} (bound 1e-6)");
    }

    #[test]
    fn mixture_s_covariance_is_full_rank() {
        // #1956: the S path scored a mixture as the single-population class-1
        // objective, so the class-2-only θ (TVCL2) and the mixing θ (P1) had
        // identically-zero score columns and `covariance_method = s` was refused as
        // singular. With the mixture score every free θ carries information.
        use crate::estimation::covariance::{
            assemble_score_cross_product, compute_covariance, CovarianceStepResult,
        };
        use crate::estimation::parameterization::{pack_params, pack_with_bounds};
        let model = crate::parser::model_parser::parse_model_string(MIX_COV_MODEL).unwrap();
        let pop = read_pop(&bimodal_csv(8, 1.0, 3.0));
        let params = &model.default_params;
        let x = pack_params(params);
        let start = pack_with_bounds(params);
        let free: Vec<usize> = (0..x.len()).filter(|&j| !start.fixed[j]).collect();
        assert_eq!(free, [0, 1, 2, 3], "the four θ are the free coordinates");
        let opts = crate::types::FitOptions {
            interaction: true,
            ..Default::default()
        };
        let eval = super::mixture_ofv(&model, &pop, params, &opts, None);
        let s = assemble_score_cross_product(
            &x,
            params,
            &model,
            &pop,
            &eval.mixest_etas,
            &eval.mixest_kappas,
            &start.bounds,
            &opts,
            &free,
        )
        .expect("finite S");
        for j in 0..4 {
            assert!(
                s[(j, j)] > 0.0,
                "S[{j},{j}] = {} must be positive",
                s[(j, j)]
            );
        }
        let eig = nalgebra::SymmetricEigen::new(s.clone()).eigenvalues;
        let (lo, hi) = (eig.min(), eig.max());
        assert!(lo > 1e-8 * hi, "S must be full rank: eigenvalues {eig:?}");

        // End to end: `s` and `rsr` both yield positive θ variances.
        for method in [
            crate::types::CovarianceMethod::CrossProduct,
            crate::types::CovarianceMethod::Sandwich,
        ] {
            let o = crate::types::FitOptions {
                covariance_method: method,
                ..opts.clone()
            };
            let matrix = match compute_covariance(
                &x,
                params,
                &model,
                &pop,
                &eval.mixest_etas,
                &eval.mixest_h_mats,
                &[],
                &o,
            ) {
                CovarianceStepResult::Success(out) => out.matrix,
                CovarianceStepResult::Unusable(msg) => panic!("{method:?} unusable: {msg}"),
                CovarianceStepResult::FailedNonPd { reason, .. } => {
                    panic!("{method:?} non-PD: {reason}")
                }
            };
            for j in 0..4 {
                let v = matrix[(j, j)];
                assert!(v.is_finite() && v > 0.0, "{method:?}: θ[{j}] variance {v}");
            }
        }
    }

    /// A class solve warm from an earlier pass's class modes takes an exactly-zero one — the
    /// prior mode of a subject with no observations — as warm, as the standard inner loop
    /// does (`PrevEtas::Solved`). The outer loop hands every evaluation after the first the
    /// previous one's class modes, so read by value alone the empty subject reran its cold
    /// multi-start in every class at every evaluation. The straddle: from no warm start the
    /// same subject is multi-started in each class, and the warm pass does solve it.
    #[test]
    fn a_mixture_class_solve_takes_an_exactly_zero_warm_mode_as_warm() {
        use crate::estimation::inner_optimizer::record_inner_solves;
        let model = crate::parser::model_parser::parse_model_string(MIX_FIT_MODEL).unwrap();
        // A dose-only subject: nothing informs its η, so its mode is exactly the prior's.
        let pop = read_pop(&format!("{}99,0,0,100,1,1,75\n", bimodal_csv(2, 1.0, 3.0)));
        let empty = "99".to_string();
        assert!(pop
            .subjects
            .iter()
            .any(|s| s.id == empty && s.observations.is_empty()));
        let opts = crate::types::FitOptions {
            inner_restarts: 1,
            ..Default::default()
        };
        let params = &model.default_params;
        let k = 2;
        let count = |ids: &[String]| ids.iter().filter(|id| **id == empty).count();

        let (cold, cold_rec) =
            record_inner_solves(|| super::mixture_ofv(&model, &pop, params, &opts, None));
        assert_eq!(
            count(&cold_rec.multistarts),
            k,
            "fixture: a cold pass multi-starts the empty subject once per class"
        );
        let at = pop.subjects.iter().position(|s| s.id == empty).unwrap();
        for (cls, etas) in cold.etas_by_class.iter().enumerate() {
            assert!(
                etas[at].iter().all(|&v| v == 0.0),
                "fixture: class {cls}'s mode of the empty subject is {:?}, not exactly zero",
                etas[at]
            );
        }

        let (_, warm_rec) = record_inner_solves(|| {
            super::mixture_ofv(&model, &pop, params, &opts, Some(&cold.etas_by_class))
        });
        let solved: Vec<String> = warm_rec
            .solve_tols
            .iter()
            .map(|(id, _)| id.clone())
            .collect();
        assert_eq!(
            count(&solved),
            k,
            "the warm pass solves the empty subject per class"
        );
        assert_eq!(
            count(&warm_rec.multistarts),
            0,
            "an exactly-zero warm class mode was multi-started as a cold start"
        );
    }

    /// #2906: the oral two-class mixture of `api/tests/outer_fd_fallback_warning_tests.rs`
    /// (lag 0.25), on the first four warfarin subjects. Subject 0's bolus is replaced by a
    /// steady-state infusion into the depot, which the adjoint declines under a lag time
    /// (`ss_lagtime_walk_unsupported`); the other three stay in scope.
    fn mixture_with_one_declined_subject() -> (crate::types::CompiledModel, crate::types::Population)
    {
        let (model, pop) = mixture_with_one_declinable_subject();
        // No regular-path shape declines since #2786, so the decline is forced: every plan of
        // subject 0 fails, and the other three stay in scope.
        crate::parser::model_parser::rhs_regcode::kernel_jit::fail_plans_of(&pop.subjects[0]);
        (model, pop)
    }

    /// [`mixture_with_one_declined_subject`] before its plans are failed: the in-scope twin.
    fn mixture_with_one_declinable_subject(
    ) -> (crate::types::CompiledModel, crate::types::Population) {
        use crate::types::{DoseEvent, RateMode};
        let model = crate::parser::model_parser::parse_model_string(
            r"
[parameters]
  theta TVCL1(0.15, 0.001, 10.0)
  theta TVCL2(0.30, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVF(0.7, 0.05, 1.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  F  = TVF
  ALAG = 0.25

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F, lagtime=ALAG)

[error_model]
  DV ~ proportional(PROP_ERR)
",
        )
        .unwrap();
        let mut pop = crate::io::datareader::read_nonmem_csv(
            std::path::Path::new("data/warfarin.csv"),
            None,
            None,
        )
        .unwrap();
        pop.subjects.truncate(4);
        let mut dose = DoseEvent::new(0.0, 100.0, 1, 50.0, true, 24.0);
        dose.rate_mode = RateMode::Fixed;
        pop.subjects[0].doses = vec![dose];
        (model, pop)
    }

    /// #3025: the covariance step's per-subject mixture score under Laplace / AGQ is the
    /// single population's quadrature score on each class, so `2 Σᵢ gᵢ` is the outer gradient
    /// of the same objective (both are the total derivative, the modes re-estimated). A score
    /// left on the FOCE/FOCEI class objective misses by the Laplace correction. Free
    /// coordinates, every fixture, both quadrature objectives.
    #[test]
    fn quadrature_mixture_scores_sum_to_the_outer_gradient() {
        use crate::estimation::parameterization::{pack_params, packed_fixed_mask};
        let mut worst = 0.0f64;
        for (tag, model, pop) in mixture_cases() {
            let params = &model.default_params;
            let x = pack_params(params);
            let fixed = packed_fixed_mask(params);
            for (kind, opts) in objective_cases() {
                if kind == "foce" || kind == "focei" {
                    continue;
                }
                let grad = exact_mixture_grad(&model, &pop, params, &opts);
                let ctx = super::MixtureScoreContext::new(&model, &pop, params, &x, &opts);
                let scorers = ctx.class_scorers(&model);
                let mut sum = vec![0.0f64; x.len()];
                for i in 0..pop.subjects.len() {
                    let g = ctx
                        .score(&model, &pop, i, &scorers)
                        .unwrap_or_else(|| panic!("{tag} {kind}: subject {i} has a score"));
                    for (a, v) in sum.iter_mut().zip(&g) {
                        *a += 2.0 * v;
                    }
                }
                let mut gap = 0.0f64;
                for j in (0..x.len()).filter(|&j| !fixed[j]) {
                    assert!(
                        sum[j].is_finite() && grad[j].is_finite(),
                        "{tag} {kind} [{j}]"
                    );
                    gap = gap.max((sum[j] - grad[j]).abs() / (1.0 + grad[j].abs()));
                }
                eprintln!("{tag} {kind}: 2 Σ scores vs outer gradient {gap:.2e}");
                worst = worst.max(gap); // nan-ok: gap is a fold of asserted-finite operands
            }
        }
        assert!(worst < 1e-6, "{worst:.2e}");
    }

    /// #3025: the per-term salvage under the quadrature objective. Subject 0's plans fail, so
    /// its (subject × class) terms are salvaged by the reconverged quadrature difference
    /// (`SubjectScoreContext::reconverged_score`; not the FOCE one: under Laplace / AGQ the two
    /// objectives differ by the Laplace correction, far above the bound). The oracle is the
    /// exact gradient of the in-scope twin of the same fixture (plans not failed) — not a
    /// whole-objective difference, which here differentiates a Hessian the declined subject's
    /// objective itself finite-differences. The fixture straddles the gate: the strict assembly
    /// declines, and subject 0's salvaged terms are live.
    #[test]
    fn a_declined_subject_is_salvaged_on_the_quadrature_objective() {
        use crate::estimation::outer_optimizer::OuterFdDeclineLog;
        use crate::estimation::parameterization::pack_params;
        let (model, pop) = mixture_with_one_declined_subject();
        let (twin, mut twin_pop) = mixture_with_one_declinable_subject();
        // The failing plans are keyed by the subject's event structure: moving one observation
        // by 1e-9 h gives the twin plans of its own, and an objective unchanged to ~1e-9.
        twin_pop.subjects[0].obs_times[0] += 1e-9;
        twin_pop.subjects[0].obs_raw_times[0] += 1e-9;
        let params = &model.default_params;
        for (kind, mut opts) in objective_cases() {
            // Laplace only: under the Gauss-Newton anchor (`agq3`) a subject whose plans fail has
            // no anchor at all, so its objective is the repel sentinel, as for a single
            // population (`agq_subject_evaluate`), and there is no term to salvage.
            if kind != "laplace" {
                continue;
            }
            opts.inner_tol = 1e-8;
            let exact = exact_mixture_grad(&twin, &twin_pop, &twin.default_params, &opts);
            let (eval, modes) = super::mixture_ofv_with_modes(&model, &pop, params, &opts, None);
            assert!(
                super::mixture_gradient_with_modes(
                    &model,
                    &pop,
                    params,
                    &opts,
                    &eval,
                    Some(&modes)
                )
                .is_none(),
                "{kind}: fixture: the strict assembly must decline subject 0"
            );
            let log = OuterFdDeclineLog::new(pop.subjects.len());
            let salvaged = super::mixture_gradient_salvaged(
                &model,
                &pop,
                params,
                &opts,
                &eval,
                Some(&modes),
                &log,
            );
            assert_eq!(
                log.declined_indices(),
                vec![0],
                "{kind}: only subject 0 declined"
            );
            let zero = |d: super::DeclinedTerm<'_>| Some(vec![0.0; pack_params(d.cp).len()]);
            let rows = super::mixture_subject_gradients(
                &model,
                &pop,
                params,
                &opts,
                &eval,
                Some(&modes),
                Some(&zero),
                |f| f(),
            )
            .unwrap();
            let mut zeroed = vec![0.0; salvaged.len()];
            for r in &rows {
                for (a, v) in zeroed.iter_mut().zip(r) {
                    *a += 2.0 * v;
                }
            }
            let (mut worst, mut live) = (0.0f64, 0.0f64);
            for (j, ((&s, &e), &z)) in salvaged.iter().zip(&exact).zip(&zeroed).enumerate() {
                assert!(
                    s.is_finite() && e.is_finite() && z.is_finite(),
                    "{kind} coord {j}"
                );
                worst = worst.max((s - e).abs() / e.abs().max(1.0));
                live = live.max((s - z).abs() / e.abs().max(1.0));
            }
            eprintln!(
                "MEASURE #3025 {kind} salvaged vs in-scope exact: worst {worst:e}; live {live:e}"
            );
            // Measured 2.1e-3 (Laplace): the declined subject's own objective finite-differences
            // its Hessian, so its reconverged difference is the noisier side. The FOCE-objective
            // salvage (the mutation this pins) is off by 3.7e-2.
            assert!(
                worst < 1e-2,
                "{kind}: salvaged vs exact worst rel {worst:e}"
            );
            assert!(
                live > 100.0 * worst,
                "{kind}: subject 0's terms must be live"
            );
        }
    }

    /// #2906: one declined subject used to send the whole mixture gradient to
    /// [`super::mixture_gradient_fd`] (`mixture_gradient_with_modes` → `None`). The salvaged
    /// gradient keeps the adjoint's terms for subjects 1–3 and differences only subject 0's
    /// (subject × class) terms. It must agree with the whole-objective central difference —
    /// the independent oracle, which re-solves every subject in every class — on every
    /// coordinate, and the subject must be recorded (structurally: no recorder serves an SS
    /// infusion under a lag) so the #1154 warning reports it with the reconverged sentence,
    /// never the held-EBE remedy.
    ///
    /// The fixture straddles the gate: the strict assembly declines it (so the salvage is
    /// what produced the gradient), and subject 0's terms are live (zeroing them moves the
    /// gradient by far more than the parity bound).
    #[test]
    fn a_declined_subject_is_salvaged_alone_and_recorded() {
        use crate::estimation::outer_optimizer::{outer_fd_fallback_warning, OuterFdDeclineLog};
        use crate::estimation::parameterization::pack_params;
        let (model, pop) = mixture_with_one_declined_subject();
        let opts = crate::types::FitOptions {
            method: crate::types::EstimationMethod::FoceI,
            interaction: true,
            // The oracle re-solves every subject cold at each probe, so it is the noisier
            // side: at the default `inner_tol` it moves by ~6e-4 on TVCL2 / Ω(KA) between
            // tolerances while the salvaged gradient holds to ~1e-5. At 1e-6 the measured
            // worst gap is 9.2e-5 (macOS arm64), hence the 5e-4 bound below.
            inner_tol: 1e-6,
            ..Default::default()
        };
        let params = &model.default_params;
        let (eval, modes) = super::mixture_ofv_with_modes(&model, &pop, params, &opts, None);
        assert!(
            super::mixture_gradient_with_modes(&model, &pop, params, &opts, &eval, Some(&modes))
                .is_none(),
            "fixture: the strict assembly must decline subject 0"
        );
        let log = OuterFdDeclineLog::new(pop.subjects.len());
        let salvaged = super::mixture_gradient_salvaged(
            &model,
            &pop,
            params,
            &opts,
            &eval,
            Some(&modes),
            &log,
        );
        assert_eq!(log.declined_indices(), vec![0], "only subject 0 declined");

        // The same assembly with subject 0's salvage zeroed: how much its terms carry.
        let zero = |d: super::DeclinedTerm<'_>| Some(vec![0.0; pack_params(d.cp).len()]);
        let zeroed: Vec<f64> = {
            let rows = super::mixture_subject_gradients(
                &model,
                &pop,
                params,
                &opts,
                &eval,
                Some(&modes),
                Some(&zero),
                |f| f(),
            )
            .unwrap();
            let mut g = vec![0.0; salvaged.len()];
            for r in &rows {
                for (a, v) in g.iter_mut().zip(r) {
                    *a += 2.0 * v;
                }
            }
            g
        };

        let x = pack_params(params);
        let fd = super::mixture_gradient_fd(&model, &pop, &x, params, &opts);
        assert_eq!(salvaged.len(), fd.len());
        let (mut worst, mut live) = (0.0f64, 0.0f64);
        for (j, ((&s, &f), &z)) in salvaged.iter().zip(&fd).zip(&zeroed).enumerate() {
            assert!(
                s.is_finite() && f.is_finite() && z.is_finite(),
                "coord {j}: {s} {f} {z}"
            );
            let rel = (s - f).abs() / f.abs().max(1.0);
            let moved = (s - z).abs() / f.abs().max(1.0);
            assert!(rel.is_finite() && moved.is_finite(), "coord {j}");
            worst = worst.max(rel);
            live = live.max(moved);
        }
        eprintln!("MEASURE #2906 salvaged vs whole-objective FD: worst rel {worst:e}; subject 0's terms move it by {live:e}");
        assert!(worst < 5e-4, "salvaged vs FD worst rel {worst:e}");
        assert!(
            live > 100.0 * worst,
            "fixture: subject 0's salvaged terms must be live ({live:e} vs {worst:e})"
        );

        let w = outer_fd_fallback_warning(&pop, &log).expect("the decline warns");
        assert!(
            w.contains("1 of 4") && w.contains(&format!("subject {}", pop.subjects[0].id)),
            "{w}"
        );
        assert!(
            w.contains("no analytic recorder serves them") && !w.contains("fixed-EBE"),
            "a structural mixture decline is reconverged, not held-EBE: {w}"
        );
    }

    /// The mixing map's derivatives are recorded once per distinct value of the covariates the
    /// mixing expressions read, not once per subject: a covariate-free map records once for all
    /// four subjects, a `WT`-dependent one once per distinct `WT` (two values over four
    /// subjects), and every subject's rows are the bits a direct call returns. Both sides of the
    /// key in one test: a cache keyed too coarsely returns one subject's rows for another's `WT`
    /// (the rows differ between the two values, asserted), one keyed per subject records four times.
    #[test]
    fn mixing_derivatives_record_once_per_distinct_covariate_value() {
        use super::{mixing_logp_grad, MixingGradCache};
        let (free_model, pop) = mixture_with_one_declined_subject();
        let cov_model = crate::parser::model_parser::parse_model_string(
            r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta MIXL(0.1, -10.0, 10.0)
  theta BWT(0.05, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)

[mixture]
  nsub = 2
  logit(1) = MIXL + BWT * (WT - 70)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
",
        )
        .unwrap();
        for (model, wts, want_records) in [
            (&free_model, [70.0, 70.0, 70.0, 70.0], 1usize),
            (&cov_model, [60.0, 90.0, 60.0, 90.0], 2),
        ] {
            let spec = model.mixture.as_ref().unwrap();
            let theta = &model.default_params.theta;
            let floor = crate::estimation::parameterization::fd_theta_step_floors(model);
            let cache = MixingGradCache::new(model);
            let mut rows = Vec::new();
            for (i, subject) in pop.subjects.iter().enumerate() {
                let mut cov = subject.covariates.clone();
                cov.insert("WT".to_string(), wts[i]);
                let got = cache.rows(model, spec, theta, &cov);
                let want = mixing_logp_grad(model, spec, theta, &cov, &floor);
                let bits = |m: &Vec<Vec<f64>>| -> Vec<u64> {
                    m.iter().flatten().map(|v| v.to_bits()).collect()
                };
                assert!(got.iter().flatten().all(|v| v.is_finite()));
                assert_eq!(bits(&got), bits(&want), "subject {i}: not the direct bits");
                rows.push(got);
            }
            assert_eq!(cache.computed(), want_records, "recordings");
            if want_records == 2 {
                let a: Vec<u64> = rows[0].iter().flatten().map(|v| v.to_bits()).collect();
                let b: Vec<u64> = rows[1].iter().flatten().map(|v| v.to_bits()).collect();
                assert_ne!(a, b, "fixture: the two WT values must give different rows");
            }
        }
    }

    /// The outer gradient's per-subject assembly takes its mixing derivatives through the
    /// shared cache: four subjects of a covariate-free mixing map make one recording.
    #[test]
    fn the_subject_gradients_record_the_mixing_map_once() {
        use crate::estimation::parameterization::pack_params;
        let (model, pop) = mixture_with_one_declined_subject();
        let opts = crate::types::FitOptions {
            method: crate::types::EstimationMethod::FoceI,
            interaction: true,
            ..Default::default()
        };
        let params = &model.default_params;
        let (eval, modes) = super::mixture_ofv_with_modes(&model, &pop, params, &opts, None);
        let zero = |d: super::DeclinedTerm<'_>| Some(vec![0.0; pack_params(d.cp).len()]);
        super::MIXING_RECORDINGS.with(|c| c.set(usize::MAX));
        let rows = super::mixture_subject_gradients(
            &model,
            &pop,
            params,
            &opts,
            &eval,
            Some(&modes),
            Some(&zero),
            |f| f(),
        )
        .expect("salvaged");
        assert_eq!(rows.len(), 4);
        assert_eq!(super::MIXING_RECORDINGS.with(|c| c.get()), 1);
    }
}
