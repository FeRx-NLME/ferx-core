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

use crate::estimation::inner_optimizer::{find_ebe_for_stage, InnerLoopStats};
use crate::estimation::parameterization::{
    fd_step_floors, fd_theta_step_floors, pack_params, pack_with_bounds, packed_segments,
    theta_packs_log, unpack_params, PackedBounds,
};
use crate::estimation::sens_outer_gradient::per_subject_packed_gradients;
use crate::parser::model_parser::{
    eval_mixing_log_probs, mixing_logp_grad, mixing_logp_grad_fd, MixtureClassGuard,
};
use crate::stats::likelihood::{foce_subject_nll_cov, foce_subject_nll_iov_cov};
use crate::types::{CompiledModel, FitOptions, MixtureParams, ModelParameters, Population};

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

/// The pairs of mixture classes that cannot be told apart at `params`, as class numbers the way
/// `MIXNUM` spells them (`(1, 2)` is the first two classes), in ascending order. A pair is
/// indistinguishable when every subject has the same mixing probability under both classes, and
/// the same individual objective at η = 0 and at a probe η (0.3 prior SD on each random effect).
/// Empty when every pair can be told apart, and for a model without a mixture.
///
/// One such pair is enough. The mixture objective is then symmetric under swapping that pair, so
/// its gradient is: a gradient optimizer started there moves both of them the same way and never
/// separates them. It stops on that saddle and reports it converged (two identical classes on
/// `tests/nonmem/mixture_iv.csv`: OFV 350.27 against the optimum's 301.58). A derivative-free
/// optimizer's probe steps break the symmetry, so `optimizer = auto` runs BOBYQA from such a
/// start instead.
///
/// Pairwise, not all-classes: with `nsub = 3`, two identical classes beside a third that differs
/// leave the identical two inseparable, so the fit can only ever reach the best solution in which
/// they stay merged. Requiring every class to match would call that start distinguishable and hand
/// it to L-BFGS, which is the failure this guard exists to prevent.
///
/// The comparison is exact (`to_bits`): identical classes are the same computation on the same
/// inputs. It can err only towards "symmetric", for classes that differ only away from the two
/// η probed or only through κ (which `individual_nll` holds at 0), and that costs nothing but
/// the faster optimizer.
///
/// Every pair is returned, not the first, so the warning can name each pair the user has to
/// separate: with three identical classes, separating classes 1 and 2 still leaves 1 and 3.
pub(crate) fn indistinguishable_class_pairs(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
) -> Vec<(usize, usize)> {
    let (Some(spec), Some(mp)) = (model.mixture.as_ref(), params.mixture.as_ref()) else {
        return Vec::new();
    };
    let k = mp.omega.len();
    if k < 2 {
        return Vec::new();
    }
    let zeros = vec![0.0; model.n_eta];
    let probe: Vec<f64> = (0..model.n_eta)
        .map(|j| 0.3 * params.omega.matrix[(j, j)].max(0.0).sqrt())
        .collect();
    let classes: Vec<ModelParameters> = (0..k).map(|cls| class_params(params, cls)).collect();
    // Every pair still symmetric on every subject seen so far. A subject that tells a pair apart
    // drops it; an empty set means no pair can be symmetric and the scan stops there.
    let mut pairs: Vec<(usize, usize)> = (0..k)
        .flat_map(|a| (a + 1..k).map(move |b| (a, b)))
        .collect();
    for subject in &population.subjects {
        let logp = eval_mixing_log_probs(spec, &params.theta, &subject.covariates);
        let nlls: Vec<Vec<u64>> = [&zeros, &probe]
            .iter()
            .map(|eta| {
                classes
                    .iter()
                    .enumerate()
                    .map(|(cls, cp)| {
                        let _guard = MixtureClassGuard::enter(cls + 1);
                        crate::stats::likelihood::individual_nll(
                            model,
                            subject,
                            &cp.theta,
                            eta,
                            &cp.omega,
                            &cp.sigma.values,
                        )
                        .to_bits()
                    })
                    .collect()
            })
            .collect();
        pairs.retain(|&(a, b)| {
            logp[a].to_bits() == logp[b].to_bits() && nlls.iter().all(|n| n[a] == n[b])
        });
        if pairs.is_empty() {
            return pairs;
        }
    }
    pairs.into_iter().map(|(a, b)| (a + 1, b + 1)).collect()
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
/// (`[class][subject]`) to warm-start each class's inner solve.
pub fn mixture_ofv(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    warm: Option<&[Vec<DVector<f64>>]>,
) -> MixtureEval {
    mixture_ofv_with_modes(model, population, params, options, warm).0
}

/// Every class's inner modes at one mixture evaluation, `[class][subject]`: the
/// per-class H and κ̂ that `MixtureEval` keeps only for the MIXEST class (its
/// [`MixtureEval::etas_by_class`] already holds every class). The covariance step's
/// per-subject mixture score (#1956) differentiates every class's `nll_ik` at its own
/// fixed modes, so it needs them all.
pub(crate) struct MixtureClassModes {
    /// Inner Hessian / linearization `H` per class per subject.
    pub(crate) h_mats: Vec<Vec<DMatrix<f64>>>,
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
            let ebe = find_ebe_for_stage(
                model,
                subject,
                &cp,
                options.inner_maxiter,
                options.inner_tol,
                warm_i,
                None,
                options.inner_restarts,
                options,
            );
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
            nll[cls][i] = if let Some(omega_iov) = cp.omega_iov.as_ref() {
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
        h_mats: hmats_by_class,
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
/// Returns `None` (→ the caller falls back to FD of the mixture OFV) when the
/// gradient is out of analytic scope: inter-occasion variability (κ), whose per-class κ̂
/// `eval` does not carry (`mixture_gradient_with_modes` takes them), or any
/// subject/class the per-subject adjoint gradient (`per_subject_packed_gradients`) declines.
pub fn mixture_gradient(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    eval: &MixtureEval,
) -> Option<Vec<f64>> {
    mixture_gradient_with_modes(model, population, params, options, eval, None)
}

/// [`mixture_gradient`] with every class's modes from the same evaluation
/// ([`mixture_ofv_with_modes`]): under IOV each class's subject gradient is taken at its
/// stacked `[η̂, κ̂]`, so an IOV mixture is exact too. The base Ω may be a block: a class
/// override can only name a diagonal base Ω's coordinate (a parse-time rule), so a block
/// Ω's coordinates route to themselves.
pub(crate) fn mixture_gradient_with_modes(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    eval: &MixtureEval,
    modes: Option<&MixtureClassModes>,
) -> Option<Vec<f64>> {
    mixture_gradient_assembled(model, population, params, options, eval, modes, |_| None)
}

/// [`mixture_gradient_with_modes`] that **salvages** a declined (subject × class) term
/// instead of returning `None` for the whole population (#2906), as #1529 does for a
/// single population: every term the adjoint serves keeps its exact gradient, and only a
/// declined `∂nll_ik/∂x` is replaced by that one subject's reconverged central
/// difference under class `k` ([`class_subject_reconverged_gradient`]) — `2·n_free`
/// solves of one subject in one class, where the whole-objective
/// [`mixture_gradient_fd`] re-solves every subject in every class per probe. A term
/// whose posterior weight is exactly zero carries no gradient and is not differenced.
///
/// Each subject that took a salvage is recorded on `declines` (structural when no
/// recorder serves the subject's data shape under that class, as
/// `outer_optimizer::is_structural_decline` asks), so the fit's #1154 warning reports
/// it. Still `None` when the mixing map declines (a model-level gap, so the caller's
/// whole-objective FD is the only gradient there is), or under IOV without `modes`.
pub(crate) fn mixture_gradient_salvaged(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    eval: &MixtureEval,
    modes: Option<&MixtureClassModes>,
    declines: &crate::estimation::outer_optimizer::OuterFdDeclineLog,
) -> Option<Vec<f64>> {
    mixture_gradient_assembled(model, population, params, options, eval, modes, |d| {
        let subject = &population.subjects[d.subject];
        declines.record(d.subject);
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
    })
}

/// One (subject × class) term [`mixture_gradient_assembled`] could not take from the
/// adjoint: what its salvage needs.
struct DeclinedTerm<'a> {
    /// The subject's index in the population.
    subject: usize,
    /// The class, 0-based.
    cls: usize,
    /// The class parameters, [`class_params`].
    cp: &'a ModelParameters,
    /// The subject's class-`cls` mode, stacked `[η̂, κ̂]` under IOV.
    eta: &'a DVector<f64>,
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
    if cp.omega_iov.is_some() {
        crate::estimation::outer_optimizer::subject_reconverged_fd_gradient_iov(
            &x_k, &cp, model, subject, eta, &bounds, options,
        )
    } else {
        crate::estimation::outer_optimizer::subject_reconverged_fd_gradient(
            &x_k, &cp, model, subject, eta, &bounds, options,
        )
    }
}

/// The posterior-weighted mixture gradient both [`mixture_gradient_with_modes`] (strict:
/// `salvage` declines every term) and [`mixture_gradient_salvaged`] assemble. `salvage`
/// fills a (subject × class) term the adjoint declined with `d(nll_ik)/dx_k` over the
/// class layout, or `None` to give up on the whole gradient. It is called only for a
/// term with non-zero posterior weight.
fn mixture_gradient_assembled(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    eval: &MixtureEval,
    modes: Option<&MixtureClassModes>,
    mut salvage: impl FnMut(DeclinedTerm<'_>) -> Option<Vec<f64>>,
) -> Option<Vec<f64>> {
    let spec = model.mixture.as_ref()?;
    let mp = params.mixture.as_ref()?;
    if params.omega_iov.is_some() && modes.is_none() {
        return None;
    }
    let k = mp.omega.len();
    let interaction = options.interaction;
    let dtheta_dx = theta_chain_factors(params);
    let mut grad = vec![0.0f64; packed_segments(params).total()];

    for cls in 0..k {
        let cp = class_params(params, cls);
        let x_k = pack_params(&cp);
        // ∂nll_ik/∂x_k over the class-k packed layout [θ, Ω(base slots), σ, …].
        // The class thread-local is set inside each rayon worker (the outer-thread
        // guard would not reach them), so pass `Some(cls + 1)` rather than relying
        // on a `MixtureClassGuard` on this thread.
        // Under IOV, every subject's class mode stacked with its κ̂.
        let stacked: Vec<DVector<f64>>;
        let etas = match modes.filter(|_| params.omega_iov.is_some()) {
            Some(m) => {
                stacked = population
                    .subjects
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        crate::estimation::covariance::stacked_mode(
                            model,
                            s,
                            &eval.etas_by_class[cls][i],
                            m.kappas[cls].get(i),
                        )
                        .map(|b| DVector::from_column_slice(&b))
                    })
                    .collect::<Option<_>>()?;
                &stacked
            }
            None => &eval.etas_by_class[cls],
        };
        let g_k = per_subject_packed_gradients(
            model,
            population,
            &cp,
            &x_k,
            etas,
            interaction,
            Some(cls + 1),
        );
        for (i, gk_i) in g_k.iter().enumerate() {
            let w = eval.pmix[i][cls];
            if w == 0.0 {
                continue;
            }
            let salvaged;
            let gi = match gk_i.as_ref().filter(|g| g.iter().all(|v| v.is_finite())) {
                Some(g) => g,
                // Out-of-scope or non-finite term → the caller's salvage (or `None`).
                None => {
                    salvaged = salvage(DeclinedTerm {
                        subject: i,
                        cls,
                        cp: &cp,
                        eta: &etas[i],
                    })?;
                    &salvaged
                }
            };
            route_class_gradient(&mut grad, gi, 2.0 * w, cls, params, mp);
        }
    }

    // Softmax mixing gradient: −2 Σ_i Σ_k w_ik ∂ln p_ik/∂θ_j (theta slots).
    let theta_floor = fd_theta_step_floors(model);
    for (i, subject) in population.subjects.iter().enumerate() {
        let dlnp = mixing_logp_grad(spec, &params.theta, &subject.covariates, &theta_floor)?;
        for (cls, dlnp_c) in dlnp.iter().enumerate() {
            let w = eval.pmix[i][cls];
            if w == 0.0 {
                continue;
            }
            subtract_mixing_gradient(&mut grad, dlnp_c, 2.0 * w, &dtheta_dx);
        }
    }

    Some(grad)
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
/// packed layout at `x_hat`, or `None` (the caller keeps the stencil) when any subject's
/// class is outside the adjoint's scope or the mixing map declines.
///
/// Per subject, with `u_k = ln p_k − F_k` (`F_k` the class objective at its own mode, the
/// covariance step's objective) and `w_k = PMIX_k`, the marginal is `−LSE_k u_k`, so
///
/// ```text
/// ∇²(−log Lᵢ) = Σ_k w_k (∇²F_k − ∇² ln p_k) − Σ_k w_k ∇u_k ∇u_kᵀ + ḡ ḡᵀ,   ḡ = Σ_k w_k ∇u_k.
/// ```
///
/// `∇F_k`, `∇²F_k` are the adjoint's exact profile gradient and Hessian on the class layout
/// (class modes, `MIXNUM = k`), routed onto the mixture layout by [`class_route`];
/// `∇ ln p_k`, `∇² ln p_k` are the mixing programs' exact θ-derivatives, chained through
/// θ's packing. The OFV is `2 Σᵢ`, as the single-population assembly scales it. The modes
/// come from one cold `mixture_ofv` at the covariance inner tolerance, the evaluation the
/// step's base OFV and its score `S` read.
pub(crate) fn mixture_cov_hessian(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x_hat: &[f64],
    options: &FitOptions,
    objective: crate::sens::ode_provider::adjoint::Objective,
) -> Option<DMatrix<f64>> {
    use crate::sens::ode_provider::adjoint::Objective;
    use crate::sens::ode_provider::adjoint::{
        subject_adjoint_cov_hessian_at, subject_adjoint_gradient_at, PackedPoint,
    };
    use rayon::prelude::*;
    // The class objective is FOCE/FOCEI's (`mixture_ofv` scores `foce_subject_nll_*`).
    if !matches!(objective, Objective::Foce | Objective::Focei) {
        return None;
    }
    let spec = model.mixture.as_ref()?;
    let params = unpack_params(x_hat, template);
    let mp = params.mixture.as_ref()?;
    let n = packed_segments(&params).total();
    if x_hat.len() != n {
        return None;
    }
    let score_options = FitOptions {
        inner_tol: options.effective_cov_inner_tol(model.uses_closed_form_ltbs_inner()),
        ..options.clone()
    };
    let (eval, modes) = mixture_ofv_with_modes(model, population, &params, &score_options, None);
    let classes: Vec<(ModelParameters, Vec<f64>, Vec<Option<usize>>)> = (0..mp.omega.len())
        .map(|cls| {
            let cp = class_params(&params, cls);
            let xc = pack_params(&cp);
            (cp, xc, class_route(cls, &params, mp))
        })
        .collect();
    let chain = theta_chain_factors(&params);
    let log_packed: Vec<bool> = params
        .theta_lower
        .iter()
        .map(|&l| theta_packs_log(l))
        .collect();
    let n_theta = params.theta.len();
    let per_subject: Vec<Option<DMatrix<f64>>> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let (dlnp, d2lnp) = crate::parser::model_parser::mixing_logp_hess_exact(
                spec,
                &params.theta,
                &subject.covariates,
            )?;
            let mut h = DMatrix::<f64>::zeros(n, n);
            let mut gbar = DVector::<f64>::zeros(n);
            for (cls, (cp, xc, route)) in classes.iter().enumerate() {
                let w = eval.pmix[i][cls];
                if w == 0.0 {
                    continue;
                }
                let _guard = MixtureClassGuard::enter(cls + 1);
                let pt = PackedPoint::new(cp, xc)?;
                let b = crate::estimation::covariance::stacked_mode(
                    model,
                    subject,
                    &eval.etas_by_class[cls][i],
                    modes.kappas[cls].get(i),
                )?;
                let g_c = subject_adjoint_gradient_at(model, subject, &pt, &b, objective)?;
                let h_c = subject_adjoint_cov_hessian_at(model, subject, &pt, &b, objective)?;
                if g_c.len() != route.len() || h_c.nrows() != route.len() {
                    return None;
                }
                // ∇u_k = ∇ln p_k − ∇F_k on the mixture layout.
                let mut du = DVector::<f64>::zeros(n);
                for (a, da) in route.iter().enumerate() {
                    if let Some(da) = *da {
                        du[da] -= g_c[a];
                        for (c, dc) in route.iter().enumerate() {
                            if let Some(dc) = *dc {
                                h[(da, dc)] += w * h_c[(a, c)];
                            }
                        }
                    }
                }
                for j in 0..n_theta {
                    du[j] += dlnp[cls][j] * chain[j];
                    for l in 0..n_theta {
                        let mut d = d2lnp[cls][(j, l)] * chain[j] * chain[l];
                        if j == l && log_packed[j] {
                            // d(θ_j)/dx_j = θ_j on a log-packed coordinate.
                            d += dlnp[cls][j] * chain[j];
                        }
                        h[(j, l)] -= w * d;
                    }
                }
                h -= w * &du * du.transpose();
                gbar += w * &du;
            }
            h += &gbar * gbar.transpose();
            h.iter().all(|v| v.is_finite()).then_some(h)
        })
        .collect();
    let mut acc = DMatrix::<f64>::zeros(n, n);
    for h in per_subject {
        acc += h?;
    }
    Some(2.0 * acc)
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
/// subjects). Each `∂nllᵢₖ/∂x` is the single-population fixed-mode score
/// (`covariance::fixed_ebe_subject_score`, so the closed-form-vs-FD dispatch and the
/// fixed-mode convention, η̂/H/κ̂ held at `x̂`, are the ones every other S uses; see
/// the convention comment in `covariance::assemble_score_cross_product`) evaluated under
/// `MIXNUM = k`, with class `k`'s own η̂/H/κ̂ and Ω/Σ, then routed onto the mixture
/// layout by `route_class_gradient` — the routing the outer gradient uses. The modes
/// come from one cold `mixture_ofv` at `x̂` and the covariance inner tolerance, the
/// same evaluation `compute_covariance` takes its base OFV from.
pub(crate) struct MixtureScoreContext {
    params: ModelParameters,
    eval: MixtureEval,
    modes: MixtureClassModes,
    classes: Vec<ClassScoreView>,
    dtheta_dx: Vec<f64>,
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
        }
    }

    /// Subject `i`'s score `∂(−log Lᵢ)/∂x` over the full mixture packed layout.
    pub(crate) fn score(
        &self,
        model: &CompiledModel,
        population: &Population,
        i: usize,
        options: &FitOptions,
    ) -> Vec<f64> {
        let spec = model
            .mixture
            .as_ref()
            .expect("mixture score on a non-mixture model");
        let mp = self.params.mixture.as_ref().expect("mixture parameters");
        let covariates = &population.subjects[i].covariates;
        // The p-form has no analytic mixing gradient; S, unlike the outer gradient, has no
        // whole-objective FD to fall back to, so difference the mixing map itself.
        let floor = fd_theta_step_floors(model);
        let dlnp = mixing_logp_grad(spec, &self.params.theta, covariates, &floor)
            .unwrap_or_else(|| mixing_logp_grad_fd(spec, &self.params.theta, covariates, &floor));
        let mut g = vec![0.0f64; packed_segments(&self.params).total()];
        for (cls, view) in self.classes.iter().enumerate() {
            let w = self.eval.pmix[i][cls];
            if w == 0.0 {
                continue;
            }
            // MIXNUM = cls+1 on this (worker) thread for the class's predictions.
            let _guard = MixtureClassGuard::enter(cls + 1);
            // Asymmetry (#1973): the class kernel here is the single-population S's,
            // not `mixture_gradient`'s `per_subject_packed_gradients`; only the routing
            // is shared, so a kernel change moves one of S and the outer gradient.
            let g_c = crate::estimation::covariance::fixed_ebe_subject_score(
                &view.x,
                &view.template,
                model,
                population,
                i,
                &self.eval.etas_by_class[cls][i],
                &self.modes.h_mats[cls][i],
                &self.modes.kappas[cls][i],
                &view.bounds,
                options,
            );
            route_class_gradient(&mut g, &g_c, w, cls, &self.params, mp);
            subtract_mixing_gradient(&mut g, &dlnp[cls], w, &self.dtheta_dx);
        }
        g
    }
}

/// Central-FD outer gradient of the mixture OFV in real packed space — the
/// fallback when [`mixture_gradient`] is out of analytic scope (#977 Phase 4).
/// Re-runs `mixture_ofv` at `x ± hᵢ` per coordinate (`2·n` cold evals).
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
pub fn mixture_gradient_fd(
    model: &CompiledModel,
    population: &Population,
    x: &[f64],
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Vec<f64> {
    let floor = fd_step_floors(init_params, model);
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

    /// #2147: the mixing-map derivative steps θ in its own units. With `TINY = 7.7e-8` in
    /// `sqrt(TINY)`, the legacy step `1e-6·max(|θ|, 1) = 1e-6` puts the minus point at
    /// `−9.2e-7`; the floor `|θ₀|` steps by `1e-6·TINY` instead. `dlnp/dTINY` against the
    /// closed form `(1 − pⱼ)·g′` with `g′ = 1/(2·sqrt(TINY))` (class 1) and `−p₁·g′`
    /// (class 2), for both mixing forms — `p(1) = sqrt(TINY)` routes to the FD map. Each
    /// side asserts the floored step and the straddle that floor 1 is wrong.
    #[test]
    fn mixing_derivative_steps_a_small_theta_relative_to_its_floor() {
        use crate::estimation::parameterization::fd_theta_step_floors;
        use crate::parser::model_parser::{mixing_logp_grad, mixing_logp_grad_fd};
        use std::collections::HashMap;

        let tiny = 7.7e-8_f64;
        let g = tiny.sqrt();
        let dg = 0.5 / g;
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
            let floor = fd_theta_step_floors(&model);
            assert_eq!(floor, vec![1.0, tiny], "{mixing}: floors");
            // The closed form: softmax with logit g, or the p form p₁ = g (both unclamped
            // here, p₁ = 2.8e-4).
            let p1 = if logit { 1.0 / (1.0 + (-g).exp()) } else { g };
            let want = if logit {
                [(1.0 - p1) * dg, -p1 * dg]
            } else {
                [dg / p1, -dg / (1.0 - p1)]
            };
            // Both forms are exact now (`mixing_logp_grad_exact`), so no step and no
            // floor enters: the outer gradient's and the S score's maps are one number.
            let got = mixing_logp_grad(spec, theta, &none, &floor).expect("exact for both forms");
            let got_s = mixing_logp_grad_fd(spec, theta, &none, &[1.0, 1.0]);
            for c in 0..2 {
                assert_eq!(
                    got[c][1].to_bits(),
                    got_s[c][1].to_bits(),
                    "{mixing}: one map"
                );
                let rel = (got[c][1] - want[c]).abs() / want[c].abs();
                // Measured at rounding (the closed form and the tape associate
                // differently); the central differences this replaced were 1e-9 to 1.4e-7
                // off here, and 71 % off at the legacy step.
                assert!(
                    rel < 1e-12,
                    "{mixing} class {}: {} vs {}",
                    c + 1,
                    got[c][1],
                    want[c]
                );
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
    fn iov_mixture_gradient_falls_back_to_fd() {
        // #985: the analytic mixture gradient does not yet emit κ-slot gradients and
        // its override-tail offset would collide with the IOV segment, so an IOV
        // mixture must route to central FD (mixture_gradient returns None). A scope
        // gap here would silently return a wrong-length / mis-slotted gradient.
        let model = crate::parser::model_parser::parse_model_string(MIX_IOV_MODEL).unwrap();
        let pop = read_pop_iov(&bimodal_iov_csv(3, 1.0, 3.0));
        let opts = crate::types::FitOptions::default();
        let eval = super::mixture_ofv(&model, &pop, &model.default_params, &opts, None);
        assert!(
            super::mixture_gradient(&model, &pop, &model.default_params, &opts, &eval).is_none(),
            "IOV + mixture must route to FD (no analytic kappa gradient)"
        );
        // The FD gradient is still well-formed: full packed length, all finite.
        use crate::estimation::parameterization::{pack_params, packed_len};
        let x = pack_params(&model.default_params);
        let fd = super::mixture_gradient_fd(&model, &pop, &x, &model.default_params, &opts);
        assert_eq!(fd.len(), packed_len(&model.default_params));
        assert!(fd.iter().all(|g| g.is_finite()));
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
    /// (`mixture_gradient_fd`); the eval-only entry point, which has no per-class κ̂, still
    /// declines rather than mis-slot.
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
            assert!(super::mixture_gradient(&model, &pop, params, &opts, &eval).is_none());
        }
        eprintln!("IOV mixture gradient vs reconverged FD: worst {worst:.2e}");
        assert!(worst < 1e-4, "{worst:.2e}");
    }

    /// The exact mixture R-matrix (`mixture_cov_hessian`) is the derivative of the exact
    /// mixture gradient: against its Richardson central difference with every class's mode
    /// re-solved at each point, on every mixture fixture and both interaction settings. A
    /// dropped outer-product term, a mis-routed override block or a missing θ-packing
    /// second derivative of `ln p` misses by the size of that term.
    #[test]
    fn the_mixture_cov_hessian_is_the_derivative_of_the_exact_gradient() {
        use crate::estimation::parameterization::{pack_params, unpack_params};
        use crate::sens::ode_provider::adjoint::Objective;
        let mut worst = 0.0f64;
        for (tag, model, pop) in mixture_cases() {
            let params = &model.default_params;
            let x = pack_params(params);
            let n = x.len();
            for interaction in [true, false] {
                let opts = crate::types::FitOptions {
                    interaction,
                    inner_tol: 1e-10,
                    ..crate::types::FitOptions::default()
                };
                let objective = if interaction {
                    Objective::Focei
                } else {
                    Objective::Foce
                };
                let h = super::mixture_cov_hessian(&model, &pop, params, &x, &opts, objective)
                    .unwrap_or_else(|| panic!("{tag}: in scope"));
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
                    assert!(a.is_finite() && b.is_finite(), "{tag}");
                    gap = gap.max((a - b).abs() / (1.0 + b.abs()));
                }
                assert!(
                    h.iter().map(|v| v.abs()).fold(0.0, f64::max) > 1.0,
                    "{tag}: a vanishing Hessian tests nothing"
                );
                eprintln!(
                    "{tag} interaction={interaction}: mixture R vs FD of exact gradient {gap:.2e}"
                );
                worst = worst.max(gap);
            }
        }
        assert!(worst < 1e-5, "{worst:.2e}");
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
        /// `auto`: follows the non-mixture rule (#2739), silently — L-BFGS on the
        /// analytic outer gradient, BOBYQA on finite differences.
        SilentAuto,
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
            Optimizer::Auto => OnMixture::SilentAuto,
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
            // The analytic flag is read only by the `auto` rule, which a mixture now
            // follows too (#2739); neither value may change what an explicit choice
            // runs or whether a warning is owed.
            for analytic in [true, false] {
                let (resolved, warning) =
                    resolve_outer_optimizer(requested, &model, true, analytic);
                match on_mixture(requested) {
                    OnMixture::Honoured => {
                        assert_eq!(resolved, requested, "{requested:?} runs as requested");
                        assert_eq!(warning, None, "{requested:?} is honoured: no warning");
                    }
                    OnMixture::SilentAuto => {
                        // Spelled out rather than read off `resolve_auto_given_analytic`,
                        // so the row does not borrow its answer from the code under test.
                        let want = if analytic {
                            Optimizer::NloptLbfgs
                        } else {
                            Optimizer::Bobyqa
                        };
                        assert_eq!(resolved, want, "auto on a mixture, analytic = {analytic}");
                        assert_eq!(warning, None, "auto resolves silently");
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

    /// Subject `i`'s mixture marginal `−log Σₖ pᵢₖ e^{−nllᵢₖ}` at packed `x`, every
    /// class held at the context's own fixed modes — the function whose gradient the
    /// score is. Built from the objective's pieces directly (`unpack_params`'s override
    /// routing, `class_params`, the per-class FOCE nll `mixture_ofv` takes, the mixing
    /// map), not from anything `MixtureScoreContext::score` calls for its routing,
    /// mixing gradient or class weights.
    fn subject_marginal_at_fixed_modes(
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
        let k = p.mixture.as_ref().unwrap().omega.len();
        let nll: Vec<f64> = (0..k)
            .map(|c| {
                let _guard = super::MixtureClassGuard::enter(c + 1);
                let cp = super::class_params(&p, c);
                let eta = &ctx.eval.etas_by_class[c][i];
                let h = &ctx.modes.h_mats[c][i];
                match cp.omega_iov.as_ref() {
                    Some(iov) => super::foce_subject_nll_iov_cov(
                        model,
                        subject,
                        &cp.theta,
                        eta,
                        h,
                        &cp.omega,
                        &cp.sigma.values,
                        model.residual_cov_at(&cp),
                        interaction,
                        &ctx.modes.kappas[c][i],
                        iov,
                    ),
                    None => super::foce_subject_nll_cov(
                        model,
                        subject,
                        &cp.theta,
                        eta,
                        h,
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
            subject_marginal_at_fixed_modes(&model, pop, template, xv, i, &ctx, interaction)
        };
        let mut worst = 0.0f64;
        for i in 0..pop.subjects.len() {
            let g = ctx.score(&model, pop, i, &opts);
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
        // `p(1)` mixing (no analytic mixing gradient → `mixing_logp_grad_fd`) with IOV:
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
                &eval.mixest_h_mats,
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
            &eval.mixest_h_mats,
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

    /// #2906: the oral two-class mixture of `tests/outer_gradient_fd_fallback_warning.rs`
    /// (lag 0.25), on the first four warfarin subjects. Subject 0's bolus is replaced by a
    /// steady-state infusion into the depot, which the adjoint declines under a lag time
    /// (`ss_lagtime_walk_unsupported`); the other three stay in scope.
    fn mixture_with_one_declined_subject() -> (crate::types::CompiledModel, crate::types::Population)
    {
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
        let log = OuterFdDeclineLog::reconverging(pop.subjects.len());
        let salvaged = super::mixture_gradient_salvaged(
            &model,
            &pop,
            params,
            &opts,
            &eval,
            Some(&modes),
            &log,
        )
        .expect("the mixing map is in scope, so the salvage forms a gradient");
        assert_eq!(log.declined_indices(), vec![0], "only subject 0 declined");

        // The same assembly with subject 0's salvage zeroed: how much its terms carry.
        let zeroed = super::mixture_gradient_assembled(
            &model,
            &pop,
            params,
            &opts,
            &eval,
            Some(&modes),
            |d| Some(vec![0.0; pack_params(d.cp).len()]),
        )
        .unwrap();

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

        let w = outer_fd_fallback_warning(&model, &pop, &log).expect("the decline warns");
        assert!(
            w.contains("1 of 4") && w.contains(&format!("subject {}", pop.subjects[0].id)),
            "{w}"
        );
        assert!(
            w.contains("no analytic recorder serves them") && !w.contains("fixed-EBE"),
            "a structural mixture decline is reconverged, not held-EBE: {w}"
        );
    }
}

/// [`indistinguishable_class_pairs`]: each case changes one thing from an indistinguishable start
/// and must flip the answer, so every input the check reads is shown to matter.
#[cfg(test)]
mod indistinguishable_tests {
    use super::indistinguishable_class_pairs;
    use crate::types::{CompiledModel, ModelParameters, Population};

    /// Whether any pair is indistinguishable: the question every case but the pair-naming ones
    /// asks.
    fn classes_indistinguishable(
        model: &CompiledModel,
        pop: &Population,
        params: &ModelParameters,
    ) -> bool {
        !indistinguishable_class_pairs(model, pop, params).is_empty()
    }

    /// The `tests/mixture_nonmem.rs` model at an indistinguishable start: both classes at
    /// CL 1.7, the mixing logit at 0 (p = ½ each).
    const SYMMETRIC: &str = r"
[parameters]
  theta TVCL1(1.7, 0.01, 100.0)
  theta TVCL2(1.7, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    fn indistinguishable(src: &str) -> bool {
        let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let pop = crate::io::datareader::read_nonmem_csv(
            std::path::Path::new("tests/nonmem/mixture_iv.csv"),
            Some(&["WT"]),
            None,
        )
        .expect("data loads");
        classes_indistinguishable(&model, &pop, &model.default_params)
    }

    fn edit(from: &str, to: &str) -> String {
        assert!(SYMMETRIC.contains(from), "the edit must apply: {from}");
        SYMMETRIC.replace(from, to)
    }

    #[test]
    fn identical_classes_at_equal_probabilities_are_indistinguishable() {
        assert!(indistinguishable(SYMMETRIC));
    }

    #[test]
    fn different_class_parameters_are_distinguishable() {
        assert!(!indistinguishable(&edit(
            "theta TVCL2(1.7, 0.01, 100.0)",
            "theta TVCL2(2.5, 0.01, 100.0)",
        )));
    }

    /// Identical classes at unequal mixing probabilities: the gradient weights the classes by
    /// p₁ and p₂, so it separates them (measured: L-BFGS reaches the optimum from here).
    #[test]
    fn unequal_mixing_probabilities_are_distinguishable() {
        assert!(!indistinguishable(&edit(
            "theta MIXL(0.0, -10.0, 10.0)",
            "theta MIXL(1.0, -10.0, 10.0)",
        )));
    }

    /// Classes that differ only in a per-class Ω: the same predictions, a different prior.
    #[test]
    fn a_class_omega_override_is_distinguishable() {
        assert!(!indistinguishable(&edit(
            "  logit(1) = MIXL\n",
            "  logit(1) = MIXL\n  omega(2) ETA_CL ~ 0.36\n",
        )));
    }

    /// Classes that differ only in how they use η: identical at η = 0, so only the probe η
    /// can tell them apart.
    #[test]
    fn classes_differing_only_through_eta_are_distinguishable() {
        assert!(!indistinguishable(&edit(
            "else TVCL2 * exp(ETA_CL)",
            "else TVCL2 * exp(2 * ETA_CL)",
        )));
    }

    /// The scan reads every subject. `tests/nonmem/mixture_iv.csv` has subjects 1–15 at WT 60
    /// and 16–30 at WT 90, and class 2's clearance is scaled by `WT / ref`, which is exactly
    /// 1.0 for the subjects at `ref`, so the classes are bit-identical on those and apart on the
    /// rest. A check that stops after the first subject calls the first start symmetric, and one
    /// that reads only the last calls the second symmetric. Both must come back `false`, and the
    /// all-symmetric control (`WT / ref` is 1.0 for every subject) must stay `true`.
    #[test]
    fn a_subject_anywhere_in_the_data_that_separates_the_classes_is_found() {
        let pop = crate::io::datareader::read_nonmem_csv(
            std::path::Path::new("tests/nonmem/mixture_iv.csv"),
            Some(&["WT"]),
            None,
        )
        .expect("data loads");
        let wt = |i: usize| pop.subjects[i].covariates["WT"];
        assert_eq!(wt(0), 60.0, "the first subject is at WT 60");
        assert_eq!(
            wt(pop.subjects.len() - 1),
            90.0,
            "the last subject is at WT 90"
        );
        let scaled = |reference: &str| {
            edit(
                "else TVCL2 * exp(ETA_CL)",
                &format!("else TVCL2 * exp(ETA_CL) * (WT / {reference})"),
            )
        };
        let judge = |src: &str| {
            let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
            classes_indistinguishable(&model, &pop, &model.default_params)
        };
        assert!(
            !judge(&scaled("60.0")),
            "subjects 16–30 separate the classes; the symmetric first subject must not decide"
        );
        assert!(
            !judge(&scaled("90.0")),
            "subjects 1–15 separate the classes; the symmetric last subject must not decide"
        );
        assert!(
            judge(&scaled("WT")),
            "WT / WT is 1.0 for every subject: nothing separates the classes"
        );
    }

    /// A three-class start where classes 1 and 2 are identical and class 3 differs, all at
    /// p = ⅓.
    const THREE: &str = r"
[parameters]
  theta TVCL1(1.7, 0.01, 100.0)
  theta TVCL2(1.7, 0.01, 100.0)
  theta TVCL3(3.1, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL1(0.0, -10.0, 10.0)
  theta MIXL2(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[mixture]
  nsub = 3
  logit(1) = MIXL1
  logit(2) = MIXL2

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else if (MIXNUM == 2) TVCL2 * exp(ETA_CL) else TVCL3 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    /// [`THREE`] with class 1 moved off and class 3 onto class 2's values: classes 2 and 3
    /// identical, class 1 different. A check that reports the first pair whatever it found
    /// names classes 1 and 2 here.
    fn three_mirrored() -> String {
        let mirror = THREE
            .replace(
                "theta TVCL1(1.7, 0.01, 100.0)",
                "theta TVCL1(3.1, 0.01, 100.0)",
            )
            .replace(
                "theta TVCL3(3.1, 0.01, 100.0)",
                "theta TVCL3(1.7, 0.01, 100.0)",
            );
        assert!(mirror.contains("TVCL1(3.1") && mirror.contains("TVCL3(1.7"));
        mirror
    }

    /// The indistinguishable pairs of a three-class model on `tests/nonmem/mixture_iv.csv`.
    fn three_class_pairs(src: &str) -> Vec<(usize, usize)> {
        let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
        assert_eq!(model.mixture.as_ref().expect("mixture").n_classes, 3);
        let pop = crate::io::datareader::read_nonmem_csv(
            std::path::Path::new("tests/nonmem/mixture_iv.csv"),
            Some(&["WT"]),
            None,
        )
        .expect("data loads");
        indistinguishable_class_pairs(&model, &pop, &model.default_params)
    }

    /// The identical pair in [`THREE`] is still permutation-symmetric, so a gradient optimizer
    /// keeps the two merged and can only reach the best solution in which they are: the guard
    /// must flag it. Requiring every class to match (the first cut) reported this
    /// distinguishable.
    ///
    /// The control is the same model with all three classes distinct, which must not be flagged,
    /// so the test cannot pass by flagging every three-class start.
    #[test]
    fn two_identical_classes_beside_a_third_are_indistinguishable() {
        let pairs = three_class_pairs;
        assert_eq!(
            pairs(THREE),
            [(1, 2)],
            "classes 1 and 2 are identical at equal p: that pair, and only it, cannot be separated"
        );
        // The third class differing is what the all-classes rule keyed on, so the control has to
        // break the *pair*: class 2 moved off class 1.
        assert_eq!(
            pairs(&THREE.replace(
                "theta TVCL2(1.7, 0.01, 100.0)",
                "theta TVCL2(2.4, 0.01, 100.0)"
            )),
            []
        );
        // Equal classes but unequal mixing probabilities: the pair is weighted differently, so
        // the gradient separates them.
        assert_eq!(
            pairs(&THREE.replace(
                "theta MIXL2(0.0, -10.0, 10.0)",
                "theta MIXL2(1.0, -10.0, 10.0)"
            )),
            []
        );
        assert_eq!(pairs(&three_mirrored()), [(2, 3)]);
        // All three identical: every pair is returned, so the warning names each one the user
        // has to separate, not just the first.
        let all = THREE.replace(
            "theta TVCL3(3.1, 0.01, 100.0)",
            "theta TVCL3(1.7, 0.01, 100.0)",
        );
        assert_eq!(pairs(&all), [(1, 2), (1, 3), (2, 3)]);
    }

    /// Classes that differ only in a per-class Σ: the same predictions and prior, a different
    /// residual variance. The base Σ is what every other case shares between the classes.
    #[test]
    fn a_class_sigma_override_is_distinguishable() {
        assert!(!indistinguishable(&edit(
            "  logit(1) = MIXL\n",
            "  logit(1) = MIXL\n  sigma(2) EPS ~ 0.09\n",
        )));
    }

    /// `W_MIXTURE_SYMMETRIC_START` names the pair the check found, through the same
    /// `symmetric_mixture_start` → `resolve_fit_optimizer` route `optimize_population` takes. The
    /// rule is pairwise, so the message must not say every class matches. Each fixture asserts
    /// the pair it has and the negatives of the pairs it does not.
    #[test]
    fn the_symmetric_start_warning_names_the_pair() {
        use crate::estimation::outer_optimizer::{resolve_fit_optimizer, symmetric_mixture_start};
        use crate::types::Optimizer;
        let warning = |src: &str| {
            let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
            let pop = crate::io::datareader::read_nonmem_csv(
                std::path::Path::new("tests/nonmem/mixture_iv.csv"),
                Some(&["WT"]),
                None,
            )
            .expect("data loads");
            let pairs =
                symmetric_mixture_start(&model, &pop, &model.default_params, Optimizer::Auto);
            let (resolved, warnings) =
                resolve_fit_optimizer(Optimizer::Auto, &model, true, true, &pairs);
            assert_eq!(resolved, Optimizer::Bobyqa, "a symmetric start runs BOBYQA");
            let [w] = warnings.as_slice() else {
                panic!("one warning: {warnings:#?}")
            };
            assert!(w.starts_with("W_MIXTURE_SYMMETRIC_START"), "{w}");
            assert!(!w.contains("every class"), "the rule is pairwise: {w}");
            // The other three sentences: what "indistinguishable" means, what ran instead, and
            // what to change.
            assert!(
                w.contains("the same mixing probability and the same objective under both classes"),
                "{w}"
            );
            assert!(w.contains("ran BOBYQA instead of L-BFGS"), "{w}");
            assert!(w.contains("different initial values"), "{w}");
            w.clone()
        };
        let w = warning(THREE);
        assert!(w.contains("classes 1 and 2"), "{w}");
        assert!(
            !w.contains("classes 1 and 3") && !w.contains("classes 2 and 3"),
            "{w}"
        );
        let w = warning(&three_mirrored());
        assert!(w.contains("classes 2 and 3"), "{w}");
        assert!(
            !w.contains("classes 1 and 2") && !w.contains("classes 1 and 3"),
            "{w}"
        );
        // All three identical: every pair is named, not only the first.
        let all = THREE.replace(
            "theta TVCL3(3.1, 0.01, 100.0)",
            "theta TVCL3(1.7, 0.01, 100.0)",
        );
        let w = warning(&all);
        assert!(
            w.contains("classes 1 and 2, classes 1 and 3, classes 2 and 3 start"),
            "{w}"
        );
    }

    /// The `tests/mixture_iov_nonmem.rs` model (IOV on CL, `p(1)` mixing): `individual_nll`
    /// holds κ at 0, so an indistinguishable start is still caught, and a distinct one is not.
    #[test]
    fn an_iov_mixture_is_judged_too() {
        const IOV: &str = r"
[parameters]
  theta TVCL1(1.8, 0.01, 100.0)
  theta TVCL2(1.8, 0.01, 100.0)
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
        let pop = crate::io::datareader::read_nonmem_csv(
            std::path::Path::new("tests/nonmem/mixture_iv_iov.csv"),
            Some(&["WT"]),
            Some("OCC"),
        )
        .expect("data loads");
        let judge = |src: &str| {
            let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
            assert_eq!(model.n_kappa, 1);
            classes_indistinguishable(&model, &pop, &model.default_params)
        };
        assert!(judge(IOV), "identical classes, p = ½");
        assert!(!judge(&IOV.replace(
            "theta TVCL2(1.8, 0.01, 100.0)",
            "theta TVCL2(3.0, 0.01, 100.0)"
        )));
    }

    #[test]
    fn a_model_without_a_mixture_is_not_flagged() {
        let plain = SYMMETRIC
            .replace("[mixture]\n  nsub = 2\n  logit(1) = MIXL\n", "")
            .replace(
                "CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)",
                "CL = TVCL1 * exp(ETA_CL + 0 * TVCL2 + 0 * MIXL)",
            );
        assert!(!plain.contains("[mixture]"));
        assert!(!indistinguishable(&plain));
    }
}
