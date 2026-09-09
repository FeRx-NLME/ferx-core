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
//! ([`foce_subject_nll`]) evaluated with class-`k` parameters and the class-`k`
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

use crate::estimation::inner_optimizer::{find_ebe, InnerLoopStats};
use crate::estimation::parameterization::{omega_packed_len, pack_params, theta_packs_log};
use crate::estimation::sens_outer_gradient::per_subject_packed_gradients;
use crate::parser::model_parser::{eval_mixing_log_probs, mixing_logp_grad, MixtureClassGuard};
use crate::stats::likelihood::{foce_subject_nll, foce_subject_nll_iov};
use crate::types::{CompiledModel, FitOptions, ModelParameters, Population};

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
    let mut hard_reject: Vec<Vec<bool>> = vec![vec![false; n]; k];

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
            let ebe = find_ebe(
                model,
                subject,
                &cp,
                options.inner_maxiter,
                options.inner_tol,
                warm_i,
                None,
                options.inner_restarts,
            );
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
                foce_subject_nll_iov(
                    model,
                    subject,
                    &cp.theta,
                    &ebe.eta,
                    &ebe.h_matrix,
                    &cp.omega,
                    &cp.sigma.values,
                    interaction,
                    &ebe.kappas,
                    omega_iov,
                )
            } else {
                foce_subject_nll(
                    model,
                    subject,
                    &cp.theta,
                    &ebe.eta,
                    &ebe.h_matrix,
                    &cp.omega,
                    &cp.sigma.values,
                    &cp.residual_correlations,
                    interaction,
                )
            };
            converged[cls][i] = ebe.converged;
            fallback[cls][i] = ebe.used_fallback;
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
    let mut mixest = Vec::with_capacity(n);
    let mut mixest_etas = Vec::with_capacity(n);
    let mut mixest_h_mats = Vec::with_capacity(n);
    let mut mixest_kappas = Vec::with_capacity(n);
    let (mut n_unconverged, mut n_fallback, mut n_start_rejected) = (0usize, 0usize, 0usize);

    for (i, subject) in population.subjects.iter().enumerate() {
        let logp = eval_mixing_log_probs(spec, &params.theta, &subject.covariates);
        let nll_i: Vec<f64> = (0..k).map(|cls| nll[cls][i]).collect();
        let (contrib, probs, best) = combine_subject(&logp, &nll_i);
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
        let reject_i: Vec<bool> = (0..k).map(|cls| hard_reject[cls][i]).collect();
        if any_contributing(&unconverged_i, &probs) {
            n_unconverged += 1;
        }
        if any_contributing(&fallback_i, &probs) {
            n_fallback += 1;
        }
        if any_contributing(&reject_i, &probs) {
            n_start_rejected += 1;
        }
        mixest_etas.push(etas_by_class[best][i].clone());
        mixest_h_mats.push(hmats_by_class[best][i].clone());
        mixest_kappas.push(kappas_by_class[best][i].clone());
        pmix.push(probs);
        mixest.push(best);
    }

    MixtureEval {
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
            n_start_rejected,
        },
    }
}

/// Analytic outer gradient of the mixture OFV in **real packed space** (#977
/// Phase 4): `∂OFV/∂x = 2 Σ_i Σ_k w_ik (∂nll_ik/∂x − ∂ln p_ik/∂x)`, where
/// `w_ik = PMIX_ik` (from `eval`), `∂nll_ik/∂x` is the per-class per-subject
/// analytic FOCE/FOCEI packed gradient, and `∂ln p_ik/∂θ` is the softmax mixing
/// gradient (theta slots only). The un-scaled gradient the objective closure
/// then multiplies by `scale[k]` for optimizer space.
///
/// Returns `None` (→ the caller falls back to FD of the mixture OFV) when the
/// gradient is out of analytic scope: a non-diagonal base Omega, the `p`
/// (direct-probability) mixing form, inter-occasion variability (κ), or any
/// subject/class the per-subject sensitivity provider declines.
pub fn mixture_gradient(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    eval: &MixtureEval,
) -> Option<Vec<f64>> {
    let spec = model.mixture.as_ref()?;
    let mp = params.mixture.as_ref()?;
    // Analytic per-class Ω-slot mapping assumes a diagonal base Omega (also the
    // rule for `omega(k)` overrides). Off-diagonal → FD fallback.
    if !params.omega.diagonal {
        return None;
    }
    // Inter-occasion variability (#985): the packed vector carries a κ (`omega_iov`)
    // segment between σ and the per-class override tail, and the per-class
    // sensitivity provider does not yet emit κ-slot gradients for a mixture. Rather
    // than assemble a partial gradient with wrong override-slot offsets, route the
    // whole IOV-mixture objective to central FD (`mixture_gradient_fd`, which packs
    // through the authoritative `pack_params` layout). Returning here means every
    // path below is non-IOV, so the κ segment is empty and `ov_base` needs no κ term.
    if params.omega_iov.is_some() {
        return None;
    }
    let k = mp.omega.len();
    let interaction = options.interaction;

    let nt = params.theta.len();
    let n_omega = omega_packed_len(params.omega.dim(), params.omega.diagonal);
    let n_sigma = params.sigma.values.len();
    // Mirror `pack_params`: the per-class override tail follows [θ, Ω, σ]. There is
    // no κ segment on this path (the `omega_iov` early-return above guarantees it).
    let ov_base = nt + n_omega + n_sigma;
    let n_omega_ov = mp.omega_override_addr.len();
    let total = ov_base + n_omega_ov + mp.sigma_override_addr.len();

    // θ → packed chain factor dθ_j/dx_j (θ_j for log-packed, 1 for identity).
    let dtheta_dx: Vec<f64> = (0..nt)
        .map(|j| {
            if theta_packs_log(params.theta_lower[j]) {
                params.theta[j]
            } else {
                1.0
            }
        })
        .collect();

    let mut grad = vec![0.0f64; total];

    for cls in 0..k {
        let cp = class_params(params, cls);
        let x_k = pack_params(&cp);
        // ∂nll_ik/∂x_k over the class-k packed layout [θ, Ω(base slots), σ, …].
        // The class thread-local is set inside each rayon worker (the outer-thread
        // guard would not reach them), so pass `Some(cls + 1)` rather than relying
        // on a `MixtureClassGuard` on this thread.
        let g_k = per_subject_packed_gradients(
            model,
            population,
            &cp,
            &x_k,
            &eval.etas_by_class[cls],
            interaction,
            Some(cls + 1),
        );
        for (i, gk_i) in g_k.iter().enumerate() {
            let gi = gk_i.as_ref()?; // out-of-scope subject → FD fallback
            let w = eval.pmix[i][cls];
            if w == 0.0 {
                continue;
            }
            // θ slots (shared across classes).
            for j in 0..nt {
                grad[j] += 2.0 * w * gi[j];
            }
            // Base-Ω diagonal slots: route to the override slot for classes that
            // override eta e, else to the shared base slot.
            for e in 0..params.omega.dim() {
                let s = nt + e;
                let dest = mp
                    .omega_override_addr
                    .iter()
                    .position(|&(c, ee)| c == cls && ee == e)
                    .map(|a| ov_base + a)
                    .unwrap_or(s);
                grad[dest] += 2.0 * w * gi[s];
            }
            // σ slots.
            for t in 0..n_sigma {
                let s = nt + n_omega + t;
                let dest = mp
                    .sigma_override_addr
                    .iter()
                    .position(|&(c, tt)| c == cls && tt == t)
                    .map(|b| ov_base + n_omega_ov + b)
                    .unwrap_or(s);
                grad[dest] += 2.0 * w * gi[s];
            }
        }
    }

    // Softmax mixing gradient: −2 Σ_i Σ_k w_ik ∂ln p_ik/∂θ_j (theta slots).
    for (i, subject) in population.subjects.iter().enumerate() {
        let dlnp = mixing_logp_grad(spec, &params.theta, &subject.covariates)?;
        for (cls, dlnp_c) in dlnp.iter().enumerate() {
            let w = eval.pmix[i][cls];
            if w == 0.0 {
                continue;
            }
            for j in 0..nt {
                grad[j] -= 2.0 * w * dlnp_c[j] * dtheta_dx[j];
            }
        }
    }

    Some(grad)
}

/// Central-FD outer gradient of the mixture OFV in real packed space — the
/// fallback when [`mixture_gradient`] is out of analytic scope (#977 Phase 4).
/// Re-runs `mixture_ofv` at `x ± h` per coordinate (`2·n` cold evals).
pub fn mixture_gradient_fd(
    model: &CompiledModel,
    population: &Population,
    x: &[f64],
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Vec<f64> {
    let h = 1e-5;
    let n = x.len();
    let mut g = vec![0.0f64; n];
    for i in 0..n {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let pp = crate::estimation::parameterization::unpack_params(&xp, init_params);
        let pm = crate::estimation::parameterization::unpack_params(&xm, init_params);
        let fp = mixture_ofv(model, population, &pp, options, None).ofv;
        let fm = mixture_ofv(model, population, &pm, options, None).ofv;
        g[i] = (fp - fm) / (2.0 * h);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::{any_contributing, combine_subject};

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
    fn mixture_gradient_pform_falls_back_to_fd() {
        // #983 item 4: the `p(k)` form has no analytic mixing-probability gradient
        // (only the softmax `logit(k)` form is differentiated), so
        // `mixture_gradient` returns None and the caller routes to central FD. A
        // scope gap here would silently return a wrong analytic gradient.
        let m = mix_form_model("theta P1(0.4, 0.001, 0.999)", "p(1) = P1");
        let model = crate::parser::model_parser::parse_model_string(&m).unwrap();
        let pop = read_pop(&bimodal_csv(4, 1.0, 3.0));
        let opts = crate::types::FitOptions::default();
        let eval = super::mixture_ofv(&model, &pop, &model.default_params, &opts, None);
        assert!(
            super::mixture_gradient(&model, &pop, &model.default_params, &opts, &eval).is_none(),
            "p(k) mixing form must route to FD (no analytic mixing gradient)"
        );
    }

    // Variance-mixture (classes differ only by their Ω/Σ overrides, no `MIXNUM`
    // branch) so the per-subject analytic sensitivity provider is in scope — the
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
                .any(|w| w.contains("BOBYQA") && w.contains("Bfgs")),
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
}
