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
//! under a [`MixtureClassGuard`]. The per-class loop is **serial** on purpose: a
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
use crate::stats::likelihood::foce_subject_nll;
use crate::types::{CompiledModel, FitOptions, ModelParameters, Population};

/// Result of one mixture-objective evaluation.
pub struct MixtureEval {
    /// Population objective, `−2 Σ_i log Σ_k p_ik exp(−nll_ik)`.
    pub ofv: f64,
    /// MIXEST-class EBE per subject (for warm-start carry / postfit).
    pub mixest_etas: Vec<DVector<f64>>,
    /// MIXEST-class inner Hessian per subject.
    pub mixest_h_mats: Vec<DMatrix<f64>>,
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
    let mut converged: Vec<Vec<bool>> = vec![vec![true; n]; k];
    let mut fallback: Vec<Vec<bool>> = vec![vec![false; n]; k];
    let mut hard_reject: Vec<Vec<bool>> = vec![vec![false; n]; k];

    for cls in 0..k {
        // MIXNUM = cls+1 for the whole class solve (serial → thread-local safe).
        let _guard = MixtureClassGuard::enter(cls + 1);
        let cp = class_params(params, cls);
        let mut etas_c = Vec::with_capacity(n);
        let mut hmats_c = Vec::with_capacity(n);
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
            nll[cls][i] = foce_subject_nll(
                model,
                subject,
                &cp.theta,
                &ebe.eta,
                &ebe.h_matrix,
                &cp.omega,
                &cp.sigma.values,
                interaction,
            );
            converged[cls][i] = ebe.converged;
            fallback[cls][i] = ebe.used_fallback;
            hard_reject[cls][i] = ebe.hard_reject;
            etas_c.push(ebe.eta);
            hmats_c.push(ebe.h_matrix);
        }
        etas_by_class.push(etas_c);
        hmats_by_class.push(hmats_c);
    }

    // Combine per subject via log-sum-exp; pick MIXEST for warm-start / postfit.
    let mut ofv = 0.0;
    let mut pmix = Vec::with_capacity(n);
    let mut mixest = Vec::with_capacity(n);
    let mut mixest_etas = Vec::with_capacity(n);
    let mut mixest_h_mats = Vec::with_capacity(n);
    let (mut n_unconverged, mut n_fallback, mut n_start_rejected) = (0usize, 0usize, 0usize);

    for (i, subject) in population.subjects.iter().enumerate() {
        let logp = eval_mixing_log_probs(spec, &params.theta, &subject.covariates);
        let nll_i: Vec<f64> = (0..k).map(|cls| nll[cls][i]).collect();
        let (contrib, probs, best) = combine_subject(&logp, &nll_i);
        ofv += contrib;
        if !converged[best][i] {
            n_unconverged += 1;
        }
        if fallback[best][i] {
            n_fallback += 1;
        }
        if hard_reject[best][i] {
            n_start_rejected += 1;
        }
        mixest_etas.push(etas_by_class[best][i].clone());
        mixest_h_mats.push(hmats_by_class[best][i].clone());
        pmix.push(probs);
        mixest.push(best);
    }

    MixtureEval {
        ofv,
        mixest_etas,
        mixest_h_mats,
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
/// (direct-probability) mixing form, or any subject/class the per-subject
/// sensitivity provider declines.
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
    let k = mp.omega.len();
    let interaction = options.interaction;

    let nt = params.theta.len();
    let n_omega = omega_packed_len(params.omega.dim(), params.omega.diagonal);
    let n_sigma = params.sigma.values.len();
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
        let _guard = MixtureClassGuard::enter(cls + 1);
        let cp = class_params(params, cls);
        let x_k = pack_params(&cp);
        // ∂nll_ik/∂x_k over the class-k packed layout [θ, Ω(base slots), σ, …].
        let g_k = per_subject_packed_gradients(
            model,
            population,
            &cp,
            &x_k,
            &eval.etas_by_class[cls],
            interaction,
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
    use super::combine_subject;

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
