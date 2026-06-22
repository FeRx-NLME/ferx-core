//! Exact analytic FOCE/FOCEI covariance Hessian (R-matrix) — the noise-free
//! replacement for the finite-difference covariance step (issue #436).
//!
//! The per-subject FOCEI objective is `Fᵢ = Φ + ½ log|H̃|` with
//!
//! ```text
//!   Φ = ½ Σⱼ (εⱼ²/Rⱼ + ln Rⱼ) + ½ η̂ᵀΩ⁻¹η̂ + ½ ln|Ω|,
//! ```
//!
//! i.e. the inner objective `lᵢ` plus the η-independent `½ ln|Ω|`. The covariance
//! Hessian is the total second derivative of the **profile** objective
//! `F̂ᵢ(x) = Fᵢ(x, η̂(x))` w.r.t. the population parameters, split into
//!
//! * **M2** — the profile Hessian of `Φ`. Because `η̂` minimises `lᵢ ⊆ Φ`, the
//!   envelope theorem gives `∂Φ/∂η|_η̂ = 0`, and the profile Hessian collapses to
//!   the Schur form
//!
//!   ```text
//!     R^M2_{ξζ} = ∂²Φ/∂ξ∂ζ|_η̂  −  M_ξᵀ H⁻¹ M_ζ ,
//!     M_ζ = ∂²Φ/∂η∂ζ ,   H = ∂²Φ/∂η² = h_inner.
//!   ```
//!
//!   Every ingredient is **second order** — `∂²f/∂θ²`, `∂²f/∂η∂θ`, `∂²f/∂η²` from
//!   the provider plus `H⁻¹` — so M2 needs no `Dual3`.
//!
//! * **M3** — the Hessian of the `½ log|H̃|` term, which carries the third-order
//!   curvature (`∂³f/∂η³`, `∂³f/∂η²∂θ`) through the moving mode. Added separately.
//!
//! This module assembles M2 in the natural parameter space (θ, Ω entries, σ); the
//! packed-space chain and M3 are layered on top in later units.
#![allow(clippy::needless_range_loop)]
// WIP (#436): the per-block assemblers are built bottom-up and validated against
// finite differences before being wired into `compute_covariance` in the final
// unit of this PR. Remove this allow once the covariance step consumes them.
#![allow(dead_code)]

use super::sens_outer_gradient::{mixed_eta_theta, Prep};
use crate::sens::provider::SubjectSens;
use crate::types::{CompiledModel, ModelParameters, Subject};
use nalgebra::{DMatrix, DVector};

/// Central-difference half-step for a σ finite difference, keeping the minus side
/// `σ − h` strictly positive near `σ = 0` (mirrors `sens_outer_gradient`'s private
/// `sigma_fd_step`). σ enters Φ only through the closed-form residual variance, so
/// these differences are exact algebra of well-conditioned functions, not AD.
fn sigma_fd_step(sigma_k: f64) -> f64 {
    let h = 1e-6 * (1.0 + sigma_k.abs());
    if sigma_k > 0.0 && h >= sigma_k {
        0.5 * sigma_k
    } else {
        h
    }
}

/// `M_θm = ∂²Φ/∂η∂θ_m` for every θ_m, and the paired `H⁻¹ M_θm`. The mixed term
/// is exactly `mixed_eta_theta` (the inner Hessian's θ-derivative), reused so the
/// θ EBE-response is identical to the gradient's `dη̂/dθ` denominator.
fn theta_m_and_u(
    prep: &Prep,
    sens: &SubjectSens,
    n_theta: usize,
) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
    let n_eta = prep.n_eta;
    let mut mvec = Vec::with_capacity(n_theta);
    let mut uvec = Vec::with_capacity(n_theta);
    for m in 0..n_theta {
        let mm = mixed_eta_theta(&sens.obs, &prep.et, n_eta, prep.n_obs, m);
        let u = &prep.h_inner_inv * &mm;
        mvec.push(mm);
        uvec.push(u);
    }
    (mvec, uvec)
}

/// The **θθ** explicit data-curvature `∂²Φ/∂θ_n∂θ_m|_η̂`, per subject:
/// `Σⱼ [ ½ α'ⱼ bⱼₙ bⱼₘ + ½ αⱼ (∂²f/∂θ²)ⱼ,ₙₘ ]` (the `d2f_dtheta2` provider field
/// supplies the structural curvature). Censored (M3-BLOQ) rows enter uniformly
/// through `½α = ∂L/∂f`, `½α' = ∂²L/∂f²` (set in `prepare`). The full θθ block
/// subtracts the EBE-response coupling `M_θnᵀ H⁻¹ M_θm` in the assembler below.
fn theta_theta_explicit(
    prep: &Prep,
    sens: &SubjectSens,
    n_theta: usize,
    n: usize,
    m: usize,
) -> f64 {
    let mut expl = 0.0;
    for (j, obs) in sens.obs.iter().enumerate() {
        let bn = obs.df_dtheta[n];
        let bm = obs.df_dtheta[m];
        let d2 = obs.d2f_dtheta2[n * n_theta + m];
        expl += 0.5 * (prep.et[j].alpha_p * bn * bm + prep.et[j].alpha * d2);
    }
    expl
}

/// Per-σ finite-difference derivatives of the residual variance and the resulting
/// mode-coupling vectors `M_σk = ∂²Φ/∂η∂σ_k`, all evaluated at the frozen mode.
/// σ enters Φ only through `R(f,σ)`, so each quantity is a difference of the
/// closed-form error functions (`variance_at`, `dvar_df`) — exact algebra.
struct SigmaDerivs {
    /// `M_σk = ½ Σⱼ (∂αⱼ/∂σ_k) aⱼ`, length `n_sigma` of `n_eta`-vectors.
    m_sigma: Vec<DVector<f64>>,
    /// `∂αⱼ/∂σ_k`, `[k][j]`.
    dalpha: Vec<Vec<f64>>,
    /// `R_k = ∂Rⱼ/∂σ_k`, `[k][j]`.
    r1: Vec<Vec<f64>>,
    /// `R_kl = ∂²Rⱼ/∂σ_k∂σ_l`, `[k][l][j]` (symmetric in k,l).
    r2: Vec<Vec<Vec<f64>>>,
}

/// Build [`SigmaDerivs`] for the (non-censored Gaussian) σ block. M3-BLOQ censored
/// rows are out of M2's σ scope and handled when the covariance step gates them.
fn sigma_derivs(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    prep: &Prep,
) -> SigmaDerivs {
    let n_eta = prep.n_eta;
    let sigma = &params.sigma.values;
    let n_sigma = sigma.len();
    let n_obs = prep.n_obs;
    let rv = |sig: &[f64], cmt: usize, f: f64| model.error_spec.variance_at(cmt, f, sig);

    let mut dalpha = vec![vec![0.0; n_obs]; n_sigma];
    let mut r1 = vec![vec![0.0; n_obs]; n_sigma];
    let mut r2 = vec![vec![vec![0.0; n_obs]; n_sigma]; n_sigma];
    let mut m_sigma = vec![DVector::<f64>::zeros(n_eta); n_sigma];

    for k in 0..n_sigma {
        let hk = sigma_fd_step(sigma[k]);
        let mut sp = sigma.clone();
        sp[k] += hk;
        let mut sm = sigma.clone();
        sm[k] -= hk;
        for (j, obs) in sens.obs.iter().enumerate() {
            let cmt = subject.obs_cmts[j];
            let f = obs.f;
            let (r, d, eps) = (prep.et[j].r, prep.et[j].d, prep.et[j].eps);
            let inv_r = 1.0 / r;
            let inv_r2 = inv_r * inv_r;
            let inv_r3 = inv_r2 * inv_r;
            // R_k, d_k = ∂R/∂σ_k, ∂d/∂σ_k by central FD of the closed forms.
            let r_sig = (rv(&sp, cmt, f) - rv(&sm, cmt, f)) / (2.0 * hk);
            let d_sig = (model.error_spec.dvar_df(cmt, f, &sp)
                - model.error_spec.dvar_df(cmt, f, &sm))
                / (2.0 * hk);
            r1[k][j] = r_sig;
            // ∂α/∂σ_k = [2ε/R² + d(2ε²−R)/R³] R_k + [(R−ε²)/R²] d_k.
            let da = (2.0 * eps * inv_r2 + d * (2.0 * eps * eps - r) * inv_r3) * r_sig
                + ((r - eps * eps) * inv_r2) * d_sig;
            dalpha[k][j] = da;
            for m in 0..n_eta {
                m_sigma[k][m] += 0.5 * da * obs.df_deta[m];
            }
            // R_kk = ∂²R/∂σ_k² by the 3-point second difference.
            r2[k][k][j] = (rv(&sp, cmt, f) - 2.0 * r + rv(&sm, cmt, f)) / (hk * hk);
        }
    }
    // Mixed R_kl (k≠l) by the 4-point stencil.
    for k in 0..n_sigma {
        let hk = sigma_fd_step(sigma[k]);
        for l in (k + 1)..n_sigma {
            let hl = sigma_fd_step(sigma[l]);
            let mut spp = sigma.clone();
            spp[k] += hk;
            spp[l] += hl;
            let mut spm = sigma.clone();
            spm[k] += hk;
            spm[l] -= hl;
            let mut smp = sigma.clone();
            smp[k] -= hk;
            smp[l] += hl;
            let mut smm = sigma.clone();
            smm[k] -= hk;
            smm[l] -= hl;
            for (j, obs) in sens.obs.iter().enumerate() {
                let cmt = subject.obs_cmts[j];
                let f = obs.f;
                let val = (rv(&spp, cmt, f) - rv(&spm, cmt, f) - rv(&smp, cmt, f)
                    + rv(&smm, cmt, f))
                    / (4.0 * hk * hl);
                r2[k][l][j] = val;
                r2[l][k][j] = val;
            }
        }
    }
    SigmaDerivs {
        m_sigma,
        dalpha,
        r1,
        r2,
    }
}

/// Free Ω entries in the optimizer's pack order: diagonal `(i,i)`; block lower
/// triangle `(r,c)` with `c ≤ r`. The natural parameter for an off-diagonal entry
/// is the single scalar setting both `Ω[r,c]` and `Ω[c,r]` (symmetric).
pub(crate) fn omega_entries(diagonal: bool, n_eta: usize) -> Vec<(usize, usize)> {
    if diagonal {
        (0..n_eta).map(|i| (i, i)).collect()
    } else {
        let mut e = Vec::new();
        for c in 0..n_eta {
            for r in c..n_eta {
                e.push((r, c));
            }
        }
        e
    }
}

/// `E_{rc}`: the symmetric single-entry derivative matrix `∂Ω/∂Ω_{rc}` — a lone 1
/// at `(r,r)` for a diagonal entry, or 1s at `(r,c)` and `(c,r)` for the symmetric
/// off-diagonal parameter.
fn e_matrix(r: usize, c: usize, n: usize) -> DMatrix<f64> {
    let mut e = DMatrix::zeros(n, n);
    e[(r, c)] = 1.0;
    e[(c, r)] = 1.0;
    e
}

/// The full M2 covariance Hessian over the natural `[θ, Ω, σ]` parameters, per
/// subject. Each entry is `∂²Φ/∂ξ∂ζ|_η̂ − M_ξᵀ H⁻¹ M_ζ` with `M_ζ = ∂²Φ/∂η∂ζ`:
///
/// * `θθ` explicit `Σⱼ ½(α' bb + α ∂²f/∂θ²)`, `M_θ = mixed_eta_theta`;
/// * `σσ`/`θσ` explicit error-variance curvature, `M_σ = ½Σ(∂α/∂σ)a`;
/// * `ΩΩ` explicit `zᵀE_ξΩ⁻¹E_ζz − ½tr(Ω⁻¹E_ξΩ⁻¹E_ζ)`, `M_Ω = −Ω⁻¹E z`,
///   `z = Ω⁻¹η̂`;
/// * `θΩ`/`Ωσ` explicit partials vanish (only the mode coupling survives).
///
/// `eta_hat` is the subject's EBE for `params`. This is the natural-space block;
/// the packed-space chain and the `½log|H̃|` (M3) curvature are layered on later.
pub(crate) fn subject_cov_hessian_m2_natural(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    prep: &Prep,
    eta_hat: &[f64],
) -> DMatrix<f64> {
    let n_eta = prep.n_eta;
    let n_theta = params.theta.len();
    let entries = omega_entries(params.omega.diagonal, n_eta);
    let n_omega = entries.len();
    let n_sigma = params.sigma.values.len();
    let nt = n_theta;
    let nw = nt + n_omega; // σ offset
    let dim = n_theta + n_omega + n_sigma;

    let omega_inv = &prep.omega_inv;
    let z = omega_inv * DVector::from_column_slice(eta_hat);
    let e_mats: Vec<DMatrix<f64>> = entries
        .iter()
        .map(|&(r, c)| e_matrix(r, c, n_eta))
        .collect();

    // Mode-coupling vectors M_ζ = ∂²Φ/∂η∂ζ for every natural parameter, in order.
    let (m_theta, _) = theta_m_and_u(prep, sens, n_theta);
    let sd = sigma_derivs(model, subject, params, sens, prep);
    let m_omega: Vec<DVector<f64>> = e_mats.iter().map(|e| -(omega_inv * (e * &z))).collect();

    let mut mall: Vec<DVector<f64>> = Vec::with_capacity(dim);
    mall.extend(m_theta);
    mall.extend(m_omega);
    mall.extend(sd.m_sigma.iter().cloned());
    let uall: Vec<DVector<f64>> = mall.iter().map(|m| &prep.h_inner_inv * m).collect();

    // Explicit cross-partial ∂²Φ/∂ξ∂ζ|_η̂ between natural params a (ζ) and b (ξ).
    let explicit = |a: usize, b: usize| -> f64 {
        let (lo, hi) = (a.min(b), a.max(b));
        if hi < nt {
            // θθ.
            theta_theta_explicit(prep, sens, n_theta, a, b)
        } else if lo < nt && hi >= nt && hi < nw {
            // θΩ — explicit vanishes.
            0.0
        } else if lo < nt && hi >= nw {
            // θσ: Σⱼ ½ (∂α/∂σ_k) bⱼₘ.
            let m = lo;
            let k = hi - nw;
            let mut s = 0.0;
            for (j, obs) in sens.obs.iter().enumerate() {
                s += 0.5 * sd.dalpha[k][j] * obs.df_dtheta[m];
            }
            s
        } else if lo >= nt && hi < nw {
            // ΩΩ: zᵀE_b Ω⁻¹ E_a z − ½ tr(Ω⁻¹ E_b Ω⁻¹ E_a).
            let ea = &e_mats[a - nt];
            let eb = &e_mats[b - nt];
            let quad = (eb * &z).dot(&(omega_inv * (ea * &z)));
            let oeb = omega_inv * eb;
            let oea = omega_inv * ea;
            let tr = (&oeb * &oea).trace();
            quad - 0.5 * tr
        } else if lo >= nt && lo < nw && hi >= nw {
            // Ωσ — explicit vanishes.
            0.0
        } else {
            // σσ: Σⱼ ½[(−1/R²+2ε²/R³)R_l R_k + (1/R−ε²/R²)R_kl].
            let k = lo - nw;
            let l = hi - nw;
            let mut s = 0.0;
            for j in 0..prep.n_obs {
                let (r, eps) = (prep.et[j].r, prep.et[j].eps);
                let inv_r = 1.0 / r;
                let inv_r2 = inv_r * inv_r;
                let inv_r3 = inv_r2 * inv_r;
                let a_term = (-inv_r2 + 2.0 * eps * eps * inv_r3) * sd.r1[l][j] * sd.r1[k][j];
                let b_term = (inv_r - eps * eps * inv_r2) * sd.r2[k][l][j];
                s += 0.5 * (a_term + b_term);
            }
            s
        }
    };

    let mut h = DMatrix::zeros(dim, dim);
    for a in 0..dim {
        for b in 0..dim {
            h[(a, b)] = explicit(a, b) - mall[a].dot(&uall[b]);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::inner_optimizer::find_ebe;
    use crate::estimation::sens_outer_gradient::prepare;
    use crate::parser::model_parser::parse_model_string;
    use crate::sens::provider::{subject_sensitivities, subject_sensitivities_cov};
    use crate::types::{CompiledModel, DoseEvent, ModelParameters, OmegaMatrix, Subject};
    use std::collections::HashMap;

    const WARFARIN: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    fn warfarin_subject(model: &CompiledModel, theta: &[f64], times: &[f64]) -> Subject {
        let n = times.len();
        let mut subject = Subject {
            id: "1".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let eta_ref = [0.12, -0.08, 0.2];
        let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, &eta_ref);
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        subject
    }

    /// Analytic-Newton EBE on the inner objective, warm-started from `find_ebe`,
    /// so the reconverged-FD reference is free of inner-solver reconvergence noise.
    fn precise_ebe(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> Vec<f64> {
        let warm = find_ebe(model, subject, params, 80, 1e-10, None, None);
        let mut eta: Vec<f64> = warm.eta.iter().copied().collect();
        let n_eta = model.n_eta;
        let sigma = &params.sigma.values;
        let omega_inv = &params.omega.inv;
        for _ in 0..60 {
            let sens = subject_sensitivities(model, subject, &params.theta, &eta).unwrap();
            let mut grad = omega_inv * DVector::from_column_slice(&eta);
            let mut hess = omega_inv.clone();
            for (j, obs) in sens.obs.iter().enumerate() {
                let f = obs.f;
                let cmt = subject.obs_cmts[j];
                let r = model.error_spec.variance_at(cmt, f, sigma);
                let d = model.error_spec.dvar_df(cmt, f, sigma);
                let d2 = model.error_spec.d2var_df2(cmt, sigma);
                let eps = subject.observations[j] - f;
                // inner gradient ½α·a, true Hessian ½(α' a aᵀ + α A).
                let inv_r = 1.0 / r;
                let inv_r2 = inv_r * inv_r;
                let inv_r3 = inv_r2 * inv_r;
                let alpha = -2.0 * eps * inv_r + d * (r - eps * eps) * inv_r2;
                let alpha_p = 2.0 * inv_r
                    + 2.0 * eps * d * inv_r2
                    + (d2 * (r - eps * eps) + d * d + 2.0 * d * eps) * inv_r2
                    - 2.0 * d * d * (r - eps * eps) * inv_r3;
                for k in 0..n_eta {
                    grad[k] += 0.5 * alpha * obs.df_deta[k];
                    for l in 0..n_eta {
                        hess[(k, l)] += 0.5
                            * (alpha_p * obs.df_deta[k] * obs.df_deta[l]
                                + alpha * obs.d2f_deta2[k * n_eta + l]);
                    }
                }
            }
            let step = hess.clone().cholesky().unwrap().solve(&grad);
            for k in 0..n_eta {
                eta[k] -= step[k];
            }
            if step.norm() < 1e-13 {
                break;
            }
        }
        eta
    }

    /// Φ natural gradient `[∂Φ/∂θ, ∂Φ/∂σ]` at the reconverged mode for `params`
    /// (the M2-relevant data part of the analytic gradient — no `log|H̃|`):
    /// `∂Φ/∂θ_m = ½ Σⱼ αⱼ bⱼₘ`, `∂Φ/∂Ω_e = ½(−zᵀE z + tr(Ω⁻¹E))`,
    /// `∂Φ/∂σ_k = ½ Σⱼ (1/R − ε²/R²) R_k`, ordered `[θ, Ω, σ]` — the M2-relevant
    /// (no `log|H̃|`) part of the analytic gradient, evaluated at the reconverged
    /// mode.
    fn phi_natural_grad(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
    ) -> Vec<f64> {
        let eta = precise_ebe(model, subject, params);
        let sens = subject_sensitivities(model, subject, &params.theta, &eta).unwrap();
        let prep = prepare(model, subject, params, &sens).unwrap();
        let n_eta = model.n_eta;
        let n_theta = params.theta.len();
        let entries = omega_entries(params.omega.diagonal, n_eta);
        let n_omega = entries.len();
        let sigma = &params.sigma.values;
        let n_sigma = sigma.len();
        let nw = n_theta + n_omega;
        let mut g = vec![0.0; n_theta + n_omega + n_sigma];
        for m in 0..n_theta {
            for (j, obs) in sens.obs.iter().enumerate() {
                g[m] += 0.5 * prep.et[j].alpha * obs.df_dtheta[m];
            }
        }
        let omega_inv = &prep.omega_inv;
        let z = omega_inv * DVector::from_column_slice(&eta);
        for (e, &(r, c)) in entries.iter().enumerate() {
            let em = e_matrix(r, c, n_eta);
            let quad = z.dot(&(&em * &z));
            let tr = (omega_inv * &em).trace();
            g[n_theta + e] = 0.5 * (-quad + tr);
        }
        for k in 0..n_sigma {
            let hk = sigma_fd_step(sigma[k]);
            let mut sp = sigma.clone();
            sp[k] += hk;
            let mut sm = sigma.clone();
            sm[k] -= hk;
            for (j, obs) in sens.obs.iter().enumerate() {
                let cmt = subject.obs_cmts[j];
                let f = obs.f;
                let (r, eps) = (prep.et[j].r, prep.et[j].eps);
                let r_k = (model.error_spec.variance_at(cmt, f, &sp)
                    - model.error_spec.variance_at(cmt, f, &sm))
                    / (2.0 * hk);
                g[nw + k] += 0.5 * (1.0 / r - eps * eps / (r * r)) * r_k;
            }
        }
        g
    }

    /// Perturb the natural parameter at flat index `p` (ordered `[θ, Ω, σ]`) by
    /// `step`, rebuilding Ω (and its cached inverse) for an Ω entry.
    fn perturb_natural(
        base: &ModelParameters,
        n_theta: usize,
        entries: &[(usize, usize)],
        p: usize,
        step: f64,
    ) -> ModelParameters {
        let mut q = base.clone();
        let n_omega = entries.len();
        if p < n_theta {
            q.theta[p] += step;
        } else if p < n_theta + n_omega {
            let (r, c) = entries[p - n_theta];
            let mut m = base.omega.matrix.clone();
            m[(r, c)] += step;
            if r != c {
                m[(c, r)] += step;
            }
            q.omega =
                OmegaMatrix::from_matrix(m, base.omega.eta_names.clone(), base.omega.diagonal);
        } else {
            let mut s = q.sigma.values.clone();
            s[p - n_theta - n_omega] += step;
            q.sigma.values = s;
        }
        q
    }

    /// Validate the full natural `[θ, Ω, σ]` M2 block against a reconverged
    /// precise-EBE finite difference of the Φ natural gradient.
    fn check_m2_natural(model: &CompiledModel, subject: &Subject, params: &ModelParameters) {
        let eta = precise_ebe(model, subject, params);
        let sens = subject_sensitivities_cov(model, subject, &params.theta, &eta).unwrap();
        let prep = prepare(model, subject, params, &sens).unwrap();
        let analytic = subject_cov_hessian_m2_natural(model, subject, params, &sens, &prep, &eta);

        let n_theta = params.theta.len();
        let entries = omega_entries(params.omega.diagonal, model.n_eta);
        let n_omega = entries.len();
        let n_sigma = params.sigma.values.len();
        let dim = n_theta + n_omega + n_sigma;

        let base_val = |p: usize| -> f64 {
            if p < n_theta {
                params.theta[p]
            } else if p < n_theta + n_omega {
                let (r, c) = entries[p - n_theta];
                params.omega.matrix[(r, c)]
            } else {
                params.sigma.values[p - n_theta - n_omega]
            }
        };

        let mut fd = DMatrix::zeros(dim, dim);
        for col in 0..dim {
            let h = 1e-6 * (1.0 + base_val(col).abs());
            let gp = phi_natural_grad(
                model,
                subject,
                &perturb_natural(params, n_theta, &entries, col, h),
            );
            let gm = phi_natural_grad(
                model,
                subject,
                &perturb_natural(params, n_theta, &entries, col, -h),
            );
            for row in 0..dim {
                fd[(row, col)] = (gp[row] - gm[row]) / (2.0 * h);
            }
        }

        for row in 0..dim {
            for col in 0..dim {
                let a = analytic[(row, col)];
                let f = fd[(row, col)];
                let tol = 1e-4 * (1.0 + a.abs());
                assert!(
                    (a - f).abs() < tol,
                    "M2[{},{}]: analytic {:.8e} vs FD {:.8e} (Δ {:.2e})",
                    row,
                    col,
                    a,
                    f,
                    (a - f).abs()
                );
            }
        }
        for row in 0..dim {
            for col in 0..dim {
                assert!((analytic[(row, col)] - analytic[(col, row)]).abs() < 1e-9);
            }
        }
    }

    /// Warfarin (1-cpt oral, diagonal Ω, proportional error): the natural M2
    /// Hessian matches reconverged FD across θθ, θΩ, θσ, ΩΩ, Ωσ, σσ.
    #[test]
    fn cov_hessian_m2_natural_matches_reconverged_fd_diagonal() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.2, 10.0, 1.5];
        let subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        let mut params = model.default_params.clone();
        params.theta = theta;
        check_m2_natural(&model, &subject, &params);
    }

    /// Block-Ω (correlated CL/V): exercises the off-diagonal ΩΩ curvature and the
    /// cross-entry Ω couplings.
    #[test]
    fn cov_hessian_m2_natural_matches_reconverged_fd_block_omega() {
        const BLOCK: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let model = parse_model_string(BLOCK).expect("parse");
        let theta = vec![0.2, 10.0, 1.5];
        let subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        let mut params = model.default_params.clone();
        params.theta = theta;
        check_m2_natural(&model, &subject, &params);
    }
}
