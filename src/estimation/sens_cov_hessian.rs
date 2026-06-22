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
use nalgebra::{DMatrix, DVector};

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

/// The **θθ** block of the M2 covariance Hessian in natural θ space, per subject:
///
/// ```text
///   R^M2_{θn,θm} = Σⱼ [ ½ α'ⱼ bⱼₙ bⱼₘ + ½ αⱼ (∂²f/∂θ²)ⱼ,ₙₘ ]  −  M_θnᵀ H⁻¹ M_θm.
/// ```
///
/// The explicit cross-partial is `∂²Φ/∂θ_n∂θ_m|_η̂` (the data term's θ-curvature,
/// which the new provider field `d2f_dtheta2` supplies); the coupling is the M2
/// EBE response `Σ_l (∂²Φ/∂θ_m∂η_l) dη̂_l/dθ_n`. Censored (M3-BLOQ) rows enter
/// uniformly through `½α = ∂L/∂f`, `½α' = ∂²L/∂f²` (set in `prepare`).
pub(crate) fn cov_hessian_m2_theta(
    prep: &Prep,
    sens: &SubjectSens,
    n_theta: usize,
) -> DMatrix<f64> {
    let (mvec, uvec) = theta_m_and_u(prep, sens, n_theta);
    let mut h = DMatrix::zeros(n_theta, n_theta);
    for n in 0..n_theta {
        for m in 0..n_theta {
            let mut expl = 0.0;
            for (j, obs) in sens.obs.iter().enumerate() {
                let bn = obs.df_dtheta[n];
                let bm = obs.df_dtheta[m];
                let d2 = obs.d2f_dtheta2[n * n_theta + m];
                expl += 0.5 * (prep.et[j].alpha_p * bn * bm + prep.et[j].alpha * d2);
            }
            h[(n, m)] = expl - mvec[n].dot(&uvec[m]);
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
    use crate::types::{CompiledModel, DoseEvent, ModelParameters, Subject};
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

    /// Φ θ-gradient `∂Φ/∂θ_m = ½ Σⱼ αⱼ bⱼₘ` at the reconverged mode for `params`
    /// (the M2-relevant data part of the analytic θ-gradient — no `log|H̃|`).
    fn phi_theta_grad(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
    ) -> Vec<f64> {
        let eta = precise_ebe(model, subject, params);
        let sens = subject_sensitivities(model, subject, &params.theta, &eta).unwrap();
        let prep = prepare(model, subject, params, &sens).unwrap();
        let n_theta = params.theta.len();
        let mut g = vec![0.0; n_theta];
        for m in 0..n_theta {
            for (j, obs) in sens.obs.iter().enumerate() {
                g[m] += 0.5 * prep.et[j].alpha * obs.df_dtheta[m];
            }
        }
        g
    }

    /// The θθ M2 block equals the reconverged finite difference of the Φ
    /// θ-gradient (its total derivative through the moving mode) on warfarin.
    #[test]
    fn cov_hessian_m2_theta_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.2, 10.0, 1.5];
        let subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        let mut params = model.default_params.clone();
        params.theta = theta.clone();

        let eta = precise_ebe(&model, &subject, &params);
        // The θθ explicit cross-partial needs `d2f_dtheta2`, which only the
        // cov-augmented provider pass populates.
        let sens = subject_sensitivities_cov(&model, &subject, &params.theta, &eta).unwrap();
        let prep = prepare(&model, &subject, &params, &sens).unwrap();
        let n_theta = params.theta.len();
        let analytic = cov_hessian_m2_theta(&prep, &sens, n_theta);

        // Reconverged central FD of the Φ θ-gradient.
        let mut fd = DMatrix::zeros(n_theta, n_theta);
        for nidx in 0..n_theta {
            let h = 1e-6 * (1.0 + theta[nidx].abs());
            let mut pp = params.clone();
            pp.theta[nidx] += h;
            let gp = phi_theta_grad(&model, &subject, &pp);
            let mut pm = params.clone();
            pm.theta[nidx] -= h;
            let gm = phi_theta_grad(&model, &subject, &pm);
            for m in 0..n_theta {
                fd[(m, nidx)] = (gp[m] - gm[m]) / (2.0 * h);
            }
        }

        for nidx in 0..n_theta {
            for m in 0..n_theta {
                let a = analytic[(nidx, m)];
                let f = fd[(nidx, m)];
                let tol = 1e-4 * (1.0 + a.abs());
                assert!(
                    (a - f).abs() < tol,
                    "θθ[{},{}]: analytic {:.8e} vs FD {:.8e} (Δ {:.2e})",
                    nidx,
                    m,
                    a,
                    f,
                    (a - f).abs()
                );
            }
        }
        // Symmetry of the analytic block (within assembly rounding).
        for nidx in 0..n_theta {
            for m in 0..n_theta {
                assert!((analytic[(nidx, m)] - analytic[(m, nidx)]).abs() < 1e-9);
            }
        }
    }
}
