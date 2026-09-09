//! Analytic total derivative `dH̃/dx` of the **Gauss-Newton (Almquist) Hessian**, for the
//! AGQ/FOCEI-quadrature grid response (`focei, n_agq > 1`) — the [`HessianAnchor::GaussNewton`]
//! sibling of [`crate::estimation::laplace_h_deriv`].
//!
//! # Why this needs no third order at all
//!
//! `H̃ = Ω⁻¹ + Σⱼ pⱼ aⱼaⱼᵀ` is **bilinear in first-order sensitivities only** — unlike the
//! exact conditional Hessian `H`, it carries no `Aⱼ = ∂²f/∂η²`-weighted curvature term. So its
//! total derivative needs only
//!
//! ```text
//!   dH̃/dx_k = dΩ⁻¹/dx_k + Σⱼ [ dpⱼ/dx_k · aⱼaⱼᵀ + pⱼ·(ψⱼaⱼᵀ + aⱼψⱼᵀ) ] ,
//!   ψⱼ = daⱼ/dx_k = Cⱼ·θ' + Aⱼ·Bₖ    (Cⱼ = ∂²f/∂η∂θ, Aⱼ = ∂²f/∂η²)
//! ```
//!
//! — the **second**-order blocks `Cⱼ`/`Aⱼ` the plain provider already returns, and `pⱼ`/`βⱼ
//! = dpⱼ/df` the [`score_core`] already computes for FOCEI's own `∂log|H̃|/∂x`. This module
//! calls the ordinary [`subject_sensitivities`] — the same call `anchor_hessian` already made
//! to build `H̃` itself — not the covariance provider's third-order sweep. Nothing here costs
//! more than one extra analytic-provider evaluation per subject per gradient, independent of
//! how many free packed coordinates there are; there is no `2·n_free`-vs-`1+2(n_theta+n_eta)`
//! trade to measure, unlike [`crate::estimation::laplace_h_deriv`].
//!
//! This mirrors nlmixr2est's `foceiGradSubjectAgqFR_` (`src/foceiGrad.cpp`), whose own doc
//! makes the same point: "the determinant Ht is the Gauss-Newton form ... so
//! `d(log|Ht|)/dtheta` needs only `d2f/(deta dtheta)`". Their AGQ is Gauss-Newton-anchored
//! only — ferx additionally has the exact-anchor Laplace case
//! ([`crate::estimation::laplace_h_deriv`]), which genuinely needs third order and has no
//! analogue in their trick.
//!
//! # `pⱼ`'s direct dependence beyond `f`
//!
//! A custom/time-varying σ magnitude (#484/#576/#486) moves `R`/`d` directly through `θ`
//! (`et.dr_dtheta`/`et.dd_dtheta`), independent of `f`; `σ` itself moves `R`/`d` directly too.
//! Both use the same closed form as `βⱼ = dpⱼ/df`, substituting the direction's `(∂R/∂v,
//! ∂d/∂v)` for `(d, d2)` — see [`dp_dv`]. That formula is specific to the **quantified**
//! functional form `p = 1/R + ½(d/R)²`; an M3-censored row's `p = g2` comes from the
//! `−logΦ(z)` kernel instead, and `dp_dv` does not apply to it — see the censored exclusion
//! below.
//!
//! # Scope
//!
//! Everything [`score_core`] itself supports analytically — M3-BLOQ (base value only, see
//! below), custom/TV σ magnitude, LTBS, `ExpressionScale`, FREM, closed-form **and ODE** —
//! minus four structural exclusions this assembly does not (yet) build:
//!
//! * **IOV / mixture** — the packed-layout decode below assumes exactly
//!   `[θ…, Ω lower-tri…, σ…]`;
//! * **`iiv_on_ruv`** — its `H̃` rows are the dedicated `(g, 2.0)` residual-η structure, not
//!   the generic `p·aaᵀ` form;
//! * **correlated residuals (`block_sigma`)** — the σ-direction `∂R/∂σ` needs the
//!   correlation-aware variance, not the plain scalar one this module differences;
//! * **M3-censored rows** — `score_core` gives a correct *base* `p = g2`/`β = dg2/df` (the
//!   `H̃` matrix `anchor_hessian` builds from them is fine), but this module's σ/magnitude
//!   `dp_dv` assumes the quantified form of `p` and would be silently wrong on `g2`.
//!
//! Anything outside returns `None` and the caller keeps the finite-difference grid response.

#![allow(clippy::needless_range_loop)]

use nalgebra::{DMatrix, DVector};

use crate::estimation::agq::analytic_score_supported;
use crate::estimation::parameterization::{lower_tri_entries, packed_len};
use crate::estimation::sens_outer_gradient::{score_core, sigma_fd_step, theta_dx_chain};
use crate::sens::provider::subject_sensitivities;
use crate::stats::likelihood::build_frem_r_override;
use crate::stats::residual_error::residual_rd;
use crate::types::{CompiledModel, ModelParameters, Subject};

/// `∂p/∂v = -Rᵥ/R² + d·dᵥ/R² - d²·Rᵥ/R³`, for any direction `v` of `p = 1/R + ½(d/R)²`
/// (`R`, `d = ∂R/∂f`). `βⱼ = dp_dv(r, d, d, d2)` — the same formula with `v = f`, `Rᵥ = d`,
/// `dᵥ = d2` — so this single closed form covers `β`, the magnitude direct-θ term
/// (`sens_outer_gradient::theta_block`'s inline `dp`) and the σ term (`sigma_block`'s inline
/// `dp`) at once. Kept here rather than three separate near-duplicates.
#[inline]
fn dp_dv(r: f64, d: f64, rv: f64, dv: f64) -> f64 {
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;
    let inv_r3 = inv_r2 * inv_r;
    -rv * inv_r2 + d * dv * inv_r2 - d * d * rv * inv_r3
}

/// The total derivative `dH̃/dx_k` of the Gauss-Newton Hessian, one `d × d` matrix per
/// packed coordinate (zero on a `fixed` coordinate, which the caller skips).
///
/// `omega_inv` is `Ω⁻¹` at `params` (non-IOV, so this is `Stack::omega_joint_inv` reduced to
/// the bare η block); `db_dx` the same implicit-function mode response
/// ([`crate::estimation::sens_outer_gradient::subject_eta_dx`]) the caller moves the grid
/// centre by.
///
/// Returns `None` — writing nothing — outside the scope in the module doc, or when the base
/// provider declines at `b_hat`.
pub(crate) fn subject_htilde_dx(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    template: &ModelParameters,
    omega_inv: &DMatrix<f64>,
    x: &[f64],
    b_hat: &[f64],
    db_dx: &[DVector<f64>],
) -> Option<Vec<DMatrix<f64>>> {
    let n_theta = params.theta.len();
    let n_eta = params.omega.dim();
    let n_sigma = params.sigma.values.len();

    let entries = lower_tri_entries(n_eta, params.omega.diagonal);
    if packed_len(template) != n_theta + entries.len() + n_sigma
        || x.len() != n_theta + entries.len() + n_sigma
        || db_dx.len() != x.len()
        || b_hat.len() != n_eta
        || n_eta == 0
        || model.n_kappa > 0
        || params.omega_iov.is_some()
        || template.mixture.is_some()
        || model.residual_error_eta.is_some()
        || !params.residual_correlations.is_empty()
        || !analytic_score_supported(model)
    {
        return None;
    }

    let sens = subject_sensitivities(model, subject, &params.theta, b_hat)?;
    let n_obs = subject.observations.len();
    if n_obs == 0 || sens.obs.len() != n_obs {
        return None;
    }
    let core = score_core(
        model,
        subject,
        params,
        &sens,
        n_eta,
        omega_inv,
        b_hat,
        model.residual_error_eta,
    )?;
    // Excluded by the model-level gate above; assert rather than re-test (CLAUDE.md: two
    // redundant gates cover for each other).
    debug_assert!(
        core.ruv.is_none(),
        "ruv excluded by residual_error_eta gate above"
    );
    // M3-censored rows: `score_core` already gives a correct BASE `p = g2`/`β = dg2/df` for
    // them (the `−logΦ(z)` kernel FD it always does), so the base `H̃` matrix this module
    // starts from is fine. But `dp_dv` below assumes the QUANTIFIED functional form
    // `p = 1/R + ½(d/R)²` — wrong for `g2`, which has no such closed form — so the σ-direct
    // and magnitude-direct terms would be silently wrong on a censored row. Decline rather
    // than build the censored kernel's own σ/θ derivative chain here; `sigma_block`'s
    // `kern_at` closure is the pattern a future extension would reuse.
    if core.et.iter().any(|t| t.censored) {
        return None;
    }

    let sigma = &params.sigma.values;
    let err_keys = model.error_spec.obs_keys(subject);

    // σ-direction `(∂R/∂σ_s, ∂d/∂σ_s)` per observation, hoisted once — independent of `k`,
    // exactly as `laplace_h_deriv::residual_row_derivs` hoists its analogue. A FREM
    // pseudo-observation row's variance is the dedicated `EPSCOV²` override, not
    // `error_spec`, so it gets its own branch (mirrors `sigma_block`'s `frem_row` arm) —
    // `d ≡ 0` there since the override is constant in `f`.
    let mut sigma_row_derivs: Vec<Vec<(f64, f64)>> = vec![Vec::with_capacity(n_obs); n_sigma];
    for s in 0..n_sigma {
        let h = sigma_fd_step(sigma[s]);
        let mut sp = sigma.to_vec();
        sp[s] += h;
        let mut sm = sigma.to_vec();
        sm[s] -= h;
        for (j, o) in sens.obs.iter().enumerate() {
            let cmt = err_keys[j];
            let frem_p = build_frem_r_override(model.frem_config.as_ref(), &subject.fremtype, &sp)
                .as_ref()
                .and_then(|ov| ov.get(j))
                .and_then(|v| *v);
            let frem_m = build_frem_r_override(model.frem_config.as_ref(), &subject.fremtype, &sm)
                .as_ref()
                .and_then(|ov| ov.get(j))
                .and_then(|v| *v);
            let (r_sig, d_sig) = if let (Some(vp), Some(vm)) = (frem_p, frem_m) {
                ((vp - vm) / (2.0 * h), 0.0)
            } else {
                let mult_row: Option<&[f64]> = core
                    .mult
                    .as_ref()
                    .and_then(|m| m.get(j))
                    .map(|v| v.as_slice());
                let (vp, dp) = residual_rd(&model.error_spec, cmt, o.f, &sp, mult_row);
                let (vm, dm) = residual_rd(&model.error_spec, cmt, o.f, &sm, mult_row);
                let scale = core.ruv_scale;
                (scale * (vp - vm) / (2.0 * h), scale * (dp - dm) / (2.0 * h))
            };
            sigma_row_derivs[s].push((r_sig, d_sig));
        }
    }

    let l = &params.omega.chol;
    let mut out: Vec<DMatrix<f64>> = Vec::with_capacity(x.len());
    let mut psi = vec![0.0f64; n_eta];
    for k in 0..x.len() {
        let mut dh = DMatrix::<f64>::zeros(n_eta, n_eta);
        let (mut theta_idx, mut dtheta) = (usize::MAX, 0.0f64);
        let (mut sigma_idx, mut dsigma) = (usize::MAX, 0.0f64);

        if k < n_theta {
            theta_idx = k;
            dtheta = theta_dx_chain(template, &params.theta, k);
        } else if k < n_theta + entries.len() {
            let (row, col) = entries[k - n_theta];
            let chain = if row == col { l[(row, row)] } else { 1.0 };
            let v = DVector::from_iterator(n_eta, (0..n_eta).map(|r| l[(r, col)]));
            let u: DVector<f64> = omega_inv.column(row).into_owned();
            let p = omega_inv * &v;
            let up = &u * p.transpose();
            dh = -(&up + up.transpose()) * chain;
        } else {
            sigma_idx = k - n_theta - entries.len();
            dsigma = sigma[sigma_idx]; // σ packs as ln σ ⇒ dσ/dx = σ.
        }

        let bk = &db_dx[k];
        for (j, o) in sens.obs.iter().enumerate() {
            let a = o.df_deta.as_slice();
            let big_a = o.d2f_deta2.as_slice();
            let et = &core.et[j];
            let p = et.p;

            let mut phi = 0.0;
            for r in 0..n_eta {
                phi += a[r] * bk[r];
            }
            if theta_idx < n_theta {
                phi += o.df_dtheta[theta_idx] * dtheta;
            }
            for i in 0..n_eta {
                let mut v = 0.0;
                for r in 0..n_eta {
                    v += big_a[i * n_eta + r] * bk[r];
                }
                if theta_idx < n_theta {
                    v += o.d2f_deta_dtheta[i * n_theta + theta_idx] * dtheta;
                }
                psi[i] = v;
            }

            let mut dp = et.beta * phi;
            // Custom-magnitude direct-θ channel: `mult(θ)` moves `R`/`d` independent of `f`.
            // `et.dr_dtheta` is empty on every row when no magnitude is active. `dr_dtheta`/
            // `dd_dtheta` are in NATURAL θ units (`magnitude_dvar_dtheta`'s own FD-pinned
            // contract), so this needs the same `dtheta` packed chain the φ/ψ terms above
            // already apply — omitting it was the bug `htilde_derivative_matches_fd_under_
            // custom_magnitude` caught.
            if theta_idx < n_theta && !et.dr_dtheta.is_empty() {
                let (rv, dv) = (et.dr_dtheta[theta_idx], et.dd_dtheta[theta_idx]);
                if rv != 0.0 || dv != 0.0 {
                    dp += dp_dv(et.r, et.d, rv, dv) * dtheta;
                }
            }
            if sigma_idx < n_sigma {
                let (rv, dv) = sigma_row_derivs[sigma_idx][j];
                dp += dp_dv(et.r, et.d, rv, dv) * dsigma;
            }

            for i in 0..n_eta {
                for m in 0..n_eta {
                    dh[(i, m)] += dp * a[i] * a[m] + p * (psi[i] * a[m] + a[i] * psi[m]);
                }
            }
        }

        if dh.iter().any(|v| !v.is_finite()) {
            return None;
        }
        // Symmetric by construction, but insure against round-off asymmetry between the two
        // triangles the same way `laplace_h_deriv` does — `RegularisedAnchor::factor_derivative`
        // debug-asserts exact symmetry.
        let sym = 0.5 * (&dh + dh.transpose());
        out.push(sym);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::parameterization::{pack_params, unpack_params};
    use crate::estimation::sens_outer_gradient::subject_eta_dx;
    use crate::parser::model_parser::parse_model_string;
    use crate::types::DoseEvent;
    use std::collections::HashMap;

    const MODEL: &str = r#"
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

    fn fixture_subject(model: &CompiledModel, theta: &[f64], times: &[f64]) -> Subject {
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
            reset_covariates: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let eta_ref = [0.12, -0.08, 0.2];
        let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, &eta_ref);
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        subject
    }

    /// `H̃(x)` — the Gauss-Newton anchor with the mode moved linearly along `db̂/dx`, i.e.
    /// exactly the function `subject_htilde_dx` claims to differentiate.
    fn htilde_at(
        model: &CompiledModel,
        subject: &Subject,
        template: &ModelParameters,
        x: &[f64],
        b: &[f64],
    ) -> DMatrix<f64> {
        let params = unpack_params(x, template);
        let omega_inv = params.omega.inv.clone();
        let sens = subject_sensitivities(model, subject, &params.theta, b).unwrap();
        score_core(
            model,
            subject,
            &params,
            &sens,
            params.omega.dim(),
            &omega_inv,
            b,
            None,
        )
        .expect("score_core")
        .htilde
    }

    fn setup(
        model_src: &str,
        theta: &[f64],
        times: &[f64],
    ) -> (CompiledModel, Subject, ModelParameters) {
        let model = parse_model_string(model_src).expect("parse");
        let subject = fixture_subject(&model, theta, times);
        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        (model, subject, template)
    }

    fn assert_matches_fd(
        model: &CompiledModel,
        subject: &Subject,
        template: &ModelParameters,
        b_hat: &[f64],
        tol: f64,
    ) {
        let x = pack_params(template);
        let params = unpack_params(&x, template);
        let omega_inv = params.omega.inv.clone();
        let db_dx = subject_eta_dx(model, subject, template, &x, b_hat).expect("eta_dx");

        let analytic = subject_htilde_dx(
            model, subject, &params, template, &omega_inv, &x, b_hat, &db_dx,
        )
        .expect("in scope");

        let n_eta = params.omega.dim();
        for k in 0..x.len() {
            let step = 1e-5 * (1.0 + x[k].abs());
            let mut xp = x.clone();
            xp[k] += step;
            let mut xm = x.clone();
            xm[k] -= step;
            let bp: Vec<f64> = (0..n_eta).map(|i| b_hat[i] + step * db_dx[k][i]).collect();
            let bm: Vec<f64> = (0..n_eta).map(|i| b_hat[i] - step * db_dx[k][i]).collect();
            let fd = (htilde_at(model, subject, template, &xp, &bp)
                - htilde_at(model, subject, template, &xm, &bm))
                / (2.0 * step);
            let scale = fd.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            for i in 0..n_eta {
                for m in 0..n_eta {
                    assert!(
                        (analytic[k][(i, m)] - fd[(i, m)]).abs() / scale < tol,
                        "coord {k} entry ({i},{m}): analytic {} vs FD {} (scale {scale})",
                        analytic[k][(i, m)],
                        fd[(i, m)]
                    );
                }
            }
        }
    }

    #[test]
    fn htilde_derivative_matches_central_difference_of_the_anchor() {
        let (model, subject, template) =
            setup(MODEL, &[0.22, 11.0, 1.4], &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        assert_matches_fd(&model, &subject, &template, &[0.05, -0.03, 0.08], 1e-5);
    }

    /// M3-censored: the base `p = g2`/`β = dg2/df` `score_core` computes are fine, but this
    /// module's σ/magnitude-direct `dp_dv` assumes the quantified functional form of `p` —
    /// wrong for `g2`. Must decline rather than build a plausible wrong `H̃` derivative.
    #[test]
    fn m3_censored_rows_decline() {
        let (mut model, subject, template) =
            setup(MODEL, &[0.22, 11.0, 1.4], &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        model.bloq_method = crate::types::BloqMethod::M3;
        let mut subject = subject;
        let n = subject.observations.len();
        subject.cens[n - 1] = 1;
        subject.cens[n - 2] = 1;
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let omega_inv = params.omega.inv.clone();
        let b_hat = [0.05, -0.03, 0.08];
        let db_dx = vec![DVector::zeros(3); x.len()];
        assert!(
            subject_htilde_dx(&model, &subject, &params, &template, &omega_inv, &x, &b_hat, &db_dx)
                .is_none(),
            "an M3 censored row must decline to the finite-difference route"
        );
    }

    /// A custom/time-varying σ magnitude: the direct-θ `dp_dv` channel must be live.
    #[test]
    fn htilde_derivative_matches_fd_under_custom_magnitude() {
        const MAG_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta RUV_LATE(2.0, 0.0, 5.0)
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
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))
"#;
        let (model, subject, template) = setup(
            MAG_MODEL,
            &[0.22, 11.0, 1.4, 2.0],
            &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0],
        );
        assert_matches_fd(&model, &subject, &template, &[0.05, -0.03, 0.08], 1e-5);
    }

    /// ODE: the base provider is representation-agnostic (no third-order tensor at all), so
    /// this needs no ODE exclusion, unlike `laplace_h_deriv`.
    #[test]
    fn htilde_derivative_matches_fd_under_ode() {
        const ODE_MODEL: &str = r#"
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
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot / V - (CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let (model, subject, template) = setup(ODE_MODEL, &[0.22, 11.0, 1.4], &[0.5, 2.0, 8.0]);
        // Looser than the closed-form cases: the FD *oracle* differences the ODE solver's
        // own output at default tolerances, so its noise floor — not the analytic formula
        // (which uses exact base sensitivities, no ODE-specific step choice at all) — sets
        // the achievable match here.
        assert_matches_fd(&model, &subject, &template, &[0.05, -0.03, 0.08], 5e-3);
    }

    #[test]
    fn htilde_derivative_is_exactly_symmetric() {
        let (model, subject, template) = setup(MODEL, &[0.22, 11.0, 1.4], &[0.5, 2.0, 8.0]);
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let omega_inv = params.omega.inv.clone();
        let b_hat = [0.05, -0.03, 0.08];
        let db_dx = subject_eta_dx(&model, &subject, &template, &x, &b_hat).expect("eta_dx");
        let analytic = subject_htilde_dx(
            &model, &subject, &params, &template, &omega_inv, &x, &b_hat, &db_dx,
        )
        .expect("in scope");
        for (k, m) in analytic.iter().enumerate() {
            assert_eq!(m, &m.transpose(), "coord {k} is not symmetric");
        }
    }

    /// `iiv_on_ruv` has its own `H̃` row structure; must decline.
    #[test]
    fn iiv_on_ruv_declines() {
        const RUV_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
"#;
        let (model, subject, template) = setup(RUV_MODEL, &[0.22, 11.0, 1.4], &[0.5, 2.0, 8.0]);
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let omega_inv = params.omega.inv.clone();
        let b_hat = [0.05, -0.03, 0.08, 0.0];
        let db_dx = vec![DVector::zeros(4); x.len()];
        assert!(
            subject_htilde_dx(&model, &subject, &params, &template, &omega_inv, &x, &b_hat, &db_dx)
                .is_none(),
            "iiv_on_ruv must decline to the finite-difference route"
        );
    }
}
