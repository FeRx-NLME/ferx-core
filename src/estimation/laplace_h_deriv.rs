//! Analytic total derivative `dH/dx` of the **exact** conditional Hessian, assembled from
//! third-order prediction sensitivities (#251 / #486 reuse).
//!
//! # What this is for
//!
//! Laplace (`method = laplace`, and adaptive Gauss–Hermite generally) anchors its grid on the
//! exact conditional Hessian
//!
//! ```text
//!   H = Ω⁻¹ + Σⱼ [ wⱼ aⱼaⱼᵀ + qⱼ Aⱼ ],   aⱼ = ∂fⱼ/∂η, Aⱼ = ∂²fⱼ/∂η²,
//!   qⱼ = ∂Lⱼ/∂f, wⱼ = ∂²Lⱼ/∂f²          (`score_core`'s `h_inner`, with q = ½α, w = ½α')
//! ```
//!
//! and the objective carries `½·log|H|`. Differentiating that needs `dH/dx`, whose structural
//! content is one derivative past the provider's second order: `∂³f/∂η³` and `∂³f/∂η²∂θ`.
//! [`crate::estimation::agq::grid_response_correction`] used to sidestep this by
//! *re-evaluating the whole anchor* at `x ± h` for every free packed coordinate — `2p`
//! perturbed provider sweeps per subject per gradient. This module supplies the same
//! derivative in closed form from **one** third-order jet.
//!
//! # Where the third order comes from
//!
//! Not from new sensitivity equations: from
//! [`crate::sens::provider::subject_sensitivities_cov`], the sweep the analytic covariance
//! (#436) already uses, which central-differences the `Dual2` second-order jet along each
//! `(θ, η)` axis. That is `1 + 2(n_theta + n_eta)` provider evaluations with **no inner
//! re-solve**, and it reaches both the closed-form kernels and the augmented ODE solve.
//! Reusing it rather than writing a second FD sweep also means the step heuristic
//! (`third_order_fd_step`, solver-tolerance aware) and the route-switch guards are shared.
//!
//! **This is the default route as of #1335, and a raw call-count comparison is still not the
//! whole story.** `2p` perturbed anchors and `1 + 2(n_theta + n_eta)` provider evaluations are
//! different work units: each call here does materially more work (a full third-order jet plus
//! the residual σ/f central differences below) than a plain anchor rebuild, so fewer calls does
//! not automatically mean less time. On the **diagonal-Ω** fixture the two sit so close to
//! parity that successive sessions measured opposite signs (+34% slower, then ~14% faster, both
//! real) — which is why this route was opt-in for as long as that was the only fixture measured.
//!
//! The **block-Ω** case decided it, and it is the one to reason from: there the call-count gap
//! is wide (`2·n_free = 20` FD rebuilds vs `1 + 2(n_theta + d) = 13` provider evaluations)
//! versus 14 FD rebuilds and 13 provider evaluations on the diagonal fixture. It measured 11520
//! → 8160 calls for −24% provider time
//! across 5/5 reps at an **identical** outer-iteration count and OFV matching to four decimals.
//! Rounded OFV agreement is not a derivative check; the grid-response and Hessian-derivative
//! parity tests supply that evidence. See
//! [`crate::estimation::agq::use_analytic_grid_response`] for the full table and for why the
//! `FERX_AGQ_GRID_RESPONSE` harness is kept rather than removed.
//!
//! # The derivative
//!
//! Write `Bₖ = db̂/dx_k` (the implicit-function mode response, supplied by the caller from
//! [`crate::estimation::sens_outer_gradient::subject_eta_dx`]), and let `x_k` move `θ` by
//! `θ'` and `σ` by `σ'` through the packed chain. Then, per observation,
//!
//! ```text
//!   φ  = dfⱼ/dx_k    = bⱼᵀθ' + aⱼᵀBₖ
//!   ψ  = daⱼ/dx_k    = Cⱼθ'  + AⱼBₖ                 (Cⱼ = ∂²f/∂η∂θ)
//!   Ψ  = dAⱼ/dx_k    = Uⱼθ'  + TⱼBₖ                 (Uⱼ = ∂³f/∂η²∂θ, Tⱼ = ∂³f/∂η³)
//!   dq = wⱼφ + (∂qⱼ/∂σ)σ'
//!   dw = (∂wⱼ/∂f)φ + (∂wⱼ/∂σ)σ'
//! ```
//!
//! and
//!
//! ```text
//!   dH/dx_k = dΩ⁻¹/dx_k + Σⱼ [ dw·aⱼaⱼᵀ + wⱼ(ψaⱼᵀ + aⱼψᵀ) + dq·Aⱼ + qⱼ·Ψ ].
//! ```
//!
//! `∂q/∂f = w` exactly, because `L` is a function of the scalar `f` alone once `σ` is fixed
//! (`ε = y − f` and `R = R(f)` both flow through it) — so the residual chain needs only one
//! *new* scalar per row, `∂w/∂f = ∂³L/∂f³`, plus the σ-derivatives of `q` and `w`.
//!
//! # Why those residual scalars are finite-differenced
//!
//! `∂³L/∂f³` needs `∂³R/∂f³`, which the [`crate::stats::residual_error::ErrorSpec`] API does
//! not expose and which is *not* identically zero (a `power(...)` loading has curvature, and
//! the `MIN_VARIANCE` floor makes every branch piecewise). Rather than duplicate the variance
//! model here — the one place a silent divergence from the objective would produce a
//! plausible wrong Hessian — the three scalars are central differences of the **closed-form
//! kernel** `err_terms(residual_rd2(...))`. No model evaluation, no inner solve: this is
//! arithmetic on `(R, ∂R/∂f, ∂²R/∂f²)`, exactly the precedent `score_core` sets for the
//! censored `dg2/df` and `sigma_block`/`subject_eta_dx` set for `∂R/∂σ`. The differenced
//! function is analytic, so the error is `~ε^{2/3} ≈ 4e-11` relative — orders below the
//! `AGQ_GRID_FD_STEP` truncation the FD route it replaces carries.
//!
//! # Scope
//!
//! Deliberately the *intersection* of the covariance provider's gate (which already excludes
//! LTBS, expression scaling, Form-C readouts, FREM, `iiv_on_ruv`, custom σ magnitude,
//! correlated residuals, non-Gaussian endpoints and `gradient = fd`) with three further
//! restrictions this assembly owns. **ODE is admitted** — `covariance_sensitivities` (via
//! `ode_analytical_supported`) reaches it, so this module does too; `third_order_fd_step`
//! widening the FD step to the solver's `reltol` there is genuine integration noise, not a
//! formula gap (`h_derivative_matches_fd_under_ode`, `tol = 5e-3` vs `1e-5` closed-form).
//!
//! * **no M3-censored row** — the `−logΦ(z)` kernel's `q`/`w` are `2·g1`/`2·g2`, whose third
//!   `f`-derivative and σ-derivatives are a different chain than `err_terms`;
//! * **no IOV / mixture** — the packed layout is asserted to be exactly
//!   `[θ…, Ω lower-tri…, σ…]`, so a coordinate segment cannot be silently mis-attributed;
//! * **no residual-η (`iiv_on_ruv`)** — its `H` rows are assembled from `(g, κ)` scalars
//!   rather than `(a, A)`, so they would need their own derivative terms.
//!
//! Anything outside returns `None` and the caller keeps the finite-difference grid response,
//! which is correct for all of them.

// Indexed loops walk parallel jet/Hessian buffers; clearer than zips.
#![allow(clippy::needless_range_loop)]

use nalgebra::{DMatrix, DVector};

use crate::estimation::parameterization::{lower_tri_entries, packed_len};
use crate::estimation::sens_outer_gradient::{
    err_terms, score_core, sigma_fd_step, theta_dx_chain,
};
use crate::sens::provider::subject_sensitivities_cov;
use crate::stats::residual_error::residual_rd2;
use crate::types::{BloqMethod, CompiledModel, ModelParameters, Subject};

/// Relative half-step for central-differencing the residual kernel in `f`.
///
/// `ε^(1/3) ≈ 6.06e-6` is the textbook optimum for a central first difference of an
/// *analytic* function: truncation falls as `h²`, round-off grows as `ε/h`. The kernel here is
/// closed-form arithmetic on `(R, ∂R/∂f, ∂²R/∂f²)`, so `ε` really is the noise floor — unlike
/// [`crate::sens::provider::third_order_fd_step`], which must widen for a solver tolerance.
#[inline]
fn kernel_step(f: f64) -> f64 {
    f64::EPSILON.cbrt() * (1.0 + f.abs())
}

/// Per-observation residual scalars this assembly needs *beyond* what `score_core` returns.
struct RowDerivs {
    /// `∂w/∂f = ∂³L/∂f³` (stored as `½·dα'/df`, matching `w = ½α'`).
    dw_df: f64,
    /// `∂q/∂σ_s` per σ slot.
    dq_dsigma: Vec<f64>,
    /// `∂w/∂σ_s` per σ slot.
    dw_dsigma: Vec<f64>,
}

/// The total derivative `dH/dx_k` of the exact conditional Hessian, one `d × d` matrix per
/// packed coordinate (a zero matrix on `fixed` coordinates, which the caller skips anyway).
///
/// `omega_inv` must be the same `Ω⁻¹` the anchor was built with (`Stack::omega_joint_inv`);
/// `db_dx` the same mode response the caller moves the grid centre by, so the analytic and
/// finite-difference routes differentiate *the same* function of `x`.
///
/// Returns `None` — without writing anything — outside the scope described in the module
/// documentation, or when the provider declines at the base point.
pub(crate) fn subject_h_inner_dx(
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

    // Layout assertion, not a preference: the coordinate index is decoded arithmetically
    // below, so an extra packed segment (IOV Ω, mixture overrides, `block_sigma` ρ) would be
    // read as a σ coordinate and produce a plausible wrong derivative rather than an error.
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
    {
        return None;
    }
    // The `−logΦ(z)` censored kernel has its own `q`/`w` chain (`2·g1`, `2·g2`); `err_terms`
    // below would silently score such a row as if it were quantified.
    if matches!(model.bloq_method, BloqMethod::M3) && subject.cens.iter().any(|&c| c != 0) {
        return None;
    }

    // The third-order jet. Its `obs` base blocks are bit-identical to
    // `subject_sensitivities`, so `score_core` below sees exactly the values that built the
    // anchor the caller passes to `build_proposal`.
    let sens = subject_sensitivities_cov(model, subject, &params.theta, b_hat)?;
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
    // `core.ruv` and every `core.et[..].censored` are already excluded by the
    // `residual_error_eta` / M3-censored checks above — `score_core` derives both directly
    // from those same two model/subject facts, so a second gate here would be redundant
    // rather than defensive (CLAUDE.md: "two redundant gates cover for each other"). Assert
    // the equivalence instead of re-testing it, so a future `score_core` change that adds a
    // new path to either flag fails loudly here rather than silently degrading `dH/dx`.
    debug_assert!(
        core.ruv.is_none(),
        "ruv excluded by residual_error_eta gate above"
    );
    debug_assert!(
        core.et.iter().all(|t| !t.censored),
        "censored rows excluded by the M3 gate above"
    );
    // Custom/time-varying σ magnitude (#484/#576/#486) is a genuinely separate condition —
    // `mult(θ)` is orthogonal to the layout/M3 checks above and needs its own direct-θ
    // channel this assembly does not build.
    if core.mult.is_some() {
        return None;
    }
    // The covariance sweep leaves these empty when a perturbed point declined; without them
    // the θ/η structural response below would be silently zero.
    for o in sens.obs.iter() {
        if o.d3f_deta3.len() != n_eta * n_eta * n_eta
            || o.d3f_deta2_dtheta.len() != n_eta * n_eta * n_theta
            || o.d2f_deta_dtheta.len() != n_eta * n_theta
            || o.d2f_deta2.len() != n_eta * n_eta
            || o.df_deta.len() != n_eta
            || o.df_dtheta.len() != n_theta
        {
            return None;
        }
    }

    let sigma = &params.sigma.values;
    let err_keys = model.error_spec.obs_keys(subject);
    let rows = residual_row_derivs(model, subject, &sens, sigma, &err_keys, n_sigma)?;

    // Ω⁻¹ response, per Ω coordinate: `Ω = L Lᵀ`, so moving `L[row,col]` moves
    // `Ω` by `e_row vᵀ + v e_rowᵀ` (`v = L[:,col]`) and hence
    // `Ω⁻¹` by `−(u pᵀ + p uᵀ)` with `u = Ω⁻¹e_row`, `p = Ω⁻¹v`. Same construction as
    // `subject_eta_dx`'s `m_l`, which is this matrix already contracted with `η̂`.
    let l = &params.omega.chol;

    let mut out: Vec<DMatrix<f64>> = Vec::with_capacity(x.len());
    let mut psi = vec![0.0f64; n_eta];
    for k in 0..x.len() {
        let mut dh = DMatrix::<f64>::zeros(n_eta, n_eta);
        let mut dtheta = 0.0f64;
        let mut theta_idx = usize::MAX;
        let mut dsigma = 0.0f64;
        let mut sigma_idx = usize::MAX;

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
            // σ packs as `ln σ`, so `dσ/dx = σ`.
            dsigma = sigma[sigma_idx];
        }

        let bk = &db_dx[k];
        for (j, o) in sens.obs.iter().enumerate() {
            let a = o.df_deta.as_slice();
            let big_a = o.d2f_deta2.as_slice();
            let et = &core.et[j];
            let (q, w) = (0.5 * et.alpha, 0.5 * et.alpha_p);

            // φ = df/dx_k, ψ = da/dx_k. (Ψ = dA/dx_k is formed inline in the accumulation
            // loop below — it is the only `d × d` object per observation and materialising
            // it would allocate once per (coordinate, observation).)
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

            let mut dq = w * phi;
            let mut dw = rows[j].dw_df * phi;
            if sigma_idx < n_sigma {
                dq += rows[j].dq_dsigma[sigma_idx] * dsigma;
                dw += rows[j].dw_dsigma[sigma_idx] * dsigma;
            }

            for i in 0..n_eta {
                for m in 0..n_eta {
                    // Ψ[i,m] = Σ_r T[i,m,r]·Bₖ[r] + U[i,m,θ]·θ'
                    let mut big_psi = 0.0;
                    let base = (i * n_eta + m) * n_eta;
                    for r in 0..n_eta {
                        big_psi += o.d3f_deta3[base + r] * bk[r];
                    }
                    if theta_idx < n_theta {
                        big_psi +=
                            o.d3f_deta2_dtheta[(i * n_eta + m) * n_theta + theta_idx] * dtheta;
                    }
                    dh[(i, m)] += dw * a[i] * a[m]
                        + w * (psi[i] * a[m] + a[i] * psi[m])
                        + dq * big_a[i * n_eta + m]
                        + q * big_psi;
                }
            }
        }

        if dh.iter().any(|v| !v.is_finite()) {
            return None;
        }
        // Symmetric by construction (every term above is), but the third-order blocks reach
        // it through two independent FD stencils, so round-off can leave the two triangles
        // differing in the last bits — and `RegularisedAnchor::factor_derivative` debug-asserts
        // exact symmetry. Project rather than assert.
        let sym = 0.5 * (&dh + dh.transpose());
        out.push(sym);
    }
    Some(out)
}

/// `(∂w/∂f, ∂q/∂σ, ∂w/∂σ)` per observation, by central difference of the closed-form
/// residual kernel. See the module note on why these three are differenced rather than
/// derived.
fn residual_row_derivs(
    model: &CompiledModel,
    subject: &Subject,
    sens: &crate::sens::provider::SubjectSens,
    sigma: &[f64],
    err_keys: &[usize],
    n_sigma: usize,
) -> Option<Vec<RowDerivs>> {
    let mut rows = Vec::with_capacity(sens.obs.len());
    for (j, o) in sens.obs.iter().enumerate() {
        let cmt = err_keys[j];
        let f = o.f;
        let y = subject.observations[j];

        // ∂w/∂f: `w = ½α'`, and `α'` is a closed form in `(R, ∂R/∂f, ∂²R/∂f², ε)`, all of
        // which move with `f`. Differencing the whole kernel picks up every one of those
        // paths, including `∂³R/∂f³` on a `power(...)` loading.
        let hf = kernel_step(f);
        let at_f = |ff: f64| -> f64 {
            let (r, d, d2) = residual_rd2(&model.error_spec, cmt, ff, sigma, None);
            err_terms(r, d, d2, y - ff).alpha_p
        };
        let dw_df = 0.5 * (at_f(f + hf) - at_f(f - hf)) / (2.0 * hf);

        // σ derivatives: `ε` is σ-independent, so only `(R, ∂R/∂f, ∂²R/∂f²)` move. Same
        // step policy as `sigma_block` / `subject_eta_dx`, which keeps `σ − h` positive.
        let mut dq_dsigma = vec![0.0f64; n_sigma];
        let mut dw_dsigma = vec![0.0f64; n_sigma];
        for s in 0..n_sigma {
            let h = sigma_fd_step(sigma[s]);
            let mut sp = sigma.to_vec();
            sp[s] += h;
            let mut sm = sigma.to_vec();
            sm[s] -= h;
            let at_sigma = |sg: &[f64]| -> (f64, f64) {
                let (r, d, d2) = residual_rd2(&model.error_spec, cmt, f, sg, None);
                let t = err_terms(r, d, d2, y - f);
                (0.5 * t.alpha, 0.5 * t.alpha_p)
            };
            let (qp, wp) = at_sigma(&sp);
            let (qm, wm) = at_sigma(&sm);
            dq_dsigma[s] = (qp - qm) / (2.0 * h);
            dw_dsigma[s] = (wp - wm) / (2.0 * h);
        }

        if !dw_df.is_finite()
            || dq_dsigma
                .iter()
                .chain(dw_dsigma.iter())
                .any(|v| !v.is_finite())
        {
            return None;
        }
        rows.push(RowDerivs {
            dw_df,
            dq_dsigma,
            dw_dsigma,
        });
    }
    Some(rows)
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

    /// `H(x)` — the exact conditional Hessian with the mode moved linearly along `db̂/dx`,
    /// i.e. exactly the function `subject_h_inner_dx` claims to differentiate (and exactly
    /// what `grid_response_correction`'s FD route re-evaluates at `x ± h`).
    fn h_at(
        model: &CompiledModel,
        subject: &Subject,
        template: &ModelParameters,
        x: &[f64],
        b: &[f64],
    ) -> DMatrix<f64> {
        let params = unpack_params(x, template);
        let omega_inv = params.omega.inv.clone();
        let sens =
            crate::sens::provider::subject_sensitivities(model, subject, &params.theta, b).unwrap();
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
        .h_inner
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

    /// Shared FD-parity assertion: every packed coordinate's `dH/dx_k` must match a central
    /// difference of the same `H`, with the same mode response applied.
    ///
    /// The reference moves the mode by `b̂ ± h·db̂/dx_k` rather than re-solving the inner
    /// loop, because that *is* the derivative under test — the implicit-function response,
    /// not a reconverged EBE. A reconverged reference would fold in inner-solver noise and
    /// test the inner tolerance instead. `b_hat` is deliberately off-mode in every caller:
    /// the assembly is a property of `H(b, x)` and its documented mode response, so it must
    /// hold away from the EBE too.
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

        let analytic = subject_h_inner_dx(
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
            let fd = (h_at(model, subject, template, &xp, &bp)
                - h_at(model, subject, template, &xm, &bm))
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
    fn h_derivative_matches_central_difference_of_the_anchor() {
        let (model, subject, template) =
            setup(MODEL, &[0.22, 11.0, 1.4], &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        assert_matches_fd(&model, &subject, &template, &[0.05, -0.03, 0.08], 1e-5);
    }

    /// Robustness: a second, unrelated `θ`/`b̂` point on the same model — the formula must
    /// hold everywhere in the domain, not just at the one point the primary test happens to
    /// probe.
    #[test]
    fn h_derivative_matches_fd_at_a_different_point() {
        let (model, subject, template) = setup(
            MODEL,
            &[0.35, 7.5, 0.9],
            &[0.25, 0.75, 3.0, 6.0, 12.0, 18.0],
        );
        assert_matches_fd(&model, &subject, &template, &[-0.15, 0.22, -0.06], 1e-5);
    }

    /// Robustness: a full 3×3 block Ω — exercises the off-diagonal Cholesky-chain path in
    /// the Ω⁻¹ direct-response term (`row != col` in the coordinate decode), which the
    /// diagonal-Ω `MODEL` fixture never touches at all.
    #[test]
    fn h_derivative_matches_fd_under_block_omega() {
        const BLOCK_OMEGA_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  block_omega (ETA_CL, ETA_V, ETA_KA) = [
    0.09,
    0.02, 0.04,
    0.01, 0.01, 0.30
  ]
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
        let (model, subject, template) = setup(
            BLOCK_OMEGA_MODEL,
            &[0.22, 11.0, 1.4],
            &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0],
        );
        assert_matches_fd(&model, &subject, &template, &[0.05, -0.03, 0.08], 1e-5);
    }

    /// Robustness: a `combined` error model (two σ slots, both the additive and
    /// proportional σ-direction chains live at once) — exercises the σ-loop's
    /// `residual_row_derivs` for more than one slot on the same observation.
    #[test]
    fn h_derivative_matches_fd_under_combined_error() {
        const COMBINED_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma ADD_ERR  ~ 0.05
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ combined(ADD_ERR, PROP_ERR)
"#;
        let (model, subject, template) = setup(
            COMBINED_MODEL,
            &[0.22, 11.0, 1.4],
            &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0],
        );
        assert_matches_fd(&model, &subject, &template, &[0.05, -0.03, 0.08], 1e-5);
    }

    /// Robustness: a single-η model (`n_eta = 1`) — the smallest dimension the coordinate
    /// decode and Ω-direction loop can take, where an off-by-one in an index arithmetic
    /// formula is likeliest to show up as a panic rather than a silently wrong value.
    #[test]
    fn h_derivative_matches_fd_with_one_eta() {
        const ONE_ETA_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let (model, subject, template) = setup(
            ONE_ETA_MODEL,
            &[0.22, 11.0, 1.4],
            &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0],
        );
        assert_matches_fd(&model, &subject, &template, &[0.05], 1e-5);
    }

    /// Every returned matrix must be exactly symmetric — `RegularisedAnchor::factor_derivative`
    /// debug-asserts it, so an asymmetric slip would abort a debug fit inside a rayon loop.
    #[test]
    fn h_derivative_is_exactly_symmetric() {
        let model = parse_model_string(MODEL).expect("parse");
        let theta = [0.22, 11.0, 1.4];
        let subject = fixture_subject(&model, &theta, &[0.5, 2.0, 8.0]);
        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let omega_inv = params.omega.inv.clone();
        let b_hat = [0.05, -0.03, 0.08];
        let db_dx = subject_eta_dx(&model, &subject, &template, &x, &b_hat).expect("eta_dx");
        let analytic = subject_h_inner_dx(
            &model, &subject, &params, &template, &omega_inv, &x, &b_hat, &db_dx,
        )
        .expect("in scope");
        for (k, m) in analytic.iter().enumerate() {
            assert_eq!(m, &m.transpose(), "coord {k} is not symmetric");
        }
    }

    /// An M3-censored row is out of scope: `err_terms` would score it as if it were
    /// quantified, which is a wrong Hessian rather than a failure.
    #[test]
    fn censored_rows_decline() {
        let model = parse_model_string(MODEL).expect("parse");
        let mut model = model;
        model.bloq_method = BloqMethod::M3;
        let theta = [0.22, 11.0, 1.4];
        let mut subject = fixture_subject(&model, &theta, &[0.5, 2.0, 8.0]);
        subject.cens[2] = 1;
        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let omega_inv = params.omega.inv.clone();
        let b_hat = [0.05, -0.03, 0.08];
        let db_dx = vec![DVector::zeros(3); x.len()];
        assert!(
            subject_h_inner_dx(
                &model, &subject, &params, &template, &omega_inv, &x, &b_hat, &db_dx,
            )
            .is_none(),
            "an M3 censored row must decline to the finite-difference route"
        );
    }

    /// ODE models: `covariance_sensitivities` (via `ode_analytical_supported`) admits them —
    /// the third-order jet is available from the augmented solve — so this module does too.
    /// `third_order_fd_step` widens the FD step to the solver's `reltol` on this route, which
    /// is genuine integration noise rather than a formula error, so the tolerance here is
    /// looser than the closed-form cases (measured, mirroring
    /// `focei_htilde_dx::tests::htilde_derivative_matches_fd_under_ode`).
    #[test]
    fn h_derivative_matches_fd_under_ode() {
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
        assert!(model.ode_spec.is_some(), "fixture must be an ODE model");
        assert_matches_fd(&model, &subject, &template, &[0.05, -0.03, 0.08], 5e-3);
    }

    /// A custom residual-magnitude model (#484/#576) needs its own direct-θ channel this
    /// assembly does not build — `core.mult.is_some()` must reject it rather than silently
    /// omitting that term.
    #[test]
    fn custom_magnitude_models_decline() {
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
        let model = parse_model_string(MAG_MODEL).expect("parse");
        let theta = [0.22, 11.0, 1.4, 2.0];
        let subject = fixture_subject(&model, &theta, &[0.5, 2.0, 8.0]);
        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let omega_inv = params.omega.inv.clone();
        let b_hat = [0.05, -0.03, 0.08];
        let db_dx = vec![DVector::zeros(3); x.len()];
        assert!(
            subject_h_inner_dx(
                &model, &subject, &params, &template, &omega_inv, &x, &b_hat, &db_dx,
            )
            .is_none(),
            "a custom-magnitude model must decline to the finite-difference route"
        );
    }

    /// A packed layout the assembly does not model (here IOV, which appends a κ segment)
    /// must decline rather than misattribute the extra coordinates as σ.
    #[test]
    fn mismatched_layout_declines() {
        let model = parse_model_string(MODEL).expect("parse");
        let theta = [0.22, 11.0, 1.4];
        let subject = fixture_subject(&model, &theta, &[0.5, 2.0, 8.0]);
        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let omega_inv = params.omega.inv.clone();
        let b_hat = [0.05, -0.03, 0.08];
        // One coordinate short of the real packed length.
        let short_x = &x[..x.len() - 1];
        let db_dx = vec![DVector::zeros(3); short_x.len()];
        assert!(
            subject_h_inner_dx(
                &model, &subject, &params, &template, &omega_inv, short_x, &b_hat, &db_dx,
            )
            .is_none(),
            "a packed-length mismatch must decline rather than misread the layout"
        );
    }
}
