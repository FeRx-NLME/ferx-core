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
//! functional form `p = 1/R + ½(d/R)²`. An M3-censored row's `p = g2` comes from the
//! `−logΦ(z)` kernel instead (`et.censored`), whose `∂g2/∂σ` has no such closed form — the
//! σ-direct precompute below central-differences `m3_censored_outer`'s `g2` output directly
//! for those rows (mirroring `sens_outer_gradient::sigma_block`'s censored `kern_at`
//! closure), rather than routing them through `dp_dv`. A censored row can never carry an
//! active magnitude on the same subject — `score_core` itself declines that combination — so
//! the direct-θ branch above never needs the equivalent swap.
//!
//! # `iiv_on_ruv`'s residual-eta row
//!
//! A quantified row's `H̃[rr,rr] += 2.0` is constant (contributes nothing to `dH̃/dx`);
//! `H̃[rr,l] += gⱼ·a_{jl}` for `l ≠ rr` is not, where `gⱼ = dⱼ/Rⱼ` is scale-free (`ruv_scale`
//! cancels in the ratio). `dg/dx_k` follows the same quotient-rule shape as `dp_dv`,
//! substituted per direction — see [`dg_dv`]. A censored row's residual-eta entries are the
//! `ruv_cz`/`ruv_cm` chain instead (`ErrTerms` doc): `H̃[rr,rr] += ruv_cz`,
//! `H̃[rr,l] += ruv_cm·a_{jl}`. `ScoreCore` already central-differences `∂(ruv_cz)/∂f` and
//! `∂(ruv_cm)/∂f` (`cens_dcz_df`/`cens_dcm_df`, used by `sens_outer_gradient::theta_block`'s
//! own censored-`iiv_on_ruv` term), so the structural chain is reused rather than
//! re-differenced; the σ-direct precompute below adds the matching `∂/∂σ` pair the same way it
//! adds `∂g2/∂σ` for a censored row's `p`. In both cases `a[rr] = 0` structurally (the
//! prediction never depends on `η_ruv`), so the generic `p·aaᵀ` loop above already leaves the
//! `rr` row/col untouched — this is a pure addition, not a special case of it. A FREM
//! pseudo-observation row has no `η_ruv` dependence at all (`score_core`'s own
//! `ruv.filter(|_| frem_var.is_none())`), so it is skipped the same way here (a censored row
//! is never a FREM row, so no combined check is needed there).
//!
//! # Correlated residuals (`block_sigma`)
//!
//! `score_core`'s own `corr_diag` branch already builds a correlation-aware `(R_jj, ∂R_jj/∂f_j,
//! ∂²R_jj/∂f_j²)` for `et.r`/`et.d` — and declines (`?`-propagates `None`) whenever the current
//! `ρ` makes the subject's `R` genuinely off-diagonal, since only the diagonal case reduces to
//! the scalar `p·aaᵀ`/`(g, 2.0)` machinery this module (and `score_core` itself) builds. The
//! σ-direct precompute mirrors that: it differences [`corr_residual_rd_at_sigma`] (the same
//! correlation-aware function `sens_outer_gradient::sigma_block` differences) instead of the
//! plain per-endpoint `residual_rd`, whenever `residual_correlations` is non-empty. `corr_diag`
//! does not apply `ruv_scale` to a correlated row either, so this module doesn't — the two
//! features are mutually exclusive in practice.
//!
//! # Scope
//!
//! Everything [`score_core`] itself supports analytically — M3-BLOQ (including the σ-direct
//! derivative), `iiv_on_ruv` (including combined with M3-BLOQ, see above), custom/TV σ
//! magnitude, correlated residuals (`block_sigma`, see above), LTBS, `ExpressionScale`, FREM,
//! closed-form **and ODE** — minus one structural exclusion this assembly does not (yet) build:
//!
//! * **mixture** — a mixture's subpopulation overrides are a different packed segment
//!   again, not decoded here.
//!
//! **IOV** has its own sibling, [`subject_htilde_dx_iov`] — the packed layout, the joint
//! `Ω⁻¹`, and the sensitivity provider all change under IOV, so it is not a branch of this
//! function.
//!
//! Anything outside returns `None` and the caller keeps the finite-difference grid response.

#![allow(clippy::needless_range_loop)]

use nalgebra::{DMatrix, DVector};

use crate::estimation::agq::analytic_score_supported;
use crate::estimation::parameterization::{
    block_chol_full, lower_tri_entries, packed_len, rho_chain,
};
use crate::estimation::sens_outer_gradient::{
    corr_residual_rd_at_sigma, rho_rd_terms, score_core, sigma_fd_step, theta_dx_chain,
};
use crate::sens::provider::subject_sensitivities;
use crate::stats::likelihood::build_frem_r_override;
use crate::stats::residual_error::{residual_rd, residual_rd2};
use crate::stats::special::m3_censored_outer;
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

/// `∂g/∂v = (dᵥ·R − d·Rᵥ)/R²`, the quotient rule for `g = d/R` (`iiv_on_ruv`'s scale-free
/// residual-eta coupling, `H̃[ruv,l] += g·a_l`). `dg/df = dg_dv(r, d, d, d2)` — same
/// substitution pattern as [`dp_dv`], reused across the structural-`f`, magnitude-direct-θ
/// and σ directions.
#[inline]
fn dg_dv(r: f64, d: f64, rv: f64, dv: f64) -> f64 {
    (dv * r - d * rv) / (r * r)
}

/// `dΩ⁻¹/dL_{row,col}` for a Cholesky factor `L` of `Ω`: `-(u pᵀ + p uᵀ)`, `u = Ω⁻¹[:,row]`,
/// `p = Ω⁻¹·L[:,col]`. The same formula [`subject_htilde_dx`]'s non-IOV Ω-coordinate branch
/// uses inline; factored out for [`subject_htilde_dx_iov`], which needs it summed over `K`
/// replica positions for an IOV Ω entry (one entry of `L_iov` moves `K` diagonal blocks of the
/// joint `Σ_b` at once).
#[inline]
fn omega_inv_deriv(omega_inv: &DMatrix<f64>, u: &DVector<f64>, v: &DVector<f64>) -> DMatrix<f64> {
    let p = omega_inv * v;
    let up = u * p.transpose();
    -(&up + up.transpose())
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
    let n_rho = params.residual_correlations.len();

    let entries = lower_tri_entries(n_eta, params.omega.diagonal);
    if packed_len(template) != n_theta + entries.len() + n_sigma + n_rho
        || x.len() != n_theta + entries.len() + n_sigma + n_rho
        || db_dx.len() != x.len()
        || b_hat.len() != n_eta
        || n_eta == 0
        || model.n_kappa > 0
        || params.omega_iov.is_some()
        || template.mixture.is_some()
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
    let sigma = &params.sigma.values;
    let err_keys = model.error_spec.obs_keys(subject);
    // FREM pseudo-observation rows carry no η_ruv dependence at all (`score_core`'s own
    // `ruv.filter(|_| frem_var.is_none())`), so the residual-eta row/col addition below must
    // skip them the same way.
    let frem_r_base = build_frem_r_override(model.frem_config.as_ref(), &subject.fremtype, sigma);
    // `∂²R/∂f²` per observation, needed only for `iiv_on_ruv`'s `g = d/R` structural chain
    // (`dg_dv(r, d, d, d2)`) — not otherwise read by this module, so left empty when
    // `iiv_on_ruv` is inactive (the common case).
    let d2_row: Vec<f64> = if core.ruv.is_some() {
        sens.obs
            .iter()
            .enumerate()
            .map(|(j, o)| {
                let cmt = err_keys[j];
                let mult_row: Option<&[f64]> = core
                    .mult
                    .as_ref()
                    .and_then(|m| m.get(j))
                    .map(|v| v.as_slice());
                let (_, _, d2) = residual_rd2(&model.error_spec, cmt, o.f, sigma, mult_row);
                d2 * core.ruv_scale
            })
            .collect()
    } else {
        Vec::new()
    };

    // `∂p_j/∂σ_s` per observation, hoisted once — independent of `k`, exactly as
    // `laplace_h_deriv::residual_row_derivs` hoists its analogue. Stores the finished `dp`
    // (not the raw `(∂R/∂σ, ∂d/∂σ)` pair) because a censored row's `p = g2` does not go
    // through `dp_dv` at all — its own kernel is central-differenced directly, mirroring
    // `sens_outer_gradient::sigma_block`'s censored `kern_at` closure. A FREM
    // pseudo-observation row's variance is the dedicated `EPSCOV²` override, not
    // `error_spec`, so it gets its own branch (mirrors `sigma_block`'s `frem_row` arm) —
    // `d ≡ 0` there since the override is constant in `f`. Censored takes priority over the
    // FREM/quantified branches, matching `sigma_block`'s row-dispatch order.
    // Indexed by `[s][j]`, not built with `.push` — a censored row skips the quantified
    // `sigma_row_derivs_g` write and a quantified row skips the censored `_cz`/`_cm` writes,
    // so `.push` would desync the vector's position from the observation index `j` on any
    // subject mixing both row kinds. Pre-sized and written by index instead.
    let mut sigma_row_derivs: Vec<Vec<f64>> = vec![vec![0.0; n_obs]; n_sigma];
    // Companion `∂g_j/∂σ_s` for a quantified row's `iiv_on_ruv` term (`g = d/R`), and
    // `∂(ruv_cz)_j/∂σ_s`/`∂(ruv_cm)_j/∂σ_s` for a censored row's (module doc). All three are
    // only populated, and only read, when `core.ruv.is_some()` — the uncommon case — so their
    // `3 · n_sigma` heap allocations are skipped otherwise, on the path this module exists to
    // make cheap.
    let ruv_active = core.ruv.is_some();
    let mut sigma_row_derivs_g: Vec<Vec<f64>> = if ruv_active {
        vec![vec![0.0; n_obs]; n_sigma]
    } else {
        Vec::new()
    };
    let mut sigma_row_derivs_cz: Vec<Vec<f64>> = if ruv_active {
        vec![vec![0.0; n_obs]; n_sigma]
    } else {
        Vec::new()
    };
    let mut sigma_row_derivs_cm: Vec<Vec<f64>> = if ruv_active {
        vec![vec![0.0; n_obs]; n_sigma]
    } else {
        Vec::new()
    };
    // Correlated residual (`block_sigma`): `score_core`'s `corr_diag` already builds the
    // correlation-aware `(R_jj, ∂R_jj/∂f_j)` for `et.r`/`et.d`, so the σ-direct precompute
    // must difference the SAME correlation-aware function, not the plain per-endpoint one —
    // mirrors `sens_outer_gradient::sigma_block`'s own `correlated` branch. Computed once per
    // σ slot (not per observation, unlike the FREM override lookup) since
    // `corr_residual_rd_at_sigma` already returns the whole per-subject array.
    let correlated = !params.residual_correlations.is_empty();
    let ipreds: Vec<f64> = sens.obs.iter().map(|o| o.f).collect();
    // `∂(R_jj, d_j)/∂ρ_k`, indexed `[k][j]` — [`rho_rd_terms`]'s own closed form
    // (`crate::stats::residual_error::dvar_drho`), not a central FD like the σ/magnitude
    // channels: ρ enters `R` polynomially, so no truncation-vs-round-off step to choose.
    let rho_row_derivs = rho_rd_terms(model, subject, &sens, sigma, &params.residual_correlations);
    for s in 0..n_sigma {
        let h = sigma_fd_step(sigma[s]);
        let mut sp = sigma.to_vec();
        sp[s] += h;
        let mut sm = sigma.to_vec();
        sm[s] -= h;
        let frem_override_p =
            build_frem_r_override(model.frem_config.as_ref(), &subject.fremtype, &sp);
        let frem_override_m =
            build_frem_r_override(model.frem_config.as_ref(), &subject.fremtype, &sm);
        let corr_at_sp = correlated.then(|| {
            corr_residual_rd_at_sigma(model, subject, &ipreds, &sp, &params.residual_correlations)
        });
        let corr_at_sm = correlated.then(|| {
            corr_residual_rd_at_sigma(model, subject, &ipreds, &sm, &params.residual_correlations)
        });
        for (j, o) in sens.obs.iter().enumerate() {
            let cmt = err_keys[j];
            let et = &core.et[j];
            if et.censored {
                let y = subject.observations[j];
                let f = o.f;
                let kern_at = |sa: &[f64]| -> (f64, f64, f64) {
                    let r = model.error_spec.variance_at(cmt, f, sa) * core.ruv_scale;
                    let d = model.error_spec.dvar_df(cmt, f, sa) * core.ruv_scale;
                    let d2 = model.error_spec.d2var_df2(cmt, f, sa) * core.ruv_scale;
                    let (_g1, g2, cz, cm) = m3_censored_outer(y, f, r, d, d2, et.cens_sign);
                    (g2, cz, cm)
                };
                let (g2p, czp, cmp) = kern_at(&sp);
                let (g2m, czm, cmm) = kern_at(&sm);
                sigma_row_derivs[s][j] = (g2p - g2m) / (2.0 * h);
                // `ruv_cz`/`ruv_cm` only matter under `iiv_on_ruv`; skip the extra kernel
                // evaluations' results otherwise (already paid for by `kern_at` above, but
                // no downstream reader when `core.ruv.is_none()`).
                if core.ruv.is_some() {
                    sigma_row_derivs_cz[s][j] = (czp - czm) / (2.0 * h);
                    sigma_row_derivs_cm[s][j] = (cmp - cmm) / (2.0 * h);
                }
                continue;
            }
            let frem_p = frem_override_p
                .as_ref()
                .and_then(|ov| ov.get(j))
                .and_then(|v| *v);
            let frem_m = frem_override_m
                .as_ref()
                .and_then(|ov| ov.get(j))
                .and_then(|v| *v);
            let (r_sig, d_sig) = if let (Some(vp), Some(vm)) = (frem_p, frem_m) {
                ((vp - vm) / (2.0 * h), 0.0)
            } else if let (Some((rp, dp)), Some((rm, dm))) = (&corr_at_sp, &corr_at_sm) {
                // Correlation-aware `R_jj`/`∂R_jj/∂f_j` — `score_core`'s own `corr_diag`
                // branch does not apply `ruv_scale` either (mutually exclusive with
                // `iiv_on_ruv` in practice, matching `anchor_hessian`'s base assembly).
                ((rp[j] - rm[j]) / (2.0 * h), (dp[j] - dm[j]) / (2.0 * h))
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
            sigma_row_derivs[s][j] = dp_dv(et.r, et.d, r_sig, d_sig);
            if core.ruv.is_some() {
                sigma_row_derivs_g[s][j] = dg_dv(et.r, et.d, r_sig, d_sig);
            }
        }
    }

    let l = &params.omega.chol;
    let mut out: Vec<DMatrix<f64>> = Vec::with_capacity(x.len());
    let mut psi = vec![0.0f64; n_eta];
    for k in 0..x.len() {
        let mut dh = DMatrix::<f64>::zeros(n_eta, n_eta);
        let (mut theta_idx, mut dtheta) = (usize::MAX, 0.0f64);
        let (mut sigma_idx, mut dsigma) = (usize::MAX, 0.0f64);
        let (mut rho_idx, mut drho) = (usize::MAX, 0.0f64);

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
        } else if k < n_theta + entries.len() + n_sigma {
            sigma_idx = k - n_theta - entries.len();
            dsigma = sigma[sigma_idx]; // σ packs as ln σ ⇒ dσ/dx = σ.
        } else {
            rho_idx = k - n_theta - entries.len() - n_sigma;
            // ρ packs through `atanh` (`pack_rho`), so `dρ/dz = 1 − ρ²` (`rho_chain`) — the
            // same Fisher-z chain `subject_theta_gradient`'s own ρ block applies.
            drho = rho_chain(params.residual_correlations[rho_idx].rho);
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
                dp += sigma_row_derivs[sigma_idx][j] * dsigma;
            }
            if rho_idx < n_rho {
                let (r_rho, d_rho) = rho_row_derivs[rho_idx][j];
                dp += dp_dv(et.r, et.d, r_rho, d_rho) * drho;
            }

            // `iiv_on_ruv`'s residual variance scale `s = exp(2·η_ruv)` moves `p` (quantified),
            // `p = g2` (censored), and — for a censored row — `ruv_cz`/`ruv_cm` too, even at
            // fixed `f`, since `r`/`d`/`d2` all carry `s`. Quantified `p`: `p(s) = 1/(s·R₀) +
            // ½(D₀/R₀)²` (the `(d/R)²` term is scale-invariant, so only the `1/R` piece
            // survives), giving `dp/ds = −1/(r·s)`; chained through `ds/dx_k = 2·s·bₖ[rr]` this
            // collapses to the closed form `−2·bₖ[rr]/r` — no `s` survives the product. `g2`,
            // `ruv_cz`, `ruv_cm` have no such closed form, so `∂/∂s` is central-differenced
            // directly by rescaling `(r, d, d2)` together — one FD pair serves all three,
            // reused by the `(rr, l)` block below. Skipped for a FREM row, whose `R = EPSCOV²`
            // never carries `ruv_scale` at all (`score_core`'s own `(v, 0.0, 0.0)` branch).
            let mut censored_ruv_scale_response: Option<(f64, f64)> = None;
            if let Some(rr) = core.ruv {
                let frem_var = frem_r_base
                    .as_ref()
                    .and_then(|ov| ov.get(j))
                    .and_then(|v| *v);
                if frem_var.is_none() {
                    if et.censored {
                        let s = core.ruv_scale;
                        // `sigma_fd_step`'s relative-with-floor step, not an absolute `1e-6`:
                        // `s = exp(2·η̂_ruv)` can be far below 1 for an outer-loop excursion
                        // (`η̂_ruv < -6.9` ⇒ `s < 1e-6`), where an absolute step made `s - hs`
                        // negative and handed `m3_censored_outer` a negative variance.
                        let hs = sigma_fd_step(s);
                        let y = subject.observations[j];
                        let f = o.f;
                        let at = |scale: f64| -> (f64, f64, f64) {
                            let ratio = scale / s;
                            let (_g1, g2, cz, cm) = m3_censored_outer(
                                y,
                                f,
                                et.r * ratio,
                                et.d * ratio,
                                d2_row[j] * ratio,
                                et.cens_sign,
                            );
                            (g2, cz, cm)
                        };
                        let (g2p, czp, cmp) = at(s + hs);
                        let (g2m, czm, cmm) = at(s - hs);
                        let ds_dx = 2.0 * s * bk[rr];
                        let dg2_ds = (g2p - g2m) / (2.0 * hs);
                        let dcz_ds = (czp - czm) / (2.0 * hs);
                        let dcm_ds = (cmp - cmm) / (2.0 * hs);
                        dp += dg2_ds * ds_dx;
                        censored_ruv_scale_response = Some((dcz_ds * ds_dx, dcm_ds * ds_dx));
                    } else {
                        dp += -2.0 * bk[rr] / et.r;
                    }
                }
            }

            for i in 0..n_eta {
                for m in 0..n_eta {
                    dh[(i, m)] += dp * a[i] * a[m] + p * (psi[i] * a[m] + a[i] * psi[m]);
                }
            }

            // `iiv_on_ruv` (#474): `H̃[rr,rr]` is constant for a quantified row (`+= 2.0`), so
            // contributes nothing there, but genuinely moves for a censored row (`+= ruv_cz`,
            // handled just above via `dp`, since `dh`'s generic accumulation below writes
            // `dh[(rr,rr)] += dp*a[rr]*a[rr] = 0` — the censored `dp` contribution needs to
            // land on `(rr,rr)` directly, added below). `H̃[rr,l] += g·a_l` (quantified,
            // `g = d/R`, scale-free) or `ruv_cm·a_l` (censored) for `l ≠ rr` — `a[rr] = 0`
            // structurally means the generic `p·aaᵀ` loop above already left `dh`'s `rr`
            // row/col untouched, so this only ADDS, never overwrites. Skipped for a FREM
            // pseudo-observation (module doc); a censored row is never a FREM row.
            if let Some(rr) = core.ruv {
                let frem_var = frem_r_base
                    .as_ref()
                    .and_then(|ov| ov.get(j))
                    .and_then(|v| *v);
                if frem_var.is_none() {
                    if et.censored {
                        let mut dcz = core.cens_dcz_df[j] * phi;
                        let mut dcm = core.cens_dcm_df[j] * phi;
                        if sigma_idx < n_sigma {
                            dcz += sigma_row_derivs_cz[sigma_idx][j] * dsigma;
                            dcm += sigma_row_derivs_cm[sigma_idx][j] * dsigma;
                        }
                        if let Some((dcz_s, dcm_s)) = censored_ruv_scale_response {
                            dcz += dcz_s;
                            dcm += dcm_s;
                        }
                        dh[(rr, rr)] += dcz;
                        for l in 0..n_eta {
                            if l == rr {
                                continue;
                            }
                            let contrib = dcm * a[l] + et.ruv_cm * psi[l];
                            dh[(rr, l)] += contrib;
                            dh[(l, rr)] += contrib;
                        }
                    } else {
                        let (r, d) = (et.r, et.d);
                        let mut dg = dg_dv(r, d, d, d2_row[j]) * phi;
                        if theta_idx < n_theta && !et.dr_dtheta.is_empty() {
                            let (rv, dv) = (et.dr_dtheta[theta_idx], et.dd_dtheta[theta_idx]);
                            if rv != 0.0 || dv != 0.0 {
                                dg += dg_dv(r, d, rv, dv) * dtheta;
                            }
                        }
                        if sigma_idx < n_sigma {
                            dg += sigma_row_derivs_g[sigma_idx][j] * dsigma;
                        }
                        let g = d / r;
                        for l in 0..n_eta {
                            if l == rr {
                                continue;
                            }
                            let contrib = dg * a[l] + g * psi[l];
                            dh[(rr, l)] += contrib;
                            dh[(l, rr)] += contrib;
                        }
                    }
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

/// The stacked-system twin of [`subject_htilde_dx`] for IOV models.
///
/// The IOV FOCEI marginal is exactly the ordinary FOCEI Laplace objective over the augmented
/// system `b = [η, κ₁..κ_K]`, prior `Σ_b = Ω_bsv ⊕ K·Ω_iov`
/// (`sens_outer_gradient::subject_theta_gradient_iov`'s own doc) — so [`score_core`] and the
/// per-observation `p`/`β` machinery this module already builds on need only `n_eta → n_st`
/// (the stacked dimension) and the joint `Ω⁻¹` to keep working; no residual-chain math is new.
/// What is genuinely new: the packed layout is `[θ, Ω_bsv lower-tri, σ, Ω_iov lower-tri]`
/// (σ sits BETWEEN the two Ω segments, not after both — mirrors `subject_eta_dx_iov`'s own
/// `sigma_start`/`iov_start`), the sensitivity provider is
/// [`crate::sens::provider::subject_sensitivities_iov`], and a BSV or IOV Ω coordinate's
/// `dΩ⁻¹/dx` reads a column of the *joint* Cholesky factor
/// ([`crate::estimation::parameterization::block_chol_full`]) rather than the bare block's —
/// an IOV entry moves `K` replicated diagonal blocks of `Σ_b` at once, so its `dΩ⁻¹/dx` sums
/// `K` such contributions ([`omega_inv_deriv`]).
///
/// `omega_inv` must be the joint `Σ_b⁻¹` (`Stack::omega_joint_inv` — the caller passes this
/// unconditionally, IOV or not, never reduced to the bare η block); `b_hat`/`db_dx` the
/// stacked mode and its response
/// ([`crate::estimation::sens_outer_gradient::subject_eta_dx_iov`]).
///
/// # Scope
///
/// Mirrors [`subject_htilde_dx`]'s scope for everything that carries over unchanged (M3-BLOQ
/// including its σ-direct derivative, custom/TV σ magnitude, `iiv_on_ruv` including combined
/// with M3-BLOQ, LTBS, `ExpressionScale`, FREM), minus:
///
/// * **mixture** — a mixture's subpopulation overrides are a different packed segment again,
///   not the stacked-prior story IOV is; kept as its own decline;
/// * **correlated residuals (`block_sigma`)** — mutually exclusive with IOV in `score_core`'s
///   own scope already, so declined here too rather than building dead code for an
///   unreachable combination.
pub(crate) fn subject_htilde_dx_iov(
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
    let n_sigma = params.sigma.values.len();
    let n_eta_bsv = model.n_eta;
    let n_iov = model.n_kappa;
    if n_eta_bsv == 0 || n_iov == 0 {
        return None;
    }
    let k_occ = crate::stats::likelihood::iov_occasion_groups(subject).len();
    let n_st = n_eta_bsv + k_occ * n_iov;
    let omega_iov = params.omega_iov.as_ref()?;

    let bsv_entries = lower_tri_entries(n_eta_bsv, params.omega.diagonal);
    let iov_entries = lower_tri_entries(n_iov, omega_iov.diagonal);
    let omega_start = n_theta;
    let sigma_start = omega_start + bsv_entries.len();
    let iov_start = sigma_start + n_sigma;
    let expected_len = iov_start + iov_entries.len();

    if packed_len(template) != expected_len
        || x.len() != expected_len
        || db_dx.len() != x.len()
        || b_hat.len() != n_st
        || template.mixture.is_some()
        || !params.residual_correlations.is_empty()
        || !analytic_score_supported(model)
    {
        return None;
    }

    let sens =
        crate::sens::provider::subject_sensitivities_iov(model, subject, &params.theta, b_hat)?;
    let n_obs = subject.observations.len();
    if n_obs == 0 || sens.obs.len() != n_obs {
        return None;
    }
    let core = score_core(
        model,
        subject,
        params,
        &sens,
        n_st,
        omega_inv,
        b_hat,
        model.residual_error_eta,
    )?;

    let sigma = &params.sigma.values;
    let err_keys = model.error_spec.obs_keys(subject);
    let frem_r_base = build_frem_r_override(model.frem_config.as_ref(), &subject.fremtype, sigma);
    let d2_row: Vec<f64> = if core.ruv.is_some() {
        sens.obs
            .iter()
            .enumerate()
            .map(|(j, o)| {
                let cmt = err_keys[j];
                let mult_row: Option<&[f64]> = core
                    .mult
                    .as_ref()
                    .and_then(|m| m.get(j))
                    .map(|v| v.as_slice());
                let (_, _, d2) = residual_rd2(&model.error_spec, cmt, o.f, sigma, mult_row);
                d2 * core.ruv_scale
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut sigma_row_derivs: Vec<Vec<f64>> = vec![vec![0.0; n_obs]; n_sigma];
    // `sigma_row_derivs_g`/`_cz`/`_cm` are only populated, and only read, when
    // `core.ruv.is_some()` — the uncommon case — so skip their `3 · n_sigma` heap allocations
    // otherwise, on the path this module exists to make cheap.
    let ruv_active = core.ruv.is_some();
    let mut sigma_row_derivs_g: Vec<Vec<f64>> = if ruv_active {
        vec![vec![0.0; n_obs]; n_sigma]
    } else {
        Vec::new()
    };
    let mut sigma_row_derivs_cz: Vec<Vec<f64>> = if ruv_active {
        vec![vec![0.0; n_obs]; n_sigma]
    } else {
        Vec::new()
    };
    let mut sigma_row_derivs_cm: Vec<Vec<f64>> = if ruv_active {
        vec![vec![0.0; n_obs]; n_sigma]
    } else {
        Vec::new()
    };
    for s in 0..n_sigma {
        let h = sigma_fd_step(sigma[s]);
        let mut sp = sigma.to_vec();
        sp[s] += h;
        let mut sm = sigma.to_vec();
        sm[s] -= h;
        let frem_override_p =
            build_frem_r_override(model.frem_config.as_ref(), &subject.fremtype, &sp);
        let frem_override_m =
            build_frem_r_override(model.frem_config.as_ref(), &subject.fremtype, &sm);
        for (j, o) in sens.obs.iter().enumerate() {
            let cmt = err_keys[j];
            let et = &core.et[j];
            if et.censored {
                let y = subject.observations[j];
                let f = o.f;
                let kern_at = |sa: &[f64]| -> (f64, f64, f64) {
                    let r = model.error_spec.variance_at(cmt, f, sa) * core.ruv_scale;
                    let d = model.error_spec.dvar_df(cmt, f, sa) * core.ruv_scale;
                    let d2 = model.error_spec.d2var_df2(cmt, f, sa) * core.ruv_scale;
                    let (_g1, g2, cz, cm) = m3_censored_outer(y, f, r, d, d2, et.cens_sign);
                    (g2, cz, cm)
                };
                let (g2p, czp, cmp) = kern_at(&sp);
                let (g2m, czm, cmm) = kern_at(&sm);
                sigma_row_derivs[s][j] = (g2p - g2m) / (2.0 * h);
                if core.ruv.is_some() {
                    sigma_row_derivs_cz[s][j] = (czp - czm) / (2.0 * h);
                    sigma_row_derivs_cm[s][j] = (cmp - cmm) / (2.0 * h);
                }
                continue;
            }
            let frem_p = frem_override_p
                .as_ref()
                .and_then(|ov| ov.get(j))
                .and_then(|v| *v);
            let frem_m = frem_override_m
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
            sigma_row_derivs[s][j] = dp_dv(et.r, et.d, r_sig, d_sig);
            if core.ruv.is_some() {
                sigma_row_derivs_g[s][j] = dg_dv(et.r, et.d, r_sig, d_sig);
            }
        }
    }

    let l_bsv = &params.omega.chol;
    let l_iov = &omega_iov.chol;
    let l_full = block_chol_full(l_bsv, l_iov, k_occ, n_eta_bsv, n_iov);

    let mut out: Vec<DMatrix<f64>> = Vec::with_capacity(x.len());
    let mut psi = vec![0.0f64; n_st];
    for kx in 0..x.len() {
        let mut dh = DMatrix::<f64>::zeros(n_st, n_st);
        let (mut theta_idx, mut dtheta) = (usize::MAX, 0.0f64);
        let (mut sigma_idx, mut dsigma) = (usize::MAX, 0.0f64);

        if kx < n_theta {
            theta_idx = kx;
            dtheta = theta_dx_chain(template, &params.theta, kx);
        } else if kx < omega_start + bsv_entries.len() {
            let (row, col) = bsv_entries[kx - omega_start];
            let chain = if row == col { l_bsv[(row, row)] } else { 1.0 };
            let v = l_full.column(col).into_owned();
            let u: DVector<f64> = omega_inv.column(row).into_owned();
            dh = omega_inv_deriv(omega_inv, &u, &v) * chain;
        } else if kx < iov_start {
            sigma_idx = kx - sigma_start;
            dsigma = sigma[sigma_idx];
        } else {
            let (i, j) = iov_entries[kx - iov_start];
            let chain = if i == j { l_iov[(i, i)] } else { 1.0 };
            let mut acc = DMatrix::<f64>::zeros(n_st, n_st);
            for occ in 0..k_occ {
                let row = n_eta_bsv + occ * n_iov + i;
                let col = n_eta_bsv + occ * n_iov + j;
                let v = l_full.column(col).into_owned();
                let u: DVector<f64> = omega_inv.column(row).into_owned();
                acc += omega_inv_deriv(omega_inv, &u, &v);
            }
            dh = acc * chain;
        }

        let bk = &db_dx[kx];
        for (j, o) in sens.obs.iter().enumerate() {
            let a = o.df_deta.as_slice();
            let big_a = o.d2f_deta2.as_slice();
            let et = &core.et[j];
            let p = et.p;

            let mut phi = 0.0;
            for r in 0..n_st {
                phi += a[r] * bk[r];
            }
            if theta_idx < n_theta {
                phi += o.df_dtheta[theta_idx] * dtheta;
            }
            for i in 0..n_st {
                let mut v = 0.0;
                for r in 0..n_st {
                    v += big_a[i * n_st + r] * bk[r];
                }
                if theta_idx < n_theta {
                    v += o.d2f_deta_dtheta[i * n_theta + theta_idx] * dtheta;
                }
                psi[i] = v;
            }

            let mut dp = et.beta * phi;
            if theta_idx < n_theta && !et.dr_dtheta.is_empty() {
                let (rv, dv) = (et.dr_dtheta[theta_idx], et.dd_dtheta[theta_idx]);
                if rv != 0.0 || dv != 0.0 {
                    dp += dp_dv(et.r, et.d, rv, dv) * dtheta;
                }
            }
            if sigma_idx < n_sigma {
                dp += sigma_row_derivs[sigma_idx][j] * dsigma;
            }

            let mut censored_ruv_scale_response: Option<(f64, f64)> = None;
            if let Some(rr) = core.ruv {
                let frem_var = frem_r_base
                    .as_ref()
                    .and_then(|ov| ov.get(j))
                    .and_then(|v| *v);
                if frem_var.is_none() {
                    if et.censored {
                        let s = core.ruv_scale;
                        // `sigma_fd_step`'s relative-with-floor step, not an absolute `1e-6`:
                        // `s = exp(2·η̂_ruv)` can be far below 1 for an outer-loop excursion
                        // (`η̂_ruv < -6.9` ⇒ `s < 1e-6`), where an absolute step made `s - hs`
                        // negative and handed `m3_censored_outer` a negative variance.
                        let hs = sigma_fd_step(s);
                        let y = subject.observations[j];
                        let f = o.f;
                        let at = |scale: f64| -> (f64, f64, f64) {
                            let ratio = scale / s;
                            let (_g1, g2, cz, cm) = m3_censored_outer(
                                y,
                                f,
                                et.r * ratio,
                                et.d * ratio,
                                d2_row[j] * ratio,
                                et.cens_sign,
                            );
                            (g2, cz, cm)
                        };
                        let (g2p, czp, cmp) = at(s + hs);
                        let (g2m, czm, cmm) = at(s - hs);
                        let ds_dx = 2.0 * s * bk[rr];
                        let dg2_ds = (g2p - g2m) / (2.0 * hs);
                        let dcz_ds = (czp - czm) / (2.0 * hs);
                        let dcm_ds = (cmp - cmm) / (2.0 * hs);
                        dp += dg2_ds * ds_dx;
                        censored_ruv_scale_response = Some((dcz_ds * ds_dx, dcm_ds * ds_dx));
                    } else {
                        dp += -2.0 * bk[rr] / et.r;
                    }
                }
            }

            for i in 0..n_st {
                for m in 0..n_st {
                    dh[(i, m)] += dp * a[i] * a[m] + p * (psi[i] * a[m] + a[i] * psi[m]);
                }
            }

            if let Some(rr) = core.ruv {
                let frem_var = frem_r_base
                    .as_ref()
                    .and_then(|ov| ov.get(j))
                    .and_then(|v| *v);
                if frem_var.is_none() {
                    if et.censored {
                        let mut dcz = core.cens_dcz_df[j] * phi;
                        let mut dcm = core.cens_dcm_df[j] * phi;
                        if sigma_idx < n_sigma {
                            dcz += sigma_row_derivs_cz[sigma_idx][j] * dsigma;
                            dcm += sigma_row_derivs_cm[sigma_idx][j] * dsigma;
                        }
                        if let Some((dcz_s, dcm_s)) = censored_ruv_scale_response {
                            dcz += dcz_s;
                            dcm += dcm_s;
                        }
                        dh[(rr, rr)] += dcz;
                        for l in 0..n_st {
                            if l == rr {
                                continue;
                            }
                            let contrib = dcm * a[l] + et.ruv_cm * psi[l];
                            dh[(rr, l)] += contrib;
                            dh[(l, rr)] += contrib;
                        }
                    } else {
                        let (r, d) = (et.r, et.d);
                        let mut dg = dg_dv(r, d, d, d2_row[j]) * phi;
                        if theta_idx < n_theta && !et.dr_dtheta.is_empty() {
                            let (rv, dv) = (et.dr_dtheta[theta_idx], et.dd_dtheta[theta_idx]);
                            if rv != 0.0 || dv != 0.0 {
                                dg += dg_dv(r, d, rv, dv) * dtheta;
                            }
                        }
                        if sigma_idx < n_sigma {
                            dg += sigma_row_derivs_g[sigma_idx][j] * dsigma;
                        }
                        let g = d / r;
                        for l in 0..n_st {
                            if l == rr {
                                continue;
                            }
                            let contrib = dg * a[l] + g * psi[l];
                            dh[(rr, l)] += contrib;
                            dh[(l, rr)] += contrib;
                        }
                    }
                }
            }
        }

        if dh.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let sym = 0.5 * (&dh + dh.transpose());
        out.push(sym);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::parameterization::{pack_params, unpack_params};
    use crate::estimation::sens_outer_gradient::{subject_eta_dx, subject_eta_dx_iov};
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
            model.residual_error_eta,
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

    /// M3-censored: `p = g2` comes from the `−logΦ(z)` kernel rather than the quantified
    /// `dp_dv` closed form, so the σ-direct precompute central-differences `m3_censored_outer`
    /// directly for these rows (module doc). `htilde_at`'s FD oracle goes through the exact
    /// same `score_core` the analytic route reads `p`/`β` from, so this is a real parity
    /// check, not a tautology.
    #[test]
    fn htilde_derivative_matches_fd_under_m3_censoring() {
        let (mut model, subject, template) =
            setup(MODEL, &[0.22, 11.0, 1.4], &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        model.bloq_method = crate::types::BloqMethod::M3;
        let mut subject = subject;
        let n = subject.observations.len();
        subject.cens[n - 1] = 1;
        subject.cens[n - 2] = 1;
        assert_matches_fd(&model, &subject, &template, &[0.05, -0.03, 0.08], 1e-6);
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

    /// `iiv_on_ruv`: `H̃[rr,rr]` is constant (contributes nothing) and `H̃[rr,l]`'s `g = d/R`
    /// quotient rule (`dg_dv`, module doc) must match `htilde_at`'s FD oracle, which reads
    /// the same `(g, 2.0)` structure straight from `score_core`.
    #[test]
    fn htilde_derivative_matches_fd_under_iiv_on_ruv() {
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
        assert_matches_fd(&model, &subject, &template, &[0.05, -0.03, 0.08, 0.0], 1e-5);
    }

    /// `iiv_on_ruv` combined with an M3-censored row: `H̃`'s residual-eta entries are the
    /// `ruv_cz`/`ruv_cm` chain instead of `(g, 2.0)` (module doc) — reuses `ScoreCore`'s own
    /// `cens_dcz_df`/`cens_dcm_df` for the structural half, central-differences the matching
    /// σ-direct pair.
    #[test]
    fn htilde_derivative_matches_fd_under_iiv_on_ruv_with_censored_row() {
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
        let mut model = parse_model_string(RUV_MODEL).expect("parse");
        model.bloq_method = crate::types::BloqMethod::M3;
        let theta = [0.22, 11.0, 1.4];
        let mut subject = fixture_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let n = subject.observations.len();
        subject.cens[n - 1] = 1;
        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        assert_matches_fd(&model, &subject, &template, &[0.05, -0.03, 0.08, 0.0], 1e-5);
    }

    /// Same fixture, but with `η̂_ruv` pushed negative enough (`ruv_scale =
    /// exp(2·η̂_ruv) ≈ 8.3e-7`, just under the old fixed `1e-6` half-step) that the
    /// σ-direct precompute's *absolute* `1e-6` half-step used to make `s - hs` negative —
    /// a negative variance handed to `m3_censored_outer`, `NaN` from the inner `sqrt`, and
    /// the whole subject silently declining to the FD anchor sweep. Regression test for the
    /// fix: the precompute now uses `sigma_fd_step`'s relative-with-floor step, the same one
    /// `sigma_block` uses.
    ///
    /// Bypasses `assert_matches_fd`'s usual `subject_eta_dx` call for `db_dx`: at this
    /// `η̂_ruv`, `subject_eta_dx`'s OWN inner Hessian (an unrelated computation this test
    /// doesn't exercise) is too ill-conditioned for `nalgebra`'s Cholesky, and declines —
    /// not a defect of the fix under test. `subject_htilde_dx` and the `htilde_at` FD
    /// oracle are both generic in `db_dx` (never invert it, just use it to move `b` along a
    /// caller-supplied direction), so any fixed vector makes a valid parity check as long as
    /// both sides use the same one. Pick one with a nonzero η_ruv component so the fixed
    /// `ds_dx = 2·s·bₖ[rr]` term is actually exercised.
    #[test]
    fn htilde_derivative_matches_fd_under_iiv_on_ruv_with_censored_row_and_tiny_ruv_scale() {
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
        let mut model = parse_model_string(RUV_MODEL).expect("parse");
        model.bloq_method = crate::types::BloqMethod::M3;
        let theta = [0.22, 11.0, 1.4];
        let mut subject = fixture_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let n = subject.observations.len();
        subject.cens[n - 1] = 1;
        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let b_hat = [0.05, -0.03, 0.08, -7.0];
        // Zero the censored row's residual at `b_hat` (`z = eps/√R = 0` regardless of how
        // small `R` gets): `R` already carries `ruv_scale ≈ 8.3e-7`, so any of the fixture's
        // usual nonzero residuals put `|z|` in the hundreds — the extreme Gaussian tail's own
        // curvature there is sharp enough that no reasonably-sized FD step recovers a stable
        // derivative, which is a limit of central-difference validation at that regime, not
        // a defect of the fix under test.
        let f_last = crate::pk::compute_predictions_with_tv(&model, &subject, &theta, &b_hat[..3]);
        subject.observations[n - 1] = f_last[n - 1];

        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let omega_inv = params.omega.inv.clone();
        let n_eta = params.omega.dim();
        // `db_dx[k][rr]` must move `b_ruv` by much less than `s ≈ 8.3e-7` itself over the
        // outer FD step (`~1e-5`), or the *outer* central difference sees the kernel's own
        // sharp curvature at that scale and stops being a valid linear approximation —
        // `1e-4` keeps the implied `Δb_ruv` two orders below `s`.
        let db_dx: Vec<DVector<f64>> = x
            .iter()
            .map(|_| DVector::from_vec(vec![0.0, 0.0, 0.0, 1e-4]))
            .collect();

        let analytic = subject_htilde_dx(
            &model, &subject, &params, &template, &omega_inv, &x, &b_hat, &db_dx,
        )
        .expect("in scope");

        // `s ≈ 8.3e-7` is below `sigma_fd_step`'s own `1e-6` relative-step floor, so it falls
        // back to a `0.5·s` HALF-step (the guard that keeps `s - hs > 0`, the fix this test
        // pins) rather than a small one — central-difference truncation error at a 50%
        // relative step is naturally percent-scale, not the `1e-5`–`1e-6` this module's other
        // parity tests hold in the well-conditioned regime. Measured worst case at this input:
        // ~1.17e-2 (coord 3, entry (0,0)); `2e-2` keeps headroom without being loose enough to
        // pass a broken derivative (an unfixed `NaN`/panicking entry fails outright, and this
        // regime has no smaller-`hs` alternative to tighten against — `sigma_fd_step` already
        // is the smallest step that keeps the argument positive).
        let tol = 2e-2;
        for k in 0..x.len() {
            let step = 1e-5 * (1.0 + x[k].abs());
            let mut xp = x.clone();
            xp[k] += step;
            let mut xm = x.clone();
            xm[k] -= step;
            let bp: Vec<f64> = (0..n_eta).map(|i| b_hat[i] + step * db_dx[k][i]).collect();
            let bm: Vec<f64> = (0..n_eta).map(|i| b_hat[i] - step * db_dx[k][i]).collect();
            let fd = (htilde_at(&model, &subject, &template, &xp, &bp)
                - htilde_at(&model, &subject, &template, &xm, &bm))
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

    /// Correlated residual (`block_sigma`): the σ-direct precompute must difference the
    /// correlation-aware `(R_jj, ∂R_jj/∂f_j)` (`corr_residual_rd_at_sigma`), not the plain
    /// per-endpoint one, once the base `H̃` itself already reads `score_core`'s `corr_diag`
    /// (module doc). Within-observation correlation (`combined` error) keeps `R` diagonal
    /// across observations, so this is the tractable case `score_core` admits.
    #[test]
    fn htilde_derivative_matches_fd_under_block_sigma() {
        const BLOCK_SIGMA_MODEL: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 10.0)
  theta TVV(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.05, 1.00]
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;
        let (model, subject, template) = setup(
            BLOCK_SIGMA_MODEL,
            &[1.1, 11.0],
            &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0],
        );
        assert_matches_fd(&model, &subject, &template, &[0.05, -0.03], 1e-5);
    }

    fn iov_fixture_subject(model: &CompiledModel, theta: &[f64]) -> Subject {
        let obs_times = vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0];
        let occasions = vec![1u32, 1, 1, 2, 2, 2];
        let n = obs_times.len();
        let mut subject = Subject {
            id: "1".to_string(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times,
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
            occasions,
            obs_l2: Vec::new(),
            dose_occasions: vec![1, 2],
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let preds = crate::pk::predict_iov(
            model,
            &subject,
            theta,
            &[0.12, -0.08, 0.2],
            &[vec![0.05], vec![-0.07]],
        );
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        subject
    }

    /// Joint `Σ_b⁻¹` the caller (`agq.rs`'s `Stack::omega_joint_inv`) would build: block
    /// diagonal `Ω_bsv ⊕ K·Ω_iov`.
    fn iov_joint_omega_inv(params: &ModelParameters, k: usize) -> DMatrix<f64> {
        let omega_iov = params.omega_iov.as_ref().expect("IOV params");
        let block = crate::stats::likelihood::build_block_diag_omega(
            &params.omega.matrix,
            &omega_iov.matrix,
            k,
        );
        block.cholesky().expect("PD joint omega").inverse()
    }

    /// IOV twin of `htilde_at`: `H̃(x)` built from the stacked provider, at the joint
    /// dimension/prior — exactly the function `subject_htilde_dx_iov` claims to
    /// differentiate.
    fn htilde_at_iov(
        model: &CompiledModel,
        subject: &Subject,
        template: &ModelParameters,
        x: &[f64],
        b: &[f64],
        k: usize,
    ) -> DMatrix<f64> {
        let params = unpack_params(x, template);
        let omega_inv = iov_joint_omega_inv(&params, k);
        let sens =
            crate::sens::provider::subject_sensitivities_iov(model, subject, &params.theta, b)
                .expect("iov sens");
        score_core(
            model,
            subject,
            &params,
            &sens,
            b.len(),
            &omega_inv,
            b,
            model.residual_error_eta,
        )
        .expect("score_core")
        .htilde
    }

    fn assert_matches_fd_iov(
        model: &CompiledModel,
        subject: &Subject,
        template: &ModelParameters,
        b_hat: &[f64],
        k: usize,
        tol: f64,
    ) {
        let x = pack_params(template);
        let params = unpack_params(&x, template);
        let omega_inv = iov_joint_omega_inv(&params, k);
        let db_dx = subject_eta_dx_iov(model, subject, template, &x, b_hat).expect("eta_dx_iov");

        let analytic = subject_htilde_dx_iov(
            model, subject, &params, template, &omega_inv, &x, b_hat, &db_dx,
        )
        .expect("in scope");

        let n_st = b_hat.len();
        for kx in 0..x.len() {
            let step = 1e-5 * (1.0 + x[kx].abs());
            let mut xp = x.clone();
            xp[kx] += step;
            let mut xm = x.clone();
            xm[kx] -= step;
            let bp: Vec<f64> = (0..n_st).map(|i| b_hat[i] + step * db_dx[kx][i]).collect();
            let bm: Vec<f64> = (0..n_st).map(|i| b_hat[i] - step * db_dx[kx][i]).collect();
            let fd = (htilde_at_iov(model, subject, template, &xp, &bp, k)
                - htilde_at_iov(model, subject, template, &xm, &bm, k))
                / (2.0 * step);
            let scale = fd.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            for i in 0..n_st {
                for m in 0..n_st {
                    assert!(
                        (analytic[kx][(i, m)] - fd[(i, m)]).abs() / scale < tol,
                        "coord {kx} entry ({i},{m}): analytic {} vs FD {} (scale {scale})",
                        analytic[kx][(i, m)],
                        fd[(i, m)]
                    );
                }
            }
        }
    }

    #[test]
    fn htilde_derivative_matches_fd_under_iov() {
        const IOV_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let model = parse_model_string(IOV_MODEL).expect("parse");
        let theta = [0.22, 11.0, 1.4];
        let subject = iov_fixture_subject(&model, &theta);
        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        // n_st = n_eta_bsv(3) + k(2 occasions) * n_iov(1) = 5.
        let b_hat = [0.05, -0.03, 0.08, 0.02, -0.015];
        assert_matches_fd_iov(&model, &subject, &template, &b_hat, 2, 1e-5);
    }
}
