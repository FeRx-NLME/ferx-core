//! Exact FOCEI and FOCE outer gradients for a subject whose residual covariance `R`
//! couples two observation rows (#1960): a paired `block_sigma` (total/unbound assays at one time,
//! or one `L2` group) or a `correlation cs|ar1|car1(RHO) by COL` block.
//!
//! The scalar assembly in the parent module needs `R` diagonal — its per-observation
//! `(r, d, d2)` reduction is exact only then (`corr_residual_diag`). This is the dense
//! generalisation of the same objective, [`foce_subject_nll_interaction_dense_cov`]:
//!
//! ```text
//!   Fᵢ = ½ [ εᵀR⁻¹ε + ln|R| + η̂ᵀΩ⁻¹η̂ + ln|Ω| + ln|H̃| ]
//!   H̃  = JᵀR⁻¹J + ½B + Ω⁻¹,   B_kl = tr(R⁻¹M_k R⁻¹M_l),   M_k = Σ_m J_mk ∂R/∂f_m
//! ```
//!
//! with `J = ∂f/∂η`. Every derivative the gradient needs — `∂Fᵢ/∂ζ` at fixed η̂, the
//! inner gradient's `∂/∂ζ` and `∂/∂η` (the mixed block and the true inner Hessian), and
//! `∂log|H̃|/∂η` — is one **directional derivative** of the same primitives `(f, J, R,
//! ∂R/∂f)` along one scalar coordinate, so one routine, [`Base::directional`], serves
//! all of them. A coordinate is described by what it moves ([`Dir`]):
//!
//! | coordinate | `δf`      | `δJ`              | direct `δR`, `δ(∂R/∂f)` at fixed `f` |
//! |------------|-----------|-------------------|---------------------------------------|
//! | `η_l`      | `J[:,l]`  | `∂²f/∂η∂η_l`      | —                                      |
//! | `θ_m`      | `∂f/∂θ_m` | `∂²f/∂η∂θ_m`      | a `correlation` line's ρ (it is a θ)   |
//! | `σ_s`, ρ   | —         | —                 | the builders differenced in `σ_s`, ρ   |
//!
//! The total derivative is then Almquist's Eq. 46 exactly as in the scalar path:
//! `dFᵢ/dζ = ∂Fᵢ/∂ζ + ½ ∂log|H̃|/∂η · dη̂/dζ`, `dη̂/dζ = −H⁻¹ ∂g/∂ζ`, with `g` the inner
//! gradient (zero at η̂, which is why `∂Fᵢ/∂η = ½ ∂log|H̃|/∂η`). Ω enters only through
//! `Ω⁻¹`, so its block is the parent's closed form fed this `H̃⁻¹`, `H⁻¹` and
//! `∂log|H̃|/∂η` ([`super::omega_packed_block_from`]).
//!
//! **The direct σ/ρ derivatives are central differences that are exact, not
//! approximate.** Away from the `MIN_VARIANCE` floor, `R` and `∂R/∂f` are polynomials
//! of degree two in each σ (every entry is `Σ c c σσ`, the within-row and cross-row
//! `block_sigma` terms `ρ σ_i σ_j c c`) and of degree one in a `block_sigma` ρ, so a
//! central difference of the builders *is* their derivative, up to rounding — the
//! reason a step as wide as `0.1·σ` is used, which keeps that rounding at ~1e-15. A
//! floored diagonal is constant (the visitors drop its `f`-terms too), so the
//! difference stays exact while every stencil point agrees on which rows are floored;
//! a stencil that straddles the floor is not a derivative, and the subject declines
//! (`floor_mask`).
//! Differencing the builders rather than re-deriving their entries keeps one home for
//! the residual covariance (`stats::residual_error`). A `correlation` line's
//! `C_jk(ρ) = ρ^Δ` is not a polynomial of fixed degree, so its ρ derivative is closed
//! form instead ([`crate::stats::residual_error::serial_rho_derivatives`]).
//!
//! `∂R/∂f` and `∂²R/∂f²` are carried in the column forms of
//! [`crate::stats::residual_error::compute_dr_df_columns_cov`] /
//! [`crate::stats::residual_error::compute_d2r_df2_compact_cov`] — `n×n` instead of the
//! `n³` / `n⁴` of the dense builders, which matters for a long `ar1` series.
//!
//! The FOCE (Sheiner–Beal) marginal has its own assembly,
//! [`subject_packed_gradient_foce_dense`], built from the same column forms and the same
//! exact σ/ρ differences; the FOCEI description above is
//! [`subject_packed_gradient_dense`].
//!
//! Scope: no IOV, no `iiv_on_ruv`, no FREM, no M3, no custom residual magnitude
//! (`weight =`), no SDE, no mixture, no non-Gaussian endpoint ([`dense_outer_supported`]).
//! Out of scope returns `None`, and the outer assembly keeps the reconverged
//! finite-difference salvage for the subject (#1536).
//!
//! [`foce_subject_nll_interaction_dense_cov`]: crate::stats::likelihood::foce_subject_nll_interaction_dense_cov

// Indexed loops over parallel n×n buffers read clearer than zips here.
#![allow(clippy::needless_range_loop)]

use super::{invert_inner_hessian, omega_packed_block_from, theta_dx_chain};
use crate::estimation::parameterization::{packed_fixed_mask, rho_chain, rho_packed_start};
use crate::sens::provider::SubjectSens;
use crate::stats::residual_error::{
    compute_d2r_df2_compact_cov, compute_dr_df_columns_cov, r_matrix_maybe_scaled,
    serial_rho_derivatives, ResidualCov, MIN_VARIANCE,
};
use crate::types::{BloqMethod, CompiledModel, ModelParameters, ResidualCorrelation, Subject};
use nalgebra::{DMatrix, DVector};

/// Can a model's cross-row-residual subjects take [`subject_packed_gradient_dense`]
/// (FOCEI) or [`subject_packed_gradient_foce_dense`] (FOCE) at all? The feature
/// exclusions of the module docs; the per-subject routing question is
/// [`crate::stats::residual_error::has_cross_observation_residual`]. Read by the outer
/// assembly to decide which declines are structural (#1536), so a model this is `false`
/// for keeps its reconverged salvage.
pub(crate) fn dense_outer_supported(model: &CompiledModel) -> bool {
    model.n_kappa == 0
        && model.residual_error_eta.is_none()
        && model.frem_config.is_none()
        && !matches!(model.bloq_in_force(), BloqMethod::M3)
        && !model.has_custom_ruv_magnitude()
        && !model.has_non_gaussian()
        && model.mixture.is_none()
        && !model.is_sde()
}

/// The exact per-subject FOCEI packed gradient `dFᵢ/dx` for a subject with a
/// cross-observation residual covariance, in `pack_params` order — the dense twin of
/// [`super::subject_packed_gradient`], which routes here. `None` out of scope
/// ([`dense_outer_supported`]) or where `R`, `H̃` or the inner Hessian `H` is not
/// positive definite.
pub(crate) fn subject_packed_gradient_dense(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    params: &ModelParameters,
    sens: &SubjectSens,
    eta_hat: &[f64],
    n_packed: usize,
) -> Option<Vec<f64>> {
    if !dense_outer_supported(model) {
        return None;
    }
    let n = subject.observations.len();
    let ne = model.n_eta;
    let n_theta = params.theta.len();
    if sens.obs.len() != n || eta_hat.len() != ne || n == 0 {
        return None;
    }
    let cov = model.residual_cov_at(params);
    let sigma = &params.sigma.values;
    let base = Base::new(model, subject, params, sens, cov)?;

    // η directions: ∂log|H̃|/∂η and the true inner Hessian H = ½ ∂G/∂η + Ω⁻¹.
    let omega_inv = &params.omega.inv;
    let mut g_eta = vec![0.0; ne];
    let mut h_inner = omega_inv.clone();
    for l in 0..ne {
        let df = base.j.column(l).into_owned();
        let dj = DMatrix::from_fn(n, ne, |j, k| sens.obs[j].d2f_deta2[k * ne + l]);
        let out = base.directional(&Dir::moving_f(df, dj));
        g_eta[l] = out.dlogdet;
        for k in 0..ne {
            h_inner[(k, l)] += 0.5 * out.dg[k];
        }
    }
    let h_inner = 0.5 * (&h_inner + h_inner.transpose());
    let h_inner_inv = invert_inner_hessian(h_inner)?;
    let g_eta_v = DVector::from_column_slice(&g_eta);
    let response = |out: &DirOut| -> f64 {
        // ½ ∂log|H̃|/∂η · dη̂/dζ with dη̂/dζ = −H⁻¹ (½ ∂G/∂ζ).
        let deta = -(&h_inner_inv * (0.5 * &out.dg));
        0.5 * g_eta_v.dot(&deta)
    };
    let total = |out: &DirOut| -> f64 { 0.5 * (out.ddata + out.dlogdet) + response(out) };

    let mut g = vec![0.0f64; n_packed];
    let is_fixed = fixed_coordinates(template);

    // θ: through f and J, plus a `correlation` line's ρ (a θ) directly through R.
    let serial_theta = model.residual_serial.as_ref().map(|s| s.rho_theta);
    for m in 0..n_theta {
        if is_fixed(m) {
            continue;
        }
        let df = DVector::from_fn(n, |j, _| sens.obs[j].df_dtheta[m]);
        let dj = DMatrix::from_fn(n, ne, |j, k| sens.obs[j].d2f_deta_dtheta[k * n_theta + m]);
        let mut dir = Dir::moving_f(df, dj);
        if serial_theta == Some(m) {
            let err_keys = model.error_spec.obs_keys(subject);
            let (dr, ddc) = serial_rho_derivatives(
                &model.error_spec,
                &base.f,
                err_keys.as_ref(),
                &subject.obs_times,
                &subject.obs_raw_times,
                &subject.obs_rgroup,
                sigma,
                cov,
                None,
            )?;
            dir.dr_direct = Some(dr);
            dir.ddc_direct = Some(ddc);
        }
        g[m] = total(&base.directional(&dir)) * theta_dx_chain(template, &params.theta, m);
    }

    // Ω: the parent's closed form (Ω enters only as Ω⁻¹).
    let og = omega_packed_block_from(
        omega_inv,
        &base.htilde_inv,
        &h_inner_inv,
        &g_eta,
        params,
        eta_hat,
    );
    let omega_start = n_theta;
    for (k, &v) in og.iter().enumerate() {
        g[omega_start + k] = v;
    }

    // σ: direct through R and ∂R/∂f (log-σ chain).
    let sigma_start = omega_start + og.len();
    let base_floor = floor_mask(&base.builders_at(model, subject, sigma, cov).0);
    for s in 0..sigma.len() {
        if is_fixed(sigma_start + s) {
            continue;
        }
        let h = sigma_step(sigma[s]);
        let mut sp = sigma.clone();
        sp[s] = sigma[s] + h;
        let plus = base.builders_at(model, subject, &sp, cov);
        sp[s] = sigma[s] - h;
        let minus = base.builders_at(model, subject, &sp, cov);
        same_floor(&base_floor, &plus.0, &minus.0)?;
        let dir = Dir::direct(&plus, &minus, h, n, ne);
        g[sigma_start + s] = total(&base.directional(&dir)) * sigma[s];
    }

    // ρ: the `block_sigma` off-diagonals, packed last (Fisher-z chain).
    let corrs = &params.residual_correlations;
    if !corrs.is_empty() {
        let rho_start = rho_packed_start(template);
        for c in 0..corrs.len() {
            if is_fixed(rho_start + c) {
                continue;
            }
            let h = RHO_STEP;
            let at = |rho: f64| -> (DMatrix<f64>, DMatrix<f64>) {
                let mut cs: Vec<ResidualCorrelation> = corrs.clone();
                cs[c].rho = rho;
                let cov_c = ResidualCov {
                    pairs: &cs,
                    serial: cov.serial,
                };
                base.builders_at(model, subject, sigma, cov_c)
            };
            let plus = at(corrs[c].rho + h);
            let minus = at(corrs[c].rho - h);
            same_floor(&base_floor, &plus.0, &minus.0)?;
            let dir = Dir::direct(&plus, &minus, h, n, ne);
            g[rho_start + c] = total(&base.directional(&dir)) * rho_chain(corrs[c].rho);
        }
    }
    Some(g)
}

/// The exact per-subject **FOCE** (Sheiner–Beal, non-interaction) packed gradient for a
/// subject with a cross-observation residual covariance — the dense twin of
/// [`super::subject_packed_gradient_foce`], which routes here.
///
/// The marginal is [`foce_subject_nll_standard_cov`]'s
///
/// ```text
///   Fᵢ = ½ [ ρᵀR̃⁻¹ρ + ln|R̃| ],   ρ = y − f(η̂) + Jη̂,   R̃ = JΩJᵀ + R⁰
/// ```
///
/// with `R⁰ = R(f(η = 0))` frozen at the population prediction, and η̂ the mode of the
/// frozen-variance inner objective `l(η) = ½ εᵀR⁰⁻¹ε + ½ ηᵀΩ⁻¹η` (#1722) — so `R⁰` moves
/// with θ (through `f(η = 0)` and a `correlation` line's ρ) and σ/ρ, never with η. As
/// in the FOCEI twin each coordinate is one directional derivative, of `Fᵢ` at fixed η̂
/// and of the inner gradient `g = −JᵀR⁰⁻¹ε + Ω⁻¹η`; then
/// `dFᵢ/dζ = ∂Fᵢ/∂ζ + ∂Fᵢ/∂η̂ · dη̂/dζ`, `dη̂/dζ = −H⁻¹ ∂g/∂ζ`, with the exact inner
/// Hessian `H = JᵀR⁰⁻¹J − Σ_j (R⁰⁻¹ε)_j ∂²f_j/∂η² + Ω⁻¹`. Unlike FOCEI, `Fᵢ` is not
/// stationary in η̂, so `∂Fᵢ/∂η̂` is the full coupling. Ω enters `R̃` directly here, so
/// its block is assembled in this function rather than borrowed.
///
/// [`foce_subject_nll_standard_cov`]: crate::stats::likelihood::foce_subject_nll_standard_cov
#[allow(clippy::too_many_arguments)]
pub(crate) fn subject_packed_gradient_foce_dense(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    params: &ModelParameters,
    sens: &SubjectSens,
    sens0: &SubjectSens,
    eta_hat: &[f64],
    n_packed: usize,
) -> Option<Vec<f64>> {
    if !dense_outer_supported(model) {
        return None;
    }
    let n = subject.observations.len();
    let ne = model.n_eta;
    let n_theta = params.theta.len();
    if sens.obs.len() != n || sens0.obs.len() != n || eta_hat.len() != ne || n == 0 {
        return None;
    }
    let es = &model.error_spec;
    let err_keys = es.obs_keys(subject);
    let cov = model.residual_cov_at(params);
    let sigma = &params.sigma.values;
    let f0: Vec<f64> = sens0.obs.iter().map(|o| o.f).collect();
    let r0_at = |sig: &[f64], c: ResidualCov<'_>| -> (DMatrix<f64>, DMatrix<f64>) {
        let r = r_matrix_maybe_scaled(es, &f0, err_keys.as_ref(), subject, sig, c, None);
        let dc = compute_dr_df_columns_cov(
            es,
            &f0,
            err_keys.as_ref(),
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            &subject.obs_l2,
            &subject.obs_rgroup,
            sig,
            c,
            None,
        );
        (r, dc)
    };
    let (r0, dc0) = r0_at(sigma, cov);
    let r0_inv = r0.clone().cholesky()?.inverse();

    let eta = DVector::from_column_slice(eta_hat);
    let j = DMatrix::from_fn(n, ne, |r_, k| sens.obs[r_].df_deta[k]);
    let omega = &params.omega.matrix;
    let omega_inv = &params.omega.inv;
    let eps = DVector::from_fn(n, |r_, _| subject.observations[r_] - sens.obs[r_].f);
    let rho_v = &eps + &j * &eta;
    let rtilde = &j * omega * j.transpose() + &r0;
    let rtilde_inv = rtilde.cholesky()?.inverse();
    let w = &rtilde_inv * &rho_v;
    let u0 = &r0_inv * &eps;
    if !(rtilde_inv.iter().all(|v| v.is_finite()) && dc0.iter().all(|v| v.is_finite())) {
        return None;
    }

    // One coordinate of the marginal at fixed η̂: from δρ and δR̃,
    // ∂F = ½[2 δρᵀw − wᵀδR̃ w + tr(R̃⁻¹δR̃)].
    let marginal = |drho: &DVector<f64>, drt: &DMatrix<f64>| -> f64 {
        let tr = trace_prod(&rtilde_inv, drt);
        0.5 * (2.0 * drho.dot(&w) - w.dot(&(drt * &w)) + tr)
    };
    // δR̃ from δJ, δΩ and δR⁰.
    let rtilde_dir = |dj: &DMatrix<f64>, domega: Option<&DMatrix<f64>>, dr0: &DMatrix<f64>| {
        let x = dj * omega * j.transpose();
        let mut out = &x + x.transpose() + dr0;
        if let Some(dom) = domega {
            out += &j * dom * j.transpose();
        }
        out
    };
    // ∂g/∂ζ from δf, δJ and δR⁰ (g = −JᵀR⁰⁻¹ε + Ω⁻¹η; δΩ⁻¹ added by the caller).
    let inner_dir = |df: &DVector<f64>, dj: &DMatrix<f64>, dr0: &DMatrix<f64>| {
        -(dj.transpose() * &u0) + j.transpose() * (&r0_inv * (dr0 * &u0 + df))
    };

    // Coupling c = ∂F/∂η̂ and the exact inner Hessian.
    let mut coupling = DVector::<f64>::zeros(ne);
    let mut h_inner = j.transpose() * &r0_inv * &j + omega_inv;
    for l in 0..ne {
        let a_l = DMatrix::from_fn(n, ne, |r_, k| sens.obs[r_].d2f_deta2[k * ne + l]);
        // ρ = y − f(η̂) + J(η̂)η̂ ⇒ ∂ρ/∂η_l = A_l η̂ (the ±J_l cancel).
        let drho = &a_l * &eta;
        let drt = rtilde_dir(&a_l, None, &DMatrix::zeros(n, n));
        coupling[l] = marginal(&drho, &drt);
        for k in 0..ne {
            let mut s = 0.0;
            for r_ in 0..n {
                s += u0[r_] * a_l[(r_, k)];
            }
            h_inner[(k, l)] -= s;
        }
    }
    let h_inner = 0.5 * (&h_inner + h_inner.transpose());
    let h_inner_inv = invert_inner_hessian(h_inner)?;
    // c · dη̂/dζ = −cᵀH⁻¹ ∂g/∂ζ = −vᵀ∂g/∂ζ.
    let v = &h_inner_inv * &coupling;
    let response = |dg: &DVector<f64>| -> f64 { -v.dot(dg) };

    let mut g = vec![0.0f64; n_packed];
    let is_fixed = fixed_coordinates(template);

    // θ: through f(η̂), J, f(η = 0) (in R⁰), and a `correlation` line's ρ directly.
    let serial_theta = model.residual_serial.as_ref().map(|s| s.rho_theta);
    for m in 0..n_theta {
        if is_fixed(m) {
            continue;
        }
        let df = DVector::from_fn(n, |r_, _| sens.obs[r_].df_dtheta[m]);
        let dj = DMatrix::from_fn(n, ne, |r_, k| sens.obs[r_].d2f_deta_dtheta[k * n_theta + m]);
        let df0: Vec<f64> = sens0.obs.iter().map(|o| o.df_dtheta[m]).collect();
        let mut dr0 = pattern_sum(&dc0, &df0);
        if serial_theta == Some(m) {
            let (dr, _) = serial_rho_derivatives(
                es,
                &f0,
                err_keys.as_ref(),
                &subject.obs_times,
                &subject.obs_raw_times,
                &subject.obs_rgroup,
                sigma,
                cov,
                None,
            )?;
            dr0 += dr;
        }
        let drho = -&df + &dj * &eta;
        let fixed = marginal(&drho, &rtilde_dir(&dj, None, &dr0));
        let resp = response(&inner_dir(&df, &dj, &dr0));
        g[m] = (fixed + resp) * theta_dx_chain(template, &params.theta, m);
    }

    // Ω (packed Cholesky): δΩ = E_rc Lᵀ + L E_cr (× L_rr on the log-packed diagonal),
    // through R̃ and, via δΩ⁻¹ = −Ω⁻¹δΩΩ⁻¹, the inner gradient's prior term.
    let l = &params.omega.chol;
    let entries = crate::estimation::parameterization::lower_tri_entries(ne, params.omega.diagonal);
    let omega_start = n_theta;
    let zero_nn = DMatrix::<f64>::zeros(n, n);
    let zero_dj = DMatrix::<f64>::zeros(n, ne);
    for (ko, &(row, col)) in entries.iter().enumerate() {
        let mut dom = DMatrix::<f64>::zeros(ne, ne);
        for c2 in 0..ne {
            dom[(row, c2)] += l[(c2, col)];
            dom[(c2, row)] += l[(c2, col)];
        }
        let chain = if row == col { l[(row, row)] } else { 1.0 };
        let fixed = marginal(
            &DVector::zeros(n),
            &rtilde_dir(&zero_dj, Some(&dom), &zero_nn),
        );
        let dg = -(omega_inv * &dom * omega_inv * &eta);
        g[omega_start + ko] = (fixed + response(&dg)) * chain;
    }

    // σ and ρ: through R⁰ only (exact central differences, see the module docs).
    let sigma_start = omega_start + entries.len();
    let zero_n = DVector::<f64>::zeros(n);
    let base_floor = floor_mask(&r0);
    for s in 0..sigma.len() {
        if is_fixed(sigma_start + s) {
            continue;
        }
        let h = sigma_step(sigma[s]);
        let mut sp = sigma.clone();
        sp[s] = sigma[s] + h;
        let plus = r0_at(&sp, cov).0;
        sp[s] = sigma[s] - h;
        let minus = r0_at(&sp, cov).0;
        same_floor(&base_floor, &plus, &minus)?;
        let dr0 = (plus - minus) / (2.0 * h);
        let fixed = marginal(&zero_n, &dr0);
        let resp = response(&inner_dir(&zero_n, &zero_dj, &dr0));
        g[sigma_start + s] = (fixed + resp) * sigma[s];
    }
    let corrs = &params.residual_correlations;
    if !corrs.is_empty() {
        let rho_start = rho_packed_start(template);
        for c in 0..corrs.len() {
            if is_fixed(rho_start + c) {
                continue;
            }
            let at = |rho: f64| -> DMatrix<f64> {
                let mut cs: Vec<ResidualCorrelation> = corrs.clone();
                cs[c].rho = rho;
                let cov_c = ResidualCov {
                    pairs: &cs,
                    serial: cov.serial,
                };
                r0_at(sigma, cov_c).0
            };
            let (plus, minus) = (at(corrs[c].rho + RHO_STEP), at(corrs[c].rho - RHO_STEP));
            same_floor(&base_floor, &plus, &minus)?;
            let dr0 = (plus - minus) / (2.0 * RHO_STEP);
            let fixed = marginal(&zero_n, &dr0);
            let resp = response(&inner_dir(&zero_n, &zero_dj, &dr0));
            g[rho_start + c] = (fixed + resp) * rho_chain(corrs[c].rho);
        }
    }
    Some(g)
}

/// `FIX`-ed packed coordinates (`packed_fixed_mask`). Their gradient entries are zeroed
/// by the population assembly anyway, so the θ/σ/ρ loops skip them — which matters
/// because those loops hold the fallible steps (a serial-ρ derivative, a σ/ρ stencil
/// that straddles the floor): a fixed coordinate must not decline the whole subject.
fn fixed_coordinates(template: &ModelParameters) -> impl Fn(usize) -> bool {
    let mask = packed_fixed_mask(template);
    move |k| mask.get(k).copied().unwrap_or(false)
}

/// The rows whose diagonal `R` sits on the `MIN_VARIANCE` floor (the builders clamp it
/// to exactly that value).
fn floor_mask(r: &DMatrix<f64>) -> Vec<bool> {
    (0..r.nrows()).map(|j| r[(j, j)] <= MIN_VARIANCE).collect()
}

/// `Some(())` when both stencil points floor exactly the rows the base point does, so
/// their difference is the derivative (module docs); `None` — decline — when the
/// stencil straddles the floor on some row.
fn same_floor(base: &[bool], plus: &DMatrix<f64>, minus: &DMatrix<f64>) -> Option<()> {
    (floor_mask(plus) == base && floor_mask(minus) == base).then_some(())
}

/// Half-width of the σ central difference: wide, because `R` is quadratic in σ and the
/// difference is exact (module docs). A zero σ has no scale to borrow.
fn sigma_step(sigma: f64) -> f64 {
    if sigma != 0.0 {
        0.1 * sigma.abs()
    } else {
        1e-3
    }
}

/// Half-width of a `block_sigma` ρ central difference — exact at any width, `R` being
/// linear in ρ. The builders do not validate ρ, so `ρ ± 0.1` past ±1 is harmless.
const RHO_STEP: f64 = 0.1;

/// The dense primitives at η̂, factored once per subject.
struct Base {
    n: usize,
    ne: usize,
    /// Predictions at η̂.
    f: Vec<f64>,
    /// `u = R⁻¹ε`.
    u: DVector<f64>,
    r_inv: DMatrix<f64>,
    /// `∂R/∂f` in column form: `dc[(k, m)] = ∂R_km/∂f_m`.
    dc: DMatrix<f64>,
    /// `∂²R_ab/∂f_a∂f_b` (`a ≠ b`).
    d2x: DMatrix<f64>,
    /// `d2y[(k, a)] = ∂²R_ka/∂f_a²`.
    d2y: DMatrix<f64>,
    /// `J = ∂f/∂η` (`n × ne`).
    j: DMatrix<f64>,
    /// `R⁻¹J`.
    rinv_j: DMatrix<f64>,
    /// `M_k = Σ_m J_mk ∂R/∂f_m`.
    m: Vec<DMatrix<f64>>,
    /// `N_k = R⁻¹M_k`.
    nk: Vec<DMatrix<f64>>,
    htilde_inv: DMatrix<f64>,
}

/// What one scalar coordinate moves (see the module table).
struct Dir {
    df: DVector<f64>,
    dj: DMatrix<f64>,
    /// `∂R/∂ζ` at fixed `f`.
    dr_direct: Option<DMatrix<f64>>,
    /// `∂(∂R/∂f)/∂ζ` at fixed `f`, column form.
    ddc_direct: Option<DMatrix<f64>>,
}

impl Dir {
    fn moving_f(df: DVector<f64>, dj: DMatrix<f64>) -> Self {
        Self {
            df,
            dj,
            dr_direct: None,
            ddc_direct: None,
        }
    }

    /// A coordinate that moves `R` only, from the builders at `ζ ± h`.
    fn direct(
        plus: &(DMatrix<f64>, DMatrix<f64>),
        minus: &(DMatrix<f64>, DMatrix<f64>),
        h: f64,
        n: usize,
        ne: usize,
    ) -> Self {
        Self {
            df: DVector::zeros(n),
            dj: DMatrix::zeros(n, ne),
            dr_direct: Some((&plus.0 - &minus.0) / (2.0 * h)),
            ddc_direct: Some((&plus.1 - &minus.1) / (2.0 * h)),
        }
    }
}

/// One coordinate's derivatives: of the data term `εᵀR⁻¹ε + ln|R|`, of `ln|H̃|`, and of
/// the data term's η-gradient `G_k = ∂(εᵀR⁻¹ε + ln|R|)/∂η_k`.
struct DirOut {
    ddata: f64,
    dlogdet: f64,
    dg: DVector<f64>,
}

/// `Σ_m w_m ∂R/∂f_m` from the column form `c` (the row-and-column-`m` pattern of each
/// `∂R/∂f_m`): with `P = c·diag(w)`, it is `P + Pᵀ − diag(P)`.
fn pattern_sum(c: &DMatrix<f64>, w: &[f64]) -> DMatrix<f64> {
    let n = c.nrows();
    let mut out = DMatrix::<f64>::zeros(n, n);
    for m in 0..n {
        let wm = w[m];
        if wm == 0.0 {
            continue;
        }
        for i in 0..n {
            let p = c[(i, m)] * wm;
            if p == 0.0 {
                continue;
            }
            out[(i, m)] += p;
            if i != m {
                out[(m, i)] += p;
            }
        }
    }
    out
}

/// `tr(AB)`.
fn trace_prod(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    let n = a.nrows();
    let mut t = 0.0;
    for p in 0..n {
        for q in 0..n {
            t += a[(p, q)] * b[(q, p)];
        }
    }
    t
}

impl Base {
    fn new(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
        sens: &SubjectSens,
        cov: ResidualCov<'_>,
    ) -> Option<Self> {
        let n = subject.observations.len();
        let ne = model.n_eta;
        let es = &model.error_spec;
        let err_keys = es.obs_keys(subject);
        let sigma = &params.sigma.values;
        let f: Vec<f64> = sens.obs.iter().map(|o| o.f).collect();
        // The very `R` the objective factors (`foce_subject_nll_interaction_dense_cov`).
        let r = r_matrix_maybe_scaled(es, &f, err_keys.as_ref(), subject, sigma, cov, None);
        let chol = r.cholesky()?;
        let r_inv = chol.inverse();
        let eps = DVector::from_fn(n, |j, _| subject.observations[j] - f[j]);
        let u = &r_inv * &eps;
        let dc = compute_dr_df_columns_cov(
            es,
            &f,
            err_keys.as_ref(),
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            &subject.obs_l2,
            &subject.obs_rgroup,
            sigma,
            cov,
            None,
        );
        let (d2x, d2y) = compute_d2r_df2_compact_cov(
            es,
            &f,
            err_keys.as_ref(),
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            &subject.obs_l2,
            &subject.obs_rgroup,
            sigma,
            cov,
            None,
        );
        if !(r_inv.iter().all(|v| v.is_finite())
            && dc.iter().all(|v| v.is_finite())
            && d2x.iter().all(|v| v.is_finite())
            && d2y.iter().all(|v| v.is_finite()))
        {
            return None;
        }
        let j = DMatrix::from_fn(n, ne, |r_, k| sens.obs[r_].df_deta[k]);
        let rinv_j = &r_inv * &j;
        let m: Vec<DMatrix<f64>> = (0..ne)
            .map(|k| pattern_sum(&dc, j.column(k).as_slice()))
            .collect();
        let nk: Vec<DMatrix<f64>> = m.iter().map(|mk| &r_inv * mk).collect();
        let mut htilde = j.transpose() * &rinv_j + &params.omega.inv;
        for k in 0..ne {
            for l in 0..ne {
                htilde[(k, l)] += 0.5 * trace_prod(&nk[k], &nk[l]);
            }
        }
        let htilde_inv = htilde.cholesky()?.inverse();
        Some(Self {
            n,
            ne,
            f,
            u,
            r_inv,
            dc,
            d2x,
            d2y,
            j,
            rinv_j,
            m,
            nk,
            htilde_inv,
        })
    }

    /// `(R, ∂R/∂f in column form)` at this subject's predictions for another σ / ρ.
    fn builders_at(
        &self,
        model: &CompiledModel,
        subject: &Subject,
        sigma: &[f64],
        cov: ResidualCov<'_>,
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let es = &model.error_spec;
        let err_keys = es.obs_keys(subject);
        let r = r_matrix_maybe_scaled(es, &self.f, err_keys.as_ref(), subject, sigma, cov, None);
        let dc = compute_dr_df_columns_cov(
            es,
            &self.f,
            err_keys.as_ref(),
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            &subject.obs_l2,
            &subject.obs_rgroup,
            sigma,
            cov,
            None,
        );
        (r, dc)
    }

    /// The derivatives of the data term, of `ln|H̃|` and of `G` along one coordinate.
    fn directional(&self, d: &Dir) -> DirOut {
        let (n, ne) = (self.n, self.ne);
        let df = d.df.as_slice();
        // δR = direct + Σ_m δf_m ∂R/∂f_m.
        let mut dr = pattern_sum(&self.dc, df);
        if let Some(x) = &d.dr_direct {
            dr += x;
        }
        // δ(∂R/∂f_m) = direct + Σ_p δf_p ∂²R/∂f_m∂f_p, in column form:
        // column m, row k = δf_m·∂²R_km/∂f_m² + [k ≠ m] δf_k·∂²R_mk/∂f_m∂f_k.
        let mut ddc = DMatrix::<f64>::zeros(n, n);
        for m in 0..n {
            for k in 0..n {
                let mut v = df[m] * self.d2y[(k, m)];
                if k != m {
                    v += df[k] * self.d2x[(m, k)];
                }
                ddc[(k, m)] = v;
            }
        }
        if let Some(x) = &d.ddc_direct {
            ddc += x;
        }

        let rinv_dr = &self.r_inv * &dr;
        let drinv = -(&rinv_dr * &self.r_inv); // δ(R⁻¹) = −R⁻¹ δR R⁻¹
        let dr_u = &dr * &self.u;
        // δ(εᵀR⁻¹ε + ln|R|) = −2 δfᵀu − uᵀδR u + tr(R⁻¹δR).
        let mut tr_rinv_dr = 0.0;
        for i in 0..n {
            tr_rinv_dr += rinv_dr[(i, i)];
        }
        let ddata = -2.0 * d.df.dot(&self.u) - self.u.dot(&dr_u) + tr_rinv_dr;
        // δu = δ(R⁻¹ε) = −R⁻¹(δR u + δf).
        let du = -(&self.r_inv * (&dr_u + &d.df));

        // δH̃ = δJᵀR⁻¹J + JᵀR⁻¹δJ + Jᵀδ(R⁻¹)J + ½δB.
        let cross = d.dj.transpose() * &self.rinv_j;
        let mut dh = &cross + cross.transpose() + self.j.transpose() * &drinv * &self.j;
        let mut dm: Vec<DMatrix<f64>> = Vec::with_capacity(ne);
        let mut dn: Vec<DMatrix<f64>> = Vec::with_capacity(ne);
        for k in 0..ne {
            let mut dmk = pattern_sum(&ddc, self.j.column(k).as_slice());
            dmk += pattern_sum(&self.dc, d.dj.column(k).as_slice());
            dn.push(&drinv * &self.m[k] + &self.r_inv * &dmk);
            dm.push(dmk);
        }
        for k in 0..ne {
            for l in 0..ne {
                dh[(k, l)] +=
                    0.5 * (trace_prod(&dn[k], &self.nk[l]) + trace_prod(&self.nk[k], &dn[l]));
            }
        }
        let dlogdet = trace_prod(&self.htilde_inv, &dh);

        // δG_k, G_k = −2uᵀJ_k − uᵀM_k u + tr(R⁻¹M_k).
        let mut dg = DVector::<f64>::zeros(ne);
        for k in 0..ne {
            let jk = self.j.column(k);
            let mk_u = &self.m[k] * &self.u;
            let mut tr_dn = 0.0;
            for i in 0..n {
                tr_dn += dn[k][(i, i)];
            }
            dg[k] = -2.0 * (du.dot(&jk) + du.dot(&mk_u))
                - 2.0 * self.u.dot(&d.dj.column(k))
                - self.u.dot(&(&dm[k] * &self.u))
                + tr_dn;
        }
        DirOut { ddata, dlogdet, dg }
    }
}
