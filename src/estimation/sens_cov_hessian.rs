//! Sensitivity-assembled FOCE/FOCEI covariance Hessian (R-matrix) — the
//! replacement for the reconverged-objective finite-difference step (issue #436).
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

use super::sens_outer_gradient::{mixed_eta_theta, subject_natural_gradient, Prep};
use crate::estimation::parameterization::{theta_packs_log, unpack_params};
use crate::estimation::sens_outer_gradient::prepare;
use crate::sens::provider::{subject_sensitivities_cov, SubjectSens};
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

/// Closed-form error-model scalars and their first **and second** `f`-derivatives
/// at one observation, built from `R`, `d = ∂R/∂f`, `d2 = ∂²R/∂f²`, `eps = y − f`.
/// Assumes `∂³R/∂f³ = 0`, which is exact for the additive / proportional /
/// combined error models away from the variance floor: `d2var_df2` is a
/// constant `2σ²` for the proportional part (`0` for additive), except where
/// `variance_at` clamps to `MIN_VARIANCE`, on which flat side `d2var_df2`
/// returns `0` (#958) — still a locally-constant `∂²R/∂f²`, so `∂³R/∂f³ = 0`
/// holds. `α`, `α'`, `β`, `p` reproduce
/// [`super::sens_outer_gradient::err_terms`]; `α''`, `β'` are the new third-order
/// pieces the M3 covariance Hessian contracts against `∂³f/∂η³` etc.
struct ErrD2 {
    alpha: f64,    // α = −2ε/R + d(R−ε²)/R²
    alpha_p: f64,  // α' = ∂α/∂f  (dε/df = −1, dR/df = d, dd/df = d2)
    alpha_pp: f64, // α'' = ∂²α/∂f²
    p: f64,        // p = 1/R + ½(d/R)²
    beta: f64,     // β = ∂p/∂f
    beta_p: f64,   // β' = ∂²p/∂f²
}

/// Build [`ErrD2`] from the raw variance derivatives. `g = R − ε²` recurs through
/// the α-algebra. The `f`-derivatives use `ε' = −1`, `R' = d`, `d' = d2`,
/// `d2' = d3 = 0`; the resulting α''/β' are the closed forms derived in the
/// module's M3 notes (validated by `err_d2_scalar_matches_fd`).
fn err_d2(r: f64, d: f64, d2: f64, eps: f64) -> ErrD2 {
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;
    let inv_r3 = inv_r2 * inv_r;
    let inv_r4 = inv_r3 * inv_r;
    let g = r - eps * eps; // R − ε²
    let alpha = -2.0 * eps * inv_r + d * g * inv_r2;
    let alpha_p = 2.0 * inv_r + 2.0 * eps * d * inv_r2 + (d2 * g + d * d + 2.0 * d * eps) * inv_r2
        - 2.0 * d * d * g * inv_r3;
    // α'' = ∂²α/∂f² (d3 = 0):
    //   (−6d + 6εd2 + 3d·d2)/R²
    //   + (−12εd² − 6d·d2·g − 4d³)/R³
    //   + 6d³g/R⁴.
    let alpha_pp = (-6.0 * d + 6.0 * eps * d2 + 3.0 * d * d2) * inv_r2
        + (-12.0 * eps * d * d - 6.0 * d * d2 * g - 4.0 * d * d * d) * inv_r3
        + 6.0 * d * d * d * g * inv_r4;
    let p = inv_r + 0.5 * (d * inv_r) * (d * inv_r);
    let beta = -d * inv_r2 + d * d2 * inv_r2 - d * d * d * inv_r3;
    // β' = ∂²p/∂f² (d3 = 0):
    //   (−d2 + d2²)/R² + (2d² − 5d²·d2)/R³ + 3d⁴/R⁴.
    let beta_p = (-d2 + d2 * d2) * inv_r2
        + (2.0 * d * d - 5.0 * d * d * d2) * inv_r3
        + 3.0 * d * d * d * d * inv_r4;
    ErrD2 {
        alpha,
        alpha_p,
        alpha_pp,
        p,
        beta,
        beta_p,
    }
}

/// Error-likelihood derivatives with respect to the prediction. Quantified rows
/// retain the closed forms above. M3 rows differentiate only the scalar
/// `-log Phi` kernel; prediction derivatives still come from the sensitivity
/// provider and are never differenced at the population-gradient level.
fn observation_err_d2(
    model: &CompiledModel,
    subject: &Subject,
    sens: &SubjectSens,
    sigma: &[f64],
    j: usize,
) -> ErrD2 {
    let cmt = subject.obs_cmts[j];
    let f = sens.obs[j].f;
    let y = subject.observations[j];
    let cens = subject.cens.get(j).copied().unwrap_or(0);
    if model.bloq_method != crate::types::BloqMethod::M3 || cens == 0 {
        return err_d2(
            model.error_spec.variance_at(cmt, f, sigma),
            model.error_spec.dvar_df(cmt, f, sigma),
            model.error_spec.d2var_df2(cmt, f, sigma),
            y - f,
        );
    }
    let at = |ff: f64| {
        crate::stats::special::m3_censored_outer(
            y,
            ff,
            model.error_spec.variance_at(cmt, ff, sigma),
            model.error_spec.dvar_df(cmt, ff, sigma),
            model.error_spec.d2var_df2(cmt, ff, sigma),
            cens,
        )
    };
    let (g1, g2, _, _) = at(f);
    let h3 = 1e-4 * (1.0 + f.abs());
    let h4 = 2e-3 * (1.0 + f.abs());
    let g2m2 = at(f - 2.0 * h3).1;
    let g2m1 = at(f - h3).1;
    let g2p1 = at(f + h3).1;
    let g2p2 = at(f + 2.0 * h3).1;
    let g3 = (g2m2 - 8.0 * g2m1 + 8.0 * g2p1 - g2p2) / (12.0 * h3);
    let q2m2 = at(f - 2.0 * h4).1;
    let q2m1 = at(f - h4).1;
    let q2p1 = at(f + h4).1;
    let q2p2 = at(f + 2.0 * h4).1;
    let g4 = (-q2p2 + 16.0 * q2p1 - 30.0 * g2 + 16.0 * q2m1 - q2m2) / (12.0 * h4 * h4);
    ErrD2 {
        alpha: 2.0 * g1,
        alpha_p: 2.0 * g2,
        alpha_pp: 2.0 * g3,
        p: g2,
        beta: g3,
        beta_p: g4,
    }
}

fn observation_data_loss(
    model: &CompiledModel,
    subject: &Subject,
    sens: &SubjectSens,
    sigma: &[f64],
    j: usize,
) -> f64 {
    let cmt = subject.obs_cmts[j];
    let f = sens.obs[j].f;
    let y = subject.observations[j];
    let r = model.error_spec.variance_at(cmt, f, sigma);
    let cens = subject.cens.get(j).copied().unwrap_or(0);
    if model.bloq_method == crate::types::BloqMethod::M3 && cens != 0 {
        -crate::stats::likelihood::m3_logcdf(y, f, r.sqrt(), cens)
    } else {
        0.5 * ((y - f).powi(2) / r + r.ln())
    }
}

struct TailJet {
    mu: f64,
    var: f64,
    mumu: f64,
    muvar: f64,
    varvar: f64,
}

/// First and second partials of the scalar FOCE M3 tail with respect to its
/// linearized marginal mean and variance.
fn foce_tail_jet(y: f64, mu: f64, var: f64, cens: i8) -> TailJet {
    let s = if cens < 0 { -1.0 } else { 1.0 };
    let w = var.sqrt();
    let z = s * (y - mu) / w;
    let h = crate::stats::special::inv_mills(z);
    let lzz = h * (z + h);
    let zm = -s / w;
    let zv = -z / (2.0 * var);
    let zmv = s / (2.0 * var * w);
    let zvv = 3.0 * z / (4.0 * var * var);
    TailJet {
        mu: -h * zm,
        var: -h * zv,
        mumu: lzz * zm * zm,
        muvar: lzz * zm * zv - h * zmv,
        varvar: lzz * zv * zv - h * zvv,
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
        let mm = mixed_eta_theta(&sens.obs, &prep.et, n_eta, prep.n_obs, m, prep.ruv);
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
    /// Direct data-likelihood `∂²L_j/∂σ_k∂σ_l`, `[k][l][j]`.
    d2loss: Vec<Vec<Vec<f64>>>,
}

/// Build [`SigmaDerivs`] for quantified Gaussian and M3-censored rows.
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
    let mut d2loss = vec![vec![vec![0.0; n_obs]; n_sigma]; n_sigma];
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
            // ∂α/∂σ_k = [2ε/R² + d(2ε²−R)/R³] R_k + [(R−ε²)/R²] d_k.
            let cens = model.bloq_method == crate::types::BloqMethod::M3
                && subject.cens.get(j).copied().unwrap_or(0) != 0;
            let da = if cens {
                (observation_err_d2(model, subject, sens, &sp, j).alpha
                    - observation_err_d2(model, subject, sens, &sm, j).alpha)
                    / (2.0 * hk)
            } else {
                (2.0 * eps * inv_r2 + d * (2.0 * eps * eps - r) * inv_r3) * r_sig
                    + ((r - eps * eps) * inv_r2) * d_sig
            };
            r1[k][j] = r_sig;
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
            for j in 0..sens.obs.len() {
                let val = (rv(&spp, subject.obs_cmts[j], sens.obs[j].f)
                    - rv(&spm, subject.obs_cmts[j], sens.obs[j].f)
                    - rv(&smp, subject.obs_cmts[j], sens.obs[j].f)
                    + rv(&smm, subject.obs_cmts[j], sens.obs[j].f))
                    / (4.0 * hk * hl);
                r2[k][l][j] = val;
                r2[l][k][j] = val;
            }
        }
    }
    for j in 0..n_obs {
        let cens = model.bloq_method == crate::types::BloqMethod::M3
            && subject.cens.get(j).copied().unwrap_or(0) != 0;
        for k in 0..n_sigma {
            for l in k..n_sigma {
                let val = if cens {
                    let hk = 1e-4 * (1.0 + sigma[k].abs());
                    let hl = 1e-4 * (1.0 + sigma[l].abs());
                    if k == l {
                        let mut sp = sigma.clone();
                        sp[k] += hk;
                        let mut sm = sigma.clone();
                        sm[k] -= hk;
                        (observation_data_loss(model, subject, sens, &sp, j)
                            - 2.0 * observation_data_loss(model, subject, sens, sigma, j)
                            + observation_data_loss(model, subject, sens, &sm, j))
                            / (hk * hk)
                    } else {
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
                        (observation_data_loss(model, subject, sens, &spp, j)
                            - observation_data_loss(model, subject, sens, &spm, j)
                            - observation_data_loss(model, subject, sens, &smp, j)
                            + observation_data_loss(model, subject, sens, &smm, j))
                            / (4.0 * hk * hl)
                    }
                } else {
                    let e = &prep.et[j];
                    let ir = 1.0 / e.r;
                    0.5 * ((-ir * ir + 2.0 * e.eps * e.eps * ir * ir * ir) * r1[k][j] * r1[l][j]
                        + (ir - e.eps * e.eps * ir * ir) * r2[k][l][j])
                };
                d2loss[k][l][j] = val;
                d2loss[l][k][j] = val;
            }
        }
    }
    SigmaDerivs {
        m_sigma,
        dalpha,
        d2loss,
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

/// Independent covariance directions. An IOV direction changes the SAME entry in
/// every occasion block; off-block zeroes never become artificial parameters.
pub(crate) fn covariance_basis<'a>(
    params: &ModelParameters,
    prep: &'a Prep,
) -> std::borrow::Cow<'a, [DMatrix<f64>]> {
    match &prep.covariance_prior {
        Some(prior) => std::borrow::Cow::Borrowed(&prior.basis),
        None => std::borrow::Cow::Owned(
            omega_entries(params.omega.diagonal, prep.n_eta)
                .into_iter()
                .map(|(r, c)| e_matrix(r, c, prep.n_eta))
                .collect(),
        ),
    }
}

pub(crate) fn covariance_sensitivities(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    b: &[f64],
) -> Option<SubjectSens> {
    if model.n_kappa > 0 {
        crate::sens::provider::subject_sensitivities_cov_iov(model, subject, theta, b)
    } else {
        subject_sensitivities_cov(model, subject, theta, b)
    }
}

/// Joint eta/kappa preparation, without duplicating the shared IOV parameters.
pub(crate) fn prepare_covariance(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    b: &[f64],
) -> Option<Prep> {
    if model.n_kappa == 0 {
        if params.omega_iov.is_some() {
            return None;
        }
        return prepare(model, subject, params, sens, b);
    }
    let iov = params.omega_iov.as_ref()?;
    let (ne, nk) = (model.n_eta, model.n_kappa);
    let occasions = crate::stats::likelihood::iov_occasion_groups(subject).len();
    let d = ne + occasions * nk;
    if b.len() != d || iov.dim() != nk || params.omega.dim() != ne {
        return None;
    }
    let mut matrix = DMatrix::zeros(d, d);
    matrix
        .view_mut((0, 0), (ne, ne))
        .copy_from(&params.omega.matrix);
    for k in 0..occasions {
        matrix
            .view_mut((ne + k * nk, ne + k * nk), (nk, nk))
            .copy_from(&iov.matrix);
    }
    let inv = crate::estimation::importance_sampling::build_joint_omega_inv(
        &params.omega.inv,
        &iov.inv,
        ne,
        nk,
        occasions,
    );
    let mut basis: Vec<_> = omega_entries(params.omega.diagonal, ne)
        .into_iter()
        .map(|(r, c)| e_matrix(r, c, d))
        .collect();
    for (r, c) in omega_entries(iov.diagonal, nk) {
        let mut e = DMatrix::zeros(d, d);
        for k in 0..occasions {
            let off = ne + k * nk;
            e[(off + r, off + c)] = 1.0;
            e[(off + c, off + r)] = 1.0;
        }
        basis.push(e);
    }
    let mut prep = crate::estimation::sens_outer_gradient::prepare_stacked(
        model, subject, params, sens, d, inv, b, None,
    )?;
    prep.covariance_prior =
        Some(crate::estimation::sens_outer_gradient::CovariancePrior { matrix, basis });
    Some(prep)
}

/// FOCEI score in the same natural direction order as the Hessian: theta,
/// base covariance entries, shared IOV covariance entries, then sigma.
fn covariance_natural_gradient(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    prep: &Prep,
    b: &[f64],
) -> Vec<f64> {
    if prep.covariance_prior.is_none() {
        return subject_natural_gradient(prep, sens, model, subject, params, b);
    }
    let core = crate::estimation::sens_outer_gradient::score_core(
        model,
        subject,
        params,
        sens,
        prep.n_eta,
        &prep.omega_inv,
        b,
        None,
    )
    .expect("prepared covariance score");
    let mut g = crate::estimation::agq_cov_hessian::fixed_b_natural_score(
        model, subject, params, sens, prep, &core, b,
    );
    let ad = subject_anchor_derivatives(model, subject, params, sens, prep, b);
    for (i, gi) in g.iter_mut().enumerate() {
        *gi += 0.5 * (&prep.htilde_inv * ad.total_first(i)).trace();
    }
    g
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
    let parts = subject_cov_hessian_parts(model, subject, params, sens, prep, eta_hat);
    parts.fuse(&prep.h_inner_inv)
}

/// The two halves of [`subject_cov_hessian_m2_natural`] **before** they are combined.
///
/// FOCEI only ever wants them fused, because its mode response is the implicit-function relation
/// `b̂_ζ = −H⁻¹M_ζ` and `C − MᵀH⁻¹M` is what that substitution yields. AGQ (#251) cannot use the
/// fused form: away from the mode a quadrature node moves as `β_{j,ζ} = b̂_ζ + √2·M_ζ(S)·z_j`,
/// which is *not* `−H⁻¹M_ζ`, so the per-node curvature has to contract `C` and `M` against `β`
/// itself. Splitting them here keeps one derivation of both quantities rather than a second copy
/// in `agq_cov_hessian` that could drift.
///
/// Nothing about the FOCEI result changes: [`CovHessianParts::fuse`] performs exactly the
/// `explicit(a,b) − mall[a]·(H⁻¹mall[b])` this function used to inline, and
/// `cov_hessian_parts_fuse_to_the_m2_natural_block` pins that.
pub(crate) struct CovHessianParts {
    /// `C[ξ,ζ] = ∂²Φ/∂ξ∂ζ` at **fixed** `b` — the explicit cross-partial, no mode response.
    pub(crate) c: DMatrix<f64>,
    /// `M_ζ = ∂²Φ/∂b∂ζ`, one `d`-vector per natural parameter, in the axis order of `c`.
    pub(crate) m: Vec<DVector<f64>>,
}

impl CovHessianParts {
    /// `C[ξ,ζ] − M_ξᵀ H⁻¹ M_ζ` — the FOCEI M2 natural block.
    pub(crate) fn fuse(&self, h_inner_inv: &DMatrix<f64>) -> DMatrix<f64> {
        let dim = self.c.nrows();
        let uall: Vec<DVector<f64>> = self.m.iter().map(|m| h_inner_inv * m).collect();
        let mut h = DMatrix::zeros(dim, dim);
        for a in 0..dim {
            for b in 0..dim {
                h[(a, b)] = self.c[(a, b)] - self.m[a].dot(&uall[b]);
            }
        }
        h
    }
}

/// `C` and `M` for one subject, evaluated at whatever `b` the caller supplies.
///
/// `eta_hat` is the evaluation point, **not** necessarily the mode: pass a quadrature node and
/// `sens`/`prep` built at that node to get the node-local `C_j`/`M_j` that #251's term (C) needs.
/// The `Ω`-block's `z = Ω⁻¹·eta_hat` is what makes that substitution correct — it is the prior
/// score at the evaluation point, which is exactly what a fixed-`b` curvature should carry.
pub(crate) fn subject_cov_hessian_parts(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    prep: &Prep,
    eta_hat: &[f64],
) -> CovHessianParts {
    let n_theta = params.theta.len();
    let e_mats = covariance_basis(params, prep);
    let n_omega = e_mats.len();
    let n_sigma = params.sigma.values.len();
    let nt = n_theta;
    let nw = nt + n_omega; // σ offset
    let dim = n_theta + n_omega + n_sigma;

    let omega_inv = &prep.omega_inv;
    let z = omega_inv * DVector::from_column_slice(eta_hat);

    // Mode-coupling vectors M_ζ = ∂²Φ/∂η∂ζ for every natural parameter, in order.
    let (m_theta, _) = theta_m_and_u(prep, sens, n_theta);
    let sd = sigma_derivs(model, subject, params, sens, prep);
    let m_omega: Vec<DVector<f64>> = e_mats.iter().map(|e| -(omega_inv * (e * &z))).collect();

    let mut mall: Vec<DVector<f64>> = Vec::with_capacity(dim);
    mall.extend(m_theta);
    mall.extend(m_omega);
    mall.extend(sd.m_sigma.iter().cloned());

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
            // σσ: direct data-likelihood curvature, Gaussian or M3.
            let k = lo - nw;
            let l = hi - nw;
            (0..prep.n_obs).map(|j| sd.d2loss[k][l][j]).sum()
        }
    };

    let mut c = DMatrix::zeros(dim, dim);
    for a in 0..dim {
        for b in 0..dim {
            c[(a, b)] = explicit(a, b);
        }
    }
    CovHessianParts { c, m: mall }
}

/// σ-derivatives of the error scalars the M3 (`½log|H̃|`) Hessian needs, all by
/// finite differences of the closed-form [`ErrD2`] over σ (exact algebra of
/// well-conditioned functions — σ enters only through `R(f,σ)`). First
/// derivatives are central; the diagonal second derivative is a 3-point stencil;
/// the mixed `σ_k σ_l` second derivative is a 4-point stencil on α and p.
struct M3Sigma {
    dp: Vec<Vec<f64>>,           // ∂p/∂σ_k        [k][j]
    dbeta: Vec<Vec<f64>>,        // ∂β/∂σ_k        [k][j]
    dalpha: Vec<Vec<f64>>,       // ∂α/∂σ_k        [k][j]
    dalpha_p: Vec<Vec<f64>>,     // ∂α'/∂σ_k       [k][j]
    d2p: Vec<Vec<Vec<f64>>>,     // ∂²p/∂σ_k∂σ_l   [k][l][j]
    d2alpha: Vec<Vec<Vec<f64>>>, // ∂²α/∂σ_k∂σ_l   [k][l][j]
}

fn m3_sigma_derivs(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
) -> M3Sigma {
    let sigma = &params.sigma.values;
    let n_sigma = sigma.len();
    let n_obs = sens.obs.len();
    // ErrD2 for observation `j` at error parameters `sig` (ε = y − f is
    // σ-independent; only R, d, d2 move).
    let ed = |sig: &[f64], j: usize| observation_err_d2(model, subject, sens, sig, j);

    let mut dp = vec![vec![0.0; n_obs]; n_sigma];
    let mut dbeta = vec![vec![0.0; n_obs]; n_sigma];
    let mut dalpha = vec![vec![0.0; n_obs]; n_sigma];
    let mut dalpha_p = vec![vec![0.0; n_obs]; n_sigma];
    let mut d2p = vec![vec![vec![0.0; n_obs]; n_sigma]; n_sigma];
    let mut d2alpha = vec![vec![vec![0.0; n_obs]; n_sigma]; n_sigma];

    for k in 0..n_sigma {
        let hk = sigma_fd_step(sigma[k]);
        let mut sp = sigma.clone();
        sp[k] += hk;
        let mut sm = sigma.clone();
        sm[k] -= hk;
        for j in 0..n_obs {
            let ep = ed(&sp, j);
            let em = ed(&sm, j);
            let e0 = ed(sigma, j);
            let inv2h = 1.0 / (2.0 * hk);
            dp[k][j] = (ep.p - em.p) * inv2h;
            dbeta[k][j] = (ep.beta - em.beta) * inv2h;
            dalpha[k][j] = (ep.alpha - em.alpha) * inv2h;
            dalpha_p[k][j] = (ep.alpha_p - em.alpha_p) * inv2h;
            let inv_h2 = 1.0 / (hk * hk);
            d2p[k][k][j] = (ep.p - 2.0 * e0.p + em.p) * inv_h2;
            d2alpha[k][k][j] = (ep.alpha - 2.0 * e0.alpha + em.alpha) * inv_h2;
        }
    }
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
            let denom = 1.0 / (4.0 * hk * hl);
            for j in 0..n_obs {
                let p_val = (ed(&spp, j).p - ed(&spm, j).p - ed(&smp, j).p + ed(&smm, j).p) * denom;
                let a_val = (ed(&spp, j).alpha - ed(&spm, j).alpha - ed(&smp, j).alpha
                    + ed(&smm, j).alpha)
                    * denom;
                d2p[k][l][j] = p_val;
                d2p[l][k][j] = p_val;
                d2alpha[k][l][j] = a_val;
                d2alpha[l][k][j] = a_val;
            }
        }
    }
    M3Sigma {
        dp,
        dbeta,
        dalpha,
        dalpha_p,
        d2p,
        d2alpha,
    }
}

/// One natural/η parameter direction, classified so the H̃- and inner-objective
/// derivative assembly can dispatch on what `p_j`, `a_j`, and `Ω⁻¹` actually
/// depend on. Natural params are ordered `[θ, Ω, σ]` (indices `0..dim`); the η
/// directions follow (`dim + l`).
#[derive(Clone, Copy, PartialEq)]
enum Dir {
    Theta(usize),
    Omega(usize),
    Sigma(usize),
    Eta(usize),
}

/// The full M3 covariance Hessian over the natural `[θ, Ω, σ]` parameters — the
/// Hessian of the `½log|H̃|` term carried through the moving mode `η̂(ζ)`:
///
/// ```text
///   M3_{ξζ} = C'_{ξζ}                                   [A explicit]
///           + Σ_l C'_{ξη_l} η̂_{l,ζ} + Σ_m C'_{ζη_m} η̂_{m,ξ}   [B mode cross]
///           + Σ_{lm} C'_{η_lη_m} η̂_{l,ζ} η̂_{m,ξ}        [C mode quadratic]
///           + Σ_l C'_{η_l} η̂_{l,ξζ},                    [D 2nd mode response]
/// ```
///
/// with `C' = ½log|H̃|`, `C'_{η_l} = ½ g_eta_l`, `η̂_{·,ξ} = −H⁻¹ M_ξ`, and the
/// second mode response `η̂_{ξζ} = −H⁻¹[S_{ξζ} + S_{ξη}η̂_ζ + S_{ζη}η̂_ξ +
/// S_{ηη}:(η̂_ξ,η̂_ζ)]` from the inner stationarity `S = ∂lᵢ/∂η = 0`. The cross
/// partials `C'_{st} = ½[tr(H̃⁻¹∂²H̃/∂s∂t) − tr(H̃⁻¹∂H̃/∂s·H̃⁻¹∂H̃/∂t)]` use the
/// provider's third-order `f`-sensitivities (`∂³f/∂η³`, `∂³f/∂η²∂θ`, `∂³f/∂η∂θ²`)
/// and the α''/β' error scalars; `S_{ηη}` adds `α''` against `∂³f/∂η³`.
///
/// The inner-mode responses to the population parameters, shared by the FOCEI
/// `½log|H̃|` (M3) and FOCE (Sheiner–Beal) covariance Hessians because they depend
/// only on the **inner objective** `lᵢ` (the posterior mode is the same for both):
///
/// * `eta_d[ζ] = η̂_{·,ζ} = −H⁻¹ M_ζ`, `M_ζ = ∂²lᵢ/∂η∂ζ` (the M2 mode-coupling);
/// * `eta_dd[ξ][ζ] = η̂_{ξζ} = −H⁻¹[S_{ξζ} + S_{ξη}η̂_ζ + S_{ζη}η̂_ξ +
///   S_{ηη}:(η̂_ξ,η̂_ζ)]`, `S = ∂lᵢ/∂η`, the second mode response.
///
/// Both are indexed over the natural `[θ, Ω, σ]` directions. `S_{ηη}` contracts
/// `∂³f/∂η³` against `α''`; the σ pieces use the FD-of-closed-form scalars.
fn inner_eta_responses(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    prep: &Prep,
    eta_hat: &[f64],
) -> (Vec<DVector<f64>>, Vec<Vec<DVector<f64>>>) {
    let ne = prep.n_eta;
    let n_obs = prep.n_obs;
    let nt = params.theta.len();
    let e_mats = covariance_basis(params, prep);
    let n_omega = e_mats.len();
    let n_sigma = params.sigma.values.len();
    let nw = nt + n_omega;
    let dim = nt + n_omega + n_sigma;
    let omega_inv = &prep.omega_inv;
    let h_inner_inv = &prep.h_inner_inv;
    let z = omega_inv * DVector::from_column_slice(eta_hat);

    let a: Vec<DVector<f64>> = sens
        .obs
        .iter()
        .map(|o| DVector::from_column_slice(&o.df_deta))
        .collect();
    let ed: Vec<ErrD2> = (0..n_obs)
        .map(|j| observation_err_d2(model, subject, sens, &params.sigma.values, j))
        .collect();
    let m3s = m3_sigma_derivs(model, subject, params, sens);
    let dir_of = |d: usize| -> Dir {
        if d < nt {
            Dir::Theta(d)
        } else if d < nw {
            Dir::Omega(d - nt)
        } else {
            Dir::Sigma(d - nw)
        }
    };
    let t_deta2_dtheta = |j: usize, r: usize, l: usize, m: usize| {
        sens.obs[j].d3f_deta2_dtheta[(r * ne + l) * nt + m]
    };
    let t_deta_dtheta2 = |j: usize, r: usize, m: usize, n: usize| {
        sens.obs[j].d3f_deta_dtheta2[(r * nt + m) * nt + n]
    };
    let t_deta3 =
        |j: usize, r: usize, l: usize, m: usize| sens.obs[j].d3f_deta3[(r * ne + l) * ne + m];

    let (m_theta, _) = theta_m_and_u(prep, sens, nt);
    let sd = sigma_derivs(model, subject, params, sens, prep);
    let m_omega: Vec<DVector<f64>> = e_mats.iter().map(|e| -(omega_inv * (e * &z))).collect();
    let mut mall: Vec<DVector<f64>> = Vec::with_capacity(dim);
    mall.extend(m_theta);
    mall.extend(m_omega);
    mall.extend(sd.m_sigma.iter().cloned());
    let eta_d: Vec<DVector<f64>> = mall.iter().map(|m| -(h_inner_inv * m)).collect();

    let spp = |xi: usize, ze: usize| -> DVector<f64> {
        let mut v = DVector::<f64>::zeros(ne);
        for j in 0..n_obs {
            let aj = &a[j];
            let bj = &sens.obs[j].df_dtheta;
            let app = ed[j].alpha_pp;
            let apr = ed[j].alpha_p;
            let alp = ed[j].alpha;
            let bmat = |k: usize, m: usize| sens.obs[j].d2f_deta_dtheta[k * nt + m];
            match (dir_of(xi), dir_of(ze)) {
                (Dir::Theta(m), Dir::Theta(n)) => {
                    let c2 = sens.obs[j].d2f_dtheta2[m * nt + n];
                    for r in 0..ne {
                        v[r] += 0.5
                            * (app * bj[m] * bj[n] * aj[r]
                                + apr * (c2 * aj[r] + bj[m] * bmat(r, n) + bj[n] * bmat(r, m))
                                + alp * t_deta_dtheta2(j, r, m, n));
                    }
                }
                (Dir::Theta(m), Dir::Sigma(k)) | (Dir::Sigma(k), Dir::Theta(m)) => {
                    for r in 0..ne {
                        v[r] += 0.5
                            * (m3s.dalpha_p[k][j] * bj[m] * aj[r] + m3s.dalpha[k][j] * bmat(r, m));
                    }
                }
                (Dir::Sigma(k), Dir::Sigma(l)) => {
                    for r in 0..ne {
                        v[r] += 0.5 * m3s.d2alpha[k][l][j] * aj[r];
                    }
                }
                _ => {}
            }
        }
        if let (Dir::Omega(e), Dir::Omega(f)) = (dir_of(xi), dir_of(ze)) {
            let ee = &e_mats[e];
            let ef = &e_mats[f];
            v += omega_inv * (ee * omega_inv * ef + ef * omega_inv * ee) * &z;
        }
        v
    };
    let spe = |xi: usize| -> DMatrix<f64> {
        let mut mtx = DMatrix::<f64>::zeros(ne, ne);
        match dir_of(xi) {
            Dir::Theta(n) => {
                for j in 0..n_obs {
                    let aj = &a[j];
                    let bn = sens.obs[j].df_dtheta[n];
                    let app = ed[j].alpha_pp;
                    let apr = ed[j].alpha_p;
                    let alp = ed[j].alpha;
                    let amat = |k: usize, l: usize| sens.obs[j].d2f_deta2[k * ne + l];
                    let bmat = |k: usize, m: usize| sens.obs[j].d2f_deta_dtheta[k * nt + m];
                    for r in 0..ne {
                        for mm in 0..ne {
                            mtx[(r, mm)] += 0.5
                                * (app * aj[mm] * bn * aj[r]
                                    + apr
                                        * (bmat(mm, n) * aj[r]
                                            + bn * amat(r, mm)
                                            + aj[mm] * bmat(r, n))
                                    + alp * t_deta2_dtheta(j, r, mm, n));
                        }
                    }
                }
            }
            Dir::Sigma(k) => {
                for j in 0..n_obs {
                    let aj = &a[j];
                    let amat = |kk: usize, l: usize| sens.obs[j].d2f_deta2[kk * ne + l];
                    for r in 0..ne {
                        for mm in 0..ne {
                            mtx[(r, mm)] += 0.5
                                * (m3s.dalpha_p[k][j] * aj[mm] * aj[r]
                                    + m3s.dalpha[k][j] * amat(r, mm));
                        }
                    }
                }
            }
            Dir::Omega(e) => {
                mtx = -(omega_inv * &e_mats[e] * omega_inv);
            }
            Dir::Eta(_) => {}
        }
        mtx
    };
    let seta = |u: &DVector<f64>, w: &DVector<f64>| -> DVector<f64> {
        let mut out = DVector::<f64>::zeros(ne);
        for j in 0..n_obs {
            let aj = &a[j];
            let app = ed[j].alpha_pp;
            let apr = ed[j].alpha_p;
            let alp = ed[j].alpha;
            let au = aj.dot(u);
            let aw = aj.dot(w);
            let mut uaw = 0.0;
            let mut au_row = DVector::<f64>::zeros(ne);
            let mut aw_row = DVector::<f64>::zeros(ne);
            for r in 0..ne {
                let mut su = 0.0;
                let mut sw = 0.0;
                for l in 0..ne {
                    let arl = sens.obs[j].d2f_deta2[r * ne + l];
                    su += arl * u[l];
                    sw += arl * w[l];
                }
                au_row[r] = su;
                aw_row[r] = sw;
                uaw += su * w[r];
            }
            for r in 0..ne {
                let mut tcontr = 0.0;
                for l in 0..ne {
                    for m in 0..ne {
                        tcontr += t_deta3(j, r, l, m) * u[l] * w[m];
                    }
                }
                out[r] += 0.5
                    * (app * aj[r] * au * aw
                        + apr * (aj[r] * uaw + aw * au_row[r] + au * aw_row[r])
                        + alp * tcontr);
            }
        }
        out
    };

    let mut eta_dd: Vec<Vec<DVector<f64>>> = vec![vec![DVector::<f64>::zeros(ne); dim]; dim];
    for xi in 0..dim {
        for ze in xi..dim {
            let mut rhs = spp(xi, ze);
            rhs += spe(ze) * &eta_d[xi];
            rhs += spe(xi) * &eta_d[ze];
            rhs += seta(&eta_d[xi], &eta_d[ze]);
            let v = -(h_inner_inv * rhs);
            eta_dd[ze][xi].clone_from(&v);
            eta_dd[xi][ze] = v;
        }
    }
    (eta_d, eta_dd)
}

/// The anchor's parameter derivatives, and the mode responses that chain them.
///
/// FOCEI consumes these only through the `½log|H̃|` traces in [`subject_cov_hessian_m3_natural`].
/// AGQ (#251) needs the **matrices themselves**: `S_ζ = propagate(dH̃/dζ)` feeds the Cholesky
/// differential that places the quadrature nodes, and no amount of trace information substitutes
/// for it. Exposing them keeps one derivation — every entry here comes from the third-order
/// `f`-sensitivities (`d3f_deta3`, `d3f_deta2_dtheta`, `d3f_deta_dtheta2`), which the provider
/// obtains by finite-differencing the exact second-order `Dual2` jet (Shi 2021). The anchor
/// and mode responses are assembled directly rather than differencing reconverged gradients.
///
/// # Directions
///
/// `dh` and `d2h` are indexed over `dim + n_eta` directions: the natural `[θ, Ω, σ]` parameters
/// first, then η. They are **partials at fixed η** — deliberately, because the mode response is
/// chained on separately via `eta_d` / `eta_dd`. Reading `dh[ζ]` as the total `dH̃/dζ` is wrong and
/// is what [`AnchorDerivatives::total_first`] exists to prevent.
pub(crate) struct AnchorDerivatives {
    /// Number of natural `[θ, Ω, σ]` parameters.
    pub(crate) dim: usize,
    /// Number of random effects; directions `dim..dim+n_eta` are the η axes.
    pub(crate) n_eta: usize,
    /// `∂H̃/∂s` at fixed η, one per direction.
    pub(crate) dh: Vec<DMatrix<f64>>,
    /// `∂²H̃/∂s∂t` at fixed η, symmetric in the pair.
    pub(crate) d2h: Vec<Vec<DMatrix<f64>>>,
    /// `C'_{st} = ½[tr(H̃⁻¹∂²H̃/∂s∂t) − tr(K_sK_t)]`, the `½log|H̃|` cross-partials.
    pub(crate) cpp: DMatrix<f64>,
    /// `b̂_ζ = −H⁻¹M_ζ`, natural directions only.
    pub(crate) eta_d: Vec<DVector<f64>>,
    /// `b̂_ζξ`, the second mode response, natural directions only.
    pub(crate) eta_dd: Vec<Vec<DVector<f64>>>,
}

impl AnchorDerivatives {
    /// `dH̃/dζ = ∂H̃/∂ζ + Σ_l (∂H̃/∂η_l)·b̂_ζ[l]` — the **total** derivative.
    ///
    /// `H̃` is evaluated at `b̂(x)`, so AGQ's `S_ζ` needs this, not the partial. Dropping the sum
    /// is the error that a frozen-mode finite difference would happily agree with.
    pub(crate) fn total_first(&self, zeta: usize) -> DMatrix<f64> {
        let mut m = self.dh[zeta].clone();
        for l in 0..self.n_eta {
            m += self.eta_d[zeta][l] * &self.dh[self.dim + l];
        }
        m
    }

    /// `d²H̃/dζdξ` — the same chain one order up, matching the A/B/C/D structure
    /// [`subject_cov_hessian_m3_natural`] applies to the scalar `½log|H̃|`.
    pub(crate) fn total_second(&self, zeta: usize, xi: usize) -> DMatrix<f64> {
        let (d, ne) = (self.dim, self.n_eta);
        let mut m = self.d2h[zeta][xi].clone(); // A: explicit
        for l in 0..ne {
            // B: one index through the mode.
            m += self.eta_d[xi][l] * &self.d2h[zeta][d + l];
            m += self.eta_d[zeta][l] * &self.d2h[d + l][xi];
            // D: second mode response.
            m += self.eta_dd[zeta][xi][l] * &self.dh[d + l];
        }
        for l in 0..ne {
            for k in 0..ne {
                // C: both indices through the mode.
                m += (self.eta_d[zeta][l] * self.eta_d[xi][k]) * &self.d2h[d + l][d + k];
            }
        }
        m
    }
}

/// The `½log|H̃|` (M3) covariance Hessian — [`AnchorDerivatives`] contracted into traces.
pub(crate) fn subject_cov_hessian_m3_natural(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    prep: &Prep,
    eta_hat: &[f64],
) -> DMatrix<f64> {
    let ad = subject_anchor_derivatives(model, subject, params, sens, prep, eta_hat);
    let (dim, ne) = (ad.dim, ad.n_eta);
    let half_geta = DVector::from_iterator(ne, prep.g_eta.iter().map(|g| 0.5 * g));
    let mut m3 = DMatrix::zeros(dim, dim);
    for xi in 0..dim {
        for ze in xi..dim {
            // A
            let mut val = ad.cpp[(xi, ze)];
            // B
            for l in 0..ne {
                val += ad.cpp[(xi, dim + l)] * ad.eta_d[ze][l]
                    + ad.cpp[(ze, dim + l)] * ad.eta_d[xi][l];
            }
            // C
            for l in 0..ne {
                for m in 0..ne {
                    val += ad.cpp[(dim + l, dim + m)] * ad.eta_d[xi][l] * ad.eta_d[ze][m];
                }
            }
            // D: ½ g_eta · η̂_{ξζ}.
            val += half_geta.dot(&ad.eta_dd[xi][ze]);

            m3[(xi, ze)] = val;
            m3[(ze, xi)] = val;
        }
    }
    m3
}

/// Build [`AnchorDerivatives`]. This is the body that used to be inlined in
/// [`subject_cov_hessian_m3_natural`]; the split is a pure extraction, and
/// `cov_hessian_m3_natural_matches_reconverged_fd_*` are its regression net.
pub(crate) fn subject_anchor_derivatives(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    prep: &Prep,
    eta_hat: &[f64],
) -> AnchorDerivatives {
    let ne = prep.n_eta;
    let n_obs = prep.n_obs;
    let nt = params.theta.len();
    let e_mats = covariance_basis(params, prep);
    let n_omega = e_mats.len();
    let n_sigma = params.sigma.values.len();
    let nw = nt + n_omega;
    let dim = nt + n_omega + n_sigma;
    let nd = dim + ne; // directions: natural params then η

    let omega_inv = &prep.omega_inv;
    let htilde_inv = &prep.htilde_inv;

    // Per-observation primitives.
    let a: Vec<DVector<f64>> = sens
        .obs
        .iter()
        .map(|o| DVector::from_column_slice(&o.df_deta))
        .collect();
    let ed: Vec<ErrD2> = (0..n_obs)
        .map(|j| observation_err_d2(model, subject, sens, &params.sigma.values, j))
        .collect();
    let p: Vec<f64> = ed.iter().map(|e| e.p).collect();
    let m3s = m3_sigma_derivs(model, subject, params, sens);

    let dir_of = |d: usize| -> Dir {
        if d < nt {
            Dir::Theta(d)
        } else if d < nw {
            Dir::Omega(d - nt)
        } else if d < dim {
            Dir::Sigma(d - nw)
        } else {
            Dir::Eta(d - dim)
        }
    };

    // Per-direction, per-obs first-order primitives:
    //   pa[d][j] = ∂p_j/∂(dir d),   av[d][j] = ∂a_j/∂(dir d)  (n_eta vector).
    let acol = |j: usize, l: usize| -> DVector<f64> {
        DVector::from_iterator(ne, (0..ne).map(|k| sens.obs[j].d2f_deta2[k * ne + l]))
    };
    let bcol = |j: usize, m: usize| -> DVector<f64> {
        DVector::from_iterator(ne, (0..ne).map(|k| sens.obs[j].d2f_deta_dtheta[k * nt + m]))
    };
    let mut pa = vec![vec![0.0; n_obs]; nd];
    let mut av = vec![vec![DVector::<f64>::zeros(ne); n_obs]; nd];
    for d in 0..nd {
        match dir_of(d) {
            Dir::Theta(m) => {
                for j in 0..n_obs {
                    pa[d][j] = ed[j].beta * sens.obs[j].df_dtheta[m];
                    av[d][j] = bcol(j, m);
                }
            }
            Dir::Eta(l) => {
                for j in 0..n_obs {
                    pa[d][j] = ed[j].beta * a[j][l];
                    av[d][j] = acol(j, l);
                }
            }
            Dir::Sigma(k) => {
                for j in 0..n_obs {
                    pa[d][j] = m3s.dp[k][j];
                }
            }
            Dir::Omega(_) => {}
        }
    }

    // ∂H̃/∂(dir d): Σ_j[ pa a aᵀ + p(av aᵀ + a avᵀ) ]  (+ Ω part for Ω directions).
    let first = |d: usize| -> DMatrix<f64> {
        let mut m = DMatrix::zeros(ne, ne);
        for j in 0..n_obs {
            let aj = &a[j];
            let avj = &av[d][j];
            for r in 0..ne {
                for c in 0..ne {
                    m[(r, c)] +=
                        pa[d][j] * aj[r] * aj[c] + p[j] * (avj[r] * aj[c] + aj[r] * avj[c]);
                }
            }
        }
        if let Dir::Omega(e) = dir_of(d) {
            m -= omega_inv * &e_mats[e] * omega_inv;
        }
        m
    };
    let dh: Vec<DMatrix<f64>> = (0..nd).map(first).collect();
    let kmat: Vec<DMatrix<f64>> = dh.iter().map(|m| htilde_inv * m).collect();

    // (∂²p/∂s∂t, ∂²a/∂s∂t) per obs for a direction pair, both drawn from {θ,σ,η}
    // (any Ω direction contributes zero here; its second-order action is the
    // Ω⁻¹ matrix term added in `second`). Symmetric in (s,t).
    let t_deta2_dtheta = |j: usize, r: usize, l: usize, m: usize| -> f64 {
        sens.obs[j].d3f_deta2_dtheta[(r * ne + l) * nt + m]
    };
    let t_deta_dtheta2 = |j: usize, r: usize, m: usize, n: usize| -> f64 {
        sens.obs[j].d3f_deta_dtheta2[(r * nt + m) * nt + n]
    };
    let t_deta3 = |j: usize, r: usize, l: usize, m: usize| -> f64 {
        sens.obs[j].d3f_deta3[(r * ne + l) * ne + m]
    };
    let pp_aa = |s: usize, t: usize, j: usize| -> (f64, DVector<f64>) {
        let aj = &a[j];
        let bj = &sens.obs[j].df_dtheta;
        let amat = |k: usize, l: usize| sens.obs[j].d2f_deta2[k * ne + l];
        let bmat = |k: usize, m: usize| sens.obs[j].d2f_deta_dtheta[k * nt + m];
        let (beta, bp) = (ed[j].beta, ed[j].beta_p);
        match (dir_of(s), dir_of(t)) {
            (Dir::Eta(l), Dir::Eta(m)) => {
                let pp = bp * aj[l] * aj[m] + beta * amat(l, m);
                let aa = DVector::from_iterator(ne, (0..ne).map(|r| t_deta3(j, r, l, m)));
                (pp, aa)
            }
            (Dir::Eta(l), Dir::Theta(m)) | (Dir::Theta(m), Dir::Eta(l)) => {
                let pp = bp * aj[l] * bj[m] + beta * bmat(l, m);
                let aa = DVector::from_iterator(ne, (0..ne).map(|r| t_deta2_dtheta(j, r, l, m)));
                (pp, aa)
            }
            (Dir::Theta(m), Dir::Theta(n)) => {
                let c2 = sens.obs[j].d2f_dtheta2[m * nt + n];
                let pp = bp * bj[m] * bj[n] + beta * c2;
                let aa = DVector::from_iterator(ne, (0..ne).map(|r| t_deta_dtheta2(j, r, m, n)));
                (pp, aa)
            }
            (Dir::Eta(l), Dir::Sigma(k)) | (Dir::Sigma(k), Dir::Eta(l)) => {
                (m3s.dbeta[k][j] * aj[l], DVector::zeros(ne))
            }
            (Dir::Theta(m), Dir::Sigma(k)) | (Dir::Sigma(k), Dir::Theta(m)) => {
                (m3s.dbeta[k][j] * bj[m], DVector::zeros(ne))
            }
            (Dir::Sigma(k), Dir::Sigma(l)) => (m3s.d2p[k][l][j], DVector::zeros(ne)),
            _ => (0.0, DVector::zeros(ne)),
        }
    };

    // ∂²H̃/∂s∂t.
    let second = |s: usize, t: usize| -> DMatrix<f64> {
        let mut m = DMatrix::zeros(ne, ne);
        for j in 0..n_obs {
            let aj = &a[j];
            let avs = &av[s][j];
            let avt = &av[t][j];
            let (pp, aa) = pp_aa(s, t, j);
            for r in 0..ne {
                for c in 0..ne {
                    m[(r, c)] += pp * aj[r] * aj[c]
                        + pa[s][j] * (avt[r] * aj[c] + aj[r] * avt[c])
                        + pa[t][j] * (avs[r] * aj[c] + aj[r] * avs[c])
                        + p[j]
                            * (aa[r] * aj[c] + avs[r] * avt[c] + avt[r] * avs[c] + aj[r] * aa[c]);
                }
            }
        }
        if let (Dir::Omega(e), Dir::Omega(f)) = (dir_of(s), dir_of(t)) {
            let ee = &e_mats[e];
            let ef = &e_mats[f];
            let inner = ee * omega_inv * ef + ef * omega_inv * ee;
            m += omega_inv * inner * omega_inv;
        }
        m
    };

    // ∂²H̃/∂s∂t materialised, and C'_{st} = ½[tr(H̃⁻¹ ∂²H̃/∂s∂t) − tr(K_s K_t)] from it.
    //
    // `second` is evaluated once per unordered pair, exactly as before — it is symmetric in
    // `(s,t)` by construction (`pp_aa` is), so the mirror is a clone rather than a second call.
    // Keeping the matrices is what lets AGQ chain them into `S_ζξ`; FOCEI only ever needed the
    // two traces, which is why they were discarded here.
    let mut d2h = vec![vec![DMatrix::<f64>::zeros(ne, ne); nd]; nd];
    let mut cpp = DMatrix::zeros(nd, nd);
    for s in 0..nd {
        for t in s..nd {
            let m = second(s, t);
            let tr2 = (htilde_inv * &m).trace();
            let trk = (&kmat[s] * &kmat[t]).trace();
            let v = 0.5 * (tr2 - trk);
            cpp[(s, t)] = v;
            cpp[(t, s)] = v;
            d2h[t][s] = m.clone();
            d2h[s][t] = m;
        }
    }

    // Inner-mode responses (shared with FOCE): η̂_{·,ζ} and η̂_{ξζ}, both functions
    // of the inner objective `lᵢ` only.
    let (eta_d, eta_dd) = inner_eta_responses(model, subject, params, sens, prep, eta_hat);

    AnchorDerivatives {
        dim,
        n_eta: ne,
        dh,
        d2h,
        cpp,
        eta_d,
        eta_dd,
    }
}

/// The full natural-space FOCEI covariance Hessian `M2 + M3` over `[θ, Ω, σ]`.
pub(crate) fn subject_cov_hessian_natural(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    prep: &Prep,
    eta_hat: &[f64],
) -> DMatrix<f64> {
    subject_cov_hessian_m2_natural(model, subject, params, sens, prep, eta_hat)
        + subject_cov_hessian_m3_natural(model, subject, params, sens, prep, eta_hat)
}

/// Per-σ derivatives of the frozen FOCE residual variance `R⁰ = R(f(η=0), σ)` by
/// finite differences of the closed-form variance (exact algebra). Returns, per
/// quant row `i`: `∂R⁰/∂σ_k` `[k][i]`, `∂²R⁰/∂σ_k∂σ_l` `[k][l][i]`, and
/// `∂(∂R⁰/∂f)/∂σ_k` `[k][i]` (the latter for the θσ cross term, since
/// `∂R⁰/∂θ = (∂R⁰/∂f)·∂f0/∂θ`).
struct Foce0Sigma {
    dr0: Vec<Vec<f64>>,       // ∂R⁰/∂σ_k       [k][i]
    d2r0: Vec<Vec<Vec<f64>>>, // ∂²R⁰/∂σ_k∂σ_l [k][l][i]
    dd0: Vec<Vec<f64>>,       // ∂(∂R⁰/∂f)/∂σ_k [k][i]
}

fn foce0_sigma(model: &CompiledModel, cmts: &[usize], f0: &[f64], sigma: &[f64]) -> Foce0Sigma {
    let n_sigma = sigma.len();
    let nq = f0.len();
    let var = |sig: &[f64], i: usize| model.error_spec.variance_at(cmts[i], f0[i], sig);
    let dvar = |sig: &[f64], i: usize| model.error_spec.dvar_df(cmts[i], f0[i], sig);
    let mut dr0 = vec![vec![0.0; nq]; n_sigma];
    let mut dd0 = vec![vec![0.0; nq]; n_sigma];
    let mut d2r0 = vec![vec![vec![0.0; nq]; n_sigma]; n_sigma];
    for k in 0..n_sigma {
        let hk = sigma_fd_step(sigma[k]);
        let mut sp = sigma.to_vec();
        sp[k] += hk;
        let mut sm = sigma.to_vec();
        sm[k] -= hk;
        for i in 0..nq {
            dr0[k][i] = (var(&sp, i) - var(&sm, i)) / (2.0 * hk);
            dd0[k][i] = (dvar(&sp, i) - dvar(&sm, i)) / (2.0 * hk);
            d2r0[k][k][i] = (var(&sp, i) - 2.0 * var(sigma, i) + var(&sm, i)) / (hk * hk);
        }
    }
    for k in 0..n_sigma {
        let hk = sigma_fd_step(sigma[k]);
        for l in (k + 1)..n_sigma {
            let hl = sigma_fd_step(sigma[l]);
            let mut spp = sigma.to_vec();
            spp[k] += hk;
            spp[l] += hl;
            let mut spm = sigma.to_vec();
            spm[k] += hk;
            spm[l] -= hl;
            let mut smp = sigma.to_vec();
            smp[k] -= hk;
            smp[l] += hl;
            let mut smm = sigma.to_vec();
            smm[k] -= hk;
            smm[l] -= hl;
            for i in 0..nq {
                let v =
                    (var(&spp, i) - var(&spm, i) - var(&smp, i) + var(&smm, i)) / (4.0 * hk * hl);
                d2r0[k][l][i] = v;
                d2r0[l][k][i] = v;
            }
        }
    }
    Foce0Sigma { dr0, d2r0, dd0 }
}

/// The **fixed-η̂** FOCE (Sheiner–Beal) covariance Hessian and gradient over the
/// natural `[θ, Ω, σ]` parameters, holding the mode `η̂` constant. The marginal is
/// `Fᵢ = ½[ρᵀR̃⁻¹ρ + log|R̃|]` with `R̃ = JΩJᵀ + diag(R⁰)`, `ρ = ε + Jη̂`,
/// `J = ∂f/∂η|_η̂`, `R⁰ = R(f(η=0), σ)`. The Hessian is the standard
/// Gaussian-marginal form
///
/// ```text
///   ∂²F/∂s∂t = ρ_st·u + ρ_s·u_t + ½tr(R̃⁻¹V_st) − ½tr(R̃⁻¹V_t R̃⁻¹V_s)
///            − u_t·V_s u − ½u·V_st u ,  u = R̃⁻¹ρ,  u_t = R̃⁻¹(ρ_t − V_t u),
/// ```
///
/// with `V = R̃`. `ρ` depends only on θ (through `f(η̂)` and `J`); `V` on θ (J, R⁰),
/// Ω (`JE_eJᵀ`, linear), and σ (R⁰). The θ second derivatives consume `∂³f/∂η∂θ²`
/// and `∂²f/∂θ²` (η̂ rows from `sens`, η=0 rows from `sens0`). Returns the
/// `(gradient, hessian)` pair; both share the SB setup. `None` outside scope or on
/// a BLOQ-censored subject (FOCE M3 censoring is a separate path). The moving-mode
/// (`η̂`-response) terms are added by the caller — this is the frozen part.
// Reachable only from this module's tests, which pin it against a frozen-mode FD
// reference. Said with `#[cfg(test)]` rather than by suppressing `dead_code`, so a future
// production caller disappearing is a compile error here instead of a silent allow (PR #953
// review, lower-priority items).
#[cfg(test)]
#[allow(clippy::type_complexity)]
fn foce_sb_fixed_natural(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    sens0: &SubjectSens,
    eta_hat: &[f64],
) -> Option<(Vec<f64>, DMatrix<f64>)> {
    let ne = model.n_eta;
    let nt = params.theta.len();
    let n_obs = subject.observations.len();
    // No-BLOQ scope: all observation rows are quantified.
    if model.bloq_method == crate::types::BloqMethod::M3 && subject.cens.iter().any(|&c| c != 0) {
        return None;
    }
    let nq = n_obs;
    if sens.obs.len() != nq || sens0.obs.len() != nq {
        return None;
    }
    let e_mats: Vec<_> = omega_entries(params.omega.diagonal, ne)
        .into_iter()
        .map(|(r, c)| e_matrix(r, c, ne))
        .collect();
    let n_omega = e_mats.len();
    let sigma = &params.sigma.values;
    let n_sigma = sigma.len();
    let dim = nt + n_omega + n_sigma;
    let nw = nt + n_omega;
    let omega = &params.omega.matrix;
    let cmts: Vec<usize> = (0..nq).map(|i| subject.obs_cmts[i]).collect();

    // J = ∂f/∂η (at η̂), ρ = ε + Jη̂, f0 = f(η=0), R⁰ and its f-derivatives.
    let mut jmat = DMatrix::<f64>::zeros(nq, ne);
    let mut rho = DVector::<f64>::zeros(nq);
    let mut f0 = vec![0.0; nq];
    let mut r0 = vec![0.0; nq];
    let mut d0 = vec![0.0; nq];
    let mut d20 = vec![0.0; nq];
    for i in 0..nq {
        let obs = &sens.obs[i];
        let mut jeta = 0.0;
        for k in 0..ne {
            jmat[(i, k)] = obs.df_deta[k];
            jeta += obs.df_deta[k] * eta_hat[k];
        }
        rho[i] = subject.observations[i] - (obs.f - jeta);
        let f0act = sens0.obs[i].f;
        f0[i] = f0act;
        let r = model.error_spec.variance_at(cmts[i], f0act, sigma);
        if !(r.is_finite() && r > 0.0) {
            return None;
        }
        r0[i] = r;
        d0[i] = model.error_spec.dvar_df(cmts[i], f0act, sigma);
        d20[i] = model.error_spec.d2var_df2(cmts[i], f0act, sigma);
    }
    let mut rtilde = &jmat * omega * jmat.transpose();
    for i in 0..nq {
        rtilde[(i, i)] += r0[i];
    }
    let rtilde_inv = rtilde.cholesky()?.inverse();
    let u = &rtilde_inv * &rho;
    let s0 = foce0_sigma(model, &cmts, &f0, sigma);

    // Per-direction ρ_s (nq) and V_s (nq×nq), s over natural [θ, Ω, σ].
    // B_m[i,l] = ∂²f/∂η_l∂θ_m (at η̂).
    let bcol = |m: usize| -> DMatrix<f64> {
        let mut bm = DMatrix::<f64>::zeros(nq, ne);
        for i in 0..nq {
            for l in 0..ne {
                bm[(i, l)] = sens.obs[i].d2f_deta_dtheta[l * nt + m];
            }
        }
        bm
    };
    let bmats: Vec<DMatrix<f64>> = (0..nt).map(bcol).collect();
    let mut rho_s: Vec<DVector<f64>> = Vec::with_capacity(dim);
    let mut v_s: Vec<DMatrix<f64>> = Vec::with_capacity(dim);
    // θ
    for m in 0..nt {
        let bm = &bmats[m];
        let mut r = DVector::<f64>::zeros(nq);
        for i in 0..nq {
            let mut be = 0.0;
            for l in 0..ne {
                be += bm[(i, l)] * eta_hat[l];
            }
            r[i] = -sens.obs[i].df_dtheta[m] + be;
        }
        rho_s.push(r);
        // V_θm = Bm Ω Jᵀ + J Ω Bmᵀ + diag(d⁰·∂f0/∂θ_m).
        let bmojt = bm * omega * jmat.transpose();
        let mut v = &bmojt + bmojt.transpose();
        for i in 0..nq {
            v[(i, i)] += d0[i] * sens0.obs[i].df_dtheta[m];
        }
        v_s.push(v);
    }
    // Ω
    for e in 0..n_omega {
        rho_s.push(DVector::zeros(nq));
        v_s.push(&jmat * &e_mats[e] * jmat.transpose());
    }
    // σ
    for k in 0..n_sigma {
        rho_s.push(DVector::zeros(nq));
        v_s.push(DMatrix::from_diagonal(&DVector::from_column_slice(
            &s0.dr0[k],
        )));
    }

    // Pre-multiplied K_s = R̃⁻¹ V_s for the trace-product term.
    let k_s: Vec<DMatrix<f64>> = v_s.iter().map(|v| &rtilde_inv * v).collect();
    let vu: Vec<DVector<f64>> = v_s.iter().map(|v| v * &u).collect();

    // ρ_st and V_st for a direction pair (only θθ, θΩ, θσ, σσ are nonzero).
    let rho_st = |a: usize, b: usize| -> DVector<f64> {
        let (lo, hi) = (a.min(b), a.max(b));
        if hi < nt {
            // θθ: ρ_θmθn[i] = −∂²f/∂θ² + Σ_l ∂³f/∂η_l∂θ_m∂θ_n · η̂_l.
            let (m, n) = (lo, hi);
            let mut r = DVector::<f64>::zeros(nq);
            for i in 0..nq {
                let mut s = -sens.obs[i].d2f_dtheta2[m * nt + n];
                for l in 0..ne {
                    s += sens.obs[i].d3f_deta_dtheta2[(l * nt + m) * nt + n] * eta_hat[l];
                }
                r[i] = s;
            }
            r
        } else {
            DVector::zeros(nq)
        }
    };
    let v_st = |a: usize, b: usize| -> DMatrix<f64> {
        let (lo, hi) = (a.min(b), a.max(b));
        if hi < nt {
            // θθ.
            let (m, n) = (lo, hi);
            let bm = &bmats[m];
            let bn = &bmats[n];
            // ∂Bm/∂θn[i,l] = ∂³f/∂η_l∂θ_m∂θ_n.
            let mut dbm = DMatrix::<f64>::zeros(nq, ne);
            for i in 0..nq {
                for l in 0..ne {
                    dbm[(i, l)] = sens.obs[i].d3f_deta_dtheta2[(l * nt + m) * nt + n];
                }
            }
            let t1 = &dbm * omega * jmat.transpose(); // (∂Bm/∂θn)ΩJᵀ
            let t2 = bm * omega * bn.transpose(); // Bm Ω Bnᵀ
            let mut v = &t1 + t1.transpose() + &t2 + t2.transpose();
            for i in 0..nq {
                let f0m = sens0.obs[i].df_dtheta[m];
                let f0n = sens0.obs[i].df_dtheta[n];
                let f0mn = sens0.obs[i].d2f_dtheta2[m * nt + n];
                v[(i, i)] += d20[i] * f0m * f0n + d0[i] * f0mn;
            }
            v
        } else if lo < nt && hi >= nt && hi < nw {
            // θΩ: Bm E_e Jᵀ + J E_e Bmᵀ.
            let m = lo;
            let e = hi - nt;
            let bm = &bmats[m];
            let t = bm * &e_mats[e] * jmat.transpose();
            &t + t.transpose()
        } else if lo < nt && hi >= nw {
            // θσ: diag((∂d⁰/∂σ_k)·∂f0/∂θ_m).
            let m = lo;
            let k = hi - nw;
            let mut v = DMatrix::<f64>::zeros(nq, nq);
            for i in 0..nq {
                v[(i, i)] = s0.dd0[k][i] * sens0.obs[i].df_dtheta[m];
            }
            v
        } else if lo >= nw {
            // σσ: diag(∂²R⁰/∂σ_k∂σ_l).
            let k = lo - nw;
            let l = hi - nw;
            DMatrix::from_diagonal(&DVector::from_column_slice(&s0.d2r0[k][l]))
        } else {
            // ΩΩ, Ωσ vanish (V linear in Ω, no σ).
            DMatrix::zeros(nq, nq)
        }
    };

    // Fixed-η̂ gradient: ∂F/∂s = ρ_s·u + ½tr(R̃⁻¹V_s) − ½u·V_s u.
    let mut grad = vec![0.0; dim];
    for s in 0..dim {
        grad[s] = rho_s[s].dot(&u) + 0.5 * k_s[s].trace() - 0.5 * u.dot(&vu[s]);
    }

    // Fixed-η̂ Hessian.
    let mut hess = DMatrix::zeros(dim, dim);
    for s in 0..dim {
        for t in s..dim {
            let rst = rho_st(s, t);
            let vst = v_st(s, t);
            // u_t = R̃⁻¹(ρ_t − V_t u).
            let u_t = &rtilde_inv * (&rho_s[t] - &vu[t]);
            let tr_vst = (&rtilde_inv * &vst).trace();
            let tr_prod = (&k_s[t] * &k_s[s]).trace();
            let uvstu = u.dot(&(&vst * &u));
            let val = rst.dot(&u) + rho_s[s].dot(&u_t) + 0.5 * tr_vst
                - 0.5 * tr_prod
                - u_t.dot(&vu[s])
                - 0.5 * uvstu;
            hess[(s, t)] = val;
            hess[(t, s)] = val;
        }
    }
    Some((grad, hess))
}

/// The full per-subject **FOCE** (Sheiner–Beal) covariance Hessian and gradient
/// over the natural `[θ, Ω, σ]` parameters, including the moving-mode response —
/// the total second/first derivative of `F̂ᵢ(ζ) = Fᵢ(ζ, η̂(ζ))`:
///
/// ```text
///   H[ξ,ζ] = F_{ξζ} + Σ_l(F_{ξη_l}η̂_{l,ζ} + F_{ζη_l}η̂_{l,ξ})
///          + Σ_{lm} F_{η_lη_m}η̂_{l,ξ}η̂_{m,ζ} + Σ_l c_l η̂_{l,ξζ},
///   g[ζ]   = F_ζ + Σ_l c_l η̂_{l,ζ} ,   c_l = F_{η_l} (the SB coupling),
/// ```
///
/// where `F_{st}` is the fixed-η̂ Gaussian-marginal second derivative
/// ([`foce_sb_fixed_natural`]) extended to η directions (`ρ_η_l = Dη̂`,
/// `V_η_l = D_lΩJᵀ + JΩD_lᵀ`, `D_l = ∂J/∂η_l = ∂²f/∂η∂η_l`; η cross-second
/// derivatives consume `∂³f/∂η³` and `∂³f/∂η²∂θ`), and `η̂_{l,ζ}`, `η̂_{l,ξζ}`
/// are the shared inner-mode responses ([`inner_eta_responses`]). `None` outside
/// scope. M3 rows contribute their scalar linearized-marginal tail while
/// quantified rows retain the Gaussian marginal block. `prep` carries the shared inner Hessian.
#[allow(clippy::type_complexity)]
fn subject_cov_hessian_foce_natural(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    sens0: &SubjectSens,
    prep: &Prep,
    eta_hat: &[f64],
) -> Option<(Vec<f64>, DMatrix<f64>)> {
    let ne = prep.n_eta;
    let nt = params.theta.len();
    let nq = subject.observations.len();
    let is_cens: Vec<bool> = (0..nq)
        .map(|i| {
            model.bloq_method == crate::types::BloqMethod::M3
                && subject.cens.get(i).copied().unwrap_or(0) != 0
        })
        .collect();
    if is_cens.iter().all(|&c| c) {
        return None;
    }
    if sens.obs.len() != nq || sens0.obs.len() != nq {
        return None;
    }
    let e_mats = covariance_basis(params, prep);
    let n_omega = e_mats.len();
    let sigma = &params.sigma.values;
    let n_sigma = sigma.len();
    let dim = nt + n_omega + n_sigma;
    let nw = nt + n_omega;
    let nd = dim + ne;
    let omega = prep
        .covariance_prior
        .as_ref()
        .map(|p| &p.matrix)
        .unwrap_or(&params.omega.matrix);
    let cmts: Vec<usize> = (0..nq).map(|i| subject.obs_cmts[i]).collect();
    let eta_vec = DVector::from_column_slice(eta_hat);

    // J, ρ, R̃, u, R⁰ and f-derivatives — as in `foce_sb_fixed_natural`.
    let mut jmat = DMatrix::<f64>::zeros(nq, ne);
    let mut rho = DVector::<f64>::zeros(nq);
    let mut linear_mean = vec![0.0; nq];
    let mut f0 = vec![0.0; nq];
    let mut r0 = vec![0.0; nq];
    let mut d0 = vec![0.0; nq];
    let mut d20 = vec![0.0; nq];
    for i in 0..nq {
        let obs = &sens.obs[i];
        let mut jeta = 0.0;
        for k in 0..ne {
            jmat[(i, k)] = obs.df_deta[k];
            jeta += obs.df_deta[k] * eta_hat[k];
        }
        linear_mean[i] = obs.f - jeta;
        rho[i] = subject.observations[i] - linear_mean[i];
        let f0act = sens0.obs[i].f;
        f0[i] = f0act;
        let r = model.error_spec.variance_at(cmts[i], f0act, sigma);
        if !(r.is_finite() && r > 0.0) {
            return None;
        }
        r0[i] = r;
        d0[i] = model.error_spec.dvar_df(cmts[i], f0act, sigma);
        d20[i] = model.error_spec.d2var_df2(cmts[i], f0act, sigma);
    }
    let mut rtilde = &jmat * omega * jmat.transpose();
    for i in 0..nq {
        rtilde[(i, i)] += r0[i];
    }
    let tail_var: Vec<f64> = (0..nq).map(|i| rtilde[(i, i)]).collect();
    // Censored rows leave the Gaussian marginal. Replacing their rows by an
    // independent unit-variance zero residual contributes exactly zero and lets
    // the quantified submatrix use the same dense formulas below.
    for i in 0..nq {
        if is_cens[i] {
            rho[i] = 0.0;
            for j in 0..nq {
                rtilde[(i, j)] = 0.0;
                rtilde[(j, i)] = 0.0;
            }
            rtilde[(i, i)] = 1.0;
        }
    }
    let rtilde_inv = rtilde.cholesky()?.inverse();
    let u = &rtilde_inv * &rho;
    let s0 = foce0_sigma(model, &cmts, &f0, sigma);

    // B_m[i,l] = ∂²f/∂η_l∂θ_m ; D_l[i,k] = ∂²f/∂η_k∂η_l (= ∂J/∂η_l).
    let bmats: Vec<DMatrix<f64>> = (0..nt)
        .map(|m| DMatrix::from_fn(nq, ne, |i, l| sens.obs[i].d2f_deta_dtheta[l * nt + m]))
        .collect();
    let dmats: Vec<DMatrix<f64>> = (0..ne)
        .map(|l| DMatrix::from_fn(nq, ne, |i, k| sens.obs[i].d2f_deta2[k * ne + l]))
        .collect();

    // ρ_s and V_s over extended directions [θ, Ω, σ, η].
    let mut rho_s: Vec<DVector<f64>> = Vec::with_capacity(nd);
    let mut v_s: Vec<DMatrix<f64>> = Vec::with_capacity(nd);
    for m in 0..nt {
        let bm = &bmats[m];
        let mut r = DVector::<f64>::zeros(nq);
        for i in 0..nq {
            let mut be = 0.0;
            for l in 0..ne {
                be += bm[(i, l)] * eta_hat[l];
            }
            r[i] = -sens.obs[i].df_dtheta[m] + be;
        }
        rho_s.push(r);
        let bmojt = bm * omega * jmat.transpose();
        let mut v = &bmojt + bmojt.transpose();
        for i in 0..nq {
            v[(i, i)] += d0[i] * sens0.obs[i].df_dtheta[m];
        }
        v_s.push(v);
    }
    for e in 0..n_omega {
        rho_s.push(DVector::zeros(nq));
        v_s.push(&jmat * &e_mats[e] * jmat.transpose());
    }
    for k in 0..n_sigma {
        rho_s.push(DVector::zeros(nq));
        v_s.push(DMatrix::from_diagonal(&DVector::from_column_slice(
            &s0.dr0[k],
        )));
    }
    for l in 0..ne {
        let dl = &dmats[l];
        rho_s.push(dl * &eta_vec); // ρ_η_l = D_l η̂
        let dlojt = dl * omega * jmat.transpose();
        v_s.push(&dlojt + dlojt.transpose());
    }
    let tail_vs: Vec<Vec<f64>> = v_s
        .iter()
        .map(|v| (0..nq).map(|i| v[(i, i)]).collect())
        .collect();
    let tail_mu_s: Vec<Vec<f64>> = rho_s
        .iter()
        .map(|r| r.iter().map(|v| -v).collect())
        .collect();
    for d in 0..nd {
        for i in 0..nq {
            if is_cens[i] {
                rho_s[d][i] = 0.0;
                for j in 0..nq {
                    v_s[d][(i, j)] = 0.0;
                    v_s[d][(j, i)] = 0.0;
                }
            }
        }
    }
    let k_s: Vec<DMatrix<f64>> = v_s.iter().map(|v| &rtilde_inv * v).collect();
    let vu: Vec<DVector<f64>> = v_s.iter().map(|v| v * &u).collect();

    let dir_of = |d: usize| -> Dir {
        if d < nt {
            Dir::Theta(d)
        } else if d < nw {
            Dir::Omega(d - nt)
        } else if d < dim {
            Dir::Sigma(d - nw)
        } else {
            Dir::Eta(d - dim)
        }
    };
    // ∂³f tensors at η̂. dB_n/∂η_l[i,k] = ∂³f/∂η_k∂η_l∂θ_n ; dD_l/∂η_m[i,k] = ∂³f/∂η_k∂η_l∂η_m.
    let dbn_deta = |n: usize, l: usize| -> DMatrix<f64> {
        DMatrix::from_fn(nq, ne, |i, k| {
            sens.obs[i].d3f_deta2_dtheta[(k * ne + l) * nt + n]
        })
    };
    let ddl_deta = |l: usize, m: usize| -> DMatrix<f64> {
        DMatrix::from_fn(nq, ne, |i, k| sens.obs[i].d3f_deta3[(k * ne + l) * ne + m])
    };

    let rho_st_raw = |aa: usize, bb: usize| -> DVector<f64> {
        let (lo, hi) = (aa.min(bb), aa.max(bb));
        match (dir_of(lo), dir_of(hi)) {
            (Dir::Theta(m), Dir::Theta(n)) => {
                let mut r = DVector::<f64>::zeros(nq);
                for i in 0..nq {
                    let mut s = -sens.obs[i].d2f_dtheta2[m * nt + n];
                    for l in 0..ne {
                        s += sens.obs[i].d3f_deta_dtheta2[(l * nt + m) * nt + n] * eta_hat[l];
                    }
                    r[i] = s;
                }
                r
            }
            (Dir::Theta(n), Dir::Eta(l)) => {
                // ρ_{θn η_l}[i] = Σ_k ∂³f/∂η_k∂η_l∂θ_n · η̂_k.
                let mut r = DVector::<f64>::zeros(nq);
                for i in 0..nq {
                    let mut s = 0.0;
                    for k in 0..ne {
                        s += sens.obs[i].d3f_deta2_dtheta[(k * ne + l) * nt + n] * eta_hat[k];
                    }
                    r[i] = s;
                }
                r
            }
            (Dir::Eta(l), Dir::Eta(m)) => {
                // ρ_{η_l η_m}[i] = Σ_k ∂³f/∂η_k∂η_l∂η_m · η̂_k + ∂²f/∂η_m∂η_l.
                let mut r = DVector::<f64>::zeros(nq);
                for i in 0..nq {
                    let mut s = sens.obs[i].d2f_deta2[m * ne + l];
                    for k in 0..ne {
                        s += sens.obs[i].d3f_deta3[(k * ne + l) * ne + m] * eta_hat[k];
                    }
                    r[i] = s;
                }
                r
            }
            _ => DVector::zeros(nq),
        }
    };
    let rho_st = |aa: usize, bb: usize| -> DVector<f64> {
        let mut out = rho_st_raw(aa, bb);
        for i in 0..nq {
            if is_cens[i] {
                out[i] = 0.0;
            }
        }
        out
    };
    let v_st = |aa: usize, bb: usize| -> DMatrix<f64> {
        let (lo, hi) = (aa.min(bb), aa.max(bb));
        match (dir_of(lo), dir_of(hi)) {
            (Dir::Theta(m), Dir::Theta(n)) => {
                let bm = &bmats[m];
                let bn = &bmats[n];
                // ∂Bm/∂θn[i,l] = ∂³f/∂η_l∂θ_m∂θ_n.
                let dbmn = DMatrix::from_fn(nq, ne, |i, l| {
                    sens.obs[i].d3f_deta_dtheta2[(l * nt + m) * nt + n]
                });
                let t1 = &dbmn * omega * jmat.transpose();
                let t2 = bm * omega * bn.transpose();
                let mut v = &t1 + t1.transpose() + &t2 + t2.transpose();
                for i in 0..nq {
                    let f0m = sens0.obs[i].df_dtheta[m];
                    let f0n = sens0.obs[i].df_dtheta[n];
                    let f0mn = sens0.obs[i].d2f_dtheta2[m * nt + n];
                    v[(i, i)] += d20[i] * f0m * f0n + d0[i] * f0mn;
                }
                v
            }
            (Dir::Theta(m), Dir::Omega(e)) => {
                let t = &bmats[m] * &e_mats[e] * jmat.transpose();
                &t + t.transpose()
            }
            (Dir::Theta(m), Dir::Sigma(k)) => {
                let mut v = DMatrix::<f64>::zeros(nq, nq);
                for i in 0..nq {
                    v[(i, i)] = s0.dd0[k][i] * sens0.obs[i].df_dtheta[m];
                }
                v
            }
            (Dir::Theta(n), Dir::Eta(l)) => {
                // V_{θn η_l} = (∂Bn/∂η_l)ΩJᵀ + Bn Ω D_lᵀ + D_l Ω Bnᵀ + JΩ(∂Bn/∂η_l)ᵀ.
                let dbnl = dbn_deta(n, l);
                let t1 = &dbnl * omega * jmat.transpose();
                let t2 = &bmats[n] * omega * dmats[l].transpose();
                &t1 + t1.transpose() + &t2 + t2.transpose()
            }
            (Dir::Omega(e), Dir::Eta(l)) => {
                let t = &dmats[l] * &e_mats[e] * jmat.transpose();
                &t + t.transpose()
            }
            (Dir::Sigma(k), Dir::Sigma(ll)) => {
                DMatrix::from_diagonal(&DVector::from_column_slice(&s0.d2r0[k][ll]))
            }
            (Dir::Eta(l), Dir::Eta(m)) => {
                // V_{η_l η_m} = (∂D_l/∂η_m)ΩJᵀ + D_lΩD_mᵀ + D_mΩD_lᵀ + JΩ(∂D_l/∂η_m)ᵀ.
                let ddlm = ddl_deta(l, m);
                let t1 = &ddlm * omega * jmat.transpose();
                let t2 = &dmats[l] * omega * dmats[m].transpose();
                &t1 + t1.transpose() + &t2 + t2.transpose()
            }
            // ΩΩ, Ωσ, ση (V σ indep of η) vanish.
            _ => DMatrix::zeros(nq, nq),
        }
    };

    // F_{st} REML second derivative for any extended directions.
    let mu_s = |s: usize, i: usize| tail_mu_s[s][i];
    let mu_st = |s: usize, t: usize, i: usize| -rho_st_raw(s, t)[i];
    let tails: Vec<Option<TailJet>> = (0..nq)
        .map(|i| {
            is_cens[i].then(|| {
                foce_tail_jet(
                    subject.observations[i],
                    linear_mean[i],
                    tail_var[i],
                    subject.cens[i],
                )
            })
        })
        .collect();
    let f_st = |s: usize, t: usize| -> f64 {
        let rst = rho_st(s, t);
        let mut vst = v_st(s, t);
        let tail_vst: Vec<f64> = (0..nq).map(|i| vst[(i, i)]).collect();
        for i in 0..nq {
            if is_cens[i] {
                for j in 0..nq {
                    vst[(i, j)] = 0.0;
                    vst[(j, i)] = 0.0;
                }
            }
        }
        let u_t = &rtilde_inv * (&rho_s[t] - &vu[t]);
        let mut out = rst.dot(&u) + rho_s[s].dot(&u_t) + 0.5 * (&rtilde_inv * &vst).trace()
            - 0.5 * (&k_s[t] * &k_s[s]).trace()
            - u_t.dot(&vu[s])
            - 0.5 * u.dot(&(&vst * &u));
        for i in 0..nq {
            if let Some(q) = &tails[i] {
                let (ms, mt) = (mu_s(s, i), mu_s(t, i));
                let (vs, vt) = (tail_vs[s][i], tail_vs[t][i]);
                out += q.mumu * ms * mt
                    + q.muvar * (ms * vt + vs * mt)
                    + q.varvar * vs * vt
                    + q.mu * mu_st(s, t, i)
                    + q.var * tail_vst[i];
            }
        }
        out
    };

    // Fixed-η̂ gradient and coupling c_l = F_{η_l}.
    let fixed_grad: Vec<f64> = (0..dim)
        .map(|s| {
            let mut out = rho_s[s].dot(&u) + 0.5 * k_s[s].trace() - 0.5 * u.dot(&vu[s]);
            for i in 0..nq {
                if let Some(q) = &tails[i] {
                    out += q.mu * mu_s(s, i) + q.var * tail_vs[s][i];
                }
            }
            out
        })
        .collect();
    let c: Vec<f64> = (0..ne)
        .map(|l| {
            let s = dim + l;
            let mut out = rho_s[s].dot(&u) + 0.5 * k_s[s].trace() - 0.5 * u.dot(&vu[s]);
            for i in 0..nq {
                if let Some(q) = &tails[i] {
                    out += q.mu * mu_s(s, i) + q.var * tail_vs[s][i];
                }
            }
            out
        })
        .collect();

    // Shared inner-mode responses.
    let (eta_d, eta_dd) = inner_eta_responses(model, subject, params, sens, prep, eta_hat);

    // Total gradient g[ζ] = F_ζ + Σ_l c_l η̂_{l,ζ}.
    let mut grad = vec![0.0; dim];
    for z in 0..dim {
        grad[z] = fixed_grad[z];
        for l in 0..ne {
            grad[z] += c[l] * eta_d[z][l];
        }
    }

    // Total Hessian.
    let mut hess = DMatrix::zeros(dim, dim);
    for xi in 0..dim {
        for ze in xi..dim {
            let mut val = f_st(xi, ze);
            for l in 0..ne {
                val += f_st(xi, dim + l) * eta_d[ze][l] + f_st(ze, dim + l) * eta_d[xi][l];
            }
            for l in 0..ne {
                for m in 0..ne {
                    val += f_st(dim + l, dim + m) * eta_d[xi][l] * eta_d[ze][m];
                }
            }
            for l in 0..ne {
                val += c[l] * eta_dd[xi][ze][l];
            }
            hess[(xi, ze)] = val;
            hess[(ze, xi)] = val;
        }
    }
    Some((grad, hess))
}

/// The exact per-subject FOCEI covariance Hessian `∂²Fᵢ/∂x²` in the optimizer's
/// **packed** space (log-θ / Cholesky-Ω / log-σ) — the analytic, finite-difference-free
/// replacement for `compute_covariance`'s per-subject contribution. Returns
/// `None` when the subject/model is outside the analytic-covariance scope (the
/// caller then falls back to the existing FD covariance for the whole population):
///
/// * `covariance_sensitivities` declines unsupported sensitivity-provider cases,
///   scaling, LTBS, closed-form Form-C readouts, custom residual magnitudes, and
///   non-Gaussian endpoints. Dual-evaluable ODE readouts are part of the ODE jet;
/// M3/BLOQ rows use scalar derivatives of their tail likelihood while retaining
/// the same prediction-sensitivity chain.
///
/// `eta_hat` must be the EBE for `unpack_params(x)`. The result is the per-subject
/// negative-log-likelihood Hessian; the covariance OFV is `2·Σᵢ Fᵢ`, so the caller
/// scales the summed contributions by 2.
pub(crate) fn subject_packed_cov_hessian(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
) -> Option<DMatrix<f64>> {
    let params = unpack_params(x, template);
    let sens = covariance_sensitivities(model, subject, &params.theta, eta_hat)?;
    let prep = prepare_covariance(model, subject, &params, &sens, eta_hat)?;
    let h_nat = subject_cov_hessian_natural(model, subject, &params, &sens, &prep, eta_hat);
    let g_nat = covariance_natural_gradient(model, subject, &params, &sens, &prep, eta_hat);
    Some(pack_natural_hessian(&h_nat, &g_nat, x, template))
}

/// Map a natural-space covariance Hessian `h_nat` and the matching natural
/// gradient `g_nat` (both ordered `[θ, Ω-entries, σ]`) into the optimizer's
/// **packed** space — log-θ (or identity-θ for sign-bearing params), Cholesky-Ω,
/// log-σ — via the exact second-order reparameterization chain
///
/// ```text
///   H^packed = Jᵀ H^nat J  +  Σ_e g_nat_e · ∇²_x ζ_e ,   J_{e,a} = ∂ζ_e/∂x_a.
/// ```
///
/// `x` is the packed point and `template` supplies the θ log/identity flags and Ω
/// structure. The θ/σ maps are scalar (`ζ = e^x` ⇒ `∂ζ/∂x = ∂²ζ/∂x² = ζ`, or
/// identity). The Ω map factors as `x → L → Ω = LLᵀ`: `L` is `e^x` on the
/// (log-packed) Cholesky diagonal and raw `x` off-diagonal, and `Ω_{rc} = Σ_k
/// L_{rk}L_{ck}` is quadratic in `L` (so `∂²Ω/∂L∂L` is constant). The natural Ω
/// entry and the packed Cholesky entry share the optimizer's lower-triangle order
/// ([`omega_entries`] = `pack_params`), so the two index sets line up.
///
/// Natural coordinates are `[θ, Ω_BSV, Ω_IOV, σ]` (omitting IOV when absent);
/// packed coordinates place the shared IOV Cholesky entries after σ.
/// `g_nat`/`h_nat` use the symmetric single-parameter Ω convention (an
/// off-diagonal entry sets both `Ω_{rc}` and `Ω_{cr}`), matching
/// [`subject_cov_hessian_natural`] and [`super::sens_outer_gradient::subject_omega_gradient`].
pub(crate) fn pack_natural_hessian(
    h_nat: &DMatrix<f64>,
    g_nat: &[f64],
    x: &[f64],
    template: &ModelParameters,
) -> DMatrix<f64> {
    let params = unpack_params(x, template);
    pack_natural_hessian_with_params(h_nat, g_nat, &params, template)
}

/// Apply the packing chain using an already-unpacked population parameter snapshot.
/// AGQ reuses this snapshot across subjects rather than unpacking it for every Hessian.
pub(crate) fn pack_natural_hessian_with_params(
    h_nat: &DMatrix<f64>,
    g_nat: &[f64],
    params: &ModelParameters,
    template: &ModelParameters,
) -> DMatrix<f64> {
    if let Some(iov) = &params.omega_iov {
        // Use one representative IOV block for the parameter transformation. The
        // occasion multiplicity is already carried by the tied natural basis.
        let nt = params.theta.len();
        let ne = params.omega.dim();
        let nk = iov.dim();
        let d = ne + nk;
        let mut l = DMatrix::zeros(d, d);
        l.view_mut((0, 0), (ne, ne)).copy_from(&params.omega.chol);
        l.view_mut((ne, ne), (nk, nk)).copy_from(&iov.chol);
        let diagonal = params.omega.diagonal && iov.diagonal;
        let mut names = params.omega.eta_names.clone();
        names.extend(iov.eta_names.iter().cloned());
        let omega = crate::types::OmegaMatrix::from_chol_factor(
            l,
            names,
            diagonal,
            DMatrix::from_element(d, d, true),
        );
        let mut expanded = params.clone();
        expanded.omega = omega;
        expanded.omega_iov = None;
        let mut expanded_template = template.clone();
        expanded_template.omega = expanded.omega.clone();
        expanded_template.omega_iov = None;
        let all = omega_entries(diagonal, d);
        let base: Vec<_> = omega_entries(params.omega.diagonal, ne)
            .iter()
            .map(|rc| nt + all.iter().position(|v| v == rc).unwrap())
            .collect();
        let occasions: Vec<_> = omega_entries(iov.diagonal, nk)
            .iter()
            .map(|&(r, c)| nt + all.iter().position(|v| *v == (ne + r, ne + c)).unwrap())
            .collect();
        let sigma: Vec<_> = (0..params.sigma.values.len())
            .map(|k| nt + all.len() + k)
            .collect();
        let natural: Vec<_> = (0..nt)
            .chain(base.iter().copied())
            .chain(occasions.iter().copied())
            .chain(sigma.iter().copied())
            .collect();
        let packed: Vec<_> = (0..nt)
            .chain(base.iter().copied())
            .chain(sigma.iter().copied())
            .chain(occasions.iter().copied())
            .collect();
        assert_eq!(h_nat.nrows(), natural.len(), "IOV natural direction count");
        let n = nt + all.len() + sigma.len();
        let mut h = DMatrix::zeros(n, n);
        let mut g = vec![0.0; n];
        for (i, &a) in natural.iter().enumerate() {
            g[a] = g_nat[i];
            for (j, &b) in natural.iter().enumerate() {
                h[(a, b)] = h_nat[(i, j)];
            }
        }
        let transformed = pack_natural_hessian_with_params(&h, &g, &expanded, &expanded_template);
        return DMatrix::from_fn(packed.len(), packed.len(), |i, j| {
            transformed[(packed[i], packed[j])]
        });
    }
    let n_eta = template.omega.dim();
    let nt = template.theta.len();
    let entries = omega_entries(template.omega.diagonal, n_eta);
    let n_omega = entries.len();
    let n_sigma = template.sigma.values.len();
    let nw = nt + n_omega;
    let dim = nt + n_omega + n_sigma;
    let theta = &params.theta;
    let l = &params.omega.chol;
    let sigma = &params.sigma.values;

    // ∂L_{is,js}/∂x_s: L_ii = e^{x} on the (log-packed) diagonal, raw x off it.
    let lp = |s: usize| -> f64 {
        let (r, c) = entries[s];
        if r == c {
            l[(r, c)]
        } else {
            1.0
        }
    };
    // ∂Ω_{rc}/∂L_{ij} = δ_{ri} L_{cj} + δ_{ci} L_{rj}.
    let domega_dl = |re: usize, ce: usize, is: usize, js: usize| -> f64 {
        (if re == is { l[(ce, js)] } else { 0.0 }) + (if ce == is { l[(re, js)] } else { 0.0 })
    };

    // Jacobian J_{e,a} = ∂ζ_e/∂x_a.
    let mut jmat = DMatrix::<f64>::zeros(dim, dim);
    for m in 0..nt {
        jmat[(m, m)] = if theta_packs_log(template.theta_lower[m]) {
            theta[m]
        } else {
            1.0
        };
    }
    for k in 0..n_sigma {
        jmat[(nw + k, nw + k)] = sigma[k];
    }
    for e in 0..n_omega {
        let (re, ce) = entries[e];
        for s in 0..n_omega {
            let (is, js) = entries[s];
            jmat[(nt + e, nt + s)] = domega_dl(re, ce, is, js) * lp(s);
        }
    }

    // Term 1: Jᵀ H^nat J.
    let mut hpack = jmat.transpose() * h_nat * &jmat;

    // Term 2: Σ_e g_nat_e · ∇²_x ζ_e (the reparameterization curvature).
    for m in 0..nt {
        if theta_packs_log(template.theta_lower[m]) {
            hpack[(m, m)] += g_nat[m] * theta[m];
        }
    }
    for k in 0..n_sigma {
        hpack[(nw + k, nw + k)] += g_nat[nw + k] * sigma[k];
    }
    for e in 0..n_omega {
        let (re, ce) = entries[e];
        let ge = g_nat[nt + e];
        if ge == 0.0 {
            continue;
        }
        for s in 0..n_omega {
            let (is, js) = entries[s];
            // Diagonal-in-x curvature ∂²L_{ii}/∂x² = L_{ii} (off-diagonal L is linear).
            if is == js {
                hpack[(nt + s, nt + s)] += ge * domega_dl(re, ce, is, js) * l[(is, js)];
            }
            // Constant ∂²Ω_{rc}/∂L_s∂L_t, scaled by the L→x chain ∂L_s/∂x ∂L_t/∂x.
            for t in 0..n_omega {
                let (it, jt) = entries[t];
                let d2 = ((re == is && ce == it && js == jt) as i32
                    + (ce == is && re == it && js == jt) as i32) as f64;
                if d2 != 0.0 {
                    hpack[(nt + s, nt + t)] += ge * d2 * lp(s) * lp(t);
                }
            }
        }
    }
    hpack
}

/// The exact per-subject **FOCE** (Sheiner–Beal) covariance Hessian `∂²Fᵢ/∂x²` in
/// packed space — the FOCE counterpart of [`subject_packed_cov_hessian`]. `None`
/// outside analytic scope. The
/// per-subject NLL Hessian; the covariance OFV is `2·Σᵢ Fᵢ`, so the caller scales
/// by 2.
pub(crate) fn subject_packed_cov_hessian_foce(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
) -> Option<DMatrix<f64>> {
    let params = unpack_params(x, template);
    let sens = covariance_sensitivities(model, subject, &params.theta, eta_hat)?;
    let zeros = vec![0.0; eta_hat.len()];
    let sens0 = covariance_sensitivities(model, subject, &params.theta, &zeros)?;
    let prep = prepare_covariance(model, subject, &params, &sens, eta_hat)?;
    let (grad, hess) =
        subject_cov_hessian_foce_natural(model, subject, &params, &sens, &sens0, &prep, eta_hat)?;
    Some(pack_natural_hessian(&hess, &grad, x, template))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::inner_optimizer::find_ebe;
    use crate::estimation::parameterization::pack_params;
    use crate::estimation::sens_outer_gradient::{
        prepare, subject_omega_gradient, subject_packed_gradient, subject_packed_gradient_foce,
        subject_sigma_gradient, subject_theta_gradient,
    };
    use crate::parser::model_parser::parse_model_string;
    use crate::sens::provider::{subject_sensitivities, subject_sensitivities_cov};
    use crate::types::{
        BloqMethod, CompiledModel, DoseEvent, GradientMethod, ModelParameters, OmegaMatrix, Subject,
    };
    use std::collections::HashMap;

    /// The new third-order error scalars α'' = ∂²α/∂f² and β' = ∂²p/∂f² (and the
    /// reproduced α', β) match a finite difference of the closed forms, taken on a
    /// combined error model `R(f) = a² + (b f)²` (so `d = 2b²f`, `d2 = 2b²`,
    /// `d3 = 0` — the assumption `err_d2` bakes in). This isolates the hand-derived
    /// α''/β' algebra from all model machinery.
    #[test]
    fn err_d2_scalar_matches_fd() {
        let (a, b, y) = (0.30_f64, 0.20_f64, 7.5_f64);
        // R, d, d2, eps as explicit functions of the prediction f.
        let rv = |f: f64| a * a + (b * f) * (b * f);
        let dv = |f: f64| 2.0 * b * b * f;
        let d2v = 2.0 * b * b;
        let epsv = |f: f64| y - f;
        let alpha_of = |f: f64| {
            let (r, d, eps) = (rv(f), dv(f), epsv(f));
            -2.0 * eps / r + d * (r - eps * eps) / (r * r)
        };
        let p_of = |f: f64| {
            let (r, d) = (rv(f), dv(f));
            1.0 / r + 0.5 * (d / r) * (d / r)
        };
        for &f in &[2.0_f64, 5.0, 9.0, 13.0] {
            let s = err_d2(rv(f), dv(f), d2v, epsv(f));
            let h = 1e-5 * (1.0 + f.abs());
            let ap_fd = (alpha_of(f + h) - alpha_of(f - h)) / (2.0 * h);
            let app_fd = (alpha_of(f + h) - 2.0 * alpha_of(f) + alpha_of(f - h)) / (h * h);
            let bp_fd = (p_of(f + h) - 2.0 * p_of(f) + p_of(f - h)) / (h * h);
            assert!(
                (s.alpha_p - ap_fd).abs() < 1e-6 * (1.0 + ap_fd.abs()),
                "α'({f}): {} vs {}",
                s.alpha_p,
                ap_fd
            );
            assert!(
                (s.alpha_pp - app_fd).abs() < 1e-4 * (1.0 + app_fd.abs()),
                "α''({f}): {} vs {}",
                s.alpha_pp,
                app_fd
            );
            assert!(
                (s.beta_p - bp_fd).abs() < 1e-4 * (1.0 + bp_fd.abs()),
                "β'({f}): {} vs {}",
                s.beta_p,
                bp_fd
            );
        }
    }

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

    const ODE_IV_COV: &str = r#"
[parameters]
  theta TVK(0.15, 0.01, 2.0)
  omega ETA_K ~ 0.09
  sigma ADD_ERR ~ 2.0
[individual_parameters]
  K = TVK * exp(ETA_K)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -K * central
[error_model]
  DV ~ additive(ADD_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    const ODE_IV_IOV_COV: &str = r#"
[parameters]
  theta TVK(0.15, 0.01, 2.0)
  omega ETA_K ~ 0.09
  kappa KAPPA_K ~ 0.04
  sigma ADD_ERR ~ 2.0
[individual_parameters]
  K = TVK * exp(ETA_K + KAPPA_K)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -K * central
[error_model]
  DV ~ additive(ADD_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    const CLOSED_FORM_IV_IOV_COV: &str = r#"
[parameters]
  theta TVK(0.15, 0.01, 2.0)
  omega ETA_K ~ 0.09
  kappa KAPPA_K ~ 0.04
  sigma ADD_ERR ~ 2.0
[individual_parameters]
  K = TVK * exp(ETA_K + KAPPA_K)
  V = 1
[structural_model]
  pk one_cpt_iv(cl=K, v=V)
[error_model]
  DV ~ additive(ADD_ERR)
"#;

    const FLIP_FLOP_TRANSIT_COV: &str = r#"
[parameters]
  theta TVCL(2.0, 0.001, 50.0)
  theta TVV(4.0, 0.1, 500.0)
  theta TVNTR(3.0, 0.0, 20.0)
  theta TVMTT(20.0, 0.05, 200.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  NTR = TVNTR
  MTT = TVMTT
[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)
[error_model]
  DV ~ proportional(PROP)
"#;

    fn ode_cov_subject(model: &CompiledModel, occasions: usize) -> Subject {
        let times: Vec<_> = (0..occasions)
            .flat_map(|k| [1.0, 3.0, 6.0, 11.0].map(|t| t + 12.0 * k as f64))
            .collect();
        let n = times.len();
        let mut subject = Subject {
            id: "ode-cov".into(),
            doses: (0..occasions)
                .map(|k| DoseEvent::new(12.0 * k as f64, 100.0, 1, 0.0, false, 0.0))
                .collect(),
            obs_times: times,
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
            occasions: (0..occasions)
                .flat_map(|k| vec![(k + 1) as u32; 4])
                .collect(),
            dose_occasions: (1..=occasions).map(|k| k as u32).collect(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_l2: Vec::new(),
            obs_records: vec![],
        };
        let eta = vec![0.10; model.n_eta];
        let preds = if model.n_kappa == 0 {
            crate::pk::compute_predictions_with_tv(
                model,
                &subject,
                &model.default_params.theta,
                &eta,
            )
        } else {
            let kappas: Vec<_> = (0..occasions)
                .map(|k| vec![if k % 2 == 0 { 0.08 } else { -0.06 }; model.n_kappa])
                .collect();
            crate::pk::predict_iov(model, &subject, &model.default_params.theta, &eta, &kappas)
        };
        subject.observations = preds
            .iter()
            .enumerate()
            .map(|(i, p)| p * (0.94 + 0.01 * (i % 3) as f64))
            .collect();
        subject
    }

    fn iov_cov_fixture(occasions: usize, block: bool) -> (CompiledModel, Subject) {
        let mut source = WARFARIN
            .replace(
                "sigma PROP_ERR ~ 0.04",
                "kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.15",
            )
            .replace(
                "CL = TVCL * exp(ETA_CL)",
                "CL = TVCL * exp(ETA_CL + KAPPA_CL)",
            );
        if block {
            source = source
                .replace(
                    "omega ETA_CL ~ 0.09\n  omega ETA_V  ~ 0.04",
                    "block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]",
                )
                .replace("omega ETA_KA ~ 0.30", "")
                .replace("KA = TVKA * exp(ETA_KA)", "KA = TVKA")
                .replace(
                    "kappa KAPPA_CL ~ 0.02",
                    "block_kappa (KAPPA_CL, KAPPA_V) = [0.02, 0.004, 0.03]",
                )
                .replace("V  = TVV  * exp(ETA_V)", "V  = TVV  * exp(ETA_V + KAPPA_V)");
        }
        let model = parse_model_string(&source).unwrap();
        let times: Vec<_> = (0..occasions)
            .flat_map(|k| [0.5, 2.0, 6.0, 12.0].map(|t| t + 24.0 * k as f64))
            .collect();
        let mut subject = warfarin_subject(&model, &model.default_params.theta, &times);
        subject.doses = (0..occasions)
            .map(|k| DoseEvent::new(24.0 * k as f64, 100.0, 1, 0.0, false, 0.0))
            .collect();
        subject.occasions = (0..occasions)
            .flat_map(|k| vec![(k + 1) as u32; 4])
            .collect();
        subject.dose_occasions = (1..=occasions).map(|k| k as u32).collect();
        let eta = vec![0.1; model.n_eta];
        let kappas: Vec<_> = (0..occasions)
            .map(|k| vec![if k % 2 == 0 { 0.12 } else { -0.08 }; model.n_kappa])
            .collect();
        let pred =
            crate::pk::predict_iov(&model, &subject, &model.default_params.theta, &eta, &kappas);
        subject.observations = pred
            .iter()
            .enumerate()
            .map(|(i, p)| p * (0.91 + 0.02 * (i % 3) as f64))
            .collect();
        (model, subject)
    }

    fn precise_iov_mode(model: &CompiledModel, subject: &Subject, p: &ModelParameters) -> Vec<f64> {
        let warm = find_ebe(model, subject, p, 200, 1e-10, None, None, 0);
        let mut b: Vec<_> = warm
            .eta
            .iter()
            .copied()
            .chain(warm.kappas.iter().flat_map(|k| k.iter().copied()))
            .collect();
        for _ in 0..30 {
            let sens =
                crate::sens::provider::subject_sensitivities_iov(model, subject, &p.theta, &b)
                    .unwrap();
            let prep = prepare_covariance(model, subject, p, &sens, &b).unwrap();
            let core = crate::estimation::sens_outer_gradient::score_core(
                model,
                subject,
                p,
                &sens,
                prep.n_eta,
                &prep.omega_inv,
                &b,
                None,
            )
            .unwrap();
            let mut g = &prep.omega_inv * DVector::from_column_slice(&b);
            for (o, e) in sens.obs.iter().zip(&core.et) {
                for k in 0..b.len() {
                    g[k] += 0.5 * e.alpha * o.df_deta[k];
                }
            }
            let step = core.h_inner.cholesky().unwrap().solve(&g);
            for k in 0..b.len() {
                b[k] -= step[k];
            }
            if step.amax() < 1e-12 {
                return b;
            }
        }
        panic!("IOV mode did not converge tightly");
    }

    #[test]
    fn iov_cov_hessian_matches_reconverged_gradient() {
        use crate::estimation::sens_outer_gradient::{
            subject_packed_gradient_foce_iov, subject_packed_gradient_iov,
        };
        for (occasions, block) in [(1, false), (2, false), (2, true)] {
            let (model, s) = iov_cov_fixture(occasions, block);
            let p = &model.default_params;
            let x = pack_params(p);
            let b = precise_iov_mode(&model, &s, p);
            for interaction in [false, true] {
                let h = if interaction {
                    subject_packed_cov_hessian(&model, &s, p, &x, &b)
                } else {
                    subject_packed_cov_hessian_foce(&model, &s, p, &x, &b)
                }
                .expect("IOV covariance in scope");
                assert_eq!(h.nrows(), x.len());
                for col in 0..x.len() {
                    let step = 2e-5 * (1.0 + x[col].abs());
                    let gradient = |sign: f64| {
                        let mut xp = x.clone();
                        xp[col] += sign * step;
                        let pp = unpack_params(&xp, p);
                        let bp = precise_iov_mode(&model, &s, &pp);
                        if interaction {
                            subject_packed_gradient_iov(&model, &s, p, &xp, &bp)
                        } else {
                            subject_packed_gradient_foce_iov(&model, &s, p, &xp, &bp)
                        }
                        .unwrap()
                    };
                    let gp = gradient(1.0);
                    let gm = gradient(-1.0);
                    for row in 0..x.len() {
                        let fd = (gp[row] - gm[row]) / (2.0 * step);
                        assert!((h[(row,col)]-fd).abs()<2e-4*h.amax().max(1.0),
                            "occasions={occasions} block={block} interaction={interaction} ({row},{col}) analytic={} FD={fd}",h[(row,col)]);
                    }
                }
            }
        }
    }

    #[test]
    fn iov_m3_cov_hessian_matches_reconverged_gradient() {
        use crate::estimation::sens_outer_gradient::{
            subject_packed_gradient_foce_iov, subject_packed_gradient_iov,
        };
        let (mut model, mut subject) = iov_cov_fixture(2, false);
        model.bloq_method = BloqMethod::M3;
        subject.cens[6] = 1;
        subject.cens[7] = -1;
        let p = &model.default_params;
        let x = pack_params(p);
        let b = precise_iov_mode(&model, &subject, p);
        for interaction in [false, true] {
            let h = if interaction {
                subject_packed_cov_hessian(&model, &subject, p, &x, &b)
            } else {
                subject_packed_cov_hessian_foce(&model, &subject, p, &x, &b)
            }
            .expect("IOV M3 covariance in scope");
            for col in 0..x.len() {
                let step = 1e-4 * (1.0 + x[col].abs());
                let gradient = |sign: f64| {
                    let mut xp = x.clone();
                    xp[col] += sign * step;
                    let pp = unpack_params(&xp, p);
                    let bp = precise_iov_mode(&model, &subject, &pp);
                    if interaction {
                        subject_packed_gradient_iov(&model, &subject, p, &xp, &bp)
                    } else {
                        subject_packed_gradient_foce_iov(&model, &subject, p, &xp, &bp)
                    }
                    .unwrap()
                };
                let gp = gradient(1.0);
                let gm = gradient(-1.0);
                for row in 0..x.len() {
                    let fd = (gp[row] - gm[row]) / (2.0 * step);
                    assert!(
                        (h[(row, col)] - fd).abs() < 3e-4 * h.amax().max(1.0),
                        "IOV M3 interaction={interaction} ({row},{col}) analytic={} FD={fd}",
                        h[(row, col)]
                    );
                }
            }
        }
    }

    #[test]
    fn iov_agq_cov_hessian_matches_reconverged_objective() {
        check_iov_agq_cov_hessian(false);
    }

    #[test]
    fn iov_m3_agq_cov_hessian_matches_reconverged_objective() {
        check_iov_agq_cov_hessian(true);
    }

    fn check_iov_agq_cov_hessian(m3: bool) {
        use crate::estimation::agq::{agq_population_nll, gauss_hermite, subject_grid_and_weights};
        use crate::estimation::agq_cov_hessian::subject_packed_agq_cov_hessian;
        use crate::types::{HessianAnchor, Population};
        let (mut model, mut s) = iov_cov_fixture(2, false);
        if m3 {
            model.bloq_method = BloqMethod::M3;
            s.cens[6] = 1;
            s.cens[7] = -1;
        }
        let p = &model.default_params;
        let x = pack_params(p);
        let b = precise_iov_mode(&model, &s, p);
        let (nodes, weights) = gauss_hermite(3);
        let (grid, pi) = subject_grid_and_weights(&model, &s, p, &b, &nodes, &weights).unwrap();
        assert_eq!(grid.len(), 3usize.pow(b.len() as u32));
        let h = subject_packed_agq_cov_hessian(&model, &s, p, p, &b, &grid, &pi).unwrap();
        let pop = Population {
            subjects: vec![s],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let f = |xv: &[f64]| {
            let pp = unpack_params(xv, p);
            let b = precise_iov_mode(&model, &pop.subjects[0], &pp);
            let eta = DVector::from_column_slice(&b[..model.n_eta]);
            let kap: Vec<_> = b[model.n_eta..]
                .chunks(model.n_kappa)
                .map(DVector::from_column_slice)
                .collect();
            agq_population_nll(
                &model,
                &pop,
                &pp,
                &[eta],
                &[kap],
                3,
                HessianAnchor::GaussNewton,
            )
        };
        let f0 = f(&x);
        for i in 0..x.len() {
            for j in i..x.len() {
                let hi = 3e-4 * (1.0 + x[i].abs());
                let hj = 3e-4 * (1.0 + x[j].abs());
                let bump = |si: f64, sj: f64| {
                    let mut xp = x.clone();
                    xp[i] += si * hi;
                    xp[j] += sj * hj;
                    f(&xp)
                };
                let fd = if i == j {
                    (bump(1.0, 0.0) - 2.0 * f0 + bump(-1.0, 0.0)) / (hi * hi)
                } else {
                    (bump(1.0, 1.0) - bump(1.0, -1.0) - bump(-1.0, 1.0) + bump(-1.0, -1.0))
                        / (4.0 * hi * hj)
                };
                assert!(
                    (h[(i, j)] - fd).abs() < 2e-3 * h.amax().max(1.0),
                    "IOV AGQ M3={m3} ({i},{j}) analytic={} FD={fd}",
                    h[(i, j)]
                );
            }
        }
    }

    #[test]
    fn iov_cov_hessian_preserves_scope_exclusions() {
        use crate::estimation::agq_cov_hessian::prepare_mode;
        let (mut model, mut subject) = iov_cov_fixture(2, false);
        let params = model.default_params.clone();
        let x = pack_params(&params);
        let b = vec![0.1; model.n_eta + 2 * model.n_kappa];
        let declined = |m: &CompiledModel, s: &Subject| {
            assert!(subject_packed_cov_hessian(m, s, &params, &x, &b).is_none());
            assert!(subject_packed_cov_hessian_foce(m, s, &params, &x, &b).is_none());
            assert!(prepare_mode(m, s, &params, &b).is_none());
        };
        model.log_transform = true;
        declined(&model, &subject);
        model.log_transform = false;
        model.bloq_method = BloqMethod::M3;
        subject.cens[0] = 1;
        let m3b = precise_iov_mode(&model, &subject, &params);
        assert!(subject_packed_cov_hessian(&model, &subject, &params, &x, &m3b).is_some());
        assert!(prepare_mode(&model, &subject, &params, &m3b).is_some());
        assert!(subject_packed_cov_hessian_foce(&model, &subject, &params, &x, &m3b).is_some());
        subject.cens[0] = 0;
        model.bloq_method = BloqMethod::Drop;
        model.gradient_method = GradientMethod::Fd;
        declined(&model, &subject);
        model.gradient_method = GradientMethod::Auto;
        assert!(covariance_sensitivities(&model, &subject, &params.theta, &b).is_some());

        let source = WARFARIN
            .replace(
                "sigma PROP_ERR ~ 0.04",
                "kappa KAPPA_CL ~ 0.02\n sigma PROP_ERR ~ 0.04",
            )
            .replace(
                "CL = TVCL * exp(ETA_CL)",
                "CL = TVCL * exp(ETA_CL + KAPPA_CL)",
            )
            .replace("proportional(PROP_ERR)", "proportional(PROP_ERR * (TVKA))");
        let custom = parse_model_string(&source).unwrap();
        assert!(custom.has_custom_ruv_magnitude());
        declined(&custom, &subject);
    }

    #[test]
    fn iov_cov_hessian_dispatch_preserves_joint_modes_and_method() {
        use crate::estimation::agq::{gauss_hermite, subject_grid_and_weights};
        use crate::estimation::agq_cov_hessian::subject_packed_agq_cov_hessian;
        use crate::estimation::covariance::analytic_cov_hessian;
        use crate::types::{EstimationMethod, FitOptions, Population};
        let (model, s2) = iov_cov_fixture(2, false);
        let (_, mut s1) = iov_cov_fixture(1, false);
        s1.id = "one-occasion".into();
        let pop = Population {
            subjects: vec![s1, s2],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let p = &model.default_params;
        let x = pack_params(p);
        let at = unpack_params(&x, p);
        let modes: Vec<_> = pop
            .subjects
            .iter()
            .map(|s| precise_iov_mode(&model, s, &at))
            .collect();
        let eta: Vec<_> = modes
            .iter()
            .map(|b| DVector::from_column_slice(&b[..model.n_eta]))
            .collect();
        let kap: Vec<Vec<_>> = modes
            .iter()
            .map(|b| {
                b[model.n_eta..]
                    .chunks(model.n_kappa)
                    .map(DVector::from_column_slice)
                    .collect()
            })
            .collect();
        for (method, n_agq, interaction) in [
            (EstimationMethod::Foce, 1, false),
            (EstimationMethod::FoceI, 1, true),
            (EstimationMethod::FoceI, 3, true),
        ] {
            let opts = FitOptions {
                method,
                n_agq,
                interaction,
                ..FitOptions::default()
            };
            let actual = analytic_cov_hessian(&model, &pop, p, &x, &eta, &kap, &opts)
                .expect("IOV route must be used");
            let mut expected = DMatrix::zeros(x.len(), x.len());
            for (s, b) in pop.subjects.iter().zip(&modes) {
                let h = if n_agq > 1 {
                    let (nodes, weights) = gauss_hermite(n_agq);
                    let (grid, pi) =
                        subject_grid_and_weights(&model, s, &at, b, &nodes, &weights).unwrap();
                    subject_packed_agq_cov_hessian(&model, s, p, &at, b, &grid, &pi)
                } else if interaction {
                    subject_packed_cov_hessian(&model, s, p, &x, b)
                } else {
                    subject_packed_cov_hessian_foce(&model, s, p, &x, b)
                }
                .unwrap();
                expected += 2.0 * h;
            }
            assert_eq!(
                actual, expected,
                "the population route must sum the selected NLL Hessians with OFV scaling"
            );
            assert!(
                analytic_cov_hessian(&model, &pop, p, &x, &eta, &[], &opts).is_none(),
                "missing joint modes must decline"
            );
        }
        for n_agq in [1, 3] {
            let opts = FitOptions {
                method: EstimationMethod::Laplace,
                n_agq,
                ..FitOptions::default()
            };
            assert!(
                analytic_cov_hessian(&model, &pop, p, &x, &eta, &kap, &opts).is_none(),
                "IOV must not admit the exact-anchor Laplace covariance"
            );
        }
    }

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
            reset_covariates: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_l2: Vec::new(),
            // Declared unconditionally since Phase 4.0 — the `#[cfg(feature = "survival")]`
            // duplicate this fixture carried over from the ported branch broke
            // `--features survival` with E0062 (PR #953 review finding 6).
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
        let warm = find_ebe(model, subject, params, 80, 1e-10, None, None, 0);
        let mut eta: Vec<f64> = warm.eta.iter().copied().collect();
        let n_eta = model.n_eta;
        let omega_inv = &params.omega.inv;
        for _ in 0..60 {
            let sens = subject_sensitivities(model, subject, &params.theta, &eta).unwrap();
            let prep = prepare(model, subject, params, &sens, &eta).unwrap();
            let mut grad = omega_inv * DVector::from_column_slice(&eta);
            let mut hess = omega_inv.clone();
            for (j, obs) in sens.obs.iter().enumerate() {
                // inner gradient ½α·a, true Hessian ½(α' a aᵀ + α A).
                let alpha = prep.et[j].alpha;
                let alpha_p = prep.et[j].alpha_p;
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
        let prep = prepare(model, subject, params, &sens, &eta).unwrap();
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

    /// The full FOCEI natural gradient `[∂F/∂θ, ∂F/∂Ω, ∂F/∂σ]` (Φ **and**
    /// `½log|H̃|`, including the EBE response) at the reconverged precise mode —
    /// the NONMEM-validated analytic gradient. Finite-differencing this over the
    /// natural parameters (reconverging the mode each step, via `precise_ebe`
    /// inside) is the gold-standard target for the total `M2 + M3` Hessian.
    fn full_natural_grad(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
    ) -> Vec<f64> {
        let eta = precise_ebe(model, subject, params);
        let gt = subject_theta_gradient(model, subject, params, &eta).unwrap();
        let go = subject_omega_gradient(model, subject, params, &eta).unwrap();
        let gs = subject_sigma_gradient(model, subject, params, &eta).unwrap();
        [gt, go, gs].concat()
    }

    /// Validate the full natural `M2 + M3` covariance Hessian against a
    /// reconverged precise-EBE finite difference of [`full_natural_grad`].
    fn check_full_natural(model: &CompiledModel, subject: &Subject, params: &ModelParameters) {
        let eta = precise_ebe(model, subject, params);
        let sens = subject_sensitivities_cov(model, subject, &params.theta, &eta).unwrap();
        let prep = prepare(model, subject, params, &sens, &eta).unwrap();
        let analytic = subject_cov_hessian_natural(model, subject, params, &sens, &prep, &eta);

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
            let gp = full_natural_grad(
                model,
                subject,
                &perturb_natural(params, n_theta, &entries, col, h),
            );
            let gm = full_natural_grad(
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
                let tol = 2e-3 * (1.0 + a.abs());
                assert!(
                    (a - f).abs() < tol,
                    "Hessian[{},{}]: analytic {:.8e} vs FD {:.8e} (Δ {:.2e})",
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

    /// The full FOCEI **packed** gradient `∂F/∂x` at the reconverged precise mode
    /// for `unpack_params(x)` — the existing NONMEM-validated analytic packed
    /// gradient. FD of this over `x` is the gold-standard target for the packed
    /// covariance Hessian.
    fn full_packed_grad(
        model: &CompiledModel,
        subject: &Subject,
        template: &ModelParameters,
        x: &[f64],
    ) -> Vec<f64> {
        let params = unpack_params(x, template);
        let eta = precise_ebe(model, subject, &params);
        subject_packed_gradient(model, subject, template, x, &eta).unwrap()
    }

    /// Validate the packed-space chain: the analytic packed Hessian (natural
    /// `M2 + M3` chained through `pack_natural_hessian`) against a reconverged
    /// precise-EBE finite difference of the analytic packed gradient.
    fn check_packed(model: &CompiledModel, subject: &Subject, params: &ModelParameters) {
        let x = pack_params(params);
        let dim = x.len();
        let eta = precise_ebe(model, subject, params);
        let sens = subject_sensitivities_cov(model, subject, &params.theta, &eta).unwrap();
        let prep = prepare(model, subject, params, &sens, &eta).unwrap();
        let h_nat = subject_cov_hessian_natural(model, subject, params, &sens, &prep, &eta);
        let g_nat = full_natural_grad(model, subject, params);
        let analytic = pack_natural_hessian(&h_nat, &g_nat, &x, params);

        let mut fd = DMatrix::zeros(dim, dim);
        for col in 0..dim {
            let h = 1e-6 * (1.0 + x[col].abs());
            let mut xp = x.clone();
            xp[col] += h;
            let mut xm = x.clone();
            xm[col] -= h;
            let gp = full_packed_grad(model, subject, params, &xp);
            let gm = full_packed_grad(model, subject, params, &xm);
            for row in 0..dim {
                fd[(row, col)] = (gp[row] - gm[row]) / (2.0 * h);
            }
        }

        for row in 0..dim {
            for col in 0..dim {
                let a = analytic[(row, col)];
                let f = fd[(row, col)];
                let tol = 2e-3 * (1.0 + a.abs());
                assert!(
                    (a - f).abs() < tol,
                    "packed H[{},{}]: analytic {:.8e} vs FD {:.8e} (Δ {:.2e})",
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

    /// Validate the **fixed-η̂** FOCE (Sheiner–Beal) natural Hessian against a
    /// frozen finite difference of its own gradient (η̂ held constant). Isolates
    /// the marginal's explicit second derivative, before the mode response.
    fn check_foce_fixed(model: &CompiledModel, subject: &Subject, params: &ModelParameters) {
        let eta = precise_ebe(model, subject, params);
        let zeros = vec![0.0; model.n_eta];
        let grad_at = |p: &ModelParameters| -> Vec<f64> {
            let sens = subject_sensitivities_cov(model, subject, &p.theta, &eta).unwrap();
            let sens0 = subject_sensitivities_cov(model, subject, &p.theta, &zeros).unwrap();
            foce_sb_fixed_natural(model, subject, p, &sens, &sens0, &eta)
                .unwrap()
                .0
        };
        let sens = subject_sensitivities_cov(model, subject, &params.theta, &eta).unwrap();
        let sens0 = subject_sensitivities_cov(model, subject, &params.theta, &zeros).unwrap();
        let (_, hess) = foce_sb_fixed_natural(model, subject, params, &sens, &sens0, &eta).unwrap();

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
            let gp = grad_at(&perturb_natural(params, n_theta, &entries, col, h));
            let gm = grad_at(&perturb_natural(params, n_theta, &entries, col, -h));
            for row in 0..dim {
                fd[(row, col)] = (gp[row] - gm[row]) / (2.0 * h);
            }
        }
        for row in 0..dim {
            for col in 0..dim {
                let a = hess[(row, col)];
                let f = fd[(row, col)];
                let tol = 1e-4 * (1.0 + a.abs());
                assert!(
                    (a - f).abs() < tol,
                    "FOCE-fixed H[{},{}]: analytic {:.8e} vs FD {:.8e} (Δ {:.2e})",
                    row,
                    col,
                    a,
                    f,
                    (a - f).abs()
                );
            }
        }
    }

    /// Validate the full FOCE (Sheiner–Beal) natural Hessian — fixed part **plus**
    /// the η̂-mode response — by chaining it to packed space and comparing against a
    /// reconverged precise-EBE FD of the exact FOCE packed gradient.
    fn check_foce_full(model: &CompiledModel, subject: &Subject, params: &ModelParameters) {
        let x = pack_params(params);
        let dim = x.len();
        let eta = precise_ebe(model, subject, params);
        let zeros = vec![0.0; model.n_eta];
        let sens = subject_sensitivities_cov(model, subject, &params.theta, &eta).unwrap();
        let sens0 = subject_sensitivities_cov(model, subject, &params.theta, &zeros).unwrap();
        let prep = prepare(model, subject, params, &sens, &eta).unwrap();
        let (grad, hess) =
            subject_cov_hessian_foce_natural(model, subject, params, &sens, &sens0, &prep, &eta)
                .unwrap();
        let analytic = pack_natural_hessian(&hess, &grad, &x, params);
        let grad_packed = |xv: &[f64]| -> Vec<f64> {
            let p = unpack_params(xv, params);
            let e = precise_ebe(model, subject, &p);
            subject_packed_gradient_foce(model, subject, params, xv, &e).unwrap()
        };
        let mut fd = DMatrix::zeros(dim, dim);
        for col in 0..dim {
            let h = if model.bloq_method == BloqMethod::M3 {
                1e-4 * (1.0 + x[col].abs())
            } else {
                1e-6 * (1.0 + x[col].abs())
            };
            let mut xp = x.clone();
            xp[col] += h;
            let mut xm = x.clone();
            xm[col] -= h;
            let gp = grad_packed(&xp);
            let gm = grad_packed(&xm);
            for row in 0..dim {
                fd[(row, col)] = (gp[row] - gm[row]) / (2.0 * h);
            }
        }
        for row in 0..dim {
            for col in 0..dim {
                let a = analytic[(row, col)];
                let f = fd[(row, col)];
                let tol = 2e-3 * (1.0 + a.abs());
                assert!(
                    (a - f).abs() < tol,
                    "FOCE-full packed H[{},{}]: analytic {:.8e} vs FD {:.8e} (Δ {:.2e})",
                    row,
                    col,
                    a,
                    f,
                    (a - f).abs()
                );
            }
        }
    }

    /// ODE covariance uses the same sensitivity-level assembly as closed form: a central
    /// difference of the augmented `Dual2` jet supplies the third-order prediction blocks.
    /// Check both conditional (FOCEI) and linearised-marginal (FOCE) Hessians against a
    /// reconverged-gradient oracle on a tightly solved one-state ODE.
    #[test]
    fn ode_cov_hessian_matches_reconverged_gradients() {
        let form_c = ODE_IV_COV
            .replace(
                "ode(obs_cmt=central, states=[central])",
                "ode(states=[central])",
            )
            .replace("[error_model]", "[scaling]\n  y = central\n[error_model]");
        for source in [ODE_IV_COV, form_c.as_str()] {
            let model = parse_model_string(source).expect("parse ODE covariance fixture");
            let subject = ode_cov_subject(&model, 1);
            check_full_natural(&model, &subject, &model.default_params);
            check_foce_full(&model, &subject, &model.default_params);
        }
    }

    /// The ODE-specific step scaling must make the result usable at the public default
    /// `ode_reltol`, rather than requiring users to discover a hidden tight-tolerance
    /// prerequisite. Compare it with the same model solved at 1e-10.
    #[test]
    fn ode_cov_hessian_is_stable_at_default_solver_tolerance() {
        let tight = parse_model_string(ODE_IV_COV).expect("parse tight ODE fixture");
        let loose_text = ODE_IV_COV.replace(
            "[fit_options]\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
            "",
        );
        let loose = parse_model_string(&loose_text).expect("parse default-tolerance ODE fixture");
        let subject = ode_cov_subject(&tight, 1);

        let packed = |model: &CompiledModel| {
            let p = &model.default_params;
            let x = pack_params(p);
            let eta = precise_ebe(model, &subject, p);
            subject_packed_cov_hessian(model, &subject, p, &x, &eta)
                .expect("ODE covariance must remain analytic at the default tolerance")
        };
        let h_tight = packed(&tight);
        let h_loose = packed(&loose);
        let scale = h_tight.amax().max(1.0);
        assert!(
            (&h_loose - &h_tight).amax() < 2e-2 * scale,
            "default-tolerance ODE Hessian drifted too far from tight solve: max Δ={}, scale={scale}",
            (&h_loose - &h_tight).amax()
        );
    }

    /// A closed-form transit model can select its ODE twin only after evaluating the
    /// current parameters. The tolerance lookup must use that same effective model;
    /// `effective_for(subject)` alone does not see this flip-flop reroute.
    #[test]
    fn ode_cov_flip_flop_reroute_uses_the_selected_twins_tolerance() {
        let model = parse_model_string(FLIP_FLOP_TRANSIT_COV).expect("parse transit fixture");
        let subject = ode_cov_subject(&model, 1);
        let eta = precise_ebe(&model, &subject, &model.default_params);
        assert!(
            crate::pk::effective_model_for_eval(
                &model,
                &subject,
                &model.default_params.theta,
                &eta,
            )
            .ode_spec
            .is_some(),
            "fixture must select the ODE twin at the fitted mode"
        );
        assert!(
            subject_sensitivities_cov(&model, &subject, &model.default_params.theta, &eta)
                .is_some(),
            "parameter-selected ODE twin must remain in analytic covariance scope"
        );
    }

    /// Production dispatch accepts ODE jets for every Gauss-Newton-anchored covariance
    /// objective, including the largest shared extension: IOV + M3 under FOCE, FOCEI, and
    /// multi-node FOCEI-AGQ. Exact-anchor Laplace remains a separate fourth-order problem.
    #[test]
    fn ode_cov_iov_m3_dispatch_matrix() {
        use crate::estimation::covariance::analytic_cov_hessian;
        use crate::types::{EstimationMethod, FitOptions, Population};

        let mut model = parse_model_string(ODE_IV_IOV_COV).expect("parse ODE IOV fixture");
        model.bloq_method = BloqMethod::M3;
        let mut reference =
            parse_model_string(CLOSED_FORM_IV_IOV_COV).expect("parse closed-form IOV reference");
        reference.bloq_method = BloqMethod::M3;
        let mut subject = ode_cov_subject(&model, 2);
        subject.cens[2] = 1;
        subject.cens[6] = -1;
        let population = Population {
            subjects: vec![subject.clone()],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let p = &model.default_params;
        let x = pack_params(p);
        let b = precise_iov_mode(&model, &subject, p);
        let eta = vec![DVector::from_column_slice(&b[..model.n_eta])];
        let kappas = vec![b[model.n_eta..]
            .chunks(model.n_kappa)
            .map(DVector::from_column_slice)
            .collect::<Vec<_>>()];
        let reference_b = precise_iov_mode(&reference, &subject, &reference.default_params);
        let reference_eta = vec![DVector::from_column_slice(&reference_b[..reference.n_eta])];
        let reference_kappas = vec![reference_b[reference.n_eta..]
            .chunks(reference.n_kappa)
            .map(DVector::from_column_slice)
            .collect::<Vec<_>>()];

        for (method, n_agq, interaction) in [
            (EstimationMethod::Foce, 1, false),
            (EstimationMethod::FoceI, 1, true),
            (EstimationMethod::FoceI, 3, true),
        ] {
            let opts = FitOptions {
                method,
                n_agq,
                interaction,
                ..FitOptions::default()
            };
            let h = analytic_cov_hessian(&model, &population, p, &x, &eta, &kappas, &opts)
                .expect("ODE IOV + M3 must stay on the analytic covariance route");
            let h_reference = analytic_cov_hessian(
                &reference,
                &population,
                &reference.default_params,
                &pack_params(&reference.default_params),
                &reference_eta,
                &reference_kappas,
                &opts,
            )
            .expect("closed-form IOV + M3 reference must be analytic");
            assert_eq!((h.nrows(), h.ncols()), (x.len(), x.len()));
            assert!(h.iter().all(|v| v.is_finite()));
            assert!((&h - h.transpose()).amax() < 1e-8);
            let scale = h_reference.amax().max(1.0);
            assert!(
                (&h - &h_reference).amax() < 3e-3 * scale,
                "{method:?} n_agq={n_agq}: ODE covariance must match its closed-form twin; max Δ={}, scale={scale}",
                (&h - &h_reference).amax()
            );
        }

        let laplace = FitOptions {
            method: EstimationMethod::Laplace,
            n_agq: 1,
            interaction: true,
            ..FitOptions::default()
        };
        assert!(
            analytic_cov_hessian(&model, &population, p, &x, &eta, &kappas, &laplace).is_none(),
            "exact-anchor Laplace still needs fourth-order prediction derivatives"
        );
    }

    /// Warfarin (1-cpt, diagonal Ω): full FOCE Hessian (with mode response) vs
    /// reconverged FD of the FOCE packed gradient.
    #[test]
    fn cov_hessian_foce_full_matches_reconverged_fd_diagonal() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.2, 10.0, 1.5];
        let subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        let mut params = model.default_params.clone();
        params.theta = theta;
        check_foce_full(&model, &subject, &params);
    }

    #[test]
    fn cov_hessian_foce_m3_matches_reconverged_gradient() {
        let mut model = parse_model_string(WARFARIN).expect("parse");
        model.bloq_method = BloqMethod::M3;
        let theta = vec![0.2, 10.0, 1.5];
        let mut subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        subject.cens[5] = 1;
        subject.cens[6] = -1;
        let mut params = model.default_params.clone();
        params.theta = theta;
        check_foce_full(&model, &subject, &params);
    }

    /// Block-Ω: full FOCE Hessian with mode response (off-diagonal Ω + η coupling).
    #[test]
    fn cov_hessian_foce_full_matches_reconverged_fd_block_omega() {
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
        check_foce_full(&model, &subject, &params);
    }

    /// Warfarin (1-cpt, diagonal Ω): fixed-η̂ FOCE SB Hessian vs frozen FD.
    #[test]
    fn cov_hessian_foce_fixed_matches_frozen_fd_diagonal() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.2, 10.0, 1.5];
        let subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        let mut params = model.default_params.clone();
        params.theta = theta;
        check_foce_fixed(&model, &subject, &params);
    }

    /// Block-Ω: fixed-η̂ FOCE SB Hessian vs frozen FD (off-diagonal Ω curvature).
    #[test]
    fn cov_hessian_foce_fixed_matches_frozen_fd_block_omega() {
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
        check_foce_fixed(&model, &subject, &params);
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
        let prep = prepare(model, subject, params, &sens, &eta).unwrap();
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

    #[test]
    fn agq_cov_hessian_mode_anchor_matches_the_direct_objective_matrix() {
        use crate::estimation::agq_cov_hessian::{prepare_mode, regularised_anchor};
        use crate::estimation::sens_outer_gradient::score_core;

        let model = parse_model_string(WARFARIN).unwrap();
        let params = model.default_params.clone();
        let subject = warfarin_subject(&model, &params.theta, &[0.5, 2.0, 8.0, 24.0]);
        let eta = [0.17, -0.11, 0.23];
        let (_, prep, anchor) = prepare_mode(&model, &subject, &params, &eta).unwrap();
        // The objective uses the ordinary sensitivity provider and the direct ScoreCore.
        let sens = subject_sensitivities(&model, &subject, &params.theta, &eta).unwrap();
        let core = score_core(
            &model,
            &subject,
            &params,
            &sens,
            model.n_eta,
            &params.omega.inv,
            &eta,
            model.residual_error_eta,
        )
        .unwrap();
        let expected = regularised_anchor(&core.htilde).unwrap();
        assert_eq!(
            anchor.s, expected.s,
            "covariance must use the objective's exact anchor bytes"
        );
        let roundtrip = regularised_anchor(&prep.htilde_inv.try_inverse().unwrap()).unwrap();
        assert_ne!(
            roundtrip.s, expected.s,
            "fixture must expose the old double-inversion roundoff"
        );
    }

    #[test]
    fn agq_cov_hessian_ignores_unusable_zero_weight_nodes() {
        use crate::estimation::agq_cov_hessian::{node_jet, prepare_mode, subject_agq_cov_hessian};
        let model = parse_model_string(WARFARIN).unwrap();
        let p = &model.default_params;
        let s = warfarin_subject(&model, &p.theta, &[0.5, 2.0, 8.0, 24.0]);
        let eta = precise_ebe(&model, &s, p);
        let central = vec![vec![0.0; model.n_eta]];
        let expected = subject_agq_cov_hessian(&model, &s, p, &eta, &central, &[1.0]).unwrap();
        let (_, _, anchor) = prepare_mode(&model, &s, p, &eta).unwrap();
        let tail = vec![1e6; model.n_eta];
        let bad = DVector::from_column_slice(&eta)
            + std::f64::consts::SQRT_2 * anchor.node_scale() * DVector::from_column_slice(&tail);
        assert!(
            node_jet(&model, &s, p, bad.as_slice()).is_none(),
            "tail fixture must be outside the provider's scope"
        );
        // Include a zero-weight tail before AND after the live node to pin alignment.
        let actual = subject_agq_cov_hessian(
            &model,
            &s,
            p,
            &eta,
            &[tail.clone(), central[0].clone(), tail],
            &[0.0, 1.0, 0.0],
        )
        .unwrap();
        assert_eq!(actual.total(), expected.total());
        assert_eq!(actual.grad, expected.grad);
    }

    #[test]
    fn agq_cov_hessian_parallel_reduction_is_bit_identical() {
        use crate::estimation::covariance::analytic_cov_hessian;
        use crate::types::{EstimationMethod, FitOptions, Population};
        let model = parse_model_string(WARFARIN).unwrap();
        let p = &model.default_params;
        let subjects: Vec<_> = (0..5)
            .map(|i| {
                let mut s = warfarin_subject(&model, &p.theta, &[0.5, 2.0, 8.0, 24.0]);
                s.id = i.to_string();
                for y in &mut s.observations {
                    *y *= 1.0 + 0.02 * i as f64;
                }
                s
            })
            .collect();
        let eta: Vec<_> = subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, p)))
            .collect();
        let pop = Population {
            subjects,
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let opts = FitOptions {
            method: EstimationMethod::FoceI,
            n_agq: 3,
            ..FitOptions::default()
        };
        let x = pack_params(p);
        let run = |n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
                .install(|| analytic_cov_hessian(&model, &pop, p, &x, &eta, &[], &opts).unwrap())
        };
        assert_eq!(run(1), run(3));
    }

    #[test]
    fn agq_cov_hessian_score_failure_rejects_s_instead_of_squaring_a_penalty() {
        use crate::estimation::covariance::{
            assemble_score_cross_product, compute_covariance, CovarianceStepResult,
        };
        use crate::estimation::parameterization::compute_bounds;
        use crate::types::{CovarianceMethod, EstimationMethod, FitOptions, Population};
        let model = parse_model_string(WARFARIN).unwrap();
        let p = &model.default_params;
        let s = warfarin_subject(&model, &p.theta, &[0.5, 2.0, 8.0, 24.0]);
        let eta = vec![DVector::from_vec(precise_ebe(&model, &s, p))];
        let ebe = find_ebe(&model, &s, p, 200, 1e-10, None, None, 0);
        let h = vec![ebe.h_matrix];
        let pop = Population {
            subjects: vec![s],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let x = pack_params(p);
        let bounds = compute_bounds(p);
        let free: Vec<_> = (0..x.len()).collect();
        for fraction in [0.0, 1.0] {
            let opts = FitOptions {
                method: EstimationMethod::FoceI,
                n_agq: 3,
                inner_maxiter: 0,
                inner_tol: 1e-12,
                reconverge_gradient_interval: 1,
                max_unconverged_frac: fraction,
                covariance_method: CovarianceMethod::Sandwich,
                verbose: false,
                ..FitOptions::default()
            };
            let mut perturbed = x.clone();
            perturbed[0] += 1e-4 * (1.0 + x[0].abs());
            let perturbed = unpack_params(&perturbed, p);
            let failed = find_ebe(
                &model,
                &pop.subjects[0],
                &perturbed,
                0,
                1e-12,
                Some(eta[0].as_slice()),
                None,
                0,
            );
            assert!(!failed.converged, "fixture must exhaust its inner budget");
            let err = assemble_score_cross_product(
                &x,
                p,
                &model,
                &pop,
                &eta,
                &h,
                &[vec![]],
                &bounds,
                &opts,
                &free,
            )
            .unwrap_err();
            assert!(
                err.contains("quadrature score") && err.contains("subject 1"),
                "{err}"
            );
            match compute_covariance(&x, p, &model, &pop, &eta, &h, &[vec![]], &opts) {
                CovarianceStepResult::Unusable(reason) => {
                    assert!(reason.contains("quadrature score"), "{reason}")
                }
                _ => panic!("an unavailable quadrature score must not produce covariance"),
            }
            let analytic = FitOptions {
                reconverge_gradient_interval: 0,
                ..opts
            };
            assert!(assemble_score_cross_product(
                &x,
                p,
                &model,
                &pop,
                &eta,
                &h,
                &[vec![]],
                &bounds,
                &analytic,
                &free
            )
            .is_ok());
        }
    }

    #[test]
    fn agq_cov_hessian_declines_custom_and_time_varying_magnitudes() {
        use crate::estimation::agq_cov_hessian::{
            node_jet, subject_agq_cov_hessian, subject_packed_agq_cov_hessian,
        };
        let eta = [0.17, -0.11, 0.23];
        let grid = vec![vec![0.0; 3]];
        let pi = vec![1.0];
        for magnitude in [None, Some("2.0"), Some("if (TIME > 4.0) TVKA else 1.0")] {
            let source = magnitude.map_or_else(
                || WARFARIN.to_owned(),
                |m| {
                    WARFARIN.replace(
                        "proportional(PROP_ERR)",
                        &format!("proportional(PROP_ERR * ({m}))"),
                    )
                },
            );
            let model = parse_model_string(&source).unwrap();
            let params = &model.default_params;
            let subject = warfarin_subject(&model, &params.theta, &[0.5, 2.0, 8.0, 24.0]);
            assert_eq!(
                model.ruv_obs_mult(&subject, &params.theta).is_some(),
                magnitude.is_some()
            );
            // Both assembly boundaries and the node helper decline; the otherwise
            // identical plain fixture must remain supported.
            let supported = magnitude.is_none();
            assert_eq!(
                node_jet(&model, &subject, params, &eta).is_some(),
                supported
            );
            assert_eq!(
                subject_agq_cov_hessian(&model, &subject, params, &eta, &grid, &pi).is_some(),
                supported
            );
            assert_eq!(
                subject_packed_agq_cov_hessian(&model, &subject, params, params, &eta, &grid, &pi)
                    .is_some(),
                supported
            );
        }
    }

    /// The split into `C` and `M` reassembles into exactly the M2 natural block, and both halves
    /// behave the way #251's term (C) needs them to.
    ///
    /// Three claims, none of which the FD parity tests above can see because they only ever
    /// observe the fused result:
    ///
    /// 1. `fuse` reproduces `subject_cov_hessian_m2_natural` **bit for bit** — the extraction is
    ///    a refactor, not a reformulation, so #436's covariance is untouched.
    /// 2. `C` is symmetric. It is a fixed-`b` second derivative, so Clairaut applies; AGQ
    ///    contracts `C_j` directly rather than through the fused form, which would mask an
    ///    asymmetry by symmetrising it away.
    /// 3. `C` is *not* already the fused answer — i.e. the mode-coupling correction is a
    ///    materially large part of the result. Without this the first two claims would hold
    ///    vacuously if `M` came back empty or zero.
    #[test]
    fn cov_hessian_parts_fuse_to_the_m2_natural_block() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.2, 10.0, 1.5];
        let subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        let mut params = model.default_params.clone();
        params.theta = theta;

        let eta = precise_ebe(&model, &subject, &params);
        let sens = subject_sensitivities_cov(&model, &subject, &params.theta, &eta).unwrap();
        let prep = prepare(&model, &subject, &params, &sens, &eta).unwrap();

        let fused = subject_cov_hessian_m2_natural(&model, &subject, &params, &sens, &prep, &eta);
        let parts = subject_cov_hessian_parts(&model, &subject, &params, &sens, &prep, &eta);
        let refused = parts.fuse(&prep.h_inner_inv);

        let dim = fused.nrows();
        assert_eq!(parts.c.nrows(), dim, "C must span the natural parameters");
        assert_eq!(parts.m.len(), dim, "one M vector per natural parameter");

        let mut max_correction = 0.0_f64;
        for a in 0..dim {
            for b in 0..dim {
                assert_eq!(
                    refused[(a, b)],
                    fused[(a, b)],
                    "fuse must reproduce the M2 natural block exactly at ({a},{b})"
                );
                assert!(
                    (parts.c[(a, b)] - parts.c[(b, a)]).abs() < 1e-9 * parts.c.amax().max(1.0),
                    "the fixed-b curvature C must be symmetric at ({a},{b})"
                );
                max_correction = max_correction.max((parts.c[(a, b)] - fused[(a, b)]).abs());
            }
        }
        assert!(
            max_correction > 1e-6,
            "the mode-coupling term MᵀH⁻¹M is ~0, so this test would pass vacuously"
        );
    }

    /// `agq_cov_hessian::node_jet` at the **mode** reproduces exactly what #436 computes there.
    ///
    /// The jet's whole premise is that `prepare` / `subject_cov_hessian_parts` / `score_core` are
    /// functions of the evaluation point rather than of the mode specifically — AGQ calls them at
    /// quadrature nodes. Nothing enforces that today, so if someone later hoists a mode-only
    /// assumption into `prepare` (a `z = Ω⁻¹η̂` that reads a cached EBE, say), the node path would
    /// silently start returning mode quantities at every node and the resulting Hessian would be
    /// wrong in a way no FD parity test on the *mode* could see. Pinning the `b = b̂` case against
    /// the direct route makes that failure loud at the one point where both are defined.
    ///
    /// Also pins `‖g(b̂)‖ ≈ 0`. That is the stationarity fact the whole `n_agq = 1` reduction
    /// rests on, and it holds only to `inner_tol` — so this doubles as a measurement of the real
    /// tolerance floor for the eventual reduction test, rather than leaving it to be discovered
    /// as an unexplained FD disagreement.
    #[test]
    fn node_jet_at_the_mode_reproduces_the_focei_parts() {
        use crate::estimation::agq_cov_hessian::node_jet;

        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.2, 10.0, 1.5];
        let subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        let mut params = model.default_params.clone();
        params.theta = theta;
        let eta = precise_ebe(&model, &subject, &params);

        let jet = node_jet(&model, &subject, &params, &eta).expect("warfarin is in scope");

        // Stationarity: the mode's inner gradient vanishes, to the inner tolerance.
        for i in 0..model.n_eta {
            assert!(
                jet.g[i].abs() < 1e-5,
                "‖g(b̂)‖ must vanish by stationarity; component {i} is {}",
                jet.g[i]
            );
        }

        let sens = subject_sensitivities_cov(&model, &subject, &params.theta, &eta).unwrap();
        let prep = prepare(&model, &subject, &params, &sens, &eta).unwrap();
        let want = subject_cov_hessian_parts(&model, &subject, &params, &sens, &prep, &eta);

        let dim = want.c.nrows();
        assert_eq!(jet.parts.c.nrows(), dim);
        for a in 0..dim {
            for b in 0..dim {
                assert_eq!(
                    jet.parts.c[(a, b)],
                    want.c[(a, b)],
                    "node jet's C must match the direct route at ({a},{b})"
                );
            }
            for i in 0..model.n_eta {
                assert_eq!(jet.parts.m[a][i], want.m[a][i], "M mismatch at ({a},{i})");
            }
        }

        // `h` is the exact conditional Hessian, i.e. the inverse of what `prep` carries.
        let h_from_prep = prep.h_inner_inv.clone().try_inverse().expect("H is PD");
        for i in 0..model.n_eta {
            for j in 0..model.n_eta {
                assert!(
                    (jet.h[(i, j)] - h_from_prep[(i, j)]).abs()
                        < 1e-7 * h_from_prep.amax().max(1.0),
                    "node jet's H must be prep's h_inner at ({i},{j}): {} vs {}",
                    jet.h[(i, j)],
                    h_from_prep[(i, j)]
                );
            }
        }
    }

    /// **The reduction, end to end.** `subject_agq_cov_hessian` at one node must reproduce #436.
    ///
    /// At `n_agq = 1` the rule is `z = 0`, so the node is the mode, `π₁ = 1`, and:
    ///
    /// * term (B) is *identically* zero — a single softmax weight has no variance;
    /// * term (C) collapses to `C − MᵀH⁻¹M`, i.e. `subject_cov_hessian_m2_natural`, because
    ///   `β_ζ = b̂_ζ = −H⁻¹M_ζ` and the `g₁ᵀ(b̂_ζξ + √2M_ζξz)` tail is killed by `g₁ = 0`.
    ///
    /// Term (A) has no #436 counterpart to compare against directly (it is the Hessian of
    /// `½log|S|`, where #436's M3 block is the Hessian of `½log|H̃|` — they differ by the jitter),
    /// so the assertion is on (B) and (C), which is exactly why `AgqCovTerms` keeps the three
    /// apart instead of returning only the sum. A test on the total could not distinguish a sign
    /// error in one term from a compensating error in another.
    ///
    /// The `g₁ = 0` tolerance is the binding one here, not the FD step: `node_jet_at_the_mode_…`
    /// measures it directly, and it is why this asserts `1e-6` relative rather than machine
    /// epsilon despite every operation in the chain being exact.
    #[test]
    fn agq_cov_hessian_reduces_to_focei_at_one_node() {
        use crate::estimation::agq_cov_hessian::subject_agq_cov_hessian;

        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.2, 10.0, 1.5];
        let subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        let mut params = model.default_params.clone();
        params.theta = theta;
        let eta = precise_ebe(&model, &subject, &params);

        // The n_agq = 1 grid: one node at z = 0 carrying all the weight.
        let grid = vec![vec![0.0; model.n_eta]];
        let pi = vec![1.0];

        let terms = subject_agq_cov_hessian(&model, &subject, &params, &eta, &grid, &pi)
            .expect("warfarin at one node is in scope");

        let sens = subject_sensitivities_cov(&model, &subject, &params.theta, &eta).unwrap();
        let prep = prepare(&model, &subject, &params, &sens, &eta).unwrap();
        let want = subject_cov_hessian_m2_natural(&model, &subject, &params, &sens, &prep, &eta);

        let dim = want.nrows();
        let scale = want.amax().max(1.0);

        for a in 0..dim {
            for b in 0..dim {
                // (B) vanishes identically at one node.
                assert!(
                    terms.softmax[(a, b)].abs() < 1e-12,
                    "term (B) must be exactly zero at one node; got {} at ({a},{b})",
                    terms.softmax[(a, b)]
                );
                // (C) is #436's M2 natural block.
                assert!(
                    (terms.node[(a, b)] - want[(a, b)]).abs() < 1e-6 * scale,
                    "term (C) at ({a},{b}): AGQ {} vs #436 M2 {}",
                    terms.node[(a, b)],
                    want[(a, b)]
                );
            }
        }

        // And the log-det curvature is a real contribution, not silently zero — otherwise the
        // assembly would "reduce to #436" only because two thirds of it did nothing.
        assert!(
            terms.logdet.amax() > 1e-6,
            "term (A) is ~0, so the reduction above is vacuous"
        );
    }

    /// **The multi-node oracle.** The packed AGQ covariance Hessian against a reconverged
    /// second difference of the AGQ objective itself, at `n_agq = 3`.
    ///
    /// This is the only check that sees the whole assembly at once, and the only one that can see
    /// the things `agq_cov_hessian_reduces_to_focei_at_one_node` structurally cannot:
    ///
    /// * the **`√2` node scaling** — at one node `z = 0`, so `√2` drops out of the placement, the
    ///   displacement `β_{j,ζ}` and the `M_ζξ z_j` tail alike. An inconsistency between them is
    ///   exactly the bug nlmixr2est#785 shipped, and it is invisible below three nodes;
    /// * **term (B)**, `Cov_π(u_ζ, u_ξ)`, which vanishes identically at one node — so the entire
    ///   softmax-response term is unexercised by the reduction test;
    /// * the **`g_jᵀ(b̂_ζξ + √2·M_ζξ z_j)` tail**, killed at the mode by `g₁ = 0`;
    /// * the **natural→packed chain** pairing this Hessian with the AGQ gradient rather than
    ///   FOCEI's.
    ///
    /// Differencing `F_i` rather than a component is the point: it is the function the fit
    /// minimises, so agreement here means the covariance describes the likelihood that was
    /// actually optimised — the property the whole module exists to establish.
    ///
    /// # Why this tolerance
    ///
    /// The oracle is the noisy side. `F_i` is smooth and evaluated to ~`1e-15`, but a second
    /// difference divides by `h²`, so at `h = 1e-4` the floor is ~`1e-7` absolute against
    /// truncation of the same order — balanced by construction. `1e-4` **relative** therefore
    /// leaves three orders of headroom over the reference's own noise, and is not a statement
    /// about the analytic side's accuracy.
    ///
    /// The reconverged EBE does *not* dominate: `∂F/∂η̂` is the posterior-mean score, which is
    /// near zero, so an EBE error enters quadratically. That is the same envelope property the
    /// `n_agq = 1` reduction leans on, working in the oracle's favour here.
    #[test]
    fn agq_cov_hessian_matches_fd_of_the_agq_objective_at_three_nodes() {
        check_agq_cov_hessian_objective(WARFARIN, 3, false);
    }

    #[test]
    fn agq_cov_hessian_matches_m3_objective() {
        check_agq_cov_hessian_objective(WARFARIN, 3, true);
    }

    #[test]
    fn agq_cov_hessian_matches_fd_with_block_omega() {
        let model = WARFARIN.replace(
            "omega ETA_CL ~ 0.09\n  omega ETA_V  ~ 0.04",
            "block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]",
        );
        assert_ne!(model, WARFARIN);
        check_agq_cov_hessian_objective(&model, 3, false);
    }

    #[test]
    fn agq_cov_hessian_matches_fd_with_combined_error_at_five_nodes() {
        let model = WARFARIN
            .replace(
                "sigma PROP_ERR ~ 0.04",
                "sigma PROP_ERR ~ 0.04\n  sigma ADD_ERR ~ 0.1",
            )
            .replace("proportional(PROP_ERR)", "combined(PROP_ERR, ADD_ERR)");
        check_agq_cov_hessian_objective(&model, 5, false);
    }

    fn check_agq_cov_hessian_objective(model_text: &str, n_agq: usize, m3: bool) {
        use crate::estimation::agq::{
            agq_subject_objective, gauss_hermite, subject_grid_and_weights,
        };
        use crate::estimation::agq_cov_hessian::subject_packed_agq_cov_hessian;
        use crate::estimation::parameterization::{pack_params, unpack_params};

        let mut model = parse_model_string(model_text).expect("parse");
        if m3 {
            model.bloq_method = BloqMethod::M3;
        }
        let theta = vec![0.2, 10.0, 1.5];
        let mut subject = warfarin_subject(&model, &theta, &[0.5, 2.0, 8.0, 24.0]);
        if m3 {
            subject.cens[2] = 1;
            subject.cens[3] = 1;
        }
        let mut params = model.default_params.clone();
        params.theta = theta;
        let template = params.clone();
        let x = pack_params(&params);
        let n = x.len();

        // Analytic, on the grid the objective evaluates.
        let eta = precise_ebe(&model, &subject, &params);
        let (nodes, weights) = gauss_hermite(n_agq);
        let (grid, pi) =
            subject_grid_and_weights(&model, &subject, &params, &eta, &nodes, &weights)
                .expect("warfarin is in the Gauss-Newton anchor's scope");
        assert_eq!(
            grid.len(),
            n_agq.pow(model.n_eta as u32),
            "premise: the tensor grid really has more than one node"
        );
        let analytic =
            subject_packed_agq_cov_hessian(&model, &subject, &template, &params, &eta, &grid, &pi)
                .expect("analytic AGQ covariance is in scope");

        // Oracle: F_i(x) with the mode reconverged at every perturbed point.
        let f = |xv: &[f64]| -> f64 {
            let q = unpack_params(xv, &template);
            let e = precise_ebe(&model, &subject, &q);
            agq_subject_objective(&model, &subject, &q, &e, n_agq)
        };
        let f0 = f(&x);
        let step: Vec<f64> = x.iter().map(|v| 1e-4 * (1.0 + v.abs())).collect();
        let bump = |i: usize, si: f64, j: usize, sj: f64| -> f64 {
            let mut q = x.clone();
            q[i] += si * step[i];
            q[j] += sj * step[j];
            f(&q)
        };

        let scale = analytic.amax().max(1.0);
        for i in 0..n {
            for j in i..n {
                let fd = if i == j {
                    (bump(i, 1.0, i, 0.0) - 2.0 * f0 + bump(i, -1.0, i, 0.0)) / (step[i] * step[i])
                } else {
                    (bump(i, 1.0, j, 1.0) - bump(i, 1.0, j, -1.0) - bump(i, -1.0, j, 1.0)
                        + bump(i, -1.0, j, -1.0))
                        / (4.0 * step[i] * step[j])
                };
                assert!(
                    (analytic[(i, j)] - fd).abs() < 1e-4 * scale,
                    "∂²F/∂x{i}∂x{j}: analytic {} vs FD-of-objective {fd}",
                    analytic[(i, j)]
                );
            }
        }
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

    /// Warfarin (1-cpt oral, diagonal Ω, proportional error): the full `M2 + M3`
    /// natural covariance Hessian (Φ **and** `½log|H̃|` curvature carried through
    /// the moving mode) matches the reconverged precise-EBE FD of the full
    /// analytic gradient across every `[θ, Ω, σ]` block.
    #[test]
    fn cov_hessian_full_natural_matches_reconverged_fd_diagonal() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.2, 10.0, 1.5];
        let subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        let mut params = model.default_params.clone();
        params.theta = theta;
        check_full_natural(&model, &subject, &params);
    }

    #[test]
    fn cov_hessian_m3_matches_reconverged_gradient() {
        let mut model = parse_model_string(WARFARIN).expect("parse");
        model.bloq_method = BloqMethod::M3;
        let theta = vec![0.2, 10.0, 1.5];
        let mut subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        subject.cens[5] = 1;
        subject.cens[6] = -1;
        let mut params = model.default_params.clone();
        params.theta = theta;
        check_full_natural(&model, &subject, &params);
    }

    /// Block-Ω (correlated CL/V): the full `M2 + M3` Hessian, exercising the
    /// off-diagonal ΩΩ third-order curvature and the η-coupled mode response.
    #[test]
    fn cov_hessian_full_natural_matches_reconverged_fd_block_omega() {
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
        check_full_natural(&model, &subject, &params);
    }

    /// Warfarin (1-cpt, diagonal Ω): the analytic **packed** covariance Hessian
    /// (natural `M2 + M3` chained through the log-θ / Cholesky-Ω / log-σ
    /// reparameterization) matches the reconverged-FD of the analytic packed
    /// gradient — the exact replacement for `compute_covariance`'s FD stencil.
    #[test]
    fn cov_hessian_packed_matches_reconverged_fd_diagonal() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.2, 10.0, 1.5];
        let subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        let mut params = model.default_params.clone();
        params.theta = theta;
        check_packed(&model, &subject, &params);
    }

    /// Block-Ω (correlated CL/V): the packed chain through the Cholesky factor's
    /// off-diagonal entries, the part the diagonal case does not exercise.
    #[test]
    fn cov_hessian_packed_matches_reconverged_fd_block_omega() {
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
        check_packed(&model, &subject, &params);
    }

    /// Speed & accuracy report for the analytic covariance Hessian (#436): the
    /// exact `M2 + M3` natural Hessian vs the finite-difference path it replaces,
    /// on warfarin (1-cpt, dim = 7). Prints with `--nocapture`. The analytic side
    /// is one forward pass per subject; the FD side reconverges the EBE and
    /// evaluates the analytic natural gradient at `2·dim` perturbed points (the
    /// cheapest FD variant in production — the OFV-Hessian path is costlier).
    #[test]
    fn cov_hessian_speed_and_accuracy_report() {
        use std::hint::black_box;
        use std::time::Instant;
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.2, 10.0, 1.5];
        let subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        let mut params = model.default_params.clone();
        params.theta = theta;
        let eta = precise_ebe(&model, &subject, &params);
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

        // Accuracy: analytic vs reconverged precise-EBE FD of the full gradient.
        let sens = subject_sensitivities_cov(&model, &subject, &params.theta, &eta).unwrap();
        let prep = prepare(&model, &subject, &params, &sens, &eta).unwrap();
        let analytic = subject_cov_hessian_natural(&model, &subject, &params, &sens, &prep, &eta);
        let mut fd = DMatrix::zeros(dim, dim);
        for col in 0..dim {
            let h = 1e-6 * (1.0 + base_val(col).abs());
            let gp = full_natural_grad(
                &model,
                &subject,
                &perturb_natural(&params, n_theta, &entries, col, h),
            );
            let gm = full_natural_grad(
                &model,
                &subject,
                &perturb_natural(&params, n_theta, &entries, col, -h),
            );
            for row in 0..dim {
                fd[(row, col)] = (gp[row] - gm[row]) / (2.0 * h);
            }
        }
        let mut max_abs = 0.0_f64;
        for r in 0..dim {
            for c in 0..dim {
                max_abs = max_abs.max((analytic[(r, c)] - fd[(r, c)]).abs());
            }
        }

        // Speed: analytic (Dual3 sens + prepare + assembly) vs FD-of-gradient
        // (2·dim reconverged-EBE analytic natural gradients).
        let n_iter = 30;
        let t0 = Instant::now();
        for _ in 0..n_iter {
            let s = subject_sensitivities_cov(&model, &subject, &params.theta, &eta).unwrap();
            let p = prepare(&model, &subject, &params, &s, &eta).unwrap();
            black_box(subject_cov_hessian_natural(
                &model, &subject, &params, &s, &p, &eta,
            ));
        }
        let ta = t0.elapsed().as_secs_f64() / n_iter as f64;
        let t1 = Instant::now();
        for _ in 0..n_iter {
            for col in 0..dim {
                let h = 1e-6 * (1.0 + base_val(col).abs());
                for &sgn in &[1.0_f64, -1.0] {
                    let pp = perturb_natural(&params, n_theta, &entries, col, sgn * h);
                    let e = find_ebe(
                        &model,
                        &subject,
                        &pp,
                        30,
                        1e-8,
                        Some(eta.as_slice()),
                        None,
                        0,
                    )
                    .eta;
                    let es: Vec<f64> = e.iter().copied().collect();
                    black_box(subject_theta_gradient(&model, &subject, &pp, &es));
                    black_box(subject_omega_gradient(&model, &subject, &pp, &es));
                    black_box(subject_sigma_gradient(&model, &subject, &pp, &es));
                }
            }
        }
        let tf = t1.elapsed().as_secs_f64() / n_iter as f64;

        eprintln!(
            "\n=== #436 covariance Hessian — speed & accuracy (warfarin 1-cpt, dim={dim}, debug) ===\n\
             accuracy : max |analytic - reconverged-FD| = {:.2e}  (analytic noise-free; no fd_hessian_step)\n\
             speed/subj: analytic {:.3} ms   FD-of-gradient {:.3} ms   -> {:.1}x faster\n",
            max_abs,
            ta * 1e3,
            tf * 1e3,
            tf / ta
        );
        assert!(max_abs < 2e-3, "analytic vs FD max Δ {max_abs:.2e}");
        assert!(ta < tf, "analytic ({ta:.4}s) should beat FD ({tf:.4}s)");
    }

    /// End-to-end through `compute_covariance`: the analytic route must reproduce the
    /// finite-difference stencil it replaces — **including the OFV scale factor**.
    ///
    /// Every other test in this module validates the per-subject assembly against a finite
    /// difference of its own gradient, which is scale-blind: `subject_packed_cov_hessian`
    /// returns `∂²Fᵢ/∂x²`, but `compute_covariance` consumes `∂²OFV/∂x²` with `OFV = 2·Σᵢ Fᵢ`
    /// (see `population_gradient_sens`'s `grad[k] += 2.0 * gi[k]`), and its
    /// `covariance = 2·H⁻¹` assumes that convention. Summing per-subject Hessians unscaled
    /// therefore yields half the right matrix and inflates every standard error by √2 — with
    /// **no other symptom**, since the result stays symmetric, positive-definite and
    /// plausibly sized. Exactly the factor-of-2 class of #209.
    ///
    /// Nothing else covers the population helper or the `compute_covariance` dispatch at all,
    /// so this is the only test that exercises the wiring rather than the mathematics.
    #[test]
    fn analytic_cov_matches_the_fd_stencil_through_compute_covariance() {
        use crate::estimation::covariance::{
            analytic_cov_hessian, compute_covariance, CovarianceStepResult,
        };
        use crate::types::{EstimationMethod, FitOptions, Population};

        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.2, 10.0, 1.5];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();

        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0];
        let subjects: Vec<Subject> = (0..4)
            .map(|k| {
                let mut s = warfarin_subject(&model, &theta, &times);
                s.id = format!("{k}");
                s
            })
            .collect();
        let eta_hats: Vec<DVector<f64>> = subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
            .collect();
        let n_subj = subjects.len();
        let population = Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let x_hat = pack_params(&params);
        let h_mats = vec![DMatrix::zeros(model.n_eta, model.n_eta); n_subj];
        let kappas = vec![vec![]; n_subj];

        // Exact-anchor Laplace is a different objective even at one node.
        for n_agq in [1, 3] {
            let opts = FitOptions {
                method: EstimationMethod::Laplace,
                n_agq,
                ..FitOptions::default()
            };
            assert!(analytic_cov_hessian(
                &model,
                &population,
                &params,
                &x_hat,
                &eta_hats,
                &[],
                &opts
            )
            .is_none());
        }
        // M3 subjects remain on the analytic AGQ route.
        let mut censored_model = parse_model_string(WARFARIN).expect("parse");
        censored_model.bloq_method = BloqMethod::M3;
        let mut censored_population = population.clone();
        censored_population.subjects[1].cens[3] = 1;
        let agq_opts = FitOptions {
            method: EstimationMethod::FoceI,
            n_agq: 3,
            ..FitOptions::default()
        };
        assert!(analytic_cov_hessian(
            &censored_model,
            &censored_population,
            &params,
            &x_hat,
            &eta_hats,
            &[],
            &agq_opts
        )
        .is_some());

        let run = |analytic: bool, interaction: bool, n_agq: usize| -> DMatrix<f64> {
            let mut opts = FitOptions {
                analytic_cov_hessian: analytic,
                interaction,
                method: if interaction {
                    EstimationMethod::FoceI
                } else {
                    EstimationMethod::Foce
                },
                n_agq,
                ..FitOptions::default()
            };
            opts.verbose = false;
            if analytic {
                assert!(analytic_cov_hessian(&model, &population, &params, &x_hat, &eta_hats, &[], &opts).is_some(),
                    "production dispatch must select analytic covariance: interaction={interaction}, n_agq={n_agq}");
            }
            match compute_covariance(
                &x_hat,
                &params,
                &model,
                &population,
                &eta_hats,
                &h_mats,
                &kappas,
                &opts,
            ) {
                CovarianceStepResult::Success(o) => o.matrix,
                other => panic!(
                    "covariance step must succeed (analytic = {analytic}, \
                     interaction = {interaction}); got {}",
                    match other {
                        CovarianceStepResult::Unusable(m) => m,
                        CovarianceStepResult::FailedNonPd { reason, .. } => reason,
                        _ => unreachable!(),
                    }
                ),
            }
        };

        // Premise: the analytic route must actually be *taken*, on BOTH entry points. If the
        // scope gate declined, `compute_covariance` would fall back to the same FD stencil and
        // this test would compare FD against FD — passing while proving nothing. Asserted at
        // the source, per estimator, because the two assemblies gate independently (the FOCE
        // one additionally needs an in-scope third-order sweep at η = 0).
        for (s, e) in population.subjects.iter().zip(eta_hats.iter()) {
            assert!(
                subject_packed_cov_hessian(&model, s, &params, &x_hat, e.as_slice()).is_some(),
                "fixture must be in FOCEI analytic scope, else this compares FD against itself"
            );
            assert!(
                subject_packed_cov_hessian_foce(&model, s, &params, &x_hat, e.as_slice()).is_some(),
                "fixture must be in FOCE analytic scope, else this compares FD against itself"
            );
        }

        // Standard errors are what a user sees, so compare those rather than raw entries.
        let se = |c: &DMatrix<f64>| -> Vec<f64> {
            (0..c.nrows()).map(|i| c[(i, i)].max(0.0).sqrt()).collect()
        };

        // Both estimators, because the analytic route dispatches on `options.interaction` into
        // two *different* marginals — FOCEI's Almquist–Laplace `Φ + ½log|H̃|` and FOCE's
        // Sheiner–Beal `(y−f₀)ᵀR̃⁻¹(y−f₀) + log|R̃|`. Running only the default (`interaction =
        // true`) left the FOCE arm of that dispatch, and the OFV `×2` scaling applied to the
        // FOCE assembly, unexercised end-to-end — which is precisely the untested-wiring shape
        // that produced the √2 SE inflation on the FOCEI side.
        for (interaction, n_agq, label) in [
            (false, 1, "FOCE"),
            (true, 1, "FOCEI"),
            (true, 3, "AGQ-FOCEI"),
        ] {
            let (se_fd, se_an) = (
                se(&run(false, interaction, n_agq)),
                se(&run(true, interaction, n_agq)),
            );
            let worst = se_fd
                .iter()
                .zip(se_an.iter())
                .filter(|(f, _)| **f > 1e-12)
                .fold(0.0f64, |m, (f, a)| m.max(((a - f) / f).abs()));

            // A √2 (41%) discrepancy is the specific failure this guards. Checked before the
            // tighter bound so a scale error reports as a scale error rather than as generic
            // disagreement.
            assert!(
                worst < 0.2,
                "{label}: SE ratio looks like an OFV scale-factor error (√2 ≈ 0.41): \
                 {worst:.3e}\nfd = {se_fd:?}\nan = {se_an:?}"
            );
            assert!(
                worst < 0.05,
                "{label}: analytic SEs must match the FD stencil to 5%: worst relative Δ \
                 {worst:.3e}\nfd = {se_fd:?}\nan = {se_an:?}"
            );
        }
    }

    #[test]
    fn audit_quadrature_score_matrix_uses_its_own_objective() {
        use crate::estimation::agq::agq_population_nll;
        use crate::estimation::covariance::assemble_score_cross_product;
        use crate::estimation::parameterization::compute_bounds;
        use crate::types::{EstimationMethod, FitOptions, Population};
        let model = parse_model_string(WARFARIN).unwrap();
        let params = &model.default_params;
        let subject = warfarin_subject(&model, &params.theta, &[0.5, 2.0, 8.0, 24.0]);
        let pop = Population {
            subjects: vec![subject],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let ebe = find_ebe(&model, &pop.subjects[0], params, 200, 1e-11, None, None, 0);
        let eta = precise_ebe(&model, &pop.subjects[0], params);
        let x = pack_params(params);
        let bounds = compute_bounds(params);
        let free: Vec<usize> = (0..x.len()).collect();
        for (method, n_agq) in [
            (EstimationMethod::FoceI, 3),
            (EstimationMethod::Laplace, 1),
            (EstimationMethod::Laplace, 3),
        ] {
            let opts = FitOptions {
                method,
                n_agq,
                interaction: true,
                inner_tol: 1e-11,
                inner_maxiter: 200,
                verbose: false,
                ..FitOptions::default()
            };
            let f = |xv: &[f64]| {
                let p = unpack_params(xv, params);
                let e = precise_ebe(&model, &pop.subjects[0], &p);
                // S is the cross-product of NLL scores, not OFV scores.
                agq_population_nll(
                    &model,
                    &pop,
                    &p,
                    &[DVector::from_vec(e)],
                    &[],
                    n_agq,
                    opts.hessian_anchor(),
                )
            };
            let mut g = DVector::zeros(x.len());
            for k in 0..x.len() {
                let h = 1e-4 * (1.0 + x[k].abs());
                let mut xp = x.clone();
                let mut xm = x.clone();
                xp[k] += h;
                xm[k] -= h;
                g[k] = (f(&xp) - f(&xm)) / (2.0 * h);
            }
            let expected = &g * g.transpose();
            // This fixture must distinguish quadrature from the old FOCEI score path,
            // independently of the OFV/NLL factor-of-two convention.
            let (_, old) = crate::estimation::gauss_newton::subject_nll_pop_grad(
                &x,
                params,
                &model,
                &pop,
                0,
                &DVector::from_vec(eta.clone()),
                &ebe.h_matrix,
                &[],
                &bounds,
                &opts,
            );
            let old = DVector::from_vec(old);
            let old_gap = (&old * old.transpose() - &expected).amax() / expected.amax().max(1.0);
            assert!(
                old_gap > 1e-2,
                "fixture must distinguish the old FOCEI NLL scores: {old_gap}"
            );
            for interval in [0, 1] {
                let opts = FitOptions {
                    reconverge_gradient_interval: interval,
                    ..opts.clone()
                };
                let actual = assemble_score_cross_product(
                    &x,
                    params,
                    &model,
                    &pop,
                    &[DVector::from_vec(eta.clone())],
                    &[ebe.h_matrix.clone()],
                    &[vec![]],
                    &bounds,
                    &opts,
                    &free,
                )
                .expect("finite converged quadrature scores");
                let gap = (&actual - &expected).amax() / expected.amax().max(1.0);
                assert!(gap < 1e-3, "{method:?} n={n_agq} interval={interval}: score matrix uses a different objective, relative gap {gap}");
                if interval == 1 {
                    let cov_tolerance = FitOptions {
                        inner_tol: 1e-3,
                        cov_inner_tol: Some(opts.inner_tol),
                        ..opts.clone()
                    };
                    let with_override = assemble_score_cross_product(
                        &x,
                        params,
                        &model,
                        &pop,
                        &[DVector::from_vec(eta.clone())],
                        &[ebe.h_matrix.clone()],
                        &[vec![]],
                        &bounds,
                        &cov_tolerance,
                        &free,
                    )
                    .unwrap();
                    assert_eq!(
                        actual, with_override,
                        "quadrature scores must honor cov_inner_tol"
                    );
                }
            }
        }
    }

    /// Safety gate: the per-subject analytic covariance Hessian (both FOCEI and
    /// FOCE entry points) must return `None` for out-of-derivation-scope models, so
    /// `compute_covariance` drops the whole population back to the finite-difference
    /// covariance. A `None` from any subject is what makes the fallback total. Here
    /// LTBS (`log_transform`) is exercised as an exclusion; M3/BLOQ is checked as
    /// an admitted path with separate FOCE and FOCEI assemblies.
    #[test]
    fn analytic_cov_hessian_gates_out_of_scope() {
        let mut model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.2, 10.0, 1.5];
        let subject = warfarin_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]);
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let eta = precise_ebe(&model, &subject, &params);
        let x = pack_params(&params);

        // In scope (plain analytical Gaussian): both methods produce a Hessian.
        assert!(subject_packed_cov_hessian(&model, &subject, &params, &x, &eta).is_some());
        assert!(subject_packed_cov_hessian_foce(&model, &subject, &params, &x, &eta).is_some());

        // LTBS (log-transform-both-sides) is out of scope → both decline.
        model.log_transform = true;
        assert!(subject_packed_cov_hessian(&model, &subject, &params, &x, &eta).is_none());
        assert!(subject_packed_cov_hessian_foce(&model, &subject, &params, &x, &eta).is_none());
        model.log_transform = false;

        // M3/BLOQ uses the conditional tail under FOCEI and the distinct
        // linearized-marginal tail under FOCE.
        model.bloq_method = BloqMethod::M3;
        let mut subj_cens = subject.clone();
        subj_cens.cens[3] = 1;
        let cens_eta = precise_ebe(&model, &subj_cens, &params);
        assert!(subject_packed_cov_hessian(&model, &subj_cens, &params, &x, &cens_eta).is_some());
        assert!(
            subject_packed_cov_hessian_foce(&model, &subj_cens, &params, &x, &cens_eta).is_some()
        );
        let mut all_cens = subj_cens.clone();
        all_cens.cens.fill(1);
        assert!(
            subject_packed_cov_hessian_foce(&model, &all_cens, &params, &x, &cens_eta).is_none()
        );
        model.bloq_method = BloqMethod::Drop;

        // ── PR #953 review: clauses added after the gate was found narrower than the
        // objective it replaces. Each is a *silent* wrong-number failure if it declines to
        // decline — the assembly returns a plausible, finite, positive-definite Hessian —
        // so each gets its own assertion rather than trusting the list is complete.

        // `gradient = fd` is the user's opt-out from analytic sensitivities (finding 9).
        model.gradient_method = GradientMethod::Fd;
        assert!(subject_packed_cov_hessian(&model, &subject, &params, &x, &eta).is_none());
        assert!(subject_packed_cov_hessian_foce(&model, &subject, &params, &x, &eta).is_none());
        model.gradient_method = GradientMethod::Auto;

        // FREM: covariate pseudo-observation rows need `EPSCOV²` via `build_frem_r_override`,
        // which this assembly does not consult — it would score them with the PK error model
        // and report wrong SEs for exactly the covariate ω block (finding 2, the covariance
        // twin of PR #844).
        model.frem_config = Some(crate::types::FremConfig {
            fremtype_to_indices: HashMap::new(),
            covariate_sigma_index: 0,
        });
        assert!(subject_packed_cov_hessian(&model, &subject, &params, &x, &eta).is_none());
        assert!(subject_packed_cov_hessian_foce(&model, &subject, &params, &x, &eta).is_none());
        model.frem_config = None;

        // A `Selected` spec keys endpoints by covariate branch, not by the CMT column the
        // assembly reads — every row would be scored against branch 1's sigma (finding 3).
        let saved_spec = std::mem::replace(
            &mut model.error_spec,
            crate::types::ErrorSpec::Selected {
                selector: crate::types::ErrorSelector {
                    eval: Box::new(|_| 0),
                    branch_labels: vec!["else".to_string()],
                },
                endpoints: HashMap::new(),
            },
        );
        assert!(subject_packed_cov_hessian(&model, &subject, &params, &x, &eta).is_none());
        assert!(subject_packed_cov_hessian_foce(&model, &subject, &params, &x, &eta).is_none());
        model.error_spec = saved_spec;

        // Back in scope once every flip is undone — otherwise the assertions above could all
        // be passing for some unrelated reason introduced between them.
        assert!(subject_packed_cov_hessian(&model, &subject, &params, &x, &eta).is_some());
        assert!(subject_packed_cov_hessian_foce(&model, &subject, &params, &x, &eta).is_some());

        // Not covered here: `has_non_gaussian()` (finding 4) is unconditionally `false`
        // without the `survival` feature, so the clause cannot be exercised from a default
        // build — it mirrors `analytic_outer_gradient_available`'s own clause, which has the
        // same limitation. The flip-flop reroute (finding 5) is likewise parameter-dependent
        // and lives in `subject_sensitivities_cov`, not in a model-level field.
    }
}
