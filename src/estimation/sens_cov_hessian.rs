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
/// combined error models (`d2var_df2` is `f`-independent — a constant `2σ²` for
/// the proportional part, `0` for additive). `α`, `α'`, `β`, `p` reproduce
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
    let ed = |sig: &[f64], j: usize| -> ErrD2 {
        let cmt = subject.obs_cmts[j];
        let f = sens.obs[j].f;
        let r = model.error_spec.variance_at(cmt, f, sig);
        let d = model.error_spec.dvar_df(cmt, f, sig);
        let d2 = model.error_spec.d2var_df2(cmt, sig);
        let eps = subject.observations[j] - f;
        err_d2(r, d, d2, eps)
    };

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
    let entries = omega_entries(params.omega.diagonal, ne);
    let n_omega = entries.len();
    let n_sigma = params.sigma.values.len();
    let nw = nt + n_omega;
    let dim = nt + n_omega + n_sigma;
    let omega_inv = &prep.omega_inv;
    let h_inner_inv = &prep.h_inner_inv;
    let z = omega_inv * DVector::from_column_slice(eta_hat);
    let e_mats: Vec<DMatrix<f64>> = entries.iter().map(|&(r, c)| e_matrix(r, c, ne)).collect();

    let a: Vec<DVector<f64>> = sens
        .obs
        .iter()
        .map(|o| DVector::from_column_slice(&o.df_deta))
        .collect();
    let ed: Vec<ErrD2> = (0..n_obs)
        .map(|j| {
            let cmt = subject.obs_cmts[j];
            let f = sens.obs[j].f;
            let r = model.error_spec.variance_at(cmt, f, &params.sigma.values);
            let d = model.error_spec.dvar_df(cmt, f, &params.sigma.values);
            let d2 = model.error_spec.d2var_df2(cmt, &params.sigma.values);
            err_d2(r, d, d2, subject.observations[j] - f)
        })
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

/// Censored (M3-BLOQ) rows are out of scope here (their inner term is `−logΦ`,
/// not the Gaussian α/p) and gated out by the covariance step.
pub(crate) fn subject_cov_hessian_m3_natural(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    prep: &Prep,
    eta_hat: &[f64],
) -> DMatrix<f64> {
    let ne = prep.n_eta;
    let n_obs = prep.n_obs;
    let nt = params.theta.len();
    let entries = omega_entries(params.omega.diagonal, ne);
    let n_omega = entries.len();
    let n_sigma = params.sigma.values.len();
    let nw = nt + n_omega;
    let dim = nt + n_omega + n_sigma;
    let nd = dim + ne; // directions: natural params then η

    let omega_inv = &prep.omega_inv;
    let htilde_inv = &prep.htilde_inv;
    let e_mats: Vec<DMatrix<f64>> = entries.iter().map(|&(r, c)| e_matrix(r, c, ne)).collect();

    // Per-observation primitives.
    let a: Vec<DVector<f64>> = sens
        .obs
        .iter()
        .map(|o| DVector::from_column_slice(&o.df_deta))
        .collect();
    let ed: Vec<ErrD2> = (0..n_obs)
        .map(|j| {
            let cmt = subject.obs_cmts[j];
            let f = sens.obs[j].f;
            let r = model.error_spec.variance_at(cmt, f, &params.sigma.values);
            let d = model.error_spec.dvar_df(cmt, f, &params.sigma.values);
            let d2 = model.error_spec.d2var_df2(cmt, &params.sigma.values);
            let eps = subject.observations[j] - f;
            err_d2(r, d, d2, eps)
        })
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

    // C'_{st} = ½[tr(H̃⁻¹ ∂²H̃/∂s∂t) − tr(K_s K_t)] for every direction pair.
    let mut cpp = DMatrix::zeros(nd, nd);
    for s in 0..nd {
        for t in s..nd {
            let tr2 = (htilde_inv * second(s, t)).trace();
            let trk = (&kmat[s] * &kmat[t]).trace();
            let v = 0.5 * (tr2 - trk);
            cpp[(s, t)] = v;
            cpp[(t, s)] = v;
        }
    }

    // Inner-mode responses (shared with FOCE): η̂_{·,ζ} and η̂_{ξζ}, both functions
    // of the inner objective `lᵢ` only.
    let (eta_d, eta_dd) = inner_eta_responses(model, subject, params, sens, prep, eta_hat);

    let half_geta = DVector::from_iterator(ne, prep.g_eta.iter().map(|g| 0.5 * g));
    let mut m3 = DMatrix::zeros(dim, dim);
    for xi in 0..dim {
        for ze in xi..dim {
            // A
            let mut val = cpp[(xi, ze)];
            // B
            for l in 0..ne {
                val += cpp[(xi, dim + l)] * eta_d[ze][l] + cpp[(ze, dim + l)] * eta_d[xi][l];
            }
            // C
            for l in 0..ne {
                for m in 0..ne {
                    val += cpp[(dim + l, dim + m)] * eta_d[xi][l] * eta_d[ze][m];
                }
            }
            // D: ½ g_eta · η̂_{ξζ}.
            val += half_geta.dot(&eta_dd[xi][ze]);

            m3[(xi, ze)] = val;
            m3[(ze, xi)] = val;
        }
    }
    m3
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
#[allow(clippy::type_complexity, dead_code)]
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
    let entries = omega_entries(params.omega.diagonal, ne);
    let n_omega = entries.len();
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
        d20[i] = model.error_spec.d2var_df2(cmts[i], sigma);
    }
    let mut rtilde = &jmat * omega * jmat.transpose();
    for i in 0..nq {
        rtilde[(i, i)] += r0[i];
    }
    let rtilde_inv = rtilde.cholesky()?.inverse();
    let u = &rtilde_inv * &rho;
    let s0 = foce0_sigma(model, &cmts, &f0, sigma);
    let e_mats: Vec<DMatrix<f64>> = entries.iter().map(|&(r, c)| e_matrix(r, c, ne)).collect();

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
/// scope / on BLOQ censoring. `prep` carries the shared inner Hessian.
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
    let ne = model.n_eta;
    let nt = params.theta.len();
    let nq = subject.observations.len();
    if model.bloq_method == crate::types::BloqMethod::M3 && subject.cens.iter().any(|&c| c != 0) {
        return None;
    }
    if sens.obs.len() != nq || sens0.obs.len() != nq {
        return None;
    }
    let entries = omega_entries(params.omega.diagonal, ne);
    let n_omega = entries.len();
    let sigma = &params.sigma.values;
    let n_sigma = sigma.len();
    let dim = nt + n_omega + n_sigma;
    let nw = nt + n_omega;
    let nd = dim + ne;
    let omega = &params.omega.matrix;
    let cmts: Vec<usize> = (0..nq).map(|i| subject.obs_cmts[i]).collect();
    let eta_vec = DVector::from_column_slice(eta_hat);

    // J, ρ, R̃, u, R⁰ and f-derivatives — as in `foce_sb_fixed_natural`.
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
        d20[i] = model.error_spec.d2var_df2(cmts[i], sigma);
    }
    let mut rtilde = &jmat * omega * jmat.transpose();
    for i in 0..nq {
        rtilde[(i, i)] += r0[i];
    }
    let rtilde_inv = rtilde.cholesky()?.inverse();
    let u = &rtilde_inv * &rho;
    let s0 = foce0_sigma(model, &cmts, &f0, sigma);
    let e_mats: Vec<DMatrix<f64>> = entries.iter().map(|&(r, c)| e_matrix(r, c, ne)).collect();

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

    let rho_st = |aa: usize, bb: usize| -> DVector<f64> {
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
    let f_st = |s: usize, t: usize| -> f64 {
        let rst = rho_st(s, t);
        let vst = v_st(s, t);
        let u_t = &rtilde_inv * (&rho_s[t] - &vu[t]);
        rst.dot(&u) + rho_s[s].dot(&u_t) + 0.5 * (&rtilde_inv * &vst).trace()
            - 0.5 * (&k_s[t] * &k_s[s]).trace()
            - u_t.dot(&vu[s])
            - 0.5 * u.dot(&(&vst * &u))
    };

    // Fixed-η̂ gradient and coupling c_l = F_{η_l}.
    let fixed_grad: Vec<f64> = (0..dim)
        .map(|s| rho_s[s].dot(&u) + 0.5 * k_s[s].trace() - 0.5 * u.dot(&vu[s]))
        .collect();
    let c: Vec<f64> = (0..ne)
        .map(|l| {
            let s = dim + l;
            rho_s[s].dot(&u) + 0.5 * k_s[s].trace() - 0.5 * u.dot(&vu[s])
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
/// * `subject_sensitivities_cov` declines (ODE, scaling, LTBS, IOV, time-varying
///   covariates, oral-infusion, resets, non-fixed doses, non-analytical PK);
/// * the subject carries M3/BLOQ censored rows (their inner term is `−logΦ`, not
///   the Gaussian α/p the M3 assembly assumes).
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
    let sens = subject_sensitivities_cov(model, subject, &params.theta, eta_hat)?;
    let prep = prepare(model, subject, &params, &sens)?;
    // Censored (M3/BLOQ) rows are out of the M3 assembly's scope.
    if !prep.censored.is_empty() {
        return None;
    }
    let h_nat = subject_cov_hessian_natural(model, subject, &params, &sens, &prep, eta_hat);
    let g_nat = subject_natural_gradient(&prep, &sens, model, subject, &params, eta_hat);
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
/// Non-IOV `[θ, Ω, σ]` block only (IOV is gated out of the analytic covariance
/// path). `g_nat`/`h_nat` use the symmetric single-parameter Ω convention (an
/// off-diagonal entry sets both `Ω_{rc}` and `Ω_{cr}`), matching
/// [`subject_cov_hessian_natural`] and [`super::sens_outer_gradient::subject_omega_gradient`].
pub(crate) fn pack_natural_hessian(
    h_nat: &DMatrix<f64>,
    g_nat: &[f64],
    x: &[f64],
    template: &ModelParameters,
) -> DMatrix<f64> {
    let n_eta = template.omega.dim();
    let nt = template.theta.len();
    let entries = omega_entries(template.omega.diagonal, n_eta);
    let n_omega = entries.len();
    let n_sigma = template.sigma.values.len();
    let nw = nt + n_omega;
    let dim = nt + n_omega + n_sigma;
    let params = unpack_params(x, template);
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
/// outside analytic scope or on BLOQ censoring (caller falls back to FD). The
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
    let sens = subject_sensitivities_cov(model, subject, &params.theta, eta_hat)?;
    let zeros = vec![0.0; model.n_eta];
    let sens0 = subject_sensitivities_cov(model, subject, &params.theta, &zeros)?;
    let prep = prepare(model, subject, &params, &sens)?;
    if !prep.censored.is_empty() {
        return None;
    }
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
    use crate::types::{CompiledModel, DoseEvent, ModelParameters, OmegaMatrix, Subject};
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
        let prep = prepare(model, subject, params, &sens).unwrap();
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
        let prep = prepare(model, subject, params, &sens).unwrap();
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
        let prep = prepare(model, subject, params, &sens).unwrap();
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
            let h = 1e-6 * (1.0 + x[col].abs());
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
        let prep = prepare(&model, &subject, &params, &sens).unwrap();
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
            let p = prepare(&model, &subject, &params, &s).unwrap();
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
                    let e =
                        find_ebe(&model, &subject, &pp, 30, 1e-8, Some(eta.as_slice()), None).eta;
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
}
