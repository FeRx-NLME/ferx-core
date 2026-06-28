//! Paper-exact FOCEI outer gradient (Almquist 2015, Eq. 23) for analytical PK
//! models, assembled in closed form from the [`crate::sens`] provider.
//!
//! The per-subject FOCEI Laplace objective is
//!
//! ```text
//!   Fᵢ = ½ Σⱼ (εⱼ²/Rⱼ + ln Rⱼ) + ½ η̂ᵀΩ⁻¹η̂ + ½ ln|Ω| + ½ log|H̃|.
//! ```
//!
//! Its total derivative w.r.t. a population parameter pulls in the EBE response
//! `dη̂/dζ` (Eq. 46). Writing `aⱼ = ∂f/∂η`, `Aⱼ = ∂²f/∂η²`, `bⱼ = ∂f/∂θ`,
//! `Bⱼ = ∂²f/∂η∂θ` — all exact from the provider — and the error-model scalars
//! `R, d = ∂R/∂f, d2 = ∂²R/∂f²`:
//!
//! * `αⱼ = −2ε/R + d(R−ε²)/R²`,  `α'ⱼ = dαⱼ/df`
//! * `pⱼ = 1/R + ½(d/R)²`,        `βⱼ = dpⱼ/df = −d/R² + d·d2/R² − d³/R³`
//! * `H̃ = Σⱼ pⱼ aⱼaⱼᵀ + Ω⁻¹`,    `wⱼ = H̃⁻¹aⱼ`,  `qⱼ = aⱼᵀwⱼ`
//! * true inner Hessian `H = ½ Σⱼ (α'ⱼ aⱼaⱼᵀ + αⱼ Aⱼ) + Ω⁻¹`
//! * mixed `M[:,m] = ½ Σⱼ (α'ⱼ bⱼₘ aⱼ + αⱼ Bⱼ[:,m])`,  `dη̂/dθₘ = −H⁻¹ M[:,m]`
//! * `∂log|H̃|/∂η_l = Σⱼ (βⱼ qⱼ a_{jl} + 2 pⱼ Σₖ w_{jk} A_{jkl})`
//!
//! giving the per-subject θ-gradient
//!
//! ```text
//!   dFᵢ/dθₘ = ½ Σⱼ (αⱼ + βⱼqⱼ) bⱼₘ          (data + a-fixed log|H̃|)
//!           +    Σⱼ pⱼ Σₖ w_{jk} B_{jkm}      (∂²f/∂η∂θ curvature)
//!           + ½ Σ_l (∂log|H̃|/∂η_l) dη̂_l/dθₘ  (Eq. 46 EBE response)
//! ```
//!
//! This is the noise-free closed-form replacement for the previous FD-over-AD
//! curvature/EBE-response path (issue #367). Scope is whatever
//! [`crate::sens::provider::subject_sensitivities`] supports (analytical
//! 1-/2-/3-cpt); callers fall back to the existing FD/Laplace path otherwise.
// Indexed loops index parallel grad/Hessian/Jacobian buffers; clearer than zips.
#![allow(clippy::needless_range_loop)]

use crate::estimation::parameterization::{packed_fixed_mask, theta_packs_log, unpack_params};
use crate::sens::provider::{subject_sensitivities, ObsSens, SubjectSens};
use crate::types::{CompiledModel, ModelParameters, Population, Subject};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Per-observation error-model scalars used throughout the assembly.
struct ErrTerms {
    r: f64,       // Rⱼ
    d: f64,       // dⱼ = ∂R/∂f
    eps: f64,     // εⱼ = y − f
    alpha: f64,   // αⱼ
    alpha_p: f64, // α'ⱼ = dαⱼ/df
    p: f64,       // pⱼ
    beta: f64,    // βⱼ = dpⱼ/df
}

/// Inverse Mills ratio `h = φ(z)/Φ(z)`, evaluated through logs so it stays finite
/// in the far tail (`Φ(z)→0` when the prediction sits well above the LLOQ).
fn inv_mills(z: f64) -> f64 {
    let ln_phi = -0.5 * z * z - 0.5 * std::f64::consts::TAU.ln();
    (ln_phi - crate::stats::special::log_normal_cdf(z)).exp()
}

/// M3 censored-row scalars `(g1, g2) = (∂L/∂f, ∂²L/∂f²)` for the data term
/// `L = −logΦ(z)`, `z = (y−f)/√R`, with `y` the LLOQ, `R(f)` the residual
/// variance, `d = ∂R/∂f`, `d2 = ∂²R/∂f²`. Uses `∂z/∂f = −m` and
/// `dh/dz = −h(h+z)`, where `m = 1/w + (y−f)d/(2w³)`, `w = √R`:
/// ```text
///   g1 = h·m
///   g2 = h(h+z)·m² + h·∂m/∂f,
///   ∂m/∂f = [−2d + (y−f)d2]/(2w³) − 3(y−f)d²/(4w⁵).
/// ```
fn m3_censored_scalars(y: f64, f: f64, r: f64, d: f64, d2: f64) -> (f64, f64) {
    let w = r.sqrt();
    let w3 = r * w; // w³
    let w5 = r * r * w; // w⁵
    let z = (y - f) / w;
    let h = inv_mills(z);
    let m = 1.0 / w + (y - f) * d / (2.0 * w3);
    let g1 = h * m;
    let dm_df = (-2.0 * d + (y - f) * d2) / (2.0 * w3) - 3.0 * (y - f) * d * d / (4.0 * w5);
    let g2 = h * (h + z) * m * m + h * dm_df;
    (g1, g2)
}

fn err_terms(r: f64, d: f64, d2: f64, eps: f64) -> ErrTerms {
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;
    let inv_r3 = inv_r2 * inv_r;
    let alpha = -2.0 * eps * inv_r + d * (r - eps * eps) * inv_r2;
    // α'ⱼ = dαⱼ/df with dε/df = −1, dR/df = d, dd/df = d2:
    //   = 2/R + 2εd/R² + [d2(R−ε²) + d² + 2dε]/R² − 2d²(R−ε²)/R³.
    let alpha_p = 2.0 * inv_r
        + 2.0 * eps * d * inv_r2
        + (d2 * (r - eps * eps) + d * d + 2.0 * d * eps) * inv_r2
        - 2.0 * d * d * (r - eps * eps) * inv_r3;
    let p = inv_r + 0.5 * (d * inv_r) * (d * inv_r);
    let beta = -d * inv_r2 + d * d2 * inv_r2 - d * d * d * inv_r3;
    ErrTerms {
        r,
        d,
        eps,
        alpha,
        alpha_p,
        p,
        beta,
    }
}

/// Shared per-subject quantities the θ/Ω/Σ gradient blocks all consume, built
/// once from the provider sensitivities at the EBE.
struct Prep {
    n_eta: usize,
    n_obs: usize,
    et: Vec<ErrTerms>,
    /// `Ω⁻¹` (copied so blocks don't borrow `params`).
    omega_inv: DMatrix<f64>,
    /// `H̃⁻¹` (first-order FOCEI Hessian inverse).
    htilde_inv: DMatrix<f64>,
    /// `H⁻¹` for the **true** inner Hessian `H = ∂²lᵢ/∂η²` (Eq. 46 denominator).
    h_inner_inv: DMatrix<f64>,
    /// `wⱼ = H̃⁻¹aⱼ`.
    w: Vec<DVector<f64>>,
    /// `qⱼ = aⱼᵀ H̃⁻¹ aⱼ`.
    q: Vec<f64>,
    /// Exact `∂log|H̃|/∂η` (a-fixed part + `∂²f/∂η²` curvature).
    g_eta: Vec<f64>,
    /// Per-observation M3-censored flag. Censored rows enter `H` (true inner
    /// Hessian) and the data gradient but carry `p = β = 0`, so they are excluded
    /// from `H̃` / `log|H̃|` — matching `gaussian_foce_accum`, which accumulates
    /// `hrh`/`ctc` over quantified rows only and adds the censored `−logΦ` to the
    /// data term. Empty `Vec` (all-false) when the subject has no censoring.
    censored: Vec<bool>,
    /// IIV-on-RUV (`Y = IPRED + EPS·EXP(η_ruv)`, #474): the random-effect index
    /// that scales the residual variance by `exp(2·η_ruv)`, or `None`. When set,
    /// the variance terms `r`/`d` in `et` already carry that factor, and the
    /// `η_ruv` row/col of `H̃`/`H` plus its `log|H̃|` derivatives are assembled
    /// from the per-observation `g_ruv`/`gp_ruv` scalars below.
    ruv: Option<usize>,
    /// `gⱼ = (∂Rⱼ/∂fⱼ)/Rⱼ = dⱼ/Rⱼ` per observation (scale-invariant) — the
    /// residual-eta `c̃` cross coupling `H̃[ruv,l] = Σⱼ gⱼ a_{jl}`. Empty when no
    /// `ruv`.
    g_ruv: Vec<f64>,
    /// `g'ⱼ = ∂gⱼ/∂fⱼ = d2ⱼ/Rⱼ − (dⱼ/Rⱼ)²` per observation (scale-invariant) — the
    /// `f`-derivative of the `c̃` coupling, needed for `∂log|H̃|/∂θ` and `/∂η`.
    /// Empty when no `ruv`.
    gp_ruv: Vec<f64>,
    /// `exp(2·η̂_ruv)` (1.0 when no `ruv`) — the residual-variance scale, used to
    /// lift the σ-block's central-FD `∂R/∂σ` / `∂d/∂σ` (taken on the *unscaled*
    /// error functions) onto the scaled variance.
    ruv_scale: f64,
}

/// Residual-eta coupling `κⱼ = ∂(1−ε²/R)/∂f = 2ε/R + ε²d/R²` — the `f`-derivative
/// of the residual-eta data gradient, used in the mixed η-θ block and the true
/// inner Hessian's residual-eta row (#474). Single source so the two assemblies
/// can't diverge.
#[inline]
fn ruv_kappa(eps: f64, r: f64, d: f64) -> f64 {
    2.0 * eps / r + eps * eps * d / (r * r)
}

fn prepare(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    eta_hat: &[f64],
) -> Option<Prep> {
    prepare_stacked(
        model,
        subject,
        params,
        sens,
        model.n_eta,
        params.omega.inv.clone(),
        eta_hat,
        model.residual_error_eta,
    )
}

/// [`prepare`] generalized over the random-effect dimension and prior precision,
/// so it serves both the non-IOV path (`n_eta = model.n_eta`, `Ω⁻¹ = params.omega.inv`)
/// and the **IOV** path, where the random effects are the stacked
/// `[η_bsv, κ₁..κ_K]` and `omega_inv` is the inverse of the block-diagonal
/// `Ω_bsv ⊕ K·Ω_iov`. Everything else (error model, σ, censoring) is shared.
#[allow(clippy::too_many_arguments)]
fn prepare_stacked(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    n_eta: usize,
    omega_inv: DMatrix<f64>,
    eta_hat: &[f64],
    ruv: Option<usize>,
) -> Option<Prep> {
    let n_obs = subject.observations.len();
    // A dosing-only subject (dose rows, no DV) contributes no data term to the
    // marginal gradient; with no observations `H̃ = Ω⁻¹` is still PD so the
    // FOCEI blocks (`theta_block`, `mixed_eta_theta`) would proceed and then
    // index `obs[0]`, panicking and aborting the fit. Decline like the FOCE
    // siblings (`subject_packed_gradient_foce`) so the caller falls back to FD,
    // which handles the empty subject correctly (PR #381 review #1).
    if n_obs == 0 {
        return None;
    }
    if sens.obs.len() != n_obs {
        return None;
    }
    let sigma = &params.sigma.values;
    // IIV-on-RUV (#474): every residual variance scales by `s = exp(2·η̂_ruv)`, so
    // `r`/`d`/`d2` carry that factor below. `η_ruv` enters the likelihood only
    // through the variance (`∂f/∂η_ruv = 0`), contributing the `c̃` interaction
    // column `c̃_{j,ruv} = 2` to `H̃` (Almquist), the true-Hessian terms
    // `∂²lᵢ/∂η_ruv² = Σ 2ε²/R`, `∂²lᵢ/∂η_ruv∂η_l = Σ κⱼ a_{jl}`, and the matching
    // `log|H̃|` derivatives. (M3 BLOQ + `iiv_on_ruv` routes to FD upstream, so no
    // censored residual-eta second derivatives are needed here.)
    // Reuse the canonical `exp(2·η_ruv)` link (the IOV path forces `ruv = None`
    // to keep its residual variance unscaled, so gate on the local `ruv`, not on
    // `model.residual_error_eta`).
    let ruv_scale = if ruv.is_some() {
        model.residual_var_scale(eta_hat)
    } else {
        1.0
    };

    // H̃ = Σ pⱼ aⱼaⱼᵀ + Ω⁻¹ ; true inner Hessian H = ½Σ(α'ⱼ aⱼaⱼᵀ + αⱼ Aⱼ) + Ω⁻¹.
    let mut htilde = omega_inv.clone();
    let mut h_inner = omega_inv.clone();
    let mut et: Vec<ErrTerms> = Vec::with_capacity(n_obs);
    let (mut g_ruv, mut gp_ruv) = if ruv.is_some() {
        (vec![0.0f64; n_obs], vec![0.0f64; n_obs])
    } else {
        (Vec::new(), Vec::new())
    };

    let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3);
    let mut censored = vec![false; n_obs];
    let mut any_cens = false;
    for obs in sens.obs.iter() {
        let f = obs.f;
        // obs index → cmt: provider obs are parallel to subject.obs_times.
        let j = et.len();
        let cmt = subject.obs_cmts[j];
        let r = model.error_spec.variance_at(cmt, f, sigma) * ruv_scale;
        if !(r.is_finite() && r > 0.0) {
            return None;
        }
        let d = model.error_spec.dvar_df(cmt, f, sigma) * ruv_scale;
        let d2 = model.error_spec.d2var_df2(cmt, sigma) * ruv_scale;
        let y = subject.observations[j];
        let is_cens = m3 && subject.cens.get(j).copied().unwrap_or(0) != 0;
        // For a censored row the data term is `−logΦ(z)`: store its f-derivatives
        // as `alpha = 2·g1`, `alpha_p = 2·g2` (so the assembly's `½α`, `½α'` recover
        // `∂L/∂f`, `∂²L/∂f²`) and force `p = β = 0` (excluded from `H̃` / `log|H̃|`).
        let t = if is_cens {
            // M3 + `iiv_on_ruv` routes to FD upstream (`analytic_outer_gradient_
            // available`), so `is_cens` and `ruv` never co-occur on a reachable call.
            // `r`/`d`/`d2` here carry `ruv_scale`; if a future caller bypassed the
            // gate, the censored scalars would get a doubly-applied scale — pin it.
            debug_assert!(
                ruv.is_none(),
                "M3 censoring with iiv_on_ruv must route to FD (gate bypassed)"
            );
            censored[j] = true;
            any_cens = true;
            let (g1, g2) = m3_censored_scalars(y, f, r, d, d2);
            ErrTerms {
                r,
                d,
                eps: y - f,
                alpha: 2.0 * g1,
                alpha_p: 2.0 * g2,
                p: 0.0,
                beta: 0.0,
            }
        } else {
            err_terms(r, d, d2, y - f)
        };

        let a = obs.df_deta.as_slice();
        for k in 0..n_eta {
            for l in 0..n_eta {
                htilde[(k, l)] += t.p * a[k] * a[l];
                h_inner[(k, l)] +=
                    0.5 * (t.alpha_p * a[k] * a[l] + t.alpha * obs.d2f_deta2[k * n_eta + l]);
            }
        }
        // Residual-eta rows/cols (`a_{j,ruv} = 0`, so the loop above left them at
        // their `Ω⁻¹` value). `c̃_{j,ruv} = 2` ⇒ `½ c̃ c̃ᵀ` gives `H̃[ruv,ruv] += 2`
        // and `H̃[ruv,l] += gⱼ a_{jl}` (`gⱼ = dⱼ/Rⱼ`); the true Hessian gets
        // `H[ruv,ruv] += 2ε²/R` and `H[ruv,l] += κⱼ a_{jl}`.
        if let Some(rr) = ruv {
            let eps = t.eps;
            let g = t.d / t.r;
            g_ruv[j] = g;
            gp_ruv[j] = d2 / t.r - g * g;
            let kappa = ruv_kappa(eps, t.r, t.d);
            htilde[(rr, rr)] += 2.0;
            h_inner[(rr, rr)] += 2.0 * eps * eps / t.r;
            for l in 0..n_eta {
                if l == rr {
                    continue;
                }
                htilde[(rr, l)] += g * a[l];
                htilde[(l, rr)] += g * a[l];
                h_inner[(rr, l)] += kappa * a[l];
                h_inner[(l, rr)] += kappa * a[l];
            }
        }
        et.push(t);
    }
    let censored = if any_cens { censored } else { Vec::new() };

    let htilde_inv = htilde.cholesky()?.inverse();
    let h_inner_inv = h_inner.cholesky()?.inverse();

    let mut w: Vec<DVector<f64>> = Vec::with_capacity(n_obs);
    let mut q = vec![0.0f64; n_obs];
    for (j, obs) in sens.obs.iter().enumerate() {
        let aj = DVector::from_column_slice(&obs.df_deta);
        let wj = &htilde_inv * &aj;
        q[j] = aj.dot(&wj);
        w.push(wj);
    }

    // ∂log|H̃|/∂η_l = Σⱼ (βⱼ qⱼ a_{jl} + 2 pⱼ Σₖ w_{jk} A_{jkl}).
    let mut g_eta = vec![0.0f64; n_eta];
    for l in 0..n_eta {
        let mut s = 0.0;
        for (j, obs) in sens.obs.iter().enumerate() {
            s += et[j].beta * q[j] * obs.df_deta[l];
            let mut curv = 0.0;
            for k in 0..n_eta {
                curv += w[j][k] * obs.d2f_deta2[k * n_eta + l];
            }
            s += 2.0 * et[j].p * curv;
        }
        g_eta[l] = s;
    }
    // Residual-eta `log|H̃|` derivative. The loop above (using `∂p/∂f` and `a_{ruv}
    // = A_{·,ruv} = 0`) left the `c̃`-column contribution out. Add, per quant obs:
    //   ordinary l: 2( g'ⱼ wⱼ[ruv] a_{jl} + gⱼ Σₖ H̃⁻¹_{ruv,k} A_{jkl} )
    //   l = ruv:    −2 qⱼ/Rⱼ  (`∂p/∂η_ruv = −2/R`; the `c̃[ruv,ruv]=2` is constant)
    if let Some(rr) = ruv {
        for (j, obs) in sens.obs.iter().enumerate() {
            let wjr = w[j][rr];
            let (g, gp) = (g_ruv[j], gp_ruv[j]);
            for l in 0..n_eta {
                if l == rr {
                    continue;
                }
                let mut sa = 0.0;
                for k in 0..n_eta {
                    sa += htilde_inv[(rr, k)] * obs.d2f_deta2[k * n_eta + l];
                }
                g_eta[l] += 2.0 * (gp * wjr * obs.df_deta[l] + g * sa);
            }
            g_eta[rr] += -2.0 * q[j] / et[j].r;
        }
    }

    Some(Prep {
        n_eta,
        n_obs,
        et,
        omega_inv,
        htilde_inv,
        h_inner_inv,
        w,
        q,
        g_eta,
        censored,
        ruv,
        g_ruv,
        gp_ruv,
        ruv_scale,
    })
}

/// The exact per-subject θ-gradient `dFᵢ/dθ` (length `n_theta`, natural θ
/// space), or `None` when the model/subject is outside the provider's scope.
///
/// `eta_hat` must be the EBE for `params` (the function evaluates the gradient
/// identity at the inner optimum; the envelope theorem and Eq. 46 both assume
/// `∂lᵢ/∂η|_η̂ = 0`).
pub fn subject_theta_gradient(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta_hat: &[f64],
) -> Option<Vec<f64>> {
    if subject.observations.is_empty() {
        return Some(vec![0.0; params.theta.len()]);
    }
    let sens = subject_sensitivities(model, subject, &params.theta, eta_hat)?;
    let prep = prepare(model, subject, params, &sens, eta_hat)?;
    Some(theta_block(&prep, &sens, params.theta.len()))
}

fn theta_block(prep: &Prep, sens: &SubjectSens, n_theta: usize) -> Vec<f64> {
    let (n_eta, n_obs) = (prep.n_eta, prep.n_obs);
    let mut grad = vec![0.0f64; n_theta];
    for m in 0..n_theta {
        // data + a-fixed log|H̃|:  ½ Σⱼ (αⱼ + βⱼqⱼ) bⱼₘ ; plus ∂²f/∂η∂θ curvature.
        let mut g = 0.0;
        for (j, obs) in sens.obs.iter().enumerate() {
            let bjm = obs.df_dtheta[m];
            g += 0.5 * (prep.et[j].alpha + prep.et[j].beta * prep.q[j]) * bjm;
            let mut curv = 0.0;
            for k in 0..n_eta {
                curv += prep.w[j][k] * obs.d2f_deta_dtheta[k * n_theta + m];
            }
            g += prep.et[j].p * curv;
        }
        // Residual-eta `log|H̃|` θ-derivative (`∂(c̃-column)/∂θ`), per quant obs:
        //   gⱼ' bⱼₘ wⱼ[ruv] + gⱼ Σ_l H̃⁻¹_{ruv,l} B_{jlm}   (B = ∂²f/∂η∂θ).
        if let Some(rr) = prep.ruv {
            for (j, obs) in sens.obs.iter().enumerate() {
                let mut sb = 0.0;
                for l in 0..n_eta {
                    sb += prep.htilde_inv[(rr, l)] * obs.d2f_deta_dtheta[l * n_theta + m];
                }
                g += prep.gp_ruv[j] * obs.df_dtheta[m] * prep.w[j][rr] + prep.g_ruv[j] * sb;
            }
        }
        // EBE response: ½ g_eta · dη̂/dθₘ,  dη̂/dθₘ = −H⁻¹ M[:,m].
        let m_vec = mixed_eta_theta(&sens.obs, &prep.et, n_eta, n_obs, m, prep.ruv);
        let deta = -(&prep.h_inner_inv * m_vec);
        let mut resp = 0.0;
        for l in 0..n_eta {
            resp += prep.g_eta[l] * deta[l];
        }
        grad[m] = g + 0.5 * resp;
    }
    grad
}

/// The exact per-subject θ-gradient for an analytical **IOV** subject, evaluated
/// over the stacked random-effects vector `[η_bsv, κ₁..κ_K]` with the
/// block-diagonal prior `Ω = Ω_bsv ⊕ K·Ω_iov`. `None` outside the IOV-analytical
/// scope (caller falls back). `stacked_eta_hat` must be the joint EBE for `params`
/// (the gradient identity holds at the inner optimum).
///
/// The IOV FOCEI marginal (`foce_subject_nll_iov`) is exactly the ordinary FOCEI
/// Laplace objective over the augmented system `b = [η, κ]` with prior `Σ_b`, so
/// the same paper-exact assembly applies — only `n_eta` and `Ω⁻¹` change.
pub fn subject_theta_gradient_iov(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    stacked_eta_hat: &[f64],
) -> Option<Vec<f64>> {
    let sens = crate::sens::provider::subject_sensitivities_iov(
        model,
        subject,
        &params.theta,
        stacked_eta_hat,
    )?;
    let k_groups = crate::stats::likelihood::split_obs_by_occasion(subject).len();
    let n_stacked = model.n_eta + k_groups * model.n_kappa;
    if stacked_eta_hat.len() != n_stacked {
        return None;
    }
    let omega_iov = params.omega_iov.as_ref()?;
    let block = crate::stats::likelihood::build_block_diag_omega(
        &params.omega.matrix,
        &omega_iov.matrix,
        k_groups,
    );
    let omega_inv = block.cholesky()?.inverse();
    let prep = prepare_stacked(
        model,
        subject,
        params,
        &sens,
        n_stacked,
        omega_inv,
        stacked_eta_hat,
        None,
    )?;
    Some(theta_block(&prep, &sens, params.theta.len()))
}

/// The exact per-subject Ω-gradient `dFᵢ/dΩ` over the free Ω entries, in the
/// same order the optimizer packs them (diagonal: `(i,i)`; block: lower triangle
/// `(i,j)`, `j ≤ i`), natural variance/covariance scale. `None` when unsupported.
///
/// Per free entry `(r,c)` with `z = Ω⁻¹η̂`, `G = Ω⁻¹H̃⁻¹Ω⁻¹`, `v = Ω⁻¹H⁻¹g_eta`:
/// fixed-η̂ part `½[−zᵀEz + tr(Ω⁻¹E) − tr(GE)]` plus EBE response `½ vᵀEz`,
/// `E = ∂Ω/∂Ω_{rc}` (symmetric).
pub fn subject_omega_gradient(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta_hat: &[f64],
) -> Option<Vec<f64>> {
    if subject.observations.is_empty() {
        let n = if params.omega.diagonal {
            model.n_eta
        } else {
            model.n_eta * (model.n_eta + 1) / 2
        };
        return Some(vec![0.0; n]);
    }
    let sens = subject_sensitivities(model, subject, &params.theta, eta_hat)?;
    let prep = prepare(model, subject, params, &sens, eta_hat)?;
    Some(omega_block(&prep, params, eta_hat))
}

fn omega_block(prep: &Prep, params: &ModelParameters, eta_hat: &[f64]) -> Vec<f64> {
    let n_eta = prep.n_eta;
    let eta = DVector::from_column_slice(eta_hat);
    let z = &prep.omega_inv * &eta;
    let g_mat = &prep.omega_inv * &prep.htilde_inv * &prep.omega_inv;
    let u = &prep.h_inner_inv * DVector::from_column_slice(&prep.g_eta);
    let v = &prep.omega_inv * u;

    let entries: Vec<(usize, usize)> = if params.omega.diagonal {
        (0..n_eta).map(|i| (i, i)).collect()
    } else {
        let mut e = Vec::new();
        for c in 0..n_eta {
            for r in c..n_eta {
                e.push((r, c));
            }
        }
        e
    };

    entries
        .iter()
        .map(|&(r, c)| {
            if r == c {
                // E has a single 1 at (r,r).
                let fixed = 0.5 * (-z[r] * z[r] + prep.omega_inv[(r, r)] - g_mat[(r, r)]);
                let resp = 0.5 * v[r] * z[r];
                fixed + resp
            } else {
                // Symmetric off-diagonal: E has 1 at (r,c) and (c,r).
                let fixed = -z[r] * z[c] + prep.omega_inv[(r, c)] - g_mat[(r, c)];
                let resp = 0.5 * (v[r] * z[c] + v[c] * z[r]);
                fixed + resp
            }
        })
        .collect()
}

/// The exact per-subject Σ-gradient `dFᵢ/dσ` (length `n_sigma`, natural σ
/// scale), or `None` when unsupported. σ enters only through the residual
/// variance, so `∂R/∂σ` and `∂d/∂σ` (`d = ∂R/∂f`) are taken by central FD of the
/// closed-form error functions — exact algebra, well-conditioned, no AD.
///
/// Per σ_k:  `½ Σⱼ Rσⱼ(Rⱼ−εⱼ²)/Rⱼ²` (data + lnR) `+ ½ Σⱼ (∂pⱼ/∂σ) qⱼ` (log|H̃|)
/// `+ ½ g_eta·dη̂/dσ`, with `dη̂/dσ = −H⁻¹ M`, `M[m] = ½ Σⱼ (∂αⱼ/∂σ) a_{jm}`.
pub fn subject_sigma_gradient(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta_hat: &[f64],
) -> Option<Vec<f64>> {
    if subject.observations.is_empty() {
        return Some(vec![0.0; params.sigma.values.len()]);
    }
    let sens = subject_sensitivities(model, subject, &params.theta, eta_hat)?;
    let prep = prepare(model, subject, params, &sens, eta_hat)?;
    Some(sigma_block(&prep, model, subject, params, &sens))
}

/// Central-difference half-step for a σ finite difference that keeps the minus
/// side `σ − h` strictly positive. The error models build the variance from `σ²`
/// and `variance_at` floors it at `MIN_VARIANCE`, so once the minus-side variance
/// underflows the floor (near a near-zero residual error) the central difference
/// is corrupted; shrinking the step near `σ = 0` keeps `∂/∂σ` well-defined
/// (PR #381 review #6). For an ordinary σ the `1e-6·(1+|σ|)` step is unchanged.
fn sigma_fd_step(sigma_k: f64) -> f64 {
    let h = 1e-6 * (1.0 + sigma_k.abs());
    if sigma_k > 0.0 && h >= sigma_k {
        0.5 * sigma_k
    } else {
        h
    }
}

fn sigma_block(
    prep: &Prep,
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
) -> Vec<f64> {
    let n_eta = prep.n_eta;
    let sigma = &params.sigma.values;
    let n_sigma = sigma.len();
    let mut grad = vec![0.0f64; n_sigma];

    for k in 0..n_sigma {
        let h = sigma_fd_step(sigma[k]);
        let mut sp = sigma.clone();
        sp[k] += h;
        let mut sm = sigma.clone();
        sm[k] -= h;

        let mut fixed = 0.0;
        let mut m_vec = DVector::<f64>::zeros(n_eta);
        for (j, obs) in sens.obs.iter().enumerate() {
            let cmt = subject.obs_cmts[j];
            let f = obs.f;
            if prep.censored.get(j).copied().unwrap_or(false) {
                // M3 censored row: data term `−logΦ((y−f)/√R(σ))` (not in `H̃`, so
                // no `log|H̃|` σ-term). `∂L/∂σ` and `∂g1/∂σ = ½∂α/∂σ` by central FD
                // of the closed-form censored scalars, mirroring the Gaussian path.
                let y = subject.observations[j];
                let l_sig = (-crate::stats::special::log_normal_cdf(
                    (y - f) / model.error_spec.variance_at(cmt, f, &sp).sqrt(),
                ) + crate::stats::special::log_normal_cdf(
                    (y - f) / model.error_spec.variance_at(cmt, f, &sm).sqrt(),
                )) / (2.0 * h);
                fixed += l_sig;
                let (g1p, _) = m3_censored_scalars(
                    y,
                    f,
                    model.error_spec.variance_at(cmt, f, &sp),
                    model.error_spec.dvar_df(cmt, f, &sp),
                    model.error_spec.d2var_df2(cmt, &sp),
                );
                let (g1m, _) = m3_censored_scalars(
                    y,
                    f,
                    model.error_spec.variance_at(cmt, f, &sm),
                    model.error_spec.dvar_df(cmt, f, &sm),
                    model.error_spec.d2var_df2(cmt, &sm),
                );
                let dg1 = (g1p - g1m) / (2.0 * h);
                for m in 0..n_eta {
                    m_vec[m] += dg1 * obs.df_deta[m];
                }
                continue;
            }
            let (r, d, eps) = (prep.et[j].r, prep.et[j].d, prep.et[j].eps);
            // Evaluate the four closed-form error functions once at σ±h and reuse
            // them for `r_sig`/`d_sig` and the residual-eta `g_sig` below.
            let vp = model.error_spec.variance_at(cmt, f, &sp);
            let vm = model.error_spec.variance_at(cmt, f, &sm);
            let dp_var = model.error_spec.dvar_df(cmt, f, &sp);
            let dm_var = model.error_spec.dvar_df(cmt, f, &sm);
            // ∂R/∂σ_k, ∂d/∂σ_k by central FD. `et.r`/`et.d` carry the `exp(2·η_ruv)`
            // scale, so lift these too.
            let r_sig = prep.ruv_scale * (vp - vm) / (2.0 * h);
            let d_sig = prep.ruv_scale * (dp_var - dm_var) / (2.0 * h);

            let inv_r = 1.0 / r;
            let inv_r2 = inv_r * inv_r;
            let inv_r3 = inv_r2 * inv_r;

            // data + lnR:  ½ Rσ (R − ε²)/R²
            fixed += 0.5 * r_sig * (r - eps * eps) * inv_r2;
            // log|H̃|:  ½ (∂p/∂σ) q ,  ∂p/∂σ = −Rσ/R² + d·dσ/R² − d²Rσ/R³
            let dp = -r_sig * inv_r2 + d * d_sig * inv_r2 - d * d * r_sig * inv_r3;
            fixed += 0.5 * dp * prep.q[j];

            // ∂α/∂σ = [2ε/R² + d(2ε²−R)/R³] Rσ + [(R−ε²)/R²] dσ
            let dalpha = (2.0 * eps * inv_r2 + d * (2.0 * eps * eps - r) * inv_r3) * r_sig
                + ((r - eps * eps) * inv_r2) * d_sig;
            for m in 0..n_eta {
                m_vec[m] += 0.5 * dalpha * obs.df_deta[m];
            }
            // Residual-eta terms (#474). `∂R/∂σ` scales `R`, so:
            //   M[ruv] = ∂(1−ε²/R)/∂σ = ε²/R² · Rσ   (the residual-eta row of M)
            //   ∂log|H̃|/∂σ gains  (∂gⱼ/∂σ)·wⱼ[ruv]  with `gⱼ = d/R` (scale-free,
            //     so FD the unscaled quotient directly).
            if let Some(rr) = prep.ruv {
                m_vec[rr] += eps * eps * inv_r2 * r_sig;
                // `gⱼ = d/R` is scale-free, so FD the unscaled quotient directly.
                let g_sig = (dp_var / vp - dm_var / vm) / (2.0 * h);
                fixed += g_sig * prep.w[j][rr];
            }
        }

        let deta = -(&prep.h_inner_inv * m_vec);
        let mut resp = 0.0;
        for l in 0..n_eta {
            resp += prep.g_eta[l] * deta[l];
        }
        grad[k] = fixed + 0.5 * resp;
    }
    grad
}

/// Per-Ω-Cholesky-entry packed gradient `∂Fᵢ/∂x` in `pack_params` order
/// (diagonal: `ln L_ii`; block: lower-triangle `(i,j)`, off-diagonals raw). The
/// fixed-η̂ part is the existing closed form (the inner factor-2 cancels the
/// outer ½, so it is the *full* ∂NLL/∂x), augmented with the Eq. 46 EBE-response
/// `tᵢ = ½·g_eta·dη̂/dL` mapped into L-space:
/// `t_{L,rc} = ½[(v·z)·s_r + z_r·(s·v)]`, `v = L[:,c]`, `s = Ω⁻¹H⁻¹g_eta`
/// (×`L_kk` for the diagonal log-chain).
fn omega_packed_block(prep: &Prep, params: &ModelParameters, eta_hat: &[f64]) -> Vec<f64> {
    let n_eta = prep.n_eta;
    let l = &params.omega.chol;
    let z = &prep.omega_inv * DVector::from_column_slice(eta_hat);
    let g_mat = &prep.omega_inv * &prep.htilde_inv * &prep.omega_inv;
    let s = &prep.omega_inv * (&prep.h_inner_inv * DVector::from_column_slice(&prep.g_eta));

    let entries: Vec<(usize, usize)> = if params.omega.diagonal {
        (0..n_eta).map(|i| (i, i)).collect()
    } else {
        let mut e = Vec::new();
        for c in 0..n_eta {
            for r in c..n_eta {
                e.push((r, c));
            }
        }
        e
    };

    entries
        .iter()
        .map(|&(row, col)| {
            let v: Vec<f64> = (0..n_eta).map(|r| l[(r, col)]).collect();
            let vz: f64 = v.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
            let gv_row: f64 = (0..n_eta).map(|c| g_mat[(row, c)] * v[c]).sum();
            let sv: f64 = v.iter().zip(s.iter()).map(|(a, b)| a * b).sum();
            let t = 0.5 * (vz * s[row] + z[row] * sv);
            if row == col {
                let l_kk = l[(row, row)];
                (-l_kk * z[row] * vz + 1.0 - l_kk * gv_row) + l_kk * t
            } else {
                (-z[row] * vz - gv_row) + t
            }
        })
        .collect()
}

/// Symmetric per-entry natural Ω-gradient `M_{rc} = ∂Fᵢ/∂Ω_{rc}` (treating every
/// entry independently), as a matrix. Built from the same closed form as
/// [`omega_block`]: fixed `½(−z zᵀ + Ω⁻¹ − G)` plus EBE response `¼(v zᵀ + z vᵀ)`,
/// with `z = Ω⁻¹η̂`, `G = Ω⁻¹H̃⁻¹Ω⁻¹`, `v = Ω⁻¹H⁻¹g_eta`. (The free-parameter
/// gradient `omega_block` returns is `M_{rc}+M_{cr}` off-diagonal; this keeps the
/// matrix form so it can be sub-blocked and Cholesky-mapped for IOV.)
fn natural_omega_grad_matrix(prep: &Prep, eta_hat: &[f64]) -> DMatrix<f64> {
    let n = prep.n_eta;
    let eta = DVector::from_column_slice(eta_hat);
    let z = &prep.omega_inv * &eta;
    let g = &prep.omega_inv * &prep.htilde_inv * &prep.omega_inv;
    let v = &prep.omega_inv * (&prep.h_inner_inv * DVector::from_column_slice(&prep.g_eta));
    let mut m = DMatrix::zeros(n, n);
    for r in 0..n {
        for c in 0..n {
            let fixed = 0.5 * (-z[r] * z[c] + prep.omega_inv[(r, c)] - g[(r, c)]);
            let resp = 0.25 * (v[r] * z[c] + z[r] * v[c]);
            m[(r, c)] = fixed + resp;
        }
    }
    m
}

/// Map a sub-block's natural symmetric gradient `M_sub` (`∂F/∂Ω_sub`) to the
/// packed Cholesky-space gradient for that block: `∂F/∂L = 2·M_sub·L` (L lower-
/// triangular), with the diagonal log-chain (`x_ii = ln L_ii ⇒ ×L_ii`) and raw
/// off-diagonals — the same convention/order as [`omega_packed_block`] /
/// `pack_params`.
fn chol_pack(m_sub: &DMatrix<f64>, l: &DMatrix<f64>, diagonal: bool) -> Vec<f64> {
    let n = l.nrows();
    let gl = (m_sub * l).scale(2.0);
    let mut out = Vec::new();
    if diagonal {
        for i in 0..n {
            out.push(gl[(i, i)] * l[(i, i)]);
        }
    } else {
        for j in 0..n {
            for i in j..n {
                if i == j {
                    out.push(gl[(i, i)] * l[(i, i)]);
                } else {
                    out.push(gl[(i, j)]);
                }
            }
        }
    }
    out
}

/// The exact per-subject FOCEI packed gradient `dFᵢ/dx` for an analytical **IOV**
/// subject, in `pack_params` order `[θ, Ω_bsv, σ, Ω_iov]`. `stacked_eta_hat` is
/// the joint EBE `[η_bsv, κ₁..κ_K]` for `unpack_params(x)`. `None` outside the
/// IOV-analytical scope.
///
/// The θ and σ blocks reuse the stacked-η assembly unchanged. The Ω blocks split
/// the **block-diagonal** `Σ_b = Ω_bsv ⊕ K·Ω_iov`: the BSV packed gradient is the
/// top-left sub-block of the natural gradient mapped through `L_bsv`; the IOV
/// packed gradient is the **sum** of the K diagonal IOV sub-blocks (the κ-variance
/// is shared across occasions — `∂F/∂L_iov = Σ_k 2·M_{block_k}·L_iov`) mapped
/// through `L_iov`.
pub fn subject_packed_gradient_iov(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    stacked_eta_hat: &[f64],
) -> Option<Vec<f64>> {
    let params = unpack_params(x, template);
    let sens = crate::sens::provider::subject_sensitivities_iov(
        model,
        subject,
        &params.theta,
        stacked_eta_hat,
    )?;
    let k = crate::stats::likelihood::split_obs_by_occasion(subject).len();
    let n_eta_bsv = model.n_eta;
    let n_iov = model.n_kappa;
    let n_stacked = n_eta_bsv + k * n_iov;
    if stacked_eta_hat.len() != n_stacked {
        return None;
    }
    let omega_iov = params.omega_iov.as_ref()?;
    let block = crate::stats::likelihood::build_block_diag_omega(
        &params.omega.matrix,
        &omega_iov.matrix,
        k,
    );
    let omega_inv = block.cholesky()?.inverse();
    let prep = prepare_stacked(
        model,
        subject,
        &params,
        &sens,
        n_stacked,
        omega_inv,
        stacked_eta_hat,
        None,
    )?;

    let n_theta = params.theta.len();
    let n_sigma = params.sigma.values.len();
    let mut g = vec![0.0f64; x.len()];

    // θ (log/identity chain).
    let g_theta = theta_block(&prep, &sens, n_theta);
    for m in 0..n_theta {
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        g[m] = g_theta[m] * dtheta_dx;
    }

    // Ω blocks from the natural symmetric gradient over the stacked Σ_b.
    let m_mat = natural_omega_grad_matrix(&prep, stacked_eta_hat);
    let m_bsv = m_mat.view((0, 0), (n_eta_bsv, n_eta_bsv)).into_owned();
    let bsv_packed = chol_pack(&m_bsv, &params.omega.chol, params.omega.diagonal);
    // Sum the K diagonal IOV sub-blocks (shared κ-variance / SAME).
    let mut m_iov = DMatrix::<f64>::zeros(n_iov, n_iov);
    for kk in 0..k {
        let off = n_eta_bsv + kk * n_iov;
        m_iov += m_mat.view((off, off), (n_iov, n_iov));
    }
    let iov_packed = chol_pack(&m_iov, &omega_iov.chol, omega_iov.diagonal);

    // σ (log-σ chain).
    let g_sigma = sigma_block(&prep, model, subject, &params, &sens);

    // Place in pack_params order: θ, Ω_bsv, σ, Ω_iov.
    let omega_start = n_theta;
    for (i, &val) in bsv_packed.iter().enumerate() {
        g[omega_start + i] = val;
    }
    let sigma_start = omega_start + bsv_packed.len();
    for kk in 0..n_sigma {
        g[sigma_start + kk] = g_sigma[kk] * params.sigma.values[kk];
    }
    let iov_start = sigma_start + n_sigma;
    for (i, &val) in iov_packed.iter().enumerate() {
        g[iov_start + i] = val;
    }

    Some(g)
}

/// The exact per-subject FOCEI gradient `dFᵢ/dx` in the **packed** optimizer
/// space (log-θ / Cholesky-Ω / log-σ), or `None` when unsupported. `eta_hat`
/// must be the EBE for `unpack_params(x)`.
pub fn subject_packed_gradient(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
) -> Option<Vec<f64>> {
    if subject.observations.is_empty() {
        return Some(vec![0.0; x.len()]);
    }
    // M3/BLOQ: censored rows enter through `prepare` (data term `−logΦ`, true
    // inner Hessian; excluded from `H̃`/`log|H̃|` — matching `gaussian_foce_accum`).
    // This is the FOCEI (interaction) path that non-IOV M3 promotes to; plain FOCE
    // with M3 has its own analytic censored path in `subject_packed_gradient_foce`
    // (guarded by `population_packed_gradient_m3_foce_matches_fd`). IOV+M3 routes to
    // FD via `iov_analytical_supported`.
    let params = unpack_params(x, template);
    let sens = subject_sensitivities(model, subject, &params.theta, eta_hat)?;
    let prep = prepare(model, subject, &params, &sens, eta_hat)?;

    let n_theta = params.theta.len();
    let n_sigma = params.sigma.values.len();
    let mut g = vec![0.0f64; x.len()];

    // θ: ∂F/∂x = ∂F/∂θ · ∂θ/∂x, ∂θ/∂x = θ (log) or 1 (identity).
    let g_theta = theta_block(&prep, &sens, n_theta);
    for m in 0..n_theta {
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        g[m] = g_theta[m] * dtheta_dx;
    }

    // Ω: packed Cholesky-L gradient (already in x-space).
    let omega_start = n_theta;
    let og = omega_packed_block(&prep, &params, eta_hat);
    let n_omega = og.len();
    for (ko, &val) in og.iter().enumerate() {
        g[omega_start + ko] = val;
    }

    // σ: ∂F/∂x = ∂F/∂σ · σ (log-σ chain).
    let sigma_start = omega_start + n_omega;
    let g_sigma = sigma_block(&prep, model, subject, &params, &sens);
    for k in 0..n_sigma {
        g[sigma_start + k] = g_sigma[k] * params.sigma.values[k];
    }

    Some(g)
}

/// The exact analytic population gradient `d(OFV)/dx = 2·Σᵢ dFᵢ/dx` in packed
/// space, or `None` if any subject is unsupported (caller falls back to FD).
/// Fixed coordinates are zeroed. `eta_hats[i]` must be subject `i`'s EBE at `x`.
pub fn population_gradient_sens(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[DVector<f64>],
) -> Option<Vec<f64>> {
    let n = x.len();
    // Per-subject gradients in parallel (the FD path this replaces was already
    // subject-parallel; PR #381 review #7). `collect::<Option<_>>` short-circuits
    // to `None` if any subject is out of analytic scope, and preserves subject
    // order so the accumulation below is bit-reproducible across runs.
    let per_subject: Vec<Vec<f64>> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            subject_packed_gradient(model, subject, template, x, eta_hats[i].as_slice())
        })
        .collect::<Option<Vec<_>>>()?;
    let mut grad = vec![0.0f64; n];
    for gi in &per_subject {
        for k in 0..n {
            grad[k] += 2.0 * gi[k];
        }
    }
    let fixed = packed_fixed_mask(template);
    for k in 0..n {
        if fixed[k] {
            grad[k] = 0.0;
        }
    }
    Some(grad)
}

/// Per-subject analytic packed gradients `dᵢ = d(nllᵢ)/dx` (FOCEI when
/// `interaction`, plain FOCE otherwise), with `None` for any subject the
/// analytic provider can't handle (SS+reset, time-varying covariates,
/// modeled-duration doses, EVID=2 reset). Unlike [`population_gradient_sens`],
/// which short-circuits the *whole* population to `None` on the first
/// out-of-scope subject, this exposes the per-subject result so the caller can
/// keep the exact analytic gradient for the in-scope subjects and fill only the
/// out-of-scope ones with a reconverged-FD gradient. One out-of-scope subject no
/// longer disables the exact gradient for the other thousands — the all-or-
/// nothing fallback dropped to the θ-only fixed-EBE gradient, whose biased Ω/σ
/// block stalled SLSQP/L-BFGS/MMA well above the derivative-free optimum
/// (focei-slsqp-fixed-ebe-gradient-bias). Caller scales each entry by 2 and
/// zeroes fixed coordinates when assembling the population sum.
pub fn per_subject_packed_gradients(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[DVector<f64>],
    interaction: bool,
) -> Vec<Option<Vec<f64>>> {
    population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            if interaction {
                subject_packed_gradient(model, subject, template, x, eta_hats[i].as_slice())
            } else {
                subject_packed_gradient_foce(model, subject, template, x, eta_hats[i].as_slice())
            }
        })
        .collect()
}

/// Per-subject analytic packed gradients for an **IOV** model — the IOV analogue of
/// [`per_subject_packed_gradients`], exposing `None` per out-of-scope subject (rather than
/// short-circuiting the whole population to FD) so the
/// caller can keep the exact gradient for in-scope subjects and fill the rest with a
/// per-subject reconverged FD (#466 review round 2). `eta_hats[i]` are the BSV EBEs and
/// `kappas[i]` the per-occasion κ̂; both are stacked into `[η_bsv, κ₁..κ_K]` per subject.
pub fn per_subject_packed_gradients_iov(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[DVector<f64>],
    kappas: &[Vec<DVector<f64>>],
    interaction: bool,
) -> Vec<Option<Vec<f64>>> {
    population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let mut stacked: Vec<f64> = eta_hats[i].iter().copied().collect();
            for kap in &kappas[i] {
                stacked.extend(kap.iter().copied());
            }
            if interaction {
                subject_packed_gradient_iov(model, subject, template, x, &stacked)
            } else {
                subject_packed_gradient_foce_iov(model, subject, template, x, &stacked)
            }
        })
        .collect()
}

/// The exact per-subject **FOCE** (non-interaction) packed gradient `dFᵢ/dx`, or
/// `None` when unsupported. ferx's FOCE objective is the Sheiner–Beal linearized
/// marginal (the algebraic equal of the paper's Laplace FOCE, Eq. 18, with the
/// residual variance independent of η):
///
/// ```text
///   Fᵢ = ½ [ ρᵀ R̃⁻¹ ρ + log|R̃| ],   ρ = y − f0,  f0 = f(η̂) − J·η̂,
///   R̃ = J Ω Jᵀ + diag(R⁰),  J = ∂f/∂η,  R⁰ⱼ = R(fⱼ(η=0)).
/// ```
///
/// The EBE η̂ is the **shared** posterior mode (the inner objective is the same
/// `individual_nll` FOCE and FOCEI both minimise), so the true inner Hessian and
/// the Eq. 46 response `dη̂/dx` are reused verbatim from [`subject_eta_dx`]; the
/// total derivative is `∂Fᵢ/∂x|_η̂ + c·dη̂/dx` with the coupling `c = ∂Fᵢ/∂η̂`.
/// Only the fixed-η̂ marginal partials and `c` are FOCE-specific (computed here).
pub fn subject_packed_gradient_foce(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
) -> Option<Vec<f64>> {
    let params = unpack_params(x, template);
    let n_eta = model.n_eta;
    let n_obs = subject.observations.len();
    if n_obs == 0 {
        return Some(vec![0.0; x.len()]);
    }
    let sens = subject_sensitivities(model, subject, &params.theta, eta_hat)?;
    // Residual variance R⁰ is frozen at the η=0 (typical-individual) prediction —
    // ferx's no-interaction semantics. One extra provider pass supplies f(η=0)
    // and ∂f(η=0)/∂θ (for ∂R⁰/∂θ); both reuse the analytic closed forms.
    let zeros = vec![0.0f64; n_eta];
    let sens0 = subject_sensitivities(model, subject, &params.theta, &zeros)?;
    if sens.obs.len() != n_obs || sens0.obs.len() != n_obs {
        return None;
    }

    let sigma = &params.sigma.values;
    let omega = &params.omega.matrix;

    // M3 BLOQ: censored rows leave the Sheiner–Beal marginal (R̃ and the quadratic
    // form are built over the quantified rows only) and re-enter as
    // `−logΦ((LLOQ − f(η̂))/√R⁰)` data terms — the same objective as
    // `foce_subject_nll_standard`. `quant` maps SB-local row i → original obs index.
    let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3)
        && subject.cens.iter().any(|&c| c != 0);
    let quant: Vec<usize> = (0..n_obs)
        .filter(|&j| !(m3 && subject.cens.get(j).copied().unwrap_or(0) != 0))
        .collect();
    let nq = quant.len();
    if nq == 0 {
        return None;
    }

    // J = ∂f/∂η (nq×n_eta), ρ = y − f0 = ε + J·η̂, R⁰ and d⁰ at f(η=0) — quant rows.
    let mut jmat = DMatrix::<f64>::zeros(nq, n_eta);
    let mut rho = DVector::<f64>::zeros(nq);
    let mut r0 = vec![0.0f64; nq];
    let mut d0 = vec![0.0f64; nq];
    for (i, &j) in quant.iter().enumerate() {
        let obs = &sens.obs[j];
        let mut jeta = 0.0;
        for k in 0..n_eta {
            jmat[(i, k)] = obs.df_deta[k];
            jeta += obs.df_deta[k] * eta_hat[k];
        }
        rho[i] = subject.observations[j] - (obs.f - jeta);
        let cmt = subject.obs_cmts[j];
        let f0act = sens0.obs[j].f;
        let r = model.error_spec.variance_at(cmt, f0act, sigma);
        if !(r.is_finite() && r > 0.0) {
            return None;
        }
        r0[i] = r;
        d0[i] = model.error_spec.dvar_df(cmt, f0act, sigma);
    }

    // Censored rows: `−logΦ(z)`, z = (LLOQ − f(η̂))/√R⁰. Precompute the inverse
    // Mills ratio h and the population variance / its f-derivative per row.
    struct Cens {
        j: usize,
        resid: f64, // LLOQ − f(η̂)
        w: f64,     // √R⁰
        h: f64,     // φ(z)/Φ(z)
        d0: f64,    // ∂R⁰/∂f at f(η=0)
        f0act: f64, // f(η=0)
        cmt: usize,
    }
    let mut cens: Vec<Cens> = Vec::new();
    if m3 {
        for j in 0..n_obs {
            if subject.cens.get(j).copied().unwrap_or(0) == 0 {
                continue;
            }
            let cmt = subject.obs_cmts[j];
            let f0act = sens0.obs[j].f;
            let r0c = model.error_spec.variance_at(cmt, f0act, sigma);
            if !(r0c.is_finite() && r0c > 0.0) {
                return None;
            }
            let w = r0c.sqrt();
            let resid = subject.observations[j] - sens.obs[j].f;
            cens.push(Cens {
                j,
                resid,
                w,
                h: inv_mills(resid / w),
                d0: model.error_spec.dvar_df(cmt, f0act, sigma),
                f0act,
                cmt,
            });
        }
    }

    // R̃ = J Ω Jᵀ + diag(R⁰) over quant rows; u = R̃⁻¹ ρ; ΩJᵀ reused throughout.
    let jo = &jmat * omega; // J Ω
    let mut rtilde = &jo * jmat.transpose();
    for i in 0..nq {
        rtilde[(i, i)] += r0[i];
    }
    let rtilde_inv = rtilde.cholesky()?.inverse();
    let u = &rtilde_inv * &rho;
    let ojt = omega * jmat.transpose(); // Ω Jᵀ (n_eta×nq)

    let n_theta = params.theta.len();
    let n_sigma = sigma.len();
    let mut fixed = vec![0.0f64; x.len()];

    // θ (fixed η̂): SB part over quant rows + censored `−logΦ` θ-gradient.
    //   SB: u·Qₘ + tr(R̃⁻¹EₘΩJᵀ) − u·(EₘΩJᵀu) + ½Σ ∂R⁰/∂θ (R̃⁻¹ᵢᵢ − u²ᵢ).
    //   censored: h·[ b̂ⱼₘ/w + (LLOQ−f̂)·∂R⁰/∂θ /(2w³) ], ∂R⁰/∂θ = d⁰·∂f(η=0)/∂θ.
    for m in 0..n_theta {
        let mut qm = DVector::<f64>::zeros(nq);
        let mut em = DMatrix::<f64>::zeros(nq, n_eta);
        let mut dvar = 0.0;
        for (i, &j) in quant.iter().enumerate() {
            let obs = &sens.obs[j];
            let mut bjeta = 0.0;
            for l in 0..n_eta {
                let bjl_m = obs.d2f_deta_dtheta[l * n_theta + m];
                em[(i, l)] = bjl_m;
                bjeta += bjl_m * eta_hat[l];
            }
            qm[i] = -obs.df_dtheta[m] + bjeta;
            let dr0 = d0[i] * sens0.obs[j].df_dtheta[m];
            dvar += dr0 * (rtilde_inv[(i, i)] - u[i] * u[i]);
        }
        let emojt = &em * &ojt;
        let tr = (&rtilde_inv * &emojt).trace();
        let uemu = u.dot(&(&emojt * &u));
        let mut nat = u.dot(&qm) + tr - uemu + 0.5 * dvar;
        for c in &cens {
            let bhat = sens.obs[c.j].df_dtheta[m];
            let dr0 = c.d0 * sens0.obs[c.j].df_dtheta[m];
            nat += c.h * (bhat / c.w + c.resid * dr0 / (2.0 * c.w * c.w * c.w));
        }
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        fixed[m] = nat * dtheta_dx;
    }

    // Ω (fixed η̂, packed Cholesky-L): SB over quant rows. The censored term has no
    // direct Ω-gradient (R⁰ and f(η̂) do not depend on Ω); it enters only via dη̂/dx.
    let l = &params.omega.chol;
    let jl = &jmat * l;
    let entries: Vec<(usize, usize)> = if params.omega.diagonal {
        (0..n_eta).map(|i| (i, i)).collect()
    } else {
        let mut e = Vec::new();
        for col in 0..n_eta {
            for r in col..n_eta {
                e.push((r, col));
            }
        }
        e
    };
    let omega_start = n_theta;
    for (ko, &(row, col)) in entries.iter().enumerate() {
        let jr = jmat.column(row);
        let jv = jl.column(col);
        let rinv_jr = &rtilde_inv * jr;
        let fixed_l = jv.dot(&rinv_jr) - jr.dot(&u) * jv.dot(&u);
        let chain = if row == col { l[(row, row)] } else { 1.0 };
        fixed[omega_start + ko] = fixed_l * chain;
    }

    // σ (fixed η̂): SB part over quant + censored. ∂R⁰/∂σ by central FD of the
    // closed-form variance at f(η=0) — works for FOCE here and FOCEI in sigma_block.
    //   censored: h·(LLOQ−f̂)·∂R⁰/∂σ /(2w³).
    let sigma_start = omega_start + entries.len();
    for k in 0..n_sigma {
        let hsig = sigma_fd_step(sigma[k]);
        let mut sp = sigma.clone();
        sp[k] += hsig;
        let mut sm = sigma.clone();
        sm[k] -= hsig;
        let mut nat = 0.0;
        for (i, &j) in quant.iter().enumerate() {
            let cmt = subject.obs_cmts[j];
            let f0act = sens0.obs[j].f;
            let dr0 = (model.error_spec.variance_at(cmt, f0act, &sp)
                - model.error_spec.variance_at(cmt, f0act, &sm))
                / (2.0 * hsig);
            nat += 0.5 * dr0 * (rtilde_inv[(i, i)] - u[i] * u[i]);
        }
        for c in &cens {
            let dr0 = (model.error_spec.variance_at(c.cmt, c.f0act, &sp)
                - model.error_spec.variance_at(c.cmt, c.f0act, &sm))
                / (2.0 * hsig);
            nat += c.h * c.resid * dr0 / (2.0 * c.w * c.w * c.w);
        }
        fixed[sigma_start + k] = nat * sigma[k];
    }

    // Coupling c = ∂F/∂η̂: SB part over quant rows + censored (∂(−logΦ)/∂η̂ = h·â/w).
    //   SB: u·P_k + tr(R̃⁻¹ Dk ΩJᵀ) − u·(Dk ΩJᵀ u),  P_k[i]=(Aⱼη̂)_k, Dk[i,l]=Aⱼ[k,l].
    let mut coupling = DVector::<f64>::zeros(n_eta);
    for k in 0..n_eta {
        let mut pk = DVector::<f64>::zeros(nq);
        let mut dk = DMatrix::<f64>::zeros(nq, n_eta);
        for (i, &j) in quant.iter().enumerate() {
            let obs = &sens.obs[j];
            let mut s = 0.0;
            for l in 0..n_eta {
                let a_kl = obs.d2f_deta2[k * n_eta + l];
                s += a_kl * eta_hat[l];
                dk[(i, l)] = a_kl; // A symmetric: Aⱼ[l,k] = Aⱼ[k,l]
            }
            pk[i] = s;
        }
        let dkojt = &dk * &ojt;
        let tr = (&rtilde_inv * &dkojt).trace();
        let udku = u.dot(&(&dkojt * &u));
        let mut ck = u.dot(&pk) + tr - udku;
        for c in &cens {
            ck += c.h * sens.obs[c.j].df_deta[k] / c.w;
        }
        coupling[k] = ck;
    }

    // Total: dFᵢ/dx_k = ∂Fᵢ/∂x_k|_η̂ + c·(dη̂/dx_k). dη̂/dx is interaction-
    // independent (shared inner objective, M3-aware), so it is reused as-is.
    let eta_dx = subject_eta_dx(model, subject, template, x, eta_hat)?;
    let mut g = vec![0.0f64; x.len()];
    for k in 0..x.len() {
        g[k] = fixed[k] + coupling.dot(&eta_dx[k]);
    }
    Some(g)
}

/// The exact analytic **FOCE** population gradient `d(OFV)/dx = 2·Σᵢ dFᵢ/dx` in
/// packed space, or `None` if any subject is unsupported. Fixed coords zeroed.
pub fn population_gradient_sens_foce(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[DVector<f64>],
) -> Option<Vec<f64>> {
    let n = x.len();
    // Subject-parallel; see `population_gradient_sens` (PR #381 review #7).
    let per_subject: Vec<Vec<f64>> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            subject_packed_gradient_foce(model, subject, template, x, eta_hats[i].as_slice())
        })
        .collect::<Option<Vec<_>>>()?;
    let mut grad = vec![0.0f64; n];
    for gi in &per_subject {
        for k in 0..n {
            grad[k] += 2.0 * gi[k];
        }
    }
    let fixed = packed_fixed_mask(template);
    for k in 0..n {
        if fixed[k] {
            grad[k] = 0.0;
        }
    }
    Some(grad)
}

/// Lower-triangle packed-entry list for an Ω of dimension `n` (diagonal: `(i,i)`;
/// block: `(r,c)`, `c ≤ r`), matching `pack_params` order.
fn lower_tri_entries(n: usize, diagonal: bool) -> Vec<(usize, usize)> {
    if diagonal {
        (0..n).map(|i| (i, i)).collect()
    } else {
        let mut e = Vec::new();
        for c in 0..n {
            for r in c..n {
                e.push((r, c));
            }
        }
        e
    }
}

/// Block-diagonal Cholesky factor `L_Σb = blkdiag(L_bsv, L_iov × K)` of the IOV
/// prior `Σ_b = Ω_bsv ⊕ K·Ω_iov`.
fn block_chol_full(
    l_bsv: &DMatrix<f64>,
    l_iov: &DMatrix<f64>,
    k: usize,
    n_eta: usize,
    n_iov: usize,
) -> DMatrix<f64> {
    let n = n_eta + k * n_iov;
    let mut l = DMatrix::zeros(n, n);
    for r in 0..n_eta {
        for c in 0..n_eta {
            l[(r, c)] = l_bsv[(r, c)];
        }
    }
    for kk in 0..k {
        let off = n_eta + kk * n_iov;
        for r in 0..n_iov {
            for c in 0..n_iov {
                l[(off + r, off + c)] = l_iov[(r, c)];
            }
        }
    }
    l
}

/// EBE response `dη̂/dx` for an analytical **IOV** subject (FOCE coupling +
/// Eq. 48 predictor), over the stacked `[η_bsv, κ₁..κ_K]` with block-Ω. Mirrors
/// [`subject_eta_dx`] but the Ω coords split: BSV packed entries map to the
/// top-left Cholesky block; the shared κ-variance packed entries sum the response
/// across the K IOV Cholesky blocks. `None` outside the IOV-analytical scope.
pub fn subject_eta_dx_iov(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    stacked_eta_hat: &[f64],
) -> Option<Vec<DVector<f64>>> {
    let params = unpack_params(x, template);
    let sens = crate::sens::provider::subject_sensitivities_iov(
        model,
        subject,
        &params.theta,
        stacked_eta_hat,
    )?;
    let k = crate::stats::likelihood::split_obs_by_occasion(subject).len();
    let n_eta_bsv = model.n_eta;
    let n_iov = model.n_kappa;
    let n_st = n_eta_bsv + k * n_iov;
    if stacked_eta_hat.len() != n_st {
        return None;
    }
    let omega_iov = params.omega_iov.as_ref()?;
    let block = crate::stats::likelihood::build_block_diag_omega(
        &params.omega.matrix,
        &omega_iov.matrix,
        k,
    );
    let omega_inv = block.cholesky()?.inverse();
    let prep = prepare_stacked(
        model,
        subject,
        &params,
        &sens,
        n_st,
        omega_inv,
        stacked_eta_hat,
        None,
    )?;
    let n_theta = params.theta.len();
    let n_sigma = params.sigma.values.len();
    let mut out: Vec<DVector<f64>> = vec![DVector::zeros(n_st); x.len()];

    // θ coords.
    for m in 0..n_theta {
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        let mvec = mixed_eta_theta(&sens.obs, &prep.et, n_st, prep.n_obs, m, prep.ruv);
        out[m] = -(&prep.h_inner_inv * mvec) * dtheta_dx;
    }

    // Ω coords (per Cholesky entry of Σ_b, pre-chain response).
    let z = &prep.omega_inv * DVector::from_column_slice(stacked_eta_hat);
    let l_bsv = &params.omega.chol;
    let l_iov = &omega_iov.chol;
    let l_full = block_chol_full(l_bsv, l_iov, k, n_eta_bsv, n_iov);
    let m_l_response = |row: usize, col: usize| -> DVector<f64> {
        let v = l_full.column(col).into_owned();
        let vz = v.dot(&z);
        let oinv_v = &prep.omega_inv * &v;
        let oinv_col_row: DVector<f64> = prep.omega_inv.column(row).into_owned();
        let m_l = -(oinv_col_row * vz + oinv_v * z[row]);
        -(&prep.h_inner_inv * m_l)
    };
    let omega_start = n_theta;
    let bsv_entries = lower_tri_entries(n_eta_bsv, params.omega.diagonal);
    for (e, &(row, col)) in bsv_entries.iter().enumerate() {
        let chain = if row == col { l_bsv[(row, row)] } else { 1.0 };
        out[omega_start + e] = m_l_response(row, col) * chain;
    }
    let sigma_start = omega_start + bsv_entries.len();
    let iov_start = sigma_start + n_sigma;
    let iov_entries = lower_tri_entries(n_iov, omega_iov.diagonal);
    for (e, &(i, j)) in iov_entries.iter().enumerate() {
        let mut resp = DVector::zeros(n_st);
        for kk in 0..k {
            resp += m_l_response(n_eta_bsv + kk * n_iov + i, n_eta_bsv + kk * n_iov + j);
        }
        let chain = if i == j { l_iov[(i, i)] } else { 1.0 };
        out[iov_start + e] = resp * chain;
    }

    // σ coords (no M3 in IOV scope).
    let sigma = &params.sigma.values;
    for kk in 0..n_sigma {
        let h = sigma_fd_step(sigma[kk]);
        let mut sp = sigma.clone();
        sp[kk] += h;
        let mut sm = sigma.clone();
        sm[kk] -= h;
        let mut mvec = DVector::<f64>::zeros(n_st);
        for (j, obs) in sens.obs.iter().enumerate() {
            let cmt = subject.obs_cmts[j];
            let f = obs.f;
            let (r, d, eps) = (prep.et[j].r, prep.et[j].d, prep.et[j].eps);
            let r_sig = (model.error_spec.variance_at(cmt, f, &sp)
                - model.error_spec.variance_at(cmt, f, &sm))
                / (2.0 * h);
            let d_sig = (model.error_spec.dvar_df(cmt, f, &sp)
                - model.error_spec.dvar_df(cmt, f, &sm))
                / (2.0 * h);
            let inv_r = 1.0 / r;
            let inv_r2 = inv_r * inv_r;
            let inv_r3 = inv_r2 * inv_r;
            let dalpha = (2.0 * eps * inv_r2 + d * (2.0 * eps * eps - r) * inv_r3) * r_sig
                + ((r - eps * eps) * inv_r2) * d_sig;
            for m in 0..n_st {
                mvec[m] += 0.5 * dalpha * obs.df_deta[m];
            }
        }
        out[sigma_start + kk] = -(&prep.h_inner_inv * mvec) * sigma[kk];
    }

    Some(out)
}

/// The exact per-subject **FOCE** (non-interaction) packed gradient for an
/// analytical **IOV** subject, in `pack_params` order `[θ, Ω_bsv, σ, Ω_iov]`. The
/// Sheiner–Beal linearized marginal `½[ρᵀR̃⁻¹ρ + log|R̃|]`, `R̃ = J Σ_b Jᵀ + R⁰`,
/// over the stacked `J = ∂f/∂[η_bsv,κ]` and block-Ω `Σ_b`. The Ω blocks split the
/// per-Cholesky-entry SB gradient over `Σ_b`'s factor (BSV block direct; the K
/// IOV blocks summed for the shared κ-variance); the coupling `∂F/∂η̂` reuses
/// [`subject_eta_dx_iov`]. `None` outside the IOV-analytical scope.
pub fn subject_packed_gradient_foce_iov(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    stacked_eta_hat: &[f64],
) -> Option<Vec<f64>> {
    let params = unpack_params(x, template);
    let sens = crate::sens::provider::subject_sensitivities_iov(
        model,
        subject,
        &params.theta,
        stacked_eta_hat,
    )?;
    let k = crate::stats::likelihood::split_obs_by_occasion(subject).len();
    let n_eta_bsv = model.n_eta;
    let n_iov = model.n_kappa;
    let n_st = n_eta_bsv + k * n_iov;
    if stacked_eta_hat.len() != n_st {
        return None;
    }
    let n_obs = subject.observations.len();
    if n_obs == 0 {
        return None;
    }
    let zeros = vec![0.0f64; n_st];
    let sens0 =
        crate::sens::provider::subject_sensitivities_iov(model, subject, &params.theta, &zeros)?;
    if sens.obs.len() != n_obs || sens0.obs.len() != n_obs {
        return None;
    }
    let sigma = &params.sigma.values;
    let omega_iov = params.omega_iov.as_ref()?;
    let omega_full = crate::stats::likelihood::build_block_diag_omega(
        &params.omega.matrix,
        &omega_iov.matrix,
        k,
    );

    // J = ∂f/∂[η,κ], ρ = ε + J·b̂, R⁰ and d⁰ at f(all-zero).
    let mut jmat = DMatrix::<f64>::zeros(n_obs, n_st);
    let mut rho = DVector::<f64>::zeros(n_obs);
    let mut r0 = vec![0.0f64; n_obs];
    let mut d0 = vec![0.0f64; n_obs];
    for j in 0..n_obs {
        let obs = &sens.obs[j];
        let mut jeta = 0.0;
        for kk in 0..n_st {
            jmat[(j, kk)] = obs.df_deta[kk];
            jeta += obs.df_deta[kk] * stacked_eta_hat[kk];
        }
        rho[j] = subject.observations[j] - (obs.f - jeta);
        let cmt = subject.obs_cmts[j];
        let f0act = sens0.obs[j].f;
        let r = model.error_spec.variance_at(cmt, f0act, sigma);
        if !(r.is_finite() && r > 0.0) {
            return None;
        }
        r0[j] = r;
        d0[j] = model.error_spec.dvar_df(cmt, f0act, sigma);
    }

    let jo = &jmat * &omega_full;
    let mut rtilde = &jo * jmat.transpose();
    for j in 0..n_obs {
        rtilde[(j, j)] += r0[j];
    }
    let rtilde_inv = rtilde.cholesky()?.inverse();
    let u = &rtilde_inv * &rho;
    let ojt = &omega_full * jmat.transpose();

    let n_theta = params.theta.len();
    let n_sigma = sigma.len();
    let mut fixed = vec![0.0f64; x.len()];

    // θ (fixed η̂).
    for m in 0..n_theta {
        let mut qm = DVector::<f64>::zeros(n_obs);
        let mut em = DMatrix::<f64>::zeros(n_obs, n_st);
        let mut dvar = 0.0;
        for j in 0..n_obs {
            let obs = &sens.obs[j];
            let mut bjeta = 0.0;
            for l in 0..n_st {
                let bjl = obs.d2f_deta_dtheta[l * n_theta + m];
                em[(j, l)] = bjl;
                bjeta += bjl * stacked_eta_hat[l];
            }
            qm[j] = -obs.df_dtheta[m] + bjeta;
            let dr0 = d0[j] * sens0.obs[j].df_dtheta[m];
            dvar += dr0 * (rtilde_inv[(j, j)] - u[j] * u[j]);
        }
        let emojt = &em * &ojt;
        let tr = (&rtilde_inv * &emojt).trace();
        let uemu = u.dot(&(&emojt * &u));
        let nat = u.dot(&qm) + tr - uemu + 0.5 * dvar;
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        fixed[m] = nat * dtheta_dx;
    }

    // Ω (fixed η̂): per Cholesky entry of Σ_b, BSV direct + K-summed IOV.
    let l_bsv = &params.omega.chol;
    let l_iov = &omega_iov.chol;
    let l_full = block_chol_full(l_bsv, l_iov, k, n_eta_bsv, n_iov);
    let jl = &jmat * &l_full;
    let entry_grad = |row: usize, col: usize| -> f64 {
        let jr = jmat.column(row);
        let jv = jl.column(col);
        let rinv_jr = &rtilde_inv * jr;
        jv.dot(&rinv_jr) - jr.dot(&u) * jv.dot(&u)
    };
    let omega_start = n_theta;
    let bsv_entries = lower_tri_entries(n_eta_bsv, params.omega.diagonal);
    for (e, &(row, col)) in bsv_entries.iter().enumerate() {
        let chain = if row == col { l_bsv[(row, row)] } else { 1.0 };
        fixed[omega_start + e] = entry_grad(row, col) * chain;
    }
    let sigma_start = omega_start + bsv_entries.len();

    // σ (fixed η̂).
    for kk in 0..n_sigma {
        let hsig = sigma_fd_step(sigma[kk]);
        let mut sp = sigma.clone();
        sp[kk] += hsig;
        let mut sm = sigma.clone();
        sm[kk] -= hsig;
        let mut nat = 0.0;
        for j in 0..n_obs {
            let cmt = subject.obs_cmts[j];
            let f0act = sens0.obs[j].f;
            let dr0 = (model.error_spec.variance_at(cmt, f0act, &sp)
                - model.error_spec.variance_at(cmt, f0act, &sm))
                / (2.0 * hsig);
            nat += 0.5 * dr0 * (rtilde_inv[(j, j)] - u[j] * u[j]);
        }
        fixed[sigma_start + kk] = nat * sigma[kk];
    }
    let iov_start = sigma_start + n_sigma;
    let iov_entries = lower_tri_entries(n_iov, omega_iov.diagonal);
    for (e, &(i, j)) in iov_entries.iter().enumerate() {
        let mut raw = 0.0;
        for kk in 0..k {
            raw += entry_grad(n_eta_bsv + kk * n_iov + i, n_eta_bsv + kk * n_iov + j);
        }
        let chain = if i == j { l_iov[(i, i)] } else { 1.0 };
        fixed[iov_start + e] = raw * chain;
    }

    // Coupling c = ∂F/∂η̂ over the stacked random effects.
    let mut coupling = DVector::<f64>::zeros(n_st);
    for kk in 0..n_st {
        let mut pk = DVector::<f64>::zeros(n_obs);
        let mut dk = DMatrix::<f64>::zeros(n_obs, n_st);
        for j in 0..n_obs {
            let obs = &sens.obs[j];
            let mut s = 0.0;
            for l in 0..n_st {
                let a_kl = obs.d2f_deta2[kk * n_st + l];
                s += a_kl * stacked_eta_hat[l];
                dk[(j, l)] = a_kl;
            }
            pk[j] = s;
        }
        let dkojt = &dk * &ojt;
        let tr = (&rtilde_inv * &dkojt).trace();
        let udku = u.dot(&(&dkojt * &u));
        coupling[kk] = u.dot(&pk) + tr - udku;
    }

    let eta_dx = subject_eta_dx_iov(model, subject, template, x, stacked_eta_hat)?;
    let mut g = vec![0.0f64; x.len()];
    for kk in 0..x.len() {
        g[kk] = fixed[kk] + coupling.dot(&eta_dx[kk]);
    }
    Some(g)
}

/// Per-packed-coordinate EBE response `dη̂/dx_k` (each a length-`n_eta` vector),
/// for the Almquist Eq. 48 warm-start predictor. Same `H⁻¹·∂²lᵢ/∂η∂x` solves the
/// gradient already forms, chained natural→packed. `None` when unsupported.
pub fn subject_eta_dx(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
) -> Option<Vec<DVector<f64>>> {
    if subject.observations.is_empty() {
        return Some(vec![DVector::zeros(model.n_eta); x.len()]);
    }
    let params = unpack_params(x, template);
    let sens = subject_sensitivities(model, subject, &params.theta, eta_hat)?;
    let prep = prepare(model, subject, &params, &sens, eta_hat)?;
    let n_eta = prep.n_eta;
    let n_theta = params.theta.len();
    let n_sigma = params.sigma.values.len();
    let mut out: Vec<DVector<f64>> = vec![DVector::zeros(n_eta); x.len()];

    // θ coords: dη̂/dx = −H⁻¹ (∂²l/∂η∂θ · ∂θ/∂x).
    for m in 0..n_theta {
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        let mvec = mixed_eta_theta(&sens.obs, &prep.et, n_eta, prep.n_obs, m, prep.ruv);
        out[m] = -(&prep.h_inner_inv * mvec) * dtheta_dx;
    }

    // Ω coords: M_L = −Ω⁻¹(e_row·(v·z) + v·z_row), v = L[:,col]; ×L_kk for diag-log.
    let z = &prep.omega_inv * DVector::from_column_slice(eta_hat);
    let l = &params.omega.chol;
    let entries: Vec<(usize, usize)> = if params.omega.diagonal {
        (0..n_eta).map(|i| (i, i)).collect()
    } else {
        let mut e = Vec::new();
        for c in 0..n_eta {
            for r in c..n_eta {
                e.push((r, c));
            }
        }
        e
    };
    let omega_start = n_theta;
    for (ko, &(row, col)) in entries.iter().enumerate() {
        let v = DVector::from_iterator(n_eta, (0..n_eta).map(|r| l[(r, col)]));
        let vz = v.dot(&z);
        let oinv_v = &prep.omega_inv * &v;
        let oinv_col_row: DVector<f64> = prep.omega_inv.column(row).into_owned();
        let m_l = -(oinv_col_row * vz + oinv_v * z[row]);
        let chain = if row == col { l[(row, row)] } else { 1.0 };
        out[omega_start + ko] = -(&prep.h_inner_inv * m_l) * chain;
    }

    // σ coords: M_σ = ½ Σⱼ ∂αⱼ/∂σ · aⱼ (∂R/∂σ,∂d/∂σ by FD of closed form); ×σ.
    let sigma_start = omega_start + entries.len();
    let sigma = &params.sigma.values;
    for k in 0..n_sigma {
        let h = sigma_fd_step(sigma[k]);
        let mut sp = sigma.clone();
        sp[k] += h;
        let mut sm = sigma.clone();
        sm[k] -= h;
        let mut mvec = DVector::<f64>::zeros(n_eta);
        for (j, obs) in sens.obs.iter().enumerate() {
            let cmt = subject.obs_cmts[j];
            let f = obs.f;
            // M3 censored inner term uses the conditional variance R(f(η̂)); its
            // `½∂α/∂σ = ∂g1/∂σ` by central FD of the closed-form censored scalar.
            if prep.censored.get(j).copied().unwrap_or(false) {
                let y = subject.observations[j];
                let (g1p, _) = m3_censored_scalars(
                    y,
                    f,
                    model.error_spec.variance_at(cmt, f, &sp),
                    model.error_spec.dvar_df(cmt, f, &sp),
                    model.error_spec.d2var_df2(cmt, &sp),
                );
                let (g1m, _) = m3_censored_scalars(
                    y,
                    f,
                    model.error_spec.variance_at(cmt, f, &sm),
                    model.error_spec.dvar_df(cmt, f, &sm),
                    model.error_spec.d2var_df2(cmt, &sm),
                );
                let dg1 = (g1p - g1m) / (2.0 * h);
                for m in 0..n_eta {
                    mvec[m] += dg1 * obs.df_deta[m];
                }
                continue;
            }
            let (r, d, eps) = (prep.et[j].r, prep.et[j].d, prep.et[j].eps);
            // `et.r`/`et.d` carry the `exp(2·η_ruv)` scale, so lift `∂R/∂σ`,`∂d/∂σ`
            // too (mirrors `sigma_block`); `ruv_scale == 1` when there is no ruv.
            let r_sig = prep.ruv_scale
                * (model.error_spec.variance_at(cmt, f, &sp)
                    - model.error_spec.variance_at(cmt, f, &sm))
                / (2.0 * h);
            let d_sig = prep.ruv_scale
                * (model.error_spec.dvar_df(cmt, f, &sp) - model.error_spec.dvar_df(cmt, f, &sm))
                / (2.0 * h);
            let inv_r = 1.0 / r;
            let inv_r2 = inv_r * inv_r;
            let inv_r3 = inv_r2 * inv_r;
            let dalpha = (2.0 * eps * inv_r2 + d * (2.0 * eps * eps - r) * inv_r3) * r_sig
                + ((r - eps * eps) * inv_r2) * d_sig;
            for m in 0..n_eta {
                mvec[m] += 0.5 * dalpha * obs.df_deta[m];
            }
            // Residual-eta row of M (#474): `M[ruv] = ∂(1−ε²/R)/∂σ = ε²/R²·Rσ`.
            if let Some(rr) = prep.ruv {
                mvec[rr] += eps * eps * inv_r2 * r_sig;
            }
        }
        out[sigma_start + k] = -(&prep.h_inner_inv * mvec) * sigma[k];
    }

    Some(out)
}

/// Per-subject `dη̂/dx` Jacobians for the whole population, or `None` if any
/// subject is unsupported.
pub fn population_eta_dx(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[DVector<f64>],
) -> Option<Vec<Vec<DVector<f64>>>> {
    population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, s)| subject_eta_dx(model, s, template, x, eta_hats[i].as_slice()))
        .collect()
}

/// Almquist Eq. 48 warm-start: `η⁰ᵢ = η̂ᵢ + Σₖ (dη̂ᵢ/dx_k)·(x_new−x_prev)_k`.
pub fn predict_warm_etas(
    prev_etas: &[DVector<f64>],
    jacs: &[Vec<DVector<f64>>],
    x_prev: &[f64],
    x_new: &[f64],
) -> Vec<DVector<f64>> {
    // Cap on the L2 norm of a single predicted η warm-start step. The inner solve
    // re-refines from the warm start, so this only needs to keep it inside a sane
    // region: on a large or ill-conditioned outer step the linear Eq.48
    // extrapolation can overshoot the basin, and if the inner BFGS then hits
    // max_iter it can land at a different mode, perturbing the reported OFV.
    // η live on the O(1) random-effects scale, so ~2 (a few IIV SDs) rarely binds
    // on a normal step but blocks a runaway one. PR #381 review finding #8.
    const MAX_PREDICT_STEP_NORM: f64 = 2.0;
    prev_etas
        .iter()
        .zip(jacs.iter())
        .map(|(eta, jac)| {
            let mut step = DVector::zeros(eta.len());
            for (k, jk) in jac.iter().enumerate() {
                let dx = x_new[k] - x_prev[k];
                if dx != 0.0 {
                    step += jk * dx;
                }
            }
            let norm = step.norm();
            if norm > MAX_PREDICT_STEP_NORM {
                step *= MAX_PREDICT_STEP_NORM / norm;
            }
            eta + step
        })
        .collect()
}

/// `M[:,m] = ∂²lᵢ/∂η∂θₘ = ½ Σⱼ (α'ⱼ bⱼₘ aⱼ + αⱼ Bⱼ[:,m])`. With IIV-on-RUV the
/// residual-eta row is `M[ruv,m] = Σⱼ κⱼ bⱼₘ`, `κⱼ = ∂(1−ε²/R)/∂f = 2ε/R + ε²d/R²`
/// (the `f`-derivative of the residual-eta data gradient; `a_{ruv}=B_{ruv}=0`, so
/// the main loop leaves that row at zero).
fn mixed_eta_theta(
    obs: &[ObsSens],
    et: &[ErrTerms],
    n_eta: usize,
    n_obs: usize,
    m: usize,
    ruv: Option<usize>,
) -> DVector<f64> {
    let n_theta_stride = obs[0].df_dtheta.len();
    let mut mk = DVector::zeros(n_eta);
    for j in 0..n_obs {
        let bjm = obs[j].df_dtheta[m];
        if let Some(rr) = ruv {
            mk[rr] += ruv_kappa(et[j].eps, et[j].r, et[j].d) * bjm;
        }
        for k in 0..n_eta {
            let b2 = obs[j].d2f_deta_dtheta[k * n_theta_stride + m];
            mk[k] += 0.5 * (et[j].alpha_p * bjm * obs[j].df_deta[k] + et[j].alpha * b2);
        }
    }
    mk
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::inner_optimizer::find_ebe;
    use crate::estimation::parameterization::pack_params;
    use crate::parser::model_parser::parse_model_string;
    use crate::stats::likelihood::{foce_subject_nll, foce_subject_nll_interaction};
    use crate::types::{DoseEvent, OmegaMatrix, Subject};
    use std::collections::HashMap;

    const TWOCPT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV1(30.0, 3.0, 300.0)
  theta TVQ(2.0, 0.1, 20.0)
  theta TVV2(50.0, 5.0, 500.0)
  theta TVKA(1.0, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk two_cpt_oral(cl=CL, v=V1, q=Q, v2=V2, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    const THREECPT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  theta TVKA(1.0, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    fn params_with_omega(model: &CompiledModel, theta: &[f64], vars: &[f64]) -> ModelParameters {
        let mut p = model.default_params.clone();
        p.theta = theta.to_vec();
        p.omega = OmegaMatrix::from_diagonal(vars, model.eta_names.clone());
        p
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

    fn subject_with_obs(model: &CompiledModel, theta: &[f64], times: &[f64]) -> Subject {
        // Build observations from the model at a reference eta so residuals are
        // realistic and nonzero (gradient identity holds at any obs).
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
        // Perturb by a fixed multiplicative factor so ε ≠ 0.
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        subject
    }

    /// A dosing-only subject (dose rows, no DV) contributes zero to the FOCE/FOCEI
    /// marginal objective: with no data rows, `log|Ω| + log|Ω⁻¹|` cancels at
    /// `η̂ = 0`. It should therefore return a zero analytic gradient instead of
    /// forcing the whole population gradient onto the FD fallback.
    #[test]
    fn zero_observation_subject_returns_zero_gradient() {
        let model = parse_model_string(TWOCPT).expect("parse");
        let template = model.default_params.clone();
        let theta = template.theta.clone();
        let mut subject = subject_with_obs(&model, &theta, &[2.0]); // build then empty out
        subject.obs_times.clear();
        subject.observations.clear();
        subject.obs_cmts.clear();
        subject.cens.clear();
        subject.occasions.clear();
        assert!(subject.observations.is_empty());

        let x = pack_params(&template);
        let eta_hat = vec![0.0; model.n_eta];
        let packed = subject_packed_gradient(&model, &subject, &template, &x, &eta_hat)
            .expect("dosing-only subject has zero packed gradient");
        assert_eq!(packed, vec![0.0; x.len()]);

        let theta_grad = subject_theta_gradient(&model, &subject, &template, &eta_hat)
            .expect("dosing-only subject has zero theta gradient");
        assert_eq!(theta_grad, vec![0.0; model.n_theta]);

        let omega_grad = subject_omega_gradient(&model, &subject, &template, &eta_hat)
            .expect("dosing-only subject has zero omega gradient");
        assert_eq!(omega_grad, vec![0.0; model.n_eta]);

        let sigma_grad = subject_sigma_gradient(&model, &subject, &template, &eta_hat)
            .expect("dosing-only subject has zero sigma gradient");
        assert_eq!(sigma_grad, vec![0.0; template.sigma.values.len()]);

        let eta_dx = subject_eta_dx(&model, &subject, &template, &x, &eta_hat)
            .expect("dosing-only subject has zero EBE predictor");
        assert_eq!(eta_dx, vec![DVector::zeros(model.n_eta); x.len()]);
    }

    /// `sigma_fd_step` keeps the central-difference minus side `σ − h` strictly
    /// positive: an ordinary σ uses the full `1e-6·(1+|σ|)` step, but a σ at/below
    /// that step shrinks to `0.5·σ` so the minus evaluation never underflows the
    /// `variance_at` floor (PR #381 review #6).
    #[test]
    fn sigma_fd_step_keeps_minus_side_positive() {
        // Ordinary σ: unchanged full step (h ≪ σ).
        let sig = 0.2;
        let h = sigma_fd_step(sig);
        assert!((h - 1e-6 * (1.0 + sig)).abs() < 1e-18);
        assert!(sig - h > 0.0);
        // Near-zero σ: step shrinks to 0.5·σ, minus side stays positive.
        let tiny = 5e-7;
        let h_tiny = sigma_fd_step(tiny);
        assert_eq!(h_tiny, 0.5 * tiny);
        assert!(tiny - h_tiny > 0.0);
        // σ = 0 leaves the base step (degenerate; no positive side to protect).
        assert_eq!(sigma_fd_step(0.0), 1e-6);
    }

    /// Precisely locate η̂ via analytic Newton on the inner objective (exact
    /// gradient ½Σαⱼaⱼ + Ω⁻¹η and true Hessian H from the provider), so the
    /// marginal-NLL finite difference is not contaminated by inner-solver
    /// reconvergence noise. Warm-started from `find_ebe`.
    fn precise_ebe(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> Vec<f64> {
        let warm = find_ebe(model, subject, params, 80, 1e-10, None, None);
        let mut eta: Vec<f64> = warm.eta.iter().copied().collect();
        let n_eta = model.n_eta;
        let sigma = &params.sigma.values;
        let omega_inv = &params.omega.inv;
        for _ in 0..50 {
            let sens =
                crate::sens::provider::subject_sensitivities(model, subject, &params.theta, &eta)
                    .unwrap();
            let mut grad = nalgebra::DVector::<f64>::from_column_slice(
                &(omega_inv * nalgebra::DVector::from_column_slice(&eta))
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
            );
            let mut hess = omega_inv.clone();
            let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3);
            for (j, obs) in sens.obs.iter().enumerate() {
                let f = obs.f;
                let cmt = subject.obs_cmts[j];
                let r = model.error_spec.variance_at(cmt, f, sigma);
                let d = model.error_spec.dvar_df(cmt, f, sigma);
                let d2 = model.error_spec.d2var_df2(cmt, sigma);
                let y = subject.observations[j];
                // (g1, g2) = (∂L/∂f, ∂²L/∂f²): the censored `−logΦ` scalars for an
                // M3 BLOQ row, else the Gaussian `½α`, `½α'`.
                let (g1, g2) = if m3 && subject.cens.get(j).copied().unwrap_or(0) != 0 {
                    m3_censored_scalars(y, f, r, d, d2)
                } else {
                    let t = err_terms(r, d, d2, y - f);
                    (0.5 * t.alpha, 0.5 * t.alpha_p)
                };
                let a = obs.df_deta.as_slice();
                for k in 0..n_eta {
                    grad[k] += g1 * a[k];
                    for l in 0..n_eta {
                        hess[(k, l)] += g2 * a[k] * a[l] + g1 * obs.d2f_deta2[k * n_eta + l];
                    }
                }
            }
            let step = hess.cholesky().unwrap().solve(&grad);
            for k in 0..n_eta {
                eta[k] -= step[k];
            }
            if step.norm() < 1e-13 {
                break;
            }
        }
        eta
    }

    /// `∂f/∂η` Jacobian (row-major `n_obs × n_eta`) at `eta` via the light
    /// provider, falling back to the full provider's `df_deta` for models the light
    /// one doesn't cover (TV-covariates, `ExpressionScale`) so the reconverged-FD
    /// marginal harness works there too. Test-only — the full provider's first
    /// derivative is exact, so the FOCE linearization is identical.
    fn eta_jacobian_any(
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        eta: &[f64],
    ) -> Vec<f64> {
        if let Some(j) = crate::sens::provider::subject_eta_jacobian(model, subject, theta, eta) {
            return j;
        }
        let s = crate::sens::provider::subject_sensitivities(model, subject, theta, eta)
            .expect("provider supports subject");
        let mut j = Vec::with_capacity(s.obs.len() * model.n_eta);
        for o in &s.obs {
            j.extend_from_slice(&o.df_deta);
        }
        j
    }

    /// Per-subject Laplace NLL Fᵢ at a *given* η̂ (no reconvergence).
    fn marginal_nll_at(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
        eta: &[f64],
    ) -> f64 {
        let eta_v = nalgebra::DVector::from_column_slice(eta);
        let ipreds = crate::pk::compute_predictions_with_tv(model, subject, &params.theta, eta);
        let jac = eta_jacobian_any(model, subject, &params.theta, eta);
        let h_matrix =
            nalgebra::DMatrix::from_row_slice(subject.obs_times.len(), model.n_eta, &jac);
        foce_subject_nll_interaction(
            subject,
            &ipreds,
            &eta_v,
            &h_matrix,
            &params.omega,
            &params.sigma.values,
            &model.error_spec,
            model.bloq_method,
            &[],
            None,
            model.residual_error_eta,
            model.ruv_obs_mult(subject, &params.theta).as_deref(),
        )
    }

    /// Reconverged marginal NLL Fᵢ(θ) at the precisely-located EBE.
    fn marginal_nll(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> f64 {
        let eta = precise_ebe(model, subject, params);
        marginal_nll_at(model, subject, params, &eta)
    }

    /// Per-subject **FOCE** (non-interaction) marginal NLL at a given η̂ — ferx's
    /// Sheiner–Beal linearized objective via `foce_subject_nll(.., interaction=false)`.
    fn marginal_nll_foce_at(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
        eta: &[f64],
    ) -> f64 {
        let eta_v = nalgebra::DVector::from_column_slice(eta);
        let jac = eta_jacobian_any(model, subject, &params.theta, eta);
        let h_matrix =
            nalgebra::DMatrix::from_row_slice(subject.obs_times.len(), model.n_eta, &jac);
        foce_subject_nll(
            model,
            subject,
            &params.theta,
            &eta_v,
            &h_matrix,
            &params.omega,
            &params.sigma.values,
            false,
        )
    }

    /// Reconverged FOCE marginal NLL at the precisely-located (shared) EBE.
    fn marginal_nll_foce(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
    ) -> f64 {
        let eta = precise_ebe(model, subject, params);
        marginal_nll_foce_at(model, subject, params, &eta)
    }

    /// FOCE analog of [`run_population_packed_gradient_check`]: the analytic FOCE
    /// packed gradient must match the reconverged-FD of ferx's FOCE OFV.
    fn run_packed_check_foce(model: &CompiledModel, theta: &[f64]) {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;

        let s1 = subject_with_obs(model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let s2 = subject_with_obs(model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]);
        let pop = Population {
            subjects: vec![s1, s2],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(model, s, &params)))
            .collect();

        let analytic =
            population_gradient_sens_foce(model, &pop, &template, &x, &ehs).expect("supported");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| marginal_nll_foce(model, s, &p))
                .sum::<f64>()
        };
        let fd_at = |k: usize, h: f64| -> f64 {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            (ofv(&xp) - ofv(&xm)) / (2.0 * h)
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let f1 = fd_at(k, h);
            let f2 = fd_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 3e-3, epsilon = 1e-5);
        }
    }

    #[test]
    fn theta_gradient_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.22, 11.0, 1.4];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let subject = subject_with_obs(&model, &theta, &times);

        let mut params = model.default_params.clone();
        params.theta = theta.clone();

        // Precisely-located EBE at the base point.
        let eta_hat = precise_ebe(&model, &subject, &params);

        let analytic =
            subject_theta_gradient(&model, &subject, &params, &eta_hat).expect("supported");

        // Richardson-extrapolated reconverged central FD of the marginal NLL
        // (cancels the O(h²) truncation; EBE is located analytically so there is
        // no inner-solver noise floor).
        let fd_at = |m: usize, h: f64| -> f64 {
            let mut pp = params.clone();
            pp.theta[m] += h;
            let mut pm = params.clone();
            pm.theta[m] -= h;
            (marginal_nll(&model, &subject, &pp) - marginal_nll(&model, &subject, &pm)) / (2.0 * h)
        };
        let n_theta = theta.len();
        for m in 0..n_theta {
            let h = 1e-4 * (1.0 + theta[m].abs());
            let f1 = fd_at(m, h);
            let f2 = fd_at(m, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                "theta[{m}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[m],
                fd,
                (analytic[m] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[m], fd, max_relative = 1e-3, epsilon = 1e-6);
        }
    }

    #[test]
    fn omega_gradient_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.22, 11.0, 1.4];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let subject = subject_with_obs(&model, &theta, &times);

        let vars = vec![0.09, 0.04, 0.30];
        let params = params_with_omega(&model, &theta, &vars);
        let eta_hat = precise_ebe(&model, &subject, &params);

        let analytic =
            subject_omega_gradient(&model, &subject, &params, &eta_hat).expect("supported");

        // Richardson reconverged FD over each natural variance entry.
        let fd_at = |i: usize, h: f64| -> f64 {
            let mut vp = vars.clone();
            vp[i] += h;
            let mut vm = vars.clone();
            vm[i] -= h;
            let pp = params_with_omega(&model, &theta, &vp);
            let pm = params_with_omega(&model, &theta, &vm);
            (marginal_nll(&model, &subject, &pp) - marginal_nll(&model, &subject, &pm)) / (2.0 * h)
        };
        for i in 0..vars.len() {
            let h = 1e-4 * (1.0 + vars[i].abs());
            let f1 = fd_at(i, h);
            let f2 = fd_at(i, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "omega[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 1e-3, epsilon = 1e-6);
        }
    }

    #[test]
    fn sigma_gradient_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.22, 11.0, 1.4];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let subject = subject_with_obs(&model, &theta, &times);

        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let eta_hat = precise_ebe(&model, &subject, &params);

        let analytic =
            subject_sigma_gradient(&model, &subject, &params, &eta_hat).expect("supported");

        let sig0 = params.sigma.values.clone();
        let fd_at = |k: usize, h: f64| -> f64 {
            let mut pp = params.clone();
            pp.sigma.values[k] += h;
            let mut pm = params.clone();
            pm.sigma.values[k] -= h;
            (marginal_nll(&model, &subject, &pp) - marginal_nll(&model, &subject, &pm)) / (2.0 * h)
        };
        for k in 0..sig0.len() {
            let h = 1e-4 * (1.0 + sig0[k].abs());
            let f1 = fd_at(k, h);
            let f2 = fd_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "sigma[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 2e-3, epsilon = 1e-6);
        }
    }

    /// Warfarin oral with a dedicated residual-error eta (`iiv_on_ruv`, #474).
    /// `ETA_RUV` is the 4th declared omega (index 3) and is not used in any
    /// individual parameter.
    const WARFARIN_RUV: &str = r#"
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

    /// Subject for an `iiv_on_ruv` model: predictions are independent of η_ruv, so
    /// build realistic nonzero residuals from the structural etas and pad the eta
    /// vector for the (prediction-irrelevant) residual eta.
    fn ruv_subject(model: &CompiledModel, theta: &[f64], times: &[f64]) -> Subject {
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
        let eta_ref = [0.12, -0.08, 0.2, 0.0];
        let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, &eta_ref);
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        subject
    }

    /// Precise EBE for an `iiv_on_ruv` model: Newton on the *scaled* inner
    /// objective (residual variance × `exp(2·η_ruv)`, plus the residual-eta
    /// gradient `1−ε²/R` and Hessian `2ε²/R` / `κ a` terms), mirroring `prepare`'s
    /// `H` so the marginal FD is not contaminated by inner-solver noise.
    fn precise_ebe_ruv(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
    ) -> Vec<f64> {
        let warm = find_ebe(model, subject, params, 80, 1e-10, None, None);
        let mut eta: Vec<f64> = warm.eta.iter().copied().collect();
        let n_eta = model.n_eta;
        let rr = model.residual_error_eta.expect("ruv model");
        let sigma = &params.sigma.values;
        let omega_inv = &params.omega.inv;
        for _ in 0..50 {
            let s = (2.0 * eta[rr]).exp();
            let sens =
                crate::sens::provider::subject_sensitivities(model, subject, &params.theta, &eta)
                    .unwrap();
            let mut grad = omega_inv * DVector::from_column_slice(&eta);
            let mut hess = omega_inv.clone();
            for (j, obs) in sens.obs.iter().enumerate() {
                let f = obs.f;
                let cmt = subject.obs_cmts[j];
                let r = model.error_spec.variance_at(cmt, f, sigma) * s;
                let d = model.error_spec.dvar_df(cmt, f, sigma) * s;
                let d2 = model.error_spec.d2var_df2(cmt, sigma) * s;
                let y = subject.observations[j];
                let eps = y - f;
                let t = err_terms(r, d, d2, eps);
                let (g1, g2) = (0.5 * t.alpha, 0.5 * t.alpha_p);
                let a = obs.df_deta.as_slice();
                for k in 0..n_eta {
                    grad[k] += g1 * a[k];
                    for l in 0..n_eta {
                        hess[(k, l)] += g2 * a[k] * a[l] + g1 * obs.d2f_deta2[k * n_eta + l];
                    }
                }
                // Residual-eta gradient / Hessian (a_{ruv} = A_{·,ruv} = 0).
                grad[rr] += 1.0 - eps * eps / r;
                hess[(rr, rr)] += 2.0 * eps * eps / r;
                let kappa = 2.0 * eps / r + eps * eps * d / (r * r);
                for l in 0..n_eta {
                    if l == rr {
                        continue;
                    }
                    hess[(rr, l)] += kappa * a[l];
                    hess[(l, rr)] += kappa * a[l];
                }
            }
            let step = hess.cholesky().unwrap().solve(&grad);
            for k in 0..n_eta {
                eta[k] -= step[k];
            }
            if step.norm() < 1e-13 {
                break;
            }
        }
        eta
    }

    /// 2-cpt IV **user-ODE** model with IIV on residual error (`iiv_on_ruv`). Same
    /// structure as `TWOCPT_ODE_OUTER` plus a dedicated `ETA_RUV` omega — exercises
    /// the residual-eta assembly through the ODE Dual2 sensitivity provider (#474).
    const TWOCPT_ODE_RUV: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  omega ETA_RUV ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// Shared FD check for an `iiv_on_ruv` model: the analytic FOCEI population
    /// packed gradient must match the Richardson-extrapolated reconverged-FD of
    /// ferx's own scaled FOCEI marginal across every θ/Ω/σ coordinate — including
    /// the residual eta's Ω entry (#474). The OFV *value* it differentiates is
    /// independently NONMEM-validated (#413).
    fn run_ruv_packed_check(model: &CompiledModel, theta: &[f64]) {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;

        let s1 = ruv_subject(model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let s2 = ruv_subject(model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]);
        let pop = Population {
            subjects: vec![s1, s2],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe_ruv(model, s, &params)))
            .collect();

        let analytic =
            population_gradient_sens(model, &pop, &template, &x, &ehs).expect("ruv is analytic");

        // 2·Σᵢ Fᵢ at the reconverged (scaled) EBE — the production FOCEI OFV.
        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| {
                    let eta = precise_ebe_ruv(model, s, &p);
                    marginal_nll_at(model, s, &p, &eta)
                })
                .sum::<f64>()
        };
        let fd_at = |k: usize, h: f64| -> f64 {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            (ofv(&xp) - ofv(&xm)) / (2.0 * h)
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let f1 = fd_at(k, h);
            let f2 = fd_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                "x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 2e-3, epsilon = 1e-5);
        }
    }

    #[test]
    fn population_packed_gradient_iiv_on_ruv_matches_fd() {
        let model = parse_model_string(WARFARIN_RUV).expect("parse");
        assert_eq!(model.residual_error_eta, Some(3));
        run_ruv_packed_check(&model, &[0.22, 11.0, 1.4]);
    }

    /// The same residual-eta gradient must be exact on an **ODE** model: the
    /// assembly is provider-agnostic, so the ODE `Dual2` sensitivities feed the
    /// residual-eta `H̃`/`H`/`log|H̃|` terms exactly as the closed-form ones do
    /// (#474). Confirms ODE + `iiv_on_ruv` is analytic, not FD.
    #[test]
    fn population_packed_gradient_ode_iiv_on_ruv_matches_fd() {
        let model = parse_model_string(TWOCPT_ODE_RUV).expect("parse ODE ruv");
        assert_eq!(model.residual_error_eta, Some(2));
        assert!(
            crate::sens::provider::analytic_outer_gradient_available(&model),
            "ODE + iiv_on_ruv must route to the analytic outer gradient (#474)"
        );
        run_ruv_packed_check(&model, &[4.0, 12.0, 2.0, 25.0]);
    }

    /// LTBS (`log_additive`) + `iiv_on_ruv` on an ODE model. The provider applies
    /// the `g = ln(f)` chain to the sensitivities, so the residual-eta variance
    /// terms (additive `R = σ²` on the log scale, `d = 0`) feed the same provider-
    /// agnostic assembly — the analytic outer gradient must still match FD (#474).
    /// (The inner EBE keeps FD for LTBS by design; the outer gradient is analytic.)
    const TWOCPT_ODE_LTBS_RUV: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  omega ETA_RUV ~ 0.10
  sigma ADD_ERR ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ log_additive(ADD_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    #[test]
    fn population_packed_gradient_ode_ltbs_iiv_on_ruv_matches_fd() {
        let model = parse_model_string(TWOCPT_ODE_LTBS_RUV).expect("parse ODE LTBS ruv");
        assert!(model.log_transform, "log_additive must set LTBS");
        assert_eq!(model.residual_error_eta, Some(2));
        assert!(
            crate::sens::provider::analytic_outer_gradient_available(&model),
            "ODE + LTBS + iiv_on_ruv must route to the analytic outer gradient (#474)"
        );
        run_ruv_packed_check(&model, &[4.0, 12.0, 2.0, 25.0]);
    }

    #[test]
    fn eta_dx_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.22, 11.0, 1.4];
        let subject = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let mut template = model.default_params.clone();
        template.theta = theta.clone();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let eta_hat = precise_ebe(&model, &subject, &params);

        let jac = subject_eta_dx(&model, &subject, &template, &x, &eta_hat).expect("supported");
        let n_eta = model.n_eta;
        for k in 0..x.len() {
            let h = 1e-5 * (1.0 + x[k].abs());
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            let ep = precise_ebe(&model, &subject, &unpack_params(&xp, &template));
            let em = precise_ebe(&model, &subject, &unpack_params(&xm, &template));
            for l in 0..n_eta {
                let fd = (ep[l] - em[l]) / (2.0 * h);
                approx::assert_relative_eq!(jac[k][l], fd, max_relative = 2e-3, epsilon = 1e-6);
            }
        }
    }

    /// #474 regression: `subject_eta_dx` for an `iiv_on_ruv` model. The σ columns
    /// of `dη̂/dx` must carry the `exp(2·η_ruv)` scale and the residual-eta row of
    /// `M_σ`, matching FD of the (scaled) EBE — this guards the parity with
    /// `sigma_block` that an earlier revision broke (dropped `ruv_scale` + the
    /// residual row, silently wrong σ columns).
    #[test]
    fn eta_dx_matches_fd_iiv_on_ruv() {
        use crate::estimation::parameterization::pack_params;
        let model = parse_model_string(WARFARIN_RUV).expect("parse");
        let theta = vec![0.22, 11.0, 1.4];
        let subject = ruv_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let mut template = model.default_params.clone();
        template.theta = theta.clone();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let eta_hat = precise_ebe_ruv(&model, &subject, &params);

        let jac = subject_eta_dx(&model, &subject, &template, &x, &eta_hat).expect("supported");
        let n_eta = model.n_eta;
        for k in 0..x.len() {
            let h = 1e-5 * (1.0 + x[k].abs());
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            let ep = precise_ebe_ruv(&model, &subject, &unpack_params(&xp, &template));
            let em = precise_ebe_ruv(&model, &subject, &unpack_params(&xm, &template));
            for l in 0..n_eta {
                let fd = (ep[l] - em[l]) / (2.0 * h);
                approx::assert_relative_eq!(jac[k][l], fd, max_relative = 2e-3, epsilon = 1e-6);
            }
        }
    }

    fn run_population_packed_gradient_check(model: &CompiledModel, theta: &[f64]) {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;

        let s1 = subject_with_obs(model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let s2 = subject_with_obs(model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]);
        let pop = Population {
            subjects: vec![s1, s2],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);

        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(model, s, &params)))
            .collect();

        let analytic =
            population_gradient_sens(model, &pop, &template, &x, &ehs).expect("supported");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| marginal_nll(model, s, &p))
                .sum::<f64>()
        };
        let fd_at = |k: usize, h: f64| -> f64 {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            (ofv(&xp) - ofv(&xm)) / (2.0 * h)
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let f1 = fd_at(k, h);
            let f2 = fd_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 3e-3, epsilon = 1e-5);
        }
    }

    #[test]
    fn population_packed_gradient_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN).expect("parse");
        run_population_packed_gradient_check(&model, &[0.22, 11.0, 1.4]);
    }

    #[test]
    fn population_packed_gradient_2cpt_matches_fd() {
        let model = parse_model_string(TWOCPT).expect("parse");
        run_population_packed_gradient_check(&model, &[5.0, 30.0, 2.0, 50.0, 1.0]);
    }

    // 2-cpt IV **user-ODE** model (Form C readout `y = central/V1`), IIV on CL+V1.
    // Exercises the armed ODE sensitivity provider (#410) through the *full* outer
    // assembly: the Dual2 augmented-RK45 jet must flow through the θ/Ω/σ blocks
    // (incl. the EBE response) and match reconverged FD exactly as the analytical
    // PK models do. Tight ODE tolerances so the propagated derivative agrees with a
    // finite difference of the (separately integrated) f64 objective.
    const TWOCPT_ODE_OUTER: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// The armed ODE outer gradient (#410) must match reconverged Richardson FD of
    /// the FOCEI marginal — the end-to-end proof that flipping `ODE_SENS_ENABLED`
    /// feeds a *correct* θ/Ω/σ gradient through the shared assembly, not just that
    /// the per-observation provider matches production (the `ode_provider` tests).
    #[test]
    fn population_packed_gradient_ode_2cpt_matches_fd() {
        let model = parse_model_string(TWOCPT_ODE_OUTER).expect("parse ODE");
        assert!(
            crate::sens::provider::sens_supported(&model),
            "2-cpt IV ODE must be armed for the analytic outer gradient (#410)"
        );
        run_population_packed_gradient_check(&model, &[4.0, 12.0, 2.0, 25.0]);
    }

    // 1-cpt IV (log-normal CL/V) used by the EVID=3/4 reset gradient checks: the
    // provider rebuilds each observation from the doses in its current reset
    // segment, so a reset subject's `∂f/∂η`, `∂²f/∂η²`, `∂f/∂θ`, `∂²f/∂η∂θ` jet —
    // and therefore the assembled θ/Ω/σ packed gradient — must still match
    // reconverged FD with no special-casing in the outer assembly.
    const ONECPT_IV_RESET: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    /// Two IV-infusion occasions separated by an EVID=4 reset at t=120: occasion-2
    /// observations must rebuild from zero (no carryover across the reset). The
    /// observations are synthesised from the production predictor at a reference η
    /// so residuals are realistic and nonzero.
    fn reset_subject_outer(
        model: &CompiledModel,
        theta: &[f64],
        eta_ref: &[f64],
        id: &str,
    ) -> Subject {
        let obs_times = vec![2.0, 4.0, 8.0, 60.0, 122.0, 126.0, 150.0];
        let n = obs_times.len();
        let mut subject = Subject {
            id: id.to_string(),
            doses: vec![
                DoseEvent::new(0.0, 1000.0, 1, 200.0, false, 0.0),
                DoseEvent::new(120.0, 1000.0, 1, 200.0, false, 0.0),
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
            reset_times: vec![120.0],
            cens: vec![0; n],
            occasions: vec![1; n],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        assert!(subject.has_resets(), "fixture must carry a reset");
        let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, eta_ref);
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        subject
    }

    // 1-cpt IV user-ODE for the SS+reset regression: SS bolus (II=24) establishes
    // steady state, an EVID 3/4 reset at t=60 zeros the carryover, and a re-dose
    // restarts. Tight tolerances so the dual jet agrees with FD of the f64 objective.
    const ONECPT_IV_ODE_SS_RESET: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// Steady-state dosing **combined with an EVID 3/4 reset** is served analytically
    /// on the ODE path: the static walk declines SS, so the subject routes to the
    /// event-driven walk (`ode_tvcov_supported`), which admits SS + reset with no joint
    /// exclusion. The analytic outer gradient must match Richardson FD of the FOCEI
    /// marginal. (The closed-form path keeps this combination on FD — the analytical
    /// superposition gates SS+reset. Pins the 2026-06-26 audit finding for #486.)
    #[test]
    fn population_packed_gradient_ode_ss_reset_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::{DoseEvent, Population};

        let model = parse_model_string(ONECPT_IV_ODE_SS_RESET).expect("parse ODE SS+reset");
        assert!(model.is_ode_based(), "must be on the ODE path");
        let theta = [0.2, 10.0];
        let eta_ref = [0.1, -0.1];
        let obs_times = vec![2.0, 8.0, 20.0, 62.0, 70.0, 90.0];
        let n = obs_times.len();
        let mut s = Subject {
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 24.0),
                DoseEvent::new(60.0, 1000.0, 1, 0.0, false, 0.0),
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
            reset_times: vec![60.0],
            cens: vec![0; n],
            occasions: vec![1; n],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        assert!(
            s.has_resets() && s.doses.iter().any(|d| d.ss),
            "fixture is SS + reset"
        );
        // Routes to the event-driven walk (static walk declines SS), and that walk
        // admits SS + reset — the precondition for the analytic gradient below.
        assert!(
            crate::sens::ode_provider::ode_tvcov_supported(&model, &s),
            "ODE event-driven walk must admit SS + reset"
        );
        assert!(
            !crate::sens::ode_provider::ode_subject_supported(&model, &s),
            "static walk declines SS, so SS + reset must route to the event-driven walk"
        );

        let preds = crate::pk::compute_predictions_with_tv(&model, &s, &theta, &eta_ref);
        s.observations = preds.iter().map(|p| p * 0.85).collect();

        let pop = Population {
            subjects: vec![s],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
            .collect();
        let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
            .expect("ODE SS + reset must be served analytically");
        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| marginal_nll(&model, s, &p))
                .sum::<f64>()
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            let f1 = (ofv(&xp) - ofv(&xm)) / (2.0 * h);
            let mut xp2 = x.clone();
            xp2[k] += h / 2.0;
            let mut xm2 = x.clone();
            xm2[k] -= h / 2.0;
            let f2 = (ofv(&xp2) - ofv(&xm2)) / (2.0 * (h / 2.0));
            let fd = (4.0 * f2 - f1) / 3.0;
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 5e-3, epsilon = 1e-5);
        }
    }

    /// FOCEI and FOCE packed gradients for a population containing a reset-bearing
    /// subject must both match Richardson reconverged-FD of their respective
    /// marginal objectives. This is the outer-assembly counterpart to the
    /// provider-vs-production reset tests in `sens::provider`: it confirms the
    /// reset segment's jet flows correctly through the θ/Ω/σ blocks (incl. the EBE
    /// response) for both estimation methods.
    #[test]
    fn population_packed_gradient_reset_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;

        let model = parse_model_string(ONECPT_IV_RESET).expect("parse");
        let theta = [0.22, 11.0];
        let eta_ref = [0.12, -0.08];

        // One reset subject + one ordinary subject, so the population mixes both.
        let s_reset = reset_subject_outer(&model, &theta, &eta_ref, "reset");
        let s_plain = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let pop = Population {
            subjects: vec![s_reset, s_plain],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
            .collect();

        // Both FOCEI (Almquist Laplace) and FOCE (Sheiner–Beal) paths.
        for interaction in [true, false] {
            let analytic = if interaction {
                population_gradient_sens(&model, &pop, &template, &x, &ehs)
            } else {
                population_gradient_sens_foce(&model, &pop, &template, &x, &ehs)
            }
            .expect("reset subject supported by analytic gradient");

            let ofv = |xv: &[f64]| -> f64 {
                let p = unpack_params(xv, &template);
                2.0 * pop
                    .subjects
                    .iter()
                    .map(|s| {
                        if interaction {
                            marginal_nll(&model, s, &p)
                        } else {
                            marginal_nll_foce(&model, s, &p)
                        }
                    })
                    .sum::<f64>()
            };
            let fd_at = |k: usize, h: f64| -> f64 {
                let mut xp = x.clone();
                xp[k] += h;
                let mut xm = x.clone();
                xm[k] -= h;
                (ofv(&xp) - ofv(&xm)) / (2.0 * h)
            };
            for k in 0..x.len() {
                let h = 1e-4 * (1.0 + x[k].abs());
                let f1 = fd_at(k, h);
                let f2 = fd_at(k, h / 2.0);
                let fd = (4.0 * f2 - f1) / 3.0;
                eprintln!(
                    "interaction={interaction} x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                    analytic[k],
                    fd,
                    (analytic[k] - fd).abs() / fd.abs().max(1e-12)
                );
                approx::assert_relative_eq!(analytic[k], fd, max_relative = 3e-3, epsilon = 1e-5);
            }
        }
    }

    /// Out-of-scope sibling of [`reset_subject_outer`]: same two IV-infusion
    /// occasions split by an EVID=4 reset, but the doses are **steady-state**.
    /// SS + reset is outside the analytic provider's scope (SS assumes an infinite
    /// periodic history a mid-record reset contradicts), so it returns `None` from
    /// `subject_sensitivities` while production still predicts it via the
    /// event-driven `f64` walk.
    fn ss_reset_subject_outer(
        model: &CompiledModel,
        theta: &[f64],
        eta_ref: &[f64],
        id: &str,
    ) -> Subject {
        let mut subject = reset_subject_outer(model, theta, eta_ref, id);
        subject.doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 24.0),
            DoseEvent::new(120.0, 1000.0, 1, 200.0, true, 24.0),
        ];
        assert!(
            subject.doses.iter().any(|d| d.ss) && subject.has_resets(),
            "fixture must be steady-state + reset"
        );
        let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, eta_ref);
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        subject
    }

    /// Regression for focei-slsqp-fixed-ebe-gradient-bias: a population mixing
    /// in-scope subjects with a single out-of-scope (SS+reset) subject must still
    /// yield the exact analytic gradient for the in-scope subjects, filling only
    /// the out-of-scope one with a reconverged per-subject FD. Before the fix one
    /// such subject forced `population_gradient_sens` to `None`, dropping the
    /// whole population onto the θ-only fixed-EBE gradient whose biased Ω/σ block
    /// left the variance components pinned at their start and stalled SLSQP/
    /// L-BFGS/MMA. The assembled `population_gradient_sens_mixed` must match
    /// reconverged-FD of the FOCEI OFV across every packed coordinate.
    #[test]
    fn mixed_gradient_with_out_of_scope_subject_matches_fd() {
        use crate::estimation::outer_optimizer::population_gradient_sens_mixed;
        use crate::estimation::parameterization::{compute_bounds, pack_params};
        use crate::types::{FitOptions, Population};

        let model = parse_model_string(ONECPT_IV_RESET).expect("parse");
        let theta = [0.22, 11.0];
        let eta_ref = [0.12, -0.08];

        // In-scope plain subject + an out-of-scope SS+reset subject.
        let s_plain = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let s_oos = ss_reset_subject_outer(&model, &theta, &eta_ref, "ss_reset");
        let pop = Population {
            subjects: vec![s_plain, s_oos],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);

        // EBE per subject: in-scope subjects use the analytic Newton polish
        // (`precise_ebe`); the out-of-scope SS+reset subject uses the production
        // inner solver (`find_ebe`), which `precise_ebe` can't because it unwraps
        // the analytic provider.
        let zeros = vec![0.0; model.n_eta];
        let in_scope = |s: &Subject| {
            crate::sens::provider::subject_sensitivities(&model, s, &params.theta, &zeros).is_some()
        };
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| {
                if in_scope(s) {
                    DVector::from_vec(precise_ebe(&model, s, &params))
                } else {
                    find_ebe(&model, s, &params, 200, 1e-12, None, None).eta
                }
            })
            .collect();

        // Pre-fix behaviour: the all-or-nothing analytic gradient declines the
        // whole population because subject 1 (SS+reset) is out of scope.
        assert!(
            population_gradient_sens(&model, &pop, &template, &x, &ehs).is_none(),
            "SS+reset subject must take the whole population out of the all-or-nothing path"
        );
        // The per-subject view keeps the in-scope subject analytic, only the
        // out-of-scope one `None`.
        let per_sub = per_subject_packed_gradients(&model, &pop, &template, &x, &ehs, true);
        assert!(per_sub[0].is_some(), "plain subject is in analytic scope");
        assert!(
            per_sub[1].is_none(),
            "SS+reset subject is out of analytic scope"
        );

        // The assembled mixed gradient (analytic in-scope + per-subject FD for the
        // out-of-scope subject) must match reconverged-FD of the FOCEI OFV.
        let options = FitOptions {
            interaction: true,
            ..Default::default()
        };
        let bounds = compute_bounds(&template);
        let mixed =
            population_gradient_sens_mixed(&x, &template, &model, &pop, &ehs, &bounds, &options);

        // FD reference, per subject mirroring the mixed assembly: in-scope
        // subjects via the analytic-EBE `marginal_nll`, the out-of-scope one via
        // the production reconverged EBE + `foce_subject_nll` (exactly what the
        // mixed FD fallback computes internally).
        let subj_marginal = |s: &Subject, p: &ModelParameters| -> f64 {
            if in_scope(s) {
                marginal_nll(&model, s, p)
            } else {
                let ebe = find_ebe(&model, s, p, 200, 1e-12, None, None);
                foce_subject_nll(
                    &model,
                    s,
                    &p.theta,
                    &ebe.eta,
                    &ebe.h_matrix,
                    &p.omega,
                    &p.sigma.values,
                    true,
                )
            }
        };
        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| subj_marginal(s, &p))
                .sum::<f64>()
        };
        let fd_at = |k: usize, h: f64| -> f64 {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            (ofv(&xp) - ofv(&xm)) / (2.0 * h)
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let f1 = fd_at(k, h);
            let f2 = fd_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            approx::assert_relative_eq!(mixed[k], fd, max_relative = 3e-3, epsilon = 1e-5);
        }
    }

    // 1-cpt oral with allometric WT-on-CL — the canonical time-varying covariate.
    // WT changes across a subject's records, so `CL = TVCL·(WT/70)^THETA_WT·exp(ETA_CL)`
    // switches mid-decay. The provider routes these to the event-driven Dual2 walk
    // and returns the standard `(η, θ)` jet, so the θ/Ω/σ packed gradient —
    // including the THETA_WT covariate coefficient and the EBE response — must match
    // reconverged FD with no special-casing in the outer assembly.
    const ONECPT_ORAL_TVCOV_OUTER: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    /// TV-cov subject: one dose with `WT` changing across observations (and an
    /// optional EVID=2 covariate breakpoint at `pk_only_times`). `dose` lets the
    /// caller pass a steady-state dose. Observations are synthesised from the
    /// production predictor at a reference η so residuals are realistic and nonzero.
    #[allow(clippy::too_many_arguments)]
    fn tvcov_subject_outer(
        model: &CompiledModel,
        theta: &[f64],
        eta_ref: &[f64],
        dose: DoseEvent,
        obs_times: &[f64],
        obs_wts: &[f64],
        pk_only_times: Vec<f64>,
        pk_only_wts: &[f64],
        id: &str,
    ) -> Subject {
        let n = obs_times.len();
        let wt_map = |w: f64| {
            let mut m = HashMap::new();
            m.insert("WT".to_string(), w);
            m
        };
        let mut subject = Subject {
            id: id.to_string(),
            doses: vec![dose],
            obs_times: obs_times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n],
            obs_cmts: vec![1; n],
            covariates: wt_map(obs_wts[0]),
            dose_covariates: vec![wt_map(obs_wts[0])],
            obs_covariates: obs_wts.iter().map(|&w| wt_map(w)).collect(),
            pk_only_times,
            pk_only_covariates: pk_only_wts.iter().map(|&w| wt_map(w)).collect(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        assert!(
            subject.has_tv_covariates(),
            "fixture must carry TV covariates"
        );
        let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, eta_ref);
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        subject
    }

    /// FOCEI and FOCE packed gradients for a population with a time-varying-covariate
    /// subject must both match Richardson reconverged-FD of their marginal
    /// objectives — the outer-assembly counterpart to the provider-vs-production
    /// TV-cov tests in `sens::provider`. One subject carries the covariate change
    /// across observations, the other carries an EVID=2 breakpoint between them, so
    /// both the covariate-θ chain and the `pk_only` walk flow through the θ/Ω/σ
    /// blocks (incl. the EBE response) for both estimation methods.
    #[test]
    fn population_packed_gradient_tvcov_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;

        let model = parse_model_string(ONECPT_ORAL_TVCOV_OUTER).expect("parse tvcov");
        let theta = [0.22, 11.0, 1.4, 0.7];
        let eta_ref = [0.12, -0.08, 0.2];

        let s_obs = tvcov_subject_outer(
            &model,
            &theta,
            &eta_ref,
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            &[1.0, 2.0, 4.0, 8.0, 24.0],
            &[70.0, 74.0, 82.0, 88.0, 95.0],
            Vec::new(),
            &[],
            "tvcov_obs",
        );
        let s_brk = tvcov_subject_outer(
            &model,
            &theta,
            &eta_ref,
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            &[1.0, 2.0, 6.0, 12.0],
            &[70.0, 70.0, 95.0, 95.0],
            vec![4.0],
            &[95.0],
            "tvcov_brk",
        );
        // Third subject: a steady-state (II=24) oral dose with WT changing across
        // observations, so the SS-equilibrated jet flows through the packed
        // gradient and its reconverged-FD reference too.
        let s_ss = tvcov_subject_outer(
            &model,
            &theta,
            &eta_ref,
            DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0),
            &[1.0, 4.0, 9.0, 15.0, 22.0],
            &[70.0, 76.0, 84.0, 90.0, 96.0],
            Vec::new(),
            &[],
            "tvcov_ss",
        );
        assert!(
            s_ss.doses.iter().any(|d| d.ss),
            "SS fixture must carry an SS dose"
        );
        let pop = Population {
            subjects: vec![s_obs, s_brk, s_ss],
            covariate_names: vec!["WT".into()],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
            .collect();

        for interaction in [true, false] {
            let analytic = if interaction {
                population_gradient_sens(&model, &pop, &template, &x, &ehs)
            } else {
                population_gradient_sens_foce(&model, &pop, &template, &x, &ehs)
            }
            .expect("TV-cov subject supported by analytic gradient");

            let ofv = |xv: &[f64]| -> f64 {
                let p = unpack_params(xv, &template);
                2.0 * pop
                    .subjects
                    .iter()
                    .map(|s| {
                        if interaction {
                            marginal_nll(&model, s, &p)
                        } else {
                            marginal_nll_foce(&model, s, &p)
                        }
                    })
                    .sum::<f64>()
            };
            let fd_at = |k: usize, h: f64| -> f64 {
                let mut xp = x.clone();
                xp[k] += h;
                let mut xm = x.clone();
                xm[k] -= h;
                (ofv(&xp) - ofv(&xm)) / (2.0 * h)
            };
            for k in 0..x.len() {
                let h = 1e-4 * (1.0 + x[k].abs());
                let f1 = fd_at(k, h);
                let f2 = fd_at(k, h / 2.0);
                let fd = (4.0 * f2 - f1) / 3.0;
                eprintln!(
                    "interaction={interaction} x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                    analytic[k],
                    fd,
                    (analytic[k] - fd).abs() / fd.abs().max(1e-12)
                );
                approx::assert_relative_eq!(analytic[k], fd, max_relative = 3e-3, epsilon = 1e-5);
            }
        }
    }

    // 1-cpt oral with a parameter-dependent central baseline (#524). The analytic
    // init impulse and its θ/η jet must flow through the packed FOCEI/FOCE
    // population gradient — i.e. the gradient that `gradient = auto` uses must
    // match Richardson FD of the marginal objective (`gradient = fd`), the
    // population-level analogue of the per-subject provider-vs-FD init test.
    const ONECPT_ORAL_INIT_OUTER: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVC0(5.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[initial_conditions]
  init(central) = TVC0 * V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    #[test]
    fn population_packed_gradient_init_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;

        let model = parse_model_string(ONECPT_ORAL_INIT_OUTER).expect("parse init outer");
        assert_eq!(model.analytical_init.len(), 1);
        assert!(
            crate::sens::provider::analytical_supported(&model),
            "init model must use the analytic outer gradient, not FD"
        );
        let theta = [0.22, 11.0, 1.4, 6.0];
        let eta_ref = [0.12, -0.08, 0.2];

        // Two plain (non-TV) subjects with a baseline-bearing oral dose; obs are
        // the init-aware prediction at eta_ref scaled down so EBEs are non-trivial.
        let make = |id: &str, times: &[f64]| -> Subject {
            let n = times.len();
            let mut s = Subject {
                id: id.to_string(),
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
            let preds = crate::pk::compute_predictions_with_tv(&model, &s, &theta, &eta_ref);
            s.observations = preds.iter().map(|p| p * 0.85).collect();
            s
        };
        let pop = Population {
            subjects: vec![
                make("init_a", &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
                make("init_b", &[1.0, 3.0, 6.0, 12.0]),
            ],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
            .collect();

        for interaction in [true, false] {
            let analytic = if interaction {
                population_gradient_sens(&model, &pop, &template, &x, &ehs)
            } else {
                population_gradient_sens_foce(&model, &pop, &template, &x, &ehs)
            }
            .expect("init subject supported by analytic gradient");

            let ofv = |xv: &[f64]| -> f64 {
                let p = unpack_params(xv, &template);
                2.0 * pop
                    .subjects
                    .iter()
                    .map(|s| {
                        if interaction {
                            marginal_nll(&model, s, &p)
                        } else {
                            marginal_nll_foce(&model, s, &p)
                        }
                    })
                    .sum::<f64>()
            };
            let fd_at = |k: usize, h: f64| -> f64 {
                let mut xp = x.clone();
                xp[k] += h;
                let mut xm = x.clone();
                xm[k] -= h;
                (ofv(&xp) - ofv(&xm)) / (2.0 * h)
            };
            for k in 0..x.len() {
                let h = 1e-4 * (1.0 + x[k].abs());
                let f1 = fd_at(k, h);
                let f2 = fd_at(k, h / 2.0);
                let fd = (4.0 * f2 - f1) / 3.0;
                approx::assert_relative_eq!(analytic[k], fd, max_relative = 3e-3, epsilon = 1e-5);
            }
        }
    }

    // 1-cpt oral with a log-normal dose lagtime (`LAGTIME = TVLAG·exp(ETA_LAG)`):
    // the lagtime θ (`TVLAG`) and ω (`ETA_LAG`) enter the packed gradient through
    // the provider's `∂f/∂θ` / `∂²f/∂η∂θ` for the lag slot, with no special-casing.
    const WARFARIN_LAG: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVLAG(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    /// Full packed-gradient check (8 params: 4 θ + 4 Ω-Cholesky + 1 σ) for a model
    /// with a differentiated dose lagtime, vs Richardson reconverged-FD of the
    /// marginal NLL. Confirms the lagtime axis flows through the Almquist assembly.
    #[test]
    fn population_packed_gradient_lagtime_matches_fd() {
        use crate::estimation::parameterization::{pack_params, unpack_params};
        use crate::types::Population;
        use std::collections::HashMap;

        let model = parse_model_string(WARFARIN_LAG).expect("parse lag");
        let theta = [0.22, 11.0, 1.4, 0.7];

        // Two subjects, observations built at a 4-component reference η (all obs
        // times comfortably past the lagged arrival so residuals are smooth).
        let build = |times: &[f64]| -> Subject {
            let n = times.len();
            let mut s = Subject {
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
            let eta_ref = [0.12, -0.08, 0.2, 0.1];
            let preds = crate::pk::compute_predictions_with_tv(&model, &s, &theta, &eta_ref);
            s.observations = preds.iter().map(|p| p * 0.85).collect();
            s
        };
        let pop = Population {
            subjects: vec![
                build(&[1.0, 2.0, 4.0, 8.0, 24.0]),
                build(&[1.5, 3.0, 6.0, 12.0, 36.0]),
            ],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
            .collect();

        let analytic =
            population_gradient_sens(&model, &pop, &template, &x, &ehs).expect("supported");
        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| marginal_nll(&model, s, &p))
                .sum::<f64>()
        };
        let fd_at = |k: usize, h: f64| -> f64 {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            (ofv(&xp) - ofv(&xm)) / (2.0 * h)
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let f1 = fd_at(k, h);
            let f2 = fd_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 3e-3, epsilon = 1e-5);
        }
    }

    /// The analytic FOCEI **M3** packed gradient (censored rows enter `prepare`'s
    /// M3 branch: data `−logΦ` + true-inner-Hessian, excluded from `H̃`) must match
    /// the reconverged-FD of ferx's M3 FOCEI objective (`foce_subject_nll_interaction`
    /// with `bloq_term`). Each subject carries both quantified and censored rows.
    #[test]
    fn population_packed_gradient_m3_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::{BloqMethod, Population};

        let mut model = parse_model_string(WARFARIN).expect("parse");
        model.bloq_method = BloqMethod::M3;
        let theta = [0.22, 11.0, 1.4];

        // Build subjects, then mark the last two observations of each as censored
        // (CENS=1, the obs cell carries the LLOQ) so every subject mixes quantified
        // and BLOQ rows. Leaves z moderate (LLOQ ≈ 0.85·f_ref), away from the tail.
        let mut s1 = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let mut s2 = subject_with_obs(&model, &theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]);
        for s in [&mut s1, &mut s2] {
            let n = s.observations.len();
            s.cens[n - 1] = 1;
            s.cens[n - 2] = 1;
        }
        assert!(s1.cens.iter().any(|&c| c != 0) && s2.cens.iter().any(|&c| c != 0));

        let pop = Population {
            subjects: vec![s1, s2],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
            .collect();

        let analytic =
            population_gradient_sens(&model, &pop, &template, &x, &ehs).expect("M3 supported");

        // marginal_nll uses foce_subject_nll_interaction with model.bloq_method = M3,
        // so the OFV carries the censored −2logΦ term; precise_ebe is M3-aware.
        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| marginal_nll(&model, s, &p))
                .sum::<f64>()
        };
        let fd_at = |k: usize, h: f64| -> f64 {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            (ofv(&xp) - ofv(&xm)) / (2.0 * h)
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let f1 = fd_at(k, h);
            let f2 = fd_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 5e-3, epsilon = 1e-5);
        }
    }

    // 1-cpt oral **user-ODE** model with M3 BLOQ, tight tolerances. ODE counterpart
    // of the closed-form `population_packed_gradient_m3_matches_fd`: the censored
    // `−logΦ` term enters `prepare`'s M3 branch on top of the ODE walk's `ObsSens`
    // (the same provider-agnostic assembly), proving non-IOV ODE+M3 is analytic on
    // the outer loop.
    const ONECPT_ODE_M3_OUTER: &str = r#"
[parameters]
  theta TVCL(0.2,  0.001, 10.0)
  theta TVV(10.0,  0.1,  500.0)
  theta TVKA(1.5,  0.01,  50.0)
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
[fit_options]
  method      = focei
  bloq_method = m3
  ode_reltol  = 1e-9
  ode_abstol  = 1e-11
"#;

    /// ODE counterpart of [`population_packed_gradient_m3_matches_fd`]: the analytic
    /// FOCEI M3 packed gradient assembled from the **event-driven ODE sensitivity
    /// walk** (censored rows enter `prepare`'s M3 branch) must match reconverged FD
    /// of the M3 FOCEI objective. Proves non-IOV ODE+M3 is analytic on the outer
    /// loop (the inner counterpart lives in `inner_optimizer.rs`).
    #[test]
    fn population_packed_gradient_ode_m3_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::{BloqMethod, Population};

        let model = parse_model_string(ONECPT_ODE_M3_OUTER).expect("parse ODE M3");
        assert!(matches!(model.bloq_method, BloqMethod::M3), "must be M3");
        assert!(model.is_ode_based(), "must be on the ODE path");
        let theta = [0.22, 11.0, 1.4];

        let mut s1 = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 8.0]);
        let mut s2 = subject_with_obs(&model, &theta, &[0.25, 1.5, 6.0, 12.0, 36.0]);
        for s in [&mut s1, &mut s2] {
            let n = s.observations.len();
            s.cens[n - 1] = 1;
            s.cens[n - 2] = 1;
        }
        assert!(s1.cens.iter().any(|&c| c != 0) && s2.cens.iter().any(|&c| c != 0));

        let pop = Population {
            subjects: vec![s1, s2],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
            .collect();

        let analytic =
            population_gradient_sens(&model, &pop, &template, &x, &ehs).expect("ODE M3 supported");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| marginal_nll(&model, s, &p))
                .sum::<f64>()
        };
        let fd_at = |k: usize, h: f64| -> f64 {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            (ofv(&xp) - ofv(&xm)) / (2.0 * h)
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let f1 = fd_at(k, h);
            let f2 = fd_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 5e-3, epsilon = 1e-5);
        }
    }

    /// The analytic **FOCE** (Sheiner–Beal, non-interaction) M3 packed gradient
    /// (censored rows excluded from R̃, added as `−logΦ((LLOQ−f̂)/√R⁰)` with the
    /// population variance) must match the reconverged-FD of ferx's FOCE-M3
    /// objective (`foce_subject_nll_standard` with the censored term).
    #[test]
    fn population_packed_gradient_m3_foce_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::{BloqMethod, Population};

        let mut model = parse_model_string(WARFARIN).expect("parse");
        model.bloq_method = BloqMethod::M3;
        let theta = [0.22, 11.0, 1.4];

        let mut s1 = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let mut s2 = subject_with_obs(&model, &theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]);
        for s in [&mut s1, &mut s2] {
            let n = s.observations.len();
            s.cens[n - 1] = 1;
            s.cens[n - 2] = 1;
        }

        let pop = Population {
            subjects: vec![s1, s2],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
            .collect();

        let analytic =
            population_gradient_sens_foce(&model, &pop, &template, &x, &ehs).expect("M3 FOCE");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| marginal_nll_foce(&model, s, &p))
                .sum::<f64>()
        };
        let fd_at = |k: usize, h: f64| -> f64 {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            (ofv(&xp) - ofv(&xm)) / (2.0 * h)
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let f1 = fd_at(k, h);
            let f2 = fd_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 5e-3, epsilon = 1e-5);
        }
    }

    #[test]
    fn population_packed_gradient_3cpt_matches_fd() {
        let model = parse_model_string(THREECPT).expect("parse");
        run_population_packed_gradient_check(&model, &[5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.0]);
    }

    // --- FOCE (non-interaction, Sheiner–Beal linearized marginal) ---

    #[test]
    fn population_packed_gradient_foce_matches_fd() {
        let model = parse_model_string(WARFARIN).expect("parse");
        run_packed_check_foce(&model, &[0.22, 11.0, 1.4]);
    }

    #[test]
    fn population_packed_gradient_foce_2cpt_matches_fd() {
        let model = parse_model_string(TWOCPT).expect("parse");
        run_packed_check_foce(&model, &[5.0, 30.0, 2.0, 50.0, 1.0]);
    }

    #[test]
    fn population_packed_gradient_foce_3cpt_matches_fd() {
        let model = parse_model_string(THREECPT).expect("parse");
        run_packed_check_foce(&model, &[5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.0]);
    }

    // --- Eq. 48 EBE warm-start predictor: is it correct & better than plain warm? ---

    /// The Eq. 48 predictor `η⁰ = η̂_prev + (dη̂/dx)·Δx` is a first-order Taylor
    /// extrapolation of the EBE as the packed parameters move x_prev → x_new. So
    /// against the *converged* EBE at x_new it must beat the plain warm-start
    /// (reuse η̂_prev): the prediction error is `O(‖Δx‖²)` while the warm-start
    /// error is `O(‖Δx‖)`. This walks several step sizes in a representative
    /// direction and checks (a) prediction strictly beats warm for small steps,
    /// and (b) the prediction/warm error ratio shrinks ∝ ‖Δx‖ (second order).
    #[test]
    fn eta_predictor_beats_warm_start() {
        use crate::estimation::parameterization::pack_params;

        let model = parse_model_string(TWOCPT).expect("parse");
        let theta = vec![5.0, 30.0, 2.0, 50.0, 1.0];
        let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let subjects = [
            subject_with_obs(&model, &theta, &times),
            subject_with_obs(&model, &theta, &[0.5, 1.5, 3.0, 6.0, 12.0, 36.0]),
        ];

        let mut template = model.default_params.clone();
        template.theta = theta.clone();
        let x0 = pack_params(&template);
        let n = x0.len();

        // A fixed, representative outer direction (unit-norm in packed space).
        let mut dir: Vec<f64> = (0..n).map(|k| 0.5 + 0.1 * k as f64).collect();
        let dnorm = dir.iter().map(|d| d * d).sum::<f64>().sqrt();
        for d in dir.iter_mut() {
            *d /= dnorm;
        }

        // Base EBEs and dη̂/dx at x0.
        let p0 = unpack_params(&x0, &template);
        let eta0: Vec<DVector<f64>> = subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &p0)))
            .collect();
        let jac: Vec<Vec<DVector<f64>>> = subjects
            .iter()
            .enumerate()
            .map(|(i, s)| subject_eta_dx(&model, s, &template, &x0, eta0[i].as_slice()).unwrap())
            .collect();

        eprintln!("  step    warm_err   pred_err   ratio");
        let mut prev_ratio: Option<f64> = None;
        for &s in &[0.20_f64, 0.10, 0.05, 0.025] {
            let x1: Vec<f64> = (0..n).map(|k| x0[k] + s * dir[k]).collect();
            let p1 = unpack_params(&x1, &template);

            let pred = predict_warm_etas(&eta0, &jac, &x0, &x1);

            let mut warm_err = 0.0;
            let mut pred_err = 0.0;
            for (i, subj) in subjects.iter().enumerate() {
                let eta1 = DVector::from_vec(precise_ebe(&model, subj, &p1));
                warm_err += (&eta0[i] - &eta1).norm();
                pred_err += (&pred[i] - &eta1).norm();
            }
            let ratio = pred_err / warm_err.max(1e-300);
            eprintln!("  {s:>5.3}  {warm_err:>9.2e}  {pred_err:>9.2e}  {ratio:>6.3}");

            // (a) the predictor must be a real improvement on warm-start.
            assert!(
                pred_err < 0.5 * warm_err,
                "predictor (err {pred_err:.3e}) should beat warm-start (err {warm_err:.3e}) at step {s}"
            );
            // (b) halving the step should shrink the ratio (second-order error).
            if let Some(pr) = prev_ratio {
                assert!(
                    ratio < pr + 1e-9,
                    "pred/warm ratio should not grow as the step shrinks ({ratio:.3} vs {pr:.3})"
                );
            }
            prev_ratio = Some(ratio);
        }
    }

    // --- IOV: analytic θ-gradient over the stacked (η_bsv, κ) with block-Ω ---

    const WARFARIN_IOV: &str = r#"
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
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    /// Two-occasion IOV subject (no washout — carryover spans the boundary), with
    /// observations synthesised from the model at a reference (η, κ) so residuals
    /// are realistic.
    fn iov_subject_outer(model: &CompiledModel, theta: &[f64]) -> Subject {
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
            cens: vec![0; n],
            occasions,
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        // Reference (η_bsv, κ_g0, κ_g1) → realistic ε ≠ 0.
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

    /// Precisely locate the joint IOV EBE by analytic Newton on the stacked inner
    /// objective (exact gradient ½Σαⱼaⱼ + Ω_block⁻¹b and true Hessian from the IOV
    /// provider), so the marginal FD is not contaminated by inner-solver
    /// reconvergence noise — the IOV analog of [`precise_ebe`]. Returns the stacked
    /// `b̂`, plus the `(η̂, κ̂, BSV H-matrix)` form `foce_subject_nll_iov` consumes
    /// (H-matrix = the provider's exact `∂f/∂η_bsv`).
    fn precise_ebe_iov(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
    ) -> (Vec<f64>, DVector<f64>, Vec<DVector<f64>>, DMatrix<f64>) {
        let k = crate::stats::likelihood::split_obs_by_occasion(subject).len();
        let n_eta = model.n_eta;
        let n_kappa = model.n_kappa;
        let n_st = n_eta + k * n_kappa;
        let warm = find_ebe(model, subject, params, 80, 1e-10, None, None);
        let mut stacked = vec![0.0; n_st];
        for i in 0..n_eta {
            stacked[i] = warm.eta[i];
        }
        for (g, kap) in warm.kappas.iter().enumerate() {
            for ki in 0..n_kappa {
                stacked[n_eta + g * n_kappa + ki] = kap[ki];
            }
        }
        let block = crate::stats::likelihood::build_block_diag_omega(
            &params.omega.matrix,
            &params.omega_iov.as_ref().unwrap().matrix,
            k,
        );
        let omega_inv = block.cholesky().unwrap().inverse();
        let sigma = &params.sigma.values;
        for _ in 0..50 {
            let sens = crate::sens::provider::subject_sensitivities_iov(
                model,
                subject,
                &params.theta,
                &stacked,
            )
            .unwrap();
            let mut g = &omega_inv * DVector::from_column_slice(&stacked);
            let mut h = omega_inv.clone();
            for (j, obs) in sens.obs.iter().enumerate() {
                let cmt = subject.obs_cmts[j];
                let f = obs.f;
                let r = model.error_spec.variance_at(cmt, f, sigma);
                let d = model.error_spec.dvar_df(cmt, f, sigma);
                let d2 = model.error_spec.d2var_df2(cmt, sigma);
                let t = err_terms(r, d, d2, subject.observations[j] - f);
                let a = &obs.df_deta;
                for kk in 0..n_st {
                    g[kk] += 0.5 * t.alpha * a[kk];
                    for ll in 0..n_st {
                        h[(kk, ll)] += 0.5
                            * (t.alpha_p * a[kk] * a[ll] + t.alpha * obs.d2f_deta2[kk * n_st + ll]);
                    }
                }
            }
            let step = h.cholesky().unwrap().solve(&g);
            for kk in 0..n_st {
                stacked[kk] -= step[kk];
            }
            if step.norm() < 1e-13 {
                break;
            }
        }
        let eta = DVector::from_column_slice(&stacked[..n_eta]);
        let kappas: Vec<DVector<f64>> = (0..k)
            .map(|gi| {
                DVector::from_column_slice(
                    &stacked[n_eta + gi * n_kappa..n_eta + (gi + 1) * n_kappa],
                )
            })
            .collect();
        let sens = crate::sens::provider::subject_sensitivities_iov(
            model,
            subject,
            &params.theta,
            &stacked,
        )
        .unwrap();
        let n_obs = subject.obs_times.len();
        let mut hm = DMatrix::zeros(n_obs, n_eta);
        for j in 0..n_obs {
            for c in 0..n_eta {
                hm[(j, c)] = sens.obs[j].df_deta[c];
            }
        }
        (stacked, eta, kappas, hm)
    }

    /// IOV marginal at the analytically-reconverged joint EBE for `params` (no
    /// inner-solver noise; the BSV H-matrix is the provider's exact Jacobian).
    /// `interaction = true` → FOCEI, `false` → FOCE (Sheiner–Beal).
    fn marginal_nll_iov_inter(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
        interaction: bool,
    ) -> f64 {
        let (_stacked, eta, kappas, hm) = precise_ebe_iov(model, subject, params);
        crate::stats::likelihood::foce_subject_nll_iov(
            model,
            subject,
            &params.theta,
            &eta,
            &hm,
            &params.omega,
            &params.sigma.values,
            interaction,
            &kappas,
            params.omega_iov.as_ref().expect("IOV model has omega_iov"),
        )
    }

    fn marginal_nll_iov(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> f64 {
        marginal_nll_iov_inter(model, subject, params, true)
    }

    /// The analytic IOV θ-gradient (paper-exact over the stacked η + block-Ω) must
    /// match the Richardson-extrapolated reconverged FD of the production IOV FOCEI
    /// marginal `foce_subject_nll_iov` — the same objective validated against NONMEM
    /// (`tests/warfarin_iov_nonmem.rs`, ferx ≈308.2 vs NONMEM 308.83). This closes
    /// the IOV outer-gradient θ block end-to-end against a NONMEM-grounded target.
    #[test]
    fn iov_theta_gradient_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_subject_outer(&model, &theta);

        // Joint EBE [η_bsv (3), κ_g0 (1), κ_g1 (1)], analytically reconverged.
        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);

        let analytic = subject_theta_gradient_iov(&model, &subject, &params, &stacked)
            .expect("IOV θ-gradient supported");

        let fd_at = |m: usize, h: f64| -> f64 {
            let mut pp = params.clone();
            pp.theta[m] += h;
            let mut pm = params.clone();
            pm.theta[m] -= h;
            (marginal_nll_iov(&model, &subject, &pp) - marginal_nll_iov(&model, &subject, &pm))
                / (2.0 * h)
        };
        for m in 0..theta.len() {
            let h = 1e-4 * (1.0 + theta[m].abs());
            let f1 = fd_at(m, h);
            let f2 = fd_at(m, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                "iov theta[{m}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[m],
                fd,
                (analytic[m] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[m], fd, max_relative = 3e-3, epsilon = 1e-5);
        }
    }

    /// The full analytic IOV **packed** gradient (`[θ, Ω_bsv, σ, Ω_iov]`, optimizer
    /// space) must match the Richardson reconverged FD of the production IOV FOCEI
    /// marginal over every packed coordinate — closing the Ω (incl. the shared
    /// κ-variance) and σ blocks against the NONMEM-grounded objective.
    #[test]
    fn iov_packed_gradient_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_subject_outer(&model, &theta);
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            .expect("IOV packed gradient supported");

        let f = |xx: &[f64]| -> f64 {
            let p = unpack_params(xx, &template);
            marginal_nll_iov(&model, &subject, &p)
        };
        for i in 0..x.len() {
            let h = 1e-4 * (1.0 + x[i].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut xp = x.clone();
                xp[i] += hh;
                let mut xm = x.clone();
                xm[i] -= hh;
                (f(&xp) - f(&xm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                "iov x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-9)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 2e-3, epsilon = 2e-5);
        }
    }

    /// The full analytic IOV **FOCE** (non-interaction) packed gradient must match
    /// the Richardson reconverged FD of the production IOV FOCE marginal
    /// (`foce_subject_nll_iov` with `interaction = false`, the Sheiner–Beal
    /// linearized objective) over every packed coordinate — the path
    /// `method = foce` (warfarin_iov's default) actually exercises.
    #[test]
    fn iov_packed_gradient_foce_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_subject_outer(&model, &theta);
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
            .expect("IOV FOCE packed gradient supported");

        let f = |xx: &[f64]| -> f64 {
            let p = unpack_params(xx, &template);
            marginal_nll_iov_inter(&model, &subject, &p, false)
        };
        for i in 0..x.len() {
            let h = 1e-4 * (1.0 + x[i].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut xp = x.clone();
                xp[i] += hh;
                let mut xm = x.clone();
                xm[i] -= hh;
                (f(&xp) - f(&xm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "iov foce x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-9)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 2e-3, epsilon = 2e-5);
        }
    }

    /// IOV with an EVID=4 washout reset at the occasion boundary: the same
    /// two-occasion subject as `iov_subject_outer`, but occasion 2 rebuilds from
    /// zero (no carryover). The full packed gradient — FOCEI **and** FOCE — must
    /// still match Richardson reconverged FD of the IOV marginal, confirming the
    /// reset jet flows through the stacked-η / block-Ω assembly unchanged.
    fn iov_subject_outer_reset(model: &CompiledModel, theta: &[f64]) -> Subject {
        let mut s = iov_subject_outer(model, theta);
        s.reset_times = vec![24.0];
        assert!(s.has_resets(), "fixture must carry a reset");
        // Re-synthesise observations through the reset-aware predict_iov so ε ≠ 0.
        let preds = crate::pk::predict_iov(
            model,
            &s,
            theta,
            &[0.12, -0.08, 0.2],
            &[vec![0.05], vec![-0.07]],
        );
        s.observations = preds.iter().map(|p| p * 0.85).collect();
        s
    }

    #[test]
    fn iov_packed_gradient_reset_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_subject_outer_reset(&model, &theta);
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);

        // FOCEI (Almquist Laplace) and FOCE (Sheiner–Beal) over the reset subject.
        for interaction in [true, false] {
            let analytic = if interaction {
                subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            } else {
                subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
            }
            .expect("IOV+reset packed gradient supported");

            let f = |xx: &[f64]| -> f64 {
                let p = unpack_params(xx, &template);
                marginal_nll_iov_inter(&model, &subject, &p, interaction)
            };
            for i in 0..x.len() {
                let h = 1e-4 * (1.0 + x[i].abs());
                let fd_at = |hh: f64| -> f64 {
                    let mut xp = x.clone();
                    xp[i] += hh;
                    let mut xm = x.clone();
                    xm[i] -= hh;
                    (f(&xp) - f(&xm)) / (2.0 * hh)
                };
                let f1 = fd_at(h);
                let f2 = fd_at(h / 2.0);
                let fd = (4.0 * f2 - f1) / 3.0; // Richardson
                eprintln!(
                    "iov reset interaction={interaction} x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                    analytic[i],
                    fd,
                    (analytic[i] - fd).abs() / fd.abs().max(1e-9)
                );
                approx::assert_relative_eq!(analytic[i], fd, max_relative = 2e-3, epsilon = 2e-5);
            }
        }
    }

    // --- IOV combined with a time-varying covariate ---

    /// IOV model that *also* carries a WT-on-CL covariate (`THETA_WT`), so a
    /// subject whose WT varies across records switches `CL` by both κ (occasion)
    /// and WT (covariate). θ = [TVCL, TVV, TVKA, THETA_WT].
    const WARFARIN_IOV_TVCOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    /// Two-occasion IOV subject carrying a WT covariate that varies across records
    /// (lighter in occasion 1, heavier in occasion 2, plus an EVID=2 breakpoint at
    /// t=18). Observations are synthesised through `predict_iov` (which seeds each
    /// event at its own covariate) so residuals are realistic on the merged path.
    fn iov_tvcov_subject_outer(model: &CompiledModel, theta: &[f64]) -> Subject {
        let obs_times = vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0];
        let occasions = vec![1u32, 1, 1, 2, 2, 2];
        let obs_wts = [70.0, 72.0, 78.0, 88.0, 90.0, 95.0];
        let n = obs_times.len();
        let mut wt_map = |w: f64| {
            let mut m = std::collections::HashMap::new();
            m.insert("WT".to_string(), w);
            m
        };
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
            covariates: wt_map(70.0),
            dose_covariates: vec![wt_map(70.0), wt_map(85.0)],
            obs_covariates: obs_wts.iter().map(|&w| wt_map(w)).collect(),
            pk_only_times: vec![18.0],
            pk_only_covariates: vec![wt_map(85.0)],
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions,
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
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

    /// The full analytic IOV+TV-cov **packed** gradient — FOCEI **and** FOCE — must
    /// match the Richardson reconverged FD of the production IOV marginal over every
    /// packed coordinate (θ incl. `THETA_WT`, Ω_bsv, σ, Ω_iov). Closes the merged
    /// IOV × time-varying-covariate path end to end against the same `predict_iov`-
    /// grounded objective the non-TV IOV tests use.
    #[test]
    fn iov_tvcov_packed_gradient_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN_IOV_TVCOV).expect("parse warfarin IOV+TVcov");
        let theta = vec![0.22, 11.0, 1.4, 0.7];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_tvcov_subject_outer(&model, &theta);
        assert!(subject.has_tv_covariates(), "fixture must carry TV cov");
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);

        for interaction in [true, false] {
            let analytic = if interaction {
                subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            } else {
                subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
            }
            .expect("IOV+TVcov packed gradient supported");

            let f = |xx: &[f64]| -> f64 {
                let p = unpack_params(xx, &template);
                marginal_nll_iov_inter(&model, &subject, &p, interaction)
            };
            for i in 0..x.len() {
                let h = 1e-4 * (1.0 + x[i].abs());
                let fd_at = |hh: f64| -> f64 {
                    let mut xp = x.clone();
                    xp[i] += hh;
                    let mut xm = x.clone();
                    xm[i] -= hh;
                    (f(&xp) - f(&xm)) / (2.0 * hh)
                };
                let f1 = fd_at(h);
                let f2 = fd_at(h / 2.0);
                let fd = (4.0 * f2 - f1) / 3.0; // Richardson
                eprintln!(
                    "iov tvcov interaction={interaction} x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                    analytic[i],
                    fd,
                    (analytic[i] - fd).abs() / fd.abs().max(1e-9)
                );
                approx::assert_relative_eq!(analytic[i], fd, max_relative = 2e-3, epsilon = 2e-5);
            }
        }
    }
}
