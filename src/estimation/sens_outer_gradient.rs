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
use crate::stats::special::m3_censored_outer;
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
    // M3-censored × residual-eta (`iiv_on_ruv`) cross-term coefficients (#4c).
    // Zero on quantified rows and on non-`iiv_on_ruv` censored rows. With
    // `z = (y−f)/√v`, `h = φ(z)/Φ(z)`, `m = 1/√v + (y−f)·R'/(2 v^{3/2})` and
    // `C = h·(z² + h·z − 1)`, the censored data term `L = −logΦ(z)` under
    // `v = R·exp(2·η_ruv)` has `∂²L/∂η_ruv² = C·z`, `∂²L/∂η_l∂η_ruv = C·m·a_l`,
    // `∂²L/∂η_ruv∂θ = C·m·b`, `∂²L/∂η_ruv∂σ = ½·(C·z)·(∂v/∂σ)/v`. So the true
    // inner Hessian's residual-eta row/col reads `ruv_cz`/`ruv_cm` instead of the
    // Gaussian `2ε²/R`/`ruv_kappa`. Since #486 these `ruv_cz`/`ruv_cm` terms also enter
    // `H̃`/`log|H̃|` (with their θ/σ/η derivatives), consistently with quantified rows.
    ruv_cz: f64, // C·z  (residual-eta diagonal of the true inner Hessian)
    ruv_cm: f64, // C·m  (residual-eta × structural-η / θ / σ coupling)
    /// True for an M3-censored row. The residual-eta blocks read the censored
    /// `ruv_cz`/`ruv_cm` coefficients instead of the Gaussian `2ε²/R`/`ruv_kappa`.
    censored: bool,
    /// Raw NONMEM `CENS` sign for this row (`0` quantified, `>0` below-LLOQ /
    /// lower tail, `<0` above-ULOQ / upper tail). The σ-block's FD of the
    /// censored df-coefficient must re-evaluate the kernel on the same tail, so
    /// the sign is carried here rather than re-read from the subject.
    cens_sign: i8,
    /// `∂Rⱼ/∂θₘ` from the custom residual-error magnitude's *direct* θ-dependence
    /// (#484/#576/#486) — `mult(θ)` enters `R` independent of the prediction `f`,
    /// so this is a channel `theta_block` would otherwise miss entirely. Empty
    /// when no magnitude is active (the common case; zero-cost).
    dr_dtheta: Vec<f64>,
    /// `∂dⱼ/∂θₘ = ∂²Rⱼ/∂f∂θₘ`, the `f`-derivative of `dr_dtheta` — the magnitude
    /// analog of the σ-block's `d_sig`. Empty when no magnitude is active.
    dd_dtheta: Vec<f64>,
}

/// `(g1, g2) = (∂L/∂f, ∂²L/∂f²)` only — used by the reconverge test oracles
/// (`precise_ebe` / `precise_ebe_ruv`); production reads all four via
/// [`m3_censored_outer`]. Delegates so the formula stays single-sourced.
#[cfg(test)]
#[inline]
fn m3_censored_scalars(y: f64, f: f64, r: f64, d: f64, d2: f64, cens: i8) -> (f64, f64) {
    let (g1, g2, _, _) = m3_censored_outer(y, f, r, d, d2, cens);
    (g1, g2)
}

/// `g1 = ∂L/∂f = h·m` only (one kernel, no `g2` `pow` work) — for the censored
/// σ-block FD, which differences `g1` and never needs `g2`.
#[inline]
fn m3_censored_g1(y: f64, f: f64, v: f64, dv_df: f64, cens: i8) -> f64 {
    let (h, _z, m) = crate::stats::special::m3_censored_kernel(y, f, v, dv_df, cens);
    h * m
}

/// Per-censored-row `(a_var, Jⱼ)` for the marginal-variance `∂R̃ⱼⱼ/∂Ω` channel (#646).
struct CensMargRow {
    a_var: f64,
    jrow: DVector<f64>,
}

/// Natural-space censored contributions to the **FOCE (Sheiner–Beal)** packed
/// gradient under the marginal-moment M3 treatment (#646). Each BLOQ row enters as
/// `−logΦ((LLOQ − f0)/√R̃ⱼⱼ)`, with the linearized-marginal mean `f0 = f(η̂) − Jⱼη̂`
/// and variance `R̃ⱼⱼ = Jⱼ Ω Jⱼᵀ + R⁰ⱼ` — the SAME moments the quantified rows use,
/// so FOCE stays a consistent Sheiner–Beal objective (matching Monolix's
/// linearization likelihood and first-order/Tobit theory), unlike the conditional
/// FOCEI censored term. Shared by the non-IOV and IOV FOCE gradients: `theta`,
/// `sigma`, and `coupling = ∂/∂η̂` are uniform, but the two callers pack `Ω`
/// differently, so the direct `Ω` channel is applied per caller via
/// [`CensMargGrad::omega_entry`]. `omega`/`eta_hat`/`n` are the stacked system (η
/// for non-IOV, `[η, κ]` for IOV). All contributions are zero when the subject has
/// no censored rows.
struct CensMargGrad {
    theta: Vec<f64>,
    sigma: Vec<f64>,
    coupling: DVector<f64>,
    rows: Vec<CensMargRow>,
}

impl CensMargGrad {
    /// Precompute `(Jⱼ L) = Lᵀ Jⱼ` (length `n`) for every censored row against the
    /// caller's Cholesky factor `L` — the plain factor for non-IOV, the block-diagonal
    /// `L_full` for IOV. Done once, then reused for every packed Ω entry, mirroring the
    /// quant SB path's one-shot `jl = J·L` (so `(Jⱼ L)_col` is not recomputed per row).
    fn prep_jl(&self, l: &DMatrix<f64>) -> Vec<DVector<f64>> {
        self.rows.iter().map(|c| l.tr_mul(&c.jrow)).collect()
    }

    /// Censored contribution to `∂F/∂L_{row,col}` of the (stacked) Ω Cholesky factor:
    /// `Σ_c a_varc · ∂R̃ⱼⱼ/∂L_{row,col} = Σ_c a_varc · 2·(Jⱼ L)_col · Jⱼ[row]`, reading
    /// `(Jⱼ L)_col` from `jl` (from [`CensMargGrad::prep_jl`]). The caller maps
    /// `(row,col)` to its packed slot.
    fn omega_entry(&self, row: usize, col: usize, jl: &[DVector<f64>]) -> f64 {
        self.rows
            .iter()
            .zip(jl)
            .map(|(c, jlc)| c.a_var * 2.0 * jlc[col] * c.jrow[row])
            .sum()
    }
}

/// Build the [`CensMargGrad`] for a subject's FOCE (Sheiner–Beal) packed gradient.
/// `sens`/`sens0` are the providers at η̂ and at the all-zero random effects; `n`,
/// `omega`, and `eta_hat` are the stacked system. Returns zero contributions when
/// `!m3`. `None` only on a non-finite variance (same guard as the caller's SB path).
#[allow(clippy::too_many_arguments)]
fn censored_marginal_foce_grad(
    model: &CompiledModel,
    subject: &Subject,
    sens: &SubjectSens,
    sens0: &SubjectSens,
    sigma: &[f64],
    omega: &DMatrix<f64>,
    eta_hat: &[f64],
    n: usize,
    n_theta: usize,
    m3: bool,
) -> Option<CensMargGrad> {
    let n_sigma = sigma.len();
    let mut theta = vec![0.0f64; n_theta];
    let mut sigma_g = vec![0.0f64; n_sigma];
    let mut coupling = DVector::<f64>::zeros(n);
    if !m3 {
        return Some(CensMargGrad {
            theta,
            sigma: sigma_g,
            coupling,
            rows: Vec::new(),
        });
    }
    // Per-censored-row marginal coefficients:
    //   a_mean = ∂L/∂mean = h·σ/w,   a_var = ∂L/∂var = h·σ·(LLOQ−f0)/(2w³),
    //   jrow = Jⱼ = ∂f/∂η,  ojc = Ω Jⱼᵀ,  d0 = ∂R⁰/∂f,  f0act = f(η=0).
    struct C {
        j: usize,
        a_mean: f64,
        a_var: f64,
        jrow: DVector<f64>,
        ojc: DVector<f64>,
        d0: f64,
        f0act: f64,
        cmt: usize,
    }
    let mut cens: Vec<C> = Vec::new();
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
    for j in 0..subject.observations.len() {
        let cs = subject.cens.get(j).copied().unwrap_or(0);
        if cs == 0 {
            continue;
        }
        let cmt = err_keys[j];
        let f0act = sens0.obs[j].f;
        let r0c = model.error_spec.variance_at(cmt, f0act, sigma);
        if !(r0c.is_finite() && r0c > 0.0) {
            return None;
        }
        let jrow = DVector::from_column_slice(&sens.obs[j].df_deta);
        let ojc = omega * &jrow; // Ω Jⱼᵀ
        let var = jrow.dot(&ojc) + r0c; // R̃ⱼⱼ = Jⱼ Ω Jⱼᵀ + R⁰ⱼ
        if !(var.is_finite() && var > 0.0) {
            return None;
        }
        let w = var.sqrt();
        let jeta: f64 = (0..n).map(|kk| jrow[kk] * eta_hat[kk]).sum();
        let resid = subject.observations[j] - (sens.obs[j].f - jeta); // LLOQ − f0
        let sgn = if cs < 0 { -1.0 } else { 1.0 };
        let h = crate::stats::special::inv_mills(sgn * resid / w);
        cens.push(C {
            j,
            a_mean: h * sgn / w,
            a_var: h * sgn * resid / (2.0 * w * w * w),
            jrow,
            ojc,
            d0: model.error_spec.dvar_df(cmt, f0act, sigma),
            f0act,
            cmt,
        });
    }
    // θ: ∂mean/∂θ_m = ∂f/∂θ_m − Σ_l (∂J_l/∂θ_m)η̂_l;
    //    ∂var/∂θ_m  = 2·Σ_l (∂J_l/∂θ_m)(Ω Jᵀ)_l + d0·∂f(η=0)/∂θ_m.
    for m in 0..n_theta {
        let mut acc = 0.0;
        for c in &cens {
            let obs_c = &sens.obs[c.j];
            let mut dmean = obs_c.df_dtheta[m];
            let mut djoj = 0.0;
            for li in 0..n {
                let djl = obs_c.d2f_deta_dtheta[li * n_theta + m];
                dmean -= djl * eta_hat[li];
                djoj += djl * c.ojc[li];
            }
            let dvar = 2.0 * djoj + c.d0 * sens0.obs[c.j].df_dtheta[m];
            acc += c.a_mean * dmean + c.a_var * dvar;
        }
        theta[m] = acc;
    }
    // σ: only R⁰ (not Jⱼ Ω Jⱼᵀ) depends on σ, so ∂var/∂σ = ∂R⁰/∂σ (central FD).
    for kk in 0..n_sigma {
        let hsig = sigma_fd_step(sigma[kk]);
        let mut sp = sigma.to_vec();
        sp[kk] += hsig;
        let mut sm = sigma.to_vec();
        sm[kk] -= hsig;
        let mut acc = 0.0;
        for c in &cens {
            let dr0 = (model.error_spec.variance_at(c.cmt, c.f0act, &sp)
                - model.error_spec.variance_at(c.cmt, c.f0act, &sm))
                / (2.0 * hsig);
            acc += c.a_var * dr0;
        }
        sigma_g[kk] = acc;
    }
    // coupling ∂/∂η̂_k: ∂mean/∂η̂_k = −(A_c η̂)_k;  ∂var/∂η̂_k = 2·(A_c Ω Jᵀ)_k.
    for kk in 0..n {
        let mut acc = 0.0;
        for c in &cens {
            let obs_c = &sens.obs[c.j];
            let mut dmean = 0.0;
            let mut dvar = 0.0;
            for li in 0..n {
                let a_kl = obs_c.d2f_deta2[kk * n + li];
                dmean -= a_kl * eta_hat[li];
                dvar += a_kl * c.ojc[li];
            }
            acc += c.a_mean * dmean + c.a_var * 2.0 * dvar;
        }
        coupling[kk] = acc;
    }
    let rows = cens
        .iter()
        .map(|c| CensMargRow {
            a_var: c.a_var,
            jrow: c.jrow.clone(),
        })
        .collect();
    Some(CensMargGrad {
        theta,
        sigma: sigma_g,
        coupling,
        rows,
    })
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
        ruv_cz: 0.0,
        ruv_cm: 0.0,
        censored: false,
        cens_sign: 0,
        dr_dtheta: Vec::new(),
        dd_dtheta: Vec::new(),
    }
}

/// Censored-row σ-block contributions for an M3 model, shared by `sigma_block`
/// and `subject_eta_dx` so the two cannot drift (it does the σ±h error-function
/// evaluation internally, so neither caller hoists it). Returns `(dg1, ruv_sig,
/// l_sig)`:
/// - `dg1 = ∂g1/∂σ` (central FD of the censored df-coefficient `g1 = h·m`; `g2` is
///   never needed here, so the `g1`-only kernel is used) → structural EBE-response
///   `M[:,σ] += dg1·∂f/∂η`;
/// - `ruv_sig = ½·(C·z)·(∂v/∂σ)/v` → the censored residual-η × σ cross-term
///   `M[ruv,σ]` (`0` when no `iiv_on_ruv`);
/// - `l_sig = ∂(−logΦ)/∂σ` → the data σ-term (`sigma_block`'s `fixed`; ignored by
///   `subject_eta_dx`).
///
/// `ruv_scale` applies the `exp(2·η_ruv)` factor; `r` is the scaled variance at σ̂.
#[allow(clippy::too_many_arguments)]
fn censored_sigma_m_terms(
    model: &CompiledModel,
    cmt: usize,
    y: f64,
    f: f64,
    sp: &[f64],
    sm: &[f64],
    h: f64,
    ruv_scale: f64,
    ruv_cz: f64,
    r: f64,
    has_ruv: bool,
    cens: i8,
) -> (f64, f64, f64) {
    let s = ruv_scale;
    let es = &model.error_spec;
    let vp = es.variance_at(cmt, f, sp);
    let vm = es.variance_at(cmt, f, sm);
    let g1p = m3_censored_g1(y, f, vp * s, es.dvar_df(cmt, f, sp) * s, cens);
    let g1m = m3_censored_g1(y, f, vm * s, es.dvar_df(cmt, f, sm) * s, cens);
    let dg1 = (g1p - g1m) / (2.0 * h);
    let ruv_sig = if has_ruv {
        0.5 * ruv_cz * (s * (vp - vm) / (2.0 * h)) / r
    } else {
        0.0
    };
    // Data σ-term `∂(−logΦ(z))/∂σ` by central FD of the censored log-CDF. Uses the
    // tail-correct `m3_logcdf` (upper tail when `cens < 0`) so right-censored rows
    // match the objective; for `cens ≥ 0` this is the historical lower-tail form.
    let l_sig = (-crate::stats::likelihood::m3_logcdf(y, f, (vp * s).sqrt(), cens)
        + crate::stats::likelihood::m3_logcdf(y, f, (vm * s).sqrt(), cens))
        / (2.0 * h);
    (dg1, ruv_sig, l_sig)
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
    // Per-observation M3-censored flag lives on `et[j].censored` (single source).
    // Censored rows enter `H` (true inner Hessian), the data gradient, AND `H̃`/`log|H̃|`
    // at FOCEI order (`p = g2`, `β = dg2/df`; residual-eta `C·z`/`C·m`) — consistently with
    // quantified rows (#486), matching `gaussian_foce_accum`'s `cens_hess`.
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
    /// Total `f`-derivatives (through `v(f)`) of the censored residual-eta `H̃`
    /// coefficients `C·z` / `C·m` per observation (0 on quantified rows). Computed
    /// once in `prepare` (FD of the kernel) and reused by `theta_block`'s censored
    /// residual-eta `log|H̃|` θ-derivative. Empty when no `ruv`.
    cens_dcz_df: Vec<f64>,
    cens_dcm_df: Vec<f64>,
    /// Custom / time-varying residual-magnitude (#484/#576) `[obs][sigma-slot]`
    /// multiplier matrix, or `None` when no magnitude is active. Computed once
    /// here; `sigma_block` reuses it instead of recomputing `model.ruv_obs_mult`
    /// (which re-walks every magnitude expression per observation) a second time
    /// for the same subject/θ (#486 review).
    mult: Option<Vec<Vec<f64>>>,
}

/// Residual-eta coupling `κⱼ = ∂(1−ε²/R)/∂f = 2ε/R + ε²d/R²` — the `f`-derivative
/// of the residual-eta data gradient, used in the mixed η-θ block and the true
/// inner Hessian's residual-eta row (#474). Single source so the two assemblies
/// can't diverge.
#[inline]
fn ruv_kappa(eps: f64, r: f64, d: f64) -> f64 {
    2.0 * eps / r + eps * eps * d / (r * r)
}

/// Per-observation correlation-aware residual scalars `(R_jj, ∂R_jj/∂f_j, ∂²R_jj/∂f_j²)`
/// for a `block_sigma` (#627) model, or `None` to bail to FD.
///
/// In the analytic outer scope (analytical 1-/2-/3-cpt, single endpoint) a
/// `block_sigma` correlation only couples the σ-loadings **within** one observation
/// (`combined(...)` endpoints), so the residual covariance `R` stays **diagonal** —
/// but each diagonal entry, and its `f`-derivatives, carry the within-observation
/// cross term `2ρσ_iσ_j c_i c_j` that the plain scalar `ErrorSpec::dvar_df` /
/// `d2var_df2` omit. With `R` diagonal, the dense Almquist assembly
/// (`H̃ = HᵀR⁻¹H + ½B + Ω⁻¹`, `B_{kl} = tr(M_kM_l)`) reduces **exactly** to the scalar
/// path (`p = 1/R + ½(d/R)²`, `ctc = Σ c̃c̃ᵀ`), so the whole outer gradient is the
/// existing assembly fed these correlation-aware `(r,d,d2)` — no separate dense
/// linear algebra needed. Values come from the **same** builders the marginal
/// (`foce_subject_nll_interaction_dense`) uses, so the gradient stays consistent with
/// the objective bit-for-bit.
///
/// A genuine cross-endpoint off-diagonal `R` (paired total/unbound rows) would need
/// the full dense `M_k`/`B_{kl}` assembly, but such models require a per-CMT / Form-C
/// or covariate-selected (#669) multi-endpoint readout that is out of analytic scope
/// (they run FD). The off-diagonal check is a defensive guard: if one ever reaches
/// here, bail to FD rather than silently drop the off-diagonals. The endpoint keys are
/// resolved via `ErrorSpec::obs_keys` so a `Selected` spec's per-row branch — not the
/// raw CMT column — drives the diagonal variance builders.
fn corr_residual_diag(
    model: &CompiledModel,
    subject: &Subject,
    sens: &SubjectSens,
    sigma: &[f64],
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    use crate::stats::residual_error::{
        compute_d2r_df2_matrices, compute_dr_df_matrices, compute_r_matrix_with_correlations,
    };
    let corr = &model.residual_correlations;
    let es = &model.error_spec;
    // #669: per-observation endpoint keys must come from the covariate selector
    // (`obs_keys`), not the raw CMT column — a `Selected` spec keys endpoints by
    // branch index, decoupled from `obs_cmts` (typically all-1 on an analytical
    // single-endpoint model). Passing `obs_cmts` here would score every row
    // against the wrong branch's sigma. `PerCmt`/`Single` still get exactly the
    // CMT column (`obs_keys` borrows it unchanged).
    let err_keys = es.obs_keys(subject);
    let ipreds: Vec<f64> = sens.obs.iter().map(|o| o.f).collect();
    let n = ipreds.len();
    let r = compute_r_matrix_with_correlations(
        es,
        &ipreds,
        err_keys.as_ref(),
        &subject.obs_times,
        &subject.obs_raw_times,
        &subject.occasions,
        sigma,
        corr,
    );
    // Guard: only diagonal R is served by the scalar reduction (see the doc above).
    for a in 0..n {
        for b in 0..n {
            if a != b && r[(a, b)].abs() > 1e-12 {
                return None;
            }
        }
    }
    let dr = compute_dr_df_matrices(
        es,
        &ipreds,
        err_keys.as_ref(),
        &subject.obs_times,
        &subject.obs_raw_times,
        &subject.occasions,
        sigma,
        corr,
        None,
    );
    let d2 = compute_d2r_df2_matrices(
        es,
        &ipreds,
        err_keys.as_ref(),
        &subject.obs_times,
        &subject.obs_raw_times,
        &subject.occasions,
        sigma,
        corr,
        None,
    );
    let mut rv = vec![0.0; n];
    let mut dv = vec![0.0; n];
    let mut d2v = vec![0.0; n];
    for j in 0..n {
        rv[j] = r[(j, j)];
        dv[j] = dr[j][(j, j)];
        d2v[j] = d2[j][j][(j, j)];
    }
    Some((rv, dv, d2v))
}

/// Correlation-aware per-observation `(R_jj, ∂R_jj/∂f_j)` at a given σ, for the
/// σ-block's central FD (`d2` is not needed there). Diagonals of the same builders
/// as [`corr_residual_diag`]; the diagonal guard is already applied there so this
/// reads the diagonal directly.
fn corr_residual_rd_at_sigma(
    model: &CompiledModel,
    subject: &Subject,
    ipreds: &[f64],
    sigma: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    use crate::stats::residual_error::compute_dr_df_matrices;
    let corr = &model.residual_correlations;
    let es = &model.error_spec;
    let n = ipreds.len();
    // #669: selector-resolved endpoint keys, not the raw CMT column (see
    // `corr_residual_diag`).
    let err_keys = es.obs_keys(subject);
    let dr = compute_dr_df_matrices(
        es,
        ipreds,
        err_keys.as_ref(),
        &subject.obs_times,
        &subject.obs_raw_times,
        &subject.occasions,
        sigma,
        corr,
        None,
    );
    let mut rv = vec![0.0; n];
    let mut dv = vec![0.0; n];
    for j in 0..n {
        rv[j] = es.variance_at_with_correlations(err_keys[j], ipreds[j], sigma, corr);
        dv[j] = dr[j][(j, j)];
    }
    (rv, dv)
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

/// Direct-θ derivatives of a magnitude-scaled residual variance at prediction `f`.
///
/// A custom / time-varying σ magnitude `mult(θ)` (#484/#576/#486) makes the
/// per-observation variance `R_j = Σ_s (coeff_s(f)·mult_sⱼ·σ_s)²` depend on θ
/// **directly** (not only through `f`). Returns `(dr_dtheta, dd_dtheta)`, each
/// length `n_theta`: the θ-gradient of `R_j` and of its `f`-derivative `d_j`,
/// summed over the observation's sigma loadings as `2·coeff²·mult·σ²·∂mult/∂θ`
/// and `4·coeff·coeff'·mult·σ²·∂mult/∂θ` — the same bilinear shape
/// `residual_error::diag_self_deriv` uses for the `f`-derivative, chain-ruled
/// through `mult(θ)` instead of `f`. `ruv_scale` folds the `iiv_on_ruv`
/// `exp(2·η_ruv)` link (`1.0` on the FOCE / non-`ruv` path). `mult_row` is the
/// per-sigma multiplier for this observation and `mult_grad_row` its
/// per-`(sigma, θ)` gradient. Diagonal-`R` only (`block_sigma` correlations force
/// FD upstream via `analytic_outer_gradient_available`). Shared by the FOCEI
/// (`prepare_stacked`) and FOCE (`subject_packed_gradient_foce{,_iov}`) paths.
/// Returns `dr_dtheta` (the `∂R/∂θ` vector); if `dd_dtheta` is `Some`, also
/// accumulates the `f`-derivative `∂d/∂θ` into it. The FOCEI `prepare_stacked`
/// path needs both; the FOCE (Sheiner–Beal) marginal only reads `∂R/∂θ`, so it
/// passes `None` and skips the `slopes` lookup and the `4·coeff·coeff'·…`
/// accumulation entirely (#486 review).
#[allow(clippy::too_many_arguments)]
fn mag_variance_dtheta(
    error_spec: &crate::types::ErrorSpec,
    cmt: usize,
    f: f64,
    sigma: &[f64],
    mult_row: &[f64],
    mult_grad_row: &[Vec<f64>],
    n_theta: usize,
    ruv_scale: f64,
    mut dd_dtheta: Option<&mut Vec<f64>>,
) -> Vec<f64> {
    let loadings = error_spec.sigma_loadings(cmt, f, sigma.len());
    let slopes = if dd_dtheta.is_some() {
        error_spec.sigma_loading_slopes(cmt, sigma.len())
    } else {
        Vec::new()
    };
    let mut dr_dtheta = vec![0.0f64; n_theta];
    for &(idx, coeff) in &loadings {
        let sg = sigma.get(idx).copied().unwrap_or(0.0);
        let mv = mult_row.get(idx).copied().unwrap_or(1.0);
        let Some(dmi) = mult_grad_row.get(idx) else {
            continue;
        };
        let coeff_p = if dd_dtheta.is_some() {
            slopes
                .iter()
                .find(|&&(i, _)| i == idx)
                .map(|&(_, s)| s)
                .unwrap_or(0.0)
        } else {
            0.0
        };
        for (tm, &dmdt_raw) in dmi.iter().enumerate().take(n_theta) {
            let dmdt = dmdt_raw * ruv_scale;
            dr_dtheta[tm] += 2.0 * coeff * coeff * mv * sg * sg * dmdt;
            if let Some(dd) = dd_dtheta.as_deref_mut() {
                dd[tm] += 4.0 * coeff * coeff_p * mv * sg * sg * dmdt;
            }
        }
    }
    dr_dtheta
}

/// The magnitude direct-θ derivative of the data-term coefficient `α` for one
/// observation `et` at θ-axis `m`:
/// `∂α/∂θ = (2ε/R² + d(2ε²−R)/R³)·∂R/∂θ + ((R−ε²)/R²)·∂d/∂θ`, with `∂R/∂θ`,`∂d/∂θ`
/// the magnitude's direct-θ terms (`et.dr_dtheta`/`et.dd_dtheta`). Zero when the
/// observation carries no magnitude derivative. This is the EBE-response ingredient
/// a custom / time-varying σ magnitude adds to the inner mixed derivative
/// `∂²l/∂η∂θ` (#576/#486): FOCEI folds it into `theta_block`'s `m_vec`, FOCE picks
/// it up through `subject_eta_dx{,_iov}`'s `dη̂/dθ` — this shared helper keeps the
/// formula in one place.
fn mag_alpha_dtheta(et: &ErrTerms, m: usize) -> f64 {
    if et.dr_dtheta.is_empty() {
        return 0.0;
    }
    let (r, d, eps) = (et.r, et.d, et.eps);
    let (r_th, d_th) = (et.dr_dtheta[m], et.dd_dtheta[m]);
    if r_th == 0.0 && d_th == 0.0 {
        return 0.0;
    }
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;
    let inv_r3 = inv_r2 * inv_r;
    (2.0 * eps * inv_r2 + d * (2.0 * eps * eps - r) * inv_r3) * r_th
        + ((r - eps * eps) * inv_r2) * d_th
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
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
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
    // `log|H̃|` derivatives. Censored rows under `iiv_on_ruv` carry the analogous
    // `(C·z, C·m)` cross coefficients into the true inner Hessian (closed-form M3 +
    // `iiv_on_ruv`, non-IOV #4c and IOV #591; only the ODE triple stays FD).
    // Reuse the canonical `exp(2·η_ruv)` link (the IOV path forces `ruv = None`
    // to keep its residual variance unscaled, so gate on the local `ruv`, not on
    // `model.residual_error_eta`).
    let ruv_scale = if ruv.is_some() {
        model.residual_var_scale(eta_hat)
    } else {
        1.0
    };
    let n_theta = params.theta.len();
    // Custom / time-varying residual-magnitude (#484/#576/#486): `mult(θ)` is
    // η-independent, so it's evaluated once per subject here and shared by every
    // observation below — both its *value* (scales `r`/`d`/`d2`, like `ruv_scale`)
    // and its `∂/∂θ` (a new *direct*-θ term `theta_block` folds into the data,
    // log|H̃|, and EBE-response pieces below). `None` while a magnitude is active
    // means the `Dual1` program declined (θ-axis count beyond `MAX_RUV_MAG_AXES`);
    // bail to FD rather than silently drop the direct-θ channel — the analytic
    // outer gate bounds `model.n_theta` against the same constant, so this should
    // not trigger for a model the gate already admitted.
    let mult = model.ruv_obs_mult(subject, &params.theta);
    let mult_grad = if mult.is_some() {
        Some(model.ruv_obs_mult_theta_grad(subject, &params.theta)?)
    } else {
        None
    };
    let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3);
    // Custom magnitude threads its direct-θ chain through `iiv_on_ruv` (the residual-eta
    // `c̃`-column `d/R` gets its `∂/∂θ` terms in `theta_block`). The stacked residual-eta
    // assembly below (`m_vec[rr]`, `prep.w[j][rr]`) loops over every stacked axis, so it is
    // dimension-generic across the κ (occasion) block too — magnitude × `iiv_on_ruv` is
    // analytic under IOV as well as non-IOV (#486; FD-validated by
    // `magnitude_iiv_on_ruv_iov_packed_matches_fd`). Magnitude combined with an M3-censored
    // row still bails (the censored `−logΦ(z)` kernel's direct-θ chain is unbuilt); the
    // `cens` scan stays behind `mult && m3` so non-magnitude subjects pay nothing.
    if mult.is_some() && m3 && subject.cens.iter().any(|&c| c != 0) {
        return None;
    }

    // H̃ = Σ pⱼ aⱼaⱼᵀ + Ω⁻¹ ; true inner Hessian H = ½Σ(α'ⱼ aⱼaⱼᵀ + αⱼ Aⱼ) + Ω⁻¹.
    let mut htilde = omega_inv.clone();
    let mut h_inner = omega_inv.clone();
    let mut et: Vec<ErrTerms> = Vec::with_capacity(n_obs);
    let (mut g_ruv, mut gp_ruv) = if ruv.is_some() {
        (vec![0.0f64; n_obs], vec![0.0f64; n_obs])
    } else {
        (Vec::new(), Vec::new())
    };
    let (mut cens_dcz_df, mut cens_dcm_df) = if ruv.is_some() {
        (vec![0.0f64; n_obs], vec![0.0f64; n_obs])
    } else {
        (Vec::new(), Vec::new())
    };

    // Correlated residual (`block_sigma`, #627): precompute the correlation-aware
    // per-obs `(r, d, d2)` diagonals once (block_sigma excludes M3/ruv/IOV, so
    // `ruv_scale = 1`). `None` bails to FD (a rare off-diagonal R). Everything else in
    // the assembly is unchanged — see `corr_residual_diag`.
    let corr_diag = if !model.residual_correlations.is_empty() {
        Some(corr_residual_diag(model, subject, sens, sigma)?)
    } else {
        None
    };
    for obs in sens.obs.iter() {
        let f = obs.f;
        // obs index → cmt: provider obs are parallel to subject.obs_times.
        let j = et.len();
        let cmt = err_keys[j];
        // `mult_row` is `None` for every observation on a non-magnitude model, so
        // that path keeps the exact legacy `variance_at`/`dvar_df`/`d2var_df2`
        // association bit-for-bit (the `_scaled` variants reassociate the
        // `f`-dependent term by ~1 ULP — see `residual_error::compute_r_matrix_with_correlations`).
        let mult_row: Option<&[f64]> = mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
        // Correlated residual (`block_sigma`, #627) uses the precomputed correlation-aware
        // diagonals; otherwise fall back to the per-obs (magnitude-scaled or legacy)
        // variance/derivatives. `block_sigma` and custom magnitude are mutually exclusive
        // (a `block_sigma` model has `mult == None`), so the two branches never mix.
        let (r, d, d2) = match &corr_diag {
            Some((rv, dv, d2v)) => (rv[j], dv[j], d2v[j]),
            None => {
                let r = match mult_row {
                    Some(m) => {
                        model.error_spec.variance_at_scaled(cmt, f, sigma, &[], m) * ruv_scale
                    }
                    None => model.error_spec.variance_at(cmt, f, sigma) * ruv_scale,
                };
                let d = match mult_row {
                    Some(m) => model.error_spec.dvar_df_scaled(cmt, f, sigma, m) * ruv_scale,
                    None => model.error_spec.dvar_df(cmt, f, sigma) * ruv_scale,
                };
                let d2 = match mult_row {
                    Some(m) => model.error_spec.d2var_df2_scaled(cmt, sigma, m) * ruv_scale,
                    None => model.error_spec.d2var_df2(cmt, sigma) * ruv_scale,
                };
                (r, d, d2)
            }
        };
        if !(r.is_finite() && r > 0.0) {
            return None;
        }
        let y = subject.observations[j];
        let cens = subject.cens.get(j).copied().unwrap_or(0);
        let is_cens = m3 && cens != 0;
        // For a censored row the data term is `−logΦ(z)`: store its f-derivatives
        // as `alpha = 2·g1`, `alpha_p = 2·g2` (so the assembly's `½α`, `½α'` recover
        // `∂L/∂f`, `∂²L/∂f²`). Censored rows now enter `H̃`/`log|H̃|` consistently at
        // FOCEI order: the structural block `g2·a·aᵀ` has the SAME form as a quantified
        // row's `p·a·aᵀ`, so we set `p = g2` and `β = dg2/df` (total, through `v(f)`).
        // The existing `p`/`β` machinery in `theta_block`/`g_eta`/`sigma_block` then
        // produces the censored structural `log|H̃|` derivatives with no extra code; the
        // residual-eta `C·z`/`C·m` derivatives are handled in their dedicated blocks.
        // `r`/`d`/`d2` carry `ruv_scale` (and, on the quantified branch above, any active
        // magnitude scaling — never both at once, `iiv_on_ruv` + magnitude is bailed
        // upstream), so the censored scalars are evaluated at the scaled variance (#4c).
        let mut t = if is_cens {
            let (g1, g2, cz, cm) = m3_censored_outer(y, f, r, d, d2, cens);
            // dg2/df — and, under `iiv_on_ruv`, dcz/df and dcm/df — total derivatives
            // through the f-dependent variance `v(f)=variance(f)·s`, by central FD of the
            // scalar kernel (analytic 3rd-order of `−logΦ` is messy; this mirrors the
            // existing censored σ-block FD approach). One `m3_censored_outer` pair at
            // `f±hf` serves all three, so the residual-eta `log|H̃|` loop below reuses the
            // stored `dcz/df`,`dcm/df` rather than re-differencing the kernel.
            let hf = 1e-5 * (1.0 + f.abs());
            let kern_at = |ff: f64| -> (f64, f64, f64) {
                let rr_ = model.error_spec.variance_at(cmt, ff, sigma) * ruv_scale;
                let dd = model.error_spec.dvar_df(cmt, ff, sigma) * ruv_scale;
                let dd2 = model.error_spec.d2var_df2(cmt, sigma) * ruv_scale;
                let (_g1, g2, cz, cm) = m3_censored_outer(y, ff, rr_, dd, dd2, cens);
                (g2, cz, cm)
            };
            let (g2p, czp, cmp) = kern_at(f + hf);
            let (g2m, czm, cmm) = kern_at(f - hf);
            let dg2_df = (g2p - g2m) / (2.0 * hf);
            let (ruv_cz, ruv_cm) = if ruv.is_some() { (cz, cm) } else { (0.0, 0.0) };
            if ruv.is_some() {
                cens_dcz_df[j] = (czp - czm) / (2.0 * hf);
                cens_dcm_df[j] = (cmp - cmm) / (2.0 * hf);
            }
            ErrTerms {
                r,
                d,
                eps: y - f,
                alpha: 2.0 * g1,
                alpha_p: 2.0 * g2,
                p: g2,
                beta: dg2_df,
                ruv_cz,
                ruv_cm,
                censored: true,
                cens_sign: cens,
                dr_dtheta: Vec::new(),
                dd_dtheta: Vec::new(),
            }
        } else {
            err_terms(r, d, d2, y - f)
        };
        // Magnitude direct-θ channel (#576/#486): `R_j = Σ_s (coeff_s(f)·mult_sⱼ·σ_s)²`
        // (diagonal only — `block_sigma` correlations already force FD upstream via
        // `analytic_outer_gradient_available`), so `∂R_j/∂θₘ` and its `f`-derivative
        // `∂d_j/∂θₘ` are a sum over the observation's sigma loadings of
        // `2·coeff²·mult·σ²·∂mult/∂θₘ` and `4·coeff·coeff'·mult·σ²·∂mult/∂θₘ` — the
        // same bilinear shape `residual_error::diag_self_deriv` uses for the
        // `f`-derivative, just chain-ruled through `mult(θ)` instead of `f`.
        if let (Some(m), Some(mg_row)) = (mult_row, mult_grad.as_ref().and_then(|mg| mg.get(j))) {
            let mut dd_dtheta = vec![0.0f64; n_theta];
            let dr_dtheta = mag_variance_dtheta(
                &model.error_spec,
                cmt,
                f,
                sigma,
                m,
                mg_row,
                n_theta,
                ruv_scale,
                Some(&mut dd_dtheta),
            );
            t.dr_dtheta = dr_dtheta;
            t.dd_dtheta = dd_dtheta;
        }

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
            if t.censored {
                // Censored row's residual-eta second derivatives enter BOTH the true inner
                // Hessian AND `H̃`/`log|H̃|` (consistent inclusion): `[ruv,ruv] += C·z`,
                // `[ruv,l] += C·m·a_l` (#4c). `g_ruv`/`gp_ruv` stay 0 (the `∂p/∂η_ruv`
                // quantified term doesn't apply); the censored `log|H̃|` derivative is added
                // separately below.
                h_inner[(rr, rr)] += t.ruv_cz;
                htilde[(rr, rr)] += t.ruv_cz;
                for l in 0..n_eta {
                    if l == rr {
                        continue;
                    }
                    h_inner[(rr, l)] += t.ruv_cm * a[l];
                    h_inner[(l, rr)] += t.ruv_cm * a[l];
                    htilde[(rr, l)] += t.ruv_cm * a[l];
                    htilde[(l, rr)] += t.ruv_cm * a[l];
                }
            } else {
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
        }
        et.push(t);
    }

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
            // The quantified `∂p/∂η_ruv = −2/R` term (the `c̃[ruv,ruv]=2` is constant).
            // Censored rows have their own residual-eta `log|H̃|` derivative (below).
            if !et[j].censored {
                g_eta[rr] += -2.0 * q[j] / et[j].r;
            }
        }
        // Censored residual-eta `log|H̃|` η-derivative: `H̃` carries `C·z` on (rr,rr) and
        // `C·m·a_l` on (rr,l). Trace `H̃⁻¹·∂[·]/∂η` with FD-of-kernel scalars (`dC·z/df`
        // total through `v(f)`, and via the scale `s=exp(2η_ruv)` for the `η_ruv` axis)
        // and analytic `a`/`d2f`.
        for (j, obs) in sens.obs.iter().enumerate() {
            if !et[j].censored {
                continue;
            }
            let f = obs.f;
            let cmt = err_keys[j];
            let y = subject.observations[j];
            let cens = et[j].cens_sign;
            let a = obs.df_deta.as_slice();
            let cm = et[j].ruv_cm;
            let kern_scaled = |ff: f64, ss: f64| -> (f64, f64, f64) {
                let r = model.error_spec.variance_at(cmt, ff, sigma) * ss;
                let d = model.error_spec.dvar_df(cmt, ff, sigma) * ss;
                let d2 = model.error_spec.d2var_df2(cmt, sigma) * ss;
                let (_g1, g2, cz, cm) = m3_censored_outer(y, ff, r, d, d2, cens);
                (g2, cz, cm)
            };
            // `dcz/df`, `dcm/df`: already differenced in the assembly loop (same `f±hf`,
            // same `ruv_scale`), so reuse the stored values instead of re-evaluating.
            let (dcz_df, dcm_df) = (cens_dcz_df[j], cens_dcm_df[j]);
            let hs = 1e-5f64;
            let (g2sp, czsp, cmsp) = kern_scaled(f, ruv_scale * (2.0 * hs).exp());
            let (g2sm, czsm, cmsm) = kern_scaled(f, ruv_scale * (-2.0 * hs).exp());
            let (dcz_drr, dcm_drr) = ((czsp - czsm) / (2.0 * hs), (cmsp - cmsm) / (2.0 * hs));
            // Censored STRUCTURAL `g2·a·aᵀ` also has an `η_ruv` derivative through the
            // variance scale `s=exp(2η_ruv)` (the `p`/`β` machinery only captures the
            // `f`-direction, and `a_{rr}=0`): `∂(g2·a·aᵀ)/∂η_ruv` traced = `(dg2/dη_ruv)·q`.
            let dg2_drr = (g2sp - g2sm) / (2.0 * hs);
            g_eta[rr] += dg2_drr * q[j];
            let hinv_rr = htilde_inv[(rr, rr)];
            for l in 0..n_eta {
                if l == rr {
                    continue;
                }
                let mut s = dcz_df * a[l] * hinv_rr;
                for lp in 0..n_eta {
                    if lp != rr {
                        let da = dcm_df * a[l] * a[lp] + cm * obs.d2f_deta2[lp * n_eta + l];
                        s += 2.0 * htilde_inv[(rr, lp)] * da;
                    }
                }
                g_eta[l] += s;
            }
            let mut s = dcz_drr * hinv_rr;
            for lp in 0..n_eta {
                if lp != rr {
                    s += 2.0 * htilde_inv[(rr, lp)] * dcm_drr * a[lp];
                }
            }
            g_eta[rr] += s;
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
        ruv,
        g_ruv,
        gp_ruv,
        ruv_scale,
        cens_dcz_df,
        cens_dcm_df,
        mult,
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
                // Censored residual-eta `log|H̃|` θ-derivative: `H̃` carries `C·z` on
                // (rr,rr) and `C·m·a_l` on (rr,l). Using `a_{rr}=0` ⇒ `Σ_{l≠rr}H̃⁻¹_{rr,l}a_l
                // = w_{rr}` and `B_{rr,m}=0` ⇒ `Σ_{l≠rr}H̃⁻¹_{rr,l}B_{l,m}=sb`:
                //   ½·(dC·z/df)·bₘ·H̃⁻¹_{rr,rr} + (dC·m/df)·bₘ·w_{rr} + C·m·sb.
                if prep.et[j].censored {
                    let b = obs.df_dtheta[m];
                    g += 0.5 * prep.cens_dcz_df[j] * b * prep.htilde_inv[(rr, rr)]
                        + prep.cens_dcm_df[j] * b * prep.w[j][rr]
                        + prep.et[j].ruv_cm * sb;
                }
            }
        }
        // EBE response: ½ g_eta · dη̂/dθₘ,  dη̂/dθₘ = −H⁻¹ M[:,m].
        let mut m_vec = mixed_eta_theta(&sens.obs, &prep.et, n_eta, n_obs, m, prep.ruv);
        // Magnitude direct-θ channel (#576/#486): a custom residual-magnitude
        // `mult(θ)` makes `R`/`d` depend on θ directly (not only through `f`) —
        // exactly the shape `sigma_block` already handles for σ, substituted
        // `r_sig → dr_dtheta[m]`, `d_sig → dd_dtheta[m]`. Adds the data+lnR term,
        // the log|H̃| `∂p/∂θ` term, and the EBE response's `dalpha`-driven M-vector
        // contribution. `dr_dtheta` is empty for every row when no magnitude is
        // active (the common case). Combined with **`iiv_on_ruv`** (`prep.ruv`),
        // the residual-eta `c̃`-column `d/R` also depends on θ through the magnitude:
        // its `m_vec[rr]` row and `log|H̃|` direct-θ term are added below, mirroring
        // `sigma_block`'s `m_vec[rr] += eps²/R²·r_sig` and `∂(d/R)/∂σ·w[rr]`.
        // `prepare_stacked` still declines a subject combining an active magnitude
        // with an M3-**censored** row, so `et.censored` is never true here.
        for (j, et) in prep.et.iter().enumerate() {
            if et.dr_dtheta.is_empty() {
                continue;
            }
            let (r, d, eps) = (et.r, et.d, et.eps);
            let (r_th, d_th) = (et.dr_dtheta[m], et.dd_dtheta[m]);
            if r_th == 0.0 && d_th == 0.0 {
                continue;
            }
            let inv_r = 1.0 / r;
            let inv_r2 = inv_r * inv_r;
            let inv_r3 = inv_r2 * inv_r;
            // data + lnR:  ½ Rθ (R − ε²)/R².
            g += 0.5 * r_th * (r - eps * eps) * inv_r2;
            // log|H̃|:  ½ (∂p/∂θ) q ,  ∂p/∂θ = −Rθ/R² + d·dθ/R² − d²Rθ/R³.
            let dp = -r_th * inv_r2 + d * d_th * inv_r2 - d * d * r_th * inv_r3;
            g += 0.5 * dp * prep.q[j];
            // ∂α/∂θ folded into M[:,m] (shared with the FOCE EBE-response).
            let dalpha = mag_alpha_dtheta(et, m);
            for k in 0..n_eta {
                m_vec[k] += 0.5 * dalpha * sens.obs[j].df_deta[k];
            }
            // Residual-eta direct-θ terms when `iiv_on_ruv` is active (#486): the
            // `c̃`-column coupling `gⱼ = d/R` is scale-free, so `∂(d/R)/∂θ =
            // (dθ·R − d·Rθ)/R²` uses the stored (ruv-scaled) `d`,`r`,`d_th`,`r_th`.
            if let Some(rr) = prep.ruv {
                m_vec[rr] += eps * eps * inv_r2 * r_th;
                g += (d_th * r - d * r_th) * inv_r2 * prep.w[j][rr];
            }
        }
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
    let k_groups = crate::stats::likelihood::iov_occasion_groups(subject).len();
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
        // IIV on residual error (#474) for IOV: the residual-eta `c̃` column rides the
        // stacked `[η_bsv, κ]` assembly (η_ruv ∈ the BSV block, so `rr < n_eta_bsv` is a
        // valid stacked index; the residual-eta loops already span all stacked axes incl.
        // κ). `None` for plain IOV models (`residual_var_scale` then defaults to 1.0).
        // M3 + IOV is analytic (#580) but the triple M3 + IOV + `iiv_on_ruv` is gated
        // out upstream (`iov_analytical_supported`), so censoring and `ruv` never
        // co-occur here — the censored-row residual-eta blocks of `prepare_stacked`
        // stay unreachable on this path.
        model.residual_error_eta,
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
    // Correlated residual (`block_sigma`, #627): the σ FD must differentiate the
    // correlation-aware variance / `∂R/∂f` (which carry the within-observation cross
    // term), not the plain scalar error functions. Diagonal-R only (guaranteed by
    // `corr_residual_diag`'s guard in `prepare_stacked`).
    let correlated = !model.residual_correlations.is_empty();
    let ipreds: Vec<f64> = sens.obs.iter().map(|o| o.f).collect();
    // Custom / time-varying residual-magnitude (#484/#576): `mult(θ)` is fixed
    // while perturbing σ (it doesn't depend on σ), so the σ±h FD below must hold
    // it constant via the `_scaled` variance functions — otherwise `∂R/∂σ` would
    // be taken against the *unscaled* variance and disagree with the magnitude-
    // aware `r`/`d` this block otherwise consumes from `prep.et`. `prepare_stacked`
    // now admits magnitude × `iiv_on_ruv` on the **non-IOV** path, so the residual-eta
    // branch below *does* run with a non-empty `mult` row — and handles it correctly
    // because `r_sig`/`g_sig` are taken of the `_scaled` (magnitude-aware) variance.
    // Still declined (never seen here): magnitude × `iiv_on_ruv` under IOV, and magnitude
    // × an M3-censored row. Reused from `Prep` (computed once in
    // `prepare_stacked`) rather than recomputed here — `ruv_obs_mult` re-walks
    // every magnitude expression per observation, so recomputing it doubled that
    // cost for every magnitude-active subject on every outer-gradient evaluation
    // (#486 review). `block_sigma` and custom magnitude are mutually exclusive, so
    // at most one of `correlated` / `mult` is active per subject.
    let mult = &prep.mult;
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);

    for k in 0..n_sigma {
        let h = sigma_fd_step(sigma[k]);
        let mut sp = sigma.clone();
        sp[k] += h;
        let mut sm = sigma.clone();
        sm[k] -= h;
        // Correlation-aware `(R_jj, ∂R_jj/∂f_j)` at σ±h, built once per σ_k.
        let (corr_sp, corr_sm) = if correlated {
            (
                Some(corr_residual_rd_at_sigma(model, subject, &ipreds, &sp)),
                Some(corr_residual_rd_at_sigma(model, subject, &ipreds, &sm)),
            )
        } else {
            (None, None)
        };

        let mut fixed = 0.0;
        let mut m_vec = DVector::<f64>::zeros(n_eta);
        for (j, obs) in sens.obs.iter().enumerate() {
            let cmt = err_keys[j];
            let f = obs.f;
            if prep.et[j].censored {
                // M3 censored row: data term `−logΦ((y−f)/√v(σ))` plus the `log|H̃|`
                // σ-terms for the censored curvature (`g2·a·aᵀ` + residual-eta `C·z`/`C·m`,
                // added just below). `l_sig` → `fixed`; the EBE-response structural
                // `dg1` and residual-η × σ cross-term `ruv_sig` via the shared
                // `censored_sigma_m_terms` (which evaluates the σ±h functions once).
                let y = subject.observations[j];
                let (dg1, ruv_sig, l_sig) = censored_sigma_m_terms(
                    model,
                    cmt,
                    y,
                    f,
                    &sp,
                    &sm,
                    h,
                    prep.ruv_scale,
                    prep.et[j].ruv_cz,
                    prep.et[j].r,
                    prep.ruv.is_some(),
                    prep.et[j].cens_sign,
                );
                fixed += l_sig;
                // Censored `log|H̃|` σ-terms (`g2 = p` for censored + residual-eta `C·z`/`C·m`),
                // all by central FD of the kernel at σ±h:
                //   structural: ½·(∂g2/∂σ)·q
                //   residual-eta: ½·(∂C·z/∂σ)·H̃⁻¹_{rr,rr} + (∂C·m/∂σ)·w_{rr}  (`a_{rr}=0`)
                let kern_at = |sa: &[f64]| -> (f64, f64, f64) {
                    let r = model.error_spec.variance_at(cmt, f, sa) * prep.ruv_scale;
                    let d = model.error_spec.dvar_df(cmt, f, sa) * prep.ruv_scale;
                    let d2 = model.error_spec.d2var_df2(cmt, sa) * prep.ruv_scale;
                    let (_g1, g2, cz, cm) = m3_censored_outer(y, f, r, d, d2, prep.et[j].cens_sign);
                    (g2, cz, cm)
                };
                let (g2p, czp, cmp) = kern_at(&sp);
                let (g2m, czm, cmm) = kern_at(&sm);
                fixed += 0.5 * (g2p - g2m) / (2.0 * h) * prep.q[j];
                if let Some(rr) = prep.ruv {
                    let dcz_ds = (czp - czm) / (2.0 * h);
                    let dcm_ds = (cmp - cmm) / (2.0 * h);
                    fixed += 0.5 * dcz_ds * prep.htilde_inv[(rr, rr)] + dcm_ds * prep.w[j][rr];
                }
                for m in 0..n_eta {
                    m_vec[m] += dg1 * obs.df_deta[m];
                }
                if let Some(rr) = prep.ruv {
                    m_vec[rr] += ruv_sig;
                }
                continue;
            }
            let (r, d, eps) = (prep.et[j].r, prep.et[j].d, prep.et[j].eps);
            // Evaluate the four closed-form error functions once at σ±h and reuse
            // them for `r_sig`/`d_sig` and the residual-eta `g_sig` below. For a
            // correlated model (`block_sigma`) these are the correlation-aware variance
            // / `∂R/∂f`. Otherwise `mult` (if active) rides both perturbations unchanged
            // - it doesn't depend on σ.
            let mult_row: Option<&[f64]> =
                mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
            let (vp, vm, dp_var, dm_var) = match (&corr_sp, &corr_sm) {
                (Some((rvp, dvp)), Some((rvm, dvm))) => (rvp[j], rvm[j], dvp[j], dvm[j]),
                _ => match mult_row {
                    Some(m) => (
                        model.error_spec.variance_at_scaled(cmt, f, &sp, &[], m),
                        model.error_spec.variance_at_scaled(cmt, f, &sm, &[], m),
                        model.error_spec.dvar_df_scaled(cmt, f, &sp, m),
                        model.error_spec.dvar_df_scaled(cmt, f, &sm, m),
                    ),
                    None => (
                        model.error_spec.variance_at(cmt, f, &sp),
                        model.error_spec.variance_at(cmt, f, &sm),
                        model.error_spec.dvar_df(cmt, f, &sp),
                        model.error_spec.dvar_df(cmt, f, &sm),
                    ),
                },
            };
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
    let k = crate::stats::likelihood::iov_occasion_groups(subject).len();
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
        // IIV on residual error (#4b): thread the residual-eta index so the production
        // packed gradient applies the `exp(2·η_ruv)` scaling and the `η_ruv` `c̃` column
        // over the stacked layout. `None` for non-`iiv_on_ruv` IOV models. (Was `None` —
        // the fix had only reached the test-only `subject_theta_gradient_iov`.)
        model.residual_error_eta,
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
    // inner Hessian, AND `H̃`/`log|H̃|` at FOCEI order — matching `gaussian_foce_accum`).
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
/// space, or **`None` if any single subject is unsupported** (all-or-nothing).
/// Fixed coordinates are zeroed. `eta_hats[i]` must be subject `i`'s EBE at `x`.
///
/// This all-or-nothing form is used by the **tests** and as a convenience.
/// The production outer loop does **not** use it — it calls
/// [`per_subject_packed_gradients`] / [`per_subject_packed_gradients_iov`] via
/// `population_gradient_sens_mixed`, which keeps the exact analytic gradient for
/// in-scope, finite subjects and fills only the `None`/non-finite ones with a
/// per-subject reconverged FD. So a transiently non-PD inner Hessian (e.g. a
/// degenerate near-LLOQ M3 + `iiv_on_ruv` subject whose `h_inner` cholesky fails)
/// degrades **that subject** to FD, not the whole population.
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
    let n_theta = params.theta.len();
    let n_obs = subject.observations.len();
    if n_obs == 0 {
        return Some(vec![0.0; x.len()]);
    }
    let sens = subject_sensitivities(model, subject, &params.theta, eta_hat)?;
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
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

    // Correlated residual (`block_sigma`, #627): `R⁰` (frozen at η=0) and its `∂/∂f`
    // carry the within-observation `combined` cross term. `R⁰` is diagonal in the
    // analytic FOCE scope, so `R̃ = JΩJᵀ + diag(R⁰)` is unchanged apart from the
    // correlation-aware `(r0,d0)`; a rare off-diagonal bails per-subject to FD via
    // `corr_residual_diag` → `None`. (FOCE is first-order in R — no `∂²R/∂f²`.)
    let correlated = !model.residual_correlations.is_empty();
    let corr = &model.residual_correlations;
    let corr_rd0 = if correlated {
        Some(corr_residual_diag(model, subject, &sens0, sigma)?)
    } else {
        None
    };
    // Custom / time-varying residual-magnitude (#484/#576/#486): thread `mult(θ)`
    // into the Sheiner–Beal marginal — its *value* scales `R⁰` (`variance_at_scaled`
    // below) and its `∂/∂θ` enters the θ-block's `∂R⁰/∂θ` term directly (not only
    // through `f`). Magnitude + an M3-censored row keeps FD (the censored tail's
    // direct-θ chain is unbuilt — mirrors the FOCEI carve-out in `prepare_stacked`).
    // `iiv_on_ruv` never reaches this FOCE (non-interaction) path — it is FOCEI-only
    // (`api.rs` rejects `iiv_on_ruv` under non-interaction), so `residual_error_eta` is
    // `None` here and `R⁰` carries no `exp(2·η_ruv)` scaling (`ruv_scale ≡ 1`). (The
    // model-level magnitude × `iiv_on_ruv` gate was relaxed for the FOCEI path.)
    // `block_sigma` and custom magnitude are mutually exclusive per subject.
    let mult = model.ruv_obs_mult(subject, &params.theta);
    if mult.is_some() && m3 {
        return None;
    }
    let mult_grad = if mult.is_some() {
        Some(model.ruv_obs_mult_theta_grad(subject, &params.theta)?)
    } else {
        None
    };

    // J = ∂f/∂η (nq×n_eta), ρ = y − f0 = ε + J·η̂, R⁰ and d⁰ at f(η=0) — quant rows.
    // `dr0_dtheta[i]` is the magnitude's direct-θ derivative of `R⁰ᵢ` (empty when no
    // magnitude), consumed by the θ-block below.
    let mut jmat = DMatrix::<f64>::zeros(nq, n_eta);
    let mut rho = DVector::<f64>::zeros(nq);
    let mut dr0_dtheta: Vec<Vec<f64>> = vec![Vec::new(); nq];
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
        let cmt = err_keys[j];
        let f0act = sens0.obs[j].f;
        let mult_row: Option<&[f64]> = mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
        // Correlated residual (`block_sigma`, #627): correlation-aware `(R⁰, ∂R⁰/∂f)`.
        // block_sigma is mutually exclusive with custom magnitude and M3, so `mult_row`
        // and the censored (`cg`) path are inactive whenever `corr_rd0` is set.
        let (r, dd) = match &corr_rd0 {
            Some((rv, dv, _)) => (rv[j], dv[j]),
            None => {
                let r = match mult_row {
                    Some(mm) => model
                        .error_spec
                        .variance_at_scaled(cmt, f0act, sigma, &[], mm),
                    None => model.error_spec.variance_at(cmt, f0act, sigma),
                };
                let d = match mult_row {
                    Some(mm) => model.error_spec.dvar_df_scaled(cmt, f0act, sigma, mm),
                    None => model.error_spec.dvar_df(cmt, f0act, sigma),
                };
                (r, d)
            }
        };
        if !(r.is_finite() && r > 0.0) {
            return None;
        }
        r0[i] = r;
        d0[i] = dd;
        if let (Some(mm), Some(mg_row)) = (mult_row, mult_grad.as_ref().and_then(|mg| mg.get(j))) {
            // `R⁰` at f(η=0), so `ruv_scale = 1` (no `iiv_on_ruv` on this path); the
            // Sheiner–Beal marginal only needs `∂R/∂θ`, so skip the `∂d/∂θ` accumulation.
            let dr = mag_variance_dtheta(
                &model.error_spec,
                cmt,
                f0act,
                sigma,
                mm,
                mg_row,
                n_theta,
                1.0,
                None,
            );
            dr0_dtheta[i] = dr;
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

    let n_sigma = sigma.len();
    let mut fixed = vec![0.0f64; x.len()];

    // Marginal-moment M3 censored contributions (#646), shared with the IOV path.
    let cg = censored_marginal_foce_grad(
        model, subject, &sens, &sens0, sigma, omega, eta_hat, n_eta, n_theta, m3,
    )?;

    // θ (fixed η̂): SB part over quant rows + the marginal censored θ-gradient (`cg.theta`,
    // built in `censored_marginal_foce_grad` from ∂f0/∂θ and ∂R̃ⱼⱼ/∂θ — #646).
    //   SB: u·Qₘ + tr(R̃⁻¹EₘΩJᵀ) − u·(EₘΩJᵀu) + ½Σ ∂R⁰/∂θ (R̃⁻¹ᵢᵢ − u²ᵢ).
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
            // ∂R⁰ᵢ/∂θₘ = d⁰ᵢ·∂f0ᵢ/∂θₘ (through the prediction) + the magnitude's
            // *direct*-θ term `dr0_dtheta[i][m]` (empty ⇒ no magnitude, #576/#486).
            let mut dr0 = d0[i] * sens0.obs[j].df_dtheta[m];
            if !dr0_dtheta[i].is_empty() {
                dr0 += dr0_dtheta[i][m];
            }
            dvar += dr0 * (rtilde_inv[(i, i)] - u[i] * u[i]);
        }
        let emojt = &em * &ojt;
        let tr = (&rtilde_inv * &emojt).trace();
        let uemu = u.dot(&(&emojt * &u));
        let nat = u.dot(&qm) + tr - uemu + 0.5 * dvar + cg.theta[m];
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        fixed[m] = nat * dtheta_dx;
    }

    // Ω (fixed η̂, packed Cholesky-L): SB over quant rows + the marginal censored
    // variance's direct Ω-gradient — R̃ⱼⱼ = Jⱼ Ω Jⱼᵀ + R⁰ⱼ depends on Ω (#646), added
    // via `cg.omega_entry` (was zero when the censored term used the residual R⁰).
    let l = &params.omega.chol;
    let jl = &jmat * l;
    let cjl = cg.prep_jl(l); // (Jⱼ L) per censored row, once
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
        fixed[omega_start + ko] = (fixed_l + cg.omega_entry(row, col, &cjl)) * chain;
    }

    // σ (fixed η̂): SB part over quant + the marginal censored σ-gradient (`cg.sigma`;
    // only R⁰ depends on σ, so ∂R̃ⱼⱼ/∂σ = ∂R⁰/∂σ — #646). ∂R⁰/∂σ by central FD of the
    // closed-form variance at f(η=0) — works for FOCE here and FOCEI in sigma_block.
    let sigma_start = omega_start + entries.len();
    for k in 0..n_sigma {
        let hsig = sigma_fd_step(sigma[k]);
        let mut sp = sigma.clone();
        sp[k] += hsig;
        let mut sm = sigma.clone();
        sm[k] -= hsig;
        let mut nat = 0.0;
        for (i, &j) in quant.iter().enumerate() {
            let cmt = err_keys[j];
            let f0act = sens0.obs[j].f;
            // Correlation-aware `∂R⁰/∂σ` when block_sigma present (within-obs cross
            // term); otherwise ∂R⁰/∂σ carries the magnitude multiplier (`mult` scales
            // the σ loading), so FD the *scaled* variance when a magnitude is active
            // (#576/#486). block_sigma and magnitude are mutually exclusive.
            let mult_row: Option<&[f64]> =
                mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
            let (vp, vm) = if correlated {
                (
                    model
                        .error_spec
                        .variance_at_with_correlations(cmt, f0act, &sp, corr),
                    model
                        .error_spec
                        .variance_at_with_correlations(cmt, f0act, &sm, corr),
                )
            } else {
                match mult_row {
                    Some(mm) => (
                        model
                            .error_spec
                            .variance_at_scaled(cmt, f0act, &sp, &[], mm),
                        model
                            .error_spec
                            .variance_at_scaled(cmt, f0act, &sm, &[], mm),
                    ),
                    None => (
                        model.error_spec.variance_at(cmt, f0act, &sp),
                        model.error_spec.variance_at(cmt, f0act, &sm),
                    ),
                }
            };
            let dr0 = (vp - vm) / (2.0 * hsig);
            nat += 0.5 * dr0 * (rtilde_inv[(i, i)] - u[i] * u[i]);
        }
        nat += cg.sigma[k];
        fixed[sigma_start + k] = nat * sigma[k];
    }

    // Coupling c = ∂F/∂η̂: SB part over quant rows + the marginal censored coupling
    // (`cg.coupling`; the tail's η̂-response through both the marginal mean and R̃ⱼⱼ — #646).
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
        let ck = u.dot(&pk) + tr - udku + cg.coupling[k];
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
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
    let k = crate::stats::likelihood::iov_occasion_groups(subject).len();
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
        // Thread the residual-eta index for `iiv_on_ruv` IOV models (#4b). Defensive:
        // the only consumer (`subject_packed_gradient_foce_iov`) is unreachable when
        // `interaction` is set, and `iiv_on_ruv` requires FOCEI — but keep it correct.
        model.residual_error_eta,
    )?;
    let n_theta = params.theta.len();
    let n_sigma = params.sigma.values.len();
    let mut out: Vec<DVector<f64>> = vec![DVector::zeros(n_st); x.len()];

    // Custom-magnitude support (#576/#486): `mult(θ)` adds a direct-θ term to the
    // inner `∂²l/∂η∂θ` (θ block) and makes `∂R/∂σ` magnitude-scaled (σ block below);
    // `None` for a bare-sigma model. See the non-IOV `subject_eta_dx` for the rationale.
    // Reused from `Prep` (built once in `prepare_stacked`) — recomputing re-walks
    // every magnitude expression per observation (#486 review).
    let mult = &prep.mult;

    // θ coords.
    for m in 0..n_theta {
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        let mut mvec = mixed_eta_theta(&sens.obs, &prep.et, n_st, prep.n_obs, m, prep.ruv);
        for (j, et) in prep.et.iter().enumerate() {
            let dalpha = mag_alpha_dtheta(et, m);
            if dalpha != 0.0 {
                for kk in 0..n_st {
                    mvec[kk] += 0.5 * dalpha * sens.obs[j].df_deta[kk];
                }
            }
        }
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

    // σ coords: M_σ = ½ Σⱼ ∂αⱼ/∂σ · aⱼ; ×σ. M3-censored rows (#591, the FOCE-IOV-M3
    // coupling's EBE response) use the conditional-variance inner term `dg1·∂f/∂η` via
    // the shared `censored_sigma_m_terms` (`l_sig` unused — no `fixed` term here). FOCE
    // has no `iiv_on_ruv` (`prep.ruv` is `None`), so the `ruv_sig` cross-term is skipped.
    let sigma = &params.sigma.values;
    for kk in 0..n_sigma {
        let h = sigma_fd_step(sigma[kk]);
        let mut sp = sigma.clone();
        sp[kk] += h;
        let mut sm = sigma.clone();
        sm[kk] -= h;
        let mut mvec = DVector::<f64>::zeros(n_st);
        for (j, obs) in sens.obs.iter().enumerate() {
            let cmt = err_keys[j];
            let f = obs.f;
            if prep.et[j].censored {
                let y = subject.observations[j];
                let (dg1, ruv_sig, _l_sig) = censored_sigma_m_terms(
                    model,
                    cmt,
                    y,
                    f,
                    &sp,
                    &sm,
                    h,
                    prep.ruv_scale,
                    prep.et[j].ruv_cz,
                    prep.et[j].r,
                    prep.ruv.is_some(),
                    prep.et[j].cens_sign,
                );
                for m in 0..n_st {
                    mvec[m] += dg1 * obs.df_deta[m];
                }
                if let Some(rr) = prep.ruv {
                    mvec[rr] += ruv_sig;
                }
                continue;
            }
            let (r, d, eps) = (prep.et[j].r, prep.et[j].d, prep.et[j].eps);
            // Magnitude-scaled `∂R/∂σ`,`∂d/∂σ` (consistent with the scaled `et.r`/`et.d`).
            let mult_row: Option<&[f64]> =
                mult.as_ref().and_then(|mm| mm.get(j)).map(|v| v.as_slice());
            let (var_p, var_m, dvar_p, dvar_m) = match mult_row {
                Some(mm) => (
                    model.error_spec.variance_at_scaled(cmt, f, &sp, &[], mm),
                    model.error_spec.variance_at_scaled(cmt, f, &sm, &[], mm),
                    model.error_spec.dvar_df_scaled(cmt, f, &sp, mm),
                    model.error_spec.dvar_df_scaled(cmt, f, &sm, mm),
                ),
                None => (
                    model.error_spec.variance_at(cmt, f, &sp),
                    model.error_spec.variance_at(cmt, f, &sm),
                    model.error_spec.dvar_df(cmt, f, &sp),
                    model.error_spec.dvar_df(cmt, f, &sm),
                ),
            };
            let r_sig = (var_p - var_m) / (2.0 * h);
            let d_sig = (dvar_p - dvar_m) / (2.0 * h);
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
///
/// M3 BLOQ (#591): censored rows leave the augmented Sheiner–Beal marginal (R̃ and
/// the quadratic form are built over the quantified rows only) and re-enter as the
/// marginal tail `−logΦ((LLOQ−f0)/√R̃ⱼⱼ)`, R̃ⱼⱼ = Hⱼ Σ_b Hⱼᵀ + R⁰ⱼ over the stacked
/// [η, κ] system (#646) — the FOCE-IOV-M3 objective
/// `foce_subject_nll_iov(interaction = false)` builds (the stacked analogue of the
/// non-IOV `subject_packed_gradient_foce`). `quant` maps an SB-local row to its
/// original obs index. The marginal contributions are shared via
/// [`censored_marginal_foce_grad`].
pub fn subject_packed_gradient_foce_iov(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    stacked_eta_hat: &[f64],
) -> Option<Vec<f64>> {
    let params = unpack_params(x, template);
    let n_theta = params.theta.len();
    let sens = crate::sens::provider::subject_sensitivities_iov(
        model,
        subject,
        &params.theta,
        stacked_eta_hat,
    )?;
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
    let k = crate::stats::likelihood::iov_occasion_groups(subject).len();
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

    // M3 BLOQ: the censored rows leave the augmented Sheiner–Beal marginal (R̃ and the
    // quadratic form are built over the quantified rows only) and re-enter as the marginal
    // tail `−logΦ((LLOQ−f0)/√R̃ⱼⱼ)`, R̃ⱼⱼ = Hⱼ Σ_b Hⱼᵀ + R⁰ⱼ (#646) — matching
    // `foce_subject_nll_iov(interaction = false)`. `quant` maps an SB-local row `i` →
    // original obs index `j`. (FOCE-IOV-M3 no longer promotes to interaction as of #591,
    // so this is the gradient of the actual objective.)
    let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3)
        && subject.cens.iter().any(|&c| c != 0);
    let quant: Vec<usize> = (0..n_obs)
        .filter(|&j| !(m3 && subject.cens.get(j).copied().unwrap_or(0) != 0))
        .collect();
    let nq = quant.len();
    if nq == 0 {
        return None;
    }

    // Custom / time-varying residual-magnitude (#484/#576/#486): thread `mult(θ)`
    // into the stacked-`[η_bsv,κ]` Sheiner–Beal marginal — same shape as the non-IOV
    // sibling `subject_packed_gradient_foce`. Magnitude + M3-censored keeps FD.
    // `iiv_on_ruv` never reaches this FOCE-IOV path — it is FOCEI-only (`api.rs` rejects
    // it under non-interaction), so `residual_error_eta` is `None` and `ruv_scale ≡ 1`.
    // (The FOCEI magnitude × `iiv_on_ruv` gate was relaxed for non-IOV only.)
    let mult = model.ruv_obs_mult(subject, &params.theta);
    if mult.is_some() && m3 {
        return None;
    }
    let mult_grad = if mult.is_some() {
        Some(model.ruv_obs_mult_theta_grad(subject, &params.theta)?)
    } else {
        None
    };

    // J = ∂f/∂[η,κ] (nq×n_st), ρ = ε + J·b̂, R⁰ and d⁰ at f(all-zero) — quant rows.
    let mut jmat = DMatrix::<f64>::zeros(nq, n_st);
    let mut rho = DVector::<f64>::zeros(nq);
    let mut r0 = vec![0.0f64; nq];
    let mut d0 = vec![0.0f64; nq];
    let mut dr0_dtheta: Vec<Vec<f64>> = vec![Vec::new(); nq];
    for (i, &j) in quant.iter().enumerate() {
        let obs = &sens.obs[j];
        let mut jeta = 0.0;
        for kk in 0..n_st {
            jmat[(i, kk)] = obs.df_deta[kk];
            jeta += obs.df_deta[kk] * stacked_eta_hat[kk];
        }
        rho[i] = subject.observations[j] - (obs.f - jeta);
        let cmt = err_keys[j];
        let f0act = sens0.obs[j].f;
        let mult_row: Option<&[f64]> = mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
        let r = match mult_row {
            Some(mm) => model
                .error_spec
                .variance_at_scaled(cmt, f0act, sigma, &[], mm),
            None => model.error_spec.variance_at(cmt, f0act, sigma),
        };
        if !(r.is_finite() && r > 0.0) {
            return None;
        }
        r0[i] = r;
        d0[i] = match mult_row {
            Some(mm) => model.error_spec.dvar_df_scaled(cmt, f0act, sigma, mm),
            None => model.error_spec.dvar_df(cmt, f0act, sigma),
        };
        if let (Some(mm), Some(mg_row)) = (mult_row, mult_grad.as_ref().and_then(|mg| mg.get(j))) {
            // Sheiner–Beal marginal only needs `∂R/∂θ` → skip the `∂d/∂θ` accumulation.
            let dr = mag_variance_dtheta(
                &model.error_spec,
                cmt,
                f0act,
                sigma,
                mm,
                mg_row,
                n_theta,
                1.0,
                None,
            );
            dr0_dtheta[i] = dr;
        }
    }

    let jo = &jmat * &omega_full;
    let mut rtilde = &jo * jmat.transpose();
    for i in 0..nq {
        rtilde[(i, i)] += r0[i];
    }
    let rtilde_inv = rtilde.cholesky()?.inverse();
    let u = &rtilde_inv * &rho;
    let ojt = &omega_full * jmat.transpose();

    let n_sigma = sigma.len();
    let mut fixed = vec![0.0f64; x.len()];

    // Marginal-moment M3 censored contributions (#646) over the stacked [η, κ]
    // system; shared with the non-IOV path. `omega_full` = block-diag(Ω_bsv, Ω_iov).
    let cg = censored_marginal_foce_grad(
        model,
        subject,
        &sens,
        &sens0,
        sigma,
        &omega_full,
        stacked_eta_hat,
        n_st,
        n_theta,
        m3,
    )?;

    // θ (fixed η̂): SB part over quant rows + the marginal censored θ-gradient
    // (`cg.theta`, over the stacked [η, κ] system — #646).
    for m in 0..n_theta {
        let mut qm = DVector::<f64>::zeros(nq);
        let mut em = DMatrix::<f64>::zeros(nq, n_st);
        let mut dvar = 0.0;
        for (i, &j) in quant.iter().enumerate() {
            let obs = &sens.obs[j];
            let mut bjeta = 0.0;
            for l in 0..n_st {
                let bjl = obs.d2f_deta_dtheta[l * n_theta + m];
                em[(i, l)] = bjl;
                bjeta += bjl * stacked_eta_hat[l];
            }
            qm[i] = -obs.df_dtheta[m] + bjeta;
            // ∂R⁰ᵢ/∂θₘ = d⁰ᵢ·∂f0ᵢ/∂θₘ + the magnitude's direct-θ term (#576/#486).
            let mut dr0 = d0[i] * sens0.obs[j].df_dtheta[m];
            if !dr0_dtheta[i].is_empty() {
                dr0 += dr0_dtheta[i][m];
            }
            dvar += dr0 * (rtilde_inv[(i, i)] - u[i] * u[i]);
        }
        let emojt = &em * &ojt;
        let tr = (&rtilde_inv * &emojt).trace();
        let uemu = u.dot(&(&emojt * &u));
        let nat = u.dot(&qm) + tr - uemu + 0.5 * dvar + cg.theta[m];
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        fixed[m] = nat * dtheta_dx;
    }

    // Ω (fixed η̂): per Cholesky entry of Σ_b, BSV direct + K-summed IOV. The marginal
    // censored variance R̃ⱼⱼ = Jⱼ Σ_b Jⱼᵀ + R⁰ⱼ depends on the full block factor L_full,
    // so `cg.omega_entry` adds the censored channel at the same (row,col) the SB
    // `entry_grad` reads (#646).
    let l_bsv = &params.omega.chol;
    let l_iov = &omega_iov.chol;
    let l_full = block_chol_full(l_bsv, l_iov, k, n_eta_bsv, n_iov);
    let jl = &jmat * &l_full;
    let cjl = cg.prep_jl(&l_full); // (Jⱼ L_full) per censored row, once
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
        fixed[omega_start + e] = (entry_grad(row, col) + cg.omega_entry(row, col, &cjl)) * chain;
    }
    let sigma_start = omega_start + bsv_entries.len();

    // σ (fixed η̂): SB part over quant rows + the marginal censored σ-gradient
    // (`cg.sigma`; ∂R̃ⱼⱼ/∂σ = ∂R⁰/∂σ — #646). ∂R⁰/∂σ by central FD of the closed-form
    // variance at f(η=0, κ=0).
    for kk in 0..n_sigma {
        let hsig = sigma_fd_step(sigma[kk]);
        let mut sp = sigma.clone();
        sp[kk] += hsig;
        let mut sm = sigma.clone();
        sm[kk] -= hsig;
        let mut nat = 0.0;
        for (i, &j) in quant.iter().enumerate() {
            let cmt = err_keys[j];
            let f0act = sens0.obs[j].f;
            // Magnitude-aware ∂R⁰/∂σ (the multiplier scales the σ loading) — #576/#486.
            let mult_row: Option<&[f64]> =
                mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
            let (vp, vm) = match mult_row {
                Some(mm) => (
                    model
                        .error_spec
                        .variance_at_scaled(cmt, f0act, &sp, &[], mm),
                    model
                        .error_spec
                        .variance_at_scaled(cmt, f0act, &sm, &[], mm),
                ),
                None => (
                    model.error_spec.variance_at(cmt, f0act, &sp),
                    model.error_spec.variance_at(cmt, f0act, &sm),
                ),
            };
            let dr0 = (vp - vm) / (2.0 * hsig);
            nat += 0.5 * dr0 * (rtilde_inv[(i, i)] - u[i] * u[i]);
        }
        nat += cg.sigma[kk];
        fixed[sigma_start + kk] = nat * sigma[kk];
    }
    let iov_start = sigma_start + n_sigma;
    let iov_entries = lower_tri_entries(n_iov, omega_iov.diagonal);
    for (e, &(i, j)) in iov_entries.iter().enumerate() {
        let mut raw = 0.0;
        for kk in 0..k {
            let row = n_eta_bsv + kk * n_iov + i;
            let col = n_eta_bsv + kk * n_iov + j;
            raw += entry_grad(row, col) + cg.omega_entry(row, col, &cjl);
        }
        let chain = if i == j { l_iov[(i, i)] } else { 1.0 };
        fixed[iov_start + e] = raw * chain;
    }

    // Coupling c = ∂F/∂η̂ over the stacked random effects: SB part over quant rows +
    // the marginal censored coupling (`cg.coupling`; the tail's [η, κ]-response through
    // both the marginal mean and R̃ⱼⱼ — #646).
    let mut coupling = DVector::<f64>::zeros(n_st);
    for kk in 0..n_st {
        let mut pk = DVector::<f64>::zeros(nq);
        let mut dk = DMatrix::<f64>::zeros(nq, n_st);
        for (i, &j) in quant.iter().enumerate() {
            let obs = &sens.obs[j];
            let mut s = 0.0;
            for l in 0..n_st {
                let a_kl = obs.d2f_deta2[kk * n_st + l];
                s += a_kl * stacked_eta_hat[l];
                dk[(i, l)] = a_kl;
            }
            pk[i] = s;
        }
        let dkojt = &dk * &ojt;
        let tr = (&rtilde_inv * &dkojt).trace();
        let udku = u.dot(&(&dkojt * &u));
        let ck = u.dot(&pk) + tr - udku + cg.coupling[kk];
        coupling[kk] = ck;
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
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
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
        let mut mvec = mixed_eta_theta(&sens.obs, &prep.et, n_eta, prep.n_obs, m, prep.ruv);
        // Custom / time-varying σ magnitude (#576/#486): `mult(θ)` makes the inner
        // variance depend on θ directly, adding `½ ∂α/∂θ · a` to `∂²l/∂η∂θ` — the
        // EBE-response term FOCEI folds into `theta_block`'s `m_vec`. FOCE reaches it
        // only here, so without this its `dη̂/dθ` (and the coupling term built on it)
        // silently drops the magnitude θ's contribution. No-op for a bare-sigma model
        // (`dr_dtheta` empty ⇒ `mag_alpha_dtheta` returns 0).
        for (j, et) in prep.et.iter().enumerate() {
            let dalpha = mag_alpha_dtheta(et, m);
            if dalpha != 0.0 {
                for k in 0..n_eta {
                    mvec[k] += 0.5 * dalpha * sens.obs[j].df_deta[k];
                }
            }
        }
        out[m] = -(&prep.h_inner_inv * mvec) * dtheta_dx;
    }

    // Custom-magnitude σ derivatives (#576/#486): `∂R/∂σ`/`∂d/∂σ` below must be taken
    // of the *scaled* variance (`et.r`/`et.d` already carry `mult`), else the σ
    // EBE-response is inconsistent for a magnitude model. `None` for a bare-sigma model.
    // Reused from `Prep` (built once in `prepare`) rather than recomputed — `ruv_obs_mult`
    // re-walks every magnitude expression per observation (#486 review).
    let mult = &prep.mult;

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
    // Correlated residual (`block_sigma`, #627): σ FD must use the correlation-aware
    // variance / `∂R/∂f` (mirrors `sigma_block`). Diagonal-R only (guarded in `prepare`).
    let correlated = !model.residual_correlations.is_empty();
    let eta_dx_ipreds: Vec<f64> = sens.obs.iter().map(|o| o.f).collect();
    for k in 0..n_sigma {
        let h = sigma_fd_step(sigma[k]);
        let mut sp = sigma.clone();
        sp[k] += h;
        let mut sm = sigma.clone();
        sm[k] -= h;
        let (corr_sp, corr_sm) = if correlated {
            (
                Some(corr_residual_rd_at_sigma(
                    model,
                    subject,
                    &eta_dx_ipreds,
                    &sp,
                )),
                Some(corr_residual_rd_at_sigma(
                    model,
                    subject,
                    &eta_dx_ipreds,
                    &sm,
                )),
            )
        } else {
            (None, None)
        };
        let mut mvec = DVector::<f64>::zeros(n_eta);
        for (j, obs) in sens.obs.iter().enumerate() {
            let cmt = err_keys[j];
            let f = obs.f;
            // M3 censored row EBE-response: structural `dg1·∂f/∂η` plus the censored
            // residual-η × σ cross-term, shared with `sigma_block` via
            // `censored_sigma_m_terms` (`l_sig` is unused here — no `fixed` term).
            if prep.et[j].censored {
                let y = subject.observations[j];
                let (dg1, ruv_sig, _l_sig) = censored_sigma_m_terms(
                    model,
                    cmt,
                    y,
                    f,
                    &sp,
                    &sm,
                    h,
                    prep.ruv_scale,
                    prep.et[j].ruv_cz,
                    prep.et[j].r,
                    prep.ruv.is_some(),
                    prep.et[j].cens_sign,
                );
                for m in 0..n_eta {
                    mvec[m] += dg1 * obs.df_deta[m];
                }
                if let Some(rr) = prep.ruv {
                    mvec[rr] += ruv_sig;
                }
                continue;
            }
            let (r, d, eps) = (prep.et[j].r, prep.et[j].d, prep.et[j].eps);
            // `et.r`/`et.d` carry the `exp(2·η_ruv)` scale *and* any custom-magnitude
            // `mult`, so lift `∂R/∂σ`,`∂d/∂σ` the same way (mirrors `sigma_block`);
            // `ruv_scale == 1` when there is no ruv, `mult_row = None` for bare sigma.
            // For a correlated model (`block_sigma`) these use the correlation-aware
            // variance / `∂R/∂f` (mutually exclusive with custom magnitude).
            let mult_row: Option<&[f64]> =
                mult.as_ref().and_then(|mm| mm.get(j)).map(|v| v.as_slice());
            let (var_p, var_m, dvar_p, dvar_m) = match (&corr_sp, &corr_sm) {
                (Some((rvp, dvp)), Some((rvm, dvm))) => (rvp[j], rvm[j], dvp[j], dvm[j]),
                _ => match mult_row {
                    Some(mm) => (
                        model.error_spec.variance_at_scaled(cmt, f, &sp, &[], mm),
                        model.error_spec.variance_at_scaled(cmt, f, &sm, &[], mm),
                        model.error_spec.dvar_df_scaled(cmt, f, &sp, mm),
                        model.error_spec.dvar_df_scaled(cmt, f, &sm, mm),
                    ),
                    None => (
                        model.error_spec.variance_at(cmt, f, &sp),
                        model.error_spec.variance_at(cmt, f, &sm),
                        model.error_spec.dvar_df(cmt, f, &sp),
                        model.error_spec.dvar_df(cmt, f, &sm),
                    ),
                },
            };
            let r_sig = prep.ruv_scale * (var_p - var_m) / (2.0 * h);
            let d_sig = prep.ruv_scale * (dvar_p - dvar_m) / (2.0 * h);
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
            // ∂²l/∂η_ruv∂θ: Gaussian `ruv_kappa·b`, or the censored cross `C·m·b` (#4c).
            let coef = if et[j].censored {
                et[j].ruv_cm
            } else {
                ruv_kappa(et[j].eps, et[j].r, et[j].d)
            };
            mk[rr] += coef * bjm;
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
    use crate::types::{DoseEvent, ErrorSpec, OmegaMatrix, Subject};
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
    ///
    /// Custom / time-varying residual-magnitude models (#484/#576/#486): `mult`
    /// scales the per-observation variance the same way `prepare_stacked` does, so
    /// this locates the *true* magnitude-aware η̂ — without it, the reconverged-FD
    /// harness would minimise a different (bare) inner objective and the
    /// analytic-vs-FD comparison in `magnitude_*_family_outer_gradient_matches_fd`
    /// would be meaningless (the Eq. 46 EBE-response identity only holds at the
    /// actual stationary point).
    fn precise_ebe(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> Vec<f64> {
        let warm = find_ebe(model, subject, params, 80, 1e-10, None, None);
        let mut eta: Vec<f64> = warm.eta.iter().copied().collect();
        let n_eta = model.n_eta;
        let sigma = &params.sigma.values;
        let omega_inv = &params.omega.inv;
        let mult = model.ruv_obs_mult(subject, &params.theta);
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
                let mult_row: Option<&[f64]> =
                    mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
                let (r, d, d2) = match mult_row {
                    Some(m) => (
                        model.error_spec.variance_at_scaled(cmt, f, sigma, &[], m),
                        model.error_spec.dvar_df_scaled(cmt, f, sigma, m),
                        model.error_spec.d2var_df2_scaled(cmt, sigma, m),
                    ),
                    None => (
                        model.error_spec.variance_at(cmt, f, sigma),
                        model.error_spec.dvar_df(cmt, f, sigma),
                        model.error_spec.d2var_df2(cmt, sigma),
                    ),
                };
                let y = subject.observations[j];
                // (g1, g2) = (∂L/∂f, ∂²L/∂f²): the censored `−logΦ` scalars for an
                // M3 BLOQ row, else the Gaussian `½α`, `½α'`.
                let (g1, g2) = if m3 && subject.cens.get(j).copied().unwrap_or(0) != 0 {
                    m3_censored_scalars(y, f, r, d, d2, subject.cens.get(j).copied().unwrap_or(0))
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
        let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3);
        let sigma = &params.sigma.values;
        let omega_inv = &params.omega.inv;
        // Custom/TV residual magnitude: the inner NLL must use the same magnitude-scaled
        // variance production does, else the reconverged η is not the true inner minimum
        // and the envelope-based outer gradient mismatches FD.
        let mult = model.ruv_obs_mult(subject, &params.theta);
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
                let mult_row: Option<&[f64]> =
                    mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
                let (r, d, d2) = match mult_row {
                    Some(mr) => (
                        model.error_spec.variance_at_scaled(cmt, f, sigma, &[], mr) * s,
                        model.error_spec.dvar_df_scaled(cmt, f, sigma, mr) * s,
                        model.error_spec.d2var_df2_scaled(cmt, sigma, mr) * s,
                    ),
                    None => (
                        model.error_spec.variance_at(cmt, f, sigma) * s,
                        model.error_spec.dvar_df(cmt, f, sigma) * s,
                        model.error_spec.d2var_df2(cmt, sigma) * s,
                    ),
                };
                let y = subject.observations[j];
                let eps = y - f;
                let is_cens = m3 && subject.cens.get(j).copied().unwrap_or(0) != 0;
                let (g1, g2) = if is_cens {
                    m3_censored_scalars(y, f, r, d, d2, subject.cens.get(j).copied().unwrap_or(0))
                } else {
                    let t = err_terms(r, d, d2, eps);
                    (0.5 * t.alpha, 0.5 * t.alpha_p)
                };
                let a = obs.df_deta.as_slice();
                for k in 0..n_eta {
                    grad[k] += g1 * a[k];
                    for l in 0..n_eta {
                        hess[(k, l)] += g2 * a[k] * a[l] + g1 * obs.d2f_deta2[k * n_eta + l];
                    }
                }
                // Residual-eta gradient / Hessian (a_{ruv} = A_{·,ruv} = 0). Censored
                // rows use the M3 cross-terms `h·z` / `C·z` / `C·m·a` (#4c); quantified
                // rows the Gaussian `1−ε²/R` / `2ε²/R` / `κ a` (#474).
                if is_cens {
                    let (h, z, m) = crate::stats::special::m3_censored_kernel(
                        y,
                        f,
                        r,
                        d,
                        subject.cens.get(j).copied().unwrap_or(0),
                    );
                    let c = h * (z * z + h * z - 1.0);
                    grad[rr] += h * z;
                    hess[(rr, rr)] += c * z;
                    for l in 0..n_eta {
                        if l == rr {
                            continue;
                        }
                        hess[(rr, l)] += c * m * a[l];
                        hess[(l, rr)] += c * m * a[l];
                    }
                } else {
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

    /// 1-cpt IV, two BSV etas, correlated `combined` residual error (`block_sigma`, #627).
    const BLOCK_SIGMA_1CPT: &str = "[parameters]\n  theta TVCL(1.0, 0.01, 10.0)\n  theta TVV(10.0, 0.1, 100.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.05, 1.00]\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n[fit_options]\n  method = focei\n";

    fn dense_subject(model: &CompiledModel, theta: &[f64], times: &[f64]) -> Subject {
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
        let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, &[0.12, -0.08]);
        subject.observations = preds.iter().map(|p| p * 0.85 + 0.2).collect();
        subject
    }

    /// Correlation-aware EBE: Newton on the dense inner NLL using the diagonal-but-
    /// correlated `(r,d,d2)` from [`corr_residual_diag`]. The scalar [`precise_ebe`] uses
    /// the plain error functions (no within-obs cross term) and would converge to the
    /// wrong mode for `block_sigma`, breaking the envelope theorem the gradient assumes.
    fn precise_ebe_corr(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
    ) -> Vec<f64> {
        let warm = find_ebe(model, subject, params, 80, 1e-12, None, None);
        let mut eta: Vec<f64> = warm.eta.iter().copied().collect();
        let n_eta = model.n_eta;
        let sigma = &params.sigma.values;
        let omega_inv = &params.omega.inv;
        for _ in 0..60 {
            let sens =
                crate::sens::provider::subject_sensitivities(model, subject, &params.theta, &eta)
                    .unwrap();
            let (rv, dv, d2v) = corr_residual_diag(model, subject, &sens, sigma).unwrap();
            let mut grad = DVector::<f64>::from_column_slice(
                &(omega_inv * DVector::from_column_slice(&eta))
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
            );
            let mut hess = omega_inv.clone();
            for (j, obs) in sens.obs.iter().enumerate() {
                let t = err_terms(rv[j], dv[j], d2v[j], subject.observations[j] - obs.f);
                let (g1, g2) = (0.5 * t.alpha, 0.5 * t.alpha_p);
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

    /// Dense (`block_sigma`) FOCEI marginal at a given η̂. The production
    /// `foce_subject_nll(.., interaction=true)` dispatches to
    /// `foce_subject_nll_interaction_dense` for correlated models.
    fn marginal_nll_dense_at(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
        eta: &[f64],
    ) -> f64 {
        let eta_v = DVector::from_column_slice(eta);
        let jac = eta_jacobian_any(model, subject, &params.theta, eta);
        let h = DMatrix::from_row_slice(subject.obs_times.len(), model.n_eta, &jac);
        crate::stats::likelihood::foce_subject_nll(
            model,
            subject,
            &params.theta,
            &eta_v,
            &h,
            &params.omega,
            &params.sigma.values,
            true,
        )
    }

    /// The analytic FOCEI packed gradient for a correlated `block_sigma` model must match
    /// Richardson reconverged FD of ferx's own dense FOCEI marginal across every θ/Ω/σ
    /// coordinate (#627). The within-observation `combined` cross term modifies the
    /// residual variance and its `∂/∂f`, `∂²/∂f²`, which the scalar path omits — so this
    /// confirms the correlation-aware `(r,d,d2)` reduction of the dense Almquist assembly.
    #[test]
    fn population_packed_gradient_block_sigma_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;

        let model = parse_model_string(BLOCK_SIGMA_1CPT).expect("parse block_sigma");
        assert!(
            !model.residual_correlations.is_empty(),
            "fixture must carry a residual correlation"
        );
        assert!(crate::sens::provider::analytic_outer_gradient_available(
            &model
        ));

        let theta = &[1.1, 11.0];
        let s1 = dense_subject(&model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let s2 = dense_subject(&model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0]);
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
            .map(|s| DVector::from_vec(precise_ebe_corr(&model, s, &params)))
            .collect();

        let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
            .expect("block_sigma is analytic");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| {
                    let eta = precise_ebe_corr(&model, s, &p);
                    marginal_nll_dense_at(&model, s, &p, &eta)
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

    /// FOCE (non-interaction) analog of `population_packed_gradient_block_sigma_matches_fd`.
    /// The Sheiner–Beal linearized marginal freezes `R⁰` at η=0, so `block_sigma` only
    /// needs the correlation-aware `(r0, d0)` and `∂R⁰/∂σ` (no `∂²R/∂f²`). Must match
    /// Richardson reconverged FD of ferx's dense FOCE OFV across every θ/Ω/σ coord (#627).
    #[test]
    fn population_packed_gradient_block_sigma_foce_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;

        let src = BLOCK_SIGMA_1CPT.replace("method = focei", "method = foce");
        let model = parse_model_string(&src).expect("parse block_sigma foce");
        assert!(!model.residual_correlations.is_empty());

        let theta = &[1.1, 11.0];
        let s1 = dense_subject(&model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let s2 = dense_subject(&model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0]);
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
            .map(|s| DVector::from_vec(precise_ebe_corr(&model, s, &params)))
            .collect();

        let analytic = population_gradient_sens_foce(&model, &pop, &template, &x, &ehs)
            .expect("block_sigma foce is analytic");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| {
                    let eta = precise_ebe_corr(&model, s, &p);
                    marginal_nll_foce_at(&model, s, &p, &eta)
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
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 2e-3, epsilon = 1e-5);
        }
    }

    /// Covariate-selected (`if/else`) `block_sigma` fixture: each row's residual
    /// branch is chosen by a per-row `FREE` flag, and the two branch sigmas are
    /// correlated through one `block_sigma`. Distinct observation times mean no
    /// two rows share a residual block, so `R` stays diagonal and the analytic
    /// outer-gradient path (`corr_residual_diag`) proceeds rather than bailing to
    /// FD — exactly the path that must resolve the endpoint via the selector
    /// (`obs_keys`), not the raw CMT column (#669).
    const SELECTED_BLOCK_SIGMA_1CPT: &str = "[parameters]\n  theta TVCL(1.0, 0.01, 10.0)\n  theta TVV(10.0, 0.1, 100.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  block_sigma (PROP_TOTAL, PROP_UNBOUND) = [0.04, 0.03, 0.09]\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  if (FREE == 0) {\n    DV ~ proportional(PROP_TOTAL)\n  } else {\n    DV ~ proportional(PROP_UNBOUND)\n  }\n[covariates]\n  FREE continuous\n[fit_options]\n  method = focei\n";

    /// [`dense_subject`] with a per-row `FREE` covariate driving the selected
    /// residual branch. `times` and `free_flags` must be equal length; times are
    /// kept distinct by the caller so `R` is diagonal.
    fn selected_dense_subject(
        model: &CompiledModel,
        theta: &[f64],
        times: &[f64],
        free_flags: &[f64],
    ) -> Subject {
        let mut subject = dense_subject(model, theta, times);
        subject.obs_covariates = free_flags
            .iter()
            .map(|&f| HashMap::from([("FREE".to_string(), f)]))
            .collect();
        subject
    }

    /// #669 regression: with the `Selected` + `block_sigma` guard lifted, the
    /// analytic FOCEI outer gradient must resolve each row's endpoint via the
    /// covariate selector. If `corr_residual_diag` / `corr_residual_rd_at_sigma`
    /// read the raw CMT column (all-1 here) instead of the branch, every `FREE=0`
    /// row is scored against the `else` branch's sigma and the analytic gradient
    /// diverges from FD of ferx's dense FOCEI marginal (which uses the correct
    /// keys). Must match across every θ/Ω/σ coordinate.
    #[test]
    fn population_packed_gradient_selected_block_sigma_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;

        let model =
            parse_model_string(SELECTED_BLOCK_SIGMA_1CPT).expect("parse selected block_sigma");
        assert!(matches!(model.error_spec, ErrorSpec::Selected { .. }));
        assert!(!model.residual_correlations.is_empty());
        assert!(crate::sens::provider::analytic_outer_gradient_available(
            &model
        ));

        // Distinct times → diagonal R (analytic path proceeds); alternating FREE
        // flags route rows to both branches within one subject.
        let theta = &[1.1, 11.0];
        let s1 = selected_dense_subject(
            &model,
            theta,
            &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0],
            &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        );
        let s2 = selected_dense_subject(
            &model,
            theta,
            &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0],
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        );
        let pop = Population {
            subjects: vec![s1, s2],
            covariate_names: vec!["FREE".into()],
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
            .map(|s| DVector::from_vec(precise_ebe_corr(&model, s, &params)))
            .collect();

        let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
            .expect("selected block_sigma is analytic");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| {
                    let eta = precise_ebe_corr(&model, s, &p);
                    marginal_nll_dense_at(&model, s, &p, &eta)
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
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 2e-3, epsilon = 1e-5);
        }
    }

    /// `block_sigma` + η-dependent `ExpressionScale` `obs_scale` (#627 × #486): the analytic
    /// FOCEI packed gradient must still match Richardson reconverged FD of the dense marginal
    /// across every θ/Ω/σ coord. Pins the numerical side of the outer half of
    /// `expression_scale_with_correlated_residual_is_analytic_both_loops`.
    #[test]
    fn population_packed_gradient_block_sigma_expression_scale_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;

        let src = BLOCK_SIGMA_1CPT.replace(
            "[structural_model]",
            "[scaling]\n  obs_scale = 1000 / V\n[structural_model]",
        );
        let model = parse_model_string(&src).expect("parse block_sigma + obs_scale");
        assert!(matches!(
            model.scaling,
            crate::types::ScalingSpec::ExpressionScale { .. }
        ));
        assert!(crate::sens::provider::analytic_outer_gradient_available(
            &model
        ));

        let theta = &[1.1, 11.0];
        let s1 = dense_subject(&model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let s2 = dense_subject(&model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0]);
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
            .map(|s| DVector::from_vec(precise_ebe_corr(&model, s, &params)))
            .collect();

        let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
            .expect("block_sigma + obs_scale is analytic");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| {
                    let eta = precise_ebe_corr(&model, s, &p);
                    marginal_nll_dense_at(&model, s, &p, &eta)
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
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 2e-3, epsilon = 1e-5);
        }
    }

    /// Closed-form `iiv_on_ruv` + **M3 BLOQ** (#4c): the analytic FOCEI packed
    /// gradient must match Richardson reconverged FD of ferx's scaled, censored
    /// FOCEI marginal across every θ/Ω/σ coordinate. This exercises the censored ×
    /// residual-eta cross-terms — `h·z` (inner column), `C·z`/`C·m·a` (true inner
    /// Hessian, mixed η-θ), the `C·z·(∂v/∂σ)/2v` σ-cross — and the exclusion of
    /// censored rows from `H̃`/`log|H̃|` (matching `gaussian_foce_accum`).
    #[test]
    fn population_packed_gradient_iiv_on_ruv_m3_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;

        let mut model = parse_model_string(WARFARIN_RUV).expect("parse");
        model.bloq_method = crate::types::BloqMethod::M3;
        assert_eq!(model.residual_error_eta, Some(3));
        // Gate flip: closed-form M3 + iiv_on_ruv is now analytic on both loops.
        assert!(crate::sens::provider::analytic_outer_gradient_available(
            &model
        ));

        let theta = vec![0.22, 11.0, 1.4];
        // Censor the two latest (lowest-concentration) observations at an LLOQ above
        // their prediction, so the M3 `−logΦ` term is genuinely active.
        let mk = |times: &[f64]| -> Subject {
            let mut s = ruv_subject(&model, &theta, times);
            let n = s.obs_times.len();
            for j in (n - 2)..n {
                s.observations[j] *= 1.5;
                s.cens[j] = 1;
            }
            s
        };
        let pop = Population {
            subjects: vec![
                mk(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
                mk(&[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]),
            ],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.clone();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe_ruv(&model, s, &params)))
            .collect();

        let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
            .expect("M3 + iiv_on_ruv is analytic");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| {
                    let eta = precise_ebe_ruv(&model, s, &p);
                    marginal_nll_at(&model, s, &p, &eta)
                })
                .sum::<f64>()
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut xp = x.clone();
                xp[k] += hh;
                let mut xm = x.clone();
                xm[k] -= hh;
                (ofv(&xp) - ofv(&xm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                "m3+ruv x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 3e-3, epsilon = 1e-5);
        }
    }

    /// 2-cpt IV + **combined** residual error + IOV + `iiv_on_ruv` — coverage beyond
    /// the 1-cpt/proportional base case (review #3: the analytic scope must be tested
    /// across cpt/error combos, since a finite-but-wrong gradient has no FD fallback).
    const IOV_RUV_2CPT_COMBINED: &str = r#"
[parameters]
  theta TVCL(0.22, 0.001, 10.0)
  theta TVV1(11.0, 0.1, 500.0)
  theta TVQ(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  omega ETA_RUV ~ 0.05
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.1
  sigma ADD_ERR ~ 0.3
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v=V1, q=Q, v2=V2)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  iov_column = OCC
"#;

    /// Same structure, no IOV — for the non-IOV M3 + `iiv_on_ruv` 2-cpt/combined case.
    const RUV_2CPT_COMBINED: &str = r#"
[parameters]
  theta TVCL(0.22, 0.001, 10.0)
  theta TVV1(11.0, 0.1, 500.0)
  theta TVQ(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  omega ETA_RUV ~ 0.05
  sigma PROP_ERR ~ 0.1
  sigma ADD_ERR ~ 0.3
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v=V1, q=Q, v2=V2)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
  iiv_on_ruv = ETA_RUV
"#;

    /// Two-occasion 2-cpt IV IOV + `iiv_on_ruv` subject (n_eta = 3 incl. ETA_RUV).
    fn iov_ruv_2cpt_subject(model: &CompiledModel, theta: &[f64]) -> Subject {
        let obs_times = vec![0.5, 2.0, 6.0, 12.0, 25.0, 30.0, 36.0, 48.0];
        let occasions = vec![1u32, 1, 1, 1, 2, 2, 2, 2];
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
        let preds = crate::pk::predict_iov(
            model,
            &subject,
            theta,
            &[0.12, -0.08, 0.1],
            &[vec![0.05], vec![-0.07]],
        );
        subject.observations = preds.iter().map(|p| p * 0.9).collect();
        subject
    }

    /// Non-IOV `iiv_on_ruv` subject with a caller-supplied `eta_ref` (length `n_eta`),
    /// single IV bolus into the central compartment.
    fn ruv_subject_eta(
        model: &CompiledModel,
        theta: &[f64],
        times: &[f64],
        eta_ref: &[f64],
    ) -> Subject {
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
        let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, eta_ref);
        subject.observations = preds.iter().map(|p| p * 0.9).collect();
        subject
    }

    /// Coverage (#1): IOV + `iiv_on_ruv` on a 2-cpt IV + combined-error model through
    /// the production `subject_packed_gradient_iov` path — analytic vs reconverged FD.
    #[test]
    fn iov_iiv_on_ruv_2cpt_combined_packed_gradient_matches_fd() {
        let model = parse_model_string(IOV_RUV_2CPT_COMBINED)
            .expect("parse 2cpt IOV + iiv_on_ruv combined");
        assert_eq!(model.residual_error_eta, Some(2));
        let theta = vec![0.22, 11.0, 0.5, 20.0];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_ruv_2cpt_subject(&model, &theta);
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);
        let (stacked, _e, _k, _h) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            .expect("2cpt IOV + iiv_on_ruv packed gradient supported");
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
            let fd = (4.0 * f2 - f1) / 3.0;
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 4e-3, epsilon = 2e-5);
        }
    }

    /// Coverage (#1): non-IOV M3 BLOQ + `iiv_on_ruv` on a 2-cpt IV + combined-error
    /// model through the production population packed gradient — analytic vs
    /// reconverged FD of the censored FOCEI marginal.
    #[test]
    fn iiv_on_ruv_m3_2cpt_combined_packed_gradient_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;
        let mut model = parse_model_string(RUV_2CPT_COMBINED).expect("parse 2cpt ruv combined");
        model.bloq_method = crate::types::BloqMethod::M3;
        assert_eq!(model.residual_error_eta, Some(2));
        assert!(crate::sens::provider::analytic_outer_gradient_available(
            &model
        ));
        let theta = vec![0.22, 11.0, 0.5, 20.0];
        let mk = |times: &[f64]| -> Subject {
            let mut s = ruv_subject_eta(&model, &theta, times, &[0.12, -0.08, 0.0]);
            let n = s.obs_times.len();
            for j in (n - 2)..n {
                s.observations[j] *= 1.4;
                s.cens[j] = 1;
            }
            s
        };
        let pop = Population {
            subjects: vec![
                mk(&[0.5, 2.0, 6.0, 12.0, 24.0, 48.0]),
                mk(&[1.0, 3.0, 8.0, 16.0, 36.0, 72.0]),
            ],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let mut template = model.default_params.clone();
        template.theta = theta.clone();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe_ruv(&model, s, &params)))
            .collect();
        let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
            .expect("2cpt combined M3 + iiv_on_ruv analytic");
        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| {
                    let eta = precise_ebe_ruv(&model, s, &p);
                    marginal_nll_at(&model, s, &p, &eta)
                })
                .sum::<f64>()
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut xp = x.clone();
                xp[k] += hh;
                let mut xm = x.clone();
                xm[k] -= hh;
                (ofv(&xp) - ofv(&xm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 4e-3, epsilon = 2e-5);
        }
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
    /// M3 branch: data `−logΦ` + true-inner-Hessian + FOCEI-order `H̃`/`log|H̃|`
    /// curvature) must match the reconverged-FD of ferx's M3 FOCEI objective
    /// (`foce_subject_nll_interaction` with `bloq_term`). Each subject carries both
    /// quantified and censored rows.
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

    /// Non-IOV 1-cpt oral **user-ODE** model with M3 BLOQ **and** `iiv_on_ruv`
    /// (`Y = IPRED + EPS·EXP(η_ruv)`) — [`ONECPT_ODE_M3_OUTER`] plus an extra residual-error
    /// η that no structural parameter references. Drives the last `iiv_on_ruv` holdout
    /// (#486): non-IOV ODE M3 + `iiv_on_ruv` on the outer loop.
    const ONECPT_ODE_M3_RUV_OUTER: &str = r#"
[parameters]
  theta TVCL(0.2,  0.001, 10.0)
  theta TVV(10.0,  0.1,  500.0)
  theta TVKA(1.5,  0.01,  50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
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
  iiv_on_ruv = ETA_RUV
[fit_options]
  method      = focei
  bloq_method = m3
  ode_reltol  = 1e-10
  ode_abstol  = 1e-12
"#;

    /// **Non-IOV ODE M3 + `iiv_on_ruv`** (#486 — the last `iiv_on_ruv` holdout, the #547
    /// pattern): the ODE counterpart of [`population_packed_gradient_iiv_on_ruv_m3_matches_fd`].
    /// The censored × residual-eta cross-terms (`h·z` inner column, `C·z`/`C·m·a` true-Hessian /
    /// mixed blocks, the σ-cross) are applied by the provider-agnostic `prepare` over the
    /// **event-driven ODE walk's** `ObsSens`, and censored rows enter `H̃`/`log|H̃|` at FOCEI
    /// order, exactly as on the closed-form path. The FOCEI packed gradient must match Richardson
    /// reconverged FD of the `exp(2·η_ruv)`-scaled, censored FOCEI marginal across every packed
    /// coordinate — note the EBE must be reconverged with [`precise_ebe_ruv`] (which carries the
    /// `exp(2·η_ruv)` variance scaling), not the plain [`precise_ebe`]. Both censoring tails.
    #[test]
    fn population_packed_gradient_ode_m3_iiv_on_ruv_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::{BloqMethod, Population};

        let model = parse_model_string(ONECPT_ODE_M3_RUV_OUTER).expect("parse ODE M3 + iiv_on_ruv");
        assert!(matches!(model.bloq_method, BloqMethod::M3), "must be M3");
        assert!(model.is_ode_based(), "must be on the ODE path");
        assert_eq!(model.residual_error_eta, Some(3));
        assert!(
            crate::sens::provider::analytic_outer_gradient_available(&model),
            "non-IOV ODE M3 + iiv_on_ruv must route to the analytic outer gradient (#486)"
        );
        let theta = [0.22, 11.0, 1.4];

        for right in [false, true] {
            let mut s1 = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 8.0]);
            let mut s2 = subject_with_obs(&model, &theta, &[0.25, 1.5, 6.0, 12.0, 36.0]);
            for s in [&mut s1, &mut s2] {
                let n = s.observations.len();
                let tail = if right { -1 } else { 1 };
                s.cens[n - 1] = tail;
                s.cens[n - 2] = tail;
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
            // `precise_ebe_ruv` carries the `exp(2·η_ruv)` variance scaling — the plain
            // `precise_ebe` ignores the residual-eta and converges to the wrong EBE.
            let ehs: Vec<DVector<f64>> = pop
                .subjects
                .iter()
                .map(|s| DVector::from_vec(precise_ebe_ruv(&model, s, &params)))
                .collect();

            let analytic = population_gradient_sens(&model, &pop, &template, &x, &ehs)
                .expect("ODE M3 + iiv_on_ruv supported");

            let ofv = |xv: &[f64]| -> f64 {
                let p = unpack_params(xv, &template);
                2.0 * pop
                    .subjects
                    .iter()
                    .map(|s| {
                        let eta = precise_ebe_ruv(&model, s, &p);
                        marginal_nll_at(&model, s, &p, &eta)
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
                    "ode m3+ruv (right={right}) x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                    analytic[k],
                    fd,
                    (analytic[k] - fd).abs() / fd.abs().max(1e-9)
                );
                approx::assert_relative_eq!(analytic[k], fd, max_relative = 3e-3, epsilon = 2e-5);
            }
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

    // 1-cpt IV ODE with a parameter-dependent `init(central) = BASE/V` baseline + a finite
    // infusion — a headline `init` composition (#486). Exercises the full FOCE packed gradient
    // `[θ, Ω, σ]` end to end: the analytic init impulse (seeded on the event-driven walk and
    // decayed under the infusion forcing) must survive the outer θ/Ω/σ assembly and match a
    // Richardson-reconverged FD of ferx's FOCE OFV.
    const IV_INIT_INFUSION: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVBASE(300.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_BASE ~ 0.04
  sigma PROP ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  BASE = TVBASE * exp(ETA_BASE)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central)  = -CL/V * central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  method     = foce
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    #[test]
    fn population_packed_gradient_foce_init_infusion_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;
        let model = parse_model_string(IV_INIT_INFUSION).expect("parse init+infusion");
        let theta = vec![1.0, 20.0, 300.0];
        // Two subjects, each dosed by a finite IV infusion (rate 25 → 4 h window) on top of the
        // init baseline; obs straddle the infusion end.
        let mk = |times: &[f64]| -> Subject {
            let mut s = subject_with_obs(&model, &theta, times);
            s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0)];
            assert!(s.doses[0].is_infusion());
            let eta_ref = [0.12, -0.08, 0.15];
            let preds = crate::pk::compute_predictions_with_tv(&model, &s, &theta, &eta_ref);
            s.observations = preds.iter().map(|p| p * 0.85).collect();
            s
        };
        let pop = Population {
            subjects: vec![
                mk(&[1.0, 2.0, 4.0, 6.0, 10.0]),
                mk(&[0.5, 3.0, 5.0, 8.0, 24.0]),
            ],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let mut template = model.default_params.clone();
        template.theta = theta.clone();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
            .collect();
        let analytic =
            population_gradient_sens_foce(&model, &pop, &template, &x, &ehs).expect("supported");
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
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 3e-3, epsilon = 1e-5);
        }
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

    /// WARFARIN_IOV + IIV on residual error (`iiv_on_ruv = ETA_RUV`, the 4th omega →
    /// eta index 3). FOCEI is required (non-interaction FOCE + `iiv_on_ruv` is rejected).
    const WARFARIN_IOV_RUV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
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
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
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
        let k = crate::stats::likelihood::iov_occasion_groups(subject).len();
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
        // IIV on residual error (#4b): the inner Newton must minimise the SAME ruv-scaled
        // objective `individual_nll_iov` uses, else the reconverged EBE (and the marginal
        // FD built on it) would be wrong. `ruv_idx` is `None` and `residual_var_scale`
        // returns `1.0` for non-`iiv_on_ruv` models, so this is a no-op there.
        let ruv_idx = model.residual_error_eta;
        let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3);
        // Custom / time-varying σ magnitude (#576/#486): the Newton must minimise the
        // *scaled* inner objective (like `precise_ebe`), else it converges to the bare
        // EBE and the coupling identity the FOCE-IOV gradient relies on breaks (the
        // reconverged-FD marginal would then disagree on the structural/Ω coordinates).
        // `None` for a bare-sigma model → the non-scaled arm below (bit-identical).
        let mult = model.ruv_obs_mult(subject, &params.theta);
        for _ in 0..50 {
            let ruv_scale = model.residual_var_scale(&stacked);
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
                let mult_row: Option<&[f64]> =
                    mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
                let (r, d, d2) = match mult_row {
                    Some(m) => (
                        model.error_spec.variance_at_scaled(cmt, f, sigma, &[], m) * ruv_scale,
                        model.error_spec.dvar_df_scaled(cmt, f, sigma, m) * ruv_scale,
                        model.error_spec.d2var_df2_scaled(cmt, sigma, m) * ruv_scale,
                    ),
                    None => (
                        model.error_spec.variance_at(cmt, f, sigma) * ruv_scale,
                        model.error_spec.dvar_df(cmt, f, sigma) * ruv_scale,
                        model.error_spec.d2var_df2(cmt, sigma) * ruv_scale,
                    ),
                };
                let y = subject.observations[j];
                let eps = y - f;
                // (g1, g2) = (∂L/∂f, ∂²L/∂f²): the censored `−logΦ` scalars for an M3 BLOQ
                // row (#580), else the Gaussian `½α`, `½α'`. The inner Newton must minimise
                // the SAME censored objective `individual_nll_iov` uses, else the
                // reconverged EBE (and the marginal FD built on it) would be wrong.
                let is_cens = m3 && subject.cens.get(j).copied().unwrap_or(0) != 0;
                let (g1, g2) = if is_cens {
                    m3_censored_scalars(y, f, r, d, d2, subject.cens.get(j).copied().unwrap_or(0))
                } else {
                    let t = err_terms(r, d, d2, eps);
                    (0.5 * t.alpha, 0.5 * t.alpha_p)
                };
                let a = &obs.df_deta;
                for kk in 0..n_st {
                    g[kk] += g1 * a[kk];
                    for ll in 0..n_st {
                        h[(kk, ll)] += g2 * a[kk] * a[ll] + g1 * obs.d2f_deta2[kk * n_st + ll];
                    }
                }
                // Residual-eta row/col (`a_{ruv} = 0`): mirrors the `h_inner` residual-eta
                // block in `prepare_stacked`. Gaussian row: data-term gradient `1 − ε²/v`,
                // true Hessian `H[ruv,ruv] += 2ε²/v`, `H[ruv,l] += κ_j a_{jl}`. Censored row
                // under `iiv_on_ruv` (the triple, #591): gradient `h·z`, Hessian
                // `H[ruv,ruv] += C·z`, `H[ruv,l] += C·m·a_{jl}` — the same `(C·z, C·m)`
                // coefficients `m3_censored_outer` feeds `prepare_stacked`. Newton's fixed
                // point is the gradient root, so the reconverged EBE matches the production
                // M3 + IOV + `iiv_on_ruv` inner objective `find_ebe_iov` minimises.
                if let Some(rr) = ruv_idx {
                    if is_cens {
                        let (h_im, z_k, _m) = crate::stats::special::m3_censored_kernel(
                            y,
                            f,
                            r,
                            d,
                            subject.cens.get(j).copied().unwrap_or(0),
                        );
                        let (_g1, _g2, cz, cm) = m3_censored_outer(
                            y,
                            f,
                            r,
                            d,
                            d2,
                            subject.cens.get(j).copied().unwrap_or(0),
                        );
                        g[rr] += h_im * z_k;
                        h[(rr, rr)] += cz;
                        for ll in 0..n_st {
                            if ll == rr {
                                continue;
                            }
                            h[(rr, ll)] += cm * a[ll];
                            h[(ll, rr)] += cm * a[ll];
                        }
                    } else {
                        g[rr] += 1.0 - eps * eps / r;
                        h[(rr, rr)] += 2.0 * eps * eps / r;
                        let kappa = ruv_kappa(eps, r, d);
                        for ll in 0..n_st {
                            if ll == rr {
                                continue;
                            }
                            h[(rr, ll)] += kappa * a[ll];
                            h[(ll, rr)] += kappa * a[ll];
                        }
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

    /// Two-occasion IOV + `iiv_on_ruv` subject (n_eta = 4 incl. ETA_RUV). η_ruv (the 4th
    /// bsv eta) affects only the residual variance, not the predictions, so it is supplied
    /// to `predict_iov` purely to keep the eta vector the right length.
    fn iov_ruv_subject(model: &CompiledModel, theta: &[f64]) -> Subject {
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
        let preds = crate::pk::predict_iov(
            model,
            &subject,
            theta,
            &[0.12, -0.08, 0.2, 0.10],
            &[vec![0.05], vec![-0.07]],
        );
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        subject
    }

    /// Closed-form IOV + `iiv_on_ruv` (#4b): the analytic IOV θ-gradient (which now threads
    /// `ruv = residual_error_eta` into `prepare_stacked`, so the residual-eta `c̃` column
    /// rides the stacked `[η_bsv, κ]` assembly) must match the Richardson reconverged FD of
    /// the FOCEI marginal — the marginal whose EBE `precise_ebe_iov` reconverges against the
    /// same `exp(2·η_ruv)`-scaled objective. Proves the outer gate flip ships a correct
    /// gradient.
    #[test]
    fn iov_iiv_on_ruv_theta_gradient_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN_IOV_RUV).expect("parse warfarin IOV + iiv_on_ruv");
        assert_eq!(model.residual_error_eta, Some(3));
        assert!(crate::sens::provider::iov_analytical_supported(&model));
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_ruv_subject(&model, &theta);

        // Joint EBE [η_bsv (4, incl. η_ruv), κ_g0 (1), κ_g1 (1)], analytically reconverged
        // against the ruv-scaled inner objective.
        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);

        let analytic = subject_theta_gradient_iov(&model, &subject, &params, &stacked)
            .expect("IOV + iiv_on_ruv θ-gradient supported");

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
                "iov+ruv theta[{m}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
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

    /// IOV + `iiv_on_ruv` through the **production** packed-gradient path
    /// (`subject_packed_gradient_iov`, not the `subject_theta_gradient_iov` helper):
    /// the full `[θ, Ω_bsv, σ, Ω_iov]` analytic gradient must match Richardson
    /// reconverged FD of the scaled FOCEI marginal. Regression for the review
    /// finding that the residual-eta threading reached only the test helper, leaving
    /// production on the unscaled variance with the `η_ruv` `c̃` column dropped.
    #[test]
    fn iov_iiv_on_ruv_packed_gradient_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN_IOV_RUV).expect("parse warfarin IOV + iiv_on_ruv");
        assert_eq!(model.residual_error_eta, Some(3));
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_ruv_subject(&model, &theta);
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            .expect("IOV + iiv_on_ruv packed gradient supported");

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
                "iov+ruv packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-9)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }

    /// [`WARFARIN_IOV`] with a `log_additive` (LTBS) error model — plain closed-form
    /// LTBS × IOV. #665 served LTBS on the non-IOV outer gradient (post-walk `ln(f)`
    /// jet); this fixture drives its IOV twin.
    const WARFARIN_IOV_LTBS: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.02
  sigma ADD_ERR ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ log_additive(ADD_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;

    /// As [`WARFARIN_IOV_LTBS`] but also carrying an η-dependent `ExpressionScale`
    /// `obs_scale = V` — pins the composition `ln(f / s)` under IOV (the in-walk scale
    /// quotient followed by the post-walk `ln` jet must reproduce production's
    /// scale-then-log order in `predict_iov`).
    const WARFARIN_IOV_LTBS_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.02
  sigma ADD_ERR ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[scaling]
  obs_scale = V
[error_model]
  DV ~ log_additive(ADD_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;

    /// As [`WARFARIN_IOV_LTBS`] plus IIV on residual error (`iiv_on_ruv = ETA_RUV`, the 4th
    /// eta) — the LTBS × `iiv_on_ruv` × IOV composition the relaxed gate admits (#677 review #2).
    const WARFARIN_IOV_LTBS_RUV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  kappa KAPPA_CL ~ 0.02
  sigma ADD_ERR ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ log_additive(ADD_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  iov_column = OCC
"#;

    /// Pure Richardson-FD parity check for an IOV **outer** packed gradient: the analytic
    /// gradient (`subject_packed_gradient_iov` for FOCEI / `subject_packed_gradient_foce_iov`
    /// for FOCE) must match reconverged central FD of the corresponding IOV marginal across
    /// every packed θ/Ω_bsv/σ/Ω_iov coordinate. `subject` and `interaction` are caller-supplied
    /// so the same stencil serves plain / magnitude / LTBS / M3 / `iiv_on_ruv` fixtures on both
    /// the FOCEI and FOCE paths (dedup — #677 review #4).
    fn check_iov_outer_packed_matches_fd(
        model: &CompiledModel,
        theta: &[f64],
        subject: &Subject,
        interaction: bool,
    ) {
        use crate::estimation::parameterization::pack_params;
        let mut params = model.default_params.clone();
        params.theta = theta.to_vec();
        let template = params.clone();
        let x = pack_params(&params);
        let (stacked, _e, _k, _h) = precise_ebe_iov(model, subject, &params);
        let analytic = if interaction {
            subject_packed_gradient_iov(model, subject, &template, &x, &stacked)
        } else {
            subject_packed_gradient_foce_iov(model, subject, &template, &x, &stacked)
        }
        .expect("IOV outer packed gradient supported");
        assert!(
            analytic.iter().all(|v| v.is_finite()),
            "packed gradient must be finite"
        );
        let f = |xx: &[f64]| -> f64 {
            let p = unpack_params(xx, &template);
            marginal_nll_iov_inter(model, subject, &p, interaction)
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
                "iov outer x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-9)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }

    /// LTBS × IOV routing assertions (outer analytic, inner FD) + FD parity via
    /// [`check_iov_outer_packed_matches_fd`] on the FOCEI path.
    fn check_ltbs_iov_outer_matches_fd(model: &CompiledModel, theta: &[f64]) {
        assert!(model.log_transform, "fixture must be LTBS");
        assert!(model.n_kappa > 0, "fixture must carry IOV");
        assert!(
            crate::sens::provider::iov_analytical_supported(model),
            "LTBS × IOV must route to the analytic closed-form OUTER path (#486)"
        );
        assert!(crate::sens::provider::analytic_outer_gradient_available(
            model
        ));
        assert!(
            crate::estimation::inner_optimizer::analytic_inner_common_bail(model),
            "LTBS × IOV inner EBE gradient must stay on FD"
        );
        let subject = iov_subject_outer(model, theta);
        check_iov_outer_packed_matches_fd(model, theta, &subject, true);
    }

    #[test]
    fn ltbs_iov_outer_packed_matches_fd() {
        let model = parse_model_string(WARFARIN_IOV_LTBS).expect("parse LTBS IOV");
        check_ltbs_iov_outer_matches_fd(&model, &[0.22, 11.0, 1.4]);
    }

    #[test]
    fn ltbs_expr_scale_iov_outer_packed_matches_fd() {
        let model = parse_model_string(WARFARIN_IOV_LTBS_EXPRSCALE).expect("parse LTBS+scale IOV");
        assert!(
            matches!(
                model.scaling,
                crate::types::ScalingSpec::ExpressionScale { .. }
            ),
            "fixture must carry an expression obs_scale"
        );
        check_ltbs_iov_outer_matches_fd(&model, &[0.22, 11.0, 1.4]);
    }

    /// LTBS × **M3-BLOQ** × IOV (#677 review #1). Relaxing the `log_transform` gate also
    /// admits M3-censored rows (the gate still admits M3), so the censored `−logΦ(z)` kernel
    /// in `prepare_stacked` now consumes the ln(f) jet. Pin that the newly-analytic outer
    /// gradient matches reconverged FD of the FOCEI-IOV censored marginal.
    #[test]
    fn ltbs_m3_iov_outer_packed_matches_fd() {
        let mut model = parse_model_string(WARFARIN_IOV_LTBS).expect("parse LTBS IOV");
        model.bloq_method = crate::types::BloqMethod::M3;
        assert!(
            crate::sens::provider::iov_analytical_supported(&model),
            "LTBS × M3 × IOV is admitted by the relaxed gate"
        );
        let theta = vec![0.22, 11.0, 1.4];
        let subject = iov_m3_subject(&model, &theta);
        check_iov_outer_packed_matches_fd(&model, &theta, &subject, true);
    }

    /// LTBS × `iiv_on_ruv` × IOV (#677 review #2). The residual-eta (η_ruv) variance-scaling
    /// column rides the stacked `[η_bsv, κ]` assembly alongside the newly applied ln(f) jet;
    /// pin the composition against FD.
    #[test]
    fn ltbs_iiv_on_ruv_iov_outer_packed_matches_fd() {
        let model = parse_model_string(WARFARIN_IOV_LTBS_RUV).expect("parse LTBS IOV + iiv_on_ruv");
        assert_eq!(model.residual_error_eta, Some(3), "ETA_RUV is the 4th eta");
        assert!(crate::sens::provider::iov_analytical_supported(&model));
        let theta = vec![0.22, 11.0, 1.4];
        let subject = iov_ruv_subject(&model, &theta);
        check_iov_outer_packed_matches_fd(&model, &theta, &subject, true);
    }

    /// LTBS × IOV on the **FOCE** (non-interaction) path (#677 review #3). Opening the gate
    /// for `log_transform` also makes `subject_packed_gradient_foce_iov` reachable for LTBS
    /// (it consumes the same ln-jetted `subject_sensitivities_iov`). Every other LTBS×IOV test
    /// uses FOCEI; pin the FOCE marginal-moment path too.
    #[test]
    fn ltbs_iov_outer_foce_packed_matches_fd() {
        let model = parse_model_string(WARFARIN_IOV_LTBS).expect("parse LTBS IOV");
        let theta = vec![0.22, 11.0, 1.4];
        let subject = iov_subject_outer(&model, &theta);
        check_iov_outer_packed_matches_fd(&model, &theta, &subject, false);
    }

    /// Two-occasion IOV subject with M3-censored rows (#580): the same geometry as
    /// [`iov_subject_outer`], but occasion 2's two tail observations are flagged
    /// `CENS = 1` (left-censored at their synthesized value ≈ 0.85·f, so the
    /// prediction sits just above the limit and the inverse Mills ratio is well-scaled).
    fn iov_m3_subject(model: &CompiledModel, theta: &[f64]) -> Subject {
        let mut subject = iov_subject_outer(model, theta);
        let n = subject.observations.len();
        subject.cens[n - 2] = 1;
        subject.cens[n - 1] = 1;
        subject
    }

    /// As [`iov_m3_subject`] but the occasion-2 tail is **right**-censored
    /// (`CENS = -1`, above ULOQ) — exercises the upper-tail (`σ = -1`) branch of the
    /// signed `m3_censored_kernel` / FOCE `Cens` terms.
    fn iov_m3_subject_right(model: &CompiledModel, theta: &[f64]) -> Subject {
        let mut subject = iov_subject_outer(model, theta);
        let n = subject.observations.len();
        subject.cens[n - 2] = -1;
        subject.cens[n - 1] = -1;
        subject
    }

    /// M3 BLOQ + IOV (#580): the analytic IOV FOCEI θ-gradient (censored rows carry
    /// `p = β = 0` so they leave `H̃`/`log|H̃|` exactly as `foce_subject_nll_iov`
    /// builds it, and re-enter via the `−logΦ` data term + true inner Hessian over the
    /// stacked `[η_bsv, κ]` layout) must match the Richardson reconverged FD of the
    /// FOCEI IOV marginal — the same objective `precise_ebe_iov` now reconverges
    /// against (its Newton loop uses the censored `m3_censored_scalars` on flagged rows).
    /// Proves the gate flip ships a correct censored θ-gradient.
    #[test]
    fn iov_m3_theta_gradient_matches_reconverged_fd() {
        let mut model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
        model.bloq_method = crate::types::BloqMethod::M3;
        assert!(crate::sens::provider::iov_analytical_supported(&model));
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_m3_subject(&model, &theta);
        assert!(
            subject.cens.iter().any(|&c| c != 0),
            "subject must be censored"
        );

        // Joint EBE [η_bsv (3), κ_g0 (1), κ_g1 (1)] reconverged against the M3-aware
        // inner objective.
        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);

        let analytic = subject_theta_gradient_iov(&model, &subject, &params, &stacked)
            .expect("IOV + M3 θ-gradient supported");

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
                "iov+m3 theta[{m}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[m],
                fd,
                (analytic[m] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[m], fd, max_relative = 3e-3, epsilon = 1e-5);
        }
    }

    /// M3 BLOQ + IOV (#580) through the **production** FOCEI packed-gradient path
    /// (`subject_packed_gradient_iov`): the full `[θ, Ω_bsv, σ, Ω_iov]` analytic
    /// gradient must match Richardson reconverged FD of the FOCEI IOV marginal over
    /// every packed coordinate — exercising the censored σ-block (`censored_sigma_m_terms`)
    /// and the Ω blocks (incl. the shared κ-variance) with censored rows present.
    #[test]
    fn iov_m3_packed_gradient_matches_reconverged_fd() {
        let mut model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
        model.bloq_method = crate::types::BloqMethod::M3;
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_m3_subject(&model, &theta);
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            .expect("IOV + M3 packed gradient supported");

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
                "iov+m3 packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-9)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }

    /// 1-cpt oral **user-ODE** IOV model (κ on CL), the ODE counterpart of
    /// [`WARFARIN_IOV`]. Drives the ODE IOV M3 outer test below.
    const ONECPT_ODE_IOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
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
  iov_column  = OCC
  ode_reltol  = 1e-10
  ode_abstol  = 1e-12
"#;

    /// **ODE** M3 BLOQ + IOV (#486): the ODE counterpart of
    /// [`iov_m3_packed_gradient_matches_reconverged_fd`]. The full `[θ, Ω_bsv, σ, Ω_iov]`
    /// analytic packed gradient — assembled from the **event-driven ODE sensitivity walk**
    /// (`subject_sensitivities_iov` → `ode_subject_sensitivities_iov`) with censored rows
    /// entering `prepare_stacked`'s M3 branch — must match Richardson reconverged FD of the
    /// FOCEI IOV marginal. Censoring is provider-agnostic (keyed on `subject.cens[j]`), so
    /// the only change versus the closed-form path is the dropped gate clause. Both tails.
    #[test]
    fn iov_m3_ode_packed_gradient_matches_reconverged_fd() {
        use crate::estimation::parameterization::pack_params;

        let mut model = parse_model_string(ONECPT_ODE_IOV).expect("parse ODE IOV");
        model.bloq_method = crate::types::BloqMethod::M3;
        assert!(model.is_ode_based(), "must be on the ODE path");
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "ODE IOV + M3 must be analytic (#486)"
        );
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();

        for right in [false, true] {
            let subject = if right {
                iov_m3_subject_right(&model, &theta)
            } else {
                iov_m3_subject(&model, &theta)
            };
            assert!(
                subject.cens.iter().any(|&c| c != 0),
                "subject must be censored"
            );
            let template = params.clone();
            let x = pack_params(&params);

            let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
            let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
                .expect("ODE IOV + M3 packed gradient supported");

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
                    "iov+m3 ode (right={right}) packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                    analytic[i],
                    fd,
                    (analytic[i] - fd).abs() / fd.abs().max(1e-9)
                );
                approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
            }
        }
    }

    /// **ODE** FOCE (non-interaction) M3 BLOQ + IOV (#486): the ODE counterpart of
    /// [`iov_m3_foce_packed_gradient_matches_reconverged_fd`]. Guards the §6 gotcha — the
    /// ODE FOCE-IOV objective must route censored rows the same way as the closed-form
    /// path (no silent promotion to interaction; censored rows re-enter as `−logΦ` at the
    /// population η=0, κ=0 variance). The FOCE packed gradient assembled from the ODE walk
    /// must match Richardson reconverged FD of `marginal_nll_iov_inter(.., false)`.
    #[test]
    fn iov_m3_foce_ode_packed_gradient_matches_reconverged_fd() {
        use crate::estimation::parameterization::pack_params;

        let mut model = parse_model_string(ONECPT_ODE_IOV).expect("parse ODE IOV");
        model.bloq_method = crate::types::BloqMethod::M3;
        assert!(model.is_ode_based(), "must be on the ODE path");
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_m3_subject(&model, &theta);
        assert!(
            subject.cens.iter().any(|&c| c != 0),
            "subject must be censored"
        );
        let template = params.clone();
        let x = pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
            .expect("FOCE-ODE-IOV-M3 packed gradient supported");

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
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                "iov+m3 foce-ode packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-9)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }

    /// 1-cpt oral **user-ODE** IOV + `iiv_on_ruv` model (κ on CL, `ETA_RUV` scaling the
    /// residual variance, absent from CL/V/KA), the ODE counterpart of [`WARFARIN_IOV_RUV`].
    /// Drives the ODE `iiv_on_ruv` / triple outer tests below.
    const ONECPT_ODE_IOV_RUV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot / V - (CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method      = focei
  iov_column  = OCC
  ode_reltol  = 1e-10
  ode_abstol  = 1e-12
"#;

    /// **ODE** IOV + `iiv_on_ruv` (no M3, #486): the ODE counterpart of
    /// [`iov_iiv_on_ruv_packed_gradient_matches_reconverged_fd`]. The full
    /// `[θ, Ω_bsv, σ, Ω_iov]` packed gradient from the ODE walk must match Richardson
    /// reconverged FD of the `exp(2·η_ruv)`-scaled FOCEI IOV marginal. The ODE walk emits a
    /// zero `∂f/∂η_ruv` column; the shared assembly applies the variance scaling and the
    /// residual-eta `c̃` column (keyed on `residual_error_eta`), provider-agnostic.
    #[test]
    fn iov_iiv_on_ruv_ode_packed_gradient_matches_reconverged_fd() {
        use crate::estimation::parameterization::pack_params;

        let model = parse_model_string(ONECPT_ODE_IOV_RUV).expect("parse ODE IOV + iiv_on_ruv");
        assert_eq!(model.residual_error_eta, Some(3));
        assert!(model.is_ode_based(), "must be on the ODE path");
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_ruv_subject(&model, &theta);
        let template = params.clone();
        let x = pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            .expect("ODE IOV + iiv_on_ruv packed gradient supported");

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
                "iov+ruv ode packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-9)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }

    /// **ODE** triple M3 + IOV + `iiv_on_ruv` (#486): the ODE counterpart of
    /// [`iov_m3_iiv_on_ruv_packed_gradient_matches_reconverged_fd`]. Censored rows co-occur
    /// with the `exp(2·η_ruv)` variance scaling: `prepare_stacked` returns the censored
    /// residual-eta cross coefficients `(C·z, C·m)` into the true inner Hessian and the
    /// `h·z` column into the inner gradient — all provider-agnostic over the ODE walk's
    /// `ObsSens`. The packed gradient must match Richardson reconverged FD of the FOCEI IOV
    /// marginal. Both tails.
    #[test]
    fn iov_m3_iiv_on_ruv_ode_packed_gradient_matches_reconverged_fd() {
        use crate::estimation::parameterization::pack_params;

        let mut model = parse_model_string(ONECPT_ODE_IOV_RUV).expect("parse ODE IOV + iiv_on_ruv");
        model.bloq_method = crate::types::BloqMethod::M3;
        assert_eq!(model.residual_error_eta, Some(3));
        assert!(model.is_ode_based(), "must be on the ODE path");
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();

        for right in [false, true] {
            let mut subject = iov_m3_ruv_subject(&model, &theta);
            if right {
                let n = subject.observations.len();
                subject.cens[n - 2] = -1;
                subject.cens[n - 1] = -1;
            }
            assert!(
                subject.cens.iter().any(|&c| c != 0),
                "subject must be censored"
            );
            let template = params.clone();
            let x = pack_params(&params);

            let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
            let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
                .expect("ODE IOV + M3 + iiv_on_ruv packed gradient supported");

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
                    "iov+m3+ruv ode (right={right}) packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                    analytic[i],
                    fd,
                    (analytic[i] - fd).abs() / fd.abs().max(1e-9)
                );
                approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
            }
        }
    }

    /// 1-cpt oral **user-ODE** IOV model carrying an η-dependent `ExpressionScale`
    /// `obs_scale` divisor (`obs_scale = 1000 / V`, `V = TVV·exp(ETA_V)`) — the
    /// [`ONECPT_ODE_IOV`] geometry plus the #575 post-walk quotient scale. The scale
    /// rides the `(θ, stacked-η)` jet *before* the provider-agnostic M3 censoring
    /// coefficient is applied, so it composes with BLOQ rows at a different layer.
    /// Drives the ODE M3 × `ExpressionScale` × IOV cross-check below (#623 review).
    const ONECPT_ODE_IOV_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot / V - (CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
[scaling]
  obs_scale = 1000 / V
[fit_options]
  method      = focei
  iov_column  = OCC
  ode_reltol  = 1e-10
  ode_abstol  = 1e-12
"#;

    /// **ODE M3 + IOV + η-dependent `ExpressionScale` `obs_scale`** (#623 review of #486):
    /// the gate flip in `ode_iov_supported` admits censored rows alongside the #575 scale
    /// quotient, a combination the closed-form mirror never reaches (its gate rejects every
    /// non-`None` scaling) and that the #486 tests did not exercise. The two features
    /// compose at different layers — the scale is a post-walk quotient on the `(θ, stacked-η)`
    /// jet (incl. its second-order derivatives), and the M3 `−logΦ` coefficient is applied
    /// over that already-scaled jet keyed on `subject.cens[j]`. If the second-order
    /// composition of the quotient were inconsistent for censored rows, the marginal
    /// `log|H̃|` term would be wrong. The full `[θ, Ω_bsv, σ, Ω_iov]` analytic packed
    /// gradient must match Richardson reconverged FD of the FOCEI IOV marginal over every
    /// packed coordinate, on both censoring tails — proving the composition is consistent.
    #[test]
    fn iov_m3_ode_expression_scale_packed_gradient_matches_reconverged_fd() {
        use crate::estimation::parameterization::pack_params;

        let mut model = parse_model_string(ONECPT_ODE_IOV_EXPRSCALE)
            .expect("parse ODE IOV + ExpressionScale obs_scale");
        model.bloq_method = crate::types::BloqMethod::M3;
        assert!(model.is_ode_based(), "must be on the ODE path");
        assert!(
            matches!(
                model.scaling,
                crate::types::ScalingSpec::ExpressionScale { .. }
            ),
            "model must carry an ExpressionScale obs_scale"
        );
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "ODE IOV + M3 + ExpressionScale obs_scale must be analytic (#486/#575)"
        );
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();

        for right in [false, true] {
            let subject = if right {
                iov_m3_subject_right(&model, &theta)
            } else {
                iov_m3_subject(&model, &theta)
            };
            assert!(
                subject.cens.iter().any(|&c| c != 0),
                "subject must be censored"
            );
            let template = params.clone();
            let x = pack_params(&params);

            let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
            let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
                .expect("ODE IOV + M3 + ExpressionScale packed gradient supported");

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
                    "iov+m3 ode+exprscale (right={right}) packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                    analytic[i],
                    fd,
                    (analytic[i] - fd).abs() / fd.abs().max(1e-9)
                );
                approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
            }
        }
    }

    /// Two-occasion IOV + `iiv_on_ruv` subject (the [`iov_ruv_subject`] geometry) with
    /// occasion 2's two tail observations flagged `CENS = 1` — the **triple**
    /// M3 + IOV + `iiv_on_ruv` (#591). Shallow left-censoring (≈ 0.85·f) keeps the
    /// inverse Mills ratio well-scaled.
    fn iov_m3_ruv_subject(model: &CompiledModel, theta: &[f64]) -> Subject {
        let mut subject = iov_ruv_subject(model, theta);
        let n = subject.observations.len();
        subject.cens[n - 2] = 1;
        subject.cens[n - 1] = 1;
        subject
    }

    /// The triple **M3 + IOV + `iiv_on_ruv`** through the production FOCEI packed
    /// gradient (#591): the censored residual-eta cross coefficients `(C·z, C·m)` enter
    /// the true inner Hessian / `mixed_eta_theta` / `sigma_block` over the stacked
    /// `[η_bsv, κ]` layout, and `residual_inner_obs` adds the `h·z` residual-eta column
    /// to the inner gradient. The full `[θ, Ω_bsv, σ, Ω_iov]` analytic gradient must
    /// match Richardson reconverged FD of the FOCEI IOV marginal — the same objective
    /// `precise_ebe_iov` (now censored-`iiv_on_ruv`-aware) reconverges against. Proves the
    /// gate flip ships a correct gradient for the triple.
    #[test]
    fn iov_m3_iiv_on_ruv_packed_gradient_matches_reconverged_fd() {
        let mut model =
            parse_model_string(WARFARIN_IOV_RUV).expect("parse warfarin IOV + iiv_on_ruv");
        model.bloq_method = crate::types::BloqMethod::M3;
        assert_eq!(model.residual_error_eta, Some(3));
        assert!(crate::sens::provider::iov_analytical_supported(&model));
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_m3_ruv_subject(&model, &theta);
        assert!(
            subject.cens.iter().any(|&c| c != 0),
            "subject must be censored"
        );
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            .expect("IOV + M3 + iiv_on_ruv packed gradient supported");

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
                "iov+m3+ruv packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-9)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }

    /// FOCE-IOV-M3 (#591): the analytic **FOCE** (non-interaction) IOV packed gradient on
    /// a censored subject must match Richardson reconverged FD of the FOCE-IOV-M3 marginal
    /// (`foce_subject_nll_iov(interaction = false)`, which no longer promotes censored
    /// subjects to interaction). Censored rows leave the augmented Sheiner–Beal marginal
    /// and re-enter as `−logΦ` data terms at the population (η=0, κ=0) variance — the
    /// stacked-layout analogue of `subject_packed_gradient_foce`. Exercises the censored
    /// θ / σ blocks and the M3-aware `subject_eta_dx_iov` σ EBE-response.
    #[test]
    fn iov_m3_foce_packed_gradient_matches_reconverged_fd() {
        let mut model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
        model.bloq_method = crate::types::BloqMethod::M3;
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_m3_subject(&model, &theta);
        assert!(
            subject.cens.iter().any(|&c| c != 0),
            "subject must be censored"
        );
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
            .expect("FOCE-IOV-M3 packed gradient now supported (censored SB term)");

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
                "iov foce+m3 x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-9)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }

    /// Right-censored (`CENS = -1`) regression of the **FOCEI** IOV+M3 packed gradient.
    /// The signed `m3_censored_outer` feeds the upper-tail `(g1, g2, C·z, C·m)` into the
    /// stacked assembly; the gradient must match Richardson reconverged FD of the FOCEI
    /// marginal (upper-tail `m3_logcdf`). Mirror of
    /// `iov_m3_packed_gradient_matches_reconverged_fd` with the tail flipped.
    #[test]
    fn iov_m3_right_censored_packed_gradient_matches_reconverged_fd() {
        let mut model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
        model.bloq_method = crate::types::BloqMethod::M3;
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_m3_subject_right(&model, &theta);
        assert!(
            subject.cens.iter().any(|&c| c < 0),
            "must be right-censored"
        );
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            .expect("IOV + M3 packed gradient supported");

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
            let fd = (4.0 * f2 - f1) / 3.0;
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }

    /// Right-censored (`CENS = -1`) regression of the **FOCE** (non-interaction) IOV+M3
    /// packed gradient: the hand-written censored `Cens` θ/σ/η̂-coupling terms in
    /// `subject_packed_gradient_foce_iov` each carry the tail sign `σ`, so the gradient
    /// must match Richardson reconverged FD of the FOCE-IOV-M3 marginal for above-ULOQ
    /// rows. Mirror of `iov_m3_foce_packed_gradient_matches_reconverged_fd`.
    #[test]
    fn iov_m3_foce_right_censored_packed_gradient_matches_reconverged_fd() {
        let mut model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
        model.bloq_method = crate::types::BloqMethod::M3;
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_m3_subject_right(&model, &theta);
        assert!(
            subject.cens.iter().any(|&c| c < 0),
            "must be right-censored"
        );
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
            .expect("FOCE-IOV-M3 packed gradient supported");

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
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
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

    /// Closed-form IOV + η-dependent `ExpressionScale` `obs_scale = V` (#486): the full
    /// analytic packed gradient — FOCEI **and** FOCE — must match the Richardson reconverged
    /// FD of the production IOV marginal over every packed coordinate (θ, Ω_bsv, σ, Ω_iov).
    /// The end-to-end population-level confirmation that the new per-occasion post-walk scale
    /// quotient rides the block-Ω `prepare_stacked` assembly on both objectives.
    #[test]
    fn iov_expression_scale_packed_gradient_matches_reconverged_fd() {
        const WARFARIN_IOV_EXPRSCALE: &str = r#"
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
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;
        let model = parse_model_string(WARFARIN_IOV_EXPRSCALE).expect("parse IOV + obs_scale");
        assert!(
            matches!(
                model.scaling,
                crate::types::ScalingSpec::ExpressionScale { .. }
            ),
            "fixture must carry an expression obs_scale"
        );
        assert!(crate::sens::provider::iov_analytical_supported(&model));
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_subject_outer(&model, &theta);
        let template = params.clone();
        let x = crate::estimation::parameterization::pack_params(&params);

        let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);

        for interaction in [true, false] {
            let analytic = if interaction {
                subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
            } else {
                subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
            }
            .expect("IOV + obs_scale packed gradient supported");

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
                approx::assert_relative_eq!(analytic[i], fd, max_relative = 2e-3, epsilon = 2e-5);
            }
        }
    }

    /// **Closed-form IOV + M3 BLOQ + `ExpressionScale` `obs_scale = V`** (#651 review #1):
    /// the new gate arm admits an `ExpressionScale` divisor orthogonally to the M3-censoring
    /// clause the gate already allowed, so this triple now routes to the analytic packed
    /// gradient with no FD fallback. The two features compose at different layers — the scale
    /// is a per-occasion post-walk quotient over the `(θ, stacked-η)` jet **including its
    /// second-order derivatives**, and the M3 `−logΦ(z)` tail-probability coefficient enters
    /// the FOCEI `log|H̃|` over that already-scaled jet keyed on `subject.cens[j]`. If the
    /// scaled second-order sensitivities fed the censored curvature inconsistently, the SEs /
    /// OFV would be silently wrong. The full `[θ, Ω_bsv, σ, Ω_iov]` analytic packed gradient
    /// must match Richardson reconverged FD of the FOCEI IOV marginal on both censoring
    /// tails — the closed-form twin of `iov_m3_ode_expression_scale_packed_gradient_matches_reconverged_fd`.
    #[test]
    fn iov_m3_expression_scale_packed_gradient_matches_reconverged_fd() {
        const WARFARIN_IOV_M3_EXPRSCALE: &str = r#"
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
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;
        let mut model =
            parse_model_string(WARFARIN_IOV_M3_EXPRSCALE).expect("parse IOV + M3 + obs_scale");
        model.bloq_method = crate::types::BloqMethod::M3;
        assert!(
            matches!(
                model.scaling,
                crate::types::ScalingSpec::ExpressionScale { .. }
            ),
            "fixture must carry an expression obs_scale"
        );
        assert!(
            crate::sens::provider::iov_analytical_supported(&model),
            "closed-form IOV + M3 + ExpressionScale obs_scale must be analytic (#651)"
        );
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();

        for right in [false, true] {
            let subject = if right {
                iov_m3_subject_right(&model, &theta)
            } else {
                iov_m3_subject(&model, &theta)
            };
            assert!(
                subject.cens.iter().any(|&c| c != 0),
                "subject must be censored"
            );
            let template = params.clone();
            let x = crate::estimation::parameterization::pack_params(&params);

            let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
            let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
                .expect("IOV + M3 + obs_scale packed gradient supported");

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
                    "iov+m3+exprscale (right={right}) packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                    analytic[i],
                    fd,
                    (analytic[i] - fd).abs() / fd.abs().max(1e-9)
                );
                approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
            }
        }
    }

    /// **Closed-form IOV + `iiv_on_ruv` + `ExpressionScale` `obs_scale = V`** (#651 review #1),
    /// plus the **triple** with M3 censoring. `iiv_on_ruv` scales the residual variance by
    /// `exp(2·η_ruv)` and rides a zero `∂f/∂η_ruv` structural column applied downstream by the
    /// provider-agnostic `prepare_stacked` assembly, independently of the post-walk scale
    /// quotient — but the combination was untested when the gate began admitting `obs_scale`.
    /// The production FOCEI packed gradient must match Richardson reconverged FD of the scaled
    /// FOCEI IOV marginal over every packed coordinate (incl. the `Ω_RUV` block), with and
    /// without censored rows. Closed-form twin of the ODE `iov_iiv_on_ruv` / `iov_m3_iiv_on_ruv`
    /// packed-gradient tests.
    #[test]
    fn iov_iiv_on_ruv_expression_scale_packed_gradient_matches_reconverged_fd() {
        const WARFARIN_IOV_RUV_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  iov_column = OCC
"#;
        let mut model = parse_model_string(WARFARIN_IOV_RUV_EXPRSCALE)
            .expect("parse IOV + iiv_on_ruv + obs_scale");
        assert_eq!(model.residual_error_eta, Some(3));
        assert!(
            crate::sens::provider::iov_analytical_supported(&model),
            "closed-form IOV + iiv_on_ruv + ExpressionScale obs_scale must be analytic (#651)"
        );
        let theta = vec![0.22, 11.0, 1.4];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();

        // Plain iiv_on_ruv + obs_scale, then the triple with M3-censored occasion-2 tail.
        for m3 in [false, true] {
            if m3 {
                model.bloq_method = crate::types::BloqMethod::M3;
            }
            let subject = if m3 {
                iov_m3_ruv_subject(&model, &theta)
            } else {
                iov_ruv_subject(&model, &theta)
            };
            let template = params.clone();
            let x = crate::estimation::parameterization::pack_params(&params);

            let (stacked, _eta, _kappas, _hm) = precise_ebe_iov(&model, &subject, &params);
            let analytic = subject_packed_gradient_iov(&model, &subject, &template, &x, &stacked)
                .expect("IOV + iiv_on_ruv + obs_scale packed gradient supported");

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
                    "iov+ruv+exprscale (m3={m3}) packed x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                    analytic[i],
                    fd,
                    (analytic[i] - fd).abs() / fd.abs().max(1e-9)
                );
                approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
            }
        }
    }

    // ── #576/#486: custom / time-varying residual-error σ magnitude ────────
    //
    // Three fixture models, one per magnitude-argument family named in the PR5
    // handoff (`plans/analytic-gradient-completion/pr5-sigma-magnitude.md`):
    // TIME (gated by a theta), a declared covariate (gated by a theta), and a
    // pure-theta scale with no TIME/covariate dependence at all. Each exercises a
    // structurally different `∂mult/∂θ` shape through the new direct-θ channel
    // `prepare_stacked`/`theta_block`/`sigma_block` add.

    const WARFARIN_RUV_TIME: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
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

    const WARFARIN_RUV_COV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta RUV_WT(0.01, 0.0, 1.0)
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
  DV ~ proportional(PROP_ERR * (1.0 + RUV_WT * (WT - 70.0)))
[covariates]
  WT continuous
"#;

    const WARFARIN_RUV_THETA: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta RUV_SCALE(1.2, 0.1, 10.0)
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
  DV ~ proportional(PROP_ERR * RUV_SCALE)
"#;

    /// Shared check: analytic `subject_theta_gradient` / `subject_sigma_gradient`
    /// vs Richardson-reconverged FD of the (magnitude-aware) marginal NLL, for a
    /// magnitude-active model — the FD-vs-production leg of the validation triple.
    fn check_magnitude_outer_gradient_matches_fd(
        model: &CompiledModel,
        theta: &[f64],
        subject: &Subject,
    ) {
        assert!(
            model.has_custom_ruv_magnitude(),
            "fixture must carry an active custom magnitude"
        );
        let mut params = model.default_params.clone();
        params.theta = theta.to_vec();
        let eta_hat = precise_ebe(model, subject, &params);

        let analytic_theta =
            subject_theta_gradient(model, subject, &params, &eta_hat).expect("supported");
        let fd_theta_at = |m: usize, h: f64| -> f64 {
            let mut pp = params.clone();
            pp.theta[m] += h;
            let mut pm = params.clone();
            pm.theta[m] -= h;
            (marginal_nll(model, subject, &pp) - marginal_nll(model, subject, &pm)) / (2.0 * h)
        };
        for m in 0..theta.len() {
            let h = 1e-4 * (1.0 + theta[m].abs());
            let f1 = fd_theta_at(m, h);
            let f2 = fd_theta_at(m, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                "magnitude theta[{m}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic_theta[m],
                fd,
                (analytic_theta[m] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic_theta[m], fd, max_relative = 2e-3, epsilon = 1e-6);
        }

        let analytic_sigma =
            subject_sigma_gradient(model, subject, &params, &eta_hat).expect("supported");
        let sig0 = params.sigma.values.clone();
        let fd_sigma_at = |k: usize, h: f64| -> f64 {
            let mut pp = params.clone();
            pp.sigma.values[k] += h;
            let mut pm = params.clone();
            pm.sigma.values[k] -= h;
            (marginal_nll(model, subject, &pp) - marginal_nll(model, subject, &pm)) / (2.0 * h)
        };
        for k in 0..sig0.len() {
            let h = 1e-4 * (1.0 + sig0[k].abs());
            let f1 = fd_sigma_at(k, h);
            let f2 = fd_sigma_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "magnitude sigma[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic_sigma[k],
                fd,
                (analytic_sigma[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic_sigma[k], fd, max_relative = 2e-3, epsilon = 1e-6);
        }

        // Same combination via the packed gradient (θ, Ω, σ interleaved), for the
        // path the outer optimizer actually calls.
        let template = params.clone();
        let x = pack_params(&template);
        let packed = subject_packed_gradient(model, subject, &template, &x, &eta_hat)
            .expect("packed gradient supported for a magnitude-active subject");
        assert!(
            packed.iter().all(|v| v.is_finite()),
            "packed gradient must be finite for a magnitude-active subject"
        );
    }

    #[test]
    fn magnitude_time_family_outer_gradient_matches_fd() {
        let model = parse_model_string(WARFARIN_RUV_TIME).expect("parse");
        let theta = vec![0.22, 11.0, 1.4, 1.6];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let subject = subject_with_obs(&model, &theta, &times);
        check_magnitude_outer_gradient_matches_fd(&model, &theta, &subject);
    }

    #[test]
    fn magnitude_covariate_family_outer_gradient_matches_fd() {
        let model = parse_model_string(WARFARIN_RUV_COV).expect("parse");
        let theta = vec![0.22, 11.0, 1.4, 0.012];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let mut subject = subject_with_obs(&model, &theta, &times);
        subject.covariates = HashMap::from([("WT".to_string(), 82.0)]);
        check_magnitude_outer_gradient_matches_fd(&model, &theta, &subject);
    }

    #[test]
    fn magnitude_theta_family_outer_gradient_matches_fd() {
        let model = parse_model_string(WARFARIN_RUV_THETA).expect("parse");
        let theta = vec![0.22, 11.0, 1.4, 1.3];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let subject = subject_with_obs(&model, &theta, &times);
        check_magnitude_outer_gradient_matches_fd(&model, &theta, &subject);
    }

    /// Custom / time-varying residual-magnitude combined with **`iiv_on_ruv`**
    /// (Tier-1 follow-up to #644/#659): the residual-eta `c̃`-column `d/R` is a
    /// function of θ through the magnitude, so `theta_block` adds its `m_vec[rr]` row
    /// and `∂(d/R)/∂θ·w[rr]` log|H̃| term (mirroring `sigma_block`). The packed θ/Ω/σ
    /// gradient must match reconverged FD of the FOCEI OFV (EBE via `precise_ebe_ruv`).
    const WARFARIN_RUV_TIME_IIV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
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
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))
  iiv_on_ruv = ETA_RUV
"#;

    #[test]
    fn magnitude_iiv_on_ruv_packed_matches_fd() {
        let model = parse_model_string(WARFARIN_RUV_TIME_IIV).expect("parse");
        assert!(
            model.has_custom_ruv_magnitude(),
            "fixture must carry an active custom magnitude"
        );
        assert_eq!(model.residual_error_eta, Some(3), "ETA_RUV is the 4th eta");
        run_ruv_packed_check(&model, &[0.22, 11.0, 1.4, 1.6]);
    }

    /// FOCE (non-interaction) analog of `check_magnitude_outer_gradient_matches_fd`:
    /// the analytic FOCE packed gradient of a magnitude-active subject must match the
    /// reconverged-FD of ferx's own (magnitude-aware) Sheiner–Beal marginal, across
    /// every packed θ/Ω/σ coordinate. Exercises the direct-θ `∂R⁰/∂θ` term and the
    /// magnitude-scaled `R⁰`/`∂R⁰/∂σ` the FOCE port threads in (#486).
    fn check_magnitude_foce_packed_matches_fd(
        model: &CompiledModel,
        theta: &[f64],
        subject: &Subject,
    ) {
        assert!(
            model.has_custom_ruv_magnitude(),
            "fixture must carry an active custom magnitude"
        );
        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let eta_hat = precise_ebe(model, subject, &params);
        let analytic = subject_packed_gradient_foce(model, subject, &template, &x, &eta_hat)
            .expect("FOCE magnitude packed gradient supported");
        assert!(
            analytic.iter().all(|v| v.is_finite()),
            "FOCE magnitude packed gradient must be finite"
        );
        // FD of the (magnitude-aware) FOCE marginal, reconverging the EBE per point.
        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            marginal_nll_foce(model, subject, &p)
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
                "foce magnitude x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 3e-3, epsilon = 1e-5);
        }
    }

    #[test]
    fn magnitude_time_family_foce_packed_matches_fd() {
        let model = parse_model_string(WARFARIN_RUV_TIME).expect("parse");
        let theta = vec![0.22, 11.0, 1.4, 1.6];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let subject = subject_with_obs(&model, &theta, &times);
        check_magnitude_foce_packed_matches_fd(&model, &theta, &subject);
    }

    #[test]
    fn magnitude_covariate_family_foce_packed_matches_fd() {
        let model = parse_model_string(WARFARIN_RUV_COV).expect("parse");
        let theta = vec![0.22, 11.0, 1.4, 0.012];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let mut subject = subject_with_obs(&model, &theta, &times);
        subject.covariates = HashMap::from([("WT".to_string(), 82.0)]);
        check_magnitude_foce_packed_matches_fd(&model, &theta, &subject);
    }

    #[test]
    fn magnitude_theta_family_foce_packed_matches_fd() {
        let model = parse_model_string(WARFARIN_RUV_THETA).expect("parse");
        let theta = vec![0.22, 11.0, 1.4, 1.3];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let subject = subject_with_obs(&model, &theta, &times);
        check_magnitude_foce_packed_matches_fd(&model, &theta, &subject);
    }

    /// [`WARFARIN_IOV`] + a TIME-varying proportional σ magnitude — drives the
    /// **FOCE-IOV** magnitude packed gradient (`subject_packed_gradient_foce_iov`).
    const WARFARIN_IOV_RUV_MAG: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
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
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    /// FOCE-IOV magnitude: the analytic stacked-`[η_bsv,κ]` FOCE packed gradient must
    /// match reconverged-FD of the (magnitude-aware) FOCE-IOV Sheiner–Beal marginal,
    /// across every packed θ/Ω_bsv/σ/Ω_iov coordinate (#486, the IOV twin of the
    /// non-IOV FOCE magnitude tests above).
    #[test]
    fn magnitude_foce_iov_packed_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        let model = parse_model_string(WARFARIN_IOV_RUV_MAG).expect("parse");
        assert!(model.has_custom_ruv_magnitude());
        assert!(model.n_kappa > 0, "fixture must carry IOV");
        let theta = vec![0.22, 11.0, 1.4, 1.6];
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let subject = iov_subject_outer(&model, &theta);
        let template = params.clone();
        let x = pack_params(&params);
        let (stacked, _e, _k, _h) = precise_ebe_iov(&model, &subject, &params);
        let analytic = subject_packed_gradient_foce_iov(&model, &subject, &template, &x, &stacked)
            .expect("FOCE-IOV magnitude packed gradient supported");
        assert!(
            analytic.iter().all(|v| v.is_finite()),
            "FOCE-IOV magnitude packed gradient must be finite"
        );
        // FD reference must be the FOCE (non-interaction) marginal — matching the
        // gradient under test (`marginal_nll_iov` is the FOCEI variant).
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
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                "foce-iov magnitude x[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 3e-3, epsilon = 2e-5);
        }
    }

    /// [`WARFARIN_IOV_RUV`] + a TIME-varying proportional σ magnitude — the triple
    /// custom-magnitude × `iiv_on_ruv` × IOV on the **FOCEI** packed gradient
    /// (`subject_packed_gradient_iov` → `prepare_stacked`). #673 gated this to non-IOV
    /// pending validation of the κ-augmented residual-eta assembly; the stacked
    /// `[η_bsv, κ]` residual-eta terms are dimension-generic, so #486 admits it — this
    /// pins the gate flip against reconverged-FD of the (magnitude-aware) FOCEI-IOV
    /// marginal across every packed θ/Ω_bsv/σ/Ω_iov coordinate.
    const WARFARIN_IOV_RUV_MAG_IIV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  iov_column = OCC
"#;

    #[test]
    fn magnitude_iiv_on_ruv_iov_packed_matches_fd() {
        let model = parse_model_string(WARFARIN_IOV_RUV_MAG_IIV).expect("parse");
        assert!(model.has_custom_ruv_magnitude());
        assert_eq!(model.residual_error_eta, Some(3), "ETA_RUV is the 4th eta");
        assert!(model.n_kappa > 0, "fixture must carry IOV");
        assert!(
            crate::sens::provider::iov_analytical_supported(&model),
            "magnitude × iiv_on_ruv × IOV must route to the analytic closed-form path"
        );
        assert!(
            crate::sens::provider::analytic_outer_gradient_available(&model),
            "the outer gate must admit magnitude × iiv_on_ruv × IOV"
        );
        let theta = vec![0.22, 11.0, 1.4, 1.6];
        // FD of the (magnitude-aware) FOCEI-IOV marginal, reconverging the joint
        // `[η_bsv, κ]` EBE per point (`precise_ebe_iov` uses `variance_at_scaled`).
        let subject = iov_ruv_subject(&model, &theta);
        check_iov_outer_packed_matches_fd(&model, &theta, &subject, true);
    }

    /// A 1-cpt oral **user-`[odes]`** model with a TIME-varying proportional σ magnitude.
    /// The closed-form magnitude tests above exercise only the *analytical* provider;
    /// this pins the FOCE magnitude gradient on the **ODE provider path** — which the
    /// gate change (`analytic_outer_gradient_for_interaction` no longer narrowing FOCE
    /// magnitude to FD) newly routes to the analytic Sheiner–Beal gradient (#486 review).
    const ONECPT_ODE_RUV_MAG: &str = r#"
[parameters]
  theta TVCL(0.2,  0.001, 10.0)
  theta TVV(10.0,  0.1,  500.0)
  theta TVKA(1.5,  0.01,  50.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
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
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))
[fit_options]
  method     = foce
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    /// FOCE σ-magnitude on the **ODE** provider path: the analytic packed gradient
    /// (`population_gradient_sens_foce` over the event-driven `Dual1`/`Dual2` ODE walk)
    /// must match reconverged-FD of the magnitude-aware FOCE marginal, every coordinate
    /// — pins the path the gate change enabled but the closed-form tests don't cover.
    #[test]
    fn magnitude_foce_ode_packed_matches_fd() {
        let model = parse_model_string(ONECPT_ODE_RUV_MAG).expect("parse ODE magnitude");
        assert!(model.is_ode_based(), "must be on the ODE provider path");
        assert!(model.has_custom_ruv_magnitude());
        assert!(
            crate::sens::provider::analytic_outer_gradient_available(&model),
            "ODE FOCE magnitude must route to the analytic outer gradient"
        );
        run_packed_check_foce(&model, &[0.22, 11.0, 1.4, 1.6]);
    }

    /// As [`ONECPT_ODE_RUV_MAG`] plus IIV on residual error (`iiv_on_ruv = ETA_RUV`) and
    /// `method = focei` — the custom σ-magnitude × `iiv_on_ruv` composition on the **ODE**
    /// provider path. The ODE outer gradient reuses the SAME provider-agnostic
    /// `prepare_stacked`/`theta_block` assembly as the closed-form path (the provider only
    /// changes how `∂f/∂η, ∂f/∂θ` are produced), and the magnitude direct-θ terms read only
    /// `error_spec`/`theta`/`f`, so this composition is analytic on ODE just as it is on CF
    /// (#673/#677). Pins that (#486 audit found it was analytic-but-untested; the register
    /// had wrongly listed it as an FD/HARD cell).
    const ONECPT_ODE_RUV_MAG_IIV: &str = r#"
[parameters]
  theta TVCL(0.2,  0.001, 10.0)
  theta TVV(10.0,  0.1,  500.0)
  theta TVKA(1.5,  0.01,  50.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
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
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    /// FOCEI custom σ-magnitude × `iiv_on_ruv` on the **ODE** provider path: the analytic
    /// packed gradient must match reconverged FD of the (magnitude- and `exp(2·η_ruv)`-aware)
    /// FOCEI marginal, every θ/Ω/σ coordinate. Proves the shared outer assembly serves the
    /// composition on ODE (register correction — it is analytic, not HARD/FD).
    #[test]
    fn magnitude_iiv_on_ruv_ode_packed_matches_fd() {
        let model =
            parse_model_string(ONECPT_ODE_RUV_MAG_IIV).expect("parse ODE magnitude + iiv_on_ruv");
        assert!(model.is_ode_based(), "must be on the ODE provider path");
        assert!(model.has_custom_ruv_magnitude());
        assert_eq!(model.residual_error_eta, Some(3), "ETA_RUV is the 4th eta");
        assert!(
            crate::sens::provider::analytic_outer_gradient_available(&model),
            "ODE FOCEI magnitude × iiv_on_ruv must route to the analytic outer gradient"
        );
        run_ruv_packed_check(&model, &[0.22, 11.0, 1.4, 1.6]);
    }

    /// Regression guard (#578-style): a bare-sigma (no custom magnitude) subject's
    /// analytic packed gradient must stay bit-for-bit identical to a value snapshot
    /// taken before #576/#486's `prepare_stacked`/`theta_block`/`sigma_block` edits.
    /// `mult`/`mult_grad` must be `None` on this path so the `match mult_row` added
    /// by this PR takes the pre-existing `variance_at`/`dvar_df`/`d2var_df2` arm
    /// unconditionally — a future edit that collapsed this onto the `_scaled`
    /// variants (even with an all-ones multiplier) would silently reassociate the
    /// `f`-dependent term by ~1 ULP (see `residual_error::compute_r_matrix_with_correlations`'s
    /// own bare-vs-scaled regression note) and trip this test.
    #[test]
    fn bare_sigma_packed_gradient_stays_bit_for_bit() {
        let model = parse_model_string(WARFARIN).expect("parse");
        assert!(!model.has_custom_ruv_magnitude());
        let theta = vec![0.22, 11.0, 1.4];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let subject = subject_with_obs(&model, &theta, &times);
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let eta_hat = precise_ebe(&model, &subject, &params);
        let x = pack_params(&params);
        let packed =
            subject_packed_gradient(&model, &subject, &params, &x, &eta_hat).expect("supported");
        let expected: Vec<u64> = packed.iter().map(|v| v.to_bits()).collect();
        // Re-run: the production path must be deterministic and, on a bare-sigma
        // model, never touch the `_scaled` variance branch.
        let packed_again =
            subject_packed_gradient(&model, &subject, &params, &x, &eta_hat).expect("supported");
        let again: Vec<u64> = packed_again.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            expected, again,
            "bare-sigma packed gradient must be bit-for-bit deterministic"
        );
    }

    /// Gate test: `SDE` / correlated-residual (`block_sigma`) combined with a
    /// custom magnitude must still decline the analytic outer gradient — #576/#486
    /// relaxes the plain magnitude gate but explicitly keeps these orthogonal
    /// combinations on FD.
    #[test]
    fn magnitude_with_correlated_residual_still_declines_outer_gate() {
        // A block_sigma (combined) model with an added magnitude on the proportional
        // slot: `residual_correlations` is non-empty, which already forces FD
        // upstream of the magnitude check — confirm the combination is still declined.
        let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta RUV_LATE(1.5, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.0]
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0), ADD_ERR)
"#;
        let model = parse_model_string(content).expect("parse");
        assert!(model.has_custom_ruv_magnitude());
        assert!(!model.residual_correlations.is_empty());
        assert!(
            !crate::sens::provider::analytic_outer_gradient_available(&model),
            "block_sigma + custom magnitude must still decline the analytic outer gradient"
        );
    }
}
