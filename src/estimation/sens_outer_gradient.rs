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

use crate::estimation::parameterization::{
    block_chol_full, chol_pack, lower_tri_entries, packed_fixed_mask, rho_chain, rho_packed_start,
    theta_packs_log, unpack_params,
};
use crate::sens::provider::{subject_sensitivities, ObsSens, SubjectSens};
use crate::stats::residual_error::{residual_rd, residual_rd2};
use crate::stats::special::m3_censored_outer;
use crate::types::{CompiledModel, ModelParameters, Population, ResidualCorrelation, Subject};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Per-observation error-model scalars used throughout the assembly.
pub(crate) struct ErrTerms {
    pub(crate) r: f64,       // Rⱼ
    pub(crate) d: f64,       // dⱼ = ∂R/∂f
    pub(crate) eps: f64,     // εⱼ = y − f
    pub(crate) alpha: f64,   // αⱼ
    pub(crate) alpha_p: f64, // α'ⱼ = dαⱼ/df
    pub(crate) p: f64,       // pⱼ
    pub(crate) beta: f64,    // βⱼ = dpⱼ/df
    // M3-censored × residual-eta (`iiv_on_ruv`) cross-term coefficients (#4c).
    // Zero on quantified rows and on non-`iiv_on_ruv` censored rows. With
    // `z = (y−f)/√v`, `h = φ(z)/Φ(z)`, `m = 1/√v + (y−f)·R'/(2 v^{3/2})` and
    // `C = h·(z² + h·z − 1)`, the censored data term `L = −logΦ(z)` under
    // `v = R·exp(2·η_ruv)` has `∂²L/∂η_ruv² = C·z`, `∂²L/∂η_l∂η_ruv = C·m·a_l`,
    // `∂²L/∂η_ruv∂θ = C·m·b`, `∂²L/∂η_ruv∂σ = ½·(C·z)·(∂v/∂σ)/v`. So the true
    // inner Hessian's residual-eta row/col reads `ruv_cz`/`ruv_cm` instead of the
    // Gaussian `2ε²/R`/`ruv_kappa`. Since #486 these `ruv_cz`/`ruv_cm` terms also enter
    // `H̃`/`log|H̃|` (with their θ/σ/η derivatives), consistently with quantified rows.
    pub(crate) ruv_cz: f64, // C·z  (residual-eta diagonal of the true inner Hessian)
    pub(crate) ruv_cm: f64, // C·m  (residual-eta × structural-η / θ / σ coupling)
    /// True for an M3-censored row. The residual-eta blocks read the censored
    /// `ruv_cz`/`ruv_cm` coefficients instead of the Gaussian `2ε²/R`/`ruv_kappa`.
    pub(crate) censored: bool,
    /// Raw NONMEM `CENS` sign for this row (`0` quantified, `>0` below-LLOQ /
    /// lower tail, `<0` above-ULOQ / upper tail). The σ-block's FD of the
    /// censored df-coefficient must re-evaluate the kernel on the same tail, so
    /// the sign is carried here rather than re-read from the subject.
    pub(crate) cens_sign: i8,
    /// `∂Rⱼ/∂θₘ` from the custom residual-error magnitude's *direct* θ-dependence
    /// (#484/#576/#486) — `mult(θ)` enters `R` independent of the prediction `f`,
    /// so this is a channel `theta_block` would otherwise miss entirely. Empty
    /// when no magnitude is active (the common case; zero-cost).
    pub(crate) dr_dtheta: Vec<f64>,
    /// `∂dⱼ/∂θₘ = ∂²Rⱼ/∂f∂θₘ`, the `f`-derivative of `dr_dtheta` — the magnitude
    /// analog of the σ-block's `d_sig`. Empty when no magnitude is active.
    pub(crate) dd_dtheta: Vec<f64>,
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
pub(crate) struct Prep {
    pub(crate) n_eta: usize,
    pub(crate) n_obs: usize,
    pub(crate) et: Vec<ErrTerms>,
    /// `Ω⁻¹` (copied so blocks don't borrow `params`).
    pub(crate) omega_inv: DMatrix<f64>,
    /// `H̃⁻¹` (first-order FOCEI Hessian inverse).
    pub(crate) htilde_inv: DMatrix<f64>,
    /// `H⁻¹` for the **true** inner Hessian `H = ∂²lᵢ/∂η²` (Eq. 46 denominator).
    pub(crate) h_inner_inv: DMatrix<f64>,
    /// `wⱼ = H̃⁻¹aⱼ`.
    pub(crate) w: Vec<DVector<f64>>,
    /// `qⱼ = aⱼᵀ H̃⁻¹ aⱼ`.
    pub(crate) q: Vec<f64>,
    /// Exact `∂log|H̃|/∂η` (a-fixed part + `∂²f/∂η²` curvature).
    pub(crate) g_eta: Vec<f64>,
    // Per-observation M3-censored flag lives on `et[j].censored` (single source).
    // Censored rows enter `H` (true inner Hessian), the data gradient, AND `H̃`/`log|H̃|`
    // at FOCEI order (`p = g2`, `β = dg2/df`; residual-eta `C·z`/`C·m`) — consistently with
    // quantified rows (#486), matching `gaussian_foce_accum`'s `cens_hess`.
    /// IIV-on-RUV (`Y = IPRED + EPS·EXP(η_ruv)`, #474): the random-effect index
    /// that scales the residual variance by `exp(2·η_ruv)`, or `None`. When set,
    /// the variance terms `r`/`d` in `et` already carry that factor, and the
    /// `η_ruv` row/col of `H̃`/`H` plus its `log|H̃|` derivatives are assembled
    /// from the per-observation `g_ruv`/`gp_ruv` scalars below.
    pub(crate) ruv: Option<usize>,
    /// `gⱼ = (∂Rⱼ/∂fⱼ)/Rⱼ = dⱼ/Rⱼ` per observation (scale-invariant) — the
    /// residual-eta `c̃` cross coupling `H̃[ruv,l] = Σⱼ gⱼ a_{jl}`. Empty when no
    /// `ruv`.
    pub(crate) g_ruv: Vec<f64>,
    /// `g'ⱼ = ∂gⱼ/∂fⱼ = d2ⱼ/Rⱼ − (dⱼ/Rⱼ)²` per observation (scale-invariant) — the
    /// `f`-derivative of the `c̃` coupling, needed for `∂log|H̃|/∂θ` and `/∂η`.
    /// Empty when no `ruv`.
    pub(crate) gp_ruv: Vec<f64>,
    /// `exp(2·η̂_ruv)` (1.0 when no `ruv`) — the residual-variance scale, used to
    /// lift the σ-block's central-FD `∂R/∂σ` / `∂d/∂σ` (taken on the *unscaled*
    /// error functions) onto the scaled variance.
    pub(crate) ruv_scale: f64,
    /// Total `f`-derivatives (through `v(f)`) of the censored residual-eta `H̃`
    /// coefficients `C·z` / `C·m` per observation (0 on quantified rows). Computed
    /// once in `prepare` (FD of the kernel) and reused by `theta_block`'s censored
    /// residual-eta `log|H̃|` θ-derivative. Empty when no `ruv`.
    pub(crate) cens_dcz_df: Vec<f64>,
    pub(crate) cens_dcm_df: Vec<f64>,
    /// Custom / time-varying residual-magnitude (#484/#576) `[obs][sigma-slot]`
    /// multiplier matrix, or `None` when no magnitude is active. Computed once
    /// here; `sigma_block` reuses it instead of recomputing `model.ruv_obs_mult`
    /// (which re-walks every magnitude expression per observation) a second time
    /// for the same subject/θ (#486 review).
    pub(crate) mult: Option<Vec<Vec<f64>>>,
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
    corr: &[ResidualCorrelation],
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    use crate::stats::residual_error::{
        compute_d2r_df2_matrices, compute_dr_df_matrices, compute_r_matrix_with_correlations,
    };
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
        &subject.obs_l2,
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
        &subject.obs_l2,
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
        &subject.obs_l2,
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
    corr: &[ResidualCorrelation],
) -> (Vec<f64>, Vec<f64>) {
    use crate::stats::residual_error::compute_dr_df_matrices;
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
        &subject.obs_l2,
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

pub(crate) fn prepare(
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
///
/// The formula itself lives with the rest of the residual-variance math in
/// [`crate::stats::residual_error::magnitude_dvar_dtheta`], which since #1182
/// also carries the `power(...)` exponent's `∂R/∂p · ∂p/∂θ` term; this is the
/// call-site alias the three gradient paths share.
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
    dd_dtheta: Option<&mut Vec<f64>>,
) -> Vec<f64> {
    crate::stats::residual_error::magnitude_dvar_dtheta(
        error_spec,
        cmt,
        f,
        sigma,
        mult_row,
        mult_grad_row,
        n_theta,
        ruv_scale,
        dd_dtheta,
    )
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

/// The σ-EBE-response data-term coefficient derivative for one observation:
/// `∂α/∂σ = [2ε/R² + d(2ε²−R)/R³]·Rσ + [(R−ε²)/R²]·dσ`, given `(r, d, eps)` and the
/// σ-FD slopes `r_sig = ∂R/∂σ`, `d_sig = ∂d/∂σ`. The σ mirror of [`mag_alpha_dtheta`],
/// shared by the three σ-blocks (`sigma_block`, `subject_eta_dx{,_iov}`).
#[inline]
fn dalpha_dsigma(r: f64, d: f64, eps: f64, r_sig: f64, d_sig: f64) -> f64 {
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;
    let inv_r3 = inv_r2 * inv_r;
    (2.0 * eps * inv_r2 + d * (2.0 * eps * eps - r) * inv_r3) * r_sig
        + ((r - eps * eps) * inv_r2) * d_sig
}

/// [`prepare`] generalized over the random-effect dimension and prior precision,
/// so it serves both the non-IOV path (`n_eta = model.n_eta`, `Ω⁻¹ = params.omega.inv`)
/// and the **IOV** path, where the random effects are the stacked
/// `[η_bsv, κ₁..κ_K]` and `omega_inv` is the inverse of the block-diagonal
/// `Ω_bsv ⊕ K·Ω_iov`. Everything else (error model, σ, censoring) is shared.
/// The per-observation scalar likelihood chain (`ErrTerms`) plus the two Hessians, for
/// an **arbitrary** `eta` — the mode is never assumed.
///
/// This is the half of [`prepare_stacked`] that does not depend on the FOCEI `log|H̃|`
/// machinery, split out so callers that only need the *score* of the conditional NLL can
/// have it without paying for two Cholesky inverses per call (#251 AGQ/Laplace, which
/// evaluates it once per quadrature node).
///
/// Two things here are exactly what an AGQ/Laplace node needs, for the FULL analytic
/// scope (M3-censored, `iiv_on_ruv`, custom σ magnitude, correlated residual, LTBS):
///
/// * `et[j].alpha = 2·∂L_j/∂f_j` — the residual chain, per endpoint family; and
/// * `h_inner = Ω⁻¹ + Σⱼ (∂²L_j/∂f² aⱼaⱼᵀ + ∂L_j/∂f Aⱼ)`, which **is** the exact
///   conditional Hessian `∂²nll/∂b²` (`L_j = ½(εⱼ²/Rⱼ + ln Rⱼ)`, so `½α` and `½α'`
///   recover `∂L/∂f` and `∂²L/∂f²`).
pub(crate) struct ScoreCore {
    pub(crate) et: Vec<ErrTerms>,
    /// `H̃ = Σ pⱼ aⱼaⱼᵀ + Ω⁻¹` — the first-order (Almquist) FOCEI Hessian.
    pub(crate) htilde: DMatrix<f64>,
    /// `H = ∂²nll/∂b²` — the **exact** conditional Hessian at `eta`.
    pub(crate) h_inner: DMatrix<f64>,
    pub(crate) g_ruv: Vec<f64>,
    pub(crate) gp_ruv: Vec<f64>,
    pub(crate) cens_dcz_df: Vec<f64>,
    pub(crate) cens_dcm_df: Vec<f64>,
    pub(crate) mult: Option<Vec<Vec<f64>>>,
    pub(crate) ruv_scale: f64,
    pub(crate) ruv: Option<usize>,
}

/// `∂(Σⱼ Lⱼ)/∂σ_k` at **fixed** predictions — the data term's σ-gradient, in natural σ.
///
/// σ enters `nll` only through the residual variance, so this is a closed-form scalar
/// computation at fixed `f`: no model evaluation, no inner solve, no Hessian. It is the
/// *data half* of [`sigma_block`], which additionally carries the FOCEI `log|H̃|` and
/// EBE-response terms that a fixed-`b` score must NOT have (#251 AGQ/Laplace).
///
/// Quantified rows use `∂L/∂R · ∂R/∂σ`. `R` is a quadratic form in σ
/// (`R = Σ_s (coeff_s(f)·mult_s·σ_s)²`), so the central difference of `R` is **exact** up
/// to rounding — there is no truncation error to trade against noise — and it inherits the
/// custom-magnitude, `iiv_on_ruv` and correlated-residual scalings for free rather than
/// re-deriving `∂R/∂σ` once per family. Censored rows use the `−logΦ(z)` kernel's own
/// σ-derivative through the shared [`censored_sigma_m_terms`], the same convention and FD
/// step `sigma_block` uses.
pub(crate) fn data_sigma_gradient(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    core: &ScoreCore,
) -> Vec<f64> {
    let sigma = &params.sigma.values;
    let n_sigma = sigma.len();
    let err_keys = model.error_spec.obs_keys(subject);
    // The **live** correlations (#847): a non-`FIX` `block_sigma` moves ρ, so the
    // σ FD has to differentiate the variance at the ρ the optimizer is proposing.
    let corrs = &params.residual_correlations;
    let correlated = !corrs.is_empty();
    let ipreds: Vec<f64> = sens.obs.iter().map(|o| o.f).collect();
    let mut out = vec![0.0f64; n_sigma];

    for k in 0..n_sigma {
        let h = sigma_fd_step(sigma[k]);
        let mut sp = sigma.clone();
        sp[k] += h;
        let mut sm = sigma.clone();
        sm[k] -= h;
        let (corr_sp, corr_sm) = if correlated {
            (
                Some(corr_residual_rd_at_sigma(
                    model, subject, &ipreds, &sp, corrs,
                )),
                Some(corr_residual_rd_at_sigma(
                    model, subject, &ipreds, &sm, corrs,
                )),
            )
        } else {
            (None, None)
        };

        let mut acc = 0.0;
        for (j, obs) in sens.obs.iter().enumerate() {
            let cmt = err_keys[j];
            let f = obs.f;
            let et = &core.et[j];
            if et.censored {
                let (_dg1, _ruv_sig, l_sig) = censored_sigma_m_terms(
                    model,
                    cmt,
                    subject.observations[j],
                    f,
                    &sp,
                    &sm,
                    h,
                    core.ruv_scale,
                    et.ruv_cz,
                    et.r,
                    core.ruv.is_some(),
                    et.cens_sign,
                );
                acc += l_sig;
                continue;
            }
            // `R` at σ±h, built exactly as `score_core` builds it: a FREM covariate
            // pseudo-observation takes the dedicated `EPSCOV` σ; otherwise the
            // correlation-aware diagonal, else the magnitude-scaled or legacy variance,
            // times the `iiv_on_ruv` scale.
            let r_at = |sa: &[f64], corr: &Option<(Vec<f64>, Vec<f64>)>| -> f64 {
                if let Some(v) = crate::stats::likelihood::build_frem_r_override(
                    model.frem_config.as_ref(),
                    &subject.fremtype,
                    sa,
                )
                .as_ref()
                .and_then(|o| o.get(j))
                .and_then(|x| *x)
                {
                    return v;
                }
                match corr {
                    Some((rv, _)) => rv[j],
                    None => match core.mult.as_ref().and_then(|m| m.get(j)) {
                        Some(m) => {
                            model.error_spec.variance_at_scaled(cmt, f, sa, &[], m) * core.ruv_scale
                        }
                        None => model.error_spec.variance_at(cmt, f, sa) * core.ruv_scale,
                    },
                }
            };
            let dr_dsig = (r_at(&sp, &corr_sp) - r_at(&sm, &corr_sm)) / (2.0 * h);
            // `∂L/∂R = (R − ε²)/(2R²)` for `L = ½(ε²/R + ln R)`.
            let (r, eps) = (et.r, et.eps);
            acc += 0.5 * (r - eps * eps) / (r * r) * dr_dsig;
        }
        out[k] = acc;
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn score_core(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
    n_eta: usize,
    omega_inv: &DMatrix<f64>,
    eta: &[f64],
    ruv: Option<usize>,
) -> Option<ScoreCore> {
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
        model.residual_var_scale(eta)
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
    let corr_diag = if !params.residual_correlations.is_empty() {
        Some(corr_residual_diag(
            model,
            subject,
            sens,
            sigma,
            &params.residual_correlations,
        )?)
    } else {
        None
    };
    // FREM covariate pseudo-observations (`fremtype > 0`): the objective scores these rows
    // against the dedicated covariate σ (`EPSCOV`), not `error_spec.variance_at(f)`. The
    // provider has already rewritten their *jet* (`apply_frem_pseudo_obs_jet` — prediction
    // `θ[ti] + η[ei]`, unit first derivatives, zero second); this is the matching *variance*
    // half. `R` is constant in `f` on such a row, hence `d = d2 = 0`. `None` for a non-FREM
    // model, so the common path allocates nothing.
    let frem_r = crate::stats::likelihood::build_frem_r_override(
        model.frem_config.as_ref(),
        &subject.fremtype,
        sigma,
    );
    for obs in sens.obs.iter() {
        let f = obs.f;
        // obs index → cmt: provider obs are parallel to subject.obs_times.
        let j = et.len();
        let cmt = err_keys[j];
        let frem_var = frem_r.as_ref().and_then(|o| o.get(j)).and_then(|x| *x);
        // `mult_row` is `None` for every observation on a non-magnitude model, so
        // that path keeps the exact legacy `variance_at`/`dvar_df`/`d2var_df2`
        // association bit-for-bit (the `_scaled` variants reassociate the
        // `f`-dependent term by ~1 ULP — see `residual_error::compute_r_matrix_with_correlations`).
        let mult_row: Option<&[f64]> = mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
        // Correlated residual (`block_sigma`, #627) uses the precomputed correlation-aware
        // diagonals; otherwise fall back to the per-obs (magnitude-scaled or legacy)
        // variance/derivatives. `block_sigma` and custom magnitude are mutually exclusive
        // (a `block_sigma` model has `mult == None`), so the two branches never mix.
        let (r, d, d2) = match (frem_var, &corr_diag) {
            // FREM pseudo-obs: `R = EPSCOV²`, independent of `f`.
            (Some(v), _) => (v, 0.0, 0.0),
            (None, Some((rv, dv, d2v))) => (rv[j], dv[j], d2v[j]),
            (None, None) => {
                let (r, d, d2) = residual_rd2(&model.error_spec, cmt, f, sigma, mult_row);
                (r * ruv_scale, d * ruv_scale, d2 * ruv_scale)
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
                let dd2 = model.error_spec.d2var_df2(cmt, ff, sigma) * ruv_scale;
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
        // `f`-derivative, just chain-ruled through `mult(θ)` instead of `f`. Never on a
        // FREM row: its `R` is the dedicated `EPSCOV²` override, independent of the PK
        // error-spec magnitude entirely, so `dr_dtheta`/`dd_dtheta` must stay empty
        // there exactly as for a censored or correlated row (#251 review #6).
        if let (Some(m), Some(mg_row)) = (
            mult_row.filter(|_| frem_var.is_none()),
            mult_grad.as_ref().and_then(|mg| mg.get(j)),
        ) {
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
        // `H[ruv,ruv] += 2ε²/R` and `H[ruv,l] += κⱼ a_{jl}`. Skipped entirely for a
        // FREM row: `individual_nll`'s FREM dispatch never applies `ruv_scale`, so such
        // a row's likelihood has zero η_ruv dependence and must leave `g_ruv`/`gp_ruv`
        // at their default `0.0` and the `(rr, ·)` block untouched (#251 review #5) —
        // otherwise the row's huge `1/R` (`R = EPSCOV²`) leaks a spurious O(1e8) term
        // into the `eta_ruv` Hessian/gradient.
        if let Some(rr) = ruv.filter(|_| frem_var.is_none()) {
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

    Some(ScoreCore {
        et,
        htilde,
        h_inner,
        g_ruv,
        gp_ruv,
        cens_dcz_df,
        cens_dcm_df,
        mult,
        ruv_scale,
        ruv,
    })
}

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
    let err_keys = model.error_spec.obs_keys(subject);
    let sigma = &params.sigma.values;
    let ScoreCore {
        et,
        htilde,
        h_inner,
        g_ruv,
        gp_ruv,
        cens_dcz_df,
        cens_dcm_df,
        mult,
        ruv_scale,
        ruv: _,
    } = score_core(
        model, subject, params, sens, n_eta, &omega_inv, eta_hat, ruv,
    )?;

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
                let d2 = model.error_spec.d2var_df2(cmt, ff, sigma) * ss;
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

    let entries: Vec<(usize, usize)> = lower_tri_entries(n_eta, params.omega.diagonal);

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
    // The **live** correlations (#847): a non-`FIX` `block_sigma` moves ρ, so the
    // σ FD has to differentiate the variance at the ρ the optimizer is proposing.
    let corrs = &params.residual_correlations;
    let correlated = !corrs.is_empty();
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
                Some(corr_residual_rd_at_sigma(
                    model, subject, &ipreds, &sp, corrs,
                )),
                Some(corr_residual_rd_at_sigma(
                    model, subject, &ipreds, &sm, corrs,
                )),
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
                    let d2 = model.error_spec.d2var_df2(cmt, f, sa) * prep.ruv_scale;
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
            // FREM covariate pseudo-observation (`fremtype > 0`): `R = EPSCOV²`,
            // independent of `f` (`d ≡ 0`), and — matching `individual_nll`'s FREM
            // dispatch — NOT scaled by `ruv_scale`/magnitude, so `r_sig` is the bare FD
            // of the covariate-σ variance, not lifted by `prep.ruv_scale` (#251 review
            // #2). A `None` `frem_row` (the common case) falls through to the ordinary
            // correlation/magnitude/legacy variance chain below.
            let frem_row = crate::stats::likelihood::build_frem_r_override(
                model.frem_config.as_ref(),
                &subject.fremtype,
                &sp,
            )
            .as_ref()
            .and_then(|o| o.get(j))
            .and_then(|x| *x)
            .zip(
                crate::stats::likelihood::build_frem_r_override(
                    model.frem_config.as_ref(),
                    &subject.fremtype,
                    &sm,
                )
                .as_ref()
                .and_then(|o| o.get(j))
                .and_then(|x| *x),
            );
            // Evaluate the four closed-form error functions once at σ±h and reuse
            // them for `r_sig`/`d_sig` and the residual-eta `g_sig` below. For a
            // correlated model (`block_sigma`) these are the correlation-aware variance
            // / `∂R/∂f`. Otherwise `mult` (if active) rides both perturbations unchanged
            // - it doesn't depend on σ.
            let mult_row: Option<&[f64]> =
                mult.as_ref().and_then(|m| m.get(j)).map(|v| v.as_slice());
            let (r_sig, d_sig) = if let Some((vp_frem, vm_frem)) = frem_row {
                ((vp_frem - vm_frem) / (2.0 * h), 0.0)
            } else {
                let (vp, vm, dp_var, dm_var) = match (&corr_sp, &corr_sm) {
                    (Some((rvp, dvp)), Some((rvm, dvm))) => (rvp[j], rvm[j], dvp[j], dvm[j]),
                    _ => {
                        let (vp, dp_var) = residual_rd(&model.error_spec, cmt, f, &sp, mult_row);
                        let (vm, dm_var) = residual_rd(&model.error_spec, cmt, f, &sm, mult_row);
                        (vp, vm, dp_var, dm_var)
                    }
                };
                // ∂R/∂σ_k, ∂d/∂σ_k by central FD. `et.r`/`et.d` carry the `exp(2·η_ruv)`
                // scale, so lift these too.
                let r_sig = prep.ruv_scale * (vp - vm) / (2.0 * h);
                let d_sig = prep.ruv_scale * (dp_var - dm_var) / (2.0 * h);
                // Residual-eta terms (#474). `∂R/∂σ` scales `R`, so:
                //   M[ruv] = ∂(1−ε²/R)/∂σ = ε²/R² · Rσ   (the residual-eta row of M)
                //   ∂log|H̃|/∂σ gains  (∂gⱼ/∂σ)·wⱼ[ruv]  with `gⱼ = d/R` (scale-free,
                //     so FD the unscaled quotient directly). Skipped for a FREM row
                //     above — its likelihood has no η_ruv dependence at all (#251
                //     review #5's principle, applied to σ).
                if let Some(rr) = prep.ruv {
                    m_vec[rr] += eps * eps / (r * r) * r_sig;
                    let g_sig = (dp_var / vp - dm_var / vm) / (2.0 * h);
                    fixed += g_sig * prep.w[j][rr];
                }
                (r_sig, d_sig)
            };

            let inv_r = 1.0 / r;
            let inv_r2 = inv_r * inv_r;
            let inv_r3 = inv_r2 * inv_r;

            // data + lnR:  ½ Rσ (R − ε²)/R²
            fixed += 0.5 * r_sig * (r - eps * eps) * inv_r2;
            // log|H̃|:  ½ (∂p/∂σ) q ,  ∂p/∂σ = −Rσ/R² + d·dσ/R² − d²Rσ/R³
            let dp = -r_sig * inv_r2 + d * d_sig * inv_r2 - d * d * r_sig * inv_r3;
            fixed += 0.5 * dp * prep.q[j];

            // ∂α/∂σ = [2ε/R² + d(2ε²−R)/R³] Rσ + [(R−ε²)/R²] dσ
            let dalpha = dalpha_dsigma(r, d, eps, r_sig, d_sig);
            for m in 0..n_eta {
                m_vec[m] += 0.5 * dalpha * obs.df_deta[m];
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

/// Per-observation `(∂R_jj/∂ρ_k, ∂d_j/∂ρ_k)` for every `block_sigma`
/// off-diagonal, indexed `[k][j]` (#847). Thin wrapper over the closed forms in
/// [`crate::stats::residual_error::dvar_drho`], transposed into the
/// correlation-major layout the ρ blocks below iterate in.
///
/// Diagonal-`R` only, which is what the analytic scope guarantees: `prepare` /
/// `prepare_stacked` bail to FD through `corr_residual_diag` when any off-diagonal
/// survives, so ρ can only reach here through the within-observation `combined`
/// cross term.
fn rho_rd_terms(
    model: &CompiledModel,
    subject: &Subject,
    sens: &SubjectSens,
    sigma: &[f64],
    corrs: &[ResidualCorrelation],
) -> Vec<Vec<(f64, f64)>> {
    let err_keys = model.error_spec.obs_keys(subject);
    let mut out = vec![vec![(0.0, 0.0); sens.obs.len()]; corrs.len()];
    let mut row = vec![(0.0, 0.0); corrs.len()];
    for (j, obs) in sens.obs.iter().enumerate() {
        crate::stats::residual_error::dvar_drho(
            &model.error_spec,
            err_keys[j],
            obs.f,
            sigma,
            corrs,
            &mut row,
        );
        for (k, &t) in row.iter().enumerate() {
            out[k][j] = t;
        }
    }
    out
}

/// Per-`block_sigma`-off-diagonal packed gradient `∂Fᵢ/∂z_k` for the **FOCEI**
/// marginal, in `pack_params` order (#847).
///
/// ρ is the exact analogue of a σ coordinate: it enters only through the residual
/// variance, so this is [`sigma_block`]'s plain (non-censored, non-FREM) row with
/// `(∂R/∂ρ, ∂d/∂ρ)` in place of `(∂R/∂σ, ∂d/∂σ)` — the same data + `lnR` term, the
/// same `log|H̃|` term, and the same EBE-response `M` accumulation. Unlike
/// `sigma_block` it needs no branches: `block_sigma` is mutually exclusive with
/// M3 censoring, FREM rows, `iiv_on_ruv`, custom residual magnitude and IOV, each
/// rejected up front, so every row here is the plain Sheiner–Beal term and
/// `prep.ruv_scale == 1` (kept in the expressions anyway, for parity with
/// `sigma_block` should one of those exclusions ever be lifted).
///
/// The optimizer coordinate is the Fisher-z `z = atanh(ρ)`, so the returned
/// gradient carries the `dρ/dz = 1 − ρ²` chain.
fn rho_block(
    prep: &Prep,
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
) -> Vec<f64> {
    let corrs = &params.residual_correlations;
    let mut grad = vec![0.0f64; corrs.len()];
    if corrs.is_empty() || sens.obs.is_empty() {
        return grad;
    }
    let n_eta = prep.n_eta;
    let terms = rho_rd_terms(model, subject, sens, &params.sigma.values, corrs);

    for (k, per_obs) in terms.iter().enumerate() {
        let mut fixed = 0.0;
        let mut m_vec = DVector::<f64>::zeros(n_eta);
        for (j, obs) in sens.obs.iter().enumerate() {
            let (r, d, eps) = (prep.et[j].r, prep.et[j].d, prep.et[j].eps);
            let (dr, dd) = per_obs[j];
            let r_rho = prep.ruv_scale * dr;
            let d_rho = prep.ruv_scale * dd;
            let inv_r = 1.0 / r;
            let inv_r2 = inv_r * inv_r;
            let inv_r3 = inv_r2 * inv_r;

            // data + lnR:  ½ Rρ (R − ε²)/R²
            fixed += 0.5 * r_rho * (r - eps * eps) * inv_r2;
            // log|H̃|:  ½ (∂p/∂ρ) q ,  ∂p/∂ρ = −Rρ/R² + d·dρ/R² − d²Rρ/R³
            let dp = -r_rho * inv_r2 + d * d_rho * inv_r2 - d * d * r_rho * inv_r3;
            fixed += 0.5 * dp * prep.q[j];
            // EBE response M[m] += ½ (∂α/∂ρ) ∂f/∂η_m
            let dalpha = dalpha_dsigma(r, d, eps, r_rho, d_rho);
            for m in 0..n_eta {
                m_vec[m] += 0.5 * dalpha * obs.df_deta[m];
            }
        }

        let deta = -(&prep.h_inner_inv * m_vec);
        let mut resp = 0.0;
        for l in 0..n_eta {
            resp += prep.g_eta[l] * deta[l];
        }
        grad[k] = (fixed + 0.5 * resp) * rho_chain(corrs[k].rho);
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

    let entries: Vec<(usize, usize)> = lower_tri_entries(n_eta, params.omega.diagonal);

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
        let dtheta_dx = theta_dx_chain(template, &params.theta, m);
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
        let dtheta_dx = theta_dx_chain(template, &params.theta, m);
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

    // ρ: the `block_sigma` off-diagonals (#847), packed last — `rho_packed_start`
    // owns that offset, so this never re-derives it from n_theta/n_omega/n_sigma
    // (which would land on the mixture segment for a mixture model). The Fisher-z
    // chain is already applied inside `rho_block`.
    let rho_start = rho_packed_start(template);
    for (k, &val) in rho_block(&prep, model, subject, &params, &sens)
        .iter()
        .enumerate()
    {
        g[rho_start + k] = val;
    }

    Some(g)
}

/// Shared all-or-nothing population sum `d(OFV)/dx = 2·Σᵢ per_subject(i)` in
/// packed space, short-circuiting to `None` if any subject is unsupported and
/// zeroing fixed coordinates. The per-subject closure is the only difference
/// between the FOCEI ([`population_gradient_sens`]) and FOCE
/// ([`population_gradient_sens_foce`]) forms.
///
/// Subject-parallel (the FD path this replaces was already subject-parallel;
/// PR #381 review #7). `collect::<Option<_>>` short-circuits to `None` if any
/// subject is out of analytic scope, and preserves subject order so the
/// accumulation below is bit-reproducible across runs.
fn population_sum(
    population: &Population,
    template: &ModelParameters,
    n: usize,
    per_subject: impl Fn(usize, &Subject) -> Option<Vec<f64>> + Sync,
) -> Option<Vec<f64>> {
    let grads: Vec<Vec<f64>> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| per_subject(i, subject))
        .collect::<Option<Vec<_>>>()?;
    let mut grad = vec![0.0f64; n];
    for gi in &grads {
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
    population_sum(population, template, x.len(), |i, subject| {
        subject_packed_gradient(model, subject, template, x, eta_hats[i].as_slice())
    })
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
///
/// `mixture_class` (1-based) is `Some(k)` for the mixture outer gradient (#977):
/// the `MIXTURE_CLASS` thread-local is set **inside** each rayon worker's closure
/// so `MIXNUM` branches in the typical value resolve to class `k`. The guard the
/// mixture objective set on the *calling* thread does not reach rayon workers, so
/// without this every class's analytic sensitivity would silently read the
/// default class 1. `None` leaves the thread-local at its default (non-mixture).
pub fn per_subject_packed_gradients(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[DVector<f64>],
    interaction: bool,
    mixture_class: Option<usize>,
) -> Vec<Option<Vec<f64>>> {
    population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            // Set the mixture-class thread-local on *this* worker thread (the
            // outer-thread guard does not propagate into rayon workers).
            let _guard = mixture_class.map(crate::parser::model_parser::MixtureClassGuard::enter);
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
    let correlated = !params.residual_correlations.is_empty();
    let corr = &params.residual_correlations;
    let corr_rd0 = if correlated {
        Some(corr_residual_diag(model, subject, &sens0, sigma, corr)?)
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
        // FREM covariate pseudo-observation (`fremtype > 0`): the Sheiner–Beal marginal
        // still needs an `R⁰ⱼ` for this row, but the objective scores it against the
        // dedicated `EPSCOV` variance, not `error_spec.variance_at`. `R⁰ = EPSCOV²` is
        // constant in `f`, hence `d⁰ = 0`, and it takes neither the correlation-aware
        // nor the magnitude-scaled branch below (#251 review #3 — `method = foce` had
        // no FREM branch at all, unlike FOCEI's `score_core`).
        let frem_r0 = crate::stats::likelihood::build_frem_r_override(
            model.frem_config.as_ref(),
            &subject.fremtype,
            sigma,
        )
        .as_ref()
        .and_then(|o| o.get(j))
        .and_then(|x| *x);
        // Correlated residual (`block_sigma`, #627): correlation-aware `(R⁰, ∂R⁰/∂f)`.
        // block_sigma is mutually exclusive with custom magnitude and M3, so `mult_row`
        // and the censored (`cg`) path are inactive whenever `corr_rd0` is set.
        let (r, dd) = match (frem_r0, &corr_rd0) {
            (Some(v), _) => (v, 0.0),
            (None, Some((rv, dv, _))) => (rv[j], dv[j]),
            (None, None) => residual_rd(&model.error_spec, cmt, f0act, sigma, mult_row),
        };
        if !(r.is_finite() && r > 0.0) {
            return None;
        }
        r0[i] = r;
        d0[i] = dd;
        // The magnitude direct-θ channel is a PK-error-spec derivative and does not
        // apply to a FREM row's dedicated `EPSCOV` variance (#251 review #6's
        // principle): `dr0_dtheta[i]` stays empty there, exactly as for a censored or
        // correlated row.
        if frem_r0.is_none() {
            if let (Some(mm), Some(mg_row)) =
                (mult_row, mult_grad.as_ref().and_then(|mg| mg.get(j)))
            {
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
        let dtheta_dx = theta_dx_chain(template, &params.theta, m);
        fixed[m] = nat * dtheta_dx;
    }

    // Ω (fixed η̂, packed Cholesky-L): SB over quant rows + the marginal censored
    // variance's direct Ω-gradient — R̃ⱼⱼ = Jⱼ Ω Jⱼᵀ + R⁰ⱼ depends on Ω (#646), added
    // via `cg.omega_entry` (was zero when the censored term used the residual R⁰).
    let l = &params.omega.chol;
    let jl = &jmat * l;
    let cjl = cg.prep_jl(l); // (Jⱼ L) per censored row, once
    let entries: Vec<(usize, usize)> = lower_tri_entries(n_eta, params.omega.diagonal);
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
            // FREM covariate pseudo-observation: `R⁰ = EPSCOV²`, so `∂R⁰/∂σ` comes from
            // the dedicated covariate σ, not the PK error model (#251 review #3). Without
            // this branch `dr0` FD's the PK variance's (zero) dependence on `EPSCOV`,
            // leaving `grad[EPSCOV] ≡ 0` under `method = foce` too.
            let frem_vp = crate::stats::likelihood::build_frem_r_override(
                model.frem_config.as_ref(),
                &subject.fremtype,
                &sp,
            )
            .as_ref()
            .and_then(|o| o.get(j))
            .and_then(|x| *x);
            let frem_vm = crate::stats::likelihood::build_frem_r_override(
                model.frem_config.as_ref(),
                &subject.fremtype,
                &sm,
            )
            .as_ref()
            .and_then(|o| o.get(j))
            .and_then(|x| *x);
            if let (Some(vp), Some(vm)) = (frem_vp, frem_vm) {
                let dr0 = (vp - vm) / (2.0 * hsig);
                nat += 0.5 * dr0 * (rtilde_inv[(i, i)] - u[i] * u[i]);
                continue;
            }
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

    // ρ (`block_sigma` off-diagonals, #847). The FOCE marginal is first-order in
    // R, so ρ enters only through `R⁰` — frozen at f(η=0), hence the closed forms
    // are taken at `sens0`, exactly like the σ loop's `f0act`. Same
    // `½ ∂R⁰/∂ρ (R̃⁻¹ᵢᵢ − uᵢ²)` shape, with the Fisher-z chain. No censored
    // companion term: `block_sigma` + M3 is rejected up front, so `cg` carries no
    // ρ block to add.
    if !params.residual_correlations.is_empty() {
        let rho_start = rho_packed_start(template);
        let rho_terms = rho_rd_terms(model, subject, &sens0, sigma, &params.residual_correlations);
        for (k, per_obs) in rho_terms.iter().enumerate() {
            let mut nat = 0.0;
            for (i, &j) in quant.iter().enumerate() {
                let (dr0, _) = per_obs[j];
                nat += 0.5 * dr0 * (rtilde_inv[(i, i)] - u[i] * u[i]);
            }
            fixed[rho_start + k] = nat * rho_chain(params.residual_correlations[k].rho);
        }
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
    population_sum(population, template, x.len(), |i, subject| {
        subject_packed_gradient_foce(model, subject, template, x, eta_hats[i].as_slice())
    })
}

/// θ→packed chain rule `∂θ/∂x`: `θ` when the parameter packs in log-space,
/// else `1.0`. Shared by every θ-loop of the packed-gradient / eta-dx functions.
#[inline]
fn theta_dx_chain(template: &ModelParameters, theta: &[f64], m: usize) -> f64 {
    if theta_packs_log(template.theta_lower[m]) {
        theta[m]
    } else {
        1.0
    }
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
        let dtheta_dx = theta_dx_chain(template, &params.theta, m);
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
            let (var_p, dvar_p) = residual_rd(&model.error_spec, cmt, f, &sp, mult_row);
            let (var_m, dvar_m) = residual_rd(&model.error_spec, cmt, f, &sm, mult_row);
            let r_sig = (var_p - var_m) / (2.0 * h);
            let d_sig = (dvar_p - dvar_m) / (2.0 * h);
            let dalpha = dalpha_dsigma(r, d, eps, r_sig, d_sig);
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
/// `censored_marginal_foce_grad`.
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
        let (r, d) = residual_rd(&model.error_spec, cmt, f0act, sigma, mult_row);
        if !(r.is_finite() && r > 0.0) {
            return None;
        }
        r0[i] = r;
        d0[i] = d;
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
        let dtheta_dx = theta_dx_chain(template, &params.theta, m);
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
        let dtheta_dx = theta_dx_chain(template, &params.theta, m);
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
    let entries: Vec<(usize, usize)> = lower_tri_entries(n_eta, params.omega.diagonal);
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
    // Live, not declared (#847): the values fed to `corr_residual_rd_at_sigma`
    // below come from `params`, so the predicate that gates them must too, or a
    // hand-built `ModelParameters` could disagree with the model about whether a
    // correlation exists at all.
    let correlated = !params.residual_correlations.is_empty();
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
                    &params.residual_correlations,
                )),
                Some(corr_residual_rd_at_sigma(
                    model,
                    subject,
                    &eta_dx_ipreds,
                    &sm,
                    &params.residual_correlations,
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
            // FREM covariate pseudo-observation: `R = EPSCOV²`, independent of `f`
            // (`d ≡ 0`) and — matching `individual_nll` — not scaled by `ruv_scale`, so
            // `r_sig` is the bare FD of the covariate-σ variance (#251 review #4). It
            // also skips the residual-eta row below: a FREM row's likelihood has no
            // η_ruv dependence (#251 review #5's principle, applied here too).
            let frem_row = crate::stats::likelihood::build_frem_r_override(
                model.frem_config.as_ref(),
                &subject.fremtype,
                &sp,
            )
            .as_ref()
            .and_then(|o| o.get(j))
            .and_then(|x| *x)
            .zip(
                crate::stats::likelihood::build_frem_r_override(
                    model.frem_config.as_ref(),
                    &subject.fremtype,
                    &sm,
                )
                .as_ref()
                .and_then(|o| o.get(j))
                .and_then(|x| *x),
            );
            // `et.r`/`et.d` carry the `exp(2·η_ruv)` scale *and* any custom-magnitude
            // `mult`, so lift `∂R/∂σ`,`∂d/∂σ` the same way (mirrors `sigma_block`);
            // `ruv_scale == 1` when there is no ruv, `mult_row = None` for bare sigma.
            // For a correlated model (`block_sigma`) these use the correlation-aware
            // variance / `∂R/∂f` (mutually exclusive with custom magnitude).
            let mult_row: Option<&[f64]> =
                mult.as_ref().and_then(|mm| mm.get(j)).map(|v| v.as_slice());
            let (r_sig, d_sig) = if let Some((vp_frem, vm_frem)) = frem_row {
                ((vp_frem - vm_frem) / (2.0 * h), 0.0)
            } else {
                let (var_p, var_m, dvar_p, dvar_m) = match (&corr_sp, &corr_sm) {
                    (Some((rvp, dvp)), Some((rvm, dvm))) => (rvp[j], rvm[j], dvp[j], dvm[j]),
                    _ => {
                        let (var_p, dvar_p) = residual_rd(&model.error_spec, cmt, f, &sp, mult_row);
                        let (var_m, dvar_m) = residual_rd(&model.error_spec, cmt, f, &sm, mult_row);
                        (var_p, var_m, dvar_p, dvar_m)
                    }
                };
                let r_sig = prep.ruv_scale * (var_p - var_m) / (2.0 * h);
                let d_sig = prep.ruv_scale * (dvar_p - dvar_m) / (2.0 * h);
                // Residual-eta row of M (#474): `M[ruv] = ∂(1−ε²/R)/∂σ = ε²/R²·Rσ`.
                // Skipped for a FREM row (handled above).
                if let Some(rr) = prep.ruv {
                    mvec[rr] += eps * eps / (r * r) * r_sig;
                }
                (r_sig, d_sig)
            };
            let dalpha = dalpha_dsigma(r, d, eps, r_sig, d_sig);
            for m in 0..n_eta {
                mvec[m] += 0.5 * dalpha * obs.df_deta[m];
            }
        }
        out[sigma_start + k] = -(&prep.h_inner_inv * mvec) * sigma[k];
    }

    // ρ coords (#847): `M_ρ = ½ Σⱼ ∂αⱼ/∂ρ · aⱼ`, the σ loop's body with
    // `(∂R/∂ρ, ∂d/∂ρ)` swapped in. The inner objective is interaction-independent,
    // so this one Jacobian serves both the FOCE and FOCEI callers — the same
    // reason the σ columns are shared. Every `sigma_block`-style branch is
    // inapplicable here (see `rho_block`), so the plain row is the only one.
    if !params.residual_correlations.is_empty() {
        let rho_start = rho_packed_start(template);
        let rho_terms = rho_rd_terms(model, subject, &sens, sigma, &params.residual_correlations);
        for (k, per_obs) in rho_terms.iter().enumerate() {
            let mut mvec = DVector::<f64>::zeros(n_eta);
            for (j, obs) in sens.obs.iter().enumerate() {
                let (r, d, eps) = (prep.et[j].r, prep.et[j].d, prep.et[j].eps);
                let (dr, dd) = per_obs[j];
                let dalpha = dalpha_dsigma(r, d, eps, prep.ruv_scale * dr, prep.ruv_scale * dd);
                for m in 0..n_eta {
                    mvec[m] += 0.5 * dalpha * obs.df_deta[m];
                }
            }
            out[rho_start + k] =
                -(&prep.h_inner_inv * mvec) * rho_chain(params.residual_correlations[k].rho);
        }
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
pub(crate) fn mixed_eta_theta(
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
#[path = "sens_outer_gradient_tests.rs"]
mod tests;

/// The full per-subject FOCEI **natural** gradient `[∂Fᵢ/∂θ, ∂Fᵢ/∂Ω, ∂Fᵢ/∂σ]` — θ, then the
/// free Ω entries in `pack_params` lower-triangle order, then σ.
///
/// Built from an already-computed `prep`/`sens` so the analytic covariance Hessian
/// ([`crate::estimation::sens_cov_hessian`]) can pair it with the natural Hessian without
/// recomputing the sensitivities. Same three blocks as the public per-axis gradients, so the
/// gradient the Hessian is paired with is the one the outer loop actually descends.
pub(crate) fn subject_natural_gradient(
    prep: &Prep,
    sens: &SubjectSens,
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta_hat: &[f64],
) -> Vec<f64> {
    let mut g = theta_block(prep, sens, params.theta.len());
    g.extend(omega_block(prep, params, eta_hat));
    g.extend(sigma_block(prep, model, subject, params, sens));
    g
}
