//! The FOCE/FOCEI subject objective `F` and the inner objective `l`, written once over a
//! generic numeric type, as functions of the predictions `f`, their η-sensitivities
//! `G = ∂f/∂η`, `η`, and the population parameters the likelihood reads ([`ParT`]: Ω by
//! its Cholesky factor, σ, `block_sigma` ρ, the serial ρ, θ for FREM rows).
//!
//! The adjoint gradient (`sens::ode_provider::adjoint`) evaluates them twice:
//! * over a forward η-dual in f64, for `∇_η F` and `H = ∇²_η l`;
//! * on the scalar reverse tape [`R`](crate::sens::rtape::R), for `Ψ = F − wᵀ∇_η l` and
//!   its partials with respect to every input, from one sweep and with no hand-derived
//!   seeds or finite differences.
//!
//! Every formula mirrors the f64 production objective term for term:
//! * FOCEI, diagonal residual: `foce_subject_nll_interaction` / `gaussian_foce_accum`,
//!   `H̃ = Σ aᵀa/V + ½·c̃ᵀc̃ + cens_hess + Ω⁻¹` ([`focei_f_grad`]: one dual pass per row
//!   gives both `F` and `∇_η l`);
//! * FOCEI, dense residual: `foce_subject_nll_interaction_dense`,
//!   `H̃ = GᵀR⁻¹G + ½B + Ω⁻¹`, `B_kl = tr(R⁻¹∂_kR R⁻¹∂_lR)`;
//! * FOCE: `foce_subject_nll_standard_cov`'s Sheiner–Beal marginal ([`foce_f`]);
//! * the inner `l` the EBE minimises (`individual_nll_into_prepared_with_schedule_frozen`),
//!   conditional or with FOCE's frozen variance ([`inner_l`], [`inner_grad`]).
//!
//! The objective-identity, parity and reconverged-FD tests pin that correspondence. On
//! the dense path, `∂R/∂η_k` is not hand-derived: `R` is built over an η-dual whose
//! predictions carry `G`, so its tangents are exactly `Σ_m G_mk ∂R/∂f_m`.
#![allow(clippy::needless_range_loop)]

use crate::sens::num::PkNum;
use crate::sens::rtape::DualT;
use crate::stats::residual_error::ResidualCov;
use crate::types::{CompiledModel, ModelParameters, Subject};

/// The population parameters the likelihood reads directly, as kernel values: Ω (by
/// its Cholesky factor, with the `Ω⁻¹` and `log|Ω|` built from it), σ, the
/// `block_sigma` ρ's (in `residual_correlations` order), the serial-correlation ρ, and
/// θ where it enters the likelihood directly (FREM pseudo-predictions).
#[derive(Clone)]
pub(crate) struct ParT<T> {
    pub omega: OmegaT<T>,
    pub sigma: Vec<T>,
    pub rho: Vec<T>,
    pub serial_rho: Option<T>,
    pub theta: Vec<T>,
    /// Per observation, the custom residual magnitude row (`RuvMagnitude::eval_obs`'s
    /// layout: σ-slot multipliers, then exponents under a `power(...)` form), as functions
    /// of `θ`; `None` without a custom magnitude.
    pub ruv_mult: Option<Vec<Vec<T>>>,
}

/// Ω as kernel values: the lower Cholesky factor `L` (`Ω = LLᵀ`, the packed parameters),
/// and `Ω⁻¹ = L⁻ᵀL⁻¹`, `log|Ω| = 2Σ ln L_kk` computed from it on `T`.
#[derive(Clone)]
pub(crate) struct OmegaT<T> {
    pub chol: Vec<Vec<T>>,
    pub inv: Vec<Vec<T>>,
    pub log_det: T,
}

impl<T: PkNum> OmegaT<T> {
    pub(crate) fn from_chol(chol: Vec<Vec<T>>) -> Option<Self> {
        let n = chol.len();
        let c = |x: f64| T::from_f64(x);
        // L⁻¹, lower triangular, by forward substitution.
        let mut li = vec![vec![c(0.0); n]; n];
        for i in 0..n {
            if !(chol[i][i].val() > 0.0) {
                return None;
            }
            li[i][i] = c(1.0) / chol[i][i];
            for j in 0..i {
                let s = T::wsum3(0.0, (j..i).map(|k| (c(1.0), chol[i][k], li[k][j])));
                li[i][j] = -s / chol[i][i];
            }
        }
        let mut inv = vec![vec![c(0.0); n]; n];
        for a in 0..n {
            for b in 0..=a {
                let e = T::wsum3(0.0, (a..n).map(|k| (c(1.0), li[k][a], li[k][b])));
                inv[a][b] = e;
                inv[b][a] = e;
            }
        }
        let mut log_det = c(0.0);
        for k in 0..n {
            log_det = log_det + c(2.0) * chol[k][k].ln();
        }
        Some(OmegaT { chol, inv, log_det })
    }
}

impl ParT<f64> {
    pub(crate) fn of(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> Self {
        let n = params.omega.chol.nrows();
        let l = &params.omega.chol;
        ParT {
            omega: OmegaT {
                chol: (0..n)
                    .map(|r| (0..n).map(|c| l[(r, c)]).collect())
                    .collect(),
                inv: (0..n)
                    .map(|r| (0..n).map(|c| params.omega.inv[(r, c)]).collect())
                    .collect(),
                log_det: params.omega.log_det,
            },
            sigma: params.sigma.values.clone(),
            rho: params.residual_correlations.iter().map(|c| c.rho).collect(),
            serial_rho: model
                .residual_serial
                .as_ref()
                .map(|s| params.theta.get(s.rho_theta).copied().unwrap_or(f64::NAN)),
            theta: params.theta.clone(),
            ruv_mult: model.ruv_obs_mult(subject, &params.theta),
        }
    }
}

impl<T: Copy> ParT<T> {
    /// The same parameters through `f` (lifting to a dual over `T`, say).
    pub(crate) fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> ParT<U> {
        let mm = |m: &Vec<Vec<T>>| {
            m.iter()
                .map(|r| r.iter().map(|&x| f(x)).collect())
                .collect()
        };
        ParT {
            omega: OmegaT {
                chol: mm(&self.omega.chol),
                inv: mm(&self.omega.inv),
                log_det: f(self.omega.log_det),
            },
            sigma: self.sigma.iter().map(|&x| f(x)).collect(),
            rho: self.rho.iter().map(|&x| f(x)).collect(),
            serial_rho: self.serial_rho.map(&f),
            theta: self.theta.iter().map(|&x| f(x)).collect(),
            ruv_mult: self.ruv_mult.as_ref().map(|m| {
                m.iter()
                    .map(|r| r.iter().map(|&x| f(x)).collect())
                    .collect()
            }),
        }
    }
}

impl<T: PkNum> ParT<T> {
    /// Lift to a first-order dual over `T` (constant in the dual's directions).
    pub(crate) fn wrap<const K: usize>(&self) -> ParT<DualT<K, T>> {
        self.map(|x| DualT {
            v: x,
            g: [T::from_f64(0.0); K],
        })
    }
}

/// Subject-static inputs of the kernel at one parameter set.
pub(crate) struct LikCtx<'a> {
    pub model: &'a CompiledModel,
    pub subject: &'a Subject,
    pub params: &'a ModelParameters,
    pub cov: ResidualCov<'a>,
    /// Per row, its σ loadings' shape `(slot, c₀, slope)`: `c(f) = c₀ + slope·f` (constant
    /// or proportional, `ErrorSpec::sigma_loadings`), f-independent, so read once here.
    load_shape: Vec<[(usize, f64, f64); MAX_LOADINGS]>,
    load_len: Vec<usize>,
}

/// More σ loadings than any error form gives one row (combined: 2).
const MAX_LOADINGS: usize = 4;

impl<'a> LikCtx<'a> {
    pub(crate) fn new(
        model: &'a CompiledModel,
        subject: &'a Subject,
        params: &'a ModelParameters,
    ) -> Self {
        let keys = model.error_spec.obs_keys(subject);
        let cov = model.residual_cov_with(&params.residual_correlations, &params.theta);
        let n_sigma = params.sigma.values.len();
        let (mut load_shape, mut load_len) = (
            Vec::with_capacity(keys.len()),
            Vec::with_capacity(keys.len()),
        );
        for &k in keys.iter() {
            let slopes = model.error_spec.sigma_loading_slopes(k, n_sigma);
            let mut row = [(0usize, 0.0, 0.0); MAX_LOADINGS];
            let at0 = model.error_spec.sigma_loadings(k, 0.0, n_sigma);
            for (i, &(idx, c0)) in at0.iter().take(MAX_LOADINGS).enumerate() {
                row[i] = (
                    idx,
                    c0,
                    slopes.iter().find(|s| s.0 == idx).map_or(0.0, |s| s.1),
                );
            }
            load_shape.push(row);
            load_len.push(at0.len().min(MAX_LOADINGS));
        }
        LikCtx {
            model,
            subject,
            params,
            cov,
            load_shape,
            load_len,
        }
    }

    /// Both `F` and `l` take the dense-`R` form when `R` can couple rows and the subject
    /// has no censored or FREM rows (`foce_subject_nll` and
    /// `individual_nll_into_prepared_with_schedule_frozen`); otherwise both score it row
    /// by row, with the live correlated variance.
    ///
    /// A FREM row is any `FREMTYPE > 0`, mapped by `[frem]` or not: production's own test
    /// (`subject.fremtype.iter().any(|&ft| ft > 0)`, `stats::likelihood`), so an unmapped
    /// non-zero `FREMTYPE` is scored row by row here too (#2093 review).
    fn dense(&self) -> bool {
        self.cov.is_dense()
            && !self.subject.fremtype.iter().any(|&ft| ft > 0)
            && !(0..self.subject.observations.len()).any(|j| self.censored(j).is_some())
    }

    /// A FREM covariate pseudo-observation row: `(θ index, η index, covariate σ index)`
    /// of its prediction `θ_i + η_e` and fixed variance `σ_c²`.
    #[inline]
    fn frem_row(&self, j: usize) -> Option<(usize, usize, usize)> {
        let fc = self.model.frem_config.as_ref()?;
        let ft = self.subject.fremtype.get(j).copied().unwrap_or(0);
        if ft == 0 {
            return None;
        }
        fc.fremtype_to_indices
            .get(&ft)
            .map(|&(ti, ei)| (ti, ei, fc.covariate_sigma_index))
    }

    /// A FREM row's `(prediction, variance)`: `θ_i + η_e` and `max(σ_c², 1e-12)`
    /// (`build_frem_r_override`).
    fn frem_pred_var<T: PkNum>(
        &self,
        (ti, ei, ci): (usize, usize, usize),
        eta: &[T],
        p: &ParT<T>,
    ) -> (T, T) {
        let s = p.sigma[ci];
        (p.theta[ti] + eta[ei], (s * s).guard_floor(1e-12))
    }

    #[inline]
    fn censored(&self, j: usize) -> Option<i8> {
        // `bloq_in_force`, not the model field: a fit can arm the censoring method per call
        // (`BloqOverrideGuard`), and the objective scores rows under that.
        let c = self.subject.cens.get(j).copied().unwrap_or(0);
        self.model.bloq_in_force().is_censored_row(c).then_some(c)
    }

    /// Row `j`'s σ loadings at prediction `f`, as generic values: a constant slot's
    /// coefficient, a proportional slot's `f` — raised to `sign(f)·|f|^p` under a
    /// `power(...)` exponent — each times the custom magnitude multiplier. The same
    /// loadings as `ErrorSpec::scaled_loadings`, built in `T` arithmetic so every order
    /// of derivative in `f`, `θ` and `η` is exact without a hand-written rule. Returned
    /// in a fixed buffer with its length (no allocation per row).
    fn loadings<T: PkNum>(
        &self,
        j: usize,
        f: T,
        p: &ParT<T>,
    ) -> ([(usize, T); MAX_LOADINGS], usize) {
        let n_sigma = p.sigma.len();
        let mult = p.ruv_mult.as_ref().and_then(|m| m.get(j));
        let mut out = [(0usize, T::from_f64(0.0)); MAX_LOADINGS];
        let len = self.load_len[j];
        for (o, &(idx, c0, slope)) in out.iter_mut().zip(&self.load_shape[j][..len]) {
            let c = if slope == 0.0 {
                T::from_f64(c0)
            } else {
                // `slot_exponent`: the exponent half of the row, when present.
                match mult
                    .filter(|m| m.len() == 2 * n_sigma)
                    .map(|m| m[n_sigma + idx])
                {
                    Some(pw) if pw.val() != 1.0 => {
                        if f.val() == 0.0 {
                            T::from_f64(0.0)
                        } else {
                            T::from_f64(if f.val() < 0.0 { -1.0 } else { 1.0 }) * f.abs().pow(pw)
                        }
                    }
                    _ => T::from_f64(c0) + f * T::from_f64(slope),
                }
            };
            let m = mult
                .and_then(|m| m.get(idx).copied())
                .unwrap_or(T::from_f64(1.0));
            *o = (idx, m * c);
        }
        (out, len)
    }

    /// `V` of row `j` at `f`: the loadings variance, unscaled.
    fn loadings_variance<T: PkNum>(&self, j: usize, f: T, p: &ParT<T>) -> T {
        let (l, n) = self.loadings(j, f, p);
        crate::stats::residual_error::loadings_variance_g(
            &l[..n],
            &p.sigma,
            &self.params.residual_correlations,
            |k| p.rho[k],
        )
    }

    /// `(V, ∂V/∂f, ∂²V/∂f²)` of row `j` at `f`, times the RUV scale: [`Self::loadings`]
    /// over a second-order dual in `f`.
    fn variance<T: PkNum>(&self, j: usize, f: T, ruv: T, p: &ParT<T>) -> (T, T, T) {
        type D2<T> = DualT<1, DualT<1, T>>;
        let one = T::from_f64(1.0);
        let zero = T::from_f64(0.0);
        let f2: D2<T> = DualT {
            v: DualT { v: f, g: [one] },
            g: [DualT { v: one, g: [zero] }],
        };
        let lift = |x: T| -> D2<T> {
            DualT {
                v: DualT { v: x, g: [zero] },
                g: [DualT { v: zero, g: [zero] }],
            }
        };
        let v = self.loadings_variance(j, f2, &p.map(lift));
        (v.v.v * ruv, v.v.g[0] * ruv, v.g[0].g[0] * ruv)
    }

    /// `V` of row `j` at `f` (times the RUV scale).
    fn variance_v<T: PkNum>(&self, j: usize, f: T, ruv: T, p: &ParT<T>) -> T {
        self.loadings_variance(j, f, p) * ruv
    }

    /// The IIV-on-RUV variance scale `exp(2η_r)` (`1` without RUV IIV).
    fn ruv_scale<T: PkNum>(&self, eta: &[T]) -> T {
        match self.model.residual_error_eta {
            Some(r) => (T::from_f64(2.0) * eta[r]).exp(),
            None => T::from_f64(1.0),
        }
    }

    /// `ηᵀΩ⁻¹η + log|Ω|`.
    fn prior<T: PkNum>(&self, eta: &[T], p: &ParT<T>) -> T {
        let n = eta.len();
        let oi = &p.omega.inv;
        T::wsum3(
            0.0,
            (0..n)
                .flat_map(|k| (0..n).map(move |l| (k, l)))
                .map(|(k, l)| (oi[k][l], eta[k], eta[l])),
        ) + p.omega.log_det
    }

    /// The residual covariance over `T` (the dense path).
    fn r_matrix<T: PkNum>(&self, f: &[T], p: &ParT<T>) -> Vec<Vec<T>> {
        crate::stats::residual_error::r_matrix_g(
            |j| {
                let (l, n) = self.loadings(j, f[j], p);
                l[..n].to_vec()
            },
            f.len(),
            self.subject,
            &p.sigma,
            self.cov,
            |k| p.rho[k],
            p.serial_rho,
        )
    }
}

/// `log Φ(z)` with production's value (`log_normal_cdf`) and its exact derivatives
/// (`h = φ/Φ`, `h′ = −h(z + h)`, `h″`).
#[inline]
fn log_phi<T: PkNum>(z: T) -> T {
    let zv = z.val();
    let h = crate::stats::special::inv_mills(zv);
    let h1 = -h * (zv + h);
    // (log Φ)‴ = h″ = −h′(z + h) − h(1 + h′): the Laplace kernel's second-order dual
    // differentiates the `(log Φ)″` factor; (log Φ)⁗ = h‴ = −h″(z + 2h) − 2h′(1 + h′), for
    // the covariance step's nested dual.
    let h2 = -h1 * (zv + h) - h * (1.0 + h1);
    z.lift4(
        crate::stats::special::log_normal_cdf(zv),
        h,
        h1,
        h2,
        -h2 * (zv + 2.0 * h) - 2.0 * h1 * (1.0 + h1),
    )
}

/// Inverse Mills ratio `h(z) = φ/Φ` with `h′ = −h(z + h)`, lifted to fourth order so the
/// exact anchor's nested covariance dual differentiates it rather than poisoning (`lift2` left
/// `h‴` unknown, #2093 review).
#[inline]
fn inv_mills<T: PkNum>(z: T) -> T {
    let zv = z.val();
    let h = crate::stats::special::inv_mills(zv);
    let h1 = -h * (zv + h);
    // h″ = −h′(z + h) − h(1 + h′); h‴ = −h″(z + 2h) − 2h′(1 + h′);
    // h⁗ = −h‴(z + 2h) − 3h″(1 + 2h′).
    let h2 = -h1 * (zv + h) - h * (1.0 + h1);
    let h3 = -h2 * (zv + 2.0 * h) - 2.0 * h1 * (1.0 + h1);
    let h4 = -h3 * (zv + 2.0 * h) - 3.0 * h2 * (1.0 + 2.0 * h1);
    z.lift4(h, h1, h2, h3, h4)
}

/// One censored row's `−2·log Φ(z)` and the `H̃` coefficients `(g2, cz, cm)` of
/// `m3_censored_outer`, generic.
fn m3_terms<T: PkNum>(y: f64, f: T, v: T, d: T, d2: T, cens: i8) -> (T, T, T, T) {
    let sgn = if cens < 0 { -1.0 } else { 1.0 };
    let c = |x: f64| T::from_f64(x);
    let w = v.sqrt();
    let r = c(y) - f;
    let z = c(sgn) * r / w;
    let h = inv_mills(z);
    let m = c(sgn) * (c(1.0) / w + r * d / (c(2.0) * v * w));
    let w3 = v * w;
    let w5 = v * v * w;
    let dm_df =
        c(sgn) * ((c(-2.0) * d + r * d2) / (c(2.0) * w3) - c(3.0) * r * d * d / (c(4.0) * w5));
    let g2 = h * (h + z) * m * m + h * dm_df;
    let cc = h * (z * z + h * z - c(1.0));
    (c(-2.0) * log_phi(z), g2, cc * z, cc * m)
}

/// FOCEI `F` and the conditional `∇_η l` together. On the diagonal path one dual pass per
/// row, in the directions the row reads (its prediction, and `η_r` under RUV IIV), gives
/// `ℓ_j`, `∂ℓ_j/∂f`, `∂ℓ_j/∂η_r`, `V` and `∂V/∂f` at once, and both assemble from them.
pub(crate) fn focei_f_grad<T: PkNum, const N: usize>(
    ctx: &LikCtx,
    f: &[T],
    g: &[[T; N]],
    eta: &[T; N],
    p: &ParT<T>,
) -> Option<(T, [T; N])> {
    if ctx.dense() {
        return Some((
            focei_f_dense(ctx, f, g, eta, p)?.0,
            inner_grad(ctx, f, g, eta, p, None)?,
        ));
    }
    let (fv, grad, _) = focei_parts(ctx, f, g, eta, p)?;
    Some((fv, grad))
}

/// FOCEI's Gauss–Newton Hessian `H̃` (the AGQ Gauss–Newton anchor).
pub(crate) fn focei_htilde<T: PkNum, const N: usize>(
    ctx: &LikCtx,
    f: &[T],
    g: &[[T; N]],
    eta: &[T; N],
    p: &ParT<T>,
) -> Option<Vec<Vec<T>>> {
    Some(focei_parts(ctx, f, g, eta, p)?.2)
}

/// `(F, ∇_η l, H̃)` — `∇l` is left zero on the dense path, whose callers take it from
/// [`inner_grad`].
fn focei_parts<T: PkNum, const N: usize>(
    ctx: &LikCtx,
    f: &[T],
    g: &[[T; N]],
    eta: &[T; N],
    p: &ParT<T>,
) -> Option<(T, [T; N], Vec<Vec<T>>)> {
    if ctx.dense() {
        let (fv, ht) = focei_f_dense(ctx, f, g, eta, p)?;
        return Some((fv, [T::from_f64(0.0); N], ht));
    }
    match ctx.model.residual_error_eta {
        None => focei_rows::<T, N, 1>(ctx, f, g, eta, p, None),
        Some(r) => focei_rows::<T, N, 2>(ctx, f, g, eta, p, Some(r)),
    }
}

/// The diagonal-path row pass behind [`focei_f_grad`], over a `K`-direction dual:
/// direction 0 is the row's prediction (or, on a FREM row, its `η_e`), direction 1 is
/// `η_r` (`K = 2` only under RUV IIV). Mirrors `foce_subject_nll_interaction` /
/// `gaussian_foce_accum` term for term, with `H̃ = Σ aᵀa/V + ½·c̃ᵀc̃ + cens_hess + Ω⁻¹`.
fn focei_rows<T: PkNum, const N: usize, const K: usize>(
    ctx: &LikCtx,
    f: &[T],
    g: &[[T; N]],
    eta: &[T; N],
    p: &ParT<T>,
    rr: Option<usize>,
) -> Option<(T, [T; N], Vec<Vec<T>>)> {
    let c = |x: f64| T::from_f64(x);
    let one = c(1.0);
    let unit = |k: usize| -> [T; N] { std::array::from_fn(|l| c(if l == k { 1.0 } else { 0.0 })) };
    let lift = |x: T| DualT::<K, T> {
        v: x,
        g: [c(0.0); K],
    };
    let pk = p.wrap::<K>();
    let ed: [DualT<K, T>; N] = std::array::from_fn(|k| {
        let mut d = lift(eta[k]);
        if K > 1 && Some(k) == rr {
            d.g[1] = one;
        }
        d
    });
    let ruv = ctx.ruv_scale(&ed);
    let n = f.len();
    let mut data = c(0.0);
    let mut bloq = c(0.0);
    let mut dl = vec![c(0.0); n];
    let mut explicit = [c(0.0); N];
    // H̃ = Ω⁻¹ + Σ_t α_t·u_t·v_tᵀ; the terms come in symmetric pairs, so H̃ stays symmetric.
    let mut terms: Vec<(T, [T; N], [T; N])> = Vec::with_capacity(n + 1);
    for j in 0..n {
        let y = ctx.subject.observations[j];
        if let Some(row) = ctx.frem_row(j) {
            // FREM: prediction θ_i + η_e (unit η-sensitivity), fixed variance; no RUV
            // scale and no `c̃` column (`gaussian_foce_accum`'s `!is_pk_row`).
            let e = row.1;
            let mut el: [DualT<K, T>; N] = std::array::from_fn(|k| lift(eta[k]));
            el[e].g[0] = one;
            let d = row_l(ctx, j, lift(f[j]), lift(f[j]), lift(one), &el, &pk)?;
            data = data + d.v;
            explicit[e] = explicit[e] + d.g[0];
            terms.push((one / ctx.frem_pred_var(row, eta, p).1, unit(e), unit(e)));
            continue;
        }
        let mut fd = lift(f[j]);
        fd.g[0] = one;
        let gj = g[j];
        match ctx.censored(j) {
            None => {
                let vd = ctx.variance_v(j, fd, ruv, &pk);
                if !(vd.v.val().is_finite() && vd.v.val() > 0.0) {
                    return None;
                }
                let r = lift(c(y)) - fd;
                let lj = r * r / vd + vd.ln();
                data = data + lj.v;
                dl[j] = lj.g[0];
                if let Some(r) = rr {
                    explicit[r] = explicit[r] + lj.g[1];
                }
                let iv = one / vd.v;
                let cs = vd.g[0] / vd.v;
                // g·gᵀ/V + ½·c̃c̃ᵀ with c̃ = (∂V/∂f / V)·g, whose η_r entry is 2 under RUV IIV.
                match rr {
                    None => terms.push((iv + c(0.5) * cs * cs, gj, gj)),
                    Some(r) => {
                        let ck: [T; N] =
                            std::array::from_fn(|k| if k == r { c(2.0) } else { cs * gj[k] });
                        terms.push((iv, gj, gj));
                        terms.push((c(0.5), ck, ck));
                    }
                }
            }
            Some(cens) => {
                let (v, d, d2) = ctx.variance(j, f[j], ruv.v, p);
                if !(v.val().is_finite() && v.val() > 0.0) {
                    return None;
                }
                let (l2, g2, cz, cm) = m3_terms(y, f[j], v, d, d2, cens);
                bloq = bloq + l2;
                // ∂ℓ/∂f and ∂ℓ/∂η_r of ℓ = −2·log Φ(±(y − f)/√V), with ∂V/∂η_r = 2V.
                let mut vk = lift(v);
                vk.g[0] = d;
                if K > 1 {
                    vk.g[1] = c(2.0) * v;
                }
                let sgn = if cens < 0 { -1.0 } else { 1.0 };
                let lj = lift(c(-2.0)) * log_phi(lift(c(sgn)) * (lift(c(y)) - fd) / vk.sqrt());
                dl[j] = lj.g[0];
                if let Some(r) = rr {
                    explicit[r] = explicit[r] + lj.g[1];
                }
                terms.push((g2, gj, gj));
                if let Some(r) = rr {
                    let mut go = gj;
                    go[r] = c(0.0);
                    terms.push((cz, unit(r), unit(r)));
                    terms.push((cm, unit(r), go));
                    terms.push((cm, go, unit(r)));
                }
            }
        }
    }
    let oi = &p.omega.inv;
    let mut ht: Vec<Vec<T>> = vec![vec![c(0.0); N]; N];
    for a in 0..N {
        for b in 0..=a {
            let e = T::wsum3(
                0.0,
                std::iter::once((oi[a][b], one, one))
                    .chain(terms.iter().map(|(al, u, v)| (*al, u[a], v[b]))),
            );
            ht[a][b] = e;
            ht[b][a] = e;
        }
    }
    let ld = chol(&ht).map(|l| chol_logdet(&l))?;
    let fv = c(0.5) * (data + ctx.prior(eta, p) + ld + bloq);
    Some((fv, assemble_grad(oi, eta, g, &dl, &explicit), ht))
}

/// `∇_k l = Σ_l Ω⁻¹_kl η_l + ½·Σ_j (∂ℓ_j/∂f)·G_jk + ½·explicit_k`.
fn assemble_grad<T: PkNum, const N: usize>(
    oi: &[Vec<T>],
    eta: &[T; N],
    g: &[[T; N]],
    dl: &[T],
    explicit: &[T; N],
) -> [T; N] {
    let one = T::from_f64(1.0);
    let half = T::from_f64(0.5);
    std::array::from_fn(|k| {
        let prior = (0..N).map(|l| (oi[k][l], eta[l], one));
        let rows = (0..dl.len()).map(|j| (half, dl[j], g[j][k]));
        T::wsum3(0.0, prior.chain(rows)) + half * explicit[k]
    })
}

/// The FOCE (no interaction) subject objective: the Sheiner–Beal linearised marginal
/// `½[(y−f₀)ᵀR̃⁻¹(y−f₀) + log|R̃| + M3]` with `f₀ = f − Gη̂` and `R̃ = GΩGᵀ + R⁰` over the
/// quantified rows (`foce_subject_nll_standard_cov`). `R⁰` is at the population
/// prediction `fpop = f(η = 0)` (`population_variance_preds`; `None` for an
/// `f`-independent variance, where `f₀` is used). A censored row contributes
/// `−2·log Φ` at mean `f₀` and variance `R̃_jj`; a FREM row has mean `θ_i + η_e`, unit
/// η-sensitivity and variance `σ_c²`.
///
/// A diagonal `R⁰` takes Woodbury's form, `rᵀR̃⁻¹r = rᵀR⁻¹r − bᵀA⁻¹b` and
/// `log|R̃| = log|R| + log|Ω| + log|A|` with `A = Ω⁻¹ + GᵀR⁻¹G`, `b = GᵀR⁻¹r`: `N × N`
/// work per subject instead of the `n × n` factorisation. A covariance coupling rows
/// factors `R̃` directly.
pub(crate) fn foce_f<T: PkNum, const N: usize>(
    ctx: &LikCtx,
    f: &[T],
    g: &[[T; N]],
    eta: &[T; N],
    fpop: Option<&[T]>,
    p: &ParT<T>,
) -> Option<T> {
    let c = |x: f64| T::from_f64(x);
    let one = c(1.0);
    let n = f.len();
    let unit = |k: usize| -> [T; N] { std::array::from_fn(|l| c(if l == k { 1.0 } else { 0.0 })) };
    // Per row: prediction, η-sensitivity, linearised mean f₀ and LᵀG (for GΩGᵀ = (LᵀG)ᵀ(LᵀG)).
    let lch = &p.omega.chol;
    let mut f0: Vec<T> = Vec::with_capacity(n);
    let mut lg: Vec<[T; N]> = Vec::with_capacity(n);
    for j in 0..n {
        let (fj, gj) = match ctx.frem_row(j) {
            Some(row) => (ctx.frem_pred_var(row, eta, p).0, unit(row.1)),
            None => (f[j], g[j]),
        };
        f0.push(fj - T::wsum3(0.0, (0..N).map(|k| (one, gj[k], eta[k]))));
        lg.push(std::array::from_fn(|cc| {
            T::wsum3(0.0, (cc..N).map(|r| (one, lch[r][cc], gj[r])))
        }));
    }
    let r_eval: Vec<T> = match fpop {
        Some(fp) => fp.to_vec(),
        None => f0.clone(),
    };
    // R⁰'s diagonal (FREM rows overridden), and the censored rows' `−2·log Φ`.
    let r0_diag = |j: usize, rd: T| -> T {
        match ctx.frem_row(j) {
            Some(row) => ctx.frem_pred_var(row, eta, p).1,
            None => rd,
        }
    };
    let dense = ctx.cov.is_dense();
    let rm: Option<Vec<Vec<T>>> = dense.then(|| ctx.r_matrix(&r_eval, p));
    let mut bloq = c(0.0);
    let mut quant: Vec<usize> = Vec::with_capacity(n);
    let mut rdiag: Vec<T> = Vec::with_capacity(n);
    for j in 0..n {
        let rd = r0_diag(
            j,
            match &rm {
                Some(m) => m[j][j],
                None => ctx.variance_v(j, r_eval[j], one, p),
            },
        );
        if !(rd.val().is_finite() && rd.val() > 0.0) {
            return None;
        }
        rdiag.push(rd);
        match ctx.censored(j) {
            Some(cens) if ctx.frem_row(j).is_none() => {
                let rt = T::wsum3(0.0, (0..N).map(|k| (one, lg[j][k], lg[j][k]))) + rd;
                let sgn = if cens < 0 { -1.0 } else { 1.0 };
                bloq = bloq
                    + c(-2.0)
                        * log_phi(c(sgn) * (c(ctx.subject.observations[j]) - f0[j]) / rt.sqrt());
            }
            _ => quant.push(j),
        }
    }
    let res: Vec<T> = quant
        .iter()
        .map(|&j| c(ctx.subject.observations[j]) - f0[j])
        .collect();
    let nq = quant.len();
    // Woodbury on a diagonal `R⁰` is a subtractive identity, accurate to `≈ eps·(1 + tr S)`
    // with `S = LᵀGᵀD⁻¹GL` (`Ω = LLᵀ`, `GL` = `lg`). A row whose `R⁰` is on the variance
    // floor while its `G` is O(1) (FOCE's `R⁰` is taken at η = 0, `G` at η̂) drives
    // `1 + tr S` to ~1e11 and the correction loses every digit (#1498): above
    // `WOODBURY_MAX_INFO_TRACE`, or when it is not finite, `R̃` is factored at observation size.
    let woodbury = rm.is_none() && {
        let tr: f64 = quant
            .iter()
            .map(|&j| (0..N).map(|k| lg[j][k].val() * lg[j][k].val()).sum::<f64>() / rdiag[j].val())
            .sum();
        1.0 + tr <= WOODBURY_MAX_INFO_TRACE || woodbury_forced()
    };
    if !woodbury {
        // R̃ = R⁰_q + (LᵀG)ᵀ(LᵀG), FREM rows decoupled (`apply_frem_r_matrix_overrides`).
        let rt: Vec<Vec<T>> = (0..nq)
            .map(|a| {
                (0..nq)
                    .map(|b| {
                        let (ja, jb) = (quant[a], quant[b]);
                        let r0 = if ja == jb {
                            rdiag[ja]
                        } else if ctx.frem_row(ja).is_some() || ctx.frem_row(jb).is_some() {
                            c(0.0)
                        } else {
                            rm.as_ref().map_or(c(0.0), |m| m[ja][jb])
                        };
                        T::wsum3(0.0, (0..N).map(|k| (one, lg[ja][k], lg[jb][k]))) + r0
                    })
                    .collect()
            })
            .collect();
        let l = chol(&rt)?;
        let z = chol_solve(&l, &res);
        let quad = T::wsum3(0.0, (0..nq).map(|i| (one, res[i], z[i])));
        return Some(c(0.5) * (quad + chol_logdet(&l) + bloq));
    }
    // Woodbury / determinant lemma on the diagonal R⁰, Ω-normalised: with `B = D⁻¹GL`,
    // `R̃⁻¹ = D⁻¹ − B(I + S)⁻¹Bᵀ` and `|R̃| = |D|·|I + S|`. Factoring `I + S` rather than
    // `Ω⁻¹ + GᵀD⁻¹G` keeps `cond(Ω)` (a near-rail `block_omega`) out of the subtraction,
    // which the trace gate above does not see.
    let iv: Vec<T> = quant.iter().map(|&j| one / rdiag[j]).collect();
    let mut a: Vec<Vec<T>> = vec![vec![c(0.0); N]; N];
    for k in 0..N {
        for l in 0..=k {
            let e = T::wsum3(
                if k == l { 1.0 } else { 0.0 },
                (0..nq).map(|i| (iv[i], lg[quant[i]][k], lg[quant[i]][l])),
            );
            a[k][l] = e;
            a[l][k] = e;
        }
    }
    let b: Vec<T> = (0..N)
        .map(|k| T::wsum3(0.0, (0..nq).map(|i| (iv[i], lg[quant[i]][k], res[i]))))
        .collect();
    let la = chol(&a)?;
    let ab = chol_solve(&la, &b);
    let quad = T::wsum3(0.0, (0..nq).map(|i| (iv[i], res[i], res[i])))
        - T::wsum3(0.0, (0..N).map(|k| (one, b[k], ab[k])));
    let mut log_r = c(0.0);
    for &j in &quant {
        log_r = log_r + rdiag[j].ln();
    }
    Some(c(0.5) * (quad + log_r + chol_logdet(&la) + bloq))
}

/// The largest `1 + tr(S)` the FOCE kernel's Woodbury branch is trusted under, `S` the
/// Ω-normalised information `LᵀGᵀD⁻¹GL` (#1498; base's `foce_rtilde_inverse`). A Cholesky
/// inverse at condition `κ` keeps `≈ eps·κ`, so `1e6` bounds the error near `2e-10`. Measured
/// on base: warfarin's worst FOCE trace 2.7e4 (error 1.6e-12); `ss_oral_q24` subject 3's
/// smallest 1.2e11 (error 0.16–0.27). Anywhere in `[1e5, 1e9]` separates them.
const WOODBURY_MAX_INFO_TRACE: f64 = 1e6;

#[cfg(test)]
thread_local! {
    /// Test switch: take the Woodbury branch whatever `1 + tr(S)` says, to show a fixture
    /// can expose the loss the gate exists for.
    pub(crate) static FORCE_WOODBURY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[inline]
fn woodbury_forced() -> bool {
    #[cfg(test)]
    return FORCE_WOODBURY.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

/// Dense-`R` FOCEI `F`: `½(rᵀR⁻¹r + log|R| + ηᵀΩ⁻¹η + log|Ω| + log|H̃|)`.
fn focei_f_dense<T: PkNum, const N: usize>(
    ctx: &LikCtx,
    f: &[T],
    g: &[[T; N]],
    eta: &[T; N],
    p: &ParT<T>,
) -> Option<(T, Vec<Vec<T>>)> {
    let c = |x: f64| T::from_f64(x);
    let n = f.len();
    let fd: Vec<DualT<N, T>> = (0..n).map(|j| DualT { v: f[j], g: g[j] }).collect();
    // η as a dual too, so `R·exp(2η_r)` (IIV on the residual) carries `∂R/∂η_r` in its
    // tangents along with `Σ_m G_mk ∂R/∂f_m`.
    let ed: [DualT<N, T>; N] = std::array::from_fn(|k| {
        let mut gk = [c(0.0); N];
        gk[k] = c(1.0);
        DualT { v: eta[k], g: gk }
    });
    let ruv = ctx.ruv_scale(&ed);
    let rd: Vec<Vec<DualT<N, T>>> = ctx
        .r_matrix(&fd, &p.wrap::<N>())
        .into_iter()
        .map(|row| row.into_iter().map(|x| x * ruv).collect())
        .collect();
    let rv: Vec<Vec<T>> = rd
        .iter()
        .map(|row| row.iter().map(|x| x.v).collect())
        .collect();
    let l = chol(&rv)?;
    let res: Vec<T> = (0..n)
        .map(|j| c(ctx.subject.observations[j]) - f[j])
        .collect();
    let z = chol_solve(&l, &res);
    let mut data = chol_logdet(&l);
    for j in 0..n {
        data = data + res[j] * z[j];
    }
    // R⁻¹G (n × N) and R⁻¹·∂_kR (n × n per k)
    let rg: Vec<Vec<T>> = (0..N)
        .map(|k| chol_solve(&l, &(0..n).map(|j| g[j][k]).collect::<Vec<T>>()))
        .collect();
    let rd_k: Vec<Vec<Vec<T>>> = (0..N)
        .map(|k| {
            (0..n)
                .map(|col| chol_solve(&l, &(0..n).map(|i| rd[i][col].g[k]).collect::<Vec<T>>()))
                .collect()
        })
        .collect();
    let mut ht: Vec<Vec<T>> = p.omega.inv.clone();
    for a in 0..N {
        for b in 0..N {
            let mut hrh = c(0.0);
            for j in 0..n {
                hrh = hrh + g[j][a] * rg[b][j];
            }
            // B_ab = tr(Y_a Y_b), Y_k = R⁻¹∂_kR, stored by columns: rd_k[k][col][row]
            let mut bab = c(0.0);
            for i in 0..n {
                for jj in 0..n {
                    bab = bab + rd_k[a][jj][i] * rd_k[b][i][jj];
                }
            }
            ht[a][b] = ht[a][b] + hrh + c(0.5) * bab;
        }
    }
    let ld = chol(&ht).map(|l| chol_logdet(&l))?;
    Some((c(0.5) * (data + ctx.prior(eta, p) + ld), ht))
}

/// One row's inner data term: `r²/V + ln V` (Gaussian), `−2·log Φ(z)` (M3-censored,
/// `z = ±(y − f)/√V`, `m3_logcdf`), or the FREM pseudo-observation term (prediction
/// `θ_i + η_e`, read from `eta`/`p`), with `V` evaluated at `vf` (`f` itself, or FOCE's
/// frozen population prediction). The single row definition behind both the value of `l`
/// and its η-gradient.
#[allow(clippy::too_many_arguments)]
fn row_l<T: PkNum, const N: usize>(
    ctx: &LikCtx,
    j: usize,
    f: T,
    vf: T,
    ruv: T,
    eta: &[T; N],
    p: &ParT<T>,
) -> Option<T> {
    let c = |x: f64| T::from_f64(x);
    let y = ctx.subject.observations[j];
    if let Some(row) = ctx.frem_row(j) {
        let (fp, v) = ctx.frem_pred_var(row, eta, p);
        let r = c(y) - fp;
        return Some(r * r / v + v.ln());
    }
    let v = ctx.variance_v(j, vf, ruv, p);
    if !(v.val().is_finite() && v.val() > 0.0) {
        return None;
    }
    Some(match ctx.censored(j) {
        None => {
            let r = c(y) - f;
            r * r / v + v.ln()
        }
        Some(cens) => {
            let sgn = if cens < 0 { -1.0 } else { 1.0 };
            c(-2.0) * log_phi(c(sgn) * (c(y) - f) / v.sqrt())
        }
    })
}

/// The inner (EBE) objective `l(η)` at the predictions `f`. `frozen` is FOCE's
/// convention (`EbeVariance::Population`): the residual variance held at the population
/// prediction, with no RUV-IIV scale; `None` is the conditional objective.
pub(crate) fn inner_l<T: PkNum, const N: usize>(
    ctx: &LikCtx,
    f: &[T],
    eta: &[T; N],
    p: &ParT<T>,
    frozen: Option<&[T]>,
) -> Option<T> {
    let c = |x: f64| T::from_f64(x);
    let ruv = if frozen.is_some() {
        c(1.0)
    } else {
        ctx.ruv_scale(eta)
    };
    let vf = frozen.unwrap_or(f);
    if ctx.dense() {
        let n = f.len();
        let r: Vec<Vec<T>> = ctx
            .r_matrix(vf, p)
            .into_iter()
            .map(|row| row.into_iter().map(|x| x * ruv).collect())
            .collect();
        let l = chol(&r)?;
        let res: Vec<T> = (0..n)
            .map(|j| c(ctx.subject.observations[j]) - f[j])
            .collect();
        let z = chol_solve(&l, &res);
        let mut data = chol_logdet(&l);
        for j in 0..n {
            data = data + res[j] * z[j];
        }
        return Some(c(0.5) * (ctx.prior(eta, p) + data));
    }
    let mut data = c(0.0);
    for j in 0..f.len() {
        data = data + row_l(ctx, j, f[j], vf[j], ruv, eta, p)?;
    }
    Some(c(0.5) * (ctx.prior(eta, p) + data))
}

/// `∇_η l` at `(f, G, η)`, under the conditional or FOCE's `frozen` convention (see
/// [`inner_l`]). Row-separable on the diagonal path: each row contributes
/// `½·(∂ℓ_j/∂f)·G_j` plus its explicit η-dependence (the RUV-IIV scale, or a FREM row's
/// `θ_i + η_e`), from a small dual over the row term [`row_l`] (the conditional case is
/// [`focei_f_grad`]'s row pass); the prior adds `Ω⁻¹η`. The dense path couples rows, so
/// there `l` itself is evaluated over an η-dual whose predictions carry `G`.
pub(crate) fn inner_grad<T: PkNum, const N: usize>(
    ctx: &LikCtx,
    f: &[T],
    g: &[[T; N]],
    eta: &[T; N],
    p: &ParT<T>,
    frozen: Option<&[T]>,
) -> Option<[T; N]> {
    let c = |x: f64| T::from_f64(x);
    if ctx.dense() {
        let fd: Vec<DualT<N, T>> = (0..f.len()).map(|j| DualT { v: f[j], g: g[j] }).collect();
        let ed: [DualT<N, T>; N] = std::array::from_fn(|k| {
            let mut gk = [c(0.0); N];
            gk[k] = c(1.0);
            DualT { v: eta[k], g: gk }
        });
        let fz: Option<Vec<DualT<N, T>>> = frozen.map(|fz| {
            fz.iter()
                .map(|&x| DualT {
                    v: x,
                    g: [c(0.0); N],
                })
                .collect()
        });
        return Some(inner_l(ctx, &fd, &ed, &p.wrap::<N>(), fz.as_deref())?.g);
    }
    let Some(fz) = frozen else {
        return Some(focei_f_grad(ctx, f, g, eta, p)?.1);
    };
    // FOCE's frozen variance: a row reads η only through f (FREM rows through η_e).
    let p1 = p.wrap::<1>();
    let lift = |x: T| DualT { v: x, g: [c(0.0)] };
    let mut dl = vec![c(0.0); f.len()];
    let mut explicit = [c(0.0); N];
    for j in 0..f.len() {
        let mut el: [DualT<1, T>; N] = std::array::from_fn(|k| lift(eta[k]));
        if let Some((_, e, _)) = ctx.frem_row(j) {
            el[e].g[0] = c(1.0);
            let d = row_l(ctx, j, lift(f[j]), lift(f[j]), lift(c(1.0)), &el, &p1)?;
            explicit[e] = explicit[e] + d.g[0];
            continue;
        }
        dl[j] = row_l(
            ctx,
            j,
            DualT {
                v: f[j],
                g: [c(1.0)],
            },
            lift(fz[j]),
            lift(c(1.0)),
            &el,
            &p1,
        )?
        .g[0];
    }
    let grad = assemble_grad(&p.omega.inv, eta, g, &dl, &explicit);
    Some(grad)
}

/// Cholesky factor `L` (`A = LLᵀ`), generic; `None` unless positive definite at the value.
pub(crate) fn chol<T: PkNum>(a: &[Vec<T>]) -> Option<Vec<Vec<T>>> {
    let n = a.len();
    let mut l: Vec<Vec<T>> = vec![vec![T::from_f64(0.0); n]; n];
    for j in 0..n {
        let mut d = a[j][j];
        for k in 0..j {
            d = d - l[j][k] * l[j][k];
        }
        if !(d.val() > 0.0) {
            return None;
        }
        let ljj = d.sqrt();
        l[j][j] = ljj;
        for i in (j + 1)..n {
            let mut s = a[i][j];
            for k in 0..j {
                s = s - l[i][k] * l[j][k];
            }
            l[i][j] = s / ljj;
        }
    }
    Some(l)
}

pub(crate) fn chol_logdet<T: PkNum>(l: &[Vec<T>]) -> T {
    let mut ld = T::from_f64(0.0);
    for j in 0..l.len() {
        ld = ld + T::from_f64(2.0) * l[j][j].ln();
    }
    ld
}

/// `A⁻¹b` from `A = LLᵀ`.
fn chol_solve<T: PkNum>(l: &[Vec<T>], b: &[T]) -> Vec<T> {
    let n = l.len();
    let mut y = b.to_vec();
    for i in 0..n {
        let mut s = y[i];
        for k in 0..i {
            s = s - l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s = s - l[k][i] * y[k];
        }
        y[i] = s / l[i][i];
    }
    y
}
