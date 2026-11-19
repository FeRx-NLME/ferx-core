//! The FOCE/FOCEI subject objective `F` and the inner objective `l`, written once over a
//! generic numeric type, as functions of the predictions `f`, their η-sensitivities
//! `G = ∂f/∂η`, `η`, and the population parameters the likelihood reads ([`ParT`]: Ω by
//! its Cholesky factor, σ, `block_sigma` ρ, the serial ρ, θ for FREM rows).
//!
//! The adjoint gradient (`sens::ode_provider::adjoint`) records them into a kernel plan
//! (`ode_adjoint_kernel`) whose leaves are every input, so one multi-seed reverse gives
//! `Ψ = F − wᵀ∇_η l`'s partials with respect to every input, with no hand-derived seeds or
//! finite differences.
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

use crate::sens::jet::DualT;
#[cfg(test)]
use crate::sens::num::NonRecording;
use crate::sens::num::{wsum3, BranchCmp, PkNum};
use crate::stats::residual_error::{LoadingForm, ResidualCov};
use crate::types::{CompiledModel, ModelParameters, Subject};
use std::borrow::Cow;

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

/// Widths [`OmegaT::from_chol_cached`] keeps a factorisation for, per thread.
const OMEGA_CACHE_WIDTHS: usize = 8;

thread_local! {
    /// The Ω [`OmegaT::from_chol_cached`] last factored at each width: Ω is one matrix for a
    /// whole outer evaluation, so every subject on a thread after the first reuses its inverse
    /// — also when the subjects' stacked widths differ (IOV with unequal occasion counts), as
    /// a single entry would miss on every alternation.
    static OMEGA_CACHE: std::cell::RefCell<Vec<OmegaT<f64>>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Factorisations the cache had to compute (test seam).
    #[cfg(test)]
    static OMEGA_MISSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl OmegaT<f64> {
    /// [`Self::from_chol`], reusing the `Ω⁻¹` and `log|Ω|` last formed at this width when `chol`
    /// is bit-for-bit the (square) factor they were formed from (`from_chol` is a function of
    /// `chol` alone, so the reuse returns exactly what it would compute).
    pub(crate) fn from_chol_cached(chol: Vec<Vec<f64>>) -> Option<Self> {
        OMEGA_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            let at = c.iter().position(|o| o.chol.len() == chol.len());
            if let Some(o) = at.map(|i| &c[i]).filter(|o| {
                o.chol
                    .iter()
                    .zip(&chol)
                    .all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()))
            }) {
                return Some(OmegaT {
                    chol,
                    inv: o.inv.clone(),
                    log_det: o.log_det,
                });
            }
            #[cfg(test)]
            OMEGA_MISSES.with(|m| m.set(m.get() + 1));
            let o = Self::from_chol(chol)?;
            match at {
                Some(i) => c[i] = o.clone(),
                None => {
                    if c.len() == OMEGA_CACHE_WIDTHS {
                        c.swap_remove(0);
                    }
                    c.push(o.clone());
                }
            }
            Some(o)
        })
    }
}

impl<T: PkNum> OmegaT<T> {
    pub(crate) fn from_chol(chol: Vec<Vec<T>>) -> Option<Self> {
        let n = chol.len();
        let c = |x: f64| T::from_f64(x);
        // L⁻¹, lower triangular, by forward substitution.
        let mut li = vec![vec![c(0.0); n]; n];
        for i in 0..n {
            if !chol[i][i].br_cmp_c(BranchCmp::Gt, 0.0) {
                return None;
            }
            li[i][i] = c(1.0) / chol[i][i];
            for j in 0..i {
                let s = wsum3(0.0, (j..i).map(|k| (c(1.0), chol[i][k], li[k][j])));
                li[i][j] = -s / chol[i][i];
            }
        }
        let mut inv = vec![vec![c(0.0); n]; n];
        for a in 0..n {
            for b in 0..=a {
                let e = wsum3(0.0, (a..n).map(|k| (c(1.0), li[k][a], li[k][b])));
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
        Self::of_with_omega(
            model,
            subject,
            params,
            OmegaT {
                chol: (0..n)
                    .map(|r| (0..n).map(|c| l[(r, c)]).collect())
                    .collect(),
                inv: (0..n)
                    .map(|r| (0..n).map(|c| params.omega.inv[(r, c)]).collect())
                    .collect(),
                log_det: params.omega.log_det,
            },
        )
    }

    /// [`Self::of`] with Ω given (the stacked prior's factor, [`OmegaT::from_chol`]): the
    /// params' own Ω is not copied just to be replaced.
    pub(crate) fn of_with_omega(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
        omega: OmegaT<f64>,
    ) -> Self {
        ParT {
            omega,
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

impl<T> ParT<T> {
    /// `Ω⁻¹` for `n` random effects. A [`ParT`] from [`Self::map_rows`] / [`Self::wrap_rows`]
    /// does not carry it, and reading one there would index an empty matrix (or, where the
    /// width came from it, silently loop over nothing): fail with the cause instead.
    #[inline]
    pub(crate) fn omega_inv(&self, n: usize) -> &Vec<Vec<T>> {
        assert!(
            self.omega.inv.len() >= n,
            "Ω⁻¹ has {} rows, {n} needed: a ParT from map_rows/wrap_rows carries no Ω",
            self.omega.inv.len()
        );
        &self.omega.inv
    }

    /// `L` (`Ω = LLᵀ`) for `n` random effects; see [`Self::omega_inv`].
    #[inline]
    pub(crate) fn omega_chol(&self, n: usize) -> &Vec<Vec<T>> {
        assert!(
            self.omega.chol.len() >= n,
            "Ω's factor has {} rows, {n} needed: a ParT from map_rows/wrap_rows carries no Ω",
            self.omega.chol.len()
        );
        &self.omega.chol
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

impl<T: Copy> ParT<T> {
    /// [`Self::map`] without Ω's factor and inverse (only `log|Ω|` is carried): for the row
    /// passes, which read σ, ρ, θ and the RUV multipliers but never Ω (the prior is scored
    /// from the unlifted `ParT`). Saves the `2n + 2` allocations of lifting Ω at every call.
    pub(crate) fn map_rows<U: Copy>(&self, f: impl Fn(T) -> U) -> ParT<U> {
        ParT {
            omega: OmegaT {
                chol: Vec::new(),
                inv: Vec::new(),
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

    /// [`Self::wrap`] for the row passes ([`Self::map_rows`]): Ω is not lifted.
    pub(crate) fn wrap_rows<const K: usize>(&self) -> ParT<DualT<K, T>> {
        self.map_rows(|x| DualT {
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
    /// Per row, its dispatch key (`ErrorSpec::obs_keys`: borrowed from the subject unless an
    /// endpoint selector computes it). A row's σ-loading shape is a function of its key alone
    /// (`ErrorSpec::sigma_loading_shape`), read on access, so building the context allocates
    /// nothing per row — it is built for every subject at every gradient.
    keys: std::borrow::Cow<'a, [usize]>,
    /// An SDE subject: the kernel's `f`/`g` (and `q`) carry `2n` rows, the `n` predictions then
    /// the EKF's per-observation process-noise variances `p_obs` with their η-jets, and each
    /// row's variance is the residual one plus its `p_obs` (`ekf_p_obs`).
    pub(crate) sde: bool,
}

/// The most σ loadings one row has: the residual-error owner's bound.
const MAX_LOADINGS: usize = crate::stats::residual_error::MAX_SIGMA_LOADINGS;

impl<'a> LikCtx<'a> {
    pub(crate) fn new(
        model: &'a CompiledModel,
        subject: &'a Subject,
        params: &'a ModelParameters,
    ) -> Self {
        let keys = model.error_spec.obs_keys(subject);
        let cov = model.residual_cov_with(&params.residual_correlations, &params.theta);
        LikCtx {
            model,
            subject,
            params,
            cov,
            keys,
            sde: model.is_sde(),
        }
    }

    /// The subject's observation rows (the kernel's `f` holds `2n` under an SDE).
    #[inline]
    fn n(&self) -> usize {
        self.subject.observations.len()
    }

    /// Row `j`'s EKF process-noise variance as the kernel sees it: `f[n + j]` under an SDE,
    /// else zero.
    #[inline]
    fn extra<T: PkNum>(&self, f: &[T], j: usize) -> T {
        if self.sde {
            f[self.n() + j]
        } else {
            T::from_f64(0.0)
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
    /// `1`, a proportional slot's `f` — raised to `sign(f)·|f|^p` under a
    /// `power(...)` exponent — and a `LogCombined` additive slot's `exp(−f)` (#2729),
    /// each times the custom magnitude multiplier. The same
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
        let shape = self
            .model
            .error_spec
            .loading_shape(self.keys[j], self.params.sigma.values.len());
        let len = shape.len();
        for (o, &(idx, form)) in out.iter_mut().zip(shape.iter()) {
            let c = if form == LoadingForm::Prop {
                // `slot_exponent`: the exponent half of the row, when present.
                match mult
                    .filter(|m| m.len() == 2 * n_sigma)
                    .map(|m| m[n_sigma + idx])
                {
                    Some(pw) if pw.br_cmp_c(BranchCmp::Ne, 1.0) => {
                        if f.br_cmp_c(BranchCmp::Eq, 0.0) {
                            T::from_f64(0.0)
                        } else {
                            let sign = if f.br_cmp_c(BranchCmp::Lt, 0.0) {
                                -1.0
                            } else {
                                1.0
                            };
                            T::from_f64(sign) * f.abs().pow(pw)
                        }
                    }
                    _ => form.coeff_g(f),
                }
            } else {
                form.coeff_g(f)
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
        let v = self.loadings_variance(j, d2_seed(f), &p.map_rows(d2_const));
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
        let oi = p.omega_inv(n);
        wsum3(
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
            self.n(),
            self.subject,
            &p.sigma,
            self.cov,
            |k| p.rho[k],
            p.serial_rho,
        )
    }
}

/// `0 < x < ∞` — a variance a row can be scored with — as a branch: `false` on `NaN`.
#[inline]
fn positive_finite<T: PkNum>(x: T) -> bool {
    x.br_cmp_c(BranchCmp::Gt, 0.0) && x.br_cmp_c(BranchCmp::Lt, f64::INFINITY)
}

/// `log Φ(z)` with production's value (`log_normal_cdf`) and its exact derivatives
/// (`h = φ/Φ`, `h′ = −h(z + h)`, `h″`).
#[inline]
fn log_phi<T: PkNum>(z: T) -> T {
    // Past `h`'s underflow the row is certain and every derivative is `h` times a polynomial:
    // limit `0` (#2772). Taken here, not left to the jet evaluator (which also returns `0`):
    // at `z = +∞` the chain `0 · z′` is `0 · ∞ = NaN` for a `z` whose own derivatives overflow
    // (`z = r/w` with `r ≈ 1e308`), so the value alone is returned. A recordable test.
    if inv_mills(z).br_cmp_c(BranchCmp::Eq, 0.0) {
        return z
            .jet_unary(crate::sens::num::JetFn::LogNormCdf)
            .value_only();
    }
    z.jet_unary(crate::sens::num::JetFn::LogNormCdf)
}

/// Inverse Mills ratio `h(z) = φ/Φ` with `h′ = −h(z + h)`, lifted to fourth order so the
/// exact anchor's nested covariance dual differentiates it rather than poisoning (`lift2` left
/// `h‴` unknown, #2093 review).
#[inline]
fn inv_mills<T: PkNum>(z: T) -> T {
    z.jet_unary(crate::sens::num::JetFn::InvMills)
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
    // A recordable test (`val` would fail a recording).
    if h.br_cmp_c(BranchCmp::Eq, 0.0) {
        // A certain row past `h`'s underflow: `g2`, `cz` and `cm` are `h` times polynomials
        // in `z` and `m`, limit `0`, and `z²` overflows from `|z| ≈ 1.3e154` (#2772).
        // `log_phi` is the value only here (its own `h == 0` arm).
        return (c(-2.0) * log_phi(z), c(0.0), c(0.0), c(0.0));
    }
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
pub(crate) fn focei_f_grad<T: PkNum>(
    ctx: &LikCtx,
    f: &[T],
    g: &[Vec<T>],
    eta: &[T],
    p: &ParT<T>,
) -> Option<(T, Vec<T>)> {
    if ctx.dense() {
        return Some((
            focei_f_dense(ctx, f, g, eta, p)?.0,
            inner_grad(ctx, f, g, eta, p, None)?,
        ));
    }
    let (fv, grad, _) = focei_parts(ctx, f, g, eta, p)?;
    Some((fv, grad))
}

/// A subject's non-Gaussian data term at `η` for the FOCE/FOCEI objective — its value `q`
/// (the raw `1×` NLL of every TTE / binary / CTMM endpoint), η-gradient and η-Hessian —
/// as production folds it in (`foce_subject_nll_interaction_with_tte`): `2q` beside the
/// Gaussian `data_ll`, `∇²q` into `H̃`, so `∇_η l` gains `∇q`.
pub(crate) struct NonGauss<T> {
    pub(crate) q: T,
    pub(crate) g: Vec<T>,
    pub(crate) h: Vec<Vec<T>>,
}

/// `(F, ∇_η l)` of the objective of a subject carrying non-Gaussian records: production's
/// one form for both interaction settings — the conditional Gaussian rows plus `ng`, with
/// `H̃`'s `½·c̃ᵀc̃` and censored curvature only under `interaction` (FOCE omits them).
/// `None` on a dense residual (production's non-Gaussian path is the diagonal one) or a
/// failed factorisation.
#[cfg(feature = "survival")]
pub(crate) fn focei_f_grad_ng<T: PkNum>(
    ctx: &LikCtx,
    f: &[T],
    g: &[Vec<T>],
    eta: &[T],
    p: &ParT<T>,
    ng: &NonGauss<T>,
    interaction: bool,
) -> Option<(T, Vec<T>)> {
    if ctx.dense() {
        return None;
    }
    let (fv, grad, _) = focei_rows_k(ctx, f, g, eta, p, Some(ng), interaction)?;
    Some((fv, grad))
}

/// FOCEI's Gauss–Newton Hessian `H̃` (the AGQ Gauss–Newton anchor).
pub(crate) fn focei_htilde<T: PkNum>(
    ctx: &LikCtx,
    f: &[T],
    g: &[Vec<T>],
    eta: &[T],
    p: &ParT<T>,
) -> Option<Vec<Vec<T>>> {
    Some(focei_parts(ctx, f, g, eta, p)?.2)
}

/// `(F, ∇_η l, H̃)` — `∇l` is left zero on the dense path, whose callers take it from
/// [`inner_grad`].
fn focei_parts<T: PkNum>(
    ctx: &LikCtx,
    f: &[T],
    g: &[Vec<T>],
    eta: &[T],
    p: &ParT<T>,
) -> Option<(T, Vec<T>, Vec<Vec<T>>)> {
    if ctx.dense() {
        let (fv, ht) = focei_f_dense(ctx, f, g, eta, p)?;
        return Some((fv, vec![T::from_f64(0.0); eta.len()], ht));
    }
    focei_rows_k(ctx, f, g, eta, p, None, true)
}

/// An uncensored Gaussian row's `ℓ = r²/V + log V` at prediction `fd` with variance `vd`
/// (`r = y − f`), over any dual the caller carries: [`focei_rows`]' and the covariance
/// Hessian's structured rows' one formula (#2980).
#[inline]
fn gauss_l<D: PkNum>(y: f64, fd: D, vd: D) -> D {
    let r = D::from_f64(y) - fd;
    r * r / vd + vd.ln()
}

/// A FOCEI row's `H̃` weight `1/V + ½·(∂V/∂f / V)²` (`c̃ = (∂V/∂f / V)·g`, no RUV IIV).
#[inline]
fn focei_alpha<T: PkNum>(v: T, vf: T) -> T {
    let cs = vf / v;
    T::from_f64(1.0) / v + T::from_f64(0.5) * cs * cs
}

/// Row `j` of the FOCEI diagonal path as the structured covariance Hessian reads it (#2980):
/// `(ℓ, ∂ℓ/∂f, α)` at prediction `f` — an uncensored Gaussian row, no RUV IIV, no SDE, no
/// FREM ([`structured_rows_supported`]). `T` carries the derivatives the caller needs (a dual
/// over the row's local inputs); the arithmetic is [`focei_rows`]'.
pub(crate) fn focei_gauss_row<T: PkNum>(
    ctx: &LikCtx,
    j: usize,
    f: T,
    p: &ParT<T>,
) -> Option<(T, T, T)> {
    let pk = p.wrap_rows::<1>();
    let fd = DualT::<1, T> {
        v: f,
        g: [T::from_f64(1.0)],
    };
    let ruv = DualT::<1, T> {
        v: T::from_f64(1.0),
        g: [T::from_f64(0.0)],
    };
    let vd = ctx.variance_v(j, fd, ruv, &pk);
    if !positive_finite(vd.v) {
        return None;
    }
    let lj = gauss_l(ctx.subject.observations[j], fd, vd);
    Some((lj.v, lj.g[0], focei_alpha(vd.v, vd.g[0])))
}

/// Row `j`'s `ℓ = r²/V + log V` alone at prediction `f` (an uncensored Gaussian row, as
/// [`focei_gauss_row`]): what a quadrature node's aggregate `Σℓ` reads (#2980) — no
/// derivative in `f` carried beyond `T`'s own.
pub(crate) fn gauss_row_l<T: PkNum>(ctx: &LikCtx, j: usize, f: T, p: &ParT<T>) -> Option<T> {
    let v = ctx.variance_v(j, f, T::from_f64(1.0), p);
    if !positive_finite(v) {
        return None;
    }
    Some(gauss_l(ctx.subject.observations[j], f, v))
}

/// Whether a subject's FOCEI objective is the row-and-reduce form [`focei_gauss_row`] and the
/// reduce `½·[ηᵀΩ⁻¹η + log|Ω| + log|Ω⁻¹ + Σ αGGᵀ|]` describe exactly: the diagonal path with
/// every row an uncensored Gaussian one, no RUV IIV, no SDE, no FREM rows, no non-Gaussian
/// records, no custom residual magnitude and a diagonal residual covariance — asked of
/// `ResidualCov::is_dense` (`stats::residual_error`), which sees every correlation source,
/// not only `block_sigma` pairs (#1081).
pub(crate) fn structured_rows_supported(ctx: &LikCtx) -> bool {
    let m = ctx.model;
    !ctx.cov.is_dense()
        && m.residual_error_eta.is_none()
        && !ctx.sde
        && m.frem_config.is_none()
        && !ctx.subject.fremtype.iter().any(|&ft| ft > 0)
        && ctx.subject.obs_records.is_empty()
        && !(0..ctx.n()).any(|j| ctx.censored(j).is_some())
        && !m.ruv_magnitude.as_ref().is_some_and(|rm| rm.is_active())
}

/// Row `j` of FOCE's Sheiner–Beal marginal as the structured covariance Hessian reads it
/// (#2980): `(iv·e², iv·e, iv, log V)` with `e = y − f`, `V = R⁰` at `r_eval` (the population
/// prediction under an `f`-dependent variance, else any — `V` does not read it) and `iv = 1/V`.
/// [`foce_f`]'s aggregates over these: `Σ iv·res²` with `res = e + G·η`, `b = Σ iv·G·res`,
/// `S = Σ iv·GGᵀ`, `Σ log V`.
pub(crate) fn foce_row<T: PkNum>(
    ctx: &LikCtx,
    j: usize,
    f: T,
    r_eval: T,
    p: &ParT<T>,
) -> Option<[T; 4]> {
    let v = ctx.variance_v(j, r_eval, T::from_f64(1.0), p);
    if !positive_finite(v) {
        return None;
    }
    let iv = T::from_f64(1.0) / v;
    let e = T::from_f64(ctx.subject.observations[j]) - f;
    Some([iv * e * e, iv * e, iv, v.ln()])
}

/// Row `j` of the Laplace objective as the structured covariance Hessian reads it (#2980):
/// `(ℓ, ℓ′, ℓ″)` of [`row_l`] in its prediction — the row term and the derivatives
/// [`inner_hessian_through_predictions`] builds the exact inner Hessian from.
pub(crate) fn laplace_row<T: PkNum>(ctx: &LikCtx, j: usize, f: T, p: &ParT<T>) -> Option<[T; 3]> {
    let pl = p.map_rows(d2_const);
    let el: [D2<T>; 1] = [d2_const(T::from_f64(0.0))];
    let fj = d2_seed(f);
    let l = row_l(
        ctx,
        j,
        fj,
        fj,
        d2_const(T::from_f64(0.0)),
        d2_const(T::from_f64(1.0)),
        &el,
        &pl,
    )?;
    Some([l.v.v, l.v.g[0], l.g[0].g[0]])
}

/// Whether [`foce_f`] takes its Woodbury form at `S = LᵀGᵀD⁻¹GL` with trace `tr_s` — the
/// form the structured FOCE kernel is (#2980): otherwise it rotates the rows into `I + S` (#2988).
pub(crate) fn foce_woodbury_ok(tr_s: f64) -> bool {
    1.0 + tr_s <= WOODBURY_MAX_INFO_TRACE || woodbury_forced()
}

/// `ηᵀΩ⁻¹η + log|Ω|` (the prior the objective reads), for the structured reduce.
pub(crate) fn prior_of<T: PkNum>(ctx: &LikCtx, eta: &[T], p: &ParT<T>) -> T {
    ctx.prior(eta, p)
}

/// [`focei_rows`] at the dual width the subject needs: direction 0 is the prediction, then
/// `η_r` under RUV IIV, then the row's `p_obs` under an SDE.
fn focei_rows_k<T: PkNum>(
    ctx: &LikCtx,
    f: &[T],
    g: &[Vec<T>],
    eta: &[T],
    p: &ParT<T>,
    ng: Option<&NonGauss<T>>,
    interaction: bool,
) -> Option<(T, Vec<T>, Vec<Vec<T>>)> {
    let rr = ctx.model.residual_error_eta;
    match (rr.is_some(), ctx.sde) {
        (false, false) => focei_rows::<T, 1>(ctx, f, g, eta, p, rr, ng, interaction),
        (true, false) | (false, true) => focei_rows::<T, 2>(ctx, f, g, eta, p, rr, ng, interaction),
        (true, true) => focei_rows::<T, 3>(ctx, f, g, eta, p, rr, ng, interaction),
    }
}

/// The diagonal-path row pass behind [`focei_f_grad`], over a `K`-direction dual:
/// direction 0 is the row's prediction (or, on a FREM row, its `η_e`), direction 1 is
/// `η_r` (`K = 2` only under RUV IIV). Mirrors `foce_subject_nll_interaction` /
/// `gaussian_foce_accum` term for term, with `H̃ = Σ aᵀa/V + ½·c̃ᵀc̃ + cens_hess + Ω⁻¹`;
/// `interaction = false` drops `½·c̃ᵀc̃` and the censored curvature (production's FOCE form
/// beside a non-Gaussian term), and `ng` adds that term (see [`NonGauss`]).
#[allow(clippy::too_many_arguments)]
fn focei_rows<T: PkNum, const K: usize>(
    ctx: &LikCtx,
    f: &[T],
    g: &[Vec<T>],
    eta: &[T],
    p: &ParT<T>,
    rr: Option<usize>,
    ng: Option<&NonGauss<T>>,
    interaction: bool,
) -> Option<(T, Vec<T>, Vec<Vec<T>>)> {
    let c = |x: f64| T::from_f64(x);
    let one = c(1.0);
    let nn = eta.len();
    let unit =
        |k: usize| -> Vec<T> { (0..nn).map(|l| c(if l == k { 1.0 } else { 0.0 })).collect() };
    let lift = |x: T| DualT::<K, T> {
        v: x,
        g: [c(0.0); K],
    };
    let pk = p.wrap_rows::<K>();
    let ed: Vec<DualT<K, T>> = (0..nn)
        .map(|k| {
            let mut d = lift(eta[k]);
            if K > 1 && Some(k) == rr {
                d.g[1] = one;
            }
            d
        })
        .collect();
    let ruv = ctx.ruv_scale(&ed);
    let n = ctx.n();
    // The `p_obs` direction (an SDE's), last.
    let ip = K - 1;
    let mut data = c(0.0);
    let mut bloq = c(0.0);
    let mut dl = vec![c(0.0); n];
    let mut explicit = vec![c(0.0); nn];
    // H̃ = Ω⁻¹ + Σ_t α_t·u_t·v_tᵀ; the terms come in symmetric pairs, so H̃ stays symmetric.
    let mut terms: Vec<(T, Cow<[T]>, Cow<[T]>)> = Vec::with_capacity(n + 1);
    for j in 0..n {
        let y = ctx.subject.observations[j];
        if let Some(row) = ctx.frem_row(j) {
            // FREM: prediction θ_i + η_e (unit η-sensitivity), fixed variance; no RUV
            // scale and no `c̃` column (`gaussian_foce_accum`'s `!is_pk_row`).
            let e = row.1;
            let mut el: Vec<DualT<K, T>> = eta.iter().map(|&x| lift(x)).collect();
            el[e].g[0] = one;
            let d = row_l(
                ctx,
                j,
                lift(f[j]),
                lift(f[j]),
                lift(c(0.0)),
                lift(one),
                &el,
                &pk,
            )?;
            data = data + d.v;
            explicit[e] = explicit[e] + d.g[0];
            terms.push((
                one / ctx.frem_pred_var(row, eta, p).1,
                Cow::Owned(unit(e)),
                Cow::Owned(unit(e)),
            ));
            continue;
        }
        let mut fd = lift(f[j]);
        fd.g[0] = one;
        let gj = &g[j];
        match ctx.censored(j) {
            None => {
                let mut vd = ctx.variance_v(j, fd, ruv, &pk);
                if ctx.sde {
                    // `V = V_resid + p_obs` (`H̃` below holds `p_obs` η-independent, as
                    // production does: `c̃` reads `∂V/∂f` only); `∇_η l` sees its η-jet.
                    let mut pd = lift(f[n + j]);
                    pd.g[ip] = one;
                    vd = vd + pd;
                }
                if !positive_finite(vd.v) {
                    return None;
                }
                let lj = gauss_l(y, fd, vd);
                data = data + lj.v;
                dl[j] = lj.g[0];
                if let Some(r) = rr {
                    explicit[r] = explicit[r] + lj.g[1];
                }
                if ctx.sde {
                    for k in 0..nn {
                        explicit[k] = explicit[k] + lj.g[ip] * g[n + j][k];
                    }
                }
                let iv = one / vd.v;
                let cs = vd.g[0] / vd.v;
                // g·gᵀ/V + ½·c̃c̃ᵀ with c̃ = (∂V/∂f / V)·g, whose η_r entry is 2 under RUV IIV.
                match (rr, interaction) {
                    (_, false) => terms.push((iv, Cow::Borrowed(gj), Cow::Borrowed(gj))),
                    (None, true) => terms.push((
                        focei_alpha(vd.v, vd.g[0]),
                        Cow::Borrowed(gj),
                        Cow::Borrowed(gj),
                    )),
                    (Some(r), true) => {
                        let ck: Vec<T> = (0..nn)
                            .map(|k| if k == r { c(2.0) } else { cs * gj[k] })
                            .collect();
                        terms.push((iv, Cow::Borrowed(gj), Cow::Borrowed(gj)));
                        terms.push((c(0.5), Cow::Owned(ck.clone()), Cow::Owned(ck)));
                    }
                }
            }
            Some(cens) => {
                let (v0, d, d2) = ctx.variance(j, f[j], ruv.v, p);
                let v = v0 + ctx.extra(f, j);
                if !positive_finite(v) {
                    return None;
                }
                let (l2, g2, cz, cm) = m3_terms(y, f[j], v, d, d2, cens);
                bloq = bloq + l2;
                // ∂ℓ/∂f, ∂ℓ/∂η_r and ∂ℓ/∂p_obs of ℓ = −2·log Φ(±(y − f)/√V), with
                // ∂V/∂η_r = 2·V_resid.
                let mut vk = lift(v);
                vk.g[0] = d;
                if rr.is_some() {
                    vk.g[1] = c(2.0) * v0;
                }
                if ctx.sde {
                    vk.g[ip] = one;
                }
                let sgn = if cens < 0 { -1.0 } else { 1.0 };
                let lj = lift(c(-2.0)) * log_phi(lift(c(sgn)) * (lift(c(y)) - fd) / vk.sqrt());
                dl[j] = lj.g[0];
                if let Some(r) = rr {
                    explicit[r] = explicit[r] + lj.g[1];
                }
                if ctx.sde {
                    for k in 0..nn {
                        explicit[k] = explicit[k] + lj.g[ip] * g[n + j][k];
                    }
                }
                if interaction {
                    terms.push((g2, Cow::Borrowed(gj), Cow::Borrowed(gj)));
                    if let Some(r) = rr {
                        let mut go = gj.clone();
                        go[r] = c(0.0);
                        terms.push((cz, Cow::Owned(unit(r)), Cow::Owned(unit(r))));
                        terms.push((cm, Cow::Owned(unit(r)), Cow::Owned(go.clone())));
                        terms.push((cm, Cow::Owned(go), Cow::Owned(unit(r))));
                    }
                }
            }
        }
    }
    let oi = p.omega_inv(nn);
    let mut ht: Vec<Vec<T>> = vec![vec![c(0.0); nn]; nn];
    for a in 0..nn {
        for b in 0..=a {
            let e = wsum3(
                0.0,
                std::iter::once((oi[a][b], one, one))
                    .chain(terms.iter().map(|(al, u, v)| (*al, u[a], v[b]))),
            );
            ht[a][b] = e;
            ht[b][a] = e;
        }
    }
    // The non-Gaussian term: `2q` in the data, `∇²q` in `H̃`, `∇q` in `∇l` (`l = ½·data`).
    if let Some(ng) = ng {
        data = data + c(2.0) * ng.q;
        for a in 0..nn {
            for b in 0..nn {
                ht[a][b] = ht[a][b] + ng.h[a][b];
            }
        }
    }
    let ld = chol(&ht).map(|l| chol_logdet(&l))?;
    let fv = c(0.5) * (data + ctx.prior(eta, p) + ld + bloq);
    let mut grad = assemble_grad(oi, eta, g, &dl, &explicit);
    if let Some(ng) = ng {
        for k in 0..nn {
            grad[k] = grad[k] + ng.g[k];
        }
    }
    Some((fv, grad, ht))
}

/// `∇_k l = Σ_l Ω⁻¹_kl η_l + ½·Σ_j (∂ℓ_j/∂f)·G_jk + ½·explicit_k`.
fn assemble_grad<T: PkNum>(
    oi: &[Vec<T>],
    eta: &[T],
    g: &[Vec<T>],
    dl: &[T],
    explicit: &[T],
) -> Vec<T> {
    let one = T::from_f64(1.0);
    let half = T::from_f64(0.5);
    (0..eta.len())
        .map(|k| {
            let prior = (0..eta.len()).map(|l| (oi[k][l], eta[l], one));
            let rows = (0..dl.len()).map(|j| (half, dl[j], g[j][k]));
            wsum3(0.0, prior.chain(rows)) + half * explicit[k]
        })
        .collect()
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
/// factors `R̃` directly; past the #1498 gate the rows are rotated into `I + S` (#2988).
pub(crate) fn foce_f<T: PkNum>(
    ctx: &LikCtx,
    f: &[T],
    g: &[Vec<T>],
    eta: &[T],
    fpop: Option<&[T]>,
    p: &ParT<T>,
) -> Option<T> {
    let c = |x: f64| T::from_f64(x);
    let one = c(1.0);
    let n = ctx.n();
    let nn = eta.len();
    let unit =
        |k: usize| -> Vec<T> { (0..nn).map(|l| c(if l == k { 1.0 } else { 0.0 })).collect() };
    // Per row: prediction, η-sensitivity, linearised mean f₀ and LᵀG (for GΩGᵀ = (LᵀG)ᵀ(LᵀG)).
    let lch = p.omega_chol(nn);
    let mut f0: Vec<T> = Vec::with_capacity(n);
    let mut lg: Vec<Vec<T>> = Vec::with_capacity(n);
    for j in 0..n {
        let (fj, gj) = match ctx.frem_row(j) {
            Some(row) => (ctx.frem_pred_var(row, eta, p).0, unit(row.1)),
            None => (f[j], g[j].clone()),
        };
        f0.push(fj - wsum3(0.0, (0..nn).map(|k| (one, gj[k], eta[k]))));
        lg.push(
            (0..nn)
                .map(|cc| wsum3(0.0, (cc..nn).map(|r| (one, lch[r][cc], gj[r]))))
                .collect(),
        );
    }
    let r_eval: Vec<T> = match fpop {
        Some(fp) => fp.to_vec(),
        None => f0.clone(),
    };
    // R⁰'s diagonal (FREM rows overridden), and the censored rows' `−2·log Φ`.
    // An SDE row adds its EKF `p_obs` (at the mode, held: the marginal linearises η out).
    let r0_diag = |j: usize, rd: T| -> T {
        match ctx.frem_row(j) {
            Some(row) => ctx.frem_pred_var(row, eta, p).1,
            None => rd + ctx.extra(f, j),
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
        if !positive_finite(rd) {
            return None;
        }
        rdiag.push(rd);
        match ctx.censored(j) {
            Some(cens) if ctx.frem_row(j).is_none() => {
                let rt = wsum3(0.0, (0..nn).map(|k| (one, lg[j][k], lg[j][k]))) + rd;
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
    // `WOODBURY_MAX_INFO_TRACE`, or when it is not finite, the rows are rotated into
    // `I + S` instead. A dense `R⁰` factors `R̃` at observation size.
    let woodbury = rm.is_none() && {
        let tr = quant.iter().fold(c(0.0), |acc, &j| {
            let s = (0..nn).fold(c(0.0), |a, k| a + lg[j][k] * lg[j][k]);
            acc + s / rdiag[j]
        });
        (one + tr).br_cmp_c(BranchCmp::Le, WOODBURY_MAX_INFO_TRACE) || woodbury_forced()
    };
    if rm.is_some() {
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
                        wsum3(0.0, (0..nn).map(|k| (one, lg[ja][k], lg[jb][k]))) + r0
                    })
                    .collect()
            })
            .collect();
        let l = chol(&rt)?;
        let z = chol_solve(&l, &res);
        let quad = wsum3(0.0, (0..nq).map(|i| (one, res[i], z[i])));
        return Some(c(0.5) * (quad + chol_logdet(&l) + bloq));
    }
    let mut log_r = c(0.0);
    for &j in &quant {
        log_r = log_r + rdiag[j].ln();
    }
    if !woodbury {
        // `min_w ‖w‖² + ‖D^-½(res − GL·w)‖²` (the same quadratic form) by Givens rotations of
        // each row into the triangular factor `R` of `I + S`, started at `I`: no subtraction of
        // large terms, every pivot ≥ 1, and `O(nq·nn²)` work where an `nq × nq` `R̃` costs a
        // compiled plan `O(nq³)` instructions (#2988).
        let mut r: Vec<Vec<T>> = (0..nn)
            .map(|k| (0..nn).map(|l| c(if k == l { 1.0 } else { 0.0 })).collect())
            .collect();
        let mut z = vec![c(0.0); nn];
        let mut quad = c(0.0);
        for (i, &j) in quant.iter().enumerate() {
            let w = one / rdiag[j].sqrt();
            let mut v: Vec<T> = lg[j].iter().map(|&x| x * w).collect();
            let mut t = res[i] * w;
            for k in 0..nn {
                let h = (r[k][k] * r[k][k] + v[k] * v[k]).sqrt();
                let (cs, sn) = (r[k][k] / h, v[k] / h);
                r[k][k] = h;
                for l in k + 1..nn {
                    (r[k][l], v[l]) = (cs * r[k][l] + sn * v[l], cs * v[l] - sn * r[k][l]);
                }
                (z[k], t) = (cs * z[k] + sn * t, cs * t - sn * z[k]);
            }
            quad = quad + t * t;
        }
        let ld = (0..nn).fold(c(0.0), |a, k| a + c(2.0) * r[k][k].ln());
        // A non-finite `S` declines, as the dense factor it replaces did.
        return positive_finite(one + quad + ld).then(|| c(0.5) * (quad + log_r + ld + bloq));
    }
    // Woodbury / determinant lemma on the diagonal R⁰, Ω-normalised: with `B = D⁻¹GL`,
    // `R̃⁻¹ = D⁻¹ − B(I + S)⁻¹Bᵀ` and `|R̃| = |D|·|I + S|`. Factoring `I + S` rather than
    // `Ω⁻¹ + GᵀD⁻¹G` keeps `cond(Ω)` (a near-rail `block_omega`) out of the subtraction,
    // which the trace gate above does not see.
    let iv: Vec<T> = quant.iter().map(|&j| one / rdiag[j]).collect();
    let mut a: Vec<Vec<T>> = vec![vec![c(0.0); nn]; nn];
    for k in 0..nn {
        for l in 0..=k {
            let e = wsum3(
                if k == l { 1.0 } else { 0.0 },
                (0..nq).map(|i| (iv[i], lg[quant[i]][k], lg[quant[i]][l])),
            );
            a[k][l] = e;
            a[l][k] = e;
        }
    }
    let b: Vec<T> = (0..nn)
        .map(|k| wsum3(0.0, (0..nq).map(|i| (iv[i], lg[quant[i]][k], res[i]))))
        .collect();
    let la = chol(&a)?;
    let ab = chol_solve(&la, &b);
    let quad = wsum3(0.0, (0..nq).map(|i| (iv[i], res[i], res[i])))
        - wsum3(0.0, (0..nn).map(|k| (one, b[k], ab[k])));
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

/// The predictions and η as one-direction duals along `η_k`: prediction `j` carries
/// `G_jk` and `η_m` the unit vector `e_k`, so a quantity built from them has `∂/∂η_k` in its
/// tangent. One pass per `k` replaces a dual wide in every η, so no width is baked into a type.
fn direction_duals<T: PkNum>(
    f: &[T],
    g: &[Vec<T>],
    eta: &[T],
    k: usize,
) -> (Vec<DualT<1, T>>, Vec<DualT<1, T>>) {
    let c = |x: f64| T::from_f64(x);
    let fd = (0..f.len())
        .map(|j| DualT {
            v: f[j],
            g: [g[j][k]],
        })
        .collect();
    let ed = (0..eta.len())
        .map(|m| DualT {
            v: eta[m],
            g: [c(if m == k { 1.0 } else { 0.0 })],
        })
        .collect();
    (fd, ed)
}

/// Dense-`R` FOCEI `F`: `½(rᵀR⁻¹r + log|R| + ηᵀΩ⁻¹η + log|Ω| + log|H̃|)`.
fn focei_f_dense<T: PkNum>(
    ctx: &LikCtx,
    f: &[T],
    g: &[Vec<T>],
    eta: &[T],
    p: &ParT<T>,
) -> Option<(T, Vec<Vec<T>>)> {
    let c = |x: f64| T::from_f64(x);
    let n = ctx.n();
    let nn = eta.len();
    let ruv = ctx.ruv_scale(eta);
    let mut rv: Vec<Vec<T>> = ctx
        .r_matrix(f, p)
        .into_iter()
        .map(|row| row.into_iter().map(|x| x * ruv).collect())
        .collect();
    for j in 0..n {
        rv[j][j] = rv[j][j] + ctx.extra(f, j);
    }
    let l = chol(&rv)?;
    let res: Vec<T> = (0..n)
        .map(|j| c(ctx.subject.observations[j]) - f[j])
        .collect();
    let z = chol_solve(&l, &res);
    let mut data = chol_logdet(&l);
    for j in 0..n {
        data = data + res[j] * z[j];
    }
    // R⁻¹G (n × N) and R⁻¹·∂_kR (n × n per k). `∂_kR` comes from `R` over a one-direction
    // dual per `k` ([`direction_duals`]): its tangent is exactly `Σ_m G_mk ∂R/∂f_m`, plus
    // `∂R/∂η_r` through the `exp(2η_r)` scale.
    let rg: Vec<Vec<T>> = (0..nn)
        .map(|k| chol_solve(&l, &(0..n).map(|j| g[j][k]).collect::<Vec<T>>()))
        .collect();
    let rd_k: Vec<Vec<Vec<T>>> = (0..nn)
        .map(|k| {
            let (fd, ed) = direction_duals(f, g, eta, k);
            let ruv = ctx.ruv_scale(&ed);
            let rd = ctx.r_matrix(&fd, &p.wrap_rows::<1>());
            (0..n)
                .map(|col| {
                    chol_solve(
                        &l,
                        &(0..n)
                            .map(|i| {
                                let x = rd[i][col] * ruv;
                                (if i == col { x + ctx.extra(&fd, i) } else { x }).g[0]
                            })
                            .collect::<Vec<T>>(),
                    )
                })
                .collect()
        })
        .collect();
    let mut ht: Vec<Vec<T>> = p.omega_inv(nn).clone();
    for a in 0..nn {
        for b in 0..nn {
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
/// frozen population prediction) plus `extra` (an SDE row's EKF `p_obs`, else zero). The single row definition behind both the value of `l`
/// and its η-gradient.
#[allow(clippy::too_many_arguments)]
fn row_l<T: PkNum>(
    ctx: &LikCtx,
    j: usize,
    f: T,
    vf: T,
    extra: T,
    ruv: T,
    eta: &[T],
    p: &ParT<T>,
) -> Option<T> {
    let c = |x: f64| T::from_f64(x);
    let y = ctx.subject.observations[j];
    if let Some(row) = ctx.frem_row(j) {
        let (fp, v) = ctx.frem_pred_var(row, eta, p);
        let r = c(y) - fp;
        return Some(r * r / v + v.ln());
    }
    let v = ctx.variance_v(j, vf, ruv, p) + extra;
    if !positive_finite(v) {
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
pub(crate) fn inner_l<T: PkNum>(
    ctx: &LikCtx,
    f: &[T],
    eta: &[T],
    p: &ParT<T>,
    frozen: Option<&[T]>,
) -> Option<T> {
    let data = data_l(ctx, f, eta, p, frozen)?;
    Some(T::from_f64(0.5) * (ctx.prior(eta, p) + data))
}

/// The data term of [`inner_l`], `−2 log p(y | f)` up to its constant: the residual
/// likelihood at the predictions `f` (dense or per row, censored rows by M3, FREM rows by
/// their own variance), with the random effects entering only through `f` and the
/// residual-error η's scale. `inner_l` is `½(prior + data_l)`.
pub(crate) fn data_l<T: PkNum>(
    ctx: &LikCtx,
    f: &[T],
    eta: &[T],
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
        let n = ctx.n();
        let mut r: Vec<Vec<T>> = ctx
            .r_matrix(vf, p)
            .into_iter()
            .map(|row| row.into_iter().map(|x| x * ruv).collect())
            .collect();
        for j in 0..n {
            r[j][j] = r[j][j] + ctx.extra(f, j);
        }
        let l = chol(&r)?;
        let res: Vec<T> = (0..n)
            .map(|j| c(ctx.subject.observations[j]) - f[j])
            .collect();
        let z = chol_solve(&l, &res);
        let mut data = chol_logdet(&l);
        for j in 0..n {
            data = data + res[j] * z[j];
        }
        return Some(data);
    }
    let mut data = c(0.0);
    for j in 0..ctx.n() {
        data = data + row_l(ctx, j, f[j], vf[j], ctx.extra(f, j), ruv, eta, p)?;
    }
    Some(data)
}

/// A 1-D second-order dual: value, first and second derivative along one direction.
type D2<T> = DualT<1, DualT<1, T>>;

/// `x` as a [`D2`] constant.
fn d2_const<T: PkNum>(x: T) -> D2<T> {
    let zero = T::from_f64(0.0);
    DualT {
        v: DualT { v: x, g: [zero] },
        g: [DualT { v: zero, g: [zero] }],
    }
}

/// `x` as the [`D2`] variable: unit first derivative, zero second.
fn d2_seed<T: PkNum>(x: T) -> D2<T> {
    let (zero, one) = (T::from_f64(0.0), T::from_f64(1.0));
    DualT {
        v: DualT { v: x, g: [one] },
        g: [DualT { v: one, g: [zero] }],
    }
}

/// Whether some row of the conditional [`inner_l`] reads η other than through its own
/// prediction: the dense residual path (rows coupled through `R`), a FREM row (`θ_i + η_e`),
/// the RUV-IIV scale `exp(2η_r)`, or an SDE's `p_obs` (a second η-dependent input per row).
/// Exactly the cases the chain rule through the
/// predictions does not cover.
fn reads_eta_directly(ctx: &LikCtx) -> bool {
    ctx.dense()
        || ctx.sde
        || ctx.model.residual_error_eta.is_some()
        || (0..ctx.subject.observations.len()).any(|j| ctx.frem_row(j).is_some())
}

/// Test watch: per watched subject `(address, closed-form count, generic count)` — how many
/// times each implementation ([`inner_hessian_through_predictions`],
/// [`inner_hessian_generic`]) ran for it, on any thread. Without it the route is
/// unobservable: both give the same matrix, so a site that silently fell back to the generic
/// path (losing the speedup) would leave every test green. Counted where each runs, not at the
/// dispatch, so a fallback inside a branch is seen too; a list, so concurrent tests watching
/// their own subjects do not overwrite each other.
#[cfg(test)]
pub(crate) static INNER_HESSIAN_ROUTE: std::sync::Mutex<Vec<(usize, usize, usize)>> =
    std::sync::Mutex::new(Vec::new());

/// Count one run of an implementation for `subject` if a test watches it.
#[cfg(test)]
fn note_inner_hessian_route(subject: &Subject, generic: bool) {
    if let Ok(mut watch) = INNER_HESSIAN_ROUTE.lock() {
        let addr = subject as *const Subject as usize;
        if let Some(e) = watch.iter_mut().find(|e| e.0 == addr) {
            if generic {
                e.2 += 1;
            } else {
                e.1 += 1;
            }
        }
    }
}

/// The exact η-Hessian `∂²l/∂η∂ηᵀ` of the conditional [`inner_l`] (`frozen = None`) at
/// predictions with values `f`, η-gradients `G` and packed η-Hessians `Q` (the full mixed
/// partials at `tri(k, l)`), for every site that anchors on it: the Laplace/AGQ gradient
/// (taped and untaped), `exact_inner_hessian`, and the covariance step.
///
/// By the chain rule through the predictions ([`inner_hessian_through_predictions`]) unless a
/// row reads η directly ([`reads_eta_directly`]), and otherwise by [`inner_l`] over a
/// second-order dual — the one decline rule and the one fallback, here rather than at each
/// site. `None` only where the objective itself is undefined at these jets (a row with no
/// valid variance): the generic path fails at the same row, so it is not re-run.
///
pub(crate) fn exact_inner_hessian_kernel<T: PkNum>(
    ctx: &LikCtx,
    f: &[T],
    g: &[Vec<T>],
    q: &[Vec<T>],
    eta: &[T],
    p: &ParT<T>,
) -> Option<Vec<Vec<T>>> {
    if reads_eta_directly(ctx) {
        inner_hessian_generic(ctx, f, g, q, eta, p)
    } else {
        inner_hessian_through_predictions(ctx, f, g, q, p)
    }
}

/// The exact inner Hessian by the chain rule through each row's prediction:
///
/// `∂²l/∂η∂ηᵀ = ½(Ω⁻¹ + Ω⁻ᵀ) + ½·Σⱼ (ℓⱼ″·GⱼGⱼᵀ + ℓⱼ′·Qⱼ)`
///
/// with `ℓⱼ′`, `ℓⱼ″` the derivatives of the row term [`row_l`] in its prediction, from one
/// pass of `row_l` over a [`D2`]. Every row kind that reads η only through its prediction is
/// covered by that one definition — Gaussian under any error model, M3-censored. A row costs
/// one 4-component pass plus the `n(n+1)/2` assembly. The caller has established that no row
/// reads η directly ([`exact_inner_hessian_kernel`]); `None` is an undefined row.
pub(crate) fn inner_hessian_through_predictions<T: PkNum>(
    ctx: &LikCtx,
    f: &[T],
    g: &[Vec<T>],
    q: &[Vec<T>],
    p: &ParT<T>,
) -> Option<Vec<Vec<T>>> {
    #[cfg(test)]
    note_inner_hessian_route(ctx.subject, false);
    let n = p.omega.inv.len();
    // The width is Ω⁻¹'s own, so a `ParT` without Ω (`map_rows`) would give `n = 0` and a
    // silently empty Hessian: the rows' η-sensitivities say how wide it must be, and a
    // mismatch declines (in every build) rather than assembling a wrong-width matrix.
    if g.first().is_some_and(|r| r.len() != n) {
        return None;
    }
    let half = T::from_f64(0.5);
    let pl = p.map_rows(d2_const);
    // A non-FREM row does not read η; the array only fills `row_l`'s signature.
    let el: [D2<T>; 1] = [d2_const(T::from_f64(0.0))];
    let ruv = d2_const(T::from_f64(1.0));
    let oi = &p.omega.inv;
    let mut h: Vec<Vec<T>> = (0..n)
        .map(|a| (0..n).map(|b| half * (oi[a][b] + oi[b][a])).collect())
        .collect();
    for j in 0..f.len() {
        let fj = d2_seed(f[j]);
        let l = row_l(ctx, j, fj, fj, d2_const(T::from_f64(0.0)), ruv, &el, &pl)?;
        // Row invariants hoisted: `½ℓ″`, `½ℓ′`, and `½ℓ″·G_a` per `a`.
        let (hd1, hd2) = (half * l.v.g[0], half * l.g[0].g[0]);
        let (gj, qj) = (&g[j], &q[j]);
        for a in 0..n {
            let ga = hd2 * gj[a];
            for b in 0..=a {
                let v = ga * gj[b] + hd1 * qj[crate::sens::jet::tri(a, b)];
                h[a][b] = h[a][b] + v;
                if a != b {
                    h[b][a] = h[b][a] + v;
                }
            }
        }
    }
    Some(h)
}

/// The exact inner Hessian as [`inner_l`] over mixed second-order duals, one per pair
/// `(a, b)`: the definition, and the route for rows that read η directly. The outer
/// direction is `η_a` and the inner `η_b`, the predictions carry `(f, G_·b, G_·a, Q_·ab)` and
/// η its two unit vectors, so `l`'s `∂²/∂η_a∂η_b` is the entry. `None` where `inner_l` is.
pub(crate) fn inner_hessian_generic<T: PkNum>(
    ctx: &LikCtx,
    f: &[T],
    g: &[Vec<T>],
    q: &[Vec<T>],
    eta: &[T],
    p: &ParT<T>,
) -> Option<Vec<Vec<T>>> {
    #[cfg(test)]
    note_inner_hessian_route(ctx.subject, true);
    let n = eta.len();
    let (zero, one) = (T::from_f64(0.0), T::from_f64(1.0));
    let pd = p.map(d2_const);
    let mut h: Vec<Vec<T>> = vec![vec![zero; n]; n];
    for a in 0..n {
        for b in 0..=a {
            let t = crate::sens::jet::tri(a, b);
            let fh: Vec<D2<T>> = (0..f.len())
                .map(|j| DualT {
                    v: DualT {
                        v: f[j],
                        g: [g[j][b]],
                    },
                    g: [DualT {
                        v: g[j][a],
                        g: [q[j][t]],
                    }],
                })
                .collect();
            let eh: Vec<D2<T>> = (0..n)
                .map(|m| DualT {
                    v: DualT {
                        v: eta[m],
                        g: [if m == b { one } else { zero }],
                    },
                    g: [DualT {
                        v: if m == a { one } else { zero },
                        g: [zero],
                    }],
                })
                .collect();
            let e = inner_l(ctx, &fh, &eh, &pd, None)?.g[0].g[0];
            h[a][b] = e;
            h[b][a] = e;
        }
    }
    Some(h)
}

/// `∇_η l` at `(f, G, η)`, under the conditional or FOCE's `frozen` convention (see
/// [`inner_l`]). Row-separable on the diagonal path: each row contributes
/// `½·(∂ℓ_j/∂f)·G_j` plus its explicit η-dependence (the RUV-IIV scale, or a FREM row's
/// `θ_i + η_e`), from a small dual over the row term [`row_l`] (the conditional case is
/// [`focei_f_grad`]'s row pass); the prior adds `Ω⁻¹η`. The dense path couples rows, so
/// there `l` itself is evaluated over an η-dual whose predictions carry `G`.
pub(crate) fn inner_grad<T: PkNum>(
    ctx: &LikCtx,
    f: &[T],
    g: &[Vec<T>],
    eta: &[T],
    p: &ParT<T>,
    frozen: Option<&[T]>,
) -> Option<Vec<T>> {
    let c = |x: f64| T::from_f64(x);
    if ctx.dense() {
        let p1 = p.wrap::<1>();
        return (0..eta.len())
            .map(|k| {
                let (fd, ed) = direction_duals(f, g, eta, k);
                let fz: Option<Vec<DualT<1, T>>> =
                    frozen.map(|fz| fz.iter().map(|&x| DualT { v: x, g: [c(0.0)] }).collect());
                Some(inner_l(ctx, &fd, &ed, &p1, fz.as_deref())?.g[0])
            })
            .collect();
    }
    let Some(fz) = frozen else {
        return Some(focei_f_grad(ctx, f, g, eta, p)?.1);
    };
    // FOCE's frozen variance: a row reads η only through f (FREM rows through η_e).
    let p1 = p.wrap_rows::<1>();
    let lift = |x: T| DualT { v: x, g: [c(0.0)] };
    let n = ctx.n();
    let mut dl = vec![c(0.0); n];
    let mut explicit = vec![c(0.0); eta.len()];
    for j in 0..n {
        let mut el: Vec<DualT<1, T>> = eta.iter().map(|&x| lift(x)).collect();
        if let Some((_, e, _)) = ctx.frem_row(j) {
            el[e].g[0] = c(1.0);
            let d = row_l(
                ctx,
                j,
                lift(f[j]),
                lift(f[j]),
                lift(c(0.0)),
                lift(c(1.0)),
                &el,
                &p1,
            )?;
            explicit[e] = explicit[e] + d.g[0];
            continue;
        }
        let px = lift(ctx.extra(f, j));
        dl[j] = row_l(
            ctx,
            j,
            DualT {
                v: f[j],
                g: [c(1.0)],
            },
            lift(fz[j]),
            px,
            lift(c(1.0)),
            &el,
            &p1,
        )?
        .g[0];
        if ctx.sde {
            // ∂ℓ/∂p_obs times the row's η-jet of `p_obs`.
            let dp = row_l(
                ctx,
                j,
                lift(f[j]),
                lift(fz[j]),
                DualT {
                    v: px.v,
                    g: [c(1.0)],
                },
                lift(c(1.0)),
                &el,
                &p1,
            )?
            .g[0];
            for k in 0..eta.len() {
                explicit[k] = explicit[k] + dp * g[n + j][k];
            }
        }
    }
    let grad = assemble_grad(p.omega_inv(eta.len()), eta, g, &dl, &explicit);
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
        if !d.br_cmp_c(BranchCmp::Gt, 0.0) {
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
pub(crate) fn chol_solve<T: PkNum>(l: &[Vec<T>], b: &[T]) -> Vec<T> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sens::dual2::Dual2;

    /// `from_chol_cached` returns exactly `from_chol`'s `Ω⁻¹` and `log|Ω|`, and says when it
    /// computed: on a miss, on a hit, after the factor changes (a stale entry would return the
    /// previous Ω's inverse), for a factor that differs only in the sign of a lower-triangle
    /// zero (a miss: reuse is by bit pattern), at widths 1, 2, 3 and 4, and when the widths
    /// alternate (IOV with unequal occasion counts: one entry per width, so the alternation
    /// after warm-up is all hits). Mutation: compare with `==` instead of `to_bits` → the
    /// `-0.0` row hits; keep one entry → the alternation rows miss; skip the width lookup
    /// (compare against the first entry) → a width change serves the wrong inverse.
    #[test]
    fn omega_cache_returns_exactly_what_from_chol_computes() {
        let m3 = vec![
            vec![0.3, 0.0, 0.0],
            vec![0.0, 0.5, 0.0],
            vec![-0.2, 0.05, 0.4],
        ];
        let mut b = m3.clone();
        b[0][0] = 0.31;
        let mut z = m3.clone();
        z[1][0] = -0.0;
        let m1 = vec![vec![0.7]];
        let m2 = vec![vec![0.3, 0.0], vec![0.1, 0.5]];
        let m4: Vec<Vec<f64>> = (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        if j <= i {
                            0.2 + 0.1 * (i + j) as f64
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        let misses = || OMEGA_MISSES.with(std::cell::Cell::get);
        let bits =
            |m: &[Vec<f64>]| -> Vec<u64> { m.iter().flatten().map(|x| x.to_bits()).collect() };
        // (factor, whether the call has to compute)
        let rows: Vec<(&str, &Vec<Vec<f64>>, bool)> = vec![
            ("first", &m3, true),
            ("again", &m3, false),
            ("other values", &b, true),
            ("back", &m3, true),
            ("sign of a lower-triangle zero", &z, true),
            ("same -0.0 again", &z, false),
            ("n = 1", &m1, true),
            ("n = 4", &m4, true),
            ("n = 4 again", &m4, false),
            ("width 3 (kept beside the others)", &z, false),
            ("n = 2", &m2, true),
            ("alternate 3", &z, false),
            ("alternate 2", &m2, false),
            ("alternate 1", &m1, false),
            ("alternate 4", &m4, false),
        ];
        for (what, chol, computes) in rows {
            let before = misses();
            let want = OmegaT::from_chol(chol.clone()).expect("positive definite");
            let got = OmegaT::from_chol_cached(chol.clone()).expect("positive definite");
            assert_eq!(misses() - before, usize::from(computes), "{what}: computed");
            assert_eq!(bits(&got.chol), bits(&want.chol), "{what}");
            assert_eq!(bits(&got.inv), bits(&want.inv), "{what}");
            assert_eq!(got.log_det.to_bits(), want.log_det.to_bits(), "{what}");
        }
        // A non-positive diagonal declines, from the cache as from `from_chol`.
        let mut bad = m3.clone();
        bad[1][1] = 0.0;
        assert!(OmegaT::from_chol_cached(bad).is_none());
        // More widths than the cache keeps evict, and every answer stays exact.
        for n in 1..=OMEGA_CACHE_WIDTHS + 2 {
            let l: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            if i == j {
                                1.0 + n as f64
                            } else if j < i {
                                0.1
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect();
            let want = OmegaT::from_chol(l.clone()).expect("positive definite");
            let got = OmegaT::from_chol_cached(l).expect("positive definite");
            assert_eq!(bits(&got.inv), bits(&want.inv), "n = {n}");
        }
    }

    /// Reading Ω from the row passes' parameters fails with the cause, in every build, not
    /// with an index panic or an empty matrix; the full parameters answer at any width up to
    /// their own. Mutation: drop the `assert!` in `omega_inv` / `omega_chol` → the `catch_unwind`
    /// rows see no panic with the cause.
    #[test]
    fn omega_readers_name_a_parameter_set_without_omega() {
        let p = ParT::<f64> {
            omega: OmegaT::from_chol(vec![vec![0.3, 0.0], vec![0.1, 0.5]]).unwrap(),
            sigma: vec![],
            rho: vec![],
            serial_rho: None,
            theta: vec![],
            ruv_mult: None,
        };
        assert_eq!(p.omega_inv(2).len(), 2);
        assert_eq!(p.omega_chol(1).len(), 2);
        let rows = p.map_rows(|x| x);
        let msg = |f: &dyn Fn()| {
            let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("panics");
            e.downcast_ref::<String>().cloned().unwrap_or_default()
        };
        assert!(msg(&|| {
            rows.omega_inv(2);
        })
        .contains("carries no Ω"));
        assert!(msg(&|| {
            rows.omega_chol(2);
        })
        .contains("carries no Ω"));
    }

    /// The row passes' lifted parameters (`map_rows`) carry every field the rows read — σ, ρ,
    /// the serial ρ, θ, the RUV multipliers, `log|Ω|` — and drop only Ω's factor and inverse.
    #[test]
    fn map_rows_keeps_what_the_rows_read_and_drops_omega() {
        let p = ParT::<f64> {
            omega: OmegaT::from_chol(vec![vec![0.3, 0.0], vec![0.1, 0.5]]).unwrap(),
            sigma: vec![0.1, 0.2],
            rho: vec![0.3],
            serial_rho: Some(0.4),
            theta: vec![1.0, 2.0, 3.0],
            ruv_mult: Some(vec![vec![1.5], vec![2.5]]),
        };
        let (full, rows) = (p.map(|x| x * 2.0), p.map_rows(|x| x * 2.0));
        assert!(rows.omega.chol.is_empty() && rows.omega.inv.is_empty());
        assert_eq!(rows.omega.log_det, full.omega.log_det);
        assert_eq!(rows.sigma, full.sigma);
        assert_eq!(rows.rho, full.rho);
        assert_eq!(rows.serial_rho, full.serial_rho);
        assert_eq!(rows.theta, full.theta);
        assert_eq!(rows.ruv_mult, full.ruv_mult);
    }

    /// #2762 on the adjoint's M3 kernel (`m3_terms`, behind the exact FOCEI gradient, `H̃`
    /// and R-matrix). It lifts the f64 `log_normal_cdf` / `inv_mills` by hand, so the value-
    /// path likelihood test never reaches it (#2767 review r1 #4). A censored limit at `±1e300`
    /// puts `z` past the `normal_cdf` overflow, where `log_normal_cdf` was `NaN`. The row is
    /// certain, so every term and its jet in `f` is exactly `0`.
    ///
    /// #2772 extends it to every term and to `z = +∞`. Past `h = φ/Φ`'s underflow the
    /// `iiv_on_ruv` cross terms `cz`/`cm` form `h·z²`, `0·∞` once `z²` overflows (`±1e300`),
    /// and at `z = +∞` (`±1e308`) `log_phi`'s own `h′ = −h(z + h)` is `0·∞` too. All their
    /// limits are `0`. The oracle is that exact `0`, not a neighbouring row: `±1e6` is
    /// `z = 1e7`, past `h`'s underflow too, so it takes the same guards and a wrong guard
    /// constant would agree with itself there (#2782 review r1 #1). `==`, not bits, so the
    /// guard's `+0` and a computed `−0` both pass while `NaN` still fails.
    #[test]
    fn m3_terms_score_a_limit_past_the_normal_cdf_overflow_as_certain() {
        // Proportional error at f = 1, σ = 0.1: v = σ²f², d = 2σ²f, d2 = 2σ², all carried
        // in the dual so the jets of the terms move with f.
        let terms = |limit: f64, cens: i8| {
            let f = Dual2::<1>::var(1.0, 0);
            let s2 = Dual2::<1>::from_f64(0.01);
            let two = Dual2::<1>::from_f64(2.0);
            m3_terms(limit, f, s2 * f * f, two * s2 * f, two * s2, cens)
        };
        let z = |limit: f64, cens: i8| (if cens < 0 { -1.0 } else { 1.0 }) * (limit - 1.0) / 0.1;
        for (near, sq_over, inf, cens) in [(1e6, 1e300, 1e308, 1), (-1e6, -1e300, -1e308, -1)] {
            // The straddles: z² finite at the control, overflowing at `sq_over`, and z
            // itself infinite at `inf`.
            assert!(z(near, cens).powi(2).is_finite(), "{near:e}");
            assert!(z(sq_over, cens).is_finite() && z(sq_over, cens).powi(2) == f64::INFINITY);
            assert_eq!(z(inf, cens), f64::INFINITY, "{inf:e}");
            for x in [near, sq_over, inf] {
                let got = terms(x, cens);
                for (name, g) in [
                    ("value", got.0),
                    ("g2", got.1),
                    ("cz", got.2),
                    ("cm", got.3),
                ] {
                    assert!(g.val() == 0.0, "{name} at {x:e}: {}", g.val());
                    assert!(g.grad[0] == 0.0, "∂{name}/∂f at {x:e}: {}", g.grad[0]);
                    assert!(
                        g.hess[0][0] == 0.0,
                        "∂²{name}/∂f² at {x:e}: {}",
                        g.hess[0][0]
                    );
                }
            }
        }
    }

    /// #2772 residual risk after the merge with #2786: `log_phi` is also called directly (the
    /// exact-anchor, the lifted nested and the value-path likelihood rows), not only through
    /// `m3_terms`. At `z = +∞` reached as `r / w` with `r ≈ 1e308`, `z`'s own derivatives
    /// overflow, so the chain `h · z′` with `h = 0` is `0 · ∞`. The oracle is the exact `0`
    /// jet; `==` so `NaN` fails. The straddle: `near` is `h > 0`-free but finite-chain
    /// (`z = 1e7`), `inf` is `z = +∞`.
    #[test]
    fn log_phi_at_a_certain_row_has_a_zero_jet_where_z_overflows() {
        for (limit, sgn) in [(1e6, 1.0), (1e300, 1.0), (1e308, 1.0), (-1e308, -1.0)] {
            let f = Dual2::<1>::var(1.0, 0);
            let s2 = Dual2::<1>::from_f64(0.01);
            let w = (s2 * f * f).sqrt();
            let z = Dual2::<1>::from_f64(sgn) * (Dual2::<1>::from_f64(limit) - f) / w;
            let g = log_phi(z);
            assert!(g.val() == 0.0, "value at {limit:e}: {}", g.val());
            assert!(g.grad[0] == 0.0, "d/df at {limit:e}: {}", g.grad[0]);
            assert!(g.hess[0][0] == 0.0, "d2/df2 at {limit:e}: {}", g.hess[0][0]);
        }
        // The straddle: the chain really is non-finite at the extreme, finite at the control.
        let f = Dual2::<1>::var(1.0, 0);
        let w = (Dual2::<1>::from_f64(0.01) * f * f).sqrt();
        let z = (Dual2::<1>::from_f64(1e308) - f) / w;
        assert!(z.val() == f64::INFINITY && !z.grad[0].is_finite());
    }
}
