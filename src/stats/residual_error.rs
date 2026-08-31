use crate::types::{ErrorModel, ErrorSpec, ResidualCorrelation, SigmaType, Subject, SubjectResult};
use nalgebra::DMatrix;

const MIN_VARIANCE: f64 = 1e-12;

/// Compute residual variance for a single observation
/// sigma_values: `[sigma1]` for additive/proportional, `[sigma1, sigma2]` for combined
pub fn residual_variance(error_model: ErrorModel, f_pred: f64, sigma_values: &[f64]) -> f64 {
    let v = match error_model {
        ErrorModel::Additive => {
            // V = sigma1^2
            sigma_values[0] * sigma_values[0]
        }
        ErrorModel::Proportional => {
            // V = (f * sigma1)^2
            let fs = f_pred * sigma_values[0];
            fs * fs
        }
        ErrorModel::Combined => {
            // V = (f * sigma1)^2 + sigma2^2
            let prop = f_pred * sigma_values[0];
            prop * prop + sigma_values[1] * sigma_values[1]
        }
    };
    v.max(MIN_VARIANCE)
}

// ── ErrorSpec residual-variance math ─────────────────────────────────────────
//
// This module is the single owner of the residual variance-and-derivatives
// math, so the `ErrorSpec` dispatch layer lives with its primitive
// (`residual_variance`, above) and its consumers (below). It previously sat in
// `types.rs`, which meant `variance_at` had to reach back across the module
// boundary for `residual_variance`; `types.rs` now keeps only the `ErrorSpec`
// data definition + `obs_key`/`obs_keys` (data/dispatch concerns).
impl ErrorSpec {
    /// Observation-level loadings onto the flat residual-error vector.
    ///
    /// For an observation with prediction `f`, additive error contributes a
    /// coefficient of `1`, proportional error contributes `f`, and combined
    /// error contributes both. Multiplying these loadings by the residual
    /// sigma covariance matrix yields the scalar variance on the diagonal and
    /// the cross-observation covariance off diagonal.
    pub fn sigma_loadings(&self, cmt: usize, f: f64, n_sigma: usize) -> Vec<(usize, f64)> {
        match self {
            ErrorSpec::Single(em) => match em {
                ErrorModel::Additive => {
                    if n_sigma > 0 {
                        vec![(0, 1.0)]
                    } else {
                        Vec::new()
                    }
                }
                ErrorModel::Proportional => {
                    if n_sigma > 0 {
                        vec![(0, f)]
                    } else {
                        Vec::new()
                    }
                }
                ErrorModel::Combined => {
                    let mut out = Vec::with_capacity(2);
                    if n_sigma > 0 {
                        out.push((0, f));
                    }
                    if n_sigma > 1 {
                        out.push((1, 1.0));
                    }
                    out
                }
            },
            ErrorSpec::PerCmt(map) | ErrorSpec::Selected { endpoints: map, .. } => {
                match map.get(&cmt) {
                    Some(ep) => match ep.error_model {
                        ErrorModel::Additive => ep
                            .sigma_idx
                            .first()
                            .copied()
                            .map(|i| vec![(i, 1.0)])
                            .unwrap_or_default(),
                        ErrorModel::Proportional => ep
                            .sigma_idx
                            .first()
                            .copied()
                            .map(|i| vec![(i, f)])
                            .unwrap_or_default(),
                        ErrorModel::Combined => {
                            let mut out = Vec::with_capacity(2);
                            if let Some(&i) = ep.sigma_idx.first() {
                                out.push((i, f));
                            }
                            if let Some(&i) = ep.sigma_idx.get(1) {
                                out.push((i, 1.0));
                            }
                            out
                        }
                    },
                    None => Vec::new(),
                }
            }
        }
    }

    /// `∂(sigma loading coefficient)/∂f` for each slot an observation loads on,
    /// parallel in shape to [`ErrorSpec::sigma_loadings`](crate::types::ErrorSpec::sigma_loadings).
    ///
    /// Every loading coefficient is affine in the prediction `f`: the
    /// proportional slot loads `f` (slope `1`), the additive slot loads the
    /// constant `1` (slope `0`). Returning the slopes with the *same slot
    /// presence* as [`ErrorSpec::sigma_loadings`](crate::types::ErrorSpec::sigma_loadings) lets the dense-`R` derivative
    /// ([`crate::stats::residual_error::compute_dr_df_matrices`]) reuse the exact
    /// bilinear cross-covariance assembly with the value loadings replaced by
    /// these slopes — the off-diagonal `R` is linear in each observation's
    /// loadings, so `∂R_jk/∂f_j = cross(slopes_j, values_k)`.
    pub fn sigma_loading_slopes(&self, cmt: usize, n_sigma: usize) -> Vec<(usize, f64)> {
        match self {
            ErrorSpec::Single(em) => match em {
                ErrorModel::Additive => {
                    if n_sigma > 0 {
                        vec![(0, 0.0)]
                    } else {
                        Vec::new()
                    }
                }
                ErrorModel::Proportional => {
                    if n_sigma > 0 {
                        vec![(0, 1.0)]
                    } else {
                        Vec::new()
                    }
                }
                ErrorModel::Combined => {
                    let mut out = Vec::with_capacity(2);
                    if n_sigma > 0 {
                        out.push((0, 1.0));
                    }
                    if n_sigma > 1 {
                        out.push((1, 0.0));
                    }
                    out
                }
            },
            ErrorSpec::PerCmt(map) | ErrorSpec::Selected { endpoints: map, .. } => {
                match map.get(&cmt) {
                    Some(ep) => match ep.error_model {
                        ErrorModel::Additive => ep
                            .sigma_idx
                            .first()
                            .copied()
                            .map(|i| vec![(i, 0.0)])
                            .unwrap_or_default(),
                        ErrorModel::Proportional => ep
                            .sigma_idx
                            .first()
                            .copied()
                            .map(|i| vec![(i, 1.0)])
                            .unwrap_or_default(),
                        ErrorModel::Combined => {
                            let mut out = Vec::with_capacity(2);
                            if let Some(&i) = ep.sigma_idx.first() {
                                out.push((i, 1.0));
                            }
                            if let Some(&i) = ep.sigma_idx.get(1) {
                                out.push((i, 0.0));
                            }
                            out
                        }
                    },
                    None => Vec::new(),
                }
            }
        }
    }

    /// Residual variance including fixed residual correlations from
    /// `block_sigma`.
    pub fn variance_at_with_correlations(
        &self,
        cmt: usize,
        f_pred: f64,
        sigma: &[f64],
        correlations: &[ResidualCorrelation],
    ) -> f64 {
        if correlations.is_empty() {
            return self.variance_at(cmt, f_pred, sigma);
        }
        // The correlated variance is exactly [`variance_at_scaled`] with an
        // all-ones multiplier: an empty `mult` makes every loading's multiplier
        // default to 1.0. Delegate so the loading + cross-covariance formula
        // lives in one place (#484 review #9) and the magnitude path can never
        // silently diverge from the bare-sigma path.
        self.variance_at_scaled(cmt, f_pred, sigma, correlations, &[])
    }

    /// Residual variance with a per-observation custom magnitude (#484).
    ///
    /// Like [`variance_at_with_correlations`] but scales each sigma loading by
    /// the per-observation multiplier `mult[idx]` (1-based on the flat sigma
    /// vector, supplied by [`crate::types::RuvMagnitude::eval_obs`]). A `mult`
    /// of all ones reproduces [`variance_at_with_correlations`] exactly. Cross
    /// terms from `block_sigma` scale by the product of the two loadings'
    /// multipliers.
    ///
    /// [`variance_at_with_correlations`]: ErrorSpec::variance_at_with_correlations
    pub fn variance_at_scaled(
        &self,
        cmt: usize,
        f_pred: f64,
        sigma: &[f64],
        correlations: &[ResidualCorrelation],
        mult: &[f64],
    ) -> f64 {
        let loadings = self.sigma_loadings(cmt, f_pred, sigma.len());
        if loadings.is_empty() {
            return f64::NAN;
        }
        let m = |idx: usize| mult.get(idx).copied().unwrap_or(1.0);
        let mut v = 0.0;
        for &(idx, coeff) in &loadings {
            let Some(&s) = sigma.get(idx) else {
                return f64::NAN;
            };
            let c = coeff * m(idx);
            v += c * c * s * s;
        }
        for corr in correlations {
            let ci = loadings
                .iter()
                .find(|(idx, _)| *idx == corr.sigma_i)
                .map(|(_, coeff)| *coeff * m(corr.sigma_i));
            let cj = loadings
                .iter()
                .find(|(idx, _)| *idx == corr.sigma_j)
                .map(|(_, coeff)| *coeff * m(corr.sigma_j));
            let (Some(ci), Some(cj)) = (ci, cj) else {
                continue;
            };
            let (Some(&si), Some(&sj)) = (sigma.get(corr.sigma_i), sigma.get(corr.sigma_j)) else {
                return f64::NAN;
            };
            v += 2.0 * ci * cj * corr.rho * si * sj;
        }
        v.max(MIN_VARIANCE)
    }

    /// `SigmaType` for each entry of the flat global sigma vector, as a `Vec`
    /// of length `n_sigma`, so `FitResult` can label/scale every sigma.
    ///
    /// `Single` stamps the error model's own sigma types into the leading
    /// slots and leaves any further declared sigmas as `Additive`. `PerCmt`
    /// stamps the type of every sigma index each endpoint owns; sigmas not
    /// referenced by any endpoint default to `Additive`. Either way the
    /// returned length always equals `n_sigma`.
    pub fn sigma_types(&self, n_sigma: usize) -> Vec<SigmaType> {
        let mut out = vec![SigmaType::Additive; n_sigma];
        match self {
            ErrorSpec::Single(em) => {
                for (i, t) in em.sigma_types().into_iter().enumerate() {
                    if i < out.len() {
                        out[i] = t;
                    }
                }
            }
            ErrorSpec::PerCmt(map) | ErrorSpec::Selected { endpoints: map, .. } => {
                for ep in map.values() {
                    let types = ep.error_model.sigma_types();
                    for (k, &idx) in ep.sigma_idx.iter().enumerate() {
                        if idx < out.len() {
                            out[idx] = types[k];
                        }
                    }
                }
            }
        }
        out
    }

    /// `d(residual variance)/d(prediction f)` for one observation at `cmt`.
    ///
    /// The score term the SAEM M-step needs alongside the variance. Additive
    /// endpoints contribute 0; proportional/combined endpoints contribute
    /// `2·f·σ_prop²` (σ_prop is the endpoint's proportional sigma, which is the
    /// first sigma for both `Proportional` and `Combined`). `Single` ignores
    /// `cmt`; `PerCmt` dispatches on the endpoint registered for `cmt`.
    pub fn dvar_df(&self, cmt: usize, f: f64, sigma: &[f64]) -> f64 {
        // Floor-aware derivative: `variance_at` clamps the raw variance to
        // `MIN_VARIANCE` (`v.max(MIN_VARIANCE)`), so on the clamped side the
        // variance is locally *constant* in `f` and its true `∂v/∂f` is 0 — not
        // the raw `2·f·σ²`. Returning the raw slope where the floor is active
        // makes every analytic path that consumes `∂R/∂f` (the inner-EBE
        // gradient's `coef`, the FOCEI Laplace `c̃` term, the outer θ-gradient)
        // disagree with the finite-difference objective, which sees the clamped
        // function. This bites a proportional-error row whose prediction is
        // driven to ~0 (drug fully eliminated) — the analytic inner gradient
        // then leads the EBE search to a different mode than FD, and the FOCEI
        // objective becomes gradient-path-dependent (#958). `variance_at`
        // returns exactly `MIN_VARIANCE` iff the raw variance was clamped; a
        // `NaN` (unregistered `PerCmt` cmt) fails the comparison and falls
        // through to the `None` arm below, which already returns 0.
        if self.variance_at(cmt, f, sigma) <= MIN_VARIANCE {
            return 0.0;
        }
        self.dvar_df_raw(cmt, f, sigma)
    }

    /// Raw (unfloored) `∂v/∂f` — the closed-form proportional slope before the
    /// `MIN_VARIANCE` floor gate. Both floored entry points share it so the
    /// closed form lives once: [`dvar_df`](Self::dvar_df) gates it on the
    /// unscaled `variance_at`, while [`dvar_df_scaled`](Self::dvar_df_scaled)
    /// gates it on the *scaled* variance it actually pairs with. Returns 0 for
    /// additive endpoints and for an unregistered `PerCmt` cmt.
    fn dvar_df_raw(&self, cmt: usize, f: f64, sigma: &[f64]) -> f64 {
        let (em, prop_sigma) = match self {
            ErrorSpec::Single(em) => (*em, sigma.first().copied().unwrap_or(0.0)),
            ErrorSpec::PerCmt(map) | ErrorSpec::Selected { endpoints: map, .. } => {
                match map.get(&cmt) {
                    Some(ep) => (
                        ep.error_model,
                        ep.sigma_idx
                            .first()
                            .and_then(|&i| sigma.get(i))
                            .copied()
                            .unwrap_or(0.0),
                    ),
                    None => return 0.0,
                }
            }
        };
        match em {
            ErrorModel::Additive => 0.0,
            ErrorModel::Proportional | ErrorModel::Combined => 2.0 * f * prop_sigma * prop_sigma,
        }
    }

    /// Flat sigma-vector index of `cmt`'s proportional sigma (the slot a
    /// magnitude multiplier scales for the `f`-derivative chain): slot `0` for
    /// `Single`, the endpoint's first `sigma_idx` for `PerCmt`. `None` when
    /// `PerCmt` has no endpoint registered for `cmt`, or that endpoint declares
    /// no sigma at all — both cases the `_scaled` derivative callers treat as
    /// "no residual error here" (`0.0`). Single source for
    /// [`dvar_df_scaled`](Self::dvar_df_scaled) and
    /// [`d2var_df2_scaled`](Self::d2var_df2_scaled) so the two can't resolve a
    /// different slot for the same `(cmt, mult)` (#486 review).
    fn prop_sigma_slot(&self, cmt: usize) -> Option<usize> {
        match self {
            ErrorSpec::Single(_) => Some(0),
            ErrorSpec::PerCmt(map) | ErrorSpec::Selected { endpoints: map, .. } => {
                map.get(&cmt)?.sigma_idx.first().copied()
            }
        }
    }

    /// `dvar_df` with a per-observation custom magnitude (#484).
    ///
    /// The proportional loading carries the multiplier `m_prop`, so the
    /// variance's `f`-dependent term is `(f·m_prop·σ_prop)²` and its `f`
    /// derivative scales by `m_prop²`. The proportional sigma is the first sigma
    /// slot for both `Proportional` and `Combined`; `mult` is indexed on the
    /// flat sigma vector. A `mult` of all ones reproduces [`dvar_df`].
    ///
    /// [`dvar_df`]: ErrorSpec::dvar_df
    pub fn dvar_df_scaled(&self, cmt: usize, f: f64, sigma: &[f64], mult: &[f64]) -> f64 {
        let Some(prop_slot) = self.prop_sigma_slot(cmt) else {
            return 0.0;
        };
        let m = mult.get(prop_slot).copied().unwrap_or(1.0);
        // Gate the floor on the *scaled* variance this derivative pairs with —
        // `variance_at_scaled` folds the magnitude `m` into the loading *before*
        // clamping, exactly as `residual_rd`/`residual_rd2` compute the R that
        // consumes this slope. Gating on the unscaled `variance_at` (via
        // `dvar_df`) puts the floor boundary at a different `f` whenever `m ≠ 1`,
        // so the objective sees a live variance while the analytic slope reads 0
        // (or vice-versa) — the #958 analytic-vs-FD gradient mismatch, on the
        // #484 magnitude path. `&[]`: correlations force FD upstream (diagonal-R
        // only here), matching the `residual_rd` call.
        if self.variance_at_scaled(cmt, f, sigma, &[], mult) <= MIN_VARIANCE {
            return 0.0;
        }
        self.dvar_df_raw(cmt, f, sigma) * m * m
    }

    /// `d²(residual variance)/d(prediction f)²` for one observation at `cmt`.
    ///
    /// Additive endpoints contribute 0 (variance is `σ_add²`, independent of f).
    /// Proportional and combined endpoints contribute `2·σ_prop²` (variance has
    /// a `f²·σ_prop²` term, so the second derivative w.r.t. f is constant).
    /// `Single` ignores `cmt`; `PerCmt` dispatches on the endpoint registered
    /// for `cmt`. Used by the Almquist Laplace FOCEI gradient's θ-axis β_j
    /// chain — keeping the per-CMT routing here lets the same closed-form
    /// gradient handle multi-endpoint models without changing the call site.
    ///
    /// Floor-aware in `f` for the same reason as [`dvar_df`](Self::dvar_df):
    /// where `variance_at` clamps the raw variance to `MIN_VARIANCE`, the
    /// clamped variance is locally constant in `f`, so `∂²v/∂f² = 0` — not the
    /// raw `2·σ²` (#958).
    pub fn d2var_df2(&self, cmt: usize, f: f64, sigma: &[f64]) -> f64 {
        if self.variance_at(cmt, f, sigma) <= MIN_VARIANCE {
            return 0.0;
        }
        self.d2var_df2_raw(cmt, sigma)
    }

    /// Raw (unfloored) `∂²v/∂f²` — the constant proportional curvature before the
    /// `MIN_VARIANCE` floor gate. Shared by [`d2var_df2`](Self::d2var_df2) (gated
    /// on the unscaled variance) and [`d2var_df2_scaled`](Self::d2var_df2_scaled)
    /// (gated on the scaled variance it pairs with). `f`-independent, so it takes
    /// no `f`. Returns 0 for additive endpoints and an unregistered `PerCmt` cmt.
    fn d2var_df2_raw(&self, cmt: usize, sigma: &[f64]) -> f64 {
        let (em, prop_sigma) = match self {
            ErrorSpec::Single(em) => (*em, sigma.first().copied().unwrap_or(0.0)),
            ErrorSpec::PerCmt(map) | ErrorSpec::Selected { endpoints: map, .. } => {
                match map.get(&cmt) {
                    Some(ep) => (
                        ep.error_model,
                        ep.sigma_idx
                            .first()
                            .and_then(|&i| sigma.get(i))
                            .copied()
                            .unwrap_or(0.0),
                    ),
                    None => return 0.0,
                }
            }
        };
        match em {
            ErrorModel::Additive => 0.0,
            ErrorModel::Proportional | ErrorModel::Combined => 2.0 * prop_sigma * prop_sigma,
        }
    }

    /// `d²(residual variance)/d(prediction f)²` with a per-observation custom
    /// magnitude (#484/#576) — the second-derivative analogue of [`dvar_df_scaled`].
    ///
    /// The proportional loading carries the multiplier `m_prop`, so the variance's
    /// `f²·(m_prop·σ_prop)²` term differentiates twice to `2·(m_prop·σ_prop)²`,
    /// i.e. [`d2var_df2`] scaled by `m_prop²` — exactly the `m²` factor
    /// [`dvar_df_scaled`] applies to `dvar_df`. Keeping the same scaling here lets
    /// the M3 censored curvature (which differentiates the same `v(f)`) stay
    /// internally consistent under a custom RUV magnitude, and the FOCEI outer
    /// θ/σ gradient's direct-θ channel (`sens_outer_gradient::prepare_stacked`)
    /// consume the same magnitude-scaled second derivative. A `mult` of all ones
    /// reproduces [`d2var_df2`].
    ///
    /// [`dvar_df_scaled`]: ErrorSpec::dvar_df_scaled
    /// [`dvar_df`]: ErrorSpec::dvar_df
    /// [`d2var_df2`]: ErrorSpec::d2var_df2
    pub fn d2var_df2_scaled(&self, cmt: usize, f: f64, sigma: &[f64], mult: &[f64]) -> f64 {
        let Some(prop_slot) = self.prop_sigma_slot(cmt) else {
            return 0.0;
        };
        let m = mult.get(prop_slot).copied().unwrap_or(1.0);
        // Floor on the scaled variance this curvature is the second derivative
        // of — same reasoning as `dvar_df_scaled`. Gating on the unscaled
        // `variance_at` (via `d2var_df2`) leaves the M3 covariance Hessian a
        // spurious `2·m²·σ²` where the scaled variance is floored constant,
        // biasing the standard errors on the #484 magnitude path.
        if self.variance_at_scaled(cmt, f, sigma, &[], mult) <= MIN_VARIANCE {
            return 0.0;
        }
        self.d2var_df2_raw(cmt, sigma) * m * m
    }

    /// `d(residual variance)/d(log σ_k)` for one observation at `cmt`, where
    /// `k` indexes the flat global sigma vector. Zero when `σ_k` does not enter
    /// this observation's endpoint, so the SAEM sigma-gradient can sum this over
    /// every observation and have each sigma pick up only its own endpoint's
    /// contributions. (`Proportional` slot → `2·σ_k²·f²`, `Additive` slot →
    /// `2·σ_k²`.)
    pub fn dvar_dlogsigma(&self, cmt: usize, k: usize, f: f64, sigma: &[f64]) -> f64 {
        let sk = match sigma.get(k) {
            Some(&s) => s,
            None => return 0.0,
        };
        let sk2 = sk * sk;
        // Resolve which `SigmaType` (if any) global index `k` plays for this
        // observation's endpoint.
        let stype = match self {
            ErrorSpec::Single(em) => em.sigma_types().get(k).copied(),
            ErrorSpec::PerCmt(map) | ErrorSpec::Selected { endpoints: map, .. } => {
                match map.get(&cmt) {
                    Some(ep) => ep
                        .sigma_idx
                        .iter()
                        .position(|&i| i == k)
                        .and_then(|p| ep.error_model.sigma_types().get(p).copied()),
                    None => None,
                }
            }
        };
        match stype {
            Some(SigmaType::Proportional) => 2.0 * sk2 * f * f,
            Some(SigmaType::Additive) => 2.0 * sk2,
            None => 0.0,
        }
    }

    /// [`dvar_dlogsigma`](Self::dvar_dlogsigma) with a per-observation custom
    /// magnitude (#484 / #1029).
    ///
    /// Slot `k`'s variance contribution is `(coeff · m_k · σ_k)²`, so its
    /// `log σ_k` derivative is the unscaled one times `m_k²` — the multiplier
    /// rides the loading, exactly as it does in
    /// [`variance_at_scaled`](Self::variance_at_scaled). A `mult` of all ones
    /// reproduces `dvar_dlogsigma`.
    pub fn dvar_dlogsigma_scaled(
        &self,
        cmt: usize,
        k: usize,
        f: f64,
        sigma: &[f64],
        mult: &[f64],
    ) -> f64 {
        let m = mult.get(k).copied().unwrap_or(1.0);
        self.dvar_dlogsigma(cmt, k, f, sigma) * m * m
    }

    /// Residual variance for one observation, dispatching on its compartment.
    ///
    /// For `Single` the `cmt` is ignored and the full `sigma` slice is used
    /// (the back-compat path). For `PerCmt` the endpoint registered for `cmt`
    /// selects the error model and slices `sigma` by its `sigma_idx`. Returns
    /// `NaN` when `cmt` has no registered endpoint, or when an endpoint's
    /// `sigma_idx` points outside `sigma` — defensive guards mirroring the
    /// scaling path. Fit-time validation rejects an uncovered CMT up front,
    /// and `build_error_spec` resolves indices against the real sigma vector,
    /// so a `NaN` here is only reachable via a hand-constructed model.
    /// Whether the residual variance depends on the prediction `f` for any
    /// endpoint (proportional or combined). When `false` (purely additive),
    /// the variance is constant in `f`, so FOCE's choice of evaluation point
    /// (linearized `f0` vs population `f(η=0)`) is irrelevant and the cheap
    /// path stays bit-identical. Used to gate the FOCE population-variance
    /// (`f(η=0)`) treatment in the marginal, analytical gradient, and
    /// covariance step.
    pub fn has_f_dependent_variance(&self) -> bool {
        match self {
            ErrorSpec::Single(em) => !matches!(em, ErrorModel::Additive),
            ErrorSpec::PerCmt(map) | ErrorSpec::Selected { endpoints: map, .. } => map
                .values()
                .any(|ep| !matches!(ep.error_model, ErrorModel::Additive)),
        }
    }

    /// Global `sigma.values` indices of the additive component of every
    /// `Combined` endpoint (the second sigma slot). De-duplicated; empty when
    /// no endpoint is combined.
    pub fn combined_additive_sigma_indices(&self) -> Vec<usize> {
        match self {
            ErrorSpec::Single(ErrorModel::Combined) => vec![1],
            ErrorSpec::Single(_) => Vec::new(),
            ErrorSpec::PerCmt(map) | ErrorSpec::Selected { endpoints: map, .. } => {
                let mut out = Vec::new();
                for endpoint in map.values() {
                    if matches!(endpoint.error_model, ErrorModel::Combined) {
                        if let Some(&idx) = endpoint.sigma_idx.get(1) {
                            if !out.contains(&idx) {
                                out.push(idx);
                            }
                        }
                    }
                }
                out
            }
        }
    }

    pub fn variance_at(&self, cmt: usize, f_pred: f64, sigma: &[f64]) -> f64 {
        match self {
            ErrorSpec::Single(em) => residual_variance(*em, f_pred, sigma),
            ErrorSpec::PerCmt(map) | ErrorSpec::Selected { endpoints: map, .. } => {
                match map.get(&cmt) {
                    Some(ep) => {
                        // Slice length is tied to the endpoint's error model
                        // (1 for additive/proportional, 2 for combined); the max
                        // is 2, so a stack buffer avoids a per-observation alloc.
                        let n = ep.error_model.n_sigma();
                        let mut buf = [0.0f64; 2];
                        for (k, slot) in buf.iter_mut().take(n.min(2)).enumerate() {
                            match ep.sigma_idx.get(k).and_then(|&i| sigma.get(i)) {
                                Some(&v) => *slot = v,
                                None => return f64::NAN, // malformed spec / sigma length
                            }
                        }
                        residual_variance(ep.error_model, f_pred, &buf[..n.min(2)])
                    }
                    None => f64::NAN,
                }
            }
        }
    }
}

/// Compute the R diagonal (vector of residual variances for all observations),
/// dispatching the error model per observation by compartment. `obs_cmts` is
/// parallel to `ipreds` (`subject.obs_cmts`); for single-endpoint models the
/// CMT is ignored.
pub fn compute_r_diag(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    sigma_values: &[f64],
) -> Vec<f64> {
    ipreds
        .iter()
        .zip(obs_cmts.iter())
        .map(|(&f, &cmt)| error_spec.variance_at(cmt, f, sigma_values))
        .collect()
}

/// [`compute_r_diag`] with the optional per-observation custom-magnitude
/// multiplier (#484 / #1029): `Some(mult)` routes each observation through
/// [`ErrorSpec::variance_at_scaled`], `None` reproduces the bare
/// [`ErrorSpec::variance_at`] association exactly (the two differ by ~1 ULP
/// under IEEE-754 reassociation, so the split is preserved, not merged).
pub fn compute_r_diag_maybe_scaled(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    sigma_values: &[f64],
    ruv_mult: Option<&[Vec<f64>]>,
) -> Vec<f64> {
    let Some(mult) = ruv_mult else {
        return compute_r_diag(error_spec, ipreds, obs_cmts, sigma_values);
    };
    ipreds
        .iter()
        .zip(obs_cmts.iter())
        .enumerate()
        .map(|(j, (&f, &cmt))| error_spec.variance_at_scaled(cmt, f, sigma_values, &[], &mult[j]))
        .collect()
}

/// Gaussian observation NLL `Σ_j ½(ln V_j + (y_j − f_j)²/V_j)` from a supplied
/// prediction vector, with the per-observation custom magnitude (#484 / #1029)
/// and the `iiv_on_ruv` variance scale folded in.
///
/// Single owner of the "score these predictions against this magnitude" formula
/// so a diagonal-`R` estimator can evaluate it at a *perturbed* parameter point
/// without duplicating it. SAEM's M-step θ gradient uses exactly that: with a
/// magnitude active it forward-differences this whole quantity — capturing both
/// the prediction channel and the magnitude's own direct-θ channel — instead of
/// chaining an analytic `∂nll/∂f` that would silently drop the second one.
///
/// `frem_var[j] = Some(v)` overrides observation `j`'s variance outright (FREM
/// covariate pseudo-observations use `EPSCOV²`, not the PK residual error, and
/// carry neither the magnitude nor `ruv_scale`). Predictions and variances are
/// floored exactly as the callers' inline loops do, so this is a drop-in for
/// them. Censored (M3) rows are **not** handled — callers with `BloqMethod::M3`
/// take their own path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gaussian_obs_nll_scaled(
    error_spec: &ErrorSpec,
    err_keys: &[usize],
    observations: &[f64],
    preds: &[f64],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
    ruv_scale: f64,
    frem_var: Option<&[Option<f64>]>,
    ruv_mult: Option<&[Vec<f64>]>,
) -> f64 {
    let mut nll = 0.0_f64;
    for (j, (&y, &f_raw)) in observations.iter().zip(preds.iter()).enumerate() {
        let f = f_raw.max(MIN_VARIANCE);
        let frem_vj = frem_var.and_then(|o| o.get(j)).and_then(|x| *x);
        let v = match frem_vj {
            Some(vv) => vv.max(MIN_VARIANCE),
            None => {
                // Same dispatch as `CompiledModel::residual_variance_at_scaled`,
                // correlations and all, so a caller's base NLL and the perturbed
                // one it differences against are built by identical arithmetic.
                let raw = match ruv_mult {
                    Some(m) => error_spec.variance_at_scaled(
                        err_keys[j],
                        f,
                        sigma_values,
                        correlations,
                        &m[j],
                    ),
                    None => error_spec.variance_at(err_keys[j], f, sigma_values),
                };
                (raw * ruv_scale).max(MIN_VARIANCE)
            }
        };
        let resid = y - f;
        nll += 0.5 * (v.ln() + resid * resid / v);
    }
    nll
}

/// Compute residual variances including fixed residual correlations from
/// `block_sigma`.
pub fn compute_r_diag_with_correlations(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> Vec<f64> {
    if correlations.is_empty() {
        return compute_r_diag(error_spec, ipreds, obs_cmts, sigma_values);
    }
    ipreds
        .iter()
        .zip(obs_cmts.iter())
        .map(|(&f, &cmt)| {
            error_spec.variance_at_with_correlations(cmt, f, sigma_values, correlations)
        })
        .collect()
}

fn observation_time_key(obs_times: &[f64], j: usize) -> u64 {
    obs_times.get(j).copied().unwrap_or(0.0).to_bits()
}

fn observation_occasion_key(occasions: &[u32], j: usize) -> u32 {
    occasions.get(j).copied().unwrap_or(0)
}

/// Do observations `j` and `k` belong to the same correlated residual block?
///
/// When the data carries an `L2` grouping id (NONMEM's level-2 data item), it is
/// authoritative: a row with a nonzero `L2` pairs **only** with rows sharing that
/// id, so the user controls exactly which records form one correlated
/// observation unit (total/unbound of a draw, replicate assays, any paired
/// endpoints). Rows with no `L2` id (`0`, or an empty `obs_l2` when the dataset
/// has no `L2` column) fall back to the implicit `(time, occasion)` rule.
fn same_residual_block(
    obs_times: &[f64],
    _obs_raw_times: &[f64],
    occasions: &[u32],
    obs_l2: &[i64],
    j: usize,
    k: usize,
) -> bool {
    let l2j = obs_l2.get(j).copied().unwrap_or(0);
    let l2k = obs_l2.get(k).copied().unwrap_or(0);
    if l2j != 0 || l2k != 0 {
        // At least one row is explicitly grouped: pair iff same L2 id.
        return l2j == l2k;
    }
    observation_time_key(obs_times, j) == observation_time_key(obs_times, k)
        && observation_occasion_key(occasions, j) == observation_occasion_key(occasions, k)
}

fn cross_observation_covariance(
    load_j: &[(usize, f64)],
    load_k: &[(usize, f64)],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> f64 {
    let mut cov = 0.0;
    for corr in correlations {
        let j_has_i = load_j.iter().any(|(idx, _)| *idx == corr.sigma_i);
        let j_has_j = load_j.iter().any(|(idx, _)| *idx == corr.sigma_j);
        let k_has_i = load_k.iter().any(|(idx, _)| *idx == corr.sigma_i);
        let k_has_j = load_k.iter().any(|(idx, _)| *idx == corr.sigma_j);
        if (j_has_i && j_has_j) || (k_has_i && k_has_j) {
            continue;
        }
        let Some(&si) = sigma_values.get(corr.sigma_i) else {
            return f64::NAN;
        };
        let Some(&sj) = sigma_values.get(corr.sigma_j) else {
            return f64::NAN;
        };
        let cov_ij = corr.rho * si * sj;
        let ci_j = load_j
            .iter()
            .find(|(idx, _)| *idx == corr.sigma_i)
            .map(|(_, coeff)| *coeff);
        let cj_k = load_k
            .iter()
            .find(|(idx, _)| *idx == corr.sigma_j)
            .map(|(_, coeff)| *coeff);
        if let (Some(ci), Some(cj)) = (ci_j, cj_k) {
            cov += ci * cj * cov_ij;
        }

        let cj_j = load_j
            .iter()
            .find(|(idx, _)| *idx == corr.sigma_j)
            .map(|(_, coeff)| *coeff);
        let ci_k = load_k
            .iter()
            .find(|(idx, _)| *idx == corr.sigma_i)
            .map(|(_, coeff)| *coeff);
        if let (Some(cj), Some(ci)) = (cj_j, ci_k) {
            cov += cj * ci * cov_ij;
        }
    }
    cov
}

/// Do observations `j` and `k` share at least one `block_sigma` cross
/// correlation, judged purely from which sigma slots their loadings occupy?
///
/// This mirrors the slot bookkeeping of [`cross_observation_covariance`]
/// (identical within-observation skip) but ignores the loading *coefficients*,
/// so the answer is independent of the prediction values. A pure proportional
/// row at `f = 0` keeps its slot (as `(idx, 0.0)`), so it still pairs with its
/// partner and the derivative builders keep emitting the nonzero slope-loading
/// cross term.
pub(crate) fn loadings_share_correlation(
    load_j: &[(usize, f64)],
    load_k: &[(usize, f64)],
    correlations: &[ResidualCorrelation],
) -> bool {
    correlations.iter().any(|corr| {
        let j_has_i = load_j.iter().any(|(idx, _)| *idx == corr.sigma_i);
        let j_has_j = load_j.iter().any(|(idx, _)| *idx == corr.sigma_j);
        let k_has_i = load_k.iter().any(|(idx, _)| *idx == corr.sigma_i);
        let k_has_j = load_k.iter().any(|(idx, _)| *idx == corr.sigma_j);
        // A row carrying both slots owns the correlation as a within-observation
        // term, not a cross-observation one (matches `cross_observation_covariance`).
        if (j_has_i && j_has_j) || (k_has_i && k_has_j) {
            return false;
        }
        (j_has_i && k_has_j) || (j_has_j && k_has_i)
    })
}

/// Match observations into cross-covariance pairs within each residual block.
///
/// [`same_residual_block`] decides which rows may pair, and the grouping source
/// sets the pairing rule:
///
/// * **Explicit `L2` group** — the rows are one user-declared correlated unit,
///   so **every** complementary pair in the group is correlated (all-to-all). A
///   genuine 3+ endpoint block (e.g. parent + two metabolites, each pair
///   correlated and jointly positive-definite) keeps its full cross-covariance
///   structure. Over-grouping true replicates into one L2 unit would make that
///   block indefinite and fail `R.cholesky()` loudly — which is the user telling
///   the fit their grouping is wrong, not a silent drop.
/// * **`(time, occasion)` fallback** — replicate assays cannot be told apart, so
///   partners are matched **one-to-one, greedily in row order**: each unmatched
///   row takes the first later unmatched row in its block with a nonzero cross
///   covariance. This yields disjoint 2×2 pairs, so a co-temporal 4-row
///   replicate block stays PD instead of collapsing to the invalid sentinel
///   (issue #827).
///
/// Returns `(j, k, cov)` triples with `j < k`. `cov` is the value-loading cross
/// covariance the two `R` builders reuse directly (avoiding a recompute); the
/// derivative builders take only the `(j, k)` pairing and recompute with slope
/// loadings. `loadings[j]` are observation `j`'s sigma value loadings (slot
/// presence is identical for slope loadings, so the same pairing drives the
/// derivative builders).
fn match_partners(
    loadings: &[Vec<(usize, f64)>],
    obs_times: &[f64],
    obs_raw_times: &[f64],
    occasions: &[u32],
    obs_l2: &[i64],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> Vec<(usize, usize, f64)> {
    let n = loadings.len();
    let mut pairs = Vec::new();
    // Fallback (no L2) pairing is one-to-one: once a row is matched it is
    // consumed and starts no further pairs. Explicit L2 rows are never consumed,
    // so each pairs with every complementary row in its group (all-to-all).
    let mut consumed = vec![false; n];
    for j in 0..n {
        if consumed[j] || loadings[j].is_empty() {
            continue;
        }
        for k in (j + 1)..n {
            if consumed[k]
                || loadings[k].is_empty()
                || !same_residual_block(obs_times, obs_raw_times, occasions, obs_l2, j, k)
            {
                continue;
            }
            // Gate the pairing on the *structural* slot overlap, not the value
            // covariance: a pure proportional row whose prediction is momentarily
            // `f = 0` has a zero value-loading covariance but a nonzero slope
            // (derivative) cross term. Deciding on `cov != 0.0` would drop the
            // pair — and with it that derivative term — exactly at `f ≈ 0`, and
            // would let the pairing flicker as `f` crosses 0 across iterations.
            if !loadings_share_correlation(&loadings[j], &loadings[k], correlations) {
                continue;
            }
            let cov = cross_observation_covariance(
                &loadings[j],
                &loadings[k],
                sigma_values,
                correlations,
            );
            pairs.push((j, k, cov));
            let explicit = obs_l2.get(j).copied().unwrap_or(0) != 0
                && obs_l2.get(k).copied().unwrap_or(0) != 0;
            if !explicit {
                // (time, occasion) fallback: consume both rows and stop scanning
                // for more partners of `j` — disjoint one-to-one pairing.
                consumed[j] = true;
                consumed[k] = true;
                break;
            }
            // Explicit L2 group: keep scanning so `j` pairs with every
            // complementary row in its group (all-to-all).
        }
    }
    pairs
}

/// Write the off-diagonal cross-covariance entries of `R` from the matched pairs
/// produced by [`match_partners`]. Shared by both `R` builders so the
/// pairing-to-matrix fill lives in one place.
fn fill_cross_covariances(r: &mut DMatrix<f64>, pairs: &[(usize, usize, f64)]) {
    for &(j, k, cov) in pairs {
        r[(j, k)] = cov;
        r[(k, j)] = cov;
    }
}

/// Build the subject-level residual covariance matrix `R`.
///
/// The diagonal is the existing per-observation residual variance, including
/// within-observation `block_sigma` terms for `combined(...)` endpoints. When a
/// `block_sigma` off-diagonal connects sigmas used by different endpoint rows,
/// rows at the same subject time and occasion receive the corresponding
/// cross-observation covariance. This mirrors NONMEM-style paired endpoint
/// records such as total/unbound assays written as separate rows.
#[allow(clippy::too_many_arguments)]
pub fn compute_r_matrix_with_correlations(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    obs_times: &[f64],
    obs_raw_times: &[f64],
    occasions: &[u32],
    obs_l2: &[i64],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> DMatrix<f64> {
    // NOTE: deliberately NOT delegated to `_scaled` with an empty multiplier.
    // The diagonal here goes through `compute_r_diag` → `residual_variance`,
    // which forms the proportional variance as `(f·σ)·(f·σ)`; `variance_at_scaled`
    // (the `_scaled` diagonal) forms it as `((f·f)·σ)·σ`. The two are equal in
    // exact arithmetic but differ by ~1 ULP under IEEE-754 reassociation on the
    // f-dependent term (~55% of proportional/combined rows), so delegating would
    // silently shift the bare-sigma R — and every OFV/CWRES built on it — off
    // its current bit-for-bit value. Keep the legacy diagonal form here. (The
    // magnitude path already uses the `_scaled` association by construction.)
    let n = ipreds.len();
    let mut r = DMatrix::<f64>::zeros(n, n);
    // Write the diagonal variance directly into `r` (mirrors
    // `compute_r_diag_with_correlations`) rather than building a throwaway Vec
    // and copying it in — this runs once per subject per FOCE inner/outer
    // iteration. The empty-correlation case delegates to `variance_at`, the
    // legacy `(f·σ)·(f·σ)` association the comment above relies on.
    if correlations.is_empty() {
        for (j, (&f, &cmt)) in ipreds.iter().zip(obs_cmts.iter()).enumerate() {
            r[(j, j)] = error_spec.variance_at(cmt, f, sigma_values);
        }
        return r;
    }
    for (j, (&f, &cmt)) in ipreds.iter().zip(obs_cmts.iter()).enumerate() {
        r[(j, j)] = error_spec.variance_at_with_correlations(cmt, f, sigma_values, correlations);
    }

    let loadings: Vec<Vec<(usize, f64)>> = ipreds
        .iter()
        .zip(obs_cmts.iter())
        .map(|(&f, &cmt)| error_spec.sigma_loadings(cmt, f, sigma_values.len()))
        .collect();
    let pairs = match_partners(
        &loadings,
        obs_times,
        obs_raw_times,
        occasions,
        obs_l2,
        sigma_values,
        correlations,
    );
    fill_cross_covariances(&mut r, &pairs);
    r
}

/// Build the subject residual covariance matrix `R` with a per-observation
/// custom magnitude (#484). `mult` is the `[obs][sigma-slot]` multiplier matrix
/// from [`crate::types::RuvMagnitude::eval_obs`]; each observation's sigma
/// loadings are scaled by its row before forming the diagonal variance and any
/// `block_sigma` cross-covariance. NOTE: a `mult` whose rows are all ones does
/// **not** reproduce [`compute_r_matrix_with_correlations`] bit-for-bit. The
/// bare path forms the diagonal as `(f·σ)·(f·σ)` whereas the scaled path uses
/// `variance_at_scaled`'s reassociated `((f·f)·σ)·σ` form, which differs by
/// ~1 ULP on ~55% of proportional/combined rows. The two diagonal builders are
/// kept separate deliberately — do NOT re-collapse the bare path into
/// `_scaled(..., &[])` (that is the exact regression this revert prevents).
#[allow(clippy::too_many_arguments)]
pub fn compute_r_matrix_with_correlations_scaled(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    obs_times: &[f64],
    obs_raw_times: &[f64],
    occasions: &[u32],
    obs_l2: &[i64],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
    mult: &[Vec<f64>],
) -> DMatrix<f64> {
    let n = ipreds.len();
    let mut r = DMatrix::<f64>::zeros(n, n);
    let ones: Vec<f64> = Vec::new();
    let row = |j: usize| -> &[f64] { mult.get(j).map(|v| v.as_slice()).unwrap_or(&ones) };
    for j in 0..n {
        let f = ipreds[j];
        let cmt = obs_cmts.get(j).copied().unwrap_or(0);
        r[(j, j)] = error_spec.variance_at_scaled(cmt, f, sigma_values, correlations, row(j));
    }
    if correlations.is_empty() {
        return r;
    }
    // Scale each observation's loadings by its multiplier row before computing
    // the cross-observation covariance.
    let scale_loadings = |j: usize| -> Vec<(usize, f64)> {
        let f = ipreds[j];
        let cmt = obs_cmts.get(j).copied().unwrap_or(0);
        let m = row(j);
        error_spec
            .sigma_loadings(cmt, f, sigma_values.len())
            .into_iter()
            .map(|(idx, coeff)| (idx, coeff * m.get(idx).copied().unwrap_or(1.0)))
            .collect()
    };
    let loadings: Vec<Vec<(usize, f64)>> = (0..n).map(scale_loadings).collect();
    let pairs = match_partners(
        &loadings,
        obs_times,
        obs_raw_times,
        occasions,
        obs_l2,
        sigma_values,
        correlations,
    );
    fill_cross_covariances(&mut r, &pairs);
    r
}

/// Dense residual covariance `R`, dispatching on the optional per-observation
/// custom-magnitude multiplier: `Some(mult)` routes through the `_scaled`
/// association ([`compute_r_matrix_with_correlations_scaled`]) and `None` through
/// the bare ([`compute_r_matrix_with_correlations`]) diagonal form. The two are
/// equal in exact arithmetic but differ by ~1 ULP under IEEE-754 reassociation,
/// so the split is preserved (not merged). `obs_times`/`obs_raw_times`/
/// `occasions`/`obs_l2` are pulled off `subject`. Folds the identical `match`
/// that the FOCE/FOCEI/data-term/CWRES paths each carried.
pub(crate) fn r_matrix_maybe_scaled(
    error_spec: &ErrorSpec,
    preds: &[f64],
    err_keys: &[usize],
    subject: &Subject,
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
    ruv_mult: Option<&[Vec<f64>]>,
) -> DMatrix<f64> {
    match ruv_mult {
        Some(mult) => compute_r_matrix_with_correlations_scaled(
            error_spec,
            preds,
            err_keys,
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            &subject.obs_l2,
            sigma_values,
            correlations,
            mult,
        ),
        None => compute_r_matrix_with_correlations(
            error_spec,
            preds,
            err_keys,
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            &subject.obs_l2,
            sigma_values,
            correlations,
        ),
    }
}

/// `∂R/∂f_m` matrices of the dense residual covariance built by
/// [`compute_r_matrix_with_correlations`] (or its `_scaled` variant) — one
/// symmetric `n×n` matrix per observation `m`.
///
/// `dr[m]` is nonzero only in row/column `m`, because every entry of `R`
/// depends on the prediction vector only through the two observations it
/// couples, and each sigma loading coefficient is affine in `f` (proportional
/// slot loads `f`, additive slot a constant). So:
///
/// * diagonal: `∂R_mm/∂f_m` — the `f`-derivative of `variance_at_scaled`,
///   including any within-observation `block_sigma` cross term;
/// * off-diagonal `(m,k)` in the same residual block: `∂R_mk/∂f_m`, which —
///   because `cross_observation_covariance` is bilinear in the two
///   observations' loadings — is exactly that cross-covariance evaluated with
///   observation `m`'s *slope* loadings ([`ErrorSpec::sigma_loading_slopes`])
///   in place of its value loadings.
///
/// Feeds the FOCEI interaction Hessian term
/// `B_kl = tr(R⁻¹ ∂R/∂η_k R⁻¹ ∂R/∂η_l)` with `∂R/∂η_k = Σ_m H[m,k]·dr[m]`.
/// `mult` is the #484 per-observation magnitude matrix (`None` ⇒ all ones),
/// applied identically to the value and slope loadings so the derivative tracks
/// the magnitude-scaled `R`.
#[allow(clippy::too_many_arguments)]
pub fn compute_dr_df_matrices(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    obs_times: &[f64],
    obs_raw_times: &[f64],
    occasions: &[u32],
    obs_l2: &[i64],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
    mult: Option<&[Vec<f64>]>,
) -> Vec<DMatrix<f64>> {
    let n = ipreds.len();
    let empty: Vec<f64> = Vec::new();
    let mrow = |j: usize| -> &[f64] {
        mult.and_then(|m| m.get(j))
            .map(|v| v.as_slice())
            .unwrap_or(&empty)
    };
    let m_at = |j: usize, idx: usize| -> f64 { mrow(j).get(idx).copied().unwrap_or(1.0) };
    let cmt_at = |j: usize| -> usize { obs_cmts.get(j).copied().unwrap_or(0) };

    // Per-observation value loadings (coeff·mult) and slope loadings
    // (∂coeff/∂f·mult). The slope loadings carry the SAME slot presence as the
    // value loadings (additive slots appear with slope 0), so the bilinear
    // cross-covariance and its within-observation skip logic behave identically.
    let vload: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|j| {
            error_spec
                .sigma_loadings(cmt_at(j), ipreds[j], sigma_values.len())
                .into_iter()
                .map(|(idx, c)| (idx, c * m_at(j, idx)))
                .collect()
        })
        .collect();
    let sload: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|j| {
            error_spec
                .sigma_loading_slopes(cmt_at(j), sigma_values.len())
                .into_iter()
                .map(|(idx, s)| (idx, s * m_at(j, idx)))
                .collect()
        })
        .collect();

    let pairs = match_partners(
        &vload,
        obs_times,
        obs_raw_times,
        occasions,
        obs_l2,
        sigma_values,
        correlations,
    );
    let mut out = vec![DMatrix::<f64>::zeros(n, n); n];
    for m in 0..n {
        if vload[m].is_empty() {
            continue;
        }
        out[m][(m, m)] = diag_self_deriv(&vload[m], &sload[m], sigma_values, correlations);
    }
    // ∂R_jk/∂f_j for each matched pair (mirrors the pairing that builds `R`):
    // slope loadings of the differentiated row, value loadings of its partner.
    // Each row picks up one cross term per partner (all-to-all within an L2
    // group), so `dr[j]` and `dr[k]` carry independent slopes.
    for &(j, k, _) in &pairs {
        let djk = cross_observation_covariance(&sload[j], &vload[k], sigma_values, correlations);
        out[j][(j, k)] = djk;
        out[j][(k, j)] = djk;
        let dkj = cross_observation_covariance(&sload[k], &vload[j], sigma_values, correlations);
        out[k][(k, j)] = dkj;
        out[k][(j, k)] = dkj;
    }
    out
}

/// `∂R_mm/∂f_m`: the `f`-derivative of the diagonal residual variance
/// [`ErrorSpec::variance_at_scaled`], built from observation `m`'s value
/// loadings and their `f`-slopes (both already magnitude-scaled).
///
/// `V_mm = Σ_s (c_s σ_s)² + Σ_corr 2 c_i c_j ρ σ_i σ_j` (the within-observation
/// `block_sigma` cross term), so with `c_s' = slope_s`:
/// `∂V_mm/∂f = Σ_s 2 c_s c_s' σ_s² + Σ_corr 2 ρ σ_i σ_j (c_i' c_j + c_i c_j')`.
/// The `.max(1e-12)` variance floor is treated as inactive here, matching the
/// diagonal interaction path's [`ErrorSpec::dvar_df`].
fn diag_self_deriv(
    vload: &[(usize, f64)],
    sload: &[(usize, f64)],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> f64 {
    let coeff = |loads: &[(usize, f64)], slot: usize| -> Option<f64> {
        loads.iter().find(|(i, _)| *i == slot).map(|(_, c)| *c)
    };
    let sig = |idx: usize| -> f64 { sigma_values.get(idx).copied().unwrap_or(0.0) };
    let mut d = 0.0;
    for &(idx, c) in vload {
        let s = coeff(sload, idx).unwrap_or(0.0);
        let sg = sig(idx);
        d += 2.0 * c * s * sg * sg;
    }
    for corr in correlations {
        let (Some(ci), Some(cj)) = (coeff(vload, corr.sigma_i), coeff(vload, corr.sigma_j)) else {
            continue;
        };
        let si_s = coeff(sload, corr.sigma_i).unwrap_or(0.0);
        let sj_s = coeff(sload, corr.sigma_j).unwrap_or(0.0);
        d += 2.0 * corr.rho * sig(corr.sigma_i) * sig(corr.sigma_j) * (si_s * cj + ci * sj_s);
    }
    d
}

/// Second-order `∂²R/∂f_a∂f_b` tensor of the dense residual covariance — the
/// curvature companion to [`compute_dr_df_matrices`], returned as `d2r[a][b]`,
/// an `n×n` symmetric matrix per ordered prediction pair `(a, b)`
/// (`d2r[a][b] == d2r[b][a]` by equality of mixed partials).
///
/// Every sigma loading coefficient is *affine* in `f` (proportional slot loads
/// `f`, additive slot is constant), so the second `f`-derivative of any loading
/// is zero. Combined with the fact that each `R` entry couples at most two
/// observations, only two families of second derivatives survive:
///
/// * `d2r[m][m]` has a single nonzero entry at `(m, m)`:
///   `∂²R_mm/∂f_m²` — the second `f`-derivative of `variance_at_scaled`. With
///   `c_s` the value loading, `c_s' = slope_s`, `c_s'' = 0`, the within-obs
///   `block_sigma` cross term `2 ρ σ_i σ_j c_i c_j` differentiates twice to
///   `4 ρ σ_i σ_j c_i' c_j'` (see `diag_self_second_deriv`). The off-diagonal
///   entries of `R` have `∂²R_mk/∂f_m² = cross(c_m'', c_k) = 0`, so they do not
///   appear here.
/// * `d2r[m][k]` for `m ≠ k` in the same residual block has nonzero entries at
///   `(m, k)` and `(k, m)`: `∂²R_mk/∂f_m∂f_k`. Because
///   `cross_observation_covariance` is bilinear in the two observations'
///   loadings, this mixed partial is exactly that cross-covariance evaluated
///   with *both* observations' slope loadings — `cross(slope_m, slope_k)`.
///
/// Feeds the dense FOCEI outer-gradient curvature coefficients (the `β`/`α'`
/// reservoir in `sens_outer_gradient`) and the inner Hessian response
/// correction, via `∂²R/∂η_k∂η_l = Σ_{m,m'} H[m,k] H[m',l] · d2r[m][m']`
/// (plus the `Σ_m (∂²f_m/∂η_k∂η_l) · ∂R/∂f_m` term carried by the first-order
/// [`compute_dr_df_matrices`]). `mult` is the #484 per-observation magnitude
/// matrix, applied to the slope loadings identically to the first-order path.
#[allow(clippy::too_many_arguments)]
pub fn compute_d2r_df2_matrices(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    obs_times: &[f64],
    obs_raw_times: &[f64],
    occasions: &[u32],
    obs_l2: &[i64],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
    mult: Option<&[Vec<f64>]>,
) -> Vec<Vec<DMatrix<f64>>> {
    let n = ipreds.len();
    let empty: Vec<f64> = Vec::new();
    let mrow = |j: usize| -> &[f64] {
        mult.and_then(|m| m.get(j))
            .map(|v| v.as_slice())
            .unwrap_or(&empty)
    };
    let m_at = |j: usize, idx: usize| -> f64 { mrow(j).get(idx).copied().unwrap_or(1.0) };
    let cmt_at = |j: usize| -> usize { obs_cmts.get(j).copied().unwrap_or(0) };

    // Value loadings gate the emptiness skip exactly as `compute_dr_df_matrices`
    // (an observation with no sigma loadings contributes nothing); slope
    // loadings carry the math (the only nonzero second derivatives).
    let vload: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|j| {
            error_spec
                .sigma_loadings(cmt_at(j), ipreds[j], sigma_values.len())
                .into_iter()
                .map(|(idx, c)| (idx, c * m_at(j, idx)))
                .collect()
        })
        .collect();
    let sload: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|j| {
            error_spec
                .sigma_loading_slopes(cmt_at(j), sigma_values.len())
                .into_iter()
                .map(|(idx, s)| (idx, s * m_at(j, idx)))
                .collect()
        })
        .collect();

    let pairs = match_partners(
        &vload,
        obs_times,
        obs_raw_times,
        occasions,
        obs_l2,
        sigma_values,
        correlations,
    );
    let mut out = vec![vec![DMatrix::<f64>::zeros(n, n); n]; n];
    for m in 0..n {
        if vload[m].is_empty() {
            continue;
        }
        // Diagonal curvature ∂²R_mm/∂f_m².
        out[m][m][(m, m)] = diag_self_second_deriv(&sload[m], sigma_values, correlations);
    }
    // Mixed partial ∂²R_jk/∂f_j∂f_k for each matched pair (both slope loadings).
    for &(j, k, _) in &pairs {
        let d = cross_observation_covariance(&sload[j], &sload[k], sigma_values, correlations);
        if d != 0.0 {
            out[j][k][(j, k)] = d;
            out[j][k][(k, j)] = d;
            out[k][j][(j, k)] = d;
            out[k][j][(k, j)] = d;
        }
    }
    out
}

/// `∂²R_mm/∂f_m²`: the second `f`-derivative of the diagonal residual variance.
/// With value loadings affine in `f` (`c_s'' = 0`),
/// `∂²V_mm/∂f² = Σ_s 2 (c_s')² σ_s² + Σ_corr 4 ρ σ_i σ_j c_i' c_j'`,
/// i.e. the slope-only companion of [`diag_self_deriv`]. The slope loadings
/// carry the same slot presence as the value loadings (additive slots appear
/// with slope 0), so iterating them is sufficient.
fn diag_self_second_deriv(
    sload: &[(usize, f64)],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> f64 {
    let slope = |slot: usize| -> f64 {
        sload
            .iter()
            .find(|(i, _)| *i == slot)
            .map(|(_, s)| *s)
            .unwrap_or(0.0)
    };
    let sig = |idx: usize| -> f64 { sigma_values.get(idx).copied().unwrap_or(0.0) };
    let mut d = 0.0;
    for &(idx, s) in sload {
        let sg = sig(idx);
        d += 2.0 * s * s * sg * sg;
    }
    for corr in correlations {
        let si_s = slope(corr.sigma_i);
        let sj_s = slope(corr.sigma_j);
        d += 4.0 * corr.rho * sig(corr.sigma_i) * sig(corr.sigma_j) * si_s * sj_s;
    }
    d
}

/// Individual weighted residual: IWRES_j = (y_j - f_j) / sqrt(V_j)
pub fn iwres(obs: f64, ipred: f64, error_model: ErrorModel, sigma_values: &[f64]) -> f64 {
    let v = residual_variance(error_model, ipred, sigma_values);
    (obs - ipred) / v.sqrt()
}

/// Compute IWRES for all observations, dispatching the error model per
/// observation by compartment (`obs_cmts` parallel to `observations`/`ipreds`).
pub fn compute_iwres(
    observations: &[f64],
    ipreds: &[f64],
    obs_cmts: &[usize],
    error_spec: &ErrorSpec,
    sigma_values: &[f64],
) -> Vec<f64> {
    observations
        .iter()
        .zip(ipreds.iter())
        .zip(obs_cmts.iter())
        .map(|((&y, &f), &cmt)| {
            let v = error_spec.variance_at(cmt, f, sigma_values);
            (y - f) / v.sqrt()
        })
        .collect()
}

/// Compute IWRES using residual variances that include fixed `block_sigma`
/// correlations. With no correlations and no custom magnitude this is exactly
/// [`compute_iwres`].
///
/// `ruv_mult` is the per-observation custom-magnitude multiplier matrix (#484)
/// from [`crate::types::CompiledModel::ruv_obs_mult`]; `None` reproduces the
/// legacy unscaled IWRES. When present, each observation's residual variance is
/// scaled by its multiplier row so the sdtab IWRES matches the magnitude-aware
/// OFV variance (otherwise late/covariate-varying rows are systematically
/// mis-scaled).
pub fn compute_iwres_with_correlations(
    observations: &[f64],
    ipreds: &[f64],
    obs_cmts: &[usize],
    error_spec: &ErrorSpec,
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
    ruv_mult: Option<&[Vec<f64>]>,
) -> Vec<f64> {
    if let Some(mult) = ruv_mult {
        // variance_at_scaled handles empty `correlations` (no cross terms) and a
        // short/empty multiplier row (slots default to 1.0), so this one path
        // covers correlated and uncorrelated custom-magnitude models alike.
        return observations
            .iter()
            .zip(ipreds.iter())
            .zip(obs_cmts.iter())
            .enumerate()
            .map(|(j, ((&y, &f), &cmt))| {
                let m = mult.get(j).map(|v| v.as_slice()).unwrap_or(&[]);
                let v = error_spec.variance_at_scaled(cmt, f, sigma_values, correlations, m);
                (y - f) / v.sqrt()
            })
            .collect();
    }
    if correlations.is_empty() {
        return compute_iwres(observations, ipreds, obs_cmts, error_spec, sigma_values);
    }
    observations
        .iter()
        .zip(ipreds.iter())
        .zip(obs_cmts.iter())
        .map(|((&y, &f), &cmt)| {
            let v = error_spec.variance_at_with_correlations(cmt, f, sigma_values, correlations);
            (y - f) / v.sqrt()
        })
        .collect()
}

/// Compute pooled lag-1 autocorrelation diagnostics on IWRES across subjects.
///
/// Subjects with fewer than 2 finite IWRES values are skipped.
/// Returns `(lag1_r, durbin_watson)` where DW = 2.0 indicates no autocorrelation.
/// Returns `(f64::NAN, f64::NAN)` when no subject has enough observations.
pub fn iwres_autocorrelation(subjects: &[SubjectResult]) -> (f64, f64) {
    // Accumulators for Durbin-Watson: Σ(eᵢ - eᵢ₋₁)², Σeᵢ²
    let mut dw_num = 0.0_f64;
    let mut dw_den = 0.0_f64;

    // Accumulators for pooled lag-1 Pearson r
    let mut sum_xy = 0.0_f64; // Σ e[t] * e[t+1]
    let mut sum_x = 0.0_f64; // Σ e[t]
    let mut sum_y = 0.0_f64; // Σ e[t+1]
    let mut sum_x2 = 0.0_f64; // Σ e[t]²
    let mut sum_y2 = 0.0_f64; // Σ e[t+1]²
    let mut n_pairs: usize = 0;

    for subj in subjects {
        let valid: Vec<f64> = subj
            .iwres
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        if valid.len() < 2 {
            continue;
        }
        // DW accumulation
        dw_den += valid.iter().map(|e| e * e).sum::<f64>();
        for w in valid.windows(2) {
            let diff = w[1] - w[0];
            dw_num += diff * diff;
        }
        // Lag-1 Pearson accumulation
        for w in valid.windows(2) {
            let x = w[0];
            let y = w[1];
            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            sum_y2 += y * y;
            sum_xy += x * y;
            n_pairs += 1;
        }
    }

    if n_pairs == 0 {
        return (f64::NAN, f64::NAN);
    }

    let n = n_pairs as f64;
    let lag1_r = {
        let num = n * sum_xy - sum_x * sum_y;
        let den = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        if den == 0.0 {
            0.0
        } else {
            num / den
        }
    };

    let dw = if dw_den == 0.0 {
        f64::NAN
    } else {
        dw_num / dw_den
    };

    (lag1_r, dw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EndpointError, GradientMethod};
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    #[test]
    fn test_additive_variance() {
        let v = residual_variance(ErrorModel::Additive, 10.0, &[0.5]);
        assert_relative_eq!(v, 0.25, epsilon = 1e-12);
    }

    #[test]
    fn test_additive_variance_independent_of_prediction() {
        let v1 = residual_variance(ErrorModel::Additive, 1.0, &[0.5]);
        let v2 = residual_variance(ErrorModel::Additive, 100.0, &[0.5]);
        assert_relative_eq!(v1, v2, epsilon = 1e-12);
    }

    #[test]
    fn test_proportional_variance() {
        // V = (f * sigma)^2 = (10 * 0.1)^2 = 1.0
        let v = residual_variance(ErrorModel::Proportional, 10.0, &[0.1]);
        assert_relative_eq!(v, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_proportional_variance_scales_with_prediction() {
        let v1 = residual_variance(ErrorModel::Proportional, 10.0, &[0.1]);
        let v2 = residual_variance(ErrorModel::Proportional, 20.0, &[0.1]);
        assert_relative_eq!(v2 / v1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_combined_variance() {
        // V = (f * sigma1)^2 + sigma2^2 = (10 * 0.1)^2 + 0.5^2 = 1.0 + 0.25 = 1.25
        let v = residual_variance(ErrorModel::Combined, 10.0, &[0.1, 0.5]);
        assert_relative_eq!(v, 1.25, epsilon = 1e-12);
    }

    #[test]
    fn test_combined_variance_with_residual_correlation() {
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        // V = (10 * 0.2)^2 + 1^2 + 2 * 10 * 0.5 * 0.2 * 1 = 7.
        let v = spec.variance_at_with_correlations(1, 10.0, &[0.2, 1.0], &[corr]);
        assert_relative_eq!(v, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_r_diag_with_correlations_empty_matches_diagonal() {
        // With no correlations the helper must be identical to compute_r_diag.
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let ipreds = [10.0, 20.0];
        let cmts = [0usize, 0];
        let sigma = [0.2, 1.0];
        let plain = compute_r_diag(&spec, &ipreds, &cmts, &sigma);
        let with = compute_r_diag_with_correlations(&spec, &ipreds, &cmts, &sigma, &[]);
        assert_eq!(plain, with);
    }

    #[test]
    fn test_compute_r_diag_with_correlations_applies_cross_term() {
        // Each observation's diagonal variance gains the 2·f·ρ·σ₁·σ₂ cross term.
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let ipreds = [10.0];
        let cmts = [0usize];
        let sigma = [0.2, 1.0];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        let with = compute_r_diag_with_correlations(&spec, &ipreds, &cmts, &sigma, &[corr]);
        assert_relative_eq!(with[0], 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_r_matrix_with_correlations_links_paired_endpoints() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0];
        let cmts = [1usize, 2, 2];
        let times = [1.0, 1.0, 2.0];
        let sigma = [0.2, 0.3];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        let r = compute_r_matrix_with_correlations(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &[],
            &sigma,
            &[corr],
        );
        assert_relative_eq!(r[(0, 0)], (50.0_f64 * 0.3).powi(2), epsilon = 1e-12);
        assert_relative_eq!(r[(1, 1)], (5.0_f64 * 0.2).powi(2), epsilon = 1e-12);
        assert_relative_eq!(r[(0, 1)], 50.0 * 5.0 * 0.5 * 0.3 * 0.2, epsilon = 1e-12);
        assert_relative_eq!(r[(1, 0)], r[(0, 1)], epsilon = 1e-12);
        assert_eq!(r[(0, 2)], 0.0);
    }

    // #827: two physical samples at the SAME subject time (replicate assays,
    // paired in NONMEM by the `L2` data item) must yield DISJOINT cross-covariance
    // pairs — each total row correlates with exactly one unbound row, not both.
    // An all-to-all 4×4 block would be indefinite and collapse the FOCEI objective
    // to the invalid sentinel; the disjoint pairing keeps `R` positive-definite.
    #[test]
    fn test_replicate_time_samples_pair_disjointly_and_stay_pd() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        // Rows in the NONMEM-contiguous layout: (total#1, unbound#1, total#2,
        // unbound#2), all at t = 1.0 — the replicate-time collision from ID 21.
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 1, 2];
        let times = [1.0, 1.0, 1.0, 1.0];
        let sigma = [0.671, 0.644];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.93,
        };
        let r = compute_r_matrix_with_correlations(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &[],
            &sigma,
            &[corr],
        );
        // Matched pairs (0,1) and (2,3) carry the cross covariance; the
        // cross-sample entries (0,3) and (1,2) must be exactly zero.
        assert!(r[(0, 1)] != 0.0 && r[(1, 0)] != 0.0);
        assert!(r[(2, 3)] != 0.0 && r[(3, 2)] != 0.0);
        assert_eq!(r[(0, 3)], 0.0);
        assert_eq!(r[(3, 0)], 0.0);
        assert_eq!(r[(1, 2)], 0.0);
        assert_eq!(r[(2, 1)], 0.0);
        // The whole matrix is PD (before the fix this cholesky failed → 2e20).
        assert!(
            r.cholesky().is_some(),
            "replicate-time residual R must be positive-definite"
        );
    }

    // #827: an explicit `L2` grouping id is authoritative — it pairs rows the
    // user grouped even across different times, and it keeps co-temporal rows in
    // *different* groups uncorrelated. Four rows all at t = 1.0, two total
    // (cmt 1) and two unbound (cmt 2), grouped L2 = {row0,row2} and {row1,row3}
    // — i.e. the pairing crosses the file order deliberately.
    #[test]
    fn test_l2_grouping_controls_pairing_over_time_and_order() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 2, 1];
        // Row times deliberately mixed: rows 0 and 2 share L2=10 but sit at
        // different times, proving L2 overrides the (time, occasion) fallback.
        let times = [1.0, 1.0, 9.0, 9.0];
        let obs_l2 = [10i64, 20, 10, 20];
        let sigma = [0.671, 0.644];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.93,
        };
        let r = compute_r_matrix_with_correlations(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &obs_l2,
            &sigma,
            &[corr],
        );
        // L2=10 pairs rows (0,2) across different times; L2=20 pairs (1,3).
        assert!(r[(0, 2)] != 0.0 && r[(2, 0)] != 0.0);
        assert!(r[(1, 3)] != 0.0 && r[(3, 1)] != 0.0);
        // Co-temporal but different-L2 (rows 0 & 1 both at t=1) stay uncorrelated,
        // as do all other cross-group entries.
        assert_eq!(r[(0, 1)], 0.0);
        assert_eq!(r[(0, 3)], 0.0);
        assert_eq!(r[(1, 2)], 0.0);
        assert_eq!(r[(2, 3)], 0.0);
        assert!(r.cholesky().is_some(), "L2-grouped residual R must be PD");
    }

    // #830: within one explicit `L2` group the rows form a single correlated
    // unit, so a genuine 3-endpoint block (parent + two metabolites, each pair
    // correlated and jointly positive-definite) must keep ALL THREE cross
    // covariances — not just one greedy pair. The disjoint greedy fallback would
    // pair only (0,1) and silently zero the (0,2)/(1,2) terms; all-to-all within
    // the L2 group restores the full block.
    #[test]
    fn test_l2_group_correlates_three_endpoints_all_to_all() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                3,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![2],
                },
            ),
        ]));
        let ipreds = [10.0, 20.0, 30.0];
        let cmts = [1usize, 2, 3];
        let times = [1.0, 1.0, 1.0];
        let obs_l2 = [7i64, 7, 7];
        let sigma = [0.3, 0.3, 0.3];
        // Three pairwise correlations at a mild rho so the 3×3 block stays PD.
        let corrs = [
            crate::types::ResidualCorrelation {
                sigma_i: 0,
                sigma_j: 1,
                rho: 0.3,
            },
            crate::types::ResidualCorrelation {
                sigma_i: 0,
                sigma_j: 2,
                rho: 0.3,
            },
            crate::types::ResidualCorrelation {
                sigma_i: 1,
                sigma_j: 2,
                rho: 0.3,
            },
        ];
        let r = compute_r_matrix_with_correlations(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &obs_l2,
            &sigma,
            &corrs,
        );
        // All three off-diagonals are populated (greedy one-to-one would leave
        // row 2 unmatched, zeroing r[(0,2)] and r[(1,2)]).
        assert!(r[(0, 1)] != 0.0 && r[(1, 0)] != 0.0);
        assert!(r[(0, 2)] != 0.0 && r[(2, 0)] != 0.0);
        assert!(r[(1, 2)] != 0.0 && r[(2, 1)] != 0.0);
        assert!(
            r.cholesky().is_some(),
            "PD 3-endpoint L2 block must stay positive-definite"
        );

        // The derivative builder must track the same all-to-all pairing: ∂R/∂f
        // matches a central difference of R for every observation.
        let dr = compute_dr_df_matrices(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &obs_l2,
            &sigma,
            &corrs,
            None,
        );
        let n = ipreds.len();
        let r_at = |f: &[f64]| {
            compute_r_matrix_with_correlations(
                &spec,
                f,
                &cmts,
                &times,
                &[],
                &[],
                &obs_l2,
                &sigma,
                &corrs,
            )
        };
        let h = 1e-4;
        for m in 0..n {
            let mut fp = ipreds.to_vec();
            let mut fm = ipreds.to_vec();
            fp[m] += h;
            fm[m] -= h;
            let fd = (r_at(&fp) - r_at(&fm)) / (2.0 * h);
            for p in 0..n {
                for q in 0..n {
                    assert_relative_eq!(
                        dr[m][(p, q)],
                        fd[(p, q)],
                        epsilon = 1e-5,
                        max_relative = 1e-4
                    );
                }
            }
        }
    }

    // #830: the pairing is structural (slot overlap), not value-based, so a
    // paired proportional row whose prediction is momentarily f = 0 keeps its
    // partner and the derivative builder still emits the (nonzero) slope-loading
    // cross term. Gating on `cov != 0.0` would drop it exactly at f ≈ 0.
    #[test]
    fn test_cross_derivative_survives_zero_prediction() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
        ]));
        // Row 0's prediction is exactly zero — its value loading (and hence the
        // value cross covariance) vanishes, but its slope loading does not.
        let ipreds = [0.0, 5.0];
        let cmts = [1usize, 2];
        let times = [1.0, 1.0];
        let sigma = [0.3, 0.4];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        // R's off-diagonal is genuinely 0 at f = 0 (value covariance), and the
        // block stays PD.
        let r = compute_r_matrix_with_correlations(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &[],
            &sigma,
            &[corr],
        );
        assert_eq!(r[(0, 1)], 0.0);
        // ∂R_01/∂f_0 = slope_0 · value_1 · ρ·σ0·σ1 = 1·5·0.5·0.3·0.4 = 0.3.
        let dr = compute_dr_df_matrices(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &[],
            &sigma,
            &[corr],
            None,
        );
        assert_relative_eq!(dr[0][(0, 1)], 0.3, epsilon = 1e-12);
        assert_relative_eq!(dr[0][(1, 0)], 0.3, epsilon = 1e-12);
        assert!(
            dr[0][(0, 1)] != 0.0,
            "cross derivative must survive a zero prediction"
        );
    }

    // The derivative builders must use the SAME disjoint pairing as
    // `compute_r_matrix_with_correlations`, or `∂R/∂f` would not match a finite
    // difference of `R`. Exercises the replicate-time (all-same-time) layout.
    #[test]
    fn test_compute_dr_df_matrices_matches_fd_replicate_time() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 1, 2];
        let times = [1.0, 1.0, 1.0, 1.0];
        let sigma = [0.671, 0.644];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.93,
        };
        let dr = compute_dr_df_matrices(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &[],
            &sigma,
            &[corr],
            None,
        );
        let n = ipreds.len();
        let r_at = |f: &[f64]| {
            compute_r_matrix_with_correlations(
                &spec,
                f,
                &cmts,
                &times,
                &[],
                &[],
                &[],
                &sigma,
                &[corr],
            )
        };
        let h = 1e-4;
        for m in 0..n {
            let mut fp = ipreds.to_vec();
            let mut fm = ipreds.to_vec();
            fp[m] += h;
            fm[m] -= h;
            let fd = (r_at(&fp) - r_at(&fm)) / (2.0 * h);
            for p in 0..n {
                for q in 0..n {
                    assert_relative_eq!(
                        dr[m][(p, q)],
                        fd[(p, q)],
                        epsilon = 1e-5,
                        max_relative = 1e-4
                    );
                }
            }
        }
    }

    // Central-difference check: ∂R/∂f_m from `compute_dr_df_matrices` must match
    // a finite-difference perturbation of `compute_r_matrix_with_correlations` for
    // every observation, on a paired-endpoint cross-correlated model.
    #[test]
    fn test_compute_dr_df_matrices_matches_finite_difference_paired() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Combined,
                    sigma_idx: vec![0, 2],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 1, 2];
        let times = [1.0, 1.0, 2.0, 2.0];
        let sigma = [0.2, 0.3, 1.5];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.4,
        };
        let dr = compute_dr_df_matrices(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &[],
            &sigma,
            &[corr],
            None,
        );
        let n = ipreds.len();
        let r_at = |f: &[f64]| {
            compute_r_matrix_with_correlations(
                &spec,
                f,
                &cmts,
                &times,
                &[],
                &[],
                &[],
                &sigma,
                &[corr],
            )
        };
        let h = 1e-4;
        for m in 0..n {
            let mut fp = ipreds.to_vec();
            let mut fm = ipreds.to_vec();
            fp[m] += h;
            fm[m] -= h;
            let fd = (r_at(&fp) - r_at(&fm)) / (2.0 * h);
            for p in 0..n {
                for q in 0..n {
                    assert_relative_eq!(
                        dr[m][(p, q)],
                        fd[(p, q)],
                        epsilon = 1e-5,
                        max_relative = 1e-4
                    );
                }
            }
        }
    }

    // With a #484 per-observation magnitude matrix, ∂R/∂f must track the
    // *scaled* covariance `compute_r_matrix_with_correlations_scaled`.
    #[test]
    fn test_compute_dr_df_matrices_matches_finite_difference_scaled() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 1, 2];
        let times = [1.0, 1.0, 2.0, 2.0];
        let sigma = [0.2, 0.3];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.4,
        };
        // Non-trivial per-observation, per-slot multipliers.
        let mult = vec![
            vec![1.0, 1.2],
            vec![0.8, 1.0],
            vec![1.1, 0.9],
            vec![1.3, 1.0],
        ];
        let dr = compute_dr_df_matrices(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &[],
            &sigma,
            &[corr],
            Some(&mult),
        );
        let n = ipreds.len();
        let r_at = |f: &[f64]| {
            compute_r_matrix_with_correlations_scaled(
                &spec,
                f,
                &cmts,
                &times,
                &[],
                &[],
                &[],
                &sigma,
                &[corr],
                &mult,
            )
        };
        let h = 1e-4;
        for m in 0..n {
            let mut fp = ipreds.to_vec();
            let mut fm = ipreds.to_vec();
            fp[m] += h;
            fm[m] -= h;
            let fd = (r_at(&fp) - r_at(&fm)) / (2.0 * h);
            for p in 0..n {
                for q in 0..n {
                    assert_relative_eq!(
                        dr[m][(p, q)],
                        fd[(p, q)],
                        epsilon = 1e-5,
                        max_relative = 1e-4
                    );
                }
            }
        }
    }

    // ∂²R/∂f_a∂f_b must match a central difference of the first-order
    // ∂R/∂f machinery: d2r[a][b] ≈ (dr(f+h·e_b)[a] − dr(f−h·e_b)[a]) / 2h.
    // Mixed CMTs (proportional + combined) and a cross-endpoint block_sigma
    // correlation exercise both the diagonal-curvature and the bilinear
    // off-diagonal mixed-partial branches.
    #[test]
    fn test_compute_d2r_df2_matrices_matches_finite_difference_paired() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Combined,
                    sigma_idx: vec![0, 2],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 1, 2];
        let times = [1.0, 1.0, 2.0, 2.0];
        let sigma = [0.2, 0.3, 1.5];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.4,
        };
        let d2r = compute_d2r_df2_matrices(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &[],
            &sigma,
            &[corr],
            None,
        );
        let n = ipreds.len();
        let dr_at = |f: &[f64]| {
            compute_dr_df_matrices(
                &spec,
                f,
                &cmts,
                &times,
                &[],
                &[],
                &[],
                &sigma,
                &[corr],
                None,
            )
        };
        let h = 1e-4;
        for b in 0..n {
            let mut fp = ipreds.to_vec();
            let mut fm = ipreds.to_vec();
            fp[b] += h;
            fm[b] -= h;
            let drp = dr_at(&fp);
            let drm = dr_at(&fm);
            for a in 0..n {
                let fd = (&drp[a] - &drm[a]) / (2.0 * h);
                for p in 0..n {
                    for q in 0..n {
                        assert_relative_eq!(
                            d2r[a][b][(p, q)],
                            fd[(p, q)],
                            epsilon = 1e-5,
                            max_relative = 1e-4
                        );
                    }
                }
            }
        }
    }

    // Same FD check with a #484 per-observation magnitude matrix: ∂²R/∂f²
    // must track the *scaled* covariance through `compute_dr_df_matrices`.
    #[test]
    fn test_compute_d2r_df2_matrices_matches_finite_difference_scaled() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 1, 2];
        let times = [1.0, 1.0, 2.0, 2.0];
        let sigma = [0.2, 0.3];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.4,
        };
        let mult = vec![
            vec![1.0, 1.2],
            vec![0.8, 1.0],
            vec![1.1, 0.9],
            vec![1.3, 1.0],
        ];
        let d2r = compute_d2r_df2_matrices(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &[],
            &sigma,
            &[corr],
            Some(&mult),
        );
        let n = ipreds.len();
        let dr_at = |f: &[f64]| {
            compute_dr_df_matrices(
                &spec,
                f,
                &cmts,
                &times,
                &[],
                &[],
                &[],
                &sigma,
                &[corr],
                Some(&mult),
            )
        };
        let h = 1e-4;
        for b in 0..n {
            let mut fp = ipreds.to_vec();
            let mut fm = ipreds.to_vec();
            fp[b] += h;
            fm[b] -= h;
            let drp = dr_at(&fp);
            let drm = dr_at(&fm);
            for a in 0..n {
                let fd = (&drp[a] - &drm[a]) / (2.0 * h);
                for p in 0..n {
                    for q in 0..n {
                        assert_relative_eq!(
                            d2r[a][b][(p, q)],
                            fd[(p, q)],
                            epsilon = 1e-5,
                            max_relative = 1e-4
                        );
                    }
                }
            }
        }
    }

    // The diagonal self-derivative must include a *within-observation*
    // `block_sigma` cross term (combined error with σ_prop ↔ σ_add correlated),
    // which `dvar_df` alone omits.
    #[test]
    fn test_compute_dr_df_matrices_within_obs_cross_term() {
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let sigma = [0.3, 1.2];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        let ipreds = [40.0];
        let cmts = [1usize];
        let times = [1.0];
        let dr = compute_dr_df_matrices(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &[],
            &sigma,
            &[corr],
            None,
        );
        // V = (f·σ0)² + σ1² + 2·f·1·ρ·σ0·σ1; ∂V/∂f = 2·f·σ0² + 2·ρ·σ0·σ1.
        let expected = 2.0 * 40.0 * 0.3 * 0.3 + 2.0 * 0.5 * 0.3 * 1.2;
        assert_relative_eq!(dr[0][(0, 0)], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_r_matrix_with_correlations_uses_shifted_time_after_reset() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 1, 2];
        let shifted_times = [1.0, 1.0, 101.0, 101.0];
        let raw_times = [1.0, 1.0, 1.0, 1.0];
        let sigma = [0.2, 0.3];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };

        let r = compute_r_matrix_with_correlations(
            &spec,
            &ipreds,
            &cmts,
            &shifted_times,
            &raw_times,
            &[],
            &[],
            &sigma,
            &[corr],
        );

        assert_relative_eq!(r[(0, 1)], 50.0 * 5.0 * 0.5 * 0.3 * 0.2, epsilon = 1e-12);
        assert_relative_eq!(r[(2, 3)], 40.0 * 4.0 * 0.5 * 0.3 * 0.2, epsilon = 1e-12);
        assert_eq!(r[(0, 3)], 0.0);
        assert_eq!(r[(1, 2)], 0.0);
    }

    #[test]
    fn test_min_variance_floor() {
        // Proportional with f=0 gives V=0, should be floored to MIN_VARIANCE
        let v = residual_variance(ErrorModel::Proportional, 0.0, &[0.1]);
        assert_relative_eq!(v, MIN_VARIANCE, epsilon = 1e-20);
    }

    /// #958: `variance_at` clamps the raw variance to `MIN_VARIANCE`, so on the
    /// clamped side the variance is *constant* in `f` and its exact `∂v/∂f` /
    /// `∂²v/∂f²` are both 0. The derivative accessors must reflect that clamp —
    /// returning the raw `2·f·σ²` / `2·σ²` there would make every analytic path
    /// that consumes `∂R/∂f` (inner-EBE gradient, FOCEI Laplace `c̃`, outer
    /// θ-gradient, covariance) disagree with the finite-difference objective,
    /// which sees the clamped function. Concretely a proportional-error row
    /// whose prediction is driven to ~0 (drug fully eliminated) made the FOCEI
    /// objective gradient-path-dependent.
    #[test]
    fn dvar_df_and_d2var_df2_are_floor_aware() {
        let spec = ErrorSpec::Single(ErrorModel::Proportional);
        let sigma = [0.15];
        // Prediction small enough that (f·σ)² < MIN_VARIANCE → variance clamped.
        let f_floored = 1e-6_f64; // (1e-6·0.15)² = 2.25e-14 < 1e-12
        assert!(
            spec.variance_at(1, f_floored, &sigma) <= MIN_VARIANCE,
            "precondition: variance must be clamped at this prediction"
        );
        assert_eq!(
            spec.dvar_df(1, f_floored, &sigma),
            0.0,
            "∂v/∂f must be 0 where the variance floor is active"
        );
        assert_eq!(
            spec.d2var_df2(1, f_floored, &sigma),
            0.0,
            "∂²v/∂f² must be 0 where the variance floor is active"
        );

        // Above the floor, the derivatives are the raw proportional values and
        // must match central finite differences of `variance_at` itself.
        let f = 10.0_f64; // (10·0.15)² = 2.25 ≫ MIN_VARIANCE
        assert!(spec.variance_at(1, f, &sigma) > MIN_VARIANCE);
        let h = 1e-4;
        let fd1 =
            (spec.variance_at(1, f + h, &sigma) - spec.variance_at(1, f - h, &sigma)) / (2.0 * h);
        assert_relative_eq!(spec.dvar_df(1, f, &sigma), fd1, max_relative = 1e-6);
        assert_relative_eq!(
            spec.dvar_df(1, f, &sigma),
            2.0 * f * 0.15 * 0.15,
            epsilon = 1e-12
        );
        let fd2 = (spec.variance_at(1, f + h, &sigma) - 2.0 * spec.variance_at(1, f, &sigma)
            + spec.variance_at(1, f - h, &sigma))
            / (h * h);
        assert_relative_eq!(spec.d2var_df2(1, f, &sigma), fd2, max_relative = 1e-4);
    }

    /// #958 scope gap on the #484 custom-magnitude path: the `_scaled` derivative
    /// accessors must gate the `MIN_VARIANCE` floor on the *scaled* variance they
    /// pair with (`variance_at_scaled`), not the unscaled `variance_at`. With a
    /// magnitude multiplier `m ≠ 1` the two floor at different predictions, so a
    /// unscaled gate zeroed the analytic slope on a band where the objective's
    /// variance is still live — reviving the analytic-vs-FD gradient mismatch.
    #[test]
    fn dvar_df_scaled_floor_gates_on_scaled_variance() {
        let spec = ErrorSpec::Single(ErrorModel::Proportional);
        let sigma = [0.15];
        let mult = [100.0]; // m = 100 on the proportional sigma slot
        let corr: [ResidualCorrelation; 0] = [];

        // Band where the UNSCALED variance is floored but the SCALED variance is
        // still live: 1e-6/(m·σ) < f < 1e-6/σ, i.e. 6.7e-8 < f < 6.7e-6.
        let f_band = 1e-6_f64;
        assert!(
            spec.variance_at(1, f_band, &sigma) <= MIN_VARIANCE,
            "precondition: unscaled variance is floored here"
        );
        assert!(
            spec.variance_at_scaled(1, f_band, &sigma, &corr, &mult) > MIN_VARIANCE,
            "precondition: scaled variance (m=100) is above the floor here"
        );
        // The old unscaled gate returned 0 here; the derivative must instead be
        // the live scaled slope, matching central FD of `variance_at_scaled`.
        assert_ne!(
            spec.dvar_df_scaled(1, f_band, &sigma, &mult),
            0.0,
            "∂v/∂f must be live where the SCALED variance is above the floor"
        );
        let h = 1e-8;
        let fd1 = (spec.variance_at_scaled(1, f_band + h, &sigma, &corr, &mult)
            - spec.variance_at_scaled(1, f_band - h, &sigma, &corr, &mult))
            / (2.0 * h);
        assert_relative_eq!(
            spec.dvar_df_scaled(1, f_band, &sigma, &mult),
            fd1,
            max_relative = 1e-5
        );
        let fd2 = (spec.variance_at_scaled(1, f_band + h, &sigma, &corr, &mult)
            - 2.0 * spec.variance_at_scaled(1, f_band, &sigma, &corr, &mult)
            + spec.variance_at_scaled(1, f_band - h, &sigma, &corr, &mult))
            / (h * h);
        assert_relative_eq!(
            spec.d2var_df2_scaled(1, f_band, &sigma, &mult),
            fd2,
            max_relative = 1e-3
        );

        // Below the SCALED floor both derivatives are 0 (variance locally flat).
        let f_floored = 1e-9_f64;
        assert!(
            spec.variance_at_scaled(1, f_floored, &sigma, &corr, &mult) <= MIN_VARIANCE,
            "precondition: scaled variance is floored here"
        );
        assert_eq!(spec.dvar_df_scaled(1, f_floored, &sigma, &mult), 0.0);
        assert_eq!(spec.d2var_df2_scaled(1, f_floored, &sigma, &mult), 0.0);

        // Well above the floor the scaled forms equal the raw m²-scaled values.
        let f_hi = 10.0_f64;
        assert_relative_eq!(
            spec.dvar_df_scaled(1, f_hi, &sigma, &mult),
            2.0 * f_hi * (100.0 * 0.15) * (100.0 * 0.15),
            max_relative = 1e-12
        );
        assert_relative_eq!(
            spec.d2var_df2_scaled(1, f_hi, &sigma, &mult),
            2.0 * (100.0 * 0.15) * (100.0 * 0.15),
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_iwres_perfect_prediction() {
        let r = iwres(10.0, 10.0, ErrorModel::Additive, &[1.0]);
        assert_relative_eq!(r, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_iwres_known_value() {
        // IWRES = (y - f) / sqrt(V) = (12 - 10) / sqrt(1) = 2.0
        let r = iwres(12.0, 10.0, ErrorModel::Additive, &[1.0]);
        assert_relative_eq!(r, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_r_diag_length() {
        // Single-endpoint model (Additive): CMT is ignored.
        let model = crate::types::test_helpers::analytical_model(GradientMethod::Auto);
        let ipreds = vec![1.0, 2.0, 3.0];
        let obs_cmts = vec![1, 1, 1];
        let r = compute_r_diag(&model.error_spec, &ipreds, &obs_cmts, &[0.5]);
        assert_eq!(r.len(), 3);
        // Additive variance is sigma^2 regardless of prediction/CMT.
        for v in &r {
            assert_relative_eq!(*v, 0.25, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_compute_iwres_vectorized() {
        let model = crate::types::test_helpers::analytical_model(GradientMethod::Auto);
        let obs = vec![12.0, 22.0];
        let ipreds = vec![10.0, 20.0];
        let obs_cmts = vec![1, 1];
        let result = compute_iwres(&obs, &ipreds, &obs_cmts, &model.error_spec, &[1.0]);
        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_iwres_with_correlations_applies_cross_term() {
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        // V = (10 * 0.2)^2 + 1^2 + 2 * 10 * 0.5 * 0.2 * 1 = 7.
        let result = compute_iwres_with_correlations(
            &[12.0],
            &[10.0],
            &[1],
            &spec,
            &[0.2, 1.0],
            &[corr],
            None,
        );
        assert_relative_eq!(result[0], 2.0 / 7.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_compute_iwres_with_correlations_empty_matches_diagonal() {
        let spec = ErrorSpec::Single(ErrorModel::Additive);
        let obs = [12.0, 22.0];
        let ipreds = [10.0, 20.0];
        let obs_cmts = [1, 1];
        let plain = compute_iwres(&obs, &ipreds, &obs_cmts, &spec, &[1.0]);
        let with =
            compute_iwres_with_correlations(&obs, &ipreds, &obs_cmts, &spec, &[1.0], &[], None);
        assert_eq!(plain, with);
    }

    #[test]
    fn compute_r_matrix_diagonal_keeps_legacy_association() {
        // Bit-reproducibility guard. The bare-sigma R diagonal must keep
        // `residual_variance`'s `(f·σ)·(f·σ)` association, NOT the
        // `((f·f)·σ)·σ` form `variance_at_scaled` uses. The two are equal in
        // exact arithmetic but differ by ~1 ULP under IEEE-754 on ~55% of
        // proportional/combined rows — so delegating `compute_r_matrix_with_correlations`
        // to `_scaled` with an empty multiplier would silently shift every
        // proportional/combined FOCE OFV and CWRES off its bit-for-bit value.
        // `f = 32.451, σ = 0.159` is one such divergent pair.
        let spec = ErrorSpec::Single(ErrorModel::Proportional);
        let f = 32.451_f64;
        let s = 0.159_f64;
        let legacy = (f * s) * (f * s);
        let reassociated = ((f * f) * s) * s;
        assert_ne!(
            legacy.to_bits(),
            reassociated.to_bits(),
            "fixture must be a pair where the two associations differ"
        );
        let r =
            compute_r_matrix_with_correlations(&spec, &[f], &[1], &[0.0], &[], &[], &[], &[s], &[]);
        assert_eq!(
            r[(0, 0)].to_bits(),
            legacy.to_bits(),
            "R diagonal must use the legacy (f·σ)·(f·σ) association"
        );
    }

    #[test]
    fn test_compute_iwres_with_correlations_applies_custom_magnitude() {
        // #484 review #4: the sdtab IWRES must use the per-observation magnitude
        // multiplier, so a row whose multiplier ≠ 1 is scaled by it. Proportional
        // error: V = (f·m·σ)², so IWRES = (y−f)/(f·m·σ).
        let spec = ErrorSpec::Single(ErrorModel::Proportional);
        let obs = [12.0, 22.0];
        let ipreds = [10.0, 20.0];
        let obs_cmts = [1, 1];
        let sigma = [0.2];
        // Row 0 bare (mult 1), row 1 inflated by 2.
        let mult = vec![vec![1.0], vec![2.0]];
        let scaled = compute_iwres_with_correlations(
            &obs,
            &ipreds,
            &obs_cmts,
            &spec,
            &sigma,
            &[],
            Some(&mult),
        );
        assert_relative_eq!(scaled[0], 2.0 / (10.0 * 1.0 * 0.2), epsilon = 1e-12);
        assert_relative_eq!(scaled[1], 2.0 / (20.0 * 2.0 * 0.2), epsilon = 1e-12);

        // An all-ones multiplier reproduces the unscaled IWRES exactly.
        let ones = vec![vec![1.0], vec![1.0]];
        let unit = compute_iwres_with_correlations(
            &obs,
            &ipreds,
            &obs_cmts,
            &spec,
            &sigma,
            &[],
            Some(&ones),
        );
        let bare =
            compute_iwres_with_correlations(&obs, &ipreds, &obs_cmts, &spec, &sigma, &[], None);
        assert_relative_eq!(unit[0], bare[0], epsilon = 1e-12);
        assert_relative_eq!(unit[1], bare[1], epsilon = 1e-12);
    }

    fn make_subject(iwres: Vec<f64>) -> SubjectResult {
        use nalgebra::DVector;
        SubjectResult {
            id: "1".to_string(),
            eta: DVector::zeros(0),
            ipred: vec![0.0; iwres.len()],
            pred: vec![0.0; iwres.len()],
            iwres,
            cwres: vec![],
            npde: vec![],
            npd: vec![],
            ofv_contribution: 0.0,
            cens: vec![],
            n_obs: 0,
            pmix: None,
            mixest: None,
            extra_columns: vec![],
            per_obs_tad: vec![],
            compartment_states: vec![],
            #[cfg(feature = "survival")]
            discrete_rows: Vec::new(),
        }
    }

    #[test]
    fn test_dw_monotone_positive_autocorrelation() {
        // Monotonically increasing → strong positive autocorrelation → DW near 0
        let subj = make_subject(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (r, dw) = iwres_autocorrelation(&[subj]);
        assert!(
            dw < 1.5,
            "expected DW < 1.5 for monotone sequence, got {dw}"
        );
        assert!(r > 0.5, "expected positive lag-1 r, got {r}");
    }

    #[test]
    fn test_dw_alternating_negative_autocorrelation() {
        // Alternating signs → strong negative autocorrelation → DW near 4
        let subj = make_subject(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let (_r, dw) = iwres_autocorrelation(&[subj]);
        assert!(
            dw > 2.5,
            "expected DW > 2.5 for alternating sequence, got {dw}"
        );
    }

    #[test]
    fn test_dw_uncorrelated_near_two() {
        // White-noise-like residuals → DW near 2 (no autocorrelation)
        let subj = make_subject(vec![1.0, -0.5, 0.2, 0.8, -0.3, -0.7, 0.4, 0.1]);
        let (_r, dw) = iwres_autocorrelation(&[subj]);
        assert!(
            dw > 1.5 && dw < 2.5,
            "expected DW near 2 for white-noise sequence, got {dw}"
        );
    }

    #[test]
    fn test_nan_iwres_skipped() {
        let subj = make_subject(vec![f64::NAN, 1.0, 2.0, f64::NAN, 3.0]);
        let (_r, dw) = iwres_autocorrelation(&[subj]);
        // Should not panic and should produce a finite result based on valid values
        assert!(
            dw.is_finite(),
            "DW should be finite after skipping NaN entries"
        );
    }

    #[test]
    fn test_single_observation_subject_skipped() {
        let single = make_subject(vec![1.0]);
        let multi = make_subject(vec![1.0, 2.0, 3.0]);
        let (r, dw) = iwres_autocorrelation(&[single, multi]);
        assert!(dw.is_finite());
        assert!(r.is_finite());
    }

    #[test]
    fn test_no_valid_subjects_returns_nan() {
        let subj = make_subject(vec![1.0]); // < 2 valid
        let (r, dw) = iwres_autocorrelation(&[subj]);
        assert!(r.is_nan());
        assert!(dw.is_nan());
    }
}

/// Residual variance `R` and its `f`-derivative `d = ∂R/∂f` for one observation,
/// dispatching on whether a custom σ-magnitude is active (`mult_row = Some`): the
/// scaled closed forms when it is, the legacy ones when it is not (`None` keeps
/// the exact `variance_at`/`dvar_df` association bit-for-bit — the `_scaled`
/// variants reassociate the `f`-dependent term by ~1 ULP). Callers apply
/// `ruv_scale` (and any FD combination) at the call site so float association is
/// unchanged. Diagonal-`R` only (`block_sigma` correlations force FD upstream).
#[inline]
pub(crate) fn residual_rd(
    es: &ErrorSpec,
    cmt: usize,
    f: f64,
    sigma: &[f64],
    mult_row: Option<&[f64]>,
) -> (f64, f64) {
    match mult_row {
        Some(m) => (
            es.variance_at_scaled(cmt, f, sigma, &[], m),
            es.dvar_df_scaled(cmt, f, sigma, m),
        ),
        None => (es.variance_at(cmt, f, sigma), es.dvar_df(cmt, f, sigma)),
    }
}

/// [`residual_rd`] plus the second `f`-derivative `d2 = ∂²R/∂f²`.
#[inline]
pub(crate) fn residual_rd2(
    es: &ErrorSpec,
    cmt: usize,
    f: f64,
    sigma: &[f64],
    mult_row: Option<&[f64]>,
) -> (f64, f64, f64) {
    match mult_row {
        Some(m) => (
            es.variance_at_scaled(cmt, f, sigma, &[], m),
            es.dvar_df_scaled(cmt, f, sigma, m),
            es.d2var_df2_scaled(cmt, f, sigma, m),
        ),
        None => (
            es.variance_at(cmt, f, sigma),
            es.dvar_df(cmt, f, sigma),
            es.d2var_df2(cmt, f, sigma),
        ),
    }
}
