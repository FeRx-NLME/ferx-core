//! Built-in absorption **input-rate functions** — `R_in(tad)` per model (#322).
//!
//! Each returns the dose-driven appearance rate into the compartment it feeds,
//! normalised so `∫₀^∞ R_in dt = dose`, where the caller folds bioavailability
//! into `dose = F · amt`. `R_in = 0` for `tad ≤ 0` (the input starts after the
//! dose); per-dose contributions are superposed by the caller.
//!
//! These are the inherently-numerical absorption models that feed an explicit
//! ODE disposition (see `plans/absorption-models.md`). They use only
//! `+ − * /`, `.ln()`, `.exp()`, and `ln_gamma` — all on the `PkNum`
//! trait. The input-rate forcing is **generic over `T: PkNum`** (the `sens/`
//! `*_g<T>` convention), so `transit()` and `igd()` models are evaluated over
//! `Dual2` by `sens/ode_provider.rs` and get **exact analytic** FOCE/FOCEI/Bayes
//! sensitivities, not finite differences (#430). (The Enzyme `autodiff` path
//! these once targeted was retired in #367/#381; `Dual2` handles `max`/`min` by
//! comparison, so the old `f64::max`/`min` restriction no longer applies.)

use crate::sens::num::PkNum;

/// `ln(2π)` — the inverse-Gaussian log-density normalisation constant. A literal
/// (rather than a runtime `(2.0 * PI).ln()`) since `f64::ln` is not `const`; the
/// `igd_matches_direct_density_formula` test pins it to the textbook density to
/// 1e-12, so a typo here cannot pass unnoticed.
const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// Savic et al. (2007) transit-compartment input rate into the **depot**, for a
/// *continuous* number of transit compartments `n`:
///
/// ```text
/// R_in(tad) = dose · KTR · (KTR·tad)^n · exp(−KTR·tad) / Γ(n + 1),
///   KTR = (n + 1) / mtt,   dose = F · amt.
/// ```
///
/// The depot then empties to central via first-order `ka` (applied in the ODE,
/// not here). `∫₀^∞ R_in dt = dose`. Returns `0` for `tad ≤ 0` and for a
/// non-positive `dose`.
///
/// Domain: `mtt > 0`, `n ≥ 0` (enforce upstream with [`validate_transit`]).
/// Evaluated in the log domain for stability with large `n` / `(KTR·tad)^n`.
///
/// This is the readable reference form (used by tests and one-shot callers); the
/// ODE hot path goes through [`InputRateForcing::prepare`] +
/// [`PreparedInputRate::rate`], which hoist the dose-invariant constants
/// (`ln Γ`, `KTR`, `ln KTR`) out of the per-dose superposition loop.
pub fn transit_input_rate(tad: f64, n: f64, mtt: f64, dose: f64) -> f64 {
    PreparedInputRate::transit(n, mtt).rate(tad, dose)
}

/// Validate transit parameters: `mtt` strictly positive, `n` non-negative.
/// The negated comparisons also reject `NaN`.
pub fn validate_transit(n: f64, mtt: f64) -> Result<(), String> {
    if !(mtt > 0.0) {
        return Err(format!(
            "transit: mtt (mean transit time) must be > 0, got {mtt}"
        ));
    }
    if !(n >= 0.0) {
        return Err(format!(
            "transit: n (number of transit compartments) must be ≥ 0, got {n}"
        ));
    }
    Ok(())
}

/// Freijer & Post (1997) inverse-Gaussian (convection–dispersion) absorption
/// input rate into the **central** compartment, for mean absorption time `mat`
/// and relative dispersion `cv2`:
///
/// ```text
/// R_in(tad) = dose · √(MAT / (2π·CV²·tad³)) · exp(−(tad−MAT)² / (2·CV²·MAT·tad)).
/// ```
///
/// This is the standard inverse-Gaussian density scaled by the dose, with mean
/// `μ = MAT` and shape `λ = MAT/CV²` (`CV²` = relative dispersion, Var/mean²);
/// unlike `transit`, it models the *entire* absorption delay and feeds central
/// directly (no downstream `ka`). `∫₀^∞ R_in dt = dose`. Returns `0` for
/// `tad ≤ 0` and for a non-positive `dose`.
///
/// Domain: `mat > 0`, `cv2 > 0` (enforce upstream with [`validate_igd`]).
/// Evaluated in the log domain for stability: the essential singularity at
/// `tad → 0` collapses to `R_in → 0` because the `−(tad−MAT)²/(2·CV²·MAT·tad)`
/// term diverges like `−MAT/(2·CV²·tad)`, dominating the `−1.5·ln tad` term.
///
/// This is the readable reference form (used by tests and one-shot callers); the
/// ODE hot path goes through [`InputRateForcing::prepare`] +
/// [`PreparedInputRate::rate`], which hoist the dose-invariant constants
/// (`c0`, `1/(2·CV²·MAT)`) out of the per-dose superposition loop.
pub fn inverse_gaussian_input_rate(tad: f64, mat: f64, cv2: f64, dose: f64) -> f64 {
    PreparedInputRate::inverse_gaussian(mat, cv2).rate(tad, dose)
}

/// Validate inverse-Gaussian parameters: `mat` and `cv2` strictly positive.
/// The negated comparisons also reject `NaN`.
pub fn validate_igd(mat: f64, cv2: f64) -> Result<(), String> {
    if !(mat > 0.0) {
        return Err(format!(
            "igd: mat (mean absorption time) must be > 0, got {mat}"
        ));
    }
    if !(cv2 > 0.0) {
        return Err(format!(
            "igd: cv2 (relative dispersion) must be > 0, got {cv2}"
        ));
    }
    Ok(())
}

/// Weibull absorption input rate into the compartment it feeds, for scale `td`
/// (> 0) and shape `beta` (> 0):
///
/// ```text
/// R_in(tad) = dose · (β/Td) · (tad/Td)^(β−1) · exp(−(tad/Td)^β),   dose = F · amt.
/// ```
///
/// This is the Weibull density scaled by the dose — the appearance-rate form of
/// the cumulative absorbed fraction `1 − exp(−(tad/Td)^β)`. `∫₀^∞ R_in dt = dose`.
/// Returns `0` for `tad ≤ 0` and for a non-positive `dose`.
///
/// Domain: `td > 0`, `beta > 0` (enforce upstream with [`validate_weibull`]).
/// Evaluated in the log domain so the power `(tad/Td)^β` is `exp(β·ln(tad/Td))` —
/// **no `powf` / new special function** is needed, and the only building blocks are
/// `ln`/`exp`/`+ − * /`, all on [`PkNum`], so a `weibull()` model carries exact
/// analytic `Dual2` sensitivities through `sens/ode_provider.rs` (Phase 2).
///
/// Weibull has **no elementary closed form** with linear disposition, so it is
/// permanently on the numerical ODE path; this `PkNum`-generic forcing is its only
/// route to exact (rather than finite-difference) gradients. The shape `β` governs
/// the `tad → 0⁺` edge: `β > 1 ⇒ R_in → 0`, `β = 1 ⇒` first-order with `ka = 1/Td`,
/// `β < 1 ⇒ R_in → ∞` (an **integrable** spike — `∫R_in` stays `= dose`); the
/// `tad ≤ 0 ⇒ 0` guard makes the input start at the dose, and the adaptive solver
/// resolves the spike with small steps rather than the value being capped (a cap
/// would break mass balance).
///
/// This is the readable reference form (used by tests and one-shot callers); the
/// ODE hot path goes through [`InputRateForcing::prepare`] +
/// [`PreparedInputRate::rate`], which hoist the dose-invariant constants
/// (`ln β − ln Td`, `ln Td`) out of the per-dose superposition loop.
pub fn weibull_input_rate(tad: f64, td: f64, beta: f64, dose: f64) -> f64 {
    PreparedInputRate::weibull(td, beta).rate(tad, dose)
}

/// Validate Weibull parameters: `td` (scale) and `beta` (shape) strictly
/// positive. The negated comparisons also reject `NaN`.
pub fn validate_weibull(td: f64, beta: f64) -> Result<(), String> {
    if !(td > 0.0) {
        return Err(format!("weibull: td (scale) must be > 0, got {td}"));
    }
    if !(beta > 0.0) {
        return Err(format!("weibull: beta (shape) must be > 0, got {beta}"));
    }
    Ok(())
}

/// Zero-order absorption input rate into the compartment it feeds, for a modeled
/// duration `dur` (> 0):
///
/// ```text
/// R_in(tad) = dose / dur   for 0 < tad ≤ dur,   0 otherwise,   dose = F · amt.
/// ```
///
/// A constant rate over `(0, dur]` — the zero-order (`D1`-style, #324) input
/// reused as an absorption term. `∫₀^∞ R_in dt = (dose/dur)·dur = dose` exactly.
/// Returns `0` for `tad ≤ 0` and for a non-positive `dose`.
///
/// Domain: `dur > 0` (enforce upstream with [`validate_zero_order`]). Unlike the
/// smooth densities, the rate has a **hard cutoff at `tad = dur`**: callers on the
/// ODE path must break the integration timeline there, and the `∂R_in/∂dur`
/// boundary impulse is FD-only until #530 (see [`InputRateKind::supported_over_dual`]).
///
/// This is the readable reference form (used by tests and one-shot callers); the
/// ODE hot path goes through [`InputRateForcing::prepare`] +
/// [`PreparedInputRate::rate`], which hoist `1/dur` out of the per-dose loop.
pub fn zero_order_input_rate(tad: f64, dur: f64, dose: f64) -> f64 {
    PreparedInputRate::zero_order(dur).rate(tad, dose)
}

/// Validate the zero-order parameter: `dur` (duration) strictly positive. The
/// negated comparison also rejects `NaN`.
pub fn validate_zero_order(dur: f64) -> Result<(), String> {
    if !(dur > 0.0) {
        return Err(format!("zero_order: dur (duration) must be > 0, got {dur}"));
    }
    Ok(())
}

/// First-order absorption input rate into the compartment it feeds, for an
/// absorption rate constant `ka` (> 0):
///
/// ```text
/// R_in(tad) = dose · ka · exp(−ka·tad),   dose = F · amt.
/// ```
///
/// This is the classic first-order (Bateman) input — the same absorption the
/// analytical `pk *_oral` solvers use, exposed here as an input-rate function so
/// it can be **composed** in `[odes]` for parallel / mixed dual-pathway models
/// (#505): `FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2)` (parallel) and
/// `FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR)` (mixed). Standalone
/// first-order absorption still uses the analytical `pk *_oral` path; this is for
/// the multi-pathway case the closed forms can't express in one term.
/// `∫₀^∞ R_in dt = dose`. Returns `0` for `tad ≤ 0` and for a non-positive `dose`.
///
/// Domain: `ka > 0` (enforce upstream with [`validate_first_order`]). The
/// `tad → 0⁺` edge is finite (`R_in → dose·ka`), so there is no singularity to
/// guard — unlike the densities above. The single building block is `exp` (on
/// [`PkNum`]), so a `first_order()` model carries exact analytic `Dual2`
/// sensitivities through `sens/ode_provider.rs` (it is identical to `transit`
/// with `n = 0`, whose middle term `0·ln x` vanishes).
///
/// This is the readable reference form (used by tests and one-shot callers); the
/// ODE hot path goes through [`InputRateForcing::prepare`] +
/// [`PreparedInputRate::rate`], which hoist `ka` out of the per-dose loop.
pub fn first_order_input_rate(tad: f64, ka: f64, dose: f64) -> f64 {
    PreparedInputRate::first_order(ka).rate(tad, dose)
}

/// Validate the first-order parameter: `ka` (absorption rate constant) strictly
/// positive. The negated comparison also rejects `NaN`.
pub fn validate_first_order(ka: f64) -> Result<(), String> {
    if !(ka > 0.0) {
        return Err(format!(
            "first_order: ka (absorption rate constant) must be > 0, got {ka}"
        ));
    }
    Ok(())
}

/// Which built-in absorption input-rate model a forcing term uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputRateKind {
    /// Savic transit-compartment chain — `transit(n, mtt)`.
    Transit,
    /// Freijer & Post inverse-Gaussian density — `igd(mat, cv2)`.
    InverseGaussian,
    /// Weibull density absorption — `weibull(td, beta)`.
    Weibull,
    /// Zero-order (constant-rate) absorption over a modeled duration —
    /// `zero_order(dur)`. `R_in = dose/dur` on `(0, dur]`, else 0 (#504). Unlike
    /// the smooth densities above, this has a **hard cutoff at `tad = dur`**: the
    /// ODE timeline must place a break there (see `predictions.rs`), and its
    /// `∂R_in/∂dur` is a moving boundary that the pointwise `Dual2` walk cannot
    /// express, so it stays on the FD fallback ([`Self::supported_over_dual`])
    /// until the boundary-impulse mechanism lands (#530).
    ZeroOrder,
    /// First-order (Bateman) absorption — `first_order(ka)`. `R_in = dose·ka·
    /// e^{−ka·tad}` (#505). The existing first-order absorption exposed as an
    /// input-rate function so it can be **composed** in `[odes]` for parallel /
    /// mixed dual-pathway models; standalone first-order still uses the analytical
    /// `pk *_oral` path. A smooth density like the three above (it uses only
    /// `exp`), so it stays on the pointwise `R_in(tad)` path and is lifted to
    /// `Dual2` ([`Self::supported_over_dual`] = `true`).
    FirstOrder,
}

impl InputRateKind {
    /// Whether this kind's input-rate forcing has been lifted to `PkNum`/`Dual2`
    /// so the analytic ODE sensitivity provider (`sens/ode_provider.rs`) can
    /// evaluate it over dual numbers; otherwise a model using it stays on the FD
    /// fallback. Inverse-Gaussian (#430 slice 1), transit (#430 slice 2, via the
    /// `ln_gamma` `Dual2` rule = digamma/trigamma), and Weibull (Phase 2 — its
    /// log-domain forcing needs only the `ln`/`exp` `Dual2` rules) are all lifted.
    pub fn supported_over_dual(self) -> bool {
        // Exhaustive (no `_` arm) so adding a kind forces a decision here, and must
        // stay consistent with [`InputRateForcing::prepare_dual`] — a kind marked
        // supported here but returning `None` there would let the ODE provider admit
        // the model and then silently bail the whole subject to FD. The
        // `supported_over_dual_agrees_with_prepare_dual` test pins that consistency
        // (#430 review #5 / #451).
        match self {
            InputRateKind::InverseGaussian => true,
            InputRateKind::Transit => true,
            InputRateKind::Weibull => true,
            // First-order is a smooth `dose·ka·e^{−ka·tad}` density (only `exp`),
            // pointwise like the three above — exact `Dual2` forcing sensitivities
            // for `ka` and the pathway fraction `frac` (#505).
            InputRateKind::FirstOrder => true,
            // Zero-order has a hard cutoff at `tad = dur`; `∂R_in/∂dur` carries a
            // Leibniz boundary term (a dual-channel impulse at the cutoff) that the
            // smooth pointwise `rate(tad) → Dual2` walk misses. Until that impulse
            // is built (#530, shared with #384's modeled-duration `Dn` infusion),
            // `dur` is differentiated by FD — so this kind is *not* lifted, and
            // `prepare_dual` returns `None` for it (pinned by
            // `supported_over_dual_agrees_with_prepare_dual`).
            InputRateKind::ZeroOrder => false,
        }
    }
}

/// A built-in absorption input-rate term attached to one ODE compartment.
///
/// Design A (see `plans/absorption-models.md`): the input-rate function is split
/// out of the `[odes]` RHS at parse time and evaluated here with dose context,
/// rather than threaded through the expression AST / bytecode VM / symbolic-AD
/// machinery. `arg_slots` index the flat individual-parameter vector for this
/// model's parameters — for [`InputRateKind::Transit`], `[n, mtt]`; for
/// [`InputRateKind::InverseGaussian`], `[mat, cv2]`; for
/// [`InputRateKind::Weibull`], `[td, beta]`; for [`InputRateKind::ZeroOrder`],
/// the single `[dur]`.
///
/// `frac_slot` is the optional **pathway fraction** multiplier for parallel /
/// biphasic absorption (#388): when a `[odes]` term is written `FR*igd(...)`, the
/// leading declared individual parameter `FR` is resolved to its slot here, and
/// the delivered rate is `frac · R_in` (so the dose mass splits across pathways by
/// linearity). `None` is the common single-pathway case (`frac = 1`). The fraction
/// is required to be a single declared parameter — not an arbitrary expression
/// like `(1-FR)` — so it binds to one slot; a two-pathway split is two terms whose
/// fractions sum to 1 (validated at fit-init).
#[derive(Debug, Clone)]
pub struct InputRateForcing {
    /// 0-based ODE compartment that receives `R_in`.
    pub cmt: usize,
    pub kind: InputRateKind,
    /// Indices into the flat individual-parameter vector for this model's args.
    pub arg_slots: Vec<usize>,
    /// Optional slot of the pathway-fraction multiplier (`FR*fn(...)`, #388).
    /// `None` ⇒ fraction `1` (single pathway). See [`Self::frac`].
    pub frac_slot: Option<usize>,
}

impl InputRateForcing {
    /// Read this forcing's argument `i` from the flat individual-parameter
    /// vector `params`, falling back to `dflt` if the slot is absent. Generic
    /// over `T: PkNum` so the same reader serves the `f64` prediction path and
    /// the `Dual2<N>` sensitivity provider (`params` is then the dual
    /// individual-parameter vector).
    #[inline]
    fn arg<T: PkNum>(&self, params: &[T], i: usize, dflt: f64) -> T {
        self.arg_slots
            .get(i)
            .and_then(|&s| params.get(s))
            .copied()
            .unwrap_or(T::from_f64(dflt))
    }

    /// The pathway-fraction multiplier for this forcing, read from the flat
    /// individual-parameter vector `params`. Returns `1` when there is no
    /// `frac_slot` (the single-pathway default) or the slot is absent, so an
    /// unfractioned `igd(...)`/`transit(...)`/etc. is unaffected. Generic over
    /// `T: PkNum` so the `Dual2` provider carries the exact `∂R_in/∂frac`
    /// sensitivity (the multiplier flows linearly into the forcing — #388).
    #[inline]
    pub fn frac<T: PkNum>(&self, params: &[T]) -> T {
        self.frac_slot
            .and_then(|s| params.get(s))
            .copied()
            .unwrap_or(T::from_f64(1.0))
    }

    /// Precompute the dose-invariant constants for this forcing's parameters
    /// (read from the flat individual-parameter vector `params`). Call **once**
    /// per RHS evaluation, then evaluate [`PreparedInputRate::rate`] per dose —
    /// this keeps the expensive `ln Γ` (and `KTR`, `ln KTR`) out of the per-dose
    /// superposition loop on the ODE hot path.
    pub fn prepare(&self, params: &[f64]) -> PreparedInputRate {
        match self.kind {
            InputRateKind::Transit => {
                PreparedInputRate::transit(self.arg(params, 0, 0.0), self.arg(params, 1, 1.0))
            }
            InputRateKind::InverseGaussian => PreparedInputRate::inverse_gaussian(
                self.arg(params, 0, 1.0),
                self.arg(params, 1, 1.0),
            ),
            InputRateKind::Weibull => {
                PreparedInputRate::weibull(self.arg(params, 0, 1.0), self.arg(params, 1, 1.0))
            }
            InputRateKind::ZeroOrder => PreparedInputRate::zero_order(self.arg(params, 0, 1.0)),
            InputRateKind::FirstOrder => PreparedInputRate::first_order(self.arg(params, 0, 1.0)),
        }
    }

    /// Validate this forcing's parameters (read from the flat individual-parameter
    /// vector `params`) against the model's domain, naming the offending value.
    /// Wired into the fit-time data checks (evaluated on typical values, η = 0) so
    /// an out-of-domain or non-finite `n`/`mtt` is rejected loudly instead of
    /// propagating as a `NaN` through the ODE RHS.
    pub fn validate(&self, params: &[f64]) -> Result<(), String> {
        match self.kind {
            InputRateKind::Transit => {
                validate_transit(self.arg(params, 0, 0.0), self.arg(params, 1, 1.0))
            }
            InputRateKind::InverseGaussian => {
                validate_igd(self.arg(params, 0, 1.0), self.arg(params, 1, 1.0))
            }
            InputRateKind::Weibull => {
                validate_weibull(self.arg(params, 0, 1.0), self.arg(params, 1, 1.0))
            }
            InputRateKind::ZeroOrder => validate_zero_order(self.arg(params, 0, 1.0)),
            InputRateKind::FirstOrder => validate_first_order(self.arg(params, 0, 1.0)),
        }
    }

    /// Build the prepared input-rate constants over `T: PkNum` (e.g.
    /// `T = Dual2<N>` for the analytic ODE sensitivity provider, #430) from the
    /// individual-parameter vector `params` — laid out identically to the `f64`
    /// [`Self::prepare`] input, so `arg_slots` index the same way (and the
    /// per-kind argument defaults match `prepare`, so the lifted constants
    /// reproduce the scalar ones for `T = f64`). The smooth kinds
    /// (inverse-Gaussian, transit, Weibull) are lifted and return `Some`;
    /// [`InputRateKind::ZeroOrder`] returns `None` (its moving-boundary `∂/∂dur`
    /// is not a pointwise `Dual2` expression — see #530), keeping that model on
    /// the FD fallback. [`InputRateKind::supported_over_dual`] gates which kinds
    /// reach here and is pinned consistent with this `match` by
    /// `supported_over_dual_agrees_with_prepare_dual`.
    pub fn prepare_dual<T: PkNum>(&self, params: &[T]) -> Option<PreparedInputRate<T>> {
        match self.kind {
            InputRateKind::Transit => Some(PreparedInputRate::transit(
                self.arg(params, 0, 0.0),
                self.arg(params, 1, 1.0),
            )),
            InputRateKind::InverseGaussian => Some(PreparedInputRate::inverse_gaussian(
                self.arg(params, 0, 1.0),
                self.arg(params, 1, 1.0),
            )),
            InputRateKind::Weibull => Some(PreparedInputRate::weibull(
                self.arg(params, 0, 1.0),
                self.arg(params, 1, 1.0),
            )),
            // Not lifted: the cutoff at `tad = dur` makes `∂/∂dur` a boundary
            // impulse the pointwise dual walk can't carry (#530). FD fallback.
            InputRateKind::ZeroOrder => None,
            // Lifted: a smooth `exp`-only density, exact `Dual2` like the others.
            InputRateKind::FirstOrder => {
                Some(PreparedInputRate::first_order(self.arg(params, 0, 1.0)))
            }
        }
    }
}

/// An input-rate forcing with its dose-invariant constants precomputed for the
/// ODE hot path. Built once per RHS evaluation by [`InputRateForcing::prepare`];
/// [`Self::rate`] then costs only the `tad`/`dose`-dependent arithmetic per dose.
/// Generic over the numeric type `T`: `T = f64` for predictions, `T = Dual2<N>`
/// for the analytic ODE sensitivity provider (#430). The default `T = f64` keeps
/// every existing scalar call site (`PreparedInputRate`) unchanged.
#[derive(Debug, Clone, Copy)]
pub enum PreparedInputRate<T = f64> {
    /// Savic transit constants: `KTR`, `ln KTR`, `n`, and `ln Γ(n + 1)`.
    Transit {
        ktr: T,
        ln_ktr: T,
        n: T,
        ln_gamma_np1: T,
    },
    /// Inverse-Gaussian constants: the mean `mat`, the dose-invariant log
    /// prefactor `c0 = ½·(ln mat − ln 2π − ln cv2)`, and `inv_2cv2mat
    /// = 1/(2·cv2·mat)`. (`cv2` is folded into `c0`/`inv_2cv2mat`, so it is not
    /// stored separately.)
    InverseGaussian { mat: T, c0: T, inv_2cv2mat: T },
    /// Weibull constants: the shape `beta`, `ln Td`, and the dose-invariant log
    /// prefactor `c0 = ln β − ln Td`. (`Td` itself is folded into `ln_td`/`c0`, so
    /// it is not stored separately.)
    Weibull { beta: T, ln_td: T, c0: T },
    /// Zero-order constants: the duration `dur` (kept for the `tad ≤ dur` cutoff)
    /// and its reciprocal `inv_dur = 1/dur` (the constant rate factor, so the
    /// per-dose `rate` is one multiply: `dose · inv_dur`).
    ZeroOrder { dur: T, inv_dur: T },
    /// First-order constant: the absorption rate constant `ka` (the per-dose
    /// `rate` is `dose · ka · exp(−ka·tad)`).
    FirstOrder { ka: T },
}

impl<T: PkNum> PreparedInputRate<T> {
    /// Domain floor for the strictly-positive input-rate parameters (transit
    /// `mtt`; inverse-Gaussian `mat`, `cv2`; Weibull `td`, `beta`) when clamping a
    /// transient mid-fit excursion (see the `transit` / `inverse_gaussian` /
    /// `weibull` constructors). Far below
    /// any realistic value, so it never perturbs a converged fit — it only keeps
    /// a transient `≤ 0` from turning a `.ln()` / `1/x` into a `NaN`/`∞`. The
    /// generic clamp is [`PkNum::guard_floor`], which for `T = f64` is identical
    /// to [`crate::types::clamp_above_floor`] (the modeled-duration floor) for
    /// all inputs incl. `NaN`, so the domain-wall clamps can't drift apart.
    const MIN_PARAM: f64 = 1e-8;

    /// Precompute the inverse-Gaussian constants for `(mat, cv2)`.
    ///
    /// The arguments are **clamped to the valid domain** (`mat > 0`, `cv2 > 0`,
    /// floor [`Self::MIN_PARAM`]) so a transient mid-search excursion (additive
    /// `eta`, wide FD step) yields a finite `R_in` at the domain wall instead of
    /// a `NaN` (`ln`/`1/0`) that would poison the ODE RHS; the converged optimum
    /// is interior, so reported estimates are unaffected. `NaN` inputs also fall
    /// to the floor. Generic over `T` (the `sens/` `*_g<T>` convention) so the
    /// `Dual2` provider gets exact analytic sensitivities for `mat`/`cv2` — the
    /// constants here use only `ln` / `+ − * /`, all on [`PkNum`].
    #[inline]
    fn inverse_gaussian(mat: T, cv2: T) -> Self {
        let mat = mat.guard_floor(Self::MIN_PARAM);
        let cv2 = cv2.guard_floor(Self::MIN_PARAM);
        PreparedInputRate::InverseGaussian {
            mat,
            c0: T::from_f64(0.5) * (mat.ln() - T::from_f64(LN_2PI) - cv2.ln()),
            inv_2cv2mat: T::from_f64(1.0) / (T::from_f64(2.0) * cv2 * mat),
        }
    }

    /// Precompute the transit constants for `(n, mtt)`.
    ///
    /// The arguments are **clamped to the valid domain** (`mtt > 0`, `n ≥ 0`).
    /// The fit-time guard ([`validate_transit`], wired into
    /// `check_absorption_dosing`) already rejects an out-of-domain *typical*
    /// value loudly; but during estimation the inner BFGS perturbs `eta` and the
    /// outer FD step perturbs `theta`, so an additive parameterisation
    /// (`MTT = TVMTT + ETA_MTT`) or a wide FD step can drive a transient
    /// `mtt ≤ 0` / `n < 0` *mid-search*. Left unclamped that yields
    /// `ktr.ln()` / `ln Γ(n+1) = NaN`, which propagates through the ODE RHS into
    /// an opaque `NaN` OFV instead of a recoverable step. Clamping keeps `R_in`
    /// finite at the domain wall so the optimiser can climb back to the interior;
    /// the converged optimum is interior, so reported estimates are unaffected.
    /// `NaN` inputs also fall to the floor (`guard_floor` floors `NaN`, matching
    /// `f64::max`; the `n` branch is false for `NaN`).
    ///
    /// Generic over `T` (the `sens/` `*_g<T>` convention) so a `transit()` model
    /// gets exact analytic `Dual2` sensitivities via [`PkNum::ln_gamma`]
    /// (digamma/trigamma) — #430 slice 2. For `T = f64` this is byte-identical to
    /// the previous f64-only constructor (`guard_floor` ≡ `clamp_above_floor`,
    /// `PkNum::ln_gamma` ≡ `special::ln_gamma`).
    #[inline]
    fn transit(n: T, mtt: T) -> Self {
        let mtt = mtt.guard_floor(Self::MIN_PARAM);
        let n = if n.val() >= 0.0 { n } else { T::from_f64(0.0) };
        let ktr = (n + T::from_f64(1.0)) / mtt;
        PreparedInputRate::Transit {
            ktr,
            ln_ktr: ktr.ln(),
            n,
            ln_gamma_np1: (n + T::from_f64(1.0)).ln_gamma(),
        }
    }

    /// Precompute the Weibull constants for `(td, beta)`.
    ///
    /// Both arguments are **clamped to the valid domain** (`td > 0`, `beta > 0`,
    /// floor [`Self::MIN_PARAM`]) for the same reason as the other constructors: a
    /// transient mid-search excursion (additive `eta`, wide FD step) could drive
    /// `td ≤ 0` or `beta ≤ 0`, turning `ln Td` / `ln β` into a `NaN` that would
    /// poison the ODE RHS. The fit-time guard ([`validate_weibull`]) rejects an
    /// out-of-domain *typical* value loudly; the clamp keeps `R_in` finite at the
    /// domain wall so the optimiser can climb back to the interior, and the
    /// converged optimum is interior so reported estimates are unaffected. `NaN`
    /// inputs also fall to the floor.
    ///
    /// Generic over `T` (the `sens/` `*_g<T>` convention) so a `weibull()` model
    /// gets exact analytic `Dual2` sensitivities for `td`/`beta`: the constants and
    /// [`Self::rate`] use only `ln`/`exp`/`+ − * /`, all on [`PkNum`] — the power
    /// `(tad/Td)^β` is evaluated as `exp(β·ln(tad/Td))`, so no `powf` rule is
    /// needed. For `T = f64` this reduces to the scalar Weibull density.
    #[inline]
    fn weibull(td: T, beta: T) -> Self {
        let td = td.guard_floor(Self::MIN_PARAM);
        let beta = beta.guard_floor(Self::MIN_PARAM);
        let ln_td = td.ln();
        PreparedInputRate::Weibull {
            beta,
            ln_td,
            c0: beta.ln() - ln_td,
        }
    }

    /// Precompute the zero-order constants for `dur`.
    ///
    /// `dur` is **clamped to the valid domain** (`dur > 0`, floor
    /// [`Self::MIN_PARAM`]) for the same reason as the other constructors: a
    /// transient mid-search excursion (additive `eta`, wide FD step) could drive
    /// `dur ≤ 0`, turning `1/dur` into `±∞`/`NaN` that would poison the ODE RHS.
    /// The fit-time guard ([`validate_zero_order`]) rejects an out-of-domain
    /// *typical* value loudly; the clamp keeps `R_in` finite at the domain wall so
    /// the optimiser can climb back to the interior, and the converged optimum is
    /// interior so reported estimates are unaffected. `NaN` falls to the floor.
    ///
    /// Generic over `T` for uniformity with the other constructors, but unlike
    /// them the `dur` parameter's gradient is **not** exact here: the cutoff at
    /// `tad = dur` ([`Self::rate`]) is a moving boundary whose `∂/∂dur` impulse the
    /// pointwise dual walk misses, so a `zero_order()` model is gated to the FD
    /// fallback ([`InputRateKind::supported_over_dual`] = `false`) and this is
    /// reached only for `T = f64` (and the consistency test's `T = f64` probe).
    #[inline]
    fn zero_order(dur: T) -> Self {
        let dur = dur.guard_floor(Self::MIN_PARAM);
        PreparedInputRate::ZeroOrder {
            dur,
            inv_dur: T::from_f64(1.0) / dur,
        }
    }

    /// Precompute the first-order constant for `ka`.
    ///
    /// `ka` is **clamped to the valid domain** (`ka > 0`, floor [`Self::MIN_PARAM`])
    /// for the same reason as the other constructors: a transient mid-search
    /// excursion (additive `eta`, wide FD step) could drive `ka ≤ 0`, turning the
    /// input rate negative / non-physical. The fit-time guard
    /// ([`validate_first_order`]) rejects an out-of-domain *typical* value loudly;
    /// the clamp keeps `R_in` finite and positive at the domain wall so the
    /// optimiser can climb back to the interior, and the converged optimum is
    /// interior so reported estimates are unaffected. `NaN` falls to the floor.
    ///
    /// Generic over `T` (the `sens/` `*_g<T>` convention) so a `first_order()`
    /// model gets exact analytic `Dual2` sensitivities for `ka` and the pathway
    /// fraction: [`Self::rate`] uses only `exp`/`* −`, all on [`PkNum`]. For
    /// `T = f64` this reduces to the scalar first-order (Bateman) input rate.
    #[inline]
    fn first_order(ka: T) -> Self {
        PreparedInputRate::FirstOrder {
            ka: ka.guard_floor(Self::MIN_PARAM),
        }
    }

    /// Appearance rate `R_in(tad)` for one dose (`dose = F · amt`). Per-dose
    /// contributions are summed by the caller; `tad ≤ 0` or `dose ≤ 0 ⇒ 0` (the
    /// guard branches on `.val()`, so for a `Dual2` it returns a flat zero). The
    /// body uses only `ln`/`exp`/`+ − * /`, so it carries exact `Dual2`
    /// sensitivities for `T = Dual2<N>` with no new special function (#430).
    #[inline]
    pub fn rate(&self, tad: T, dose: T) -> T {
        if tad.val() <= 0.0 || dose.val() <= 0.0 {
            return T::from_f64(0.0);
        }
        match *self {
            // ln R_in = ln dose + ln KTR + n·ln(KTR·tad) − KTR·tad − ln Γ(n + 1).
            // For n = 0 the middle term is 0·ln x = 0, reducing to the first-order
            // (Bateman) input dose·KTR·exp(−KTR·tad).
            PreparedInputRate::Transit {
                ktr,
                ln_ktr,
                n,
                ln_gamma_np1,
            } => {
                let x = ktr * tad; // > 0 (tad > 0, ktr > 0 for valid params)
                (dose.ln() + ln_ktr + n * x.ln() - x - ln_gamma_np1).exp()
            }
            // ln R_in = ln dose + ½(ln mat − ln 2π − ln cv2) − 1.5·ln tad
            //           − (tad − mat)² / (2·cv2·mat·tad).
            // tad > 0 here. As tad → 0⁺ the last term → −mat/(2·cv2·tad) = −∞,
            // dominating the +∞ from −1.5·ln tad, so R_in → 0 (the essential
            // singularity); large tad underflows the same way.
            PreparedInputRate::InverseGaussian {
                mat,
                c0,
                inv_2cv2mat,
            } => {
                let d = tad - mat;
                (dose.ln() + c0 - T::from_f64(1.5) * tad.ln() - d * d * inv_2cv2mat / tad).exp()
            }
            // ln R_in = ln dose + (ln β − ln Td) + (β − 1)·u − exp(β·u),
            //   u = ln(tad/Td) = ln tad − ln Td.
            // The power (tad/Td)^β is exp(β·u), so no `powf` is needed. tad > 0
            // here. As tad → 0⁺ (u → −∞): β > 1 ⇒ (β−1)·u → −∞ ⇒ R_in → 0;
            // β = 1 ⇒ R_in = dose/Td·exp(−tad/Td) (first-order, ka = 1/Td);
            // β < 1 ⇒ (β−1)·u → +∞ ⇒ R_in → ∞, an *integrable* spike (∫R_in = dose
            // still holds), which the adaptive solver resolves with small steps.
            PreparedInputRate::Weibull { beta, ln_td, c0 } => {
                let u = tad.ln() - ln_td;
                (dose.ln() + c0 + (beta - T::from_f64(1.0)) * u - (beta * u).exp()).exp()
            }
            // Constant rate `dose/dur` over the window `(0, dur]`, else 0 — a
            // rectangle, so `∫R_in = dose` exactly. `tad > 0` is already ensured by
            // the guard above; the cutoff branches on `.val()`, so for a `Dual2`
            // the inactive side is a flat zero (the boundary jump in `∂/∂dur` is
            // *not* captured — hence the FD gate, #530). The ODE timeline places a
            // break at `tad = dur` so the adaptive solver resolves the step rather
            // than smearing it across one RK45 step.
            PreparedInputRate::ZeroOrder { dur, inv_dur } => {
                if tad.val() <= dur.val() {
                    dose * inv_dur
                } else {
                    T::from_f64(0.0)
                }
            }
            // R_in = dose · ka · exp(−ka·tad) — the first-order (Bateman) input.
            // tad > 0 here (guarded above). Evaluated directly (not in the log
            // domain): the only building block is `exp`, so it carries exact
            // `Dual2` sensitivities for `ka` with no special function. As tad → 0⁺
            // it is finite (R_in → dose·ka); there is no singularity to guard.
            PreparedInputRate::FirstOrder { ka } => dose * ka * (-(ka * tad)).exp(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Coarse trapezoidal `∫₀^upper R_in dt` — enough to check normalisation.
    fn integrate(n: f64, mtt: f64, dose: f64, upper: f64, dt: f64) -> f64 {
        let steps = (upper / dt) as usize;
        let mut sum = 0.0;
        let mut prev = transit_input_rate(0.0, n, mtt, dose);
        for i in 1..=steps {
            let t = i as f64 * dt;
            let cur = transit_input_rate(t, n, mtt, dose);
            sum += 0.5 * (prev + cur) * dt;
            prev = cur;
        }
        sum
    }

    #[test]
    fn transit_mass_balance_integrates_to_dose() {
        // ∫₀^∞ R_in dt = dose across a range of (n, mtt) — the invariant that
        // catches a wrong normalisation constant (the whole point of ln Γ).
        for &(n, mtt) in &[(0.0, 1.0), (1.0, 2.0), (3.0, 1.5), (7.3, 4.0), (20.0, 6.0)] {
            let dose = 100.0;
            let mass = integrate(n, mtt, dose, 80.0, 0.002);
            assert_relative_eq!(mass, dose, max_relative = 2e-3);
        }
    }

    #[test]
    fn transit_n_zero_is_first_order() {
        // n = 0 ⇒ R_in = dose·ktr·exp(−ktr·tad) with ktr = 1/mtt (Bateman input).
        let (mtt, dose) = (2.0_f64, 50.0_f64);
        let ktr = 1.0 / mtt;
        for &tad in &[0.1, 0.5, 1.0, 3.0, 8.0] {
            let want = dose * ktr * (-ktr * tad).exp();
            assert_relative_eq!(
                transit_input_rate(tad, 0.0, mtt, dose),
                want,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn transit_peaks_at_the_gamma_mode() {
        // For n > 0 the chain output peaks at KTR·tad = n ⇒ tad = n·mtt/(n+1).
        let (n, mtt, dose) = (4.0, 3.0, 100.0);
        let mode = n * mtt / (n + 1.0);
        let peak = transit_input_rate(mode, n, mtt, dose);
        assert!(peak > transit_input_rate(mode * 0.5, n, mtt, dose));
        assert!(peak > transit_input_rate(mode * 1.5, n, mtt, dose));
    }

    #[test]
    fn transit_zero_before_dose_and_for_zero_dose() {
        assert_eq!(transit_input_rate(0.0, 3.0, 2.0, 100.0), 0.0);
        assert_eq!(transit_input_rate(-1.0, 3.0, 2.0, 100.0), 0.0);
        assert_eq!(transit_input_rate(1.0, 3.0, 2.0, 0.0), 0.0);
    }

    #[test]
    fn validate_transit_domain() {
        assert!(validate_transit(3.0, 2.0).is_ok());
        assert!(validate_transit(0.0, 1.0).is_ok()); // n = 0 allowed (first-order)
        assert!(validate_transit(3.0, 0.0).is_err());
        assert!(validate_transit(3.0, -1.0).is_err());
        assert!(validate_transit(-1.0, 2.0).is_err());
        assert!(validate_transit(f64::NAN, 2.0).is_err());
        assert!(validate_transit(3.0, f64::NAN).is_err());
    }

    /// A *transient* domain excursion (`mtt ≤ 0`, `n < 0`, or `NaN`) — reachable
    /// mid-fit when an additive `eta` or a wide FD step leaves the domain — must
    /// yield a **finite, non-negative** `R_in`, never a `NaN` that silently
    /// poisons the ODE RHS / OFV. The loud fit-start `validate_transit` still
    /// rejects out-of-domain *typical* values; this guards the search path
    /// (`PreparedInputRate::transit` clamps to the domain).
    #[test]
    fn transit_rate_is_finite_for_domain_excursions() {
        for &(n, mtt) in &[
            (3.0, 0.0),      // mtt = 0  → ktr = +∞ unclamped
            (3.0, -1.0),     // mtt < 0  → ktr < 0, ln(ktr) = NaN unclamped
            (-1.0, 2.0),     // n  < 0
            (f64::NAN, 2.0), // NaN n
            (3.0, f64::NAN), // NaN mtt
        ] {
            for &tad in &[0.5, 2.0, 10.0] {
                let r = transit_input_rate(tad, n, mtt, 100.0);
                assert!(
                    r.is_finite() && r >= 0.0,
                    "R_in must be finite & non-negative at n={n}, mtt={mtt}, tad={tad}, got {r}"
                );
            }
        }
    }

    /// Dual-path counterpart to `transit_rate_is_finite_for_domain_excursions`:
    /// the same transient excursions (`mtt ≤ 0`, `n < 0`, `NaN`), but evaluated
    /// over `Dual2` (here `DualMixed`), must yield a finite **value, gradient, and
    /// Hessian** — not merely a finite `f64` value. This is the failure mode that
    /// turns a mid-search excursion *on the analytic FOCE/FOCEI/Bayes path* into a
    /// `NaN` gradient → `NaN` OFV (the f64 value test above can't see it — it has
    /// no jet). The clamp ([`PreparedInputRate::transit`], via `guard_floor` and
    /// the `n.val() >= 0` branch) makes the clamped region flat, so the **clamped
    /// parameter's gradient entry is exactly zero**; a regression that let
    /// `mtt ≤ 0` / `n < 0` reach `ln` / `ln_gamma` would surface here as a `NaN`/
    /// `∞` jet. `n = 0` (Bateman) is included as an interior case: it is the
    /// `0·ln x` product-rule edge (value 0, jet `∂/∂n = ln x ≠ 0`) and must stay
    /// finite with both parameters' jets live.
    #[test]
    fn transit_dual_jets_finite_at_domain_excursions() {
        use crate::sens::dual_mixed::DualMixed;
        type D = DualMixed<2, 2>;
        let forcing = InputRateForcing {
            cmt: 0,
            kind: InputRateKind::Transit,
            arg_slots: vec![6, 7], // n @ 6 (dim 0), mtt @ 7 (dim 1)
            frac_slot: None,
        };
        // (n, mtt, label, clamped_dim): clamped_dim is the seeded dim whose jet the
        // clamp must zero out (None = interior, both jets live).
        let cases: &[(f64, f64, &str, Option<usize>)] = &[
            (-1.0, 2.0, "n<0", Some(0)),
            (3.0, 0.0, "mtt=0", Some(1)),
            (3.0, -1.0, "mtt<0", Some(1)),
            (f64::NAN, 2.0, "NaN n", Some(0)),
            (3.0, f64::NAN, "NaN mtt", Some(1)),
            (0.0, 2.0, "n=0 Bateman (interior)", None),
        ];
        for &(n, mtt, label, clamped) in cases {
            let mut params = vec![D::constant(0.0); crate::types::MAX_PK_PARAMS];
            params[6] = D::var(n, 0); // seed n   → dim 0
            params[7] = D::var(mtt, 1); // seed mtt → dim 1
            let prep = forcing
                .prepare_dual::<D>(&params)
                .expect("transit lifts over PkNum (slice 2)");
            for &tad in &[0.5, 2.0, 10.0] {
                let r = prep.rate(D::constant(tad), D::constant(100.0));
                assert!(
                    r.value.is_finite(),
                    "{label}: value not finite at tad={tad}: {}",
                    r.value
                );
                assert!(
                    r.grad.iter().all(|g| g.is_finite()),
                    "{label}: gradient not finite at tad={tad}: {:?}",
                    r.grad
                );
                assert!(
                    r.hess.iter().flatten().all(|h| h.is_finite()),
                    "{label}: Hessian not finite at tad={tad}: {:?}",
                    r.hess
                );
                if let Some(d) = clamped {
                    let got = r.grad[d];
                    assert_eq!(
                        got, 0.0,
                        "{label}: clamped dim {d} must have a flat (zero) jet at tad={tad}, got {got}",
                    );
                }
            }
        }
    }

    /// `prepare(...).rate(...)` (the hoisted ODE-hot-path form) must agree bit-for-bit
    /// with the readable reference `transit_input_rate` — guards the two from drifting
    /// and pins the `arg_slots` wiring in `prepare`.
    #[test]
    fn prepared_rate_matches_reference_and_reads_slots() {
        let forcing = InputRateForcing {
            cmt: 0,
            kind: InputRateKind::Transit,
            arg_slots: vec![6, 7], // n @ 6, mtt @ 7
            frac_slot: None,
        };
        let mut params = vec![0.0; crate::types::MAX_PK_PARAMS];
        params[6] = 3.0; // n
        params[7] = 2.0; // mtt
        let prepared = forcing.prepare(&params);
        for &tad in &[0.0, 0.1, 1.0, 4.0, 12.0] {
            assert_eq!(
                prepared.rate(tad, 100.0),
                transit_input_rate(tad, 3.0, 2.0, 100.0)
            );
        }
    }

    /// `InputRateForcing::validate` reads `n`/`mtt` from the right slots and
    /// surfaces the domain error — the hook the fit-time check relies on.
    #[test]
    fn forcing_validate_reads_slots_and_flags_domain() {
        let forcing = InputRateForcing {
            cmt: 0,
            kind: InputRateKind::Transit,
            arg_slots: vec![6, 7],
            frac_slot: None,
        };
        let mut ok = vec![0.0; crate::types::MAX_PK_PARAMS];
        ok[6] = 3.0;
        ok[7] = 2.0;
        assert!(forcing.validate(&ok).is_ok());

        let mut bad_mtt = ok.clone();
        bad_mtt[7] = -1.0; // mtt ≤ 0
        assert!(forcing.validate(&bad_mtt).unwrap_err().contains("mtt"));

        let mut bad_n = ok.clone();
        bad_n[6] = -2.0; // n < 0
        assert!(forcing.validate(&bad_n).unwrap_err().contains("n "));
    }

    // ── Inverse-Gaussian (Freijer & Post) `igd(mat, cv2)` ────────────────────

    /// Direct (non-log-domain) inverse-Gaussian density × dose, the independent
    /// reference the log-domain [`inverse_gaussian_input_rate`] is checked against.
    fn ig_density_ref(tad: f64, mat: f64, cv2: f64, dose: f64) -> f64 {
        if tad <= 0.0 || dose <= 0.0 {
            return 0.0;
        }
        dose * (mat / (std::f64::consts::TAU * cv2 * tad.powi(3))).sqrt()
            * (-(tad - mat).powi(2) / (2.0 * cv2 * mat * tad)).exp()
    }

    /// Inverse-Gaussian mode (peak of the density): `μ·(√(1+(1.5·CV²)²) − 1.5·CV²)`.
    fn ig_mode(mat: f64, cv2: f64) -> f64 {
        let k = 1.5 * cv2;
        mat * ((1.0 + k * k).sqrt() - k)
    }

    /// Coarse trapezoidal `∫₀^upper R_in dt` for the mass-balance invariant.
    fn integrate_ig(mat: f64, cv2: f64, dose: f64, upper: f64, dt: f64) -> f64 {
        let steps = (upper / dt) as usize;
        let mut sum = 0.0;
        let mut prev = inverse_gaussian_input_rate(0.0, mat, cv2, dose);
        for i in 1..=steps {
            let t = i as f64 * dt;
            let cur = inverse_gaussian_input_rate(t, mat, cv2, dose);
            sum += 0.5 * (prev + cur) * dt;
            prev = cur;
        }
        sum
    }

    #[test]
    fn igd_matches_direct_density_formula() {
        // The log-domain evaluation must agree with the textbook density form to
        // machine precision — guards the algebra of the `c0` / `inv_2cv2mat` hoist.
        let dose = 100.0;
        for &(mat, cv2) in &[(2.0, 0.3), (6.0, 1.87), (1.0, 0.5), (4.0, 0.2)] {
            for &tad in &[0.05, 0.5, 1.0, 2.0, 5.0, 12.0, 30.0] {
                assert_relative_eq!(
                    inverse_gaussian_input_rate(tad, mat, cv2, dose),
                    ig_density_ref(tad, mat, cv2, dose),
                    max_relative = 1e-12
                );
            }
        }
    }

    #[test]
    fn igd_mass_balance_integrates_to_dose() {
        // ∫₀^∞ R_in dt = dose: the inverse-Gaussian is a proper density (mean MAT),
        // so the dose is fully delivered. `upper`/`dt` are sized per case from the
        // mean and SD (= MAT·√CV²) to capture the right-skewed tail.
        for &(mat, cv2) in &[
            (2.0_f64, 0.3_f64),
            (6.0, 1.87),
            (1.0, 0.5),
            (4.0, 0.2),
            (3.0, 1.0),
        ] {
            let dose = 100.0;
            let sd = mat * cv2.sqrt();
            let upper = mat + 40.0 * sd;
            let dt = (ig_mode(mat, cv2).min(sd)) / 400.0;
            let mass = integrate_ig(mat, cv2, dose, upper, dt);
            assert_relative_eq!(mass, dose, max_relative = 2e-3);
        }
    }

    #[test]
    fn igd_peaks_at_the_ig_mode() {
        // The density peaks at its mode; flanks at half / 1.5× the mode are lower.
        for &(mat, cv2) in &[(2.0, 0.3), (6.0, 1.0), (4.0, 0.2)] {
            let mode = ig_mode(mat, cv2);
            let peak = inverse_gaussian_input_rate(mode, mat, cv2, 100.0);
            assert!(peak > inverse_gaussian_input_rate(mode * 0.5, mat, cv2, 100.0));
            assert!(peak > inverse_gaussian_input_rate(mode * 1.5, mat, cv2, 100.0));
        }
    }

    #[test]
    fn igd_zero_before_dose_and_for_zero_dose() {
        assert_eq!(inverse_gaussian_input_rate(0.0, 2.0, 0.3, 100.0), 0.0);
        assert_eq!(inverse_gaussian_input_rate(-1.0, 2.0, 0.3, 100.0), 0.0);
        assert_eq!(inverse_gaussian_input_rate(1.0, 2.0, 0.3, 0.0), 0.0);
    }

    #[test]
    fn igd_essential_singularity_vanishes_at_tiny_tad() {
        // The essential singularity at tad → 0⁺ collapses to R_in → 0 (not NaN/∞):
        // values must be finite, non-negative, and far below the peak as tad shrinks.
        let (mat, cv2) = (2.0, 0.3);
        let peak = inverse_gaussian_input_rate(ig_mode(mat, cv2), mat, cv2, 100.0);
        for &tad in &[1e-2, 1e-4, 1e-6, 1e-10, 1e-300] {
            let r = inverse_gaussian_input_rate(tad, mat, cv2, 100.0);
            assert!(
                r.is_finite() && r >= 0.0,
                "R_in must be finite ≥ 0, got {r}"
            );
            assert!(
                r < peak,
                "R_in at tiny tad={tad} ({r}) must stay below the peak"
            );
        }
        assert_eq!(inverse_gaussian_input_rate(1e-300, mat, cv2, 100.0), 0.0);
    }

    #[test]
    fn validate_igd_domain() {
        assert!(validate_igd(2.0, 0.3).is_ok());
        assert!(validate_igd(0.0, 0.3).is_err());
        assert!(validate_igd(-1.0, 0.3).is_err());
        assert!(validate_igd(2.0, 0.0).is_err());
        assert!(validate_igd(2.0, -0.3).is_err());
        assert!(validate_igd(f64::NAN, 0.3).is_err());
        assert!(validate_igd(2.0, f64::NAN).is_err());
    }

    /// A transient domain excursion (`mat ≤ 0`, `cv2 ≤ 0`, or `NaN`) must yield a
    /// finite, non-negative `R_in` (the clamp in `PreparedInputRate::inverse_gaussian`),
    /// never a `NaN`/`∞` poisoning the ODE RHS — the IG analogue of the transit guard.
    #[test]
    fn igd_rate_is_finite_for_domain_excursions() {
        for &(mat, cv2) in &[
            (0.0, 0.3),
            (-1.0, 0.3),
            (2.0, 0.0),
            (2.0, -0.3),
            (f64::NAN, 0.3),
            (2.0, f64::NAN),
        ] {
            for &tad in &[0.5, 2.0, 10.0] {
                let r = inverse_gaussian_input_rate(tad, mat, cv2, 100.0);
                assert!(
                    r.is_finite() && r >= 0.0,
                    "R_in must be finite & non-negative at mat={mat}, cv2={cv2}, tad={tad}, got {r}"
                );
            }
        }
    }

    /// `prepare(...).rate(...)` must agree bit-for-bit with the reference
    /// `inverse_gaussian_input_rate`, and read `mat`/`cv2` from the right slots.
    #[test]
    fn prepared_igd_rate_matches_reference_and_reads_slots() {
        let forcing = InputRateForcing {
            cmt: 1,
            kind: InputRateKind::InverseGaussian,
            arg_slots: vec![4, 5], // mat @ 4, cv2 @ 5
            frac_slot: None,
        };
        let mut params = vec![0.0; crate::types::MAX_PK_PARAMS];
        params[4] = 2.0; // mat
        params[5] = 0.3; // cv2
        let prepared = forcing.prepare(&params);
        for &tad in &[0.0, 0.1, 1.0, 4.0, 12.0] {
            assert_eq!(
                prepared.rate(tad, 100.0),
                inverse_gaussian_input_rate(tad, 2.0, 0.3, 100.0)
            );
        }
    }

    /// `InputRateForcing::validate` reads `mat`/`cv2` from the right slots for the
    /// IG kind and surfaces the domain error.
    #[test]
    fn forcing_validate_igd_reads_slots_and_flags_domain() {
        let forcing = InputRateForcing {
            cmt: 1,
            kind: InputRateKind::InverseGaussian,
            arg_slots: vec![4, 5],
            frac_slot: None,
        };
        let mut ok = vec![0.0; crate::types::MAX_PK_PARAMS];
        ok[4] = 2.0;
        ok[5] = 0.3;
        assert!(forcing.validate(&ok).is_ok());

        let mut bad_mat = ok.clone();
        bad_mat[4] = -1.0;
        assert!(forcing.validate(&bad_mat).unwrap_err().contains("mat"));

        let mut bad_cv2 = ok.clone();
        bad_cv2[5] = 0.0;
        assert!(forcing.validate(&bad_cv2).unwrap_err().contains("cv2"));
    }

    // ── Weibull `weibull(td, beta)` ──────────────────────────────────────────

    /// Direct (non-log-domain) Weibull density × dose, via `powf` — the independent
    /// reference the log-domain [`weibull_input_rate`] (which uses `exp(β·ln x)`) is
    /// checked against, so a typo in either route is caught.
    fn weibull_density_ref(tad: f64, td: f64, beta: f64, dose: f64) -> f64 {
        if tad <= 0.0 || dose <= 0.0 {
            return 0.0;
        }
        let x = tad / td;
        dose * (beta / td) * x.powf(beta - 1.0) * (-x.powf(beta)).exp()
    }

    /// Analytic Weibull survival `S(t) = exp(−(t/Td)^β)`; `dose·(1 − S)` is the
    /// cumulative absorbed amount, and `R_in = −dose·S'(t)`.
    fn weibull_survival(t: f64, td: f64, beta: f64) -> f64 {
        (-(t / td).powf(beta)).exp()
    }

    #[test]
    fn weibull_matches_direct_density_formula() {
        // Log-domain evaluation vs the textbook `powf` density to machine precision,
        // across β < 1, β = 1, β > 1 — guards the `c0`/`ln_td` hoist algebra and the
        // `(tad/Td)^β = exp(β·ln(tad/Td))` rewrite.
        let dose = 100.0;
        for &(td, beta) in &[(2.0, 0.5), (1.5, 1.0), (3.0, 1.5), (2.0, 3.0), (4.0, 0.8)] {
            for &tad in &[0.05, 0.5, 1.0, 2.0, 5.0, 12.0, 30.0] {
                assert_relative_eq!(
                    weibull_input_rate(tad, td, beta, dose),
                    weibull_density_ref(tad, td, beta, dose),
                    max_relative = 1e-12
                );
            }
        }
    }

    #[test]
    fn weibull_is_negative_derivative_of_survival() {
        // The strongest normalisation check: R_in(tad) = −dose·S'(t), with
        // S = exp(−(t/Td)^β). Pins the whole functional form (constant *and*
        // exponent), and — unlike a trapezoidal mass integral — stays accurate for
        // β < 1 because it is evaluated pointwise at finite tad (no singular cell).
        let dose = 100.0;
        let h = 1e-6;
        for &(td, beta) in &[(2.0, 0.7), (1.5, 1.0), (3.0, 1.4), (2.0, 2.5)] {
            for &tad in &[0.2, 0.6, 1.5, 4.0, 9.0] {
                let s_prime = (weibull_survival(tad + h, td, beta)
                    - weibull_survival(tad - h, td, beta))
                    / (2.0 * h);
                assert_relative_eq!(
                    weibull_input_rate(tad, td, beta, dose),
                    -dose * s_prime,
                    max_relative = 1e-5
                );
            }
        }
    }

    #[test]
    fn weibull_mass_balance_integrates_to_dose() {
        // ∫₀^∞ R_in dt = dose for β ≥ 1, where the integrand is finite at tad → 0
        // so a plain trapezoidal sum from 0 converges. (β < 1 has an integrable
        // singularity at 0 that trapezoidal-from-0 cannot resolve; its normalisation
        // is covered exactly by `weibull_is_negative_derivative_of_survival` and the
        // integrable-spike test below.)
        for &(td, beta) in &[(1.5_f64, 1.0_f64), (3.0, 1.5), (2.0, 3.0), (4.0, 2.0)] {
            let dose = 100.0;
            let upper = td * 8.0;
            let dt = upper / 400_000.0;
            let steps = (upper / dt) as usize;
            let mut sum = 0.0;
            let mut prev = weibull_input_rate(0.0, td, beta, dose);
            for i in 1..=steps {
                let t = i as f64 * dt;
                let cur = weibull_input_rate(t, td, beta, dose);
                sum += 0.5 * (prev + cur) * dt;
                prev = cur;
            }
            assert_relative_eq!(sum, dose, max_relative = 2e-3);
        }
    }

    #[test]
    fn weibull_beta_one_is_first_order() {
        // β = 1 ⇒ R_in = dose/Td·exp(−tad/Td): first-order absorption with ka = 1/Td.
        let (td, dose) = (2.0_f64, 50.0_f64);
        let ka = 1.0 / td;
        for &tad in &[0.0, 0.1, 0.5, 1.0, 3.0, 8.0] {
            let want = if tad <= 0.0 {
                0.0
            } else {
                dose * ka * (-ka * tad).exp()
            };
            assert_relative_eq!(
                weibull_input_rate(tad, td, 1.0, dose),
                want,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn weibull_peaks_at_the_mode_for_shape_gt_one() {
        // For β > 1 the density has an interior mode at tad = Td·((β−1)/β)^(1/β);
        // flanks at half / 1.5× the mode are lower.
        for &(td, beta) in &[(2.0_f64, 1.5_f64), (3.0, 2.0), (1.5, 4.0)] {
            let mode = td * ((beta - 1.0) / beta).powf(1.0 / beta);
            let peak = weibull_input_rate(mode, td, beta, 100.0);
            assert!(peak > weibull_input_rate(mode * 0.5, td, beta, 100.0));
            assert!(peak > weibull_input_rate(mode * 1.5, td, beta, 100.0));
        }
    }

    #[test]
    fn weibull_beta_lt_one_integrable_spike() {
        // β < 1: R_in → ∞ as tad → 0⁺, but the spike is *integrable*. Concretely,
        // as tad shrinks toward 0: (i) R_in is finite & positive at every tad > 0;
        // (ii) R_in rises (the value is *not* capped — a cap would break mass
        // balance); (iii) the cumulative absorbed amount dose·(1 − S(t)) *falls*
        // toward 0 — so the mass under the ever-taller spike vanishes as the window
        // shrinks. The integral, not the value, is what stays finite.
        let (td, beta, dose) = (2.0_f64, 0.5_f64, 100.0_f64);
        let mut prev_r = 0.0_f64; // rises as tad shrinks
        let mut prev_absorbed = f64::INFINITY; // falls as tad shrinks
        for &tad in &[1e-1, 1e-3, 1e-6, 1e-9] {
            let r = weibull_input_rate(tad, td, beta, dose);
            assert!(
                r.is_finite() && r > 0.0,
                "R_in must be finite > 0 at tad={tad}, got {r}"
            );
            assert!(
                r > prev_r,
                "R_in must rise toward the tad→0 singularity (tad={tad}: {r} vs prev {prev_r})"
            );
            prev_r = r;

            let absorbed = dose * (1.0 - weibull_survival(tad, td, beta));
            assert!(
                absorbed < prev_absorbed,
                "cumulative absorbed must fall toward 0 as tad→0 (tad={tad}: {absorbed} vs prev {prev_absorbed})"
            );
            prev_absorbed = absorbed;
        }
        // At the smallest tad the absorbed mass is a vanishing fraction of the dose,
        // confirming the singularity is integrable (∫₀^t R_in → 0).
        assert!(dose * (1.0 - weibull_survival(1e-9, td, beta)) < dose * 1e-3);
        // tad = 0 itself is the guarded start (input begins at the dose), R_in = 0.
        assert_eq!(weibull_input_rate(0.0, td, beta, dose), 0.0);
    }

    #[test]
    fn weibull_zero_before_dose_and_for_zero_dose() {
        assert_eq!(weibull_input_rate(0.0, 2.0, 1.5, 100.0), 0.0);
        assert_eq!(weibull_input_rate(-1.0, 2.0, 1.5, 100.0), 0.0);
        assert_eq!(weibull_input_rate(1.0, 2.0, 1.5, 0.0), 0.0);
    }

    #[test]
    fn validate_weibull_domain() {
        assert!(validate_weibull(2.0, 1.5).is_ok());
        assert!(validate_weibull(2.0, 0.5).is_ok()); // β < 1 is valid
        assert!(validate_weibull(0.0, 1.5).is_err());
        assert!(validate_weibull(-1.0, 1.5).is_err());
        assert!(validate_weibull(2.0, 0.0).is_err());
        assert!(validate_weibull(2.0, -0.5).is_err());
        assert!(validate_weibull(f64::NAN, 1.5).is_err());
        assert!(validate_weibull(2.0, f64::NAN).is_err());
    }

    /// A transient domain excursion (`td ≤ 0`, `beta ≤ 0`, or `NaN`) — reachable
    /// mid-fit via an additive `eta` / wide FD step — must yield a finite,
    /// non-negative `R_in` (the clamp in `PreparedInputRate::weibull`), never a
    /// `NaN`/`∞` poisoning the ODE RHS.
    #[test]
    fn weibull_rate_is_finite_for_domain_excursions() {
        for &(td, beta) in &[
            (0.0, 1.5),
            (-1.0, 1.5),
            (2.0, 0.0),
            (2.0, -0.5),
            (f64::NAN, 1.5),
            (2.0, f64::NAN),
        ] {
            for &tad in &[0.5, 2.0, 10.0] {
                let r = weibull_input_rate(tad, td, beta, 100.0);
                assert!(
                    r.is_finite() && r >= 0.0,
                    "R_in must be finite & non-negative at td={td}, beta={beta}, tad={tad}, got {r}"
                );
            }
        }
    }

    /// Dual-path counterpart: the same transient excursions over `Dual2` must yield
    /// a finite **value, gradient, and Hessian** (the analytic FOCE/FOCEI/Bayes
    /// failure mode the f64 test can't see), and the clamped parameter's jet must be
    /// exactly flat (zero gradient) since the clamp pins it to a constant.
    #[test]
    fn weibull_dual_jets_finite_at_domain_excursions() {
        use crate::sens::dual_mixed::DualMixed;
        type D = DualMixed<2, 2>;
        let forcing = InputRateForcing {
            cmt: 1,
            kind: InputRateKind::Weibull,
            arg_slots: vec![4, 5], // td @ 4 (dim 0), beta @ 5 (dim 1)
            frac_slot: None,
        };
        // (td, beta, label, clamped_dim)
        let cases: &[(f64, f64, &str, Option<usize>)] = &[
            (0.0, 1.5, "td=0", Some(0)),
            (-1.0, 1.5, "td<0", Some(0)),
            (2.0, 0.0, "beta=0", Some(1)),
            (2.0, -0.5, "beta<0", Some(1)),
            (f64::NAN, 1.5, "NaN td", Some(0)),
            (2.0, f64::NAN, "NaN beta", Some(1)),
            (2.0, 1.5, "interior", None),
        ];
        for &(td, beta, label, clamped) in cases {
            let mut params = vec![D::constant(0.0); crate::types::MAX_PK_PARAMS];
            params[4] = D::var(td, 0);
            params[5] = D::var(beta, 1);
            let prep = forcing
                .prepare_dual::<D>(&params)
                .expect("weibull lifts over PkNum (Phase 2)");
            for &tad in &[0.5, 2.0, 10.0] {
                let r = prep.rate(D::constant(tad), D::constant(100.0));
                assert!(
                    r.value.is_finite(),
                    "{label}: value not finite at tad={tad}: {}",
                    r.value
                );
                assert!(
                    r.grad.iter().all(|g| g.is_finite()),
                    "{label}: gradient not finite at tad={tad}: {:?}",
                    r.grad
                );
                assert!(
                    r.hess.iter().flatten().all(|h| h.is_finite()),
                    "{label}: Hessian not finite at tad={tad}: {:?}",
                    r.hess
                );
                if let Some(d) = clamped {
                    assert_eq!(
                        r.grad[d], 0.0,
                        "{label}: clamped dim {d} must have a flat jet at tad={tad}, got {}",
                        r.grad[d]
                    );
                }
            }
        }
    }

    /// The analytic `Dual2` gradient `∂R_in/∂(td, beta)` must match central finite
    /// differences of the scalar `weibull_input_rate` at interior points — the
    /// per-PR backstop (default `--features ci`, since `Dual2` is the `gradient =
    /// auto` path) against a wrong log-domain dual rule, the #317-class regression
    /// FD-only checks miss. Covers β < 1, β = 1, β > 1.
    #[test]
    fn weibull_dual_gradient_matches_central_fd() {
        use crate::sens::dual_mixed::DualMixed;
        type D = DualMixed<2, 2>;
        let forcing = InputRateForcing {
            cmt: 1,
            kind: InputRateKind::Weibull,
            arg_slots: vec![4, 5], // td @ 4 (dim 0), beta @ 5 (dim 1)
            frac_slot: None,
        };
        let dose = 100.0;
        let h = 1e-6;
        for &(td, beta) in &[(1.5, 0.8), (2.0, 1.0), (3.0, 1.5), (2.0, 2.5)] {
            let mut params = vec![D::constant(0.0); crate::types::MAX_PK_PARAMS];
            params[4] = D::var(td, 0);
            params[5] = D::var(beta, 1);
            let prep = forcing.prepare_dual::<D>(&params).unwrap();
            for &tad in &[0.3, 1.0, 4.0] {
                let r = prep.rate(D::constant(tad), D::constant(dose));
                let d_td = (weibull_input_rate(tad, td + h, beta, dose)
                    - weibull_input_rate(tad, td - h, beta, dose))
                    / (2.0 * h);
                let d_beta = (weibull_input_rate(tad, td, beta + h, dose)
                    - weibull_input_rate(tad, td, beta - h, dose))
                    / (2.0 * h);
                assert_relative_eq!(r.grad[0], d_td, max_relative = 1e-5, epsilon = 1e-7);
                assert_relative_eq!(r.grad[1], d_beta, max_relative = 1e-5, epsilon = 1e-7);
            }
        }
    }

    /// `prepare(...).rate(...)` must agree bit-for-bit with the reference
    /// `weibull_input_rate`, reading `td`/`beta` from the right slots.
    #[test]
    fn prepared_weibull_rate_matches_reference_and_reads_slots() {
        let forcing = InputRateForcing {
            cmt: 1,
            kind: InputRateKind::Weibull,
            arg_slots: vec![4, 5], // td @ 4, beta @ 5
            frac_slot: None,
        };
        let mut params = vec![0.0; crate::types::MAX_PK_PARAMS];
        params[4] = 2.0; // td
        params[5] = 1.5; // beta
        let prepared = forcing.prepare(&params);
        for &tad in &[0.0, 0.1, 1.0, 4.0, 12.0] {
            assert_eq!(
                prepared.rate(tad, 100.0),
                weibull_input_rate(tad, 2.0, 1.5, 100.0)
            );
        }
    }

    /// `InputRateForcing::validate` reads `td`/`beta` from the right slots for the
    /// Weibull kind and surfaces the domain error.
    #[test]
    fn forcing_validate_weibull_reads_slots_and_flags_domain() {
        let forcing = InputRateForcing {
            cmt: 1,
            kind: InputRateKind::Weibull,
            arg_slots: vec![4, 5],
            frac_slot: None,
        };
        let mut ok = vec![0.0; crate::types::MAX_PK_PARAMS];
        ok[4] = 2.0;
        ok[5] = 1.5;
        assert!(forcing.validate(&ok).is_ok());

        let mut bad_td = ok.clone();
        bad_td[4] = -1.0;
        assert!(forcing.validate(&bad_td).unwrap_err().contains("td"));

        let mut bad_beta = ok.clone();
        bad_beta[5] = 0.0;
        assert!(forcing.validate(&bad_beta).unwrap_err().contains("beta"));
    }

    // ── Zero-order `zero_order(dur)` ─────────────────────────────────────────

    #[test]
    fn zero_order_is_constant_over_window_then_zero() {
        // R_in = dose/dur on (0, dur], 0 after — a rectangle.
        let (dur, dose) = (4.0_f64, 100.0_f64);
        let rate = dose / dur;
        for &tad in &[0.001, 1.0, 2.0, dur - 1e-9, dur] {
            assert_relative_eq!(
                zero_order_input_rate(tad, dur, dose),
                rate,
                max_relative = 1e-12
            );
        }
        // Strictly past the cutoff the input has stopped.
        for &tad in &[dur + 1e-9, dur + 1.0, 100.0] {
            assert_eq!(zero_order_input_rate(tad, dur, dose), 0.0);
        }
    }

    #[test]
    fn zero_order_mass_balance_is_exact() {
        // ∫₀^∞ R_in dt = (dose/dur)·dur = dose, exactly (it's a rectangle, no
        // quadrature error). Catches a wrong normalisation/window.
        for &(dur, dose) in &[(1.0, 50.0), (4.0, 100.0), (0.25, 10.0), (12.0, 250.0)] {
            assert_relative_eq!(zero_order_input_rate(1e-9, dur, dose) * dur, dose);
        }
    }

    #[test]
    fn zero_order_zero_before_dose_and_for_zero_dose() {
        assert_eq!(zero_order_input_rate(0.0, 4.0, 100.0), 0.0);
        assert_eq!(zero_order_input_rate(-1.0, 4.0, 100.0), 0.0);
        assert_eq!(zero_order_input_rate(1.0, 4.0, 0.0), 0.0);
    }

    #[test]
    fn validate_zero_order_domain() {
        assert!(validate_zero_order(4.0).is_ok());
        assert!(validate_zero_order(0.0).is_err());
        assert!(validate_zero_order(-1.0).is_err());
        assert!(validate_zero_order(f64::NAN).is_err());
    }

    /// A transient domain excursion (`dur ≤ 0` or `NaN`) — reachable mid-fit via an
    /// additive `eta` / wide FD step — must yield a finite, non-negative `R_in` (the
    /// clamp in `PreparedInputRate::zero_order`), never the `1/0 = ∞` / `NaN` that
    /// would poison the ODE RHS.
    #[test]
    fn zero_order_rate_is_finite_for_domain_excursions() {
        for &dur in &[0.0, -1.0, f64::NAN] {
            for &tad in &[0.5, 2.0, 10.0] {
                let r = zero_order_input_rate(tad, dur, 100.0);
                assert!(
                    r.is_finite() && r >= 0.0,
                    "R_in must be finite & non-negative at dur={dur}, tad={tad}, got {r}"
                );
            }
        }
    }

    /// `prepare(...).rate(...)` (the hoisted ODE-hot-path form) must agree
    /// bit-for-bit with the reference `zero_order_input_rate`, reading `dur` from
    /// the right (single) slot.
    #[test]
    fn prepared_zero_order_rate_matches_reference_and_reads_slot() {
        let forcing = InputRateForcing {
            cmt: 0,
            kind: InputRateKind::ZeroOrder,
            arg_slots: vec![4], // dur @ 4
            frac_slot: None,
        };
        let mut params = vec![0.0; crate::types::MAX_PK_PARAMS];
        params[4] = 4.0; // dur
        let prepared = forcing.prepare(&params);
        for &tad in &[0.0, 0.1, 1.0, 4.0, 4.5, 12.0] {
            assert_eq!(
                prepared.rate(tad, 100.0),
                zero_order_input_rate(tad, 4.0, 100.0)
            );
        }
    }

    /// `InputRateForcing::validate` reads `dur` from the right slot for the
    /// zero-order kind and surfaces the domain error.
    #[test]
    fn forcing_validate_zero_order_reads_slot_and_flags_domain() {
        let forcing = InputRateForcing {
            cmt: 0,
            kind: InputRateKind::ZeroOrder,
            arg_slots: vec![4],
            frac_slot: None,
        };
        let mut ok = vec![0.0; crate::types::MAX_PK_PARAMS];
        ok[4] = 4.0;
        assert!(forcing.validate(&ok).is_ok());

        let mut bad = ok.clone();
        bad[4] = -1.0;
        assert!(forcing.validate(&bad).unwrap_err().contains("dur"));
    }

    /// Zero-order is **not** lifted over `Dual2` (its moving-boundary `∂/∂dur` is
    /// FD-only, #530): the gate must say so and `prepare_dual` must return `None`,
    /// so the ODE provider keeps a `zero_order()` subject on the FD fallback.
    #[test]
    fn zero_order_is_not_lifted_over_dual() {
        assert!(!InputRateKind::ZeroOrder.supported_over_dual());
        let forcing = InputRateForcing {
            cmt: 0,
            kind: InputRateKind::ZeroOrder,
            arg_slots: vec![4],
            frac_slot: None,
        };
        let params = vec![1.0; crate::types::MAX_PK_PARAMS];
        assert!(forcing.prepare_dual::<f64>(&params).is_none());
    }

    /// `prepare_dual` lifts the **smooth** forcings (IG, transit, Weibull,
    /// first-order) to a `PkNum` type (here `T = f64`), reproducing the scalar
    /// `prepare` exactly — the `T = f64` byte-identity that lets the analytic ODE
    /// provider evaluate them over `Dual2` without drifting from the production
    /// predictor (#430; Weibull = Phase 2; first-order = #505). Zero-order is the
    /// one non-lifted kind, checked separately (`zero_order_is_not_lifted_over_dual`).
    #[test]
    fn prepare_dual_lifts_all_kinds() {
        let mut params = vec![0.0; crate::types::MAX_PK_PARAMS];
        params[4] = 2.0; // mat / td
        params[5] = 0.3; // cv2 / (beta below uses its own slots)
        params[6] = 3.0; // n
        params[7] = 1.0; // mtt
        params[2] = 1.5; // beta (td reuses slot 4)
        params[3] = 1.2; // ka (first-order)

        let ig = InputRateForcing {
            cmt: 1,
            kind: InputRateKind::InverseGaussian,
            arg_slots: vec![4, 5], // mat @ 4, cv2 @ 5
            frac_slot: None,
        };
        let transit = InputRateForcing {
            cmt: 0,
            kind: InputRateKind::Transit,
            arg_slots: vec![6, 7], // n @ 6, mtt @ 7
            frac_slot: None,
        };
        let weibull = InputRateForcing {
            cmt: 0,
            kind: InputRateKind::Weibull,
            arg_slots: vec![4, 2], // td @ 4, beta @ 2
            frac_slot: None,
        };
        let first_order = InputRateForcing {
            cmt: 1,
            kind: InputRateKind::FirstOrder,
            arg_slots: vec![3], // ka @ 3
            frac_slot: None,
        };
        for forcing in [&ig, &transit, &weibull, &first_order] {
            let lifted = forcing
                .prepare_dual::<f64>(&params)
                .expect("all kinds lift over PkNum");
            let scalar = forcing.prepare(&params);
            for &tad in &[0.0, 0.1, 1.0, 4.0, 12.0] {
                assert_eq!(lifted.rate(tad, 100.0), scalar.rate(tad, 100.0));
            }
        }
    }

    /// Drift tripwire: `InputRateKind::supported_over_dual` (the gate the ODE
    /// provider reads) must agree with whether `prepare_dual` actually lifts the
    /// kind. A kind marked supported but returning `None` would let
    /// `ode_analytical_supported` admit the model, then the `?` in
    /// `integrate_subject_duals` silently bails the whole subject to FD with no
    /// error. Adding a kind: extend `ALL_KINDS` here too (#430 review #5 / #451).
    #[test]
    fn supported_over_dual_agrees_with_prepare_dual() {
        const ALL_KINDS: &[InputRateKind] = &[
            InputRateKind::InverseGaussian,
            InputRateKind::Transit,
            InputRateKind::Weibull,
            InputRateKind::ZeroOrder,
            InputRateKind::FirstOrder,
        ];
        let params = vec![1.0; crate::types::MAX_PK_PARAMS];
        for &kind in ALL_KINDS {
            let forcing = InputRateForcing {
                cmt: 1,
                kind,
                arg_slots: vec![4, 5],
                frac_slot: None,
            };
            assert_eq!(
                kind.supported_over_dual(),
                forcing.prepare_dual::<f64>(&params).is_some(),
                "supported_over_dual must match prepare_dual liftability for {kind:?}"
            );
        }
    }

    // ── First-order `first_order(ka)` ─────────────────────────────────────────

    #[test]
    fn first_order_matches_bateman_input_and_is_zero_before_dose() {
        // R_in = dose·ka·exp(−ka·tad) for tad > 0; 0 for tad ≤ 0 and dose ≤ 0.
        let (ka, dose) = (1.3_f64, 80.0_f64);
        for &tad in &[0.1, 0.5, 1.0, 3.0, 8.0] {
            let want = dose * ka * (-ka * tad).exp();
            assert_relative_eq!(
                first_order_input_rate(tad, ka, dose),
                want,
                max_relative = 1e-12
            );
        }
        // The input starts at the dose (no negative-tad leak) and a non-positive
        // dose contributes nothing.
        assert_eq!(first_order_input_rate(0.0, ka, dose), 0.0);
        assert_eq!(first_order_input_rate(-1.0, ka, dose), 0.0);
        assert_eq!(first_order_input_rate(1.0, ka, 0.0), 0.0);
        // The peak rate is at the dose instant (finite, no singularity): dose·ka.
        assert_relative_eq!(
            first_order_input_rate(1e-9, ka, dose),
            dose * ka,
            max_relative = 1e-6
        );
    }

    #[test]
    fn first_order_mass_balance_integrates_to_dose() {
        // ∫₀^∞ dose·ka·e^{−ka·tad} dt = dose. The integrand is finite at tad → 0,
        // so a plain trapezoidal sum from 0 converges (like the β ≥ 1 Weibull case).
        for &ka in &[0.5_f64, 1.0, 2.5] {
            let dose = 100.0;
            let upper = 30.0 / ka; // ≫ several 1/ka time-constants
            let dt = upper / 400_000.0;
            let steps = (upper / dt) as usize;
            let mut sum = 0.0;
            let mut prev = first_order_input_rate(0.0, ka, dose);
            for i in 1..=steps {
                let t = i as f64 * dt;
                let cur = first_order_input_rate(t, ka, dose);
                sum += 0.5 * (prev + cur) * dt;
                prev = cur;
            }
            assert_relative_eq!(sum, dose, max_relative = 2e-3);
        }
    }

    #[test]
    fn first_order_equals_transit_n_zero() {
        // first_order(ka) is exactly transit(n = 0, mtt = 1/ka): with n = 0 the
        // transit chain's gamma density collapses to the Bateman input KTR·e^{−KTR·tad}
        // with KTR = (0+1)/mtt = ka. A cheap cross-check that the new kind agrees with
        // the established transit evaluator.
        for &ka in &[0.4_f64, 1.0, 2.0] {
            let mtt = 1.0 / ka;
            for &tad in &[0.0, 0.25, 1.0, 4.0, 10.0] {
                assert_relative_eq!(
                    first_order_input_rate(tad, ka, 100.0),
                    transit_input_rate(tad, 0.0, mtt, 100.0),
                    max_relative = 1e-12,
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    fn validate_first_order_domain() {
        assert!(validate_first_order(1.0).is_ok());
        assert!(validate_first_order(0.0).unwrap_err().contains("ka"));
        assert!(validate_first_order(-1.0).is_err());
        assert!(validate_first_order(f64::NAN).is_err());
    }

    /// A transient domain excursion (`ka ≤ 0` or `NaN`) — reachable mid-fit via an
    /// additive `eta` / wide FD step — must yield a finite, non-negative `R_in` (the
    /// clamp in `PreparedInputRate::first_order`), never a `NaN`/negative rate.
    #[test]
    fn first_order_rate_is_finite_for_domain_excursions() {
        for &ka in &[0.0, -1.0, f64::NAN] {
            for &tad in &[0.5, 2.0, 10.0] {
                let r = first_order_input_rate(tad, ka, 100.0);
                assert!(
                    r.is_finite() && r >= 0.0,
                    "R_in must be finite & non-negative at ka={ka}, tad={tad}, got {r}"
                );
            }
        }
    }

    /// `prepare(...).rate(...)` (the hoisted ODE-hot-path form) must agree
    /// bit-for-bit with the reference `first_order_input_rate`, reading `ka` from the
    /// right (single) slot.
    #[test]
    fn prepared_first_order_rate_matches_reference_and_reads_slot() {
        let forcing = InputRateForcing {
            cmt: 0,
            kind: InputRateKind::FirstOrder,
            arg_slots: vec![3], // ka @ 3
            frac_slot: None,
        };
        let mut params = vec![0.0; crate::types::MAX_PK_PARAMS];
        params[3] = 1.1; // ka
        let prepared = forcing.prepare(&params);
        for &tad in &[0.0, 0.1, 1.0, 4.0, 12.0] {
            assert_eq!(
                prepared.rate(tad, 100.0),
                first_order_input_rate(tad, 1.1, 100.0)
            );
        }
    }

    /// `InputRateForcing::validate` reads `ka` from the right slot for the
    /// first-order kind and surfaces the domain error.
    #[test]
    fn forcing_validate_first_order_reads_slot_and_flags_domain() {
        let forcing = InputRateForcing {
            cmt: 0,
            kind: InputRateKind::FirstOrder,
            arg_slots: vec![3],
            frac_slot: None,
        };
        let mut ok = vec![0.0; crate::types::MAX_PK_PARAMS];
        ok[3] = 1.0;
        assert!(forcing.validate(&ok).is_ok());
        let mut bad = ok.clone();
        bad[3] = -1.0;
        assert!(forcing.validate(&bad).unwrap_err().contains("ka"));
    }

    /// First-order **is** lifted over `Dual2` (a smooth `exp`-only density): the
    /// gate says so and `prepare_dual` returns `Some`, so a pure-`first_order()`
    /// (e.g. `parallel`) model gets exact analytic ODE sensitivities (#505).
    #[test]
    fn first_order_is_lifted_over_dual() {
        assert!(InputRateKind::FirstOrder.supported_over_dual());
        let forcing = InputRateForcing {
            cmt: 0,
            kind: InputRateKind::FirstOrder,
            arg_slots: vec![3],
            frac_slot: None,
        };
        let params = vec![1.0; crate::types::MAX_PK_PARAMS];
        assert!(forcing.prepare_dual::<f64>(&params).is_some());
    }

    /// The analytic `Dual2` gradient of the **delivered** first-order forcing
    /// (`frac · R_in`) matches central finite differences for both `ka` and the
    /// pathway fraction `frac` — the #505 / issue requirement "analytic ≡ FD incl.
    /// ∂/∂frac". The fraction enters linearly (`acc · frac` in
    /// `add_prepared_input_rate_forcing`), so `∂/∂frac = R_in`; `ka` flows through
    /// the `exp` density. Exercised through the same `frac()` reader the ODE
    /// provider uses, so a wrong dual rule (a missing `exp` jet or a dropped
    /// fraction channel) would surface here in the default `--features ci` build.
    #[test]
    fn first_order_dual_gradient_matches_central_fd() {
        use crate::sens::dual_mixed::DualMixed;
        type D = DualMixed<2, 2>;
        let forcing = InputRateForcing {
            cmt: 0,
            kind: InputRateKind::FirstOrder,
            arg_slots: vec![4], // ka @ 4 (dim 0)
            frac_slot: Some(2), // frac @ 2 (dim 1)
        };
        let dose = 100.0;
        let h = 1e-6;
        for &(ka, frac) in &[(0.5, 0.3), (1.2, 0.6), (2.5, 0.45)] {
            let mut params = vec![D::constant(0.0); crate::types::MAX_PK_PARAMS];
            params[4] = D::var(ka, 0);
            params[2] = D::var(frac, 1);
            let prep = forcing.prepare_dual::<D>(&params).unwrap();
            for &tad in &[0.3, 1.0, 4.0] {
                // Delivered rate = R_in(tad) · frac (the per-dose contribution).
                let delivered =
                    prep.rate(D::constant(tad), D::constant(dose)) * forcing.frac(&params);
                // ∂/∂ka = frac · ∂R_in/∂ka; ∂/∂frac = R_in (linear multiplier).
                let d_ka = frac
                    * (first_order_input_rate(tad, ka + h, dose)
                        - first_order_input_rate(tad, ka - h, dose))
                    / (2.0 * h);
                let d_frac = first_order_input_rate(tad, ka, dose);
                assert_relative_eq!(delivered.grad[0], d_ka, max_relative = 1e-5, epsilon = 1e-7);
                assert_relative_eq!(
                    delivered.grad[1],
                    d_frac,
                    max_relative = 1e-5,
                    epsilon = 1e-7
                );
            }
        }
    }
}
