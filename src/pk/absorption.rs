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

/// Which built-in absorption input-rate model a forcing term uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputRateKind {
    /// Savic transit-compartment chain — `transit(n, mtt)`.
    Transit,
    /// Freijer & Post inverse-Gaussian density — `igd(mat, cv2)`.
    InverseGaussian,
}

impl InputRateKind {
    /// Whether this kind's input-rate forcing has been lifted to `PkNum`/`Dual2`
    /// so the analytic ODE sensitivity provider (`sens/ode_provider.rs`) can
    /// evaluate it over dual numbers; otherwise a model using it stays on the FD
    /// fallback. Inverse-Gaussian (#430 slice 1) and transit (#430 slice 2, via
    /// the `ln_gamma` `Dual2` rule = digamma/trigamma) are both lifted; Weibull
    /// follows with Phase 2 (its `powf` `Dual2` rule).
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
/// [`InputRateKind::InverseGaussian`], `[mat, cv2]`.
#[derive(Debug, Clone)]
pub struct InputRateForcing {
    /// 0-based ODE compartment that receives `R_in`.
    pub cmt: usize,
    pub kind: InputRateKind,
    /// Indices into the flat individual-parameter vector for this model's args.
    pub arg_slots: Vec<usize>,
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
        }
    }

    /// Build the prepared input-rate constants over `T: PkNum` (e.g.
    /// `T = Dual2<N>` for the analytic ODE sensitivity provider, #430) from the
    /// individual-parameter vector `params` — laid out identically to the `f64`
    /// [`Self::prepare`] input, so `arg_slots` index the same way (and the
    /// per-kind argument defaults match `prepare`, so the lifted constants
    /// reproduce the scalar ones for `T = f64`). Both lifted kinds
    /// (inverse-Gaussian, transit) return `Some`; a future unlifted kind (Weibull,
    /// pre-Phase-2) would return `None`, keeping that model on the FD fallback.
    /// [`InputRateKind::supported_over_dual`] gates which kinds reach here and is
    /// pinned consistent with this `match` by
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
}

impl<T: PkNum> PreparedInputRate<T> {
    /// Domain floor for the strictly-positive input-rate parameters (transit
    /// `mtt`; inverse-Gaussian `mat`, `cv2`) when clamping a transient mid-fit
    /// excursion (see the `transit` / `inverse_gaussian` constructors). Far below
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

    /// `prepare_dual` lifts **both** forcings (IG and transit) to a `PkNum` type
    /// (here `T = f64`), reproducing the scalar `prepare` exactly — the `T = f64`
    /// byte-identity that lets the analytic ODE provider evaluate them over `Dual2`
    /// without drifting from the production predictor (#430; transit = slice 2).
    #[test]
    fn prepare_dual_lifts_ig_and_transit() {
        let mut params = vec![0.0; crate::types::MAX_PK_PARAMS];
        params[4] = 2.0; // mat
        params[5] = 0.3; // cv2
        params[6] = 3.0; // n
        params[7] = 1.0; // mtt

        let ig = InputRateForcing {
            cmt: 1,
            kind: InputRateKind::InverseGaussian,
            arg_slots: vec![4, 5], // mat @ 4, cv2 @ 5
        };
        let transit = InputRateForcing {
            cmt: 0,
            kind: InputRateKind::Transit,
            arg_slots: vec![6, 7], // n @ 6, mtt @ 7
        };
        for forcing in [&ig, &transit] {
            let lifted = forcing
                .prepare_dual::<f64>(&params)
                .expect("both kinds lift over PkNum in slice 2");
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
        const ALL_KINDS: &[InputRateKind] =
            &[InputRateKind::InverseGaussian, InputRateKind::Transit];
        let params = vec![1.0; crate::types::MAX_PK_PARAMS];
        for &kind in ALL_KINDS {
            let forcing = InputRateForcing {
                cmt: 1,
                kind,
                arg_slots: vec![4, 5],
            };
            assert_eq!(
                kind.supported_over_dual(),
                forcing.prepare_dual::<f64>(&params).is_some(),
                "supported_over_dual must match prepare_dual liftability for {kind:?}"
            );
        }
    }
}
