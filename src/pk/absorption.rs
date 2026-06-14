//! Built-in absorption **input-rate functions** — `R_in(tad)` per model (#322).
//!
//! Each returns the dose-driven appearance rate into the compartment it feeds,
//! normalised so `∫₀^∞ R_in dt = dose`, where the caller folds bioavailability
//! into `dose = F · amt`. `R_in = 0` for `tad ≤ 0` (the input starts after the
//! dose); per-dose contributions are superposed by the caller.
//!
//! These are the inherently-numerical absorption models that feed an explicit
//! ODE disposition (see `plans/absorption-models.md`). They are AD/Enzyme-safe
//! (only `+ − * /`, `.ln()`, `.exp()`; no `f64::max`/`min` intrinsics — see
//! CLAUDE.md). Written for `f64` for now; a shared numeric trait for the
//! `Dual`/Enzyme paths follows when these are wired into the autodiff ODE
//! gradient (the roadmap's escape hatch — duplicate-free generics later).

use crate::stats::special::ln_gamma;

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
pub fn transit_input_rate(tad: f64, n: f64, mtt: f64, dose: f64) -> f64 {
    if tad <= 0.0 || dose <= 0.0 {
        return 0.0;
    }
    let ktr = (n + 1.0) / mtt;
    let x = ktr * tad; // > 0 (tad > 0, ktr > 0 for valid params)

    // ln R_in = ln dose + ln KTR + n·ln(KTR·tad) − KTR·tad − ln Γ(n + 1).
    // For n = 0 the middle term is 0·ln x = 0, reducing to the first-order
    // (Bateman) input dose·KTR·exp(−KTR·tad).
    let ln_rin = dose.ln() + ktr.ln() + n * x.ln() - x - ln_gamma(n + 1.0);
    ln_rin.exp()
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

/// Which built-in absorption input-rate model a forcing term uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputRateKind {
    /// Savic transit-compartment chain — `transit(n, mtt)`.
    Transit,
}

/// A built-in absorption input-rate term attached to one ODE compartment.
///
/// Design A (see `plans/absorption-models.md`): the input-rate function is split
/// out of the `[odes]` RHS at parse time and evaluated here with dose context,
/// rather than threaded through the expression AST / bytecode VM / symbolic-AD
/// machinery. `arg_slots` index the flat individual-parameter vector for this
/// model's parameters — for [`InputRateKind::Transit`], `[n, mtt]`.
#[derive(Debug, Clone)]
pub struct InputRateForcing {
    /// 0-based ODE compartment that receives `R_in`.
    pub cmt: usize,
    pub kind: InputRateKind,
    /// Indices into the flat individual-parameter vector for this model's args.
    pub arg_slots: Vec<usize>,
}

impl InputRateForcing {
    /// Appearance rate `R_in(tad)` into [`Self::cmt`] for one dose, where
    /// `dose = F · amt` and `params` is the flat individual-parameter vector.
    /// Per-dose contributions are summed by the caller; `tad ≤ 0 ⇒ 0`.
    pub fn rate(&self, tad: f64, dose: f64, params: &[f64]) -> f64 {
        let arg = |i: usize, dflt: f64| {
            self.arg_slots
                .get(i)
                .and_then(|&s| params.get(s))
                .copied()
                .unwrap_or(dflt)
        };
        match self.kind {
            InputRateKind::Transit => transit_input_rate(tad, arg(0, 0.0), arg(1, 1.0), dose),
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
}
