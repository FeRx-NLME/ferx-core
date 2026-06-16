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
    /// Read this forcing's argument `i` from the flat individual-parameter
    /// vector `params`, falling back to `dflt` if the slot is absent.
    #[inline]
    fn arg(&self, params: &[f64], i: usize, dflt: f64) -> f64 {
        self.arg_slots
            .get(i)
            .and_then(|&s| params.get(s))
            .copied()
            .unwrap_or(dflt)
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
        }
    }
}

/// An input-rate forcing with its dose-invariant constants precomputed for the
/// ODE hot path. Built once per RHS evaluation by [`InputRateForcing::prepare`];
/// [`Self::rate`] then costs only the `tad`/`dose`-dependent arithmetic per dose.
#[derive(Debug, Clone, Copy)]
pub enum PreparedInputRate {
    /// Savic transit constants: `KTR`, `ln KTR`, `n`, and `ln Γ(n + 1)`.
    Transit {
        ktr: f64,
        ln_ktr: f64,
        n: f64,
        ln_gamma_np1: f64,
    },
}

impl PreparedInputRate {
    /// Domain floor for `mtt` when clamping a transient mid-fit excursion (see
    /// [`Self::transit`]). Far below any realistic mean transit time, so it never
    /// perturbs a converged fit — it only keeps a transient `mtt ≤ 0` from
    /// turning `ktr.ln()` into a `NaN`.
    const MIN_MTT: f64 = 1e-8;

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
    /// `NaN` inputs also fall to the floor (every `>`/`>=` is false for `NaN`).
    #[inline]
    fn transit(n: f64, mtt: f64) -> Self {
        let mtt = if mtt > Self::MIN_MTT {
            mtt
        } else {
            Self::MIN_MTT
        };
        let n = if n >= 0.0 { n } else { 0.0 };
        let ktr = (n + 1.0) / mtt;
        PreparedInputRate::Transit {
            ktr,
            ln_ktr: ktr.ln(),
            n,
            ln_gamma_np1: ln_gamma(n + 1.0),
        }
    }

    /// Appearance rate `R_in(tad)` for one dose (`dose = F · amt`). Per-dose
    /// contributions are summed by the caller; `tad ≤ 0` or `dose ≤ 0 ⇒ 0`.
    #[inline]
    pub fn rate(&self, tad: f64, dose: f64) -> f64 {
        if tad <= 0.0 || dose <= 0.0 {
            return 0.0;
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
}
