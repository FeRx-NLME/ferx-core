// Analytic hazard functions for Phase 1 parametric TTE families.
//
// Parameter layout by family (all in natural scale, not log):
//   Exponential: [lambda, loghr_term]
//                lambda     = constant hazard rate (h = lambda * exp(loghr))
//                loghr_term = Σ(β·covariate) added on the log-hazard scale; 0.0 when none
//   Weibull:     [scale, shape, loghr_term]   — scale parameterization: H=(t/scale)^shape
//                scale      = scale parameter (a time; larger = slower events)
//                shape      = shape parameter (> 0; 1 = Exponential, > 1 = increasing hazard)
//                loghr_term = same as above; multiplies entire hazard (PH form)
//   Gompertz:    [alpha, gamma, loghr_term]
//                alpha      = baseline hazard at t=0
//                gamma      = hazard growth rate (> 0)
//                loghr_term = same as above
//
// The loghr_term is always at the last index for each family and defaults to 0.0
// when not provided (i.e. params.get(index) returns None).
//
// All functions guard against t ≤ 0 to avoid log/pow domain errors.

use crate::types::HazardFamily;

/// Returns (h(t), H(t)) for the given family and parameter vector.
///
/// Returns (0.0, 0.0) for t ≤ 0.
pub fn hazard_and_cum_hazard(family: HazardFamily, t: f64, params: &[f64]) -> (f64, f64) {
    if t <= 0.0 {
        return (0.0, 0.0);
    }
    match family {
        HazardFamily::Exponential => {
            let lambda = params[0];
            let loghr = params.get(1).copied().unwrap_or(0.0);
            // h(t) = lambda * exp(loghr);  H(t) = lambda * exp(loghr) * t
            let eff_lambda = lambda * loghr.exp();
            (eff_lambda, eff_lambda * t)
        }
        HazardFamily::Weibull => {
            let scale = params[0];
            let shape = params[1];
            let loghr = params.get(2).copied().unwrap_or(0.0);
            let exp_lhr = loghr.exp();
            // PH form: h(t) = (shape/scale)*(t/scale)^(shape-1) * exp(loghr)
            //          H(t) = (t/scale)^shape * exp(loghr)
            let t_scaled = t / scale;
            let h_val = (shape / scale) * t_scaled.powf(shape - 1.0) * exp_lhr;
            let cum_h = t_scaled.powf(shape) * exp_lhr;
            (h_val, cum_h)
        }
        HazardFamily::Gompertz => {
            let alpha = params[0];
            let gamma = params[1];
            let loghr = params.get(2).copied().unwrap_or(0.0);
            let exp_loghr = loghr.exp();
            if gamma == 0.0 {
                // γ=0: limit of (alpha/gamma)*(exp(gamma*t)-1) as γ→0 is alpha*t
                let eff = alpha * exp_loghr;
                return (eff, eff * t);
            }
            // h(t) = alpha * exp(gamma*t) * exp(loghr)
            // H(t) = (alpha/gamma) * (exp(gamma*t) - 1) * exp(loghr)
            let exp_gt = (gamma * t).exp();
            let h_val = alpha * exp_gt * exp_loghr;
            let cum_h = (alpha / gamma) * (exp_gt - 1.0) * exp_loghr;
            (h_val, cum_h)
        }
    }
}

/// Cumulative hazard H(t) only (cheaper when h is not needed).
pub fn cum_hazard(family: HazardFamily, t: f64, params: &[f64]) -> f64 {
    hazard_and_cum_hazard(family, t, params).1
}

/// Median survival time: the unique T satisfying H(T) = ln 2.
///
/// Computed analytically for all three families.  Returns `f64::NAN` if the
/// parameters are degenerate (e.g. `alpha ≤ 0` for Gompertz) or the median is
/// undefined — a declining-hazard (`γ < 0`) Gompertz has no median when its cure
/// fraction `S(∞) = exp(−α·e^{loghr}/|γ|) ≥ 0.5` (more than half never event); when
/// `S(∞) < 0.5` the median is finite and returned (#805).
pub fn median_survival(family: HazardFamily, params: &[f64]) -> f64 {
    use std::f64::consts::LN_2;
    match family {
        HazardFamily::Exponential => {
            let lambda = params[0];
            let loghr = params.get(1).copied().unwrap_or(0.0);
            let eff = lambda * loghr.exp();
            if eff > 0.0 {
                LN_2 / eff
            } else {
                f64::NAN
            }
        }
        HazardFamily::Weibull => {
            let scale = params[0];
            let shape = params[1];
            let loghr = params.get(2).copied().unwrap_or(0.0);
            let exp_lhr = loghr.exp();
            // H(T) = (T/scale)^shape * exp(loghr) = ln2
            // T = scale * (ln2 / exp(loghr))^(1/shape)
            if scale > 0.0 && shape > 0.0 && exp_lhr > 0.0 {
                scale * (LN_2 / exp_lhr).powf(1.0 / shape)
            } else {
                f64::NAN
            }
        }
        HazardFamily::Gompertz => {
            let alpha = params[0];
            let gamma = params[1];
            let loghr = params.get(2).copied().unwrap_or(0.0);
            let exp_lhr = loghr.exp();
            if alpha <= 0.0 || !(exp_lhr > 0.0) {
                return f64::NAN;
            }
            // H(T) = (alpha/gamma)*(exp(gamma*T)-1)*exp(loghr) = ln2
            // exp(gamma*T) = 1 + x  where  x = ln2·γ / (α·exp(loghr))
            if gamma == 0.0 {
                // γ=0: Gompertz degenerates to Exponential with rate α·exp(loghr)
                LN_2 / (alpha * exp_lhr)
            } else {
                // γ ≠ 0: exp(γT) = 1 + x with x = ln2·γ/(α·e^{loghr}). ln_1p is numerically
                // stable when |γ| is small (avoids cancellation in (1+x).ln()).
                //   γ > 0 ⇒ x > 0, so x > −1 always: finite median (behavior unchanged).
                //   γ < 0 ⇒ x < 0, with a finite limiting cumulative hazard H(∞)=α·e^{loghr}/|γ|.
                //     The median exists iff ln2 < H(∞) ⟺ x > −1 ⟺ cure fraction S(∞) < 0.5
                //     (then ln_1p(x) < 0 and γ < 0 give T > 0); otherwise it is +∞/undefined
                //     → NaN. Same γ<0 boundary as the samplers (#803/#804); here in the
                //     diagnostic path (#805).
                let x = LN_2 * gamma / (alpha * exp_lhr);
                if x > -1.0 {
                    x.ln_1p() / gamma
                } else {
                    f64::NAN
                }
            }
        }
    }
}

/// Mean survival time E[T] = ∫₀^∞ S(t) dt.
///
/// Uses the analytic form `1 / (λ · exp(loghr))` for the Exponential family and
/// the midpoint rule (2 000 steps to 40 × median) for Weibull and Gompertz.
/// Returns `f64::NAN` for degenerate parameters, and for a declining-hazard
/// (`γ < 0`) Gompertz — whose positive cure fraction makes the mean genuinely
/// infinite (#805).
pub fn mean_survival(family: HazardFamily, params: &[f64]) -> f64 {
    match family {
        HazardFamily::Exponential => {
            let lambda = params[0];
            let loghr = params.get(1).copied().unwrap_or(0.0);
            let eff = lambda * loghr.exp();
            if eff > 0.0 {
                1.0 / eff
            } else {
                f64::NAN
            }
        }
        _ => {
            // Gompertz γ=0 degenerates to Exponential with rate α·exp(loghr).
            // hazard_and_cum_hazard has its own γ=0 guard, but the integration
            // upper bound would be 40×median which overshoots the analytic limit —
            // so we return the exact value directly.
            if matches!(family, HazardFamily::Gompertz) && params[1] == 0.0 {
                let alpha = params[0];
                let loghr = params.get(2).copied().unwrap_or(0.0);
                let exp_lhr = loghr.exp();
                return if alpha > 0.0 && exp_lhr > 0.0 {
                    1.0 / (alpha * exp_lhr)
                } else {
                    f64::NAN
                };
            }
            // Gompertz γ<0 is a declining hazard with a positive cure fraction
            // S(∞)=exp(−α·e^{loghr}/|γ|) > 0, so S(t) never decays to 0 and the mean
            // E[T]=∫₀^∞ S diverges. Return NaN explicitly: since #805 median_survival now
            // yields a *finite* γ<0 median (when S(∞)<0.5), the median-based integration
            // below would otherwise run over [0, 40·median], miss the non-decaying tail,
            // and report a bogus finite mean.
            if matches!(family, HazardFamily::Gompertz) && params[1] < 0.0 {
                return f64::NAN;
            }
            let t_med = median_survival(family, params);
            if !t_med.is_finite() || t_med <= 0.0 {
                return f64::NAN;
            }
            let t_max = 40.0 * t_med;
            let n = 2000usize;
            let dt = t_max / n as f64;
            let mut sum = 0.0;
            for i in 0..n {
                let t = (i as f64 + 0.5) * dt;
                let (_, cum_h) = hazard_and_cum_hazard(family, t, params);
                sum += (-cum_h).exp();
            }
            sum * dt
        }
    }
}

/// Sample an unconditional event time via analytic inverse-CDF.
///
/// u must be in (0, 1).  Returns a finite positive value; for extremely
/// small u (very early events) or extreme parameters the result is clamped
/// at f64::MAX to avoid infinity.
pub fn sample_event_time(family: HazardFamily, params: &[f64], u: f64) -> f64 {
    // Clamp away from 0 and 1: Standard distribution can yield exactly 0, making
    // -ln(0) = +∞.  Callers should prefer Open01; this is a defence-in-depth guard.
    let u = u.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    let neg_log_u = -u.ln(); // -log U, always positive for u ∈ (0,1)
    let t = match family {
        HazardFamily::Exponential => {
            let lambda = params[0];
            let loghr = params.get(1).copied().unwrap_or(0.0);
            // H(T) = lambda*exp(loghr)*T = -log U  →  T = -log(U) / (lambda*exp(loghr))
            neg_log_u / (lambda * loghr.exp())
        }
        HazardFamily::Weibull => {
            let scale = params[0];
            let shape = params[1];
            let loghr = params.get(2).copied().unwrap_or(0.0);
            // H(T) = (T/scale)^shape * exp(loghr) = -log U
            // T = scale * (-log U / exp(loghr))^(1/shape)
            scale * (neg_log_u / loghr.exp()).powf(1.0 / shape)
        }
        HazardFamily::Gompertz => {
            let alpha = params[0];
            let gamma = params[1];
            let loghr = params.get(2).copied().unwrap_or(0.0);
            let exp_loghr = loghr.exp();
            if gamma == 0.0 {
                // γ=0: degenerate to Exponential; H(T) = alpha*exp(loghr)*T = -log U
                return neg_log_u / (alpha * exp_loghr);
            }
            // H(T) = -log U  =>  (alpha/gamma)*(exp(gamma*T)-1)*exp(loghr) = neg_log_u
            // exp(gamma*T) = 1 + neg_log_u * gamma / (alpha * exp(loghr))
            let inner = 1.0 + neg_log_u * gamma / (alpha * exp_loghr);
            // γ < 0 is a declining hazard with a finite limiting cumulative hazard
            // H(∞) = −α·exp(loghr)/γ, i.e. a genuine cure fraction S(∞) = exp(−H(∞)) > 0.
            // A draw whose −log U exceeds H(∞) (u below the cure fraction) yields inner ≤ 0 and
            // has no event; inner ∈ (0, 1) is a valid finite event (T = ln(inner)/γ > 0). For
            // γ > 0 the hazard grows unboundedly and inner > 1 always, so this guard then only
            // catches numerically degenerate draws. Rejecting inner ≤ 1 would wrongly censor
            // *every* γ < 0 event (#803).
            if inner <= 0.0 {
                return f64::MAX;
            }
            inner.ln() / gamma
        }
    };
    if t.is_finite() && t > 0.0 {
        t
    } else {
        f64::MAX
    }
}

/// Sample a conditional event time given survival past `entry_time`.
///
/// Solves H(T) - H(entry_time) = -log U for T using the exact form for each family.
/// Falls back to `entry_time + sample_event_time(...)` for Exponential (memoryless property).
pub fn sample_conditional_event_time(
    family: HazardFamily,
    params: &[f64],
    entry_time: f64,
    u: f64,
) -> f64 {
    let u = u.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    let neg_log_u = -u.ln();
    let t = match family {
        HazardFamily::Exponential => {
            let loghr = params.get(1).copied().unwrap_or(0.0);
            let eff_lambda = params[0] * loghr.exp();
            // Memoryless with PH: effective rate is lambda*exp(loghr); shift by entry_time.
            entry_time + neg_log_u / eff_lambda
        }
        HazardFamily::Weibull => {
            let scale = params[0];
            let shape = params[1];
            let loghr = params.get(2).copied().unwrap_or(0.0);
            let exp_lhr = loghr.exp();
            // H(T) - H(entry) = -log U
            // (T/scale)^shape * exp_lhr - (entry/scale)^shape * exp_lhr = neg_log_u
            // (T/scale)^shape = (entry/scale)^shape + neg_log_u / exp_lhr
            let h_entry = (entry_time / scale).powf(shape);
            let h_target = h_entry + neg_log_u / exp_lhr;
            scale * h_target.powf(1.0 / shape)
        }
        HazardFamily::Gompertz => {
            let alpha = params[0];
            let gamma = params[1];
            let loghr = params.get(2).copied().unwrap_or(0.0);
            let exp_loghr = loghr.exp();
            if gamma == 0.0 {
                // γ=0: memoryless Exponential limit; shift by entry_time
                return entry_time + neg_log_u / (alpha * exp_loghr);
            }
            // exp(gamma*T) = exp(gamma*entry) + neg_log_u * gamma / (alpha * exp_loghr)
            let exp_entry = (gamma * entry_time).exp();
            let inner = exp_entry + neg_log_u * gamma / (alpha * exp_loghr);
            // As in `sample_event_time`, γ < 0 has a cure fraction beyond `entry`: inner ≤ 0 means
            // −log U exceeds the remaining cumulative hazard H(∞) − H(entry), so there is no event.
            // inner ∈ (0, exp_entry) is a valid event with T > entry; γ > 0 keeps inner > 1. The
            // old inner ≤ 1 guard censored every γ < 0 event (#803).
            if inner <= 0.0 {
                return f64::MAX;
            }
            inner.ln() / gamma
        }
    };
    if t.is_finite() && t > entry_time {
        t
    } else {
        f64::MAX
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_hazard_values() {
        let (h, cum) = hazard_and_cum_hazard(HazardFamily::Exponential, 2.0, &[0.1]);
        assert!((h - 0.1).abs() < 1e-12, "h = {h}");
        assert!((cum - 0.2).abs() < 1e-12, "H = {cum}");
    }

    #[test]
    fn weibull_hazard_values() {
        // scale=1, shape=2: H(1) = 1.0, h(1) = 2.0
        let (h, cum) = hazard_and_cum_hazard(HazardFamily::Weibull, 1.0, &[1.0, 2.0]);
        assert!((h - 2.0).abs() < 1e-12, "h = {h}");
        assert!((cum - 1.0).abs() < 1e-12, "H = {cum}");
    }

    #[test]
    fn gompertz_hazard_at_zero_is_alpha() {
        // At t→0+: h ≈ alpha, H ≈ 0
        let (h, cum) = hazard_and_cum_hazard(HazardFamily::Gompertz, 1e-8, &[0.002, 0.005, 0.0]);
        assert!((h - 0.002).abs() < 1e-6, "h = {h}");
        assert!(cum < 1e-7, "H = {cum}");
    }

    #[test]
    fn exponential_inverse_cdf_roundtrip() {
        // Sample T, then check H(T) = -log U
        let u = 0.3_f64;
        let params = &[0.1];
        let t = sample_event_time(HazardFamily::Exponential, params, u);
        let (_, cum) = hazard_and_cum_hazard(HazardFamily::Exponential, t, params);
        let expected = -u.ln();
        assert!(
            (cum - expected).abs() < 1e-10,
            "H = {cum}, expected {expected}"
        );
    }

    #[test]
    fn weibull_inverse_cdf_roundtrip() {
        let u = 0.5_f64;
        let params = &[20.0, 2.0];
        let t = sample_event_time(HazardFamily::Weibull, params, u);
        let (_, cum) = hazard_and_cum_hazard(HazardFamily::Weibull, t, params);
        let expected = -u.ln();
        assert!(
            (cum - expected).abs() < 1e-9,
            "H = {cum}, expected {expected}"
        );
    }

    #[test]
    fn gompertz_inverse_cdf_roundtrip() {
        let u = 0.4_f64;
        let params = &[0.002, 0.005, 0.0];
        let t = sample_event_time(HazardFamily::Gompertz, params, u);
        let (_, cum) = hazard_and_cum_hazard(HazardFamily::Gompertz, t, params);
        let expected = -u.ln();
        assert!(
            (cum - expected).abs() < 1e-9,
            "H = {cum}, expected {expected}"
        );
    }

    #[test]
    fn gompertz_gamma_negative_samples_events_and_cure_fraction() {
        // Declining-hazard Gompertz (#803): γ < 0 has a finite limiting cumulative hazard
        // H(∞) = −α·exp(loghr)/γ and a genuine cure fraction S(∞) = exp(−H(∞)).
        //   α = 0.002, γ = −0.005, loghr = 0  ⇒  H(∞) = 0.002/0.005 = 0.4, S(∞) = e^{-0.4} ≈ 0.6703.
        let params = &[0.002, -0.005, 0.0];
        let cure = (-0.4_f64).exp();

        // u above the cure fraction ⇒ a valid finite event that round-trips H(T) = −ln u.
        // Under the old `inner <= 1.0` guard this returned the `f64::MAX` no-event sentinel
        // for every draw. The `t < f64::MAX` bound is the direct regression signal (an actual
        // event was produced, not the sentinel — note `f64::MAX` is itself finite and > 0, so
        // the earlier checks alone would not catch it); the round-trip below then confirms the
        // event time is numerically correct.
        let u = 0.8_f64;
        assert!(u > cure, "test setup: u must exceed the cure fraction");
        let t = sample_event_time(HazardFamily::Gompertz, params, u);
        assert!(
            t.is_finite() && t > 0.0 && t < f64::MAX,
            "γ<0 above-cure draw must yield a real event, not the no-event sentinel; got {t}"
        );
        let (_, cum) = hazard_and_cum_hazard(HazardFamily::Gompertz, t, params);
        assert!(
            (cum - (-u.ln())).abs() < 1e-9,
            "H(T) = {cum}, expected {}",
            -u.ln()
        );

        // u below the cure fraction ⇒ no event (the true cure fraction), sentinel returned.
        let u_cured = 0.5_f64;
        assert!(
            u_cured < cure,
            "test setup: u must be below the cure fraction"
        );
        let t_cured = sample_event_time(HazardFamily::Gompertz, params, u_cured);
        assert_eq!(t_cured, f64::MAX, "γ<0 below-cure draw must be censored");
    }

    #[test]
    fn gompertz_gamma_negative_conditional_events_and_cure_fraction() {
        // Conditional (left-truncated) declining-hazard Gompertz (#803). The remaining cumulative
        // hazard beyond `entry` is H(∞) − H(entry); a draw with −ln u above it has no event.
        let params = &[0.002, -0.005, 0.0];
        let entry = 20.0_f64;
        let h_entry = cum_hazard(HazardFamily::Gompertz, entry, params);
        let remaining = 0.4 - h_entry; // H(∞) = 0.4 for these params
        let cond_cure = (-remaining).exp();

        // u above the conditional cure fraction ⇒ finite event later than entry, round-tripping
        // H(T) − H(entry) = −ln u. The old guard returned the `f64::MAX` sentinel for all of
        // these; `t < f64::MAX` is the direct regression signal (the sentinel is finite and
        // > entry, so it would otherwise slip past), the round-trip confirms correctness.
        let u = 0.9_f64;
        assert!(
            u > cond_cure,
            "test setup: u must exceed the conditional cure fraction"
        );
        let t = sample_conditional_event_time(HazardFamily::Gompertz, params, entry, u);
        assert!(
            t.is_finite() && t > entry && t < f64::MAX,
            "conditional γ<0 above-cure draw must yield a real event > entry, not the sentinel; got {t}"
        );
        let h_t = cum_hazard(HazardFamily::Gompertz, t, params);
        assert!(
            (h_t - h_entry - (-u.ln())).abs() < 1e-8,
            "H(T)-H(entry) = {}, expected {}",
            h_t - h_entry,
            -u.ln()
        );

        // u below the conditional cure fraction ⇒ censored at the sentinel.
        let u_cured = 0.5_f64;
        assert!(
            u_cured < cond_cure,
            "test setup: u must be below the conditional cure fraction"
        );
        let t_cured = sample_conditional_event_time(HazardFamily::Gompertz, params, entry, u_cured);
        assert_eq!(
            t_cured,
            f64::MAX,
            "conditional γ<0 below-cure draw must be censored"
        );
    }

    #[test]
    fn conditional_exponential_shifts_by_entry() {
        // Memoryless: T|T>entry ~ entry + Exp(lambda)
        let params = &[0.1];
        let t = sample_conditional_event_time(HazardFamily::Exponential, params, 5.0, 0.5);
        // Unconditional from entry 0: T = -log(0.5)/0.1 ≈ 6.93; conditional adds 5.0 → ≈ 11.93
        let unconditional = sample_event_time(HazardFamily::Exponential, params, 0.5);
        assert!((t - (entry_shift(unconditional, 5.0, params))).abs() < 1e-9);
        fn entry_shift(uncond: f64, entry: f64, _params: &[f64]) -> f64 {
            entry + uncond // memoryless; exponential is shift-invariant
        }
    }

    #[test]
    fn left_truncation_regression() {
        // H(T) - H(entry_time) should equal -log U by construction.
        let u = 0.3_f64;
        let entry = 5.0;
        for &family in &[
            HazardFamily::Exponential,
            HazardFamily::Weibull,
            HazardFamily::Gompertz,
        ] {
            let params: Vec<f64> = match family {
                HazardFamily::Exponential => vec![0.1],
                HazardFamily::Weibull => vec![20.0, 2.0],
                HazardFamily::Gompertz => vec![0.002, 0.005, 0.0],
            };
            let t = sample_conditional_event_time(family, &params, entry, u);
            let h_t = cum_hazard(family, t, &params);
            let h_entry = cum_hazard(family, entry, &params);
            let expected = -u.ln();
            assert!(
                (h_t - h_entry - expected).abs() < 1e-8,
                "{family:?}: H(T)-H(entry)={}, expected {}",
                h_t - h_entry,
                expected
            );
        }
    }

    #[test]
    fn exponential_loghr_doubles_hazard() {
        // loghr = ln(2) should double both h and H compared to no loghr.
        let loghr = 2.0_f64.ln();
        let params_base = [0.1_f64];
        let params_lhr = [0.1_f64, loghr];
        let (h0, cum0) = hazard_and_cum_hazard(HazardFamily::Exponential, 3.0, &params_base);
        let (h1, cum1) = hazard_and_cum_hazard(HazardFamily::Exponential, 3.0, &params_lhr);
        assert!((h1 / h0 - 2.0).abs() < 1e-12, "h ratio = {}", h1 / h0);
        assert!(
            (cum1 / cum0 - 2.0).abs() < 1e-12,
            "H ratio = {}",
            cum1 / cum0
        );
    }

    #[test]
    fn exponential_loghr_inverse_cdf_roundtrip() {
        // With loghr, H(T) = lambda*exp(loghr)*T; sample must satisfy H(T) = -log U.
        let u = 0.4_f64;
        let params = [0.1_f64, 0.5_f64]; // loghr = 0.5
        let t = sample_event_time(HazardFamily::Exponential, &params, u);
        let (_, cum) = hazard_and_cum_hazard(HazardFamily::Exponential, t, &params);
        let expected = -u.ln();
        assert!(
            (cum - expected).abs() < 1e-10,
            "H = {cum}, expected {expected}"
        );
    }

    #[test]
    fn weibull_loghr_doubles_hazard() {
        // loghr = ln(2) doubles h and H for Weibull (PH form).
        let loghr = 2.0_f64.ln();
        let params_base = [20.0_f64, 2.0_f64];
        let params_lhr = [20.0_f64, 2.0_f64, loghr];
        let (h0, cum0) = hazard_and_cum_hazard(HazardFamily::Weibull, 5.0, &params_base);
        let (h1, cum1) = hazard_and_cum_hazard(HazardFamily::Weibull, 5.0, &params_lhr);
        assert!((h1 / h0 - 2.0).abs() < 1e-12, "h ratio = {}", h1 / h0);
        assert!(
            (cum1 / cum0 - 2.0).abs() < 1e-12,
            "H ratio = {}",
            cum1 / cum0
        );
    }

    #[test]
    fn weibull_loghr_inverse_cdf_roundtrip() {
        // Sample T with loghr, confirm H(T) = -log U.
        let u = 0.6_f64;
        let params = [20.0_f64, 2.0_f64, 0.3_f64]; // loghr = 0.3
        let t = sample_event_time(HazardFamily::Weibull, &params, u);
        let (_, cum) = hazard_and_cum_hazard(HazardFamily::Weibull, t, &params);
        let expected = -u.ln();
        assert!(
            (cum - expected).abs() < 1e-9,
            "H = {cum}, expected {expected}"
        );
    }

    #[test]
    fn left_truncation_regression_with_loghr() {
        // H(T) - H(entry) = -log U must hold even with nonzero loghr.
        let u = 0.3_f64;
        let entry = 5.0;
        let cases: &[(&str, HazardFamily, Vec<f64>)] = &[
            (
                "Exponential+loghr",
                HazardFamily::Exponential,
                vec![0.1, 0.5],
            ),
            ("Weibull+loghr", HazardFamily::Weibull, vec![20.0, 2.0, 0.3]),
            (
                "Gompertz+loghr",
                HazardFamily::Gompertz,
                vec![0.002, 0.005, 0.4],
            ),
        ];
        for (label, family, params) in cases {
            let t = sample_conditional_event_time(*family, params, entry, u);
            let h_t = cum_hazard(*family, t, params);
            let h_entry = cum_hazard(*family, entry, params);
            let expected = -u.ln();
            assert!(
                (h_t - h_entry - expected).abs() < 1e-8,
                "{label}: H(T)-H(entry)={}, expected {}",
                h_t - h_entry,
                expected
            );
        }
    }

    #[test]
    fn median_exponential_is_ln2_over_lambda() {
        // Exp(0.1): median = ln(2)/0.1 ≈ 6.931
        let t_50 = median_survival(HazardFamily::Exponential, &[0.1]);
        let expected = std::f64::consts::LN_2 / 0.1;
        assert!((t_50 - expected).abs() < 1e-12, "median = {t_50}");
        // Verify S(t_50) ≈ 0.5
        let (_, cum) = hazard_and_cum_hazard(HazardFamily::Exponential, t_50, &[0.1]);
        assert!(
            ((-cum).exp() - 0.5).abs() < 1e-12,
            "S(median) = {}",
            (-cum).exp()
        );
    }

    #[test]
    fn median_weibull_consistency() {
        let params = [20.0_f64, 2.0_f64];
        let t_50 = median_survival(HazardFamily::Weibull, &params);
        // S(t_50) must equal 0.5 by construction
        let (_, cum) = hazard_and_cum_hazard(HazardFamily::Weibull, t_50, &params);
        assert!(
            ((-cum).exp() - 0.5).abs() < 1e-12,
            "S(median)={} for Weibull",
            (-cum).exp()
        );
    }

    #[test]
    fn median_gompertz_consistency() {
        let params = [0.002_f64, 0.005_f64, 0.0_f64];
        let t_50 = median_survival(HazardFamily::Gompertz, &params);
        let (_, cum) = hazard_and_cum_hazard(HazardFamily::Gompertz, t_50, &params);
        assert!(
            ((-cum).exp() - 0.5).abs() < 1e-10,
            "S(median)={} for Gompertz",
            (-cum).exp()
        );
    }

    #[test]
    fn median_gompertz_gamma_zero_matches_exponential() {
        // γ=0: Gompertz degenerates to Exponential; median must equal ln2/α.
        // (hazard_and_cum_hazard has a 0/0 form at γ=0, so verify analytically.)
        let alpha = 0.05_f64;
        let t_gompertz = median_survival(HazardFamily::Gompertz, &[alpha, 0.0, 0.0]);
        let t_exp = median_survival(HazardFamily::Exponential, &[alpha]);
        assert!(
            t_gompertz.is_finite(),
            "Gompertz γ=0 median must be finite, got {t_gompertz}"
        );
        assert!(
            (t_gompertz - t_exp).abs() < 1e-10,
            "Gompertz(γ=0) median {t_gompertz} should equal Exponential median {t_exp}"
        );
    }

    #[test]
    fn mean_gompertz_gamma_zero_matches_exponential() {
        // γ=0: mean must equal 1/α (analytic Exponential limit), not NaN.
        let alpha = 0.05_f64;
        let m_gompertz = mean_survival(HazardFamily::Gompertz, &[alpha, 0.0, 0.0]);
        let m_exp = mean_survival(HazardFamily::Exponential, &[alpha]);
        assert!(
            m_gompertz.is_finite(),
            "Gompertz γ=0 mean must be finite, got {m_gompertz}"
        );
        assert!(
            (m_gompertz - m_exp).abs() < 1e-10,
            "Gompertz(γ=0) mean {m_gompertz} should equal Exponential mean {m_exp}"
        );
    }

    #[test]
    fn median_gompertz_small_gamma_stable() {
        // Very small γ — ln_1p path must not return NaN
        let params = [0.002_f64, 1e-10_f64, 0.0_f64];
        let t_50 = median_survival(HazardFamily::Gompertz, &params);
        assert!(
            t_50.is_finite(),
            "Gompertz small-γ median must be finite, got {t_50}"
        );
        let (_, cum) = hazard_and_cum_hazard(HazardFamily::Gompertz, t_50, &params);
        assert!(
            ((-cum).exp() - 0.5).abs() < 1e-8,
            "S(median)={} for small-γ Gompertz",
            (-cum).exp()
        );
    }

    #[test]
    fn median_gompertz_gamma_negative_finite_when_cure_below_half() {
        // Declining-hazard Gompertz (#805): γ<0 has a finite limiting cumulative hazard
        // H(∞)=α·e^{loghr}/|γ| and cure fraction S(∞)=exp(−H(∞)). When S(∞)<0.5 (fewer than
        // half are "cured") a finite median genuinely exists and must be returned — the old
        // code blanket-returned NaN for every γ<0.
        //   α=0.002, γ=−0.001, loghr=0 ⇒ H(∞)=2.0, S(∞)=e^{−2}≈0.1353 < 0.5.
        let params = [0.002_f64, -0.001_f64, 0.0_f64];
        let s_inf = (-(0.002_f64 / 0.001)).exp(); // exp(−H(∞))
        assert!(s_inf < 0.5, "test setup: cure fraction must be below 0.5");
        let t_50 = median_survival(HazardFamily::Gompertz, &params);
        assert!(
            t_50.is_finite() && t_50 > 0.0,
            "γ<0 median must be finite and positive when S(∞)<0.5, got {t_50}"
        );
        // Round-trips the defining equation H(median)=ln2 ⟺ S(median)=0.5.
        let (_, cum) = hazard_and_cum_hazard(HazardFamily::Gompertz, t_50, &params);
        assert!(
            (cum - std::f64::consts::LN_2).abs() < 1e-10,
            "H(median)={cum}, expected ln2"
        );
        assert!(
            ((-cum).exp() - 0.5).abs() < 1e-10,
            "S(median)={} for γ<0 Gompertz",
            (-cum).exp()
        );
    }

    #[test]
    fn median_gompertz_gamma_negative_nan_when_cure_at_least_half() {
        // γ<0 with S(∞)≥0.5: more than half never event, so the median is genuinely
        // +∞/undefined and NaN is correct.
        //   α=0.002, γ=−0.005, loghr=0 ⇒ H(∞)=0.4, S(∞)=e^{−0.4}≈0.6703 ≥ 0.5.
        let params = [0.002_f64, -0.005_f64, 0.0_f64];
        let s_inf = (-(0.002_f64 / 0.005)).exp();
        assert!(s_inf >= 0.5, "test setup: cure fraction must be ≥ 0.5");
        let t_50 = median_survival(HazardFamily::Gompertz, &params);
        assert!(
            t_50.is_nan(),
            "γ<0 median must be NaN when S(∞)≥0.5, got {t_50}"
        );
    }

    #[test]
    fn median_gompertz_gamma_negative_boundary_s_inf_half() {
        // Boundary S(∞)=0.5 (⟺ x=−1): the median is undefined (+∞). Bracket it — a cure
        // fraction a hair below 0.5 (H(∞) just above ln2) yields a large finite median; a hair
        // above 0.5 (H(∞) just below ln2) yields NaN. Construct α from a target H(∞) via
        // α = H(∞)·|γ| (loghr=0).
        let gamma = -0.001_f64;
        let just_below_half = {
            let h_inf = std::f64::consts::LN_2 * (1.0 + 1e-6); // S(∞) just below 0.5
            [h_inf * 0.001, gamma, 0.0_f64]
        };
        let just_above_half = {
            let h_inf = std::f64::consts::LN_2 * (1.0 - 1e-6); // S(∞) just above 0.5
            [h_inf * 0.001, gamma, 0.0_f64]
        };
        let m_below = median_survival(HazardFamily::Gompertz, &just_below_half);
        assert!(
            m_below.is_finite() && m_below > 0.0,
            "S(∞) just below 0.5 must give a finite median, got {m_below}"
        );
        let m_above = median_survival(HazardFamily::Gompertz, &just_above_half);
        assert!(
            m_above.is_nan(),
            "S(∞) just above 0.5 must give NaN median, got {m_above}"
        );
    }

    #[test]
    fn mean_gompertz_gamma_negative_is_nan_even_with_finite_median() {
        // The #805 trap: a γ<0 Gompertz with S(∞)<0.5 has a *finite median* (so the
        // median-based integration bound would fire) but an *infinite mean* — S(t) floors at
        // S(∞)>0 and never decays, so ∫₀^∞ S diverges. mean_survival must return NaN via its
        // explicit γ<0 guard, NOT a finite value from integrating [0, 40·median]. Uses the same
        // finite-median params as `median_gompertz_gamma_negative_finite_when_cure_below_half`,
        // so this test fails the moment that guard is removed.
        let params = [0.002_f64, -0.001_f64, 0.0_f64];
        // Precondition: the median really is finite here (else this wouldn't guard the trap).
        assert!(
            median_survival(HazardFamily::Gompertz, &params).is_finite(),
            "test precondition: median must be finite for the trap to bite"
        );
        let m = mean_survival(HazardFamily::Gompertz, &params);
        assert!(
            m.is_nan(),
            "γ<0 Gompertz mean is infinite (positive cure fraction) ⇒ must be NaN, got {m}"
        );
    }

    #[test]
    fn mean_exponential_analytic() {
        // Exp(λ): mean = 1/λ
        let m = mean_survival(HazardFamily::Exponential, &[0.1]);
        assert!((m - 10.0).abs() < 1e-12, "mean = {m}");
    }

    #[test]
    fn mean_weibull_numerical_plausible() {
        // shape=1 → Exponential with scale=20: mean should be ~20
        let m = mean_survival(HazardFamily::Weibull, &[20.0, 1.0]);
        assert!(
            (m - 20.0).abs() < 0.05,
            "Weibull(scale=20, shape=1) mean = {m}"
        );
    }

    #[test]
    fn mean_exceeds_median() {
        // For right-skewed distributions, mean > median
        for (fam, params) in [
            (HazardFamily::Exponential, vec![0.1_f64]),
            (HazardFamily::Weibull, vec![20.0_f64, 1.5_f64]),
            (HazardFamily::Gompertz, vec![0.002_f64, 0.005_f64, 0.0_f64]),
        ] {
            let med = median_survival(fam, &params);
            let mu = mean_survival(fam, &params);
            assert!(mu > med, "{fam:?}: mean {mu} should exceed median {med}");
        }
    }
}
