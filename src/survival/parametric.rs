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
            // H(T) = -log U  =>  (alpha/gamma)*(exp(gamma*T)-1)*exp(loghr) = neg_log_u
            // exp(gamma*T) = 1 + neg_log_u * gamma / (alpha * exp(loghr))
            let inner = 1.0 + neg_log_u * gamma / (alpha * exp_loghr);
            if inner <= 1.0 {
                // Numerically degenerate (should not happen for valid params)
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
            // exp(gamma*T) = exp(gamma*entry) + neg_log_u * gamma / (alpha * exp_loghr)
            let exp_entry = (gamma * entry_time).exp();
            let inner = exp_entry + neg_log_u * gamma / (alpha * exp_loghr);
            if inner <= 1.0 {
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
}
