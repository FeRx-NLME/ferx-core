use crate::sens::two_cpt::{
    two_cpt_infusion_g, two_cpt_infusion_ss_g, two_cpt_iv_bolus_g, two_cpt_iv_bolus_ss_g,
    two_cpt_oral_g, two_cpt_oral_ss_g,
};
use crate::types::DoseEvent;

// The 2-cpt single-dose concentration closed forms are the single source of truth
// in `crate::sens::two_cpt` (generic over `PkNum`); these f64 entry points delegate
// to the generic `*_g` at `T = f64` (issue #408 / Ron review #9). `#[inline]` keeps
// the hot path identical. The peripheral-amount helpers (`*_peripheral`) and the
// local `macro_rates` they share have no concentration analogue and stay here.

/// Compute macro-rate constants (alpha, beta, k21) from micro-constants —
/// delegates to the single generic source `sens::two_cpt::macro_rates_g` at
/// `T = f64` (the peripheral-amount helpers below are the only remaining f64
/// callers).
#[inline]
fn macro_rates(cl: f64, v1: f64, q: f64, v2: f64) -> (f64, f64, f64) {
    crate::sens::two_cpt::macro_rates_g::<f64>(cl, v1, q, v2)
}

/// Two-compartment IV bolus
/// C(t) = A*exp(-alpha*t) + B*exp(-beta*t)
#[inline]
pub fn two_cpt_iv_bolus(dose: &DoseEvent, t: f64, cl: f64, v1: f64, q: f64, v2: f64) -> f64 {
    two_cpt_iv_bolus_g::<f64>(dose.amt, t, cl, v1, q, v2)
}

/// Two-compartment infusion
#[inline]
pub fn two_cpt_infusion(dose: &DoseEvent, t: f64, cl: f64, v1: f64, q: f64, v2: f64) -> f64 {
    two_cpt_infusion_g::<f64>(dose.rate, dose.duration, dose.amt, t, cl, v1, q, v2)
}

/// Two-compartment oral absorption
/// C(t) = P*exp(-alpha*t) + Q*exp(-beta*t) + R*exp(-ka*t)
#[inline]
pub fn two_cpt_oral(dose: &DoseEvent, t: f64, cl: f64, v1: f64, q: f64, v2: f64, ka: f64) -> f64 {
    two_cpt_oral_f(dose, t, cl, v1, q, v2, ka, 1.0)
}

#[inline]
pub fn two_cpt_oral_f(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    ka: f64,
    f_bio: f64,
) -> f64 {
    two_cpt_oral_g::<f64>(dose.amt, t, cl, v1, q, v2, ka, f_bio)
}

/// Predict concentration from a single dose at elapsed time t using 2-cmt model.
pub fn two_cpt_predict(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    ka: Option<f64>,
    f_bio: Option<f64>,
) -> f64 {
    if dose.is_infusion() {
        two_cpt_infusion(dose, t, cl, v1, q, v2)
    } else if let Some(ka_val) = ka {
        two_cpt_oral_f(dose, t, cl, v1, q, v2, ka_val, f_bio.unwrap_or(1.0))
    } else {
        two_cpt_iv_bolus(dose, t, cl, v1, q, v2)
    }
}

// --- Steady-state (SS=1) variants ---
//
// For every exponential A·exp(-λ·t) in the single-dose response, the
// geometric-series sum Σ_{n=0}^∞ A·exp(-λ·(τ + n·II)) closed form is
// A·exp(-λ·τ) / (1 - exp(-λ·II)). The 2-cpt formulas just apply this
// substitution per eigenvalue (α, β, and ka for oral).

/// Helper: SS coefficient `1 / (1 - exp(-λ·ii))`. Returns 0 when the
/// denominator collapses (λ·ii ≤ 0 — should already be guarded out by
/// callers' `ii > 0` and `λ > 0` checks).
#[inline]
fn ss_coeff(lambda: f64, ii: f64) -> f64 {
    let denom = 1.0 - (-lambda * ii).exp();
    if denom > 0.0 {
        1.0 / denom
    } else {
        0.0
    }
}

/// Two-compartment IV bolus at steady state.
#[inline]
pub fn two_cpt_iv_bolus_ss(dose: &DoseEvent, t: f64, cl: f64, v1: f64, q: f64, v2: f64) -> f64 {
    two_cpt_iv_bolus_ss_g::<f64>(dose.amt, t, dose.ii, cl, v1, q, v2)
}

/// Two-compartment infusion at steady state, for any `T_inf` — including
/// overlapping pulses (`T_inf > II`). Evaluated at phase `t ∈ [0, II)`.
#[inline]
pub fn two_cpt_infusion_ss(dose: &DoseEvent, t: f64, cl: f64, v1: f64, q: f64, v2: f64) -> f64 {
    two_cpt_infusion_ss_g::<f64>(
        dose.rate,
        dose.duration,
        dose.amt,
        t,
        dose.ii,
        cl,
        v1,
        q,
        v2,
    )
}

/// Two-compartment oral absorption at steady state (with bioavailability).
#[inline]
pub fn two_cpt_oral_f_ss(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    ka: f64,
    f_bio: f64,
) -> f64 {
    two_cpt_oral_ss_g::<f64>(dose.amt, t, dose.ii, cl, v1, q, v2, ka, f_bio)
}

/// Two-compartment oral absorption at steady state (F = 1).
#[inline]
pub fn two_cpt_oral_ss(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    ka: f64,
) -> f64 {
    two_cpt_oral_f_ss(dose, t, cl, v1, q, v2, ka, 1.0)
}

// --- Peripheral compartment concentration (for compartment_states) ---

/// Peripheral concentration for a 2-cpt IV single dose at elapsed time tau.
/// Handles bolus, infusion, and SS=1 doses.
///
/// Derivation: C2(t) = k12/v2/diff * dose.amt * (exp(-β·t) − exp(-α·t))  for bolus.
/// Infusion uses the convolution integral; SS uses geometric-series factor.
pub(crate) fn two_cpt_iv_peripheral(
    dose: &DoseEvent,
    tau: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
) -> f64 {
    if tau < 0.0 || v1 <= 0.0 || v2 <= 0.0 || cl <= 0.0 || q < 0.0 {
        return 0.0;
    }
    let (alpha, beta, _k21) = macro_rates(cl, v1, q, v2);
    let diff = alpha - beta;
    if diff.abs() < 1e-12 || alpha.abs() < 1e-12 || beta.abs() < 1e-12 {
        return 0.0;
    }
    // Shared coefficient: (q/v1)/v2/diff = k12/(v2*diff)
    let k12_over_v2_diff = q / (v1 * v2 * diff);

    if dose.ss && dose.ii > 0.0 {
        let ii = dose.ii;
        if dose.is_infusion() {
            let dur = dose.duration;
            if dur <= 0.0 {
                // Degenerate infusion → treat as bolus SS
                return k12_over_v2_diff
                    * dose.amt
                    * ((-beta * tau).exp() * ss_coeff(beta, ii)
                        - (-alpha * tau).exp() * ss_coeff(alpha, ii));
            }
            if dur > ii {
                return 0.0;
            }
            // c2_coeff for infusion: rate * k12/(v2*diff)
            let c2 = dose.rate / diff * q / (v1 * v2);
            if tau <= dur {
                // Current pulse (during infusion) + past pulses (always "after").
                let current = c2
                    * ((1.0 - (-beta * tau).exp()) / beta - (1.0 - (-alpha * tau).exp()) / alpha);
                let past_b = c2 * (1.0 - (-beta * dur).exp()) / beta
                    * (-beta * (tau - dur)).exp()
                    * (-beta * ii).exp()
                    * ss_coeff(beta, ii);
                let past_a = c2 * (1.0 - (-alpha * dur).exp()) / alpha
                    * (-alpha * (tau - dur)).exp()
                    * (-alpha * ii).exp()
                    * ss_coeff(alpha, ii);
                current + past_b - past_a
            } else {
                let dt = tau - dur;
                c2 * ((1.0 - (-beta * dur).exp()) * (-beta * dt).exp() * ss_coeff(beta, ii) / beta
                    - (1.0 - (-alpha * dur).exp()) * (-alpha * dt).exp() * ss_coeff(alpha, ii)
                        / alpha)
            }
        } else {
            // Bolus SS
            k12_over_v2_diff
                * dose.amt
                * ((-beta * tau).exp() * ss_coeff(beta, ii)
                    - (-alpha * tau).exp() * ss_coeff(alpha, ii))
        }
    } else if dose.is_infusion() {
        let dur = dose.duration;
        if dur <= 0.0 {
            return k12_over_v2_diff * dose.amt * ((-beta * tau).exp() - (-alpha * tau).exp());
        }
        let c2 = dose.rate / diff * q / (v1 * v2);
        if tau <= dur {
            c2 * ((1.0 - (-beta * tau).exp()) / beta - (1.0 - (-alpha * tau).exp()) / alpha)
        } else {
            let dt = tau - dur;
            c2 * ((1.0 - (-beta * dur).exp()) * (-beta * dt).exp() / beta
                - (1.0 - (-alpha * dur).exp()) * (-alpha * dt).exp() / alpha)
        }
    } else {
        // Single-dose bolus
        k12_over_v2_diff * dose.amt * ((-beta * tau).exp() - (-alpha * tau).exp())
    }
}

/// Peripheral concentration for a 2-cpt oral single dose at elapsed time tau.
///
/// **Derivation — why only 3 terms (no exp(-k21·τ) term):**
///
/// In Laplace space:
/// ```text
///   A_central(s)  = F·D·ka·(s+k21) / [(s+ka)(s+α)(s+β)]
///   A_periph(s)   = k12 · A_central(s) / (s+k21)
///                 = k12·F·D·ka / [(s+ka)(s+α)(s+β)]
/// ```
/// The `(s+k21)` factor in `A_central(s)` cancels the `(s+k21)` pole of
/// `1/(s+k21)`, leaving only 3 poles: `−ka`, `−α`, `−β`. Partial-fraction
/// expansion gives the 3-term Bateman result below. The `exp(-ka·τ)` term is
/// implicitly correct: because `A_periph(0) = 0` (no drug in peripheral at
/// time 0), the residues must sum to zero, so the `exp(-ka·τ)` coefficient
/// equals minus the sum of the eigenvalue residues.
///
/// Formula (non-L'Hôpital):
///   D = f_bio·dose.amt·ka/v1
///   C2(τ) = (q/v2)·D · [ exp(-α·τ)/((ka-α)(β-α)) + exp(-β·τ)/((ka-β)(α-β)) + exp(-ka·τ)/((α-ka)(β-ka)) ]
///
/// L'Hôpital case (ka≈α or ka≈β): returns 0 (peripheral formula undefined at singularity).
pub(crate) fn two_cpt_oral_peripheral(
    dose: &DoseEvent,
    tau: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    ka: f64,
    f_bio: f64,
) -> f64 {
    if tau < 0.0 || v1 <= 0.0 || v2 <= 0.0 || cl <= 0.0 || q < 0.0 || ka <= 0.0 {
        return 0.0;
    }
    let (alpha, beta, _k21) = macro_rates(cl, v1, q, v2);
    let diff = alpha - beta; // alpha - beta
    if diff.abs() < 1e-12 {
        return 0.0;
    }
    if (ka - alpha).abs() < 1e-6 || (ka - beta).abs() < 1e-6 {
        return 0.0; // L'Hôpital case; peripheral formula reduces to 0 at this singularity
    }

    let d = f_bio * dose.amt * ka / v1;
    // Peripheral formula coefficients (exp terms mirror the central formula, but with k21 cancelled):
    // C2(τ) = (q/v2)·D·[exp(-α·τ)/((ka-α)(β-α)) + exp(-β·τ)/((ka-β)(α-β)) + exp(-ka·τ)/((α-ka)(β-ka))]
    let q_over_v2 = q / v2;

    if dose.ss && dose.ii > 0.0 {
        let ii = dose.ii;
        q_over_v2
            * d
            * ((-alpha * tau).exp() * ss_coeff(alpha, ii) / ((ka - alpha) * (beta - alpha))
                + (-beta * tau).exp() * ss_coeff(beta, ii) / ((ka - beta) * (alpha - beta))
                + (-ka * tau).exp() * ss_coeff(ka, ii) / ((alpha - ka) * (beta - ka)))
    } else {
        q_over_v2
            * d
            * ((-alpha * tau).exp() / ((ka - alpha) * (beta - alpha))
                + (-beta * tau).exp() / ((ka - beta) * (alpha - beta))
                + (-ka * tau).exp() / ((alpha - ka) * (beta - ka)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn bolus_dose(amt: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0)
    }

    fn infusion_dose(amt: f64, rate: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, rate, false, 0.0)
    }

    // Typical 2-cpt PK parameters
    const CL: f64 = 10.0;
    const V1: f64 = 100.0;
    const Q: f64 = 5.0;
    const V2: f64 = 200.0;

    // --- Macro rates ---

    #[test]
    fn test_macro_rates_positive() {
        let (alpha, beta, k21) = macro_rates(CL, V1, Q, V2);
        assert!(alpha > beta);
        assert!(alpha > 0.0);
        assert!(beta > 0.0);
        assert!(k21 > 0.0);
    }

    #[test]
    fn test_macro_rates_vieta() {
        // alpha * beta = k10 * k21 (Vieta's formula)
        let k10 = CL / V1;
        let k21 = Q / V2;
        let (alpha, beta, _) = macro_rates(CL, V1, Q, V2);
        assert_relative_eq!(alpha * beta, k10 * k21, epsilon = 1e-10);
    }

    // --- IV Bolus ---

    #[test]
    fn test_iv_bolus_at_time_zero() {
        let dose = bolus_dose(1000.0);
        let c = two_cpt_iv_bolus(&dose, 0.0, CL, V1, Q, V2);
        assert_relative_eq!(c, 1000.0 / V1, epsilon = 1e-10);
    }

    #[test]
    fn test_iv_bolus_approaches_zero() {
        let dose = bolus_dose(1000.0);
        let c = two_cpt_iv_bolus(&dose, 10000.0, CL, V1, Q, V2);
        assert!(c < 1e-20);
    }

    #[test]
    fn test_iv_bolus_monotone_decrease_eventually() {
        // After distribution phase, concentrations should decrease
        let dose = bolus_dose(1000.0);
        let c1 = two_cpt_iv_bolus(&dose, 50.0, CL, V1, Q, V2);
        let c2 = two_cpt_iv_bolus(&dose, 100.0, CL, V1, Q, V2);
        assert!(c2 < c1);
    }

    #[test]
    fn test_iv_bolus_guard_clauses() {
        let dose = bolus_dose(1000.0);
        assert_eq!(two_cpt_iv_bolus(&dose, -1.0, CL, V1, Q, V2), 0.0);
        assert_eq!(two_cpt_iv_bolus(&dose, 1.0, CL, 0.0, Q, V2), 0.0);
        assert_eq!(two_cpt_iv_bolus(&dose, 1.0, 0.0, V1, Q, V2), 0.0);
    }

    // --- Infusion ---

    #[test]
    fn test_infusion_during() {
        let dose = infusion_dose(1000.0, 100.0); // dur=10
        let c = two_cpt_infusion(&dose, 5.0, CL, V1, Q, V2);
        assert!(c > 0.0);
    }

    #[test]
    fn test_infusion_continuity_at_end() {
        let dose = infusion_dose(1000.0, 100.0); // dur=10
        let dur = 10.0;
        let c_at = two_cpt_infusion(&dose, dur, CL, V1, Q, V2);
        let c_after = two_cpt_infusion(&dose, dur + 1e-10, CL, V1, Q, V2);
        assert_relative_eq!(c_at, c_after, epsilon = 1e-5);
    }

    #[test]
    fn test_infusion_after_decays() {
        let dose = infusion_dose(1000.0, 100.0); // dur=10
        let c1 = two_cpt_infusion(&dose, 50.0, CL, V1, Q, V2);
        let c2 = two_cpt_infusion(&dose, 100.0, CL, V1, Q, V2);
        assert!(c2 < c1);
    }

    // --- Oral ---

    #[test]
    fn test_oral_at_time_zero() {
        let dose = bolus_dose(1000.0);
        let c = two_cpt_oral(&dose, 0.0, CL, V1, Q, V2, 1.5);
        assert_relative_eq!(c, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_oral_positive_at_peak() {
        let dose = bolus_dose(1000.0);
        let c = two_cpt_oral(&dose, 2.0, CL, V1, Q, V2, 1.5);
        assert!(c > 0.0);
    }

    #[test]
    fn test_oral_approaches_zero() {
        let dose = bolus_dose(1000.0);
        let c = two_cpt_oral(&dose, 10000.0, CL, V1, Q, V2, 1.5);
        assert!(c < 1e-20);
    }

    #[test]
    fn test_oral_bioavailability_scaling() {
        let dose = bolus_dose(1000.0);
        let c_full = two_cpt_oral_f(&dose, 2.0, CL, V1, Q, V2, 1.5, 1.0);
        let c_half = two_cpt_oral_f(&dose, 2.0, CL, V1, Q, V2, 1.5, 0.5);
        assert_relative_eq!(c_half / c_full, 0.5, epsilon = 1e-10);
    }

    // --- Predict dispatcher ---

    #[test]
    fn test_predict_routes_iv_bolus() {
        let dose = bolus_dose(1000.0);
        let direct = two_cpt_iv_bolus(&dose, 2.0, CL, V1, Q, V2);
        let via_predict = two_cpt_predict(&dose, 2.0, CL, V1, Q, V2, None, None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_routes_oral() {
        let dose = bolus_dose(1000.0);
        let direct = two_cpt_oral(&dose, 2.0, CL, V1, Q, V2, 1.5);
        let via_predict = two_cpt_predict(&dose, 2.0, CL, V1, Q, V2, Some(1.5), None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_routes_infusion() {
        let dose = infusion_dose(1000.0, 100.0);
        let direct = two_cpt_infusion(&dose, 2.0, CL, V1, Q, V2);
        let via_predict = two_cpt_predict(&dose, 2.0, CL, V1, Q, V2, None, None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    // --- Steady-state variants ---
    //
    // Each 2-cpt SS closed form is verified against a 300-term numerical sum
    // of the single-dose response shifted by n·II. 300 (vs. 200 used for the
    // 1-cpt suite) gives margin for the slower β eigenvalue tail.

    fn ss_bolus_dose(amt: f64, ii: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, 0.0, true, ii)
    }

    fn ss_infusion_dose(amt: f64, rate: f64, ii: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, rate, true, ii)
    }

    fn ss_numerical_sum<F: Fn(f64) -> f64>(t: f64, ii: f64, c_single: F) -> f64 {
        const N: usize = 300;
        (0..N).map(|n| c_single(t + (n as f64) * ii)).sum()
    }

    #[test]
    fn test_ss_iv_bolus_matches_numerical_sum() {
        let ii: f64 = 12.0;
        let dose = ss_bolus_dose(1000.0, ii);
        let single = bolus_dose(1000.0);
        for &t in &[0.0, 0.5, 3.0, 8.0, 11.9, 12.0, 24.0, 48.0] {
            let cf = two_cpt_iv_bolus_ss(&dose, t, CL, V1, Q, V2);
            let num = ss_numerical_sum(t, ii, |tt| two_cpt_iv_bolus(&single, tt, CL, V1, Q, V2));
            assert_relative_eq!(cf, num, epsilon = 1e-7, max_relative = 1e-7);
        }
    }

    #[test]
    fn test_ss_oral_matches_numerical_sum() {
        let ii: f64 = 24.0;
        let ka = 1.0;
        let dose = ss_bolus_dose(500.0, ii);
        let single = bolus_dose(500.0);
        for &t in &[0.0, 0.5, 2.0, 5.0, 12.0, 23.0, 48.0] {
            let cf = two_cpt_oral_ss(&dose, t, CL, V1, Q, V2, ka);
            let num = ss_numerical_sum(t, ii, |tt| two_cpt_oral(&single, tt, CL, V1, Q, V2, ka));
            assert_relative_eq!(cf, num, epsilon = 1e-7, max_relative = 1e-7);
        }
    }

    #[test]
    fn test_ss_oral_lhopital_ka_near_alpha_matches_numerical_sum() {
        // Pick ka exactly equal to α to trigger the L'Hopital branch.
        let ii: f64 = 24.0;
        let (alpha, _, _) = macro_rates(CL, V1, Q, V2);
        let ka = alpha;
        let dose = ss_bolus_dose(500.0, ii);
        let single = bolus_dose(500.0);
        for &t in &[0.5, 2.0, 5.0, 12.0, 23.0] {
            let cf = two_cpt_oral_ss(&dose, t, CL, V1, Q, V2, ka);
            let num = ss_numerical_sum(t, ii, |tt| two_cpt_oral(&single, tt, CL, V1, Q, V2, ka));
            assert_relative_eq!(cf, num, epsilon = 1e-6, max_relative = 1e-6);
        }
    }

    #[test]
    fn test_ss_oral_lhopital_ka_near_beta_matches_numerical_sum() {
        let ii: f64 = 24.0;
        let (_, beta, _) = macro_rates(CL, V1, Q, V2);
        let ka = beta;
        let dose = ss_bolus_dose(500.0, ii);
        let single = bolus_dose(500.0);
        for &t in &[0.5, 2.0, 12.0, 23.0] {
            let cf = two_cpt_oral_ss(&dose, t, CL, V1, Q, V2, ka);
            let num = ss_numerical_sum(t, ii, |tt| two_cpt_oral(&single, tt, CL, V1, Q, V2, ka));
            assert_relative_eq!(cf, num, epsilon = 1e-6, max_relative = 1e-6);
        }
    }

    #[test]
    fn test_oral_lhopital_matches_nonsingular_limit_independent() {
        // Independent (non-self-referential) check for the ka≈α / ka≈β L'Hôpital
        // limits. At ka = λ the singular branch must equal the limit of the
        // NON-singular formula as ka → λ — and the non-singular branch never
        // touches the L'Hôpital code, so it is independent truth (the prior tests
        // built "truth" from the same singular branch and so missed the bug).
        // Central average cancels the O(δ) term. Regression-guards the ~14% defect
        // (true ≈ 8.309 vs buggy ≈ 9.504) from the PR #381 review.
        let (cl, v1, q, v2, amt) = (10.0, 50.0, 15.0, 100.0, 1000.0);
        let (alpha, beta, _) = macro_rates(cl, v1, q, v2);
        let single = bolus_dose(amt);
        let delta = 1e-3;
        for &lambda in &[alpha, beta] {
            for &t in &[0.5, 2.0, 6.0] {
                let c_sing = two_cpt_oral(&single, t, cl, v1, q, v2, lambda);
                let c_lo = two_cpt_oral(&single, t, cl, v1, q, v2, lambda - delta);
                let c_hi = two_cpt_oral(&single, t, cl, v1, q, v2, lambda + delta);
                let truth = 0.5 * (c_lo + c_hi);
                assert_relative_eq!(c_sing, truth, max_relative = 2e-4, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_ss_oral_lhopital_matches_nonsingular_limit_independent() {
        // SS analog of the independent continuity check: the singular SS branch at
        // ka = λ must match the non-singular SS formula limit as ka → λ.
        let ii: f64 = 24.0;
        let (cl, v1, q, v2, amt) = (10.0, 50.0, 15.0, 100.0, 500.0);
        let (alpha, beta, _) = macro_rates(cl, v1, q, v2);
        let dose = ss_bolus_dose(amt, ii);
        let delta = 1e-3;
        for &lambda in &[alpha, beta] {
            for &t in &[0.5, 2.0, 12.0] {
                let c_sing = two_cpt_oral_ss(&dose, t, cl, v1, q, v2, lambda);
                let c_lo = two_cpt_oral_ss(&dose, t, cl, v1, q, v2, lambda - delta);
                let c_hi = two_cpt_oral_ss(&dose, t, cl, v1, q, v2, lambda + delta);
                let truth = 0.5 * (c_lo + c_hi);
                assert_relative_eq!(c_sing, truth, max_relative = 2e-4, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_ss_infusion_during_matches_numerical_sum() {
        let ii: f64 = 24.0;
        let rate = 100.0; // dur=10
        let dose = ss_infusion_dose(1000.0, rate, ii);
        let single = infusion_dose(1000.0, rate);
        for &t in &[0.0, 1.0, 5.0, 9.0, 10.0] {
            let cf = two_cpt_infusion_ss(&dose, t, CL, V1, Q, V2);
            let num = ss_numerical_sum(t, ii, |tt| two_cpt_infusion(&single, tt, CL, V1, Q, V2));
            assert_relative_eq!(cf, num, epsilon = 1e-7, max_relative = 1e-7);
        }
    }

    #[test]
    fn test_ss_infusion_after_matches_numerical_sum() {
        let ii: f64 = 24.0;
        let rate = 100.0; // dur=10
        let dose = ss_infusion_dose(1000.0, rate, ii);
        let single = infusion_dose(1000.0, rate);
        for &t in &[10.001, 15.0, 23.5, 48.0, 72.0] {
            let cf = two_cpt_infusion_ss(&dose, t, CL, V1, Q, V2);
            let num = ss_numerical_sum(t, ii, |tt| two_cpt_infusion(&single, tt, CL, V1, Q, V2));
            assert_relative_eq!(cf, num, epsilon = 1e-7, max_relative = 1e-7);
        }
    }

    #[test]
    fn test_ss_infusion_continuity_at_end_of_infusion() {
        let dose = ss_infusion_dose(1000.0, 100.0, 24.0);
        let c_at = two_cpt_infusion_ss(&dose, 10.0, CL, V1, Q, V2);
        let c_after = two_cpt_infusion_ss(&dose, 10.0 + 1e-10, CL, V1, Q, V2);
        assert_relative_eq!(c_at, c_after, epsilon = 1e-5);
    }

    #[test]
    fn test_ss_infusion_overlapping_matches_numerical_sum() {
        // Overlapping infusions (T_inf > II) must equal the explicit pulse-train
        // superposition (#379). amt=1000/rate=500 → duration=2, ii=1 (2 overlap);
        // amt=900/rate=300 → duration=3, ii=2.
        // A 2-cpt terminal eigenvalue can be small (≈0.016 here), so the past
        // pulse train decays slowly — sum far more pulses than `ss_numerical_sum`
        // for a converged reference.
        let big_sum = |t: f64, ii: f64, c_single: &dyn Fn(f64) -> f64| -> f64 {
            (0..50_000).map(|n| c_single(t + (n as f64) * ii)).sum()
        };
        for &(rate, amt, ii) in &[(500.0_f64, 1000.0_f64, 1.0_f64), (300.0, 900.0, 2.0)] {
            let dose = ss_infusion_dose(amt, rate, ii);
            let single = infusion_dose(amt, rate);
            assert!(dose.duration > ii, "fixture must overlap");
            for &t in &[0.0, 0.3 * ii, 0.5 * ii, 0.9 * ii, 0.999 * ii] {
                let cf = two_cpt_infusion_ss(&dose, t, CL, V1, Q, V2);
                let num = big_sum(t, ii, &|tt| two_cpt_infusion(&single, tt, CL, V1, Q, V2));
                assert_relative_eq!(cf, num, epsilon = 1e-6, max_relative = 1e-6);
            }
        }
    }

    #[test]
    fn test_ss_oral_with_bioavailability_scales() {
        let dose = ss_bolus_dose(500.0, 24.0);
        let c_full = two_cpt_oral_f_ss(&dose, 4.0, CL, V1, Q, V2, 1.0, 1.0);
        let c_half = two_cpt_oral_f_ss(&dose, 4.0, CL, V1, Q, V2, 1.0, 0.5);
        assert_relative_eq!(c_half / c_full, 0.5, epsilon = 1e-10);
    }
}
