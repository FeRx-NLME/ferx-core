use crate::types::DoseEvent;

/// Compute macro-rate constants (alpha, beta) from micro-constants.
/// Uses Vieta's formula for beta to avoid catastrophic cancellation
/// when s >> sqrt(s^2 - 4d).
fn macro_rates(cl: f64, v1: f64, q: f64, v2: f64) -> (f64, f64, f64) {
    let k10 = cl / v1;
    let k12 = q / v1;
    let k21 = q / v2;
    let s = k10 + k12 + k21;
    let d = k10 * k21;
    let disc = {
        let x = s * s - 4.0 * d;
        if x > 0.0 {
            x.sqrt()
        } else {
            0.0
        }
    };
    let alpha = (s + disc) / 2.0;
    // Vieta's formula: alpha * beta = d, so beta = d / alpha
    // This avoids subtracting two nearly-equal large numbers.
    let beta = if alpha > 1e-30 { d / alpha } else { 0.0 };
    (alpha, beta, k21)
}

/// Two-compartment IV bolus
/// C(t) = A*exp(-alpha*t) + B*exp(-beta*t)
pub fn two_cpt_iv_bolus(dose: &DoseEvent, t: f64, cl: f64, v1: f64, q: f64, v2: f64) -> f64 {
    if t < 0.0 || v1 <= 0.0 || cl <= 0.0 {
        return 0.0;
    }
    let (alpha, beta, k21) = macro_rates(cl, v1, q, v2);
    let diff = alpha - beta;
    if diff.abs() < 1e-12 {
        return 0.0;
    }

    let a = (dose.amt / v1) * (alpha - k21) / diff;
    let b = (dose.amt / v1) * (k21 - beta) / diff;

    a * (-alpha * t).exp() + b * (-beta * t).exp()
}

/// Two-compartment infusion
pub fn two_cpt_infusion(dose: &DoseEvent, t: f64, cl: f64, v1: f64, q: f64, v2: f64) -> f64 {
    if t < 0.0 || v1 <= 0.0 || cl <= 0.0 {
        return 0.0;
    }
    let (alpha, beta, k21) = macro_rates(cl, v1, q, v2);
    let diff = alpha - beta;
    if diff.abs() < 1e-12 || alpha.abs() < 1e-12 || beta.abs() < 1e-12 {
        return 0.0;
    }

    let rate = dose.rate;
    let dur = dose.duration;
    if dur <= 0.0 {
        return two_cpt_iv_bolus(dose, t, cl, v1, q, v2);
    }

    let a_coeff = (rate / v1) * (alpha - k21) / (diff * alpha);
    let b_coeff = (rate / v1) * (k21 - beta) / (diff * beta);

    if t <= dur {
        a_coeff * (1.0 - (-alpha * t).exp()) + b_coeff * (1.0 - (-beta * t).exp())
    } else {
        let dt = t - dur;
        a_coeff * (1.0 - (-alpha * dur).exp()) * (-alpha * dt).exp()
            + b_coeff * (1.0 - (-beta * dur).exp()) * (-beta * dt).exp()
    }
}

/// Two-compartment oral absorption
/// C(t) = P*exp(-alpha*t) + Q*exp(-beta*t) + R*exp(-ka*t)
pub fn two_cpt_oral(dose: &DoseEvent, t: f64, cl: f64, v1: f64, q: f64, v2: f64, ka: f64) -> f64 {
    two_cpt_oral_f(dose, t, cl, v1, q, v2, ka, 1.0)
}

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
    if t < 0.0 || v1 <= 0.0 || cl <= 0.0 || ka <= 0.0 {
        return 0.0;
    }
    let (alpha, beta, k21) = macro_rates(cl, v1, q, v2);
    let diff = alpha - beta;
    if diff.abs() < 1e-12 {
        return 0.0;
    }

    let d = f_bio * dose.amt * ka / v1;

    // Standard formula:
    //   C(t) = d * [ (k21-α)/((ka-α)(β-α)) · e^{-αt}
    //              + (k21-β)/((ka-β)(α-β)) · e^{-βt}
    //              + (k21-ka)/((α-ka)(β-ka)) · e^{-ka·t} ]
    //
    // Handle singularities when ka ≈ alpha or ka ≈ beta via L'Hopital limits.
    let p = if (ka - alpha).abs() < 1e-6 {
        d * (alpha - k21) / diff * t * (-alpha * t).exp()
    } else {
        d * (k21 - alpha) / ((ka - alpha) * (beta - alpha)) * (-alpha * t).exp()
    };

    let q_val = if (ka - beta).abs() < 1e-6 {
        d * (k21 - beta) / diff * t * (-beta * t).exp()
    } else {
        d * (k21 - beta) / ((ka - beta) * (alpha - beta)) * (-beta * t).exp()
    };

    let r = if (ka - alpha).abs() < 1e-6 || (ka - beta).abs() < 1e-6 {
        0.0
    } else {
        d * (k21 - ka) / ((alpha - ka) * (beta - ka)) * (-ka * t).exp()
    };

    p + q_val + r
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
pub fn two_cpt_iv_bolus_ss(dose: &DoseEvent, t: f64, cl: f64, v1: f64, q: f64, v2: f64) -> f64 {
    if t < 0.0 || v1 <= 0.0 || cl <= 0.0 || v2 <= 0.0 || q < 0.0 || dose.ii <= 0.0 {
        return 0.0;
    }
    let (alpha, beta, k21) = macro_rates(cl, v1, q, v2);
    let diff = alpha - beta;
    if diff.abs() < 1e-12 {
        return 0.0;
    }
    let ii = dose.ii;
    let a = (dose.amt / v1) * (alpha - k21) / diff;
    let b = (dose.amt / v1) * (k21 - beta) / diff;
    a * (-alpha * t).exp() * ss_coeff(alpha, ii) + b * (-beta * t).exp() * ss_coeff(beta, ii)
}

/// Two-compartment infusion at steady state.
///
/// Closed form requires `T_inf ≤ II` (non-overlapping infusions); returns 0.0
/// otherwise. The `api.rs` warning catches this case for users.
pub fn two_cpt_infusion_ss(dose: &DoseEvent, t: f64, cl: f64, v1: f64, q: f64, v2: f64) -> f64 {
    if t < 0.0 || v1 <= 0.0 || cl <= 0.0 || v2 <= 0.0 || q < 0.0 || dose.ii <= 0.0 {
        return 0.0;
    }
    let dur = dose.duration;
    if dur <= 0.0 {
        return two_cpt_iv_bolus_ss(dose, t, cl, v1, q, v2);
    }
    if dur > dose.ii {
        return 0.0;
    }
    let (alpha, beta, k21) = macro_rates(cl, v1, q, v2);
    let diff = alpha - beta;
    if diff.abs() < 1e-12 || alpha.abs() < 1e-12 || beta.abs() < 1e-12 {
        return 0.0;
    }
    let ii = dose.ii;
    let rate = dose.rate;
    let a_coeff = (rate / v1) * (alpha - k21) / (diff * alpha);
    let b_coeff = (rate / v1) * (k21 - beta) / (diff * beta);

    // Past-pulses contribution (n ≥ 1): always "after-infusion".
    let exp_neg_a_ii = (-alpha * ii).exp();
    let exp_neg_b_ii = (-beta * ii).exp();
    let past_a = a_coeff
        * (1.0 - (-alpha * dur).exp())
        * (-alpha * (t - dur)).exp()
        * exp_neg_a_ii
        * ss_coeff(alpha, ii);
    let past_b = b_coeff
        * (1.0 - (-beta * dur).exp())
        * (-beta * (t - dur)).exp()
        * exp_neg_b_ii
        * ss_coeff(beta, ii);
    if t <= dur {
        // Current pulse is during infusion.
        a_coeff * (1.0 - (-alpha * t).exp()) + b_coeff * (1.0 - (-beta * t).exp()) + past_a + past_b
    } else {
        // Current pulse is after — all pulses are "after"; combine with the
        // past-pulses tail by replacing the (n ≥ 1) sum with (n ≥ 0).
        let dt = t - dur;
        a_coeff * (1.0 - (-alpha * dur).exp()) * (-alpha * dt).exp() * ss_coeff(alpha, ii)
            + b_coeff * (1.0 - (-beta * dur).exp()) * (-beta * dt).exp() * ss_coeff(beta, ii)
    }
}

/// Two-compartment oral absorption at steady state (with bioavailability).
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
    if t < 0.0 || v1 <= 0.0 || cl <= 0.0 || ka <= 0.0 || v2 <= 0.0 || q < 0.0 || dose.ii <= 0.0 {
        return 0.0;
    }
    let (alpha, beta, k21) = macro_rates(cl, v1, q, v2);
    let diff = alpha - beta;
    if diff.abs() < 1e-12 {
        return 0.0;
    }
    let ii = dose.ii;
    let d = f_bio * dose.amt * ka / v1;

    // Standard 2-cpt oral with per-eigenvalue SS geometric-series factor.
    // L'Hopital limits apply when ka ≈ α or ka ≈ β; the SS sum of
    // (τ + n·II)·exp(-λ·(τ + n·II)) closed form is
    //   exp(-λ·τ) · [τ/(1-x) + II·x/(1-x)²]  with x = exp(-λ·ii).
    fn lhopital_ss_sum(tau: f64, lambda: f64, ii: f64) -> f64 {
        let x = (-lambda * ii).exp();
        let one_minus_x = 1.0 - x;
        if one_minus_x <= 0.0 {
            return 0.0;
        }
        (-lambda * tau).exp() * (tau / one_minus_x + ii * x / (one_minus_x * one_minus_x))
    }

    let p = if (ka - alpha).abs() < 1e-6 {
        d * (alpha - k21) / diff * lhopital_ss_sum(t, alpha, ii)
    } else {
        d * (k21 - alpha) / ((ka - alpha) * (beta - alpha))
            * (-alpha * t).exp()
            * ss_coeff(alpha, ii)
    };
    let q_val = if (ka - beta).abs() < 1e-6 {
        d * (k21 - beta) / diff * lhopital_ss_sum(t, beta, ii)
    } else {
        d * (k21 - beta) / ((ka - beta) * (alpha - beta)) * (-beta * t).exp() * ss_coeff(beta, ii)
    };
    let r = if (ka - alpha).abs() < 1e-6 || (ka - beta).abs() < 1e-6 {
        0.0
    } else {
        d * (k21 - ka) / ((alpha - ka) * (beta - ka)) * (-ka * t).exp() * ss_coeff(ka, ii)
    };
    p + q_val + r
}

/// Two-compartment oral absorption at steady state (F = 1).
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
    fn test_ss_infusion_with_t_inf_gt_ii_returns_zero() {
        // amt=1000, rate=500 → duration=2; ii=1 → t_inf > ii.
        let dose = DoseEvent::new(0.0, 1000.0, 1, 500.0, true, 1.0);
        assert_eq!(two_cpt_infusion_ss(&dose, 0.5, CL, V1, Q, V2), 0.0);
    }

    #[test]
    fn test_ss_oral_with_bioavailability_scales() {
        let dose = ss_bolus_dose(500.0, 24.0);
        let c_full = two_cpt_oral_f_ss(&dose, 4.0, CL, V1, Q, V2, 1.0, 1.0);
        let c_half = two_cpt_oral_f_ss(&dose, 4.0, CL, V1, Q, V2, 1.0, 0.5);
        assert_relative_eq!(c_half / c_full, 0.5, epsilon = 1e-10);
    }
}
