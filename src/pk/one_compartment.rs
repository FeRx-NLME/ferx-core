use crate::sens::one_cpt::{
    one_cpt_infusion_g, one_cpt_infusion_ss_g, one_cpt_iv_bolus_g, one_cpt_iv_bolus_ss_g,
    one_cpt_oral_g, one_cpt_oral_ss_g,
};
use crate::types::DoseEvent;

// The closed forms below are the single source of truth: they live once, generic
// over `PkNum`, in `crate::sens::one_cpt` (the "sens model"). These f64 entry
// points delegate to the generic `*_g` at `T = f64`, so the prediction path and
// the `Dual2` sensitivity path can never drift (issue #408 / Ron review #9). The
// `#[inline]` delegators monomorphise to the same machine code the hand-written
// f64 forms produced, so the hot superposition path is unchanged.

/// One-compartment IV bolus: C(t) = (Dose/V) * exp(-k*t)
#[inline]
pub fn one_cpt_iv_bolus(dose: &DoseEvent, t: f64, cl: f64, v: f64) -> f64 {
    one_cpt_iv_bolus_g::<f64>(dose.amt, t, cl, v)
}

/// One-compartment infusion
/// During infusion (t <= T): C(t) = (Rate/CL) * (1 - exp(-k*t))
/// After infusion (t > T):   C(t) = (Rate/CL) * (1 - exp(-k*T)) * exp(-k*(t-T))
#[inline]
pub fn one_cpt_infusion(dose: &DoseEvent, t: f64, cl: f64, v: f64) -> f64 {
    one_cpt_infusion_g::<f64>(dose.rate, dose.duration, dose.amt, t, cl, v)
}

/// One-compartment oral absorption
/// C(t) = (F*Dose*KA) / (V*(KA - k)) * [exp(-k*t) - exp(-KA*t)]
/// Handles singularity when KA ≈ k via L'Hopital limit
#[inline]
pub fn one_cpt_oral(dose: &DoseEvent, t: f64, cl: f64, v: f64, ka: f64) -> f64 {
    one_cpt_oral_f(dose, t, cl, v, ka, 1.0)
}

#[inline]
pub fn one_cpt_oral_f(dose: &DoseEvent, t: f64, cl: f64, v: f64, ka: f64, f_bio: f64) -> f64 {
    one_cpt_oral_g::<f64>(dose.amt, t, cl, v, ka, f_bio)
}

/// Predict concentration from a single dose at elapsed time t using 1-cmt model.
/// Parameters are passed as a HashMap-like slice: [cl, v, ka (optional), f (optional)]
pub fn one_cpt_predict(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v: f64,
    ka: Option<f64>,
    f_bio: Option<f64>,
) -> f64 {
    if dose.is_infusion() {
        one_cpt_infusion(dose, t, cl, v)
    } else if let Some(ka_val) = ka {
        one_cpt_oral_f(dose, t, cl, v, ka_val, f_bio.unwrap_or(1.0))
    } else {
        one_cpt_iv_bolus(dose, t, cl, v)
    }
}

// --- Steady-state (SS=1) variants ---
//
// NONMEM SS=1 semantics: at the time of the SS dose, the compartmental state
// is initialised to the steady-state value from an infinite-past pulse train
// at interval `dose.ii`. After the SS dose, no further pulses are implicitly
// continued — subsequent dynamics are the natural decay of the SS-loaded
// system (or the superposition with any explicit follow-up dose records).
//
// Each closed form is the limit of the geometric series
//
//     C_ss(τ) = Σ_{n=0}^∞ C_single(τ + n · II)
//
// where C_single is the corresponding single-dose response. The series
// converges as a geometric series in `exp(-λ·II)` for every disposition
// rate `λ`, so the closed form is valid for all τ ≥ 0 (not just τ ∈ [0, II)).

/// One-compartment IV bolus at steady state.
#[inline]
pub fn one_cpt_iv_bolus_ss(dose: &DoseEvent, t: f64, cl: f64, v: f64) -> f64 {
    one_cpt_iv_bolus_ss_g::<f64>(dose.amt, t, dose.ii, cl, v)
}

/// One-compartment oral absorption at steady state (with bioavailability).
#[inline]
pub fn one_cpt_oral_f_ss(dose: &DoseEvent, t: f64, cl: f64, v: f64, ka: f64, f_bio: f64) -> f64 {
    one_cpt_oral_ss_g::<f64>(dose.amt, t, dose.ii, cl, v, ka, f_bio)
}

/// One-compartment oral absorption at steady state (F = 1).
#[inline]
pub fn one_cpt_oral_ss(dose: &DoseEvent, t: f64, cl: f64, v: f64, ka: f64) -> f64 {
    one_cpt_oral_f_ss(dose, t, cl, v, ka, 1.0)
}

/// Depot amount for a single oral bolus (or SS oral) dose at elapsed time tau.
/// Returns 0 for infusion doses (infusions bypass the depot compartment).
pub(crate) fn one_cpt_oral_depot(dose: &DoseEvent, tau: f64, ka: f64, f_bio: f64) -> f64 {
    if tau < 0.0 || ka <= 0.0 || dose.is_infusion() {
        return 0.0;
    }
    let a = f_bio * dose.amt * (-ka * tau).exp();
    if dose.ss && dose.ii > 0.0 {
        let denom = 1.0 - (-ka * dose.ii).exp();
        if denom > 0.0 {
            a / denom
        } else {
            0.0
        }
    } else {
        a
    }
}

/// One-compartment infusion at steady state, for any `T_inf` — including
/// overlapping pulses (`T_inf > II`), where several infusions are simultaneously
/// active. Evaluated at phase `t ∈ [0, II)`.
#[inline]
pub fn one_cpt_infusion_ss(dose: &DoseEvent, t: f64, cl: f64, v: f64) -> f64 {
    one_cpt_infusion_ss_g::<f64>(dose.rate, dose.duration, dose.amt, t, dose.ii, cl, v)
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

    /// Numerical reference: sum the single-dose response at τ + n·II for
    /// n = 0..N. With N=200 pulses, even λ·II as small as 0.05 leaves a
    /// truncation tail below 1e-4 of the SS value — adequate at the 1e-6
    /// tolerance the closed-form tests use.
    fn ss_numerical_sum<F: Fn(f64) -> f64>(t: f64, ii: f64, c_single: F) -> f64 {
        const N: usize = 200;
        (0..N).map(|n| c_single(t + (n as f64) * ii)).sum()
    }

    // --- IV Bolus ---

    #[test]
    fn test_iv_bolus_at_time_zero() {
        let dose = bolus_dose(1000.0);
        let c = one_cpt_iv_bolus(&dose, 0.0, 10.0, 100.0);
        assert_relative_eq!(c, 10.0, epsilon = 1e-10); // Dose/V = 1000/100
    }

    #[test]
    fn test_iv_bolus_decay() {
        let dose = bolus_dose(1000.0);
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let k = cl / v; // 0.1
        let t = 5.0;
        let expected = (1000.0 / v) * (-k * t).exp();
        let c = one_cpt_iv_bolus(&dose, t, cl, v);
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_iv_bolus_approaches_zero() {
        let dose = bolus_dose(1000.0);
        let c = one_cpt_iv_bolus(&dose, 1000.0, 10.0, 100.0);
        assert!(c < 1e-30);
    }

    #[test]
    fn test_iv_bolus_negative_time() {
        let dose = bolus_dose(1000.0);
        assert_eq!(one_cpt_iv_bolus(&dose, -1.0, 10.0, 100.0), 0.0);
    }

    #[test]
    fn test_iv_bolus_zero_volume() {
        let dose = bolus_dose(1000.0);
        assert_eq!(one_cpt_iv_bolus(&dose, 1.0, 10.0, 0.0), 0.0);
    }

    // --- Infusion ---

    #[test]
    fn test_infusion_during() {
        let dose = infusion_dose(1000.0, 100.0); // duration = 10h
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let k = cl / v;
        let t: f64 = 5.0; // during infusion
        let expected = (100.0 / cl) * (1.0 - (-k * t).exp());
        let c = one_cpt_infusion(&dose, t, cl, v);
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_infusion_after() {
        let dose = infusion_dose(1000.0, 100.0); // duration = 10h
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let k = cl / v;
        let dur: f64 = 10.0;
        let t: f64 = 15.0; // after infusion
        let expected = (100.0 / cl) * (1.0 - (-k * dur).exp()) * (-k * (t - dur)).exp();
        let c = one_cpt_infusion(&dose, t, cl, v);
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_infusion_continuity_at_end() {
        let dose = infusion_dose(1000.0, 100.0); // duration = 10
        let cl = 10.0;
        let v = 100.0;
        let dur = 10.0;
        let c_at = one_cpt_infusion(&dose, dur, cl, v);
        let c_after = one_cpt_infusion(&dose, dur + 1e-10, cl, v);
        assert_relative_eq!(c_at, c_after, epsilon = 1e-6);
    }

    // --- Oral ---

    #[test]
    fn test_oral_at_time_zero() {
        let dose = bolus_dose(1000.0);
        let c = one_cpt_oral(&dose, 0.0, 10.0, 100.0, 1.0);
        assert_relative_eq!(c, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_oral_known_value() {
        let dose = bolus_dose(1000.0);
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let ka: f64 = 1.5;
        let k = cl / v;
        let t: f64 = 2.0;
        let expected = (1000.0 * ka / (v * (ka - k))) * ((-k * t).exp() - (-ka * t).exp());
        let c = one_cpt_oral(&dose, t, cl, v, ka);
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_oral_singularity_ka_equals_ke() {
        // When ka ≈ k, L'Hopital: C(t) = (D*ka/V) * t * exp(-k*t)
        let dose = bolus_dose(1000.0);
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let k = cl / v; // 0.1
        let ka = k; // singularity
        let t: f64 = 5.0;
        let expected = (1000.0 * ka / v) * t * (-k * t).exp();
        let c = one_cpt_oral(&dose, t, cl, v, ka);
        assert_relative_eq!(c, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_oral_with_bioavailability() {
        let dose = bolus_dose(1000.0);
        let c_full = one_cpt_oral_f(&dose, 2.0, 10.0, 100.0, 1.5, 1.0);
        let c_half = one_cpt_oral_f(&dose, 2.0, 10.0, 100.0, 1.5, 0.5);
        assert_relative_eq!(c_half / c_full, 0.5, epsilon = 1e-10);
    }

    // --- Predict dispatcher ---

    #[test]
    fn test_predict_routes_iv_bolus() {
        let dose = bolus_dose(1000.0);
        let direct = one_cpt_iv_bolus(&dose, 2.0, 10.0, 100.0);
        let via_predict = one_cpt_predict(&dose, 2.0, 10.0, 100.0, None, None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_routes_oral() {
        let dose = bolus_dose(1000.0);
        let direct = one_cpt_oral(&dose, 2.0, 10.0, 100.0, 1.5);
        let via_predict = one_cpt_predict(&dose, 2.0, 10.0, 100.0, Some(1.5), None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_routes_infusion() {
        let dose = infusion_dose(1000.0, 100.0);
        let direct = one_cpt_infusion(&dose, 2.0, 10.0, 100.0);
        let via_predict = one_cpt_predict(&dose, 2.0, 10.0, 100.0, None, None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    // --- Steady-state variants ---
    //
    // Each closed-form SS function is verified against a 200-term numerical
    // sum of the single-dose response shifted by n·II. This is the exact
    // definition the closed forms came from, so agreement to ~1e-9 is the
    // expected tolerance (set looser to leave margin for the truncation
    // tail when λ·II is small).

    fn ss_bolus_dose(amt: f64, ii: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, 0.0, true, ii)
    }

    fn ss_infusion_dose(amt: f64, rate: f64, ii: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, rate, true, ii)
    }

    #[test]
    fn test_ss_iv_bolus_matches_numerical_sum() {
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let ii: f64 = 12.0;
        let dose = ss_bolus_dose(1000.0, ii);
        let single = bolus_dose(1000.0);
        for &t in &[0.0, 0.5, 3.0, 8.0, 11.9, 12.0, 24.0] {
            let cf = one_cpt_iv_bolus_ss(&dose, t, cl, v);
            let num = ss_numerical_sum(t, ii, |tt| one_cpt_iv_bolus(&single, tt, cl, v));
            assert_relative_eq!(cf, num, epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    #[test]
    fn test_ss_iv_bolus_at_zero_equals_steady_state_amount() {
        // C_ss(0) = (D/V) / (1 - exp(-k·II))
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let ii: f64 = 12.0;
        let k = cl / v;
        let dose = ss_bolus_dose(1000.0, ii);
        let c = one_cpt_iv_bolus_ss(&dose, 0.0, cl, v);
        let expected = (1000.0 / v) / (1.0 - (-k * ii).exp());
        assert_relative_eq!(c, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_ss_oral_matches_numerical_sum() {
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let ka: f64 = 1.0;
        let ii: f64 = 24.0;
        let dose = ss_bolus_dose(500.0, ii);
        let single = bolus_dose(500.0);
        for &t in &[0.0, 0.5, 2.0, 5.0, 12.0, 23.0, 48.0] {
            let cf = one_cpt_oral_ss(&dose, t, cl, v, ka);
            let num = ss_numerical_sum(t, ii, |tt| one_cpt_oral(&single, tt, cl, v, ka));
            assert_relative_eq!(cf, num, epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    #[test]
    fn test_ss_oral_singularity_ka_equals_ke_matches_numerical_sum() {
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let k = cl / v; // 0.1
        let ka = k; // singularity
        let ii: f64 = 24.0;
        let dose = ss_bolus_dose(500.0, ii);
        let single = bolus_dose(500.0);
        for &t in &[0.5, 2.0, 5.0, 12.0, 23.0] {
            let cf = one_cpt_oral_ss(&dose, t, cl, v, ka);
            let num = ss_numerical_sum(t, ii, |tt| one_cpt_oral(&single, tt, cl, v, ka));
            assert_relative_eq!(cf, num, epsilon = 1e-8, max_relative = 1e-8);
        }
    }

    #[test]
    fn test_ss_oral_with_bioavailability_scales() {
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let ka: f64 = 1.0;
        let ii: f64 = 24.0;
        let dose = ss_bolus_dose(500.0, ii);
        let c_full = one_cpt_oral_f_ss(&dose, 4.0, cl, v, ka, 1.0);
        let c_half = one_cpt_oral_f_ss(&dose, 4.0, cl, v, ka, 0.5);
        assert_relative_eq!(c_half / c_full, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_ss_infusion_during_matches_numerical_sum() {
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let rate: f64 = 100.0; // duration = 10
        let ii: f64 = 24.0;
        let dose = ss_infusion_dose(1000.0, rate, ii);
        let single = infusion_dose(1000.0, rate);
        for &t in &[0.0, 1.0, 5.0, 9.0, 10.0] {
            let cf = one_cpt_infusion_ss(&dose, t, cl, v);
            let num = ss_numerical_sum(t, ii, |tt| one_cpt_infusion(&single, tt, cl, v));
            assert_relative_eq!(cf, num, epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    #[test]
    fn test_ss_infusion_after_matches_numerical_sum() {
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let rate: f64 = 100.0; // duration = 10
        let ii: f64 = 24.0;
        let dose = ss_infusion_dose(1000.0, rate, ii);
        let single = infusion_dose(1000.0, rate);
        for &t in &[10.001, 15.0, 23.5, 48.0, 72.0] {
            let cf = one_cpt_infusion_ss(&dose, t, cl, v);
            let num = ss_numerical_sum(t, ii, |tt| one_cpt_infusion(&single, tt, cl, v));
            assert_relative_eq!(cf, num, epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    #[test]
    fn test_ss_infusion_continuity_at_end_of_infusion() {
        let cl = 10.0;
        let v = 100.0;
        let rate = 100.0; // duration = 10
        let ii = 24.0;
        let dose = ss_infusion_dose(1000.0, rate, ii);
        let c_at = one_cpt_infusion_ss(&dose, 10.0, cl, v);
        let c_after = one_cpt_infusion_ss(&dose, 10.0 + 1e-10, cl, v);
        assert_relative_eq!(c_at, c_after, epsilon = 1e-6);
    }

    #[test]
    fn test_ss_with_zero_ii_returns_zero() {
        // SS=1 with II=0 is ill-defined; expect 0 (api.rs warns separately).
        let dose = DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 0.0);
        assert_eq!(one_cpt_iv_bolus_ss(&dose, 5.0, 10.0, 100.0), 0.0);
    }

    #[test]
    fn test_ss_infusion_overlapping_matches_numerical_sum() {
        // Overlapping infusions (T_inf > II): several pulses are simultaneously
        // active, so the closed form must still equal the explicit superposition
        // of single-dose responses (#379). Two regimes: duration=5/II=2 (≈2.5
        // pulses overlap) and duration=7/II=3.
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        for &(rate, amt, ii) in &[(200.0_f64, 1000.0_f64, 2.0_f64), (140.0, 980.0, 3.0)] {
            let dose = ss_infusion_dose(amt, rate, ii);
            let single = infusion_dose(amt, rate);
            assert!(dose.duration > ii, "fixture must overlap");
            // Phase τ ∈ [0, II): the physically meaningful SS sampling window.
            for &t in &[0.0, 0.3, 0.5, 0.9 * ii, 0.999 * ii] {
                let cf = one_cpt_infusion_ss(&dose, t, cl, v);
                let num = ss_numerical_sum(t, ii, |tt| one_cpt_infusion(&single, tt, cl, v));
                assert_relative_eq!(cf, num, epsilon = 1e-8, max_relative = 1e-7);
            }
        }
    }
}
