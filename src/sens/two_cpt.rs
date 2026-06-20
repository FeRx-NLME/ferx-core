//! Two-compartment analytic solutions generic over [`PkNum`] — the 2-cpt
//! counterpart to [`super::one_cpt`]. Written once and monomorphised for `f64`
//! (prediction) and [`Dual2`](super::dual2::Dual2) (exact PK-parameter
//! sensitivities). Mirrors [`crate::pk::two_compartment`]; only the central-
//! compartment concentration is needed for the likelihood.
// PK closed forms carry many positional params (dose descriptors + 4 PK params);
// the wide signatures are inherent, not a smell.
#![allow(clippy::too_many_arguments)]

use super::num::PkNum;
use crate::types::DoseEvent;

/// Macro-rate constants `(α, β, k21)` from the PK params, via Vieta's `β = d/α`
/// (avoids catastrophic cancellation). Branches on `.val()` only.
pub(crate) fn macro_rates_g<T: PkNum>(cl: T, v1: T, q: T, v2: T) -> (T, T, T) {
    let k10 = cl / v1;
    let k12 = q / v1;
    let k21 = q / v2;
    let s = k10 + k12 + k21;
    let d = k10 * k21;
    let disc = {
        let x = s * s - T::from_f64(4.0) * d;
        if x.val() > 0.0 {
            x.sqrt()
        } else {
            T::from_f64(0.0)
        }
    };
    let alpha = (s + disc) * T::from_f64(0.5);
    let beta = if alpha.val() > 1e-30 {
        d / alpha
    } else {
        T::from_f64(0.0)
    };
    (alpha, beta, k21)
}

/// 2-cpt IV bolus: `C = A·e^{−αt} + B·e^{−βt}`. `t` is generic so the caller can
/// seed it as a dual carrying the lagtime sensitivity (`∂t/∂lagtime = −1`).
pub fn two_cpt_iv_bolus_g<T: PkNum>(amt: f64, t: T, cl: T, v1: T, q: T, v2: T) -> T {
    if t.val() < 0.0 || v1.val() <= 0.0 || cl.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    let (alpha, beta, k21) = macro_rates_g(cl, v1, q, v2);
    let diff = alpha - beta;
    // Confluent eigenvalues (α≈β): the spectral form carries 1/(α−β) residue
    // poles, undefined at exact root coalescence. Measure-zero for physical PK
    // params (distinct real eigenvalues); return 0 rather than the confluent
    // (A+B·t)·e^{−αt} limit — a deliberate edge (PR #381, review #1).
    if diff.val().abs() < 1e-12 {
        return T::from_f64(0.0);
    }
    let amt_v1 = T::from_f64(amt) / v1;
    let a = amt_v1 * (alpha - k21) / diff;
    let b = amt_v1 * (k21 - beta) / diff;
    a * (-(alpha * t)).exp() + b * (-(beta * t)).exp()
}

/// 2-cpt infusion (zero-order input, duration `dur`, rate `rate`).
pub fn two_cpt_infusion_g<T: PkNum>(
    rate: f64,
    dur: f64,
    amt: f64,
    t: T,
    cl: T,
    v1: T,
    q: T,
    v2: T,
) -> T {
    if t.val() < 0.0 || v1.val() <= 0.0 || cl.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    if dur <= 0.0 {
        return two_cpt_iv_bolus_g(amt, t, cl, v1, q, v2);
    }
    let (alpha, beta, k21) = macro_rates_g(cl, v1, q, v2);
    let diff = alpha - beta;
    if diff.val().abs() < 1e-12 || alpha.val().abs() < 1e-12 || beta.val().abs() < 1e-12 {
        return T::from_f64(0.0);
    }
    let r_v1 = T::from_f64(rate) / v1;
    let a_coeff = r_v1 * (alpha - k21) / (diff * alpha);
    let b_coeff = r_v1 * (k21 - beta) / (diff * beta);
    let one = T::from_f64(1.0);
    let dd = T::from_f64(dur);
    if t.val() <= dur {
        a_coeff * (one - (-(alpha * t)).exp()) + b_coeff * (one - (-(beta * t)).exp())
    } else {
        let dt = t - dd;
        a_coeff * (one - (-(alpha * dd)).exp()) * (-(alpha * dt)).exp()
            + b_coeff * (one - (-(beta * dd)).exp()) * (-(beta * dt)).exp()
    }
}

/// 2-cpt oral (first-order absorption), with `ka ≈ α`/`ka ≈ β` L'Hôpital limits.
pub fn two_cpt_oral_g<T: PkNum>(amt: f64, t: T, cl: T, v1: T, q: T, v2: T, ka: T, f_bio: T) -> T {
    if t.val() < 0.0 || v1.val() <= 0.0 || cl.val() <= 0.0 || ka.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    let (alpha, beta, k21) = macro_rates_g(cl, v1, q, v2);
    let diff = alpha - beta;
    // Confluent eigenvalues (α≈β): the spectral form carries 1/(α−β) residue
    // poles, undefined at exact root coalescence. Measure-zero for physical PK
    // params (distinct real eigenvalues); return 0 rather than the confluent
    // (A+B·t)·e^{−αt} limit — a deliberate edge (PR #381, review #1).
    if diff.val().abs() < 1e-12 {
        return T::from_f64(0.0);
    }
    let d = f_bio * T::from_f64(amt) * ka / v1;
    let tt = t;

    // Combined ka→α (or ka→β) L'Hôpital limit: the e^{-ka·t} term shares the
    // 1/(ka-α) pole and is folded in, contributing −(k21-β)/diff²·e^{-αt} (the
    // piece a t-only limit drops). Mirror of `pk::two_compartment::two_cpt_oral_f`.
    let p = if (ka.val() - alpha.val()).abs() < 1e-6 {
        d * (-(alpha * tt)).exp() * ((alpha - k21) / diff * tt - (k21 - beta) / (diff * diff))
    } else {
        d * (k21 - alpha) / ((ka - alpha) * (beta - alpha)) * (-(alpha * tt)).exp()
    };
    let q_val = if (ka.val() - beta.val()).abs() < 1e-6 {
        d * (-(beta * tt)).exp() * ((k21 - beta) / diff * tt - (k21 - alpha) / (diff * diff))
    } else {
        d * (k21 - beta) / ((ka - beta) * (alpha - beta)) * (-(beta * tt)).exp()
    };
    let r = if (ka.val() - alpha.val()).abs() < 1e-6 || (ka.val() - beta.val()).abs() < 1e-6 {
        T::from_f64(0.0)
    } else {
        d * (k21 - ka) / ((alpha - ka) * (beta - ka)) * (-(ka * tt)).exp()
    };
    p + q_val + r
}

/// SS geometric-series factor `1/(1 − e^{−λ·ii})`.
fn ss_coeff_g<T: PkNum>(lambda: T, ii: f64) -> T {
    let denom = T::from_f64(1.0) - (-(lambda * T::from_f64(ii))).exp();
    if denom.val() > 0.0 {
        T::from_f64(1.0) / denom
    } else {
        T::from_f64(0.0)
    }
}

/// 2-cpt IV bolus at steady state.
pub fn two_cpt_iv_bolus_ss_g<T: PkNum>(amt: f64, t: T, ii: f64, cl: T, v1: T, q: T, v2: T) -> T {
    // `q < 0` (e.g. a step or additive IIV driving Q negative) makes a disposition
    // eigenvalue negative, so the SS geometric `1/(1−e^{−λΙΙ})` flips sign and the
    // closed form returns a non-physical (or NaN, via a negative discriminant)
    // concentration. The production SS path returned 0 here; mirror it (PR #381 #3).
    if t.val() < 0.0
        || v1.val() <= 0.0
        || cl.val() <= 0.0
        || v2.val() <= 0.0
        || q.val() < 0.0
        || ii <= 0.0
    {
        return T::from_f64(0.0);
    }
    let (alpha, beta, k21) = macro_rates_g(cl, v1, q, v2);
    let diff = alpha - beta;
    // Confluent eigenvalues (α≈β): the spectral form carries 1/(α−β) residue
    // poles, undefined at exact root coalescence. Measure-zero for physical PK
    // params (distinct real eigenvalues); return 0 rather than the confluent
    // (A+B·t)·e^{−αt} limit — a deliberate edge (PR #381, review #1).
    if diff.val().abs() < 1e-12 {
        return T::from_f64(0.0);
    }
    let amt_v1 = T::from_f64(amt) / v1;
    let a = amt_v1 * (alpha - k21) / diff;
    let b = amt_v1 * (k21 - beta) / diff;
    let tt = t;
    a * (-(alpha * tt)).exp() * ss_coeff_g(alpha, ii)
        + b * (-(beta * tt)).exp() * ss_coeff_g(beta, ii)
}

/// 2-cpt oral at steady state, with `ka ≈ α`/`ka ≈ β` L'Hôpital limits.
pub fn two_cpt_oral_ss_g<T: PkNum>(
    amt: f64,
    t: T,
    ii: f64,
    cl: T,
    v1: T,
    q: T,
    v2: T,
    ka: T,
    f_bio: T,
) -> T {
    if t.val() < 0.0
        || v1.val() <= 0.0
        || cl.val() <= 0.0
        || ka.val() <= 0.0
        || v2.val() <= 0.0
        || q.val() < 0.0
        || ii <= 0.0
    {
        return T::from_f64(0.0);
    }
    let (alpha, beta, k21) = macro_rates_g(cl, v1, q, v2);
    let diff = alpha - beta;
    // Confluent eigenvalues (α≈β): the spectral form carries 1/(α−β) residue
    // poles, undefined at exact root coalescence. Measure-zero for physical PK
    // params (distinct real eigenvalues); return 0 rather than the confluent
    // (A+B·t)·e^{−αt} limit — a deliberate edge (PR #381, review #1).
    if diff.val().abs() < 1e-12 {
        return T::from_f64(0.0);
    }
    let d = f_bio * T::from_f64(amt) * ka / v1;
    let tt = t;

    // L'Hôpital SS sum of (τ+nII)e^{−λ(τ+nII)} = e^{−λτ}[τ/(1−x) + II·x/(1−x)²].
    let lhop = |lambda: T| -> T {
        let x = (-(lambda * T::from_f64(ii))).exp();
        let omx = T::from_f64(1.0) - x;
        if omx.val() <= 0.0 {
            return T::from_f64(0.0);
        }
        (-(lambda * tt)).exp() * (tt / omx + (x * T::from_f64(ii)) / (omx * omx))
    };

    // Combined ka→α (or ka→β) L'Hôpital limit, SS form: add the
    // −(k21-β)/diff²·e^{-ατ}·ss_coeff term the t-only limit drops (see
    // `pk::two_compartment::two_cpt_oral_f_ss`).
    let p = if (ka.val() - alpha.val()).abs() < 1e-6 {
        d * ((alpha - k21) / diff * lhop(alpha)
            - (k21 - beta) / (diff * diff) * (-(alpha * tt)).exp() * ss_coeff_g(alpha, ii))
    } else {
        d * (k21 - alpha) / ((ka - alpha) * (beta - alpha))
            * (-(alpha * tt)).exp()
            * ss_coeff_g(alpha, ii)
    };
    let q_val = if (ka.val() - beta.val()).abs() < 1e-6 {
        d * ((k21 - beta) / diff * lhop(beta)
            - (k21 - alpha) / (diff * diff) * (-(beta * tt)).exp() * ss_coeff_g(beta, ii))
    } else {
        d * (k21 - beta) / ((ka - beta) * (alpha - beta))
            * (-(beta * tt)).exp()
            * ss_coeff_g(beta, ii)
    };
    let r = if (ka.val() - alpha.val()).abs() < 1e-6 || (ka.val() - beta.val()).abs() < 1e-6 {
        T::from_f64(0.0)
    } else {
        d * (k21 - ka) / ((alpha - ka) * (beta - ka)) * (-(ka * tt)).exp() * ss_coeff_g(ka, ii)
    };
    p + q_val + r
}

/// 2-cpt infusion at steady state (interval `ii`), for any `dur` — including
/// overlapping pulses (`dur > ii`). Mirrors production `two_cpt_infusion_ss`.
#[allow(clippy::too_many_arguments)]
pub fn two_cpt_infusion_ss_g<T: PkNum>(
    rate: f64,
    dur: f64,
    amt: f64,
    t: T,
    ii: f64,
    cl: T,
    v1: T,
    q: T,
    v2: T,
) -> T {
    // See `two_cpt_iv_bolus_ss_g` for why SS guards `q < 0` (PR #381 #3).
    if t.val() < 0.0
        || v1.val() <= 0.0
        || cl.val() <= 0.0
        || v2.val() <= 0.0
        || q.val() < 0.0
        || ii <= 0.0
    {
        return T::from_f64(0.0);
    }
    if dur <= 0.0 {
        return two_cpt_iv_bolus_ss_g(amt, t, ii, cl, v1, q, v2);
    }
    let (alpha, beta, k21) = macro_rates_g(cl, v1, q, v2);
    let diff = alpha - beta;
    if diff.val().abs() < 1e-12 || alpha.val().abs() < 1e-12 || beta.val().abs() < 1e-12 {
        return T::from_f64(0.0);
    }
    let r_v1 = T::from_f64(rate) / v1;
    let a_coeff = r_v1 * (alpha - k21) / (diff * alpha);
    let b_coeff = r_v1 * (k21 - beta) / (diff * beta);
    let one = T::from_f64(1.0);
    let dd = T::from_f64(dur);
    if dur > ii {
        // Overlapping infusions: superpose the past pulse train per eigenvalue
        // (mirror of `crate::pk::two_cpt_infusion_ss`). `N` (count of pulses still
        // infusing at phase `t`) is a locally-constant integer, seeded as a value.
        let n_active = (((dur - t.val()) / ii).floor() + 1.0).max(0.0);
        let nii = T::from_f64(n_active * ii);
        let overlap = |c: T, lambda: T| -> T {
            let sc = ss_coeff_g(lambda, ii);
            let a = T::from_f64(n_active)
                - (-(lambda * t)).exp() * (one - (-(lambda * nii)).exp()) * sc;
            let d = (one - (-(lambda * dd)).exp()) * (-(lambda * (t - dd + nii))).exp() * sc;
            c * (a + d)
        };
        return overlap(a_coeff, alpha) + overlap(b_coeff, beta);
    }
    let dt = t - dd;
    // Past pulses (n ≥ 1): always "after-infusion".
    let past_a = a_coeff
        * (one - (-(alpha * dd)).exp())
        * (-(alpha * dt)).exp()
        * (-(alpha * T::from_f64(ii))).exp()
        * ss_coeff_g(alpha, ii);
    let past_b = b_coeff
        * (one - (-(beta * dd)).exp())
        * (-(beta * dt)).exp()
        * (-(beta * T::from_f64(ii))).exp()
        * ss_coeff_g(beta, ii);
    if t.val() <= dur {
        a_coeff * (one - (-(alpha * t)).exp())
            + b_coeff * (one - (-(beta * t)).exp())
            + past_a
            + past_b
    } else {
        a_coeff * (one - (-(alpha * dd)).exp()) * (-(alpha * dt)).exp() * ss_coeff_g(alpha, ii)
            + b_coeff * (one - (-(beta * dd)).exp()) * (-(beta * dt)).exp() * ss_coeff_g(beta, ii)
    }
}

/// Central-compartment concentration contribution of a single dose at elapsed
/// time `t`, dispatching bolus / infusion / oral and their SS variants — the
/// 2-cpt counterpart to [`super::one_cpt::one_cpt_conc_g`].
#[allow(clippy::too_many_arguments)]
pub fn two_cpt_conc_g<T: PkNum>(
    dose: &DoseEvent,
    t: T,
    cl: T,
    v1: T,
    q: T,
    v2: T,
    ka: T,
    f_bio: T,
    oral: bool,
) -> T {
    // `F` rule mirrors production `route_f_scale` (#327): oral-depot bolus bakes
    // `F` in; IV bolus and every infusion are linear in dose → post-multiply.
    let oral_bolus = oral && !dose.is_infusion();
    let raw = if dose.ss && dose.ii > 0.0 {
        if dose.is_infusion() {
            two_cpt_infusion_ss_g(
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
        } else if oral {
            two_cpt_oral_ss_g(dose.amt, t, dose.ii, cl, v1, q, v2, ka, f_bio)
        } else {
            two_cpt_iv_bolus_ss_g(dose.amt, t, dose.ii, cl, v1, q, v2)
        }
    } else if dose.is_infusion() {
        two_cpt_infusion_g(dose.rate, dose.duration, dose.amt, t, cl, v1, q, v2)
    } else if oral {
        two_cpt_oral_g(dose.amt, t, cl, v1, q, v2, ka, f_bio)
    } else {
        two_cpt_iv_bolus_g(dose.amt, t, cl, v1, q, v2)
    };
    if oral_bolus {
        raw
    } else {
        raw * f_bio
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sens::dual2::Dual2;

    /// A negative inter-compartmental clearance `Q` (reachable via a step or
    /// additive IIV) must yield 0 from the SS forms — not a non-physical or NaN
    /// concentration — matching the production SS guard (PR #381 review #3).
    #[test]
    fn ss_forms_return_zero_and_finite_for_negative_q() {
        let (cl, v1, v2, ka) = (1.2f64, 12.0, 30.0, 1.5);
        let q = -0.5f64; // negative inter-compartmental clearance
        let ii = 12.0;
        let bolus = two_cpt_iv_bolus_ss_g::<f64>(100.0, 2.0, ii, cl, v1, q, v2);
        let oral = two_cpt_oral_ss_g::<f64>(100.0, 2.0, ii, cl, v1, q, v2, ka, 1.0);
        let inf = two_cpt_infusion_ss_g::<f64>(50.0, 2.0, 100.0, 2.0, ii, cl, v1, q, v2);
        for (name, v) in [("bolus", bolus), ("oral", oral), ("infusion", inf)] {
            assert!(v.is_finite() && v == 0.0, "{name} SS with q<0 = {v}");
        }
        // Sanity: a physical q>0 still produces a positive SS concentration.
        assert!(two_cpt_iv_bolus_ss_g::<f64>(100.0, 2.0, ii, cl, v1, 2.0, v2) > 0.0);
    }

    /// Coverage + correctness for `two_cpt_conc_g` across every dose kind. Reached
    /// only via the `Dual2` provider (the f64 production walk dispatches through
    /// `single_dose_concentration`, not `conc_g`), so otherwise uncovered. Checks
    /// the `Dual2` value against the f64 value (gradient finite), also exercising
    /// the `F` post-multiply.
    #[test]
    fn conc_g_all_dose_kinds_dual_matches_f64() {
        let (cl, v1, q, v2, ka, fb) = (1.2, 12.0, 2.0, 30.0, 1.5, 0.8);
        let t = 2.0;
        let mk = |rate: f64, ss: bool, ii: f64| DoseEvent::new(0.0, 100.0, 1, rate, ss, ii);
        let cases: [(DoseEvent, bool); 6] = [
            (mk(0.0, false, 0.0), false),
            (mk(50.0, false, 0.0), false),
            (mk(0.0, false, 0.0), true),
            (mk(0.0, true, 12.0), false),
            (mk(0.0, true, 12.0), true),
            (mk(50.0, true, 24.0), false),
        ];
        for (dose, oral) in &cases {
            let v64 = two_cpt_conc_g::<f64>(dose, t, cl, v1, q, v2, ka, fb, *oral);
            let d = two_cpt_conc_g::<Dual2<6>>(
                dose,
                Dual2::constant(t),
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q, 2),
                Dual2::var(v2, 3),
                Dual2::var(ka, 4),
                Dual2::var(fb, 5),
                *oral,
            );
            assert!(
                d.value.is_finite() && (d.value - v64).abs() <= 1e-9 * (1.0 + v64.abs()),
                "value {} vs f64 {v64} for {dose:?}",
                d.value
            );
            assert!(
                d.grad.iter().all(|g| g.is_finite()),
                "non-finite grad for {dose:?}"
            );
        }
    }
}
