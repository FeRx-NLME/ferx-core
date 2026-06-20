//! Three-compartment analytic solutions generic over [`PkNum`] — the 3-cpt
//! counterpart to [`super::two_cpt`]. Written once and monomorphised for `f64`
//! (prediction) and [`Dual2`](super::dual2::Dual2) (exact PK-parameter
//! sensitivities). Mirrors [`crate::pk::three_compartment`]; only the central-
//! compartment concentration is needed for the likelihood. The macro-rate cubic
//! is solved by the same trigonometric (Vieta) method, which is why the dual
//! number needs `cos`/`acos` (see [`super::num::PkNum`]).
// PK closed forms carry many positional params (dose descriptors + 6 PK params);
// the wide signatures are inherent, not a smell.
#![allow(clippy::too_many_arguments)]

use super::num::PkNum;
use crate::types::DoseEvent;

/// Macro-rate constants `(α, β, γ, k21, k31)` with `α > β > γ > 0`, via the
/// trigonometric solution of the disposition cubic. Branches on `.val()` only,
/// so the dual derivatives flow through the selected roots unchanged.
#[allow(clippy::many_single_char_names)]
pub(crate) fn macro_rates_three_cpt_g<T: PkNum>(
    cl: T,
    v1: T,
    q2: T,
    v2: T,
    q3: T,
    v3: T,
) -> (T, T, T, T, T) {
    let k10 = cl / v1;
    let k12 = q2 / v1;
    let k21 = q2 / v2;
    let k13 = q3 / v1;
    let k31 = q3 / v3;

    // Symmetric functions of the roots (Vieta's formulas).
    let s2 = k10 + k12 + k13 + k21 + k31;
    let s1 = k10 * k21 + k10 * k31 + k21 * k31 + k12 * k31 + k13 * k21;
    let s0 = k10 * k21 * k31;

    // Depress the cubic: lambda = x + s2/3.
    let third = T::from_f64(1.0 / 3.0);
    let h = s2 * third;
    let p = s1 - s2 * s2 * third;
    let q = s1 * s2 * third - s2 * s2 * s2 * T::from_f64(2.0 / 27.0) - s0;

    // Guard p = 0 (matches production `p.min(-1e-30)`); inactive for valid PK.
    let p_safe = if p.val() < -1e-30 {
        p
    } else {
        T::from_f64(-1e-30)
    };
    let m = (-(p_safe * third)).sqrt() * T::from_f64(2.0);
    // arg = 3q / (p_safe·m), clamped to [-1, 1] (identity in the interior).
    let arg_raw = (q * T::from_f64(3.0)) / (p_safe * m);
    let arg = if arg_raw.val() > 1.0 {
        T::from_f64(1.0)
    } else if arg_raw.val() < -1.0 {
        T::from_f64(-1.0)
    } else {
        arg_raw
    };
    let phi = arg.acos() * third;

    let pi_2_3 = 2.0 * std::f64::consts::FRAC_PI_3;
    let lambda0 = m * phi.cos() + h;
    let lambda1 = m * (phi - T::from_f64(pi_2_3)).cos() + h;
    let lambda2 = m * (phi - T::from_f64(2.0 * pi_2_3)).cos() + h;

    // Sort by value: alpha > beta > gamma. The concentration is symmetric under
    // root permutation, so selecting the dual by `.val()` is exact at distinct
    // roots (the sort is locally constant there).
    let (l0v, l1v, l2v) = (lambda0.val(), lambda1.val(), lambda2.val());
    let alpha = if l0v >= l1v && l0v >= l2v {
        lambda0
    } else if l1v >= l2v {
        lambda1
    } else {
        lambda2
    };
    let gamma = if l0v <= l1v && l0v <= l2v {
        lambda0
    } else if l1v <= l2v {
        lambda1
    } else {
        lambda2
    };
    let beta = s2 - alpha - gamma;

    (alpha, beta, gamma, k21, k31)
}

/// 3-cpt IV bolus: `C = A·e^{−αt} + B·e^{−βt} + G·e^{−γt}`.
#[allow(clippy::too_many_arguments)]
pub fn three_cpt_iv_bolus_g<T: PkNum>(
    amt: f64,
    t: T,
    cl: T,
    v1: T,
    q2: T,
    v2: T,
    q3: T,
    v3: T,
) -> T {
    if t.val() < 0.0 || v1.val() <= 0.0 || v2.val() <= 0.0 || v3.val() <= 0.0 || cl.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt_g(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    // Confluent eigenvalues (any of α≈β, α≈γ, β≈γ): the spectral form carries
    // 1/(λ_i−λ_j) residue poles, undefined at exact root coalescence. Measure-zero
    // for physical PK params (distinct real eigenvalues); return 0 rather than the
    // confluent limit form — a deliberate edge (PR #381, review #1).
    if ab.val().abs() < 1e-12 || ag.val().abs() < 1e-12 || bg.val().abs() < 1e-12 {
        return T::from_f64(0.0);
    }
    let d = T::from_f64(amt) / v1;
    let a = d * (alpha - k21) * (alpha - k31) / (ab * ag);
    let b = d * (beta - k21) * (beta - k31) / (-(ab) * bg);
    let g = d * (gamma - k21) * (gamma - k31) / (ag * bg);
    let tt = t;
    a * (-(alpha * tt)).exp() + b * (-(beta * tt)).exp() + g * (-(gamma * tt)).exp()
}

/// 3-cpt constant-rate IV infusion (duration `dur`, rate `rate`).
#[allow(clippy::too_many_arguments)]
pub fn three_cpt_infusion_g<T: PkNum>(
    rate: f64,
    dur: f64,
    amt: f64,
    t: T,
    cl: T,
    v1: T,
    q2: T,
    v2: T,
    q3: T,
    v3: T,
) -> T {
    if t.val() < 0.0 || v1.val() <= 0.0 || v2.val() <= 0.0 || v3.val() <= 0.0 || cl.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    if dur <= 0.0 {
        return three_cpt_iv_bolus_g(amt, t, cl, v1, q2, v2, q3, v3);
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt_g(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    if ab.val().abs() < 1e-12
        || ag.val().abs() < 1e-12
        || bg.val().abs() < 1e-12
        || alpha.val().abs() < 1e-12
        || beta.val().abs() < 1e-12
        || gamma.val().abs() < 1e-12
    {
        return T::from_f64(0.0);
    }
    let rv = T::from_f64(rate) / v1;
    let a_coeff = rv * (alpha - k21) * (alpha - k31) / (ab * ag * alpha);
    let b_coeff = rv * (beta - k21) * (beta - k31) / (-(ab) * bg * beta);
    let g_coeff = rv * (gamma - k21) * (gamma - k31) / (ag * bg * gamma);
    let one = T::from_f64(1.0);
    let dd = T::from_f64(dur);
    if t.val() <= dur {
        let tt = t;
        a_coeff * (one - (-(alpha * tt)).exp())
            + b_coeff * (one - (-(beta * tt)).exp())
            + g_coeff * (one - (-(gamma * tt)).exp())
    } else {
        let dt = t - dd;
        a_coeff * (one - (-(alpha * dd)).exp()) * (-(alpha * dt)).exp()
            + b_coeff * (one - (-(beta * dd)).exp()) * (-(beta * dt)).exp()
            + g_coeff * (one - (-(gamma * dd)).exp()) * (-(gamma * dt)).exp()
    }
}

/// 3-cpt oral (first-order absorption), with per-eigenvalue `ka ≈ λ` L'Hôpital
/// limits in the Bateman function.
#[allow(clippy::too_many_arguments)]
pub fn three_cpt_oral_g<T: PkNum>(
    amt: f64,
    t: T,
    cl: T,
    v1: T,
    q2: T,
    v2: T,
    q3: T,
    v3: T,
    ka: T,
    f_bio: T,
) -> T {
    if t.val() < 0.0
        || v1.val() <= 0.0
        || v2.val() <= 0.0
        || v3.val() <= 0.0
        || cl.val() <= 0.0
        || ka.val() <= 0.0
    {
        return T::from_f64(0.0);
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt_g(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    // Confluent eigenvalues (any of α≈β, α≈γ, β≈γ): the spectral form carries
    // 1/(λ_i−λ_j) residue poles, undefined at exact root coalescence. Measure-zero
    // for physical PK params (distinct real eigenvalues); return 0 rather than the
    // confluent limit form — a deliberate edge (PR #381, review #1).
    if ab.val().abs() < 1e-12 || ag.val().abs() < 1e-12 || bg.val().abs() < 1e-12 {
        return T::from_f64(0.0);
    }
    let coeff = f_bio * T::from_f64(amt) * ka / v1;
    let a = (alpha - k21) * (alpha - k31) / (ab * ag);
    let b = (beta - k21) * (beta - k31) / (-(ab) * bg);
    let c = (gamma - k21) * (gamma - k31) / (ag * bg);
    let tt = t;

    // Bateman per eigenvalue λ with L'Hôpital limit when ka ≈ λ.
    let bateman = |lambda: T| -> T {
        if (ka.val() - lambda.val()).abs() < 1e-6 {
            tt * (-(lambda * tt)).exp()
        } else {
            ((-(lambda * tt)).exp() - (-(ka * tt)).exp()) / (ka - lambda)
        }
    };

    coeff * (a * bateman(alpha) + b * bateman(beta) + c * bateman(gamma))
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

/// 3-cpt IV bolus at steady state.
#[allow(clippy::too_many_arguments)]
pub fn three_cpt_iv_bolus_ss_g<T: PkNum>(
    amt: f64,
    t: T,
    ii: f64,
    cl: T,
    v1: T,
    q2: T,
    v2: T,
    q3: T,
    v3: T,
) -> T {
    if t.val() < 0.0
        || v1.val() <= 0.0
        || v2.val() <= 0.0
        || v3.val() <= 0.0
        || cl.val() <= 0.0
        || ii <= 0.0
    {
        return T::from_f64(0.0);
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt_g(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    // Confluent eigenvalues (any of α≈β, α≈γ, β≈γ): the spectral form carries
    // 1/(λ_i−λ_j) residue poles, undefined at exact root coalescence. Measure-zero
    // for physical PK params (distinct real eigenvalues); return 0 rather than the
    // confluent limit form — a deliberate edge (PR #381, review #1).
    if ab.val().abs() < 1e-12 || ag.val().abs() < 1e-12 || bg.val().abs() < 1e-12 {
        return T::from_f64(0.0);
    }
    let d = T::from_f64(amt) / v1;
    let a = d * (alpha - k21) * (alpha - k31) / (ab * ag);
    let b = d * (beta - k21) * (beta - k31) / (-(ab) * bg);
    let g = d * (gamma - k21) * (gamma - k31) / (ag * bg);
    let tt = t;
    a * (-(alpha * tt)).exp() * ss_coeff_g(alpha, ii)
        + b * (-(beta * tt)).exp() * ss_coeff_g(beta, ii)
        + g * (-(gamma * tt)).exp() * ss_coeff_g(gamma, ii)
}

/// 3-cpt oral at steady state, with per-eigenvalue `ka ≈ λ` L'Hôpital limits.
#[allow(clippy::too_many_arguments)]
pub fn three_cpt_oral_ss_g<T: PkNum>(
    amt: f64,
    t: T,
    ii: f64,
    cl: T,
    v1: T,
    q2: T,
    v2: T,
    q3: T,
    v3: T,
    ka: T,
    f_bio: T,
) -> T {
    if t.val() < 0.0
        || v1.val() <= 0.0
        || v2.val() <= 0.0
        || v3.val() <= 0.0
        || cl.val() <= 0.0
        || ka.val() <= 0.0
        || ii <= 0.0
    {
        return T::from_f64(0.0);
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt_g(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    // Confluent eigenvalues (any of α≈β, α≈γ, β≈γ): the spectral form carries
    // 1/(λ_i−λ_j) residue poles, undefined at exact root coalescence. Measure-zero
    // for physical PK params (distinct real eigenvalues); return 0 rather than the
    // confluent limit form — a deliberate edge (PR #381, review #1).
    if ab.val().abs() < 1e-12 || ag.val().abs() < 1e-12 || bg.val().abs() < 1e-12 {
        return T::from_f64(0.0);
    }
    let coeff = f_bio * T::from_f64(amt) * ka / v1;
    let a = (alpha - k21) * (alpha - k31) / (ab * ag);
    let b = (beta - k21) * (beta - k31) / (-(ab) * bg);
    let c = (gamma - k21) * (gamma - k31) / (ag * bg);
    let tt = t;

    // SS Bateman per eigenvalue λ, with L'Hôpital limit when ka ≈ λ:
    //   Σ_n (τ + n·II)·e^{−λ(τ+nII)} = e^{−λτ}·[τ/(1−x) + II·x/(1−x)²], x=e^{−λII}.
    let bateman_ss = |lambda: T| -> T {
        if (ka.val() - lambda.val()).abs() < 1e-6 {
            let x = (-(lambda * T::from_f64(ii))).exp();
            let omx = T::from_f64(1.0) - x;
            if omx.val() <= 0.0 {
                return T::from_f64(0.0);
            }
            (-(lambda * tt)).exp() * (tt / omx + (x * T::from_f64(ii)) / (omx * omx))
        } else {
            ((-(lambda * tt)).exp() * ss_coeff_g(lambda, ii)
                - (-(ka * tt)).exp() * ss_coeff_g(ka, ii))
                / (ka - lambda)
        }
    };

    coeff * (a * bateman_ss(alpha) + b * bateman_ss(beta) + c * bateman_ss(gamma))
}

/// 3-cpt infusion at steady state (interval `ii`), for any `dur` — including
/// overlapping pulses (`dur > ii`). Mirrors production `three_cpt_infusion_ss`.
#[allow(clippy::too_many_arguments)]
pub fn three_cpt_infusion_ss_g<T: PkNum>(
    rate: f64,
    dur: f64,
    amt: f64,
    t: T,
    ii: f64,
    cl: T,
    v1: T,
    q2: T,
    v2: T,
    q3: T,
    v3: T,
) -> T {
    if t.val() < 0.0
        || v1.val() <= 0.0
        || v2.val() <= 0.0
        || v3.val() <= 0.0
        || cl.val() <= 0.0
        || ii <= 0.0
    {
        return T::from_f64(0.0);
    }
    if dur <= 0.0 {
        return three_cpt_iv_bolus_ss_g(amt, t, ii, cl, v1, q2, v2, q3, v3);
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt_g(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    if ab.val().abs() < 1e-12
        || ag.val().abs() < 1e-12
        || bg.val().abs() < 1e-12
        || alpha.val().abs() < 1e-12
        || beta.val().abs() < 1e-12
        || gamma.val().abs() < 1e-12
    {
        return T::from_f64(0.0);
    }
    let rv = T::from_f64(rate) / v1;
    let a_coeff = rv * (alpha - k21) * (alpha - k31) / (ab * ag * alpha);
    let b_coeff = rv * (beta - k21) * (beta - k31) / (-(ab) * bg * beta);
    let g_coeff = rv * (gamma - k21) * (gamma - k31) / (ag * bg * gamma);
    let one = T::from_f64(1.0);
    let dd = T::from_f64(dur);

    if dur > ii {
        // Overlapping infusions: superpose the past pulse train per eigenvalue
        // (mirror of `crate::pk::three_cpt_infusion_ss`). `N` = count of pulses
        // still infusing at phase `t`, a locally-constant integer seeded by value.
        let n_active = (((dur - t.val()) / ii).floor() + 1.0).max(0.0);
        let nii = T::from_f64(n_active * ii);
        let overlap = |c: T, lambda: T| -> T {
            let sc = ss_coeff_g(lambda, ii);
            let a = T::from_f64(n_active)
                - (-(lambda * t)).exp() * (one - (-(lambda * nii)).exp()) * sc;
            let d = (one - (-(lambda * dd)).exp()) * (-(lambda * (t - dd + nii))).exp() * sc;
            c * (a + d)
        };
        return overlap(a_coeff, alpha) + overlap(b_coeff, beta) + overlap(g_coeff, gamma);
    }

    let dt = t - dd;
    // Past pulses (n ≥ 1): always "after-infusion" since τ + n·II ≥ II ≥ dur.
    let past = |coeff: T, lambda: T| -> T {
        coeff
            * (one - (-(lambda * dd)).exp())
            * (-(lambda * dt)).exp()
            * (-(lambda * T::from_f64(ii))).exp()
            * ss_coeff_g(lambda, ii)
    };

    if t.val() <= dur {
        let tt = t;
        a_coeff * (one - (-(alpha * tt)).exp())
            + b_coeff * (one - (-(beta * tt)).exp())
            + g_coeff * (one - (-(gamma * tt)).exp())
            + past(a_coeff, alpha)
            + past(b_coeff, beta)
            + past(g_coeff, gamma)
    } else {
        a_coeff * (one - (-(alpha * dd)).exp()) * (-(alpha * dt)).exp() * ss_coeff_g(alpha, ii)
            + b_coeff * (one - (-(beta * dd)).exp()) * (-(beta * dt)).exp() * ss_coeff_g(beta, ii)
            + g_coeff
                * (one - (-(gamma * dd)).exp())
                * (-(gamma * dt)).exp()
                * ss_coeff_g(gamma, ii)
    }
}

/// Central-compartment concentration contribution of a single dose at elapsed
/// time `t`, dispatching bolus / infusion / oral and their SS variants — the
/// 3-cpt counterpart to [`super::two_cpt::two_cpt_conc_g`].
#[allow(clippy::too_many_arguments)]
pub fn three_cpt_conc_g<T: PkNum>(
    dose: &DoseEvent,
    t: T,
    cl: T,
    v1: T,
    q2: T,
    v2: T,
    q3: T,
    v3: T,
    ka: T,
    f_bio: T,
    oral: bool,
) -> T {
    // `F` rule mirrors production `route_f_scale` (#327): oral-depot bolus bakes
    // `F` in; IV bolus and every infusion are linear in dose → post-multiply.
    let oral_bolus = oral && !dose.is_infusion();
    let raw = if dose.ss && dose.ii > 0.0 {
        if dose.is_infusion() {
            three_cpt_infusion_ss_g(
                dose.rate,
                dose.duration,
                dose.amt,
                t,
                dose.ii,
                cl,
                v1,
                q2,
                v2,
                q3,
                v3,
            )
        } else if oral {
            three_cpt_oral_ss_g(dose.amt, t, dose.ii, cl, v1, q2, v2, q3, v3, ka, f_bio)
        } else {
            three_cpt_iv_bolus_ss_g(dose.amt, t, dose.ii, cl, v1, q2, v2, q3, v3)
        }
    } else if dose.is_infusion() {
        three_cpt_infusion_g(
            dose.rate,
            dose.duration,
            dose.amt,
            t,
            cl,
            v1,
            q2,
            v2,
            q3,
            v3,
        )
    } else if oral {
        three_cpt_oral_g(dose.amt, t, cl, v1, q2, v2, q3, v3, ka, f_bio)
    } else {
        three_cpt_iv_bolus_g(dose.amt, t, cl, v1, q2, v2, q3, v3)
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

    /// Coverage + correctness for `three_cpt_conc_g` across every dose kind.
    /// Reached only via the `Dual2` provider (the f64 production walk dispatches
    /// through `single_dose_concentration`, not `conc_g`), so otherwise uncovered.
    /// Checks the `Dual2` value against the f64 value (gradient finite), also
    /// exercising the `F` post-multiply.
    #[test]
    fn conc_g_all_dose_kinds_dual_matches_f64() {
        let (cl, v1, q2, v2, q3, v3, ka, fb) = (1.2, 12.0, 2.0, 30.0, 1.0, 50.0, 1.5, 0.8);
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
            let v64 = three_cpt_conc_g::<f64>(dose, t, cl, v1, q2, v2, q3, v3, ka, fb, *oral);
            let d = three_cpt_conc_g::<Dual2<8>>(
                dose,
                Dual2::constant(t),
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q2, 2),
                Dual2::var(v2, 3),
                Dual2::var(q3, 4),
                Dual2::var(v3, 5),
                Dual2::var(ka, 6),
                Dual2::var(fb, 7),
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
