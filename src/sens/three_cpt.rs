//! Three-compartment analytic solutions generic over [`PkNum`] — the 3-cpt
//! counterpart to [`super::two_cpt`]. Written once and monomorphised for `f64`
//! (prediction) and [`Dual2`](super::dual2::Dual2) (exact PK-parameter
//! sensitivities). Mirrors [`crate::pk::three_compartment`]; only the central-
//! compartment concentration is needed for the likelihood. The macro-rate cubic
//! starts from the trigonometric solution on the values and is polished in the
//! number type ([`macro_rates_three_cpt_g`]).
// PK closed forms carry many positional params (dose descriptors + 6 PK params);
// the wide signatures are inherent, not a smell.
#![allow(clippy::too_many_arguments)]

use super::num::{bateman, bateman_ss, ss_factor, PkNum};
use super::one_cpt::{modal_sum, mode_infusion, mode_infusion_ss};
use crate::types::DoseEvent;

/// Macro-rate constants `(α, β, γ, k21, k31)` with `α ≥ β ≥ γ ≥ 0`: the roots of the
/// disposition cubic, ordered by construction (no sort), with exact jets of every order.
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

    // Each root is polished in `T` by three Newton steps on the cubic from a start computed on
    // the values (`value_only`): from a jet-free start each step doubles the orders its jets are
    // exact to (1, 3, 7 — past every type's 4), and the value converges with them. So no jet
    // flows through `acos` (unbounded as two roots meet) or a discriminant's square root (which
    // cancels as they do); each root's jets are the implicit function's, `−∂P/∂θ / P'(λ)`.
    let newton = |x0: T| {
        (0..3).fold(x0, |x, _| {
            let f = ((x - s2) * x + s1) * x - s0;
            let df = (T::from_f64(3.0) * x - T::from_f64(2.0) * s2) * x + s1;
            x - f / df
        })
    };
    let (c2, c1, c0) = (s2.value_only(), s1.value_only(), s0.value_only());
    let third = T::from_f64(1.0 / 3.0);
    let h = c2 * third;
    let p = c1 - c2 * c2 * third;
    let q = c1 * c2 * third - c2 * c2 * c2 * T::from_f64(2.0 / 27.0) - c0;
    // Guard p = 0 (a triple root; inactive for valid PK) and the rounding of `arg` past ±1:
    // the start only, which the Newton steps then converge from.
    let p_safe = if p.br_cmp_c(crate::sens::num::BranchCmp::Lt, -1e-30) {
        p
    } else {
        T::from_f64(-1e-30)
    };
    let m = (-(p_safe * third)).sqrt() * T::from_f64(2.0);
    let arg_raw = (q * T::from_f64(3.0)) / (p_safe * m);
    let arg = if arg_raw.br_cmp_c(crate::sens::num::BranchCmp::Gt, 1.0) {
        T::from_f64(1.0)
    } else if arg_raw.br_cmp_c(crate::sens::num::BranchCmp::Lt, -1.0) {
        T::from_f64(-1.0)
    } else {
        arg_raw
    };
    // `α`: φ ∈ [0, π/3] makes `m·cos φ + h` the largest of the three roots.
    let alpha = newton(m * (arg.acos() * third).cos() + h);
    // `β`: the larger root of the deflated quadratic `λ² − (β+γ)·λ + βγ` (`βγ = s0/α`,
    // `β + γ = (s1 − βγ)/α`), a sum of positives; `γ = s0/(αβ)` by Vieta, so the slow
    // terminal root does not cancel.
    let a0 = alpha.value_only();
    let bg0 = c0 / a0;
    let beta = newton(crate::sens::two_cpt::quadratic_roots((c1 - bg0) / a0, bg0).0);
    let gamma = s0 / (alpha * beta);

    (alpha, beta, gamma, k21, k31)
}

/// A three-compartment disposition: the micro parameters the closed forms read directly
/// and the macro rates ([`macro_rates_three_cpt_g`]), computed once for a parameter set and
/// shared by every dose and observation evaluated at it. `Q2`/`Q3` enter only through the
/// rates, so they are not kept.
#[derive(Clone, Copy)]
pub(crate) struct ThreeCpt<T> {
    pub cl: T,
    pub v1: T,
    pub v2: T,
    pub v3: T,
    pub alpha: T,
    pub beta: T,
    pub gamma: T,
    pub k21: T,
    pub k31: T,
}

impl<T: PkNum> ThreeCpt<T> {
    pub(crate) fn new(cl: T, v1: T, q2: T, v2: T, q3: T, v3: T) -> Self {
        let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt_g(cl, v1, q2, v2, q3, v3);
        Self {
            cl,
            v1,
            v2,
            v3,
            alpha,
            beta,
            gamma,
            k21,
            k31,
        }
    }

    /// The disposition modes `(c, λ)` of the central impulse response `(1/V1)·Σ c·e^{−λt}`,
    /// `c_λ = (λ − k21)(λ − k31)/Π_{μ≠λ}(λ − μ)` — all `≥ 0` (the roots interlace `k21`,
    /// `k31`), so no closed form built on them cancels. `None` at a repeated root, which needs
    /// a zero inter-compartmental clearance (a positive one keeps the roots of the
    /// irreducible arrowhead generator distinct).
    pub(crate) fn modes(&self) -> Option<[(T, T); 3]> {
        let ThreeCpt {
            alpha: a,
            beta: b,
            gamma: g,
            k21,
            k31,
            ..
        } = *self;
        use crate::sens::num::BranchCmp::Le;
        if (a - b).br_cmp_c(Le, 0.0) || (b - g).br_cmp_c(Le, 0.0) {
            return None;
        }
        let c = |l: T, m: T, n: T| (l - k21) * (l - k31) / ((l - m) * (l - n));
        Some([(c(a, b, g), a), (c(b, a, g), b), (c(g, a, b), g)])
    }
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
    three_cpt_iv_bolus_d(amt, t, &ThreeCpt::new(cl, v1, q2, v2, q3, v3))
}

/// [`three_cpt_iv_bolus_g`] over a precomputed disposition (its macro rates computed once per
/// parameter set, not once per dose and observation).
#[allow(clippy::too_many_arguments)]
pub(crate) fn three_cpt_iv_bolus_d<T: PkNum>(amt: f64, t: T, d: &ThreeCpt<T>) -> T {
    three_cpt_iv_bolus_amt_d(T::from_f64(amt), t, d)
}

/// As [`three_cpt_iv_bolus_g`] but with a generic amount `amt` (issue #524), so
/// an analytical initial condition `A₀(θ,η)` threads its sensitivity through the
/// central impulse. The single-dose path delegates here with a constant `amt`.
#[allow(clippy::too_many_arguments)]
pub fn three_cpt_iv_bolus_amt_g<T: PkNum>(
    amt: T,
    t: T,
    cl: T,
    v1: T,
    q2: T,
    v2: T,
    q3: T,
    v3: T,
) -> T {
    three_cpt_iv_bolus_amt_d(amt, t, &ThreeCpt::new(cl, v1, q2, v2, q3, v3))
}

/// [`three_cpt_iv_bolus_amt_g`] over a precomputed disposition (its macro rates computed once per
/// parameter set, not once per dose and observation).
#[allow(clippy::too_many_arguments)]
pub(crate) fn three_cpt_iv_bolus_amt_d<T: PkNum>(amt: T, t: T, d: &ThreeCpt<T>) -> T {
    let ThreeCpt { cl, v1, v2, v3, .. } = *d;

    if t.br_cmp_c(crate::sens::num::BranchCmp::Lt, 0.0) {
        return T::from_f64(0.0);
    }
    // The shared #2104 rule: degenerate → 0, `NaN` → `NaN`, ahead of this form's own
    // numerical guards (which would otherwise read a `NaN` disposition as 0).
    if let Some(c) = crate::sens::propagate::dose_override_g(cl, v1) {
        return c;
    }
    if v2.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
        || v3.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
    {
        return T::from_f64(0.0);
    }

    let Some(m) = d.modes() else {
        return T::from_f64(0.0);
    };
    (amt / v1) * modal_sum(&m, |lam| (-(lam * t)).exp())
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
    three_cpt_infusion_d(rate, dur, amt, t, &ThreeCpt::new(cl, v1, q2, v2, q3, v3))
}

/// [`three_cpt_infusion_g`] over a precomputed disposition (its macro rates computed once per
/// parameter set, not once per dose and observation).
#[allow(clippy::too_many_arguments)]
pub(crate) fn three_cpt_infusion_d<T: PkNum>(
    rate: f64,
    dur: f64,
    amt: f64,
    t: T,
    d: &ThreeCpt<T>,
) -> T {
    let ThreeCpt { cl, v1, v2, v3, .. } = *d;

    if t.br_cmp_c(crate::sens::num::BranchCmp::Lt, 0.0) {
        return T::from_f64(0.0);
    }
    // The shared #2104 rule: degenerate → 0, `NaN` → `NaN`, ahead of this form's own
    // numerical guards (which would otherwise read a `NaN` disposition as 0).
    if let Some(c) = crate::sens::propagate::dose_override_g(cl, v1) {
        return c;
    }
    if v2.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
        || v3.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
    {
        return T::from_f64(0.0);
    }
    if dur <= 0.0 {
        return three_cpt_iv_bolus_d(amt, t, d);
    }

    let Some(m) = d.modes() else {
        return T::from_f64(0.0);
    };
    let (rv, dd) = (T::from_f64(rate) / v1, T::from_f64(dur));
    modal_sum(&m, |lam| mode_infusion(lam, rv, dd, t))
}

/// 3-cpt oral (first-order absorption): `(F·D·KA/V1)·Σ c·B(λ, KA, t)` over the disposition
/// modes, the Bateman function one smooth function through `KA = λ` ([`bateman`]).
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
    three_cpt_oral_d(amt, t, &ThreeCpt::new(cl, v1, q2, v2, q3, v3), ka, f_bio)
}

/// [`three_cpt_oral_g`] over a precomputed disposition (its macro rates computed once per
/// parameter set, not once per dose and observation).
#[allow(clippy::too_many_arguments)]
pub(crate) fn three_cpt_oral_d<T: PkNum>(amt: f64, t: T, d: &ThreeCpt<T>, ka: T, f_bio: T) -> T {
    three_cpt_oral_amt_d(T::from_f64(amt), t, d, ka, f_bio)
}

/// As [`three_cpt_oral_g`] but with a generic amount `amt` (issue #524); the
/// initial-condition path passes a pre-loaded depot `A₀` as a dual with `F = 1`.
#[allow(clippy::too_many_arguments)]
pub fn three_cpt_oral_amt_g<T: PkNum>(
    amt: T,
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
    three_cpt_oral_amt_d(amt, t, &ThreeCpt::new(cl, v1, q2, v2, q3, v3), ka, f_bio)
}

/// [`three_cpt_oral_amt_g`] over a precomputed disposition (its macro rates computed once per
/// parameter set, not once per dose and observation).
#[allow(clippy::too_many_arguments)]
pub(crate) fn three_cpt_oral_amt_d<T: PkNum>(amt: T, t: T, d: &ThreeCpt<T>, ka: T, f_bio: T) -> T {
    let ThreeCpt { cl, v1, v2, v3, .. } = *d;

    if t.br_cmp_c(crate::sens::num::BranchCmp::Lt, 0.0) {
        return T::from_f64(0.0);
    }
    // The shared #2104 rule: degenerate → 0, `NaN` → `NaN`, ahead of this form's own
    // numerical guards (which would otherwise read a `NaN` disposition as 0).
    if let Some(c) = crate::sens::propagate::dose_override_g(cl, v1) {
        return c;
    }
    if v2.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
        || v3.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
        || ka.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
    {
        return T::from_f64(0.0);
    }

    let Some(m) = d.modes() else {
        return T::from_f64(0.0);
    };
    (f_bio * amt * ka / v1) * modal_sum(&m, |lam| bateman(lam, ka, t))
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
    three_cpt_iv_bolus_ss_d(amt, t, ii, &ThreeCpt::new(cl, v1, q2, v2, q3, v3))
}

/// [`three_cpt_iv_bolus_ss_g`] over a precomputed disposition (its macro rates computed once per
/// parameter set, not once per dose and observation).
#[allow(clippy::too_many_arguments)]
pub(crate) fn three_cpt_iv_bolus_ss_d<T: PkNum>(amt: f64, t: T, ii: f64, d: &ThreeCpt<T>) -> T {
    let ThreeCpt { cl, v1, v2, v3, .. } = *d;

    if t.br_cmp_c(crate::sens::num::BranchCmp::Lt, 0.0) {
        return T::from_f64(0.0);
    }
    // The shared #2104 rule: degenerate → 0, `NaN` → `NaN`, ahead of this form's own
    // numerical guards (which would otherwise read a `NaN` disposition as 0).
    if let Some(c) = crate::sens::propagate::dose_override_g(cl, v1) {
        return c;
    }
    if v2.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
        || v3.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
        || ii <= 0.0
    {
        return T::from_f64(0.0);
    }

    let Some(m) = d.modes() else {
        return T::from_f64(0.0);
    };
    (T::from_f64(amt) / v1) * modal_sum(&m, |lam| (-(lam * t)).exp() * ss_factor(lam, ii))
}

/// 3-cpt oral at steady state: `(F·D·KA/V1)·Σ c·B_ss(λ, KA, t)`, smooth through `KA = λ`
/// ([`bateman_ss`]).
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
    three_cpt_oral_ss_d(
        amt,
        t,
        ii,
        &ThreeCpt::new(cl, v1, q2, v2, q3, v3),
        ka,
        f_bio,
    )
}

/// [`three_cpt_oral_ss_g`] over a precomputed disposition (its macro rates computed once per
/// parameter set, not once per dose and observation).
#[allow(clippy::too_many_arguments)]
pub(crate) fn three_cpt_oral_ss_d<T: PkNum>(
    amt: f64,
    t: T,
    ii: f64,
    d: &ThreeCpt<T>,
    ka: T,
    f_bio: T,
) -> T {
    let ThreeCpt { cl, v1, v2, v3, .. } = *d;

    if t.br_cmp_c(crate::sens::num::BranchCmp::Lt, 0.0) {
        return T::from_f64(0.0);
    }
    // The shared #2104 rule: degenerate → 0, `NaN` → `NaN`, ahead of this form's own
    // numerical guards (which would otherwise read a `NaN` disposition as 0).
    if let Some(c) = crate::sens::propagate::dose_override_g(cl, v1) {
        return c;
    }
    if v2.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
        || v3.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
        || ka.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
        || ii <= 0.0
    {
        return T::from_f64(0.0);
    }

    let Some(m) = d.modes() else {
        return T::from_f64(0.0);
    };
    (f_bio * T::from_f64(amt) * ka / v1) * modal_sum(&m, |lam| bateman_ss(lam, ka, t, ii))
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
    three_cpt_infusion_ss_d(
        rate,
        dur,
        amt,
        t,
        ii,
        &ThreeCpt::new(cl, v1, q2, v2, q3, v3),
    )
}

/// [`three_cpt_infusion_ss_g`] over a precomputed disposition (its macro rates computed once per
/// parameter set, not once per dose and observation).
#[allow(clippy::too_many_arguments)]
pub(crate) fn three_cpt_infusion_ss_d<T: PkNum>(
    rate: f64,
    dur: f64,
    amt: f64,
    t: T,
    ii: f64,
    d: &ThreeCpt<T>,
) -> T {
    let ThreeCpt { cl, v1, v2, v3, .. } = *d;

    if t.br_cmp_c(crate::sens::num::BranchCmp::Lt, 0.0) {
        return T::from_f64(0.0);
    }
    // The shared #2104 rule: degenerate → 0, `NaN` → `NaN`, ahead of this form's own
    // numerical guards (which would otherwise read a `NaN` disposition as 0).
    if let Some(c) = crate::sens::propagate::dose_override_g(cl, v1) {
        return c;
    }
    if v2.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
        || v3.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0)
        || ii <= 0.0
    {
        return T::from_f64(0.0);
    }
    if dur <= 0.0 {
        return three_cpt_iv_bolus_ss_d(amt, t, ii, d);
    }

    let Some(m) = d.modes() else {
        return T::from_f64(0.0);
    };
    let rv = T::from_f64(rate) / v1;
    modal_sum(&m, |lam| mode_infusion_ss(lam, rv, dur, ii, t))
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
    three_cpt_conc_d(
        dose,
        t,
        &ThreeCpt::new(cl, v1, q2, v2, q3, v3),
        ka,
        f_bio,
        oral,
    )
}

/// [`three_cpt_conc_g`] over a precomputed disposition (its macro rates computed once per
/// parameter set, not once per dose and observation).
#[allow(clippy::too_many_arguments)]
pub(crate) fn three_cpt_conc_d<T: PkNum>(
    dose: &DoseEvent,
    t: T,
    d: &ThreeCpt<T>,
    ka: T,
    f_bio: T,
    oral: bool,
) -> T {
    // The shared #2104 rule decides an arrived dose before `F` scales it, as
    // `pk::single_dose_concentration` does: a degenerate `CL`/`V` drops the dose (0 even
    // when `F` is `NaN` or infinite, which the post-multiply below would turn `NaN`), a
    // `NaN` one serves `NaN`. Before arrival the leaf kernels read 0.
    if t.br_cmp_c(crate::sens::num::BranchCmp::Ge, 0.0) {
        if let Some(c) = crate::sens::propagate::dose_override_g(d.cl, d.v1) {
            return c;
        }
    }
    // `F` rule mirrors production `route_f_scale` (#327): oral-depot bolus bakes
    // `F` in; IV bolus and every infusion are linear in dose → post-multiply.
    let oral_bolus = oral && !dose.is_infusion();
    let raw = if dose.ss && dose.ii > 0.0 {
        if dose.is_infusion() {
            three_cpt_infusion_ss_d(dose.rate, dose.duration, dose.amt, t, dose.ii, d)
        } else if oral {
            three_cpt_oral_ss_d(dose.amt, t, dose.ii, d, ka, f_bio)
        } else {
            three_cpt_iv_bolus_ss_d(dose.amt, t, dose.ii, d)
        }
    } else if dose.is_infusion() {
        three_cpt_infusion_d(dose.rate, dose.duration, dose.amt, t, d)
    } else if oral {
        three_cpt_oral_d(dose.amt, t, d, ka, f_bio)
    } else {
        three_cpt_iv_bolus_d(dose.amt, t, d)
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
