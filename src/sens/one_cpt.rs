//! One-compartment analytic sensitivities via `Dual2` (issue #367 clean slate).
//!
//! The closed-form solutions are written **once**, generic over [`PkNum`]:
//! evaluating in `f64` gives the prediction, evaluating in `Dual2<N>` (seeded on
//! the PK parameters) gives `f`, `∂f/∂pk`, `∂²f/∂pk²` exactly, no hand-derivation.
//! The `*_hardcoded` form is kept only as an independent cross-check in tests.

use super::dual2::Dual2;
use super::num::PkNum;
use crate::pk::analytical_absorption::{convolve_1cpt, TransitAbsorption};
use crate::types::DoseEvent;

// ── Generic closed-form single-dose solutions ────────────────────────────────

/// 1-cpt IV bolus: `C(t) = (D/V)·exp(−(CL/V)·t)`. `t` (elapsed time since dose) is
/// generic so the caller can seed it as a dual carrying the lagtime sensitivity
/// (`∂t/∂lagtime = −1`); pass an `f64` for the plain prediction.
pub fn one_cpt_iv_bolus_g<T: PkNum>(amt: f64, t: T, cl: T, v: T) -> T {
    one_cpt_iv_bolus_amt_g(T::from_f64(amt), t, cl, v)
}

/// As [`one_cpt_iv_bolus_g`] but the amount `amt` is itself generic over
/// [`PkNum`], so an analytical initial condition `A₀(θ,η)` (issue #524) threads
/// its parameter sensitivity through the impulse. The single-dose path passes a
/// constant `amt` via [`one_cpt_iv_bolus_g`]; the only formula lives here.
pub fn one_cpt_iv_bolus_amt_g<T: PkNum>(amt: T, t: T, cl: T, v: T) -> T {
    if t.val() < 0.0 || v.val() <= 0.0 || cl.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    let k = cl / v;
    (amt / v) * (-(k * t)).exp()
}

/// 1-cpt oral (first-order absorption), with the `KA ≈ k` L'Hôpital limit.
pub fn one_cpt_oral_g<T: PkNum>(amt: f64, t: T, cl: T, v: T, ka: T, f_bio: T) -> T {
    one_cpt_oral_amt_g(T::from_f64(amt), t, cl, v, ka, f_bio)
}

/// As [`one_cpt_oral_g`] but with a generic amount `amt` (issue #524). The
/// initial-condition path passes `A₀` as a dual with `F = 1`.
pub fn one_cpt_oral_amt_g<T: PkNum>(amt: T, t: T, cl: T, v: T, ka: T, f_bio: T) -> T {
    if t.val() < 0.0 || v.val() <= 0.0 || cl.val() <= 0.0 || ka.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    let k = cl / v;
    let d = f_bio * amt;
    if (ka.val() - k.val()).abs() < 1e-6 {
        (d * ka / v) * t * (-(k * t)).exp()
    } else {
        (d * ka / (v * (ka - k))) * ((-(k * t)).exp() - (-(ka * t)).exp())
    }
}

/// 1-cpt with Savic transit-compartment absorption (`n` transit compartments,
/// mean transit time `mtt`) — the analytic closed form of #386. Like
/// [`one_cpt_oral_g`] this is the single exact `Dual2`-differentiable model; it
/// routes through [`convolve_1cpt`] over a [`TransitAbsorption`] (Gamma) density,
/// so `T = f64` gives the concentration and `T = Dual2<N>` gives exact
/// `∂C/∂{cl,v,n,mtt,f}` (+ 2nd order). With `n = 0` it reduces to first-order oral.
pub fn one_cpt_transit_g<T: PkNum>(amt: f64, t: T, cl: T, v: T, n: T, mtt: T, f_bio: T) -> T {
    one_cpt_transit_amt_g(T::from_f64(amt), t, cl, v, n, mtt, f_bio)
}

/// As [`one_cpt_transit_g`] but with a generic amount `amt` (issue #524 init path).
///
/// Domain: the exponential-tilting closed form converges only for `ke = CL/V` below
/// the transit rate `KTR = (n+1)/mtt` (the absorption-rate-limited regime). Outside
/// it — invalid params, or flip-flop `ke ≥ KTR` — this returns `0.0`, matching the
/// sibling closed forms' invalid-parameter convention (penalising the optimiser back
/// into the valid region rather than letting `convolve_1cpt` emit a NaN; `mgf`'s own
/// `debug_assert` then never fires on this guarded path).
pub fn one_cpt_transit_amt_g<T: PkNum>(amt: T, t: T, cl: T, v: T, n: T, mtt: T, f_bio: T) -> T {
    if t.val() < 0.0 || v.val() <= 0.0 || cl.val() <= 0.0 || n.val() < 0.0 || mtt.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    let ke = cl / v;
    let ktr = (n + T::from_f64(1.0)) / mtt;
    if ke.val() >= ktr.val() {
        return T::from_f64(0.0);
    }
    let abs = TransitAbsorption { n, mtt };
    convolve_1cpt(&abs, t, ke, (f_bio * amt) / v)
}

/// 1-cpt infusion (zero-order input of duration `dur`, rate `rate`).
pub fn one_cpt_infusion_g<T: PkNum>(rate: f64, dur: f64, amt: f64, t: T, cl: T, v: T) -> T {
    if t.val() < 0.0 || v.val() <= 0.0 || cl.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    if dur <= 0.0 {
        return one_cpt_iv_bolus_g(amt, t, cl, v);
    }
    let k = cl / v;
    let r_cl = T::from_f64(rate) / cl;
    let one = T::from_f64(1.0);
    if t.val() <= dur {
        r_cl * (one - (-(k * t)).exp())
    } else {
        r_cl * (one - (-(k * T::from_f64(dur))).exp()) * (-(k * (t - T::from_f64(dur)))).exp()
    }
}

/// 1-cpt IV bolus at steady state (interval `ii`).
pub fn one_cpt_iv_bolus_ss_g<T: PkNum>(amt: f64, t: T, ii: f64, cl: T, v: T) -> T {
    if t.val() < 0.0 || v.val() <= 0.0 || cl.val() <= 0.0 || ii <= 0.0 {
        return T::from_f64(0.0);
    }
    let k = cl / v;
    let denom = T::from_f64(1.0) - (-(k * T::from_f64(ii))).exp();
    if denom.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    (T::from_f64(amt) / v) * (-(k * t)).exp() / denom
}

/// 1-cpt oral at steady state (interval `ii`), with the `KA ≈ k` L'Hôpital limit.
pub fn one_cpt_oral_ss_g<T: PkNum>(amt: f64, t: T, ii: f64, cl: T, v: T, ka: T, f_bio: T) -> T {
    if t.val() < 0.0 || v.val() <= 0.0 || cl.val() <= 0.0 || ka.val() <= 0.0 || ii <= 0.0 {
        return T::from_f64(0.0);
    }
    let k = cl / v;
    let d = f_bio * T::from_f64(amt);
    if (ka.val() - k.val()).abs() < 1e-6 {
        // Σ_{n≥0} (τ+nII)·e^{-k(τ+nII)} = e^{-kτ}·[τ/(1-x) + II·x/(1-x)²], x=e^{-kII}.
        let x = (-(k * T::from_f64(ii))).exp();
        let omx = T::from_f64(1.0) - x;
        if omx.val() <= 0.0 {
            return T::from_f64(0.0);
        }
        let s = t / omx + (x * T::from_f64(ii)) / (omx * omx);
        (d * ka / v) * (-(k * t)).exp() * s
    } else {
        let denom_k = T::from_f64(1.0) - (-(k * T::from_f64(ii))).exp();
        let denom_ka = T::from_f64(1.0) - (-(ka * T::from_f64(ii))).exp();
        if denom_k.val() <= 0.0 || denom_ka.val() <= 0.0 {
            return T::from_f64(0.0);
        }
        (d * ka / (v * (ka - k))) * ((-(k * t)).exp() / denom_k - (-(ka * t)).exp() / denom_ka)
    }
}

/// 1-cpt infusion at steady state (interval `ii`), for any `dur` — including
/// overlapping pulses (`dur > ii`). Mirrors the production
/// [`crate::pk::one_cpt_infusion_ss`]; the `dur > ii` branch superposes the
/// infinite past pulse train (`N` simultaneously-active infusions at phase `t`).
pub fn one_cpt_infusion_ss_g<T: PkNum>(
    rate: f64,
    dur: f64,
    amt: f64,
    t: T,
    ii: f64,
    cl: T,
    v: T,
) -> T {
    if t.val() < 0.0 || v.val() <= 0.0 || cl.val() <= 0.0 || ii <= 0.0 {
        return T::from_f64(0.0);
    }
    if dur <= 0.0 {
        return one_cpt_iv_bolus_ss_g(amt, t, ii, cl, v);
    }
    let k = cl / v;
    let denom = T::from_f64(1.0) - (-(k * T::from_f64(ii))).exp();
    if denom.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    let r_over_cl = T::from_f64(rate) / cl;
    let one_minus_e_kt_inf = T::from_f64(1.0) - (-(k * T::from_f64(dur))).exp();
    if dur > ii {
        // Overlapping infusions: C = (R/CL)·[A + D] with `N` pulses still
        // infusing at phase `t` (see `crate::pk::one_cpt_infusion_ss`). `N`
        // depends on `t` only through its value (a locally-constant integer), so
        // it is seeded as a constant and the dual derivatives flow through the
        // exponentials.
        let n_active = (((dur - t.val()) / ii).floor() + 1.0).max(0.0);
        let nii = T::from_f64(n_active * ii);
        let a = T::from_f64(n_active)
            - (-(k * t)).exp() * (T::from_f64(1.0) - (-(k * nii)).exp()) / denom;
        let d = one_minus_e_kt_inf * (-(k * (t - T::from_f64(dur) + nii))).exp() / denom;
        return r_over_cl * (a + d);
    }
    // Past pulses (n ≥ 1) are always "after-infusion".
    let past = r_over_cl * one_minus_e_kt_inf * (-(k * (t + T::from_f64(ii - dur)))).exp() / denom;
    if t.val() <= dur {
        r_over_cl * (T::from_f64(1.0) - (-(k * t)).exp()) + past
    } else {
        r_over_cl * one_minus_e_kt_inf * (-(k * (t - T::from_f64(dur)))).exp() / denom
    }
}

/// Concentration contribution of a single dose at elapsed time `t` since the
/// dose, dispatching bolus / infusion / oral and their steady-state variants —
/// the generic counterpart to [`crate::pk::one_cpt_predict`]. `oral` selects the
/// absorption route (the PK model is per-subject-static, so this is a flag, not
/// per-dose). `cl`/`v`/`ka`/`f_bio` are the seeded PK params.
///
/// Bioavailability `F` follows the same `route_f_scale` rule as the production
/// predictor ([`crate::pk::predict_concentration`]): the oral-depot bolus form
/// bakes `F` in (so it takes the `1.0` branch), while IV bolus and every infusion
/// (which bypass the depot even on oral models) use an `F`-agnostic closed form
/// and get `F` by post-multiplying the linear-in-dose result — seeded through
/// `f_bio` so `∂C/∂F` is exact (#327).
pub fn one_cpt_conc_g<T: PkNum>(
    dose: &DoseEvent,
    t: T,
    cl: T,
    v: T,
    ka: T,
    f_bio: T,
    oral: bool,
) -> T {
    let oral_bolus = oral && !dose.is_infusion();
    let raw = if dose.ss && dose.ii > 0.0 {
        if dose.is_infusion() {
            one_cpt_infusion_ss_g(dose.rate, dose.duration, dose.amt, t, dose.ii, cl, v)
        } else if oral {
            one_cpt_oral_ss_g(dose.amt, t, dose.ii, cl, v, ka, f_bio)
        } else {
            one_cpt_iv_bolus_ss_g(dose.amt, t, dose.ii, cl, v)
        }
    } else if dose.is_infusion() {
        one_cpt_infusion_g(dose.rate, dose.duration, dose.amt, t, cl, v)
    } else if oral {
        one_cpt_oral_g(dose.amt, t, cl, v, ka, f_bio)
    } else {
        one_cpt_iv_bolus_g(dose.amt, t, cl, v)
    };
    if oral_bolus {
        raw // `F` already baked into the oral-depot form
    } else {
        raw * f_bio // IV bolus / infusion: linear in dose, scale by `F`
    }
}

/// Transit-absorption counterpart to [`one_cpt_conc_g`] for the analytic
/// `one_cpt_transit` model (#386). Transit rejects infusion and SS doses at parse,
/// so only the absorbed-bolus route exists — a thin wrapper over
/// [`one_cpt_transit_amt_g`] with `F` baked into the kernel (no post-multiply, the
/// `oral_bolus` branch of [`one_cpt_conc_g`]). Generic over [`PkNum`] so prediction
/// (`T = f64`) and the `Dual2` sensitivity share one definition.
pub fn one_cpt_transit_conc_g<T: PkNum>(
    dose: &DoseEvent,
    t: T,
    cl: T,
    v: T,
    n: T,
    mtt: T,
    f_bio: T,
) -> T {
    one_cpt_transit_amt_g(T::from_f64(dose.amt), t, cl, v, n, mtt, f_bio)
}

// ── Sensitivity extraction (seed the active PK params as Dual2 variables) ─────

/// `(f, ∂f/∂[CL,V], ∂²f/∂[CL,V]²)` for the IV bolus.
pub fn iv_bolus_sens(amt: f64, t: f64, cl: f64, v: f64) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    let f = one_cpt_iv_bolus_g::<Dual2<2>>(
        amt,
        Dual2::constant(t),
        Dual2::var(cl, 0),
        Dual2::var(v, 1),
    );
    (f.value, f.grad, f.hess)
}

/// `(f, ∂f/∂[CL,V,KA,F], ∂²f/∂[CL,V,KA,F]²)` for oral.
pub fn oral_sens(
    amt: f64,
    t: f64,
    cl: f64,
    v: f64,
    ka: f64,
    f_bio: f64,
) -> (f64, [f64; 4], [[f64; 4]; 4]) {
    let f = one_cpt_oral_g::<Dual2<4>>(
        amt,
        Dual2::constant(t),
        Dual2::var(cl, 0),
        Dual2::var(v, 1),
        Dual2::var(ka, 2),
        Dual2::var(f_bio, 3),
    );
    (f.value, f.grad, f.hess)
}

/// Independent hand-derived IV-bolus derivatives, used only to cross-check the
/// dual path in tests (`k=CL/V`, `f=(D/V)e^{−kt}`).
#[cfg(test)]
fn iv_bolus_hardcoded(amt: f64, t: f64, cl: f64, v: f64) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    if t < 0.0 || v <= 0.0 || cl <= 0.0 {
        return (0.0, [0.0; 2], [[0.0; 2]; 2]);
    }
    let k = cl / v;
    let f = (amt / v) * (-k * t).exp();
    let v2 = v * v;
    let grad = [f * (-t / v), f * (k * t - 1.0) / v];
    let hess = [
        [f * t * t / v2, f * t * (2.0 - k * t) / v2],
        [
            f * t * (2.0 - k * t) / v2,
            f * (k * k * t * t - 4.0 * k * t + 2.0) / v2,
        ],
    ];
    (f, grad, hess)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central FD grad + 4-point Hessian of an `N`-arg value closure.
    fn fd<const N: usize>(p: [f64; N], val: impl Fn([f64; N]) -> f64) -> ([f64; N], [[f64; N]; N]) {
        let h: [f64; N] = std::array::from_fn(|i| 1e-5 * (1.0 + p[i].abs()));
        let bump = |d: [f64; N]| {
            let mut q = p;
            for i in 0..N {
                q[i] += d[i];
            }
            val(q)
        };
        let mut g = [0.0; N];
        let mut hh = [[0.0; N]; N];
        for i in 0..N {
            let mut u = [0.0; N];
            u[i] = h[i];
            let mut dn = [0.0; N];
            dn[i] = -h[i];
            g[i] = (bump(u) - bump(dn)) / (2.0 * h[i]);
        }
        for i in 0..N {
            for j in 0..N {
                let mut pp = [0.0; N];
                pp[i] += h[i];
                pp[j] += h[j];
                let mut pm = [0.0; N];
                pm[i] += h[i];
                pm[j] -= h[j];
                let mut mp = [0.0; N];
                mp[i] -= h[i];
                mp[j] += h[j];
                let mut mm = [0.0; N];
                mm[i] -= h[i];
                mm[j] -= h[j];
                hh[i][j] = (bump(pp) - bump(pm) - bump(mp) + bump(mm)) / (4.0 * h[i] * h[j]);
            }
        }
        (g, hh)
    }

    #[test]
    fn iv_bolus_dual_matches_hardcoded_and_fd() {
        for &(amt, t, cl, v) in &[(100.0, 0.5, 3.0, 30.0), (50.0, 9.0, 5.0, 40.0)] {
            let (fd_v, gd, hd) = iv_bolus_sens(amt, t, cl, v);
            let (fh, gh, hh) = iv_bolus_hardcoded(amt, t, cl, v);
            approx::assert_relative_eq!(fd_v, fh, max_relative = 1e-12);
            for i in 0..2 {
                approx::assert_relative_eq!(gd[i], gh[i], max_relative = 1e-10);
                for j in 0..2 {
                    approx::assert_relative_eq!(hd[i][j], hh[i][j], max_relative = 1e-10);
                }
            }
        }
    }

    #[test]
    fn oral_dual_matches_fd() {
        let (amt, t, cl, v, ka, fb) = (100.0, 2.0, 1.2, 12.0, 0.8, 0.9);
        let (_, gd, hd) = oral_sens(amt, t, cl, v, ka, fb);
        let (gfd, hfd) = fd([cl, v, ka, fb], |p| {
            one_cpt_oral_g::<f64>(amt, t, p[0], p[1], p[2], p[3])
        });
        for i in 0..4 {
            approx::assert_relative_eq!(gd[i], gfd[i], max_relative = 1e-4, epsilon = 1e-9);
            for j in 0..4 {
                approx::assert_relative_eq!(
                    hd[i][j],
                    hfd[i][j],
                    max_relative = 3e-3,
                    epsilon = 1e-8
                );
            }
        }
    }

    /// The analytic `one_cpt_transit_amt_g` wrapper (#386) seeds the *model*
    /// parameters CL, V, N, MTT, F as dual axes and packs them into `convolve_1cpt`
    /// (`ke = CL/V`, `KTR = (N+1)/MTT`, scale `F·amt/V`). #606 FD-checks the
    /// `convolve_1cpt` kernel directly, but in terms of `ke` / `F·Dose/V` — so a
    /// `CL/V → ke` slip or an `N`/`MTT` packing swap in *this* wrapper would not
    /// show up there. This is the exact sensitivity the FOCEI provider seeds for an
    /// estimable transit fit, so check `∂f/∂{CL,V,N,MTT,F}` (and the Hessian)
    /// against a central difference of the f64 form. Guards the headline
    /// "continuous (estimable) N" against a silent gradient regression.
    #[test]
    fn transit_amt_dual_matches_fd() {
        // ke = CL/V = 0.10 well below KTR = (N+1)/MTT = 4/1.5 ≈ 2.67 (valid regime).
        let (amt, t, cl, v, n, mtt, fb) = (100.0, 2.0, 1.2, 12.0, 3.0, 1.5, 0.9);
        let d = one_cpt_transit_amt_g::<Dual2<5>>(
            Dual2::constant(amt),
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
            Dual2::var(n, 2),
            Dual2::var(mtt, 3),
            Dual2::var(fb, 4),
        );
        let (gfd, hfd) = fd([cl, v, n, mtt, fb], |p| {
            one_cpt_transit_amt_g::<f64>(amt, t, p[0], p[1], p[2], p[3], p[4])
        });
        // A real (positive, finite) concentration, so the gradient check is non-trivial.
        assert!(d.value > 0.0 && d.value.is_finite(), "value = {}", d.value);
        // First order is the load-bearing check (these are the exact ∂C/∂{CL,V,N,MTT,F}
        // the estimator seeds); assert it tightly.
        for (g, gf) in d.grad.iter().zip(&gfd) {
            approx::assert_relative_eq!(*g, *gf, max_relative = 2e-4, epsilon = 1e-9);
        }
        // Second order as a sanity check on the wrapper composition. `C` is exactly
        // linear in `F`, so `∂²C/∂F²` is a structural 0 against which the FD reference
        // is pure rounding noise — an absolute floor (not just a relative bound)
        // keeps that term, and any other near-zero entry, from flaking. The exact
        // kernel Hessian is FD-validated in #606's `convolve_1cpt_dual_gradients_match_fd`.
        for (hrow, hfrow) in d.hess.iter().zip(&hfd) {
            for (h, hf) in hrow.iter().zip(hfrow) {
                approx::assert_relative_eq!(*h, *hf, max_relative = 5e-3, epsilon = 1e-4);
            }
        }
    }

    /// Out-of-domain / invalid params return `0.0` (the sibling closed forms'
    /// invalid-parameter convention): bad `n`/`mtt`, and the flip-flop
    /// `ke = CL/V ≥ KTR = (n+1)/mtt`.
    #[test]
    fn one_cpt_transit_amt_guards_return_zero() {
        let z = 0.0_f64;
        // n < 0 → invalid-param guard
        assert_eq!(
            one_cpt_transit_g::<f64>(100.0, 1.0, 1.0, 10.0, -0.5, 1.5, 1.0),
            z
        );
        // mtt = 0 → invalid-param guard
        assert_eq!(
            one_cpt_transit_g::<f64>(100.0, 1.0, 1.0, 10.0, 3.0, 0.0, 1.0),
            z
        );
        // ke = CL/V = 3.0 ≥ KTR = (3+1)/1.5 ≈ 2.67 → flip-flop guard
        assert_eq!(
            one_cpt_transit_g::<f64>(100.0, 1.0, 30.0, 10.0, 3.0, 1.5, 1.0),
            z
        );
    }

    /// Force the full `f+grad+hess` of an IV-bolus sensitivity at dual width `N`
    /// (seed CL@0, V@1; the other N−2 dims stay zero but still cost O(N²) work).
    /// Returns a reduction over every component so nothing is optimised away.
    fn iv_bolus_width<const N: usize>(amt: f64, t: f64, cl: f64, v: f64) -> f64 {
        let f = one_cpt_iv_bolus_g::<Dual2<N>>(
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
        );
        let mut s = f.value;
        for i in 0..N {
            s += f.grad[i];
            for j in 0..N {
                s += f.hess[i][j];
            }
        }
        s
    }

    /// Same for 1-cpt oral (seed CL@0, V@1, KA@2, F@3).
    fn oral_width<const N: usize>(amt: f64, t: f64, cl: f64, v: f64, ka: f64, fb: f64) -> f64 {
        let f = one_cpt_oral_g::<Dual2<N>>(
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
            Dual2::var(ka, 2),
            Dual2::var(fb, 3),
        );
        let mut s = f.value;
        for i in 0..N {
            s += f.grad[i];
            for j in 0..N {
                s += f.hess[i][j];
            }
        }
        s
    }

    /// Reduction over the explicit (Option-B) IV-bolus f+grad+hess, parity with
    /// `iv_bolus_width` (sum value + 2 grad + 4 hess entries).
    fn iv_bolus_explicit_reduce(amt: f64, t: f64, cl: f64, v: f64) -> f64 {
        let (f, g, h) = iv_bolus_hardcoded(amt, t, cl, v);
        f + g[0] + g[1] + h[0][0] + h[0][1] + h[1][0] + h[1][1]
    }

    /// Option B (explicit closed-form derivatives) vs `Dual2<N>` forward-mode, for
    /// the per-observation `f + ∂f/∂pk + ∂²f/∂pk²` kernel. Shows (a) the explicit
    /// path vs the minimal-width dual, and (b) the O(N²) penalty the provider pays
    /// by seeding `Dual2<8>` for every model regardless of active parameter count.
    #[test]
    #[ignore = "bench: run with -- --ignored --nocapture"]
    fn option_b_vs_dual2_widths() {
        use std::time::Instant;
        let n = 20_000_000u64;
        let (amt, cl, v, ka, fb) = (100.0, 1.2, 12.0, 0.8, 0.9);
        let time_it = |label: &str, f: &dyn Fn(f64) -> f64| {
            let t0 = Instant::now();
            let mut acc = 0.0;
            for i in 0..n {
                let t = (i % 24) as f64 * 0.5;
                acc += f(t);
            }
            let ns = t0.elapsed().as_nanos() as f64 / n as f64;
            std::hint::black_box(acc);
            eprintln!("  {label:<34} {ns:6.2} ns/eval");
            ns
        };

        eprintln!("IV bolus (2 active params: CL, V) — f + grad + hess:");
        let exp = time_it("Option B (explicit f64)", &|t| {
            iv_bolus_explicit_reduce(amt, t, cl, v)
        });
        let d2 = time_it("Dual2<2> (minimal width)", &|t| {
            iv_bolus_width::<2>(amt, t, cl, v)
        });
        let d4 = time_it("Dual2<4>", &|t| iv_bolus_width::<4>(amt, t, cl, v));
        let d8 = time_it("Dual2<8> (provider width)", &|t| {
            iv_bolus_width::<8>(amt, t, cl, v)
        });
        eprintln!(
            "  → explicit is {:.1}x faster than Dual2<2>, {:.1}x faster than Dual2<8>",
            d2 / exp,
            d8 / exp
        );
        eprintln!(
            "  → Dual2<8> is {:.1}x slower than Dual2<2> (O(N^2) width penalty)\n",
            d8 / d2
        );

        eprintln!("1-cpt oral (3-4 active params: CL, V, KA[, F]) — f + grad + hess:");
        let o4 = time_it("Dual2<4> (minimal width)", &|t| {
            oral_width::<4>(amt, t, cl, v, ka, fb)
        });
        let o8 = time_it("Dual2<8> (provider width)", &|t| {
            oral_width::<8>(amt, t, cl, v, ka, fb)
        });
        eprintln!(
            "  → Dual2<8> is {:.1}x slower than Dual2<4> for oral (provider over-seeds)\n",
            o8 / o4
        );
    }

    /// Coverage + correctness for the `one_cpt_conc_g` dispatcher across every
    /// dose kind (bolus / infusion / oral and their SS variants, with `F`). This
    /// path is reached only via the `Dual2` provider — the f64 production walk
    /// dispatches through `single_dose_concentration`, not `conc_g` — so it is
    /// otherwise uncovered. Checks the `Dual2` value/grad against the f64 value
    /// and FD of the same f64 dispatcher (also exercises the `F` post-multiply).
    #[test]
    fn conc_g_all_dose_kinds_match_f64_and_fd() {
        let (cl, v, ka, fb) = (1.2, 12.0, 1.5, 0.8);
        let t = 2.0;
        let mk = |rate: f64, ss: bool, ii: f64| DoseEvent::new(0.0, 100.0, 1, rate, ss, ii);
        let cases: [(DoseEvent, bool); 6] = [
            (mk(0.0, false, 0.0), false),  // IV bolus
            (mk(50.0, false, 0.0), false), // IV infusion (dur = amt/rate = 2)
            (mk(0.0, false, 0.0), true),   // oral bolus
            (mk(0.0, true, 12.0), false),  // SS IV bolus
            (mk(0.0, true, 12.0), true),   // SS oral
            (mk(50.0, true, 24.0), false), // SS infusion (dur 2 < II)
        ];
        for (dose, oral) in &cases {
            let v64 = one_cpt_conc_g::<f64>(dose, t, cl, v, ka, fb, *oral);
            let d = one_cpt_conc_g::<Dual2<4>>(
                dose,
                Dual2::constant(t),
                Dual2::var(cl, 0),
                Dual2::var(v, 1),
                Dual2::var(ka, 2),
                Dual2::var(fb, 3),
                *oral,
            );
            assert!(
                (d.value - v64).abs() <= 1e-9 * (1.0 + v64.abs()),
                "value {} vs f64 {v64} for {dose:?}",
                d.value
            );
            let (g, _) = fd([cl, v, ka, fb], |p| {
                one_cpt_conc_g::<f64>(dose, t, p[0], p[1], p[2], p[3], *oral)
            });
            for k in 0..4 {
                assert!(
                    (d.grad[k] - g[k]).abs() <= 1e-4 * (1.0 + g[k].abs()),
                    "grad[{k}] {} vs FD {} for {dose:?}",
                    d.grad[k],
                    g[k]
                );
            }
        }
    }

    /// Overhead of value + gradient + Hessian (`Dual2`) vs the bare `f64` value.
    #[test]
    #[ignore = "bench: run with -- --ignored --nocapture"]
    fn dual2_overhead() {
        use std::time::Instant;
        let n = 5_000_000u64;
        let (amt, cl, v, ka, fb) = (100.0, 1.2, 12.0, 0.8, 0.9);

        let t0 = Instant::now();
        let mut a = 0.0;
        for i in 0..n {
            let t = (i % 24) as f64 * 0.5;
            a += one_cpt_oral_g::<f64>(amt, t, cl, v, ka, fb);
        }
        let f64_ns = t0.elapsed().as_nanos() as f64 / n as f64;

        let t1 = Instant::now();
        let mut b = 0.0;
        for i in 0..n {
            let t = (i % 24) as f64 * 0.5;
            let (f, _, _) = oral_sens(amt, t, cl, v, ka, fb);
            b += f;
        }
        let dual_ns = t1.elapsed().as_nanos() as f64 / n as f64;

        eprintln!(
            "1-cpt oral (4 params): f64 value = {f64_ns:.1} ns; Dual2<4> value+grad+hess = {dual_ns:.1} ns; overhead = {:.1}x",
            dual_ns / f64_ns
        );
        std::hint::black_box((a, b));
    }
}
