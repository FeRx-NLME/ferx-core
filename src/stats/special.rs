//! Special functions: erf, normal CDF, log normal CDF (M3 BLOQ likelihood),
//! ln Γ (Lanczos), and the regularized lower incomplete gamma `P(a, x)` — the
//! last two for the transit-compartment absorption model.
//!
//! The differentiable ones are built from polynomial/rational approximations
//! (Abramowitz & Stegun 7.1.26 for erf) and elementary operations only —
//! `+ − * /`, `.exp()`, `.ln()`, and `ln Γ` — with no `f64::max`/`min` branch
//! ambiguity, so a generic `PkNum`/`Dual2` instantiation differentiates them
//! cleanly. (`normal_inv_cdf` is `f64`-only and off the gradient path.)

use crate::sens::num::PkNum;

/// 1 / sqrt(2)
const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// 1 / sqrt(2*pi)
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
/// Smallest probability retained before taking `ln`. Prevents `-inf` contamination
/// of the likelihood when a BLOQ observation lies many SDs above the prediction.
const MIN_PROB: f64 = 1e-300;
/// ½·ln(2π) — the constant term of the Lanczos `ln_gamma` formula.
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_74;
/// Lanczos `g` parameter (g = 7) shared by [`ln_gamma`] and its analytic
/// derivatives [`digamma`] / [`trigamma`], so the value and its 1st/2nd
/// derivatives are computed from the *same* approximation and can't drift apart.
const LANCZOS_G: f64 = 7.0;
/// Lanczos coefficients for g = 7 (n = 9 terms); see [`LANCZOS_G`].
const LANCZOS_COEF: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
];

/// Abramowitz & Stegun 7.1.26 — max error ~1.5e-7 over the whole real line.
/// Entirely polynomial in t = 1/(1 + p*|x|) and exp(-x²), so cleanly differentiable.
pub fn erf(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = if x < 0.0 { -x } else { x };
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

/// Complementary error function. Uses `erfc(x) = 1 - erf(x)` directly; for very
/// large positive x this loses precision but the `log_normal_cdf` path below
/// uses the asymptotic form directly, so the precision loss never matters.
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// Standard normal CDF: Φ(z) = 0.5 * (1 + erf(z / √2)).
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z * INV_SQRT_2))
}

/// Numerically stable log Φ(z).
///
/// For z > -5 we use `ln(max(Φ(z), MIN_PROB))` directly. For very negative z the
/// naive form underflows to `ln(0) = -inf`, so we switch to the asymptotic
/// expansion of the Mills ratio:
///
///   log Φ(z) ≈ log φ(z) + log(-1/z) + log(1 - 1/z² + 3/z⁴ - 15/z⁶ + …)
///
/// where φ(z) = exp(-z²/2) / √(2π). We truncate after the 1/z⁶ term, which is
/// accurate to ~1e-12 for z < -5 (the branch threshold).
pub fn log_normal_cdf(z: f64) -> f64 {
    if z > -5.0 {
        let p = normal_cdf(z);
        if p < MIN_PROB {
            MIN_PROB.ln()
        } else {
            p.ln()
        }
    } else {
        let log_phi = -0.5 * z * z - (1.0_f64 / INV_SQRT_2PI).ln();
        let inv_z2 = 1.0 / (z * z);
        let series = 1.0 - inv_z2 + 3.0 * inv_z2 * inv_z2 - 15.0 * inv_z2 * inv_z2 * inv_z2;
        // z < 0, so -z > 0 and ln(-z) is well-defined; series > 0 for z < -5.
        log_phi - (-z).ln() + series.ln()
    }
}

/// Inverse Mills ratio `h = φ(z)/Φ(z)`, evaluated through logs (via
/// [`log_normal_cdf`]) so it stays finite in the far tail where `Φ(z) → 0`.
/// Single source for the M3-censored kernels (the closed-form and ODE inner
/// df-coefficients, the residual-η column, and the cross-curvature terms).
#[inline]
pub fn inv_mills(z: f64) -> f64 {
    let ln_phi = -0.5 * z * z - 0.5 * std::f64::consts::TAU.ln();
    (ln_phi - log_normal_cdf(z)).exp()
}

/// Shared M3-censored kernel at a censored row with limit `y`, prediction `f`,
/// residual variance `v`, `dv_df = ∂v/∂f`, and censoring sign `cens` (NONMEM
/// convention: `cens > 0` = below-LLOQ / lower tail, `cens < 0` = above-ULOQ /
/// upper tail). Returns the **signed** `(h, z, m)` where, with `σ = sign(cens)`
/// (`+1` for left, `−1` for right):
/// `z = σ·(y − f)/√v` is the tail argument the objective scores
/// (`−logΦ(z)`, matching [`m3_logcdf`](crate::stats::likelihood::m3_logcdf)),
/// `h = φ(z)/Φ(z)` ([`inv_mills`]), and the signed
/// `m = σ·[1/√v + (y − f)·dv_df / (2·v^{3/2})]`.
///
/// Signing `z` and `m` together keeps every downstream censored derivative
/// tail-correct with **no** per-caller branch: the data-term f-coefficient
/// `∂(−logΦ)/∂f = h·m`; the `iiv_on_ruv` residual-η column
/// `∂(−logΦ)/∂η_ruv = h·z`; and the censored × residual-η cross coefficients via
/// `C = h·(z² + h·z − 1)` (`C·z`, `C·m`). (The second-order f-curvature `g2` in
/// [`m3_censored_outer`] also signs its `∂m/∂f` term by the same `σ`.) For the
/// lower tail (`cens ≥ 0`) `σ = +1` and
/// these reduce to the historical unsigned forms.
#[inline]
pub fn m3_censored_kernel(y: f64, f: f64, v: f64, dv_df: f64, cens: i8) -> (f64, f64, f64) {
    let w = v.sqrt();
    let sgn = if cens < 0 { -1.0 } else { 1.0 };
    let z = sgn * (y - f) / w;
    let h = inv_mills(z);
    let m = sgn * (1.0 / w + (y - f) * dv_df / (2.0 * v * w));
    (h, z, m)
}

/// All M3-censored-row coefficients from a **single** [`m3_censored_kernel`] eval:
/// `(g1, g2, cz, cm)` = `(∂L/∂f, ∂²L/∂f², C·z, C·m)` for `L = −logΦ(z)`,
/// `z = σ·(y−f)/√v` (`σ = sign(cens)`), `(h, z, m)` the signed kernel,
/// `C = h(z²+h·z−1)`:
/// ```text
///   g1 = h·m
///   g2 = h(h+z)·m² + h·∂m/∂f,   ∂m/∂f = σ·{[−2d + (y−f)d2]/(2w³) − 3(y−f)d²/(4w⁵)}
///   (cz, cm) = (C·z, C·m)       (residual-eta cross coefficients, `iiv_on_ruv`)
/// ```
/// `d = ∂v/∂f`, `d2 = ∂²v/∂f²`. `z` and `m` carry the tail sign from the kernel;
/// `∂m/∂f` is signed here by the same `σ` so the curvature matches the upper tail
/// for right-censored (`cens < 0`) rows. One kernel (one `erfc`/`exp` pair) per
/// censored row. Single-sourced here so the FOCEI marginal
/// ([`gaussian_foce_accum`](crate::stats::likelihood)) and the analytic outer
/// gradient ([`sens_outer_gradient`](crate::estimation::sens_outer_gradient))
/// cannot drift.
#[inline]
pub fn m3_censored_outer(
    y: f64,
    f: f64,
    v: f64,
    d: f64,
    d2: f64,
    cens: i8,
) -> (f64, f64, f64, f64) {
    let (h, z, m) = m3_censored_kernel(y, f, v, d, cens);
    let sgn = if cens < 0 { -1.0 } else { 1.0 };
    let w = v.sqrt();
    let w3 = v * w; // w³
    let w5 = v * v * w; // w⁵
    let g1 = h * m;
    let dm_df = sgn * ((-2.0 * d + (y - f) * d2) / (2.0 * w3) - 3.0 * (y - f) * d * d / (4.0 * w5));
    let g2 = h * (h + z) * m * m + h * dm_df;
    let c = h * (z * z + h * z - 1.0);
    (g1, g2, c * z, c * m)
}

/// Inverse standard normal CDF (quantile / probit function): the `z` such that
/// `Φ(z) = p`, for `p ∈ (0, 1)`. Peter Acklam's rational approximation, with a
/// maximum relative error of ~1.15e-9 over the open interval. Returns `-∞` at
/// `p ≤ 0` and `+∞` at `p ≥ 1`.
///
/// (A Halley refinement step is intentionally omitted: it would have to evaluate
/// the forward CDF, and our [`normal_cdf`] carries the A&S 7.1.26 error of
/// ~1.5e-7, so refining against it degrades rather than improves the result.)
///
/// Used by the simulation-based NPDE/NPD diagnostics ([`crate::stats::npde`]) to
/// inverse-normal-transform empirical CDF probabilities. `f64`-only — not on any
/// gradient path.
pub fn normal_inv_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Acklam coefficients.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    // Break-points between the central rational region and the two tails.
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        // Lower tail.
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        // Central region.
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper tail.
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Natural log of the Gamma function, ln Γ(x), via the Lanczos approximation
/// (g = 7, n = 9), accurate to ~1e-13 (relative) for x > 0.
///
/// Added for the transit-compartment absorption model (Savic et al. 2007): its
/// gamma-density input rate needs `ln Γ(n + 1)` for a *continuous* number of
/// transit compartments `n`, where `n!` is undefined. Bare Stirling errs ~8% at
/// n = 1 — enough to bias the absorption peak — so Lanczos is used instead.
///
/// Cleanly differentiable: only `+`, `-`, `*`, `/`, `.ln()`, and (on the
/// reflection branch) `.sin()` — no `f64::max`/`min` branch ambiguity. The
/// reflection branch (x < 0.5) is never exercised by the transit path
/// (n ≥ 0 ⇒ argument ≥ 1) but keeps the function correct over the whole
/// domain x > 0.
pub fn ln_gamma(x: f64) -> f64 {
    // Reflection for x < 0.5: ln Γ(x) = ln(π / sin(πx)) − ln Γ(1 − x).
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let y = x - 1.0;
    let mut a = LANCZOS_COEF[0];
    for (i, &c) in LANCZOS_COEF.iter().enumerate().skip(1) {
        a += c / (y + i as f64);
    }
    let t = y + LANCZOS_G + 0.5;
    HALF_LN_2PI + (y + 0.5) * t.ln() - t + a.ln()
}

/// Digamma ψ(x) = d/dx ln Γ(x) — the exact analytic derivative of [`ln_gamma`]'s
/// Lanczos form (same [`LANCZOS_COEF`] / [`LANCZOS_G`]), so a finite difference of
/// `ln_gamma` and `digamma` agree to ~1e-12 (an independent ψ approximation would
/// not). The first-order `Dual2` rule for `ln Γ` on the transit absorption
/// sensitivity path (#430). Reflection mirrors `ln_gamma`:
/// ψ(x) = ψ(1 − x) − π·cot(πx) for x < 0.5.
pub fn digamma(x: f64) -> f64 {
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return digamma(1.0 - x) - pi / (pi * x).tan();
    }

    let y = x - 1.0;
    let mut a = LANCZOS_COEF[0];
    let mut da = 0.0;
    for (i, &c) in LANCZOS_COEF.iter().enumerate().skip(1) {
        let d = y + i as f64;
        a += c / d;
        da -= c / (d * d);
    }
    let t = y + LANCZOS_G + 0.5;
    t.ln() + (y + 0.5) / t - 1.0 + da / a
}

/// Trigamma ψ′(x) = d²/dx² ln Γ(x) — the exact second derivative of [`ln_gamma`]'s
/// Lanczos form; the second-order `Dual2` rule for `ln Γ` (#430). Reflection:
/// ψ′(x) = π²/sin²(πx) − ψ′(1 − x) for x < 0.5.
pub fn trigamma(x: f64) -> f64 {
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        let s = (pi * x).sin();
        return pi * pi / (s * s) - trigamma(1.0 - x);
    }

    let y = x - 1.0;
    let mut a = LANCZOS_COEF[0];
    let mut da = 0.0;
    let mut dda = 0.0;
    for (i, &c) in LANCZOS_COEF.iter().enumerate().skip(1) {
        let d = y + i as f64;
        a += c / d;
        da -= c / (d * d);
        dda += 2.0 * c / (d * d * d);
    }
    let t = y + LANCZOS_G + 0.5;
    let a_ratio = da / a;
    2.0 / t - (y + 0.5) / (t * t) + dda / a - a_ratio * a_ratio
}

/// Regularized lower incomplete gamma `P(a, x) = γ(a, x) / Γ(a)` — the CDF of a
/// Gamma(shape = `a`, rate = 1) distribution at `x`, defined for `a > 0`,
/// `x ≥ 0` (returns `0` for `x ≤ 0`).
///
/// **Boundary `x ≤ 0`** is a flat clamp: value `0` with a **zero** gradient. That is
/// exact for `a > 1` (the Gamma pdf `∂P/∂x → 0`) and a deliberate clamp for `a ≤ 1`
/// (where the true `∂P/∂x|_{x→0⁺}` is `1` at `a = 1` and `+∞` for `a < 1`, neither
/// representable). The transit closed form only reaches `a = n + 1 = 1` (single
/// transit) at `t = 0`, where the concentration is `0` regardless, so this clamp is
/// never on a live sensitivity path there.
///
/// Added for the **Phase 3** analytical absorption closed forms (#386): the
/// exponential-tilting convolution of a Savic transit (Gamma) input with a linear
/// 1-/2-cpt disposition is `P(n + 1, (KTR − k)·t)` (see `plans/absorption-models.md`
/// §Phase 3), so this is the special function the `convolve_*` closed forms evaluate.
///
/// **Generic over [`PkNum`] on purpose.** Unlike [`ln_gamma`] — whose parameter
/// derivative has a closed form ([`digamma`] / [`trigamma`]) — the incomplete
/// gamma's `∂P/∂a` is *not* elementary. So rather than hand-derive a dual rule, the
/// series / continued-fraction iteration is written **once** over `T: PkNum` and the
/// dual numbers carry the exact `∂P/∂a`, `∂P/∂x` (and 2nd order) through the very
/// same arithmetic — the `sens/` `*_g<T>` convention, with no second copy to drift.
/// Only the branch and convergence *decisions* read `T::val()` (the value part); the
/// body is `+ − * / .exp() .ln()` and [`PkNum::ln_gamma`], all of which the dual
/// types already differentiate. `T = f64` yields the plain value; `T = Dual2<N>`
/// yields the value plus exact 1st/2nd-order sensitivities.
///
/// Algorithm (Numerical Recipes §6.2; Abramowitz & Stegun 6.5): the power series for
/// `x < a + 1`, else Lentz's continued fraction for the complementary upper tail
/// `Q(a, x)` with `P = 1 − Q` — each form converges fastest on its own side of the
/// `x = a + 1` switch. Transit fits sit mostly in the continued-fraction regime
/// (`x = (KTR − k)·t` grows with the dosing interval while `a = n + 1` stays small).
pub fn regularized_gamma_p<T: PkNum>(a: T, x: T) -> T {
    if x.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    if x.val() < a.val() + 1.0 {
        gamma_p_series(a, x)
    } else {
        T::from_f64(1.0) - gamma_q_cf(a, x)
    }
}

/// Shared prefactor `x^a · e^{−x} / Γ(a) = exp(a·ln x − x − ln Γ(a))` of both the
/// series and the continued fraction. Caller guarantees `x > 0`, `a > 0`.
fn gamma_pref<T: PkNum>(a: T, x: T) -> T {
    (a * x.ln() - x - a.ln_gamma()).exp()
}

/// Power series for `P(a, x)`, used when `x < a + 1`:
/// `P = pref · Σ_{n≥0} xⁿ / [a(a+1)…(a+n)]`, with term₀ = 1/a and
/// termₙ = termₙ₋₁ · x/(a+n). Converges in `O(x)` terms in this regime.
fn gamma_p_series<T: PkNum>(a: T, x: T) -> T {
    const EPS: f64 = 1e-15;
    const MAX_ITER: usize = 300;
    let one = T::from_f64(1.0);
    let mut ap = a;
    let mut del = one / a;
    let mut sum = del;
    for _ in 0..MAX_ITER {
        ap = ap + one;
        del = del * x / ap;
        sum = sum + del;
        // Convergence is judged on the value part; the dual jets of `sum` are
        // resolved at least as tightly (the per-term derivatives decay with the
        // term), and the gradient-agreement tests confirm it.
        if del.val().abs() < sum.val().abs() * EPS {
            break;
        }
    }
    sum * gamma_pref(a, x)
}

/// Lentz's continued fraction for the upper incomplete gamma `Q(a, x)`, used when
/// `x ≥ a + 1`; the caller returns `P = 1 − Q`. `aₙ = −n·(n − a)`,
/// `bₙ = x + 1 − a + 2n`.
fn gamma_q_cf<T: PkNum>(a: T, x: T) -> T {
    const EPS: f64 = 1e-15;
    const FPMIN: f64 = 1e-300;
    const MAX_ITER: usize = 300;
    /// Consecutive converged steps required before breaking. Numerical Recipes
    /// breaks the f64 CF on a single `|del−1| < EPS` step, but **differentiating**
    /// the truncated CF needs more care: near the switch `x ≈ a + 1` the CF *value*
    /// converges several iterations before its parameter-derivative jets (`∂/∂a`,
    /// `∂²/∂a²`) do, and the single-step test can even fire at iteration 1 for
    /// integer `a` — there `aₙ = −n·(n − a) = 0` makes `del ≡ 1` while the value is
    /// exact only by coincidence and the jet is far from converged. Requiring the
    /// step to stay converged for a sustained streak runs the extra iterations the
    /// jets need (empirically ≤ ~95 total at the worst near-seam point; the FD
    /// gradient floor ~3e-9 is reached by a streak of 15, so 20 carries margin).
    const SETTLE: usize = 20;
    let one = T::from_f64(1.0);
    let two = T::from_f64(2.0);
    let mut b = x + one - a;
    let mut c = T::from_f64(1.0 / FPMIN);
    let mut d = one / b;
    let mut h = d;
    let mut settled = 0usize;
    for i in 1..=MAX_ITER {
        let ai = T::from_f64(i as f64);
        let an = -ai * (ai - a); // aₙ = −n·(n − a)
        b = b + two;
        d = an * d + b;
        // Guard a vanishing denominator (Lentz). This fires with probability ~0 for
        // valid `a, x` (here `d, c` are O(1)); the constant rescue zeros the local
        // jet, but the flat region is never reached on the tested paths.
        if d.val().abs() < FPMIN {
            d = T::from_f64(FPMIN);
        }
        c = b + an / c;
        if c.val().abs() < FPMIN {
            c = T::from_f64(FPMIN);
        }
        d = one / d;
        let del = d * c;
        h = h * del;
        if (del.val() - 1.0).abs() < EPS {
            settled += 1;
            if settled >= SETTLE {
                break;
            }
        } else {
            settled = 0;
        }
    }
    h * gamma_pref(a, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn erf_zero() {
        // A&S 7.1.26 is a max-error-1.5e-7 rational approximation; erf(0) is
        // close to but not exactly zero.
        assert!(erf(0.0).abs() < 2e-7);
    }

    #[test]
    fn erf_one() {
        // erf(1) = 0.8427007929... — A&S 7.1.26 matches to ~1.5e-7.
        let v = erf(1.0);
        assert!((v - 0.842_700_792_949_715).abs() < 2e-7, "got {}", v);
    }

    #[test]
    fn erf_symmetry() {
        for &x in &[0.1, 0.5, 1.0, 2.0, 3.5] {
            assert_relative_eq!(erf(-x), -erf(x), epsilon = 1e-12);
        }
    }

    #[test]
    fn erf_bounds() {
        assert!(erf(-10.0) > -1.0 - 1e-6);
        assert!(erf(-10.0) < -1.0 + 1e-6);
        assert!(erf(10.0) < 1.0 + 1e-6);
        assert!(erf(10.0) > 1.0 - 1e-6);
    }

    #[test]
    fn normal_cdf_zero() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 2e-7);
    }

    #[test]
    fn normal_cdf_standard_values() {
        // Φ(1.96) ≈ 0.975 (classic) — allow A&S 1.5e-7 error.
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-5);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-5);
    }

    #[test]
    fn normal_inv_cdf_known_quantiles() {
        // Standard quantiles — Acklam's rational approximation, ~1.15e-9.
        assert_relative_eq!(normal_inv_cdf(0.975), 1.959_963_98, epsilon = 1e-6);
        assert_relative_eq!(normal_inv_cdf(0.025), -1.959_963_98, epsilon = 1e-6);
        assert_relative_eq!(normal_inv_cdf(0.5), 0.0, epsilon = 1e-9);
        assert_relative_eq!(normal_inv_cdf(0.9), 1.281_551_57, epsilon = 1e-6);
        assert_relative_eq!(normal_inv_cdf(0.1), -1.281_551_57, epsilon = 1e-6);
    }

    #[test]
    fn normal_inv_cdf_inverts_normal_cdf() {
        // Round-trip Φ⁻¹(Φ(z)) ≈ z across the central region and into the tails.
        // The tolerance is set by normal_cdf's A&S error (~1.5e-7), amplified by
        // 1/φ(z) in the tails — not by normal_inv_cdf, which is good to ~1e-9.
        for &z in &[-3.5, -2.0, -0.7, 0.0, 0.4, 1.5, 3.0] {
            let round = normal_inv_cdf(normal_cdf(z));
            assert_relative_eq!(round, z, epsilon = 1e-4);
        }
    }

    #[test]
    fn normal_inv_cdf_edge_cases() {
        assert_eq!(normal_inv_cdf(0.0), f64::NEG_INFINITY);
        assert_eq!(normal_inv_cdf(1.0), f64::INFINITY);
        assert_eq!(normal_inv_cdf(-0.1), f64::NEG_INFINITY);
        assert_eq!(normal_inv_cdf(1.1), f64::INFINITY);
    }

    #[test]
    fn log_normal_cdf_matches_direct_for_moderate_z() {
        // For z > -5 the two branches should agree closely.
        for &z in &[-4.99, -3.0, -1.0, 0.0, 1.0, 3.0] {
            let direct = normal_cdf(z).ln();
            let stable = log_normal_cdf(z);
            assert_relative_eq!(direct, stable, epsilon = 1e-4);
        }
    }

    #[test]
    fn log_normal_cdf_stable_at_extreme_negative() {
        // At z = -20 the direct CDF underflows to zero, but the asymptotic form
        // yields approximately -203.92 (see Mills-ratio expansion).
        let v = log_normal_cdf(-20.0);
        assert!(v.is_finite(), "log_normal_cdf(-20) must be finite");
        assert!(
            (v - (-203.917)).abs() < 0.01,
            "expected ≈ -203.917, got {}",
            v
        );
    }

    #[test]
    fn log_normal_cdf_is_monotone_increasing() {
        let mut prev = log_normal_cdf(-30.0);
        for z in [-25.0, -20.0, -10.0, -5.5, -5.01, -4.99, -2.0, 0.0, 2.0].iter() {
            let v = log_normal_cdf(*z);
            assert!(
                v >= prev - 1e-6,
                "non-monotone at z={}: {} < {}",
                z,
                v,
                prev
            );
            prev = v;
        }
    }

    #[test]
    fn ln_gamma_integer_factorials() {
        // ln Γ(n+1) = ln(n!). Includes the near-zero cases (0! = 1! = 1).
        let cases = [
            (1.0, 0.0),                    // 0! = 1
            (2.0, 0.0),                    // 1! = 1
            (3.0, std::f64::consts::LN_2), // 2! = 2
            (5.0, 24.0_f64.ln()),          // 4! = 24
            (6.0, 120.0_f64.ln()),         // 5! = 120
            (11.0, 3_628_800.0_f64.ln()),  // 10! = 3628800
        ];
        for (x, want) in cases {
            assert_relative_eq!(ln_gamma(x), want, epsilon = 1e-9, max_relative = 1e-10);
        }
    }

    #[test]
    fn ln_gamma_half_integers_closed_form() {
        // ln Γ(1/2) = ln √π (main branch, x = 0.5 exactly).
        assert_relative_eq!(
            ln_gamma(0.5),
            0.5 * std::f64::consts::PI.ln(),
            epsilon = 1e-12
        );
        // ln Γ(3/2) = ln(√π / 2).
        assert_relative_eq!(
            ln_gamma(1.5),
            (std::f64::consts::PI.sqrt() / 2.0).ln(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn ln_gamma_reflection_branch_small_x() {
        // x < 0.5 exercises the reflection formula. ln Γ(1/4) = 1.2880225246980776.
        assert_relative_eq!(ln_gamma(0.25), 1.288_022_524_698_077_6, epsilon = 1e-9);
    }

    #[test]
    fn ln_gamma_recurrence() {
        // Functional equation: ln Γ(x+1) = ln Γ(x) + ln x (spans both branches).
        for &x in &[0.3, 0.7, 1.0, 2.5, 4.2, 9.0] {
            assert_relative_eq!(
                ln_gamma(x + 1.0),
                ln_gamma(x) + x.ln(),
                epsilon = 1e-9,
                max_relative = 1e-10
            );
        }
    }

    #[test]
    fn ln_gamma_large_argument_no_overflow() {
        // Γ(100) overflows f64, but ln Γ(100) = 359.1342053695754 is finite.
        let v = ln_gamma(100.0);
        assert!(v.is_finite());
        assert_relative_eq!(v, 359.134_205_369_575_4, max_relative = 1e-11);
    }

    #[test]
    fn ln_gamma_legendre_duplication() {
        // Legendre duplication formula — an independent identity that needs no
        // external reference table and exercises both branches (z = 0.3 takes
        // the reflection path):
        //   ln Γ(z) + ln Γ(z+½) = (1−2z)·ln2 + ½·ln π + ln Γ(2z).
        let ln2 = std::f64::consts::LN_2;
        let half_ln_pi = 0.5 * std::f64::consts::PI.ln();
        for &z in &[0.3, 0.7, 1.3, 2.0, 3.5, 6.1] {
            let lhs = ln_gamma(z) + ln_gamma(z + 0.5);
            let rhs = (1.0 - 2.0 * z) * ln2 + half_ln_pi + ln_gamma(2.0 * z);
            assert_relative_eq!(lhs, rhs, epsilon = 1e-9, max_relative = 1e-10);
        }
    }

    #[test]
    fn digamma_known_values() {
        // ψ(1) = −γ, ψ(2) = 1 − γ, ψ(½) = −γ − 2 ln 2.
        let gamma = 0.577_215_664_901_532_9; // Euler–Mascheroni
        let ln2 = std::f64::consts::LN_2;
        assert_relative_eq!(digamma(1.0), -gamma, max_relative = 1e-10);
        assert_relative_eq!(digamma(2.0), 1.0 - gamma, max_relative = 1e-10);
        assert_relative_eq!(digamma(0.5), -gamma - 2.0 * ln2, max_relative = 1e-10);
    }

    #[test]
    fn trigamma_known_values() {
        // ψ′(1) = π²/6, ψ′(2) = π²/6 − 1, ψ′(½) = π²/2.
        let pi = std::f64::consts::PI;
        assert_relative_eq!(trigamma(1.0), pi * pi / 6.0, max_relative = 1e-9);
        assert_relative_eq!(trigamma(2.0), pi * pi / 6.0 - 1.0, max_relative = 1e-9);
        assert_relative_eq!(trigamma(0.5), pi * pi / 2.0, max_relative = 1e-9);
    }

    #[test]
    fn digamma_trigamma_recurrence() {
        // ψ(x+1) = ψ(x) + 1/x ; ψ′(x+1) = ψ′(x) − 1/x².
        for &x in &[0.7, 1.3, 3.0, 6.5] {
            assert_relative_eq!(digamma(x + 1.0), digamma(x) + 1.0 / x, max_relative = 1e-11);
            assert_relative_eq!(
                trigamma(x + 1.0),
                trigamma(x) - 1.0 / (x * x),
                max_relative = 1e-10
            );
        }
    }

    /// digamma/trigamma must be the finite-difference derivatives of `ln_gamma`,
    /// including the reflection branch (x < 0.5) where `ln_gamma` flips to the sine
    /// form — the branch the transit path never hits but correctness still demands.
    #[test]
    fn digamma_trigamma_match_fd_of_ln_gamma_incl_reflection() {
        for &x in &[0.2, 0.35, 0.8, 1.0, 2.5, 5.0, 12.0, 15.0] {
            let h1 = 1e-6;
            let fd1 = (ln_gamma(x + h1) - ln_gamma(x - h1)) / (2.0 * h1);
            assert_relative_eq!(digamma(x), fd1, max_relative = 1e-5, epsilon = 1e-8);
            let h2 = 1e-4;
            let fd2 = (digamma(x + h2) - digamma(x - h2)) / (2.0 * h2);
            assert_relative_eq!(trigamma(x), fd2, max_relative = 1e-5, epsilon = 1e-8);
        }
    }

    /// digamma/trigamma must be continuous across the `x = 0.5` branch seam
    /// (reflection below, Lanczos at/above) — a wrong reflection constant or sign
    /// surfaces as a jump here, which neither side's anchor alone catches. The two
    /// sides differ only by `O(ψ′)·2ε` / `O(ψ″)·2ε`; a real discontinuity is O(1).
    #[test]
    fn digamma_trigamma_continuous_across_reflection_seam() {
        let eps = 1e-7;
        assert_relative_eq!(digamma(0.5 - eps), digamma(0.5 + eps), epsilon = 1e-5);
        assert_relative_eq!(trigamma(0.5 - eps), trigamma(0.5 + eps), epsilon = 1e-4);
    }

    /// `P(a, x)` against two independent closed forms, spanning both algorithm
    /// branches (`x < a+1` series, `x ≥ a+1` continued fraction):
    /// `P(1, x) = 1 − e^{−x}` (exponential CDF) and `P(½, x) = erf(√x)`.
    #[test]
    fn regularized_gamma_p_matches_closed_forms() {
        for &x in &[0.05_f64, 0.5, 1.0, 1.9, 2.0, 3.0, 8.0, 20.0] {
            let want = 1.0 - (-x).exp();
            assert_relative_eq!(
                regularized_gamma_p::<f64>(1.0, x),
                want,
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
        // P(1/2, x) = erf(√x); tolerance set by erf's A&S ~1.5e-7 error.
        for &x in &[0.1, 0.7, 1.5, 4.0] {
            assert_relative_eq!(
                regularized_gamma_p::<f64>(0.5, x),
                erf(x.sqrt()),
                epsilon = 1e-6,
                max_relative = 1e-6
            );
        }
    }

    /// Boundary behaviour: `P(a, 0) = 0`, `x ≤ 0 ⇒ 0`, and `P(a, x) → 1` as `x → ∞`.
    #[test]
    fn regularized_gamma_p_boundary_values() {
        for &a in &[0.5, 1.0, 3.0, 7.5] {
            assert_eq!(regularized_gamma_p::<f64>(a, 0.0), 0.0);
            assert_eq!(regularized_gamma_p::<f64>(a, -1.0), 0.0);
            let tail = regularized_gamma_p::<f64>(a, a + 80.0);
            assert!((tail - 1.0).abs() < 1e-9, "P({a}, {}) = {tail}", a + 80.0);
        }
    }

    /// `P(a, ·)` is a CDF: it stays in `[0, 1]` and is monotone non-decreasing in `x`,
    /// across the series/CF switch at `x = a + 1`.
    #[test]
    fn regularized_gamma_p_is_a_monotone_cdf() {
        let a = 3.0;
        let mut prev = -1.0;
        for k in 0..=40 {
            let x = k as f64 * 0.5;
            let p = regularized_gamma_p::<f64>(a, x);
            assert!((-1e-15..=1.0 + 1e-12).contains(&p), "P out of [0,1]: {p}");
            assert!(p >= prev - 1e-12, "non-monotone at x={x}");
            prev = p;
        }
    }

    /// The series (`x < a+1`) and continued fraction (`x ≥ a+1`) must agree across the
    /// seam: stepping `±ε` over `x = a + 1` may move `P` only by `O(ε·pdf)` (the true
    /// derivative), never the `O(1)` jump a branch mismatch would produce.
    #[test]
    fn regularized_gamma_p_continuous_across_branch_seam() {
        for &a in &[0.7, 2.0, 4.5] {
            let seam = a + 1.0;
            let eps = 1e-6;
            let below = regularized_gamma_p::<f64>(a, seam - eps); // series
            let above = regularized_gamma_p::<f64>(a, seam + eps); // continued fraction
            assert!(
                (below - above).abs() < 1e-5,
                "branch jump at a={a}: {below} vs {above}"
            );
        }
    }

    /// The point of the generic `PkNum` form: the `Dual2` 1st **and** 2nd-order
    /// sensitivities in *both* arguments (and the cross term) must match central
    /// finite differences of the `f64` value — including the non-elementary `∂P/∂a`.
    /// Cases span both branches and the transit-typical large-`x` (CF) regime; a
    /// mis-transcribed series/CF coefficient or a broken `ln_gamma` dual rule fails
    /// here. (FD-only checks would miss exactly this — cf. #317 / #458.)
    #[test]
    fn regularized_gamma_p_dual_gradients_match_central_fd() {
        use crate::sens::dual2::Dual2;
        let p = |av: f64, xv: f64| regularized_gamma_p::<f64>(av, xv);
        // (a, x): series (x<a+1) for (2,1)/(3.5,2), CF (x≥a+1) for the rest incl.
        // transit-like large x (6,16). (Near-seam CF is covered by its own test below.)
        for &(av, xv) in &[(2.0, 1.0), (3.5, 2.0), (4.0, 9.0), (1.5, 6.0), (6.0, 16.0)] {
            let dual = regularized_gamma_p(Dual2::<2>::var(av, 0), Dual2::<2>::var(xv, 1));
            assert_relative_eq!(dual.value, p(av, xv), max_relative = 1e-12);

            let h = 1e-5;
            let dpda = (p(av + h, xv) - p(av - h, xv)) / (2.0 * h);
            let dpdx = (p(av, xv + h) - p(av, xv - h)) / (2.0 * h);
            assert_relative_eq!(dual.grad[0], dpda, max_relative = 1e-5, epsilon = 1e-9);
            assert_relative_eq!(dual.grad[1], dpdx, max_relative = 1e-5, epsilon = 1e-9);

            let h2 = 1e-3;
            let d2a = (p(av + h2, xv) - 2.0 * p(av, xv) + p(av - h2, xv)) / (h2 * h2);
            let d2x = (p(av, xv + h2) - 2.0 * p(av, xv) + p(av, xv - h2)) / (h2 * h2);
            let dax = (p(av + h2, xv + h2) - p(av + h2, xv - h2) - p(av - h2, xv + h2)
                + p(av - h2, xv - h2))
                / (4.0 * h2 * h2);
            assert_relative_eq!(dual.hess[0][0], d2a, max_relative = 2e-3, epsilon = 1e-6);
            assert_relative_eq!(dual.hess[1][1], d2x, max_relative = 2e-3, epsilon = 1e-6);
            assert_relative_eq!(dual.hess[0][1], dax, max_relative = 2e-3, epsilon = 1e-6);
        }
    }

    /// Exercise the second-order chain through a **composite** input (`a = e^θ`, so the
    /// seeded variable carries a nonzero input Hessian `x″`): `∂²P/∂θ²` must match the
    /// central second difference of `θ ↦ P(e^θ, x)`. This is the `ψ·x″`-style term the
    /// bare-`var` case in the test above does not exercise (cf. the `ln_gamma` chain
    /// test in `sens::num`).
    #[test]
    fn regularized_gamma_p_dual_chain_rule_with_composite_input() {
        use crate::sens::dual2::Dual2;
        let f = |theta: f64, xv: f64| regularized_gamma_p::<f64>(theta.exp(), xv);
        for &(theta, xv) in &[(0.4_f64, 1.0), (1.0, 9.0), (-0.3, 3.0)] {
            let dual =
                regularized_gamma_p(Dual2::<1>::var(theta, 0).exp(), Dual2::<1>::from_f64(xv));
            assert_relative_eq!(dual.value, f(theta, xv), max_relative = 1e-12);
            let h = 1e-5;
            let fd1 = (f(theta + h, xv) - f(theta - h, xv)) / (2.0 * h);
            assert_relative_eq!(dual.grad[0], fd1, max_relative = 1e-5, epsilon = 1e-9);
            let h2 = 1e-3;
            let fd2 = (f(theta + h2, xv) - 2.0 * f(theta, xv) + f(theta - h2, xv)) / (h2 * h2);
            assert_relative_eq!(dual.hess[0][0], fd2, max_relative = 2e-3, epsilon = 1e-6);
        }
    }

    /// Regression for the continued-fraction `SETTLE` streak (see `gamma_q_cf`): just
    /// inside the CF branch (`x` barely above `a + 1`) the CF *value* converges several
    /// iterations before its `∂/∂a` jet, and for **integer `a`** the naive single-step
    /// `|del−1| < EPS` break fires at iteration 1 (`aₙ = −n·(n − a) = 0 ⇒ del ≡ 1`),
    /// truncating the jet. This pins both 1st- and 2nd-order `∂/∂a` (the transit-`N`
    /// gradient direction) against central FD in exactly that regime. Without the streak
    /// fix, `∂P/∂a` at `(1, 2.001)` is wrong by ~1.8e-3 (≈0.8% relative) — this test
    /// fails; the deeper-CF points of the test above do not exercise it.
    #[test]
    fn regularized_gamma_p_dual_da_accurate_just_inside_cf_seam() {
        use crate::sens::dual2::Dual2;
        let p = |av: f64, xv: f64| regularized_gamma_p::<f64>(av, xv);
        // Integer `a` (the aₙ=0 trap) and a non-integer, each just inside the CF branch.
        for &(av, xv) in &[
            (1.0, 2.001),
            (2.0, 3.001),
            (4.0, 5.02),
            (1.0, 2.5),
            (3.0, 4.1),
        ] {
            assert!(xv >= av + 1.0, "case ({av},{xv}) must be in the CF branch");
            let dual = regularized_gamma_p(Dual2::<2>::var(av, 0), Dual2::<2>::var(xv, 1));
            let h = 1e-6;
            let dpda = (p(av + h, xv) - p(av - h, xv)) / (2.0 * h);
            let dpdx = (p(av, xv + h) - p(av, xv - h)) / (2.0 * h);
            assert_relative_eq!(dual.grad[0], dpda, max_relative = 1e-4, epsilon = 1e-7);
            assert_relative_eq!(dual.grad[1], dpdx, max_relative = 1e-4, epsilon = 1e-7);
            // ALL second-order components must be converged near the seam — not just
            // ∂²/∂a² (the SETTLE streak covers every jet, but the slow `a`-direction
            // also drives the cross term, so pin ∂²/∂a∂x and ∂²/∂x² too).
            let h2 = 1e-3;
            let d2a = (p(av + h2, xv) - 2.0 * p(av, xv) + p(av - h2, xv)) / (h2 * h2);
            let d2x = (p(av, xv + h2) - 2.0 * p(av, xv) + p(av, xv - h2)) / (h2 * h2);
            let d2ax = (p(av + h2, xv + h2) - p(av + h2, xv - h2) - p(av - h2, xv + h2)
                + p(av - h2, xv - h2))
                / (4.0 * h2 * h2);
            assert_relative_eq!(dual.hess[0][0], d2a, max_relative = 5e-3, epsilon = 1e-5);
            assert_relative_eq!(dual.hess[1][1], d2x, max_relative = 5e-3, epsilon = 1e-5);
            assert_relative_eq!(dual.hess[0][1], d2ax, max_relative = 5e-3, epsilon = 1e-5);
        }
    }
}
