//! `PkNum` — the small numeric interface a closed-form PK solution needs, so a
//! single generic implementation serves both the scalar prediction (`T = f64`)
//! and its exact sensitivities (`T = Dual2<N>`). No Enzyme, no codegen: the same
//! source is monomorphised for each numeric type.

use super::dual1::Dual1;
use super::dual_mixed::DualMixed;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait PkNum:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Lift a constant into the numeric type (zero derivatives for duals).
    fn from_f64(x: f64) -> Self;
    /// Seed dual dimension `dim` as an independent variable at value `x`. `f64`
    /// carries no derivatives, so it ignores `dim` and behaves like `from_f64`.
    fn var(x: f64, dim: usize) -> Self;
    /// The underlying value — for guards / branch conditions only.
    fn val(self) -> f64;
    /// `exp`.
    fn exp(self) -> Self;
    /// `ln`.
    fn ln(self) -> Self;
    /// `sqrt`.
    fn sqrt(self) -> Self;
    /// `self^e` (binary power; the bytecode `Op::Pow`).
    fn pow(self, e: Self) -> Self;
    /// `|self|`.
    fn abs(self) -> Self;
    /// `inv_logit(self) = 1/(1+e^{−x})` (logistic sigmoid).
    fn inv_logit(self) -> Self;
    /// `logit(self) = ln(x/(1−x))`.
    fn logit(self) -> Self;
    /// `cos` (3-cpt trigonometric cubic root solve).
    fn cos(self) -> Self;
    /// `acos` (3-cpt trigonometric cubic root solve).
    fn acos(self) -> Self;
    /// Lower-clamp the value to `lo` (branch on `.val()`), used by the bytecode
    /// VM to reproduce the `Op::Ln`/`Op::Sqrt` domain guards (`v.max(lo)`) before
    /// the transcendental call. For duals the clamped region is flat (zero jet).
    fn guard_floor(self, lo: f64) -> Self;
    /// `ln Γ(self)` (log-gamma). The transit absorption forcing's `ln Γ(n + 1)`
    /// constant rides this on the analytic ODE sensitivity path, so for the dual
    /// types it must carry 1st/2nd-order derivatives (digamma/trigamma) (#430).
    fn ln_gamma(self) -> Self;
}

impl PkNum for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn var(x: f64, _dim: usize) -> Self {
        x
    }
    #[inline]
    fn val(self) -> f64 {
        self
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn pow(self, e: Self) -> Self {
        f64::powf(self, e)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn inv_logit(self) -> Self {
        1.0 / (1.0 + f64::exp(-self))
    }
    #[inline]
    fn logit(self) -> Self {
        f64::ln(self / (1.0 - self))
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline]
    fn guard_floor(self, lo: f64) -> Self {
        self.max(lo)
    }
    #[inline]
    fn ln_gamma(self) -> Self {
        crate::stats::special::ln_gamma(self)
    }
}

impl<const N: usize> PkNum for Dual1<N> {
    #[inline]
    fn from_f64(x: f64) -> Self {
        Dual1::constant(x)
    }
    #[inline]
    fn var(x: f64, dim: usize) -> Self {
        Dual1::var(x, dim)
    }
    #[inline]
    fn val(self) -> f64 {
        self.value
    }
    #[inline]
    fn exp(self) -> Self {
        Dual1::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        Dual1::ln(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        Dual1::sqrt(self)
    }
    #[inline]
    fn pow(self, e: Self) -> Self {
        Dual1::powd(self, e)
    }
    #[inline]
    fn abs(self) -> Self {
        Dual1::abs(self)
    }
    #[inline]
    fn inv_logit(self) -> Self {
        Dual1::inv_logit(self)
    }
    #[inline]
    fn logit(self) -> Self {
        Dual1::logit(self)
    }
    #[inline]
    fn cos(self) -> Self {
        Dual1::cos(self)
    }
    #[inline]
    fn acos(self) -> Self {
        Dual1::acos(self)
    }
    #[inline]
    fn guard_floor(self, lo: f64) -> Self {
        // `!(value >= lo)` (not `value < lo`) so a `NaN` value floors too, matching
        // `f64::max(lo)` on the scalar path (`NaN.max(lo) == lo`) — otherwise a
        // transient `NaN` would give a finite f64 prediction but a `NaN` dual
        // gradient (the clamped region is flat, so the floored jet is zero) (#430).
        if !(self.value >= lo) {
            Dual1::constant(lo)
        } else {
            self
        }
    }
    #[inline]
    fn ln_gamma(self) -> Self {
        Dual1::ln_gamma(self)
    }
}

impl<const NA: usize, const N: usize> PkNum for DualMixed<NA, N> {
    #[inline]
    fn from_f64(x: f64) -> Self {
        DualMixed::constant(x)
    }
    #[inline]
    fn var(x: f64, dim: usize) -> Self {
        DualMixed::var(x, dim)
    }
    #[inline]
    fn val(self) -> f64 {
        self.value
    }
    #[inline]
    fn exp(self) -> Self {
        DualMixed::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        DualMixed::ln(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        DualMixed::sqrt(self)
    }
    #[inline]
    fn pow(self, e: Self) -> Self {
        DualMixed::powd(self, e)
    }
    #[inline]
    fn abs(self) -> Self {
        DualMixed::abs(self)
    }
    #[inline]
    fn inv_logit(self) -> Self {
        DualMixed::inv_logit(self)
    }
    #[inline]
    fn logit(self) -> Self {
        DualMixed::logit(self)
    }
    #[inline]
    fn cos(self) -> Self {
        DualMixed::cos(self)
    }
    #[inline]
    fn acos(self) -> Self {
        DualMixed::acos(self)
    }
    #[inline]
    fn guard_floor(self, lo: f64) -> Self {
        // `!(value >= lo)` floors `NaN` too, matching `f64::max(lo)` — see the
        // `Dual1` impl above (#430).
        if !(self.value >= lo) {
            DualMixed::constant(lo)
        } else {
            self
        }
    }
    #[inline]
    fn ln_gamma(self) -> Self {
        DualMixed::ln_gamma(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sens::dual1::Dual1;
    use crate::sens::dual2::Dual2;
    use crate::sens::dual_mixed::DualMixed;

    /// Exercise every `PkNum` method on a value once, so the per-impl delegators
    /// (and the underlying `Dual1`/`Dual2` ops they call) are covered. `0.7` is a
    /// safe argument for all: positive (exp/ln/sqrt/pow), in `(0,1)` (logit/
    /// inv_logit) and in `[-1,1]` (acos).
    fn exercise<T: PkNum>(x: T) {
        let _ = T::from_f64(1.0).val();
        let _ = x.val();
        let _ = x.exp().val();
        let _ = x.ln().val();
        let _ = x.sqrt().val();
        let _ = x.pow(T::from_f64(2.0)).val();
        let _ = x.abs().val();
        let _ = (-x.abs()).abs().val(); // negative branch of abs
        let _ = x.cos().val();
        let _ = x.acos().val();
        let _ = x.inv_logit().val();
        let _ = x.logit().val();
        let _ = x.guard_floor(1e-6).val();
        let _ = x.guard_floor(10.0).val(); // floor-active branch
        let _ = x.ln_gamma().val(); // 0.7 > 0, in the ln_gamma domain
    }

    #[test]
    fn pknum_all_methods_covered_for_each_impl() {
        exercise::<f64>(0.7);
        exercise::<Dual1<1>>(Dual1::var(0.7, 0));
        exercise::<Dual2<1>>(Dual2::var(0.7, 0));
        exercise::<DualMixed<1, 2>>(DualMixed::var(0.7, 0));
    }

    /// `guard_floor(NaN)` must floor (return `lo`) on the duals exactly as it does on
    /// `f64` (`NaN.max(lo) == lo`) — a sub-floor / `NaN` value lands in the flat
    /// clamped region, so the dual's value is `lo` and its jet is zero. Without this,
    /// a transient `NaN` would give a finite f64 prediction but a `NaN` dual gradient
    /// (#430 review #2).
    #[test]
    fn guard_floor_floors_nan_consistently_across_impls() {
        let lo = 1e-6;
        assert_eq!(f64::NAN.guard_floor(lo), lo);
        assert_eq!(Dual1::<1>::var(f64::NAN, 0).guard_floor(lo).value, lo);
        assert_eq!(Dual2::<1>::var(f64::NAN, 0).guard_floor(lo).value, lo);
        assert_eq!(
            DualMixed::<1, 2>::var(f64::NAN, 0).guard_floor(lo).value,
            lo
        );
        // The floored dual carries a zero jet (flat clamped region).
        let g = Dual1::<1>::var(f64::NAN, 0).guard_floor(lo);
        assert_eq!(g.grad[0], 0.0);
        // A value already above the floor is untouched.
        assert_eq!(Dual1::<1>::var(5.0, 0).guard_floor(lo).value, 5.0);
    }

    /// The `ln_gamma` `Dual2` rule: its analytic 1st (digamma) and 2nd (trigamma)
    /// derivatives must match central finite differences of `special::ln_gamma`,
    /// since the transit forcing's `ln Γ(n + 1)` constant rides this on the analytic
    /// ODE sensitivity path (#430 slice 2). This is exactly the `Dual2`-rule
    /// regression an FD-only check would miss (cf. the `guard_floor` NaN bug, #2).
    #[test]
    fn ln_gamma_dual2_matches_central_fd() {
        use crate::stats::special::ln_gamma;
        for &x in &[0.6, 1.0, 2.5, 4.0, 7.3] {
            let d = Dual2::<1>::var(x, 0).ln_gamma();
            approx::assert_relative_eq!(d.value, ln_gamma(x), max_relative = 1e-12);
            // 1st derivative (digamma) vs central FD of ln_gamma.
            let h1 = 1e-5;
            let fd1 = (ln_gamma(x + h1) - ln_gamma(x - h1)) / (2.0 * h1);
            approx::assert_relative_eq!(d.grad[0], fd1, max_relative = 1e-6, epsilon = 1e-9);
            // 2nd derivative (trigamma) vs central second difference of ln_gamma.
            let h2 = 1e-4;
            let fd2 = (ln_gamma(x + h2) - 2.0 * ln_gamma(x) + ln_gamma(x - h2)) / (h2 * h2);
            approx::assert_relative_eq!(d.hess[0][0], fd2, max_relative = 1e-4, epsilon = 1e-6);
        }
    }

    /// The dual `ln_gamma` chain rule has two second-order terms — `ψ′·x′⊗x′`
    /// **and** `ψ·x″`. `ln_gamma_dual2_matches_central_fd` seeds a bare `var`
    /// (x″ = 0), so it exercises only the first. Feed a dual with a NONZERO input
    /// Hessian via `x ↦ ln Γ(eˣ)` (the inner `exp` carries x′ and x″) so the
    /// `ψ·x″` term is exercised too — the path real transit fits hit when an
    /// absorption param is composite (e.g. `MTT = TVMTT·exp(ETA_MTT)`). Negative
    /// `x` (eˣ < 0.5) additionally drives the **reflection** branch of the dual
    /// rule (#458 review #1/#2). Without `ψ·x″`, the main-branch points fail; with
    /// a wrong reflection rule, the negative points fail.
    #[test]
    fn ln_gamma_dual2_chain_rule_nonzero_input_hessian_and_reflection() {
        use crate::stats::special::ln_gamma;
        let f = |x: f64| ln_gamma(x.exp());
        // x = −1.5, −1.0 → eˣ = 0.22, 0.37 < 0.5 (reflection); the rest main branch.
        for &x in &[-1.5_f64, -1.0, 0.1, 0.5, 1.2, 2.0] {
            let u = Dual2::<1>::var(x, 0).exp().ln_gamma();
            approx::assert_relative_eq!(u.value, f(x), max_relative = 1e-12);
            let h = 1e-5;
            let fd1 = (f(x + h) - f(x - h)) / (2.0 * h);
            approx::assert_relative_eq!(u.grad[0], fd1, max_relative = 1e-6, epsilon = 1e-9);
            let h2 = 1e-4;
            let fd2 = (f(x + h2) - 2.0 * f(x) + f(x - h2)) / (h2 * h2);
            approx::assert_relative_eq!(u.hess[0][0], fd2, max_relative = 1e-4, epsilon = 1e-6);
        }
    }
}
