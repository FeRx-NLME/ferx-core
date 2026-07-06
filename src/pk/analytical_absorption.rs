//! Analytical closed forms for absorption models that are **closed under
//! exponential tilting** — the #386 / absorption-Phase-3 acceleration that moves
//! `transit()` (and, later, `igd()`) off the numerical ODE forcing path onto the
//! analytic `pk` path, where sensitivities come from `sens/`'s `Dual2` jets
//! rather than finite differences.
//!
//! ## The tilting identity
//!
//! A 1-cpt central compartment fed by an absorption-time density `f` (with
//! `∫₀^∞ f = 1`, so the whole `F·Dose` is eventually delivered) and eliminated at
//! rate `ke` has
//!
//! ```text
//!   A_central(t) = F·Dose · ∫₀ᵗ f(u) e^{-ke(t-u)} du
//!                = F·Dose · e^{-ke t} ∫₀ᵗ f(u) e^{ke u} du.
//! ```
//!
//! Writing the MGF `M(k) = ∫₀^∞ e^{ku} f(u) du` and the CDF of the
//! exponentially-tilted density `g(u) = e^{ku} f(u) / M(k)` as
//! `G(t; k) = (1/M(k)) ∫₀ᵗ e^{ku} f(u) du`, the inner integral is `M(k)·G(t; k)`,
//! so
//!
//! ```text
//!   C(t) = A_central(t)/V = (F·Dose/V) · M(ke) · e^{-ke t} · G(t; ke).
//! ```
//!
//! Any density that is *closed under tilting* — its tilted form stays in the same
//! family with a closed-form CDF — therefore gives an elementary `C(t)`. The
//! [`TiltedAbsorption`] trait captures the two pieces `M` and `G`; [`convolve_1cpt`]
//! assembles them. A 2-cpt disposition decomposes into the same tilting form at
//! its two macro-rates `α`, `β` — see [`convolve_2cpt`] (#386 Phase-3 PR D).
//!
//! ## Generic over `PkNum`
//!
//! Every function here is written once over `T: PkNum` (the `sens/ *_g<T>`
//! convention), so `T = f64` gives the plain concentration and `T = Dual2<N>`
//! gives the exact `∂C/∂θ`, `∂C/∂η` (and 2nd-order) gradients FOCE/FOCEI need —
//! no hand-derived absorption-gradient rule and no ODE solve. The non-elementary
//! `∂/∂a` of the incomplete gamma rides through [`regularized_gamma_p`], which is
//! itself generic over `PkNum` (#386 PR B).
//!
//! ## Domain
//!
//! The tilting identity needs `ke` below the MGF's abscissa of convergence
//! (`ke < KTR` for transit). Physically this is "elimination slower than the
//! absorption rate constant" — the usual absorption-rate-limited regime. Callers
//! that may violate it must guard upstream: the analytic dispatch keeps such a
//! model on the numerical ODE forcing path instead.

use crate::sens::num::PkNum;
use crate::stats::special::regularized_gamma_p;

/// An absorption-time distribution that is **closed under exponential tilting**:
/// both its MGF `M(k) = E[e^{kX}]` and the CDF of the `e^{kt}`-tilted density have
/// closed forms. Implementors plug into [`convolve_1cpt`] / [`convolve_2cpt`]
/// to give an elementary central-compartment concentration.
///
/// Generic over the numeric type `T` so one implementation serves both the
/// scalar prediction (`T = f64`) and its `Dual2` sensitivities.
pub trait TiltedAbsorption<T: PkNum> {
    /// Moment-generating function `M(k) = E[e^{kX}] = ∫₀^∞ e^{ku} f(u) du`, for
    /// `k` below the distribution's abscissa of convergence.
    fn mgf(&self, k: T) -> T;

    /// CDF at `t` of the `e^{ku}`-tilted density `g(u) = e^{ku} f(u) / M(k)`,
    /// i.e. `G(t; k) = (1/M(k)) ∫₀ᵗ e^{ku} f(u) du`.
    fn tilted_cdf(&self, t: T, k: T) -> T;
}

/// Savic et al. (2007) transit-compartment absorption: the dose's first-passage
/// time into central is `Gamma(shape = n+1, rate = KTR)` with `KTR = (n+1)/mtt`.
/// The Gamma family is closed under exponential tilting, so both
/// [`TiltedAbsorption`] pieces are elementary.
///
/// This is the *continuous-N* Savic approximation (`n` need not be an integer) —
/// the same `transit(n, mtt)` density the ODE forcing path implements (see
/// [`crate::pk::absorption::PreparedInputRate`]). With `n = 0` the chain reduces
/// to first-order absorption with `ka = KTR = 1/mtt`.
pub struct TransitAbsorption<T: PkNum> {
    /// Number of transit compartments (continuous), `n ≥ 0`.
    pub n: T,
    /// Mean transit time, `mtt > 0`.
    pub mtt: T,
}

impl<T: PkNum> TransitAbsorption<T> {
    /// Transit rate constant `KTR = (n+1)/mtt`.
    #[inline]
    fn ktr(&self) -> T {
        (self.n + T::from_f64(1.0)) / self.mtt
    }
}

impl<T: PkNum> TiltedAbsorption<T> for TransitAbsorption<T> {
    fn mgf(&self, k: T) -> T {
        // Gamma(n+1, KTR) MGF: M(k) = (KTR/(KTR−k))^{n+1}, converges for k < KTR.
        let ktr = self.ktr();
        // Enforce the domain at the point of violation: above the abscissa the base
        // KTR/(KTR−k) goes negative and `.pow` of a non-integer exponent is NaN,
        // which would otherwise propagate silently into the likelihood. The analytic
        // dispatch guards `ke < KTR` upstream (routing to the ODE path otherwise), so
        // this never fires on a valid call — it catches a guard regression in tests.
        debug_assert!(
            k.val() < ktr.val(),
            "transit MGF diverges for k ≥ KTR ({} ≥ {}); caller must guard ke < KTR",
            k.val(),
            ktr.val()
        );
        (ktr / (ktr - k)).pow(self.n + T::from_f64(1.0))
    }

    fn tilted_cdf(&self, t: T, k: T) -> T {
        // The e^{ku}-tilted Gamma(n+1, KTR) is Gamma(n+1, KTR−k); its CDF at t is
        // the regularized lower incomplete gamma P(n+1, (KTR−k)·t).
        let ktr = self.ktr();
        regularized_gamma_p(self.n + T::from_f64(1.0), (ktr - k) * t)
    }
}

/// Central-compartment concentration at time `t` for a single dose absorbed into
/// a **1-cpt** disposition through `abs`, eliminated at rate `ke = CL/V`:
///
/// ```text
///   C(t) = (F·Dose/V) · M(ke) · e^{-ke t} · G(t; ke).
/// ```
///
/// `f_dose_over_v = F·Dose/V`. Requires `ke` below `abs`'s MGF abscissa (for
/// transit, `ke < KTR`) — the caller guards this upstream.
#[inline]
pub fn convolve_1cpt<T: PkNum, A: TiltedAbsorption<T>>(
    abs: &A,
    t: T,
    ke: T,
    f_dose_over_v: T,
) -> T {
    f_dose_over_v * abs.mgf(ke) * (-(ke * t)).exp() * abs.tilted_cdf(t, ke)
}

/// Central-compartment concentration at time `t` for a single dose absorbed into a
/// **2-cpt** disposition through `abs`, with macro-rate constants `α ≥ β` and
/// peripheral micro-rate `k21` (all from
/// [`crate::sens::two_cpt::macro_rates_g`]):
///
/// ```text
///   C(t) = (F·Dose/V1) · [ cα·M(α)·e^{-α t}·G(t; α) + cβ·M(β)·e^{-β t}·G(t; β) ],
///   cα = (α−k21)/(α−β),   cβ = (k21−β)/(α−β)    (cα + cβ = 1).
/// ```
///
/// The 2-cpt IV-bolus central impulse response is the bi-exponential
/// `(1/V1)[cα e^{-αt} + cβ e^{-βt}]` (the same `cα`, `cβ` as
/// [`crate::sens::two_cpt::two_cpt_iv_bolus_amt_g`]); convolving each disposition
/// exponential with the absorption density turns `e^{-rate·t}` into the tilting
/// form `M(rate)·e^{-rate·t}·G(t; rate)`, so each term is one [`convolve_1cpt`]
/// call. With a degenerate absorption (`M ≡ 1`, `G ≡ 1`, e.g. transit `mtt → 0`)
/// it reduces to the IV bolus; with first-order absorption (transit `n = 0`) it
/// reduces to [`crate::sens::two_cpt::two_cpt_oral_amt_g`].
///
/// `f_dose_over_v1 = F·Dose/V1`. Requires **both** macro-rates below `abs`'s MGF
/// abscissa; since `α ≥ β`, the caller's single `α < KTR` guard
/// ([`crate::sens::two_cpt::two_cpt_transit_amt_g`]) suffices. The caller also
/// excludes the confluent `α = β` case (`diff → 0`).
#[inline]
pub fn convolve_2cpt<T: PkNum, A: TiltedAbsorption<T>>(
    abs: &A,
    t: T,
    alpha: T,
    beta: T,
    k21: T,
    f_dose_over_v1: T,
) -> T {
    let diff = alpha - beta;
    let c_alpha = (alpha - k21) / diff;
    let c_beta = (k21 - beta) / diff;
    // Each term is c·M(rate)·e^{-rate·t}·G(t;rate) — one convolve_1cpt with the
    // coefficient passed as its `f_dose_over_v` weight; the shared F·Dose/V1 scales
    // the sum. No second copy of the tilting algebra.
    f_dose_over_v1 * (convolve_1cpt(abs, t, alpha, c_alpha) + convolve_1cpt(abs, t, beta, c_beta))
}

/// Peripheral-compartment **concentration** at time `t` (`A2(t)/V2`) for a single
/// dose absorbed into a 2-cpt disposition through `abs`, with macro-rates `α ≥ β`
/// and central→peripheral micro-rate `k12 = Q/V1`:
///
/// ```text
///   C2(t) = (F·Dose/V2) · (k12/(α−β)) · [ M(β)·e^{-β t}·G(t; β) − M(α)·e^{-α t}·G(t; α) ].
/// ```
///
/// The peripheral IV-bolus impulse response (amount) is
/// `(k12/(α−β))·(e^{-βt} − e^{-αt})`; convolving with the absorption density and
/// dividing by `V2` gives the two-term tilting form above (each a [`convolve_1cpt`]
/// call with the shared coefficient). Used only for `[derived]` peripheral amounts
/// — the likelihood needs only the central concentration ([`convolve_2cpt`]).
/// Same domain requirement and `α ≠ β` exclusion as [`convolve_2cpt`].
#[inline]
pub(crate) fn convolve_2cpt_peripheral<T: PkNum, A: TiltedAbsorption<T>>(
    abs: &A,
    t: T,
    alpha: T,
    beta: T,
    k12: T,
    f_dose_over_v2: T,
) -> T {
    let coeff = f_dose_over_v2 * k12 / (alpha - beta);
    convolve_1cpt(abs, t, beta, coeff) - convolve_1cpt(abs, t, alpha, coeff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sens::dual2::Dual2;
    use crate::stats::special::ln_gamma;
    use approx::assert_relative_eq;

    /// `f64` concentration for a transit model, for the value-level tests.
    fn conc(n: f64, mtt: f64, t: f64, ke: f64, f_dose_over_v: f64) -> f64 {
        let abs = TransitAbsorption { n, mtt };
        convolve_1cpt(&abs, t, ke, f_dose_over_v)
    }

    /// Same concentration with `n`, `mtt`, `ke` seeded as the three `Dual2`
    /// variables (dims 0, 1, 2), so `.grad`/`.hess` carry the exact derivatives.
    fn dual_conc(n: f64, mtt: f64, t: f64, ke: f64, f_dose_over_v: f64) -> Dual2<3> {
        let abs = TransitAbsorption {
            n: Dual2::<3>::var(n, 0),
            mtt: Dual2::<3>::var(mtt, 1),
        };
        convolve_1cpt(
            &abs,
            Dual2::<3>::from_f64(t),
            Dual2::<3>::var(ke, 2),
            Dual2::<3>::from_f64(f_dose_over_v),
        )
    }

    /// With `n = 0` the transit chain is exponential absorption with
    /// `ka = KTR = 1/mtt`, so `convolve_1cpt` must reproduce the Bateman
    /// (one-compartment first-order) equation exactly.
    #[test]
    fn transit_n0_recovers_bateman() {
        let mtt = 2.0;
        let ka = 1.0 / mtt; // KTR for n = 0
        let ke = 0.1;
        let f_dose_over_v = 2.0;
        let bateman = |t: f64| f_dose_over_v * ka / (ka - ke) * ((-ke * t).exp() - (-ka * t).exp());
        for &t in &[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0] {
            assert_relative_eq!(
                conc(0.0, mtt, t, ke, f_dose_over_v),
                bateman(t),
                max_relative = 1e-12,
                epsilon = 1e-14
            );
        }
    }

    /// The closed form must equal the defining convolution
    /// `(F·Dose/V) ∫₀ᵗ f(u) e^{-ke(t-u)} du` with `f` the `Gamma(n+1, KTR)`
    /// transit density — checked by fine quadrature for *non-integer* `n` (so this
    /// is independent of the `n=0` Bateman identity and of the ODE path).
    #[test]
    fn convolve_1cpt_matches_numerical_convolution() {
        let f_dose_over_v = 2.0;
        // (n, mtt, ke) — the last case is near the seam (ke = 0.9·KTR, KTR = 3.5),
        // where the MGF factor (KTR/(KTR−ke))^{n+1} ≈ 10^{3.5} is large; because the
        // closed form is a *product* M(ke)·G(t;ke) (not a difference), it must still
        // match the convolution there.
        for &(n, mtt, ke) in &[
            (2.5, 1.0, 0.10),
            (1.3, 0.7, 0.20),
            (4.0, 2.0, 0.05),
            (0.4, 3.0, 0.15),
            (2.5, 1.0, 0.9 * 3.5),
        ] {
            let ktr = (n + 1.0) / mtt;
            // Gamma(n+1, KTR) density f(u) = KTR^{n+1} u^n e^{-KTR u} / Γ(n+1).
            let ln_norm = (n + 1.0) * ktr.ln() - ln_gamma(n + 1.0);
            let dens = |u: f64| (ln_norm + n * u.ln() - ktr * u).exp();
            for &t in &[0.5, 1.0, 2.0, 4.0, 8.0] {
                // Composite trapezoid for ∫₀ᵗ f(u) e^{-ke(t-u)} du.
                let steps = 200_000usize;
                let h = t / steps as f64;
                let mut acc = 0.0;
                for i in 0..=steps {
                    let u = i as f64 * h;
                    // f(0)=0 for n>0; the u^n factor handles the u=0 endpoint.
                    let integrand = if u == 0.0 {
                        0.0
                    } else {
                        dens(u) * (-ke * (t - u)).exp()
                    };
                    let w = if i == 0 || i == steps { 0.5 } else { 1.0 };
                    acc += w * integrand;
                }
                let numeric = f_dose_over_v * acc * h;
                assert_relative_eq!(
                    conc(n, mtt, t, ke, f_dose_over_v),
                    numeric,
                    max_relative = 1e-4,
                    epsilon = 1e-9
                );
            }
        }
    }

    /// MGF / tilted-CDF sanity: `M(0) = 1`, `G(0; k) = 0`, `G` rises to 1, and the
    /// concentration starts at 0 and is positive thereafter.
    #[test]
    fn mgf_and_tilted_cdf_sanity() {
        let abs = TransitAbsorption { n: 3.0, mtt: 1.5 };
        assert_relative_eq!(abs.mgf(0.0), 1.0, max_relative = 1e-12);
        assert_relative_eq!(abs.tilted_cdf(0.0, 0.2), 0.0, epsilon = 1e-12);
        assert_relative_eq!(abs.tilted_cdf(1e6, 0.2), 1.0, max_relative = 1e-9);
        assert_relative_eq!(conc(3.0, 1.5, 0.0, 0.1, 2.0), 0.0, epsilon = 1e-12);
        assert!(conc(3.0, 1.5, 2.0, 0.1, 2.0) > 0.0);
    }

    /// The reason for the closed form: exact `Dual2` sensitivities. We validate a
    /// two-rung ladder so each order is checked against the one below:
    ///   * 1st order `∂C/∂{n,mtt,ke}` vs a central difference of the `f64` value;
    ///   * 2nd order (all three diagonals plus the `∂²C/∂n∂mtt`, `∂²C/∂n∂ke`,
    ///     `∂²C/∂mtt∂ke` cross terms) vs a central difference of the **exact dual
    ///     1st-derivative** — this avoids the `1/h²` roundoff blow-up of a
    ///     value-based second difference (differencing values that nearly cancel),
    ///     the trap that a naive 2nd-order FD reference falls into.
    ///
    /// `t` and `F·Dose/V` are held constant here; their sensitivities are checked
    /// in [`convolve_1cpt_t_and_dose_sensitivities`].
    #[test]
    fn convolve_1cpt_dual_gradients_match_fd() {
        let fdv = 2.0;
        // (n, mtt, ke, t), all with ke < KTR = (n+1)/mtt.
        for &(nv, mv, kv, tv) in &[
            (2.5, 1.0, 0.10, 1.5),
            (1.3, 0.7, 0.20, 3.0),
            (4.0, 2.0, 0.05, 6.0),
            (0.4, 3.0, 0.15, 2.0),
        ] {
            let d = dual_conc(nv, mv, tv, kv, fdv);

            // 1st order vs central difference of the value.
            let h = 1e-6;
            let dn = (conc(nv + h, mv, tv, kv, fdv) - conc(nv - h, mv, tv, kv, fdv)) / (2.0 * h);
            let dm = (conc(nv, mv + h, tv, kv, fdv) - conc(nv, mv - h, tv, kv, fdv)) / (2.0 * h);
            let dk = (conc(nv, mv, tv, kv + h, fdv) - conc(nv, mv, tv, kv - h, fdv)) / (2.0 * h);
            assert_relative_eq!(d.grad[0], dn, max_relative = 1e-4, epsilon = 1e-8);
            assert_relative_eq!(d.grad[1], dm, max_relative = 1e-4, epsilon = 1e-8);
            assert_relative_eq!(d.grad[2], dk, max_relative = 1e-4, epsilon = 1e-8);

            // 2nd order vs central difference of the exact dual 1st-derivative.
            let h2 = 1e-4;
            let d2n = (dual_conc(nv + h2, mv, tv, kv, fdv).grad[0]
                - dual_conc(nv - h2, mv, tv, kv, fdv).grad[0])
                / (2.0 * h2);
            let d2m = (dual_conc(nv, mv + h2, tv, kv, fdv).grad[1]
                - dual_conc(nv, mv - h2, tv, kv, fdv).grad[1])
                / (2.0 * h2);
            let d2k = (dual_conc(nv, mv, tv, kv + h2, fdv).grad[2]
                - dual_conc(nv, mv, tv, kv - h2, fdv).grad[2])
                / (2.0 * h2);
            // cross term ∂²C/∂n∂ke: difference ∂C/∂n in the ke direction.
            let dnk = (dual_conc(nv, mv, tv, kv + h2, fdv).grad[0]
                - dual_conc(nv, mv, tv, kv - h2, fdv).grad[0])
                / (2.0 * h2);
            // cross term ∂²C/∂n∂mtt: difference ∂C/∂n in the mtt direction.
            let dnm = (dual_conc(nv, mv + h2, tv, kv, fdv).grad[0]
                - dual_conc(nv, mv - h2, tv, kv, fdv).grad[0])
                / (2.0 * h2);
            // cross term ∂²C/∂mtt∂ke: difference ∂C/∂mtt in the ke direction.
            let dmk = (dual_conc(nv, mv, tv, kv + h2, fdv).grad[1]
                - dual_conc(nv, mv, tv, kv - h2, fdv).grad[1])
                / (2.0 * h2);
            assert_relative_eq!(d.hess[0][0], d2n, max_relative = 1e-4, epsilon = 1e-7);
            assert_relative_eq!(d.hess[1][1], d2m, max_relative = 1e-4, epsilon = 1e-7);
            assert_relative_eq!(d.hess[2][2], d2k, max_relative = 1e-4, epsilon = 1e-7);
            assert_relative_eq!(d.hess[0][2], dnk, max_relative = 1e-4, epsilon = 1e-7);
            assert_relative_eq!(d.hess[0][1], dnm, max_relative = 1e-4, epsilon = 1e-7);
            assert_relative_eq!(d.hess[1][2], dmk, max_relative = 1e-4, epsilon = 1e-7);
        }
    }

    /// `n = 0` is the Bateman boundary: `a = n+1 = 1`, exactly where `special.rs`
    /// clamps `∂P/∂x` at `x → 0`. The `n=0` *value* is anchored against Bateman above;
    /// here we confirm the clamp doesn't corrupt the *live* (`t > 0`) gradient path —
    /// `∂C/∂mtt` and `∂C/∂ke` at `n = 0` match a central difference. (`∂C/∂n` is
    /// one-sided at the `n ≥ 0` boundary, not a meaningful two-sided derivative there,
    /// so it is intentionally not asserted.)
    #[test]
    fn transit_n0_gradients_match_fd() {
        let fdv = 2.0;
        let h = 1e-6;
        for &(mv, kv, tv) in &[(2.0, 0.10, 1.5), (0.7, 0.20, 3.0), (1.5, 0.05, 6.0)] {
            let d = dual_conc(0.0, mv, tv, kv, fdv);
            let dm = (conc(0.0, mv + h, tv, kv, fdv) - conc(0.0, mv - h, tv, kv, fdv)) / (2.0 * h);
            let dk = (conc(0.0, mv, tv, kv + h, fdv) - conc(0.0, mv, tv, kv - h, fdv)) / (2.0 * h);
            assert_relative_eq!(d.grad[1], dm, max_relative = 1e-4, epsilon = 1e-8);
            assert_relative_eq!(d.grad[2], dk, max_relative = 1e-4, epsilon = 1e-8);
        }
    }

    /// Sensitivities in the two arguments the gradient ladder holds constant:
    ///   * `∂C/∂t` (and `∂²C/∂t²`) — the only direct exercise of the incomplete
    ///     gamma's `∂P/∂x` through the closed form — vs the same two-rung FD ladder;
    ///   * `F·Dose/V`, in which `C` must be *exactly* linear (`∂C/∂fdv = C/fdv`,
    ///     `∂²C/∂fdv² = 0`).
    #[test]
    fn convolve_1cpt_t_and_dose_sensitivities() {
        let h = 1e-6;
        let h2 = 1e-4;
        for &(n, mtt, ke, t) in &[
            (2.5, 1.0, 0.10, 1.5),
            (1.3, 0.7, 0.20, 3.0),
            (0.4, 3.0, 0.15, 2.0),
        ] {
            let abs = TransitAbsorption {
                n: Dual2::<1>::from_f64(n),
                mtt: Dual2::<1>::from_f64(mtt),
            };
            let ke_c = Dual2::<1>::from_f64(ke);
            let fdv_c = Dual2::<1>::from_f64(2.0);
            let d = convolve_1cpt(&abs, Dual2::<1>::var(t, 0), ke_c, fdv_c);
            // 1st order ∂C/∂t vs central difference of the value.
            let dt = (conc(n, mtt, t + h, ke, 2.0) - conc(n, mtt, t - h, ke, 2.0)) / (2.0 * h);
            assert_relative_eq!(d.grad[0], dt, max_relative = 1e-4, epsilon = 1e-8);
            // 2nd order ∂²C/∂t² vs central difference of the exact dual 1st-derivative.
            let gp = convolve_1cpt(&abs, Dual2::<1>::var(t + h2, 0), ke_c, fdv_c).grad[0];
            let gm = convolve_1cpt(&abs, Dual2::<1>::var(t - h2, 0), ke_c, fdv_c).grad[0];
            assert_relative_eq!(
                d.hess[0][0],
                (gp - gm) / (2.0 * h2),
                max_relative = 1e-4,
                epsilon = 1e-7
            );
        }
        // C is exactly linear in F·Dose/V: ∂C/∂fdv = C/fdv, ∂²C/∂fdv² = 0.
        let fdv = 2.0;
        let abs = TransitAbsorption {
            n: Dual2::<1>::from_f64(2.5),
            mtt: Dual2::<1>::from_f64(1.0),
        };
        let d = convolve_1cpt(
            &abs,
            Dual2::<1>::from_f64(1.5),
            Dual2::<1>::from_f64(0.1),
            Dual2::<1>::var(fdv, 0),
        );
        assert_relative_eq!(
            d.grad[0],
            conc(2.5, 1.0, 1.5, 0.1, fdv) / fdv,
            max_relative = 1e-12
        );
        assert_relative_eq!(d.hess[0][0], 0.0, epsilon = 1e-12);
    }

    // ── convolve_2cpt (2-cpt disposition) ────────────────────────────────────

    use crate::sens::two_cpt::{macro_rates_g, two_cpt_oral_amt_g};

    /// `f64` central concentration for a 2-cpt transit model, `F·Dose/V1` folded
    /// in from `amt`/`f_bio` so seeding `v1` captures the full `∂C/∂v1`.
    fn tc(cl: f64, v1: f64, q: f64, v2: f64, n: f64, mtt: f64, t: f64, amt: f64) -> f64 {
        let (a, b, k21) = macro_rates_g(cl, v1, q, v2);
        let abs = TransitAbsorption { n, mtt };
        convolve_2cpt(&abs, t, a, b, k21, amt / v1)
    }

    /// Same with all six PK params seeded as `Dual2<6>` vars (cl,v1,q,v2,n,mtt =
    /// dims 0..6), so `.grad`/`.hess` carry the exact derivatives.
    fn tc_dual(cl: f64, v1: f64, q: f64, v2: f64, n: f64, mtt: f64, t: f64, amt: f64) -> Dual2<6> {
        let cld = Dual2::<6>::var(cl, 0);
        let v1d = Dual2::<6>::var(v1, 1);
        let qd = Dual2::<6>::var(q, 2);
        let v2d = Dual2::<6>::var(v2, 3);
        let nd = Dual2::<6>::var(n, 4);
        let mttd = Dual2::<6>::var(mtt, 5);
        let (a, b, k21) = macro_rates_g(cld, v1d, qd, v2d);
        let abs = TransitAbsorption { n: nd, mtt: mttd };
        convolve_2cpt(
            &abs,
            Dual2::<6>::from_f64(t),
            a,
            b,
            k21,
            Dual2::<6>::from_f64(amt) / v1d,
        )
    }

    /// With `n = 0` the transit chain is first-order absorption with
    /// `ka = KTR = 1/mtt`, so the 2-cpt convolution must reproduce the independent,
    /// already-validated `two_cpt_oral_amt_g` (the 2-cpt oral Bateman) **exactly** —
    /// the strongest check of the `cα`/`cβ` bi-exponential decomposition. Params are
    /// chosen absorption-rate-limited (`ka = 1/mtt > α`, i.e. `α < KTR`) so the
    /// tilting closed form converges — small `mtt` makes absorption faster than the
    /// fast disposition macro-rate.
    #[test]
    fn convolve_2cpt_n0_recovers_two_cpt_oral() {
        let amt = 100.0;
        // (cl, v1, q, v2, mtt) with ka = 1/mtt ABOVE the fast macro-rate α (n=0 ⇒
        // KTR = 1/mtt; the tilting form needs α < KTR).
        for &(cl, v1, q, v2, mtt) in &[
            (5.0, 20.0, 10.0, 40.0, 0.3), // α≈0.93 < ka≈3.33
            (3.0, 30.0, 6.0, 60.0, 0.5),  // α≈0.37 < ka=2.0
            (8.0, 50.0, 20.0, 30.0, 0.4), // α≈1.13 < ka=2.5
        ] {
            let ka = 1.0 / mtt;
            for &t in &[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0] {
                let got = tc(cl, v1, q, v2, 0.0, mtt, t, amt);
                let want = two_cpt_oral_amt_g::<f64>(amt, t, cl, v1, q, v2, ka, 1.0);
                assert_relative_eq!(got, want, max_relative = 1e-10, epsilon = 1e-12);
            }
        }
    }

    /// The 2-cpt closed form must equal the defining convolution
    /// `(F·Dose/V1) ∫₀ᵗ f(u)(cα e^{-α(t-u)} + cβ e^{-β(t-u)}) du` with `f` the
    /// `Gamma(n+1, KTR)` transit density — fine quadrature, *non-integer* `n` (so this
    /// is independent of the `n=0 ↔ two_cpt_oral` identity).
    #[test]
    fn convolve_2cpt_matches_numerical_convolution() {
        let amt = 100.0;
        for &(cl, v1, q, v2, n, mtt) in &[
            (5.0, 20.0, 10.0, 40.0, 2.5, 1.0),
            (3.0, 30.0, 6.0, 60.0, 1.3, 0.7),
            (8.0, 50.0, 20.0, 30.0, 4.0, 2.0),
        ] {
            let (alpha, beta, k21) = macro_rates_g::<f64>(cl, v1, q, v2);
            let diff = alpha - beta;
            let c_alpha = (alpha - k21) / diff;
            let c_beta = (k21 - beta) / diff;
            let ktr = (n + 1.0) / mtt;
            let ln_norm = (n + 1.0) * ktr.ln() - ln_gamma(n + 1.0);
            let dens = |u: f64| (ln_norm + n * u.ln() - ktr * u).exp();
            for &t in &[0.5, 1.0, 2.0, 4.0, 8.0] {
                let steps = 200_000usize;
                let h = t / steps as f64;
                let mut acc = 0.0;
                for i in 0..=steps {
                    let u = i as f64 * h;
                    let kernel =
                        c_alpha * (-alpha * (t - u)).exp() + c_beta * (-beta * (t - u)).exp();
                    let integrand = if u == 0.0 { 0.0 } else { dens(u) * kernel };
                    let w = if i == 0 || i == steps { 0.5 } else { 1.0 };
                    acc += w * integrand;
                }
                let numeric = (amt / v1) * acc * h;
                assert_relative_eq!(
                    tc(cl, v1, q, v2, n, mtt, t, amt),
                    numeric,
                    max_relative = 1e-4,
                    epsilon = 1e-9
                );
            }
        }
    }

    /// Exact `Dual2` `∂C/∂{cl,v1,q,v2,n,mtt}` (the FOCE/FOCEI gradients) vs a central
    /// difference of the `f64` value, plus the six diagonal 2nd derivatives vs a
    /// central difference of the exact dual 1st-derivative (the two-rung ladder that
    /// avoids the `1/h²` value-difference blow-up).
    #[test]
    fn convolve_2cpt_dual_gradients_match_fd() {
        let amt = 100.0;
        let p = [5.0, 20.0, 10.0, 40.0, 2.5, 1.0]; // cl,v1,q,v2,n,mtt
        for &t in &[1.0, 3.0, 6.0] {
            let d = tc_dual(p[0], p[1], p[2], p[3], p[4], p[5], t, amt);
            let h = 1e-6;
            for dim in 0..6 {
                let mut pp = p;
                let mut pm = p;
                pp[dim] += h;
                pm[dim] -= h;
                let fd = (tc(pp[0], pp[1], pp[2], pp[3], pp[4], pp[5], t, amt)
                    - tc(pm[0], pm[1], pm[2], pm[3], pm[4], pm[5], t, amt))
                    / (2.0 * h);
                assert_relative_eq!(d.grad[dim], fd, max_relative = 1e-4, epsilon = 1e-7);
                // 2nd-order diagonal vs central difference of the exact dual grad.
                let h2 = 1e-4;
                let mut qp = p;
                let mut qm = p;
                qp[dim] += h2;
                qm[dim] -= h2;
                let gp = tc_dual(qp[0], qp[1], qp[2], qp[3], qp[4], qp[5], t, amt).grad[dim];
                let gm = tc_dual(qm[0], qm[1], qm[2], qm[3], qm[4], qm[5], t, amt).grad[dim];
                assert_relative_eq!(
                    d.hess[dim][dim],
                    (gp - gm) / (2.0 * h2),
                    max_relative = 2e-4,
                    epsilon = 1e-6
                );
            }
        }
    }

    /// The peripheral concentration `C2 = A2/V2` must equal its defining convolution
    /// `(F·Dose/V2)(k12/(α−β)) ∫₀ᵗ f(u)(e^{-β(t-u)} − e^{-α(t-u)}) du` (the peripheral
    /// IV-bolus impulse response convolved with the transit density), `k12 = Q/V1`.
    #[test]
    fn convolve_2cpt_peripheral_matches_numerical() {
        let amt = 100.0;
        for &(cl, v1, q, v2, n, mtt) in &[
            (5.0, 20.0, 10.0, 40.0, 2.5, 1.0),
            (3.0, 30.0, 6.0, 60.0, 1.3, 0.7),
        ] {
            let (alpha, beta, _k21) = macro_rates_g::<f64>(cl, v1, q, v2);
            let k12 = q / v1;
            let abs = TransitAbsorption { n, mtt };
            let ktr = (n + 1.0) / mtt;
            let ln_norm = (n + 1.0) * ktr.ln() - ln_gamma(n + 1.0);
            let dens = |u: f64| (ln_norm + n * u.ln() - ktr * u).exp();
            for &t in &[0.5, 1.0, 2.0, 4.0, 8.0] {
                let steps = 200_000usize;
                let h = t / steps as f64;
                let mut acc = 0.0;
                for i in 0..=steps {
                    let u = i as f64 * h;
                    let kernel = (-beta * (t - u)).exp() - (-alpha * (t - u)).exp();
                    let integrand = if u == 0.0 { 0.0 } else { dens(u) * kernel };
                    let w = if i == 0 || i == steps { 0.5 } else { 1.0 };
                    acc += w * integrand;
                }
                let numeric = (amt / v2) * (k12 / (alpha - beta)) * acc * h;
                let got = convolve_2cpt_peripheral(&abs, t, alpha, beta, k12, amt / v2);
                assert_relative_eq!(got, numeric, max_relative = 1e-4, epsilon = 1e-9);
            }
        }
    }
}
