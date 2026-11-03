//! Exact η-derivatives of a scalar function written once over any VM number type (#2437).
//!
//! A non-Gaussian data term — a TTE hazard likelihood, a binary logit term, a CTMM
//! transition chain — is an [`EtaFunction`]: one generic `eval::<T>` that the value path
//! runs at `T = f64` and that [`eta_hessian`] runs on the adjoint's tape-free η-jets
//! (`vfree::DualF<N, N(N+1)/2>`, θ constant, η free), at the same width buckets the recorders use.
//! The derivatives are those of the function the value path evaluates, by construction:
//! there is no hand-written chain rule, closed-form curvature or second Fréchet derivative
//! to drift from it, and no finite-difference step to choose.
//!
//! The one way left to break that is a branch on [`PkNum::val`] that switches *formula*:
//! both arms agree in value, but the jets follow the arm taken, so a limit written as its
//! value alone (`γ = 0 ⇒ H = α·t`) drops every derivative the other arm carries. A `.val()`
//! branch in a function reached from here may only guard (a sentinel, a domain check);
//! a limit is written in a form that keeps its derivatives (a series), as
//! `survival::parametric::hazard_and_cum_hazard_g` does for Gompertz.

use super::num::vm::VmNum;
use super::num::PkNum;
use super::ode_provider::adjoint::{by_bucket, WIDTH_BUCKETS};
use super::vfree::DualF;
use super::vtape::tri;
use nalgebra::DMatrix;

/// A scalar function of η, written once over any VM number type.
pub(crate) trait EtaFunction {
    /// Its value at `eta` over `T`, or `None` where it has no derivative to give: the
    /// value path's degeneracy sentinel, a declined program, an unrepresentable endpoint.
    fn eval<T: VmNum>(&self, eta: &[T]) -> Option<T>;
}

/// [`EtaFunction::eval`]'s value at `eta` and its exact η-Hessian. `None` when the function
/// declines, any jet is non-finite, or `eta` is wider than the widest dispatch bucket — the
/// caller then finite-differences, as for a subject outside the recorders' scope.
pub(crate) fn eta_hessian<F: EtaFunction>(f: &F, eta: &[f64]) -> Option<(f64, DMatrix<f64>)> {
    let bucket = WIDTH_BUCKETS.into_iter().find(|&b| b >= eta.len().max(1))?;
    by_bucket!(bucket, run, (f, eta))
}

fn run<const N: usize, const M: usize>(
    f: &impl EtaFunction,
    eta: &[f64],
) -> Option<(f64, DMatrix<f64>)> {
    let n = eta.len();
    let x: Vec<DualF<N, M>> = eta
        .iter()
        .enumerate()
        .map(|(k, &v)| {
            let mut g = [0.0; N];
            g[k] = 1.0;
            DualF::free(v, g, [0.0; M])
        })
        .collect();
    let y = f.eval(&x)?;
    if !y.v.is_finite() || !y.jets_finite() {
        return None;
    }
    Some((y.v, DMatrix::from_fn(n, n, |k, l| y.h[tri(k, l)])))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f(η) = exp(η₀·η₁) + η_{n−1}²`, any width: a cross term and a pure square.
    struct Probe;
    impl EtaFunction for Probe {
        fn eval<T: VmNum>(&self, eta: &[T]) -> Option<T> {
            let last = *eta.last()?;
            let cross = if eta.len() > 1 {
                (eta[0] * eta[1]).exp()
            } else {
                T::from_f64(0.0)
            };
            Some(cross + last * last)
        }
    }

    /// A function that declines: [`eta_hessian`] must pass the `None` through.
    struct Declines;
    impl EtaFunction for Declines {
        fn eval<T: VmNum>(&self, _eta: &[T]) -> Option<T> {
            None
        }
    }

    #[test]
    fn hessian_is_exact_at_every_bucket_and_padded_width() {
        // Every bucket, and a width between buckets (5 pads to 6, 47 to 48), so a wrong
        // seeding of the padded axes or a wrong packed index shows at some width.
        for n in [1usize, 2, 3, 5, 8, 13, 24, 47, 48] {
            let eta: Vec<f64> = (0..n).map(|k| 0.3 - 0.1 * k as f64).collect();
            let (v, h) = eta_hessian(&Probe, &eta).expect("within the widest bucket");
            let want_v = Probe.eval(&eta).unwrap();
            assert_eq!(
                v.to_bits(),
                want_v.to_bits(),
                "n = {n}: value is the f64 path's"
            );
            let mut want = DMatrix::<f64>::zeros(n, n);
            if n > 1 {
                let (a, b) = (eta[0], eta[1]);
                let e = (a * b).exp();
                want[(0, 0)] = b * b * e;
                want[(1, 1)] = a * a * e;
                want[(0, 1)] = (1.0 + a * b) * e;
                want[(1, 0)] = want[(0, 1)];
            }
            want[(n - 1, n - 1)] += 2.0;
            let err = (&h - &want).abs().max();
            assert!(
                err.is_finite() && err < 1e-14,
                "n = {n}: |H − exact| = {err:e}"
            );
        }
    }

    #[test]
    fn declines_past_the_widest_bucket_or_when_the_function_does() {
        let wide = vec![0.1; WIDTH_BUCKETS[WIDTH_BUCKETS.len() - 1] + 1];
        assert!(eta_hessian(&Probe, &wide).is_none());
        assert!(eta_hessian(&Declines, &[0.1, 0.2]).is_none());
    }
}
