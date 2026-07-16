//! Dense linear solve over the sensitivity numeric type ([`PkNum`]).
//!
//! A single partial-pivoted Gaussian elimination, generic over `f64` (value-only) and the
//! dual types (`Dual1`/`Dual2`). Because every arithmetic step is a `PkNum` `+ − × ÷`, the
//! derivative jets ride through the elimination automatically: solving `A·x = b` over a dual
//! `A`, `b` yields the exact implicit-function derivative
//! `∂x/∂p = A⁻¹·(∂b/∂p − ∂A/∂p·x)` — and its 2nd order for the FOCEI outer Hessian — with no
//! hand-assembled `dA`/`db`. This is what carries `∂u_ss/∂(θ,η)` through the steady-state
//! absorption fixed point `u_ss = (I − M)⁻¹·b`: production instantiates it at `T = f64`
//! ([`crate::ode::predictions`]) and the analytic-sensitivity walk at `T = Dual1/Dual2`
//! ([`crate::sens::ode_provider`]) — one body, one source of truth (#835).

use crate::sens::num::PkNum;

/// Solve the dense `n×n` system `A·x = b` (row-major `a`, length-`n` `b`) by partial-pivoted
/// Gaussian elimination, returning `x` (length `n`) or `None` when the value-part matrix is
/// singular to a scaled pivot tolerance.
///
/// Pivoting branches on the **value part** ([`PkNum::val`]) only, so the elimination sequence
/// is identical to the `f64` solve of `A.val()`; the dual jets then propagate through the same
/// `× ÷ −` operations, giving the exact derivative (and 2nd order) of the solution. On a
/// singular value-part matrix the caller falls back to the iterative equilibration, mirroring
/// how the `f64` fixed point declines a non-invertible `I − M`.
pub(crate) fn solve_linear_system_g<T: PkNum>(a: &[T], b: &[T], n: usize) -> Option<Vec<T>> {
    debug_assert_eq!(a.len(), n * n, "coefficient matrix must be n×n row-major");
    debug_assert_eq!(b.len(), n, "right-hand side must have length n");
    if n == 0 {
        return Some(Vec::new());
    }
    let mut m = a.to_vec();
    let mut rhs = b.to_vec();
    // Singular tolerance relative to the largest value-part magnitude, floored at 1.0 so a
    // uniformly small-entry system is not spuriously rejected.
    let scale = a.iter().fold(0.0_f64, |s, x| s.max(x.val().abs())).max(1.0);
    let tol = 1e-12 * scale;
    for col in 0..n {
        // Partial pivot: largest value-part magnitude in this column, at or below the diagonal.
        let mut piv = col;
        let mut best = m[col * n + col].val().abs();
        for r in (col + 1)..n {
            let v = m[r * n + col].val().abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best <= tol {
            return None;
        }
        if piv != col {
            for c in 0..n {
                m.swap(col * n + c, piv * n + c);
            }
            rhs.swap(col, piv);
        }
        let pivot = m[col * n + col];
        for r in (col + 1)..n {
            let factor = m[r * n + col] / pivot;
            for c in col..n {
                let sub = factor * m[col * n + c];
                m[r * n + c] = m[r * n + c] - sub;
            }
            let sub = factor * rhs[col];
            rhs[r] = rhs[r] - sub;
        }
    }
    // Back-substitution.
    let mut x = vec![T::from_f64(0.0); n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for c in (i + 1)..n {
            s = s - m[i * n + c] * x[c];
        }
        x[i] = s / m[i * n + i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sens::dual2::Dual2;

    /// A 2×2 system whose entries mix both parameters (so `x(p)` is genuinely nonlinear in
    /// `p` and the Hessian is nonzero), built generically so the same source runs at `f64`
    /// (finite-difference reference) and `Dual2` (analytic grad + Hessian under test).
    fn build<T: PkNum>(p0: T, p1: T) -> ([T; 4], [T; 2]) {
        let a = [
            T::from_f64(2.0) + p0,
            T::from_f64(1.0),
            p1 * p1,
            T::from_f64(3.0) + p0 * p1,
        ];
        let b = [p0, T::from_f64(1.0) + p1];
        (a, b)
    }

    fn solve_val(p0: f64, p1: f64, comp: usize) -> f64 {
        let (a, b) = build::<f64>(p0, p1);
        solve_linear_system_g::<f64>(&a, &b, 2).expect("nonsingular")[comp]
    }

    #[test]
    fn dual_solve_grad_and_hessian_match_fd() {
        let (p0, p1) = (1.3, 0.7);
        let (a, b) = build::<Dual2<2>>(Dual2::<2>::var(p0, 0), Dual2::<2>::var(p1, 1));
        let x = solve_linear_system_g::<Dual2<2>>(&a, &b, 2).expect("nonsingular");

        for comp in 0..2 {
            approx::assert_relative_eq!(
                x[comp].value,
                solve_val(p0, p1, comp),
                max_relative = 1e-12
            );
        }

        // First order vs central FD of the f64 solve.
        let hg = 1e-6;
        for comp in 0..2 {
            let g0 = (solve_val(p0 + hg, p1, comp) - solve_val(p0 - hg, p1, comp)) / (2.0 * hg);
            let g1 = (solve_val(p0, p1 + hg, comp) - solve_val(p0, p1 - hg, comp)) / (2.0 * hg);
            approx::assert_relative_eq!(x[comp].grad[0], g0, max_relative = 1e-6, epsilon = 1e-8);
            approx::assert_relative_eq!(x[comp].grad[1], g1, max_relative = 1e-6, epsilon = 1e-8);
        }

        // Second order vs FD (3-point diagonal, 4-point mixed) — the FOCEI Hessian path.
        let hh = 1e-4;
        for comp in 0..2 {
            let f = |q0: f64, q1: f64| solve_val(q0, q1, comp);
            let hxx = (f(p0 + hh, p1) - 2.0 * f(p0, p1) + f(p0 - hh, p1)) / (hh * hh);
            let hyy = (f(p0, p1 + hh) - 2.0 * f(p0, p1) + f(p0, p1 - hh)) / (hh * hh);
            let hxy = (f(p0 + hh, p1 + hh) - f(p0 + hh, p1 - hh) - f(p0 - hh, p1 + hh)
                + f(p0 - hh, p1 - hh))
                / (4.0 * hh * hh);
            approx::assert_relative_eq!(
                x[comp].hess[0][0],
                hxx,
                max_relative = 1e-4,
                epsilon = 1e-5
            );
            approx::assert_relative_eq!(
                x[comp].hess[1][1],
                hyy,
                max_relative = 1e-4,
                epsilon = 1e-5
            );
            approx::assert_relative_eq!(
                x[comp].hess[0][1],
                hxy,
                max_relative = 1e-4,
                epsilon = 1e-5
            );
            // Symmetry of the returned Hessian.
            approx::assert_relative_eq!(
                x[comp].hess[0][1],
                x[comp].hess[1][0],
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn singular_matrix_returns_none() {
        // Rank-deficient: row 2 = 2 × row 1.
        let a = [1.0, 2.0, 2.0, 4.0];
        let b = [1.0, 2.0];
        assert!(solve_linear_system_g::<f64>(&a, &b, 2).is_none());
    }

    #[test]
    fn pivoting_handles_zero_leading_diagonal() {
        // A zero (0,0) pivot forces a row swap; the solve must still be exact.
        // [[0,1],[1,0]]·x = [2,3]  ⇒  x = [3,2].
        let a = [0.0, 1.0, 1.0, 0.0];
        let b = [2.0, 3.0];
        let x = solve_linear_system_g::<f64>(&a, &b, 2).expect("nonsingular after pivot");
        approx::assert_relative_eq!(x[0], 3.0, max_relative = 1e-12);
        approx::assert_relative_eq!(x[1], 2.0, max_relative = 1e-12);
    }

    #[test]
    fn identity_solve_returns_rhs() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [3.5, -2.0];
        let x = solve_linear_system_g::<f64>(&a, &b, 2).expect("nonsingular");
        assert_eq!(x, vec![3.5, -2.0]);
    }
}
