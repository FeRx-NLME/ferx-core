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
    let mut piv = vec![0usize; n];
    if !lu_factor_in_place_g(&mut m, &mut piv, n) {
        return None;
    }
    let mut x = b.to_vec();
    lu_solve_in_place_g(&m, &piv, n, &mut x);
    Some(x)
}

/// Partial-pivoted LU factorization of the row-major `n×n` matrix `m`, **in place**: the
/// unit-lower multipliers land in the strict lower triangle and `U` in the upper, with
/// `piv[col]` recording the row swapped into `col`. Returns `false` (leaving `m` partially
/// eliminated) when the value-part matrix is singular to the same scaled pivot tolerance
/// [`solve_linear_system_g`] uses.
///
/// Split out from [`solve_linear_system_g`] so a caller with **many right-hand sides against
/// one matrix** pays the `O(n³)` elimination once and only `O(n²)` per solve: the Rosenbrock
/// steppers ([`crate::ode::rosenbrock`]) factor `W = I/(γh) − J` once per step attempt and
/// back-solve it once per stage, which is the whole cost argument for a linearly-implicit
/// method over a Newton-iterating one.
///
/// Pivoting branches on [`PkNum::val`] only — see the module note; a dual-part comparison
/// would make the elimination order differ between the `f64` and `Dual2` runs and silently
/// desynchronise the sensitivities from the prediction.
// `col` indexes `m`, `piv` and the elimination bounds together; iterating `piv` would hide
// that. The `!(best > tol)` guard is deliberately a negated comparison, so NaN is rejected.
#[allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]
pub(crate) fn lu_factor_in_place_g<T: PkNum>(m: &mut [T], piv: &mut [usize], n: usize) -> bool {
    debug_assert_eq!(m.len(), n * n, "coefficient matrix must be n×n row-major");
    debug_assert_eq!(piv.len(), n, "pivot vector must have length n");
    // Singular tolerance relative to the largest value-part magnitude, floored at 1.0 so a
    // uniformly small-entry system is not spuriously rejected.
    let scale = m.iter().fold(0.0_f64, |s, x| s.max(x.val().abs())).max(1.0);
    let tol = 1e-12 * scale;
    for col in 0..n {
        // Partial pivot: largest value-part magnitude in this column, at or below the diagonal.
        let mut p = col;
        let mut best = m[col * n + col].val().abs();
        for r in (col + 1)..n {
            let v = m[r * n + col].val().abs();
            if v > best {
                best = v;
                p = r;
            }
        }
        // `!(best > tol)` rather than `best <= tol` so a NaN pivot (a diverged Jacobian)
        // is rejected instead of sliding through the comparison.
        if !(best > tol) {
            return false;
        }
        piv[col] = p;
        if p != col {
            for c in 0..n {
                m.swap(col * n + c, p * n + c);
            }
        }
        let pivot = m[col * n + col];
        for r in (col + 1)..n {
            let factor = m[r * n + col] / pivot;
            m[r * n + col] = factor;
            for c in (col + 1)..n {
                let sub = factor * m[col * n + c];
                m[r * n + c] = m[r * n + c] - sub;
            }
        }
    }
    true
}

/// Solve `A·x = b` in place given the [`lu_factor_in_place_g`] factorization of `A`: `b` is
/// overwritten with `x`. Applies the recorded row swaps, forward-substitutes through the
/// unit-lower multipliers, then back-substitutes through `U`.
///
/// The permutation is applied as a **complete pass before** any elimination (LAPACK's
/// `dgetrs` order), not interleaved column by column. `lu_factor_in_place_g` swaps whole
/// rows — the stored multipliers of earlier columns move with them — so `L` is expressed in
/// the fully permuted row order and `b` has to be brought into that order first. Interleaving
/// silently produces a wrong answer as soon as a swap happens at a column other than the
/// first (it is invisible at `n = 2`, and on the Rosenbrock `W = I/(γh) − J`, which is
/// diagonally dominant at small `h` and rarely pivots at all).
// `col` walks the permutation, the multiplier column and the right-hand side together.
#[allow(clippy::needless_range_loop)]
pub(crate) fn lu_solve_in_place_g<T: PkNum>(m: &[T], piv: &[usize], n: usize, b: &mut [T]) {
    debug_assert_eq!(m.len(), n * n, "factorization must be n×n row-major");
    debug_assert_eq!(b.len(), n, "right-hand side must have length n");
    for col in 0..n {
        let p = piv[col];
        if p != col {
            b.swap(col, p);
        }
    }
    for col in 0..n {
        for r in (col + 1)..n {
            let sub = m[r * n + col] * b[col];
            b[r] = b[r] - sub;
        }
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for c in (i + 1)..n {
            s = s - m[i * n + c] * b[c];
        }
        b[i] = s / m[i * n + i];
    }
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

    /// One factorization reused across several right-hand sides must be **bit-identical** to
    /// re-eliminating per right-hand side — that equality is what lets the Rosenbrock
    /// steppers pay for the `O(n³)` elimination once per step and back-solve it per stage.
    ///
    /// The matrix is chosen to pivot **twice**: a zero leading entry swaps at column 0 and
    /// the growth in column 1 swaps again at column 1. That second swap is the case an
    /// interleaved (rather than up-front) permutation gets wrong, and it cannot be caught at
    /// `n = 2` — hence the residual check against `A` itself rather than only cross-checking
    /// the two entry points, which share an implementation.
    #[test]
    fn lu_reuse_matches_per_rhs_elimination() {
        let a = [0.0, 2.0, 1.0, 3.0, -1.0, 4.0, 2.0, 5.0, -3.0];
        let rhs_set = [[1.0, 2.0, 3.0], [-4.0, 0.5, 7.0], [0.0, 0.0, 1.0]];

        let mut lu = a.to_vec();
        let mut piv = vec![0usize; 3];
        assert!(lu_factor_in_place_g::<f64>(&mut lu, &mut piv, 3));

        for b in rhs_set {
            let mut x = b.to_vec();
            lu_solve_in_place_g::<f64>(&lu, &piv, 3, &mut x);
            let reference = solve_linear_system_g::<f64>(&a, &b, 3).expect("nonsingular");
            assert_eq!(x, reference);
            // …and it really solves the system.
            for r in 0..3 {
                let ax: f64 = (0..3).map(|c| a[r * 3 + c] * x[c]).sum();
                approx::assert_relative_eq!(ax, b[r], epsilon = 1e-12);
            }
        }
    }

    /// A NaN entry (a diverged Jacobian reaching the Rosenbrock `W`) must be rejected by the
    /// pivot test rather than sliding through the comparison and producing NaN "solutions".
    #[test]
    fn non_finite_matrix_is_rejected() {
        let a = [f64::NAN, 1.0, 1.0, 1.0];
        let mut m = a.to_vec();
        let mut piv = vec![0usize; 2];
        assert!(!lu_factor_in_place_g::<f64>(&mut m, &mut piv, 2));
        assert!(solve_linear_system_g::<f64>(&a, &[1.0, 1.0], 2).is_none());
    }
}
