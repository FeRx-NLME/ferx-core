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

/// A value that is exactly 0 and carries no derivative at all — no jet, no tape node
/// ([`PkNum::is_constant`]). Its products contribute exactly 0 to every component, so the
/// dense kernels here and in `ode::linear` skip them without changing a bit (up to the sign
/// of a zero). A value that is merely *numerically* 0 (`θ − θ₀` at `θ = θ₀`) has a
/// derivative and is not skipped; a type that cannot tell cheaply reports non-constant, so
/// nothing is skipped for it.
#[inline]
pub(crate) fn structural_zero<T: PkNum>(x: T) -> bool {
    x.val() == 0.0 && x.is_constant()
}

/// Value and every derivative component finite. A structural zero is skipped only against a
/// finite factor: `0·inf` and `0·NaN` are `NaN` in IEEE arithmetic, and dropping them would
/// turn a diverged solve that used to propagate `NaN` (and be rejected) into a finite one.
/// Against a finite factor the skipped product is exactly `±0`.
#[inline]
pub(crate) fn all_finite<T: PkNum>(x: T) -> bool {
    x.val().is_finite() && x.jets_finite()
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
    // The same body at a literal `n` for the small systems an ODE stepper factors every
    // attempt, so the compiler unrolls it: the operations, and their order, are unchanged.
    match n {
        2 => lu_factor_body(m, piv, 2),
        3 => lu_factor_body(m, piv, 3),
        4 => lu_factor_body(m, piv, 4),
        _ => lu_factor_body(m, piv, n),
    }
}

#[inline(always)]
fn lu_factor_body<T: PkNum>(m: &mut [T], piv: &mut [usize], n: usize) -> bool {
    debug_assert_eq!(m.len(), n * n, "coefficient matrix must be n×n row-major");
    debug_assert_eq!(piv.len(), n, "pivot vector must have length n");
    if let Some(ok) = T::lu_factor_hook(m, piv, n) {
        return ok;
    }
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
        // The pivot row to the right of the pivot, which every update below multiplies.
        let pivot_row_finite = ((col + 1)..n).all(|c| all_finite(m[col * n + c]));
        for r in (col + 1)..n {
            // A structural zero below the pivot (value 0, no jet, no tape) has multiplier
            // exactly 0 in every component, so its row update adds `±0`: skipped, against a
            // finite pivot row (see `all_finite`). The multiplier itself is still stored, so
            // its sign is the one the update would have written.
            if pivot_row_finite && structural_zero(m[r * n + col]) {
                m[r * n + col] = m[r * n + col] / pivot;
                continue;
            }
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
    // As in `lu_factor_in_place_g`: the same body, unrolled for the small `n` a stepper solves
    // once per stage.
    match n {
        2 => lu_solve_body(m, piv, 2, b),
        3 => lu_solve_body(m, piv, 3, b),
        4 => lu_solve_body(m, piv, 4, b),
        _ => lu_solve_body(m, piv, n, b),
    }
}

#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn lu_solve_body<T: PkNum>(m: &[T], piv: &[usize], n: usize, b: &mut [T]) {
    debug_assert_eq!(m.len(), n * n, "factorization must be n×n row-major");
    debug_assert_eq!(b.len(), n, "right-hand side must have length n");
    if T::lu_solve_hook(m, piv, n, b) {
        return;
    }
    for col in 0..n {
        let p = piv[col];
        if p != col {
            b.swap(col, p);
        }
    }
    for col in 0..n {
        // `b[col]` a structural zero multiplies the whole multiplier column below it.
        if structural_zero(b[col]) && ((col + 1)..n).all(|r| all_finite(m[r * n + col])) {
            continue;
        }
        let b_finite = all_finite(b[col]);
        for r in (col + 1)..n {
            if b_finite && structural_zero(m[r * n + col]) {
                continue;
            }
            let sub = m[r * n + col] * b[col];
            b[r] = b[r] - sub;
        }
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for c in (i + 1)..n {
            if structural_zero(m[i * n + c]) && all_finite(b[c]) {
                continue;
            }
            s = s - m[i * n + c] * b[c];
        }
        b[i] = s / m[i * n + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sens::dual2::Dual2;

    /// The pre-skip factor and solve: every update taken, structural zeros included.
    fn lu_reference<T: PkNum>(m: &[T], n: usize, b: &[T]) -> Option<Vec<T>> {
        let mut m = m.to_vec();
        let mut piv = vec![0usize; n];
        let scale = m.iter().fold(0.0_f64, |s, x| s.max(x.val().abs())).max(1.0);
        for col in 0..n {
            let mut p = col;
            let mut best = m[col * n + col].val().abs();
            for r in (col + 1)..n {
                if m[r * n + col].val().abs() > best {
                    best = m[r * n + col].val().abs();
                    p = r;
                }
            }
            if !(best > 1e-12 * scale) {
                return None;
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
        let mut x = b.to_vec();
        for col in 0..n {
            x.swap(col, piv[col]);
        }
        for col in 0..n {
            for r in (col + 1)..n {
                let sub = m[r * n + col] * x[col];
                x[r] = x[r] - sub;
            }
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for c in (i + 1)..n {
                s = s - m[i * n + c] * x[c];
            }
            x[i] = s / m[i * n + i];
        }
        Some(x)
    }

    /// The structural-zero skips in the factor and both substitutions are exact on a sparse
    /// system — bit-identical to taking every update (`±0` identified) — and a zero that
    /// carries a derivative is still used, in `L` and in `U`. The system forces a row swap
    /// (column 0's largest entry is below the diagonal) and has non-zero upper entries, so the
    /// back substitution sees both kinds of zero. Mutations: drop `is_constant` from
    /// `structural_zero` → the jet-carrying zeros are skipped and the `DualF` solve differs;
    /// skip every back-substitution term → the non-zero `U` entries are lost.
    #[test]
    fn lu_skips_structural_zeros_exactly_and_keeps_a_jet_carrying_zero() {
        use crate::sens::vfree::DualF;
        let n = 4;
        let c = |v: f64| DualF::<2, 0>::from_f64(v);
        let d = |v: f64, g0: f64, g1: f64| DualF::<2, 0> {
            v,
            g: [g0, g1],
            h: [],
        };
        let mut m = vec![c(0.0); n * n];
        m[0] = d(1.0, 1.0, 0.2); // pivot-swap: row 1 has the larger column-0 entry
        m[n] = d(3.0, 0.0, 0.5);
        m[n + 1] = d(2.5, 0.3, 0.0);
        m[1] = d(0.7, 0.0, 1.0); // non-zero U entry
        m[2] = d(0.0, 2.0, 0.0); // jet-carrying zero in U
        m[2 * n + 2] = d(4.0, 1.0, 0.0);
        m[2 * n + 1] = d(0.0, 0.0, 3.0); // jet-carrying zero in L
        m[n + 3] = c(0.25); // non-zero U entry, constant
        m[3 * n + 3] = d(5.0, 0.0, 0.4);
        m[3 * n + 2] = c(0.5);
        let b = vec![d(1.0, 0.0, 0.0), c(0.0), d(0.0, 0.5, 0.0), c(2.0)];
        let mut got = b.clone();
        let mut mm = m.clone();
        let mut piv = vec![0usize; n];
        assert!(lu_factor_in_place_g(&mut mm, &mut piv, n));
        assert_eq!(piv[0], 1, "premise: column 0 pivots on row 1");
        lu_solve_in_place_g(&mm, &piv, n, &mut got);
        let want = lu_reference(&m, n, &b).unwrap();
        let norm = |x: f64| if x == 0.0 { 0 } else { x.to_bits() };
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(norm(g.v), norm(w.v));
            for k in 0..2 {
                assert_eq!(norm(g.g[k]), norm(w.g[k]));
            }
        }
    }

    /// A skipped row still stores its multiplier, so the factorization keeps the dense
    /// code's sign: `+0` under a negative pivot is stored as `0 / −2 = −0`, exactly as the
    /// row update would have written it. Mutation: skip without storing → `+0` stays.
    #[test]
    fn a_skipped_lu_row_keeps_the_dense_multiplier_sign() {
        let n = 2;
        let mut m = vec![-2.0, 1.0, 0.0, 3.0];
        let mut piv = vec![0usize; n];
        assert!(lu_factor_in_place_g(&mut m, &mut piv, n));
        assert_eq!(piv[0], 0, "premise: no swap");
        let want = 0.0_f64 / -2.0;
        assert!(
            want.is_sign_negative(),
            "premise: the dense multiplier is −0"
        );
        assert_eq!(m[2].to_bits(), want.to_bits(), "multiplier {}", m[2]);
    }

    /// A structural zero is skipped only against a finite factor. With an infinite multiplier
    /// column and a structurally-zero right-hand side entry, the dense solve propagates
    /// `0·inf = NaN`; skipping the zero would hand back a finite answer to a diverged system.
    /// Mutation: drop the `all_finite` guards → the solution is finite and this fails.
    #[test]
    fn lu_does_not_skip_a_zero_against_an_infinite_factor() {
        let n = 2;
        let m = vec![1.0, 0.0, f64::INFINITY, 1.0];
        let mut piv = vec![0usize; n];
        let mut mm = m.clone();
        // Pivoting would move the inf row up; bypass the factor and solve the unit-lower
        // system directly, as a Rosenbrock stage solve with an overflowed entry does.
        piv[0] = 0;
        piv[1] = 1;
        mm[2] = f64::INFINITY;
        let mut b = vec![0.0, 1.0];
        lu_solve_in_place_g(&mm, &piv, n, &mut b);
        let want = lu_reference_solve_only(&mm, &piv, n, &[0.0, 1.0]);
        assert!(
            want.iter().any(|v| v.is_nan()),
            "premise: the dense solve is NaN"
        );
        assert!(b.iter().any(|v| v.is_nan()), "the skip hid a NaN: {b:?}");
    }

    /// The factor's skip is guarded by the pivot row. An `inf` there can only arise mid-
    /// elimination (any `inf` input makes the singularity tolerance infinite and the factor
    /// rejects up front), so: finite entries near `f64::MAX` whose column-0 update overflows
    /// row 1's `U` entry to `inf`, with a structural zero below it in column 1. The dense
    /// update writes `1e300 − 0·inf = NaN` there. Mutation: drop `pivot_row_finite` → the
    /// row is skipped and stays `1e300`.
    #[test]
    fn lu_factor_does_not_skip_a_zero_against_an_overflowed_pivot_row() {
        let n = 3;
        let mut m = vec![
            1e300, 0.0, 1.5e308, //
            -1e300, 1e300, 1.5e308, //
            0.0, 0.0, 1e300,
        ];
        let mut piv = vec![0usize; n];
        // Column 2's pivot is then NaN, so the factor rejects; what it wrote must match dense.
        let _ = lu_factor_in_place_g(&mut m, &mut piv, n);
        assert_eq!(piv[..2], [0, 1], "premise: no swaps");
        assert!(
            m[5].is_infinite(),
            "premise: row 1's U entry overflowed: {m:?}"
        );
        assert!(m[8].is_nan(), "the skip hid 1e300 − 0·inf: {m:?}");
    }

    /// The forward substitution's per-entry skip is guarded by the solved entry: a
    /// structural-zero multiplier below an `inf` right-hand side is `0·inf = NaN` in the
    /// dense update. Mutation: drop `b_finite` → the update is skipped and `b[1]` stays 1.
    #[test]
    fn lu_forward_substitution_does_not_skip_a_zero_against_an_infinite_entry() {
        let n = 2;
        let m = vec![1.0, 0.0, 0.0, 1.0];
        let piv = vec![0usize, 1];
        let mut b = vec![f64::INFINITY, 1.0];
        lu_solve_in_place_g(&m, &piv, n, &mut b);
        let want = lu_reference_solve_only(&m, &piv, n, &[f64::INFINITY, 1.0]);
        assert!(want[1].is_nan(), "premise: dense {want:?}");
        assert!(b[1].is_nan(), "the skip hid 0·inf: {b:?}");
    }

    /// The back substitution's skip is guarded by the solved entry: a structural zero in
    /// `U` against an `inf` component of the solution is `NaN` in the dense sum. Mutation:
    /// drop `all_finite(b[c])` → the term is skipped and `x[0]` comes out finite.
    #[test]
    fn lu_back_substitution_does_not_skip_a_zero_against_an_infinite_entry() {
        let n = 2;
        let m = vec![1.0, 0.0, 0.0, 1e-300];
        let piv = vec![0usize, 1];
        let mut b = vec![1.0, 1e10];
        lu_solve_in_place_g(&m, &piv, n, &mut b);
        let want = lu_reference_solve_only(&m, &piv, n, &[1.0, 1e10]);
        assert!(
            want[1].is_infinite() && want[0].is_nan(),
            "premise: dense {want:?}"
        );
        assert!(b[0].is_nan(), "the skip hid 0·inf: {b:?}");
    }

    /// The pre-skip substitutions alone, on a given factorization.
    fn lu_reference_solve_only(m: &[f64], piv: &[usize], n: usize, b: &[f64]) -> Vec<f64> {
        let mut x = b.to_vec();
        for col in 0..n {
            x.swap(col, piv[col]);
        }
        for col in 0..n {
            for r in (col + 1)..n {
                x[r] -= m[r * n + col] * x[col];
            }
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for c in (i + 1)..n {
                s -= m[i * n + c] * x[c];
            }
            x[i] = s / m[i * n + i];
        }
        x
    }

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
