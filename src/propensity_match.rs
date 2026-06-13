//! Propensity-score matching of simulated random effects to fitted (posthoc)
//! random effects, for VPCs on adaptively-dosed real-world data.
//!
//! Background: in real-world data, therapy is often adapted in response to a
//! patient's own PK (e.g. longer dosing intervals for high-clearance patients).
//! A standard VPC draws each subject's `eta` independently of its design, so the
//! design↔eta association present in the observed data is lost and the VPC shows
//! spurious misspecification. Propensity-score matching restores that
//! association: per replicate, freshly drawn etas are reassigned to subjects so
//! that each subject's *design* is paired with a drawn eta close (in eta-space)
//! to that subject's *fitted* eta. See the PAGE poster "Visual Predictive Checks
//! for Real-World Data using Propensity-Score Matching" (Keizer, Bergstrand,
//! Hughes).
//!
//! The metric is the Mahalanobis distance under the model `Ω`,
//! `d²(a,b) = (a−b)ᵀ Ω⁻¹ (a−b)`, and matching is **optimal** (global minimum of
//! the total matched distance, i.e. the linear assignment problem) and 1:1
//! without replacement, mirroring `MatchIt(method = "optimal")`.

use nalgebra::{DMatrix, DVector};

/// Squared Mahalanobis distance between two eta vectors under `Ω⁻¹`.
///
/// `omega_inv` must be `d × d` where `d == a.len() == b.len()`.
pub fn mahalanobis_sq(a: &[f64], b: &[f64], omega_inv: &DMatrix<f64>) -> f64 {
    let d = a.len();
    debug_assert_eq!(b.len(), d);
    debug_assert_eq!(omega_inv.nrows(), d);
    debug_assert_eq!(omega_inv.ncols(), d);
    // Form the difference once, then evaluate the quadratic form diffᵀ Ω⁻¹ diff.
    let diff: Vec<f64> = (0..d).map(|k| a[k] - b[k]).collect();
    let mut acc = 0.0;
    for i in 0..d {
        let mut row = 0.0;
        for j in 0..d {
            row += omega_inv[(i, j)] * diff[j];
        }
        acc += diff[i] * row;
    }
    acc
}

/// Solve the square linear assignment problem: given an `n × n` cost matrix,
/// return `assign` of length `n` where `assign[i] = j` pairs row `i` with
/// column `j`, minimising `Σ_i cost[(i, assign[i])]`.
///
/// Uses the O(n³) Hungarian algorithm with potentials (Jonker–Volgenant-style
/// shortest augmenting paths). Costs are finite `f64`; ties are broken
/// arbitrarily but deterministically.
pub fn optimal_assignment(cost: &DMatrix<f64>) -> Vec<usize> {
    let n = cost.nrows();
    assert_eq!(
        cost.ncols(),
        n,
        "optimal_assignment requires a square matrix"
    );
    // Non-finite costs would make the shortest-path search never terminate (a
    // NaN compares false against every candidate). Callers must screen them out
    // (e.g. `simulate_with_options` rejects non-finite posthoc etas upstream).
    debug_assert!(
        cost.iter().all(|x| x.is_finite()),
        "optimal_assignment requires all costs to be finite"
    );
    if n == 0 {
        return Vec::new();
    }

    // 1-indexed working arrays (index 0 is a sentinel), following the classic
    // potentials formulation. `p[j]` is the row currently matched to column j.
    let inf = f64::INFINITY;
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    // cost is 0-indexed; working arrays are 1-indexed.
                    let cur = cost[(i0 - 1, j - 1)] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // Augment along the alternating path.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // p[j] = row matched to column j; invert to row -> column.
    let mut assign = vec![0usize; n];
    for j in 1..=n {
        assign[p[j] - 1] = j - 1;
    }
    assign
}

/// For each subject (indexed by `fitted`), pick the drawn pool index whose eta
/// optimally matches the subject's fitted eta under the `Ω⁻¹` metric.
///
/// `pool` and `fitted` must have equal length `n`; the return value `assign`
/// has length `n` with `assign[i]` the `pool` index assigned to subject `i`
/// (a permutation of `0..n`).
pub fn match_draws_to_fitted(
    pool: &[DVector<f64>],
    fitted: &[DVector<f64>],
    omega_inv: &DMatrix<f64>,
) -> Vec<usize> {
    let n = fitted.len();
    assert_eq!(pool.len(), n, "pool and fitted must have equal length");
    // Rows = subjects (fitted), columns = drawn pool.
    let mut cost = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            cost[(i, j)] = mahalanobis_sq(fitted[i].as_slice(), pool[j].as_slice(), omega_inv);
        }
    }
    optimal_assignment(&cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Brute-force minimum-cost permutation for small n, for cross-checking.
    fn brute_force(cost: &DMatrix<f64>) -> f64 {
        let n = cost.nrows();
        let mut idx: Vec<usize> = (0..n).collect();
        let mut best = f64::INFINITY;
        permute(&mut idx, 0, cost, &mut best);
        best
    }
    fn permute(idx: &mut [usize], k: usize, cost: &DMatrix<f64>, best: &mut f64) {
        let n = idx.len();
        if k == n {
            let total: f64 = (0..n).map(|i| cost[(i, idx[i])]).sum();
            if total < *best {
                *best = total;
            }
            return;
        }
        for s in k..n {
            idx.swap(k, s);
            permute(idx, k + 1, cost, best);
            idx.swap(k, s);
        }
    }

    fn total_cost(cost: &DMatrix<f64>, assign: &[usize]) -> f64 {
        (0..assign.len()).map(|i| cost[(i, assign[i])]).sum()
    }

    #[test]
    fn mahalanobis_identity_is_squared_euclidean() {
        let omega_inv = DMatrix::identity(2, 2);
        let d = mahalanobis_sq(&[1.0, 2.0], &[-1.0, 0.0], &omega_inv);
        assert!((d - (4.0 + 4.0)).abs() < 1e-12);
    }

    #[test]
    fn mahalanobis_diagonal_scales_each_axis() {
        // Ω⁻¹ = diag(1/4, 1) ⇒ d² = (Δ0)²/4 + (Δ1)².
        let omega_inv = DMatrix::from_diagonal(&DVector::from_vec(vec![0.25, 1.0]));
        let d = mahalanobis_sq(&[2.0, 3.0], &[0.0, 0.0], &omega_inv);
        assert!((d - (4.0 * 0.25 + 9.0)).abs() < 1e-12);
    }

    #[test]
    fn assignment_identity_when_diagonal_cheapest() {
        // Cheapest to match i->i.
        let cost = DMatrix::from_row_slice(3, 3, &[0.0, 5.0, 5.0, 5.0, 0.0, 5.0, 5.0, 5.0, 0.0]);
        let a = optimal_assignment(&cost);
        assert_eq!(a, vec![0, 1, 2]);
    }

    #[test]
    fn assignment_finds_off_diagonal_optimum() {
        // Optimal is the anti-diagonal (total 0), not the diagonal (total 3).
        let cost = DMatrix::from_row_slice(3, 3, &[1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let a = optimal_assignment(&cost);
        assert_eq!(total_cost(&cost, &a), 0.0);
        assert_eq!(a, vec![2, 1, 0]);
    }

    #[test]
    fn assignment_matches_brute_force_on_varied_matrices() {
        // A handful of deterministic, irregular matrices up to n = 7.
        let sizes = [1usize, 2, 3, 4, 5, 6, 7];
        for &n in &sizes {
            let mut cost = DMatrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    // Deterministic pseudo-random-ish values with ties.
                    let v = ((i * 7 + j * 3 + (i * j) % 5) % 11) as f64;
                    cost[(i, j)] = v;
                }
            }
            let a = optimal_assignment(&cost);
            // Valid permutation.
            let mut seen = vec![false; n];
            for &j in &a {
                assert!(!seen[j]);
                seen[j] = true;
            }
            let got = total_cost(&cost, &a);
            let want = brute_force(&cost);
            assert!((got - want).abs() < 1e-9, "n={n}: got {got}, want {want}");
        }
    }

    #[test]
    fn matching_is_identity_when_pool_equals_fitted() {
        // If the drawn pool equals the fitted etas, optimal matching is the
        // identity (each subject gets its own value, distance 0).
        let omega_inv = DMatrix::identity(2, 2);
        let fitted = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.5, -0.5]),
            DVector::from_vec(vec![-2.0, 1.0]),
        ];
        let pool = fitted.clone();
        let assign = match_draws_to_fitted(&pool, &fitted, &omega_inv);
        assert_eq!(assign, vec![0, 1, 2]);
    }

    #[test]
    fn empty_assignment() {
        let cost = DMatrix::<f64>::zeros(0, 0);
        assert!(optimal_assignment(&cost).is_empty());
    }
}
