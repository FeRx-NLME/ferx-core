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
//! `d²(a,b) = (a−b)ᵀ Ω⁻¹ (a−b)`. Three 1:1-without-replacement matching methods
//! are offered (see [`MatchMethod`]):
//!
//! - **optimal** — global minimum of the total matched distance (the linear
//!   assignment problem), mirroring `MatchIt(method = "optimal")`. Simulation
//!   studies found this best on average, so it is the recommended default.
//! - **nearest** — greedy nearest-neighbour in subject order, mirroring
//!   `MatchIt(method = "nearest", distance = "mahalanobis")`.
//! - **rank** — pair subjects and draws by the rank of their Mahalanobis norm
//!   `‖eta‖ = √(etaᵀ Ω⁻¹ eta)` (k-th order statistic to k-th), greedily on rank
//!   distance.
//!
//! "optimal" was best on average in simulation, but other methods can win on
//! particular datasets, hence all three are exposed (#396).

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

/// Greedy 1:1 assignment without replacement: for each row in index order,
/// pick the still-available column with least cost (ties → smallest column
/// index). `cost` must be square `n × n`; the return value `assign` has length
/// `n` and is a permutation of `0..n`.
///
/// Unlike [`optimal_assignment`] this minimises greedily, not globally — the
/// total matched cost can exceed the optimum. It mirrors the greedy nearest-
/// neighbour matching of `MatchIt(method = "nearest")`.
pub fn greedy_assignment(cost: &DMatrix<f64>) -> Vec<usize> {
    let n = cost.nrows();
    assert_eq!(
        cost.ncols(),
        n,
        "greedy_assignment requires a square matrix"
    );
    let mut used = vec![false; n];
    let mut assign = vec![0usize; n];
    for i in 0..n {
        // n − i ≥ 1 columns remain free when processing row i, so this always
        // finds a candidate; an all-NaN row would leave best_j unset, hence the
        // debug_assert below mirrors `optimal_assignment`'s finiteness contract.
        let mut best_j = usize::MAX;
        let mut best_c = f64::INFINITY;
        for j in 0..n {
            if !used[j] && cost[(i, j)] < best_c {
                best_c = cost[(i, j)];
                best_j = j;
            }
        }
        debug_assert!(best_j != usize::MAX, "greedy_assignment: no free column");
        used[best_j] = true;
        assign[i] = best_j;
    }
    assign
}

/// Mahalanobis norm of a single eta from the origin: `√(etaᵀ Ω⁻¹ eta)`.
fn mahalanobis_norm(a: &[f64], omega_inv: &DMatrix<f64>) -> f64 {
    mahalanobis_sq(a, &vec![0.0; a.len()], omega_inv).sqrt()
}

/// Dense rank of each eta by its Mahalanobis norm, ascending. Returns a vector
/// of rank positions `0..n` (ties broken by index, so a stable permutation).
fn ranks_by_norm(etas: &[DVector<f64>], omega_inv: &DMatrix<f64>) -> Vec<f64> {
    let scores: Vec<f64> = etas
        .iter()
        .map(|e| mahalanobis_norm(e.as_slice(), omega_inv))
        .collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut rank = vec![0.0f64; scores.len()];
    for (r, &idx) in order.iter().enumerate() {
        rank[idx] = r as f64;
    }
    rank
}

/// Squared-Mahalanobis cost matrix: rows = subjects (`fitted`), columns = drawn
/// `pool`. `cost[(i, j)] = d²(fitted[i], pool[j])` under `Ω⁻¹`.
fn mahalanobis_cost(
    pool: &[DVector<f64>],
    fitted: &[DVector<f64>],
    omega_inv: &DMatrix<f64>,
) -> DMatrix<f64> {
    let n = fitted.len();
    let mut cost = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            cost[(i, j)] = mahalanobis_sq(fitted[i].as_slice(), pool[j].as_slice(), omega_inv);
        }
    }
    cost
}

/// How drawn etas are reassigned to subjects per replicate. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMethod {
    /// Global minimum total Mahalanobis distance (linear assignment).
    /// `MatchIt(method = "optimal")`. Recommended default.
    Optimal,
    /// Greedy nearest-neighbour in subject order, 1:1 without replacement.
    /// `MatchIt(method = "nearest", distance = "mahalanobis")`.
    Nearest,
    /// Pair by Mahalanobis-norm rank (k-th order statistic to k-th).
    Rank,
}

/// For each subject (indexed by `fitted`), pick the drawn pool index that
/// matches the subject's fitted eta under the `Ω⁻¹` metric, using `method`.
///
/// `pool` and `fitted` must have equal length `n`; the return value `assign`
/// has length `n` with `assign[i]` the `pool` index assigned to subject `i`
/// (a permutation of `0..n`).
pub fn match_draws_to_fitted(
    pool: &[DVector<f64>],
    fitted: &[DVector<f64>],
    omega_inv: &DMatrix<f64>,
    method: MatchMethod,
) -> Vec<usize> {
    let n = fitted.len();
    assert_eq!(pool.len(), n, "pool and fitted must have equal length");
    match method {
        MatchMethod::Optimal => optimal_assignment(&mahalanobis_cost(pool, fitted, omega_inv)),
        MatchMethod::Nearest => greedy_assignment(&mahalanobis_cost(pool, fitted, omega_inv)),
        MatchMethod::Rank => {
            // Pair on the rank of each eta's Mahalanobis norm: greedily assign
            // the pool draw whose rank is closest to the subject's. With a
            // complete pool (distinct ranks 0..n) this reduces to matching the
            // k-th order statistic to the k-th, independent of subject order.
            let rank_fitted = ranks_by_norm(fitted, omega_inv);
            let rank_pool = ranks_by_norm(pool, omega_inv);
            let mut cost = DMatrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    cost[(i, j)] = (rank_fitted[i] - rank_pool[j]).abs();
                }
            }
            greedy_assignment(&cost)
        }
    }
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
        for method in [
            MatchMethod::Optimal,
            MatchMethod::Nearest,
            MatchMethod::Rank,
        ] {
            let assign = match_draws_to_fitted(&pool, &fitted, &omega_inv, method);
            assert_eq!(assign, vec![0, 1, 2], "method {method:?}");
        }
    }

    #[test]
    fn empty_assignment() {
        let cost = DMatrix::<f64>::zeros(0, 0);
        assert!(optimal_assignment(&cost).is_empty());
        assert!(greedy_assignment(&cost).is_empty());
    }

    #[test]
    fn greedy_picks_least_cost_per_row_in_order() {
        // Row 0's cheapest is col 1; once taken, row 1 must fall back to col 0.
        let cost = DMatrix::from_row_slice(2, 2, &[5.0, 1.0, 2.0, 3.0]);
        assert_eq!(greedy_assignment(&cost), vec![1, 0]);
    }

    #[test]
    fn greedy_can_be_suboptimal_vs_optimal() {
        // Greedy in row order takes (0->0, cost 0) then is forced into (1->1,
        // cost 10): total 10. The optimum is the anti-diagonal: total 2.
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 10.0]);
        assert_eq!(greedy_assignment(&cost), vec![0, 1]);
        assert_eq!(total_cost(&cost, &greedy_assignment(&cost)), 10.0);
        assert_eq!(total_cost(&cost, &optimal_assignment(&cost)), 2.0);
    }

    #[test]
    fn greedy_returns_valid_permutation() {
        let n = 6;
        let mut cost = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                cost[(i, j)] = ((i * 5 + j * 2 + (i + j) % 3) % 7) as f64;
            }
        }
        let a = greedy_assignment(&cost);
        let mut seen = vec![false; n];
        for &j in &a {
            assert!(!seen[j], "column {j} used twice");
            seen[j] = true;
        }
    }

    #[test]
    fn rank_pairs_by_mahalanobis_norm_order_statistics() {
        // Identity Ω⁻¹ ⇒ norm is Euclidean length. Fitted norms ascending:
        // idx2 (0) < idx0 (1) < idx1 (2). Pool norms: idx1 (0.5) < idx2 (1.5)
        // < idx0 (3). Rank-matching pairs the k-th smallest fitted with the
        // k-th smallest pool: fitted2->pool1, fitted0->pool2, fitted1->pool0.
        let omega_inv = DMatrix::identity(1, 1);
        let fitted = vec![
            DVector::from_vec(vec![1.0]),  // rank 1
            DVector::from_vec(vec![-2.0]), // rank 2 (norm 2)
            DVector::from_vec(vec![0.0]),  // rank 0
        ];
        let pool = vec![
            DVector::from_vec(vec![3.0]),  // rank 2
            DVector::from_vec(vec![0.5]),  // rank 0
            DVector::from_vec(vec![-1.5]), // rank 1 (norm 1.5)
        ];
        let assign = match_draws_to_fitted(&pool, &fitted, &omega_inv, MatchMethod::Rank);
        assert_eq!(assign, vec![2, 0, 1]);
    }
}
