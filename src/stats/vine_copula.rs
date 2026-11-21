//! Regular-vine (R-vine) copula + zero-mean flexible marginal distribution for SAEM.
//!
//! Implements Dissmann, Brechmann, Czado & Kurowicka (2013) *Computational Statistics
//! & Data Analysis* 59:52-69:
//! - Sequential maximum-spanning-tree R-vine structure selection by empirical `|tau|`.
//! - Leaf-peeling canonical simulation ordering `perm[0..d]` with verified conditioning
//!   pointers (`PartnerSource`), guaranteeing exact agreement between:
//!   - [`RVine::log_density`] (Algorithm 2.1),
//!   - [`RVine::rosenblatt_forward`] (`u -> w`), and
//!   - [`RVine::rosenblatt_inverse`] (`w -> u`, Algorithm 2.2),
//!   across arbitrary dimension `d` and asymmetric `90`/`270`-degree rotated copulas.
//! - Zero-mean constrained marginals ([`MarginalDist`]) + Robbins-Monro score &
//!   observed-information stochastic approximation ([`MarginalSaState`], [`CopulaSaState`]).

use crate::stats::copula::{empirical_kendall_tau, CopulaFamily, CopulaRotation, CopulaSaState};
use crate::stats::marginals::{
    select_marginal_bic, MarginalDist, MarginalSaState, MarginalSummary, MARGINAL_SCALE_FLOOR,
};
use crate::stats::random_effects::RandomEffectDistribution;
use crate::types::{ModelParameters, OmegaMatrix};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, Open01};
use serde::{Deserialize, Serialize};

const U_CLAMP: f64 = 1e-12;

/// Transpose a bivariate copula (`(U, V) -> (V, U)`): symmetric families are invariant,
/// while `90`-degree and `270`-degree rotations swap (`c_90(v, u) == c_270(u, v)`).
#[inline]
fn transpose_copula(cop: CopulaFamily) -> CopulaFamily {
    match cop {
        CopulaFamily::Clayton { theta, rotation: CopulaRotation::R90 } => {
            CopulaFamily::Clayton { theta, rotation: CopulaRotation::R270 }
        }
        CopulaFamily::Clayton { theta, rotation: CopulaRotation::R270 } => {
            CopulaFamily::Clayton { theta, rotation: CopulaRotation::R90 }
        }
        CopulaFamily::Gumbel { theta, rotation: CopulaRotation::R90 } => {
            CopulaFamily::Gumbel { theta, rotation: CopulaRotation::R270 }
        }
        CopulaFamily::Gumbel { theta, rotation: CopulaRotation::R270 } => {
            CopulaFamily::Gumbel { theta, rotation: CopulaRotation::R90 }
        }
        other => other,
    }
}

/// Exact source of `F(partner | D)` in an earlier column `< i` of the R-vine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum PartnerSource {
    /// Tree 0 (`D = empty`): read unconditional `u_int[var]` directly.
    Root(usize),
    /// Tree `t >= 1`, `partner` was `var1` of `columns[col][tree]` -> read `h1_out[col][tree]`.
    H1 { col: usize, tree: usize },
    /// Tree `t >= 1`, `partner` was `var2` of `columns[col][tree]` -> read `h2_out[col][tree]`.
    H2 { col: usize, tree: usize },
}

/// One directed edge inside column `i` (`var1_int == i`) at tree level `t` (`0..i`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RVineEdge {
    /// 0-based tree index (`0` = unconditional MST, `1` = conditioned on 1 variable, ...).
    pub tree_level: usize,
    /// First conditioned variable in internal ordering (`== i`).
    pub var1_int: usize,
    /// Second conditioned variable in internal ordering (`< i`).
    pub var2_int: usize,
    /// Sorted internal indices of the conditioning set `D` (length `tree_level`).
    pub cond_set_int: Vec<usize>,
    /// Pre-verified pointer locating `F(var2_int | D)` in columns `0..i`.
    pub partner_src: PartnerSource,
    /// Fitted bivariate pair-copula on `(F(var1_int | D), F(var2_int | D))`.
    pub copula: CopulaFamily,
}

/// Regular vine on `d` variables in canonical leaf-peeled simulation ordering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RVine {
    /// Dimension `d`.
    pub d: usize,
    /// Permutation mapping internal index `a in 0..d` -> original ETA index `perm[a]`.
    pub perm: Vec<usize>,
    /// Inverse permutation mapping original ETA index `orig` -> internal index `inv_perm[orig]`.
    pub inv_perm: Vec<usize>,
    /// Per-column edge stacks: `columns[i]` has length `i` (`t = 0..i`), empty for `i == 0`.
    pub(crate) columns: Vec<Vec<RVineEdge>>,
}

impl RVine {
    /// Construct a default natural-order D-vine (`perm = 0..d`) initialized at independence.
    pub fn independence_dvine(d: usize) -> Self {
        let perm: Vec<usize> = (0..d).collect();
        let inv_perm: Vec<usize> = (0..d).collect();
        let mut columns: Vec<Vec<RVineEdge>> = vec![Vec::new(); d];
        for (i, col_edges) in columns.iter_mut().enumerate().skip(1) {
            // Natural D-vine: column i connects i with m = i - 1 - t given {i - t, ..., i - 1}.
            for t in 0..i {
                let m = i - 1 - t;
                let cond_set_int: Vec<usize> = ((m + 1)..i).collect();
                let partner_src = if t == 0 {
                    PartnerSource::Root(m)
                } else {
                    // For D-vine: {m} U {m+1..i-1} = {m..i-1}, produced by column i-1 at tree t-1
                    // where var1 = i-1 and var2 = m, so F(m | {m+1..i-1}) is H2 of (i-1, t-1)!
                    PartnerSource::H2 { col: i - 1, tree: t - 1 }
                };
                col_edges.push(RVineEdge {
                    tree_level: t,
                    var1_int: i,
                    var2_int: m,
                    cond_set_int,
                    partner_src,
                    copula: CopulaFamily::Independence,
                });
            }
        }
        Self { d, perm, inv_perm, columns }
    }

    /// Total number of free scalar parameters across all pair-copulas in the R-vine.
    pub fn n_free_params(&self) -> usize {
        self.columns.iter().flat_map(|col| col.iter()).map(|e| e.copula.n_params()).sum()
    }

    /// Set the pair-copula family on internal column `col_int` (`1..d`) at 0-based tree `tree` (`0..col_int`).
    pub fn set_edge_copula(&mut self, col_int: usize, tree: usize, copula: CopulaFamily) {
        if let Some(edge) = self.columns.get_mut(col_int).and_then(|c| c.get_mut(tree)) {
            edge.copula = copula;
        }
    }

    #[inline]
    fn read_partner(
        src: PartnerSource, u_int: &[f64], h1_out: &[Vec<f64>], h2_out: &[Vec<f64>],
    ) -> f64 {
        match src {
            PartnerSource::Root(m) => u_int[m],
            PartnerSource::H1 { col, tree } => h1_out[col][tree],
            PartnerSource::H2 { col, tree } => h2_out[col][tree],
        }
    }

    /// Evaluate exact R-vine copula log-density `ln c(u_orig)` for `u_orig in (0, 1)^d`.
    pub fn log_density(&self, u_orig: &[f64]) -> f64 {
        let d = self.d;
        if d <= 1 {
            return 0.0;
        }
        let u_int: Vec<f64> = (0..d)
            .map(|a| {
                let x = u_orig[self.perm[a]];
                if x.is_finite() {
                    x.clamp(U_CLAMP, 1.0 - U_CLAMP)
                } else {
                    0.5
                }
            })
            .collect();

        let mut h1_out: Vec<Vec<f64>> = (0..d).map(|i| vec![0.0; i]).collect();
        let mut h2_out: Vec<Vec<f64>> = (0..d).map(|i| vec![0.0; i]).collect();
        let mut log_c = 0.0_f64;

        for i in 1..d {
            let mut v_curr = u_int[i];
            for (t, edge) in self.columns[i].iter().enumerate() {
                let v_part = Self::read_partner(edge.partner_src, &u_int, &h1_out, &h2_out);
                log_c += edge.copula.log_density(v_curr, v_part);
                let next_h1 = edge.copula.h1(v_curr, v_part);
                let next_h2 = edge.copula.h2(v_part, v_curr);
                h1_out[i][t] = next_h1;
                h2_out[i][t] = next_h2;
                v_curr = next_h1;
            }
        }
        if log_c.is_finite() {
            log_c
        } else {
            -crate::stats::likelihood::SUBJECT_SENTINEL_NLL
        }
    }

    /// Forward Rosenblatt transform `u_orig in (0, 1)^d -> w in (0, 1)^d` (independent uniforms
    /// under `self`). Used for diagnostics and exact round-trip verification.
    pub fn rosenblatt_forward(&self, u_orig: &[f64]) -> Vec<f64> {
        let d = self.d;
        if d == 0 {
            return Vec::new();
        }
        let u_int: Vec<f64> =
            (0..d).map(|a| u_orig[self.perm[a]].clamp(U_CLAMP, 1.0 - U_CLAMP)).collect();
        let mut h1_out: Vec<Vec<f64>> = (0..d).map(|i| vec![0.0; i]).collect();
        let mut h2_out: Vec<Vec<f64>> = (0..d).map(|i| vec![0.0; i]).collect();
        let mut w_int = vec![0.0; d];
        w_int[0] = u_int[0];

        for i in 1..d {
            let mut v_curr = u_int[i];
            for (t, edge) in self.columns[i].iter().enumerate() {
                let v_part = Self::read_partner(edge.partner_src, &u_int, &h1_out, &h2_out);
                let next_h1 = edge.copula.h1(v_curr, v_part);
                let next_h2 = edge.copula.h2(v_part, v_curr);
                h1_out[i][t] = next_h1;
                h2_out[i][t] = next_h2;
                v_curr = next_h1;
            }
            w_int[i] = v_curr;
        }
        w_int
    }

    /// Inverse Rosenblatt transform `w in (0, 1)^d -> u_orig in (0, 1)^d` (Dissmann Algorithm 2.2).
    pub fn rosenblatt_inverse(&self, w: &[f64]) -> Vec<f64> {
        let d = self.d;
        if d == 0 {
            return Vec::new();
        }
        let mut u_int = vec![0.0; d];
        let mut h1_out: Vec<Vec<f64>> = (0..d).map(|i| vec![0.0; i]).collect();
        let mut h2_out: Vec<Vec<f64>> = (0..d).map(|i| vec![0.0; i]).collect();

        u_int[0] = w[0].clamp(U_CLAMP, 1.0 - U_CLAMP);
        for i in 1..d {
            // All conditioning values `conds[t]` depend strictly on columns `0..i`, which
            // are already realized!
            let conds: Vec<f64> = self.columns[i]
                .iter()
                .map(|edge| Self::read_partner(edge.partner_src, &u_int, &h1_out, &h2_out))
                .collect();

            let mut v_curr = w[i].clamp(U_CLAMP, 1.0 - U_CLAMP);
            for t in (0..i).rev() {
                v_curr = self.columns[i][t].copula.h1_inv(v_curr, conds[t]);
            }
            u_int[i] = v_curr;

            // Populate h1_out / h2_out of column `i` for columns `> i`.
            let mut v_fwd = u_int[i];
            for t in 0..i {
                let next_h1 = self.columns[i][t].copula.h1(v_fwd, conds[t]);
                let next_h2 = self.columns[i][t].copula.h2(conds[t], v_fwd);
                h1_out[i][t] = next_h1;
                h2_out[i][t] = next_h2;
                v_fwd = next_h1;
            }
        }

        let mut u_orig = vec![0.0; d];
        for a in 0..d {
            u_orig[self.perm[a]] = u_int[a];
        }
        u_orig
    }

    /// Extract per-edge pseudo-observation pairs `(u1_vec, u2_vec)` on a sample matrix
    /// `u_samples[s][orig_dim]` using the frozen R-vine structure, optionally updating each
    /// edge's copula via `CopulaSaState` along the sequential pass.
    pub(crate) fn forward_pseudo_obs_and_step(
        &mut self, u_samples: &[Vec<f64>], mut sa_states: Option<(&mut [Vec<CopulaSaState>], f64)>,
        zero_fixed: &[bool],
    ) -> Vec<Vec<(Vec<f64>, Vec<f64>)>> {
        let d = self.d;
        let n = u_samples.len();
        if d <= 1 || n == 0 {
            return vec![Vec::new(); d];
        }

        // u_int_cols[a][s] = sample s of internal variable a.
        let u_int_cols: Vec<Vec<f64>> = (0..d)
            .map(|a| {
                let orig = self.perm[a];
                u_samples.iter().map(|row| row[orig].clamp(1e-10, 1.0 - 1e-10)).collect()
            })
            .collect();

        let mut h1_cols: Vec<Vec<Vec<f64>>> = (0..d).map(|i| vec![vec![0.0; n]; i]).collect();
        let mut h2_cols: Vec<Vec<Vec<f64>>> = (0..d).map(|i| vec![vec![0.0; n]; i]).collect();
        let mut stored_pairs: Vec<Vec<(Vec<f64>, Vec<f64>)>> =
            (0..d).map(|i| Vec::with_capacity(i)).collect();

        for i in 1..d {
            let mut v_curr_col = u_int_cols[i].clone();
            let orig_i = self.perm[i];
            let fixed_zero_i = zero_fixed.get(orig_i).copied().unwrap_or(false);

            for t in 0..i {
                let src = self.columns[i][t].partner_src;
                let orig_m = self.perm[self.columns[i][t].var2_int];
                let fixed_zero_m = zero_fixed.get(orig_m).copied().unwrap_or(false);
                let v_part_col: Vec<f64> = match src {
                    PartnerSource::Root(m) => u_int_cols[m].clone(),
                    PartnerSource::H1 { col, tree } => h1_cols[col][tree].clone(),
                    PartnerSource::H2 { col, tree } => h2_cols[col][tree].clone(),
                };

                if fixed_zero_i || fixed_zero_m {
                    self.columns[i][t].copula = CopulaFamily::Independence;
                } else if let Some((ref mut states, gamma)) = sa_states {
                    states[i][t].step(
                        &mut self.columns[i][t].copula,
                        &v_curr_col,
                        &v_part_col,
                        gamma,
                    );
                }

                let cop = self.columns[i][t].copula;
                for s in 0..n {
                    let u1 = v_curr_col[s];
                    let u2 = v_part_col[s];
                    h1_cols[i][t][s] = cop.h1(u1, u2);
                    h2_cols[i][t][s] = cop.h2(u2, u1);
                }
                let next_v = h1_cols[i][t].clone();
                stored_pairs[i].push((v_curr_col, v_part_col));
                v_curr_col = next_v;
            }
        }
        stored_pairs
    }
}

/// Select an R-vine structure and pair-copula families by Dissmann et al. (2013)
/// sequential maximum-spanning-tree (`|tau|`) + BIC on `u_samples[s][orig_dim]`.
pub fn select_rvine_dissmann(
    u_samples: &[Vec<f64>], weight: f64, n_subjects: usize, zero_fixed: &[bool],
    gaussian_only: bool,
) -> (RVine, Vec<Vec<(Vec<f64>, Vec<f64>)>>) {
    let d = u_samples.first().map_or(0, Vec::len);
    if d <= 1 || u_samples.len() < 6 {
        let rv = RVine::independence_dvine(d);
        return (rv, vec![Vec::new(); d]);
    }
    let n = u_samples.len();

    #[derive(Clone)]
    struct RawEdge {
        var_a: usize,
        var_b: usize,
        cond_set: Vec<usize>,
        full_set: Vec<usize>,
        copula: CopulaFamily,
        u_a_given_b: Vec<f64>,
        u_b_given_a: Vec<f64>,
    }

    let u_cols: Vec<Vec<f64>> = (0..d)
        .map(|j| u_samples.iter().map(|r| r[j].clamp(1e-10, 1.0 - 1e-10)).collect())
        .collect();

    let fit_edge_copula = |u1: &[f64], u2: &[f64], a: usize, b: usize| -> CopulaFamily {
        if zero_fixed.get(a).copied().unwrap_or(false)
            || zero_fixed.get(b).copied().unwrap_or(false)
        {
            return CopulaFamily::Independence;
        }
        if gaussian_only {
            CopulaFamily::gaussian(0.0).fit_mle(u1, u2)
        } else {
            CopulaFamily::select_bic(u1, u2, weight, n_subjects)
        }
    };

    let mut trees_raw: Vec<Vec<RawEdge>> = Vec::with_capacity(d - 1);

    // ── Tree 0: MST over variables 0..d weighted by |tau| ─────────────────────
    {
        let mut adj = vec![vec![-1.0_f64; d]; d];
        for a in 0..d {
            for b in (a + 1)..d {
                let w = if zero_fixed.get(a).copied().unwrap_or(false)
                    || zero_fixed.get(b).copied().unwrap_or(false)
                {
                    0.0
                } else {
                    empirical_kendall_tau(&u_cols[a], &u_cols[b]).abs()
                };
                adj[a][b] = w;
                adj[b][a] = w;
            }
        }
        let mst_pairs = prim_mst(d, &adj);
        let mut t0 = Vec::with_capacity(d - 1);
        for (u, v) in mst_pairs {
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            let cop = fit_edge_copula(&u_cols[a], &u_cols[b], a, b);
            let mut u_ab = vec![0.0; n];
            let mut u_ba = vec![0.0; n];
            for s in 0..n {
                u_ab[s] = cop.h1(u_cols[a][s], u_cols[b][s]);
                u_ba[s] = cop.h2(u_cols[b][s], u_cols[a][s]);
            }
            t0.push(RawEdge {
                var_a: a,
                var_b: b,
                cond_set: Vec::new(),
                full_set: vec![a, b],
                copula: cop,
                u_a_given_b: u_ab,
                u_b_given_a: u_ba,
            });
        }
        trees_raw.push(t0);
    }

    // ── Trees 1..d-2: MST on proximity graph of previous tree's edges ─────────
    for t in 1..(d - 1) {
        let prev = &trees_raw[t - 1];
        let m_nodes = prev.len(); // == d - t
        let mut adj = vec![vec![-1.0_f64; m_nodes]; m_nodes];
        // Cache candidate intersection metadata.
        #[allow(clippy::type_complexity)]
        let mut meta: Vec<Vec<Option<(usize, usize, Vec<usize>)>>> =
            vec![vec![None; m_nodes]; m_nodes];

        for p in 0..m_nodes {
            for q in (p + 1)..m_nodes {
                let inter = sorted_intersection(&prev[p].full_set, &prev[q].full_set);
                if inter.len() == t {
                    let a = unique_diff(&prev[p].full_set, &inter);
                    let b = unique_diff(&prev[q].full_set, &inter);
                    let u_a = if a == prev[p].var_a {
                        &prev[p].u_a_given_b
                    } else {
                        &prev[p].u_b_given_a
                    };
                    let u_b = if b == prev[q].var_a {
                        &prev[q].u_a_given_b
                    } else {
                        &prev[q].u_b_given_a
                    };
                    let w = if zero_fixed.get(a).copied().unwrap_or(false)
                        || zero_fixed.get(b).copied().unwrap_or(false)
                    {
                        0.0
                    } else {
                        empirical_kendall_tau(u_a, u_b).abs()
                    };
                    adj[p][q] = w;
                    adj[q][p] = w;
                    meta[p][q] = Some((a, b, inter));
                }
            }
        }

        let mst_pairs = prim_mst(m_nodes, &adj);
        let mut cur_tree = Vec::with_capacity(d - 1 - t);
        for (u_node, v_node) in mst_pairs {
            let (p, q) = if u_node < v_node { (u_node, v_node) } else { (v_node, u_node) };
            let (a, b, cond_set) =
                meta[p][q].clone().expect("MST edge must satisfy proximity condition");
            let u_a = if a == prev[p].var_a { &prev[p].u_a_given_b } else { &prev[p].u_b_given_a };
            let u_b = if b == prev[q].var_a { &prev[q].u_a_given_b } else { &prev[q].u_b_given_a };
            let cop = fit_edge_copula(u_a, u_b, a, b);
            let mut u_ab = vec![0.0; n];
            let mut u_ba = vec![0.0; n];
            for s in 0..n {
                u_ab[s] = cop.h1(u_a[s], u_b[s]);
                u_ba[s] = cop.h2(u_b[s], u_a[s]);
            }
            let mut full_set = cond_set.clone();
            full_set.push(a);
            full_set.push(b);
            full_set.sort_unstable();
            cur_tree.push(RawEdge {
                var_a: a,
                var_b: b,
                cond_set,
                full_set,
                copula: cop,
                u_a_given_b: u_ab,
                u_b_given_a: u_ba,
            });
        }
        trees_raw.push(cur_tree);
    }

    // ── Leaf peeling from m = d down to 2 to establish canonical ordering ─────
    let mut active = trees_raw;
    let mut perm = vec![0_usize; d];
    // Peeled raw edges per internal index `i` (`t = 0..i`), still holding original var IDs.
    let mut peeled_orig: Vec<Vec<(usize, Vec<usize>, CopulaFamily)>> = vec![Vec::new(); d];

    for m in (2..=d).rev() {
        let top_edge = &active[m - 2][0];
        let leaf_orig = top_edge.var_a.max(top_edge.var_b);
        perm[m - 1] = leaf_orig;
        if m == 2 {
            perm[0] = if top_edge.var_a == leaf_orig { top_edge.var_b } else { top_edge.var_a };
        }
        let mut col_raw = Vec::with_capacity(m - 1);
        for active_tree in active.iter_mut().take(m - 1) {
            let pos = active_tree
                .iter()
                .position(|e| e.full_set.binary_search(&leaf_orig).is_ok())
                .expect("peeled leaf must belong to exactly one edge per active tree");
            let e = active_tree.swap_remove(pos);
            let (partner_orig, oriented_cop) = if e.var_a == leaf_orig {
                (e.var_b, e.copula)
            } else {
                debug_assert_eq!(e.var_b, leaf_orig);
                (e.var_a, transpose_copula(e.copula))
            };
            col_raw.push((partner_orig, e.cond_set, oriented_cop));
        }
        peeled_orig[m - 1] = col_raw;
    }

    let mut inv_perm = vec![0_usize; d];
    for (int_idx, &orig_idx) in perm.iter().enumerate() {
        inv_perm[orig_idx] = int_idx;
    }

    // Resolve internal variable indices and verified `PartnerSource` pointers.
    let mut columns: Vec<Vec<RVineEdge>> = vec![Vec::new(); d];
    for i in 1..d {
        let mut col_edges = Vec::with_capacity(i);
        for (t, (partner_orig, cond_orig, cop)) in peeled_orig[i].iter().enumerate() {
            let m_int = inv_perm[*partner_orig];
            debug_assert!(m_int < i, "partner internal index must be < i");
            let mut cond_int: Vec<usize> = cond_orig.iter().map(|&c| inv_perm[c]).collect();
            cond_int.sort_unstable();
            debug_assert!(cond_int.iter().all(|&c| c < i));

            let partner_src = if t == 0 {
                PartnerSource::Root(m_int)
            } else {
                // Find the unique column `col < i` at `tree = t - 1` whose output matches
                // `(conditioned = m_int, cond_set = cond_int)`.
                let mut target_full = cond_int.clone();
                target_full.push(m_int);
                target_full.sort_unstable();
                let max_col = *target_full.last().expect("non-empty set");
                let prev_edge = &columns[max_col][t - 1];
                if prev_edge.var1_int == m_int {
                    let mut check = prev_edge.cond_set_int.clone();
                    check.push(prev_edge.var2_int);
                    check.sort_unstable();
                    debug_assert_eq!(check, cond_int);
                    PartnerSource::H1 { col: max_col, tree: t - 1 }
                } else {
                    debug_assert_eq!(prev_edge.var2_int, m_int);
                    let mut check = prev_edge.cond_set_int.clone();
                    check.push(prev_edge.var1_int);
                    check.sort_unstable();
                    debug_assert_eq!(check, cond_int);
                    PartnerSource::H2 { col: max_col, tree: t - 1 }
                }
            };

            col_edges.push(RVineEdge {
                tree_level: t,
                var1_int: i,
                var2_int: m_int,
                cond_set_int: cond_int,
                partner_src,
                copula: *cop,
            });
        }
        columns[i] = col_edges;
    }

    let mut rvine = RVine { d, perm, inv_perm, columns };
    let pairs = rvine.forward_pseudo_obs_and_step(u_samples, None, zero_fixed);
    (rvine, pairs)
}

fn sorted_intersection(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

fn unique_diff(full: &[usize], inter: &[usize]) -> usize {
    for &x in full {
        if inter.binary_search(&x).is_err() {
            return x;
        }
    }
    panic!("set difference must contain exactly one variable");
}

/// Prim's Maximum Spanning Tree on `m` vertices (`adj[u][v] < 0` means non-adjacent).
fn prim_mst(m: usize, adj: &[Vec<f64>]) -> Vec<(usize, usize)> {
    if m <= 1 {
        return Vec::new();
    }
    let mut in_tree = vec![false; m];
    in_tree[0] = true;
    let mut edges = Vec::with_capacity(m - 1);
    for _ in 0..(m - 1) {
        let mut best_w = -1.0_f64;
        let mut best_edge = None;
        for u in 0..m {
            if !in_tree[u] {
                continue;
            }
            for (v, &in_v) in in_tree.iter().enumerate() {
                if in_v {
                    continue;
                }
                let w = adj[u][v];
                if w > best_w {
                    best_w = w;
                    best_edge = Some((u, v));
                }
            }
        }
        let (u, v) = best_edge.expect("R-vine proximity graph must be connected");
        in_tree[v] = true;
        edges.push((u, v));
    }
    edges
}

// ---------------------------------------------------------------------------
// Reporting & serialization structs (`FitResult` / YAML / `.fitrx` / `ferx-r`)
// ---------------------------------------------------------------------------

/// Summary of one fitted bivariate pair-copula.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairCopulaSummary {
    /// Canonical family label (e.g. `"clayton"`, `"clayton_90"`, `"gaussian"`, `"independence"`).
    pub family: String,
    /// Rotation degrees (`0`, `90`, `180`, or `270`).
    #[serde(default)]
    pub rotation: u16,
    /// Named natural parameters (`[]` for `"independence"`).
    pub params: Vec<(String, f64)>,
    /// Approximate observed-information SEs paired with `params`.
    #[serde(default)]
    pub se: Vec<(String, f64)>,
    /// Kendall's rank correlation `tau in [-1, 1]`.
    pub kendall_tau: f64,
    /// Lower diagonal tail-dependence coefficient `lambda_L in [0, 1]`.
    pub tail_dep_lower: f64,
    /// Upper diagonal tail-dependence coefficient `lambda_U in [0, 1]`.
    pub tail_dep_upper: f64,
}

/// One pair-copula edge inside a vine tree summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VinePairEntry {
    /// Human-readable edge label (e.g. `"ETA_CL ~ ETA_V"` or `"ETA_KA ~ ETA_CL | ETA_V"`).
    pub label: String,
    /// Internal edge coordinate `(col_int, tree_level)` for exact R-vine reconstruction.
    pub edge_coord: (usize, usize),
    /// Fitted pair-copula summary.
    pub copula: PairCopulaSummary,
}

/// Summary of all pair-copulas at one vine tree level (`1..=d-1`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VineTreeSummary {
    /// 1-based tree index (`1` = unconditional MST, `2` = conditioned on 1 ETA, ...).
    pub tree: usize,
    /// Pair-copula edges at this tree level.
    pub pairs: Vec<VinePairEntry>,
}

/// Complete serializable summary of a fitted [`VineCopulaOmega`] distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VineFitParams {
    /// Per-ETA fitted zero-mean marginal summaries in model ETA declaration order.
    pub marginals: Vec<MarginalSummary>,
    /// Vine tree summaries (`1..=d-1`).
    pub trees: Vec<VineTreeSummary>,
    /// Exact R-vine structure for lossless simulation/density reconstruction across FFI and `.fitrx`.
    pub rvine: RVine,
    /// Per-ETA `FIX` flags preserved across `.fitrx` and post-fit consumers (`simulate`, `npde`).
    #[serde(default)]
    pub omega_fixed: Vec<bool>,
    /// Per-ETA `0 FIX` flags (`omega ETA_X ~ 0 FIX`) preserved across `.fitrx` and post-fit consumers.
    #[serde(default)]
    pub zero_fixed: Vec<bool>,
}

impl VineFitParams {
    /// Total number of free estimated parameters across all non-fixed marginals and pair-copulas.
    pub fn n_free_params(&self) -> usize {
        let m_free: usize = self
            .marginals
            .iter()
            .enumerate()
            .filter(|&(j, m)| {
                !self.omega_fixed.get(j).copied().unwrap_or(false)
                    && !self.zero_fixed.get(j).copied().unwrap_or(false)
                    && m.sd > 0.0
            })
            .map(|(_, m)| m.params.len())
            .sum();
        let c_free: usize =
            self.trees.iter().flat_map(|t| t.pairs.iter()).map(|p| p.copula.params.len()).sum();
        m_free + c_free
    }
}

// ---------------------------------------------------------------------------
// VineCopulaOmega — full random-effect distribution
// ---------------------------------------------------------------------------

/// Zero-mean flexible marginals + Dissmann regular-vine random-effect distribution.
#[derive(Debug, Clone)]
pub struct VineCopulaOmega {
    /// Number of random effects `d = n_eta`.
    pub d: usize,
    /// Per-ETA zero-mean marginal distributions (in model declaration order `0..d`).
    pub marginals: Vec<MarginalDist>,
    /// Regular-vine copula over `u in (0, 1)^d`.
    pub rvine: RVine,
    /// Whether BIC family & R-vine structure selection has executed.
    pub selected: bool,
    /// Whether flexible non-Gaussian marginals (`StudentT`, `SkewNormal`, `SkewT`) are permitted.
    pub allow_flexible_marginals: bool,
    /// Per-ETA Robbins-Monro score/info SA states.
    marginal_sa: Vec<MarginalSaState>,
    /// Per-R-vine-edge Robbins-Monro score/info SA states (`copula_sa[i][t]`).
    copula_sa: Vec<Vec<CopulaSaState>>,
    /// Dense Robbins-Monro sample second-moment matrix `(1/N) sum eta_i eta_i^T`.
    sample_s2: DMatrix<f64>,
    /// Dense working covariance / Gaussian-equivalent `OmegaMatrix` (never off-diagonal wiped!).
    omega_equiv: OmegaMatrix,
    /// Initial `OmegaMatrix` template (`eta_names`, `diagonal` provenance, fixed values).
    initial_omega: OmegaMatrix,
    /// Per-ETA `FIX` flags.
    omega_fixed: Vec<bool>,
    /// Per-ETA flag: `FIX` at exact variance `0.0`.
    zero_fixed: Vec<bool>,
    /// Buffered pseudo-observations from the most recent selection/update for SE reporting
    /// (`(pairs_by_col, sample_weight, marginal_samples_by_eta)`).
    last_pair_obs: Vec<Vec<(Vec<f64>, Vec<f64>)>>,
    last_marginal_obs: Vec<Vec<f64>>,
    last_obs_weight: f64,
}

impl VineCopulaOmega {
    /// Construct from initial model parameters (starts as a zero-mean Gaussian copula).
    pub fn from_init_params(init_params: &ModelParameters, allow_flexible_marginals: bool) -> Self {
        let omega = init_params.omega.clone();
        let d = omega.dim();
        let struct_zeros = omega.structural_zeros();
        let mut zero_fixed = vec![false; d];
        let marginals: Vec<MarginalDist> = (0..d)
            .map(|i| {
                let v = omega.matrix[(i, i)];
                let fixed = init_params.omega_fixed.get(i).copied().unwrap_or(false);
                if fixed && (struct_zeros.get(i).copied().unwrap_or(false) || v <= 1e-12) {
                    zero_fixed[i] = true;
                }
                MarginalDist::normal(v.max(MARGINAL_SCALE_FLOOR * MARGINAL_SCALE_FLOOR).sqrt())
            })
            .collect();

        let rvine = RVine::independence_dvine(d);
        let copula_sa: Vec<Vec<CopulaSaState>> =
            (0..d).map(|i| vec![CopulaSaState::default(); i]).collect();
        let marginal_sa = vec![MarginalSaState::new(); d];

        // Build dense free_mask on `omega_equiv` so `proposal_chol()` adapts to full correlation
        // even when the `.ferx` file declared standalone `omega` lines!
        let dense_mask = DMatrix::from_element(d, d, true);
        let omega_equiv = OmegaMatrix::from_matrix_with_mask(
            omega.matrix.clone(),
            omega.eta_names.clone(),
            false,
            dense_mask,
        );

        Self {
            d,
            marginals,
            rvine,
            selected: false,
            allow_flexible_marginals,
            marginal_sa,
            copula_sa,
            sample_s2: omega.matrix.clone(),
            omega_equiv,
            initial_omega: omega,
            omega_fixed: init_params.omega_fixed.clone(),
            zero_fixed,
            last_pair_obs: vec![Vec::new(); d],
            last_marginal_obs: vec![Vec::new(); d],
            last_obs_weight: 1.0,
        }
    }

    /// True when ETA coordinate `j` is `FIX` at exact variance `0.0`.
    #[inline]
    pub fn is_zero_fixed(&self, j: usize) -> bool {
        self.zero_fixed.get(j).copied().unwrap_or(false)
    }

    /// True once family & R-vine structure selection has executed.
    #[inline]
    pub fn is_selected(&self) -> bool {
        self.selected
    }

    /// Total free parameters owned by the vine random-effect model:
    /// `sum_{j: !fixed_j} marginals[j].n_params() + rvine.n_free_params()`.
    pub fn n_free_params(&self) -> usize {
        let marg_k: usize = (0..self.d)
            .filter(|&j| !self.omega_fixed.get(j).copied().unwrap_or(false))
            .map(|j| self.marginals[j].n_params())
            .sum();
        marg_k + self.rvine.n_free_params()
    }

    /// Transform `eta` vectors into pseudo-observations `u in (0, 1)^d` via current marginals.
    pub fn pit_samples(&self, etas: &[Vec<f64>]) -> Vec<Vec<f64>> {
        etas.iter()
            .map(|eta| {
                (0..self.d)
                    .map(|j| {
                        if self.zero_fixed[j] {
                            0.5
                        } else {
                            self.marginals[j].cdf(eta[j]).clamp(1e-10, 1.0 - 1e-10)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Run one-time structure & family selection on pooled exploration draws
    /// (`pooled_etas` of length `M * N_subjects`, weighted by `weight = 1 / M`).
    pub fn select_structure_and_families(
        &mut self, pooled_etas: &[Vec<f64>], n_subjects: usize, weight: f64,
    ) {
        if pooled_etas.is_empty() || self.d == 0 {
            self.selected = true;
            return;
        }

        // Stage 1: Select and fit zero-mean marginals per non-fixed ETA by BIC.
        let mut marg_obs = vec![Vec::with_capacity(pooled_etas.len()); self.d];
        for eta in pooled_etas {
            for j in 0..self.d {
                marg_obs[j].push(eta[j]);
            }
        }
        for j in 0..self.d {
            let fixed = self.omega_fixed.get(j).copied().unwrap_or(false);
            if !fixed {
                self.marginals[j] = select_marginal_bic(
                    &marg_obs[j],
                    weight,
                    n_subjects,
                    self.allow_flexible_marginals,
                );
            }
            self.marginal_sa[j].reset();
        }

        // Stage 2: Exact PIT -> Dissmann R-vine MST + pair-copula BIC selection.
        let u_pooled = self.pit_samples(pooled_etas);
        let (rvine, pair_obs) =
            select_rvine_dissmann(&u_pooled, weight, n_subjects, &self.omega_fixed, false);
        self.rvine = rvine;
        self.copula_sa = (0..self.d).map(|i| vec![CopulaSaState::default(); i]).collect();
        self.last_pair_obs = pair_obs;
        self.last_marginal_obs = marg_obs;
        self.last_obs_weight = weight;
        self.selected = true;
    }

    /// Warm-up Gaussian R-vine update used during Phase 1a (`k < K_select`): keeps Gaussian
    /// marginals and updates Gaussian pair-copulas smoothly via Robbins-Monro SA.
    pub fn warmup_gaussian_update(&mut self, sampled_etas: &[Vec<f64>], gamma: f64) {
        if sampled_etas.is_empty() || gamma <= 0.0 {
            return;
        }
        self.update_omega_equiv(sampled_etas, gamma);
        for j in 0..self.d {
            let fixed = self.omega_fixed.get(j).copied().unwrap_or(false);
            if !fixed {
                let sd = self.omega_equiv.matrix[(j, j)]
                    .max(MARGINAL_SCALE_FLOOR * MARGINAL_SCALE_FLOOR)
                    .sqrt();
                self.marginals[j] = MarginalDist::normal(sd);
            }
        }
        // Keep a Gaussian D-vine synchronized with `omega_equiv` during warm-up so Phase 1a
        // targets the exact Gaussian prior while adapting correlations.
        if self.d >= 2 && self.rvine.columns[1][0].copula == CopulaFamily::Independence {
            for i in 1..self.d {
                for t in 0..i {
                    let orig_i = self.rvine.perm[i];
                    let orig_m = self.rvine.perm[self.rvine.columns[i][t].var2_int];
                    let fixed_i = self.omega_fixed.get(orig_i).copied().unwrap_or(false);
                    let fixed_m = self.omega_fixed.get(orig_m).copied().unwrap_or(false);
                    if !fixed_i && !fixed_m {
                        self.rvine.columns[i][t].copula = CopulaFamily::gaussian(0.0);
                    }
                }
            }
        }
        let u_samples = self.pit_samples(sampled_etas);
        self.last_pair_obs = self.rvine.forward_pseudo_obs_and_step(
            &u_samples,
            Some((&mut self.copula_sa, gamma)),
            &self.omega_fixed,
        );
    }

    fn update_omega_equiv(&mut self, sampled_etas: &[Vec<f64>], gamma: f64) {
        let d = self.d;
        if d == 0 || sampled_etas.is_empty() {
            return;
        }
        let mut eta_outer = DMatrix::zeros(d, d);
        for eta in sampled_etas {
            let ev = DVector::from_column_slice(eta);
            eta_outer += &ev * ev.transpose();
        }
        eta_outer /= sampled_etas.len() as f64;
        self.sample_s2 = (1.0 - gamma) * &self.sample_s2 + gamma * &eta_outer;

        let mut mat = self.sample_s2.clone();
        for i in 0..d {
            let fi = self.omega_fixed.get(i).copied().unwrap_or(false);
            if fi {
                let init_v = self.initial_omega.matrix[(i, i)];
                for j in 0..d {
                    if i != j {
                        mat[(i, j)] = 0.0;
                        mat[(j, i)] = 0.0;
                    }
                }
                mat[(i, i)] = init_v;
            } else if mat[(i, i)] < 1e-6 {
                mat[(i, i)] = 1e-6;
            }
        }
        let dense_mask = DMatrix::from_element(d, d, true);
        self.omega_equiv = OmegaMatrix::from_matrix_with_mask(
            mat,
            self.initial_omega.eta_names.clone(),
            false,
            dense_mask,
        );
    }

    /// Map a point `w in (0, 1)^d` from the unit hypercube into `eta in R^d` via exact
    /// R-vine inverse Rosenblatt + marginal quantiles (used by both PRNG and Sobol QMC IS!).
    pub fn draw_eta_from_uniforms(&self, w: &[f64]) -> Vec<f64> {
        if self.d == 0 {
            return Vec::new();
        }
        let u_orig = self.rvine.rosenblatt_inverse(w);
        (0..self.d)
            .map(|j| if self.zero_fixed[j] { 0.0 } else { self.marginals[j].quantile(u_orig[j]) })
            .collect()
    }

    /// Draw one `d`-dimensional `eta` vector using an RNG.
    pub fn draw_eta<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let w: Vec<f64> = (0..self.d).map(|_| Open01.sample(rng)).collect();
        self.draw_eta_from_uniforms(&w)
    }

    /// Joint density `p(eta) = exp(log_density(eta))`.
    pub fn density(&self, eta: &[f64]) -> f64 {
        self.log_density(eta).exp()
    }

    /// Pack all free unconstrained parameters `(psi_marginals, psi_copulas)` of `self`
    /// in canonical order for parameter-trajectory settling checks and joint Louis/BHHH covariance.
    pub fn pack_free_psi(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_free_params());
        for j in 0..self.d {
            if !self.omega_fixed.get(j).copied().unwrap_or(false) && !self.zero_fixed[j] {
                out.extend(self.marginals[j].pack_psi());
            }
        }
        for i in 1..self.d {
            for edge in &self.rvine.columns[i] {
                if edge.copula != CopulaFamily::Independence {
                    out.extend(edge.copula.pack_psi());
                }
            }
        }
        out
    }

    /// Evaluate `log_density(eta)` at a perturbed free unconstrained parameter vector `psi`
    /// without cloning `sample_s2` or observation buffers.
    fn log_density_at_free_psi(&self, eta: &[f64], psi: &[f64]) -> f64 {
        if self.d == 0 {
            return 0.0;
        }
        let mut pos = 0usize;
        let mut lp = 0.0_f64;
        let mut u = vec![0.5_f64; self.d];
        for j in 0..self.d {
            if self.zero_fixed[j] {
                continue;
            }
            let m = if !self.omega_fixed.get(j).copied().unwrap_or(false) {
                let k = self.marginals[j].n_params();
                let next = self.marginals[j].from_psi(&psi[pos..pos + k]);
                pos += k;
                next
            } else {
                self.marginals[j]
            };
            lp += m.log_pdf(eta[j]);
            u[j] = m.cdf(eta[j]).clamp(U_CLAMP, 1.0 - U_CLAMP);
        }
        if self.d > 1 {
            let u_int: Vec<f64> =
                (0..self.d).map(|a| u[self.rvine.perm[a]].clamp(U_CLAMP, 1.0 - U_CLAMP)).collect();
            let mut h1_out: Vec<Vec<f64>> = (0..self.d).map(|i| vec![0.0; i]).collect();
            let mut h2_out: Vec<Vec<f64>> = (0..self.d).map(|i| vec![0.0; i]).collect();
            for i in 1..self.d {
                let mut v_curr = u_int[i];
                for (t, edge) in self.rvine.columns[i].iter().enumerate() {
                    let cop = if edge.copula != CopulaFamily::Independence {
                        let k = edge.copula.n_params();
                        let next = edge.copula.from_psi(&psi[pos..pos + k]);
                        pos += k;
                        next
                    } else {
                        CopulaFamily::Independence
                    };
                    let v_part = RVine::read_partner(edge.partner_src, &u_int, &h1_out, &h2_out);
                    lp += cop.log_density(v_curr, v_part);
                    let next_h1 = cop.h1(v_curr, v_part);
                    let next_h2 = cop.h2(v_part, v_curr);
                    h1_out[i][t] = next_h1;
                    h2_out[i][t] = next_h2;
                    v_curr = next_h1;
                }
            }
        }
        lp
    }

    /// Total gradient `grad_psi ln p_vine(eta; psi)` with respect to [`Self::pack_free_psi`],
    /// including the Jacobian `du_j / dpsi_j` through the R-vine copula density.
    pub fn parameter_score(&self, eta: &[f64]) -> Vec<f64> {
        let mut psi = self.pack_free_psi();
        let n = psi.len();
        let mut score = vec![0.0_f64; n];
        for q in 0..n {
            let orig = psi[q];
            let h = 1e-5 * (1.0 + orig.abs());
            psi[q] = orig + h;
            let lp = self.log_density_at_free_psi(eta, &psi);
            psi[q] = orig - h;
            let lm = self.log_density_at_free_psi(eta, &psi);
            psi[q] = orig;
            let deriv = (lp - lm) / (2.0 * h);
            if deriv.is_finite() {
                score[q] = deriv;
            }
        }
        score
    }

    /// Extract a complete [`VineFitParams`] summary for `FitResult`, YAML, `.fitrx`, and R.
    pub fn to_fit_params(&self, eta_names: &[String]) -> VineFitParams {
        let name_of = |orig_idx: usize| -> String {
            eta_names.get(orig_idx).cloned().unwrap_or_else(|| format!("ETA_{}", orig_idx + 1))
        };

        let marginals: Vec<MarginalSummary> = (0..self.d)
            .map(|j| {
                if self.zero_fixed[j] {
                    MarginalSummary {
                        eta: name_of(j),
                        family: "normal".to_string(),
                        mean: 0.0,
                        sd: 0.0,
                        skewness: Some(0.0),
                        excess_kurtosis: Some(0.0),
                        params: vec![("sd".to_string(), 0.0)],
                        se: Vec::new(),
                    }
                } else {
                    let obs = self.last_marginal_obs.get(j).map_or(&[][..], Vec::as_slice);
                    let mut s =
                        self.marginals[j].to_summary(&name_of(j), obs, self.last_obs_weight);
                    if self.omega_fixed.get(j).copied().unwrap_or(false) {
                        s.se.clear();
                    }
                    s
                }
            })
            .collect();

        let mut trees: Vec<VineTreeSummary> = Vec::with_capacity(self.d.saturating_sub(1));
        for t in 0..self.d.saturating_sub(1) {
            let mut pairs = Vec::new();
            for i in (t + 1)..self.d {
                let edge = &self.rvine.columns[i][t];
                let orig_1 = self.rvine.perm[edge.var1_int];
                let orig_2 = self.rvine.perm[edge.var2_int];
                let cond_names: Vec<String> =
                    edge.cond_set_int.iter().map(|&c| name_of(self.rvine.perm[c])).collect();
                let label = if cond_names.is_empty() {
                    format!("{} ~ {}", name_of(orig_1), name_of(orig_2))
                } else {
                    format!("{} ~ {} | {}", name_of(orig_1), name_of(orig_2), cond_names.join(", "))
                };

                let cop = edge.copula;
                let (td_l, td_u) = cop.tail_dependence();
                let se = self
                    .last_pair_obs
                    .get(i)
                    .and_then(|col| col.get(t))
                    .map(|(u1, u2)| cop.observed_se(u1, u2, self.last_obs_weight))
                    .unwrap_or_default();

                pairs.push(VinePairEntry {
                    label,
                    edge_coord: (i, t),
                    copula: PairCopulaSummary {
                        family: cop.family_name().to_string(),
                        rotation: cop.rotation_degrees(),
                        params: cop.named_params(),
                        se,
                        kendall_tau: cop.kendall_tau(),
                        tail_dep_lower: td_l,
                        tail_dep_upper: td_u,
                    },
                });
            }
            trees.push(VineTreeSummary { tree: t + 1, pairs });
        }

        VineFitParams {
            marginals,
            trees,
            rvine: self.rvine.clone(),
            omega_fixed: self.omega_fixed.clone(),
            zero_fixed: self.zero_fixed.clone(),
        }
    }

    /// Reconstruct a usable [`VineCopulaOmega`] from its [`VineFitParams`] summary
    /// (for simulation, NPDE, prediction, and R density evaluation).
    pub fn from_fit_params(params: &VineFitParams) -> Self {
        let d = params.marginals.len();
        let eta_names: Vec<String> = params.marginals.iter().map(|m| m.eta.clone()).collect();
        let marginals: Vec<MarginalDist> =
            params.marginals.iter().map(MarginalDist::from_summary).collect();
        let rvine =
            if params.rvine.d == d { params.rvine.clone() } else { RVine::independence_dvine(d) };

        let zero_fixed: Vec<bool> = if params.zero_fixed.len() == d {
            params.zero_fixed.clone()
        } else {
            params.marginals.iter().map(|m| m.sd == 0.0).collect()
        };
        let omega_fixed: Vec<bool> = if params.omega_fixed.len() == d {
            params.omega_fixed.clone()
        } else {
            zero_fixed.clone()
        };

        let mut diag = DMatrix::zeros(d, d);
        for i in 0..d {
            diag[(i, i)] = if zero_fixed[i] {
                0.0
            } else {
                let sd = marginals[i].sd();
                sd * sd
            };
        }
        let omega = OmegaMatrix::from_matrix(diag.clone(), eta_names, true);

        Self {
            d,
            marginals,
            rvine,
            selected: true,
            allow_flexible_marginals: true,
            marginal_sa: vec![MarginalSaState::new(); d],
            copula_sa: (0..d).map(|i| vec![CopulaSaState::default(); i]).collect(),
            sample_s2: diag,
            omega_equiv: omega.clone(),
            initial_omega: omega,
            omega_fixed,
            zero_fixed,
            last_pair_obs: vec![Vec::new(); d],
            last_marginal_obs: vec![Vec::new(); d],
            last_obs_weight: 1.0,
        }
    }
}

impl RandomEffectDistribution for VineCopulaOmega {
    fn dim(&self) -> usize {
        self.d
    }

    fn prior_nll(&self, eta: &[f64]) -> f64 {
        let d = self.d;
        if d == 0 {
            return 0.0;
        }
        let half_ln_2pi = 0.5 * std::f64::consts::TAU.ln();
        let mut nll = 0.0_f64;
        let mut u = vec![0.5_f64; d];

        for j in 0..d {
            if self.zero_fixed[j] {
                continue;
            }
            // Strip -(1/2) ln(2*pi) per active dimension so prior_nll matches FOCE's
            // `0.5 * (eta' Omega^-1 eta + log|Omega|)` convention identically!
            nll += -self.marginals[j].log_pdf(eta[j]) - half_ln_2pi;
            u[j] = self.marginals[j].cdf(eta[j]).clamp(U_CLAMP, 1.0 - U_CLAMP);
        }
        if d > 1 {
            nll -= self.rvine.log_density(&u);
        }
        if nll.is_finite() {
            nll
        } else {
            crate::stats::likelihood::SUBJECT_SENTINEL_NLL
        }
    }

    fn log_density(&self, eta: &[f64]) -> f64 {
        let d = self.d;
        if d == 0 {
            return 0.0;
        }
        let mut lp = 0.0_f64;
        let mut u = vec![0.5_f64; d];
        for j in 0..d {
            if self.zero_fixed[j] {
                continue;
            }
            lp += self.marginals[j].log_pdf(eta[j]);
            u[j] = self.marginals[j].cdf(eta[j]).clamp(U_CLAMP, 1.0 - U_CLAMP);
        }
        if d > 1 {
            lp += self.rvine.log_density(&u);
        }
        if lp.is_finite() {
            lp
        } else {
            -crate::stats::likelihood::SUBJECT_SENTINEL_NLL
        }
    }

    fn prior_nll_grad(&self, eta: &[f64]) -> Vec<f64> {
        let d = self.d;
        let mut grad = vec![0.0_f64; d];
        if d == 0 {
            return grad;
        }
        let mut u = vec![0.5_f64; d];
        let mut pdf = vec![0.0_f64; d];
        for j in 0..d {
            if self.zero_fixed[j] {
                continue;
            }
            grad[j] = -self.marginals[j].d_log_pdf_d_eta(eta[j]);
            u[j] = self.marginals[j].cdf(eta[j]).clamp(U_CLAMP, 1.0 - U_CLAMP);
            pdf[j] = self.marginals[j].pdf(eta[j]);
        }
        if d > 1 {
            let mut u_work = u.clone();
            for j in 0..d {
                if self.zero_fixed[j] || pdf[j] <= 1e-300 {
                    continue;
                }
                let margin = (u[j] - U_CLAMP).min(1.0 - U_CLAMP - u[j]);
                let du = 1e-6_f64.min(0.45 * margin);
                if du <= 1e-15 {
                    continue;
                }
                u_work[j] = u[j] + du;
                let lp = self.rvine.log_density(&u_work);
                u_work[j] = u[j] - du;
                let lm = self.rvine.log_density(&u_work);
                u_work[j] = u[j];
                let d_log_c_du = (lp - lm) / (2.0 * du);
                if d_log_c_du.is_finite() {
                    grad[j] -= pdf[j] * d_log_c_du;
                }
            }
        }
        for g in &mut grad {
            if !g.is_finite() {
                *g = 0.0;
            }
        }
        grad
    }

    fn mstep_update(&mut self, sampled_etas: &[Vec<f64>], gamma: f64) {
        if sampled_etas.is_empty() || gamma <= 0.0 || self.d == 0 {
            return;
        }
        if !self.selected {
            self.select_structure_and_families(sampled_etas, sampled_etas.len(), 1.0);
            self.update_omega_equiv(sampled_etas, gamma);
            return;
        }

        self.update_omega_equiv(sampled_etas, gamma);

        // Stage 1: Robbins-Monro score/info SA update on each non-fixed zero-mean marginal.
        let mut marg_obs = vec![Vec::with_capacity(sampled_etas.len()); self.d];
        for eta in sampled_etas {
            for j in 0..self.d {
                marg_obs[j].push(eta[j]);
            }
        }
        for j in 0..self.d {
            let fixed = self.omega_fixed.get(j).copied().unwrap_or(false);
            if !fixed {
                self.marginal_sa[j].step(&mut self.marginals[j], &marg_obs[j], gamma);
            }
        }

        // Stage 2: Exact PIT -> Robbins-Monro score/info SA update on frozen R-vine pair-copulas.
        let u_samples = self.pit_samples(sampled_etas);
        self.last_pair_obs = self.rvine.forward_pseudo_obs_and_step(
            &u_samples,
            Some((&mut self.copula_sa, gamma)),
            &self.omega_fixed,
        );
        self.last_marginal_obs = marg_obs;
        self.last_obs_weight = 1.0;
    }

    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.draw_eta(rng)).collect()
    }

    fn proposal_chol(&self) -> &DMatrix<f64> {
        &self.omega_equiv.chol
    }

    fn marginal_sds(&self) -> Vec<f64> {
        (0..self.d).map(|j| self.marginals[j].sd().max(MARGINAL_SCALE_FLOOR)).collect()
    }

    fn to_omega_matrix(&self) -> &OmegaMatrix {
        &self.omega_equiv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Build a non-trivial 4-D non-D-vine (star / C-vine in Tree 0!) with mixed
    /// asymmetric rotated copulas (`Clayton90`, `Gumbel270`, `StudentT`, `Frank`)
    /// via Dissmann selection and verify BOTH:
    ///   1. `rosenblatt_forward(rosenblatt_inverse(w)) == w` pointwise to `< 1e-8`, AND
    ///   2. `|det D(rosenblatt_forward)(u)| == exp(rvine.log_density(u))` pointwise!
    #[test]
    fn rvine_4d_star_and_rotated_copulas_satisfy_rosenblatt_and_jacobian_identities() {
        let mut rng = StdRng::seed_from_u64(777);
        let d = 4;
        let n = 500;
        // Generate data where var 1 is the hub strongly coupled to vars 0, 2, 3 (a star in Tree 0,
        // NOT a natural-order path D-vine!), with negative asymmetric coupling on (1, 2).
        let c01 = CopulaFamily::clayton(2.8, CopulaRotation::R0);
        let c12 = CopulaFamily::clayton(2.4, CopulaRotation::R90);
        let c13 = CopulaFamily::gumbel(2.3, CopulaRotation::R270);
        let mut u_data = Vec::with_capacity(n);
        for _ in 0..n {
            let u1 = rng.random::<f64>().clamp(1e-4, 1.0 - 1e-4);
            let u0 = c01.h1_inv(rng.random::<f64>().clamp(1e-4, 1.0 - 1e-4), u1);
            let u2 = c12.h2_inv(rng.random::<f64>().clamp(1e-4, 1.0 - 1e-4), u1);
            let u3 = c13.h2_inv(rng.random::<f64>().clamp(1e-4, 1.0 - 1e-4), u1);
            u_data.push(vec![u0, u1, u2, u3]);
        }

        let (rvine, _) = select_rvine_dissmann(&u_data, 1.0, n, &[false; 4], false);

        // 1. Pointwise Rosenblatt round-trip for 50 random 4-D points!
        for _ in 0..50 {
            let w: Vec<f64> = (0..d).map(|_| rng.random::<f64>().clamp(0.02, 0.98)).collect();
            let u = rvine.rosenblatt_inverse(&w);
            let w_back = rvine.rosenblatt_forward(&u);
            for k in 0..d {
                assert!(
                    (w[k] - w_back[k]).abs() < 1e-7,
                    "4-D Rosenblatt mismatch at dim {k}: w={} vs w_back={}",
                    w[k],
                    w_back[k]
                );
            }
        }

        // 2. Pointwise Jacobian determinant of `rosenblatt_forward(u)` == `exp(log_density(u))`!
        //    This mathematically proves `log_density` and `rosenblatt_forward`/`inverse`
        //    describe the EXACT same 4-D probability distribution with zero indexing bugs!
        let eps = 1e-6;
        for _ in 0..20 {
            let u0: Vec<f64> = (0..d).map(|_| rng.random::<f64>().clamp(0.10, 0.90)).collect();
            let mut jac = DMatrix::<f64>::zeros(d, d);
            for j in 0..d {
                let mut up = u0.clone();
                let mut um = u0.clone();
                up[j] += eps;
                um[j] -= eps;
                let wp = rvine.rosenblatt_forward(&up);
                let wm = rvine.rosenblatt_forward(&um);
                for i in 0..d {
                    jac[(i, j)] = (wp[i] - wm[i]) / (2.0 * eps);
                }
            }
            let det_jac = jac.determinant().abs();
            let dens = rvine.log_density(&u0).exp();
            assert!(
                (det_jac - dens).abs() < 1e-4 * (1.0 + dens),
                "4-D Jacobian |det DR(u)| = {det_jac} != c_rvine(u) = {dens} at u={u0:?}"
            );
        }
    }

    #[test]
    fn vine_sample_uses_rvine_inverse_rosenblatt_not_gaussian_omega() {
        let omega = OmegaMatrix::from_diagonal(
            &[0.09, 0.16, 0.25],
            vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
        );
        let params = crate::types::ModelParameters {
            omega_fixed: vec![false; 3],
            ..crate::types::test_helpers::bare_params(omega)
        };
        let mut vine = VineCopulaOmega::from_init_params(&params, true);
        vine.marginals[0] = MarginalDist::skew_t(0.45, 5.0, 4.5);
        vine.rvine.columns[1][0].copula = CopulaFamily::clayton(3.0, CopulaRotation::R0);
        vine.selected = true;

        let mut rng = StdRng::seed_from_u64(99);
        let draws = vine.sample(3_000, &mut rng);
        let mean0: f64 = draws.iter().map(|e| e[0]).sum::<f64>() / draws.len() as f64;
        assert!(mean0.abs() < 0.03, "zero-mean skew-t sample mean should be ~0, got {mean0}");

        // Check round-trip serialization -> reconstruction preserves log_density exactly.
        let summary = vine.to_fit_params(&params.omega.eta_names);
        let rebuilt = VineCopulaOmega::from_fit_params(&summary);
        let pt = vec![0.15, -0.22, 0.08];
        assert!(
            (vine.log_density(&pt) - rebuilt.log_density(&pt)).abs() < 1e-12,
            "from_fit_params must preserve exact R-vine + marginal log_density"
        );

        // Verify prior_nll_grad matches central finite differences of prior_nll.
        let grad = vine.prior_nll_grad(&pt);
        let h = 1e-6;
        for j in 0..3 {
            let mut ep = pt.clone();
            let mut em = pt.clone();
            ep[j] += h;
            em[j] -= h;
            let fd = (vine.prior_nll(&ep) - vine.prior_nll(&em)) / (2.0 * h);
            assert!(
                (grad[j] - fd).abs() < 1e-6 * (1.0 + fd.abs()),
                "prior_nll_grad[{j}] = {} vs FD = {fd}",
                grad[j]
            );
        }
    }

    #[test]
    fn rvine_2d_to_5d_rosenblatt_and_jacobian_invariants() {
        let mut rng = StdRng::seed_from_u64(314159);
        for d in [2usize, 3, 5] {
            let n = 350;
            let mut u_data = Vec::with_capacity(n);
            for _ in 0..n {
                let u0 = rng.random::<f64>().clamp(1e-4, 1.0 - 1e-4);
                let mut row = vec![0.5; d];
                row[0] = u0;
                for j in 1..d {
                    let cop = match j % 4 {
                        1 => CopulaFamily::clayton(2.2, CopulaRotation::R0),
                        2 => CopulaFamily::gumbel(2.0, CopulaRotation::R90),
                        3 => CopulaFamily::frank(-4.5),
                        _ => CopulaFamily::student_t(0.65, 4.0),
                    };
                    row[j] = cop.h2_inv(rng.random::<f64>().clamp(1e-4, 1.0 - 1e-4), row[j - 1]);
                }
                u_data.push(row);
            }
            let (rvine, _) = select_rvine_dissmann(&u_data, 1.0, n, &vec![false; d], false);
            for _ in 0..20 {
                let w: Vec<f64> = (0..d).map(|_| rng.random::<f64>().clamp(0.03, 0.97)).collect();
                let u = rvine.rosenblatt_inverse(&w);
                let w_back = rvine.rosenblatt_forward(&u);
                for k in 0..d {
                    assert!(
                        (w[k] - w_back[k]).abs() < 1e-7,
                        "{d}-D Rosenblatt mismatch at dim {k}: {} vs {}",
                        w[k],
                        w_back[k]
                    );
                }
            }
            let eps = 1e-6;
            for _ in 0..10 {
                let u0: Vec<f64> = (0..d).map(|_| rng.random::<f64>().clamp(0.12, 0.88)).collect();
                let mut jac = DMatrix::<f64>::zeros(d, d);
                for j in 0..d {
                    let mut up = u0.clone();
                    let mut um = u0.clone();
                    up[j] += eps;
                    um[j] -= eps;
                    let wp = rvine.rosenblatt_forward(&up);
                    let wm = rvine.rosenblatt_forward(&um);
                    for i in 0..d {
                        jac[(i, j)] = (wp[i] - wm[i]) / (2.0 * eps);
                    }
                }
                let det_jac = jac.determinant().abs();
                let dens = rvine.log_density(&u0).exp();
                assert!(
                    (det_jac - dens).abs() < 2e-4 * (1.0 + dens),
                    "{d}-D Jacobian |det DR(u)| = {det_jac} != c_rvine(u) = {dens}"
                );
            }
        }
    }

    #[test]
    fn zero_fixed_eta_log_density_and_from_fit_params_roundtrip() {
        let names = vec!["ETA_CL".to_string(), "ETA_ZERO".to_string(), "ETA_V".to_string()];
        let mut mat = DMatrix::zeros(3, 3);
        mat[(0, 0)] = 0.09;
        mat[(1, 1)] = 0.0;
        mat[(2, 2)] = 0.04;
        let om = OmegaMatrix::from_matrix(mat, names.clone(), true);
        let mut init = crate::types::test_helpers::bare_params(om);
        init.omega_fixed = vec![false, true, true]; // ETA_ZERO is `0 FIX`, ETA_V is `0.04 FIX`

        let mut vine = VineCopulaOmega::from_init_params(&init, true);
        vine.marginals[0] = MarginalDist::skew_t(0.32, 1.8, 5.5);
        vine.rvine.set_edge_copula(2, 0, CopulaFamily::clayton(1.7, CopulaRotation::R0));
        vine.selected = true;

        let eta = [0.21, 0.0, -0.14];
        // Active dimension count is 2 (ETA_CL and ETA_V; ETA_ZERO is `0 FIX`).
        let expected_half_d_ln_2pi = 0.5 * 2.0 * std::f64::consts::TAU.ln();
        let lp = vine.log_density(&eta);
        let pnll = vine.prior_nll(&eta);
        assert!(
            (lp - (-pnll - expected_half_d_ln_2pi)).abs() < 1e-13,
            "log_density ({lp}) and prior_nll ({pnll}) must agree on d_active = 2"
        );
        assert_eq!(vine.n_free_params(), 4); // SkewT (3) + Clayton (1); ETA_ZERO and ETA_V are FIX

        // Round-trip through `VineFitParams` + serde JSON (including legacy JSON without flags).
        let summary = vine.to_fit_params(&names);
        assert_eq!(summary.n_free_params(), 4);
        assert_eq!(summary.marginals[1].sd, 0.0);

        let rebuilt = VineCopulaOmega::from_fit_params(&summary);
        assert!(rebuilt.is_zero_fixed(1));
        assert!(!rebuilt.is_zero_fixed(0));
        assert!(!rebuilt.is_zero_fixed(2));
        assert_eq!(rebuilt.n_free_params(), 4);
        assert!((rebuilt.log_density(&eta) - lp).abs() < 1e-13);

        let mut rng = StdRng::seed_from_u64(99);
        for draw in rebuilt.sample(25, &mut rng) {
            assert_eq!(draw[1], 0.0, "rebuilt 0-FIX ETA must sample exact 0.0");
            assert_ne!(draw[2], 0.0, "non-zero FIX ETA must retain its 0.2 SD");
        }

        // Also test legacy JSON missing `omega_fixed`/`zero_fixed` -> inferred from `sd == 0.0`.
        let mut legacy = summary;
        legacy.omega_fixed.clear();
        legacy.zero_fixed.clear();
        let rebuilt_legacy = VineCopulaOmega::from_fit_params(&legacy);
        assert!(rebuilt_legacy.is_zero_fixed(1));
        for draw in rebuilt_legacy.sample(10, &mut rng) {
            assert_eq!(draw[1], 0.0);
        }

        // Verify `parameter_score` matches central FD of `log_density`.
        let score = vine.parameter_score(&eta);
        assert_eq!(score.len(), 4);
        let base_psi = vine.pack_free_psi();
        for q in 0..score.len() {
            let mut p_up = base_psi.clone();
            let mut p_dn = base_psi.clone();
            let h = 1e-6;
            p_up[q] += h;
            p_dn[q] -= h;
            let fd = (vine.log_density_at_free_psi(&eta, &p_up)
                - vine.log_density_at_free_psi(&eta, &p_dn))
                / (2.0 * h);
            assert!((score[q] - fd).abs() < 1e-5 * (1.0 + fd.abs()));
        }
    }
}
