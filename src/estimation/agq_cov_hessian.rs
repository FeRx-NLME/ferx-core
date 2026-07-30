//! Exact analytic covariance R-matrix for the **FOCEI-anchored quadrature** (`method = focei`
//! with `n_agq > 1`) — issue #251, following the FOCE/FOCEI R-matrix of #436.
//!
//! # Why this is not #436 with a wider gate
//!
//! #436 assembles the second derivative of the FOCE/FOCEI marginal. AGQ minimises a
//! *different* objective: the quadrature marginal carries a grid whose centre and scale both
//! move with `x`, and a softmax over nodes whose weights move too. Neither has a FOCEI
//! analogue, so `analytic_cov_hessian` declines every `agq_nodes().is_some()` fit (#953 review
//! finding 1) rather than reporting standard errors for a likelihood the fit never optimised.
//! This module supplies the missing object.
//!
//! Only the **Gauss-Newton** anchor is in scope. `method = laplace` anchors on the exact
//! conditional Hessian `H = ∂²nll/∂b²`, which already contains `∂²f/∂η²`; its second
//! derivative would need **fourth**-order sensitivities. The Gauss-Newton anchor
//! `H̃ = Ω⁻¹ + Σⱼ pⱼaⱼaⱼᵀ` is built from *first* derivatives of `f`, so `∂²H̃/∂x²` needs only
//! third order — exactly what #436 already computes. That asymmetry is the whole reason this
//! is tractable for FOCEI-AGQ and not for Laplace.
//!
//! # The objective
//!
//! Matching `agq::agq_subject_nll` term for term, with `S ≡ H_reg = H̃ + Λ` the *regularised*
//! anchor and `Λ` the per-dimension jitter `importance_sampling::build_proposal` applies:
//!
//! ```text
//!   F   = ½·d·log π + ½·log|S| − LSE_j(t_j)
//!   t_j = log w_j + ‖z_j‖² − n_j ,        n_j = nll(b_j ; x)
//!   b_j = b̂(x) + √2 · M(x) · z_j ,        M ≡ L^{-T},  M Mᵀ = S⁻¹,  L Lᵀ = S
//! ```
//!
//! The Gauss–Hermite nodes `z_j` and weights `w_j` are constants. Everything that moves with
//! `x` moves through `b̂`, `S`, and the explicit θ/Ω/σ dependence of `nll`.
//!
//! Write `πⱼ = softmax(t)ⱼ` for the **softmax** weights — deliberately not `p`, which in
//! `sens_outer_gradient::ErrTerms` already means the residual weight inside `H̃`. Confusing the
//! two is the easiest way to write a plausible wrong assembly here.
//!
//! # First derivative
//!
//! With `u_{j,k} ≡ dn_j/dx_k` the node's **total** score,
//!
//! ```text
//!   u_{j,k} = s_{j,k} + g_jᵀ β_{j,k}
//!   s_{j,k} = ∂nll/∂x_k |_{b = b_j}          (fixed-b packed score)
//!   g_j     = ∇_b nll |_{b_j}
//!   β_{j,k} = db_j/dx_k = b̂_k + √2 · M_k · z_j
//! ```
//!
//! ```text
//!   ∂F/∂x_k = ½ tr(S⁻¹ S_k) + Σ_j π_j u_{j,k}
//! ```
//!
//! This is the same object `agq::agq_subject_packed_gradient` computes, which is a useful
//! cross-check: that one splits it as "analytic fixed-b score" plus a finite-differenced
//! `grid_response_correction`, and the two must agree.
//!
//! # Second derivative
//!
//! ```text
//!   ∂²F/∂x_k∂x_l = ½[ tr(S⁻¹ S_kl) − tr(S⁻¹ S_l S⁻¹ S_k) ]        (A) log-det curvature
//!                − Cov_π(u_k, u_l)                                 (B) softmax response
//!                + Σ_j π_j · ∂u_{j,k}/∂x_l                         (C) per-node curvature
//! ```
//!
//! Term (B) is `Σ_j π_j u_{j,k}u_{j,l} − ū_k ū_l` with `ū_k = Σ_j π_j u_{j,k}`. It is the
//! **only** term with no FOCEI counterpart, and it vanishes identically at one node (a single
//! softmax weight has zero variance) — which is the quadrature content, isolated.
//!
//! Term (C), per node, with `C_j = ∂²nll/∂x²|_b`, `M_j = ∂²nll/∂b∂x`, `H_j = ∂²nll/∂b²`, all
//! evaluated **at the node `b_j`, not at the mode**:
//!
//! ```text
//!   ∂u_{j,k}/∂x_l = C_j[k,l]
//!                 + M_j[:,k]ᵀ β_{j,l} + M_j[:,l]ᵀ β_{j,k}
//!                 + β_{j,l}ᵀ H_j β_{j,k}
//!                 + g_jᵀ ( b̂_kl + √2 · M_kl · z_j )
//! ```
//!
//! Symmetric in `(k,l)` by inspection: `C_j` is symmetric, the two `M_j` terms exchange, the
//! `H_j` form is symmetric, and `∂²b_j/∂x_k∂x_l` is. Asymmetry in an implementation of this is
//! therefore a bug, not a rounding artefact — worth asserting rather than symmetrising away.
//!
//! # Check: it must collapse onto #436 at one node
//!
//! At `n_agq = 1` the rule is `z = 0`, `w = √π`, so `b_1 = b̂`, `π_1 = 1`, and — decisively —
//! `g_1 = ∇_b nll(b̂) = 0` by stationarity of the mode. That kills the `g_jᵀ(b̂_kl + …)` term
//! outright, i.e. the entire second-derivative-of-the-mode and second-derivative-of-the-
//! Cholesky-factor machinery, which is the most expensive part of (C). Term (B) is zero. With
//! the implicit-function relation `b̂_k = −H⁻¹M_k`,
//!
//! ```text
//!   M_kᵀb̂_l + M_lᵀb̂_k + b̂_lᵀH b̂_k
//!     = −M_kᵀH⁻¹M_l − M_lᵀH⁻¹M_k + M_lᵀH⁻¹ H H⁻¹M_k
//!     = −M_kᵀH⁻¹M_l
//! ```
//!
//! so
//!
//! ```text
//!   ∂²F/∂x_k∂x_l  =  ½[tr(S⁻¹S_kl) − tr(S⁻¹S_lS⁻¹S_k)]  +  C[k,l]  −  M_kᵀH⁻¹M_l
//! ```
//!
//! which is exactly the structure of [`crate::estimation::sens_cov_hessian`]: the fixed-`b`
//! Hessian, minus the M2 envelope term `−M_ξᵀH⁻¹M_ζ`, plus the log-determinant curvature.
//! The derivation therefore reduces to the covariance that shipped in #436 — the strongest
//! available check on it. `agq_cov_hessian_reduces_to_focei_at_one_node` will be the first test
//! written against the assembly (step 5 below), before the multi-node path is trusted at all;
//! it does not exist yet, because the assembly does not.
//!
//! # The jitter is not optional
//!
//! `build_proposal` regularises with a per-dimension **relative** jitter
//! `Λᵢᵢ = max(1e-6·|H̃ᵢᵢ|, 1e-10)`, so the objective's log-determinant is `½log|H̃ + Λ|`. On
//! the 3-η warfarin fixture that is a systematic `2.1e-6` offset from `½log|H̃|`
//! (`agq_one_node_gauss_newton_is_the_focei_marginal_up_to_the_proposal_jitter` measures it).
//! Differentiating `H̃` and calling the result exact would be wrong at that same relative
//! order — and it would present as a consistent bias no step-size tuning removes, i.e. the
//! failure mode that survives every convergence test.
//!
//! In the ordinary branch (`H̃ᵢᵢ > 0` and above the floor) the jitter is *linear* in `H̃`:
//!
//! ```text
//!   S    = H̃ + 1e-6·diag(H̃)
//!   S_k  = H̃_k  + 1e-6·diag(H̃_k)
//!   S_kl = H̃_kl + 1e-6·diag(H̃_kl)
//! ```
//!
//! so carrying it costs nothing.
//!
//! Both *branches* of the `max` are individually smooth — that is worth stating precisely,
//! because the loose version of this claim ("the floor branch is non-differentiable") is wrong
//! and would narrow the scope for no reason:
//!
//! * relative branch (`1e-6·|H̃ᵢᵢ| > 1e-10`, i.e. `|H̃ᵢᵢ| > 1e-4`): `Λᵢᵢ = 1e-6·sgn·H̃ᵢᵢ`, so
//!   `Λ'ᵢᵢ = 1e-6·sgn·H̃'ᵢᵢ` — linear.
//! * floor branch (`|H̃ᵢᵢ| < 1e-4`): `Λᵢᵢ = 1e-10`, a **constant**, so `Λ'ᵢᵢ = 0` — also linear.
//!
//! What is not differentiable is the *crossing* at `|H̃ᵢᵢ| = 1e-4`, and the sign flip at
//! `H̃ᵢᵢ = 0`. Both are handled by declining when a diagonal sits within a factor
//! [`JITTER_BRANCH_MARGIN`] of either, which for a real model is never: an `H̃` diagonal is a
//! curvature in the packed coordinates and runs `O(1)`–`O(10²)`, six-plus orders above the
//! crossing.
//!
//! And if `build_proposal` fell back to the broad `Ω⁻¹` proposal (`H_reg` not
//! positive-definite), `S` is a different function of `x` altogether; the derivative has to
//! follow the branch actually taken, so that case declines too.
//!
//! # Ingredients, and where they come from
//!
//! | quantity | source |
//! |---|---|
//! | `aⱼ = ∂f/∂η`, `∂²f/∂η²`, `∂²f/∂η∂θ` | [`crate::sens::provider::ObsSens`] |
//! | `pⱼ`, `βⱼ = dp/df`, `αⱼ`, `α'ⱼ` | `ErrTerms` |
//! | `Hⱼ = ∂²nll/∂b²` at an **arbitrary** `b` | `score_core`'s `h_inner` (exposed by #951) |
//! | `Mⱼ = ∂²nll/∂b∂x` | `mixed_eta_theta` (θ) + the σ/Ω chains in `sigma_block`/`omega_block` |
//! | `b̂_k` | `sens_outer_gradient::subject_eta_dx` |
//! | `∂³f/∂η³`, `∂³f/∂η²∂θ`, `∂³f/∂η∂θ²`, `∂²f/∂θ²` | `provider::subject_sensitivities_cov` (#436) |
//! | `α''`, `β'` | `sens_cov_hessian::err_d2` (#436) |
//! | `H̃_k`, `H̃_kl` as explicit matrices | **new** — assembled below |
//! | `b̂_kl` | **new** — second differentiation of the IFT relation |
//! | `M_k`, `M_kl` (Cholesky-factor derivatives) | **new** |
//!
//! `H̃_k` is a total derivative — `H̃` is evaluated at `b̂(x)`, so the mode response is part of
//! it, exactly as `grid_response_correction` moves the mode when it differences the anchor:
//!
//! ```text
//!   H̃_k = (Ω⁻¹)_k + Σ_j [ ṗ_{j,k}·aⱼaⱼᵀ + pⱼ·(ȧ_{j,k}aⱼᵀ + aⱼȧ_{j,k}ᵀ) ]
//!   ṗ_{j,k} = βⱼ·ḟ_{j,k} + ∂pⱼ/∂σ_k          ḟ_{j,k} = ∂fⱼ/∂x_k|_b + aⱼᵀb̂_k
//!   ȧ_{j,k} = ∂aⱼ/∂x_k|_b + (∂²fⱼ/∂b²)·b̂_k
//!   (Ω⁻¹)_k = −Ω⁻¹ Ω_k Ω⁻¹
//! ```
//!
//! and `b̂_kl` from differentiating `∇_b nll(b̂(x); x) = 0` a second time:
//!
//! ```text
//!   b̂_kl = −H⁻¹ [ ∂³nll/∂b∂x_k∂x_l
//!                + (∂³nll/∂b²∂x_l)·b̂_k + (∂³nll/∂b²∂x_k)·b̂_l
//!                + ∂³nll/∂b³[b̂_k, b̂_l] ]
//! ```
//!
//! Every third derivative of `nll` reduces to a third derivative of `f` (from #436) contracted
//! with the residual scalars `α'', β'` (also #436). **No fourth-order sensitivity appears
//! anywhere** — that is the structural claim this module rests on.
//!
//! # Cost
//!
//! `G + O(N)` provider evaluations per subject (`G = n_agq^d` nodes, `N = n_theta + n_eta`),
//! once per fit at the converged point. The finite-difference stencil it replaces costs
//! `~2·n_free²` reconverged population objectives, **each** of which sweeps all `G` nodes for
//! every subject. So the saving is `O(n_free²) → O(1)` in the number of grid sweeps, which is
//! the quantity that actually hurts as `n_agq` grows.

// # Implementation route — revised after nlmixr2est#787
//
// nlmixr2est ships this (`est = "agq"`, `covType = "analytic"`) as
//
//     R_i = ld + E_π[Φ_pq] − Cov_π(Φ_p, Φ_q)
//
// which is the three-term result above, independently derived: `ld` is (A), `E_π[Φ_pq]` is (C),
// `Cov_π` is (B). The framing there is better than the one used to derive it here, and it
// deletes the largest piece of planned work:
//
//   > The AGQ objective is the FOCEi objective with **one term swapped**: the data half becomes
//   > `−log(Σ_k a_k)` over the quadrature nodes; the `log|Ht|`, Omega and tbs terms are
//   > **unchanged**.
//
// So `ld` is the FOCEI log-determinant half **reused verbatim**. There is no need to build `H̃_k`
// and `H̃_kl` as explicit matrices for term (A) — which would have meant factoring `∂p/∂σ` and
// `∂Ω⁻¹/∂x_k` out of `sigma_block`/`omega_block`, both of which currently form only contractions.
//
// ferx already has the seam. `sens_cov_hessian` splits the FOCEI Hessian exactly this way:
//
//   * `subject_cov_hessian_m2_natural` → `∂²Φ/∂ξ∂ζ|_η̂ − M_ξᵀH⁻¹M_ζ` — the **data half**,
//     mode response included. This is the piece AGQ replaces.
//   * the M3 block → the Hessian of `½log|H̃|`. This is `ld`, and it carries over unchanged.
//
// The correspondence is exact rather than approximate: `Φ_p` is the per-node *total* score
// `u_{j,k}` (mode movement included), so at one node `E_π[Φ_pq]` collapses to
// `∂²Φ/∂ξ∂ζ − M_ξᵀH⁻¹M_ζ` — literally `subject_cov_hessian_m2_natural` — and `Cov_π` vanishes.
// That is the same `n_agq = 1` reduction derived above, arrived at from the assembly side.
//
// Remaining steps, each landing behind its own FD parity test before the next builds on it (the
// repo's analytic-sensitivity rule — a wrong sensitivity here compiles and runs silently):
//
//   2. Per-node `Φ_p` and `Φ_pq` at `b_j`, reusing `score_core_at` (#951) for `H_j` and
//      `mixed_eta_theta` for `M_j`.
//   3. `b̂_kl`, from differentiating `∇_b nll(b̂(x); x) = 0` twice; vs FD of `subject_eta_dx`.
//   4. `M_k`, `M_kl` — Cholesky-factor derivatives from `S_k`, `S_kl`. Still needed: the node
//      placement depends on them even though `ld` does not. nlmixr2est#787 verifies its
//      equivalent (`d2Ginv`) to 1e-9 and Clairaut-symmetric to 9e-19; match that.
//   5. Assemble `ld + E_π[Φ_pq] − Cov_π`, gate on `hessian_anchor() == GaussNewton` plus #953's
//      scope, and wire into `compute_covariance`.
//
// A `√2` caution from the sibling PR nlmixr2est#785 ("AGQ quadrature node scaling converges to
// the wrong limit"): the `√2` must be carried through the node placement, the log-weight untilt
// **and** the `dGinv`/`d2Ginv` node-displacement terms in lockstep. The `n_agq = 1` identity test
// cannot see an error here — its node is `z = 0`, so `√2` drops out — so step 4 needs a
// multi-node check of its own.

use nalgebra::DMatrix;

/// Absolute jitter floor in `build_proposal`: `Λᵢᵢ = max(1e-6·|H̃ᵢᵢ|, JITTER_FLOOR)`.
///
/// Mirrored here rather than imported because `build_proposal` writes both constants inline.
/// [`regularised_anchor_matches_build_proposal`] pins the two together, so a change there
/// fails loudly here instead of silently differentiating a different `S`.
const JITTER_FLOOR: f64 = 1e-10;

/// Relative jitter coefficient in `build_proposal`.
const JITTER_REL: f64 = 1e-6;

/// The two `max` branches cross at `|H̃ᵢᵢ| = JITTER_FLOOR / JITTER_REL = 1e-4`.
const JITTER_CROSSING: f64 = JITTER_FLOOR / JITTER_REL;

/// How far a diagonal must sit from the branch crossing (and from zero) for `S` to be
/// differentiable in the ordinary sense.
///
/// A factor of 10 either side. This is not a tolerance to be tuned: at the crossing the second
/// derivative of `Λ` is a delta, so nothing finite is correct there, and near it the *third*
/// derivative that term (A) contracts against is unbounded. Declining is the only honest
/// answer. Real models are nowhere near — an `H̃` diagonal is a packed-coordinate curvature of
/// order `1`–`10²`, versus a crossing at `1e-4`.
const JITTER_BRANCH_MARGIN: f64 = 10.0;

/// Which branch of `build_proposal`'s `max` a given diagonal is on, and hence how `Λᵢᵢ`
/// responds to `H̃ᵢᵢ`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum JitterBranch {
    /// `Λᵢᵢ = JITTER_REL·sgn·H̃ᵢᵢ`; carries `sgn` so the `abs` is differentiated correctly.
    Relative { sign: f64 },
    /// `Λᵢᵢ = JITTER_FLOOR`, a constant — so this diagonal contributes **nothing** to `S_k`.
    Floor,
}

impl JitterBranch {
    /// `dΛᵢᵢ/dH̃ᵢᵢ` — the linear coefficient this branch contributes.
    #[inline]
    fn coefficient(self) -> f64 {
        match self {
            JitterBranch::Relative { sign } => JITTER_REL * sign,
            JitterBranch::Floor => 0.0,
        }
    }
}

/// `S = H̃ + Λ` together with the linear map that turns any `H̃` derivative into the matching
/// `S` derivative.
///
/// The point of bundling them is that the *same* branch decision governs `S`, `S_k` and `S_kl`.
/// Recomputing it per derivative order would let them disagree, which is exactly the class of
/// error that produces a plausible wrong Hessian.
#[derive(Debug, Clone)]
pub(crate) struct RegularisedAnchor {
    /// `S = H̃ + Λ` — bit-identical to the matrix `build_proposal` Cholesky-factors.
    pub(crate) s: DMatrix<f64>,
    /// Per-diagonal `dΛᵢᵢ/dH̃ᵢᵢ`.
    coef: Vec<f64>,
}

impl RegularisedAnchor {
    /// Propagate a derivative of `H̃` (to any order) through the jitter:
    /// `S_• = H̃_• + diag(coefᵢ · H̃_•[i,i])`.
    ///
    /// Exact, because the jitter is linear on each branch. Applies unchanged to `S_k` and
    /// `S_kl` — the linearity is why the second derivative needs no extra term.
    pub(crate) fn propagate(&self, dh: &DMatrix<f64>) -> DMatrix<f64> {
        let mut ds = dh.clone();
        for (i, &c) in self.coef.iter().enumerate() {
            ds[(i, i)] += c * dh[(i, i)];
        }
        ds
    }
}

/// Build `S = H̃ + Λ` and its jitter response, or `None` when the anchor is not
/// differentiable — the caller then keeps the finite-difference covariance.
///
/// Declines in exactly three situations, each for a stated reason rather than out of caution:
///
/// * a diagonal within [`JITTER_BRANCH_MARGIN`] of the branch crossing `|H̃ᵢᵢ| = 1e-4`, where
///   `Λ''` is a delta and no finite answer is right;
/// * a diagonal within the same margin of zero, where the `abs` flips sign;
/// * `S` not positive-definite, because `build_proposal` then silently substitutes the broad
///   `Σ = Ω` proposal — a **different function of `x`** — and differentiating `H̃ + Λ` would be
///   answering about an objective the fit never evaluated.
pub(crate) fn regularised_anchor(htilde: &DMatrix<f64>) -> Option<RegularisedAnchor> {
    let d = htilde.nrows();
    if d == 0 || htilde.ncols() != d {
        return None;
    }
    let mut s = htilde.clone();
    let mut coef = Vec::with_capacity(d);
    for i in 0..d {
        let hii = htilde[(i, i)];
        if !hii.is_finite() {
            return None;
        }
        let a = hii.abs();
        // Bail near the sign flip and near the branch crossing.
        if a < JITTER_CROSSING * JITTER_BRANCH_MARGIN && a > JITTER_CROSSING / JITTER_BRANCH_MARGIN
        {
            return None;
        }
        let branch = if a * JITTER_REL > JITTER_FLOOR {
            if a < JITTER_CROSSING / JITTER_BRANCH_MARGIN {
                // Unreachable given the guard above, but keep the invariant explicit.
                return None;
            }
            JitterBranch::Relative {
                sign: if hii >= 0.0 { 1.0 } else { -1.0 },
            }
        } else {
            JitterBranch::Floor
        };
        // Reproduce `build_proposal`'s arithmetic exactly, in its order, rather than via the
        // branch — so `s` is bit-identical to the matrix production factorises.
        s[(i, i)] += (JITTER_REL * a).max(JITTER_FLOOR);
        coef.push(branch.coefficient());
    }
    // `build_proposal` falls back to the broad proposal when this fails; see above.
    s.clone().cholesky()?;
    Some(RegularisedAnchor { s, coef })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::importance_sampling::build_proposal;

    fn spd(d: usize, scale: f64) -> DMatrix<f64> {
        let mut m = DMatrix::<f64>::zeros(d, d);
        for i in 0..d {
            for j in 0..d {
                m[(i, j)] = if i == j {
                    scale * (i as f64 + 2.0)
                } else {
                    0.1 * scale / ((i + j) as f64 + 2.0)
                };
            }
        }
        m
    }

    /// `regularised_anchor` must produce the **same** `S` that `build_proposal` factorises.
    ///
    /// This is the load-bearing test of this step. Everything downstream differentiates `S`; if
    /// `S` is not the matrix the objective actually used, every derivative is of the wrong
    /// function and no amount of FD parity elsewhere would reveal it — the FD reference would
    /// be differentiating my `S` too.
    ///
    /// Compared through `log|S|`, which is what `build_proposal` exposes and precisely what
    /// enters the objective as `½·log_det_inv_scale`.
    #[test]
    fn regularised_anchor_matches_build_proposal() {
        for d in 1..=4 {
            for scale in [1e-2, 1.0, 25.0, 1e3] {
                let h = spd(d, scale);
                let omega_inv = DMatrix::<f64>::identity(d, d);
                let reg = regularised_anchor(&h).expect("well-conditioned anchor is in scope");
                let proposal = build_proposal(&h, &omega_inv, d).expect("proposal");

                let mine = reg.s.clone().cholesky().expect("S is PD").l();
                let log_det_mine = 2.0 * (0..d).map(|i| mine[(i, i)].ln()).sum::<f64>();
                assert!(
                    (log_det_mine - proposal.log_det_inv_scale).abs() < 1e-12,
                    "d={d} scale={scale}: log|S| {log_det_mine} != build_proposal's {}",
                    proposal.log_det_inv_scale
                );
            }
        }
    }

    /// The jitter really is applied — `S != H̃`. Guards against the whole step becoming a
    /// no-op, which would make every downstream derivative silently wrong at the `1e-6`
    /// relative order the identity test measured.
    #[test]
    fn regularised_anchor_actually_jitters() {
        let h = spd(3, 1.0);
        let reg = regularised_anchor(&h).unwrap();
        for i in 0..3 {
            let delta = reg.s[(i, i)] - h[(i, i)];
            let want = JITTER_REL * h[(i, i)].abs();
            // Tolerance is set by the cancellation in `s − h`, not by the jitter's own size:
            // `s = h·(1 + 1e-6)`, so recovering a `1e-6`-relative quantity by subtracting two
            // `O(h)` numbers costs ~ULP(h) ≈ 4.4e-16 at h ≈ 2. A `1e-18` bound is unreachable
            // by construction — it would be testing float arithmetic, not the jitter.
            let tol = 8.0 * f64::EPSILON * h[(i, i)].abs();
            assert!(
                (delta - want).abs() <= tol,
                "diagonal {i}: jitter {delta} != {want} (tol {tol:.2e})"
            );
        }
        // Off-diagonals untouched.
        assert!((reg.s[(0, 1)] - h[(0, 1)]).abs() < 1e-18);
    }

    /// `propagate` is the exact derivative of `S` with respect to `H̃`, checked against a
    /// central difference of `regularised_anchor` itself along a perturbation direction.
    ///
    /// Exact to machine precision rather than to a truncation tolerance, because the map is
    /// linear — so the assertion is tight enough to catch a wrong coefficient (`1e-6` vs
    /// `2e-6`, or a missed sign) that a loose tolerance would pass.
    #[test]
    fn propagate_is_the_exact_derivative_of_s() {
        let d = 3;
        let h = spd(d, 4.0);
        let reg = regularised_anchor(&h).unwrap();

        // An arbitrary non-symmetric-free direction, including diagonal movement.
        let mut dir = DMatrix::<f64>::zeros(d, d);
        for i in 0..d {
            for j in 0..d {
                dir[(i, j)] = 0.3 * (i as f64 + 1.0) - 0.17 * (j as f64);
            }
        }

        let step = 1e-3;
        let sp = regularised_anchor(&(&h + step * &dir)).unwrap().s;
        let sm = regularised_anchor(&(&h - step * &dir)).unwrap().s;
        let fd = (sp - sm) / (2.0 * step);
        let analytic = reg.propagate(&dir);

        for i in 0..d {
            for j in 0..d {
                assert!(
                    (analytic[(i, j)] - fd[(i, j)]).abs() < 1e-12,
                    "dS[{i},{j}]: analytic {} vs FD {}",
                    analytic[(i, j)],
                    fd[(i, j)]
                );
            }
        }
    }

    /// A diagonal sitting on the branch crossing declines, and one safely on the **floor**
    /// branch does not.
    ///
    /// The second half matters: an earlier draft of the module doc claimed the floor branch was
    /// non-differentiable, which would have narrowed the scope for no reason. `Λᵢᵢ` is
    /// *constant* there, so it is perfectly differentiable — with zero derivative.
    #[test]
    fn declines_only_at_the_branch_crossing_not_on_the_floor() {
        let d = 2;
        // On the crossing (|H̃ᵢᵢ| = 1e-4) → decline.
        let mut h = DMatrix::<f64>::identity(d, d);
        h[(0, 0)] = JITTER_CROSSING;
        assert!(
            regularised_anchor(&h).is_none(),
            "a diagonal on the jitter branch crossing must decline"
        );

        // Well below the crossing → floor branch, in scope, zero jitter response.
        let mut h = DMatrix::<f64>::identity(d, d);
        h[(0, 0)] = JITTER_CROSSING / (JITTER_BRANCH_MARGIN * 10.0);
        let reg = regularised_anchor(&h).expect("floor branch is differentiable");
        assert_eq!(
            reg.coef[0], 0.0,
            "floor branch must have zero jitter response"
        );
        assert!(
            reg.coef[1] > 0.0,
            "the other diagonal is on the relative branch"
        );
        // And the floor really is applied as a constant.
        assert!((reg.s[(0, 0)] - (h[(0, 0)] + JITTER_FLOOR)).abs() < 1e-18);
    }

    /// A non-positive-definite anchor declines, because `build_proposal` substitutes the broad
    /// `Σ = Ω` proposal there — a different function of `x`, whose derivative is not what this
    /// module computes.
    #[test]
    fn declines_when_build_proposal_would_fall_back_to_the_broad_proposal() {
        let d = 2;
        let mut h = DMatrix::<f64>::identity(d, d);
        h[(0, 0)] = -5.0; // indefinite
        h[(1, 1)] = 3.0;
        assert!(
            regularised_anchor(&h).is_none(),
            "an indefinite anchor must decline rather than differentiate H̃ + Λ"
        );
        // Confirm the premise: `build_proposal` does still return something there (the broad
        // proposal), so declining is a real choice and not merely mirroring a `None`.
        let omega_inv = DMatrix::<f64>::identity(d, d);
        assert!(
            build_proposal(&h, &omega_inv, d).is_some(),
            "premise: build_proposal falls back rather than failing, which is why we decline"
        );
    }
}
