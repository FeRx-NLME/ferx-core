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
//! ## The `b̂_kl` term does not vanish by Bartlett — do not drop it
//!
//! `agq::grid_response_correction`'s doc decomposes the *first* derivative as
//!
//! ```text
//!   dF/dx = ∂Φ/∂x|_H  +  ∂Φ/∂H·dH/dx  +  ∂Φ/∂η̂·dη̂/dx
//! ```
//!
//! and notes the third factor is `Σ_j π_j ∇_b nll(b_j)` — the posterior-mean score — which is
//! "zero by the Bartlett identity". Read quickly, that invites the conclusion that every `b̂`
//! response drops out here too, taking [`mode_second_derivative`] with it. It does not.
//!
//! Bartlett gives `∫ p(b)∇_b nll(b) db = 0` for the **exact** posterior. What appears in `F` is
//! the *quadrature sum* `Σ_j π_j g_j`, which equals that integral only up to the rule's own
//! error. It is exactly zero at `n_agq = 1`, where `π_1 = 1` and `g_1 = ∇_b nll(b̂) = 0` by
//! stationarity — the same fact that drives the reduction above — but at `n_agq > 1` it is a
//! small non-zero residual, and `F`'s derivative is what it is regardless of what the underlying
//! marginal's would be. Since the whole point of this module is `n_agq > 1`, the term stays.
//!
//! (The existing gradient is not affected by this: `grid_response_correction` rebuilds `η̂` *and*
//! `H` at the perturbed parameters, so its finite difference captures both responses together
//! and the Bartlett remark is an aside about one of them, not an omission.)
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
//! What is not differentiable is the *crossing* at `|H̃ᵢᵢ| = 1e-4`, handled by declining when a
//! diagonal sits within a factor [`JITTER_BRANCH_MARGIN`] of it — which for a real model is
//! never: an `H̃` diagonal is a curvature in the packed coordinates and runs `O(1)`–`O(10²)`,
//! six-plus orders above the crossing.
//!
//! The sign flip at `H̃ᵢᵢ = 0` is **not** a second non-differentiable point, and an earlier
//! version of this section wrongly said it was. Any diagonal near zero is far below the
//! crossing, hence on the floor branch, where `Λᵢᵢ = 1e-10` is a constant and `Λ'ᵢᵢ = 0`
//! regardless of the sign of `H̃ᵢᵢ` — the `abs` is never differentiated there. `sgn` is carried
//! in [`JitterBranch::Relative`] for the relative branch alone, where it is genuine.
//!
//! Near zero the real hazard is **conditioning**, not differentiability: `H̃ᵢᵢ = 1e-11` gives
//! `Sᵢᵢ ≈ 1e-10` and `S⁻¹` entries of order `1e10`, which term (A) then contracts against `S_k`
//! and `S_kl`. That is a property of the assembled Hessian, not of the jitter, so the guard
//! belongs at the step-5 gate where `S⁻¹` is actually formed — see [`regularised_anchor`], which
//! deliberately does not screen for it and says so.
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
// `Cov_π` is (B). The framing there is better than the one used to derive it here:
//
//   > The AGQ objective is the FOCEi objective with **one term swapped**: the data half becomes
//   > `−log(Σ_k a_k)` over the quadrature nodes; the `log|Ht|`, Omega and tbs terms are
//   > **unchanged**.
//
// So `ld` is the FOCEI log-determinant half **reused verbatim**, and term (A) needs no explicit
// `H̃_k`/`H̃_kl`.
//
// **This does not delete the work, and an earlier revision of this note claimed it did.** Step 4
// below still needs `M_k`, `M_kl` — derivatives of the Cholesky factor of `S` — because the node
// placement `b_j = b̂ + √2·M·z_j` depends on them. Those come from `S_k`, `S_kl`, and
// `S_• = H̃_• + Λ_•` requires `H̃_k`/`H̃_kl` as *actual matrices*: a Cholesky differential needs
// the full `dS`, not contractions of it, and no other square root (symmetric or otherwise) avoids
// that. So factoring `∂p/∂σ` and `∂Ω⁻¹/∂x_k` out of `sigma_block`/`omega_block` is still owed for
// any `n_agq > 1`. What #787 removes is their use in term (A) only.
//
// Two consequences, both of which cut against the route as first revised:
//
//   * The cost argument for abandoning the **split route** — `∂²F_focei/∂x²` from #436 plus FD of
//     the quadrature correction `Δ = F_agq − F_focei`, every ingredient of which exists today —
//     rested on that premise and is correspondingly weaker. It deserves re-deciding on its
//     merits before steps 2–4 are written, since under the split route they are not written at
//     all.
//   * Because `H̃_k` gets built regardless, carrying `Λ` into term (A) costs one diagonal scaling
//     ([`RegularisedAnchor::propagate`]) rather than a new derivation. Reusing FOCEI's `ld`
//     verbatim would leave a systematic `1e-6`-relative bias — FOCEI's log-det is `½log|H̃|`
//     (`likelihood.rs`, no jitter), AGQ's is `½log|H̃ + Λ|` — which an FD parity test cannot see,
//     the same trap class as the `√2` caution below. So take the `ld` *structure* from #787, but
//     evaluate it on `S`, not on `H̃`.
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

// Every item here is consumed only by this module's tests until the step-5 assembly lands.
// Without this, `pub mod` exported nine `dead_code` warnings into every non-test build of the
// crate and of downstream consumers (ferx-r's `src/rust`). Remove when `compute_covariance`
// calls `regularised_anchor`.
#![allow(dead_code)]

use nalgebra::{Cholesky, DMatrix, DVector, Dyn};

use crate::estimation::sens_cov_hessian::CovHessianParts;
use crate::types::{CompiledModel, ModelParameters, Subject};

/// Absolute jitter floor in `build_proposal`: `Λᵢᵢ = max(1e-6·|H̃ᵢᵢ|, JITTER_FLOOR)`.
///
/// Mirrored here rather than imported because `build_proposal` writes both constants inline.
/// The test `regularised_anchor_matches_build_proposal` pins the two together, so a change
/// there fails loudly here instead of silently differentiating a different `S`. That test
/// carries a dedicated floor-branch fixture — without one it would pin [`JITTER_REL`] only,
/// since every well-scaled anchor sits on the relative branch (a plain intra-doc link is not
/// used above because the target lives in `#[cfg(test)]` and cannot resolve under `cargo doc`).
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
    ///
    /// `sign = -1.0` cannot currently reach a returned [`RegularisedAnchor`]: a negative diagonal
    /// leaves `S` indefinite, so the definiteness screen rejects it first. It is kept, and
    /// covered by `coefficient_carries_the_sign_on_the_relative_branch`, because step 5 has to
    /// face the indefinite anchor — and the natural way to widen that scope (a modified Cholesky
    /// or an eigenvalue floor in place of the screen) would put this coefficient into service for
    /// the first time. An error there flips one diagonal's jitter response at the `1e-6` relative
    /// order, which a coarse FD parity check would not catch.
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
    /// The factor `L` with `L Lᵀ = S`, kept rather than discarded after the definiteness screen.
    ///
    /// The assembly needs exactly this: nodes are placed at `M = L⁻ᵀ`, and term (A)'s
    /// `tr(S⁻¹S_kl)` and `tr(S⁻¹S_lS⁻¹S_k)` both solve against it. Mirrors
    /// `importance_sampling::Proposal`, which likewise carries `chol_h` alongside its log-det.
    pub(crate) chol: Cholesky<f64, Dyn>,
    /// `L⁻¹`, formed once because every factor differential below needs it two or three times.
    l_inv: DMatrix<f64>,
    /// Per-diagonal `dΛᵢᵢ/dH̃ᵢᵢ`.
    coef: Vec<f64>,
}

/// `Φ(A)` — lower triangle of `A` with the diagonal halved.
///
/// The projector that makes a Cholesky differential well-posed. `dS = dL·Lᵀ + L·dLᵀ` determines
/// `dL` only once you impose that `L⁻¹dL` is lower triangular; `Φ` is exactly the operator that
/// splits a symmetric matrix into that lower-triangular part plus its transpose, so
/// `Φ(W) + Φ(W)ᵀ = W` for symmetric `W`.
fn half_tril(a: &DMatrix<f64>) -> DMatrix<f64> {
    let d = a.nrows();
    let mut out = DMatrix::<f64>::zeros(d, d);
    for i in 0..d {
        for j in 0..=i {
            out[(i, j)] = if i == j { 0.5 * a[(i, i)] } else { a[(i, j)] };
        }
    }
    out
}

impl RegularisedAnchor {
    /// Packed dimension `d` of the anchor.
    pub(crate) fn dim(&self) -> usize {
        self.coef.len()
    }

    /// Whether `S` is conditioned well enough for `S⁻¹` to be contracted against its own
    /// derivatives without the result being dominated by round-off.
    ///
    /// Estimated from the Cholesky diagonal: `cond(S) ≈ (max Lᵢᵢ / min Lᵢᵢ)²`, exact for a
    /// diagonal `S` and a sound lower bound generally, at no extra factorisation cost.
    ///
    /// The threshold is `1e6`, and it is set by term (A), not by `f64`. `tr(S⁻¹S_ξS⁻¹S_ζ)`
    /// contracts `S⁻¹` **twice**, so round-off is amplified by `cond²`: at `cond = 1e6` the
    /// relative error is `cond²·ε ≈ 2e-4`, which is about where an analytic Hessian stops being
    /// better than the finite-difference covariance it would replace. `1e8` would already give
    /// `≈ 2.2` — no correct digits at all. Anything near the `f64` limit is meaningless here.
    ///
    /// Real models sit far below: an `H̃` diagonal is a packed-coordinate curvature running
    /// `O(1)`–`O(10²)`, so `cond` is typically `1e2`–`1e4`. The two orders of headroom are for
    /// poorly-identified but legitimate fits; beyond that, declining to the FD covariance is the
    /// honest answer for a direction the data does not identify.
    ///
    /// This is a property of the assembled Hessian, not of the jitter, which is why
    /// [`regularised_anchor`] does not screen for it: `S` there is exactly what it claims to be
    /// and is differentiable regardless. The distinction matters because the two failures want
    /// different answers — a jitter branch crossing is *undefined*, an ill-conditioned anchor is
    /// well-defined but useless.
    pub(crate) fn is_well_conditioned(&self) -> bool {
        const MAX_COND: f64 = 1e6;
        let l = self.chol.l();
        let (mut lo, mut hi) = (f64::INFINITY, 0.0_f64);
        for i in 0..self.coef.len() {
            let d = l[(i, i)].abs();
            lo = lo.min(d);
            hi = hi.max(d);
        }
        lo > 0.0 && (hi / lo).powi(2) <= MAX_COND
    }

    /// `log|S|`, the quantity entering the objective as `½·log_det_inv_scale`.
    pub(crate) fn log_det(&self) -> f64 {
        let l = self.chol.l();
        2.0 * (0..self.coef.len()).map(|i| l[(i, i)].ln()).sum::<f64>()
    }

    /// Propagate a derivative of `H̃` (to any order) through the jitter:
    /// `S_• = H̃_• + diag(coefᵢ · H̃_•[i,i])`.
    ///
    /// Exact, because the jitter is linear on each branch. Applies unchanged to `S_k` and
    /// `S_kl` — the linearity is why the second derivative needs no extra term.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `dh` is `d × d`. The packed dimension is `n_eta + n_occ·n_kappa`, which
    /// under IOV varies from subject to subject, so an anchor cached across subjects and fed a
    /// smaller `dh` would otherwise index out of bounds — inside a rayon loop that aborts the
    /// process and loses a converged fit, instead of degrading to the FD covariance.
    pub(crate) fn propagate(&self, dh: &DMatrix<f64>) -> DMatrix<f64> {
        debug_assert_eq!(
            (dh.nrows(), dh.ncols()),
            (self.coef.len(), self.coef.len()),
            "propagate: derivative is {}×{} but the anchor is {d}×{d}",
            dh.nrows(),
            dh.ncols(),
            d = self.coef.len()
        );
        let mut ds = dh.clone();
        for (i, &c) in self.coef.iter().enumerate() {
            ds[(i, i)] += c * dh[(i, i)];
        }
        ds
    }

    /// `M = L⁻ᵀ`, the node-placement scale: `b_j = b̂ + √2·M·z_j`, and `M Mᵀ = S⁻¹`.
    pub(crate) fn node_scale(&self) -> DMatrix<f64> {
        self.l_inv.transpose()
    }

    /// `L_k = L·Φ(L⁻¹ S_k L⁻ᵀ)` — the first differential of the Cholesky factor.
    ///
    /// `ds` is `S_k`, i.e. the output of [`Self::propagate`] applied to `H̃_k`, **not** `H̃_k`
    /// itself. Feeding the unjittered derivative here is the `1e-6`-relative error the module doc
    /// warns about, and it is invisible to an FD parity check that differences the same wrong
    /// matrix.
    pub(crate) fn factor_derivative(&self, ds: &DMatrix<f64>) -> DMatrix<f64> {
        debug_assert!(
            is_symmetric(ds),
            "factor_derivative: S_k must be symmetric; a Cholesky differential is not defined \
             otherwise"
        );
        let w = &self.l_inv * ds * self.l_inv.transpose();
        self.chol.l() * half_tril(&w)
    }

    /// `L_kl = L·Φ(L⁻¹ S_kl L⁻ᵀ − A_k A_lᵀ − A_l A_kᵀ)` with `A_• = L⁻¹L_•`.
    ///
    /// The two `A` terms are the whole content of the second differential: along a path where
    /// `S` is linear (`S_kl = 0`) they are the entire answer, which is what
    /// `factor_second_derivative_matches_fd` exercises. Symmetric in `(k,l)` by construction —
    /// see `factor_second_derivative_is_clairaut_symmetric`.
    pub(crate) fn factor_second_derivative(
        &self,
        lk: &DMatrix<f64>,
        ll: &DMatrix<f64>,
        ds_kl: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        debug_assert!(
            is_symmetric(ds_kl),
            "factor_second_derivative: S_kl must be symmetric"
        );
        let ak = &self.l_inv * lk;
        let al = &self.l_inv * ll;
        let w = &self.l_inv * ds_kl * self.l_inv.transpose()
            - &ak * al.transpose()
            - &al * ak.transpose();
        self.chol.l() * half_tril(&w)
    }

    /// `M_k = −M L_kᵀ M`, from `d(L⁻ᵀ) = −L⁻ᵀ(dL)ᵀL⁻ᵀ`.
    pub(crate) fn node_scale_derivative(&self, lk: &DMatrix<f64>) -> DMatrix<f64> {
        let m = self.node_scale();
        -(&m * lk.transpose() * &m)
    }

    /// `M_kl = M L_lᵀ M L_kᵀ M + M L_kᵀ M L_lᵀ M − M L_klᵀ M`.
    ///
    /// Differentiating `M_k = −M L_kᵀ M` again: the two triple products come from `M_l` on either
    /// side, the last from `L_kl`. This is what carries the `√2` node displacement to second
    /// order — the term nlmixr2est#785 got wrong, and which the `n_agq = 1` identity test cannot
    /// see because its node is `z = 0`.
    pub(crate) fn node_scale_second_derivative(
        &self,
        lk: &DMatrix<f64>,
        ll: &DMatrix<f64>,
        lkl: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let m = self.node_scale();
        let mlk = &m * lk.transpose();
        let mll = &m * ll.transpose();
        &mll * &mlk * &m + &mlk * &mll * &m - &m * lkl.transpose() * &m
    }
}

/// `β_{j,k} = b̂_k + √2·M_k·z_j` — how node `j` moves when parameter `k` moves.
///
/// The `√2` is the one nlmixr2est#785 got wrong, and it has to be the *same* `√2` that placed the
/// node (`agq::agq_nodes_and_terms` applies it through `apply_l_sigma`). Placement and
/// displacement disagreeing would partly cancel and present as a small gradient error rather than
/// a failure — which is why this is one function rather than a constant repeated at both sites.
///
/// At `z = 0` this is exactly `b̂_k`, which is the whole of the `n_agq = 1` reduction: the grid
/// cannot move if there is only one node at the centre.
pub(crate) fn node_displacement(
    b_hat_k: &DVector<f64>,
    m_k: &DMatrix<f64>,
    z: &DVector<f64>,
) -> DVector<f64> {
    b_hat_k + std::f64::consts::SQRT_2 * (m_k * z)
}

/// Term (C) for one node, **less** the `g_jᵀ(b̂_kl + √2·M_kl·z_j)` tail:
///
/// ```text
///   Φ_pq(j)[k,l] = C_j[k,l] + M_j[:,k]ᵀβ_{j,l} + M_j[:,l]ᵀβ_{j,k} + β_{j,l}ᵀ H_j β_{j,k}
/// ```
///
/// Everything is evaluated **at the node** `b_j`, not at the mode — `C_j`/`M_j` come from
/// [`sens_cov_hessian::subject_cov_hessian_parts`] built on `sens`/`prep` at `b_j`, and `H_j`
/// from `agq::score_core_at`'s `h_inner` there.
///
/// The omitted tail is not an approximation: it is step 3's `b̂_kl` and step 4's `M_kl`, which
/// enter linearly and are added by the step-5 assembly. At the mode `g_j = 0` kills it outright,
/// so it is precisely the part that has no `n_agq = 1` counterpart — which is why splitting it
/// off keeps this function testable against #436 today.
///
/// [`sens_cov_hessian::subject_cov_hessian_parts`]: super::sens_cov_hessian::subject_cov_hessian_parts
pub(crate) fn node_curvature(
    c: &DMatrix<f64>,
    m: &[DVector<f64>],
    h: &DMatrix<f64>,
    beta: &[DVector<f64>],
) -> DMatrix<f64> {
    let dim = c.nrows();
    debug_assert_eq!(m.len(), dim, "one M vector per natural parameter");
    debug_assert_eq!(beta.len(), dim, "one β vector per natural parameter");
    let mut out = DMatrix::zeros(dim, dim);
    for k in 0..dim {
        let hbk = h * &beta[k];
        for l in 0..dim {
            out[(k, l)] = c[(k, l)] + m[k].dot(&beta[l]) + m[l].dot(&beta[k]) + beta[l].dot(&hbk);
        }
    }
    out
}

/// Term (B): `Cov_π(u_k, u_l) = Σ_j π_j u_{j,k}u_{j,l} − ū_k ū_l`.
///
/// `u[j][k]` is node `j`'s **total** score for parameter `k` (mode movement included) and `pi` the
/// softmax weights, both on the grid the objective actually evaluated. The assembly *subtracts*
/// this — see the sign in the module doc's term (B).
///
/// Vanishes identically at one node, where a single softmax weight has zero variance. That is not
/// a numerical accident but the defining property of the term, and
/// `softmax_covariance_vanishes_at_one_node` pins it.
pub(crate) fn softmax_covariance(pi: &[f64], u: &[Vec<f64>]) -> DMatrix<f64> {
    let dim = u.first().map_or(0, Vec::len);
    let mut mean = vec![0.0; dim];
    for (&w, uj) in pi.iter().zip(u.iter()) {
        for k in 0..dim {
            mean[k] += w * uj[k];
        }
    }
    let mut out = DMatrix::zeros(dim, dim);
    for (&w, uj) in pi.iter().zip(u.iter()) {
        for k in 0..dim {
            for l in 0..dim {
                out[(k, l)] += w * uj[k] * uj[l];
            }
        }
    }
    for k in 0..dim {
        for l in 0..dim {
            out[(k, l)] -= mean[k] * mean[l];
        }
    }
    out
}

/// Term (A): `½[tr(S⁻¹S_kl) − tr(S⁻¹S_l S⁻¹S_k)]`, the log-determinant curvature.
///
/// `s_d[k]` is `S_k` and `s_dd[k][l]` is `S_kl` — both **already through**
/// [`RegularisedAnchor::propagate`], i.e. jittered. Passing raw `H̃` derivatives here is the
/// `1e-6`-relative bias the module doc warns about, and no FD parity test would show it.
///
/// Uses the stored factor to solve rather than forming `S⁻¹`: `tr(S⁻¹A) = tr(chol.solve(A))`.
pub(crate) fn logdet_curvature(
    anchor: &RegularisedAnchor,
    s_d: &[DMatrix<f64>],
    s_dd: &[Vec<DMatrix<f64>>],
) -> DMatrix<f64> {
    let dim = s_d.len();
    // `S⁻¹S_k` once per parameter — the double loop below would otherwise re-solve `dim` times.
    let sinv_sd: Vec<DMatrix<f64>> = s_d.iter().map(|sk| anchor.chol.solve(sk)).collect();
    let mut out = DMatrix::zeros(dim, dim);
    for k in 0..dim {
        for l in 0..dim {
            let first = anchor.chol.solve(&s_dd[k][l]).trace();
            let second = (&sinv_sd[l] * &sinv_sd[k]).trace();
            out[(k, l)] = 0.5 * (first - second);
        }
    }
    out
}

/// Everything term (B) and term (C) need at one quadrature node.
///
/// The point of bundling is that all four come from **one** provider sweep at `b_j`. Building
/// them separately would evaluate the model three times per node, and `G = n_agq^d` nodes is
/// already the dominant cost.
pub(crate) struct NodeJet {
    /// `s_{j,ζ} = ∂nll/∂ζ|_b` — the fixed-`b` natural score.
    pub(crate) s: Vec<f64>,
    /// `g_j = ∇_b nll|_{b_j}`. Zero at the mode by stationarity; **not** zero at other nodes.
    pub(crate) g: DVector<f64>,
    /// `H_j = ∂²nll/∂b²|_{b_j}` — the *exact* conditional Hessian, not the Gauss-Newton anchor.
    /// Term (C)'s `β_lᵀH_jβ_k` is a curvature of `nll`, so it takes `h_inner` even though the
    /// grid that placed `b_j` was scaled by `H̃`.
    pub(crate) h: DMatrix<f64>,
    /// `C_j` and `M_j` at the node.
    pub(crate) parts: CovHessianParts,
}

/// Build [`NodeJet`] at an arbitrary `b`, not necessarily the mode.
///
/// Non-IOV only, matching #953's covariance scope (`analytic_cov_hessian` already declines
/// `n_kappa > 0`), so `b` is plain `η` and the prior precision is `Ω⁻¹`.
///
/// The load-bearing assumption is that none of the three sources is secretly mode-bound.
/// `agq::accumulate_fixed_eta_packed_gradient` states it for the provider — *"neither provider
/// entry point assumes the mode; the `η = η̂` requirement in `sens_outer_gradient` lives in its
/// `theta_block`, not here"* — and `theta_block` is exactly what this path does **not** call:
/// [`subject_cov_hessian_parts`] reads `prep.et`, `prep.omega_inv` and `z = Ω⁻¹·b`, all of which
/// are ordinary functions of the evaluation point. `node_jet_at_the_mode_reproduces_the_focei_parts`
/// pins the mode case against #436 so a future mode-only assumption in `prepare` fails loudly.
///
/// [`subject_cov_hessian_parts`]: super::sens_cov_hessian::subject_cov_hessian_parts
pub(crate) fn node_jet(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    b: &[f64],
) -> Option<NodeJet> {
    use crate::estimation::inner_optimizer::analytic_eta_nll_gradient_with_schedule;
    use crate::estimation::sens_cov_hessian::subject_cov_hessian_parts;
    use crate::estimation::sens_outer_gradient::{prepare, score_core};
    use crate::sens::provider::subject_sensitivities_cov;

    let sens = subject_sensitivities_cov(model, subject, &params.theta, b)?;
    let prep = prepare(model, subject, params, &sens, b)?;
    let core = score_core(
        model,
        subject,
        params,
        &sens,
        model.n_eta,
        &params.omega.inv,
        b,
        model.residual_error_eta,
    )?;
    let g = analytic_eta_nll_gradient_with_schedule(
        model,
        subject,
        &params.theta,
        b,
        &params.omega,
        &params.sigma.values,
        None,
        None,
    )?;
    let parts = subject_cov_hessian_parts(model, subject, params, &sens, &prep, b);
    let s = fixed_b_natural_score(model, subject, params, &sens, &prep, &core, b);
    Some(NodeJet {
        s,
        g: DVector::from_vec(g),
        h: core.h_inner,
        parts,
    })
}

/// The three terms of the AGQ covariance Hessian, kept separate.
///
/// Separate rather than summed because they are checked separately: at one node (B) must be
/// exactly zero and (C) must equal #436's M2 natural block, and a test that only saw the total
/// could not tell a sign error in one from a compensating error in another.
pub(crate) struct AgqCovTerms {
    /// (A) `½[tr(S⁻¹S_ζξ) − tr(S⁻¹S_ξS⁻¹S_ζ)]`.
    pub(crate) logdet: DMatrix<f64>,
    /// (B) `Cov_π(u_ζ, u_ξ)` — **subtracted** in the total.
    pub(crate) softmax: DMatrix<f64>,
    /// (C) `Σ_j π_j ∂u_{j,ζ}/∂ξ`.
    pub(crate) node: DMatrix<f64>,
    /// `∂F/∂ζ = ½tr(S⁻¹S_ζ) + Σ_j π_j u_{j,ζ}` — the AGQ natural **gradient**.
    ///
    /// Carried because the natural→packed chain needs the gradient of *the objective being
    /// differentiated*: its second-order reparameterisation term contracts `∂²(natural)/∂(packed)²`
    /// against it. Pairing this Hessian with `sens_outer_gradient::subject_natural_gradient`
    /// would silently mix in FOCEI's gradient, which differs from AGQ's by the quadrature
    /// correction — small, systematic, and invisible to any test that only checks the Hessian.
    pub(crate) grad: Vec<f64>,
}

impl AgqCovTerms {
    /// `(A) − (B) + (C)`.
    pub(crate) fn total(&self) -> DMatrix<f64> {
        &self.logdet - &self.softmax + &self.node
    }
}

/// The per-subject AGQ covariance Hessian in natural `[θ, Ω, σ]` space.
///
/// `grid[j]` is node `j`'s `d`-vector of Gauss-Hermite abscissae `z_j` and `pi[j]` its softmax
/// weight — both supplied by the caller, which already builds them for the objective
/// (`agq::agq_nodes_and_terms`). Taking them as inputs rather than rebuilding keeps this function
/// differentiating *the grid the objective actually evaluated*, which is the whole reason the
/// jitter is carried, and makes the `n_agq = 1` case directly constructible in a test.
///
/// # Order of accuracy
///
/// **Fully analytic in the parameters.** `H̃_ζ`, `H̃_ζξ`, `b̂_ζ` and `b̂_ζξ` all come from
/// [`AnchorDerivatives`], which #436 assembles in closed form from the third-order
/// `f`-sensitivities. Those in turn are obtained by finite-differencing the exact second-order
/// `Dual2` jet (Shi 2021) inside `subject_sensitivities_cov` — the *only* finite difference in
/// this path, and the same one FOCEI's own covariance already rides on.
///
/// An earlier draft reached the two second-order objects by differencing assembled `H̃_ζ`/`b̂_ζ`
/// over the natural parameters with the EBE reconverged at each point. That was wrong twice over:
/// it is FD at a far higher level than the sensitivity jet, so it injects inner-solver
/// convergence error into the top order, and it duplicated machinery that already existed one
/// function away. The `2·dim` reconverged inner solves it cost are now zero.
///
/// [`AnchorDerivatives`]: super::sens_cov_hessian::AnchorDerivatives
pub(crate) fn subject_agq_cov_hessian(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta_hat: &[f64],
    grid: &[Vec<f64>],
    pi: &[f64],
) -> Option<AgqCovTerms> {
    use crate::estimation::sens_cov_hessian::{omega_entries, subject_anchor_derivatives};
    use crate::estimation::sens_outer_gradient::prepare;
    use crate::sens::provider::subject_sensitivities_cov;
    use std::f64::consts::SQRT_2;

    let n_eta = model.n_eta;
    let n_theta = params.theta.len();
    let entries = omega_entries(params.omega.diagonal, n_eta);
    let dim = n_theta + entries.len() + params.sigma.values.len();

    // ── at the mode ────────────────────────────────────────────────────────────────────────
    let sens = subject_sensitivities_cov(model, subject, &params.theta, eta_hat)?;
    let prep = prepare(model, subject, params, &sens, eta_hat)?;
    let htilde = prep.htilde_inv.clone().try_inverse()?;
    let anchor = regularised_anchor(&htilde)?;
    // The conditioning screen `regularised_anchor` deliberately does not perform (review
    // finding 1). It is owed *here*, where `S⁻¹` is actually formed: a near-zero `H̃` diagonal —
    // a flat or unidentifiable η direction — is perfectly differentiable, so `regularised_anchor`
    // rightly returns `Some`, but term (A) then contracts `S⁻¹` entries of order `1/Sᵢᵢ` against
    // `S_ζ`/`S_ζξ` and reports standard errors inflated by many orders of magnitude, as an
    // analytic result. Declining hands the subject to the FD covariance instead, which is the
    // correct answer for a direction the data does not identify.
    if !anchor.is_well_conditioned() {
        return None;
    }
    let ad = subject_anchor_derivatives(model, subject, params, &sens, &prep, eta_hat);

    // `ad.dh`/`ad.d2h` are partials at **fixed η**; `total_first`/`total_second` chain the mode
    // response on, doing to the matrix `H̃` what M3's A/B/C/D assembly does to the scalar
    // `½log|H̃|`. Using the partials directly here would silently drop the mode movement.
    let b_hat_d = &ad.eta_d;
    let b_hat_dd = &ad.eta_dd;
    let s_d: Vec<DMatrix<f64>> = (0..dim)
        .map(|z| anchor.propagate(&ad.total_first(z)))
        .collect();
    let l_d: Vec<DMatrix<f64>> = s_d.iter().map(|s| anchor.factor_derivative(s)).collect();
    let m_d: Vec<DMatrix<f64>> = l_d
        .iter()
        .map(|l| anchor.node_scale_derivative(l))
        .collect();

    let s_dd: Vec<Vec<DMatrix<f64>>> = (0..dim)
        .map(|z| {
            (0..dim)
                .map(|x| anchor.propagate(&ad.total_second(z, x)))
                .collect()
        })
        .collect();
    let mut m_dd = vec![vec![DMatrix::<f64>::zeros(n_eta, n_eta); dim]; dim];
    for zeta in 0..dim {
        for xi in 0..dim {
            let l_zx = anchor.factor_second_derivative(&l_d[zeta], &l_d[xi], &s_dd[zeta][xi]);
            m_dd[zeta][xi] = anchor.node_scale_second_derivative(&l_d[zeta], &l_d[xi], &l_zx);
        }
    }

    // ── the grid ───────────────────────────────────────────────────────────────────────────
    let node_scale = anchor.node_scale();
    let mut u: Vec<Vec<f64>> = Vec::with_capacity(grid.len());
    let mut node_acc = DMatrix::<f64>::zeros(dim, dim);

    for (j, z) in grid.iter().enumerate() {
        let zv = DVector::from_column_slice(z);
        let b_j: Vec<f64> = (eta_hat_vec(eta_hat) + SQRT_2 * (&node_scale * &zv))
            .iter()
            .copied()
            .collect();
        let jet = node_jet(model, subject, params, &b_j)?;

        let beta: Vec<DVector<f64>> = (0..dim)
            .map(|zeta| node_displacement(&b_hat_d[zeta], &m_d[zeta], &zv))
            .collect();

        u.push(
            (0..dim)
                .map(|zeta| jet.s[zeta] + jet.g.dot(&beta[zeta]))
                .collect(),
        );

        let mut curv = node_curvature(&jet.parts.c, &jet.parts.m, &jet.h, &beta);
        // The tail: g_jᵀ(b̂_ζξ + √2·M_ζξ·z_j). Zero at the mode by stationarity, non-zero
        // elsewhere — see the module doc on why Bartlett does not kill it.
        for zeta in 0..dim {
            for xi in 0..dim {
                let disp = &b_hat_dd[zeta][xi] + SQRT_2 * (&m_dd[zeta][xi] * &zv);
                curv[(zeta, xi)] += jet.g.dot(&disp);
            }
        }
        node_acc += pi[j] * curv;
    }

    // The AGQ natural gradient, from the same `S_ζ` and the same grid the Hessian used, so the
    // packing chain cannot pair this Hessian with a gradient of a different objective.
    let grad: Vec<f64> = (0..dim)
        .map(|zeta| {
            let half_tr = 0.5 * anchor.chol.solve(&s_d[zeta]).trace();
            half_tr
                + u.iter()
                    .zip(pi.iter())
                    .map(|(uj, &w)| w * uj[zeta])
                    .sum::<f64>()
        })
        .collect();

    Some(AgqCovTerms {
        logdet: logdet_curvature(&anchor, &s_d, &s_dd),
        softmax: softmax_covariance(pi, &u),
        node: node_acc,
        grad,
    })
}

fn eta_hat_vec(eta: &[f64]) -> DVector<f64> {
    DVector::from_column_slice(eta)
}

/// The per-subject AGQ covariance Hessian in the optimizer's **packed** space.
///
/// Mirrors `sens_cov_hessian::subject_packed_cov_hessian` — same censored-row scope check, same
/// `pack_natural_hessian` chain — but pairs the AGQ Hessian with the **AGQ** natural gradient.
/// That pairing is the point: the chain's second-order term contracts `∂²(natural)/∂(packed)²`
/// against the gradient of the objective being differentiated, and FOCEI's differs from AGQ's by
/// the quadrature correction.
pub(crate) fn subject_packed_agq_cov_hessian(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
    grid: &[Vec<f64>],
    pi: &[f64],
) -> Option<DMatrix<f64>> {
    use crate::estimation::parameterization::unpack_params;
    use crate::estimation::sens_cov_hessian::pack_natural_hessian;
    use crate::estimation::sens_outer_gradient::prepare;
    use crate::sens::provider::subject_sensitivities_cov;

    let params = unpack_params(x, template);
    // Censored (M3/BLOQ) rows are out of the M3 assembly's scope, and `AnchorDerivatives` is
    // that assembly's machinery — so the same exclusion applies here.
    let sens = subject_sensitivities_cov(model, subject, &params.theta, eta_hat)?;
    let prep = prepare(model, subject, &params, &sens, eta_hat)?;
    if prep.et.iter().any(|t| t.censored) {
        return None;
    }
    let terms = subject_agq_cov_hessian(model, subject, &params, eta_hat, grid, pi)?;
    Some(pack_natural_hessian(
        &terms.total(),
        &terms.grad,
        x,
        template,
    ))
}

/// `s_{j,ζ} = ∂nll/∂ζ|_b` — the fixed-`b` score in **natural** `[θ, Ω, σ]` space.
///
/// The other half of `u_{j,ζ} = s_{j,ζ} + g_jᵀβ_{j,ζ}`, which term (B) averages over the grid.
///
/// `agq::accumulate_fixed_eta_packed_gradient` computes the same object in *packed* space and is
/// the validated reference for the θ and σ blocks — this is those blocks with the packing
/// Jacobians (`dθ/dx`, `σ_k`) left off, because #436's Hessian is assembled in natural space and
/// chained to packed once, at the end. Mixing the two conventions mid-assembly is the sort of
/// error that produces a plausible wrong covariance, so the split is kept explicit.
///
/// The Ω block is written out rather than borrowed: `nll` contains `½(bᵀΩ⁻¹b + log|Ω|)`, so
/// `∂nll/∂Ω_e = ½[tr(Ω⁻¹E_e) − bᵀΩ⁻¹E_eΩ⁻¹b]` at fixed `b` — no data channel at all, since `f`
/// does not depend on Ω.
pub(crate) fn fixed_b_natural_score(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &crate::sens::provider::SubjectSens,
    prep: &crate::estimation::sens_outer_gradient::Prep,
    core: &crate::estimation::sens_outer_gradient::ScoreCore,
    b: &[f64],
) -> Vec<f64> {
    use crate::estimation::sens_cov_hessian::omega_entries;
    use crate::estimation::sens_outer_gradient::data_sigma_gradient;

    let n_eta = prep.n_eta;
    let n_theta = params.theta.len();
    let entries = omega_entries(params.omega.diagonal, n_eta);
    let mut out = Vec::with_capacity(n_theta + entries.len() + params.sigma.values.len());

    // θ: the residual chain through `f`, plus the DIRECT channel a custom / time-varying σ
    // magnitude opens (`R` depending on θ without passing through `f`). Omitting the second is
    // not an approximation — it is a missing channel, which is how it read before #486.
    for m in 0..n_theta {
        let mut d = 0.0;
        for (j, obs) in sens.obs.iter().enumerate() {
            let e = &prep.et[j];
            d += 0.5 * e.alpha * obs.df_dtheta[m];
            if !e.dr_dtheta.is_empty() {
                d += 0.5 * (e.r - e.eps * e.eps) / (e.r * e.r) * e.dr_dtheta[m];
            }
        }
        out.push(d);
    }

    // Ω: prior only.
    let bv = DVector::from_column_slice(b);
    let oi = &prep.omega_inv;
    let oib = oi * &bv;
    for &(r, c) in &entries {
        let mut e = DMatrix::<f64>::zeros(n_eta, n_eta);
        e[(r, c)] = 1.0;
        e[(c, r)] = 1.0;
        let quad = oib.dot(&(&e * &oib));
        let tr = (oi * &e).trace();
        out.push(0.5 * (tr - quad));
    }

    // σ: reaches `nll` only through the residual variance, so a closed-form scalar computation
    // at fixed `f` — the function whose doc names this exact use (#251 AGQ/Laplace).
    out.extend(data_sigma_gradient(model, subject, params, sens, core));
    out
}

/// Symmetry to a relative tolerance, for the debug preconditions on the factor differentials.
fn is_symmetric(a: &DMatrix<f64>) -> bool {
    if a.nrows() != a.ncols() {
        return false;
    }
    let scale = a.amax().max(1.0);
    (0..a.nrows()).all(|i| (0..i).all(|j| (a[(i, j)] - a[(j, i)]).abs() <= 1e-10 * scale))
}

/// Build `S = H̃ + Λ` and its jitter response, or `None` when the anchor is not
/// differentiable — the caller then keeps the finite-difference covariance.
///
/// Declines in exactly two situations, each for a stated reason rather than out of caution:
///
/// * a diagonal within [`JITTER_BRANCH_MARGIN`] of the branch crossing `|H̃ᵢᵢ| = 1e-4`, where
///   `Λ''` is a delta and no finite answer is right;
/// * `S` not positive-definite, because `build_proposal` then silently substitutes the broad
///   `Σ = Ω` proposal — a **different function of `x`** — and differentiating `H̃ + Λ` would be
///   answering about an objective the fit never evaluated.
///
/// # What this deliberately does *not* screen for
///
/// A near-zero diagonal is **in scope and correctly handled**: it lands on the floor branch,
/// where `Λᵢᵢ` is constant, so `S` is differentiable there with zero jitter response. An earlier
/// version of this contract listed a third decline "within the same margin of zero, where the
/// `abs` flips sign" — that condition was never implemented and is not needed, because the `abs`
/// is only ever differentiated on the relative branch.
///
/// What a near-zero diagonal *does* threaten is the **conditioning** of the assembled Hessian:
/// `H̃ᵢᵢ = 1e-11` yields `S⁻¹` entries of order `1e10`, and term (A) contracts those against
/// `S_k`/`S_kl`. Returning `Some` here is still right — `S` and its derivative are exactly what
/// this function claims — but a step-5 caller that treats `Some` as "analytic standard errors are
/// trustworthy" would report values inflated by many orders of magnitude. **That gate is owed at
/// the assembly**, where `S⁻¹` is formed and its condition number is available; it is not owed
/// here, and is recorded so it is not mistaken for already-done.
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
        // Bail near the branch crossing, where `Λ''` is a delta.
        if a < JITTER_CROSSING * JITTER_BRANCH_MARGIN && a > JITTER_CROSSING / JITTER_BRANCH_MARGIN
        {
            return None;
        }
        let branch = if a * JITTER_REL > JITTER_FLOOR {
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
    // `build_proposal` falls back to the broad proposal when this fails; see above. The factor is
    // kept rather than discarded — the assembly needs it for the node placement and for term (A).
    let chol = s.clone().cholesky()?;
    let l_inv = chol
        .l()
        .solve_lower_triangular(&DMatrix::<f64>::identity(d, d))?;
    Some(RegularisedAnchor {
        s,
        chol,
        l_inv,
        coef,
    })
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

                let log_det_mine = reg.log_det();
                assert!(
                    (log_det_mine - proposal.log_det_inv_scale).abs() < 1e-12,
                    "d={d} scale={scale}: log|S| {log_det_mine} != build_proposal's {}",
                    proposal.log_det_inv_scale
                );
            }
        }

        // The sweep above pins `JITTER_REL` only. Its smallest diagonal is `1e-2 · 2 = 0.02`,
        // whose relative jitter is `2e-8` — 200× the `1e-10` floor — so every fixture sits on the
        // relative branch and `JITTER_FLOOR` never enters the compared `log|S|`. A retune of the
        // inline `.max(1e-10)` in `build_proposal` would leave the suite green while this module
        // differentiated a different `S`, which is precisely the failure the constant's doc claims
        // is covered. This fixture puts one diagonal on the floor branch so it is.
        for d in 1..=3 {
            let mut h = DMatrix::<f64>::identity(d, d);
            h[(0, 0)] = 1e-6; // 1e-6 · 1e-6 = 1e-12 < 1e-10 ⟹ floor branch
            let omega_inv = DMatrix::<f64>::identity(d, d);
            let reg = regularised_anchor(&h).expect("floor branch is in scope");
            let proposal = build_proposal(&h, &omega_inv, d).expect("proposal");
            assert_eq!(
                reg.coef[0], 0.0,
                "d={d}: fixture must be on the floor branch"
            );
            assert!(
                (reg.log_det() - proposal.log_det_inv_scale).abs() < 1e-12,
                "d={d}: floor-branch log|S| {} != build_proposal's {}",
                reg.log_det(),
                proposal.log_det_inv_scale
            );
        }
    }

    /// The sign is carried on the relative branch, including the negative case that the
    /// definiteness screen currently prevents `regularised_anchor` from returning.
    ///
    /// Exercised through [`JitterBranch`] directly for exactly that reason: routing it through
    /// `regularised_anchor` is impossible today, and the coefficient must not go into service
    /// uncovered when step 5 widens the screen to a modified Cholesky. See the variant's doc.
    #[test]
    fn coefficient_carries_the_sign_on_the_relative_branch() {
        assert_eq!(
            JitterBranch::Relative { sign: 1.0 }.coefficient(),
            JITTER_REL
        );
        assert_eq!(
            JitterBranch::Relative { sign: -1.0 }.coefficient(),
            -JITTER_REL,
            "a negative diagonal's jitter must shrink |S| as H̃ᵢᵢ grows, not inflate it"
        );
        assert_eq!(JitterBranch::Floor.coefficient(), 0.0);
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
    ///
    /// The linearity is also why `step` is large. A central difference of a linear map carries
    /// **no truncation error at all**, so the only error is subtractive cancellation in `sp − sm`:
    /// roughly `ULP(max|S|)/step`. An earlier `step = 1e-3` against a fixed `1e-12` bound left a
    /// residual of `~7.3e-13` — a margin of 1.4×, close enough that a fourth dimension, a larger
    /// fixture, or a nalgebra rounding change would turn this red on an unrelated PR. At
    /// `step = 1e-1` the floor drops to `~2e-14`, and the bound below is relative so it scales
    /// with the fixture instead of being retuned alongside it. Nothing real is masked: a wrong
    /// coefficient moves the diagonal by `~1e-6·|dir|`, eight orders above the bound.
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

        let step = 1e-1;
        let sp = regularised_anchor(&(&h + step * &dir)).unwrap().s;
        let sm = regularised_anchor(&(&h - step * &dir)).unwrap().s;
        let fd = (sp - sm) / (2.0 * step);
        let analytic = reg.propagate(&dir);

        for i in 0..d {
            for j in 0..d {
                let tol = 1e-12 * (1.0 + fd[(i, j)].abs());
                assert!(
                    (analytic[(i, j)] - fd[(i, j)]).abs() < tol,
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

    /// A symmetric perturbation direction, for the factor-differential tests.
    ///
    /// Symmetry is required, not cosmetic: `dS` is the differential of a symmetric matrix, and a
    /// Cholesky differential is undefined otherwise (the `debug_assert`s catch it).
    fn sym_dir(d: usize) -> DMatrix<f64> {
        let mut dir = DMatrix::<f64>::zeros(d, d);
        for i in 0..d {
            for j in 0..d {
                dir[(i, j)] = 0.4 * ((i + j) as f64 + 1.0).sin() + if i == j { 0.7 } else { 0.0 };
            }
        }
        (&dir + dir.transpose()) * 0.5
    }

    /// `M = L⁻ᵀ` really is the node-placement scale: `M Mᵀ = S⁻¹`.
    ///
    /// The property the whole grid rests on — `b_j = b̂ + √2·M·z_j` is a draw from the proposal
    /// only if this holds. Cheap, and it pins the transpose convention: `L⁻¹` (not `L⁻ᵀ`) here
    /// would still be a valid square root of *something*, just not of `S⁻¹`.
    #[test]
    fn node_scale_is_an_inverse_square_root_of_s() {
        for d in 1..=4 {
            let h = spd(d, 3.0);
            let reg = regularised_anchor(&h).unwrap();
            let m = reg.node_scale();
            let recon = &m * m.transpose();
            let s_inv = reg.s.clone().try_inverse().expect("S is PD");
            for i in 0..d {
                for j in 0..d {
                    assert!(
                        (recon[(i, j)] - s_inv[(i, j)]).abs() < 1e-10 * s_inv.amax().max(1.0),
                        "d={d}: M Mᵀ != S⁻¹ at ({i},{j})"
                    );
                }
            }
        }
    }

    /// `factor_derivative` against a central difference of the factor itself.
    ///
    /// `step = 1e-4` balances the two error sources: cancellation `~ε|L|/step ≈ 2e-11` against
    /// truncation `~step²|L'''|/6 ≈ 1e-10`. The asserted `1e-8` therefore sits about two orders
    /// above the FD reference's own noise, which is the limiting side here — not the analytic
    /// derivative under test.
    #[test]
    fn factor_derivative_matches_fd() {
        for d in 1..=4 {
            let h = spd(d, 4.0);
            let dir = sym_dir(d);
            let reg = regularised_anchor(&h).unwrap();

            let analytic = reg.factor_derivative(&reg.propagate(&dir));

            let step = 1e-4;
            let lp = regularised_anchor(&(&h + step * &dir)).unwrap().chol.l();
            let lm = regularised_anchor(&(&h - step * &dir)).unwrap().chol.l();
            let fd = (lp - lm) / (2.0 * step);

            for i in 0..d {
                for j in 0..d {
                    assert!(
                        (analytic[(i, j)] - fd[(i, j)]).abs() < 1e-8,
                        "d={d} L_k[{i},{j}]: analytic {} vs FD {}",
                        analytic[(i, j)],
                        fd[(i, j)]
                    );
                }
            }
        }
    }

    /// `factor_second_derivative` against a second central difference along one direction.
    ///
    /// Along the straight path `H̃ + t·dir` the anchor is linear in `t` (on a fixed jitter
    /// branch), so `S_kl = 0` and the analytic answer comes **entirely** from the
    /// `−A_kA_lᵀ − A_lA_kᵀ` correction. That makes this a direct test of the only non-obvious
    /// part of the formula: dropping those terms leaves `L_kl = 0`, which the assertion below
    /// rejects by a wide margin.
    #[test]
    fn factor_second_derivative_matches_fd() {
        for d in 1..=4 {
            let h = spd(d, 4.0);
            let dir = sym_dir(d);
            let reg = regularised_anchor(&h).unwrap();

            let lk = reg.factor_derivative(&reg.propagate(&dir));
            let zero = DMatrix::<f64>::zeros(d, d);
            let analytic = reg.factor_second_derivative(&lk, &lk, &zero);

            let step = 1e-3;
            let lp = regularised_anchor(&(&h + step * &dir)).unwrap().chol.l();
            let l0 = reg.chol.l();
            let lm = regularised_anchor(&(&h - step * &dir)).unwrap().chol.l();
            let fd = (lp - 2.0 * l0 + lm) / (step * step);

            let mut max_abs = 0.0_f64;
            for i in 0..d {
                for j in 0..d {
                    max_abs = max_abs.max(analytic[(i, j)].abs());
                    assert!(
                        (analytic[(i, j)] - fd[(i, j)]).abs() < 1e-7,
                        "d={d} L_kl[{i},{j}]: analytic {} vs FD {}",
                        analytic[(i, j)],
                        fd[(i, j)]
                    );
                }
            }
            // The correction is not vanishing — otherwise the check above would pass trivially
            // against a zero matrix and prove nothing.
            assert!(
                max_abs > 1e-3,
                "d={d}: second differential is ~0, so the FD agreement is vacuous"
            );
        }
    }

    /// `L_kl` is symmetric under `k ↔ l`, to the bit.
    ///
    /// Not a numerical claim: `Φ(L⁻¹S_klL⁻ᵀ − A_kA_lᵀ − A_lA_kᵀ)` is manifestly symmetric in the
    /// pair, so any asymmetry is an implementation bug rather than round-off. nlmixr2est#787
    /// verifies its equivalent to `9e-19`; exact equality is the stronger statement available
    /// here because both orderings execute the same instruction sequence with swapped operands.
    #[test]
    fn factor_second_derivative_is_clairaut_symmetric() {
        let d = 4;
        let h = spd(d, 4.0);
        let reg = regularised_anchor(&h).unwrap();

        let dir_k = sym_dir(d);
        let mut dir_l = sym_dir(d);
        dir_l[(0, 0)] = -1.3; // make the two directions genuinely different
        dir_l[(d - 1, d - 1)] = 2.9;

        let lk = reg.factor_derivative(&reg.propagate(&dir_k));
        let ll = reg.factor_derivative(&reg.propagate(&dir_l));
        let mixed = reg.propagate(&sym_dir(d)); // any symmetric S_kl

        let kl = reg.factor_second_derivative(&lk, &ll, &mixed);
        let lk_order = reg.factor_second_derivative(&ll, &lk, &mixed);

        for i in 0..d {
            for j in 0..d {
                assert_eq!(
                    kl[(i, j)],
                    lk_order[(i, j)],
                    "L_kl must be exactly symmetric in (k,l) at ({i},{j})"
                );
            }
        }
    }

    /// `M_k` and `M_kl` against central differences of `M = L⁻ᵀ`.
    ///
    /// `M` is what actually places the nodes, so an error here moves the grid rather than
    /// mis-scaling a determinant — the failure mode nlmixr2est#785 hit. Checked along a single
    /// direction, where `S_kl = 0` again, so `M_kl`'s two triple products carry the whole result.
    #[test]
    fn node_scale_derivatives_match_fd() {
        for d in 1..=4 {
            let h = spd(d, 4.0);
            let dir = sym_dir(d);
            let reg = regularised_anchor(&h).unwrap();

            let lk = reg.factor_derivative(&reg.propagate(&dir));
            let zero = DMatrix::<f64>::zeros(d, d);
            let lkl = reg.factor_second_derivative(&lk, &lk, &zero);

            let d_analytic = reg.node_scale_derivative(&lk);
            let d2_analytic = reg.node_scale_second_derivative(&lk, &lk, &lkl);

            let step = 1e-4;
            let mp = regularised_anchor(&(&h + step * &dir))
                .unwrap()
                .node_scale();
            let m0 = reg.node_scale();
            let mm = regularised_anchor(&(&h - step * &dir))
                .unwrap()
                .node_scale();
            let d_fd = (&mp - &mm) / (2.0 * step);

            let step2 = 1e-3;
            let mp2 = regularised_anchor(&(&h + step2 * &dir))
                .unwrap()
                .node_scale();
            let mm2 = regularised_anchor(&(&h - step2 * &dir))
                .unwrap()
                .node_scale();
            let d2_fd = (&mp2 - 2.0 * &m0 + &mm2) / (step2 * step2);

            for i in 0..d {
                for j in 0..d {
                    assert!(
                        (d_analytic[(i, j)] - d_fd[(i, j)]).abs() < 1e-8,
                        "d={d} M_k[{i},{j}]: analytic {} vs FD {}",
                        d_analytic[(i, j)],
                        d_fd[(i, j)]
                    );
                    assert!(
                        (d2_analytic[(i, j)] - d2_fd[(i, j)]).abs() < 1e-6,
                        "d={d} M_kl[{i},{j}]: analytic {} vs FD {}",
                        d2_analytic[(i, j)],
                        d2_fd[(i, j)]
                    );
                }
            }
        }
    }

    /// **The reduction, proven at the level of the formula.**
    ///
    /// Substituting the implicit-function relation `β_k = −H⁻¹M_k` — which is what the node
    /// displacement becomes at `z = 0`, i.e. `β_k = b̂_k` — [`node_curvature`] must collapse onto
    /// `C − MᵀH⁻¹M`, exactly `CovHessianParts::fuse` and hence #436's M2 natural block:
    ///
    /// ```text
    ///   M_kᵀβ_l + M_lᵀβ_k + β_lᵀHβ_k  =  −X − X + X  =  −X,   X = M_kᵀH⁻¹M_l
    /// ```
    ///
    /// Done on synthetic matrices rather than a fitted model deliberately: this is an algebraic
    /// identity, so a model fixture would only add EBE-convergence noise to a claim that holds
    /// exactly. The model-level version is step 5's
    /// `agq_cov_hessian_reduces_to_focei_at_one_node`.
    #[test]
    fn node_curvature_collapses_to_the_m2_envelope_at_the_mode() {
        let dim = 5; // natural parameters
        let d = 3; // random effects
        let h = spd(d, 2.0);
        let h_inv = h.clone().try_inverse().unwrap();

        let c = {
            let raw = spd(dim, 1.5);
            (&raw + raw.transpose()) * 0.5
        };
        let m: Vec<DVector<f64>> = (0..dim)
            .map(|k| {
                DVector::from_iterator(d, (0..d).map(|i| 0.3 * (k + 1) as f64 - 0.11 * i as f64))
            })
            .collect();
        let beta: Vec<DVector<f64>> = m.iter().map(|mk| -(&h_inv * mk)).collect();

        let got = node_curvature(&c, &m, &h, &beta);

        for k in 0..dim {
            for l in 0..dim {
                let want = c[(k, l)] - m[k].dot(&(&h_inv * &m[l]));
                assert!(
                    (got[(k, l)] - want).abs() < 1e-10 * want.abs().max(1.0),
                    "({k},{l}): node curvature {} != M2 envelope {want}",
                    got[(k, l)]
                );
            }
        }
    }

    /// Term (C) is symmetric in `(k,l)` for an arbitrary displacement, not only at the mode.
    ///
    /// The module doc calls asymmetry here "a bug, not a rounding artefact". Checked with `β`
    /// unrelated to `−H⁻¹M` so the symmetry cannot come from the envelope structure.
    #[test]
    fn node_curvature_is_symmetric_away_from_the_mode() {
        let dim = 4;
        let d = 3;
        let h = spd(d, 2.0);
        let c = {
            let raw = spd(dim, 1.5);
            (&raw + raw.transpose()) * 0.5
        };
        let m: Vec<DVector<f64>> = (0..dim)
            .map(|k| {
                DVector::from_iterator(d, (0..d).map(|i| 0.7 * (k as f64 + 1.0).sin() + i as f64))
            })
            .collect();
        let beta: Vec<DVector<f64>> = (0..dim)
            .map(|k| {
                DVector::from_iterator(d, (0..d).map(|i| 1.3 * (i as f64 - 0.5 * k as f64).cos()))
            })
            .collect();

        let got = node_curvature(&c, &m, &h, &beta);
        for k in 0..dim {
            for l in 0..dim {
                assert!(
                    (got[(k, l)] - got[(l, k)]).abs() < 1e-12 * got.amax().max(1.0),
                    "term (C) must be symmetric at ({k},{l})"
                );
            }
        }
    }

    /// `β_{j,k} = b̂_k` exactly when `z = 0` — the single node cannot move.
    #[test]
    fn node_displacement_is_the_mode_response_at_the_central_node() {
        let d = 3;
        let b_hat_k = DVector::from_vec(vec![0.4, -1.2, 3.0]);
        let m_k = spd(d, 2.0);
        let z = DVector::zeros(d);
        let beta = node_displacement(&b_hat_k, &m_k, &z);
        for i in 0..d {
            assert_eq!(beta[i], b_hat_k[i], "z=0 must give exactly b̂_k at {i}");
        }
        // …and a non-zero node really does move, so the check above is not vacuous.
        let z1 = DVector::from_vec(vec![1.0, -0.5, 0.25]);
        let moved = node_displacement(&b_hat_k, &m_k, &z1);
        assert!(
            (0..d).any(|i| (moved[i] - b_hat_k[i]).abs() > 1e-6),
            "a non-central node must be displaced from the mode"
        );
    }

    /// Term (B) vanishes identically at one node — the property that makes AGQ(1) reduce to
    /// FOCEI — and is a genuine covariance otherwise.
    #[test]
    fn softmax_covariance_vanishes_at_one_node() {
        let u_one = vec![vec![1.5, -2.0, 0.75]];
        let cov = softmax_covariance(&[1.0], &u_one);
        for k in 0..3 {
            for l in 0..3 {
                assert!(
                    cov[(k, l)].abs() < 1e-14,
                    "a single softmax weight has zero variance, got {} at ({k},{l})",
                    cov[(k, l)]
                );
            }
        }

        // Two nodes: compare against the definition written out directly.
        let pi = [0.3, 0.7];
        let u = vec![vec![1.0, 2.0], vec![-3.0, 0.5]];
        let cov = softmax_covariance(&pi, &u);
        for k in 0..2 {
            for l in 0..2 {
                let mean_k: f64 = pi.iter().zip(&u).map(|(w, uj)| w * uj[k]).sum();
                let mean_l: f64 = pi.iter().zip(&u).map(|(w, uj)| w * uj[l]).sum();
                let raw: f64 = pi.iter().zip(&u).map(|(w, uj)| w * uj[k] * uj[l]).sum();
                assert!((cov[(k, l)] - (raw - mean_k * mean_l)).abs() < 1e-12);
            }
        }
    }

    /// Term (A) against a second central difference of `½log|S|` along a straight path.
    ///
    /// `S` is linear in `t` there, so `S_kl = 0` and the analytic value is entirely
    /// `−½tr((S⁻¹S')²)` — the second trace, which is the term an implementation is most likely to
    /// drop or sign-flip. A version omitting it would return `0` here.
    #[test]
    fn logdet_curvature_matches_fd_of_half_log_det() {
        for d in 2..=4 {
            let h = spd(d, 4.0);
            let dir = sym_dir(d);
            let anchor = regularised_anchor(&h).unwrap();

            let s_k = anchor.propagate(&dir);
            let zero = DMatrix::<f64>::zeros(d, d);
            let analytic = logdet_curvature(&anchor, &[s_k], &[vec![zero]])[(0, 0)];

            let step = 1e-3;
            let lp = regularised_anchor(&(&h + step * &dir)).unwrap().log_det();
            let l0 = anchor.log_det();
            let lm = regularised_anchor(&(&h - step * &dir)).unwrap().log_det();
            let fd = 0.5 * (lp - 2.0 * l0 + lm) / (step * step);

            assert!(
                (analytic - fd).abs() < 1e-6 * fd.abs().max(1.0),
                "d={d}: term (A) {analytic} vs FD {fd}"
            );
            assert!(
                analytic.abs() > 1e-3,
                "d={d}: curvature is ~0, so the comparison is vacuous"
            );
        }
    }

    /// The conditioning screen accepts an ordinary anchor and rejects a flat η direction.
    ///
    /// The second case is the one review finding 1 raised: `H̃ᵢᵢ = 1e-11` is *differentiable* —
    /// it sits on the jitter's floor branch, so `regularised_anchor` correctly returns `Some` —
    /// but `S⁻¹` then carries entries of order `1e10`, and term (A) contracts them twice. Both
    /// halves are asserted, because a screen that rejected everything would satisfy the second
    /// claim alone while silently disabling the feature.
    #[test]
    fn conditioning_screen_admits_real_anchors_and_rejects_flat_directions() {
        for d in 1..=4 {
            for scale in [1e-2, 1.0, 25.0, 1e3] {
                assert!(
                    regularised_anchor(&spd(d, scale))
                        .unwrap()
                        .is_well_conditioned(),
                    "d={d} scale={scale}: an ordinary curvature must be admitted"
                );
            }
        }

        // A near-zero diagonal: in scope for `regularised_anchor`, out of scope for the assembly.
        let mut h = DMatrix::<f64>::identity(3, 3);
        h[(0, 0)] = 1e-11;
        let reg = regularised_anchor(&h).expect("floor branch is differentiable");
        assert_eq!(
            reg.coef[0], 0.0,
            "premise: this diagonal is on the floor branch"
        );
        assert!(
            !reg.is_well_conditioned(),
            "a flat η direction must be declined before S⁻¹ is contracted"
        );

        // The threshold is derived, not fitted to this fixture. `cond(S) ≈ 1/1.1e-10 ≈ 9e9`, so
        // term (A)'s double contraction amplifies round-off by `cond²·ε ≈ 1e4` — no correct
        // digits. An earlier `1e12` cutoff admitted exactly this case; asserting the fixture's
        // conditioning here keeps the two facts (how bad it is, that it is rejected) together, so
        // a future retune cannot quietly pass by loosening the bound past it again.
        let l = reg.chol.l();
        let hi = (0..3).fold(0.0_f64, |m, i| m.max(l[(i, i)]));
        let lo = (0..3).fold(f64::INFINITY, |m, i| m.min(l[(i, i)]));
        let cond = (hi / lo).powi(2);
        assert!(
            cond > 1e9,
            "premise: the fixture must really be ill-conditioned; cond ≈ {cond}"
        );
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
