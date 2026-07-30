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
//! available check on it, and the reason `agq_cov_hessian_reduces_to_focei_at_one_node` is the
//! first test in this module rather than an afterthought.
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
//! so carrying it costs nothing. Two branches are **not** differentiable and must decline
//! rather than be approximated:
//!
//! * a diagonal pinned at the `1e-10` absolute floor (the `max` switches), and
//! * `H̃ᵢᵢ ≤ 0` (the `abs` switches).
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

// Implementation follows in subsequent commits, in the order the derivation above needs it:
//   1. `S` and its jitter branch predicate (this is where a silent wrong answer would start)
//   2. `H̃_k` as explicit matrices, vs FD of `score_core(...).htilde` with the mode moved
//   3. `b̂_kl`, vs FD of `subject_eta_dx`
//   4. `M_k`, `M_kl` from `S_k`, `S_kl`
//   5. terms (A), (B), (C) and the packed assembly
// Each step lands with its own FD parity test before the next builds on it, per the repo's
// analytic-sensitivity rule — a wrong sensitivity here compiles and runs silently.
