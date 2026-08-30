//! Adaptive Gauss–Hermite quadrature (AGQ) — the marginal-likelihood objective (#251).
//!
//! # What it computes
//!
//! FOCE/FOCEI approximate each subject's marginal likelihood
//! `L_i = ∫ p(y_i|η) φ(η; 0, Ω) dη` with a *single* Gaussian centred at the empirical-Bayes
//! mode. AGQ keeps that centring but evaluates the integrand on a Gauss–Hermite grid laid
//! around the mode, so the approximation improves with the node count instead of being
//! fixed. Writing `l_i(η) = log p(y_i|η) + log φ(η; 0, Ω)` for the exact conditional
//! log-likelihood, `η̂` for the mode and `H = −∇²l_i(η̂)`:
//!
//! ```text
//!   L_i ≈ 2^(d/2) · |Σ^{1/2}| · Σ_j [ (Π_k w_{j,k}) · exp(‖z_j‖²) · exp(l_i(η̂ + √2·Σ^{1/2}·z_j)) ]
//! ```
//!
//! with `Σ = H⁻¹`, `z_j` the tensor-product Gauss–Hermite nodes and `w` the GH weights. The
//! `exp(‖z_j‖²)` factor undoes the `e^{−z²}` that the Hermite weights carry, so what is
//! actually integrated is the *full* integrand rather than a Gaussian-shaped surrogate.
//!
//! **Two properties follow, and they are the whole point of the method:**
//!
//! 1. **`n_agq = 1` is exactly Laplace.** The one-point rule is `z = 0`, `w = √π`, so the
//!    sum collapses to `(2π)^(d/2) · |H|^(−1/2) · exp(l_i(η̂))` — the Laplace approximation,
//!    term for term. This is not an approximation of an approximation; it is an identity,
//!    and [`tests::one_node_agq_equals_laplace`] pins it.
//! 2. **No Gaussian-residual assumption.** `l_i` is evaluated through
//!    [`individual_nll_into_with_schedule`], the model's *actual* likelihood — so
//!    time-to-event and categorical endpoints are integrated as faithfully as Gaussian
//!    ones. That is what FOCE/FOCEI structurally cannot do, and why AGQ exists here.
//!
//! # Constant convention
//!
//! `individual_nll_*` omits the `(d + n_obs)/2 · log(2π)` constants (NONMEM's "objective
//! function without constant"). Writing `G_i = ∫ exp(−nll(η)) dη`, the OFV in that same
//! convention is `−2·Σ_i log G_i + Σ_i d·log(2π)`, so this module's per-subject
//! contribution — the quantity the outer loop doubles into an OFV — is
//!
//! ```text
//!   agq_nll_i = (d/2)·log(π) + ½·log|H| − logΣexp_j[ Σ_k log w_{j,k} + ‖z_j‖² − nll(η_j) ]
//! ```
//!
//! which at one node reduces to `nll(η̂) + ½·log|H|`: the FOCEI Laplace per-subject NLL,
//! in the same units as [`crate::stats::likelihood::foce_subject_nll`]. AGQ OFVs are
//! therefore directly comparable to FOCE/FOCEI OFVs from this engine.
//!
//! # Why the Hessian is finite-differenced
//!
//! `H` is the Hessian of the *true* integrand, obtained by central differences of
//! `individual_nll`. It is deliberately **not**
//! [`crate::estimation::importance_sampling::compute_posterior_hessian`], which builds the
//! Gauss-Newton form `Ω⁻¹ + JᵀR⁻¹J`: that carries no curvature at all from TTE or
//! categorical endpoints, i.e. it is blind on exactly the models AGQ is here to serve, and
//! would scale their grids by `Ω` alone. Note the grid scaling only affects *accuracy at
//! finite n* — AGQ is consistent under any invertible scaling — but at `n_agq = 1` it enters
//! the objective directly through `½·log|H|`, so a blind `H` would silently not be Laplace.
//!
//! This is confirmed externally. On warfarin, AGQ at one node reproduces NONMEM
//! `$EST METHOD=1 LAPLACIAN INTER` to five significant figures (7.8994 both), while ferx
//! FOCEI and NONMEM `METHOD=1 INTER` agree with each other on a *different* value (7.4967)
//! — so the 0.40-unit AGQ(1)-vs-FOCEI gap is FOCEI's Gauss-Newton Hessian, reproduced by
//! the reference implementation, not drift here. NONMEM's LAPLACIAN **without** `INTER`
//! misses by 9.3 units, which is the same point from the other side: the exact Hessian has
//! to carry the curvature of the η-dependent residual variance (`ln V(f(η))`), and the
//! Gauss-Newton form would have silently dropped it. See `docs/estimation/agq.qmd`.
//!
//! # Grid pruning: why there isn't any
//!
//! Dropping low-weight nodes looks free and is not. The *corrected* weight
//! `w_k · exp(z_k²)` is `Θ(1)` across the grid — the Hermite weight's `e^{−z²}` decay is
//! exactly what the correction cancels — so a threshold on `w_k` prunes nodes whose actual
//! contribution is not small. The contribution is only known after `nll(η_j)` is evaluated,
//! which is the entire cost. Dimensionality is therefore bounded by [`MAX_AGQ_GRID`]
//! instead, and a sparse (Smolyak) rule is the real answer for `d = 5..8` (issue #251).

use nalgebra::DMatrix;
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::estimation::importance_sampling::build_proposal;
use crate::estimation::inner_optimizer::cacheable_schedule;
use crate::pk;
use crate::stats::likelihood::individual_nll_into_with_schedule;
use crate::stats::util::log_sum_exp as logsumexp;
use crate::types::{CompiledModel, HessianAnchor, ModelParameters, Population, Subject};

/// Upper bound on `n_agq`. Beyond ~20 nodes the Golub–Welsch eigenproblem starts to lose
/// the extreme nodes to round-off, and the marginal accuracy gain is nil for any realistic
/// NLME integrand.
pub const MAX_AGQ_NODES: usize = 21;

/// Hard cap on the tensor-grid size `n_agq^n_eta`, enforced at model-check time by
/// [`crate::api::check_model_options`]. The tensor rule costs one full likelihood
/// evaluation per node per subject per outer iteration, so the grid — not the node count —
/// is the quantity a user needs protecting from. 100k nodes on an ODE model is already
/// hours; past that a fit is not slow, it is wrong to have started.
pub const MAX_AGQ_GRID: usize = 100_000;

/// The sentinel a non-finite likelihood collapses to, matching `individual_nll`'s own
/// convention so a diverged node sorts as "impossible" rather than poisoning the sum.
const NLL_SENTINEL: f64 = 1e20;

/// Relative step for central-differencing the grid-response term (and the mixed
/// `∂²nll/∂η∂x` in [`eta_dx`]).
///
/// **Truncation-dominated, not noise-dominated** — the opposite of the intuition. It
/// differences `log|H|`, which curves sharply in the packed coordinates, so a "safe" large
/// step is catastrophic: `1e-2` puts the θ gradient 5× off, `1e-3` still 4%. The `H` inside
/// is itself finite-differenced, but its error floor turns out to sit well below where
/// truncation bites, so the step wants to be *small*. Measured on warfarin, the answer is
/// converged by `1e-4` and unchanged at `1e-5`. Calibrated by
/// `analytic_gradient_matches_fd_at_every_node_count`, which would catch a regression here
/// immediately.
const AGQ_GRID_FD_STEP: f64 = 1e-5;

/// Number of nodes in the tensor grid, saturating at [`usize::MAX`] rather than
/// overflowing — callers compare against [`MAX_AGQ_GRID`] and reject long before that.
pub fn grid_size(n_nodes: usize, n_eta: usize) -> usize {
    n_nodes.saturating_pow(n_eta as u32)
}

/// Gauss–Hermite nodes and weights for the *physicists'* weight `e^{−x²}` (so the weights
/// sum to `√π`), via Golub–Welsch: the nodes are the eigenvalues of the symmetric
/// tridiagonal Jacobi matrix with zero diagonal and off-diagonal `β_k = √(k/2)`, and the
/// weights are `√π · v_{0,k}²` from the first component of each unit eigenvector.
///
/// Computing them beats a hard-coded table: it is exact at every `n` (no transcription
/// risk), and `n = 1` falls out as the 1×1 zero matrix → node `0`, weight `√π`, which is
/// precisely what makes AGQ collapse to Laplace.
pub(crate) fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    debug_assert!(n >= 1, "gauss_hermite needs at least one node");
    let mut j = DMatrix::<f64>::zeros(n, n);
    for k in 1..n {
        let beta = (k as f64 / 2.0).sqrt();
        j[(k - 1, k)] = beta;
        j[(k, k - 1)] = beta;
    }
    let eig = j.symmetric_eigen();
    let sqrt_pi = PI.sqrt();
    let mut pairs: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let v0 = eig.eigenvectors[(0, k)];
            (eig.eigenvalues[k], sqrt_pi * v0 * v0)
        })
        .collect();
    // Ascending node order: the grid is then deterministic and reproducible run to run,
    // independent of whatever order the eigensolver happened to converge in.
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("GH nodes are finite"));
    pairs.into_iter().unzip()
}

/// The per-subject **integration variable** and its prior.
///
/// AGQ integrates over whatever random effects the subject has. Without IOV that is `b = η`
/// with prior `Ω`. With IOV it is the **stacked** vector
///
/// ```text
///   b = [η, κ₁, …, κ_K]        (dimension d = n_eta + K·n_kappa)
/// ```
///
/// whose prior is the block-diagonal `Ω_joint = Ω ⊕ Ω_iov ⊕ … ⊕ Ω_iov` (K copies). That is
/// not a new model — it is exactly what
/// [`crate::stats::likelihood::individual_nll_iov`] already scores, since
///
/// ```text
///   nll_iov = ½·(ηᵀΩ⁻¹η + log|Ω| + Σ_k κ_kᵀΩ_iov⁻¹κ_k + K·log|Ω_iov| + data)
///           = ½·(bᵀΩ_joint⁻¹b + log|Ω_joint| + data)
/// ```
///
/// So **every AGQ formula in this module carries over verbatim** with `d` the stacked
/// dimension: the marginal, the `(d/2)·logπ + ½log|H|` normaliser, the node transform, the
/// Fisher-identity gradient. IOV is a change of dimension, not of method.
///
/// The cost is that `d` grows with the number of occasions, and the tensor grid is
/// `n_agq^d` — see [`MAX_AGQ_GRID`]. `method = laplace` (one node) is unaffected: its grid is
/// a single point no matter how large `d` gets, so Laplace + IOV is always tractable.
pub(crate) struct Stack {
    n_eta: usize,
    n_kappa: usize,
    /// Occasions for *this* subject (0 when the model or subject has no IOV).
    n_occ: usize,
    /// `Ω_joint⁻¹`, the prior precision over `b`. Feeds `build_proposal`'s fallback.
    omega_joint_inv: DMatrix<f64>,
    /// `√diag(Ω_joint)` — each coordinate's natural scale, for finite-difference steps.
    prior_sd: Vec<f64>,
}

impl Stack {
    /// Build the stack for a subject with `n_occ` occasions (`0` ⇒ no IOV).
    fn new(model: &CompiledModel, params: &ModelParameters, n_occ: usize) -> Self {
        let n_eta = model.n_eta;
        let n_kappa = if n_occ == 0 { 0 } else { model.n_kappa };
        let d = n_eta + n_occ * n_kappa;

        let (omega_joint_inv, prior_sd) = match (&params.omega_iov, n_kappa) {
            (Some(iov), k) if k > 0 => {
                let inv = crate::estimation::importance_sampling::build_joint_omega_inv(
                    &params.omega.inv,
                    &iov.inv,
                    n_eta,
                    k,
                    n_occ,
                );
                let mut sd = Vec::with_capacity(d);
                for i in 0..n_eta {
                    sd.push(params.omega.matrix[(i, i)].max(0.0).sqrt());
                }
                for _ in 0..n_occ {
                    for i in 0..k {
                        sd.push(iov.matrix[(i, i)].max(0.0).sqrt());
                    }
                }
                (inv, sd)
            }
            _ => {
                let sd = (0..n_eta)
                    .map(|i| params.omega.matrix[(i, i)].max(0.0).sqrt())
                    .collect();
                (params.omega.inv.clone(), sd)
            }
        };
        Self {
            n_eta,
            n_kappa,
            n_occ,
            omega_joint_inv,
            prior_sd,
        }
    }

    /// Dimension of the integration variable.
    fn d(&self) -> usize {
        self.n_eta + self.n_occ * self.n_kappa
    }

    fn is_iov(&self) -> bool {
        self.n_occ > 0 && self.n_kappa > 0
    }

    /// Split `b` back into `(η, [κ₁ … κ_K])`.
    fn split<'a>(&self, b: &'a [f64]) -> (&'a [f64], Vec<Vec<f64>>) {
        let eta = &b[..self.n_eta];
        let kappas = (0..self.n_occ)
            .map(|k| {
                let s = self.n_eta + k * self.n_kappa;
                b[s..s + self.n_kappa].to_vec()
            })
            .collect();
        (eta, kappas)
    }

    /// The exact conditional NLL at an arbitrary `b` — the AGQ integrand. Dispatches to the
    /// IOV-aware likelihood when the subject has occasions; both are the model's *real*
    /// likelihood, not a surrogate.
    fn nll_at(
        &self,
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
        b: &[f64],
        scratch: &mut pk::EventPkParams,
        schedule: Option<&pk::event_driven::EventSchedule>,
    ) -> f64 {
        if !self.is_iov() {
            return individual_nll_into_with_schedule(
                model,
                subject,
                &params.theta,
                b,
                &params.omega,
                &params.sigma.values,
                scratch,
                schedule,
            );
        }
        let (eta, kappas) = self.split(b);
        crate::stats::likelihood::individual_nll_iov(
            model,
            subject,
            &params.theta,
            eta,
            &kappas,
            &params.omega,
            params.omega_iov.as_ref(),
            &params.sigma.values,
        )
    }
}

/// Central-difference Hessian of the exact conditional NLL w.r.t. the integration variable
/// `b` at `b_hat` (see [`Stack`] — `b = η`, or `[η, κ₁..κ_K]` under IOV).
///
/// Step `hᵢ = max(ε^¼ · √Ω_joint,ᵢᵢ, 1e-4)`: `ε^¼` is the standard second-difference step (it
/// balances truncation against the `ε/h²` round-off blow-up), scaled by the prior SD because
/// that is each coordinate's natural unit. The `1e-4` floor keeps a near-zero (or FIXed-small)
/// prior variance from driving `h` so small that `ε/h²` swamps the curvature.
fn fd_posterior_hessian(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    stack: &Stack,
    b_hat: &[f64],
    scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> DMatrix<f64> {
    let d = stack.d();
    let mut nll_at =
        |b: &[f64]| -> f64 { stack.nll_at(model, subject, params, b, scratch, schedule) };

    let steps: Vec<f64> = (0..d)
        .map(|i| (f64::EPSILON.powf(0.25) * stack.prior_sd[i]).max(1e-4))
        .collect();

    let f0 = nll_at(b_hat);
    let mut h = DMatrix::<f64>::zeros(d, d);
    let mut b = b_hat.to_vec();

    for i in 0..d {
        let hi = steps[i];
        b[i] = b_hat[i] + hi;
        let f_plus = nll_at(&b);
        b[i] = b_hat[i] - hi;
        let f_minus = nll_at(&b);
        b[i] = b_hat[i];
        h[(i, i)] = (f_plus - 2.0 * f0 + f_minus) / (hi * hi);
    }

    for i in 0..d {
        for j in (i + 1)..d {
            let (hi, hj) = (steps[i], steps[j]);
            let mut at = |si: f64, sj: f64| -> f64 {
                b[i] = b_hat[i] + si * hi;
                b[j] = b_hat[j] + sj * hj;
                let v = nll_at(&b);
                b[i] = b_hat[i];
                b[j] = b_hat[j];
                v
            };
            let mixed =
                (at(1.0, 1.0) - at(1.0, -1.0) - at(-1.0, 1.0) + at(-1.0, -1.0)) / (4.0 * hi * hj);
            h[(i, j)] = mixed;
            h[(j, i)] = mixed;
        }
    }
    h
}

/// Both anchor Hessians, from **one** [`score_core`](crate::estimation::sens_outer_gradient::score_core)
/// sweep. That function already assembles both matrices from a single provider evaluation:
///
/// * `htilde` = `H̃ = Ω⁻¹ + Σⱼ pⱼaⱼaⱼᵀ` — the first-order (Almquist) FOCEI Hessian; and
/// * `h_inner` = `Ω⁻¹ + Σⱼ (∂²Lⱼ/∂f² aⱼaⱼᵀ + ∂Lⱼ/∂f Aⱼ)` — the **exact** conditional Hessian
///   `∂²nll/∂b²`, which additionally carries the `∂²f/∂b²` term Gauss-Newton drops.
///
/// `None` when the model is outside the provider's scope, or `score_core` declines this
/// subject at runtime (off-diagonal correlated residual, magnitude × M3-censored).
fn score_core_at(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    stack: &Stack,
    b_hat: &[f64],
) -> Option<crate::estimation::sens_outer_gradient::ScoreCore> {
    // The jet is the stacked (η, κ) one under IOV, so this serves both regimes.
    let sens = if stack.is_iov() {
        crate::sens::provider::subject_sensitivities_iov(model, subject, &params.theta, b_hat)?
    } else {
        crate::sens::provider::subject_sensitivities(model, subject, &params.theta, b_hat)?
    };
    crate::estimation::sens_outer_gradient::score_core(
        model,
        subject,
        params,
        &sens,
        stack.d(),
        &stack.omega_joint_inv,
        b_hat,
        model.residual_error_eta,
    )
}

/// The Hessian that scales the quadrature grid and enters `½log|H|`, per the anchor.
///
/// * [`HessianAnchor::Exact`] → the exact conditional Hessian `∂²nll/∂b²`. This is
///   Laplace / adaptive GH quadrature. Taken **analytically** from `score_core`'s `h_inner`
///   where the provider reaches, and from the `2d²+1`-evaluation
///   [`fd_posterior_hessian`] sweep only outside it (TTE, categorical, and the two
///   per-subject `score_core` declines) — where FD is the only thing that works, since the
///   likelihood has no `PkNum` twin.
/// * [`HessianAnchor::GaussNewton`] → the Almquist `H̃`, the *same* Hessian FOCEI's own
///   objective and gradient use, so `focei` with `n_agq > 1` is a genuine quadrature
///   refinement of FOCEI.
///
/// **The two anchors stay different matrices — that is the estimator distinction.** Sharing
/// `score_core` unifies the *computation*, not the result: `laplace, n_agq = 1` is NONMEM
/// `LAPLACIAN INTER` and `focei, n_agq = 1` is plain FOCEI, and it is exactly the anchor that
/// separates them. Reading `htilde` here for `Exact` would silently turn Laplace into FOCEI
/// while it kept reporting Laplace OFVs.
///
/// Returns `None` when a Gauss-Newton anchor is out of the provider's scope (the caller then
/// yields the NLL sentinel, and `api::check_model_options` rejects `focei, n_agq > 1` outside
/// that scope up front so this does not surprise a fit at runtime). The `Exact` arm always
/// yields a matrix, falling back to FD.
fn anchor_hessian(
    anchor: HessianAnchor,
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    stack: &Stack,
    b_hat: &[f64],
    scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> Option<DMatrix<f64>> {
    match anchor {
        HessianAnchor::Exact => {
            // Gate on the **model-level** predicate before attempting the provider: an
            // out-of-scope model (TTE, categorical — precisely the ones quadrature exists
            // for) would otherwise pay a doomed `subject_sensitivities` + `score_core` call
            // on every evaluation before falling back.
            if analytic_score_supported(model) {
                if let Some(core) = score_core_at(model, subject, params, stack, b_hat) {
                    return Some(core.h_inner);
                }
            }
            Some(fd_posterior_hessian(
                model, subject, params, stack, b_hat, scratch, schedule,
            ))
        }
        HessianAnchor::GaussNewton => {
            Some(score_core_at(model, subject, params, stack, b_hat)?.htilde)
        }
    }
}

/// AGQ marginal NLL for one subject, in the same "without constant" units as
/// [`crate::stats::likelihood::foce_subject_nll`] — see the module docs for the derivation.
///
/// `eta_hat` is the converged EBE mode from the shared inner loop; AGQ does **not**
/// re-optimise it, it only lays the grid around it. `nodes`/`log_weights` are the 1-D
/// Gauss–Hermite rule, hoisted by the caller so the eigenproblem is solved once per
/// population rather than once per subject.
pub(crate) fn agq_subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    stack: &Stack,
    b_hat: &[f64],
    nodes: &[f64],
    log_weights: &[f64],
    anchor: HessianAnchor,
) -> f64 {
    let d = stack.d();
    let mut scratch = pk::EventPkParams::with_capacity_for(subject);
    let schedule = cacheable_schedule(model, subject);

    // No random effects: the "integral" is a point mass and AGQ degenerates to the
    // conditional likelihood itself. (The formula below would agree — an empty tensor
    // product is the single empty node — but there is no Σ to factor, so short-circuit.)
    if d == 0 {
        return stack.nll_at(
            model,
            subject,
            params,
            b_hat,
            &mut scratch,
            schedule.as_ref(),
        );
    }

    let Some(h) = anchor_hessian(
        anchor,
        model,
        subject,
        params,
        stack,
        b_hat,
        &mut scratch,
        schedule.as_ref(),
    ) else {
        // Gauss-Newton anchor out of the provider's scope (guarded up front by
        // `check_model_options`, so this is only reachable if that guard is bypassed).
        return NLL_SENTINEL;
    };
    // `build_proposal` applies the relative jitter and — if the FD Hessian came back
    // indefinite (a loosely-converged mode, a flat direction) — falls back to the prior-scale
    // factor (`Ω`, or the block-diagonal `Ω_joint` under IOV). That fallback keeps AGQ
    // *consistent* (any invertible scale integrates to the same limit); it only costs
    // nodes-worth of efficiency, which is the right failure mode.
    let Some(proposal) = build_proposal(&h, &stack.omega_joint_inv, d) else {
        return NLL_SENTINEL;
    };

    let (_bs, terms) = agq_nodes_and_terms(
        model,
        subject,
        params,
        stack,
        b_hat,
        nodes,
        log_weights,
        &proposal,
        &mut scratch,
        schedule.as_ref(),
    );

    let nll = 0.5 * d as f64 * PI.ln() + 0.5 * proposal.log_det_inv_scale - logsumexp(&terms);
    if nll.is_finite() {
        nll
    } else {
        NLL_SENTINEL
    }
}

/// Sweep the tensor grid once, returning each node's `η_j` and its log-term
/// `t_j = Σ_k log w_{j,k} + ‖z_j‖² − nll(η_j)`.
///
/// The single place the grid is materialised. The objective ([`agq_subject_nll`]) reduces
/// the terms with `logsumexp`; the gradient ([`agq_subject_packed_gradient`]) additionally
/// needs the `η_j` themselves and turns the same terms into softmax weights. Sharing one
/// sweep is what guarantees the gradient is differentiating the objective that was actually
/// evaluated, rather than a grid that drifted from it.
#[allow(clippy::too_many_arguments)]
fn agq_nodes_and_terms(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    stack: &Stack,
    b_hat: &[f64],
    nodes: &[f64],
    log_weights: &[f64],
    proposal: &crate::estimation::importance_sampling::Proposal,
    scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let d = stack.d();
    let n = nodes.len();
    let cap = grid_size(n, d);
    let mut bs = Vec::with_capacity(cap);
    let mut terms = Vec::with_capacity(cap);
    let mut idx = vec![0usize; d];
    let mut z = vec![0.0f64; d];
    let mut step = vec![0.0f64; d];

    loop {
        let mut log_w = 0.0;
        let mut z_sq = 0.0;
        for k in 0..d {
            let zk = nodes[idx[k]];
            z[k] = zk;
            z_sq += zk * zk;
            log_w += log_weights[idx[k]];
        }
        // b_j = b̂ + √2 · Σ^{1/2} · z_j — the adaptive transform, over the whole stacked
        // vector (η, and every occasion's κ under IOV).
        proposal.apply_l_sigma(&z, &mut step, std::f64::consts::SQRT_2);
        let b: Vec<f64> = (0..d).map(|k| b_hat[k] + step[k]).collect();

        let nll = stack.nll_at(model, subject, params, &b, scratch, schedule);
        // A diverged node returns `individual_nll`'s 1e20 sentinel, which lands here as a
        // ~−1e20 log-term — negligible under logsumexp unless *every* node diverged, in
        // which case the subject correctly reports the sentinel back.
        terms.push(log_w + z_sq - nll);
        bs.push(b);

        // Mixed-radix increment over the d-dimensional tensor grid.
        let mut k = 0;
        while k < d {
            idx[k] += 1;
            if idx[k] < n {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
        if k == d {
            break;
        }
    }
    (bs, terms)
}

/// Population AGQ objective: `Σ_i agq_subject_nll_i`. The outer loop doubles this into an
/// OFV, exactly as it does [`crate::estimation::outer_optimizer::pop_nll`].
///
/// Parallel over subjects (the grid sweep stays serial *within* a subject, matching
/// importance sampling), then reduced **serially in subject order** — a rayon `.sum()`
/// folds along thread-count-dependent split boundaries and f64 addition is not associative,
/// which would make the OFV depend on the thread count (#703).
pub fn agq_population_nll(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[nalgebra::DVector<f64>],
    kappas: &[Vec<nalgebra::DVector<f64>>],
    n_nodes: usize,
    anchor: HessianAnchor,
) -> f64 {
    let (nodes, weights) = gauss_hermite(n_nodes);
    let log_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();

    let per_subject: Vec<f64> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let subj_kappas = kappas.get(i).map(Vec::as_slice).unwrap_or(&[]);
            let stack = Stack::new(model, params, subj_kappas.len());
            let b_hat = stack_mode(eta_hats[i].as_slice(), subj_kappas);
            agq_subject_nll(
                model,
                subject,
                params,
                &stack,
                &b_hat,
                &nodes,
                &log_weights,
                anchor,
            )
        })
        .collect();
    per_subject.iter().sum()
}

/// Assemble the stacked mode `b̂ = [η̂, κ̂₁ … κ̂_K]` from what the shared inner loop already
/// returns. AGQ does not re-optimise either piece — `find_ebe_iov` converges the joint
/// (η, κ) mode, which is exactly the point the grid is centred on.
fn stack_mode(eta_hat: &[f64], kappas: &[nalgebra::DVector<f64>]) -> Vec<f64> {
    let mut b = eta_hat.to_vec();
    for k in kappas {
        b.extend_from_slice(k.as_slice());
    }
    b
}

// ---------------------------------------------------------------------------
// Analytic outer gradient
// ---------------------------------------------------------------------------
//
// # The identity
//
// AGQ's per-subject objective is `F = (d/2)·logπ + ½·log|H| − LSE`, and the node
// log-terms are `t_j = log w_j + ‖z_j‖² − nll(η_j)`. Differentiating w.r.t. the packed
// population parameters `x` with the **nodes held fixed** (η̂ and H constant), the first
// two terms drop out and the third is a softmax:
//
// ```text
//   ∂F/∂x  =  −∂LSE/∂x  =  Σ_j ŵ_j · ∂nll(η_j; x)/∂x ,     ŵ_j = exp(t_j) / Σ_k exp(t_k)
// ```
//
// The `ŵ_j` are exactly the posterior weights at the nodes, so this is the **Fisher /
// Louis identity**: `∂(−log ∫e^{−nll})/∂x = E_{p(η|y)}[∂nll/∂x]`, with the quadrature
// supplying the expectation. Each `∂nll(η_j)/∂x` is a *fixed-η* gradient — no EBE re-solve,
// no `log|H̃|` term, no EBE-response — which the `Dual2` provider already gives us exactly.
//
// # The grid also moves, and that term is not optional
//
// The nodes are built from `(η̂(x), H(x))`, so the total derivative also carries
// `∂F/∂η̂ · dη̂/dx` and `∂F/∂H · dH/dx`. Both **vanish in the exact-quadrature limit** — an
// exactly-integrated `∫exp(−nll)` does not care where you centre or how you scale the grid;
// it is a change of variables — so they are identically the quadrature error's sensitivity,
// and the explicit `½·log|H|` term cancels against the nodes' `H`-dependence rather than
// either vanishing alone.
//
// At finite `n` they do not vanish, and they are **not** small. The fixed-node score alone,
// measured against a finite-difference of the real objective on warfarin, is **26% wrong at
// n = 1** (where the rule *is* Laplace and the cancellation fails completely), 0.8% at n = 3.
// So [`grid_response_correction`] supplies the term, and the gradient is the exact total
// derivative at every `n_agq`, n = 1 included: 0.17% / 0.16% / 0.002% / 0.0006% at
// n = 1/3/5/7, i.e. at the FD reference's own noise floor.
//
// The subtlety that makes or breaks it: `H = ∂²nll/∂η²|_{η̂(x)}` depends on `x` **twice** —
// explicitly through θ/Ω/σ, and implicitly through the mode. Differencing only the explicit
// half is not a partial improvement, it is *worse than omitting the term* (26% → 62%). The
// mode must be moved along `dη̂/dx` too. See [`grid_response_correction`].
//
// # Cost
//
// The score is one `subject_sensitivities` call per node — the same order as one objective
// evaluation. The correction adds `2·n_free` Hessian rebuilds and grid sweeps, with **no**
// inner re-solve (that is what `dη̂/dx` buys). The `reconverged_fd_gradient` it replaces
// costs `2·n_free` *full population objective* evaluations, each re-solving every subject's
// inner loop.

/// Alias kept for `build_info::gradient_method_outer`, which reports scope by model.
/// Identical to [`analytic_score_supported`] — the predicate is purely model-level.
pub fn analytic_score_supported_model(model: &CompiledModel) -> bool {
    analytic_score_supported(model)
}

/// Whether the AGQ/Laplace analytic score applies to `model`.
///
/// This is **the same predicate FOCE/FOCEI gate on** — a single call to
/// [`analytic_outer_gradient_available`](crate::sens::provider::analytic_outer_gradient_available),
/// with no separate copy. That is deliberate and load-bearing for scope parity: widening the
/// analytic scope (a new PK model, a new endpoint family, a relaxed magnitude bound) in that
/// one function extends FOCE, FOCEI, AGQ and Laplace **together**, with no per-estimator list
/// here to drift out of step (#251).
///
/// It already covers IOV without a separate arm. `analytic_outer_gradient_available`'s final
/// clause is `sens_supported(model) || iov_sens_supported(model)`, and `sens_supported` is
/// always false under IOV (both `analytical_supported` and `ode_analytical_supported` bail on
/// `n_kappa != 0`), so for an IOV model it reduces exactly to `iov_sens_supported` AND the
/// shared `!fd` / `!has_non_gaussian` / magnitude bails — the whole IOV scope
/// (`iov_sens_supported` itself folds in the closed-form, ODE-IOV and transit-via-ODE-twin
/// cases). The earlier hand-replicated IOV arm (#251 review #9) was therefore pure redundancy
/// and a drift hazard; delegating supersedes it.
pub fn analytic_score_supported(model: &CompiledModel) -> bool {
    let provider = crate::sens::provider::analytic_outer_gradient_available(model);
    // Everything the FOCE/FOCEI outer gradient can do analytically, the AGQ/Laplace score
    // can now do too: the per-observation residual chain comes from the SAME
    // `sens_outer_gradient::score_core`, so M3-BLOQ, `iiv_on_ruv`, a custom / time-varying
    // σ magnitude, LTBS and a correlated residual all ride the analytic path rather than
    // falling back to the fixed-b FD score. Scope parity holds by construction — there is
    // no per-family list here to drift out of step with `analytic_outer_gradient_available`.
    //
    // FREM included: a covariate pseudo-observation row is scored against `θ[ti] + η[ei]`
    // with the dedicated `EPSCOV` variance, and BOTH halves are now threaded through the
    // shared machinery — the jet by `provider::apply_frem_pseudo_obs_jet`, the variance by
    // `score_core`'s FREM override. (This was a live defect in the FOCE/FOCEI outer
    // gradient too, not only a missing AGQ capability: the analytic gradient was
    // differentiating the PK likelihood on rows the objective scores as covariate
    // pseudo-observations. It went unnoticed because FREM models are conventionally fit
    // with SAEM, which never asks for that gradient.)
    //
    // Two runtime (per-subject, not caught by this model-level predicate) declines still
    // land on FD: a genuinely off-diagonal correlated subject, and magnitude ×
    // M3-censored — both inside `score_core`, returning `None`. `accumulate_fixed_eta_
    // packed_gradient` falls back to `accumulate_fixed_b_packed_gradient_fd` for just
    // that subject when this happens, not the whole population (#251 review #8).
    provider
}

/// AGQ always drives the outer loop with its **own** gradient — for every model, and at every
/// `n_agq`.
///
/// There is deliberately no scope gate here. The two ingredients both work universally:
///
/// * the fixed-η score is analytic where the provider reaches
///   ([`analytic_score_supported`]) and finite-differenced *at fixed η* otherwise
///   ([`accumulate_fixed_b_packed_gradient_fd`]) — the latter is correct for any likelihood
///   ferx can evaluate, including TTE and categorical; and
/// * `dη̂/dx` comes from the implicit-function theorem (`−H⁻¹·∂²nll/∂η∂x`), analytic where the
///   provider reaches and finite-differenced otherwise ([`eta_dx`]).
///
/// Crucially **neither path re-solves the inner loop**, which is the cost that makes
/// `reconverged_fd_gradient` expensive. Letting the models AGQ exists for — non-Gaussian
/// endpoints, which are precisely the ones outside the `Dual2` provider — fall back to that
/// gradient would have made the headline use case the slow one.
///
/// **Node-count independent.** With [`grid_response_correction`] supplying the `∂Φ/∂H·dH/dx`
/// term, the gradient is the exact total derivative at every `n_agq`, `n_agq = 1` included
/// (where that term is exactly `½·d log|H|/dx`, the Laplace log-determinant).
pub fn analytic_gradient_available(model: &CompiledModel) -> bool {
    // Kept as a predicate (rather than inlining `true`) so a future model class that genuinely
    // cannot supply `∂nll/∂x` at fixed η has one place to opt out; `population_gradient`'s
    // `reconverged_fd_gradient` fallback stays wired up behind it.
    let _ = model;
    true
}

/// Finite-differenced θ/σ score at a **fixed η** — the universal fallback when the analytic
/// chain rule does not apply (TTE, categorical, M3, LTBS, FREM, `block_sigma`, ODE models).
///
/// Central-differences `individual_nll(η; θ, σ)` in the *packed* coordinates, so the log-θ and
/// log-σ chain rules come out automatically. The Ω block is **not** redone here: it is closed
/// form from the η-prior and the caller has already accumulated it exactly.
///
/// Predictions are σ-independent, so the σ perturbations reuse nothing extra — each is one
/// likelihood evaluation. No inner re-solve, which is the whole point.
#[allow(clippy::too_many_arguments)]
fn accumulate_fixed_b_packed_gradient_fd(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    template: &ModelParameters,
    stack: &Stack,
    b: &[f64],
    weight: f64,
    n_theta: usize,
    sigma_start: usize,
    out: &mut [f64],
) -> Option<()> {
    use crate::estimation::parameterization::{pack_params, packed_fixed_mask, unpack_params};

    let x = pack_params(params);
    let fixed = packed_fixed_mask(template);
    let n_sigma = params.sigma.values.len();
    let mut scratch = pk::EventPkParams::with_capacity_for(subject);
    let schedule = cacheable_schedule(model, subject);

    let mut nll_at = |p: &ModelParameters| -> f64 {
        stack.nll_at(model, subject, p, b, &mut scratch, schedule.as_ref())
    };

    // θ and σ coordinates only — the Ω block is exact in closed form and already added.
    let coords = (0..n_theta).chain(sigma_start..sigma_start + n_sigma);
    for k in coords {
        if fixed[k] {
            continue;
        }
        let h = 1e-6 * (1.0 + x[k].abs());
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[k] += h;
        xm[k] -= h;
        let d = (nll_at(&unpack_params(&xp, template)) - nll_at(&unpack_params(&xm, template)))
            / (2.0 * h);
        if d.is_finite() {
            out[k] += weight * d;
        }
    }
    Some(())
}

/// Packed gradient of `individual_nll` at a **fixed** `eta`, accumulated as
/// `out += weight · ∂nll(η)/∂x`.
///
/// This is the fixed-η score: the θ block via the provider's exact `∂f/∂θ` chained through
/// `∂nll/∂f`, the σ block in closed form, and the Ω block from the η-prior alone
/// (`½(−zzᵀ + Ω⁻¹)`, mapped to Cholesky-packed space by the shared
/// [`crate::estimation::parameterization::chol_pack`]). Deliberately carries **no**
/// `log|H̃|` and **no** EBE-response term — those belong to the Laplace marginal, not to
/// `nll(η)` at a fixed η.
///
/// Returns `None` when the subject is outside the provider's scope, which drops the whole
/// population to the FD gradient (all-or-nothing, like `population_gradient_sens`).
fn accumulate_fixed_eta_packed_gradient(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    template: &ModelParameters,
    stack: &Stack,
    eta: &[f64],
    weight: f64,
    out: &mut [f64],
) -> Option<()> {
    use crate::estimation::parameterization::theta_packs_log;

    let b = eta; // the integration variable: η, or the stacked [η, κ₁..κ_K] under IOV
    let n_obs = subject.observations.len();
    let n_theta = params.theta.len();
    let n_sigma = params.sigma.values.len();
    let n_eta = stack.n_eta;
    let (eta_part, kappas) = stack.split(b);

    // ── Ω block ──────────────────────────────────────────────────────────────────────
    // From the η-prior `½(ηᵀΩ⁻¹η + log|Ω|)`, the only place Ω enters `nll` at a fixed b:
    //   ∂/∂Ω [½ ηᵀΩ⁻¹η] = −½·zzᵀ   and   ∂/∂Ω [½ log|Ω|] = ½·Ω⁻¹,   z = Ω⁻¹η.
    // Data-independent, so it is accumulated even for a subject with no observations.
    let z = &params.omega.inv * nalgebra::DVector::from_column_slice(eta_part);
    let mut m_omega = DMatrix::zeros(n_eta, n_eta);
    for r in 0..n_eta {
        for c in 0..n_eta {
            m_omega[(r, c)] = 0.5 * (-z[r] * z[c] + params.omega.inv[(r, c)]);
        }
    }
    let omega_start = n_theta;
    let og = crate::estimation::parameterization::chol_pack(
        &m_omega,
        &params.omega.chol,
        params.omega.diagonal,
    );
    for (k, &v) in og.iter().enumerate() {
        out[omega_start + k] += weight * v;
    }
    let sigma_start = omega_start + og.len();

    // ── Ω_iov block ──────────────────────────────────────────────────────────────────
    // The κ prior is `½(Σ_k κ_kᵀΩ_iov⁻¹κ_k + K·log|Ω_iov|)` — the *same* shape as the η
    // prior, but summed over the K occasions, which share one Ω_iov:
    //   ∂/∂Ω_iov = ½·(−Σ_k z_k z_kᵀ + K·Ω_iov⁻¹),   z_k = Ω_iov⁻¹κ_k.
    // Mapped to Cholesky-packed space by the same `chol_pack`. This is the only block that
    // is genuinely new under IOV; θ and σ are unchanged (see below).
    let iov_start = sigma_start + n_sigma;
    if stack.is_iov() {
        let iov = params
            .omega_iov
            .as_ref()
            .expect("stack.is_iov() ⟹ omega_iov present");
        let k_occ = kappas.len();
        let n_iov = stack.n_kappa;
        let mut m_iov = DMatrix::zeros(n_iov, n_iov);
        for kap in &kappas {
            let zk = &iov.inv * nalgebra::DVector::from_column_slice(kap);
            for r in 0..n_iov {
                for c in 0..n_iov {
                    m_iov[(r, c)] -= 0.5 * zk[r] * zk[c];
                }
            }
        }
        for r in 0..n_iov {
            for c in 0..n_iov {
                m_iov[(r, c)] += 0.5 * (k_occ as f64) * iov.inv[(r, c)];
            }
        }
        let ig = crate::estimation::parameterization::chol_pack(&m_iov, &iov.chol, iov.diagonal);
        for (k, &v) in ig.iter().enumerate() {
            out[iov_start + k] += weight * v;
        }
    }

    if n_obs == 0 {
        return Some(());
    }

    // Off the analytic score's scope (a non-Gaussian endpoint, M3, LTBS, an ODE model, …):
    // finite-difference the θ and σ blocks of `nll` at this **fixed b** instead. Still no
    // inner re-solve — that is the expensive thing, and neither path does one — so AGQ keeps
    // its own gradient for *every* model rather than dropping to `reconverged_fd_gradient`.
    // This matters most for exactly the endpoints AGQ exists to serve: TTE and categorical
    // are outside the `Dual2` provider, and it would be perverse for them to be the slow case.
    if !analytic_score_supported(model) {
        return accumulate_fixed_b_packed_gradient_fd(
            model,
            subject,
            params,
            template,
            stack,
            b,
            weight,
            n_theta,
            sigma_start,
            out,
        );
    }

    // Exact ∂f/∂θ at *this* b. Neither provider entry point assumes the mode — the η = η̂
    // requirement in `sens_outer_gradient` lives in its `theta_block`, not here. The IOV
    // entry point takes the *stacked* (η, κ) vector and returns the same `ObsSens`, so
    // everything below is identical for both.
    //
    // `analytic_score_supported` is a *model*-level scope check; `score_core` (and, more
    // rarely, the sensitivity provider itself) can still decline a specific *subject* at
    // runtime — a genuinely off-diagonal correlated residual, or magnitude × an
    // M3-censored row. Falling through those `?`s used to propagate `None` out of this
    // whole function, which `agq_population_gradient` treats as "any subject declined" and
    // answers by dropping the **entire population** onto `reconverged_fd_gradient` — the
    // `2·n_free` full-objective, every-subject-inner-resolve fallback this module exists to
    // avoid. FOCE/FOCEI stopped doing that population-wide bail for exactly this cost
    // (`population_gradient_sens_mixed`'s per-subject `subject_reconverged_fd_gradient`
    // salvage); AGQ's fixed-b FD score is this function's own equivalent per-subject
    // salvage, and it is already right here — the Ω/Ω_iov block above is unaffected by
    // which θ/σ path runs, so falling back for just this subject is exactly the state
    // `accumulate_fixed_b_packed_gradient_fd`'s own docs assume it's called in (#251
    // review #8).
    let sens = if stack.is_iov() {
        crate::sens::provider::subject_sensitivities_iov(model, subject, &params.theta, b)
    } else {
        crate::sens::provider::subject_sensitivities(model, subject, &params.theta, b)
    };
    let Some(sens) = sens else {
        return accumulate_fixed_b_packed_gradient_fd(
            model,
            subject,
            params,
            template,
            stack,
            b,
            weight,
            n_theta,
            sigma_start,
            out,
        );
    };

    // The per-observation scalar likelihood chain. `score_core` is the half of
    // `sens_outer_gradient::prepare_stacked` that does NOT build the FOCEI `log|H̃|`
    // machinery, so a node pays no Cholesky inverse — and it already carries the
    // M3-censored (`−logΦ(z)`), `iiv_on_ruv`, custom-σ-magnitude, LTBS and
    // correlated-residual chains that this function used to hand-roll for the plain
    // Gaussian case alone. That hand-rolled chain is why AGQ used to gate those five
    // families onto the fixed-b FD score even though FOCE/FOCEI had them analytic.
    //
    // `n_eta` is the STACKED dimension and `Ω⁻¹` the stacked prior precision, so IOV needs
    // no special case: `ObsSens` is already stacked, and `η_ruv` lives in the BSV block, so
    // `residual_error_eta` stays a valid stacked index (the same argument the IOV caller of
    // `prepare_stacked` makes).
    let core = crate::estimation::sens_outer_gradient::score_core(
        model,
        subject,
        params,
        &sens,
        stack.d(),
        &stack.omega_joint_inv,
        b,
        model.residual_error_eta,
    );
    let Some(core) = core else {
        return accumulate_fixed_b_packed_gradient_fd(
            model,
            subject,
            params,
            template,
            stack,
            b,
            weight,
            n_theta,
            sigma_start,
            out,
        );
    };
    let et = &core.et;

    // θ block. Two channels:
    //
    //  (a) through the prediction — `∂L/∂f · ∂f/∂θ`. `ErrTerms` defines `α = 2·∂L/∂f` for
    //      EVERY endpoint family (for a censored row it is `2·g1` off the `−logΦ` kernel),
    //      so `½α` is the residual chain the old code spelled out for Gaussian only.
    //
    //  (b) DIRECT — a custom / time-varying σ magnitude `mult(θ)` makes `R` depend on θ
    //      without passing through `f`, contributing `∂L/∂R · ∂R/∂θ`. This term was not
    //      approximated before, it was ABSENT: the θ gradient of a magnitude model was
    //      simply missing a channel. `score_core` declines magnitude × M3 upstream, so the
    //      quantified `∂L/∂R` below is the only form needed here.
    for m in 0..n_theta {
        let mut d = 0.0;
        for j in 0..n_obs {
            d += 0.5 * et[j].alpha * sens.obs[j].df_dtheta[m];
            if !et[j].dr_dtheta.is_empty() {
                let (r, eps) = (et[j].r, et[j].eps);
                d += 0.5 * (r - eps * eps) / (r * r) * et[j].dr_dtheta[m];
            }
        }
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        out[m] += weight * d * dtheta_dx;
    }

    // σ block, log-packed. σ reaches `nll` only through the residual variance, so this is a
    // closed-form scalar computation at fixed `f` — no model evaluation, no inner solve.
    let g_sigma = crate::estimation::sens_outer_gradient::data_sigma_gradient(
        model, subject, params, &sens, &core,
    );
    for k in 0..n_sigma {
        out[sigma_start + k] += weight * g_sigma[k] * params.sigma.values[k];
    }

    Some(())
}

/// The **grid-response** term `∂Φ/∂H · dH/dx`, the one piece the fixed-node score omits.
///
/// Writing `F = Φ(x, H(x))` with the nodes built from `H`, the exact total derivative is
///
/// ```text
///   dF/dx = ∂Φ/∂x|_H          ← the fixed-node score (analytic; see above)
///         + ∂Φ/∂H · dH/dx     ← this function
///         + ∂Φ/∂η̂ · dη̂/dx     ← = Σ_j ŵ_j ∇_η nll(η_j), the posterior-mean score = 0
/// ```
///
/// The `∂Φ/∂η̂` factor is the posterior-mean score, which is zero by the Bartlett identity —
/// exactly so at `n_agq = 1`, where η̂ *is* the mode. So the `H`-response is all that stands
/// between the fixed-node score and the exact gradient. It is not a small correction: without
/// it the gradient is 26% wrong at `n_agq = 1`.
///
/// It is computed by central-differencing `Φ` in `x` **through the grid alone**: the grid
/// (`η̂`, `H`) is rebuilt at the perturbed parameters while `nll` stays at the original ones,
/// which isolates the response term from the direct dependence already covered analytically.
///
/// **`H` depends on `x` twice, and both halves matter.** `H = ∂²nll/∂η²|_{η̂(x)}` varies with
/// `x` explicitly (through θ/Ω/σ) *and* implicitly through the mode `η̂(x)`. Differencing only
/// the explicit half is not a partial improvement — it is **worse than omitting the term
/// entirely** (26% → 62% on warfarin at n = 1), because the two halves substantially cancel.
/// So the mode is moved too, along the exact analytic `dη̂/dx` from
/// [`crate::estimation::sens_outer_gradient::subject_eta_dx`] (the implicit-function
/// derivative `−H⁻¹·∂²nll/∂η∂x`, which the EBE predictor already relies on). That is what
/// keeps this **free of any inner re-solve**: it costs `2·n_free` Hessian rebuilds and grid
/// sweeps, not the `2·n_free` *full population objective* evaluations `reconverged_fd_gradient`
/// pays.
///
/// A fully analytic `dH/dx` would need `∂³nll/∂η²∂θ`; the provider stops at `∂²f/∂η∂θ`, so
/// this third order is finite-differenced. `analytic_gradient_matches_fd_at_every_node_count`
/// measures the result against a finite-difference of the real objective.
///
/// At `n_agq = 1` the grid objective collapses to `½·log|H|` (the single node sits at η̂), so
/// this reduces to `½·d log|H|/dx` — precisely the Laplace log-determinant term FOCE/FOCEI
/// carry analytically.
///
/// # The node-response term is analytic, not a grid re-sweep
///
/// Writing `Φ = ½log|Σ⁻¹| − logΣⱼexp(tⱼ)` with `tⱼ = log wⱼ + ‖zⱼ‖² − nll(bⱼ; x₀)` and
/// `bⱼ(x) = b̂(x) + √2·Σ^{1/2}(H(x))·zⱼ`, the exact derivative separates:
///
/// ```text
///   dΦ/dx_k = ½·d log|Σ⁻¹|/dx_k  +  Σⱼ softmaxⱼ · ⟨ ∂nll/∂b|_{bⱼ} , dbⱼ/dx_k ⟩
/// ```
///
/// because `nll` is held at `x₀` — only the node *positions* move. Two consequences:
///
/// * `∂nll/∂b|_{bⱼ}` is the **inner EBE gradient**, one call per node, and it does not depend
///   on `k` — so it is computed **once** and reused across every packed coordinate.
/// * `x ↦ (log|Σ⁻¹|, bⱼ)` is a cheap *analytic* map (a Cholesky and `d` back-solves per node,
///   no likelihood), so finite-differencing **it** costs no predictions at all.
///
/// That replaces `2·n_free · G` likelihood evaluations with `G` gradient evaluations plus
/// `2·n_free` Hessian rebuilds — the dominant cost at `n_agq > 1`, where `G = n_agq^d` grows
/// as a tensor product (243 nodes at `d = 5, n = 3`). It also removes the *only* place the
/// old code differenced a **finite-differenced** quantity in `x`: with the analytic `h_inner`
/// anchor, `H` is exact and the remaining FD is of an exact function.
///
/// Falls back to the previous `phi_grid` re-sweep when the per-node gradient is out of the
/// provider's scope (TTE, categorical, …) — the same all-or-nothing boundary the anchor uses.
#[allow(clippy::too_many_arguments)]
fn grid_response_correction(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    template: &ModelParameters,
    stack: &Stack,
    anchor: HessianAnchor,
    x: &[f64],
    b_hat: &[f64],
    nodes: &[f64],
    log_weights: &[f64],
    bs: &[Vec<f64>],
    softmax: &[f64],
    scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
    out: &mut [f64],
) -> Option<()> {
    use crate::estimation::parameterization::{packed_fixed_mask, unpack_params};

    let fixed = packed_fixed_mask(template);

    // `H = ∂²nll/∂η²|_{η̂(x)}` depends on `x` **twice**: explicitly through θ/Ω/σ, and
    // implicitly through the mode `η̂(x)`. Differencing only the explicit half is not a
    // partial improvement — it is *wrong*, and measurably worse than omitting the term
    // (26% → 62% on warfarin at n = 1). So the mode is moved too, along the exact analytic
    // `dη̂/dx` (the implicit-function-theorem derivative `−H⁻¹·∂²nll/∂η∂x`, which the
    // sensitivity provider already supplies and the EBE predictor already relies on). No
    // inner re-solve is needed.
    let db_dx = eta_dx(
        model, subject, params, template, stack, x, b_hat, scratch, schedule,
    )?;

    // `∂nll/∂b` at every base node — the node-response factor. Independent of `k`, so this is
    // the whole per-node likelihood cost of the correction, paid once instead of `2·n_free`
    // times. `None` if any node is out of the inner provider's scope; the loop then takes the
    // `phi_grid` re-sweep for every coordinate (all-or-nothing, matching the anchor).
    // Hoisted once for the whole sweep — see `node_nll_gradient`.
    let mult = model.ruv_obs_mult(subject, &params.theta);
    let node_grads: Option<Vec<Vec<f64>>> = bs
        .iter()
        .map(|b| node_nll_gradient(model, subject, params, stack, b, schedule, mult.as_deref()))
        .collect();

    let d = stack.d();
    let mut z = vec![0.0f64; d];
    let (mut off_p, mut off_m) = (vec![0.0f64; d], vec![0.0f64; d]);

    for k in 0..x.len() {
        if fixed[k] {
            continue;
        }
        // Truncation-dominated: `Φ` differences `log|H|`, which curves sharply in the packed
        // coordinates, so the step wants to be *small* (see `AGQ_GRID_FD_STEP`). With the
        // analytic `h_inner` anchor there is no FD error floor underneath it to trade against.
        let step = AGQ_GRID_FD_STEP * (1.0 + x[k].abs());
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[k] += step;
        xm[k] -= step;

        // The mode moves with the parameters: b̂(x ± h) ≈ b̂ ± h · db̂/dx_k.
        let ep: Vec<f64> = (0..b_hat.len())
            .map(|i| b_hat[i] + step * db_dx[k][i])
            .collect();
        let em: Vec<f64> = (0..b_hat.len())
            .map(|i| b_hat[i] - step * db_dx[k][i])
            .collect();

        let pp = unpack_params(&xp, template);
        let pm = unpack_params(&xm, template);
        // Ω_joint moves with the parameters too, so rebuild the stack at each perturbed point.
        let sp = Stack::new(model, &pp, stack.n_occ);
        let sm = Stack::new(model, &pm, stack.n_occ);
        let (Some(hp), Some(hm)) = (
            anchor_hessian(anchor, model, subject, &pp, &sp, &ep, scratch, schedule),
            anchor_hessian(anchor, model, subject, &pm, &sm, &em, scratch, schedule),
        ) else {
            continue; // GN anchor out of scope at this perturbed point — no correction
        };

        // `nll` stays at the ORIGINAL params — the direct x-dependence is already covered
        // analytically by the fixed-η score — but the grid (centre and scale) is the
        // perturbed one, which is exactly the response term we are after.
        let r = match &node_grads {
            // Analytic node response: difference only the cheap `x ↦ (log|Σ⁻¹|, bⱼ)` map and
            // contract each node's displacement with its own `∂nll/∂b`. No likelihood
            // evaluations inside this loop at all.
            Some(gs) => {
                let (Some(prop_p), Some(prop_m)) = (
                    build_proposal(&hp, &sp.omega_joint_inv, d),
                    build_proposal(&hm, &sm.omega_joint_inv, d),
                ) else {
                    continue; // degenerate perturbed Hessian contributes no correction
                };
                let mut acc =
                    0.5 * (prop_p.log_det_inv_scale - prop_m.log_det_inv_scale) / (2.0 * step);
                for (j, g) in gs.iter().enumerate() {
                    if softmax[j] == 0.0 {
                        continue; // underflowed node — contributes nothing
                    }
                    grid_z_at(j, nodes, d, &mut z);
                    prop_p.apply_l_sigma(&z, &mut off_p, std::f64::consts::SQRT_2);
                    prop_m.apply_l_sigma(&z, &mut off_m, std::f64::consts::SQRT_2);
                    let mut dot = 0.0;
                    for i in 0..d {
                        // dbⱼ/dx_k = db̂/dx_k + √2·d(Σ^{1/2})/dx_k · zⱼ, both differenced here.
                        let dbj = ((ep[i] + off_p[i]) - (em[i] + off_m[i])) / (2.0 * step);
                        dot += g[i] * dbj;
                    }
                    acc += softmax[j] * dot;
                }
                acc
            }
            // Out of the inner provider's scope: difference the grid objective itself.
            None => {
                let phip = phi_grid(
                    model,
                    subject,
                    params,
                    stack,
                    &ep,
                    nodes,
                    log_weights,
                    &hp,
                    &sp.omega_joint_inv,
                    scratch,
                    schedule,
                );
                let phim = phi_grid(
                    model,
                    subject,
                    params,
                    stack,
                    &em,
                    nodes,
                    log_weights,
                    &hm,
                    &sm.omega_joint_inv,
                    scratch,
                    schedule,
                );
                let (Some(phip), Some(phim)) = (phip, phim) else {
                    continue; // a degenerate perturbed Hessian contributes no correction
                };
                (phip - phim) / (2.0 * step)
            }
        };
        if r.is_finite() {
            out[k] += r;
        }
    }
    Some(())
}

/// `zⱼ` for tensor-grid node `j`, reconstructed from its mixed-radix index.
///
/// [`agq_nodes_and_terms`] enumerates the grid with a mixed-radix counter whose **digit 0
/// increments fastest**, so node `j` has `idx[k] = (j / nᵏ) mod n`. Recomputing it here (rather
/// than storing every `zⱼ`) keeps the grid-response term allocation-free at `MAX_AGQ_GRID`
/// nodes. `grid_z_at_matches_the_sweep_enumeration` pins the two orderings together — they
/// must not drift, or each node's displacement would be contracted with another node's
/// gradient.
#[inline]
fn grid_z_at(j: usize, nodes: &[f64], d: usize, out: &mut [f64]) {
    let n = nodes.len();
    let mut rem = j;
    for slot in out.iter_mut().take(d) {
        *slot = nodes[rem % n];
        rem /= n;
    }
}

/// `∂nll/∂b` at one quadrature node — the **inner EBE gradient**, which is exactly the
/// derivative the node-response term contracts against.
///
/// This is the same entry point the inner loop minimises with, so the gradient and the `nll`
/// [`Stack::nll_at`] scores are consistent by construction rather than by a second derivation.
/// `None` outside the provider's scope; the caller then falls back to the grid re-sweep.
///
/// `schedule` and `mult` are **hoisted by the caller** and threaded through, because this runs
/// once per quadrature node — up to [`MAX_AGQ_GRID`] times per subject. The bare
/// `analytic_eta_nll_gradient` rebuilds the `EventSchedule` and recomputes the residual-magnitude
/// multiplier on every call, which is per-call setup the inner BFGS loop already learned to
/// hoist (#449 re-review #6); paying it per node made the sweep several times its own cost.
fn node_nll_gradient(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    stack: &Stack,
    b: &[f64],
    schedule: Option<&pk::event_driven::EventSchedule>,
    mult: Option<&[Vec<f64>]>,
) -> Option<Vec<f64>> {
    if stack.is_iov() {
        let iov = params.omega_iov.as_ref()?;
        crate::estimation::inner_optimizer::analytic_eta_nll_gradient_iov(
            model,
            subject,
            &params.theta,
            b,
            &params.omega,
            iov,
            &params.sigma.values,
            stack.n_eta,
            stack.n_kappa,
            stack.n_occ,
            mult,
        )
    } else {
        crate::estimation::inner_optimizer::analytic_eta_nll_gradient_with_schedule(
            model,
            subject,
            &params.theta,
            b,
            &params.omega,
            &params.sigma.values,
            schedule,
            mult,
        )
    }
}

/// `dη̂/dx` — how the EBE mode moves with the population parameters, per packed coordinate.
///
/// From the implicit function theorem: η̂ solves `∇_η nll(η̂; x) = 0`, so
///
/// ```text
///   dη̂/dx = −H⁻¹ · ∂²nll/∂η∂x ,     H = ∂²nll/∂η²|_{η̂}
/// ```
///
/// Uses the provider's exact version
/// ([`crate::estimation::sens_outer_gradient::subject_eta_dx`]) where it applies, and
/// otherwise finite-differences the mixed term `∂²nll/∂η∂x` directly — perturb `x`, take the
/// η-gradient at the *same* η̂, solve against `H`. That costs `2·n_free · 2·d` likelihood
/// evaluations and, critically, **no inner re-solve**: it is the derivative of the mode, not a
/// re-computation of it. So models outside the provider (TTE, categorical, ODE) still get the
/// cheap gradient rather than dropping to `reconverged_fd_gradient`.
#[allow(clippy::too_many_arguments)]
fn eta_dx(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    template: &ModelParameters,
    stack: &Stack,
    x: &[f64],
    b_hat: &[f64],
    scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> Option<Vec<nalgebra::DVector<f64>>> {
    use crate::estimation::parameterization::{packed_fixed_mask, unpack_params};

    // `db̂/dx` from the provider — the stacked (η, κ) entry point under IOV, the η-only one
    // otherwise. Both are the exact implicit-function derivative `−H_inner⁻¹·∂²nll/∂b∂x`, and
    // neither re-solves the inner loop: this is the *derivative* of the mode, not a
    // recomputation of it. The finite difference below is the fallback for models outside the
    // provider's scope, and uses the same exact `H` the grid is scaled by.
    if analytic_score_supported(model) {
        let exact = if stack.is_iov() {
            crate::estimation::sens_outer_gradient::subject_eta_dx_iov(
                model, subject, template, x, b_hat,
            )
        } else {
            crate::estimation::sens_outer_gradient::subject_eta_dx(
                model, subject, template, x, b_hat,
            )
        };
        if let Some(v) = exact {
            return Some(v);
        }
    }

    let d = stack.d();
    let h_mat = fd_posterior_hessian(model, subject, params, stack, b_hat, scratch, schedule);
    let h_inv = h_mat.clone().try_inverse()?;
    let fixed = packed_fixed_mask(template);

    // ∇_b nll(b̂; p) — central differences in the stacked variable, at the given parameters.
    let mut eta_grad = |p: &ModelParameters, out: &mut [f64]| {
        let mut e = b_hat.to_vec();
        for i in 0..d {
            let step = (f64::EPSILON.powf(0.25) * stack.prior_sd[i]).max(1e-5);
            e[i] = b_hat[i] + step;
            let fp = stack.nll_at(model, subject, p, &e, scratch, schedule);
            e[i] = b_hat[i] - step;
            let fm = stack.nll_at(model, subject, p, &e, scratch, schedule);
            e[i] = b_hat[i];
            out[i] = (fp - fm) / (2.0 * step);
        }
    };

    let mut out = vec![nalgebra::DVector::zeros(d); x.len()];
    let (mut gp, mut gm) = (vec![0.0f64; d], vec![0.0f64; d]);
    for k in 0..x.len() {
        if fixed[k] {
            continue;
        }
        let step = AGQ_GRID_FD_STEP * (1.0 + x[k].abs());
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[k] += step;
        xm[k] -= step;
        eta_grad(&unpack_params(&xp, template), &mut gp);
        eta_grad(&unpack_params(&xm, template), &mut gm);
        // ∂²nll/∂η∂x_k, then dη̂/dx_k = −H⁻¹ · that.
        let mixed =
            nalgebra::DVector::from_iterator(d, (0..d).map(|i| (gp[i] - gm[i]) / (2.0 * step)));
        out[k] = -(&h_inv * mixed);
    }
    Some(out)
}

/// `Φ_grid(H)` — the part of the AGQ objective that depends on `x` *only* through `H`:
/// `½·log|H| − logΣexp_j[ log w_j + ‖z_j‖² − nll(η̂ + √2·Σ^{1/2}(H)·z_j ; x₀) ]`.
///
/// `nll` is evaluated at the **original** `params`, so differencing this in `x` isolates the
/// grid's response to `H` from the direct `x`-dependence that
/// [`accumulate_fixed_eta_packed_gradient`] already covers analytically. The `(d/2)·logπ`
/// constant cancels in the difference and is omitted.
#[allow(clippy::too_many_arguments)]
fn phi_grid(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    stack: &Stack,
    b_hat: &[f64],
    nodes: &[f64],
    log_weights: &[f64],
    h_mat: &DMatrix<f64>,
    omega_inv: &DMatrix<f64>,
    scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> Option<f64> {
    let d = stack.d();
    let proposal = build_proposal(h_mat, omega_inv, d)?;
    let (_bs, terms) = agq_nodes_and_terms(
        model,
        subject,
        params,
        stack,
        b_hat,
        nodes,
        log_weights,
        &proposal,
        scratch,
        schedule,
    );
    let lse = logsumexp(&terms);
    lse.is_finite()
        .then(|| 0.5 * proposal.log_det_inv_scale - lse)
}

/// Analytic packed gradient of the AGQ objective for one subject: the posterior-weighted
/// average of fixed-η scores over the quadrature nodes, **plus** the grid-response term
/// [`grid_response_correction`] that makes it the exact total derivative at every `n_agq`
/// (see the module note above).
///
/// `out` is the population accumulator; this adds `∂F_i/∂x` into it.
#[allow(clippy::too_many_arguments)]
fn agq_subject_packed_gradient(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    template: &ModelParameters,
    stack: &Stack,
    x: &[f64],
    b_hat: &[f64],
    nodes: &[f64],
    log_weights: &[f64],
    anchor: HessianAnchor,
    out: &mut [f64],
) -> Option<()> {
    let d = stack.d();
    let mut scratch = pk::EventPkParams::with_capacity_for(subject);
    let schedule = cacheable_schedule(model, subject);

    if d == 0 {
        return accumulate_fixed_eta_packed_gradient(
            model, subject, params, template, stack, b_hat, 1.0, out,
        );
    }

    let h = anchor_hessian(
        anchor,
        model,
        subject,
        params,
        stack,
        b_hat,
        &mut scratch,
        schedule.as_ref(),
    )?;
    let proposal = build_proposal(&h, &stack.omega_joint_inv, d)?;

    // Sweep the grid once, keeping each node's b and its log-term, so the softmax weights
    // and the scores are computed on exactly the same nodes the objective used.
    let (bs, terms) = agq_nodes_and_terms(
        model,
        subject,
        params,
        stack,
        b_hat,
        nodes,
        log_weights,
        &proposal,
        &mut scratch,
        schedule.as_ref(),
    );
    let lse = logsumexp(&terms);
    if !lse.is_finite() {
        return None;
    }

    // Posterior (softmax) weights, materialised once: the fixed-node score averages against
    // them here, and the grid-response term contracts each node's displacement against the
    // same values — so both differentiate the grid the objective actually evaluated.
    let softmax: Vec<f64> = terms.iter().map(|&t| (t - lse).exp()).collect();

    for (b_j, &w) in bs.iter().zip(softmax.iter()) {
        if w == 0.0 {
            continue; // exp underflow — contributes nothing to the average
        }
        accumulate_fixed_eta_packed_gradient(model, subject, params, template, stack, b_j, w, out)?;
    }

    // …plus the grid's response to H — the only term the fixed-node score omits, and the
    // whole of the gap at n_agq = 1 (where it is exactly ½·d log|H|/dx).
    grid_response_correction(
        model,
        subject,
        params,
        template,
        stack,
        anchor,
        x,
        b_hat,
        nodes,
        log_weights,
        &bs,
        &softmax,
        &mut scratch,
        schedule.as_ref(),
        out,
    )?;
    Some(())
}

/// Analytic packed gradient of the AGQ **OFV** (`2 · Σᵢ Fᵢ`), or `None` if any subject is
/// outside the provider's scope (all-or-nothing, matching `population_gradient_sens`).
///
/// Parallel over subjects; the per-subject gradients are reduced **serially in subject
/// order** so the result cannot depend on the thread count (#703), exactly as
/// [`agq_population_nll`] does for the objective.
pub fn agq_population_gradient(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[nalgebra::DVector<f64>],
    kappas: &[Vec<nalgebra::DVector<f64>>],
    n_nodes: usize,
    anchor: HessianAnchor,
) -> Option<Vec<f64>> {
    let n_packed = x.len();
    let (nodes, weights) = gauss_hermite(n_nodes);
    let log_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();

    let per_subject: Vec<Option<Vec<f64>>> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let subj_kappas = kappas.get(i).map(Vec::as_slice).unwrap_or(&[]);
            let stack = Stack::new(model, params, subj_kappas.len());
            let b_hat = stack_mode(eta_hats[i].as_slice(), subj_kappas);
            let mut g = vec![0.0f64; n_packed];
            agq_subject_packed_gradient(
                model,
                subject,
                params,
                template,
                &stack,
                x,
                &b_hat,
                &nodes,
                &log_weights,
                anchor,
                &mut g,
            )
            .map(|()| g)
        })
        .collect();

    let mut out = vec![0.0f64; n_packed];
    for g in per_subject {
        let g = g?;
        for (o, v) in out.iter_mut().zip(g.iter()) {
            *o += 2.0 * v; // OFV = 2 · Σᵢ Fᵢ
        }
    }
    if out.iter().all(|v| v.is_finite()) {
        Some(out)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::model_parser::parse_model_string;

    // --- Phase 0 (#251): the analytic fixed-b score reaches full FOCE/FOCEI scope -------
    //
    // AGQ's score used to hand-roll a *Gaussian-only* residual chain, so five endpoint
    // families that FOCE/FOCEI already handled analytically (M3-BLOQ, `iiv_on_ruv`, a
    // custom sigma magnitude, LTBS, correlated residual) fell back to the fixed-b FD score.
    // It now shares `sens_outer_gradient::score_core`, so the chain is the SAME one
    // FOCE/FOCEI uses and the per-family gates are gone.
    //
    // The reference below is a central difference of `Stack::nll_at` at a **fixed, non-mode
    // b**. That matters: it differences the exact conditional NLL the AGQ objective
    // integrates, with NO inner re-solve, so -- unlike the outer-gradient tests -- the
    // reference carries no inner-solver noise floor and the step can be chosen for
    // truncation alone.

    const M3_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    const RUV_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
"#;

    const LTBS_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma ADD_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ log_additive(ADD_ERR)
"#;

    /// FREM: 3 PK etas + 2 covariate etas under one block Ω, with the covariate θs and the
    /// `EPSCOV` σ left **free** so the gradient test actually exercises those channels (the
    /// shipped `examples/warfarin_frem.ferx` fixes them, which would mask the very terms
    /// under test). `EPSCOV` is also given a sane magnitude rather than the near-zero value
    /// a production FREM model uses to pin `η_cov` to the observed covariate.
    const FREM_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TV_WT(72.0, 1.0, 500.0)
  theta TV_AGE(37.6, 1.0, 200.0)

  block_omega (ETA_CL, ETA_V, ETA_KA, ETA_WT_FREM, ETA_AGE_FREM) = [
    0.09,
    0.0, 0.04,
    0.0, 0.0, 0.30,
    0.0, 0.0, 0.0, 111.56,
    0.0, 0.0, 0.0, 0.0, 99.38
  ]

  sigma PROP_ERR ~ 0.02
  sigma EPSCOV   ~ 0.30
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  frem_predictions = TV_WT/ETA_WT_FREM:100, TV_AGE/ETA_AGE_FREM:200
  frem_sigma       = EPSCOV
"#;

    /// A subject with realistic nonzero residuals, built from the model at a reference eta.
    fn score_subject(model: &CompiledModel, theta: &[f64], times: &[f64]) -> Subject {
        use crate::types::DoseEvent;
        use std::collections::HashMap;
        let n = times.len();
        let mut subject = Subject {
            id: "1".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let eta_ref = [0.12, -0.08, 0.2];
        let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, &eta_ref);
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        subject
    }

    /// The analytic fixed-b score vs a central difference of `Stack::nll_at` -- at a `b`
    /// that is deliberately NOT the mode, since the score is a property of `nll(b; x)`
    /// alone and must hold everywhere, not only at the EBE.
    fn assert_fixed_b_score_matches_fd(
        model: &CompiledModel,
        subject: &Subject,
        template: &ModelParameters,
        b: &[f64],
        tol: f64,
        label: &str,
    ) {
        use crate::estimation::parameterization::{pack_params, packed_fixed_mask, unpack_params};

        let x = pack_params(template);
        let params = unpack_params(&x, template);
        let stack = Stack::new(model, &params, usize::from(model.n_kappa > 0));
        assert!(
            analytic_score_supported(model),
            "{label}: expected the ANALYTIC score path, got the FD fallback"
        );

        let mut analytic = vec![0.0f64; x.len()];
        accumulate_fixed_eta_packed_gradient(
            model,
            subject,
            &params,
            template,
            &stack,
            b,
            1.0,
            &mut analytic,
        )
        .unwrap_or_else(|| panic!("{label}: analytic score declined"));

        let fixed = packed_fixed_mask(template);
        let mut scratch = crate::pk::EventPkParams::default();
        let mut nll_at_x = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, template);
            let st = Stack::new(model, &p, stack.n_occ);
            st.nll_at(model, subject, &p, b, &mut scratch, None)
        };

        for k in 0..x.len() {
            if fixed[k] {
                continue;
            }
            let h = 1e-6 * (1.0 + x[k].abs());
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[k] += h;
            xm[k] -= h;
            let fd = (nll_at_x(&xp) - nll_at_x(&xm)) / (2.0 * h);
            let scale = fd.abs().max(analytic[k].abs()).max(1.0);
            assert!(
                (analytic[k] - fd).abs() / scale < tol,
                "{label}: coord {k}: analytic {} vs FD {} (rel {:.2e})",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / scale
            );
        }
    }

    /// M3-BLOQ. Was gated to the FD score; FOCE/FOCEI have had it analytic since #486.
    #[test]
    fn fixed_b_score_is_analytic_and_exact_under_m3() {
        use crate::types::BloqMethod;
        let mut model = parse_model_string(M3_MODEL).expect("parse");
        model.bloq_method = BloqMethod::M3;
        let theta = [0.22, 11.0, 1.4];
        let mut subject = score_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let n = subject.observations.len();
        subject.cens[n - 1] = 1;
        subject.cens[n - 2] = 1;

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        assert_fixed_b_score_matches_fd(
            &model,
            &subject,
            &template,
            &[0.05, -0.03, 0.08],
            1e-6,
            "M3",
        );
    }

    /// `iiv_on_ruv` -- eta enters the residual VARIANCE directly (`df/deta_ruv = 0`), a
    /// channel the old Gaussian chain had no term for at all.
    #[test]
    fn fixed_b_score_is_analytic_and_exact_under_iiv_on_ruv() {
        let model = parse_model_string(RUV_MODEL).expect("parse");
        assert!(
            model.residual_error_eta.is_some(),
            "fixture must set iiv_on_ruv"
        );
        let theta = [0.22, 11.0, 1.4];
        let subject = score_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        assert_fixed_b_score_matches_fd(
            &model,
            &subject,
            &template,
            &[0.05, -0.03, 0.08, 0.06],
            1e-6,
            "iiv_on_ruv",
        );
    }

    /// LTBS. The provider already returns the log-transformed jet
    /// (`apply_ltbs_transform_outer`), so the shared chain needs no LTBS special case --
    /// this pins that.
    #[test]
    fn fixed_b_score_is_analytic_and_exact_under_ltbs() {
        let model = parse_model_string(LTBS_MODEL).expect("parse");
        assert!(model.log_transform, "log_additive must set LTBS");
        let theta = [0.22, 11.0, 1.4];
        let subject = score_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        assert_fixed_b_score_matches_fd(
            &model,
            &subject,
            &template,
            &[0.05, -0.03, 0.08],
            1e-6,
            "LTBS",
        );
    }

    /// FREM covariate pseudo-observations. REGRESSION TEST — this fails without the jet +
    /// variance fix, and it fails for FOCE/FOCEI too, not just AGQ.
    ///
    /// `individual_nll` scores a `fremtype > 0` row against a completely different model:
    /// prediction `θ[ti] + η[ei]` (not the PK model's `f`) and variance `EPSCOV²` (not
    /// `error_spec.variance_at(f)`). The sensitivity provider returned the ordinary PK jet
    /// for those rows and `prepare_stacked` read the ordinary variance, so the analytic
    /// outer gradient was differentiating a likelihood the objective never evaluates.
    /// Latent because FREM is conventionally fit with SAEM, which never asks for it.
    ///
    /// The FD reference here is a central difference of the real conditional NLL at a fixed
    /// `b`, so it exercises exactly the likelihood the objective uses — including the
    /// pseudo-observation rows.
    #[test]
    fn frem_pseudo_obs_rows_get_the_right_analytic_score() {
        use crate::types::DoseEvent;
        use std::collections::HashMap;

        let model = parse_model_string(FREM_MODEL).expect("parse");
        let fc = model.frem_config.as_ref().expect("fixture must be FREM");
        assert_eq!(model.n_eta, 5, "3 PK etas + 2 covariate etas");
        assert!(
            analytic_score_supported_model(&model),
            "FREM must now take the ANALYTIC score path"
        );

        // Two PK rows plus one pseudo-observation per covariate (FREMTYPE 100 / 200).
        let mut fts: Vec<u16> = fc.fremtype_to_indices.keys().copied().collect();
        fts.sort_unstable();
        assert_eq!(fts.len(), 2, "fixture has WT and AGE pseudo-obs");

        let theta = &model.default_params.theta;
        let pk_times = [1.0, 6.0, 24.0];
        let n = pk_times.len() + fts.len();
        let mut subject = Subject {
            id: "1".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: pk_times
                .iter()
                .copied()
                .chain(fts.iter().map(|_| 0.0))
                .collect(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: pk_times
                .iter()
                .map(|_| 0u16)
                .chain(fts.iter().copied())
                .collect(),
            obs_records: vec![],
        };
        // PK rows: realistic residuals off the model. Pseudo-obs rows: the covariate value,
        // offset from θ so the residual is nonzero.
        let eta_ref = vec![0.1, -0.05, 0.15, 0.0, 0.0];
        let preds = crate::pk::compute_predictions_with_tv(&model, &subject, theta, &eta_ref);
        for j in 0..pk_times.len() {
            subject.observations[j] = preds[j] * 0.85;
        }
        for (k, &ft) in fts.iter().enumerate() {
            let (ti, _ei) = fc.fremtype_to_indices[&ft];
            subject.observations[pk_times.len() + k] = theta[ti] * 1.10;
        }

        let template = model.default_params.clone();
        let b = [0.05, -0.03, 0.08, 1.5, -2.0];
        assert_fixed_b_score_matches_fd(&model, &subject, &template, &b, 1e-6, "FREM");

        // The INNER η-gradient has the same root cause and is the more damaging half: it
        // sets the EBEs. `analytic_eta_nll_gradient` fed `residual_inner_obs` the PK
        // prediction and the PK variance on pseudo-observation rows, so the mode it drove
        // to was not the mode of the likelihood being integrated.
        let params = model.default_params.clone();
        let analytic = crate::estimation::inner_optimizer::analytic_eta_nll_gradient(
            &model,
            &subject,
            &params.theta,
            &b,
            &params.omega,
            &params.sigma.values,
        )
        .expect("FREM inner gradient must be analytic");

        let mut scratch = crate::pk::EventPkParams::default();
        for k in 0..b.len() {
            let h = 1e-6 * (1.0 + b[k].abs());
            let (mut bp, mut bm) = (b.to_vec(), b.to_vec());
            bp[k] += h;
            bm[k] -= h;
            let nll = |e: &[f64], s: &mut crate::pk::EventPkParams| {
                crate::stats::likelihood::individual_nll_into_with_schedule(
                    &model,
                    &subject,
                    &params.theta,
                    e,
                    &params.omega,
                    &params.sigma.values,
                    s,
                    None,
                )
            };
            let fd = (nll(&bp, &mut scratch) - nll(&bm, &mut scratch)) / (2.0 * h);
            let scale = fd.abs().max(analytic[k].abs()).max(1.0);
            assert!(
                (analytic[k] - fd).abs() / scale < 1e-6,
                "FREM inner grad η{k}: analytic {} vs FD {}",
                analytic[k],
                fd
            );
        }
    }

    /// The gate itself: none of these families may route to the FD score any more.
    #[test]
    fn analytic_score_no_longer_gates_the_focei_scope() {
        use crate::types::BloqMethod;
        let mut m3 = parse_model_string(M3_MODEL).expect("parse");
        m3.bloq_method = BloqMethod::M3;
        for (model, label) in [
            (m3, "M3"),
            (parse_model_string(RUV_MODEL).expect("parse"), "iiv_on_ruv"),
            (parse_model_string(LTBS_MODEL).expect("parse"), "LTBS"),
        ] {
            assert!(
                analytic_score_supported_model(&model),
                "{label} must take the analytic score path"
            );
        }
    }

    /// [`grid_z_at`] must reproduce **exactly** the node ordering [`agq_nodes_and_terms`]
    /// sweeps.
    ///
    /// The grid-response term contracts node `j`'s displacement against node `j`'s stored
    /// `∂nll/∂b` and node `j`'s softmax weight. If the two enumerations ever drift, every node
    /// would be paired with *another* node's gradient — each individually valid, so the result
    /// stays finite and plausible while being a wrong gradient. That is exactly the failure
    /// class this module's "share one sweep" rule exists to prevent, and reconstructing `zⱼ`
    /// from the index (rather than storing it) is the one place the rule is enforced by
    /// agreement rather than by construction. Hence this test.
    #[test]
    fn grid_z_at_matches_the_sweep_enumeration() {
        for (n, d) in [(1usize, 1usize), (3, 1), (2, 3), (3, 4), (5, 2), (4, 3)] {
            let (nodes, _w) = gauss_hermite(n);
            // The mixed-radix counter from `agq_nodes_and_terms`, digit 0 fastest.
            let mut idx = vec![0usize; d];
            let mut j = 0usize;
            let mut got = vec![0.0f64; d];
            loop {
                let expect: Vec<f64> = (0..d).map(|k| nodes[idx[k]]).collect();
                grid_z_at(j, &nodes, d, &mut got);
                assert_eq!(got, expect, "n={n} d={d}: node {j} z-vector");
                j += 1;
                let mut k = 0;
                while k < d {
                    idx[k] += 1;
                    if idx[k] < n {
                        break;
                    }
                    idx[k] = 0;
                    k += 1;
                }
                if k == d {
                    break;
                }
            }
            assert_eq!(j, grid_size(n, d), "n={n} d={d}: swept node count");
        }
    }

    /// Golub–Welsch must reproduce the textbook physicists' Hermite rule.
    #[test]
    fn gauss_hermite_matches_known_rules() {
        // n = 1: the rule that makes AGQ collapse to Laplace.
        let (x, w) = gauss_hermite(1);
        assert!((x[0] - 0.0).abs() < 1e-14);
        assert!((w[0] - PI.sqrt()).abs() < 1e-14);

        // n = 3: nodes ±√(3/2), 0; weights √π/6, 2√π/3, √π/6.
        let (x, w) = gauss_hermite(3);
        let r = (1.5f64).sqrt();
        for (got, want) in x.iter().zip([-r, 0.0, r]) {
            assert!((got - want).abs() < 1e-12, "node {got} != {want}");
        }
        let sp = PI.sqrt();
        for (got, want) in w.iter().zip([sp / 6.0, 2.0 * sp / 3.0, sp / 6.0]) {
            assert!((got - want).abs() < 1e-12, "weight {got} != {want}");
        }
    }

    /// Weights sum to `√π = ∫ e^{−x²} dx`, and the rule is exact for polynomials up to
    /// degree `2n − 1` — the defining property of an n-point Gaussian rule.
    #[test]
    fn gauss_hermite_integrates_polynomials_exactly() {
        for n in 1..=MAX_AGQ_NODES {
            let (x, w) = gauss_hermite(n);
            let total: f64 = w.iter().sum();
            assert!(
                (total - PI.sqrt()).abs() < 1e-10,
                "n={n}: weights sum to {total}, want √π"
            );
            // ∫ x² e^{−x²} dx = √π/2 — needs degree 2, exact for every n ≥ 2.
            if n >= 2 {
                let m2: f64 = x.iter().zip(w.iter()).map(|(xi, wi)| wi * xi * xi).sum();
                assert!(
                    (m2 - PI.sqrt() / 2.0).abs() < 1e-10,
                    "n={n}: ∫x²e^{{−x²}} = {m2}, want √π/2"
                );
            }
        }
    }

    /// The identity the whole method rests on: with one node the AGQ formula *is* the
    /// Laplace approximation. Checked here on the raw arithmetic — a 1-D integrand whose
    /// exact Laplace value is known in closed form — so a regression in the constants
    /// (the `(d/2)·log π`, the `√2` scaling, the `exp(‖z‖²)` correction) is caught by a
    /// unit test rather than by a converging fit.
    #[test]
    fn one_node_agq_equals_laplace() {
        // Integrand exp(−nll) with nll(η) = ½·a·(η − m)² + c, i.e. an exact Gaussian.
        // Laplace is exact here: −log ∫ = c + ½·log(a) − ½·log(2π), and in the
        // "without constant" convention the per-subject NLL is nll(m) + ½·log|H| = c + ½·log a.
        let (a, m, c) = (2.5f64, 0.3f64, 1.7f64);
        let nll = |eta: f64| 0.5 * a * (eta - m).powi(2) + c;

        let (nodes, weights) = gauss_hermite(1);
        let log_w: Vec<f64> = weights.iter().map(|w| w.ln()).collect();

        // H = a (exact second derivative); Σ^{1/2} = a^{−½}.
        let h = a;
        let terms: Vec<f64> = (0..1)
            .map(|j| {
                let z = nodes[j];
                let eta = m + std::f64::consts::SQRT_2 * h.powf(-0.5) * z;
                log_w[j] + z * z - nll(eta)
            })
            .collect();
        let agq = 0.5 * PI.ln() + 0.5 * h.ln() - logsumexp(&terms);
        let laplace = nll(m) + 0.5 * h.ln();
        assert!(
            (agq - laplace).abs() < 1e-12,
            "1-node AGQ {agq} != Laplace {laplace}"
        );
    }

    /// With enough nodes AGQ must recover the *exact* marginal on an integrand where the
    /// truth is known — here a Gaussian, where Laplace is already exact, plus a genuinely
    /// non-Gaussian (skewed) integrand where it is not. The second case is the one that
    /// proves the extra nodes are doing real work.
    #[test]
    fn many_nodes_recover_exact_marginal_on_skewed_integrand() {
        // nll(η) = ½η² + η⁴/12  ⇒  ∫exp(−nll) has no closed form; get truth by fine
        // trapezoid on a wide grid (the integrand decays like e^{−η⁴/12}).
        let nll = |e: f64| 0.5 * e * e + e.powi(4) / 12.0;
        let (lo, hi, steps) = (-12.0f64, 12.0f64, 2_000_000);
        let dx = (hi - lo) / steps as f64;
        let truth: f64 = (0..=steps)
            .map(|i| {
                let e = lo + i as f64 * dx;
                let f = (-nll(e)).exp();
                if i == 0 || i == steps {
                    0.5 * f
                } else {
                    f
                }
            })
            .sum::<f64>()
            * dx;
        // Reference in the module's convention: (d/2)·log(2π) − log G, with d = 1.
        let want = 0.5 * (2.0 * PI).ln() - truth.ln();

        // Mode is η = 0; H = nll''(0) = 1.
        let (m, h) = (0.0f64, 1.0f64);
        let agq_with = |n: usize| -> f64 {
            let (nodes, weights) = gauss_hermite(n);
            let terms: Vec<f64> = (0..n)
                .map(|j| {
                    let z = nodes[j];
                    let eta = m + std::f64::consts::SQRT_2 * h.powf(-0.5) * z;
                    weights[j].ln() + z * z - nll(eta)
                })
                .collect();
            0.5 * PI.ln() + 0.5 * h.ln() - logsumexp(&terms)
        };

        // Laplace is badly biased on this integrand — it is *exactly* 0 here (the mode
        // term and the ½log|H| term cancel), against a true marginal NLL of ~0.1368.
        let laplace = agq_with(1);
        assert!(
            (laplace - want).abs() > 1e-3,
            "Laplace {laplace} unexpectedly matched truth {want}; the test integrand is not \
             non-Gaussian enough to prove the node sweep does anything"
        );

        // Adding nodes closes that gap monotonically. The 21-node residual (~3e-6) is
        // genuine Gauss–Hermite truncation, not a defect: the transformed integrand here is
        // exp(−z⁴/3), which no finite polynomial rule integrates exactly. A regression in
        // the transform or the constants would miss by O(0.1), not O(1e-6).
        let errs: Vec<f64> = [1usize, 3, 7, 21]
            .iter()
            .map(|&n| (agq_with(n) - want).abs())
            .collect();
        for w in errs.windows(2) {
            assert!(
                w[1] < w[0],
                "AGQ error must shrink with node count, got {errs:?}"
            );
        }
        assert!(
            *errs.last().unwrap() < 1e-5,
            "21-node AGQ error {} too large (truth {want})",
            errs.last().unwrap()
        );
    }

    #[test]
    fn grid_size_saturates_instead_of_overflowing() {
        assert_eq!(grid_size(3, 4), 81);
        assert_eq!(grid_size(1, 50), 1);
        // 21^50 overflows u64/usize many times over; must saturate, not wrap to something
        // small that would sneak past the MAX_AGQ_GRID check.
        assert_eq!(grid_size(21, 50), usize::MAX);
    }
}
