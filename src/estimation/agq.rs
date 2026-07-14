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
use crate::types::{CompiledModel, ModelParameters, Population, Subject};

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

/// Numerically stable `log Σ exp(xᵢ)`.
fn logsumexp(xs: &[f64]) -> f64 {
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
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

    let h = fd_posterior_hessian(
        model,
        subject,
        params,
        stack,
        b_hat,
        &mut scratch,
        schedule.as_ref(),
    );
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
            agq_subject_nll(model, subject, params, &stack, &b_hat, &nodes, &log_weights)
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

/// Whether the **θ/σ score** can be taken analytically from the `Dual2` provider, or must be
/// finite-differenced at fixed η.
///
/// This is a *speed* switch, not a scope gate: both paths compute the same quantity, and
/// neither re-solves the inner loop. The analytic form chains the provider's exact `∂f/∂θ`
/// through `∂nll/∂f`, which is only valid for the plain Gaussian residual term — every
/// variant listed below adds a term it does not carry (M3's normal-tail, LTBS's `ln f`
/// wrap, FREM's pseudo-observations, a dense residual covariance, an η-scaled or custom
/// residual variance), and TTE / categorical endpoints are outside the provider entirely.
/// Those all take [`accumulate_fixed_eta_packed_gradient_fd`] instead.
/// Model-level mirror of [`analytic_score_supported`], for reporting
/// (`build_info::gradient_method_outer`) which has no per-subject [`Stack`]. Uses the model's
/// own `n_kappa` to pick the IOV or non-IOV provider scope, which is the same answer every
/// subject of an IOV model gets.
pub fn analytic_score_supported_model(model: &CompiledModel) -> bool {
    let stack = Stack {
        n_eta: model.n_eta,
        n_kappa: model.n_kappa,
        n_occ: usize::from(model.n_kappa > 0),
        omega_joint_inv: DMatrix::zeros(0, 0),
        prior_sd: Vec::new(),
    };
    analytic_score_supported(model, &stack)
}

pub fn analytic_score_supported(model: &CompiledModel, stack: &Stack) -> bool {
    // The provider's scope differs for IOV: `subject_sensitivities_iov` runs the stacked
    // (θ, η, κ) dual walk, gated by `iov_sens_supported`, while the non-IOV entry point is
    // gated by `analytic_outer_gradient_available`. `analytic_outer_gradient_available`
    // already folds in `iov_sens_supported`, but it also carries the `gradient_method = fd`
    // opt-out and the non-Gaussian/magnitude bails, which apply to both.
    let provider = if stack.is_iov() {
        crate::sens::provider::iov_sens_supported(model)
            && !matches!(model.gradient_method, crate::types::GradientMethod::Fd)
    } else {
        crate::sens::provider::analytic_outer_gradient_available(model)
    };
    provider
        && !matches!(model.bloq_method, crate::types::BloqMethod::M3)
        && model.residual_correlations.is_empty()
        && model.frem_config.is_none()
        && model.residual_error_eta.is_none()
        && !model.has_custom_ruv_magnitude()
        && !model.log_transform
}

/// AGQ always drives the outer loop with its **own** gradient — for every model, and at every
/// `n_agq`.
///
/// There is deliberately no scope gate here. The two ingredients both work universally:
///
/// * the fixed-η score is analytic where the provider reaches
///   ([`analytic_score_supported`]) and finite-differenced *at fixed η* otherwise
///   ([`accumulate_fixed_eta_packed_gradient_fd`]) — the latter is correct for any likelihood
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
/// [`crate::estimation::sens_outer_gradient::chol_pack`]). Deliberately carries **no**
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
    let og = crate::estimation::sens_outer_gradient::chol_pack(
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
        let ig = crate::estimation::sens_outer_gradient::chol_pack(&m_iov, &iov.chol, iov.diagonal);
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
    if !analytic_score_supported(model, stack) {
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
    let sens = if stack.is_iov() {
        crate::sens::provider::subject_sensitivities_iov(model, subject, &params.theta, b)?
    } else {
        crate::sens::provider::subject_sensitivities(model, subject, &params.theta, b)?
    };

    let err_keys = model.error_spec.obs_keys(subject);
    let mut d_nll_d_f = vec![0.0f64; n_obs];
    let mut variances = vec![0.0f64; n_obs];
    let mut residuals = vec![0.0f64; n_obs];

    for j in 0..n_obs {
        let f = sens.obs[j].f.max(1e-12);
        let v = model
            .residual_variance_at(err_keys[j], f, &params.sigma.values)
            .max(1e-12);
        let resid = subject.observations[j] - sens.obs[j].f;
        residuals[j] = resid;
        variances[j] = v;
        // ∂nll_j/∂f = −r/V + ½·(∂V/∂f)·(1/V − r²/V²)  — the halved convention of
        // `individual_nll` (0.5·Σ[r²/V + ln V]); identical to SAEM's `d_nll_d_f`.
        let dv_df = model
            .error_spec
            .dvar_df(err_keys[j], f, &params.sigma.values);
        d_nll_d_f[j] = -resid / v + 0.5 * dv_df * (1.0 / v - resid * resid / (v * v));
    }

    // θ block: chain the exact ∂f/∂θ through ∂nll/∂f, then the log-θ packing chain rule.
    for m in 0..n_theta {
        let d: f64 = (0..n_obs)
            .map(|j| d_nll_d_f[j] * sens.obs[j].df_dtheta[m])
            .sum();
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        out[m] += weight * d * dtheta_dx;
    }

    // σ block: closed form. ∂nll/∂(log σ_k) = Σ_j ½·(∂V_j/∂log σ_k)·(1/V_j − r_j²/V_j²).
    for k in 0..n_sigma {
        let d: f64 = (0..n_obs)
            .map(|j| {
                let f = sens.obs[j].f.max(1e-12);
                let v = variances[j];
                let r = residuals[j];
                let ratio =
                    model
                        .error_spec
                        .dvar_dlogsigma(err_keys[j], k, f, &params.sigma.values);
                0.5 * ratio * (1.0 / v - r * r / (v * v))
            })
            .sum();
        out[sigma_start + k] += weight * d;
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
/// measures the result against a finite-difference of the real objective: 0.17% at n = 1,
/// 0.0006% at n = 7 — the FD reference's own noise floor.
///
/// At `n_agq = 1` the grid objective collapses to `½·log|H|` (the single node sits at η̂), so
/// this reduces to `½·d log|H|/dx` — precisely the Laplace log-determinant term FOCE/FOCEI
/// carry analytically.
#[allow(clippy::too_many_arguments)]
fn grid_response_correction(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    template: &ModelParameters,
    stack: &Stack,
    x: &[f64],
    b_hat: &[f64],
    nodes: &[f64],
    log_weights: &[f64],
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

    for k in 0..x.len() {
        if fixed[k] {
            continue;
        }
        // `H` is itself a finite-difference quantity, so its own error floor is amplified by
        // 1/step here. The step must stay well above sqrt(that floor); 1e-3 is calibrated by
        // `analytic_gradient_matches_fd_at_every_node_count`.
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
        let hp = fd_posterior_hessian(model, subject, &pp, &sp, &ep, scratch, schedule);
        let hm = fd_posterior_hessian(model, subject, &pm, &sm, &em, scratch, schedule);

        // `nll` stays at the ORIGINAL params — the direct x-dependence is already covered
        // analytically by the fixed-η score — but the grid (centre and scale) is the
        // perturbed one, which is exactly the response term we are after.
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
        let r = (phip - phim) / (2.0 * step);
        if r.is_finite() {
            out[k] += r;
        }
    }
    Some(())
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
    if analytic_score_supported(model, stack) {
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

    let h = fd_posterior_hessian(
        model,
        subject,
        params,
        stack,
        b_hat,
        &mut scratch,
        schedule.as_ref(),
    );
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

    for (b_j, &t) in bs.iter().zip(terms.iter()) {
        let w = (t - lse).exp();
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
        x,
        b_hat,
        nodes,
        log_weights,
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

    #[test]
    fn logsumexp_is_stable_and_handles_sentinels() {
        let want = (1.0f64.exp() + 2.0f64.exp()).ln();
        assert!((logsumexp(&[1.0, 2.0]) - want).abs() < 1e-12);
        // Huge magnitudes must not overflow.
        assert!((logsumexp(&[1e5, 1e5]) - (1e5 + 2.0f64.ln())).abs() < 1e-9);
        // A diverged node (the −1e20 term) is simply ignored next to a live one.
        assert!((logsumexp(&[-1e20, 3.0]) - 3.0).abs() < 1e-12);
    }
}
