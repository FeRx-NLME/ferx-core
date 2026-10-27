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
//!    and `tests::one_node_agq_equals_laplace` pins it.
//! 2. **No Gaussian-residual assumption.** `l_i` is evaluated through
//!    [`crate::stats::likelihood::individual_nll_into_with_schedule`], the model's *actual*
//!    likelihood — so
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
//! `crate::estimation::importance_sampling::compute_posterior_hessian`, which builds the
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

use crate::stats::stacked_layout::{kappa_range, stacked_width};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::estimation::importance_sampling::build_proposal;
use crate::estimation::inner_optimizer::cacheable_schedule;
use crate::pk;
use crate::stats::likelihood::{individual_nll_into_prepared_with_schedule, SUBJECT_SENTINEL_NLL};
use crate::stats::util::log_sum_exp as logsumexp;
use crate::types::{CompiledModel, HessianAnchor, ModelParameters, Population, Subject};

/// Upper bound on `n_agq`. Beyond ~20 nodes the Golub–Welsch eigenproblem starts to lose
/// the extreme nodes to round-off, and the marginal accuracy gain is nil for any realistic
/// NLME integrand.
pub const MAX_AGQ_NODES: usize = 21;

static GAUSS_HERMITE_RULES: [OnceLock<(Vec<f64>, Vec<f64>, Vec<f64>)>; MAX_AGQ_NODES] =
    [const { OnceLock::new() }; MAX_AGQ_NODES];

pub(crate) fn cached_gauss_hermite(n: usize) -> (&'static [f64], &'static [f64], &'static [f64]) {
    assert!((1..=MAX_AGQ_NODES).contains(&n));
    let (nodes, weights, log_weights) = GAUSS_HERMITE_RULES[n - 1].get_or_init(|| {
        let (nodes, weights) = gauss_hermite(n);
        let log_weights = weights.iter().map(|w| w.ln()).collect();
        (nodes, weights, log_weights)
    });
    (nodes, weights, log_weights)
}

fn cached_log_weights_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FERX_NO_AGQ_LOG_WEIGHT_CACHE")
            .map(|v| v != "1")
            .unwrap_or(true)
    })
}

fn laplace_one_point_fast_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FERX_NO_LAPLACE_ONE_POINT_FAST")
            .map(|v| v != "1")
            .unwrap_or(true)
    })
}

fn laplace_mode_nll_reuse_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FERX_NO_LAPLACE_MODE_NLL_REUSE")
            .map(|v| v != "1")
            .unwrap_or(true)
    })
}

/// Hard cap on the tensor-grid size `n_agq^n_eta`, enforced at model-check time by
/// [`crate::api::check_model_options`]. The tensor rule costs one full likelihood
/// evaluation per node per subject per outer iteration, so the grid — not the node count —
/// is the quantity a user needs protecting from. 100k nodes on an ODE model is already
/// hours; past that a fit is not slow, it is wrong to have started.
pub const MAX_AGQ_GRID: usize = 100_000;

/// Work already produced by the immediately preceding inner solve at the same
/// population parameters and mode: optional exact curvature plus the mode NLL.
pub(crate) type TerminalLaplaceWork = (Option<DMatrix<f64>>, f64);

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

// Test-only A/B switch for measuring the former per-node allocation without
// carrying a production branch. Numeric work remains identical.
#[cfg(test)]
static LEGACY_KAPPA_SLICE_ALLOCATION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// AGQ population objective and matching gradient computed from the same subject-local grids.
pub(crate) struct PopulationEvaluation {
    pub(crate) nll: f64,
    gradient: Option<Vec<f64>>,
    /// How each subject's score was formed, in subject order (for the outer FD-fallback
    /// warning).
    routes: Vec<ScoreRoute>,
}

/// How [`SubjectScoreContext::score_routed`] formed a subject's score.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ScoreRoute {
    /// The adjoint's exact `dFᵢ/dx`.
    Exact,
    /// The adjoint's held-EBE score (no EBE response):
    /// `adjoint::subject_adjoint_held_ebe_score`.
    HeldEbe,
    /// The reconverged finite-difference score.
    Reconverged,
}

impl ScoreRoute {
    /// Record a non-exact route for subject `i` in the outer FD-fallback log.
    fn record(self, log: &crate::estimation::outer_optimizer::OuterFdDeclineLog, i: usize) {
        match self {
            ScoreRoute::Exact => {}
            ScoreRoute::HeldEbe => log.record(i),
            ScoreRoute::Reconverged => {
                log.record(i);
                log.record_structural(i);
            }
        }
    }
}

impl Stack {
    /// Build the stack for a subject with `n_occ` occasions (`0` ⇒ no IOV).
    fn new(model: &CompiledModel, params: &ModelParameters, n_occ: usize) -> Self {
        let n_eta = model.n_eta;
        let n_kappa = if n_occ == 0 { 0 } else { model.n_kappa };
        let d = stacked_width(n_eta, n_kappa, n_occ);

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
        stacked_width(self.n_eta, self.n_kappa, self.n_occ)
    }

    fn is_iov(&self) -> bool {
        self.n_occ > 0 && self.n_kappa > 0
    }

    fn split<'a>(&self, b: &'a [f64]) -> (&'a [f64], Vec<&'a [f64]>) {
        let eta = &b[..self.n_eta];
        let kappas = (0..self.n_occ)
            .map(|k| &b[kappa_range(self.n_eta, self.n_kappa, k)])
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
        // Low-call-count sites (FD sweeps, single-node scoring): hoist nothing, just
        // allocate scratch for this one call and dispatch through the shared branch.
        let err_keys = (!self.is_iov()).then(|| model.error_spec.obs_keys(subject));
        let ruv_mult = (!self.is_iov())
            .then(|| model.ruv_obs_mult(subject, &params.theta))
            .flatten();
        self.nll_at_with_recycle(
            model,
            subject,
            params,
            b,
            scratch,
            schedule,
            err_keys.as_deref(),
            ruv_mult.as_deref(),
            &mut Vec::new(),
            &mut DVector::zeros(self.n_eta),
            &mut DVector::zeros(self.n_eta),
        )
    }

    /// Same dispatch as [`Stack::nll_at`], but takes the `err_keys`/`ruv_mult`/scratch as
    /// caller-hoisted arguments instead of computing or allocating them per call. The hot
    /// per-node sweep in [`agq_terms`] hoists these once for the whole grid; routing both
    /// call shapes through this one
    /// IOV/non-IOV branch keeps a single copy of the dispatch instead of two that must be
    /// kept in sync.
    #[allow(clippy::too_many_arguments)]
    fn nll_at_with_recycle(
        &self,
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
        b: &[f64],
        scratch: &mut pk::EventPkParams,
        schedule: Option<&pk::event_driven::EventSchedule>,
        err_keys: Option<&[usize]>,
        ruv_mult: Option<&[Vec<f64>]>,
        pred_recycle: &mut Vec<f64>,
        eta_work: &mut DVector<f64>,
        prior_work: &mut DVector<f64>,
    ) -> f64 {
        if !self.is_iov() {
            return individual_nll_into_prepared_with_schedule(
                model,
                subject,
                &params.theta,
                b,
                &params.omega,
                &params.sigma.values,
                // AGQ holds the `block_sigma` off-diagonals at their declared
                // value (#847); `params` carries exactly that.
                model.residual_cov_at(params),
                scratch,
                schedule,
                err_keys.expect("non-IOV error keys"),
                ruv_mult,
                pred_recycle,
                eta_work,
                prior_work,
            );
        }
        let eta = &b[..self.n_eta];
        const INLINE_OCCASIONS: usize = 16;
        #[cfg(test)]
        let force_legacy = LEGACY_KAPPA_SLICE_ALLOCATION.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(test))]
        let force_legacy = false;
        if self.n_occ <= INLINE_OCCASIONS && !force_legacy {
            let kappas: [&[f64]; INLINE_OCCASIONS] = std::array::from_fn(|k| {
                if k < self.n_occ {
                    &b[kappa_range(self.n_eta, self.n_kappa, k)]
                } else {
                    &[]
                }
            });
            crate::stats::likelihood::individual_nll_iov_with_scratch(
                model,
                subject,
                &params.theta,
                eta,
                &kappas[..self.n_occ],
                &params.omega,
                params.omega_iov.as_ref(),
                &params.sigma.values,
                scratch,
            )
        } else {
            let (_, kappas) = self.split(b);
            crate::stats::likelihood::individual_nll_iov_with_scratch(
                model,
                subject,
                &params.theta,
                eta,
                &kappas,
                &params.omega,
                params.omega_iov.as_ref(),
                &params.sigma.values,
                scratch,
            )
        }
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

/// The Hessian that scales the quadrature grid and enters `½log|H|`, per the anchor.
///
/// * [`HessianAnchor::Exact`] → the exact conditional Hessian `∂²nll/∂b²`. This is
///   Laplace / adaptive GH quadrature. Taken **analytically** from the unified recording
///   (`adjoint::exact_inner_hessian`) wherever it serves the subject, and from the
///   `2d²+1`-evaluation [`fd_posterior_hessian`] sweep only outside it (TTE, categorical),
///   whose likelihood has no `PkNum` twin.
/// * [`HessianAnchor::GaussNewton`] → the Almquist `H̃`, the *same* Hessian FOCEI's own
///   objective and gradient use, so `focei` with `n_agq > 1` is a genuine quadrature
///   refinement of FOCEI.
///
/// **The two anchors stay different matrices — that is the estimator distinction.** Sharing
/// one recorder (the adjoint's `run_agq`, `Objective::Agq { exact }`) unifies the
/// *computation*, not the result: `laplace, n_agq = 1` is NONMEM
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
            // The exact Hessian from the unified recording (every engine it serves), before
            // any finite difference.
            if let Some(h) = crate::sens::ode_provider::adjoint::exact_inner_hessian(
                model, subject, params, b_hat,
            ) {
                return Some(h);
            }
            Some(fd_posterior_hessian(
                model, subject, params, stack, b_hat, scratch, schedule,
            ))
        }
        HessianAnchor::GaussNewton => {
            crate::sens::ode_provider::adjoint::exact_htilde(model, subject, params, b_hat)
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn agq_subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    stack: &Stack,
    b_hat: &[f64],
    nodes: &[f64],
    log_weights: &[f64],
    anchor: HessianAnchor,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> f64 {
    agq_subject_evaluate(
        model,
        subject,
        params,
        stack,
        b_hat,
        nodes,
        log_weights,
        anchor,
        schedule,
        None,
        None,
    )
}

/// Same as the body [`agq_subject_nll`] delegates to, but `schedule` is now a caller-supplied
/// parameter instead of a fresh [`cacheable_schedule`] rebuild on every call — see
/// [`crate::estimation::inner_optimizer::build_schedule_cache`]. It depends only on
/// `model` + `subject`, so the two population-level callers ([`agq_population_nll`],
/// [`agq_population_evaluate`]) build the cache once per outer-loop evaluation and hand each
/// subject its entry, rather than every subject re-walking its own dose/event timeline.
/// If no schedule is supplied, the caller continues using the legacy per-call
/// [`cacheable_schedule`] path.
#[allow(clippy::too_many_arguments)]
fn agq_subject_evaluate(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    stack: &Stack,
    b_hat: &[f64],
    nodes: &[f64],
    log_weights: &[f64],
    anchor: HessianAnchor,
    schedule: Option<&pk::event_driven::EventSchedule>,
    terminal_hessian: Option<&DMatrix<f64>>,
    terminal_mode_nll: Option<f64>,
) -> f64 {
    let d = stack.d();
    let mut scratch = pk::EventPkParams::with_capacity_for(subject);
    let local_schedule = if schedule.is_none() {
        cacheable_schedule(model, subject)
    } else {
        None
    };
    let schedule = schedule.or(local_schedule.as_ref());

    // No random effects: the "integral" is a point mass and AGQ degenerates to the
    // conditional likelihood itself. (The formula below would agree — an empty tensor
    // product is the single empty node — but there is no Σ to factor, so short-circuit.)
    if d == 0 {
        return stack.nll_at(model, subject, params, b_hat, &mut scratch, schedule);
    }

    let h = matches!(anchor, HessianAnchor::Exact)
        .then(|| terminal_hessian.cloned())
        .flatten()
        .or_else(|| {
            anchor_hessian(
                anchor,
                model,
                subject,
                params,
                stack,
                b_hat,
                &mut scratch,
                schedule,
            )
        });
    let Some(h) = h else {
        // Gauss-Newton anchor out of the recorder's scope (guarded up front by
        // `check_model_options`, so this is only reachable if that guard is bypassed).
        return SUBJECT_SENTINEL_NLL;
    };
    // `build_proposal` applies the relative jitter and — if the FD Hessian came back
    // indefinite (a loosely-converged mode, a flat direction) — falls back to the prior-scale
    // factor (`Ω`, or the block-diagonal `Ω_joint` under IOV). That fallback keeps AGQ
    // *consistent* (any invertible scale integrates to the same limit); it only costs
    // nodes-worth of efficiency, which is the right failure mode.
    let Some(proposal) = build_proposal(&h, &stack.omega_joint_inv, d) else {
        return SUBJECT_SENTINEL_NLL;
    };

    // One-point AGQ is Laplace exactly.  Avoid constructing the one-element tensor
    // grid and, when the inner solve handed it to us, avoid re-evaluating the
    // conditional NLL at the very same mode. At z=0 the transformed node is
    // b_hat, while the sqrt(PI) rule weight cancels the d/2*log(PI)
    // normalization term algebraically.
    if nodes.len() == 1 && laplace_one_point_fast_enabled() {
        let reusable_nll = terminal_mode_nll.filter(|_| laplace_mode_nll_reuse_enabled());
        let mode_nll = reusable_nll.map_or_else(
            || stack.nll_at(model, subject, params, b_hat, &mut scratch, schedule),
            |nll| nll,
        );
        let nll = mode_nll + 0.5 * proposal.log_det_inv_scale;
        return if nll.is_finite() {
            nll
        } else {
            SUBJECT_SENTINEL_NLL
        };
    }

    let terms = agq_terms(
        model,
        subject,
        params,
        stack,
        b_hat,
        nodes,
        log_weights,
        &proposal,
        &mut scratch,
        schedule,
    );
    let nll = 0.5 * d as f64 * PI.ln() + 0.5 * proposal.log_det_inv_scale - logsumexp(&terms);
    if nll.is_finite() {
        nll
    } else {
        SUBJECT_SENTINEL_NLL
    }
}

/// Sweep the tensor grid once, returning every log-term `log w + ‖z‖² − l(b_j)` — the one
/// place the grid is materialised; the objective ([`agq_subject_nll`]) reduces the terms
/// with `logsumexp`.
#[allow(clippy::too_many_arguments)]
fn agq_terms(
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
) -> Vec<f64> {
    let d = stack.d();
    let n = nodes.len();
    let mut terms = Vec::with_capacity(grid_size(n, d));
    let mut idx = vec![0usize; d];
    let mut z = vec![0.0f64; d];
    let mut step = vec![0.0f64; d];
    let mut b = vec![0.0f64; d];
    let err_keys = (!stack.is_iov()).then(|| model.error_spec.obs_keys(subject));
    let ruv_mult = (!stack.is_iov())
        .then(|| model.ruv_obs_mult(subject, &params.theta))
        .flatten();
    let mut pred_recycle = Vec::new();
    let mut eta_work = DVector::zeros(stack.n_eta);
    let mut prior_work = DVector::zeros(stack.n_eta);
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
        for k in 0..d {
            b[k] = b_hat[k] + step[k];
        }
        // A diverged node returns `individual_nll`'s 1e20 sentinel, which lands here as a
        // ~−1e20 log-term — negligible under logsumexp unless *every* node diverged, in
        // which case the subject correctly reports the sentinel back.
        let nll = stack.nll_at_with_recycle(
            model,
            subject,
            params,
            &b,
            scratch,
            schedule,
            err_keys.as_deref(),
            ruv_mult.as_deref(),
            &mut pred_recycle,
            &mut eta_work,
            &mut prior_work,
        );
        terms.push(log_w + z_sq - nll);
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
            return terms;
        }
    }
}

/// [`agq_subject_nll`] under a chosen Hessian anchor (`Exact` is Laplace's).
#[cfg(test)]
pub(crate) fn agq_subject_objective_anchored(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta_hat: &[f64],
    n_agq: usize,
    anchor: HessianAnchor,
) -> f64 {
    let stack = Stack::new(model, params, 0);
    let (nodes, weights) = gauss_hermite(n_agq);
    let log_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
    agq_subject_nll(
        model,
        subject,
        params,
        &stack,
        eta_hat,
        &nodes,
        &log_weights,
        anchor,
        None,
    )
}

/// Population AGQ objective: `Σ_i agq_subject_nll_i`. The outer loop doubles this into an
/// OFV, exactly as it does `crate::estimation::outer_optimizer::pop_nll`.
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
    agq_population_nll_impl(
        model, population, params, eta_hats, kappas, n_nodes, anchor, None, None,
    )
}

/// Same objective as [`agq_population_nll`], for the FOCEI/Laplace/AGQ outer hot loop
/// (`outer_optimizer.rs`): a caller-hoisted
/// [`crate::estimation::inner_optimizer::build_schedule_cache`] skips every subject's
/// per-eval schedule rebuild, and `terminal_hessians` — each subject's exact conditional
/// Hessian retained from its inner solve (`EbeResult::terminal_hessian`) — replaces the
/// anchor recomputation under [`HessianAnchor::Exact`]. Either may be `None`; every other
/// caller goes through [`agq_population_nll`] and pays both.
///
/// A retained Hessian is trusted verbatim, so the producer must guarantee it is the matrix
/// `anchor_hessian(Exact)` would compute at `eta_hats[i]` — `analytic_terminal_work` gates
/// on the same `analytic_score_supported` predicate for exactly that reason.
#[allow(clippy::too_many_arguments)]
pub(crate) fn agq_population_nll_prepared(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[nalgebra::DVector<f64>],
    kappas: &[Vec<nalgebra::DVector<f64>>],
    n_nodes: usize,
    anchor: HessianAnchor,
    schedules: Option<&[Option<pk::event_driven::EventSchedule>]>,
    terminal_work: Option<&[TerminalLaplaceWork]>,
) -> f64 {
    agq_population_nll_impl(
        model,
        population,
        params,
        eta_hats,
        kappas,
        n_nodes,
        anchor,
        schedules,
        terminal_work,
    )
}

#[allow(clippy::too_many_arguments)]
fn agq_population_nll_impl(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[nalgebra::DVector<f64>],
    kappas: &[Vec<nalgebra::DVector<f64>>],
    n_nodes: usize,
    anchor: HessianAnchor,
    schedules: Option<&[Option<pk::event_driven::EventSchedule>]>,
    terminal_work: Option<&[TerminalLaplaceWork]>,
) -> f64 {
    let (nodes, weights, cached_log_weights) = cached_gauss_hermite(n_nodes);
    let uncached_log_weights;
    let log_weights = if cached_log_weights_enabled() {
        cached_log_weights
    } else {
        uncached_log_weights = weights.iter().map(|w| w.ln()).collect::<Vec<_>>();
        &uncached_log_weights
    };
    let per_subject: Vec<f64> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let subj_kappas = kappas.get(i).map(Vec::as_slice).unwrap_or(&[]);
            let stack = Stack::new(model, params, subj_kappas.len());
            let b_hat = stack_mode(eta_hats[i].as_slice(), subj_kappas);
            let schedule = schedules.and_then(|s| s[i].as_ref());
            agq_subject_evaluate(
                model,
                subject,
                params,
                &stack,
                &b_hat,
                nodes,
                log_weights,
                anchor,
                schedule,
                terminal_work.and_then(|all| all[i].0.as_ref()),
                terminal_work.map(|all| all[i].1),
            )
        })
        .collect();
    per_subject.iter().sum()
}

/// Evaluate the AGQ population objective and its matching gradient in one subject pass.
/// Each worker consumes and drops a subject's retained grid before taking another subject;
/// only the small per-subject objective and packed score survive for ordered reduction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn agq_population_evaluate(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[nalgebra::DVector<f64>],
    kappas: &[Vec<nalgebra::DVector<f64>>],
    bounds: &crate::estimation::parameterization::PackedBounds,
    options: &crate::types::FitOptions,
    n_nodes: usize,
    anchor: HessianAnchor,
) -> PopulationEvaluation {
    agq_population_evaluate_impl(
        model, population, params, template, x, eta_hats, kappas, bounds, options, n_nodes, anchor,
        None, None,
    )
}

/// Same evaluation as [`agq_population_evaluate`], but with a caller-hoisted
/// [`crate::estimation::inner_optimizer::build_schedule_cache`] — see
/// [`agq_population_nll_prepared`]. Used only by the FOCEI/Laplace/AGQ outer hot loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn agq_population_evaluate_with_schedules(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[nalgebra::DVector<f64>],
    kappas: &[Vec<nalgebra::DVector<f64>>],
    bounds: &crate::estimation::parameterization::PackedBounds,
    options: &crate::types::FitOptions,
    n_nodes: usize,
    anchor: HessianAnchor,
    schedules: &[Option<pk::event_driven::EventSchedule>],
    terminal_work: &[TerminalLaplaceWork],
) -> PopulationEvaluation {
    agq_population_evaluate_impl(
        model,
        population,
        params,
        template,
        x,
        eta_hats,
        kappas,
        bounds,
        options,
        n_nodes,
        anchor,
        Some(schedules),
        Some(terminal_work),
    )
}

#[allow(clippy::too_many_arguments)]
fn agq_population_evaluate_impl(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[nalgebra::DVector<f64>],
    kappas: &[Vec<nalgebra::DVector<f64>>],
    bounds: &crate::estimation::parameterization::PackedBounds,
    options: &crate::types::FitOptions,
    n_nodes: usize,
    anchor: HessianAnchor,
    schedules: Option<&[Option<pk::event_driven::EventSchedule>]>,
    terminal_work: Option<&[TerminalLaplaceWork]>,
) -> PopulationEvaluation {
    let (nodes, weights) = gauss_hermite(n_nodes);
    let log_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
    let context = SubjectScoreContext::new(model, template, x, options, bounds, false);

    let per_subject: Vec<(f64, Option<(Vec<f64>, ScoreRoute)>)> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let subj_kappas = kappas.get(i).map(Vec::as_slice).unwrap_or(&[]);
            let stack = Stack::new(model, params, subj_kappas.len());
            let b_hat = stack_mode(eta_hats[i].as_slice(), subj_kappas);
            let cached_schedule = schedules.and_then(|s| s[i].as_ref());
            let local_schedule = if cached_schedule.is_none() {
                cacheable_schedule(model, subject)
            } else {
                None
            };
            let schedule = cached_schedule.or(local_schedule.as_ref());
            let nll = agq_subject_evaluate(
                model,
                subject,
                params,
                &stack,
                &b_hat,
                &nodes,
                &log_weights,
                anchor,
                schedule,
                None,
                terminal_work.map(|all| all[i].1),
            );
            let score = context.score_routed(subject, eta_hats[i].as_slice(), subj_kappas, true);
            (nll, score)
        })
        .collect();
    let nll = per_subject.iter().map(|(nll, _)| nll).sum();
    let gradient = per_subject.iter().try_fold(
        vec![0.0; x.len()],
        |mut result, (_, score)| -> Option<Vec<f64>> {
            for (out, g) in result.iter_mut().zip(&score.as_ref()?.0) {
                *out += 2.0 * g;
            }
            Some(result)
        },
    );
    let routes = per_subject
        .iter()
        .map(|(_, s)| s.as_ref().map_or(ScoreRoute::Reconverged, |s| s.1))
        .collect();
    PopulationEvaluation {
        nll,
        gradient,
        routes,
    }
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

// # The analytic outer gradient
//
// Every subject's `dF/dx` comes from the adjoint (`sens::ode_provider::adjoint::run_agq`):
// at the mode, `Ψ = F − wᵀ∇_η l` with `w = H⁻¹∇_η F`, its partial derivative at fixed `η̂`
// the total derivative. The node displacement `b_j = η̂ + √2·L⁻ᵀz_j` and the anchor `H_a`
// sit on the reverse tape, so the grid response — 26% of the gradient at `n_agq = 1`,
// 0.8% at 3, on warfarin — is differentiated with the fixed-node score, with no Hessian
// rebuild and no inner re-solve. A subject outside the adjoint's scope takes the
// reconverged per-subject finite difference (`SubjectScoreContext::finite_difference_score`).

/// Alias kept for `build_info::gradient_method_outer`, which reports scope by model.
/// Identical to [`analytic_score_supported`] — the predicate is purely model-level.
pub fn analytic_score_supported_model(model: &CompiledModel) -> bool {
    analytic_score_supported(model)
}

/// Whether the AGQ/Laplace analytic score applies to `model`: **the same predicate
/// FOCE/FOCEI gate on** ([`analytic_outer_gradient_available`](crate::sens::provider::analytic_outer_gradient_available)),
/// with no separate copy, so widening the adjoint's scope extends FOCE, FOCEI, AGQ and Laplace
/// together (#251). Model-level only: a subject the adjoint declines at run time takes the
/// per-subject finite-difference score.
pub fn analytic_score_supported(model: &CompiledModel) -> bool {
    crate::sens::provider::analytic_outer_gradient_available(model)
}

/// Whether AGQ/Laplace supplies its own outer gradient for `model` — always: each
/// subject's score is the adjoint's exact `dFᵢ/dx` where it serves the subject (every block,
/// `block_sigma`'s ρ included) and the per-subject finite-difference score
/// (`SubjectScoreContext::finite_difference_score`) where it does not, a non-Gaussian
/// endpoint's subjects among them. So a gradient-based optimizer stays available for the
/// models AGQ exists for. Kept as a predicate so a model class with no gradient at all has
/// one place to opt out. Which subjects were analytic is reported per fit (the outer FD
/// fallback warning), and [`analytic_score_supported`] is the model-level scope.
pub fn analytic_gradient_available(model: &CompiledModel) -> bool {
    let _ = model;
    true
}

/// Analytic packed gradient of the AGQ objective for one subject: the posterior-weighted
/// average of fixed-η scores over the quadrature nodes, **plus** the grid-response term that
/// makes it the exact total derivative at every `n_agq` — both recorded on the adjoint's
/// tape (`adjoint::run_agq`, see the module note above).
///
/// `out` is the population accumulator; this adds `∂F_i/∂x` into it.
fn agq_subject_packed_gradient(
    model: &CompiledModel,
    subject: &Subject,
    pt: &crate::sens::ode_provider::adjoint::PackedPoint<'_>,
    b_hat: &[f64],
    n_agq: usize,
    anchor: HessianAnchor,
    out: &mut [f64],
) -> Option<()> {
    // AGQ (either anchor, any node count) by the adjoint (`sens::ode_provider::adjoint`).
    use crate::sens::ode_provider::adjoint::{subject_adjoint_gradient_at, Objective};
    let objective = Objective::Agq {
        n_agq,
        exact: matches!(anchor, HessianAnchor::Exact),
    };
    let g = subject_adjoint_gradient_at(model, subject, pt, b_hat, objective)?;
    if g.len() != out.len() || !g.iter().all(|v| v.is_finite()) {
        return None;
    }
    for (o, v) in out.iter_mut().zip(&g) {
        *o += v;
    }
    Some(())
}

/// Shared quadrature inputs for subject scores. No synthetic population or optimizer
/// penalty is involved in the numerical score: failed inner solves decline the score.
pub(crate) struct SubjectScoreContext<'a> {
    model: &'a CompiledModel,
    template: &'a ModelParameters,
    x: &'a [f64],
    options: &'a crate::types::FitOptions,
    bounds: &'a crate::estimation::parameterization::PackedBounds,
    nodes: Vec<f64>,
    log_weights: Vec<f64>,
    fixed: Vec<bool>,
    force_fd: bool,
    /// The inner tolerance of the finite-difference score's re-solves. Its probes start at
    /// the subject's mode, so for the outer gradient it is the stencil tolerance
    /// (`stencil_inner_tol`, #2261); the covariance step's S-matrix keeps its own
    /// (`with_fd_inner_tol`).
    fd_inner_tol: f64,
    /// `x` unpacked once for every subject this context scores (`None` when `x` is not
    /// `template`'s packed length, which the adjoint declines).
    point: Option<crate::sens::ode_provider::adjoint::PackedPoint<'a>>,
}

impl<'a> SubjectScoreContext<'a> {
    pub(crate) fn new(
        model: &'a CompiledModel,
        template: &'a ModelParameters,
        x: &'a [f64],
        options: &'a crate::types::FitOptions,
        bounds: &'a crate::estimation::parameterization::PackedBounds,
        force_fd: bool,
    ) -> Self {
        let (nodes, weights, _) =
            cached_gauss_hermite(options.agq_nodes().expect("quadrature stage"));
        Self {
            model,
            template,
            x,
            options,
            bounds,
            nodes: nodes.to_vec(),
            log_weights: weights.iter().map(|w| w.ln()).collect(),
            fixed: crate::estimation::parameterization::packed_fixed_mask(template),
            force_fd: force_fd || !analytic_gradient_available(model),
            fd_inner_tol: crate::estimation::inner_optimizer::stencil_inner_tol(options.inner_tol),
            point: crate::sens::ode_provider::adjoint::PackedPoint::new(template, x),
        }
    }

    /// This context with its finite-difference score re-solving at `tol`.
    pub(crate) fn with_fd_inner_tol(self, tol: f64) -> Self {
        Self {
            fd_inner_tol: tol,
            ..self
        }
    }

    /// Negative-log-likelihood score: the exact `dFᵢ/dx`, else the reconverged
    /// finite-difference score. The covariance step's form, which never takes a held-EBE
    /// score into `S`.
    pub(crate) fn score(
        &self,
        subject: &Subject,
        eta: &[f64],
        kappas: &[nalgebra::DVector<f64>],
    ) -> Option<Vec<f64>> {
        self.score_routed(subject, eta, kappas, false)
            .map(|(g, _)| g)
    }

    /// [`Self::score`], and with `held_ok` the optimizer's salvage between the two: a
    /// subject whose mode identity the adjoint declines takes the held-EBE score
    /// ([`ScoreRoute::HeldEbe`]) rather than `2·n_x` EBE re-solves — what FOCE/FOCEI do with
    /// the same subject. Not under IOV, whose FOCEI salvage is the reconverged one.
    pub(crate) fn score_routed(
        &self,
        subject: &Subject,
        eta: &[f64],
        kappas: &[nalgebra::DVector<f64>],
        held_ok: bool,
    ) -> Option<(Vec<f64>, ScoreRoute)> {
        if crate::cancel::is_cancelled(&self.options.cancel) {
            return None;
        }
        let joint_warm = stack_mode(eta, kappas);
        if !self.force_fd {
            // The adjoint (any node count, either anchor): the exact `dFᵢ/dx`.
            use crate::sens::ode_provider::adjoint::{
                subject_adjoint_gradient_at, subject_adjoint_held_ebe_score_at, Objective,
            };
            let objective = Objective::Agq {
                n_agq: self.nodes.len(),
                exact: matches!(self.options.hessian_anchor(), HessianAnchor::Exact),
            };
            let accept = |g: Option<Vec<f64>>| {
                let mut g = g?;
                if g.len() != self.x.len() || !g.iter().all(|v| v.is_finite()) {
                    return None;
                }
                for (v, fixed) in g.iter_mut().zip(&self.fixed) {
                    if *fixed {
                        *v = 0.0;
                    }
                }
                Some(g)
            };
            if let Some(pt) = &self.point {
                let b = &joint_warm[..];
                let exact = subject_adjoint_gradient_at(self.model, subject, pt, b, objective);
                if let Some(g) = accept(exact) {
                    return Some((g, ScoreRoute::Exact));
                }
                if held_ok && self.model.n_kappa == 0 {
                    let held =
                        subject_adjoint_held_ebe_score_at(self.model, subject, pt, b, objective);
                    if let Some(g) = accept(held) {
                        return Some((g, ScoreRoute::HeldEbe));
                    }
                }
            }
        }
        self.finite_difference_score(subject, &joint_warm)
            .map(|g| (g, ScoreRoute::Reconverged))
    }

    fn reconverged_nll(&self, subject: &Subject, xv: &[f64], joint_warm: &[f64]) -> Option<f64> {
        use crate::estimation::parameterization::{compute_mu_k, unpack_params};
        if crate::cancel::is_cancelled(&self.options.cancel) {
            return None;
        }
        let p = unpack_params(xv, self.template);
        let mu = compute_mu_k(self.model, &p.theta, self.options.mu_referencing);
        let ebe = crate::estimation::inner_optimizer::find_ebe(
            self.model,
            subject,
            &p,
            self.options.inner_maxiter,
            self.fd_inner_tol,
            Some(joint_warm),
            Some(&mu),
            self.options.inner_restarts,
        );
        if ebe.hard_reject
            || !ebe.converged
            || !ebe.nll.is_finite()
            || ebe.nll >= SUBJECT_SENTINEL_NLL
        {
            return None;
        }
        let stack = Stack::new(self.model, &p, ebe.kappas.len());
        let b = stack_mode(ebe.eta.as_slice(), &ebe.kappas);
        let nll = agq_subject_nll(
            self.model,
            subject,
            &p,
            &stack,
            &b,
            &self.nodes,
            &self.log_weights,
            self.options.hessian_anchor(),
            None,
        );
        (nll.is_finite() && nll < SUBJECT_SENTINEL_NLL).then_some(nll)
    }

    fn finite_difference_score(&self, subject: &Subject, joint_warm: &[f64]) -> Option<Vec<f64>> {
        self.reconverged_nll(subject, self.x, joint_warm)?;
        let mut g = vec![0.0; self.x.len()];
        for k in 0..self.x.len() {
            if self.fixed[k] {
                continue;
            }
            let floor = crate::estimation::parameterization::fd_step_floor(
                self.template,
                &self.model.default_params,
                k,
            );
            let h = crate::estimation::parameterization::fd_step(1e-4, self.x[k], floor);
            let mut xp = self.x.to_vec();
            let mut xm = self.x.to_vec();
            xp[k] = (self.x[k] + h).min(self.bounds.upper[k]);
            xm[k] = (self.x[k] - h).max(self.bounds.lower[k]);
            let width = xp[k] - xm[k];
            if width.abs() < 1e-16 {
                continue;
            }
            let fp = self.reconverged_nll(subject, &xp, joint_warm)?;
            let fm = self.reconverged_nll(subject, &xm, joint_warm)?;
            g[k] = (fp - fm) / width;
            if !g[k].is_finite() {
                return None;
            }
        }
        Some(g)
    }
}

/// Use analytic scores where possible and reconverge only subjects needing numerical scores.
#[allow(clippy::too_many_arguments)]
pub(crate) fn population_gradient_mixed(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[nalgebra::DVector<f64>],
    kappas: &[Vec<nalgebra::DVector<f64>>],
    bounds: &crate::estimation::parameterization::PackedBounds,
    options: &crate::types::FitOptions,
    force_fd: bool,
    evaluation: Option<PopulationEvaluation>,
    declines: Option<&crate::estimation::outer_optimizer::OuterFdDeclineLog>,
) -> Option<Vec<f64>> {
    if let Some(evaluation) = evaluation {
        if let Some(log) = declines {
            for (i, route) in evaluation.routes.iter().enumerate() {
                route.record(log, i);
            }
        }
        return evaluation.gradient;
    }
    let context = SubjectScoreContext::new(model, template, x, options, bounds, force_fd);
    let scores: Vec<_> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            context.score_routed(
                subject,
                eta_hats[i].as_slice(),
                kappas.get(i).map(Vec::as_slice).unwrap_or(&[]),
                true,
            )
        })
        .collect();
    let mut result = vec![0.0; x.len()];
    for (i, score) in scores.into_iter().enumerate() {
        let (score, route) = score?;
        if let Some(log) = declines {
            route.record(log, i);
        }
        for (out, g) in result.iter_mut().zip(score) {
            *out += 2.0 * g;
        }
    }
    result.iter().all(|v| v.is_finite()).then_some(result)
}

/// Packed gradient of the AGQ **OFV** (`2 · Σᵢ Fᵢ`), or `None` if a complete finite gradient
/// cannot be formed for any subject. Unavailable analytic ingredients use the numerical
/// routes above; a failure there declines the whole gradient rather than omitting terms.
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
    let _ = params;
    // One unpack for the population: every subject is at the same packed point.
    let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(template, x)?;
    let per_subject: Vec<Option<Vec<f64>>> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let subj_kappas = kappas.get(i).map(Vec::as_slice).unwrap_or(&[]);
            let b_hat = stack_mode(eta_hats[i].as_slice(), subj_kappas);
            let mut g = vec![0.0f64; n_packed];
            agq_subject_packed_gradient(model, subject, &pt, &b_hat, n_nodes, anchor, &mut g)
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

    #[test]
    fn inline_iov_slice_table_is_bitwise_identical_to_legacy_allocation() {
        use crate::io::datareader::read_nonmem_csv;
        use crate::parser::model_parser::parse_model_file;
        use std::path::Path;

        let model = parse_model_file(Path::new("examples/warfarin_iov.ferx")).expect("model");
        let population =
            read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC")).expect("data");
        let eta_hats = vec![DVector::zeros(model.n_eta); population.subjects.len()];
        let kappas: Vec<Vec<DVector<f64>>> = population
            .subjects
            .iter()
            .map(|subject| {
                vec![
                    DVector::zeros(model.n_kappa);
                    crate::stats::likelihood::iov_occasion_groups(subject).len()
                ]
            })
            .collect();
        let evaluate = || {
            agq_population_nll(
                &model,
                &population,
                &model.default_params,
                &eta_hats,
                &kappas,
                1,
                HessianAnchor::GaussNewton,
            )
        };

        LEGACY_KAPPA_SLICE_ALLOCATION.store(true, std::sync::atomic::Ordering::Relaxed);
        let legacy = evaluate();
        LEGACY_KAPPA_SLICE_ALLOCATION.store(false, std::sync::atomic::Ordering::Relaxed);
        let inline = evaluate();
        assert_eq!(legacy.to_bits(), inline.to_bits());
    }

    use crate::parser::model_parser::parse_model_string;

    // --- Phase 0 (#251): the analytic fixed-b score reaches full FOCE/FOCEI scope -------
    //
    // AGQ's score used to hand-roll a *Gaussian-only* residual chain, so five endpoint
    // families that FOCE/FOCEI already handled analytically (M3-BLOQ, `iiv_on_ruv`, a
    // custom sigma magnitude, LTBS, correlated residual) fell back to the fixed-b FD score.
    // It now shares the adjoint recorder (`sens::ode_provider::adjoint`), so the chain is the
    // SAME one FOCE/FOCEI uses and the per-family gates are gone.
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

    /// A subject with realistic nonzero residuals, built from the model at a reference eta.
    fn score_subject(model: &CompiledModel, theta: &[f64], times: &[f64]) -> Subject {
        use crate::types::DoseEvent;
        use std::collections::HashMap;
        let n = times.len();
        let mut subject = Subject {
            obs_rgroup: Vec::new(),
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
            pk_only_raw_times: Vec::new(),
            reset_period_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            pk_only_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
            endpoint_covariates: Vec::new(),
            filled_cells: Vec::new(),
            simulated_mixnum: None,
        };
        let eta_ref = [0.12, -0.08, 0.2];
        let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, &eta_ref);
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        subject
    }

    #[test]
    fn fused_objective_work_reproduces_objective_and_gradient() {
        use crate::estimation::inner_optimizer::find_ebe;
        use crate::estimation::parameterization::{compute_bounds, pack_params, unpack_params};
        use crate::types::{EstimationMethod, FitOptions, Population};

        let model = parse_model_string(M3_MODEL).unwrap();
        let template = &model.default_params;
        let x = pack_params(template);
        // The optimizer callback evaluates both objective and gradient from unpacked `x`.
        // Mirror that path: pack(default) can move a log-transformed value by one ULP.
        let params = unpack_params(&x, template);
        let subject = score_subject(&model, &params.theta, &[0.5, 1.0, 2.0, 4.0, 8.0]);
        let ebe = find_ebe(&model, &subject, &params, 200, 1e-11, None, None, 0);
        assert!(ebe.converged, "fixture EBE must converge");
        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let eta_hats = vec![ebe.eta];
        let kappas = vec![Vec::new()];
        let bounds = compute_bounds(template);
        for (method, anchor) in [
            (EstimationMethod::FoceI, HessianAnchor::GaussNewton),
            (EstimationMethod::Laplace, HessianAnchor::Exact),
        ] {
            let options = FitOptions {
                method,
                n_agq: 3,
                ..FitOptions::default()
            };
            let evaluation = agq_population_evaluate(
                &model,
                &population,
                &params,
                template,
                &x,
                &eta_hats,
                &kappas,
                &bounds,
                &options,
                3,
                anchor,
            );
            let terms_only =
                agq_population_nll(&model, &population, &params, &eta_hats, &kappas, 3, anchor);
            assert!(evaluation.gradient.is_some());
            assert!(
                (evaluation.nll - terms_only).abs() <= 1e-13 * (1.0 + terms_only.abs()),
                "{anchor:?} fused objective {} differs from terms-only {}",
                evaluation.nll,
                terms_only
            );

            let uncached = population_gradient_mixed(
                &model,
                &population,
                template,
                &x,
                &eta_hats,
                &kappas,
                &bounds,
                &options,
                false,
                None,
                None,
            )
            .expect("uncached gradient");
            let fused = population_gradient_mixed(
                &model,
                &population,
                template,
                &x,
                &eta_hats,
                &kappas,
                &bounds,
                &options,
                false,
                Some(evaluation),
                None,
            )
            .expect("fused gradient");
            assert_eq!(fused.len(), uncached.len());
            for (i, (fused, uncached)) in fused.iter().zip(&uncached).enumerate() {
                assert!(
                    (fused - uncached).abs() <= 1e-13 * (1.0 + uncached.abs()),
                    "{anchor:?} packed gradient coordinate {i}: \
                     fused={fused:e}, uncached={uncached:e}"
                );
            }
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

    #[test]
    fn cached_gauss_hermite_reuses_the_exact_rule_storage() {
        let (nodes_a, weights_a, logs_a) = cached_gauss_hermite(3);
        let (nodes_b, weights_b, logs_b) = cached_gauss_hermite(3);
        assert!(std::ptr::eq(nodes_a.as_ptr(), nodes_b.as_ptr()));
        assert!(std::ptr::eq(weights_a.as_ptr(), weights_b.as_ptr()));
        assert!(std::ptr::eq(logs_a.as_ptr(), logs_b.as_ptr()));
        let (expected_nodes, expected_weights) = gauss_hermite(3);
        assert_eq!(nodes_a, expected_nodes);
        assert_eq!(weights_a, expected_weights);
        assert_eq!(logs_a, weights_a.iter().map(|w| w.ln()).collect::<Vec<_>>());
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

    /// `AGQ(1, GaussNewton)` **is** the FOCEI marginal — pinned, and its residual measured.
    ///
    /// This identity is the framing the analytic AGQ covariance Hessian (#251) is built on:
    /// `F_agq = F_focei + Δ`, where `Δ` is purely the quadrature refinement and vanishes at
    /// one node. If the identity did not hold, the decomposition would be assembling a
    /// second derivative of the wrong object. Nothing in the repo pinned it — the existing
    /// tests pin the *Exact*-anchor AGQ(1) against NONMEM `LAPLACIAN`, which is the other
    /// anchor, and the `d == 0` degenerate case.
    ///
    /// The residual is **not zero**, and the reason is load-bearing for the derivative work:
    /// `build_proposal` regularises with a per-dimension relative jitter
    /// `Λᵢᵢ = max(1e-6·|Hᵢᵢ|, 1e-10)`, so the objective's log-determinant term is
    /// `½log|H̃ + Λ|`, not `½log|H̃|`. To first order the gap is `½·1e-6·tr((H̃+Λ)⁻¹diag|H̃|)`,
    /// i.e. `~½·1e-6·d` — small, but a *systematic* offset, not noise. Any "exact" analytic
    /// derivative of this objective must therefore differentiate `H̃ + Λ`; differentiating
    /// `H̃` alone would be wrong at the same relative order and would show up as a
    /// consistent bias no amount of step-size tuning removes.
    ///
    /// The assertion is two-sided on purpose: an upper bound catches the identity breaking,
    /// and a lower bound catches the jitter silently disappearing (which would make a future
    /// `H̃`-only derivative look correct here while being wrong in general).
    #[test]
    fn agq_one_node_gauss_newton_is_the_focei_marginal_up_to_the_proposal_jitter() {
        use crate::estimation::inner_optimizer::find_ebe;
        use crate::stats::likelihood::foce_subject_nll;

        let model = parse_model_string(M3_MODEL).expect("parse");
        let theta = [0.22, 11.0, 1.4];
        let subject = score_subject(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let mut params = model.default_params.clone();
        params.theta = theta.to_vec();

        let ebe = find_ebe(&model, &subject, &params, 200, 1e-11, None, None, 0);
        let focei = foce_subject_nll(
            &model,
            &subject,
            &params.theta,
            &ebe.eta,
            &ebe.h_matrix,
            &params.omega,
            &params.sigma.values,
            &params.residual_correlations,
            true,
        );

        let stack = Stack::new(&model, &params, 0);
        // `gauss_hermite` returns **weights**, not log-weights — the production call sites
        // take `.ln()` themselves. Destructuring straight into `log_weights` costs a silent
        // `d·(√π − ln√π) = 1.2·d` offset in the objective.
        let (nodes, weights) = gauss_hermite(1);
        let log_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
        let agq = agq_subject_nll(
            &model,
            &subject,
            &params,
            &stack,
            ebe.eta.as_slice(),
            &nodes,
            &log_weights,
            HessianAnchor::GaussNewton,
            None,
        );

        let d = stack.d();
        assert_eq!(d, 3, "fixture must have 3 random effects");

        // The closed form AGQ(1) must reduce to, stated independently of `agq_subject_nll`
        // so the test pins the *reduction* and not merely its own arithmetic.
        let mut scratch = pk::EventPkParams::with_capacity_for(&subject);
        let schedule = cacheable_schedule(&model, &subject);
        let nll_at_mode = stack.nll_at(
            &model,
            &subject,
            &params,
            ebe.eta.as_slice(),
            &mut scratch,
            schedule.as_ref(),
        );
        let h_anchor = anchor_hessian(
            HessianAnchor::GaussNewton,
            &model,
            &subject,
            &params,
            &stack,
            ebe.eta.as_slice(),
            &mut scratch,
            schedule.as_ref(),
        )
        .expect("gauss-newton anchor in scope");
        let proposal = build_proposal(&h_anchor, &stack.omega_joint_inv, d).expect("proposal");
        let laplace_closed_form = nll_at_mode + 0.5 * proposal.log_det_inv_scale;
        assert!(
            (agq - laplace_closed_form).abs() < 1e-12,
            "AGQ(1) must reduce to nll(b̂) + ½log|H_reg| exactly: agq={agq}, \
             closed form={laplace_closed_form}"
        );

        let gap = (agq - focei).abs();

        // Upper bound: a few times the first-order jitter estimate `½·1e-6·d`. Anything
        // larger means the two are no longer the same marginal.
        let jitter_scale = 0.5e-6 * d as f64;
        assert!(
            gap < 20.0 * jitter_scale,
            "AGQ(1, GaussNewton) must be the FOCEI marginal: agq={agq}, focei={focei}, \
             gap={gap:.3e}, expected < {:.3e}",
            20.0 * jitter_scale
        );
        // Lower bound: the jitter is really there. If this trips, `build_proposal` stopped
        // regularising and the derivative work below can drop its `Λ` terms.
        assert!(
            gap > 1e-9,
            "expected a non-zero proposal-jitter offset, got gap={gap:.3e} — if \
             `build_proposal` no longer jitters, the analytic AGQ derivative may \
             differentiate H̃ directly"
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

    /// `agq_population_nll_prepared` / `agq_population_evaluate_with_schedules` must be
    /// bit-identical to the uncached `agq_population_nll` / `agq_population_evaluate` they
    /// wrap — with the schedule cache alone, and with the terminal Hessians the inner solve
    /// actually retains (`analytic_terminal_work`, the production producer; not
    /// `anchor_hessian` fed back into itself, which would only assert `f(x) == f(x)`). Mixes a subject with an `EVID 3/4` reset (`cacheable_schedule` returns `Some`)
    /// and one with neither a reset nor a time-varying covariate (`None`) in the same
    /// population, since that mix is what would expose the cache's per-subject index ever
    /// drifting out of alignment with `population.subjects`.
    #[test]
    fn cached_schedule_agq_population_matches_uncached() {
        use crate::estimation::inner_optimizer::{build_schedule_cache, find_ebe};
        use crate::estimation::parameterization::{compute_bounds, pack_params};
        use crate::types::{DoseEvent, EstimationMethod, FitOptions, Population};

        let model = parse_model_string(M3_MODEL).unwrap();
        let template = &model.default_params;
        let x = pack_params(template);
        let params = crate::estimation::parameterization::unpack_params(&x, template);

        let no_reset = score_subject(&model, &params.theta, &[0.5, 1.0, 2.0, 4.0, 8.0]);
        assert!(!no_reset.has_resets());

        let mut reset_subject = score_subject(&model, &params.theta, &[0.5, 1.0, 2.0, 4.0, 8.0]);
        reset_subject
            .doses
            .push(DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0));
        reset_subject.reset_times = vec![12.0];
        let preds = crate::pk::compute_predictions_with_tv(
            &model,
            &reset_subject,
            &params.theta,
            &[0.12, -0.08, 0.2],
        );
        reset_subject.observations = preds.iter().map(|p| p * 0.85).collect();
        assert!(reset_subject.has_resets());

        let population = Population {
            subjects: vec![no_reset, reset_subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let schedules = build_schedule_cache(&model, &population);
        assert!(schedules[0].is_none());
        assert!(schedules[1].is_some());

        let eta_hats: Vec<_> = population
            .subjects
            .iter()
            .map(|s| find_ebe(&model, s, &params, 200, 1e-11, None, None, 0).eta)
            .collect();
        let kappas = vec![Vec::new(); population.subjects.len()];
        let bounds = compute_bounds(template);

        for (method, anchor) in [
            (EstimationMethod::FoceI, HessianAnchor::GaussNewton),
            (EstimationMethod::Laplace, HessianAnchor::Exact),
        ] {
            let options = FitOptions {
                method,
                n_agq: 3,
                ..FitOptions::default()
            };
            let uncached_nll =
                agq_population_nll(&model, &population, &params, &eta_hats, &kappas, 3, anchor);
            let cached_nll = agq_population_nll_prepared(
                &model,
                &population,
                &params,
                &eta_hats,
                &kappas,
                3,
                anchor,
                Some(&schedules),
                None,
            );
            assert_eq!(
                uncached_nll.to_bits(),
                cached_nll.to_bits(),
                "{anchor:?}: cached population NLL diverged from the uncached rebuild"
            );
            let terminal_work: Vec<_> = population
                .subjects
                .iter()
                .enumerate()
                .map(|(i, subject)| {
                    let stack = Stack::new(&model, &params, 0);
                    let mut scratch = pk::EventPkParams::with_capacity_for(subject);
                    let hessian = matches!(anchor, HessianAnchor::Exact)
                        .then(|| {
                            anchor_hessian(
                                anchor,
                                &model,
                                subject,
                                &params,
                                &stack,
                                eta_hats[i].as_slice(),
                                &mut scratch,
                                schedules[i].as_ref(),
                            )
                        })
                        .flatten();
                    let mode_nll = stack.nll_at(
                        &model,
                        subject,
                        &params,
                        eta_hats[i].as_slice(),
                        &mut scratch,
                        schedules[i].as_ref(),
                    );
                    (hessian, mode_nll)
                })
                .collect();
            if matches!(anchor, HessianAnchor::Exact) {
                let retained_terminal_work: Vec<_> = population
                    .subjects
                    .iter()
                    .enumerate()
                    .map(|(i, subject)| {
                        let retained = crate::estimation::inner_optimizer::analytic_terminal_work(
                            &model,
                            subject,
                            &params,
                            eta_hats[i].as_slice(),
                        )
                        .map(|(_, h)| h)
                        .expect("in-scope Gaussian subject retains a terminal Hessian");
                        // The retained matrix must be *the* anchor, not merely close to it:
                        // the objective trusts it verbatim in place of `anchor_hessian`.
                        let stack = Stack::new(&model, &params, 0);
                        let mut scratch = pk::EventPkParams::with_capacity_for(subject);
                        let anchored = anchor_hessian(
                            anchor,
                            &model,
                            subject,
                            &params,
                            &stack,
                            eta_hats[i].as_slice(),
                            &mut scratch,
                            schedules[i].as_ref(),
                        )
                        .expect("exact anchor always yields a matrix");
                        assert_eq!(
                            retained, anchored,
                            "subject {i}: retained terminal Hessian is not the exact anchor"
                        );
                        (Some(retained), terminal_work[i].1)
                    })
                    .collect();
                let reused = agq_population_nll_prepared(
                    &model,
                    &population,
                    &params,
                    &eta_hats,
                    &kappas,
                    3,
                    anchor,
                    Some(&schedules),
                    Some(&retained_terminal_work),
                );
                assert_eq!(
                    cached_nll.to_bits(),
                    reused.to_bits(),
                    "terminal Hessian reuse changed the exact-anchor objective"
                );

                let fresh_one = agq_population_nll_impl(
                    &model,
                    &population,
                    &params,
                    &eta_hats,
                    &kappas,
                    1,
                    anchor,
                    Some(&schedules),
                    None,
                );
                let reused_one = agq_population_nll_impl(
                    &model,
                    &population,
                    &params,
                    &eta_hats,
                    &kappas,
                    1,
                    anchor,
                    Some(&schedules),
                    Some(&terminal_work),
                );
                assert_eq!(
                    fresh_one.to_bits(),
                    reused_one.to_bits(),
                    "terminal mode NLL reuse changed the one-node Laplace objective"
                );
            }

            let uncached_eval = agq_population_evaluate(
                &model,
                &population,
                &params,
                template,
                &x,
                &eta_hats,
                &kappas,
                &bounds,
                &options,
                3,
                anchor,
            );
            let cached_eval = agq_population_evaluate_with_schedules(
                &model,
                &population,
                &params,
                template,
                &x,
                &eta_hats,
                &kappas,
                &bounds,
                &options,
                3,
                anchor,
                &schedules,
                &terminal_work,
            );
            assert_eq!(uncached_eval.nll.to_bits(), cached_eval.nll.to_bits());
            let (uncached_grad, cached_grad) = (
                uncached_eval.gradient.expect("uncached gradient"),
                cached_eval.gradient.expect("cached gradient"),
            );
            assert_eq!(uncached_grad.len(), cached_grad.len());
            for (i, (u, c)) in uncached_grad.iter().zip(&cached_grad).enumerate() {
                assert_eq!(
                    u.to_bits(),
                    c.to_bits(),
                    "{anchor:?} gradient coordinate {i}: uncached={u:e}, cached={c:e}"
                );
            }

            if matches!(anchor, HessianAnchor::Exact) {
                let fresh_one = agq_population_evaluate_impl(
                    &model,
                    &population,
                    &params,
                    template,
                    &x,
                    &eta_hats,
                    &kappas,
                    &bounds,
                    &options,
                    1,
                    anchor,
                    Some(&schedules),
                    None,
                );
                let reused_one = agq_population_evaluate_impl(
                    &model,
                    &population,
                    &params,
                    template,
                    &x,
                    &eta_hats,
                    &kappas,
                    &bounds,
                    &options,
                    1,
                    anchor,
                    Some(&schedules),
                    Some(&terminal_work),
                );
                assert_eq!(fresh_one.nll.to_bits(), reused_one.nll.to_bits());
                for (fresh, reused) in fresh_one
                    .gradient
                    .expect("fresh one-node gradient")
                    .iter()
                    .zip(reused_one.gradient.expect("reused one-node gradient"))
                {
                    assert_eq!(fresh.to_bits(), reused.to_bits());
                }
            }
        }
    }

    /// #2261: the reconverged FD score of an IOV subject (forced here; production reaches it
    /// when the adjoint declines) starts every probe at the subject's joint mode and
    /// re-solves at the stencil tolerance.
    #[test]
    fn agq_fd_score_probes_start_at_the_iov_mode_at_the_stencil_tol() {
        use crate::estimation::inner_optimizer::record_inner_solves;
        use crate::estimation::inner_optimizer::test_helpers::{
            assert_solves_at_stencil_tol, assert_starts_at_modes, on_one_thread,
            warfarin_iov_fixture,
        };
        let (model, pop, opts) = warfarin_iov_fixture();
        let opts = crate::types::FitOptions {
            method: crate::types::EstimationMethod::Laplace,
            ..opts
        };
        let p = &model.default_params;
        let x = crate::estimation::parameterization::pack_params(p);
        let bounds = crate::estimation::parameterization::compute_bounds(p);
        let (etas, _, _, kappas) = crate::estimation::inner_optimizer::run_inner_loop_warm(
            &model, &pop, p, 200, 1e-8, None, None, 0, 0,
        );
        let subject = &pop.subjects[0];
        let rec = on_one_thread(|| {
            record_inner_solves(|| {
                let context = SubjectScoreContext::new(&model, p, &x, &opts, &bounds, true);
                context
                    .score(subject, etas[0].as_slice(), &kappas[0])
                    .expect("the FD score")
            })
            .1
        });
        assert_starts_at_modes("AGQ FD score", &rec, |_| true, &pop, &etas, &kappas);
        assert_solves_at_stencil_tol("AGQ FD score", &rec, |_| true, opts.inner_tol);
    }
}
