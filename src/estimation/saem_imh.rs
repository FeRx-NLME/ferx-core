//! Independent Metropolis–Hastings (f-SAEM) E-step kernel for SAEM (#1449).
//!
//! The default SAEM E-step proposes η from the **prior**: the block kernel
//! ([`crate::estimation::saem::mh_steps`]) moves along `δ_i · chol(Ω)·z` and the
//! componentwise sweep along `δ_ij · √Ω_jj`. The conditional posterior
//! `p(η_i | y_i, θ)` is far narrower than the prior in every identified
//! direction, so a prior-shaped random walk either gets rejected or takes tiny
//! steps, and successive draws are strongly autocorrelated.
//!
//! This module implements the alternative of Karimi, Lavielle & Moulines
//! (*CSDA* 2020, arXiv:1910.12222): an **independent** proposal
//!
//! ```text
//!   η' ~ (1 − α)·N(η̂_i, c·Γ_i) + α·N(0, Ω),     Γ_i = H̃_i⁻¹,
//!   H̃_i = Hᵀ R⁻¹ H + ½ C̃ᵀC̃ + Ω⁻¹
//! ```
//!
//! centred on the subject's conditional mode η̂_i with the linearised
//! conditional covariance, plus a small defensive prior component (see
//! [`ImhProposal`]). Because the proposal does not depend on the current state,
//! accepted draws are close to i.i.d. from the conditional whenever the model is
//! near-linear in η.
//!
//! Three things make this cheap in ferx:
//!
//! * η̂_i is tracked by **Gauss-Newton** steps warm-started from the previous
//!   refresh, each one solving `H̃ δ = g` with the exact `Dual2` η-gradient
//!   ([`analytic_eta_nll_gradient`]) and the very `H̃` the proposal needs anyway;
//! * `H = ∂f/∂η` at η̂ is the same analytic Jacobian
//!   ([`crate::sens::provider::subject_eta_jacobian`]) the FOCEI inner loop
//!   builds, and H̃ is assembled by
//!   [`crate::stats::likelihood::conditional_eta_precision`] — the FOCEI
//!   curvature itself, not a second copy of the formula;
//! * `(η̂_i, H̃_i)` are cached across iterations and refreshed only every
//!   `saem_imh_refresh` iterations. A stale proposal costs *efficiency*, never
//!   correctness: IMH targets the current conditional for **any** `q` fixed
//!   before the move. That is why a refresh starts from the previous centre or
//!   the prior mean and is never handed the chain's current η
//!   ([`ImhSlot::refresh_if_due`]): the Gauss-Newton search is capped and can
//!   stop short of the mode, so a search seeded at the current state would make
//!   `q` a function of that state and void the `q(η)/q(η')` correction.
//!
//! The target density is scored by the same prepared individual NLL the
//! random-walk kernels use ([`MhScratch::score`]), so the two kernels share one
//! invariant distribution and one cached `nll` per subject.
//!
//! Two structural safety properties make an independence sampler usable here:
//!
//! 1. `H̃ ⪰ Ω⁻¹` (both `hrh` and `C̃ᵀC̃` are Gram matrices, hence PSD), so
//!    `Γ ⪯ Ω`: the main component can never be **broader** than the prior.
//! 2. A light-tailed independence sampler can still get *stuck* when the true
//!    conditional has heavier tails than the Laplace approximation — and on a
//!    `CL = TVCL·e^η` model with informative data it *will*, measurably: see
//!    [`ImhProposal`]'s note for the subject where a pure `N(η̂, cΓ)` sampler
//!    accepts 0 of 200 proposals. Hence the defensive mixture, and the fixed
//!    inflation [`IMH_INFLATE`] on Γ.

use nalgebra::DMatrix;
use rand::{Rng, RngExt};
use rand_distr::StandardNormal;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::estimation::inner_optimizer::analytic_eta_nll_gradient;
use crate::estimation::saem::MhScratch;
use crate::pk::event_driven::EventSchedule;
use crate::stats::likelihood::conditional_eta_precision;
use crate::types::{CompiledModel, OmegaMatrix, Subject};

/// Inflation `c` of the main component's covariance, `N(η̂, c·Γ)`. Slightly
/// above 1 is the standard guard against an independence sampler under-covering
/// a conditional with heavier tails than its Laplace approximation. 1.2 is the
/// value every #1449 measurement was taken at; it is not exposed as an option
/// because nothing was measured that would tell a user how to move it.
pub(crate) const IMH_INFLATE: f64 = 1.2;

/// Gauss-Newton steps per refresh. The Metropolis correction absorbs a
/// partially-converged mode, so the search only has to *track* η̂ across
/// refreshes; a step shorter than [`GN_CONVERGED_MAHA`] ends it early, so a warm
/// refresh rarely takes more than one.
pub(crate) const IMH_GN_MAX_STEPS: usize = 5;

/// Relative eigenvalue floor applied to the conditional precision when its
/// Cholesky fails. `H̃ ⪰ Ω⁻¹` holds in exact arithmetic, so a failure here is a
/// numerical artefact (a near-singular `Hᵀ R⁻¹ H` swamping `Ω⁻¹` in magnitude),
/// not a genuinely indefinite curvature.
const PRECISION_EIGEN_FLOOR_REL: f64 = 1e-10;

/// Absolute floor for the same repair, so a precision that is numerically all
/// zeros still yields an invertible (if useless) proposal rather than a `None`
/// that silently drops the subject to the random walk on every iteration.
const PRECISION_EIGEN_FLOOR_ABS: f64 = 1e-12;

/// Trust region on one Gauss-Newton MAP step, measured in the step's own `H̃`
/// metric (`‖δ‖_H̃ = √(gᵀδ)`, free once `δ` is solved). Three conditional SDs is
/// far enough that a warm subject never feels it and short enough that a single
/// Newton step on a badly non-quadratic objective cannot fling the proposal
/// centre somewhere the Metropolis correction then has to rescue.
const GN_STEP_MAHA_CAP: f64 = 3.0;

/// If the accepted GN steps have moved further than this in the same metric
/// from where `H̃` was last built, the curvature is rebuilt at the new η (one
/// extra Jacobian). Once the cached η̂ tracks the mode, steps fall well below
/// this and the rebuild stops happening — which is what makes a steady-state
/// refresh cost one gradient and one Jacobian.
const GN_RECURVE_MAHA: f64 = 0.25;

/// Newton-step length, in the same `H̃` metric, below which the MAP search is
/// converged: the step is a rounding error on the conditional's own scale.
const GN_CONVERGED_MAHA: f64 = 1e-3;

/// Per-η cap on `ηᵀΩ⁻¹η` for an accepted GN step, mirroring
/// `inner_optimizer`'s runaway-EBE guard. A step that leaves this ball is
/// rejected outright: the objective there is not one this proposal should be
/// centred on, and rejecting costs nothing (no extra evaluation).
const GN_RUNAWAY_MAHA_PER_ETA: f64 = 100.0;

// ---------------------------------------------------------------------------
// Cost counters
// ---------------------------------------------------------------------------

/// Measured E-step cost of the IMH kernel for **one** SAEM run, counted rather
/// than derived from the kernel definition. Atomic because the E-step runs the
/// subjects on rayon workers; owned by the run (not a process global) so two
/// fits in one process — a bootstrap, a test binary — never read each other's
/// counts.
#[derive(Default, Debug)]
pub(crate) struct ImhCounters {
    /// Analytic `Dual2` η-gradient evaluations spent inside the MAP refresh.
    map_grad_evals: AtomicU64,
    /// Analytic η-Jacobian (+ `H̃` assembly) evaluations.
    map_jac_evals: AtomicU64,
    /// IMH proposals drawn (each costs exactly one individual-NLL evaluation).
    proposals: AtomicU64,
    /// IMH proposals accepted.
    accepts: AtomicU64,
    /// Per-subject refreshes that produced a usable proposal.
    refreshes: AtomicU64,
    /// Per-subject refresh attempts that declined (no usable curvature or
    /// gradient at the current point).
    declines: AtomicU64,
    /// Wall nanoseconds inside the analytic η-gradient calls.
    map_grad_ns: AtomicU64,
    /// Wall nanoseconds inside the η-Jacobian + `H̃` assembly.
    map_jac_ns: AtomicU64,
    /// Wall nanoseconds inside the IMH proposals' individual-NLL calls. Paired
    /// with `proposals`, this is the per-NLL cost against which the gradient
    /// and Jacobian costs are priced — all measured on the same subjects, in
    /// the same run, under the same machine contention.
    imh_nll_ns: AtomicU64,
}

/// A snapshot of [`ImhCounters`], for differencing.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct ImhSnapshot {
    pub map_grad_evals: u64,
    pub map_jac_evals: u64,
    pub proposals: u64,
    pub accepts: u64,
    pub refreshes: u64,
    pub declines: u64,
    pub map_grad_ns: u64,
    pub map_jac_ns: u64,
    pub imh_nll_ns: u64,
}

fn bump(c: &AtomicU64, by: u64) {
    c.fetch_add(by, Ordering::Relaxed);
}

impl ImhCounters {
    pub(crate) fn snapshot(&self) -> ImhSnapshot {
        let ld = |c: &AtomicU64| c.load(Ordering::Relaxed);
        ImhSnapshot {
            map_grad_evals: ld(&self.map_grad_evals),
            map_jac_evals: ld(&self.map_jac_evals),
            proposals: ld(&self.proposals),
            accepts: ld(&self.accepts),
            refreshes: ld(&self.refreshes),
            declines: ld(&self.declines),
            map_grad_ns: ld(&self.map_grad_ns),
            map_jac_ns: ld(&self.map_jac_ns),
            imh_nll_ns: ld(&self.imh_nll_ns),
        }
    }

    fn decline(&self) {
        bump(&self.declines, 1);
    }
}

// ---------------------------------------------------------------------------
// The proposal
// ---------------------------------------------------------------------------

/// A cached independent proposal for one subject: the **defensive mixture**
///
/// ```text
///   q(η) = (1 − α) · N(η̂, c·H̃⁻¹)   +   α · N(0, Ω)
/// ```
///
/// The second component is what makes this usable rather than merely
/// attractive. Measured on a 1-cpt IV subject whose data put the conditional
/// mode at `η_CL = 1.50`: the Laplace curvature there is `sd = 0.074`, so the
/// chain's starting state `η = 0` sits **20 proposal SDs** out with
/// `log q(0) = −293`, while the objective is only `15.4` above the mode — the
/// conditional is far flatter than its own curvature at the mode, because
/// `CL = TVCL·e^η` makes the sensitivity collapse as η falls. A pure
/// `N(η̂, cΓ)` independence sampler *correctly* refuses to leave that state
/// (`log α ≈ −278`) and the kernel is inert for the whole run. Mixing in the
/// prior — the distribution η is guaranteed to live under — bounds the
/// importance ratio `p/q` and restores `log α ≈ +9` for the same move.
///
/// Every density below is fully normalised, because a *mixture* proposal's
/// normalising constants no longer cancel between the two components.
#[derive(Clone, Debug)]
pub(crate) struct ImhProposal {
    /// Conditional mode — the centre of the main component.
    pub eta_hat: Vec<f64>,
    /// Lower Cholesky factor `L` of the **precision** `H̃ = L Lᵀ` (before
    /// inflation). Storing the precision factor rather than the covariance keeps
    /// both the draw (`x = √c · L⁻ᵀ z`) and the Mahalanobis form
    /// (`‖Lᵀ(η − η̂)‖² / c`) to one triangular solve / one triangular product.
    chol_l: DMatrix<f64>,
    /// Inflation `c` applied to the main component's **covariance**.
    inflate: f64,
    /// Mixture weight `α` on the prior component. `0.0` disables it.
    defensive: f64,
    /// Lower Cholesky factor of Ω (the prior component's covariance).
    omega_chol: DMatrix<f64>,
    /// `Ω⁻¹`, for the prior component's density.
    omega_inv: DMatrix<f64>,
    /// `log` of the main component's normalising constant, excluding the shared
    /// `−(p/2)·log(2π)` (dropped from *both* components, so the mixture's
    /// log-sum-exp stays consistent).
    log_norm_main: f64,
    /// The same for the prior component.
    log_norm_prior: f64,
}

impl ImhProposal {
    /// The one constructor: every normalising constant is derived here, from
    /// the factors it normalises, so no caller can pair a factor with another
    /// factor's constant.
    fn from_parts(
        eta_hat: Vec<f64>,
        chol_l: DMatrix<f64>,
        inflate: f64,
        defensive: f64,
        omega: &OmegaMatrix,
    ) -> Self {
        // One gate for the weight: `saem::imh_knobs` sanitises it before a run
        // starts, so a value outside [0, 0.5] here is a caller bug, not input.
        debug_assert!((0.0..=0.5).contains(&defensive), "defensive = {defensive}");
        Self {
            eta_hat,
            log_norm_main: log_norm_main_component(&chol_l, inflate),
            chol_l,
            inflate,
            defensive,
            omega_chol: omega.chol.clone(),
            omega_inv: omega.inv.clone(),
            log_norm_prior: -0.5 * omega.log_det,
        }
    }

    fn n(&self) -> usize {
        self.eta_hat.len()
    }

    /// `(η − η̂)ᵀ (H̃/c) (η − η̂)` — the Mahalanobis distance under the *inflated*
    /// main-component scale matrix.
    fn maha(&self, eta: &[f64]) -> f64 {
        let n = self.n();
        // `w = Lᵀ d`, column-walked so `chol_l`'s lower-triangular storage is
        // read in its natural order.
        let mut w = vec![0.0_f64; n];
        for j in 0..n {
            let dj = eta[j] - self.eta_hat[j];
            if dj == 0.0 {
                continue;
            }
            for (i, wi) in w.iter_mut().enumerate().take(j + 1) {
                *wi += self.chol_l[(j, i)] * dj;
            }
        }
        w.iter().map(|wi| wi * wi).sum::<f64>() / self.inflate
    }

    /// `log` of the main component's density at `eta`, normalised except for the
    /// shared `−(p/2)·log(2π)`.
    fn log_main(&self, eta: &[f64]) -> f64 {
        let m = self.maha(eta);
        if !m.is_finite() {
            return f64::NEG_INFINITY;
        }
        self.log_norm_main - 0.5 * m
    }

    /// `log` of the prior component's density at `eta`, same convention.
    fn log_prior(&self, eta: &[f64]) -> f64 {
        let m = quad_form(&self.omega_inv, eta);
        if !m.is_finite() {
            return f64::NEG_INFINITY;
        }
        self.log_norm_prior - 0.5 * m
    }

    /// `log q(η)` for the mixture, up to the `−(p/2)·log(2π)` both components
    /// share (and which therefore cancels in the acceptance ratio).
    fn log_q(&self, eta: &[f64]) -> f64 {
        if self.defensive <= 0.0 {
            return self.log_main(eta);
        }
        let a = self.log_main(eta) + (1.0 - self.defensive).ln();
        let b = self.log_prior(eta) + self.defensive.ln();
        log_sum_exp2(a, b)
    }

    /// Draw `η' ~ q` into `out`. The main component solves `Lᵀ x = z` by back
    /// substitution, so `cov(x) = (L Lᵀ)⁻¹ = H̃⁻¹`; the prior component uses
    /// `chol(Ω)·z`, the same move the block random walk makes at scale 1.
    fn draw_into(&self, rng: &mut impl Rng, z: &mut [f64], out: &mut [f64]) {
        let n = self.n();
        for slot in z.iter_mut() {
            *slot = rng.sample(StandardNormal);
        }
        // The mixture indicator is drawn unconditionally (not behind a
        // short-circuit on `defensive > 0`), so a run's RNG stream does not
        // depend on the mixture weight — `defensive = 0` and `defensive > 0`
        // stay comparable proposal for proposal.
        let u: f64 = rng.random::<f64>();
        if self.defensive > 0.0 && u < self.defensive {
            for i in 0..n {
                let mut s = 0.0;
                for (j, &zj) in z.iter().enumerate().take(i + 1) {
                    s += self.omega_chol[(i, j)] * zj;
                }
                out[i] = s;
            }
            return;
        }
        for i in (0..n).rev() {
            let mut s = z[i];
            for k in (i + 1)..n {
                s -= self.chol_l[(k, i)] * out[k];
            }
            out[i] = s / self.chol_l[(i, i)];
        }
        let scale = self.inflate.sqrt();
        for j in 0..n {
            out[j] = self.eta_hat[j] + scale * out[j];
        }
    }
}

/// `xᵀ A x` for a dense symmetric `A`.
fn quad_form(a: &DMatrix<f64>, x: &[f64]) -> f64 {
    let n = x.len();
    let mut acc = 0.0;
    for i in 0..n {
        let mut s = 0.0;
        for (j, &xj) in x.iter().enumerate().take(n) {
            s += a[(i, j)] * xj;
        }
        acc += x[i] * s;
    }
    acc
}

/// `log(e^a + e^b)`, overflow-safe. Both arms may be `-inf`.
fn log_sum_exp2(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if !m.is_finite() {
        // Both `-inf` (η impossible under either component) or a poisoned value;
        // either way there is no usable density here.
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// `log` of the main component's normalising constant, minus the shared
/// `−(p/2)·log(2π)`: `½·log|Σ⁻¹| = Σ log L_ii − (p/2)·log c`.
fn log_norm_main_component(chol_l: &DMatrix<f64>, inflate: f64) -> f64 {
    let p = chol_l.nrows() as f64;
    let half_log_det_prec: f64 = (0..chol_l.nrows()).map(|i| chol_l[(i, i)].ln()).sum();
    half_log_det_prec - 0.5 * p * inflate.ln()
}

// ---------------------------------------------------------------------------
// Building the proposal
// ---------------------------------------------------------------------------

/// Cholesky of `p`, repairing a numerically indefinite matrix with a symmetric
/// eigen-floor first. Returns the lower factor `L` with `p ≈ L Lᵀ`.
fn floored_cholesky(p: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    // Symmetrise: `hrh`/`ctc` are accumulated as outer products and can carry
    // an asymmetry of a few ULP, which `SymmetricEigen` would silently ignore
    // and `cholesky` would resolve arbitrarily.
    let sym = (p + p.transpose()) * 0.5;
    if let Some(c) = sym.clone().cholesky() {
        return Some(c.l());
    }
    let eig = sym.symmetric_eigen();
    let max_ev = eig
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_ev.is_finite() {
        return None;
    }
    let floor = (max_ev.abs() * PRECISION_EIGEN_FLOOR_REL).max(PRECISION_EIGEN_FLOOR_ABS);
    let mut vals = eig.eigenvalues.clone();
    for v in vals.iter_mut() {
        if !v.is_finite() || *v < floor {
            *v = floor;
        }
    }
    let repaired = &eig.eigenvectors * DMatrix::from_diagonal(&vals) * eig.eigenvectors.transpose();
    repaired.cholesky().map(|c| c.l())
}

/// Recompute `(η̂_i, H̃_i)` for one subject by warm-started Gauss-Newton and
/// package them as an [`ImhProposal`].
///
/// Each GN step `η ← η − H̃⁻¹ g` uses the exact `Dual2` η-gradient and the
/// conditional precision the proposal needs anyway, so a refresh spends **no**
/// individual-NLL evaluations; in steady state (a warm η̂ whose step is shorter
/// than [`GN_RECURVE_MAHA`]) it costs one gradient and one Jacobian. The search
/// does not have to converge: a partially-converged η̂ is a perfectly good
/// proposal *centre*, since the Metropolis correction makes an inaccurate `q`
/// cost sampling efficiency, never correctness.
///
/// Returns `None` when the gradient or curvature is unavailable at this point —
/// the caller then keeps the random-walk kernel for this subject.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refresh_proposal(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    eta_warm: &[f64],
    max_steps: usize,
    defensive: f64,
    counters: &ImhCounters,
) -> Option<ImhProposal> {
    let n_eta = eta_warm.len();
    if n_eta == 0 || !omega.log_det.is_finite() {
        counters.decline();
        return None;
    }

    // Jacobian + precision assembly, counted and timed as one unit (they are
    // never used apart). SAEM's individual objective carries the η-dependence of
    // the residual variance, so the curvature is the interaction form.
    let curvature_at = |eta: &[f64]| -> Option<DMatrix<f64>> {
        let t0 = std::time::Instant::now();
        let sigma = model.is_sde().then_some(sigma_values);
        let out =
            eta_jacobian_and_preds(model, subject, theta, sigma, eta).and_then(|(h, f, p)| {
                conditional_eta_precision(
                    model,
                    subject,
                    theta,
                    eta,
                    &f,
                    &p,
                    &h,
                    omega,
                    sigma_values,
                    true,
                )
            });
        bump(&counters.map_jac_ns, t0.elapsed().as_nanos() as u64);
        bump(&counters.map_jac_evals, 1);
        out
    };
    let gradient_at = |eta: &[f64]| -> Option<Vec<f64>> {
        let t0 = std::time::Instant::now();
        let g = analytic_eta_nll_gradient(model, subject, theta, eta, omega, sigma_values)
            .filter(|g| g.iter().all(|v| v.is_finite()));
        bump(&counters.map_grad_ns, t0.elapsed().as_nanos() as u64);
        bump(&counters.map_grad_evals, 1);
        g
    };

    let Some((eta_hat, precision)) = gn_map_refine(
        &gradient_at,
        &curvature_at,
        eta_warm,
        omega,
        max_steps.max(1),
    ) else {
        counters.decline();
        return None;
    };
    let Some(chol_l) = floored_cholesky(&precision) else {
        counters.decline();
        return None;
    };

    bump(&counters.refreshes, 1);
    Some(ImhProposal::from_parts(
        eta_hat,
        chol_l,
        IMH_INFLATE,
        defensive,
        omega,
    ))
}

/// One subject's IMH state, carried across SAEM iterations: the cached proposal
/// (`None` = the subject takes the block random walk this iteration) and when a
/// refresh was last *attempted*, successful or not.
#[derive(Clone, Debug, Default)]
pub(crate) struct ImhSlot {
    pub(crate) prop: Option<ImhProposal>,
    attempted_at: Option<usize>,
}

impl ImhSlot {
    /// Rebuild the proposal if `refresh` iterations have passed since the last
    /// attempt (or there has been none), via `build(warm_start)`.
    ///
    /// Two properties the chain's correctness and cost rest on, both here so a
    /// test reaches them without a fit:
    ///
    /// * **The proposal never depends on the chain's current state.** An
    ///   independence sampler's `q(η)/q(η')` correction is exact only for a `q`
    ///   fixed before the move; the Gauss-Newton search stops after
    ///   `IMH_GN_MAX_STEPS`, short of the mode on a far start, so seeding it from
    ///   the current η would make `q` a function of that η. It starts from the
    ///   previous proposal's centre — a function of θ and of *past* draws only,
    ///   the usual adaptive-MCMC dependence, like the random walk's scale — and,
    ///   with none, from the prior mean `0`. This function is not handed the
    ///   current η at all.
    /// * **A declined refresh waits its turn.** A decline counts as an attempt,
    ///   so a subject whose curvature is unavailable retries every `refresh`
    ///   iterations, not every iteration on top of its random-walk fallback.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn refresh_if_due(
        &mut self,
        k: usize,
        refresh: usize,
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        omega: &OmegaMatrix,
        sigma_values: &[f64],
        defensive: f64,
        counters: &ImhCounters,
    ) {
        self.refresh_if_due_with(k, refresh, model.n_eta, |warm| {
            refresh_proposal(
                model,
                subject,
                theta,
                omega,
                sigma_values,
                warm,
                IMH_GN_MAX_STEPS,
                defensive,
                counters,
            )
        });
    }

    /// [`Self::refresh_if_due`] with the build injected, so a test can see the
    /// warm start without a model. Production reaches it only through
    /// `refresh_if_due`, whose signature carries no η of the chain.
    pub(crate) fn refresh_if_due_with(
        &mut self,
        k: usize,
        refresh: usize,
        n_eta: usize,
        build: impl FnOnce(&[f64]) -> Option<ImhProposal>,
    ) {
        let due = self
            .attempted_at
            .is_none_or(|a| k.saturating_sub(a) >= refresh.max(1));
        if !due {
            return;
        }
        let warm = self
            .prop
            .as_ref()
            .map_or_else(|| vec![0.0; n_eta], |p| p.eta_hat.clone());
        self.prop = build(&warm);
        self.attempted_at = Some(k);
    }
}

/// The η-Jacobian `H` (`n_obs × n_eta`) and the predictions `f` it was taken at,
/// from one exact recording, and with `sigma` an SDE subject's EKF variances `p_obs`
/// off the same recording (empty otherwise). The curvature needs all of them; reading
/// them off one plan run spares the refresh a second solve. `None` when the
/// recording is out of scope or any entry is non-finite.
pub(crate) fn eta_jacobian_and_preds(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: Option<&[f64]>,
    eta: &[f64],
) -> Option<(DMatrix<f64>, Vec<f64>, Vec<f64>)> {
    let rows = crate::sens::ode_provider::adjoint::exact_obs_grads_in(
        model, subject, None, theta, sigma, eta,
    )?;
    let (grads, var) = rows.split_at(rows.len().min(subject.observations.len()));
    let n_eta = eta.len();
    let f: Vec<f64> = grads.iter().map(|o| o.f).collect();
    let p: Vec<f64> = var.iter().map(|o| o.f).collect();
    let h = DMatrix::from_row_iterator(
        grads.len(),
        n_eta,
        grads.iter().flat_map(|o| o.df_deta.iter().copied()),
    );
    (h.iter().chain(&f).chain(&p).all(|v| v.is_finite())).then_some((h, f, p))
}

/// Solve `H̃ δ = g` given the lower Cholesky factor `L` of `H̃` (forward then
/// backward substitution).
fn chol_solve(l: &DMatrix<f64>, g: &[f64]) -> Vec<f64> {
    let n = g.len();
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut acc = g[i];
        for k in 0..i {
            acc -= l[(i, k)] * y[k];
        }
        y[i] = acc / l[(i, i)];
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut acc = y[i];
        for k in (i + 1)..n {
            acc -= l[(k, i)] * x[k];
        }
        x[i] = acc / l[(i, i)];
    }
    x
}

/// Gauss-Newton MAP refinement, returning `(η̂, H̃ at η̂)`.
///
/// Each step solves `H̃ δ = g` and moves to `η − δ`, trust-regioned at
/// [`GN_STEP_MAHA_CAP`] in the `H̃` metric (`‖δ‖²_H̃ = gᵀδ`) and rejected
/// outright if it would leave the prior ball ([`GN_RUNAWAY_MAHA_PER_ETA`]).
///
/// Convergence ([`GN_CONVERGED_MAHA`]) and re-curvature ([`GN_RECURVE_MAHA`])
/// are separate decisions: an earlier version used the re-curvature threshold as
/// the convergence test too, which stopped the search after one short step and
/// left it 7 prior SDs from the mode on the stress fixture. The curvature
/// returned is always one built within `GN_RECURVE_MAHA` of the returned η̂.
fn gn_map_refine(
    gradient_at: &dyn Fn(&[f64]) -> Option<Vec<f64>>,
    curvature_at: &dyn Fn(&[f64]) -> Option<DMatrix<f64>>,
    eta_warm: &[f64],
    omega: &OmegaMatrix,
    steps: usize,
) -> Option<(Vec<f64>, DMatrix<f64>)> {
    let n_eta = eta_warm.len();
    let runaway = GN_RUNAWAY_MAHA_PER_ETA * n_eta as f64;
    let mut eta = eta_warm.to_vec();
    let mut precision = curvature_at(&eta)?;
    // Distance, in the `H̃` metric, from the point `precision` was evaluated at
    // to the current `eta`. Zero means the held curvature is current.
    let mut stale_by = 0.0_f64;

    for _ in 0..steps {
        let g = gradient_at(&eta)?;
        let l = floored_cholesky(&precision)?;
        let delta = chol_solve(&l, &g);
        // `gᵀδ = δᵀ H̃ δ` is the squared step length in the curvature metric,
        // non-negative because `H̃` is PD after the floor.
        let len2: f64 = g.iter().zip(delta.iter()).map(|(a, b)| a * b).sum();
        if !len2.is_finite() || !delta.iter().all(|v| v.is_finite()) {
            break;
        }
        let len = len2.max(0.0).sqrt();
        if len < GN_CONVERGED_MAHA {
            break;
        }
        let scale = if len > GN_STEP_MAHA_CAP {
            GN_STEP_MAHA_CAP / len
        } else {
            1.0
        };
        let cand: Vec<f64> = (0..n_eta).map(|j| eta[j] - scale * delta[j]).collect();
        if !cand.iter().all(|v| v.is_finite()) || quad_form(&omega.inv, &cand) > runaway {
            break;
        }
        eta = cand;
        stale_by += len * scale;
        if stale_by > GN_RECURVE_MAHA {
            precision = curvature_at(&eta)?;
            stale_by = 0.0;
        }
    }
    Some((eta, precision))
}

// ---------------------------------------------------------------------------
// The cost report
// ---------------------------------------------------------------------------

/// What one iteration of the IMH kernel costs for one subject, and what the
/// block random walk it displaces would have cost, both in individual-NLL
/// equivalents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ImhCostEstimate {
    /// Measured cost of one analytic η-gradient, in NLL evaluations.
    pub grad_ratio: f64,
    /// Measured cost of one η-Jacobian + `H̃` assembly, in NLL evaluations.
    pub jac_ratio: f64,
    /// `m` proposals plus the amortised refresh, per subject per iteration.
    pub imh_cost: f64,
    /// `n_mh_steps` — what the block kernel would have spent instead.
    pub rw_cost: f64,
}

impl ImhCostEstimate {
    pub(crate) fn ratio(&self) -> f64 {
        if self.rw_cost > 0.0 {
            self.imh_cost / self.rw_cost
        } else {
            f64::NAN
        }
    }
}

/// The ratio above which [`imh_cost_warning`] speaks up.
pub(crate) const IMH_COST_WARN_RATIO: f64 = 1.5;

/// Price a run's measured counters into [`ImhCostEstimate`].
///
/// The per-call costs come from the nanosecond counters, so they are measured on
/// *this* model, in this run — not assumed. That matters: the gradient cost
/// ranges from ~3× an NLL (busulfan, closed-form 2-cpt) to **~22×**
/// (pembrolizumab, whose `TIME`-dependent clearance routes through the
/// event-driven predictor), and at the high end the refresh cannot amortise.
///
/// Returns `None` without a complete measurement (no refresh, no proposal).
pub(crate) fn estimate_imh_cost(
    d: &ImhSnapshot,
    n_proposals: usize,
    refresh_interval: usize,
    n_mh_steps: usize,
) -> Option<ImhCostEstimate> {
    if d.refreshes == 0 || d.proposals == 0 || d.imh_nll_ns == 0 {
        return None;
    }
    if d.map_grad_evals == 0 || d.map_jac_evals == 0 {
        return None;
    }
    let ns_per_nll = d.imh_nll_ns as f64 / d.proposals as f64;
    let grad_ratio = (d.map_grad_ns as f64 / d.map_grad_evals as f64) / ns_per_nll;
    let jac_ratio = (d.map_jac_ns as f64 / d.map_jac_evals as f64) / ns_per_nll;
    if !(grad_ratio.is_finite() && jac_ratio.is_finite()) {
        return None;
    }
    // Gradient and Jacobian calls per refresh ATTEMPT — the eval counters include
    // the declined ones — amortised over the `refresh_interval` subject-iterations
    // each attempt serves (`ImhSlot::refresh_if_due` retries a decline on the
    // same interval).
    let refresh = refresh_interval.max(1) as f64;
    let events = (d.refreshes + d.declines) as f64;
    let grad_per_si = d.map_grad_evals as f64 / events / refresh;
    let jac_per_si = d.map_jac_evals as f64 / events / refresh;
    Some(ImhCostEstimate {
        grad_ratio,
        jac_ratio,
        imh_cost: n_proposals as f64 + grad_per_si * grad_ratio + jac_per_si * jac_ratio,
        rw_cost: n_mh_steps as f64,
    })
}

/// The `FitResult.warnings` line for a run whose IMH kernel spent more than
/// [`IMH_COST_WARN_RATIO`] times the random walk it replaced, or `None`.
///
/// **A report, never a switch.** The measurement is wall-clock, so acting on it
/// would make a seeded fit's estimates depend on machine load; the kernel the
/// user asked for runs for the whole fit, and this tells them what it cost.
pub(crate) fn imh_cost_warning(est: &ImhCostEstimate) -> Option<String> {
    if !(est.ratio() > IMH_COST_WARN_RATIO) {
        return None;
    }
    Some(format!(
        "saem_kernel = imh cost {:.1} individual-NLL evaluations per subject per iteration \
         against the {:.0} the block random walk it replaced would have spent ({:.1}x): on \
         this model one analytic eta-gradient costs {:.1} NLL evaluations and one \
         eta-Jacobian {:.1}. Raising `saem_imh_refresh` amortises the proposal refresh \
         further; `saem_kernel = rw` is the cheaper E-step here.",
        est.imh_cost,
        est.rw_cost,
        est.ratio(),
        est.grad_ratio,
        est.jac_ratio,
    ))
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

/// The `FitResult.warnings` line for an IMH kernel that almost never moved the
/// chain over the **convergence phase** — the iterations whose draws the
/// estimates average — with acceptance below `stuck` (the random walk's own
/// "not mixing" threshold). A run-wide rate would let a healthy exploration
/// phase hide a frozen convergence phase. The mixing tally reads only the
/// random-walk kernels against their targets, so without this a stuck
/// independence sampler — on a one-η model, with no componentwise sweep, the
/// only move there is — would pass silently. Measured healthy acceptance on the
/// #1449 benches is ~0.8.
pub(crate) fn imh_stuck_warning(accepts: u64, proposals: u64, stuck: f64) -> Option<String> {
    if proposals == 0 {
        return None;
    }
    let rate = accepts as f64 / proposals as f64;
    (rate < stuck).then(|| {
        format!(
            "saem_kernel = imh accepted {:.2}% of its {} proposals over the convergence \
             phase — the independent proposal sits where the conditional has almost no \
             mass, so its draws barely moved the chain and the estimates rest on whatever \
             mixing the rest of the E-step managed. `saem_kernel = rw` is the safer kernel \
             for this model.",
            rate * 100.0,
            proposals
        )
    })
}

/// Run `n_steps` independent-Metropolis–Hastings proposals for one subject
/// in-place. Returns `(n_accepted, updated_nll)`.
///
/// Unlike [`crate::estimation::saem::mh_steps`] the proposal is **not
/// symmetric**, so the proposal density does not cancel: the acceptance
/// log-ratio is
///
/// ```text
///   log α = [nll(η) − nll(η')] + [log q(η) − log q(η')]
/// ```
///
/// Both `nll` terms are scored by [`MhScratch::score`], the prepared NLL the
/// random-walk kernels use, so the two kernels target the identical conditional
/// and share one cached `nll` per subject.
#[allow(clippy::too_many_arguments)]
pub(crate) fn imh_steps(
    eta: &mut [f64],
    nll_current: f64,
    prop: &ImhProposal,
    subject: &Subject,
    model: &CompiledModel,
    theta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    rng: &mut impl Rng,
    n_steps: usize,
    scratch: &mut MhScratch,
    schedule: Option<&EventSchedule>,
    counters: &ImhCounters,
) -> (usize, f64) {
    let n = eta.len();
    let mut nll = nll_current;
    let mut log_q_cur = prop.log_q(eta);
    let mut n_accepted = 0;
    let mut z = vec![0.0_f64; n];
    let mut eta_prop = vec![0.0_f64; n];
    // Accumulated locally and folded into the shared counters once per call, so
    // the rayon workers do not contend on the atomics per proposal.
    let mut nll_ns: u64 = 0;

    for _ in 0..n_steps {
        prop.draw_into(rng, &mut z, &mut eta_prop);
        let t0 = std::time::Instant::now();
        let nll_prop = scratch.score(
            model,
            subject,
            theta,
            &eta_prop,
            omega,
            sigma_values,
            schedule,
        );
        nll_ns += t0.elapsed().as_nanos() as u64;
        let log_q_prop = prop.log_q(&eta_prop);
        let log_alpha = (nll - nll_prop) + (log_q_cur - log_q_prop);
        let log_u: f64 = rng.random::<f64>().ln();
        if log_alpha.is_finite() && log_u < log_alpha {
            eta.copy_from_slice(&eta_prop);
            nll = nll_prop;
            log_q_cur = log_q_prop;
            n_accepted += 1;
        }
    }

    bump(&counters.imh_nll_ns, nll_ns);
    bump(&counters.proposals, n_steps as u64);
    bump(&counters.accepts, n_accepted as u64);
    (n_accepted, nll)
}

#[cfg(test)]
impl ImhProposal {
    /// Build a proposal straight from a precision matrix — the unit tests need
    /// a `q` with a known covariance, without a model or a subject. `omega`
    /// defaults to the identity when `None`. Goes through the production
    /// constructor, so a test of its densities is a test of the real ones.
    pub(crate) fn from_precision(
        eta_hat: Vec<f64>,
        precision: &DMatrix<f64>,
        inflate: f64,
        defensive: f64,
        omega: Option<&OmegaMatrix>,
    ) -> Option<Self> {
        let n = eta_hat.len();
        let chol_l = floored_cholesky(precision)?;
        let identity = OmegaMatrix::from_diagonal(&vec![1.0; n], vec![String::new(); n]);
        Some(Self::from_parts(
            eta_hat,
            chol_l,
            inflate,
            defensive,
            omega.unwrap_or(&identity),
        ))
    }

    /// The proposal's main-component covariance `c · H̃⁻¹`.
    pub(crate) fn scale_matrix(&self) -> DMatrix<f64> {
        let prec = &self.chol_l * self.chol_l.transpose();
        prec.try_inverse().expect("proposal precision is PD") * self.inflate
    }

    pub(crate) fn maha_for_test(&self, eta: &[f64]) -> f64 {
        self.maha(eta)
    }

    pub(crate) fn log_q_for_test(&self, eta: &[f64]) -> f64 {
        self.log_q(eta)
    }

    pub(crate) fn draw_for_test(&self, rng: &mut impl Rng) -> Vec<f64> {
        let n = self.n();
        let (mut z, mut out) = (vec![0.0; n], vec![0.0; n]);
        self.draw_into(rng, &mut z, &mut out);
        out
    }
}

#[cfg(test)]
#[path = "saem_imh_tests.rs"]
mod tests;
