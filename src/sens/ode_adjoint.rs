//! The exact outer gradient — and, in `hessian`, the covariance R-matrix — of every
//! likelihood ferx estimates (FOCE, FOCEI, Laplace, AGQ), by the adjoint method, on one
//! recorder over the production predictor.
//!
//! At the EBE `η̂` (the minimiser of the inner objective `l`), the total subject gradient
//! — the curvature term and the EBE response `dη̂/dx` included — is the partial
//! derivative, at fixed `η̂` and fixed `w`, of one scalar:
//!
//! ```text
//!   Ψ(x) = F(x, η̂) − wᵀ ∇_η l(x, η̂),      w = H⁻¹ ∇_η F   (H = ∇²_η l, exact)
//! ```
//!
//! This is the implicit-function / Lagrangian identity: `dF/dx = ∂ₓF + (∇_η F)ᵀ dη̂/dx`
//! with `dη̂/dx = −H⁻¹ ∂ₓ∇_η l`, which is `∂ₓΨ`. It holds for any objective `F` built on
//! the EBE of `l`, so FOCEI (`F = l + ½log|H̃|`) and FOCE (the Sheiner–Beal marginal,
//! with FOCE's frozen-variance `l`) differ only in the kernel function. `Ψ` depends on
//! the structural parameters only through the predictions `f` and `G = ∂f/∂η`, so no
//! second-order sensitivity in θ is ever formed. The steps:
//!
//! 1. **Record** the predictions over `DualV<N>` (`sens::vtape`): value and η-tangents
//!    on the vector tape, an η-Hessian payload alongside. θ are tape leaves, η free inputs, and the individual-parameter program itself runs on the tape
//!    (`eval_params_g`), so `∂p/∂θ` tables are never built. The solve is the production
//!    integrator's, so this is a discrete adjoint of exactly the predicted trajectory.
//! 2. **`w`**, from `(F, ∇_η l)` evaluated over a plain forward η-dual (`f` carrying `G`,
//!    `G` carrying the payload `∂²f/∂η²`): its dual parts are `∇_η F` and `H`. No tape.
//! 3. **`Ψ`** on the scalar reverse tape ([`R`]), with `f`, `G`, `η`, Ω's packed Cholesky
//!    entries, σ, ρ, the serial ρ and (FREM) θ as leaves; one single-lane sweep gives
//!    `∂Ψ` for all of them. The kernel (`focei_kernel`) is the likelihood written once,
//!    generic over the number type, and pinned to the f64 production objective.
//! 4. `∂Ψ/∂f`, `∂Ψ/∂G` seed one single-lane sweep of the vector tape back to the θ leaves.
//!
//! Its cost does not grow with the number of θ or individual parameters. IOV runs the same steps over the stacked
//! `[η, κ_0, …, κ_{K−1}]` with the block prior `Ω ⊕ (I_K ⊗ Ω_IOV)`.
//!
//! The same machinery serves FOCE, FOCEI and AGQ (both anchors; one exact-anchor node is
//! Laplace) — see [`run_agq`].
//!
//! The same recorder runs tape-free (`DualV::free`) for the inner loop's η-Jacobians,
//! inner gradients and Hessians ([`exact_obs_grads`], [`exact_inner_hessian`], …).
//!
//! Out of scope (`None`, and the caller salvages the subject by finite differences): SDE,
//! subjects with non-Gaussian records, IOV with FREM rows, engine shapes no recorder
//! serves, and more than 48 random effects (stacked under IOV).
#![allow(clippy::needless_range_loop)]

use super::*;
use crate::estimation::parameterization::{lower_tri_entries, unpack_params};
use crate::sens::focei_kernel::{foce_f, focei_f_grad, inner_grad, inner_l, LikCtx, ParT};
use crate::sens::num::vm::VmNum;
use crate::sens::num::PkNum;
use crate::sens::rtape::{rinput, rsweep, rtape_reset, DualT, R};
use crate::sens::vfree::DualF;
use crate::sens::vtape::{
    vadj_index, vadj_index_h, vadj_init, vsweep, vsweep_h, vtape_reset, DualV, NONE,
};
use crate::types::ModelParameters;
use nalgebra::{DMatrix, DVector};

/// The dual widths the recorders are instantiated at. A subject runs in the smallest
/// bucket that holds its width (`n_eta`, or the stacked `n_eta + K·n_kappa` under IOV), the
/// extra η directions inert (unit prior, read by no prediction). Every bucket is a full
/// monomorphisation of the recorder stack, so the ladder trades a little padding at
/// in-between widths for compile time and binary size; the cap is the η-Hessian payload's
/// `N(N+1)/2` per operation, not any derivative limit.
/// A model without random effects pads to 1: an inert direction with a unit prior, as
/// every padded width carries, so `Ψ = F` exactly without a width-0 copy of the recorders.
const WIDTH_BUCKETS: [usize; 11] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48];

/// The widest `[η, κ]` width a recording serves (the last dispatch bucket); wider subjects
/// go to finite differences.
pub(crate) const MAX_WIDTH: usize = WIDTH_BUCKETS[WIDTH_BUCKETS.len() - 1];

/// The likelihood-side scope: a Gaussian diagonal-residual FOCEI subject whose `Ψ` is
/// the plain form below. Engine-side scope is decided by each engine.
pub(crate) fn likelihood_supported(
    model: &CompiledModel,
    _subject: &Subject,
    _params: &ModelParameters,
) -> bool {
    model.n_eta <= WIDTH_BUCKETS[WIDTH_BUCKETS.len() - 1]
        && !model.is_sde()
        && _subject.obs_records.is_empty()
}

/// Whether the model routes to the exact outer gradient through the adjoint: a recorder
/// serves its predictions ([`recorder_in_scope`]) and its likelihood is Gaussian (a
/// non-Gaussian endpoint's subjects decline, so the outer loop stays derivative-free).
/// Individual subjects the recorders decline still fall back, per subject.
pub(crate) fn model_in_scope(model: &CompiledModel) -> bool {
    recorder_in_scope(model) && !model.has_non_gaussian()
}

/// Whether a recorder serves the model's Gaussian predictions: the model-level half of
/// [`likelihood_supported`], plus the dual ODE walk or a closed form (or compartment-free
/// readout) with a parameter program over `[η, κ]`. The inner loop's scope: a subject
/// without non-Gaussian records is served even on a joint model.
pub(crate) fn recorder_in_scope(model: &CompiledModel) -> bool {
    model.n_eta <= WIDTH_BUCKETS[WIDTH_BUCKETS.len() - 1]
        && !model.is_sde()
        && !nested_levels_outside_scope(model)
        && match model.ode_spec.as_ref() {
            _ if model.is_algebraic() => model
                .indiv_param_partials
                .indiv_param_program
                .as_ref()
                .is_some_and(|p| p.n_eta_axis() == model.n_eta + model.n_kappa),
            Some(_) => {
                crate::sens::ode_provider::ode_walk_model_scope(model)
                    && (model.n_kappa == 0 || model.frem_config.is_none())
            }
            None => {
                crate::sens::provider::closed_form_model_scope(model)
                    && crate::sens::provider::analytic_param_program(model)
                        .is_some_and(|p| p.n_eta_axis() == model.n_eta + model.n_kappa)
            }
        }
}

/// The subject objective whose gradient is wanted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Objective {
    /// FOCE (Sheiner–Beal marginal; frozen-variance EBE).
    Foce,
    /// FOCEI (interaction; conditional EBE).
    Focei,
    /// Adaptive Gauss–Hermite quadrature with `n_agq` nodes per dimension around the
    /// conditional mode, scaled by the exact Hessian (`exact`: Laplace at one node) or by
    /// FOCEI's Gauss–Newton `H̃`.
    Agq { n_agq: usize, exact: bool },
}

/// Exact per-subject FOCE/FOCEI packed gradient `dFᵢ/dx` (`pack_params` order, the same
/// quantity as `subject_packed_gradient` / `_foce`) by the adjoint method, or `None`
/// outside its scope.
#[cfg(test)]
pub(crate) fn subject_adjoint_packed_gradient(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
    interaction: bool,
) -> Option<Vec<f64>> {
    subject_adjoint_gradient(
        model,
        subject,
        template,
        x,
        eta_hat,
        if interaction {
            Objective::Focei
        } else {
            Objective::Foce
        },
    )
}

/// A subject inside the adjoint's scope: its effective model (a closed-form absorption
/// model may route to its ODE twin, exactly as the predictor resolves it), the unpacked
/// parameters, the active dual width (`n_eta`, or the stacked `n_eta + K·n_kappa` under
/// IOV), the dispatch bucket and `η̂` padded to it.
pub(super) struct Scoped<'m> {
    pub model: &'m CompiledModel,
    /// The engine production evaluates FOCE's typical-value variance `R⁰ = f(η = 0)` on: the
    /// model resolved at `η = 0`, which a closed-form absorption model near its flip-flop switch
    /// may route differently from `model` (resolved at `η̂`).
    pub pop_model: &'m CompiledModel,
    pub params: &'m ModelParameters,
    pub width: usize,
    pub bucket: usize,
    pub eta: Vec<f64>,
    /// The subject's layout under IOV, built once for every recording of the subject.
    pub layout: Option<crate::stats::stacked_layout::StackedLayout>,
}

/// A packed point unpacked once: `template`, the packed `x` and `unpack_params(x, template)`.
///
/// Every subject of a population gradient, covariance matrix or AGQ score sweep is
/// differentiated at the same packed point, so the population loops build one of these
/// before `par_iter` and hand it to the `*_at` entry points ([`subject_adjoint_gradient_at`],
/// [`subject_adjoint_cov_hessian_at`], …). Unpacking per subject rebuilt the whole
/// `ModelParameters` each time, every θ name a fresh `String`: on a 1,317-θ covariate network
/// that was a third of the gradient. The per-subject entry points without `_at` build their
/// own.
pub(crate) struct PackedPoint<'a> {
    pub(crate) template: &'a ModelParameters,
    pub(crate) x: &'a [f64],
    params: ModelParameters,
}

impl<'a> PackedPoint<'a> {
    /// `None` when `x` is not `template`'s packed length, the one shape `unpack_params`
    /// cannot read (it indexes `x` unchecked).
    pub(crate) fn new(template: &'a ModelParameters, x: &'a [f64]) -> Option<Self> {
        #[cfg(test)]
        if let Ok(mut watch) = PACKED_POINT_WATCH.lock() {
            if let Some((addr, n)) = watch.as_mut() {
                if *addr == template as *const ModelParameters as usize {
                    *n += 1;
                }
            }
        }
        let seg = crate::estimation::parameterization::packed_segments(template);
        let len = seg.theta
            + seg.omega
            + seg.sigma
            + seg.rho
            + seg.iov
            + seg.mixture_omega
            + seg.mixture_sigma;
        (x.len() == len).then(|| PackedPoint {
            template,
            x,
            params: unpack_params(x, template),
        })
    }
}

/// Test watch: `(template address, count)` — the [`PackedPoint`]s built for that one
/// template, on **any** thread (a per-subject unpack would run on the rayon workers, where a
/// thread-local counter on the caller cannot see it). Keyed by address so concurrent tests
/// with their own templates do not count.
#[cfg(test)]
pub(crate) static PACKED_POINT_WATCH: std::sync::Mutex<Option<(usize, usize)>> =
    std::sync::Mutex::new(None);

/// [`Scoped`] for a subject, or `None` outside the adjoint's scope: the shared prelude of
/// the gradient and the covariance Hessian.
pub(super) fn scope<'m>(
    model: &'m CompiledModel,
    subject: &Subject,
    pt: &'m PackedPoint<'_>,
    eta_hat: &[f64],
) -> Option<Scoped<'m>> {
    let (template, x, params) = (pt.template, pt.x, &pt.params);
    let pop_model = if model.n_kappa > 0 {
        crate::pk::effective_model_for_eval_iov(
            model,
            subject,
            &params.theta,
            &vec![0.0; eta_hat.len()],
        )
    } else {
        crate::pk::effective_model_for_eval(model, subject, &params.theta, &vec![0.0; model.n_eta])
    };
    let (model, width, eta, lay) = resolve(model, subject, &params.theta, eta_hat, None)?;
    if !likelihood_supported(model, subject, &params) {
        return None;
    }
    let seg = crate::estimation::parameterization::packed_segments(template);
    let n_iov_packed = if model.n_kappa > 0 {
        // IOV + FREM is scored by a sentinel, not the kernel's objective. A dense residual
        // (a `correlation … by` line, #1082 PR5) is: the IOV marginal is the non-IOV dense
        // one over the stacked `[η, κ]` system, on the same row-wise predicate.
        if subject.fremtype.iter().any(|&ft| ft > 0) {
            return None;
        }
        lower_tri_entries(model.n_kappa, params.omega_iov.as_ref()?.diagonal).len()
    } else {
        0
    };
    // A mixture's per-class gradient (`mixture::mixture_gradient`) arrives in the class
    // layout — the class's Ω and σ in the base slots, the class read under the caller's
    // `MixtureClassGuard` — so its override tail is inert here and gets zeros.
    if seg.iov != n_iov_packed
        || seg.omega != lower_tri_entries(model.n_eta, params.omega.diagonal).len()
        || x.len()
            != seg.theta
                + seg.omega
                + seg.sigma
                + seg.rho
                + seg.iov
                + seg.mixture_omega
                + seg.mixture_sigma
    {
        return None;
    }
    // `eta` is in the smallest bucket that holds the width, padded with inert directions.
    Some(Scoped {
        model,
        pop_model,
        params,
        width,
        bucket: eta.len(),
        eta,
        layout: lay.map(std::borrow::Cow::into_owned),
    })
}

/// `$f::<N, N(N+1)/2>(args)` for a dispatch bucket ([`WIDTH_BUCKETS`]); `None` for any
/// other.
macro_rules! by_bucket {
    ($w:expr, $f:ident, ($($arg:expr),*)) => {
        match $w {
            1 => $f::<1, 1>($($arg),*),
            2 => $f::<2, 3>($($arg),*),
            3 => $f::<3, 6>($($arg),*),
            4 => $f::<4, 10>($($arg),*),
            6 => $f::<6, 21>($($arg),*),
            8 => $f::<8, 36>($($arg),*),
            12 => $f::<12, 78>($($arg),*),
            16 => $f::<16, 136>($($arg),*),
            24 => $f::<24, 300>($($arg),*),
            32 => $f::<32, 528>($($arg),*),
            48 => $f::<48, 1176>($($arg),*),
            _ => None,
        }
    };
}

/// `subject_adjoint_packed_gradient` for any [`Objective`].
#[cfg(test)]
pub(crate) fn subject_adjoint_gradient(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
    objective: Objective,
) -> Option<Vec<f64>> {
    let pt = PackedPoint::new(template, x)?;
    subject_adjoint_gradient_at(model, subject, &pt, eta_hat, objective)
}

/// `subject_adjoint_gradient` (the per-subject entry point, which unpacks its own `x`) at a [`PackedPoint`] the caller unpacked once for its
/// population.
pub(crate) fn subject_adjoint_gradient_at(
    model: &CompiledModel,
    subject: &Subject,
    pt: &PackedPoint<'_>,
    eta_hat: &[f64],
    objective: Objective,
) -> Option<Vec<f64>> {
    adjoint_gradient(model, subject, pt, eta_hat, objective, false)
}

/// The **held-EBE** quadrature score: `∂Aᵢ/∂x` at a fixed `η̂`, for an AGQ subject whose mode
/// identity [`subject_adjoint_gradient_at`] declines — an indefinite exact `H` at an `η̂` the
/// inner search's fallback left short of stationary ([`mode_solve`]). Without the identity
/// the EBE response is not available exactly, and the alternative is the reconverged
/// finite-difference score, `2·n_x` EBE re-solves per subject: on a 147-weight covariate
/// network under Laplace that turned a 7.5 s fit into a 90 s one. This is the AGQ twin of
/// FOCE/FOCEI's held-EBE salvage, omits the same EBE-response term, and is reported by the
/// same outer FD-fallback warning. `None` wherever [`subject_adjoint_gradient_at`] declines for
/// any other reason (scope, a non-finite recording) and for the FOCE/FOCEI objectives,
/// whose salvage lives in the outer optimizer.
#[cfg(test)]
pub(crate) fn subject_adjoint_held_ebe_score(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
    objective: Objective,
) -> Option<Vec<f64>> {
    let pt = PackedPoint::new(template, x)?;
    subject_adjoint_held_ebe_score_at(model, subject, &pt, eta_hat, objective)
}

/// `subject_adjoint_held_ebe_score` (the per-subject entry point, which unpacks its own `x`) at a [`PackedPoint`].
pub(crate) fn subject_adjoint_held_ebe_score_at(
    model: &CompiledModel,
    subject: &Subject,
    pt: &PackedPoint<'_>,
    eta_hat: &[f64],
    objective: Objective,
) -> Option<Vec<f64>> {
    matches!(objective, Objective::Agq { .. })
        .then(|| adjoint_gradient(model, subject, pt, eta_hat, objective, true))?
}

fn adjoint_gradient(
    model: &CompiledModel,
    subject: &Subject,
    pt: &PackedPoint<'_>,
    eta_hat: &[f64],
    objective: Objective,
    held: bool,
) -> Option<Vec<f64>> {
    let (template, x) = (pt.template, pt.x);
    let mut sc = scope(model, subject, pt, eta_hat)?;
    let objective = match objective {
        // No random effects: both anchors are the empty `H`, and `Ψ = F`.
        Objective::Agq { n_agq, exact: true } if sc.width == 0 => Objective::Agq {
            n_agq,
            exact: false,
        },
        // The exact anchor's recording runs at its own buckets: pad to one.
        Objective::Agq { exact: true, .. } => {
            sc.bucket = EXACT_ANCHOR_BUCKETS.into_iter().find(|&b| b >= sc.width)?;
            sc.eta.resize(sc.bucket, 0.0);
            objective
        }
        objective => objective,
    };
    fn go<const N: usize, const M: usize>(
        sc: &Scoped,
        subject: &Subject,
        template: &ModelParameters,
        x: &[f64],
        objective: Objective,
        held: bool,
    ) -> Option<Vec<f64>>
    where
        Bucket<N, M>: ExactAnchor<N, M>,
    {
        match objective {
            Objective::Agq { n_agq, exact } => run_agq::<N, M>(
                sc.model,
                subject,
                sc.layout.as_ref(),
                template,
                sc.params,
                x,
                &sc.eta,
                n_agq,
                exact,
                sc.width,
                held,
            ),
            _ => run::<N, M>(
                sc.model,
                sc.pop_model,
                subject,
                sc.layout.as_ref(),
                template,
                sc.params,
                x,
                &sc.eta,
                objective != Objective::Foce,
                sc.width,
            ),
        }
    }
    by_bucket!(sc.bucket, go, (&sc, subject, template, x, objective, held))
}

/// `H⁻¹b` for the mode system `H w = ∇_η F` (`H` the exact inner Hessian at `η̂`): a
/// Cholesky solve at a strict mode, else a pivoted LU. The identity `Ψ = F − wᵀ∇_η l` needs
/// `η̂` stationary and `H` nonsingular, not definite, and an EBE that the inner search's
/// fallback left at a stationary point whose exact `H` is indefinite is still one: declining
/// it sent the subject to the reconverged finite-difference score, `2·n_x` EBE re-solves (on
/// a 141-weight covariate network under Laplace, 43 such subject-iterations were most of the
/// fit). `None` when `H` is singular or the solve is not finite.
///
/// The indefinite branch also needs the stationarity the identity rests on: there `η̂` came
/// from the inner search's fallback, which can stop short (Nelder–Mead, max-iter), and the
/// gradient `wᵀ∇_η l` drops is then not zero. `grad_l` is `∇_η l(η̂)`; above
/// [`MODE_STATIONARY_TOL`] the subject declines (#2093 review) — under AGQ to the held-EBE
/// score ([`subject_adjoint_held_ebe_score_at`]), not the reconverged one.
///
/// The SPD branch takes no such gate, deliberately. Near a strict mode `η̂ = η* + ε` and
/// `∂Ψ/∂x` misses the converged objective's total derivative by `O(|∇_η l|)` only, while
/// every salvage a decline could take drops the EBE response, an `O(1)` error: measured off
/// a Laplace mode, 2.1e-3 vs 3.0 at `|ε| ≈ 1e-3` and 2.1e-2 vs 2.6 at 1e-2
/// (`off_the_mode_the_identity_beats_the_held_ebe_gradient`). An indefinite `H` off the mode
/// is not near a minimum at all, which is what the gate above refuses.
fn mode_solve(h: &DMatrix<f64>, b: &DVector<f64>, grad_l: &[f64]) -> Option<DVector<f64>> {
    if let Some(c) = h.clone().cholesky() {
        return Some(c.solve(b));
    }
    if !(crate::sens::num::nan_max_abs(grad_l.iter().copied()) <= MODE_STATIONARY_TOL) {
        return None;
    }
    let w = h.clone().full_piv_lu().solve(b)?;
    w.iter().all(|v| v.is_finite()).then_some(w)
}

/// The largest `|∇_η l(η̂)|` [`mode_solve`] treats as a stationary point on its indefinite
/// branch: ten times the inner search's default gradient tolerance (`inner_tol = 1e-5`),
/// which a certified fallback end point meets.
const MODE_STATIONARY_TOL: f64 = 1e-4;

/// The θ leaves of a recording (value leaves, no tangent) for each θ the parameter
/// program reads; shared by every recording on one tape.
fn theta_leaves<const N: usize, const M: usize, const TH: bool>(
    n_theta_axis: usize,
    theta: &[f64],
) -> Vec<DualV<N, M, TH>> {
    theta
        .iter()
        .enumerate()
        .map(|(m, &v)| {
            if m < n_theta_axis {
                DualV::leaf(v, [0.0; N], [0.0; M])
            } else {
                DualV::constant(v)
            }
        })
        .collect()
}

/// The identity directions: tangent `k` is `∂/∂η_k`.
fn unit_dirs<const N: usize>() -> [[f64; N]; N] {
    std::array::from_fn(|k| std::array::from_fn(|m| if m == k { 1.0 } else { 0.0 }))
}

/// The θ leaves of one recording and, when the weights reach the likelihood through the
/// networks' outputs alone ([`hessian::NnReduction`]), those outputs on the **same** number
/// type: the recording reads them in place of evaluating the networks over the weights. The
/// type is the guarantee — a recording can only be handed outputs of its own type, so there
/// is no thread-local channel to miss or to nest (#2293 review).
pub(super) struct ThetaLeaves<'a, T> {
    pub(super) theta: &'a [T],
    #[cfg(feature = "nn")]
    pub(super) nn: Option<&'a [Vec<T>]>,
}

impl<'a, T> ThetaLeaves<'a, T> {
    /// θ leaves alone: every network is evaluated on the recording's number type.
    pub(super) fn theta(theta: &'a [T]) -> Self {
        ThetaLeaves {
            theta,
            #[cfg(feature = "nn")]
            nn: None,
        }
    }

    /// θ leaves with the networks' outputs (`None`: evaluate the networks).
    #[cfg(feature = "nn")]
    pub(super) fn with_nn(theta: &'a [T], nn: Option<&'a [Vec<T>]>) -> Self {
        ThetaLeaves { theta, nn }
    }
}

impl<T> std::ops::Deref for ThetaLeaves<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.theta
    }
}

/// The [`hessian::NnReduction`] of `subject` at `theta` and its outputs as leaves on the
/// outer tape (one value leaf per output, no η-tangent: the inputs are subject-static, so
/// the outputs do not move with η), handed to the recordings in their [`ThetaLeaves`].
/// `None` when the reduction does not hold, and the networks are taped over their weights
/// as before.
#[cfg(feature = "nn")]
#[allow(clippy::type_complexity)]
fn nn_output_leaves<'s, const N: usize, const M: usize, const TH: bool>(
    model: &CompiledModel,
    subject: &'s Subject,
    template: &ModelParameters,
    theta: &[f64],
) -> Option<(hessian::NnReduction<'s>, Vec<Vec<DualV<N, M, TH>>>)> {
    #[cfg(test)]
    if NN_OUTPUT_LEAVES_OFF.with(std::cell::Cell::get) {
        return None;
    }
    let red = hessian::NnReduction::build(model, subject, template, theta)?;
    #[cfg(test)]
    NN_OUTPUT_LEAF_HITS.with(|c| c.set(c.get() + 1));
    let z = red
        .z
        .iter()
        .map(|net| {
            net.iter()
                .map(|&v| DualV::leaf(v, [0.0; N], [0.0; M]))
                .collect()
        })
        .collect();
    Some((red, z))
}

/// The weights' share of the θ adjoint under [`nn_output_leaves`]:
/// `Σ_k (∂Ψ/∂z_k)·∂z_k/∂w`, each network's output adjoints (`at`, read off the sweep)
/// pulled back through it in one backward pass. `None` when a network refuses the product.
#[cfg(feature = "nn")]
fn add_nn_weight_adjoint<L>(
    model: &CompiledModel,
    red: &hessian::NnReduction,
    leaves: &[Vec<L>],
    theta: &[f64],
    at: impl Fn(&L) -> f64,
    theta_adj: &mut [f64],
) -> Option<()> {
    for (net, ((&(off, n_w, _), fwd), zl)) in
        red.nets.iter().zip(&red.forward).zip(leaves).enumerate()
    {
        let zbar: Vec<f64> = zl.iter().map(&at).collect();
        let gw =
            model.covariate_nns[net]
                .mapper
                .mlp()
                .vjp_at(fwd, &theta[off..off + n_w], &zbar)?;
        for (t, g) in theta_adj[off..off + n_w].iter_mut().zip(gw) {
            *t += g;
        }
    }
    Some(())
}

#[cfg(all(test, feature = "nn"))]
thread_local! {
    /// Test switch: tape every network over its weights in the outer gradient.
    pub(crate) static NN_OUTPUT_LEAVES_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Outer-gradient recordings on this thread that read the networks' outputs as leaves.
    pub(crate) static NN_OUTPUT_LEAF_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The individual parameters over the tape, placed in their PK slots (`None` where the
/// program writes no slot): the parameter program itself evaluated on the recording's number type, so its
/// η-tangents are `∂p/∂η`, its payload `∂²p/∂η²`, and its θ-dependence is on the tape.
fn taped_params<T: VmNum>(
    model: &CompiledModel,
    prog: &crate::parser::model_parser::IndivParamProgram,
    n_slots: usize,
    theta: &ThetaLeaves<'_, T>,
    eta: &[T],
    cov: &std::collections::HashMap<String, f64>,
) -> Vec<Option<T>> {
    // `[covariate_nn]` outputs on `T`, handed to the program as an argument: the ones the
    // recording was given (`ThetaLeaves::nn`), or the networks over the θ (weight) inputs.
    #[cfg(feature = "nn")]
    let computed: Vec<Vec<T>>;
    #[cfg(feature = "nn")]
    let nn: &[Vec<T>] = match theta.nn {
        Some(z) => z,
        None if model.covariate_nns.is_empty() => &[],
        None => {
            computed = model
                .covariate_nns
                .iter()
                .map(|nn| {
                    let w =
                        &theta[nn.weights_offset..nn.weights_offset + nn.mapper.mlp().n_weights()];
                    // Weights with no derivative (every inner recording): the network on
                    // `f64` — memoised across the solve — lifted, not on the dual.
                    if w.iter().all(|t| t.is_constant()) {
                        let wf: Vec<f64> = w.iter().map(|t| t.val()).collect();
                        if let Ok(y) = nn.mapper.forward_raw(&wf, cov) {
                            return y.into_iter().map(T::from_f64).collect();
                        }
                    }
                    nn.mapper.forward_raw_g(w, cov)
                })
                .collect();
            &computed
        }
    };
    #[cfg(not(feature = "nn"))]
    let nn: &[Vec<T>] = {
        let _ = model;
        &[]
    };
    let vals = prog.eval_params_g(theta, eta, cov, nn);
    let mut by_slot = vec![None; n_slots];
    for (&(slot, _), v) in prog.pk_var_slots().iter().zip(vals) {
        if slot < n_slots {
            by_slot[slot] = Some(v);
        }
    }
    by_slot
}

/// An expression `obs_scale` divisor `s(θ, η, p)`, applied after the walk as the
/// providers do: the scale program evaluated on the tape over the (subject-static)
/// taped parameters, and every prediction becomes `pred / s`. `None` for a scale this
/// cannot serve (a per-CMT scale, or LTBS on top of it).
fn apply_expression_scale<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[T],
    eta: &[T],
    by_slot: &[Option<T>],
    mut preds: Vec<T>,
) -> Option<Vec<T>> {
    match &model.scaling {
        ScalingSpec::ExpressionScale {
            deriv: Some(prog), ..
        } => {
            let vars: Vec<T> = prog
                .var_to_pk_slot()
                .iter()
                .map(|&sl| {
                    by_slot
                        .get(sl)
                        .copied()
                        .flatten()
                        .unwrap_or(T::from_f64(0.0))
                })
                .collect();
            let s = prog.eval_scale_g(theta, eta, &subject.covariates, &vars);
            if !(s.val() > 0.0 && s.val().is_finite()) {
                return None;
            }
            for y in preds.iter_mut() {
                *y = *y / s;
                // Production's order: the scale, then the LTBS log (`apply_output_transform`
                // left it for here).
                if model.log_transform {
                    *y = crate::sens::ode_provider::ltbs_readout(*y);
                }
            }
            Some(preds)
        }
        ScalingSpec::ExpressionScale { .. } | ScalingSpec::PerCmt(_) => None,
        _ => Some(preds),
    }
}

/// ODE engine: the static superposition walk, or (time-varying covariates, lag times,
/// steady state, modeled doses, `TIME`, resets) the event-driven walk, with `init(...)`
/// on either through the shared Taylor seed.
fn record_ode<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    th: &ThetaLeaves<'_, T>,
    et: &[T],
) -> Option<Vec<T>> {
    let ode = model.ode_spec.as_ref()?;
    let prog = ode.indiv_param_program.as_ref()?;
    if static_walk_subject_supported(model, subject) {
        let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
        let by_slot = taped_params(model, prog, pk.values.len(), th, et, &subject.covariates);
        let params_dual: Vec<T> = by_slot
            .iter()
            .zip(&pk.values)
            .map(|(d, &v)| d.unwrap_or(T::from_f64(v)))
            .collect();
        let init_state = init_seed::<T>(ode, &params_dual, &model.pk_indices)
            .unwrap_or_else(|| vec![T::from_f64(0.0); ode.n_states]);
        let preds = integrate_seeded_duals(model, subject, &pk.values, &params_dual, &init_state)?;
        return apply_expression_scale(model, subject, th, et, &by_slot, preds);
    }
    if !event_walk_subject_supported(model, subject) {
        return None;
    }
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let (pk_at_dose, pk_at_obs, pk_at_pk_only, pk_at_reset) =
        seed_tvcov_snapshots::<T>(subject, uses_time, |cov, time| {
            // One taped parameter set per distinct covariate snapshot (the seeder caches by key).
            let _time_guard =
                crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
            let pk = (model.pk_param_fn)(theta, eta, cov, time);
            let by_slot = taped_params(model, prog, pk.values.len(), th, et, cov);
            Some(
                by_slot
                    .iter()
                    .zip(&pk.values)
                    .map(|(d, &v)| d.unwrap_or(T::from_f64(v)))
                    .collect(),
            )
        })?;
    let preds = integrate_tvcov_readout::<T>(
        model,
        subject,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
        &pk_at_reset,
    );
    if !matches!(
        model.scaling,
        ScalingSpec::ExpressionScale { .. } | ScalingSpec::PerCmt(_)
    ) {
        return Some(preds);
    }
    // The scale is subject-static even under TV covariates (production's `apply_scaling`
    // reads `subject.covariates`): its parameters are at the subject covariates.
    let n_slots = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0)
        .values
        .len();
    let by_slot = taped_params(model, prog, n_slots, th, et, &subject.covariates);
    apply_expression_scale(model, subject, th, et, &by_slot, preds)
}

/// Closed-form engine: static superposition (lag, SS, readout, scalar scale, LTBS).
fn record_analytic<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    th: &ThetaLeaves<'_, T>,
    et: &[T],
) -> Option<Vec<T>> {
    if !closed_form_serves(model, subject) || closed_form_reshapes_f_window(model, subject) {
        return None;
    }
    let prog = crate::sens::provider::analytic_param_program(model)?;
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let by_slot = taped_params(model, prog, pk.values.len(), th, et, &subject.covariates);
    let preds = match pk_space_predictions(model, subject, &pk, &by_slot) {
        Some(preds) => preds,
        None => crate::sens::provider::analytic_static_predictions(
            model,
            subject,
            &pk,
            |slot| by_slot.get(slot).copied().flatten(),
            th,
            et,
        )?,
    };
    closed_form_finish(model, subject, theta, eta, th, et, prog, preds)
}

#[cfg(test)]
thread_local! {
    /// Test switch: take the generic closed form even where the PK-space path applies, for
    /// the twin that pins the two against each other.
    pub(super) static PK_SPACE_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Recordings the PK-space path served, so a test can see it ran.
    pub(super) static PK_SPACE_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The static closed form in **PK space**: each observation's concentration as a function
/// of the subject's `P` live PK parameters `p` (the slots `by_slot` tapes), over a tape-free
/// `P`-direction second-order dual, then lifted onto `T` through its second-order Taylor
/// polynomial in `δp = p − p₀`, recorded as one node per observation.
///
/// The static subject's parameters are one set for every observation, so `f_j(η, θ) =
/// c_j(p(η, θ))`; the polynomial reproduces every jet of `c_j ∘ p` of total order ≤ 2 —
/// value, `∂/∂η`, `∂²/∂η²`, and the tape's first θ-derivative of the value and η-tangents —
/// which is everything a [`PkNum::TAYLOR2_EXACT`] type carries. The generic path instead
/// runs the closed form itself over `T`: every operation carries the η-Hessian payload and,
/// where it meets θ, a tape node, so its cost grows with `n_η`; this one's grows with `P`
/// and writes `n_obs` nodes (#2093 review: the forward provider's PK-space kernels).
///
/// `None` — the caller takes the generic path — for a type that is not
/// [`PkNum::TAYLOR2_EXACT`] (the exact anchor's taped Hessian, the covariance step's nested
/// duals), for more live slots than the widest small bucket, when an analytic `init(...)`
/// reads θ and η directly, or outside the static path's scope.
fn pk_space_predictions<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    pk: &crate::types::PkParams,
    by_slot: &[Option<T>],
) -> Option<Vec<T>> {
    #[cfg(test)]
    if PK_SPACE_OFF.with(|c| c.get()) {
        return None;
    }
    if !T::TAYLOR2_EXACT || !model.analytical_init.is_empty() {
        return None;
    }
    let live: Vec<(usize, T)> = by_slot
        .iter()
        .enumerate()
        .filter_map(|(s, d)| d.map(|d| (s, d)))
        .collect();
    // The polynomial's order is the jet's: a tape-free first-order recording (the inner
    // loop's η-gradient) needs `∂c/∂p` alone; the η-Hessian payload, or a tape that
    // differentiates the η-tangents once more in θ, needs `∂²c/∂p²` too.
    let second = T::ETA_HESSIAN || live.iter().any(|&(_, d)| d.is_taped());
    // A tape-free first-order recording (the inner loop's η-gradient) records nothing either
    // way, and runs the closed form over about as many lanes: there the generic path is the
    // cheaper one.
    if !second {
        return None;
    }
    fn go<const P: usize, const MP: usize, T: VmNum>(
        model: &CompiledModel,
        subject: &Subject,
        pk: &crate::types::PkParams,
        live: &[(usize, T)],
    ) -> Option<Vec<T>> {
        let unit = unit_dirs::<P>();
        let jets = crate::sens::provider::analytic_static_predictions::<DualV<P, MP>>(
            model,
            subject,
            pk,
            |slot| {
                let i = live.iter().position(|&(s, _)| s == slot)?;
                Some(DualV::free(live[i].1.val(), unit[i], [0.0; MP]))
            },
            &[],
            &[],
        )?;
        // The polynomial's basis, once per subject: `δp_i`, then (second order) `δp_i·δp_j`
        // (i ≥ j), each with value 0 and the full jet of its product.
        let n = live.len();
        let dp: Vec<T> = live
            .iter()
            .map(|&(_, d)| d - T::from_f64(d.val()))
            .collect();
        let mut basis = dp.clone();
        if MP > 0 {
            for i in 0..n {
                for j in 0..=i {
                    basis.push(dp[i] * dp[j]);
                }
            }
        }
        let mut coef = vec![0.0; basis.len()];
        jets.iter()
            .map(|q| {
                coef[..n].copy_from_slice(&q.g[..n]);
                if MP > 0 {
                    let mut t = n;
                    for i in 0..n {
                        for j in 0..=i {
                            let hij = q.h[crate::sens::vtape::tri(i, j)];
                            coef[t] = if i == j { 0.5 * hij } else { hij };
                            t += 1;
                        }
                    }
                }
                Some(T::lincomb(T::from_f64(q.v), 1.0, &coef, &basis))
            })
            .collect()
    }
    #[cfg(test)]
    PK_SPACE_HITS.with(|c| c.set(c.get() + 1));
    // Second order only: the `!second` return above already sent a first-order recording
    // to the generic path, so no `go::<P, 0>` is instantiated.
    match live.len() {
        1 => go::<1, 1, T>(model, subject, pk, &live),
        2 => go::<2, 3, T>(model, subject, pk, &live),
        3 => go::<3, 6, T>(model, subject, pk, &live),
        4 => go::<4, 10, T>(model, subject, pk, &live),
        5 | 6 => go::<6, 21, T>(model, subject, pk, &live),
        7 | 8 => go::<8, 36, T>(model, subject, pk, &live),
        _ => None,
    }
}

/// ODE IOV engine: the event walk fed per-occasion-group parameter sets
/// (`seed_iov_events`, as `predict_iov` does), each the parameter program evaluated on
/// the combined `[η_bsv, κ_g]` leaves of its group (κ held at zero for an ungrouped
/// pk-only row). `eta`/`et` are the stacked `[η_bsv, κ_0, …, κ_{K−1}]`.
fn record_ode_iov<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    layout: &crate::stats::stacked_layout::StackedLayout,
    theta: &[f64],
    eta: &[f64],
    th: &ThetaLeaves<'_, T>,
    et: &[T],
) -> Option<Vec<T>> {
    let ode = model.ode_spec.as_ref()?;
    let prog = ode.indiv_param_program.as_ref()?;
    ode_iov_walk_scope(model, subject, layout)?;
    if layout.n_stacked() > et.len() {
        return None;
    }
    let n_eta = model.n_eta;
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let combined = |g: Option<usize>| -> (Vec<f64>, Vec<T>) {
        let mut d: Vec<T> = et[..n_eta].to_vec();
        match g {
            Some(g) => d.extend_from_slice(&et[layout.kappa_range(g)]),
            None => d.extend(std::iter::repeat_n(T::from_f64(0.0), model.n_kappa)),
        }
        let v = match g {
            Some(g) => layout.combined_effect(eta, g),
            None => layout.combined_pk_only(eta),
        };
        (v, d)
    };
    let seed = |g: Option<usize>,
                cov: &std::collections::HashMap<String, f64>,
                time: f64|
     -> Option<Vec<T>> {
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
        let (v, d) = combined(g);
        let pk = (model.pk_param_fn)(theta, &v, cov, time);
        let by_slot = taped_params(model, prog, pk.values.len(), th, &d, cov);
        Some(
            by_slot
                .iter()
                .zip(&pk.values)
                .map(|(d, &v)| d.unwrap_or(T::from_f64(v)))
                .collect(),
        )
    };
    let cov = &subject.covariates;
    let k = layout.n_groups();
    let static_groups =
        || -> Option<Vec<Vec<T>>> { (0..k).map(|g| seed(Some(g), cov, 0.0)).collect() };
    let per_event = iov_walk_per_event(model, subject);
    let shared = if per_event {
        None
    } else {
        Some(static_groups()?)
    };
    let (pk_at_dose, pk_at_obs, pk_at_pk_only, pk_at_reset) = seed_iov_events::<T>(
        subject,
        layout.occ_to_group(),
        k,
        per_event,
        cov,
        shared.as_deref(),
        |g, c, t| seed(Some(g), c, t),
        |c, t| seed(None, c, t),
    )?;
    let mut preds = integrate_tvcov_readout::<T>(
        model,
        subject,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
        &pk_at_reset,
    );
    match &model.scaling {
        // One scale per occasion group at the subject covariates (`predict_iov`'s
        // `apply_scaling`), applied to that group's observations.
        ScalingSpec::ExpressionScale {
            deriv: Some(sprog), ..
        } => {
            let groups = match shared {
                Some(g) => g,
                None => static_groups()?,
            };
            let scales: Vec<T> = groups
                .iter()
                .map(|params| {
                    let vars: Vec<T> = sprog
                        .var_to_pk_slot()
                        .iter()
                        .map(|&sl| params.get(sl).copied().unwrap_or(T::from_f64(0.0)))
                        .collect();
                    sprog.eval_scale_g(th, &et[..n_eta], cov, &vars)
                })
                .collect();
            for (j, y) in preds.iter_mut().enumerate() {
                let s = scales[layout.group_of(*subject.occasions.get(j)?)?];
                if !(s.val() > 0.0 && s.val().is_finite()) {
                    return None;
                }
                *y = *y / s;
                if model.log_transform {
                    *y = crate::sens::ode_provider::ltbs_readout(*y);
                }
            }
        }
        ScalingSpec::ExpressionScale { .. } | ScalingSpec::PerCmt(_) => return None,
        _ => {}
    }
    Some(preds)
}

/// A rate-defined infusion under an estimated bioavailability, which production
/// *reshapes* (#419: the rate held, the window stretched to `F·dur`). The superposition
/// kernels take each window as a fixed f64, so these subjects take the closed-form event
/// walk, whose dose resolution carries the stretched window as a moving boundary.
fn closed_form_reshapes_f_window(model: &CompiledModel, subject: &Subject) -> bool {
    model.has_bioavailability() && subject.has_rate_defined_infusion()
}

/// What either closed-form recorder needs from the model and subject, beyond its own
/// engine: the closed form's model scope (`closed_form_model_scope`), and a subject it
/// predicts at all (`subject_feeds_analytical_pk`).
fn closed_form_serves(model: &CompiledModel, subject: &Subject) -> bool {
    crate::sens::provider::closed_form_model_scope(model)
        && crate::pk::subject_feeds_analytical_pk(model, subject)
}

/// The closed-form output stage on the tape, as production's `apply_scaling` then
/// `apply_log_transform`: each observation divided by its scale — none, a constant, an
/// expression of the subject-static `t = 0` parameters, or per compartment any of those
/// — then the LTBS log. `None` for an invalid scale or one with no compiled program.
#[allow(clippy::too_many_arguments)]
fn closed_form_finish<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    th: &ThetaLeaves<'_, T>,
    et: &[T],
    prog: &crate::parser::model_parser::IndivParamProgram,
    mut preds: Vec<T>,
) -> Option<Vec<T>> {
    let n_eta = model.n_eta;
    let mut static_slots: Option<Vec<Option<T>>> = None;
    let mut scale_of = |spec: &ScalingSpec| -> Option<Option<T>> {
        Some(match spec {
            ScalingSpec::None => None,
            ScalingSpec::ScalarScale(k) => Some(T::from_f64(*k)),
            ScalingSpec::ExpressionScale {
                deriv: Some(sp), ..
            } => {
                let slots = static_slots.get_or_insert_with(|| {
                    let n = (model.pk_param_fn)(theta, &eta[..n_eta], &subject.covariates, 0.0)
                        .values
                        .len();
                    taped_params(model, prog, n, th, &et[..], &subject.covariates)
                });
                let vars: Vec<T> = sp
                    .var_to_pk_slot()
                    .iter()
                    .map(|&sl| slots.get(sl).copied().flatten().unwrap_or(T::from_f64(0.0)))
                    .collect();
                Some(sp.eval_scale_g(th, &et[..n_eta], &subject.covariates, &vars))
            }
            _ => return None,
        })
    };
    let scales: Vec<Option<T>> = match &model.scaling {
        ScalingSpec::PerCmt(map) => subject
            .obs_cmts
            .iter()
            .map(|c| map.get(c).map_or(Some(None), &mut scale_of))
            .collect::<Option<_>>()?,
        spec => vec![scale_of(spec)?; preds.len()],
    };
    for (y, s) in preds.iter_mut().zip(scales) {
        if let Some(s) = s {
            if !(s.val() > 0.0 && s.val().is_finite()) {
                return None;
            }
            *y = *y / s;
        }
    }
    if model.log_transform {
        preds
            .iter_mut()
            .for_each(|y| *y = crate::pk::ltbs_log_g(*y));
    }
    Some(preds)
}

/// Closed-form event engine: the event-driven walk (`closed_form_event_predictions`) fed
/// each dose, observation and pk-only row its own parameter set — its covariate snapshot
/// and time, and under IOV its occasion's `[η, κ_g]` — exactly as `fill_record_params` /
/// `fill_iov_record_params` build them. Serves time-varying covariates, `TIME`, oral
/// infusion and closed-form IOV, with analytic `init(...)` impulses, a Form C readout
/// (at each observation's own parameters, as production evaluates it under IOV too) and
/// output scaling (per occasion group under IOV).
fn record_analytic_events<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
    th: &ThetaLeaves<'_, T>,
    et: &[T],
) -> Option<Vec<T>> {
    use crate::sens::propagate::PkDual;
    use crate::sens::provider as pv;
    use crate::types::{
        PK_IDX_CL, PK_IDX_F, PK_IDX_KA, PK_IDX_Q, PK_IDX_Q3, PK_IDX_V, PK_IDX_V2, PK_IDX_V3,
    };
    let iov = model.n_kappa > 0;
    // The program must cover every slot the closed form reads, and read the combined
    // `[η, κ]` the records hand it.
    let prog = pv::analytic_param_program(model)
        .filter(|p| p.n_eta_axis() == model.n_eta + model.n_kappa)?;
    // Only what the shared walk or production semantics require — the forward provider's
    // dispatch-width, readout-slot and IOV gates do not apply to a recorder that hands the
    // walk taped parameters: a compartment-free model has no walk; the walk lacks the
    // pre-arrival SS overlay under a lag; a modeled dose needs its slot.
    if model.is_algebraic()
        || !closed_form_serves(model, subject)
        || pv::ss_lagtime_walk_unsupported(model, subject)
        || !pv::modeled_dose_analytic_gate(model, subject)
    {
        return None;
    }
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let n_eta = model.n_eta;
    let layout = if iov { Some(lay?) } else { None };
    // The `[η, κ_g]` a record of occasion `occ` reads (κ = 0 for an ungrouped occasion):
    // values for `pk_param_fn`, tape leaves for the program.
    let combined = |occ: u32| -> (Vec<f64>, Vec<T>) {
        let Some(l) = layout else {
            return (eta.to_vec(), et.to_vec());
        };
        let mut v = eta[..n_eta].to_vec();
        let mut d = et[..n_eta].to_vec();
        match l.group_of(occ) {
            Some(g) => {
                v.extend_from_slice(&eta[l.kappa_range(g)]);
                d.extend_from_slice(&et[l.kappa_range(g)]);
            }
            None => {
                v.extend(std::iter::repeat_n(0.0, model.n_kappa));
                d.extend(std::iter::repeat_n(T::from_f64(0.0), model.n_kappa));
            }
        }
        (v, d)
    };
    let at = |occ: u32,
              cov: &std::collections::HashMap<String, f64>,
              time: f64|
     -> (crate::types::PkParams, Vec<T>) {
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
        let (v, d) = combined(occ);
        let pk = (model.pk_param_fn)(theta, &v, cov, time);
        let by_slot = taped_params(model, prog, pk.values.len(), th, &d, cov);
        let full = by_slot
            .iter()
            .zip(&pk.values)
            .map(|(d, &v)| d.unwrap_or(T::from_f64(v)))
            .collect();
        (pk, full)
    };
    // With subject-static covariates and no `TIME` read, every record of one occasion
    // group reads the same parameters: evaluate them once per group (the per-group
    // sources `predict_iov` and the forward provider share) rather than per record.
    let per_record = subject.has_tv_covariates()
        || !subject.pk_only_covariates.is_empty()
        || !subject.reset_covariates.is_empty()
        || uses_time;
    let mut shared: std::collections::HashMap<Option<usize>, (crate::types::PkParams, Vec<T>)> =
        std::collections::HashMap::new();
    let mut at_record = |occ: u32,
                         cov: &std::collections::HashMap<String, f64>,
                         time: f64|
     -> (crate::types::PkParams, Vec<T>) {
        if per_record {
            return at(occ, cov, time);
        }
        let key = layout.and_then(|l| l.group_of(occ));
        shared
            .entry(key)
            .or_insert_with(|| at(occ, cov, time))
            .clone()
    };
    let doses: Vec<_> = (0..subject.doses.len())
        .map(|k| {
            at_record(
                crate::pk::dose_row_occasion(subject, k),
                subject.dose_cov(k),
                subject.param_time_at(subject.doses[k].time),
            )
        })
        .collect();
    let obs: Vec<_> = (0..subject.obs_times.len())
        .map(|j| {
            at_record(
                subject.occasions.get(j).copied().unwrap_or(0),
                subject.obs_cov(j),
                subject.param_time_obs(j),
            )
        })
        .collect();
    let pk_only: Vec<_> = (0..subject.pk_only_times.len())
        .map(|m| {
            at_record(
                crate::pk::pk_only_row_occasion(subject, m).unwrap_or(u32::MAX),
                subject.pk_only_cov(m),
                subject.param_time_pk_only(m),
            )
        })
        .collect();
    let dual = |p: &[T]| PkDual {
        cl: p[PK_IDX_CL],
        v: p[PK_IDX_V],
        q: p[PK_IDX_Q],
        v2: p[PK_IDX_V2],
        ka: p[PK_IDX_KA],
        q3: p[PK_IDX_Q3],
        v3: p[PK_IDX_V3],
        f: p[PK_IDX_F],
    };
    let dose_pk: Vec<crate::types::PkParams> = doses.iter().map(|(pk, _)| *pk).collect();
    let at_dose: Vec<_> = doses.iter().map(|(_, p)| dual(p)).collect();
    let at_obs: Vec<_> = obs.iter().map(|(_, p)| dual(p)).collect();
    let at_pk_only: Vec<_> = pk_only.iter().map(|(_, p)| dual(p)).collect();
    let slot_dual =
        |k: usize, slot: usize| doses[k].1.get(slot).copied().unwrap_or(T::from_f64(0.0));
    let obs_params = |j: usize| -> [T; crate::types::MAX_PK_PARAMS] {
        std::array::from_fn(|s| obs[j].1.get(s).copied().unwrap_or(T::from_f64(0.0)))
    };
    let readout = model
        .analytic_readout
        .as_ref()
        .and_then(|ar| ar.program.as_ref());
    let mut preds = pv::closed_form_event_predictions(
        model,
        subject,
        &dose_pk,
        &at_dose,
        &at_obs,
        &at_pk_only,
        &slot_dual,
        &obs_params,
        readout,
    )?;
    // Analytic `init(...)` impulses before scaling: the amount at the subject-static `t = 0`
    // parameters (BSV η only under IOV, as `predict_iov`); the decay at those parameters,
    // or under IOV at each observation's own occasion parameters.
    if !model.analytical_init.is_empty() {
        let n_slots = (model.pk_param_fn)(theta, &eta[..n_eta], &subject.covariates, 0.0)
            .values
            .len();
        let pk0 = (model.pk_param_fn)(theta, &eta[..n_eta], &subject.covariates, 0.0);
        let mut e0: Vec<T> = et[..n_eta].to_vec();
        e0.extend(std::iter::repeat_n(
            T::from_f64(0.0),
            if iov { model.n_kappa } else { 0 },
        ));
        let amount: Vec<T> = taped_params(model, prog, n_slots, th, &e0, &subject.covariates)
            .iter()
            .zip(&pk0.values)
            .map(|(d, &v)| d.unwrap_or(T::from_f64(v)))
            .collect();
        let decay: Option<Vec<Vec<T>>> = iov.then(|| obs.iter().map(|(_, p)| p.clone()).collect());
        pv::add_closed_form_init_g(
            model,
            subject,
            th,
            &et[..n_eta],
            &amount,
            decay.as_deref(),
            &mut preds,
        )?;
    }
    // Scaling then LTBS: per occasion group for an IOV expression scale, else the shared
    // closed-form output stage.
    if iov && matches!(model.scaling, ScalingSpec::ExpressionScale { .. }) {
        // One scale per occasion group (`predict_iov`'s per-occasion `apply_scaling`): that
        // group's parameters at the subject covariates and `t = 0`.
        let ScalingSpec::ExpressionScale {
            deriv: Some(sprog), ..
        } = &model.scaling
        else {
            return None;
        };
        let l = layout?;
        let group_scale = |occ: u32| -> T {
            let (_, params) = at(occ, &subject.covariates, 0.0);
            let vars: Vec<T> = sprog
                .var_to_pk_slot()
                .iter()
                .map(|&sl| params.get(sl).copied().unwrap_or(T::from_f64(0.0)))
                .collect();
            sprog.eval_scale_g(th, &et[..n_eta], &subject.covariates, &vars)
        };
        let mut cache: std::collections::HashMap<Option<usize>, T> =
            std::collections::HashMap::new();
        for (j, y) in preds.iter_mut().enumerate() {
            let occ = subject.occasions.get(j).copied().unwrap_or(u32::MAX);
            let s = *cache
                .entry(l.group_of(occ))
                .or_insert_with(|| group_scale(occ));
            if !(s.val() > 0.0 && s.val().is_finite()) {
                return None;
            }
            *y = *y / s;
        }
        if model.log_transform {
            preds
                .iter_mut()
                .for_each(|y| *y = crate::pk::ltbs_log_g(*y));
        }
        return Some(preds);
    }
    closed_form_finish(model, subject, theta, eta, th, et, prog, preds)
}

/// Compartment-free engine (`$PRED`-equivalent): the readout program over the
/// individual parameters, per observation, with production's cadence
/// (`pk::apply_analytic_readout`): one subject-static snapshot unless a covariate or the
/// `TIME` built-in moves the parameters between rows, each row at its occasion's
/// `[η, κ_g]` under IOV, both clocks the TIME as written, then the output transform.
fn record_algebraic<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
    th: &ThetaLeaves<'_, T>,
    et: &[T],
) -> Option<Vec<T>> {
    let readout = model.analytic_readout.as_ref()?;
    let prog_out = match &readout.readout {
        crate::ode::OdeReadout::Single(_) => {
            readout.program.as_ref().filter(|p| p.is_dual_evaluable())?
        }
        _ => return None,
    };
    let prog = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .filter(|p| p.n_eta_axis() == model.n_eta + model.n_kappa)?;
    let iov = model.n_kappa > 0;
    let layout = if iov { Some(lay?) } else { None };
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let per_obs = iov || subject.has_tv_covariates() || uses_time;
    let n_eta = model.n_eta;
    let at = |j: Option<usize>| -> Option<Vec<T>> {
        let (cov, time) = match j {
            Some(j) => (subject.obs_cov(j), subject.param_time_obs(j)),
            None => (&subject.covariates, 0.0),
        };
        // The row's `[η, κ_g]` (κ = 0 for an unlabelled row): values and tape inputs.
        let (v, d): (Vec<f64>, Vec<T>) = match (layout, j) {
            (Some(l), Some(j)) => {
                let g = match subject.occasions.get(j) {
                    Some(&occ) => Some(l.group_of(occ)?),
                    None => None,
                };
                let mut d: Vec<T> = et[..n_eta].to_vec();
                match g {
                    Some(g) => d.extend_from_slice(&et[l.kappa_range(g)]),
                    None => d.extend(std::iter::repeat_n(T::from_f64(0.0), model.n_kappa)),
                }
                (
                    g.map_or_else(|| l.combined_pk_only(eta), |g| l.combined_effect(eta, g)),
                    d,
                )
            }
            _ => (eta.to_vec(), et.to_vec()),
        };
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
        let pk = (model.pk_param_fn)(theta, &v, cov, time);
        let by_slot = taped_params(model, prog, pk.values.len(), th, &d, cov);
        Some(
            by_slot
                .iter()
                .zip(&pk.values)
                .map(|(d, &v)| d.unwrap_or(T::from_f64(v)))
                .collect(),
        )
    };
    let shared = if per_obs { None } else { Some(at(None)?) };
    let (mut vars, mut stack) = (Vec::new(), Vec::new());
    (0..subject.obs_times.len())
        .map(|j| {
            let owned;
            let params: &[T] = match &shared {
                Some(p) => p,
                None => {
                    owned = at(Some(j))?;
                    &owned
                }
            };
            let _guard =
                crate::parser::model_parser::ModelTimeGuard::enter(subject.readout_time(j));
            let y =
                prog_out.eval_output_g::<T>(&[], params, subject.obs_cov(j), &mut vars, &mut stack);
            Some(crate::sens::ode_provider::apply_output_transform(model, y))
        })
        .collect()
}

/// The predictions at `η` over the tape (either engine), from the shared leaves.
///
/// `lay` is the subject's layout under IOV
/// ([`StackedLayout::sens_of`](crate::stats::stacked_layout::StackedLayout::sens_of)): the
/// caller's, so an EBE search builds it once rather than once per recording (#2037), or
/// built here when the caller has none.
fn record<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
    th: &ThetaLeaves<'_, T>,
    et: &[T],
) -> Option<Vec<T>> {
    let built;
    let lay = match lay {
        None if model.n_kappa > 0 => {
            built = crate::stats::stacked_layout::StackedLayout::sens_of(model, subject)?;
            Some(&built)
        }
        lay => lay,
    };
    if let Some((eta, et)) = expand_nested(model, lay, eta, et)? {
        return record_cells(model, subject, lay, theta, &eta, th, &et);
    }
    record_cells(model, subject, lay, theta, eta, th, et)
}

/// [`record`] on inputs laid out per cell: the engine dispatch.
fn record_cells<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
    th: &ThetaLeaves<'_, T>,
    et: &[T],
) -> Option<Vec<T>> {
    if model.is_algebraic() {
        return record_algebraic(model, subject, lay, theta, eta, th, et);
    }
    match (model.ode_spec.is_some(), model.n_kappa > 0) {
        (true, true) => record_ode_iov(model, subject, lay?, theta, eta, th, et),
        (true, false) => record_ode(model, subject, theta, eta, th, et),
        (false, true) => record_analytic_events(model, subject, lay, theta, eta, th, et),
        // Production's own route: the event walk for TV covariates / `TIME` / oral
        // infusion, the `t = 0` static superposition otherwise.
        (false, false)
            if crate::sens::provider::subject_routes_to_event_walk(model, subject)
                || closed_form_reshapes_f_window(model, subject) =>
        {
            record_analytic_events(model, subject, None, theta, eta, th, et)
        }
        (false, false) => record_analytic(model, subject, theta, eta, th, et),
    }
}

/// The subject objective `F` and the inner gradient `∇_η l` its EBE zeroes: FOCEI's
/// conditional pair, or FOCE's Sheiner–Beal marginal with the frozen-variance `l`
/// (`EbeVariance::Population`, `fpop = f(η = 0)`).
#[allow(clippy::too_many_arguments)]
fn objective<T: crate::sens::num::PkNum, const N: usize>(
    ctx: &LikCtx,
    interaction: bool,
    f: &[T],
    g: &[[T; N]],
    eta: &[T; N],
    fpop: Option<&[T]>,
    p: &ParT<T>,
) -> Option<(T, [T; N])> {
    if interaction {
        focei_f_grad(ctx, f, g, eta, p)
    } else {
        Some((
            foce_f(ctx, f, g, eta, fpop, p)?,
            inner_grad(ctx, f, g, eta, p, fpop)?,
        ))
    }
}

/// The canonical recording at `eta` (any engine the adjoint serves): per observation the
/// prediction, its η-gradient and its packed η-Hessian, or `None` outside the recorders'
/// scope. Shared by [`exact_eta_jacobian`] and [`exact_inner_hessian`].
fn canonical_jets<const N: usize, const M: usize>(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<DualV<N, M>>> {
    // Tape-free: θ constants, η free inputs, so every operation is plain forward
    // arithmetic (`M = 0`: `(f, ∂f/∂η)`; `M = N(N+1)/2`: with `∂²f/∂η²`).
    let th: Vec<DualV<N, M>> = theta.iter().map(|&v| DualV::constant(v)).collect();
    let unit = unit_dirs::<N>();
    let et: [DualV<N, M>; N] = std::array::from_fn(|k| DualV::free(eta[k], unit[k], [0.0; M]));
    record(
        model,
        subject,
        lay,
        theta,
        eta,
        &ThetaLeaves::theta(&th),
        &et,
    )
    .filter(|p| p.len() == subject.observations.len())
}

/// [`canonical_jets`] at first order (`(f, ∂f/∂η)`), the inner loop's per-step recording, on
/// the index-free [`DualF`] (`sens::vfree`): the same jets as a free `DualV<N, 0>`, kept in
/// registers. Only this order gets its own instantiation of the recorders; the
/// second-order jets share the outer gradient's `DualV<N, M>` ones.
fn canonical_jets1<const N: usize>(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<DualF<N, 0>>> {
    let th: Vec<DualF<N, 0>> = theta.iter().map(|&v| DualF::constant(v)).collect();
    let unit = unit_dirs::<N>();
    let et: [DualF<N, 0>; N] = std::array::from_fn(|k| DualF::free(eta[k], unit[k], []));
    record(
        model,
        subject,
        lay,
        theta,
        eta,
        &ThetaLeaves::theta(&th),
        &et,
    )
    .filter(|p| p.len() == subject.observations.len())
}

/// The model the predictor would evaluate for this subject (a closed-form absorption model
/// may route to its ODE twin), the dual width, and `eta` in the recordings' basis padded to
/// its dispatch bucket. The width is `n_eta`, the stacked `[η, κ]` under IOV, or — under
/// nested levels (#1082) — the compact axis count: `eta` arrives per cell (as every
/// reader of the stack sees it) and is recorded over the axes it expands from
/// ([`StackedLayout::sens_of`](crate::stats::stacked_layout::StackedLayout::sens_of)), so
/// each derivative is over the vector the EBE search moves. `None` when `eta` is not the
/// subject's width or a dispatch bucket's, and for a nested-level subject no recording
/// lays out (see [`nested_levels_outside_scope`]).
#[allow(clippy::type_complexity)]
fn resolve<'m, 'l>(
    model: &'m CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    layout: Option<&'l crate::stats::stacked_layout::StackedLayout>,
) -> Option<(
    &'m CompiledModel,
    usize,
    Vec<f64>,
    Option<std::borrow::Cow<'l, crate::stats::stacked_layout::StackedLayout>>,
)> {
    let (eff, w, mut e, lay) = if model.n_kappa > 0 {
        if nested_levels_outside_scope(model) {
            return None;
        }
        let eff = crate::pk::effective_model_for_eval_iov(model, subject, theta, eta);
        // The caller's layout is the subject's under any route: an ODE twin has the
        // model's `n_eta` / `n_kappa`, and one without the model's levels is declined above.
        let lay = match layout {
            Some(l) => std::borrow::Cow::Borrowed(l),
            None => std::borrow::Cow::Owned(crate::stats::stacked_layout::StackedLayout::sens_of(
                eff, subject,
            )?),
        };
        if model.has_nested_levels() && lay.nested().is_none() {
            return None;
        }
        (eff, lay.n_axes(), lay.to_axes(eta)?, Some(lay))
    } else {
        if eta.len() != model.n_eta {
            return None;
        }
        (
            crate::pk::effective_model_for_eval(model, subject, theta, eta),
            model.n_eta,
            eta.to_vec(),
            None,
        )
    };
    e.resize(WIDTH_BUCKETS.into_iter().find(|&b| b >= w)?, 0.0);
    Some((eff, w, e, lay))
}

/// The number of random effects a recording of `subject` is over: `n_eta`, the stacked
/// `[η, κ]` under IOV, or the compact axes under nested levels; `None` for a nested-level
/// subject never given its cells.
pub(crate) fn subject_width(model: &CompiledModel, subject: &Subject) -> Option<usize> {
    if model.n_kappa == 0 {
        return Some(model.n_eta);
    }
    Some(crate::stats::stacked_layout::StackedLayout::sens_of(model, subject)?.n_axes())
}

/// Whether a model with nested levels (#1082) lies outside the recordings: one whose IOV
/// subjects may run on a closed-form transit/IG model's ODE twin
/// (`absorption_ode_equivalent`), a separate `CompiledModel` parsed from emitted source
/// that is not known to carry the levels — a recording on it would lay κ out per cell,
/// one κ per arm where the model draws one per study. `false` without nested levels.
pub(crate) fn nested_levels_outside_scope(model: &CompiledModel) -> bool {
    model.has_nested_levels() && model.absorption_ode_equivalent.is_some()
}

/// A nested-level recording's inputs per cell (#1082): `eta` and `et` over the compact
/// axes, expanded to the stacked `[η, κ_cell0, κ_cell1, …]` every engine reads, each
/// cell's κ the *same* leaf as its group's axis — so the cells of one group share one
/// derivative direction. `None` without nested levels (the inputs are the stack itself).
fn expand_nested<T: Copy>(
    model: &CompiledModel,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    eta: &[f64],
    et: &[T],
) -> Option<Option<(Vec<f64>, Vec<T>)>> {
    if !model.has_nested_levels() {
        return Some(None);
    }
    let layout = lay?;
    layout.nested()?;
    let n_eta = layout.n_eta();
    if et.len() < layout.n_axes() || eta.len() < layout.n_axes() {
        return None;
    }
    let mut v = eta[..n_eta].to_vec();
    let mut d = et[..n_eta].to_vec();
    for g in 0..layout.n_groups() {
        for k in 0..layout.n_kappa() {
            let ax = layout.kappa_axis(g, k);
            v.push(eta[ax]);
            d.push(et[ax]);
        }
    }
    Some(Some((v, d)))
}

/// The exact `∂f/∂η` (`n_obs × width`, stacked `[η, κ]` under IOV) from one first-order
/// recording — the analytic replacement for a finite-difference η-Jacobian wherever the
/// recorders serve the subject.
pub(crate) fn exact_eta_jacobian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<DMatrix<f64>> {
    fn go<const N: usize, const M: usize>(
        model: &CompiledModel,
        subject: &Subject,
        lay: Option<&crate::stats::stacked_layout::StackedLayout>,
        theta: &[f64],
        eta: &[f64],
        w: usize,
    ) -> Option<DMatrix<f64>> {
        let p = canonical_jets1::<N>(model, subject, lay, theta, eta)?;
        Some(DMatrix::from_fn(p.len(), w, |j, k| p[j].g[k]))
    }
    let (model, w, eta, lay) = resolve(model, subject, theta, eta, None)?;
    if model.is_sde() {
        return None;
    }
    by_bucket!(
        eta.len(),
        go,
        (model, subject, lay.as_deref(), theta, &eta, w)
    )
}

/// Per observation `(f, ∂f/∂η)` at `eta` (the stacked `[η, κ]` under IOV) from one
/// tape-free first-order recording: the inner loop's η-Jacobian, in the light
/// provider's row layout.
pub(crate) fn exact_obs_grads(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<crate::sens::provider::ObsGrad>> {
    exact_obs_grads_in(model, subject, None, theta, eta)
}

/// [`exact_obs_grads`] over a caller-built `layout` (the subject's
/// [`StackedLayout::sens_of`](crate::stats::stacked_layout::StackedLayout::sens_of) under
/// IOV), so an EBE search builds it once rather than once per gradient (#2037).
pub(crate) fn exact_obs_grads_in(
    model: &CompiledModel,
    subject: &Subject,
    layout: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<crate::sens::provider::ObsGrad>> {
    let mut rows = Vec::new();
    exact_obs_grads_into(model, subject, layout, theta, eta, &mut rows).then_some(rows)
}

/// [`exact_obs_grads`] into `rows`, reusing its allocations (the inner loop calls this
/// once per BFGS iteration). `false` outside the recorders' scope or on a non-finite row,
/// leaving `rows` unspecified.
pub(crate) fn exact_obs_grads_into(
    model: &CompiledModel,
    subject: &Subject,
    layout: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
    rows: &mut Vec<crate::sens::provider::ObsGrad>,
) -> bool {
    fn go<const N: usize, const M: usize>(
        model: &CompiledModel,
        subject: &Subject,
        lay: Option<&crate::stats::stacked_layout::StackedLayout>,
        theta: &[f64],
        eta: &[f64],
        w: usize,
        rows: &mut Vec<crate::sens::provider::ObsGrad>,
    ) -> Option<()> {
        let p = canonical_jets1::<N>(model, subject, lay, theta, eta)?;
        rows.resize_with(p.len(), || crate::sens::provider::ObsGrad {
            f: 0.0,
            df_deta: Vec::new(),
        });
        for (row, q) in rows.iter_mut().zip(&p) {
            if !(q.v.is_finite() && q.g.iter().all(|v| v.is_finite())) {
                return None;
            }
            row.f = q.v;
            row.df_deta.clear();
            row.df_deta.extend_from_slice(&q.g[..w]);
        }
        Some(())
    }
    let Some((model, w, eta_p, lay)) = resolve(model, subject, theta, eta, layout) else {
        return false;
    };
    if model.is_sde() {
        return false;
    }
    if by_bucket!(
        eta_p.len(),
        go,
        (model, subject, lay.as_deref(), theta, &eta_p, w, rows)
    )
    .is_none()
    {
        return false;
    }
    // A FREM covariate pseudo-observation predicts `θ_i + η_e`, with `∂f/∂η = e_e`.
    if let Some(fc) = model.frem_config.as_ref() {
        for (j, row) in rows.iter_mut().enumerate() {
            let Some(&(ti, ei)) = subject
                .fremtype
                .get(j)
                .filter(|&&ft| ft != 0)
                .and_then(|ft| fc.fremtype_to_indices.get(ft))
            else {
                continue;
            };
            row.f = theta.get(ti).copied().unwrap_or(0.0) + eta.get(ei).copied().unwrap_or(0.0);
            row.df_deta
                .iter_mut()
                .enumerate()
                .for_each(|(k, v)| *v = if k == ei { 1.0 } else { 0.0 });
        }
    }
    true
}

/// The exact conditional Hessian `∇²_η l` at `eta`: the kernel's `l` over a second-order
/// η-dual whose predictions carry `(f, G, Q)` from one canonical recording. The analytic
/// replacement for a finite-difference posterior Hessian.
pub(crate) fn exact_inner_hessian(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta: &[f64],
) -> Option<DMatrix<f64>> {
    fn go<const N: usize, const M: usize>(
        model: &CompiledModel,
        subject: &Subject,
        lay: Option<&crate::stats::stacked_layout::StackedLayout>,
        params: &ModelParameters,
        eta: &[f64],
        w: usize,
    ) -> Option<DMatrix<f64>> {
        use crate::sens::rtape::DualH;
        let p = canonical_jets::<N, M>(model, subject, lay, &params.theta, eta)?;
        let ctx = LikCtx::new(model, subject, params);
        let mut pv = ParT::of(model, subject, params);
        pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(
            stacked_omega::<N>(model, subject, params, params, w).chol,
        )?;
        let fh: Vec<DualH<N, M, f64>> = p
            .iter()
            .map(|q| DualH {
                v: q.v,
                g: q.g,
                h: q.h,
            })
            .collect();
        let unit = unit_dirs::<N>();
        let eh: [DualH<N, M, f64>; N] = std::array::from_fn(|k| DualH {
            v: eta[k],
            g: unit[k],
            h: [0.0; M],
        });
        let l = inner_l(
            &ctx,
            &fh,
            &eh,
            &pv.map(|x| DualH {
                v: x,
                g: [0.0; N],
                h: [0.0; M],
            }),
            None,
        )?;
        let h = DMatrix::from_fn(w, w, |a, b| l.hess(a, b));
        h.iter().all(|v| v.is_finite()).then_some(h)
    }
    let (model, w, eta, lay) = resolve(model, subject, &params.theta, eta, None)?;
    // IOV scores FREM rows by a sentinel, not the kernel's `l` (as `exact_htilde` and
    // `exact_inner_gradient_in` decline).
    let iov_mismatch = model.n_kappa > 0 && subject.fremtype.iter().any(|&ft| ft > 0);
    if iov_mismatch || !likelihood_supported(model, subject, params) {
        return None;
    }
    by_bucket!(
        eta.len(),
        go,
        (model, subject, lay.as_deref(), params, &eta, w)
    )
}

/// FOCEI's Gauss–Newton `H̃` at `eta` (the stacked `[η, κ]` under IOV), the kernel's on
/// one tape-free first-order recording: the AGQ grid's `H̃` anchor.
pub(crate) fn exact_htilde(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta: &[f64],
) -> Option<DMatrix<f64>> {
    fn go<const N: usize, const M: usize>(
        model: &CompiledModel,
        subject: &Subject,
        lay: Option<&crate::stats::stacked_layout::StackedLayout>,
        params: &ModelParameters,
        eta: &[f64],
        w: usize,
    ) -> Option<DMatrix<f64>> {
        let p = canonical_jets1::<N>(model, subject, lay, &params.theta, eta)?;
        let ctx = LikCtx::new(model, subject, params);
        let mut pv = ParT::of(model, subject, params);
        pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(
            stacked_omega::<N>(model, subject, params, params, w).chol,
        )?;
        let f: Vec<f64> = p.iter().map(|q| q.v).collect();
        let g: Vec<[f64; N]> = p.iter().map(|q| q.g).collect();
        let e: [f64; N] = std::array::from_fn(|k| eta[k]);
        let ht = crate::sens::focei_kernel::focei_htilde(&ctx, &f, &g, &e, &pv)?;
        let h = DMatrix::from_fn(w, w, |a, b| ht[a][b]);
        h.iter().all(|v| v.is_finite()).then_some(h)
    }
    let (model, w, eta, lay) = resolve(model, subject, &params.theta, eta, None)?;
    // IOV scores FREM rows by a sentinel, not the kernel's `l`.
    let iov_mismatch = model.n_kappa > 0 && subject.fremtype.iter().any(|&ft| ft > 0);
    if iov_mismatch || !likelihood_supported(model, subject, params) {
        return None;
    }
    by_bucket!(
        eta.len(),
        go,
        (model, subject, lay.as_deref(), params, &eta, w)
    )
}

/// The exact inner gradient `∇_η l` at `eta` (the stacked `[η, κ]` under IOV): the
/// kernel's `inner_grad` on one canonical recording, under the conditional or FOCE's
/// frozen-variance convention (`frozen`, `EbeVariance::Population`). The analytic
/// replacement for an inner finite-difference gradient wherever the recorders serve the
/// subject.
pub(crate) fn exact_inner_gradient(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta: &[f64],
    frozen: Option<&[f64]>,
) -> Option<Vec<f64>> {
    exact_inner_gradient_in(model, subject, None, params, eta, frozen)
}

/// [`exact_inner_gradient`] over a caller-built `layout`, as [`exact_obs_grads_in`].
pub(crate) fn exact_inner_gradient_in(
    model: &CompiledModel,
    subject: &Subject,
    layout: Option<&crate::stats::stacked_layout::StackedLayout>,
    params: &ModelParameters,
    eta: &[f64],
    frozen: Option<&[f64]>,
) -> Option<Vec<f64>> {
    fn go<const N: usize, const M: usize>(
        model: &CompiledModel,
        subject: &Subject,
        lay: Option<&crate::stats::stacked_layout::StackedLayout>,
        params: &ModelParameters,
        eta: &[f64],
        frozen: Option<&[f64]>,
        w: usize,
    ) -> Option<Vec<f64>> {
        let p = canonical_jets1::<N>(model, subject, lay, &params.theta, eta)?;
        let ctx = LikCtx::new(model, subject, params);
        let mut pv = ParT::of(model, subject, params);
        pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(
            stacked_omega::<N>(model, subject, params, params, w).chol,
        )?;
        let f: Vec<f64> = p.iter().map(|q| q.v).collect();
        let g: Vec<[f64; N]> = p.iter().map(|q| q.g).collect();
        let e: [f64; N] = std::array::from_fn(|k| eta[k]);
        let gr = inner_grad(&ctx, &f, &g, &e, &pv, frozen)?;
        gr.iter().all(|v| v.is_finite()).then(|| gr[..w].to_vec())
    }
    let (model, w, eta, lay) = resolve(model, subject, &params.theta, eta, layout)?;
    // IOV scores FREM rows by a sentinel, not the kernel's `l`.
    let iov_mismatch = model.n_kappa > 0 && subject.fremtype.iter().any(|&ft| ft > 0);
    if iov_mismatch || !likelihood_supported(model, subject, params) {
        return None;
    }
    by_bucket!(
        eta.len(),
        go,
        (model, subject, lay.as_deref(), params, &eta, frozen, w)
    )
}

/// How many leading θ the recordings take as leaves: those the parameter program reads,
/// or every θ for a `[covariate_nn]` model, whose generated weights only the networks read.
fn theta_axis(model: &CompiledModel) -> Option<usize> {
    let n = param_program(model)?.n_theta_axis();
    Some(if model.has_covariate_nn() {
        usize::MAX
    } else {
        n
    })
}

/// The individual-parameter program the engine evaluates (its θ axes are the θ leaves).
fn param_program(model: &CompiledModel) -> Option<&crate::parser::model_parser::IndivParamProgram> {
    if model.is_algebraic() {
        return model.indiv_param_partials.indiv_param_program.as_ref();
    }
    match model.ode_spec.as_ref() {
        Some(ode) => ode.indiv_param_program.as_ref(),
        None => crate::sens::provider::analytic_param_program(model),
    }
}

/// Ω as the stacked prior's Cholesky factor (`L ⊕ (I_K ⊗ L_IOV)` under IOV), and per
/// packed Ω / Ω_IOV entry its packed index, the stacked positions it fills (one per
/// occasion block for Ω_IOV) and its packing chain (`L_kk` for the log-diagonal).
struct StackedOmega {
    chol: Vec<Vec<f64>>,
    placed: Vec<(usize, Vec<(usize, usize)>, f64)>,
}

fn stacked_omega<const N: usize>(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    params: &ModelParameters,
    active: usize,
) -> StackedOmega {
    let (n_eta, omega) = (model.n_eta, &params.omega);
    let seg = crate::estimation::parameterization::packed_segments(template);
    let mut chol = vec![vec![0.0; N]; N];
    let mut placed = Vec::new();
    for (e, &(r, c)) in lower_tri_entries(n_eta, omega.diagonal).iter().enumerate() {
        chol[r][c] = omega.chol[(r, c)];
        placed.push((
            seg.omega_start() + e,
            vec![(r, c)],
            if r == c { omega.chol[(r, r)] } else { 1.0 },
        ));
    }
    if let Some(oi) = params.omega_iov.as_ref().filter(|_| model.n_kappa > 0) {
        let nk = model.n_kappa;
        // Under nested levels (#1082) each entry sits once per group of its κs' level, on
        // the compact axes; a pair across two levels is a structural zero, with no leaf.
        let nested = model
            .has_nested_levels()
            .then(|| crate::stats::stacked_layout::StackedLayout::sens_of(model, subject))
            .flatten();
        for (e, &(r, c)) in lower_tri_entries(nk, oi.diagonal).iter().enumerate() {
            let at: Vec<(usize, usize)> = match &nested {
                Some(l) => l.iov_entry_axes(r, c),
                None => (0..(active - n_eta) / nk)
                    .map(|g| {
                        let off = crate::stats::stacked_layout::kappa_offset(n_eta, nk, g);
                        (off + r, off + c)
                    })
                    .collect(),
            };
            if at.is_empty() {
                continue;
            }
            for &(i, j) in &at {
                chol[i][j] = oi.chol[(r, c)];
            }
            placed.push((
                seg.iov_start() + e,
                at,
                if r == c { oi.chol[(r, r)] } else { 1.0 },
            ));
        }
    }
    // Inert padding directions (a width bucket above the active width): a unit prior,
    // which no prediction or data term reads.
    for (i, row) in chol.iter_mut().enumerate().skip(active) {
        row[i] = 1.0;
    }
    StackedOmega { chol, placed }
}

/// The population parameters the likelihood reads, as scalar-tape leaves: Ω by its
/// packed Cholesky entries (one leaf per entry, shared across occasion blocks), σ, ρ,
/// the serial ρ, and θ where FREM rows read it.
fn tape_params(
    model: &CompiledModel,
    subject: &Subject,
    pv: &ParT<f64>,
    so: &StackedOmega,
) -> Option<(ParT<R>, Vec<R>)> {
    // Constants first (the inert padding's unit diagonal), then the packed leaves.
    let mut lr: Vec<Vec<R>> = so
        .chol
        .iter()
        .map(|row| row.iter().map(|&x| R::c(x)).collect())
        .collect();
    let om: Vec<R> = so
        .placed
        .iter()
        .map(|(_, at, _)| {
            let leaf = rinput(so.chol[at[0].0][at[0].1]);
            for &(i, j) in at {
                lr[i][j] = leaf;
            }
            leaf
        })
        .collect();
    let pr = ParT {
        omega: crate::sens::focei_kernel::OmegaT::from_chol(lr)?,
        sigma: pv.sigma.iter().map(|&x| rinput(x)).collect(),
        rho: pv.rho.iter().map(|&x| rinput(x)).collect(),
        serial_rho: pv.serial_rho.map(rinput),
        theta: if theta_read_directly(model) {
            pv.theta.iter().map(|&x| rinput(x)).collect()
        } else {
            pv.theta.iter().map(|&x| R::c(x)).collect()
        },
        ruv_mult: None,
    };
    // A custom residual magnitude, per observation, from its compiled programs over the
    // θ leaves: its direct θ channel is then on the tape with everything else.
    let ruv_mult = match model.ruv_magnitude.as_ref().filter(|rm| rm.is_active()) {
        Some(rm) => Some(
            (0..subject.observations.len())
                .map(|j| {
                    rm.eval_obs_g(
                        &pr.theta,
                        subject.obs_cov(j),
                        subject.readout_time(j),
                        rm.tad_for(subject, j),
                    )
                })
                .collect::<Option<Vec<_>>>()?,
        ),
        None => None,
    };
    Some((ParT { ruv_mult, ..pr }, om))
}

/// Whether the likelihood reads θ directly, not only through the predictions: FREM
/// pseudo-observations (`θ_i + η_e`) and a θ-dependent custom residual magnitude.
fn theta_read_directly(model: &CompiledModel) -> bool {
    model.frem_config.is_some()
        || model
            .ruv_magnitude
            .as_ref()
            .is_some_and(|rm| rm.is_active() && rm.theta_dependent)
}

/// The packed gradient `[θ, Ω, σ, Ω_IOV, ρ]` from `∂Ψ/∂θ` through the predictions
/// (`theta_adj`, per θ) and the scalar tape's adjoints (`psi_of`), through each block's
/// packing chain.
#[allow(clippy::too_many_arguments)]
fn packed_gradient(
    model: &CompiledModel,
    template: &ModelParameters,
    params: &ModelParameters,
    len: usize,
    theta_adj: &[f64],
    psi_of: &dyn Fn(R) -> f64,
    pr: &ParT<R>,
    om: &[R],
    so: &StackedOmega,
    pv: &ParT<f64>,
) -> Vec<f64> {
    let theta = &params.theta;
    let seg = crate::estimation::parameterization::packed_segments(template);
    let chain =
        |m: usize| crate::estimation::sens_outer_gradient::theta_dx_chain(template, theta, m);
    let mut out = vec![0.0; len];
    // θ through the predictions, and directly (FREM rows, serial ρ).
    for (m, &a) in theta_adj.iter().enumerate() {
        out[m] += a * chain(m);
    }
    for (m, &leaf) in pr.theta.iter().enumerate() {
        if leaf.idx().is_some() {
            out[m] += psi_of(leaf) * chain(m);
        }
    }
    if let (Some(serial), Some(leaf)) = (model.residual_serial.as_ref(), pr.serial_rho) {
        out[serial.rho_theta] += psi_of(leaf) * chain(serial.rho_theta);
    }
    // Ω and Ω_IOV: packed diagonal `ln L_kk`, off-diagonal `L_rc`.
    for ((idx, _, ch), &leaf) in so.placed.iter().zip(om) {
        out[*idx] = psi_of(leaf) * ch;
    }
    // σ (log), ρ (Fisher z).
    for (k, &leaf) in pr.sigma.iter().enumerate() {
        out[seg.sigma_start() + k] = psi_of(leaf) * params.sigma.values[k];
    }
    let rho_start = crate::estimation::parameterization::rho_packed_start(template);
    for (k, &leaf) in pr.rho.iter().enumerate() {
        out[rho_start + k] =
            psi_of(leaf) * crate::estimation::parameterization::rho_chain(pv.rho[k]);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run<const N: usize, const M: usize>(
    model: &CompiledModel,
    pop_model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    template: &ModelParameters,
    params: &ModelParameters,
    x: &[f64],
    eta: &[f64],
    interaction: bool,
    active: usize,
) -> Option<Vec<f64>> {
    let theta = &params.theta;

    // ---- 1. forward: record the predictions with η-tangents and Hessian payload ---------
    // Only the θ leaves are swept to, so η enters as free (untaped) inputs: an η-only
    // subexpression is plain forward arithmetic, taped only where it meets θ.
    vtape_reset(N + 1);
    let th = theta_leaves::<N, M, false>(theta_axis(model)?, theta);
    // `[covariate_nn]` outputs as tape leaves when the weights reach the likelihood through
    // them alone (`NnReduction`): the recordings read the leaves, and the weights' adjoint
    // is one backward pass through each network below, in place of every network taped
    // over its weights on the outer dual.
    #[cfg(feature = "nn")]
    let nn_out = nn_output_leaves::<N, M, false>(model, subject, template, theta);
    #[cfg(feature = "nn")]
    let lv = ThetaLeaves::with_nn(&th, nn_out.as_ref().map(|(_, z)| z.as_slice()));
    #[cfg(not(feature = "nn"))]
    let lv = ThetaLeaves::theta(&th);
    let unit = unit_dirs::<N>();
    let et: [DualV<N, M>; N] = std::array::from_fn(|k| DualV::free(eta[k], unit[k], [0.0; M]));
    let preds = record(model, subject, lay, theta, eta, &lv, &et)?;
    let n_obs = preds.len();
    if n_obs != subject.observations.len() {
        return None;
    }
    // FOCE scores R⁰ at the population prediction f(η = 0): the same engine, on the same
    // θ leaves, with η held at zero (values only).
    let pop = if !interaction && model.error_spec.has_f_dependent_variance() {
        let zeros = [0.0; N];
        let p0 = record(
            pop_model,
            subject,
            lay,
            theta,
            &zeros,
            &lv,
            &[DualV::constant(0.0); N],
        )?;
        (p0.len() == n_obs).then_some(p0)?.into()
    } else {
        None
    };

    // ---- 2. w = H⁻¹∇_η F, from (F, ∇_η l) over a forward η-dual: no tape --------------
    // f carries its η-tangents G, and G its tangents ∂G/∂η = ∂²f/∂η² (the Hessian payload),
    // so the dual parts of F and ∇l are the total ∇_η F and H = ∇²_η l. (f_pop does not
    // depend on η.)
    let ctx = LikCtx::new(model, subject, params);
    let so = stacked_omega::<N>(model, subject, template, params, active);
    let mut pv = ParT::of(model, subject, params);
    pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(so.chol.clone())?;
    let pop_v: Option<Vec<f64>> = pop
        .as_ref()
        .map(|p0: &Vec<DualV<N, M>>| p0.iter().map(|p| p.v).collect());
    let (grad_f, h, grad_l) =
        eta_derivatives(&ctx, interaction, &preds, pop_v.as_deref(), eta, &pv)?;
    let w = mode_solve(&h, &grad_f, &grad_l)?;

    // ---- 3. Ψ = F − wᵀ∇_η l on the scalar tape, one single-lane sweep -----------------
    rtape_reset();
    let fr: Vec<R> = preds.iter().map(|p| rinput(p.v)).collect();
    let gr: Vec<[R; N]> = preds
        .iter()
        .map(|p| std::array::from_fn(|k| rinput(p.g[k])))
        .collect();
    let pr0: Option<Vec<R>> = pop
        .as_ref()
        .map(|p0: &Vec<DualV<N, M>>| p0.iter().map(|p| rinput(p.v)).collect());
    let er: [R; N] = std::array::from_fn(|k| rinput(eta[k]));
    let (pr, om) = tape_params(model, subject, &pv, &so)?;
    let (fval, lg) = objective::<R, N>(&ctx, interaction, &fr, &gr, &er, pr0.as_deref(), &pr)?;
    let psi = fval
        - <R as crate::sens::num::PkNum>::wsum3(
            0.0,
            (0..N).map(|k| (R::c(w[k]), lg[k], R::c(1.0))),
        );
    let sadj = rsweep::<1>(&[psi]);
    let psi_of = |x: R| -> f64 { x.idx().map_or(0.0, |i| sadj[i]) };

    // ---- 4. Ψ seeds on the taped predictions, one single-lane sweep to the θ leaves -----
    let vadj = vadj_init::<1>();
    let mut top = 0usize;
    let mut seed = |node: u32, comp: usize, v: f64| {
        if node != NONE {
            top = top.max(node as usize + 1);
            vadj[vadj_index::<N, 1>(node, comp, 0)] += v;
        }
    };
    for (j, p) in preds.iter().enumerate() {
        seed(p.i, 0, psi_of(fr[j]));
        for k in 0..N {
            seed(p.i, 1 + k, psi_of(gr[j][k]));
        }
    }
    if let (Some(p0), Some(r0)) = (pop.as_ref(), pr0.as_ref()) {
        for (p, &x) in p0.iter().zip(r0) {
            seed(p.i, 0, psi_of(x));
        }
    }
    let adj = vsweep::<N, 1>(top);
    // A leaf's value-component adjoint (θ, and the networks' outputs under the reduction).
    let at = |l: &DualV<N, M>| {
        if l.i == NONE {
            0.0
        } else {
            adj[vadj_index::<N, 1>(l.i, 0, 0)]
        }
    };
    #[cfg_attr(not(feature = "nn"), allow(unused_mut))]
    let mut theta_adj: Vec<f64> = th.iter().map(at).collect();
    #[cfg(feature = "nn")]
    if let Some((red, z)) = &nn_out {
        add_nn_weight_adjoint(model, red, z, theta, at, &mut theta_adj)?;
    }
    let out = packed_gradient(
        model,
        template,
        params,
        x.len(),
        &theta_adj,
        &psi_of,
        &pr,
        &om,
        &so,
        &pv,
    );
    Some(out)
}

/// `∇_η F` and `H = ∂(∇_η l)/∂η` at `eta`, from `(F, ∇_η l)` over a forward η-dual: `f`
/// carries its tangents `G`, and `G` its tangents `∂²f/∂η²` (the payload), so the dual
/// parts are the total derivatives. `fpop` (FOCE's `f(η = 0)`) does not depend on η.
pub(super) fn eta_derivatives<const N: usize, const M: usize>(
    ctx: &LikCtx,
    interaction: bool,
    preds: &[DualV<N, M>],
    fpop: Option<&[f64]>,
    eta: &[f64],
    pv: &ParT<f64>,
) -> Option<(DVector<f64>, DMatrix<f64>, [f64; N])> {
    type D<const N: usize> = DualT<N, f64>;
    let fd: Vec<D<N>> = preds.iter().map(|p| DualT { v: p.v, g: p.g }).collect();
    let gd: Vec<[D<N>; N]> = preds
        .iter()
        .map(|p| {
            std::array::from_fn(|k| DualT {
                v: p.g[k],
                g: std::array::from_fn(|m| p.hess(k, m)),
            })
        })
        .collect();
    let ed: [D<N>; N] = std::array::from_fn(|k| {
        let mut g = [0.0; N];
        g[k] = 1.0;
        DualT { v: eta[k], g }
    });
    let pd: Option<Vec<D<N>>> =
        fpop.map(|p0| p0.iter().map(|&v| DualT { v, g: [0.0; N] }).collect());
    let (fa, la) = objective::<D<N>, N>(
        ctx,
        interaction,
        &fd,
        &gd,
        &ed,
        pd.as_deref(),
        &pv.wrap::<N>(),
    )?;
    let h = DMatrix::from_fn(N, N, |k, m| 0.5 * (la[k].g[m] + la[m].g[k]));
    Some((DVector::from_column_slice(&fa.g), h, la.map(|l| l.v)))
}

/// Theta leaves (shared by every recording on the tape) and η leaves with tangent rows
/// `dirs`, for one recording at `eta`.
fn eta_leaves<const N: usize, const M: usize, const TH: bool>(
    eta: &[f64],
    dirs: &[[f64; N]; N],
) -> [DualV<N, M, TH>; N] {
    std::array::from_fn(|k| DualV::leaf(eta[k], dirs[k], [0.0; M]))
}

/// The jitter production's proposal adds to the anchor Hessian (`build_proposal`):
/// `H_ii += max(10⁻⁶|H_ii|, 10⁻¹⁰)`, on any number type.
fn regularise<T: crate::sens::num::PkNum>(h: &mut [Vec<T>]) {
    for (i, row) in h.iter_mut().enumerate() {
        let d = row[i];
        row[i] = if 1e-6 * d.val().abs() > 1e-10 {
            d + T::from_f64(1e-6) * d.abs()
        } else {
            d + T::from_f64(1e-10)
        };
    }
}

/// Exact AGQ subject gradient by the adjoint method — `n_agq` Gauss–Hermite nodes per
/// dimension around the conditional mode `η̂`, scaled by the exact Hessian (`exact`; one
/// node is Laplace) or FOCEI's Gauss–Newton `H̃`.
///
/// Production's objective (`agq_subject_evaluate`) is
///
/// ```text
///   A = (d/2)·ln π + ½·log|H_reg| − LSE_j[c_j − l(b_j)],   b_j = η̂ + √2·L⁻ᵀz_j,
/// ```
///
/// with `H_reg = H_a + jitter` the regularised anchor and `L = chol(H_reg)`; one node is
/// `l(η̂) + ½log|H_reg|`. The node predictions enter as `f_j + G_j·(b − b_j)`, so the grid's
/// movement with `x` and `η̂` (through `H_a`) is differentiated on the scalar tape, the
/// hand-written grid response is not needed. `Ψ = A − wᵀ∇_η l(η̂)`, `w = H⁻¹∇_η̂A` with
/// `H` the exact inner Hessian, and `dA/dx = ∂ₓΨ`.
///
/// Under the exact anchor the second-order sensitivities `Q = ∂²f/∂η²` at `η̂` enter only
/// through `H_a = Σ_j (∂l/∂f_j)·Q_j + …`, so each row's adjoint is `c_j·S` for one
/// symmetric `S`; the taped recording at `η̂` runs along `S`'s eigenvectors `u_m`, to
/// second order along each (`1 + 2N` components), and is seeded `u_mᵀ adj(Q_j) u_m`. The
/// Gauss–Newton anchor reads no `Q`, so its recording is first order.
#[allow(clippy::too_many_arguments)]
fn run_agq<const N: usize, const M: usize>(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    template: &ModelParameters,
    params: &ModelParameters,
    x: &[f64],
    eta: &[f64],
    n_agq: usize,
    exact: bool,
    active: usize,
    held: bool,
) -> Option<Vec<f64>>
where
    Bucket<N, M>: ExactAnchor<N, M>,
{
    use crate::sens::rtape::DualH;
    let theta = &params.theta;
    let ctx = LikCtx::new(model, subject, params);
    let so = stacked_omega::<N>(model, subject, template, params, active);
    let mut pv = ParT::of(model, subject, params);
    pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(so.chol.clone())?;
    let unit = unit_dirs::<N>();
    let eta_a: [f64; N] = std::array::from_fn(|k| eta[k]);

    // ---- 1. values at η̂ (an untaped canonical pass): f, G, Q; the exact H and the anchor
    let p1 = canonical_jets::<N, M>(model, subject, lay, theta, eta)?;
    let lift_h = |x: f64| DualH {
        v: x,
        g: [0.0; N],
        h: [0.0; M],
    };
    let h_exact = {
        let fh: Vec<DualH<N, M, f64>> = p1
            .iter()
            .map(|p| DualH {
                v: p.v,
                g: p.g,
                h: p.h,
            })
            .collect();
        let eh: [DualH<N, M, f64>; N] = std::array::from_fn(|k| DualH {
            v: eta[k],
            g: unit[k],
            h: [0.0; M],
        });
        let l = inner_l(&ctx, &fh, &eh, &pv.map(lift_h), None)?;
        DMatrix::from_fn(N, N, |a, b| l.hess(a, b))
    };
    let h_inner = h_exact.clone();
    let mut h_a: Vec<Vec<f64>> = if exact {
        (0..N)
            .map(|a| (0..N).map(|b| h_exact[(a, b)]).collect())
            .collect()
    } else {
        let f: Vec<f64> = p1.iter().map(|p| p.v).collect();
        let g: Vec<[f64; N]> = p1.iter().map(|p| p.g).collect();
        crate::sens::focei_kernel::focei_htilde(&ctx, &f, &g, &eta_a, &pv)?
    };
    regularise(&mut h_a);
    // Production's fallback when the regularised anchor is not positive-definite (a mode
    // the inner search left loosely converged, a flat direction): the prior precision `Ω⁻¹`
    // scales the grid and its log-determinant enters `F` (`build_proposal`). It is part of
    // the objective, so the gradient follows it rather than declining.
    let lh = crate::sens::focei_kernel::chol(&h_a)
        .or_else(|| crate::sens::focei_kernel::chol(&pv.omega.inv))?;
    // The tensor grid: per node its `z`, `c_j = Σ log w + ‖z‖²`, and `b_j` (f64).
    let (gh_nodes, _, gh_logw) = crate::estimation::agq::cached_gauss_hermite(n_agq);
    let one_point = n_agq == 1;
    let mut grid: Vec<([f64; N], f64, [f64; N])> = Vec::new();
    if !one_point {
        // Over the active dimensions only; an inert padding direction stays at z = 0.
        let mut idx = [0usize; N];
        loop {
            let z: [f64; N] =
                std::array::from_fn(|k| if k < active { gh_nodes[idx[k]] } else { 0.0 });
            let c = (0..active)
                .map(|k| gh_logw[idx[k]] + z[k] * z[k])
                .sum::<f64>();
            // b = η̂ + √2·L⁻ᵀz
            let mut step = [0.0; N];
            for i in (0..N).rev() {
                let mut acc = z[i];
                for j in (i + 1)..N {
                    acc -= lh[j][i] * step[j];
                }
                step[i] = acc / lh[i][i];
            }
            grid.push((
                z,
                c,
                std::array::from_fn(|k| eta[k] + std::f64::consts::SQRT_2 * step[k]),
            ));
            let mut k = 0;
            while k < active {
                idx[k] += 1;
                if idx[k] < n_agq {
                    break;
                }
                idx[k] = 0;
                k += 1;
            }
            if k == active {
                break;
            }
        }
    }
    if exact {
        <Bucket<N, M> as ExactAnchor<N, M>>::taped(
            model, subject, lay, template, params, x, eta, &ctx, &so, &pv, &p1, &grid, h_inner,
            active, held,
        )
    } else {
        agq_taped::<N, M, M, false>(
            model, subject, lay, template, params, x, eta, &ctx, &so, &pv, &p1, &grid, h_inner,
            exact, active, held,
        )
    }
}

/// The exact anchor's dispatch widths: its own, shorter ladder. Its recording carries `1 + 2N`
/// components per scalar on a taped Hessian sweep, its own dual family (`DualV<N, N, true>`)
/// that every width instantiates in full — so it pads to these rather than to every
/// [`WIDTH_BUCKETS`] entry (#2093 review), at the cost of a few inert directions on the
/// widths between.
const EXACT_ANCHOR_BUCKETS: [usize; 4] = [2, 4, 8, 24];

/// A [`WIDTH_BUCKETS`] width as a type, so whether the exact anchor's taped recording is
/// instantiated at it is decided at compile time ([`ExactAnchor`]).
struct Bucket<const N: usize, const M: usize>;

/// The exact anchor's taped half ([`agq_taped`] in its directional second-order mode) at
/// one bucket: the recording at an [`EXACT_ANCHOR_BUCKETS`] width, `None` at any other —
/// which [`subject_adjoint_gradient_at`] never reaches, since it pads an exact-anchor subject
/// to an exact-anchor bucket first.
trait ExactAnchor<const N: usize, const M: usize> {
    #[allow(clippy::too_many_arguments)]
    fn taped(
        model: &CompiledModel,
        subject: &Subject,
        lay: Option<&crate::stats::stacked_layout::StackedLayout>,
        template: &ModelParameters,
        params: &ModelParameters,
        x: &[f64],
        eta: &[f64],
        ctx: &LikCtx,
        so: &StackedOmega,
        pv: &ParT<f64>,
        p1: &[DualV<N, M>],
        grid: &[([f64; N], f64, [f64; N])],
        h_inner: DMatrix<f64>,
        active: usize,
        held: bool,
    ) -> Option<Vec<f64>>;
}

macro_rules! exact_anchor_buckets {
    (taped: $(($n:literal, $m:literal)),*; declined: $(($dn:literal, $dm:literal)),*) => {
        $(impl ExactAnchor<$n, $m> for Bucket<$n, $m> {
            fn taped(
                model: &CompiledModel,
                subject: &Subject,
                lay: Option<&crate::stats::stacked_layout::StackedLayout>,
                template: &ModelParameters,
                params: &ModelParameters,
                x: &[f64],
                eta: &[f64],
                ctx: &LikCtx,
                so: &StackedOmega,
                pv: &ParT<f64>,
                p1: &[DualV<$n, $m>],
                grid: &[([f64; $n], f64, [f64; $n])],
                h_inner: DMatrix<f64>,
                active: usize,
                held: bool,
            ) -> Option<Vec<f64>> {
                agq_taped::<$n, $m, $n, true>(
                    model, subject, lay, template, params, x, eta, ctx, so, pv, p1, grid,
                    h_inner, true, active, held,
                )
            }
        })*
        $(impl ExactAnchor<$dn, $dm> for Bucket<$dn, $dm> {
            fn taped(
                _: &CompiledModel,
                _: &Subject,
                _: Option<&crate::stats::stacked_layout::StackedLayout>,
                _: &ModelParameters,
                _: &ModelParameters,
                _: &[f64],
                _: &[f64],
                _: &LikCtx,
                _: &StackedOmega,
                _: &ParT<f64>,
                _: &[DualV<$dn, $dm>],
                _: &[([f64; $dn], f64, [f64; $dn])],
                _: DMatrix<f64>,
                _: usize,
                _: bool,
            ) -> Option<Vec<f64>> {
                None
            }
        })*
    };
}

exact_anchor_buckets!(
    taped: (2, 3), (4, 10), (8, 36), (24, 300);
    declined: (1, 1), (3, 6), (6, 21), (12, 78), (16, 136), (32, 528), (48, 1176)
);

/// The taped half of [`run_agq`]: the node recordings, the scalar-tape objective, the
/// recording at `η̂` along the seed directions, and the two sweeps. `M2`/`TH` is the
/// recording's second-order mode (directional under the exact anchor, none otherwise).
#[allow(clippy::too_many_arguments)]
fn agq_taped<const N: usize, const M: usize, const M2: usize, const TH: bool>(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    template: &ModelParameters,
    params: &ModelParameters,
    x: &[f64],
    eta: &[f64],
    ctx: &LikCtx,
    so: &StackedOmega,
    pv: &ParT<f64>,
    p1: &[DualV<N, M>],
    grid: &[([f64; N], f64, [f64; N])],
    h_inner: DMatrix<f64>,
    exact: bool,
    active: usize,
    held: bool,
) -> Option<Vec<f64>> {
    use crate::sens::rtape::DualH;
    let theta = &params.theta;
    let n_obs = subject.observations.len();
    let unit = unit_dirs::<N>();
    let vidx = |node: u32, comp: usize| {
        if TH {
            vadj_index_h::<N, M2>(node, comp)
        } else {
            vadj_index::<N, 1>(node, comp, 0)
        }
    };
    vtape_reset(if TH { 1 + 2 * N } else { N + 1 });
    let th = theta_leaves::<N, M2, TH>(theta_axis(model)?, theta);
    // `[covariate_nn]` outputs as leaves, as `run` does, for both recordings below.
    #[cfg(feature = "nn")]
    let nn_out = nn_output_leaves::<N, M2, TH>(model, subject, template, theta);
    #[cfg(feature = "nn")]
    let lv = ThetaLeaves::with_nn(&th, nn_out.as_ref().map(|(_, z)| z.as_slice()));
    #[cfg(not(feature = "nn"))]
    let lv = ThetaLeaves::theta(&th);
    // Node predictions at their own b_j, taped for their θ path (canonical tangents: G at
    // the node, for the linearisation in b).
    let node_preds: Vec<Vec<DualV<N, M2, TH>>> = grid
        .iter()
        .map(|(_, _, b)| {
            record(
                model,
                subject,
                lay,
                theta,
                b,
                &lv,
                &eta_leaves::<N, M2, TH>(b, &unit),
            )
            .filter(|p| p.len() == n_obs)
        })
        .collect::<Option<_>>()?;

    // ---- the objective on the scalar tape -------------------------------------------
    rtape_reset();
    let one = R::c(1.0);
    let fr: Vec<R> = p1.iter().map(|p| rinput(p.v)).collect();
    let gr: Vec<[R; N]> = p1
        .iter()
        .map(|p| std::array::from_fn(|k| rinput(p.g[k])))
        .collect();
    let qr: Vec<[R; M]> = p1
        .iter()
        .map(|p| std::array::from_fn(|t| if exact { rinput(p.h[t]) } else { R::c(p.h[t]) }))
        .collect();
    let er: [R; N] = std::array::from_fn(|k| rinput(eta[k]));
    let (pr, om) = tape_params(model, subject, pv, so)?;
    // The anchor H_a on the tape: the exact Hessian over a second-order dual whose
    // predictions carry (f, G, Q), or FOCEI's H̃.
    let mut h_a: Vec<Vec<R>> = if exact {
        let fh: Vec<DualH<N, M, R>> = (0..n_obs)
            .map(|j| DualH {
                v: fr[j],
                g: gr[j],
                h: qr[j],
            })
            .collect();
        let eh: [DualH<N, M, R>; N] = std::array::from_fn(|k| DualH {
            v: er[k],
            g: std::array::from_fn(|m| R::c(unit[k][m])),
            h: [R::c(0.0); M],
        });
        let l = inner_l(
            ctx,
            &fh,
            &eh,
            &pr.map(|x| DualH {
                v: x,
                g: [R::c(0.0); N],
                h: [R::c(0.0); M],
            }),
            None,
        )?;
        (0..N)
            .map(|a| (0..N).map(|b| l.hess(a, b)).collect())
            .collect()
    } else {
        crate::sens::focei_kernel::focei_htilde(ctx, &fr, &gr, &er, &pr)?
    };
    regularise(&mut h_a);
    // The same anchor fallback as `run_agq`, on the tape (both see the same values, so
    // both take the same branch).
    let lh = crate::sens::focei_kernel::chol(&h_a)
        .or_else(|| crate::sens::focei_kernel::chol(&pr.omega.inv))?;
    let half_logdet = R::c(0.5) * crate::sens::focei_kernel::chol_logdet(&lh);
    // Node prediction rows f_j(b) = f_j + G_j·(b − b_j), with the node values as leaves.
    let node_fr: Vec<Vec<R>> = node_preds
        .iter()
        .map(|ps| ps.iter().map(|p| rinput(p.v)).collect())
        .collect();
    let a = if grid.is_empty() {
        inner_l(ctx, &fr, &er, &pr, None)? + half_logdet
    } else {
        let mut terms: Vec<R> = Vec::with_capacity(grid.len());
        for ((z, c, b0), (ps, fl)) in grid.iter().zip(node_preds.iter().zip(&node_fr)) {
            // b = η̂ + √2·L⁻ᵀz on the tape.
            let mut step = [R::c(0.0); N];
            for i in (0..N).rev() {
                let mut acc = R::c(z[i]);
                for j in (i + 1)..N {
                    acc = acc - lh[j][i] * step[j];
                }
                step[i] = acc / lh[i][i];
            }
            let b: [R; N] =
                std::array::from_fn(|k| er[k] + R::c(std::f64::consts::SQRT_2) * step[k]);
            let fb: Vec<R> = ps
                .iter()
                .zip(fl)
                .map(|(p, &f)| {
                    f + <R as crate::sens::num::PkNum>::wsum3(
                        0.0,
                        (0..N).map(|k| (R::c(p.g[k]), b[k] - R::c(b0[k]), one)),
                    )
                })
                .collect();
            terms.push(R::c(*c) - inner_l(ctx, &fb, &b, &pr, None)?);
        }
        let mx = terms.iter().map(|t| t.v).fold(f64::NEG_INFINITY, f64::max);
        let sum = <R as crate::sens::num::PkNum>::wsum3(
            0.0,
            terms.iter().map(|&t| (one, (t - R::c(mx)).exp(), one)),
        );
        R::c(0.5 * active as f64 * std::f64::consts::PI.ln()) + half_logdet - (R::c(mx) + sum.ln())
    };
    let lg = inner_grad(ctx, &fr, &gr, &er, &pr, None)?;

    // ---- sweep 1: A's adjoints; under the exact anchor, S's eigen-directions --------
    let s1 = rsweep::<1>(&[a]).to_vec();
    let ra = |adj: &[f64], x: R| x.idx().map_or(0.0, |i| adj[i]);
    let dirs: [[f64; N]; N] = if TH {
        // Every row's Q-adjoint is c_j·S: the eigenvectors of the largest one are S's.
        let adj_q = |j: usize| {
            DMatrix::from_fn(N, N, |k, l| {
                let v = ra(&s1, qr[j][crate::sens::vtape::tri(k, l)]);
                if k == l {
                    v
                } else {
                    0.5 * v
                }
            })
        };
        // A subject with no observation rows has no Q-adjoint at all (its objective is the
        // prior alone), so any basis is exact; the unit one is taken rather than declining.
        match (0..n_obs).max_by(|&a, &b| adj_q(a).norm().total_cmp(&adj_q(b).norm())) {
            Some(j_max) => {
                let eig = adj_q(j_max).symmetric_eigen();
                std::array::from_fn(|k| std::array::from_fn(|m| eig.eigenvectors[(k, m)]))
            }
            None => unit,
        }
    } else {
        unit
    };
    // The recording at η̂, along `dirs` (tangent m is the derivative along column m).
    let et = eta_leaves::<N, M2, TH>(eta, &dirs);
    let p2 = record(model, subject, lay, theta, eta, &lv, &et)?;
    if p2.len() != n_obs {
        return None;
    }
    // `weights`: whether the θ adjoint is wanted (sweep 2), so the networks' backward
    // passes run once.
    let sweep = |adj: &[f64], weights: bool| -> Option<([f64; N], Vec<f64>)> {
        let vadj = vadj_init::<1>();
        let mut top = 0usize;
        let mut put = |node: u32, comp: usize, v: f64| {
            if node != NONE && v != 0.0 {
                top = top.max(node as usize + 1);
                vadj[vidx(node, comp)] += v;
            }
        };
        for j in 0..n_obs {
            let p = &p2[j];
            put(p.i, 0, ra(adj, fr[j]));
            for m in 0..N {
                put(
                    p.i,
                    1 + m,
                    (0..N).map(|k| ra(adj, gr[j][k]) * dirs[k][m]).sum(),
                );
            }
            if TH {
                for m in 0..N {
                    let mut s = 0.0;
                    for k in 0..N {
                        for l in 0..N {
                            let t = crate::sens::vtape::tri(k, l);
                            let w = if k == l { 1.0 } else { 0.5 };
                            s += dirs[k][m] * dirs[l][m] * w * ra(adj, qr[j][t]);
                        }
                    }
                    put(p.i, 1 + N + m, s);
                }
            }
        }
        for (ps, fl) in node_preds.iter().zip(&node_fr) {
            for (p, &f) in ps.iter().zip(fl) {
                put(p.i, 0, ra(adj, f));
            }
        }
        let vs = if TH {
            vsweep_h::<N, M2>(top)
        } else {
            vsweep::<N, 1>(top)
        };
        let at = |d: &DualV<N, M2, TH>| if d.i == NONE { 0.0 } else { vs[vidx(d.i, 0)] };
        #[cfg_attr(not(feature = "nn"), allow(unused_mut))]
        let mut theta_adj: Vec<f64> = th.iter().map(at).collect();
        #[cfg(feature = "nn")]
        if let (Some((red, z)), true) = (&nn_out, weights) {
            add_nn_weight_adjoint(model, red, z, theta, at, &mut theta_adj)?;
        }
        #[cfg(not(feature = "nn"))]
        let _ = weights;
        Some((
            std::array::from_fn(|k| ra(adj, er[k]) + at(&et[k])),
            theta_adj,
        ))
    };
    let (grad_eta, _) = sweep(&s1, false)?;
    let grad_l: Vec<f64> = lg.iter().map(|l| l.v).collect();
    // `held`: past a declined mode solve, `w = 0` makes `Ψ = A`, the held-EBE score
    // (`subject_adjoint_held_ebe_score`).
    let w = match mode_solve(&h_inner, &DVector::from_column_slice(&grad_eta), &grad_l) {
        Some(w) => w,
        None if held => DVector::zeros(N),
        None => return None,
    };

    // ---- sweep 2: Ψ = A − wᵀ∇_η l(η̂) ----------------------------------------------
    let psi =
        a - <R as crate::sens::num::PkNum>::wsum3(0.0, (0..N).map(|k| (R::c(w[k]), lg[k], one)));
    let s2 = rsweep::<1>(&[psi]).to_vec();
    let (_, theta_adj) = sweep(&s2, true)?;
    let psi_of = |x: R| -> f64 { ra(&s2, x) };
    Some(packed_gradient(
        model,
        template,
        params,
        x.len(),
        &theta_adj,
        &psi_of,
        &pr,
        &om,
        so,
        pv,
    ))
}

#[path = "ode_adjoint_hessian.rs"]
mod hessian;
pub(crate) use hessian::{
    cov_width_declines, subject_adjoint_cov_block_at, subject_adjoint_cov_diag_at,
    subject_adjoint_cov_hessian_at, subject_adjoint_eta_dx_at,
};
#[cfg(test)]
pub(crate) use hessian::{subject_adjoint_cov_hessian, subject_adjoint_eta_dx};

#[cfg(test)]
#[path = "ode_adjoint_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "ode_adjoint_phenomena_tests.rs"]
mod phenomena_tests;
