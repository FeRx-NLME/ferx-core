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
use crate::sens::focei_kernel::{data_l, foce_f, focei_f_grad, inner_grad, inner_l, LikCtx, ParT};
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

/// [`model_in_scope`] for the FOCE/FOCEI outer gradient, which also serves a model whose
/// non-Gaussian endpoints all have an exact channel (`run` folds their terms into the
/// objective, over the BSV block under IOV). AGQ and the EBE predictor keep [`model_in_scope`].
pub(crate) fn outer_model_in_scope(model: &CompiledModel) -> bool {
    (recorder_in_scope(model) || endpoint_only(model))
        && (!model.has_non_gaussian()
            || crate::estimation::inner_optimizer::non_gaussian_endpoints_served(model))
}

/// A model with no structural part — only non-Gaussian endpoints (`[markov_model]`,
/// `[binary_model]` alone): no Gaussian rows to record, so no recorder is needed.
fn endpoint_only(model: &CompiledModel) -> bool {
    model.ode_spec.is_none()
        && !model.is_algebraic()
        && model.pk_indices.is_empty()
        && model.has_non_gaussian()
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
            let addr = template as *const ModelParameters as usize;
            if let Some(e) = watch.iter_mut().find(|e| e.0 == addr) {
                e.1 += 1;
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

/// Test watch: per watched template `(address, count)` — the [`PackedPoint`]s built for it, on
/// **any** thread (a per-subject unpack would run on the rayon workers, where a thread-local
/// counter on the caller cannot see it). A list, so concurrent tests watching their own
/// templates neither count each other's unpacks nor overwrite each other's slot
/// ([`watch_packed_points`]).
#[cfg(test)]
pub(crate) static PACKED_POINT_WATCH: std::sync::Mutex<Vec<(usize, usize)>> =
    std::sync::Mutex::new(Vec::new());

/// The [`PackedPoint`]s built for `template` while `f` runs (see [`PACKED_POINT_WATCH`]).
#[cfg(test)]
pub(crate) fn watch_packed_points(template: &ModelParameters, f: &dyn Fn()) -> usize {
    let addr = template as *const ModelParameters as usize;
    PACKED_POINT_WATCH.lock().unwrap().push((addr, 0));
    f();
    let mut watch = PACKED_POINT_WATCH.lock().unwrap();
    let i = watch.iter().position(|e| e.0 == addr).expect("watched");
    watch.swap_remove(i).1
}

/// [`Scoped`] for a subject, or `None` outside the adjoint's scope: the shared prelude of
/// the gradient and the covariance Hessian. A subject carrying non-Gaussian records is
/// served when every one has an exact channel (`non_gaussian_terms_supported`): the
/// objectives fold the term in, over the BSV block under IOV.
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
    let supported = if !subject.obs_records.is_empty() {
        non_gaussian_terms_supported(model, subject)
    } else {
        likelihood_supported(model, subject, &params)
    };
    if !supported {
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

/// The fixed-EBE objective and its exact packed gradient at `η̂` with the linearisation
/// `H` held as given: `F(θ, Ω, σ; η̂, H)` — the analytic kernel's FOCE (`interaction =
/// false`, `R⁰` at `f(η = 0)` where the variance is f-dependent) or FOCEI objective, with
/// `H` and `η̂` constants. This is the score the outer optimizer's fixed-EBE paths
/// difference (GN, trust region, the S-matrix, the held-EBE salvage), taken exactly: one
/// scalar-tape sweep of the objective gives its adjoints in the predictions, `Ω` and `σ`,
/// and the predictions' θ-derivatives are contracted with them by one recording and one
/// reverse sweep each ([`fixed_eta_theta_contractions`]). Every feature the kernel scores —
/// M3 censoring, a dense residual, FREM rows, a residual-error η, a residual magnitude —
/// is covered by construction. `None` outside the kernel's or the recorders' scope, under
/// IOV (its linearisation is not a held `H`), or on a shape mismatch.
pub(crate) fn subject_fixed_ebe_score_at(
    model: &CompiledModel,
    subject: &Subject,
    pt: &PackedPoint<'_>,
    eta_hat: &[f64],
    h: &DMatrix<f64>,
    interaction: bool,
) -> Option<(f64, Vec<f64>)> {
    if model.n_kappa > 0 {
        // `eta_hat` is the stacked `[η̂, κ̂]`; `h` holds the BSV columns only.
        let sc = scope(model, subject, pt, eta_hat)?;
        if h.nrows() != subject.observations.len() || h.ncols() != model.n_eta {
            return None;
        }
        fn go_iov<const N: usize, const M: usize>(
            sc: &Scoped,
            subject: &Subject,
            pt: &PackedPoint<'_>,
            h: &DMatrix<f64>,
            interaction: bool,
        ) -> Option<(f64, Vec<f64>)> {
            fixed_ebe_score_iov::<N>(sc, subject, pt, h, interaction)
        }
        return by_bucket!(sc.bucket, go_iov, (&sc, subject, pt, h, interaction));
    }
    let sc = scope(model, subject, pt, eta_hat)?;
    if h.nrows() != subject.observations.len() || h.ncols() != sc.width {
        return None;
    }
    fn go<const N: usize, const M: usize>(
        sc: &Scoped,
        subject: &Subject,
        pt: &PackedPoint<'_>,
        eta_hat: &[f64],
        h: &DMatrix<f64>,
        interaction: bool,
    ) -> Option<(f64, Vec<f64>)>
    where
        Bucket<N, M>: ExactAnchor<N, M>,
    {
        fixed_ebe_score::<N>(sc, subject, pt, eta_hat, h, interaction)
    }
    by_bucket!(sc.bucket, go, (&sc, subject, pt, eta_hat, h, interaction))
}

fn fixed_ebe_score<const N: usize>(
    sc: &Scoped,
    subject: &Subject,
    pt: &PackedPoint<'_>,
    eta_hat: &[f64],
    h: &DMatrix<f64>,
    interaction: bool,
) -> Option<(f64, Vec<f64>)> {
    let (model, params, template) = (sc.model, sc.params, pt.template);
    let theta = &params.theta;
    let n_obs = subject.observations.len();
    // A subject's non-Gaussian records: `(q, ∇q, ∇²q)` at `η̂`. Beside them production scores
    // the conditional variance under either interaction setting (`focei_f_grad_ng`), so
    // FOCE's population variance point is not read.
    let n_bsv = model.n_eta;
    let ng = if subject.obs_records.is_empty() {
        None
    } else {
        if n_bsv > N {
            return None;
        }
        Some(non_gaussian_eta_jets2(
            model,
            subject,
            theta,
            &eta_hat[..n_bsv],
        )?)
    };
    let use_pop = ng.is_none() && !interaction && model.error_spec.has_f_dependent_variance();
    // FOCE's variance point f(η = 0) is needed before the objective can be scored; its
    // values come from one plain evaluation, its θ-derivatives from its own recording below.
    let fpop = if use_pop {
        let zeros = vec![0.0; sc.eta.len()];
        let f0 = record::<f64>(
            sc.pop_model,
            subject,
            None,
            theta,
            &zeros,
            &ThetaLeaves::theta(theta),
            &zeros,
        )?;
        Some((f0.len() == n_obs).then_some(f0)?)
    } else {
        None
    };
    let ctx = LikCtx::new(model, subject, params);
    let so = stacked_omega::<N>(model, subject, template, params, sc.width);
    let mut pv = ParT::of(model, subject, params);
    pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(so.chol.clone())?;

    // The objective on the scalar tape (`H` and `η̂` constants), scored inside the η̂
    // recording's weights callback on that recording's own prediction values — one
    // recording serves both the value and its θ-derivatives. Everything but the θ block
    // is assembled there; the θ block is linear in `theta_adj` and added afterwards.
    let scored: std::cell::RefCell<Option<(f64, Vec<f64>, Vec<f64>)>> = Default::default();
    let ng_adj: std::cell::RefCell<Option<(f64, Vec<f64>, Vec<Vec<f64>>)>> = Default::default();
    let score = |f: &[f64]| -> Option<Vec<Vec<f64>>> {
        rtape_reset();
        let fr: Vec<R> = f.iter().map(|&v| rinput(v)).collect();
        let gr: Vec<[R; N]> = (0..n_obs)
            .map(|j| std::array::from_fn(|k| R::c(if k < sc.width { h[(j, k)] } else { 0.0 })))
            .collect();
        let er: [R; N] = std::array::from_fn(|k| R::c(sc.eta.get(k).copied().unwrap_or(0.0)));
        let pr0: Option<Vec<R>> = fpop
            .as_ref()
            .map(|p| p.iter().map(|&v| rinput(v)).collect());
        let (pr, om) = tape_params(model, subject, &pv, &so)?;
        let ngr = ng.as_ref().map(|(q, g, hq)| {
            ng_tape_inputs::<N>(&NgJets {
                q: *q,
                g: g.clone(),
                h: (0..g.len())
                    .map(|k| (0..g.len()).map(|l| hq[(k, l)]).collect())
                    .collect(),
                t: Vec::new(),
            })
        });
        let fval = match ngr.as_ref() {
            Some(ngr) => {
                crate::sens::focei_kernel::focei_f_grad_ng(
                    &ctx,
                    &fr,
                    &gr,
                    &er,
                    &pr,
                    ngr,
                    interaction,
                )?
                .0
            }
            None => objective::<R, N>(&ctx, interaction, &fr, &gr, &er, pr0.as_deref(), &pr)?.0,
        };
        let sadj = rsweep::<1>(&[fval]);
        let psi_of = |x: R| -> f64 { x.idx().map_or(0.0, |i| sadj[i]) };
        if let Some(ngr) = ngr.as_ref() {
            *ng_adj.borrow_mut() = Some(ng_adjoints(ngr, n_bsv, psi_of));
        }
        let wf: Vec<f64> = fr.iter().map(|&x| psi_of(x)).collect();
        let wp: Vec<f64> = pr0
            .as_ref()
            .map(|p0| p0.iter().map(|&x| psi_of(x)).collect())
            .unwrap_or_default();
        let base = packed_gradient(
            model,
            template,
            params,
            pt.x.len(),
            &[],
            &psi_of,
            &pr,
            &om,
            &so,
            &pv,
        );
        *scored.borrow_mut() = Some((fval.val(), base, wp));
        Some(vec![wf])
    };
    // An endpoint-only subject has no prediction to record: its θ enter through the term.
    let theta_mode = if n_obs == 0 && ng.is_some() {
        score(&[])?;
        vec![0.0; theta.len()]
    } else {
        fixed_eta_theta_contractions(model, subject, theta, eta_hat, score)?.pop()?
    };
    let (nll, mut grad, wp) = scored.into_inner()?;
    let mut theta_adj = theta_mode;
    if let Some((a_q, a_g, a_h)) = ng_adj.into_inner() {
        let d = ng_theta_block(model, subject, theta, &eta_hat[..n_bsv], a_q, &a_g, &a_h)?;
        for (t, dv) in theta_adj.iter_mut().zip(d) {
            *t += dv;
        }
    }
    if use_pop {
        let zeros_eta = vec![0.0; eta_hat.len()];
        let at_pop =
            fixed_eta_theta_contractions(sc.pop_model, subject, theta, &zeros_eta, |_| {
                Some(vec![wp])
            })?
            .pop()?;
        for (a, b) in theta_adj.iter_mut().zip(at_pop) {
            *a += b;
        }
    }
    for (m, &a) in theta_adj.iter().enumerate() {
        grad[m] += a * crate::estimation::sens_outer_gradient::theta_dx_chain(template, theta, m);
    }
    (nll.is_finite() && grad.iter().all(|g| g.is_finite())).then_some((nll, grad))
}

/// [`fixed_ebe_score`] under IOV: production's held objective (`foce_subject_nll_iov`) holds
/// the BSV columns of `H` and `b̂ = [η̂, κ̂]`, and rebuilds the κ columns `∂f/∂κ` at `b̂` from
/// θ. So the predictions are recorded with their κ-tangents on the vector tape (θ leaves,
/// `b` free), the kernel's objective over the stacked system is scored on the scalar tape
/// with the BSV columns constant and the κ columns as inputs, and one sweep seeded with both
/// adjoints carries `∂f/∂θ` and `∂²f/∂κ∂θ` to the θ leaves together.
fn fixed_ebe_score_iov<const N: usize>(
    sc: &Scoped,
    subject: &Subject,
    pt: &PackedPoint<'_>,
    h: &DMatrix<f64>,
    interaction: bool,
) -> Option<(f64, Vec<f64>)> {
    let (model, params, template) = (sc.model, sc.params, pt.template);
    let theta = &params.theta;
    let lay = sc.layout.as_ref();
    let eta = &sc.eta[..];
    let n_bsv = model.n_eta;
    vtape_reset(N + 1);
    let axis = theta_axis(model).or_else(|| endpoint_only(model).then_some(theta.len()))?;
    let th = theta_leaves::<N, 0, false>(axis, theta);
    #[cfg(feature = "nn")]
    let nn_out = nn_output_leaves::<N, 0, false>(model, subject, template, theta);
    #[cfg(feature = "nn")]
    let lv = ThetaLeaves::with_nn(&th, nn_out.as_ref().map(|(_, z)| z.as_slice()));
    #[cfg(not(feature = "nn"))]
    let lv = ThetaLeaves::theta(&th);
    let unit = unit_dirs::<N>();
    let et: [DualV<N, 0>; N] = std::array::from_fn(|k| DualV::free(eta[k], unit[k], []));
    let preds = if subject.observations.is_empty() && !subject.obs_records.is_empty() {
        Vec::new()
    } else {
        record(model, subject, lay, theta, eta, &lv, &et)?
    };
    let n_obs = preds.len();
    if n_obs != subject.observations.len() {
        return None;
    }
    // Non-Gaussian records: `(q, ∇q, ∇²q)` on the BSV block, production's conditional
    // variance under either interaction setting (as `fixed_ebe_score`).
    let ng = if subject.obs_records.is_empty() {
        None
    } else {
        if n_bsv > N {
            return None;
        }
        Some(non_gaussian_eta_jets2(
            model,
            subject,
            theta,
            &eta[..n_bsv],
        )?)
    };
    let pop = if ng.is_none() && !interaction && model.error_spec.has_f_dependent_variance() {
        let p0 = record(
            sc.pop_model,
            subject,
            lay,
            theta,
            &[0.0; N],
            &lv,
            &[DualV::constant(0.0); N],
        )?;
        (p0.len() == n_obs).then_some(p0)?.into()
    } else {
        None
    };
    let ctx = LikCtx::new(model, subject, params);
    let so = stacked_omega::<N>(model, subject, template, params, sc.width);
    let mut pv = ParT::of(model, subject, params);
    pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(so.chol.clone())?;

    rtape_reset();
    let fr: Vec<R> = preds.iter().map(|p| rinput(p.v)).collect();
    let gr: Vec<[R; N]> = preds
        .iter()
        .enumerate()
        .map(|(j, p)| {
            std::array::from_fn(|k| {
                if k < n_bsv {
                    R::c(h[(j, k)])
                } else if k < sc.width {
                    rinput(p.g[k])
                } else {
                    R::c(0.0)
                }
            })
        })
        .collect();
    let pr0: Option<Vec<R>> = pop
        .as_ref()
        .map(|p0: &Vec<DualV<N, 0>>| p0.iter().map(|p| rinput(p.v)).collect());
    let er: [R; N] = std::array::from_fn(|k| R::c(eta[k]));
    let (pr, om) = tape_params(model, subject, &pv, &so)?;
    let ngr = ng.as_ref().map(|(q, g, hq)| {
        ng_tape_inputs::<N>(&NgJets {
            q: *q,
            g: g.clone(),
            h: (0..g.len())
                .map(|k| (0..g.len()).map(|l| hq[(k, l)]).collect())
                .collect(),
            t: Vec::new(),
        })
    });
    let fval = match ngr.as_ref() {
        Some(ngr) => {
            crate::sens::focei_kernel::focei_f_grad_ng(&ctx, &fr, &gr, &er, &pr, ngr, interaction)?
                .0
        }
        None => objective::<R, N>(&ctx, interaction, &fr, &gr, &er, pr0.as_deref(), &pr)?.0,
    };
    let sadj = rsweep::<1>(&[fval]);
    let psi_of = |x: R| -> f64 { x.idx().map_or(0.0, |i| sadj[i]) };
    let ng_adj = ngr.as_ref().map(|ngr| ng_adjoints(ngr, n_bsv, psi_of));

    let vadj = vadj_init::<1>();
    let mut top = 0usize;
    let mut seed = |node: u32, comp: usize, v: f64| {
        if node != NONE && v != 0.0 {
            top = top.max(node as usize + 1);
            vadj[vadj_index::<N, 1>(node, comp, 0)] += v;
        }
    };
    for (j, p) in preds.iter().enumerate() {
        seed(p.i, 0, psi_of(fr[j]));
        for k in n_bsv..sc.width {
            seed(p.i, 1 + k, psi_of(gr[j][k]));
        }
    }
    if let (Some(p0), Some(r0)) = (pop.as_ref(), pr0.as_ref()) {
        for (p, &x) in p0.iter().zip(r0) {
            seed(p.i, 0, psi_of(x));
        }
    }
    let adj = vsweep::<N, 1>(top);
    let at = |l: &DualV<N, 0>| {
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
    if let Some((a_q, a_g, a_h)) = ng_adj {
        let d = ng_theta_block(model, subject, theta, &eta[..n_bsv], a_q, &a_g, &a_h)?;
        for (t, dv) in theta_adj.iter_mut().zip(d) {
            *t += dv;
        }
    }
    let grad = packed_gradient(
        model,
        template,
        params,
        pt.x.len(),
        &theta_adj,
        &psi_of,
        &pr,
        &om,
        &so,
        &pv,
    );
    let nll = fval.val();
    (nll.is_finite() && grad.iter().all(|g| g.is_finite())).then_some((nll, grad))
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
    // Not inlined: every bucket's recording lives on the stack, and inlined together they
    // made this dispatcher a 2.4 MB frame that every call — every subject, every width —
    // probed page by page. Out of line, a call pays for its own width's frame only.
    #[inline(never)]
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

/// [`mode_solve`], with a declined stationarity gate judged again by `inner` — the gradient
/// the inner search converged on ([`inner_search_gradient`]) — which is evaluated only
/// then: the definite branch never reads a gradient.
fn mode_solve_judged<const N: usize>(
    h: &DMatrix<f64>,
    b: &DVector<f64>,
    grad_l: &[f64],
    inner: impl FnOnce() -> Option<[f64; N]>,
) -> Option<DVector<f64>> {
    mode_solve(h, b, grad_l).or_else(|| mode_solve(h, b, &inner()?))
}

/// `∇_η l(η̂)` by the engine the inner search converged with ([`exact_inner_gradient`]'s
/// η-space recording), for [`mode_solve`]'s stationarity gate. The identity's own recording
/// may have been lifted from PK space, and both are exact derivatives of the solve — but of
/// two different floating-point computations, which part by the solver's tolerance where
/// the tangents are not error-controlled: an explicit method stepping at its stability
/// edge amplifies their rounding as it amplifies the values' (measured on a two-compartment
/// RK45 subject, `3.5e-5` of the derivative row at the last observation, values bit for
/// bit). Whether `η̂` is stationary is a statement about the gradient the search drove to
/// zero, so on a declined gate it is judged by that gradient. `None` outside its scope.
#[allow(clippy::too_many_arguments)]
fn inner_search_gradient<const N: usize>(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
    ctx: &LikCtx,
    pv: &ParT<f64>,
    frozen: Option<&[f64]>,
) -> Option<[f64; N]> {
    let p = canonical_jets1::<N>(model, subject, lay, theta, eta)?;
    let f: Vec<f64> = p.iter().map(|q| q.v).collect();
    let g: Vec<[f64; N]> = p.iter().map(|q| q.g).collect();
    let e: [f64; N] = std::array::from_fn(|k| eta[k]);
    inner_grad(ctx, &f, &g, &e, pv, frozen)
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
/// on either through the compiled init program (`init_seed`).
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
        // The subject's one parameter set makes its predictions a function of the live PK
        // parameters alone: solved once in PK space and lifted, where `T`'s order is served.
        let preds = match pk_space_predictions(model, subject, &pk, &by_slot, PkEngine::Ode) {
            Some(p) => p,
            None => {
                let init_state = init_seed::<T>(ode, &params_dual)
                    .unwrap_or_else(|| vec![T::from_f64(0.0); ode.n_states]);
                integrate_seeded_duals(model, subject, &pk.values, &params_dual, &init_state)?
            }
        };
        return apply_expression_scale(model, subject, th, et, &by_slot, preds);
    }
    if !event_walk_subject_supported(model, subject) {
        return None;
    }
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let (pk_at_dose, pk_at_obs, pk_at_pk_only, pk_at_reset) =
        seed_tvcov_snapshots::<T>(model, subject, uses_time, |cov, time| {
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
    let preds = match pk_space_predictions(model, subject, &pk, &by_slot, PkEngine::ClosedForm) {
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
    /// Test switch: take the PK-space path past the work rule, for the twins that pin the
    /// lift's jets on fixtures the rule would leave on the generic path.
    pub(super) static PK_SPACE_FORCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Recordings the PK-space path served, so a test can see it ran.
    pub(super) static PK_SPACE_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Which engine a PK-space solve runs: the static closed form, or the static ODE walk.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PkEngine {
    ClosedForm,
    Ode,
}

/// A static subject's predictions in **PK space**: each observation as a function `c_j(p)`
/// of the subject's `P` live PK parameters (the slots `by_slot` holds on `T`), solved once
/// over a tape-free `P`-direction jet of the order `T` observes, then lifted onto `T`
/// through its Taylor polynomial in `δp = p − p₀`, one node per observation.
///
/// The static subject's parameters are one set for every observation, so `f_j(η, θ) =
/// c_j(p(η, θ))`, and the polynomial of order `k = T::JET_ORDER` reproduces every jet of
/// `c_j ∘ p` of total order ≤ `k` — everything `T` can observe. Its cost grows with `P` and
/// writes `n_obs` nodes; the generic path runs the engine itself over `T`, every operation
/// carrying `T`'s jet and tape, so its cost grows with `n_η` and with every pass over it
/// (#2093 review: the forward provider's PK-space kernels).
///
/// - order 2 (the adjoint's recordings, [`PkNum::TAYLOR2_EXACT`]): the jet is a free
///   `DualV<P, P(P+1)/2>`;
/// - order 3 or 4 (the covariance step's nested duals, the Laplace exact anchor's taped
///   Hessian): a [`crate::sens::taylor::Taylor`].
///
/// `None` — the caller takes the generic path — where the lift's jet type is more work per
/// operation than `T` ([`PkNum::WORK`](crate::sens::num::PkNum::WORK), so the lift is taken
/// as the recording's directions outnumber the live slots or a tape would record the solve),
/// for a first-order tape-free recording, past the widest bucket of live slots or order 4,
/// when an analytic `init(...)` reads θ and η directly, or outside the engine's static scope.
fn pk_space_predictions<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    pk: &crate::types::PkParams,
    by_slot: &[Option<T>],
    engine: PkEngine,
) -> Option<Vec<T>> {
    #[cfg(test)]
    if PK_SPACE_OFF.with(|c| c.get()) {
        return None;
    }
    if !model.analytical_init.is_empty() {
        return None;
    }
    let live: Vec<(usize, T)> = by_slot
        .iter()
        .enumerate()
        .filter_map(|(s, d)| d.map(|d| (s, d)))
        .collect();
    let order = T::JET_ORDER;
    // A tape-free first-order recording (the inner loop's η-gradient) records nothing either
    // way, and runs the engine over about as many lanes: there the generic path is the
    // cheaper one. An order-2 type that carries no η-Hessian and meets no tape is one too.
    let second = T::ETA_HESSIAN || live.iter().any(|&(_, d)| d.is_taped());
    if order < 2 || (order == 2 && !second) || order > 4 || live.is_empty() {
        return None;
    }
    let n = live.len();
    // The lift runs the engine over its own jet type, once: taken when that type is no more
    // work per operation than the recording's. A taped η-recording over as many PK slots
    // gains (the tape is what the lift drops); an order-3 jet over more slots than `T` has
    // η-directions loses — `Taylor<4, 3>`'s 165-term product against a two-η directional
    // recording's 24, measured slower on a Michaelis–Menten Laplace fit.
    let lift_work = pk_space_work(order, n)?;
    #[cfg(test)]
    let forced = PK_SPACE_FORCE.with(|c| c.get());
    #[cfg(not(test))]
    let forced = false;
    if lift_work > T::WORK && !forced {
        return None;
    }
    #[cfg(test)]
    PK_SPACE_HITS.with(|c| c.set(c.get() + 1));
    // The jets depend on the subject's timeline, the live slots' values and the jet type
    // alone, so the covariance step's passes at one mode share one solve.
    let key = pk_space_key(model, subject, pk, &live, engine, order);
    macro_rules! taylor {
        ($p:literal, $ord:literal) => {{
            const L: usize = crate::sens::taylor::binom($p + $ord, $ord);
            type J = crate::sens::taylor::Taylor<$p, $ord, L>;
            let coefs = pk_space_cached(key, || {
                let jets =
                    pk_space_jets::<J>(model, subject, pk, &live, engine, |i, v| J::seed(v, i))?;
                Some(jets.iter().map(|q| q.c.to_vec()).collect())
            })?;
            lift_polynomial(&coefs, &live, crate::sens::taylor::table($p, $ord))
        }};
    }
    macro_rules! quad {
        ($p:literal, $mp:literal) => {{
            let unit = unit_dirs::<$p>();
            let t = crate::sens::taylor::table($p, 2);
            let coefs = pk_space_cached(key, || {
                let jets =
                    pk_space_jets::<DualV<$p, $mp>>(model, subject, pk, &live, engine, |i, v| {
                        DualV::free(v, unit[i], [0.0; $mp])
                    })?;
                // A second-order dual's derivatives as the normalised Taylor coefficients of
                // the order-2 table: `c_{e_i} = g_i`, `c_{e_i+e_j} = h_ij`, `c_{2e_i} = h_ii/2`.
                Some(
                    jets.iter()
                        .map(|q| {
                            t.exps
                                .iter()
                                .map(|e| {
                                    let nz: Vec<usize> = (0..$p).filter(|&v| e[v] > 0).collect();
                                    match (
                                        e.iter().map(|&x| x as usize).sum::<usize>(),
                                        nz.as_slice(),
                                    ) {
                                        (0, _) => q.v,
                                        (1, [i]) => q.g[*i],
                                        (2, [i]) => 0.5 * q.h[crate::sens::vtape::tri(*i, *i)],
                                        (2, [i, j]) => q.h[crate::sens::vtape::tri(*i, *j)],
                                        _ => unreachable!("degree ≤ 2"),
                                    }
                                })
                                .collect()
                        })
                        .collect(),
                )
            })?;
            lift_polynomial(&coefs, &live, t)
        }};
    }
    match (order, pk_space_bucket(order, n)?) {
        (2, 1) => quad!(1, 1),
        (2, 2) => quad!(2, 3),
        (2, 3) => quad!(3, 6),
        (2, 4) => quad!(4, 10),
        (2, 6) => quad!(6, 21),
        (2, 8) => quad!(8, 36),
        (3, 2) => taylor!(2, 3),
        (3, 4) => taylor!(4, 3),
        (3, 6) => taylor!(6, 3),
        (3, 8) => taylor!(8, 3),
        (4, 2) => taylor!(2, 4),
        (4, 4) => taylor!(4, 4),
        (4, 6) => taylor!(6, 4),
        _ => None,
    }
}

/// [`PkNum::WORK`](crate::sens::num::PkNum::WORK) of the jet type the lift solves `n` live
/// slots at `order` over: the second-order dual, or the Taylor type, at the dispatch bucket.
fn pk_space_work(order: usize, n: usize) -> Option<usize> {
    let p = pk_space_bucket(order, n)?;
    match order {
        // An untaped `DualV<P, P(P+1)/2>`: the value, the gradient and the packed Hessian.
        2 => Some(1 + p + p * (p + 1) / 2),
        3 | 4 => Some(crate::sens::taylor::mul_len(p, order)),
        _ => None,
    }
}

/// The live-slot width [`pk_space_predictions`] solves `n` live slots at for `order`: the
/// dispatch bucket, `None` past the widest.
fn pk_space_bucket(order: usize, n: usize) -> Option<usize> {
    match (order, n) {
        (2, 1..=4) => Some(n),
        (2, 5 | 6) | (3 | 4, 5 | 6) => Some(6),
        (2 | 3, 7 | 8) => Some(8),
        (3 | 4, 1 | 2) => Some(2),
        (3 | 4, 3 | 4) => Some(4),
        _ => None,
    }
}

/// What a PK-space solve depends on, by exact bits: the model (its address and solver
/// options), the subject (its address and its timeline — dose records and observation
/// times, which with the engine fix the solve),
/// the live slots and every parameter value, the engine and the order (with the slot
/// count, the jet type).
fn pk_space_key<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    pk: &crate::types::PkParams,
    live: &[(usize, T)],
    engine: PkEngine,
    order: usize,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (
        model as *const CompiledModel as usize,
        subject as *const Subject as usize,
        engine == PkEngine::Ode,
        order,
    )
        .hash(&mut h);
    if let Some(ode) = model.ode_spec.as_ref() {
        let o = ode.effective_solver_opts();
        (
            o.reltol.to_bits(),
            o.abstol.to_bits(),
            o.max_steps,
            o.method.as_str(),
        )
            .hash(&mut h);
    }
    for &(s, _) in live {
        s.hash(&mut h);
    }
    let bits = |h: &mut std::collections::hash_map::DefaultHasher, xs: &[f64]| {
        for x in xs {
            x.to_bits().hash(h);
        }
    };
    bits(&mut h, &pk.values);
    bits(&mut h, &subject.obs_times);
    bits(&mut h, &subject.pk_only_times);
    bits(&mut h, &subject.reset_times);
    subject.obs_cmts.hash(&mut h);
    for d in &subject.doses {
        bits(&mut h, &[d.time, d.amt, d.rate, d.duration, d.ii]);
        (d.cmt_raw(), d.ss).hash(&mut h);
    }
    h.finish()
}

thread_local! {
    /// The last few PK-space solves' coefficients by [`pk_space_key`].
    #[allow(clippy::type_complexity)]
    static PK_SPACE_CACHE: std::cell::RefCell<Vec<(u64, std::rc::Rc<Vec<Vec<f64>>>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// `solve()`'s coefficients, or the cached ones for `key`.
fn pk_space_cached(
    key: u64,
    solve: impl FnOnce() -> Option<Vec<Vec<f64>>>,
) -> Option<std::rc::Rc<Vec<Vec<f64>>>> {
    let hit = PK_SPACE_CACHE.with(|c| {
        c.borrow()
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v.clone())
    });
    if hit.is_some() {
        return hit;
    }
    let v = std::rc::Rc::new(solve()?);
    PK_SPACE_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() == 4 {
            c.remove(0);
        }
        c.push((key, v.clone()));
    });
    Some(v)
}

/// The subject's predictions over the jet `J`, each live slot `i` seeded by `seed(i, p_i)`
/// (inert directions past the live count are never seeded), on `engine`.
fn pk_space_jets<J: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    pk: &crate::types::PkParams,
    live: &[(usize, impl Copy)],
    engine: PkEngine,
    seed: impl Fn(usize, f64) -> J,
) -> Option<Vec<J>> {
    let at = |slot: usize| live.iter().position(|&(s, _)| s == slot);
    match engine {
        PkEngine::ClosedForm => crate::sens::provider::analytic_static_predictions::<J>(
            model,
            subject,
            pk,
            |slot| at(slot).map(|i| seed(i, pk.values[slot])),
            &[],
            &[],
        ),
        PkEngine::Ode => {
            let ode = model.ode_spec.as_ref()?;
            let params: Vec<J> = pk
                .values
                .iter()
                .enumerate()
                .map(|(s, &v)| at(s).map_or(J::from_f64(v), |i| seed(i, v)))
                .collect();
            let init = crate::sens::ode_provider::init_seed::<J>(ode, &params)
                .unwrap_or_else(|| vec![J::from_f64(0.0); ode.n_states]);
            crate::sens::ode_provider::integrate_seeded_duals(
                model, subject, &pk.values, &params, &init,
            )
        }
    }
}

/// Each observation's Taylor polynomial `Σ_α c_α δp^α` (normalised coefficients in
/// `table`'s graded order) evaluated on `T`, `δp_i = p_i − p_i.val()` the live slots: the
/// monomials once per subject, then one linear node per observation. Monomials of an inert
/// padding direction are skipped (their `δp` is zero).
fn lift_polynomial<T: VmNum>(
    coefs: &[Vec<f64>],
    live: &[(usize, T)],
    table: &crate::sens::taylor::Table,
) -> Option<Vec<T>> {
    let n = live.len();
    let dp: Vec<T> = live
        .iter()
        .map(|&(_, d)| d - T::from_f64(d.val()))
        .collect();
    let m = table.exps.len();
    // `keep[k]`: monomial `k` reads only live directions.
    let keep: Vec<bool> = table
        .exps
        .iter()
        .map(|e| e.iter().skip(n).all(|&x| x == 0))
        .collect();
    let mut basis: Vec<T> = vec![T::from_f64(1.0); m];
    for k in 1..m {
        if keep[k] {
            let (b, v) = table.parent[k];
            basis[k] = if b == 0 {
                dp[v as usize]
            } else {
                basis[b as usize] * dp[v as usize]
            };
        }
    }
    let idx: Vec<usize> = (1..m).filter(|&k| keep[k]).collect();
    let used: Vec<T> = idx.iter().map(|&k| basis[k]).collect();
    let mut w = vec![0.0; idx.len()];
    coefs
        .iter()
        .map(|c| {
            for (o, &k) in w.iter_mut().zip(&idx) {
                *o = c[k];
            }
            if !(c[0].is_finite() && w.iter().all(|x| x.is_finite())) {
                return None;
            }
            // Finite coefficients can still overflow against the basis at a far trial
            // point; the generic recording then gets the call, as it would a solve whose
            // jets diverged.
            let f = T::lincomb(T::from_f64(c[0]), 1.0, &w, &used);
            (f.val().is_finite() && f.jets_finite()).then_some(f)
        })
        .collect()
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
        model,
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
    let mut doses: Vec<_> = (0..subject.doses.len())
        .map(|k| {
            at_record(
                crate::pk::dose_row_occasion(subject, k),
                subject.dose_cov(k),
                subject.dose_param_time(k),
            )
        })
        .collect();
    // An `ADDL`-implied dose's lag is its parent row's (#1909), value and dual alike:
    // `inherit_parent_dose_attrs{,_g}` over the `(PkParams, Vec<T>)` pair, so the copies
    // come from the same `parent_dose_attr_copies`.
    for (k, p, slot) in crate::pk::parent_dose_attr_copies(model, subject) {
        doses[k].0.values[slot] = doses[p].0.values[slot];
        debug_assert!(
            slot < doses[p].1.len() && slot < doses[k].1.len(),
            "dual seed too short for parent-dose slot {slot}"
        );
        if slot < doses[p].1.len() && slot < doses[k].1.len() {
            doses[k].1[slot] = doses[p].1[slot];
        }
    }
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

/// [`objective`] of a subject carrying non-Gaussian records: `F` is production's one form
/// for both interaction settings (`focei_f_grad_ng`: conditional variance, `½·c̃ᵀc̃` under
/// FOCEI only, the term folded in); `∇_η l` is the inner objective's the EBE is stationary
/// for — under FOCE the frozen-variance one at `fpop` (#1722) — plus `∇q`.
pub(super) fn objective_ng<T: crate::sens::num::PkNum, const N: usize>(
    ctx: &LikCtx,
    interaction: bool,
    f: &[T],
    g: &[[T; N]],
    eta: &[T; N],
    fpop: Option<&[T]>,
    ng: &crate::sens::focei_kernel::NonGauss<T, N>,
    p: &ParT<T>,
) -> Option<(T, [T; N])> {
    let (fv, lg) = crate::sens::focei_kernel::focei_f_grad_ng(ctx, f, g, eta, p, ng, interaction)?;
    match fpop {
        Some(f0) if !interaction => {
            let mut frozen = inner_grad(ctx, f, g, eta, p, Some(f0))?;
            for k in 0..N {
                frozen[k] = frozen[k] + ng.g[k];
            }
            Some((fv, frozen))
        }
        _ => Some((fv, lg)),
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

/// The predictions and their η-Jacobian from one recording — [`exact_eta_jacobian`] with
/// the values it already computes.
fn eta_values_and_jacobian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(Vec<f64>, DMatrix<f64>)> {
    fn go<const N: usize, const M: usize>(
        model: &CompiledModel,
        subject: &Subject,
        lay: Option<&crate::stats::stacked_layout::StackedLayout>,
        theta: &[f64],
        eta: &[f64],
        w: usize,
    ) -> Option<(Vec<f64>, DMatrix<f64>)> {
        let p = canonical_jets1::<N>(model, subject, lay, theta, eta)?;
        Some((
            p.iter().map(|q| q.v).collect(),
            DMatrix::from_fn(p.len(), w, |j, k| p[j].g[k]),
        ))
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

/// The exact θ-derivatives of the predictions at a fixed `eta`, contracted with weights
/// the caller computes from them: `Σ_j w_j ∂f_j/∂θ_m` for every θ coordinate (raw θ, not
/// packed), by one recording on the vector tape — θ as leaves, η as plain values — and one
/// reverse sweep seeded with `w = weights(f)`. With `want_jac`, also every `∂f_j/∂θ_m`
/// (row-major `n_obs × n_theta`), from sweeps of eight observations at a time.
///
/// The fixed-η score of SAEM's M-step and of variational inference, where η is a draw
/// rather than a mode: the cost of one solve and one sweep however many θ there are, where
/// a difference quotient pays one solve per θ. `None` outside the recorders' scope (the
/// caller keeps its own path), on a recording whose length is not the subject's
/// observation count, or when `weights` declines.
pub(crate) struct FixedEtaTheta {
    pub(crate) grad: Vec<f64>,
    pub(crate) jac: Option<Vec<f64>>,
}

pub(crate) fn fixed_eta_theta_adjoint(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    want_jac: bool,
    weights: impl FnOnce(&[f64]) -> Option<Vec<f64>>,
) -> Option<FixedEtaTheta> {
    let mut n_obs = 0;
    let rows = fixed_eta_theta_contractions(model, subject, theta, eta, |f| {
        n_obs = f.len();
        let mut rows = vec![weights(f)?];
        if want_jac {
            rows.extend((0..f.len()).map(|j| {
                let mut e = vec![0.0; f.len()];
                e[j] = 1.0;
                e
            }));
        }
        Some(rows)
    })?;
    let mut rows = rows.into_iter();
    let grad = rows.next()?;
    let jac = want_jac.then(|| rows.flatten().collect::<Vec<f64>>());
    debug_assert!(jac.as_ref().is_none_or(|j| j.len() == n_obs * grad.len()));
    Some(FixedEtaTheta { grad, jac })
}

/// `Σ_j w_j ∂f_j/∂θ_m` for each weight row `w` the caller computes from the predictions
/// `f` at a fixed `eta`, for every θ coordinate (raw θ; a θ no recording reads has a zero
/// derivative). One recording on the vector tape — θ as leaves, η as plain values — and one
/// reverse sweep per eight rows, however many θ there are: the exact replacement for a
/// difference quotient of the predictions in θ. `None` outside the recorders' scope, on a
/// recording whose length is not the subject's observation count, when `weights`
/// declines, or on a non-finite result.
pub(crate) fn fixed_eta_theta_contractions(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    weights: impl FnOnce(&[f64]) -> Option<Vec<Vec<f64>>>,
) -> Option<Vec<Vec<f64>>> {
    // One inert tangent: η is fixed, so no η-direction is carried.
    type T = DualV<1, 0>;
    if !recorder_in_scope(model) {
        return None;
    }
    vtape_reset(2);
    let th = theta_leaves::<1, 0, false>(theta_axis(model)?, theta);
    let et: Vec<T> = eta.iter().map(|&e| DualV::constant(e)).collect();
    let preds = record(
        model,
        subject,
        None,
        theta,
        eta,
        &ThetaLeaves::theta(&th),
        &et,
    )?;
    let n_obs = preds.len();
    if n_obs != subject.observations.len() {
        return None;
    }
    // A FREM covariate pseudo-observation predicts `θ_i + η_e` (the recorders return the
    // PK prediction there, as every caller of theirs patches): its derivative is the unit
    // vector on `θ_i`, added after the sweep rather than seeded.
    let frem: Vec<Option<usize>> = (0..n_obs)
        .map(|j| {
            let fc = model.frem_config.as_ref()?;
            let ft = subject.fremtype.get(j).filter(|&&ft| ft != 0)?;
            fc.fremtype_to_indices.get(ft).map(|&(ti, _)| ti)
        })
        .collect();
    let f: Vec<f64> = preds
        .iter()
        .enumerate()
        .map(|(j, p)| match (frem[j], model.frem_config.as_ref()) {
            (Some(ti), Some(fc)) => {
                let ei = fc.fremtype_to_indices[&subject.fremtype[j]].1;
                theta.get(ti).copied().unwrap_or(0.0) + eta.get(ei).copied().unwrap_or(0.0)
            }
            _ => p.v,
        })
        .collect();
    let rows = weights(&f)?;
    if rows.iter().any(|w| w.len() != n_obs) {
        return None;
    }
    const L: usize = 8;
    let at = |a: &[f64], l: &T, lane: usize| -> f64 {
        if l.i == NONE {
            0.0
        } else {
            a[(l.i as usize * 2) * L + lane]
        }
    };
    let mut out = Vec::with_capacity(rows.len());
    for chunk in rows.chunks(L) {
        let adj = vadj_init::<L>();
        let mut top = 0;
        for (lane, w) in chunk.iter().enumerate() {
            for (p, (&wj, fr)) in preds.iter().zip(w.iter().zip(&frem)) {
                if p.i != NONE && fr.is_none() && wj != 0.0 {
                    adj[vadj_index::<1, L>(p.i, 0, lane)] += wj;
                    top = top.max(p.i as usize + 1);
                }
            }
        }
        let a = vsweep::<1, L>(top);
        for (lane, w) in chunk.iter().enumerate() {
            let mut g: Vec<f64> = (0..theta.len())
                .map(|m| th.get(m).map_or(0.0, |l| at(a, l, lane)))
                .collect();
            for (fr, &wj) in frem.iter().zip(w) {
                if let Some(gm) = fr.and_then(|ti| g.get_mut(ti)) {
                    *gm += wj;
                }
            }
            out.push(g);
        }
    }
    out.iter().flatten().all(|v| v.is_finite()).then_some(out)
}

/// Whether [`ode_cumhaz_hazard_g`] serves `subject`: an autonomous RHS (so `h` is the RHS at
/// the saved state, with no clock anchor), no steady-state dose (the dual equilibration has
/// no accumulator handling, #1210), absorption forcings the dual walk lifts, and the static
/// walk's own scope (no time-varying covariates, resets or lag). The routing predicates ask
/// this, so a subject the walk declines is reported on FD rather than labelled analytic.
#[cfg(feature = "survival")]
pub(crate) fn ode_hazard_walk_in_scope(model: &CompiledModel, subject: &Subject) -> bool {
    let Some(ode) = model.ode_spec.as_ref() else {
        return false;
    };
    ode.rhs_program.is_some()
        && ode.indiv_param_program.is_some()
        && ode.input_rate.iter().all(|f| f.kind.supported_over_dual())
        && (static_hazard_walk(model, subject) || ss_hazard_walk(model, subject))
}

/// The static walk serves the subject's state solve: no steady-state dose (the walk has no
/// run-in).
#[cfg(feature = "survival")]
fn static_hazard_walk(model: &CompiledModel, subject: &Subject) -> bool {
    !subject.has_ss_doses() && static_walk_subject_supported(model, subject)
}

/// A subject the static walk would serve but for its steady-state dose: one parameter
/// snapshot for every event (no time-varying covariates, no `TIME` in the parameters, no
/// IOV), fixed unlagged doses — so the event walk, which carries the dual SS run-in, solves
/// it with that one snapshot at every event.
#[cfg(feature = "survival")]
fn ss_hazard_walk(model: &CompiledModel, subject: &Subject) -> bool {
    subject.has_periodic_ss_dose()
        && model.n_kappa == 0
        && !subject.has_tv_covariates()
        && !crate::parser::model_parser::compiled_model_uses_time_builtin(model)
        && !model.has_lagtime()
        && !model.has_route_absorption_lag()
        && subject.all_doses_fixed()
        && !(model.has_bioavailability() && subject.has_rate_defined_infusion())
        && event_walk_subject_supported(model, subject)
}

/// The augmented system's cumulative hazard `H(t)` and hazard `h(t) = dH/dt` for the
/// injected accumulator `chz_state`, at `times`, on any VM number type — the joint
/// PK-TTE likelihood's curves with the derivatives the θ and η seeds carry. The
/// solve is the static dual walk (`integrate_g`, saving at `times` instead of the
/// observations) and `h` the RHS at the saved state. The value path
/// (`survival::ode_cumhaz_hazard`) integrates densely with per-dose reads; the two agree to
/// the solver tolerance, which is all a derivative needs.
///
/// `None` — the caller keeps its stencil — outside the static walk's scope or under a
/// steady-state dose (the dual equilibration has no accumulator handling, #1210). A RHS
/// reading model time (`TIME`, `TAD`, `TAFD`) is integrated by the walk's clock, and `h(t)`
/// reads the same anchors the value path does.
#[cfg(feature = "survival")]
pub(crate) fn ode_cumhaz_hazard_g<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    chz_state: usize,
    theta: &[T],
    eta: &[T],
    times: &[f64],
) -> Option<(Vec<T>, Vec<T>)> {
    let ode = model.ode_spec.as_ref()?;
    let program = ode.rhs_program.as_ref()?;
    let DualStates {
        states,
        params,
        first_dose_time,
    } = ode_states_at_g(model, subject, theta, eta, times)?;
    let mut du = vec![T::from_f64(0.0); ode.n_states];
    let (mut vars, mut stack) = (Vec::new(), Vec::new());
    let mut cum = Vec::with_capacity(times.len());
    let mut haz = Vec::with_capacity(times.len());
    for (st, &t) in states.iter().zip(times) {
        cum.push(st[chz_state]);
        // `h` reads the RHS's clocks at `t` as the value path does (`rhs_ext_params_at`):
        // `TAFD` from the first dose record, `TAD` from the last arrival by `t`. The walk's
        // scope has no compartment lag, so the arrivals are the records and the anchor is a
        // constant.
        let tad_anchor = crate::ode::predictions::tad_anchor_for(&subject.doses, &[], t);
        super::eval_rhs_anchored::<T>(
            program,
            st,
            &params,
            t,
            first_dose_time,
            T::from_f64(tad_anchor),
            &mut du,
            &mut vars,
            &mut stack,
        );
        haz.push(du[chz_state]);
    }
    Some((cum, haz))
}

/// [`ode_states_at_g`]'s result: the ODE state at each requested time, with the parameter
/// vector and first-dose time the solve ran under.
#[cfg(feature = "survival")]
pub(crate) struct DualStates<T> {
    pub(crate) states: Vec<Vec<T>>,
    pub(crate) params: Vec<T>,
    pub(crate) first_dose_time: f64,
}

/// The subject's ODE state at `times` (sorted, non-decreasing) on any VM number type: the
/// static dual walk (`integrate_g`) saving at `times` instead of the observations — what a
/// state-driven endpoint (an ODE-accumulated hazard, a drug-driven CTMM intensity) reads.
/// `None` outside [`ode_hazard_walk_in_scope`], under a compartment lag, or when the walk
/// does not return one state per time.
#[cfg(feature = "survival")]
pub(crate) fn ode_states_at_g<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[T],
    eta: &[T],
    times: &[f64],
) -> Option<DualStates<T>> {
    let ode = model.ode_spec.as_ref()?;
    let program = ode.rhs_program.as_ref()?;
    let prog = ode.indiv_param_program.as_ref()?;
    if !ode_hazard_walk_in_scope(model, subject) || times.is_empty() {
        return None;
    }
    let eta_v: Vec<f64> = eta.iter().map(|e| e.val()).collect();
    let theta_v: Vec<f64> = theta.iter().map(|t| t.val()).collect();
    let pk = (model.pk_param_fn)(&theta_v, &eta_v, &subject.covariates, 0.0);
    if pk.values[crate::types::PK_IDX_LAGTIME].abs() > 1e-12 {
        return None;
    }
    let by_slot = taped_params(
        model,
        prog,
        pk.values.len(),
        &ThetaLeaves::theta(theta),
        eta,
        &subject.covariates,
    );
    let params: Vec<T> = by_slot
        .iter()
        .zip(&pk.values)
        .map(|(d, &v)| d.unwrap_or(T::from_f64(v)))
        .collect();
    let init =
        super::init_seed::<T>(ode, &params).unwrap_or_else(|| vec![T::from_f64(0.0); ode.n_states]);
    let dose_f_bio: Vec<T> = subject
        .doses
        .iter()
        .map(|d| params[super::f_bio_slot(ode, d.cmt_raw())])
        .collect();
    let first_dose_time = subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min);
    let forcings = super::PreparedForcings::shared(ode, &params, super::prepare_dual_forcing::<T>);
    // The walk saves at the subject's observation times: hand it the requested times.
    let mut at_times = subject.clone();
    at_times.obs_times = times.to_vec();
    let states = if static_hazard_walk(model, subject) {
        super::integrate_g::<T>(
            program,
            ode.n_states,
            &at_times,
            ode,
            &forcings,
            &params,
            &dose_f_bio,
            &init,
            first_dose_time,
            &ode.effective_solver_opts(),
        )?
    } else {
        // A steady-state subject: the event walk, every event under the one snapshot (the
        // requested times read the subject's covariates, as its records do).
        at_times.obs_covariates.clear();
        at_times.obs_raw_times.clear();
        let snap = |n: usize| vec![params.clone(); n];
        super::integrate_tvcov_g::<T>(
            program,
            ode,
            ode.n_states,
            &at_times,
            &snap(subject.doses.len()),
            &snap(times.len()),
            &snap(subject.pk_only_times.len()),
            &snap(subject.reset_times.len()),
            &dose_f_bio,
            &init,
            first_dose_time,
            &[],
            &[],
            &ode.effective_solver_opts(),
        )
    };
    // A non-finite state declines: the static walk anchors `TAD` by a fold over the doses
    // at or before each segment, so a hazard reading it before the first dose integrates a
    // `NaN` the value path (anchored at the first arrival) does not.
    (states.len() == times.len() && states.iter().flatten().all(|v| v.val().is_finite())).then_some(
        DualStates {
            states,
            params,
            first_dose_time,
        },
    )
}

/// The fixed-η data objective `½·data_l` — `−log p(y | f(θ, η))` up to its constant, the
/// conditional NLL SAEM's M-step and VI score at a draw `η` — with its exact gradient in
/// raw θ and raw σ. The analytic kernel's data term on the scalar tape (σ as leaves, the
/// serial ρ as a leaf on its θ, θ itself where the likelihood reads it directly: FREM rows
/// and a residual magnitude), one sweep for its adjoints in the predictions and the
/// parameters, and the predictions' θ-derivatives contracted with them by one recording and
/// one reverse sweep ([`fixed_eta_theta_contractions`]). M3 censoring, a dense or serial
/// residual and FREM rows are scored by construction; a subject's TTE, binary and CTMM records
/// add their own term's exact θ-gradient ([`non_gaussian_theta_score`]). `eta` is the
/// recordings' vector (the stacked `[η, κ…]` under IOV). `None` outside the recorders' or the
/// kernel's scope, or on a non-finite result.
pub(crate) fn fixed_eta_data_score(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>, Vec<f64>)> {
    if subject.obs_records.is_empty() {
        return gaussian_fixed_eta_data_score(model, subject, theta, sigma, eta);
    }
    // A subject with non-Gaussian records: its Gaussian rows by the kernel, plus the
    // records' own term over the BSV η, each exact.
    let (q, dq) = non_gaussian_theta_score(model, subject, theta, eta.get(..model.n_eta)?)?;
    let (v, mut d_theta, d_sigma) = if subject.observations.is_empty() {
        (0.0, vec![0.0; theta.len()], vec![0.0; sigma.len()])
    } else {
        gaussian_fixed_eta_data_score(model, subject, theta, sigma, eta)?
    };
    for (d, e) in d_theta.iter_mut().zip(&dq) {
        *d += e;
    }
    Some((v + q, d_theta, d_sigma))
}

/// [`fixed_eta_data_score`]'s Gaussian (and M3) rows: the kernel's data term only.
fn gaussian_fixed_eta_data_score(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>, Vec<f64>)> {
    if model.n_eta > MAX_WIDTH || model.is_sde() {
        return None;
    }
    let mut params = model.default_params.clone();
    params.theta = theta.to_vec();
    params.sigma.values = sigma.to_vec();
    params.residual_correlations = model.residual_correlations.clone();
    let n_obs = subject.observations.len();
    let ctx = LikCtx::new(model, subject, &params);
    let pv = ParT::of(model, subject, &params);
    // The data term on the scalar tape, scored inside the recording's weights callback on
    // its own prediction values: one recording gives the value and the θ-derivatives.
    let scored: std::cell::RefCell<Option<(f64, Vec<f64>, Vec<f64>)>> = Default::default();
    let d_theta_f = fixed_eta_theta_contractions(model, subject, theta, eta, |f| {
        rtape_reset();
        let mut pr: ParT<R> = pv.map(R::c);
        pr.sigma = pv.sigma.iter().map(|&x| rinput(x)).collect();
        pr.serial_rho = pv.serial_rho.map(rinput);
        if theta_read_directly(model) {
            pr.theta = pv.theta.iter().map(|&x| rinput(x)).collect();
        }
        pr.ruv_mult = match model.ruv_magnitude.as_ref().filter(|rm| rm.is_active()) {
            Some(rm) => Some(
                (0..n_obs)
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
        let fr: Vec<R> = f.iter().map(|&v| rinput(v)).collect();
        let er: Vec<R> = eta.iter().map(|&v| R::c(v)).collect();
        let nll = R::c(0.5) * data_l(&ctx, &fr, &er, &pr, None)?;
        let sadj = rsweep::<1>(&[nll]);
        let adj = |x: R| -> f64 { x.idx().map_or(0.0, |i| sadj[i]) };
        // θ read directly (FREM rows, a residual magnitude), and the serial ρ's θ.
        let mut d_direct: Vec<f64> = pr.theta.iter().map(|&t| adj(t)).collect();
        if let (Some(sr), Some(rs)) = (pr.serial_rho, model.residual_serial.as_ref()) {
            if let Some(d) = d_direct.get_mut(rs.rho_theta) {
                *d += adj(sr);
            }
        }
        let d_sigma: Vec<f64> = pr.sigma.iter().map(|&x| adj(x)).collect();
        let wf: Vec<f64> = fr.iter().map(|&x| adj(x)).collect();
        *scored.borrow_mut() = Some((nll.val(), d_direct, d_sigma));
        Some(vec![wf])
    })?
    .pop()?;
    let (v, d_direct, d_sigma) = scored.into_inner()?;
    let mut d_theta = d_theta_f;
    for (d, e) in d_theta.iter_mut().zip(&d_direct) {
        *d += e;
    }
    (v.is_finite() && d_theta.iter().chain(&d_sigma).all(|g| g.is_finite()))
        .then_some((v, d_theta, d_sigma))
}

/// `∂(½·data_l)/∂η` at fixed θ and σ — the gradient of the conditional data NLL
/// `−log p(y | f(θ, η))` in the random effects (the stacked `[η, κ…]` under IOV), which VI
/// differences its ELBO's data term by. The predictions' η-Jacobian is the inner search's
/// own recording ([`exact_eta_jacobian`]); the kernel's data term on the scalar tape gives
/// its adjoints in the predictions and in η itself (a residual-error η's scale, FREM rows),
/// so M3 censoring and a dense or serial residual are scored by construction. `None`
/// outside the recorders' or the kernel's scope, or on a non-finite result.
pub(crate) fn fixed_theta_data_eta_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: &[f64],
    eta: &[f64],
) -> Option<Vec<f64>> {
    if !likelihood_supported(model, subject, &model.default_params) {
        return None;
    }
    let mut params = model.default_params.clone();
    params.theta = theta.to_vec();
    params.sigma.values = sigma.to_vec();
    params.residual_correlations = model.residual_correlations.clone();
    let n_obs = subject.observations.len();
    // One recording: the predictions' values and their η-Jacobian.
    let (f, jac) = eta_values_and_jacobian(model, subject, theta, eta)?;
    if f.len() != n_obs || jac.nrows() != n_obs || jac.ncols() != eta.len() {
        return None;
    }
    let ctx = LikCtx::new(model, subject, &params);
    let pr: ParT<R> = ParT::of(model, subject, &params).map(R::c);
    rtape_reset();
    let fr: Vec<R> = f.iter().map(|&v| rinput(v)).collect();
    let er: Vec<R> = eta.iter().map(|&v| rinput(v)).collect();
    let data = data_l(&ctx, &fr, &er, &pr, None)?;
    let sadj = rsweep::<1>(&[R::c(0.5) * data]);
    let adj = |x: R| -> f64 { x.idx().map_or(0.0, |i| sadj[i]) };
    let g: Vec<f64> = (0..eta.len())
        .map(|k| adj(er[k]) + (0..n_obs).map(|j| adj(fr[j]) * jac[(j, k)]).sum::<f64>())
        .collect();
    g.iter().all(|v| v.is_finite()).then_some(g)
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
        // An endpoint-only subject has no Gaussian rows to record.
        let p = if subject.observations.is_empty() && !subject.obs_records.is_empty() {
            Vec::new()
        } else {
            canonical_jets::<N, M>(model, subject, lay, &params.theta, eta)?
        };
        let ctx = LikCtx::new(model, subject, params);
        let mut pv = ParT::of(model, subject, params);
        pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(
            stacked_omega::<N>(model, subject, params, params, w).chol,
        )?;
        let f: Vec<f64> = p.iter().map(|q| q.v).collect();
        let g: Vec<[f64; N]> = p.iter().map(|q| q.g).collect();
        let q: Vec<[f64; M]> = p.iter().map(|q| q.h).collect();
        let e: [f64; N] = std::array::from_fn(|k| eta[k]);
        let h =
            crate::sens::focei_kernel::exact_inner_hessian_kernel(&ctx, &f, &g, &q, &e, &pv, w)?;
        let h = DMatrix::from_fn(w, w, |a, b| h[a][b]);
        h.iter().all(|v| v.is_finite()).then_some(h)
    }
    let (model, w, eta, lay) = resolve(model, subject, &params.theta, eta, None)?;
    // IOV scores FREM rows by a sentinel, not the kernel's `l` (as `exact_htilde` and
    // `exact_inner_gradient_in` decline).
    let iov_mismatch = model.n_kappa > 0 && subject.fremtype.iter().any(|&ft| ft > 0);
    if iov_mismatch {
        return None;
    }
    // A subject's non-Gaussian records add their term's exact η-Hessian.
    let ng = if subject.obs_records.is_empty() {
        if !likelihood_supported(model, subject, params) {
            return None;
        }
        None
    } else {
        if !non_gaussian_terms_supported(model, subject) {
            return None;
        }
        Some(non_gaussian_eta_hessian(
            model,
            subject,
            &params.theta,
            &eta[..model.n_eta],
        )?)
    };
    let mut h = by_bucket!(
        eta.len(),
        go,
        (model, subject, lay.as_deref(), params, &eta, w)
    )?;
    if let Some(q) = ng {
        let n = q.nrows();
        let mut blk = h.view_mut((0, 0), (n, n));
        blk += q;
    }
    Some(h)
}

/// Whether the adjoint's kernels serve a subject's non-Gaussian records: every one has an
/// exact channel (`inner_optimizer::non_gaussian_eta_grad_served`), and the model has no SDE
/// and at most [`WIDTH_BUCKETS`]' widest random effects. The terms read BSV η only, so under
/// IOV they occupy the leading block of the stacked `[η, κ]`.
fn non_gaussian_terms_supported(model: &CompiledModel, subject: &Subject) -> bool {
    model.n_eta <= WIDTH_BUCKETS[WIDTH_BUCKETS.len() - 1]
        && !model.is_sde()
        && crate::estimation::inner_optimizer::non_gaussian_eta_grad_served(model, subject)
}

/// [`crate::survival::non_gaussian_eta_hessian`], `None` without the `survival` feature.
fn non_gaussian_eta_hessian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<DMatrix<f64>> {
    non_gaussian_eta_jets2(model, subject, theta, eta).map(|(_, _, h)| h)
}

/// [`crate::survival::non_gaussian_eta_jets2`], `None` without the `survival` feature.
fn non_gaussian_eta_jets2(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>, DMatrix<f64>)> {
    #[cfg(feature = "survival")]
    return crate::survival::non_gaussian_eta_jets2(model, subject, theta, eta);
    #[cfg(not(feature = "survival"))]
    {
        let _ = (model, subject, theta, eta);
        None
    }
}

/// FOCEI's Gauss–Newton `H̃` at `eta` (the stacked `[η, κ]` under IOV), the kernel's on
/// one tape-free first-order recording, plus a subject's non-Gaussian term's exact
/// η-Hessian over the BSV block: the AGQ grid's `H̃` anchor.
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
        // An endpoint-only subject has no Gaussian rows to record.
        let p = if subject.observations.is_empty() && !subject.obs_records.is_empty() {
            Vec::new()
        } else {
            canonical_jets1::<N>(model, subject, lay, &params.theta, eta)?
        };
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
    if iov_mismatch {
        return None;
    }
    // A subject's non-Gaussian records add their term's exact η-Hessian, as FOCEI's own
    // objective adds it to `H̃` (`foce_subject_nll_cov_split`).
    let ng = if subject.obs_records.is_empty() {
        if !likelihood_supported(model, subject, params) {
            return None;
        }
        None
    } else {
        if !non_gaussian_terms_supported(model, subject) {
            return None;
        }
        Some(non_gaussian_eta_hessian(
            model,
            subject,
            &params.theta,
            &eta[..model.n_eta],
        )?)
    };
    let mut h = by_bucket!(
        eta.len(),
        go,
        (model, subject, lay.as_deref(), params, &eta, w)
    )?;
    if let Some(q) = ng {
        let n = q.nrows();
        let mut blk = h.view_mut((0, 0), (n, n));
        blk += q;
    }
    Some(h)
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

/// A subject's non-Gaussian term at `η` (θ fixed) with its η-derivatives to third order —
/// what the FOCE/FOCEI adjoint needs of it: `q`, `∇q` and `∇²q` enter the objective, and
/// their η-derivatives (`∇²q`, `∇³q`) the mode system `H w = ∇_η F`. From one evaluation on
/// a third-order Taylor type in η (`Taylor<P, 3>`, `P` the smallest bucket ≥ `n`).
pub(super) struct NgJets {
    q: f64,
    g: Vec<f64>,
    h: Vec<Vec<f64>>,
    t: Vec<Vec<Vec<f64>>>,
}

#[cfg(not(feature = "survival"))]
fn ng_jets3(_: &CompiledModel, _: &Subject, _: &[f64], _: &[f64]) -> Option<NgJets> {
    None
}

#[cfg(feature = "survival")]
fn ng_jets3(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<NgJets> {
    use crate::survival::{EtaScalar, FixedTheta, NonGaussianTerm};
    let n = eta.len();
    let term = NonGaussianTerm { model, subject };
    let f = FixedTheta { theta, term: &term };
    macro_rules! go {
        ($p:literal) => {{
            const L: usize = crate::sens::taylor::binom($p + 3, 3);
            type J = crate::sens::taylor::Taylor<$p, 3, L>;
            let et: Vec<J> = (0..n).map(|k| J::seed(eta[k], k)).collect();
            let y = f.eval::<J>(&et)?;
            let tab = crate::sens::taylor::table($p, 3);
            let mut jets = NgJets {
                q: y.c[0],
                g: vec![0.0; n],
                h: vec![vec![0.0; n]; n],
                t: vec![vec![vec![0.0; n]; n]; n],
            };
            for (i, e) in tab.exps.iter().enumerate() {
                if e.iter().skip(n).any(|&x| x > 0) {
                    continue;
                }
                // ∂^α f = α!·c_α (normalised coefficients), placed at every ordering of α.
                let fact: f64 = e
                    .iter()
                    .map(|&x| (1..=x as u32).product::<u32>() as f64)
                    .product();
                let d = y.c[i] * fact;
                let vars: Vec<usize> = e
                    .iter()
                    .enumerate()
                    .flat_map(|(v, &x)| std::iter::repeat_n(v, x as usize))
                    .collect();
                match vars.as_slice() {
                    [a] => jets.g[*a] = d,
                    [a, b] => {
                        jets.h[*a][*b] = d;
                        jets.h[*b][*a] = d;
                    }
                    [a, b, c] => {
                        for (x, y2, z) in [
                            (*a, *b, *c),
                            (*a, *c, *b),
                            (*b, *a, *c),
                            (*b, *c, *a),
                            (*c, *a, *b),
                            (*c, *b, *a),
                        ] {
                            jets.t[x][y2][z] = d;
                        }
                    }
                    _ => {}
                }
            }
            Some(jets)
        }};
    }
    let jets = match n {
        // No random effects: the value, with empty derivatives.
        0 | 1 => go!(1),
        2 => go!(2),
        3 | 4 => go!(4),
        5 | 6 => go!(6),
        7 | 8 => go!(8),
        _ => None,
    }?;
    (jets.q.is_finite() && jets.q < 1e19).then_some(jets)
}

/// The non-Gaussian term's `q`, `∇_η q` and symmetric `∇²_η q` as scalar-tape inputs (one
/// input per `k ≥ l`, mirrored), zero past the jets' width.
fn ng_tape_inputs<const N: usize>(j: &NgJets) -> crate::sens::focei_kernel::NonGauss<R, N> {
    let n = j.g.len();
    let rq = rinput(j.q);
    let rg: [R; N] = std::array::from_fn(|k| if k < n { rinput(j.g[k]) } else { R::c(0.0) });
    let mut rh = [[R::c(0.0); N]; N];
    for k in 0..n {
        for l in 0..=k {
            let x = rinput(j.h[k][l]);
            rh[k][l] = x;
            rh[l][k] = x;
        }
    }
    crate::sens::focei_kernel::NonGauss {
        q: rq,
        g: rg,
        h: rh,
    }
}

/// The adjoints `(a_q, a_g, a_h)` of [`ng_tape_inputs`]' inputs, `a_h[k][l]` for `k ≥ l`.
fn ng_adjoints<const N: usize>(
    ngr: &crate::sens::focei_kernel::NonGauss<R, N>,
    n: usize,
    adj: impl Fn(R) -> f64,
) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
    let a_g = (0..n).map(|k| adj(ngr.g[k])).collect();
    let a_h = (0..n)
        .map(|k| {
            (0..n)
                .map(|l| if l <= k { adj(ngr.h[k][l]) } else { 0.0 })
                .collect()
        })
        .collect();
    (adj(ngr.q), a_g, a_h)
}

/// `∂/∂η_m` of the term's tape inputs contracted with their adjoints:
/// `a_q·∂_m q + Σ_k a_g,k·∂²_km q + Σ_{k≥l} a_h,kl·∂³_klm q`, from the third-order jets.
fn ng_eta_contraction<const N: usize>(
    j: &NgJets,
    ngr: &crate::sens::focei_kernel::NonGauss<R, N>,
    adj: impl Fn(R) -> f64,
) -> Vec<f64> {
    let n = j.g.len();
    let (a_q, a_g, a_h) = ng_adjoints(ngr, n, adj);
    (0..n)
        .map(|m| {
            let mut acc = a_q * j.g[m];
            for k in 0..n {
                acc += a_g[k] * j.h[k][m];
                for l in 0..=k {
                    acc += a_h[k][l] * j.t[k][l][m];
                }
            }
            acc
        })
        .collect()
}

/// `a·∂q/∂θ` at `eta`: the non-Gaussian term's θ-gradient (η held), scaled by its adjoint
/// `a` — an AGQ node's share of the θ block.
fn ng_theta_score(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    a: f64,
) -> Option<Vec<f64>> {
    if a == 0.0 {
        return Some(vec![0.0; theta.len()]);
    }
    let (_, g) = non_gaussian_theta_score(model, subject, theta, eta)?;
    Some(g.into_iter().map(|v| a * v).collect())
}

/// A subject's non-Gaussian data term `q(θ, η)` (TTE, binary and CTMM summed, 1×) and its
/// exact θ-gradient at fixed `eta` (the BSV η): θ in chunks of 8 forward directions. `(0, 0)`
/// for a subject with no non-Gaussian records; `None` when a record has no exact channel or
/// on a non-finite result.
#[cfg(feature = "survival")]
pub(crate) fn non_gaussian_theta_score(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>)> {
    use crate::sens::rtape::DualT;
    use crate::survival::{NonGaussianTerm, SubjectTerm};
    const C: usize = 8;
    let mut out = vec![0.0; theta.len()];
    if subject.obs_records.is_empty() {
        return Some((0.0, out));
    }
    let term = NonGaussianTerm { model, subject };
    let et: Vec<DualT<C, f64>> = eta.iter().map(|&v| DualT { v, g: [0.0; C] }).collect();
    let mut q = 0.0;
    // A model without θ still has a value to report.
    for start in (0..theta.len().max(1)).step_by(C) {
        let th: Vec<DualT<C, f64>> = theta
            .iter()
            .enumerate()
            .map(|(m, &v)| {
                let mut g = [0.0; C];
                if (start..start + C).contains(&m) {
                    g[m - start] = 1.0;
                }
                DualT { v, g }
            })
            .collect();
        let y = term.eval(&th, &et)?;
        q = y.v;
        for (lane, m) in (start..(start + C).min(theta.len())).enumerate() {
            out[m] = y.g[lane];
        }
    }
    (q.is_finite()
        && q < crate::survival::REPEL_SENTINEL_FLOOR
        && out.iter().all(|v| v.is_finite()))
    .then_some((q, out))
}

#[cfg(not(feature = "survival"))]
pub(crate) fn non_gaussian_theta_score(
    _: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    _: &[f64],
) -> Option<(f64, Vec<f64>)> {
    subject
        .obs_records
        .is_empty()
        .then(|| (0.0, vec![0.0; theta.len()]))
}

/// `Σ_m` of the scalar-tape adjoints `(a_q, a_g, a_h)` of `(q, ∇_η q, ∇²_η q)` against their
/// θ-derivatives — the non-Gaussian term's share of the adjoint's θ block: `∂q/∂θ`,
/// `∂²q/∂θ∂η` and `∂³q/∂θ∂η²`, from forward passes of four θ directions each over a
/// second-order η-dual (`DualT<4, DualF<NB, MB>>`). `a_h[k][l]` is read for `k ≥ l` (each
/// symmetric entry is one tape input).
#[cfg(not(feature = "survival"))]
fn ng_theta_block(
    _: &CompiledModel,
    _: &Subject,
    _: &[f64],
    _: &[f64],
    _: f64,
    _: &[f64],
    _: &[Vec<f64>],
) -> Option<Vec<f64>> {
    None
}

#[cfg(feature = "survival")]
fn ng_theta_block(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    a_q: f64,
    a_g: &[f64],
    a_h: &[Vec<f64>],
) -> Option<Vec<f64>> {
    use crate::sens::rtape::DualT;
    use crate::sens::vfree::DualF;
    use crate::survival::{NonGaussianTerm, SubjectTerm};
    let n = eta.len();
    fn go<const NB: usize, const MB: usize>(
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        eta: &[f64],
        a_q: f64,
        a_g: &[f64],
        a_h: &[Vec<f64>],
    ) -> Option<Vec<f64>> {
        const C: usize = 4;
        type B<const NB: usize, const MB: usize> = DualF<NB, MB>;
        let n = eta.len();
        let cst = |v: f64| B::<NB, MB>::free(v, [0.0; NB], [0.0; MB]);
        let et: Vec<DualT<C, B<NB, MB>>> = (0..n)
            .map(|k| {
                let mut g = [0.0; NB];
                g[k] = 1.0;
                DualT {
                    v: B::<NB, MB>::free(eta[k], g, [0.0; MB]),
                    g: [cst(0.0); C],
                }
            })
            .collect();
        let term = NonGaussianTerm { model, subject };
        let mut out = vec![0.0; theta.len()];
        for start in (0..theta.len()).step_by(C) {
            let th: Vec<DualT<C, B<NB, MB>>> = theta
                .iter()
                .enumerate()
                .map(|(m, &v)| {
                    let mut g = [cst(0.0); C];
                    if (start..start + C).contains(&m) {
                        g[m - start] = cst(1.0);
                    }
                    DualT { v: cst(v), g }
                })
                .collect();
            let y = term.eval(&th, &et)?;
            for (lane, m) in (start..(start + C).min(theta.len())).enumerate() {
                let d = &y.g[lane];
                let mut acc = a_q * d.v;
                for k in 0..n {
                    acc += a_g[k] * d.g[k];
                    for l in 0..=k {
                        acc += a_h[k][l] * d.h[crate::sens::vtape::tri(k, l)];
                    }
                }
                out[m] = acc;
            }
        }
        out.iter().all(|v| v.is_finite()).then_some(out)
    }
    match n {
        // No random effects: no `∇q`/`∇²q` inputs, so `a_q·∂q/∂θ` alone, on a first-order
        // θ-dual rather than the second-order η-jets.
        0 => ng_theta_score(model, subject, theta, eta, a_q),
        1 => go::<1, 1>(model, subject, theta, eta, a_q, a_g, a_h),
        2 => go::<2, 3>(model, subject, theta, eta, a_q, a_g, a_h),
        3 | 4 => go::<4, 10>(model, subject, theta, eta, a_q, a_g, a_h),
        5..=8 => go::<8, 36>(model, subject, theta, eta, a_q, a_g, a_h),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
/// TEMPORARY diagnosis.
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
    // An endpoint-only model has no parameter program (and no Gaussian rows to record).
    let axis = theta_axis(model).or_else(|| endpoint_only(model).then_some(theta.len()))?;
    let th = theta_leaves::<N, M, false>(axis, theta);
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
    // A subject with no Gaussian rows (an endpoint-only one) has nothing to record.
    let preds = if subject.observations.is_empty() && !subject.obs_records.is_empty() {
        Vec::new()
    } else {
        record(model, subject, lay, theta, eta, &lv, &et)?
    };
    let n_obs = preds.len();
    if n_obs != subject.observations.len() {
        return None;
    }
    // A subject's non-Gaussian records (TTE, binary, CTMM): the term and its η-derivatives
    // to third order, for every step below. It reads BSV η only, so under IOV it lives on the
    // stacked system's leading block (the jets pad past it).
    let n_bsv = model.n_eta;
    let ng = if subject.obs_records.is_empty() {
        None
    } else {
        if n_bsv > N {
            return None;
        }
        Some(ng_jets3(model, subject, theta, &eta[..n_bsv])?)
    };
    // FOCE scores R⁰ at the population prediction f(η = 0): the same engine, on the same
    // θ leaves, with η held at zero (values only). Beside a non-Gaussian term production's
    // objective takes the conditional variance, but the EBE is still the frozen-variance
    // inner objective's stationary point (#1722), so `∇_η l` reads this point either way.
    let pop = if n_obs > 0 && !interaction && model.error_spec.has_f_dependent_variance() {
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
    let (grad_f, h, grad_l) = eta_derivatives(
        &ctx,
        interaction,
        &preds,
        pop_v.as_deref(),
        eta,
        &pv,
        ng.as_ref(),
    )?;
    let fz = if interaction { None } else { pop_v.as_deref() };
    let w = mode_solve_judged(&h, &grad_f, &grad_l, || {
        let mut g = inner_search_gradient::<N>(model, subject, lay, theta, eta, &ctx, &pv, fz)?;
        if let Some(j) = ng.as_ref() {
            for (gk, qk) in g.iter_mut().zip(&j.g) {
                *gk += qk;
            }
        }
        Some(g)
    })?;

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
    // The non-Gaussian term's value, η-gradient and (symmetric, one input per k ≥ l)
    // η-Hessian as tape inputs; their θ-derivatives are contracted with the adjoints in
    // step 4.
    let ng_r = ng.as_ref().map(ng_tape_inputs::<N>);
    let (fval, lg) = match ng_r.as_ref() {
        Some(ngr) => {
            objective_ng::<R, N>(&ctx, interaction, &fr, &gr, &er, pr0.as_deref(), ngr, &pr)?
        }
        None => objective::<R, N>(&ctx, interaction, &fr, &gr, &er, pr0.as_deref(), &pr)?,
    };
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
    if let Some(ngr) = ng_r.as_ref() {
        let n = n_bsv;
        let (a_q, a_g, a_h) = ng_adjoints(ngr, n, psi_of);
        let d = ng_theta_block(model, subject, theta, &eta[..n], a_q, &a_g, &a_h)?;
        for (t, dv) in theta_adj.iter_mut().zip(d) {
            *t += dv;
        }
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
    ng: Option<&NgJets>,
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
    let (fa, la) = match ng {
        // The term's value, η-gradient and η-Hessian with their own η-derivatives.
        Some(j) => {
            let pad = |k: usize| k < j.g.len();
            let at = |v: &[f64], k: usize| if pad(k) { v[k] } else { 0.0 };
            let ngd = crate::sens::focei_kernel::NonGauss::<D<N>, N> {
                q: DualT {
                    v: j.q,
                    g: std::array::from_fn(|k| at(&j.g, k)),
                },
                g: std::array::from_fn(|k| DualT {
                    v: at(&j.g, k),
                    g: std::array::from_fn(|l| if pad(k) { at(&j.h[k], l) } else { 0.0 }),
                }),
                h: std::array::from_fn(|k| {
                    std::array::from_fn(|l| DualT {
                        v: if pad(k) { at(&j.h[k], l) } else { 0.0 },
                        g: std::array::from_fn(|m| {
                            if pad(k) && pad(l) {
                                at(&j.t[k][l], m)
                            } else {
                                0.0
                            }
                        }),
                    })
                }),
            };
            objective_ng::<D<N>, N>(
                ctx,
                interaction,
                &fd,
                &gd,
                &ed,
                pd.as_deref(),
                &ngd,
                &pv.wrap::<N>(),
            )?
        }
        None => objective::<D<N>, N>(
            ctx,
            interaction,
            &fd,
            &gd,
            &ed,
            pd.as_deref(),
            &pv.wrap::<N>(),
        )?,
    };
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
    let theta = &params.theta;
    let ctx = LikCtx::new(model, subject, params);
    let so = stacked_omega::<N>(model, subject, template, params, active);
    let mut pv = ParT::of(model, subject, params);
    pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(so.chol.clone())?;
    let eta_a: [f64; N] = std::array::from_fn(|k| eta[k]);

    // ---- 1. values at η̂ (an untaped canonical pass): f, G, Q; the exact H and the anchor
    // An endpoint-only subject has no Gaussian rows to record.
    let p1 = if subject.observations.is_empty() && !subject.obs_records.is_empty() {
        Vec::new()
    } else {
        canonical_jets::<N, M>(model, subject, lay, theta, eta)?
    };
    let f1: Vec<f64> = p1.iter().map(|p| p.v).collect();
    let g1: Vec<[f64; N]> = p1.iter().map(|p| p.g).collect();
    let q1: Vec<[f64; M]> = p1.iter().map(|p| p.h).collect();
    // A subject's non-Gaussian records (TTE, binary, CTMM): the term's η-jets to third order
    // at η̂, over the BSV block (the leading one of the stacked `[η, κ]` under IOV). Both
    // anchors carry its `∇²q`, as production's do (`exact_inner_hessian`, `exact_htilde`).
    let n_bsv = model.n_eta;
    let ng = if subject.obs_records.is_empty() {
        None
    } else {
        if n_bsv > N {
            return None;
        }
        Some(ng_jets3(model, subject, theta, &eta[..n_bsv])?)
    };
    let h_exact = {
        let h = crate::sens::focei_kernel::exact_inner_hessian_kernel(
            &ctx, &f1, &g1, &q1, &eta_a, &pv, active,
        )?;
        let mut h = DMatrix::from_fn(N, N, |a, b| h[a][b]);
        if let Some(j) = ng.as_ref() {
            for a in 0..n_bsv {
                for b in 0..n_bsv {
                    h[(a, b)] += j.h[a][b];
                }
            }
        }
        h
    };
    let h_inner = h_exact.clone();
    let mut h_a: Vec<Vec<f64>> = if exact {
        (0..N)
            .map(|a| (0..N).map(|b| h_exact[(a, b)]).collect())
            .collect()
    } else {
        let mut h = crate::sens::focei_kernel::focei_htilde(&ctx, &f1, &g1, &eta_a, &pv)?;
        if let Some(j) = ng.as_ref() {
            for a in 0..n_bsv {
                for b in 0..n_bsv {
                    h[a][b] += j.h[a][b];
                }
            }
        }
        h
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
    // The term's value and η-gradient at every node, for its linearisation in `b` there.
    let ng = match ng {
        Some(jets) => {
            let nodes = grid
                .iter()
                .map(|(_, _, b)| ng_value_grad(model, subject, theta, &b[..n_bsv]))
                .collect::<Option<Vec<_>>>()?;
            Some(AgqNg { jets, nodes })
        }
        None => None,
    };
    if exact {
        <Bucket<N, M> as ExactAnchor<N, M>>::taped(
            model,
            subject,
            lay,
            template,
            params,
            x,
            eta,
            &ctx,
            &so,
            &pv,
            &p1,
            &grid,
            h_inner,
            active,
            held,
            ng.as_ref(),
        )
    } else {
        agq_taped::<N, M, M, false>(
            model,
            subject,
            lay,
            template,
            params,
            x,
            eta,
            &ctx,
            &so,
            &pv,
            &p1,
            &grid,
            h_inner,
            exact,
            active,
            held,
            ng.as_ref(),
        )
    }
}

/// A subject's non-Gaussian term on the AGQ grid: its third-order η-jets at `η̂` and its
/// value and η-gradient at each node.
struct AgqNg {
    jets: NgJets,
    nodes: Vec<(f64, Vec<f64>)>,
}

/// The non-Gaussian term's value and η-gradient at `eta` (θ fixed).
fn ng_value_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>)> {
    #[cfg(feature = "survival")]
    {
        use crate::survival::{eta_gradient_by_bucket, FixedTheta, NonGaussianTerm};
        let term = NonGaussianTerm { model, subject };
        eta_gradient_by_bucket(eta, &FixedTheta { theta, term: &term })
    }
    #[cfg(not(feature = "survival"))]
    {
        let _ = (model, subject, theta, eta);
        None
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
        ng: Option<&AgqNg>,
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
                ng: Option<&AgqNg>,
            ) -> Option<Vec<f64>> {
                agq_taped::<$n, $m, $n, true>(
                    model, subject, lay, template, params, x, eta, ctx, so, pv, p1, grid,
                    h_inner, true, active, held, ng,
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
                _: Option<&AgqNg>,
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
    ng: Option<&AgqNg>,
) -> Option<Vec<f64>> {
    let theta = &params.theta;
    let n_obs = subject.observations.len();
    let n_bsv = model.n_eta;
    let unit = unit_dirs::<N>();
    let vidx = |node: u32, comp: usize| {
        if TH {
            vadj_index_h::<N, M2>(node, comp)
        } else {
            vadj_index::<N, 1>(node, comp, 0)
        }
    };
    vtape_reset(if TH { 1 + 2 * N } else { N + 1 });
    // An endpoint-only model has no parameter program (and no Gaussian rows to record).
    let axis = theta_axis(model).or_else(|| endpoint_only(model).then_some(theta.len()))?;
    let th = theta_leaves::<N, M2, TH>(axis, theta);
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
            if n_obs == 0 {
                return Some(Vec::new());
            }
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
    // The non-Gaussian term at η̂ as tape inputs, as `run` takes it: `q`, `∇q` and the
    // symmetric `∇²q` (one input per k ≥ l), and its value at each node.
    let ng_r = ng.map(|ng| ng_tape_inputs::<N>(&ng.jets));
    let node_qr: Vec<R> = ng
        .map(|ng| ng.nodes.iter().map(|(q, _)| rinput(*q)).collect())
        .unwrap_or_default();
    // The anchor H_a on the tape: the exact inner Hessian from the predictions' (f, G, Q),
    // or FOCEI's H̃.
    let mut h_a: Vec<Vec<R>> = if exact {
        crate::sens::focei_kernel::exact_inner_hessian_kernel(ctx, &fr, &gr, &qr, &er, &pr, active)?
    } else {
        crate::sens::focei_kernel::focei_htilde(ctx, &fr, &gr, &er, &pr)?
    };
    if let Some(ngr) = ng_r.as_ref() {
        for k in 0..n_bsv {
            for l in 0..n_bsv {
                h_a[k][l] = h_a[k][l] + ngr.h[k][l];
            }
        }
    }
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
    let ng_q = ng_r.as_ref().map_or(R::c(0.0), |n| n.q);
    let a = if grid.is_empty() {
        inner_l(ctx, &fr, &er, &pr, None)? + ng_q + half_logdet
    } else {
        let mut terms: Vec<R> = Vec::with_capacity(grid.len());
        for (j, ((z, c, b0), (ps, fl))) in
            grid.iter().zip(node_preds.iter().zip(&node_fr)).enumerate()
        {
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
            // The term at the node, linearised in `b` there as the predictions are.
            let qb = match ng {
                Some(ng) => {
                    let gq = &ng.nodes[j].1;
                    node_qr[j]
                        + <R as crate::sens::num::PkNum>::wsum3(
                            0.0,
                            (0..n_bsv).map(|k| (R::c(gq[k]), b[k] - R::c(b0[k]), one)),
                        )
                }
                None => R::c(0.0),
            };
            terms.push(R::c(*c) - (inner_l(ctx, &fb, &b, &pr, None)? + qb));
        }
        let mx = terms.iter().map(|t| t.v).fold(f64::NEG_INFINITY, f64::max);
        let sum = <R as crate::sens::num::PkNum>::wsum3(
            0.0,
            terms.iter().map(|&t| (one, (t - R::c(mx)).exp(), one)),
        );
        R::c(0.5 * active as f64 * std::f64::consts::PI.ln()) + half_logdet - (R::c(mx) + sum.ln())
    };
    let mut lg = inner_grad(ctx, &fr, &gr, &er, &pr, None)?;
    if let Some(ngr) = ng_r.as_ref() {
        for k in 0..N {
            lg[k] = lg[k] + ngr.g[k];
        }
    }

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
    let p2 = if n_obs == 0 {
        Vec::new()
    } else {
        record(model, subject, lay, theta, eta, &lv, &et)?
    };
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
    let (mut grad_eta, _) = sweep(&s1, false)?;
    // The term's inputs at η̂ move with η̂ through their own η-derivatives (the node values'
    // movement is the linearisation's, already on the tape).
    if let (Some(ngr), Some(ng)) = (ng_r.as_ref(), ng) {
        let d = ng_eta_contraction(&ng.jets, ngr, |x| ra(&s1, x));
        for k in 0..n_bsv {
            grad_eta[k] += d[k];
        }
    }
    let grad_l: Vec<f64> = lg.iter().map(|l| l.v).collect();
    // `held`: past a declined mode solve, `w = 0` makes `Ψ = A`, the held-EBE score
    // (`subject_adjoint_held_ebe_score`).
    let b = DVector::from_column_slice(&grad_eta);
    let w = match mode_solve_judged(&h_inner, &b, &grad_l, || {
        let mut g = inner_search_gradient::<N>(model, subject, lay, theta, eta, ctx, pv, None)?;
        if let Some(ng) = ng {
            for (gk, qk) in g.iter_mut().zip(&ng.jets.g) {
                *gk += qk;
            }
        }
        Some(g)
    }) {
        Some(w) => w,
        None if held => DVector::zeros(N),
        None => return None,
    };

    // ---- sweep 2: Ψ = A − wᵀ∇_η l(η̂) ----------------------------------------------
    let psi =
        a - <R as crate::sens::num::PkNum>::wsum3(0.0, (0..N).map(|k| (R::c(w[k]), lg[k], one)));
    let s2 = rsweep::<1>(&[psi]).to_vec();
    let (_, mut theta_adj) = sweep(&s2, true)?;
    let psi_of = |x: R| -> f64 { ra(&s2, x) };
    // The term's θ-derivatives: at η̂ through `(q, ∇q, ∇²q)`, at each node through its value.
    if let (Some(ngr), Some(_)) = (ng_r.as_ref(), ng) {
        let (a_q, a_g, a_h) = ng_adjoints(ngr, n_bsv, psi_of);
        let mut add = |d: Vec<f64>| {
            for (t, dv) in theta_adj.iter_mut().zip(d) {
                *t += dv;
            }
        };
        add(ng_theta_block(
            model,
            subject,
            theta,
            &eta[..n_bsv],
            a_q,
            &a_g,
            &a_h,
        )?);
        for ((_, _, b), &qr) in grid.iter().zip(&node_qr) {
            add(ng_theta_score(
                model,
                subject,
                theta,
                &b[..n_bsv],
                psi_of(qr),
            )?);
        }
    }
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
    cov_non_gaussian_served, cov_width_declines, subject_adjoint_cov_block_at,
    subject_adjoint_cov_diag_at, subject_adjoint_cov_hessian_at, subject_adjoint_eta_dx_at,
};
#[cfg(test)]
pub(crate) use hessian::{subject_adjoint_cov_hessian, subject_adjoint_eta_dx};

#[cfg(test)]
#[path = "ode_adjoint_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "ode_adjoint_phenomena_tests.rs"]
mod phenomena_tests;
