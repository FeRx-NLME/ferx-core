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
//! 1. **Record** the predictions into the subject's compiled plan (`parser::kernel_jit`): value and η-tangents
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
//! The same recorder runs tape-free for the inner loop's η-Jacobians,
//! inner gradients and Hessians ([`exact_obs_grads`], [`exact_inner_hessian`], …).
//!
//! Out of scope (`None`, and the caller salvages the subject by finite differences): SDE,
//! subjects with non-Gaussian records, IOV with FREM rows, engine shapes no recorder
//! serves.
#![allow(clippy::needless_range_loop)]

use super::*;
use crate::estimation::parameterization::{lower_tri_entries, unpack_params};
use crate::sens::focei_kernel::{data_l, foce_f, focei_f_grad, inner_grad, inner_l, LikCtx, ParT};
use crate::sens::num::vm::VmNum;
use crate::types::ModelParameters;
use nalgebra::{DMatrix, DVector};

/// The likelihood-side scope: a Gaussian diagonal-residual FOCEI subject whose `Ψ` is
/// the plain form below. Engine-side scope is decided by each engine.
pub(crate) fn likelihood_supported(
    _model: &CompiledModel,
    _subject: &Subject,
    _params: &ModelParameters,
) -> bool {
    _subject.obs_records.is_empty()
}

/// A recorder serves the model's predictions ([`recorder_in_scope`]) and its likelihood is
/// Gaussian. The production gates are [`outer_model_in_scope`] (which also serves non-Gaussian
/// endpoints) and [`recorder_in_scope`]; this narrower predicate is kept for the tests that
/// pin a Gaussian model's routing.
#[cfg(test)]
pub(crate) fn model_in_scope(model: &CompiledModel) -> bool {
    recorder_in_scope(model) && !model.has_non_gaussian()
}

/// The FOCE/FOCEI outer gradient's model scope: a recorder serves the model's predictions
/// (or it is endpoint-only), and it also serves a model whose
/// non-Gaussian endpoints all have an exact channel (`run` folds their terms into the
/// objective, over the BSV block under IOV). AGQ (`analytic_score_supported`) and the EBE
/// predictor (`optimize_bfgs`) take the same scope.
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
    !nested_levels_outside_scope(model)
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
/// parameters, the dual width (`n_eta`, or the stacked `n_eta + K·n_kappa` under
/// IOV) and `η̂`.
pub(super) struct Scoped<'m> {
    pub model: &'m CompiledModel,
    /// The engine production evaluates FOCE's typical-value variance `R⁰ = f(η = 0)` on: the
    /// model resolved at `η = 0`, which a closed-form absorption model near its flip-flop switch
    /// may route differently from `model` (resolved at `η̂`).
    pub pop_model: &'m CompiledModel,
    pub params: &'m ModelParameters,
    pub width: usize,
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
    Some(Scoped {
        model,
        pop_model,
        params,
        width,
        eta,
        layout: lay.map(std::borrow::Cow::into_owned),
    })
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
    adjoint_gradient(model, subject, pt, eta_hat, objective, false, None)
}

/// [`subject_adjoint_gradient_at`] for FOCE/FOCEI, and with it the exact mode response
/// `dη̂/dx` of the fit's own inner problem — one vector of the BSV effects per packed
/// coordinate, the EBE warm-start predictor's Jacobian (Almquist Eq. 48) — formed from the
/// gradient's own recording: one more sweep per BSV effect, no second recording (#2549).
/// The response is `None` where the gradient is (the adjoint declined) or the inner Hessian
/// is singular.
#[cfg(test)]
pub(crate) fn subject_adjoint_gradient_with_eta_dx_at(
    model: &CompiledModel,
    subject: &Subject,
    pt: &PackedPoint<'_>,
    eta_hat: &[f64],
    interaction: bool,
) -> (Option<Vec<f64>>, Option<Vec<DVector<f64>>>) {
    let objective = if interaction {
        Objective::Focei
    } else {
        Objective::Foce
    };
    subject_adjoint_gradient_and_eta_dx_at(model, subject, pt, eta_hat, objective)
}

/// [`subject_adjoint_gradient_with_eta_dx_at`] for any objective the adjoint serves: under
/// Laplace/AGQ (`Objective::Agq`) the response of the conditional mode the quadrature is
/// centred on, from the exact inner Hessian, out of the quadrature gradient's own tape.
#[cfg(test)]
pub(crate) fn subject_adjoint_gradient_and_eta_dx_at(
    model: &CompiledModel,
    subject: &Subject,
    pt: &PackedPoint<'_>,
    eta_hat: &[f64],
    objective: Objective,
) -> (Option<Vec<f64>>, Option<Vec<DVector<f64>>>) {
    let mut out = ModeOut {
        want_dx: true,
        ..ModeOut::default()
    };
    let g = subject_adjoint_gradient_and_mode_at(model, subject, pt, eta_hat, objective, &mut out);
    (g, out.eta_dx)
}

/// [`subject_adjoint_gradient_at`], with whatever `out` asks for about the subject's mode
/// ([`ModeOut`]) formed from the same pass. Both are `None` where the gradient is.
pub(crate) fn subject_adjoint_gradient_and_mode_at(
    model: &CompiledModel,
    subject: &Subject,
    pt: &PackedPoint<'_>,
    eta_hat: &[f64],
    objective: Objective,
    out: &mut ModeOut,
) -> Option<Vec<f64>> {
    let g = adjoint_gradient(
        model,
        subject,
        pt,
        eta_hat,
        objective,
        false,
        Some(&mut *out),
    );
    if g.is_none() {
        out.clear();
    }
    g
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
        .then(|| adjoint_gradient(model, subject, pt, eta_hat, objective, true, None))?
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
        return fixed_ebe_score_iov(&sc, subject, pt, h, interaction);
    }
    let sc = scope(model, subject, pt, eta_hat)?;
    if h.nrows() != subject.observations.len() || h.ncols() != sc.width {
        return None;
    }
    fixed_ebe_score(&sc, subject, pt, eta_hat, h, interaction)
}

fn fixed_ebe_score(
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
    let so = stacked_omega(model, subject, template, params, sc.width);
    let mut pv = ParT::of(model, subject, params);
    pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(so.chol.clone())?;

    // The objective on the scalar tape (`H` and `η̂` constants), scored inside the η̂
    // recording's weights callback on that recording's own prediction values — one
    // recording serves both the value and its θ-derivatives. Everything but the θ block
    // is assembled there; the θ block is linear in `theta_adj` and added afterwards.
    let scored: std::cell::RefCell<Option<(f64, Vec<f64>, Vec<f64>)>> = Default::default();
    let ng_adj: std::cell::RefCell<Option<(f64, Vec<f64>, Vec<Vec<f64>>)>> = Default::default();
    let score = |f: &[f64]| -> Option<Vec<Vec<f64>>> {
        // The kernel plan's inputs: `H` and `η̂` held, so leaves the plan replays at new
        // values, their adjoints unread.
        let mut inp = kernel::Inputs::default();
        let fr: Vec<kernel::Leaf> = f.iter().map(|&v| inp.leaf(v)).collect();
        // An SDE subject's variance rows (`f` holds `2n`) carry no η-jets here: `η̂` is held.
        let gr: Vec<Vec<kernel::Leaf>> = (0..f.len())
            .map(|j| {
                (0..sc.width)
                    .map(|k| inp.leaf(if j < n_obs { h[(j, k)] } else { 0.0 }))
                    .collect()
            })
            .collect();
        let er: Vec<kernel::Leaf> = sc.eta.iter().map(|&v| inp.leaf(v)).collect();
        let pr0: Option<Vec<kernel::Leaf>> = fpop
            .as_ref()
            .map(|p| p.iter().map(|&v| inp.leaf(v)).collect());
        let (pr, om) = param_leaves(&pv, &so, &mut inp);
        let ngr = ng.as_ref().map(|(q, g, hq)| {
            ng_leaves(
                sc.width,
                &NgJets {
                    q: *q,
                    g: g.clone(),
                    h: (0..g.len())
                        .map(|k| (0..g.len()).map(|l| hq[(k, l)]).collect())
                        .collect(),
                    t: NgThird::Full(Vec::new()),
                },
                &mut inp,
            )
        });
        let variant = kernel::variant_of(&(
            1u8,
            interaction,
            sc.width,
            pr0.is_some(),
            ng.as_ref().map(|j| j.1.len()),
            &so.placed.iter().map(|p| p.1.clone()).collect::<Vec<_>>(),
        ));
        let kr = kernel::kernel_run(
            model,
            subject,
            variant,
            crate::sens::num::vm::Layout::F64,
            &inp.vals,
            1,
            &|lv| {
                let t = |l: kernel::Leaf| l.tr(lv);
                let f: Vec<kj_tr::Tr> = fr.iter().map(|&l| t(l)).collect();
                let g: Vec<Vec<kj_tr::Tr>> = gr
                    .iter()
                    .map(|r| r.iter().map(|&l| t(l)).collect())
                    .collect();
                let e: Vec<kj_tr::Tr> = er.iter().map(|&l| t(l)).collect();
                let p0: Option<Vec<kj_tr::Tr>> =
                    pr0.as_ref().map(|v| v.iter().map(|&l| t(l)).collect());
                let p = param_tr(model, subject, &pr, lv)?;
                let fv = match ngr.as_ref() {
                    Some(n) => ng_f_grad(&ctx, &f, &g, &e, &p, &ng_tr(n, lv), interaction)?.0,
                    None => objective(&ctx, interaction, &f, &g, &e, p0.as_deref(), &p)?.0,
                };
                Some(vec![fv])
            },
        )?;
        let sadj = kr.reverse_one(&[1.0])?;
        let psi_of = |x: kernel::Leaf| -> f64 { x.adj(&sadj) };
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
        *scored.borrow_mut() = Some((kr.value(0), base, wp));
        Some(vec![wf])
    };
    // An endpoint-only subject has no prediction to record: its θ enter through the term.
    let theta_mode = if n_obs == 0 && ng.is_some() {
        score(&[])?;
        vec![0.0; theta.len()]
    } else {
        fixed_eta_theta_sigma_contractions(
            model,
            subject,
            theta,
            &params.sigma.values,
            eta_hat,
            score,
        )?
        .pop()?
    };
    let (nll, mut grad, wp) = scored.into_inner()?;
    let mut theta_adj = theta_mode;
    if let Some((a_q, a_g, a_h)) = ng_adj.into_inner() {
        let part = NgSeed {
            eta: &eta_hat[..n_bsv],
            a_q,
            a_g: &a_g,
            a_h: &a_h,
        };
        let d = ng_theta_adjoint(model, subject, theta, &[part])?;
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
    add_theta_sigma_adjoint(template, params, &theta_adj, &mut grad);
    (nll.is_finite() && grad.iter().all(|g| g.is_finite())).then_some((nll, grad))
}

/// `∂/∂θ` (raw θ, then σ for an SDE subject's variance rows) added onto a packed gradient
/// through each block's packing chain, as [`packed_gradient`] does for its `theta_adj`.
fn add_theta_sigma_adjoint(
    template: &ModelParameters,
    params: &ModelParameters,
    theta_adj: &[f64],
    grad: &mut [f64],
) {
    let theta = &params.theta;
    for (m, &a) in theta_adj.iter().take(theta.len()).enumerate() {
        grad[m] += a * crate::estimation::sens_outer_gradient::theta_dx_chain(template, theta, m);
    }
    let sigma_start = crate::estimation::parameterization::packed_segments(template).sigma_start();
    for (k, &a) in theta_adj.iter().skip(theta.len()).enumerate() {
        grad[sigma_start + k] += a * params.sigma.values[k];
    }
}

/// [`fixed_ebe_score`] under IOV: production's held objective (`foce_subject_nll_iov`) holds
/// the BSV columns of `H` and `b̂ = [η̂, κ̂]`, and rebuilds the κ columns `∂f/∂κ` at `b̂` from
/// θ. So the predictions are recorded with their κ-tangents on the vector tape (θ leaves,
/// `b` free), the kernel's objective over the stacked system is scored on the scalar tape
/// with the BSV columns constant and the κ columns as inputs, and one sweep seeded with both
/// adjoints carries `∂f/∂θ` and `∂²f/∂κ∂θ` to the θ leaves together.
fn fixed_ebe_score_iov(
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
    // The plan's first-order forward: the predictions and their `[η, κ]` columns, kept for
    // one reverse to θ. An endpoint-only subject has no prediction rows.
    let n_obs = subject.observations.len();
    let l1 = terms::layout1(sc.width);
    let lw = l1.lanes();
    let planned = if n_obs == 0 {
        None
    } else {
        Some(plan_forward(model, subject, lay, theta, eta, l1)?)
    };
    let lanes = planned.as_ref().map_or(&[][..], |p| p.1.as_slice());
    // An SDE subject's EKF variance rows (BSV only, as production's `ekf_p_obs` reads `η̂`):
    // a plan of their own over θ then σ, the kernel's rows `n..2n`.
    let planned_p = if n_obs > 0 && model.is_sde() {
        Some(plan_forward_in(
            model,
            subject,
            lay,
            theta,
            Some(&params.sigma.values),
            eta,
            l1,
        )?)
    } else {
        None
    };
    let p_lanes = planned_p.as_ref().map_or(&[][..], |p| p.1.as_slice());
    // Non-Gaussian records: `(q, ∇q, ∇²q)` on the BSV block, production's conditional
    // variance under either interaction setting (as `fixed_ebe_score`).
    let ng = if subject.obs_records.is_empty() {
        None
    } else {
        Some(non_gaussian_eta_jets2(
            model,
            subject,
            theta,
            &eta[..n_bsv],
        )?)
    };
    let pop =
        if n_obs > 0 && ng.is_none() && !interaction && model.error_spec.has_f_dependent_variance()
        {
            let zeros = vec![0.0; sc.width];
            let f64l = crate::sens::num::vm::Layout::F64;
            Some(plan_forward(
                sc.pop_model,
                subject,
                lay,
                theta,
                &zeros,
                f64l,
            )?)
        } else {
            None
        };
    let ctx = LikCtx::new(model, subject, params);
    let so = stacked_omega(model, subject, template, params, sc.width);
    let mut pv = ParT::of(model, subject, params);
    pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(so.chol.clone())?;

    let mut inp = kernel::Inputs::default();
    let mut fr: Vec<kernel::Leaf> = (0..n_obs).map(|j| inp.leaf(lanes[j * lw])).collect();
    for j in 0..p_lanes.len() / lw {
        fr.push(inp.leaf(p_lanes[j * lw]));
    }
    // The variance rows' η-jets are not read by the held objective's value.
    let gr: Vec<Vec<kernel::Leaf>> = (0..fr.len())
        .map(|j| {
            (0..sc.width)
                .map(|k| {
                    if j >= n_obs {
                        inp.leaf(0.0)
                    } else if k < n_bsv {
                        inp.leaf(h[(j, k)])
                    } else {
                        inp.leaf(lanes[j * lw + 1 + k])
                    }
                })
                .collect()
        })
        .collect();
    let pr0: Option<Vec<kernel::Leaf>> = pop
        .as_ref()
        .map(|(_, v, _)| v.iter().map(|&x| inp.leaf(x)).collect());
    let er: Vec<kernel::Leaf> = eta.iter().map(|&v| inp.leaf(v)).collect();
    let (pr, om) = param_leaves(&pv, &so, &mut inp);
    let ngr = ng.as_ref().map(|(q, g, hq)| {
        ng_leaves(
            sc.width,
            &NgJets {
                q: *q,
                g: g.clone(),
                h: (0..g.len())
                    .map(|k| (0..g.len()).map(|l| hq[(k, l)]).collect())
                    .collect(),
                t: NgThird::Full(Vec::new()),
            },
            &mut inp,
        )
    });
    let variant = kernel::variant_of(&(
        2u8,
        interaction,
        sc.width,
        pr0.is_some(),
        ng.as_ref().map(|j| j.1.len()),
        &so.placed.iter().map(|p| p.1.clone()).collect::<Vec<_>>(),
    ));
    let kr = kernel::kernel_run(
        model,
        subject,
        variant,
        crate::sens::num::vm::Layout::F64,
        &inp.vals,
        1,
        &|lv| {
            let t = |l: kernel::Leaf| l.tr(lv);
            let f: Vec<kj_tr::Tr> = fr.iter().map(|&l| t(l)).collect();
            let g: Vec<Vec<kj_tr::Tr>> = gr
                .iter()
                .map(|r| r.iter().map(|&l| t(l)).collect())
                .collect();
            let e: Vec<kj_tr::Tr> = er.iter().map(|&l| t(l)).collect();
            let p0: Option<Vec<kj_tr::Tr>> =
                pr0.as_ref().map(|v| v.iter().map(|&l| t(l)).collect());
            let p = param_tr(model, subject, &pr, lv)?;
            let fv = match ngr.as_ref() {
                Some(n) => ng_f_grad(&ctx, &f, &g, &e, &p, &ng_tr(n, lv), interaction)?.0,
                None => objective(&ctx, interaction, &f, &g, &e, p0.as_deref(), &p)?.0,
            };
            Some(vec![fv])
        },
    )?;
    let sadj = kr.reverse_one(&[1.0])?;
    let psi_of = |x: kernel::Leaf| -> f64 { x.adj(&sadj) };
    let ng_adj = ngr.as_ref().map(|ngr| ng_adjoints(ngr, n_bsv, psi_of));

    // θ's adjoint: the predictions' (their value and κ-column seeds) through the plan's
    // reverse, and FOCE's population predictions' through theirs.
    let n_th = theta.len();
    let n_sigma = if planned_p.is_some() {
        params.sigma.values.len()
    } else {
        0
    };
    let mut theta_adj = vec![0.0; n_th + n_sigma];
    if let Some((plan, _, fwd)) = planned_p.as_ref() {
        let mut du_bar = vec![0.0; n_obs * lw];
        for j in 0..n_obs {
            du_bar[j * lw] = psi_of(fr[n_obs + j]);
        }
        let (mut ub, mut pb) = (vec![0.0; sc.width * lw], vec![0.0; (n_th + n_sigma) * lw]);
        if !plan.reverse_from(fwd, l1, 1, &mut ub, &mut pb, &du_bar) {
            return None;
        }
        for (m, t) in theta_adj.iter_mut().enumerate() {
            *t += pb[m * lw];
        }
    }
    if let Some((plan, _, fwd)) = planned.as_ref() {
        let mut du_bar = vec![0.0; n_obs * lw];
        for j in 0..n_obs {
            du_bar[j * lw] = psi_of(fr[j]);
            for k in n_bsv..sc.width {
                du_bar[j * lw + 1 + k] = psi_of(gr[j][k]);
            }
        }
        let (mut ub, mut pb) = (vec![0.0; sc.width * lw], vec![0.0; n_th * lw]);
        if !plan.reverse_from(fwd, l1, 1, &mut ub, &mut pb, &du_bar) {
            return None;
        }
        for (m, t) in theta_adj.iter_mut().take(n_th).enumerate() {
            *t += pb[m * lw];
        }
    }
    if let (Some((pp, _, pf)), Some(r0)) = (pop.as_ref(), pr0.as_ref()) {
        let du_bar: Vec<f64> = r0.iter().map(|&x| psi_of(x)).collect();
        let (mut ub, mut pb) = (vec![0.0; sc.width], vec![0.0; n_th]);
        if !pp.reverse_from(
            pf,
            crate::sens::num::vm::Layout::F64,
            1,
            &mut ub,
            &mut pb,
            &du_bar,
        ) {
            return None;
        }
        for (t, d) in theta_adj.iter_mut().zip(&pb) {
            *t += d;
        }
    }
    if let Some((a_q, a_g, a_h)) = ng_adj {
        let part = NgSeed {
            eta: &eta[..n_bsv],
            a_q,
            a_g: &a_g,
            a_h: &a_h,
        };
        let d = ng_theta_adjoint(model, subject, theta, &[part])?;
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
    let nll = kr.value(0);
    (nll.is_finite() && grad.iter().all(|g| g.is_finite())).then_some((nll, grad))
}

fn adjoint_gradient(
    model: &CompiledModel,
    subject: &Subject,
    pt: &PackedPoint<'_>,
    eta_hat: &[f64],
    objective: Objective,
    held: bool,
    eta_dx: Option<&mut ModeOut>,
) -> Option<Vec<f64>> {
    let (template, x) = (pt.template, pt.x);
    // A reused `out` starts each pass without a cause, so a decline elsewhere cannot
    // inherit the last pass's (#2713).
    let mut eta_dx = eta_dx;
    if let Some(o) = eta_dx.as_deref_mut() {
        o.mode_system_declined = false;
    }
    let sc = scope(model, subject, pt, eta_hat)?;
    let objective = match objective {
        // No random effects: both anchors are the empty `H`, and `Ψ = F`.
        Objective::Agq { n_agq, exact: true } if sc.width == 0 => Objective::Agq {
            n_agq,
            exact: false,
        },
        objective => objective,
    };
    fn go(
        sc: &Scoped,
        subject: &Subject,
        template: &ModelParameters,
        x: &[f64],
        objective: Objective,
        held: bool,
        eta_dx: Option<&mut ModeOut>,
    ) -> Option<Vec<f64>> {
        let go_run = |eta_dx: Option<&mut ModeOut>| match objective {
            Objective::Agq { n_agq, exact } => run_agq(
                sc.model,
                subject,
                sc.layout.as_ref(),
                template,
                sc.params,
                x,
                &sc.eta,
                n_agq,
                exact,
                held,
                eta_dx,
            ),
            _ => run(
                sc.model,
                sc.pop_model,
                subject,
                sc.layout.as_ref(),
                template,
                sc.params,
                x,
                &sc.eta,
                objective != Objective::Foce,
                eta_dx,
            ),
        };
        match eta_dx {
            None => go_run(None),
            // The mode response shares the gradient's sweeps, and a lane that fails there (a
            // non-finite row adjoint, say from a near-singular `H`) fails the whole pass. The
            // gradient must not depend on whether the response was asked for, so a failed
            // fused pass is re-run as the gradient alone, whose Ψ is the same arithmetic, and
            // reports no response (#2549 review r2 #2).
            Some(o) => match go_run(Some(&mut *o)) {
                Some(g) => Some(g),
                None if o.want_dx => {
                    let mut lite = ModeOut {
                        want_hessian: o.want_hessian,
                        ..ModeOut::default()
                    };
                    let g = go_run(Some(&mut lite));
                    o.eta_dx = None;
                    o.hessian = lite.hessian;
                    g
                }
                None => None,
            },
        }
    }
    go(&sc, subject, template, x, objective, held, eta_dx)
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
/// from a certified or iteration-limited inner search, which need not be stationary in
/// `‖g‖∞` (see [`MODE_STATIONARY_TOL`]), and the gradient `wᵀ∇_η l` drops is then not zero. `grad_l` is `∇_η l(η̂)`; above
/// [`MODE_STATIONARY_TOL`] the subject declines (#2093 review) — under AGQ to the held-EBE
/// score ([`subject_adjoint_held_ebe_score_at`]), not the reconverged one.
///
/// The SPD branch takes no such gate, deliberately. Near a strict mode `η̂ = η* + ε` and
/// `∂Ψ/∂x` misses the converged objective's total derivative by `O(|∇_η l|)` only, while
/// every salvage a decline could take drops the EBE response, an `O(1)` error: measured off
/// a Laplace mode, 2.1e-3 vs 3.0 at `|ε| ≈ 1e-3` and 2.1e-2 vs 2.6 at 1e-2
/// (`off_the_mode_the_identity_beats_the_held_ebe_gradient`). An indefinite `H` off the mode
/// is not near a minimum at all, which is what the gate above refuses.
///
/// A declined gate is judged again by `inner` — the gradient the inner search converged on
/// ([`inner_search_gradient`]) — which is evaluated only then: a definite `H` never reads a
/// gradient, and a singular `H` or a non-finite `w` declines whatever the gradient, so
/// neither pays its recording.
fn mode_solve(
    h: &DMatrix<f64>,
    b: &DVector<f64>,
    grad_l: &[f64],
    inner: impl FnOnce() -> Option<Vec<f64>>,
) -> Option<DVector<f64>> {
    if let Some(c) = h.clone().cholesky() {
        return Some(c.solve(b));
    }
    let w = h.clone().full_piv_lu().solve(b)?;
    if !w.iter().all(|v| v.is_finite()) {
        return None;
    }
    let stationary =
        |g: &[f64]| crate::sens::num::nan_max_abs(g.iter().copied()) <= MODE_STATIONARY_TOL;
    (stationary(grad_l) || inner().is_some_and(|g| stationary(&g))).then_some(w)
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
fn inner_search_gradient(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
    ctx: &LikCtx,
    pv: &ParT<f64>,
    frozen: Option<&[f64]>,
) -> Option<Vec<f64>> {
    let p = canonical_jets1(
        model,
        subject,
        lay,
        theta,
        Some(&ctx.params.sigma.values),
        eta,
    )?;
    let (f, g) = p.f_g();
    inner_grad(ctx, &f, &g, eta, pv, frozen)
}

/// The largest `|∇_η l(η̂)|∞` [`mode_solve`] treats as a stationary point on its indefinite
/// branch: ten times the inner search's default gradient tolerance (`inner_tol = 1e-5`).
///
/// A gradient test of its own, **not** the inner search's certificate restated: a certified
/// EBE can fail it (#2557). `FallbackCertifier`'s gradient arm (`‖g‖ < tol`) meets it at the
/// default `inner_tol` in the raw metric, but not above `inner_tol = 1e-4`, and not under
/// FREM's preconditioned stop (#406), whose posterior-variance weights (~1e-6 on the covariate
/// pseudo-observation dimensions) admit a raw `|g|∞` near 1e-2. Its Newton-step arm
/// (`‖H⁻¹g‖ < tol`) admits `|g|` up to `c·tol` along a direction of curvature `c`, and its
/// resolution arm (`½gᵀH⁻¹g ≤ INNER_FTOL_REL·(1 + |f|)`) bounds no gradient. Measured under
/// `inner_optimizer = nelder_mead`: a `one_cpt_infusion` EBE certified by the Newton arm at
/// `‖g‖∞ = 0.18`, curvature ~3e4. The stall stop on a numerical objective bounds no gradient
/// either. Such a point declines here, to the same route as any other decline.
///
/// Kept rather than widened to accept those arms. The bound the identity's error obeys at a
/// certified point, `√(2·INNER_FTOL_REL·(1 + |f|))` SEs, rests on a positive definite `H`, and
/// both Newton arms need the certifier's finite-difference Hessian to factor. A certified
/// point that reaches this branch is one where that Hessian and the exact one disagree on
/// definiteness: the case the bound says nothing about. `mode_solve` also does not know
/// whether `η̂` was certified.
const MODE_STATIONARY_TOL: f64 = 1e-4;

/// The θ leaves of one recording and, when the weights reach the likelihood through the
/// networks' outputs alone ([`hessian::NnReduction`]), those outputs on the **same** number
/// type: the recording reads them in place of evaluating the networks over the weights. The
/// type is the guarantee — a recording can only be handed outputs of its own type, so there
/// is no thread-local channel to miss or to nest (#2293 review).
pub(super) struct ThetaLeaves<'a, T> {
    pub(super) theta: &'a [T],
    /// σ as leaves, for an SDE subject's recording of its EKF variance rows ([`record_sde_p`]).
    pub(super) sigma: Option<&'a [T]>,
    #[cfg(feature = "nn")]
    pub(super) nn: Option<&'a [Vec<T>]>,
}

impl<'a, T> ThetaLeaves<'a, T> {
    /// θ leaves alone: every network is evaluated on the recording's number type.
    pub(super) fn theta(theta: &'a [T]) -> Self {
        ThetaLeaves {
            theta,
            sigma: None,
            #[cfg(feature = "nn")]
            nn: None,
        }
    }

    /// These leaves with σ as parameter slots after θ.
    pub(super) fn with_sigma(mut self, sigma: Option<&'a [T]>) -> Self {
        self.sigma = sigma;
        self
    }

    /// θ leaves with the networks' outputs (`None`: evaluate the networks; without the `nn`
    /// feature a model has no networks to stand aside).
    pub(super) fn with_nn(theta: &'a [T], nn: Option<&'a [Vec<T>]>) -> Self {
        #[cfg(not(feature = "nn"))]
        let _ = nn;
        ThetaLeaves {
            theta,
            sigma: None,
            #[cfg(feature = "nn")]
            nn,
        }
    }
}

impl<T> std::ops::Deref for ThetaLeaves<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.theta
    }
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
    let mut by_slot = crate::scratch::take_for_opt::<T>(n_slots);
    by_slot.resize(n_slots, None);
    for (&(slot, _), &v) in prog.pk_var_slots().iter().zip(vals.iter()) {
        if slot < n_slots {
            by_slot[slot] = Some(v);
        }
    }
    crate::scratch::give_for(vals);
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
            // `0 < s < ∞`: a `NaN` fails both.
            if !(s.br_cmp_c(crate::sens::num::BranchCmp::Gt, 0.0)
                && s.br_cmp_c(crate::sens::num::BranchCmp::Lt, f64::INFINITY))
            {
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
        let init_state = init_seed::<T>(ode, &params_dual)
            .unwrap_or_else(|| vec![T::from_f64(0.0); ode.n_states]);
        let preds = integrate_seeded_duals(model, subject, &pk.values, &params_dual, &init_state)?;
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
        true,
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

/// An SDE subject's EKF process-noise variances `p_obs_j` ([`crate::stats::likelihood::ekf_p_obs`]
/// — the pre-update `P[obs, obs]` at each record) as recorded rows: the one filter
/// ([`crate::ode::ekf::solve_ekf_g`]) over the recording type, its individual parameters the
/// parameter program on the η and θ leaves, its Kalman `R` the residual variance at the
/// filter's own mean (production's: one solve) with σ as the leaves after θ, and its diffusion variances the θ leaves from
/// `diffusion_theta_start`, floored at zero. A modeled `RATE=-1/-2` dose is resolved on the
/// values, as production resolves it, and its `R`/`D` handed to the filter as the parameter it
/// is, so the moving window end carries its derivatives.
fn record_sde_p<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    th: &ThetaLeaves<'_, T>,
    et: &[T],
    sigma: &[T],
) -> Option<Vec<T>> {
    use crate::sens::num::BranchCmp;
    let ode = model.ode_spec.as_ref()?;
    let prog = ode.indiv_param_program.as_ref()?;
    let start = model.diffusion_theta_start?;
    // Production's `ekf_p_obs` is one filter pass over one parameter set: the BSV η̂ alone
    // under IOV (κ = 0), the subject's static covariates, `TIME = 0`; its Kalman `R` reads
    // the filter's own mean, before any output scale or LTBS. So none of those
    // is a decline here, and tv covariates, toggles and `TIME` are read exactly as it reads
    // them (not at all).
    let (eta, et) = (&eta[..model.n_eta], &et[..model.n_eta]);
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let by_slot = taped_params(model, prog, pk.values.len(), th, et, &subject.covariates);
    let params: Vec<T> = by_slot
        .iter()
        .zip(&pk.values)
        .map(|(d, &v)| d.unwrap_or(T::from_f64(v)))
        .collect();
    let zero = T::from_f64(0.0);
    let init = init_seed::<T>(ode, &params).unwrap_or_else(|| vec![zero; ode.n_states]);
    let modeled: Vec<Option<T>> = subject
        .doses
        .iter()
        .map(|d| modeled_slot_for(&ode.dose_attr_map, d).map(|(_, slot)| params[slot]))
        .collect();
    let resolved = crate::dosing::resolve_subject_doses(subject, &ode.dose_attr_map, &pk.values);
    let mut diffusion = vec![zero; ode.n_states];
    for (k, &state) in model.diffusion_state_indices.iter().enumerate() {
        if let (Some(&x), true) = (th.get(start + k), state < ode.n_states) {
            diffusion[state] = if x.br_cmp_c(BranchCmp::Lt, 0.0) {
                zero
            } else {
                x
            };
        }
    }
    let dose_f: Vec<T> = resolved
        .doses
        .iter()
        .map(|d| params[f_bio_slot(ode, d.cmt_raw())])
        .collect();
    // `ErrorSpec::Single` (an SDE model has no per-CMT readout): the key is unread.
    let shape = model.error_spec.loading_shape(0, sigma.len());
    let r_of = |_: usize, f: T| -> T {
        let loads: Vec<(usize, T)> = shape
            .iter()
            .map(|&(i, form)| (i, form.coeff_g(f)))
            .collect();
        let v = crate::stats::residual_error::loadings_variance_g(&loads, sigma, &[], |_| zero);
        // `ode_predictions_ekf_with_diffusion`'s guard: an unusable variance reads as 1.
        if v.br_cmp_c(BranchCmp::Gt, 0.0) && v.br_cmp_c(BranchCmp::Lt, f64::INFINITY) {
            v
        } else {
            T::from_f64(1.0)
        }
    };
    let pts = crate::ode::ekf::solve_ekf_g::<T, _>(
        &crate::ode::ekf::ProgramRhs(ode.rhs_program.as_ref()?),
        ode.n_states,
        ode.obs_cmt_idx()?,
        &diffusion,
        &params,
        &dose_f,
        &modeled,
        &init,
        &resolved.doses,
        &resolved.obs_times,
        &r_of,
        ode.effective_solver_opts(),
    );
    Some(pts.iter().map(|p| p.p_obs).collect())
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
    let preds = crate::sens::provider::analytic_static_predictions(
        model,
        subject,
        &pk,
        |slot| by_slot.get(slot).copied().flatten(),
        th,
        et,
    );
    crate::scratch::give_opt_for::<T>(by_slot);
    let preds = preds?;
    closed_form_finish(model, subject, theta, eta, th, et, prog, preds)
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
        true,
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
                // `0 < s < ∞`: a `NaN` fails both.
                if !(s.br_cmp_c(crate::sens::num::BranchCmp::Gt, 0.0)
                    && s.br_cmp_c(crate::sens::num::BranchCmp::Lt, f64::INFINITY))
                {
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
    let apply = |y: &mut T, s: Option<T>| -> Option<()> {
        if let Some(s) = s {
            // `0 < s < ∞`: a `NaN` fails both.
            if !(s.br_cmp_c(crate::sens::num::BranchCmp::Gt, 0.0)
                && s.br_cmp_c(crate::sens::num::BranchCmp::Lt, f64::INFINITY))
            {
                return None;
            }
            *y = *y / s;
        }
        Some(())
    };
    match &model.scaling {
        ScalingSpec::PerCmt(map) => {
            let scales: Vec<Option<T>> = subject
                .obs_cmts
                .iter()
                .map(|c| map.get(c).map_or(Some(None), &mut scale_of))
                .collect::<Option<_>>()?;
            for (y, s) in preds.iter_mut().zip(scales) {
                apply(y, s)?;
            }
        }
        // One scale for every observation: applied directly, no per-observation copy
        // (#2549 W1). The same divisions in the same order.
        spec => {
            let s = scale_of(spec)?;
            for y in preds.iter_mut() {
                apply(y, s)?;
            }
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
    // readout-slot and IOV gates do not apply to a recorder that hands the
    // walk taped parameters: a compartment-free model has no walk; a modeled dose needs its
    // slot.
    if model.is_algebraic()
        || !closed_form_serves(model, subject)
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
    // Every record's `$PK` inputs: doses, then observations, then PK-only rows.
    let (n_dose, n_obs) = (subject.doses.len(), subject.obs_times.len());
    let records = crate::pk::record_pk_inputs(subject);
    let group = |occ: u32| layout.and_then(|l| l.group_of(occ));
    // `first[i]`: the record whose evaluation record `i` reuses (itself when none);
    // `None` when every record is its own. With subject-static covariates and no `TIME`
    // read, every record of one occasion group reads the same parameters, so it shares by
    // group alone (#2558: `SharingScope`).
    let first: Option<std::rc::Rc<[usize]>> =
        crate::pk::sharing_scope::first_record_with_same_pk_inputs(
            model,
            subject,
            crate::pk::sharing_scope::ShareSite::Recorder,
            records.iter().map(|&(occ, cov, _)| (group(occ), cov)),
        );
    // The distinct evaluations, and which one each record reads. Reserved for one per
    // record: a `PkParams` is 1 KiB, and growing into the vector re-copied every
    // evaluation already made, about 2% of a FOCEI fit on a model whose records cannot
    // share (#2543 review r1, row 3).
    let mut snaps: Vec<(crate::types::PkParams, Vec<T>)> = Vec::with_capacity(records.len());
    let mut snap_of: Vec<usize> = Vec::with_capacity(records.len());
    for (i, &(occ, cov, time)) in records.iter().enumerate() {
        match first.as_ref().map(|f| f[i]).filter(|&f| f != i) {
            None => {
                snap_of.push(snaps.len());
                snaps.push(at(occ, cov, time));
            }
            Some(f) => snap_of.push(snap_of[f]),
        }
    }
    // An `ADDL`-implied dose's lag is its parent row's (#1909), value and dual alike:
    // `inherit_parent_dose_attrs{,_g}` over the `(PkParams, Vec<T>)` pair, so the copies
    // come from the same `parent_dose_attr_copies`. An implied dose whose evaluation other
    // records read gets its own copy first, so they keep their lag.
    let mut readers: Option<Vec<u32>> = None;
    for (k, p, slot) in crate::pk::parent_dose_attr_copies(model, subject) {
        if snap_of[k] == snap_of[p] {
            continue;
        }
        let readers = readers.get_or_insert_with(|| {
            let mut n = vec![0; snaps.len()];
            snap_of.iter().for_each(|&s| n[s] += 1);
            n
        });
        if readers[snap_of[k]] > 1 {
            readers[snap_of[k]] -= 1;
            snaps.push(snaps[snap_of[k]].clone());
            readers.push(1);
            snap_of[k] = snaps.len() - 1;
        }
        let (dk, dp) = (snap_of[k], snap_of[p]);
        snaps[dk].0.values[slot] = snaps[dp].0.values[slot];
        debug_assert!(
            slot < snaps[dp].1.len() && slot < snaps[dk].1.len(),
            "dual seed too short for parent-dose slot {slot}"
        );
        if slot < snaps[dp].1.len() && slot < snaps[dk].1.len() {
            snaps[dk].1[slot] = snaps[dp].1[slot];
        }
    }
    let doses = &snap_of[..n_dose];
    let obs = &snap_of[n_dose..n_dose + n_obs];
    let pk_only = &snap_of[n_dose + n_obs..];
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
    let dose_pk: Vec<crate::types::PkParams> = doses.iter().map(|&s| snaps[s].0).collect();
    let at_dose: Vec<_> = doses.iter().map(|&s| dual(&snaps[s].1)).collect();
    let at_obs: Vec<_> = obs.iter().map(|&s| dual(&snaps[s].1)).collect();
    let at_pk_only: Vec<_> = pk_only.iter().map(|&s| dual(&snaps[s].1)).collect();
    let slot_dual = |k: usize, slot: usize| {
        snaps[doses[k]]
            .1
            .get(slot)
            .copied()
            .unwrap_or(T::from_f64(0.0))
    };
    let obs_params = |j: usize| -> [T; crate::types::MAX_PK_PARAMS] {
        let p = &snaps[obs[j]].1;
        std::array::from_fn(|s| p.get(s).copied().unwrap_or(T::from_f64(0.0)))
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
        let decay: Option<Vec<Vec<T>>> =
            iov.then(|| obs.iter().map(|&s| snaps[s].1.clone()).collect());
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
            // `0 < s < ∞`: a `NaN` fails both.
            if !(s.br_cmp_c(crate::sens::num::BranchCmp::Gt, 0.0)
                && s.br_cmp_c(crate::sens::num::BranchCmp::Lt, f64::INFINITY))
            {
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
    // An SDE subject's variance rows: the filter, not the predictions.
    if let (true, Some(sigma)) = (model.is_sde(), th.sigma) {
        return record_sde_p(model, subject, theta, eta, th, et, sigma);
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
fn objective<T: crate::sens::num::PkNum>(
    ctx: &LikCtx,
    interaction: bool,
    f: &[T],
    g: &[Vec<T>],
    eta: &[T],
    fpop: Option<&[T]>,
    p: &ParT<T>,
) -> Option<(T, Vec<T>)> {
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
#[cfg(feature = "survival")]
pub(super) fn objective_ng<T: crate::sens::num::PkNum>(
    ctx: &LikCtx,
    interaction: bool,
    f: &[T],
    g: &[Vec<T>],
    eta: &[T],
    fpop: Option<&[T]>,
    ng: &crate::sens::focei_kernel::NonGauss<T>,
    p: &ParT<T>,
) -> Option<(T, Vec<T>)> {
    let (fv, lg) = ng_f_grad(ctx, f, g, eta, p, ng, interaction)?;
    match fpop {
        Some(f0) if !interaction => {
            let mut frozen = inner_grad(ctx, f, g, eta, p, Some(f0))?;
            for k in 0..ng.g.len() {
                frozen[k] = frozen[k] + ng.g[k];
            }
            Some((fv, frozen))
        }
        _ => Some((fv, lg)),
    }
}

/// Without the `survival` feature there are no non-Gaussian records, so no term: a stub, so a
/// default build compiles none of the term's objective for any width.
#[cfg(not(feature = "survival"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn objective_ng<T: crate::sens::num::PkNum>(
    _: &LikCtx,
    _: bool,
    _: &[T],
    _: &[Vec<T>],
    _: &[T],
    _: Option<&[T]>,
    _: &crate::sens::focei_kernel::NonGauss<T>,
    _: &ParT<T>,
) -> Option<(T, Vec<T>)> {
    None
}

/// `focei_f_grad_ng` — production's objective with a non-Gaussian term — under `survival`, a
/// stub (no term exists) without it.
#[cfg(feature = "survival")]
fn ng_f_grad<T: crate::sens::num::PkNum>(
    ctx: &LikCtx,
    f: &[T],
    g: &[Vec<T>],
    eta: &[T],
    p: &ParT<T>,
    ng: &crate::sens::focei_kernel::NonGauss<T>,
    interaction: bool,
) -> Option<(T, Vec<T>)> {
    crate::sens::focei_kernel::focei_f_grad_ng(ctx, f, g, eta, p, ng, interaction)
}

#[cfg(not(feature = "survival"))]
fn ng_f_grad<T: crate::sens::num::PkNum>(
    _: &LikCtx,
    _: &[T],
    _: &[Vec<T>],
    _: &[T],
    _: &ParT<T>,
    _: &crate::sens::focei_kernel::NonGauss<T>,
    _: bool,
) -> Option<(T, Vec<T>)> {
    None
}

/// The canonical recording at `eta` (any engine the adjoint serves): per observation the
/// prediction, its η-gradient and its packed η-Hessian, or `None` outside the recorders'
/// scope. Shared by [`exact_inner_hessian`] and the AGQ/Laplace anchor.
fn canonical_jets(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    sigma: Option<&[f64]>,
    eta: &[f64],
) -> Option<PredLanes<Vec<f64>>> {
    let (_, mut du, _) = plan_forward(model, subject, lay, theta, eta, terms::layout2(eta.len()))?;
    if let Some(sigma) = sigma.filter(|_| model.is_sde()) {
        // The kernel's SDE rows: the EKF's variances after the predictions.
        let (_, dp, _) = plan_forward_in(
            model,
            subject,
            lay,
            theta,
            Some(sigma),
            eta,
            terms::layout2(eta.len()),
        )?;
        du.extend(dp);
    }
    Some(PredLanes::new(du, eta.len(), true))
}

/// [`canonical_jets`] at first order (`(f, ∂f/∂η)`), the inner loop's per-step recording: the
/// plan's forward at the first-order layout.
fn canonical_jets1(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    sigma: Option<&[f64]>,
    eta: &[f64],
) -> Option<PredLanes<Vec<f64>>> {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    use std::sync::atomic::Ordering::Relaxed;
    let n_obs = subject.observations.len();
    let mut p = with_plan(model, subject, lay, theta, eta, &mut |plan| {
        kjit_run(plan, theta, eta, n_obs)
    })?;
    if let Some(sigma) = sigma.filter(|_| model.is_sde()) {
        let ts: Vec<f64> = theta.iter().chain(sigma).copied().collect();
        let pp = with_plan_in(model, subject, lay, theta, Some(sigma), eta, &mut |plan| {
            kjit_run(plan, &ts, eta, n_obs)
        })?;
        p.du.extend(pp.du);
    }
    kj::STATS.runs.fetch_add(1, Relaxed);
    Some(p)
}

/// The subject's predictions at `eta` (the stacked `[η, κ]` under IOV) into `out`, from the
/// compiled plan's value-only forward ([`kj_fwd::Plan::run_values`]) instead of the native
/// predictor; `false` leaves the native predictor to answer. Served only where the plan is
/// straight-line code (`chunks_only`): a closed form, or an `[odes]` model on the exact linear
/// path. Both are one implementation with the native walk (the same generic kernels over `f64`
/// and over the recording type, host analyses shared), so the values are bit-equal to it (pinned
/// by `plan_values_are_bit_equal_to_the_native_predictor` and
/// `linear_ode_plan_values_are_bit_equal_to_the_native_predictor`). A nonlinear ODE drives a
/// solver loop, so its plan is not `chunks_only` and declines; so do a closed form rerouted to
/// its ODE twin, an SDE and a FREM model, and a recording that is open.
pub(crate) fn plan_values(
    model: &CompiledModel,
    subject: &Subject,
    layout: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
    out: &mut Vec<f64>,
) -> bool {
    if subject.observations.is_empty()
        || model.is_sde()
        || model.frem_config.is_some()
        || kj_fwd::recording_open()
    {
        return false;
    }
    let Some((eff, _, eta_p, lay)) = resolve(model, subject, theta, eta, layout) else {
        return false;
    };
    // A closed form rerouted to its ODE twin is a different formula from the native walk's.
    if eff.ode_spec.is_some() && model.ode_spec.is_none() {
        crate::scratch::give(eta_p);
        return false;
    }
    let n_obs = subject.observations.len();
    out.clear();
    out.resize(n_obs, 0.0);
    let served = with_plan_in(
        eff,
        subject,
        lay.as_deref(),
        theta,
        None,
        &eta_p,
        &mut |plan| {
            if plan.n_out() != n_obs {
                return None;
            }
            // Declined, not unserved: the plan is fine for the gradient, and must not be marked bad.
            if !plan.chunks_only() {
                return Some(false);
            }
            plan.run_values(theta, &eta_p, out).then_some(true)
        },
    );
    crate::scratch::give(eta_p);
    served == Some(true)
}

/// An SDE subject's variance-row plan's `kj::SubjectKey` kind (kernels are 8, the covariance
/// step's reduced basis 9).
const SDE_P_KIND: u8 = 10;

/// A subject's compiled plan (#2786).
pub(super) type PlanArc =
    std::sync::Arc<crate::parser::model_parser::rhs_regcode::kernel_jit::Plan>;

/// The first of the subject's plans for `eta.len()` random effects that `serves` — the caller
/// runs the plan at its own layout and returns `None` where the plan's guard does not hold
/// there — recording a variant at `(theta, eta)` when no cached one does (#2786). `None`
/// where the subject has no prediction rows or cannot be recorded.
fn with_plan<R>(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
    serves: &mut dyn FnMut(&PlanArc) -> Option<R>,
) -> Option<R> {
    with_plan_in(model, subject, lay, theta, None, eta, serves)
}

/// [`with_plan`] for an SDE subject's variance rows when `sigma` is given: a plan of its own
/// kind whose parameter slots are θ then σ (the Kalman `R` reads σ), its outputs the EKF's
/// `p_obs_j` ([`record_sde_p`]).
fn with_plan_in<R>(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    sigma: Option<&[f64]>,
    eta: &[f64],
    serves: &mut dyn FnMut(&PlanArc) -> Option<R>,
) -> Option<R> {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    use std::sync::atomic::Ordering::Relaxed;
    // An endpoint-only subject has no prediction rows to record (its terms have their own
    // plans, `terms::term_plans`).
    if subject.observations.is_empty() {
        return None;
    }
    let (n, n_obs) = (eta.len(), subject.observations.len());
    let n_in = theta.len() + sigma.map_or(0, <[f64]>::len);
    let mut key = kj::SubjectKey::new(model, subject, n, n_in).with_layout(lay);
    if sigma.is_some() {
        key = key.with_kind(SDE_P_KIND);
    }
    let fits = |p: &PlanArc| p.n_out() == n_obs && p.n_eta() == n && p.n_theta() == n_in;
    match kj::lookup(&key) {
        kj::Lookup::Bad => return None,
        kj::Lookup::Found(ks) => {
            for (i, plan) in ks.order().filter(|(_, p)| fits(p)) {
                if let Some(r) = serves(plan) {
                    ks.hit(i);
                    return Some(r);
                }
            }
            kj::STATS.guard_fail.fetch_add(1, Relaxed);
        }
        kj::Lookup::Missing => {
            if let Some(why) = kj::model_bad(&key) {
                kj::mark_bad(&key, &why);
                return None;
            }
        }
    }
    let plan = kjit_record(model, subject, lay, theta, eta, None, sigma, &key)?;
    let r = serves(&plan);
    if r.is_none() {
        // A recording that does not serve at its own point never will: recording again on
        // every call would only repeat it.
        kj::mark_bad(&key, "the plan does not serve at its own recording point");
    }
    r
}

/// [`with_plan`] for the covariance step's reduced basis ([`hessian::NnReduction`]): the
/// predictions recorded with the networks' raw outputs `nn_z` as parameter slots after θ, so
/// they are inputs a direction can move — with `sigma`, an SDE subject's variance rows
/// ([`with_plan_in`]), σ's slots after the outputs'. Keyed apart from the subject's plain plans.
pub(super) fn with_plan_nn<R>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    nn_z: &[Vec<f64>],
    sigma: Option<&[f64]>,
    serves: &mut dyn FnMut(&PlanArc) -> Option<R>,
) -> Option<R> {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    if subject.observations.is_empty() {
        return None;
    }
    let n_in =
        theta.len() + nn_z.iter().map(Vec::len).sum::<usize>() + sigma.map_or(0, <[f64]>::len);
    let shape = kernel::variant_of(&(
        nn_z.iter().map(Vec::len).collect::<Vec<_>>(),
        sigma.map(<[f64]>::len),
    ));
    let key = kj::SubjectKey::new(model, subject, eta.len(), n_in)
        .with_kind(9)
        .with_variant(shape);
    let fits = |p: &PlanArc| p.n_eta() == eta.len() && p.n_theta() == n_in;
    match kj::lookup(&key) {
        kj::Lookup::Bad => return None,
        kj::Lookup::Found(ks) => {
            for plan in ks.iter().filter(|p| fits(p)) {
                if let Some(r) = serves(plan) {
                    return Some(r);
                }
            }
        }
        kj::Lookup::Missing => {}
    }
    let plan = kjit_record(model, subject, None, theta, eta, Some(nn_z), sigma, &key)?;
    let r = serves(&plan);
    if r.is_none() {
        kj::mark_bad(&key, "the plan does not serve at its own recording point");
    }
    r
}

/// [`with_plan`] with `serves` also told the plan's width, `eta.len()`.
pub(super) fn with_plan_width<R>(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
    serves: &mut dyn FnMut(usize, &PlanArc) -> Option<R>,
) -> Option<R> {
    let n = eta.len();
    with_plan(model, subject, lay, theta, eta, &mut |p| serves(n, p))
}

/// Record the subject on the recording type at `(theta, eta)` and plan it (#2786): the
/// variant joins `key`'s. `None` — and the subject (or, for a reason of the model's, the
/// model) marked so no caller tries again — where the recording fails.
#[allow(clippy::too_many_arguments)]
fn kjit_record(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
    nn_z: Option<&[Vec<f64>]>,
    sigma: Option<&[f64]>,
    key: &crate::parser::model_parser::rhs_regcode::kernel_jit::SubjectKey,
) -> Option<PlanArc> {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    use std::sync::atomic::Ordering::Relaxed;
    let n_obs = subject.observations.len();
    let t = std::time::Instant::now();
    // The parameter slots: θ, then the reduced basis's network outputs and an SDE subject's σ.
    let slot = theta.len()
        + nn_z.map_or(0, |z| z.iter().map(Vec::len).sum())
        + sigma.map_or(0, <[f64]>::len);
    let rec = kj::recorded(&key.family(), || {
        let mut slot = theta.len();
        let th: Vec<kj::Tr> = theta
            .iter()
            .enumerate()
            .map(|(j, &v)| kj::Tr::theta(j, v))
            .collect();
        let et: Vec<kj::Tr> = eta
            .iter()
            .enumerate()
            .map(|(k, &v)| kj::Tr::eta(k, v))
            .collect();
        // The reduced basis's network outputs, parameter slots after θ: the networks stand aside.
        let z: Option<Vec<Vec<kj::Tr>>> = nn_z.map(|z| {
            z.iter()
                .map(|net| {
                    net.iter()
                        .map(|&v| {
                            slot += 1;
                            kj::Tr::theta(slot - 1, v)
                        })
                        .collect()
                })
                .collect()
        });
        // An SDE subject's σ, parameter slots after θ.
        let sg: Option<Vec<kj::Tr>> = sigma.map(|s| {
            s.iter()
                .map(|&v| {
                    slot += 1;
                    kj::Tr::theta(slot - 1, v)
                })
                .collect()
        });
        record(
            model,
            subject,
            lay,
            theta,
            eta,
            &ThetaLeaves::with_nn(&th, z.as_deref()).with_sigma(sg.as_deref()),
            &et,
        )
        .filter(|p| p.len() == n_obs)
        .map(|p| p.iter().map(|x| x.reg()).collect())
    });
    kj::STATS
        .record_us
        .fetch_add(t.elapsed().as_micros() as usize, Relaxed);
    let rec = match rec {
        Ok(r) => r,
        Err((why, model_wide)) => {
            if kj::debug() {
                eprintln!("kjit: subject not recorded: {why}");
            }
            kj::mark_bad(key, &why);
            if model_wide {
                kj::mark_model_bad(key, &why);
            }
            return None;
        }
    };
    if kj::debug() {
        rec.replay_check();
    }
    let layout = terms::layout1(eta.len());
    let Some(plan) = rec.plan(layout, eta.len(), slot).map(std::sync::Arc::new) else {
        kj::mark_bad(key, "the recording could not be planned");
        return None;
    };
    kj::add(key, plan.clone());
    Some(plan)
}

/// The subject's plan whose guard holds at `(theta, eta)` ([`with_plan`]) and its forward at
/// `layout` (η seeded one-hot, θ values): the observations' lanes and the forward, kept for
/// the plan's reverse passes.
fn plan_forward(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
    layout: crate::sens::num::vm::Layout,
) -> Option<(PlanArc, Vec<f64>, kj_fwd::Fwd)> {
    plan_forward_in(model, subject, lay, theta, None, eta, layout)
}

/// [`plan_forward`] of an SDE subject's variance rows when `sigma` is given ([`with_plan_in`]):
/// σ's slots carry values only, their adjoints in the reverse's `p_bar` after θ's.
fn plan_forward_in(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    sigma: Option<&[f64]>,
    eta: &[f64],
    layout: crate::sens::num::vm::Layout,
) -> Option<(PlanArc, Vec<f64>, kj_fwd::Fwd)> {
    let (n, n_obs) = (eta.len(), subject.observations.len());
    let l = layout.lanes();
    let mut u = kj_fwd::Buf::new(n * l);
    for k in 0..n {
        u[k * l] = eta[k];
        if layout.n > 0 {
            u[k * l + 1 + k] = 1.0;
        }
    }
    let mut p = kj_fwd::Buf::new((theta.len() + sigma.map_or(0, <[f64]>::len)) * l);
    for (j, &v) in theta.iter().chain(sigma.unwrap_or(&[])).enumerate() {
        p[j * l] = v;
    }
    let owner = subject as *const Subject as usize;
    with_plan_in(model, subject, lay, theta, sigma, eta, &mut |plan| {
        // The forward the subject's inner solve ended on, at exactly this point: adopted, not
        // computed again ([`keep_forward`]). Only the plain predictions' plan.
        if sigma.is_none() {
            if let Some(fwd) = plan.adopt(owner, layout, theta, eta) {
                let mut du = vec![0.0; n_obs * l];
                for i in 0..n_obs {
                    du[i * l..(i + 1) * l].copy_from_slice(plan.kept_out(&fwd, i));
                }
                if !plan.kept_guard(&fwd) {
                    return None;
                }
                return Some((plan.clone(), du, fwd));
            }
        }
        serve_forward(plan, layout, &u, &p, n_obs)
    })
}

/// `plan`'s forward at `layout` over the inputs `u` (η) and `p` (θ, σ): its observation lanes
/// and the kept forward, `None` where its guard does not hold at them.
fn serve_forward(
    plan: &PlanArc,
    layout: crate::sens::num::vm::Layout,
    u: &[f64],
    p: &[f64],
    n_obs: usize,
) -> Option<(PlanArc, Vec<f64>, kj_fwd::Fwd)> {
    let l = layout.lanes();
    let mut du = vec![0.0; (n_obs + 1) * l];
    let fwd = plan.forward_layout(layout, u, p, &mut du)?;
    if du[n_obs * l] != 1.0 {
        return None;
    }
    du.truncate(n_obs * l);
    Some((plan.clone(), du, fwd))
}

/// [`plan_forward`] at the values-only layout, trying `first` (a plan that served this
/// subject at its mode) before the registry: a quadrature node is the same subject at another
/// η, so its guard usually holds and the lookup is skipped; where it does not, the lookup
/// serves as before.
pub(super) fn plan_forward_f64(
    first: Option<&PlanArc>,
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    eta: &[f64],
) -> Option<(PlanArc, Vec<f64>, kj_fwd::Fwd)> {
    let layout = crate::sens::num::vm::Layout::F64;
    if let Some(plan) = first.filter(|p| {
        p.n_out() == subject.observations.len()
            && p.n_eta() == eta.len()
            && p.n_theta() == theta.len()
    }) {
        let mut u = kj_fwd::Buf::new(eta.len());
        u.copy_from_slice(eta);
        let mut p = kj_fwd::Buf::new(theta.len());
        p.copy_from_slice(theta);
        if let Some(r) = serve_forward(plan, layout, &u, &p, subject.observations.len()) {
            return Some(r);
        }
    }
    plan_forward(model, subject, lay, theta, eta, layout)
}

/// Keep, for the outer gradient, the forward the subject's inner solve last took at `eta`
/// ([`kj_fwd::keep_last_run`]): [`plan_forward`] adopts it at the same `(θ, η̂)` instead of
/// running the plan again.
pub(crate) fn keep_forward(subject: &Subject, theta: &[f64], eta: &[f64]) {
    kj_fwd::keep_last_run(subject as *const Subject as usize, theta, eta);
}

/// The observations' lanes of a plan's forward over `n` random effects: per observation the
/// value, the η-gradient and (second order) the packed η-Hessian, `l` lanes each.
pub(crate) struct PredLanes<D> {
    du: D,
    n: usize,
    l: usize,
}

impl<D: AsRef<[f64]>> PredLanes<D> {
    fn new(du: D, n: usize, second: bool) -> Self {
        let l = 1 + n + if second { n * (n + 1) / 2 } else { 0 };
        PredLanes { du, n, l }
    }
    fn rows(&self) -> usize {
        self.du.as_ref().len() / self.l
    }
    fn lanes(&self, j: usize) -> &[f64] {
        &self.du.as_ref()[j * self.l..(j + 1) * self.l]
    }
    fn v(&self, j: usize) -> f64 {
        self.lanes(j)[0]
    }
    /// Observation `j`'s η-gradient.
    fn row_g(&self, j: usize) -> &[f64] {
        &self.lanes(j)[1..1 + self.n]
    }
    fn g(&self, j: usize, k: usize) -> f64 {
        self.lanes(j)[1 + k]
    }
    /// Observation `j`'s packed η-Hessian.
    fn row_q(&self, j: usize) -> &[f64] {
        &self.lanes(j)[1 + self.n..]
    }
    fn q(&self, j: usize, t: usize) -> f64 {
        self.row_q(j)[t]
    }
    /// The predictions and their η-gradients, as the likelihood kernel reads them.
    fn f_g(&self) -> (Vec<f64>, Vec<Vec<f64>>) {
        (
            (0..self.rows()).map(|j| self.v(j)).collect(),
            (0..self.rows()).map(|j| self.row_g(j).to_vec()).collect(),
        )
    }
}

#[path = "ode_adjoint_kernel.rs"]
mod kernel;
#[path = "ode_adjoint_terms.rs"]
pub(crate) mod terms;
use crate::parser::model_parser::rhs_regcode::kernel_jit as kj_tr;

#[path = "ode_adjoint_agq.rs"]
mod agq;

use crate::parser::model_parser::rhs_regcode::kernel_jit as kj_fwd;

/// Run a subject's compiled plan at `(theta, eta)`: its predictions' value and η-gradient
/// lanes, `None` when a guard fails. Every branch a parameter can move — an absorption lag
/// crossing an observation (`lagged_elapsed`), the event schedule's arrival order, a modeled
/// infusion window — is a guard in the recording (#2786), so a plan whose guard holds is exact.
fn kjit_run(
    plan: &crate::parser::model_parser::rhs_regcode::kernel_jit::Plan,
    theta: &[f64],
    eta: &[f64],
    n_obs: usize,
) -> Option<PredLanes<Vec<f64>>> {
    if plan.n_out() != n_obs {
        return None;
    }
    let mut out = PredLanes::new(vec![0.0; n_obs * (1 + eta.len())], eta.len(), false);
    if !plan.run(theta, eta, &mut out.du) {
        return None;
    }
    Some(out)
}

/// The model the predictor would evaluate for this subject (a closed-form absorption model
/// may route to its ODE twin), the dual width and `eta` in the recordings' basis. The width is
/// `n_eta`, the stacked `[η, κ]` under IOV, or — under nested levels (#1082) — the compact axis
/// count: `eta` arrives per cell and is recorded over the axes it expands from
/// ([`StackedLayout::sens_of`](crate::stats::stacked_layout::StackedLayout::sens_of)), so each
/// derivative is over the vector the EBE search moves. `None` when `eta` is not the subject's
/// width, and for a nested-level subject no recording lays out (see
/// [`nested_levels_outside_scope`]).
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
    let (eff, w, e, lay) = if model.n_kappa > 0 {
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
        let mut e = crate::scratch::take_with_capacity(eta.len());
        e.extend_from_slice(eta);
        (
            crate::pk::effective_model_for_eval(model, subject, theta, eta),
            model.n_eta,
            e,
            None,
        )
    };
    Some((eff, w, e, lay))
}

/// The number of random effects a recording of `subject` is over: `n_eta`, the stacked
/// `[η, κ]` under IOV, or the compact axes under nested levels (tests' fixtures only).
#[cfg(test)]
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
    let (model, w, eta, lay) = resolve(model, subject, theta, eta, None)?;
    let p = canonical_jets1(model, subject, lay.as_deref(), theta, None, &eta)?;
    Some((
        (0..p.rows()).map(|j| p.v(j)).collect(),
        DMatrix::from_fn(p.rows(), w, |j, k| p.g(j, k)),
    ))
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
    let (model, w, eta, lay) = resolve(model, subject, theta, eta, None)?;
    let p = canonical_jets1(model, subject, lay.as_deref(), theta, None, &eta)?;
    Some(DMatrix::from_fn(p.rows(), w, |j, k| p.g(j, k)))
}

/// [`exact_eta_jacobian`]'s κ columns alone (`n_obs × width`, the BSV η columns left zero),
/// for callers that already hold the η columns (the IOV linearisation, the
/// importance-sampling proposal): the subject's first-order plan's (#2786), which carries
/// every column. `None` without IOV, or where [`exact_eta_jacobian`] declines.
pub(crate) fn exact_kappa_jacobian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<DMatrix<f64>> {
    if model.n_kappa == 0 {
        return None;
    }
    let mut jac = exact_eta_jacobian(model, subject, theta, eta)?;
    let n_bsv = model.n_eta.min(jac.ncols());
    if jac.ncols() == n_bsv {
        return None;
    }
    jac.columns_mut(0, n_bsv).fill(0.0);
    Some(jac)
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
    exact_obs_grads_in(model, subject, None, theta, None, eta)
}

/// [`exact_obs_grads`] over a caller-built `layout` (the subject's
/// [`StackedLayout::sens_of`](crate::stats::stacked_layout::StackedLayout::sens_of) under
/// IOV), so an EBE search builds it once rather than once per gradient (#2037). With `sigma`,
/// an SDE subject's rows are followed by its EKF variance rows ([`exact_obs_grads_into`]).
pub(crate) fn exact_obs_grads_in(
    model: &CompiledModel,
    subject: &Subject,
    layout: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    sigma: Option<&[f64]>,
    eta: &[f64],
) -> Option<Vec<crate::sens::provider::ObsGrad>> {
    let mut rows = Vec::new();
    exact_obs_grads_into(model, subject, layout, theta, sigma, eta, &mut rows).then_some(rows)
}

/// The exact θ-derivatives of the predictions at a fixed `eta`, contracted with the
/// caller's per-observation `weights`: `Σ_j w_j ∂f_j/∂θ_m` for every θ coordinate (raw θ,
/// not packed), by one recording on the vector tape — θ as leaves, η as plain values — and
/// one reverse sweep seeded with `w`. With `want_jac`, also every `∂f_j/∂θ_m`
/// (row-major `n_obs × n_theta`), from sweeps of eight observations at a time.
///
/// The fixed-η score of SAEM's M-step and of variational inference, where η is a draw
/// rather than a mode: the cost of one solve and one sweep however many θ there are, where
/// a difference quotient pays one solve per θ. `None` outside the recorders' scope (the
/// caller keeps its own path), on a recording whose length is not the subject's
/// observation count, or when `weights` is not one per observation.
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
    weights: &[f64],
) -> Option<FixedEtaTheta> {
    let (mut rows, jac) =
        fixed_eta_theta_sweeps(model, subject, theta, None, eta, want_jac, |f| {
            (weights.len() == f.len()).then(|| vec![weights.to_vec()])
        })?;
    let grad = rows.pop()?;
    Some(FixedEtaTheta {
        jac: jac.map(|j| j.into_iter().flatten().collect()),
        grad,
    })
}

/// `Σ_j w_j ∂f_j/∂θ_m` for each weight row `w` the caller computes from the predictions
/// `f` at a fixed `eta`, for every θ coordinate (raw θ; a θ no recording reads has a zero
/// derivative). One recording on the vector tape — θ as leaves, η as plain values — and one
/// reverse sweep per (up to) eight rows, however many θ there are: the exact replacement for
/// a difference quotient of the predictions in θ. `None` outside the recorders' scope, on a
/// recording whose length is not the subject's observation count, when `weights`
/// declines, or on a non-finite result.
pub(crate) fn fixed_eta_theta_contractions(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    weights: impl FnOnce(&[f64]) -> Option<Vec<Vec<f64>>>,
) -> Option<Vec<Vec<f64>>> {
    fixed_eta_theta_sweeps(model, subject, theta, None, eta, false, weights).map(|(rows, _)| rows)
}

/// [`fixed_eta_theta_contractions`] of an SDE subject's kernel rows: `weights` gets the `2n`
/// rows the kernel scores (the predictions `f`, then the EKF's variances `p_obs`, both at the
/// fixed `eta`) and returns weight rows of that length; each contraction is over `θ` then `σ`
/// (the Kalman `R` reads σ), the variance rows' adjoints reversed through their own plan
/// ([`record_sde_p`]). Any other model is [`fixed_eta_theta_contractions`]'s.
pub(crate) fn fixed_eta_theta_sigma_contractions(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: &[f64],
    eta: &[f64],
    weights: impl FnOnce(&[f64]) -> Option<Vec<Vec<f64>>>,
) -> Option<Vec<Vec<f64>>> {
    fixed_eta_theta_sweeps(model, subject, theta, Some(sigma), eta, false, weights)
        .map(|(rows, _)| rows)
}

/// The plan reverses behind [`fixed_eta_theta_contractions`] and [`fixed_eta_theta_adjoint`]:
/// the subject's plan, its value forward at `(θ, eta)` kept, and one value
/// reverse seeded with every weight row (and, with `want_jac`, one with every observation's
/// unit seed): the rows' contractions over raw θ.
fn fixed_eta_theta_sweeps(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: Option<&[f64]>,
    eta: &[f64],
    want_jac: bool,
    weights: impl FnOnce(&[f64]) -> Option<Vec<Vec<f64>>>,
) -> Option<(Vec<Vec<f64>>, Option<Vec<Vec<f64>>>)> {
    use crate::sens::num::vm::Layout;
    // An SDE subject's score has variance rows beside the predictions: the callers that hand σ
    // ([`fixed_eta_theta_sigma_contractions`]) weight them; the rest contract the predictions
    // alone (FOCE's population prediction at η = 0, whose variance rows are not in the score).
    if !recorder_in_scope(model) || (model.is_sde() && want_jac && sigma.is_none()) {
        return None;
    }
    let sigma = sigma.filter(|_| model.is_sde());
    let n_sig = sigma.map_or(0, <[f64]>::len);
    // The recordings' basis, as every sibling entry point resolves it: the model the
    // predictor evaluates (a transit/IG twin), and under nested levels `eta` taken from per
    // cell to the compact axes `record` expands from.
    let (rec_model, _, axes, lay) = resolve(model, subject, theta, eta, None)?;
    if !std::ptr::eq(rec_model, model) && !recorder_in_scope(rec_model) {
        return None;
    }
    let n_obs = subject.observations.len();
    if n_obs == 0 {
        // No prediction rows (an endpoint-only subject): every contraction is zero.
        let rows = weights(&[])?;
        let zero = |n: usize| vec![vec![0.0; theta.len() + n_sig]; n];
        return Some((zero(rows.len()), want_jac.then(|| zero(0))));
    }
    let (n_eta, plan, vals, fwd) = with_plan_width(
        rec_model,
        subject,
        lay.as_deref(),
        theta,
        &axes,
        &mut |pn, plan| {
            let mut u = axes.clone();
            u.resize(pn, 0.0);
            let mut du = vec![0.0; n_obs + 1];
            let fwd = plan.forward_layout(Layout::F64, &u, theta, &mut du)?;
            (du[n_obs] == 1.0).then(|| (pn, plan.clone(), du[..n_obs].to_vec(), fwd))
        },
    )?;
    // A FREM covariate pseudo-observation predicts `θ_i + η_e` (the recorders return the
    // PK prediction there, as every caller of theirs patches): its derivative is the unit
    // vector on `θ_i`, added after the reverse rather than seeded.
    let frem: Vec<Option<usize>> = (0..n_obs)
        .map(|j| {
            let fc = model.frem_config.as_ref()?;
            let ft = subject.fremtype.get(j).filter(|&&ft| ft != 0)?;
            fc.fremtype_to_indices.get(ft).map(|&(ti, _)| ti)
        })
        .collect();
    let f: Vec<f64> = (0..n_obs)
        .map(|j| match (frem[j], model.frem_config.as_ref()) {
            (Some(ti), Some(fc)) => {
                let ei = fc.fremtype_to_indices[&subject.fremtype[j]].1;
                theta.get(ti).copied().unwrap_or(0.0) + eta.get(ei).copied().unwrap_or(0.0)
            }
            _ => vals[j],
        })
        .collect();
    // An SDE subject's variance rows: the EKF's `p_obs` at the same point, a plan of their
    // own over θ then σ, scored after the predictions.
    let planned_p = match sigma {
        Some(s) => Some(plan_forward_in(
            rec_model,
            subject,
            lay.as_deref(),
            theta,
            Some(s),
            &axes,
            Layout::F64,
        )?),
        None => None,
    };
    let f_all: Vec<f64> = f
        .iter()
        .copied()
        .chain(planned_p.iter().flat_map(|p| p.1.iter().copied()))
        .collect();
    let rows = weights(&f_all)?;
    if rows.iter().any(|w| w.len() != f_all.len()) {
        return None;
    }
    let n_th = theta.len();
    let n_cols = n_th + n_sig;
    // Every row a seed of one value reverse; a FREM row's weight goes to its θ directly.
    let contract = |rows: &dyn Fn(usize, usize) -> f64, n_rows: usize| -> Option<Vec<Vec<f64>>> {
        let mut du_bar = vec![0.0; n_rows * n_obs];
        for r in 0..n_rows {
            for j in (0..n_obs).filter(|&j| frem[j].is_none()) {
                du_bar[r * n_obs + j] = rows(r, j);
            }
        }
        let (mut ub, mut pb) = (vec![0.0; n_rows * n_eta], vec![0.0; n_rows * n_th]);
        if !plan.reverse_from(&fwd, Layout::F64, n_rows, &mut ub, &mut pb, &du_bar) {
            return None;
        }
        // The variance rows' seeds through their plan, to θ then σ.
        let mut pv = vec![0.0; n_rows * n_cols];
        if let Some((pp, _, pf)) = planned_p.as_ref() {
            let mut du_bar = vec![0.0; n_rows * n_obs];
            for r in 0..n_rows {
                for j in 0..n_obs {
                    du_bar[r * n_obs + j] = rows(r, n_obs + j);
                }
            }
            let mut ub = vec![0.0; n_rows * n_eta];
            if !pp.reverse_from(pf, Layout::F64, n_rows, &mut ub, &mut pv, &du_bar) {
                return None;
            }
        }
        Some(
            (0..n_rows)
                .map(|r| {
                    let mut g = vec![0.0; n_cols];
                    g[..n_th].copy_from_slice(&pb[r * n_th..(r + 1) * n_th]);
                    for (a, b) in g.iter_mut().zip(&pv[r * n_cols..(r + 1) * n_cols]) {
                        *a += b;
                    }
                    for (j, ti) in frem.iter().enumerate() {
                        if let Some(gm) = ti.and_then(|ti| g.get_mut(ti)) {
                            *gm += rows(r, j);
                        }
                    }
                    g
                })
                .collect(),
        )
    };
    let dense = if rows.is_empty() {
        Vec::new()
    } else {
        contract(&|r, j| rows[r][j], rows.len())?
    };
    let jac = if want_jac {
        Some(contract(&|r, j| if r == j { 1.0 } else { 0.0 }, n_obs)?)
    } else {
        None
    };
    let finite = |x: &Vec<Vec<f64>>| x.iter().flatten().all(|v| v.is_finite());
    (finite(&dense) && jac.as_ref().is_none_or(finite)).then_some((dense, jac))
}

/// Whether the dual hazard walk ([`ode_states_at_g`], [`cumhaz_hazard_from_states`]) serves
/// `subject`: a compiled RHS and parameter program, absorption forcings the dual walk lifts,
/// and either walk's scope — the static walk's, or the event-driven walk's for what the
/// static one has no run-in or snapshot for (a `TIME`-reading `$PK`, a lag, a steady state,
/// a modeled dose, resets). The routing predicates ask this, so a subject the walks decline
/// is reported on FD rather than labelled analytic.
#[cfg(feature = "survival")]
pub(crate) fn ode_hazard_walk_in_scope(model: &CompiledModel, subject: &Subject) -> bool {
    let Some(ode) = model.ode_spec.as_ref() else {
        return false;
    };
    // The compiled RHS program is required by both walk scopes below (a second copy of that
    // gate here could never change the answer).
    ode.indiv_param_program.is_some()
        && ode.input_rate.iter().all(|f| f.kind.supported_over_dual())
        && (static_hazard_walk(model, subject) || event_walk_subject_supported(model, subject))
}

/// Whether the hazard solve takes the static walk: no steady state and the static walk's
/// scope. Anything else takes the event-driven walk.
#[cfg(feature = "survival")]
fn static_hazard_walk(model: &CompiledModel, subject: &Subject) -> bool {
    !subject.has_ss_doses() && static_walk_subject_supported(model, subject)
}

/// `H(t) = u[chz_state]` and `h(t) = u̇[chz_state]` at `times` from a solve saved there
/// ([`ode_states_at_g`]): `h` is the RHS at the saved state, read at the value path's clock
/// anchors (`TAFD` from the first dose record, `TAD` from the last arrival by `t`, each
/// arrival carrying its dose's lag — [`tad_anchor_g`]).
#[cfg(feature = "survival")]
pub(crate) fn cumhaz_hazard_from_states<T: VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    chz_state: usize,
    times: &[f64],
    solved: &DualStates<T>,
) -> Option<(Vec<T>, Vec<T>)> {
    let ode = model.ode_spec.as_ref()?;
    let program = ode.rhs_program.as_ref()?;
    if solved.states.len() != times.len() {
        return None;
    }
    let mut du = vec![T::from_f64(0.0); ode.n_states];
    let (mut vars, mut stack) = (Vec::new(), Vec::new());
    let mut cum = Vec::with_capacity(times.len());
    let mut haz = Vec::with_capacity(times.len());
    for (st, &t) in solved.states.iter().zip(times) {
        cum.push(st[chz_state]);
        super::eval_rhs_anchored::<T>(
            program,
            st,
            &solved.params,
            T::from_f64(t),
            solved.first_dose_time,
            tad_anchor_g(&subject.doses, &solved.dose_lags, t),
            &mut du,
            &mut vars,
            &mut stack,
        );
        haz.push(du[chz_state]);
    }
    Some((cum, haz))
}

/// [`crate::ode::predictions::tad_anchor_for`] with each dose's lag a number of type `T`:
/// the referent [`crate::dosing::tad_referent`] picks — an arrival, a steady state's last
/// cycle start before `t`, or a seeded steady state's pre-record pulse — as an expression in
/// its dose's lag, and every choice between doses through the branch API, so a recording
/// follows the lags. `lags` may be shorter than `doses` (a missing entry is no lag).
#[cfg(feature = "survival")]
fn tad_anchor_g<T: VmNum>(doses: &[crate::types::DoseEvent], lags: &[T], t: f64) -> T {
    use crate::sens::num::BranchCmp::{Ge, Gt, Le, Lt};
    let tf = T::from_f64;
    let lag = |i: usize| lags.get(i).copied().unwrap_or(tf(0.0));
    let eps = 1e-12; // `dosing::TAD_EPS`
    let referent = |i: usize, d: &crate::types::DoseEvent| -> Option<T> {
        let arrival = tf(d.time) + lag(i);
        if arrival.br_cmp_c(Le, t + eps) {
            if d.ss && d.ii > 0.0 {
                // `t − (t − arrival).rem_euclid(II)`: the arrival plus its whole cycles by `t`.
                let elapsed = tf(t) - arrival;
                let m = (elapsed.seen_value() / d.ii).floor();
                let _ = elapsed.br_cmp_c(Ge, m * d.ii);
                let _ = elapsed.br_cmp_c(Lt, (m + 1.0) * d.ii);
                return Some(arrival + tf(m * d.ii));
            }
            return Some(arrival);
        }
        (crate::dosing::ss_seeded_at_record_g(d, lag(i)) && d.time <= t + eps)
            .then(|| tf(d.time) - crate::dosing::ss_seed_phase_g(d, lag(i)))
    };
    let mut best: Option<T> = None;
    for (i, d) in doses.iter().enumerate() {
        if let Some(r) = referent(i, d) {
            best = Some(match best {
                Some(b) if !r.br_cmp(Gt, b) => b,
                _ => r,
            });
        }
    }
    // No dose has arrived: the earliest arrival (`NaN` for a dose-free subject).
    best.unwrap_or_else(|| {
        doses
            .iter()
            .enumerate()
            .map(|(i, d)| tf(d.time) + lag(i))
            .reduce(|a, b| if b.br_cmp(Lt, a) { b } else { a })
            .unwrap_or(tf(f64::NAN))
    })
}

/// [`ode_states_at_g`]'s result: the ODE state at each requested time, with the parameter
/// vector (the disposition) and first-dose time the solve ran under, and each dose's lag
/// (empty without one).
#[cfg(feature = "survival")]
pub(crate) struct DualStates<T> {
    pub(crate) states: Vec<Vec<T>>,
    pub(crate) params: Vec<T>,
    pub(crate) first_dose_time: f64,
    pub(crate) dose_lags: Vec<T>,
}

/// The subject's ODE state at `times` (sorted, non-decreasing) on any VM number type — what a
/// state-driven endpoint (an ODE-accumulated hazard, a drug-driven CTMM intensity) reads —
/// solved as production's single-pass hazard solve (`survival::ode_cumhaz_hazard`) solves it:
/// one disposition snapshot, `$PK` at `TIME = 0` (#610), with every dose quantity (kernel, lag,
/// `F`, modeled `D`/`R`, the steady-state run-in) read at its own dose record (#1575). The
/// static dual walk (`integrate_g`) serves the subjects in its scope, where the two readings
/// coincide; the event-driven walk (`integrate_tvcov_g`, every segment on the disposition)
/// every other. `None` outside both, or on a non-finite state.
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
    // The values serve only the slots the tape shows constant in θ and η (`seen_value`); a
    // live slot takes its tape value below.
    let eta_v: Vec<f64> = eta.iter().map(|e| e.seen_value()).collect();
    let theta_v: Vec<f64> = theta.iter().map(|t| t.seen_value()).collect();
    let snapshot = |cov: &std::collections::HashMap<String, f64>, time: f64| -> Vec<T> {
        let pk = (model.pk_param_fn)(&theta_v, &eta_v, cov, time);
        let by_slot = taped_params(
            model,
            prog,
            pk.values.len(),
            &ThetaLeaves::theta(theta),
            eta,
            cov,
        );
        by_slot
            .iter()
            .zip(&pk.values)
            .map(|(d, &v)| d.unwrap_or(T::from_f64(v)))
            .collect()
    };
    let params = {
        let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, 0.0);
        snapshot(&subject.covariates, 0.0)
    };
    let init =
        super::init_seed::<T>(ode, &params).unwrap_or_else(|| vec![T::from_f64(0.0); ode.n_states]);
    let first_dose_time = subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min);
    let (states, dose_lags) = if static_hazard_walk(model, subject) {
        if params[crate::types::PK_IDX_LAGTIME]
            .abs()
            .br_cmp_c(crate::sens::num::BranchCmp::Gt, 1e-12)
        {
            return None;
        }
        let dose_f_bio: Vec<T> = subject
            .doses
            .iter()
            .map(|d| params[super::f_bio_slot(ode, d.cmt_raw())])
            .collect();
        let forcings =
            super::PreparedForcings::shared(ode, &params, super::prepare_dual_forcing::<T>);
        // The walks save at the requested times rather than the subject's observations.
        let states = super::integrate_g::<T>(
            program,
            ode.n_states,
            subject,
            times,
            ode,
            &forcings,
            &params,
            &dose_f_bio,
            &init,
            first_dose_time,
            &ode.effective_solver_opts(),
        )?;
        (states, Vec::new())
    } else {
        let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
        let (pk_at_dose, ..) =
            seed_tvcov_snapshots::<T>(model, subject, uses_time, |cov, time| {
                let _time_guard =
                    crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
                Some(snapshot(cov, time))
            })?;
        let attr_map = model.active_dose_attr_map();
        let lag_slot: Vec<usize> = if model.has_lagtime() {
            subject
                .doses
                .iter()
                .map(|d| attr_map.lag_slot(d.cmt_raw()))
                .collect()
        } else {
            Vec::new()
        };
        let modeled_slot: Vec<_> = if subject.all_doses_fixed() {
            Vec::new()
        } else {
            subject
                .doses
                .iter()
                .map(|d| modeled_slot_for(attr_map, d))
                .collect()
        };
        let f_bio: Vec<T> = subject
            .doses
            .iter()
            .zip(&pk_at_dose)
            .map(|(d, p)| p[super::f_bio_slot(ode, d.cmt_raw())])
            .collect();
        let same = |n: usize| vec![params.clone(); n];
        let states = integrate_tvcov_g::<T>(
            program,
            ode,
            ode.n_states,
            subject,
            times,
            &pk_at_dose,
            &same(times.len()),
            &same(subject.pk_only_times.len()),
            &same(subject.reset_times.len()),
            &f_bio,
            &init,
            first_dose_time,
            &lag_slot,
            &modeled_slot,
            &ode.effective_solver_opts(),
            Some(&params),
        );
        let lags = lag_slot
            .iter()
            .zip(&pk_at_dose)
            .map(|(&s, p)| p[s])
            .collect();
        (states, lags)
    };
    // A non-finite state declines: the static walk anchors `TAD` by a fold over the doses
    // at or before each segment, so a hazard reading it before the first dose integrates a
    // `NaN` the value path (anchored at the first arrival) does not.
    (states.len() == times.len() && states.iter().flatten().all(|v| v.br_abs_lt(f64::INFINITY)))
        .then_some(DualStates {
            states,
            params,
            first_dose_time,
            dose_lags,
        })
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
    let mut params = model.default_params.clone();
    params.theta = theta.to_vec();
    params.sigma.values = sigma.to_vec();
    params.residual_correlations = model.residual_correlations.clone();
    let n_obs = subject.observations.len();
    let ctx = LikCtx::new(model, subject, &params);
    let pv = ParT::of(model, subject, &params);
    // The data term on the scalar tape, scored inside the recording's weights callback on
    // its own prediction values (an SDE subject's `2n` rows: the predictions, then the EKF
    // variances): one recording gives the value and the θ-derivatives, σ's through the
    // variance rows' plan after θ's.
    let scored: std::cell::RefCell<Option<(f64, Vec<f64>, Vec<f64>)>> = Default::default();
    let d_theta_f = fixed_eta_theta_sigma_contractions(model, subject, theta, sigma, eta, |f| {
        let mut inp = kernel::Inputs::default();
        let (pr, _) = param_leaves(&pv, &StackedOmega::empty(), &mut inp);
        let fr: Vec<kernel::Leaf> = f.iter().map(|&v| inp.leaf(v)).collect();
        let er: Vec<kernel::Leaf> = eta.iter().map(|&v| inp.leaf(v)).collect();
        let kr = kernel::kernel_run(
            model,
            subject,
            kernel::variant_of(&(3u8, n_obs, eta.len())),
            crate::sens::num::vm::Layout::F64,
            &inp.vals,
            1,
            &|lv| {
                let t = |l: kernel::Leaf| l.tr(lv);
                let f: Vec<kj_tr::Tr> = fr.iter().map(|&l| t(l)).collect();
                let e: Vec<kj_tr::Tr> = er.iter().map(|&l| t(l)).collect();
                let p = param_tr(model, subject, &pr, lv)?;
                let half = <kj_tr::Tr as crate::sens::num::PkNum>::from_f64(0.5);
                Some(vec![half * data_l(&ctx, &f, &e, &p, None)?])
            },
        )?;
        let sadj = kr.reverse_one(&[1.0])?;
        let adj = |x: kernel::Leaf| -> f64 { x.adj(&sadj) };
        // θ read directly (FREM rows, a residual magnitude), and the serial ρ's θ.
        let mut d_direct: Vec<f64> = pr.theta.iter().map(|&t| adj(t)).collect();
        if let (Some(sr), Some(rs)) = (pr.serial_rho, model.residual_serial.as_ref()) {
            if let Some(d) = d_direct.get_mut(rs.rho_theta) {
                *d += adj(sr);
            }
        }
        let d_sigma: Vec<f64> = pr.sigma.iter().map(|&x| adj(x)).collect();
        let wf: Vec<f64> = fr.iter().map(|&x| adj(x)).collect();
        *scored.borrow_mut() = Some((kr.value(0), d_direct, d_sigma));
        Some(vec![wf])
    })?
    .pop()?;
    let (v, d_direct, mut d_sigma) = scored.into_inner()?;
    let mut d_theta = d_theta_f;
    for (d, e) in d_theta.iter_mut().zip(&d_direct) {
        *d += e;
    }
    // The variance rows' σ-adjoints, after θ's.
    for (d, e) in d_sigma.iter_mut().zip(d_theta.split_off(theta.len())) {
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
    // One recording: the predictions' values and their η-Jacobian — for an SDE subject also
    // its EKF variance rows' (the kernel's rows `n..2n`).
    let (f, jac) = if model.is_sde() {
        let (rec, w, eta_p, lay) = resolve(model, subject, theta, eta, None)?;
        let p = canonical_jets1(rec, subject, lay.as_deref(), theta, Some(sigma), &eta_p)?;
        (
            (0..p.rows()).map(|j| p.v(j)).collect::<Vec<f64>>(),
            DMatrix::from_fn(p.rows(), w, |j, k| p.g(j, k)),
        )
    } else {
        eta_values_and_jacobian(model, subject, theta, eta)?
    };
    let n_rows = if model.is_sde() { 2 * n_obs } else { n_obs };
    if f.len() != n_rows || jac.nrows() != n_rows || jac.ncols() != eta.len() {
        return None;
    }
    let ctx = LikCtx::new(model, subject, &params);
    let pv = ParT::of(model, subject, &params);
    let mut inp = kernel::Inputs::default();
    let (pr, _) = param_leaves(&pv, &StackedOmega::empty(), &mut inp);
    let fr: Vec<kernel::Leaf> = f.iter().map(|&v| inp.leaf(v)).collect();
    let er: Vec<kernel::Leaf> = eta.iter().map(|&v| inp.leaf(v)).collect();
    let kr = kernel::kernel_run(
        model,
        subject,
        kernel::variant_of(&(3u8, n_obs, eta.len())),
        crate::sens::num::vm::Layout::F64,
        &inp.vals,
        1,
        &|lv| {
            let t = |l: kernel::Leaf| l.tr(lv);
            let f: Vec<kj_tr::Tr> = fr.iter().map(|&l| t(l)).collect();
            let e: Vec<kj_tr::Tr> = er.iter().map(|&l| t(l)).collect();
            let p = param_tr(model, subject, &pr, lv)?;
            let half = <kj_tr::Tr as crate::sens::num::PkNum>::from_f64(0.5);
            Some(vec![half * data_l(&ctx, &f, &e, &p, None)?])
        },
    )?;
    let sadj = kr.reverse_one(&[1.0])?;
    let adj = |x: kernel::Leaf| -> f64 { x.adj(&sadj) };
    let g: Vec<f64> = (0..eta.len())
        .map(|k| adj(er[k]) + (0..n_rows).map(|j| adj(fr[j]) * jac[(j, k)]).sum::<f64>())
        .collect();
    g.iter().all(|v| v.is_finite()).then_some(g)
}

/// [`exact_obs_grads`] into `rows`, reusing its allocations (the inner loop calls this
/// once per BFGS iteration). With `sigma`, an SDE subject's `n` prediction rows are followed
/// by its `n` EKF variance rows `p_obs` and their η-jets (the kernel's rows `n..2n`, from the
/// same canonical recording). `false` outside the recorders' scope or on a non-finite row,
/// leaving `rows` unspecified.
pub(crate) fn exact_obs_grads_into(
    model: &CompiledModel,
    subject: &Subject,
    layout: Option<&crate::stats::stacked_layout::StackedLayout>,
    theta: &[f64],
    sigma: Option<&[f64]>,
    eta: &[f64],
    rows: &mut Vec<crate::sens::provider::ObsGrad>,
) -> bool {
    let Some((model, _, eta_p, lay)) = resolve(model, subject, theta, eta, layout) else {
        return false;
    };
    let p = canonical_jets1(model, subject, lay.as_deref(), theta, sigma, &eta_p);
    crate::scratch::give(eta_p);
    let Some(p) = p else {
        return false;
    };
    rows.resize_with(p.rows(), || crate::sens::provider::ObsGrad {
        f: 0.0,
        df_deta: Vec::new(),
    });
    for (j, row) in rows.iter_mut().enumerate() {
        let (f, g) = (p.v(j), p.row_g(j));
        if !(f.is_finite() && g.iter().all(|v| v.is_finite())) {
            crate::scratch::give(p.du);
            return false;
        }
        row.f = f;
        row.df_deta.clear();
        row.df_deta.extend_from_slice(g);
    }
    crate::scratch::give(p.du);
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

/// IOV scores FREM rows by a sentinel, not the kernel's `l`, so the kernel's exact inner
/// Hessian is not that subject's (as `exact_htilde` and `exact_inner_gradient_in` decline).
/// The one decline of [`exact_inner_hessian`] that a quadrature gradient pass does not
/// share, so that pass hands back no Hessian there either (#2585 review #4).
fn iov_frem_mismatch(model: &CompiledModel, subject: &Subject) -> bool {
    model.n_kappa > 0 && subject.fremtype.iter().any(|&ft| ft > 0)
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
    let (model, w, eta, lay) = resolve(model, subject, &params.theta, eta, None)?;
    if iov_frem_mismatch(&model, subject) {
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
    // An endpoint-only subject has no Gaussian rows to record.
    let p = if subject.observations.is_empty() && !subject.obs_records.is_empty() {
        PredLanes::new(Vec::new(), w, true)
    } else {
        canonical_jets(
            model,
            subject,
            lay.as_deref(),
            &params.theta,
            Some(&params.sigma.values),
            &eta,
        )?
    };
    let ctx = LikCtx::new(model, subject, params);
    let mut pv = ParT::of(model, subject, params);
    pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(
        stacked_omega(model, subject, params, params, w).chol,
    )?;
    let f: Vec<f64> = (0..p.rows()).map(|j| p.v(j)).collect();
    let g: Vec<Vec<f64>> = (0..p.rows()).map(|j| p.row_g(j).to_vec()).collect();
    let q: Vec<Vec<f64>> = (0..p.rows()).map(|j| p.row_q(j).to_vec()).collect();
    let h = crate::sens::focei_kernel::exact_inner_hessian_kernel(&ctx, &f, &g, &q, &eta, &pv)?;
    let mut h = DMatrix::from_fn(w, w, |a, b| h[a][b]);
    if !h.iter().all(|v| v.is_finite()) {
        return None;
    }
    if let Some(q) = ng {
        let n = q.nrows();
        let mut blk = h.view_mut((0, 0), (n, n));
        blk += q;
    }
    Some(h)
}

/// Whether the adjoint's kernels serve a subject's non-Gaussian records: every one has an
/// exact channel (`inner_optimizer::non_gaussian_eta_grad_served`) and the model has no SDE.
/// The terms read BSV η only, so under IOV they occupy the leading block of the stacked
/// `[η, κ]`.
fn non_gaussian_terms_supported(model: &CompiledModel, subject: &Subject) -> bool {
    crate::estimation::inner_optimizer::non_gaussian_eta_grad_served(model, subject)
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
    // An endpoint-only subject has no Gaussian rows to record.
    let p = if subject.observations.is_empty() && !subject.obs_records.is_empty() {
        PredLanes::new(Vec::new(), w, false)
    } else {
        canonical_jets1(
            model,
            subject,
            lay.as_deref(),
            &params.theta,
            Some(&params.sigma.values),
            &eta,
        )?
    };
    let ctx = LikCtx::new(model, subject, params);
    let mut pv = ParT::of(model, subject, params);
    pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(
        stacked_omega(model, subject, params, params, w).chol,
    )?;
    let (f, g) = p.f_g();
    let ht = crate::sens::focei_kernel::focei_htilde(&ctx, &f, &g, &eta, &pv)?;
    let mut h = DMatrix::from_fn(w, w, |a, b| ht[a][b]);
    if !h.iter().all(|v| v.is_finite()) {
        return None;
    }
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
    let (model, w, eta, lay) = resolve(model, subject, &params.theta, eta, layout)?;
    // IOV scores FREM rows by a sentinel, not the kernel's `l`.
    let iov_mismatch = model.n_kappa > 0 && subject.fremtype.iter().any(|&ft| ft > 0);
    if iov_mismatch || !likelihood_supported(model, subject, params) {
        return None;
    }
    let p = canonical_jets1(
        model,
        subject,
        lay.as_deref(),
        &params.theta,
        Some(&params.sigma.values),
        &eta,
    )?;
    let ctx = LikCtx::new(model, subject, params);
    let mut pv = ParT::of(model, subject, params);
    pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(
        stacked_omega(model, subject, params, params, w).chol,
    )?;
    let (f, g) = p.f_g();
    let gr = inner_grad(&ctx, &f, &g, &eta, &pv, frozen)?;
    gr.iter().all(|v| v.is_finite()).then_some(gr)
}

/// Ω as the stacked prior's Cholesky factor (`L ⊕ (I_K ⊗ L_IOV)` under IOV), and per
/// packed Ω / Ω_IOV entry its packed index, the stacked positions it fills (one per
/// occasion block for Ω_IOV) and its packing chain (`L_kk` for the log-diagonal).
struct StackedOmega {
    chol: Vec<Vec<f64>>,
    placed: Vec<(usize, Vec<(usize, usize)>, f64)>,
}

impl StackedOmega {
    /// No prior: for a kernel that scores the data term alone.
    fn empty() -> Self {
        StackedOmega {
            chol: Vec::new(),
            placed: Vec::new(),
        }
    }
}

fn stacked_omega(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    params: &ModelParameters,
    active: usize,
) -> StackedOmega {
    let (n_eta, omega) = (model.n_eta, &params.omega);
    let seg = crate::estimation::parameterization::packed_segments(template);
    let mut chol = vec![vec![0.0; active]; active];
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
    StackedOmega { chol, placed }
}

/// [`tape_params`] as a kernel plan's inputs: Ω by its packed Cholesky entries (one leaf per
/// entry, shared across occasion blocks), σ, ρ, the serial ρ
/// and every θ — a plan replays at new values, so nothing parameter-dependent may enter as a
/// constant (a θ the kernel does not read costs nothing: the plan drops it). Ω's inverse and
/// log-determinant are formed from the factor inside the recording ([`param_tr`]).
fn param_leaves(
    pv: &ParT<f64>,
    so: &StackedOmega,
    inp: &mut kernel::Inputs,
) -> (ParT<kernel::Leaf>, Vec<kernel::Leaf>) {
    use kernel::Leaf;
    let mut lr: Vec<Vec<Leaf>> = so
        .chol
        .iter()
        .map(|row| row.iter().map(|&x| Leaf::c(x)).collect())
        .collect();
    let om: Vec<Leaf> = so
        .placed
        .iter()
        .map(|(_, at, _)| {
            let leaf = inp.leaf(so.chol[at[0].0][at[0].1]);
            for &(i, j) in at {
                lr[i][j] = leaf;
            }
            leaf
        })
        .collect();
    let pr = ParT {
        omega: crate::sens::focei_kernel::OmegaT {
            chol: lr,
            inv: Vec::new(),
            log_det: Leaf::c(0.0),
        },
        sigma: pv.sigma.iter().map(|&x| inp.leaf(x)).collect(),
        rho: pv.rho.iter().map(|&x| inp.leaf(x)).collect(),
        serial_rho: pv.serial_rho.map(|x| inp.leaf(x)),
        theta: pv.theta.iter().map(|&x| inp.leaf(x)).collect(),
        ruv_mult: None,
    };
    (pr, om)
}

/// [`param_leaves`]' parameters on the recording: Ω from its factor, and a custom residual
/// magnitude from its compiled programs over the θ leaves.
fn param_tr(
    model: &CompiledModel,
    subject: &Subject,
    pr: &ParT<kernel::Leaf>,
    lv: &[kj_tr::Tr],
) -> Option<ParT<kj_tr::Tr>> {
    let t = |l: kernel::Leaf| l.tr(lv);
    let chol: Vec<Vec<kj_tr::Tr>> = pr
        .omega
        .chol
        .iter()
        .map(|row| row.iter().map(|&l| t(l)).collect())
        .collect();
    let theta: Vec<kj_tr::Tr> = pr.theta.iter().map(|&l| t(l)).collect();
    let ruv_mult = match model.ruv_magnitude.as_ref().filter(|rm| rm.is_active()) {
        Some(rm) => Some(
            (0..subject.observations.len())
                .map(|j| rm.eval_subject_obs_g(&model.error_spec, subject, j, &theta))
                .collect::<Option<Vec<_>>>()?,
        ),
        None => None,
    };
    Some(ParT {
        omega: crate::sens::focei_kernel::OmegaT::from_chol(chol)?,
        sigma: pr.sigma.iter().map(|&l| t(l)).collect(),
        rho: pr.rho.iter().map(|&l| t(l)).collect(),
        serial_rho: pr.serial_rho.map(t),
        theta,
        ruv_mult,
    })
}

/// [`ng_tape_inputs`] as a kernel plan's inputs, padded with constants to the stacked `width`
/// (the term reads the leading BSV block).
fn ng_leaves(
    width: usize,
    j: &NgJets,
    inp: &mut kernel::Inputs,
) -> crate::sens::focei_kernel::NonGauss<kernel::Leaf> {
    use kernel::Leaf;
    let n = j.g.len();
    let q = inp.leaf(j.q);
    let g: Vec<Leaf> = (0..width)
        .map(|k| {
            if k < n {
                inp.leaf(j.g[k])
            } else {
                Leaf::c(0.0)
            }
        })
        .collect();
    let mut h = vec![vec![Leaf::c(0.0); width]; width];
    for k in 0..n {
        for l in 0..=k {
            let x = inp.leaf(j.h[k][l]);
            h[k][l] = x;
            h[l][k] = x;
        }
    }
    crate::sens::focei_kernel::NonGauss { q, g, h }
}

/// [`ng_leaves`] on the recording.
fn ng_tr(
    n: &crate::sens::focei_kernel::NonGauss<kernel::Leaf>,
    lv: &[kj_tr::Tr],
) -> crate::sens::focei_kernel::NonGauss<kj_tr::Tr> {
    crate::sens::focei_kernel::NonGauss {
        q: n.q.tr(lv),
        g: n.g.iter().map(|l| l.tr(lv)).collect(),
        h: n.h
            .iter()
            .map(|r| r.iter().map(|l| l.tr(lv)).collect())
            .collect(),
    }
}

/// The packed gradient `[θ, Ω, σ, Ω_IOV, ρ]` from `∂Ψ/∂θ` through the predictions
/// (`theta_adj`, per θ, then per σ for an SDE subject's variance rows) and the kernel inputs' adjoints (`psi_of`), through each block's
/// packing chain.
#[allow(clippy::too_many_arguments)]
fn packed_gradient(
    model: &CompiledModel,
    template: &ModelParameters,
    params: &ModelParameters,
    len: usize,
    theta_adj: &[f64],
    psi_of: &dyn Fn(kernel::Leaf) -> f64,
    pr: &ParT<kernel::Leaf>,
    om: &[kernel::Leaf],
    so: &StackedOmega,
    pv: &ParT<f64>,
) -> Vec<f64> {
    let theta = &params.theta;
    let seg = crate::estimation::parameterization::packed_segments(template);
    let chain =
        |m: usize| crate::estimation::sens_outer_gradient::theta_dx_chain(template, theta, m);
    let mut out = vec![0.0; len];
    // θ through the predictions, and directly (FREM rows, serial ρ); then, for an SDE subject,
    // σ through the EKF's variance rows (the adjoints after θ's).
    for (m, &a) in theta_adj.iter().take(theta.len()).enumerate() {
        out[m] += a * chain(m);
    }
    for (k, &a) in theta_adj.iter().skip(theta.len()).enumerate() {
        out[seg.sigma_start() + k] += a * params.sigma.values[k];
    }
    for (m, &leaf) in pr.theta.iter().enumerate() {
        if leaf.i.is_some() {
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
        out[seg.sigma_start() + k] += psi_of(leaf) * params.sigma.values[k];
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
/// its compiled plan.
pub(super) struct NgJets {
    q: f64,
    g: Vec<f64>,
    h: Vec<Vec<f64>>,
    t: NgThird,
}

/// The third order of [`NgJets`], which every consumer only contracts
/// ([`NgJets::contract_t`]): the full tensor, or the term's plan with its kept second-order
/// forward, whose reverse seeded on the `∇²q` lanes is the contraction (#2786).
pub(super) enum NgThird {
    Full(Vec<Vec<Vec<f64>>>),
    #[cfg(feature = "survival")]
    Planned(
        std::sync::Arc<crate::parser::model_parser::rhs_regcode::kernel_jit::Plan>,
        crate::parser::model_parser::rhs_regcode::kernel_jit::Fwd,
    ),
}

impl NgJets {
    /// `Σ_{k≥l} a_h[k][l]·∂³q/∂η_k∂η_l∂η_m` for every `m`.
    fn contract_t(&self, a_h: &[Vec<f64>]) -> Option<Vec<f64>> {
        let n = self.g.len();
        match &self.t {
            NgThird::Full(t) => Some(
                (0..n)
                    .map(|m| {
                        let mut acc = 0.0;
                        for k in 0..n {
                            for l in 0..=k {
                                acc += a_h[k][l] * t[k][l][m];
                            }
                        }
                        acc
                    })
                    .collect(),
            ),
            #[cfg(feature = "survival")]
            NgThird::Planned(plan, fwd) => terms::contract_hessian_lanes(plan, fwd, n, a_h),
        }
    }
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
    // The subject's compiled term (#2786): a forward and one multi-seed reverse.
    terms::non_gaussian_jets3_planned(model, subject, theta, eta)
}

/// The adjoints `(a_q, a_g, a_h)` of [`ng_tape_inputs`]' inputs, `a_h[k][l]` for `k ≥ l`.
fn ng_adjoints<H: Copy>(
    ngr: &crate::sens::focei_kernel::NonGauss<H>,
    n: usize,
    adj: impl Fn(H) -> f64,
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
fn ng_eta_contraction<H: Copy>(
    j: &NgJets,
    ngr: &crate::sens::focei_kernel::NonGauss<H>,
    adj: impl Fn(H) -> f64,
) -> Option<Vec<f64>> {
    let n = j.g.len();
    let (a_q, a_g, a_h) = ng_adjoints(ngr, n, adj);
    let third = if a_h.iter().flatten().any(|&a| a != 0.0) {
        j.contract_t(&a_h)?
    } else {
        vec![0.0; n]
    };
    Some(
        (0..n)
            .map(|m| {
                let mut acc = a_q * j.g[m];
                for k in 0..n {
                    acc += a_g[k] * j.h[k][m];
                }
                acc + third[m]
            })
            .collect(),
    )
}

/// One contribution to [`ng_theta_adjoint`]: the term at `eta` (the BSV η), weighted by the
/// adjoints of its value (`a_q`), η-gradient (`a_g`) and symmetric η-Hessian (`a_h[k][l]`,
/// read for `k ≥ l`). Empty `a_g`/`a_h` weight the value alone.
// Without `survival` the stub ignores its parts: the seeds are built, never read.
#[cfg_attr(not(feature = "survival"), allow(dead_code))]
struct NgSeed<'a> {
    eta: &'a [f64],
    a_q: f64,
    a_g: &'a [f64],
    a_h: &'a [Vec<f64>],
}

/// `Σ_parts ∂/∂θ [a_q·q + Σ_k a_g,k·∂_k q + Σ_{k≥l} a_h,kl·∂²_kl q]` at each part's `eta`: the
/// non-Gaussian term's share of an adjoint's θ block, from the term's plan's reverses (#2786).
/// `None` where the term does not record, or on a non-finite result.
#[cfg(feature = "survival")]
fn ng_theta_adjoint(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    parts: &[NgSeed<'_>],
) -> Option<Vec<f64>> {
    terms::non_gaussian_theta_adjoint_planned(model, subject, theta, parts)
}

/// [`ng_theta_adjoint`] for several seeds at one `eta`: the term's θ-block contraction for
/// each `(a_q, a_g, a_h)`, one forward and one multi-seed reverse of the term's plan.
#[cfg(feature = "survival")]
#[allow(clippy::type_complexity)]
fn ng_theta_adjoint_seeds(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    seeds: &[(f64, Vec<f64>, Vec<Vec<f64>>)],
) -> Option<Vec<Vec<f64>>> {
    let by_ref: Vec<(f64, &[f64], &[Vec<f64>])> = seeds
        .iter()
        .map(|(a_q, a_g, a_h)| (*a_q, a_g.as_slice(), a_h.as_slice()))
        .collect();
    terms::non_gaussian_theta_seeds_planned(model, subject, theta, eta, &by_ref)
}

#[cfg(not(feature = "survival"))]
#[allow(clippy::type_complexity)]
fn ng_theta_adjoint_seeds(
    _: &CompiledModel,
    _: &Subject,
    _: &[f64],
    _: &[f64],
    _: &[(f64, Vec<f64>, Vec<Vec<f64>>)],
) -> Option<Vec<Vec<f64>>> {
    None
}

#[cfg(not(feature = "survival"))]
fn ng_theta_adjoint(
    _: &CompiledModel,
    _: &Subject,
    _: &[f64],
    _: &[NgSeed<'_>],
) -> Option<Vec<f64>> {
    None
}

/// A subject's non-Gaussian data term `q(θ, η)` (TTE, binary and CTMM summed, 1×) and its
/// exact θ-gradient at fixed `eta` (the BSV η): one scalar-tape recording with θ as inputs and
/// one reverse sweep. `(0, 0)` for a subject with no non-Gaussian records; `None` when a record
/// has no exact channel or on a non-finite result. Overwrites the scalar tape.
#[cfg(feature = "survival")]
pub(crate) fn non_gaussian_theta_score(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>)> {
    if subject.obs_records.is_empty() {
        return Some((0.0, vec![0.0; theta.len()]));
    }
    let (q, out) = terms::non_gaussian_theta_score_planned(model, subject, theta, eta)?;
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

#[allow(clippy::too_many_arguments)]
fn run(
    model: &CompiledModel,
    pop_model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    template: &ModelParameters,
    params: &ModelParameters,
    x: &[f64],
    eta: &[f64],
    interaction: bool,
    eta_dx: Option<&mut ModeOut>,
) -> Option<Vec<f64>> {
    let theta = &params.theta;
    let n = eta.len();

    // ---- 1. forward: the subject's plan at the first-order layout ---------------------
    // Its forward gives the predictions with their η-tangents; its reverse (step 2) gives
    // every η- and θ-derivative through them, the η-Hessian of `f` only ever contracted
    // with an adjoint of `G`, never formed. An endpoint-only subject has no prediction rows.
    let n_obs = subject.observations.len();
    let fl = terms::layout1(n);
    let planned = if n_obs == 0 {
        None
    } else {
        Some(plan_forward(model, subject, lay, theta, eta, fl)?)
    };
    // An SDE subject's EKF variance rows, a plan of their own (θ then σ): the kernel's rows
    // `n_obs..2·n_obs`, whose adjoints step 4 reverses to θ and σ.
    let planned_p = if n_obs > 0 && model.is_sde() {
        Some(plan_forward_in(
            model,
            subject,
            lay,
            theta,
            Some(&params.sigma.values),
            eta,
            fl,
        )?)
    } else {
        None
    };
    let preds = PredLanes::new(
        planned
            .iter()
            .chain(&planned_p)
            .flat_map(|p| p.1.iter().copied())
            .collect::<Vec<f64>>(),
        n,
        false,
    );
    let n_rows = preds.rows();
    // A subject's non-Gaussian records (TTE, binary, CTMM): the term and its η-derivatives
    // to third order, for every step below. It reads BSV η only, so under IOV it lives on the
    // stacked system's leading block (the jets pad past it).
    let n_bsv = model.n_eta;
    let ng = if subject.obs_records.is_empty() {
        None
    } else {
        Some(ng_jets3(model, subject, theta, &eta[..n_bsv])?)
    };
    // FOCE scores R⁰ at the population prediction f(η = 0): the population model's plan
    // there, values only, its forward kept for step 4's reverse. Beside a non-Gaussian term
    // production's objective takes the conditional variance, but the EBE is still the
    // frozen-variance inner objective's stationary point (#1722), so `∇_η l` reads this
    // point either way.
    let planned_pop = if n_obs > 0 && !interaction && model.error_spec.has_f_dependent_variance() {
        let zeros = vec![0.0; n];
        let f64l = crate::sens::num::vm::Layout::F64;
        Some(plan_forward(pop_model, subject, lay, theta, &zeros, f64l)?)
    } else {
        None
    };

    // ---- 2. F and ∇_η l on the scalar tape, one (1 + N)-lane sweep ---------------------
    // Every derivative below is a combination of this sweep's input adjoints: `∇_η F` and
    // `H = ∂(∇_η l)/∂η` through the predictions (one multi-seed reverse of their plans,
    // seeded with each output's adjoints of `f` and `G`; f_pop does not depend on η), and
    // Ψ = F − wᵀ∇_η l and the mode-response rows as fixed linear combinations of the lanes.
    let ctx = LikCtx::new(model, subject, params);
    let so = stacked_omega(model, subject, template, params, n);
    let mut pv = ParT::of(model, subject, params);
    pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(so.chol.clone())?;
    let mut inp = kernel::Inputs::default();
    let fr: Vec<kernel::Leaf> = (0..n_rows).map(|j| inp.leaf(preds.v(j))).collect();
    let gr: Vec<Vec<kernel::Leaf>> = (0..n_rows)
        .map(|j| (0..n).map(|k| inp.leaf(preds.g(j, k))).collect())
        .collect();
    let pr0: Option<Vec<kernel::Leaf>> = planned_pop
        .as_ref()
        .map(|(_, v, _)| v.iter().map(|&x| inp.leaf(x)).collect());
    let er: Vec<kernel::Leaf> = eta.iter().map(|&v| inp.leaf(v)).collect();
    let (pr, om) = param_leaves(&pv, &so, &mut inp);
    // The non-Gaussian term's value, η-gradient and (symmetric, one input per k ≥ l)
    // η-Hessian as kernel inputs; their θ-derivatives are contracted with the adjoints in
    // step 4.
    let ng_r = ng.as_ref().map(|j| ng_leaves(n, j, &mut inp));
    // The kernel's code: everything but its leaves that shapes it.
    let variant = kernel::variant_of(&(
        0u8,
        interaction,
        n,
        pr0.is_some(),
        ng.as_ref().map(|j| j.g.len()),
        &so.placed.iter().map(|p| p.1.clone()).collect::<Vec<_>>(),
    ));
    let kr = kernel::kernel_run(
        model,
        subject,
        variant,
        crate::sens::num::vm::Layout::F64,
        &inp.vals,
        1 + n,
        &|lv| {
            let t = |l: kernel::Leaf| l.tr(lv);
            let f: Vec<kj_tr::Tr> = fr.iter().map(|&l| t(l)).collect();
            let g: Vec<Vec<kj_tr::Tr>> = gr
                .iter()
                .map(|r| r.iter().map(|&l| t(l)).collect())
                .collect();
            let p0: Option<Vec<kj_tr::Tr>> =
                pr0.as_ref().map(|v| v.iter().map(|&l| t(l)).collect());
            let e: Vec<kj_tr::Tr> = er.iter().map(|&l| t(l)).collect();
            let p = param_tr(model, subject, &pr, lv)?;
            let (fv, lg) = match ng_r.as_ref() {
                Some(n) => objective_ng(
                    &ctx,
                    interaction,
                    &f,
                    &g,
                    &e,
                    p0.as_deref(),
                    &ng_tr(n, lv),
                    &p,
                )?,
                None => objective(&ctx, interaction, &f, &g, &e, p0.as_deref(), &p)?,
            };
            Some(std::iter::once(fv).chain(lg).collect())
        },
    )?;
    // Every output's adjoint at every leaf: one reverse seeded with each output.
    let unit = |o: usize| -> Vec<f64> { (0..=n).map(|c| if c == o { 1.0 } else { 0.0 }).collect() };
    let base = kr.reverse_values(&(0..=n).map(unit).collect::<Vec<_>>())?;
    let ba = |o: usize, x: kernel::Leaf| x.adj(&base[o]);
    // Every lane's adjoints of the predictions through their plans, one reverse each seeded
    // with all `1 + N` lanes: per lane its η-derivative (value lanes of η's slots) and its θ
    // (then an SDE's σ) derivative, which step 4 combines per output.
    let n_sigma = if planned_p.is_some() {
        params.sigma.values.len()
    } else {
        0
    };
    let n_lanes = 1 + n;
    let mut pred_eta = vec![vec![0.0; n]; n_lanes];
    let mut pred_theta = vec![vec![0.0; theta.len() + n_sigma]; n_lanes];
    for (row0, p) in [(0, planned.as_ref()), (n_obs, planned_p.as_ref())] {
        let Some((plan, _, fwd)) = p else { continue };
        let l = fl.lanes();
        let w = n_obs * l;
        let mut du_bar = vec![0.0; n_lanes * w];
        for o in 0..n_lanes {
            for j in 0..n_obs {
                du_bar[o * w + j * l] = ba(o, fr[row0 + j]);
                for k in 0..n {
                    du_bar[o * w + j * l + 1 + k] = ba(o, gr[row0 + j][k]);
                }
            }
        }
        let np = plan.n_theta();
        let mut u_bar = vec![0.0; n_lanes * n * l];
        let mut p_bar = vec![0.0; n_lanes * np * l];
        if !plan.reverse_from(fwd, fl, n_lanes, &mut u_bar, &mut p_bar, &du_bar) {
            return None;
        }
        for o in 0..n_lanes {
            for (k, d) in pred_eta[o].iter_mut().enumerate() {
                *d += u_bar[(o * n + k) * l];
            }
            for (j, t) in pred_theta[o].iter_mut().take(np).enumerate() {
                *t += p_bar[(o * np + j) * l];
            }
        }
    }
    // Lane `o`'s total η-gradient (0: F, 1 + k: (∇_η l)_k).
    let d_eta = |o: usize| -> Option<Vec<f64>> {
        let mut d: Vec<f64> = er.iter().map(|&l| ba(o, l)).collect();
        for (dm, p) in d.iter_mut().zip(&pred_eta[o]) {
            *dm += p;
        }
        if let (Some(j), Some(ngr)) = (ng.as_ref(), ng_r.as_ref()) {
            for (dm, c) in d.iter_mut().zip(ng_eta_contraction(j, ngr, |x| ba(o, x))?) {
                *dm += c;
            }
        }
        Some(d)
    };
    let grad_f = DVector::from_column_slice(&d_eta(0)?);
    let rows: Vec<Vec<f64>> = (0..n).map(|k| d_eta(1 + k)).collect::<Option<_>>()?;
    let h = DMatrix::from_fn(n, n, |k, m| 0.5 * (rows[k][m] + rows[m][k]));
    let grad_l: Vec<f64> = (0..n).map(|k| kr.value(1 + k)).collect();
    let fz = if interaction {
        None
    } else {
        planned_pop.as_ref().map(|(_, v, _)| v.clone())
    };
    let Some(w) = mode_solve(&h, &grad_f, &grad_l, || {
        let mut g =
            inner_search_gradient(model, subject, lay, theta, eta, &ctx, &pv, fz.as_deref())?;
        if let Some(j) = ng.as_ref() {
            for (gk, qk) in g.iter_mut().zip(&j.g) {
                *gk += qk;
            }
        }
        Some(g)
    }) else {
        if let Some(o) = eta_dx {
            o.mode_system_declined = true;
        }
        return None;
    };

    // ---- 3. Ψ = F − wᵀ∇_η l and (with the mode response) its rows, from the lanes -----
    // Row `r` is `y_r = Σ_k Z_rk (∇_η l)_k`, `Z` the BSV rows of `H⁻¹`, so
    // `dη̂_r/dx = −∂y_r/∂x` (`Z` held constant: `H` is the derivative of `∇_η l` at the
    // mode): one lane per BSV effect, however wide the stacked `[η, κ]` system. Ψ in lane 0.
    let mut coefs: Vec<Vec<f64>> = vec![std::iter::once(1.0).chain(w.iter().map(|x| -x)).collect()];
    if eta_dx.as_ref().is_some_and(|o| o.want_dx) {
        coefs.extend(eta_dx_coefs(&h, n_bsv));
    }
    let n_outs = coefs.len();
    // Each output's adjoint row: its combination of the kernel outputs' rows.
    // Combined per leaf, at the leaves that are read: most of a recording's inputs are not.
    let row_at =
        |o: usize, i: usize| -> f64 { coefs[o].iter().zip(&base).map(|(a, r)| a * r[i]).sum() };
    #[cfg(test)]
    if n_outs > 1 && FAIL_RESPONSE_LANES.with(|f| f.get()) {
        return None;
    }
    let out_adj = |o: usize, x: kernel::Leaf| -> f64 { x.adj_with(|i| row_at(o, i)) };
    let psi_of = |x: kernel::Leaf| out_adj(0, x);

    // ---- 4. the outputs' θ-derivatives: through the predictions, step 2's per-lane ------
    // derivatives combined; FOCE's population prediction and the non-Gaussian term below.
    let mut lane_theta: Vec<Vec<f64>> = coefs
        .iter()
        .map(|c| {
            (0..theta.len() + n_sigma)
                .map(|i| c.iter().zip(&pred_theta).map(|(a, t)| a * t[i]).sum())
                .collect()
        })
        .collect();
    // FOCE's population prediction: its adjoints through its plan at η = 0, values only.
    if let (Some((pp, _, pf)), Some(p0)) = (planned_pop.as_ref(), pr0.as_ref()) {
        let f64l = crate::sens::num::vm::Layout::F64;
        let no = n_outs;
        let mut du_bar = vec![0.0; no * n_obs];
        for o in 0..no {
            for (j, &x) in p0.iter().enumerate() {
                du_bar[o * n_obs + j] = out_adj(o, x);
            }
        }
        let nt = theta.len();
        let mut u_bar = vec![0.0; no * n];
        let mut p_bar = vec![0.0; no * nt];
        if !pp.reverse_from(pf, f64l, no, &mut u_bar, &mut p_bar, &du_bar) {
            return None;
        }
        for (o, ta) in lane_theta.iter_mut().enumerate() {
            for (t, &d) in ta.iter_mut().zip(&p_bar[o * nt..(o + 1) * nt]) {
                *t += d;
            }
        }
    }
    // The non-Gaussian term's θ block for every output: one recording of the term at η̂,
    // swept per seed.
    if let Some(ngr) = ng_r.as_ref() {
        let n = n_bsv;
        let seeds: Vec<_> = (0..n_outs)
            .map(|o| ng_adjoints(ngr, n, |x: kernel::Leaf| out_adj(o, x)))
            .collect();
        let d = if seeds.len() == 1 {
            let (a_q, a_g, a_h) = &seeds[0];
            let part = NgSeed {
                eta: &eta[..n],
                a_q: *a_q,
                a_g,
                a_h,
            };
            vec![ng_theta_adjoint(model, subject, theta, &[part])?]
        } else {
            ng_theta_adjoint_seeds(model, subject, theta, &eta[..n], &seeds)?
        };
        for (ta, d) in lane_theta.iter_mut().zip(d) {
            for (t, dv) in ta.iter_mut().zip(d) {
                *t += dv;
            }
        }
    }
    let out = packed_gradient(
        model,
        template,
        params,
        x.len(),
        &lane_theta[0],
        &psi_of,
        &pr,
        &om,
        &so,
        &pv,
    );
    if n_outs > 1 {
        // ---- 5. the mode response from the same sweeps ------------------------------------
        let n_rows = n_outs - 1;
        let mut cols = vec![DVector::<f64>::zeros(n_rows); x.len()];
        for r in 0..n_rows {
            let of = |x: kernel::Leaf| out_adj(r + 1, x);
            let row = packed_gradient(
                model,
                template,
                params,
                x.len(),
                &lane_theta[r + 1],
                &of,
                &pr,
                &om,
                &so,
                &pv,
            );
            for (c, v) in cols.iter_mut().zip(row) {
                c[r] = -v;
            }
        }
        let ok = cols.iter().all(|c| c.iter().all(|v| v.is_finite()));
        if let Some(o) = eta_dx {
            o.eta_dx = ok.then_some(cols);
        }
    }
    Some(out)
}

#[cfg(test)]
thread_local! {
    /// Test switch: sweep the quadrature's Ψ on its own, as before the fused read.
    pub(crate) static PSI_SWEEP_UNFUSED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Quadrature gradients that read Ψ's θ adjoint off A's and the response rows'.
    pub(crate) static PSI_FUSED_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// What a gradient pass can also hand back about the subject's mode, from its own recording
/// (#2549): the exact mode response `dη̂/dx` (the EBE warm-start predictor's Jacobian) and,
/// under the quadrature, the exact inner Hessian `∇²_η l` at the mode, which the quadrature
/// objective's anchor and the next warm inner solve's seed would otherwise each recompute.
#[derive(Debug, Default)]
pub(crate) struct ModeOut {
    /// Form the mode response.
    pub(crate) want_dx: bool,
    /// Hand back the exact inner Hessian (quadrature objectives only).
    pub(crate) want_hessian: bool,
    pub(crate) eta_dx: Option<Vec<DVector<f64>>>,
    pub(crate) hessian: Option<DMatrix<f64>>,
    /// The FOCE/FOCEI pass declined at its mode system `H w = ∇_η F` ([`mode_solve`]): `H`
    /// singular at `η̂`, or indefinite at an `η̂` that is not stationary. Any other decline
    /// (scope, a failed or non-finite recording) leaves it `false` (#2713). Reset at the
    /// start of every pass; [`Self::clear`] keeps it.
    pub(crate) mode_system_declined: bool,
}

impl ModeOut {
    /// Drop the products, keeping what is wanted.
    pub(crate) fn clear(&mut self) {
        self.eta_dx = None;
        self.hessian = None;
    }
}

#[cfg(test)]
thread_local! {
    /// Test hook: fail [`run`] whenever it carries mode-response lanes, as a non-finite row
    /// adjoint would, to pin that the gradient survives a failed response.
    pub(crate) static FAIL_RESPONSE_LANES: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// The mode response's rows (step 3 of [`run`]): `y_r = Σ_k Z_rk g_k` for `r < n_rows`,
/// `Z = (H⁻¹)_{r,:}`, as coefficients over the lanes `[F, g_0, …, g_{n−1}]`. Empty when `H`
/// is singular.
fn eta_dx_coefs(h: &DMatrix<f64>, n_rows: usize) -> Vec<Vec<f64>> {
    let n_act = h.nrows();
    let rhs = DMatrix::<f64>::identity(n_act, n_rows);
    let z = match h.clone().cholesky() {
        Some(c) => Some(c.solve(&rhs)),
        None => h.clone().lu().solve(&rhs),
    };
    // `H` symmetric: column `r` of `H⁻¹E` is row `r` of `H⁻¹`.
    let Some(z) = z.filter(|z| z.iter().all(|v| v.is_finite())) else {
        return Vec::new();
    };
    // Over [`run`]'s base lanes `[F, (∇_η l)_0, …]`.
    (0..n_rows)
        .map(|r| {
            let mut c = vec![0.0; 1 + n_act];
            for k in 0..n_act {
                c[1 + k] = z[(k, r)];
            }
            c
        })
        .collect()
}

/// The anchor's Cholesky factor, by production's rule (`agq::agq_proposal`, #2538): `H_a`
/// itself when it is positive-definite, so one node is the Laplace approximation
/// `l(η̂) + ½log|H_a|` exactly; else `H_a` with the jitter `H_ii += max(10⁻⁶|H_ii|, 10⁻¹⁰)`
/// (`build_proposal`); else the prior precision `Ω⁻¹`. Branches on values only, so the
/// tape differentiates whichever objective production evaluated.
fn factor_anchor<T: crate::sens::num::PkNum>(
    h: &mut [Vec<T>],
    omega_inv: &[Vec<T>],
) -> Option<Vec<Vec<T>>> {
    if let Some(l) = crate::sens::focei_kernel::chol(h) {
        return Some(l);
    }
    for (i, row) in h.iter_mut().enumerate() {
        let d = row[i];
        row[i] = if (T::from_f64(1e-6) * d.abs()).br_cmp_c(crate::sens::num::BranchCmp::Gt, 1e-10) {
            d + T::from_f64(1e-6) * d.abs()
        } else {
            d + T::from_f64(1e-10)
        };
    }
    crate::sens::focei_kernel::chol(h).or_else(|| crate::sens::focei_kernel::chol(omega_inv))
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
fn run_agq(
    model: &CompiledModel,
    subject: &Subject,
    lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    template: &ModelParameters,
    params: &ModelParameters,
    x: &[f64],
    eta: &[f64],
    n_agq: usize,
    exact: bool,
    held: bool,
    eta_dx: Option<&mut ModeOut>,
) -> Option<Vec<f64>> {
    let theta = &params.theta;
    let ctx = LikCtx::new(model, subject, params);
    let so = stacked_omega(model, subject, template, params, eta.len());
    let mut pv = ParT::of(model, subject, params);
    pv.omega = crate::sens::focei_kernel::OmegaT::from_chol(so.chol.clone())?;
    let n = eta.len();

    // ---- 1. values at η̂ (the plan's second-order forward): f, G, Q; the exact H and the
    // anchor. An endpoint-only subject has no Gaussian rows to record.
    let n_obs = subject.observations.len();
    // Only the exact anchor (which differentiates through `H`) reads `Q`; the Gauss-Newton
    // anchor runs first order and `agq_planned` contracts the exact inner Hessian by reverse.
    let lay_mode = if exact {
        terms::layout2(n)
    } else {
        terms::layout1(n)
    };
    let p1 = if n_obs == 0 {
        None
    } else {
        Some(plan_forward(model, subject, lay, theta, eta, lay_mode)?)
    };
    // An SDE subject's EKF variance rows, the kernel's rows `n..2n`, a plan of their own.
    let p_var = if n_obs > 0 && model.is_sde() {
        Some(plan_forward_in(
            model,
            subject,
            lay,
            theta,
            Some(&params.sigma.values),
            eta,
            lay_mode,
        )?)
    } else {
        None
    };
    let lanes_all: Vec<f64> = p1
        .iter()
        .chain(&p_var)
        .flat_map(|p| p.1.iter().copied())
        .collect();
    let pl = PredLanes::new(lanes_all.as_slice(), n, exact);
    let n_rows = pl.rows();
    let f1: Vec<f64> = (0..n_rows).map(|j| pl.v(j)).collect();
    let g1: Vec<Vec<f64>> = (0..n_rows).map(|j| pl.row_g(j).to_vec()).collect();
    // A subject's non-Gaussian records (TTE, binary, CTMM): the term's η-jets to third order
    // at η̂, over the BSV block (the leading one of the stacked `[η, κ]` under IOV). Both
    // anchors carry its `∇²q`, as production's do (`exact_inner_hessian`, `exact_htilde`).
    let n_bsv = model.n_eta;
    let ng = if subject.obs_records.is_empty() {
        None
    } else {
        Some(ng_jets3(model, subject, theta, &eta[..n_bsv])?)
    };
    let h_exact = if exact {
        let q1: Vec<Vec<f64>> = (0..n_rows).map(|j| pl.row_q(j).to_vec()).collect();
        let h =
            crate::sens::focei_kernel::exact_inner_hessian_kernel(&ctx, &f1, &g1, &q1, eta, &pv)?;
        let mut h = DMatrix::from_fn(n, n, |a, b| h[a][b]);
        if let Some(j) = ng.as_ref() {
            for a in 0..n_bsv {
                for b in 0..n_bsv {
                    h[(a, b)] += j.h[a][b];
                }
            }
        }
        Some(h)
    } else {
        None
    };
    let mut h_a: Vec<Vec<f64>> = if let Some(h_exact) = h_exact.as_ref() {
        (0..n)
            .map(|a| (0..n).map(|b| h_exact[(a, b)]).collect())
            .collect()
    } else {
        let mut h = crate::sens::focei_kernel::focei_htilde(&ctx, &f1, &g1, eta, &pv)?;
        if let Some(j) = ng.as_ref() {
            for a in 0..n_bsv {
                for b in 0..n_bsv {
                    h[a][b] += j.h[a][b];
                }
            }
        }
        h
    };
    // Production's fallbacks when the anchor is not positive-definite (a mode the inner
    // search left loosely converged, a flat direction): the jittered anchor, then the prior
    // precision `Ω⁻¹`. Each is part of the objective, so the gradient follows it rather
    // than declining.
    let lh = factor_anchor(&mut h_a, &pv.omega.inv)?;
    // The tensor grid: per node its `z`, `c_j = Σ log w + ‖z‖²`, and `b_j` (f64).
    let one_point = n_agq == 1;
    let grid: Vec<(Vec<f64>, f64, Vec<f64>)> = if one_point {
        Vec::new()
    } else {
        agq_grid(n_agq, n)
            .into_iter()
            .map(|(z, c)| {
                // b = η̂ + √2·L⁻ᵀz
                let mut step = vec![0.0; n];
                for i in (0..n).rev() {
                    let mut acc = z[i];
                    for j in (i + 1)..n {
                        acc -= lh[j][i] * step[j];
                    }
                    step[i] = acc / lh[i][i];
                }
                let b = (0..n)
                    .map(|k| eta[k] + std::f64::consts::SQRT_2 * step[k])
                    .collect();
                (z, c, b)
            })
            .collect()
    };
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
    agq::agq_planned(
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
        &grid,
        h_exact.as_ref(),
        exact,
        held,
        ng.as_ref(),
        eta_dx,
        p1.as_ref(),
        p_var.as_ref(),
        &pl,
    )
}

/// The tensor Gauss–Hermite grid over `n` dimensions: per node its `z` and
/// `c = Σ log w + ‖z‖²`.
fn agq_grid(n_agq: usize, n: usize) -> Vec<(Vec<f64>, f64)> {
    let (nodes, _, logw) = crate::estimation::agq::cached_gauss_hermite(n_agq);
    let mut out = Vec::new();
    let mut idx = vec![0usize; n];
    loop {
        let z: Vec<f64> = idx.iter().map(|&i| nodes[i]).collect();
        out.push((z.clone(), (0..n).map(|k| logw[idx[k]] + z[k] * z[k]).sum()));
        let mut k = 0;
        while k < n {
            idx[k] += 1;
            if idx[k] < n_agq {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
        if k == n {
            return out;
        }
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
        terms::non_gaussian_gradient_planned(model, subject, theta, eta)
    }
    #[cfg(not(feature = "survival"))]
    {
        let _ = (model, subject, theta, eta);
        None
    }
}

#[path = "ode_adjoint_hessian.rs"]
mod hessian;
pub(crate) use hessian::{
    cov_non_gaussian_served, subject_adjoint_cov_block_at, subject_adjoint_cov_diag_at,
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
