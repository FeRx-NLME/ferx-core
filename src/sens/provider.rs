//! Per-observation analytic sensitivities for analytical 1-/2-/3-cpt PK models.
//!
//! Given a [`CompiledModel`], a [`Subject`], population `theta` and a candidate
//! `eta`, [`subject_sensitivities`] returns, for every observation, the exact
//!
//!   * value `f`,
//!   * `∂f/∂η`, `∂²f/∂η²`,
//!   * `∂f/∂θ`, `∂²f/∂η∂θ`
//!
//! built by seeding the PK parameters as [`Dual2`](super::dual2::Dual2) variables
//! through the closed-form PK solution (exact `∂f/∂pk`, `∂²f/∂pk²`) and then
//! applying the **closed-form η/θ chain rule**: the model exposes the relation
//! `pk_i = tv_i·exp(Σ_k sel[i,k]·η_k)`, so
//!
//!   * `∂pk_i/∂η_k        = pk_i · sel[i,k]`,
//!   * `∂²pk_i/∂η_k∂η_l   = pk_i · sel[i,k] · sel[i,l]`,
//!   * `∂pk_i/∂θ_m        = pk_i · ρ[i,m]`,        ρ[i,m] = (∂tv_i/∂θ_m)/tv_i,
//!   * `∂²pk_i/∂η_k∂θ_m   = pk_i · sel[i,k] · ρ[i,m]`,
//!
//! with `∂tv_i/∂θ_m` from a finite difference of the runtime `tv_fn` closure
//! (irreducible — `tv_fn` is user-assembled at parse time). `sel[i,k]`,
//! `pk_indices`, and `tv_fn` all already live on `CompiledModel`.
//!
//! Scope (issue #367): analytical 1-/2-/3-cpt (IV bolus/infusion + oral, incl.
//! steady state — SS infusion for any `T_inf`, including overlapping `T_inf > II`),
//! single endpoint, log-normal η, optional scalar output scaling / LTBS, and dose
//! lagtime (seeded as an extra dual axis through the elapsed-time argument). No
//! IOV, no time-varying covariates, no resets mixed with steady state.
//! [`analytical_supported`] (+ per-subject gates in [`subject_sensitivities`])
//! gate exactly that; everything else returns `None` so the caller falls back
//! to the gradient-free path.
#![allow(clippy::needless_range_loop)]

use super::dual1::Dual1;
use super::dual2::Dual2;
use super::num::PkNum;
use super::one_cpt::one_cpt_conc_g;
use super::three_cpt::three_cpt_conc_g;
use super::two_cpt::two_cpt_conc_g;
use crate::types::{
    CompiledModel, DoseEvent, GradientMethod, PkModel, ScalingSpec, Subject, PK_IDX_CL, PK_IDX_F,
    PK_IDX_KA, PK_IDX_LAGTIME, PK_IDX_Q, PK_IDX_Q3, PK_IDX_V, PK_IDX_V2, PK_IDX_V3,
};

/// Exact sensitivities of one observation w.r.t. η and θ. Hessian-shaped fields
/// are stored row-major (`[k*n + l]`).
#[derive(Debug, Clone)]
pub struct ObsSens {
    pub f: f64,
    /// `∂f/∂η_k`, length `n_eta`.
    pub df_deta: Vec<f64>,
    /// `∂²f/∂η_k∂η_l`, row-major `n_eta × n_eta`.
    pub d2f_deta2: Vec<f64>,
    /// `∂f/∂θ_m`, length `n_theta`.
    pub df_dtheta: Vec<f64>,
    /// `∂²f/∂η_k∂θ_m`, row-major `n_eta × n_theta`.
    pub d2f_deta_dtheta: Vec<f64>,
}

/// All observations' sensitivities for one subject, parallel to
/// `subject.obs_times`.
#[derive(Debug, Clone)]
pub struct SubjectSens {
    pub obs: Vec<ObsSens>,
}

/// Map a fixed PK slot to its seed dimension. The analytical 1-/2-/3-cpt
/// solutions read `CL, V1, Q2, V2, KA, F, Q3, V3` (slots 0,1,2,3,4,5,6,7) — an
/// identity map; `LAGTIME` (slot 8) is differentiated too, entering each dose's
/// concentration through the elapsed-time argument (`∂elapsed/∂lagtime = −1`).
#[inline]
fn slot_to_dim(slot: usize) -> Option<usize> {
    match slot {
        PK_IDX_CL => Some(0),
        PK_IDX_V => Some(1),
        PK_IDX_Q => Some(2),
        PK_IDX_V2 => Some(3),
        PK_IDX_KA => Some(4),
        PK_IDX_F => Some(5),
        PK_IDX_Q3 => Some(6),
        PK_IDX_V3 => Some(7),
        PK_IDX_LAGTIME => Some(8),
        _ => None,
    }
}

/// Number of seeded dimensions (`CL, V1, Q2, V2, KA, F, Q3, V3, LAGTIME`).
const N_PK: usize = 9;

/// True when the compiled `[individual_parameters]` program emits a differentiable
/// row for every structural PK output `model` requires — the precondition for
/// driving the analytic sensitivity chain (`param_derivatives_from_prog` /
/// `pd_from_program`) over its `pk_slots()`-ordered rows.
///
/// Checks the model's *required* PK slots, NOT `pk_indices.len()`. `pk_indices`
/// is parallel to every `[individual_parameters]` assignment, so it also counts
/// intermediate rows (e.g. `WTREL = WT/70` before `CL = ...`); comparing its
/// length against `pk_slots().len()` (structural outputs only) wrongly rejected
/// any model with intermediate assignments and routed it to a fallback that
/// mis-seeds the structural slots (#455/#456). Every non-literal structural
/// parameter is bound as an individual parameter and therefore present in
/// `pk_slots()`; a literal-constant slot is correctly absent and seeded as a
/// constant downstream.
fn prog_covers_required_pk_slots(
    model: &CompiledModel,
    prog: &crate::parser::model_parser::IndivParamProgram,
) -> bool {
    let prog_slots = prog.pk_slots_ref();
    model
        .pk_model
        .required_pk_params()
        .iter()
        .all(|(slot, _)| prog_slots.contains(slot) && slot_to_dim(*slot).is_some())
}

/// Elapsed time since a dose's *lagged* arrival, as a dual carrying the lagtime
/// sensitivity (`∂elapsed/∂lagtime = −1`). Mirrors [`crate::pk::predict_concentration`]:
/// the dose contributes from `dose.time + lagtime` onward, and a steady-state
/// dose additionally shows its pre-arrival tail (the previous interval's pulse)
/// by wrapping the negative elapsed time up into `[0, II)`. Returns `None` when
/// the dose does not contribute at `t_obs`. `lag_d` is the seeded lagtime dual;
/// `lag_val` its value (kept separate to branch on `.val()` without a borrow).
#[inline]
fn lagged_elapsed<T: PkNum>(dose: &DoseEvent, t_obs: f64, lag_val: f64, lag_d: T) -> Option<T> {
    let t_eff = dose.time + lag_val;
    if t_eff <= t_obs {
        // `elapsed = (t_obs − dose.time) − lagtime`.
        Some(T::from_f64(t_obs - dose.time) - lag_d)
    } else if dose.ss && dose.ii > 0.0 && t_obs >= dose.time {
        // Pre-arrival steady-state tail: the most recent pulse landed at
        // `t_eff − n·II`; wrap the negative elapsed up into `[0, II)`. `n` is
        // locally constant, so `∂elapsed/∂lagtime = −1` still holds.
        let raw = t_obs - t_eff;
        let n = (-raw / dose.ii).ceil();
        let wrapped = T::from_f64(t_obs - dose.time + n * dose.ii) - lag_d;
        if wrapped.val() < 0.0 {
            None
        } else {
            Some(wrapped)
        }
    } else {
        None
    }
}

/// Master switch for the user-ODE sensitivity path (issue #367, Option A). The
/// provider in [`crate::sens::ode_provider`] supplies the analytic `∂f/∂θ` and
/// `∂f/∂η` for RHS-program ODE models via an augmented `Dual2` RK45, feeding the
/// same FOCE/FOCEI outer-gradient assembly the analytical PK models use.
///
/// Armed in **#410** after hardening the observation/break-time matching
/// (`obs_time_matches`, tolerance instead of bit-exact keying). The scope is
/// limited by [`crate::sens::ode_provider::ode_analytical_supported`] (RHS-program
/// models, simple readout, bolus + finite infusion, `F`, resets, covariates,
/// within the dual-axis cap); anything outside it falls back to the prior path
/// (gradient-free outer, FD inner). Both routes are armed: the **outer** η/θ
/// gradient via the `Dual2` walk, and the **inner** EBE η-gradient via the light
/// `Dual1` walk (`subject_eta_grad_impl` → `ode_subject_eta_grad`, gated by
/// `ode_inner_grad_supported`); out-of-scope subjects still fall back to FD.
const ODE_SENS_ENABLED: bool = true;

/// Escape hatch: `FERX_DISABLE_EXPLICIT_SENS=1` forces every subject onto the
/// generic `Dual2<N>` provider path instead of the hand-written explicit-
/// derivative kernels (`*_explicit`). The explicit kernels are the default — they
/// compute the same `(f, ∂f/∂pk, ∂²f/∂pk²)` to ~1e-8 (validated per-kernel
/// against the dual oracle) at a fraction of the cost. This toggle exists to A/B
/// the two on a real fit and as a fallback if a kernel is ever suspected wrong.
fn explicit_sens_disabled() -> bool {
    std::env::var("FERX_DISABLE_EXPLICIT_SENS")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Which explicit-kernel model class serves a subject. Every dose kind — bolus /
/// infusion / oral and their steady-state variants — has a hand-written kernel,
/// so the explicit path covers any in-scope subject; the genuinely unsupported SS
/// edges (overlapping SS infusion, SS mixed with resets) are screened earlier in
/// [`subject_sensitivities`] and never reach the kernels. The per-observation
/// chain is identical to the `Dual2<N>` path — only `(f, ∂f/∂pk, ∂²f/∂pk²)` is
/// sourced differently.
#[derive(Clone, Copy)]
enum ExKind {
    /// 1-cpt IV: bolus + infusion.
    OneCptIv,
    /// 1-cpt oral: first-order absorption + infusion-into-central.
    OneCptOral,
    /// 2-cpt IV: bolus + infusion.
    TwoCptIv,
    /// 2-cpt oral: first-order absorption + infusion-into-central.
    TwoCptOral,
    /// 3-cpt IV: bolus + infusion.
    ThreeCptIv,
    /// 3-cpt oral: first-order absorption + infusion-into-central.
    ThreeCptOral,
}

/// True when [`subject_sensitivities`] can serve this model: any analytical
/// 1-/2-/3-cpt model (IV bolus/infusion + oral), `tv_fn` present, no ODE.
/// Per-subject gates (TV covariates) are checked separately in
/// [`subject_sensitivities`].
pub fn analytical_supported(model: &CompiledModel) -> bool {
    matches!(
        model.pk_model,
        PkModel::OneCptIv
            | PkModel::OneCptOral
            | PkModel::TwoCptIv
            | PkModel::TwoCptOral
            | PkModel::ThreeCptIv
            | PkModel::ThreeCptOral
    ) && model.ode_spec.is_none()
        && model.tv_fn.is_some()
        && model.n_kappa == 0
        && scaling_supported(model)
        && init_supported(model)
        // Every individual-parameter slot must be one we differentiate. A
        // LAGTIME (slot 8) routes to fall back.
        && model.pk_indices.iter().all(|&s| slot_to_dim(s).is_some())
}

/// Whether the model's `[initial_conditions]` baseline (#521) is one the Dual2
/// provider differentiates exactly (#524): every init carries a compiled
/// `amount_deriv` program and the `(θ, η)` axis count fits the init dispatch
/// table (`1..=MAX_SCALE_AXES`, shared with the scale program). An init-free
/// model is trivially supported; a hand-built init with no program, or a model
/// with more axes than the table covers, routes to FD (correct, just slower).
fn init_supported(model: &CompiledModel) -> bool {
    model.analytical_init.is_empty()
        || (model
            .analytical_init
            .iter()
            .all(|i| i.amount_deriv.is_some())
            && (1..=MAX_SCALE_AXES).contains(&(model.n_theta + model.n_eta)))
}

/// Maximum `(θ, η)` axis count for the differentiable `ExpressionScale` program
/// (the `Dual2<M>` dispatch table). Beyond this the scale falls back to FD.
pub(crate) const MAX_SCALE_AXES: usize = 16;

/// Monomorphize an axis-parametrized helper on a runtime axis count, dispatching
/// `$apply::<K>(args…)` over the `1..=MAX_SCALE_AXES` table. Written **once** and
/// shared by the analytic `[initial_conditions]` impulse (inner
/// [`apply_analytical_init_inner`] on `n_eta`, outer [`apply_analytical_init_outer`]
/// on `n_theta + n_eta`) and the inner `ExpressionScale` quotient
/// ([`apply_expression_scale_inner`] on `n_eta`) so none of them can drift to a
/// different axis table. The `_` arm is unreachable for a supported model:
/// `init_supported` / `scaling_supported` bound `n_theta + n_eta` to
/// `1..=MAX_SCALE_AXES`, and each path's axis count is `≤ n_theta + n_eta`.
macro_rules! dispatch_init_impulse {
    ($axes:expr, $apply:ident, $($arg:expr),+ $(,)?) => {
        match $axes {
            1 => $apply::<1>($($arg),+),
            2 => $apply::<2>($($arg),+),
            3 => $apply::<3>($($arg),+),
            4 => $apply::<4>($($arg),+),
            5 => $apply::<5>($($arg),+),
            6 => $apply::<6>($($arg),+),
            7 => $apply::<7>($($arg),+),
            8 => $apply::<8>($($arg),+),
            9 => $apply::<9>($($arg),+),
            10 => $apply::<10>($($arg),+),
            11 => $apply::<11>($($arg),+),
            12 => $apply::<12>($($arg),+),
            13 => $apply::<13>($($arg),+),
            14 => $apply::<14>($($arg),+),
            15 => $apply::<15>($($arg),+),
            16 => $apply::<16>($($arg),+),
            _ => {}
        }
    };
}

// The `dispatch_init_impulse!` table is hand-written for `1..=16`. If the axis cap
// changes, the table (and the scale-program dispatch tables) must change with it —
// fail to compile rather than silently drop the init impulse on the `_` arm.
const _: () = assert!(
    MAX_SCALE_AXES == 16,
    "dispatch_init_impulse! enumerates 1..=16; update it when MAX_SCALE_AXES changes",
);

/// Maximum `(θ, η)` axis count (`n_theta + n_eta`) for the TV-cov event-driven dual
/// walk. The outer `run_obs_tvcov` (`m_dim`) and inner `run_obs_grad_tvcov` (`n_eta
/// ≤ m_dim`) dispatch tables both enumerate `1..=MAX_TVCOV_AXES`, and
/// `tvcov_analytical_supported` bounds the model here, so both resolve and the
/// inner/outer analytic scope stays matched (#449 re-review #2).
const MAX_TVCOV_AXES: usize = 24;

// Five `disp!(1..=24)` dispatch tables key on `MAX_TVCOV_AXES` with a silent `_ => None`:
// `lognormal_param_derivatives`, `subject_sensitivities_iov`,
// `subject_eta_grad_iov_analytical`, `subject_sensitivities_tvcov`, and
// `subject_eta_grad_tvcov`. Keep all five in lockstep with the const — bumping it without
// widening every arm would let an in-scope wider model hit `_ => None` and silently fall
// back to FD. The mirror of the `MAX_ODE_AXES` tripwire in `ode_provider.rs` (#466 review
// round 4 #12).
const _: () = assert!(
    MAX_TVCOV_AXES == 24,
    "MAX_TVCOV_AXES changed: widen the disp!(1..=24) tables in lognormal_param_derivatives, \
     subject_sensitivities_iov, subject_eta_grad_iov_analytical, subject_sensitivities_tvcov, \
     and subject_eta_grad_tvcov to match, then update this assert"
);

/// Whether the model's output scaling is one the provider differentiates exactly:
/// `None` / constant `ScalarScale` (a per-jet divisor, `∂k/∂η = ∂k/∂θ = 0`), or an
/// `ExpressionScale` carrying a `Dual2`-differentiable program whose axis counts
/// match the model and fit the dispatch table. `ExpressionScale` without a program
/// and `PerCmt` route to FD.
fn scaling_supported(model: &CompiledModel) -> bool {
    match &model.scaling {
        ScalingSpec::None | ScalingSpec::ScalarScale(_) => true,
        ScalingSpec::ExpressionScale { deriv: Some(p), .. } => {
            p.n_theta_axis() == model.n_theta
                && p.n_eta_axis() == model.n_eta
                && (1..=MAX_SCALE_AXES).contains(&p.n_axes())
        }
        _ => false,
    }
}

/// True when the exact `sens` outer gradient applies to this model: either the
/// analytical PK provider ([`analytical_supported`]) or the ODE sensitivity
/// provider ([`ode_analytical_supported`](crate::sens::ode_provider::ode_analytical_supported)).
/// Used to gate the gradient dispatch and the Eq. 48 EBE predictor.
pub fn sens_supported(model: &CompiledModel) -> bool {
    analytical_supported(model)
        || (ODE_SENS_ENABLED && crate::sens::ode_provider::ode_analytical_supported(model))
}

/// Whether the exact analytic **outer** (population) FOCE/FOCEI gradient is
/// available for `model`: it is in the sensitivity provider's scope (non-IOV
/// [`sens_supported`] or [`iov_analytical_supported`]) and the user did not
/// force finite differences via `gradient_method = fd`.
///
/// Single source of truth for the analytic-vs-FD outer-gradient decision. The
/// outer-loop gradient dispatch (`outer_optimizer::population_gradient`),
/// [`Optimizer::resolve_auto`](crate::types::Optimizer::resolve_auto), and
/// `build_info::gradient_method_outer` all consult this so the `auto` optimizer
/// can never resolve to a gradient-based optimizer while the loop actually
/// computes an FD gradient (a drift that would run a gradient optimizer on a
/// noisy FD gradient). The per-eval `reconverge_gradient_interval` override is
/// orthogonal and handled at the dispatch site, not here.
///
/// TTE (`[event_model]`) endpoints are excluded: the hazard log-likelihood has no
/// analytic outer gradient (the sensitivity provider only covers the structural
/// PK/PD model), so a TTE — or mixed PK+TTE — objective must be differentiated by
/// finite differences. Without this guard a TTE model would report an analytic
/// gradient it cannot supply, so `resolve_auto` would pick a gradient-based
/// optimizer that then stalls on a meaningless gradient (TTE is FD-only — see
/// `docs/estimation/tte.qmd`).
pub fn analytic_outer_gradient_available(model: &CompiledModel) -> bool {
    !matches!(model.gradient_method, GradientMethod::Fd)
        && model.residual_correlations.is_empty()
        && !model.has_tte()
        // Custom residual-error magnitude (#484): θ-dependent variance not yet in
        // the analytic outer θ/σ kernels — FD gradient only (it is magnitude-aware).
        && !model.has_custom_ruv_magnitude()
        // `iov_sens_supported` (not just the closed-form `iov_analytical_supported`) so
        // the predicate also recognizes the ODE IOV outer gradient (#439 ODE IOV / #466).
        && (sens_supported(model) || iov_sens_supported(model))
        // IIV on residual error (#474): the analytic gradient (inner η-column +
        // outer θ/Ω/σ variance terms) is provider-agnostic, so it serves the
        // closed-form AND ODE paths. IOV and M3-BLOQ `iiv_on_ruv` keep the FD
        // gradient on BOTH loops — the IOV inner gradient does not carry the
        // variance scaling, and the residual-eta censored second derivatives are
        // not assembled.
        && !model.iiv_on_ruv_forces_fd()
}

/// Whether the light **ODE inner** η-gradient (`Dual1`) serves this model+subject:
/// the master switch is armed and the subject is in the per-subject ODE scope —
/// either the static superposition walk ([`ode_subject_supported`]) or the
/// event-driven TV-cov walk ([`ode_tvcov_supported`]). Both are wired in
/// [`ode_subject_eta_grad`], so the inner EBE loop takes the analytic η-gradient
/// for TV-cov subjects too, matching the outer scope rather than splitting to an
/// FD inner (#410; #449 review — the TV-cov inner gate was previously missing).
///
/// [`ode_subject_supported`]: crate::sens::ode_provider::ode_subject_supported
/// [`ode_tvcov_supported`]: crate::sens::ode_provider::ode_tvcov_supported
/// [`ode_subject_eta_grad`]: crate::sens::ode_provider::ode_subject_eta_grad
pub(crate) fn ode_inner_grad_supported(model: &CompiledModel, subject: &Subject) -> bool {
    ODE_SENS_ENABLED
        && (crate::sens::ode_provider::ode_subject_supported(model, subject)
            || crate::sens::ode_provider::ode_tvcov_supported(model, subject))
}

/// The per-observation `∂f/∂η` Jacobian (`n_obs × n_eta`, row-major) as a flat
/// vector, or `None` when unsupported. Convenience for the inner loop, whose
/// `h_matrix` is exactly this Jacobian at the converged η̂. Uses the light
/// first-order provider ([`subject_eta_grad`]) — this is `∂f/∂η` only, so the
/// second-order `Dual2` work the full provider does would be wasted here.
pub fn subject_eta_jacobian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<f64>> {
    let sens = subject_eta_grad(model, subject, theta, eta)?;
    let n_eta = model.n_eta;
    let mut jac = Vec::with_capacity(sens.len() * n_eta);
    for obs in &sens {
        jac.extend_from_slice(&obs.df_deta);
    }
    debug_assert_eq!(jac.len(), subject.obs_times.len() * n_eta);
    Some(jac)
}

/// `∂tv_i/∂θ_m` by bound-agnostic central finite difference of `tv_fn`. Returns
/// a row-major `n_tv × n_theta` matrix. `tv_fn` folds covariates and evaluates
/// at η = 0, so this is purely the θ → typical-value Jacobian.
pub(crate) fn tv_theta_jacobian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
) -> Vec<Vec<f64>> {
    let tv_fn = model
        .tv_fn
        .as_ref()
        .expect("analytical_supported guarantees tv_fn is present");
    let n_theta = model.n_theta;
    let base = tv_fn(theta, &subject.covariates);
    let n_tv = base.len();
    let mut jac = vec![vec![0.0; n_theta]; n_tv];
    let mut th = theta.to_vec();
    for m in 0..n_theta {
        let h = 1e-6 * (1.0 + theta[m].abs());
        let orig = th[m];
        th[m] = orig + h;
        let up = tv_fn(&th, &subject.covariates);
        th[m] = orig - h;
        let dn = tv_fn(&th, &subject.covariates);
        th[m] = orig;
        for i in 0..n_tv {
            jac[i][m] = (up[i] - dn[i]) / (2.0 * h);
        }
    }
    jac
}

/// Fallback `∂p/∂(θ,η)` for the closed-form log-normal parameterization
/// `pk_i = tv_i·exp(Σ_k sel[i,k]·η_k)`, used when the exact `Dual2`-over-program
/// path is unavailable (NN-weight θ or IOV kappa make the program's axis counts
/// disagree with the model's θ/η). The η chain is closed form; the θ chain uses
/// the FD `tv_theta_jacobian` (ρ). Produces the same `ParamDerivs` shape the
/// program path returns so the downstream chain is identical:
///   `∂p_i/∂η_k = pk_i·sel_ik`, `∂p_i/∂θ_m = pk_i·ρ_im`,
///   `∂²p_i/∂η_k∂η_l = pk_i·sel_ik·sel_il`, `∂²p_i/∂η_k∂θ_m = pk_i·sel_ik·ρ_im`.
fn lognormal_param_derivatives(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    pk: &crate::types::PkParams,
) -> crate::sens::ode_provider::ParamDerivs {
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let tv = (model.tv_fn.as_ref().unwrap())(theta, &subject.covariates);
    let tv_jac = tv_theta_jacobian(model, subject, theta);
    let ni = model.pk_indices.len();
    let mut dp_deta = vec![vec![0.0; n_eta]; ni];
    let mut dp_dtheta = vec![vec![0.0; n_theta]; ni];
    let mut d2p_deta2 = vec![vec![vec![0.0; n_eta]; n_eta]; ni];
    let mut d2p_detadtheta = vec![vec![vec![0.0; n_theta]; n_eta]; ni];
    for (i, &slot) in model.pk_indices.iter().enumerate() {
        let pk_val = pk.values[slot];
        let tv_i = tv[i];
        let sel: Vec<f64> = (0..n_eta).map(|k| model.sel_flat[i * n_eta + k]).collect();
        let rho: Vec<f64> = if tv_i.abs() > 0.0 {
            (0..n_theta).map(|m| tv_jac[i][m] / tv_i).collect()
        } else {
            vec![0.0; n_theta]
        };
        for m in 0..n_theta {
            dp_dtheta[i][m] = pk_val * rho[m];
        }
        for k in 0..n_eta {
            dp_deta[i][k] = pk_val * sel[k];
            for l in 0..n_eta {
                d2p_deta2[i][k][l] = pk_val * sel[k] * sel[l];
            }
            for m in 0..n_theta {
                d2p_detadtheta[i][k][m] = pk_val * sel[k] * rho[m];
            }
        }
    }
    crate::sens::ode_provider::ParamDerivs {
        dp_deta,
        dp_dtheta,
        d2p_deta2,
        d2p_detadtheta,
    }
}

// ─── IOV (inter-occasion variability) analytic sensitivities ──────────
//
// IOV makes the PK parameters switch mid-decay at occasion boundaries (NONMEM
// #104): a dose given in occasion 1 decays with occasion-1 params until the
// boundary, then the carried-over amount continues with occasion-2 params. Dose
// superposition with a single per-subject `pk` (the `subject_sensitivities` path)
// cannot represent that, so IOV runs through the **event-driven sensitivity
// walk** (`crate::sens::propagate::event_driven_sens_one_cpt_g`) instead — the
// same engine production's `predict_iov` uses for the f64 prediction, here over
// `Dual2`.
//
// The inner/outer "η" for an IOV subject is the **stacked** random-effects vector
//   `[η_bsv (n_eta) , κ_group0 (n_kappa) , κ_group1 , … , κ_group(K−1)]`
// with `K = iov_occasion_groups(subject).len()`. The returned [`SubjectSens`]
// is over that stacked vector (plus the usual θ block), so the caller's block-Ω
// (BSV ⊕ K·IOV) assembly consumes it directly.

/// Per-occasion individual-parameter derivatives in the **combined** layout
/// `(θ, η_bsv, κ)` — the program's native axes for an IOV model
/// (`n_eff = n_eta_bsv + n_kappa`). One of these is built per occasion group from
/// that group's combined effect vector; the chain then scatters its η_bsv columns
/// to the shared BSV block and its κ columns to the group's own κ block.
///
/// Shared between the analytical IOV provider ([`subject_sensitivities_iov`]) and the
/// ODE IOV provider ([`crate::sens::ode_provider::ode_subject_sensitivities_iov`]) —
/// both seed the same compiled `[individual_parameters]` program over the combined
/// axes, then scatter onto a stacked-η dual; only the downstream walk (closed-form
/// vs RK45) differs (#439 ODE IOV).
pub(crate) struct CombinedDerivs {
    /// `∂p_i/∂(η_bsv, κ)`, row-major `n_rows × n_eff`.
    pub(crate) deta: Vec<Vec<f64>>,
    /// `∂p_i/∂θ_m`, `n_rows × n_theta`.
    pub(crate) dtheta: Vec<Vec<f64>>,
    /// `∂²p_i/∂(η_bsv,κ)²`, `n_rows × n_eff × n_eff`.
    pub(crate) d2eta: Vec<Vec<Vec<f64>>>,
    /// `∂²p_i/∂(η_bsv,κ)∂θ`, `n_rows × n_eff × n_theta`.
    pub(crate) d2eta_theta: Vec<Vec<Vec<f64>>>,
}

/// Build [`CombinedDerivs`] dispatching on the program's combined axis count
/// `MP = n_theta + n_eff` at runtime — the `const`-generic [`iov_combined_derivs`]
/// wrapped in the same `1..=24` table the analytical IOV walk uses, so the ODE IOV
/// provider can reuse the per-occasion derivative source without re-spelling the
/// dispatch. `None` when `MP` exceeds the table (caller falls back to FD).
pub(crate) fn iov_combined_derivs_dyn(
    prog: &crate::parser::model_parser::IndivParamProgram,
    n_theta: usize,
    n_eff: usize,
    n_rows: usize,
    cov: &std::collections::HashMap<String, f64>,
    theta: &[f64],
    combined: &[f64],
) -> Option<CombinedDerivs> {
    macro_rules! disp {
        ($($m:literal),+) => {
            match n_theta + n_eff {
                $($m => Some(iov_combined_derivs::<$m>(
                    prog, n_theta, n_eff, n_rows, cov, theta, combined,
                )),)+
                _ => None,
            }
        };
    }
    disp!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
}

/// Evaluate the compiled `[individual_parameters]` program over `Dual2<MP>` seeded
/// on `(θ, combined)` (`combined = [η_bsv, κ]`, `MP = n_theta + n_eff`) and pack
/// the per-row derivatives in the combined layout.
fn iov_combined_derivs<const MP: usize>(
    prog: &crate::parser::model_parser::IndivParamProgram,
    n_theta: usize,
    n_eff: usize,
    n_rows: usize,
    cov: &std::collections::HashMap<String, f64>,
    theta: &[f64],
    combined: &[f64],
) -> CombinedDerivs {
    let p = prog.eval_param_duals::<MP>(theta, combined, cov);
    let mut deta = vec![vec![0.0; n_eff]; n_rows];
    let mut dtheta = vec![vec![0.0; n_theta]; n_rows];
    let mut d2eta = vec![vec![vec![0.0; n_eff]; n_eff]; n_rows];
    let mut d2eta_theta = vec![vec![vec![0.0; n_theta]; n_eff]; n_rows];
    for i in 0..n_rows {
        let g = &p[i].grad;
        let h = &p[i].hess;
        for m in 0..n_theta {
            dtheta[i][m] = g[m];
        }
        for a in 0..n_eff {
            deta[i][a] = g[n_theta + a];
            for b in 0..n_eff {
                d2eta[i][a][b] = h[n_theta + a][n_theta + b];
            }
            for m in 0..n_theta {
                d2eta_theta[i][a][m] = h[n_theta + a][m];
            }
        }
    }
    CombinedDerivs {
        deta,
        dtheta,
        d2eta,
        d2eta_theta,
    }
}

/// True when [`subject_sensitivities_iov`] can serve this model: any analytical
/// 1-/2-/3-cpt IOV model (`n_kappa > 0`), no ODE, no scaling/LTBS/lagtime, a usable
/// `[individual_parameters]` program whose axes are `(n_theta, n_eta_bsv+n_kappa)`.
/// Time-varying covariates ARE supported (each event's PK-param derivatives are
/// seeded at that event's covariate snapshot). Narrowly scoped on purpose —
/// anything outside falls back to the gradient-free path (matching the rest of the
/// provider's gating).
pub fn iov_analytical_supported(model: &CompiledModel) -> bool {
    if model.n_kappa == 0 || model.ode_spec.is_some() {
        return false;
    }
    // M3 BLOQ: the IOV objective promotes M3 to the interaction (censored) marginal
    // (`foce_subject_nll_iov`), but the IOV analytic gradient assembly carries no
    // censored-row term — it would differentiate a different function than it
    // minimises. Route IOV+M3 to the FD/Laplace path until the censored IOV
    // gradient lands. (Non-IOV M3 is fully analytic.)
    if matches!(model.bloq_method, crate::types::BloqMethod::M3) {
        return false;
    }
    // IIV on residual error (`iiv_on_ruv`) IS analytic for closed-form IOV models:
    // `η_ruv` enters only through the variance (`v = R(f)·exp(2·η_ruv)`, `∂f/∂η_ruv = 0`),
    // and both halves now carry that scaling — the IOV inner gradient
    // (`analytic_eta_nll_gradient_iov`) scales `v`/`dv_df` and adds the `η_ruv` variance
    // column, and the IOV outer assembly threads `ruv = residual_error_eta` into
    // `prepare_stacked` (the residual-eta `c̃` column rides the stacked `[η_bsv, κ]`
    // layout). Mirrors the non-IOV `iiv_on_ruv` path (#474). Only IOV + M3-BLOQ still
    // forces FD — the censored residual-eta second derivatives are not assembled (gated by
    // the M3 check above, and by `iiv_on_ruv_forces_fd`). (ODE IOV + `iiv_on_ruv` stays FD
    // via `ode_iov_supported`, which is unchanged.)
    // FREM + IOV: the analytic IOV inner gradient uses the ordinary residual variance for
    // every observation row, never the FREM covariate pseudo-obs variance the objective
    // (`individual_nll_iov`) uses for those rows; and the IOV objective returns a `1e18`
    // sentinel for FREM+IOV anyway. Route to FD. (#466 review round 2.)
    if model.frem_config.is_some() {
        return false;
    }
    if !matches!(
        model.pk_model,
        PkModel::OneCptIv
            | PkModel::OneCptOral
            | PkModel::TwoCptIv
            | PkModel::TwoCptOral
            | PkModel::ThreeCptIv
            | PkModel::ThreeCptOral
    ) {
        return false;
    }
    if !matches!(model.scaling, ScalingSpec::None) || model.log_transform || model.has_lagtime() {
        return false;
    }
    // Initial-compartment amounts (#521) ARE differentiable by the analytic kernels
    // now (#524), but only on the dose-superposition path: the init impulse is
    // layered per-subject in `subject_sensitivities` / `subject_eta_grad`, not in the
    // IOV event-driven walk, which would have to re-seed the `A₀·kernel` baseline into
    // each occasion block. Until that occasion-block seeding lands, an IOV + init
    // model falls back to FD (see `analytical_supported` for the non-IOV exact path).
    if !model.analytical_init.is_empty() {
        return false;
    }
    let n_eff = model.n_eta + model.n_kappa;
    match model.indiv_param_partials.indiv_param_program.as_ref() {
        Some(prog) => {
            prog_covers_required_pk_slots(model, prog)
                && prog.n_theta_axis() == model.n_theta
                && prog.n_eta_axis() == n_eff
        }
        None => false,
    }
}

/// True when the exact analytic IOV outer gradient applies to this model: either the
/// closed-form analytical IOV provider ([`iov_analytical_supported`]) or the ODE IOV
/// provider ([`crate::sens::ode_provider::ode_iov_supported`]). Gates the IOV branch
/// of the outer-gradient dispatch (`population_gradient_sens_iov_mixed`, which assembles
/// per subject with per-subject reconverged-FD salvage), the IOV analogue of
/// [`sens_supported`] (#439 ODE IOV).
pub fn iov_sens_supported(model: &CompiledModel) -> bool {
    iov_analytical_supported(model)
        || (ODE_SENS_ENABLED && crate::sens::ode_provider::ode_iov_supported(model))
}

/// Light **inner** η-gradient (`∂f/∂(stacked-η)` per observation) for an IOV subject
/// over the stacked vector `[η_bsv, κ₁..κ_K]`, or `None` when no analytic inner serves
/// the model (caller falls back to the FD inner). Dispatches to the ODE IOV provider
/// for RHS-program models and the closed-form Dual1 walk for analytical 1-/2-/3-cpt
/// IOV models — so both get an exact analytic inner EBE gradient (#439 IOV inner).
pub(crate) fn subject_eta_grad_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_eta: &[f64],
) -> Option<Vec<ObsGrad>> {
    if model.ode_spec.is_some() {
        if ODE_SENS_ENABLED {
            return crate::sens::ode_provider::ode_subject_eta_grad_iov(
                model,
                subject,
                theta,
                stacked_eta,
            );
        }
        return None;
    }
    subject_eta_grad_iov_analytical(model, subject, theta, stacked_eta)
}

/// Exact analytic sensitivities for an analytical 1-/2-/3-cpt **IOV** subject, over
/// the stacked random-effects vector `[η_bsv, κ_group0, …, κ_group(K−1)]` (plus the θ
/// block). Returns `None` outside the supported scope (caller falls back).
///
/// `stacked_eta` must have length `n_eta_bsv + K·n_kappa`. Each occasion group's
/// PK parameters are seeded on their own `Dual2` axis block; the event-driven walk
/// carries the dual amounts across occasion boundaries (exact carryover), and the
/// `run_obs`-style two-level chain maps `∂conc/∂pk` back to the stacked η and θ
/// through each group's [`CombinedDerivs`].
pub fn subject_sensitivities_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_eta: &[f64],
) -> Option<SubjectSens> {
    // ODE IOV: route RHS-program models to the ODE provider, which runs the same
    // stacked-`(θ, η_bsv, κ)` layout over the event-driven RK45 walk (the TV-cov
    // walk fed per-occasion params). Returns the identical `SubjectSens` shape, so
    // the block-Ω assembly (`prepare_stacked`) consumes it unchanged. Out-of-scope
    // ODE subjects return `None` → the population drops to FD, mirroring the
    // analytical IOV gate (#439 ODE IOV).
    if model.ode_spec.is_some() {
        if ODE_SENS_ENABLED {
            return crate::sens::ode_provider::ode_subject_sensitivities_iov(
                model,
                subject,
                theta,
                stacked_eta,
            );
        }
        return None;
    }
    if !iov_analytical_supported(model) {
        return None;
    }
    let s = build_iov_sources(model, subject, theta, stacked_eta)?;
    // Run the walk over `Dual2<M>` (M = n_theta + n_stacked); the dual width tracks
    // the *unknowns* (n_eta + K·n_kappa + n_theta), not the PK axes, so it stays
    // narrow for many occasions whenever n_kappa < n_diff (the usual κ-on-CL case).
    let m_dim = s.n_theta + s.n_stacked;
    macro_rules! disp {
        ($($m:literal),+) => {
            match m_dim {
                $($m => run_obs_iov::<$m>(
                    model, subject, &s.sources, &s.dose_src, &s.obs_src, &s.pkonly_src,
                    &s.slot_row, s.n_eta, s.n_kappa, s.n_eff, s.n_stacked, s.n_theta,
                ),)+
                _ => None,
            }
        };
    }
    disp!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
}

/// Per-event seed sources for the analytical IOV walk — the per-subject gates, the
/// occasion split, and one [`CombinedDerivs`] per event (or per occasion group when
/// covariates are static). Shared by the outer Dual2 walk ([`subject_sensitivities_iov`])
/// and the inner Dual1 walk ([`subject_eta_grad_iov_analytical`]) so the per-subject
/// scope and the κ-axis sources stay identical across the two dual orders. Callers
/// must have checked [`iov_analytical_supported`] first. `None` out of scope (#439).
struct IovSources {
    /// `(pk, cd, group)` per source; `group = Some(g)` scatters κ to occasion group
    /// `g`'s stacked block, `None` (pk_only) drops it.
    sources: Vec<(crate::types::PkParams, CombinedDerivs, Option<usize>)>,
    dose_src: Vec<usize>,
    obs_src: Vec<usize>,
    pkonly_src: Vec<usize>,
    slot_row: [Option<usize>; N_PK],
    n_eta: usize,
    n_kappa: usize,
    n_eff: usize,
    n_stacked: usize,
    n_theta: usize,
}

fn build_iov_sources(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_eta: &[f64],
) -> Option<IovSources> {
    // #419: decline a rate-defined infusion under `F ≠ 1` to the FD gradient — the
    // Dual2 walk applies `F` as an inline magnitude scale on the rate
    // (`propagate.rs`: `pk.f * rate`) over the unscaled window, so it can't
    // represent the `F`-scaled window (see `subject_sensitivities_impl` for the
    // full rationale).
    if model.has_bioavailability() && subject.has_rate_defined_infusion() {
        return None;
    }
    // EVID=3/4 resets are honoured by the event-driven walk: it zeros the dual
    // state at each reset and rebuilds the post-reset occasion from the schedule,
    // exactly as production's `event_driven_predictions` does (the `f64` instance
    // of the same walk). Steady-state doses assume an infinite periodic history
    // that a mid-record reset contradicts, so a subject mixing SS with resets
    // falls back to FD — mirroring the non-IOV provider.
    if subject.has_resets() && subject.doses.iter().any(|d| d.ss) {
        return None;
    }
    // Modeled-duration doses (`RATE=-2`) are read unresolved here; route to FD
    // (mirrors the non-IOV provider gate).
    if !subject.all_doses_fixed() {
        return None;
    }

    let n_eta = model.n_eta; // BSV
    let n_kappa = model.n_kappa;
    let n_theta = model.n_theta;
    let n_eff = n_eta + n_kappa;

    let occ_groups = crate::stats::likelihood::iov_occasion_groups(subject);
    let k_groups = occ_groups.len();
    if k_groups == 0 {
        return None;
    }
    let n_stacked = n_eta + k_groups * n_kappa;
    if stacked_eta.len() != n_stacked {
        return None;
    }
    let occ_to_k = crate::stats::likelihood::iov_occ_to_k(&occ_groups);
    // Combined effect vector for group `k`: `[η_bsv, κ_k]` (shared κ-axis layout).
    let combined_for =
        |k: usize| crate::stats::likelihood::iov_combined_effect(stacked_eta, n_eta, n_kappa, k);
    // EVID=2 (`pk_only`) rows carry no occasion label → BSV η with zero κ (matches
    // production `predict_iov`). Their κ derivatives are dropped from the stacked
    // axes (group `None` below), so the prediction holds κ fixed at 0. Single-sourced
    // with the ODE IOV provider via the shared helper (#598 review).
    let combined_pk_only: Vec<f64> =
        crate::stats::likelihood::iov_combined_pk_only(stacked_eta, n_eta, n_kappa);

    let prog = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .expect("iov_analytical_supported guarantees the program");
    let slots = prog.pk_slots_ref();
    let n_diff = slots.len();
    // PK slot → differentiated-row index (for seeding the dual axis).
    let mut slot_row: [Option<usize>; N_PK] = [None; N_PK];
    for (i, &sl) in slots.iter().enumerate() {
        if sl < N_PK {
            slot_row[sl] = Some(i);
        }
    }

    // Combined derivatives at `(theta, combined)` evaluated at covariate map `cov`,
    // dispatching the program-eval width `MP = n_theta + n_eff`.
    let cd_at = |combined: &[f64], cov: &std::collections::HashMap<String, f64>| {
        crate::sens::provider::iov_combined_derivs_dyn(
            prog, n_theta, n_eff, n_diff, cov, theta, combined,
        )
    };

    // Per-event seed sources `(pk, cd, group)`. Each event's derivatives are evaluated
    // at that event's covariate snapshot, so a time-varying covariate is exact (no
    // per-group caching across events). When covariates are subject-static, one source
    // per occasion group is built and shared, preserving the non-TV cost.
    let has_tv = subject.has_tv_covariates();
    let cov_static = &subject.covariates;
    let mut sources: Vec<(crate::types::PkParams, CombinedDerivs, Option<usize>)> = Vec::new();
    let mut dose_src = vec![0usize; subject.doses.len()];
    let mut obs_src = vec![0usize; subject.obs_times.len()];
    let mut pkonly_src = vec![0usize; subject.pk_only_times.len()];

    if has_tv {
        for d in 0..subject.doses.len() {
            let occ = subject.dose_occasions.get(d).copied()?;
            let g = *occ_to_k.get(&occ)?;
            let combined = combined_for(g);
            let cov = subject.dose_cov(d);
            let pk = (model.pk_param_fn)(theta, &combined, cov);
            let cd = cd_at(&combined, cov)?;
            dose_src[d] = sources.len();
            sources.push((pk, cd, Some(g)));
        }
        for j in 0..subject.obs_times.len() {
            let occ = subject.occasions.get(j).copied()?;
            let g = *occ_to_k.get(&occ)?;
            let combined = combined_for(g);
            let cov = subject.obs_cov(j);
            let pk = (model.pk_param_fn)(theta, &combined, cov);
            let cd = cd_at(&combined, cov)?;
            obs_src[j] = sources.len();
            sources.push((pk, cd, Some(g)));
        }
        for m in 0..subject.pk_only_times.len() {
            let cov = subject.pk_only_cov(m);
            let pk = (model.pk_param_fn)(theta, &combined_pk_only, cov);
            let cd = cd_at(&combined_pk_only, cov)?;
            pkonly_src[m] = sources.len();
            sources.push((pk, cd, None));
        }
    } else {
        // One source per occasion group, at the subject-static covariates.
        let mut group_source = vec![usize::MAX; k_groups];
        for g in 0..k_groups {
            let combined = combined_for(g);
            let pk = (model.pk_param_fn)(theta, &combined, cov_static);
            let cd = cd_at(&combined, cov_static)?;
            group_source[g] = sources.len();
            sources.push((pk, cd, Some(g)));
        }
        for d in 0..subject.doses.len() {
            let occ = subject.dose_occasions.get(d).copied()?;
            dose_src[d] = group_source[*occ_to_k.get(&occ)?];
        }
        for j in 0..subject.obs_times.len() {
            let occ = subject.occasions.get(j).copied()?;
            obs_src[j] = group_source[*occ_to_k.get(&occ)?];
        }
        if !subject.pk_only_times.is_empty() {
            let pk = (model.pk_param_fn)(theta, &combined_pk_only, cov_static);
            let cd = cd_at(&combined_pk_only, cov_static)?;
            let idx = sources.len();
            sources.push((pk, cd, None));
            for m in 0..subject.pk_only_times.len() {
                pkonly_src[m] = idx;
            }
        }
    }

    Some(IovSources {
        sources,
        dose_src,
        obs_src,
        pkonly_src,
        slot_row,
        n_eta,
        n_kappa,
        n_eff,
        n_stacked,
        n_theta,
    })
}

/// Light **inner** η-gradient (`∂f/∂(stacked-η)` per observation) for an analytical
/// IOV subject — the closed-form counterpart of
/// [`crate::sens::ode_provider::ode_subject_eta_grad_iov`] and the inner sibling of
/// [`subject_sensitivities_iov`]. Runs the event-driven sensitivity walk over the light
/// `Dual1<N>` (`N = n_stacked`), reading only `∂f/∂(stacked-η)` (no θ block, no Hessian).
/// `None` outside the IOV-analytical scope (#439 closed-form IOV inner).
fn subject_eta_grad_iov_analytical(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_eta: &[f64],
) -> Option<Vec<ObsGrad>> {
    if !iov_analytical_supported(model) {
        return None;
    }
    let s = build_iov_sources(model, subject, theta, stacked_eta)?;
    let n_dim = s.n_stacked;
    macro_rules! disp {
        ($($n:literal),+) => {
            match n_dim {
                $($n => run_obs_iov_eta::<$n>(
                    model, subject, &s.sources, &s.dose_src, &s.obs_src, &s.pkonly_src,
                    &s.slot_row, s.n_eta, s.n_kappa, s.n_stacked,
                ),)+
                _ => None,
            }
        };
    }
    disp!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
}

/// The dual-width-`M` inner of [`subject_sensitivities_iov`] (`M = n_theta +
/// n_stacked`). Builds each event's PK-param duals seeded directly on the stacked
/// `(θ, η_bsv, κ)` unknowns (from that event's [`CombinedDerivs`] source), runs the
/// event-driven sensitivity walk over `Dual2<M>`, and reads `∂conc/∂unknowns`
/// straight off the resulting dual — the walk composes the whole chain, so there
/// is no separate two-level assembly. Dual dimension `m < n_theta` is `θ_m`;
/// `n_theta + p` is stacked-η axis `p`.
///
/// `sources[src] = (pk, cd, group)`; `dose_src`/`obs_src`/`pkonly_src` map each
/// event to its source index. `group = Some(g)` scatters the κ columns to occasion
/// group `g`'s stacked block; `None` (a pk_only / EVID=2 event) drops them, so the
/// prediction holds κ fixed at 0. One dual is built per *source* and cached, so a
/// subject-static-covariate subject still pays one build per occasion group.
#[allow(clippy::too_many_arguments)]
fn run_obs_iov<const M: usize>(
    model: &CompiledModel,
    subject: &Subject,
    sources: &[(crate::types::PkParams, CombinedDerivs, Option<usize>)],
    dose_src: &[usize],
    obs_src: &[usize],
    pkonly_src: &[usize],
    slot_row: &[Option<usize>; N_PK],
    n_eta: usize,
    n_kappa: usize,
    n_eff: usize,
    n_stacked: usize,
    n_theta: usize,
) -> Option<SubjectSens> {
    use crate::pk::event_driven::EventSchedule;
    use crate::sens::propagate::{event_driven_sens_g, PkDual};

    // Build the `Dual2<M>` for a differentiated PK row `i` of source `cd`/`group`,
    // carrying `val` and `∂/∂(θ, stacked-η)`. The combined column `c` maps to a
    // stacked axis: η_bsv (`c < n_eta`) → shared `n_theta + c`; κ (`c ≥ n_eta`) →
    // group g's block `n_theta + n_eta + g·n_kappa + (c−n_eta)`, or is dropped when
    // `group` is `None` (pk_only event, κ fixed at 0). The θ-θ Hessian block is
    // unused downstream (left zero).
    let seed = |cd: &CombinedDerivs, group: Option<usize>, i: usize, val: f64| -> Dual2<M> {
        let kappa_base = group.map(|g| n_theta + n_eta + g * n_kappa);
        let stacked_axis = |c: usize| -> Option<usize> {
            if c < n_eta {
                Some(n_theta + c)
            } else {
                kappa_base.map(|kb| kb + (c - n_eta))
            }
        };
        let mut grad = [0.0; M];
        let mut hess = [[0.0; M]; M];
        for m in 0..n_theta.min(M) {
            grad[m] = cd.dtheta[i][m];
        }
        for c in 0..n_eff {
            let ax = match stacked_axis(c) {
                Some(ax) if ax < M => ax,
                _ => continue,
            };
            grad[ax] = cd.deta[i][c];
            for d in 0..n_eff {
                if let Some(bx) = stacked_axis(d) {
                    if bx < M {
                        hess[ax][bx] = cd.d2eta[i][c][d];
                    }
                }
            }
            for m in 0..n_theta.min(M) {
                let v = cd.d2eta_theta[i][c][m];
                hess[ax][m] = v;
                hess[m][ax] = v;
            }
        }
        Dual2 {
            value: val,
            grad,
            hess,
        }
    };

    // Per-source PK param duals: seed differentiated slots, constants otherwise.
    let mk = |src: usize| -> PkDual<Dual2<M>> {
        let (pk, cd, group) = &sources[src];
        let dv = |slot: usize, val: f64| -> Dual2<M> {
            match slot_row[slot] {
                Some(i) => seed(cd, *group, i, val),
                None => Dual2::<M>::constant(val),
            }
        };
        PkDual {
            cl: dv(PK_IDX_CL, pk.cl()),
            v: dv(PK_IDX_V, pk.v()),
            q: dv(PK_IDX_Q, pk.q()),
            v2: dv(PK_IDX_V2, pk.v2()),
            ka: dv(PK_IDX_KA, pk.ka()),
            q3: dv(PK_IDX_Q3, pk.q3()),
            v3: dv(PK_IDX_V3, pk.v3()),
            f: dv(PK_IDX_F, pk.f_bio()),
        }
    };

    // One dual per source, cached across the events that share it.
    let mut src_dual: Vec<Option<PkDual<Dual2<M>>>> = vec![None; sources.len()];
    let mut event_dual = |src: usize| -> PkDual<Dual2<M>> {
        if src_dual[src].is_none() {
            src_dual[src] = Some(mk(src));
        }
        src_dual[src].unwrap()
    };
    let mut pk_at_dose: Vec<PkDual<Dual2<M>>> = Vec::with_capacity(subject.doses.len());
    for &src in dose_src {
        pk_at_dose.push(event_dual(src));
    }
    let mut pk_at_obs: Vec<PkDual<Dual2<M>>> = Vec::with_capacity(subject.obs_times.len());
    for &src in obs_src {
        pk_at_obs.push(event_dual(src));
    }
    let mut pk_at_pk_only: Vec<PkDual<Dual2<M>>> = Vec::with_capacity(subject.pk_only_times.len());
    for &src in pkonly_src {
        pk_at_pk_only.push(event_dual(src));
    }

    // No lagtime in IOV scope → zero dose lagtimes.
    let dose_lagtimes = vec![0.0; subject.doses.len()];
    let schedule =
        EventSchedule::for_subject(subject, model.pk_model, &subject.doses, &dose_lagtimes);
    let conc = event_driven_sens_g::<Dual2<M>>(
        model.pk_model,
        subject,
        &schedule,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    let mut obs_out = Vec::with_capacity(conc.len());
    for c in &conc {
        // Clamp parity with production `conc.max(0.0)`: a negative value's
        // derivatives vanish (consistency with the OFV).
        let neg = c.value < 0.0;
        let mut df_deta = vec![0.0; n_stacked];
        let mut df_dtheta = vec![0.0; n_theta];
        let mut d2f_deta2 = vec![0.0; n_stacked * n_stacked];
        let mut d2f_deta_dtheta = vec![0.0; n_stacked * n_theta];
        if !neg {
            for p in 0..n_stacked {
                df_deta[p] = c.grad[n_theta + p];
                for q in 0..n_stacked {
                    d2f_deta2[p * n_stacked + q] = c.hess[n_theta + p][n_theta + q];
                }
                for m in 0..n_theta {
                    d2f_deta_dtheta[p * n_theta + m] = c.hess[n_theta + p][m];
                }
            }
            for m in 0..n_theta {
                df_dtheta[m] = c.grad[m];
            }
        }
        obs_out.push(ObsSens {
            f: if neg { 0.0 } else { c.value },
            df_deta,
            d2f_deta2,
            df_dtheta,
            d2f_deta_dtheta,
        });
    }
    Some(SubjectSens { obs: obs_out })
}

/// Light first-order (`Dual1<N>`, `N = n_stacked`) inner of
/// [`subject_eta_grad_iov_analytical`] — the η-only counterpart of [`run_obs_iov`].
/// Seeds each event's PK params on the stacked-η axes (no θ, no Hessian) from its
/// [`CombinedDerivs`], runs the event-driven sensitivity walk, and reads
/// `∂conc/∂(stacked-η)` straight off the `Dual1`. Same per-source caching and clamp
/// parity as the Dual2 walk (#439 closed-form IOV inner).
#[allow(clippy::too_many_arguments)]
fn run_obs_iov_eta<const N: usize>(
    model: &CompiledModel,
    subject: &Subject,
    sources: &[(crate::types::PkParams, CombinedDerivs, Option<usize>)],
    dose_src: &[usize],
    obs_src: &[usize],
    pkonly_src: &[usize],
    slot_row: &[Option<usize>; N_PK],
    n_eta: usize,
    n_kappa: usize,
    n_stacked: usize,
) -> Option<Vec<ObsGrad>> {
    use crate::pk::event_driven::EventSchedule;
    use crate::sens::propagate::{event_driven_sens_g, PkDual};

    // First-order stacked-η seed (no θ axes): η_bsv column `c` → axis `c`; κ column
    // `c` (group g) → `n_eta + g·n_kappa + (c−n_eta)`, dropped when `group` is `None`
    // (pk_only, κ fixed at 0).
    let seed = |cd: &CombinedDerivs, group: Option<usize>, i: usize, val: f64| -> Dual1<N> {
        let kappa_base = group.map(|g| n_eta + g * n_kappa);
        let stacked_axis = |c: usize| -> Option<usize> {
            if c < n_eta {
                Some(c)
            } else {
                kappa_base.map(|kb| kb + (c - n_eta))
            }
        };
        let mut grad = [0.0; N];
        for c in 0..cd.deta[i].len() {
            if let Some(ax) = stacked_axis(c) {
                if ax < N {
                    grad[ax] = cd.deta[i][c];
                }
            }
        }
        Dual1 { value: val, grad }
    };
    let mk = |src: usize| -> PkDual<Dual1<N>> {
        let (pk, cd, group) = &sources[src];
        let dv = |slot: usize, val: f64| -> Dual1<N> {
            match slot_row[slot] {
                Some(i) => seed(cd, *group, i, val),
                None => Dual1::<N>::constant(val),
            }
        };
        PkDual {
            cl: dv(PK_IDX_CL, pk.cl()),
            v: dv(PK_IDX_V, pk.v()),
            q: dv(PK_IDX_Q, pk.q()),
            v2: dv(PK_IDX_V2, pk.v2()),
            ka: dv(PK_IDX_KA, pk.ka()),
            q3: dv(PK_IDX_Q3, pk.q3()),
            v3: dv(PK_IDX_V3, pk.v3()),
            f: dv(PK_IDX_F, pk.f_bio()),
        }
    };
    let mut src_dual: Vec<Option<PkDual<Dual1<N>>>> = vec![None; sources.len()];
    let mut event_dual = |src: usize| -> PkDual<Dual1<N>> {
        if src_dual[src].is_none() {
            src_dual[src] = Some(mk(src));
        }
        src_dual[src].unwrap()
    };
    let mut pk_at_dose: Vec<PkDual<Dual1<N>>> = Vec::with_capacity(subject.doses.len());
    for &src in dose_src {
        pk_at_dose.push(event_dual(src));
    }
    let mut pk_at_obs: Vec<PkDual<Dual1<N>>> = Vec::with_capacity(subject.obs_times.len());
    for &src in obs_src {
        pk_at_obs.push(event_dual(src));
    }
    let mut pk_at_pk_only: Vec<PkDual<Dual1<N>>> = Vec::with_capacity(subject.pk_only_times.len());
    for &src in pkonly_src {
        pk_at_pk_only.push(event_dual(src));
    }

    let dose_lagtimes = vec![0.0; subject.doses.len()];
    let schedule =
        EventSchedule::for_subject(subject, model.pk_model, &subject.doses, &dose_lagtimes);
    let conc = event_driven_sens_g::<Dual1<N>>(
        model.pk_model,
        subject,
        &schedule,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    let mut out = Vec::with_capacity(conc.len());
    for c in &conc {
        // Clamp parity with production `conc.max(0.0)`: a negative value's derivatives
        // vanish (consistency with the OFV / the Dual2 walk).
        let neg = c.value < 0.0;
        let mut df_deta = vec![0.0; n_stacked];
        if !neg {
            for (p, df) in df_deta.iter_mut().enumerate() {
                *df = c.grad[p];
            }
        }
        out.push(ObsGrad {
            f: if neg { 0.0 } else { c.value },
            df_deta,
        });
    }
    Some(out)
}

/// True when [`subject_sensitivities_tvcov`] can serve this model: an analytical
/// 1-/2-/3-cpt model whose individual parameters carry **time-varying covariates**
/// (the covariate enters the program, so the PK parameters switch mid-decay — the
/// same shape as IOV, handled by the event-driven Dual2 walk rather than dose
/// superposition).
///
/// First cut (the rest routes to FD until the per-event dual walk is extended):
/// no IOV (the analytical gate already requires `n_kappa == 0`), no dose lagtime,
/// no LTBS, and only `None` / constant `ScalarScale` output scaling. A constant
/// `ScalarScale` divisor is covariate-independent, so it divides the whole jet
/// uniformly; an `ExpressionScale` could itself reference the time-varying
/// covariate and would need a per-observation scale jet (deferred). Requires the
/// compiled individual-parameter program with `(θ, η)` axis counts matching the
/// model, so each event's `∂p/∂(θ, η)` (+ second order) can be evaluated at *that
/// event's* covariate snapshot via
/// [`pd_from_program`](crate::sens::ode_provider::pd_from_program).
pub fn tvcov_analytical_supported(model: &CompiledModel) -> bool {
    if !analytical_supported(model) || model.has_lagtime() || model.log_transform {
        return false;
    }
    // The TV-cov event-driven walk does not yet layer the analytic
    // `[initial_conditions]` impulse — #524 added exact init gradients only to the
    // dose-superposition path (`subject_sensitivities` / `subject_eta_grad`), not to
    // `run_obs_tvcov`. Production's `compute_predictions_with_tv`, however, always
    // adds the `A₀·kernel` baseline. Admitting an init model here would walk a
    // gradient that omits the init while the objective keeps it — a silent mismatch
    // that biases the FOCE/FOCEI gradient (outer and inner). Decline init models so
    // their TV-cov / oral-infusion subjects fall back to FD (which differentiates the
    // true objective); their non-TV-cov subjects still take the exact init path. Lift
    // this once the walk carries the init impulse (#524 follow-up).
    if !model.analytical_init.is_empty() {
        return false;
    }
    // Bound total axes to the dual-walk dispatch cap so the outer (`m_dim`) and inner
    // (`n_eta`) TV-cov tables both resolve — matched analytic scope, no fixed-EBE FD
    // inner split (#449 re-review #2).
    if model.n_theta + model.n_eta > MAX_TVCOV_AXES {
        return false;
    }
    // Constant `ScalarScale` is a covariate-independent divisor (applied to the
    // whole jet below); `ExpressionScale` / `PerCmt` need a per-event scale jet
    // and route to FD for now.
    if !matches!(
        model.scaling,
        ScalingSpec::None | ScalingSpec::ScalarScale(_)
    ) {
        return false;
    }
    match model.indiv_param_partials.indiv_param_program.as_ref() {
        Some(prog) => {
            prog_covers_required_pk_slots(model, prog)
                && prog.n_theta_axis() == model.n_theta
                && prog.n_eta_axis() == model.n_eta
        }
        None => false,
    }
}

/// Exact analytic sensitivities for an analytical subject with **time-varying
/// covariates**, over the ordinary `(η, θ)` blocks — the standard
/// [`SubjectSens`] shape, identical to the non-TV provider, so the outer gradient
/// and inner Jacobian consume it unchanged.
///
/// This is the IOV walk minus κ: each event's PK-param duals are seeded at *that
/// event's* covariate snapshot (via [`pd_from_program`](crate::sens::ode_provider::pd_from_program),
/// which evaluates `∂p/∂(θ, η)` + second order at an arbitrary covariate map), and
/// the event-driven Dual2 walk carries the dual amounts across the covariate
/// breakpoints — switching to each event's params, exactly as production's
/// `compute_predictions_with_tv` does over `f64`. EVID=2 (`pk_only`) covariate
/// breakpoints between observations are seeded and walked too. Returns `None`
/// outside the supported scope (caller falls back to FD).
pub fn subject_sensitivities_tvcov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<SubjectSens> {
    // Routed here for time-varying covariates *or* oral infusion (#350/#400) — both
    // need the state-propagating walk rather than dose superposition.
    if !tvcov_analytical_supported(model)
        || !(subject.has_tv_covariates() || subject_has_oral_infusion(model, subject))
    {
        return None;
    }
    // Steady-state doses equilibrate per-event in the walk (`equilibrate_ss_g`,
    // at each dose's covariate snapshot), exactly as production's event-driven
    // predictor does. A steady-state dose assumes an infinite periodic history,
    // which a mid-record reset contradicts, so a subject mixing SS with resets
    // falls back to FD — mirroring the non-IOV / IOV providers.
    if subject.has_resets() && subject.doses.iter().any(|d| d.ss) {
        return None;
    }

    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let m_dim = n_theta + n_eta;

    let prog = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .expect("tvcov_analytical_supported guarantees the program");
    let slots = prog.pk_slots_ref();
    // PK slot → differentiated-row index of `pd_from_program` (for seeding the
    // dual axis). `pd` rows follow `pk_slots()` order, so row `i` ↔ slot `slots[i]`.
    let mut slot_row: [Option<usize>; N_PK] = [None; N_PK];
    for (i, &s) in slots.iter().enumerate() {
        if s < N_PK {
            slot_row[s] = Some(i);
        }
    }

    macro_rules! disp {
        ($($m:literal),+) => {
            match m_dim {
                $($m => run_obs_tvcov::<$m>(
                    model, subject, theta, eta, prog, &slot_row, n_eta, n_theta,
                ),)+
                _ => None,
            }
        };
    }
    let mut sens = disp!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    )?;

    // Constant `ScalarScale` output divisor `f_scaled = f/k`: every derivative is
    // linear in `f` and `k` is constant, so the whole jet divides by `k` — matches
    // `pk::apply_scaling` (`pred /= s`) on the production TV-cov path. The gate
    // admits only `None` / `ScalarScale`, so no other scaling reaches here.
    if let ScalingSpec::ScalarScale(k) = model.scaling {
        if k != 1.0 {
            let inv = 1.0 / k;
            for o in sens.obs.iter_mut() {
                o.f *= inv;
                for v in o
                    .df_deta
                    .iter_mut()
                    .chain(o.d2f_deta2.iter_mut())
                    .chain(o.df_dtheta.iter_mut())
                    .chain(o.d2f_deta_dtheta.iter_mut())
                {
                    *v *= inv;
                }
            }
        }
    }
    Some(sens)
}

/// The dual-width-`M` inner of [`subject_sensitivities_tvcov`] (`M = n_theta +
/// n_eta`). For each event, evaluates the individual-parameter program's
/// `∂p/∂(θ, η)` at that event's covariate snapshot, seeds the PK-param duals on the
/// `(θ, η)` axes (`θ_m → m`, `η_k → n_theta + k`), runs the event-driven
/// sensitivity walk over `Dual2<M>`, and reads `∂conc/∂(θ, η)` straight off into
/// the standard `(n_eta, n_theta)` [`SubjectSens`].
#[allow(clippy::too_many_arguments)]
fn run_obs_tvcov<const M: usize>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    prog: &crate::parser::model_parser::IndivParamProgram,
    slot_row: &[Option<usize>; N_PK],
    n_eta: usize,
    n_theta: usize,
) -> Option<SubjectSens> {
    use crate::pk::event_driven::EventSchedule;
    use crate::sens::ode_provider::pd_from_program;
    use crate::sens::propagate::{event_driven_sens_g, PkDual};

    // Build the per-event PK-param duals at a covariate snapshot: evaluate the
    // program's `∂p/∂(θ, η)` (+ 2nd order) at `cov`, then seed each differentiated
    // PK slot on its `(θ, η)` dual axis (`θ_m → m`, `η_k → n_theta + k`); constants
    // otherwise. The θ-θ Hessian block is unused downstream (left zero), mirroring
    // the IOV / scale seeders.
    let mk = |cov: &std::collections::HashMap<String, f64>| -> PkDual<Dual2<M>> {
        let pd = pd_from_program::<M>(prog, model, cov, theta, eta);
        let pk = (model.pk_param_fn)(theta, eta, cov);
        let seed_row = |i: usize, val: f64| -> Dual2<M> {
            let mut grad = [0.0; M];
            let mut hess = [[0.0; M]; M];
            for m in 0..n_theta.min(M) {
                grad[m] = pd.dp_dtheta[i][m];
            }
            for k in 0..n_eta {
                if n_theta + k < M {
                    grad[n_theta + k] = pd.dp_deta[i][k];
                }
                for l in 0..n_eta {
                    if n_theta + k < M && n_theta + l < M {
                        hess[n_theta + k][n_theta + l] = pd.d2p_deta2[i][k][l];
                    }
                }
                for m in 0..n_theta {
                    if n_theta + k < M && m < M {
                        let v = pd.d2p_detadtheta[i][k][m];
                        hess[n_theta + k][m] = v;
                        hess[m][n_theta + k] = v;
                    }
                }
            }
            Dual2 {
                value: val,
                grad,
                hess,
            }
        };
        let dv = |slot: usize, val: f64| -> Dual2<M> {
            match slot_row[slot] {
                Some(i) => seed_row(i, val),
                None => Dual2::<M>::constant(val),
            }
        };
        PkDual {
            cl: dv(PK_IDX_CL, pk.cl()),
            v: dv(PK_IDX_V, pk.v()),
            q: dv(PK_IDX_Q, pk.q()),
            v2: dv(PK_IDX_V2, pk.v2()),
            ka: dv(PK_IDX_KA, pk.ka()),
            q3: dv(PK_IDX_Q3, pk.q3()),
            v3: dv(PK_IDX_V3, pk.v3()),
            f: dv(PK_IDX_F, pk.f_bio()),
        }
    };

    // Per-event params: doses / observations / EVID=2 breakpoints each at their own
    // covariate snapshot (the `*_cov` accessors fall back to the static map when a
    // particular event carries no snapshot).
    let pk_at_dose: Vec<PkDual<Dual2<M>>> = (0..subject.doses.len())
        .map(|k| mk(subject.dose_cov(k)))
        .collect();
    let pk_at_obs: Vec<PkDual<Dual2<M>>> = (0..subject.obs_times.len())
        .map(|j| mk(subject.obs_cov(j)))
        .collect();
    let pk_at_pk_only: Vec<PkDual<Dual2<M>>> = (0..subject.pk_only_times.len())
        .map(|m| mk(subject.pk_only_cov(m)))
        .collect();

    // No lagtime in TV-cov scope → zero dose lagtimes.
    let dose_lagtimes = vec![0.0; subject.doses.len()];
    let schedule =
        EventSchedule::for_subject(subject, model.pk_model, &subject.doses, &dose_lagtimes);
    let conc = event_driven_sens_g::<Dual2<M>>(
        model.pk_model,
        subject,
        &schedule,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    let mut obs_out = Vec::with_capacity(conc.len());
    for c in &conc {
        // Clamp parity with production `conc.max(0.0)`: a negative value's
        // derivatives vanish (consistency with the OFV).
        let neg = c.value < 0.0;
        let mut df_deta = vec![0.0; n_eta];
        let mut df_dtheta = vec![0.0; n_theta];
        let mut d2f_deta2 = vec![0.0; n_eta * n_eta];
        let mut d2f_deta_dtheta = vec![0.0; n_eta * n_theta];
        if !neg {
            for k in 0..n_eta {
                df_deta[k] = c.grad[n_theta + k];
                for l in 0..n_eta {
                    d2f_deta2[k * n_eta + l] = c.hess[n_theta + k][n_theta + l];
                }
                for m in 0..n_theta {
                    d2f_deta_dtheta[k * n_theta + m] = c.hess[n_theta + k][m];
                }
            }
            for m in 0..n_theta {
                df_dtheta[m] = c.grad[m];
            }
        }
        obs_out.push(ObsSens {
            f: if neg { 0.0 } else { c.value },
            df_deta,
            d2f_deta2,
            df_dtheta,
            d2f_deta_dtheta,
        });
    }
    Some(SubjectSens { obs: obs_out })
}

/// Light (`Dual1`, `N = n_eta`) inner η-gradient for time-varying-covariate /
/// oral-infusion subjects — the first-order mirror of [`subject_sensitivities_tvcov`]
/// (`run_obs_grad_tvcov` is to `run_obs_tvcov` as `run_obs_grad` is to `run_obs`).
/// The outer θ/Ω/σ gradient already serves these via the `Dual2` event-driven walk;
/// this gives the inner EBE loop the matching exact η-gradient instead of FD,
/// closing the inner half of #447. Same gate as the outer TV-cov path; declines
/// (→ FD inner) the #419 `F`+rate-infusion and modeled-duration cases the walk
/// can't serve.
pub fn subject_eta_grad_tvcov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<ObsGrad>> {
    subject_eta_grad_tvcov_with_schedule(model, subject, theta, eta, None)
}

/// As [`subject_eta_grad_tvcov`], but reusing a per-subject `EventSchedule` the inner
/// optimizer cached once (η-invariant) instead of rebuilding it every inner BFGS step
/// (#449 re-review #6). `None` rebuilds locally (identical result).
pub(crate) fn subject_eta_grad_tvcov_with_schedule(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    cached_schedule: Option<&crate::pk::event_driven::EventSchedule>,
) -> Option<Vec<ObsGrad>> {
    if !tvcov_analytical_supported(model)
        || !(subject.has_tv_covariates() || subject_has_oral_infusion(model, subject))
    {
        return None;
    }
    if subject.has_resets() && subject.doses.iter().any(|d| d.ss) {
        return None;
    }
    if !subject.all_doses_fixed() {
        return None;
    }
    if model.has_bioavailability() && subject.has_rate_defined_infusion() {
        return None;
    }

    let n_eta = model.n_eta;
    let prog = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .expect("tvcov_analytical_supported guarantees the program");
    let slots = prog.pk_slots_ref();
    let mut slot_row: [Option<usize>; N_PK] = [None; N_PK];
    for (i, &s) in slots.iter().enumerate() {
        if s < N_PK {
            slot_row[s] = Some(i);
        }
    }

    macro_rules! disp {
        ($($n:literal),+) => {
            match n_eta {
                $($n => run_obs_grad_tvcov::<$n>(model, subject, theta, eta, prog, &slot_row, n_eta, cached_schedule),)+
                _ => None,
            }
        };
    }
    // Match the outer `run_obs_tvcov` cap (`m_dim = n_theta + n_eta` over `1..=24`):
    // since `n_eta ≤ m_dim`, bounding the gate at 24 (below) makes both dispatch
    // tables resolve, so the inner and outer analytic scope stay matched rather than
    // splitting to a fixed-EBE FD inner (#449 re-review #2).
    let mut out = disp!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    )?;

    // Constant `ScalarScale` divisor: `∂(f/k)/∂η = (∂f/∂η)/k` (η-independent `k`),
    // matching the outer TV-cov path and `pk::apply_scaling`. (LTBS / ExpressionScale
    // keep the FD inner — gated upstream in `analytic_inner_grad_supported_model`.)
    if let ScalingSpec::ScalarScale(k) = model.scaling {
        if k != 1.0 {
            for o in out.iter_mut() {
                o.f /= k;
                for g in o.df_deta.iter_mut() {
                    *g /= k;
                }
            }
        }
    }
    Some(out)
}

/// Light dual-width-`N` (`= n_eta`) inner of [`subject_eta_grad_tvcov`]: per-event
/// PK-param `Dual1` seeded on η (`∂p/∂η_k` on axis `k`), the event-driven walk over
/// `Dual1<N>`, and `∂conc/∂η` read straight into `ObsGrad`.
#[allow(clippy::too_many_arguments)]
fn run_obs_grad_tvcov<const N: usize>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    prog: &crate::parser::model_parser::IndivParamProgram,
    slot_row: &[Option<usize>; N_PK],
    n_eta: usize,
    cached_schedule: Option<&crate::pk::event_driven::EventSchedule>,
) -> Option<Vec<ObsGrad>> {
    use crate::pk::event_driven::EventSchedule;
    use crate::sens::ode_provider::param_derivatives_at_cov;
    use crate::sens::propagate::{event_driven_sens_g, PkDual};

    // The dispatch sizes `N = n_eta` exactly, so the `.min(N)` clamps below are
    // no-ops — flat `0..n_eta` loops (#449 re-review #5, mirroring #15).
    debug_assert_eq!(N, n_eta);

    let mk = |cov: &std::collections::HashMap<String, f64>| -> Option<PkDual<Dual1<N>>> {
        // `None` above the param-derivative dispatch cap (n_axes > 16): decline so
        // the inner loop falls back to FD rather than panicking (#449 review #1).
        let pd = param_derivatives_at_cov(prog, model, cov, theta, eta)?;
        let pk = (model.pk_param_fn)(theta, eta, cov);
        let seed_row = |i: usize, val: f64| -> Dual1<N> {
            let mut grad = [0.0; N];
            for k in 0..n_eta {
                grad[k] = pd.dp_deta[i][k];
            }
            Dual1 { value: val, grad }
        };
        let dv = |slot: usize, val: f64| -> Dual1<N> {
            match slot_row[slot] {
                Some(i) => seed_row(i, val),
                None => Dual1::<N>::constant(val),
            }
        };
        Some(PkDual {
            cl: dv(PK_IDX_CL, pk.cl()),
            v: dv(PK_IDX_V, pk.v()),
            q: dv(PK_IDX_Q, pk.q()),
            v2: dv(PK_IDX_V2, pk.v2()),
            ka: dv(PK_IDX_KA, pk.ka()),
            q3: dv(PK_IDX_Q3, pk.q3()),
            v3: dv(PK_IDX_V3, pk.v3()),
            f: dv(PK_IDX_F, pk.f_bio()),
        })
    };

    let pk_at_dose: Vec<PkDual<Dual1<N>>> = (0..subject.doses.len())
        .map(|k| mk(subject.dose_cov(k)))
        .collect::<Option<Vec<_>>>()?;
    let pk_at_obs: Vec<PkDual<Dual1<N>>> = (0..subject.obs_times.len())
        .map(|j| mk(subject.obs_cov(j)))
        .collect::<Option<Vec<_>>>()?;
    let pk_at_pk_only: Vec<PkDual<Dual1<N>>> = (0..subject.pk_only_times.len())
        .map(|m| mk(subject.pk_only_cov(m)))
        .collect::<Option<Vec<_>>>()?;

    // The event schedule is invariant across inner BFGS steps (it depends only on the
    // subject + doses + zero lagtimes, not on η). Reuse the schedule the inner
    // optimizer cached once per subject when available, instead of rebuilding it every
    // gradient step; fall back to building it locally otherwise (#449 re-review #6).
    let owned_schedule;
    let schedule: &EventSchedule = match cached_schedule {
        Some(s) => s,
        None => {
            let dose_lagtimes = vec![0.0; subject.doses.len()];
            owned_schedule =
                EventSchedule::for_subject(subject, model.pk_model, &subject.doses, &dose_lagtimes);
            &owned_schedule
        }
    };
    let conc = event_driven_sens_g::<Dual1<N>>(
        model.pk_model,
        subject,
        schedule,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    let mut out = Vec::with_capacity(conc.len());
    for c in &conc {
        let neg = c.value < 0.0;
        let mut df_deta = vec![0.0; n_eta];
        if !neg {
            for k in 0..n_eta {
                df_deta[k] = c.grad[k];
            }
        }
        out.push(ObsGrad {
            f: if neg { 0.0 } else { c.value },
            df_deta,
        });
    }
    Some(out)
}

/// Accumulated `subject_sensitivities` call count and wall-time across a fit,
/// for `FERX_PROFILE=1` (printed by the CLI via [`profile_report`]). No overhead
/// when off (the `Instant` is only taken when profiling is enabled).
pub static PROFILE_SENS_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROFILE_SENS_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn sens_profile_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("FERX_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Print the accumulated analytic-provider profile (no-op unless `FERX_PROFILE=1`).
pub fn profile_report() {
    if !sens_profile_enabled() {
        return;
    }
    let c = PROFILE_SENS_CALLS.load(std::sync::atomic::Ordering::Relaxed);
    let n = PROFILE_SENS_NANOS.load(std::sync::atomic::Ordering::Relaxed);
    if c > 0 {
        eprintln!(
            "[profile] analytic provider (subject_sensitivities): {} calls, {:.3}s total, {:.1} ns/call",
            c,
            n as f64 / 1e9,
            n as f64 / c as f64
        );
    }
    let ec = PROFILE_ETA_CALLS.load(std::sync::atomic::Ordering::Relaxed);
    let en = PROFILE_ETA_NANOS.load(std::sync::atomic::Ordering::Relaxed);
    if ec > 0 {
        eprintln!(
            "[profile] light η provider (subject_eta_grad): {} calls, {:.3}s total, {:.1} ns/call",
            ec,
            en as f64 / 1e9,
            en as f64 / ec as f64
        );
    }
}

/// Value and first-order η-gradient of one observation — the light-provider
/// counterpart of [`ObsSens`] (no `∂²f/∂η²`, no θ-block).
#[derive(Debug, Clone)]
pub struct ObsGrad {
    pub f: f64,
    /// `∂f/∂η_k`, length `n_eta`.
    pub df_deta: Vec<f64>,
}

/// Accumulated [`subject_eta_grad`] call count and wall-time (`FERX_PROFILE=1`).
pub static PROFILE_ETA_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROFILE_ETA_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// **Light first-order provider** for the inner EBE loop: per observation, the
/// value `f` and `∂f/∂η` only — never the second-order `∂²f/∂η²` or the θ-block.
/// Seeds the PK parameters as [`Dual1<N>`] (gradient, no Hessian) through the same
/// generic closed-form PK solution the full [`subject_sensitivities`] uses, then
/// applies the closed-form log-normal η chain `∂p_i/∂η_k = pk_i·sel[i,k]`. Roughly
/// halves the per-op cost of the inner gradient (the hot path: millions of calls).
///
/// Scope is a strict subset of [`subject_sensitivities`]: it additionally requires
/// every η to be log-normal (so the closed-form `pk·sel` chain is exact). Anything
/// else returns `None`, and the caller falls back to the full provider / FD.
pub fn subject_eta_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<ObsGrad>> {
    subject_eta_grad_with_schedule(model, subject, theta, eta, None)
}

/// As [`subject_eta_grad`], but threading a per-subject cached `EventSchedule` (built
/// once by the inner optimizer) into the TV-cov walk so it isn't rebuilt every inner
/// BFGS step (#449 re-review #6). `None` (and every non-TV-cov / ODE route) rebuilds
/// locally — identical result.
pub(crate) fn subject_eta_grad_with_schedule(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    cached_schedule: Option<&crate::pk::event_driven::EventSchedule>,
) -> Option<Vec<ObsGrad>> {
    if !sens_profile_enabled() {
        return subject_eta_grad_impl(model, subject, theta, eta, cached_schedule);
    }
    let t0 = std::time::Instant::now();
    let r = subject_eta_grad_impl(model, subject, theta, eta, cached_schedule);
    PROFILE_ETA_NANOS.fetch_add(
        t0.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    PROFILE_ETA_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    r
}

fn subject_eta_grad_impl(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    cached_schedule: Option<&crate::pk::event_driven::EventSchedule>,
) -> Option<Vec<ObsGrad>> {
    // ODE models: the light `Dual1` inner η-gradient (#410), gated by the master
    // switch. Out-of-scope ODE subjects decline (→ FD inner), the same per-subject
    // scope the outer provider uses, so inner and outer stay on the same route.
    if model.ode_spec.is_some() {
        if ODE_SENS_ENABLED {
            return crate::sens::ode_provider::ode_subject_eta_grad(model, subject, theta, eta);
        }
        return None;
    }
    // TV-cov / oral infusion: the light event-driven walk (#447). The outer gradient
    // already serves these via `subject_sensitivities_tvcov`; route the inner to the
    // matching `Dual1` walk (out-of-scope cases return `None` → FD per point).
    if subject.has_tv_covariates() || subject_has_oral_infusion(model, subject) {
        return subject_eta_grad_tvcov_with_schedule(model, subject, theta, eta, cached_schedule);
    }
    // Same model/subject scope as the full provider …
    if !analytical_supported(model) {
        return None;
    }
    if subject.has_resets() && subject.doses.iter().any(|d| d.ss) {
        return None;
    }
    // Modeled-duration doses (`RATE=-2` → `D{cmt}`) resolve `rate`/`duration` from
    // the PK params in the prediction path; the provider iterates `subject.doses`
    // directly, so the unresolved dose would be a bolus/zero-input surrogate. Route
    // these to FD until the resolved-duration sensitivity lands.
    if !subject.all_doses_fixed() {
        return None;
    }
    // #419: a rate-defined infusion under `F ≠ 1` reshapes (rate held, window
    // `F·dur`); the superposition kernels here apply `F` as a magnitude scale, so
    // decline to the FD inner Jacobian (matches the full provider's #419 gate).
    if model.has_bioavailability() && subject.has_rate_defined_infusion() {
        return None;
    }
    // (Oral infusion is handled by the TV-cov event-driven walk above, #447.)
    // `ExpressionScale` obs_scale is served below via the η-only quotient rule
    // (`apply_expression_scale_inner`) — the light counterpart of the outer
    // `apply_expression_scale`. `scaling_supported` (checked inside
    // `analytical_supported`) already bounded the scale program to the dispatch table.
    //
    // EXCEPT LTBS + η-dependent `ExpressionScale`: route that combination to the FD inner
    // Jacobian (#534 review #5). The EBE itself already converges on FD for any LTBS model
    // (`analytic_inner_common_bail` includes `log_transform`); `find_ebe` then builds the
    // covariance H-matrix from this provider's Jacobian (`subject_eta_jacobian`, gated only
    // on `Some`/`None`). Returning the analytic scale+log Jacobian here would pair it with
    // that FD-converged EBE — and under the `g = ln(f)` wrap the closed-form EBE's ~1e-9
    // offset is exactly what corrupts the covariance Hessian (the reason LTBS reverts the
    // inner gradient at all). Declining restores the pre-#486 behaviour for this combo and
    // matches the ODE path's `!log_transform` gate for `ExpressionScale`. (Plain LTBS and
    // plain `ExpressionScale` are unaffected; the analytic *outer* gradient still serves
    // LTBS + `ExpressionScale`.)
    if model.log_transform && matches!(model.scaling, ScalingSpec::ExpressionScale { .. }) {
        return None;
    }

    let n_eta = model.n_eta;
    let oral = matches!(
        model.pk_model,
        PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral
    );
    let two_cpt = matches!(model.pk_model, PkModel::TwoCptIv | PkModel::TwoCptOral);
    let three_cpt = matches!(model.pk_model, PkModel::ThreeCptIv | PkModel::ThreeCptOral);

    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates);

    // First-order `∂p_i/∂η_k`, exact for ANY parameterization — the η-block of the
    // same `pd` the full provider builds. The compiled `[individual_parameters]`
    // program path (`param_derivatives_from_prog`, evaluated once per subject)
    // covers log-normal, logit-normal F, additive, … ; its `slots` follow the
    // program order. Falls back to the closed-form log-normal `pk·sel` chain (rows
    // in `pk_indices` order) when the program path is unavailable, and to `None`
    // (→ caller's per-point FD) only when neither applies — so the light provider
    // serves exactly the analytical scope the full `subject_sensitivities` does.
    let (dp_deta, slots): (Vec<Vec<f64>>, Vec<usize>) = match model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .filter(|prog| prog_covers_required_pk_slots(model, prog))
        .and_then(|prog| {
            // Light `Dual1<n_eta>` η-gradient: the inner EBE loop consumes only
            // `∂p/∂η`, so seeding η alone avoids the θ-axes and second-order
            // Hessian the full `Dual2` `param_derivatives_from_prog` carries (#485
            // follow-up).
            crate::sens::ode_provider::param_eta_derivatives_from_prog(
                prog, model, subject, theta, eta,
            )
            .map(|dp_deta| (dp_deta, prog.pk_slots()))
        }) {
        Some(v) => v,
        None => {
            if !model
                .eta_param_info
                .iter()
                .all(|e| e.param_type == crate::types::EtaParamType::LogNormal)
            {
                return None;
            }
            let pd = lognormal_param_derivatives(model, subject, theta, &pk);
            (pd.dp_deta, model.pk_indices.clone())
        }
    };
    let mut seed_dim: [Option<usize>; N_PK] = [None; N_PK];
    for (i, &slot) in slots.iter().enumerate() {
        if slot < N_PK {
            seed_dim[slot] = Some(i);
        }
    }

    macro_rules! disp {
        ($($n:literal),+) => {
            match slots.len() {
                $($n => Some(run_obs_grad::<$n>(
                    &seed_dim, &pk, oral, two_cpt, three_cpt, subject, &dp_deta, n_eta,
                )),)+
                _ => None,
            }
        };
    }
    let mut out = disp!(1, 2, 3, 4, 5, 6, 7, 8, 9)?;
    // Analytic `[initial_conditions]` impulse (#524): η-gradient only, layered on
    // BEFORE scaling — same insertion order as the f64 `pk::add_analytical_init`.
    // The inner path runs on `Dual1<n_eta>`, so it dispatches on `n_eta` (≤ the
    // `n_theta + n_eta` that `init_supported` bounds to the table), so the `_` arm is
    // unreachable for a supported init model.
    if !model.analytical_init.is_empty() {
        dispatch_init_impulse!(
            n_eta,
            apply_analytical_init_inner,
            &mut out,
            model,
            subject,
            &pk,
            &dp_deta,
            &slots,
            theta,
            eta,
            &subject.covariates,
            n_eta
        );
    }
    // Constant `ScalarScale` output divisor: `f_scaled = f/k`, and `∂f/∂η` is
    // linear in `f` so it divides by the same `k` (η-independent). Matches
    // `pk::apply_scaling` (`pred /= s`). Other scaling variants are gated out.
    if let ScalingSpec::ScalarScale(k) = model.scaling {
        if k != 1.0 {
            for o in out.iter_mut() {
                o.f /= k;
                for g in o.df_deta.iter_mut() {
                    *g /= k;
                }
            }
        }
    }
    // η-dependent `ExpressionScale` output divisor `s(θ, η)`: `f_scaled = f/s`, with the
    // η-only quotient rule `∂(f/s)/∂η_k = (∂f/∂η_k)/s − f·(∂s/∂η_k)/s²`. The light
    // counterpart of the outer `apply_expression_scale` (which also carries the θ and
    // second-order blocks the outer gradient needs). `s` is subject-static, evaluated
    // once over `Dual1<n_eta>`; dispatches on `n_eta` (≤ the `n_theta + n_eta` that
    // `scaling_supported` bounds to the table), so the `_` arm is unreachable for a
    // supported scale. Applied after the init impulse / `ScalarScale` and before LTBS,
    // matching `pk::apply_scaling`'s `pred /= s` order.
    if let ScalingSpec::ExpressionScale {
        deriv: Some(prog), ..
    } = &model.scaling
    {
        apply_expression_scale_inner_dispatch(
            &mut out,
            prog,
            &pk,
            &dp_deta,
            &slots,
            theta,
            eta,
            &subject.covariates,
            n_eta,
        );
    }
    // LTBS: `g = ln(f)`, so `∂g/∂η = ∂f/∂η / f`. Applied after scaling. The value
    // half goes through the shared `pk::ltbs_log_g` — the same floor-then-log the f64
    // predictor and the ODE dual walk use — so the analytical gradient can't silently
    // drift from production either (#451 review #5). The gradient still keys on the
    // strict `> LTBS_FLOOR` boundary: below the floor the transform clamps to a
    // constant, so the gradient vanishes.
    if model.log_transform {
        for o in out.iter_mut() {
            if o.f > crate::pk::LTBS_FLOOR {
                let inv = 1.0 / o.f;
                for g in o.df_deta.iter_mut() {
                    *g *= inv;
                }
            } else {
                for g in o.df_deta.iter_mut() {
                    *g = 0.0;
                }
            }
            o.f = crate::pk::ltbs_log_g(o.f);
        }
    }
    Some(out)
}

/// Per-observation `(f, ∂f/∂η)` chain at dual width `N`, via `Dual1<N>` (grad,
/// no Hessian) — the light counterpart of [`run_obs`]. `seed_dim[s]` is the
/// compact dual axis for PK slot `s`; `dp_deta` rows are in compact-axis order.
#[allow(clippy::too_many_arguments)]
fn run_obs_grad<const N: usize>(
    seed_dim: &[Option<usize>; N_PK],
    pk: &crate::types::PkParams,
    oral: bool,
    two_cpt: bool,
    three_cpt: bool,
    subject: &Subject,
    dp_deta: &[Vec<f64>],
    n_eta: usize,
) -> Vec<ObsGrad> {
    let (cl, v1, q, v2, ka, f_bio, q3, v3) = (
        pk.cl(),
        pk.v(),
        pk.q(),
        pk.v2(),
        pk.ka(),
        pk.f_bio(),
        pk.q3(),
        pk.v3(),
    );
    let dv = |slot: usize, value: f64| -> Dual1<N> {
        match seed_dim[slot] {
            Some(k) => Dual1::<N>::var(value, k),
            None => Dual1::<N>::constant(value),
        }
    };
    let cl_d = dv(PK_IDX_CL, cl);
    let v1_d = dv(PK_IDX_V, v1);
    let q_d = dv(PK_IDX_Q, q);
    let v2_d = dv(PK_IDX_V2, v2);
    let ka_d = dv(PK_IDX_KA, ka);
    let f_d = dv(PK_IDX_F, f_bio);
    let q3_d = dv(PK_IDX_Q3, q3);
    let v3_d = dv(PK_IDX_V3, v3);
    // Lagtime enters each dose's concentration through the elapsed-time argument
    // (`elapsed = (t_obs − dose.time) − lagtime`), so seed it as its own dual axis.
    let lag_val = pk.lagtime();
    let lag_d = dv(PK_IDX_LAGTIME, lag_val);

    let mut out = Vec::with_capacity(subject.obs_times.len());
    for &t_obs in subject.obs_times.iter() {
        let reset_floor = subject
            .reset_times
            .iter()
            .copied()
            .filter(|&r| r <= t_obs)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut fd = Dual1::<N>::constant(0.0);
        for dose in &subject.doses {
            // Exclude doses washed out by a reset — keyed on the *lagged arrival*
            // `dose.time + lag`, not the record time: a dose recorded before a
            // reset but arriving after it (via lagtime) contributes to the new
            // segment, exactly as the event-driven walk applies it (PR #381 #2).
            if dose.time + lag_val < reset_floor {
                continue;
            }
            let Some(elapsed) = lagged_elapsed(dose, t_obs, lag_val, lag_d) else {
                continue;
            };
            let c = if three_cpt {
                three_cpt_conc_g(
                    dose, elapsed, cl_d, v1_d, q_d, v2_d, q3_d, v3_d, ka_d, f_d, oral,
                )
            } else if two_cpt {
                two_cpt_conc_g(dose, elapsed, cl_d, v1_d, q_d, v2_d, ka_d, f_d, oral)
            } else {
                one_cpt_conc_g(dose, elapsed, cl_d, v1_d, ka_d, f_d, oral)
            };
            fd = fd + c;
        }

        // Mirror production's `conc.max(0.0)`: a negative closed-form value clamps
        // to 0, so its η-gradient is 0 there too (consistency with the objective).
        let (fval, g) = if fd.value < 0.0 {
            (0.0, [0.0; N])
        } else {
            (fd.value, fd.grad)
        };

        let mut df_deta = vec![0.0; n_eta];
        for i in 0..N {
            let gi = g[i];
            for k in 0..n_eta {
                df_deta[k] += gi * dp_deta[i][k];
            }
        }
        out.push(ObsGrad { f: fval, df_deta });
    }
    out
}

/// Compute per-observation analytic sensitivities, or `None` if this
/// model/subject is outside the supported analytical 1-cpt scope (caller falls
/// back to the gradient-free path).
pub fn subject_sensitivities(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<SubjectSens> {
    if !sens_profile_enabled() {
        return subject_sensitivities_impl(model, subject, theta, eta);
    }
    let t0 = std::time::Instant::now();
    let r = subject_sensitivities_impl(model, subject, theta, eta);
    PROFILE_SENS_NANOS.fetch_add(
        t0.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    PROFILE_SENS_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    r
}

/// Apply an `ExpressionScale` divisor `s(θ, η)` to a subject's already-computed
/// jet in place: `scaled_f = f / s`. The scale's own value, `∂s/∂(θ,η)` and
/// Hessian come from the differentiable scale program evaluated over `Dual2<M>`
/// (`M = n_theta + n_eta`); the individual parameters it references are fed in as
/// duals built from the provider's `ParamDerivs`. The quotient rule
///   `∂(f/s)/∂x      = f_x/s − f·s_x/s²`
///   `∂²(f/s)/∂x∂y   = f_xy/s − f_x s_y/s² − f_y s_x/s² − f s_xy/s² + 2 f s_x s_y/s³`
/// is applied over the η-η and η-θ blocks (the only second-order blocks the outer
/// gradient consumes). `s` is subject-static, so it is evaluated once and reused
/// for every observation. Dual dimension `m < n_theta` is `θ_m`; `n_theta + k` is
/// `η_k`.
///
/// The first-order **η** value/gradient here (`f/s`, `f_k/s − f·s_k/s²`) is the same
/// formula [`apply_expression_scale_inner`] applies for the light inner provider; the two
/// are kept in sync by the `light_provider_expression_scale_matches_full` /
/// `ode_light_inner_eta_grad_matches_full_provider` parity tests (mirroring the
/// `apply_analytical_init` / `_inner` split, #534 review #6). Edit both together.
#[allow(clippy::too_many_arguments)]
fn apply_expression_scale<const M: usize>(
    sens: &mut SubjectSens,
    prog: &crate::parser::model_parser::ScaleDerivProgram,
    pk: &crate::types::PkParams,
    pd: &crate::sens::ode_provider::ParamDerivs,
    slots: &[usize],
    theta: &[f64],
    eta: &[f64],
    cov: &std::collections::HashMap<String, f64>,
    n_theta: usize,
    n_eta: usize,
) {
    // Build the dual for each PK slot the scale references: value + ∂p/∂(θ,η) +
    // η-η / η-θ Hessian (the quotient rule only reads those). PK params not in
    // `slots` enter as constants. Shared with the analytic init impulse (#524).
    let var_duals: Vec<Dual2<M>> = prog
        .var_to_pk_slot()
        .iter()
        .map(|&s| {
            pk_slot_dual_outer::<M>(
                s,
                pk.values.get(s).copied().unwrap_or(0.0),
                pd,
                slots,
                n_theta,
                n_eta,
            )
        })
        .collect();

    let s = prog.eval_scale_dual::<M>(theta, eta, cov, &var_duals);

    // Reuse two scratch buffers across observations rather than cloning per row — `2`
    // allocations per call instead of `2·n_obs` on subjects with many observations
    // (#534 review #2). `s` is subject-static here, so the same jet applies to every row.
    let mut fk: Vec<f64> = Vec::with_capacity(n_eta);
    let mut fm: Vec<f64> = Vec::with_capacity(n_theta);
    for o in sens.obs.iter_mut() {
        apply_scale_quotient_row::<M>(o, &s, n_theta, n_eta, &mut fk, &mut fm);
    }
}

/// Apply the `ExpressionScale` quotient `f ↦ f/s` to one [`ObsSens`] row given the
/// precomputed scale jet `s` (value + `∂s/∂(θ, axes)` + Hessian). `n_axes` is the η-axis
/// count: `n_eta` for the non-IOV path (one subject-static `s`) and `n_stacked` for the
/// IOV path (a per-occasion-group `s`); the scale's axis `k` is read at dual index
/// `n_theta + k`, its θ axis `m` at `m`. `fk`/`fm` are caller-owned scratch buffers,
/// reused across rows — the second-order update reads the *original* `∂f/∂η` / `∂f/∂θ`
/// across the whole k/l double loop, so they are snapshotted before `o` is rewritten in
/// place. Single source for the quotient rule shared by the closed-form/ODE non-IOV
/// loop and the ODE IOV per-group caller (#575 review — no second copy to keep in sync).
pub(crate) fn apply_scale_quotient_row<const M: usize>(
    o: &mut ObsSens,
    s: &Dual2<M>,
    n_theta: usize,
    n_axes: usize,
    fk: &mut Vec<f64>,
    fm: &mut Vec<f64>,
) {
    let f = o.f;
    let inv = 1.0 / s.value;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    fk.clear();
    fk.extend_from_slice(&o.df_deta); // original ∂f/∂(axes)
    fm.clear();
    fm.extend_from_slice(&o.df_dtheta); // original ∂f/∂θ
                                        // η-η Hessian.
    for k in 0..n_axes {
        for l in 0..n_axes {
            let idx = k * n_axes + l;
            let s_k = s.grad[n_theta + k];
            let s_l = s.grad[n_theta + l];
            let s_kl = s.hess[n_theta + k][n_theta + l];
            o.d2f_deta2[idx] =
                o.d2f_deta2[idx] * inv - fk[k] * s_l * inv2 - fk[l] * s_k * inv2 - f * s_kl * inv2
                    + 2.0 * f * s_k * s_l * inv3;
        }
    }
    // η-θ Hessian.
    for k in 0..n_axes {
        for m in 0..n_theta {
            let idx = k * n_theta + m;
            let s_k = s.grad[n_theta + k];
            let s_m = s.grad[m];
            let s_km = s.hess[n_theta + k][m];
            o.d2f_deta_dtheta[idx] = o.d2f_deta_dtheta[idx] * inv
                - fk[k] * s_m * inv2
                - fm[m] * s_k * inv2
                - f * s_km * inv2
                + 2.0 * f * s_k * s_m * inv3;
        }
    }
    // First derivatives and value.
    for k in 0..n_axes {
        o.df_deta[k] = fk[k] * inv - f * s.grad[n_theta + k] * inv2;
    }
    for m in 0..n_theta {
        o.df_dtheta[m] = fm[m] * inv - f * s.grad[m] * inv2;
    }
    o.f = f * inv;
}

/// Apply an `ExpressionScale` divisor to a subject's `(θ, η)`-space [`SubjectSens`],
/// monomorphising [`apply_expression_scale`] on the scale program's axis count
/// `prog.n_axes()` (`= n_theta + n_eta`) over the shared `1..=MAX_SCALE_AXES` dispatch
/// table ([`dispatch_init_impulse!`], so it stays coupled to the axis cap and can't
/// silently drop the scale through a `_` arm if the cap is later widened — #534 review
/// #4). Shared by the closed-form provider ([`subject_sensitivities`]) and the ODE
/// provider ([`crate::sens::ode_provider::ode_subject_sensitivities`], #486) — both
/// produce the same `SubjectSens` shape, so the quotient-rule application is
/// provider-agnostic. `slots`/`pd` must be the paired `(prog.pk_slots(),
/// param_derivatives_from_prog(prog))` the caller already built for the η/θ chain. A
/// scale wider than the table is a no-op (`scaling_supported` bounds `n_axes ≤
/// MAX_SCALE_AXES`, so unreachable in scope).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_expression_scale_outer(
    sens: &mut SubjectSens,
    prog: &crate::parser::model_parser::ScaleDerivProgram,
    pk: &crate::types::PkParams,
    pd: &crate::sens::ode_provider::ParamDerivs,
    slots: &[usize],
    theta: &[f64],
    eta: &[f64],
    cov: &std::collections::HashMap<String, f64>,
    n_theta: usize,
    n_eta: usize,
) {
    dispatch_init_impulse!(
        prog.n_axes(),
        apply_expression_scale,
        sens,
        prog,
        pk,
        pd,
        slots,
        theta,
        eta,
        cov,
        n_theta,
        n_eta
    );
}

/// Build the `Dual2<M>` for PK slot `slot` in `(θ, η)`-axis layout (axes
/// `0..n_theta` are θ, `n_theta + k` is `η_k`), carrying value + `∂p/∂(θ,η)` +
/// the η-η / η-θ Hessian blocks from the outer `ParamDerivs`. A slot the program
/// references but the provider does not differentiate (`slots` miss) enters as a
/// constant. Shared by the `ExpressionScale` quotient path and the analytic init
/// impulse (#524) so both seed PK params identically.
fn pk_slot_dual_outer<const M: usize>(
    slot: usize,
    value: f64,
    pd: &crate::sens::ode_provider::ParamDerivs,
    slots: &[usize],
    n_theta: usize,
    n_eta: usize,
) -> Dual2<M> {
    match slots.iter().position(|&x| x == slot) {
        Some(j) => {
            let mut grad = [0.0; M];
            let mut hess = [[0.0; M]; M];
            for m in 0..n_theta.min(M) {
                grad[m] = pd.dp_dtheta[j][m];
            }
            for k in 0..n_eta {
                if n_theta + k < M {
                    grad[n_theta + k] = pd.dp_deta[j][k];
                }
            }
            for k in 0..n_eta {
                for l in 0..n_eta {
                    if n_theta + k < M && n_theta + l < M {
                        hess[n_theta + k][n_theta + l] = pd.d2p_deta2[j][k][l];
                    }
                }
                for m in 0..n_theta {
                    if n_theta + k < M && m < M {
                        let v = pd.d2p_detadtheta[j][k][m];
                        hess[n_theta + k][m] = v;
                        hess[m][n_theta + k] = v;
                    }
                }
            }
            Dual2 { value, grad, hess }
        }
        None => Dual2::constant(value),
    }
}

/// First-order counterpart of [`pk_slot_dual_outer`] for the inner η-gradient:
/// a `Dual1<N>` (`N = n_eta`) seeded with `∂p/∂η_k` on axis `k` (from `dp_deta`).
/// The inner EBE loop consumes only `∂f/∂η`, so this drops the θ axes and the
/// `Dual2` Hessian the outer path carries — the η axes alone (no `n_theta` offset).
/// A slot the program references but the provider does not differentiate
/// (`slots` miss) enters as a constant.
fn pk_slot_dual_inner<const N: usize>(
    slot: usize,
    value: f64,
    dp_deta: &[Vec<f64>],
    slots: &[usize],
    n_eta: usize,
) -> Dual1<N> {
    match slots.iter().position(|&x| x == slot) {
        Some(j) => {
            let mut grad = [0.0; N];
            for k in 0..n_eta.min(N) {
                grad[k] = dp_deta[j][k];
            }
            Dual1 { value, grad }
        }
        None => Dual1::constant(value),
    }
}

/// The seven PK-param duals the init impulse reads, built from a per-slot dual
/// constructor. Order matches [`crate::pk::analytical_init_concentration_g`].
/// Generic over the seeded number type `T` so the outer path runs it as
/// `Dual2<M>` (full `(θ,η)` jet + Hessian) and the inner path as `Dual1<n_eta>`
/// (η-gradient only, no θ axes, no Hessian).
struct InitPkDuals<T: crate::sens::num::PkNum> {
    cl: T,
    v: T,
    q: T,
    v2: T,
    ka: T,
    q3: T,
    v3: T,
}

/// Add the analytic `[initial_conditions]` impulse to an already-computed jet
/// (#524). The init contributes `A₀ · kernel(t, pk)` to the prediction *before*
/// scaling — the same insertion point as the f64 `pk::add_analytical_init` — so
/// this runs after `run_obs`/`run_obs_grad` and before the scale/LTBS transforms.
///
/// `a0_dual(prog)` evaluates the baseline amount `A₀` and its jet from the init
/// program; `pk` supplies the seven PK-param duals. The impulse jet comes from
/// [`crate::pk::analytical_init_concentration_g`] over the seeded type `T`, so
/// `∂C/∂A₀` and `∂C/∂(CL,V,…)` are exact. `add(j, &c)` folds observation `j`'s
/// impulse into the caller's jet (full `Dual2` Hessian for the outer path, the
/// `Dual1` η-gradient for the inner). A reset (EVID=3/4) wipes the baseline, so
/// observations at/after the first reset get none — matching `add_analytical_init_with`.
fn add_init_impulse<T, FA, FAdd>(
    model: &CompiledModel,
    subject: &Subject,
    pk: &InitPkDuals<T>,
    mut a0_dual: FA,
    mut add: FAdd,
) where
    T: crate::sens::num::PkNum,
    FA: FnMut(&crate::parser::model_parser::ScaleDerivProgram) -> T,
    FAdd: FnMut(usize, &T),
{
    let first_reset = subject
        .reset_times
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    for init in &model.analytical_init {
        let Some(prog) = &init.amount_deriv else {
            continue;
        };
        let a0 = a0_dual(prog);
        // Skip only a non-finite amount. Unlike the f64 value path
        // (`add_analytical_init_with`), which drops `A₀ == 0` because it adds zero
        // concentration, the gradient must NOT be dropped at `A₀ == 0`: the impulse
        // is `C = A₀·kernel(t, pk)`, so `∂C/∂(θ,η) = kernel·∂A₀/∂(θ,η)` is nonzero
        // wherever the amount has nonzero parameter sensitivity (e.g. an additive
        // form passing through zero). Skipping there would zero a gradient component
        // an FD of the objective still sees. With `A₀.val() == 0` the kernel jet
        // folds value 0 and the correct `kernel·∂A₀/∂(θ,η)` gradient.
        if !a0.val().is_finite() {
            continue;
        }
        for (j, &t) in subject.obs_times.iter().enumerate() {
            if t < 0.0 || t >= first_reset {
                continue;
            }
            let c = crate::pk::analytical_init_concentration_g::<T>(
                model.pk_model,
                init.cmt,
                a0,
                T::from_f64(t),
                pk.cl,
                pk.v,
                pk.q,
                pk.v2,
                pk.ka,
                pk.q3,
                pk.v3,
            );
            add(j, &c);
        }
    }
}

/// Outer (Dual2) analytic init impulse: builds PK-param + amount duals from the
/// full `ParamDerivs` and folds value + `∂/∂(θ,η)` + η-η / η-θ Hessian into `sens`.
#[allow(clippy::too_many_arguments)]
fn apply_analytical_init_outer<const M: usize>(
    sens: &mut SubjectSens,
    model: &CompiledModel,
    subject: &Subject,
    pk: &crate::types::PkParams,
    pd: &crate::sens::ode_provider::ParamDerivs,
    slots: &[usize],
    theta: &[f64],
    eta: &[f64],
    cov: &std::collections::HashMap<String, f64>,
    n_theta: usize,
    n_eta: usize,
) {
    let dual =
        |slot: usize, val: f64| pk_slot_dual_outer::<M>(slot, val, pd, slots, n_theta, n_eta);
    let pkd = InitPkDuals {
        cl: dual(PK_IDX_CL, pk.cl()),
        v: dual(PK_IDX_V, pk.v()),
        q: dual(PK_IDX_Q, pk.q()),
        v2: dual(PK_IDX_V2, pk.v2()),
        ka: dual(PK_IDX_KA, pk.ka()),
        q3: dual(PK_IDX_Q3, pk.q3()),
        v3: dual(PK_IDX_V3, pk.v3()),
    };
    add_init_impulse::<Dual2<M>, _, _>(
        model,
        subject,
        &pkd,
        |prog| {
            let var_duals: Vec<Dual2<M>> = prog
                .var_to_pk_slot()
                .iter()
                .map(|&s| dual(s, pk.values.get(s).copied().unwrap_or(0.0)))
                .collect();
            prog.eval_scale_dual::<M>(theta, eta, cov, &var_duals)
        },
        |j, c| {
            if let Some(o) = sens.obs.get_mut(j) {
                o.f += c.value;
                for k in 0..n_eta {
                    o.df_deta[k] += c.grad[n_theta + k];
                }
                for m in 0..n_theta {
                    o.df_dtheta[m] += c.grad[m];
                }
                for k in 0..n_eta {
                    for l in 0..n_eta {
                        o.d2f_deta2[k * n_eta + l] += c.hess[n_theta + k][n_theta + l];
                    }
                }
                for k in 0..n_eta {
                    for m in 0..n_theta {
                        o.d2f_deta_dtheta[k * n_theta + m] += c.hess[n_theta + k][m];
                    }
                }
            }
        },
    );
}

/// Inner analytic init impulse: η-gradient only, folded into `out`. Runs on
/// `Dual1<N>` (`N = n_eta`) via [`pk_slot_dual_inner`] and
/// [`ScaleDerivProgram::eval_scale_dual1`] — no θ axes, no Hessian, so the dual
/// width is `n_eta` rather than the outer path's `n_theta + n_eta`, and `c.grad[k]`
/// is `∂C/∂η_k` directly (no `n_theta` offset).
#[allow(clippy::too_many_arguments)]
fn apply_analytical_init_inner<const N: usize>(
    out: &mut [ObsGrad],
    model: &CompiledModel,
    subject: &Subject,
    pk: &crate::types::PkParams,
    dp_deta: &[Vec<f64>],
    slots: &[usize],
    theta: &[f64],
    eta: &[f64],
    cov: &std::collections::HashMap<String, f64>,
    n_eta: usize,
) {
    let dual = |slot: usize, val: f64| pk_slot_dual_inner::<N>(slot, val, dp_deta, slots, n_eta);
    let pkd = InitPkDuals {
        cl: dual(PK_IDX_CL, pk.cl()),
        v: dual(PK_IDX_V, pk.v()),
        q: dual(PK_IDX_Q, pk.q()),
        v2: dual(PK_IDX_V2, pk.v2()),
        ka: dual(PK_IDX_KA, pk.ka()),
        q3: dual(PK_IDX_Q3, pk.q3()),
        v3: dual(PK_IDX_V3, pk.v3()),
    };
    add_init_impulse::<Dual1<N>, _, _>(
        model,
        subject,
        &pkd,
        |prog| {
            let var_duals: Vec<Dual1<N>> = prog
                .var_to_pk_slot()
                .iter()
                .map(|&s| dual(s, pk.values.get(s).copied().unwrap_or(0.0)))
                .collect();
            prog.eval_scale_dual1::<N>(theta, eta, cov, &var_duals)
        },
        |j, c| {
            if let Some(o) = out.get_mut(j) {
                o.f += c.value;
                for k in 0..n_eta.min(N) {
                    o.df_deta[k] += c.grad[k];
                }
            }
        },
    );
}

/// Inner (`Dual1<N>`, `N = n_eta`) `ExpressionScale` divisor: scale each observation
/// by `1/s(θ, η)` and apply the η-only quotient rule
///   `∂(f/s)/∂η_k = (∂f/∂η_k)/s − f·(∂s/∂η_k)/s²`.
/// The η-block counterpart of the outer [`apply_expression_scale`] (which also carries
/// the θ and the η-η / η-θ second-order blocks the outer gradient consumes). The inner
/// EBE loop reads only `(f, ∂f/∂η)`, so this drops the θ axes and the `Dual2` Hessian:
/// the individual-parameter vars enter as `Dual1<N>` built from `dp_deta` (η-gradient
/// only, via [`pk_slot_dual_inner`]) and the scale is evaluated once over
/// [`ScaleDerivProgram::eval_scale_dual1`]. `s` is subject-static, so it is evaluated
/// once and reused for every observation, matching the outer path.
///
/// This is the same first-order η quotient [`apply_expression_scale`] applies for the
/// `Dual2` outer provider; the two hand-written copies are kept in sync by the
/// `light_provider_expression_scale_matches_full` /
/// `ode_light_inner_eta_grad_matches_full_provider` parity tests (#534 review #6). Edit
/// both together.
#[allow(clippy::too_many_arguments)]
fn apply_expression_scale_inner<const N: usize>(
    out: &mut [ObsGrad],
    prog: &crate::parser::model_parser::ScaleDerivProgram,
    pk: &crate::types::PkParams,
    dp_deta: &[Vec<f64>],
    slots: &[usize],
    theta: &[f64],
    eta: &[f64],
    cov: &std::collections::HashMap<String, f64>,
    n_eta: usize,
) {
    // One `Dual1<N>` per individual-parameter var the scale program can reference
    // (value + ∂p/∂η; a slot the program references but the provider does not
    // differentiate enters constant — the same per-slot constructor the inner init
    // impulse uses). The count is `var_to_pk_slot().len()` = the number of
    // `[individual_parameters]` vars, which is NOT axis-bounded (a model may declare
    // many intermediates). Use a stack buffer on the common `<= MAX_SCALE_AXES` path
    // and fall back to a heap `Vec` for models with more individual parameters
    // (#534 review #3; the fixed-size buffer alone would panic on the `[..nvar]` slice).
    let var_slots = prog.var_to_pk_slot();
    let nvar = var_slots.len();
    let mk = |s: usize| {
        pk_slot_dual_inner::<N>(
            s,
            pk.values.get(s).copied().unwrap_or(0.0),
            dp_deta,
            slots,
            n_eta,
        )
    };
    let mut buf = [Dual1::<N>::constant(0.0); MAX_SCALE_AXES];
    let heap: Vec<Dual1<N>>;
    let var_duals: &[Dual1<N>] = if nvar <= MAX_SCALE_AXES {
        for (d, &s) in buf.iter_mut().zip(var_slots.iter()) {
            *d = mk(s);
        }
        &buf[..nvar]
    } else {
        heap = var_slots.iter().map(|&s| mk(s)).collect();
        &heap
    };
    let s = prog.eval_scale_dual1::<N>(theta, eta, cov, var_duals);
    let inv = 1.0 / s.value;
    let inv2 = inv * inv;
    for o in out.iter_mut() {
        let f = o.f;
        for k in 0..n_eta.min(N) {
            o.df_deta[k] = o.df_deta[k] * inv - f * s.grad[k] * inv2;
        }
        o.f = f * inv;
    }
}

/// Apply an `ExpressionScale` divisor to a subject's inner η-gradient (`Vec<ObsGrad>`),
/// monomorphising [`apply_expression_scale_inner`] on `n_eta` over the
/// `1..=MAX_SCALE_AXES` dispatch table. The inner counterpart of
/// [`apply_expression_scale_outer`], shared by the closed-form inner provider
/// ([`subject_eta_grad`]) and the ODE inner provider
/// ([`crate::sens::ode_provider::ode_subject_eta_grad`], #486). `slots`/`dp_deta` are
/// the paired `(prog.pk_slots(), param_eta_derivatives_from_prog(prog))` η-block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_expression_scale_inner_dispatch(
    out: &mut [ObsGrad],
    prog: &crate::parser::model_parser::ScaleDerivProgram,
    pk: &crate::types::PkParams,
    dp_deta: &[Vec<f64>],
    slots: &[usize],
    theta: &[f64],
    eta: &[f64],
    cov: &std::collections::HashMap<String, f64>,
    n_eta: usize,
) {
    dispatch_init_impulse!(
        n_eta,
        apply_expression_scale_inner,
        out,
        prog,
        pk,
        dp_deta,
        slots,
        theta,
        eta,
        cov,
        n_eta
    );
}

/// True when an **oral** model carries an **infusion** dose (RATE>0 into the
/// depot/central, or a modeled-duration `D{cmt}`). Such subjects switch the
/// compartment dynamics that dose superposition can't express — the depot
/// zero-order forced response (#400) and the depot-bypass central infusion (#350)
/// — so the full provider routes them through the state-propagating `Dual2` walk
/// (whose oral propagators now carry `rate_central`/`rate_depot`), exactly as
/// production's `compute_predictions` routes oral-depot infusion to the
/// event-driven path. The light first-order inner provider has no walk, so it
/// still falls back to the FD inner Jacobian for these subjects.
pub(crate) fn subject_has_oral_infusion(model: &CompiledModel, subject: &Subject) -> bool {
    matches!(
        model.pk_model,
        PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral
    ) && subject.doses.iter().any(|d| d.is_infusion())
}

fn subject_sensitivities_impl(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<SubjectSens> {
    // ODE models route to the ODE sensitivity provider (issue #367, Option A;
    // armed in #410) when in its supported scope; out-of-scope ODE subjects return
    // `None` and fall back to the prior path (gradient-free outer, FD inner). The
    // `ODE_SENS_ENABLED` master switch stays as a single kill-switch for the path.
    if model.ode_spec.is_some() {
        if ODE_SENS_ENABLED {
            return crate::sens::ode_provider::ode_subject_sensitivities(
                model, subject, theta, eta,
            );
        }
        return None;
    }
    if !analytical_supported(model) {
        return None;
    }
    // #419: a rate-defined infusion (`RATE>0`, `RATE=-1`) under bioavailability
    // `F ≠ 1` *reshapes* the infusion — the rate is held and the window is scaled
    // to `F·dur` — rather than scaling its magnitude. The analytic paths apply `F`
    // only as a magnitude scale (the superposition kernels via `route_f_scale`, the
    // Dual2 walk via an inline `pk.f * rate`), which is exact only when the
    // concentration is linear in `F`; neither can represent the reshaped window.
    // Route such subjects to the FD gradient, whose `event_driven_predictions`
    // already applies the #419 rule. (A duration-defined `RATE=-2` infusion is
    // unaffected: `F` scales its rate, a magnitude both mechanisms handle.)
    if model.has_bioavailability() && subject.has_rate_defined_infusion() {
        return None;
    }
    // Modeled-duration doses (`RATE=-2` → `D{cmt}`) resolve `rate`/`duration` from
    // the PK params in the prediction path; the provider reads `subject.doses`
    // directly, so the unresolved dose would be a bolus/zero-input surrogate. Route
    // to FD (matches the early gate in `subject_eta_grad_impl`, so the inner
    // gradient and Jacobian stay on the same scope).
    if !subject.all_doses_fixed() {
        return None;
    }
    // Time-varying covariates make the PK parameters switch mid-decay, which dose
    // superposition can't express — route them to the event-driven Dual2 walk
    // (IOV-minus-κ). The same walk carries the oral-infusion forced responses
    // (#350 depot-bypass central / #400 zero-order depot) that the superposition
    // kernels don't, so oral-infusion subjects route there too — mirroring
    // production's `compute_predictions` dispatch. The returned `SubjectSens` has
    // the same `(η, θ)` shape, so the outer gradient / inner Jacobian consume it
    // identically.
    if subject.has_tv_covariates() || subject_has_oral_infusion(model, subject) {
        return subject_sensitivities_tvcov(model, subject, theta, eta);
    }
    // EVID=3/4 resets are handled by restricting dose superposition to the
    // current reset segment (see `reset_floor` in the obs loop): for linear PK a
    // reset zeros the compartments, so the prediction at an observation is the
    // superposition of only the doses since the most recent reset — which carries
    // the exact `∂f/∂pk` through the same closed forms. Steady-state doses assume
    // an infinite periodic history that a mid-record reset contradicts, so a
    // subject mixing SS with resets falls back to FD.
    if subject.has_resets() && subject.doses.iter().any(|d| d.ss) {
        return None;
    }

    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let oral = matches!(
        model.pk_model,
        PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral
    );
    let two_cpt = matches!(model.pk_model, PkModel::TwoCptIv | PkModel::TwoCptOral);
    let three_cpt = matches!(model.pk_model, PkModel::ThreeCptIv | PkModel::ThreeCptOral);

    // PK parameter values at (θ, η): pk_s = tv_s·exp(sel·η). pk_param_fn folds η.
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates);

    // `∂p/∂(θ,η)` (and second order) plus `slots`: the PK slot each `pd` row maps
    // to, in `pd`-row order. Analytical where possible: evaluate the compiled
    // `[individual_parameters]` program over `Dual2` seeded on (θ, η) — exact for
    // ANY parameterization (log-normal, logit-normal F, additive), no finite
    // differences (issue #367). Its rows follow the program's `pk_var_slots` order
    // (analytical: alphabetical by PK name), NOT `pk_indices` (declaration) order.
    // Falls back to the closed-form log-normal `sel` η chain plus the FD
    // `tv_theta_jacobian` θ chain (`ρ`) — whose rows ARE in `pk_indices` order —
    // when the program path is unavailable (NN-weight θ / IOV kappa axis mismatch,
    // or a literal-const PK slot the program omits).
    let (pd, slots): (crate::sens::ode_provider::ParamDerivs, Vec<usize>) = match model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .filter(|prog| prog_covers_required_pk_slots(model, prog))
        .and_then(|prog| {
            crate::sens::ode_provider::param_derivatives_from_prog(prog, model, subject, theta, eta)
                .map(|pd| (pd, prog.pk_slots()))
        }) {
        Some((pd, slots)) => (pd, slots),
        None => {
            // The closed-form fallback assumes `∂p/∂η = pk·sel` (log-normal). It is
            // only valid when every PK-param eta is LogNormal; for additive / logit
            // / custom etas the exact `∂p/∂η` must come from the program chain
            // above, and if that is unavailable (NN-weight θ or IOV kappa axis
            // mismatch) we fall back to FD rather than mis-apply the log-normal
            // chain (which would be off by a factor of `pk` vs `1`). PR #381 #4.
            if !model
                .eta_param_info
                .iter()
                .all(|e| e.param_type == crate::types::EtaParamType::LogNormal)
            {
                return None;
            }
            let pd = lognormal_param_derivatives(model, subject, theta, &pk);
            (pd, model.pk_indices.clone())
        }
    };

    // Right-size the dual width to the number of differentiated PK parameters
    // (issue #367): seed only those on a compact `0..N`, so a 2-cpt IV model runs
    // `Dual2<4>` (4× fewer Hessian entries per op than the former fixed `Dual2<8>`),
    // 3-cpt `Dual2<6>`, etc. `seed_dim[s]` is the compact dual axis for PK slot `s`
    // (`None` = constant, not differentiated); `pd` row `i` ↔ compact axis `i`.
    let mut seed_dim: [Option<usize>; N_PK] = [None; N_PK];
    for (i, &slot) in slots.iter().enumerate() {
        if slot < N_PK {
            seed_dim[slot] = Some(i);
        }
    }

    // Explicit-kernel fast path (the default): pick the model class, then take it
    // only if a hand-written kernel covers every dose (else the whole subject uses
    // `Dual2<N>`). `FERX_DISABLE_EXPLICIT_SENS=1` forces the dual path everywhere.
    // Lagtime forces the generic path too: the explicit kernels read a plain `f64`
    // elapsed time and so can't carry the `∂elapsed/∂lagtime` sensitivity.
    // The explicit IV-bolus / infusion kernels don't carry bioavailability `F`
    // (it is only baked into the oral-depot bolus form). So when `F != 1` and the
    // subject has any non-oral-bolus dose (IV bolus, or an infusion — which
    // bypasses the depot even on oral models), route the whole subject to the
    // `Dual2` path, whose `*_conc_g` applies the production `route_f_scale`
    // post-multiply (#327). When `f_bio == 1` the explicit path is `F`-exact.
    let f_affects_non_oral =
        pk.f_bio() != 1.0 && subject.doses.iter().any(|d| !(oral && !d.is_infusion()));
    let explicit_kind = if explicit_sens_disabled() || model.has_lagtime() || f_affects_non_oral {
        None
    } else {
        let kind = match model.pk_model {
            PkModel::OneCptIv => ExKind::OneCptIv,
            PkModel::OneCptOral => ExKind::OneCptOral,
            PkModel::TwoCptIv => ExKind::TwoCptIv,
            PkModel::TwoCptOral => ExKind::TwoCptOral,
            PkModel::ThreeCptIv => ExKind::ThreeCptIv,
            PkModel::ThreeCptOral => ExKind::ThreeCptOral,
        };
        Some(kind)
    };

    // Dispatch on the differentiated-parameter count so the dual width is
    // right-sized. `pk_indices.len()` ≤ `N_PK` (the fixed PK slot table).
    macro_rules! disp {
        ($($n:literal),+) => {
            match slots.len() {
                $($n => Some(SubjectSens {
                    obs: run_obs::<$n>(
                        &seed_dim, &pk, oral, two_cpt, three_cpt, explicit_kind, subject, &pd,
                        n_eta, n_theta,
                    ),
                }),)+
                _ => None,
            }
        };
    }
    let mut sens = disp!(1, 2, 3, 4, 5, 6, 7, 8, 9)?;
    // Analytic `[initial_conditions]` impulse (#524): layer `A₀ · kernel(t, pk)`
    // and its exact `(θ, η)` jet onto every observation BEFORE scaling — the same
    // insertion order as the f64 `pk::add_analytical_init`. Dispatched on the
    // `(θ, η)` axis count `n_theta + n_eta` (init programs use the same layout as
    // the scale program); `init_supported` bounds it to the table.
    if !model.analytical_init.is_empty() {
        dispatch_init_impulse!(
            n_theta + n_eta,
            apply_analytical_init_outer,
            &mut sens,
            model,
            subject,
            &pk,
            &pd,
            &slots,
            theta,
            eta,
            &subject.covariates,
            n_theta,
            n_eta
        );
    }
    // Constant `ScalarScale` output divisor: `f_scaled = f/k`. Every derivative is
    // linear in `f` and `k` is constant (`∂k/∂η = ∂k/∂θ = 0`), so the whole jet
    // divides by `k`. Matches `pk::apply_scaling` (`pred /= s`).
    if let ScalingSpec::ScalarScale(k) = model.scaling {
        if k != 1.0 {
            let inv = 1.0 / k;
            for o in sens.obs.iter_mut() {
                o.f *= inv;
                for v in o
                    .df_deta
                    .iter_mut()
                    .chain(o.d2f_deta2.iter_mut())
                    .chain(o.df_dtheta.iter_mut())
                    .chain(o.d2f_deta_dtheta.iter_mut())
                {
                    *v *= inv;
                }
            }
        }
    }
    // ExpressionScale: divide by a per-subject scale `s(θ, η)` whose own jet is
    // computed exactly from the differentiable scale program; quotient-combine
    // `scaled_f = f / s` over every η-η / η-θ block (`apply_expression_scale`).
    // Dispatched on the scale program's `(θ, η)` axis count.
    if let ScalingSpec::ExpressionScale {
        deriv: Some(prog), ..
    } = &model.scaling
    {
        apply_expression_scale_outer(
            &mut sens,
            prog,
            &pk,
            &pd,
            &slots,
            theta,
            eta,
            &subject.covariates,
            n_theta,
            n_eta,
        );
    }
    // LTBS: transform the full jet to `g = ln(f)` (after scaling), mirroring
    // `pk::apply_log_transform`. With `inv = 1/f`:
    //   g          = ln(f)
    //   ∂g/∂x      = inv · ∂f/∂x
    //   ∂²g/∂x∂y   = inv · ∂²f/∂x∂y − inv² · ∂f/∂x · ∂f/∂y     (x,y ∈ {η,θ})
    // Second derivatives are computed from the *original* first derivatives, so
    // those are read before `df_deta`/`df_dtheta` are overwritten. The value half
    // goes through the shared `pk::ltbs_log_g` (same floor-then-log as the f64
    // predictor and the ODE dual walk), applied after the jet is transformed
    // (#451 review #5). Below the floor the transform clamps to a constant ⇒ all
    // derivatives vanish.
    if model.log_transform {
        let n_eta = sens.obs.first().map_or(0, |o| o.df_deta.len());
        let n_theta = sens.obs.first().map_or(0, |o| o.df_dtheta.len());
        for o in sens.obs.iter_mut() {
            if o.f > crate::pk::LTBS_FLOOR {
                let inv = 1.0 / o.f;
                let inv2 = inv * inv;
                // η–η Hessian: g_kl = inv·f_kl − inv²·f_k·f_l.
                for k in 0..n_eta {
                    for l in 0..n_eta {
                        let idx = k * n_eta + l;
                        o.d2f_deta2[idx] =
                            inv * o.d2f_deta2[idx] - inv2 * o.df_deta[k] * o.df_deta[l];
                    }
                }
                // η–θ cross: g_km = inv·f_km − inv²·f_k(η)·f_m(θ).
                for k in 0..n_eta {
                    for m in 0..n_theta {
                        let idx = k * n_theta + m;
                        o.d2f_deta_dtheta[idx] =
                            inv * o.d2f_deta_dtheta[idx] - inv2 * o.df_deta[k] * o.df_dtheta[m];
                    }
                }
                for g in o.df_deta.iter_mut().chain(o.df_dtheta.iter_mut()) {
                    *g *= inv;
                }
            } else {
                for v in o
                    .df_deta
                    .iter_mut()
                    .chain(o.d2f_deta2.iter_mut())
                    .chain(o.df_dtheta.iter_mut())
                    .chain(o.d2f_deta_dtheta.iter_mut())
                {
                    *v = 0.0;
                }
            }
            o.f = crate::pk::ltbs_log_g(o.f);
        }
    }
    Some(sens)
}

/// Per-observation value/grad/Hessian chain at a right-sized dual width `N`
/// (= number of differentiated PK parameters). `seed_dim[s]` is the compact dual
/// axis for PK slot `s` (`None` = constant); `pd` rows are in compact-axis order,
/// so the chain reads `g[i]`/`h[i][j]` directly (identity dims).
#[allow(clippy::too_many_arguments)]
fn run_obs<const N: usize>(
    seed_dim: &[Option<usize>; N_PK],
    pk: &crate::types::PkParams,
    oral: bool,
    two_cpt: bool,
    three_cpt: bool,
    explicit_kind: Option<ExKind>,
    subject: &Subject,
    pd: &crate::sens::ode_provider::ParamDerivs,
    n_eta: usize,
    n_theta: usize,
) -> Vec<ObsSens> {
    let (cl, v1, q, v2, ka, f_bio, q3, v3) = (
        pk.cl(),
        pk.v(),
        pk.q(),
        pk.v2(),
        pk.ka(),
        pk.f_bio(),
        pk.q3(),
        pk.v3(),
    );
    let mut out = Vec::with_capacity(subject.obs_times.len());
    for &t_obs in subject.obs_times.iter() {
        // Reset segment: the most recent EVID=3/4 reset at or before this
        // observation (−∞ when the subject has no resets). Doses before it were
        // zeroed out of the compartments, so they're excluded from the
        // superposition below — mirroring the event-driven walker's
        // `event_reset_floor`. A reset+dose record (EVID=4) sits exactly at the
        // floor and is kept (`dose.time >= reset_floor`).
        let reset_floor = subject
            .reset_times
            .iter()
            .copied()
            .filter(|&r| r <= t_obs)
            .fold(f64::NEG_INFINITY, f64::max);

        // `(f, ∂f/∂pk, ∂²f/∂pk²)` in the compact `0..N` layout — from the explicit
        // kernels when applicable, else the generic `Dual2<N>` path.
        let (fval, g, h): (f64, [f64; N], [[f64; N]; N]) = if let Some(kind) = explicit_kind {
            let mut gv = [0.0; N];
            let mut hv = [[0.0; N]; N];
            let mut val = 0.0;
            for dose in &subject.doses {
                let elapsed = t_obs - dose.time;
                if elapsed < 0.0 || dose.time < reset_floor {
                    continue;
                }
                val += eval_dose_explicit(
                    kind, dose, elapsed, cl, v1, q, v2, ka, f_bio, q3, v3, seed_dim, &mut gv,
                    &mut hv,
                );
            }
            (val, gv, hv)
        } else {
            // Seed only the differentiated PK params as `Dual2<N>` on their compact
            // axes; everything else is a constant. `dv(slot, value)` does the lookup.
            let dv = |slot: usize, value: f64| -> Dual2<N> {
                match seed_dim[slot] {
                    Some(k) => Dual2::<N>::var(value, k),
                    None => Dual2::<N>::constant(value),
                }
            };
            let cl_d = dv(PK_IDX_CL, cl);
            let v1_d = dv(PK_IDX_V, v1);
            let q_d = dv(PK_IDX_Q, q);
            let v2_d = dv(PK_IDX_V2, v2);
            let ka_d = dv(PK_IDX_KA, ka);
            let f_d = dv(PK_IDX_F, f_bio);
            let q3_d = dv(PK_IDX_Q3, q3);
            let v3_d = dv(PK_IDX_V3, v3);
            // Lagtime enters each dose's concentration through the elapsed-time
            // argument; seed it as its own dual axis (`∂elapsed/∂lagtime = −1`).
            let lag_val = pk.lagtime();
            let lag_d = dv(PK_IDX_LAGTIME, lag_val);

            // Superpose dose contributions: f = Σ conc(dose, elapsed), restricted
            // to the current reset segment (`dose.time >= reset_floor`); `elapsed`
            // carries the lagtime shift and the SS pre-arrival tail wrap.
            let mut fd = Dual2::<N>::constant(0.0);
            for dose in &subject.doses {
                // Lagged-arrival reset exclusion — see the value path above (#2).
                if dose.time + lag_val < reset_floor {
                    continue;
                }
                let Some(elapsed) = lagged_elapsed(dose, t_obs, lag_val, lag_d) else {
                    continue;
                };
                let c = if three_cpt {
                    three_cpt_conc_g(
                        dose, elapsed, cl_d, v1_d, q_d, v2_d, q3_d, v3_d, ka_d, f_d, oral,
                    )
                } else if two_cpt {
                    two_cpt_conc_g(dose, elapsed, cl_d, v1_d, q_d, v2_d, ka_d, f_d, oral)
                } else {
                    one_cpt_conc_g(dose, elapsed, cl_d, v1_d, ka_d, f_d, oral)
                };
                fd = fd + c;
            }
            (fd.value, fd.grad, fd.hess)
        };

        // Match production's `conc.max(0.0)` clamp (`pk/mod.rs`): when the closed
        // form goes slightly negative (cancellation at extreme params during a
        // line-search step), the objective uses `f = 0`, so the gradient of
        // `max(f, 0)` is also 0 there. Returning the raw negative `f` and its
        // derivatives would make the analytic gradient inconsistent with the
        // objective it differentiates (PR #381 review finding #5).
        let (fval, g, h) = if fval < 0.0 {
            (0.0, [0.0; N], [[0.0; N]; N])
        } else {
            (fval, g, h)
        };

        let mut df_deta = vec![0.0; n_eta];
        let mut d2f_deta2 = vec![0.0; n_eta * n_eta];
        let mut df_dtheta = vec![0.0; n_theta];
        let mut d2f_deta_dtheta = vec![0.0; n_eta * n_theta];

        // Chain ∂f/∂p, ∂²f/∂p² (exact, from the seeded PK Dual2 in compact layout)
        // with ∂p/∂(θ,η) (from `pd`, analytical or FD fallback):
        //   ∂f/∂η_k      = Σ_i g[i]·pᵢ,η_k
        //   ∂²f/∂η_k∂η_l = Σ_ij H[i][j]·pᵢ,η_k·pⱼ,η_l + Σ_i g[i]·pᵢ,η_kη_l
        // and likewise with θ in one slot. Compact axis `i` ↔ `pd` row `i`.
        for i in 0..N {
            let gi = g[i];
            for k in 0..n_eta {
                df_deta[k] += gi * pd.dp_deta[i][k];
            }
            for m in 0..n_theta {
                df_dtheta[m] += gi * pd.dp_dtheta[i][m];
            }
        }
        for k in 0..n_eta {
            for l in 0..n_eta {
                let mut acc = 0.0;
                for i in 0..N {
                    for j in 0..N {
                        acc += h[i][j] * pd.dp_deta[i][k] * pd.dp_deta[j][l];
                    }
                    acc += g[i] * pd.d2p_deta2[i][k][l];
                }
                d2f_deta2[k * n_eta + l] = acc;
            }
        }
        for k in 0..n_eta {
            for m in 0..n_theta {
                let mut acc = 0.0;
                for i in 0..N {
                    for j in 0..N {
                        acc += h[i][j] * pd.dp_deta[i][k] * pd.dp_dtheta[j][m];
                    }
                    acc += g[i] * pd.d2p_detadtheta[i][k][m];
                }
                d2f_deta_dtheta[k * n_theta + m] = acc;
            }
        }

        out.push(ObsSens {
            f: fval,
            df_deta,
            d2f_deta2,
            df_dtheta,
            d2f_deta_dtheta,
        });
    }
    out
}

/// One dose's explicit-kernel contribution to `(f, ∂f/∂pk, ∂²f/∂pk²)` at the
/// compact axis layout: dispatches on `(kind, dose)` to the right hand-written
/// kernel (bolus / infusion / oral), scatters its `(grad, hess)` into `gv`/`hv`
/// via [`scatter_compact`], and returns the value. Only `kind`-covered doses
/// reach here ([`ExKind::covers`] gates the subject), so the match is total over
/// the covered set; the mirror of [`one_cpt_conc_g`]/[`two_cpt_conc_g`] dispatch.
#[allow(clippy::too_many_arguments)]
fn eval_dose_explicit<const N: usize>(
    kind: ExKind,
    dose: &DoseEvent,
    elapsed: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    ka: f64,
    f_bio: f64,
    q3: f64,
    v3: f64,
    seed_dim: &[Option<usize>; N_PK],
    gv: &mut [f64; N],
    hv: &mut [[f64; N]; N],
) -> f64 {
    // Steady-state doses (`ss` flag + positive interval) take the SS kernels;
    // otherwise the single-dose kernels. The scatter map (which PK slots the
    // kernel differentiates) is the same for a kind's SS and non-SS variants.
    let ss = dose.ss && dose.ii > 0.0;
    let ii = dose.ii;
    use super::{one_cpt_explicit as e1, three_cpt_explicit as e3, two_cpt_explicit as e2};
    match kind {
        ExKind::OneCptIv | ExKind::OneCptOral => {
            if dose.is_infusion() {
                let (f, gs, hs) = if ss {
                    e1::infusion_ss_explicit(
                        dose.rate,
                        dose.duration,
                        dose.amt,
                        elapsed,
                        ii,
                        cl,
                        v1,
                    )
                } else {
                    e1::infusion_explicit(dose.rate, dose.duration, dose.amt, elapsed, cl, v1)
                };
                scatter_compact(gv, hv, &gs, &hs, &[PK_IDX_CL, PK_IDX_V], seed_dim);
                f
            } else if matches!(kind, ExKind::OneCptOral) {
                let (f, gs, hs) = if ss {
                    e1::oral_ss_explicit(dose.amt, elapsed, ii, cl, v1, ka, f_bio)
                } else {
                    e1::oral_explicit(dose.amt, elapsed, cl, v1, ka, f_bio)
                };
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[PK_IDX_CL, PK_IDX_V, PK_IDX_KA, PK_IDX_F],
                    seed_dim,
                );
                f
            } else {
                let (f, gs, hs) = if ss {
                    e1::iv_bolus_ss_explicit(dose.amt, elapsed, ii, cl, v1)
                } else {
                    e1::iv_bolus_explicit(dose.amt, elapsed, cl, v1)
                };
                scatter_compact(gv, hv, &gs, &hs, &[PK_IDX_CL, PK_IDX_V], seed_dim);
                f
            }
        }
        ExKind::TwoCptIv | ExKind::TwoCptOral => {
            if dose.is_infusion() {
                let (f, gs, hs) = if ss {
                    e2::infusion_ss_explicit(
                        dose.rate,
                        dose.duration,
                        dose.amt,
                        elapsed,
                        ii,
                        cl,
                        v1,
                        q,
                        v2,
                    )
                } else {
                    e2::infusion_explicit(
                        dose.rate,
                        dose.duration,
                        dose.amt,
                        elapsed,
                        cl,
                        v1,
                        q,
                        v2,
                    )
                };
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2],
                    seed_dim,
                );
                f
            } else if matches!(kind, ExKind::TwoCptOral) {
                let (f, gs, hs) = if ss {
                    e2::oral_ss_explicit(dose.amt, elapsed, ii, cl, v1, q, v2, ka, f_bio)
                } else {
                    e2::oral_explicit(dose.amt, elapsed, cl, v1, q, v2, ka, f_bio)
                };
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[
                        PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2, PK_IDX_KA, PK_IDX_F,
                    ],
                    seed_dim,
                );
                f
            } else {
                let (f, gs, hs) = if ss {
                    e2::iv_bolus_ss_explicit(dose.amt, elapsed, ii, cl, v1, q, v2)
                } else {
                    e2::iv_bolus_explicit(dose.amt, elapsed, cl, v1, q, v2)
                };
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2],
                    seed_dim,
                );
                f
            }
        }
        ExKind::ThreeCptIv | ExKind::ThreeCptOral => {
            if dose.is_infusion() {
                let (f, gs, hs) = if ss {
                    e3::infusion_ss_explicit(
                        dose.rate,
                        dose.duration,
                        dose.amt,
                        elapsed,
                        ii,
                        cl,
                        v1,
                        q,
                        v2,
                        q3,
                        v3,
                    )
                } else {
                    e3::infusion_explicit(
                        dose.rate,
                        dose.duration,
                        dose.amt,
                        elapsed,
                        cl,
                        v1,
                        q,
                        v2,
                        q3,
                        v3,
                    )
                };
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[
                        PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2, PK_IDX_Q3, PK_IDX_V3,
                    ],
                    seed_dim,
                );
                f
            } else if matches!(kind, ExKind::ThreeCptOral) {
                let (f, gs, hs) = if ss {
                    e3::oral_ss_explicit(dose.amt, elapsed, ii, cl, v1, q, v2, q3, v3, ka, f_bio)
                } else {
                    e3::oral_explicit(dose.amt, elapsed, cl, v1, q, v2, q3, v3, ka, f_bio)
                };
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[
                        PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2, PK_IDX_Q3, PK_IDX_V3, PK_IDX_KA,
                        PK_IDX_F,
                    ],
                    seed_dim,
                );
                f
            } else {
                let (f, gs, hs) = if ss {
                    e3::iv_bolus_ss_explicit(dose.amt, elapsed, ii, cl, v1, q, v2, q3, v3)
                } else {
                    e3::iv_bolus_explicit(dose.amt, elapsed, cl, v1, q, v2, q3, v3)
                };
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[
                        PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2, PK_IDX_Q3, PK_IDX_V3,
                    ],
                    seed_dim,
                );
                f
            }
        }
    }
}

/// Scatter an explicit kernel's `M`-parameter `(grad, hess)` into the compact
/// `N`-axis layout via `seed_dim` (PK slot → compact axis). `map8[a]` is the PK
/// slot of explicit param `a`; a param whose slot isn't differentiated
/// (`seed_dim = None`, e.g. a literal-const PK value) is dropped — its derivative
/// is not chained.
fn scatter_compact<const M: usize, const N: usize>(
    g: &mut [f64; N],
    h: &mut [[f64; N]; N],
    gs: &[f64; M],
    hs: &[[f64; M]; M],
    map8: &[usize; M],
    seed_dim: &[Option<usize>; N_PK],
) {
    for a in 0..M {
        let ca = match seed_dim[map8[a]] {
            Some(c) => c,
            None => continue,
        };
        g[ca] += gs[a];
        for b in 0..M {
            if let Some(cb) = seed_dim[map8[b]] {
                h[ca][cb] += hs[a][b];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::model_parser::parse_model_string;
    use crate::pk::compute_predictions_with_tv;
    use crate::types::{test_helpers, DoseEvent, Subject};
    use std::collections::HashMap;

    #[test]
    fn analytic_outer_gradient_available_tracks_scope_and_fd() {
        // Analytical PK model with `gradient = auto` → analytic outer gradient.
        assert!(analytic_outer_gradient_available(
            &test_helpers::analytical_model(GradientMethod::Auto)
        ));
        // `gradient = fd` forces FD even for an analytical model.
        assert!(!analytic_outer_gradient_available(
            &test_helpers::analytical_model(GradientMethod::Fd)
        ));
        // An ODE model is outside the analytic scope.
        assert!(!analytic_outer_gradient_available(
            &test_helpers::ode_model(GradientMethod::Auto)
        ));
        // A closed-form `iiv_on_ruv` model is analytic (#474)…
        let mut ruv = test_helpers::analytical_model(GradientMethod::Auto);
        ruv.residual_error_eta = Some(0);
        assert!(analytic_outer_gradient_available(&ruv));
        // …and closed-form M3 BLOQ + `iiv_on_ruv` is now analytic too (#4c — the
        // censored × residual-eta cross-terms are assembled). Only ODE M3 +
        // `iiv_on_ruv` keeps FD (via `iiv_on_ruv_forces_fd`, which gates on
        // `ode_spec.is_some()`).
        let mut ruv_m3 = test_helpers::analytical_model(GradientMethod::Auto);
        ruv_m3.residual_error_eta = Some(0);
        ruv_m3.bloq_method = crate::types::BloqMethod::M3;
        assert!(analytic_outer_gradient_available(&ruv_m3));
    }

    /// A TTE (`[event_model]`) objective has no analytic outer gradient (the
    /// provider only covers the structural PK/PD model), so the predicate must be
    /// `false` even with `gradient = auto` — and `resolve_auto` must therefore
    /// pick the derivative-free Bobyqa, not a gradient-based optimizer that would
    /// stall on a gradient TTE cannot supply (#490 auto-optimizer × TTE).
    #[cfg(feature = "survival")]
    #[test]
    fn analytic_outer_gradient_unavailable_for_tte() {
        use crate::types::Optimizer;
        const TTE: &str = r"
[parameters]
  theta TVLAMBDA(0.1, 0.001, 10.0)
  omega ETA ~ 0.09

[event_model e]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA)
";
        let m = parse_model_string(TTE).expect("TTE model must parse");
        assert!(m.has_tte(), "model must register a TTE endpoint");
        assert!(
            !analytic_outer_gradient_available(&m),
            "TTE objective is FD-only: no analytic outer gradient"
        );
        assert_eq!(
            Optimizer::Auto.resolve_auto(&m),
            Optimizer::Bobyqa,
            "auto must resolve to derivative-free Bobyqa for a TTE model"
        );
    }

    const WARFARIN: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    fn oral_subject(times: &[f64]) -> Subject {
        let n = times.len();
        Subject {
            id: "1".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    const TWOCPT_IV: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    const TWOCPT_ORAL: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta TVKA(1.0, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk two_cpt_oral(cl=CL, v=V1, q=Q, v2=V2, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    const THREECPT_IV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
[structural_model]
  pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    // ── [initial_conditions] analytic gradient fixtures (#524) ──────────────
    // 1-cpt oral with a parameter-dependent CENTRAL baseline `A₀ = TVC0 · V`
    // (IV-bolus impulse kernel). A₀ depends on θ_TVC0, θ_TVV, and η_V, so the
    // analytic ∂C/∂A₀ · ∂A₀/∂(θ,η) chain is exercised.
    const ONECPT_ORAL_INIT_CENTRAL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVC0(5.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[initial_conditions]
  init(central) = TVC0 * V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    // 1-cpt oral with a pre-loaded DEPOT baseline (oral first-order kernel, F=1).
    const ONECPT_ORAL_INIT_DEPOT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVD0(80.0, 0.01, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[initial_conditions]
  init(depot) = TVD0
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    // 2-cpt IV with a central baseline (2-cpt IV-bolus impulse kernel).
    const TWOCPT_IV_INIT_CENTRAL: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta TVC0(3.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[initial_conditions]
  init(central) = TVC0 * V1
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    // 3-cpt IV with a central baseline (3-cpt IV-bolus impulse kernel).
    const THREECPT_IV_INIT_CENTRAL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  theta TVC0(4.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
[structural_model]
  pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)
[initial_conditions]
  init(central) = TVC0 * V1
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    /// The analytic `[initial_conditions]` impulse (#524) and its `(θ, η)` jet
    /// must match central finite differences of the production predictor
    /// `compute_predictions_with_tv` (which layers the f64 init), across the
    /// 1-/2-/3-cpt central kernels and the oral-depot kernel. The light inner
    /// η-gradient provider must agree with the full outer one too.
    #[test]
    fn analytical_init_provider_matches_fd() {
        let cases: &[(&str, &str, Vec<f64>, Vec<f64>)] = &[
            (
                "1cpt oral central",
                ONECPT_ORAL_INIT_CENTRAL,
                vec![0.2, 10.0, 1.5, 5.0],
                vec![0.15, -0.10, 0.25],
            ),
            (
                "1cpt oral depot",
                ONECPT_ORAL_INIT_DEPOT,
                vec![0.2, 10.0, 1.5, 80.0],
                vec![0.15, -0.10, 0.25],
            ),
            (
                "2cpt iv central",
                TWOCPT_IV_INIT_CENTRAL,
                vec![10.0, 50.0, 15.0, 100.0, 3.0],
                vec![0.12, -0.08],
            ),
            (
                "3cpt iv central",
                THREECPT_IV_INIT_CENTRAL,
                vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 4.0],
                vec![0.12, -0.08],
            ),
        ];
        for (label, src, theta, eta) in cases {
            let m = parse_model_string(src).unwrap_or_else(|e| panic!("{label}: parse: {e}"));
            assert_eq!(m.analytical_init.len(), 1, "{label}: init parsed");
            assert!(
                analytical_supported(&m),
                "{label}: init model must use the analytic provider, not FD"
            );
            let s = oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
            // Full outer jet (value, ∂η, ∂²η², ∂θ, ∂²η∂θ) vs central FD of the
            // init-aware production predictor.
            check_full_provider_vs_fd(&m, &s, theta, eta);
            // Light inner η-gradient must agree with the full provider.
            let full = subject_sensitivities(&m, &s, theta, eta).expect("full supported");
            let light = subject_eta_grad(&m, &s, theta, eta).expect("light supported");
            assert_eq!(light.len(), full.obs.len());
            for (lo, fo) in light.iter().zip(full.obs.iter()) {
                for k in 0..m.n_eta {
                    approx::assert_relative_eq!(
                        lo.df_deta[k],
                        fo.df_deta[k],
                        max_relative = 1e-9,
                        epsilon = 1e-12
                    );
                }
            }
        }
    }

    // 1-cpt IV with WT-on-CL *and* an `[initial_conditions]` baseline. Regression
    // for the #527/#524 review: the TV-cov event-driven walk does not layer the init
    // impulse, so an init model must decline TV-cov analytic support and route its
    // TV-cov subjects to FD — otherwise the analytic gradient omits the init baseline
    // while the objective keeps it.
    const ONECPT_IV_TVCOV_INIT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVC0(4.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[initial_conditions]
  init(central) = TVC0 * V
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    /// #527/#524 review (finding #1): an `[initial_conditions]` model whose subject
    /// has time-varying covariates must fall back to FD, because the TV-cov walk
    /// never layers the init impulse. Before the fix `tvcov_analytical_supported`
    /// returned `true` for the model and the walk silently dropped the `A₀·kernel`
    /// baseline from the FOCE/FOCEI gradient (outer and inner) while the objective —
    /// which uses the init-aware f64 predictor — kept it, biasing the estimates. The
    /// non-TV-cov subjects of the same model must still take the exact init path.
    #[test]
    fn analytical_init_tvcov_subject_falls_back_to_fd() {
        let m = parse_model_string(ONECPT_IV_TVCOV_INIT).expect("parse");
        assert_eq!(m.analytical_init.len(), 1, "init parsed");
        assert!(
            !tvcov_analytical_supported(&m),
            "init model must decline TV-cov analytic support (walk omits the init)"
        );

        let theta = vec![0.2, 10.0, 0.75, 4.0];
        let eta = vec![0.15, -0.10];

        // TV-cov subject (WT changes across records): outer full provider AND inner
        // η-gradient must decline so the caller falls back to FD.
        let tv = tvcov_subject(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &[70.0],
            &[1.0, 2.0, 4.0, 8.0],
            &[70.0, 80.0, 90.0, 100.0],
            Vec::new(),
            Vec::new(),
            &[],
        );
        assert!(tv.has_tv_covariates(), "fixture must carry TV covariates");
        assert!(
            subject_sensitivities(&m, &tv, &theta, &eta).is_none(),
            "TV-cov init subject must fall back to FD, not the init-less walk"
        );
        assert!(
            subject_eta_grad(&m, &tv, &theta, &eta).is_none(),
            "inner η-gradient must fall back to FD too"
        );

        // A static-covariate subject of the SAME model still takes the exact analytic
        // init path (dose superposition + layered impulse).
        let mut stat = oral_subject(&[1.0, 2.0, 4.0, 8.0]);
        stat.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        stat.covariates = wt_map(70.0);
        assert!(!stat.has_tv_covariates(), "static-covariate subject");
        assert!(
            subject_sensitivities(&m, &stat, &theta, &eta).is_some(),
            "static-covariate init subject keeps the exact analytic init path"
        );
    }

    // 1-cpt IV with `init(central) = TVC0 * V` and a `TVC0` bound that admits zero.
    // At `TVC0 = 0` the baseline amount `A₀` is exactly 0 but `∂A₀/∂TVC0 = V ≠ 0`.
    const ONECPT_IV_INIT_ZERO: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVC0(0.0, -50.0, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[initial_conditions]
  init(central) = TVC0 * V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    /// #527/#524 review (finding #2): when the init amount evaluates to exactly 0 but
    /// has nonzero parameter sensitivity, the gradient must NOT be dropped. Evaluated
    /// at `TVC0 = 0`, `A₀ = 0` yet `∂A₀/∂TVC0 = V`, so the init contributes 0 to the
    /// value but `V·kernel` to `∂f/∂TVC0` (and the mixed `∂²f/∂η∂TVC0`). Before the
    /// fix `add_init_impulse` skipped the whole impulse on `A₀ == 0`, zeroing those
    /// derivatives; the FD of the objective sees the nonzero slope, so the full
    /// provider jet (checked here) diverged from FD on the `TVC0` axis.
    #[test]
    fn analytical_init_zero_amount_keeps_gradient() {
        let m = parse_model_string(ONECPT_IV_INIT_ZERO).expect("parse");
        assert_eq!(m.analytical_init.len(), 1, "init parsed");
        assert!(
            analytical_supported(&m),
            "model must use the analytic provider"
        );
        let mut s = oral_subject(&[1.0, 2.0, 4.0, 8.0, 24.0]);
        s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        // TVC0 = 0 → A₀ = 0 exactly, the boundary the dropped-gradient bug lived on.
        check_full_provider_vs_fd(&m, &s, &[0.2, 10.0, 0.0], &[0.1, -0.1]);
    }

    // 2-cpt IV with an *additive* η on V1 (`V1 = TVV1 + ETA_V1`): a non-log-normal
    // parameterization, so `∂V1/∂η = 1` (not `V1·sel`). Forces both providers down
    // the compiled-program `∂p/∂η` path.
    const TWOCPT_IV_ADDITIVE_V1: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 9.0
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 + ETA_V1
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    const THREECPT_ORAL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  theta TVKA(1.5, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    fn subject_with_dose(dose: DoseEvent, times: &[f64]) -> Subject {
        let n = times.len();
        Subject {
            id: "1".to_string(),
            doses: vec![dose],
            obs_times: times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    /// Check the provider's `f` matches the production predictor exactly, and
    /// `∂f/∂η`, `∂f/∂θ` match its finite differences.
    fn check_provider_vs_production(
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        eta: &[f64],
    ) {
        let sens = subject_sensitivities(model, subject, theta, eta).expect("supported");
        let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
            compute_predictions_with_tv(model, subject, th, e)[j]
        };
        let n_eta = model.n_eta;
        let n_theta = theta.len();
        let he = 1e-6;
        for (j, obs) in sens.obs.iter().enumerate() {
            // f must equal the production prediction (the closed forms agree).
            approx::assert_relative_eq!(
                obs.f,
                pred(eta, theta, j),
                max_relative = 1e-9,
                epsilon = 1e-10
            );
            for k in 0..n_eta {
                let mut ep = eta.to_vec();
                ep[k] += he;
                let mut em = eta.to_vec();
                em[k] -= he;
                let g = (pred(&ep, theta, j) - pred(&em, theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 3e-4, epsilon = 1e-7);
            }
            for m in 0..n_theta {
                let h = he * (1.0 + theta[m].abs());
                let mut tp = theta.to_vec();
                tp[m] += h;
                let mut tm = theta.to_vec();
                tm[m] -= h;
                let g = (pred(eta, &tp, j) - pred(eta, &tm, j)) / (2.0 * h);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 3e-4,
                    epsilon = 1e-7
                );
            }
        }
    }

    /// The light provider [`subject_eta_grad`] must return exactly the `f` and
    /// `∂f/∂η` the full [`subject_sensitivities`] does — same generic PK source,
    /// same log-normal η chain, just first-order. Checked across 1-/2-/3-cpt IV +
    /// oral and a steady-state case.
    #[test]
    fn light_provider_matches_full_provider_eta_grad() {
        let times = [0.25, 1.0, 4.0, 12.0];
        let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
            {
                let m = parse_model_string(WARFARIN).unwrap();
                let s = oral_subject(&times);
                (m, s, vec![0.2, 10.0, 1.5], vec![0.15, -0.10, 0.25])
            },
            {
                let m = parse_model_string(TWOCPT_IV).unwrap();
                let s =
                    subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
                (m, s, vec![10.0, 50.0, 15.0, 100.0], vec![0.12, -0.08])
            },
            {
                let m = parse_model_string(THREECPT_IV).unwrap();
                let s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
                (
                    m,
                    s,
                    vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0],
                    vec![0.12, -0.08],
                )
            },
            {
                let m = parse_model_string(THREECPT_ORAL).unwrap();
                let s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 24.0), &times);
                (
                    m,
                    s,
                    vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.5],
                    vec![0.12, -0.08, 0.2],
                )
            },
            {
                // Non-log-normal η (additive on V1): exercises the program-path
                // `∂p/∂η` branch, not the closed-form `pk·sel` chain.
                let m = parse_model_string(TWOCPT_IV_ADDITIVE_V1).unwrap();
                let s =
                    subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
                (m, s, vec![10.0, 50.0, 15.0, 100.0], vec![0.1, 3.0])
            },
        ];
        for (m, s, theta, eta) in &cases {
            let full = subject_sensitivities(m, s, theta, eta).expect("full supported");
            let light = subject_eta_grad(m, s, theta, eta).expect("light supported");
            assert_eq!(full.obs.len(), light.len());
            for (fo, lo) in full.obs.iter().zip(light.iter()) {
                approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-14);
                for k in 0..m.n_eta {
                    approx::assert_relative_eq!(
                        fo.df_deta[k],
                        lo.df_deta[k],
                        max_relative = 1e-12,
                        epsilon = 1e-14
                    );
                }
            }
        }
    }

    /// A constant `obs_scale` divisor (`ScalarScale`) must flow through `f` and
    /// every η/θ derivative — the provider divides the whole jet by `k`, and the
    /// production predictor (`compute_predictions_with_tv` → `apply_scaling`)
    /// divides its predictions by the same `k`, so they (and the FD derivatives)
    /// must still agree.
    #[test]
    fn provider_scalar_scale_matches_production() {
        let scaled = WARFARIN.replace(
            "[error_model]",
            "[scaling]\n  obs_scale = 1000\n[error_model]",
        );
        let model = parse_model_string(&scaled).expect("parse");
        assert!(
            matches!(model.scaling, ScalingSpec::ScalarScale(k) if (k - 1000.0).abs() < 1e-9),
            "model must carry the ScalarScale"
        );
        assert!(
            analytical_supported(&model),
            "ScalarScale must be supported"
        );
        let subject = oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        check_provider_vs_production(&model, &subject, &[0.2, 10.0, 1.5], &[0.15, -0.10, 0.25]);

        // The light η-provider must agree with the full provider under scaling too.
        let full = subject_sensitivities(&model, &subject, &[0.2, 10.0, 1.5], &[0.15, -0.10, 0.25])
            .expect("full");
        let light = subject_eta_grad(&model, &subject, &[0.2, 10.0, 1.5], &[0.15, -0.10, 0.25])
            .expect("light");
        for (fo, lo) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-14);
            for k in 0..model.n_eta {
                approx::assert_relative_eq!(
                    fo.df_deta[k],
                    lo.df_deta[k],
                    max_relative = 1e-12,
                    epsilon = 1e-14
                );
            }
        }
    }

    #[test]
    fn provider_2cpt_bolus_infusion_oral_match_production() {
        let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];
        // 2-cpt IV: bolus (rate=0) and infusion (rate>0, dur=2).
        let iv = parse_model_string(TWOCPT_IV).expect("parse");
        let theta_iv = vec![10.0, 50.0, 15.0, 100.0];
        let eta_iv = vec![0.12, -0.08];
        let bolus = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
        let infusion = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
        check_provider_vs_production(&iv, &bolus, &theta_iv, &eta_iv);
        check_provider_vs_production(&iv, &infusion, &theta_iv, &eta_iv);

        // 2-cpt oral (first-order absorption).
        let oral_m = parse_model_string(TWOCPT_ORAL).expect("parse");
        let theta_or = vec![10.0, 50.0, 15.0, 100.0, 1.0];
        let eta_or = vec![0.12, -0.08, 0.2];
        let oral_s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
        check_provider_vs_production(&oral_m, &oral_s, &theta_or, &eta_or);
    }

    /// Regression: bioavailability `F` on an IV bolus / infusion must be applied
    /// in the sensitivity path too. Production scales non-oral routes by `F` since
    /// #327 (`route_f_scale`); before the `*_conc_g` post-multiply fix the sens IV
    /// branch ignored `F`, so the analytic gradient/Jacobian was computed for a
    /// different (unscaled) prediction surface than the FOCEI objective.
    #[test]
    fn provider_iv_with_bioavailability_matches_production() {
        const ONECPT_IV_F: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVF(0.7, 0.05, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_F  ~ 0.05
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  F  = TVF  * exp(ETA_F)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V, f=F)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let times = [0.25, 1.0, 2.0, 4.0, 8.0];
        let m = parse_model_string(ONECPT_IV_F).expect("parse");
        let theta = vec![10.0, 50.0, 0.7];
        let eta = vec![0.1, -0.05, 0.2];
        // F on an IV *bolus* is a magnitude scale, so the analytic post-multiply
        // still matches production.
        let bolus = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
        check_provider_vs_production(&m, &bolus, &theta, &eta);
        // #419: an IV *infusion* under F ≠ 1 reshapes (rate held, window F·dur)
        // rather than scaling its magnitude, so the analytic `route_f_scale`
        // post-multiply no longer matches production — both providers decline it to
        // the FD gradient (whose `event_driven_predictions` applies the #419 rule).
        let infusion = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
        assert!(
            subject_sensitivities(&m, &infusion, &theta, &eta).is_none(),
            "F≠1 rate-defined infusion must decline to FD (full provider, #419)"
        );
        assert!(
            subject_eta_grad(&m, &infusion, &theta, &eta).is_none(),
            "F≠1 rate-defined infusion must decline to FD (light provider, #419)"
        );
    }

    /// Regression: a modeled-duration dose (`RATE=-2` → `D{cmt}`) is read with
    /// unresolved `rate`/`duration` by the provider, so the analytic path must
    /// decline (→ FD) rather than optimize a bolus/zero-input surrogate.
    #[test]
    fn provider_modeled_duration_dose_falls_back_to_fd() {
        let iv = parse_model_string(TWOCPT_IV).expect("parse");
        let mut dose = DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0);
        dose.rate_mode = crate::types::RateMode::ModeledDuration;
        let subj = subject_with_dose(dose, &[0.5, 2.0, 6.0]);
        let theta = vec![10.0, 50.0, 15.0, 100.0];
        let eta = vec![0.1, -0.05];
        assert!(
            subject_eta_grad(&iv, &subj, &theta, &eta).is_none(),
            "modeled-duration dose must fall back to FD (light provider)"
        );
        assert!(
            subject_sensitivities(&iv, &subj, &theta, &eta).is_none(),
            "modeled-duration dose must fall back to FD (full provider)"
        );
    }

    #[test]
    fn provider_2cpt_steady_state_matches_production() {
        // SS bolus (II=12) and SS oral (II=24) — exercises the *_ss_g branches.
        let times = [0.5, 2.0, 6.0, 11.5];
        let iv = parse_model_string(TWOCPT_IV).expect("parse");
        let ss_bolus = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 12.0), &times);
        check_provider_vs_production(&iv, &ss_bolus, &[10.0, 50.0, 15.0, 100.0], &[0.1, -0.05]);

        let oral_m = parse_model_string(TWOCPT_ORAL).expect("parse");
        let ss_oral = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 24.0),
            &[2.0, 6.0, 12.0, 23.0],
        );
        check_provider_vs_production(
            &oral_m,
            &ss_oral,
            &[10.0, 50.0, 15.0, 100.0, 1.0],
            &[0.1, -0.05, 0.15],
        );
    }

    #[test]
    fn provider_2cpt_ss_infusion_matches_production() {
        // Non-overlapping SS infusion (rate=200, amt=1000 → dur=5; II=12 → dur<II).
        let iv = parse_model_string(TWOCPT_IV).expect("parse");
        let ss_inf = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 12.0),
            &[1.0, 4.0, 6.0, 8.0, 11.0],
        );
        check_provider_vs_production(&iv, &ss_inf, &[10.0, 50.0, 15.0, 100.0], &[0.1, -0.05]);
    }

    #[test]
    fn provider_1cpt_ss_infusion_matches_production() {
        let m = parse_model_string(
            "[parameters]\n  theta TVCL(10.0,1.0,100.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .expect("parse");
        let ss_inf = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 12.0),
            &[1.0, 4.0, 6.0, 8.0, 11.0],
        );
        check_provider_vs_production(&m, &ss_inf, &[10.0, 50.0], &[0.1, -0.05]);
    }

    #[test]
    fn provider_3cpt_bolus_infusion_oral_match_production() {
        let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];
        let iv = parse_model_string(THREECPT_IV).expect("parse");
        let theta_iv = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0];
        let eta_iv = vec![0.12, -0.08];
        let bolus = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
        let infusion = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
        check_provider_vs_production(&iv, &bolus, &theta_iv, &eta_iv);
        check_provider_vs_production(&iv, &infusion, &theta_iv, &eta_iv);

        let oral_m = parse_model_string(THREECPT_ORAL).expect("parse");
        let theta_or = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.5];
        let eta_or = vec![0.12, -0.08, 0.2];
        let oral_s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
        check_provider_vs_production(&oral_m, &oral_s, &theta_or, &eta_or);
    }

    #[test]
    fn provider_3cpt_steady_state_matches_production() {
        // SS bolus (II=12), SS oral (II=24), SS infusion (dur<II) — exercises
        // every *_ss_g branch for 3-cpt.
        let iv = parse_model_string(THREECPT_IV).expect("parse");
        let theta_iv = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0];
        let ss_bolus = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 12.0),
            &[0.5, 2.0, 6.0, 11.5],
        );
        check_provider_vs_production(&iv, &ss_bolus, &theta_iv, &[0.1, -0.05]);

        let ss_inf = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 12.0),
            &[1.0, 4.0, 6.0, 8.0, 11.0],
        );
        check_provider_vs_production(&iv, &ss_inf, &theta_iv, &[0.1, -0.05]);

        let oral_m = parse_model_string(THREECPT_ORAL).expect("parse");
        let theta_or = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.5];
        let ss_oral = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 24.0),
            &[2.0, 6.0, 12.0, 23.0],
        );
        check_provider_vs_production(&oral_m, &ss_oral, &theta_or, &[0.1, -0.05, 0.15]);
    }

    #[test]
    fn provider_overlapping_ss_infusion_matches_production() {
        // Overlapping SS infusion (rate=200, amt=1000 → dur=5; II=2 → dur>II): the
        // provider now carries the same superposed closed form as production (#379),
        // so its value/η/θ sensitivities match FD of the production predictor.
        // Observations sampled within the dosing interval [0, II).
        let iv = parse_model_string(TWOCPT_IV).expect("parse");
        let ss_inf = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 2.0),
            &[0.3, 0.8, 1.2, 1.7],
        );
        assert!(
            subject_sensitivities(&iv, &ss_inf, &[10.0, 50.0, 15.0, 100.0], &[0.1, -0.05])
                .is_some(),
            "overlapping SS infusion is now provider-supported"
        );
        check_provider_vs_production(&iv, &ss_inf, &[10.0, 50.0, 15.0, 100.0], &[0.1, -0.05]);

        // 1-cpt IV overlapping too (dur = 1000/200 = 5 > II = 2).
        let one = parse_model_string(
            "[parameters]\n  theta TVCL(10.0,1.0,100.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .expect("parse");
        let one_inf = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 2.0),
            &[0.3, 0.8, 1.2, 1.7],
        );
        check_provider_vs_production(&one, &one_inf, &[10.0, 50.0], &[0.1, -0.05]);
    }

    /// Build a subject carrying explicit doses and EVID=3/4 reset times (no
    /// covariates, no IOV) for the reset-superposition tests.
    fn subject_with_doses_and_resets(
        doses: Vec<DoseEvent>,
        times: &[f64],
        reset_times: Vec<f64>,
    ) -> Subject {
        let n = times.len();
        Subject {
            id: "1".to_string(),
            doses,
            obs_times: times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times,
            cens: vec![0; n],
            occasions: vec![1; n],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    /// Oral models with an **infusion** dose (#350 depot-bypass central, RATE>0
    /// into cmt 2; #400 zero-order into the depot, RATE>0 into cmt 1) route through
    /// the state-propagating Dual2 walk — whose oral propagators now carry
    /// `rate_central`/`rate_depot` — rather than dose superposition. The provider's
    /// value/∂η/∂²η/∂θ/∂²η∂θ must match central FD of the production predictor
    /// (`compute_predictions_with_tv`, the independent infusion-correct f64 path)
    /// across 1-/2-/3-cpt and both infusion compartments.
    #[test]
    fn oral_infusion_provider_matches_fd_of_production() {
        const ONECPT_ORAL: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.0, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_KA ~ 0.10
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
        const THREECPT_ORAL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  theta TVKA(1.0, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        // Infusion of amt 1000 over 8 h (rate 125), then a later oral bolus.
        let inf = |cmt: usize| {
            vec![
                DoseEvent::new(0.0, 1000.0, cmt, 125.0, false, 0.0),
                DoseEvent::new(12.0, 500.0, 1, 0.0, false, 0.0),
            ]
        };
        let times = [1.0, 4.0, 7.0, 10.0, 14.0, 24.0];
        let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
            // 1-cpt oral, zero-order into the **depot** (cmt 1, #400).
            {
                let m = parse_model_string(ONECPT_ORAL).expect("parse 1cpt oral");
                let s = subject_with_doses_and_resets(inf(1), &times, Vec::new());
                (m, s, vec![10.0, 50.0, 1.0], vec![0.1, -0.05, 0.08])
            },
            // 1-cpt oral, depot-bypass infusion into **central** (cmt 2, #350).
            {
                let m = parse_model_string(ONECPT_ORAL).expect("parse 1cpt oral");
                let s = subject_with_doses_and_resets(inf(2), &times, Vec::new());
                (m, s, vec![10.0, 50.0, 1.0], vec![0.1, -0.05, 0.08])
            },
            // 2-cpt oral, zero-order into the depot (cmt 1).
            {
                let m = parse_model_string(TWOCPT_ORAL).expect("parse 2cpt oral");
                let s = subject_with_doses_and_resets(inf(1), &times, Vec::new());
                (
                    m,
                    s,
                    vec![10.0, 50.0, 15.0, 100.0, 1.0],
                    vec![0.1, -0.05, 0.08],
                )
            },
            // 3-cpt oral, zero-order into the depot (cmt 1).
            {
                let m = parse_model_string(THREECPT_ORAL).expect("parse 3cpt oral");
                let s = subject_with_doses_and_resets(inf(1), &times, Vec::new());
                (
                    m,
                    s,
                    vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.0],
                    vec![0.1, -0.05, 0.08],
                )
            },
        ];
        for (m, s, theta, eta) in &cases {
            assert!(
                subject_has_oral_infusion(m, s),
                "fixture must carry an oral infusion"
            );
            assert!(
                subject_sensitivities(m, s, theta, eta).is_some(),
                "oral-infusion subject must take the analytic provider (via the walk)"
            );
            check_provider_vs_production(m, s, theta, eta);
        }
    }

    /// Two infusion occasions on a 3-cpt IV model separated by an EVID=4 reset:
    /// occasion-2 observations must rebuild from zero (no occasion-1 carryover).
    /// The provider's reset-segment superposition must reproduce the production
    /// event-driven predictor and its FD sensitivities.
    #[test]
    fn provider_3cpt_two_occasion_reset_matches_production() {
        let iv = parse_model_string(THREECPT_IV).expect("parse");
        let theta = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0];
        let eta = vec![0.12, -0.08];
        // Occasion 1: infusion at t=0 (rate 200, amt 1000 → 5 h). Occasion 2:
        // same infusion at t=120, opened by an EVID=4 reset at t=120.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 200.0, false, 0.0),
            DoseEvent::new(120.0, 1000.0, 1, 200.0, false, 0.0),
        ];
        let times = [2.0, 4.0, 8.0, 60.0, 122.0, 126.0, 150.0];
        let subject = subject_with_doses_and_resets(doses, &times, vec![120.0]);
        assert!(subject.has_resets(), "fixture must carry a reset");
        check_provider_vs_production(&iv, &subject, &theta, &eta);
    }

    /// A reset that lands mid-infusion (1-cpt IV): the ongoing infusion is turned
    /// off and the compartment zeroed, so post-reset observations see only doses
    /// from the new segment. Exercises the `dose.time < reset_floor` exclusion of
    /// an in-flight infusion.
    #[test]
    fn provider_1cpt_reset_midinfusion_matches_production() {
        let m = parse_model_string(
            "[parameters]\n  theta TVCL(10.0,1.0,100.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .expect("parse");
        // Infusion 0–8 h (rate 125, amt 1000); reset at t=4 mid-infusion; a fresh
        // bolus opens the new segment at t=4.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 125.0, false, 0.0),
            DoseEvent::new(4.0, 500.0, 1, 0.0, false, 0.0),
        ];
        let times = [1.0, 3.0, 5.0, 7.0, 10.0];
        let subject = subject_with_doses_and_resets(doses, &times, vec![4.0]);
        check_provider_vs_production(&m, &subject, &[10.0, 50.0], &[0.1, -0.05]);
    }

    /// Provider's exact η/θ sensitivities (value, ∂/∂η, ∂²/∂η², ∂/∂θ, ∂²/∂η∂θ)
    /// must match central finite differences of the production predictor
    /// `compute_predictions_with_tv`. Shared by the natural-scale and LTBS checks —
    /// for an LTBS model the production predictor returns `ln(f)`, and the provider
    /// applies the matching `g = ln(f)` jet transform, so the same FD check covers
    /// the log-scale value, gradient, and Hessian.
    fn check_full_provider_vs_fd(
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        eta: &[f64],
    ) {
        let n_eta = model.n_eta;
        let n_theta = theta.len();

        let sens = subject_sensitivities(model, subject, theta, eta).expect("supported");

        // FD helpers over the full prediction vector (returns obs j's value).
        let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
            compute_predictions_with_tv(model, subject, th, e)[j]
        };
        let he = 1e-6; // first-derivative step
        let ht = 1e-6;
        let heh = 1e-4; // second-derivative step (4-point central is roundoff-prone)

        for (j, obs) in sens.obs.iter().enumerate() {
            // value
            let f0 = pred(&eta, &theta, j);
            approx::assert_relative_eq!(obs.f, f0, max_relative = 1e-9, epsilon = 1e-12);

            // ∂f/∂η and ∂²f/∂η²
            for k in 0..n_eta {
                let mut ep = eta.to_vec();
                ep[k] += he;
                let mut em = eta.to_vec();
                em[k] -= he;
                let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 2e-4, epsilon = 1e-7);
                for l in 0..n_eta {
                    let mut pp = eta.to_vec();
                    pp[k] += heh;
                    pp[l] += heh;
                    let mut pm = eta.to_vec();
                    pm[k] += heh;
                    pm[l] -= heh;
                    let mut mp = eta.to_vec();
                    mp[k] -= heh;
                    mp[l] += heh;
                    let mut mm = eta.to_vec();
                    mm[k] -= heh;
                    mm[l] -= heh;
                    let hh = (pred(&pp, &theta, j) - pred(&pm, &theta, j) - pred(&mp, &theta, j)
                        + pred(&mm, &theta, j))
                        / (4.0 * heh * heh);
                    approx::assert_relative_eq!(
                        obs.d2f_deta2[k * n_eta + l],
                        hh,
                        max_relative = 3e-3,
                        epsilon = 1e-5
                    );
                }
            }

            // ∂f/∂θ
            for m in 0..n_theta {
                let mut tp = theta.to_vec();
                tp[m] += ht * (1.0 + theta[m].abs());
                let mut tm = theta.to_vec();
                tm[m] -= ht * (1.0 + theta[m].abs());
                let step = ht * (1.0 + theta[m].abs());
                let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * step);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 2e-4,
                    epsilon = 1e-7
                );
            }

            // ∂²f/∂η∂θ (mixed 4-point)
            for k in 0..n_eta {
                for m in 0..n_theta {
                    let s = heh * (1.0 + theta[m].abs());
                    let mut ep = eta.to_vec();
                    ep[k] += heh;
                    let mut em = eta.to_vec();
                    em[k] -= heh;
                    let mut tp = theta.to_vec();
                    tp[m] += s;
                    let mut tm = theta.to_vec();
                    tm[m] -= s;
                    let hh = (pred(&ep, &tp, j) - pred(&ep, &tm, j) - pred(&em, &tp, j)
                        + pred(&em, &tm, j))
                        / (4.0 * heh * s);
                    approx::assert_relative_eq!(
                        obs.d2f_deta_dtheta[k * n_theta + m],
                        hh,
                        max_relative = 3e-3,
                        epsilon = 1e-5
                    );
                }
            }
        }
    }

    /// Provider's exact η/θ sensitivities must match central finite differences
    /// of the production predictor `compute_predictions_with_tv`.
    #[test]
    fn provider_matches_fd_of_production_predictor() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let subject = oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        check_full_provider_vs_fd(&model, &subject, &[0.2, 10.0, 1.5], &[0.15, -0.10, 0.25]);
    }

    /// Regression for #455/#456: an analytical model whose `[individual_parameters]`
    /// block has **intermediate** assignments before the structural PK outputs must
    /// drive the exact program-based sensitivity path even on a **static-covariate**
    /// subject (the non-TV `subject_sensitivities` provider). Before the fix, those
    /// gates compared `prog.pk_slots().len() == model.pk_indices.len()`; the
    /// intermediate rows make `pk_indices` longer, so the gate rejected the program
    /// path and fell back to the log-normal closed form keyed by `pk_indices`, whose
    /// slot-0-aliased intermediate rows overwrote CL's seed and silently zeroed
    /// `∂f/∂η_CL`. With the gate keyed on the required structural slots, the program
    /// path runs and matches FD exactly.
    #[test]
    fn static_cov_intermediate_params_uses_program_path_and_matches_fd() {
        let bolus = |t: f64| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0);
        let m = parse_model_string(TWOCPT_IV_TVCOV_INTERMEDIATE)
            .expect("parse 2cpt iv tvcov intermediate");

        // Static WT (a single `subject.covariates` snapshot, no per-event covariate
        // vectors) → NOT a TV-covariate subject, so this routes through the non-TV
        // `subject_sensitivities` / `subject_eta_grad` gates that the fix repaired.
        let mut s = subject_with_dose(bolus(0.0), &[0.5, 2.0, 6.0, 12.0]);
        s.covariates = wt_map(70.0);
        assert!(
            !s.has_tv_covariates(),
            "fixture must be a static-covariate subject so the non-TV provider runs"
        );

        // The fixture genuinely exposes intermediate rows (the precondition that the
        // old `len == len` gate tripped on), and the repaired gate admits it.
        let prog = m
            .indiv_param_partials
            .indiv_param_program
            .as_ref()
            .expect("compiled individual program");
        assert!(
            m.pk_indices.len() > prog.pk_slots().len(),
            "fixture must expose intermediate individual-parameter rows"
        );
        assert!(
            prog_covers_required_pk_slots(&m, prog),
            "repaired gate must admit the intermediate-parameter program path"
        );

        let theta = vec![10.0, 50.0, 15.0, 100.0, 0.75];
        let eta = vec![0.12, -0.08];

        // Sanity: the η_CL gradient must be non-zero — the exact symptom the old
        // mis-seeded fallback produced was a zeroed CL gradient.
        let sens = subject_sensitivities(&m, &s, &theta, &eta).expect("supported");
        let max_cl_grad = sens
            .obs
            .iter()
            .map(|o| o.df_deta[0].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_cl_grad > 1e-6,
            "∂f/∂η_CL must be non-zero (was silently zeroed by the mis-seeded fallback)"
        );

        // Full provider (and, via the harness's reference, the production predictor)
        // must match central finite differences exactly.
        check_full_provider_vs_fd(&m, &s, &theta, &eta);

        // The light inner η-gradient provider must agree with the full provider too.
        let light = subject_eta_grad(&m, &s, &theta, &eta).expect("light supported");
        assert_eq!(light.len(), sens.obs.len());
        for (lo, fo) in light.iter().zip(sens.obs.iter()) {
            for k in 0..m.n_eta {
                approx::assert_relative_eq!(
                    lo.df_deta[k],
                    fo.df_deta[k],
                    max_relative = 1e-9,
                    epsilon = 1e-12
                );
            }
        }
    }

    /// LTBS (`log(DV) ~ additive(...)`): the production predictor returns `ln(f)`,
    /// and the provider applies the matching `g = ln(f)` jet transform. The full
    /// value/gradient/Hessian must still match FD of the (log-scale) production
    /// predictor, and the light η-provider must agree with the full one — over
    /// 1-/2-/3-cpt so the second-order `g_kl = f_kl/f − f_k·f_l/f²` chain is covered.
    #[test]
    fn provider_ltbs_matches_production() {
        let ltbs = |src: &str| {
            src.replace(
                "[error_model]\n  DV ~ proportional(PROP_ERR)",
                "[error_model]\n  log(DV) ~ additive(PROP_ERR)",
            )
        };
        let times = [0.25, 1.0, 4.0, 12.0];
        let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
            {
                let m = parse_model_string(&WARFARIN.replace(
                    "[error_model]\n  DV ~ proportional(PROP_ERR)",
                    "[error_model]\n  log(DV) ~ additive(PROP_ERR)",
                ))
                .expect("parse warfarin LTBS");
                assert!(m.log_transform, "LTBS flag must be set");
                (
                    m,
                    oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
                    vec![0.2, 10.0, 1.5],
                    vec![0.15, -0.10, 0.25],
                )
            },
            {
                let m = parse_model_string(&ltbs(TWOCPT_IV)).expect("parse 2cpt LTBS");
                let s =
                    subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
                (m, s, vec![10.0, 50.0, 15.0, 100.0], vec![0.12, -0.08])
            },
            {
                let m = parse_model_string(&ltbs(THREECPT_ORAL)).expect("parse 3cpt LTBS");
                let s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 24.0), &times);
                (
                    m,
                    s,
                    vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.5],
                    vec![0.12, -0.08, 0.2],
                )
            },
        ];
        for (m, s, theta, eta) in &cases {
            assert!(analytical_supported(m), "LTBS must be provider-supported");
            check_full_provider_vs_fd(m, s, theta, eta);

            // Light η-provider must equal the full provider's log-scale f and ∂g/∂η.
            let full = subject_sensitivities(m, s, theta, eta).expect("full");
            let light = subject_eta_grad(m, s, theta, eta).expect("light");
            for (fo, lo) in full.obs.iter().zip(light.iter()) {
                approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-14);
                for k in 0..m.n_eta {
                    approx::assert_relative_eq!(
                        fo.df_deta[k],
                        lo.df_deta[k],
                        max_relative = 1e-12,
                        epsilon = 1e-14
                    );
                }
            }
        }
    }

    // 1-cpt oral with a log-normal dose lagtime (`LAGTIME = TVLAG·exp(ETA_LAG)`).
    const ONECPT_ORAL_LAG: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVLAG(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    /// The exact ODE twin of [`ONECPT_ORAL_LAG`] (depot → central, `LAGTIME` on the
    /// depot bolus). Same θ/Ω/σ and parameterization, so the ODE provider's full
    /// `SubjectSens` (value, `∂f/∂η`, `∂f/∂θ`, and the **2nd-order** blocks) must equal
    /// the closed-form analytical provider's to RK45 accuracy — validating the
    /// event-time (saltation) sensitivity, including its Hessian, against an independent
    /// path (#439 lagtime).
    const ONECPT_ORAL_LAG_ODE_TWIN: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVLAG(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;

    #[test]
    fn ode_lagtime_full_sens_matches_analytical_twin() {
        use crate::types::DoseEvent;
        let ana = parse_model_string(ONECPT_ORAL_LAG).expect("parse analytical oral lag");
        let ode = parse_model_string(ONECPT_ORAL_LAG_ODE_TWIN).expect("parse ODE oral lag");
        assert!(
            analytical_supported(&ana),
            "analytical lag must be supported"
        );
        assert!(
            crate::sens::ode_provider::ode_analytical_supported(&ode),
            "ODE bare-lag must be supported"
        );
        let n = 6usize;
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 4.0, 6.0, 9.0, 12.0],
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        // θ = [TVCL, TVV, TVKA, TVLAG]; η = [ETA_CL, ETA_V, ETA_KA, ETA_LAG] (lag carries IIV).
        let theta = [0.2, 10.0, 1.5, 0.75];
        let eta = [0.1, -0.05, 0.15, 0.08];
        let a =
            subject_sensitivities(&ana, &subject, &theta, &eta).expect("analytical sens supported");
        let o = crate::sens::ode_provider::ode_subject_sensitivities(&ode, &subject, &theta, &eta)
            .expect("ODE sens supported");
        assert_eq!(a.obs.len(), o.obs.len());
        for (oa, oo) in a.obs.iter().zip(o.obs.iter()) {
            approx::assert_relative_eq!(oa.f, oo.f, max_relative = 1e-6, epsilon = 1e-9);
            for (x, y) in oa.df_deta.iter().zip(oo.df_deta.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-8);
            }
            for (x, y) in oa.df_dtheta.iter().zip(oo.df_dtheta.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-8);
            }
            for (x, y) in oa.d2f_deta2.iter().zip(oo.d2f_deta2.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-4, epsilon = 1e-7);
            }
            for (x, y) in oa.d2f_deta_dtheta.iter().zip(oo.d2f_deta_dtheta.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-4, epsilon = 1e-7);
            }
        }
    }

    // 1-cpt IV bolus with a log-normal lagtime (`alag=` alias, IV route).
    const ONECPT_IV_LAG: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVLAG(1.0, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V, alag=LAGTIME)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    // 2-cpt oral with a log-normal lagtime.
    const TWOCPT_ORAL_LAG: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta TVKA(1.0, 0.05, 20.0)
  theta TVLAG(0.6, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  pk two_cpt_oral(cl=CL, v=V1, q=Q, v2=V2, ka=KA, lagtime=LAGTIME)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    /// Dose lagtime is now a differentiated PK slot: it enters every dose through
    /// the elapsed-time argument (`∂elapsed/∂lagtime = −1`), seeded as its own dual
    /// axis. The provider's exact value/∂η/∂²η/∂θ/∂η∂θ must match FD of the
    /// production predictor for IV bolus, 1-/2-cpt oral, and — crucially — a
    /// steady-state oral dose with an observation in the pre-arrival window
    /// `[dose.time, dose.time + lagtime)`, which exercises the SS tail wrap.
    #[test]
    fn provider_lagtime_matches_production() {
        let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
            {
                let m = parse_model_string(ONECPT_ORAL_LAG).expect("parse 1cpt oral lag");
                assert!(m.has_lagtime(), "model must carry a lagtime");
                (
                    m,
                    oral_subject(&[1.0, 2.0, 4.0, 8.0, 24.0]),
                    vec![0.2, 10.0, 1.5, 0.75],
                    vec![0.15, -0.10, 0.25, 0.12],
                )
            },
            {
                let m = parse_model_string(ONECPT_IV_LAG).expect("parse 1cpt iv lag");
                let s = subject_with_dose(
                    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                    &[1.5, 3.0, 6.0, 12.0],
                );
                (m, s, vec![10.0, 50.0, 1.0], vec![0.1, -0.05, 0.2])
            },
            {
                let m = parse_model_string(TWOCPT_ORAL_LAG).expect("parse 2cpt oral lag");
                (
                    m,
                    oral_subject(&[1.0, 2.0, 6.0, 12.0, 24.0]),
                    vec![10.0, 50.0, 15.0, 100.0, 1.0, 0.6],
                    vec![0.12, -0.08, 0.15, 0.1],
                )
            },
            {
                // Steady-state oral with an observation at t=0.5 inside the
                // pre-arrival window (lagtime ≈ 0.8 h): the SS tail wrap branch.
                let m = parse_model_string(ONECPT_ORAL_LAG).expect("parse 1cpt oral lag ss");
                let s = subject_with_dose(
                    DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0),
                    &[0.5, 2.0, 6.0, 12.0, 23.0],
                );
                (
                    m,
                    s,
                    vec![0.2, 10.0, 1.5, 0.75],
                    vec![0.15, -0.10, 0.25, 0.12],
                )
            },
        ];
        for (m, s, theta, eta) in &cases {
            assert!(
                analytical_supported(m),
                "lagtime must be provider-supported"
            );
            check_full_provider_vs_fd(m, s, theta, eta);

            // Light η-provider must equal the full provider's f and ∂f/∂η.
            let full = subject_sensitivities(m, s, theta, eta).expect("full");
            let light = subject_eta_grad(m, s, theta, eta).expect("light");
            for (fo, lo) in full.obs.iter().zip(light.iter()) {
                approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-10, epsilon = 1e-12);
                for k in 0..m.n_eta {
                    approx::assert_relative_eq!(
                        fo.df_deta[k],
                        lo.df_deta[k],
                        max_relative = 1e-9,
                        epsilon = 1e-11
                    );
                }
            }
        }
    }

    /// Reset + lagtime: a dose recorded *before* a reset but *arriving after* it
    /// (via lagtime) must contribute to the post-reset segment, exactly as the
    /// production event-driven walk applies it. The reset exclusion keys on the
    /// lagged arrival `dose.time + lag`, not the record time (PR #381 review #2).
    /// Dose at t=4 with lag≈0.75 arrives ≈4.75, past the reset at t=4.5; the
    /// earlier t=0 dose (arrives ≈0.75) is correctly washed out. Validated against
    /// `compute_predictions_with_tv` via `check_full_provider_vs_fd` (value 1e-9).
    #[test]
    fn provider_reset_with_lagged_post_reset_dose_matches_production() {
        let m = parse_model_string(ONECPT_ORAL_LAG).expect("parse 1cpt oral lag");
        let s = subject_with_doses_and_resets(
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(4.0, 100.0, 1, 0.0, false, 0.0),
            ],
            &[5.0, 6.0, 8.0, 12.0],
            vec![4.5],
        );
        // eta_lag = 0 → LAGTIME = TVLAG = 0.75; arrival of the t=4 dose is 4.75 > 4.5.
        let theta = vec![0.2, 10.0, 1.5, 0.75];
        let eta = vec![0.1, -0.05, 0.2, 0.0];
        check_full_provider_vs_fd(&m, &s, &theta, &eta);
    }

    /// `[scaling] obs_scale = 1000 / V` with `V = TVV·exp(ETA_V)`: an
    /// η/θ-dependent `ExpressionScale`. The provider divides the whole jet by the
    /// scale via the differentiable scale program (quotient rule), so its exact
    /// value/∂η/∂²η/∂θ/∂²η∂θ must match FD of the production predictor (which
    /// applies the same scale through `apply_scaling`).
    #[test]
    fn provider_expression_scale_matches_production() {
        let src = WARFARIN.replace(
            "[error_model]\n  DV ~ proportional(PROP_ERR)",
            "[error_model]\n  DV ~ proportional(PROP_ERR)\n[scaling]\n  obs_scale = 1000 / V",
        );
        let model = parse_model_string(&src).expect("scaling model parses");
        assert!(
            matches!(
                model.scaling,
                ScalingSpec::ExpressionScale { deriv: Some(_), .. }
            ),
            "model must carry a differentiable scale program"
        );
        assert!(
            analytical_supported(&model),
            "η/θ-dependent ExpressionScale must be provider-supported"
        );
        let subject = oral_subject(&[1.0, 2.0, 4.0, 8.0, 24.0]);
        check_full_provider_vs_fd(&model, &subject, &[0.2, 10.0, 1.5], &[0.15, -0.10, 0.25]);
    }

    /// The **light** inner η-provider (`subject_eta_grad`) must carry the same
    /// `ExpressionScale` η-only quotient rule as the full provider for the
    /// `obs_scale = 1000 / V` model — `apply_expression_scale_inner` is the η-block of
    /// `apply_expression_scale`. Since `provider_expression_scale_matches_production`
    /// already pins the full provider's scaled `f`/`∂f/∂η` to FD of the production
    /// predictor, light ≡ full here transitively validates the inner gradient against
    /// production. Guards the inner EBE loop (the BFGS gradient and the H-matrix Jacobian
    /// both read `subject_eta_grad`), which previously reverted `ExpressionScale` to FD.
    #[test]
    fn light_provider_expression_scale_matches_full() {
        let src = WARFARIN.replace(
            "[error_model]\n  DV ~ proportional(PROP_ERR)",
            "[error_model]\n  DV ~ proportional(PROP_ERR)\n[scaling]\n  obs_scale = 1000 / V",
        );
        let model = parse_model_string(&src).expect("scaling model parses");
        assert!(
            matches!(
                model.scaling,
                ScalingSpec::ExpressionScale { deriv: Some(_), .. }
            ),
            "model must carry a differentiable scale program"
        );
        // The model-level inner gate must now serve `ExpressionScale` analytically
        // (no longer a common bail).
        assert!(
            crate::estimation::inner_optimizer::analytic_inner_grad_supported_model(&model),
            "ExpressionScale inner gradient must be in analytic scope"
        );
        let subject = oral_subject(&[1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = [0.2, 10.0, 1.5];
        let eta = [0.15, -0.10, 0.25];
        let full = subject_sensitivities(&model, &subject, &theta, &eta).expect("full");
        let light = subject_eta_grad(&model, &subject, &theta, &eta).expect("light supported");
        assert_eq!(full.obs.len(), light.len());
        for (fo, lo) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-14);
            for k in 0..model.n_eta {
                approx::assert_relative_eq!(
                    fo.df_deta[k],
                    lo.df_deta[k],
                    max_relative = 1e-12,
                    epsilon = 1e-14
                );
            }
        }
    }

    /// Regression for the finding-3 stack-buffer bound (#534 audit): a scale program's
    /// `var_to_pk_slot().len()` is the number of `[individual_parameters]` vars, NOT the
    /// axis count — a model may declare more than `MAX_SCALE_AXES` individual parameters.
    /// `apply_expression_scale_inner` must not panic on the fixed-size buffer there (it
    /// falls back to a heap `Vec`). Build a chain of 18 vars (> 16) and confirm the light
    /// inner gradient runs and still matches the full provider.
    #[test]
    fn light_provider_expression_scale_many_indiv_params_no_panic() {
        let mut ip = String::from("[individual_parameters]\n  A0 = 1.0\n");
        for i in 1..16 {
            ip.push_str(&format!("  A{i} = A{}\n", i - 1));
        }
        // 16 A-vars (A0..A15) + CL + V = 18 individual-parameter vars > MAX_SCALE_AXES.
        ip.push_str("  CL = TVCL * exp(ETA_CL) * A15\n  V = TVV * exp(ETA_V)\n");
        let src = format!(
            "[parameters]\n  theta TVCL(0.13,0.01,1.0)\n  theta TVV(8.0,1.0,50.0)\n  \
             omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.05\n{ip}\
             [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = 1000 / V\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n"
        );
        let model = parse_model_string(&src).expect("parse many-param scaling model");
        assert!(
            matches!(
                model.scaling,
                ScalingSpec::ExpressionScale { deriv: Some(_), .. }
            ),
            "model must carry a differentiable scale program"
        );
        let subject = oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0]);
        let theta = [0.13, 8.0];
        let eta = [0.1, -0.05];
        let full = subject_sensitivities(&model, &subject, &theta, &eta).expect("full");
        let light = subject_eta_grad(&model, &subject, &theta, &eta).expect("light supported");
        assert_eq!(full.obs.len(), light.len());
        for (fo, lo) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-14);
            for k in 0..model.n_eta {
                approx::assert_relative_eq!(
                    fo.df_deta[k],
                    lo.df_deta[k],
                    max_relative = 1e-12,
                    epsilon = 1e-14
                );
            }
        }
    }

    /// LTBS combined with an η-dependent `ExpressionScale` routes the **inner** gradient to
    /// FD (#534 review #5): the analytic outer still serves it, but the light provider
    /// declines so the covariance H-matrix Jacobian (`subject_eta_jacobian`, gated only on
    /// `Some`/`None`) isn't built from an analytic scale+log jet paired with the
    /// FD-converged LTBS EBE. Restores the pre-#486 behaviour for this combo and matches the
    /// ODE path's `!log_transform` gate.
    #[test]
    fn ltbs_plus_expression_scale_inner_falls_back_to_fd() {
        let src = WARFARIN.replace(
            "[error_model]\n  DV ~ proportional(PROP_ERR)",
            "[error_model]\n  DV ~ proportional(PROP_ERR)\n[scaling]\n  obs_scale = 1000 / V",
        );
        let mut model = parse_model_string(&src).expect("parse");
        model.log_transform = true;
        let subject = oral_subject(&[1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = [0.2, 10.0, 1.5];
        let eta = [0.15, -0.10, 0.25];
        // Inner declines → FD H-matrix, matching the model-level gate.
        assert!(
            subject_eta_grad(&model, &subject, &theta, &eta).is_none(),
            "LTBS + ExpressionScale inner gradient must fall back to FD"
        );
        assert!(
            !crate::estimation::inner_optimizer::analytic_inner_grad_supported_model(&model),
            "LTBS keeps the model out of analytic inner scope"
        );
        // The analytic OUTER gradient still serves LTBS + ExpressionScale.
        assert!(
            subject_sensitivities(&model, &subject, &theta, &eta).is_some(),
            "analytic outer gradient still serves LTBS + ExpressionScale"
        );
    }

    // ── Time-varying covariate analytic sensitivities ─────────────────

    // Allometric WT-on-CL, the canonical time-varying covariate: `WT` changes
    // across a subject's records, so `CL = TVCL·(WT/70)^THETA_WT·exp(ETA_CL)`
    // switches mid-decay. θ = [TVCL, TVV, TVKA, THETA_WT].
    const ONECPT_ORAL_TVCOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    // 1-cpt oral with WT-on-CL **and a constant `obs_scale` divisor** — the scale
    // is covariate-independent, so the whole jet divides by it. θ = [TVCL, TVV,
    // TVKA, THETA_WT].
    const ONECPT_ORAL_TVCOV_SCALED: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[scaling]
  obs_scale = 1000
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    // 1-cpt IV with WT-on-CL, used for the **steady-state + TV-cov** case. θ =
    // [TVCL, TVV, THETA_WT].
    const ONECPT_IV_TVCOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    // 2-cpt IV with WT-on-CL. θ = [TVCL, TVV1, TVQ, TVV2, THETA_WT].
    const TWOCPT_IV_TVCOV: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    // 2-cpt IV with WT-on-CL through intermediate individual-parameter assignments.
    // Regression for #455: the TV-cov Dual2 gate must look at the compiled PK
    // outputs (`prog.pk_slots()`), not `model.pk_indices`, because `pk_indices` is
    // parallel to all unconditional assignments and contains intermediate rows.
    const TWOCPT_IV_TVCOV_INTERMEDIATE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  WTREL = WT / 70
  WTCL  = WTREL ^ THETA_WT
  BASECL = TVCL * WTCL
  CL = BASECL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  QBASE = TVQ
  Q  = QBASE
  V2BASE = TVV2
  V2 = V2BASE
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    // 3-cpt oral with WT-on-CL. θ = [TVCL, TVV1, TVQ2, TVV2, TVQ3, TVV3, TVKA,
    // THETA_WT].
    const THREECPT_ORAL_TVCOV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  theta TVKA(1.5, 0.05, 20.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    fn wt_map(wt: f64) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("WT".to_string(), wt);
        m
    }

    /// Build a single-subject TV-covariate fixture with per-event `WT` snapshots.
    /// `dose_wts`/`obs_wts`/`pk_only_wts` are parallel to `doses`/`obs_times`/
    /// `pk_only_times`; populating `dose_covariates`/`obs_covariates` is what makes
    /// `has_tv_covariates()` true (and routes production + provider through the
    /// event-driven walk).
    #[allow(clippy::too_many_arguments)]
    fn tvcov_subject(
        doses: Vec<DoseEvent>,
        dose_wts: &[f64],
        obs_times: &[f64],
        obs_wts: &[f64],
        reset_times: Vec<f64>,
        pk_only_times: Vec<f64>,
        pk_only_wts: &[f64],
    ) -> Subject {
        let n = obs_times.len();
        Subject {
            id: "1".to_string(),
            doses,
            obs_times: obs_times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates: wt_map(obs_wts[0]),
            dose_covariates: dose_wts.iter().map(|&w| wt_map(w)).collect(),
            obs_covariates: obs_wts.iter().map(|&w| wt_map(w)).collect(),
            pk_only_times,
            pk_only_covariates: pk_only_wts.iter().map(|&w| wt_map(w)).collect(),
            reset_times,
            cens: vec![0; n],
            occasions: vec![1; n],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    /// The TV-covariate provider's exact value/∂η/∂²η/∂θ/∂²η∂θ must match central
    /// finite differences of the production predictor `compute_predictions_with_tv`
    /// (the independent f64 event-driven path), across 1-/2-/3-cpt and the three
    /// scenarios the walk must cover: (a) the covariate changing at observations,
    /// (b) a covariate breakpoint carried by an EVID=2 (`pk_only`) record between
    /// observations, and (c) a covariate change combined with an EVID=4 reset.
    #[test]
    fn tvcov_provider_matches_fd_of_production() {
        let bolus = |t: f64| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0);
        let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
            // (a) 1-cpt oral, WT changing at each observation.
            {
                let m = parse_model_string(ONECPT_ORAL_TVCOV).expect("parse 1cpt oral tvcov");
                let s = tvcov_subject(
                    vec![bolus(0.0)],
                    &[70.0],
                    &[1.0, 2.0, 4.0, 8.0, 24.0],
                    &[70.0, 72.0, 80.0, 85.0, 90.0],
                    Vec::new(),
                    Vec::new(),
                    &[],
                );
                (m, s, vec![0.2, 10.0, 1.5, 0.75], vec![0.15, -0.10, 0.25])
            },
            // (b) 1-cpt oral, covariate breakpoint at an EVID=2 record (t=3) that
            // falls between observations — the WT jumps 70→95 there, switching CL
            // mid-decay with no observation at the breakpoint.
            {
                let m =
                    parse_model_string(ONECPT_ORAL_TVCOV).expect("parse 1cpt oral tvcov pkonly");
                let s = tvcov_subject(
                    vec![bolus(0.0)],
                    &[70.0],
                    &[1.0, 2.0, 4.0, 8.0],
                    &[70.0, 70.0, 95.0, 95.0],
                    Vec::new(),
                    vec![3.0],
                    &[95.0],
                );
                (m, s, vec![0.2, 10.0, 1.5, 0.75], vec![0.12, 0.08, -0.15])
            },
            // (c) 1-cpt oral, WT change combined with an EVID=4 reset at t=12.
            {
                let m = parse_model_string(ONECPT_ORAL_TVCOV).expect("parse 1cpt oral tvcov reset");
                let s = tvcov_subject(
                    vec![bolus(0.0), bolus(12.0)],
                    &[70.0, 90.0],
                    &[1.0, 3.0, 13.0, 15.0, 18.0],
                    &[70.0, 70.0, 90.0, 90.0, 90.0],
                    vec![12.0],
                    Vec::new(),
                    &[],
                );
                (m, s, vec![0.2, 10.0, 1.5, 0.75], vec![0.10, -0.05, 0.20])
            },
            // (d) 2-cpt IV bolus, WT changing at each observation.
            {
                let m = parse_model_string(TWOCPT_IV_TVCOV).expect("parse 2cpt iv tvcov");
                let s = tvcov_subject(
                    vec![bolus(0.0)],
                    &[70.0],
                    &[0.5, 2.0, 6.0, 12.0, 24.0],
                    &[70.0, 75.0, 82.0, 88.0, 95.0],
                    Vec::new(),
                    Vec::new(),
                    &[],
                );
                (m, s, vec![10.0, 50.0, 15.0, 100.0, 0.75], vec![0.12, -0.08])
            },
            // (e) 2-cpt IV with intermediate individual-parameter assignments and
            // an EVID=2-style covariate breakpoint. `model.pk_indices` contains
            // extra intermediate rows here; the TV-cov path must use
            // `prog.pk_slots()` to seed/scatter the four structural PK outputs.
            {
                let m = parse_model_string(TWOCPT_IV_TVCOV_INTERMEDIATE)
                    .expect("parse 2cpt iv tvcov intermediate");
                let s = tvcov_subject(
                    vec![bolus(0.0)],
                    &[70.0],
                    &[0.5, 2.0, 6.0, 12.0],
                    &[70.0, 70.0, 95.0, 95.0],
                    Vec::new(),
                    vec![3.0],
                    &[95.0],
                );
                assert!(
                    m.pk_indices.len()
                        > m.indiv_param_partials
                            .indiv_param_program
                            .as_ref()
                            .expect("compiled individual program")
                            .pk_slots()
                            .len(),
                    "fixture must expose intermediate individual-parameter rows"
                );
                (m, s, vec![10.0, 50.0, 15.0, 100.0, 0.75], vec![0.12, -0.08])
            },
            // (f) 3-cpt oral, WT changing at each observation (widest dual, M=11).
            {
                let m = parse_model_string(THREECPT_ORAL_TVCOV).expect("parse 3cpt oral tvcov");
                let s = tvcov_subject(
                    vec![bolus(0.0)],
                    &[70.0],
                    &[1.0, 2.0, 6.0, 12.0, 24.0],
                    &[70.0, 73.0, 80.0, 86.0, 92.0],
                    Vec::new(),
                    Vec::new(),
                    &[],
                );
                (
                    m,
                    s,
                    vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.5, 0.75],
                    vec![0.15, -0.10, 0.25],
                )
            },
            // (g) 1-cpt oral with a constant `obs_scale = 1000` divisor — the whole
            // jet divides by the (covariate-independent) scale.
            {
                let m = parse_model_string(ONECPT_ORAL_TVCOV_SCALED)
                    .expect("parse 1cpt oral tvcov scaled");
                assert!(
                    matches!(m.scaling, ScalingSpec::ScalarScale(k) if (k - 1000.0).abs() < 1e-9),
                    "model must carry a constant ScalarScale"
                );
                let s = tvcov_subject(
                    vec![bolus(0.0)],
                    &[70.0],
                    &[1.0, 2.0, 4.0, 8.0, 24.0],
                    &[70.0, 72.0, 80.0, 85.0, 90.0],
                    Vec::new(),
                    Vec::new(),
                    &[],
                );
                (m, s, vec![0.2, 10.0, 1.5, 0.75], vec![0.15, -0.10, 0.25])
            },
            // (h) 1-cpt IV **steady-state** bolus (II=24) with WT changing across
            // observations: the walk equilibrates the SS state per-event at the
            // dose's covariate snapshot, then the covariate switches the decay.
            {
                let m = parse_model_string(ONECPT_IV_TVCOV).expect("parse 1cpt iv tvcov ss");
                let s = tvcov_subject(
                    vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)],
                    &[70.0],
                    &[1.0, 6.0, 12.0, 18.0, 23.0],
                    &[70.0, 78.0, 86.0, 92.0, 98.0],
                    Vec::new(),
                    Vec::new(),
                    &[],
                );
                assert!(
                    s.doses.iter().any(|d| d.ss),
                    "fixture must carry an SS dose"
                );
                (m, s, vec![0.2, 10.0, 0.75], vec![0.12, -0.09])
            },
        ];
        for (m, s, theta, eta) in &cases {
            assert!(
                tvcov_analytical_supported(m),
                "TV-cov model must be provider-supported"
            );
            assert!(s.has_tv_covariates(), "fixture must carry TV covariates");
            assert!(
                subject_sensitivities(m, s, theta, eta).is_some(),
                "TV-cov subject must take the analytic provider"
            );
            check_full_provider_vs_fd(m, s, theta, eta);
        }
    }

    /// #447: the light `Dual1` inner η-gradient ([`subject_eta_grad_tvcov`]) must
    /// equal the full `Dual2` outer `df_deta` (η-block) for TV-cov subjects — both
    /// run the same event-driven walk, and the outer is FD-validated above. Covers
    /// 1-cpt oral, 2-cpt IV, and a steady-state bolus.
    #[test]
    fn tvcov_eta_grad_matches_full() {
        let bolus = |t: f64| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0);
        let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
            {
                let m = parse_model_string(ONECPT_ORAL_TVCOV).expect("parse 1cpt oral tvcov");
                let s = tvcov_subject(
                    vec![bolus(0.0)],
                    &[70.0],
                    &[1.0, 2.0, 4.0, 8.0, 24.0],
                    &[70.0, 72.0, 80.0, 85.0, 90.0],
                    Vec::new(),
                    Vec::new(),
                    &[],
                );
                (m, s, vec![0.2, 10.0, 1.5, 0.75], vec![0.15, -0.10, 0.25])
            },
            {
                let m = parse_model_string(TWOCPT_IV_TVCOV).expect("parse 2cpt iv tvcov");
                let s = tvcov_subject(
                    vec![bolus(0.0)],
                    &[70.0],
                    &[0.5, 2.0, 6.0, 12.0, 24.0],
                    &[70.0, 75.0, 82.0, 88.0, 95.0],
                    Vec::new(),
                    Vec::new(),
                    &[],
                );
                (m, s, vec![10.0, 50.0, 15.0, 100.0, 0.75], vec![0.12, -0.08])
            },
            {
                let m = parse_model_string(TWOCPT_IV_TVCOV_INTERMEDIATE)
                    .expect("parse 2cpt iv tvcov intermediate");
                let s = tvcov_subject(
                    vec![bolus(0.0)],
                    &[70.0],
                    &[0.5, 2.0, 6.0, 12.0],
                    &[70.0, 70.0, 95.0, 95.0],
                    Vec::new(),
                    vec![3.0],
                    &[95.0],
                );
                (m, s, vec![10.0, 50.0, 15.0, 100.0, 0.75], vec![0.12, -0.08])
            },
            {
                let m = parse_model_string(ONECPT_IV_TVCOV).expect("parse 1cpt iv tvcov ss");
                let s = tvcov_subject(
                    vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)],
                    &[70.0],
                    &[1.0, 6.0, 12.0, 18.0, 23.0],
                    &[70.0, 78.0, 86.0, 92.0, 98.0],
                    Vec::new(),
                    Vec::new(),
                    &[],
                );
                (m, s, vec![0.2, 10.0, 0.75], vec![0.12, -0.09])
            },
            {
                // Constant `ScalarScale` (`obs_scale = 1000`) on the TV-cov **inner**:
                // exercises `run_obs_grad_tvcov`'s `∂(f/k)/∂η = (∂f/∂η)/k` division,
                // which the other inner cases (no output scaling) leave uncovered
                // (#451 / #449 review #10).
                let m = parse_model_string(ONECPT_ORAL_TVCOV_SCALED)
                    .expect("parse 1cpt oral tvcov scaled");
                let s = tvcov_subject(
                    vec![bolus(0.0)],
                    &[70.0],
                    &[1.0, 2.0, 4.0, 8.0, 24.0],
                    &[70.0, 72.0, 80.0, 85.0, 90.0],
                    Vec::new(),
                    Vec::new(),
                    &[],
                );
                (m, s, vec![0.2, 10.0, 1.5, 0.75], vec![0.15, -0.10, 0.25])
            },
        ];
        for (model, subject, theta, eta) in &cases {
            let full =
                subject_sensitivities_tvcov(model, subject, theta, eta).expect("outer tvcov");
            let light =
                subject_eta_grad_tvcov(model, subject, theta, eta).expect("light tvcov inner");
            assert_eq!(full.obs.len(), light.len());
            for (a, b) in full.obs.iter().zip(light.iter()) {
                approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-12, epsilon = 1e-12);
                for k in 0..model.n_eta {
                    approx::assert_relative_eq!(
                        a.df_deta[k],
                        b.df_deta[k],
                        max_relative = 1e-10,
                        epsilon = 1e-11
                    );
                }
            }
        }
    }

    // ── IOV analytic sensitivities ───────────────────────────────────

    const WARFARIN_IOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    /// Two-occasion IOV subject: a dose + observations in occasion 1, then a dose +
    /// observations in occasion 2 (no washout — carryover spans the boundary).
    fn iov_subject() -> Subject {
        let obs_times = vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0];
        let occasions = vec![1u32, 1, 1, 2, 2, 2];
        let n = obs_times.len();
        Subject {
            id: "1".to_string(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times,
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions,
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    const WARFARIN_IOV_2CPT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVQ(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk two_cpt_oral(cl=CL, v=V, q=Q, v2=V2, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    /// Shared FD-vs-predict_iov check for a two-occasion IOV model: value, gradient,
    /// and Hessian over the stacked random-effects vector `[η_bsv, κ_g0, κ_g1]` and
    /// the θ block must match central differences of the production `predict_iov`
    /// (an independent f64 path), validating the whole walk + (η,κ,θ) chain.
    fn check_iov_provider_vs_fd(
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        stacked: &[f64],
    ) {
        let n_eta = model.n_eta;
        let n_kappa = model.n_kappa;
        let n_theta = theta.len();
        let k_groups = crate::stats::likelihood::iov_occasion_groups(subject).len();
        assert_eq!(
            stacked.len(),
            n_eta + k_groups * n_kappa,
            "stacked vector must match IOV occasion groups"
        );
        let sens = subject_sensitivities_iov(model, subject, theta, stacked).expect("supported");

        // Map a stacked-η vector to predict_iov's (η_bsv, kappas-per-group) form.
        let pred = |st: &[f64], th: &[f64], j: usize| -> f64 {
            let eta_bsv = st[..n_eta].to_vec();
            let kappas: Vec<Vec<f64>> = (0..k_groups)
                .map(|g| {
                    let base = n_eta + g * n_kappa;
                    st[base..base + n_kappa].to_vec()
                })
                .collect();
            crate::pk::predict_iov(model, subject, th, &eta_bsv, &kappas)[j]
        };

        let theta = theta.to_vec();
        let stacked = stacked.to_vec();
        let n_st = stacked.len();
        let he = 1e-6;
        let heh = 1e-4;
        for (j, obs) in sens.obs.iter().enumerate() {
            approx::assert_relative_eq!(
                obs.f,
                pred(&stacked, &theta, j),
                max_relative = 1e-9,
                epsilon = 1e-12
            );
            // ∂f/∂stacked and ∂²f/∂stacked².
            for k in 0..n_st {
                let mut sp = stacked.clone();
                sp[k] += he;
                let mut sm = stacked.clone();
                sm[k] -= he;
                let g = (pred(&sp, &theta, j) - pred(&sm, &theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 2e-4, epsilon = 1e-7);
                for l in 0..n_st {
                    let mut pp = stacked.clone();
                    pp[k] += heh;
                    pp[l] += heh;
                    let mut pm = stacked.clone();
                    pm[k] += heh;
                    pm[l] -= heh;
                    let mut mp = stacked.clone();
                    mp[k] -= heh;
                    mp[l] += heh;
                    let mut mm = stacked.clone();
                    mm[k] -= heh;
                    mm[l] -= heh;
                    let hh = (pred(&pp, &theta, j) - pred(&pm, &theta, j) - pred(&mp, &theta, j)
                        + pred(&mm, &theta, j))
                        / (4.0 * heh * heh);
                    approx::assert_relative_eq!(
                        obs.d2f_deta2[k * n_st + l],
                        hh,
                        max_relative = 3e-3,
                        epsilon = 1e-5
                    );
                }
            }
            // ∂f/∂θ and ∂²f/∂stacked∂θ.
            for m in 0..n_theta {
                let s = he * (1.0 + theta[m].abs());
                let mut tp = theta.clone();
                tp[m] += s;
                let mut tm = theta.clone();
                tm[m] -= s;
                let g = (pred(&stacked, &tp, j) - pred(&stacked, &tm, j)) / (2.0 * s);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 2e-4,
                    epsilon = 1e-7
                );
                for k in 0..n_st {
                    let sh = heh * (1.0 + theta[m].abs());
                    let mut ep = stacked.clone();
                    ep[k] += heh;
                    let mut em = stacked.clone();
                    em[k] -= heh;
                    let mut tp2 = theta.clone();
                    tp2[m] += sh;
                    let mut tm2 = theta.clone();
                    tm2[m] -= sh;
                    let hh = (pred(&ep, &tp2, j) - pred(&ep, &tm2, j) - pred(&em, &tp2, j)
                        + pred(&em, &tm2, j))
                        / (4.0 * heh * sh);
                    approx::assert_relative_eq!(
                        obs.d2f_deta_dtheta[k * n_theta + m],
                        hh,
                        max_relative = 3e-3,
                        epsilon = 1e-5
                    );
                }
            }
        }
    }

    /// 1-cpt oral IOV: provider == FD of `predict_iov` over `[η_bsv, κ_g0, κ_g1]`.
    #[test]
    fn iov_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
        assert_eq!(model.n_kappa, 1, "model must carry one kappa");
        assert!(
            iov_analytical_supported(&model),
            "warfarin IOV must be IOV-provider supported"
        );
        let subject = iov_subject();
        // stacked = [η_cl, η_v, η_ka, κ_g0, κ_g1].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 1.5],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
    }

    /// The closed-form **inner** IOV walk (`Dual1`, `subject_eta_grad_iov_analytical`)
    /// must produce the same per-observation value and `∂f/∂(stacked-η)` as the
    /// **outer** walk (`Dual2`, `subject_sensitivities_iov`) — whose `df_deta` is
    /// already validated against FD of `predict_iov`. Confirms the new first-order
    /// analytical IOV inner agrees with the FD-validated second-order path (#439
    /// closed-form IOV inner).
    #[test]
    fn analytical_iov_inner_eta_grad_matches_outer() {
        // 1-/2-/3-cpt oral closed-form IOV — the new Dual1 inner must track the
        // FD-validated Dual2 outer's first-order block on each.
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV).expect("parse 1cpt"),
            &iov_subject(),
            &[0.2, 10.0, 1.5],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV_2CPT).expect("parse 2cpt"),
            &iov_subject(),
            &[0.2, 10.0, 0.5, 20.0, 1.5],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV_3CPT).expect("parse 3cpt"),
            &iov_subject(),
            &[0.2, 10.0, 0.5, 20.0, 0.3, 50.0, 1.5],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
    }

    /// 1-cpt IV IOV written as a user `[odes]` model (κ on CL, Form-C readout
    /// `y = central/V`). Routes through `subject_sensitivities_iov` → the ODE IOV
    /// provider, validated against central FD of the production `predict_iov` (which
    /// integrates the same ODE via `ode_predictions_event_driven`). This is the ODE
    /// counterpart of `iov_provider_matches_fd_of_predict_iov`, proving the
    /// per-occasion κ-axis seeding + event-driven dual walk compose the exact stacked
    /// (η_bsv, κ, θ) gradient (#439 ODE IOV).
    const WARFARIN_IOV_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    #[test]
    fn ode_iov_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
        assert_eq!(model.n_kappa, 1, "model must carry one kappa");
        assert!(model.ode_spec.is_some(), "must be an ODE model");
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "1-cpt IV ODE IOV must be ODE-IOV-provider supported"
        );
        let subject = iov_subject();
        // stacked = [η_cl, η_v, κ_g0, κ_g1] (n_eta = 2, n_kappa = 1, K = 2).
        check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
    }

    #[test]
    fn ode_iov_above_legacy_axis_cap_stays_analytic() {
        let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
        let n_occ = 15;
        let obs_times: Vec<f64> = (0..n_occ).map(|i| i as f64 * 24.0 + 1.0).collect();
        let occasions: Vec<u32> = (1..=n_occ as u32).collect();
        let doses: Vec<DoseEvent> = (0..n_occ)
            .map(|i| DoseEvent::new(i as f64 * 24.0, 100.0, 1, 0.0, false, 0.0))
            .collect();
        let n = obs_times.len();
        let subject = Subject {
            id: "wide-iov".to_string(),
            doses,
            obs_times,
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions,
            dose_occasions: (1..=n_occ as u32).collect(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let theta = vec![0.2, 10.0];
        let mut stacked = vec![0.0; model.n_eta + n_occ * model.n_kappa];
        stacked[0] = 0.12;
        stacked[1] = -0.08;
        for g in 0..n_occ {
            stacked[model.n_eta + g] = 0.02 * (g as f64 - 7.0);
        }
        let m_dim = model.n_theta + stacked.len();
        assert!(
            m_dim > crate::sens::ode_provider::MAX_ODE_AXES,
            "fixture must exceed the legacy 16-axis cap"
        );
        assert!(
            m_dim <= crate::sens::ode_provider::MAX_ODE_IOV_AXES,
            "fixture must stay within the widened IOV cap"
        );
        let full = crate::sens::ode_provider::ode_subject_sensitivities_iov(
            &model, &subject, &theta, &stacked,
        )
        .expect("wide ODE IOV outer gradient should be analytic");
        let light =
            crate::sens::ode_provider::ode_subject_eta_grad_iov(&model, &subject, &theta, &stacked)
                .expect("wide ODE IOV inner gradient should be analytic");
        assert_eq!(full.obs.len(), light.len());
        for (outer, inner) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(outer.f, inner.f, max_relative = 1e-10, epsilon = 1e-12);
            for k in 0..stacked.len() {
                approx::assert_relative_eq!(
                    outer.df_deta[k],
                    inner.df_deta[k],
                    max_relative = 1e-8,
                    epsilon = 1e-10
                );
            }
        }
    }

    /// Regression guard for the ODE IOV worker-stack overflow (#601): a PNA-scale,
    /// 86-occasion subject yields a `Dual2<90>` (90×90 Hessian per dual) whose
    /// event-walk frames overflow the platform-default (~2 MiB) Rayon worker stack. The
    /// gradient is run on [`crate::api::default_fit_pool`] — the *same* pool `fit()` uses
    /// by default — so dropping the 32 MiB stack from that pool re-introduces the crash
    /// here. Heavy (full wide-`M` sensitivity through RK45), so it is gated to the
    /// nightly slow-tests tier rather than the fast per-PR job.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn fit_rayon_stack_handles_pna_scale_ode_iov_gradient() {
        let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
        let n_occ = 86;
        let obs_times: Vec<f64> = (0..n_occ).map(|i| i as f64 * 24.0 + 1.0).collect();
        let occasions: Vec<u32> = (1..=n_occ as u32).collect();
        let doses: Vec<DoseEvent> = (0..n_occ)
            .map(|i| DoseEvent::new(i as f64 * 24.0, 100.0, 1, 0.0, false, 0.0))
            .collect();
        let n = obs_times.len();
        let subject = Subject {
            id: "pna-scale-iov".to_string(),
            doses,
            obs_times,
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions,
            dose_occasions: (1..=n_occ as u32).collect(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let theta = vec![0.2, 10.0];
        let stacked = vec![0.0; model.n_eta + n_occ * model.n_kappa];
        let m_dim = model.n_theta + stacked.len();
        assert_eq!(m_dim, 90, "fixture mirrors the PNA-scale occasion width");

        // Run on the actual default fit pool, so a regression that drops the big stack
        // from `default_fit_pool` (or `fit_thread_pool_builder`) overflows here.
        let pool = crate::api::default_fit_pool().expect("ferx default fit pool");
        pool.install(|| {
            crate::sens::ode_provider::ode_subject_sensitivities_iov(
                &model, &subject, &theta, &stacked,
            )
            .expect("PNA-scale ODE IOV gradient should fit on ferx worker stack");
        });
    }

    /// A dose in an occasion that carries no sampled observations still gets its own κ
    /// axis. That kappa can affect later observations through carryover, so the ODE IOV
    /// provider must keep the subject on the analytic path rather than falling back to FD.
    #[test]
    fn ode_iov_dose_only_occasion_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
        let mut subject = iov_subject();
        // Add a dose between the two observed occasions. Occasion 3 has no observations, but
        // its CL kappa affects the post-dose amount that carries into occasion 2.
        subject
            .doses
            .push(DoseEvent::new(18.0, 100.0, 1, 0.0, false, 0.0));
        subject.dose_occasions.push(3);
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0],
            &[0.12, -0.08, 0.05, -0.10, 0.08],
        );
    }

    /// **ODE IOV + EVID 3/4 reset.** A two-occasion ODE IOV subject with a washout reset
    /// (+ re-dose) at the occasion boundary. The event-driven walk zeros the dual state at
    /// the reset (no cross-occasion carryover) and the per-occasion κ seeding continues on
    /// the post-reset occasion. Validated vs FD of `predict_iov` (#439 IOV × reset).
    #[test]
    fn ode_iov_reset_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        let subject = iov_reset_subject();
        assert!(subject.has_resets());
        check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
    }

    /// **ODE IOV + infusion.** Two-occasion IOV with finite-duration infusions; the
    /// event-driven walk applies the per-occasion `F·rate` forcing over each window.
    /// Validated vs FD of `predict_iov` (#439 IOV × infusion).
    #[test]
    fn ode_iov_infusion_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        let mut subject = iov_subject();
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 50.0, false, 0.0),
        ];
        assert!(subject.doses[0].is_infusion());
        check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
    }

    /// Regression (#575 review): a plain `ScalingSpec::None` IOV ODE model under LTBS
    /// (`log_transform`, no `obs_scale`) must stay on FD. The #575 gate rewrite replaced
    /// the `|| model.log_transform` bail with a `match`, and the `None` arm initially
    /// admitted LTBS — re-routing IOV + LTBS onto the analytic IOV walk, whose in-PK-param
    /// log can't reproduce the production scale-then-log order. The `None` arm carries an
    /// explicit `if !model.log_transform` guard; this pins it.
    #[test]
    fn ode_iov_ltbs_no_scale_falls_back_to_fd() {
        let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
        assert!(
            matches!(model.scaling, ScalingSpec::None),
            "base model must have no obs_scale (None scaling)"
        );
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "non-LTBS None-scaling IOV ODE is analytic"
        );
        let mut ltbs = model;
        ltbs.log_transform = true;
        assert!(
            !crate::sens::ode_provider::ode_iov_supported(&ltbs),
            "IOV + LTBS (None scaling) must fall back to FD"
        );
    }

    /// 1-cpt IV ODE IOV with an η-dependent `ExpressionScale` `obs_scale = V` divisor
    /// (κ on CL, `ObsCmt` readout on the amount). The post-walk per-occasion-group
    /// quotient (#575) must reproduce central FD of `predict_iov`, which applies the
    /// same divisor per occasion (κ-aware). The ExpressionScale counterpart of
    /// `ode_iov_provider_matches_fd_of_predict_iov`.
    const WARFARIN_IOV_ODE_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    #[test]
    fn ode_iov_expr_scale_supported_and_gated() {
        let model = parse_model_string(WARFARIN_IOV_ODE_EXPRSCALE).expect("parse expr-scale IOV");
        assert_eq!(model.n_kappa, 1);
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "ODE IOV + ExpressionScale obs_scale must be on the analytic path (#575)"
        );
        // + LTBS still routes to FD (the in-walk log transform is not composed with the
        // post-walk quotient on the IOV path).
        let mut ltbs = parse_model_string(WARFARIN_IOV_ODE_EXPRSCALE).expect("parse");
        ltbs.log_transform = true;
        assert!(
            !crate::sens::ode_provider::ode_iov_supported(&ltbs),
            "ExpressionScale + LTBS under IOV must fall back to FD"
        );
    }

    #[test]
    fn ode_iov_expr_scale_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_ODE_EXPRSCALE).expect("parse expr-scale IOV");
        let subject = iov_subject();
        // stacked = [η_cl, η_v, κ_g0, κ_g1] (n_eta = 2, n_kappa = 1, K = 2).
        check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
    }

    #[test]
    fn ode_iov_expr_scale_inner_eta_grad_matches_outer() {
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV_ODE_EXPRSCALE).expect("parse expr-scale IOV"),
            &iov_subject(),
            &[0.2, 10.0],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    /// The `obs_scale = V` divisor form is a numerical twin of the Form-C readout
    /// `y = central / V` (already analytic + FD-validated): both compute `central/V`
    /// and its exact stacked-(η,κ,θ) sensitivities, the divisor post-walk and the
    /// readout in-walk. Per-observation value and `∂f/∂stacked` must agree (#575).
    #[test]
    fn ode_iov_expr_scale_equals_formc_readout() {
        let divisor = parse_model_string(WARFARIN_IOV_ODE_EXPRSCALE).expect("parse divisor");
        let formc = parse_model_string(WARFARIN_IOV_ODE).expect("parse Form-C");
        let subject = iov_subject();
        let theta = [0.2, 10.0];
        let stacked = [0.12, -0.08, 0.05, -0.10];
        let a = subject_sensitivities_iov(&divisor, &subject, &theta, &stacked).expect("divisor");
        let b = subject_sensitivities_iov(&formc, &subject, &theta, &stacked).expect("formc");
        assert_eq!(a.obs.len(), b.obs.len());
        for (oa, ob) in a.obs.iter().zip(b.obs.iter()) {
            approx::assert_relative_eq!(oa.f, ob.f, max_relative = 1e-8, epsilon = 1e-10);
            for (x, y) in oa.df_deta.iter().zip(ob.df_deta.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-7, epsilon = 1e-9);
            }
        }
    }

    /// **ODE IOV + steady-state bolus.** Each occasion's SS dose equilibrates with that
    /// occasion's κ-seeded params (dual SS-equilibration), then the per-occasion walk
    /// continues. Validated vs FD of `predict_iov` (#439 IOV × SS).
    #[test]
    fn ode_iov_ss_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        let mut subject = iov_subject();
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, true, 12.0),
        ];
        assert!(subject.doses[0].ss && subject.doses[0].ii > 0.0);
        check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
    }

    /// **ODE IOV + rate-defined infusion under bioavailability `F ≠ 1`** (#419 × IOV). The
    /// bioavailable window length `F·amt/rate` is a moving rate-off boundary per occasion;
    /// the event-driven walk carries it with the rate held. Validated vs FD of `predict_iov`.
    #[test]
    fn ode_iov_rate_defined_infusion_under_f_matches_fd_of_predict_iov() {
        // WARFARIN IOV ODE with a bioavailability parameter `F`.
        const WARFARIN_IOV_F_ODE: &str = r#"
[parameters]
  theta TVCL(0.13, 0.01, 1.0)
  theta TVV(8.0, 1.0, 50.0)
  theta TVF(0.7, 0.05, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  iov_column OCC
  kappa KAPPA_CL ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV * exp(ETA_V)
  F  = TVF
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
        let model = parse_model_string(WARFARIN_IOV_F_ODE).expect("parse IOV+F ODE");
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        let mut subject = iov_subject();
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 50.0, false, 0.0),
        ];
        assert!(subject.has_rate_defined_infusion());
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.13, 8.0, 0.7],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    /// **IOV × estimated lagtime.** 1-cpt IV IOV `[odes]` model (κ on CL) with a bare
    /// `LAGTIME`. The dose arrives per occasion at `t_dose + lag`; the lag sensitivity is
    /// the event-time saltation injected at each dose and propagated through the
    /// occasion-switching event-driven walk (`integrate_tvcov_g`, shared with the TV-cov
    /// path). Validates the full stacked-η + θ (incl. the `TVLAG` column) gradient and
    /// Hessian against FD of `predict_iov`, which handles IOV + lagtime in production
    /// (#439 lagtime × IOV).
    const WARFARIN_IOV_LAG_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  LAGTIME = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    #[test]
    fn ode_iov_lagtime_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_LAG_ODE).expect("parse ODE IOV+lag");
        assert_eq!(model.n_kappa, 1);
        assert!(model.has_lagtime());
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "ODE IOV + bare lagtime must be supported"
        );
        let subject = iov_subject();
        // stacked = [η_cl, η_v, κ_g0, κ_g1]; θ = [TVCL, TVV, TVLAG].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.5],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    /// **IOV × lagtime × infusion × reset** — the combined path through `integrate_tvcov_g`
    /// (both rate-on/off saltations, the `reset_floor` guard, and per-occasion κ seeding) is
    /// otherwise covered only piecewise. Full stacked-η + θ gradient and Hessian vs FD of
    /// `predict_iov` (#472 review round 2 #5).
    #[test]
    fn ode_iov_lagtime_infusion_reset_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_LAG_ODE).expect("parse ODE IOV+lag");
        assert!(model.has_lagtime());
        let mut subject = iov_reset_subject(); // 2 occasions, EVID=4 reset at t=24
                                               // Per-occasion infusions (rate>0): occ-1 window starts at 0, occ-2 re-dose at the
                                               // reset; the lagtime shifts both windows.
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 50.0, false, 0.0),
        ];
        assert!(subject.has_resets() && subject.doses[0].is_infusion());
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.5],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    /// 1-cpt IV IOV `[odes]` model with a WT covariate on CL (`(WT/70)^θ_WT`) under
    /// **time-varying covariates**: each event's PK params are seeded at its own
    /// (occasion, WT-snapshot), so the individual CL switches both by κ and by WT.
    /// Validates the per-event IOV+TV-cov seeding vs FD of `predict_iov` (#439 ODE IOV).
    const WARFARIN_IOV_TVCOV_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    /// Same as `WARFARIN_IOV_TVCOV_ODE`, but with the readout expressed as a
    /// post-walk divisor instead of Form C. The subject has time-varying covariates
    /// in the event walk, and the `obs_scale = CL` quotient references **both** the
    /// TV covariate `WT` and `KAPPA_CL` — so the scale jet would *differ* if it read
    /// per-event covariates (it must use the subject-static snapshot, like production
    /// `predict_iov`'s `apply_scaling`) and *differs per occasion group* via κ. A
    /// covariate- and κ-free scale (`obs_scale = V`) could not distinguish those (#590).
    const WARFARIN_IOV_TVCOV_ODE_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = CL
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    #[test]
    fn ode_iov_tvcov_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_TVCOV_ODE).expect("parse ODE IOV+TV-cov");
        assert_eq!(model.n_kappa, 1);
        assert!(model.ode_spec.is_some());
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "1-cpt IV ODE IOV+TV-cov must be ODE-IOV-provider supported"
        );
        let subject = iov_tvcov_subject(false);
        assert!(
            subject.has_tv_covariates(),
            "subject must carry TV covariates"
        );
        // stacked = [η_cl, η_v, κ_g0, κ_g1]; θ = [TVCL, TVV, THETA_WT].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.75],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    #[test]
    fn ode_iov_tvcov_expr_scale_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_TVCOV_ODE_EXPRSCALE)
            .expect("parse ODE IOV+TV-cov+expr-scale");
        assert_eq!(model.n_kappa, 1);
        assert!(model.ode_spec.is_some());
        assert!(
            matches!(model.scaling, ScalingSpec::ExpressionScale { .. }),
            "fixture must use an expression obs_scale"
        );
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "ODE IOV + TV covariates + ExpressionScale must stay analytic (#590)"
        );
        let subject = iov_tvcov_subject(false);
        assert!(
            subject.has_tv_covariates(),
            "subject must carry TV covariates"
        );
        // θ = [TVCL, TVV, THETA_WT]; stacked = [η_cl, η_v, κ_g0, κ_g1].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.75],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    #[test]
    fn ode_iov_tvcov_expr_scale_inner_eta_grad_matches_outer() {
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV_TVCOV_ODE_EXPRSCALE)
                .expect("parse ODE IOV+TV-cov+expr-scale"),
            &iov_tvcov_subject(false),
            &[0.2, 10.0, 0.75],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    #[test]
    fn ode_iov_tvcov_pkonly_breakpoint_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_TVCOV_ODE).expect("parse ODE IOV+TV-cov");
        let subject = iov_tvcov_subject(true);
        assert!(
            !subject.pk_only_times.is_empty(),
            "fixture must carry an EVID=2 covariate breakpoint"
        );
        assert!(subject.has_tv_covariates(), "fixture must carry TV cov");
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "model-level ODE IOV gate must admit the fixture"
        );
        assert!(
            subject_sensitivities_iov(
                &model,
                &subject,
                &[0.2, 10.0, 0.75],
                &[0.12, -0.08, 0.05, -0.10],
            )
            .is_some(),
            "EVID=2 breakpoint must stay on the analytic ODE IOV path (#590)"
        );
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.75],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    #[test]
    fn ode_iov_tvcov_pkonly_inner_eta_grad_matches_outer() {
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV_TVCOV_ODE).expect("parse ODE IOV+TV-cov"),
            &iov_tvcov_subject(true),
            &[0.2, 10.0, 0.75],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    /// **Static-covariate** ODE IOV subject with an EVID=2 pk-only breakpoint. The TV-cov
    /// pk-only tests above always take `seed_iov_events`' per-event (TV) branch; a subject
    /// with no dose/obs TV covariates **and an empty `pk_only_covariates`** instead takes the
    /// static-cov else-branch — the pk-only event is seeded once at the subject-static snapshot
    /// and shared (`vec![seeded; len]`). This is reachable in production (e.g. a pk-only
    /// breakpoint whose covariate was pruned as irrelevant while its time remained), so it must
    /// stay analytic and match FD of `predict_iov` (#598 review — covers the else-branch).
    #[test]
    fn ode_iov_static_cov_pkonly_breakpoint_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
        let mut subject = iov_subject();
        // EVID=2 breakpoint at t=18 (occasion 2), no covariate snapshot → static-cov path.
        subject.pk_only_times = vec![18.0];
        assert!(
            subject.pk_only_covariates.is_empty() && !subject.has_tv_covariates(),
            "fixture must hit the static-cov pk-only branch (no TV cov, empty pk_only_covariates)"
        );
        assert!(
            crate::sens::ode_provider::ode_subject_sensitivities_iov(
                &model,
                &subject,
                &[0.2, 10.0],
                &[0.12, -0.08, 0.05, -0.10],
            )
            .is_some(),
            "static-cov EVID=2 breakpoint must stay on the analytic ODE IOV path"
        );
        check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
    }

    /// Inner η-gradient parity for the same static-cov pk-only subject — exercises the
    /// `Dual1` seeder's static-cov else-branch (#598 review).
    #[test]
    fn ode_iov_static_cov_pkonly_inner_eta_grad_matches_outer() {
        let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
        let mut subject = iov_subject();
        subject.pk_only_times = vec![18.0];
        check_iov_inner_matches_outer(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
    }

    /// 2-cpt IV IOV `[odes]` model (κ on CL) — higher state/axis coverage for the ODE
    /// IOV walk: stacked dual width M = n_θ(4) + n_η(2) + K(2)·n_κ(1) = 8 (#439 ODE IOV).
    const WARFARIN_IOV_2CPT_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVQ(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V) * central - (Q/V) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    #[test]
    fn ode_iov_2cpt_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_2CPT_ODE).expect("parse 2cpt ODE IOV");
        assert_eq!(model.n_kappa, 1);
        assert!(model.ode_spec.is_some());
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        let subject = iov_subject();
        // stacked = [η_cl, η_v, κ_g0, κ_g1]; θ = [TVCL, TVV, TVQ, TVV2].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.5, 20.0],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    /// 1-cpt **oral** IOV `[odes]` model (depot → central first-order absorption, κ on
    /// CL): bolus into the depot (cmt 1), readout `central/V`. Exercises the multi-state
    /// absorption ODE under per-occasion κ seeding (#439 ODE IOV).
    const WARFARIN_IOV_ORAL_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    #[test]
    fn ode_iov_oral_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_ORAL_ODE).expect("parse oral ODE IOV");
        assert_eq!(model.n_kappa, 1);
        assert!(model.ode_spec.is_some());
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        let subject = iov_subject();
        // stacked = [η_cl, η_v, η_ka, κ_g0, κ_g1]; θ = [TVCL, TVV, TVKA].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 1.5],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
    }

    /// 3-cpt IV IOV `[odes]` model (κ on CL) — highest state/axis coverage for the ODE
    /// IOV walk: 3 ODE states, dual width M = n_θ(6) + n_η(2) + K(2)·n_κ(1) = 10
    /// (#439 ODE IOV).
    const WARFARIN_IOV_3CPT_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVQ2(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  theta TVQ3(0.3, 0.001, 50.0)
  theta TVV3(50.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  Q  = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
[structural_model]
  ode(states=[central, periph1, periph2])
[odes]
  d/dt(central) = -(CL/V)*central - (Q/V)*central + (Q/V2)*periph1 - (Q3/V)*central + (Q3/V3)*periph2
  d/dt(periph1) =  (Q/V)*central - (Q/V2)*periph1
  d/dt(periph2) =  (Q3/V)*central - (Q3/V3)*periph2
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    #[test]
    fn ode_iov_3cpt_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_3CPT_ODE).expect("parse 3cpt ODE IOV");
        assert_eq!(model.n_kappa, 1);
        assert!(model.ode_spec.is_some());
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        let subject = iov_subject();
        // stacked = [η_cl, η_v, κ_g0, κ_g1]; θ = [TVCL, TVV, TVQ2, TVV2, TVQ3, TVV3].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.5, 20.0, 0.3, 50.0],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    /// The light **inner** IOV walk (`Dual1`, via the `subject_eta_grad_iov` dispatch —
    /// ODE or closed-form) must produce the same per-observation value and
    /// `∂f/∂(stacked-η)` as the **outer** walk (`Dual2`, `subject_sensitivities_iov`),
    /// whose `df_deta` is already validated against FD of `predict_iov`. A
    /// bit-for-bit-close cross-check that the first-order seeding/readout is consistent
    /// across the two dual orders, for both providers (#439 IOV inner).
    fn check_iov_inner_matches_outer(
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        stacked: &[f64],
    ) {
        let outer =
            subject_sensitivities_iov(model, subject, theta, stacked).expect("outer IOV supported");
        let inner =
            subject_eta_grad_iov(model, subject, theta, stacked).expect("inner IOV supported");
        assert_eq!(outer.obs.len(), inner.len());
        for (o, i) in outer.obs.iter().zip(inner.iter()) {
            approx::assert_relative_eq!(o.f, i.f, max_relative = 1e-12, epsilon = 1e-12);
            assert_eq!(o.df_deta.len(), i.df_deta.len());
            for (a, b) in o.df_deta.iter().zip(i.df_deta.iter()) {
                approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-11);
            }
        }
    }

    #[test]
    fn ode_iov_inner_eta_grad_matches_outer() {
        // 1-cpt IV, oral, and TV-cov variants — the inner Dual1 walk must track the
        // FD-validated outer Dual2 walk's first-order block on each.
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV_ODE).expect("parse"),
            &iov_subject(),
            &[0.2, 10.0],
            &[0.12, -0.08, 0.05, -0.10],
        );
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV_ORAL_ODE).expect("parse"),
            &iov_subject(),
            &[0.2, 10.0, 1.5],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV_TVCOV_ODE).expect("parse"),
            &iov_tvcov_subject(false),
            &[0.2, 10.0, 0.75],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    /// Two-occasion IOV subject with a washout: an EVID=4 reset at t=24 zeros the
    /// state and opens occasion 2, so there is NO carryover across the boundary
    /// (the complement of `iov_subject`, which carries occasion-1 amounts forward).
    /// Exercises the walk's reset handling under per-occasion κ seeding.
    fn iov_reset_subject() -> Subject {
        let obs_times = vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0];
        let occasions = vec![1u32, 1, 1, 2, 2, 2];
        let n = obs_times.len();
        Subject {
            id: "1".to_string(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times,
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: vec![24.0],
            cens: vec![0; n],
            occasions,
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    /// 1-cpt oral IOV **with an EVID=4 washout reset** at the occasion boundary:
    /// the provider's value/grad/Hessian over `[η_bsv, κ_g0, κ_g1]` + θ must still
    /// match FD of `predict_iov` (which routes the reset through the same
    /// event-driven walk). Confirms ungating resets in `subject_sensitivities_iov`
    /// keeps the (η, κ, θ) chain exact across the reset.
    #[test]
    fn iov_provider_with_reset_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
        let subject = iov_reset_subject();
        assert!(subject.has_resets(), "fixture must carry a reset");
        assert!(
            subject_sensitivities_iov(
                &model,
                &subject,
                &[0.2, 10.0, 1.5],
                &[0.1, 0.0, 0.1, 0.0, 0.0]
            )
            .is_some(),
            "IOV + reset subject must be analytic-supported"
        );
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 1.5],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
    }

    /// 2-cpt oral IOV: same FD check, exercising the generic 2-cpt event-driven
    /// sensitivity walk (eigen-decomposition propagators) under occasion carryover.
    #[test]
    fn iov_provider_2cpt_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_2CPT).expect("parse 2cpt warfarin IOV");
        assert_eq!(model.n_kappa, 1);
        assert!(
            iov_analytical_supported(&model),
            "2-cpt warfarin IOV must be IOV-provider supported"
        );
        let subject = iov_subject();
        // θ = [TVCL, TVV, TVQ, TVV2, TVKA]; stacked = [η_cl, η_v, η_ka, κ_g0, κ_g1].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.5, 20.0, 1.5],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
    }

    const WARFARIN_IOV_3CPT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVQ2(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  theta TVQ3(0.3, 0.001, 50.0)
  theta TVV3(50.0, 0.1, 1000.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  Q  = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk three_cpt_oral(cl=CL, v=V, q=Q, v2=V2, q3=Q3, v3=V3, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    /// 3-cpt oral IOV: same FD check, exercising the generic 3-cpt eigenmode
    /// event-driven sensitivity walk under occasion carryover.
    #[test]
    fn iov_provider_3cpt_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_3CPT).expect("parse 3cpt warfarin IOV");
        assert_eq!(model.n_kappa, 1);
        assert!(
            iov_analytical_supported(&model),
            "3-cpt warfarin IOV must be IOV-provider supported"
        );
        let subject = iov_subject();
        // θ = [TVCL,TVV,TVQ2,TVV2,TVQ3,TVV3,TVKA]; stacked = [η_cl,η_v,η_ka,κ_g0,κ_g1].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.5, 20.0, 0.3, 50.0, 1.5],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
    }

    // ── IOV combined with time-varying covariates ────────────────────────
    //
    // These models carry BOTH a kappa (IOV) and a WT-on-CL covariate that varies
    // within the subject. Each event's PK-param duals must be seeded at that
    // event's covariate snapshot *and* at the right occasion's κ — the per-event
    // `sources` refactor in `subject_sensitivities_iov`. The FD reference is the
    // production `predict_iov`, which already seeds per-event covariates, so the
    // check validates the merged (η_bsv, κ, θ, WT) chain end to end.

    const WARFARIN_IOV_TVCOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    const WARFARIN_IOV_TVCOV_2CPT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVQ(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk two_cpt_oral(cl=CL, v=V, q=Q, v2=V2, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    const WARFARIN_IOV_TVCOV_3CPT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVQ2(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  theta TVQ3(0.3, 0.001, 50.0)
  theta TVV3(50.0, 0.1, 1000.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  Q  = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk three_cpt_oral(cl=CL, v=V, q=Q, v2=V2, q3=Q3, v3=V3, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    /// Two-occasion IOV subject with a WT covariate that varies across records
    /// (occasion-1 doses/obs at a lighter weight, occasion-2 heavier), so the
    /// individual `CL` switches both by κ (occasion) and by WT (covariate). When
    /// `pk_only` is set, an EVID=2 covariate breakpoint (WT jump at t=18, no
    /// occasion) sits between the occasion-2 observations — exercising the κ=0
    /// `pk_only` source on the IOV+TV-cov path.
    fn iov_tvcov_subject(pk_only: bool) -> Subject {
        let obs_times = vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0];
        let occasions = vec![1u32, 1, 1, 2, 2, 2];
        let obs_wts = [70.0, 72.0, 78.0, 88.0, 90.0, 95.0];
        let n = obs_times.len();
        let (pk_only_times, pk_only_covariates) = if pk_only {
            (vec![18.0], vec![wt_map(85.0)])
        } else {
            (Vec::new(), Vec::new())
        };
        Subject {
            id: "1".to_string(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times,
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates: wt_map(70.0),
            dose_covariates: vec![wt_map(70.0), wt_map(85.0)],
            obs_covariates: obs_wts.iter().map(|&w| wt_map(w)).collect(),
            pk_only_times,
            pk_only_covariates,
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions,
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    /// 1-cpt oral IOV **+ WT-on-CL time-varying covariate**: the provider's
    /// value/grad/Hessian over `[η_bsv, κ_g0, κ_g1]` + θ (now including `THETA_WT`)
    /// must match FD of `predict_iov`, which seeds each event at its own covariate
    /// snapshot and occasion κ. Validates the per-event `sources` merge.
    #[test]
    fn iov_tvcov_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_TVCOV).expect("parse warfarin IOV+TVcov");
        assert_eq!(model.n_kappa, 1);
        assert!(iov_analytical_supported(&model));
        let subject = iov_tvcov_subject(false);
        assert!(subject.has_tv_covariates(), "fixture must carry TV cov");
        // θ = [TVCL, TVV, TVKA, THETA_WT]; stacked = [η_cl, η_v, η_ka, κ_g0, κ_g1].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 1.5, 0.75],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
    }

    /// 1-cpt oral IOV + TV-cov **with an EVID=2 covariate breakpoint**: a WT jump
    /// carried by a `pk_only` record (no occasion → κ fixed at 0) between the
    /// occasion-2 observations. Exercises the new `pk_only` source on the IOV path,
    /// which the previous code bailed out of.
    #[test]
    fn iov_tvcov_pkonly_breakpoint_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_TVCOV).expect("parse warfarin IOV+TVcov");
        let subject = iov_tvcov_subject(true);
        assert!(
            !subject.pk_only_times.is_empty(),
            "fixture must carry EVID=2"
        );
        assert!(subject.has_tv_covariates(), "fixture must carry TV cov");
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 1.5, 0.75],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
    }

    /// 2-cpt oral IOV + WT-on-CL TV covariate: same FD check through the generic
    /// 2-cpt event-driven sensitivity walk under per-event covariate seeding.
    #[test]
    fn iov_tvcov_2cpt_matches_fd_of_predict_iov() {
        let model =
            parse_model_string(WARFARIN_IOV_TVCOV_2CPT).expect("parse 2cpt warfarin IOV+TVcov");
        assert_eq!(model.n_kappa, 1);
        let subject = iov_tvcov_subject(false);
        // θ = [TVCL, TVV, TVQ, TVV2, TVKA, THETA_WT].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.5, 20.0, 1.5, 0.75],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
    }

    /// 3-cpt oral IOV + WT-on-CL TV covariate: same FD check through the generic
    /// 3-cpt eigenmode walk (widest dual on the IOV+TV-cov path).
    #[test]
    fn iov_tvcov_3cpt_matches_fd_of_predict_iov() {
        let model =
            parse_model_string(WARFARIN_IOV_TVCOV_3CPT).expect("parse 3cpt warfarin IOV+TVcov");
        assert_eq!(model.n_kappa, 1);
        let subject = iov_tvcov_subject(false);
        // θ = [TVCL, TVV, TVQ2, TVV2, TVQ3, TVV3, TVKA, THETA_WT].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.5, 20.0, 0.3, 50.0, 1.5, 0.75],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
    }
}
