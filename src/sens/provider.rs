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
    let prog_slots = prog.pk_slots();
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
        // Every individual-parameter slot must be one we differentiate. A
        // LAGTIME (slot 8) routes to fall back.
        && model.pk_indices.iter().all(|&s| slot_to_dim(s).is_some())
}

/// Maximum `(θ, η)` axis count for the differentiable `ExpressionScale` program
/// (the `Dual2<M>` dispatch table). Beyond this the scale falls back to FD.
const MAX_SCALE_AXES: usize = 16;

/// Maximum `(θ, η)` axis count (`n_theta + n_eta`) for the TV-cov event-driven dual
/// walk. The outer `run_obs_tvcov` (`m_dim`) and inner `run_obs_grad_tvcov` (`n_eta
/// ≤ m_dim`) dispatch tables both enumerate `1..=MAX_TVCOV_AXES`, and
/// `tvcov_analytical_supported` bounds the model here, so both resolve and the
/// inner/outer analytic scope stays matched (#449 re-review #2).
const MAX_TVCOV_AXES: usize = 24;

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
pub fn analytic_outer_gradient_available(model: &CompiledModel) -> bool {
    !matches!(model.gradient_method, GradientMethod::Fd)
        && (sens_supported(model) || iov_analytical_supported(model))
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
// with `K = split_obs_by_occasion(subject).len()`. The returned [`SubjectSens`]
// is over that stacked vector (plus the usual θ block), so the caller's block-Ω
// (BSV ⊕ K·IOV) assembly consumes it directly.

/// Per-occasion individual-parameter derivatives in the **combined** layout
/// `(θ, η_bsv, κ)` — the program's native axes for an IOV model
/// (`n_eff = n_eta_bsv + n_kappa`). One of these is built per occasion group from
/// that group's combined effect vector; the chain then scatters its η_bsv columns
/// to the shared BSV block and its κ columns to the group's own κ block.
struct CombinedDerivs {
    /// `∂p_i/∂(η_bsv, κ)`, row-major `n_rows × n_eff`.
    deta: Vec<Vec<f64>>,
    /// `∂p_i/∂θ_m`, `n_rows × n_theta`.
    dtheta: Vec<Vec<f64>>,
    /// `∂²p_i/∂(η_bsv,κ)²`, `n_rows × n_eff × n_eff`.
    d2eta: Vec<Vec<Vec<f64>>>,
    /// `∂²p_i/∂(η_bsv,κ)∂θ`, `n_rows × n_eff × n_theta`.
    d2eta_theta: Vec<Vec<Vec<f64>>>,
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
    if !iov_analytical_supported(model) {
        return None;
    }
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

    let occ_groups = crate::stats::likelihood::split_obs_by_occasion(subject);
    let k_groups = occ_groups.len();
    if k_groups == 0 {
        return None;
    }
    let n_stacked = n_eta + k_groups * n_kappa;
    if stacked_eta.len() != n_stacked {
        return None;
    }
    let mut occ_to_k: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::with_capacity(k_groups);
    for (k, (occ_id, _)) in occ_groups.iter().enumerate() {
        occ_to_k.insert(*occ_id, k);
    }
    // Combined effect vector for group `k`: `[η_bsv, κ_k]`.
    let eta_bsv = &stacked_eta[..n_eta];
    let combined_for = |k: usize| -> Vec<f64> {
        let mut c = Vec::with_capacity(n_eff);
        c.extend_from_slice(eta_bsv);
        let base = n_eta + k * n_kappa;
        c.extend_from_slice(&stacked_eta[base..base + n_kappa]);
        c
    };
    // EVID=2 (`pk_only`) rows carry no occasion label → BSV η with zero κ (matches
    // production `predict_iov`). Their κ derivatives are dropped from the stacked
    // axes (group `None` below), so the prediction holds κ fixed at 0.
    let combined_pk_only: Vec<f64> = {
        let mut c = Vec::with_capacity(n_eff);
        c.extend_from_slice(eta_bsv);
        c.extend(std::iter::repeat(0.0).take(n_kappa));
        c
    };

    let prog = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .expect("iov_analytical_supported guarantees the program");
    let slots = prog.pk_slots();
    let n_diff = slots.len();
    // PK slot → differentiated-row index (for seeding the dual axis).
    let mut slot_row: [Option<usize>; N_PK] = [None; N_PK];
    for (i, &s) in slots.iter().enumerate() {
        if s < N_PK {
            slot_row[s] = Some(i);
        }
    }

    let mp = n_theta + n_eff;
    // Combined derivatives at `(theta, combined)` evaluated at covariate map `cov`.
    macro_rules! cd_at {
        ($combined:expr, $cov:expr) => {{
            let combined = $combined;
            let cov = $cov;
            match mp {
                1 => Some(iov_combined_derivs::<1>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                2 => Some(iov_combined_derivs::<2>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                3 => Some(iov_combined_derivs::<3>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                4 => Some(iov_combined_derivs::<4>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                5 => Some(iov_combined_derivs::<5>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                6 => Some(iov_combined_derivs::<6>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                7 => Some(iov_combined_derivs::<7>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                8 => Some(iov_combined_derivs::<8>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                9 => Some(iov_combined_derivs::<9>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                10 => Some(iov_combined_derivs::<10>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                11 => Some(iov_combined_derivs::<11>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                12 => Some(iov_combined_derivs::<12>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                13 => Some(iov_combined_derivs::<13>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                14 => Some(iov_combined_derivs::<14>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                15 => Some(iov_combined_derivs::<15>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                16 => Some(iov_combined_derivs::<16>(
                    prog, n_theta, n_eff, n_diff, cov, theta, &combined,
                )),
                _ => None,
            }
        }};
    }

    // Per-event seed sources `(pk, cd, group)` — `group = Some(g)` maps the κ
    // columns to occasion group `g`'s stacked block; `None` (pk_only) drops them.
    // Each event's derivatives are evaluated at that event's covariate snapshot, so
    // a time-varying covariate is exact (no per-group caching across events). When
    // covariates are subject-static, one source per occasion group is built and
    // shared, preserving the non-TV cost.
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
            let cd = cd_at!(combined, cov)?;
            dose_src[d] = sources.len();
            sources.push((pk, cd, Some(g)));
        }
        for j in 0..subject.obs_times.len() {
            let occ = subject.occasions.get(j).copied()?;
            let g = *occ_to_k.get(&occ)?;
            let combined = combined_for(g);
            let cov = subject.obs_cov(j);
            let pk = (model.pk_param_fn)(theta, &combined, cov);
            let cd = cd_at!(combined, cov)?;
            obs_src[j] = sources.len();
            sources.push((pk, cd, Some(g)));
        }
        for m in 0..subject.pk_only_times.len() {
            let cov = subject.pk_only_cov(m);
            let pk = (model.pk_param_fn)(theta, &combined_pk_only, cov);
            let cd = cd_at!(combined_pk_only.clone(), cov)?;
            pkonly_src[m] = sources.len();
            sources.push((pk, cd, None));
        }
    } else {
        // One source per occasion group, at the subject-static covariates.
        let mut group_source = vec![usize::MAX; k_groups];
        for g in 0..k_groups {
            let combined = combined_for(g);
            let pk = (model.pk_param_fn)(theta, &combined, cov_static);
            let cd = cd_at!(combined, cov_static)?;
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
            let cd = cd_at!(combined_pk_only.clone(), cov_static)?;
            let idx = sources.len();
            sources.push((pk, cd, None));
            for m in 0..subject.pk_only_times.len() {
                pkonly_src[m] = idx;
            }
        }
    }

    // Run the walk over `Dual2<M>` (M = n_theta + n_stacked); the dual width tracks
    // the *unknowns* (n_eta + K·n_kappa + n_theta), not the PK axes, so it stays
    // narrow for many occasions whenever n_kappa < n_diff (the usual κ-on-CL case).
    let m_dim = n_theta + n_stacked;
    macro_rules! disp {
        ($($m:literal),+) => {
            match m_dim {
                $($m => run_obs_iov::<$m>(
                    model, subject, &sources, &dose_src, &obs_src, &pkonly_src, &slot_row,
                    n_eta, n_kappa, n_eff, n_stacked, n_theta,
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
    let slots = prog.pk_slots();
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
    let slots = prog.pk_slots();
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
    // The light (first-order) provider doesn't carry the `ExpressionScale`
    // quotient-rule, so the inner EBE loop reverts to FD there (the analytic
    // *outer* gradient still serves these models). Mirrors the LTBS inner choice.
    if matches!(model.scaling, ScalingSpec::ExpressionScale { .. }) {
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
    // Build the dual for each PK slot the scale references: value + ∂p/∂(θ,η),
    // with the unused θ-θ Hessian block left zero (the quotient rule only reads
    // η-η and η-θ). PK params not in `slots` (literal constants / undifferentiated)
    // enter as constants.
    let var_duals: Vec<Dual2<M>> = prog
        .var_to_pk_slot()
        .iter()
        .map(|&s| match slots.iter().position(|&x| x == s) {
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
                Dual2 {
                    value: pk.values.get(s).copied().unwrap_or(0.0),
                    grad,
                    hess,
                }
            }
            None => Dual2::constant(pk.values.get(s).copied().unwrap_or(0.0)),
        })
        .collect();

    let s = prog.eval_scale_dual::<M>(theta, eta, cov, &var_duals);
    let sv = s.value;
    let inv = 1.0 / sv;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;

    for o in sens.obs.iter_mut() {
        let f = o.f;
        let fk = o.df_deta.clone(); // original ∂f/∂η
        let fm = o.df_dtheta.clone(); // original ∂f/∂θ
                                      // η-η Hessian.
        for k in 0..n_eta {
            for l in 0..n_eta {
                let idx = k * n_eta + l;
                let s_k = s.grad[n_theta + k];
                let s_l = s.grad[n_theta + l];
                let s_kl = s.hess[n_theta + k][n_theta + l];
                o.d2f_deta2[idx] = o.d2f_deta2[idx] * inv
                    - fk[k] * s_l * inv2
                    - fk[l] * s_k * inv2
                    - f * s_kl * inv2
                    + 2.0 * f * s_k * s_l * inv3;
            }
        }
        // η-θ Hessian.
        for k in 0..n_eta {
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
        for k in 0..n_eta {
            o.df_deta[k] = fk[k] * inv - f * s.grad[n_theta + k] * inv2;
        }
        for m in 0..n_theta {
            o.df_dtheta[m] = fm[m] * inv - f * s.grad[m] * inv2;
        }
        o.f = f * inv;
    }
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
        macro_rules! disp_scale {
            ($($mm:literal),+) => {
                match prog.n_axes() {
                    $($mm => apply_expression_scale::<$mm>(
                        &mut sens, prog, &pk, &pd, &slots, theta, eta,
                        &subject.covariates, n_theta, n_eta,
                    ),)+
                    _ => {}
                }
            };
        }
        disp_scale!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
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
        let n_theta = theta.len();
        let sens = subject_sensitivities_iov(model, subject, theta, stacked).expect("supported");

        // Map a stacked-η vector to predict_iov's (η_bsv, kappas-per-group) form.
        let pred = |st: &[f64], th: &[f64], j: usize| -> f64 {
            let eta_bsv = st[..n_eta].to_vec();
            let kappas: Vec<Vec<f64>> = vec![vec![st[n_eta]], vec![st[n_eta + 1]]];
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
