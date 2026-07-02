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
use super::one_cpt::{one_cpt_conc_g, one_cpt_transit_conc_g};
use super::three_cpt::three_cpt_conc_g;
use super::two_cpt::two_cpt_conc_g;
use crate::types::{
    CompiledModel, DoseEvent, GradientMethod, PkModel, ScalingSpec, Subject, PK_IDX_CL, PK_IDX_F,
    PK_IDX_KA, PK_IDX_LAGTIME, PK_IDX_MTT, PK_IDX_N, PK_IDX_Q, PK_IDX_Q3, PK_IDX_V, PK_IDX_V2,
    PK_IDX_V3,
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
        // Transit `n`/`mtt` (#386); the analytic `one_cpt_transit` closed form reads
        // them, so they are differentiated like the other structural slots.
        PK_IDX_N => Some(9),
        PK_IDX_MTT => Some(10),
        _ => None,
    }
}

/// Width of the per-model PK-slot lookup tables — spans the full PK slot space
/// (`CL, V1, Q2, V2, KA, F, Q3, V3, LAGTIME`, the transit `N`/`MTT` slots #386, and
/// the modeled `D{cmt}`/`R{cmt}` slots the parser assigns from `PK_IDX_LAGTIME+1`
/// upward, #486). Sized to [`crate::types::MAX_PK_PARAMS`] so a subject with three or
/// more modeled infusions (slots ≥ 11) is indexed in-bounds rather than panicking
/// (#486 review #1). This is the slot-space size, not the per-model dual width `M`
/// (the compact count of *seeded* params — e.g. `Dual2<4>` for a 2-cpt IV).
const N_PK: usize = crate::types::MAX_PK_PARAMS;

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
///
/// A model that reads the event-time built-in `TIME` in a structural parameter
/// (`compiled_model_uses_time_builtin`) IS admitted here: the parameter is then
/// piecewise/time-varying, so — like a time-varying covariate — the subject is
/// routed through the per-event event-driven `Dual2` walk
/// ([`subject_sensitivities_tvcov`]) rather than dose superposition, which can't
/// express a mid-decay parameter switch. The routing (`uses_time_builtin`) is in
/// [`subject_sensitivities_impl`] / [`subject_eta_grad_impl`] (#486 / #610).
pub fn analytical_supported(model: &CompiledModel) -> bool {
    // A `TIME` model does NOT use the static dose-superposition path — its subjects
    // route through the per-event event-driven walk. So being model-level "analytic"
    // additionally requires that walk to serve it (`tvcov_analytical_supported`);
    // otherwise the direct `pk(...=TIME)` mapping (whose mapped slot is not desugared
    // into the program's `pk_slots`, so `tvcov_analytical_supported` declines it) would
    // report "analytic" through `sens_supported` / `analytic_outer_gradient_available` /
    // `analytic_inner_grad_supported_model` while every subject actually falls back to FD
    // — a route/report drift (#637 review #1). `tvcov_analytical_supported` calls the
    // non-recursive [`analytical_supported_core`], so there is no cycle.
    analytical_supported_core(model)
        && (!crate::parser::model_parser::compiled_model_uses_time_builtin(model)
            || tvcov_analytical_supported(model))
}

/// The model-level closed-form scope check shared by [`analytical_supported`] and
/// [`tvcov_analytical_supported`] — everything except the `TIME`-routing clause, so the
/// two can compose without recursion (#637 review #1).
fn analytical_supported_core(model: &CompiledModel) -> bool {
    matches!(
        model.pk_model,
        PkModel::OneCptIv
            | PkModel::OneCptOral
            | PkModel::OneCptTransit
            | PkModel::TwoCptIv
            | PkModel::TwoCptOral
            | PkModel::ThreeCptIv
            | PkModel::ThreeCptOral
    ) && model.ode_spec.is_none()
        && model.tv_fn.is_some()
        && model.n_kappa == 0
        && scaling_supported(model)
        && init_supported(model)
        && analytic_readout_dual_supported(model)
        // Every individual-parameter slot must be one we differentiate. A
        // LAGTIME (slot 8) routes to fall back.
        && model.pk_indices.iter().all(|&s| slot_to_dim(s).is_some())
}

/// Whether an analytic Form C readout (`[scaling] y = <expr>`, #650), if present,
/// is one the Dual2 provider differentiates exactly. `true` when there is no
/// readout (the built-in concentration output), or when the readout is:
/// - a **uniform** `Single` readout (per-CMT `y[CMT=N]` routes to FD for now),
/// - carrying a **dual-evaluable** program (no bare θ/η / NN output — those are
///   left un-desugared on the analytic path, so they fall back to FD), and
/// - not referencing the oral **depot** amount (the static superposition jet
///   reconstructs the central amount as `conc × V` but not the depot amount),
///   and the model has no `[initial_conditions]` baseline (the init impulse is
///   layered onto the concentration, not the post-readout output).
///
/// When `false`, the model's Form C readout routes to the finite-difference
/// gradient — which differentiates the readout-aware f64 predictor, so it is
/// correct, just slower (a documented fallback, not silent: the parser attaches
/// a warning). Keeping the check here means `analytical_supported` reports FD
/// honestly rather than claiming analytic and then FD-ing every subject (#637).
fn analytic_readout_dual_supported(model: &CompiledModel) -> bool {
    let Some(ar) = &model.analytic_readout else {
        return true;
    };
    if !model.analytical_init.is_empty() {
        return false;
    }
    match (&ar.readout, &ar.program) {
        (crate::ode::OdeReadout::Single(_), Some(prog)) => {
            // Depot is state slot 0 only in the first-order-oral
            // `["depot", "central"]` layout. IV models (and the transit model,
            // whose analytic layout is central-only) have no depot slot, so their
            // slot 0 is `central` — guard the depot check on the layout, not on
            // `is_oral()` (which also matches transit).
            let has_depot_slot = matches!(
                model.pk_model,
                PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral
            );
            prog.is_dual_evaluable() && !(has_depot_slot && prog.references_state(0))
        }
        _ => false,
    }
}

/// Whether an analytic Form C readout can be served analytically on the
/// **tv-covariate / oral-infusion / TIME** event-walk path (#650). The walk seeds
/// only the eight structural `PkDual` slots (`CL..V3`), so in addition to the
/// static-path scope ([`analytic_readout_dual_supported`]) every parameter the
/// readout references must map to a slot `<= PK_IDX_V3`. A readout referencing a
/// higher slot (only reachable when many non-structural readout params are
/// allocated) falls back to FD on this path — the static path still serves it.
fn readout_tvcov_supported(model: &CompiledModel) -> bool {
    let Some(ar) = &model.analytic_readout else {
        return true;
    };
    match &ar.program {
        Some(prog) => {
            analytic_readout_dual_supported(model)
                && prog.max_indiv_pk_slot().is_none_or(|s| s <= PK_IDX_V3)
        }
        None => false,
    }
}

/// Evaluate an analytic Form C readout jet at one observation (#650), generic over
/// the [`PkNum`](crate::sens::num::PkNum) dual type.
///
/// Reconstructs the compartment-amount state the readout expects — the central
/// amount is `conc × V` (`params[PK_IDX_V]`), any depot slot is left at zero
/// (depot-referencing readouts route to FD, so the walks never reach here with
/// one) — then runs the readout program. Shared by the four dual paths
/// (`run_obs` / `run_obs_grad` static, `run_obs_tvcov` / `run_obs_grad_tvcov`
/// event-walk) so the value and gradient jets cannot drift. `params` is the flat
/// PK-slot dual vector; `ro_state` / `ro_vars` / `ro_stack` are caller-owned
/// scratch, cleared here.
#[inline]
fn eval_readout_jet<T: crate::sens::num::PkNum>(
    prog: &crate::parser::model_parser::OdeOutputProgram,
    conc: T,
    params: &[T],
    cov: &std::collections::HashMap<String, f64>,
    ro_state: &mut Vec<T>,
    ro_vars: &mut Vec<T>,
    ro_stack: &mut Vec<T>,
) -> T {
    let n_states = prog.n_states();
    let central_slot = n_states.saturating_sub(1);
    ro_state.clear();
    ro_state.resize(n_states, T::from_f64(0.0));
    ro_state[central_slot] = conc * params[PK_IDX_V];
    prog.eval_output_g::<T>(ro_state, params, cov, ro_vars, ro_stack)
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
        // Correlated residual (`block_sigma`, #627) now has an analytic outer gradient
        // for the analytical (diagonal-R) scope: the dense Almquist assembly reduces to
        // the scalar path fed correlation-aware `(r,d,d2)` (`corr_residual_diag`). A rare
        // off-diagonal-R subject bails per-subject to FD via `corr_residual_diag` → `None`.
        && !model.has_tte()
        // Custom / time-varying residual-error magnitude (#484/#576/#486): `mult(θ)`
        // makes `R` depend on θ directly, which `sens_outer_gradient::theta_block`
        // now carries via a `Dual1`-differentiated direct-θ channel — bounded by the
        // same `Dual1<M>` dispatch table `ScaleDerivProgram`/`ExpressionScale` uses
        // (`MAX_RUV_MAG_AXES`). Beyond that axis count, or combined with `iiv_on_ruv`
        // (whose residual-eta `c̃`-column coupling `d/R` would need its own direct-θ
        // chain, not yet assembled — see `prepare_stacked`), still routes to FD.
        // Also combined with a **correlated** residual (`block_sigma`, #627): the dense
        // assembly does not thread the magnitude's direct-θ channel, so the orthogonal
        // combination stays on FD even though each is analytic on its own.
        // (An M3-censored row's `−logΦ(z)` direct-θ chain is a *per-subject* gap
        // `prepare_stacked` bails at runtime — see its own doc — not a model-level
        // one, so it is not gated here.)
        && (!model.has_custom_ruv_magnitude()
            || (model.n_theta <= crate::parser::model_parser::MAX_RUV_MAG_AXES
                && model.residual_error_eta.is_none()
                && model.residual_correlations.is_empty()))
        // `iov_sens_supported` (not just the closed-form `iov_analytical_supported`) so
        // the predicate also recognizes the ODE IOV outer gradient (#439 ODE IOV / #466).
        && (sens_supported(model) || iov_sens_supported(model))
        // IIV on residual error (#474): the analytic gradient (inner η-column +
        // outer θ/Ω/σ variance terms) is provider-agnostic, so it serves the
        // closed-form AND ODE paths — including IOV and the M3-BLOQ triple
        // (closed-form #4b/#591, ODE IOV #486). Only **non-IOV** ODE M3 + `iiv_on_ruv`
        // still routes to FD, which is all `iiv_on_ruv_forces_fd` gates now.
        && !model.iiv_on_ruv_forces_fd()
}

/// [`analytic_outer_gradient_available`], kept as the shared entry point that
/// [`crate::types::Optimizer::resolve_auto`] and `build_info::gradient_method_outer`
/// consult so `auto`'s pick and the reported gradient method can never disagree
/// with the outer loop's actual dispatch.
///
/// As of the FOCE σ-magnitude work (#486), a custom / time-varying residual-error
/// magnitude (`σ = expr(TIME, cov, θ)`) is analytic on **both** loops — the
/// Sheiner–Beal FOCE (non-interaction) assembly (`subject_packed_gradient_foce` /
/// `subject_packed_gradient_foce_iov`) now threads `mult(θ)` through its marginal
/// `R⁰` (value + direct-θ derivative), matching the FOCEI direct-θ channel. So the
/// predicate no longer narrows by `interaction`: every exclusion in
/// `analytic_outer_gradient_available` is interaction-independent (a magnitude
/// combined with an M3-censored row is a *per-subject* FD fallback, as under
/// FOCEI, not a model-level decline). The `interaction` parameter is retained for
/// call-site stability and in case a future interaction-only channel must
/// re-narrow here.
pub fn analytic_outer_gradient_for_interaction(model: &CompiledModel, interaction: bool) -> bool {
    let _ = interaction;
    analytic_outer_gradient_available(model)
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
#[derive(Clone)]
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
    // A `TIME`-built-in structural parameter is served analytically: `build_iov_sources`
    // seeds each occasion's PK-param value AND its `CombinedDerivs` at that event's time
    // (per-event branch, gated on `uses_time_builtin`), so both IOV loops evaluate the
    // switch at the right time (#486). No early-return needed.
    // M3 BLOQ + IOV is analytic as of #580. The IOV objective promotes M3 to the
    // censored marginal (`foce_subject_nll_iov` / `individual_nll_iov`, data term
    // `−logΦ(z)`), and the analytic gradient now differentiates that same function:
    // the censored-row coefficients ride the stacked `[η_bsv, κ]` layout via the
    // shared `prepare_stacked` (true inner Hessian + `mixed_eta_theta` + `sigma_block`;
    // censored rows carry `p = β = 0`, so they leave `H̃`/`log|H̃|` exactly as in the
    // non-IOV assembly and as `foce_subject_nll_iov` builds it) and the IOV inner
    // gradient `analytic_eta_nll_gradient_iov` (censored `h·m` coefficient over the
    // stacked Jacobian).
    //
    // The triple **M3 + IOV + `iiv_on_ruv`** is analytic too as of #591. The closed-form
    // assembly already carried the censored residual-eta cross coefficients `(C·z, C·m)`
    // generically (#4c added them to `prepare_stacked` over any random-effect dimension):
    // the censored row's `H[η_ruv,η_ruv] += C·z`, `H[η_ruv,l] += C·m·a_l` enter the true
    // inner Hessian, `mixed_eta_theta`/`sigma_block` read the `C·m`/`C·z` cross-terms, and
    // the IOV inner gradient's `residual_inner_obs` emits the `h·z` residual-eta column —
    // all keyed on the residual-eta index, which lives in the BSV block of the stacked
    // layout. So opening the gate lights up both loops; the censored rows still leave
    // `H̃`/`log|H̃|` (no `c̃` residual-eta column), matching the objective. The ODE *IOV*
    // triple is analytic too (#486); only the **non-IOV ODE** M3 + `iiv_on_ruv` combo stays
    // FD (via `iiv_on_ruv_forces_fd`, `ode_spec`-AND-`n_kappa == 0`-gated).
    //
    // IIV on residual error (`iiv_on_ruv`) IS analytic for closed-form IOV models:
    // `η_ruv` enters only through the variance (`v = R(f)·exp(2·η_ruv)`, `∂f/∂η_ruv = 0`),
    // and both halves now carry that scaling — the IOV inner gradient
    // (`analytic_eta_nll_gradient_iov`) scales `v`/`dv_df` and adds the `η_ruv` variance
    // column, and the IOV outer assembly threads `ruv = residual_error_eta` into
    // `prepare_stacked` (the residual-eta `c̃` column rides the stacked `[η_bsv, κ]`
    // layout). Mirrors the non-IOV `iiv_on_ruv` path (#474). (ODE IOV + `iiv_on_ruv`
    // stays FD via `ode_iov_supported`, which is unchanged.)
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
    // Output scaling: `None` (raw jet) or a differentiable `ExpressionScale` `obs_scale`
    // divisor whose (θ, η) axis counts match the model and fit the scale dispatch table.
    // The `ExpressionScale` quotient is a per-occasion-group post-walk step in
    // `run_obs_iov` / `run_obs_iov_eta` (the closed-form twin of the ODE-IOV path,
    // #575/#590). `ScalarScale` still declines (the IOV walk has no constant-divisor step)
    // and LTBS declines below (the in-walk log can't compose with the post-walk quotient),
    // matching the ODE-IOV gate `expression_scale_axes_admissible` (#486).
    match &model.scaling {
        ScalingSpec::None => {}
        ScalingSpec::ExpressionScale { deriv: Some(p), .. }
            if p.n_theta_axis() == model.n_theta
                && p.n_eta_axis() == model.n_eta
                && (1..=MAX_SCALE_AXES).contains(&p.n_axes()) => {}
        _ => return false,
    }
    if model.log_transform || model.has_lagtime() {
        return false;
    }
    // Initial-compartment amounts (#521) are now differentiable on the IOV event-driven walk
    // too (#486): `run_obs_iov` / `run_obs_iov_eta` layer the `A₀·kernel(t, pk)` impulse per
    // occasion — the amount `A₀` from the BSV-only snapshot (`bsv_amount`, κ = 0) and the decay
    // kernel from each observation's occasion source — mirroring production's
    // `add_analytical_init_with(..., amount_pk = BSV, Some(obs_params))` in `predict_iov`.
    // A hand-built init with no `amount_deriv` program still routes to FD (the impulse loop
    // would otherwise silently drop it), so require every init to carry its amount program; the
    // stacked axis count `n_theta + n_stacked` is bounded by the walk's `1..=24` dispatch (out
    // of range → the walk returns `None` → FD).
    if model
        .analytical_init
        .iter()
        .any(|i| i.amount_deriv.is_none())
    {
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
    // Analytic Form C readout (#650) under IOV: the IOV dual walk does not yet
    // route the amount state through the readout, so fall back to FD (correct via
    // the readout-aware f64 predictor). See `subject_sensitivities_tvcov`.
    if model.analytic_readout.is_some() {
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
                    model, subject, theta, stacked_eta, &s.sources, &s.dose_src, &s.obs_src,
                    &s.pkonly_src, &s.slot_row, s.n_eta, s.n_kappa, s.n_eff, s.n_stacked,
                    s.n_theta, s.scale_groups.as_deref(), s.bsv_amount.as_ref(),
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
    /// For an `ExpressionScale` `obs_scale` divisor: one `(pk, combined-derivs)` per
    /// occasion group, evaluated at the **subject-static covariate snapshot** (`t = 0`),
    /// so the walk can build a per-group scale jet. `None` when the model carries no
    /// differentiable `ExpressionScale`. The scale divisor is subject-static under IOV
    /// (production `predict_iov` calls `apply_scaling` at `t = 0` inside each occasion),
    /// so it is built at static cov even for a TV-covariate subject whose event sources
    /// are per-event — the closed-form twin of the ODE-IOV `build_all_groups` scale seed
    /// (#575/#590, ported in #486).
    scale_groups: Option<Vec<(crate::types::PkParams, CombinedDerivs)>>,
    /// For an `[initial_conditions]` baseline under IOV (#486): the `(pk, combined-derivs)`
    /// at the **BSV-only** snapshot (`κ = 0`, subject-static covariates, `t = 0`), driving the
    /// baseline *amount* `A₀`. Production's `predict_iov` evaluates the amount with `eta_bsv`
    /// (no κ) while the *decay* kernel uses each observation's occasion PK params, so the amount
    /// carries `∂A₀/∂(θ, η_bsv)` but zero `∂A₀/∂κ`. `None` when the model has no
    /// `[initial_conditions]`.
    bsv_amount: Option<(crate::types::PkParams, CombinedDerivs)>,
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
    // Modeled-`RATE=-1/-2` doses are served analytically now (#486): the walk carries
    // the dual rate + moving infusion-end boundary (`run_obs_iov` resolves each dose at
    // its occasion's PK jet). The steady-state combination is analytic too (#486:
    // `equilibrate_ss_g` threads the per-occasion modeled-window jet into its per-cycle
    // active/quiet split); only the missing-slot case declines to FD via the shared gate.
    if !modeled_dose_analytic_gate(model, subject) {
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

    // Combined derivatives at `(theta, combined)` evaluated at covariate map `cov`
    // and event time `time`, dispatching the program-eval width `MP = n_theta +
    // n_eff`. The `TIME` built-in resolves `Op::PushTime` from the model-time
    // thread-local, which `iov_combined_derivs`' `Dual2` walk reads; seed it with the
    // per-event time (gated on `uses_time`, like the f64 `pk_param_fn` closure) so
    // each occasion's PK-param derivatives are evaluated at that event's TIME (#486).
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let cd_at = |time: f64, combined: &[f64], cov: &std::collections::HashMap<String, f64>| {
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
        crate::sens::provider::iov_combined_derivs_dyn(
            prog, n_theta, n_eff, n_diff, cov, theta, combined,
        )
    };

    // Per-event seed sources `(pk, cd, group)`. Each event's derivatives are evaluated
    // at that event's covariate snapshot, so a time-varying covariate is exact (no
    // per-group caching across events). When covariates are subject-static, one source
    // per occasion group is built and shared, preserving the non-TV cost.
    // A `TIME`-built-in structural parameter is per-event dynamic even with no TV
    // covariates, so it must take the per-event branch (the one-source-per-group
    // fast path below would freeze every occasion at `t = 0`) — mirroring the
    // non-IOV `run_obs_tvcov` and the f64 `compute_event_pk_params_into` (#486).
    // `has_tv_covariates()` covers dose/obs snapshots but NOT EVID=2 pk-only snapshots,
    // so include them explicitly — otherwise a subject whose only per-event covariates
    // are on pk-only records would take the shared static source and evaluate those
    // events at the wrong covariates (matches the ODE `seed_iov_events` per-event gate;
    // #637 Copilot #1).
    let has_tv = subject.has_tv_covariates() || !subject.pk_only_covariates.is_empty();
    let per_event = has_tv || uses_time;
    let cov_static = &subject.covariates;
    let mut sources: Vec<(crate::types::PkParams, CombinedDerivs, Option<usize>)> = Vec::new();
    let mut dose_src = vec![0usize; subject.doses.len()];
    let mut obs_src = vec![0usize; subject.obs_times.len()];
    let mut pkonly_src = vec![0usize; subject.pk_only_times.len()];

    // `ExpressionScale` `obs_scale`: one `(pk, cd)` reference per occasion group at the
    // subject-static covariate snapshot (`t = 0`) — the divisor depends on the group's κ
    // through the PK params, but not on the per-event covariate/time, matching production
    // `predict_iov`'s subject-static `apply_scaling` inside each occasion. Built (with
    // `theta`/`combined_for`/`cd_at` in scope) branch-side and consumed dual-typed in the
    // walk fns. In the static-covariate branch the per-group sources ARE that `t = 0`
    // snapshot, so reuse them rather than recomputing `k_groups` full-order `cd_at`
    // on every gradient call (#651 review #3); the per-event branch has no such snapshot
    // in `sources` (those sit at each event's own covariate/time) and builds it fresh.
    let want_scale = matches!(
        &model.scaling,
        ScalingSpec::ExpressionScale { deriv: Some(_), .. }
    );
    let mut scale_groups: Option<Vec<(crate::types::PkParams, CombinedDerivs)>> = None;

    if per_event {
        for d in 0..subject.doses.len() {
            let occ = subject.dose_occasions.get(d).copied()?;
            let g = *occ_to_k.get(&occ)?;
            let combined = combined_for(g);
            let cov = subject.dose_cov(d);
            let t = subject.doses[d].time;
            let pk = (model.pk_param_fn)(theta, &combined, cov, t);
            let cd = cd_at(t, &combined, cov)?;
            dose_src[d] = sources.len();
            sources.push((pk, cd, Some(g)));
        }
        for j in 0..subject.obs_times.len() {
            let occ = subject.occasions.get(j).copied()?;
            let g = *occ_to_k.get(&occ)?;
            let combined = combined_for(g);
            let cov = subject.obs_cov(j);
            let t = subject.obs_times[j];
            let pk = (model.pk_param_fn)(theta, &combined, cov, t);
            let cd = cd_at(t, &combined, cov)?;
            obs_src[j] = sources.len();
            sources.push((pk, cd, Some(g)));
        }
        for m in 0..subject.pk_only_times.len() {
            let cov = subject.pk_only_cov(m);
            let t = subject.pk_only_times[m];
            let pk = (model.pk_param_fn)(theta, &combined_pk_only, cov, t);
            let cd = cd_at(t, &combined_pk_only, cov)?;
            pkonly_src[m] = sources.len();
            sources.push((pk, cd, None));
        }
        if want_scale {
            let mut groups = Vec::with_capacity(k_groups);
            for g in 0..k_groups {
                let combined = combined_for(g);
                let pk = (model.pk_param_fn)(theta, &combined, cov_static, 0.0);
                let cd = cd_at(0.0, &combined, cov_static)?;
                groups.push((pk, cd));
            }
            scale_groups = Some(groups);
        }
    } else {
        // One source per occasion group, at the subject-static covariates.
        let mut group_source = vec![usize::MAX; k_groups];
        for g in 0..k_groups {
            let combined = combined_for(g);
            let pk = (model.pk_param_fn)(theta, &combined, cov_static, 0.0);
            let cd = cd_at(0.0, &combined, cov_static)?;
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
            let pk = (model.pk_param_fn)(theta, &combined_pk_only, cov_static, 0.0);
            let cd = cd_at(0.0, &combined_pk_only, cov_static)?;
            let idx = sources.len();
            sources.push((pk, cd, None));
            for m in 0..subject.pk_only_times.len() {
                pkonly_src[m] = idx;
            }
        }
        if want_scale {
            // Reuse the per-group static sources (identical `(pk, cd)` at `t = 0`) —
            // clone is a Vec copy, far cheaper than re-evaluating the derivative program.
            scale_groups = Some(
                (0..k_groups)
                    .map(|g| {
                        let (pk, cd, _) = &sources[group_source[g]];
                        (*pk, cd.clone())
                    })
                    .collect(),
            );
        }
    }

    // `[initial_conditions]` baseline amount snapshot (#486): the BSV-only effect
    // (`combined_pk_only` = `[η_bsv, κ = 0]`) at the subject-static covariates and `t = 0`,
    // matching production's `add_analytical_init_with(..., amount_pk = pk_param_fn(θ, eta_bsv,
    // subject.covariates, 0), ...)`. The decay kernel uses each observation's occasion source
    // (built above); only the amount uses this κ-less snapshot.
    let bsv_amount = if model.analytical_init.is_empty() {
        None
    } else {
        let pk = (model.pk_param_fn)(theta, &combined_pk_only, cov_static, 0.0);
        let cd = cd_at(0.0, &combined_pk_only, cov_static)?;
        Some((pk, cd))
    };

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
        scale_groups,
        bsv_amount,
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
    // Analytic Form C readout (#650) under IOV: the IOV dual walk does not yet
    // route the amount state through the readout, so fall back to FD (correct via
    // the readout-aware f64 predictor). See `subject_sensitivities_tvcov`.
    if model.analytic_readout.is_some() {
        return None;
    }
    let s = build_iov_sources(model, subject, theta, stacked_eta)?;
    let n_dim = s.n_stacked;
    macro_rules! disp {
        ($($n:literal),+) => {
            match n_dim {
                $($n => run_obs_iov_eta::<$n>(
                    model, subject, theta, stacked_eta, &s.sources, &s.dose_src, &s.obs_src,
                    &s.pkonly_src, &s.slot_row, s.n_eta, s.n_kappa, s.n_stacked,
                    s.scale_groups.as_deref(), s.bsv_amount.as_ref(),
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
    theta: &[f64],
    stacked_eta: &[f64],
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
    scale_groups: Option<&[(crate::types::PkParams, CombinedDerivs)]>,
    bsv_amount: Option<&(crate::types::PkParams, CombinedDerivs)>,
) -> Option<SubjectSens> {
    use crate::pk::event_driven::EventSchedule;
    use crate::sens::propagate::{event_driven_sens_with_doses_g, PkDual};

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
    // Modeled-`RATE=-1/-2` doses (#486): resolve each dose at its occasion's PK jet
    // (`sources[dose_src[k]].0` is already `pk_param_fn` at that occasion's combined
    // effect) for the schedule's break times — shared with the non-IOV walk via
    // `resolve_eff_doses` — and build the per-dose duals seeded on the stacked
    // `(η_bsv, κ)` axes (via the same `seed` used for the PkDuals). Fixed subjects
    // borrow `subject.doses` and pass no duals.
    let eff_doses = resolve_eff_doses(model, subject, |k| sources[dose_src[k]].0);
    // Distinct-slot coincident infusion ends aren't representable by the single-`dt`
    // walk — decline to FD (#486 review #2).
    if !modeled_ends_separable(model, subject, &eff_doses) {
        return None;
    }
    let dose_inf_dual = modeled_dose_inf_duals::<Dual2<M>>(model, subject, |k, dr_slot| {
        let (pk, cd, group) = &sources[dose_src[k]];
        let val = pk.values.get(dr_slot).copied().unwrap_or(0.0);
        match slot_row.get(dr_slot).copied().flatten() {
            Some(i) => seed(cd, *group, i, val),
            None => Dual2::<M>::constant(val),
        }
    });
    let schedule = EventSchedule::for_subject(subject, model.pk_model, &eff_doses, &dose_lagtimes);
    let conc = event_driven_sens_with_doses_g::<Dual2<M>>(
        model.pk_model,
        subject,
        &schedule,
        &eff_doses,
        &dose_inf_dual,
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

    // Analytic `[initial_conditions]` impulse under IOV (#486): layer `A₀·kernel(t, pk)` onto
    // every pre-reset observation BEFORE scaling — the closed-form IOV twin of the non-IOV
    // `apply_analytical_init_outer` and of production's `add_analytical_init_with(..., amount_pk
    // = BSV snapshot, Some(obs_params))`. The amount `A₀` is built from the BSV-only snapshot
    // (`bsv_amount`, κ = 0), so it carries `∂A₀/∂(θ, η_bsv)` but zero `∂A₀/∂κ`; the decay kernel
    // uses each observation's occasion source (`sources[obs_src[j]]`), so it carries the
    // occasion κ. Their `Dual2<M>` product on the stacked axes is exact.
    if let Some((pk_bsv, cd_bsv)) = bsv_amount {
        let first_reset = subject
            .reset_times
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let eta_bsv = &stacked_eta[..n_eta];
        let cov = &subject.covariates;
        for init in &model.analytical_init {
            let Some(prog) = &init.amount_deriv else {
                continue;
            };
            // Baseline amount `A₀` seeded on the stacked axes from the BSV snapshot (group
            // `None` → κ axes dropped, matching production's κ-less amount). `eval_scale_dual`
            // seeds direct θ/η references on axes `0..n_theta` / `n_theta..n_theta+n_eta` — the
            // stacked η_bsv block — so the amount's jet lands on the correct axes.
            let a0_var_duals: Vec<Dual2<M>> = prog
                .var_to_pk_slot()
                .iter()
                .map(|&s| match slot_row[s] {
                    Some(i) => seed(
                        cd_bsv,
                        None,
                        i,
                        pk_bsv.values.get(s).copied().unwrap_or(0.0),
                    ),
                    None => Dual2::<M>::constant(pk_bsv.values.get(s).copied().unwrap_or(0.0)),
                })
                .collect();
            let a0 = prog.eval_scale_dual::<M>(theta, eta_bsv, cov, &a0_var_duals);
            if !a0.value.is_finite() {
                continue;
            }
            for (j, o) in obs_out.iter_mut().enumerate() {
                let t = subject.obs_times[j];
                if t < 0.0 || t >= first_reset {
                    continue;
                }
                // Decay kernel PK duals at observation `j`'s own occasion source (κ carried).
                let (pk_d, cd_d, group_d) = &sources[obs_src[j]];
                let dv = |slot: usize, val: f64| -> Dual2<M> {
                    match slot_row[slot] {
                        Some(i) => seed(cd_d, *group_d, i, val),
                        None => Dual2::<M>::constant(val),
                    }
                };
                let c = crate::pk::analytical_init_concentration_g::<Dual2<M>>(
                    model.pk_model,
                    init.cmt,
                    a0,
                    Dual2::<M>::constant(t),
                    dv(PK_IDX_CL, pk_d.cl()),
                    dv(PK_IDX_V, pk_d.v()),
                    dv(PK_IDX_Q, pk_d.q()),
                    dv(PK_IDX_V2, pk_d.v2()),
                    dv(PK_IDX_KA, pk_d.ka()),
                    dv(PK_IDX_Q3, pk_d.q3()),
                    dv(PK_IDX_V3, pk_d.v3()),
                );
                o.f += c.value;
                for p in 0..n_stacked {
                    o.df_deta[p] += c.grad[n_theta + p];
                    for q in 0..n_stacked {
                        o.d2f_deta2[p * n_stacked + q] += c.hess[n_theta + p][n_theta + q];
                    }
                    for m in 0..n_theta {
                        o.d2f_deta_dtheta[p * n_theta + m] += c.hess[n_theta + p][m];
                    }
                }
                for m in 0..n_theta {
                    o.df_dtheta[m] += c.grad[m];
                }
            }
        }
    }

    // η-dependent `ExpressionScale` `obs_scale` divisor: one scale jet per occasion group
    // (the divisor depends on the group's κ through the PK params), applied as a post-walk
    // quotient over the stacked `(θ, η_bsv, κ)` axes — the closed-form twin of the ODE-IOV
    // per-group quotient (#575/#590). The scale reference `(pk, cd)` is at the subject-static
    // covariate snapshot (`t = 0`), matching production `predict_iov`'s `apply_scaling`.
    if let Some(scale_groups) = scale_groups {
        let ScalingSpec::ExpressionScale {
            deriv: Some(sprog), ..
        } = &model.scaling
        else {
            return None; // gate guarantees this arm; defensive
        };
        let eta_bsv = &stacked_eta[..n_eta];
        let cov = &subject.covariates;
        // Per-group scale jets, seeding the `(θ, η_bsv, κ_g)` axes (mirroring
        // `seed_pk_dual2_iov`) — shared with the inner walk via `build_iov_group_scale_jets`.
        let group_scale = build_iov_group_scale_jets::<Dual2<M>>(
            scale_groups,
            slot_row,
            sprog.var_to_pk_slot(),
            |cd, g, i, v| seed(cd, Some(g), i, v),
            |var_duals| sprog.eval_scale_dual::<M>(theta, eta_bsv, cov, var_duals),
        );
        let mut fk: Vec<f64> = Vec::with_capacity(n_stacked);
        let mut fm: Vec<f64> = Vec::with_capacity(n_theta);
        for (j, o) in obs_out.iter_mut().enumerate() {
            let g = sources[obs_src[j]].2?;
            apply_scale_quotient_row::<M>(o, &group_scale[g], n_theta, n_stacked, &mut fk, &mut fm);
        }
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
    theta: &[f64],
    stacked_eta: &[f64],
    sources: &[(crate::types::PkParams, CombinedDerivs, Option<usize>)],
    dose_src: &[usize],
    obs_src: &[usize],
    pkonly_src: &[usize],
    slot_row: &[Option<usize>; N_PK],
    n_eta: usize,
    n_kappa: usize,
    n_stacked: usize,
    scale_groups: Option<&[(crate::types::PkParams, CombinedDerivs)]>,
    bsv_amount: Option<&(crate::types::PkParams, CombinedDerivs)>,
) -> Option<Vec<ObsGrad>> {
    use crate::pk::event_driven::EventSchedule;
    use crate::sens::propagate::{event_driven_sens_with_doses_g, PkDual};

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
    // Modeled-`RATE=-1/-2` doses (#486): resolve each dose at its occasion's PK jet
    // (shared with the non-IOV walk via `resolve_eff_doses`) and build the per-dose
    // duals seeded on the stacked-η axes (η-only, the inner mirror of `run_obs_iov`).
    // Fixed subjects borrow `subject.doses` and pass no duals.
    let eff_doses = resolve_eff_doses(model, subject, |k| sources[dose_src[k]].0);
    // Distinct-slot coincident infusion ends aren't representable by the single-`dt`
    // walk — decline to FD (#486 review #2).
    if !modeled_ends_separable(model, subject, &eff_doses) {
        return None;
    }
    let dose_inf_dual = modeled_dose_inf_duals::<Dual1<N>>(model, subject, |k, dr_slot| {
        let (pk, cd, group) = &sources[dose_src[k]];
        let val = pk.values.get(dr_slot).copied().unwrap_or(0.0);
        match slot_row.get(dr_slot).copied().flatten() {
            Some(i) => seed(cd, *group, i, val),
            None => Dual1::<N>::constant(val),
        }
    });
    let schedule = EventSchedule::for_subject(subject, model.pk_model, &eff_doses, &dose_lagtimes);
    let conc = event_driven_sens_with_doses_g::<Dual1<N>>(
        model.pk_model,
        subject,
        &schedule,
        &eff_doses,
        &dose_inf_dual,
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

    // Analytic `[initial_conditions]` impulse under IOV — η-gradient only (#486): the inner
    // twin of the `run_obs_iov` outer impulse. Amount `A₀` from the BSV snapshot (`bsv_amount`,
    // κ = 0) via `eval_scale_dual1` (seeds η_bsv on axes `0..n_eta`), decay kernel per obs from
    // that observation's occasion source (κ carried); their `Dual1<N>` product's `∂/∂stacked-η`
    // is folded onto every pre-reset observation's η-gradient, keeping the inner EBE gradient
    // consistent with the outer walk and with production's `add_analytical_init_with`.
    if let Some((pk_bsv, cd_bsv)) = bsv_amount {
        let first_reset = subject
            .reset_times
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let eta_bsv = &stacked_eta[..n_eta];
        let cov = &subject.covariates;
        for init in &model.analytical_init {
            let Some(prog) = &init.amount_deriv else {
                continue;
            };
            let a0_var_duals: Vec<Dual1<N>> = prog
                .var_to_pk_slot()
                .iter()
                .map(|&s| match slot_row[s] {
                    Some(i) => seed(
                        cd_bsv,
                        None,
                        i,
                        pk_bsv.values.get(s).copied().unwrap_or(0.0),
                    ),
                    None => Dual1::<N>::constant(pk_bsv.values.get(s).copied().unwrap_or(0.0)),
                })
                .collect();
            let a0 = prog.eval_scale_dual1::<N>(theta, eta_bsv, cov, &a0_var_duals);
            if !a0.value.is_finite() {
                continue;
            }
            for (j, o) in out.iter_mut().enumerate() {
                let t = subject.obs_times[j];
                if t < 0.0 || t >= first_reset {
                    continue;
                }
                let (pk_d, cd_d, group_d) = &sources[obs_src[j]];
                let dv = |slot: usize, val: f64| -> Dual1<N> {
                    match slot_row[slot] {
                        Some(i) => seed(cd_d, *group_d, i, val),
                        None => Dual1::<N>::constant(val),
                    }
                };
                let c = crate::pk::analytical_init_concentration_g::<Dual1<N>>(
                    model.pk_model,
                    init.cmt,
                    a0,
                    Dual1::<N>::constant(t),
                    dv(PK_IDX_CL, pk_d.cl()),
                    dv(PK_IDX_V, pk_d.v()),
                    dv(PK_IDX_Q, pk_d.q()),
                    dv(PK_IDX_V2, pk_d.v2()),
                    dv(PK_IDX_KA, pk_d.ka()),
                    dv(PK_IDX_Q3, pk_d.q3()),
                    dv(PK_IDX_V3, pk_d.v3()),
                );
                o.f += c.value;
                for p in 0..n_stacked {
                    o.df_deta[p] += c.grad[p];
                }
            }
        }
    }

    // η-only `ExpressionScale` `obs_scale` quotient, one scale jet per occasion group —
    // the first-order counterpart of the `run_obs_iov` post-walk step (#575/#590).
    if let Some(scale_groups) = scale_groups {
        let ScalingSpec::ExpressionScale {
            deriv: Some(sprog), ..
        } = &model.scaling
        else {
            return None; // gate guarantees this arm; defensive
        };
        let eta_bsv = &stacked_eta[..n_eta];
        let cov = &subject.covariates;
        // η-only per-group scale jets — same seeding as the outer walk (shared helper),
        // over the stacked-η axes only.
        let group_scale = build_iov_group_scale_jets::<Dual1<N>>(
            scale_groups,
            slot_row,
            sprog.var_to_pk_slot(),
            |cd, g, i, v| seed(cd, Some(g), i, v),
            |var_duals| sprog.eval_scale_dual1::<N>(theta, eta_bsv, cov, var_duals),
        );
        for (j, o) in out.iter_mut().enumerate() {
            let g = sources[obs_src[j]].2?;
            apply_scale_quotient_grad_iov::<N>(o, &group_scale[g], n_stacked);
        }
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
    // `analytical_supported_core` (not `analytical_supported`) so a `TIME` model doesn't
    // recurse: `analytical_supported` calls back here for the `uses_time_builtin` case
    // (#637 review #1).
    if !analytical_supported_core(model) || model.has_lagtime() || model.log_transform {
        return false;
    }
    // The TV-cov event-driven walk now layers the analytic `[initial_conditions]` impulse
    // (#486, branch H): `subject_sensitivities_tvcov` / `subject_eta_grad_tvcov` fold in
    // `A₀·kernel(t, pk)` via `apply_tvcov_init_{outer,inner}` — the exact `(θ,η)` jet at the
    // subject-static `t = 0` snapshot — BEFORE scaling, matching production's
    // `add_analytical_init` on `compute_predictions_with_tv` (which uses that same snapshot for
    // both amount and decay, so the impulse is independent of the TV-cov switching). Require
    // `init_supported` so the `(θ,η)` axis count fits the `dispatch_init_impulse!` table (the
    // impulse dispatch's `_` arm would otherwise silently drop it); a hand-built init with no
    // `amount_deriv` program still routes to FD there.
    if !init_supported(model) {
        return false;
    }
    // Bound total axes to the dual-walk dispatch cap so the outer (`m_dim`) and inner
    // (`n_eta`) TV-cov tables both resolve — matched analytic scope, no fixed-EBE FD
    // inner split (#449 re-review #2).
    if model.n_theta + model.n_eta > MAX_TVCOV_AXES {
        return false;
    }
    // Output scaling: `None` / constant `ScalarScale` (a uniform per-jet divisor), or an
    // η-dependent `ExpressionScale` whose subject-static quotient is applied post-walk by
    // `subject_sensitivities_tvcov` / `subject_eta_grad_tvcov` (the same shared
    // `apply_expression_scale_outer` / `_inner_dispatch` the dose-superposition path uses,
    // #486 — closes the TV-cov + expression-scale gap). `scaling_supported` bounds the
    // scale program to the dispatch table; `PerCmt` still routes to FD. LTBS is already
    // declined above, so LTBS + `ExpressionScale` (whose scale-then-log order the post-walk
    // quotient cannot reproduce) never reaches here.
    if !scaling_supported(model) {
        return false;
    }
    // An analytic Form C readout must also fit the event-walk's structural
    // `PkDual` slots here (#650). Without this, a readout whose params spill past
    // `PK_IDX_V3` would make `analytical_supported` (via the `uses_time` clause)
    // report "analytic" while every event-walk subject silently falls back to FD —
    // exactly the route/report drift #637 guards against.
    if !readout_tvcov_supported(model) {
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

/// Clamp a **dual** modeled `D`/`R` to its domain floor, keeping the jet in the
/// interior and dropping it at the wall — the dual mirror of
/// [`crate::types::clamp_above_floor`] used by the f64 [`DoseEvent::resolve_rate`].
/// A converged fit is interior, so this is the identity there; it only keeps a
/// transient `D ≤ 0` / `R ≤ 0` from turning `amt/D` into a non-finite rate (#486).
#[inline]
fn clamp_dual_floor<T: crate::sens::num::PkNum>(x: T, floor: f64) -> T {
    if x.val() > floor {
        x
    } else {
        T::from_f64(floor)
    }
}

/// True when every modeled `RATE=-1/-2` dose on `subject` has its backing
/// `D{cmt}`/`R{cmt}` PK slot declared. A modeled dose with no slot is an invalid
/// model (`check_model_data` rejects it upstream), but the analytic providers still
/// guard it here so they decline to FD rather than panicking in
/// [`DoseEvent::resolve_rate`]'s slot `.expect` (#486). Fixed subjects are trivially
/// `true`. Mirrors the ODE `ode_tvcov_supported` slot-presence clause.
fn modeled_doses_resolvable(model: &CompiledModel, subject: &Subject) -> bool {
    if subject.all_doses_fixed() {
        return true;
    }
    let attr_map = model.active_dose_attr_map();
    subject
        .doses
        .iter()
        .all(|d| d.is_fixed() || crate::sens::ode_provider::modeled_slot_for(attr_map, d).is_some())
}

/// Resolve each dose's modeled `RATE=-1/-2` (`R{cmt}`/`D{cmt}`) to a concrete
/// `rate`/`duration` for the walk's **value** path (the schedule's infusion break
/// times + the per-sub-interval active-window checks), using the caller-supplied f64
/// PK params at each dose (`dose_pk(k)`). A fixed subject borrows `subject.doses`
/// unchanged (#486). Single-sourced across the non-IOV and IOV walks — each supplies
/// `dose_pk` from its own already-evaluated per-dose params (so `pk_param_fn` is not
/// re-run here, #486 review). The matching derivative twin is
/// [`modeled_dose_inf_duals`].
fn resolve_eff_doses<'a>(
    model: &CompiledModel,
    subject: &'a Subject,
    dose_pk: impl Fn(usize) -> crate::types::PkParams,
) -> std::borrow::Cow<'a, [crate::types::DoseEvent]> {
    if subject.all_doses_fixed() {
        return std::borrow::Cow::Borrowed(&subject.doses);
    }
    let attr_map = model.active_dose_attr_map();
    std::borrow::Cow::Owned(
        (0..subject.doses.len())
            .map(|k| subject.doses[k].resolve_rate(attr_map, &dose_pk(k).values))
            .collect(),
    )
}

/// Shared FD-fallback gate for modeled `RATE=-1/-2` doses on the analytic closed-form
/// walk (#486, deduped from the per-provider guards). Returns `false` (⇒ decline to
/// FD) when a modeled dose has no backing `D{cmt}`/`R{cmt}` slot — an invalid model
/// `check_model_data` rejects upstream, guarded here so the provider declines rather
/// than panics in [`DoseEvent::resolve_rate`].
///
/// A modeled dose combined with a **steady-state** dose is now analytic too (#486):
/// [`crate::sens::propagate::equilibrate_ss_g`] threads the per-dose modeled-window dual
/// `(rate_bare, dur_bare)` into each equilibration cycle's active/quiet split, so the
/// moving infusion-end jet flows through the SS trough exactly as it does through the
/// current pulse (matching the ODE provider's `equilibrate_ss_state_g`).
fn modeled_dose_analytic_gate(model: &CompiledModel, subject: &Subject) -> bool {
    modeled_doses_resolvable(model, subject)
}

/// True when the modeled-infusion **end** times are separable — no two modeled doses
/// backed by *distinct* `D`/`R` PK slots resolve to the same infusion-end time. The
/// dual-`dt` walk encodes each moving end as one break carrying that dose's window
/// jet; two distinct-slot ends coinciding would need the single sub-interval to carry
/// two different dual lengths (one per compartment), which it cannot — so such a
/// (degenerate, measure-zero) subject declines to FD to stay exact (#486 review #2).
/// Same-slot coincident ends are fine: they share the `D`/`R` value ⇒ an identical
/// jet ⇒ they move together, and the walk's first-match `dual_pos` is exact. `eff`
/// are the resolved doses the walk sees; the modeled-slot identity is read from the
/// original `subject.doses` (`resolve_rate` erases `rate_mode` to `Fixed`).
fn modeled_ends_separable(
    model: &CompiledModel,
    subject: &Subject,
    eff: &[crate::types::DoseEvent],
) -> bool {
    let attr_map = model.active_dose_attr_map();
    let mut ends: Vec<(f64, usize)> = Vec::new();
    for (k, d) in subject.doses.iter().enumerate() {
        if let Some((_mode, slot)) = crate::sens::ode_provider::modeled_slot_for(attr_map, d) {
            let end = eff[k].time + eff[k].duration;
            if ends
                .iter()
                .any(|&(e, s)| s != slot && (e - end).abs() < 1e-9)
            {
                return false;
            }
            ends.push((end, slot));
        }
    }
    true
}

/// Per-dose modeled-infusion duals `(rate_bare, dur_bare)` for the event walk
/// ([`crate::sens::propagate::event_driven_sens_with_doses_g`], #486): `None` for a
/// fixed dose (or a fixed subject → empty), `Some` for a modeled `RATE=-1/-2` dose.
/// `rate_bare` is the F-agnostic infusion rate (`amt/D` for `RATE=-2`, `R` for
/// `RATE=-1`) whose PK-param jet feeds the injected rate magnitude; `dur_bare` is
/// the window length (`D`, or `amt/R`) whose jet makes the infusion **end** a moving
/// boundary. `slot_dual(dose_k, pk_slot)` supplies the seeded dual of the `D`/`R` PK
/// slot at dose `k`'s snapshot (`Dual2<M>` outer, `Dual1<N>` inner). A transient
/// non-positive `D`/`R` is clamped to its floor, mirroring the f64
/// [`DoseEvent::resolve_rate`].
fn modeled_dose_inf_duals<T: crate::sens::num::PkNum>(
    model: &CompiledModel,
    subject: &Subject,
    slot_dual: impl Fn(usize, usize) -> T,
) -> Vec<Option<(T, T)>> {
    if subject.all_doses_fixed() {
        return Vec::new();
    }
    let attr_map = model.active_dose_attr_map();
    subject
        .doses
        .iter()
        .enumerate()
        .map(|(k, d)| {
            let (mode, slot) = crate::sens::ode_provider::modeled_slot_for(attr_map, d)?;
            let sd = slot_dual(k, slot);
            let amt = T::from_f64(d.amt);
            match mode {
                crate::types::RateMode::ModeledDuration => {
                    let dur = clamp_dual_floor(sd, crate::types::DoseEvent::DURATION_FLOOR);
                    Some((amt / dur, dur))
                }
                crate::types::RateMode::ModeledRate => {
                    let rate = clamp_dual_floor(sd, crate::types::DoseEvent::RATE_FLOOR);
                    Some((rate, amt / rate))
                }
                crate::types::RateMode::Fixed => None,
            }
        })
        .collect()
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
/// Layer the analytic `[initial_conditions]` impulse onto a **TV-cov walk**'s `SubjectSens`
/// (#486, branch H — the closed-form analogue of #649's ODE `tvcov_init_state`). Production's
/// `add_analytical_init` (`pk/mod.rs`) adds `A₀·kernel(t, pk)` with a single subject-static
/// `t = 0` snapshot — `pk_param_fn(θ, η, subject.covariates, 0)` — for both the baseline amount
/// and its decay kernel, even for a TV-cov subject (the init impulse is independent of the
/// time-varying-covariate switching). So compute that snapshot's `ParamDerivs` and fold in the
/// impulse via the SAME [`apply_analytical_init_outer`] the dose-superposition provider uses,
/// BEFORE scaling. Returns `Some(())` (no-op) when the model has no `[initial_conditions]`, and
/// `None` when the exact `∂p/∂(θ,η)` can't be built (non-log-normal η with no program) so the
/// caller falls back to FD. `tvcov_analytical_supported`'s `init_supported` clause bounds the
/// axis count to the `dispatch_init_impulse!` table, so the `_` arm never silently drops it.
fn apply_tvcov_init_outer(
    sens: &mut SubjectSens,
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<()> {
    if model.analytical_init.is_empty() {
        return Some(());
    }
    let n_theta = model.n_theta;
    let n_eta = model.n_eta;
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let (pd, slots): (crate::sens::ode_provider::ParamDerivs, Vec<usize>) = match model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .filter(|prog| prog_covers_required_pk_slots(model, prog))
        .and_then(|prog| {
            crate::sens::ode_provider::param_derivatives_from_prog(prog, model, subject, theta, eta)
                .map(|pd| (pd, prog.pk_slots()))
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
            (pd, model.pk_indices.clone())
        }
    };
    dispatch_init_impulse!(
        n_theta + n_eta,
        apply_analytical_init_outer,
        sens,
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
    Some(())
}

/// Inner (`Dual1`, η-gradient only) counterpart of [`apply_tvcov_init_outer`] — layers the
/// analytic `[initial_conditions]` impulse onto a TV-cov walk's `ObsGrad` list before scaling,
/// via [`apply_analytical_init_inner`] on the subject-static `t = 0` snapshot's `∂p/∂η`.
fn apply_tvcov_init_inner(
    out: &mut [ObsGrad],
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<()> {
    if model.analytical_init.is_empty() {
        return Some(());
    }
    let n_eta = model.n_eta;
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let (dp_deta, slots): (Vec<Vec<f64>>, Vec<usize>) = match model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .filter(|prog| prog_covers_required_pk_slots(model, prog))
        .and_then(|prog| {
            crate::sens::ode_provider::param_derivatives_from_prog(prog, model, subject, theta, eta)
                .map(|pd| (pd.dp_deta, prog.pk_slots()))
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
    dispatch_init_impulse!(
        n_eta,
        apply_analytical_init_inner,
        out,
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
    Some(())
}

pub fn subject_sensitivities_tvcov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<SubjectSens> {
    // Routed here for time-varying covariates *or* oral infusion (#350/#400) — both
    // need the state-propagating walk rather than dose superposition. A structural
    // parameter reading the `TIME` built-in is piecewise/time-varying for the same
    // reason, so it routes here too even with no TV covariates (#486 / #610).
    if !tvcov_analytical_supported(model) || !subject_routes_to_event_walk(model, subject) {
        return None;
    }
    // Analytic Form C readout (#650) on a TV-cov / oral-infusion / TIME subject:
    // the event-driven Dual2 walk carries the compartment amount as a dual, so the
    // readout is served analytically here — provided every parameter it references
    // fits the eight structural `PkDual` slots (`readout_tvcov_supported`).
    // Otherwise fall back to FD (correct via the readout-aware f64 predictor).
    if model.analytic_readout.is_some() && !readout_tvcov_supported(model) {
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
    // Modeled-`RATE=-1/-2` doses — including combined with steady state (#486:
    // `equilibrate_ss_g` threads the modeled-window jet into its per-cycle active/quiet
    // split) — are served analytically below via the per-dose duals; only the
    // missing-slot case declines to FD via the shared gate.
    if !modeled_dose_analytic_gate(model, subject) {
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

    // Analytic Form C readout program (#650); `readout_tvcov_supported` (checked at
    // the gate) guarantees it fits the eight structural `PkDual` slots here.
    let readout = model
        .analytic_readout
        .as_ref()
        .and_then(|ar| ar.program.as_ref());
    macro_rules! disp {
        ($($m:literal),+) => {
            match m_dim {
                $($m => run_obs_tvcov::<$m>(
                    model, subject, theta, eta, prog, &slot_row, n_eta, n_theta, readout,
                ),)+
                _ => None,
            }
        };
    }
    let mut sens = disp!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    )?;

    // Analytic `[initial_conditions]` impulse (#486): layer `A₀·kernel(t, pk)` and its exact
    // `(θ,η)` jet onto every pre-reset observation BEFORE scaling — the same insertion order as
    // the f64 `pk::add_analytical_init` on the production TV-cov path (`compute_predictions_with_tv`).
    apply_tvcov_init_outer(&mut sens, model, subject, theta, eta)?;

    // Constant `ScalarScale` output divisor `f_scaled = f/k`: every derivative is
    // linear in `f` and `k` is constant, so the whole jet divides by `k` — matches
    // `pk::apply_scaling` (`pred /= s`) on the production TV-cov path.
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
    // η-dependent `ExpressionScale` divisor `s(θ, η)`: apply the subject-static quotient
    // `scaled_f = f/s` on the walked jet — the SAME shared quotient the dose-superposition
    // path uses (#486, closing the TV-cov + expression-scale gap). Production
    // `pk::apply_scaling` evaluates the scale once at the subject-static covariates and
    // `t = 0` params, mirrored inside the helper.
    apply_event_walk_expression_scale_outer(&mut sens, model, subject, prog, theta, eta)?;
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
    readout: Option<&crate::parser::model_parser::OdeOutputProgram>,
) -> Option<SubjectSens> {
    use crate::pk::event_driven::EventSchedule;
    use crate::sens::ode_provider::pd_from_program;
    use crate::sens::propagate::{event_driven_sens_with_doses_g, PkDual};

    // PK slot order the program differentiates (for seeding the modeled `D`/`R`
    // slot dual via `pk_slot_dual_outer`, #486).
    let slots = prog.pk_slots_ref();

    // Build the per-event PK-param duals at a covariate snapshot: evaluate the
    // program's `∂p/∂(θ, η)` (+ 2nd order) at `cov`, then seed each differentiated
    // PK slot on its `(θ, η)` dual axis (`θ_m → m`, `η_k → n_theta + k`); constants
    // otherwise. The θ-θ Hessian block is unused downstream (left zero), mirroring
    // the IOV / scale seeders.
    // A `TIME`-built-in structural parameter resolves `Op::PushTime` from the
    // model-time thread-local, which `pd_from_program`'s `Dual2` walk reads. Seed
    // it with the per-event time (gated on `uses_time`, exactly like the f64
    // `pk_param_fn` closure) so each event's PK-param duals — value AND derivatives
    // — are evaluated at that event's `TIME` (#486 / #610).
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let mk = |time: f64, cov: &std::collections::HashMap<String, f64>| -> PkDual<Dual2<M>> {
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
        let pd = pd_from_program::<M>(prog, model, cov, theta, eta);
        let pk = (model.pk_param_fn)(theta, eta, cov, time);
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
        .map(|k| mk(subject.doses[k].time, subject.dose_cov(k)))
        .collect();
    let pk_at_obs: Vec<PkDual<Dual2<M>>> = (0..subject.obs_times.len())
        .map(|j| mk(subject.obs_times[j], subject.obs_cov(j)))
        .collect();
    let pk_at_pk_only: Vec<PkDual<Dual2<M>>> = (0..subject.pk_only_times.len())
        .map(|m| mk(subject.pk_only_times[m], subject.pk_only_cov(m)))
        .collect();

    // No lagtime in TV-cov scope → zero dose lagtimes.
    let dose_lagtimes = vec![0.0; subject.doses.len()];
    // Modeled-`RATE=-1/-2` doses (#486): resolve to concrete rate/duration for the
    // schedule's break times, and build the per-dose duals so the injected rate and
    // the moving infusion-end boundary carry their `(θ,η)` jets. The per-dose f64 PK
    // params are evaluated once (`dose_pk`) and shared by the resolved window and the
    // dual's value, rather than re-running `pk_param_fn` per dose (#486 review #4).
    // Fixed subjects borrow `subject.doses` and pass no duals (identical to the
    // pre-#486 path).
    let dose_pk: Vec<crate::types::PkParams> = (0..subject.doses.len())
        .map(|k| {
            let _guard = crate::parser::model_parser::ModelTimeGuard::enter_if(
                uses_time,
                subject.doses[k].time,
            );
            (model.pk_param_fn)(theta, eta, subject.dose_cov(k), subject.doses[k].time)
        })
        .collect();
    let eff_doses = resolve_eff_doses(model, subject, |k| dose_pk[k]);
    // Distinct-slot coincident infusion ends aren't representable by the single-`dt`
    // walk — decline to FD (#486 review #2).
    if !modeled_ends_separable(model, subject, &eff_doses) {
        return None;
    }
    let schedule = EventSchedule::for_subject(subject, model.pk_model, &eff_doses, &dose_lagtimes);
    let dose_inf_dual = modeled_dose_inf_duals::<Dual2<M>>(model, subject, |k, dr_slot| {
        let cov = subject.dose_cov(k);
        let _guard =
            crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, subject.doses[k].time);
        let pd = pd_from_program::<M>(prog, model, cov, theta, eta);
        pk_slot_dual_outer::<M>(
            dr_slot,
            dose_pk[k].values.get(dr_slot).copied().unwrap_or(0.0),
            &pd,
            slots,
            n_theta,
            n_eta,
        )
    });
    let conc = event_driven_sens_with_doses_g::<Dual2<M>>(
        model.pk_model,
        subject,
        &schedule,
        &eff_doses,
        &dose_inf_dual,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    // Analytic Form C readout (#650): scratch for the per-observation dual eval.
    let mut ro_state: Vec<Dual2<M>> = Vec::new();
    let mut ro_vars: Vec<Dual2<M>> = Vec::new();
    let mut ro_stack: Vec<Dual2<M>> = Vec::new();

    let mut obs_out = Vec::with_capacity(conc.len());
    for (j, c) in conc.iter().enumerate() {
        // Analytic Form C readout: replace the central concentration jet with
        // `y = <expr>`. The per-observation PK duals (`pk_at_obs[j]`) already carry
        // each parameter's `∂/∂(θ,η)` in the walk basis — including a non-structural
        // `BMAX`/`KD` seeded into its allocated structural slot — so the readout's
        // derivatives compose directly. The central amount is `concentration × V`.
        let c: Dual2<M> = if let Some(ro) = readout {
            let pkd = &pk_at_obs[j];
            // Flat PK-slot dual vector for `eval_output_g`'s `indiv_to_pk` lookups;
            // the eight structural slots come from the walk's `PkDual`, every higher
            // slot is a static constant (`readout_tvcov_supported` guarantees no
            // readout param maps past `PK_IDX_V3`).
            let params: [Dual2<M>; N_PK] = std::array::from_fn(|s| match s {
                PK_IDX_CL => pkd.cl,
                PK_IDX_V => pkd.v,
                PK_IDX_Q => pkd.q,
                PK_IDX_V2 => pkd.v2,
                PK_IDX_KA => pkd.ka,
                PK_IDX_F => pkd.f,
                PK_IDX_Q3 => pkd.q3,
                PK_IDX_V3 => pkd.v3,
                _ => Dual2::<M>::constant(0.0),
            });
            eval_readout_jet::<Dual2<M>>(
                ro,
                *c,
                &params,
                subject.obs_cov(j),
                &mut ro_state,
                &mut ro_vars,
                &mut ro_stack,
            )
        } else {
            *c
        };
        let c = &c;
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
    if !tvcov_analytical_supported(model) || !subject_routes_to_event_walk(model, subject) {
        return None;
    }
    // Analytic Form C readout (#650): served analytically on the event-walk when it
    // fits the structural `PkDual` slots (matches the outer gate); otherwise FD.
    if model.analytic_readout.is_some() && !readout_tvcov_supported(model) {
        return None;
    }
    if subject.has_resets() && subject.doses.iter().any(|d| d.ss) {
        return None;
    }
    // Modeled-`RATE=-1/-2` doses are served analytically by the walk now (#486): the
    // dual rate + moving infusion-end boundary (`run_obs_grad_tvcov` builds the
    // per-dose duals). The steady-state combination is analytic too (#486:
    // `equilibrate_ss_g` threads the modeled-window jet into its per-cycle active/quiet
    // split); only the missing-slot case declines to FD via the shared gate.
    if !modeled_dose_analytic_gate(model, subject) {
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

    let readout = model
        .analytic_readout
        .as_ref()
        .and_then(|ar| ar.program.as_ref());
    macro_rules! disp {
        ($($n:literal),+) => {
            match n_eta {
                $($n => run_obs_grad_tvcov::<$n>(model, subject, theta, eta, prog, &slot_row, n_eta, cached_schedule, readout),)+
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

    // Analytic `[initial_conditions]` impulse (#486): η-gradient only, layered BEFORE scaling —
    // the inner twin of the outer TV-cov init impulse (keeps the inner EBE gradient consistent
    // with the outer, and with production's `add_analytical_init`).
    apply_tvcov_init_inner(&mut out, model, subject, theta, eta)?;

    // Constant `ScalarScale` divisor: `∂(f/k)/∂η = (∂f/∂η)/k` (η-independent `k`),
    // matching the outer TV-cov path and `pk::apply_scaling`. (LTBS keeps the FD inner —
    // gated upstream; `ExpressionScale` is applied below.)
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
    // η-dependent `ExpressionScale` divisor: the η-only quotient — the light counterpart
    // of the outer path, matching the static inner path (#486). Subject-static scale from
    // `pk`/`∂p/∂η` at `t = 0`, applied to every observation (shared helper).
    apply_event_walk_expression_scale_inner(&mut out, model, subject, prog, theta, eta)?;
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
    readout: Option<&crate::parser::model_parser::OdeOutputProgram>,
) -> Option<Vec<ObsGrad>> {
    use crate::pk::event_driven::EventSchedule;
    use crate::sens::ode_provider::param_derivatives_at_cov;
    use crate::sens::propagate::{event_driven_sens_with_doses_g, PkDual};

    // The dispatch sizes `N = n_eta` exactly, so the `.min(N)` clamps below are
    // no-ops — flat `0..n_eta` loops (#449 re-review #5, mirroring #15).
    debug_assert_eq!(N, n_eta);

    // PK slot order the program differentiates (for the modeled `D`/`R` slot dual).
    let slots = prog.pk_slots_ref();

    // Seed the model-time thread-local with the per-event time so a `TIME`-built-in
    // structural parameter resolves to that event's time in both the f64 value and
    // the `Dual1` η-derivative walk (gated on `uses_time`, like the f64 closure;
    // #486 / #610).
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let mk = |time: f64,
              cov: &std::collections::HashMap<String, f64>|
     -> Option<PkDual<Dual1<N>>> {
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
        // `None` above the param-derivative dispatch cap (n_axes > 16): decline so
        // the inner loop falls back to FD rather than panicking (#449 review #1).
        let pd = param_derivatives_at_cov(prog, model, cov, theta, eta)?;
        let pk = (model.pk_param_fn)(theta, eta, cov, time);
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
        .map(|k| mk(subject.doses[k].time, subject.dose_cov(k)))
        .collect::<Option<Vec<_>>>()?;
    let pk_at_obs: Vec<PkDual<Dual1<N>>> = (0..subject.obs_times.len())
        .map(|j| mk(subject.obs_times[j], subject.obs_cov(j)))
        .collect::<Option<Vec<_>>>()?;
    let pk_at_pk_only: Vec<PkDual<Dual1<N>>> = (0..subject.pk_only_times.len())
        .map(|m| mk(subject.pk_only_times[m], subject.pk_only_cov(m)))
        .collect::<Option<Vec<_>>>()?;

    // Modeled-`RATE=-1/-2` doses (#486): resolve to concrete rate/duration and build
    // the per-dose duals (dual rate + moving infusion-end boundary). A modeled dose's
    // resolved window varies with η across inner BFGS steps, so its schedule cannot be
    // cached — rebuild from the resolved `eff_doses` each call. Fixed subjects keep the
    // cached (η-invariant) schedule.
    // The per-dose f64 PK params are evaluated once (`dose_pk`) and shared by the
    // resolved window and the dual's value, rather than re-running `pk_param_fn` per
    // dose on the inner hot path (#486 review #4).
    let dose_pk: Vec<crate::types::PkParams> = (0..subject.doses.len())
        .map(|k| {
            let _guard = crate::parser::model_parser::ModelTimeGuard::enter_if(
                uses_time,
                subject.doses[k].time,
            );
            (model.pk_param_fn)(theta, eta, subject.dose_cov(k), subject.doses[k].time)
        })
        .collect();
    let eff_doses = resolve_eff_doses(model, subject, |k| dose_pk[k]);
    // Distinct-slot coincident infusion ends aren't representable by the single-`dt`
    // walk — decline to FD (#486 review #2).
    if !modeled_ends_separable(model, subject, &eff_doses) {
        return None;
    }
    let dose_inf_dual = modeled_dose_inf_duals::<Dual1<N>>(model, subject, |k, dr_slot| {
        let cov = subject.dose_cov(k);
        let _guard =
            crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, subject.doses[k].time);
        let val = dose_pk[k].values.get(dr_slot).copied().unwrap_or(0.0);
        match param_derivatives_at_cov(prog, model, cov, theta, eta) {
            Some(pd) => pk_slot_dual_inner::<N>(dr_slot, val, &pd.dp_deta, slots, n_eta),
            None => Dual1::constant(val),
        }
    });

    // The event schedule is invariant across inner BFGS steps for a fixed subject (it
    // depends only on the subject + doses + zero lagtimes, not on η). Reuse the schedule
    // the inner optimizer cached once per subject when available; rebuild locally for
    // modeled doses (window depends on η) or when no cache was passed (#449 / #486).
    let owned_schedule;
    let schedule: &EventSchedule = match cached_schedule {
        Some(s) if subject.all_doses_fixed() => s,
        _ => {
            let dose_lagtimes = vec![0.0; subject.doses.len()];
            owned_schedule =
                EventSchedule::for_subject(subject, model.pk_model, &eff_doses, &dose_lagtimes);
            &owned_schedule
        }
    };
    let conc = event_driven_sens_with_doses_g::<Dual1<N>>(
        model.pk_model,
        subject,
        schedule,
        &eff_doses,
        &dose_inf_dual,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    // Analytic Form C readout (#650): per-observation dual eval scratch (Dual1).
    let mut ro_state: Vec<Dual1<N>> = Vec::new();
    let mut ro_vars: Vec<Dual1<N>> = Vec::new();
    let mut ro_stack: Vec<Dual1<N>> = Vec::new();

    let mut out = Vec::with_capacity(conc.len());
    for (j, c) in conc.iter().enumerate() {
        // Analytic Form C readout (η-gradient): mirror the outer `run_obs_tvcov`
        // over `Dual1<N>`. Central amount = concentration × V; the readout's
        // `∂y/∂η` composes from the per-obs PK duals and the readout expression.
        let c: Dual1<N> = if let Some(ro) = readout {
            let pkd = &pk_at_obs[j];
            let params: [Dual1<N>; N_PK] = std::array::from_fn(|s| match s {
                PK_IDX_CL => pkd.cl,
                PK_IDX_V => pkd.v,
                PK_IDX_Q => pkd.q,
                PK_IDX_V2 => pkd.v2,
                PK_IDX_KA => pkd.ka,
                PK_IDX_F => pkd.f,
                PK_IDX_Q3 => pkd.q3,
                PK_IDX_V3 => pkd.v3,
                _ => Dual1::<N>::constant(0.0),
            });
            eval_readout_jet::<Dual1<N>>(
                ro,
                *c,
                &params,
                subject.obs_cov(j),
                &mut ro_state,
                &mut ro_vars,
                &mut ro_stack,
            )
        } else {
            *c
        };
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
    // A `TIME`-built-in structural parameter routes through the per-event walk
    // too (piecewise/time-varying), mirroring the outer provider (#486 / #610).
    if subject_routes_to_event_walk(model, subject) {
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
    let transit = matches!(model.pk_model, PkModel::OneCptTransit);

    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);

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

    // Analytic Form C readout program (#650); the gate guarantees it is dual-
    // evaluable and depot-free, so the inner `run_obs_grad` serves it exactly.
    let readout = model
        .analytic_readout
        .as_ref()
        .and_then(|ar| ar.program.as_ref());
    macro_rules! disp {
        ($($n:literal),+) => {
            match slots.len() {
                $($n => Some(run_obs_grad::<$n>(
                    &seed_dim, &pk, oral, two_cpt, three_cpt, transit, subject, &dp_deta, n_eta,
                    readout,
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
    transit: bool,
    subject: &Subject,
    dp_deta: &[Vec<f64>],
    n_eta: usize,
    readout: Option<&crate::parser::model_parser::OdeOutputProgram>,
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
    // Transit `n`/`mtt` (#386), seeded like the other structural params.
    let n_d = dv(PK_IDX_N, pk.n_transit());
    let mtt_d = dv(PK_IDX_MTT, pk.mtt());

    // Analytic Form C readout (#650): PK-slot dual vector + scratch, built once
    // (subject-static). Mirrors the outer `run_obs`, on `Dual1<N>` (η-grad only).
    let ro_pk_duals: Option<[Dual1<N>; N_PK]> = readout.map(|_| {
        std::array::from_fn(|s| {
            let val = pk.values.get(s).copied().unwrap_or(0.0);
            match seed_dim[s] {
                Some(k) => Dual1::<N>::var(val, k),
                None => Dual1::<N>::constant(val),
            }
        })
    });
    let mut ro_state: Vec<Dual1<N>> = Vec::new();
    let mut ro_vars: Vec<Dual1<N>> = Vec::new();
    let mut ro_stack: Vec<Dual1<N>> = Vec::new();
    let mut out = Vec::with_capacity(subject.obs_times.len());
    for (obs_i, &t_obs) in subject.obs_times.iter().enumerate() {
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
            let c = if transit {
                one_cpt_transit_conc_g(dose, elapsed, cl_d, v1_d, n_d, mtt_d, f_d)
            } else if three_cpt {
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

        // Analytic Form C readout (#650): replace the central concentration with
        // `y = <expr>` over `Dual1<N>` (central amount = conc × V), mirroring the
        // outer `run_obs`. Its `∂y/∂pk` then rides the `dp_deta` chain below.
        let (fval, g) = if let (Some(prog), Some(pkd)) = (readout, ro_pk_duals.as_ref()) {
            let conc = Dual1::<N> {
                value: fval,
                grad: g,
            };
            let y = eval_readout_jet::<Dual1<N>>(
                prog,
                conc,
                pkd,
                subject.obs_cov(obs_i),
                &mut ro_state,
                &mut ro_vars,
                &mut ro_stack,
            );
            (y.value, y.grad)
        } else {
            (fval, g)
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

/// First-order, η-only counterpart of [`apply_scale_quotient_row`] for the inner EBE
/// gradient: `∂(f/s)/∂η_k = (∂f/∂η_k)/s − f·(∂s/∂η_k)/s²` over the stacked η axes
/// (`s.grad[k]`, no `n_theta` offset — the inner scale jet has no θ block). Shared by the
/// closed-form IOV inner ([`run_obs_iov_eta`]) and the ODE IOV inner
/// ([`crate::sens::ode_provider::run_subject_iov_eta`]) — the IOV analogue of the η-block
/// of `apply_expression_scale_inner` (#575/#590).
pub(crate) fn apply_scale_quotient_grad_iov<const N: usize>(
    o: &mut ObsGrad,
    s: &Dual1<N>,
    n_stacked: usize,
) {
    let f = o.f;
    let inv = 1.0 / s.value;
    let inv2 = inv * inv;
    for k in 0..n_stacked {
        o.df_deta[k] = o.df_deta[k] * inv - f * s.grad[k] * inv2;
    }
    o.f = f * inv;
}

/// Build one `ExpressionScale` scale jet per occasion group for the closed-form IOV
/// walk. For each group's static-snapshot `(pk, cd)` it seeds a slot-indexed PK-param
/// dual vector — constant everywhere except the slots that carry a random effect
/// (`slot_row[slot] = Some(row)`), which are differentiated via `seed_slot(cd, group,
/// row, value)` — then hands the vector to the shared [`build_iov_scale_jets`] to
/// evaluate the scale program's dual. Single source for the per-group seeding shared by
/// the `Dual2` outer ([`run_obs_iov`]) and `Dual1` inner ([`run_obs_iov_eta`]) walks, so
/// a change to how a group's scale jet is seeded (a new PK slot, a snapshot fix) applies
/// to both at once and the inner/outer scale handling cannot silently diverge (#651
/// review #2). `seed_slot` and `eval` supply the dual-type-specific pieces (θ+η+κ vs
/// η-only seeding; `eval_scale_dual` vs `eval_scale_dual1`).
///
/// [`build_iov_scale_jets`]: crate::sens::ode_provider::build_iov_scale_jets
fn build_iov_group_scale_jets<T: crate::sens::num::PkNum>(
    scale_groups: &[(crate::types::PkParams, CombinedDerivs)],
    slot_row: &[Option<usize>; N_PK],
    var_to_pk_slot: &[usize],
    seed_slot: impl Fn(&CombinedDerivs, usize, usize, f64) -> T,
    eval: impl FnMut(&[T]) -> T,
) -> Vec<T> {
    let groups: Vec<Vec<T>> = scale_groups
        .iter()
        .enumerate()
        .map(|(g, (pk, cd))| {
            let mut seeded: Vec<T> = pk.values.iter().map(|&v| T::from_f64(v)).collect();
            for slot in 0..N_PK {
                if let Some(i) = slot_row[slot] {
                    seeded[slot] = seed_slot(cd, g, i, pk.values[slot]);
                }
            }
            seeded
        })
        .collect();
    crate::sens::ode_provider::build_iov_scale_jets::<T>(&groups, var_to_pk_slot, eval)
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

/// Apply an η-dependent `ExpressionScale` `obs_scale` to an already-walked **outer** jet
/// on the event-driven path (time-varying covariate / `TIME`). Builds the subject-static
/// scale reference — `pk` (value) AND `pd` (`∂p/∂(θ,η)`) at the subject covariates and
/// `t = 0`, both under one explicit [`ModelTimeGuard`] so a nested outer guard can never
/// leave the value pinned at `t = 0` while the derivative reads a stale `TIME` (#637
/// review #3) — then applies the shared quotient. A no-op unless the model carries a
/// differentiable `ExpressionScale`; `None` if the scale program's axis count is outside
/// the dispatch table (caller falls back to FD). One home for the `t = 0` wiring shared by
/// the closed-form (`subject_sensitivities_tvcov`) and ODE (`run_subject_tvcov`) walks so
/// a future change lands in one place (#637 review #6).
pub(crate) fn apply_event_walk_expression_scale_outer(
    sens: &mut SubjectSens,
    model: &CompiledModel,
    subject: &Subject,
    prog: &crate::parser::model_parser::IndivParamProgram,
    theta: &[f64],
    eta: &[f64],
) -> Option<()> {
    let ScalingSpec::ExpressionScale {
        deriv: Some(scale_prog),
        ..
    } = &model.scaling
    else {
        return Some(());
    };
    let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter(0.0);
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let pd =
        crate::sens::ode_provider::param_derivatives_from_prog(prog, model, subject, theta, eta)?;
    apply_expression_scale_outer(
        sens,
        scale_prog,
        &pk,
        &pd,
        prog.pk_slots_ref(),
        theta,
        eta,
        &subject.covariates,
        model.n_theta,
        model.n_eta,
    );
    Some(())
}

/// Inner (`Dual1`, η-only) counterpart of [`apply_event_walk_expression_scale_outer`] —
/// the `∂(f/s)/∂η` quotient on the light gradient, same `t = 0` guarded reference (#637
/// review #3 / #6).
pub(crate) fn apply_event_walk_expression_scale_inner(
    out: &mut [ObsGrad],
    model: &CompiledModel,
    subject: &Subject,
    prog: &crate::parser::model_parser::IndivParamProgram,
    theta: &[f64],
    eta: &[f64],
) -> Option<()> {
    let ScalingSpec::ExpressionScale {
        deriv: Some(scale_prog),
        ..
    } = &model.scaling
    else {
        return Some(());
    };
    let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter(0.0);
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let dp_deta = crate::sens::ode_provider::param_eta_derivatives_from_prog(
        prog, model, subject, theta, eta,
    )?;
    apply_expression_scale_inner_dispatch(
        out,
        scale_prog,
        &pk,
        &dp_deta,
        prog.pk_slots_ref(),
        theta,
        eta,
        &subject.covariates,
        model.n_eta,
    );
    Some(())
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

/// True when a subject must be served by the event-driven per-event walk rather than
/// dose superposition: it carries **time-varying covariates**, an **oral infusion**
/// (#350/#400 forced responses), or the model reads the **`TIME` built-in** in a
/// structural parameter — all of which make the PK parameters per-event dynamic. Single
/// source of truth so the outer gradient (`subject_sensitivities_impl`), the inner EBE
/// gradient (`subject_eta_grad_impl` and `inner_optimizer::analytic_inner_grad_supported`),
/// and the two TV-walk triggers cannot drift to different routing decisions (#637 round-2
/// review #3).
pub(crate) fn subject_routes_to_event_walk(model: &CompiledModel, subject: &Subject) -> bool {
    subject.has_tv_covariates()
        || subject_has_oral_infusion(model, subject)
        || crate::parser::model_parser::compiled_model_uses_time_builtin(model)
        // A modeled-`RATE=-1/-2` dose (`R{cmt}`/`D{cmt}`) resolves its rate/window
        // from PK params, so the infusion *end* is a moving boundary — the
        // event-driven walk carries it via the exact dual window length, which
        // dose superposition can't (#486). This is the closed-form twin of the ODE
        // `ode_tvcov_supported`'s `has_modeled_dose` clause. An oral modeled infusion
        // is already covered by `subject_has_oral_infusion`; this adds the IV route.
        || !subject.all_doses_fixed()
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
    // Time-varying covariates make the PK parameters switch mid-decay, which dose
    // superposition can't express — route them to the event-driven Dual2 walk
    // (IOV-minus-κ). The same walk carries the oral-infusion forced responses
    // (#350 depot-bypass central / #400 zero-order depot) that the superposition
    // kernels don't, so oral-infusion subjects route there too — mirroring
    // production's `compute_predictions` dispatch. The returned `SubjectSens` has
    // the same `(η, θ)` shape, so the outer gradient / inner Jacobian consume it
    // identically.
    //
    // A `TIME`-built-in structural parameter is piecewise/time-varying too — it
    // must route through the per-event walk even with no TV covariates, since the
    // dose-superposition path below would freeze it at one `t=0` snapshot (#486 /
    // #610).
    if subject_routes_to_event_walk(model, subject) {
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
    let transit = matches!(model.pk_model, PkModel::OneCptTransit);

    // PK parameter values at (θ, η): pk_s = tv_s·exp(sel·η). pk_param_fn folds η.
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);

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
        match model.pk_model {
            PkModel::OneCptIv => Some(ExKind::OneCptIv),
            PkModel::OneCptOral => Some(ExKind::OneCptOral),
            PkModel::TwoCptIv => Some(ExKind::TwoCptIv),
            PkModel::TwoCptOral => Some(ExKind::TwoCptOral),
            PkModel::ThreeCptIv => Some(ExKind::ThreeCptIv),
            PkModel::ThreeCptOral => Some(ExKind::ThreeCptOral),
            // Transit has no hand-written explicit kernel; use the generic Dual2 path.
            PkModel::OneCptTransit => None,
        }
    };

    // Dispatch on the differentiated-parameter count so the dual width is
    // right-sized. `pk_indices.len()` ≤ `N_PK` (the fixed PK slot table).
    // Analytic Form C readout program (#650), if this model has one; the
    // `analytical_supported` gate guarantees it is `Single` + dual-evaluable and
    // does not read the depot amount, so `run_obs` can serve it exactly.
    let readout = model
        .analytic_readout
        .as_ref()
        .and_then(|ar| ar.program.as_ref());
    macro_rules! disp {
        ($($n:literal),+) => {
            match slots.len() {
                $($n => Some(SubjectSens {
                    obs: run_obs::<$n>(
                        &seed_dim, &pk, oral, two_cpt, three_cpt, transit, explicit_kind, subject,
                        &pd, n_eta, n_theta, readout,
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
    transit: bool,
    explicit_kind: Option<ExKind>,
    subject: &Subject,
    pd: &crate::sens::ode_provider::ParamDerivs,
    n_eta: usize,
    n_theta: usize,
    readout: Option<&crate::parser::model_parser::OdeOutputProgram>,
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
    // Analytic Form C readout (#650): the PK-parameter dual vector (indexed by PK
    // slot, so `eval_output_g` can pull `V`, binding constants, … via its
    // `indiv_to_pk` plan) and reusable scratch. Subject-static, so built once.
    let ro_pk_duals: Option<[Dual2<N>; N_PK]> = readout.map(|_| {
        std::array::from_fn(|s| {
            let val = pk.values.get(s).copied().unwrap_or(0.0);
            match seed_dim[s] {
                Some(k) => Dual2::<N>::var(val, k),
                None => Dual2::<N>::constant(val),
            }
        })
    });
    let mut ro_state: Vec<Dual2<N>> = Vec::new();
    let mut ro_vars: Vec<Dual2<N>> = Vec::new();
    let mut ro_stack: Vec<Dual2<N>> = Vec::new();
    let mut out = Vec::with_capacity(subject.obs_times.len());
    for (obs_i, &t_obs) in subject.obs_times.iter().enumerate() {
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
            // Transit `n`/`mtt` (#386), seeded like the other structural params.
            let n_d = dv(PK_IDX_N, pk.n_transit());
            let mtt_d = dv(PK_IDX_MTT, pk.mtt());

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
                let c = if transit {
                    one_cpt_transit_conc_g(dose, elapsed, cl_d, v1_d, n_d, mtt_d, f_d)
                } else if three_cpt {
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

        // Analytic Form C readout (#650): replace the central concentration jet
        // with `y = <expr>` over `Dual2<N>`. The central compartment **amount** is
        // `concentration × V` (so a `central / V` readout recovers the
        // concentration and an additive term layers on); the readout's `∂y/∂pk` /
        // `∂²y/∂pk²` then ride the same `pd` chain to `(θ, η)` below. Covariates
        // come from the per-observation snapshot (a per-row `FREE`-style flag).
        let (fval, g, h) = if let (Some(prog), Some(pkd)) = (readout, ro_pk_duals.as_ref()) {
            let conc = Dual2::<N> {
                value: fval,
                grad: g,
                hess: h,
            };
            let y = eval_readout_jet::<Dual2<N>>(
                prog,
                conc,
                pkd,
                subject.obs_cov(obs_i),
                &mut ro_state,
                &mut ro_vars,
                &mut ro_stack,
            );
            (y.value, y.grad, y.hess)
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
        // Correlated residual (`block_sigma`) is now analytic on the outer loop (#627):
        // the dense assembly reduces to the scalar path fed correlation-aware `(r,d,d2)`.
        // (Previously this predicate short-circuited to FD on any residual correlation.)
        let block_sigma = parse_model_string(
            "[parameters]\n  theta TVCL(1.0, 0.01, 10.0)\n  theta TVV(10.0, 0.1, 100.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.05, 1.00]\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n[fit_options]\n  method = focei\n",
        )
        .expect("parse block_sigma model");
        assert!(!block_sigma.residual_correlations.is_empty());
        assert!(analytic_outer_gradient_available(&block_sigma));
    }

    /// The `TIME` built-in makes a structural parameter piecewise/time-varying, so
    /// the subject routes through the event-driven per-event PK walk rather than
    /// dose superposition. As of #486 every analytic provider seeds the per-event time
    /// and serves TIME: **closed-form non-IOV** (`analytical_supported` /
    /// `tvcov_analytical_supported`), **closed-form IOV** (`iov_analytical_supported`),
    /// **non-IOV ODE** (`ode_analytical_supported`), and **ODE IOV** (`ode_iov_supported`,
    /// via the per-event stacked walk) — each validated against FD of production
    /// (`time_builtin_provider_matches_fd_of_production`,
    /// `iov_time_builtin_provider_matches_fd_of_predict_iov`,
    /// `ode_time_builtin_provider_matches_fd_of_production`,
    /// `ode_iov_time_builtin_provider_matches_fd_of_predict_iov`).
    ///
    /// `TIME` composes with an η-dependent `ExpressionScale` obs_scale too — the
    /// event-driven walk applies the subject-static scale quotient post-walk (validated
    /// in `expression_scale_on_event_walk_matches_fd_closed_form`,
    /// `ode_expression_scale_on_event_walk_matches_production`, and
    /// `ode_iov_time_expression_scale_matches_fd_of_predict_iov`). The direct
    /// `pk(...=TIME)` mapping is now desugared into a synthetic `__ferx_pktime_*`
    /// individual parameter and served analytically too (see the `ANALYTICAL_TIME_DIRECT`
    /// block below and `time_builtin_direct_pk_mapping_matches_fd_of_production`). `TIME`
    /// combined with an ODE `init(...)` baseline (#662) or a built-in input-rate forcing
    /// (#643) is now analytic on the event-driven walk too — the old model-level declines
    /// were stale once those features landed (validated by `ode_time_builtin_with_init_*`
    /// and `ode_time_builtin_with_first_order_*`); this test now pins the analytic route for
    /// `TIME + init(...)` at the model gate. The pre-existing scale fallbacks (LTBS +
    /// `ExpressionScale`; closed-form IOV + any scaling) are unchanged and independent of
    /// `TIME`. The non-`TIME` twin of each model must stay supported, proving the guards
    /// are specific (#486 / #610).
    #[test]
    fn time_builtin_indiv_params_analytic_routes() {
        // Analytical 1-cpt IV: a `$PK IF(TIME...)`-style switch on CL.
        const ANALYTICAL_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(5.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        const ANALYTICAL_NO_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        // IOV (n_kappa > 0) 1-cpt oral with the same switch.
        const IOV_TIME: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVCL_LATE(0.1, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  if (TIME > 24.0) {
    CL = TVCL_LATE * exp(ETA_CL + KAPPA_CL)
  } else {
    CL = TVCL * exp(ETA_CL + KAPPA_CL)
  }
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
        // ODE 1-cpt with the switch in the individual parameters (so the
        // uses_time_builtin flag is set, not merely the ODE-RHS clock).
        const ODE_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(5.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        const ODE_NO_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin;
        let ode_supported = crate::sens::ode_provider::ode_analytical_supported;

        let ana_t = parse_model_string(ANALYTICAL_TIME).expect("parses analytical TIME");
        let ana_n = parse_model_string(ANALYTICAL_NO_TIME).expect("parses analytical control");
        assert!(
            uses_time(&ana_t),
            "TIME switch sets the uses_time_builtin flag"
        );
        assert!(!uses_time(&ana_n), "control model must not set the flag");
        // Closed-form non-IOV now serves TIME via the per-event walk (#486): the
        // model is admitted and routes through `tvcov_analytical_supported`.
        assert!(
            analytical_supported(&ana_t),
            "closed-form non-IOV now admits TIME (routed to the per-event walk)"
        );
        assert!(
            tvcov_analytical_supported(&ana_t),
            "TIME routes through the TV-cov event-driven walk"
        );
        assert!(
            analytical_supported(&ana_n),
            "non-TIME twin stays analytic (guard is specific)"
        );

        let iov_t = parse_model_string(IOV_TIME).expect("parses IOV TIME");
        let iov_n = parse_model_string(WARFARIN_IOV).expect("parses IOV control");
        assert!(
            iov_t.n_kappa > 0 && iov_n.n_kappa > 0,
            "both IOV models carry a kappa"
        );
        // Closed-form IOV now serves TIME (per-event stacked seeding in
        // `build_iov_sources`, #486).
        assert!(
            iov_analytical_supported(&iov_t),
            "closed-form IOV now serves TIME via the per-event walk"
        );
        assert!(
            iov_analytical_supported(&iov_n),
            "non-TIME IOV twin stays analytic"
        );

        let ode_t = parse_model_string(ODE_TIME).expect("parses ODE TIME");
        let ode_n = parse_model_string(ODE_NO_TIME).expect("parses ODE control");
        assert!(
            uses_time(&ode_t),
            "ODE indiv-param TIME switch sets the flag"
        );
        // Non-IOV ODE now serves TIME via the event-driven TV-cov walk (#486).
        assert!(
            ode_supported(&ode_t),
            "non-IOV ODE now serves indiv-param TIME via the per-event walk"
        );
        assert!(ode_supported(&ode_n), "non-TIME ODE twin stays analytic");

        // ODE **IOV** now serves TIME via the per-event stacked walk (#486). Build an
        // ODE + kappa + TIME model to pin `ode_iov_supported == true`.
        const ODE_IOV_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(5.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL + KAPPA_CL)
  } else {
    CL = TVCL * exp(ETA_CL + KAPPA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;
        let ode_iov_t = parse_model_string(ODE_IOV_TIME).expect("parses ODE IOV TIME");
        assert!(ode_iov_t.n_kappa > 0 && uses_time(&ode_iov_t));
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&ode_iov_t),
            "ODE IOV now serves TIME via the per-event stacked walk (#486)"
        );

        // #486: a TIME + `init(...)` ODE model is now analytic at the model level. TIME forces
        // the event-driven walk (`integrate_tvcov_g`), which now seeds a non-zero `init(...)`
        // state (#662) alongside the per-event TIME seeding (#637) — so the model reports
        // analytic, not FD. (The old #637 round-2 decline assumed the walk seeded compartments
        // at zero, which was true before #662.)
        const ODE_TIME_INIT: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(5.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = 1000.0 / V
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let ode_time_init = parse_model_string(ODE_TIME_INIT).expect("parses ODE TIME init");
        assert!(uses_time(&ode_time_init) && ode_time_init.ode_spec.is_some());
        assert!(
            ode_supported(&ode_time_init),
            "TIME + init(...) ODE is analytic on the event-driven walk now (#486)"
        );
        assert!(
            analytic_outer_gradient_available(&ode_time_init),
            "TIME + init(...) ODE outer route is analytic now (#486)"
        );

        // Direct `pk(...=TIME)` mapping (not an `[individual_parameters]` statement): the
        // parser now desugars the `=TIME` binding into a synthetic
        // `__ferx_pktime_<slot> = TIME` individual parameter (mirroring the #631 Form-C
        // readout desugaring), so the mapped slot enters the program's `pk_slots` and
        // rides the same per-event analytic walk as an `[individual_parameters]` TIME
        // switch — no longer FD (#486 direct-mapping follow-up). Use a 2-cpt `q=TIME`
        // mapping so the mapped slot is not a denominator (a `v=TIME` model divides by
        // `V = 0` at the `t = 0` dose — a user degeneracy, not a gate concern).
        const ANALYTICAL_TIME_DIRECT: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=TIME, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let direct = parse_model_string(ANALYTICAL_TIME_DIRECT).expect("parses direct pk=TIME");
        assert!(
            uses_time(&direct),
            "direct pk(...=TIME) mapping sets the uses_time_builtin flag"
        );
        assert!(
            analytical_supported(&direct) && tvcov_analytical_supported(&direct),
            "direct pk(...=TIME) mapping is now served by the per-event analytic walk (desugared)"
        );
        assert!(
            analytic_outer_gradient_available(&direct),
            "direct pk(...=TIME) outer gradient route is now analytic"
        );
    }

    /// The closed-form non-IOV provider's exact value/∂η/∂²η/∂θ/∂²η∂θ for a model
    /// whose structural parameter reads the `TIME` built-in must match central
    /// finite differences of the production predictor `compute_predictions_with_tv`
    /// (the independent f64 event-driven path that threads the same per-event TIME).
    /// No TV covariates are present — the subject is routed to the per-event walk
    /// purely by `uses_time_builtin` (#486 / #610). Covers a `$PK IF(TIME...)` switch
    /// (1-cpt IV, 2-cpt IV) and a continuous `TVCL + c·TIME` term (1-cpt oral).
    #[test]
    fn time_builtin_provider_matches_fd_of_production() {
        // (a) 1-cpt IV: CL switches at TIME = 45 (NONMEM `IF (TIME.GE.45) CL=...`).
        const ONECPT_IV_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(6.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        // (b) 1-cpt oral: CL varies continuously with TIME (`TVCL + 0.05·TIME`), so
        // both ∂CL/∂TVCL and the TIME-scaled ∂CL/∂ETA_CL are exercised per event.
        const ONECPT_ORAL_TIME: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVKA(1.5, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = (TVCL + 0.05 * TIME) * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        // (c) 2-cpt IV: same TIME switch on CL, widening the dual to M = 7.
        const TWOCPT_IV_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(6.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(5.0, 0.5, 50.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let iv_bolus = |t: f64| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0);
        // Observations straddle the TIME = 45 switch so both arms of the `if` drive
        // at least one prediction (the switch is data-driven, so it is fixed under
        // the η/θ perturbation — the prediction is smooth in η/θ within each arm).
        let straddle = [10.0, 30.0, 50.0, 70.0, 90.0];

        let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
            {
                let m = parse_model_string(ONECPT_IV_TIME).expect("parse 1cpt iv TIME");
                let s = subject_with_doses_and_resets(vec![iv_bolus(0.0)], &straddle, Vec::new());
                (m, s, vec![10.0, 6.0, 50.0], vec![0.15, -0.10])
            },
            {
                let m = parse_model_string(ONECPT_ORAL_TIME).expect("parse 1cpt oral TIME");
                let s = subject_with_doses_and_resets(
                    vec![iv_bolus(0.0)],
                    &[1.0, 4.0, 10.0, 24.0, 48.0],
                    Vec::new(),
                );
                (m, s, vec![1.0, 10.0, 1.5], vec![0.15, -0.10, 0.20])
            },
            {
                let m = parse_model_string(TWOCPT_IV_TIME).expect("parse 2cpt iv TIME");
                let s = subject_with_doses_and_resets(vec![iv_bolus(0.0)], &straddle, Vec::new());
                (m, s, vec![10.0, 6.0, 50.0, 5.0, 100.0], vec![0.15, -0.10])
            },
        ];

        for (m, s, theta, eta) in &cases {
            assert!(
                crate::parser::model_parser::compiled_model_uses_time_builtin(m),
                "fixture must read the TIME built-in"
            );
            assert!(
                !s.has_tv_covariates(),
                "fixture must stay on the no-TV path (routed purely by uses_time_builtin)"
            );
            assert!(
                tvcov_analytical_supported(m),
                "TIME model must be provider-supported via the per-event walk"
            );
            assert!(
                subject_sensitivities(m, s, theta, eta).is_some(),
                "TIME subject must take the analytic per-event provider"
            );
            check_full_provider_vs_fd(m, s, theta, eta);
        }
    }

    /// #486: the light `Dual1` inner η-gradient must equal the full `Dual2` outer
    /// `df_deta` (η-block) for a `TIME`-built-in subject — both run the same
    /// event-driven walk, and the outer is FD-validated by
    /// [`time_builtin_provider_matches_fd_of_production`].
    #[test]
    fn time_builtin_eta_grad_matches_full() {
        const ONECPT_IV_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(6.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let m = parse_model_string(ONECPT_IV_TIME).expect("parse 1cpt iv TIME");
        let s = subject_with_doses_and_resets(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &[10.0, 30.0, 50.0, 70.0, 90.0],
            Vec::new(),
        );
        let theta = [10.0, 6.0, 50.0];
        let eta = [0.15, -0.10];
        let full = subject_sensitivities(&m, &s, &theta, &eta).expect("full provider");
        let light = subject_eta_grad(&m, &s, &theta, &eta).expect("light inner provider");
        assert_eq!(full.obs.len(), light.len());
        for (o, g) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(o.f, g.f, max_relative = 1e-12, epsilon = 1e-12);
            for (a, b) in o.df_deta.iter().zip(g.df_deta.iter()) {
                approx::assert_relative_eq!(a, b, max_relative = 1e-10, epsilon = 1e-12);
            }
        }
    }

    /// #486: a **direct `pk(...=TIME)` structural mapping** (here `q=TIME` on a 2-cpt IV,
    /// so the mapped slot is not a denominator — `v=TIME` would divide by `V = 0` at the
    /// `t = 0` dose). The parser desugars it into a synthetic `__ferx_pktime_q = TIME`
    /// individual parameter, so `Q` = event time per event (`∂Q/∂θ = ∂Q/∂η = 0`) while
    /// `CL`/`V1`'s derivatives stay exact. value + ∂η + ∂²η + ∂θ + ∂²η∂θ must match central
    /// FD of production (which threads the same per-event TIME through the desugared param).
    #[test]
    fn time_builtin_direct_pk_mapping_matches_fd_of_production() {
        const DIRECT_Q_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=TIME, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let m = parse_model_string(DIRECT_Q_TIME).expect("parse direct q=TIME");
        assert!(crate::parser::model_parser::compiled_model_uses_time_builtin(&m));
        assert!(
            tvcov_analytical_supported(&m),
            "direct pk(q=TIME) desugars to an indiv-param → analytic"
        );
        let s = subject_with_doses_and_resets(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &[1.0, 4.0, 10.0, 24.0, 48.0],
            Vec::new(),
        );
        assert!(!s.has_tv_covariates());
        check_full_provider_vs_fd(&m, &s, &[10.0, 50.0, 100.0], &[0.10, -0.05]);
    }

    /// #486: the direct `pk(q=TIME)` desugaring must be *exactly* the explicit
    /// `[individual_parameters] QT = TIME; pk(q=QT)` form — a pure rename into a synthetic
    /// parameter. This ties the direct mapping to the `[individual_parameters]` TIME path
    /// that #637 validated live against NONMEM (`METHOD=1 INTER`, ~5 sig figs), so the
    /// direct form inherits that numerical validation by transitivity. All sensitivity
    /// outputs (value + ∂η + ∂θ + 2nd-order) must agree to 1e-12.
    #[test]
    fn time_builtin_direct_pk_mapping_equivalent_to_explicit_indiv_param() {
        const DIRECT: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=TIME, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        const EXPLICIT: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  V2 = TVV2
  QT = TIME
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=QT, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let m_direct = parse_model_string(DIRECT).expect("parse direct");
        let m_explicit = parse_model_string(EXPLICIT).expect("parse explicit");
        let s = subject_with_doses_and_resets(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &[1.0, 4.0, 10.0, 24.0, 48.0],
            Vec::new(),
        );
        let theta = [10.0, 50.0, 100.0];
        let eta = [0.10, -0.05];
        let sd = subject_sensitivities(&m_direct, &s, &theta, &eta).expect("direct sens");
        let se = subject_sensitivities(&m_explicit, &s, &theta, &eta).expect("explicit sens");
        assert_eq!(sd.obs.len(), se.obs.len());
        for (a, b) in sd.obs.iter().zip(se.obs.iter()) {
            approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-12, epsilon = 1e-13);
            for (x, y) in a.df_deta.iter().zip(b.df_deta.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-12, epsilon = 1e-13);
            }
            for (x, y) in a.df_dtheta.iter().zip(b.df_dtheta.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-12, epsilon = 1e-13);
            }
            for (x, y) in a.d2f_deta2.iter().zip(b.d2f_deta2.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-12, epsilon = 1e-13);
            }
            // The mixed η-θ second derivative feeds the FOCEI Laplace `log|H̃|`
            // θ-gradient this work targets, so it must match too.
            for (x, y) in a.d2f_deta_dtheta.iter().zip(b.d2f_deta_dtheta.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-12, epsilon = 1e-13);
            }
        }
    }

    /// #486: an η-dependent `ExpressionScale` `obs_scale` on the **event-driven walk** —
    /// for a `TIME` switch AND a time-varying covariate. The walk now applies the same
    /// subject-static scale quotient (`apply_expression_scale_outer` / `_inner_dispatch`)
    /// the dose-superposition path uses, so value/∂η/∂²η/∂θ/∂²η∂θ must match FD of
    /// production and the light inner must track the outer η-block. (Closes the TV-cov +
    /// expression-scale gap, not just the TIME one.)
    #[test]
    fn expression_scale_on_event_walk_matches_fd_closed_form() {
        const IV_TIME_SCALED: &str = r#"
[parameters]
  theta TVCL(1.0, 0.05, 50.0)
  theta TVCL_LATE(0.5, 0.05, 50.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  obs_scale = 1000 / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        const IV_TVCOV_SCALED: &str = r#"
[parameters]
  theta TVCL(1.0, 0.05, 50.0)
  theta TVV(20.0, 1.0, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[covariates]
  WT continuous
[scaling]
  obs_scale = 1000 / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let check_parity = |m: &CompiledModel, s: &Subject, theta: &[f64], eta: &[f64]| {
            assert!(
                matches!(m.scaling, ScalingSpec::ExpressionScale { .. }),
                "fixture must carry an ExpressionScale obs_scale"
            );
            assert!(
                tvcov_analytical_supported(m),
                "ExpressionScale event-walk model must be provider-supported"
            );
            check_full_provider_vs_fd(m, s, theta, eta);
            // Light inner η-gradient must track the outer η-block (both apply the scale).
            let full = subject_sensitivities(m, s, theta, eta).expect("outer");
            let light = subject_eta_grad(m, s, theta, eta).expect("inner");
            for (o, g) in full.obs.iter().zip(light.iter()) {
                approx::assert_relative_eq!(o.f, g.f, max_relative = 1e-12, epsilon = 1e-12);
                for (a, b) in o.df_deta.iter().zip(g.df_deta.iter()) {
                    approx::assert_relative_eq!(a, b, max_relative = 1e-10, epsilon = 1e-12);
                }
            }
        };
        // TIME switch + obs_scale (no TV covariates → routed by uses_time_builtin).
        let m_time = parse_model_string(IV_TIME_SCALED).expect("parse IV TIME scaled");
        let s_time = subject_with_doses_and_resets(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &[10.0, 30.0, 50.0, 70.0, 90.0],
            Vec::new(),
        );
        check_parity(&m_time, &s_time, &[1.0, 0.5, 20.0], &[0.15, -0.10]);
        // Time-varying covariate + obs_scale (the broader gap this closes).
        let m_tv = parse_model_string(IV_TVCOV_SCALED).expect("parse IV tvcov scaled");
        let s_tv = tvcov_subject(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &[70.0],
            &[1.0, 2.0, 4.0, 8.0, 24.0],
            &[70.0, 72.0, 80.0, 85.0, 90.0],
            Vec::new(),
            Vec::new(),
            &[],
        );
        assert!(s_tv.has_tv_covariates());
        check_parity(&m_tv, &s_tv, &[1.0, 20.0, 0.75], &[0.15, -0.10]);
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
            Optimizer::Auto.resolve_auto(&m, true),
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

    /// **Closed-form `init(...)` on the event-driven TV-cov walk** (#486, branch H). An
    /// `init(central) = TVC0·V` baseline under a WT covariate on CL (TV-cov). Production's
    /// `compute_predictions_with_tv` layers `A₀·kernel(t, pk)` with the subject-static `t = 0`
    /// snapshot; the TV-cov provider now folds in the same impulse via `apply_tvcov_init_outer` /
    /// `apply_tvcov_init_inner`. The full provider's value / `∂η` / `∂²η` / `∂θ` / `∂²η∂θ` must
    /// match FD of the production predictor (which keeps the init), the inner `∂f/∂η` must equal
    /// the outer η-block, and a static-covariate subject of the same model still takes the exact
    /// dose-superposition init path.
    #[test]
    fn analytical_init_tvcov_subject_matches_fd() {
        let m = parse_model_string(ONECPT_IV_TVCOV_INIT).expect("parse");
        assert_eq!(m.analytical_init.len(), 1, "init parsed");
        assert!(
            tvcov_analytical_supported(&m),
            "init model is now analytic on the TV-cov walk (#486)"
        );

        let theta = vec![0.2, 10.0, 0.75, 4.0];
        let eta = vec![0.15, -0.10];

        // (a) TV-cov subject (WT changes across records): outer full provider matches FD of
        // production (which includes the init), and the inner η-gradient matches the outer.
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
            subject_sensitivities(&m, &tv, &theta, &eta).is_some(),
            "TV-cov init subject now takes the analytic provider (#486)"
        );
        check_full_provider_vs_fd(&m, &tv, &theta, &eta);
        let outer = subject_sensitivities_tvcov(&m, &tv, &theta, &eta).expect("outer tvcov init");
        let inner = subject_eta_grad_tvcov(&m, &tv, &theta, &eta).expect("inner tvcov init");
        assert_eq!(outer.obs.len(), inner.len());
        for (o, i) in outer.obs.iter().zip(inner.iter()) {
            approx::assert_relative_eq!(o.f, i.f, max_relative = 1e-12, epsilon = 1e-12);
            for k in 0..m.n_eta {
                approx::assert_relative_eq!(
                    o.df_deta[k],
                    i.df_deta[k],
                    max_relative = 1e-9,
                    epsilon = 1e-10
                );
            }
        }

        // (b) TV-cov + EVID 3/4 reset: the closed-form init impulse contributes only to
        // observations strictly before the first reset (production `add_analytical_init`), and
        // the reset itself is carried by the TV-cov walk — the combination must still match FD.
        let tv_reset = tvcov_subject(
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ],
            &[70.0, 90.0],
            &[2.0, 6.0, 13.0, 18.0],
            &[70.0, 70.0, 90.0, 90.0],
            vec![12.0],
            Vec::new(),
            &[],
        );
        check_full_provider_vs_fd(&m, &tv_reset, &theta, &eta);

        // (c) A static-covariate subject of the SAME model still takes the exact analytic init
        // path (dose superposition + layered impulse).
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

    /// 1-cpt IV closed-form model with a **modeled-duration** dose (`RATE=-2` → `D1`),
    /// `D1 = TVD1·exp(ETA_D1)`. The infusion window end `t_dose + D1` is a moving
    /// boundary in `D1`; the event-driven walk carries it via the exact dual window
    /// length (#486). Used by the analytic-vs-FD and routing tests below.
    const ONECPT_IV_MODELED_DUR: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(2.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    /// A modeled-duration `RATE=-2` subject must now route to the event-driven walk
    /// and be served analytically (the closed-form twin of the ODE #635 path), not
    /// fall back to FD (#486). Both the full outer provider and the light inner
    /// gradient must accept it.
    #[test]
    fn provider_modeled_duration_routes_to_event_walk() {
        let model = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse");
        let subject = subject_with_dose(
            DoseEvent::modeled(
                0.0,
                1000.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
            &[0.5, 1.5, 3.0, 6.0],
        );
        assert!(!subject.all_doses_fixed(), "dose must be modeled");
        assert!(
            subject_routes_to_event_walk(&model, &subject),
            "a modeled-RATE subject must route to the event-driven walk (#486)"
        );
        assert!(
            subject_sensitivities(&model, &subject, &[10.0, 50.0, 2.0], &[0.1, -0.05, 0.05])
                .is_some(),
            "modeled-duration dose must be served analytically by the walk (full provider)"
        );
        assert!(
            subject_eta_grad(&model, &subject, &[10.0, 50.0, 2.0], &[0.1, -0.05, 0.05]).is_some(),
            "modeled-duration dose must be served analytically (light inner provider)"
        );
    }

    /// The exact outer `(η, θ)` sensitivities of a modeled-duration `RATE=-2` dose must
    /// match central finite differences of the production predictor
    /// (`compute_predictions_with_tv`, which resolves `D1` per evaluation). Observations
    /// straddle the window end (`D1 ≈ 2.1`: `t = 0.5, 1.5` inside; `t = 3, 6` after), so
    /// the moving infusion-end boundary is genuinely exercised (#486).
    #[test]
    fn provider_modeled_duration_iv_matches_fd_of_production() {
        let model = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse");
        let subject = subject_with_dose(
            DoseEvent::modeled(
                0.0,
                1000.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
            &[0.5, 1.5, 3.0, 6.0],
        );
        check_full_provider_vs_fd(&model, &subject, &[10.0, 50.0, 2.0], &[0.12, -0.08, 0.05]);
    }

    /// Modeled **rate** (`RATE=-1` → `R1`), the sign-mirror: `duration = amt/R1`, so the
    /// window end still moves with the PK param. Analytic outer sensitivities vs FD of
    /// production (#486).
    #[test]
    fn provider_modeled_rate_iv_matches_fd_of_production() {
        const ONECPT_IV_MODELED_RATE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVR1(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_R1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  R1 = TVR1 * exp(ETA_R1)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let model = parse_model_string(ONECPT_IV_MODELED_RATE).expect("parse");
        // amt=1000, R1≈500 → window ≈ 2h. Obs straddle it.
        let subject = subject_with_dose(
            DoseEvent::modeled(
                0.0,
                1000.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledRate,
            ),
            &[0.5, 1.5, 3.0, 6.0],
        );
        check_full_provider_vs_fd(&model, &subject, &[10.0, 50.0, 500.0], &[0.12, -0.08, 0.05]);
    }

    /// The light inner η-gradient (`subject_eta_grad`) for a modeled-duration dose must
    /// match the `df_deta` block of the FD-validated full outer provider
    /// (`subject_sensitivities`) — the inner and outer walks must agree (#486).
    #[test]
    fn provider_modeled_duration_inner_matches_outer() {
        let model = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse");
        let subject = subject_with_dose(
            DoseEvent::modeled(
                0.0,
                1000.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
            &[0.5, 1.5, 3.0, 6.0],
        );
        let theta = [10.0, 50.0, 2.0];
        let eta = [0.12, -0.08, 0.05];
        let outer = subject_sensitivities(&model, &subject, &theta, &eta).expect("outer some");
        let inner = subject_eta_grad(&model, &subject, &theta, &eta).expect("inner some");
        assert_eq!(outer.obs.len(), inner.len());
        for (o, i) in outer.obs.iter().zip(inner.iter()) {
            approx::assert_relative_eq!(o.f, i.f, max_relative = 1e-9, epsilon = 1e-12);
            for (a, b) in o.df_deta.iter().zip(i.df_deta.iter()) {
                approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-12);
            }
        }
    }

    /// **Bit-parity cross-check vs the ODE twin.** The closed-form modeled-duration
    /// walk and the ODE `[odes]` modeled-duration walk (already analytic + NONMEM-
    /// anchored via #630/#635) are two independent implementations of the same
    /// moving-boundary sensitivity. Their outer `(η, θ)` jets must agree to ODE
    /// integration tolerance — the strongest confirmation the closed-form saltation is
    /// right (#486).
    #[test]
    fn provider_modeled_duration_matches_ode_twin() {
        const ODE_TWIN: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(2.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
        let cf = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse cf");
        let ode = parse_model_string(ODE_TWIN).expect("parse ode twin");
        let subject = subject_with_dose(
            DoseEvent::modeled(
                0.0,
                1000.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
            &[0.5, 1.5, 3.0, 6.0],
        );
        let theta = [10.0, 50.0, 2.0];
        let eta = [0.12, -0.08, 0.05];
        let s_cf = subject_sensitivities(&cf, &subject, &theta, &eta).expect("cf some");
        let s_ode = subject_sensitivities(&ode, &subject, &theta, &eta).expect("ode some");
        assert_eq!(s_cf.obs.len(), s_ode.obs.len());
        for (a, b) in s_cf.obs.iter().zip(s_ode.obs.iter()) {
            approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-6, epsilon = 1e-8);
            for (x, y) in a.df_deta.iter().zip(b.df_deta.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-7);
            }
            for (x, y) in a.df_dtheta.iter().zip(b.df_dtheta.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-7);
            }
            for (x, y) in a.d2f_deta2.iter().zip(b.d2f_deta2.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-4, epsilon = 1e-6);
            }
        }
    }

    /// **Closed-form modeled-duration dose × steady state** (#486 — the last modeled-dose
    /// gap after #652). The SS trough is equilibrated by `equilibrate_ss_g`, which now threads
    /// the modeled window dual `(rate_bare, dur_bare)` into each cycle's active/quiet split, so
    /// the moving infusion-end jet (`∂D1`) flows through the SS trough exactly as it does
    /// through the current pulse. `D1 ≈ 2.1 < II = 12`; obs straddle the window end within the
    /// SS interval (0.5, 1.5 inside; 3, 6, 11 after). Validated vs central FD of the production
    /// predictor (`compute_predictions_with_tv`, which resolves `D1` per evaluation and
    /// equilibrates the same SS train).
    #[test]
    fn provider_modeled_duration_ss_matches_fd_of_production() {
        let model = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse");
        let subject = subject_with_dose(
            DoseEvent::modeled(
                0.0,
                1000.0,
                1,
                true,
                12.0,
                crate::types::RateMode::ModeledDuration,
            ),
            &[0.5, 1.5, 3.0, 6.0, 11.0],
        );
        assert!(subject.has_periodic_ss_dose());
        assert!(
            subject_sensitivities(&model, &subject, &[10.0, 50.0, 2.0], &[0.12, -0.08, 0.05])
                .is_some(),
            "modeled-duration + SS must be served analytically now (#486)"
        );
        check_full_provider_vs_fd(&model, &subject, &[10.0, 50.0, 2.0], &[0.12, -0.08, 0.05]);
    }

    /// Modeled **rate** (`RATE=-1`) × steady state — the sign-mirror of the duration case,
    /// same `equilibrate_ss_g` window-jet path (`duration = amt/R1` moves the SS window end).
    #[test]
    fn provider_modeled_rate_ss_matches_fd_of_production() {
        const ONECPT_IV_MODELED_RATE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVR1(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_R1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  R1 = TVR1 * exp(ETA_R1)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let model = parse_model_string(ONECPT_IV_MODELED_RATE).expect("parse");
        // amt=1000, R1≈500 → window ≈ 2h < II = 12. Obs straddle the SS window end.
        let subject = subject_with_dose(
            DoseEvent::modeled(
                0.0,
                1000.0,
                1,
                true,
                12.0,
                crate::types::RateMode::ModeledRate,
            ),
            &[0.5, 1.5, 3.0, 6.0, 11.0],
        );
        check_full_provider_vs_fd(&model, &subject, &[10.0, 50.0, 500.0], &[0.12, -0.08, 0.05]);
    }

    /// **Bit-parity cross-check vs the ODE twin for SS + modeled-duration.** The ODE
    /// `[odes]` SS + modeled-duration path is already analytic (#642) and independently
    /// NONMEM-anchored, so its outer `(η, θ)` jets are a strong oracle for the closed-form
    /// SS window-jet just added — they must agree to ODE integration tolerance (#486).
    #[test]
    fn provider_modeled_duration_ss_matches_ode_twin() {
        const ODE_TWIN: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(2.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
        let cf = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse cf");
        let ode = parse_model_string(ODE_TWIN).expect("parse ode twin");
        let subject = subject_with_dose(
            DoseEvent::modeled(
                0.0,
                1000.0,
                1,
                true,
                12.0,
                crate::types::RateMode::ModeledDuration,
            ),
            &[0.5, 1.5, 3.0, 6.0, 11.0],
        );
        let theta = [10.0, 50.0, 2.0];
        let eta = [0.12, -0.08, 0.05];
        let s_cf = subject_sensitivities(&cf, &subject, &theta, &eta).expect("cf some");
        let s_ode = subject_sensitivities(&ode, &subject, &theta, &eta).expect("ode some");
        assert_eq!(s_cf.obs.len(), s_ode.obs.len());
        for (a, b) in s_cf.obs.iter().zip(s_ode.obs.iter()) {
            approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-6, epsilon = 1e-8);
            for (x, y) in a.df_deta.iter().zip(b.df_deta.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-7);
            }
            for (x, y) in a.df_dtheta.iter().zip(b.df_dtheta.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-7);
            }
        }
    }

    /// **Closed-form modeled-duration × SS under IOV — κ-coupled window** (#486 review #1).
    /// The relaxed `modeled_dose_analytic_gate` is shared by the closed-form **IOV** entry
    /// (`build_iov_sources`), so a modeled SS dose is now analytic there too. Here `D1` is
    /// κ-coupled (`D1 = TVD1·exp(ETA_D1 + KAPPA_D1)`), so each occasion's SS window length
    /// differs — this pins that the per-occasion modeled-window jet threaded into
    /// `equilibrate_ss_g` lands in the correct stacked-κ axis. FD of `predict_iov` is the
    /// independent oracle.
    #[test]
    fn provider_modeled_duration_ss_iov_kappa_coupled_matches_fd_of_predict_iov() {
        const CF_IOV_MODELED_SS: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_D1 ~ 0.04
  kappa KAPPA_D1 ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1 + KAPPA_D1)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;
        let model = parse_model_string(CF_IOV_MODELED_SS).expect("parse cf IOV modeled SS");
        let mut subject = iov_subject();
        subject.doses = vec![
            DoseEvent::modeled(
                0.0,
                1000.0,
                1,
                true,
                12.0,
                crate::types::RateMode::ModeledDuration,
            ),
            DoseEvent::modeled(
                24.0,
                1000.0,
                1,
                true,
                12.0,
                crate::types::RateMode::ModeledDuration,
            ),
        ];
        let theta = [0.2, 10.0, 5.0];
        // stacked = [η_cl, η_v, η_d1, κ_g0, κ_g1]; κ on D1 → each occasion's SS window differs.
        let stacked = [0.12, -0.08, 0.05, 0.06, -0.11];
        assert!(
            subject_sensitivities_iov(&model, &subject, &theta, &stacked).is_some(),
            "closed-form IOV modeled-duration + SS must be analytic now (#486)"
        );
        check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
    }

    /// **Closed-form modeled-duration × SS under time-varying covariates** (#486 review #1).
    /// The relaxed gate is also shared by the non-IOV TV-cov entry
    /// (`subject_sensitivities_tvcov`); a covariate (`WT` on `CL`) varies across the records
    /// of a single SS modeled dose. Validated vs central FD of `compute_predictions_with_tv`
    /// (which resolves `D1` and equilibrates the SS train per evaluation).
    #[test]
    fn provider_modeled_duration_ss_tvcov_matches_fd_of_production() {
        const CF_TVCOV_MODELED_SS: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(2.0, 0.1, 24.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let model = parse_model_string(CF_TVCOV_MODELED_SS).expect("parse cf tvcov modeled SS");
        let mut subject = subject_with_dose(
            DoseEvent::modeled(
                0.0,
                1000.0,
                1,
                true,
                12.0,
                crate::types::RateMode::ModeledDuration,
            ),
            &[1.0, 4.0, 6.0, 8.0, 11.0],
        );
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        subject.dose_covariates = vec![wt(60.0)];
        subject.obs_covariates = vec![wt(65.0), wt(70.0), wt(75.0), wt(80.0), wt(85.0)];
        assert!(subject.has_tv_covariates());
        assert!(
            subject_sensitivities(
                &model,
                &subject,
                &[10.0, 50.0, 2.0, 0.75],
                &[0.12, -0.08, 0.05]
            )
            .is_some(),
            "closed-form TV-cov modeled-duration + SS must be analytic now (#486)"
        );
        check_full_provider_vs_fd(
            &model,
            &subject,
            &[10.0, 50.0, 2.0, 0.75],
            &[0.12, -0.08, 0.05],
        );
    }

    /// **Reset regression (#486 review #3).** A modeled-duration dose that starts
    /// *before* an EVID3/4 reset must not thread its window jet into the post-reset
    /// sub-intervals: the walk's rate loop drops a pre-reset dose (`t_start <
    /// reset_floor`), and `dual_pos` must drop it too — otherwise the post-reset window
    /// length carries a spurious `∂D`. The first infusion (`t = 0`, `D1 ≈ 2.1`) ends
    /// *after* the reset at `t = 1`, so its stale infusion-end break at `≈ 2.1` lands in
    /// a post-reset interval that the obs at `t = 2.5` straddles; a fresh modeled dose at
    /// the reset supplies the legitimate post-reset moving boundary. Validated against FD
    /// of the production predictor (which cancels the pre-reset infusion at the reset).
    #[test]
    fn provider_modeled_duration_reset_matches_fd_of_production() {
        let model = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse");
        let doses = vec![
            DoseEvent::modeled(
                0.0,
                1000.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
            DoseEvent::modeled(
                1.0,
                800.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
        ];
        let subject = subject_with_doses_and_resets(doses, &[0.5, 1.5, 2.5, 4.0], vec![1.0]);
        assert!(subject.has_resets(), "subject must carry a reset");
        assert!(!subject.all_doses_fixed(), "doses must be modeled");
        check_full_provider_vs_fd(&model, &subject, &[10.0, 50.0, 2.0], &[0.12, -0.08, 0.05]);
    }

    /// **Separability guard (#486 review #2).** Two modeled-duration doses into
    /// *distinct* compartments (`D1` into cmt 1, `D2` into cmt 2) whose infusion ends
    /// coincide can't be represented by the single-`dt` walk — each end wants a different
    /// per-compartment dual window length — so the subject declines to FD
    /// (`subject_sensitivities` → `None`). Moving `D2` so the ends separate restores the
    /// analytic path. Same-slot coincidences (identical jet) are unaffected.
    #[test]
    fn provider_modeled_distinct_slot_coincident_ends_decline() {
        const TWOCPT_IV_D1_D2: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(5.0, 0.5, 50.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta TVD1(2.0, 0.1, 24.0)
  theta TVD2(2.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  D1 = TVD1
  D2 = TVD2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let model = parse_model_string(TWOCPT_IV_D1_D2).expect("parse");
        let doses = vec![
            DoseEvent::modeled(
                0.0,
                1000.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
            DoseEvent::modeled(
                0.0,
                800.0,
                2,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
        ];
        // TVD1 == TVD2 == 2.0, both doses at t=0 → both infusions end at t=2 with
        // distinct backing slots ⇒ not separable ⇒ decline to FD.
        let coincident = subject_with_doses_and_resets(doses.clone(), &[0.5, 1.5, 3.0], Vec::new());
        assert!(
            subject_sensitivities(
                &model,
                &coincident,
                &[10.0, 50.0, 5.0, 100.0, 2.0, 2.0],
                &[0.1, -0.05]
            )
            .is_none(),
            "distinct-slot coincident infusion ends must decline to FD (#486 review #2)"
        );
        // TVD2 = 3.0 → ends at t=2 (cmt1) and t=3 (cmt2), separable ⇒ analytic path.
        assert!(
            subject_sensitivities(
                &model,
                &coincident,
                &[10.0, 50.0, 5.0, 100.0, 2.0, 3.0],
                &[0.1, -0.05]
            )
            .is_some(),
            "separable infusion ends must be served analytically"
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

    // ── analytic Form C readout (#650) exact sensitivities ───────────────────

    /// A nonlinear analytic Form C readout — a saturable protein-binding total
    /// concentration `y = C + BMAX·C/(KD + C)` with `C = central/V` — must be
    /// differentiated exactly by the provider: value, `∂/∂η`, `∂²/∂η²`, `∂/∂θ`,
    /// and `∂²/∂η∂θ` all match central FD of the readout-aware production
    /// predictor. The readout carries η through both `C` (via CL, V) and the
    /// `central/V` amount→conc map, and θ through BMAX/KD/CL/V. The `.expect`
    /// inside the harness also asserts the analytic path is taken (not FD).
    const ONECPT_IV_BINDING_READOUT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBMAX(3.0, 0.01, 100.0)
  theta TVKD(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  BMAX = TVBMAX
  KD   = TVKD
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = central / V + BMAX * (central / V) / (KD + central / V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    #[test]
    fn form_c_binding_readout_provider_matches_fd() {
        let m = parse_model_string(ONECPT_IV_BINDING_READOUT).expect("parse binding readout");
        // Must be in the analytic Dual2 scope (not routed to FD).
        assert!(
            analytical_supported(&m),
            "central-only dual-evaluable Form C readout must stay analytic"
        );
        let s = subject_with_dose(
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            &[0.5, 2.0, 6.0, 12.0],
        );
        let theta = [0.2, 10.0, 3.0, 2.0];
        let eta = [0.12, -0.08];
        check_full_provider_vs_fd(&m, &s, &theta, &eta);

        // Light inner η-gradient must agree with the full provider's η block.
        let full = subject_sensitivities(&m, &s, &theta, &eta).expect("supported");
        // The θ-gradient for the non-structural BMAX (θ index 2) and KD (θ index 3)
        // must be non-zero — proving they are first-class differentiable params
        // (#650 basis extension), not aliased onto CL. Before the fix they read the
        // CL slot, so ∂y/∂θ_BMAX / ∂y/∂θ_KD were identically zero.
        let max_bmax = full
            .obs
            .iter()
            .map(|o| o.df_dtheta[2].abs())
            .fold(0.0_f64, f64::max);
        let max_kd = full
            .obs
            .iter()
            .map(|o| o.df_dtheta[3].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_bmax > 1e-6,
            "∂y/∂BMAX must be non-zero (BMAX is a real readout param, not aliased to CL)"
        );
        assert!(
            max_kd > 1e-6,
            "∂y/∂KD must be non-zero (KD is a real readout param, not aliased to CL)"
        );
        let light = subject_eta_grad(&m, &s, &theta, &eta).expect("light supported");
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

    /// The fluconazole case: a readout gated on a **per-row** covariate (`FREE`)
    /// makes the subject a time-varying-covariate subject, so it routes to the
    /// event-walk provider. The readout is served analytically there too (#650):
    /// value + all η/θ first/second derivatives match FD of the (event-walk)
    /// production predictor, and BMAX/KD (non-structural, in their allocated slots)
    /// stay differentiable through the walk.
    #[test]
    fn form_c_binding_readout_tvcov_matches_fd() {
        let m = parse_model_string(ONECPT_IV_BINDING_READOUT_FREE).expect("parse");
        let mut s = subject_with_dose(
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            &[0.5, 2.0, 6.0, 12.0],
        );
        // Alternating per-row FREE flag → per-observation covariate snapshots, so
        // the subject routes to the event-walk (TV-cov) provider.
        s.obs_covariates = vec![
            HashMap::from([("FREE".to_string(), 0.0)]),
            HashMap::from([("FREE".to_string(), 1.0)]),
            HashMap::from([("FREE".to_string(), 0.0)]),
            HashMap::from([("FREE".to_string(), 1.0)]),
        ];
        assert!(s.has_tv_covariates(), "fixture must be a TV-cov subject");
        assert!(
            readout_tvcov_supported(&m),
            "central-only binding readout fits PkDual slots"
        );
        let theta = [0.2, 10.0, 3.0, 2.0];
        let eta = [0.12, -0.08];
        check_full_provider_vs_fd(&m, &s, &theta, &eta);
    }

    const ONECPT_IV_BINDING_READOUT_FREE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBMAX(3.0, 0.01, 100.0)
  theta TVKD(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  BMAX = TVBMAX
  KD   = TVKD
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = if (FREE == 0) central / V + BMAX * (central / V) / (KD + central / V) else central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    /// An **oral** Form C readout `y = central / V` (central at state slot 1,
    /// depot slot 0 unreferenced) must also differentiate exactly.
    #[test]
    fn form_c_oral_readout_provider_matches_fd() {
        let src = WARFARIN.replace(
            "[error_model]",
            "[scaling]\n  y = central / V\n[error_model]",
        );
        let m = parse_model_string(&src).expect("parse oral readout");
        assert!(
            analytical_supported(&m),
            "oral central-only readout stays analytic"
        );
        let s = oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        check_full_provider_vs_fd(&m, &s, &[0.2, 10.0, 1.5], &[0.15, -0.10, 0.25]);
    }

    /// A readout that references the oral **depot** amount is out of the static
    /// jet's scope (the depot amount isn't reconstructed as a dual), so the model
    /// routes to the FD gradient — `analytical_supported` reports that honestly.
    #[test]
    fn form_c_depot_readout_routes_to_fd() {
        let src = WARFARIN.replace(
            "[error_model]",
            "[scaling]\n  y = central / V + depot / V\n[error_model]",
        );
        let m = parse_model_string(&src).expect("parse depot readout");
        assert!(
            !analytical_supported(&m),
            "a depot-referencing analytic readout must fall back to FD (no dual depot amount)"
        );
        // The FD path still predicts correctly (readout applied in the f64 predictor).
        let s = oral_subject(&[1.0, 4.0]);
        let preds = compute_predictions_with_tv(&m, &s, &[0.2, 10.0, 1.5], &[0.0, 0.0, 0.0]);
        assert!(preds.iter().all(|p| p.is_finite()));
    }

    /// #650 review: a readout whose **non-structural** parameter depends on a
    /// time-varying covariate (`BMAX = TVBMAX·WT/70`, WT per row) must be read
    /// **per observation** by the f64 predictor, matching the per-observation
    /// snapshot the event-walk provider differentiates. Before the fix the
    /// predictor froze `BMAX` at the t=0 covariate while the analytic gradient used
    /// the per-row value, so `check_full_provider_vs_fd` (analytic vs FD-of-predictor)
    /// diverged. The `.expect` inside the harness also asserts the analytic path is
    /// taken (BMAX→slot 2, KD→slot 3, both ≤ V3, so `readout_tvcov_supported`).
    #[test]
    fn form_c_tvcov_readout_param_matches_fd() {
        const SRC: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBMAX(3.0, 0.01, 100.0)
  theta TVKD(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  BMAX = TVBMAX * (WT / 70)
  KD   = TVKD
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = central / V + BMAX * (central / V) / (KD + central / V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let m = parse_model_string(SRC).expect("parse tv-cov readout param");
        let mut s = subject_with_dose(
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            &[0.5, 2.0, 6.0, 12.0],
        );
        // Per-row body weight → time-varying covariate → event-walk provider, and a
        // readout parameter (BMAX) that actually moves between rows.
        s.obs_covariates = vec![
            HashMap::from([("WT".to_string(), 60.0)]),
            HashMap::from([("WT".to_string(), 80.0)]),
            HashMap::from([("WT".to_string(), 70.0)]),
            HashMap::from([("WT".to_string(), 95.0)]),
        ];
        assert!(s.has_tv_covariates(), "fixture must be a TV-cov subject");
        assert!(readout_tvcov_supported(&m), "BMAX/KD fit the PkDual slots");
        check_full_provider_vs_fd(&m, &s, &[0.2, 10.0, 3.0, 2.0], &[0.12, -0.08]);
    }

    /// #650 review: a readout that references the oral **depot** amount is rejected
    /// on a subject carrying an EVID=3/4 reset (superposition can't restart the
    /// depot across the reset), rather than silently reading a zero depot. A
    /// `central`-only readout on the same reset subject is accepted (the central
    /// concentration is already reset-correct).
    #[test]
    fn depot_readout_with_reset_is_rejected() {
        let depot_src = WARFARIN.replace(
            "[error_model]",
            "[scaling]\n  y = central / V + depot / V\n[error_model]",
        );
        let depot_m = parse_model_string(&depot_src).expect("parse depot readout");
        assert!(
            depot_m
                .analytic_readout
                .as_ref()
                .expect("analytic readout")
                .references_depot(),
            "readout must be detected as depot-referencing"
        );
        let reset_subj = subject_with_doses_and_resets(
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
            ],
            &[1.0, 4.0, 8.0],
            vec![5.0],
        );
        let pop = crate::types::Population {
            subjects: vec![reset_subj],
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        assert!(
            crate::api::check_analytic_readout_support(&depot_m, &pop).is_some(),
            "depot readout + reset subject must be rejected"
        );

        // Central-only readout is fine on the very same reset subject.
        let central_src = WARFARIN.replace(
            "[error_model]",
            "[scaling]\n  y = central / V\n[error_model]",
        );
        let central_m = parse_model_string(&central_src).expect("parse central readout");
        assert!(
            crate::api::check_analytic_readout_support(&central_m, &pop).is_none(),
            "central-only readout must be accepted on a reset subject"
        );
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

    /// M3 BLOQ + IOV scope (#580): a closed-form IOV model with M3 BLOQ is analytic
    /// (the censored coefficients ride the stacked `[η_bsv, κ]` layout). The triple
    /// **M3 + IOV + `iiv_on_ruv`** is analytic too as of #591 — the closed-form assembly
    /// already carried the censored residual-eta cross coefficients `(C·z, C·m)`, so
    /// `iov_analytical_supported` admits it (and `analytic_outer_gradient_available`
    /// follows). The ODE IOV triple is analytic too (#486); only the *non-IOV ODE* triple
    /// stays FD (via `iiv_on_ruv_forces_fd`). Plain IOV and IOV + `iiv_on_ruv` (no M3) stay
    /// analytic.
    #[test]
    fn iov_analytical_supported_admits_m3_but_not_the_ruv_triple() {
        let mut model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
        // Plain IOV: analytic.
        assert!(iov_analytical_supported(&model));
        // M3 + IOV (no iiv_on_ruv): analytic as of #580.
        model.bloq_method = crate::types::BloqMethod::M3;
        assert!(iov_analytical_supported(&model));
        assert!(analytic_outer_gradient_available(&model));
        // M3 + IOV + iiv_on_ruv (the triple): analytic as of #591 (the closed-form
        // assembly carried the censored residual-eta cross coefficients all along).
        model.residual_error_eta = Some(0);
        assert!(iov_analytical_supported(&model));
        assert!(analytic_outer_gradient_available(&model));
        // Only the *non-IOV ODE* triple stays FD, gated by `iiv_on_ruv_forces_fd`
        // (ode_spec-AND-n_kappa==0); the closed-form triple here does not trip it.
        assert!(!model.iiv_on_ruv_forces_fd());
        // IOV + iiv_on_ruv without M3: analytic (#4b).
        model.bloq_method = crate::types::BloqMethod::Drop;
        assert!(iov_analytical_supported(&model));
    }

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

    // 1-cpt IV closed-form IOV with an `init(central) = TVC0·V` baseline (#486, branch
    // G-closed-form). `CL = TVCL·exp(ETA_CL + KAPPA_CL)` carries the occasion κ, so the init
    // *decay* kernel depends on κ (via each observation's occasion clearance) while the init
    // *amount* `A₀ = TVC0·V` is BSV-only (no κ) — exactly production's split in `predict_iov`
    // (`add_analytical_init_with(amount_pk = BSV, Some(obs_params))`).
    const WARFARIN_IOV_INIT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVC0(5.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[initial_conditions]
  init(central) = TVC0 * V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    /// **`init(...)` × IOV on the closed-form walk** (#486, branch G-closed-form). The IOV
    /// provider now layers the `A₀·kernel(t, pk)` init impulse per occasion: the amount from the
    /// BSV snapshot (κ = 0), the decay from each observation's occasion clearance (κ carried).
    /// The full provider's value / `∂(stacked-η)` / Hessian / `∂θ` must match FD of `predict_iov`
    /// (which includes the init), and the inner η-gradient must equal the outer η-block.
    #[test]
    fn iov_init_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_INIT).expect("parse warfarin IOV+init");
        assert_eq!(model.n_kappa, 1);
        assert_eq!(model.analytical_init.len(), 1, "init parsed");
        assert!(model.ode_spec.is_none(), "closed-form model");
        assert!(
            iov_analytical_supported(&model),
            "closed-form IOV + init must be analytic (#486)"
        );
        let subject = iov_subject();
        // θ = [TVCL, TVV, TVC0]; stacked = [η_cl, η_v, κ_g0, κ_g1].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 5.0],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    #[test]
    fn iov_init_inner_eta_grad_matches_outer() {
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV_INIT).expect("parse warfarin IOV+init"),
            &iov_subject(),
            &[0.2, 10.0, 5.0],
            &[0.12, -0.08, 0.05, -0.10],
        );
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

    /// Closed-form **IOV + modeled-duration** dose (`RATE=-2` → `D1`, #486). The
    /// per-occasion `D1` sets each infusion window; the walk resolves it from the
    /// occasion's stacked PK jet and the moving infusion-end boundary carries `∂/∂D1`
    /// over θ, η, and κ. Doses at 0 and 24 (`D1 ≈ 5`), obs straddling each window end
    /// (1 inside / 6 after; 25 inside / 30 after). Validated vs central FD of
    /// `predict_iov` (which resolves `D1` per occasion) — the closed-form twin of the
    /// ODE `ode_iov_modeled_duration_provider_matches_fd_of_predict_iov` (#635).
    const WARFARIN_IOV_MODELED_DUR: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_D1 ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;

    fn iov_modeled_dur_subject() -> Subject {
        let mut s = iov_subject();
        s.doses = vec![
            DoseEvent::modeled(
                0.0,
                100.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
            DoseEvent::modeled(
                24.0,
                100.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
        ];
        s
    }

    #[test]
    fn iov_modeled_duration_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_MODELED_DUR).expect("parse modeled-dur IOV");
        assert_eq!(model.n_kappa, 1);
        assert_eq!(model.n_eta, 3);
        assert!(model.ode_spec.is_none(), "must be a closed-form model");
        let subject = iov_modeled_dur_subject();
        assert!(!subject.all_doses_fixed(), "doses must be modeled");
        // stacked = [η_cl, η_v, η_d1, κ_g0, κ_g1] (n_eta = 3, n_kappa = 1, K = 2).
        assert!(
            subject_sensitivities_iov(
                &model,
                &subject,
                &[0.2, 10.0, 5.0],
                &[0.12, -0.08, 0.05, 0.05, -0.10],
            )
            .is_some(),
            "modeled-duration closed-form IOV subject (no SS) must be served analytically (#486)"
        );
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 5.0],
            &[0.12, -0.08, 0.05, 0.05, -0.10],
        );
    }

    /// Inner (`Dual1`) IOV modeled-duration walk must match the FD-validated outer.
    #[test]
    fn iov_modeled_duration_inner_matches_outer() {
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV_MODELED_DUR).expect("parse modeled-dur IOV"),
            &iov_modeled_dur_subject(),
            &[0.2, 10.0, 5.0],
            &[0.12, -0.08, 0.05, 0.05, -0.10],
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

    // ------------------------------------------------------------------------------------
    // Closed-form IOV + `ExpressionScale` `obs_scale` (#486): the closed-form twin of the
    // ODE-IOV path (#575/#590). The scale divisor is a per-occasion-group post-walk quotient
    // on the stacked `(θ, η_bsv, κ)` axes; each group's divisor rides its κ through the PK
    // params. Validated against central FD of `predict_iov` (the independent production f64
    // path), inner-vs-outer parity, and bit-parity with the ODE-IOV twin.
    // ------------------------------------------------------------------------------------

    /// 1-cpt oral closed-form IOV (κ on CL) with an η-dependent `obs_scale = V` divisor.
    const WARFARIN_IOV_EXPRSCALE: &str = r#"
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
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    /// 2-cpt oral closed-form IOV (κ on CL) with an `obs_scale = V` divisor — exercises the
    /// per-group quotient over the wider PK-param chain.
    const WARFARIN_IOV_EXPRSCALE_2CPT: &str = r#"
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
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    /// 1-cpt oral closed-form IOV + **WT-on-CL time-varying covariate** + `obs_scale = CL`.
    /// The scale references a covariate-carrying PK param, but the divisor is still built at
    /// the subject-static covariate snapshot (`t = 0`) — matching production `predict_iov`.
    const WARFARIN_IOV_TVCOV_EXPRSCALE: &str = r#"
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
[scaling]
  obs_scale = CL
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    /// 1-cpt IV closed-form IOV (κ on CL) with `obs_scale = V` — the closed-form half of the
    /// bit-parity twin with the ODE-IOV path. Closed-form `one_cpt_iv` yields `A/V`
    /// (concentration), so `obs_scale = V` gives `A/V²`.
    const WARFARIN_IOV_IV_EXPRSCALE: &str = r#"
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
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

    /// The gate admits a closed-form IOV model carrying a differentiable `ExpressionScale`
    /// `obs_scale`, but keeps declining LTBS, `ScalarScale`, and lagtime (out of scope for
    /// the post-walk quotient) — the closed-form mirror of `ode_iov_expr_scale_supported_and_gated`.
    #[test]
    fn iov_analytical_expr_scale_supported_and_gated() {
        let model = parse_model_string(WARFARIN_IOV_EXPRSCALE).expect("parse expr-scale IOV");
        assert_eq!(model.n_kappa, 1);
        assert!(
            matches!(model.scaling, ScalingSpec::ExpressionScale { .. }),
            "fixture must use an expression obs_scale"
        );
        assert!(
            iov_analytical_supported(&model),
            "closed-form IOV + ExpressionScale obs_scale must be on the analytic path (#486)"
        );
        assert!(analytic_outer_gradient_available(&model));
        // + LTBS still routes to FD (the in-walk log is not composed with the post-walk
        // quotient on the IOV path).
        let mut ltbs = parse_model_string(WARFARIN_IOV_EXPRSCALE).expect("parse");
        ltbs.log_transform = true;
        assert!(
            !iov_analytical_supported(&ltbs),
            "ExpressionScale + LTBS under closed-form IOV must fall back to FD"
        );
        // A constant `ScalarScale` under IOV still declines (the IOV walk has no
        // constant-divisor step; that combination stays FD as before).
        let mut scalar = parse_model_string(WARFARIN_IOV).expect("parse");
        scalar.scaling = ScalingSpec::ScalarScale(2.0);
        assert!(
            !iov_analytical_supported(&scalar),
            "ScalarScale under closed-form IOV stays FD"
        );
    }

    /// Outer packed sensitivities of a closed-form IOV + `obs_scale = V` model must match
    /// central FD of `predict_iov` (value, `∂f/∂stacked`, `∂f/∂θ`, and both Hessian blocks)
    /// over `[η_bsv, κ_g0, κ_g1]` + θ — 1-cpt and 2-cpt oral.
    #[test]
    fn iov_analytical_expr_scale_outer_matches_fd() {
        check_iov_provider_vs_fd(
            &parse_model_string(WARFARIN_IOV_EXPRSCALE).expect("parse 1cpt expr-scale IOV"),
            &iov_subject(),
            &[0.2, 10.0, 1.5],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
        check_iov_provider_vs_fd(
            &parse_model_string(WARFARIN_IOV_EXPRSCALE_2CPT).expect("parse 2cpt expr-scale IOV"),
            &iov_subject(),
            &[0.2, 10.0, 0.5, 20.0, 1.5],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
    }

    /// Closed-form IOV + **TV covariate** + `obs_scale = CL`: the scale divisor stays
    /// subject-static (built at `t = 0`) even though the walk seeds each event per-covariate.
    /// Value/grad/Hessian over `[η_bsv, κ_g0, κ_g1]` + θ (incl. `THETA_WT`) vs FD of `predict_iov`.
    #[test]
    fn iov_analytical_tvcov_expr_scale_outer_matches_fd() {
        let model =
            parse_model_string(WARFARIN_IOV_TVCOV_EXPRSCALE).expect("parse TV-cov expr-scale IOV");
        let subject = iov_tvcov_subject(false);
        assert!(subject.has_tv_covariates(), "fixture must carry TV cov");
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 1.5, 0.75],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
    }

    /// The closed-form IOV **inner** η-gradient (`Dual1`) with an `obs_scale` divisor must
    /// track the FD-validated **outer** walk's first-order block — plain and TV-cov.
    #[test]
    fn iov_analytical_expr_scale_inner_eta_grad_matches_outer() {
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV_EXPRSCALE).expect("parse expr-scale IOV"),
            &iov_subject(),
            &[0.2, 10.0, 1.5],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV_TVCOV_EXPRSCALE).expect("parse TV-cov expr-scale IOV"),
            &iov_tvcov_subject(false),
            &[0.2, 10.0, 1.5, 0.75],
            &[0.12, -0.08, 0.20, 0.05, -0.10],
        );
    }

    /// **Bit-parity with the ODE-IOV twin (#575/#590).** Closed-form `one_cpt_iv` + `obs_scale
    /// = V` computes `A/V²`; the same physical model as a user `[odes]` model reaches `A/V²`
    /// two independent, already-analytic ways: an `obs_scale = V*V` divisor (the ODE half of
    /// the very quotient this PR ports) and a Form-C readout `y = central/(V*V)` (the in-walk
    /// division). Per-observation value and `∂f/∂stacked` must agree across all three paths —
    /// confirming the closed-form quotient reproduces the NONMEM-validated ODE result.
    #[test]
    fn iov_analytical_expr_scale_equals_ode_twin() {
        const ODE_OBS_SCALE: &str = r#"
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
  obs_scale = V * V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
        const ODE_FORMC: &str = r#"
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
  y = central / (V * V)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
        let cf = parse_model_string(WARFARIN_IOV_IV_EXPRSCALE).expect("parse closed-form IV");
        let ode_scale = parse_model_string(ODE_OBS_SCALE).expect("parse ODE obs_scale twin");
        let ode_formc = parse_model_string(ODE_FORMC).expect("parse ODE Form-C twin");
        assert!(iov_analytical_supported(&cf));
        assert!(crate::sens::ode_provider::ode_iov_supported(&ode_scale));
        assert!(crate::sens::ode_provider::ode_iov_supported(&ode_formc));
        let subject = iov_subject();
        let theta = [0.2, 10.0];
        let stacked = [0.12, -0.08, 0.05, -0.10];
        let a = subject_sensitivities_iov(&cf, &subject, &theta, &stacked).expect("closed-form");
        for twin in [&ode_scale, &ode_formc] {
            let b = subject_sensitivities_iov(twin, &subject, &theta, &stacked).expect("ODE twin");
            assert_eq!(a.obs.len(), b.obs.len());
            for (oa, ob) in a.obs.iter().zip(b.obs.iter()) {
                approx::assert_relative_eq!(oa.f, ob.f, max_relative = 1e-6, epsilon = 1e-9);
                for (x, y) in oa.df_deta.iter().zip(ob.df_deta.iter()) {
                    approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-8);
                }
            }
        }
    }

    /// 1-cpt IV IOV written as a user `[odes]` model (κ on CL, Form-C readout
    /// `y = central/V`). Routes through `subject_sensitivities_iov` → the ODE IOV
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

    /// #486: a `TIME`-switched CL on an **ODE IOV** model (Form-C `y = central/V`
    /// readout). With no TV covariates the per-event stacked walk is reached purely by
    /// `uses_time_builtin`; the value/∂/∂² over `[η_bsv, κ_g0, κ_g1]` + θ must match
    /// central FD of `predict_iov` (which threads the same per-event TIME through the
    /// ODE event-driven predictor).
    #[test]
    fn ode_iov_time_builtin_provider_matches_fd_of_predict_iov() {
        const WARFARIN_IOV_ODE_TIME: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVCL_LATE(0.1, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  if (TIME > 20.0) {
    CL = TVCL_LATE * exp(ETA_CL + KAPPA_CL)
  } else {
    CL = TVCL * exp(ETA_CL + KAPPA_CL)
  }
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
        let model = parse_model_string(WARFARIN_IOV_ODE_TIME).expect("parse ODE IOV TIME");
        assert_eq!(model.n_kappa, 1);
        assert!(model.ode_spec.is_some());
        assert!(!iov_subject().has_tv_covariates());
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "ODE IOV TIME (Form-C readout) must be provider-supported"
        );
        // obs [1,6,12 | 25,30,36] straddle TIME=20. stacked = [η_cl, η_v, κ_g0, κ_g1].
        check_iov_provider_vs_fd(
            &model,
            &iov_subject(),
            &[0.2, 0.1, 10.0],
            &[0.12, -0.08, 0.05, -0.10],
        );
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

    /// 1-cpt IV ODE IOV with a **modeled-duration** dose (`RATE=-2` → `D1`). `D1` is a
    /// structural individual parameter (`D1 = TVD1·exp(ETA_D1)`), so the infusion window
    /// end `t_dose + D1` is a moving boundary in `D1`; the per-occasion rate-off saltation
    /// carries its derivative on the IOV stacked axes exactly as on the non-IOV TV-cov walk
    /// (#486 / #530). κ rides on CL here (the modeled slot itself is η-only).
    const WARFARIN_IOV_ODE_MODELED_DUR: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_D1 ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
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

    /// **ODE IOV + modeled-duration dose** (#486, design A). A two-occasion IOV subject
    /// whose per-occasion modeled `D1` sets each infusion's window length. The walk resolves
    /// `D1` from the per-occasion stacked PK jet (`inf_eff` → `pk_at_dose[k][slot]`) and the
    /// moving infusion-end saltation carries `∂/∂D1` over θ, η, and κ. Validated vs central
    /// FD of `predict_iov` (which resolves `D1` per occasion). Observations straddle each
    /// window end (`D1 ≈ 5`: obs at 1 inside, 6 after the first window; 25 inside, 30 after
    /// the second) so the moving boundary is genuinely exercised.
    #[test]
    fn ode_iov_modeled_duration_provider_matches_fd_of_predict_iov() {
        let model =
            parse_model_string(WARFARIN_IOV_ODE_MODELED_DUR).expect("parse modeled-dur IOV");
        assert_eq!(model.n_kappa, 1);
        assert_eq!(model.n_eta, 3);
        assert!(model.ode_spec.is_some());
        let mut subject = iov_subject();
        subject.doses = vec![
            DoseEvent::modeled(
                0.0,
                100.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
            DoseEvent::modeled(
                24.0,
                100.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
        ];
        assert!(
            !subject.all_doses_fixed(),
            "doses must be modeled, not fixed"
        );
        assert!(
            crate::sens::ode_provider::ode_subject_sensitivities_iov(
                &model,
                &subject,
                &[0.2, 10.0, 5.0],
                &[0.12, -0.08, 0.05, 0.05, -0.10],
            )
            .is_some(),
            "modeled-duration ODE IOV subject (no SS) must be served analytically (#486)"
        );
        // stacked = [η_cl, η_v, η_d1, κ_g0, κ_g1] (n_eta = 3, n_kappa = 1, K = 2).
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 5.0],
            &[0.12, -0.08, 0.05, 0.05, -0.10],
        );
    }

    /// **ODE IOV + κ-coupled modeled-duration dose** (#486, design A — the κ-coupling guard).
    /// Here the modeled window itself varies by occasion: `D1 = TVD1·exp(ETA_D1 + KAPPA_D1)`,
    /// so each occasion's infusion has a *different* length. This pins the concern that the
    /// `∂/∂D1` moving-boundary column lands in the correct κ-group axis — `inf_eff` reads the
    /// per-occasion `seed_pk_dual2_iov` jet, and central FD of `predict_iov` (which rebuilds
    /// `D1` from each occasion's κ) is the independent oracle.
    #[test]
    fn ode_iov_modeled_duration_kappa_coupled_matches_fd_of_predict_iov() {
        const KCOUPLED: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_D1 ~ 0.04
  kappa KAPPA_D1 ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1 + KAPPA_D1)
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
        let model = parse_model_string(KCOUPLED).expect("parse kappa-coupled modeled-dur IOV");
        assert_eq!(model.n_kappa, 1);
        let mut subject = iov_subject();
        subject.doses = vec![
            DoseEvent::modeled(
                0.0,
                100.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
            DoseEvent::modeled(
                24.0,
                100.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            ),
        ];
        // stacked = [η_cl, η_v, η_d1, κ_g0, κ_g1]; κ on D1 → each occasion's window differs.
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 5.0],
            &[0.12, -0.08, 0.05, 0.06, -0.11],
        );
    }

    /// **Still-FD edge: modeled-duration dose + steady-state** (#486, design B not yet done).
    /// The dual SS equilibration reads a fixed per-cycle `t_inf` with no modeled-window jet,
    /// so a modeled + SS subject must route to FD on BOTH the outer sensitivity walk and the
    /// inner η-gradient (scope parity). Pins the `has_ss` arm of the relaxed IOV gate.
    /// **ODE IOV + modeled-duration dose × steady-state is now analytic (#486, PR3
    /// sub-case (d)).** `equilibrate_ss_state_g` threads the per-occasion `inf_eff` jet
    /// (`D1` seeded per occasion group, same as the non-SS modeled-duration IOV test) into
    /// its per-cycle active/quiet split. Validated vs central FD of `predict_iov` (both
    /// outer and inner, scope parity).
    #[test]
    fn ode_iov_modeled_duration_ss_matches_fd_of_predict_iov() {
        let model =
            parse_model_string(WARFARIN_IOV_ODE_MODELED_DUR).expect("parse modeled-dur IOV");
        let mut subject = iov_subject();
        subject.doses = vec![
            DoseEvent::modeled(
                0.0,
                100.0,
                1,
                true,
                12.0,
                crate::types::RateMode::ModeledDuration,
            ),
            DoseEvent::modeled(
                24.0,
                100.0,
                1,
                true,
                12.0,
                crate::types::RateMode::ModeledDuration,
            ),
        ];
        let theta = [0.2, 10.0, 5.0];
        let stacked = [0.12, -0.08, 0.05, 0.05, -0.10];
        assert!(
            crate::sens::ode_provider::ode_subject_sensitivities_iov(
                &model, &subject, &theta, &stacked
            )
            .is_some(),
            "modeled-duration + SS must be analytic now (outer, #486)"
        );
        assert!(
            crate::sens::ode_provider::ode_subject_eta_grad_iov(&model, &subject, &theta, &stacked)
                .is_some(),
            "modeled-duration + SS must be analytic now (inner, scope parity)"
        );
        // stacked = [η_cl, η_v, η_d1, κ_g0, κ_g1] (n_eta = 3, n_kappa = 1, K = 2).
        check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
    }

    const ZERO_ORDER_IOV_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVDUR(5.0, 0.1, 24.0)
  omega ETA_CL  ~ 0.09
  omega ETA_V   ~ 0.04
  omega ETA_DUR ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL + KAPPA_CL)
  V   = TVV  * exp(ETA_V)
  DUR = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - (CL/V)*central
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

    /// **ODE IOV + `zero_order(dur)` absorption** (#486). The last zero-order gap after
    /// #653 ported the moving-boundary window to the non-IOV event-driven walk: the IOV
    /// walk is the *same* `integrate_tvcov_g`, so once `ode_iov_supported` admits a
    /// `ZeroOrder` forcing the κ-coupled window rides through — its rate `F·amt/dur` and
    /// window end `t_dose + dur` are rebuilt from each dose's own per-occasion stacked PK
    /// jet (`pk_at_dose[k]`, seeded by `seed_pk_dual2_iov`). Validated vs central FD of
    /// `predict_iov`. `DUR ≈ 5`, so obs at 1 (inside) / 6 (after) straddle the first window
    /// end and 25 / 30 the second, exercising the rate-off saltation per occasion.
    #[test]
    fn ode_iov_zero_order_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(ZERO_ORDER_IOV_ODE).expect("parse zero_order IOV");
        assert_eq!(model.n_kappa, 1);
        assert_eq!(model.n_eta, 3);
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "zero_order model must be admitted under IOV (#486)"
        );
        let subject = iov_subject();
        let theta = [0.2, 10.0, 5.0];
        // stacked = [η_cl, η_v, η_dur, κ_g0, κ_g1] (n_eta = 3, n_kappa = 1, K = 2).
        let stacked = [0.12, -0.08, 0.05, 0.05, -0.10];
        assert!(
            crate::sens::ode_provider::ode_subject_sensitivities_iov(
                &model, &subject, &theta, &stacked
            )
            .is_some(),
            "zero_order ODE IOV subject (no SS) must be served analytically (#486)"
        );
        check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
    }

    /// **ODE IOV + κ-coupled `zero_order(dur)`** (#486 — the κ-axis-placement guard). Here
    /// the window itself varies by occasion (`DUR = TVDUR·exp(ETA_DUR + KAPPA_DUR)`), so each
    /// occasion's zero-order window has a different length. This pins that the `∂/∂DUR`
    /// rate-off column lands in the correct κ-group axis of the stacked vector (central FD of
    /// `predict_iov`, which rebuilds `DUR` from each occasion's κ, is the independent oracle).
    #[test]
    fn ode_iov_zero_order_kappa_coupled_matches_fd_of_predict_iov() {
        const KCOUPLED: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVDUR(5.0, 0.1, 24.0)
  omega ETA_CL  ~ 0.09
  omega ETA_V   ~ 0.04
  omega ETA_DUR ~ 0.04
  kappa KAPPA_DUR ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  DUR = TVDUR * exp(ETA_DUR + KAPPA_DUR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - (CL/V)*central
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
        let model = parse_model_string(KCOUPLED).expect("parse κ-coupled zero_order IOV");
        let subject = iov_subject();
        // stacked = [η_cl, η_v, η_dur, κ_g0, κ_g1]; κ on DUR → each occasion's window differs.
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 5.0],
            &[0.12, -0.08, 0.05, 0.06, -0.11],
        );
    }

    const MIXED_IOV_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVDUR(5.0, 0.1, 24.0)
  omega ETA_CL  ~ 0.09
  omega ETA_DUR ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL + KAPPA_CL)
  V    = TVV
  FZO  = TVFZO
  FZO1 = 1 - TVFZO
  KA   = TVKA
  DUR  = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - (CL/V)*central
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

    /// **ODE IOV + `mixed` (first-order + zero-order) absorption** (#486). Exercises both
    /// admitted forcing kinds together: the pointwise first-order `R_in` and the zero-order
    /// window are delivered concurrently per occasion, and the `frac` (FZO / FZO1) multiplier
    /// rides each rate. Validated vs central FD of `predict_iov`.
    #[test]
    fn ode_iov_mixed_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(MIXED_IOV_ODE).expect("parse mixed IOV");
        assert_eq!(model.n_kappa, 1);
        assert_eq!(model.n_eta, 2);
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "mixed (first_order + zero_order) must be admitted under IOV (#486)"
        );
        let subject = iov_subject();
        let theta = [0.2, 10.0, 0.4, 1.0, 5.0];
        // stacked = [η_cl, η_dur, κ_g0, κ_g1] (n_eta = 2, n_kappa = 1, K = 2).
        let stacked = [0.12, 0.05, 0.06, -0.11];
        check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
    }

    /// **Still-FD edge: built-in absorption + steady state under IOV** (#486). The dual SS
    /// equilibration (`equilibrate_ss_state_g`) seeds a bolus/infusion trough and does not
    /// spread a periodic zero-order window (or a first-order `R_in` tail) over the cycle, so
    /// a `zero_order` + SS subject routes to FD on BOTH loops. Also pins that a non-admitted
    /// forcing kind (`weibull`) under IOV stays FD at the model gate.
    #[test]
    fn ode_iov_zero_order_ss_falls_back_to_fd() {
        let model = parse_model_string(ZERO_ORDER_IOV_ODE).expect("parse zero_order IOV");
        let mut subject = iov_subject();
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, true, 12.0),
        ];
        let theta = [0.2, 10.0, 5.0];
        let stacked = [0.12, -0.08, 0.05, 0.05, -0.10];
        assert!(
            crate::sens::ode_provider::ode_subject_sensitivities_iov(
                &model, &subject, &theta, &stacked
            )
            .is_none(),
            "zero_order + SS must route to FD under IOV (outer, #486)"
        );
        assert!(
            crate::sens::ode_provider::ode_subject_eta_grad_iov(&model, &subject, &theta, &stacked)
                .is_none(),
            "zero_order + SS must route to FD under IOV (inner, scope parity)"
        );
        // A non-admitted input-rate kind (weibull) stays FD at the model gate.
        const WEIBULL_IOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVTD(2.0, 0.05, 24.0)
  theta TVBETA(1.5, 0.1, 10.0)
  omega ETA_CL   ~ 0.09
  omega ETA_BETA ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL + KAPPA_CL)
  V    = TVV
  TD   = TVTD
  BETA = TVBETA * exp(ETA_BETA)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;
        let weibull = parse_model_string(WEIBULL_IOV).expect("parse weibull IOV");
        assert!(
            !crate::sens::ode_provider::ode_iov_supported(&weibull),
            "weibull forcing under IOV stays FD (#486 scope)"
        );
    }

    /// **ODE IOV + `parallel` dual-pathway absorption** (#486 review — the multi-forcing
    /// case). Two concurrent `first_order` forcings (`FR1·first_order(ka1) +
    /// FR2·first_order(ka2)`) feed the central compartment. The gate admits it (both kinds
    /// are `FirstOrder`); this pins that the per-occasion accumulation of two `R_in`
    /// contributions stays analytic and matches central FD of `predict_iov`.
    #[test]
    fn ode_iov_parallel_provider_matches_fd_of_predict_iov() {
        const PARALLEL_IOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA1(1.5, 0.05, 24.0)
  theta TVKA2(0.3, 0.01, 24.0)
  theta TVFR1(0.6, 0.05, 0.95)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL + KAPPA_CL)
  V   = TVV  * exp(ETA_V)
  KA1 = TVKA1
  KA2 = TVKA2
  FR1 = TVFR1
  FR2 = 1 - TVFR1
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - (CL/V)*central
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
        let model = parse_model_string(PARALLEL_IOV).expect("parse parallel IOV");
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "parallel (two first_order) must be admitted under IOV (#486)"
        );
        let subject = iov_subject();
        // stacked = [η_cl, η_v, κ_g0, κ_g1] (n_eta = 2, n_kappa = 1, K = 2).
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 1.5, 0.3, 0.6],
            &[0.12, -0.08, 0.06, -0.11],
        );
    }

    /// **ODE IOV + `first_order` absorption × estimated (bare) lagtime** (#486 review). The
    /// non-IOV path admits `first_order` + lagtime (#643 onset saltation `Δr = R_in(0⁺)`);
    /// the IOV walk is the same, so it must too. Pins that the per-occasion rate-on onset
    /// saltation at the lagged arrival stays analytic under κ (FD of `predict_iov` is the
    /// oracle). A **bare** `LAGTIME` is used — a compartment-indexed `ALAGn` is a distinct
    /// per-dose shift the single-slot IOV walk cannot represent and is declined upstream.
    #[test]
    fn ode_iov_first_order_lagtime_matches_fd_of_predict_iov() {
        const FO_LAG_IOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL      = TVCL * exp(ETA_CL + KAPPA_CL)
  V       = TVV  * exp(ETA_V)
  KA      = TVKA
  LAGTIME = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - (CL/V)*central
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
        let model = parse_model_string(FO_LAG_IOV).expect("parse first_order+lag IOV");
        assert!(model.has_lagtime());
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "first_order + lagtime must be admitted under IOV (#486)"
        );
        let subject = iov_subject();
        // stacked = [η_cl, η_v, κ_g0, κ_g1].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 1.0, 0.5],
            &[0.12, -0.08, 0.06, -0.11],
        );
    }

    /// **ODE IOV + `zero_order` absorption × EVID 3/4 reset** (#486 review). Resets are
    /// admitted by `ode_iov_subject_supported` and the zero-order window is kind-agnostic to
    /// the `reset_floor` cutoff (#653), so the combination stays analytic under IOV. A reset
    /// at `t = 15` (after the first window, before the second dose) zeros the state; central
    /// FD of `predict_iov` (which cancels the pre-reset trajectory) is the oracle.
    #[test]
    fn ode_iov_zero_order_reset_matches_fd_of_predict_iov() {
        let model = parse_model_string(ZERO_ORDER_IOV_ODE).expect("parse zero_order IOV");
        let mut subject = iov_subject();
        subject.reset_times = vec![15.0];
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 5.0],
            &[0.12, -0.08, 0.05, 0.05, -0.10],
        );
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

    /// ODE M3 BLOQ + IOV + `iiv_on_ruv` scope (#486): the ODE-path counterpart of
    /// [`iov_analytical_supported_admits_m3_but_not_the_ruv_triple`]. After the gate flips,
    /// `ode_iov_supported` admits M3, `iiv_on_ruv`, and the full **triple** M3 + IOV +
    /// `iiv_on_ruv` — all provider-agnostic over the stacked `[η_bsv, κ]` layout (the ODE
    /// walk emits a zero `∂f/∂η_ruv` column; the shared assembly applies the variance
    /// scaling and the residual-eta column). Only the **non-IOV** ODE M3 + `iiv_on_ruv`
    /// combo stays FD, gated by `iiv_on_ruv_forces_fd` (`n_kappa == 0`). LTBS still declines.
    #[test]
    fn ode_iov_supported_admits_m3_and_the_ruv_triple() {
        let mut model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
        assert_eq!(model.n_kappa, 1);
        assert!(model.ode_spec.is_some(), "must be an ODE model");
        // Plain ODE IOV: analytic.
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        // ODE IOV + iiv_on_ruv (no M3): analytic as of #486.
        model.residual_error_eta = Some(1);
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "ODE IOV + iiv_on_ruv must be on the analytic path (#486)"
        );
        assert!(
            !model.iiv_on_ruv_forces_fd(),
            "IOV (n_kappa > 0) is not forced to FD"
        );
        // The full triple M3 + ODE IOV + iiv_on_ruv: analytic as of #486.
        model.bloq_method = crate::types::BloqMethod::M3;
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "ODE IOV + M3 + iiv_on_ruv (the triple) must be analytic (#486)"
        );
        assert!(
            !model.iiv_on_ruv_forces_fd(),
            "the IOV triple is not forced to FD"
        );
        assert!(
            iov_sens_supported(&model),
            "iov_sens_supported follows ode_iov_supported"
        );
        // LTBS still declines (the in-walk transform is not composed with the post-walk
        // quotient on the IOV path).
        let mut ltbs = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
        ltbs.residual_error_eta = Some(1);
        ltbs.bloq_method = crate::types::BloqMethod::M3;
        ltbs.log_transform = true;
        assert!(
            !crate::sens::ode_provider::ode_iov_supported(&ltbs),
            "ODE IOV + M3 + iiv_on_ruv + LTBS stays FD"
        );
    }

    #[test]
    fn ode_iov_expr_scale_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_ODE_EXPRSCALE).expect("parse expr-scale IOV");
        let subject = iov_subject();
        // stacked = [η_cl, η_v, κ_g0, κ_g1] (n_eta = 2, n_kappa = 1, K = 2).
        check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
    }

    /// #486: ODE IOV + `TIME` switch + η-dependent `ExpressionScale` `obs_scale = V`.
    /// The per-event stacked walk (TIME) composes with the per-occasion post-walk scale
    /// quotient (built at `t = 0`, matching production's `apply_scaling`); value/∂/∂²
    /// over `[η_bsv, κ_g0, κ_g1]` + θ must match FD of `predict_iov`.
    #[test]
    fn ode_iov_time_expression_scale_matches_fd_of_predict_iov() {
        const M: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVCL_LATE(0.1, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  if (TIME > 20.0) {
    CL = TVCL_LATE * exp(ETA_CL + KAPPA_CL)
  } else {
    CL = TVCL * exp(ETA_CL + KAPPA_CL)
  }
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
        let model = parse_model_string(M).expect("parse ODE IOV TIME expr-scale");
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        // obs [1,6,12 | 25,30,36] straddle TIME=20. stacked = [η_cl, η_v, κ_g0, κ_g1].
        check_iov_provider_vs_fd(
            &model,
            &iov_subject(),
            &[0.2, 0.1, 10.0],
            &[0.12, -0.08, 0.05, -0.10],
        );
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

    /// **IOV × steady-state × estimated lagtime is now analytic (#486, PR3 sub-case (a)).**
    /// Each occasion's SS dose gets its own `K_SS_SEED` pre-arrival seed (phase `II − lag`,
    /// `ss_state_at_phase_g` seeded with that occasion's κ) — validated against the dense
    /// (non-event-driven) predictor for this exact combination in
    /// `ode_provider_ss_lagtime_matches_production`/`..._infusion_...` (including an
    /// observation strictly inside the pre-arrival window). `predict_iov`, this test's
    /// oracle, has no such seed of its own (a pre-existing, orthogonal gap — it would read
    /// zero for a pre-arrival observation regardless of gate/gradient correctness), so this
    /// test uses the default `iov_subject()` times (post-arrival only) to isolate the SS
    /// dose's own event-time saltation under IOV.
    #[test]
    fn ode_iov_ss_lagtime_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_LAG_ODE).expect("parse ODE IOV+lag");
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "ODE IOV + bare lagtime must be supported"
        );
        let mut subject = iov_subject();
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, true, 12.0),
        ];
        assert!(subject.doses[0].ss && subject.doses[0].ii > 0.0);
        let theta = [0.2, 10.0, 0.5];
        let stacked = [0.12, -0.08, 0.05, -0.10];
        assert!(
            crate::sens::ode_provider::ode_subject_sensitivities_iov(
                &model, &subject, &theta, &stacked,
            )
            .is_some(),
            "SS + lagtime IOV subject must be analytic now (#486)"
        );
        // stacked = [η_cl, η_v, κ_g0, κ_g1]; θ = [TVCL, TVV, TVLAG].
        check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
    }

    /// **IOV × rate-defined SS infusion under bioavailability `F ≠ 1` is now analytic**
    /// (#486, PR3 sub-case (b)). `equilibrate_ss_state_g` reads the per-occasion `inf_eff`
    /// jet (window `F·duration`, rate held) instead of the fixed raw duration.
    #[test]
    fn ode_iov_ss_rate_defined_infusion_under_f_matches_fd_of_predict_iov() {
        const WARFARIN_IOV_SS_F_ODE: &str = r#"
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
        let model = parse_model_string(WARFARIN_IOV_SS_F_ODE).expect("parse IOV+SS+F ODE");
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        let mut subject = iov_subject();
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 50.0, true, 12.0),
            DoseEvent::new(24.0, 100.0, 1, 50.0, true, 12.0),
        ];
        assert!(subject.doses[0].ss && subject.has_rate_defined_infusion());
        let theta = [0.13, 8.0, 0.7];
        let stacked = [0.12, -0.08, 0.05, -0.10];
        assert!(
            crate::sens::ode_provider::ode_subject_sensitivities_iov(
                &model, &subject, &theta, &stacked,
            )
            .is_some(),
            "rate-defined SS infusion under F IOV subject must be analytic now (#486)"
        );
        check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
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

    // 1-cpt IV ODE IOV + TV-cov with a **κ-coupled `init(...)` baseline**: `init(central) =
    // BASE` where `BASE = TVBASE·exp(ETA_V + KAPPA_CL)` carries η_v *and* the occasion κ, so the
    // event-driven IOV walk's `tvcov_init_state` seed (built from the first-record snapshot's
    // stacked `(θ, η_bsv, κ)` duals) must fold `∂init/∂κ` onto the correct occasion axis (#486).
    const WARFARIN_IOV_ODE_INIT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVBASE(40.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL   = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V    = TVV  * exp(ETA_V)
  BASE = TVBASE * exp(ETA_V + KAPPA_CL)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE
  d/dt(central) = -(CL/V) * central
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

    /// **`init(...)` × IOV on the ODE walk** (#486, branch G-ODE). A κ-coupled `init(central) =
    /// BASE` baseline (`BASE = TVBASE·exp(ETA_V + KAPPA_CL)`) on a 1-cpt IV ODE IOV + TV-cov
    /// subject. The IOV outer/inner run through the same `integrate_tvcov_readout` the non-IOV
    /// walk uses, so `tvcov_init_state` already seeds `init` from the first-record snapshot's
    /// stacked `(θ, η_bsv, κ)` duals — the Taylor deltas carry `∂init/∂κ` onto the correct
    /// occasion axis with no IOV-specific code. Validated against FD of `predict_iov` over the
    /// full stacked layout, plus inner/outer η-parity.
    #[test]
    fn ode_iov_init_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_ODE_INIT).expect("parse ODE IOV+init");
        assert_eq!(model.n_kappa, 1);
        assert!(model.ode_spec.is_some());
        assert!(
            model.ode_spec.as_ref().unwrap().init_fn.is_some(),
            "fixture must declare an init(...)"
        );
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "ODE IOV + init must be analytic (#486)"
        );
        let subject = iov_tvcov_subject(false);
        // θ = [TVCL, TVV, THETA_WT, TVBASE]; stacked = [η_cl, η_v, κ_g0, κ_g1].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.75, 40.0],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    #[test]
    fn ode_iov_init_inner_eta_grad_matches_outer() {
        check_iov_inner_matches_outer(
            &parse_model_string(WARFARIN_IOV_ODE_INIT).expect("parse ODE IOV+init"),
            &iov_tvcov_subject(false),
            &[0.2, 10.0, 0.75, 40.0],
            &[0.12, -0.08, 0.05, -0.10],
        );
    }

    // ODE IOV with a `first_order` input-rate forcing AND a κ-coupled `init(central) = BASE`
    // baseline — the composition the #660 (`FirstOrder` under IOV) and #486 (`init` under IOV)
    // gate relaxations jointly admit. Both ride the same `integrate_tvcov_readout` walk: the
    // dose feeds `R_in = KA·(remaining)` and the init baseline is seeded from the first-record
    // stacked snapshot, so the two are orthogonal. This pins that they compose correctly.
    const WARFARIN_IOV_ODE_INIT_FO: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVBASE(40.0, 1.0, 500.0)
  theta TVKA(1.2, 0.01, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL   = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V    = TVV  * exp(ETA_V)
  KA   = TVKA
  BASE = TVBASE * exp(ETA_V + KAPPA_CL)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE
  d/dt(central)  = first_order(ka=KA) - (CL/V) * central
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

    /// **`init(...)` × `first_order` input-rate forcing × IOV on the ODE walk** (#486, the
    /// combination the #660 and #486 IOV relaxations jointly admit). Validated against FD of
    /// `predict_iov` over the stacked layout, plus inner/outer parity.
    #[test]
    fn ode_iov_init_first_order_provider_matches_fd_of_predict_iov() {
        let model = parse_model_string(WARFARIN_IOV_ODE_INIT_FO).expect("parse ODE IOV+init+FO");
        assert!(!model.ode_spec.as_ref().unwrap().input_rate.is_empty());
        assert!(model.ode_spec.as_ref().unwrap().init_fn.is_some());
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "ODE IOV + init + first_order must be analytic"
        );
        let subject = iov_tvcov_subject(false);
        // θ = [TVCL, TVV, THETA_WT, TVBASE, TVKA]; stacked = [η_cl, η_v, κ_g0, κ_g1].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.75, 40.0, 1.2],
            &[0.12, -0.08, 0.05, -0.10],
        );
        check_iov_inner_matches_outer(
            &model,
            &subject,
            &[0.2, 10.0, 0.75, 40.0, 1.2],
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

    /// 1-cpt IV IOV `[odes]` model whose Form-C readout references a θ (`TVBASE`) **and**
    /// an η (`ETA_CL`) directly (#486). The parser desugars each bare θ/η into a synthetic
    /// individual parameter (`__ferx_ro_*`); under IOV those synthetics ride the stacked
    /// `(θ, η_bsv, κ)` chain like any other individual parameter. `ETA_CL` is the BSV η that
    /// also carries the per-occasion κ, so the readout's `∂y/∂η_cl` couples the explicit
    /// `(1 + ETA_CL)` term with the κ-driven state — exercising the synthetic-param seeding
    /// on the IOV walk (the path the #631 review flagged as previously FD-only).
    const WARFARIN_IOV_ODE_DIRECT_THETA_ETA: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBASE(0.5, 0.0, 100.0)
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
  y = central / V * (1.0 + ETA_CL) + TVBASE
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    /// #486 + #439: a direct-θ/η Form-C readout under **IOV** must take the analytic ODE
    /// IOV path and match FD of `predict_iov` (value, η-Hessian, θ-grad, η×θ cross). Before
    /// #631 such a readout was not `dual_evaluable`, so `ode_iov_supported` returned false
    /// and the subject fell back to finite differences; this confirms the synthetic
    /// readout parameters seed the stacked `(θ, η_bsv, κ)` chain correctly on the IOV walk.
    #[test]
    fn ode_iov_form_c_direct_theta_eta_matches_production() {
        let model = parse_model_string(WARFARIN_IOV_ODE_DIRECT_THETA_ETA)
            .expect("parse ODE IOV direct θ/η");
        assert_eq!(model.n_kappa, 1);
        assert!(model.ode_spec.is_some());
        // 2 real (CL, V) + 2 synthetic (__ferx_ro_th2, __ferx_ro_eta0) individual params,
        // and no new omega for the direct η reference.
        assert_eq!(
            model.n_eta, 2,
            "direct ETA_CL reuses the existing BSV η (no new omega)"
        );
        assert_eq!(
            model.pk_indices.len(),
            4,
            "CL, V + 2 synthetic readout params"
        );
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "direct-θ/η Form-C readout under IOV should be analytic (#486)"
        );
        let subject = iov_subject();
        // stacked = [η_cl, η_v, κ_g0, κ_g1]; θ = [TVCL, TVV, TVBASE].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 0.5],
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

    /// #486: a `TIME`-switched CL on a **closed-form IOV** model. With no TV
    /// covariates the subject routes to the per-event stacked walk purely by
    /// `uses_time_builtin`; the value/∂/∂² over `[η_bsv, κ_g0, κ_g1]` + θ must match
    /// central FD of `predict_iov` (which threads the same per-event TIME).
    #[test]
    fn iov_time_builtin_provider_matches_fd_of_predict_iov() {
        const IOV_ORAL_TIME: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVCL_LATE(0.1, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  if (TIME > 20.0) {
    CL = TVCL_LATE * exp(ETA_CL + KAPPA_CL)
  } else {
    CL = TVCL * exp(ETA_CL + KAPPA_CL)
  }
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
        let model = parse_model_string(IOV_ORAL_TIME).expect("parse IOV oral TIME");
        assert_eq!(model.n_kappa, 1);
        let subject = iov_subject(); // obs [1,6,12 | 25,30,36] straddle TIME=20
        assert!(
            !subject.has_tv_covariates(),
            "no TV cov: routed by uses_time only"
        );
        assert!(
            iov_analytical_supported(&model),
            "closed-form IOV TIME must be provider-supported"
        );
        // θ = [TVCL, TVCL_LATE, TVV, TVKA]; stacked = [η_cl, η_v, η_ka, κ_g0, κ_g1].
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 0.1, 10.0, 1.5],
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
