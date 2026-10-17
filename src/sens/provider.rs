//! The closed-form engines the adjoint recorder (`sens::ode_provider::adjoint`) runs on
//! any number type: the static superposition predictions (`analytic_static_predictions`),
//! the event-driven walk (`closed_form_event_predictions`) and the analytic `init(...)`
//! impulses, each the same kernels and conventions as the `f64` predictor, generic over
//! `PkNum` — plus their scope predicates and the public per-observation η-gradient
//! (`subject_eta_grad`) the recorder backs.
#![allow(clippy::needless_range_loop)]

use super::num::PkNum;
use super::one_cpt::{one_cpt_conc_g, one_cpt_ig_conc_g, one_cpt_transit_conc_g};
use super::three_cpt::three_cpt_conc_d;
use super::two_cpt::{two_cpt_conc_d, two_cpt_ig_conc_g, two_cpt_transit_conc_g};
use crate::types::{
    CompiledModel, DoseEvent, GradientMethod, PkModel, ScalingSpec, Subject, PK_IDX_CL, PK_IDX_CV2,
    PK_IDX_F, PK_IDX_KA, PK_IDX_LAGTIME, PK_IDX_MAT, PK_IDX_MTT, PK_IDX_N, PK_IDX_Q, PK_IDX_Q3,
    PK_IDX_V, PK_IDX_V2, PK_IDX_V3,
};

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
        // Inverse-Gaussian `mat`/`cv2` (#790); the analytic `one_cpt_ig` closed form
        // reads them, so they are differentiated too. WITHOUT these arms an IG model's
        // `pk_indices` (slots 11/12) fail `analytical_supported_core`'s
        // `slot_to_dim(s).is_some()` check and the whole model silently falls back to
        // finite-difference gradients — defeating the closed form's exact-`Dual2` point.
        PK_IDX_MAT => Some(11),
        PK_IDX_CV2 => Some(12),
        _ => None,
    }
}

/// Width of the per-model PK-slot lookup tables — spans the full PK slot space
/// (`CL, V1, Q2, V2, KA, F, Q3, V3, LAGTIME`, the transit `N`/`MTT` slots #386, and
/// the modeled `D{cmt}`/`R{cmt}` slots the parser assigns from `PK_IDX_LAGTIME+1`
/// upward, #486). Sized to `crate::types::MAX_PK_PARAMS` so a subject with three or
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
/// `pk_slots()`; a literal-constant slot (`ka = 1.0`) is absent from it, recorded in
/// `const_pk_slots`, and seeded as a constant by the recorders, so it is covered too.
/// The parser leaves no third case: an argument that is neither is a parse error.
fn prog_covers_required_pk_slots(
    model: &CompiledModel,
    prog: &crate::parser::model_parser::IndivParamProgram,
) -> bool {
    let prog_slots = prog.pk_slots_ref();
    let literal = model.indiv_param_partials.const_pk_slots();
    model.pk_model.required_pk_params().iter().all(|(slot, _)| {
        (prog_slots.contains(slot) || literal.contains(slot)) && slot_to_dim(*slot).is_some()
    })
}

/// Elapsed time since a dose's *lagged* arrival, as a dual carrying the lagtime
/// sensitivity (`∂elapsed/∂lagtime = −1`). Mirrors `crate::pk::predict_concentration`:
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

/// The seven PK-class booleans that select which closed-form concentration kernel a dose
/// superposition dispatches to. Bundled into one `Copy` argument so `superpose_doses`
/// takes it in place of seven flags, and the outer (`run_obs`) and inner
/// (`run_obs_grad`) walkers build it from the same field set.
#[derive(Clone, Copy)]
struct PkClassFlags {
    oral: bool,
    two_cpt: bool,
    three_cpt: bool,
    transit: bool,
    two_cpt_transit: bool,
    ig: bool,
    two_cpt_ig: bool,
}

/// The 14 PK-parameter duals a dose superposition reads, seeded once from `seed_dim`
/// (which slot maps to which compact dual axis; `None` = constant). Generic over the
/// seeded number type so the outer walker instantiates it at `Dual2<N>` (full `(θ,η)`
/// jet + Hessian) and the inner at `Dual1<N>` (η-gradient only) from identical code.
/// `lag_val` is kept alongside `lag_d` so the lagged-arrival reset test can branch on the
/// plain value without a borrow.
struct SeededPkDuals<T: PkNum> {
    cl_d: T,
    v1_d: T,
    q_d: T,
    v2_d: T,
    ka_d: T,
    f_d: T,
    q3_d: T,
    v3_d: T,
    lag_d: T,
    lag_val: f64,
    n_d: T,
    mtt_d: T,
    mat_d: T,
    cv2_d: T,
    /// The 2-/3-cpt disposition (micro parameters and macro rates), built once per
    /// subject by [`Self::with_disposition`] for the multi-compartment closed forms.
    two: Option<super::two_cpt::TwoCpt<T>>,
    three: Option<super::three_cpt::ThreeCpt<T>>,
}

impl<T: PkNum> SeededPkDuals<T> {
    /// Seed every PK slot through `dv(slot, value)` — the general form of `Self::seed`,
    /// used by the adjoint gradient to seed each slot with its η-tangents directly.
    fn seed_with(pk: &crate::types::PkParams, dv: impl Fn(usize, f64) -> T) -> Self {
        let lag_val = pk.lagtime();
        Self {
            cl_d: dv(PK_IDX_CL, pk.cl()),
            v1_d: dv(PK_IDX_V, pk.v()),
            q_d: dv(PK_IDX_Q, pk.q()),
            v2_d: dv(PK_IDX_V2, pk.v2()),
            ka_d: dv(PK_IDX_KA, pk.ka()),
            f_d: dv(PK_IDX_F, pk.f_bio()),
            q3_d: dv(PK_IDX_Q3, pk.q3()),
            v3_d: dv(PK_IDX_V3, pk.v3()),
            lag_d: dv(PK_IDX_LAGTIME, lag_val),
            lag_val,
            n_d: dv(PK_IDX_N, pk.n_transit()),
            mtt_d: dv(PK_IDX_MTT, pk.mtt()),
            mat_d: dv(PK_IDX_MAT, pk.mat()),
            cv2_d: dv(PK_IDX_CV2, pk.cv2()),
            two: None,
            three: None,
        }
    }

    /// Build the disposition the `flags` class superposes over, once.
    fn with_disposition(mut self, flags: PkClassFlags) -> Self {
        if flags.three_cpt {
            self.three = Some(super::three_cpt::ThreeCpt::new(
                self.cl_d, self.v1_d, self.q_d, self.v2_d, self.q3_d, self.v3_d,
            ));
        } else if flags.two_cpt {
            self.two = Some(super::two_cpt::TwoCpt::new(
                self.cl_d, self.v1_d, self.q_d, self.v2_d,
            ));
        }
        self
    }
}

/// The reset segment floor at `t_obs`: the most recent EVID=3/4 reset at or before this
/// observation (`−∞` when the subject has no resets). Doses before it were zeroed out of
/// the compartments, so the superposition excludes them — mirroring the event-driven
/// walker's `event_reset_floor`.
#[inline]
fn reset_floor_at(subject: &Subject, t_obs: f64) -> f64 {
    subject
        .reset_times
        .iter()
        .copied()
        .filter(|&r| r <= t_obs)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Superpose one observation's dose contributions through the closed-form PK kernel the
/// `flags` select: `f = Σ conc(dose, elapsed)`, restricted to the current reset segment
/// (a dose **recorded** before `reset_floor` is excluded, even one whose lagged arrival
/// comes after the reset: NONMEM cancels it, and so does the event-driven walk, #1587 —
/// before which both kept it, PR #381 #2). `elapsed`
/// carries the lagtime shift and the SS pre-arrival tail wrap (`lagged_elapsed`).
/// Generic over the seeded dual type, so the `Dual2` outer (`run_obs`) and `Dual1`
/// inner (`run_obs_grad`) walkers share the byte-identical dispatch ladder.
#[inline]
fn superpose_doses<T: PkNum>(
    d: &SeededPkDuals<T>,
    flags: PkClassFlags,
    subject: &Subject,
    t_obs: f64,
    reset_floor: f64,
) -> T {
    let mut fd = T::from_f64(0.0);
    // The value path's `SS=1` reset (#1576): `crate::pk::predict_concentration`.
    let ss_cutoff = crate::dosing::ss_reset_cutoff(&subject.doses, t_obs);
    for (k, dose) in subject.doses.iter().enumerate() {
        if !crate::dosing::evid_reset_live(dose.time, reset_floor)
            || !crate::dosing::ss_reset_live(&subject.doses, ss_cutoff, k)
        {
            continue;
        }
        let Some(elapsed) = lagged_elapsed(dose, t_obs, d.lag_val, d.lag_d) else {
            continue;
        };
        let c = if flags.transit {
            one_cpt_transit_conc_g(dose, elapsed, d.cl_d, d.v1_d, d.n_d, d.mtt_d, d.f_d)
        } else if flags.two_cpt_transit {
            two_cpt_transit_conc_g(
                dose, elapsed, d.cl_d, d.v1_d, d.q_d, d.v2_d, d.n_d, d.mtt_d, d.f_d,
            )
        } else if flags.ig {
            one_cpt_ig_conc_g(dose, elapsed, d.cl_d, d.v1_d, d.mat_d, d.cv2_d, d.f_d)
        } else if flags.two_cpt_ig {
            two_cpt_ig_conc_g(
                dose, elapsed, d.cl_d, d.v1_d, d.q_d, d.v2_d, d.mat_d, d.cv2_d, d.f_d,
            )
        } else if let Some(disp) = d.three.as_ref().filter(|_| flags.three_cpt) {
            three_cpt_conc_d(dose, elapsed, disp, d.ka_d, d.f_d, flags.oral)
        } else if let Some(disp) = d.two.as_ref().filter(|_| flags.two_cpt) {
            two_cpt_conc_d(dose, elapsed, disp, d.ka_d, d.f_d, flags.oral)
        } else {
            one_cpt_conc_g(dose, elapsed, d.cl_d, d.v1_d, d.ka_d, d.f_d, flags.oral)
        };
        fd = fd + c;
    }
    fd
}

/// The model-level closed-form scope check shared by `analytical_supported` and
/// `tvcov_analytical_supported` — everything except the `TIME`-routing clause, so the
/// two can compose without recursion (#637 review #1).
fn analytical_supported_core(model: &CompiledModel) -> bool {
    // A compartment-free model (#811) carries a *placeholder* `pk_model` that the
    // `matches!` below would happily admit — and then every walk here would
    // evaluate a closed form the model does not have, against doses it does not
    // carry. Decline before the match so the placeholder can never be dispatched
    // on; such a model is served by `sens::algebraic`, which needs no closed form.
    if model.is_algebraic() {
        return false;
    }
    matches!(
        model.pk_model,
        PkModel::OneCptIv
            | PkModel::OneCptOral
            | PkModel::OneCptTransit
            | PkModel::OneCptIg
            | PkModel::TwoCptIv
            | PkModel::TwoCptOral
            | PkModel::TwoCptTransit
            | PkModel::TwoCptIg
            | PkModel::ThreeCptIv
            | PkModel::ThreeCptOral
    ) && model.ode_spec.is_none()
        && model.tv_fn.is_some()
        && model.n_kappa == 0
        && scaling_supported(model)
        && init_supported(model)
        && analytic_readout_dual_supported(model)
        // Every individual-parameter slot must be one we differentiate; a slot outside
        // `slot_to_dim`'s map (e.g. a modeled-dose `D{cmt}`/`R{cmt}` slot) routes to FD.
        && model.pk_indices.iter().all(|&s| slot_to_dim(s).is_some())
        && closed_form_slot_count_supported(model)
}

/// Widest differentiated-slot count the closed-form dispatch tables instantiate. Both the
/// outer (`subject_sensitivities` → `run_obs`) and inner (`subject_eta_grad` →
/// `run_obs_grad`) tables enumerate `1..=9`; a wider model falls through their `_ => None`
/// arm. **Keep these three in step** — raising this constant without adding the matching
/// `const_dispatch!` arms would re-open the misreport below.
const MAX_CLOSED_FORM_SLOTS: usize = 9;

const ANALYTIC_MIXED_N: [usize; 2] = [4, 6];

const _: () = assert!(ANALYTIC_MIXED_N[0] == 4 && ANALYTIC_MIXED_N[1] == 6);

#[cfg(test)]
thread_local! {
    static MIXED_ANALYTIC_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Observations whose jet came from the explicit-kernel loop in `run_obs` rather
    /// than `superpose_doses` — the discriminator a two-loop parity test asserts (#1576).
    static EXPLICIT_ANALYTIC_OBS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Whether the model's differentiated-slot count fits the closed-form dispatch tables.
///
/// Without this the count is unbounded: `analytical_supported_core` checks only that every
/// slot *maps* via `slot_to_dim`, never how many there are. A readout that claims several
/// spare slots (`allocate_readout_extra_slots` — text-referenced parameters, and synthetic
/// direct θ/η as of #486) can push `pk_slots()` past 9 on top of the structural set, so
/// `analytical_supported` returned `true`, `build_info::gradient_method{,_inner}` persisted
/// "analytic", and then *every* subject fell through `const_dispatch!`'s `_ => None` to
/// finite differences at ~10× the cost — precisely the "claim analytic then FD every
/// subject" drift that `analytic_readout_dual_supported` exists to prevent (#637). Declining
/// here changes no numbers (those subjects already ran on FD); it makes the reported method
/// match the route (PR #950 review #2).
///
/// Bounds the **program** slot list, which is what the readout allocator grows and what the
/// providers dispatch on. Since #1899 the `lognormal_param_derivatives` fallback dispatches on
/// the same list (`CompiledModel::pk_slot_rows` is the program's `(pk_slot, var_slot)` pairs
/// whenever a program exists), so this bound covers both routes; before that it keyed on
/// `model.pk_indices`, which this gate never saw.
fn closed_form_slot_count_supported(model: &CompiledModel) -> bool {
    model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .is_none_or(|prog| prog.pk_slots_ref().len() <= MAX_CLOSED_FORM_SLOTS)
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
pub(crate) fn analytic_readout_dual_supported(model: &CompiledModel) -> bool {
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
/// **tv-covariate / oral-infusion / TIME** event-walk path (#650).
///
/// The walk's `PkDual` carries only the eight structural slots (`CL..V3`), so a readout
/// referencing a 9th+ individual parameter (reachable once several non-structural readout
/// params are allocated — a 3-cpt-oral sigmoid-Emax `y = <expr>` exhausts the low slots
/// immediately) used to fall back to FD here. Since #486 the walk materializes those higher
/// slots per observation (`ro_extra` in `run_obs_tvcov` / `run_obs_grad_tvcov`) — value from
/// `pk_param_fn`, jet from the same program derivatives — so the slot ceiling is gone and
/// this now reduces to the static-path scope.
pub(crate) fn readout_tvcov_supported(model: &CompiledModel) -> bool {
    let Some(ar) = &model.analytic_readout else {
        return true;
    };
    match &ar.program {
        Some(_) => analytic_readout_dual_supported(model),
        None => false,
    }
}

/// Evaluate an analytic Form C readout jet at one observation (#650), generic over
/// the `PkNum` dual type.
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
fn eval_readout_jet<T: crate::sens::num::vm::VmNum>(
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

/// Apply the analytic Form-C readout (#650) to one observation's (already clamped)
/// concentration jet: replace `conc` with `y = <expr>` over the seeded dual (central
/// amount = conc × V), under this observation's `ModelTimeGuard` so a `TIME`-referencing
/// readout resolves `Op::PushTime` to `t_obs` (matching production's
/// `apply_analytic_readout` guard; a no-op for a `TIME`-free readout). Returns `conc`
/// unchanged with no readout, touching the covariate/time snapshots only on the readout
/// path. Generic over the dual width, so the `Dual2` outer (`run_obs`) and `Dual1` inner
/// (`run_obs_grad`) walkers share the guard/eval wiring (#637); the resulting `∂y/∂pk`
/// (and `∂²y/∂pk²` on the outer width) then rides the same `pd`/`dp_deta` chain.
#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_readout_jet<T: crate::sens::num::vm::VmNum>(
    conc: T,
    readout: Option<&crate::parser::model_parser::OdeOutputProgram>,
    ro_pk_duals: Option<&[T; N_PK]>,
    subject: &Subject,
    obs_i: usize,
    ro_state: &mut Vec<T>,
    ro_vars: &mut Vec<T>,
    ro_stack: &mut Vec<T>,
) -> T {
    if let (Some(prog), Some(pkd)) = (readout, ro_pk_duals) {
        // A `TIME`-referencing readout resolves `Op::PushTime` from the model-time
        // thread-local; enter this observation's time to match production's
        // `apply_analytic_readout` guard (no-op for a `TIME`-free readout).
        // `readout_time`, like production — the raw data-file clock.
        let _time_guard =
            crate::parser::model_parser::ModelTimeGuard::enter(subject.readout_time(obs_i));
        eval_readout_jet::<T>(
            prog,
            conc,
            pkd,
            subject.obs_cov(obs_i),
            ro_state,
            ro_vars,
            ro_stack,
        )
    } else {
        conc
    }
}

/// Whether the model's `[initial_conditions]` baseline (#521) is one the Dual2
/// provider differentiates exactly (#524): every init carries a compiled
/// `amount_deriv` program and the `(θ, η)` axis count fits the init dispatch
/// table (`1..=MAX_SCALE_AXES`, shared with the scale program). An init-free
/// model is trivially supported; a hand-built init with no program routes to FD.
fn init_supported(model: &CompiledModel) -> bool {
    model.analytical_init.is_empty()
        || (model
            .analytical_init
            .iter()
            .all(|i| i.amount_deriv.is_some()))
}

/// Whether the model's output scaling is one the provider differentiates exactly:
/// `None` / constant `ScalarScale` (a per-jet divisor, `∂k/∂η = ∂k/∂θ = 0`), or an
/// `ExpressionScale` carrying a `Dual2`-differentiable program whose axis counts
/// match the model. `ExpressionScale` without a program
/// and `PerCmt` route to FD.
fn scaling_supported(model: &CompiledModel) -> bool {
    match &model.scaling {
        ScalingSpec::None | ScalingSpec::ScalarScale(_) => true,
        ScalingSpec::ExpressionScale { deriv: Some(p), .. } => {
            p.n_theta_axis() == model.n_theta && p.n_eta_axis() == model.n_eta
        }
        _ => false,
    }
}

/// Whether the exact analytic **outer** (population) FOCE/FOCEI gradient is
/// available for `model`: it is in the sensitivity provider's scope (non-IOV
/// `sens_supported` or `iov_analytical_supported`) and the user did not
/// force finite differences via `gradient_method = fd`.
///
/// Single source of truth for the analytic-vs-FD outer-gradient decision. The
/// outer-loop gradient dispatch (`outer_optimizer::population_gradient`),
/// `Optimizer::resolve_auto`, and
/// `build_info::gradient_method_outer` all consult this so the `auto` optimizer
/// can never resolve to a gradient-based optimizer while the loop actually
/// computes an FD gradient (a drift that would run a gradient optimizer on a
/// noisy FD gradient). The per-eval `reconverge_gradient_interval` override is
/// orthogonal and handled at the dispatch site, not here.
///
/// Non-Gaussian endpoints — `[event_model]` TTE and `[binary_model]` logistic (#760) —
/// are excluded: their data log-likelihood has no analytic outer gradient (the
/// sensitivity provider only covers the structural PK/PD model), so such an objective
/// (or a mixed PK + non-Gaussian objective) must be differentiated by finite differences.
/// Without this guard the model would report an analytic gradient it cannot supply, so
/// `resolve_auto` would pick a gradient-based optimizer that then stalls on a meaningless
/// gradient (these endpoints are FD-only — see `docs/estimation/tte.qmd`).
pub fn analytic_outer_gradient_available(model: &CompiledModel) -> bool {
    !GradientMethod::forced_fd(model) && analytic_outer_gradient_in_scope(model)
}

/// The *scope* half of `analytic_outer_gradient_available`: whether the model's
/// structure admits the exact analytic outer gradient, ignoring the user's
/// `gradient_method` override.
///
/// Split out so the `gradient = fd` ⇒ `optimizer = auto` coupling (#1381) can ask
/// the counterfactual — "what would `auto` have picked had the gradient not been
/// forced?" — off the *same* predicate the live decision uses, rather than a second
/// copy of the scope list that would drift from it.
pub(crate) fn analytic_outer_gradient_in_scope(model: &CompiledModel) -> bool {
    // The adjoint gradient's model scope: it carries none of the forward provider's width
    // or slot limits. A residual `correlation … by` line with kappa (#1082 PR5) is the one
    // exclusion: its IOV marginal scores a dense `R`, which the adjoint's IOV objective does
    // not yet carry (`adjoint::kappa_with_residual_correlation`).
    crate::sens::ode_provider::adjoint::model_in_scope(model)
        && !crate::sens::ode_provider::adjoint::kappa_with_residual_correlation(model)
}

/// `analytic_outer_gradient_available`, kept as the shared entry point that
/// `crate::types::Optimizer::resolve_auto` and `build_info::gradient_method_outer`
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

/// Dual width of one IOV program evaluation on the outer builder: the declared θ, the
/// combined `[η_bsv, κ]`, and — for a `[covariate_nn]` model — one axis per network
/// output. The IOV twin of `tvcov_program_axes` (#1339).
fn iov_program_axes(model: &CompiledModel) -> usize {
    n_program_theta(model) + model.n_eta + model.n_kappa + nn_output_count(model)
}

/// Shared body. `require_theta_axis` distinguishes the outer predicate from the
/// η-only one; every other clause applies to both.
fn iov_analytical_supported_core(model: &CompiledModel, require_theta_axis: bool) -> bool {
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
    // triple is analytic too (#486); the **non-IOV ODE** M3 + `iiv_on_ruv` combo is analytic
    // as well (#623) — every `iiv_on_ruv` combination is served, with no FD carve-out.
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
    // The four closed-form absorption models ride the same stacked-κ walk through
    // `absorption_walk::closed_form_sens_walk_g` (#1560) — but only with an ODE twin, which
    // is what serves a subject whose (interval, open dose) pairs leave the tilting domain.
    // A twin-less transit/IG model with IOV is rejected at validation anyway.
    let absorption = crate::pk::absorption_walk::is_absorption_closed_form(model.pk_model)
        && model.absorption_ode_equivalent.is_some();
    if !absorption
        && !matches!(
            model.pk_model,
            PkModel::OneCptIv
                | PkModel::OneCptOral
                | PkModel::TwoCptIv
                | PkModel::TwoCptOral
                | PkModel::ThreeCptIv
                | PkModel::ThreeCptOral
        )
    {
        return false;
    }
    // Output scaling: `None` (raw jet), a constant `ScalarScale` divisor, or a differentiable
    // `ExpressionScale` `obs_scale` divisor whose (θ, η) axis counts match the model and fit
    // the scale dispatch table. The `ExpressionScale` quotient is a per-occasion-group
    // post-walk step in `run_obs_iov` / `run_obs_iov_eta` (the closed-form twin of the ODE-IOV
    // path, #575/#590); a constant `ScalarScale k` is the trivial case of that quotient — a
    // covariate/η/κ-independent `f/k` applied uniformly to every jet entry (#486 IOV-scope
    // parity, mirroring the non-IOV `f_scaled = f/k`). LTBS composes with the OUTER gradient
    // now (#486): `subject_sensitivities_iov` applies the `ln(f)` jet LAST — after the in-walk
    // scale quotient — reproducing production's scale-then-log order `ln(f/s)` (`predict_iov`
    // logs last, after `apply_scaling`). The inner IOV EBE gradient serves LTBS too as of #486
    // (`run_obs_iov_eta` applies the same shared jet last), so this gate admits `log_transform`
    // for both loops and `analytic_inner_common_bail` no longer carries an LTBS × IOV clause.
    match &model.scaling {
        ScalingSpec::None | ScalingSpec::ScalarScale(_) => {}
        // No `n_axes() <= MAX_SCALE_AXES` clause, unlike the non-IOV gate: that bound is
        // the width of the *absolute* `(θ, η)` dispatch `apply_expression_scale_outer`
        // monomorphises on, and the IOV walk does not use it. `run_obs_iov` evaluates the
        // scale through `eval_scale_dual_cols` at the chunk's own width `M = nc +
        // n_stacked`, which `const_dispatch!` already bounds at 24 — so a θ block wider
        // than the table is chunked, exactly like the `[initial_conditions]` amount
        // program right below, whose gate carries no width clause either (#1339).
        ScalingSpec::ExpressionScale { deriv: Some(p), .. }
            if p.n_theta_axis() == model.n_theta && p.n_eta_axis() == model.n_eta => {}
        _ => return false,
    }
    // Lagtime is now carried on the IOV walk (#486): the dose arrival `t + ALAG` is threaded
    // as a moving dual boundary, exactly like a modeled infusion end, so the saltation falls
    // out of the closed-form flow. A subject that *also* carries an SS dose still declines
    // (per-subject, `ss_lagtime_walk_unsupported`) — production's pre-arrival SS overlay has
    // no dual twin yet.
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
    // Analytic Form C readout (`[scaling] y = <expr>`, #655): served on the IOV
    // event-driven walk when it fits the eight structural `PkDual` slots — the same
    // walk-capable scope the TV-cov path uses (`readout_tvcov_supported`: a uniform
    // dual-evaluable `Single` readout, no depot reference, no `[initial_conditions]`,
    // no slot past `PK_IDX_V3`). `run_obs_iov` / `run_obs_iov_eta` seed the readout PK
    // params from the observation's own occasion source, so they carry κ just as the
    // walk's concentration does — matching production's per-occasion
    // `apply_analytic_readout(..., [η_bsv, κ_g], ...)` (#1079). A
    // readout outside that scope (per-CMT, depot, direct θ/η, slot overflow) declines to
    // FD here so the reported method stays honest — the closed-form IOV twin of
    // `analytical_supported_core`'s `analytic_readout_dual_supported` clause.
    // `readout_tvcov_supported` is `true` for a plain (no-readout) model.
    if !readout_tvcov_supported(model) {
        return false;
    }
    let n_eff = model.n_eta + model.n_kappa;
    // The outer θ-axis clause is on the *declared* θ (`n_program_theta`), not
    // `model.n_theta`: a `[covariate_nn]` model's generated weights are not referenceable
    // from the program and reach the PK parameters only through the network outputs,
    // which the outer builder seeds on their own axes and chains back to the weights
    // (`iov_nn_combined_derivs_dyn`, #1339 — the IOV twin of #1300). For a model with
    // no network `n_program_theta == model.n_theta`, so this is the former clause
    // exactly. Two DCM-only conditions ride along: the output-channel factorization the
    // chain relies on (a direct read of a generated weight θ breaks it — same predicate
    // `tvcov_analytical_supported` / `nn_theta_gradient` use), and the builder's own
    // dual width (declared θ + `[η, κ]` + outputs) fitting its `1..=24` dispatch.
    // The walk's θ columns are chunked (`subject_sensitivities_iov`), so `model.n_theta`
    // itself is unbounded here.
    if require_theta_axis && nn_weight_theta_count(model) > 0 {
        if !nn_output_chain_supported(model) || iov_program_axes(model) > ABSORPTION_WALK_AXES {
            return false;
        }
        // No clause here for the three post-walk steps (Form C readout,
        // `[initial_conditions]` impulse, `ExpressionScale` quotient). They used to be
        // representable only on the identity θ chunk, which made this gate unanswerable:
        // whether a subject lands on one chunk depends on its occasion count `K`
        // (`n_theta + n_eta + K·n_kappa ≤ 24`), and a model-level predicate cannot see
        // `K`. Any bound written here — `K = 1`, as a first cut did — reports analytic for
        // a band of models whose every real (`K ≥ 2`) subject then reconverges on FD,
        // exactly the route/report drift #637 forbids. `run_obs_iov` now runs all three in
        // the chunk's own basis instead, so there is nothing left to gate (#1339, PR #1340
        // review P2).
    }
    match model.indiv_param_partials.indiv_param_program.as_ref() {
        Some(prog) => {
            prog_covers_required_pk_slots(model, prog)
                && (!require_theta_axis || prog.n_theta_axis() == n_program_theta(model))
                && prog.n_eta_axis() == n_eff
        }
        None => false,
    }
}

/// The widest `(θ, η[, κ])` the closed-form absorption walk is admitted at (#1560): a
/// **prediction routing** rule — wider subjects take the ODE twin — kept at its historical
/// value so routing, and hence every prediction, is unchanged. No derivative reads it.
const ABSORPTION_WALK_AXES: usize = 24;

/// Whether the analytic providers can differentiate `subject` on the transit/IG absorption
/// walk (#1560) — the scope half of `crate::pk::absorption_walk::walk_eligible`. The walk
/// takes over a subject from the ODE twin only where its gradient is analytic, so no subject
/// trades the twin's analytic ODE sensitivities for finite differences:
///
/// * IOV: the closed-form IOV scope, and a stacked width `n_eta + K·n_kappa` below the dual
///   walk's cap (`MAX_TVCOV_AXES = 24`; the outer seeds at least one θ column beside it). The
///   twin's ODE IOV provider reaches 96, so a many-occasion subject past 24 stays there;
/// * time-varying covariates: the closed-form TV-walk scope (`tvcov_analytical_supported`).
pub(crate) fn absorption_walk_sens_in_scope(model: &CompiledModel, subject: &Subject) -> bool {
    if model.n_kappa > 0 {
        iov_analytical_supported_core(model, true)
            && crate::stats::stacked_layout::stacked_width(
                model.n_eta,
                model.n_kappa,
                crate::stats::stacked_layout::group_count(subject),
            ) < ABSORPTION_WALK_AXES
    } else {
        tvcov_analytical_supported(model)
    }
}

/// True when `subject_sensitivities_tvcov` can serve this model: an analytical
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
/// `pd_from_program`.
pub(crate) fn tvcov_analytical_supported(model: &CompiledModel) -> bool {
    // `analytical_supported_core` (not `analytical_supported`) so a `TIME` model doesn't
    // recurse: `analytical_supported` calls back here for the `uses_time_builtin` case
    // (#637 review #1).
    // Lagtime is carried on the TV-cov / `TIME` event walk since #486 (moving dual arrival
    // boundary); an SS dose alongside it still declines per-subject via
    // `ss_lagtime_walk_unsupported`.
    if !analytical_supported_core(model) {
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
    // Bound the *program's* axes to the dual-walk dispatch cap so the outer (`m_dim`) and
    // inner (`n_eta`) TV-cov tables both resolve — matched analytic scope, no fixed-EBE FD
    // inner split (#449 re-review #2). For a `[covariate_nn]` model the bound is on the
    // axes the walk actually seeds — the user-declared θ, η, and one axis per network
    // output — not on `model.n_theta`, which counts every generated weight (#1300): the
    // weights reach the PK parameters only through the network output, so their columns
    // are chained in after the program walk (`nn_param_derivatives_at_cov`) and the outer
    // walk seeds them in column chunks of `MAX_TVCOV_AXES - n_eta`
    // (`subject_sensitivities_tvcov`). The chunking needs at least one θ column per
    // chunk, hence `n_eta < MAX_TVCOV_AXES` whenever there is a θ column at all.
    if tvcov_program_axes(model) > ABSORPTION_WALK_AXES
        || (model.n_theta > 0 && model.n_eta >= ABSORPTION_WALK_AXES)
    {
        return false;
    }
    // A model that reads a generated weight θ *directly* (`… + B_TYPICAL_PK_2_1`) breaks
    // the output-channel factorization the weight chain relies on; the parser records it
    // and the model routes to FD, loudly (same predicate `nn_theta_gradient` uses).
    if nn_weight_theta_count(model) > 0 && !nn_output_chain_supported(model) {
        return false;
    }
    // Output scaling: `None` / constant `ScalarScale` (a uniform per-jet divisor), or an
    // η-dependent `ExpressionScale` whose subject-static quotient is applied post-walk by
    // `subject_sensitivities_tvcov` / `subject_eta_grad_tvcov` (the same shared
    // `apply_expression_scale_outer` / `_inner_dispatch` the dose-superposition path uses,
    // #486 — closes the TV-cov + expression-scale gap). `scaling_supported` bounds the
    // scale program to the dispatch table; `PerCmt` still routes to FD. LTBS is now admitted
    // (#486): `subject_sensitivities_tvcov` applies the `ln(f)` jet transform LAST, after any
    // scale quotient — reproducing production's scale-then-log order `ln(f/s)` exactly (the
    // whole transform is post-walk on the closed-form path, unlike the ODE in-walk log). The
    // inner EBE gradient is served analytically here too (`subject_eta_grad_tvcov` applies the
    // same `ln(f)` inner jet LAST, #673); since #486 LTBS × IOV is served on the inner as well
    // (`run_obs_iov_eta`), so no LTBS combination routes the inner to FD any more.
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
                // The program's θ axes are the user-declared thetas; generated
                // `[covariate_nn]` weights are not referenceable by name, so for a DCM this
                // is `model.n_theta` minus the weight block (`n_program_theta`).
                && prog.n_theta_axis() == n_program_theta(model)
                && prog.n_eta_axis() == model.n_eta
        }
        None => false,
    }
}

/// Number of auto-generated `[covariate_nn]` weight thetas on the model — `0` without a
/// network (and always `0` without the `nn` feature, where the field does not exist).
#[cfg(feature = "nn")]
pub(crate) fn nn_weight_theta_count(model: &CompiledModel) -> usize {
    model
        .covariate_nns
        .iter()
        .map(|nn| nn.mapper.mlp().n_weights())
        .sum()
}

#[cfg(not(feature = "nn"))]
pub(crate) fn nn_weight_theta_count(_model: &CompiledModel) -> usize {
    0
}

/// Total `[covariate_nn]` output count across every network, in the flat order
/// `ModelNnAxisGuard` seeds them (`covariate_nns` order, each network's outputs in
/// declaration order). `0` without a network.
#[cfg(feature = "nn")]
fn nn_output_count(model: &CompiledModel) -> usize {
    model
        .covariate_nns
        .iter()
        .map(|nn| nn.mapper.mlp().n_outputs())
        .sum()
}

#[cfg(not(feature = "nn"))]
fn nn_output_count(_model: &CompiledModel) -> usize {
    0
}

/// The θ count the compiled `[individual_parameters]` program can seed: every declared
/// theta, i.e. `model.n_theta` less the generated `[covariate_nn]` weight block.
pub(crate) fn n_program_theta(model: &CompiledModel) -> usize {
    model.n_theta - nn_weight_theta_count(model)
}

/// Dual width of one TV-cov program evaluation: the declared θ, the η, and — for a
/// `[covariate_nn]` model — one axis per network output (`ModelNnAxisGuard`), so the
/// program walk yields `∂p/∂z` alongside `∂p/∂(θ,η)`. This, not `model.n_theta + n_eta`,
/// is what the `1..=MAX_TVCOV_AXES` dispatch must accommodate (#1300).
fn tvcov_program_axes(model: &CompiledModel) -> usize {
    n_program_theta(model) + model.n_eta + nn_output_count(model)
}

/// Whether every `[covariate_nn]` weight reaches the likelihood through its network's
/// output alone — the invariant behind chaining `∂p/∂w = ∂p/∂z · ∂z/∂w`. The parser
/// records a direct read of a generated weight θ under
/// `NN_WEIGHT_DIRECT_REFERENCE_MARKER`; such a model declines to FD here exactly as
/// `nn_theta_gradient::NnGradPlan::build` does.
pub(crate) fn nn_output_chain_supported(model: &CompiledModel) -> bool {
    !model
        .parse_warnings
        .iter()
        .any(|w| w.contains(crate::parser::model_parser::NN_WEIGHT_DIRECT_REFERENCE_MARKER))
}

/// Clamp a **dual** modeled `D`/`R` to its domain floor, keeping the jet in the
/// interior and dropping it at the wall — the dual mirror of
/// `crate::types::clamp_above_floor` used by the f64 `DoseEvent::resolve_rate`.
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
/// `DoseEvent::resolve_rate`'s slot `.expect` (#486). Fixed subjects are trivially
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
/// `modeled_dose_inf_duals`.
fn resolve_eff_doses<'a>(
    model: &CompiledModel,
    subject: &'a Subject,
    dose_pk: impl Fn(usize) -> crate::types::PkParams,
) -> std::borrow::Cow<'a, [crate::types::DoseEvent]> {
    let bio = model.has_bioavailability() && subject.doses.iter().any(|d| d.is_infusion());
    if subject.all_doses_fixed() && !bio {
        return std::borrow::Cow::Borrowed(&subject.doses);
    }
    let attr_map = model.active_dose_attr_map();
    std::borrow::Cow::Owned(
        (0..subject.doses.len())
            .map(|k| {
                let pk = dose_pk(k);
                let d = subject.doses[k].resolve_rate(attr_map, &pk.values);
                // `F` applied once, up front, at the dose's own time, exactly as production's
                // `bioavailable_doses` does (#419): a rate-defined infusion keeps its rate and
                // its window stretches to `F·dur`; a duration-defined one keeps its window and
                // its rate scales. The walk then injects the rate as-is.
                if bio && d.is_infusion() {
                    d.with_bioavailable_infusion(pk.f_bio())
                } else {
                    d
                }
            })
            .collect(),
    )
}

/// Shared FD-fallback gate for modeled `RATE=-1/-2` doses on the analytic closed-form
/// walk (#486, deduped from the per-provider guards). Returns `false` (⇒ decline to
/// FD) when a modeled dose has no backing `D{cmt}`/`R{cmt}` slot — an invalid model
/// `check_model_data` rejects upstream, guarded here so the provider declines rather
/// than panics in `DoseEvent::resolve_rate`.
///
/// A modeled dose combined with a **steady-state** dose is now analytic too (#486):
/// `crate::sens::propagate::equilibrate_ss_g` threads the per-dose modeled-window dual
/// `(rate_bare, dur_bare)` into each equilibration cycle's active/quiet split, so the
/// moving infusion-end jet flows through the SS trough exactly as it does through the
/// current pulse (matching the ODE provider's `equilibrate_ss_state_g`).
pub(crate) fn modeled_dose_analytic_gate(model: &CompiledModel, subject: &Subject) -> bool {
    modeled_doses_resolvable(model, subject)
}

/// True when this subject combines a **steady-state infusion with a lagtime** on the event walk.
///
/// Production loads a lagged SS dose's periodic state at the dose **record** and lets the walk
/// carry it to the lagged arrival (#1121, `EventKind::DoseRecord` in
/// `pk::event_driven`). The dual walk now mirrors that seed for boluses. An infusion also has
/// a previous-cycle residual rate window after the record; until that forcing is dualized, it
/// remains a hard decline to FD rather than an approximation.
///
/// This gate is consulted from the four **walk** entry points only, so widening it leaves the
/// CF *static* superposition path alone — that path reaches SS × lagtime without passing here.
/// Do not read that as the static path being correct. `lagged_elapsed` still implements the
/// pre-#1121 semantics on both sides of the arrival: it *wraps* the pre-arrival phase where
/// production `crate::dosing::ss_seed_phase` clamps (wrapping was measured wrong against
/// NONMEM 7.6.0 — `nonmem_anchor/results/ss_lag_ge_ii`), and it always takes the collapsed
/// `C_ss(t − t_eff)` where production splits into tail + arrival once
/// `!crate::dosing::ss_arrival_is_trough`. Both divergences are confined to `ALAG > II`, where
/// `subject_sensitivities` returns `Some` on predictions measured up to ~19 % from
/// production's. Pre-existing, and untouched by #1311, which changes the walk alone; it needs
/// its own fix and its own anchor.
pub(crate) fn ss_lagtime_walk_unsupported(model: &CompiledModel, subject: &Subject) -> bool {
    model.has_lagtime() && subject.doses.iter().any(|d| d.ss && d.is_infusion())
}

/// Per-dose lagtime **values** for the event schedule (`τ_k = t_k + ALAG`), read from each
/// dose's own PK snapshot so a TV-covariate or `TIME`-dependent lag is honoured per dose —
/// production parity (`pk::compute_predictions_event_driven` builds `dose_lagtimes` the same
/// way, `p.lagtime()`, unclamped). All-zero when the model carries no lagtime, which is the
/// schedule's no-lag fast path (#486).
fn dose_lagtime_values(model: &CompiledModel, dose_pk: &[crate::types::PkParams]) -> Vec<f64> {
    if !model.has_lagtime() {
        return vec![0.0; dose_pk.len()];
    }
    dose_pk.iter().map(|p| p.lagtime()).collect()
}

/// Per-dose lagtime **duals** for the event walk (#486): `ALAG`'s `(θ, η)` jet at each dose's
/// snapshot, which turns the dose arrival into a moving boundary (see
/// `crate::sens::propagate::event_driven_sens_with_doses_g`). Empty when the model has no
/// lagtime — the walk then treats every arrival as a fixed break, exactly as before.
///
/// `slot_dual(dose_k, pk_slot)` is the same seeder `modeled_dose_inf_duals` takes, so both
/// read the identical per-dose snapshot.
fn dose_lag_duals<T: crate::sens::num::PkNum>(
    model: &CompiledModel,
    subject: &Subject,
    slot_dual: impl Fn(usize, usize) -> T,
) -> Vec<T> {
    if !model.has_lagtime() {
        return Vec::new();
    }
    (0..subject.doses.len())
        .map(|k| slot_dual(k, PK_IDX_LAGTIME))
        .collect()
}

/// True when the walk's **moving** breaks are separable, i.e. representable by the single-`dt`
/// dual walk. A break moves when the model has an estimated lagtime (every dose arrival, and
/// every infusion end riding it) or when a dose is a modeled `RATE=-1/-2` infusion (its end).
///
/// The walk resolves a break time to **one** dual position (`dual_pos`, first match within 1e-9).
/// So the subject declines to FD when either
///
/// 1. **two distinct moving breaks coincide.** They generally carry *different* jets, and the walk
///    would silently hand the first one's jet to both.
///
///    This is deliberately conservative — it declines even breaks that happen to share a `D`/`R`
///    slot. It has to be: since #486 each dose's lagtime and window are seeded from **that dose's
///    own snapshot** (its occasion under IOV, its covariates under TV-cov, its event time under
///    `TIME`), so two breaks backed by the same PK slot no longer imply the same dual. Keying the
///    check on the slot — as it used to — would admit, say, a κ-on-`ALAG` subject whose occasion-2
///    arrival lands on occasion-1's infusion end, and differentiate one with respect to the other's
///    jet. The duals themselves are not comparable here (this runs on `f64` times, before any is
///    built), so coincidence alone is the decline.
///
/// 2. a moving break coincides with a **fixed** break — a reset, a covariate-change (`PkOnly`), or,
///    when the model has no lagtime, a **dose event time** (with a lagtime every arrival is itself a
///    moving break, so those are covered by (1) instead). Those times do not move with `(θ, η)`, but
///    the walk resolves one dual position per break, so it would drag the fixed event along with the
///    moving one.
///
///    **Observations are exempt**: the walk *corrects* them instead of declining, stepping the
///    read-out state back across the sliver between the sample and the boundary
///    (`crate::sens::propagate::event_driven_sens_with_doses_g`). An observation landing on *two*
///    moving breaks is still declined — by (1), since those two breaks coincide with each other.
///
/// Both are degenerate, measure-zero coincidences. `eff` are the resolved doses the walk sees.
fn moving_bounds_separable(
    model: &CompiledModel,
    subject: &Subject,
    eff: &[crate::types::DoseEvent],
    dose_lagtimes: &[f64],
) -> bool {
    let attr_map = model.active_dose_attr_map();
    let has_lag = model.has_lagtime();
    let mut moving: Vec<f64> = Vec::new();
    for (k, d) in subject.doses.iter().enumerate() {
        // The walk's break times are the *lagged* arrivals, so compare on those.
        let t_start = eff[k].time + dose_lagtimes.get(k).copied().unwrap_or(0.0);
        // Dose arrival: moves only under a lagtime.
        if has_lag {
            moving.push(t_start);
        }
        // Infusion end: moves under a lagtime (riding the arrival) and/or when modeled.
        if eff[k].rate > 0.0 && eff[k].duration > 0.0 {
            let modeled = crate::sens::ode_provider::modeled_slot_for(attr_map, d).is_some();
            if modeled || has_lag {
                moving.push(t_start + eff[k].duration);
            }
        }
    }
    if moving.is_empty() {
        return true;
    }
    // (1) Two distinct moving breaks may not coincide.
    for (i, &t) in moving.iter().enumerate() {
        if moving[..i].iter().any(|&u| (u - t).abs() < 1e-9) {
            return false;
        }
    }
    // (2) A moving break may not land on a fixed one. Observations are corrected, not declined.
    let mut fixed: Vec<f64> = subject
        .pk_only_times
        .iter()
        .chain(subject.reset_times.iter())
        .copied()
        .collect();
    if !has_lag {
        // Without a lagtime a dose lands at its record time, which is a fixed break the walk must
        // not drag along with someone else's moving infusion end.
        fixed.extend(eff.iter().map(|d| d.time));
    }
    for f in fixed {
        if moving.iter().any(|&t| (t - f).abs() < 1e-9) {
            return false;
        }
    }
    true
}

/// Per-dose infusion duals `(rate, window)` for the event walk
/// (`crate::sens::propagate::event_driven_sens_with_doses_g`, #486): `Some` for a
/// modeled `RATE=-1/-2` dose, and under bioavailability for every infusion; `None`
/// otherwise (an empty vector when no dose needs one). The rate's jet feeds the injected
/// magnitude and the window's makes the infusion **end** a moving boundary. Under
/// bioavailability the dose-time `F` is applied exactly as `resolve_eff_doses` applies
/// it to the f64 window (#419): a rate-defined window stretches to `F·dur`, a
/// duration-defined rate scales by `F`; the walk injects the rate as-is. `slot_dual(dose_k, pk_slot)` supplies the seeded dual of the `D`/`R` PK
/// slot at dose `k`'s snapshot (`Dual2<M>` outer, `Dual1<N>` inner). A transient
/// non-positive `D`/`R` is clamped to its floor, mirroring the f64
/// `DoseEvent::resolve_rate`.
fn modeled_dose_inf_duals<T: crate::sens::num::PkNum>(
    model: &CompiledModel,
    subject: &Subject,
    slot_dual: impl Fn(usize, usize) -> T,
) -> Vec<Option<(T, T)>> {
    let bio = model.has_bioavailability() && subject.doses.iter().any(|d| d.is_infusion());
    if subject.all_doses_fixed() && !bio {
        return Vec::new();
    }
    let attr_map = model.active_dose_attr_map();
    subject
        .doses
        .iter()
        .enumerate()
        .map(|(k, d)| {
            let amt = T::from_f64(d.amt);
            // The infusion's bare `(rate, window)`: from its modeled `D`/`R` slot, or the
            // record's own values.
            let (rate, dur) = match crate::sens::ode_provider::modeled_slot_for(attr_map, d) {
                Some((crate::types::RateMode::ModeledDuration, slot)) => {
                    let dur = clamp_dual_floor(
                        slot_dual(k, slot),
                        crate::types::DoseEvent::DURATION_FLOOR,
                    );
                    (amt / dur, dur)
                }
                Some((crate::types::RateMode::ModeledRate, slot)) => {
                    let rate =
                        clamp_dual_floor(slot_dual(k, slot), crate::types::DoseEvent::RATE_FLOOR);
                    (rate, amt / rate)
                }
                _ if bio && d.is_infusion() => (T::from_f64(d.rate), T::from_f64(d.duration)),
                _ => return None,
            };
            if !bio {
                return Some((rate, dur));
            }
            // `F` at the dose's time, applied as `resolve_eff_doses` applies it (#419): it
            // stretches a rate-defined window — a boundary moving with `F` — or scales a
            // duration-defined rate.
            let f = slot_dual(k, crate::types::PK_IDX_F);
            Some(match d.infusion_def {
                crate::types::InfusionDef::RateDefined => (rate, f * dur),
                crate::types::InfusionDef::DurationDefined => (f * rate, dur),
            })
        })
        .collect()
}

/// This thread's last walk schedule and the exact inputs it was built from.
struct ScheduleCacheEntry {
    key: Vec<u64>,
    schedule: std::rc::Rc<crate::pk::event_driven::EventSchedule>,
}

thread_local! {
    static SCHEDULE_CACHE: std::cell::RefCell<(Vec<u64>, Option<ScheduleCacheEntry>)> =
        const { std::cell::RefCell::new((Vec::new(), None)) };
}

/// `EventSchedule::for_subject(subject, pk_model, doses, lagtimes)`, reused while every
/// input it reads is bit-identical to the last build on this thread. The walk is recorded
/// many times per subject at the same doses and lags — each inner-loop iteration, the
/// outer gradient's passes, the covariance step — so rebuilding it each time is pure
/// allocation. The key is everything the build reads (the PK model, each effective dose's
/// timing and infusion fields, the lag times, and the observation, pk-only and reset
/// times), so a hit is exactly a rebuild's result; a mutated subject or a moved lag misses.
fn cached_event_schedule(
    subject: &Subject,
    pk_model: PkModel,
    doses: &[DoseEvent],
    lagtimes: &[f64],
) -> std::rc::Rc<crate::pk::event_driven::EventSchedule> {
    SCHEDULE_CACHE.with(|cell| {
        let (scratch, entry) = &mut *cell.borrow_mut();
        scratch.clear();
        scratch.push(pk_model as u64);
        scratch.push(doses.len() as u64);
        for d in doses {
            // Exactly the dose fields the schedule reads (its event times, and the
            // infusion rates of `active_rates_at` / `ss_residual_infusion_end`); the
            // amount is not one of them — the walk reads it from `doses` directly. `rate`
            // is kept although no edit isolates it: a real rate change moves `duration`
            // too, and the walk revalidates the cached rates itself (`rates_valid_for`).
            // A cache key covers the full read set rather than the minimal one.
            scratch.extend([
                d.time.to_bits(),
                d.cmt_raw() as u64,
                d.rate.to_bits(),
                d.duration.to_bits(),
                u64::from(d.ss),
                d.ii.to_bits(),
                d.infusion_def as u64,
            ]);
        }
        for times in [
            lagtimes,
            subject.obs_times.as_slice(),
            subject.pk_only_times.as_slice(),
            subject.reset_times.as_slice(),
        ] {
            scratch.push(times.len() as u64);
            scratch.extend(times.iter().map(|t| t.to_bits()));
        }
        if let Some(e) = entry.as_ref().filter(|e| e.key == *scratch) {
            return e.schedule.clone();
        }
        let schedule = std::rc::Rc::new(crate::pk::event_driven::EventSchedule::for_subject(
            subject, pk_model, doses, lagtimes,
        ));
        *entry = Some(ScheduleCacheEntry {
            key: scratch.clone(),
            schedule: schedule.clone(),
        });
        schedule
    })
}

/// The closed-form event walk with its output readout, over any number type: modeled
/// doses resolved to their windows, lag-shifted arrivals as moving boundaries, the walk
/// itself (`closed_form_sens_walk_g`), the production clamp of the concentration at zero,
/// then the analytic Form C readout (#650) at each observation. Shared by the forward
/// TV-cov provider (`Dual2`) and the adjoint gradient (`DualV`), so neither carries its
/// own copy of the event semantics.
///
/// `dose_pk` are the per-dose f64 PK params (window resolution and lag values),
/// `pk_at_*` the per-event params as `T`, `slot_dual(k, slot)` dose `k`'s PK slot as `T`
/// (modeled `D`/`R`, `ALAG`), and `obs_params(j)` the flat PK-slot vector the readout
/// reads at observation `j`. `None` where the walk cannot serve the subject (coincident
/// moving breaks, or an absorption walk the closed form declines).
#[allow(clippy::too_many_arguments)]
pub(crate) fn closed_form_event_predictions<T: crate::sens::num::vm::VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    dose_pk: &[crate::types::PkParams],
    pk_at_dose: &[crate::sens::propagate::PkDual<T>],
    pk_at_obs: &[crate::sens::propagate::PkDual<T>],
    pk_at_pk_only: &[crate::sens::propagate::PkDual<T>],
    slot_dual: &dyn Fn(usize, usize) -> T,
    obs_params: &dyn Fn(usize) -> [T; N_PK],
    readout: Option<&crate::parser::model_parser::OdeOutputProgram>,
) -> Option<Vec<T>> {
    // Modeled-`RATE=-1/-2` doses (#486): resolve to concrete rate/duration for the
    // schedule's break times, and build the per-dose duals so the injected rate and
    // the moving infusion-end boundary carry their `(θ,η)` jets. The per-dose f64 PK
    // params are the caller's hoisted `pk_dose` (one `pk_param_fn` per dose per subject,
    // shared with the dual's value and every chunk). Fixed subjects borrow
    // `subject.doses` and pass no duals (identical to the pre-#486 path).
    let eff_doses = resolve_eff_doses(model, subject, |k| dose_pk[k]);
    // Lagtime on the event walk (#486): the schedule shifts each arrival to `t + ALAG`
    // (production parity) and the walk threads `∂ALAG` through the dual sub-interval
    // lengths, so the arrival is a moving boundary like a modeled infusion end.
    let dose_lagtimes = dose_lagtime_values(model, dose_pk);
    // Coincident moving breaks carrying different jets — and, under a lagtime, a moving
    // break landing on a fixed obs/reset/cov-change time — aren't representable by the
    // single-`dt` walk; decline to FD (#486 review #2).
    if !moving_bounds_separable(model, subject, &eff_doses, &dose_lagtimes) {
        return None;
    }
    let schedule = cached_event_schedule(subject, model.pk_model, &eff_doses, &dose_lagtimes);
    let dose_inf_dual = modeled_dose_inf_duals::<T>(model, subject, slot_dual);
    let dose_lag_dual = dose_lag_duals::<T>(model, subject, slot_dual);
    let conc = crate::pk::absorption_walk::closed_form_sens_walk_g::<T>(
        model.pk_model,
        subject,
        &schedule,
        &eff_doses,
        &dose_lag_dual,
        &dose_inf_dual,
        pk_at_dose,
        pk_at_obs,
        pk_at_pk_only,
        slot_dual,
    )?;

    // Analytic Form C readout (#650): scratch for the per-observation dual eval.
    let mut ro_state: Vec<T> = Vec::new();
    let mut ro_vars: Vec<T> = Vec::new();
    let mut ro_stack: Vec<T> = Vec::new();
    Some(
        conc.iter()
            .enumerate()
            .map(|(j, c)| {
                // Clamp the concentration jet to ≥ 0 BEFORE any readout — production clamps
                // `conc.max(0.0)` and only then applies the readout (returned unclamped).
                let c = if c.val() < 0.0 { T::from_f64(0.0) } else { *c };
                let Some(ro) = readout else {
                    return c;
                };
                // A `TIME`-referencing readout resolves `Op::PushTime` from the model-time
                // thread-local; enter this observation's time to match production's
                // `apply_analytic_readout` guard (no-op for a `TIME`-free readout).
                let _time_guard =
                    crate::parser::model_parser::ModelTimeGuard::enter(subject.readout_time(j));
                eval_readout_jet::<T>(
                    ro,
                    c,
                    &obs_params(j),
                    subject.obs_cov(j),
                    &mut ro_state,
                    &mut ro_vars,
                    &mut ro_stack,
                )
            })
            .collect(),
    )
}

/// Per observation the prediction and its exact `∂f/∂η` at `(θ, η)` (the stacked `[η, κ]`
/// under IOV), from one tape-free recording of the unified recorder
/// (`sens::ode_provider::adjoint`) — closed form, event walks, user ODEs, compartment-free
/// models alike — or `None` for a subject no recorder serves. FREM pseudo-observation rows
/// carry their exact `θ_i + η_e`.
pub fn subject_eta_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<ObsGrad>> {
    crate::sens::ode_provider::adjoint::exact_obs_grads(model, subject, theta, eta)
}

/// `subject_eta_grad`'s `∂f/∂η` as a row-major `n_obs × n_eta` matrix.
pub fn subject_eta_jacobian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<f64>> {
    Some(
        subject_eta_grad(model, subject, theta, eta)?
            .into_iter()
            .flat_map(|o| o.df_deta)
            .collect(),
    )
}

/// Value and first-order η-gradient of one observation (`subject_eta_grad`).
#[derive(Debug, Clone)]
pub struct ObsGrad {
    pub f: f64,
    /// `∂f/∂η_k`, length `n_eta`.
    pub df_deta: Vec<f64>,
}

/// The closed-form `init(...)` impulses over `T` (`pk::add_analytical_init_with`): each
/// `A₀ = amount(θ, η, p)` from its compiled program, decayed by the closed-form kernel at
/// every observation before the first reset, added to `preds` — before any scaling, as
/// production adds them. `amount` is the PK-slot vector the amounts read; `decay[j]` the
/// one observation `j` decays with (`None`: `amount`'s, the non-IOV convention). `None`
/// when an init has no compiled amount program.
pub(crate) fn add_closed_form_init_g<T: crate::sens::num::vm::VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[T],
    eta: &[T],
    amount: &[T],
    decay: Option<&[Vec<T>]>,
    preds: &mut [T],
) -> Option<()> {
    let first_reset = subject
        .reset_times
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let slot = |p: &[T], s: usize| p.get(s).copied().unwrap_or(T::from_f64(0.0));
    for init in &model.analytical_init {
        let prog = init.amount_deriv.as_ref()?;
        let vars: Vec<T> = prog
            .var_to_pk_slot()
            .iter()
            .map(|&s| slot(amount, s))
            .collect();
        let a0 = prog.eval_scale_g(theta, eta, &subject.covariates, &vars);
        if a0.val() == 0.0 || !a0.val().is_finite() {
            continue;
        }
        for (j, &t) in subject.obs_times.iter().enumerate() {
            if t < 0.0 || t >= first_reset {
                continue;
            }
            let p: &[T] = decay
                .and_then(|d| d.get(j))
                .map_or(amount, |v| v.as_slice());
            let c = crate::pk::analytical_init_concentration_g(
                model.pk_model,
                init.cmt,
                a0,
                T::from_f64(t),
                slot(p, PK_IDX_CL),
                slot(p, PK_IDX_V),
                slot(p, PK_IDX_Q),
                slot(p, PK_IDX_V2),
                slot(p, PK_IDX_KA),
                slot(p, PK_IDX_Q3),
                slot(p, PK_IDX_V3),
            );
            preds[j] = preds[j] + c;
        }
    }
    Some(())
}

/// The closed-form engines' model-level scope, as the adjoint gradient's recorders see
/// it: a closed form the superposition and event walk implement, no ODE, and a Form C
/// readout the shared `eval_readout_jet` can evaluate.
/// Unlike `analytical_supported` it carries none of the forward provider's
/// dispatch-width or slot limits.
pub(crate) fn closed_form_model_scope(model: &CompiledModel) -> bool {
    !model.is_algebraic()
        && model.ode_spec.is_none()
        && matches!(
            model.pk_model,
            PkModel::OneCptIv
                | PkModel::OneCptOral
                | PkModel::OneCptTransit
                | PkModel::OneCptIg
                | PkModel::TwoCptIv
                | PkModel::TwoCptOral
                | PkModel::TwoCptTransit
                | PkModel::TwoCptIg
                | PkModel::ThreeCptIv
                | PkModel::ThreeCptOral
        )
        && analytic_readout_dual_supported(model)
}

/// The closed-form static-superposition predictions at every observation over a
/// caller-seeded numeric type: `slot_dual(slot)` gives each PK slot's dual (or `None` for
/// a constant). The same kernels, clamp, readout, scalar scale and LTBS as the η-gradient
/// path `run_obs_grad` and the f64 predictor, with no `dp_deta` chain. The adjoint
/// gradient (`sens::ode_provider::adjoint`) seeds the slots with `DualV` η-tangents.
///
/// `None` outside that path's scope (event-driven walk, algebraic or ODE models,
/// resets with SS, unresolved modeled doses, rate-defined infusions under F, analytic
/// `init(...)` impulses). An expression `obs_scale` divisor is not applied here (the
/// adjoint divides by it as its own leaf). FREM pseudo-observation rows are
/// predicted here like any row; the adjoint's likelihood kernel replaces them.
pub(crate) fn analytic_static_predictions<T: crate::sens::num::vm::VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    pk: &crate::types::PkParams,
    slot_dual: impl Fn(usize) -> Option<T>,
    theta: &[T],
    eta: &[T],
) -> Option<Vec<T>> {
    if model.is_algebraic()
        || model.ode_spec.is_some()
        || subject_routes_to_event_walk(model, subject)
    {
        return None;
    }
    // The superposition kernels take each dose's f64 window, so an `F` that stretches a
    // rate-defined one (#419) cannot move it: the caller routes those to the event walk.
    if !closed_form_model_scope(model)
        || model.n_kappa != 0
        || (subject.has_resets() && subject.doses.iter().any(|d| d.ss))
        || !subject.all_doses_fixed()
        || (model.has_bioavailability() && subject.has_rate_defined_infusion())
    {
        return None;
    }
    let flags = PkClassFlags {
        oral: matches!(
            model.pk_model,
            PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral
        ),
        two_cpt: matches!(model.pk_model, PkModel::TwoCptIv | PkModel::TwoCptOral),
        three_cpt: matches!(model.pk_model, PkModel::ThreeCptIv | PkModel::ThreeCptOral),
        transit: matches!(model.pk_model, PkModel::OneCptTransit),
        two_cpt_transit: matches!(model.pk_model, PkModel::TwoCptTransit),
        ig: matches!(model.pk_model, PkModel::OneCptIg),
        two_cpt_ig: matches!(model.pk_model, PkModel::TwoCptIg),
    };
    let dual_of = |slot: usize, value: f64| slot_dual(slot).unwrap_or_else(|| T::from_f64(value));
    let seeded = SeededPkDuals::<T>::seed_with(pk, &dual_of).with_disposition(flags);
    let readout = model
        .analytic_readout
        .as_ref()
        .and_then(|ar| ar.program.as_ref());
    let ro_pk: Option<[T; N_PK]> = readout
        .map(|_| std::array::from_fn(|s| dual_of(s, pk.values.get(s).copied().unwrap_or(0.0))));
    let mut ro_state: Vec<T> = Vec::new();
    let mut ro_vars: Vec<T> = Vec::new();
    let mut ro_stack: Vec<T> = Vec::new();
    // The clamped concentrations, then the analytic `init(...)` impulses on top (production
    // adds them before the readout and scaling; the two never coexist on the closed form).
    let mut conc: Vec<T> = subject
        .obs_times
        .iter()
        .map(|&t_obs| {
            let fd = superpose_doses(
                &seeded,
                flags,
                subject,
                t_obs,
                reset_floor_at(subject, t_obs),
            );
            // Mirror production's `conc.max(0.0)`.
            if fd.val() < 0.0 {
                T::from_f64(0.0)
            } else {
                fd
            }
        })
        .collect();
    if !model.analytical_init.is_empty() {
        let amount: Vec<T> = (0..pk.values.len())
            .map(|s| dual_of(s, pk.values[s]))
            .collect();
        add_closed_form_init_g(model, subject, theta, eta, &amount, None, &mut conc)?;
    }
    let mut out = Vec::with_capacity(subject.obs_times.len());
    for (obs_i, conc) in conc.into_iter().enumerate() {
        let y = apply_readout_jet(
            conc,
            readout,
            ro_pk.as_ref(),
            subject,
            obs_i,
            &mut ro_state,
            &mut ro_vars,
            &mut ro_stack,
        );
        out.push(y);
    }
    Some(out)
}

/// The analytic individual-parameter program, when it covers every slot the closed form
/// reads (the same gate the η-gradient path applies).
pub(crate) fn analytic_param_program(
    model: &CompiledModel,
) -> Option<&crate::parser::model_parser::IndivParamProgram> {
    model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .filter(|prog| prog_covers_required_pk_slots(model, prog))
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
    // Only models the event-driven walk can state-propagate take that route — mirror
    // production's `compute_predictions_with_tv`, which gates the event-driven predictor on
    // `supports_event_driven` and otherwise falls to the t=0 static superposition.
    // `one_cpt_transit` is analytical (core-supported) but NOT walk-supported: its
    // continuous-`N` Gamma absorption is a closed-form convolution, not a finite linear
    // state (`propagate::walk_supports` / `event_driven::supports_event_driven` both omit
    // it). Routing a transit subject here would hit the walk's unsupported-model
    // early-return, which yields all-ZERO predictions and sensitivities (a zero `Vec`, not
    // an error). Declining here keeps transit on the dose-superposition analytic path
    // (`run_obs`), whose t=0 snapshot reproduces production's transit prediction exactly —
    // so a transit subject with time-varying covariates / a `TIME` switch stays analytic
    // (matching production), rather than silently zeroing or dropping to FD.
    //
    // Since #1560 a transit/IG subject whose closed form cannot serve it only because of
    // time-varying covariates or IOV *does* take the walk — its own absorption walk, which
    // `absorption_walk::closed_form_sens_walk_g` dispatches to. The same structural test
    // the value path applies; a subject whose pairs leave the tilting domain never gets
    // here, since `effective_model_for_eval` has already sent it to the twin.
    if crate::pk::absorption_walk::walk_eligible(model, subject) {
        return crate::pk::subject_feeds_analytical_pk(model, subject);
    }
    if !crate::pk::event_driven::supports_event_driven(model.pk_model) {
        return false;
    }
    // #905: mirror the value path's declination. A subject with no Gaussian
    // observation on an analytical model feeds the predictor nothing, so
    // `compute_predictions_with_tv_*` returns NaN without walking. The gradient must
    // follow — otherwise FOCE/FOCEI would enter the event-driven walk (and its `sens`
    // twin, `propagate::active_rates_g` and the bolus `cmt_idx` guard, both of which
    // `panic!` on the very unroutable dose the value path just skipped). The Gaussian
    // gradient of a subject with no Gaussian obs is an empty sum, so declining the
    // walk loses nothing. Keyed on the primary `ode_spec`, matching the value gate.
    if model.ode_spec.is_none() && !crate::pk::subject_feeds_analytical_pk(model, subject) {
        return false;
    }
    subject.has_tv_covariates()
        || subject_has_oral_infusion(model, subject)
        // A dose the superposition dispatch cannot place — any bolus outside
        // compartment 1, any infusion outside central (#375). Production reroutes
        // exactly these subjects to the walk (`pk::compute_predictions*`), so the
        // gradient must follow or FOCE/FOCEI would differentiate a different
        // dosing history than it predicted. Same single source of truth.
        || crate::pk::dose_needs_event_walk(model.pk_model, subject)
        || crate::parser::model_parser::compiled_model_uses_time_builtin(model)
        // A modeled-`RATE=-1/-2` dose (`R{cmt}`/`D{cmt}`) resolves its rate/window
        // from PK params, so the infusion *end* is a moving boundary — the
        // event-driven walk carries it via the exact dual window length, which
        // dose superposition can't (#486). This is the closed-form twin of the ODE
        // `ode_tvcov_supported`'s `has_modeled_dose` clause. An oral modeled infusion
        // is already covered by `subject_has_oral_infusion`; this adds the IV route.
        || !subject.all_doses_fixed()
        // **SS + EVID 3/4 reset** (#486). Dose *superposition* genuinely cannot express this
        // (an SS dose folds in an infinite periodic history that a mid-record reset truncates),
        // which is why the static path declines it — but the *walk* can: it equilibrates the SS
        // dose per event (`equilibrate_ss_g`) and zeros the state at each reset, and production
        // already computes exactly these subjects with the f64 instance of this same walk
        // (`pk::compute_predictions*` route on `subject.has_resets()`). So route them here and
        // let the dual walk mirror production, instead of falling to FD.
        || (subject.has_resets() && subject.doses.iter().any(|d| d.ss))
}
