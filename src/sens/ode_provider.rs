//! Per-observation analytic sensitivities for **user-specified `[odes]` models**
//! (issue #367, Option A). The closed-form provider ([`super::provider`]) covers
//! the analytical 1-/2-/3-cpt PK models; this is its ODE counterpart.
//!
//! The state is integrated as [`Dual2<N>`](super::dual2::Dual2) seeded on the
//! `N` individual parameters: the compiled RHS program
//! ([`OdeRhsProgram`](crate::parser::model_parser::OdeRhsProgram)) is evaluated
//! over the dual numbers by the generic bytecode VM, and the generic RK45
//! ([`solve_ode_g`](crate::ode::solver::solve_ode_g)) propagates `∂u/∂p` and
//! `∂²u/∂p²` through the integration with **value-based step control**. The
//! readout then yields `∂f/∂p, ∂²f/∂p²` per observation, which feed the η/θ chain
//! via the **general** individual-parameter derivatives `∂p/∂η, ∂p/∂θ` (FD of
//! `pk_param_fn` — see [`param_derivatives`]; no log-normal assumption).
//!
//! **Supported:** single-endpoint `ObsCmt`, uniform Form C (`y = central/V1`), or
//! per-CMT Form C (`y[CMT=N] = <expr>` — each endpoint differentiated over the dual,
//! #439) readout, including **covariate references** in the Form C expression
//! (e.g. a free→total protein-binding readout that branches on a `FREE` flag, #540)
//! — covariates thread in as constants from the per-observation snapshot; **bolus
//! and infusion** doses; **bioavailability F** (incl.
//! estimated, any parameterization — log-normal, logit-normal, additive — and the
//! compartment-indexed `F{cmt}` form, #486); **EVID 3/4 resets / multi-occasion**;
//! **non-zero `init(...)` initial conditions**; static covariates; a constant
//! `obs_scale` divisor, an **η-dependent expression `obs_scale`** divisor (`obs_scale =
//! expr(θ,η)`, applied as the subject-static quotient on the static walk, #486), and
//! **LTBS** (`log(DV) ~ …`) output transforms; all five built-in input-rate
//! forcings (igd/transit/weibull/first_order/zero_order, #430/#468/#530);
//! **estimated lagtime** (incl. compartment-indexed `ALAG{cmt}`) for every forcing
//! except `weibull()`; up to [`MAX_ODE_SENS_DIM`] individual parameters. Both the full
//! `Dual2` **outer** gradient and a light `Dual1` **inner** η-gradient
//! ([`ode_subject_eta_grad`]) are served (#410). On the event-driven walk these compose
//! with **time-varying covariates**, **steady-state dosing** (dual SS-equilibration), and
//! **θ/η referenced *directly* in a Form C readout** (auto-desugared into synthetic
//! individual parameters, #486); **IOV** (per-occasion κ) is served by the parallel IOV
//! provider (#439/#486).
//!
//! **Not yet supported** (falls back to the gradient-free / FD path): SDE/diffusion,
//! `weibull()` **combined with an estimated lagtime** (its `β < 1` onset has no
//! closed-form rate-on saltation), an **expression `obs_scale` combined with LTBS**, and
//! a few narrow compositions — **steady-state combined with a time-dependent
//! (`TIME`/`TAD`) RHS**, and **IOV combined with FREM, LTBS, or a steady-state
//! input-rate forcing**.
#![allow(clippy::needless_range_loop)]

use super::dual1::Dual1;
use super::dual2::Dual2;
use super::dual_mixed::DualMixed;
use super::provider::{ObsGrad, ObsSens, SubjectSens};
use crate::ode::predictions::{input_rate_consumes_cmt, OdeReadout, OdeSpec};
use crate::ode::solver::solve_ode_g;
use crate::pk::absorption::PreparedInputRate;
use crate::types::{CompiledModel, ScalingSpec, Subject, PK_IDX_F, PK_IDX_LAGTIME};
use std::cell::RefCell;

/// Largest individual-parameter count for which the `Dual2<N>` path is
/// monomorphised; models wider than this fall back to the gradient-free path.
const MAX_ODE_SENS_DIM: usize = 12;

// The `pk_indices.len()` dispatch tables in `ode_subject_sensitivities` and
// `ode_subject_eta_grad` enumerate `1..=12` explicitly with a silent `_ => None`
// fallback. Keep that table in lockstep with `MAX_ODE_SENS_DIM`: bumping the const
// without extending both `dispatch!` arms would let an in-scope wider model pass
// the gate, hit `_ => None`, and silently fall back to FD with no error. This
// compile-time tripwire forces an edit here — and a look at the tables — before the
// const can change (#438 review).
const _: () = assert!(
    MAX_ODE_SENS_DIM == 12,
    "MAX_ODE_SENS_DIM changed: extend the pk_indices.len() dispatch tables in \
     ode_subject_sensitivities and ode_subject_eta_grad to match, then update this assert"
);

/// Largest (θ + η) axis count for which the analytical η/θ chain (the
/// individual-parameter program over `Dual2<M>`) is monomorphised.
pub(crate) const MAX_ODE_AXES: usize = 16;

/// Largest stacked `(θ, η_bsv, κ_1..κ_K)` axis count for ODE IOV subjects.
/// Kept separate from [`MAX_ODE_AXES`] because high-occasion IOV subjects are wide
/// only in the kappa stack; the per-event individual-parameter derivative program
/// still runs over `(θ, η_bsv, κ_current)` and stays under the normal ODE axis cap.
pub(crate) const MAX_ODE_IOV_AXES: usize = 96;

// Four `disp!`/`dispatch_tv!(1, 2, …, 16)` dispatch tables are keyed on `MAX_ODE_AXES` and
// enumerate `1..=16` explicitly with a silent `_ => None` — they live in the **entry-point
// callers** (the `run_subject_*<const M>` workers are const-generic and carry no table):
//   1. `ode_subject_sensitivities`     (TV-cov outer, `dispatch_tv!`)
//   2. `ode_subject_eta_grad`          (TV-cov inner, `dispatch_tv!`)
//   3. `param_eta_derivatives`         (`disp!`)
//   4. `param_derivatives_at_cov`      (`disp!`)
// Keep all four in lockstep with the const: bumping `MAX_ODE_AXES` without widening every
// arm would let an in-scope wider model pass the gate, hit `_ => None`, and silently fall
// back to FD with no error. This compile-time tripwire forces an edit here — and a look at
// all four tables — before the const can change (#438 / #466 review round 1 #13 + round 2).
const _: () = assert!(
    MAX_ODE_AXES == 16,
    "MAX_ODE_AXES changed: widen the disp!(1..=16) / dispatch_tv!(1..=16) tables in \
     ode_subject_sensitivities, ode_subject_eta_grad, param_eta_derivatives, \
     and param_derivatives_at_cov to match, then update this assert"
);

// An η-dependent `ExpressionScale` admitted by `ode_analytical_supported` (bounded by
// `MAX_ODE_AXES` above) has its quotient applied *post-walk* through
// `provider::apply_expression_scale_outer` / `_inner_dispatch`, whose `dispatch_init_impulse!`
// tables are bounded by `MAX_SCALE_AXES` with a **silent `_ => {}`** (no-op, not `None`/FD).
// The static walk itself dispatches on the PK-param count (`n_indiv ≤ 12`), independent of
// `n_axes = n_theta + n_eta`, so a many-θ model can build the walk while `n_axes` exceeds the
// scale table — and the scale would be silently dropped, yielding an *unscaled* analytic
// gradient rather than an FD fallback. Couple the two caps so widening the ODE axis cap
// without widening the scale dispatch fails to compile (#534 adversarial audit).
const _: () = assert!(
    MAX_ODE_AXES <= crate::sens::provider::MAX_SCALE_AXES,
    "MAX_ODE_AXES exceeds MAX_SCALE_AXES: an ODE ExpressionScale model with n_axes in \
     (MAX_SCALE_AXES, MAX_ODE_AXES] passes ode_analytical_supported but hits the silent `_` \
     arm of dispatch_init_impulse! and silently drops the obs_scale quotient. Widen \
     MAX_SCALE_AXES (and its dispatch_init_impulse! table) to at least MAX_ODE_AXES."
);

const _: () = assert!(
    MAX_ODE_IOV_AXES == 96,
    "MAX_ODE_IOV_AXES changed: update dispatch_ode_iov_axes! to enumerate the same range"
);

macro_rules! dispatch_ode_iov_axes {
    ($dim:expr, $worker:ident, $($arg:expr),+ $(,)?) => {
        match $dim {
            1 => $worker::<1>($($arg),+),
            2 => $worker::<2>($($arg),+),
            3 => $worker::<3>($($arg),+),
            4 => $worker::<4>($($arg),+),
            5 => $worker::<5>($($arg),+),
            6 => $worker::<6>($($arg),+),
            7 => $worker::<7>($($arg),+),
            8 => $worker::<8>($($arg),+),
            9 => $worker::<9>($($arg),+),
            10 => $worker::<10>($($arg),+),
            11 => $worker::<11>($($arg),+),
            12 => $worker::<12>($($arg),+),
            13 => $worker::<13>($($arg),+),
            14 => $worker::<14>($($arg),+),
            15 => $worker::<15>($($arg),+),
            16 => $worker::<16>($($arg),+),
            17 => $worker::<17>($($arg),+),
            18 => $worker::<18>($($arg),+),
            19 => $worker::<19>($($arg),+),
            20 => $worker::<20>($($arg),+),
            21 => $worker::<21>($($arg),+),
            22 => $worker::<22>($($arg),+),
            23 => $worker::<23>($($arg),+),
            24 => $worker::<24>($($arg),+),
            25 => $worker::<25>($($arg),+),
            26 => $worker::<26>($($arg),+),
            27 => $worker::<27>($($arg),+),
            28 => $worker::<28>($($arg),+),
            29 => $worker::<29>($($arg),+),
            30 => $worker::<30>($($arg),+),
            31 => $worker::<31>($($arg),+),
            32 => $worker::<32>($($arg),+),
            33 => $worker::<33>($($arg),+),
            34 => $worker::<34>($($arg),+),
            35 => $worker::<35>($($arg),+),
            36 => $worker::<36>($($arg),+),
            37 => $worker::<37>($($arg),+),
            38 => $worker::<38>($($arg),+),
            39 => $worker::<39>($($arg),+),
            40 => $worker::<40>($($arg),+),
            41 => $worker::<41>($($arg),+),
            42 => $worker::<42>($($arg),+),
            43 => $worker::<43>($($arg),+),
            44 => $worker::<44>($($arg),+),
            45 => $worker::<45>($($arg),+),
            46 => $worker::<46>($($arg),+),
            47 => $worker::<47>($($arg),+),
            48 => $worker::<48>($($arg),+),
            49 => $worker::<49>($($arg),+),
            50 => $worker::<50>($($arg),+),
            51 => $worker::<51>($($arg),+),
            52 => $worker::<52>($($arg),+),
            53 => $worker::<53>($($arg),+),
            54 => $worker::<54>($($arg),+),
            55 => $worker::<55>($($arg),+),
            56 => $worker::<56>($($arg),+),
            57 => $worker::<57>($($arg),+),
            58 => $worker::<58>($($arg),+),
            59 => $worker::<59>($($arg),+),
            60 => $worker::<60>($($arg),+),
            61 => $worker::<61>($($arg),+),
            62 => $worker::<62>($($arg),+),
            63 => $worker::<63>($($arg),+),
            64 => $worker::<64>($($arg),+),
            65 => $worker::<65>($($arg),+),
            66 => $worker::<66>($($arg),+),
            67 => $worker::<67>($($arg),+),
            68 => $worker::<68>($($arg),+),
            69 => $worker::<69>($($arg),+),
            70 => $worker::<70>($($arg),+),
            71 => $worker::<71>($($arg),+),
            72 => $worker::<72>($($arg),+),
            73 => $worker::<73>($($arg),+),
            74 => $worker::<74>($($arg),+),
            75 => $worker::<75>($($arg),+),
            76 => $worker::<76>($($arg),+),
            77 => $worker::<77>($($arg),+),
            78 => $worker::<78>($($arg),+),
            79 => $worker::<79>($($arg),+),
            80 => $worker::<80>($($arg),+),
            81 => $worker::<81>($($arg),+),
            82 => $worker::<82>($($arg),+),
            83 => $worker::<83>($($arg),+),
            84 => $worker::<84>($($arg),+),
            85 => $worker::<85>($($arg),+),
            86 => $worker::<86>($($arg),+),
            87 => $worker::<87>($($arg),+),
            88 => $worker::<88>($($arg),+),
            89 => $worker::<89>($($arg),+),
            90 => $worker::<90>($($arg),+),
            91 => $worker::<91>($($arg),+),
            92 => $worker::<92>($($arg),+),
            93 => $worker::<93>($($arg),+),
            94 => $worker::<94>($($arg),+),
            95 => $worker::<95>($($arg),+),
            96 => $worker::<96>($($arg),+),
            _ => None,
        }
    };
}

/// Whether an `ExpressionScale` `obs_scale` divisor is admissible on an analytic ODE
/// walk: non-LTBS (the in-walk log can't compose with the post-walk quotient), program
/// (θ, η) axis counts matching the model's, and within the dual-width cap. Shared by the
/// non-IOV ([`ode_analytical_supported`]) and IOV ([`ode_iov_supported`]) gate arms so
/// the admissibility rule lives in one place — a future narrowing can't drift between the
/// two routes and admit a scale on one path that the other rejects (#575 review).
fn expression_scale_axes_admissible(
    p: &crate::parser::model_parser::ScaleDerivProgram,
    model: &CompiledModel,
) -> bool {
    !model.log_transform
        && p.n_theta_axis() == model.n_theta
        && p.n_eta_axis() == model.n_eta
        && (1..=MAX_ODE_AXES).contains(&p.n_axes())
}

/// True when [`ode_subject_sensitivities`] can serve this model: an ODE model
/// with a compiled RHS program, single `ObsCmt` readout, no built-in absorption,
/// no `init(...)`, no IOV/SDE, no output transform, and an individual-parameter
/// count within [`MAX_ODE_SENS_DIM`]. Per-subject gates (bolus-only doses, no TV
/// covariates/resets) are checked in [`ode_subject_sensitivities`].
pub fn ode_analytical_supported(model: &CompiledModel) -> bool {
    // A `TIME`-built-in structural parameter is served analytically on the ODE path
    // too: the subject routes through the event-driven TV-cov walk (`ode_tvcov_supported`
    // admits `uses_time_builtin`), which seeds each event's PK-param duals at that
    // event's time (#486). The RHS-clock `TIME` (`rhs_program.uses_time_vars`) was
    // already analytic; this covers `TIME` read in `[individual_parameters]`.
    let Some(ode) = model.ode_spec.as_ref() else {
        return false;
    };
    if ode.rhs_program.is_none() {
        return false;
    }
    // A `TIME`-built-in structural parameter is served ONLY via the event-driven TV walk
    // (`ode_subject_supported` declines it so it can't take the static superposition path).
    // A `TIME`-built-in model routes to the event-driven walk (`integrate_tvcov_g`), which now
    // carries BOTH a built-in absorption input-rate forcing (`R_in`, #643) and a seeded
    // `init(...)` state (#662) alongside the per-event `TIME` seeding (#637) — so a
    // TIME + input-rate or TIME + `init(...)` model IS analytic on that walk (#486). (The old
    // decline here assumed the walk carried neither, which was true before #643/#662.)
    // Readout: the state directly (`ObsCmt`), a simple Form C output program
    // (`y = <expr>` over states/indiv params, e.g. `central / V1`), or a per-CMT
    // Form C (`y[CMT=N] = <expr>`) where every endpoint carries a simple program
    // (#439 — each observation reads its CMT's program over the dual state).
    let readout_ok = match &ode.readout {
        OdeReadout::ObsCmt(_) => true,
        OdeReadout::Single(_) => ode
            .readout_program
            .as_ref()
            .is_some_and(|p| p.is_dual_evaluable()),
        OdeReadout::PerCmt(map) => {
            !map.is_empty()
                && map
                    .values()
                    .all(|r| r.program.as_ref().is_some_and(|p| p.is_dual_evaluable()))
        }
    };
    if !readout_ok {
        return false;
    }
    if !ode.diffusion_var.is_empty() {
        return false;
    }
    // Built-in absorption input-rate forcing is evaluated over Dual2 only for
    // kinds lifted to PkNum: inverse-Gaussian (#430), transit (#468, riding the
    // `ln_gamma` Dual2 rule #458), and Weibull (#498, log-domain `exp(β·ln x)`)
    // are all lifted. The check stays kind-agnostic via `supported_over_dual()`
    // so a *future* unlifted kind keeps the FD fallback (a model using it is not
    // "supported" here) without editing this gate.
    if ode.input_rate.iter().any(|f| !f.kind.supported_over_dual()) {
        return false;
    }
    if model.n_kappa != 0 {
        return false;
    }
    // Output transforms applied over the dual prediction in `run_subject`: `None`, a
    // constant `ScalarScale` divisor (`f/k`), and the LTBS log (`ln f`). An η-dependent
    // `ExpressionScale` divisor (`obs_scale = expr(θ,η)`) is ALSO analytic now (#486):
    // it is applied on the final `(θ,η)`-space `SubjectSens` via the shared
    // `apply_expression_scale_outer` (the closed-form provider's quotient rule), on the
    // **static** walk AND the **TV-cov event-driven** walk — the scale is subject-static
    // even under time-varying covariates (production `apply_scaling` reads
    // `subject.covariates`, not a per-event snapshot), so `ode_tvcov_supported` admits it
    // and applies one subject-static quotient post-walk (#486). It still requires
    // `!log_transform` (the walk applies LTBS in PK-param space *before* the η/θ chain, so
    // the production scale-then-log order can't be reproduced by a post-walk quotient —
    // LTBS keeps the FD fallback). Both compose with a Form-C readout (`y = state/V`), the
    // other supported route. Allowlist (not denylist) so a future scaling variant can only
    // *narrow* the analytic scope, never silently admit an unhandled one.
    match &model.scaling {
        ScalingSpec::None | ScalingSpec::ScalarScale(_) => {}
        ScalingSpec::ExpressionScale { deriv: Some(p), .. }
            if expression_scale_axes_admissible(p, model) => {}
        _ => return false,
    }
    // (ODE models have no `tv_fn` — typical values come from `pk_param_fn` at
    // η = 0 instead; see `run_subject`.)
    // Estimated lagtime — bare `LAGTIME`/`ALAG` and compartment-indexed `ALAG{n}` (#369)
    // — IS supported: lagtime is an *event-time* sensitivity (the dose arrives at
    // `t_dose + lag`), handled on the event-driven walk via the per-dose shift and the
    // event-time saltation, with the indexed slot resolved through `DoseAttrMap::lag_slot`
    // (so per-compartment / non-uniform lags are exact); `ode_subject_supported` routes any
    // lagtime subject to that walk rather than the static superposition walk (#439/#472).
    // Per-compartment bioavailability (`F1`/`F2`, #369/#486) is ALSO supported: both the
    // static `integrate_g` and the TV-cov walk resolve `F` per dose compartment via
    // `f_bio_slot` (the indexed `F{cmt}` slot, else the bare `PK_IDX_F`), mirroring
    // production's `DoseAttrMap::f_bio`, so the analytic gradient carries `∂/∂F{cmt}`
    // exactly. Both indexed slots are ordinary individual parameters seeded in
    // `params_dual` like any other — so lagtime and indexed-`F` compose.
    // Lagtime + a built-in absorption **input rate** now composes on the event-driven
    // walk (`integrate_tvcov_g`, #486): an estimated lagtime forces that walk, which
    // now carries `R_in` — the continuous `∂R_in/∂lag` flows through the shared
    // `add_prepared_input_rate_forcing` helper (its `tad` carries the lag jet), and
    // the *onset* term (the forcing switching on at the dose's lagged arrival, a
    // discontinuity in `du/dt` at that moving boundary) is injected as an exact
    // rate-on saltation, mirroring the existing lagged-infusion rate-on injection.
    // `InverseGaussian`'s onset vanishes for every valid `(mat, cv2)` (an essential
    // singularity dominates); `Transit`'s vanishes for `n > 0` and is a finite
    // `ktr·dose` jump at the degenerate `n = 0`; `FirstOrder`'s onset is always the
    // finite `ka·dose`. `Weibull`'s onset **diverges** for shape `β < 1` (an
    // integrable spike, not a finite jump) — no closed-form saltation exists there,
    // so a model combining lagtime with `Weibull` stays on the FD fallback.
    // `ZeroOrder`'s moving-boundary window is now carried on the event-driven walk too
    // (#486): its two boundaries — the rate-on at the lagged arrival `t_dose + lag` and
    // the rate-off at `t_dose + lag + dur` — both shift with `lag`, injected as the
    // rate-on / rate-off saltations `integrate_tvcov_g` already uses for infusions (the
    // rate-off additionally carries `∂/∂dur`). So `zero_order` + lagtime is admitted,
    // leaving only `Weibull` (whose onset diverges for shape `β < 1`) on the FD fallback.
    if model.has_lagtime()
        && ode.input_rate.iter().any(|f| {
            !matches!(
                f.kind,
                crate::pk::absorption::InputRateKind::Transit
                    | crate::pk::absorption::InputRateKind::InverseGaussian
                    | crate::pk::absorption::InputRateKind::FirstOrder
                    | crate::pk::absorption::InputRateKind::ZeroOrder
            )
        })
    {
        return false;
    }
    // The η/θ chain evaluates the individual-parameter program over `Dual2`
    // seeded on (θ, η); require it present, with matching axis counts (no NN-θ /
    // IOV), and within the analytic-chain dual-width cap.
    match ode.indiv_param_program.as_ref() {
        Some(p) => {
            if p.n_theta_axis() != model.n_theta
                || p.n_eta_axis() != model.n_eta
                || p.n_axes() > MAX_ODE_AXES
            {
                return false;
            }
        }
        None => return false,
    }
    let n = model.pk_indices.len();
    (1..=MAX_ODE_SENS_DIM).contains(&n)
}

/// Per-subject scope gate, shared by the full (outer `Dual2`) and light (inner
/// `Dual1`) ODE providers so a subject is served analytically for **both** the
/// outer gradient and the inner EBE loop, or neither (the inner/outer scope must
/// match — a split would mix an analytic gradient with an FD Jacobian).
pub(crate) fn ode_subject_supported(model: &CompiledModel, subject: &Subject) -> bool {
    // Model-level scope + time-varying covariates (the static dual walk holds the PK
    // params constant across the integration). A `TIME`-built-in structural parameter
    // is per-event dynamic for the same reason a TV covariate is, so the static walk
    // cannot serve it — decline here (it routes to the event-driven TV walk via
    // `ode_tvcov_supported`, or to FD if that also declines) (#486).
    if !ode_analytical_supported(model)
        || subject.has_tv_covariates()
        || crate::parser::model_parser::compiled_model_uses_time_builtin(model)
    {
        return false;
    }
    // Steady-state dosing is not yet supported over the dual loop (needs dual
    // SS-equilibration); bolus and (finite-duration) infusion doses are handled.
    if subject.has_periodic_ss_dose() {
        return false;
    }
    // Modeled-`RATE` doses (`RATE=-1`→`R{cmt}` rate, `RATE=-2`→`D{cmt}` duration)
    // arrive *unresolved* — the production ODE path resolves them from the PK params
    // per evaluation (`resolve_modeled_doses`, #324), but the dual walk reads
    // `subject.doses` directly. An unresolved infusion would integrate with the raw
    // coded rate/duration (a bolus/zero-input surrogate), so route these subjects to
    // FD, mirroring the analytical provider's `all_doses_fixed` gate (#410 fallback
    // hardening).
    if !subject.all_doses_fixed() {
        return false;
    }
    // #419: a *rate-defined* infusion under bioavailability `F ≠ 1` reshapes the
    // infusion window (NONMEM holds the rate and scales the duration to `F·amt/rate`)
    // rather than scaling the magnitude. The dual walk applies `F` as a rate
    // magnitude scale (`f_bio · rate` over the original window), which diverges from
    // the production predictor for these subjects — route them to FD so the analytic
    // gradient stays the gradient of the actual objective.
    if model.has_bioavailability() && subject.has_rate_defined_infusion() {
        return false;
    }
    // Built-in absorption forcing (igd/transit/weibull/etc., #430) + EVID 3/4 resets:
    // the dual forcing loop in `integrate_g` now threads the same tracked
    // `reset_floor` the f64 path uses, turning off a dose's pre-reset tail exactly
    // like production's `active_infusions` (#486) — so this combination is analytic
    // on both the outer θ-sensitivities and the inner η-gradient (the shared scope
    // gate keeps the two in lockstep, #430 review #1).
    // Estimated lagtime always routes to the **event-driven** walk (`ode_tvcov_supported`
    // / `run_subject_tvcov`), where the per-dose event-time saltation handles it exactly
    // (uniform or per-compartment lags). The static superposition walk here assumes a
    // single param set with no event-time shift, so it never serves a lagtime subject.
    if model.has_lagtime() {
        return false;
    }
    true
}

/// True when an infusion with (lagged) window start `start` and length `duration` fully
/// spans the integration segment `[seg_start, seg_end]` and has not been turned off by an
/// intervening EVID 3/4 reset (`start >= reset_floor`). The boolean predicate shared by both
/// analytic-sensitivity walks (`integrate_tvcov_g`, `integrate_g`) so the `reset_floor` guard
/// and the production `INFUSION_EPS` window tolerance stay single-sourced (#472 review [7]).
fn infusion_spans_segment(
    start: f64,
    duration: f64,
    seg_start: f64,
    seg_end: f64,
    reset_floor: f64,
) -> bool {
    let eps = crate::ode::predictions::INFUSION_EPS;
    start >= reset_floor && start <= seg_start + eps && start + duration >= seg_end - eps
}

/// Find dose `d`'s zero-order absorption forcing (the parser admits at most one `ZeroOrder`
/// per compartment) and its prepared duration jet from a PK snapshot `params`, if any. Shared
/// by the static (`integrate_g`) and event-driven (`integrate_tvcov_g`) walks so their
/// zero-order window construction — `rate = F·amt·frac/dur` delivered over
/// `[w_start, w_start+dur]` — cannot drift (#653 review #7). Returns `(forcing, dur)`; the
/// caller forms the rate and window (the two walks differ only in the window START: the static
/// walk applies doses at `d.time`, the event-driven walk at the lagged `d.time + lag`).
fn zero_order_forcing_for_dose<'f, T: crate::sens::num::PkNum>(
    ode: &'f OdeSpec,
    d: &crate::types::DoseEvent,
    params: &[T],
) -> Option<(&'f crate::pk::absorption::InputRateForcing, T)> {
    let f = ode.input_rate.iter().find(|f| {
        f.kind == crate::pk::absorption::InputRateKind::ZeroOrder && f.cmt + 1 == d.cmt
    })?;
    let dur = match f.prepare_dual::<T>(params)? {
        crate::pk::absorption::PreparedInputRate::ZeroOrder { dur, .. } => dur,
        _ => return None,
    };
    Some((f, dur))
}

/// True when the time-varying-covariate ODE walk ([`run_subject_tvcov`] /
/// [`run_subject_tvcov_eta`]) can serve this `(model, subject)`: an in-scope analytic
/// ODE model whose subject carries TV covariates and uses the **bolus** dose subset.
/// Non-IOV EVID=2 covariate breakpoints (#636) are analytic; `init(...)` is analytic for the
/// **plain-bolus** subset (seeded at the first-record snapshot, #486) and routes to FD when
/// combined with a reset / lagtime / finite infusion / input-rate forcing / steady-state /
/// modeled-dose (see `ode_tvcov_supported`). The IOV dual walk carries EVID=2 breakpoints
/// separately since #590. Checked by *both* the outer and inner entry points so the analytic
/// scope stays matched (#439).
/// Resolve a dose's modeled `RATE` to its PK slot (#530). A `RATE=-2` (`D{cmt}`) dose reads
/// its duration slot, a `RATE=-1` (`R{cmt}`) dose its rate slot; a `Fixed` dose — or a modeled
/// dose whose `D{cmt}`/`R{cmt}` slot is undeclared — returns `None`. Single-sourced so the
/// `ode_tvcov_supported` slot-presence gate and the `integrate_tvcov_readout` jet resolution
/// can never resolve the same dose differently if a new `RateMode` / `DoseAttr` mapping lands
/// (#530 review finding 5).
pub(crate) fn modeled_slot_for(
    attr_map: &crate::types::DoseAttrMap,
    d: &crate::types::DoseEvent,
) -> Option<(crate::types::RateMode, usize)> {
    match d.rate_mode {
        crate::types::RateMode::Fixed => None,
        crate::types::RateMode::ModeledDuration => attr_map
            .indexed_slot(crate::types::DoseAttr::Duration, d.cmt)
            .map(|s| (crate::types::RateMode::ModeledDuration, s)),
        crate::types::RateMode::ModeledRate => attr_map
            .indexed_slot(crate::types::DoseAttr::Rate, d.cmt)
            .map(|s| (crate::types::RateMode::ModeledRate, s)),
    }
}

pub(crate) fn ode_tvcov_supported(model: &CompiledModel, subject: &Subject) -> bool {
    // The event-driven walk serves a subject with time-varying covariates, an estimated
    // lagtime (per-dose event-time saltation), a steady-state dose (dual SS equilibration),
    // **or** a rate-defined infusion under `F ≠ 1` (#419: the bioavailable window length is
    // a moving boundary in `F`, carried by the rate-off saltation) — anything the static
    // superposition walk can't do. A subject with none of those uses the cheaper static walk.
    let has_ss = subject.has_periodic_ss_dose();
    let has_rate_defined_under_f =
        model.has_bioavailability() && subject.has_rate_defined_infusion();
    // #530: a modeled-`RATE`/duration dose (`RATE=-1`/`-2`, `R{cmt}`/`D{cmt}`) resolves its
    // rate/window from a PK slot, so the infusion *end* is a moving boundary in that
    // parameter — the event-driven walk carries it via the rate-off saltation exactly as
    // lagtime carries the moving *start*. The static superposition walk can't, so route
    // these subjects here (`ode_subject_supported` keeps gating them off the static walk).
    let has_modeled_dose = !subject.all_doses_fixed();
    // A `TIME`-built-in structural parameter is per-event dynamic (the switch fires at
    // event times), so it routes through the event-driven walk even with no TV
    // covariates — mirroring the closed-form `subject_sensitivities_tvcov` (#486).
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    if !ode_analytical_supported(model)
        || !(subject.has_tv_covariates()
            || model.has_lagtime()
            || has_ss
            || has_rate_defined_under_f
            || has_modeled_dose
            || uses_time)
    {
        return false;
    }
    // An η-dependent `ExpressionScale` `obs_scale` divisor IS served on this non-IOV TV-cov
    // walk now (#486): the scale is **subject-static even for TV-cov subjects** (production
    // `apply_scaling` reads `subject.covariates`), so `run_subject_tvcov` /
    // `run_subject_tvcov_eta` apply it as a single subject-static post-walk quotient — the
    // IOV per-occasion-group machinery (#590) collapses to one jet because there is no κ.
    // Admissibility (`!log_transform`, axis counts, axis cap) is already enforced upstream
    // by `ode_analytical_supported`'s scaling allowlist (only `None`/`ScalarScale`/an
    // axis-admissible `ExpressionScale` reach here), so no extra guard is needed; LTBS
    // stays FD via that allowlist's `!log_transform` admissibility. A `TIME`-built-in
    // structural parameter is served here too (per-event dynamic, mirrors TV-cov) and
    // composes with the same subject-static scale quotient.
    // Estimated lagtime IS supported here (bare or per-compartment `ALAGn`):
    // `integrate_tvcov_g` shifts each dose to `t_dose + lag` and injects the event-time
    // (saltation) sensitivity, propagated exactly through the per-event params (#439).
    // Bound total axes so BOTH TV-cov dispatch tables resolve: the outer
    // `run_subject_tvcov` dispatches `M = n_theta + n_eta` and the inner
    // `run_subject_tvcov_eta` dispatches `n_eta`, each over `1..=MAX_ODE_AXES`. With
    // `n_eta ≤ n_theta + n_eta ≤ MAX_ODE_AXES`, both succeed — so the inner and outer
    // analytic scope stay matched (never an analytic outer with an FD inner, and no
    // silent `_ => None` downgrade) (#449 review #4).
    if model.n_theta + model.n_eta > MAX_ODE_AXES {
        return false;
    }
    let Some(ode) = model.ode_spec.as_ref() else {
        return false;
    };
    // Built-in absorption input-rate forcing (transit/igd/weibull/first_order, #486):
    // the TV-cov event-driven walk (`integrate_tvcov_g`) now carries `R_in` via the
    // shared `add_prepared_input_rate_forcing` helper, exactly as the static walk
    // does. A bolus dose's (lagged) arrival is a *fixed* (non-dual) boundary when
    // there is no estimated lagtime, so the pure TV-cov case needs no saltation; the
    // combined lagtime + input-rate saltation is gated separately by
    // `ode_analytical_supported` (which this function calls above), so a model
    // reaching here with both already carries only the boundary-safe kinds.
    // `ZeroOrder`'s moving-boundary window (`#530`) is now ported to this walk too (#486):
    // `integrate_tvcov_g` rebuilds the constant `F·amt·frac/dur` window from each dose's own
    // PK snapshot and delivers it as a per-segment constant (`active_zero`, mirroring
    // `integrate_g`'s `zero_windows` and production's `active_zero_order_inputs`), with the
    // rate-off saltation at the moving window end `d.time+lag+dur` (and, under an estimated
    // lagtime, the rate-on saltation at the moving start). `add_prepared_input_rate_forcing`
    // still skips `ZeroOrder` — the per-segment constant channel owns its delivery, exactly
    // as on the static walk — so there is no longer any dropped forcing to guard against.
    // `init(...)` initial conditions are seeded on the event-driven walk at the subject's
    // first-record covariate snapshot (`tvcov_init_state`, matching production's `init_pk`),
    // so a TV-cov + `init(...)` **bolus** subject is analytic on both loops (#649). As of #486
    // `init(...)` is **fully analytic** on this non-IOV event-driven walk — it composes with
    // every dosing feature the walk carries (finite infusion, estimated lagtime, built-in
    // input-rate forcing, steady-state, modeled-duration/rate dose, and EVID 3/4 reset), each
    // validated by an FD-vs-production test (below). The one-line summary of why each is safe:
    // A **finite infusion** now composes with `init` (#486): the walk seeds `init_state` and
    // then accumulates the `active_infusions` forcing on top of the running (init-derived)
    // state — the forcing is a `du/dt` additive term with no zero-baseline assumption, so the
    // analytic gradient already matches FD-of-production on a non-zero baseline (validated by
    // `ode_provider_tvcov_init_infusion_matches_production`).
    // An estimated **lagtime** also composes with `init` (#486): the dose-arrival saltation
    // reads the *actual* pre-arrival state `x⁻ = u` (the decaying `init` baseline) and its
    // velocity `g(x⁻)` — the "`x⁻ = 0` first-dose reduction" in the comment there is only a
    // reduction, not an assumption, so a non-zero `init` baseline flows through the saltation
    // unchanged (validated by `ode_provider_tvcov_init_lagtime_matches_production`).
    // A **modeled-duration/rate dose** (`RATE=-1/-2`) also composes with `init` (#486): the
    // moving infusion-end boundary is carried by the rate-off saltation reading the live
    // `inf_eff` jet, on top of the init-seeded running state — no zero-baseline assumption
    // (validated by `ode_provider_tvcov_init_modeled_duration_matches_production`; the
    // slot-presence invariant below still guards an undeclared `D{cmt}`/`R{cmt}`).
    // A built-in **input-rate forcing** also composes with `init` (#486). For continuous kinds
    // (igd/transit/weibull/first_order) `add_prepared_input_rate_forcing` adds `R_in(tad)` to
    // `du/dt` on top of the init-seeded running state; for `ZeroOrder` (now carried on this walk
    // since #653) the per-segment constant `active_zero` channel delivers `F·amt·frac/dur` and
    // the moving-end rate-off saltation reads the same running state. Both are orthogonal to the
    // starting-state offset (validated by `ode_provider_tvcov_init_input_rate_matches_production`
    // and `ode_provider_tvcov_init_zero_order_matches_production`).
    // **Steady-state** also composes with `init` (#486): production's `equilibrate_ss_state`
    // seeds the SS trough from zero and *overwrites* the whole state at the SS dose, discarding
    // `init` there — so the dual `equilibrate_ss_state_g` (which also seeds from zero) matches
    // it; `init` only affects the pre-SS-dose segments, which the walk carries via `init_state`
    // (validated by `ode_provider_tvcov_init_ss_matches_production`).
    // An EVID 3/4 **reset** also composes with `init` (#486): production re-applies
    // `init(&last_pk.values)` at each reset, and the `K_RESET` branch of `integrate_tvcov_g`
    // now mirrors that by re-seeding the state from the reset-event snapshot (via the shared
    // `init_taylor_seed_at`, carrying `∂init/∂(θ,η)` at that snapshot) instead of zeroing —
    // validated by `ode_provider_tvcov_init_reset_matches_production`.
    // With A–F all composing, `init(...)` is fully analytic on the non-IOV event-driven walk;
    // no per-feature FD clause remains here (IOV is gated separately in `ode_iov_supported`).
    // #530: modeled-`RATE`/duration doses are resolved from their PK slot as a live jet
    // inside the walk (`inf_eff` reads `pk_at_dose[k][slot]`), so the moving infusion-end
    // boundary is analytic via the rate-off saltation. Modeled-dose × steady-state IS now
    // analytic too (#486): `equilibrate_ss_state_g` threads the same `inf_eff` jet into its
    // per-cycle active/quiet split. Decline to FD only when a modeled dose's `D{cmt}`/`R{cmt}`
    // slot is absent — `check_model_data` rejects such a model, but if one slips past, emit
    // FD rather than a wrong gradient (the walk's `inf_eff` would silently fall to the
    // unresolved fixed arm).
    if !subject.all_doses_fixed() {
        let attr_map = model.active_dose_attr_map();
        let all_slots_present = subject.doses.iter().all(|d| {
            matches!(d.rate_mode, crate::types::RateMode::Fixed)
                || modeled_slot_for(attr_map, d).is_some()
        });
        if !all_slots_present {
            return false;
        }
    }
    // #419: a rate-defined infusion under `F ≠ 1` reshapes the *window* to `F·dur` (a
    // moving boundary) — handled by the rate-off saltation everywhere now, including a
    // **steady-state** rate-defined infusion (#486: `equilibrate_ss_state_g`'s per-cycle
    // active/quiet split reads the same `F`-scaled `inf_eff` window as the main walk).
    // Steady-state (`SS=1`, `II>0`) — bolus *and* infusion — is handled via the dual
    // equilibration (the SS infusion runs an active-rate window + quiet window per cycle).
    // SS combined with an estimated **lagtime** IS now analytic too (#486): a lagged SS dose
    // arrives at `t_dose + lag`, so observations in the pre-arrival window
    // `[t_dose, t_dose + lag)` must read the *previous* interval's steady-state tail — the
    // walk seeds it via a `K_SS_SEED` timeline break at the dose record time, calling
    // `ss_state_at_phase_g` (the dual mirror of production's `ss_state_at_phase`) at phase
    // `II − lag`. The dose's *own* arrival still goes through the **unmodified** general
    // lagtime saltation (`x⁻ = ` the freshly re-equilibrated trough, which carries no
    // explicit `∂/∂lag` jet of its own): even though the trough's *value* is lag-invariant
    // (an autonomous RHS, required below, makes the periodic recurrence anchored to the
    // pulse, not to wall-clock arrival time), a later *fixed-absolute-time* observation's
    // elapsed time since arrival still shifts with `lag`, and the saltation's `D·δlag` term
    // captures exactly that — mirroring the existing `x⁻ = 0` first-dose reduction.
    // SS combined with a **non-autonomous RHS** (one that reads `TIME`/`TAFD`/`TAD`) routes
    // to FD: the SS dual equilibration expands a time-*invariant* pulse train (cycle-relative
    // time, anchor 0), so a time/TAD-dependent RHS breaks the steady-state cycle recurrence —
    // the dual walk's monotonic TAD diverges from production's per-interval anchor, giving a
    // wrong prediction *and* gradient (#473 review #1, verified vs the production predictor).
    if has_ss && ode.rhs_program.as_ref().is_some_and(|p| p.uses_time_vars()) {
        return false;
    }
    // EVID 3/4 resets and finite-duration infusions ARE handled (resets zero the state;
    // infusions add `F·rate` forcing over their lagged window, with rate-boundary lagtime
    // saltation). EVID=2 pk-only breakpoints are ALSO carried now (#486): the timeline in
    // `integrate_tvcov_g` pushes each `K_PKONLY` event and integrates its segment with the
    // PK params seeded at that record's covariate snapshot (`pk_at_pk_only`), exactly as
    // the obs/dose breakpoints — with no κ there is no `iov_combined_pk_only` analogue to
    // build (mirrors the IOV path's #590 support).
    true
}

/// True when the ODE **IOV** outer gradient ([`ode_subject_sensitivities_iov`]) can
/// serve this model — the ODE counterpart of
/// [`crate::sens::provider::iov_analytical_supported`]. A model-level gate; the
/// per-subject scope (bolus-only, occasion split, axis cap with `K`) is checked in
/// [`ode_subject_sensitivities_iov`].
///
/// Deliberately a *parallel* gate to [`ode_analytical_supported`] rather than lifting
/// its `n_kappa == 0` bail: the non-IOV inner η-gradient and outer walk seed `n_eta`
/// axes from a program whose `n_eta_axis()` would be `n_eta + n_kappa` under IOV, so
/// admitting IOV there would mis-seed κ at zero. IOV is its own analytic-outer-only
/// path (the inner EBE loop stays FD, exactly as the analytical IOV path leaves it —
/// `analytical_supported` requires `n_kappa == 0`). First cut: bolus-only (time-varying
/// covariates supported), no scaling/LTBS/lagtime/absorption/init — mirroring the narrow
/// TV-cov scope; anything outside routes to FD (#439 ODE IOV).
pub fn ode_iov_supported(model: &CompiledModel) -> bool {
    if model.n_kappa == 0 {
        return false;
    }
    let Some(ode) = model.ode_spec.as_ref() else {
        return false;
    };
    if ode.rhs_program.is_none() {
        return false;
    }
    // M3 BLOQ is analytic on the ODE IOV path (#486, mirroring closed-form #580/#591).
    // Censoring is provider-agnostic: the inner (`residual_inner_obs`) and outer
    // (`prepare_stacked`) assemblies apply the censored `−logΦ` coefficient keyed on
    // `subject.cens[j]`, over whatever `ObsSens`/`ObsGrad` the walk emits — the ODE IOV
    // walk emits the standard shape, so M3 rides the same stacked `[η_bsv, κ]` assembly
    // the closed-form path uses. No M3 clause here. (The ODE M3 + `iiv_on_ruv` triple
    // stays FD via the residual-eta clause below.)
    // IIV on residual error (`iiv_on_ruv`) is analytic on the ODE IOV path (#486). `η_ruv`
    // scales the residual variance by `exp(2·η_ruv)` but does not enter the structural
    // prediction, so the ODE walk emits a zero `∂f/∂η_ruv` column on the η_ruv axis (its
    // `CombinedDerivs.deta` is 0 for every PK param, exactly as on the closed-form walk).
    // The variance scaling and the `η_ruv` gradient column are then applied downstream by
    // the provider-agnostic assembly (`residual_inner_obs` / `prepare_stacked` via
    // `residual_var_scale`, keyed on `residual_error_eta`) — the same shared gradient the
    // closed-form IOV `iiv_on_ruv` path uses. Combined with M3 this is the full triple.
    // A `TIME`-built-in structural parameter is threaded through the ODE IOV walk
    // (`run_subject_iov`/`_eta` seed each occasion's stacked PK duals at that event's
    // time via the per-event branch, #486). An η-dependent `ExpressionScale` obs_scale
    // composes with it: the per-occasion scale jet is built at the static `t = 0`
    // snapshot, which matches production's `apply_scaling` (also evaluated at `t = 0`,
    // i.e. the switch's early value) — so no special-casing is needed here.
    // FREM + IOV: the analytic IOV inner gradient never substitutes the FREM covariate
    // pseudo-obs variance, and the IOV objective returns a `1e18` sentinel for FREM+IOV.
    // Route to FD. (#466 review round 2.)
    if model.frem_config.is_some() {
        return false;
    }
    // Readout: state directly, simple Form-C, or per-CMT — same set as the non-IOV gate.
    let readout_ok = match &ode.readout {
        OdeReadout::ObsCmt(_) => true,
        OdeReadout::Single(_) => ode
            .readout_program
            .as_ref()
            .is_some_and(|p| p.is_dual_evaluable()),
        OdeReadout::PerCmt(map) => {
            !map.is_empty()
                && map
                    .values()
                    .all(|r| r.program.as_ref().is_some_and(|p| p.is_dual_evaluable()))
        }
    };
    if !readout_ok {
        return false;
    }
    if !ode.diffusion_var.is_empty() {
        return false;
    }
    // Built-in absorption input-rate forcing under IOV (#486). The shared
    // `integrate_tvcov_g` walk delivers each forcing per-occasion — its rate/window is
    // rebuilt from that dose's own occasion-seeded `pk_at_dose[k]` jet, so κ rides through
    // exactly as η/θ do. So EVERY kind the non-IOV gate carries is analytic under IOV too:
    // the smooth densities `igd`/`transit`/`weibull`/`first_order` (pointwise `R_in`), the
    // moving-window `zero_order`, and any composition (`mixed`, `parallel`). Mirror the
    // non-IOV `ode_analytical_supported`'s kind-agnostic `supported_over_dual()` gate exactly,
    // so the IOV input-rate scope tracks the non-IOV scope rather than an arbitrary
    // `{ZeroOrder, FirstOrder}` subset (#486 IOV-scope parity). (The SS × input-rate
    // combination is still declined per subject in `ode_iov_subject_supported` — the SS dual
    // equilibration has no built-in-forcing channel; that is a real gap, not this decision.)
    if ode.input_rate.iter().any(|f| !f.kind.supported_over_dual()) {
        return false;
    }
    // `Weibull` + estimated lagtime stays FD on every path (IOV included): its onset diverges
    // for shape `β < 1` (an integrable spike, no finite rate-on saltation), exactly as the
    // non-IOV gate declines it above. Every other kind composes with lagtime under IOV.
    // Mirror the non-IOV `ode_analytical_supported` lagtime gate's exhaustive *whitelist*
    // (not a `== Weibull` blacklist), so a future `InputRateKind` variant defaults to the FD
    // fallback on both paths rather than being silently admitted here (#486 IOV-scope parity).
    if model.has_lagtime()
        && ode.input_rate.iter().any(|f| {
            !matches!(
                f.kind,
                crate::pk::absorption::InputRateKind::Transit
                    | crate::pk::absorption::InputRateKind::InverseGaussian
                    | crate::pk::absorption::InputRateKind::FirstOrder
                    | crate::pk::absorption::InputRateKind::ZeroOrder
            )
        })
    {
        return false;
    }
    // No constant `ScalarScale`/LTBS output transform. Estimated **lagtime IS supported**
    // (bare and compartment-indexed `ALAG{cmt}`, see below): the IOV walk runs through
    // `integrate_tvcov_readout`/`integrate_tvcov_g`, which applies the dose-time shift +
    // event-time saltation per occasion-seeded dose (#439 lagtime × IOV).
    // (The per-subject gate
    // `ode_iov_subject_supported` now ADMITS finite-duration infusions, EVID 3/4 resets,
    // and EVID=2 pk-only breakpoints
    // — the shared `integrate_tvcov_g` walk carries the rate-boundary saltation and the
    // `reset_floor` per occasion — and declines only SS+lagtime, SS+time-dependent RHS,
    // and rate-defined SS infusion under F (#472 review round 2 follow-up #2).)
    //
    // An η-dependent `ExpressionScale` `obs_scale` divisor (`obs_scale = expr(θ,η)`) IS
    // supported (#575): like the non-IOV ODE static walk (#534) it is applied as a
    // post-walk quotient on the final `(θ, stacked-η)` jet — here per occasion group,
    // since the divisor depends on the group's κ through the PK params (see
    // `apply_expression_scale_iov` / `run_subject_iov`). A constant `ScalarScale k` divisor
    // is supported too (#486 IOV-scope parity): unlike `ExpressionScale`, it is
    // κ-independent, so `resolve_obs_readout`/`apply_output_transform` already divide the
    // in-walk readout `p/k` over the stacked `(θ, η, κ)` dual — the exact same in-walk step
    // the non-IOV walk uses (`ode_analytical_supported` admits it), needing no post-walk
    // handling. LTBS still stays FD (the in-walk log can't compose with the per-group
    // post-walk `ExpressionScale` quotient). Allowlist, not denylist, so a future scaling
    // variant can only narrow scope.
    match &model.scaling {
        // `None`/`ScalarScale` only when NOT LTBS: the IOV walk applies the LTBS log in
        // PK-param space *before* the η/θ/κ chain, so the production scale-then-log order
        // can't be reproduced post-walk — LTBS (`log(DV) ~ additive`) stays FD for IOV,
        // matching the pre-#575 `|| model.log_transform` guard.
        ScalingSpec::None | ScalingSpec::ScalarScale(_) if !model.log_transform => {}
        ScalingSpec::ExpressionScale { deriv: Some(p), .. }
            if expression_scale_axes_admissible(p, model) => {}
        _ => return false,
    }
    // Compartment-indexed bioavailability `F{cmt}` and lagtime `ALAG{cmt}` ARE supported under
    // IOV (#486 IOV-scope parity): the shared `integrate_tvcov_readout`/`integrate_tvcov_g`
    // walk resolves each dose's own compartment slot — `f_bio_slot(ode, d.cmt)` and
    // `dose_lag_slot = attr_map.lag_slot(d.cmt)` — exactly as the non-IOV
    // `ode_analytical_supported` walk does (an indexed slot is an ordinary individual-parameter
    // output seeded per occasion by `seed_pk_dual2_iov`). The old bail here assumed a single
    // `PK_IDX_LAGTIME` slot the walk never actually used.
    // `init(...)` initial conditions are analytic on the ODE IOV walk too (#486): the IOV
    // outer/inner (`run_subject_iov` / `_eta`) run through the SAME `integrate_tvcov_readout`
    // the non-IOV walk uses, which seeds `init` via `tvcov_init_state` at the first-record
    // snapshot and re-seeds it at each EVID 3/4 reset. Because that snapshot's PK duals carry
    // the stacked `(θ, η_bsv, κ)` jets, the Taylor seed's deltas fold `∂init/∂κ` into the
    // correct occasion axis with no IOV-specific seeding code — a κ-coupled baseline
    // (`init(central) = C0·exp(η + κ)`) lands its derivative on the right axis. Validated by
    // `ode_iov_init_provider_matches_fd_of_predict_iov`.
    // The η/θ/κ chain evaluates the individual-parameter program over the **combined**
    // `(θ, η_bsv, κ)` axes (`n_eff = n_eta + n_kappa`); require it present with matching
    // axes and a program-eval width within the dispatch table. (The per-subject stacked
    // walk width `n_theta + n_eta + K·n_kappa` is bounded separately, per subject.)
    let n_eff = model.n_eta + model.n_kappa;
    match ode.indiv_param_program.as_ref() {
        Some(p) => {
            if p.n_theta_axis() != model.n_theta
                || p.n_eta_axis() != n_eff
                || model.n_theta + n_eff > MAX_ODE_AXES
            {
                return false;
            }
        }
        None => return false,
    }
    (1..=MAX_ODE_SENS_DIM).contains(&model.pk_indices.len())
}

/// Compute per-observation analytic sensitivities for an ODE model, or `None` if
/// it is outside the supported scope (caller falls back to the gradient-free
/// path).
pub fn ode_subject_sensitivities(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<SubjectSens> {
    // Time-varying covariates: the `(θ,η)`-seeded event-driven walk (dual width
    // `M = n_theta + n_eta ≤ MAX_ODE_AXES`), mirroring the analytical TV-cov path.
    if ode_tvcov_supported(model, subject) {
        macro_rules! dispatch_tv {
            ($($m:literal),+) => {
                match model.n_theta + model.n_eta {
                    $($m => run_subject_tvcov::<$m>(model, subject, theta, eta),)+
                    _ => None,
                }
            };
        }
        return dispatch_tv!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    }
    if !ode_subject_supported(model, subject) {
        return None;
    }
    // PK params at (θ, η). A bare estimated lagtime IS handled now (the dose-time
    // shift + saltation in `integrate_g`); per-compartment / infusion / SS / reset
    // lagtime is excluded by `ode_subject_supported`, so no runtime short-circuit is
    // needed here. `pk` and `pd` are each evaluated once and threaded into the drivers,
    // so neither recomputes them.
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    // Individual-parameter η/θ derivatives (cheap: one dual eval, no integration).
    // Besides feeding the chain, the `∂p/∂η` rows tell us which individual parameters
    // carry IIV, which decides the dual's Hessian width.
    let pd = param_derivatives(model, subject, theta, eta)?;
    let n_indiv = model.pk_indices.len();

    // IIV-bearing parameters: those with any nonzero `∂p/∂η`. The η/θ chain reads
    // `∂²f/∂p_i∂p_j` only when at least one of `i, j` is IIV-bearing (FOCEI never
    // uses `∂²f/∂θ²`), so seeding the IIV-bearing parameters as the leading `na`
    // dual axes lets the second-order block among the IIV-free axes be dropped —
    // the per-step Hessian work falls from `n²` to `na·n` (issue #445). All buffers
    // are stack arrays bounded by `MAX_ODE_SENS_DIM` (`n_indiv ≤ 12`, enforced by
    // `ode_subject_supported`) — no per-subject heap allocation (#448 review #6).
    let mut iiv_buf = [0usize; MAX_ODE_SENS_DIM];
    let mut is_iiv = [false; MAX_ODE_SENS_DIM];
    let mut na = 0usize;
    for i in 0..n_indiv {
        if pd.dp_deta[i].iter().any(|&v| v != 0.0) {
            iiv_buf[na] = i;
            is_iiv[i] = true;
            na += 1;
        }
    }
    let iiv = &iiv_buf[..na];

    // Axis permutation: IIV-bearing parameters take axes `0..na` (full Hessian rows);
    // the IIV-free parameters take `na..n_indiv` (gradient only).
    let mut axis_buf = [0usize; MAX_ODE_SENS_DIM];
    let mut next = 0usize;
    for &i in iiv {
        axis_buf[i] = next;
        next += 1;
    }
    for i in 0..n_indiv {
        if !is_iiv[i] {
            axis_buf[i] = next;
            next += 1;
        }
    }
    let axis_of = &axis_buf[..n_indiv];

    macro_rules! full {
        ($n:literal) => {
            run_subject::<$n>(model, subject, theta, eta, &pk.values, &pd)
        };
    }
    macro_rules! mixed {
        ($na:literal, $n:literal) => {
            run_subject_mixed::<$na, $n>(model, subject, theta, eta, &pk.values, &pd, axis_of, iiv)
        };
    }
    // For each individual-parameter count `n`, route to the mixed-order dual for
    // every `0 < na < n` (the arm lists below enumerate `na` up to `MIXED_NA_CAP =
    // n_max − 1`, so all are covered); the full `Dual2<n>` path handles only
    // `na == n` (no IIV-free block to drop) and `na == 0` (no IIV) via the `_` arm.
    macro_rules! by_n {
        ($n:literal; $($na:literal),*) => {
            match na {
                $( $na => mixed!($na, $n), )*
                _ => full!($n),
            }
        };
    }
    let mut sens = match n_indiv {
        1 => full!(1),
        2 => by_n!(2; 1),
        3 => by_n!(3; 1, 2),
        4 => by_n!(4; 1, 2, 3),
        5 => by_n!(5; 1, 2, 3, 4),
        6 => by_n!(6; 1, 2, 3, 4, 5),
        7 => by_n!(7; 1, 2, 3, 4, 5, 6),
        8 => by_n!(8; 1, 2, 3, 4, 5, 6, 7),
        9 => by_n!(9; 1, 2, 3, 4, 5, 6, 7, 8),
        10 => by_n!(10; 1, 2, 3, 4, 5, 6, 7, 8, 9),
        11 => by_n!(11; 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        12 => by_n!(12; 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        _ => None,
    }?;
    // η-dependent `ExpressionScale` divisor (#486): apply the subject-static quotient
    // on the final `(θ,η)`-space jet — the SAME `apply_expression_scale_outer` the
    // closed-form provider uses, since both produce an identical `SubjectSens`. This is the
    // **static** walk; the TV-cov event-driven walk applies the identical quotient at its
    // own tail (`run_subject_tvcov`), since the scale is subject-static even under TV
    // covariates (#486). `!log_transform` is gated (LTBS stays FD), and `pd`/`pk.values` are
    // already on hand. `slots = prog.pk_slots()` pairs with `pd` (from `param_derivatives`).
    if let ScalingSpec::ExpressionScale {
        deriv: Some(prog), ..
    } = &model.scaling
    {
        let slots = model
            .ode_spec
            .as_ref()?
            .indiv_param_program
            .as_ref()?
            .pk_slots_ref();
        crate::sens::provider::apply_expression_scale_outer(
            &mut sens,
            prog,
            &pk,
            &pd,
            slots,
            theta,
            eta,
            &subject.covariates,
            model.n_theta,
            model.n_eta,
        );
    }
    Some(sens)
}

/// Largest IIV-bearing-parameter count (`na`) for which the mixed-order dual
/// ([`DualMixed`](crate::sens::dual_mixed::DualMixed)) is monomorphised. Subjects
/// whose model has more than this many IIV-bearing individual parameters fall back
/// to the full `Dual2` path — correct, just not accelerated. Bounds the `(na, n)`
/// monomorphisation count; raise it only if models with many IIV parameters become
/// a measured bottleneck. Set to `MAX_ODE_SENS_DIM - 1` so that **every** `0 < na <
/// n` is specialised (the largest possible `na` is `n - 1 ≤ 11`): no in-scope model
/// silently falls back to the full `Dual2` path for being over the cap — only `na ==
/// n` (no IIV-free block to drop) and `na == 0` (no IIV) take the full path, both
/// correctly (#445 review #6). The cost is the `(na, n)` monomorphisation count
/// (`Σ min(n-1, cap)` over `n ≤ MAX_ODE_SENS_DIM`); lower it if compile time bites.
pub const MIXED_NA_CAP: usize = MAX_ODE_SENS_DIM - 1;

// The `by_n!` arm lists in `ode_subject_sensitivities` enumerate `na` up to
// `MIXED_NA_CAP` explicitly (a macro can't iterate a const). This tripwire fails the
// build if the const is changed without the arms being updated to match — the cap
// was previously `#[cfg(doc)]`-only and could silently drift (#445 review #4).
const _: () = assert!(MIXED_NA_CAP == 11);

/// Light **inner** η-gradient for an ODE model: per-observation `(f, ∂f/∂η)` via a
/// `Dual1` (gradient-only) augmented RK45 — the ODE counterpart of the analytical
/// light provider ([`super::provider::subject_eta_grad`]). The inner EBE loop needs
/// only `∂f/∂η`, so this skips the `Dual2` Hessian *and* the θ-chain: one `Dual1`
/// integration (≈`N`-cost) replaces FD's `2·n_eta+1` plain integrations. Same scope
/// as [`ode_subject_sensitivities`]; `None` falls back to the FD inner gradient
/// (issue #410).
pub fn ode_subject_eta_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<ObsGrad>> {
    // Time-varying covariates: the light η-only walk (`Dual1<n_eta>`), mirroring the
    // outer TV-cov dispatch so the inner/outer analytic scope stays matched.
    if ode_tvcov_supported(model, subject) {
        macro_rules! dispatch_tv {
            ($($n:literal),+) => {
                match model.n_eta {
                    $($n => run_subject_tvcov_eta::<$n>(model, subject, theta, eta),)+
                    _ => None,
                }
            };
        }
        // Up to MAX_ODE_AXES (matches the outer `run_subject_tvcov` M-dispatch and
        // the `ode_tvcov_supported` axis bound), so inner/outer stay matched (#449 #4).
        return dispatch_tv!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    }
    if !ode_subject_supported(model, subject) {
        return None;
    }
    // `pk` and the η-block `∂p/∂η` are evaluated once here and threaded into the driver
    // (mirroring the outer `ode_subject_sensitivities`, which threads `pk`/`pd`), so the
    // light walk doesn't recompute them — and the `ExpressionScale` quotient below reuses
    // the same `dp_deta` rather than running the individual-parameter Dual1 program a
    // second time in the inner BFGS hot loop (#534 review #3).
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let dp_deta = param_eta_derivatives(model, subject, theta, eta)?;
    macro_rules! dispatch {
        ($($n:literal),+) => {
            match model.pk_indices.len() {
                $($n => run_subject_eta::<$n>(model, subject, &pk, &dp_deta),)+
                _ => None,
            }
        };
    }
    let mut out = dispatch!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)?;
    // η-dependent `ExpressionScale` divisor (#486): apply the η-only quotient on the
    // light gradient, via the SAME `apply_expression_scale_inner_dispatch` the
    // closed-form inner provider uses (the η-block of `apply_expression_scale_outer`).
    // `dp_deta` is `∂p/∂η` in `prog.pk_slots()` order, paired with `slots =
    // prog.pk_slots()`; `pk` supplies the referenced PK-param values.
    if let ScalingSpec::ExpressionScale {
        deriv: Some(prog), ..
    } = &model.scaling
    {
        let slots = model
            .ode_spec
            .as_ref()?
            .indiv_param_program
            .as_ref()?
            .pk_slots_ref();
        crate::sens::provider::apply_expression_scale_inner_dispatch(
            &mut out,
            prog,
            &pk,
            &dp_deta,
            slots,
            theta,
            eta,
            &subject.covariates,
            model.n_eta,
        );
    }
    Some(out)
}

/// Exact `∂p/∂η`, `∂p/∂θ` (and second order) of the individual parameters,
/// obtained by evaluating the compiled `[individual_parameters]` program over
/// `Dual2` seeded on (θ, η) — **analytical**, any parameterization (log-normal,
/// logit-normal F, additive, …), no finite differences. (The FD fallback for
/// unsupported models is the existing gradient-free path.)
pub(crate) struct ParamDerivs {
    /// `∂p_i/∂η_k`.
    pub(crate) dp_deta: Vec<Vec<f64>>,
    /// `∂p_i/∂θ_m`.
    pub(crate) dp_dtheta: Vec<Vec<f64>>,
    /// `∂²p_i/∂η_k∂η_l`.
    pub(crate) d2p_deta2: Vec<Vec<Vec<f64>>>,
    /// `∂²p_i/∂η_k∂θ_m`.
    pub(crate) d2p_detadtheta: Vec<Vec<Vec<f64>>>,
}

fn param_derivatives(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<ParamDerivs> {
    let prog = model.ode_spec.as_ref()?.indiv_param_program.as_ref()?;
    param_derivatives_from_prog(prog, model, subject, theta, eta)
}

/// First-order `∂p/∂η` only (the η-block of [`ParamDerivs::dp_deta`]) over a
/// `Dual1<M>` seeded on η, `M = n_eta` — the light inner counterpart of
/// [`param_derivatives`]. Skips the θ-axes and the second-order Hessian the full
/// `Dual2` path computes, since the inner η-gradient consumes only `dp_deta`
/// (#410). Returns `None` on the same axis-count mismatch as the full path.
fn param_eta_derivatives(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<Vec<f64>>> {
    let prog = model.ode_spec.as_ref()?.indiv_param_program.as_ref()?;
    param_eta_derivatives_from_prog(prog, model, subject, theta, eta)
}

/// First-order `∂p/∂η` (the η-block of [`ParamDerivs::dp_deta`]) from an explicit
/// individual-parameter program, over a `Dual1<M>` seeded on η (`M = n_eta`). The
/// light inner counterpart of [`param_derivatives_from_prog`], shared by the ODE
/// provider (program on `ode_spec`) and the analytical PK provider (program on
/// `indiv_param_partials`): it skips the θ-axes and second-order Hessian the full
/// `Dual2` path computes, since the inner EBE η-gradient consumes only `dp_deta`
/// (#410). Dispatches on `n_eta` alone — so unlike the `Dual2`
/// [`param_derivatives_from_prog`] it still serves models whose combined
/// `n_theta + n_eta` exceeds the dual dispatch ceiling, as long as `n_eta` does
/// not. Returns `None` on the same axis-count mismatch as the full path.
pub(crate) fn param_eta_derivatives_from_prog(
    prog: &crate::parser::model_parser::IndivParamProgram,
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<Vec<f64>>> {
    if prog.n_theta_axis() != model.n_theta || prog.n_eta_axis() != model.n_eta {
        return None;
    }
    let ne = model.n_eta;
    let ni = prog.pk_slots_ref().len();
    macro_rules! disp {
        ($($mm:literal),+) => {
            match ne {
                $($mm => {
                    let p = prog.eval_param_eta_grad::<$mm>(theta, eta, &subject.covariates);
                    let mut dp_deta = vec![vec![0.0; ne]; ni];
                    for i in 0..ni {
                        for k in 0..ne {
                            dp_deta[i][k] = p[i].grad[k];
                        }
                    }
                    Some(dp_deta)
                })+
                _ => None,
            }
        };
    }
    disp!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
}

/// Analytical `∂p/∂(θ,η)` (+ second order) from an explicit individual-parameter
/// program, shared by the ODE provider (program on `ode_spec`) and the analytical
/// PK provider (program on `indiv_param_partials`). Returns `None` — caller falls
/// back to FD — when the program's axis counts don't match the model's θ/η (e.g.
/// NN-weight θ or IOV kappa present) or the axis count exceeds the dispatch table.
pub(crate) fn param_derivatives_from_prog(
    prog: &crate::parser::model_parser::IndivParamProgram,
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<ParamDerivs> {
    // Thin wrapper over the cov-taking [`param_derivatives_at_cov`] at the subject's
    // static covariates — single dispatch table, no second `1..=16` copy to widen in
    // lockstep (#449 review #12).
    param_derivatives_at_cov(prog, model, &subject.covariates, theta, eta)
}

/// Pack `∂p/∂(θ,η)` and `∂²p/∂(θ,η)²` from the `Dual2<M>` individual parameters,
/// where dual dimension `m` is `θ_m` (`m < n_theta`) and `n_theta + k` is `η_k`.
pub(crate) fn pd_from_program<const M: usize>(
    prog: &crate::parser::model_parser::IndivParamProgram,
    model: &CompiledModel,
    cov: &std::collections::HashMap<String, f64>,
    theta: &[f64],
    eta: &[f64],
) -> ParamDerivs {
    let p = prog.eval_param_duals::<M>(theta, eta, cov);
    let nt = model.n_theta;
    let ne = model.n_eta;
    let ni = p.len();
    let mut dp_deta = vec![vec![0.0; ne]; ni];
    let mut dp_dtheta = vec![vec![0.0; nt]; ni];
    let mut d2p_deta2 = vec![vec![vec![0.0; ne]; ne]; ni];
    let mut d2p_detadtheta = vec![vec![vec![0.0; nt]; ne]; ni];
    for i in 0..ni {
        let g = &p[i].grad;
        let h = &p[i].hess;
        for k in 0..ne {
            dp_deta[i][k] = g[nt + k];
        }
        for m in 0..nt {
            dp_dtheta[i][m] = g[m];
        }
        for k in 0..ne {
            for l in 0..ne {
                d2p_deta2[i][k][l] = h[nt + k][nt + l];
            }
            for m in 0..nt {
                d2p_detadtheta[i][k][m] = h[nt + k][m];
            }
        }
    }
    ParamDerivs {
        dp_deta,
        dp_dtheta,
        d2p_deta2,
        d2p_detadtheta,
    }
}

/// Central finite differences of `init_fn` over the differentiated PK slots — the shared
/// stencil behind [`dual_init_state`], [`dual1_init_state`], and [`tvcov_init_state`], so the
/// `he`/`h2` steps and the 2-/3-/4-point formulas live in **one** place. Returns
/// `(base, d1, d2)` where `d1[s][i] = ∂init_s/∂p_{pk_indices[i]}` and, when `want_d2`, the
/// symmetric `d2[s][i][j] = ∂²init_s/∂p_i∂p_j`; `d2` is empty otherwise (gradient-only
/// callers). `init_fn` is a cheap HashMap eval, so the FD cost is negligible.
fn init_fd_derivs(
    init_fn: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    pk: &[f64],
    pk_indices: &[usize],
    n_states: usize,
    want_d2: bool,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
    let he = 1e-6;
    let h2 = 1e-4;
    let np = pk_indices.len();
    let base = init_fn(pk);
    let mut d1 = vec![vec![0.0; np]; n_states];
    for (i, &si) in pk_indices.iter().enumerate() {
        let mut pp = pk.to_vec();
        pp[si] += he;
        let mut pm = pk.to_vec();
        pm[si] -= he;
        let (up, dn) = (init_fn(&pp), init_fn(&pm));
        for s in 0..n_states {
            d1[s][i] = (up[s] - dn[s]) / (2.0 * he);
        }
    }
    let mut d2: Vec<Vec<Vec<f64>>> = Vec::new();
    if want_d2 {
        d2 = vec![vec![vec![0.0; np]; np]; n_states];
        for (i, &si) in pk_indices.iter().enumerate() {
            let mut pp = pk.to_vec();
            pp[si] += h2;
            let mut pm = pk.to_vec();
            pm[si] -= h2;
            let (up, dn) = (init_fn(&pp), init_fn(&pm));
            for s in 0..n_states {
                d2[s][i][i] = (up[s] - 2.0 * base[s] + dn[s]) / (h2 * h2);
            }
            for (j, &sj) in pk_indices.iter().enumerate().skip(i + 1) {
                let mut a = pk.to_vec();
                a[si] += h2;
                a[sj] += h2;
                let mut b = pk.to_vec();
                b[si] += h2;
                b[sj] -= h2;
                let mut c = pk.to_vec();
                c[si] -= h2;
                c[sj] += h2;
                let mut d = pk.to_vec();
                d[si] -= h2;
                d[sj] -= h2;
                let (va, vb, vc, vd) = (init_fn(&a), init_fn(&b), init_fn(&c), init_fn(&d));
                for s in 0..n_states {
                    let v = (va[s] - vb[s] - vc[s] + vd[s]) / (4.0 * h2 * h2);
                    d2[s][i][j] = v;
                    d2[s][j][i] = v;
                }
            }
        }
    }
    (base, d1, d2)
}

/// The `DualMixed<NA, N>` initial state from a model's `init(...)` directives,
/// seeding each compartment's value **and its PK-parameter derivatives** (via the shared
/// [`init_fd_derivs`] stencil) mapped onto the dual axes.
///
/// Individual parameter `i` seeds dual axis `axis_of[i]` — identity (`= i`) for the
/// full `Dual2<N>` path (`NA == N`, `axis_of == None`), or the IIV-leading
/// permutation for the mixed path (#445). The first-order block fills the gradient
/// at every axis; the second-order block fills only the retained Hessian rows
/// (axis `< NA`) — the dropped rows are computed but not written (`init_fn` is cheap).
fn dual_init_state<const NA: usize, const N: usize>(
    init_fn: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    pk: &[f64],
    pk_indices: &[usize],
    n_states: usize,
    axis_of: Option<&[usize]>,
) -> Vec<DualMixed<NA, N>> {
    let ax = |i: usize| axis_of.map_or(i, |p| p[i]);
    let (base, d1, d2) = init_fd_derivs(init_fn, pk, pk_indices, n_states, true);
    let np = pk_indices.len();
    let mut out: Vec<DualMixed<NA, N>> = (0..n_states)
        .map(|s| DualMixed::constant(base.get(s).copied().unwrap_or(0.0)))
        .collect();

    // First order: gradient at every parameter's axis.
    for i in 0..np {
        let ai = ax(i);
        for s in 0..n_states {
            out[s].grad[ai] = d1[s][i];
        }
    }
    // Second order: only the retained Hessian rows (axis < NA).
    for i in 0..np {
        let ai = ax(i);
        if ai < NA {
            for s in 0..n_states {
                out[s].hess[ai][ai] = d2[s][i][i];
            }
        }
        for j in (i + 1)..np {
            let aj = ax(j);
            if ai >= NA && aj >= NA {
                continue;
            }
            for s in 0..n_states {
                let v = d2[s][i][j];
                if ai < NA {
                    out[s].hess[ai][aj] = v;
                }
                if aj < NA {
                    out[s].hess[aj][ai] = v;
                }
            }
        }
    }
    out
}

/// `Dual1<N>` initial state from a model's `init(...)` directives (gradient only) —
/// the light counterpart of [`dual_init_state`]: seeds each compartment's value and
/// its first-order PK-parameter derivatives via the shared [`init_fd_derivs`] stencil
/// (skipping the second-order block).
fn dual1_init_state<const N: usize>(
    init_fn: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    pk: &[f64],
    pk_indices: &[usize],
    n_states: usize,
) -> Vec<Dual1<N>> {
    let (base, d1, _) = init_fd_derivs(init_fn, pk, pk_indices, n_states, false);
    let mut out: Vec<Dual1<N>> = (0..n_states)
        .map(|s| Dual1::constant(base.get(s).copied().unwrap_or(0.0)))
        .collect();
    for i in 0..pk_indices.len() {
        for s in 0..n_states {
            out[s].grad[i] = d1[s][i];
        }
    }
    out
}

/// Seed the ODE `init(...)` initial state for the **event-driven TV-cov walk**, on the
/// walk's own dual basis (`Dual2<M>` outer / `Dual1<N>` inner — whatever `T` the caller
/// propagates). Unlike the static walk, which seeds `init` on the PK-parameter axes
/// ([`dual_init_state`]) and chains to (θ,η) at the readout, the TV-cov walk propagates
/// its state on (θ,η) directly (`seed_pk_dual2`/`seed_pk_dual1`), so `init` must be
/// seeded on that basis here — and is consumed by [`integrate_tvcov_g`] as the initial
/// state (`ode_tvcov_supported` keeps `init(...)` + EVID 3/4 reset on FD, since a reset
/// re-seeds production at the reset-event snapshot but the walk carries one subject jet).
///
/// Production seeds `init` at the subject's **first record** covariate snapshot
/// (`init_pk`, `predictions.rs`: smallest record time across dose / obs / pk-only), so
/// we pick that snapshot's already-seeded PK duals `p` and expand `init` as a
/// second-order Taylor series in the PK params, evaluated with `T` arithmetic:
///
/// ```text
/// init_s ≈ init_fn(p̄)_s + Σ_i (∂init_s/∂p_i)·(p_i − p̄_i)
///                       + ½ Σ_ij (∂²init_s/∂p_i∂p_j)·(p_i − p̄_i)(p_j − p̄_j)
/// ```
///
/// The `∂init/∂p` derivatives come from the shared [`init_fd_derivs`] stencil. Each delta
/// `(p_i − p̄_i)` has value 0 and carries the snapshot's θ/η jet, so `T`'s own product rule
/// propagates it to the walk basis exactly. A `Dual1` inner walk (`T::SECOND_ORDER == false`)
/// **skips the second-order block and term entirely** — its ε² is zero, so the quadratic
/// term would contribute nothing — leaving the same first-order η-gradient the `Dual2` outer
/// walk carries. All-zero when the subject has no records (defensive).
fn tvcov_init_state<T: crate::sens::num::PkNum>(
    init_fn: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    subject: &Subject,
    pk_at_dose: &[Vec<T>],
    pk_at_obs: &[Vec<T>],
    pk_at_pk_only: &[Vec<T>],
    pk_indices: &[usize],
    n_states: usize,
) -> Vec<T> {
    // First-record snapshot (smallest record time), matching production's `init_pk` selection.
    let snap = match first_record_pk::<T>(subject, pk_at_dose, pk_at_obs, pk_at_pk_only) {
        Some(p) => p,
        None => return vec![T::from_f64(0.0); n_states],
    };
    init_taylor_seed_at::<T>(init_fn, snap, pk_indices, n_states)
}

/// The PK-param snapshot at the subject's **first record** (smallest record time), matching
/// production's `init_pk` selection (`ode/predictions.rs`) and its `last_pk` seed for a reset
/// that is itself the first event. Doses / obs / pk-only are scanned in that order under a
/// strict `t < best_t`, so at an exact tie the earliest-considered snapshot wins — dose over
/// obs over pk-only, identical to production's `consider`. `None` when the subject has no
/// records. Shared by [`tvcov_init_state`] (the initial `init` seed) and the event-driven
/// walk's `last_params` seed (the reset re-seed's snapshot, #486 review) so the two can't drift.
fn first_record_pk<'a, T: crate::sens::num::PkNum>(
    subject: &Subject,
    pk_at_dose: &'a [Vec<T>],
    pk_at_obs: &'a [Vec<T>],
    pk_at_pk_only: &'a [Vec<T>],
) -> Option<&'a [T]> {
    let mut best_t = f64::INFINITY;
    let mut best: Option<&[T]> = None;
    for (k, d) in subject.doses.iter().enumerate() {
        if d.time < best_t {
            best_t = d.time;
            best = Some(&pk_at_dose[k]);
        }
    }
    for (j, &t) in subject.obs_times.iter().enumerate() {
        if t < best_t {
            best_t = t;
            best = Some(&pk_at_obs[j]);
        }
    }
    for (m, &t) in subject.pk_only_times.iter().enumerate() {
        if t < best_t {
            best_t = t;
            best = Some(&pk_at_pk_only[m]);
        }
    }
    best
}

/// Seed the ODE `init(...)` state on the walk's own dual basis from a **given** PK-param
/// snapshot `snap` (already seeded on `(θ, η[, κ])`), as a second-order Taylor of `init` in the
/// PK params about `snap.val()`. Shared by [`tvcov_init_state`] (which selects the first-record
/// snapshot, production's `init_pk`) and the event-driven walk's **EVID 3/4 reset** re-seed
/// (which uses the reset-event snapshot — production's `last_pk`, #486): production re-applies
/// `init(&last_pk.values)` at each reset, so the dual walk must rebuild the seed from the params
/// in effect there, not carry the one subject-level jet.
///
/// The `∂init/∂p` derivatives come from the shared [`init_fd_derivs`] stencil. Each delta
/// `(p_i − p̄_i)` has value 0 and carries the snapshot's θ/η[/κ] jet, so `T`'s own product rule
/// propagates it to the walk basis exactly. A `Dual1` inner walk (`T::SECOND_ORDER == false`)
/// **skips the second-order block and term entirely** — its ε² is zero, so the quadratic term
/// contributes nothing — leaving the same first-order gradient the `Dual2` outer walk carries.
fn init_taylor_seed_at<T: crate::sens::num::PkNum>(
    init_fn: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    snap: &[T],
    pk_indices: &[usize],
    n_states: usize,
) -> Vec<T> {
    // Central FD of `init_fn` at that snapshot via the shared stencil. The inner `Dual1`
    // walk (`T::SECOND_ORDER == false`) discards the quadratic term, so skip the `d2` block
    // entirely there rather than build a Hessian it drops.
    let want_2nd = T::SECOND_ORDER;
    let pk_vals: Vec<f64> = snap.iter().map(|x| x.val()).collect();
    let (base, d1, d2) = init_fd_derivs(init_fn, &pk_vals, pk_indices, n_states, want_2nd);
    let np = pk_indices.len();

    // Deltas δ_i = p_i − p̄_i (value 0, carry the snapshot's θ/η jets).
    let deltas: Vec<T> = pk_indices
        .iter()
        .map(|&si| snap[si] - T::from_f64(pk_vals[si]))
        .collect();

    // Taylor-compose on the walk basis via `T` arithmetic:
    //   init_s ≈ base_s + Σ_i d1_i·δ_i + ½ Σ_i d2_ii·δ_i² + Σ_{i<j} d2_ij·δ_iδ_j
    // The linear term contributes `Σ_i d1_i H_i` to the Hessian; the diagonal ½·δ_i² gives
    // `d2_ii g_i⊗g_i` and each off-diagonal `δ_iδ_j` gives `g_i⊗g_j + g_j⊗g_i` — visiting
    // `i<j` once (with the full `d2_ij` coefficient, `d2` symmetric) yields the exact
    // `Σ_ij d2_ij g_i⊗g_j` at half the O(np²·M²) Dual2 work of the all-ordered form.
    let mut out = Vec::with_capacity(n_states);
    for s in 0..n_states {
        let mut acc = T::from_f64(base.get(s).copied().unwrap_or(0.0));
        for i in 0..np {
            if d1[s][i] != 0.0 {
                acc = acc + T::from_f64(d1[s][i]) * deltas[i];
            }
        }
        if want_2nd {
            for i in 0..np {
                let cd = 0.5 * d2[s][i][i];
                if cd != 0.0 {
                    acc = acc + T::from_f64(cd) * deltas[i] * deltas[i];
                }
                for j in (i + 1)..np {
                    let c = d2[s][i][j];
                    if c != 0.0 {
                        acc = acc + T::from_f64(c) * deltas[i] * deltas[j];
                    }
                }
            }
        }
        out.push(acc);
    }
    out
}

/// Apply the model's output transforms to a dual prediction, in PK-parameter dual
/// space (before the η/θ chain): a constant `ScalarScale` divisor `f/k` and/or the
/// LTBS log `ln(max(f, floor))`. Both are smooth functions of the prediction, so the
/// `Dual2` ops carry `∂f/∂pk` and `∂²f/∂pk²` exactly — the η/θ chain that follows is
/// unchanged. An η-dependent `ExpressionScale` divisor is NOT applied here — it can
/// reference θ/η directly (not only PK params), so it is applied on the final
/// `(θ,η)`-space jet after the chain (`apply_expression_scale_outer`, #486). Mirrors
/// `pk::apply_scaling` (`pred /= k`) and `pk::apply_log_transform`
/// (`p = max(p, LTBS_FLOOR).ln()`; below the floor the value is clamped to a
/// constant, so the jet vanishes).
fn apply_output_transform<T: crate::sens::num::PkNum>(model: &CompiledModel, p: T) -> T {
    // A NaN readout (e.g. a per-CMT map miss — rejected upstream by fit-time
    // `validate_per_cmt_scaling`, so unreachable in a real fit) must stay NaN as a
    // visible tripwire: neither the `ScalarScale` divisor nor the LTBS floor below
    // may silently convert it to a finite value with zero derivatives (#449 review).
    if p.val().is_nan() {
        return p;
    }
    // `ScalarScale` is the only scaling the gate (`ode_analytical_supported`) admits
    // over duals — `ExpressionScale`/`PerCmt` scaling route to the Form-C `y = state/V`
    // readout instead, so production's full `build_obs_scale_array` need not be lifted
    // here. Divide (not multiply by `1/k`) to match production's `pred /= s` exactly.
    let p = match model.scaling {
        ScalingSpec::ScalarScale(k) if k != 1.0 => p / T::from_f64(k),
        _ => p,
    };
    // LTBS log. The value goes through the shared generic transform — the same
    // floor-then-log production runs on f64 (#451); the `NaN` pre-check above keeps a
    // `NaN` readout visible rather than letting the floor convert it to `ln(LTBS_FLOOR)`.
    // The gradient keys on the strict `> LTBS_FLOOR` boundary, matching the analytical
    // path (`provider.rs`) and `apply_log_transform`'s clamp semantics: at or below the
    // floor the readout is clamped to a constant so its derivatives vanish, rather than
    // `guard_floor` retaining the jet exactly at the floor (#460 review). Above the
    // floor the dual `ln` carries the jet (and `ltbs_log_g` is then just `p.ln()`).
    if model.log_transform {
        if p.val() > crate::pk::LTBS_FLOOR {
            crate::pk::ltbs_log_g(p)
        } else {
            T::from_f64(crate::pk::ltbs_log_g(p.val()))
        }
    } else {
        p
    }
}

/// Resolve the ODE readout for observation `j` over the dual state `st`, then apply
/// the negative-readout clamp and the output transform — the single readout site
/// shared by the static [`integrate_subject_duals`] and the TV-cov
/// [`integrate_tvcov_readout`] walks (#449 re-review #7). `params` is the flat
/// PK-param dual vector the Form-C / per-CMT program reads: `params_dual` for the
/// static walk, the per-event `pk_at_obs[j]` snapshot for the TV-cov walk. The
/// `ObsCmt` arm ignores it (reads the state compartment directly).
fn resolve_obs_readout<T: crate::sens::num::PkNum>(
    model: &CompiledModel,
    ode: &crate::ode::OdeSpec,
    subject: &Subject,
    st: &[T],
    j: usize,
    params: &[T],
    ro_vars: &mut Vec<T>,
    ro_stack: &mut Vec<T>,
) -> T {
    // Per-observation covariate snapshot the readout's covariate references read
    // (#540); for static covariates this is the subject map. Threaded as constants
    // — a covariate carries no derivative in the individual-parameter dual basis.
    let obs_cov = subject.obs_cov(j);
    let raw = match &ode.readout {
        OdeReadout::ObsCmt(idx) => st.get(*idx).copied().unwrap_or(T::from_f64(0.0)),
        OdeReadout::Single(_) => ode
            .readout_program
            .as_ref()
            .map(|p| p.eval_output_g::<T>(st, params, obs_cov, ro_vars, ro_stack))
            .unwrap_or(T::from_f64(0.0)),
        // Per-CMT (#439): observation j reads its own CMT's output program.
        OdeReadout::PerCmt(cmt_map) => subject
            .obs_cmts
            .get(j)
            .and_then(|cmt| cmt_map.get(cmt))
            .and_then(|r| r.program.as_ref())
            .map(|p| p.eval_output_g::<T>(st, params, obs_cov, ro_vars, ro_stack))
            .unwrap_or(T::from_f64(f64::NAN)),
    };
    // Negative-readout clamp (ODE overshoot guard), parity with production's
    // `conc.max(0)` (predictions.rs) and the dual walks: a clamped value carries zero
    // derivatives. A NaN readout is `< 0.0` → false, so it passes through and
    // `apply_output_transform` preserves it as a tripwire (#449 review).
    let raw = if raw.val() < 0.0 {
        T::from_f64(0.0)
    } else {
        raw
    };
    apply_output_transform::<T>(model, raw)
}

/// PK-parameter slot holding bioavailability `F` for a dose into 1-based `cmt`:
/// the compartment-indexed `F{cmt}` slot when the model declared one (#369), else
/// the bare [`PK_IDX_F`] (default 1.0). Mirrors production's
/// [`DoseAttrMap::f_bio`](crate::types::DoseAttrMap::f_bio) slot resolution so the
/// dual walk applies the same per-compartment `F` the f64 predictor does —
/// `params_dual[slot]` then carries `∂/∂F{cmt}`, since an indexed `F{cmt}` is an
/// ordinary seeded individual parameter (#486).
fn f_bio_slot(ode: &OdeSpec, cmt: usize) -> usize {
    ode.dose_attr_map
        .indexed_slot(crate::types::DoseAttr::F, cmt)
        .unwrap_or(PK_IDX_F)
}

/// Shared setup for both ODE drivers, generic over the dual type `T` (`Dual2<N>`
/// for the full outer walk, `Dual1<N>` for the light inner η-gradient): seed the
/// flat PK-parameter vector (individual parameter `i` → dual dimension `i`),
/// resolve bioavailability `F`, integrate the augmented state through the subject's
/// events, and apply the readout + output transforms per observation. `init_state`
/// is supplied by the caller (its FD seeding is order-specific:
/// [`dual_init_state`] carries the Hessian, [`dual1_init_state`] only the gradient).
/// Returns one transformed prediction `T` per observation — the caller reads its
/// `grad`/`hess` and chains with `∂p/∂(η[,θ])`. `None` on lagtime (not yet supported
/// over the dual loop) or an integration that fails to record every observation.
fn integrate_subject_duals<T: crate::sens::num::PkNum>(
    model: &CompiledModel,
    subject: &Subject,
    pk_values: &[f64],
    init_state: &[T],
    axis_of: Option<&[usize]>,
) -> Option<Vec<T>> {
    let ode = model.ode_spec.as_ref()?;
    let program = ode.rhs_program.as_ref()?;
    let opts = ode.solver_opts;

    // Seed the flat PK-parameter vector: individual parameter `i` (PK slot
    // `pk_indices[i]`) carries dual axis `axis_of[i]` — identity (`= i`) for the full
    // `Dual2`/`Dual1` paths, or the IIV-leading permutation for the mixed-order
    // `DualMixed` path (#445); everything else is constant.
    let mut params_dual: Vec<T> = pk_values.iter().map(|&v| T::from_f64(v)).collect();
    for (i, &slot) in model.pk_indices.iter().enumerate() {
        let ax = axis_of.map_or(i, |p| p[i]);
        params_dual[slot] = T::var(pk_values[slot], ax);
    }

    // An estimated lagtime routes to the event-driven walk (`integrate_tvcov_g`), where
    // the per-dose event-time saltation handles it — never this static superposition walk.
    // The `PK_IDX_LAGTIME` guard is a defensive backstop (→ FD) for any bare-lag subject
    // that reached here: the static dual loop applies no dose-time shift (#451 / #472).
    if pk_values[PK_IDX_LAGTIME].abs() > 1e-12 {
        return None;
    }
    // Bioavailability F scales the dosed amount/rate (NONMEM F·AMT / F·RATE), resolved
    // *per dose compartment*: `F{cmt}` if the model declared a compartment-indexed
    // bioavailability for that dose's compartment, else the bare `PK_IDX_F` (#369 / #486).
    // When F is an estimated individual parameter its derivative flows via
    // `params_dual[slot]`. Use the raw slot — mirroring production's `DoseAttrMap::f_bio`
    // (the 1.0 default baked into the bare slot at construction) — so a transient F ≤ 0
    // mid-fit scales the dose by F exactly as the f64 predictor does, rather than
    // substituting 1.0 and dropping ∂/∂F (#451 / #433 review #3).
    let dose_f_bio: Vec<T> = subject
        .doses
        .iter()
        .map(|d| params_dual[f_bio_slot(ode, d.cmt)])
        .collect();

    // Dose-time anchors for TAFD/TAD (constants w.r.t. the parameters).
    let first_dose_time = subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min);

    // Built-in absorption input-rate forcings (#430), parallel to `ode.input_rate`,
    // built over the dual type `T` (so they thread through `Dual2`/`Dual1`/`DualMixed`
    // alike). The gate (`ode_analytical_supported`) admits only kinds lifted to
    // `PkNum`, so `prepare_dual` returns `Some` for each; `?` bails to FD otherwise.
    let mut prepared_forcings: Vec<PreparedInputRate<T>> = Vec::with_capacity(ode.input_rate.len());
    for f in &ode.input_rate {
        prepared_forcings.push(f.prepare_dual::<T>(&params_dual)?);
    }

    // Integrate the dual state through bolus + infusion + absorption-forcing events,
    // capturing the full state at each observation time.
    let states = integrate_g::<T>(
        program,
        ode.n_states,
        subject,
        ode,
        &prepared_forcings,
        &params_dual,
        &dose_f_bio,
        init_state,
        first_dose_time,
        &opts,
    )?;

    // Apply the readout per observation, then the output transforms (`ScalarScale`
    // divisor / LTBS log). The static walk reads every observation against the same
    // `params_dual`.
    let mut ro_vars: Vec<T> = Vec::new();
    let mut ro_stack: Vec<T> = Vec::new();
    let preds: Vec<T> = states
        .iter()
        .enumerate()
        .map(|(j, st)| {
            resolve_obs_readout::<T>(
                model,
                ode,
                subject,
                st,
                j,
                &params_dual,
                &mut ro_vars,
                &mut ro_stack,
            )
        })
        .collect();

    Some(preds)
}

fn run_subject<const N: usize>(
    model: &CompiledModel,
    subject: &Subject,
    _theta: &[f64],
    _eta: &[f64],
    pk_values: &[f64],
    pd: &ParamDerivs,
) -> Option<SubjectSens> {
    let ode = model.ode_spec.as_ref()?;
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;

    // `pk_values` (PK params at (θ, η)) and `pd` (∂p/∂(θ,η) + 2nd order) are both
    // supplied by the dispatcher — already evaluated there for the lagtime check and
    // the IIV-axis classification — so neither is recomputed here (#445 review #8).

    // Initial state from `init(...)` (dual-seeded by FD of init_fn, value + grad +
    // Hessian); zeros when none is declared. Re-applied at every EVID 3/4 reset.
    let init_state: Vec<Dual2<N>> = match ode.init_fn.as_ref() {
        Some(f) => {
            dual_init_state::<N, N>(f.as_ref(), pk_values, &model.pk_indices, ode.n_states, None)
        }
        None => vec![Dual2::constant(0.0); ode.n_states],
    };

    // Seed + integrate the Dual2 state and apply the readout/transforms.
    let preds = integrate_subject_duals::<Dual2<N>>(model, subject, pk_values, &init_state, None)?;

    // Chain ∂f/∂p, ∂²f/∂p² (exact, from the dual) with ∂p/∂η, ∂p/∂θ (general,
    // from `param_derivatives`) → ∂f/∂η, ∂²f/∂η², ∂f/∂θ, ∂²f/∂η∂θ:
    //   ∂f/∂η_k        = Σ_i  g_i · pᵢ,η_k
    //   ∂²f/∂η_k∂η_l   = Σ_ij h_ij · pᵢ,η_k · pⱼ,η_l  +  Σ_i g_i · pᵢ,η_kη_l
    // and likewise with θ in one slot.
    let n_indiv = model.pk_indices.len();
    let mut out = Vec::with_capacity(subject.obs_times.len());
    for fd in &preds {
        let g = &fd.grad; // ∂f/∂p_i
        let h = &fd.hess; // ∂²f/∂p_i∂p_j

        let mut df_deta = vec![0.0; n_eta];
        let mut d2f_deta2 = vec![0.0; n_eta * n_eta];
        let mut df_dtheta = vec![0.0; n_theta];
        let mut d2f_deta_dtheta = vec![0.0; n_eta * n_theta];

        for i in 0..n_indiv {
            for k in 0..n_eta {
                df_deta[k] += g[i] * pd.dp_deta[i][k];
            }
            for m in 0..n_theta {
                df_dtheta[m] += g[i] * pd.dp_dtheta[i][m];
            }
        }
        for k in 0..n_eta {
            for l in 0..n_eta {
                let mut acc = 0.0;
                for i in 0..n_indiv {
                    for j in 0..n_indiv {
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
                for i in 0..n_indiv {
                    for j in 0..n_indiv {
                        acc += h[i][j] * pd.dp_deta[i][k] * pd.dp_dtheta[j][m];
                    }
                    acc += g[i] * pd.d2p_detadtheta[i][k][m];
                }
                d2f_deta_dtheta[k * n_theta + m] = acc;
            }
        }

        out.push(ObsSens {
            f: fd.value,
            df_deta,
            d2f_deta2,
            df_dtheta,
            d2f_deta_dtheta,
        });
    }

    Some(SubjectSens { obs: out })
}

/// Mixed-order variant of [`run_subject`] for models with IIV-free individual
/// parameters (issue #445). The integrated dual carries a full `N`-gradient but a
/// Hessian only over the `NA` IIV-bearing parameters, which are seeded as the
/// leading dual axes — `axis_of[i]` is the dual axis of individual parameter `i`
/// (IIV-bearing parameters occupy `0..NA`), and `iiv` lists the IIV-bearing `i`
/// (`iiv.len() == NA`). The result is numerically identical to `run_subject`: the
/// only entries skipped are the `∂²f/∂p_i∂p_j` with both `i, j` IIV-free, which the
/// η/θ chain never reads (FOCEI uses no `∂²f/∂θ²`).
#[allow(clippy::too_many_arguments)]
fn run_subject_mixed<const NA: usize, const N: usize>(
    model: &CompiledModel,
    subject: &Subject,
    _theta: &[f64],
    _eta: &[f64],
    pk_values: &[f64],
    pd: &ParamDerivs,
    axis_of: &[usize],
    iiv: &[usize],
) -> Option<SubjectSens> {
    let ode = model.ode_spec.as_ref()?;
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;

    // Contract from the dispatcher: the `NA` IIV-bearing parameters occupy dual axes
    // `0..NA` (`iiv.len() == NA`, `axis_of[i] < NA` for `i` in `iiv`). The chain's
    // `h[axis_of[i]][..]` access relies on it (#448 review #5).
    debug_assert!(
        iiv.len() == NA && iiv.iter().all(|&i| axis_of[i] < NA),
        "run_subject_mixed: the NA IIV-bearing parameters must occupy dual axes 0..NA"
    );

    // `pk_values` / `pd` are supplied by the dispatcher (already evaluated there for
    // the lagtime check + IIV classification). Initial state with the axis-mapped
    // seeding (IIV params on the leading Hessian rows).
    let init_state: Vec<DualMixed<NA, N>> = match ode.init_fn.as_ref() {
        Some(f) => dual_init_state::<NA, N>(
            f.as_ref(),
            pk_values,
            &model.pk_indices,
            ode.n_states,
            Some(axis_of),
        ),
        None => vec![DualMixed::constant(0.0); ode.n_states],
    };

    // Same shared seed → integrate → readout driver as `run_subject`/`run_subject_eta`,
    // with the IIV-leading axis permutation — no forked copy (#448 review #1).
    let preds = integrate_subject_duals::<DualMixed<NA, N>>(
        model,
        subject,
        pk_values,
        &init_state,
        Some(axis_of),
    )?;

    let n_indiv = model.pk_indices.len();
    let mut out = Vec::with_capacity(subject.obs_times.len());
    for fd in &preds {
        let g = &fd.grad; // ∂f/∂p_a, indexed by dual axis
        let h = &fd.hess; // ∂²f/∂p_a∂p_b: rows = IIV axes (0..NA), cols = all axes

        let mut df_deta = vec![0.0; n_eta];
        let mut d2f_deta2 = vec![0.0; n_eta * n_eta];
        let mut df_dtheta = vec![0.0; n_theta];
        let mut d2f_deta_dtheta = vec![0.0; n_eta * n_theta];

        // First order — every parameter contributes (the gradient spans all axes):
        //   ∂f/∂η_k = Σ_i g[axis_i]·(∂p_i/∂η_k),  ∂f/∂θ_m likewise.
        for i in 0..n_indiv {
            let ai = axis_of[i];
            for k in 0..n_eta {
                df_deta[k] += g[ai] * pd.dp_deta[i][k];
            }
            for m in 0..n_theta {
                df_dtheta[m] += g[ai] * pd.dp_dtheta[i][m];
            }
        }
        // Second-order `g·∂²p` terms — all parameters (these read only the gradient,
        // never a Hessian row, so they are safe for IIV-free parameters too).
        for i in 0..n_indiv {
            let ai = axis_of[i];
            for k in 0..n_eta {
                for l in 0..n_eta {
                    d2f_deta2[k * n_eta + l] += g[ai] * pd.d2p_deta2[i][k][l];
                }
                for m in 0..n_theta {
                    d2f_deta_dtheta[k * n_theta + m] += g[ai] * pd.d2p_detadtheta[i][k][m];
                }
            }
        }
        // Second-order `h·∂p·∂p` terms — the row index `i` always carries a `∂p/∂η`
        // factor, so it ranges only over the IIV-bearing parameters (`iiv`), whose
        // axes are `< NA` and therefore have a Hessian row in `h`. The column index
        // `j` ranges over all parameters (`h` has all `N` columns).
        for &i in iiv {
            let ai = axis_of[i];
            for j in 0..n_indiv {
                let hij = h[ai][axis_of[j]];
                for k in 0..n_eta {
                    for l in 0..n_eta {
                        d2f_deta2[k * n_eta + l] += hij * pd.dp_deta[i][k] * pd.dp_deta[j][l];
                    }
                    for m in 0..n_theta {
                        d2f_deta_dtheta[k * n_theta + m] +=
                            hij * pd.dp_deta[i][k] * pd.dp_dtheta[j][m];
                    }
                }
            }
        }

        out.push(ObsSens {
            f: fd.value,
            df_deta,
            d2f_deta2,
            df_dtheta,
            d2f_deta_dtheta,
        });
    }

    Some(SubjectSens { obs: out })
}

/// Light `Dual1<N>` driver: integrate the state carrying only first-order
/// `∂state/∂pk`, apply the readout + output transforms, and chain `∂f/∂pk · ∂pk/∂η`
/// → `∂f/∂η` (η only — no θ, no Hessian). The ODE counterpart of
/// [`super::provider`]'s `run_obs_grad`.
fn run_subject_eta<const N: usize>(
    model: &CompiledModel,
    subject: &Subject,
    pk: &crate::types::PkParams,
    dp_deta: &[Vec<f64>],
) -> Option<Vec<ObsGrad>> {
    let ode = model.ode_spec.as_ref()?;
    let n_eta = model.n_eta;

    // `pk` and the η-block `∂p/∂η` are evaluated once by the caller
    // (`ode_subject_eta_grad`) and threaded in, so the inner BFGS hot loop doesn't
    // recompute them per gradient evaluation (#534 review #3).

    // Initial state from `init(...)` (dual-seeded by FD of init_fn, value + grad);
    // zeros when none is declared. Re-applied at every EVID 3/4 reset.
    let init_state: Vec<Dual1<N>> = match ode.init_fn.as_ref() {
        Some(f) => dual1_init_state::<N>(f.as_ref(), &pk.values, &model.pk_indices, ode.n_states),
        None => vec![Dual1::constant(0.0); ode.n_states],
    };

    // Seed + integrate the Dual1 state and apply the readout/transforms.
    let preds = integrate_subject_duals::<Dual1<N>>(model, subject, &pk.values, &init_state, None)?;

    let n_indiv = model.pk_indices.len();
    let mut out = Vec::with_capacity(preds.len());
    for fd in &preds {
        // ∂f/∂η_k = Σ_i (∂f/∂pk_i)·(∂pk_i/∂η_k) — first order, η only.
        let g = &fd.grad;
        let mut df_deta = vec![0.0; n_eta];
        for i in 0..n_indiv {
            for k in 0..n_eta {
                df_deta[k] += g[i] * dp_deta[i][k];
            }
        }
        out.push(ObsGrad {
            f: fd.value,
            df_deta,
        });
    }
    Some(out)
}

/// Per-event flat PK-slot duals seeded on `(θ,η)` at a covariate snapshot — the ODE
/// analogue of the analytical `run_obs_tvcov`'s `mk`/`seed_row`. The PK slot for
/// individual parameter `i` carries `∂p/∂θ_m` on axis `m` and `∂p/∂η_k` on axis
/// `n_theta+k` (plus the η-η / η-θ 2nd-order blocks); every other slot is a
/// constant. The returned `Vec` is indexed by PK slot (what the ODE RHS reads).
fn seed_pk_dual2<const M: usize>(
    model: &CompiledModel,
    prog: &crate::parser::model_parser::IndivParamProgram,
    theta: &[f64],
    eta: &[f64],
    cov: &std::collections::HashMap<String, f64>,
    time: f64,
) -> Vec<Dual2<M>> {
    // Seed the model-time thread-local so a `TIME`-built-in structural parameter
    // resolves to this event's time in both the f64 values and the `Dual2` walk
    // (gated on `uses_time_builtin`, like the f64 `pk_param_fn` closure; #486).
    let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(
        crate::parser::model_parser::compiled_model_uses_time_builtin(model),
        time,
    );
    let n_theta = model.n_theta;
    let n_eta = model.n_eta;
    // The dispatch sizes `M = n_theta + n_eta` exactly (θ on axes `0..n_theta`, η on
    // `n_theta..M`), so the index guards are always satisfied — flat loops, no `< M`
    // / `.min(M)` (#449 review #15). The assert pins the invariant.
    debug_assert_eq!(M, n_theta + n_eta);
    // `pd` (the dual program eval) carries the individual-parameter *values* too, so
    // the separate `pk_param_fn` call below looks redundant (#451 re-review #9). It is
    // retained deliberately: `pk_param_fn` returns the **full** slot vector including
    // the non-individual-parameter slots (reserved `F`/lag defaults, etc.) that the
    // indiv-param program — hence `pd` — never produces. Reconstructing those from a
    // defaults base would re-encode `pk_param_fn`'s slot semantics here and risk silent
    // gradient divergence for any model that fills a non-indiv slot non-trivially,
    // while saving only the cheap f64 eval (the M²-Hessian dual eval dominates, and the
    // covariate-snapshot dedup already elides repeats). Not worth that trade.
    let pd = pd_from_program::<M>(prog, model, cov, theta, eta);
    let pk = (model.pk_param_fn)(theta, eta, cov, time);
    let mut out: Vec<Dual2<M>> = pk.values.iter().map(|&v| Dual2::constant(v)).collect();
    for (i, &slot) in model.pk_indices.iter().enumerate() {
        let mut grad = [0.0; M];
        let mut hess = [[0.0; M]; M];
        for m in 0..n_theta {
            grad[m] = pd.dp_dtheta[i][m];
        }
        for k in 0..n_eta {
            grad[n_theta + k] = pd.dp_deta[i][k];
            for l in 0..n_eta {
                hess[n_theta + k][n_theta + l] = pd.d2p_deta2[i][k][l];
            }
            for m in 0..n_theta {
                let v = pd.d2p_detadtheta[i][k][m];
                hess[n_theta + k][m] = v;
                hess[m][n_theta + k] = v;
            }
        }
        out[slot] = Dual2 {
            value: pk.values[slot],
            grad,
            hess,
        };
    }
    out
}

/// Shared TV-cov walk + readout for both ODE drivers, generic over the dual type
/// `T` (`Dual2<M>` outer, `Dual1<N>` inner): resolve per-dose bioavailability, run
/// the bolus event-driven walk over the per-event-seeded params, then per
/// observation apply the readout (`ObsCmt` / `Single` / per-CMT), the
/// negative-readout clamp, and the output transform. Returns one transformed
/// prediction per observation; the caller reads its `grad`/`hess` and chains. The
/// per-event PK-param duals are built by the caller, since the seeding differs by
/// order (`seed_pk_dual2` carries the Hessian, `seed_pk_dual1` only the gradient) —
/// the TV-cov analogue of [`integrate_subject_duals`] (#449 review #13).
fn integrate_tvcov_readout<T: crate::sens::num::PkNum>(
    model: &CompiledModel,
    subject: &Subject,
    pk_at_dose: &[Vec<T>],
    pk_at_obs: &[Vec<T>],
    pk_at_pk_only: &[Vec<T>],
) -> Vec<T> {
    // `ode_tvcov_supported` (checked by both TV-cov entry points before reaching
    // here) calls `ode_analytical_supported`, which declines a model whose `ode_spec`
    // or `rhs_program` is `None` — so both are guaranteed present and this readout is
    // infallible (the former `Option` return was dead) (#451 re-review #12).
    let ode = model
        .ode_spec
        .as_ref()
        .expect("ode_analytical_supported (via ode_tvcov_supported) guarantees ode_spec");
    let program = ode
        .rhs_program
        .as_ref()
        .expect("ode_analytical_supported (via ode_tvcov_supported) guarantees rhs_program");
    let opts = ode.solver_opts;

    // Per dose compartment, mirroring production's `DoseAttrMap::f_bio`: `F{cmt}` if
    // declared else the bare `PK_IDX_F` slot (#369 / #486), read from that dose's own
    // covariate snapshot `pk_at_dose[k]`. Raw slot (1.0 default baked in at
    // construction) — a transient F ≤ 0 scales the dose by F like the f64 predictor,
    // not 1.0 (#451 / #433 review #3).
    let f_bio_at_dose: Vec<T> = subject
        .doses
        .iter()
        .zip(pk_at_dose.iter())
        .map(|(d, p)| p[f_bio_slot(ode, d.cmt)])
        .collect();
    let first_dose_time = subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min);
    // Initial compartment state from `init(...)`, seeded on the walk's own dual basis at
    // the subject's first-record covariate snapshot (matching production `init_pk`); all
    // zeros when no `init(...)` is declared (#486). `ode_tvcov_supported` keeps `init` +
    // EVID 3/4 reset on FD, so a single subject-level init jet is exact here.
    let init_state: Vec<T> = match ode.init_fn.as_ref() {
        Some(f) => tvcov_init_state::<T>(
            f.as_ref(),
            subject,
            pk_at_dose,
            pk_at_obs,
            pk_at_pk_only,
            &model.pk_indices,
            ode.n_states,
        ),
        None => vec![T::from_f64(0.0); ode.n_states],
    };

    // Per-dose lagtime slot: the bare `PK_IDX_LAGTIME`, or a compartment-indexed
    // `ALAG{cmt}` slot when declared (#369). Empty when the model has no lagtime (the
    // walk then skips the dose-time shift / saltation entirely).
    let dose_lag_slot: Vec<usize> = if model.has_lagtime() {
        let attr_map = model.active_dose_attr_map();
        subject
            .doses
            .iter()
            .map(|d| attr_map.lag_slot(d.cmt))
            .collect()
    } else {
        Vec::new()
    };

    // #530: per-dose modeled-rate/duration slot. A `RATE=-2`/`D{cmt}` dose reads its window
    // length from the PK `D{cmt}` slot, a `RATE=-1`/`R{cmt}` dose its rate from the `R{cmt}`
    // slot — as a live `pk_at_dose[k]` jet so the moving infusion-end boundary carries
    // `∂/∂dur` (resp. `∂/∂rate`). `None` for a fixed dose (the walk then uses the resolved
    // `d.rate`/`d.duration` as before). The slot's existence is an invariant enforced by
    // `check_model_data`; mirrors the f64 `resolve_rate`, which drops the derivative. Empty
    // when every dose is fixed (byte-identical to the pre-#530 walk).
    let dose_modeled_slot: Vec<Option<(crate::types::RateMode, usize)>> =
        if subject.all_doses_fixed() {
            Vec::new()
        } else {
            let attr_map = model.active_dose_attr_map();
            subject
                .doses
                .iter()
                .map(|d| modeled_slot_for(attr_map, d))
                .collect()
        };

    let states = integrate_tvcov_g::<T>(
        program,
        ode,
        ode.n_states,
        subject,
        pk_at_dose,
        pk_at_obs,
        pk_at_pk_only,
        &f_bio_at_dose,
        &init_state,
        &model.pk_indices,
        first_dose_time,
        &dose_lag_slot,
        &dose_modeled_slot,
        &opts,
    );

    // Each observation reads against its own per-event covariate snapshot `pk_at_obs[j]`.
    let mut ro_vars: Vec<T> = Vec::new();
    let mut ro_stack: Vec<T> = Vec::new();
    let preds = states
        .iter()
        .enumerate()
        .map(|(j, st)| {
            resolve_obs_readout::<T>(
                model,
                ode,
                subject,
                st,
                j,
                &pk_at_obs[j],
                &mut ro_vars,
                &mut ro_stack,
            )
        })
        .collect();
    preds
}

/// Seed the per-event PK duals for a TV-cov subject's doses and observations,
/// deduplicating identical covariate snapshots. With TV covariates that change at
/// only a few breakpoints, most dose/obs events share a snapshot, so a full dual
/// eval per event re-does identical work; memoising by snapshot collapses that. The
/// seed is deterministic in the snapshot, so a cache hit is bit-identical to
/// re-seeding. The dose and obs vectors share one cache, so a snapshot common to
/// both is evaluated once. `seed` is fallible (`None` aborts the whole subject →
/// FD fallback); an infallible seeder wraps its result in `Some`.
///
/// One generic home for both the outer (`Dual2`) and inner (`Dual1`) TV-cov walks,
/// so the memoisation policy isn't maintained as two near-identical closures
/// (#451 re-review #8 / #451 review #3). The cache is a `HashMap` keyed on the
/// snapshot's canonical bit form — names sorted, values as `f64::to_bits` — giving
/// O(1) amortised lookup (not a linear scan) and, because `to_bits` is total, making a
/// snapshot with a missing (`NaN`) covariate deduplicate correctly: the seed is
/// deterministic in the snapshot, so sharing one bit-identical result is exactly a
/// re-seed (#460 review).
fn seed_tvcov_snapshots<T: Clone>(
    subject: &Subject,
    key_time: bool,
    mut seed: impl FnMut(&std::collections::HashMap<String, f64>, f64) -> Option<Vec<T>>,
) -> Option<(Vec<Vec<T>>, Vec<Vec<T>>, Vec<Vec<T>>)> {
    use std::collections::HashMap;
    // Canonical, hashable key for a covariate snapshot. `f64` is neither `Hash` nor
    // `Eq`, so key on `to_bits` (name-sorted); `to_bits` is total, so `NaN` keys are
    // well-defined and equal NaNs collapse — unlike `f64` `==`, which never matches a
    // `NaN` to itself (which left the old `Vec` cache scanning dead, unmatchable entries).
    fn snapshot_key(cov: &HashMap<String, f64>) -> Vec<(String, u64)> {
        let mut kv: Vec<(String, u64)> =
            cov.iter().map(|(k, v)| (k.clone(), v.to_bits())).collect();
        kv.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        kv
    }
    // When the model reads the `TIME` built-in, the seed depends on the event time as
    // well as the covariate snapshot, so two events with identical covariates at
    // different times must NOT share a seed — fold the event time into the cache key
    // (`key_time`). Non-TIME models leave it out, so TV-cov dedup is unchanged (#486
    // risk: dedup-cache key collision).
    let mut cache: HashMap<(Vec<(String, u64)>, Option<u64>), Vec<T>> = HashMap::new();
    let mut seed_for = |cov: &HashMap<String, f64>, time: f64| -> Option<Vec<T>> {
        let key = (snapshot_key(cov), key_time.then(|| time.to_bits()));
        if let Some(v) = cache.get(&key) {
            return Some(v.clone());
        }
        let v = seed(cov, time)?;
        cache.insert(key, v.clone());
        Some(v)
    };
    let pk_at_dose: Vec<Vec<T>> = (0..subject.doses.len())
        .map(|k| seed_for(subject.dose_cov(k), subject.doses[k].time))
        .collect::<Option<_>>()?;
    let pk_at_obs: Vec<Vec<T>> = (0..subject.obs_times.len())
        .map(|j| seed_for(subject.obs_cov(j), subject.obs_times[j]))
        .collect::<Option<_>>()?;
    let pk_at_pk_only: Vec<Vec<T>> = (0..subject.pk_only_times.len())
        .map(|m| seed_for(subject.pk_only_cov(m), subject.pk_only_times[m]))
        .collect::<Option<_>>()?;
    Some((pk_at_dose, pk_at_obs, pk_at_pk_only))
}

/// Time-varying-covariate outer (`Dual2<M>`, `M = n_theta + n_eta`) sensitivities
/// for an ODE model — the ODE counterpart of `run_obs_tvcov`. Seeds the per-event
/// PK params on `(θ,η)`, runs the shared TV-cov walk + readout, and reads
/// `∂f/∂(θ,η)` (+ 2nd order) straight off the dual (#439).
fn run_subject_tvcov<const M: usize>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<SubjectSens> {
    let ode = model.ode_spec.as_ref()?;
    let prog = ode.indiv_param_program.as_ref()?;
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;

    // Seed each event's per-snapshot PK duals via the shared dedup helper. When the
    // model reads `TIME`, fold the event time into the dedup key so events with equal
    // covariates but different times don't share a (time-dependent) seed (#486).
    // `seed_pk_dual2` is infallible, so wrap it in `Some`; the `?` never fires here.
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let (pk_at_dose, pk_at_obs, pk_at_pk_only) =
        seed_tvcov_snapshots::<Dual2<M>>(subject, uses_time, |cov, time| {
            Some(seed_pk_dual2::<M>(model, prog, theta, eta, cov, time))
        })?;

    let preds = integrate_tvcov_readout::<Dual2<M>>(
        model,
        subject,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    let mut out = Vec::with_capacity(preds.len());
    for fd in &preds {
        let g = &fd.grad;
        let h = &fd.hess;
        let mut df_deta = vec![0.0; n_eta];
        let mut df_dtheta = vec![0.0; n_theta];
        let mut d2f_deta2 = vec![0.0; n_eta * n_eta];
        let mut d2f_deta_dtheta = vec![0.0; n_eta * n_theta];
        for k in 0..n_eta {
            df_deta[k] = g[n_theta + k];
            for l in 0..n_eta {
                d2f_deta2[k * n_eta + l] = h[n_theta + k][n_theta + l];
            }
            for m in 0..n_theta {
                d2f_deta_dtheta[k * n_theta + m] = h[n_theta + k][m];
            }
        }
        for m in 0..n_theta {
            df_dtheta[m] = g[m];
        }
        out.push(ObsSens {
            f: fd.value,
            df_deta,
            d2f_deta2,
            df_dtheta,
            d2f_deta_dtheta,
        });
    }
    let mut sens = SubjectSens { obs: out };
    // η-dependent `ExpressionScale` divisor: apply the subject-static quotient on the
    // walked jet — the SAME shared helper the closed-form walk uses (#486, closing the
    // TV-cov + expression-scale gap on the ODE event-driven path; a `TIME`-built-in
    // parameter rides the same walk and composes with it). The scale is subject-static at
    // `t = 0`, mirrored inside the helper.
    crate::sens::provider::apply_event_walk_expression_scale_outer(
        &mut sens, model, subject, prog, theta, eta,
    )?;
    Some(sens)
}

/// Per-subject IOV scope + dimensions, shared by the outer (`Dual2`) and inner
/// (`Dual1`) ODE IOV walks so their analytic scope stays matched (a subject is served
/// analytically for both, or neither). Mirrors `ode_tvcov_supported`'s bolus-only
/// screen; time-varying covariates ARE supported (each event is seeded at its own
/// covariate snapshot). Returns `(occasion groups, n_stacked = n_eta + K·n_kappa,
/// m_dim = n_theta + n_stacked)`, or `None` out of scope.
///
/// The cap is on `m_dim` (the *outer* dual width); since `n_stacked ≤ m_dim`, the
/// inner walk's `Dual1<n_stacked>` always resolves too — so capping here keeps the
/// inner and outer on the same route (#439 ODE IOV).
fn ode_iov_subject_supported(
    model: &CompiledModel,
    subject: &Subject,
) -> Option<(Vec<(u32, Vec<usize>)>, usize, usize)> {
    if !ode_iov_supported(model) {
        return None;
    }
    // Single scan for the periodic steady-state predicate reused by every SS gate
    // below (the modeled-dose screen and the SS+lagtime / SS+time-RHS bails), rather
    // than re-scanning `subject.doses` per branch on this hot per-subject path.
    let has_ss = subject.has_periodic_ss_dose();
    // #530/#486: modeled-`RATE`/duration doses (`RATE=-1`/`-2`, `R{cmt}`/`D{cmt}`) are
    // resolved from their **per-occasion** PK slot as a live jet inside the event walk
    // (`inf_eff` reads `pk_at_dose[k][slot]`, seeded per occasion group by
    // `seed_pk_dual2_iov` so a `κ`-coupled modeled window — `D1 = TVD1·exp(η + κ)` —
    // lands its sensitivity in the correct stacked axis), so the moving infusion-end
    // boundary is analytic via the rate-off saltation, exactly as on the non-IOV TV-cov
    // walk (`ode_tvcov_supported`). Modeled-dose × steady-state IS now analytic too (#486):
    // `equilibrate_ss_state_g` threads the same `inf_eff` jet into its per-cycle active/quiet
    // split. Decline to FD only when a modeled dose's `D{cmt}`/`R{cmt}` slot is absent
    // (`check_model_data` rejects such a model, but emit FD rather than a wrong gradient if
    // one slips past — the walk's `inf_eff` would silently fall to the unresolved fixed arm).
    if !subject.all_doses_fixed() {
        let attr_map = model.active_dose_attr_map();
        let all_slots_present = subject.doses.iter().all(|d| {
            matches!(d.rate_mode, crate::types::RateMode::Fixed)
                || modeled_slot_for(attr_map, d).is_some()
        });
        if !all_slots_present {
            return None;
        }
    }
    // IOV + `ExpressionScale` `obs_scale` is served as a post-walk quotient. The scale
    // materialisation mirrors production `predict_iov`: one scale per occasion group,
    // evaluated at the subject-level covariate snapshot. A TV-cov subject may still use
    // this route; the event walk gets TV-cov PK params, while scaling follows the live
    // subject-static semantics (#590).
    // #419: rate-defined infusion under `F ≠ 1` is handled via the rate-off saltation
    // (moving window boundary) — including a steady-state rate-defined infusion (#486:
    // `equilibrate_ss_state_g`'s per-cycle active/quiet split reads the same `F`-scaled
    // `inf_eff` window as the main walk).
    // Steady-state (bolus and infusion) is handled via the dual equilibration. SS combined
    // with an estimated lagtime IS now analytic too (mirrors `ode_tvcov_supported`, #486):
    // the shared `integrate_tvcov_g` walk carries the `K_SS_SEED` pre-arrival seed, and the
    // dose's own arrival still goes through the unmodified general lagtime saltation,
    // regardless of outer/IOV dispatch.
    // SS combined with a non-autonomous RHS (reads `TIME`/`TAFD`/`TAD`) → FD: the SS
    // equilibration assumes a time-invariant pulse train, so the cycle recurrence breaks
    // (mirrors `ode_tvcov_supported`, #473 review #1).
    if has_ss
        && model
            .ode_spec
            .as_ref()
            .and_then(|o| o.rhs_program.as_ref())
            .is_some_and(|p| p.uses_time_vars())
    {
        return None;
    }
    // #486: a steady-state dose combined with a built-in absorption input-rate forcing
    // (`zero_order`/`first_order`, admitted under IOV by `ode_iov_supported`) is declined —
    // the dual SS equilibration (`equilibrate_ss_state_g`) seeds a bolus/infusion trough and
    // does not spread a periodic zero-order window (or a first-order R_in tail) over the
    // cycle, so its κ-coupled sensitivity is not yet validated. Non-SS input-rate + IOV is
    // the analytic scope this PR opens; SS × input-rate stays FD (as it effectively is on the
    // non-IOV walk for the same equilibration reason).
    if has_ss
        && model
            .ode_spec
            .as_ref()
            .is_some_and(|o| !o.input_rate.is_empty())
    {
        return None;
    }
    // EVID 3/4 resets, finite-duration infusions, and EVID=2 pk-only breakpoints are
    // handled by the event-driven walk.
    let occ_groups = crate::stats::likelihood::iov_occasion_groups(subject);
    let k_groups = occ_groups.len();
    if k_groups == 0 {
        return None;
    }
    let n_stacked = model.n_eta + k_groups * model.n_kappa;
    // Stacked dual width `M = n_theta + n_eta + K·n_kappa`. Bounded here (per subject,
    // since `K` is per subject) so an extremely many-occasion subject routes to FD rather
    // than a silent `_ => None` downgrade.
    let m_dim = model.n_theta + n_stacked;
    if !(1..=MAX_ODE_IOV_AXES).contains(&m_dim) {
        return None;
    }
    Some((occ_groups, n_stacked, m_dim))
}

/// Exact analytic sensitivities for an ODE **IOV** subject over the stacked
/// random-effects vector `[η_bsv, κ_group0, …, κ_group(K−1)]` (plus the θ block), or
/// `None` outside the supported scope (caller falls back to FD). The ODE counterpart
/// of [`crate::sens::provider::subject_sensitivities_iov`]; the returned [`SubjectSens`]
/// has the identical stacked layout, so the block-Ω (`Ω_bsv ⊕ K·Ω_iov`) assembly
/// consumes it unchanged. The inner EBE η-gradient is served analytically too
/// ([`ode_subject_eta_grad_iov`]), on the matched per-subject scope.
///
/// `stacked_eta` must have length `n_eta + K·n_kappa` with
/// `K = iov_occasion_groups(subject).len()` (#439 ODE IOV).
pub fn ode_subject_sensitivities_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_eta: &[f64],
) -> Option<SubjectSens> {
    let (occ_groups, n_stacked, m_dim) = ode_iov_subject_supported(model, subject)?;
    if stacked_eta.len() != n_stacked {
        return None;
    }
    dispatch_ode_iov_axes!(
        m_dim,
        run_subject_iov,
        model,
        subject,
        theta,
        stacked_eta,
        &occ_groups,
    )
}

/// Light **inner** η-gradient (`Dual1<N>`, `N = n_stacked = n_eta + K·n_kappa`) for an
/// ODE IOV subject — the IOV counterpart of [`run_subject_tvcov_eta`] and the inner
/// sibling of [`ode_subject_sensitivities_iov`]. Returns `∂f/∂(stacked-η)` per
/// observation (no θ block, no Hessian), or `None` outside the matched IOV scope. The
/// caller (`analytic_eta_nll_gradient_iov`) assembles the conditional-NLL gradient over
/// the stacked vector; the BSV columns also give the analytic FOCE H-matrix (#439 ODE IOV).
pub fn ode_subject_eta_grad_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_eta: &[f64],
) -> Option<Vec<ObsGrad>> {
    let (occ_groups, n_stacked, _m_dim) = ode_iov_subject_supported(model, subject)?;
    if stacked_eta.len() != n_stacked {
        return None;
    }
    dispatch_ode_iov_axes!(
        n_stacked,
        run_subject_iov_eta,
        model,
        subject,
        theta,
        stacked_eta,
        &occ_groups,
    )
}

/// Seed an occasion group's per-event PK-slot duals on the **stacked**
/// `(θ, η_bsv, κ)` axes from its [`CombinedDerivs`] — the IOV analogue of
/// [`seed_pk_dual2`]. The combined column `c` of the program maps to a stacked dual
/// axis: η_bsv (`c < n_eta`) → shared `n_theta + c`; κ (`c ≥ n_eta`) → group `g`'s
/// block `n_theta + n_eta + g·n_kappa + (c − n_eta)`. For EVID=2 pk-only events
/// `group` is `None`, matching production's zero-κ `combined_for(u32::MAX)`; κ columns
/// are then dropped. Non-individual-parameter slots are seeded as constants
/// (`pk.values`), exactly as the non-IOV seeder does. `cd` rows are parallel to
/// `model.pk_indices` (the program-eval row order shared with [`pd_from_program`]).
fn seed_pk_dual2_iov<const M: usize>(
    model: &CompiledModel,
    pk: &crate::types::PkParams,
    cd: &crate::sens::provider::CombinedDerivs,
    group: Option<usize>,
    n_eta: usize,
    n_kappa: usize,
    n_theta: usize,
) -> Vec<Dual2<M>> {
    let n_eff = n_eta + n_kappa;
    let kappa_base = group.map(|g| n_theta + n_eta + g * n_kappa);
    let stacked_axis = |c: usize| -> Option<usize> {
        if c < n_eta {
            Some(n_theta + c)
        } else {
            kappa_base.map(|base| base + (c - n_eta))
        }
    };
    let mut out: Vec<Dual2<M>> = pk.values.iter().map(|&v| Dual2::constant(v)).collect();
    for (i, &slot) in model.pk_indices.iter().enumerate() {
        let mut grad = [0.0; M];
        let mut hess = [[0.0; M]; M];
        for m in 0..n_theta.min(M) {
            grad[m] = cd.dtheta[i][m];
        }
        for c in 0..n_eff {
            let Some(ax) = stacked_axis(c) else {
                continue;
            };
            if ax >= M {
                continue;
            }
            grad[ax] = cd.deta[i][c];
            for d in 0..n_eff {
                if let Some(bx) = stacked_axis(d).filter(|&bx| bx < M) {
                    hess[ax][bx] = cd.d2eta[i][c][d];
                }
            }
            for m in 0..n_theta.min(M) {
                let v = cd.d2eta_theta[i][c][m];
                hess[ax][m] = v;
                hess[m][ax] = v;
            }
        }
        out[slot] = Dual2 {
            value: pk.values[slot],
            grad,
            hess,
        };
    }
    out
}

/// Build one `ExpressionScale` `obs_scale` jet per occasion group from per-group seeded
/// PK duals: gather the scale program's referenced PK slots (`slots`) into a scratch
/// buffer and evaluate the scale via `eval`. Generic over the dual type so the outer
/// (`Dual2`) and inner (`Dual1`) IOV walks share one jet-assembly loop instead of keeping
/// two copies in lockstep (#590 review). The per-type difference (`eval_scale_dual` vs
/// `eval_scale_dual1`) is supplied by the `eval` closure. Shared with the closed-form
/// IOV walk (`provider::run_obs_iov` / `run_obs_iov_eta`), which builds its per-group
/// seeded PK duals via its own `PkDual`-per-source `seed` closure and feeds them here
/// (#486).
pub(crate) fn build_iov_scale_jets<T: crate::sens::num::PkNum>(
    groups: &[Vec<T>],
    slots: &[usize],
    mut eval: impl FnMut(&[T]) -> T,
) -> Vec<T> {
    let mut jets = Vec::with_capacity(groups.len());
    let mut var_duals: Vec<T> = Vec::with_capacity(slots.len());
    for seeded in groups {
        var_duals.clear();
        var_duals.extend(
            slots
                .iter()
                .map(|&s| seeded.get(s).copied().unwrap_or_else(|| T::from_f64(0.0))),
        );
        jets.push(eval(&var_duals));
    }
    jets
}

/// Whether the IOV ODE walk must seed each event individually (its own occasion ×
/// covariate snapshot × time) rather than sharing one source per occasion group: TV
/// covariates, EVID=2 covariate breakpoints, or a `TIME`-built-in structural parameter.
/// Shared by `run_subject_iov` (outer `Dual2`) and `run_subject_iov_eta` (inner `Dual1`)
/// so their per-event seeding decisions can't desync (#637 round-2 review #4).
fn iov_walk_per_event(model: &CompiledModel, subject: &Subject) -> bool {
    subject.has_tv_covariates()
        || !subject.pk_only_covariates.is_empty()
        || crate::parser::model_parser::compiled_model_uses_time_builtin(model)
}

/// IOV outer (`Dual2<M>`, `M = n_theta + n_eta + K·n_kappa`) sensitivities for an ODE
/// model — the IOV counterpart of [`run_subject_tvcov`]. Seeds each event's stacked
/// PK duals at its (occasion, covariate-snapshot) — one source per occasion group when
/// covariates are static — maps each dose/observation to its source, runs the shared
/// event-driven walk +
/// readout ([`integrate_tvcov_readout`], which production's `predict_iov` mirrors by
/// feeding per-occasion params to the same `ode_predictions_event_driven`), and reads
/// `∂f/∂(θ, stacked-η)` (+ 2nd order) straight off the dual.
fn run_subject_iov<const M: usize>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_eta: &[f64],
    occ_groups: &[(u32, Vec<usize>)],
) -> Option<SubjectSens> {
    let ode = model.ode_spec.as_ref()?;
    let prog = ode.indiv_param_program.as_ref()?;
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let n_theta = model.n_theta;
    let n_eff = n_eta + n_kappa;
    let k_groups = occ_groups.len();
    let n_stacked = n_eta + k_groups * n_kappa;
    let cov = &subject.covariates;

    let occ_to_k = crate::stats::likelihood::iov_occ_to_k(&occ_groups);
    let combined_for =
        |g: usize| crate::stats::likelihood::iov_combined_effect(stacked_eta, n_eta, n_kappa, g);

    // Seed an occasion group's stacked PK duals at a covariate snapshot. `n_rows`
    // matches the program eval's `model.pk_indices`-parallel rows (the convention
    // `seed_pk_dual2` uses).
    // A `TIME`-built-in structural parameter resolves `Op::PushTime` from the model-time
    // thread-local, which `iov_combined_derivs`' walk reads; seed it with the per-event
    // time (gated on `uses_time`, like the f64 `pk_param_fn`) so each occasion's stacked
    // PK derivatives are evaluated at that event's TIME. It also forces per-event seeding
    // below (the switch fires per event even with static covariates) (#486).
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let n_rows = model.pk_indices.len();
    let seed_group_cov = |g: usize,
                          cov: &std::collections::HashMap<String, f64>,
                          time: f64|
     -> Option<Vec<Dual2<M>>> {
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
        let combined = combined_for(g);
        let pk = (model.pk_param_fn)(theta, &combined, cov, time);
        let cd = crate::sens::provider::iov_combined_derivs_dyn(
            prog, n_theta, n_eff, n_rows, cov, theta, &combined,
        )?;
        Some(seed_pk_dual2_iov::<M>(
            model,
            &pk,
            &cd,
            Some(g),
            n_eta,
            n_kappa,
            n_theta,
        ))
    };
    // EVID=2 pk-only events carry no occasion → κ held at 0 (single-sourced with the
    // closed-form provider, #598 review). Built lazily inside the closure so the common
    // IOV subject with no EVID=2 records pays no allocation — the closure is only invoked
    // when `seed_iov_events` actually has pk-only records to seed.
    let seed_pk_only_cov = |cov: &std::collections::HashMap<String, f64>,
                            time: f64|
     -> Option<Vec<Dual2<M>>> {
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
        let combined_pk_only =
            crate::stats::likelihood::iov_combined_pk_only(stacked_eta, n_eta, n_kappa);
        let pk = (model.pk_param_fn)(theta, &combined_pk_only, cov, time);
        let cd = crate::sens::provider::iov_combined_derivs_dyn(
            prog,
            n_theta,
            n_eff,
            n_rows,
            cov,
            theta,
            &combined_pk_only,
        )?;
        Some(seed_pk_dual2_iov::<M>(
            model, &pk, &cd, None, n_eta, n_kappa, n_theta,
        ))
    };

    // Build every occasion group's stacked PK seeding at the subject-static covariate
    // snapshot (`t = 0`). Shared by `static_group_dual` (the static-cov event-walk source)
    // and the `ExpressionScale` scale jets below — one source for both. For a `TIME` model
    // `static_group_dual` is `None` (per-event forced), but the scale-jet path still calls
    // this at `t = 0` — which matches production `apply_scaling` (also `t = 0`), so
    // `ExpressionScale` + TIME composes correctly under IOV.
    let build_all_groups = || -> Option<Vec<Vec<Dual2<M>>>> {
        (0..k_groups).map(|g| seed_group_cov(g, cov, 0.0)).collect()
    };

    // For static-covariate subjects the per-occasion-group stacked PK seeding is the same
    // source the event walk uses, so build it once here and share it with `seed_iov_events`
    // — no double seeding (#575 review). `None` for TV-cov OR `TIME` subjects (each event
    // seeds at its own snapshot/time in `seed_iov_events`).
    let per_event = iov_walk_per_event(model, subject);
    let static_group_dual: Option<Vec<Vec<Dual2<M>>>> = if per_event {
        None
    } else {
        Some(build_all_groups()?)
    };

    let (pk_at_dose, pk_at_obs, pk_at_pk_only) = seed_iov_events::<Dual2<M>>(
        subject,
        &occ_to_k,
        k_groups,
        per_event,
        cov,
        static_group_dual.as_deref(),
        seed_group_cov,
        seed_pk_only_cov,
    )?;

    // η-dependent `ExpressionScale` `obs_scale` divisor: one scale jet per occasion group,
    // matching production `predict_iov`'s subject-static `apply_scaling` call inside each
    // occasion. Static subjects reuse `static_group_dual`; TV-cov subjects seed a static-cov
    // scale jet for each group here.
    let group_scale: Option<Vec<Dual2<M>>> = match &model.scaling {
        ScalingSpec::ExpressionScale {
            deriv: Some(sprog), ..
        } => {
            let eta_bsv = &stacked_eta[..n_eta];
            let owned;
            let groups: &[Vec<Dual2<M>>] = match static_group_dual.as_deref() {
                Some(groups) => groups,
                None => {
                    owned = build_all_groups()?;
                    &owned
                }
            };
            Some(build_iov_scale_jets::<Dual2<M>>(
                groups,
                sprog.var_to_pk_slot(),
                |var_duals| sprog.eval_scale_dual::<M>(theta, eta_bsv, cov, var_duals),
            ))
        }
        _ => None,
    };

    let preds = integrate_tvcov_readout::<Dual2<M>>(
        model,
        subject,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    // Read `∂f/∂(θ, stacked-η)` (+ 2nd order) off the dual — the negative-readout clamp
    // and output transform are already applied inside `integrate_tvcov_readout`.
    let mut out = Vec::with_capacity(preds.len());
    for fd in &preds {
        let g = &fd.grad;
        let h = &fd.hess;
        let mut df_deta = vec![0.0; n_stacked];
        let mut df_dtheta = vec![0.0; n_theta];
        let mut d2f_deta2 = vec![0.0; n_stacked * n_stacked];
        let mut d2f_deta_dtheta = vec![0.0; n_stacked * n_theta];
        for p in 0..n_stacked {
            df_deta[p] = g[n_theta + p];
            for q in 0..n_stacked {
                d2f_deta2[p * n_stacked + q] = h[n_theta + p][n_theta + q];
            }
            for m in 0..n_theta {
                d2f_deta_dtheta[p * n_theta + m] = h[n_theta + p][m];
            }
        }
        for m in 0..n_theta {
            df_dtheta[m] = g[m];
        }
        out.push(ObsSens {
            f: fd.value,
            df_deta,
            d2f_deta2,
            df_dtheta,
            d2f_deta_dtheta,
        });
    }
    // Apply the `ExpressionScale` quotient per observation, using the observation's
    // occasion-group scale. Two scratch buffers reused across rows (not `2·n_obs` clones).
    if let Some(group_scale) = group_scale {
        let mut fk: Vec<f64> = Vec::with_capacity(n_stacked);
        let mut fm: Vec<f64> = Vec::with_capacity(n_theta);
        for (j, o) in out.iter_mut().enumerate() {
            let g = *occ_to_k.get(&subject.occasions.get(j).copied()?)?;
            crate::sens::provider::apply_scale_quotient_row::<M>(
                o,
                &group_scale[g],
                n_theta,
                n_stacked,
                &mut fk,
                &mut fm,
            );
        }
    }
    Some(SubjectSens { obs: out })
}

/// Map each dose/observation to its occasion group's seeded PK duals, generic over the
/// dual type `T` so the outer (`Dual2`) and inner (`Dual1`) IOV walks share one policy.
/// With time-varying covariates each event is seeded at its own (occasion, snapshot) —
/// the individual parameter switches both by κ (occasion) and by covariate; when
/// covariates are subject-static, one source per occasion group is built and shared,
/// preserving the non-TV cost (mirrors the analytical IOV provider). `seed_group_cov`
/// is fallible (`None` aborts the subject → FD fallback).
#[allow(clippy::too_many_arguments)]
fn seed_iov_events<T: Clone>(
    subject: &Subject,
    occ_to_k: &std::collections::HashMap<u32, usize>,
    k_groups: usize,
    per_event: bool,
    static_cov: &std::collections::HashMap<String, f64>,
    precomputed_static: Option<&[Vec<T>]>,
    mut seed_group_cov: impl FnMut(
        usize,
        &std::collections::HashMap<String, f64>,
        f64,
    ) -> Option<Vec<T>>,
    mut seed_pk_only_cov: impl FnMut(&std::collections::HashMap<String, f64>, f64) -> Option<Vec<T>>,
) -> Option<(Vec<Vec<T>>, Vec<Vec<T>>, Vec<Vec<T>>)> {
    // `per_event`: seed each event at its own (occasion, covariate snapshot, time)
    // rather than sharing one source per occasion group. True for TV covariates,
    // EVID=2 covariate breakpoints, OR a `TIME`-built-in structural parameter (the
    // switch fires per event even with static covariates, #486). Each event's time is
    // threaded to `seed_group_cov`/`seed_pk_only_cov` so `TIME` resolves per event.
    if per_event {
        let pk_at_dose = (0..subject.doses.len())
            .map(|d| {
                let g = *occ_to_k.get(&subject.dose_occasions.get(d).copied()?)?;
                seed_group_cov(g, subject.dose_cov(d), subject.doses[d].time)
            })
            .collect::<Option<_>>()?;
        let pk_at_obs = (0..subject.obs_times.len())
            .map(|j| {
                let g = *occ_to_k.get(&subject.occasions.get(j).copied()?)?;
                seed_group_cov(g, subject.obs_cov(j), subject.obs_times[j])
            })
            .collect::<Option<_>>()?;
        let pk_at_pk_only = (0..subject.pk_only_times.len())
            .map(|m| seed_pk_only_cov(subject.pk_only_cov(m), subject.pk_only_times[m]))
            .collect::<Option<_>>()?;
        Some((pk_at_dose, pk_at_obs, pk_at_pk_only))
    } else {
        // Reuse the caller's per-group seeding when supplied (the scale path already built
        // it), else build it here. Same source either way (#575 review — no double seed).
        let owned;
        let group_dual: &[Vec<T>] = match precomputed_static {
            Some(g) => g,
            None => {
                owned = (0..k_groups)
                    .map(|g| seed_group_cov(g, static_cov, 0.0))
                    .collect::<Option<Vec<_>>>()?;
                &owned
            }
        };
        let pk_at_dose = (0..subject.doses.len())
            .map(|d| {
                Some(group_dual[*occ_to_k.get(&subject.dose_occasions.get(d).copied()?)?].clone())
            })
            .collect::<Option<_>>()?;
        let pk_at_obs = (0..subject.obs_times.len())
            .map(|j| Some(group_dual[*occ_to_k.get(&subject.occasions.get(j).copied()?)?].clone()))
            .collect::<Option<_>>()?;
        let pk_at_pk_only = if subject.pk_only_times.is_empty() {
            Vec::new()
        } else {
            let seeded = seed_pk_only_cov(static_cov, 0.0)?;
            vec![seeded; subject.pk_only_times.len()]
        };
        Some((pk_at_dose, pk_at_obs, pk_at_pk_only))
    }
}

/// First-order (`Dual1<N>`, `N = n_stacked`) IOV seeder — the light counterpart of
/// [`seed_pk_dual2_iov`]. Seeds only `∂p/∂(stacked-η)` (no θ axes, no Hessian): the
/// combined column `c` maps to stacked axis `c` (η_bsv, `c < n_eta`) or
/// `n_eta + group·n_kappa + (c − n_eta)` (κ). For EVID=2 pk-only events
/// `group = None`, so κ columns are dropped. Reuses [`CombinedDerivs::deta`].
fn seed_pk_dual1_iov<const N: usize>(
    model: &CompiledModel,
    pk: &crate::types::PkParams,
    cd: &crate::sens::provider::CombinedDerivs,
    group: Option<usize>,
    n_eta: usize,
    n_kappa: usize,
) -> Vec<Dual1<N>> {
    let n_eff = n_eta + n_kappa;
    let kappa_base = group.map(|g| n_eta + g * n_kappa);
    let stacked_axis = |c: usize| -> Option<usize> {
        if c < n_eta {
            Some(c)
        } else {
            kappa_base.map(|base| base + (c - n_eta))
        }
    };
    let mut out: Vec<Dual1<N>> = pk.values.iter().map(|&v| Dual1::constant(v)).collect();
    for (i, &slot) in model.pk_indices.iter().enumerate() {
        let mut grad = [0.0; N];
        for c in 0..n_eff {
            if let Some(ax) = stacked_axis(c).filter(|&ax| ax < N) {
                grad[ax] = cd.deta[i][c];
            }
        }
        out[slot] = Dual1 {
            value: pk.values[slot],
            grad,
        };
    }
    out
}

/// Light **inner** IOV walk (`Dual1<N>`, `N = n_stacked`) — the first-order, η-only
/// counterpart of [`run_subject_iov`]. Seeds each event's stacked PK duals (per
/// occasion×snapshot, or one per group when static), runs the shared event-driven
/// walk + readout, and reads `∂f/∂(stacked-η)` straight off the dual (#439 ODE IOV).
fn run_subject_iov_eta<const N: usize>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_eta: &[f64],
    occ_groups: &[(u32, Vec<usize>)],
) -> Option<Vec<ObsGrad>> {
    let ode = model.ode_spec.as_ref()?;
    let prog = ode.indiv_param_program.as_ref()?;
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let n_theta = model.n_theta;
    let n_eff = n_eta + n_kappa;
    let k_groups = occ_groups.len();
    let n_stacked = n_eta + k_groups * n_kappa;
    let cov = &subject.covariates;

    let occ_to_k = crate::stats::likelihood::iov_occ_to_k(&occ_groups);
    let combined_for =
        |g: usize| crate::stats::likelihood::iov_combined_effect(stacked_eta, n_eta, n_kappa, g);

    // Per-event `TIME` seeding — the inner counterpart of the outer `run_subject_iov`
    // (#486). Same thread-local guard + per-event forcing.
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let n_rows = model.pk_indices.len();
    let seed_group_cov = |g: usize,
                          cov: &std::collections::HashMap<String, f64>,
                          time: f64|
     -> Option<Vec<Dual1<N>>> {
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
        let combined = combined_for(g);
        let pk = (model.pk_param_fn)(theta, &combined, cov, time);
        let cd = crate::sens::provider::iov_combined_derivs_dyn(
            prog, n_theta, n_eff, n_rows, cov, theta, &combined,
        )?;
        Some(seed_pk_dual1_iov::<N>(
            model,
            &pk,
            &cd,
            Some(g),
            n_eta,
            n_kappa,
        ))
    };
    // EVID=2 pk-only events carry no occasion → κ held at 0 (single-sourced with the
    // closed-form provider, #598 review). Built lazily inside the closure so the common
    // IOV subject with no EVID=2 records pays no allocation.
    let seed_pk_only_cov = |cov: &std::collections::HashMap<String, f64>,
                            time: f64|
     -> Option<Vec<Dual1<N>>> {
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
        let combined_pk_only =
            crate::stats::likelihood::iov_combined_pk_only(stacked_eta, n_eta, n_kappa);
        let pk = (model.pk_param_fn)(theta, &combined_pk_only, cov, time);
        let cd = crate::sens::provider::iov_combined_derivs_dyn(
            prog,
            n_theta,
            n_eff,
            n_rows,
            cov,
            theta,
            &combined_pk_only,
        )?;
        Some(seed_pk_dual1_iov::<N>(
            model, &pk, &cd, None, n_eta, n_kappa,
        ))
    };

    // Build every occasion group's stacked PK seeding at the subject-static covariate
    // snapshot (`t = 0`) — the inner counterpart of the outer `build_all_groups`. Shared by
    // `static_group_dual` and the `ExpressionScale` scale jets below. For a `TIME` model the
    // scale-jet path still calls this at `t = 0`, matching production `apply_scaling`, so
    // `ExpressionScale` + TIME composes correctly under IOV.
    let build_all_groups = || -> Option<Vec<Vec<Dual1<N>>>> {
        (0..k_groups).map(|g| seed_group_cov(g, cov, 0.0)).collect()
    };

    // Static-cov per-group seeding built once and shared by the event walk — the inner
    // counterpart of the outer `static_group_dual` (#575 review). `None` for TV-cov OR
    // `TIME` subjects.
    let per_event = iov_walk_per_event(model, subject);
    let static_group_dual: Option<Vec<Vec<Dual1<N>>>> = if per_event {
        None
    } else {
        Some(build_all_groups()?)
    };

    let (pk_at_dose, pk_at_obs, pk_at_pk_only) = seed_iov_events::<Dual1<N>>(
        subject,
        &occ_to_k,
        k_groups,
        per_event,
        cov,
        static_group_dual.as_deref(),
        seed_group_cov,
        seed_pk_only_cov,
    )?;

    // η-only `ExpressionScale` scale jets, one per occasion group. Mirrors production's
    // subject-static `apply_scaling` materialisation under IOV.
    let group_scale: Option<Vec<Dual1<N>>> = match &model.scaling {
        ScalingSpec::ExpressionScale {
            deriv: Some(sprog), ..
        } => {
            let eta_bsv = &stacked_eta[..n_eta];
            let owned;
            let groups: &[Vec<Dual1<N>>] = match static_group_dual.as_deref() {
                Some(groups) => groups,
                None => {
                    owned = build_all_groups()?;
                    &owned
                }
            };
            Some(build_iov_scale_jets::<Dual1<N>>(
                groups,
                sprog.var_to_pk_slot(),
                |var_duals| sprog.eval_scale_dual1::<N>(theta, eta_bsv, cov, var_duals),
            ))
        }
        _ => None,
    };

    let preds = integrate_tvcov_readout::<Dual1<N>>(
        model,
        subject,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    let mut out = Vec::with_capacity(preds.len());
    for fd in &preds {
        let g = &fd.grad;
        let mut df_deta = vec![0.0; n_stacked];
        for (p, df) in df_deta.iter_mut().enumerate() {
            *df = g[p];
        }
        out.push(ObsGrad {
            f: fd.value,
            df_deta,
        });
    }
    // Apply the η-only `ExpressionScale` quotient per observation (#575/#590).
    if let Some(group_scale) = group_scale {
        for (j, o) in out.iter_mut().enumerate() {
            let g = *occ_to_k.get(&subject.occasions.get(j).copied()?)?;
            crate::sens::provider::apply_scale_quotient_grad_iov::<N>(
                o,
                &group_scale[g],
                n_stacked,
            );
        }
    }
    Some(out)
}

/// `ParamDerivs` (`∂p/∂(θ,η)` + 2nd order) at an explicit covariate snapshot,
/// dispatching on the program's axis count — the cov-taking sibling of
/// [`param_derivatives`] (which reads `subject.covariates`), needed for per-event
/// TV-cov snapshots (#439). Also used by the analytical light TV-cov inner (#447).
pub(crate) fn param_derivatives_at_cov(
    prog: &crate::parser::model_parser::IndivParamProgram,
    model: &CompiledModel,
    cov: &std::collections::HashMap<String, f64>,
    theta: &[f64],
    eta: &[f64],
) -> Option<ParamDerivs> {
    if prog.n_theta_axis() != model.n_theta || prog.n_eta_axis() != model.n_eta {
        return None;
    }
    macro_rules! disp {
        ($($m:literal),+) => {
            match prog.n_axes() {
                $($m => Some(pd_from_program::<$m>(prog, model, cov, theta, eta)),)+
                _ => None,
            }
        };
    }
    disp!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
}

/// Per-event flat PK-slot duals seeded on **η only** (`Dual1<N>`, `N = n_eta`) at a
/// covariate snapshot — the light-inner counterpart of [`seed_pk_dual2`]. Slot for
/// individual parameter `i` carries `∂p/∂η_k` on axis `k`; other slots are constant.
fn seed_pk_dual1<const N: usize>(
    model: &CompiledModel,
    prog: &crate::parser::model_parser::IndivParamProgram,
    theta: &[f64],
    eta: &[f64],
    cov: &std::collections::HashMap<String, f64>,
    time: f64,
) -> Option<Vec<Dual1<N>>> {
    // Seed the model-time thread-local with this event's time so a `TIME`-built-in
    // parameter resolves per event in both the f64 value and the `Dual1` η-walk (#486).
    let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(
        crate::parser::model_parser::compiled_model_uses_time_builtin(model),
        time,
    );
    let n_eta = model.n_eta;
    // The dispatch sizes `N = n_eta` exactly, so the `.min(N)` guard is always a no-op
    // — flat loop (#449 review #15).
    debug_assert_eq!(N, n_eta);
    let pd = param_derivatives_at_cov(prog, model, cov, theta, eta)?;
    let pk = (model.pk_param_fn)(theta, eta, cov, time);
    let mut out: Vec<Dual1<N>> = pk.values.iter().map(|&v| Dual1::constant(v)).collect();
    for (i, &slot) in model.pk_indices.iter().enumerate() {
        let mut grad = [0.0; N];
        for k in 0..n_eta {
            grad[k] = pd.dp_deta[i][k];
        }
        out[slot] = Dual1 {
            value: pk.values[slot],
            grad,
        };
    }
    Some(out)
}

/// Time-varying-covariate **inner** η-gradient for an ODE model (light `Dual1<N>`,
/// `N = n_eta`) — the TV-cov counterpart of [`run_subject_eta`]. Seeds the per-event
/// PK params on η, runs the bolus event-driven walk, and reads `∂f/∂η` off the dual
/// (#439).
fn run_subject_tvcov_eta<const N: usize>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<ObsGrad>> {
    let ode = model.ode_spec.as_ref()?;
    let prog = ode.indiv_param_program.as_ref()?;
    let n_eta = model.n_eta;

    // Dedup identical covariate snapshots via the shared helper (#451 re-review #8).
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let (pk_at_dose, pk_at_obs, pk_at_pk_only) =
        seed_tvcov_snapshots::<Dual1<N>>(subject, uses_time, |cov, time| {
            seed_pk_dual1::<N>(model, prog, theta, eta, cov, time)
        })?;

    let preds = integrate_tvcov_readout::<Dual1<N>>(
        model,
        subject,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    let mut out = Vec::with_capacity(preds.len());
    for fd in &preds {
        // `seed_pk_dual1` seeds η on axes `0..n_eta`, so `grad[k] = ∂f/∂η_k` directly.
        let g = &fd.grad;
        let mut df_deta = vec![0.0; n_eta];
        for k in 0..n_eta {
            df_deta[k] = g[k];
        }
        out.push(ObsGrad {
            f: fd.value,
            df_deta,
        });
    }
    // η-dependent `ExpressionScale` divisor: the η-only quotient, mirroring the outer
    // `run_subject_tvcov` and the static inner path (#486; composes with a `TIME`-built-in
    // structural parameter on this same walk). Subject-static scale at `t = 0` (shared
    // helper). Inner and outer MUST move in lockstep (shared gate) — else the inner EBE
    // gradient silently diverges from the outer.
    crate::sens::provider::apply_event_walk_expression_scale_inner(
        &mut out, model, subject, prog, theta, eta,
    )?;
    Some(out)
}

/// Evaluate the ODE RHS at `t` with the time-after-first-dose / time-after-last-dose
/// anchors lifted as parameter-independent constants — the shared inner of the
/// static ([`integrate_g`]) and TV-cov ([`integrate_tvcov_g`]) walk RHS closures, so
/// the anchor-and-evaluate body is written once (#449 review #11). The static walk's
/// infusion rate forcing is applied by its caller after this returns (the TV-cov
/// subset is bolus-only, so it has none).
#[inline]
#[allow(clippy::too_many_arguments)]
fn eval_rhs_anchored<T: crate::sens::num::PkNum>(
    program: &crate::parser::model_parser::OdeRhsProgram,
    us: &[T],
    ps: &[T],
    t: f64,
    first_dose_time: f64,
    last_dose_eff: f64,
    du: &mut [T],
    vars: &mut Vec<T>,
    stack: &mut Vec<T>,
) {
    let tafd = if first_dose_time.is_finite() {
        t - first_dose_time
    } else {
        f64::NAN
    };
    let tad = if last_dose_eff.is_finite() {
        t - last_dose_eff
    } else {
        f64::NAN
    };
    program.eval_rhs_g::<T>(us, ps, t, tafd, tad, du, vars, stack);
}

/// Exact `J·g` (the time-derivative of the velocity, `ẍ = dẋ/dt`) at a state, **value
/// only**, with no finite differences: one directional RHS evaluation over `Dual1<1>`
/// whose state seed is `x.val` with tangent `g.val` (so `∂RHS/∂ε|_{x+εg} = J·g`). The
/// parameters are held constant (we want the state-Jacobian only). Used by the
/// estimated-lagtime corrections, where `ẍ` enters only through `δlag²` (value 0, zero
/// gradient) so only its value is needed (#439 lagtime).
#[allow(clippy::too_many_arguments)]
fn jdotg_value<T: crate::sens::num::PkNum>(
    program: &crate::parser::model_parser::OdeRhsProgram,
    n_states: usize,
    x: &[T],
    g: &[T],
    params_d1: &[Dual1<1>],
    t: f64,
    first_dose_time: f64,
    anchor: f64,
    d1_vars: &mut Vec<Dual1<1>>,
    d1_stack: &mut Vec<Dual1<1>>,
) -> Vec<f64> {
    let x_tan: Vec<Dual1<1>> = x
        .iter()
        .zip(g.iter())
        .map(|(s, gi)| Dual1 {
            value: s.val(),
            grad: [gi.val()],
        })
        .collect();
    let mut out = vec![Dual1::<1>::constant(0.0); n_states];
    eval_rhs_anchored::<Dual1<1>>(
        program,
        &x_tan,
        params_d1,
        t,
        first_dose_time,
        anchor,
        &mut out,
        d1_vars,
        d1_stack,
    );
    out.iter().map(|o| o.grad[0]).collect()
}

/// The jet-only part of a moving-boundary time `x` (an estimated lagtime or
/// modeled infusion duration): `x − x.val()` has value `0` (the boundary's f64
/// position is unchanged) but the same derivatives as `x`, so it is the `δ` fed
/// into a saltation correction (e.g. `δlag`, `δt_inf`). Shared by every call site
/// so the convention can't drift between them.
#[inline]
fn jet_only<T: crate::sens::num::PkNum>(x: T) -> T {
    x - T::from_f64(x.val())
}

/// Estimated-lagtime event-time saltation at an **infusion rate boundary** (the rate
/// turning on at `t_dose + lag` or off at `t_dose + lag + dur`), where the window shifts
/// with `lag`. Unlike a bolus, the state is continuous and only `ẋ` jumps by the forcing
/// `Δr = F·rate` in `cmt`, so the injection is exact in closed form (no pre/post RHS
/// evals): with `s = −1` at the rate-on boundary and `s = +1` at rate-off,
///   `u[cmt] += s·Δr·δlag`,  `u += −s·½·J·(Δr·e_cmt)·δlag²`,
/// matching the general `D·δlag + (½ẋ̇⁻+½ẋ̇⁺−J⁺ẋ⁻)·δlag²` with `D = s·Δr·e_cmt` and
/// `J⁻ = J⁺ = J` (state continuous). `J·(Δr·e_cmt)` is the exact directional RHS
/// derivative along the rate vector, via a `Dual1<1>` eval — no finite differences (#439).
#[allow(clippy::too_many_arguments)]
fn inject_rate_saltation<T: crate::sens::num::PkNum>(
    u: &mut [T],
    cmt_idx: usize,
    dr: T,
    dlag: T,
    s: f64,
    program: &crate::parser::model_parser::OdeRhsProgram,
    params: &[T],
    t_event: f64,
    first_dose_time: f64,
    anchor: f64,
    d1_vars: &mut Vec<Dual1<1>>,
    d1_stack: &mut Vec<Dual1<1>>,
) {
    let n = u.len();
    if cmt_idx >= n {
        return;
    }
    // First-order (D term): u[cmt] += s·Δr·δlag.
    u[cmt_idx] = u[cmt_idx] + T::from_f64(s) * dr * dlag;
    // Second-order: `J·(Δr·e_cmt)` — the exact directional RHS derivative along the rate
    // vector, via the shared `jdotg_value` primitive (the rate vector has `Δr` in `cmt`,
    // zero elsewhere). Single-sourced with the bolus-saltation `J·g` evals (#472 review #6).
    let mut rate_dir = vec![T::from_f64(0.0); n];
    rate_dir[cmt_idx] = dr;
    let params_d1: Vec<Dual1<1>> = params.iter().map(|p| Dual1::constant(p.val())).collect();
    let jg = jdotg_value::<T>(
        program,
        n,
        u,
        &rate_dir,
        &params_d1,
        t_event,
        first_dose_time,
        anchor,
        d1_vars,
        d1_stack,
    );
    let dlag2 = dlag * dlag;
    for (c, uc) in u.iter_mut().enumerate() {
        // δlag² coefficient = −s·½·(J·(Δr·e_cmt))[c].
        let coef2 = T::from_f64(-s * 0.5 * jg[c]);
        *uc = *uc + coef2 * dlag2;
    }
}

/// General event-time saltation at a **rate boundary** (an infusion or zero-order window
/// turning off) where — unlike the closed-form [`inject_rate_saltation`] — the RHS
/// **Jacobian may jump** across the boundary because a time-varying covariate changes the
/// segment's PK params from `pre_params`→`post_params` (NONMEM end-of-interval convention:
/// the segment ending at the boundary uses the previous record's params, the segment
/// starting there the next record's). The state is continuous (`x⁻ = x⁺ = u`); only `du/dt`
/// jumps from the pre-side velocity `v_minus` to the post-side `v_plus`. With boundary shift
/// `d_off` (jet-only; `δlag + δdur`/`δt_inf`):
///   u += (v⁻ − v⁺)·δ + (½ẋ̇⁻ + ½ẋ̇⁺ − J⁺·ẋ⁻)·δ²,
/// where `ẋ̇± = J(pre/post)·v±` and the cross term is `J(post)·v⁻`. Both velocities must
/// already include **every** concurrently-active forcing (other infusions / zero-order
/// windows and pointwise input rates), so the curvature term is exact when `J⁺ ≠ J⁻`; the
/// boundary's own toggling forcing is then the only difference between `v⁻` and `v⁺`. When
/// `pre_params == post_params` this reduces exactly (analytically) to `inject_rate_saltation`
/// — so callers take that cheaper closed form via [`pk_snapshot_equal`] and only reach here
/// on a genuine Jacobian jump (#653 review). The three `J·v` directional evals are exact
/// `Dual1` derivatives — no finite differences (the rate-boundary twin of the bolus-lagtime
/// saltation).
#[allow(clippy::too_many_arguments)]
fn general_rate_off_saltation<T: crate::sens::num::PkNum>(
    u: &mut [T],
    program: &crate::parser::model_parser::OdeRhsProgram,
    n_states: usize,
    pre_params: &[T],
    post_params: &[T],
    v_minus: &[T],
    v_plus: &[T],
    d_off: T,
    t_event: f64,
    first_dose_time: f64,
    anchor: f64,
    d1_vars: &mut Vec<Dual1<1>>,
    d1_stack: &mut Vec<Dual1<1>>,
) {
    // `J·v` directional evals (Dual1), with pre/post params. Any additive forcing carried in
    // `v` is state-constant (zero own-Jacobian), so `program`'s Jacobian is the full one.
    let pre_d1: Vec<Dual1<1>> = pre_params
        .iter()
        .map(|x| Dual1::constant(x.val()))
        .collect();
    let post_d1: Vec<Dual1<1>> = post_params
        .iter()
        .map(|x| Dual1::constant(x.val()))
        .collect();
    let jg_minus = jdotg_value::<T>(
        program,
        n_states,
        u,
        v_minus,
        &pre_d1,
        t_event,
        first_dose_time,
        anchor,
        d1_vars,
        d1_stack,
    );
    let jg_plus = jdotg_value::<T>(
        program,
        n_states,
        u,
        v_plus,
        &post_d1,
        t_event,
        first_dose_time,
        anchor,
        d1_vars,
        d1_stack,
    );
    // Cross term J⁺·ẋ⁻ = J(post)·v⁻ (post-side Jacobian along the pre-side velocity).
    let jg_cross = jdotg_value::<T>(
        program,
        n_states,
        u,
        v_minus,
        &post_d1,
        t_event,
        first_dose_time,
        anchor,
        d1_vars,
        d1_stack,
    );
    let d_off2 = d_off * d_off;
    for c in 0..n_states {
        // δ² coefficient = ½ẋ̇⁻ + ½ẋ̇⁺ − J⁺·ẋ⁻.
        let coef2 = T::from_f64(0.5 * (jg_minus[c] + jg_plus[c]) - jg_cross[c]);
        u[c] = u[c] + (v_minus[c] - v_plus[c]) * d_off + coef2 * d_off2;
    }
}

/// Whether two per-segment PK snapshots are equal (by value). Under a covariate that is
/// constant across a boundary — or a model with no TV covariates at all — consecutive
/// records share their covariate values, so the deterministic (non-IOV) PK map yields
/// identical duals (values *and* jets); value-equality is therefore sufficient to detect
/// "no Jacobian jump". Lets a rate-off boundary take the cheap closed-form
/// [`inject_rate_saltation`] instead of the general `g⁻−g⁺` saltation whenever the Jacobian
/// does not actually jump (#653 review — avoids the extra RHS evals on every lagtime-only /
/// covariate-constant inner-EBE and outer-gradient evaluation).
#[inline]
fn pk_snapshot_equal<T: crate::sens::num::PkNum>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.val() == y.val())
}

/// Dual steady-state equilibration: the analytic-sensitivity counterpart of production's
/// `equilibrate_ss_state`. NONMEM SS=1 loads the compartments with the steady-state
/// amounts of an infinite-past pulse train of interval `II`. There is no closed form for
/// a general ODE, so production expands the train as a **finite**
/// [`crate::ode::predictions::SS_EQUILIBRATION_CYCLES`] loop of `(apply dose; integrate II)`
/// from a zero state, returning the pre-pulse trough (the shared const keeps this trough
/// from drifting from the f64 predictor). Because the loop is finite and explicit, running
/// it over the dual type `T` propagates `∂(SS state)/∂(θ,η)` directly — no implicit
/// fixed-point differentiation. The caller then applies the SS dose's own pulse normally.
///
/// Handles SS **bolus** doses (pulse + decay per cycle) and SS **infusions** (an active-rate
/// window `[0, t_inf]` then a quiet `[0, II−t_inf]` decay per cycle), where `t_inf`/the active
/// forcing are the caller's mode-aware `(inf_rate, inf_window)` jet (`inf_eff`, #530/#419) —
/// so a modeled-duration/rate dose (`RATE=-1/-2`) and a rate-defined infusion under `F ≠ 1`
/// (whose window is `F·duration`, a moving boundary) are both analytic via the same per-cycle
/// rate-off saltation as the main walk's `K_INF_END` (#486). `eval_rhs_anchored` uses
/// cycle-relative time (anchor 0), matching production for a TAD-independent RHS (#439 SS).
#[allow(clippy::too_many_arguments)]
fn equilibrate_ss_state_g<T: crate::sens::num::PkNum>(
    program: &crate::parser::model_parser::OdeRhsProgram,
    n_states: usize,
    dose: &crate::types::DoseEvent,
    f_bio: T,
    params: &[T],
    // `Some((rate, window))` for an infusion (the caller's `inf_eff[idx]` jet, #530/#419);
    // `None` for a bolus. Replacing the old `(is_inf, inf_rate, inf_window)` triple removes the
    // meaningless `T::from_f64(0.0)` sentinels the bolus case had to pass (#642 review #5).
    inf: Option<(T, T)>,
    opts: &crate::ode::solver::OdeSolverOptions,
    d1_vars: &mut Vec<Dual1<1>>,
    d1_stack: &mut Vec<Dual1<1>>,
) -> Vec<T> {
    let mut u = vec![T::from_f64(0.0); n_states];
    if dose.ii <= 0.0 || dose.cmt == 0 {
        return u;
    }
    let cmt_idx = dose.cmt - 1;
    if cmt_idx >= n_states {
        return u;
    }
    let vars_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());
    let stack_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());
    let bare_rhs = |us: &[T], ps: &[T], t: f64, du: &mut [T]| {
        eval_rhs_anchored::<T>(
            program,
            us,
            ps,
            t,
            0.0,
            0.0,
            du,
            &mut vars_cell.borrow_mut(),
            &mut stack_cell.borrow_mut(),
        );
    };
    if let Some((inf_rate, inf_window)) = inf {
        // SS infusion: each cycle is an active-rate window `[0, t_inf]` (the wrapped RHS
        // injects the mode-aware bioavailable forcing into the dosing compartment) followed
        // by a quiet decay window `[0, II − t_inf]`. `(inf_rate, inf_window)` is the caller's
        // `inf_eff[idx]` (#530/#419): a modeled dose (`RATE=-1/-2`) rebuilds both from its PK
        // slot, and a rate-defined infusion under `F ≠ 1` carries `F`'s jet in the *window*
        // (`inf_window = F·duration`, rate held) rather than in the magnitude — so the
        // active/quiet split below is generic over every case, not just the F=1 / duration-
        // defined subset the old fixed-`dose.duration` split covered.
        let t_inf_val = inf_window.val();
        if t_inf_val > dose.ii {
            return u; // overlapping pulses — no simple equilibration (mirrors production)
        }
        let rate_forcing = inf_rate;
        let quiet_val = dose.ii - t_inf_val;
        let saveat_inf = [t_inf_val];
        let saveat_q = [quiet_val];
        // Shared early stop (#519): break once the trough converges, on the same mixed
        // atol/rtol criterion the f64 predictor uses (driver shared with `equilibrate_ss_g`,
        // #532 #9/#10).
        let mut prev = vec![0.0_f64; n_states];
        let mut cur = vec![0.0_f64; n_states];
        let mut cycles_run = 0usize;
        for cycle in 0..crate::ode::predictions::SS_EQUILIBRATION_CYCLES {
            let rhs_active = |us: &[T], ps: &[T], t: f64, du: &mut [T]| {
                bare_rhs(us, ps, t, du);
                if cmt_idx < du.len() {
                    du[cmt_idx] = du[cmt_idx] + rate_forcing;
                }
            };
            let sol = solve_ode_g(&rhs_active, &u, (0.0, t_inf_val), params, &saveat_inf, opts);
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
            // Per-cycle rate-off window saltation (#530/#419): the active/quiet boundary is
            // evaluated at the fixed nominal `t_inf_val`, but the true window `inf_window` may
            // carry a jet (modeled `D`/`R`, or `F·duration`) — inject the same event-time
            // correction the main walk's `K_INF_END` uses (`d_off = inf_window − t_inf_val`,
            // mechanically identical to `dtinf` there), so each cycle's boundary sensitivity
            // is exact, not just the boundary's *value*.
            let dtinf = inf_window - T::from_f64(t_inf_val);
            inject_rate_saltation::<T>(
                &mut u,
                cmt_idx,
                rate_forcing,
                dtinf,
                1.0,
                program,
                params,
                t_inf_val,
                0.0,
                0.0,
                d1_vars,
                d1_stack,
            );
            if quiet_val > 0.0 {
                let sol = solve_ode_g(&bare_rhs, &u, (0.0, quiet_val), params, &saveat_q, opts);
                if let Some(last) = sol.last() {
                    u.copy_from_slice(&last.u);
                }
            }
            cycles_run = cycle + 1;
            if crate::sens::propagate::ss_dual_cycle_should_stop(cycle, &u, &mut cur, &mut prev) {
                break;
            }
        }
        crate::ode::predictions::record_ss_equilibration_cycles(cycles_run);
        return u;
    }
    // Bolus SS: each cycle applies the pulse `F·amt`, then decays over one interval.
    let amt = T::from_f64(dose.amt);
    let saveat = [dose.ii];
    let mut prev = vec![0.0_f64; n_states];
    let mut cur = vec![0.0_f64; n_states];
    let mut cycles_run = 0usize;
    for cycle in 0..crate::ode::predictions::SS_EQUILIBRATION_CYCLES {
        u[cmt_idx] = u[cmt_idx] + f_bio * amt;
        let sol = solve_ode_g(&bare_rhs, &u, (0.0, dose.ii), params, &saveat, opts);
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
        cycles_run = cycle + 1;
        if crate::sens::propagate::ss_dual_cycle_should_stop(cycle, &u, &mut cur, &mut prev) {
            break;
        }
    }
    crate::ode::predictions::record_ss_equilibration_cycles(cycles_run);
    u
}

/// Taylor-extend a state `u` — already integrated to the fixed nominal bound `t_val` under
/// `rhs` — by a further **dual-valued, zero-value** duration `dt`: the value `u` would have
/// from continuing to flow under the SAME `rhs` for `dt` more time, to 2nd order in `dt`
/// (`u += g(u)·dt + ½·(J·g)(u)·dt²`, `g` = `rhs`'s velocity at `(u, t_val)`). Unlike
/// [`inject_rate_saltation`] (a forcing *jump* at an RHS switch), there is no switch here —
/// this is the sensitivity of a continuous flow's endpoint to its own duration, needed when
/// the duration itself (not a forcing) carries a derivative jet: the SS+lagtime pre-arrival
/// phase seed (#486, [`ss_state_at_phase_g`]), where `phase = II − lag` and its active/quiet
/// split are dual-valued durations advanced from an otherwise lag-invariant trough. `rhs`'s
/// possible constant forcing (e.g. the SS active window) has zero state-Jacobian, so the
/// *bare* program's `J·g` (via [`jdotg_value`]) is exact here too — no separate forced-
/// Jacobian needed, mirroring how `inject_rate_saltation` itself only ever uses the bare
/// program.
#[allow(clippy::too_many_arguments)]
fn extend_flow_by_dual_duration_g<T: crate::sens::num::PkNum>(
    u: &mut [T],
    dt: T,
    program: &crate::parser::model_parser::OdeRhsProgram,
    rhs: impl Fn(&[T], &[T], f64, &mut [T]),
    params: &[T],
    t_val: f64,
    first_dose_time: f64,
    anchor: f64,
    d1_vars: &mut Vec<Dual1<1>>,
    d1_stack: &mut Vec<Dual1<1>>,
) {
    let n = u.len();
    let mut g = vec![T::from_f64(0.0); n];
    rhs(u, params, t_val, &mut g);
    let params_d1: Vec<Dual1<1>> = params.iter().map(|p| Dual1::constant(p.val())).collect();
    let jg = jdotg_value::<T>(
        program,
        n,
        u,
        &g,
        &params_d1,
        t_val,
        first_dose_time,
        anchor,
        d1_vars,
        d1_stack,
    );
    let dt2 = dt * dt;
    for c in 0..n {
        u[c] = u[c] + g[c] * dt + T::from_f64(0.5 * jg[c]) * dt2;
    }
}

/// Dual counterpart of production's `ss_state_at_phase`: advances a precomputed SS `trough`
/// (the pre-pulse state, phase `0⁻ ≡ II`, from [`equilibrate_ss_state_g`]) forward to
/// `phase ∈ [0, II)`, measured from the pulse at phase 0. Used for the SS+lagtime pre-arrival
/// seed (#486): observations between an SS dose's record time and its lagged arrival read the
/// *previous* interval's tail at `phase = II − lag`, which carries `lag`'s jet
/// (`∂phase/∂lag = −1`). Each duration derived from `phase` is integrated to its fixed nominal
/// (`.val()`) bound and Taylor-extended by [`extend_flow_by_dual_duration_g`] for the remaining
/// dual part — the same technique `equilibrate_ss_state_g`'s per-cycle rate-off saltation uses,
/// but for a continuous (non-switching) flow rather than a forcing jump; crossing the
/// active/quiet boundary itself (when `phase > t_inf`) still uses [`inject_rate_saltation`],
/// exactly as the main cycle loop does. Mirrors production's "phase ≥ dose.duration" scope
/// note: assumes the lagtime doesn't leave the prior infusion still active at `phase` (the
/// realistic regime; overlapping infusions are already rejected upstream by
/// `equilibrate_ss_state_g`).
///
/// The `trough` is taken as a parameter (not recomputed) so the caller can equilibrate once and
/// reuse it at both this pre-arrival seed and the dose's own later `K_DOSE` event, halving the
/// up-to-50-cycle dual SS loop for a lagged SS dose (#642 review #4). `inf` is `Some((rate,
/// window))` for an infusion, `None` for a bolus (#642 review #5).
#[allow(clippy::too_many_arguments)]
fn ss_state_at_phase_g<T: crate::sens::num::PkNum>(
    program: &crate::parser::model_parser::OdeRhsProgram,
    n_states: usize,
    dose: &crate::types::DoseEvent,
    f_bio: T,
    params: &[T],
    inf: Option<(T, T)>,
    phase: T,
    trough: Vec<T>,
    opts: &crate::ode::solver::OdeSolverOptions,
    d1_vars: &mut Vec<Dual1<1>>,
    d1_stack: &mut Vec<Dual1<1>>,
) -> Vec<T> {
    let mut u = trough;
    let phase_val = phase.val();
    if phase_val <= 0.0 {
        return u;
    }
    // CMT is 1-based; a malformed `CMT=0` must no-op (return the trough unchanged), matching
    // `equilibrate_ss_state_g`'s own `dose.cmt == 0` zero-guard. `saturating_sub(1)` would
    // instead alias it to the valid index 0 and silently dose compartment 0 (#642 review #1).
    if dose.cmt == 0 {
        return u;
    }
    let cmt_idx = dose.cmt - 1;
    if cmt_idx >= n_states {
        return u;
    }
    let vars_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());
    let stack_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());
    let bare_rhs = |us: &[T], ps: &[T], t: f64, du: &mut [T]| {
        eval_rhs_anchored::<T>(
            program,
            us,
            ps,
            t,
            0.0,
            0.0,
            du,
            &mut vars_cell.borrow_mut(),
            &mut stack_cell.borrow_mut(),
        );
    };
    if let Some((inf_rate, inf_window)) = inf {
        let t_inf_val = inf_window.val();
        let active_val = phase_val.min(t_inf_val);
        let rhs_active = |us: &[T], ps: &[T], t: f64, du: &mut [T]| {
            bare_rhs(us, ps, t, du);
            if cmt_idx < du.len() {
                du[cmt_idx] = du[cmt_idx] + inf_rate;
            }
        };
        let sol = solve_ode_g(
            &rhs_active,
            &u,
            (0.0, active_val),
            params,
            &[active_val],
            opts,
        );
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
        if phase_val <= t_inf_val {
            // Still inside the active window at `phase`: extend along the SAME (active)
            // flow by the remaining dual part of `phase` itself.
            let dt = phase - T::from_f64(phase_val);
            extend_flow_by_dual_duration_g::<T>(
                &mut u, dt, program, rhs_active, params, active_val, 0.0, 0.0, d1_vars, d1_stack,
            );
        } else {
            // Crossing the active/quiet boundary: the window itself may carry a jet
            // (modeled `D`/`R`, or `F·duration`) — inject the same rate-off saltation
            // `equilibrate_ss_state_g`'s cycle loop uses (`d_off = inf_window − t_inf_val`).
            let dtinf = inf_window - T::from_f64(t_inf_val);
            inject_rate_saltation::<T>(
                &mut u, cmt_idx, inf_rate, dtinf, 1.0, program, params, t_inf_val, 0.0, 0.0,
                d1_vars, d1_stack,
            );
            let quiet_val = phase_val - t_inf_val;
            let sol = solve_ode_g(&bare_rhs, &u, (0.0, quiet_val), params, &[quiet_val], opts);
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
            // Quiet-leg duration `phase − inf_window` also carries `phase`'s (and the
            // window's) jet beyond the nominal `quiet_val` — extend along the quiet flow.
            let dt_quiet = (phase - inf_window) - T::from_f64(quiet_val);
            extend_flow_by_dual_duration_g::<T>(
                &mut u, dt_quiet, program, bare_rhs, params, quiet_val, 0.0, 0.0, d1_vars, d1_stack,
            );
        }
    } else {
        u[cmt_idx] = u[cmt_idx] + f_bio * T::from_f64(dose.amt);
        let sol = solve_ode_g(&bare_rhs, &u, (0.0, phase_val), params, &[phase_val], opts);
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
        let dt = phase - T::from_f64(phase_val);
        extend_flow_by_dual_duration_g::<T>(
            &mut u, dt, program, bare_rhs, params, phase_val, 0.0, 0.0, d1_vars, d1_stack,
        );
    }
    u
}

/// Time-varying-covariate event-driven walk over the dual state (#439), the ODE
/// mirror of the analytical [`super::provider::subject_sensitivities_tvcov`] /
/// `event_driven_sens_g`. For the **bolus** subset it reproduces production's
/// `ode_predictions_event_driven`: a merged dose+obs timeline (dose sorts before a
/// co-timed obs), each segment `[cur_t, t_event]` integrated with the params
/// evaluated **at** `t_event` (NONMEM end-of-interval), boluses applied after the
/// segment, and the state captured at each observation. `pk_at_dose` / `pk_at_obs`
/// are the per-event flat PK-slot duals pre-seeded by the caller on `(θ,η)` (outer)
/// or `η` (inner); `f_bio_at_dose[k]` is dose `k`'s bioavailability dual. Returns
/// one state vector per observation (parallel to `subject.obs_times`).
///
/// **Deliberately kept separate from [`integrate_g`]** (the #451 fold was assessed
/// and declined): the two have divergent control flow that can't merge without
/// either a regression or a net complexity *increase*. Production's static walk
/// (`ode_predictions`) — which the static dual `integrate_g` must match to 1e-9 —
/// integrates each break-time segment under **constant** params and records
/// observations as **in-segment solver save points**. This TV-cov walk instead
/// switches params at **every event** (NONMEM end-of-interval), so observations
/// must be **segment boundaries**, not interior save points. Forcing one
/// obs-handling form onto both would make the static dual diverge from production
/// (the 1e-9 risk); keeping both behind a mode branch is just two control flows
/// glued together. The genuinely shareable pieces — `eval_rhs_anchored`,
/// `resolve_obs_readout`, `solve_ode_g` — are already factored out and used by both.
///
/// `has_lagtime`: when true, each dose `k` arrives at `t_dose + pk_at_dose[k][LAGTIME]`
/// and carries the event-time (saltation) lagtime sensitivity. The time-shift identity
/// the static walk uses is invalid here (params switch on an absolute occasion/covariate
/// timeline), so the lag sensitivity is injected **at each dose** as
/// `x⁺ += D·δlag + ½·(dD/dt)·δlag²` (`D = g(x⁻)−g(x⁺)`, `dD/dt = J·g(x⁻)−J·g(x⁺)`) and
/// propagated by the event-driven integrator — exact, no finite differences (#439).
#[allow(clippy::too_many_arguments)]
fn integrate_tvcov_g<T: crate::sens::num::PkNum>(
    program: &crate::parser::model_parser::OdeRhsProgram,
    ode: &OdeSpec,
    n_states: usize,
    subject: &Subject,
    pk_at_dose: &[Vec<T>],
    pk_at_obs: &[Vec<T>],
    pk_at_pk_only: &[Vec<T>],
    f_bio_at_dose: &[T],
    init_state: &[T],
    // Differentiated PK slots, for re-seeding `init(...)` at an EVID 3/4 reset (#486). Only
    // read when `ode.init_fn` is present; a no-`init` model zeros the state at a reset as before.
    pk_indices: &[usize],
    first_dose_time: f64,
    dose_lag_slot: &[usize],
    dose_modeled_slot: &[Option<(crate::types::RateMode, usize)>],
    opts: &crate::ode::solver::OdeSolverOptions,
) -> Vec<Vec<T>> {
    let n_obs = subject.obs_times.len();
    let mut states: Vec<Vec<T>> = vec![vec![T::from_f64(0.0); n_states]; n_obs];

    debug_assert_eq!(pk_at_pk_only.len(), subject.pk_only_times.len());

    // Per-dose lagtime: dose `k` arrives at `d.time + lag_val(k)`, with its lag read from
    // `pk_at_dose[k][dose_lag_slot[k]]` — the bare `PK_IDX_LAGTIME` slot or, for a
    // compartment-indexed `ALAG{cmt}`, that compartment's slot (so per-dose differing lags
    // are exact). `dose_lag_slot` is empty when the model has no lagtime (byte-identical to
    // the pre-lag walk).
    let has_lagtime = !dose_lag_slot.is_empty();
    // Carries `∂lag/∂(θ,η)` (incl. the lagtime axis itself) — fed into
    // `add_prepared_input_rate_forcing` as `dose_lagtimes` so a built-in input-rate
    // forcing's `tad = t − (dose.time + lag)` carries the exact continuous
    // `∂R_in/∂lag` for `t` after the dose's (lagged) arrival (#486); the *onset*
    // discontinuity at the arrival itself is injected separately as a rate-on
    // saltation (mirroring the lagged-infusion rate-on injection below). A flat
    // constant (`T::from_f64(0.0)`) when there is no lagtime, so `tad` reduces to the
    // pre-#486 fixed-boundary computation.
    let lag_dual = |k: usize| -> T {
        if has_lagtime {
            pk_at_dose[k][dose_lag_slot[k]]
        } else {
            T::from_f64(0.0)
        }
    };
    // Value-only counterpart of `lag_dual`, for callers that only need `lag.val()`.
    let lag_val = |k: usize| -> f64 { lag_dual(k).val() };

    // #530: per-dose modeled rate/duration slot (empty when every dose is fixed). A modeled
    // dose is unresolved in `subject.doses` (`rate`/`duration == 0`), so its effective
    // forcing/window is rebuilt from `pk_at_dose[k][slot]` as a jet in `inf_eff` below.
    let has_modeled = !dose_modeled_slot.is_empty();
    let modeled_at = |k: usize| -> Option<(crate::types::RateMode, usize)> {
        if has_modeled {
            dose_modeled_slot[k]
        } else {
            None
        }
    };
    // Slot-aware infusion predicate: a modeled dose (`RATE=-1`/`-2`) is always an infusion,
    // but is unresolved here so `is_real_infusion`'s `is_fixed` tripwire would fire — gate on
    // `!is_fixed()` first. A fixed dose defers to the production `is_real_infusion`.
    let is_inf = |d: &crate::types::DoseEvent| -> bool {
        !d.is_fixed() || crate::ode::predictions::is_real_infusion(d)
    };
    // Mode-aware bioavailability for infusions (#419). A duration-defined infusion
    // (`RATE=-2` / `D{cmt}`) scales its *rate* by `F` over a fixed window; a rate-defined
    // infusion (`RATE>0` / `R{cmt}` / `RATE=-1`) holds its rate and scales the *window
    // length* to `F·amt/rate`. So `F`'s derivative jet lives in the effective rate
    // (duration-defined) or in the effective window length (rate-defined) — in the latter
    // the window end is a moving boundary in `F`, carried by the rate-off saltation exactly
    // as a lagtime shift. Non-infusions get `0` placeholders.
    // Computed once per subject: `n_infusion_ends` is reused below for the timeline capacity
    // reservation, and `has_any_infusion` (= it > 0) gates all infusion-specific work — the
    // per-dose effective forcing/window below, and the per-segment `active_inf` scan — so the
    // common bolus-only / oral case does a single predicate scan, not several
    // (#472 review #7 / round 2 #10 / #473 review #7).
    let n_infusion_ends = subject.doses.iter().filter(|d| is_inf(d)).count();
    let has_any_infusion = n_infusion_ends > 0;
    // Per-dose effective `(rate, window length)` — the effective rate and window are
    // physically one infusion's bioavailable forcing, so they share one pass (a divergence
    // between them would be a rate inconsistent with its window). `0` placeholders for
    // non-infusion doses; empty when the subject has no infusion (never indexed then —
    // every read is behind an `is_real_infusion` / `has_any_infusion` guard). #473 review #7.
    let inf_eff: Vec<(T, T)> = if !has_any_infusion {
        Vec::new()
    } else {
        subject
            .doses
            .iter()
            .enumerate()
            .map(|(k, d)| {
                if !is_inf(d) {
                    (T::from_f64(0.0), T::from_f64(0.0))
                } else if let Some((mode, slot)) = modeled_at(k) {
                    // #530: rebuild the modeled rate/duration from its PK slot as a live jet,
                    // mirroring the f64 `resolve_rate` (incl. its domain-wall floor clamp).
                    // `amt` is resolved; `F`'s jet enters via `f_bio_at_dose[k]` exactly as
                    // the fixed arms below.
                    let amt = T::from_f64(d.amt);
                    match mode {
                        // `RATE=-2` / `D{cmt}`: window = `D` jet, rate = `F·amt/D`.
                        crate::types::RateMode::ModeledDuration => {
                            let dur = pk_at_dose[k][slot]
                                .guard_floor(crate::types::DoseEvent::DURATION_FLOOR);
                            (f_bio_at_dose[k] * amt / dur, dur)
                        }
                        // `RATE=-1` / `R{cmt}`: rate = `R` jet, window = `F·amt/R`.
                        crate::types::RateMode::ModeledRate => {
                            let rate = pk_at_dose[k][slot]
                                .guard_floor(crate::types::DoseEvent::RATE_FLOOR);
                            (rate, f_bio_at_dose[k] * amt / rate)
                        }
                        crate::types::RateMode::Fixed => {
                            unreachable!("modeled_at returns None for a Fixed dose")
                        }
                    }
                } else {
                    match d.infusion_def {
                        // Rate-defined: rate held, window `F·amt/rate` carries `F`'s jet.
                        crate::types::InfusionDef::RateDefined => (
                            T::from_f64(d.rate),
                            f_bio_at_dose[k] * T::from_f64(d.duration),
                        ),
                        // Duration-defined: rate `F·rate` carries `F`'s jet, window fixed.
                        crate::types::InfusionDef::DurationDefined => (
                            f_bio_at_dose[k] * T::from_f64(d.rate),
                            T::from_f64(d.duration),
                        ),
                    }
                }
            })
            .collect()
    };
    let inf_window_len = |k: usize| -> f64 { inf_eff[k].1.val() };

    // Built-in absorption input-rate forcing (#486): `dose_lagtimes_dual[k]` feeds
    // `add_prepared_input_rate_forcing`'s `tad` computation with the dual lag (see
    // `lag_dual` above); empty when the model has none (the common case — every
    // per-segment closure below then skips the forcing loop entirely).
    let has_input_rate = !ode.input_rate.is_empty();
    let dose_lagtimes_dual: Vec<T> = if has_input_rate {
        (0..subject.doses.len()).map(lag_dual).collect()
    } else {
        Vec::new()
    };

    // #486: zero-order absorption windows on the event-driven walk — the port of the static
    // `integrate_g`'s `zero_windows` (and the dual mirror of production's `zero_order_windows`
    // + `active_zero_order_inputs`). A dose feeding a `zero_order(dur)` forcing delivers a
    // constant `F·amt·frac/dur` over `[d.time+lag, d.time+lag+dur]` — mechanically an
    // infusion, but with the window END (and, under an estimated lagtime, the START) a moving
    // boundary in the estimated `dur`/`lag`. The constant rate and window are fixed from the
    // dose's OWN PK snapshot (`pk_at_dose[k]` / `f_bio_at_dose[k]`), matching production (the
    // rate is mass-exact even when `dur` rides a time-varying covariate — a per-segment
    // recompute would drift). Delivered as a per-segment constant (`active_zero` below), with
    // the moving boundaries carried by the rate-on (`K_DOSE`, lagtime only) and rate-off
    // (`K_ZO_END`) saltations. Stored as `(cmt, rate_jet, w_start, w_end, dur_jet, dose_idx)`.
    // Empty for the common non-zero-order subject (every downstream read is then skipped).
    let has_zero_order = has_input_rate
        && ode
            .input_rate
            .iter()
            .any(|f| f.kind == crate::pk::absorption::InputRateKind::ZeroOrder);
    let zero_windows: Vec<(usize, T, f64, f64, T, usize)> = if !has_zero_order {
        Vec::new()
    } else {
        subject
            .doses
            .iter()
            .enumerate()
            .filter_map(|(k, d)| {
                if d.amt <= 0.0 {
                    return None;
                }
                // One zero-order forcing per dose compartment (the parser rejects > 1),
                // matching production's `zero_order_dur_and_frac_for_dose`; shared with the
                // static walk via `zero_order_forcing_for_dose` (#653 review #7).
                let (f, dur) = zero_order_forcing_for_dose(ode, d, &pk_at_dose[k])?;
                // `frac` = 1 for an unfractioned `zero_order`; the declared pathway fraction
                // for a `mixed` `FR*zero_order` leg (#505) — a linear multiplier on the rate.
                let frac = f.frac::<T>(&pk_at_dose[k]);
                let rate = f_bio_at_dose[k] * T::from_f64(d.amt) * frac / dur;
                let w_start = d.time + lag_val(k);
                let w_end = w_start + dur.val();
                Some((d.cmt.saturating_sub(1), rate, w_start, w_end, dur, k))
            })
            .collect()
    };

    // Whether dose `k` is a periodic steady-state dose (`SS=1`, `II>0`) — single source of
    // truth for the walk, mirroring `Subject::has_periodic_ss_dose`'s per-dose predicate.
    let is_ss_dose = |d: &crate::types::DoseEvent| -> bool { d.ss && d.ii > 0.0 };

    // The SS-equilibration infusion jet for dose `idx`: `Some(inf_eff[idx])` for an infusion,
    // `None` for a bolus. One shared source for both SS call sites (`K_DOSE` and `K_SS_SEED`),
    // so their infusion-jet handling can't desync (#642 review #5). `inf_eff[idx]` is only read
    // when `is_inf(d)` (which implies `has_any_infusion`, so `inf_eff` is populated).
    let ss_inf = |d: &crate::types::DoseEvent, idx: usize| -> Option<(T, T)> {
        if is_inf(d) {
            Some(inf_eff[idx])
        } else {
            None
        }
    };

    // Merged timeline: (time, kind, idx), kind ∈ {Reset=0, SsSeed=1, Dose=2, PkOnly=3, Obs=4,
    // InfEnd=5} — the sort key matching production's `kind_order` (Reset before a co-timed
    // Dose so an EVID=4 reset+dose zeros the state before its own dose lands; a per-dose
    // SS pre-arrival seed — see below — before its own later Dose event; Dose before PkOnly
    // before Obs; infusion-end last so an obs at the end reads the infusion still
    // contributing). Doses (and infusion windows) sit at their lagged arrival
    // `d.time + lag_val(k)`; resets and pk-only records are at their record time (fixed, not
    // lag-shifted).
    const K_RESET: u8 = 0;
    const K_SS_SEED: u8 = 1;
    const K_DOSE: u8 = 2;
    const K_PKONLY: u8 = 3;
    const K_OBS: u8 = 4;
    const K_INF_END: u8 = 5;
    // #486: zero-order absorption window end. Sorts after `K_OBS` (like `K_INF_END`) so an
    // observation exactly at the window end reads the constant rate still on, matching the
    // static walk and production's `active_zero_order_inputs` full-containment convention.
    const K_ZO_END: u8 = 6;
    // Capacity includes one `K_INF_END` slot per infusion (each dose adds its window-end
    // event below) and one `K_SS_SEED` slot per lagged SS dose, matching production's
    // timeline reservation. `n_infusion_ends` was computed once above (and reused for
    // `has_any_infusion`).
    let n_ss_seeds = subject
        .doses
        .iter()
        .enumerate()
        .filter(|(k, d)| is_ss_dose(d) && lag_val(*k) > 0.0)
        .count();
    let mut tl: Vec<(f64, u8, usize)> = Vec::with_capacity(
        subject.doses.len()
            + n_obs
            + subject.pk_only_times.len()
            + subject.reset_times.len()
            + n_infusion_ends
            + n_ss_seeds
            + zero_windows.len(),
    );
    for &rt in &subject.reset_times {
        tl.push((rt, K_RESET, 0));
    }
    for (k, d) in subject.doses.iter().enumerate() {
        tl.push((d.time + lag_val(k), K_DOSE, k));
        if is_inf(d) {
            // Window end uses the bioavailable length (`F·dur` for a rate-defined infusion,
            // the modeled `D`/`F·amt/R` for a modeled dose, #530).
            tl.push((d.time + lag_val(k) + inf_window_len(k), K_INF_END, k));
        }
        // SS + estimated lagtime (#486): the dose arrives at `d.time + lag`, so observations
        // in the pre-arrival window `[d.time, d.time + lag)` must read the *previous*
        // interval's steady-state tail, not the (empty) running state. Break at the raw
        // record time `d.time` and seed it there via `ss_state_at_phase_g` (phase
        // `II − lag`, the point the prior pulse's tail has decayed to by the record time) —
        // mirrors production's dense-path break (`ode/predictions.rs` `ss_state_at_phase`
        // call sites). Only when this dose's own resolved lag is positive; a model with
        // lagtime elsewhere but zero lag on this SS dose needs no seed.
        if is_ss_dose(d) && lag_val(k) > 0.0 {
            tl.push((d.time, K_SS_SEED, k));
        }
    }
    for (j, &t) in subject.obs_times.iter().enumerate() {
        tl.push((t, K_OBS, j));
    }
    for (m, &t) in subject.pk_only_times.iter().enumerate() {
        tl.push((t, K_PKONLY, m));
    }
    // #486: break at each zero-order window end `w_end` (idx = the `zero_windows` index) so
    // every segment is fully inside or outside every window — the invariant the per-segment
    // full-containment filter (`active_zero`) relies on — and the rate-off saltation lands on
    // an exact break. Mirrors production's `Kind::InfusionEnd` break for a zero-order cutoff.
    for (wi, &(_, _, _, w_end, _, _)) in zero_windows.iter().enumerate() {
        tl.push((w_end, K_ZO_END, wi));
    }
    tl.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    if tl.is_empty() {
        return states;
    }

    let mut cur_t = tl[0].0;
    // With an `init(...)` baseline the state is non-zero from the subject's true integration
    // start (`subject_integration_start`, the earliest *record* time). If the first timeline
    // event is later — e.g. a lagged first dose (whose event sits at `d.time + lag`) with a
    // pre-arrival observation, or any pre-dose observation — the init baseline must first decay
    // over the `[start, first_event]` gap, exactly as production's dense walk does (it always
    // starts at `subject_integration_start`). Without this the pre-event obs reads the
    // *undecayed* seed and every later segment inherits the wrong baseline. Gated on
    // `init_fn`: with a zero baseline the gap integrates 0 → 0 (no dose is active before its
    // own arrival), so the non-init walk is byte-identical. The first event's own segment
    // `[start, t_event]` then integrates with that event's end-of-interval params, matching the
    // convention every other segment uses.
    if ode.init_fn.is_some() {
        let start = crate::ode::predictions::subject_integration_start(subject);
        if start.is_finite() && start < cur_t {
            cur_t = start;
        }
    }
    let mut u = init_state.to_vec();
    let vars_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());
    let stack_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());
    // Scratch for the exact `J·g` directional evals in the lagtime saltation (unused
    // when `has_lagtime` is false).
    let mut d1_vars: Vec<Dual1<1>> = Vec::new();
    let mut d1_stack: Vec<Dual1<1>> = Vec::new();
    // Per-dose SS-equilibration trough cache (#642 review #4): a lagged SS dose needs its
    // trough at both the `K_SS_SEED` pre-arrival seed and its later `K_DOSE` event. The seed
    // (processed first, earlier `t_event`) equilibrates once and stashes the trough here; the
    // `K_DOSE` event reuses it instead of re-running the up-to-50-cycle dual SS loop. Entries
    // stay `None` for non-lagged SS doses (which never hit `K_SS_SEED`).
    let mut ss_trough_cache: Vec<Option<Vec<T>>> = vec![None; subject.doses.len()];
    // #653: co-terminating zero-order windows (two doses whose `t+lag+dur` coincide) are
    // processed as one cohort at the first of their `K_ZO_END` events — the general
    // saltation's covariate (`J⁺−J⁻`) correction must fire once for the shared boundary, not
    // once per window. This marks every window already folded into a cohort so its own later
    // `K_ZO_END` event is a no-op (`false` for the common single-window-per-time case).
    let mut zo_end_done: Vec<bool> = vec![false; zero_windows.len()];

    // Full RHS velocity at the current boundary time for one side of a rate-off boundary:
    // the program RHS at `side_params` plus EVERY forcing active (or ending) there — other
    // infusions, other zero-order windows, and the pointwise input rates. The caller builds
    // `v⁻`/`v⁺` for [`general_rate_off_saltation`] by evaluating this on each side's params
    // and then subtracting the boundary's OWN toggling rate from the post side, so all other
    // (frozen) forcings appear identically on both sides — cancelling in the first-order jump
    // yet contributing to the curvature term when the Jacobian jumps across a TV-cov boundary
    // (#653 review #1/#3). Membership is inclusive at the window end (`w_end ≥ t − EPS`), so a
    // co-ending frozen forcing is kept on both sides; only the caller's explicit subtraction
    // turns the boundary's own forcing off. `state`/`t_ev`/`last_dose`/`r_floor` are passed
    // per-call (they change each iteration) while the immutable model/subject data and the
    // shared RHS scratch are captured. `side_prep` is the input-rate forcing set prepared for
    // `side_params` (empty when the model has none); `add_prepared_input_rate_forcing` skips
    // `ZeroOrder` internally, so the zero-order windows below are its sole delivery here.
    let boundary_velocity = |state: &[T],
                             side_params: &[T],
                             side_prep: &[PreparedInputRate<T>],
                             t_ev: f64,
                             last_dose: f64,
                             r_floor: f64|
     -> Vec<T> {
        let eps = crate::ode::predictions::INFUSION_EPS;
        let mut v = vec![T::from_f64(0.0); n_states];
        eval_rhs_anchored::<T>(
            program,
            state,
            side_params,
            t_ev,
            first_dose_time,
            last_dose,
            &mut v,
            &mut vars_cell.borrow_mut(),
            &mut stack_cell.borrow_mut(),
        );
        // Zero-order windows active or ending at `t_ev` (their rate jet is fixed from the
        // dose's own snapshot, so it is identical on both sides — see the helper doc).
        for &(zc, zr, zws, zwe, _, _) in &zero_windows {
            if zc < n_states && zws >= r_floor && zws <= t_ev + eps && zwe >= t_ev - eps {
                v[zc] = v[zc] + zr;
            }
        }
        // Infusions active or ending at `t_ev` (mode-aware effective forcing `inf_eff[k].0`).
        if has_any_infusion {
            for (k, d) in subject.doses.iter().enumerate() {
                if !is_inf(d) || d.cmt < 1 {
                    continue;
                }
                let iws = d.time + lag_val(k);
                let iwe = iws + inf_window_len(k);
                let ci = d.cmt - 1;
                if ci < n_states && iws >= r_floor && iws <= t_ev + eps && iwe >= t_ev - eps {
                    v[ci] = v[ci] + inf_eff[k].0;
                }
            }
        }
        // Pointwise input-rate forcings (first_order / transit / igd / weibull), evaluated at
        // `side_params` — their pre/post difference is the covariate jump for a `mixed` leg.
        if !side_prep.is_empty() {
            crate::ode::predictions::add_prepared_input_rate_forcing::<T>(
                ode,
                side_prep,
                side_params,
                &subject.doses,
                &dose_lagtimes_dual,
                f_bio_at_dose,
                r_floor,
                t_ev,
                &mut v,
            );
        }
        v
    };

    // Prepare the built-in input-rate forcings for an arbitrary PK snapshot (the post-side
    // params at a rate-off boundary). Mirrors the per-event `prepared_forcings` build at the
    // top of the loop; empty when the model has no input-rate forcing (#653 review).
    let prep_for = |params: &[T]| -> Vec<PreparedInputRate<T>> {
        if has_input_rate {
            ode.input_rate
                .iter()
                .map(|f| {
                    f.prepare_dual::<T>(params).expect(
                        "ode_analytical_supported's supported_over_dual() allowlist \
                         guarantees prepare_dual succeeds for every admitted kind",
                    )
                })
                .collect()
        } else {
            Vec::new()
        }
    };

    // TAD anchor: the most recent dose at or before the current segment start. The
    // timeline is sorted and doses sort before a co-timed obs, so this only advances
    // as dose events pass — track it incrementally instead of re-scanning all doses
    // per segment (#451 re-review #6). A dose at the segment start is applied *after*
    // that segment integrates, so it anchors the *next* segment — matching the prior
    // `dt <= cur_t` scan.
    let mut last_dose_eff = f64::NEG_INFINITY;

    // Most-recent EVID 3/4 reset time (`NEG_INFINITY` until the first reset). Infusions
    // whose window started before it are turned off — the reset zeroed the compartments,
    // and production drops them from the active set the same way (`active_infusions(...
    // reset_floor)`, predictions.rs). Without this an infusion straddling a reset would
    // keep adding `F·rate` to the post-reset segments, corrupting `f` and the gradient
    // (#472 review #1).
    let mut reset_floor = f64::NEG_INFINITY;

    // Most-recent record's params, used to integrate a segment ending at a **reset** or
    // infusion-end (neither carries a PK record — mirrors production's `last_pk`). Seeded with
    // the **first-record** snapshot (production's `init_pk`, dose-preferred on ties), not an
    // arbitrary array-first slice: a reset that is itself the first event (an EVID=4 reset+dose
    // at `t = 0`) re-seeds `init` from `last_params` here (`init_taylor_seed_at` in the
    // `K_RESET` branch), and production re-applies `init(&last_pk.values)` with
    // `last_pk = init_pk.unwrap_or_default()` there — so the snapshot must match production's
    // `init_pk`, or the reset re-seed (hence its gradient) diverges for that edge case
    // (#486 review — Copilot). For every later reset `last_params` is overwritten by the most
    // recent record, exactly as production updates `last_pk` (#486).
    let mut last_params: &[T] =
        first_record_pk::<T>(subject, pk_at_dose, pk_at_obs, pk_at_pk_only).unwrap_or(&[]);

    for p in 0..tl.len() {
        let (t_event, kind, idx) = tl[p];
        // Segment `[cur_t, t_event]` uses the params evaluated at `t_event` (NONMEM
        // end-of-interval convention); a reset reuses the previous record's params.
        let params: &[T] = match kind {
            // `K_SS_SEED` shares `pk_at_dose[idx]` with its dose's own (later) `K_DOSE`
            // event — both read the same underlying record's PK snapshot (#486).
            K_DOSE | K_SS_SEED => &pk_at_dose[idx],
            K_PKONLY => &pk_at_pk_only[idx],
            K_OBS => &pk_at_obs[idx],
            _ => last_params, // K_RESET / K_INF_END (not records)
        };
        // Built-in absorption input-rate forcing (#486): hoisted once per event from
        // this event's own PK snapshot `params` — unlike the static walk's single
        // subject-constant snapshot, TV-cov can change params at every event, so the
        // dose-invariant constants (`ln Γ`, `KTR`, …) must be rebuilt per snapshot,
        // mirroring production's per-segment `prepare_input_rates`
        // (`ode_predictions_event_driven`). Computed here (rather than only inside
        // the `t_event > cur_t` segment below) so a `K_DOSE` event whose own segment
        // is degenerate (zero-length, e.g. the very first event) still has it
        // available for the onset saltation below. `ode_analytical_supported`'s
        // `supported_over_dual()` allowlist guarantees every kind reaching here
        // prepares successfully.
        let prepared_forcings: Vec<PreparedInputRate<T>> = if has_input_rate {
            ode.input_rate
                .iter()
                .map(|f| {
                    f.prepare_dual::<T>(params).expect(
                        "ode_analytical_supported's supported_over_dual() allowlist \
                         guarantees prepare_dual succeeds for every admitted kind",
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        if t_event > cur_t {
            // Infusions whose (lagged) window fully spans this segment add a constant
            // forcing `F·rate` to their compartment (the timeline breaks at every window
            // start/end, so a segment is fully inside or outside each window). `F` carries
            // its derivative jet (`f_bio_at_dose[k]`).
            let active_inf: Vec<(usize, T)> = if !has_any_infusion {
                Vec::new()
            } else {
                subject
                    .doses
                    .iter()
                    .enumerate()
                    // `d.cmt >= 1`: a malformed `CMT=0` infusion must not saturate to
                    // compartment 0 and force the wrong state, matching the dose-application
                    // guard (the datareader rejects `CMT=0` upstream) (#472 #6 / #473 #3).
                    .filter(|(_, d)| is_inf(d) && d.cmt >= 1)
                    .filter(|(k, d)| {
                        // (Lagged) window start; an infusion before the most recent reset is
                        // off (#472 review #1) and the window tolerance is production's
                        // `INFUSION_EPS` — both via the shared predicate (#472 review #5/[7]).
                        // The window LENGTH is the bioavailable `inf_window_len` (mode-aware
                        // `F`-scaling, #419), not the raw duration.
                        infusion_spans_segment(
                            d.time + lag_val(*k),
                            inf_window_len(*k),
                            cur_t,
                            t_event,
                            reset_floor,
                        )
                    })
                    // Effective forcing `inf_eff[k].0` (mode-aware: `F·rate` for a
                    // duration-defined infusion, held `rate` for a rate-defined one) (#419).
                    .map(|(k, d)| (d.cmt.saturating_sub(1), inf_eff[k].0))
                    .collect()
            };
            // #486: zero-order windows fully containing this segment add their constant
            // `F·amt·frac/dur` jet — the same full-containment test production's
            // `active_zero_order_inputs` uses (`w_start ≤ cur_t`, `w_end ≥ t_event`), made
            // artifact-free by the `w_end` timeline break above (every segment is wholly
            // inside or outside each window). Reset-aware: a window opened before the most
            // recent EVID 3/4 reset (`w_start < reset_floor`) is off, matching the infusion
            // rule and the static walk. The post-cutoff segment (right end past `w_end`) is
            // excluded here — the rate turns off there, its boundary derivative supplied by
            // the rate-off saltation at `K_ZO_END`.
            let active_zero: Vec<(usize, T)> = if !has_zero_order {
                Vec::new()
            } else {
                zero_windows
                    .iter()
                    .filter(|&&(_, _, w_start, w_end, _, _)| {
                        w_start >= reset_floor
                            && w_start <= cur_t + crate::ode::predictions::INFUSION_EPS
                            && w_end >= t_event - crate::ode::predictions::INFUSION_EPS
                    })
                    .map(|&(cmt, rate, _, _, _, _)| (cmt, rate))
                    .collect()
            };
            let rhs = |us: &[T], ps: &[T], t: f64, du: &mut [T]| {
                eval_rhs_anchored::<T>(
                    program,
                    us,
                    ps,
                    t,
                    first_dose_time,
                    last_dose_eff,
                    du,
                    &mut vars_cell.borrow_mut(),
                    &mut stack_cell.borrow_mut(),
                );
                for &(cmt, fr) in &active_inf {
                    if cmt < du.len() {
                        du[cmt] = du[cmt] + fr;
                    }
                }
                // #486: zero-order windows delivered as a per-segment constant (like an
                // infusion), NOT pointwise — `add_prepared_input_rate_forcing` skips
                // `ZeroOrder` (it `continue`s) precisely so this constant-rate delivery
                // (matching the f64 path) owns the moving-boundary cutoff.
                for &(cmt, rate) in &active_zero {
                    if cmt < du.len() {
                        du[cmt] = du[cmt] + rate;
                    }
                }
                // R_in(tad), via the shared generic helper — `dose_lagtimes_dual`
                // carries the exact continuous `∂R_in/∂lag` through `tad` when the
                // model has an estimated lagtime; the onset saltation at the dose's
                // own arrival is injected separately at the `K_DOSE` event below.
                if !prepared_forcings.is_empty() {
                    crate::ode::predictions::add_prepared_input_rate_forcing::<T>(
                        ode,
                        &prepared_forcings,
                        ps,
                        &subject.doses,
                        &dose_lagtimes_dual,
                        f_bio_at_dose,
                        reset_floor,
                        t,
                        du,
                    );
                }
            };
            // Single save point per segment — a stack array avoids the per-segment
            // heap allocation of `vec![t_event]` (#449 review #14).
            let saveat = [t_event];
            let sol = solve_ode_g(&rhs, &u, (cur_t, t_event), params, &saveat, opts);
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
            cur_t = t_event;
        }
        if kind == K_DOSE {
            let d = &subject.doses[idx];
            // Steady-state (SS=1) dose: load the compartments with the infinite-past
            // pulse train's trough (dual equilibration carries `∂SS/∂(θ,η)`), replacing
            // the running state, *before* the SS dose's own pulse is applied below
            // (mirrors production). `equilibrate_ss_state_g` handles **both** SS bolus and
            // SS infusion (active-rate + quiet window per cycle); only a rate-defined SS
            // infusion under `F ≠ 1`, SS + lagtime, and SS + a non-autonomous RHS route to
            // FD upstream (#473 review #7).
            if is_ss_dose(d) {
                // Reuse the trough already equilibrated at this dose's `K_SS_SEED` pre-arrival
                // seed (lagged SS dose); otherwise equilibrate now (non-lagged SS dose). Both
                // produce the identical trough for the same `pk_at_dose[idx]` — the cache just
                // avoids the second up-to-50-cycle dual SS loop (#642 review #4).
                u = match ss_trough_cache[idx].take() {
                    Some(trough) => trough,
                    None => equilibrate_ss_state_g::<T>(
                        program,
                        n_states,
                        d,
                        f_bio_at_dose[idx],
                        &pk_at_dose[idx],
                        ss_inf(d, idx),
                        opts,
                        &mut d1_vars,
                        &mut d1_stack,
                    ),
                };
            }
            // CMT is 1-based; a malformed `CMT=0` must not silently dose compartment
            // 0 (the datareader rejects it upstream) (#449 review #8).
            if d.cmt >= 1 {
                let cmt_idx = d.cmt - 1;
                if cmt_idx < n_states {
                    if has_input_rate && input_rate_consumes_cmt(ode, d.cmt) {
                        // The dose feeds a built-in absorption forcing (`R_in`), not a
                        // bolus (#430) — its mass flows in continuously via
                        // `add_prepared_input_rate_forcing` in the RHS above, not as an
                        // instantaneous jump here (mirrors the static walk's
                        // `input_rate_consumes_cmt` bolus skip). With an estimated
                        // lagtime, the forcing's *onset* is a discontinuity in `du/dt`
                        // at the dose's lagged arrival (`R_in` switches from inactive to
                        // `R_in(0⁺)` there, since the forcing's domain guard treats
                        // `tad ≤ 0` as flat-zero) — inject the exact rate-on saltation,
                        // the sign-mirror of the lagged-infusion injection below but with
                        // `Δr = Σ frac·R_in(0⁺)` (summed over every forcing feeding this
                        // compartment) in place of a constant infusion rate. Without
                        // lagtime the dose's arrival is a fixed (non-dual) boundary, so
                        // no saltation is needed — the continuous forcing above already
                        // carries every other parameter's exact sensitivity.
                        if has_lagtime {
                            let lag = pk_at_dose[idx][dose_lag_slot[idx]];
                            let dlag = jet_only(lag);
                            let dose_mass = f_bio_at_dose[idx] * T::from_f64(d.amt);
                            let mut onset = T::from_f64(0.0);
                            for (f, prep) in ode.input_rate.iter().zip(&prepared_forcings) {
                                if f.cmt == cmt_idx {
                                    onset = onset
                                        + f.frac(&pk_at_dose[idx]) * prep.rate_at_zero(dose_mass);
                                }
                            }
                            // #486: a zero-order window feeding this compartment also switches
                            // on at the lagged arrival — its constant rate is the window's
                            // rate-on `Δr` (`rate_at_zero` is zero for `ZeroOrder`, so the loop
                            // above contributed nothing for it). One window per dose, so match
                            // this dose's own window (`zk == idx`); a `mixed` biphasic dose
                            // (first-order + zero-order on one compartment) sums both onsets
                            // here, exactly as the two forcings switch on together.
                            for &(zcmt, zrate, _, _, _, zk) in &zero_windows {
                                if zk == idx && zcmt == cmt_idx {
                                    onset = onset + zrate;
                                }
                            }
                            inject_rate_saltation::<T>(
                                &mut u,
                                cmt_idx,
                                onset,
                                dlag,
                                -1.0,
                                program,
                                &pk_at_dose[idx],
                                t_event,
                                first_dose_time,
                                t_event,
                                &mut d1_vars,
                                &mut d1_stack,
                            );
                        }
                    } else if is_inf(d) {
                        // Infusion: no bolus — the rate `F·rate` enters via the segment
                        // forcing above over `[t_dose+lag, t_dose+lag+dur]`. With lagtime,
                        // the window's *start* shifts, so inject the rate-on event-time
                        // saltation (`s = −1`). This ALSO applies to an SS dose (#486), but
                        // needs one extra step first (below): unlike a regular lagged dose,
                        // whose pre-arrival state acquires an `∂/∂lag = -g(x⁻)` jet "for
                        // free" by chaining through a fixed-duration prior segment,
                        // `equilibrate_ss_state_g`'s trough has *zero* `∂/∂lag` (the periodic
                        // recurrence is anchored to the pulse, not to wall-clock arrival
                        // time) — so that embedded jet must be given to it explicitly before
                        // the (otherwise unmodified) rate-on saltation is exact.
                        if has_lagtime {
                            let lag = pk_at_dose[idx][dose_lag_slot[idx]];
                            let dlag = jet_only(lag);
                            if is_ss_dose(d) {
                                // The rate-on saltation below assumes `u`'s own `∂/∂lag`
                                // already equals `-g(u)` (the "embedded jet" a genuinely
                                // flowing pre-arrival residual acquires for free, by chaining
                                // through a fixed-duration prior segment — see the module's
                                // #486 dose-time-saltation derivation). `u` is the SS trough
                                // here, whose `∂/∂lag` is exactly zero instead — so give it
                                // that embedded jet explicitly, by flowing it `-δlag` under
                                // the bare (unforced) RHS, before the unmodified rate-on
                                // injection below (which is then exact, same as a regular
                                // lagged infusion).
                                let bare_rhs = |us: &[T], ps: &[T], t: f64, du: &mut [T]| {
                                    eval_rhs_anchored::<T>(
                                        program,
                                        us,
                                        ps,
                                        t,
                                        first_dose_time,
                                        t_event,
                                        du,
                                        &mut vars_cell.borrow_mut(),
                                        &mut stack_cell.borrow_mut(),
                                    );
                                };
                                extend_flow_by_dual_duration_g::<T>(
                                    &mut u,
                                    -dlag,
                                    program,
                                    bare_rhs,
                                    &pk_at_dose[idx],
                                    t_event,
                                    first_dose_time,
                                    t_event,
                                    &mut d1_vars,
                                    &mut d1_stack,
                                );
                            }
                            // Rate-on at `t+lag`: the start shifts with `lag` only (not with
                            // the bioavailable window length); `dr` = effective forcing. Its
                            // `J·g` eval is anchored at `t_event` (TAD=0, this dose just
                            // arrived), not the stale previous-dose `last_dose_eff` — which
                            // gave a TAD-referencing RHS the wrong TAD (#472 review #4).
                            inject_rate_saltation::<T>(
                                &mut u,
                                cmt_idx,
                                inf_eff[idx].0,
                                dlag,
                                -1.0,
                                program,
                                &pk_at_dose[idx],
                                t_event,
                                first_dose_time,
                                t_event,
                                &mut d1_vars,
                                &mut d1_stack,
                            );
                        }
                    } else if has_lagtime {
                        // Estimated-lagtime event-time injection. The dose arrives at
                        // `τ = t_dose + lag`; the corrected post-dose state, as a function
                        // of `δlag = lag − lag.val()` (value 0), is the pre-dose state time-
                        // shifted to the true arrival and then flowed back over the fixed
                        // integration step (`x_inject = Ψ_{−δlag}(x⁻(τ) + Δ)`):
                        //   x⁺ += D·δlag + (½ẋ̇⁻ + ½ẋ̇⁺ − J⁺·ẋ⁻)·δlag²,
                        // D = g(x⁻) − g(x⁺), ẋ̇± = J(x±)·g(x±), and the cross term J⁺·ẋ⁻ is
                        // the post-dose Jacobian applied to the *pre*-dose velocity. The
                        // integrator then propagates this exactly (across occasion /
                        // covariate boundaries, where the static time-shift identity fails).
                        // `δlag` has value 0, so the f64 value (dose at `t_event`) is
                        // unchanged. (For the first dose `x⁻ = 0`, `g(x⁻) = 0`, so this
                        // reduces to `−g(x⁺)·δlag + ½ẋ̇⁺·δlag²` — the single-dose time-shift.)
                        let params = &pk_at_dose[idx];
                        let lag = params[dose_lag_slot[idx]];
                        let dlag = jet_only(lag);
                        // TAD anchor for the *pre*-dose velocity `g(x⁻)`: the most recent
                        // earlier dose. On the first dose `last_dose_eff` is `NEG_INFINITY`,
                        // which `eval_rhs_anchored` turns into `TAD = NaN` — fine for a
                        // TAD-independent RHS (the comment's `g(x⁻)=0`), but it poisons the
                        // saltation for an RHS that references the `TAD` builtin. Fall back
                        // to `t_event` (TAD=0) so `g_minus` stays finite (#472 review #3).
                        let pre_anchor = if last_dose_eff.is_finite() {
                            last_dose_eff
                        } else {
                            t_event
                        };
                        // `x⁻` = the pre-bolus running state. For a plain dose this is the
                        // continuing residual trajectory, and `g(x⁻)` is its own velocity —
                        // the term that (via `J·g(x⁻)`, propagated by the ordinary sensitivity
                        // equation over the *fixed* duration to any later event) reproduces the
                        // "the incoming segment's own duration also depends on lag" effect. For
                        // an **SS** dose, `u` holds the periodic trough, which by construction
                        // does *not* flow toward the event as lag shifts (the recurrence is
                        // anchored to the pulse, not to wall-clock time) — so there is no
                        // incoming segment to account for, exactly like a genuine first dose
                        // with no prior residual: `g(x⁻)` is treated as zero (skip the eval),
                        // leaving only the `−g(x⁺)·δlag` term (the "later fixed-time
                        // observations see a shifted elapsed time since arrival" effect, #486).
                        let u_minus = u.clone();
                        let mut g_minus = vec![T::from_f64(0.0); n_states];
                        if !is_ss_dose(d) {
                            eval_rhs_anchored::<T>(
                                program,
                                &u_minus,
                                params,
                                t_event,
                                first_dose_time,
                                pre_anchor,
                                &mut g_minus,
                                &mut vars_cell.borrow_mut(),
                                &mut stack_cell.borrow_mut(),
                            );
                        }
                        u[cmt_idx] = u[cmt_idx] + f_bio_at_dose[idx] * T::from_f64(d.amt);
                        let mut g_plus = vec![T::from_f64(0.0); n_states];
                        eval_rhs_anchored::<T>(
                            program,
                            &u,
                            params,
                            t_event,
                            first_dose_time,
                            t_event,
                            &mut g_plus,
                            &mut vars_cell.borrow_mut(),
                            &mut stack_cell.borrow_mut(),
                        );
                        // dD/dt values via exact `J·g` directional evals (Dual1<1>).
                        let params_d1: Vec<Dual1<1>> =
                            params.iter().map(|p| Dual1::constant(p.val())).collect();
                        let jg_minus = jdotg_value::<T>(
                            program,
                            n_states,
                            &u_minus,
                            &g_minus,
                            &params_d1,
                            t_event,
                            first_dose_time,
                            pre_anchor,
                            &mut d1_vars,
                            &mut d1_stack,
                        );
                        let jg_plus = jdotg_value::<T>(
                            program,
                            n_states,
                            &u,
                            &g_plus,
                            &params_d1,
                            t_event,
                            first_dose_time,
                            t_event,
                            &mut d1_vars,
                            &mut d1_stack,
                        );
                        // Cross term J⁺·ẋ⁻: the post-dose Jacobian applied to the pre-dose
                        // velocity (directional eval at the post-dose state `u` along `g_minus`).
                        let jg_cross = jdotg_value::<T>(
                            program,
                            n_states,
                            &u,
                            &g_minus,
                            &params_d1,
                            t_event,
                            first_dose_time,
                            t_event,
                            &mut d1_vars,
                            &mut d1_stack,
                        );
                        let dlag2 = dlag * dlag;
                        for c in 0..n_states {
                            // δlag² coefficient = ½ẋ̇⁻ + ½ẋ̇⁺ − J⁺·ẋ⁻.
                            let coef2 = T::from_f64(0.5 * (jg_minus[c] + jg_plus[c]) - jg_cross[c]);
                            u[c] = u[c] + (g_minus[c] - g_plus[c]) * dlag + coef2 * dlag2;
                        }
                    } else {
                        u[cmt_idx] = u[cmt_idx] + f_bio_at_dose[idx] * T::from_f64(d.amt);
                    }
                }
            }
            // This dose now anchors TAD for every later segment, at its lagged arrival
            // `d.time + lag_val(idx)` (= `t_event` for a dose), matching production.
            last_dose_eff = last_dose_eff.max(t_event);
            last_params = &pk_at_dose[idx];
        } else if kind == K_OBS {
            states[idx].copy_from_slice(&u);
            last_params = &pk_at_obs[idx];
        } else if kind == K_PKONLY {
            // EVID=2 covariate-only record: no state jump or observation, but `$PK` has
            // run at this record, and the next segment must use its params (with κ fixed
            // at zero under IOV, matching production `predict_iov`).
            last_params = &pk_at_pk_only[idx];
        } else if kind == K_SS_SEED {
            // SS + estimated lagtime pre-arrival seed (#486): between this dose's raw
            // record time (this event) and its lagged arrival (the later `K_DOSE` event
            // for the same `idx`), the running state is the *previous* interval's
            // steady-state tail — `ss_state_at_phase_g` advances the trough forward by
            // `phase = II − lag` (the dual carries `lag`'s jet exactly as
            // `equilibrate_ss_state_g`'s per-cycle rate-off saltation does, #530). The
            // segment from here to the lagged arrival then flows this seed forward
            // through the walk's ordinary integration.
            let d = &subject.doses[idx];
            let lag = pk_at_dose[idx][dose_lag_slot[idx]];
            let phase = T::from_f64(d.ii) - lag;
            let inf = ss_inf(d, idx);
            // Equilibrate the SS trough once here and cache it, so this lagged SS dose's later
            // `K_DOSE` event reuses it instead of re-running the dual SS loop (#642 review #4).
            let trough = equilibrate_ss_state_g::<T>(
                program,
                n_states,
                d,
                f_bio_at_dose[idx],
                &pk_at_dose[idx],
                inf,
                opts,
                &mut d1_vars,
                &mut d1_stack,
            );
            ss_trough_cache[idx] = Some(trough.clone());
            u = ss_state_at_phase_g::<T>(
                program,
                n_states,
                d,
                f_bio_at_dose[idx],
                &pk_at_dose[idx],
                inf,
                phase,
                trough,
                opts,
                &mut d1_vars,
                &mut d1_stack,
            );
            last_params = &pk_at_dose[idx];
        } else if kind == K_INF_END {
            // Infusion window end: the rate turns off (the next segment's `active_inf`
            // excludes it). Not a record — no state change, no `last_params` update. The
            // window end `t+lag+t_inf` is a moving boundary: it shifts with `lag` (any
            // lagtime) and, for a rate-defined infusion, with `F` (the bioavailable window
            // length `F·amt/rate`, #419). Inject the rate-off saltation (`s = +1`) with the
            // combined shift `δ = δlag + δt_inf` (the single dual carries the lag×F cross
            // terms) — but only if the infusion is still active: one whose window was cut
            // off by an intervening EVID 3/4 reset (`start < reset_floor`) was already turned
            // off, so its rate-off correction must not fire (#472 review #2).
            //
            // Like the zero-order window end below, when a time-varying covariate changes the
            // segment's PK params across this (non-record) boundary the RHS **Jacobian jumps**,
            // so the closed-form forcing-only `inject_rate_saltation` (one param set) misses the
            // `(J⁺−J⁻)·x` term. Take it only when there is no jump (`pre == post`, the common
            // case — byte-identical to the pre-#653 path); otherwise route through the general
            // `g⁻−g⁺` saltation with full pre/post velocities (#653 review #3).
            let d = &subject.doses[idx];
            let is_rate_defined = matches!(d.infusion_def, crate::types::InfusionDef::RateDefined);
            // #530: a modeled-duration dose's window end `t+dur` moves with `D` (the `dtinf`
            // jet below carries `∂/∂D`); a modeled-rate dose is already `is_rate_defined`.
            let is_modeled = modeled_at(idx).is_some();
            if (has_lagtime || is_rate_defined || is_modeled)
                && d.time + lag_val(idx) >= reset_floor
                && d.cmt >= 1
                && d.cmt - 1 < n_states
            {
                let cmt = d.cmt - 1;
                let dlag = if has_lagtime {
                    jet_only(pk_at_dose[idx][dose_lag_slot[idx]])
                } else {
                    T::from_f64(0.0)
                };
                let dtinf = jet_only(inf_eff[idx].1);
                let d_off = dlag + dtinf;
                // Pre-boundary params = the segment ending here (`last_params`, this being a
                // non-record boundary). Post-boundary params = the next real record's snapshot
                // at/after this instant (NONMEM end-of-interval); co-located rate-off siblings
                // (`K_INF_END`/`K_ZO_END` at the same time) carry no params and are skipped, and
                // a later-time boundary segment keeps `last_params` (#653 review #2).
                let pre_params = last_params;
                let post_params: &[T] = 'lookahead: {
                    let leps = crate::ode::predictions::INFUSION_EPS;
                    for q in (p + 1)..tl.len() {
                        let (tq, kq, iq) = tl[q];
                        if (kq == K_INF_END || kq == K_ZO_END) && (tq - t_event).abs() <= leps {
                            continue;
                        }
                        break 'lookahead match kq {
                            K_DOSE | K_SS_SEED => &pk_at_dose[iq],
                            K_PKONLY => &pk_at_pk_only[iq],
                            K_OBS => &pk_at_obs[iq],
                            _ => last_params,
                        };
                    }
                    last_params
                };
                if pk_snapshot_equal(pre_params, post_params) {
                    inject_rate_saltation::<T>(
                        &mut u,
                        cmt,
                        inf_eff[idx].0,
                        d_off,
                        1.0,
                        program,
                        pre_params,
                        t_event,
                        first_dose_time,
                        last_dose_eff,
                        &mut d1_vars,
                        &mut d1_stack,
                    );
                } else {
                    let prep_post = prep_for(post_params);
                    let v_minus = boundary_velocity(
                        &u,
                        pre_params,
                        &prepared_forcings,
                        t_event,
                        last_dose_eff,
                        reset_floor,
                    );
                    let mut v_plus = boundary_velocity(
                        &u,
                        post_params,
                        &prep_post,
                        t_event,
                        last_dose_eff,
                        reset_floor,
                    );
                    // Turn this infusion's own forcing off on the post side; every other
                    // (frozen) forcing stays identical on both sides.
                    v_plus[cmt] = v_plus[cmt] - inf_eff[idx].0;
                    general_rate_off_saltation::<T>(
                        &mut u,
                        program,
                        n_states,
                        pre_params,
                        post_params,
                        &v_minus,
                        &v_plus,
                        d_off,
                        t_event,
                        first_dose_time,
                        last_dose_eff,
                        &mut d1_vars,
                        &mut d1_stack,
                    );
                }
            }
        } else if kind == K_ZO_END {
            // #486: zero-order window end (`idx` = the `zero_windows` index). The constant
            // rate turns off (the next segment's `active_zero` excludes it). The window end
            // `w_end = d.time + lag + dur` is a moving boundary: it shifts with `dur` (always)
            // and, under an estimated lagtime, with `lag`. Only fired if the window is still
            // active — one turned off by an intervening EVID 3/4 reset (`w_start < reset_floor`)
            // was already dropped from `active_zero`, so its rate-off correction must not.
            //
            // The state is continuous across `w_end`; only `du/dt` jumps. Unlike the static
            // walk (one param set) or a lagtime-only walk, TV covariates make the walk break at
            // `w_end` (which is NOT a record) with the *previous* record's params on the pre-side
            // (`last_params`) and the *next* record's params on the post-side (NONMEM
            // end-of-interval convention) — so the RHS **Jacobian jumps** across `w_end`, not
            // just the forcing. A forcing-only saltation (`inject_rate_saltation`, one param
            // set) would then miss the `(J⁺−J⁻)·x` term (empirically a several-percent error).
            // So on a genuine jump use the **general** `g⁻−g⁺` saltation
            // ([`general_rate_off_saltation`]) with velocities `v⁻`/`v⁺` that include EVERY
            // concurrently-active forcing (other zero-order windows — e.g. `dur` > dosing
            // interval — infusions, and a `mixed` leg's pointwise first-order R_in), so the
            // curvature is exact when `J⁺ ≠ J⁻` (#653 review #1). When `pre == post` (no TV-cov,
            // or `w_end` between equal covariate records) it reduces exactly to the closed-form
            // `+rate·δ`, so take that cheaper path (#653 review #5).
            //
            // Co-terminating windows (two doses whose `t+lag+dur` coincide) share one physical
            // boundary: they are processed together as a cohort at the first of their `K_ZO_END`
            // events so the covariate (`J⁺−J⁻`) correction fires ONCE, not once per window
            // (#653 review #2); `zo_end_done` no-ops the siblings. (Residual exotic limitation:
            // a zero-order window and an infusion — or a window with an *independent* `dur` jet —
            // co-terminating at the exact same instant under a covariate varying across it would
            // double-count that one covariate correction; the rates themselves stay exact.)
            if !zo_end_done[idx] {
                let ceps = crate::ode::predictions::INFUSION_EPS;
                // Active windows sharing this boundary instant (the cohort); mark them (and this
                // `idx`, even if reset-cut) done so their own later events are no-ops.
                let cohort: Vec<usize> = (0..zero_windows.len())
                    .filter(|&j| {
                        !zo_end_done[j]
                            && zero_windows[j].0 < n_states
                            && zero_windows[j].2 >= reset_floor
                            && (zero_windows[j].3 - t_event).abs() <= ceps
                    })
                    .collect();
                for &j in &cohort {
                    zo_end_done[j] = true;
                }
                zo_end_done[idx] = true;
                if !cohort.is_empty() {
                    // Shared shift `δ = δlag + δdur` (co-terminating ⇒ same `t+lag+dur` ⇒ same
                    // jet in the common case). Use the representative (first) cohort window.
                    let (_, _, _, _, dur_rep, k_rep) = zero_windows[cohort[0]];
                    let dlag = if has_lagtime {
                        jet_only(pk_at_dose[k_rep][dose_lag_slot[k_rep]])
                    } else {
                        T::from_f64(0.0)
                    };
                    let d_off = dlag + jet_only(dur_rep);
                    // Pre-boundary params = the segment ending at `w_end` (`last_params`, this
                    // being a non-record boundary). Post-boundary params = the next real record
                    // at/after this instant, skipping co-located rate-off siblings (#653 #2).
                    let pre_params = last_params;
                    let post_params: &[T] = 'lookahead: {
                        for q in (p + 1)..tl.len() {
                            let (tq, kq, iq) = tl[q];
                            if (kq == K_INF_END || kq == K_ZO_END) && (tq - t_event).abs() <= ceps {
                                continue;
                            }
                            break 'lookahead match kq {
                                K_DOSE | K_SS_SEED => &pk_at_dose[iq],
                                K_PKONLY => &pk_at_pk_only[iq],
                                K_OBS => &pk_at_obs[iq],
                                _ => last_params,
                            };
                        }
                        last_params
                    };
                    if pk_snapshot_equal(pre_params, post_params) {
                        // No Jacobian jump — cheap closed-form saltation per cohort window (each
                        // carries its own shift, in case cohort members differ in `dur`/`lag`).
                        for &j in &cohort {
                            let (cmt, rate, _, _, dur_j, k_j) = zero_windows[j];
                            let dlag_j = if has_lagtime {
                                jet_only(pk_at_dose[k_j][dose_lag_slot[k_j]])
                            } else {
                                T::from_f64(0.0)
                            };
                            let d_off_j = dlag_j + jet_only(dur_j);
                            inject_rate_saltation::<T>(
                                &mut u,
                                cmt,
                                rate,
                                d_off_j,
                                1.0,
                                program,
                                pre_params,
                                t_event,
                                first_dose_time,
                                last_dose_eff,
                                &mut d1_vars,
                                &mut d1_stack,
                            );
                        }
                    } else {
                        // Genuine Jacobian jump: build full pre/post velocities (all concurrent
                        // forcings included) and turn every cohort window's own rate off on the
                        // post side; frozen forcings then appear identically on both sides.
                        let prep_post = prep_for(post_params);
                        let v_minus = boundary_velocity(
                            &u,
                            pre_params,
                            &prepared_forcings,
                            t_event,
                            last_dose_eff,
                            reset_floor,
                        );
                        let mut v_plus = boundary_velocity(
                            &u,
                            post_params,
                            &prep_post,
                            t_event,
                            last_dose_eff,
                            reset_floor,
                        );
                        for &j in &cohort {
                            let (cmt, rate, _, _, _, _) = zero_windows[j];
                            v_plus[cmt] = v_plus[cmt] - rate;
                        }
                        general_rate_off_saltation::<T>(
                            &mut u,
                            program,
                            n_states,
                            pre_params,
                            post_params,
                            &v_minus,
                            &v_plus,
                            d_off,
                            t_event,
                            first_dose_time,
                            last_dose_eff,
                            &mut d1_vars,
                            &mut d1_stack,
                        );
                    }
                }
            }
        } else {
            // EVID 3/4 reset. Production re-applies the initial conditions here —
            // `u = ode.initial_state(&last_pk.values)` — which restores each `init(...)`
            // compartment to its value evaluated with the params in effect at the reset
            // (`last_pk`) and zeros every compartment without an `init`. Mirror that on the dual
            // walk (#486): with an `init(...)` present, re-seed the state from the reset-event
            // snapshot via the shared Taylor seed — `params` is `last_params` for a `K_RESET`
            // event (the `_` arm of the segment-params match), i.e. the most-recent record's PK,
            // matching production's `last_pk` — so the post-reset jet carries the correct
            // `∂init/∂(θ,η)` at *that* snapshot rather than the subject-level first-record seed.
            // With no `init(...)` the seed's `base`/derivatives are all zero, so this reduces to
            // the previous "zero every compartment" behaviour (byte-identical). For EVID=4 the
            // same-time dose sorts after the reset (`K_RESET < K_DOSE`), so it lands on the
            // re-seeded state.
            match ode.init_fn.as_ref() {
                Some(f) => u = init_taylor_seed_at::<T>(f.as_ref(), params, pk_indices, n_states),
                None => {
                    for x in u.iter_mut() {
                        *x = T::from_f64(0.0);
                    }
                }
            }
            // Turn off any infusion that started before this reset (matches production's
            // `reset_floor`) — the active-set filter and the rate-off saltation below both
            // consult it (#472 review #1/#2).
            reset_floor = t_event;
        }
    }
    states
}

/// True when an observation time `ot` coincides with a segment break / solver
/// save time `t`. Both are produced by arithmetic on the same CSV time values
/// (`dose.time`, `t_end`, the solver's interpolated save points), so value-equal
/// times can differ by a few ULPs; matching on `f64::to_bits` would silently miss
/// them and leave the observation's state (hence its sensitivity) at zero — the
/// hardening called for in issue #410. The tolerance is scaled to the time
/// magnitude and is many orders of magnitude tighter than any real
/// inter-observation spacing, so it never conflates distinct observations.
#[inline]
fn obs_time_matches(ot: f64, t: f64) -> bool {
    (ot - t).abs() <= 1e-9 * (1.0 + ot.abs().max(t.abs()))
}

/// Integrate the dual state through the subject's bolus + infusion events,
/// capturing the full state vector at every observation time. Returns one state
/// vector per observation (parallel to `subject.obs_times`); the caller applies
/// the readout. `f_bio` is the bioavailability (scales bolus amount and infusion
/// rate, carrying its derivative). Generic over the dual type `T`: `Dual2<N>` for
/// the full outer gradient (value + grad + Hessian), `Dual1<N>` for the light inner
/// η-gradient (value + grad only) — issue #410.
#[allow(clippy::too_many_arguments)]
fn integrate_g<T: crate::sens::num::PkNum>(
    program: &crate::parser::model_parser::OdeRhsProgram,
    n_states: usize,
    subject: &Subject,
    ode: &OdeSpec,
    prepared_forcings: &[PreparedInputRate<T>],
    params_dual: &[T],
    dose_f_bio: &[T],
    init_state: &[T],
    first_dose_time: f64,
    opts: &crate::ode::solver::OdeSolverOptions,
) -> Option<Vec<Vec<T>>> {
    let n_obs = subject.obs_times.len();
    let mut states: Vec<Vec<T>> = vec![vec![T::from_f64(0.0); n_states]; n_obs];
    let mut recorded = vec![false; n_obs];
    let mut u = init_state.to_vec();
    // (Estimated lagtime is handled on the event-driven walk, not here — see
    // `ode_subject_supported`. This static walk applies doses at their record times.)

    // Sorted `(obs_time, index)` for O(log n) tolerance lookup at each break time
    // and solver save point, replacing the per-query linear scan over all
    // observations (PR #438 review). The precise `obs_time_matches` test still
    // gates each candidate; the sort only narrows the search window.
    let mut sorted_obs: Vec<(f64, usize)> = subject.obs_times.iter().copied().zip(0..).collect();
    sorted_obs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Record `src` at every not-yet-recorded observation whose time matches `q`.
    let record_at = |q: f64, src: &[T], states: &mut [Vec<T>], recorded: &mut [bool]| {
        // Candidates lie within the relative tolerance band; widen slightly for the
        // binary-search bounds, then confirm each with the exact `obs_time_matches`.
        let slack = 2e-9 * (1.0 + q.abs());
        let lo = sorted_obs.partition_point(|&(t, _)| t < q - slack);
        for &(t, j) in &sorted_obs[lo..] {
            if t > q + slack {
                break;
            }
            if !recorded[j] && obs_time_matches(t, q) {
                states[j].copy_from_slice(src);
                recorded[j] = true;
            }
        }
    };

    // #530: zero-order absorption windows. A dose feeding a `zero_order(dur)` forcing
    // delivers a constant `F·amt·frac/dur` over `[dose.time, dose.time + dur]` — mechanically
    // an infusion, but `dur` is an estimated parameter so the window END is a moving boundary.
    // The constant rate jet carries the smooth magnitude term (`∂/∂dur` of `F·amt·frac/dur`);
    // the boundary term is the rate-off saltation injected at `w_end` below. Mirrors the f64
    // `zero_order_windows` / `active_zero_order_inputs` (lagtime is gated off this static walk,
    // so `w_start = dose.time`). Stored as `(cmt, rate_jet, w_start, dur_jet)` (#530).
    let zero_windows: Vec<(usize, T, f64, T)> = if ode
        .input_rate
        .iter()
        .any(|f| f.kind == crate::pk::absorption::InputRateKind::ZeroOrder)
    {
        subject
            .doses
            .iter()
            .enumerate()
            .filter_map(|(k, d)| {
                if d.amt <= 0.0 {
                    return None;
                }
                // One zero-order forcing per dose compartment (the parser rejects > 1); shared
                // with the event-driven walk via `zero_order_forcing_for_dose` (#653 review #7).
                // Re-preparing from the subject-static `params_dual` yields the same duration jet
                // as the previously-indexed `prepared_forcings[fi]`.
                let (f, dur) = zero_order_forcing_for_dose(ode, d, params_dual)?;
                let frac = f.frac::<T>(params_dual);
                let rate = dose_f_bio[k] * T::from_f64(d.amt) * frac / dur;
                Some((d.cmt.saturating_sub(1), rate, d.time, dur))
            })
            .collect()
    } else {
        Vec::new()
    };
    let has_zero_order = !zero_windows.is_empty();

    // Break the timeline at every dose time and — for infusions — the
    // infusion-end time, so each segment is fully inside or outside every
    // infusion window (the rate forcing is then constant over a segment).
    let t_last = subject.obs_times.iter().cloned().fold(0.0_f64, f64::max);
    // Start integration at the subject's first event (NONMEM semantics), not at a
    // fixed t = 0 — so an off-zero TIME column is not integrated over a phantom
    // `[0, first_record]` window. Mirrors the production dense walk and the
    // event-driven `cur_t = timeline[0]` start (#573).
    let mut break_times: Vec<f64> =
        vec![crate::ode::predictions::subject_integration_start(subject)];
    for dose in &subject.doses {
        break_times.push(dose.time);
        if dose.is_infusion() {
            break_times.push(dose.time + dose.duration);
        }
    }
    // #530: break at each zero-order window end `w_start + dur` so every segment is fully
    // inside or outside the window (the full-containment filter below relies on this), and
    // the rate-off saltation lands on an exact break.
    for &(_, _, w_start, dur) in &zero_windows {
        break_times.push(w_start + dur.val());
    }
    // EVID 3/4 reset times also break the timeline so the state can be zeroed
    // there (the datareader places obs/dose/reset on one absolute timeline).
    for &rt in &subject.reset_times {
        break_times.push(rt);
    }
    break_times.push(t_last);
    // NaN-safe sort: a malformed dose/reset time (e.g. `duration = amt/rate = NaN`)
    // must not panic on the `None` `partial_cmp` returns — mirrors the production
    // f64 walk (`pk::event_driven`) (PR #381 review #13).
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    // Degenerate single-instant timeline (one observation, no dose, off zero):
    // keep a second identical break so the loop runs once and `record_at(t_start)`
    // captures the observation at the first record from the initial state.
    if break_times.len() < 2 {
        break_times.push(break_times[0]);
    }

    // Reusable scratch for the RHS evaluation across all stages.
    let vars_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());
    let stack_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());
    // Scratch for the zero-order rate-off saltation's `J·g` directional eval (#530).
    let mut d1_vars: Vec<Dual1<1>> = Vec::new();
    let mut d1_stack: Vec<Dual1<1>> = Vec::new();

    // `dose_f_bio` (one bioavailability per dose, resolved per compartment by the
    // caller — `F{cmt}` else bare `PK_IDX_F`, #486) is built once per subject and
    // indexed by dose position throughout: the bolus load, the infusion rate forcing,
    // and the shared absorption-forcing helper all read `dose_f_bio[k]` (#451 / #433
    // review #6).
    debug_assert_eq!(dose_f_bio.len(), subject.doses.len());

    // Skip the per-segment active-infusion scan/alloc entirely for the common bolus-only /
    // oral subject (no infusion → empty active set every segment) — mirrors the
    // `integrate_tvcov_g` short-circuit (#472 review round 2 #7).
    let has_any_infusion = subject
        .doses
        .iter()
        .any(crate::ode::predictions::is_real_infusion);

    // Most-recent EVID 3/4 reset time (`NEG_INFINITY` until the first reset). An infusion
    // whose window *straddles* a reset must stop contributing afterward — the reset zeroed
    // the state, and production drops such infusions from the active set via `reset_floor`
    // (`active_infusions`, predictions.rs). Without this the static walk leaks `F·rate` into
    // the post-reset segments (the event-driven walk's #472 review-1 fix, mirrored here for
    // the static `integrate_g` twin) (#472 review round 2 #1).
    let mut reset_floor = f64::NEG_INFINITY;

    for w in 0..(break_times.len() - 1) {
        let t_start = break_times[w];
        let t_end = break_times[w + 1];

        // EVID 3/4 reset: re-seed the state to the initial conditions at this time, *before*
        // the same-time dose (EVID=4 = reset + dose), and record the reset time so an
        // infusion whose window straddles it is turned off below.
        if subject
            .reset_times
            .iter()
            .any(|&rt| (rt - t_start).abs() < 1e-12)
        {
            u.copy_from_slice(init_state);
            reset_floor = t_start;
        }

        // Apply bolus doses (non-infusions) at t_start: u[cmt] += F·amt. CMT is 1-based;
        // a malformed `CMT=0` must not silently dose compartment 0 (#449 #8). A compartment
        // fed by a built-in absorption input rate is skipped here — the dose feeds R_in
        // (the forcing in the RHS below), not a bolus (#430). `F` is per dose compartment
        // via `dose_f_bio[k]` (#486).
        for (k, dose) in subject.doses.iter().enumerate() {
            if !dose.is_infusion()
                && (dose.time - t_start).abs() < 1e-12
                && dose.cmt >= 1
                && !input_rate_consumes_cmt(ode, dose.cmt)
            {
                let cmt_idx = dose.cmt - 1;
                if cmt_idx < n_states {
                    u[cmt_idx] = u[cmt_idx] + dose_f_bio[k] * T::from_f64(dose.amt);
                }
            }
        }

        // Record any observation at t_start (after the dose). `t_start` is a break
        // time built by arithmetic on dose/reset times, so an observation that
        // coincides with it can be value-equal but bit-different — match by
        // tolerance, not bit pattern (issue #410).
        record_at(t_start, &u, &mut states, &mut recorded);

        // Last effective dose at or before the segment start (the TAD anchor). Shared by the
        // zero-order saltation below and the RHS forcing — one fold over `subject.doses` per
        // segment, so the two consumers can never drift on a different filter epsilon (#530
        // review finding 4).
        let last_dose_eff = subject
            .doses
            .iter()
            .map(|d| d.time)
            .filter(|&dt| dt <= t_start + 1e-12)
            .fold(f64::NEG_INFINITY, f64::max);

        // #530: zero-order rate-off saltation. At a window end `w_end = dose.time + dur`, the
        // constant rate turns off; the boundary moves with `dur`, so inject the moving-boundary
        // sensitivity (`s = +1`, the sign-mirror of the lagtime dose-start saltation). Fired
        // *after* recording an obs at `w_end` (so an obs at the boundary reads the rate still
        // on, matching the closed `(0, dur]` window) and only for a window not turned off by an
        // intervening EVID 3/4 reset (`w_start >= reset_floor`). The state value is continuous
        // — only its `∂/∂dur` jet changes.
        if has_zero_order {
            for &(cmt, rate, w_start, dur) in &zero_windows {
                let w_end = w_start + dur.val();
                if w_start >= reset_floor && (w_end - t_start).abs() < 1e-9 {
                    let ddur = dur - T::from_f64(dur.val());
                    inject_rate_saltation::<T>(
                        &mut u,
                        cmt,
                        rate,
                        ddur,
                        1.0,
                        program,
                        params_dual,
                        t_start,
                        first_dose_time,
                        last_dose_eff,
                        &mut d1_vars,
                        &mut d1_stack,
                    );
                }
            }
        }

        if (t_end - t_start).abs() < 1e-15 {
            continue;
        }

        // Observation times in (t_start, t_end]; always include t_end so `u`
        // advances for the next segment.
        let mut saveat: Vec<f64> = subject
            .obs_times
            .iter()
            .filter(|&&t| t > t_start + 1e-12 && t <= t_end + 1e-12)
            .cloned()
            .collect();
        if saveat.last().map_or(true, |&l| (l - t_end).abs() > 1e-12) {
            saveat.push(t_end);
        }
        saveat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        // `F·rate` to their compartment (the break times guarantee a segment is fully
        // inside or outside each infusion window). `F` is resolved per dose compartment
        // (`dose_f_bio[k]`, #486); pre-scale `F·rate` as a dual once per segment so the RHS
        // closure (every RK45 stage) just adds it. Skipped for bolus-only subjects; the
        // `cmt >= 1` guard, the `reset_floor` (an infusion before the most recent reset is
        // off — its window may straddle the reset, #472 review #1/#6), and production's
        // `INFUSION_EPS` window tolerance all come via the shared `infusion_spans_segment`
        // predicate (#472 review [7]).
        let active_inf: Vec<(usize, T)> = if !has_any_infusion {
            Vec::new()
        } else {
            subject
                .doses
                .iter()
                .enumerate()
                .filter(|(_, d)| d.is_infusion() && d.cmt >= 1)
                .filter(|(_, d)| {
                    infusion_spans_segment(d.time, d.duration, t_start, t_end, reset_floor)
                })
                .map(|(k, d)| (d.cmt.saturating_sub(1), dose_f_bio[k] * T::from_f64(d.rate)))
                .collect()
        };

        // #530: zero-order windows fully containing this segment add their constant
        // `F·amt·frac/dur` jet — the full-containment filter (`w_start ≤ t_start`,
        // `w_end ≥ t_end`) + the `w_end` break above mean each segment is wholly inside or
        // outside every window, matching the f64 `active_zero_order_inputs`. The post-cutoff
        // segment (right end past `w_end`) is excluded, so the rate turns off there; its
        // boundary derivative is supplied by the saltation injected above.
        let active_zero: Vec<(usize, T)> = if !has_zero_order {
            Vec::new()
        } else {
            zero_windows
                .iter()
                .filter(|&&(_, _, w_start, dur)| {
                    let w_end = w_start + dur.val();
                    w_start >= reset_floor
                        && w_start <= t_start + crate::ode::predictions::INFUSION_EPS
                        && w_end >= t_end - crate::ode::predictions::INFUSION_EPS
                })
                .map(|&(cmt, rate, _, _)| (cmt, rate))
                .collect()
        };

        // (`last_dose_eff`, the TAD anchor, was computed once above the saltation block.)
        let rhs = |us: &[T], ps: &[T], t: f64, du: &mut [T]| {
            eval_rhs_anchored::<T>(
                program,
                us,
                ps,
                t,
                first_dose_time,
                last_dose_eff,
                du,
                &mut vars_cell.borrow_mut(),
                &mut stack_cell.borrow_mut(),
            );
            // Infusion rate forcing (static walk only; the TV-cov subset is bolus-only).
            // `scaled_rate` already carries `F·rate` for this dose's compartment (#486).
            for &(cmt, scaled_rate) in &active_inf {
                if cmt < du.len() {
                    du[cmt] = du[cmt] + scaled_rate;
                }
            }
            // #530: zero-order windows delivered as a per-segment constant (like an infusion),
            // NOT pointwise — `add_prepared_input_rate_forcing` skips `ZeroOrder` (it `continue`s)
            // precisely so this constant-rate delivery (matching the f64 path) owns the cutoff.
            for &(cmt, rate) in &active_zero {
                if cmt < du.len() {
                    du[cmt] = du[cmt] + rate;
                }
            }
            // Built-in absorption input-rate forcing R_in(tad), via the shared
            // generic helper — the same superposition loop production runs on `f64`,
            // now monomorphised on the dual type `T`. Lagtime is excluded from this
            // provider (`&[]` → tad = t − dose.time); reset+absorption now threads the
            // tracked `reset_floor` (#486) so a dose delivered before the most recent
            // EVID 3/4 reset is correctly turned off here too, matching production's
            // `active_infusions` rule for infusions (#430 review #4 / #451 / #486).
            if !prepared_forcings.is_empty() {
                crate::ode::predictions::add_prepared_input_rate_forcing::<T>(
                    ode,
                    prepared_forcings,
                    ps,
                    &subject.doses,
                    &[],
                    dose_f_bio,
                    reset_floor,
                    t,
                    du,
                );
            }
        };

        let sol = solve_ode_g(&rhs, &u, (t_start, t_end), params_dual, &saveat, opts);

        // Capture state at the requested observation times; advance u to t_end.
        // `pt.t` is the solver's reported save time — match observations by
        // tolerance rather than bit pattern (issue #410).
        for pt in &sol {
            record_at(pt.t, &pt.u, &mut states, &mut recorded);
            if (pt.t - t_end).abs() < 1e-12 {
                u.copy_from_slice(&pt.u);
            }
        }
    }

    // Every observation must have been captured at a break time or a solver save
    // point. An unmatched one — e.g. a negative observation time below the timeline
    // floor (`t_last` clamps to 0, so it lies in no segment), or a save point the
    // solver dropped/realigned — would keep its zero-initialised state and feed a
    // silent `f = 0`, `∂f = 0` into the gradient. Decline so the caller falls back
    // to FD for this subject rather than return a wrong `Some`.
    if recorded.iter().any(|&r| !r) {
        return None;
    }

    Some(states)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::model_parser::parse_model_string;
    use crate::pk::compute_predictions_with_tv;

    /// Hardening for issue #410: an observation time and the segment break / solver
    /// save time it coincides with are produced by different arithmetic, so they can
    /// be value-equal yet bit-different. `0.1 + 0.2 ≠ 0.3` (IEEE-754) is the canonical
    /// case the old `f64::to_bits` keying would silently miss — leaving the
    /// observation's state (and its sensitivity) at zero. `obs_time_matches` must
    /// match these while still separating genuinely distinct observation times.
    #[test]
    fn obs_time_matches_tolerates_ulp_differences_not_distinct_times() {
        let a = 0.3_f64;
        let b = 0.1_f64 + 0.2; // 0.30000000000000004
        assert_ne!(a.to_bits(), b.to_bits(), "precondition: bit-different");
        assert!(
            obs_time_matches(a, b),
            "ULP-apart times must match — bit-exact keying would drop this observation"
        );
        // Tolerance scales with magnitude (a late, large dosing time).
        let big = 168.0_f64;
        assert!(obs_time_matches(big, big + big * 1e-12));
        // Genuinely distinct observation times must NOT be conflated.
        assert!(!obs_time_matches(24.0, 24.001));
        assert!(!obs_time_matches(0.0, 0.5));
        assert!(!obs_time_matches(big, big + 0.01));
    }
    use crate::types::DoseEvent;
    use std::collections::HashMap;

    // 2-cpt IV bolus as a user ODE, with a Form C concentration readout
    // (`y = central / V1`). CL/V1 carry IIV; Q/V2 are fixed individual params.
    const TWOCPT_ODE: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    fn bolus_subject(times: &[f64]) -> Subject {
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

    // 2-cpt ODE with an allometric weight covariate on CL and V1 — exercises the
    // covariate path: typical values (and their θ-Jacobian) must fold WT.
    const TWOCPT_ODE_COV: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70)^0.75 * exp(ETA_CL)
  V1 = TVV1 * (WT / 70) * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    fn bolus_subject_wt(times: &[f64], wt: f64) -> Subject {
        let mut s = bolus_subject(times);
        s.covariates.insert("WT".to_string(), wt);
        s
    }

    /// The ODE provider's `f`, `∂f/∂η`, `∂f/∂θ` must match the production
    /// predictor (`compute_predictions_with_tv`) and its finite differences.
    #[test]
    fn ode_provider_2cpt_matches_production() {
        let model = parse_model_string(TWOCPT_ODE).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "2-cpt ODE with Form C readout should be supported"
        );
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = vec![4.0, 12.0, 2.0, 25.0];
        let eta = vec![0.12, -0.08];

        let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
        let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
            compute_predictions_with_tv(&model, &subject, th, e)[j]
        };
        let n_eta = model.n_eta;
        let n_theta = model.n_theta;
        let he = 1e-6;

        for (j, obs) in sens.obs.iter().enumerate() {
            // Value matches the production prediction.
            approx::assert_relative_eq!(
                obs.f,
                pred(&eta, &theta, j),
                max_relative = 1e-6,
                epsilon = 1e-9
            );
            // ∂f/∂η vs central FD.
            for k in 0..n_eta {
                let mut ep = eta.clone();
                ep[k] += he;
                let mut em = eta.clone();
                em[k] -= he;
                let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 1e-3, epsilon = 1e-6);
            }
            // ∂f/∂θ vs central FD.
            for m in 0..n_theta {
                let s = he * (1.0 + theta[m].abs());
                let mut tp = theta.clone();
                tp[m] += s;
                let mut tm = theta.clone();
                tm[m] -= s;
                let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * s);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 1e-3,
                    epsilon = 1e-6
                );
            }
        }
    }

    // 2-cpt ODE whose Form C readout references a **covariate** (`FREE`) — a
    // free/total protein-binding readout (#540, the fluconazole_radboudumc shape).
    // The saturable bound term is gated off for free assays (FREE==1) and on for
    // total assays (FREE==0). `BMAX`/`KD` are individual parameters; the covariate
    // threads into the dual readout as a constant from the per-observation snapshot,
    // so the analytic gradient must still match production + FD.
    const TWOCPT_ODE_READOUT_COV: &str = r#"
[parameters]
  theta TVCL(4.0,   0.1, 100.0)
  theta TVV1(12.0,  1.0, 500.0)
  theta TVQ(2.0,    0.01, 100.0)
  theta TVV2(25.0,  1.0, 500.0)
  theta TVBMAX(3.0, 0.0, 100.0)
  theta TVKD(5.0,   0.01, 100.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V1   = TVV1 * exp(ETA_V1)
  Q    = TVQ
  V2   = TVV2
  BMAX = TVBMAX
  KD   = TVKD
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1 + (1.0 - FREE) * BMAX * (central / V1) / (KD + central / V1)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    // 2-cpt ODE whose Form C readout references a θ and an η **directly** (#486):
    // a multiplicative `ETA_CL` on the concentration plus an additive baseline
    // `TVBASE`. The parser desugars each bare θ/η into a synthetic individual
    // parameter (`__ferx_ro_*`), so the readout's ∂y/∂θ, ∂y/∂η ride the
    // individual-parameter sensitivity chain. `ETA_CL` is multiplicative AND also
    // drives `CL`, so the analytic gradient must capture both the explicit term and
    // the cross-coupling through the state — the 2nd-order blocks are non-trivial.
    const TWOCPT_ODE_READOUT_DIRECT_THETA_ETA: &str = r#"
[parameters]
  theta TVCL(4.0,   0.1, 100.0)
  theta TVV1(12.0,  1.0, 500.0)
  theta TVQ(2.0,    0.01, 100.0)
  theta TVV2(25.0,  1.0, 500.0)
  theta TVBASE(0.5, 0.0, 100.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1 * (1.0 + ETA_CL) + TVBASE
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// #486: a Form C readout referencing a θ/η directly is now analytic — the parser
    /// desugars each into a synthetic individual parameter, so the provider gradient
    /// (value, ∂f/∂η, ∂f/∂θ) and the 2nd-order blocks must match production + FD.
    #[test]
    fn ode_provider_form_c_direct_theta_eta_matches_production() {
        let model = parse_model_string(TWOCPT_ODE_READOUT_DIRECT_THETA_ETA).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "Form C readout referencing θ/η directly should be analytic (#486)"
        );
        // The two synthetic readout parameters extend the individual-parameter set.
        assert_eq!(
            model.pk_indices.len(),
            6,
            "4 real + 2 synthetic (__ferx_ro_th4, __ferx_ro_eta0) individual parameters"
        );
        let theta = vec![4.0, 12.0, 2.0, 25.0, 0.5];
        let eta = vec![0.12, -0.08];
        let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];

        let subj = bolus_subject(&times);
        check_vs_production(&model, &subj, &theta, &eta);
        check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
    }

    // Direct θ/η readout (#486) combined with a time-varying covariate (`FREE`), so
    // the subject routes through the (θ,η)-basis TV-cov walk (`run_subject_tvcov`)
    // rather than the individual-parameter-basis static walk. The synthetic readout
    // parameters must serve the analytic gradient on that path too.
    const TWOCPT_ODE_READOUT_DIRECT_TVCOV: &str = r#"
[parameters]
  theta TVCL(4.0,   0.1, 100.0)
  theta TVV1(12.0,  1.0, 500.0)
  theta TVQ(2.0,    0.01, 100.0)
  theta TVV2(25.0,  1.0, 500.0)
  theta TVBASE(0.5, 0.0, 100.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1 * (1.0 + ETA_CL) + (1.0 - FREE) * TVBASE
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// #486 + #540: a direct θ/η readout whose subject also carries a time-varying
    /// covariate routes through the TV-cov `(θ,η)`-basis walk. The desugared synthetic
    /// parameters must match production + FD there as well.
    #[test]
    fn ode_provider_form_c_direct_theta_eta_tvcov_matches_production() {
        let model = parse_model_string(TWOCPT_ODE_READOUT_DIRECT_TVCOV).expect("parse");
        let theta = vec![4.0, 12.0, 2.0, 25.0, 0.5];
        let eta = vec![0.12, -0.08];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0];

        let mut subj = bolus_subject(&times);
        subj.obs_covariates = (0..times.len())
            .map(|i| HashMap::from([("FREE".to_string(), (i % 2) as f64)]))
            .collect();
        assert!(
            subj.has_tv_covariates(),
            "alternating FREE must register as time-varying"
        );
        assert!(
            ode_tvcov_supported(&model, &subj),
            "TV-cov direct-θ/η Form C readout should be analytic (#486)"
        );
        check_vs_production(&model, &subj, &theta, &eta);
        check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
    }

    /// #540: a Form C readout referencing a covariate is now analytic. With the
    /// covariate held constant per subject (`obs_covariates` empty → the static
    /// walk), the provider gradient must match production + FD for both the total
    /// assay (bound term active) and the free assay (bound term zeroed).
    #[test]
    fn ode_provider_form_c_static_covariate_matches_production() {
        let model = parse_model_string(TWOCPT_ODE_READOUT_COV).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "Form C readout referencing a covariate should be analytic (#540)"
        );
        let theta = vec![4.0, 12.0, 2.0, 25.0, 3.0, 5.0];
        let eta = vec![0.12, -0.08];
        let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];

        let mut total = bolus_subject(&times);
        total.covariates.insert("FREE".to_string(), 0.0);
        check_vs_production(&model, &total, &theta, &eta);

        let mut free = bolus_subject(&times);
        free.covariates.insert("FREE".to_string(), 1.0);
        check_vs_production(&model, &free, &theta, &eta);
    }

    /// #540: the readout covariate read per observation. `FREE` alternates row to
    /// row (paired free/total assays on one subject), so the bound term switches on
    /// and off per observation — the TV-cov walk's `obs_cov(j)` snapshot path. The
    /// analytic gradient must still match the production predictor + FD.
    #[test]
    fn ode_provider_form_c_per_obs_covariate_matches_production() {
        let model = parse_model_string(TWOCPT_ODE_READOUT_COV).expect("parse");
        let theta = vec![4.0, 12.0, 2.0, 25.0, 3.0, 5.0];
        let eta = vec![0.12, -0.08];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0];

        let mut subj = bolus_subject(&times);
        subj.obs_covariates = (0..times.len())
            .map(|i| HashMap::from([("FREE".to_string(), (i % 2) as f64)]))
            .collect();
        assert!(
            subj.has_tv_covariates(),
            "alternating FREE must register as time-varying"
        );
        assert!(
            ode_tvcov_supported(&model, &subj),
            "TV-cov Form C readout referencing a covariate should be analytic (#540)"
        );
        check_vs_production(&model, &subj, &theta, &eta);
    }

    /// Shared check: provider `f`/`∂f/∂η`/`∂f/∂θ` vs production predictor + FD.
    fn check_vs_production(model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64]) {
        let sens = ode_subject_sensitivities(model, subject, theta, eta).expect("supported");
        let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
            compute_predictions_with_tv(model, subject, th, e)[j]
        };
        let he = 1e-6;
        for (j, obs) in sens.obs.iter().enumerate() {
            approx::assert_relative_eq!(
                obs.f,
                pred(eta, theta, j),
                max_relative = 1e-6,
                epsilon = 1e-9
            );
            for k in 0..model.n_eta {
                let mut ep = eta.to_vec();
                ep[k] += he;
                let mut em = eta.to_vec();
                em[k] -= he;
                let g = (pred(&ep, theta, j) - pred(&em, theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 2e-3, epsilon = 1e-6);
            }
            for m in 0..model.n_theta {
                let s = he * (1.0 + theta[m].abs());
                let mut tp = theta.to_vec();
                tp[m] += s;
                let mut tm = theta.to_vec();
                tm[m] -= s;
                let g = (pred(eta, &tp, j) - pred(eta, &tm, j)) / (2.0 * s);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
        }
    }

    /// Inner/outer parity guardrail: the light `Dual1` inner provider
    /// (`ode_subject_eta_grad`) must reproduce the full `Dual2` outer provider's value
    /// and `∂f/∂η` (`ode_subject_sensitivities`) exactly — both are exact analytic, only
    /// the dual order differs (and `solve_ode_g` uses value-based step control, so the
    /// trajectories match). This is what keeps the inner EBE loop's η-gradient consistent
    /// with the outer gradient across every ODE variant (Form-C, per-CMT, TV-cov, EVID=2
    /// pk-only, `ExpressionScale`); factored out so a tolerance change touches one site,
    /// not every parity test.
    fn check_inner_outer_eta_parity(
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        eta: &[f64],
    ) {
        let full = ode_subject_sensitivities(model, subject, theta, eta).expect("full outer sens");
        let light = ode_subject_eta_grad(model, subject, theta, eta).expect("light inner grad");
        assert_eq!(full.obs.len(), light.len());
        for (a, b) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-12, epsilon = 1e-12);
            for k in 0..model.n_eta {
                approx::assert_relative_eq!(
                    a.df_deta[k],
                    b.df_deta[k],
                    max_relative = 1e-9,
                    epsilon = 1e-10
                );
            }
        }
    }

    /// Validate the analytic 2nd-order blocks (`d2f_deta2`, `d2f_deta_dtheta`) against
    /// central finite differences of the analytic *first*-order gradient `df_deta` from the
    /// same provider — the Hessian must equal the derivative of the gradient.
    /// `check_vs_production` only FD-checks the value / first order against the values-only
    /// production predictor, so the δlag² saltation coefficients (the rate-boundary `coef2`
    /// and the bolus `jg_cross` cross term) get no independent 2nd-order check otherwise
    /// (#472 review round 2 #3/#4).
    /// Validate the analytic 2nd-order blocks (`d2f_deta2`, `d2f_deta_dtheta`) against
    /// **double central finite differences of the production predictor** — the ground-truth
    /// Hessian, not the self-FD of the analytic gradient. This is the check the #486
    /// init-composition scope needs: a small Hessian element sitting between large neighbours
    /// (e.g. `∂²f/∂η_V²` crossing zero mid-window for an `init = BASE/V` decay) makes the
    /// he=1e-6 self-FD of `check_hessian_vs_fd_of_grad` catastrophically cancel, even though
    /// the analytic Hessian is exact — double-FD of the predictor with an absolute floor is
    /// robust there. The `epsilon` floor (`5e-3`) only relaxes elements whose magnitude is
    /// below it (where a 4-point stencil is dominated by its own truncation/solver noise); a
    /// genuinely wrong large element is still caught by `max_relative`.
    fn check_hessian_vs_production_fd(
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        eta: &[f64],
    ) {
        let n_eta = model.n_eta;
        let n_theta = model.n_theta;
        let base = ode_subject_sensitivities(model, subject, theta, eta).expect("supported");
        let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
            compute_predictions_with_tv(model, subject, th, e)[j]
        };
        let h = 1e-4;
        // ∂²f/∂η_k∂η_p via a 4-point cross stencil.
        for p in 0..n_eta {
            for (j, o) in base.obs.iter().enumerate() {
                for k in 0..n_eta {
                    let mut epp = eta.to_vec();
                    epp[k] += h;
                    epp[p] += h;
                    let mut epm = eta.to_vec();
                    epm[k] += h;
                    epm[p] -= h;
                    let mut emp = eta.to_vec();
                    emp[k] -= h;
                    emp[p] += h;
                    let mut emm = eta.to_vec();
                    emm[k] -= h;
                    emm[p] -= h;
                    let fd = (pred(&epp, theta, j) - pred(&epm, theta, j) - pred(&emp, theta, j)
                        + pred(&emm, theta, j))
                        / (4.0 * h * h);
                    approx::assert_relative_eq!(
                        o.d2f_deta2[k * n_eta + p],
                        fd,
                        max_relative = 5e-3,
                        epsilon = 5e-3
                    );
                }
            }
        }
        // ∂²f/∂η_k∂θ_m via a 4-point cross stencil.
        for m in 0..n_theta {
            let sm = h * (1.0 + theta[m].abs());
            for (j, o) in base.obs.iter().enumerate() {
                for k in 0..n_eta {
                    let mut epp = eta.to_vec();
                    epp[k] += h;
                    let mut tp = theta.to_vec();
                    tp[m] += sm;
                    let mut epm = eta.to_vec();
                    epm[k] += h;
                    let mut tm = theta.to_vec();
                    tm[m] -= sm;
                    let fd = (pred(&epp, &tp, j)
                        - pred(&epp, &tm, j)
                        - pred(&emm_eta(eta, k, h), &tp, j)
                        + pred(&emm_eta(eta, k, h), &tm, j))
                        / (4.0 * h * sm);
                    approx::assert_relative_eq!(
                        o.d2f_deta_dtheta[k * n_theta + m],
                        fd,
                        max_relative = 5e-3,
                        epsilon = 5e-3
                    );
                }
            }
        }
    }

    /// `eta` with `eta[k] -= h` (helper for the mixed η×θ stencil above).
    fn emm_eta(eta: &[f64], k: usize, h: f64) -> Vec<f64> {
        let mut e = eta.to_vec();
        e[k] -= h;
        e
    }

    fn check_hessian_vs_fd_of_grad(
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        eta: &[f64],
    ) {
        let n_eta = model.n_eta;
        let n_theta = model.n_theta;
        let base = ode_subject_sensitivities(model, subject, theta, eta).expect("supported");
        let he = 1e-6;
        // ∂(∂f/∂η_k)/∂η_p == d2f_deta2[k, p].
        for p in 0..n_eta {
            let mut ep = eta.to_vec();
            ep[p] += he;
            let mut em = eta.to_vec();
            em[p] -= he;
            let sp = ode_subject_sensitivities(model, subject, theta, &ep).expect("supported");
            let sm = ode_subject_sensitivities(model, subject, theta, &em).expect("supported");
            for (j, o) in base.obs.iter().enumerate() {
                for k in 0..n_eta {
                    let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * he);
                    approx::assert_relative_eq!(
                        o.d2f_deta2[k * n_eta + p],
                        fd,
                        max_relative = 2e-3,
                        epsilon = 1e-6
                    );
                }
            }
        }
        // ∂(∂f/∂η_k)/∂θ_m == d2f_deta_dtheta[k, m].
        for m in 0..n_theta {
            let s = he * (1.0 + theta[m].abs());
            let mut tp = theta.to_vec();
            tp[m] += s;
            let mut tm = theta.to_vec();
            tm[m] -= s;
            let sp = ode_subject_sensitivities(model, subject, &tp, eta).expect("supported");
            let sm = ode_subject_sensitivities(model, subject, &tm, eta).expect("supported");
            for (j, o) in base.obs.iter().enumerate() {
                for k in 0..n_eta {
                    let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * s);
                    approx::assert_relative_eq!(
                        o.d2f_deta_dtheta[k * n_theta + m],
                        fd,
                        max_relative = 2e-3,
                        epsilon = 1e-6
                    );
                }
            }
        }
    }

    /// 2nd-order saltation validation (#472 review round 2 #3/#4): the δlag² coefficients
    /// for the **rate-boundary** (infusion) and **bolus `jg_cross`** (multi-dose) cases are
    /// FD-checked against the analytic gradient. All use `ETA_LAG` (lag-on-IIV) so the
    /// lagtime η-jet — hence the saltation η-Hessian rows — is non-zero.
    #[test]
    fn ode_provider_lagtime_infusion_hessian_matches_fd_of_grad() {
        // Infusion + lag-on-IIV → exercises the rate-on/off `coef2 = −s·½·J·(Δr·e_cmt)`.
        let model = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag+inf ODE");
        let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0, 12.0]);
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, false, 0.0)];
        check_hessian_vs_fd_of_grad(&model, &subject, &[1.0, 10.0, 0.5], &[0.1, 0.05]);
    }

    #[test]
    fn ode_provider_multidose_bolus_lagtime_hessian_matches_fd_of_grad() {
        // ≥2 bolus doses + lag-on-IIV → the 2nd dose has a non-zero pre-dose state, so the
        // `jg_cross` (post-dose Jacobian on pre-dose velocity) term fires (#472 review #4).
        let model = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag ODE");
        let mut subject = bolus_subject(&[1.0, 3.0, 7.0, 10.0, 14.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
        ];
        check_hessian_vs_fd_of_grad(&model, &subject, &[1.0, 10.0, 0.5], &[0.1, 0.05]);
    }

    /// Reset + lag-on-IIV: the rate-off saltation is skipped after the reset, and the bolus
    /// saltation's 2nd order across the reset boundary must still match (#472 review #3).
    #[test]
    fn ode_provider_lagtime_reset_hessian_matches_fd_of_grad() {
        let model = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag ODE");
        let mut subject = bolus_subject(&[2.0, 5.0, 9.0, 13.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(8.0, 100.0, 1, 0.0, false, 0.0),
        ];
        subject.reset_times = vec![8.0];
        check_hessian_vs_fd_of_grad(&model, &subject, &[1.0, 10.0, 0.5], &[0.1, 0.05]);
    }

    // 1-cpt **Michaelis–Menten (strongly nonlinear)** elimination with an estimated lagtime
    // carrying IIV. The MM curvature `∂(ẋ)/∂central = −VM·KM/(KM+central)²` is large when
    // `central ~ KM`, so a concurrent high-rate infusion forcing into the same compartment
    // contributes a dominant `J·(rate·e_central)` to the bolus saltation's δlag² term — the
    // term the bare-user-RHS velocity drops (#472 review [1]).
    const MM_LAG_INF_ODE: &str = r#"
[parameters]
  theta TVVM(30.0, 1.0, 300.0)
  theta TVKM(10.0, 0.5, 100.0)
  theta TVV(10.0,  1.0, 200.0)
  theta TVLAG(0.5, 0.01,  5.0)

  omega ETA_VM  ~ 0.09
  omega ETA_LAG ~ 0.04

  sigma PROP_ERR ~ 0.05 (sd)
[individual_parameters]
  VM      = TVVM * exp(ETA_VM)
  KM      = TVKM
  V       = TVV
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -VM * central / (KM + central)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;

    /// **Concurrent bolus + infusion under lagtime, nonlinear RHS** (the coverage gap from
    /// #472 review [2]). A bolus co-timed with a finite-duration infusion into the same
    /// MM-eliminated compartment, both shifted by an estimated lagtime with IIV. Exercises the
    /// bolus saltation's full δlag² Hessian (`d²f/dη_LAG²`, `d²f/dη_LAG dθ`) in the presence of
    /// a concurrent infusion forcing on a strongly nonlinear RHS — validated vs FD of the
    /// analytic gradient. (Review finding [1] proposed adding the infusion forcing to the
    /// saltation velocity `J·ẋ`; this test refutes it — the forcing is continuous across the
    /// bolus event and does not shift with the bolus lag, so adding it to the event-time
    /// saltation makes `d²f/dη_LAG²` diverge from FD here. The bare user-RHS velocity is
    /// correct.)
    #[test]
    fn ode_provider_bolus_concurrent_infusion_lagtime_hessian_matches_fd() {
        let model = parse_model_string(MM_LAG_INF_ODE).expect("parse MM lag+inf ODE");
        assert!(model.has_lagtime());
        let mut subject = bolus_subject(&[0.75, 1.0, 1.5, 2.0, 3.0, 4.5]);
        // Co-timed bolus + high-rate finite-duration infusion into the MM compartment, both
        // shifted by LAGTIME. Infusion rate 60, amt 240 → 4 h window; obs sit in the steep
        // MM region just after the lagged arrival.
        subject.doses = vec![
            DoseEvent::new(0.0, 30.0, 1, 0.0, false, 0.0),
            DoseEvent::new(0.0, 240.0, 1, 60.0, false, 0.0),
        ];
        assert!(subject.doses[1].is_infusion() && model.has_lagtime());
        assert!(ode_tvcov_supported(&model, &subject));
        // η = [ETA_VM, ETA_LAG]; θ = [TVVM, TVKM, TVV, TVLAG].
        // First validate the 1st-order gradient vs FD of the predictions (independent of the
        // analytic Hessian internals): if df is correct, FD-of-df is the true d²f.
        check_vs_production(&model, &subject, &[30.0, 10.0, 10.0, 0.5], &[0.1, 0.05]);
        check_hessian_vs_fd_of_grad(&model, &subject, &[30.0, 10.0, 10.0, 0.5], &[0.1, 0.05]);
    }

    // 1-cpt oral ODE with an **estimated lagtime** on the depot dose. The dose arrives
    // at `t + LAGTIME`; the lagtime sensitivity (`∂f/∂TVLAG`, and `∂f/∂η` if lag carries
    // IIV) comes from the event-time saltation injected at the dose. Tier 2 (#439).
    const ONECPT_ORAL_LAG_ODE: &str = r#"
[parameters]
  theta TVCL(1.0,  0.01, 100.0)
  theta TVV(10.0,  1.0, 500.0)
  theta TVKA(1.0,  0.01, 50.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA
  LAGTIME = TVLAG
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
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    /// **Per-compartment lagtime `ALAG1`.** Lagtime declared as a compartment-indexed
    /// `ALAG1` (not the bare `LAGTIME` slot) — each dose reads its lag from its own
    /// compartment's slot (`indexed_slot(Lag, cmt)`), so per-dose differing lags are
    /// exact. Validates `f`/`∂f/∂η`/`∂f/∂θ` (incl. the `θ_ALAG` column) against the
    /// production predictor + FD (#439 / #369).
    const ONECPT_ORAL_ALAG1_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVKA(1.2, 0.01, 50.0)
  theta TVALAG(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  ALAG1 = TVALAG
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
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    #[test]
    fn ode_provider_alag1_matches_production() {
        let model = parse_model_string(ONECPT_ORAL_ALAG1_ODE).expect("parse ALAG1 ODE");
        assert!(model.has_lagtime());
        assert!(
            model
                .active_dose_attr_map()
                .has_indexed_attr(crate::types::DoseAttr::Lag),
            "ALAG1 must be a compartment-indexed lag"
        );
        assert!(ode_analytical_supported(&model));
        let subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0]);
        // θ = [TVCL, TVV, TVKA, TVALAG]; the TVALAG column is the per-compartment lag.
        check_vs_production(&model, &subject, &[1.0, 10.0, 1.2, 0.4], &[0.1]);
    }

    #[test]
    fn ode_provider_lagtime_matches_production() {
        let model = parse_model_string(ONECPT_ORAL_LAG_ODE).expect("parse oral lag ODE");
        assert!(model.has_lagtime(), "model must declare a lagtime");
        assert!(
            ode_analytical_supported(&model),
            "bare-lagtime oral ODE must be analytic-supported"
        );
        // Single bolus into the depot at t=0; observations span the lagged onset.
        let subject = bolus_subject(&[0.25, 0.75, 1.5, 3.0, 6.0, 10.0]);
        // θ = [TVCL, TVV, TVKA, TVLAG]; the TVLAG column is driven entirely by the
        // event-time saltation, so it is the key check.
        check_vs_production(&model, &subject, &[1.0, 10.0, 1.0, 0.5], &[0.12, -0.08]);
    }

    // 2-cpt IV ODE under LTBS (`log(DV) ~ additive`): the readout is log-transformed
    // (`p = ln f`), so the provider's f/∂f/∂η/∂f/∂θ must match the (also
    // log-transformed) production predictor. Tier 1 output transform (#410).
    const TWOCPT_ODE_LTBS: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma ADD_LOG ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  log(DV) ~ additive(ADD_LOG)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    #[test]
    fn ode_provider_ltbs_matches_production() {
        let model = parse_model_string(TWOCPT_ODE_LTBS).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "LTBS ODE should be supported (Tier 1)"
        );
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
    }

    // 2-cpt IV ODE with a constant `ScalarScale` output divisor (`obs_scale = 50`)
    // over the central-amount readout: `f = central / 50`. Tier 1 output transform.
    const TWOCPT_ODE_SCALARSCALE: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(obs_cmt=central, states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  obs_scale = 50
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    #[test]
    fn ode_provider_scalar_scale_matches_production() {
        let model = parse_model_string(TWOCPT_ODE_SCALARSCALE).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "constant ScalarScale ODE should be supported (Tier 1)"
        );
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
    }

    // 2-cpt IV ODE with an η-dependent `ExpressionScale` divisor `obs_scale = 1000 / V1`
    // (`V1` carries `ETA_V1`) over the central-amount (`ObsCmt`) readout. The scale is
    // applied as the subject-static quotient on the final `(θ,η)`-space jet (#486),
    // reusing the closed-form provider's `apply_expression_scale_outer`.
    const TWOCPT_ODE_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(obs_cmt=central, states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  obs_scale = 1000 / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// η-dependent `ExpressionScale` `obs_scale = 1000 / V1` on the ODE path (#486): the
    /// outer provider's scaled `f` / `∂f/∂η` / `∂f/∂θ` must match FD of the production
    /// predictor (which divides by the same scale through `apply_scaling`), and the
    /// 2nd-order blocks must match FD of the analytic gradient — exercising the
    /// `apply_expression_scale_outer` quotient rule layered onto the ODE jet.
    #[test]
    fn ode_provider_expression_scale_matches_production() {
        let model = parse_model_string(TWOCPT_ODE_EXPRSCALE).expect("parse");
        assert!(
            matches!(
                model.scaling,
                ScalingSpec::ExpressionScale { deriv: Some(_), .. }
            ),
            "model must carry a differentiable scale program"
        );
        assert!(
            ode_analytical_supported(&model),
            "η-dependent ExpressionScale ODE must be supported (#486)"
        );
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
        check_hessian_vs_fd_of_grad(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
    }

    /// `ExpressionScale` + **LTBS** must route to FD (`None`): the walk applies the LTBS
    /// log in PK-param space *before* the η/θ chain, so the production scale-then-log order
    /// can't be reproduced by a post-walk quotient — `expression_scale_axes_admissible`
    /// gates it on `!log_transform`, so the analytic scope declines it (#486).
    #[test]
    fn ode_provider_expression_scale_combos_fall_back_to_fd() {
        // + LTBS → out of analytic scope (the walk applies LTBS pre-chain, so the
        // production scale-then-log order can't be reproduced by a post-walk quotient).
        // This is the only `ExpressionScale` combination that still declines: since #486
        // the event-driven walk applies the subject-static scale quotient, so
        // `ExpressionScale` + time-varying covariate / `TIME` is analytic (validated in
        // `ode_expression_scale_on_event_walk_matches_production`).
        let mut ltbs = parse_model_string(TWOCPT_ODE_EXPRSCALE).expect("parse");
        ltbs.log_transform = true;
        assert!(
            !ode_analytical_supported(&ltbs),
            "ExpressionScale + LTBS must fall back to FD"
        );
    }

    /// A 2-cpt IV ODE with **time-varying covariates** (`WT` on `V1`) **and** an η-dependent
    /// `ExpressionScale` divisor `obs_scale = 1000 / V1`. The scale is **subject-static even
    /// under TV covariates** (production `apply_scaling` reads `subject.covariates`), so the
    /// non-IOV TV-cov walk applies it as a single subject-static post-walk quotient (#486 —
    /// the IOV per-occasion-group machinery of #590 collapses to one jet, no κ). The walk
    /// itself runs the per-event TV-cov PK params; only the divisor is subject-static.
    const TWOCPT_ODE_EXPRSCALE_TVCOV: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^0.75 * exp(ETA_CL)
  V1 = TVV1 * (WT/70) * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(obs_cmt=central, states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  obs_scale = 1000 / V1
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// Build a TV-cov subject (WT on dose + per-obs) for the 2-cpt ExpressionScale model,
    /// with `subject.covariates` carrying the subject-static WT the scale divisor reads.
    fn exprscale_tvcov_subject() -> Subject {
        let mut subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        subject.covariates = wt(70.0);
        subject.dose_covariates = vec![wt(62.0)];
        subject.obs_covariates = vec![wt(62.0), wt(66.0), wt(70.0), wt(74.0), wt(80.0), wt(88.0)];
        subject
    }

    /// TV-cov + `ExpressionScale` on the **non-IOV** ODE walk (#486): the outer provider's
    /// scaled `f` / `∂f/∂η` / `∂f/∂θ` must match FD of the production TV-cov predictor (which
    /// divides by the subject-static `obs_scale` via `apply_scaling`), and the 2nd-order
    /// blocks must match FD of the analytic gradient — exercising the subject-static
    /// `apply_expression_scale_outer` quotient layered onto the per-event TV-cov jet.
    #[test]
    fn ode_provider_tvcov_expression_scale_matches_production() {
        let model = parse_model_string(TWOCPT_ODE_EXPRSCALE_TVCOV).expect("parse");
        assert!(
            matches!(
                model.scaling,
                ScalingSpec::ExpressionScale { deriv: Some(_), .. }
            ),
            "model must carry a differentiable scale program"
        );
        let subject = exprscale_tvcov_subject();
        assert!(subject.has_tv_covariates());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "TV-cov + ExpressionScale must be analytic on the non-IOV ODE walk (#486)"
        );
        let theta = vec![4.0, 12.0, 2.0, 25.0];
        let eta = vec![0.12, -0.08];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
    }

    /// Inner/outer parity for TV-cov + `ExpressionScale` (#486): the light `Dual1` inner
    /// η-gradient (`run_subject_tvcov_eta` + `apply_expression_scale_inner_dispatch`) must
    /// equal the full `Dual2` outer `df_deta` (`run_subject_tvcov` + the outer quotient).
    /// This is the guardrail: both walks share `ode_tvcov_supported`, so the inner quotient
    /// must move in lockstep with the outer or the EBE gradient silently diverges.
    #[test]
    fn ode_provider_tvcov_expression_scale_inner_eta_grad_matches_outer() {
        let model = parse_model_string(TWOCPT_ODE_EXPRSCALE_TVCOV).expect("parse");
        let subject = exprscale_tvcov_subject();
        let theta = vec![4.0, 12.0, 2.0, 25.0];
        let eta = vec![0.12, -0.08];
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// Build the TV-cov subject for the 1-cpt `ONECPT_ODE_TVCOV` model with an EVID=2
    /// covariate-only breakpoint (`pk_only`) at t=3 carrying its own WT snapshot.
    fn tvcov_pkonly_subject() -> Subject {
        let mut subject = tvcov_subject();
        subject.pk_only_times = vec![3.0];
        subject.pk_only_covariates = vec![HashMap::from([("WT".to_string(), 75.0)])];
        subject
    }

    /// TV-cov + EVID=2 pk-only covariate breakpoint on the **non-IOV** ODE walk (#486): the
    /// analytic outer `SubjectSens` (value, `∂f/∂η`, `∂f/∂θ`, + 2nd order) must match FD of
    /// the production TV-cov predictor, which carries the same `K_PKONLY` breakpoint. With no
    /// κ there is no `iov_combined_pk_only` analogue to build — pk-only events seed at their
    /// own η-snapshot exactly like obs/dose (the IOV path got this in #590).
    #[test]
    fn ode_provider_tvcov_pkonly_matches_production() {
        let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
        let subject = tvcov_pkonly_subject();
        assert!(
            !subject.pk_only_times.is_empty(),
            "fixture must carry an EVID=2 covariate breakpoint"
        );
        assert!(subject.has_tv_covariates());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "TV-cov + EVID=2 pk-only must be analytic on the non-IOV ODE walk (#486)"
        );
        let theta = vec![1.0, 20.0, 0.75];
        let eta = vec![0.1];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
    }

    /// Inner/outer parity for TV-cov + EVID=2 pk-only (#486): the light inner η-gradient must
    /// equal the full outer `df_deta`. Both walks share `ode_tvcov_supported`, so opening the
    /// pk-only gate enables both — this confirms the inner walk consumes the breakpoint too.
    #[test]
    fn ode_provider_tvcov_pkonly_inner_eta_grad_matches_outer() {
        let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
        let subject = tvcov_pkonly_subject();
        let theta = vec![1.0, 20.0, 0.75];
        let eta = vec![0.1];
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// The light `Dual1` inner provider's `f` / `∂f/∂η` must equal the full `Dual2`
    /// outer provider's `f` / `df_deta` exactly — both are exact analytic, only the
    /// dual order differs (and `solve_ode_g` uses value-based step control, so the
    /// trajectories match). This is what makes the inner EBE loop's analytic
    /// η-gradient correct (#410). Run across the readout/dose variants so the light
    /// driver's branches are exercised: Form-C readout (`TWOCPT_ODE`), an `ObsCmt`
    /// model with non-zero `init(...)` (exercises `dual1_init_state`), estimated
    /// bioavailability `F` (`BIOAV_ODE` — the `f_bio` path + `ObsCmt` arm), and LTBS.
    #[test]
    fn ode_light_inner_eta_grad_matches_full_provider() {
        // Form-C readout, IV bolus.
        let m = parse_model_string(TWOCPT_ODE).expect("parse");
        check_inner_outer_eta_parity(
            &m,
            &bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
            &[4.0, 12.0, 2.0, 25.0],
            &[0.12, -0.08],
        );

        // ObsCmt readout + non-zero init(...) → exercises `dual1_init_state`.
        let m = parse_model_string(INIT_ODE).expect("parse");
        let mut s = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        s.doses = vec![];
        check_inner_outer_eta_parity(&m, &s, &[1.0, 20.0], &[0.1, -0.05]);

        // Estimated bioavailability F + ObsCmt readout, oral depot → `f_bio` path.
        let m = parse_model_string(BIOAV_ODE).expect("parse");
        let mut s = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        check_inner_outer_eta_parity(&m, &s, &[5.0, 50.0, 1.5, 0.70], &[0.15, 0.2]);

        // Compartment-indexed F1 (with IIV) → per-compartment `f_bio_slot` path (#486).
        let m = parse_model_string(F1_ODE).expect("parse");
        let mut s = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        check_inner_outer_eta_parity(&m, &s, &[5.0, 50.0, 1.5, 0.70], &[0.15, 0.2]);

        // LTBS output transform over the Dual1 readout.
        let m = parse_model_string(TWOCPT_ODE_LTBS).expect("parse");
        check_inner_outer_eta_parity(
            &m,
            &bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
            &[4.0, 12.0, 2.0, 25.0],
            &[0.12, -0.08],
        );

        // η-dependent `ExpressionScale` divisor (#486) → the inner η-only quotient
        // (`apply_expression_scale_inner_dispatch`) must equal the full provider's
        // scaled `df_deta`.
        let m = parse_model_string(TWOCPT_ODE_EXPRSCALE).expect("parse");
        check_inner_outer_eta_parity(
            &m,
            &bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
            &[4.0, 12.0, 2.0, 25.0],
            &[0.12, -0.08],
        );

        // Estimated lagtime with IIV (`ETA_LAG`), multi-dose — exercises the event-time
        // saltation (incl. the 2nd-dose `jg_cross`) in BOTH the Dual1 inner and the Dual2
        // outer walk, which must agree to provider tolerance (#472 review round 2 #8).
        let m = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag ODE");
        let mut s = bolus_subject(&[1.0, 3.0, 7.0, 10.0]);
        s.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
        ];
        check_inner_outer_eta_parity(&m, &s, &[1.0, 10.0, 0.5], &[0.1, 0.05]);
    }

    /// The inner EBE loop must actually *resolve* to the analytic η-gradient for an
    /// in-scope ODE subject (not merely be correct when called) — i.e. the wiring in
    /// `analytic_inner_grad_supported` / `resolve_gradient_method` engages (#410).
    #[test]
    fn ode_inner_gradient_route_resolves_analytic() {
        use crate::estimation::inner_optimizer::{resolve_gradient_method, InnerGradientMethod};
        let model = parse_model_string(TWOCPT_ODE).expect("parse");
        let subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0]);
        assert_eq!(
            resolve_gradient_method(&model, &subject),
            InnerGradientMethod::Analytic,
            "in-scope ODE subject must use the analytic inner η-gradient (#410)"
        );
        // The provider entry the inner loop actually calls (`subject_eta_grad`) must
        // route the ODE model to the light Dual1 provider, not decline.
        let g = crate::sens::provider::subject_eta_grad(
            &model,
            &subject,
            &[4.0, 12.0, 2.0, 25.0],
            &[0.1, -0.05],
        );
        assert!(
            g.is_some_and(|v| v.len() == subject.obs_times.len()),
            "subject_eta_grad must serve an in-scope ODE subject via the light provider"
        );
    }

    // 1-cpt oral ODE with estimated, logit-normal bioavailability F — the dose
    // loads `F·AMT` into the depot, so F's derivative must flow through the
    // injection. Mirrors examples/bioavailability_ode.ferx.
    const BIOAV_ODE: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1,  50.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVKA(1.5,  0.05,  20.0)
  theta THETA_F(0.70, 0.001, 0.999)
  omega ETA_CL ~ 0.09
  omega ETA_F  ~ 0.10
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = inv_logit(logit(THETA_F) + ETA_F)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V - CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// Estimated bioavailability F: the provider must propagate F's derivative
    /// through the `F·AMT` depot loading (and the logit/inv_logit individual-F
    /// map), matching the production predictor and its FD.
    #[test]
    fn ode_provider_oral_bioavailability_matches_production() {
        let model = parse_model_string(BIOAV_ODE).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "estimated F should be in scope"
        );
        let mut subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        check_vs_production(&model, &subject, &[5.0, 50.0, 1.5, 0.70], &[0.15, 0.2]);
    }

    // 1-cpt oral ODE with a *compartment-indexed* bioavailability `F1` (dose into the
    // depot, cmt 1) carrying IIV (logit-normal). `F1` lands in the dose-attr map at its
    // own slot, NOT the bare `PK_IDX_F` (which stays 1.0): so a walk that read the bare
    // slot would apply F = 1.0 and diverge from production's `f_bio(cmt=1)` — both in
    // value and in ∂/∂F. The IIV (`ETA_F`) routes F1 into the inner η-gradient too (#486).
    const F1_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.5, 0.05, 20.0)
  theta THETA_F1(0.70, 0.001, 0.999)
  omega ETA_CL ~ 0.09
  omega ETA_F  ~ 0.10
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F1 = inv_logit(logit(THETA_F1) + ETA_F)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V - CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    // 2-cpt IV ODE with *distinct* per-compartment bioavailabilities `F1` (central,
    // cmt 1) and `F2` (peripheral, cmt 2). With one dose into each compartment, a walk
    // that did not key F by the dose's own compartment (e.g. applied dose 0's F to
    // both) would diverge from production. `obs_cmt = central`, so both F1 (direct) and
    // F2 (via peripheral→central redistribution) move the observed concentration, hence
    // both ∂/∂F1 and ∂/∂F2 are observable (#486).
    const F1F2_IV_ODE: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0, 0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  theta THETA_F1(0.80, 0.001, 0.999)
  theta THETA_F2(0.50, 0.001, 0.999)
  omega ETA_CL ~ 0.10
  sigma PROP_ERR ~ 0.05 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q  = TVQ
  V2 = TVV2
  F1 = THETA_F1
  F2 = THETA_F2
[structural_model]
  ode(obs_cmt=central, states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// `f_bio_slot` resolves a dose's bioavailability slot per compartment: the
    /// indexed `F{cmt}` slot when declared, else the bare `PK_IDX_F` (#486).
    #[test]
    fn f_bio_slot_resolves_indexed_then_bare() {
        let m = parse_model_string(F1F2_IV_ODE).expect("parse F1F2");
        let ode = m.ode_spec.as_ref().expect("ode_spec");
        let bare = crate::types::PK_IDX_F;
        let s1 = f_bio_slot(ode, 1);
        let s2 = f_bio_slot(ode, 2);
        assert_ne!(s1, bare, "F1 must resolve to its own indexed slot");
        assert_ne!(s2, bare, "F2 must resolve to its own indexed slot");
        assert_ne!(s1, s2, "F1 and F2 occupy distinct slots");
        assert_eq!(
            s1,
            ode.dose_attr_map
                .indexed_slot(crate::types::DoseAttr::F, 1)
                .unwrap()
        );
        // A compartment with no indexed `F` falls back to the bare slot.
        assert_eq!(f_bio_slot(ode, 3), bare);
    }

    /// Compartment-indexed bioavailability (`F1`/`F2`, #369) is now served analytically
    /// (#486): the dual walks resolve `F` per dose compartment, so analytic f / ∂f/∂η /
    /// ∂f/∂θ match the production predictor and its FD. Covers the single-indexed depot
    /// case (with IIV on F1) and distinct F1≠F2 into two compartments.
    #[test]
    fn ode_provider_compartment_indexed_f_matches_production() {
        // Single indexed F1 into the depot, with IIV.
        let model = parse_model_string(F1_ODE).expect("parse F1");
        assert!(
            ode_analytical_supported(&model),
            "compartment-indexed F1 should now be in scope"
        );
        let mut subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        check_vs_production(&model, &subject, &[5.0, 50.0, 1.5, 0.70], &[0.15, 0.2]);

        // Distinct F1 (central) and F2 (peripheral) with a dose into each compartment.
        let model = parse_model_string(F1F2_IV_ODE).expect("parse F1F2");
        assert!(
            ode_analytical_supported(&model),
            "distinct per-compartment F1/F2 should be in scope"
        );
        let mut subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(0.0, 50.0, 2, 0.0, false, 0.0),
        ];
        check_vs_production(
            &model,
            &subject,
            &[4.0, 12.0, 2.0, 25.0, 0.80, 0.50],
            &[0.1],
        );
    }

    /// Compartment-indexed lag (`ALAG1`) IS supported (#472): it is handled on the
    /// event-driven saltation walk, which reads each dose's lag from its own slot, so
    /// the model passes the analytic gate and routes to the event-driven walk (not the
    /// static superposition walk). (Indexed `F` is also supported — parity test above.)
    #[test]
    fn ode_analytical_supports_per_compartment_lag() {
        const ALAG1_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.5, 0.05, 20.0)
  theta THETA_ALAG(0.3, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  ALAG1 = THETA_ALAG
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V - CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let m = parse_model_string(ALAG1_ODE).expect("parse ALAG1");
        // Compartment-indexed `ALAG1` IS supported now: lagtime is handled on the
        // event-driven saltation walk, which reads each dose's lag from its own slot
        // (`indexed_slot(Lag, cmt)`), so per-compartment / per-dose lags are exact (#439).
        assert!(
            ode_analytical_supported(&m),
            "compartment-indexed ALAG1 is supported (event-time saltation, per-dose lag slot)"
        );
        // It routes to the event-driven walk, not the static superposition walk.
        let subj = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0]);
        assert!(ode_tvcov_supported(&m, &subj), "ALAG1 → event-driven walk");
        assert!(
            !ode_subject_supported(&m, &subj),
            "ALAG1 not on the static walk"
        );
    }

    /// Indexed `F` is now in model-level scope (parity test above), but the
    /// *per-subject* gate must still route a **rate-defined infusion under `F ≠ 1`**
    /// to FD. NONMEM reshapes such an infusion's window (holds the rate, scales the
    /// duration to `F·amt/rate`, #419), whereas the dual walk scales the rate
    /// magnitude over the *original* window — so the analytic gradient would diverge
    /// from the f64 predictor. The model-level gate admits the indexed-`F` model; the
    /// subject gate (`has_bioavailability() && has_rate_defined_infusion()`) declines
    /// it. Crucially `has_bioavailability()` detects the indexed `F{cmt}` form too, so
    /// dropping the `ode_analytical_supported` indexed-`F` decline (#486) does *not*
    /// open this infusion path. A bolus of the same model stays in scope, so the
    /// decline is attributable to the infusion, not the `F`.
    #[test]
    fn ode_subject_declines_indexed_f_rate_defined_infusion() {
        let model = parse_model_string(F1F2_IV_ODE).expect("parse F1F2");
        // Model-level scope admits indexed F (the indexed-F decline gate is gone, #486).
        assert!(
            ode_analytical_supported(&model),
            "indexed F1/F2 model is in model-level scope"
        );

        // A bolus subject of this model IS served analytically.
        let bolus = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        assert!(
            !bolus.doses[0].is_infusion(),
            "control dose must be a bolus"
        );
        assert!(
            ode_subject_supported(&model, &bolus),
            "indexed-F bolus subject should be served analytically"
        );

        // The same model with a *rate-defined* infusion (RATE>0) into the dosed
        // compartment must decline to FD: `F` reshapes the window, which the dual rate
        // scale does not reproduce (#419).
        let mut infusion = bolus.clone();
        infusion.doses = vec![DoseEvent::new(0.0, 1000.0, 1, 200.0, false, 0.0)];
        assert!(
            infusion.doses[0].is_infusion(),
            "rate=200 must be a rate-defined infusion"
        );
        assert!(
            !ode_subject_supported(&model, &infusion),
            "rate-defined infusion under indexed F must decline to FD (#419)"
        );
    }

    /// Infusion doses (RATE>0): the dual loop must add the rate forcing over the
    /// infusion window and match the production predictor through during- and
    /// post-infusion observations.
    #[test]
    fn ode_provider_2cpt_infusion_matches_production() {
        let model = parse_model_string(TWOCPT_ODE).expect("parse");
        // amt=1000, rate=200 → 5 h infusion into central; obs during and after.
        let mut subject = bolus_subject(&[1.0, 3.0, 5.0, 6.0, 9.0, 24.0]);
        subject.doses = vec![DoseEvent::new(0.0, 1000.0, 1, 200.0, false, 0.0)];
        check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
    }

    // 1-cpt with a non-zero `init(central) = 1000/V` baseline (depends on V), no
    // dose — exercises the dual-seeded initial state and its V derivative.
    const INIT_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = 1000.0 / V
  d/dt(central) = -CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// Non-zero `init(...)`: the dual initial state (value + parameter derivative)
    /// must match the production predictor + FD across the decay from baseline.
    #[test]
    fn ode_provider_init_matches_production() {
        let model = parse_model_string(INIT_ODE).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "init(...) should be in scope"
        );
        let mut subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        subject.doses = vec![]; // baseline comes from init, not a dose
        check_vs_production(&model, &subject, &[1.0, 20.0], &[0.1, -0.05]);
    }

    /// #530: a **modeled-duration infusion** (`RATE=-2` → `D1`) gets analytic
    /// sensitivities on the event-driven walk. The infusion window end `t_dose + D1` is a
    /// moving boundary in `D1`; the rate-off saltation carries its derivative (the
    /// sign-mirror of the lagtime dose-start saltation). `D1` is η-coupled here
    /// (`D1 = TVD1·exp(ETA_D1)`) so both the θ-block (`df_dtheta[TVD1]`) and the η-block
    /// (`df_deta[ETA_D1]`) of the moving-boundary term are FD-checked against the production
    /// predictor, which resolves `D1` per evaluation. Observations straddle the window end
    /// (window `[0, 5]` at `D1=5`): 1, 3 inside; 6, 10 after — the post-window obs are the
    /// ones the moving boundary moves.
    #[test]
    fn ode_provider_modeled_duration_matches_production() {
        const MODELED_DUR_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(MODELED_DUR_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
        subject.doses = vec![DoseEvent::modeled(
            0.0,
            100.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        )];
        // A modeled dose declines the static superposition walk and routes to the
        // event-driven walk (where the moving-boundary saltation lives).
        assert!(
            ode_tvcov_supported(&model, &subject) && !ode_subject_supported(&model, &subject),
            "a modeled-duration dose must route to the event-driven walk"
        );
        let theta = [5.0, 50.0, 5.0];
        let eta = [0.12, -0.08];
        check_vs_production(&model, &subject, &theta, &eta);
        // Inner/outer scope parity: the light `Dual1` inner η-gradient shares the same
        // event-driven walk, so it must serve this subject too (never analytic-outer /
        // FD-inner) and its `∂f/∂η` must equal the outer provider's η-block exactly.
        let inner = ode_subject_eta_grad(&model, &subject, &theta, &eta).expect("inner served");
        let outer =
            ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("outer served");
        for (io, oo) in inner.iter().zip(outer.obs.iter()) {
            for k in 0..model.n_eta {
                approx::assert_relative_eq!(
                    io.df_deta[k],
                    oo.df_deta[k],
                    max_relative = 1e-9,
                    epsilon = 1e-12
                );
            }
        }
    }

    /// #530 mirror: a **modeled-rate infusion** (`RATE=-1` → `R1`) — the window length is
    /// `F·amt/R1`, a moving boundary in `R1`, carried by the same rate-off saltation. `R1`
    /// is η-coupled so both blocks are FD-checked.
    #[test]
    fn ode_provider_modeled_rate_matches_production() {
        const MODELED_RATE_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVR1(20.0, 0.1, 200.0)
  omega ETA_CL ~ 0.09
  omega ETA_R1 ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  R1 = TVR1 * exp(ETA_R1)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(MODELED_RATE_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
        subject.doses = vec![DoseEvent::modeled(
            0.0,
            100.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledRate,
        )];
        assert!(
            ode_tvcov_supported(&model, &subject) && !ode_subject_supported(&model, &subject),
            "a modeled-rate dose must route to the event-driven walk"
        );
        check_vs_production(&model, &subject, &[5.0, 50.0, 20.0], &[0.12, -0.08]);
    }

    /// #530 cross-term: a **modeled-duration dose under an estimated lagtime**. Both the
    /// dose *start* (`t_dose + lag`, rate-on, moves with `lag`) and the window *end*
    /// (`t_dose + lag + D1`, rate-off, moves with `lag + D1`) are moving boundaries. The
    /// event-driven walk carries the start saltation with shift `δlag` and the end saltation
    /// with the combined shift `δlag + δD1` (`integrate_tvcov_g` `d_off = dlag + dtinf`,
    /// `ode_provider.rs:3475`). Both `D1` and `LAGTIME` are η-coupled so the cross-term's
    /// θ- and η-blocks are FD-checked against the production predictor (which resolves the
    /// modeled window and the lag per evaluation). This guards the combination the gate
    /// admits to the analytic walk — it is *not* an FD fallback (#530 review finding 1).
    /// Observations straddle the lagged window end (window `[1, 6]` at `lag=1, D1=5`): 4
    /// inside, 7 and 11 after, 0.5 before the lagged dose arrival.
    #[test]
    fn ode_provider_modeled_duration_with_lagtime_matches_production() {
        const MODELED_DUR_LAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  theta TVLAG(1.0, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_D1 ~ 0.04
  omega ETA_LAG ~ 0.02
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  D1 = TVD1 * exp(ETA_D1)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(MODELED_DUR_LAG_ODE).expect("parse");
        let mut subject = bolus_subject(&[0.5, 4.0, 7.0, 11.0]);
        subject.doses = vec![DoseEvent::modeled(
            0.0,
            100.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        )];
        // Modeled dose + lagtime both route to the event-driven walk; the combination must
        // be admitted there (not declined to FD) — the saltation carries the cross-term.
        assert!(
            model.has_lagtime()
                && ode_tvcov_supported(&model, &subject)
                && !ode_subject_supported(&model, &subject),
            "modeled-duration + lagtime must ride the event-driven walk analytically"
        );
        let theta = [5.0, 50.0, 5.0, 1.0];
        let eta = [0.12, -0.08, 0.05];
        check_vs_production(&model, &subject, &theta, &eta);
        // Inner/outer scope parity on the same walk.
        let inner = ode_subject_eta_grad(&model, &subject, &theta, &eta).expect("inner served");
        let outer =
            ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("outer served");
        for (io, oo) in inner.iter().zip(outer.obs.iter()) {
            for k in 0..model.n_eta {
                approx::assert_relative_eq!(
                    io.df_deta[k],
                    oo.df_deta[k],
                    max_relative = 1e-9,
                    epsilon = 1e-12
                );
            }
        }
    }

    /// #530: **zero-order absorption** (`zero_order(dur)`) gets analytic sensitivities on the
    /// static walk. The forcing delivers `F·amt/dur` over `[t_dose, t_dose+dur]`; the window
    /// end is a moving boundary in `dur`, carried by the rate-off saltation injected at the
    /// cutoff (`integrate_g`), exactly as a modeled-duration infusion. `DUR` is η-coupled
    /// (`DUR = TVDUR·exp(ETA_DUR)`) so both the θ-block (`df_dtheta[TVDUR]`) and the η-block
    /// (`df_deta[ETA_DUR]`) of the moving-boundary term are FD-checked against the production
    /// predictor. A bolus into `central` (the forcing's compartment) feeds `R_in`, not a
    /// bolus. Observations straddle the window end (window `[0, 4]` at `DUR=4`).
    #[test]
    fn ode_provider_zero_order_absorption_matches_production() {
        const ZERO_ORDER_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(ZERO_ORDER_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
        // A plain bolus into the forcing's compartment feeds R_in (not a bolus).
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        assert!(
            ode_subject_supported(&model, &subject),
            "a zero-order model must be served analytically on the static walk (#530)"
        );
        let theta = [5.0, 50.0, 4.0];
        let eta = [0.12, -0.08];
        check_vs_production(&model, &subject, &theta, &eta);
        // Inner/outer scope parity: the `Dual1` inner η-gradient shares `integrate_g`, so it
        // must serve this subject too and its `∂f/∂η` must equal the outer provider's η-block.
        let inner = ode_subject_eta_grad(&model, &subject, &theta, &eta).expect("inner served");
        let outer = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("outer");
        for (io, oo) in inner.iter().zip(outer.obs.iter()) {
            for k in 0..model.n_eta {
                approx::assert_relative_eq!(
                    io.df_deta[k],
                    oo.df_deta[k],
                    max_relative = 1e-9,
                    epsilon = 1e-12
                );
            }
        }
    }

    /// #530 finding 2: **multi-dose `zero_order(dur)`**. The global `ZeroOrder` Dual2 lift
    /// serves more than the single-dose case the first zero-order test covers — every dose
    /// opens its own constant-rate window with its own moving end. Two doses (t=0 and t=12)
    /// each deliver `F·amt/DUR` over their own `[t_dose, t_dose+DUR]` window; the second
    /// window's saltation must fire independently. `DUR` is η-coupled. Observations straddle
    /// both window ends (`DUR=4`: windows `[0,4]` and `[12,16]`; obs 3, 6 around the first,
    /// 14, 18 around the second).
    #[test]
    fn ode_provider_zero_order_multi_dose_matches_production() {
        const ZERO_ORDER_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(ZERO_ORDER_ODE).expect("parse");
        let mut subject = bolus_subject(&[3.0, 6.0, 14.0, 18.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ];
        assert!(
            ode_subject_supported(&model, &subject),
            "multi-dose zero-order must be served analytically on the static walk"
        );
        check_vs_production(&model, &subject, &[5.0, 50.0, 4.0], &[0.12, -0.08]);
    }

    /// #530 finding 2: **mixed (zero-order + first-order) absorption**. `FZO*zero_order(dur=DUR)`
    /// `+ FZO1*first_order(ka=KA)` feeds one compartment; the docs claim the whole `mixed`
    /// family is now analytic (the zero-order `∂/∂DUR` boundary term carried by the saltation,
    /// the first-order pathway smooth). This checks the composite against production FD with
    /// `DUR` η-coupled — the saltation must fire for the zero-order pathway while the
    /// first-order forcing rides the smooth dual path alongside it.
    #[test]
    fn ode_provider_mixed_absorption_matches_production() {
        const MIXED_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVDUR(3.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  FZO  = TVFZO
  FZO1 = 1 - TVFZO
  KA   = TVKA
  DUR  = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(MIXED_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        assert!(
            ode_subject_supported(&model, &subject),
            "mixed zero+first-order absorption must be served analytically (#530)"
        );
        check_vs_production(
            &model,
            &subject,
            &[5.0, 50.0, 0.4, 1.0, 3.0],
            &[0.12, -0.08],
        );
    }

    /// EVID 3/4 reset: a two-occasion subject (reset + re-dose at t=10) must zero
    /// the dual state at the reset and match the production event-driven path
    /// across both occasions.
    #[test]
    fn ode_provider_2cpt_reset_matches_production() {
        let model = parse_model_string(TWOCPT_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 11.0, 13.0, 16.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        subject.reset_times = vec![10.0];
        check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
    }

    /// **Static walk: infusion straddling an EVID 3/4 reset.** A plain subject (no TV-cov,
    /// no lagtime) with an infusion window straddling a reset routes to the *static*
    /// `integrate_g` walk via `ode_subject_supported` — whose `active_inf` must drop the
    /// pre-reset infusion afterward (`reset_floor`), else `F·rate` leaks into the post-reset
    /// segments. The PR fixed the event-driven twin; this guards the static path so a future
    /// edit can't silently reintroduce the pre-PR leak (#472 review round 2 #1). The
    /// post-reset observations (3, 4, 8) are the ones that catch it.
    #[test]
    fn ode_provider_static_infusion_reset_matches_production() {
        let model = parse_model_string(TWOCPT_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 4.0, 8.0]);
        subject.doses = vec![
            // Infusion rate 200, amt 1000 → 5 h window [0, 5], straddling the reset at t=2.
            DoseEvent::new(0.0, 1000.0, 1, 200.0, false, 0.0),
            // EVID=4 re-dose (bolus) at the reset.
            DoseEvent::new(2.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        subject.reset_times = vec![2.0];
        assert!(subject.doses[0].is_infusion() && subject.has_resets());
        // Plain subject → the static `integrate_g` path, NOT the event-driven walk.
        assert!(
            !ode_tvcov_supported(&model, &subject) && ode_subject_supported(&model, &subject),
            "infusion+reset with no TV-cov/lagtime must route to the static integrate_g walk"
        );
        check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
    }

    /// **Time-varying covariates + EVID 3/4 reset.** A TV-cov subject with a reset +
    /// re-dose routes to the event-driven walk, which must zero the dual state at the
    /// reset and match production across the reset boundary (#439 reset).
    #[test]
    fn ode_provider_tvcov_reset_matches_production() {
        let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 9.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(5.0, 100.0, 1, 0.0, false, 0.0),
        ];
        subject.dose_covariates = vec![wt(60.0), wt(75.0)];
        subject.obs_covariates = vec![wt(60.0), wt(65.0), wt(80.0), wt(85.0)];
        subject.reset_times = vec![5.0];
        assert!(subject.has_tv_covariates() && subject.has_resets());
        assert!(ode_tvcov_supported(&model, &subject));
        check_vs_production(&model, &subject, &[1.0, 20.0, 0.75], &[0.1]);
    }

    /// **Estimated lagtime + EVID 3/4 reset.** Lagtime routes to the event-driven walk;
    /// the reset (fixed time) zeros the dual state, and the post-reset re-dose's lagtime
    /// saltation lands on it. Full `SubjectSens` vs production FD (#439 lagtime × reset).
    #[test]
    fn ode_provider_lagtime_reset_matches_production() {
        let model = parse_model_string(ONECPT_ORAL_LAG_ODE).expect("parse oral lag ODE");
        let mut subject = bolus_subject(&[1.0, 6.0, 12.0, 25.0, 30.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ];
        subject.reset_times = vec![24.0];
        assert!(model.has_lagtime() && subject.has_resets());
        assert!(ode_tvcov_supported(&model, &subject));
        check_vs_production(&model, &subject, &[1.0, 10.0, 1.0, 0.5], &[0.12, -0.08]);
    }

    /// **Time-varying covariates + infusion.** A TV-cov subject with a finite-duration
    /// infusion (`rate>0`, window `[0, amt/rate]`) routes to the event-driven walk, which
    /// must apply the `F·rate` forcing over the in-window segments and match production
    /// (#439 infusion).
    #[test]
    fn ode_provider_tvcov_infusion_matches_production() {
        let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
        // Infusion into cmt 1: rate 50, amt 100 → 2 h window [0, 2]; obs 1 is in-window.
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0)];
        subject.dose_covariates = vec![wt(70.0)];
        subject.obs_covariates = vec![wt(60.0), wt(70.0), wt(80.0), wt(90.0)];
        assert!(subject.doses[0].is_infusion() && subject.has_tv_covariates());
        assert!(ode_tvcov_supported(&model, &subject));
        check_vs_production(&model, &subject, &[1.0, 20.0, 0.75], &[0.1]);
    }

    /// **Infusion straddling an EVID 3/4 reset.** An infusion window `[0, 4]` crossing a
    /// reset at `t=2` must stop contributing after the reset (the reset zeroes the state and
    /// turns the infusion off — production's `reset_floor`). If the dual walk kept adding
    /// `F·rate` to the post-reset segments, the *prediction* (not just the gradient) would
    /// diverge — the dominant defect this guards (#472 review #1). Post-reset obs (3, 6, 9)
    /// are the ones that catch it; validated vs production FD.
    #[test]
    fn ode_provider_tvcov_infusion_reset_matches_production() {
        let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 9.0]);
        subject.doses = vec![
            // Infusion rate 50, amt 200 → 4 h window [0, 4], straddling the reset at t=2.
            DoseEvent::new(0.0, 200.0, 1, 50.0, false, 0.0),
            // EVID=4 re-dose (bolus) at the reset.
            DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0),
        ];
        subject.reset_times = vec![2.0];
        subject.dose_covariates = vec![wt(70.0), wt(70.0)];
        subject.obs_covariates = vec![wt(60.0), wt(75.0), wt(80.0), wt(85.0)];
        assert!(subject.doses[0].is_infusion() && subject.has_resets());
        assert!(ode_tvcov_supported(&model, &subject));
        check_vs_production(&model, &subject, &[1.0, 20.0, 0.75], &[0.1]);
    }

    /// **Estimated lagtime + infusion + reset.** Combines the moving infusion window with a
    /// reset that cuts it off: after the reset the rate-off saltation at the window end must
    /// *not* fire (the infusion was already stopped), and a post-reset re-dose infusion has
    /// its own (lagged) window. Full `SubjectSens` vs production FD (#472 review #2).
    #[test]
    fn ode_provider_lagtime_infusion_reset_matches_production() {
        let model = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag+inf ODE");
        let mut subject = bolus_subject(&[2.0, 4.0, 8.0, 12.0]);
        subject.doses = vec![
            // Lagged infusion (rate 50, amt 200 → 4 h window) straddling the reset at 5.
            DoseEvent::new(0.0, 200.0, 1, 50.0, false, 0.0),
            // Post-reset re-dose infusion.
            DoseEvent::new(5.0, 100.0, 1, 40.0, false, 0.0),
        ];
        subject.reset_times = vec![5.0];
        assert!(subject.doses[0].is_infusion() && subject.has_resets() && model.has_lagtime());
        assert!(ode_tvcov_supported(&model, &subject));
        check_vs_production(&model, &subject, &[1.0, 10.0, 0.5], &[0.1, 0.05]);
    }

    /// **#419: rate-defined infusion under bioavailability `F ≠ 1`.** NONMEM holds the rate
    /// and scales the *window* to `F·amt/rate`, so `F`'s sensitivity is a moving rate-off
    /// boundary (not a rate-magnitude scale). The subject routes to the event-driven walk
    /// (`has_rate_defined_under_f`), which carries it via the rate-off saltation with
    /// `δ = δt_inf`. Validated vs production FD, with `F` on IIV (`ETA_F`) for the 2nd order.
    const ONECPT_IV_F_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVF(0.7, 0.05, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_F ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = TVF * exp(ETA_F)
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

    #[test]
    fn ode_provider_rate_defined_infusion_under_f_matches_production() {
        let model = parse_model_string(ONECPT_IV_F_ODE).expect("parse F+inf ODE");
        assert!(model.has_bioavailability());
        let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
        // Rate-defined infusion (rate 40): under F≈0.7 the window is F·100/40 = 1.75 h.
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, false, 0.0)];
        assert!(subject.doses[0].is_infusion() && subject.has_rate_defined_infusion());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "rate-defined infusion under F → event-driven walk (#419)"
        );
        // η = [ETA_CL, ETA_F]; θ = [TVCL, TVV, TVF].
        check_vs_production(&model, &subject, &[1.0, 10.0, 0.7], &[0.1, 0.05]);
        // 2nd order: the rate-off-under-F `coef2 = -s·½·J·(Δr·e_cmt)` saltation block (#473
        // review #2 — the F-window-shift Hessian was previously unvalidated).
        check_hessian_vs_fd_of_grad(&model, &subject, &[1.0, 10.0, 0.7], &[0.1, 0.05]);
    }

    /// Rate-defined infusion under `F` combined with **estimated lagtime**: the rate-on
    /// boundary shifts with `lag`, the rate-off boundary with `lag` *and* `F` (combined
    /// `δ = δlag + δt_inf`). Validated vs production FD (#419 × lagtime).
    #[test]
    fn ode_provider_rate_defined_infusion_under_f_with_lag_matches_production() {
        const ONECPT_IV_F_LAG_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVF(0.7, 0.05, 1.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = TVF
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
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
        let model = parse_model_string(ONECPT_IV_F_LAG_ODE).expect("parse F+lag+inf ODE");
        assert!(model.has_bioavailability() && model.has_lagtime());
        let mut subject = bolus_subject(&[2.0, 4.0, 6.0, 10.0]);
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, false, 0.0)];
        assert!(ode_tvcov_supported(&model, &subject));
        // θ = [TVCL, TVV, TVF, TVLAG].
        check_vs_production(&model, &subject, &[1.0, 10.0, 0.7, 0.5], &[0.1]);
    }

    /// **Estimated lagtime + infusion.** The infusion *window* `[t+lag, t+lag+dur]` shifts
    /// with `lag`, so the lagtime sensitivity is the event-time saltation at **both** rate
    /// boundaries (rate-on and rate-off). Full `SubjectSens` vs production FD, with lag on
    /// IIV (`ETA_LAG`) to exercise the 2nd-order rate-boundary term (#439 lagtime × infusion).
    const ONECPT_IV_LAG_INF_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  LAGTIME = TVLAG * exp(ETA_LAG)
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

    /// **Steady-state (SS=1) bolus dose.** The dual SS-equilibration loads the
    /// infinite-past pulse-train trough (carrying `∂SS/∂(θ,η)`) at the SS dose, then the
    /// dose's own pulse applies. Validated `f`/`∂f/∂η`/`∂f/∂θ` vs the production predictor
    /// (which equilibrates the same way) + FD (#439 Tier 2 steady state).
    const ONECPT_IV_SS_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
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

    #[test]
    fn ode_provider_ss_bolus_matches_production() {
        let model = parse_model_string(ONECPT_IV_SS_ODE).expect("parse SS ODE");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 9.0]);
        // SS=1 bolus into central, II = 12 h.
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
        assert!(subject.doses[0].ss && subject.doses[0].ii > 0.0);
        assert!(
            ode_tvcov_supported(&model, &subject),
            "SS bolus → event-driven walk"
        );
        check_vs_production(&model, &subject, &[1.0, 10.0], &[0.1]);
        // 2nd order: the SS equilibration's `∂²(SS state)/∂(θ,η)²` (#473 review #2 — the SS
        // dual-equilibration Hessian was previously unvalidated).
        check_hessian_vs_fd_of_grad(&model, &subject, &[1.0, 10.0], &[0.1]);
    }

    /// **Early-stop is value-preserving and gradient-faithful to well within validation
    /// precision** (#532 review #1/#2/#4). On a scale-separated 2-cpt SS=1 fit — small `V2` puts
    /// the peripheral compartment ~50× below central, the regime where a magnitude-only floor
    /// could short-circuit the stop — the early-stopped analytic sensitivities are compared
    /// against a *forced-full-budget* equilibration. The dual decides convergence on the value
    /// parts, so:
    ///
    /// - **Predictions** match to f64 precision: the value has reached its fixed point, so the
    ///   elided cycles do not move it.
    /// - **Gradients / Hessian blocks** match to `< 1e-6` relative (measured `~1e-8` on this
    ///   stressed model). The derivative tails contract at the value's geometric rate but lag by
    ///   a constant few cycles, so a small tail survives the value stop (#532 review #2). That
    ///   tail is 3–4 orders below the `1e-3` FD gradient-validation tolerance, the `1e-9` ODE
    ///   solver `reltol`, and NONMEM's ~`1e-5` SE-matching precision — i.e. invisible to every
    ///   reported number, which is the precise sense in which SEs are "unchanged".
    ///
    /// Running here also exercises the dual stop end-to-end (#532 review #5).
    #[test]
    fn ode_provider_ss_early_stop_matches_full_budget() {
        let model = parse_model_string(TWOCPT_ODE).expect("parse 2-cpt SS ODE");
        let mut subject = bolus_subject(&[1.0, 4.0, 8.0, 11.0, 20.0]);
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
        assert!(
            ode_tvcov_supported(&model, &subject),
            "SS bolus → analytic walk"
        );
        // CL/V1 fast central; small V2 → a peripheral compartment ~50× below central (scale
        // separation) whose slow mode still equilibrates inside the cycle budget.
        let theta = [4.0, 50.0, 8.0, 1.0];
        let eta = [0.1, 0.05];

        let early = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
        let early_cycles = crate::ode::predictions::last_ss_equilibration_cycles();
        let full = crate::ode::predictions::with_full_ss_equilibration(|| {
            ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported")
        });
        let full_cycles = crate::ode::predictions::last_ss_equilibration_cycles();

        // The dual stop must actually fire on this model, or the comparison is vacuous (#532 #5).
        assert_eq!(
            full_cycles,
            crate::ode::predictions::SS_EQUILIBRATION_CYCLES,
            "forced-full must run the whole budget"
        );
        assert!(
            early_cycles < full_cycles,
            "early stop should run fewer cycles ({early_cycles}) than the full budget ({full_cycles})"
        );

        // Predictions: value reached its fixed point → preserved tightly.
        for (e, f) in early.obs.iter().zip(&full.obs) {
            approx::assert_relative_eq!(e.f, f.f, max_relative = 1e-9, epsilon = 1e-12);
        }
        // Gradients / Hessian blocks: the dropped derivative tail is below validation precision.
        for (e, f) in early.obs.iter().zip(&full.obs) {
            for (a, b) in e.df_deta.iter().zip(&f.df_deta) {
                approx::assert_relative_eq!(*a, *b, max_relative = 1e-6, epsilon = 1e-9);
            }
            for (a, b) in e.df_dtheta.iter().zip(&f.df_dtheta) {
                approx::assert_relative_eq!(*a, *b, max_relative = 1e-6, epsilon = 1e-9);
            }
            for (a, b) in e.d2f_deta2.iter().zip(&f.d2f_deta2) {
                approx::assert_relative_eq!(*a, *b, max_relative = 1e-6, epsilon = 1e-9);
            }
            for (a, b) in e.d2f_deta_dtheta.iter().zip(&f.d2f_deta_dtheta) {
                approx::assert_relative_eq!(*a, *b, max_relative = 1e-6, epsilon = 1e-9);
            }
        }
    }

    // 1-cpt IV with a **TAD-dependent RHS** (`-(CL/V)·central·(1+0.02·TAD)`). Used to
    // verify the #473 review (13:22) finding #1: does an SS dose + a `TAD`-referencing RHS
    // diverge from production for observations beyond one `II`?
    const TAD_SS_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central * (1.0 + 0.02 * TAD)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    /// **SS=1 dose with a `TAD`-dependent (non-autonomous) RHS routes to FD** (#473 review #1).
    /// The SS dual equilibration expands a *time-invariant* pulse train (cycle-relative time),
    /// so a `TAD`/`TIME`-dependent RHS breaks the steady-state cycle recurrence — the analytic
    /// walk was verified to diverge ~40× from the production predictor on this model — so the
    /// gate must decline it. (A non-SS `TAD` RHS is fine — the TV-cov/static walks anchor TAD
    /// correctly; see `ode_provider_tvcov_tad_dependent_rhs_matches_production`.)
    #[test]
    fn ode_provider_ss_tad_dependent_rhs_routes_to_fd() {
        let model = parse_model_string(TAD_SS_ODE).expect("parse TAD SS ODE");
        let mut subject = bolus_subject(&[2.0, 8.0, 20.0]);
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
        assert!(subject.doses[0].ss);
        assert!(
            model
                .ode_spec
                .as_ref()
                .and_then(|o| o.rhs_program.as_ref())
                .is_some_and(|p| p.uses_time_vars()),
            "TAD_SS_ODE's RHS must read TAD (precondition for the gate)"
        );
        assert!(
            !ode_tvcov_supported(&model, &subject),
            "SS + TAD-dependent RHS must route to FD (#473 review #1)"
        );
    }

    /// **Steady-state (SS=1) infusion.** Each equilibration cycle runs an active-rate
    /// window (`F·rate` forcing) then a quiet decay window; the dual carries `∂SS/∂(θ,η)`
    /// through both, and the SS dose's own current-cycle window is applied via the segment
    /// forcing. Validated vs production FD (#439 SS infusion).
    #[test]
    fn ode_provider_ss_infusion_matches_production() {
        let model = parse_model_string(ONECPT_IV_SS_ODE).expect("parse SS ODE");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 9.0]);
        // SS=1 infusion into central: rate 40, amt 100 → 2.5 h window, II = 12 h.
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, true, 12.0)];
        assert!(subject.doses[0].ss && subject.doses[0].is_infusion());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "SS infusion → event-driven walk"
        );
        check_vs_production(&model, &subject, &[1.0, 10.0], &[0.1]);
    }

    /// **SS × time-varying covariates.** The SS equilibration uses the SS dose's covariate
    /// snapshot, and the post-dose obs read per-event params — both via the event-driven
    /// walk. Validated vs production FD (#439 SS composing with TV-cov).
    #[test]
    fn ode_provider_ss_tvcov_matches_production() {
        let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 9.0]);
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
        subject.dose_covariates = vec![wt(70.0)];
        subject.obs_covariates = vec![wt(60.0), wt(70.0), wt(80.0), wt(90.0)];
        assert!(subject.doses[0].ss && subject.has_tv_covariates());
        assert!(ode_tvcov_supported(&model, &subject));
        check_vs_production(&model, &subject, &[1.0, 20.0, 0.75], &[0.1]);
    }

    /// **SS × estimated lagtime is now analytic (#486, PR3 sub-case (a)).** The SS dose
    /// arrives at `t_dose + lag`, so observations in the pre-arrival window
    /// `[t_dose, t_dose+lag)` must read the previous interval's steady-state tail — the
    /// walk seeds it via a `K_SS_SEED` timeline break (`ss_state_at_phase_g`, phase
    /// `II − lag`), mirroring production's `ss_state_at_phase` at the dose record time. Obs
    /// at `0.2` (< lag ≈ 0.5) exercises the pre-arrival seed directly; the rest exercise the
    /// dose's own (unmodified) general lagtime saltation and later cycles.
    #[test]
    fn ode_provider_ss_lagtime_matches_production() {
        let model = parse_model_string(ONECPT_ORAL_LAG_ODE).expect("parse oral lag ODE");
        let mut subject = bolus_subject(&[0.2, 1.0, 4.0, 8.0, 13.0]);
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
        assert!(subject.doses[0].ss && model.has_lagtime());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "SS + lagtime → event-driven walk (#486)"
        );
        // θ = [TVCL, TVV, TVKA, TVLAG].
        let theta = [1.0, 10.0, 1.0, 0.5];
        let eta = [0.12, -0.08];
        check_vs_production(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
        // 2nd order: the pre-arrival seed's `extend_flow_by_dual_duration_g` Taylor term and
        // the dose's own (now g_minus=0) lagtime saltation coefficient are both new — neither
        // is exercised by the existing non-SS lagtime Hessian test.
        check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
    }

    /// **SS × lagtime × infusion is also analytic now** (same pre-arrival seed, #486). Here
    /// `phase = II − lag ≈ 11.5` exceeds the infusion's own duration (2.5 h), so the seed
    /// exercises `ss_state_at_phase_g`'s harder branch: crossing the active/quiet boundary
    /// (the `inject_rate_saltation` reuse) before extending the quiet leg by the remaining
    /// dual duration. Obs at `0.3` (< lag) lands in the pre-arrival window.
    #[test]
    fn ode_provider_ss_lagtime_infusion_matches_production() {
        let model = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag+inf ODE");
        let mut subject = bolus_subject(&[0.3, 3.0, 5.0, 8.0, 11.0]);
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, true, 12.0)];
        assert!(subject.doses[0].ss && subject.doses[0].is_infusion() && model.has_lagtime());
        assert!(ode_tvcov_supported(&model, &subject));
        // θ = [TVCL, TVV, TVLAG]; η = [ETA_CL, ETA_LAG].
        let theta = [1.0, 10.0, 0.5];
        let eta = [0.1, 0.05];
        check_vs_production(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
        // 2nd order: the rate-on pre-advance (`extend_flow_by_dual_duration_g`, feeding the
        // *unmodified* `inject_rate_saltation`) is new machinery, specific to this combo.
        check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
    }

    /// **Rate-defined SS infusion under `F ≠ 1` is now analytic (#486, PR3 sub-case (b)).**
    /// `equilibrate_ss_state_g` reads the caller's `inf_eff` jet (window `F·duration`, rate
    /// held) instead of the raw fixed `dose.duration`, and injects the same per-cycle
    /// rate-off saltation the main walk's `K_INF_END` uses.
    #[test]
    fn ode_provider_ss_rate_defined_infusion_under_f_matches_production() {
        let model = parse_model_string(ONECPT_IV_F_ODE).expect("parse F ODE");
        assert!(model.has_bioavailability());
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0]);
        // SS=1 rate-defined infusion (rate 40) under F≈0.7.
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, true, 12.0)];
        assert!(
            subject.doses[0].ss
                && subject.doses[0].is_infusion()
                && subject.has_rate_defined_infusion()
        );
        assert!(
            ode_tvcov_supported(&model, &subject),
            "rate-defined SS infusion under F → event-driven walk (#486)"
        );
        // η = [ETA_CL, ETA_F]; θ = [TVCL, TVV, TVF].
        let theta = [1.0, 10.0, 0.7];
        let eta = [0.1, 0.05];
        check_vs_production(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
        check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
    }

    /// **Early-stop stays gradient-faithful with the new per-cycle rate-off saltation**
    /// (#486, mirroring `ode_provider_ss_early_stop_matches_full_budget`): the (b) sub-case's
    /// per-cycle `inject_rate_saltation` call accumulates over however many cycles the SS
    /// loop runs, so an early-stopped run must still match a forced-full-budget run to well
    /// within FD-validation precision — not just for the value, but for the `F`-window
    /// saltation's gradient/Hessian contribution specifically.
    #[test]
    fn ode_provider_ss_rate_under_f_early_stop_matches_full_budget() {
        let model = parse_model_string(ONECPT_IV_F_ODE).expect("parse F ODE");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 9.0]);
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, true, 12.0)];
        assert!(ode_tvcov_supported(&model, &subject), "SS + F → analytic");
        let theta = [1.0, 10.0, 0.7];
        let eta = [0.1, 0.05];

        let early = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
        let early_cycles = crate::ode::predictions::last_ss_equilibration_cycles();
        let full = crate::ode::predictions::with_full_ss_equilibration(|| {
            ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported")
        });
        let full_cycles = crate::ode::predictions::last_ss_equilibration_cycles();

        assert_eq!(
            full_cycles,
            crate::ode::predictions::SS_EQUILIBRATION_CYCLES,
            "forced-full must run the whole budget"
        );
        assert!(
            early_cycles < full_cycles,
            "early stop should run fewer cycles ({early_cycles}) than the full budget ({full_cycles})"
        );
        for (e, f) in early.obs.iter().zip(&full.obs) {
            approx::assert_relative_eq!(e.f, f.f, max_relative = 1e-9, epsilon = 1e-12);
            for (a, b) in e.df_deta.iter().zip(&f.df_deta) {
                approx::assert_relative_eq!(*a, *b, max_relative = 1e-6, epsilon = 1e-9);
            }
            for (a, b) in e.df_dtheta.iter().zip(&f.df_dtheta) {
                approx::assert_relative_eq!(*a, *b, max_relative = 1e-6, epsilon = 1e-9);
            }
            // The per-cycle rate-off saltation accumulates into the 2nd-order term across every
            // cycle, so the Hessian blocks — not just the gradient — must survive early stop
            // (this is what the docstring claims; #642 review #2).
            for (a, b) in e.d2f_deta2.iter().zip(&f.d2f_deta2) {
                approx::assert_relative_eq!(*a, *b, max_relative = 1e-6, epsilon = 1e-9);
            }
            for (a, b) in e.d2f_deta_dtheta.iter().zip(&f.d2f_deta_dtheta) {
                approx::assert_relative_eq!(*a, *b, max_relative = 1e-6, epsilon = 1e-9);
            }
        }
    }

    /// **Modeled-duration dose × SS is now analytic (#486, PR3 sub-case (d) — cheapest,
    /// implemented first).** `equilibrate_ss_state_g` threads the `inf_eff` jet (rebuilt
    /// from the `D1` PK slot, exactly as the non-SS modeled-duration walk does) into its
    /// per-cycle active/quiet split and rate-off saltation. Observations straddle the SS
    /// dose's own (post-equilibration) window end at `D1=5`.
    #[test]
    fn ode_provider_ss_modeled_duration_matches_production() {
        const MODELED_DUR_SS_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(MODELED_DUR_SS_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
        subject.doses = vec![DoseEvent::modeled(
            0.0,
            100.0,
            1,
            true,
            12.0,
            crate::types::RateMode::ModeledDuration,
        )];
        assert!(subject.doses[0].ss && !subject.doses[0].is_fixed());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "modeled-duration × SS → event-driven walk (#486)"
        );
        let theta = [5.0, 50.0, 5.0];
        let eta = [0.12, -0.08];
        check_vs_production(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
        check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
    }

    #[test]
    fn ode_provider_lagtime_infusion_matches_production() {
        let model = parse_model_string(ONECPT_IV_LAG_INF_ODE).expect("parse lag+inf ODE");
        assert!(model.has_lagtime());
        let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0, 12.0]);
        // Infusion into central: rate 40, amt 100 → 2.5 h window, shifted by the lagtime.
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 40.0, false, 0.0)];
        assert!(subject.doses[0].is_infusion());
        assert!(ode_tvcov_supported(&model, &subject));
        // η = [ETA_CL, ETA_LAG]; θ = [TVCL, TVV, TVLAG].
        check_vs_production(&model, &subject, &[1.0, 10.0, 0.5], &[0.1, 0.05]);
    }

    /// Covariate models: the provider must fold the subject's covariate-adjusted
    /// typical values (here WT on CL/V1) into both `f` and `∂f/∂θ`. Validated
    /// against the production predictor, which folds WT the same way.
    // Same 2-cpt ODE as `TWOCPT_ODE`, but the individual parameters are declared
    // with an IIV-free parameter *first* (Q, then CL, then V2, then V1). This forces
    // the mixed-order axis permutation to be non-trivial (`axis_of != identity`):
    // the IIV-bearing CL/V1 must be relocated to the leading dual axes 0/1.
    const TWOCPT_ODE_REORDER: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  Q  = TVQ
  CL = TVCL * exp(ETA_CL)
  V2 = TVV2
  V1 = TVV1 * exp(ETA_V1)
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// The mixed-order dual (`run_subject_mixed`, dropping the IIV-free Hessian
    /// block) must reproduce the full `Dual2` provider (`run_subject`) on every
    /// `ObsSens` field. Covers both the identity-permutation case (`TWOCPT_ODE`:
    /// CL/V1 declared first) and a non-trivial permutation (`TWOCPT_ODE_REORDER`:
    /// CL/V1 relocated to the leading axes). Issue #445.
    #[test]
    fn ode_mixed_matches_full_dual2() {
        for src in [TWOCPT_ODE, TWOCPT_ODE_REORDER] {
            let model = parse_model_string(src).expect("parse");
            let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
            let theta = vec![4.0, 12.0, 2.0, 25.0];
            let eta = vec![0.12, -0.08];

            // Full reference: force the 4-axis `Dual2` path directly.
            let pd = param_derivatives(&model, &subject, &theta, &eta).expect("pd");
            let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates, 0.0);
            let full =
                run_subject::<4>(&model, &subject, &theta, &eta, &pk.values, &pd).expect("full");
            // Mixed path via the dispatcher: na = 2 (CL, V1) < n = 4 routes to
            // `run_subject_mixed::<2, 4>`, dropping the Q/V2 Hessian block.
            let mixed = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("mixed");

            assert_eq!(full.obs.len(), mixed.obs.len());
            let close = |a: &[f64], b: &[f64], what: &str| {
                assert_eq!(a.len(), b.len(), "{what} length");
                for (x, y) in a.iter().zip(b) {
                    approx::assert_relative_eq!(x, y, max_relative = 1e-9, epsilon = 1e-12);
                }
            };
            for (fo, mo) in full.obs.iter().zip(&mixed.obs) {
                approx::assert_relative_eq!(fo.f, mo.f, max_relative = 1e-12);
                close(&fo.df_deta, &mo.df_deta, "df_deta");
                close(&fo.d2f_deta2, &mo.d2f_deta2, "d2f_deta2");
                close(&fo.df_dtheta, &mo.df_dtheta, "df_dtheta");
                close(&fo.d2f_deta_dtheta, &mo.d2f_deta_dtheta, "d2f_deta_dtheta");
            }
        }
    }

    /// The mixed-order dual must reproduce the full `Dual2` provider for an
    /// **`init(...)`-bearing** model routed through the mixed path (`na < n`), which
    /// exercises the axis-mapped FD initial-state seeding in `dual_init_state` — the
    /// one new numerical path the identity/reorder parity test above never hits (its
    /// models have no `init` block) (#445 review #1). Here only `CL` carries IIV, so
    /// `na = 1 < n = 3` (`V` and `KA` are IIV-free): `init(central) = 1000/V` seeds an
    /// IIV-free gradient-only axis, and the two IIV-free parameters exercise both the
    /// skipped diagonal (#448 review #8) and the both-axes-dropped cross-term
    /// `continue` in the second-order FD seeding (#448 review #4).
    #[test]
    fn ode_mixed_init_matches_full_dual2() {
        const INIT_ODE_MIXED: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.0, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  init(central) = 1000.0 / V
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(INIT_ODE_MIXED).expect("parse");
        assert_eq!(model.n_eta, 1);
        assert_eq!(model.pk_indices.len(), 3);
        let mut subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        subject.doses = vec![]; // no dose; the `init(...)` baseline is the sole input.
        let theta = vec![1.0, 20.0, 1.0];
        let eta = vec![0.1];

        let pd = param_derivatives(&model, &subject, &theta, &eta).expect("pd");
        let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates, 0.0);
        let full = run_subject::<3>(&model, &subject, &theta, &eta, &pk.values, &pd).expect("full");
        // na = 1 (CL) < n = 3 → run_subject_mixed::<1, 3> (2 IIV-free params).
        let mixed = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("mixed");

        assert_eq!(full.obs.len(), mixed.obs.len());
        for (fo, mo) in full.obs.iter().zip(&mixed.obs) {
            approx::assert_relative_eq!(fo.f, mo.f, max_relative = 1e-12, epsilon = 1e-12);
            for (a, b) in fo.df_deta.iter().zip(&mo.df_deta) {
                approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-12);
            }
            for (a, b) in fo.d2f_deta2.iter().zip(&mo.d2f_deta2) {
                approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-12);
            }
            for (a, b) in fo.df_dtheta.iter().zip(&mo.df_dtheta) {
                approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-12);
            }
            for (a, b) in fo.d2f_deta_dtheta.iter().zip(&mo.d2f_deta_dtheta) {
                approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-12);
            }
        }
    }

    /// Micro-benchmark: outer-gradient sensitivities via the full `Dual2<4>` path
    /// vs the mixed `DualMixed<2, 4>` path (Q/V2 Hessian block dropped) on the
    /// 2-cpt ODE. Reports ns/call and the speedup. Run with
    /// `cargo test --release -- --ignored --nocapture bench_mixed_vs_full`.
    #[test]
    #[ignore = "micro-benchmark; run with --release --ignored --nocapture"]
    fn bench_mixed_vs_full() {
        use std::hint::black_box;
        use std::time::Instant;
        let model = parse_model_string(TWOCPT_ODE).expect("parse");
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = vec![4.0, 12.0, 2.0, 25.0];
        let eta = vec![0.12, -0.08];
        let pd = param_derivatives(&model, &subject, &theta, &eta).expect("pd");
        let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates, 0.0);
        let iiv = vec![0usize, 1]; // CL, V1
        let axis_of = vec![0usize, 1, 2, 3]; // identity (IIV declared first)

        let iters = 50_000;
        // Warm up.
        for _ in 0..1000 {
            black_box(run_subject::<4>(
                &model, &subject, &theta, &eta, &pk.values, &pd,
            ));
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            black_box(run_subject::<4>(
                &model, &subject, &theta, &eta, &pk.values, &pd,
            ));
        }
        let full = t0.elapsed();
        let t1 = Instant::now();
        for _ in 0..iters {
            black_box(run_subject_mixed::<2, 4>(
                &model, &subject, &theta, &eta, &pk.values, &pd, &axis_of, &iiv,
            ));
        }
        let mixed = t1.elapsed();
        let fns = full.as_nanos() as f64 / iters as f64;
        let mns = mixed.as_nanos() as f64 / iters as f64;
        eprintln!("full  Dual2<4>      = {fns:8.1} ns/call");
        eprintln!("mixed DualMixed<2,4> = {mns:8.1} ns/call");
        eprintln!(
            "speedup = {:.2}x  ({:.0}% faster)",
            fns / mns,
            100.0 * (fns - mns) / fns
        );
    }

    #[test]
    fn ode_provider_2cpt_covariate_matches_production() {
        let model = parse_model_string(TWOCPT_ODE_COV).expect("parse");
        assert!(ode_analytical_supported(&model));
        // A subject whose weight differs from the 70 kg reference, so the
        // covariate genuinely shifts CL/V1 and their θ-Jacobian.
        let subject = bolus_subject_wt(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0], 90.0);
        let theta = vec![4.0, 12.0, 2.0, 25.0];
        let eta = vec![0.12, -0.08];

        let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
        let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
            compute_predictions_with_tv(&model, &subject, th, e)[j]
        };
        let he = 1e-6;
        for (j, obs) in sens.obs.iter().enumerate() {
            approx::assert_relative_eq!(
                obs.f,
                pred(&eta, &theta, j),
                max_relative = 1e-6,
                epsilon = 1e-9
            );
            for k in 0..model.n_eta {
                let mut ep = eta.clone();
                ep[k] += he;
                let mut em = eta.clone();
                em[k] -= he;
                let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 1e-3, epsilon = 1e-6);
            }
            for m in 0..model.n_theta {
                let s = he * (1.0 + theta[m].abs());
                let mut tp = theta.clone();
                tp[m] += s;
                let mut tm = theta.clone();
                tm[m] -= s;
                let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * s);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 1e-3,
                    epsilon = 1e-6
                );
            }
        }
    }

    // Per-CMT Form-C readout (#439): a 2-cpt model observed at two endpoints —
    // central concentration at CMT 1 (`central/V1`) and peripheral concentration
    // at CMT 2 (`peripheral/V2`). Each observation reads its own CMT's output
    // program over the dual state, selected by `subject.obs_cmts`.
    const TWOCPT_ODE_PERCMT: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y[CMT=1] = central / V1
  y[CMT=2] = peripheral / V2
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    fn percmt_subject(times: &[f64], cmts: &[usize]) -> Subject {
        let mut s = bolus_subject(times);
        s.obs_cmts = cmts.to_vec();
        s
    }

    /// The per-CMT provider's `f`/`∂f/∂η`/`∂f/∂θ` must match the production predictor
    /// + FD, with each observation routed through its CMT's output program.
    #[test]
    fn ode_provider_percmt_matches_production() {
        let model = parse_model_string(TWOCPT_ODE_PERCMT).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "per-CMT Form-C ODE readout should be supported (#439)"
        );
        // Observations alternate between the two endpoints.
        let subject = percmt_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0], &[1, 2, 1, 2, 1, 2]);
        check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
    }

    /// The `PerCmt` gate's negative branches must drop the model out of analytic
    /// scope (→ FD), not silently admit it: an empty endpoint map, or an endpoint
    /// whose `program` is `None` (hand-constructed / non-`is_dual_evaluable`, which the dual
    /// provider can't evaluate) (#446 review — patch coverage on the reject path).
    #[test]
    fn ode_provider_percmt_gate_rejects_incomplete_map() {
        // Empty per-CMT map → declined.
        let mut empty = parse_model_string(TWOCPT_ODE_PERCMT).expect("parse");
        empty.ode_spec.as_mut().expect("ode").readout =
            OdeReadout::PerCmt(std::collections::HashMap::new());
        assert!(
            !ode_analytical_supported(&empty),
            "empty per-CMT map must decline to FD"
        );

        // One endpoint with `program: None` (keeps its f64 `out_fn`) → declined,
        // since the dual provider has no differentiable program to evaluate.
        let mut no_prog = parse_model_string(TWOCPT_ODE_PERCMT).expect("parse");
        match &mut no_prog.ode_spec.as_mut().expect("ode").readout {
            OdeReadout::PerCmt(map) => {
                let any_cmt = *map.keys().next().expect("at least one endpoint");
                map.get_mut(&any_cmt).expect("entry").program = None;
            }
            _ => panic!("expected PerCmt readout"),
        }
        assert!(
            !ode_analytical_supported(&no_prog),
            "per-CMT endpoint with no differentiable program must decline to FD"
        );
    }

    /// An observation whose CMT is absent from the per-CMT map hits the defensive
    /// NaN fallback in the readout (fit-time `validate_per_cmt_scaling` rejects this
    /// upstream, so it is unreachable in a real fit — but the provider must produce
    /// NaN, not panic or silently zero it) (#446 review — patch coverage on the
    /// fallback arm, shared by the Dual2 and Dual1 walks via `integrate_subject_duals`).
    #[test]
    fn ode_provider_percmt_missing_cmt_yields_nan() {
        let model = parse_model_string(TWOCPT_ODE_PERCMT).expect("parse");
        // CMT 3 has no readout entry (the map covers 1 and 2); the gate still passes
        // (map non-empty, every present program simple).
        let subject = percmt_subject(&[0.5, 1.0, 2.0], &[1, 3, 2]);
        let theta = [4.0, 12.0, 2.0, 25.0];
        let eta = [0.12, -0.08];
        let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta)
            .expect("gate passes: map non-empty, programs simple");
        assert!(
            sens.obs[1].f.is_nan(),
            "obs on uncovered CMT 3 → NaN readout"
        );
        assert!(
            sens.obs[0].f.is_finite() && sens.obs[2].f.is_finite(),
            "covered CMTs stay finite"
        );
    }

    /// The light `Dual1` inner η-gradient must equal the full `Dual2` outer
    /// `df_deta` for a per-CMT model too (each endpoint's program over both duals).
    #[test]
    fn ode_provider_percmt_light_matches_full() {
        let model = parse_model_string(TWOCPT_ODE_PERCMT).expect("parse");
        let subject = percmt_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0], &[1, 2, 1, 2, 1, 2]);
        let theta = vec![4.0, 12.0, 2.0, 25.0];
        let eta = vec![0.12, -0.08];
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    // Time-varying covariate (#439): WT on CL changes across observations, so the
    // PK params vary along the trajectory. The (θ,η)-seeded TV-cov walk must match
    // production's event-driven predictor (`ode_predictions_event_driven`) + FD.
    const ONECPT_ODE_TVCOV: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -CL/V * central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// #486: a `TIME`-switched CL on a **non-IOV ODE** model. With no TV covariates
    /// the subject routes to the event-driven TV walk purely by `uses_time_builtin`
    /// (`ode_tvcov_supported`); the full `ode_subject_sensitivities` value/∂η/∂θ (and
    /// the 2nd-order blocks) must match central FD of `compute_predictions_with_tv`
    /// (which threads the same per-event TIME), and the light inner `ode_subject_eta_grad`
    /// must track the outer η-block.
    #[test]
    fn ode_time_builtin_provider_matches_fd_of_production() {
        const ONECPT_ODE_TIME: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVCL_LATE(0.5, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 5.0) {
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
        let model = parse_model_string(ONECPT_ODE_TIME).expect("parse ODE TIME");
        assert!(crate::parser::model_parser::compiled_model_uses_time_builtin(&model));
        let subject = bolus_subject(&[1.0, 3.0, 8.0, 16.0, 30.0]); // straddle TIME=5
        assert!(!subject.has_tv_covariates());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "TIME ODE subject must route to the event-driven walk"
        );
        let theta = vec![1.0, 0.5, 20.0];
        let eta = vec![0.15, -0.10];

        let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
        let pred = |e: &[f64], th: &[f64], j: usize| {
            compute_predictions_with_tv(&model, &subject, th, e)[j]
        };
        let he = 1e-6;
        for (j, obs) in sens.obs.iter().enumerate() {
            approx::assert_relative_eq!(
                obs.f,
                pred(&eta, &theta, j),
                max_relative = 1e-6,
                epsilon = 1e-9
            );
            for k in 0..model.n_eta {
                let mut ep = eta.clone();
                ep[k] += he;
                let mut em = eta.clone();
                em[k] -= he;
                let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 1e-3, epsilon = 1e-6);
            }
            for m in 0..model.n_theta {
                let s = he * (1.0 + theta[m].abs());
                let mut tp = theta.clone();
                tp[m] += s;
                let mut tm = theta.clone();
                tm[m] -= s;
                let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * s);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 1e-3,
                    epsilon = 1e-6
                );
            }
            // 2nd-order blocks via central FD of the analytic first-order gradient.
            let grad_at = |e: &[f64]| -> (Vec<f64>, Vec<f64>) {
                let s = ode_subject_sensitivities(&model, &subject, &theta, e).expect("ode");
                (s.obs[j].df_deta.clone(), s.obs[j].df_dtheta.clone())
            };
            for k in 0..model.n_eta {
                let mut ep = eta.clone();
                ep[k] += he;
                let mut em = eta.clone();
                em[k] -= he;
                let (de_p, dt_p) = grad_at(&ep);
                let (de_m, dt_m) = grad_at(&em);
                for l in 0..model.n_eta {
                    let d2 = (de_p[l] - de_m[l]) / (2.0 * he);
                    approx::assert_relative_eq!(
                        obs.d2f_deta2[k * model.n_eta + l],
                        d2,
                        max_relative = 2e-3,
                        epsilon = 1e-6
                    );
                }
                for m in 0..model.n_theta {
                    let d2 = (dt_p[m] - dt_m[m]) / (2.0 * he);
                    approx::assert_relative_eq!(
                        obs.d2f_deta_dtheta[k * model.n_theta + m],
                        d2,
                        max_relative = 2e-3,
                        epsilon = 1e-6
                    );
                }
            }
        }

        // Inner light Dual1 η-gradient tracks the outer η-block.
        let light = ode_subject_eta_grad(&model, &subject, &theta, &eta).expect("inner");
        assert_eq!(light.len(), sens.obs.len());
        for (o, g) in sens.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(o.f, g.f, max_relative = 1e-9, epsilon = 1e-9);
            for (a, b) in o.df_deta.iter().zip(g.df_deta.iter()) {
                approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-9);
            }
        }
    }

    /// #486: a `TIME`-switched CL combined with a **built-in `first_order` absorption
    /// forcing**. The event-driven walk carries both the per-event `TIME` seeding (#637) and
    /// the `R_in` forcing (#643), so this composition is analytic — the old
    /// `ode_analytical_supported` decline (which assumed the walk carried no `R_in`) was
    /// stale. Value / ∂η / ∂θ + the 2nd-order blocks must match FD of production, and the
    /// inner Dual1 must track the outer.
    #[test]
    fn ode_time_builtin_with_first_order_matches_production() {
        const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVCL_LATE(0.5, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 5.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V  = TVV * exp(ETA_V)
  KA = TVKA
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(M).expect("parse TIME+first_order");
        assert!(
            ode_analytical_supported(&model),
            "TIME + input-rate forcing must be analytic now (#486)"
        );
        let subject = bolus_subject(&[1.0, 3.0, 8.0, 16.0, 30.0]); // straddle TIME = 5
        assert!(ode_tvcov_supported(&model, &subject));
        let theta = vec![1.0, 0.5, 20.0, 1.0];
        let eta = vec![0.15, -0.10];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// #486: a `TIME`-switched CL combined with a non-zero **`init(...)` baseline**. The
    /// event-driven walk seeds the `init` state (#662) and threads per-event `TIME` (#637)
    /// together, so this composition is analytic — the old decline (assuming the walk seeded
    /// compartments at zero) was stale. Value / ∂η / ∂θ + 2nd-order vs FD of production.
    #[test]
    fn ode_time_builtin_with_init_matches_production() {
        const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVCL_LATE(0.5, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 5.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V  = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = 1000.0 / V
  d/dt(central) = -(CL/V)*central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(M).expect("parse TIME+init");
        assert!(model.ode_spec.as_ref().unwrap().init_fn.is_some());
        assert!(
            ode_analytical_supported(&model),
            "TIME + init(...) must be analytic now (#486)"
        );
        let mut subject = bolus_subject(&[1.0, 3.0, 8.0, 16.0, 30.0]); // straddle TIME = 5
        subject.doses = vec![]; // the init baseline is the sole input
        assert!(ode_tvcov_supported(&model, &subject));
        let theta = vec![1.0, 0.5, 20.0];
        let eta = vec![0.15, -0.10];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// #486 (route-completeness): the **three-way** composition of a `TIME`-switched CL,
    /// a built-in `first_order` absorption **input-rate forcing**, AND an estimated
    /// compartment-indexed **lagtime** (`ALAG1`). Each pair is covered elsewhere
    /// (`ode_time_builtin_with_first_order_matches_production` = TIME + `R_in`;
    /// `ode_provider_first_order_with_alag_matches_production` = `R_in` + lag), but the
    /// gate removed in this PR newly admits all three at once, so the per-event `TIME`
    /// PK-param seeding (#637), the `R_in` forcing (#643), and the lagged rate-on
    /// saltation `Δr = dose·ka` at the moving arrival `t_dose + lag` must compose. If they
    /// did not, `ode_subject_sensitivities` would return `Some` with a silently wrong
    /// gradient (no FD fallback), so value/∂η/∂θ + the 2nd-order blocks are checked against
    /// FD of production and the inner `Dual1` against the outer `Dual2`.
    #[test]
    fn ode_time_builtin_with_first_order_and_lagtime_matches_production() {
        const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVCL_LATE(0.5, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVLAG(0.3, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_KA ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 5.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V     = TVV
  KA    = TVKA * exp(ETA_KA)
  ALAG1 = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(M).expect("parse TIME+first_order+lagtime");
        assert!(model.has_lagtime());
        assert!(
            ode_analytical_supported(&model),
            "TIME + input-rate + lagtime must be analytic now (#486)"
        );
        // Obs deliberately straddle both the `TIME = 5` switch and the `t_dose + TVLAG`
        // arrival, while avoiding landing exactly on `t_dose + TVLAG` (a central-FD kink
        // in `lag`, see `ode_provider_first_order_with_alag_matches_production`).
        let subject = bolus_subject(&[0.1, 0.2, 0.4, 3.0, 8.0, 16.0, 30.0]);
        assert!(!ode_subject_supported(&model, &subject));
        assert!(ode_tvcov_supported(&model, &subject));
        let theta = vec![1.0, 0.5, 20.0, 1.0, 0.3];
        let eta = vec![0.15, -0.10];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// #486 (newly-analytic composed cell): `TIME`-switched CL + a non-zero `init(...)`
    /// baseline + a **steady-state** dose. Unlike SS + input-rate (declined above), SS +
    /// `init` IS supported by production: `equilibrate_ss_state` seeds the SS trough from
    /// zero and overwrites the state at the SS dose (discarding `init` there), while `init`
    /// still governs the **pre-SS-dose** segments — so the walk must reproduce both the
    /// decaying-`init` pre-dose predictions (crossing the `TIME` switch) and the post-dose
    /// steady state. This composition was FD before this PR (the removed gate) and is
    /// **computed** by the walk now, so it is the one genuinely-new cell that could be
    /// silently wrong; value/∂η/∂θ + 2nd-order must match FD of production.
    #[test]
    fn ode_time_builtin_with_init_and_ss_matches_production() {
        const M: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVCL_LATE(0.5, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 5.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V  = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central)  = 1000.0 / V
  d/dt(central) = -(CL/V)*central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(M).expect("parse TIME+init+SS");
        assert!(model.ode_spec.as_ref().unwrap().init_fn.is_some());
        assert!(ode_analytical_supported(&model));
        // Pre-SS-dose obs (2, 4) decay the `init` baseline and straddle TIME = 5; the SS
        // bolus lands at t = 8 (II = 12), and later obs (10, 14, 20) read the steady state.
        let mut subject = bolus_subject(&[2.0, 4.0, 10.0, 14.0, 20.0]);
        subject.doses = vec![DoseEvent::new(8.0, 100.0, 1, 0.0, true, 12.0)];
        assert!(subject.doses[0].ss && subject.has_periodic_ss_dose());
        assert!(!ode_subject_supported(&model, &subject));
        assert!(ode_tvcov_supported(&model, &subject));
        let theta = vec![1.0, 0.5, 20.0];
        let eta = vec![0.15, -0.10];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// #486: an η-dependent `ExpressionScale` `obs_scale = 1000/V` on the **ODE
    /// event-driven walk** — for a `TIME` switch AND a time-varying covariate. The walk
    /// now applies the subject-static scale quotient (`apply_expression_scale_outer`)
    /// post-walk, the SAME one the static ODE walk uses; value/∂η/∂θ (+2nd order) must
    /// match FD of `compute_predictions_with_tv` (which divides by the same scale).
    #[test]
    fn ode_expression_scale_on_event_walk_matches_production() {
        const ODE_IV_TIME_SCALED: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVCL_LATE(0.5, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 5.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = 1000 / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
        const ODE_IV_TVCOV_SCALED: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[covariates]
  WT continuous
[scaling]
  obs_scale = 1000 / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
        // TIME switch + obs_scale (routed to the event walk by uses_time_builtin).
        let m_time = parse_model_string(ODE_IV_TIME_SCALED).expect("parse ODE TIME scaled");
        assert!(matches!(
            m_time.scaling,
            ScalingSpec::ExpressionScale { .. }
        ));
        let s_time = bolus_subject(&[1.0, 3.0, 8.0, 16.0, 30.0]); // straddle TIME=5
        assert!(ode_tvcov_supported(&m_time, &s_time));
        check_vs_production(&m_time, &s_time, &[1.0, 0.5, 20.0], &[0.15, -0.10]);
        // Time-varying covariate + obs_scale (the broader gap this closes).
        let m_tv = parse_model_string(ODE_IV_TVCOV_SCALED).expect("parse ODE tvcov scaled");
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        let mut s_tv = bolus_subject(&[1.0, 2.0, 4.0, 8.0, 24.0]);
        s_tv.dose_covariates = vec![wt(60.0)];
        s_tv.obs_covariates = vec![wt(60.0), wt(70.0), wt(80.0), wt(85.0), wt(90.0)];
        assert!(s_tv.has_tv_covariates() && ode_tvcov_supported(&m_tv, &s_tv));
        check_vs_production(&m_tv, &s_tv, &[1.0, 20.0, 0.75], &[0.15, -0.10]);

        // #637 review #2: the light inner η-gradient (`run_subject_tvcov_eta` →
        // `apply_expression_scale_inner_dispatch`) must track the outer η-block — this
        // exercises the new ODE inner ExpressionScale quotient, which `check_vs_production`
        // (outer only) does not.
        let parity = |m: &CompiledModel, s: &Subject, theta: &[f64], eta: &[f64]| {
            let full = ode_subject_sensitivities(m, s, theta, eta).expect("outer");
            let light = ode_subject_eta_grad(m, s, theta, eta).expect("inner");
            assert_eq!(full.obs.len(), light.len());
            for (o, g) in full.obs.iter().zip(light.iter()) {
                approx::assert_relative_eq!(o.f, g.f, max_relative = 1e-12, epsilon = 1e-12);
                for (a, b) in o.df_deta.iter().zip(g.df_deta.iter()) {
                    approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-10);
                }
            }
        };
        parity(&m_time, &s_time, &[1.0, 0.5, 20.0], &[0.15, -0.10]);
        parity(&m_tv, &s_tv, &[1.0, 20.0, 0.75], &[0.15, -0.10]);
    }

    #[test]
    fn ode_provider_tvcov_matches_production() {
        let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
        assert_eq!(model.n_theta, 3);
        assert_eq!(model.n_eta, 1); // M = n_theta + n_eta = 4
        let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        subject.dose_covariates = vec![wt(60.0)];
        subject.obs_covariates = vec![wt(60.0), wt(70.0), wt(80.0), wt(90.0)];
        assert!(subject.has_tv_covariates());
        let theta = vec![1.0, 20.0, 0.75];
        let eta = vec![0.1];

        let sens = run_subject_tvcov::<4>(&model, &subject, &theta, &eta).expect("tvcov supported");
        let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
            compute_predictions_with_tv(&model, &subject, th, e)[j]
        };
        let he = 1e-6;
        for (j, obs) in sens.obs.iter().enumerate() {
            // Value matches production's event-driven (per-event-cov) predictor.
            approx::assert_relative_eq!(
                obs.f,
                pred(&eta, &theta, j),
                max_relative = 1e-6,
                epsilon = 1e-9
            );
            for k in 0..model.n_eta {
                let mut ep = eta.clone();
                ep[k] += he;
                let mut em = eta.clone();
                em[k] -= he;
                let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 1e-3, epsilon = 1e-6);
            }
            for m in 0..model.n_theta {
                let s = he * (1.0 + theta[m].abs());
                let mut tp = theta.clone();
                tp[m] += s;
                let mut tm = theta.clone();
                tm[m] -= s;
                let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * s);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 1e-3,
                    epsilon = 1e-6
                );
            }
            // Second order (#449 review #3): the Hessian blocks `d2f_deta2` /
            // `d2f_deta_dtheta` feed FOCEI's `log|H̃|` gradient and covariance, but
            // were untested. Validate them by central-differencing the analytic
            // first-order `df_deta` / `df_dtheta` (themselves checked above) w.r.t. η.
            let grad_at = |e: &[f64]| -> (Vec<f64>, Vec<f64>) {
                let s = run_subject_tvcov::<4>(&model, &subject, &theta, e).expect("tvcov");
                (s.obs[j].df_deta.clone(), s.obs[j].df_dtheta.clone())
            };
            for k in 0..model.n_eta {
                let mut ep = eta.clone();
                ep[k] += he;
                let mut em = eta.clone();
                em[k] -= he;
                let (de_p, dt_p) = grad_at(&ep);
                let (de_m, dt_m) = grad_at(&em);
                for l in 0..model.n_eta {
                    let d2 = (de_p[l] - de_m[l]) / (2.0 * he);
                    approx::assert_relative_eq!(
                        obs.d2f_deta2[k * model.n_eta + l],
                        d2,
                        max_relative = 2e-3,
                        epsilon = 1e-6
                    );
                }
                for m in 0..model.n_theta {
                    let d2 = (dt_p[m] - dt_m[m]) / (2.0 * he);
                    approx::assert_relative_eq!(
                        obs.d2f_deta_dtheta[k * model.n_theta + m],
                        d2,
                        max_relative = 2e-3,
                        epsilon = 1e-6
                    );
                }
            }
        }
    }

    fn tvcov_subject() -> Subject {
        let mut subject = bolus_subject(&[1.0, 2.0, 4.0, 8.0]);
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        subject.dose_covariates = vec![wt(60.0)];
        subject.obs_covariates = vec![wt(60.0), wt(70.0), wt(80.0), wt(90.0)];
        subject
    }

    // TV-cov + `init(...)` (#486): a WT covariate on CL makes it a TV-cov subject, and
    // `init(central) = BASE / V` seeds an η-dependent (via `V`) and *nonlinear* baseline —
    // so the event-driven walk's dual init seed carries `∂/∂(θ,η)` and a non-zero
    // second-order block (`∂²(BASE/V)/∂V² ≠ 0`), exercising both Taylor terms of
    // `tvcov_init_state`.
    const TVCOV_INIT_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V    = TVV * exp(ETA_V)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central)  = -CL/V * central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// TV-cov + `init(...)` on the non-IOV ODE walk (#486): `init(central) = BASE/V` is
    /// η-dependent (through `V`) and nonlinear, so the event-driven walk's dual init seed
    /// must carry `∂/∂(θ,η)` and its second-order block. The analytic `SubjectSens` (value,
    /// `∂f/∂η`, `∂f/∂θ`, + 2nd order) must match FD of the production TV-cov predictor,
    /// which seeds `init` at the first-record covariate snapshot (`init_pk`).
    #[test]
    fn ode_provider_tvcov_init_matches_production() {
        let model = parse_model_string(TVCOV_INIT_ODE).expect("parse");
        assert_eq!(model.n_theta, 4);
        assert_eq!(model.n_eta, 2); // M = n_theta + n_eta = 6
        let subject = tvcov_subject();
        assert!(subject.has_tv_covariates());
        assert!(
            subject.reset_times.is_empty(),
            "fixture must have no reset — init + reset stays FD"
        );
        assert!(
            ode_tvcov_supported(&model, &subject),
            "TV-cov + init(...) must be analytic on the non-IOV ODE walk (#486)"
        );
        let theta = vec![1.0, 20.0, 0.75, 500.0];
        let eta = vec![0.1, -0.05];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
    }

    /// Inner/outer parity for TV-cov + `init(...)` (#486): the light `Dual1` inner
    /// η-gradient — which seeds `init` through the SAME `tvcov_init_state` (its second-order
    /// term vanishes at `Dual1`) — must equal the full `Dual2` outer `df_deta`. Both walks
    /// share `ode_tvcov_supported`, so the init seed moves in lockstep or the inner EBE
    /// gradient silently diverges.
    #[test]
    fn ode_provider_tvcov_init_inner_eta_grad_matches_outer() {
        let model = parse_model_string(TVCOV_INIT_ODE).expect("parse");
        let subject = tvcov_subject();
        let theta = vec![1.0, 20.0, 0.75, 500.0];
        let eta = vec![0.1, -0.05];
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// TV-cov + `init(...)` analytic scope (#486): `init(...)` is **fully analytic** on the
    /// non-IOV event-driven walk — it composes with a finite infusion, an EVID 3/4 reset, and
    /// (via their own model fixtures) lagtime / input-rate / SS / modeled-dose. This pins the
    /// two compositions constructible from the plain bolus fixture (infusion, reset), which
    /// previously routed to FD; the others are validated by their dedicated
    /// `ode_provider_tvcov_init_*_matches_production` tests.
    #[test]
    fn ode_tvcov_init_compositions_are_analytic() {
        let model = parse_model_string(TVCOV_INIT_ODE).expect("parse");
        // Baseline: plain-bolus TV-cov + init is analytic.
        assert!(
            ode_tvcov_supported(&model, &tvcov_subject()),
            "init + bolus is analytic"
        );

        // + finite infusion → analytic (#486): the walk seeds init and adds the infusion
        // forcing on top of the running state.
        let mut s = tvcov_subject();
        s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 20.0, false, 0.0)];
        assert!(
            s.doses[0].is_infusion(),
            "fixture must be a finite infusion"
        );
        assert!(
            ode_tvcov_supported(&model, &s),
            "init + finite infusion is analytic (#486)"
        );

        // + EVID 3/4 reset → analytic (#486): the K_RESET branch re-seeds init at the
        // reset-event snapshot (matching production's `initial_state(&last_pk.values)`).
        let mut s = tvcov_subject();
        s.reset_times = vec![4.0];
        assert!(
            ode_tvcov_supported(&model, &s),
            "init + EVID 3/4 reset is analytic (#486)"
        );
    }

    // Tight-tolerance twin of `TVCOV_INIT_ODE` for the infusion branch: the `1e-11`/`1e-13`
    // solver tolerance keeps the Dual2 Hessian's value-controlled step error far below the
    // `2e-3` FD-of-gradient check. `F = 1` (bare slot), so the infusion is a fixed-boundary
    // rate-defined window (`amt/rate`) — the branch-A question (does the forcing accumulate
    // correctly on a non-zero init baseline?) without the rate-defined-under-`F` moving
    // boundary, which is a separate (#419) mechanism.
    const TVCOV_INIT_INF_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V    = TVV * exp(ETA_V)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central)  = -CL/V * central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;

    /// **TV-cov + `init(...)` + finite infusion** (#486, branch A). A 5 h IV infusion
    /// (amt 100, rate 20, `F = 1`) into the init-seeded `central` compartment, under a WT
    /// covariate that makes CL time-varying. The walk seeds `init(central) = BASE/V` and then
    /// accumulates the `rate` forcing over the infusion window on top of that non-zero running
    /// state — so the analytic `SubjectSens` (value, `∂f/∂(θ,η)`, both Hessian blocks) must
    /// match FD of the production TV-cov predictor, and the light inner `Dual1` walk must match
    /// the outer `Dual2` walk. Confirms the forcing branch carries no zero-baseline assumption.
    #[test]
    fn ode_provider_tvcov_init_infusion_matches_production() {
        let model = parse_model_string(TVCOV_INIT_INF_ODE).expect("parse");
        assert_eq!(model.n_theta, 4);
        assert_eq!(model.n_eta, 2);
        let mut subject = tvcov_subject(); // obs at 1,2,4,8; infusion runs [0,5]
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 20.0, false, 0.0)];
        assert!(
            subject.doses[0].is_infusion(),
            "fixture must be an infusion"
        );
        assert!(
            ode_tvcov_supported(&model, &subject),
            "TV-cov + init + infusion must be analytic (#486)"
        );
        let theta = vec![1.0, 20.0, 0.75, 500.0];
        let eta = vec![0.1, -0.05];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    // 1-cpt IV with an estimated lagtime carrying IIV and an `init(central) = BASE/V` baseline.
    // The lagged bolus arrives at `t + lag`; before arrival the compartment decays from `init`,
    // so the dose-arrival saltation's pre-arrival state `x⁻` is the *non-zero* init baseline.
    // `CL` carries IIV (so the init decay — hence `x⁻` and its velocity `g(x⁻)`) depends on
    // `ETA_CL`), and `LAGTIME` carries IIV (so the saltation's `∂/∂η_LAG` fires). Tight
    // tolerance so the FD-of-predict Hessian check is clean.
    const ONECPT_IV_LAG_INIT_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVLAG(0.5, 0.01, 5.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  LAGTIME = TVLAG * exp(ETA_LAG)
  BASE    = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central) = -CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;

    /// **`init(...)` + estimated lagtime** (#486, branch C). A lagged bolus (arrives at
    /// `t + lag ≈ 0.5`) on top of an `init(central) = BASE/V` decaying baseline. The obs at
    /// t=0.25 sits in the pre-arrival window (pure init decay), the rest after arrival. The
    /// dose-arrival saltation must use the actual pre-arrival state `x⁻ = u` (the init
    /// baseline) and its velocity — validated at all orders against FD of the production
    /// predictor, plus inner/outer parity. Confirms the saltation makes no zero-baseline
    /// assumption.
    #[test]
    fn ode_provider_tvcov_init_lagtime_matches_production() {
        let model = parse_model_string(ONECPT_IV_LAG_INIT_ODE).expect("parse");
        assert!(model.has_lagtime());
        let subject = bolus_subject(&[0.25, 1.0, 2.0, 4.0]);
        assert!(
            ode_tvcov_supported(&model, &subject),
            "init + lagtime must be analytic on the event-driven walk (#486)"
        );
        let theta = vec![1.0, 20.0, 0.5, 500.0];
        let eta = vec![0.1, 0.05];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// **`init(...)` + modeled-duration dose** (#486, branch E). A `RATE=-2` / `D1`-modeled
    /// infusion (window `[0, D1]`, moving boundary in `D1 = TVD1·exp(ETA_D1)`) delivering into
    /// an `init(central) = BASE/V` compartment. The moving infusion-end saltation reads the
    /// live `inf_eff` `D1` jet on top of the init-seeded running state; obs straddle the window
    /// end (`D1 = 5`: t=1,3 inside; t=6,10 after). Validated at all orders against FD of the
    /// production predictor (which resolves `D1` per evaluation), plus inner/outer parity.
    #[test]
    fn ode_provider_tvcov_init_modeled_duration_matches_production() {
        const MODELED_DUR_INIT_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  theta TVBASE(200.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  D1   = TVD1 * exp(ETA_D1)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central) = -CL/V * central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
        let model = parse_model_string(MODELED_DUR_INIT_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
        subject.doses = vec![DoseEvent::modeled(
            0.0,
            100.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        )];
        assert!(
            ode_tvcov_supported(&model, &subject) && !ode_subject_supported(&model, &subject),
            "init + modeled-duration dose must route to the event-driven walk (#486)"
        );
        let theta = vec![5.0, 50.0, 5.0, 200.0];
        let eta = vec![0.12, -0.08];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// **`init(...)` + built-in `first_order` input-rate forcing + TV-cov** (#486, branch B).
    /// A first-order absorption forcing `R_in = KA·(dose remaining)` feeds `central`, which also
    /// carries an `init(central) = BASE/V` baseline; a WT covariate on CL makes the subject
    /// time-varying (forcing the event-driven walk). `add_prepared_input_rate_forcing` adds
    /// `R_in(tad)` to `du/dt` on top of the init-seeded state — validated at all orders against
    /// FD of the production predictor, plus inner/outer parity.
    #[test]
    fn ode_provider_tvcov_init_input_rate_matches_production() {
        const FIRST_ORDER_INIT_TVCOV_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.0, 0.05, 10.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_KA ~ 0.05
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V    = TVV
  KA   = TVKA * exp(ETA_KA)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central)  = first_order(ka=KA) - CL/V * central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
        let model = parse_model_string(FIRST_ORDER_INIT_TVCOV_ODE).expect("parse");
        assert!(!model.ode_spec.as_ref().unwrap().input_rate.is_empty());
        let subject = tvcov_subject(); // WT covariate → TV-cov; obs at 1,2,4,8
        assert!(subject.has_tv_covariates());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "init + input-rate forcing + TV-cov must be analytic (#486)"
        );
        let theta = vec![1.0, 20.0, 1.0, 0.75, 500.0];
        let eta = vec![0.1, -0.05];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// **`init(...)` + `zero_order` absorption + TV-cov** (#486, branch B). `zero_order` was
    /// ported to the event-driven walk by #653; with `init` its per-segment constant `active_zero`
    /// delivery (`F·amt·frac/DUR` over `[dose, dose+DUR]`, moving end in `DUR`) rides on top of the
    /// init-seeded state, and the moving-end rate-off saltation reads that running state. A WT
    /// covariate makes the subject TV-cov. Obs straddle the window end (`DUR=4`). Validated at all
    /// orders against FD of production, plus inner/outer parity.
    #[test]
    fn ode_provider_tvcov_init_zero_order_matches_production() {
        const ZERO_ORDER_INIT_TVCOV_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.05, 24.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  theta TVBASE(2000.0, 10.0, 50000.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V    = TVV
  DUR  = TVDUR * exp(ETA_DUR)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
        let model = parse_model_string(ZERO_ORDER_INIT_TVCOV_ODE).expect("parse");
        let mut subject = tvcov_subject(); // WT covariate → TV-cov
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        assert!(subject.has_tv_covariates());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "init + zero_order + TV-cov must be analytic (#486/#653)"
        );
        let theta = vec![5.0, 50.0, 4.0, 0.75, 2000.0];
        let eta = vec![0.12, -0.08];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// **`init(...)` + steady-state dose** (#486, branch D). Production's `equilibrate_ss_state`
    /// seeds the SS trough from zero and *overwrites the whole state* at the SS dose — so it
    /// discards `init` there, exactly as the dual `equilibrate_ss_state_g` does. `init` therefore
    /// only affects the **pre-SS-dose** segments: here the SS bolus is at t=5 (`II=12`), with obs
    /// at t=1,2 in the pre-dose window reading the decaying `init(central) = BASE/V` baseline
    /// (their `∂/∂(θ,η)` carry the init seed's derivatives) and obs at t=8,14 reading the SS
    /// decay (init-independent). The walk seeds `init_state`, decays it from the true integration
    /// start (t=0) to t=5, then overwrites at the SS dose — matching production at all orders.
    #[test]
    fn ode_provider_tvcov_init_ss_matches_production() {
        const ONECPT_IV_SS_INIT_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV * exp(ETA_V)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central)  = -CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
        let model = parse_model_string(ONECPT_IV_SS_INIT_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 2.0, 8.0, 14.0]);
        // SS bolus at t=5, II=12: obs at 1,2 are pre-dose (init decay); 8,14 read the SS decay.
        subject.doses = vec![DoseEvent::new(5.0, 100.0, 1, 0.0, true, 12.0)];
        assert!(
            subject.has_periodic_ss_dose(),
            "fixture must be a periodic SS dose"
        );
        assert!(
            ode_tvcov_supported(&model, &subject),
            "init + steady-state must be analytic on the event-driven walk (#486)"
        );
        let theta = vec![1.0, 20.0, 500.0];
        let eta = vec![0.1, -0.05];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// **`init(...)` + EVID 3/4 reset** (#486, branch F). Production re-applies
    /// `init(&last_pk.values)` at each reset (the `K_RESET` branch re-seeds the state from the
    /// reset-event snapshot rather than zeroing). Here an `init(central) = BASE/V` model under a
    /// WT covariate (TV-cov, so the reset-time snapshot differs from the first-record one) takes
    /// a bolus at t=0, a reset at t=4 (which restores the init baseline), and a second bolus at
    /// t=5. Obs at t=2 (pre-reset decay), t=4.5 (post-reset init baseline, pre-2nd-dose), and
    /// t=8,12 (2nd-dose decay on the re-seeded baseline). The re-seeded jet must carry
    /// `∂init/∂(θ,η)` at the reset snapshot — validated at all orders against FD of production,
    /// plus inner/outer parity.
    #[test]
    fn ode_provider_tvcov_init_reset_matches_production() {
        let model = parse_model_string(TVCOV_INIT_ODE).expect("parse");
        // tvcov_subject has a WT covariate (TV-cov) and a bolus at t=0. Add a reset + 2nd dose,
        // and place obs around the reset. obs_covariates must match obs_times length.
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        let mut subject = tvcov_subject();
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(5.0, 100.0, 1, 0.0, false, 0.0),
        ];
        subject.dose_covariates = vec![wt(60.0), wt(85.0)];
        subject.obs_times = vec![2.0, 4.5, 8.0, 12.0];
        subject.obs_covariates = vec![wt(65.0), wt(75.0), wt(80.0), wt(90.0)];
        subject.observations = vec![1.0; 4];
        subject.obs_cmts = vec![1; 4];
        subject.cens = vec![0; 4];
        subject.occasions = vec![1; 4];
        subject.reset_times = vec![4.0];
        assert!(
            ode_tvcov_supported(&model, &subject),
            "init + reset must be analytic on the event-driven walk (#486)"
        );
        let theta = vec![1.0, 20.0, 0.75, 500.0];
        let eta = vec![0.1, -0.05];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    // 1-cpt IV with an `init(central) = BASE/V` baseline whose `V` (hence `init`) depends on a
    // WT covariate — so the *snapshot* the reset re-seed reads is observable in the gradient.
    const ONECPT_IV_INIT_WTV_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVBASE(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV * (WT / 70) * exp(ETA_V)
  BASE = TVBASE
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE / V
  d/dt(central)  = -CL/V * central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;

    /// **`init(...)` + EVID=4 reset+dose at `t = 0` (the reset is the first timeline event)**
    /// (#486 review — Copilot). Production re-applies `init(&last_pk.values)` at the reset with
    /// `last_pk = init_pk` (the first-record snapshot, dose-preferred on ties). The walk seeds
    /// `last_params` from the *same* `first_record_pk` selection, so the reset re-seed reads the
    /// dose@0 snapshot — not an arbitrary array-first obs slice. Here `init = BASE/V` and `V`
    /// depends on WT, and the dose@0 covariate (WT=60) differs from the observations' (WT=90),
    /// so a wrong snapshot at the re-seed would shift the baseline and its `∂/∂(θ,η)`. Validated
    /// against FD of the production predictor, plus inner/outer parity. (Before the fix
    /// `last_params` was `pk_at_obs.first()` = the WT=90 obs snapshot → mismatch.)
    #[test]
    fn ode_provider_init_reset_dose_at_zero_matches_production() {
        let model = parse_model_string(ONECPT_IV_INIT_WTV_ODE).expect("parse");
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        let mut subject = bolus_subject(&[1.0, 2.0, 4.0]);
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        subject.dose_covariates = vec![wt(60.0)]; // dose@0 snapshot: WT=60
        subject.obs_covariates = vec![wt(90.0), wt(90.0), wt(90.0)]; // obs: WT=90
        subject.covariates = wt(60.0);
        subject.reset_times = vec![0.0]; // EVID=4 reset+dose at t=0 → reset is the first event
        assert!(subject.has_tv_covariates());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "init + reset must be analytic (#486)"
        );
        let theta = vec![1.0, 20.0, 500.0];
        let eta = vec![0.1, -0.05];
        check_vs_production(&model, &subject, &theta, &eta);
        check_hessian_vs_production_fd(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// **Estimated lagtime × time-varying covariates.** A 1-cpt oral ODE with a WT
    /// covariate on CL *and* a bare `LAGTIME`. The static time-shift identity is invalid
    /// here (WT switches on an absolute timeline), so the lag sensitivity comes from the
    /// event-time saltation injected at the dose and propagated through the per-event
    /// (TV-cov) params. Validates the full `SubjectSens` (value, `∂f/∂η`, `∂f/∂θ`, and the
    /// 2nd-order blocks via central differences of the analytic gradient) against the
    /// production TV-cov+lagtime predictor (#439 lagtime × TV-cov).
    #[test]
    fn ode_provider_lagtime_tvcov_matches_production() {
        const ONECPT_ORAL_LAG_TVCOV_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.2, 0.01, 50.0)
  theta TVLAG(0.5, 0.01, 5.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[scaling]
  y = central / V
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
        let model = parse_model_string(ONECPT_ORAL_LAG_TVCOV_ODE).expect("parse");
        assert!(model.has_lagtime());
        let subject = tvcov_subject();
        assert!(subject.has_tv_covariates());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "TV-cov + lagtime supported"
        );
        let theta = vec![1.0, 20.0, 1.2, 0.5, 0.75];
        // η = [ETA_CL, ETA_LAG] — lag carries IIV, so this exercises the ∂²/∂η_LAG²
        // (event-time saltation 2nd-order) × TV-cov boundary interaction.
        let eta = vec![0.1, 0.05];
        let sens =
            ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("tvcov+lag supported");
        let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
            compute_predictions_with_tv(&model, &subject, th, e)[j]
        };
        let he = 1e-6;
        for (j, obs) in sens.obs.iter().enumerate() {
            approx::assert_relative_eq!(
                obs.f,
                pred(&eta, &theta, j),
                max_relative = 1e-6,
                epsilon = 1e-9
            );
            for m in 0..model.n_theta {
                let s = he * (1.0 + theta[m].abs());
                let mut tp = theta.clone();
                tp[m] += s;
                let mut tm = theta.clone();
                tm[m] -= s;
                let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * s);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
            for k in 0..model.n_eta {
                let mut ep = eta.clone();
                ep[k] += he;
                let mut em = eta.clone();
                em[k] -= he;
                let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 2e-3, epsilon = 1e-6);
            }
            // 2nd order via central differences of the analytic gradient w.r.t. η.
            let grad_at = |e: &[f64]| -> (Vec<f64>, Vec<f64>) {
                let s = ode_subject_sensitivities(&model, &subject, &theta, e).expect("supported");
                (s.obs[j].df_deta.clone(), s.obs[j].df_dtheta.clone())
            };
            for k in 0..model.n_eta {
                let mut ep = eta.clone();
                ep[k] += he;
                let mut em = eta.clone();
                em[k] -= he;
                let (de_p, dt_p) = grad_at(&ep);
                let (de_m, dt_m) = grad_at(&em);
                for l in 0..model.n_eta {
                    let d2 = (de_p[l] - de_m[l]) / (2.0 * he);
                    approx::assert_relative_eq!(
                        obs.d2f_deta2[k * model.n_eta + l],
                        d2,
                        max_relative = 3e-3,
                        epsilon = 1e-6
                    );
                }
                for m in 0..model.n_theta {
                    let d2 = (dt_p[m] - dt_m[m]) / (2.0 * he);
                    approx::assert_relative_eq!(
                        obs.d2f_deta_dtheta[k * model.n_theta + m],
                        d2,
                        max_relative = 3e-3,
                        epsilon = 1e-6
                    );
                }
            }
        }
    }

    /// A TV-cov model whose RHS references the `TAD` (time-after-dose) builtin, so
    /// the event-driven TV-cov walk's `last_dose_eff` / time-anchoring is exercised
    /// — the other TV-cov parity tests use a `t`-independent RHS, leaving the
    /// anchoring covered only through a constant (#451 / #449 review #10). Same
    /// parameter shape as `ONECPT_ODE_TVCOV`, so `tvcov_subject` + the same θ/η apply.
    #[test]
    fn ode_provider_tvcov_tad_dependent_rhs_matches_production() {
        const TVCOV_TAD_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central * (1.0 + 0.02 * TAD)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(TVCOV_TAD_ODE).expect("parse");
        assert!(model.ode_spec.as_ref().unwrap().input_rate.is_empty());
        let subject = tvcov_subject();
        assert!(ode_tvcov_supported(&model, &subject));
        // Analytic TV-cov walk (f / ∂η / ∂θ) must match the production predictor + FD
        // with the TAD-dependent RHS.
        check_vs_production(&model, &subject, &[1.0, 20.0, 0.75], &[0.1]);
    }

    /// The light `Dual1` inner η-gradient must equal the full `Dual2` outer
    /// `df_deta` for a TV-cov subject too — exercised through the dispatch, so this
    /// also covers `ode_tvcov_supported` routing both entry points (#439).
    #[test]
    fn ode_provider_tvcov_light_matches_full() {
        let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
        let subject = tvcov_subject();
        let theta = vec![1.0, 20.0, 0.75];
        let eta = vec![0.1];
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// The **inner EBE gate** admits a TV-cov bolus subject (#449 review #2): before
    /// the fix, `ode_inner_grad_supported` → `ode_subject_supported` returned false
    /// for `has_tv_covariates`, so the inner loop silently ran on FD while the outer
    /// was analytic. It must now be on, so the analytic `Dual1` TV-cov walk drives
    /// EBE convergence — matching the outer analytic scope.
    #[test]
    fn ode_tvcov_inner_gate_wired() {
        let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
        let s = tvcov_subject();
        assert!(ode_tvcov_supported(&model, &s));
        assert!(
            crate::sens::provider::ode_inner_grad_supported(&model, &s),
            "TV-cov bolus subject must take the analytic inner gradient, not FD"
        );
    }

    /// The TV-cov gate admits a bolus TV-cov subject and declines a static-covariate
    /// subject (which the normal pk-seeded path serves) and an infusion (FD fallback).
    #[test]
    fn ode_tvcov_gate_scope() {
        let model = parse_model_string(ONECPT_ODE_TVCOV).expect("parse");
        // Static covariates → not the TV-cov path.
        assert!(!ode_tvcov_supported(&model, &bolus_subject(&[1.0, 2.0])));
        // Bolus TV-cov → supported.
        let tv = tvcov_subject();
        assert!(ode_tvcov_supported(&model, &tv));
        // TV-cov + a real infusion → now supported (finite-duration infusion forcing).
        let mut inf = tv.clone();
        inf.doses[0].duration = 1.0;
        inf.doses[0].rate = inf.doses[0].amt;
        assert!(crate::ode::predictions::is_real_infusion(&inf.doses[0]));
        assert!(ode_tvcov_supported(&model, &inf));
        // TV-cov + EVID 3/4 reset → now supported (state zeroed at the reset).
        let mut rst = tv.clone();
        rst.reset_times = vec![3.0];
        assert!(ode_tvcov_supported(&model, &rst));
        // TV-cov + EVID=2 pk-only breakpoints → now supported (the walk carries `K_PKONLY`
        // events seeded at the breakpoint's covariate snapshot; no κ, so no combined-pk-only
        // analogue to build — #486).
        let mut pko = tv.clone();
        pko.pk_only_times = vec![1.5];
        pko.pk_only_covariates = vec![HashMap::from([("WT".to_string(), 75.0)])];
        assert!(ode_tvcov_supported(&model, &pko));
        // TV-cov + an η-dependent `ExpressionScale` divisor → now supported (subject-static
        // post-walk quotient, #486). The `ONECPT_ODE_TVCOV` Form-C model has no `obs_scale`,
        // so check the gate on the dedicated 2-cpt ExpressionScale + TV-cov fixture.
        let es_model = parse_model_string(TWOCPT_ODE_EXPRSCALE_TVCOV).expect("parse");
        let es_subj = exprscale_tvcov_subject();
        assert!(es_subj.has_tv_covariates());
        assert!(ode_tvcov_supported(&es_model, &es_subj));
    }

    /// A model whose individual-parameter program carries more than `MAX_ODE_AXES`
    /// (16) axes must make `param_derivatives_at_cov` return `None` gracefully (its
    /// dispatch only specializes `1..=16`, hitting the `_ => None` arm) rather than
    /// panic — the seeders propagate that `None` via `?`, so the caller falls back to
    /// FD. (The gate caps `n_theta + n_eta ≤ 16`, so this `_ => None` is otherwise
    /// reachable only through intermediate-axis inflation — see #455.) (#451 re-review #10)
    #[test]
    fn param_derivatives_at_cov_declines_over_max_axes_gracefully() {
        // 16 thetas + 1 eta = 17 axes (> MAX_ODE_AXES). All thetas feed CL so the
        // program carries every θ-axis.
        let n_th = MAX_ODE_AXES;
        let mut src = String::from("[parameters]\n");
        for i in 1..=n_th {
            src += &format!("  theta T{i}(1.0, 0.1, 10.0)\n");
        }
        src += "  omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04 (sd)\n\
                [individual_parameters]\n  CL = exp(ETA_CL) * (";
        src += &(1..=n_th)
            .map(|i| format!("T{i}"))
            .collect::<Vec<_>>()
            .join(" + ");
        src += ")\n  V = T2\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n\
                [odes]\n  d/dt(central) = -CL/V * central\n[error_model]\n  \
                DV ~ proportional(PROP_ERR)\n";
        let model = parse_model_string(&src).expect("parse");
        assert_eq!(model.n_theta, n_th);
        assert_eq!(model.n_eta, 1);
        let prog = model
            .ode_spec
            .as_ref()
            .expect("ode")
            .indiv_param_program
            .as_ref()
            .expect("prog");
        assert!(prog.n_axes() > MAX_ODE_AXES, "fixture must exceed the cap");
        let theta = vec![1.0; n_th];
        let pd = param_derivatives_at_cov(prog, &model, &HashMap::new(), &theta, &[0.1]);
        assert!(
            pd.is_none(),
            "> MAX_ODE_AXES axes must decline to FD, not panic"
        );
    }

    // ---- #430 slice 1: built-in inverse-Gaussian absorption forcing over Dual2 ----

    // 1-cpt oral disposition with Freijer & Post inverse-Gaussian absorption via
    // the built-in `igd()` input rate (mirrors examples/igd_inverse_gaussian.ferx).
    // MAT/CV2 are θ-only and appear *only* inside `igd()`, so `∂f/∂(TVMAT,TVCV2)`
    // flows entirely through the forcing — the parity check fails if the Dual2
    // forcing is wrong. Tight ODE tolerances so analytic ≡ FD is clean.
    const IGD_ODE: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVMAT(2.0, 0.05, 24.0)
  theta TVCV2(0.3, 0.001, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  MAT = TVMAT
  CV2 = TVCV2
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    // Same as IGD_ODE but with an estimated bioavailability F. The dose into the
    // igd() compartment is suppressed as a bolus and fed to `R_in` as `F·amt`, so
    // F appears *only* inside the forcing — `∂f/∂THETA_F` exercises the F
    // derivative carried by the Dual2 forcing's `f_bio` (uncovered by IGD_ODE,
    // which has no F; #430 review finding 2).
    const IGD_ODE_F: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVMAT(2.0, 0.05, 24.0)
  theta TVCV2(0.3, 0.001, 10.0)
  theta THETA_F(0.7, 0.001, 0.999)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  MAT = TVMAT
  CV2 = TVCV2
  F   = THETA_F
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    // Same as IGD_ODE but with a compartment-indexed absorption lag `ALAG1` on
    // the igd() compartment. The lag is wired through the `DoseAttrMap`, *not*
    // `pk_indices` (and not the bare `PK_IDX_LAGTIME` slot), so the provider gate
    // must consult `has_lagtime()` to exclude it (#430 review finding 1).
    const IGD_ALAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVMAT(2.0, 0.05, 24.0)
  theta TVCV2(0.3, 0.001, 10.0)
  theta TVLAG(0.3, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV  * exp(ETA_V)
  MAT   = TVMAT
  CV2   = TVCV2
  ALAG1 = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    // Same disposition shape but with a `transit()` forcing — lifted to Dual2 in
    // #430 slice 2, so it is served by the analytic provider (its `ln Γ(n+1)`
    // constant rides the `ln_gamma` Dual2 rule).
    // IIV on N (the gamma argument) is deliberate: it is what routes the transit
    // forcing's `ln Γ(n+1)` derivatives into the FOCEI Hessian — ∂²f/∂η_N² rides the
    // *trigamma* (2nd-order `ln_gamma`) rule. With IIV only on CL the second-order
    // transit test would be vacuous for trigamma (∂²/∂N² would land in the dropped
    // θ-θ block). Tight ODE tols so analytic ≡ central-FD is clean.
    const TRANSIT_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0, 0.1, 20.0)
  theta TVKA(1.0, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_N  ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  MTT = TVMTT
  N   = TVN * exp(ETA_N)
  KA  = TVKA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = transit(n=N, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot - CL/V*central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    // 1-cpt disposition with Weibull absorption via the built-in `weibull()` input
    // rate (Phase 2; mirrors examples/weibull_absorption.ferx). TD/BETA appear
    // *only* inside `weibull()`, so `∂f/∂(TVTD,TVBETA)` and `∂f/∂ETA_BETA` flow
    // entirely through the forcing — the parity check fails if the log-domain Dual2
    // forcing is wrong. IIV on BETA (the forcing param) routes the forcing's `ln`/
    // `exp` 2nd-order rules into the FOCEI Hessian (the transit-N analogue). β = 1.5
    // (> 1) so the integrand is smooth at the dose and analytic ≡ central-FD is
    // clean (the β < 1 integrable spike is unit-tested in pk/absorption.rs). Tight
    // ODE tolerances so analytic ≡ FD is crisp.
    const WEIBULL_ODE: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVTD(2.0,  0.05,  24.0)
  theta TVBETA(1.5, 0.1,  10.0)
  omega ETA_CL   ~ 0.09
  omega ETA_BETA ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  TD   = TVTD
  BETA = TVBETA * exp(ETA_BETA)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    // Same as WEIBULL_ODE but with an estimated bioavailability F. The dose into
    // the weibull() compartment is suppressed as a bolus and fed to `R_in` as
    // `F·amt`, so F appears *only* inside the forcing — `∂f/∂THETA_F` exercises the
    // F derivative carried by the Dual2 forcing (uncovered by WEIBULL_ODE).
    const WEIBULL_ODE_F: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVTD(2.0,  0.05,  24.0)
  theta TVBETA(1.5, 0.1,  10.0)
  theta THETA_F(0.7, 0.001, 0.999)
  omega ETA_CL   ~ 0.09
  omega ETA_BETA ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  TD   = TVTD
  BETA = TVBETA * exp(ETA_BETA)
  F    = THETA_F
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    // Same as WEIBULL_ODE but with a compartment-indexed absorption lag `ALAG1` on
    // the weibull() compartment — wired through the `DoseAttrMap`, not `pk_indices`,
    // so the provider gate must consult `has_lagtime()` to exclude it (the
    // kind-agnostic #430 finding-1 fix, here exercised for Weibull). The dual loop
    // never applies the dose-attr lag, so an admitted model would get a no-lag
    // gradient diverging from the f64 predictor.
    const WEIBULL_ALAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVTD(2.0,  0.05,  24.0)
  theta TVBETA(1.5, 0.1,  10.0)
  theta TVLAG(0.3, 0.01,   5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV  * exp(ETA_V)
  TD    = TVTD
  BETA  = TVBETA
  ALAG1 = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    /// The kind gate: inverse-Gaussian (#430 slice 1), transit (#430 slice 2), and
    /// Weibull (Phase 2 — log-domain forcing over `ln`/`exp`) are all lifted to
    /// Dual2, so every built-in input-rate kind is served by the analytic provider.
    #[test]
    fn input_rate_kind_supported_over_dual_gates_kinds() {
        use crate::pk::absorption::InputRateKind;
        assert!(InputRateKind::InverseGaussian.supported_over_dual());
        assert!(InputRateKind::Transit.supported_over_dual());
        assert!(InputRateKind::Weibull.supported_over_dual());
    }

    /// With the IG forcing lifted to Dual2, an `igd()` model is served by the
    /// analytic provider, and its `f`/`∂f/∂η`/`∂f/∂θ` match the production
    /// predictor + central FD — including `∂f/∂(TVMAT,TVCV2)`, which flow only
    /// through the forcing.
    #[test]
    fn ode_provider_igd_absorption_matches_production() {
        let model = parse_model_string(IGD_ODE).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "igd() model should be supported once the IG forcing is lifted to Dual2"
        );
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = vec![5.0, 50.0, 2.0, 0.3];
        let eta = vec![0.1, -0.05];
        check_vs_production(&model, &subject, &theta, &eta);
    }

    /// Slice 2 lifts transit: a `transit()` model is now served by the analytic
    /// provider, and `f`/`∂f/∂η`/`∂f/∂θ` match the production predictor + central
    /// FD — including `∂f/∂(TVMTT,TVN)` and `∂f/∂ETA_N`, which flow only through the
    /// transit forcing's `ln Γ(n+1)` constant (so this exercises the new `ln_gamma`
    /// `Dual2` = digamma rule end-to-end through the ODE integration).
    #[test]
    fn ode_provider_transit_absorption_matches_production() {
        let model = parse_model_string(TRANSIT_ODE).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "transit() should be supported once its ln_gamma forcing is lifted to Dual2 (#430 slice 2)"
        );
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = vec![5.0, 50.0, 1.0, 3.0, 1.0]; // TVCL, TVV, TVMTT, TVN, TVKA
        let eta = vec![0.1, 0.05]; // ETA_CL, ETA_N (N feeds the forcing)
        check_vs_production(&model, &subject, &theta, &eta);
    }

    /// Second-order blocks of the transit forcing: FOCEI consumes `d2f_deta2` and
    /// `d2f_deta_dtheta`, which for transit ride the **trigamma** (2nd-order
    /// `ln_gamma`) rule. `N` carries IIV (`ETA_N`), so `∂²f/∂ETA_N²` flows through
    /// `trigamma(N+1)` — a wrong trigamma rule fails here while first-order parity
    /// still passes. Validated against central FD of the analytic (already
    /// FD-checked) `df_deta`.
    #[test]
    fn ode_provider_transit_second_order_matches_fd_of_gradient() {
        let model = parse_model_string(TRANSIT_ODE).expect("parse");
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = vec![5.0, 50.0, 1.0, 3.0, 1.0];
        let eta = vec![0.1, 0.05];
        let n_eta = model.n_eta;
        let n_theta = model.n_theta;
        let base = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");

        // η-η block: FD of df_deta over η (ETA_N → trigamma through the forcing).
        let he = 1e-5;
        for l in 0..n_eta {
            let mut ep = eta.clone();
            ep[l] += he;
            let mut em = eta.clone();
            em[l] -= he;
            let sp = ode_subject_sensitivities(&model, &subject, &theta, &ep).expect("supported");
            let sm = ode_subject_sensitivities(&model, &subject, &theta, &em).expect("supported");
            for (j, obs) in base.obs.iter().enumerate() {
                for k in 0..n_eta {
                    let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * he);
                    approx::assert_relative_eq!(
                        obs.d2f_deta2[k * n_eta + l],
                        fd,
                        max_relative = 2e-3,
                        epsilon = 1e-6
                    );
                }
            }
        }

        // η-θ cross block: FD of df_deta over θ (TVMTT/TVN flow only through the forcing).
        for m in 0..n_theta {
            let s = 1e-5 * (1.0 + theta[m].abs());
            let mut tp = theta.clone();
            tp[m] += s;
            let mut tm = theta.clone();
            tm[m] -= s;
            let sp = ode_subject_sensitivities(&model, &subject, &tp, &eta).expect("supported");
            let sm = ode_subject_sensitivities(&model, &subject, &tm, &eta).expect("supported");
            for (j, obs) in base.obs.iter().enumerate() {
                for k in 0..n_eta {
                    let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * s);
                    approx::assert_relative_eq!(
                        obs.d2f_deta_dtheta[k * n_theta + m],
                        fd,
                        max_relative = 2e-3,
                        epsilon = 1e-6
                    );
                }
            }
        }
    }

    /// Phase 2 lifts Weibull: a `weibull()` model is served by the analytic
    /// provider, and `f`/`∂f/∂η`/`∂f/∂θ` match the production predictor + central FD
    /// — including `∂f/∂(TVTD,TVBETA)` and `∂f/∂ETA_BETA`, which flow only through the
    /// Weibull forcing (so this exercises the log-domain `exp(β·ln(tad/Td))` Dual2
    /// evaluation end-to-end through the ODE integration).
    #[test]
    fn ode_provider_weibull_absorption_matches_production() {
        let model = parse_model_string(WEIBULL_ODE).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "weibull() should be supported once its log-domain forcing is lifted to Dual2 (Phase 2)"
        );
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = vec![5.0, 50.0, 2.0, 1.5]; // TVCL, TVV, TVTD, TVBETA
        let eta = vec![0.1, 0.05]; // ETA_CL, ETA_BETA (BETA feeds the forcing)
        check_vs_production(&model, &subject, &theta, &eta);
    }

    /// Second-order blocks of the Weibull forcing: FOCEI consumes `d2f_deta2` and
    /// `d2f_deta_dtheta`. `BETA` carries IIV (`ETA_BETA`) and appears only inside the
    /// forcing, so `∂²f/∂ETA_BETA²` flows through the forcing's `ln`/`exp` 2nd-order
    /// `Dual2` rules — a wrong 2nd-order rule fails here while first-order parity
    /// still passes. Validated against central FD of the analytic (already
    /// FD-checked) `df_deta`.
    #[test]
    fn ode_provider_weibull_second_order_matches_fd_of_gradient() {
        let model = parse_model_string(WEIBULL_ODE).expect("parse");
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = vec![5.0, 50.0, 2.0, 1.5];
        let eta = vec![0.1, 0.05];
        let n_eta = model.n_eta;
        let n_theta = model.n_theta;
        let base = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");

        // η-η block: FD of df_deta over η (ETA_BETA → forcing 2nd-order rules).
        let he = 1e-5;
        for l in 0..n_eta {
            let mut ep = eta.clone();
            ep[l] += he;
            let mut em = eta.clone();
            em[l] -= he;
            let sp = ode_subject_sensitivities(&model, &subject, &theta, &ep).expect("supported");
            let sm = ode_subject_sensitivities(&model, &subject, &theta, &em).expect("supported");
            for (j, obs) in base.obs.iter().enumerate() {
                for k in 0..n_eta {
                    let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * he);
                    approx::assert_relative_eq!(
                        obs.d2f_deta2[k * n_eta + l],
                        fd,
                        max_relative = 2e-3,
                        epsilon = 1e-6
                    );
                }
            }
        }

        // η-θ cross block: FD of df_deta over θ (TVTD/TVBETA flow only through the forcing).
        for m in 0..n_theta {
            let s = 1e-5 * (1.0 + theta[m].abs());
            let mut tp = theta.clone();
            tp[m] += s;
            let mut tm = theta.clone();
            tm[m] -= s;
            let sp = ode_subject_sensitivities(&model, &subject, &tp, &eta).expect("supported");
            let sm = ode_subject_sensitivities(&model, &subject, &tm, &eta).expect("supported");
            for (j, obs) in base.obs.iter().enumerate() {
                for k in 0..n_eta {
                    let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * s);
                    approx::assert_relative_eq!(
                        obs.d2f_deta_dtheta[k * n_theta + m],
                        fd,
                        max_relative = 2e-3,
                        epsilon = 1e-6
                    );
                }
            }
        }
    }

    /// Bioavailability F on a weibull() model flows *only* through the input-rate
    /// forcing (the bolus into the absorption compartment is suppressed and fed to
    /// `R_in` as `F·amt`), so the analytic `∂f/∂THETA_F` here exercises the F
    /// derivative carried by the Dual2 forcing — the path WEIBULL_ODE (no F) leaves
    /// untested.
    #[test]
    fn ode_provider_weibull_absorption_with_f_matches_production() {
        let model = parse_model_string(WEIBULL_ODE_F).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "weibull()+F should be supported (F scales the dose as a dual)"
        );
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = vec![5.0, 50.0, 2.0, 1.5, 0.7];
        let eta = vec![0.1, 0.05];
        check_vs_production(&model, &subject, &theta, &eta);
    }

    /// A `weibull()` model **with a compartment-indexed lag `ALAG1`** must stay on
    /// the FD fallback: the dual loop never applies the dose-attr lag, so admitting
    /// it would give a no-lag gradient diverging from the f64 predictor. The gate is
    /// kind-agnostic (`has_lagtime()`), so Weibull inherits the #430 finding-1 fix —
    /// this pins it (the Weibull analogue of `ode_provider_igd_with_alag_*`).
    #[test]
    fn ode_provider_weibull_with_alag_stays_on_fd_fallback() {
        let model = parse_model_string(WEIBULL_ALAG_ODE).expect("parse");
        assert!(
            model.has_lagtime(),
            "ALAG1 must enable has_lagtime() (precondition for the gate)"
        );
        assert!(
            !ode_analytical_supported(&model),
            "weibull()+ALAG1 must stay on the FD fallback (#430 finding 1, kind-agnostic)"
        );
    }

    /// A `weibull()` model **with an EVID 3/4 reset** is now analytic on both the
    /// outer θ-sensitivities and the inner η-gradient (#486): the dual forcing loop
    /// threads the tracked `reset_floor` into the shared `add_prepared_input_rate_forcing`
    /// helper, exactly like the f64 predictor's `active_infusions` rule, so a dose's
    /// pre-reset tail is correctly turned off. Reset is kind-agnostic (keyed on
    /// `!input_rate.is_empty()`), so Weibull inherits the fix — pinned here (the
    /// Weibull analogue of the igd reset test).
    #[test]
    fn ode_provider_weibull_with_reset_matches_production() {
        let model = parse_model_string(WEIBULL_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 11.0, 13.0, 16.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 100.0, 1, 0.0, false, 0.0),
        ];
        subject.reset_times = vec![10.0];
        let theta = vec![5.0, 50.0, 2.0, 1.5];
        let eta = vec![0.1, 0.05];
        assert!(
            ode_subject_supported(&model, &subject),
            "weibull() + reset is now shared scope for both outer and inner (#486)"
        );
        check_vs_production(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// Built-in absorption + an EVID 3/4 reset is now analytic (#486): the dual
    /// forcing loop threads the tracked `reset_floor`, turning off a dose's
    /// pre-reset tail exactly like the f64 predictor's `active_infusions` rule.
    #[test]
    fn ode_provider_igd_with_reset_matches_production() {
        let model = parse_model_string(IGD_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 11.0, 13.0, 16.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 100.0, 1, 0.0, false, 0.0),
        ];
        subject.reset_times = vec![10.0];
        let theta = vec![5.0, 50.0, 2.0, 0.3];
        let eta = vec![0.1, -0.05];
        // The inner η-gradient shares the scope gate, so it must be analytic too —
        // else the EBE loop would run FD while the outer runs analytic (#430 review #1
        // — same shared-scope invariant, now on the "supported" side of the gate).
        assert!(
            ode_subject_supported(&model, &subject),
            "IG + reset is now shared scope for both outer and inner (#486)"
        );
        check_vs_production(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// Bioavailability F on an igd() model flows *only* through the input-rate
    /// forcing (the dose into the absorption compartment is suppressed as a bolus
    /// and fed to `R_in` as `F·amt`), so the analytic `∂f/∂THETA_F` here exercises
    /// the F derivative carried by the Dual2 forcing — the path IGD_ODE (no F)
    /// leaves untested (#430 review finding 2).
    #[test]
    fn ode_provider_igd_absorption_with_f_matches_production() {
        let model = parse_model_string(IGD_ODE_F).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "igd()+F should be supported (F scales the dose as a dual)"
        );
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = vec![5.0, 50.0, 2.0, 0.3, 0.7];
        let eta = vec![0.1, -0.05];
        check_vs_production(&model, &subject, &theta, &eta);
    }

    /// Multi-dose superposition through the IG dual forcing: with two doses the
    /// forcing loop sums `R_in(tad)` over both, and the analytic ∂f/∂(η,θ) must
    /// still match the production predictor + FD. The single-dose IGD_ODE parity
    /// test never exercises the superposition sum.
    #[test]
    fn ode_provider_igd_multidose_matches_production() {
        let model = parse_model_string(IGD_ODE).expect("parse");
        let mut subject = bolus_subject(&[0.5, 1.5, 4.0, 8.0, 13.0, 16.0, 25.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 80.0, 1, 0.0, false, 0.0),
        ];
        let theta = vec![5.0, 50.0, 2.0, 0.3];
        let eta = vec![0.1, -0.05];
        check_vs_production(&model, &subject, &theta, &eta);
    }

    /// Second-order blocks of the IG forcing: `check_vs_production` only checks
    /// first order, but FOCEI consumes `d2f_deta2` and `d2f_deta_dtheta`. Validate
    /// both against central FD of the analytic (already FD-checked) `df_deta` — if
    /// the forcing's Dual2 second-order content were wrong, this fails while the
    /// first-order parity still passes. TVMAT/TVCV2 are θ-only and live solely in
    /// the forcing, so the θ-cross block exercises the forcing's curvature.
    #[test]
    fn ode_provider_igd_second_order_matches_fd_of_gradient() {
        let model = parse_model_string(IGD_ODE).expect("parse");
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = vec![5.0, 50.0, 2.0, 0.3];
        let eta = vec![0.1, -0.05];
        let n_eta = model.n_eta;
        let n_theta = model.n_theta;
        let base = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");

        // η-η block: FD of df_deta over η.
        let he = 1e-5;
        for l in 0..n_eta {
            let mut ep = eta.clone();
            ep[l] += he;
            let mut em = eta.clone();
            em[l] -= he;
            let sp = ode_subject_sensitivities(&model, &subject, &theta, &ep).expect("supported");
            let sm = ode_subject_sensitivities(&model, &subject, &theta, &em).expect("supported");
            for (j, obs) in base.obs.iter().enumerate() {
                for k in 0..n_eta {
                    let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * he);
                    approx::assert_relative_eq!(
                        obs.d2f_deta2[k * n_eta + l],
                        fd,
                        max_relative = 2e-3,
                        epsilon = 1e-6
                    );
                }
            }
        }

        // η-θ cross block: FD of df_deta over θ.
        for m in 0..n_theta {
            let s = 1e-5 * (1.0 + theta[m].abs());
            let mut tp = theta.clone();
            tp[m] += s;
            let mut tm = theta.clone();
            tm[m] -= s;
            let sp = ode_subject_sensitivities(&model, &subject, &tp, &eta).expect("supported");
            let sm = ode_subject_sensitivities(&model, &subject, &tm, &eta).expect("supported");
            for (j, obs) in base.obs.iter().enumerate() {
                for k in 0..n_eta {
                    let fd = (sp.obs[j].df_deta[k] - sm.obs[j].df_deta[k]) / (2.0 * s);
                    approx::assert_relative_eq!(
                        obs.d2f_deta_dtheta[k * n_theta + m],
                        fd,
                        max_relative = 2e-3,
                        epsilon = 1e-6
                    );
                }
            }
        }
    }

    /// Regression for #430 review finding 1, now closed by #486: an igd() model that
    /// also declares a compartment-indexed `ALAG{n}` lag is analytic. The lag is
    /// wired through the `DoseAttrMap` (not `pk_indices`/`PK_IDX_LAGTIME`), and
    /// routes the subject to the event-driven walk (`ode_tvcov_supported`), which now
    /// carries `R_in`: the continuous `∂R_in/∂lag` flows through the dual `tad`, and
    /// the forcing's onset at the dose's lagged arrival is injected as an exact
    /// rate-on saltation. Inverse-Gaussian's onset vanishes identically (the
    /// essential singularity at `tad → 0⁺` dominates for every valid `(mat, cv2)`),
    /// so this combination has no boundary-safety caveat (contrast Weibull, still FD
    /// for `β < 1`, see `ode_provider_weibull_with_alag_stays_on_fd_fallback`).
    #[test]
    fn ode_provider_igd_with_alag_matches_production() {
        let model = parse_model_string(IGD_ALAG_ODE).expect("parse");
        assert!(
            model.has_lagtime(),
            "ALAG1 must enable has_lagtime() (precondition for the gate)"
        );
        let subj = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0]);
        assert!(
            !ode_subject_supported(&model, &subj),
            "igd()+lagtime: the static walk still declines lagtime (routes to event-driven, #439)"
        );
        assert!(
            ode_tvcov_supported(&model, &subj),
            "igd()+lagtime: the event-driven walk now carries R_in (#486)"
        );
        let theta = vec![5.0, 50.0, 2.0, 0.3, 0.3];
        let eta = vec![0.1, -0.05];
        check_vs_production(&model, &subj, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subj, &theta, &eta);
        // The onset saltation's 2nd-order coefficient (`coef2` in `inject_rate_saltation`,
        // fed `dr = rate_at_zero(...)` here) is otherwise only value/1st-order checked
        // above; FOCEI's inner Newton step and Laplace `log|H̃|` consume the 2nd-order
        // blocks, so validate them against FD of the analytic 1st-order gradient too.
        check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
    }

    // A one-compartment `first_order()` absorption model (feeds `central` directly,
    // no separate depot) — the textbook Bateman shape with an estimated lagtime.
    // Unlike inverse-Gaussian, `FirstOrder`'s onset `R_in(0⁺) = dose·ka` is *always*
    // finite and nonzero, so this is the sharpest check of the new rate-on
    // saltation's `Δr ≠ 0` arm (`ode_provider_igd_with_alag_matches_production`
    // exercises only the `Δr = 0` case, since IG's onset always vanishes).
    const FIRST_ORDER_ALAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.0,  0.05, 20.0)
  theta TVLAG(0.3, 0.01,  5.0)
  omega ETA_CL ~ 0.09
  omega ETA_KA ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  KA    = TVKA * exp(ETA_KA)
  ALAG1 = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// `first_order()` + an estimated compartment-indexed lag (#486): the onset
    /// saltation's `Δr = dose·ka` is always finite and nonzero, so this is the
    /// sharpest check of the new rate-on injection (contrast the IG test, whose
    /// onset always vanishes). Value/`∂f/∂η`/`∂f/∂θ` (incl. `∂f/∂TVLAG` and
    /// `∂f/∂TVKA`) must match the production predictor + central FD, and the inner
    /// `Dual1` walk must reproduce the outer `Dual2` walk's η-gradient exactly.
    ///
    /// Observation times deliberately avoid landing exactly on `dose.time + TVLAG`
    /// (`= 0.3` here): a nonzero-onset forcing's prediction has a genuine kink there
    /// in `lag` (0 for `lag ≥ obs.time`, smoothly increasing for `lag < obs.time`),
    /// so central FD at that exact point averages the two one-sided derivatives
    /// (one of which is 0) instead of probing the analytic gradient's right-continuous
    /// convention (`R_in` active from `tad ≥ 0⁺`) — a test artifact, not a provider bug.
    #[test]
    fn ode_provider_first_order_with_alag_matches_production() {
        let model = parse_model_string(FIRST_ORDER_ALAG_ODE).expect("parse");
        assert!(model.has_lagtime());
        let subj = bolus_subject(&[0.1, 0.2, 0.4, 0.6, 1.0, 2.0, 4.0, 8.0]);
        assert!(!ode_subject_supported(&model, &subj));
        assert!(
            ode_tvcov_supported(&model, &subj),
            "first_order()+lagtime: the event-driven walk now carries R_in (#486)"
        );
        let theta = vec![5.0, 50.0, 1.0, 0.3];
        let eta = vec![0.1, -0.05];
        check_vs_production(&model, &subj, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subj, &theta, &eta);
        // The sharpest 2nd-order check of the new onset saltation: `Δr = dose·ka` is
        // nonzero here (unlike IG/transit's vanishing onset), so `coef2`'s `rate_at_zero`
        // contribution is live. FOCEI's inner Newton step / Laplace `log|H̃|` consume
        // `d2f_deta2`/`d2f_deta_dtheta`, so validate them against FD of the analytic
        // 1st-order gradient (only value/1st-order is checked above).
        check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
    }

    // TRANSIT_ODE (depot → central, forcing on depot) with an estimated
    // compartment-indexed lag on the *depot* (the forcing's own compartment). With
    // `N > 0` (TVN = 3.0) the onset vanishes (any `n > 0` dominates to zero, mirroring
    // IG), so this exercises the `Transit` `n > 0` arm of `rate_at_zero` end-to-end.
    const TRANSIT_ALAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0, 0.1, 20.0)
  theta TVKA(1.0, 0.05, 20.0)
  theta TVLAG(0.3, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_N  ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  MTT   = TVMTT
  N     = TVN * exp(ETA_N)
  KA    = TVKA
  ALAG1 = TVLAG
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = transit(n=N, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot - CL/V*central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// `transit()` + an estimated lag on the forcing's own compartment (#486): `N`
    /// carries IIV, so the second-order `trigamma` forcing terms also flow through
    /// the event-driven walk here (not just the static one). `N > 0` throughout, so
    /// the onset saltation is the `rate_at_zero` zero-jump arm.
    #[test]
    fn ode_provider_transit_with_alag_matches_production() {
        let model = parse_model_string(TRANSIT_ALAG_ODE).expect("parse");
        assert!(model.has_lagtime());
        let subj = bolus_subject(&[0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        assert!(!ode_subject_supported(&model, &subj));
        assert!(
            ode_tvcov_supported(&model, &subj),
            "transit()+lagtime: the event-driven walk now carries R_in (#486)"
        );
        let theta = vec![5.0, 50.0, 1.0, 3.0, 1.0, 0.3];
        let eta = vec![0.1, 0.05];
        check_vs_production(&model, &subj, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subj, &theta, &eta);
        // 2nd-order check of the `rate_at_zero` zero-jump arm's saltation coefficient —
        // only value/1st-order is checked above, but FOCEI's inner Newton step / Laplace
        // `log|H̃|` consume the 2nd-order blocks.
        check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
    }

    /// `transit()` + an EVID 3/4 reset (#486, cell (b)): the static walk still
    /// declines (no lagtime here, but the subject carries a reset), and the fix is
    /// kind-agnostic, so transit inherits it exactly like igd/weibull.
    #[test]
    fn ode_provider_transit_with_reset_matches_production() {
        let model = parse_model_string(TRANSIT_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 11.0, 13.0, 16.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 100.0, 1, 0.0, false, 0.0),
        ];
        subject.reset_times = vec![10.0];
        let theta = vec![5.0, 50.0, 1.0, 3.0, 1.0];
        let eta = vec![0.1, 0.05];
        assert!(
            ode_subject_supported(&model, &subject),
            "transit() + reset is shared scope for both outer and inner (#486)"
        );
        check_vs_production(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// `zero_order()` + an EVID 3/4 reset (#486 review follow-up): the kind-agnostic
    /// gate relaxation in `ode_subject_supported` (dropped for every `InputRateKind`,
    /// not just the "smooth density" forcings this PR's forcing loop touches) also
    /// admits a pure `zero_order()` model. Its own moving-boundary cutoff is a
    /// separate, pre-existing per-segment mechanism (`zero_windows`/`reset_floor`,
    /// #530) untouched by this PR, so this closes an undocumented capability rather
    /// than adding new machinery. The reset fires *inside* the first dose's open
    /// window (`[0, DUR=4]` cut short at `t=2`) — the exact straddling case that
    /// would leak pre-reset mass if `reset_floor` weren't already threaded through
    /// `zero_windows`.
    #[test]
    fn ode_provider_zero_order_with_reset_matches_production() {
        const ZERO_ORDER_ODE_RESET: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(ZERO_ORDER_ODE_RESET).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 4.0, 8.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0),
        ];
        subject.reset_times = vec![2.0];
        let theta = vec![5.0, 50.0, 4.0];
        let eta = vec![0.1, 0.05];
        assert!(
            ode_subject_supported(&model, &subject),
            "zero_order() + reset is shared scope for both outer and inner (#486, kind-agnostic gate)"
        );
        check_vs_production(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    /// `mixed` (zero-order + first-order) absorption + an EVID 3/4 reset (#486
    /// review follow-up): the same kind-agnostic gate relaxation, now exercised for
    /// the `mixed` combination — the `zero_order` term's window is what's exposed to
    /// the reset-cutting hazard; the `first_order` term rides the general
    /// smooth-density reset fix this PR adds.
    #[test]
    fn ode_provider_mixed_with_reset_matches_production() {
        const MIXED_ODE_RESET: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVDUR(3.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  FZO  = TVFZO
  FZO1 = 1 - TVFZO
  KA   = TVKA
  DUR  = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = parse_model_string(MIXED_ODE_RESET).expect("parse");
        let mut subject = bolus_subject(&[1.0, 2.5, 4.0, 8.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(1.5, 100.0, 1, 0.0, false, 0.0),
        ];
        subject.reset_times = vec![1.5];
        let theta = vec![5.0, 50.0, 0.4, 1.0, 3.0];
        let eta = vec![0.12, -0.08];
        assert!(
            ode_subject_supported(&model, &subject),
            "mixed absorption + reset is shared scope for both outer and inner (#486, kind-agnostic gate)"
        );
        check_vs_production(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    // WEIBULL_ODE with an allometric weight covariate on CL — exercises cell (c):
    // TV-cov + input-rate on the event-driven walk (no lagtime involved at all, so
    // the dose's arrival is a fixed, non-dual boundary; the continuous forcing's
    // `∂f/∂(TVTD,TVBETA,ETA_BETA)` must still match production under changing
    // per-event PK params).
    const WEIBULL_ODE_TVCOV: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVTD(2.0,  0.05,  24.0)
  theta TVBETA(1.5, 0.1,  10.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL   ~ 0.09
  omega ETA_BETA ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL   = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V    = TVV
  TD   = TVTD
  BETA = TVBETA * exp(ETA_BETA)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central
[covariates]
  WT continuous
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// `weibull()` + a time-varying covariate (#486, cell (c)): the event-driven walk
    /// now carries `R_in`, hoisting the forcing's dose-invariant constants fresh per
    /// segment as `CL` (hence the per-event PK snapshot) changes with `WT`.
    #[test]
    fn ode_provider_weibull_tvcov_matches_production() {
        let model = parse_model_string(WEIBULL_ODE_TVCOV).expect("parse");
        let mut subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]);
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        subject.dose_covariates = vec![wt(60.0)];
        subject.obs_covariates = vec![wt(60.0), wt(65.0), wt(70.0), wt(75.0), wt(80.0), wt(85.0)];
        assert!(subject.has_tv_covariates());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "weibull()+TV-cov must be analytic on the event-driven walk (#486)"
        );
        let theta = vec![5.0, 50.0, 2.0, 1.5, 0.75];
        let eta = vec![0.1, 0.05];
        check_vs_production(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
    }

    const ZERO_ORDER_ODE_TVCOV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.05, 24.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// `zero_order(dur)` + a time-varying covariate (#486): the event-driven walk now
    /// carries the moving-boundary window — the port of the static `integrate_g`'s
    /// `zero_windows`. The constant `F·amt·frac/dur` rate is delivered as a per-segment
    /// constant (`active_zero`, rebuilt per dose from `pk_at_dose[k]` as `CL` rides `WT`),
    /// and the rate-off saltation at the moving end `d.time + dur` carries `∂/∂DUR` (hence
    /// `∂/∂ETA_DUR` and `∂/∂TVDUR`). Value / `∂f/∂η` / `∂f/∂θ` must match the production
    /// predictor + central FD, inner `Dual1` must reproduce the outer `Dual2` η-gradient,
    /// and the 2nd-order blocks (the rate-off `coef2`) must match FD of the analytic gradient.
    #[test]
    fn ode_provider_zero_order_tvcov_matches_production() {
        let model = parse_model_string(ZERO_ORDER_ODE_TVCOV).expect("parse");
        // Obs times avoid landing on the window end `w_end ≈ TVDUR·exp(ETA_DUR) ≈ 4.2`, where
        // the prediction has a genuine kink in `dur`: central FD there would average the two
        // one-sided derivatives instead of probing the right-continuous analytic convention.
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 10.0]);
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        subject.dose_covariates = vec![wt(60.0)];
        subject.obs_covariates = vec![wt(60.0), wt(65.0), wt(70.0), wt(75.0)];
        assert!(subject.has_tv_covariates());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "zero_order()+TV-cov must be analytic on the event-driven walk (#486)"
        );
        let theta = vec![5.0, 50.0, 4.0, 0.75];
        let eta = vec![0.1, 0.05];
        check_vs_production(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
        // The rate-off saltation's δdur² coefficient (`coef2`) is live here (`ETA_DUR`/`TVDUR`
        // carry IIV/θ into the moving boundary), so validate the analytic 2nd-order blocks
        // against FD of the analytic first-order gradient — the value/1st-order parity above
        // does not exercise `coef2`.
        check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
    }

    const ZERO_ORDER_ALAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(4.0, 0.05, 24.0)
  theta TVLAG(0.3, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  DUR   = TVDUR * exp(ETA_DUR)
  ALAG1 = TVLAG
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// `zero_order(dur)` + an estimated compartment-indexed lag (#486): the sharpest check of
    /// the two coupled moving boundaries. The window is `[t_dose+lag, t_dose+lag+dur]`, so a
    /// rate-on saltation (`Δr = F·amt/dur`, the same constant `active_zero` rate) fires at the
    /// lagged arrival and a rate-off saltation (offset `δlag + δdur`) at the window end. Both
    /// the value / `∂f/∂η` / `∂f/∂θ` (incl. `∂f/∂TVLAG`, `∂f/∂TVDUR`) and the 2nd-order blocks
    /// must match the production predictor + FD, and the inner `Dual1` walk must reproduce the
    /// outer `Dual2` η-gradient exactly.
    ///
    /// Observation times avoid landing exactly on the lagged arrival (`lag = 0.3`) and the
    /// window end (`lag + dur ≈ 4.5`), where the prediction has a genuine kink in `lag`/`dur`.
    #[test]
    fn ode_provider_zero_order_lagtime_matches_production() {
        let model = parse_model_string(ZERO_ORDER_ALAG_ODE).expect("parse");
        assert!(model.has_lagtime());
        let subj = bolus_subject(&[0.1, 0.2, 0.5, 1.0, 2.0, 6.0, 10.0]);
        assert!(
            !ode_subject_supported(&model, &subj),
            "lagtime routes off the static walk"
        );
        assert!(
            ode_tvcov_supported(&model, &subj),
            "zero_order()+lagtime: the event-driven walk now carries the moving window (#486)"
        );
        let theta = vec![5.0, 50.0, 4.0, 0.3];
        let eta = vec![0.1, 0.05];
        check_vs_production(&model, &subj, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subj, &theta, &eta);
        check_hessian_vs_fd_of_grad(&model, &subj, &theta, &eta);
    }

    const MIXED_ODE_TVCOV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVDUR(4.0, 0.05, 24.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V    = TVV
  FZO  = TVFZO
  FZO1 = 1 - TVFZO
  KA   = TVKA
  DUR  = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central
[covariates]
  WT continuous
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// `mixed` (FZO1·first_order + FZO·zero_order) absorption + a time-varying covariate
    /// (#653 review #1/#4). Two doses spaced closer than `DUR` overlap their zero-order
    /// windows, so at each window end the *other* dose's window — plus both doses' still-
    /// flowing first-order `R_in` — are concurrently active. The general rate-off saltation
    /// must fold every concurrently-active forcing into its curvature term once the WT-driven
    /// Jacobian jumps across the (non-record) window end; a forcing-only saltation would drop
    /// those terms (`coef2` several-percent off). This is also the first regression covering
    /// the advertised `mixed` pathway and its `frac` (FZO/FZO1) rate multiplier — both new
    /// zero_order tests above use plain `zero_order(dur)` with `frac == 1`.
    #[test]
    fn ode_provider_mixed_tvcov_matches_production() {
        let model = parse_model_string(MIXED_ODE_TVCOV).expect("parse");
        // Window ends ≈ 4.2 and 6.2 (`DUR = TVDUR·exp(ETA_DUR)`); windows [0, 4.2] and
        // [2, 6.2] overlap on [2, 4.2]. Obs avoid both window ends (genuine `DUR` kinks) and
        // the dose times, and straddle each window end so WT (hence CL, hence the Jacobian)
        // changes across it.
        let mut subject = bolus_subject(&[1.0, 3.0, 5.0, 8.0, 12.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0),
        ];
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        subject.dose_covariates = vec![wt(60.0), wt(62.0)];
        subject.obs_covariates = vec![wt(65.0), wt(70.0), wt(75.0), wt(80.0), wt(85.0)];
        assert!(subject.has_tv_covariates());
        assert!(
            ode_tvcov_supported(&model, &subject),
            "mixed()+TV-cov must be analytic on the event-driven walk (#486/#653)"
        );
        let theta = vec![5.0, 50.0, 0.4, 1.0, 4.0, 0.75];
        let eta = vec![0.1, 0.05];
        check_vs_production(&model, &subject, &theta, &eta);
        check_inner_outer_eta_parity(&model, &subject, &theta, &eta);
        // The window ends fall between records where WT (hence the RHS Jacobian) changes, so
        // the general saltation's covariate-jump curvature term — fed by the concurrent
        // forcings — is live; validate the 2nd-order blocks against FD of the analytic
        // gradient (the value/1st-order parity above does not exercise `coef2`).
        check_hessian_vs_fd_of_grad(&model, &subject, &theta, &eta);
    }
}
