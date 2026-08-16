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
//! (`TIME`/`TAD`) RHS**, and **IOV combined with FREM or LTBS**. (A steady-state dose into a
//! built-in absorption compartment is analytic since #835, including under IOV; only SS into a
//! `zero_order` window or SS + an absorption lagtime remain out of scope, and both are rejected
//! upstream rather than routed here.)
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
///
/// Raised 16 → 24 (#486). This const is the most load-bearing in `sens/`: besides
/// gating the ODE walk it also bounds `param_derivatives_at_cov`, which the
/// **closed-form** provider calls for its exact θ-chain — past the cap the CF path
/// silently fell back to `lognormal_param_derivatives`, whose `∂p/∂θ` is a central
/// *finite difference* of `tv_fn`, while still reporting "analytic". A 5-structural-θ
/// + 6-covariate-θ + 5-η model sits at exactly 16, i.e. right on the old edge.
pub(crate) const MAX_ODE_AXES: usize = 24;

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
    MAX_ODE_AXES == 24,
    "MAX_ODE_AXES changed: widen the disp!(1..=24) / dispatch_tv!(1..=24) tables in \
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

/// Monomorphised `(Dual1, Dual2)` widths for the ODE IOV dispatch ladder (#971).
///
/// Enumerating every width `1..=MAX_ODE_IOV_AXES` instantiated the entire
/// ODE-integration and sensitivity stack 96 times per worker — 62 % of the crate's LLVM
/// IR sat on widths `13..=96` alone, and ~93 % of the lib compile is LLVM (#969/#970). So the ladder is
/// **bucketed**: the runtime axis count is rounded up to the next width here and the extra
/// lanes are left zero (see [`crate::sens::widths`] for why padding is semantically inert —
/// every seeder guards its axis writes with `ax < N` and every readout indexes by the
/// runtime `n_theta` / `n_stacked`, never by `N`).
///
/// Padding is not free at *runtime*: the outer walk is `Dual2<M>` with an `M×M` Hessian per
/// value, so rounding `M` up costs about `(bucket/exact)²` (measured, #971 — the inner
/// `Dual1<N>` walk is `O(N)` and two orders of magnitude cheaper either way). The ladder is
/// therefore split where the two costs actually sit:
///
/// * **`1..=24` exact.** These widths are already instantiated by the `MAX_ODE_AXES` /
///   `MAX_SCALE_AXES = 24` ladders, so the IOV ladder shares their monomorphisations and
///   each extra width here costs only ~16 k LLVM lines (~0.2 % of the crate). Exactness is
///   nearly free, and this range covers the ordinary IOV model.
/// * **`> 24` bucketed.** Past the shared cap the IOV ladder is the sole user, at ~115 k
///   lines per width (~1.2 % of the crate each), so the tail rounds — with a step chosen to
///   hold the `O(M²)` padding penalty at ≤ ~1.5× rather than the ~2.1× a coarser
///   `32/48/64/96` ladder costs.
///
/// The ladder is a tuning parameter: widen it where padding cost bites, narrow it where
/// compile time does.
pub(crate) const ODE_IOV_WIDTH_BUCKETS: [usize; 32] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 32,
    40, 48, 56, 64, 80, 96,
];

// Tripwire, replacing the old `MAX_ODE_IOV_AXES == 96` literal check: the ladder must be
// strictly ascending and reach exactly the cap `ode_iov_subject_supported` enforces.
// Otherwise a subject inside the gate but past the last bucket would hit the ladder's
// `_ => None` arm and fall **silently** back to FD instead of being declined loudly by the
// gate (the #438 / #466 / #534 convention, carried over to the bucketed form).
const _: () = assert!(
    crate::sens::widths::buckets_well_formed(&ODE_IOV_WIDTH_BUCKETS, MAX_ODE_IOV_AXES),
    "ODE_IOV_WIDTH_BUCKETS must be strictly ascending and end at MAX_ODE_IOV_AXES: the \
     dispatch ladder has to cover the whole range ode_iov_subject_supported admits, or an \
     in-scope subject silently falls back to FD"
);

/// The bucketed const-generic ladder behind [`dispatch_ode_iov_axes`]. Both the runtime
/// bucket lookup and the `match` arms are driven by the same literal list, and the
/// `slices_eq` assert pins that list to [`ODE_IOV_WIDTH_BUCKETS`] — so editing the const
/// without editing the arms (or vice versa) fails to compile rather than dropping a bucket
/// into the `_ => None` FD arm.
macro_rules! dispatch_iov_widths {
    ([$($w:literal),+ $(,)?], $dim:expr, $worker:ident, $args:tt) => {{
        const _: () = assert!(
            crate::sens::widths::slices_eq(&[$($w),+], &ODE_IOV_WIDTH_BUCKETS),
            "dispatch_iov_widths! arms are out of sync with ODE_IOV_WIDTH_BUCKETS"
        );
        match crate::sens::widths::bucket_for($dim, &ODE_IOV_WIDTH_BUCKETS) {
            // The argument list arrives as one `tt` group, not a second repetition —
            // `$(…)+` cannot nest two repetitions of different lengths, so `call_at_width!`
            // re-parses it once per width arm.
            $(Some($w) => call_at_width!($worker::<$w>, $args),)+
            // Unreachable for `1..=MAX_ODE_IOV_AXES` (the assert above pins the ladder to
            // the cap); a wider subject is declined by `ode_iov_subject_supported` before
            // it reaches here. Kept as the belt-and-suspenders FD route.
            _ => None,
        }
    }};
}

/// Apply a width-instantiated worker to a parenthesised argument list.
macro_rules! call_at_width {
    ($f:expr, ($($arg:expr),* $(,)?)) => {
        $f($($arg),*)
    };
}

/// Dispatch `$worker::<W>` at the bucketed width `W ≥ $dim`, or `None` (→ FD) when `$dim`
/// is zero or past the last bucket.
macro_rules! dispatch_ode_iov_axes {
    ($dim:expr, $worker:ident, $($arg:expr),+ $(,)?) => {
        dispatch_iov_widths!(
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 28, 32, 40, 48, 56, 64, 80, 96
            ],
            $dim,
            $worker,
            ($($arg),+)
        )
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

/// The model-level scaling allowlist [`ode_analytical_supported`] requires: `None`
/// and a constant `ScalarScale` divisor are always analytic (handled per-observation
/// by [`apply_output_transform`]); an `ExpressionScale` divisor is analytic only when
/// [`expression_scale_axes_admissible`] (which also excludes it under LTBS); anything
/// else (e.g. `PerCmt`) declines. Allowlist, not denylist, so a future scaling variant
/// can only *narrow* the analytic scope, never silently admit an unhandled one.
///
/// Factored out (not just inlined in `ode_analytical_supported`) so
/// `pk::modified_release`'s closed-form MR gradient fast path (#860 Phase A6) — which
/// additionally applies [`apply_output_transform`] itself, since it does not go through
/// `run_subject`'s per-observation loop — can require the identical scaling scope
/// without a second, driftable copy of this match.
pub(crate) fn ode_scaling_supported(model: &CompiledModel) -> bool {
    match &model.scaling {
        ScalingSpec::None | ScalingSpec::ScalarScale(_) => true,
        ScalingSpec::ExpressionScale { deriv: Some(p), .. } => {
            expression_scale_axes_admissible(p, model)
        }
        _ => false,
    }
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
    // Per-route absorption lag (`fn(..., lag=L)`, #857): the analytic event-driven walk
    // (`integrate_tvcov_g`) now emits a per-route onset saltation — each route-lagged
    // forcing switches on at its own `t_dose + lag_cmt + lag_route`, injected as a rate-on
    // saltation at that (moving) boundary with the combined `∂/∂(lag_cmt + lag_route)` jet
    // (#859). The continuous `∂R_in/∂lag_route` already flows through
    // `add_prepared_input_rate_forcing` (which adds `route_lag` to `t_eff` over any `T`);
    // the walk adds only the missing onset-discontinuity term. Admitted per kernel, mirroring
    // the `has_lagtime` classifier below:
    //   • `first_order` — onset is the finite `ka·dose` jump the rate-on saltation handles.
    //   • `zero_order` — the route-lagged window shifts BOTH boundaries: its rate-on saltation
    //     at `t_dose + lag_cmt + lag_route` (`K_ROUTE_ONSET`) and its rate-off at `+ dur`
    //     (`K_ZO_END`) both carry the `lag_route` jet.
    //   • `transit` / `igd` — continuous onset (`R_in → 0` at the boundary), so no saltation:
    //     the continuous `∂R_in/∂lag_route` plus the timeline break at the onset suffice.
    //   • `weibull` — onset **diverges** for shape `β < 1` (an integrable spike, no finite
    //     rate-on saltation), so a route-lagged `weibull` stays on the FD fallback, exactly as
    //     `weibull` + a compartment lagtime does below.
    // IOV (`n_kappa != 0`) route lag is declined *here* by the `n_kappa != 0` clause below (an IOV
    // model is off the non-IOV analytic gate outright), but IS analytic on the IOV path since #877
    // — `ode_iov_supported` admits it via this same per-kernel `route_lag_analytic()` classifier.
    if ode
        .input_rate
        .iter()
        .any(|f| f.lag_slot.is_some() && !f.kind.route_lag_analytic())
    {
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
    if !ode_scaling_supported(model) {
        return false;
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

/// True when `subject` has an infusion (RATE>0) into a compartment fed by a built-in absorption
/// input-rate forcing (#719 gap 2). The f64 predictor serves this exactly — the dose is a
/// zero-order source feeding the kernel (`R_in_inf`), with its plain `+rate` injection suppressed
/// — but the dual sensitivity walk still injects that `+rate`, so it would double-count. The
/// analytic gates therefore decline these subjects to the FD fallback, which differences the
/// (exact, cheap) f64 prediction. An analytic infusion-into-kernel sensitivity is a follow-up
/// (`rate_infused` is already `Dual2`-differentiable; only the walk's `+rate` suppression and the
/// `F`-reshaped-window boundary jet remain).
pub(crate) fn has_infusion_into_input_rate(model: &CompiledModel, subject: &Subject) -> bool {
    let Some(ode) = model.ode_spec.as_ref() else {
        return false;
    };
    !ode.input_rate.is_empty()
        && subject
            .doses
            .iter()
            .any(|d| d.is_infusion() && ode.input_rate.iter().any(|f| f.cmt == d.cmt_idx()))
}

/// Per-subject scope gate, shared by the full (outer `Dual2`) and light (inner
/// `Dual1`) ODE providers so a subject is served analytically for **both** the
/// outer gradient and the inner EBE loop, or neither (the inner/outer scope must
/// match — a split would mix an analytic gradient with an FD Jacobian).
pub(crate) fn ode_subject_supported(model: &CompiledModel, subject: &Subject) -> bool {
    // Infusion into a built-in absorption compartment (#719 gap 2) → FD fallback (see
    // `has_infusion_into_input_rate`): the f64 prediction is exact, but the dual walk's `+rate`
    // injection would double-count the mass the convolved `R_in_inf` already delivers.
    if has_infusion_into_input_rate(model, subject) {
        return false;
    }
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
    // A per-route absorption lag (`fn(..., lag=L)`, #859) is the same kind of event-time
    // shift — the route's onset is a moving boundary carrying a rate-on saltation — so it
    // too routes to the event-driven walk, never the static one. (A *pure* route lag has
    // `has_lagtime() == false`, so this needs its own decline, else the subject would fall
    // through to the static walk which emits no onset saltation.)
    if model.has_lagtime() || model.has_route_absorption_lag() {
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

/// The per-route absorption lag (`zero_order(..., lag=L)`, #859) of the single zero-order
/// forcing feeding 0-based `cmt`, read from a PK snapshot, or `None` when that forcing is
/// unlagged or absent. One zero-order forcing per compartment (parser-enforced), so the match
/// is unambiguous. Shared by the event-driven walk's route-lagged window sites: the boundary
/// shift on the rate-off saltation (`K_ZO_END`) and — for symmetry — the rate-on onset builder.
fn zero_order_route_lag<T: crate::sens::num::PkNum>(
    ode: &OdeSpec,
    cmt: usize,
    params: &[T],
) -> Option<T> {
    ode.input_rate
        .iter()
        .find(|f| {
            f.kind == crate::pk::absorption::InputRateKind::ZeroOrder
                && f.cmt == cmt
                && f.lag_slot.is_some()
        })
        .map(|f| f.route_lag(params))
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
        // 0-based match so a `CMT=0` dose resolves compartment 1's zero-order window (#899, #913
        // review) — the gradient twin of the value-path fix in `zero_order_dur_and_frac_for_dose`.
        f.kind == crate::pk::absorption::InputRateKind::ZeroOrder && f.cmt == d.cmt_idx()
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
            .indexed_slot(crate::types::DoseAttr::Duration, d.cmt_raw())
            .map(|s| (crate::types::RateMode::ModeledDuration, s)),
        crate::types::RateMode::ModeledRate => attr_map
            .indexed_slot(crate::types::DoseAttr::Rate, d.cmt_raw())
            .map(|s| (crate::types::RateMode::ModeledRate, s)),
    }
}

pub(crate) fn ode_tvcov_supported(model: &CompiledModel, subject: &Subject) -> bool {
    // Infusion into a built-in absorption compartment (#719 gap 2) → FD fallback (the f64
    // prediction is exact; the dual walk would double-count the suppressed `+rate`). Same decline
    // as the static gate above — see `has_infusion_into_input_rate`.
    if has_infusion_into_input_rate(model, subject) {
        return false;
    }
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
    // A per-route absorption lag (`fn(..., lag=L)`, #859) makes each route's onset a moving
    // boundary (`t_dose + lag_cmt + lag_route`) with its own rate-on saltation — the same
    // event-time-shift family as a compartment lagtime — so it forces the event-driven walk
    // even when the subject has no TV covariates and no compartment lagtime.
    let has_route_lag = model.has_route_absorption_lag();
    if !ode_analytical_supported(model)
        || !(subject.has_tv_covariates()
            || model.has_lagtime()
            || has_route_lag
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
    // Steady-state dosing into a built-in absorption input-rate compartment is analytic (#835):
    // the dual walk equilibrates the trough through the closed-form fixed point
    // (`equilibrate_ss_input_rate_state_g`), carrying `∂u_ss/∂(θ,η)` through `u_ss = (I − M)⁻¹·b`
    // (linear) or the pulse-train fallback (nonlinear) — bit-identical in value to the f64
    // predictor, so the forward periodic `R_in` superposition composes on top exactly as on
    // production. SS into a `zero_order` window and SS + an absorption lagtime stay out of scope
    // and are hard-rejected upstream (`E_ABSORPTION_SS_ZERO_ORDER` / `E_ABSORPTION_SS_LAG`,
    // `api::check_absorption_dosing`); the belt-and-suspenders decline below keeps this gate
    // self-consistent with those scope limits if either upstream check is relaxed. Shared
    // single source of truth: `CompiledModel::ss_absorption_out_of_scope`.
    if model.ss_absorption_out_of_scope(subject) {
        return false;
    }
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
    // Per-route absorption lag (`fn(..., lag=L)`, #857) under IOV — admitted per kernel (#877),
    // mirroring the non-IOV gate's own route-lag classifier (`ode_tvcov_supported`). The shared
    // `integrate_tvcov_g` walk injects each route's onset saltation at its own
    // `t_dose + lag_cmt + lag_route` (`K_ROUTE_ONSET`); the κ-sensitivity rides through the
    // per-occasion `pk_at_dose[k]` jet exactly as η/θ do, and the onset-slope curvature term
    // (`½·∂Δr/∂tad`, #880/#883) makes the FOCEI Hessian exact for the decaying `first_order`
    // onset (whose δ² term is nonzero on any axis the lag carries — η or κ). So
    // `first_order`/`zero_order`/`transit`/`igd` route lags are analytic under IOV, validated
    // against central FD of `predict_iov` (`check_iov_provider_vs_fd`, value + gradient +
    // Hessian). Only `weibull` + route lag stays FD — its onset diverges for shape `β < 1` (an
    // integrable spike, no finite rate-on saltation), exactly as on the non-IOV path and as the
    // compartment-lagtime gate below declines `weibull`. `route_lag_analytic()` is the
    // exhaustive per-kind classifier (no `_` arm), shared with `ode_tvcov_supported` so the two
    // paths cannot drift. (SS × route lag stays FD regardless, declined per subject in
    // `ode_iov_subject_supported` via `ss_absorption_out_of_scope`'s `lag_slot` operand — the SS
    // dual seed does not carry the per-route onset.)
    if ode
        .input_rate
        .iter()
        .any(|f| f.lag_slot.is_some() && !f.kind.route_lag_analytic())
    {
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
        return dispatch_tv!(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        );
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
        return dispatch_tv!(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        );
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
    disp!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
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
///
/// Basis-agnostic: it only divides/logs the dual `p` as a whole, so it works
/// identically whether `p`'s axes are PK-parameter space (the static walk,
/// chained to `(θ,η)` afterward) or already `(θ,η)`-space (the TV-cov walk's
/// `seed_pk_dual2` convention) — already proven by [`resolve_obs_readout`]
/// calling it from both. `pub(crate)`: also reused by
/// `pk::modified_release::mr_sens_dual`/`mr_eta_grad_dual` (#860 Phase A6),
/// whose `mr_observable_g::<Dual2<M>>` result is `(θ,η)`-seeded the same way.
pub(crate) fn apply_output_transform<T: crate::sens::num::PkNum>(model: &CompiledModel, p: T) -> T {
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
        .map(|d| params_dual[f_bio_slot(ode, d.cmt_raw())])
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
            ..Default::default()
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
            ..Default::default()
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
///
/// `pub(crate)`: also reused by the closed-form MR gradient path
/// (`pk::modified_release::mr_subject_sensitivities`, #860 Phase A6), which
/// needs the identical `(θ,η)`-seeded PK-slot duals `mr_observable_g::<Dual2<M>>`
/// consumes — not a second copy of this seeding logic.
pub(crate) fn seed_pk_dual2<const M: usize>(
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
        .map(|(d, p)| p[f_bio_slot(ode, d.cmt_raw())])
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
            .map(|d| attr_map.lag_slot(d.cmt_raw()))
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
        out.push(crate::sens::provider::obs_sens_from_dual2::<M>(
            fd, n_theta, n_eta,
        ));
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
    // Infusion into a built-in absorption compartment (#719 gap 2) → FD fallback under IOV too:
    // the f64 prediction is exact, but the dual walk's `+rate` would double-count the mass the
    // convolved `R_in_inf` already delivers (see `has_infusion_into_input_rate`).
    if has_infusion_into_input_rate(model, subject) {
        return None;
    }
    // #835: a steady-state dose into a built-in absorption input-rate compartment is analytic
    // under IOV too — the shared `integrate_tvcov_g` walk equilibrates the trough via
    // `equilibrate_ss_input_rate_state_g`, whose fixed-point / pulse-train carries κ's jet through
    // `params` exactly as it does η/θ. Only SS into a `zero_order` window and SS + an absorption
    // lagtime stay out of scope; the decline below is the same belt-and-suspenders guard as the
    // non-IOV gate (`ode_tvcov_supported`), sharing `CompiledModel::ss_absorption_out_of_scope`.
    if model.ss_absorption_out_of_scope(subject) {
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
            ..Default::default()
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
    disp!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
}

/// Per-event flat PK-slot duals seeded on **η only** (`Dual1<N>`, `N = n_eta`) at a
/// covariate snapshot — the light-inner counterpart of [`seed_pk_dual2`]. Slot for
/// individual parameter `i` carries `∂p/∂η_k` on axis `k`; other slots are constant.
///
/// `pub(crate)`: also reused by `pk::modified_release::mr_subject_eta_grad`
/// (#860 Phase A6), the closed-form counterpart of `ode_subject_eta_grad`.
pub(crate) fn seed_pk_dual1<const N: usize>(
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
///   `u[cmt] += s·Δr·δlag`,  `u += (−s·½·J·(Δr·e_cmt) + ½·(∂Δr/∂tad)·e_cmt)·δlag²`,
/// matching the general `D·δlag + (½ẋ̇⁻+½ẋ̇⁺−J⁺ẋ⁻)·δlag²` with `D = s·Δr·e_cmt` and
/// `J⁻ = J⁺ = J` (state continuous). The δlag² coefficient is `½(v⁺−v⁻)`'s **full**
/// second time-derivative jump `½(ẋ̇⁺−ẋ̇⁻)`: the state-Jacobian part `J·(Δr·e_cmt)` (the
/// exact directional RHS derivative along the rate vector, via a `Dual1<1>` eval — no
/// finite differences, #439) **plus** the forcing's own explicit time-variation at the
/// boundary, `∂Δr/∂tad` (`dr_dtad`). For a **constant**-rate forcing (infusion,
/// zero-order window) `∂Δr/∂tad = 0` and this reduces to the old `−s·½·jg`; it is the
/// decaying `first_order`/Bateman onset (`∂Δr/∂tad = −Δr·ka ≠ 0`) whose omitted term
/// biased the analytic Hessian — the near sign-mirror-of-FD `d²f/∂η²` of #880. `dr_dtad`
/// is the exact onset slope from [`PreparedInputRate::rate_dtad_at_zero`], summed over
/// every forcing feeding `cmt` exactly as `dr` sums their onset values.
#[allow(clippy::too_many_arguments)]
fn inject_rate_saltation<T: crate::sens::num::PkNum>(
    u: &mut [T],
    cmt_idx: usize,
    dr: T,
    dr_dtad: T,
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
        // δlag² coefficient = −s·½·(J·(Δr·e_cmt))[c], plus — in the forced compartment
        // only — the forcing's explicit onset time-variation ½·∂Δr/∂tad (#880). The
        // latter enters `ẋ̇` on whichever side the forcing is active and is `s`-invariant
        // (rate-on: post-side ẋ̇⁺; rate-off: pre-side ẋ̇⁻), so it is added, not scaled by `s`.
        let mut coef2 = T::from_f64(-s * 0.5 * jg[c]);
        if c == cmt_idx {
            coef2 = coef2 + T::from_f64(0.5) * dr_dtad;
        }
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
/// [`crate::dosing::SS_EQUILIBRATION_CYCLES`] loop of `(apply dose; integrate II)`
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
    if dose.ii <= 0.0 {
        return u;
    }
    // `CMT=0` equilibrates compartment 1 — NONMEM's default dose compartment
    // (#899) — mirroring the f64 `equilibrate_ss_state`. The gradient twin must
    // move with the value path or FOCEI would differentiate a different dosing
    // history than it predicts.
    let cmt_idx = dose.cmt_idx();
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

    // ---- #914 exact solve: closed-form periodic fixed point (linear disposition). ----
    // The affine fixed point `u_ss = (I − M)⁻¹·b`, carried over the dual `T` so
    // `∂u_ss/∂(θ,η[,κ])` (and the 2nd order) fall out of the linear solve — the exact analytic
    // parity with the f64 `equilibrate_ss_state`, the input-rate twin (#835) and the analytical
    // walk (#908). `advance_forced` integrates one cycle *with* the dose; `advance_unforced`
    // integrates one `II` of disposition alone (the propagator `M`; the infusion's active and
    // quiet windows share the same homogeneous propagator, so a full-`II` decay reconstructs it
    // and the window length enters only through `b`). The moving active/quiet boundary carries
    // `inf_window`'s jet via the same `inject_rate_saltation` the fallback loop uses
    // (`dtinf = inf_window − t_inf_val`); its value part is zero, so the value map stays linear
    // while the window sensitivity rides `b`. A nonlinear RHS fails the self-check and falls to
    // the pulse train below (the f64 value walk owns the #867 warning — this twin must not
    // double-count it, matching `equilibrate_ss_input_rate_state_g`).
    let eq_opts = crate::ode::predictions::ss_equilibration_opts(opts);
    let d1v: RefCell<Vec<Dual1<1>>> = RefCell::new(Vec::new());
    let d1s: RefCell<Vec<Dual1<1>>> = RefCell::new(Vec::new());
    let advance_unforced = |u0: &[T]| -> Option<Vec<T>> {
        solve_ode_g(&bare_rhs, u0, (0.0, dose.ii), params, &[dose.ii], &eq_opts)
            .last()
            .map(|p| p.u.clone())
    };
    let advance_forced = |u0: &[T]| -> Option<Vec<T>> {
        match inf {
            Some((inf_rate, inf_window)) => {
                let t_inf_val = inf_window.val();
                if t_inf_val > dose.ii {
                    return None; // overlapping pulses — no simple equilibration (→ fallback → zero)
                }
                let rate_forcing = inf_rate;
                let rhs_active = |us: &[T], ps: &[T], t: f64, du: &mut [T]| {
                    bare_rhs(us, ps, t, du);
                    if cmt_idx < du.len() {
                        du[cmt_idx] = du[cmt_idx] + rate_forcing;
                    }
                };
                let mut y = solve_ode_g(
                    &rhs_active,
                    u0,
                    (0.0, t_inf_val),
                    params,
                    &[t_inf_val],
                    &eq_opts,
                )
                .last()
                .map(|p| p.u.clone())?;
                // Boundary-shift saltation for the (possibly jet-carrying) window `inf_window`,
                // identical to the fallback loop's per-cycle `inject_rate_saltation`.
                let dtinf = inf_window - T::from_f64(t_inf_val);
                inject_rate_saltation::<T>(
                    &mut y,
                    cmt_idx,
                    rate_forcing,
                    T::from_f64(0.0),
                    dtinf,
                    1.0,
                    program,
                    params,
                    t_inf_val,
                    0.0,
                    0.0,
                    &mut d1v.borrow_mut(),
                    &mut d1s.borrow_mut(),
                );
                let quiet_val = dose.ii - t_inf_val;
                if quiet_val > 0.0 {
                    y = solve_ode_g(
                        &bare_rhs,
                        &y,
                        (0.0, quiet_val),
                        params,
                        &[quiet_val],
                        &eq_opts,
                    )
                    .last()
                    .map(|p| p.u.clone())?;
                }
                Some(y)
            }
            None => {
                let mut y = u0.to_vec();
                y[cmt_idx] = y[cmt_idx] + f_bio * T::from_f64(dose.amt);
                solve_ode_g(&bare_rhs, &y, (0.0, dose.ii), params, &[dose.ii], &eq_opts)
                    .last()
                    .map(|p| p.u.clone())
            }
        }
    };
    if let Some(u_ss) = crate::dosing::periodic_ss_fixed_point_g::<T, _, _>(
        n_states,
        dose.ii,
        eq_opts.reltol,
        eq_opts.abstol,
        advance_unforced,
        advance_forced,
    ) {
        crate::dosing::record_ss_equilibration_cycles(1);
        return u_ss;
    }

    // ---- Fallback: explicit pulse-train iteration (nonlinear disposition / singular `I − M`). ----
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
        for cycle in 0..crate::dosing::SS_EQUILIBRATION_CYCLES {
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
                // Constant infusion rate ⇒ no onset time-variation (#880).
                T::from_f64(0.0),
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
        crate::dosing::record_ss_equilibration_cycles(cycles_run);
        return u;
    }
    // Bolus SS: each cycle applies the pulse `F·amt`, then decays over one interval.
    let amt = T::from_f64(dose.amt);
    let saveat = [dose.ii];
    let mut prev = vec![0.0_f64; n_states];
    let mut cur = vec![0.0_f64; n_states];
    let mut cycles_run = 0usize;
    for cycle in 0..crate::dosing::SS_EQUILIBRATION_CYCLES {
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
    crate::dosing::record_ss_equilibration_cycles(cycles_run);
    u
}

/// Dual steady-state trough for an `SS=1` **bolus** dose into a built-in absorption input-rate
/// compartment (#835) — the analytic-sensitivity counterpart of production's
/// `equilibrate_ss_state` input-rate branch, and the input-rate sibling of
/// [`equilibrate_ss_state_g`] (which serves the bolus/infusion-into-a-plain-compartment case).
///
/// The dose does not enter as an instantaneous bolus; it drives the compartment through the
/// absorption kernel `R_in(tad)` (transit/igd/weibull/first_order). On a **linear** disposition
/// the periodic steady state is the closed-form fixed point `u_ss = (I − M)⁻¹·b`, carried over `T`
/// by [`periodic_ss_fixed_point_g`](crate::dosing::periodic_ss_fixed_point_g)
/// so `∂u_ss/∂(θ,η[,κ])` (and the 2nd order) fall out of the linear solve — exact analytic parity
/// with production's fast path, and bit-identical in value. A **nonlinear** disposition fails the
/// fixed point's self-check, so [`equilibrate_ss_input_rate_g`](crate::ode::predictions::equilibrate_ss_input_rate_g)
/// solves the same `u = P(u)` by Anderson acceleration (#867), carrying the jets through its
/// recursion + the dual Newton derivative correction. Only when *that* also declines — `ρ ≥ 1`, no
/// periodic steady state — does this fall back to the explicit pulse-train iteration, carrying the
/// same jets through its finite loop.
///
/// Either way the returned trough is the pre-pulse SS carryover; the forward walk's
/// `add_prepared_input_rate_forcing` superposes the current + prior pulses' still-arriving tails
/// on top (disjoint from this trough), exactly as on the f64 path. Bolus-record SS only — SS
/// infusion into absorption is #719 gap-2 out of scope (`has_infusion_into_input_rate` gates it to
/// FD before the walk), and the caller's `!is_inf(d)` guard mirrors that. `dose.ii > 0` and
/// `dose.cmt` valid are caller-guaranteed; a stray out-of-range dose returns the zero trough.
fn equilibrate_ss_input_rate_state_g<T: crate::sens::num::PkNum>(
    program: &crate::parser::model_parser::OdeRhsProgram,
    ode: &OdeSpec,
    n_states: usize,
    dose: &crate::types::DoseEvent,
    f_bio: T,
    params: &[T],
    opts: &crate::ode::solver::OdeSolverOptions,
) -> Vec<T> {
    let n = n_states;
    let ii = dose.ii;
    // `CMT=0` equilibrates compartment 1 — the default dose compartment — like the f64 twin
    // `equilibrate_ss_input_rate` and the plain-disposition `equilibrate_ss_state_g` (#899). This
    // previously bailed on `dose.cmt == 0` and returned the zero (unequilibrated) trough, so an
    // `SS=1` bolus written `CMT=0` into a built-in-absorption compartment was *differentiated* as a
    // single unaccumulated dose while production equilibrated it — a silent value≠gradient FOCEI
    // error (#913 review). The body resolves the forcing via `d.cmt.saturating_sub(1)`, so `cmt=0`
    // equilibrates identically to `cmt=1` once the early bail is gone.
    if ii <= 0.0 || dose.cmt_idx() >= n {
        return vec![T::from_f64(0.0); n];
    }
    // Built-in absorption forcings prepared from THIS SS dose's snapshot (`params`), mirroring the
    // f64 `prepare_input_rates`. The gate's `supported_over_dual()` allowlist guarantees success.
    let prepared: Vec<PreparedInputRate<T>> = ode
        .input_rate
        .iter()
        .map(|f| {
            f.prepare_dual::<T>(params)
                .expect("gate's supported_over_dual() allowlist guarantees prepare_dual succeeds")
        })
        .collect();

    let vars_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());
    let stack_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());
    // Disposition alone, cycle-relative time (anchor 0). An SS-admissible RHS is autonomous
    // (a time/TAD-reading RHS is gated to FD upstream), so the anchor is immaterial — matching
    // `equilibrate_ss_state_g`'s `eval_rhs_anchored(.., 0.0, 0.0, ..)`.
    let disposition = |us: &[T], ps: &[T], t: f64, du: &mut [T]| {
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

    // ---- Fast path: closed-form fixed point (linear disposition). ----
    // A single local SS pulse at t = 0 drives the periodic `R_in` (its SS branch superposes the
    // infinite-past pulse tails) — the dual mirror of the f64 `wrap_rhs_with_forcings(local_ss)`.
    let local_ss = [crate::types::DoseEvent::new(
        0.0,
        dose.amt,
        dose.cmt_raw(),
        0.0,
        true,
        ii,
    )];
    let lag0 = [T::from_f64(0.0)];
    let fbio1 = [f_bio];
    let forced = |us: &[T], ps: &[T], t: f64, du: &mut [T]| {
        disposition(us, ps, t, du);
        crate::ode::predictions::add_prepared_input_rate_forcing::<T>(
            ode,
            &prepared,
            ps,
            &local_ss,
            &lag0,
            &fbio1,
            f64::NEG_INFINITY,
            t,
            du,
        );
    };
    // Tightened equilibration tolerance so the fixed-point trough — and its dual jets — stay
    // accurate as `ρ → 1` (mirrors the f64 `equilibrate_ss_input_rate`; #867). Bit-identical to the
    // f64 path because both floor the same way.
    let eq_opts = crate::ode::predictions::ss_equilibration_opts(opts);
    let advance = |rhs: &dyn Fn(&[T], &[T], f64, &mut [T]), u0: &[T]| -> Option<Vec<T>> {
        solve_ode_g(rhs, u0, (0.0, ii), params, &[ii], &eq_opts)
            .last()
            .map(|p| p.u.clone())
    };
    if let Some(u_ss) = crate::ode::predictions::equilibrate_ss_input_rate_g::<T, _, _>(
        n,
        ii,
        eq_opts.reltol,
        eq_opts.abstol,
        |u0| advance(&disposition, u0),
        |u0| advance(&forced, u0),
    ) {
        return u_ss;
    }

    // ---- Fallback: explicit pulse-train iteration (no periodic steady state, ρ ≥ 1). ----
    // Reached only when both the linear closed form and the Anderson solve declined (a nonlinear
    // disposition *with* a steady state is solved by Anderson above, not here).
    // Lay out `SS_EQUILIBRATION_CYCLES` past **non-SS** pulses at 0, II, 2II, … and integrate
    // segment-by-segment from a zero state; `R_in` is re-evaluated by absolute pulse age so an
    // absorption tail longer than `II` keeps contributing across cycle boundaries (mirrors the
    // f64 `equilibrate_ss_state` input-rate fallback). The dual jets ride the finite loop.
    let n_pulses = crate::dosing::SS_EQUILIBRATION_CYCLES;
    let local_doses: Vec<crate::types::DoseEvent> = (0..n_pulses)
        .map(|m| {
            crate::types::DoseEvent::new(m as f64 * ii, dose.amt, dose.cmt_raw(), 0.0, false, 0.0)
        })
        .collect();
    let lags = vec![T::from_f64(0.0); n_pulses];
    let fbios = vec![f_bio; n_pulses];
    let train = |us: &[T], ps: &[T], t: f64, du: &mut [T]| {
        disposition(us, ps, t, du);
        crate::ode::predictions::add_prepared_input_rate_forcing::<T>(
            ode,
            &prepared,
            ps,
            &local_doses,
            &lags,
            &fbios,
            f64::NEG_INFINITY,
            t,
            du,
        );
    };
    let mut u = vec![T::from_f64(0.0); n];
    let mut prev = vec![0.0_f64; n];
    let mut cur = vec![0.0_f64; n];
    let mut cycles_run = 0usize;
    for m in 0..n_pulses {
        let seg_start = m as f64 * ii;
        let seg_end = seg_start + ii;
        let sol = solve_ode_g(&train, &u, (seg_start, seg_end), params, &[seg_end], opts);
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
        cycles_run = m + 1;
        if crate::sens::propagate::ss_dual_cycle_should_stop(m, &u, &mut cur, &mut prev) {
            break;
        }
    }
    crate::dosing::record_ss_equilibration_cycles(cycles_run);
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
    // CMT is 1-based, and `CMT=0` is NONMEM's *default dose compartment* — state index 0 —
    // not a malformed value. `saturating_sub(1)` is therefore the correct mapping, matching
    // `equilibrate_ss_state_g` and the f64 path (#899). This previously no-op'd on the
    // premise that aliasing to index 0 would "silently dose compartment 0" (#642 review #1);
    // #375 settled the opposite convention on the analytical engine, and #899 brought the ODE
    // engine into line, so the no-op was the silent drop rather than the guard against one.
    let cmt_idx = dose.cmt_idx();
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
            // Constant infusion rate ⇒ no onset time-variation (#880).
            inject_rate_saltation::<T>(
                &mut u,
                cmt_idx,
                inf_rate,
                T::from_f64(0.0),
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
    // #859: does any built-in forcing carry a per-route lag (`fn(..., lag=L)`)? Route-lagged
    // forcings switch on at their own `t_dose + lag_cmt + lag_route`, handled by the
    // `K_ROUTE_ONSET` timeline events below rather than the shared `K_DOSE` onset. `false`
    // (and byte-identical to the pre-#859 walk) for the common no-route-lag case. Computed
    // from `ode` (this walk has no `&CompiledModel`); equals `model.has_route_absorption_lag()`.
    let has_route_lag = ode.has_route_lag();
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
                // #859: a route-lagged zero-order window (`zero_order(..., lag=L)`) opens at
                // `t_dose + lag_cmt + lag_route` — its whole `[w_start, w_end]` slides by the
                // per-route lag (`0` for an unlagged window, byte-identical to the pre-#859
                // walk). The rate-on saltation fires at this shifted `w_start` (`K_ROUTE_ONSET`),
                // the rate-off at the shifted `w_end` (`K_ZO_END`), both carrying its jet.
                let w_start = d.time + lag_val(k) + f.route_lag(&pk_at_dose[k]).val();
                let w_end = w_start + dur.val();
                Some((d.cmt_idx(), rate, w_start, w_end, dur, k))
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

    // Merged timeline: (time, kind, idx), kind ∈ {Reset=0, SsSeed=1, Dose=2, RouteOnset=3,
    // PkOnly=4, Obs=5, InfEnd=6, ZoEnd=7} — the sort key matching production's `kind_order`
    // (Reset before a co-timed Dose so an EVID=4 reset+dose zeros the state before its own dose
    // lands; a per-dose SS pre-arrival seed — see below — before its own later Dose event; Dose
    // before a per-route absorption onset (#859) before PkOnly before Obs; infusion-end and
    // zero-order window-end last so an obs at the end reads the rate still contributing). Doses
    // (and infusion windows) sit at their lagged arrival `d.time + lag_val(k)`; resets and
    // pk-only records are at their record time (fixed, not lag-shifted).
    const K_RESET: u8 = 0;
    const K_SS_SEED: u8 = 1;
    const K_DOSE: u8 = 2;
    // #859: per-route absorption onset (`fn(..., lag=L)`). A route-lagged forcing switches on
    // at `t_dose + lag_cmt + lag_route`, PAST the dose's own `K_DOSE` arrival. Sorts right
    // after `K_DOSE` and before the record/obs events (value 3), so — like a dose arrival —
    // its rate-on saltation lands before any observation at the same instant reads the state.
    const K_ROUTE_ONSET: u8 = 3;
    const K_PKONLY: u8 = 4;
    const K_OBS: u8 = 5;
    const K_INF_END: u8 = 6;
    // #486: zero-order absorption window end. Sorts after `K_OBS` (like `K_INF_END`) so an
    // observation exactly at the window end reads the constant rate still on, matching the
    // static walk and production's `active_zero_order_inputs` full-containment convention.
    const K_ZO_END: u8 = 7;
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
    // #859: per-route absorption onset events. Each forcing carrying its own `lag=`
    // (`lag_slot`) switches on at `d.time + lag_cmt + lag_route`, past the dose's `K_DOSE`
    // arrival. Collect `(dose_idx, forcing_idx)` for every (route-lagged forcing × dose
    // feeding its compartment) — the `K_ROUTE_ONSET` handler injects that one forcing's
    // rate-on saltation at its own onset. Empty (and byte-identical to the pre-#859 walk)
    // when no forcing carries a `lag_slot` — the common case. Filter mirrors the f64
    // `push_route_lag_break_times` (`amt > 0`, compartment match) so the break time matches.
    let route_onsets: Vec<(usize, usize)> = if has_route_lag {
        let mut v = Vec::new();
        for (fi, f) in ode.input_rate.iter().enumerate() {
            if f.lag_slot.is_none() {
                continue;
            }
            for (k, d) in subject.doses.iter().enumerate() {
                if d.amt > 0.0 && d.cmt_idx() == f.cmt {
                    v.push((k, fi));
                }
            }
        }
        v
    } else {
        Vec::new()
    };
    let mut tl: Vec<(f64, u8, usize)> = Vec::with_capacity(
        subject.doses.len()
            + n_obs
            + subject.pk_only_times.len()
            + subject.reset_times.len()
            + n_infusion_ends
            + n_ss_seeds
            + zero_windows.len()
            + route_onsets.len(),
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
    // #859: per-route onset break at `d.time + lag_cmt + lag_route` (value part — the solver
    // integrates the f64 timeline; the onset's `∂/∂(lag_cmt + lag_route)` sensitivity is the
    // saltation injected at the `K_ROUTE_ONSET` handler). Payload is the `route_onsets` index,
    // not a dose index. The break time equals the f64 predictor's `push_route_lag_break_times`
    // value, so the analytic value path stays bit-identical to the FD reference.
    for (ri, &(k, fi)) in route_onsets.iter().enumerate() {
        let route_lag = ode.input_rate[fi].route_lag(&pk_at_dose[k]).val();
        tl.push((
            subject.doses[k].time + lag_val(k) + route_lag,
            K_ROUTE_ONSET,
            ri,
        ));
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
                // `d.cmt < 1`: an infusion with `CMT=0` has no target and is rejected up
                // front on both engines (#375 / #899) — see the `filter` in the saltation
                // walk below. Unreachable from a validated call.
                if !is_inf(d) || d.cmt_raw() < 1 {
                    continue;
                }
                let iws = d.time + lag_val(k);
                let iwe = iws + inf_window_len(k);
                let ci = d.cmt_idx();
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
                    // `d.cmt >= 1`: an infusion with `CMT=0` has no target — the default dose
                    // compartment is defined for a bolus but not for a zero-order input, so
                    // both engines reject it up front (analytical since #375, ODE since #899;
                    // the datareader does *not* reject it, contrary to #472 #6 / #473 #3 —
                    // only a *missing* CMT column defaults to 1). Unreachable from a validated
                    // call, kept for hand-built specs that run no validation.
                    .filter(|(_, d)| is_inf(d) && d.cmt_raw() >= 1)
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
                    .map(|(k, d)| (d.cmt_idx(), inf_eff[k].0))
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
                    // SS into a built-in absorption compartment (#835): the dose drives the
                    // kernel `R_in`, not an instantaneous bolus, so equilibrate through the dual
                    // fixed point (linear) / pulse-train (nonlinear), carrying `∂u_ss/∂(θ,η[,κ])`.
                    // `input_rate_consumes_cmt` is the same predicate the forward bolus-skip below
                    // uses; `!is_inf(d)` mirrors the upstream FD gate on SS infusion into
                    // absorption (#719 gap-2), so this arm is bolus-record only. The cache is
                    // populated only by the `K_SS_SEED` (SS+lagtime) branch, which is out of scope
                    // here (rejected upstream), so a non-lagged SS-absorption dose always lands in
                    // one of these `None` arms.
                    None if has_input_rate
                        && input_rate_consumes_cmt(ode, d.cmt_raw())
                        && !is_inf(d) =>
                    {
                        equilibrate_ss_input_rate_state_g::<T>(
                            program,
                            ode,
                            n_states,
                            d,
                            f_bio_at_dose[idx],
                            &pk_at_dose[idx],
                            opts,
                        )
                    }
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
            // CMT is 1-based, and `CMT=0` is NONMEM's *default dose compartment*, which
            // resolves to state index 0 on both engines (#899). The `d.cmt >= 1` skip this
            // replaces rested on the datareader rejecting `CMT=0` upstream (#449 review #8);
            // it does not — a missing `CMT` column defaults to 1, but an explicitly written
            // `CMT=0` reaches here unchanged, and was silently dropped from the gradient.
            {
                let cmt_idx = d.cmt_idx();
                if cmt_idx < n_states {
                    if has_input_rate && input_rate_consumes_cmt(ode, d.cmt_raw()) {
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
                            // Onset-segment snapshot (#880). Production's `R_in` turns on in the
                            // segment STARTING at the lagged arrival, integrated (NONMEM
                            // end-of-interval) with the NEXT record's PK snapshot — not the dose
                            // record's. So the onset jump's kernel (`ka`/shape, via `prep`) and
                            // pathway `frac` must be read from that post-arrival snapshot, exactly
                            // as `add_prepared_input_rate_forcing` reads them for the continuous
                            // forcing on that segment; under a TV covariate crossing the onset the
                            // dose snapshot diverges (a several-percent gradient error). The dose
                            // **mass** `F·amt` stays fixed at dose time (`f_bio_at_dose`,
                            // mass-exact), matching production. Skip co-located rate-off siblings
                            // (no record params); fall back to the dose snapshot if no later
                            // record exists (the onset then feeds no observed segment anyway).
                            let onset_params: &[T] = 'onset_snap: {
                                let leps = crate::ode::predictions::INFUSION_EPS;
                                for q in (p + 1)..tl.len() {
                                    let (tq, kq, iq) = tl[q];
                                    if (kq == K_INF_END || kq == K_ZO_END)
                                        && (tq - t_event).abs() <= leps
                                    {
                                        continue;
                                    }
                                    break 'onset_snap match kq {
                                        K_DOSE | K_SS_SEED => &pk_at_dose[iq],
                                        K_PKONLY => &pk_at_pk_only[iq],
                                        K_OBS => &pk_at_obs[iq],
                                        _ => &pk_at_dose[idx],
                                    };
                                }
                                &pk_at_dose[idx]
                            };
                            let prep_onset = prep_for(onset_params);
                            let mut onset = T::from_f64(0.0);
                            // Onset **slope** `∂Δr/∂tad` (#880), summed over the same forcings
                            // exactly as `onset` sums their values — the curvature companion the
                            // rate-on saltation's δlag² term needs (`½·∂Δr/∂tad`). Zero for the
                            // vanishing-onset kernels (`transit n>0`, IG) and for the constant
                            // zero-order window below, non-zero for the decaying `first_order`
                            // /Bateman onset that biased the Hessian.
                            let mut onset_dtad = T::from_f64(0.0);
                            for (f, prep) in ode.input_rate.iter().zip(&prep_onset) {
                                // #859: a forcing carrying its own `lag=` switches on later, at
                                // `t_dose + lag_cmt + lag_route` — its onset saltation is injected
                                // at its `K_ROUTE_ONSET` event with the combined jet, not summed
                                // into this shared `t_dose + lag_cmt` onset. Skip it here (an
                                // unlagged forcing has `lag_slot == None` — the common case, byte-
                                // identical to the pre-#859 walk).
                                if f.cmt == cmt_idx && f.lag_slot.is_none() {
                                    onset =
                                        onset + f.frac(onset_params) * prep.rate_at_zero(dose_mass);
                                    onset_dtad = onset_dtad
                                        + f.frac(onset_params) * prep.rate_dtad_at_zero(dose_mass);
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
                                    // #859: a route-lagged zero-order window opens later, at its
                                    // own `K_ROUTE_ONSET` (`w_start = t_dose + lag_cmt + lag_route`),
                                    // not at this `K_DOSE` arrival — its rate-on fires there. Skip
                                    // it here (unlagged windows, the common case, still fire here).
                                    let route_lagged = ode.input_rate.iter().any(|f| {
                                        f.kind == crate::pk::absorption::InputRateKind::ZeroOrder
                                            && f.cmt == zcmt
                                            && f.lag_slot.is_some()
                                    });
                                    if !route_lagged {
                                        onset = onset + zrate;
                                    }
                                }
                            }
                            inject_rate_saltation::<T>(
                                &mut u,
                                cmt_idx,
                                onset,
                                onset_dtad,
                                dlag,
                                -1.0,
                                program,
                                // Post-side Jacobian uses the onset-segment snapshot (#880),
                                // matching where the forcing actually turns on.
                                onset_params,
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
                                // Constant infusion rate ⇒ no onset time-variation (#880).
                                T::from_f64(0.0),
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
        } else if kind == K_ROUTE_ONSET {
            // #859: a route-lagged forcing (`fn(..., lag=L)`) switches on here, at
            // `t_dose + lag_cmt + lag_route` — its onset is a discontinuity in `du/dt` (the
            // forcing jumps from inactive to `R_in(0⁺)`), a moving boundary in BOTH lags.
            // Inject that single forcing's rate-on saltation with the combined jet
            // `∂/∂(lag_cmt + lag_route)` — the exact analogue of the shared-onset injection at
            // `K_DOSE` (which now skips route-lagged forcings). Fires with or without a
            // compartment lagtime: a *pure* route lag has `has_lagtime == false`, so `lag_cmt`
            // is zero and this is the ONLY onset term. The magnitude is this forcing's own
            // `frac·R_in(0⁺)` (`rate_at_zero`), not the summed onset — a finite `ka·dose` for
            // `first_order`; zero for the smooth kernels (`transit`/`igd`, whose continuous
            // onset needs only the timeline break). A `zero_order` route-lagged window adds its
            // constant window rate here as the rate-on (its `rate_at_zero` is zero), with the
            // matching rate-off carrying the same `lag_route` jet at `K_ZO_END` (#859 Slice 2).
            let (dose_idx, fi) = route_onsets[idx];
            let d = &subject.doses[dose_idx];
            // `saturating_sub`: `CMT=0` is the default dose compartment, state index 0 (#899).
            // The former `d.cmt >= 1` gate dropped this onset jet for such a dose, which is a
            // *wrong gradient* rather than a visibly zero value.
            let cmt_idx = d.cmt_idx();
            if cmt_idx < n_states {
                let f = &ode.input_rate[fi];
                // #880: read the onset kernel (`ka`/shape via `prep`), pathway `frac`, and the
                // post-side Jacobian from the POST-ARRIVAL segment snapshot — the record ending
                // the segment where this route's `R_in` turns on (NONMEM end-of-interval) — not
                // the pre-onset `last_params` (which `prepared_forcings` here is built from) nor
                // the dose record's snapshot. Under a TV covariate crossing the route onset those
                // diverge (a several-percent gradient error), exactly as at the shared `K_DOSE`
                // onset. The dose **mass** `F·amt` stays fixed at dose time (mass-exact). Skip
                // co-located rate-off siblings; fall back to the dose snapshot if no later record.
                let onset_params: &[T] = 'route_snap: {
                    let leps = crate::ode::predictions::INFUSION_EPS;
                    for q in (p + 1)..tl.len() {
                        let (tq, kq, iq) = tl[q];
                        if (kq == K_INF_END || kq == K_ZO_END) && (tq - t_event).abs() <= leps {
                            continue;
                        }
                        break 'route_snap match kq {
                            K_DOSE | K_SS_SEED => &pk_at_dose[iq],
                            K_PKONLY => &pk_at_pk_only[iq],
                            K_OBS => &pk_at_obs[iq],
                            _ => &pk_at_dose[dose_idx],
                        };
                    }
                    &pk_at_dose[dose_idx]
                };
                let prep_onset = prep_for(onset_params);
                let prep = &prep_onset[fi];
                let lag_cmt = if has_lagtime {
                    pk_at_dose[dose_idx][dose_lag_slot[dose_idx]]
                } else {
                    T::from_f64(0.0)
                };
                let dlag = jet_only(lag_cmt + f.route_lag(&pk_at_dose[dose_idx]));
                let dose_mass = f_bio_at_dose[dose_idx] * T::from_f64(d.amt);
                let mut onset = f.frac(onset_params) * prep.rate_at_zero(dose_mass);
                // #880: onset **slope** `∂Δr/∂tad` — the δlag² curvature companion the rate-on
                // saltation needs (`−dose·ka²` for `first_order`; 0 for the smooth kernels and
                // for the constant `zero_order` window rate below).
                let onset_dtad = f.frac(onset_params) * prep.rate_dtad_at_zero(dose_mass);
                // #859 Slice 2: a route-lagged `zero_order` window's rate-on is its constant
                // window rate (`rate_at_zero` is zero for `ZeroOrder`, so the term above is zero).
                // One zero-order forcing per compartment, so match this dose's own window here.
                if f.kind == crate::pk::absorption::InputRateKind::ZeroOrder {
                    for &(zcmt, zrate, _, _, _, zk) in &zero_windows {
                        if zk == dose_idx && zcmt == cmt_idx {
                            onset = onset + zrate;
                        }
                    }
                }
                // TAD anchor for the saltation's `J·(Δr·e_cmt)` term = the most recent dose's
                // (lagged) arrival (`last_dose_eff`), so a TAD-referencing RHS Jacobian sees the
                // same `TAD` it uses throughout the segment ending here. Fall back to `t_event`
                // before any dose has anchored TAD (mirrors the `K_DOSE` bolus-saltation guard).
                let anchor = if last_dose_eff.is_finite() {
                    last_dose_eff
                } else {
                    t_event
                };
                inject_rate_saltation::<T>(
                    &mut u,
                    cmt_idx,
                    onset,
                    onset_dtad,
                    dlag,
                    -1.0,
                    program,
                    // Post-side Jacobian from the onset-segment snapshot (#880).
                    onset_params,
                    t_event,
                    first_dose_time,
                    anchor,
                    &mut d1_vars,
                    &mut d1_stack,
                );
            }
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
            // `saturating_sub`: `CMT=0` is the default dose compartment, state index 0 (#899).
            // The former `d.cmt >= 1` gate dropped this saltation term for such a dose — again
            // a wrong gradient rather than a visibly zero value.
            if (has_lagtime || is_rate_defined || is_modeled)
                && d.time + lag_val(idx) >= reset_floor
                && d.cmt_idx() < n_states
            {
                let cmt = d.cmt_idx();
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
                        // Constant infusion rate ⇒ no onset time-variation (#880).
                        T::from_f64(0.0),
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
                    // Shared shift `δ = δlag + δdur (+ δlag_route)` (co-terminating ⇒ same
                    // `t + lag + lag_route + dur` ⇒ same jet in the common case). Use the
                    // representative (first) cohort window.
                    let (cmt_rep, _, _, _, dur_rep, k_rep) = zero_windows[cohort[0]];
                    let dlag = if has_lagtime {
                        jet_only(pk_at_dose[k_rep][dose_lag_slot[k_rep]])
                    } else {
                        T::from_f64(0.0)
                    };
                    // #859 Slice 2: a route-lagged window's end `w_end = t + lag_cmt + lag_route
                    // + dur` also moves with `lag_route`, so its jet joins the rate-off shift.
                    let route_lag_rep = zero_order_route_lag(ode, cmt_rep, &pk_at_dose[k_rep])
                        .map_or_else(|| T::from_f64(0.0), jet_only);
                    let d_off = dlag + jet_only(dur_rep) + route_lag_rep;
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
                            // #859 Slice 2: this window's own route-lag jet (its end shifts with
                            // `lag_route` too); `0` for an unlagged window (the common case).
                            let route_lag_j = zero_order_route_lag(ode, cmt, &pk_at_dose[k_j])
                                .map_or_else(|| T::from_f64(0.0), jet_only);
                            let d_off_j = dlag_j + jet_only(dur_j) + route_lag_j;
                            inject_rate_saltation::<T>(
                                &mut u,
                                cmt,
                                rate,
                                // Constant zero-order rate ⇒ no onset time-variation (#880).
                                T::from_f64(0.0),
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
                Some((d.cmt_idx(), rate, d.time, dur))
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

        // Apply bolus doses (non-infusions) at t_start: u[cmt] += F·amt. CMT is 1-based, and
        // `CMT=0` is NONMEM's default dose compartment — state index 0 — on both engines
        // (#899). The `dose.cmt >= 1` skip this replaces rested on the datareader rejecting
        // `CMT=0` upstream (#449 #8); it does not, so the analytic gradient saw an *undosed*
        // subject while production dosed it — `f = 0` against a production `PRED` of ~7.8 on
        // the 2-cpt parity fixture. A compartment fed by a built-in absorption input rate is
        // skipped here — the dose feeds R_in (the forcing in the RHS below), not a bolus
        // (#430). `F` is per dose compartment via `dose_f_bio[k]` (#486).
        for (k, dose) in subject.doses.iter().enumerate() {
            if !dose.is_infusion()
                && (dose.time - t_start).abs() < 1e-12
                && !input_rate_consumes_cmt(ode, dose.cmt_raw())
            {
                let cmt_idx = dose.cmt_idx();
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
                        // Constant zero-order rate ⇒ no onset time-variation (#880).
                        T::from_f64(0.0),
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
                .filter(|(_, d)| d.is_infusion() && d.cmt_raw() >= 1)
                .filter(|(_, d)| {
                    infusion_spans_segment(d.time, d.duration, t_start, t_end, reset_floor)
                })
                .map(|(k, d)| (d.cmt_idx(), dose_f_bio[k] * T::from_f64(d.rate)))
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
#[path = "ode_provider_tests.rs"]
mod tests;
