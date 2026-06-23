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
//! #439) readout; **bolus and infusion** doses; **bioavailability F** (incl.
//! estimated, any parameterization — log-normal, logit-normal, additive); **EVID 3/4
//! resets / multi-occasion**; **non-zero `init(...)` initial conditions**; static
//! covariates; a constant `obs_scale` divisor and **LTBS** (`log(DV) ~ …`) output
//! transforms; up to [`MAX_ODE_SENS_DIM`] individual parameters. Both the full
//! `Dual2` **outer** gradient and a light `Dual1` **inner** η-gradient
//! ([`ode_subject_eta_grad`]) are served (#410).
//!
//! **Not yet supported** (falls back to the gradient-free / FD path): steady-state
//! dosing, lagtime, built-in input-rate absorption, IOV, SDE/diffusion, expression
//! `obs_scale`, time-varying covariates.
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
const MAX_ODE_AXES: usize = 16;

/// True when [`ode_subject_sensitivities`] can serve this model: an ODE model
/// with a compiled RHS program, single `ObsCmt` readout, no built-in absorption,
/// no `init(...)`, no IOV/SDE, no output transform, and an individual-parameter
/// count within [`MAX_ODE_SENS_DIM`]. Per-subject gates (bolus-only doses, no TV
/// covariates/resets) are checked in [`ode_subject_sensitivities`].
pub fn ode_analytical_supported(model: &CompiledModel) -> bool {
    let Some(ode) = model.ode_spec.as_ref() else {
        return false;
    };
    if ode.rhs_program.is_none() {
        return false;
    }
    // Readout: the state directly (`ObsCmt`), a simple Form C output program
    // (`y = <expr>` over states/indiv params, e.g. `central / V1`), or a per-CMT
    // Form C (`y[CMT=N] = <expr>`) where every endpoint carries a simple program
    // (#439 — each observation reads its CMT's program over the dual state).
    let readout_ok = match &ode.readout {
        OdeReadout::ObsCmt(_) => true,
        OdeReadout::Single(_) => ode.readout_program.as_ref().is_some_and(|p| p.is_simple()),
        OdeReadout::PerCmt(map) => {
            !map.is_empty()
                && map
                    .values()
                    .all(|r| r.program.as_ref().is_some_and(|p| p.is_simple()))
        }
    };
    if !readout_ok {
        return false;
    }
    if !ode.diffusion_var.is_empty() {
        return false;
    }
    // Built-in absorption input-rate forcing is evaluated over Dual2 only for
    // kinds lifted to PkNum (#430: inverse-Gaussian). Other kinds — transit
    // until its `ln_gamma` dual rule, Weibull until Phase 2 — keep the FD
    // fallback, so a model using one is not "supported" here.
    if ode.input_rate.iter().any(|f| !f.kind.supported_over_dual()) {
        return false;
    }
    if model.n_kappa != 0 {
        return false;
    }
    // Output transforms: `None` and a constant `ScalarScale` divisor (`f/k`) are
    // applied over the dual prediction in `run_subject`, as is the LTBS log
    // (`ln f`). The `ExpressionScale` divisor form (`obs_scale = expr`) is not yet
    // handled over Dual2 — the equivalent Form-C readout (`y = state/V`) is the
    // supported route. Allowlist (not denylist) so a future scaling variant can
    // only *narrow* the analytic scope, never silently admit an unhandled one.
    if !matches!(
        model.scaling,
        ScalingSpec::None | ScalingSpec::ScalarScale(_)
    ) {
        return false;
    }
    // (ODE models have no `tv_fn` — typical values come from `pk_param_fn` at
    // η = 0 instead; see `run_subject`.)
    // Lagtime shifts the dosing timeline; supporting an estimated lagtime needs
    // ∂(timeline)/∂θ, which is not yet wired — exclude models that estimate it.
    // Use `has_lagtime()`, not a raw `pk_indices`/`PK_IDX_LAGTIME` scan: an ODE
    // model wires lagtime by name (bare `LAGTIME`/`ALAG`, or a compartment-indexed
    // `ALAG{n}` routed through the `DoseAttrMap`), and neither a named bare lag nor
    // an `ALAG{n}` lands in `pk_indices` (see `CompiledModel::has_lagtime`). The
    // dual loop never applies the dose-attr lag, so a missed `ALAG{n}` would yield a
    // no-lag gradient that diverges from the f64 predictor (#430 / #449 review #1).
    // Bioavailability F *is* supported (it scales the dose amount/rate as a dual).
    if model.has_lagtime() {
        return false;
    }
    // Per-compartment bioavailability (`F1`/`F2`, #369): production resolves
    // `f_bio(d.cmt)` per dose compartment, but both the static `integrate_g` and the
    // TV-cov walk apply the single bare `PK_IDX_F` slot, so a compartment-indexed `F`
    // would give the wrong analytic gradient — decline to FD (the bare / no-`F` case
    // is unaffected). (Per-compartment lag is already covered by `has_lagtime()`
    // above.) (#449 review #7, #1)
    if model
        .active_dose_attr_map()
        .has_indexed_attr(crate::types::DoseAttr::F)
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
    // Model-level scope + time-varying covariates (the dual walk holds the PK
    // params constant across the integration).
    if !ode_analytical_supported(model) || subject.has_tv_covariates() {
        return false;
    }
    // Steady-state dosing is not yet supported over the dual loop (needs dual
    // SS-equilibration); bolus and (finite-duration) infusion doses are handled.
    if subject.doses.iter().any(|d| d.ss && d.ii > 0.0) {
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
    // Built-in absorption forcing (igd, #430) + EVID 3/4 resets: the f64 path turns
    // off pre-reset dose tails via a `reset_floor`, which the dual forcing loop in
    // `integrate_g` doesn't yet replicate (it sums `R_in` over every dose with
    // `tad > 0`). Keep reset+absorption subjects on the FD fallback. This is the
    // SHARED scope gate for both the outer θ-sensitivities and the inner η-gradient,
    // so the inner EBE loop can't silently run an analytic no-`reset_floor` gradient
    // while the outer correctly falls back to FD (#430 review #1).
    if let Some(ode) = model.ode_spec.as_ref() {
        if !ode.input_rate.is_empty() && !subject.reset_times.is_empty() {
            return false;
        }
    }
    true
}

/// True when the time-varying-covariate ODE walk ([`run_subject_tvcov`] /
/// [`run_subject_tvcov_eta`]) can serve this `(model, subject)`: an in-scope analytic
/// ODE model whose subject carries TV covariates and uses the **bolus** dose subset.
/// Infusion / steady-state / reset / EVID=2 / `init(...)` route to the FD fallback —
/// production's TV-cov walk (`ode_predictions_event_driven`) handles those via
/// forcing/SS machinery the dual walk does not yet mirror. Checked by *both* the
/// outer and inner entry points so the analytic scope stays matched (#439).
pub(crate) fn ode_tvcov_supported(model: &CompiledModel, subject: &Subject) -> bool {
    if !ode_analytical_supported(model) || !subject.has_tv_covariates() {
        return false;
    }
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
    // Built-in absorption input-rate forcing (igd, #430): the TV-cov event-driven
    // walk (`integrate_tvcov_g`) does not carry the `R_in` forcing, so a TV-cov igd
    // model must route to the static / FD path instead of silently dropping it.
    if !ode.input_rate.is_empty() {
        return false;
    }
    // The bolus walk seeds compartments at zero; `init(...)` needs the seeded
    // initial-state machinery the static path uses, so route those to FD.
    if ode.init_fn.is_some() {
        return false;
    }
    // Bolus subset only. `all_doses_fixed` first — `is_real_infusion` debug-asserts
    // every dose is already resolved to `Fixed`.
    if !subject.all_doses_fixed() {
        return false;
    }
    if subject
        .doses
        .iter()
        .any(crate::ode::predictions::is_real_infusion)
    {
        return false;
    }
    if subject.doses.iter().any(|d| d.ss && d.ii > 0.0) {
        return false;
    }
    if subject.has_resets() || !subject.pk_only_times.is_empty() {
        return false;
    }
    true
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
    // PK params at (θ, η). The runtime lagtime short-circuit runs *before* the
    // (more expensive) param-derivative eval: a nonzero lagtime isn't supported over
    // the dual loop, so a lagtime-bearing subject declines here instead of computing
    // `pd` only to discard it downstream (#445 review #8). `pk` and `pd` are each
    // evaluated once and threaded into the drivers, so neither recomputes them.
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates);
    if pk.values[PK_IDX_LAGTIME].abs() > 1e-12 {
        return None;
    }
    // (reset+absorption FD fallback is enforced by the shared `ode_subject_supported`
    // gate above, so both the outer and inner paths decline it together — #430 review #1.)
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
    match n_indiv {
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
    }
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
    macro_rules! dispatch {
        ($($n:literal),+) => {
            match model.pk_indices.len() {
                $($n => run_subject_eta::<$n>(model, subject, theta, eta),)+
                _ => None,
            }
        };
    }
    dispatch!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
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
    if prog.n_theta_axis() != model.n_theta || prog.n_eta_axis() != model.n_eta {
        return None;
    }
    let ne = model.n_eta;
    let ni = prog.pk_slots().len();
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

/// The `DualMixed<NA, N>` initial state from a model's `init(...)` directives,
/// seeding each compartment's value **and its PK-parameter derivatives** by central
/// FD of the f64 `init_fn` over the differentiated PK slots. `init_fn` is a cheap
/// HashMap eval (no integration), so the FD cost is negligible.
///
/// Individual parameter `i` seeds dual axis `axis_of[i]` — identity (`= i`) for the
/// full `Dual2<N>` path (`NA == N`, `axis_of == None`), or the IIV-leading
/// permutation for the mixed path (#445). The first-order block fills the gradient
/// at every axis; the second-order block fills only the retained Hessian rows
/// (axis `< NA`), and skips the FD evaluations whose result would be dropped.
fn dual_init_state<const NA: usize, const N: usize>(
    init_fn: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    pk: &[f64],
    pk_indices: &[usize],
    n_states: usize,
    axis_of: Option<&[usize]>,
) -> Vec<DualMixed<NA, N>> {
    let ax = |i: usize| axis_of.map_or(i, |p| p[i]);
    let base = init_fn(pk);
    let he = 1e-6;
    let h2 = 1e-4;
    let mut out: Vec<DualMixed<NA, N>> = (0..n_states)
        .map(|s| DualMixed::constant(base.get(s).copied().unwrap_or(0.0)))
        .collect();

    // First order: gradient at every parameter's axis.
    for (i, &si) in pk_indices.iter().enumerate() {
        let ai = ax(i);
        let mut pp = pk.to_vec();
        pp[si] += he;
        let mut pm = pk.to_vec();
        pm[si] -= he;
        let (up, dn) = (init_fn(&pp), init_fn(&pm));
        for s in 0..n_states {
            out[s].grad[ai] = (up[s] - dn[s]) / (2.0 * he);
        }
    }
    // Second order: only the retained Hessian rows (axis < NA).
    for (i, &si) in pk_indices.iter().enumerate() {
        let ai = ax(i);
        // Diagonal — skip the FD evaluation entirely when this axis carries no
        // Hessian row (an IIV-free parameter in the mixed path) (#448 review #8).
        if ai < NA {
            let mut pp = pk.to_vec();
            pp[si] += h2;
            let mut pm = pk.to_vec();
            pm[si] -= h2;
            let (up, dn) = (init_fn(&pp), init_fn(&pm));
            for s in 0..n_states {
                out[s].hess[ai][ai] = (up[s] - 2.0 * base[s] + dn[s]) / (h2 * h2);
            }
        }
        for (j, &sj) in pk_indices.iter().enumerate().skip(i + 1) {
            let aj = ax(j);
            // Both axes in the dropped block — no Hessian row to fill.
            if ai >= NA && aj >= NA {
                continue;
            }
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
/// its first-order PK-parameter derivatives by central FD of the f64 `init_fn`.
fn dual1_init_state<const N: usize>(
    init_fn: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    pk: &[f64],
    pk_indices: &[usize],
    n_states: usize,
) -> Vec<Dual1<N>> {
    let base = init_fn(pk);
    let he = 1e-6;
    let mut out: Vec<Dual1<N>> = (0..n_states)
        .map(|s| Dual1::constant(base.get(s).copied().unwrap_or(0.0)))
        .collect();
    for (i, &si) in pk_indices.iter().enumerate() {
        let mut pp = pk.to_vec();
        pp[si] += he;
        let mut pm = pk.to_vec();
        pm[si] -= he;
        let (up, dn) = (init_fn(&pp), init_fn(&pm));
        for s in 0..n_states {
            out[s].grad[i] = (up[s] - dn[s]) / (2.0 * he);
        }
    }
    out
}

/// Apply the model's output transforms to a dual prediction, in PK-parameter dual
/// space (before the η/θ chain): a constant `ScalarScale` divisor `f/k` and/or the
/// LTBS log `ln(max(f, floor))`. Both are smooth functions of the prediction, so the
/// `Dual2` ops carry `∂f/∂pk` and `∂²f/∂pk²` exactly — the η/θ chain that follows is
/// unchanged. `ExpressionScale` is gated out upstream (use a Form-C readout). Mirrors
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
    let raw = match &ode.readout {
        OdeReadout::ObsCmt(idx) => st.get(*idx).copied().unwrap_or(T::from_f64(0.0)),
        OdeReadout::Single(_) => ode
            .readout_program
            .as_ref()
            .map(|p| p.eval_output_g::<T>(st, params, ro_vars, ro_stack))
            .unwrap_or(T::from_f64(0.0)),
        // Per-CMT (#439): observation j reads its own CMT's output program.
        OdeReadout::PerCmt(cmt_map) => subject
            .obs_cmts
            .get(j)
            .and_then(|cmt| cmt_map.get(cmt))
            .and_then(|r| r.program.as_ref())
            .map(|p| p.eval_output_g::<T>(st, params, ro_vars, ro_stack))
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

    // Lagtime (a nonzero dose-time shift) is not yet supported over the dual loop.
    if pk_values[PK_IDX_LAGTIME].abs() > 1e-12 {
        return None;
    }
    // Bioavailability F scales the dosed amount/rate (NONMEM F·AMT / F·RATE). F
    // lives at PK_IDX_F (pk_param_fn defaults it to 1 when undeclared); when F is an
    // estimated individual parameter, its derivative flows via `params_dual`. Use the
    // raw slot — mirroring production's `DoseAttrMap::f_bio` (raw `params[PK_IDX_F]`,
    // with the 1.0 default baked into the slot at construction) — so a transient
    // F ≤ 0 mid-fit scales the dose by F exactly as the f64 predictor does, rather
    // than substituting 1.0 and dropping ∂/∂F (#451 / #433 review #3).
    let f_bio = params_dual[PK_IDX_F];

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
        f_bio,
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
    theta: &[f64],
    eta: &[f64],
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
    theta: &[f64],
    eta: &[f64],
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
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<ObsGrad>> {
    let ode = model.ode_spec.as_ref()?;
    let n_eta = model.n_eta;

    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates);

    // First-order `∂p/∂η` only — the η-block, over a `Dual1` (no θ-axes, no Hessian).
    let dp_deta = param_eta_derivatives(model, subject, theta, eta)?;

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
) -> Vec<Dual2<M>> {
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
    let pk = (model.pk_param_fn)(theta, eta, cov);
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

    // Raw slot, mirroring production's `DoseAttrMap::f_bio` (1.0 default baked in at
    // construction) — a transient F ≤ 0 scales the dose by F like the f64 predictor,
    // not 1.0 (#451 / #433 review #3).
    let f_bio_at_dose: Vec<T> = pk_at_dose.iter().map(|p| p[PK_IDX_F]).collect();
    let first_dose_time = subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min);
    let init_state: Vec<T> = vec![T::from_f64(0.0); ode.n_states];

    let states = integrate_tvcov_g::<T>(
        program,
        ode.n_states,
        subject,
        pk_at_dose,
        pk_at_obs,
        &f_bio_at_dose,
        &init_state,
        first_dose_time,
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
    mut seed: impl FnMut(&std::collections::HashMap<String, f64>) -> Option<Vec<T>>,
) -> Option<(Vec<Vec<T>>, Vec<Vec<T>>)> {
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
    let mut cache: HashMap<Vec<(String, u64)>, Vec<T>> = HashMap::new();
    let mut seed_for = |cov: &HashMap<String, f64>| -> Option<Vec<T>> {
        let key = snapshot_key(cov);
        if let Some(v) = cache.get(&key) {
            return Some(v.clone());
        }
        let v = seed(cov)?;
        cache.insert(key, v.clone());
        Some(v)
    };
    let pk_at_dose: Vec<Vec<T>> = (0..subject.doses.len())
        .map(|k| seed_for(subject.dose_cov(k)))
        .collect::<Option<_>>()?;
    let pk_at_obs: Vec<Vec<T>> = (0..subject.obs_times.len())
        .map(|j| seed_for(subject.obs_cov(j)))
        .collect::<Option<_>>()?;
    Some((pk_at_dose, pk_at_obs))
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

    // Seed each event's per-snapshot PK duals via the shared dedup helper.
    // `seed_pk_dual2` is infallible, so wrap it in `Some`; the `?` never fires here.
    let (pk_at_dose, pk_at_obs) = seed_tvcov_snapshots::<Dual2<M>>(subject, |cov| {
        Some(seed_pk_dual2::<M>(model, prog, theta, eta, cov))
    })?;

    let preds = integrate_tvcov_readout::<Dual2<M>>(model, subject, &pk_at_dose, &pk_at_obs);

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
    Some(SubjectSens { obs: out })
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
) -> Option<Vec<Dual1<N>>> {
    let n_eta = model.n_eta;
    // The dispatch sizes `N = n_eta` exactly, so the `.min(N)` guard is always a no-op
    // — flat loop (#449 review #15).
    debug_assert_eq!(N, n_eta);
    let pd = param_derivatives_at_cov(prog, model, cov, theta, eta)?;
    let pk = (model.pk_param_fn)(theta, eta, cov);
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
    let (pk_at_dose, pk_at_obs) = seed_tvcov_snapshots::<Dual1<N>>(subject, |cov| {
        seed_pk_dual1::<N>(model, prog, theta, eta, cov)
    })?;

    let preds = integrate_tvcov_readout::<Dual1<N>>(model, subject, &pk_at_dose, &pk_at_obs);

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
#[allow(clippy::too_many_arguments)]
fn integrate_tvcov_g<T: crate::sens::num::PkNum>(
    program: &crate::parser::model_parser::OdeRhsProgram,
    n_states: usize,
    subject: &Subject,
    pk_at_dose: &[Vec<T>],
    pk_at_obs: &[Vec<T>],
    f_bio_at_dose: &[T],
    init_state: &[T],
    first_dose_time: f64,
    opts: &crate::ode::solver::OdeSolverOptions,
) -> Vec<Vec<T>> {
    let n_obs = subject.obs_times.len();
    let mut states: Vec<Vec<T>> = vec![vec![T::from_f64(0.0); n_states]; n_obs];

    // This walk is the bolus-only subset of production's event-driven predictor — it
    // omits infusion forcing, EVID 3/4 resets, EVID=2 pk-only breakpoints, and
    // lagtime, all of which `ode_tvcov_supported` already excludes. Assert the
    // invariant so a future gate change can't silently feed an unsupported subject to
    // this simplified walk (the divergence would otherwise surface only as a wrong
    // gradient) (#449 review #11).
    debug_assert!(
        !subject
            .doses
            .iter()
            .any(crate::ode::predictions::is_real_infusion)
            && !subject.has_resets()
            && subject.pk_only_times.is_empty(),
        "integrate_tvcov_g is bolus-only; ode_tvcov_supported must exclude infusion/reset/pk-only"
    );

    // Merged timeline: (time, sort-order, is_dose, idx). Bolus-only — the gate
    // excludes infusion / reset / pk-only + TV-cov — so order is just Dose(1) <
    // Obs(3) (matching production's `kind_order`) to break time ties dose-first.
    let mut tl: Vec<(f64, u8, bool, usize)> = Vec::with_capacity(subject.doses.len() + n_obs);
    for (k, d) in subject.doses.iter().enumerate() {
        tl.push((d.time, 1, true, k));
    }
    for (j, &t) in subject.obs_times.iter().enumerate() {
        tl.push((t, 3, false, j));
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
    let mut u = init_state.to_vec();
    let vars_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());
    let stack_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());

    // TAD anchor: the most recent dose at or before the current segment start. The
    // timeline is sorted and doses sort before a co-timed obs, so this only advances
    // as dose events pass — track it incrementally instead of re-scanning all doses
    // per segment (#451 re-review #6). A dose at the segment start is applied *after*
    // that segment integrates, so it anchors the *next* segment — matching the prior
    // `dt <= cur_t` scan.
    let mut last_dose_eff = f64::NEG_INFINITY;

    for &(t_event, _order, is_dose, idx) in &tl {
        // Segment `[cur_t, t_event]` uses the params evaluated at `t_event`.
        let params: &[T] = if is_dose {
            &pk_at_dose[idx]
        } else {
            &pk_at_obs[idx]
        };
        if t_event > cur_t {
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
        if is_dose {
            let d = &subject.doses[idx];
            // CMT is 1-based; a malformed `CMT=0` must not silently dose compartment
            // 0 (the datareader rejects it upstream) (#449 review #8).
            if d.cmt >= 1 {
                let cmt_idx = d.cmt - 1;
                if cmt_idx < n_states {
                    u[cmt_idx] = u[cmt_idx] + f_bio_at_dose[idx] * T::from_f64(d.amt);
                }
            }
            // This dose now anchors TAD for every later segment (all dose times count,
            // matching the prior all-doses scan, regardless of compartment validity).
            last_dose_eff = last_dose_eff.max(d.time);
        } else {
            states[idx].copy_from_slice(&u);
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
    f_bio: T,
    init_state: &[T],
    first_dose_time: f64,
    opts: &crate::ode::solver::OdeSolverOptions,
) -> Option<Vec<Vec<T>>> {
    let n_obs = subject.obs_times.len();
    let mut states: Vec<Vec<T>> = vec![vec![T::from_f64(0.0); n_states]; n_obs];
    let mut recorded = vec![false; n_obs];
    let mut u = init_state.to_vec();

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

    // Break the timeline at every dose time and — for infusions — the
    // infusion-end time, so each segment is fully inside or outside every
    // infusion window (the rate forcing is then constant over a segment).
    let t_last = subject.obs_times.iter().cloned().fold(0.0_f64, f64::max);
    let mut break_times: Vec<f64> = vec![0.0];
    for dose in &subject.doses {
        break_times.push(dose.time);
        if dose.is_infusion() {
            break_times.push(dose.time + dose.duration);
        }
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

    // Reusable scratch for the RHS evaluation across all stages.
    let vars_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());
    let stack_cell: RefCell<Vec<T>> = RefCell::new(Vec::new());

    // Per-dose bioavailability for the shared absorption-forcing helper. The static
    // walk applies one `f_bio` to every dose (per-compartment F is gated off), built
    // once per subject rather than per RK45 stage (#451 / #433 review #6).
    let dose_f_bio_all: Vec<T> = vec![f_bio; subject.doses.len()];

    for w in 0..(break_times.len() - 1) {
        let t_start = break_times[w];
        let t_end = break_times[w + 1];

        // EVID 3/4 reset: re-seed the state to the initial conditions at this
        // time, *before* the same-time dose (EVID=4 = reset + dose). Infusions
        // from a prior occasion live at earlier absolute times, so they are
        // naturally no longer active after the reset.
        if subject
            .reset_times
            .iter()
            .any(|&rt| (rt - t_start).abs() < 1e-12)
        {
            u.copy_from_slice(init_state);
        }

        // Apply bolus doses (non-infusions) at t_start: u[cmt] += F·amt. CMT is
        // 1-based; a malformed `CMT=0` must not silently dose compartment 0 (#449 #8).
        // A compartment fed by a built-in absorption input rate is skipped here — the
        // dose feeds R_in (the forcing in the RHS below), not a bolus (#430, mirroring
        // production's `input_rate_consumes_cmt` routing).
        for dose in &subject.doses {
            if !dose.is_infusion()
                && (dose.time - t_start).abs() < 1e-12
                && dose.cmt >= 1
                && !input_rate_consumes_cmt(ode, dose.cmt)
            {
                let cmt_idx = dose.cmt - 1;
                if cmt_idx < n_states {
                    u[cmt_idx] = u[cmt_idx] + f_bio * T::from_f64(dose.amt);
                }
            }
        }

        // Record any observation at t_start (after the dose). `t_start` is a break
        // time built by arithmetic on dose/reset times, so an observation that
        // coincides with it can be value-equal but bit-different — match by
        // tolerance, not bit pattern (issue #410).
        record_at(t_start, &u, &mut states, &mut recorded);

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

        // Infusions spanning this whole segment add a constant rate forcing
        // F·rate to their compartment (the break times guarantee a segment is
        // fully inside or outside each infusion window).
        let active_inf: Vec<(usize, f64)> = subject
            .doses
            .iter()
            .filter(|d| d.is_infusion())
            .filter(|d| d.time <= t_start + 1e-9 && d.time + d.duration >= t_end - 1e-9)
            .map(|d| (d.cmt.saturating_sub(1), d.rate))
            .collect();

        // Last effective dose at or before the segment start, for TAD.
        let last_dose_eff = subject
            .doses
            .iter()
            .map(|d| d.time)
            .filter(|&dt| dt <= t_start + 1e-12)
            .fold(f64::NEG_INFINITY, f64::max);

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
            for &(cmt, rate) in &active_inf {
                if cmt < du.len() {
                    du[cmt] = du[cmt] + f_bio * T::from_f64(rate);
                }
            }
            // Built-in absorption input-rate forcing R_in(tad), via the shared
            // generic helper — the same superposition loop production runs on `f64`,
            // now monomorphised on the dual type `T`. Lagtime is excluded from this
            // provider (`&[]` → tad = t − dose.time), and reset+absorption is gated to
            // FD (`NEG_INFINITY` floor → no pre-reset skip) (#430 review #4 / #451).
            if !prepared_forcings.is_empty() {
                crate::ode::predictions::add_prepared_input_rate_forcing::<T>(
                    ode,
                    prepared_forcings,
                    &subject.doses,
                    &[],
                    &dose_f_bio_all,
                    f64::NEG_INFINITY,
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
        fn check(model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64]) {
            let full = ode_subject_sensitivities(model, subject, theta, eta).expect("full");
            let light = ode_subject_eta_grad(model, subject, theta, eta).expect("light");
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

        // Form-C readout, IV bolus.
        let m = parse_model_string(TWOCPT_ODE).expect("parse");
        check(
            &m,
            &bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
            &[4.0, 12.0, 2.0, 25.0],
            &[0.12, -0.08],
        );

        // ObsCmt readout + non-zero init(...) → exercises `dual1_init_state`.
        let m = parse_model_string(INIT_ODE).expect("parse");
        let mut s = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        s.doses = vec![];
        check(&m, &s, &[1.0, 20.0], &[0.1, -0.05]);

        // Estimated bioavailability F + ObsCmt readout, oral depot → `f_bio` path.
        let m = parse_model_string(BIOAV_ODE).expect("parse");
        let mut s = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        check(&m, &s, &[5.0, 50.0, 1.5, 0.70], &[0.15, 0.2]);

        // LTBS output transform over the Dual1 readout.
        let m = parse_model_string(TWOCPT_ODE_LTBS).expect("parse");
        check(
            &m,
            &bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
            &[4.0, 12.0, 2.0, 25.0],
            &[0.12, -0.08],
        );
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

    /// Compartment-indexed bioavailability (`F1`) and lag time (`ALAG1`) land in the
    /// dose-attr map, not the bare `PK_IDX_F` / `PK_IDX_LAGTIME` slots — the dual
    /// walks apply only the bare `F` and dose at the bare `d.time`, so the analytic
    /// gradient would diverge from production's per-compartment `f_bio(cmt)` /
    /// `d.time + lagtime(cmt)`. The gate must decline both to FD (#449 re-review #1, #3).
    #[test]
    fn ode_analytical_declines_per_compartment_f_and_lag() {
        const F1_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.5, 0.05, 20.0)
  theta THETA_F1(0.70, 0.001, 0.999)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F1 = THETA_F1
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V - CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let m = parse_model_string(F1_ODE).expect("parse F1");
        assert!(
            !ode_analytical_supported(&m),
            "compartment-indexed F1 must decline to FD"
        );

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
        assert!(
            !ode_analytical_supported(&m),
            "compartment-indexed ALAG1 must decline to FD"
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
            let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates);
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
        let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates);
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
        let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates);
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
    /// whose `program` is `None` (hand-constructed / non-`is_simple`, which the dual
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
        let full = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("full");
        let light = ode_subject_eta_grad(&model, &subject, &theta, &eta).expect("light");
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
        let full = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("outer tvcov");
        let light = ode_subject_eta_grad(&model, &subject, &theta, &eta).expect("inner tvcov");
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
        // TV-cov + a real infusion → FD fallback (out of the bolus subset).
        let mut inf = tv.clone();
        inf.doses[0].duration = 1.0;
        inf.doses[0].rate = inf.doses[0].amt;
        assert!(crate::ode::predictions::is_real_infusion(&inf.doses[0]));
        assert!(!ode_tvcov_supported(&model, &inf));
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

    /// The kind gate: inverse-Gaussian (#430 slice 1) and transit (#430 slice 2)
    /// are both lifted to Dual2; a future kind (Weibull) stays on the FD fallback
    /// until its own dual rule lands.
    #[test]
    fn input_rate_kind_supported_over_dual_gates_kinds() {
        use crate::pk::absorption::InputRateKind;
        assert!(InputRateKind::InverseGaussian.supported_over_dual());
        assert!(InputRateKind::Transit.supported_over_dual());
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

    /// Built-in absorption + an EVID 3/4 reset is kept on the FD fallback in
    /// slice 1: the dual loop doesn't yet apply the `reset_floor` that turns off
    /// pre-reset dose tails, so `ode_subject_sensitivities` declines the subject.
    #[test]
    fn ode_provider_igd_with_reset_falls_back_to_fd() {
        let model = parse_model_string(IGD_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 11.0, 13.0, 16.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 100.0, 1, 0.0, false, 0.0),
        ];
        subject.reset_times = vec![10.0];
        let theta = [5.0, 50.0, 2.0, 0.3];
        let eta = [0.1, -0.05];
        assert!(
            ode_subject_sensitivities(&model, &subject, &theta, &eta).is_none(),
            "IG + reset must fall back to FD on the outer θ-sensitivity path (#430)"
        );
        // The inner η-gradient shares the scope gate, so it must decline too — else
        // the EBE loop would run an analytic no-`reset_floor` gradient while the outer
        // falls back to FD (#430 review #1). Guarded by `ode_subject_supported`.
        assert!(
            !ode_subject_supported(&model, &subject),
            "IG + reset must be out of shared scope (covers the inner η-gradient)"
        );
        assert!(
            ode_subject_eta_grad(&model, &subject, &theta, &eta).is_none(),
            "IG + reset must fall back to FD on the inner η-gradient path too (#430 review #1)"
        );
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

    /// Regression for #430 review finding 1: an igd() model that also declares a
    /// compartment-indexed `ALAG{n}` lag must stay on the FD fallback. The lag is
    /// wired through the `DoseAttrMap`, so it lands in neither `pk_indices` nor the
    /// bare `PK_IDX_LAGTIME` slot — the gate must consult `has_lagtime()`, or it
    /// would serve the model and the dual loop would compute a no-lag gradient
    /// that diverges from the f64 predictor (which shifts the forcing by the lag).
    #[test]
    fn ode_provider_igd_with_alag_stays_on_fd_fallback() {
        let model = parse_model_string(IGD_ALAG_ODE).expect("parse");
        assert!(
            model.has_lagtime(),
            "ALAG1 must enable has_lagtime() (precondition for the gate)"
        );
        assert!(
            !ode_analytical_supported(&model),
            "igd()+ALAG1 must stay on the FD fallback (#430 finding 1)"
        );
    }
}
