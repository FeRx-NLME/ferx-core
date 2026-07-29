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
use super::one_cpt::{one_cpt_conc_g, one_cpt_ig_conc_g, one_cpt_transit_conc_g};
use super::three_cpt::three_cpt_conc_g;
use super::two_cpt::{two_cpt_conc_g, two_cpt_ig_conc_g, two_cpt_transit_conc_g};
use crate::types::{
    CompiledModel, DoseEvent, GradientMethod, PkModel, ScalingSpec, Subject, PK_IDX_CL, PK_IDX_CV2,
    PK_IDX_F, PK_IDX_KA, PK_IDX_LAGTIME, PK_IDX_MAT, PK_IDX_MTT, PK_IDX_N, PK_IDX_Q, PK_IDX_Q3,
    PK_IDX_V, PK_IDX_V2, PK_IDX_V3,
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

/// Scatter one observation's `(θ,η)`-seeded `Dual2<M>` jet into an [`ObsSens`],
/// with the canonical axis layout every provider uses: θ on dual axes
/// `0..n_theta`, η on `n_theta..M` (`M = n_theta + n_eta`). The second-order
/// blocks kept are the η-η Hessian and the η-θ cross block (the only ones the
/// FOCEI outer gradient consumes). The dual must already carry any output
/// transform (`apply_output_transform` / readout) — this only reshapes.
///
/// Single home for a loop that was written verbatim in three places
/// (`run_subject_tvcov`, `subject_sensitivities_tvcov`, and the closed-form MR
/// `mr_sens_dual`, #860); a drift in the axis convention here would otherwise be
/// silently wrong in whichever copy was missed.
pub(crate) fn obs_sens_from_dual2<const M: usize>(
    fd: &Dual2<M>,
    n_theta: usize,
    n_eta: usize,
) -> ObsSens {
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
    ObsSens {
        f: fd.value,
        df_deta,
        d2f_deta2,
        df_dtheta,
        d2f_deta_dtheta,
    }
}

/// Shared accessor letting the constant-`ScalarScale` divide run once over both the
/// outer full-jet ([`ObsSens`]) and the inner η-grad ([`ObsGrad`]) observation types.
/// `for_each_scaled_axis` visits every derivative the scale multiplies by `1/k`, in the
/// same order the two hand-written copies did: the outer visits `∂f/∂η`, `∂²f/∂η²`,
/// `∂f/∂θ`, `∂²f/∂η∂θ`; the inner only `∂f/∂η` (that width carries no θ / Hessian block).
trait ScalableObs {
    fn scale_f_mut(&mut self) -> &mut f64;
    fn for_each_scaled_axis(&mut self, f: impl FnMut(&mut f64));
}

impl ScalableObs for ObsSens {
    #[inline]
    fn scale_f_mut(&mut self) -> &mut f64 {
        &mut self.f
    }
    #[inline]
    fn for_each_scaled_axis(&mut self, f: impl FnMut(&mut f64)) {
        self.df_deta
            .iter_mut()
            .chain(self.d2f_deta2.iter_mut())
            .chain(self.df_dtheta.iter_mut())
            .chain(self.d2f_deta_dtheta.iter_mut())
            .for_each(f);
    }
}

impl ScalableObs for ObsGrad {
    #[inline]
    fn scale_f_mut(&mut self) -> &mut f64 {
        &mut self.f
    }
    #[inline]
    fn for_each_scaled_axis(&mut self, f: impl FnMut(&mut f64)) {
        self.df_deta.iter_mut().for_each(f);
    }
}

/// Divide a subject's jet by a constant `ScalarScale` output divisor `k` (`f_scaled = f/k`)
/// in place, on either observation width. Every derivative is linear in `f` and `k` is
/// constant (`∂k/∂η = ∂k/∂θ = 0`), so value, gradient, and (for the outer width) both
/// Hessian blocks divide by the same `k`. Matches `pk::apply_scaling` (`pred /= s`); no-op
/// at `k == 1.0`. Per `pk::apply_scaling` / `ScalingSpec::build_obs_scale_array`, a
/// non-positive or non-finite scale is invalid and NaN-outs the whole jet (loud-failure
/// semantics) rather than emitting an `inf`/sign-flipped jet, keeping the analytic path in
/// lock-step with the production predictor the FD oracle differentiates. Shared by the
/// closed-form, TV-cov, and IOV outer/inner walks so the field set stays in sync (#486).
#[inline]
fn scale_obs_sens<J: ScalableObs>(obs: &mut [J], k: f64) {
    if k == 1.0 {
        return;
    }
    let inv = if k.is_finite() && k > 0.0 {
        1.0 / k
    } else {
        f64::NAN
    };
    for o in obs.iter_mut() {
        *o.scale_f_mut() *= inv;
        o.for_each_scaled_axis(|v| *v *= inv);
    }
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

/// The seven PK-class booleans that select which closed-form concentration kernel a dose
/// superposition dispatches to. Bundled into one `Copy` argument so [`superpose_doses`]
/// takes it in place of seven flags, and the outer ([`run_obs`]) and inner
/// ([`run_obs_grad`]) walkers build it from the same field set.
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
}

impl<T: PkNum> SeededPkDuals<T> {
    /// Seed each PK slot as an independent dual axis (`seed_dim[slot] = Some(k)`) or a
    /// constant (`None`), reading the values off `pk`. Mirrors the per-slot `dv` closure
    /// the two walkers previously inlined verbatim.
    fn seed(seed_dim: &[Option<usize>; N_PK], pk: &crate::types::PkParams) -> Self {
        let dv = |slot: usize, value: f64| -> T {
            match seed_dim[slot] {
                Some(k) => T::var(value, k),
                None => T::from_f64(value),
            }
        };
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
        }
    }
}

/// Seed the full `N_PK`-slot PK-parameter dual array the analytic Form-C readout reads
/// (indexed by PK slot, so `eval_output_g` can pull `V`, binding constants, … via its
/// `indiv_to_pk` plan). Same per-slot var/constant seeding as [`SeededPkDuals::seed`],
/// over every slot. Generic so the outer/inner walkers build it at `Dual2<N>`/`Dual1<N>`.
fn ro_pk_dual_array<T: PkNum>(
    seed_dim: &[Option<usize>; N_PK],
    pk: &crate::types::PkParams,
) -> [T; N_PK] {
    std::array::from_fn(|s| {
        let val = pk.values.get(s).copied().unwrap_or(0.0);
        match seed_dim[s] {
            Some(k) => T::var(val, k),
            None => T::from_f64(val),
        }
    })
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
/// (`dose.time + lag < reset_floor` excludes washed-out doses, keyed on the *lagged
/// arrival* — a dose recorded before a reset but arriving after it via lagtime still
/// contributes, exactly as the event-driven walk applies it, PR #381 #2). `elapsed`
/// carries the lagtime shift and the SS pre-arrival tail wrap ([`lagged_elapsed`]).
/// Generic over the seeded dual type, so the `Dual2` outer ([`run_obs`]) and `Dual1`
/// inner ([`run_obs_grad`]) walkers share the byte-identical dispatch ladder.
#[inline]
fn superpose_doses<T: PkNum>(
    d: &SeededPkDuals<T>,
    flags: PkClassFlags,
    subject: &Subject,
    t_obs: f64,
    reset_floor: f64,
) -> T {
    let mut fd = T::from_f64(0.0);
    for dose in &subject.doses {
        if dose.time + d.lag_val < reset_floor {
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
        } else if flags.three_cpt {
            three_cpt_conc_g(
                dose, elapsed, d.cl_d, d.v1_d, d.q_d, d.v2_d, d.q3_d, d.v3_d, d.ka_d, d.f_d,
                flags.oral,
            )
        } else if flags.two_cpt {
            two_cpt_conc_g(
                dose, elapsed, d.cl_d, d.v1_d, d.q_d, d.v2_d, d.ka_d, d.f_d, flags.oral,
            )
        } else {
            one_cpt_conc_g(dose, elapsed, d.cl_d, d.v1_d, d.ka_d, d.f_d, flags.oral)
        };
        fd = fd + c;
    }
    fd
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
/// outer ([`subject_sensitivities`] → `run_obs`) and inner ([`subject_eta_grad`] →
/// `run_obs_grad`) tables enumerate `1..=9`; a wider model falls through their `_ => None`
/// arm. **Keep these three in step** — raising this constant without adding the matching
/// `const_dispatch!` arms would re-open the misreport below.
const MAX_CLOSED_FORM_SLOTS: usize = 9;

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
/// providers dispatch on when the program path is live. The `lognormal_param_derivatives`
/// fallback (`model.pk_indices`) is unchanged from before #486 and left alone here.
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
/// **tv-covariate / oral-infusion / TIME** event-walk path (#650).
///
/// The walk's `PkDual` carries only the eight structural slots (`CL..V3`), so a readout
/// referencing a 9th+ individual parameter (reachable once several non-structural readout
/// params are allocated — a 3-cpt-oral sigmoid-Emax `y = <expr>` exhausts the low slots
/// immediately) used to fall back to FD here. Since #486 the walk materializes those higher
/// slots per observation (`ro_extra` in `run_obs_tvcov` / `run_obs_grad_tvcov`) — value from
/// `pk_param_fn`, jet from the same program derivatives — so the slot ceiling is gone and
/// this now reduces to the static-path scope.
fn readout_tvcov_supported(model: &CompiledModel) -> bool {
    let Some(ar) = &model.analytic_readout else {
        return true;
    };
    match &ar.program {
        Some(_) => analytic_readout_dual_supported(model),
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

/// Apply the analytic Form-C readout (#650) to one observation's (already clamped)
/// concentration jet: replace `conc` with `y = <expr>` over the seeded dual (central
/// amount = conc × V), under this observation's `ModelTimeGuard` so a `TIME`-referencing
/// readout resolves `Op::PushTime` to `t_obs` (matching production's
/// `apply_analytic_readout` guard; a no-op for a `TIME`-free readout). Returns `conc`
/// unchanged with no readout, touching the covariate/time snapshots only on the readout
/// path. Generic over the dual width, so the `Dual2` outer ([`run_obs`]) and `Dual1` inner
/// ([`run_obs_grad`]) walkers share the guard/eval wiring (#637); the resulting `∂y/∂pk`
/// (and `∂²y/∂pk²` on the outer width) then rides the same `pd`/`dp_deta` chain.
#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_readout_jet<T: PkNum>(
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
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter(
            subject.obs_times.get(obs_i).copied().unwrap_or(0.0),
        );
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

/// κ = 0 (BSV-only) readout-parameter snapshots for the analytic IOV walk (#655).
/// Production's [`crate::pk::apply_analytic_readout`] builds the readout PK params from
/// `eta_bsv` (no κ) — only the concentration carries the occasion κ — so these snapshots
/// carry `∂/∂(θ, η_bsv)` and zero `∂/∂κ`. When covariates / `TIME` vary across observations
/// the snapshot is per-observation; otherwise a single shared snapshot suffices (the readout
/// param jet is then identical across rows, so a per-obs `Vec` would deep-clone the nested
/// `CombinedDerivs` `N − 1` times per gradient eval on a hot path — `bsv_amount` already
/// carries a single snapshot for the same reason).
enum ReadoutSnapshots {
    Static((crate::types::PkParams, CombinedDerivs)),
    PerObs(Vec<(crate::types::PkParams, CombinedDerivs)>),
}

impl ReadoutSnapshots {
    /// The κ = 0 readout snapshot for observation `j` (the shared one when `Static`).
    #[inline]
    fn at(&self, j: usize) -> &(crate::types::PkParams, CombinedDerivs) {
        match self {
            Self::Static(s) => s,
            Self::PerObs(v) => &v[j],
        }
    }
}

/// Evaluate the analytic Form C readout at one IOV observation (#655) — shared by the outer
/// (`Dual2`) and inner (`Dual1`) walks so the κ = 0 seeding convention lives in one place.
/// Seeds the readout PK params from the BSV-only snapshot (`group = None`, so the κ axes are
/// dropped) via the caller's monomorphized `seed`, then runs the readout jet under this
/// observation's model time.
///
/// `conc` **must already be clamped to ≥ 0** — production clamps `conc.max(0.0)` *before* the
/// readout ([`crate::pk::event_driven`] → [`crate::pk::apply_analytic_readout`]) and returns
/// the readout output unclamped, so the caller clamps the concentration jet and does not
/// re-clamp the result. A readout referencing `TIME` resolves `Op::PushTime` from the
/// model-time thread-local, so it is entered at `obs_time` to match the guard
/// `apply_analytic_readout` sets (a no-op read for a `TIME`-free readout).
#[allow(clippy::too_many_arguments)]
#[inline]
fn eval_readout_at_obs<T: crate::sens::num::PkNum>(
    prog: &crate::parser::model_parser::OdeOutputProgram,
    conc: T,
    snapshot: &(crate::types::PkParams, CombinedDerivs),
    slot_row: &[Option<usize>; N_PK],
    cov: &std::collections::HashMap<String, f64>,
    obs_time: f64,
    seed: impl Fn(&CombinedDerivs, Option<usize>, usize, f64) -> T,
    ro_state: &mut Vec<T>,
    ro_vars: &mut Vec<T>,
    ro_stack: &mut Vec<T>,
) -> T {
    let (pk_ro, cd_ro) = snapshot;
    let params: [T; N_PK] = std::array::from_fn(|s| {
        let val = pk_ro.values.get(s).copied().unwrap_or(0.0);
        match slot_row[s] {
            Some(i) => seed(cd_ro, None, i, val),
            None => T::from_f64(val),
        }
    });
    let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter(obs_time);
    eval_readout_jet::<T>(prog, conc, &params, cov, ro_state, ro_vars, ro_stack)
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
///
/// Raised 16 → 24 (#486): a mid-sized covariate model (5 structural θ + 6
/// covariate-effect θ + 5 η) sits at exactly 16 axes, so the old cap declined the
/// *next* covariate a modeller added. Must stay `>= MAX_ODE_AXES` (static assert in
/// `ode_provider.rs`).
pub(crate) const MAX_SCALE_AXES: usize = 24;

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
            17 => $apply::<17>($($arg),+),
            18 => $apply::<18>($($arg),+),
            19 => $apply::<19>($($arg),+),
            20 => $apply::<20>($($arg),+),
            21 => $apply::<21>($($arg),+),
            22 => $apply::<22>($($arg),+),
            23 => $apply::<23>($($arg),+),
            24 => $apply::<24>($($arg),+),
            _ => {}
        }
    };
}

// The `dispatch_init_impulse!` table is hand-written for `1..=24`. If the axis cap
// changes, the table (and the scale-program dispatch tables) must change with it —
// fail to compile rather than silently drop the init impulse on the `_` arm.
const _: () = assert!(
    MAX_SCALE_AXES == 24,
    "dispatch_init_impulse! enumerates 1..=24; update it when MAX_SCALE_AXES changes",
);

/// Monomorphize a `const`-generic dispatcher on a runtime dimension: expand
/// `$dim` against the literal table `$($n),+`, binding the matched literal as a
/// block-local `const $N: usize` for `$body` to use as its const-generic
/// argument, with a `_ => None` fallback. Written once and shared by the seven
/// `run_*::<N>` dispatch ladders below (formerly a per-function `macro_rules!
/// disp` re-declared at each site). The generated `match` arms are the same
/// tokens as the hand-written ladders, so the dispatch is bit-identical.
macro_rules! const_dispatch {
    ($dim:expr; $($n:literal),+ $(,)?; |$N:ident| $body:expr $(,)?) => {
        match $dim {
            $($n => { const $N: usize = $n; $body })+
            _ => None,
        }
    };
}

/// Maximum `(θ, η)` axis count (`n_theta + n_eta`) for the TV-cov event-driven dual
/// walk. The outer `run_obs_tvcov` (`m_dim`) and inner `run_obs_grad_tvcov` (`n_eta
/// ≤ m_dim`) dispatch tables both enumerate `1..=MAX_TVCOV_AXES`, and
/// `tvcov_analytical_supported` bounds the model here, so both resolve and the
/// inner/outer analytic scope stays matched (#449 re-review #2).
/// Raised 16 → 24 alongside `MAX_ODE_AXES` (#486).
///
/// **Must not exceed `MAX_ODE_AXES`** (static assert below). The *inner* TV-cov walk
/// (`run_obs_grad_tvcov`) obtains its `∂p/∂η` from `param_derivatives_at_cov`, whose dispatch
/// table is bounded by `MAX_ODE_AXES`; a model admitted here but wider than that would take the
/// analytic **outer** gradient while the inner declined via `?` — an analytic outer against a
/// fixed-EBE FD inner, exactly the split `subject_eta_grad_tvcov_with_schedule` forbids
/// (#449 re-review #2). Raise the two together, never this one alone.
///
/// That decline is now the *only* consequence of a cap divergence: `run_obs_grad_tvcov` hoists
/// the program derivatives, so every consumer inside it is total and a miss can no longer be
/// papered over with a zero jet (it used to be, in the modeled-dose slot dual and the readout's
/// high slots — a wrong gradient reported as analytic). The assert still stands, because an
/// inner-FD / outer-analytic *scope split* remains wrong on its own.
const MAX_TVCOV_AXES: usize = 24;

const _: () = assert!(
    MAX_TVCOV_AXES <= crate::sens::ode_provider::MAX_ODE_AXES,
    "MAX_TVCOV_AXES exceeds MAX_ODE_AXES: the TV-cov walk would admit a model whose inner      `param_derivatives_at_cov` dispatch declines, splitting inner/outer analytic scope"
);

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
/// Non-Gaussian endpoints — `[event_model]` TTE and `[binary_model]` logistic (#760) —
/// are excluded: their data log-likelihood has no analytic outer gradient (the
/// sensitivity provider only covers the structural PK/PD model), so such an objective
/// (or a mixed PK + non-Gaussian objective) must be differentiated by finite differences.
/// Without this guard the model would report an analytic gradient it cannot supply, so
/// `resolve_auto` would pick a gradient-based optimizer that then stalls on a meaningless
/// gradient (these endpoints are FD-only — see `docs/estimation/tte.qmd`).
pub fn analytic_outer_gradient_available(model: &CompiledModel) -> bool {
    !matches!(model.gradient_method, GradientMethod::Fd)
        // Correlated residual (`block_sigma`, #627) now has an analytic outer gradient
        // for the analytical (diagonal-R) scope: the dense Almquist assembly reduces to
        // the scalar path fed correlation-aware `(r,d,d2)` (`corr_residual_diag`). A rare
        // off-diagonal-R subject bails per-subject to FD via `corr_residual_diag` → `None`.
        && !model.has_non_gaussian()
        // Custom / time-varying residual-error magnitude (#484/#576/#486): `mult(θ)`
        // makes `R` depend on θ directly, which `sens_outer_gradient::theta_block`
        // now carries via a `Dual1`-differentiated direct-θ channel — bounded by the
        // same `Dual1<M>` dispatch table `ScaleDerivProgram`/`ExpressionScale` uses
        // (`MAX_RUV_MAG_AXES`). Beyond that axis count, or combined with a **correlated**
        // residual (`block_sigma`, #627 — the dense assembly does not thread the
        // magnitude's direct-θ channel), still routes to FD. `iiv_on_ruv` is now analytic
        // combined with the magnitude (the residual-eta `c̃`-column coupling `d/R` gets its
        // direct-θ terms in `theta_block`). (An M3-censored row's `−logΦ(z)` direct-θ chain
        // is threaded per-subject too now; where it can't be — a subject with censored rows
        // under the FOCEI magnitude path — `prepare_stacked` bails at runtime, not here.)
        && (!model.has_custom_ruv_magnitude()
            || (model.n_theta <= crate::parser::model_parser::MAX_RUV_MAG_AXES
                && model.residual_correlations.is_empty()))
        // `iov_sens_supported` (not just the closed-form `iov_analytical_supported`) so
        // the predicate also recognizes the ODE IOV outer gradient (#439 ODE IOV / #466).
        && (sens_supported(model) || iov_sens_supported(model))
    // IIV on residual error (#474) needs no gate here: the analytic gradient (inner η-column +
    // outer θ/Ω/σ variance terms) is provider-agnostic, so it serves every `iiv_on_ruv`
    // combination — closed-form and ODE, plain / IOV / M3-BLOQ including the triples
    // (#474/#4b/#591/#623/#677) — with no FD carve-out.
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

/// Model-level umbrella for [`ode_inner_grad_supported`]: whether the light `Dual1`
/// ODE inner η-gradient can serve the *best-case* subject of this model. Both
/// per-subject gates ([`ode_subject_supported`], [`ode_tvcov_supported`]) require
/// [`ode_analytical_supported`] as their subject-independent core, so a servable
/// subject always exists when this holds — a plain-bolus, fixed-dose, no-TV-cov
/// subject takes the static walk, and a lagtime / TV-cov / SS / modeled-dose subject
/// routes to the event-driven walk. **Non-IOV only:** `ode_analytical_supported`
/// requires `n_kappa == 0` (the ODE IOV inner is [`ode_iov_supported`]) and
/// `ode_spec.is_some()` (false for every closed-form / endpoint-only model), so this
/// never overlaps the closed-form or IOV report branches.
///
/// It deliberately does **not** fold in the escape hatches (`gradient = fd`, SDE,
/// magnitude × `block_sigma`): the caller applies `analytic_inner_common_bail`, whose
/// hatches now coincide exactly with the ones the live per-subject ODE branch in
/// `analytic_inner_grad_supported` hand-inlines (the `log_transform && n_kappa > 0`
/// clause that used to make the bail a strict superset was dropped when the closed-form
/// IOV inner learned the `ln` jet, #486). Consumed (via the shared
/// `inner_reports_analytic_model`) by `build_info::gradient_method_inner` to report
/// the ODE inner route at model level without a per-subject scan — the inner analog of
/// how [`sens_supported`] already lets the *outer* report recognize ODE (#378 task B).
///
/// [`ode_analytical_supported`]: crate::sens::ode_provider::ode_analytical_supported
/// [`ode_subject_supported`]: crate::sens::ode_provider::ode_subject_supported
/// [`ode_tvcov_supported`]: crate::sens::ode_provider::ode_tvcov_supported
/// [`ode_iov_supported`]: crate::sens::ode_provider::ode_iov_supported
pub(crate) fn ode_inner_grad_supported_model(model: &CompiledModel) -> bool {
    ODE_SENS_ENABLED && crate::sens::ode_provider::ode_analytical_supported(model)
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

/// Resolve the exact `(∂p/∂(θ,η)` full [`ParamDerivs`], `slots)` pair for a subject:
/// evaluate the compiled `[individual_parameters]` program over `Dual2` seeded on
/// `(θ, η)` when it covers the required PK slots, else fall back to the closed-form
/// log-normal chain (`pk·sel` / `tv_theta_jacobian`) whose rows follow
/// `pk_indices` order. Returns `None` — the caller falls back to FD — only when the
/// program path is unavailable AND some PK-param η is not `LogNormal` (the
/// log-normal chain would then be wrong). Hoisted from the two byte-identical outer
/// call sites (`apply_tvcov_init_outer`, `subject_sensitivities`) — see F4.
fn resolve_param_derivs(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    pk: &crate::types::PkParams,
) -> Option<(crate::sens::ode_provider::ParamDerivs, Vec<usize>)> {
    match model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .filter(|prog| prog_covers_required_pk_slots(model, prog))
        .and_then(|prog| {
            crate::sens::ode_provider::param_derivatives_from_prog(prog, model, subject, theta, eta)
                .map(|pd| (pd, prog.pk_slots()))
        }) {
        Some(v) => Some(v),
        None => {
            if !model
                .eta_param_info
                .iter()
                .all(|e| e.param_type == crate::types::EtaParamType::LogNormal)
            {
                return None;
            }
            let pd = lognormal_param_derivatives(model, subject, theta, pk);
            Some((pd, model.pk_indices.clone()))
        }
    }
}

/// PK slot → differentiated-row index of a `pd`/`dp_deta` whose rows follow `slots`
/// order: `out[slots[i]] = Some(i)` for every in-range slot, `None` otherwise.
/// Hoisted from the five identical `[None; N_PK]` build loops (F4).
fn seed_dim_from_slots(slots: &[usize]) -> [Option<usize>; N_PK] {
    let mut seed_dim: [Option<usize>; N_PK] = [None; N_PK];
    for (i, &slot) in slots.iter().enumerate() {
        if slot < N_PK {
            seed_dim[slot] = Some(i);
        }
    }
    seed_dim
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
    const_dispatch!(
        n_theta + n_eff;
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24;
        |M| Some(iov_combined_derivs::<M>(
            prog, n_theta, n_eff, n_rows, cov, theta, combined,
        ))
    )
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
        ScalingSpec::ExpressionScale { deriv: Some(p), .. }
            if p.n_theta_axis() == model.n_theta
                && p.n_eta_axis() == model.n_eta
                && (1..=MAX_SCALE_AXES).contains(&p.n_axes()) => {}
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
    // params from the κ = 0 (BSV-only) per-obs snapshot while the walk's concentration
    // carries κ, matching production `apply_analytic_readout(..., eta_bsv, ...)`. A
    // readout outside that scope (per-CMT, depot, direct θ/η, slot overflow) declines to
    // FD here so the reported method stays honest — the closed-form IOV twin of
    // `analytical_supported_core`'s `analytic_readout_dual_supported` clause.
    // `readout_tvcov_supported` is `true` for a plain (no-readout) model.
    if !readout_tvcov_supported(model) {
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
    // A closed-form transit/IG IOV model is served by its ODE twin (issue #719, routed
    // per-subject by `CompiledModel::effective_for`), so its analytic-IOV outer-gradient scope
    // is the twin's ODE-IOV scope. (`get_or_build` is cached; the twin is built for the fit
    // anyway.)
    if model.n_kappa > 0 {
        if let Some(eq) = &model.absorption_ode_equivalent {
            return ODE_SENS_ENABLED
                && crate::sens::ode_provider::ode_iov_supported(eq.get_or_build());
        }
    }
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
    // Closed-form transit/IG IOV → its ODE twin (issue #719); no-op otherwise. See
    // `subject_sensitivities_iov`.
    let model = model.effective_for(subject);
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
    // #905: IOV twin of the non-IOV `subject_routes_to_event_walk` gate. An analytical
    // subject with no Gaussian observation feeds the predictor nothing; declining here
    // (→ the caller drops it to FD) keeps FOCE/FOCEI out of the IOV sens walk, whose
    // dose-routing `panic!`s (`propagate::active_rates_g` / the bolus `cmt_idx` guard)
    // are the same twins the value gate fences off. Routed TTE-only subjects never reach
    // here (they carry `obs_records`, so the analytic-IOV gates already decline them);
    // this fires only on the model-blind loader that put the TTE rows in `obs_times`.
    if model.ode_spec.is_none() && !crate::pk::subject_feeds_analytical_pk(model, subject) {
        return None;
    }
    // Cheap, model/subject-only decline before any dual seeding (#822 review #9) — see
    // `subject_sensitivities_tvcov`. Harmless for the ODE-twin branch below: an ODE model has no
    // closed-form walk to decline for, and `ss_lagtime_walk_unsupported` is about that walk's
    // missing pre-arrival SS overlay.
    if !model.ode_spec.is_some() && ss_lagtime_walk_unsupported(model, subject) {
        return None;
    }
    // Closed-form transit/IG IOV is served by its ODE twin (issue #719) — the twin carries
    // cross-occasion dose amounts exactly (#104/#663), which the closed-form superposition
    // cannot. A no-op for every other model, so the `ode_spec` branch below then picks up the
    // twin's analytic ODE-IOV sensitivities.
    let model = model.effective_for(subject);
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
    // Analytic Form C readout program (#655), threaded into the walk to replace each
    // observation's central concentration with `y = <expr>`. `iov_analytical_supported`
    // has already confirmed the readout (if any) fits the walk. `None` for a plain model.
    let readout = model
        .analytic_readout
        .as_ref()
        .and_then(|ar| ar.program.as_ref());
    let s = build_iov_sources(model, subject, theta, stacked_eta)?;
    // Run the walk over `Dual2<M>` (M = n_theta + n_stacked); the dual width tracks
    // the *unknowns* (n_eta + K·n_kappa + n_theta), not the PK axes, so it stays
    // narrow for many occasions whenever n_kappa < n_diff (the usual κ-on-CL case).
    let m_dim = s.n_theta + s.n_stacked;
    let mut sens = const_dispatch!(
        m_dim;
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24;
        |M| run_obs_iov::<M>(
            model, subject, theta, stacked_eta, &s.sources, &s.dose_src, &s.obs_src,
            &s.pkonly_src, &s.slot_row, s.n_eta, s.n_kappa, s.n_eff, s.n_stacked,
            s.n_theta, s.scale_groups.as_deref(), s.bsv_amount.as_ref(),
            readout, s.readout_obs.as_ref(),
        )
    )?;
    // LTBS (#486): apply the `ln(f)` jet LAST — after the in-walk scale quotient / Form C
    // readout / init impulse — so the outer θ/Ω/σ gradient matches production's scale-then-log
    // `ln(f/s)` (`predict_iov` logs last, after `apply_scaling`). Post-walk on the full stacked
    // jet, so it is dimension-generic over the κ axes; a no-op when `log_transform` is false.
    // `run_obs_iov_eta` applies the inner twin of this step last, so the IOV inner EBE gradient
    // serves LTBS too (#486) — the two loops differentiate the same `ln(f/s)`.
    apply_ltbs_transform_outer(&mut sens, model.log_transform);
    Some(sens)
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
    /// For an analytic Form C readout (`[scaling] y = <expr>`) under IOV (#655): one
    /// **BSV-only** (`κ = 0`) `(pk, combined-derivs)` snapshot per observation,
    /// evaluated at that observation's covariate / `TIME` snapshot. Production's
    /// `apply_analytic_readout` builds the readout PK params (the amount's `V`, plus
    /// non-structural `BMAX`/`KD`) from `eta_bsv` (no κ) — only the concentration jet
    /// carries κ — so this snapshot drives the readout param jet with `∂/∂(θ, η_bsv)`
    /// and zero `∂/∂κ` (seeded `group = None`), while the walk's κ-bearing concentration
    /// supplies the central amount. `Static` when covariates / `TIME` are subject-static
    /// (one shared snapshot), `PerObs` otherwise; `None` when the model has no analytic
    /// readout. See [`ReadoutSnapshots`].
    readout_obs: Option<ReadoutSnapshots>,
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
    // of the same walk). Since #486 that includes a subject mixing SS with resets:
    // the walk equilibrates each SS dose per event and the reset zeros the state, so
    // it mirrors production exactly (only the *static* superposition path, which folds
    // in an infinite periodic history, cannot express it — see `subject_sensitivities_impl`).
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
    let slot_row = seed_dim_from_slots(slots);

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

    // Analytic Form C readout param snapshots (#655): one κ = 0 (BSV-only) `(pk, cd)`
    // per observation, matching production `apply_analytic_readout(..., eta_bsv,
    // obs_cov(j), obs_time(j))`. Per-event when covariates / `TIME` vary (each obs at its
    // own snapshot); otherwise one static snapshot reused across observations (the
    // derivative-program eval is the cost, so clone the shared `cd`). Built only when a
    // readout is present — such a model is init-free with no `[scaling]` divisor, so
    // `bsv_amount`/`scale_groups` are `None` and the post-walk transform blocks are inert.
    let readout_obs = if model.analytic_readout.is_none() {
        None
    } else if per_event {
        let mut v = Vec::with_capacity(subject.obs_times.len());
        for j in 0..subject.obs_times.len() {
            let cov = subject.obs_cov(j);
            let t = subject.obs_times[j];
            let pk = (model.pk_param_fn)(theta, &combined_pk_only, cov, t);
            let cd = cd_at(t, &combined_pk_only, cov)?;
            v.push((pk, cd));
        }
        Some(ReadoutSnapshots::PerObs(v))
    } else {
        // Subject-static covariates / `TIME`: one shared snapshot, so the walk reads the same
        // κ = 0 readout params at every observation (no per-obs `CombinedDerivs` clone).
        let pk = (model.pk_param_fn)(theta, &combined_pk_only, cov_static, 0.0);
        let cd = cd_at(0.0, &combined_pk_only, cov_static)?;
        Some(ReadoutSnapshots::Static((pk, cd)))
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
        readout_obs,
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
    // #905: decline a no-Gaussian-observation analytical subject (→ FD), so the IOV
    // inner gradient never enters the sens walk on the exempt subject's unroutable dose.
    // This is the twin the model-blind loader reaches: `analytic_iov_inner`'s
    // `!subject_has_survival_records` guard fails there (TTE rows sit in `obs_times`, not
    // `obs_records`), so without this the walk panics at `propagate.rs` bolus/infusion.
    if model.ode_spec.is_none() && !crate::pk::subject_feeds_analytical_pk(model, subject) {
        return None;
    }
    // Cheap decline before any dual seeding (#822 review #9) — see `subject_sensitivities_tvcov`.
    if ss_lagtime_walk_unsupported(model, subject) {
        return None;
    }
    if !iov_analytical_supported(model) {
        return None;
    }
    // Analytic Form C readout program (#655); `iov_analytical_supported` has already
    // confirmed the readout (if any) fits the walk. `None` for a plain model.
    let readout = model
        .analytic_readout
        .as_ref()
        .and_then(|ar| ar.program.as_ref());
    let s = build_iov_sources(model, subject, theta, stacked_eta)?;
    let n_dim = s.n_stacked;
    const_dispatch!(
        n_dim;
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24;
        |N| run_obs_iov_eta::<N>(
            model, subject, theta, stacked_eta, &s.sources, &s.dose_src, &s.obs_src,
            &s.pkonly_src, &s.slot_row, s.n_eta, s.n_kappa, s.n_stacked,
            s.scale_groups.as_deref(), s.bsv_amount.as_ref(),
            readout, s.readout_obs.as_ref(),
        )
    )
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
    readout: Option<&crate::parser::model_parser::OdeOutputProgram>,
    readout_obs: Option<&ReadoutSnapshots>,
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

    // Modeled-`RATE=-1/-2` doses (#486): resolve each dose at its occasion's PK jet
    // (`sources[dose_src[k]].0` is already `pk_param_fn` at that occasion's combined
    // effect) for the schedule's break times — shared with the non-IOV walk via
    // `resolve_eff_doses` — and build the per-dose duals seeded on the stacked
    // `(η_bsv, κ)` axes (via the same `seed` used for the PkDuals). Fixed subjects
    // borrow `subject.doses` and pass no duals.
    let dose_pk: Vec<crate::types::PkParams> = (0..subject.doses.len())
        .map(|k| sources[dose_src[k]].0)
        .collect();
    let eff_doses = resolve_eff_doses(model, subject, |k| dose_pk[k]);
    // Lagtime under IOV (#486): each dose's lag is read at *its own occasion's* snapshot,
    // so an occasion-varying `ALAG` (κ on the lag) moves that occasion's arrivals only —
    // it falls out of the per-dose seeding for free.
    let dose_lagtimes = dose_lagtime_values(model, &dose_pk);
    // Coincident moving breaks carrying different jets — and, under a lagtime, a moving
    // break landing on a fixed obs/reset/cov-change time — aren't representable by the
    // single-`dt` walk; decline to FD (#486 review #2).
    if !moving_bounds_separable(model, subject, &eff_doses, &dose_lagtimes) {
        return None;
    }
    let slot_dual = |k: usize, slot: usize| -> Dual2<M> {
        let (pk, cd, group) = &sources[dose_src[k]];
        let val = pk.values.get(slot).copied().unwrap_or(0.0);
        match slot_row.get(slot).copied().flatten() {
            Some(i) => seed(cd, *group, i, val),
            None => Dual2::<M>::constant(val),
        }
    };
    let dose_inf_dual = modeled_dose_inf_duals::<Dual2<M>>(model, subject, &slot_dual);
    let dose_lag_dual = dose_lag_duals::<Dual2<M>>(model, subject, &slot_dual);
    let schedule = EventSchedule::for_subject(subject, model.pk_model, &eff_doses, &dose_lagtimes);
    let conc = event_driven_sens_with_doses_g::<Dual2<M>>(
        model.pk_model,
        subject,
        &schedule,
        &eff_doses,
        &dose_lag_dual,
        &dose_inf_dual,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    // Analytic Form C readout (#655): scratch for the per-observation dual eval.
    let mut ro_state: Vec<Dual2<M>> = Vec::new();
    let mut ro_vars: Vec<Dual2<M>> = Vec::new();
    let mut ro_stack: Vec<Dual2<M>> = Vec::new();

    let mut obs_out = Vec::with_capacity(conc.len());
    for (j, c) in conc.iter().enumerate() {
        // Clamp the concentration jet to ≥ 0 BEFORE any readout — production clamps
        // `conc.max(0.0)` in `event_driven_predictions` and only then applies the readout
        // (which is returned unclamped). A negative value's derivatives vanish, matching the
        // OFV. For a plain (no-readout) model this reproduces the old post-loop clamp exactly
        // (`constant(0.0)` has zero grad/hess).
        let c_clamped: Dual2<M> = if c.value < 0.0 {
            Dual2::<M>::constant(0.0)
        } else {
            *c
        };
        // Analytic Form C readout (#655): replace the (clamped) central concentration jet
        // with `y = <expr>`. The concentration carries the occasion κ (through the walk);
        // the readout PK params come from the κ = 0 (BSV-only) snapshot (`readout_obs.at(j)`,
        // seeded `group = None`), matching production `apply_analytic_readout(..., eta_bsv)`.
        // The readout output is NOT re-clamped (production does not re-clamp it).
        let c: Dual2<M> = match (readout, readout_obs) {
            (Some(ro), Some(ro_obs)) => eval_readout_at_obs::<Dual2<M>>(
                ro,
                c_clamped,
                ro_obs.at(j),
                slot_row,
                subject.obs_cov(j),
                subject.obs_times[j],
                &seed,
                &mut ro_state,
                &mut ro_vars,
                &mut ro_stack,
            ),
            _ => c_clamped,
        };
        let c = &c;
        let mut df_deta = vec![0.0; n_stacked];
        let mut df_dtheta = vec![0.0; n_theta];
        let mut d2f_deta2 = vec![0.0; n_stacked * n_stacked];
        let mut d2f_deta_dtheta = vec![0.0; n_stacked * n_theta];
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
        obs_out.push(ObsSens {
            f: c.value,
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

    // Constant `ScalarScale` output divisor `f/k`: covariate/η/κ-independent, so every jet
    // entry (value, gradient, Hessian) divides by the same `k` — the IOV twin of the non-IOV
    // `f_scaled = f/k` and production `apply_scaling` (#486 IOV-scope parity).
    if let ScalingSpec::ScalarScale(k) = model.scaling {
        scale_obs_sens(&mut obs_out, k);
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
    readout: Option<&crate::parser::model_parser::OdeOutputProgram>,
    readout_obs: Option<&ReadoutSnapshots>,
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

    // Modeled-`RATE=-1/-2` doses (#486): resolve each dose at its occasion's PK jet
    // (shared with the non-IOV walk via `resolve_eff_doses`) and build the per-dose
    // duals seeded on the stacked-η axes (η-only, the inner mirror of `run_obs_iov`).
    // Fixed subjects borrow `subject.doses` and pass no duals.
    let dose_pk: Vec<crate::types::PkParams> = (0..subject.doses.len())
        .map(|k| sources[dose_src[k]].0)
        .collect();
    let eff_doses = resolve_eff_doses(model, subject, |k| dose_pk[k]);
    // Lagtime under IOV — the inner (η-only) mirror of `run_obs_iov` (#486).
    let dose_lagtimes = dose_lagtime_values(model, &dose_pk);
    // Coincident moving breaks carrying different jets — and, under a lagtime, a moving
    // break landing on a fixed obs/reset/cov-change time — aren't representable by the
    // single-`dt` walk; decline to FD (#486 review #2).
    if !moving_bounds_separable(model, subject, &eff_doses, &dose_lagtimes) {
        return None;
    }
    let slot_dual = |k: usize, slot: usize| -> Dual1<N> {
        let (pk, cd, group) = &sources[dose_src[k]];
        let val = pk.values.get(slot).copied().unwrap_or(0.0);
        match slot_row.get(slot).copied().flatten() {
            Some(i) => seed(cd, *group, i, val),
            None => Dual1::<N>::constant(val),
        }
    };
    let dose_inf_dual = modeled_dose_inf_duals::<Dual1<N>>(model, subject, &slot_dual);
    let dose_lag_dual = dose_lag_duals::<Dual1<N>>(model, subject, &slot_dual);
    let schedule = EventSchedule::for_subject(subject, model.pk_model, &eff_doses, &dose_lagtimes);
    let conc = event_driven_sens_with_doses_g::<Dual1<N>>(
        model.pk_model,
        subject,
        &schedule,
        &eff_doses,
        &dose_lag_dual,
        &dose_inf_dual,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    // Analytic Form C readout (#655): scratch for the per-observation dual eval.
    let mut ro_state: Vec<Dual1<N>> = Vec::new();
    let mut ro_vars: Vec<Dual1<N>> = Vec::new();
    let mut ro_stack: Vec<Dual1<N>> = Vec::new();

    let mut out = Vec::with_capacity(conc.len());
    for (j, c) in conc.iter().enumerate() {
        // Clamp the concentration jet to ≥ 0 BEFORE the readout (production ordering), then
        // apply the Form C readout unclamped — the η-only mirror of the `run_obs_iov` step.
        // The readout PK params come from the κ = 0 (BSV-only) snapshot (`readout_obs.at(j)`,
        // `group = None`) while the clamped concentration supplies the κ-bearing central amount.
        let c_clamped: Dual1<N> = if c.value < 0.0 {
            Dual1::<N>::constant(0.0)
        } else {
            *c
        };
        let c: Dual1<N> = match (readout, readout_obs) {
            (Some(ro), Some(ro_obs)) => eval_readout_at_obs::<Dual1<N>>(
                ro,
                c_clamped,
                ro_obs.at(j),
                slot_row,
                subject.obs_cov(j),
                subject.obs_times[j],
                &seed,
                &mut ro_state,
                &mut ro_vars,
                &mut ro_stack,
            ),
            _ => c_clamped,
        };
        let c = &c;
        let mut df_deta = vec![0.0; n_stacked];
        for (p, df) in df_deta.iter_mut().enumerate() {
            *df = c.grad[p];
        }
        out.push(ObsGrad {
            f: c.value,
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

    // Constant `ScalarScale` output divisor `f/k` (first-order): `f` and each `∂f/∂stacked-η`
    // divide by `k` (#486 IOV-scope parity — the inner twin of the `run_obs_iov` step).
    if let ScalingSpec::ScalarScale(k) = model.scaling {
        scale_obs_sens(&mut out, k);
    }

    // LTBS `g = ln(f)` LAST — after the per-occasion scale quotient — the inner twin of
    // `subject_sensitivities_iov`'s `apply_ltbs_transform_outer`, so the EBE gradient and the
    // outer θ/Ω/σ gradient differentiate the same `ln(f/s)` production computes (#486).
    // The transform is a per-observation scalar quotient `∂g/∂x = (∂f/∂x)/f`, identical on
    // every stacked axis, so it applies to the κ columns exactly as it does to η_bsv — there
    // is nothing occasion-specific about it and no per-group jet is needed. A no-op when
    // `log_transform` is false.
    apply_ltbs_transform_inner(&mut out, model.log_transform);

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

/// True when this subject combines a **steady-state dose with a lagtime** on the event walk,
/// which the dual walk does not serve yet (#486).
///
/// Production overlays the pre-arrival SS tail for observations falling in `[t, t + ALAG)` —
/// it recomputes them from `pk::event_driven::ss_state_at_phase_event_driven` *after* the
/// walk — and `event_driven_sens_with_doses_g` has no dual twin of that overlay. Without it
/// the walk disagrees with production in **value**, not merely in derivative, so this is a
/// hard decline to FD rather than an approximation. (The CF *static* superposition path does
/// serve SS × lagtime, via `lagged_elapsed`'s pre-arrival wrap — only the walk is affected.)
fn ss_lagtime_walk_unsupported(model: &CompiledModel, subject: &Subject) -> bool {
    model.has_lagtime() && subject.doses.iter().any(|d| d.ss)
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
/// [`crate::sens::propagate::event_driven_sens_with_doses_g`]). Empty when the model has no
/// lagtime — the walk then treats every arrival as a fixed break, exactly as before.
///
/// `slot_dual(dose_k, pk_slot)` is the same seeder [`modeled_dose_inf_duals`] takes, so both
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
///    ([`crate::sens::propagate::event_driven_sens_with_doses_g`]). An observation landing on *two*
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
    let (pd, slots) = resolve_param_derivs(model, subject, theta, eta, &pk)?;
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
    // Full `ParamDerivs` via the shared resolver, then project the η-block (the inner
    // init impulse consumes `∂p/∂η` only) — bit-identical to the former inline
    // `param_derivatives_from_prog(...).map(|pd| pd.dp_deta)` + log-normal fallback (F4).
    let (pd, slots) = resolve_param_derivs(model, subject, theta, eta, &pk)?;
    let dp_deta = pd.dp_deta;
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
    // Cheap, model/subject-only declines go **before** any dual seeding (#822 review #9): a
    // permanently-declining subject (a lagtime model with an SS dose, say) would otherwise pay a
    // full second-order dual pass on every gradient evaluation and every inner BFGS step, throw it
    // away, and then pay the FD fallback on top.
    if ss_lagtime_walk_unsupported(model, subject) {
        return None;
    }
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
    // predictor does — including alongside an EVID 3/4 reset (#486), which the walk
    // handles by zeroing the state while the static superposition path cannot.
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
    let slot_row = seed_dim_from_slots(slots);

    // Analytic Form C readout program (#650); `readout_tvcov_supported` (checked at
    // the gate) guarantees it fits the eight structural `PkDual` slots here.
    let readout = model
        .analytic_readout
        .as_ref()
        .and_then(|ar| ar.program.as_ref());
    let mut sens = const_dispatch!(
        m_dim;
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24;
        |M| run_obs_tvcov::<M>(
            model, subject, theta, eta, prog, &slot_row, n_eta, n_theta, readout,
        )
    )?;

    // Analytic `[initial_conditions]` impulse (#486): layer `A₀·kernel(t, pk)` and its exact
    // `(θ,η)` jet onto every pre-reset observation BEFORE scaling — the same insertion order as
    // the f64 `pk::add_analytical_init` on the production TV-cov path (`compute_predictions_with_tv`).
    apply_tvcov_init_outer(&mut sens, model, subject, theta, eta)?;

    // Constant `ScalarScale` output divisor `f_scaled = f/k`: every derivative is
    // linear in `f` and `k` is constant, so the whole jet divides by `k` — matches
    // `pk::apply_scaling` (`pred /= s`) on the production TV-cov path.
    if let ScalingSpec::ScalarScale(k) = model.scaling {
        scale_obs_sens(&mut sens.obs, k);
    }
    // η-dependent `ExpressionScale` divisor `s(θ, η)`: apply the subject-static quotient
    // `scaled_f = f/s` on the walked jet — the SAME shared quotient the dose-superposition
    // path uses (#486, closing the TV-cov + expression-scale gap). Production
    // `pk::apply_scaling` evaluates the scale once at the subject-static covariates and
    // `t = 0` params, mirrored inside the helper.
    apply_event_walk_expression_scale_outer(&mut sens, model, subject, prog, theta, eta)?;
    // LTBS: transform the walked (and scaled) jet to `g = ln(f)` LAST, via the same shared
    // helper the dose-superposition outer uses — so the two paths cannot drift, and the
    // scale-then-log order matches production `ln(f/s)` (#486). The inner counterpart is
    // `apply_ltbs_transform_inner`, applied last by the matching inner walk (#673/#486).
    apply_ltbs_transform_outer(&mut sens, model.log_transform);
    // FREM covariate pseudo-observations are not PK rows at all — overwrite their jet.
    apply_frem_pseudo_obs_jet(&mut sens, model, subject, theta, eta);
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
    // Lagtime on the event walk (#486): the schedule shifts each arrival to `t + ALAG`
    // (production parity) and the walk threads `∂ALAG` through the dual sub-interval
    // lengths, so the arrival is a moving boundary like a modeled infusion end.
    let dose_lagtimes = dose_lagtime_values(model, &dose_pk);
    // Coincident moving breaks carrying different jets — and, under a lagtime, a moving
    // break landing on a fixed obs/reset/cov-change time — aren't representable by the
    // single-`dt` walk; decline to FD (#486 review #2).
    if !moving_bounds_separable(model, subject, &eff_doses, &dose_lagtimes) {
        return None;
    }
    let schedule = EventSchedule::for_subject(subject, model.pk_model, &eff_doses, &dose_lagtimes);
    let slot_dual = |k: usize, slot: usize| -> Dual2<M> {
        let cov = subject.dose_cov(k);
        let _guard =
            crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, subject.doses[k].time);
        let pd = pd_from_program::<M>(prog, model, cov, theta, eta);
        pk_slot_dual_outer::<M>(
            slot,
            dose_pk[k].values.get(slot).copied().unwrap_or(0.0),
            &pd,
            slots,
            n_theta,
            n_eta,
        )
    };
    let dose_inf_dual = modeled_dose_inf_duals::<Dual2<M>>(model, subject, &slot_dual);
    let dose_lag_dual = dose_lag_duals::<Dual2<M>>(model, subject, &slot_dual);
    let conc = event_driven_sens_with_doses_g::<Dual2<M>>(
        model.pk_model,
        subject,
        &schedule,
        &eff_doses,
        &dose_lag_dual,
        &dose_inf_dual,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    // Readout params past the eight structural slots (#486). The walk's `PkDual` carries only
    // CL/V/Q/V2/KA/F/Q3/V3, so a readout referencing a 9th+ individual parameter (a 3-cpt-oral
    // sigmoid-Emax `y = <expr>`, say) used to be gated out of the walk entirely. Materialize
    // those slots at each observation's snapshot — value from `pk_param_fn`, jet from the same
    // program derivatives the structural slots are seeded from — so `eval_readout_jet` sees them
    // exactly as the static superposition path already does. Empty for a readout-free model.
    // Which slots above the eight structural ones does the program actually differentiate? Almost
    // always none — and then there is nothing to materialize, so skip the per-observation pass
    // entirely rather than running a full second-order `pd_from_program` per observation and
    // discarding it (#822 review #8). The readout never reads those slots in that case, so the
    // `_` arm's `constant(0.0)` is unobservable: `allocate_readout_extra_slots` /
    // `pk_assignment_mapping` (`parser::model_parser`) slot *every* readout-referenced
    // individual parameter, constant or not, so a parameter's own slot is always a member of
    // `ro_slots` whenever the readout actually reads it — `ro_slots` cannot be empty while a
    // live high-slot read exists (#822 review #1, refuted: see
    // `form_c_tvcov_readout_constant_param_matches_fd`, which forces a constant parameter past
    // `V3` and confirms `ro_slots` is non-empty there).
    let ro_slots: Vec<usize> = if readout.is_some() {
        (PK_IDX_V3 + 1..N_PK)
            .filter(|&s| slot_row[s].is_some())
            .collect()
    } else {
        Vec::new()
    };
    let ro_extra: Vec<(crate::types::PkParams, Vec<(usize, Dual2<M>)>)> = if ro_slots.is_empty() {
        Vec::new()
    } else {
        (0..subject.obs_times.len())
            .map(|j| {
                let cov = subject.obs_cov(j);
                let t = subject.obs_times[j];
                let _guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, t);
                let pk = (model.pk_param_fn)(theta, eta, cov, t);
                let pd = pd_from_program::<M>(prog, model, cov, theta, eta);
                let extras = ro_slots
                    .iter()
                    .map(|&s| {
                        let d = pk_slot_dual_outer::<M>(
                            s,
                            pk.values.get(s).copied().unwrap_or(0.0),
                            &pd,
                            slots,
                            n_theta,
                            n_eta,
                        );
                        (s, d)
                    })
                    .collect();
                (pk, extras)
            })
            .collect()
    };

    // Analytic Form C readout (#650): scratch for the per-observation dual eval.
    let mut ro_state: Vec<Dual2<M>> = Vec::new();
    let mut ro_vars: Vec<Dual2<M>> = Vec::new();
    let mut ro_stack: Vec<Dual2<M>> = Vec::new();

    let mut obs_out = Vec::with_capacity(conc.len());
    for (j, c) in conc.iter().enumerate() {
        // Clamp the concentration jet to ≥ 0 BEFORE any readout — production clamps
        // `conc.max(0.0)` and only then applies the readout (returned unclamped). For a
        // plain model this reproduces the old post-loop clamp (`constant(0.0)` zeros grad/hess).
        let c_clamped: Dual2<M> = if c.value < 0.0 {
            Dual2::<M>::constant(0.0)
        } else {
            *c
        };
        // Analytic Form C readout: replace the (clamped) central concentration jet with
        // `y = <expr>`. The per-observation PK duals (`pk_at_obs[j]`) already carry each
        // parameter's `∂/∂(θ,η)` in the walk basis — including a non-structural `BMAX`/`KD`
        // seeded into its allocated structural slot — so the readout's derivatives compose
        // directly. The central amount is `concentration × V`; the output is not re-clamped.
        let c: Dual2<M> = if let Some(ro) = readout {
            let pkd = &pk_at_obs[j];
            // Flat PK-slot dual vector for `eval_output_g`'s `indiv_to_pk` lookups: the eight
            // structural slots come from the walk's `PkDual`; a higher slot the program
            // differentiates comes from `ro_extra` (seeded on the same `(θ,η)` axes), and any
            // other higher slot is its static value — not zero, which is what the walk used to
            // substitute (#486).
            let extra = ro_extra.get(j);
            let params: [Dual2<M>; N_PK] = std::array::from_fn(|s| match s {
                PK_IDX_CL => pkd.cl,
                PK_IDX_V => pkd.v,
                PK_IDX_Q => pkd.q,
                PK_IDX_V2 => pkd.v2,
                PK_IDX_KA => pkd.ka,
                PK_IDX_F => pkd.f,
                PK_IDX_Q3 => pkd.q3,
                PK_IDX_V3 => pkd.v3,
                // Empty when the readout differentiates no slot up here — it then never reads
                // them, so the zero is unobservable (see `ro_slots`).
                _ => match extra {
                    Some((ro_pk, ro_ext)) => ro_ext
                        .iter()
                        .find(|&&(slot, _)| slot == s)
                        .map(|&(_, d)| d)
                        .unwrap_or_else(|| {
                            Dual2::<M>::constant(ro_pk.values.get(s).copied().unwrap_or(0.0))
                        }),
                    None => Dual2::<M>::constant(0.0),
                },
            });
            // A `TIME`-referencing readout resolves `Op::PushTime` from the model-time
            // thread-local; enter this observation's time to match production's
            // `apply_analytic_readout` guard (no-op for a `TIME`-free readout).
            let _time_guard =
                crate::parser::model_parser::ModelTimeGuard::enter(subject.obs_times[j]);
            eval_readout_jet::<Dual2<M>>(
                ro,
                c_clamped,
                &params,
                subject.obs_cov(j),
                &mut ro_state,
                &mut ro_vars,
                &mut ro_stack,
            )
        } else {
            c_clamped
        };
        obs_out.push(obs_sens_from_dual2::<M>(&c, n_theta, n_eta));
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
    // Cheap, model/subject-only declines go **before** any dual seeding (#822 review #9): a
    // permanently-declining subject (a lagtime model with an SS dose, say) would otherwise pay a
    // full second-order dual pass on every gradient evaluation and every inner BFGS step, throw it
    // away, and then pay the FD fallback on top.
    if ss_lagtime_walk_unsupported(model, subject) {
        return None;
    }
    if !tvcov_analytical_supported(model) || !subject_routes_to_event_walk(model, subject) {
        return None;
    }
    // LTBS on the TV-cov inner walk is now analytic (#665 follow-up): the `g = ln(f)` jet
    // is applied LAST below (after the init impulse and scale quotient), mirroring the outer
    // TV-cov path (`subject_sensitivities_tvcov` → `apply_ltbs_transform_outer`). Reproduces
    // production's scale-then-log order; the covariance step reconverges these EBEs at the
    // tighter `cov_inner_tol` (closed-form LTBS, IOV included since #486), so the SEs stay clean.
    // Analytic Form C readout (#650): served analytically on the event-walk when it
    // fits the structural `PkDual` slots (matches the outer gate); otherwise FD.
    if model.analytic_readout.is_some() && !readout_tvcov_supported(model) {
        return None;
    }
    // SS + reset is served by the walk since #486 (see `subject_sensitivities_tvcov`).
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
    let slot_row = seed_dim_from_slots(slots);

    let readout = model
        .analytic_readout
        .as_ref()
        .and_then(|ar| ar.program.as_ref());
    // Match the outer `run_obs_tvcov` cap (`m_dim = n_theta + n_eta` over `1..=24`):
    // since `n_eta ≤ m_dim`, bounding the gate at 24 (below) makes both dispatch
    // tables resolve, so the inner and outer analytic scope stay matched rather than
    // splitting to a fixed-EBE FD inner (#449 re-review #2).
    let mut out = const_dispatch!(
        n_eta;
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24;
        |N| run_obs_grad_tvcov::<N>(model, subject, theta, eta, prog, &slot_row, n_eta, cached_schedule, readout)
    )?;

    // Analytic `[initial_conditions]` impulse (#486): η-gradient only, layered BEFORE scaling —
    // the inner twin of the outer TV-cov init impulse (keeps the inner EBE gradient consistent
    // with the outer, and with production's `add_analytical_init`).
    apply_tvcov_init_inner(&mut out, model, subject, theta, eta)?;

    // Constant `ScalarScale` divisor: `∂(f/k)/∂η = (∂f/∂η)/k` (η-independent `k`),
    // matching the outer TV-cov path and `pk::apply_scaling`. (LTBS keeps the FD inner —
    // gated upstream; `ExpressionScale` is applied below.)
    if let ScalingSpec::ScalarScale(k) = model.scaling {
        scale_obs_sens(&mut out, k);
    }
    // η-dependent `ExpressionScale` divisor: the η-only quotient — the light counterpart
    // of the outer path, matching the static inner path (#486). Subject-static scale from
    // `pk`/`∂p/∂η` at `t = 0`, applied to every observation (shared helper).
    apply_event_walk_expression_scale_inner(&mut out, model, subject, prog, theta, eta)?;
    // LTBS `g = ln(f)` LAST — after the init impulse and scale quotient — mirroring the
    // outer TV-cov path (#665 follow-up). Shared inner helper; no-op when not LTBS.
    apply_ltbs_transform_inner(&mut out, model.log_transform);
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

    // Per-event program derivatives (`∂p/∂η`), evaluated **once per event** and shared by every
    // consumer below: the event `PkDual`s, the modeled-dose / lagtime slot duals, and the Form-C
    // readout's high slots.
    //
    // `param_derivatives_at_cov` declines (`None`) purely on the *program's* axis count —
    // `n_theta + n_eta` above the `disp!` cap (#449 review #1) — which is a property of `prog`,
    // never of `cov` (`ode_provider::param_derivatives_at_cov`). Its `None` is therefore uniform
    // across a subject's events, so hoisting it collapses the decline into the single `?` below
    // that drops the whole subject to FD. The consumers then take a `&ParamDerivs` and are
    // *total*: there is no longer a site that can substitute a **zero jet** for a missing
    // derivative and report the result as analytic. Two such fallbacks used to exist (the
    // modeled-dose slot dual and the readout's high slots); both were unreachable only by
    // coincidence, and a `MAX_TVCOV_AXES > MAX_ODE_AXES` cap divergence would have made them a
    // silently-wrong gradient rather than an FD fallback (#822 follow-up).
    let pd_at = |time: f64,
                 cov: &std::collections::HashMap<String, f64>|
     -> Option<crate::sens::ode_provider::ParamDerivs> {
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
        param_derivatives_at_cov(prog, model, cov, theta, eta)
    };
    let pd_dose: Vec<crate::sens::ode_provider::ParamDerivs> = (0..subject.doses.len())
        .map(|k| pd_at(subject.doses[k].time, subject.dose_cov(k)))
        .collect::<Option<Vec<_>>>()?;
    let pd_obs: Vec<crate::sens::ode_provider::ParamDerivs> = (0..subject.obs_times.len())
        .map(|j| pd_at(subject.obs_times[j], subject.obs_cov(j)))
        .collect::<Option<Vec<_>>>()?;
    let pd_pk_only: Vec<crate::sens::ode_provider::ParamDerivs> = (0..subject.pk_only_times.len())
        .map(|m| pd_at(subject.pk_only_times[m], subject.pk_only_cov(m)))
        .collect::<Option<Vec<_>>>()?;

    let mk = |pd: &crate::sens::ode_provider::ParamDerivs,
              time: f64,
              cov: &std::collections::HashMap<String, f64>|
     -> PkDual<Dual1<N>> {
        let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, time);
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

    let pk_at_dose: Vec<PkDual<Dual1<N>>> = (0..subject.doses.len())
        .map(|k| mk(&pd_dose[k], subject.doses[k].time, subject.dose_cov(k)))
        .collect();
    let pk_at_obs: Vec<PkDual<Dual1<N>>> = (0..subject.obs_times.len())
        .map(|j| mk(&pd_obs[j], subject.obs_times[j], subject.obs_cov(j)))
        .collect();
    let pk_at_pk_only: Vec<PkDual<Dual1<N>>> = (0..subject.pk_only_times.len())
        .map(|m| {
            mk(
                &pd_pk_only[m],
                subject.pk_only_times[m],
                subject.pk_only_cov(m),
            )
        })
        .collect();

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
    let dose_lagtimes = dose_lagtime_values(model, &dose_pk);
    // Coincident moving breaks carrying different jets — and, under a lagtime, a moving
    // break landing on a fixed obs/reset/cov-change time — aren't representable by the
    // single-`dt` walk; decline to FD (#486 review #2).
    if !moving_bounds_separable(model, subject, &eff_doses, &dose_lagtimes) {
        return None;
    }
    // Reads the dose's already-computed `∂p/∂η` — no program re-evaluation, and no `None` arm
    // to swallow (see the `pd_dose` hoist above).
    let slot_dual = |k: usize, slot: usize| -> Dual1<N> {
        let val = dose_pk[k].values.get(slot).copied().unwrap_or(0.0);
        pk_slot_dual_inner::<N>(slot, val, &pd_dose[k].dp_deta, slots, n_eta)
    };
    let dose_inf_dual = modeled_dose_inf_duals::<Dual1<N>>(model, subject, &slot_dual);
    let dose_lag_dual = dose_lag_duals::<Dual1<N>>(model, subject, &slot_dual);

    // The event schedule is invariant across inner BFGS steps for a fixed subject (it
    // depends only on the subject + doses + lagtimes, not on η). Reuse the schedule
    // the inner optimizer cached once per subject when available; rebuild locally for
    // modeled doses (window depends on η), for an **η-dependent lagtime** (which moves
    // every arrival across inner steps — the same staleness rule `inner_optimizer`'s
    // own cache applies), or when no cache was passed (#449 / #486).
    let owned_schedule;
    let schedule: &EventSchedule = match cached_schedule {
        Some(s) if subject.all_doses_fixed() && !model.has_lagtime() => s,
        _ => {
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
        &dose_lag_dual,
        &dose_inf_dual,
        &pk_at_dose,
        &pk_at_obs,
        &pk_at_pk_only,
    );

    // Analytic Form C readout (#650): per-observation dual eval scratch (Dual1).
    // Readout params past the eight structural slots — the inner (`Dual1`, η-only) mirror of
    // `run_obs_tvcov`'s `ro_extra` (#486). `ro_slots` is empty when the readout differentiates
    // nothing above `PK_IDX_V3`, in which case the per-observation seeding below is a no-op;
    // the derivatives it seeds from are `pd_obs[j]`, already computed once for the walk.
    let ro_slots: Vec<usize> = if readout.is_some() {
        (PK_IDX_V3 + 1..N_PK)
            .filter(|&s| slot_row[s].is_some())
            .collect()
    } else {
        Vec::new()
    };
    let ro_extra: Vec<(crate::types::PkParams, Vec<(usize, Dual1<N>)>)> = if readout.is_some() {
        (0..subject.obs_times.len())
            .map(|j| {
                let cov = subject.obs_cov(j);
                let t = subject.obs_times[j];
                let _guard = crate::parser::model_parser::ModelTimeGuard::enter_if(uses_time, t);
                let pk = (model.pk_param_fn)(theta, eta, cov, t);
                let extras: Vec<(usize, Dual1<N>)> = ro_slots
                    .iter()
                    .map(|&s| {
                        let val = pk.values.get(s).copied().unwrap_or(0.0);
                        (
                            s,
                            pk_slot_dual_inner::<N>(s, val, &pd_obs[j].dp_deta, slots, n_eta),
                        )
                    })
                    .collect();
                (pk, extras)
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut ro_state: Vec<Dual1<N>> = Vec::new();
    let mut ro_vars: Vec<Dual1<N>> = Vec::new();
    let mut ro_stack: Vec<Dual1<N>> = Vec::new();

    let mut out = Vec::with_capacity(conc.len());
    for (j, c) in conc.iter().enumerate() {
        // Clamp conc ≥ 0 BEFORE the readout (production ordering), then apply the readout
        // unclamped — the η-gradient mirror of the outer `run_obs_tvcov` over `Dual1<N>`.
        // Central amount = concentration × V; `∂y/∂η` composes from the per-obs PK duals.
        let c_clamped: Dual1<N> = if c.value < 0.0 {
            Dual1::<N>::constant(0.0)
        } else {
            *c
        };
        let c: Dual1<N> = if let Some(ro) = readout {
            let pkd = &pk_at_obs[j];
            let extra = ro_extra.get(j);
            let params: [Dual1<N>; N_PK] = std::array::from_fn(|s| match s {
                PK_IDX_CL => pkd.cl,
                PK_IDX_V => pkd.v,
                PK_IDX_Q => pkd.q,
                PK_IDX_V2 => pkd.v2,
                PK_IDX_KA => pkd.ka,
                PK_IDX_F => pkd.f,
                PK_IDX_Q3 => pkd.q3,
                PK_IDX_V3 => pkd.v3,
                _ => match extra {
                    Some((ro_pk, ro_ext)) => ro_ext
                        .iter()
                        .find(|&&(slot, _)| slot == s)
                        .map(|&(_, d)| d)
                        .unwrap_or_else(|| {
                            Dual1::<N>::constant(ro_pk.values.get(s).copied().unwrap_or(0.0))
                        }),
                    None => Dual1::<N>::constant(0.0),
                },
            });
            let _time_guard =
                crate::parser::model_parser::ModelTimeGuard::enter(subject.obs_times[j]);
            eval_readout_jet::<Dual1<N>>(
                ro,
                c_clamped,
                &params,
                subject.obs_cov(j),
                &mut ro_state,
                &mut ro_vars,
                &mut ro_stack,
            )
        } else {
            c_clamped
        };
        let mut df_deta = vec![0.0; n_eta];
        for k in 0..n_eta {
            df_deta[k] = c.grad[k];
        }
        out.push(ObsGrad {
            f: c.value,
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
    let mut r = if !sens_profile_enabled() {
        subject_eta_grad_impl(model, subject, theta, eta, cached_schedule)
    } else {
        let t0 = std::time::Instant::now();
        let r = subject_eta_grad_impl(model, subject, theta, eta, cached_schedule);
        PROFILE_ETA_NANOS.fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        PROFILE_ETA_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        r
    };
    // FREM covariate pseudo-observations, inner (η-gradient) counterpart of
    // `apply_frem_pseudo_obs_jet`. Applied here rather than inside `subject_eta_grad_impl`
    // so it covers every inner route at one choke point — closed-form, TV-cov event-driven
    // and the light `Dual1` ODE walk — none of which knows the row is not a PK row.
    if let Some(g) = r.as_mut() {
        apply_frem_pseudo_obs_grad(g, model, subject, theta, eta);
    }
    r
}

/// Inner (η-gradient) counterpart of [`apply_frem_pseudo_obs_jet`]: a FREM covariate
/// pseudo-observation's prediction is `θ[ti] + η[ei]`, so `∂f/∂η = e_ei` exactly — the
/// same `{0, 1}` Jacobian `inner_optimizer::overwrite_frem_pseudo_obs_rows` already
/// stamps into the FOCE H-matrix. Without it the inner EBE gradient differentiates the PK
/// model on a row the objective scores against a covariate, so the EBEs themselves are
/// wrong. A no-op for a non-FREM model or a subject with no pseudo-observations.
fn apply_frem_pseudo_obs_grad(
    sens: &mut [ObsGrad],
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) {
    let Some(fc) = model.frem_config.as_ref() else {
        return;
    };
    if subject.fremtype.is_empty() {
        return;
    }
    for (j, obs) in sens.iter_mut().enumerate() {
        let ft = subject.fremtype.get(j).copied().unwrap_or(0);
        if ft == 0 {
            continue;
        }
        let Some(&(ti, ei)) = fc.fremtype_to_indices.get(&ft) else {
            continue;
        };
        obs.f = theta.get(ti).copied().unwrap_or(0.0) + eta.get(ei).copied().unwrap_or(0.0);
        let n_eta = obs.df_deta.len();
        obs.df_deta.iter_mut().for_each(|v| *v = 0.0);
        if ei < n_eta {
            obs.df_deta[ei] = 1.0;
        }
    }
}

fn subject_eta_grad_impl(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    cached_schedule: Option<&crate::pk::event_driven::EventSchedule>,
) -> Option<Vec<ObsGrad>> {
    // Transit subject the closed form can't serve → its ODE `transit()` equivalent, which
    // takes the ODE-provider branch below (that branch ignores the analytical
    // `cached_schedule`, so a stale one is harmless). Unchanged for every other model (#486).
    let model = crate::pk::effective_model_for_eval(model, subject, theta, eta);
    // ODE models: the light `Dual1` inner η-gradient (#410), gated by the master
    // switch. Out-of-scope ODE subjects decline (→ FD inner), the same per-subject
    // scope the outer provider uses, so inner and outer stay on the same route.
    if model.ode_spec.is_some() {
        if ODE_SENS_ENABLED {
            // Closed-form modified-release fast path (#860 Phase A6): a *static*
            // multi-route subject `mr_scope` admits skips the ODE integration
            // entirely (same "analytic" report as `ode_subject_eta_grad` below —
            // `ode_analytical_supported` already covers this model — just a
            // faster computation of the identical jet). `None` falls through to
            // the ODE provider unchanged.
            if let Some(g) =
                crate::pk::modified_release::mr_subject_eta_grad(model, subject, theta, eta)
            {
                return Some(g);
            }
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
    // SS + EVID 3/4 reset on a model the walk CANNOT propagate (transit/IG: `supports_event_
    // driven` is false, so `subject_routes_to_event_walk` returns false before its own
    // `has_resets() && any ss` clause is ever consulted) still reaches the static superposition
    // path here — which cannot express it. Decline. This guard is NOT dead code.
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
    // LTBS + η-dependent `ExpressionScale` is now analytic on the inner loop too (#665
    // follow-up): the `ExpressionScale` η-quotient is applied below, then the `g = ln(f)`
    // jet LAST — reproducing production's scale-then-log order `ln(f/s)`, mirroring the
    // outer path (`apply_expression_scale_outer` → `apply_ltbs_transform_outer`). The
    // covariance step reconverges these EBEs at the tighter `cov_inner_tol` (LTBS is a
    // closed-form non-IOV path here), so the ~1e-9 provider-vs-`compute_predictions`
    // offset does not corrupt the Hessian.

    let n_eta = model.n_eta;
    let oral = matches!(
        model.pk_model,
        PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral
    );
    let two_cpt = matches!(model.pk_model, PkModel::TwoCptIv | PkModel::TwoCptOral);
    let three_cpt = matches!(model.pk_model, PkModel::ThreeCptIv | PkModel::ThreeCptOral);
    let transit = matches!(model.pk_model, PkModel::OneCptTransit);
    let two_cpt_transit = matches!(model.pk_model, PkModel::TwoCptTransit);
    let ig = matches!(model.pk_model, PkModel::OneCptIg);
    let two_cpt_ig = matches!(model.pk_model, PkModel::TwoCptIg);

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
    let seed_dim = seed_dim_from_slots(&slots);

    // Analytic Form C readout program (#650); the gate guarantees it is dual-
    // evaluable and depot-free, so the inner `run_obs_grad` serves it exactly.
    let readout = model
        .analytic_readout
        .as_ref()
        .and_then(|ar| ar.program.as_ref());
    let mut out = const_dispatch!(
        slots.len();
        1, 2, 3, 4, 5, 6, 7, 8, 9;
        |N| Some(run_obs_grad::<N>(
            &seed_dim, &pk, oral, two_cpt, three_cpt, transit, two_cpt_transit, ig,
            two_cpt_ig, subject, &dp_deta, n_eta, readout,
        ))
    )?;
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
        scale_obs_sens(&mut out, k);
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
    // LTBS: `g = ln(f)`, applied LAST (after any scale quotient) via the shared inner
    // helper — same floor-then-log as the f64 predictor / ODE dual walk (#451 review #5).
    apply_ltbs_transform_inner(&mut out, model.log_transform);
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
    two_cpt_transit: bool,
    ig: bool,
    two_cpt_ig: bool,
    subject: &Subject,
    dp_deta: &[Vec<f64>],
    n_eta: usize,
    readout: Option<&crate::parser::model_parser::OdeOutputProgram>,
) -> Vec<ObsGrad> {
    let seeded = SeededPkDuals::<Dual1<N>>::seed(seed_dim, pk);
    let flags = PkClassFlags {
        oral,
        two_cpt,
        three_cpt,
        transit,
        two_cpt_transit,
        ig,
        two_cpt_ig,
    };

    // Analytic Form C readout (#650): PK-slot dual vector + scratch, built once
    // (subject-static). Mirrors the outer `run_obs`, on `Dual1<N>` (η-grad only).
    let ro_pk_duals: Option<[Dual1<N>; N_PK]> =
        readout.map(|_| ro_pk_dual_array::<Dual1<N>>(seed_dim, pk));
    let mut ro_state: Vec<Dual1<N>> = Vec::new();
    let mut ro_vars: Vec<Dual1<N>> = Vec::new();
    let mut ro_stack: Vec<Dual1<N>> = Vec::new();
    let mut out = Vec::with_capacity(subject.obs_times.len());
    for (obs_i, &t_obs) in subject.obs_times.iter().enumerate() {
        let reset_floor = reset_floor_at(subject, t_obs);
        let fd = superpose_doses(&seeded, flags, subject, t_obs, reset_floor);

        // Mirror production's `conc.max(0.0)`: a negative closed-form value clamps
        // to 0, so its η-gradient is 0 there too (consistency with the objective).
        let (fval, g) = if fd.value < 0.0 {
            (0.0, [0.0; N])
        } else {
            (fd.value, fd.grad)
        };

        // Analytic Form C readout (#650): replace the central concentration with
        // `y = <expr>` over `Dual1<N>` (central amount = conc × V), mirroring the outer
        // `run_obs`. Its `∂y/∂pk` then rides the `dp_deta` chain below.
        let conc = Dual1::<N> {
            value: fval,
            grad: g,
        };
        let y = apply_readout_jet(
            conc,
            readout,
            ro_pk_duals.as_ref(),
            subject,
            obs_i,
            &mut ro_state,
            &mut ro_vars,
            &mut ro_stack,
        );
        let (fval, g) = (y.value, y.grad);

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
    let mut r = if !sens_profile_enabled() {
        subject_sensitivities_impl(model, subject, theta, eta)
    } else {
        let t0 = std::time::Instant::now();
        let r = subject_sensitivities_impl(model, subject, theta, eta);
        PROFILE_SENS_NANOS.fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        PROFILE_SENS_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        r
    };
    // FREM covariate pseudo-observations, at the one choke point that covers every
    // route `subject_sensitivities_impl` can take — the no-TV closed form, the TV-cov
    // event-driven walk (which also applies this internally so its own direct callers
    // stay correct), and the ODE provider (`ode_subject_sensitivities`), which returns
    // early from `subject_sensitivities_impl` and previously never saw this pass at all
    // (#251 review #1): an ODE FREM subject took the analytic outer gradient with the raw
    // PK jet on pseudo-obs rows while `score_core` still overrode their variance to
    // `EPSCOV²`, mismatching jet and variance by orders of magnitude.
    if let Some(sens) = r.as_mut() {
        apply_frem_pseudo_obs_jet(sens, model, subject, theta, eta);
    }
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

fn subject_sensitivities_impl(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<SubjectSens> {
    // A `one_cpt_transit` subject the closed form can't serve (TIME switch / TV covariates)
    // routes to its exact ODE `transit()` equivalent, which then takes the ODE-provider
    // branch below; every other model is unchanged (#486).
    let model = crate::pk::effective_model_for_eval(model, subject, theta, eta);
    // ODE models route to the ODE sensitivity provider (issue #367, Option A;
    // armed in #410) when in its supported scope; out-of-scope ODE subjects return
    // `None` and fall back to the prior path (gradient-free outer, FD inner). The
    // `ODE_SENS_ENABLED` master switch stays as a single kill-switch for the path.
    if model.ode_spec.is_some() {
        if ODE_SENS_ENABLED {
            // Closed-form modified-release fast path (#860 Phase A6): try the
            // no-integration superposition first for the *static* subjects
            // `mr_scope` admits — a pure speed swap inside the branch
            // `ode_analytical_supported` already reports "analytic" for, not a
            // new supported/declined predicate (so `sens_supported` /
            // `analytic_outer_gradient_available` need no change). `None` (out
            // of `mr_scope`'s narrower static scope, or an axis-count mismatch)
            // falls through to the ODE provider unchanged.
            if let Some(sens) =
                crate::pk::modified_release::mr_subject_sensitivities(model, subject, theta, eta)
            {
                return Some(sens);
            }
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
    // the exact `∂f/∂pk` through the same closed forms. Steady-state doses assume an infinite
    // periodic history that a mid-record reset contradicts, so a subject mixing SS with resets
    // falls back to FD **here**. This is not dead code: `subject_routes_to_event_walk` bails to
    // `false` for a model the walk cannot state-propagate (transit/IG — `supports_event_driven`)
    // *before* it ever consults its own `has_resets() && any ss` clause, so such a subject lands
    // on this dose-superposition path, which cannot express SS + reset.
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
    let two_cpt_transit = matches!(model.pk_model, PkModel::TwoCptTransit);
    let ig = matches!(model.pk_model, PkModel::OneCptIg);
    let two_cpt_ig = matches!(model.pk_model, PkModel::TwoCptIg);

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
    let (pd, slots) = resolve_param_derivs(model, subject, theta, eta, &pk)?;

    // Right-size the dual width to the number of differentiated PK parameters
    // (issue #367): seed only those on a compact `0..N`, so a 2-cpt IV model runs
    // `Dual2<4>` (4× fewer Hessian entries per op than the former fixed `Dual2<8>`),
    // 3-cpt `Dual2<6>`, etc. `seed_dim[s]` is the compact dual axis for PK slot `s`
    // (`None` = constant, not differentiated); `pd` row `i` ↔ compact axis `i`.
    let seed_dim = seed_dim_from_slots(&slots);

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
            // Transit / IG have no hand-written explicit kernel; use the generic Dual2 path.
            PkModel::OneCptTransit
            | PkModel::TwoCptTransit
            | PkModel::OneCptIg
            | PkModel::TwoCptIg => None,
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
    let mut sens = const_dispatch!(
        slots.len();
        1, 2, 3, 4, 5, 6, 7, 8, 9;
        |N| Some(SubjectSens {
            obs: run_obs::<N>(
                &seed_dim, &pk, oral, two_cpt, three_cpt, transit, two_cpt_transit, ig,
                two_cpt_ig, explicit_kind, subject, &pd, n_eta, n_theta, readout,
            ),
        })
    )?;
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
        scale_obs_sens(&mut sens.obs, k);
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
    // LTBS: transform the full jet to `g = ln(f)` (after scaling), via the shared helper
    // that the TV-cov event-driven outer (`subject_sensitivities_tvcov`) also uses. Applied
    // last, matching production's scale-then-log order.
    apply_ltbs_transform_outer(&mut sens, model.log_transform);
    Some(sens)
}

/// Replace the jet of every FREM **covariate pseudo-observation** row (`fremtype > 0`).
///
/// `individual_nll` does not score such a row against the PK model at all: its prediction
/// is `θ[theta_idx] + η[eta_idx]` and its variance is the dedicated covariate σ (`EPSCOV`),
/// not `error_spec.variance_at(f)`. The sensitivity walk knows nothing about that — it
/// returns the ordinary PK jet for every observation — so without this pass the analytic
/// outer gradient differentiates a likelihood the objective never evaluates on those rows.
///
/// The corrected row is trivially linear-Gaussian:
///
/// ```text
///   f = θ[ti] + η[ei] ,   ∂f/∂η = e_ei ,   ∂f/∂θ = e_ti ,   every 2nd derivative = 0
/// ```
///
/// The *variance* half is handled by the consumers, which read the FREM σ override rather
/// than `variance_at` for these rows (`sens_outer_gradient::score_core`); `d = ∂R/∂f` and
/// `d2` are then zero because `R` does not depend on `f` here.
///
/// Applied **after** [`apply_ltbs_transform_outer`], because a FREM row's prediction is not
/// log-transformed by the objective either (`individual_nll` takes the `fremtype` branch
/// before it ever touches `preds`), so the LTBS jet for that row must be overwritten, not
/// composed with. A no-op for a non-FREM model or a subject with no pseudo-observations.
fn apply_frem_pseudo_obs_jet(
    sens: &mut SubjectSens,
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) {
    let Some(fc) = model.frem_config.as_ref() else {
        return;
    };
    if subject.fremtype.is_empty() {
        return;
    }
    for (j, obs) in sens.obs.iter_mut().enumerate() {
        let ft = subject.fremtype.get(j).copied().unwrap_or(0);
        if ft == 0 {
            continue;
        }
        let Some(&(ti, ei)) = fc.fremtype_to_indices.get(&ft) else {
            continue;
        };
        let n_eta = obs.df_deta.len();
        let n_theta = obs.df_dtheta.len();
        obs.f = theta.get(ti).copied().unwrap_or(0.0) + eta.get(ei).copied().unwrap_or(0.0);
        obs.df_deta.iter_mut().for_each(|v| *v = 0.0);
        if ei < n_eta {
            obs.df_deta[ei] = 1.0;
        }
        obs.df_dtheta.iter_mut().for_each(|v| *v = 0.0);
        if ti < n_theta {
            obs.df_dtheta[ti] = 1.0;
        }
        obs.d2f_deta2.iter_mut().for_each(|v| *v = 0.0);
        obs.d2f_deta_dtheta.iter_mut().for_each(|v| *v = 0.0);
    }
}

/// Apply the log-transform-both-sides (LTBS) `g = ln(f)` output transform to an outer
/// [`SubjectSens`] jet in place, mirroring `pk::apply_log_transform`. With `inv = 1/f`:
///   g          = ln(f)
///   ∂g/∂x      = inv · ∂f/∂x
///   ∂²g/∂x∂y   = inv · ∂²f/∂x∂y − inv² · ∂f/∂x · ∂f/∂y     (x, y ∈ {η, θ})
/// The second derivatives read the *original* first derivatives, so they are computed
/// before `df_deta`/`df_dtheta` are overwritten. The value half goes through the shared
/// `pk::ltbs_log_g` (same floor-then-log as the f64 predictor and the ODE dual walk, #451),
/// applied after the jet is transformed; below `LTBS_FLOOR` the transform clamps to a
/// constant so all derivatives vanish. A no-op when `log_transform` is false. Shared by the
/// dose-superposition outer (`subject_sensitivities_impl`) and the TV-cov event-driven outer
/// (`subject_sensitivities_tvcov`), so the two paths cannot drift (#486). Applied last, after
/// any `ScalarScale`/`ExpressionScale` quotient, matching production's scale-then-log order.
/// [`apply_ltbs_transform_inner`] is the first-order counterpart every inner walk applies at
/// the same point, so the two loops differentiate the same `ln(f/s)` (#673/#486).
fn apply_ltbs_transform_outer(sens: &mut SubjectSens, log_transform: bool) {
    if !log_transform {
        return;
    }
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
                    o.d2f_deta2[idx] = inv * o.d2f_deta2[idx] - inv2 * o.df_deta[k] * o.df_deta[l];
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

/// Inner (η-gradient) counterpart of [`apply_ltbs_transform_outer`]: applies the
/// `g = ln(f)` jet to a `Dual1` inner gradient in place — `∂g/∂η = ∂f/∂η / f`, with the
/// same floor-then-log as the f64 predictor (`pk::ltbs_log_g`). Below `LTBS_FLOOR` the
/// transform clamps to a constant so the gradient vanishes. Applied LAST, after any scale
/// quotient (so `g = ln(f/s)` matches production's scale-then-log order). Shared by the
/// plain (`subject_eta_grad_impl`) and TV-cov (`subject_eta_grad_tvcov_with_schedule`)
/// inner paths; a no-op when `log_transform` is false.
///
/// `#[inline(never)]` keeps this a real, attributable coverage region: the coverage
/// profile inherits `release` (opt-level 3), so without it this small helper is inlined
/// into every call site and its body records zero direct executions — reading as
/// uncovered on Codecov even though the unit test and the LTBS FD-parity tests exercise
/// both branches.
#[inline(never)]
fn apply_ltbs_transform_inner(out: &mut [ObsGrad], log_transform: bool) {
    if !log_transform {
        return;
    }
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
    two_cpt_transit: bool,
    ig: bool,
    two_cpt_ig: bool,
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
    let flags = PkClassFlags {
        oral,
        two_cpt,
        three_cpt,
        transit,
        two_cpt_transit,
        ig,
        two_cpt_ig,
    };
    // Analytic Form C readout (#650): the PK-parameter dual vector (indexed by PK
    // slot, so `eval_output_g` can pull `V`, binding constants, … via its
    // `indiv_to_pk` plan) and reusable scratch. Subject-static, so built once.
    let ro_pk_duals: Option<[Dual2<N>; N_PK]> =
        readout.map(|_| ro_pk_dual_array::<Dual2<N>>(seed_dim, pk));
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
        let reset_floor = reset_floor_at(subject, t_obs);

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
            // Seed only the differentiated PK params as `Dual2<N>` and superpose the
            // dose contributions restricted to the current reset segment; `elapsed`
            // carries the lagtime shift and the SS pre-arrival tail wrap.
            let seeded = SeededPkDuals::<Dual2<N>>::seed(seed_dim, pk);
            let fd = superpose_doses(&seeded, flags, subject, t_obs, reset_floor);
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

        // Analytic Form C readout (#650): replace the central concentration jet with
        // `y = <expr>` over `Dual2<N>`. The central compartment **amount** is
        // `concentration × V` (so a `central / V` readout recovers the concentration and
        // an additive term layers on); the readout's `∂y/∂pk` / `∂²y/∂pk²` then ride the
        // same `pd` chain to `(θ, η)` below. Covariates come from the per-observation
        // snapshot (a per-row `FREE`-style flag).
        let conc = Dual2::<N> {
            value: fval,
            grad: g,
            hess: h,
        };
        let y = apply_readout_jet(
            conc,
            readout,
            ro_pk_duals.as_ref(),
            subject,
            obs_i,
            &mut ro_state,
            &mut ro_vars,
            &mut ro_stack,
        );
        let (fval, g, h) = (y.value, y.grad, y.hess);

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
#[path = "provider_tests.rs"]
mod tests;
