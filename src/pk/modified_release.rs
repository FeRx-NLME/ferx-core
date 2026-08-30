//! Closed-form modified-release absorption (#860).
//!
//! Multi-route / modified-release absorption (parallel and mixed pathways #505,
//! per-route lag #856/#859) is authored as an `[odes]` model whose central
//! compartment receives a superposition of built-in input-rate forcings
//! (`first_order`/`transit`/`igd`/`zero_order`, each optionally `FR*…` split and
//! `lag=…` shifted). Historically every such model is **ODE-integrated** — there
//! is no closed-form twin, because the fixed closed-form PK layout has a single
//! `KA` slot and every `PkModel` variant is single-route.
//!
//! This module supplies the closed form. The disposition (central [+ peripheral])
//! is a linear time-invariant system, so its inputs **superpose**:
//!
//! ```text
//!   y(t) = Σ_k  FR_k · single_route_closed_form_k( t − LAG_k )
//! ```
//!
//! each term an *existing* single-route kernel (`sens/one_cpt.rs`,
//! `sens/two_cpt.rs`), scaled by its pathway fraction and shifted by its onset.
//! No joint / pairwise closed form is ever derived — superposition into a shared
//! LTI disposition is additive, term by term. Written once, generic over
//! [`PkNum`]: `T = f64` gives the value, `T = Dual2<N>` gives the value **and**
//! the exact `∂y/∂{ka, frac, lag, cl, v, …}` FOCE/FOCEI consume — the same
//! `∂y/∂LAG_k` #859 restored, now with no integration at all.
//!
//! ## Identification (no AST / name matching)
//!
//! The disposition is recovered from the compiled `OdeSpec` **by behaviour, not
//! by spelling** — pattern-matching `-CL/V*central` (or guessing which parameter
//! is "CL") is a happy-path trap that false-positives on nonlinear or
//! reparameterised RHSs. [`identify_disposition`] instead probes the compiled RHS
//! program (which the integrator evaluates *without* the input-rate forcings —
//! those are a separate wrapper) at basis states to extract the constant Jacobian
//! `A`, verifies linearity + time-invariance + a canonical 1-/2-cpt sign pattern,
//! and reads the observable's central slope `m = ∂y/∂central` off the readout
//! program. Volumes cancel: for the Form-C convention `y = central / V` the
//! observable equals `Σ mass·kernel_conc` exactly when the kernel is evaluated
//! with `v = 1/m`, `cl = ke/m` (see [`recover_disp_params_g`]).
//!
//! Identification is deliberately **conservative** — it declines on any
//! nonlinearity, time-variance, non-canonical sign pattern, or non-central /
//! nonlinear readout — and the ODE path is always a correct fallback, so an
//! un-admitted model is only ever *slower*, never wrong. The remaining risk (a
//! model that identifies but whose parameters are recovered wrongly) is closed
//! by the **reduction-to-ODE test anchor**: the admitted closed form is asserted
//! bit-for-bit (to solver tolerance) against its own integrated `OdeSpec` twin
//! across a model zoo, and transitively against the NONMEM `$DES` anchor the ODE
//! path already matches. [`mr_predictions`] is the routing entry; it returns
//! `None` (→ ODE) for anything outside scope.

use crate::ode::predictions::OdeSpec;
use crate::pk::absorption::{InputRateForcing, InputRateKind};
use crate::sens::num::PkNum;
use crate::sens::one_cpt::{one_cpt_ig_g, one_cpt_oral_g, one_cpt_transit_g, one_cpt_zero_order_g};
use crate::sens::two_cpt::{two_cpt_ig_g, two_cpt_oral_g, two_cpt_transit_g, two_cpt_zero_order_g};
use crate::types::DoseEvent;

/// Linear disposition shape recovered from an [`OdeSpec`] by numeric probing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MrDispositionKind {
    /// Single central compartment (`d central/dt = R_in − ke·central`).
    OneCpt,
    /// Central + one peripheral (`k12`/`k21` reversible coupling).
    TwoCpt,
}

/// The identified disposition: its shape plus the state indices of central and
/// (for two-compartment) peripheral. Produced by [`identify_disposition`] from
/// the immutable [`OdeSpec`] structure, so it is **model-invariant** (independent
/// of subject, `θ`, `η`). It is currently recovered afresh on each [`mr_scope`]
/// call rather than cached on the model — a memoisation opportunity, since the
/// numeric Jacobian probing repeats every value / gradient evaluation (tracked as
/// a follow-up; the fast path still avoids all ODE integration, so the probing is
/// a second-order cost). The per-subject rates are recovered separately over `T`
/// by [`recover_disp_params_g`] so `T = Dual2` carries their sensitivities.
#[derive(Debug, Clone, Copy)]
pub struct MrDisposition {
    pub kind: MrDispositionKind,
    /// State index of the central compartment (the forcings' target).
    pub central: usize,
    /// State index of the peripheral compartment (two-compartment only).
    pub periph: Option<usize>,
}

/// Disposition parameters synthesised at a parameter point, over `T` so a
/// `Dual2` carries `∂/∂p`. Parameterised in the `cl`/`v`(/`q`/`v2`) form the
/// existing kernels expect; `v = 1/m` and `cl = ke/m` fold the observable slope
/// in so `Σ mass·kernel_conc` *is* the observable (volumes cancel — see module
/// docs).
#[derive(Debug, Clone, Copy)]
pub struct DispParams<T: PkNum> {
    pub cl: T,
    pub v: T,
    /// `(q, v2)` for two-compartment; `None` for one-compartment.
    pub periph: Option<(T, T)>,
}

/// Tolerances for the numeric structure checks. Loose enough to admit honest
/// floating-point noise in the Jacobian probes, tight enough that a genuinely
/// nonlinear or non-canonical RHS fails. These only pick *which* closed form to
/// try; what makes admitting one safe is the scope gate in [`mr_scope`] (which
/// rejects, statically, the classes these numeric probes cannot decide) plus
/// [`verify_against_ode_twin`], which re-checks every admitted subject against
/// its integrated twin in debug builds.
///
/// Until #1124 this comment named a "runtime verify-gate" as the real safety
/// net. There was no such gate — three comments referred to it and nothing
/// implemented it, which is how a `TAD`-reading RHS reached production 8× wrong.
/// It exists now; keep it that way, and do not loosen these tolerances on the
/// strength of a release-build guarantee it does not provide.
const LINEARITY_TOL: f64 = 1e-7;
/// A cross-coupling rate below this is treated as "absent" (distinguishes a
/// two-compartment `k12`/`k21` pair from a one-way depot→central drain).
const COUPLING_EPS: f64 = 1e-9;

/// Probe the forcing-free disposition RHS `du = f(u, p, t)` in `f64` at state
/// `u`. Returns `None` when the spec has no compiled RHS program (hand-built
/// specs / out of analytic scope). The forcings are applied by a separate
/// integrator wrapper, so the program yields the pure linear disposition.
fn probe_rhs_f64(spec: &OdeSpec, u: &[f64], p: &[f64], t: f64) -> Option<Vec<f64>> {
    let prog = spec.rhs_program.as_ref()?;
    let mut du = vec![0.0; spec.n_states];
    let mut vars: Vec<f64> = Vec::new();
    let mut stack: Vec<f64> = Vec::new();
    // `tad` is pinned to `0.0` — deliberately, and it is NOT safe on its own.
    //
    // The comment that used to sit here claimed "a supported (static) disposition
    // is autonomous so their value is irrelevant", which assumes the very thing
    // `identify_disposition`'s two-time probe exists to establish. For `TAD` the
    // assumption is circular and was false: pinning it made a `TAD`-reading RHS
    // evaluate identically at both probe times, so it was admitted as
    // time-invariant and its `TAD` term vanished from the predictions (#1124).
    //
    // Varying `tad` here does not fix that, and makes things worse — measured:
    // passing `t` for both `tafd` and `tad` lets `TAFD − TAD` cancel to a
    // constant, flipping a correctly-*declined* RHS to wrongly *admitted*, while
    // a `TAD` read from an untaken `if` branch still slips past (two probe times
    // never enter it). The behavioural probe cannot decide this question at all.
    //
    // It is settled statically instead, in `mr_scope`, which declines any RHS
    // that reads model time in any spelling before identification is attempted —
    // so no `TAD`-reading program ever reaches this probe, and the pin is inert.
    prog.eval_rhs_g::<f64>(u, p, t, t, 0.0, &mut du, &mut vars, &mut stack);
    Some(du)
}

/// Identify the linear disposition of `spec` by numeric probing — its shape and
/// the central/peripheral state indices — or `None` if it is not a canonical
/// linear 1-/2-cpt system with all input-rate forcings into one central
/// compartment. Structure only; per-subject rates come from
/// [`recover_disp_params_g`]. `p` is a representative (positive) parameter point
/// and `t` a probe time; the checks are parameter-point-agnostic in structure,
/// and [`verify_against_ode_twin`] re-validates numerically in debug builds.
///
/// The probe is behavioural by design, but it cannot decide **model-time
/// dependence**: it samples two times with `tad` pinned, so a `TAD` term is
/// invisible and any time variable inside an untaken `if` branch is unreachable.
/// [`mr_scope`] rejects those statically before calling this (#1124).
pub fn identify_disposition(spec: &OdeSpec, p: &[f64], t: f64) -> Option<MrDisposition> {
    // Must have a differentiable RHS + a dual-evaluable readout (else no analytic
    // path exists anyway).
    spec.rhs_program.as_ref()?;
    let ro = spec.readout_program.as_ref()?;
    if !ro.is_dual_evaluable() {
        return None;
    }
    let n = spec.n_states;
    if n == 0 || n > 2 {
        return None;
    }

    // All forcings must target the same compartment — that is "central".
    let mut cmts = spec.input_rate.iter().map(|f| f.cmt);
    let central = cmts.next()?;
    if !cmts.all(|c| c == central) || central >= n {
        return None;
    }

    // No autonomous term: f(0, p, t) ≈ 0. The `!is_finite()` guard is load-bearing:
    // a `NaN` probe (a pathological mid-fit param excursion, e.g. `V → 0`) would
    // slip past a bare `d.abs() > TOL` because `NaN > TOL` is false — decline it so
    // the fast path never serves NaN where the sign/linearity checks below (also
    // `.abs() > …` comparisons) would silently admit it.
    let du0 = probe_rhs_f64(spec, &vec![0.0; n], p, t)?;
    if du0
        .iter()
        .any(|d| !d.is_finite() || d.abs() > LINEARITY_TOL)
    {
        return None;
    }

    // Linear + time-invariant: f(2·e_i) = 2·f(e_i), and equal at two times.
    let t2 = t + 1.0;
    let mut basis = vec![0.0; n];
    let mut jac = vec![vec![0.0; n]; n]; // jac[i] = ∂f/∂u_i column (= A·e_i)
    for i in 0..n {
        basis[i] = 1.0;
        let ei = probe_rhs_f64(spec, &basis, p, t)?;
        basis[i] = 2.0;
        let two_ei = probe_rhs_f64(spec, &basis, p, t)?;
        basis[i] = 1.0;
        let ei_t2 = probe_rhs_f64(spec, &basis, p, t2)?;
        basis[i] = 0.0;
        for j in 0..n {
            if !ei[j].is_finite() || !two_ei[j].is_finite() || !ei_t2[j].is_finite() {
                return None; // NaN/Inf probe — see the `du0` guard above
            }
            if (two_ei[j] - 2.0 * ei[j]).abs() > LINEARITY_TOL * (1.0 + two_ei[j].abs()) {
                return None; // nonlinear in u_i
            }
            if (ei_t2[j] - ei[j]).abs() > LINEARITY_TOL * (1.0 + ei[j].abs()) {
                return None; // time-varying
            }
            jac[i][j] = ei[j];
        }
    }

    // Readout must depend on central only, linearly.
    for s in 0..n {
        if s != central && ro.references_state(s) {
            return None;
        }
    }
    if !ro.references_state(central) {
        return None;
    }

    match n {
        1 => {
            // central self-rate must be elimination (< 0).
            if jac[central][central] >= -COUPLING_EPS {
                return None;
            }
            Some(MrDisposition {
                kind: MrDispositionKind::OneCpt,
                central,
                periph: None,
            })
        }
        2 => {
            let periph = (0..n).find(|&s| s != central)?;
            // A (amounts):  dA_c = −(k10+k12)·A_c + k21·A_p ;  dA_p = k12·A_c − k21·A_p
            let k12 = jac[central][periph]; // central → periph
            let k21 = jac[periph][central]; // periph → central
            let c_self = jac[central][central]; // −(k10+k12)
            let p_self = jac[periph][periph]; // −k21
                                              // Canonical reversible two-compartment sign pattern, with *both*
                                              // cross-couplings present (a one-way depot→central drain, k21≈0, is
                                              // NOT two-compartment — it is an oral depot-state model with its own
                                              // analytic path, and must be declined here).
            if k12 <= COUPLING_EPS || k21 <= COUPLING_EPS {
                return None;
            }
            if c_self >= 0.0 || p_self >= 0.0 {
                return None;
            }
            // Peripheral self-rate must be −k21 (mass-conserving coupling).
            if (p_self + k21).abs() > LINEARITY_TOL * (1.0 + k21.abs()) {
                return None;
            }
            // Implied elimination k10 = −(c_self) − k12 must be non-negative.
            if -c_self - k12 < -COUPLING_EPS {
                return None;
            }
            Some(MrDisposition {
                kind: MrDispositionKind::TwoCpt,
                central,
                periph: Some(periph),
            })
        }
        _ => None,
    }
}

/// Recover the disposition parameters at parameter point `p` over `T`, so a
/// `Dual2` carries their `∂/∂(θ,η)`. Uses the compiled RHS program (disposition
/// rates from the Jacobian) and readout program (central slope `m`), then folds
/// the volume in as `v = 1/m`, `cl = ke/m` (`q = k12/m`, `v2 = q/k21` for
/// two-compartment). Returns `None` if the readout slope is non-positive
/// (degenerate) — the caller ([`mr_scope`]) then declines to the ODE path.
pub fn recover_disp_params_g<T: PkNum>(
    spec: &OdeSpec,
    disp: &MrDisposition,
    p: &[T],
    cov: &std::collections::HashMap<String, f64>,
) -> Option<DispParams<T>> {
    let prog = spec.rhs_program.as_ref()?;
    let ro = spec.readout_program.as_ref()?;
    // A `TIME`-reading readout (`y = central/V + BETA*TIME/(TIME+T50)`, now that
    // #1028 makes `TIME` resolve per observation) is not a function of the state
    // alone, so the linearity probe below cannot characterise it: the probe
    // evaluates the readout under whatever the *ambient* model-time thread-local
    // holds — `0.0` at this gate — which makes an additive time term vanish, passes
    // `y0 ≈ 0`, and would let `mr_observable_g`'s `m·amount` silently drop the term
    // for every MR-scoped subject. It would also make scope membership depend on
    // ambient thread-local state (a non-zero ambient time flips the same model from
    // admitted to declined). Decline outright and take the ODE path, which enters
    // the per-observation guard (#1028).
    if ro.reads_time_builtin() {
        return None;
    }
    let n = spec.n_states;
    let mut vars: Vec<T> = Vec::new();
    let mut stack: Vec<T> = Vec::new();

    let probe = |u: &[T], vars: &mut Vec<T>, stack: &mut Vec<T>| -> Vec<T> {
        let mut du = vec![T::from_f64(0.0); n];
        prog.eval_rhs_g::<T>(u, p, 0.0, 0.0, 0.0, &mut du, vars, stack);
        du
    };

    // Central slope m = readout(e_central) − readout(0), over T.
    let zero_state = vec![T::from_f64(0.0); n];
    let mut e_c = vec![T::from_f64(0.0); n];
    e_c[disp.central] = T::from_f64(1.0);
    let mut e_c2 = vec![T::from_f64(0.0); n];
    e_c2[disp.central] = T::from_f64(2.0);
    let y0 = ro.eval_output_g::<T>(&zero_state, p, cov, &mut vars, &mut stack);
    let y1 = ro.eval_output_g::<T>(&e_c, p, cov, &mut vars, &mut stack);
    let y2 = ro.eval_output_g::<T>(&e_c2, p, cov, &mut vars, &mut stack);
    // The observable must be **linear through the origin** in the central amount:
    // `mr_observable_g` sums `mass·kernel_conc = m·amount`, so a readout with a
    // non-zero intercept (a baseline `y = central/V + BASE`) or any curvature in
    // central (`y = central²/V`) would be silently mispredicted — the `identify`
    // readout check only pins *which* state is read, not that it is read linearly.
    // Require `readout(0) ≈ 0` and `readout(2) ≈ 2·readout(1)`, else decline.
    if y0.val().abs() > LINEARITY_TOL * (1.0 + y1.val().abs()) {
        return None;
    }
    if (y2.val() - 2.0 * y1.val()).abs() > LINEARITY_TOL * (1.0 + y2.val().abs()) {
        return None;
    }
    let m = y1 - y0;
    if m.val() <= 0.0 {
        return None;
    }
    let v = T::from_f64(1.0) / m;

    let du_ec = probe(&e_c, &mut vars, &mut stack);
    match disp.kind {
        MrDispositionKind::OneCpt => {
            let ke = -du_ec[disp.central]; // du = −ke·1
            if ke.val() <= 0.0 {
                return None;
            }
            Some(DispParams {
                cl: ke * v,
                v,
                periph: None,
            })
        }
        MrDispositionKind::TwoCpt => {
            let periph = disp.periph?;
            let mut e_p = vec![T::from_f64(0.0); n];
            e_p[periph] = T::from_f64(1.0);
            let du_ep = probe(&e_p, &mut vars, &mut stack);
            let k12 = du_ec[periph]; // central → periph
            let k21 = du_ep[disp.central]; // periph → central
            let k10 = -du_ec[disp.central] - k12; // −(c_self) − k12
            if k10.val() <= 0.0 || k12.val() <= 0.0 || k21.val() <= 0.0 {
                return None;
            }
            let v1 = v;
            let cl = k10 * v1;
            let q = k12 * v1;
            let v2 = q / k21;
            Some(DispParams {
                cl,
                v: v1,
                periph: Some((q, v2)),
            })
        }
    }
}

/// Read forcing argument `i` from the flat individual-parameter vector, over `T`.
/// Mirrors the private `InputRateForcing::arg` (kept local so this module does
/// not widen that visibility).
#[inline]
fn forcing_arg<T: PkNum>(f: &InputRateForcing, p: &[T], i: usize, dflt: f64) -> T {
    f.arg_slots
        .get(i)
        .and_then(|&s| p.get(s))
        .copied()
        .unwrap_or_else(|| T::from_f64(dflt))
}

/// One route's contribution to the observable: `mass · kernel_conc(τ)`.
///
/// The kernel is evaluated at unit amount / unit `F` (its response is linear in
/// dose mass) and multiplied by `mass = FR·F·D` outside — so a `Dual2` `mass`
/// threads the exact `∂/∂FR` (and `∂/∂F`) via the product rule, and the kernel's
/// own `τ.val() < 0 → 0` guard supplies the pre-onset gate. Covers the smooth
/// kernels (`first_order`/`transit`/`igd`) and the box-car `zero_order` kernel
/// (`∂/∂dur` exact on whichever side of the `t == dur` boundary is selected,
/// see [`crate::sens::one_cpt::one_cpt_zero_order_g`]); `weibull` has no
/// elementary closed form and is excluded by the support gate, returning `0`
/// here as a defensive no-op.
fn route_conc_g<T: PkNum>(
    kind: InputRateKind,
    f: &InputRateForcing,
    p: &[T],
    dp: &DispParams<T>,
    tau: T,
    mass: T,
) -> T {
    let one = T::from_f64(1.0);
    let unit = match dp.periph {
        None => match kind {
            InputRateKind::FirstOrder => {
                let ka = forcing_arg(f, p, 0, 1.0);
                one_cpt_oral_g(1.0, tau, dp.cl, dp.v, ka, one)
            }
            InputRateKind::Transit => {
                let n = forcing_arg(f, p, 0, 0.0);
                let mtt = forcing_arg(f, p, 1, 1.0);
                one_cpt_transit_g(1.0, tau, dp.cl, dp.v, n, mtt, one)
            }
            InputRateKind::InverseGaussian => {
                let mat = forcing_arg(f, p, 0, 1.0);
                let cv2 = forcing_arg(f, p, 1, 1.0);
                one_cpt_ig_g(1.0, tau, dp.cl, dp.v, mat, cv2, one)
            }
            InputRateKind::ZeroOrder => {
                let dur = forcing_arg(f, p, 0, 1.0);
                one_cpt_zero_order_g(1.0, tau, dp.cl, dp.v, dur, one)
            }
            InputRateKind::Weibull => T::from_f64(0.0),
        },
        Some((q, v2)) => match kind {
            InputRateKind::FirstOrder => {
                let ka = forcing_arg(f, p, 0, 1.0);
                two_cpt_oral_g(1.0, tau, dp.cl, dp.v, q, v2, ka, one)
            }
            InputRateKind::Transit => {
                let n = forcing_arg(f, p, 0, 0.0);
                let mtt = forcing_arg(f, p, 1, 1.0);
                two_cpt_transit_g(1.0, tau, dp.cl, dp.v, q, v2, n, mtt, one)
            }
            InputRateKind::InverseGaussian => {
                let mat = forcing_arg(f, p, 0, 1.0);
                let cv2 = forcing_arg(f, p, 1, 1.0);
                two_cpt_ig_g(1.0, tau, dp.cl, dp.v, q, v2, mat, cv2, one)
            }
            InputRateKind::ZeroOrder => {
                let dur = forcing_arg(f, p, 0, 1.0);
                two_cpt_zero_order_g(1.0, tau, dp.cl, dp.v, q, v2, dur, one)
            }
            InputRateKind::Weibull => T::from_f64(0.0),
        },
    };
    mass * unit
}

/// The closed-form modified-release observable at time `t`, over `T`.
///
/// Superposes every route of every past dose: `Σ_dose Σ_route FR·F·D ·
/// kernel_conc(t − dose − lag_cmt − lag_route)`. `lag_cmt` is the (already
/// resolved) shared compartment lag for the central compartment; `f_bio` the
/// overall bioavailability `F` (one dose record → `F` scales the whole dose
/// before the `FR` split). Clamped at `0` like the analytic superposition. Pure
/// superposition — the disposition parameters `dp` are recovered by the caller,
/// which keeps this testable in isolation against manual kernel sums.
#[allow(clippy::too_many_arguments)]
pub fn mr_observable_g<T: PkNum>(
    dp: &DispParams<T>,
    forcings: &[InputRateForcing],
    doses: &[DoseEvent],
    t: T,
    params: &[T],
    lag_cmt: T,
    f_bio: T,
) -> T {
    let mut c = T::from_f64(0.0);
    for dose in doses {
        let base = T::from_f64(dose.time) + lag_cmt;
        for f in forcings {
            let onset = base + f.route_lag(params);
            let tau = t - onset;
            let mass = f.frac(params) * f_bio * T::from_f64(dose.amt);
            c = c + route_conc_g::<T>(f.kind, f, params, dp, tau, mass);
        }
    }
    c.guard_floor(0.0)
}

/// Whether this input-rate kind has an elementary closed-form central response
/// that can be superposed. `first_order`/`transit`/`igd` (smooth kernels) and
/// `zero_order` (box-car convolution, #860 Phase B) do; `weibull` has no
/// elementary convolution with an exponential disposition and stays on the ODE
/// path permanently. Exhaustive so a new kind forces a decision here.
pub(crate) fn is_closed_form_kind(kind: InputRateKind) -> bool {
    match kind {
        InputRateKind::FirstOrder
        | InputRateKind::Transit
        | InputRateKind::InverseGaussian
        | InputRateKind::ZeroOrder => true,
        InputRateKind::Weibull => false,
    }
}

/// Whether a `transit`/`igd` route is in its flip-flop domain, where the
/// exponential-tilting closed form does not converge and the kernel returns `0`.
/// Such a subject must integrate (the ODE forcing is valid for any params), so
/// its presence declines the whole model from the closed-form path — mirroring
/// the single-route flip-flop reroute (`absorption_flip_flop_at`). `first_order`
/// never flips (the `ka ≈ ke` limit is handled inside the kernel).
// `!(ke < bound)` is deliberate (not `ke >= bound`): it declines a transient `NaN`
// `ke`/`bound` too (`NaN < x` is false → `!false` = true → decline), matching the
// kernels' own `!(ke < ktr)` NaN-safe guards. Suppress the partial-ord lint that
// would "simplify" it to the NaN-unsafe `>=` form.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn route_flips(f: &InputRateForcing, dp: &DispParams<f64>, p: &[f64]) -> bool {
    match dp.periph {
        None => {
            // One-compartment: the convergence bound is on `ke = CL/V`, matching
            // `one_cpt_transit_amt_g` / `one_cpt_ig_amt_g` exactly.
            let ke = dp.cl / dp.v;
            match f.kind {
                InputRateKind::Transit => {
                    let n = forcing_arg(f, p, 0, 0.0);
                    let mtt = forcing_arg(f, p, 1, 1.0);
                    mtt <= 0.0 || !(ke < (n + 1.0) / mtt)
                }
                InputRateKind::InverseGaussian => {
                    let mat = forcing_arg(f, p, 0, 1.0);
                    let cv2 = forcing_arg(f, p, 1, 1.0);
                    mat <= 0.0 || cv2 <= 0.0 || !(ke < 1.0 / (2.0 * mat * cv2))
                }
                _ => false,
            }
        }
        Some((q, v2)) => {
            // Two-compartment: the tilting converges on the **fast macro-rate α**
            // (an eigenvalue, α > k10 = CL/V1) and needs distinct eigenvalues —
            // NOT `ke = CL/V1`. Reuse the 2-cpt kernel's OWN domain predicate so
            // `route_flips` can never disagree with what the kernel actually does
            // (a route that fails it would otherwise return a silent 0).
            match f.kind {
                InputRateKind::Transit => {
                    let n = forcing_arg(f, p, 0, 0.0);
                    let mtt = forcing_arg(f, p, 1, 1.0);
                    crate::sens::two_cpt::transit_2cpt_domain_ok::<f64>(dp.cl, dp.v, q, v2, n, mtt)
                        .is_none()
                }
                InputRateKind::InverseGaussian => {
                    let mat = forcing_arg(f, p, 0, 1.0);
                    let cv2 = forcing_arg(f, p, 1, 1.0);
                    crate::sens::two_cpt::ig_2cpt_domain_ok::<f64>(dp.cl, dp.v, q, v2, mat, cv2)
                        .is_none()
                }
                _ => false,
            }
        }
    }
}

/// Everything the closed-form MR path needs once a subject is admitted: the
/// compiled ODE spec, the identified disposition, its recovered `f64` rates
/// (used for the `route_flips` domain check both the value and gradient path
/// share), and the `f64` PK-parameter snapshot at `(θ, η, t=0)`.
pub(crate) struct MrScope<'a> {
    pub spec: &'a OdeSpec,
    pub disp: MrDisposition,
    pub dp: DispParams<f64>,
    pub pk: crate::types::PkParams,
}

/// The single closed-form MR scope gate, shared by the value path
/// ([`mr_predictions`]) and the gradient path (`mr_subject_sensitivities` /
/// `mr_subject_eta_grad`) so the two can never admit a different set of
/// subjects — a value/gradient scope drift is exactly the silent-wrong class
/// the reduction-to-ODE tests guard against, and it can only be prevented by
/// having one gate, not two independently-maintained copies.
///
/// Scope (all must hold): an `[odes]` model with ≥1 input-rate forcing, every one
/// of a closed-form-able kind ([`is_closed_form_kind`]); a canonical linear
/// 1-/2-cpt disposition ([`identify_disposition`]); no IOV, time-varying
/// covariates, resets, `init(...)`, steady-state or infusion doses, or
/// compartment-indexed `F{c}`/`ALAG{c}` (the `dose_attr_map`, which would make
/// the per-dose `F`/lag non-uniform); and no route in its flip-flop domain
/// ([`route_flips`]). Returns `None` when out of scope, in which case the
/// caller falls back to the ODE path (always correct).
pub(crate) fn mr_scope<'a>(
    model: &'a crate::types::CompiledModel,
    subject: &crate::types::Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<MrScope<'a>> {
    let spec = model.ode_spec.as_ref()?;
    if model.n_kappa > 0 || subject.has_tv_covariates() || subject.has_resets() {
        return None;
    }
    // Any model-time dependence — a `TIME`-reading *individual parameter*, or a
    // non-autonomous `[odes]` RHS reading `TAD`/`TAFD`/`T`/`TIME` — is out of
    // scope, and both are decided **statically**, before `identify_disposition`,
    // because that behavioural probe provably cannot decide either one.
    //
    // For a `TIME`-dependent parameter: the single `t = 0` parameter snapshot
    // below cannot represent it, and the dependence is invisible to the probe's
    // fixed-`p` Jacobian.
    //
    // For a non-autonomous RHS: superposition needs a **time-invariant**
    // disposition, and `-CL/V*central*(1 + k*TAD)` is linear in the state but not
    // time-invariant, so `Σ FR_k·kernel_k(t − LAG_k)` cannot represent it. The
    // probe pins `tad = 0.0`, so a `TAD` term evaluates identically at both probe
    // times and is admitted as time-invariant — measured bit-identical to the same
    // model with the term deleted, 8× wrong by 12 h (#1124). And it samples two
    // times only, so *any* of the four spellings read from an untaken `if` branch
    // slips past as well (measured 4.0e-1 each). See the note in `probe_rhs_f64`
    // for why varying `tad` in the probe is not the fix.
    //
    // One call, not two hand-paired flags: `model_uses_time_anywhere` is the same
    // predicate the value path's routing site uses, and
    // `OdeRhsProgram::reads_model_time` inside it is the same one the SS gates
    // use. A fifth spelling is then wired in one place. (Pairing the flags by hand
    // per call site is exactly how #1124's `TIME` gap survived.)
    //
    // This is the **gradient-side** half of the fix. The value path does not reach
    // here at all — the routing site sends these models to the event-driven
    // predictor first — but `mr_subject_sensitivities` / `mr_subject_eta_grad` are
    // called from `sens/provider.rs` and never pass through that routing. Without
    // this gate the value would carry `TAD` while the gradient dropped it: a
    // value/gradient mismatch worse than either alone.
    //
    // Over-declines only where a time variable provably cancels (`T − TAFD` is the
    // first dose time, a constant). Correct, just slower.
    if crate::pk::model_uses_time_anywhere(model) {
        return None;
    }
    // …and the same for `TIME` in the Form C *readout*, which
    // `model_uses_time_builtin` does not see (it inspects the individual-parameter
    // program only). `mr_observable_g` reads the observable as `m·amount`, with `m`
    // recovered by a state-space linearity probe that evaluates the readout at the
    // ambient model time — so a time term would be silently dropped rather than
    // rejected. Mirrors the same decline inside `recover_disp_params_g`, kept here
    // too so scope membership never depends on the ambient thread-local (#1028).
    if spec
        .readout_program
        .as_ref()
        .is_some_and(|ro| ro.reads_time_builtin())
    {
        return None;
    }
    if spec.init_fn.is_some() || !spec.dose_attr_map.is_empty() {
        return None;
    }
    // An SDE (`[diffusion]` block on one or more states): `identify_disposition`'s
    // Jacobian probe reads only `rhs_program` (the drift), so it is structurally blind
    // to a diffusion term on an otherwise-canonical-linear state and would admit it.
    // For the *value* this is harmless (the SDE mean prediction IS the drift solution —
    // `likelihood::model_predictions` returns the drift, EKF noise enters only the
    // variance `p_obs`), so the closed form would give the correct mean. For the
    // *gradient* it is not: the analytic jet of the drift omits `∂p_obs/∂θ`, which the
    // FD gradient of the full OFV carries — so an SDE must take FD, exactly as
    // `ode_analytical_supported` enforces (`!diffusion_var.is_empty() → false`). Higher
    // gates (`sens_supported` / `analytic_inner_grad_supported_model`) already route SDE
    // to FD before either provider is reached, so this is the same defensive re-check
    // `ode_subject_sensitivities` keeps (fail loudly to FD, never a silent wrong jet;
    // the #637 layering principle) — placed in the shared gate so value and gradient
    // stay on one scope.
    if !spec.diffusion_var.is_empty() {
        return None;
    }
    if spec.input_rate.is_empty() || !spec.input_rate.iter().all(|f| is_closed_form_kind(f.kind)) {
        return None;
    }
    if subject.doses.iter().any(|d| d.ss || d.is_infusion()) {
        return None;
    }
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let disp = identify_disposition(spec, &pk.values, 1.0)?;
    // Every dose must feed the central compartment (the forcings' shared target).
    // `mr_observable_g` splits each dose across all forcings, mirroring the ODE
    // forcing seam — which applies each forcing only to doses into *its own*
    // compartment (`predictions.rs`: `d.cmt-1 != forcing.cmt → continue`). A dose
    // into any other compartment is a bolus the superposition does not represent,
    // so decline it. (`dose.cmt` is 1-based; `disp.central` is a 0-based state.)
    if subject.doses.iter().any(|d| d.cmt_idx() != disp.central) {
        return None;
    }
    let dp = recover_disp_params_g::<f64>(spec, &disp, &pk.values, &subject.covariates)?;
    if spec
        .input_rate
        .iter()
        .any(|f| route_flips(f, &dp, &pk.values))
    {
        return None;
    }
    Some(MrScope { spec, disp, dp, pk })
}

/// Verify-gate memo key: `OdeSpec` identity, subject id, and a **coarse bucket of
/// the recovered disposition**, one `i32` per parameter.
///
/// The bucket is what keeps the memo honest. Admission is *not* purely a property
/// of the model and subject shape — `mr_scope` evaluates `pk_param_fn(θ, η)`,
/// probes `identify_disposition` numerically at that point, declines in
/// `recover_disp_params_g` on a non-positive slope, and runs the explicitly
/// parameter-dependent `route_flips` domain check. Keyed on `(spec, subject)`
/// alone the gate would validate only the **first** `(θ, η)` a fit visits — the
/// initial estimates, where models are best behaved — and never re-check a closed
/// form degrading toward a `ka ≈ ke` or `α ≈ β` confluence as the optimiser walks.
///
/// Bucketing on `round(10 · log₁₀ p)` re-verifies whenever any disposition
/// parameter moves by more than ~26% (a decade split into ten), which is far
/// finer than the drift that turns a well-conditioned closed form marginal, while
/// still collapsing the thousands of near-identical evaluations within one
/// optimiser step.
#[cfg(debug_assertions)]
type MrVerifyKey = (usize, String, Vec<i32>);

/// The coarse disposition bucket of [`MrVerifyKey`]. `log₁₀` of a non-positive or
/// non-finite parameter is not meaningful; such a point gets its own sentinel
/// bucket so it is verified rather than silently folded in with a valid one.
#[cfg(debug_assertions)]
fn disp_bucket(dp: &DispParams<f64>) -> Vec<i32> {
    let mut out = Vec::with_capacity(4);
    let mut push = |p: f64| {
        out.push(if p.is_finite() && p > 0.0 {
            (p.log10() * 10.0).round() as i32
        } else {
            i32::MIN
        });
    };
    push(dp.cl);
    push(dp.v);
    if let Some((q, v2)) = dp.periph {
        push(q);
        push(v2);
    }
    out
}

#[cfg(debug_assertions)]
thread_local! {
    /// Keys the verify-gate has already checked on this thread — see
    /// [`MrVerifyKey`] for why the parameter bucket is part of the key.
    ///
    /// Measured on `cargo test --lib --features ci`: **577 s unmemoized**, against
    /// 281–470 s memoized over three runs and a single 296 s run with no gate at
    /// all. Wall-clock on this machine is too noisy to separate the memoized gate
    /// from no gate; the unmemoized ~2× penalty is well outside that spread, and
    /// is what this exists to avoid.
    ///
    /// Keyed on the `OdeSpec`'s address, so a freed spec whose address is reused
    /// by a *different* spec under a subject of the same id **and** the same
    /// disposition bucket would skip one check. That direction is safe — it misses
    /// a check, it never raises a false alarm — and the `reduces_to_ode_*` zoo
    /// covers the same reduction unconditionally.
    static MR_TWIN_VERIFIED: std::cell::RefCell<std::collections::HashSet<MrVerifyKey>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

/// Runtime verify-gate (#1124): re-validate an admitted closed form against its
/// own integrated `OdeSpec` twin, in debug builds only.
///
/// [`identify_disposition`]'s structural checks only pick *which* closed form to
/// try; this is what is supposed to make admitting one safe (see the tolerance
/// constants). It was cited as the real safety net for three years and never
/// existed — which is how #1124 shipped: a `TAD`-reading RHS passed every
/// structural check, the fast path dropped the term, and predictions were 8×
/// wrong. The `reduces_to_ode_*` zoo is the same reduction, but it only covers
/// fixtures somebody thought to write, and nobody wrote one that read `TAD`.
/// This gate runs on the models callers actually fit, so it does not depend on
/// anyone having anticipated the case.
///
/// It **asserts**; it never declines. A gate that changed the route in debug
/// would make debug and release disagree about what ferx predicts, which is a
/// worse failure than the one it guards. Release builds compile it out entirely,
/// so the fast path stays a fast path.
///
/// Non-finite twin entries are skipped rather than reported: a twin that fails to
/// integrate is its own defect (`ode/predictions.rs`), not evidence that the
/// closed form is wrong, and panicking here would misattribute it. Skipping can
/// make a whole call **vacuous** (a length mismatch, or every twin entry
/// non-finite), so a vacuous call releases its memo key again rather than
/// recording a check that never happened — otherwise one `NaN` twin would retire
/// that `(spec, subject, bucket)` permanently and silently.
#[cfg(debug_assertions)]
fn verify_against_ode_twin(
    s: &MrScope<'_>,
    subject: &crate::types::Subject,
    theta: &[f64],
    eta: &[f64],
    mr: &[f64],
) {
    // The bound has to scale with the model's *own* solver accuracy: the twin is
    // integrated at `spec.solver_opts`, so it cannot be a tighter reference than
    // the user asked for. A fixed bound is the wrong shape here, and a fixed
    // `1e-4` was measurably wrong — the zoo fixtures pin `reltol = 1e-10`, but a
    // model that leaves the default `1e-4` produces a twin that honestly differs
    // from the exact closed form by far more than that.
    //
    // Measured across the `reduces_to_ode_*` zoo with the pinned tolerances
    // stripped (so `reltol = 1e-4`, the default), worst `|closed form − twin|`
    // relative over all observations:
    //
    // | model                | 1 dose  | 3 doses     |
    // |----------------------|---------|-------------|
    // | 1-cpt parallel       | 9.4e-7  | 3.4e-3      |
    // | 1-cpt parallel, η≠0  | 1.0e-6  | 2.0e-3      |
    // | 1-cpt per-route lag  | 2.9e-3  | **6.4e-3**  |
    // | 2-cpt parallel       | 1.3e-6  | 3.0e-3      |
    //
    // — worst 6.4e-3, i.e. 64× `reltol`, and it grows with dose count as the
    // integration error accumulates across cycles. `500 × reltol` leaves ~8×
    // headroom over that while still catching the defect this exists for by ~12×
    // (#1124 diverged by 5.8e-1). A false positive here panics a debug build, so
    // the headroom is deliberately on that side.
    //
    // The floor keeps the bound meaningful for a model that pins its tolerances
    // very tight, where `500 × reltol` would drop below honest last-bit noise.
    //
    // Read the net's coarseness honestly: at the shipped default `reltol = 1e-4`
    // the bound is `0.05` relative, so this catches only a gross disagreement
    // there. The ~12× detection margin above is against #1124's **worst**
    // observation (5.8e-1); the same defect was 1.4e-1 at `t = 4.4` and 2.2e-4 at
    // `t = 0.5`, i.e. under the default bound at the early times. The gate is
    // tight only for a model that pins `ode_reltol` — it is a backstop against a
    // structurally wrong closed form, not a general accuracy guarantee.
    //
    // Once per (spec, subject, disposition bucket) — see `MR_TWIN_VERIFIED`.
    // Recorded *before* the integration so a panicking assert does not re-enter on
    // unwind; released again below if the comparison turns out vacuous.
    let key = (
        s.spec as *const OdeSpec as usize,
        subject.id.clone(),
        disp_bucket(&s.dp),
    );
    if !MR_TWIN_VERIFIED.with(|seen| seen.borrow_mut().insert(key.clone())) {
        return;
    }
    let release = || MR_TWIN_VERIFIED.with(|seen| seen.borrow_mut().remove(&key));
    let verify_tol = (500.0 * s.spec.solver_opts.reltol).max(1e-4);
    let twin = crate::ode::ode_predictions(s.spec, &s.pk.values, theta, eta, subject);
    if twin.len() != mr.len() {
        release();
        return;
    }
    let mut compared = 0usize;
    for (i, (&a, &b)) in mr.iter().zip(&twin).enumerate() {
        if !b.is_finite() {
            continue;
        }
        compared += 1;
        debug_assert!(
            (a - b).abs() <= verify_tol * (1.0 + b.abs()),
            "modified-release closed form disagrees with its ODE twin at obs {i} \
             (t = {}): closed form {a}, twin {b}. The subject was admitted by \
             `mr_scope` but the two do not reduce to each other — the identified \
             disposition does not represent this model.",
            subject.obs_times.get(i).copied().unwrap_or(f64::NAN),
        );
    }
    if compared == 0 {
        release();
    }
}

/// Closed-form modified-release predictions for `subject` at its observation
/// times, or `None` when out of scope ([`mr_scope`]). This is the **value**
/// path: `predict` / `simulate` and the FOCE/FOCEI marginal-objective value
/// ([`crate::stats`]'s `model_predictions` →
/// [`crate::pk::compute_predictions_with_tv`]). For an FD-method fit the
/// gradient finite-differences these same values, so it is fully
/// self-consistent. For an analytic-sensitivity fit the gradient comes from
/// [`mr_subject_sensitivities`]/[`mr_subject_eta_grad`] when in scope
/// (identical `mr_scope`, so never a different subject set), else the ODE
/// provider — either way value and gradient **agree to solver tolerance** (the
/// closed form reduces to the same integrated twin), so the objective stays
/// consistent to far within `inner_tol`.
pub(crate) fn mr_predictions(
    model: &crate::types::CompiledModel,
    subject: &crate::types::Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<f64>> {
    let s = mr_scope(model, subject, theta, eta)?;
    let lag_cmt = s.pk.lagtime();
    let f_bio = s.pk.f_bio();
    let preds: Vec<f64> = subject
        .obs_times
        .iter()
        .map(|&t| {
            mr_observable_g(
                &s.dp,
                &s.spec.input_rate,
                &subject.doses,
                t,
                &s.pk.values,
                lag_cmt,
                f_bio,
            )
        })
        .collect();
    #[cfg(debug_assertions)]
    verify_against_ode_twin(&s, subject, theta, eta, &preds);
    Some(preds)
}

/// Analytic-provider axis-count guard shared by [`mr_subject_sensitivities`] and
/// [`mr_subject_eta_grad`]: the compiled individual-parameter program's `(θ,η)`
/// axis counts must match the model's exactly (a mismatch is an NN-θ / IOV-κ
/// desugaring this fast path does not carry) — mirrors
/// [`crate::sens::ode_provider::ode_analytical_supported`]'s identical check.
/// Declining here (→ `None` → the ODE provider, which re-checks the same
/// invariant) is cheap insurance against a silently-wrong jet; it is not a new
/// "is this subject analytic" report, since `ode_analytical_supported` already
/// requires this for the model to be reported analytic at all.
fn mr_prog_axes_match(
    prog: &crate::parser::model_parser::IndivParamProgram,
    n_theta: usize,
    n_eta: usize,
) -> bool {
    prog.n_theta_axis() == n_theta
        && prog.n_eta_axis() == n_eta
        && prog.n_axes() <= crate::sens::ode_provider::MAX_ODE_AXES
}

/// Closed-form analytic FOCE/FOCEI outer sensitivities for a modified-release
/// subject (#860 Phase A6) — the fast-path alternative to
/// [`crate::sens::ode_provider::ode_subject_sensitivities`] for the same
/// (already-analytic, per `ode_analytical_supported`) subjects `mr_scope`
/// additionally admits. `mr_observable_g` is evaluated at `T = Dual2<M>`
/// seeded directly on `(θ, η)` via
/// [`crate::sens::ode_provider::seed_pk_dual2`] — the same seeding the TV-cov
/// event-driven walk (`run_subject_tvcov`) uses — so no ODE integration and no
/// separate outer chain-rule step: composing `mr_observable_g`'s arithmetic
/// over an already-`(θ,η)`-seeded `Dual2` *is* the chain rule. `None` when
/// `mr_scope` declines, or the axis-count guard fails — either way the caller
/// falls back to `ode_subject_sensitivities`.
pub(crate) fn mr_subject_sensitivities(
    model: &crate::types::CompiledModel,
    subject: &crate::types::Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<crate::sens::provider::SubjectSens> {
    let s = mr_scope(model, subject, theta, eta)?;
    let prog = s.spec.indiv_param_program.as_ref()?;
    if !mr_prog_axes_match(prog, model.n_theta, model.n_eta)
        || !crate::sens::ode_provider::ode_scaling_supported(model)
    {
        return None;
    }
    macro_rules! dispatch {
        ($($m:literal),+) => {
            match model.n_theta + model.n_eta {
                $($m => mr_sens_dual::<$m>(model, subject, &s, prog, theta, eta),)+
                _ => None,
            }
        };
    }
    let mut sens = dispatch!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    )?;
    crate::sens::provider::apply_event_walk_expression_scale_outer(
        &mut sens, model, subject, prog, theta, eta,
    )?;
    Some(sens)
}

/// `Dual2<M>`-width monomorphisation of [`mr_subject_sensitivities`]. Seeds the
/// PK-slot duals, recovers the disposition params over the same duals (so
/// `v = 1/m`, `cl = ke/m`, and the 2-cpt `v2 = q/k21` division all carry exact
/// `(θ,η)` sensitivities), evaluates `mr_observable_g` per observation, then
/// unpacks the jet into [`crate::sens::provider::ObsSens`] — the identical
/// field layout `run_subject_tvcov` uses (axes `0..n_theta` = θ,
/// `n_theta..M` = η).
fn mr_sens_dual<const M: usize>(
    model: &crate::types::CompiledModel,
    subject: &crate::types::Subject,
    s: &MrScope,
    prog: &crate::parser::model_parser::IndivParamProgram,
    theta: &[f64],
    eta: &[f64],
) -> Option<crate::sens::provider::SubjectSens> {
    use crate::sens::dual2::Dual2;
    let n_theta = model.n_theta;
    let n_eta = model.n_eta;
    let p: Vec<Dual2<M>> = crate::sens::ode_provider::seed_pk_dual2::<M>(
        model,
        prog,
        theta,
        eta,
        &subject.covariates,
        0.0,
    );
    let dp = recover_disp_params_g::<Dual2<M>>(s.spec, &s.disp, &p, &subject.covariates)?;
    let lag_cmt = p[crate::types::PK_IDX_LAGTIME];
    let f_bio = p[crate::types::PK_IDX_F];
    let mut out = Vec::with_capacity(subject.obs_times.len());
    for &t in &subject.obs_times {
        let fd = mr_observable_g::<Dual2<M>>(
            &dp,
            &s.spec.input_rate,
            &subject.doses,
            Dual2::constant(t),
            &p,
            lag_cmt,
            f_bio,
        );
        // `ScalarScale`/LTBS output transform, mirroring the ODE provider's
        // `resolve_obs_readout` → `apply_output_transform` (same per-observation
        // site, basis-agnostic over which space the dual's axes represent — see
        // that function's doc). `ExpressionScale` is handled separately, after
        // the whole `SubjectSens` is assembled, by `apply_event_walk_expression_scale_outer`.
        let fd = crate::sens::ode_provider::apply_output_transform(model, fd);
        // Shared jet→ObsSens scatter (identical axis layout in every provider).
        out.push(crate::sens::provider::obs_sens_from_dual2::<M>(
            &fd, n_theta, n_eta,
        ));
    }
    Some(crate::sens::provider::SubjectSens { obs: out })
}

/// Closed-form light **inner** η-gradient for a modified-release subject (#860
/// Phase A6) — the fast-path alternative to
/// [`crate::sens::ode_provider::ode_subject_eta_grad`], mirroring
/// [`mr_subject_sensitivities`] at `Dual1<N>` (`N = n_eta`) width via
/// [`crate::sens::ode_provider::seed_pk_dual1`]. Per the shared per-subject
/// scope contract ([`crate::sens::ode_provider::ode_subject_supported`]'s doc),
/// this and [`mr_subject_sensitivities`] must serve exactly the same subjects —
/// both share `mr_scope` and `mr_prog_axes_match`, so that holds structurally.
pub(crate) fn mr_subject_eta_grad(
    model: &crate::types::CompiledModel,
    subject: &crate::types::Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<crate::sens::provider::ObsGrad>> {
    let s = mr_scope(model, subject, theta, eta)?;
    let prog = s.spec.indiv_param_program.as_ref()?;
    if !mr_prog_axes_match(prog, model.n_theta, model.n_eta)
        || !crate::sens::ode_provider::ode_scaling_supported(model)
    {
        return None;
    }
    macro_rules! dispatch {
        ($($n:literal),+) => {
            match model.n_eta {
                $($n => mr_eta_grad_dual::<$n>(model, subject, &s, prog, theta, eta),)+
                _ => None,
            }
        };
    }
    let mut out = dispatch!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    )?;
    crate::sens::provider::apply_event_walk_expression_scale_inner(
        &mut out, model, subject, prog, theta, eta,
    )?;
    Some(out)
}

/// `Dual1<N>`-width monomorphisation of [`mr_subject_eta_grad`].
fn mr_eta_grad_dual<const N: usize>(
    model: &crate::types::CompiledModel,
    subject: &crate::types::Subject,
    s: &MrScope,
    prog: &crate::parser::model_parser::IndivParamProgram,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<crate::sens::provider::ObsGrad>> {
    use crate::sens::dual1::Dual1;
    let p: Vec<Dual1<N>> = crate::sens::ode_provider::seed_pk_dual1::<N>(
        model,
        prog,
        theta,
        eta,
        &subject.covariates,
        0.0,
    )?;
    let dp = recover_disp_params_g::<Dual1<N>>(s.spec, &s.disp, &p, &subject.covariates)?;
    let lag_cmt = p[crate::types::PK_IDX_LAGTIME];
    let f_bio = p[crate::types::PK_IDX_F];
    let mut out = Vec::with_capacity(subject.obs_times.len());
    for &t in &subject.obs_times {
        let fd = mr_observable_g::<Dual1<N>>(
            &dp,
            &s.spec.input_rate,
            &subject.doses,
            Dual1::constant(t),
            &p,
            lag_cmt,
            f_bio,
        );
        // See the matching comment in `mr_sens_dual`: `ScalarScale`/LTBS via the
        // same shared, basis-agnostic `apply_output_transform`.
        let fd = crate::sens::ode_provider::apply_output_transform(model, fd);
        out.push(crate::sens::provider::ObsGrad {
            f: fd.value,
            df_deta: fd.grad.to_vec(),
        });
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn forcing(
        kind: InputRateKind,
        arg_slots: Vec<usize>,
        frac: Option<usize>,
        lag: Option<usize>,
    ) -> InputRateForcing {
        InputRateForcing {
            cmt: 0,
            kind,
            arg_slots,
            frac_slot: frac,
            lag_slot: lag,
        }
    }

    fn dose(time: f64, amt: f64) -> DoseEvent {
        DoseEvent::new(time, amt, 1, 0.0, false, 0.0)
    }

    // ── Layer 1: bit-reduction — a single unlagged, full-fraction route equals
    //    the single-route closed form directly. ──────────────────────────────
    #[test]
    fn single_first_order_route_reduces_to_bateman() {
        // params: [ka] at slot 0.
        let p = vec![1.3_f64];
        let dp = DispParams {
            cl: 5.0,
            v: 50.0,
            periph: None,
        };
        let f = vec![forcing(InputRateKind::FirstOrder, vec![0], None, None)];
        let doses = vec![dose(0.0, 100.0)];
        for &t in &[0.5, 1.0, 2.0, 4.0, 8.0, 12.0] {
            let got = mr_observable_g(&dp, &f, &doses, t, &p, 0.0, 1.0);
            let want = one_cpt_oral_g(100.0, t, dp.cl, dp.v, p[0], 1.0);
            assert!((got - want).abs() < 1e-12, "t={t}: {got} vs {want}");
        }
    }

    // ── Layer 2: superposition oracle — parallel first_order routes equal the
    //    independent sum of per-route single-route closed forms, split by frac
    //    and shifted by per-route lag. ────────────────────────────────────────
    #[test]
    fn parallel_routes_equal_independent_superposition() {
        // params: [ka1, ka2, fr1, fr2, lag2] slots 0..4.
        let ka1 = 1.5;
        let ka2 = 0.4;
        let fr1 = 0.6;
        let fr2 = 0.4;
        let lag2 = 2.0;
        let p = vec![ka1, ka2, fr1, fr2, lag2];
        let dp = DispParams {
            cl: 4.0,
            v: 40.0,
            periph: None,
        };
        let f = vec![
            forcing(InputRateKind::FirstOrder, vec![0], Some(2), None),
            forcing(InputRateKind::FirstOrder, vec![1], Some(3), Some(4)),
        ];
        let d = 100.0;
        let doses = vec![dose(0.0, d)];
        for &t in &[0.5, 1.0, 3.0, 6.0, 10.0, 16.0] {
            let got = mr_observable_g(&dp, &f, &doses, t, &p, 0.0, 1.0);
            // Independent oracle: FR1·oral(ka1)(t) + FR2·oral(ka2)(t − lag2).
            let term1 = fr1 * one_cpt_oral_g(d, t, dp.cl, dp.v, ka1, 1.0);
            let term2 = fr2 * one_cpt_oral_g(d, t - lag2, dp.cl, dp.v, ka2, 1.0);
            let want = (term1 + term2).max(0.0);
            assert!((got - want).abs() < 1e-12, "t={t}: {got} vs {want}");
        }
    }

    // Mixed kinds (transit + igd) also superpose additively.
    #[test]
    fn mixed_transit_igd_routes_superpose() {
        // params: [n, mtt, mat, cv2, fr1, fr2] slots 0..6.
        let p = vec![3.0, 1.5, 1.2, 0.3, 0.5, 0.5];
        let dp = DispParams {
            cl: 3.0,
            v: 30.0,
            periph: None,
        };
        let f = vec![
            forcing(InputRateKind::Transit, vec![0, 1], Some(4), None),
            forcing(InputRateKind::InverseGaussian, vec![2, 3], Some(5), None),
        ];
        let d = 100.0;
        let doses = vec![dose(0.0, d)];
        for &t in &[0.5, 1.0, 2.0, 4.0, 8.0] {
            let got = mr_observable_g(&dp, &f, &doses, t, &p, 0.0, 1.0);
            let want = (0.5 * one_cpt_transit_g(d, t, dp.cl, dp.v, p[0], p[1], 1.0)
                + 0.5 * one_cpt_ig_g(d, t, dp.cl, dp.v, p[2], p[3], 1.0))
            .max(0.0);
            assert!((got - want).abs() < 1e-12, "t={t}: {got} vs {want}");
        }
    }

    // Multi-dose superposition: two doses add by linearity.
    #[test]
    fn multi_dose_superposes() {
        let p = vec![1.0];
        let dp = DispParams {
            cl: 5.0,
            v: 50.0,
            periph: None,
        };
        let f = vec![forcing(InputRateKind::FirstOrder, vec![0], None, None)];
        let doses = vec![dose(0.0, 100.0), dose(12.0, 100.0)];
        for &t in &[1.0, 6.0, 13.0, 18.0, 24.0] {
            let got = mr_observable_g(&dp, &f, &doses, t, &p, 0.0, 1.0);
            let want = (one_cpt_oral_g(100.0, t, dp.cl, dp.v, p[0], 1.0)
                + one_cpt_oral_g(100.0, t - 12.0, dp.cl, dp.v, p[0], 1.0))
            .max(0.0);
            assert!((got - want).abs() < 1e-12, "t={t}: {got} vs {want}");
        }
    }

    // f_bio scales the whole dose before the frac split (one dose record → F).
    #[test]
    fn f_bio_scales_whole_dose() {
        let p = vec![1.2, 0.6, 0.4];
        let dp = DispParams {
            cl: 4.0,
            v: 40.0,
            periph: None,
        };
        let f = vec![
            forcing(InputRateKind::FirstOrder, vec![0], Some(1), None),
            forcing(InputRateKind::FirstOrder, vec![0], Some(2), None),
        ];
        let doses = vec![dose(0.0, 100.0)];
        let f_bio = 0.7;
        for &t in &[1.0, 4.0, 8.0] {
            let got = mr_observable_g(&dp, &f, &doses, t, &p, 0.0, f_bio);
            let want =
                ((0.6 + 0.4) * f_bio * one_cpt_oral_g(100.0, t, dp.cl, dp.v, p[0], 1.0)).max(0.0);
            assert!((got - want).abs() < 1e-12, "t={t}: {got} vs {want}");
        }
    }

    // ── Layer A1: behavioural disposition identification on *compiled* models —
    //    the numeric probe must recover CL/V(/Q/V2) from the RHS/readout alone,
    //    with no name or AST matching. ────────────────────────────────────────
    const PARALLEL_1CPT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVKA1(1.5, 0.05, 24.0)
  theta TVKA2(0.3, 0.01, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  FR1 = TVFR1
  FR2 = 1 - TVFR1
  KA1 = TVKA1
  KA2 = TVKA2
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    const PARALLEL_2CPT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(10.0, 0.1, 100.0)
  theta TVV2(100.0, 5.0, 1000.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVKA1(1.5, 0.05, 24.0)
  theta TVKA2(0.3, 0.01, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q = TVQ
  V2 = TVV2
  FR1 = TVFR1
  FR2 = 1 - TVFR1
  KA1 = TVKA1
  KA2 = TVKA2
[structural_model]
  ode(states=[central, periph])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V1*central - Q/V1*central + Q/V2*periph
  d/dt(periph) = Q/V1*central - Q/V2*periph
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    fn flat_params(model: &crate::types::CompiledModel, theta: &[f64], eta: &[f64]) -> Vec<f64> {
        let cov = std::collections::HashMap::new();
        (model.pk_param_fn)(theta, eta, &cov, 0.0).values.to_vec()
    }

    #[test]
    fn identifies_one_cpt_and_recovers_cl_v() {
        let model = crate::parser::model_parser::parse_model_string(PARALLEL_1CPT).unwrap();
        let spec = model.ode_spec.as_ref().expect("ode model");
        let theta = [5.0, 50.0, 0.6, 1.5, 0.3];
        let eta = [0.0, 0.0];
        let p = flat_params(&model, &theta, &eta);
        let disp = identify_disposition(spec, &p, 1.0).expect("identifies 1-cpt");
        assert_eq!(disp.kind, MrDispositionKind::OneCpt);
        let cov = std::collections::HashMap::new();
        let dp = recover_disp_params_g::<f64>(spec, &disp, &p, &cov).expect("recovers");
        assert!((dp.cl - 5.0).abs() < 1e-9, "cl={}", dp.cl);
        assert!((dp.v - 50.0).abs() < 1e-9, "v={}", dp.v);
        assert!(dp.periph.is_none());
    }

    #[test]
    fn identifies_two_cpt_and_recovers_cl_v1_q_v2() {
        let model = crate::parser::model_parser::parse_model_string(PARALLEL_2CPT).unwrap();
        let spec = model.ode_spec.as_ref().expect("ode model");
        let theta = [5.0, 50.0, 10.0, 100.0, 0.6, 1.5, 0.3];
        let eta = [0.0];
        let p = flat_params(&model, &theta, &eta);
        let disp = identify_disposition(spec, &p, 1.0).expect("identifies 2-cpt");
        assert_eq!(disp.kind, MrDispositionKind::TwoCpt);
        let cov = std::collections::HashMap::new();
        let dp = recover_disp_params_g::<f64>(spec, &disp, &p, &cov).expect("recovers");
        let (q, v2) = dp.periph.expect("two-cpt periph");
        assert!((dp.cl - 5.0).abs() < 1e-9, "cl={}", dp.cl);
        assert!((dp.v - 50.0).abs() < 1e-9, "v1={}", dp.v);
        assert!((q - 10.0).abs() < 1e-9, "q={q}");
        assert!((v2 - 100.0).abs() < 1e-9, "v2={v2}");
    }

    // A nonlinear (Michaelis–Menten) elimination must be DECLINED — the numeric
    // linearity probe is what protects the fast path from a silent wrong value.
    const MM_1CPT: &str = r#"
[parameters]
  theta TVVMAX(10.0, 0.1, 100.0)
  theta TVKM(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.5, 0.05, 24.0)
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  VMAX = TVVMAX
  KM = TVKM
  V = TVV * exp(ETA_V)
  KA = TVKA
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - VMAX*central/(KM + central)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    #[test]
    fn declines_nonlinear_michaelis_menten() {
        let model = crate::parser::model_parser::parse_model_string(MM_1CPT).unwrap();
        let spec = model.ode_spec.as_ref().expect("ode model");
        let theta = [10.0, 5.0, 50.0, 1.5];
        let eta = [0.0];
        let p = flat_params(&model, &theta, &eta);
        assert!(
            identify_disposition(spec, &p, 1.0).is_none(),
            "MM elimination must be declined (nonlinear)"
        );
    }

    // ── Layer 3 (the anchor): the admitted closed form must reduce to its own
    //    integrated OdeSpec twin, to solver tolerance. This is what ties the
    //    fast path to the (NONMEM-anchored) ODE path. Tight solver tol so the
    //    integrator's own onset-kink error (~1e-3 at default reltol) does not
    //    mask a real discrepancy. ───────────────────────────────────────────
    fn mk_subject(doses: Vec<DoseEvent>, obs_times: Vec<f64>) -> crate::types::Subject {
        let n = obs_times.len();
        let nd = doses.len();
        crate::types::Subject {
            id: "R".into(),
            doses,
            obs_times,
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n],
            obs_cmts: vec![1; n],
            covariates: std::collections::HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            obs_l2: Vec::new(),
            dose_occasions: vec![1; nd],
            fremtype: Vec::new(),
            obs_records: vec![],
        }
    }

    fn assert_reduces_to_ode(model_src: &str, theta: &[f64], eta: &[f64], doses: Vec<DoseEvent>) {
        let model = crate::parser::model_parser::parse_model_string(model_src).unwrap();
        let spec = model.ode_spec.as_ref().expect("ode model");
        let obs: Vec<f64> = (1..=60).map(|i| i as f64 * 0.4).collect(); // 0.4 .. 24 h
        let subject = mk_subject(doses, obs.clone());
        let mr = mr_predictions(&model, &subject, theta, eta).expect("MR-supported");
        let p = flat_params(&model, theta, eta);
        let ode = crate::ode::ode_predictions(spec, &p, theta, eta, &subject);
        assert_eq!(mr.len(), ode.len());
        for (i, (&a, &b)) in mr.iter().zip(&ode).enumerate() {
            assert!(
                (a - b).abs() < 1e-5 * (1.0 + b.abs()),
                "obs {i} t={}: MR {a} vs ODE {b}",
                obs[i]
            );
        }
    }

    const REDUCE_1CPT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVKA1(1.5, 0.05, 24.0)
  theta TVKA2(0.3, 0.01, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  FR1 = TVFR1
  FR2 = 1 - TVFR1
  KA1 = TVKA1
  KA2 = TVKA2
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    // IR + delayed-release: a per-route lag on the second pathway (the classic
    // modified-release picture, #856). The closed form shifts τ; the twin lags
    // the forcing onset — they must agree.
    const REDUCE_1CPT_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFR1(0.5, 0.05, 0.95)
  theta TVKA1(1.2, 0.05, 24.0)
  theta TVKA2(0.8, 0.05, 24.0)
  theta TVLAG2(3.0, 0.001, 12.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  FR1 = TVFR1
  FR2 = 1 - TVFR1
  KA1 = TVKA1
  KA2 = TVKA2
  LAG2 = TVLAG2
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2, lag=LAG2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    const REDUCE_2CPT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(10.0, 0.1, 100.0)
  theta TVV2(100.0, 5.0, 1000.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVKA1(1.5, 0.05, 24.0)
  theta TVKA2(0.3, 0.01, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q = TVQ
  V2 = TVV2
  FR1 = TVFR1
  FR2 = 1 - TVFR1
  KA1 = TVKA1
  KA2 = TVKA2
[structural_model]
  ode(states=[central, periph])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V1*central - Q/V1*central + Q/V2*periph
  d/dt(periph) = Q/V1*central - Q/V2*periph
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    #[test]
    fn reduces_to_ode_1cpt_parallel() {
        assert_reduces_to_ode(
            REDUCE_1CPT,
            &[5.0, 50.0, 0.6, 1.5, 0.3],
            &[0.0, 0.0],
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        );
    }

    #[test]
    fn reduces_to_ode_1cpt_parallel_nonzero_eta() {
        // η ≠ 0 exercises the recovered ke/V carrying the individual shift.
        assert_reduces_to_ode(
            REDUCE_1CPT,
            &[5.0, 50.0, 0.6, 1.5, 0.3],
            &[0.3, -0.2],
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        );
    }

    #[test]
    fn reduces_to_ode_1cpt_per_route_lag() {
        assert_reduces_to_ode(
            REDUCE_1CPT_LAG,
            &[5.0, 50.0, 0.5, 1.2, 0.8, 3.0],
            &[0.0],
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        );
    }

    #[test]
    fn reduces_to_ode_1cpt_multi_dose() {
        assert_reduces_to_ode(
            REDUCE_1CPT,
            &[5.0, 50.0, 0.6, 1.5, 0.3],
            &[0.0, 0.0],
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ],
        );
    }

    #[test]
    fn reduces_to_ode_2cpt_parallel() {
        assert_reduces_to_ode(
            REDUCE_2CPT,
            &[5.0, 50.0, 10.0, 100.0, 0.6, 1.5, 0.3],
            &[0.0],
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        );
    }

    // #860 Phase B, 2-cpt: a zero_order route mixed with first_order into a
    // two-compartment disposition — exercises `two_cpt_zero_order_g`'s macro-rate
    // (α/β) form, not just the 1-cpt kernel.
    const MIXED_ZERO_ORDER_2CPT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(10.0, 0.1, 100.0)
  theta TVV2(100.0, 5.0, 1000.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.5, 0.05, 24.0)
  theta TVDUR(2.0, 0.1, 12.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q = TVQ
  V2 = TVV2
  FZO = TVFZO
  FZO1 = 1 - TVFZO
  KA = TVKA
  DUR = TVDUR
[structural_model]
  ode(states=[central, periph])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V1*central - Q/V1*central + Q/V2*periph
  d/dt(periph) = Q/V1*central - Q/V2*periph
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    #[test]
    fn reduces_to_ode_2cpt_zero_order_mixed() {
        assert_reduces_to_ode(
            MIXED_ZERO_ORDER_2CPT,
            &[5.0, 50.0, 10.0, 100.0, 0.4, 1.5, 2.0],
            &[0.0],
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        );
    }

    // A *static* covariate (WT on CL) is baked into the parameters at the single
    // `t = 0` snapshot — valid because it does not vary in time — and the recovered
    // `ke` must carry it, matching the twin which reads the same covariate.
    const REDUCE_1CPT_COV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVKA1(1.5, 0.05, 24.0)
  theta TVKA2(0.3, 0.01, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL) * WT / 70
  V = TVV
  FR1 = TVFR1
  FR2 = 1 - TVFR1
  KA1 = TVKA1
  KA2 = TVKA2
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    // Bioavailability F (< 1) scales the whole dose before the FR split, in both
    // the closed form (`mass = FR·F·D`) and the twin (`dose_f_bio`).
    const REDUCE_1CPT_F: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVKA1(1.5, 0.05, 24.0)
  theta TVKA2(0.3, 0.01, 24.0)
  theta TVF(0.7, 0.05, 1.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  FR1 = TVFR1
  FR2 = 1 - TVFR1
  KA1 = TVKA1
  KA2 = TVKA2
  F = TVF
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

    #[test]
    fn reduces_to_ode_1cpt_static_covariate() {
        let model = crate::parser::model_parser::parse_model_string(REDUCE_1CPT_COV).unwrap();
        let spec = model.ode_spec.as_ref().unwrap();
        let theta = [5.0, 50.0, 0.6, 1.5, 0.3];
        let eta = [0.1];
        let obs: Vec<f64> = (1..=60).map(|i| i as f64 * 0.4).collect();
        let mut subject = mk_subject(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs.clone(),
        );
        subject.covariates = std::collections::HashMap::from([("WT".to_string(), 140.0)]);
        let mr = mr_predictions(&model, &subject, &theta, &eta).expect("MR-supported");
        // Flat params WITH the covariate baked in, for the ODE reference.
        let pk = (model.pk_param_fn)(&theta, &eta, &subject.covariates, 0.0);
        let ode = crate::ode::ode_predictions(spec, &pk.values, &theta, &eta, &subject);
        for (i, (&a, &b)) in mr.iter().zip(&ode).enumerate() {
            assert!(
                (a - b).abs() < 1e-5 * (1.0 + b.abs()),
                "cov obs {i} t={}: MR {a} vs ODE {b}",
                obs[i]
            );
        }
    }

    #[test]
    fn reduces_to_ode_1cpt_bioavailability() {
        assert_reduces_to_ode(
            REDUCE_1CPT_F,
            &[5.0, 50.0, 0.6, 1.5, 0.3, 0.7],
            &[0.0],
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        );
    }

    // ── Decline branches (the "no happy paths" safety): a subject/model outside
    //    scope must return None so the caller integrates. A gate that silently
    //    fails to fire is exactly the bug these guard against. ─────────────────
    const MIXED_ZERO_ORDER: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.5, 0.05, 24.0)
  theta TVDUR(2.0, 0.1, 12.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  FZO = TVFZO
  FZO1 = 1 - TVFZO
  KA = TVKA
  DUR = TVDUR
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    // TIME in the RHS makes the disposition non-autonomous — must decline.
    const TIME_DEPENDENT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.5, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  KA = TVKA
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central*(1 + 0.01*TIME)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    // Single transit route whose ke = CL/V has crossed the flip-flop bound
    // KTR = (n+1)/mtt — the closed form does not converge, must decline.
    const TRANSIT_FLIPFLOP: &str = r#"
[parameters]
  theta TVCL(60.0, 0.1, 200.0)
  theta TVV(10.0, 1.0, 500.0)
  theta TVN(2.0, 0.1, 20.0)
  theta TVMTT(1.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  N = TVN
  MTT = TVMTT
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = transit(n=N, mtt=MTT) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    fn parse(src: &str) -> crate::types::CompiledModel {
        crate::parser::model_parser::parse_model_string(src).unwrap()
    }

    #[test]
    fn admits_zero_order_pathway_and_reduces_to_ode() {
        // #860 Phase B: a mixed first_order + zero_order model no longer declines —
        // it must both admit the closed form AND reduce to the ODE twin.
        assert_reduces_to_ode(
            MIXED_ZERO_ORDER,
            &[5.0, 50.0, 0.4, 1.5, 2.0],
            &[0.0],
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        );
    }

    #[test]
    fn declines_time_dependent_disposition() {
        let model = parse(TIME_DEPENDENT);
        let subject = mk_subject(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            vec![1.0, 2.0, 4.0],
        );
        assert!(
            mr_predictions(&model, &subject, &[5.0, 50.0, 1.5], &[0.0]).is_none(),
            "a TIME-dependent disposition must decline"
        );
    }

    #[test]
    fn declines_time_dependent_readout() {
        // #1028: the Form C readout resolves `TIME` per observation, so a readout with
        // an additive time term is no longer a function of the state alone. The MR fast
        // path reads the observable as `m·amount`, with `m` recovered by a linearity
        // probe (`readout(0) ≈ 0`, `readout(2·e_c) ≈ 2·readout(e_c)`) evaluated under
        // the *ambient* model-time thread-local — `0.0` here — so the time term
        // evaluates to zero at the probe, `y0 == 0` passes, and the closed form would
        // then silently serve predictions with the whole term missing. Decline instead.
        let src = REDUCE_1CPT.replace("y = central / V", "y = central / V + 0.5 * TIME");
        let model = parse(&src);
        let subject = mk_subject(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            vec![1.0, 2.0, 4.0],
        );
        let theta = [5.0, 50.0, 0.6, 1.5, 0.3];
        let eta = [0.0, 0.0];

        // Baseline: the same model *without* the time term is MR-admitted, so the
        // decline below is attributable to the readout and nothing else.
        assert!(
            mr_scope(&parse(REDUCE_1CPT), &subject, &theta, &eta).is_some(),
            "REDUCE_1CPT must be MR-admitted before the TIME term is added"
        );
        assert!(
            mr_scope(&model, &subject, &theta, &eta).is_none(),
            "a TIME-reading Form C readout must decline the closed-form MR scope"
        );
        assert!(
            mr_predictions(&model, &subject, &theta, &eta).is_none(),
            "…and the MR value path with it"
        );
        assert!(
            mr_subject_sensitivities(&model, &subject, &theta, &eta).is_none(),
            "…and the MR analytic-gradient path"
        );

        // The term is real on the path that now serves this model: the ODE readout
        // (which enters the per-observation guard) differs from the time-free twin by
        // exactly `0.5·t`. Had the fast path stayed in scope it would have returned the
        // time-free values — which is the silent-wrong this decline prevents.
        let spec = model.ode_spec.as_ref().expect("ode model");
        let p = flat_params(&model, &theta, &eta);
        let with_time = crate::ode::ode_predictions(spec, &p, &theta, &eta, &subject);
        let base_model = parse(REDUCE_1CPT);
        let base_spec = base_model.ode_spec.as_ref().expect("ode model");
        let base_p = flat_params(&base_model, &theta, &eta);
        let without_time = crate::ode::ode_predictions(base_spec, &base_p, &theta, &eta, &subject);
        for (i, t) in subject.obs_times.iter().enumerate() {
            let delta = with_time[i] - without_time[i];
            assert!(
                (delta - 0.5 * t).abs() < 1e-6,
                "obs {i} t={t}: readout time term is {delta}, expected {}",
                0.5 * t
            );
        }
    }

    #[test]
    fn declines_sde_diffusion() {
        // A parsed SDE model always uses an `ObsCmt` readout (Form-C `y = <expr>`
        // is rejected on SDE at parse), and `ObsCmt` has no `readout_program`, so
        // `identify_disposition` already declines it at its `readout_program.as_ref()?`
        // — the `diffusion_var` guard can't be reached through the normal parse path
        // *today*. It is forward-defensive: were `ObsCmt` (or a future readout) ever
        // made MR-identifiable, an SDE's linear *drift* would identify and the guard
        // is the only thing keeping the stochastic model off the deterministic closed
        // form. Exercise it directly by injecting a diffusion term into a model
        // `mr_scope` otherwise admits, proving the guard is load-bearing.
        let mut model = parse(REDUCE_1CPT);
        let subject = mk_subject(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            vec![1.0, 2.0, 4.0],
        );
        let theta = [5.0, 50.0, 0.6, 1.5, 0.3];
        let eta = [0.0, 0.0];
        // Baseline: this model IS admitted (sanity — else the test proves nothing).
        assert!(
            mr_scope(&model, &subject, &theta, &eta).is_some(),
            "REDUCE_1CPT must be MR-admitted before the diffusion injection"
        );
        // Inject a state diffusion term without touching the (still canonical-linear)
        // drift RHS the disposition probe reads. The guard keys on `diffusion_var`,
        // exactly as the ODE provider's `ode_analytical_supported` does (both read
        // `diffusion_var`, not `is_sde()`'s `diffusion_theta_start`), so setting this
        // one field is what exercises the guard.
        let spec = model.ode_spec.as_mut().unwrap();
        spec.diffusion_var = vec![0.01; spec.n_states];
        assert!(
            mr_scope(&model, &subject, &theta, &eta).is_none(),
            "an SDE (diffusion term) must decline the closed-form MR scope"
        );
        assert!(
            mr_predictions(&model, &subject, &theta, &eta).is_none(),
            "an SDE must decline the MR value path"
        );
        assert!(
            mr_subject_sensitivities(&model, &subject, &theta, &eta).is_none(),
            "an SDE must decline the MR analytic-gradient path"
        );
    }

    #[test]
    fn declines_flip_flop_transit_route() {
        let model = parse(TRANSIT_FLIPFLOP);
        // ke = CL/V = 60/10 = 6 ≥ KTR = (2+1)/1 = 3 → flip-flop, no closed form.
        let subject = mk_subject(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            vec![0.5, 1.0, 2.0],
        );
        assert!(
            mr_predictions(&model, &subject, &[60.0, 10.0, 2.0, 1.0], &[0.0]).is_none(),
            "a transit route past its flip-flop bound must decline"
        );
    }

    #[test]
    fn declines_steady_state_and_infusion_doses() {
        let model = parse(REDUCE_1CPT);
        let theta = [5.0, 50.0, 0.6, 1.5, 0.3];
        let obs = vec![1.0, 4.0, 8.0];
        // Steady-state dose.
        let ss = mk_subject(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)],
            obs.clone(),
        );
        assert!(
            mr_predictions(&model, &ss, &theta, &[0.0, 0.0]).is_none(),
            "SS dose declines"
        );
        // Infusion dose (RATE > 0).
        let inf = mk_subject(vec![DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0)], obs);
        assert!(
            mr_predictions(&model, &inf, &theta, &[0.0, 0.0]).is_none(),
            "infusion dose declines"
        );
    }

    #[test]
    fn route_conc_g_weibull_is_inert() {
        // The defensive arm in `route_conc_g` (the gate excludes this kind, so it
        // is never reached in production) returns 0 for both dispositions.
        let p = vec![2.0_f64];
        let one = DispParams {
            cl: 5.0,
            v: 50.0,
            periph: None,
        };
        let two = DispParams {
            cl: 5.0,
            v: 50.0,
            periph: Some((10.0, 100.0)),
        };
        for dp in [one, two] {
            let f = forcing(InputRateKind::Weibull, vec![0], None, None);
            let got = route_conc_g::<f64>(InputRateKind::Weibull, &f, &p, &dp, 2.0, 100.0);
            assert_eq!(got, 0.0, "Weibull must be inert in route_conc_g");
        }
    }

    // ── #860 Phase B: zero_order box-car kernel. ──────────────────────────────

    #[test]
    fn single_zero_order_route_matches_infusion_kernel() {
        // A single, unlagged, full-fraction zero_order route must match the
        // fixed-duration infusion kernel exactly (same box-car-into-1cpt physics,
        // `dur` generic here vs `f64` there) — the bit-reduction anchor for the
        // new kernel, mirroring `single_first_order_route_reduces_to_bateman`.
        let dur = 2.5_f64;
        let p = vec![dur];
        let dp = DispParams {
            cl: 5.0,
            v: 50.0,
            periph: None,
        };
        let f = vec![forcing(InputRateKind::ZeroOrder, vec![0], None, None)];
        let doses = vec![dose(0.0, 100.0)];
        let rate = 100.0 / dur;
        for &t in &[0.5, 1.0, dur, 2.0 * dur, 8.0] {
            let got = mr_observable_g(&dp, &f, &doses, t, &p, 0.0, 1.0);
            let want = crate::sens::one_cpt::one_cpt_infusion_g(rate, dur, 100.0, t, dp.cl, dp.v);
            assert!((got - want).abs() < 1e-9, "t={t}: {got} vs {want}");
        }
    }

    #[test]
    fn zero_order_dual2_matches_fd_near_dur_boundary() {
        // The real risk in #860 Phase B: `∂/∂dur` near the moving t == dur
        // boundary. Central-FD the f64 kernel against the analytic Dual2 kernel
        // at observation times straddling `dur` (just below/just above), 1-cpt
        // and 2-cpt. `t == dur` exactly is deliberately excluded: there, central
        // FD perturbs `dur` past the *fixed* `t`, so the finite difference mixes
        // the during- and after-window branches (a genuine secant across a kink,
        // not a one-sided derivative) — no analytic one-sided derivative can
        // match it, and no real observation time lands there exactly.
        use crate::sens::dual2::Dual2;
        let h = 1e-5;
        let cl = 5.0_f64;
        let v = 50.0_f64;
        let q = 3.0_f64;
        let v2 = 80.0_f64;
        let dur0 = 3.0_f64;
        for &t in &[1.0, 2.999, 3.001, 6.0] {
            // 1-cpt.
            let plus = one_cpt_zero_order_g(1.0, t, cl, v, dur0 + h, 1.0);
            let minus = one_cpt_zero_order_g(1.0, t, cl, v, dur0 - h, 1.0);
            let fd = (plus - minus) / (2.0 * h);
            let dur_d = Dual2::<1>::var(dur0, 0);
            let got = one_cpt_zero_order_g(
                1.0,
                Dual2::<1>::constant(t),
                Dual2::constant(cl),
                Dual2::constant(v),
                dur_d,
                Dual2::constant(1.0),
            );
            assert!(
                (got.grad[0] - fd).abs() < 1e-4 * (1.0 + fd.abs()),
                "1cpt t={t}: analytic {} vs FD {fd}",
                got.grad[0]
            );
            // 2-cpt.
            let plus2 = two_cpt_zero_order_g(1.0, t, cl, v, q, v2, dur0 + h, 1.0);
            let minus2 = two_cpt_zero_order_g(1.0, t, cl, v, q, v2, dur0 - h, 1.0);
            let fd2 = (plus2 - minus2) / (2.0 * h);
            let got2 = two_cpt_zero_order_g(
                1.0,
                Dual2::<1>::constant(t),
                Dual2::constant(cl),
                Dual2::constant(v),
                Dual2::constant(q),
                Dual2::constant(v2),
                dur_d,
                Dual2::constant(1.0),
            );
            assert!(
                (got2.grad[0] - fd2).abs() < 1e-4 * (1.0 + fd2.abs()),
                "2cpt t={t}: analytic {} vs FD {fd2}",
                got2.grad[0]
            );
        }
    }

    // ── Review regressions (PR #889): each of these was silently mispredicted
    //    before the fix, and is outside the #505/#856 shape the other tests use. ─

    // H1: a readout with a non-zero intercept (baseline) — `Σ mass·kernel` drops
    // the baseline, so it must decline (the readout is not linear-through-origin).
    const BASELINE_READOUT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.5, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  KA = TVKA
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central
[scaling]
  y = central / V + 5
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    #[test]
    fn declines_baseline_readout() {
        let model = parse(BASELINE_READOUT);
        let subject = mk_subject(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            vec![1.0, 2.0, 4.0],
        );
        assert!(
            mr_predictions(&model, &subject, &[5.0, 50.0, 1.5], &[0.0]).is_none(),
            "a readout with a baseline (non-zero intercept) must decline"
        );
    }

    // H2: a dose into a non-central compartment is a bolus the superposition does
    // not represent — must decline (2-cpt model so cmt 2 is a real state).
    #[test]
    fn declines_dose_into_noncentral_compartment() {
        let model = parse(REDUCE_2CPT);
        let theta = [5.0, 50.0, 10.0, 100.0, 0.6, 1.5, 0.3];
        let central_ok = mk_subject(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            vec![1.0, 4.0],
        );
        assert!(
            mr_predictions(&model, &central_ok, &theta, &[0.0]).is_some(),
            "sanity: a central dose is served"
        );
        let periph_dose = mk_subject(
            vec![DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0)],
            vec![1.0, 4.0],
        );
        assert!(
            mr_predictions(&model, &periph_dose, &theta, &[0.0]).is_none(),
            "a dose into the peripheral compartment must decline"
        );
    }

    // H3: a 2-cpt transit route whose ke = CL/V1 is inside the (1-cpt) bound but
    // whose fast macro-rate α is NOT — the kernel would return a silent 0, so the
    // 2-cpt domain check must decline it. Positive twin: an in-domain 2-cpt
    // transit reduces to ODE.
    const TWOCPT_TRANSIT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(10.0, 0.1, 100.0)
  theta TVV2(100.0, 5.0, 1000.0)
  theta TVN(3.0, 0.1, 20.0)
  theta TVMTT(1.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q = TVQ
  V2 = TVV2
  N = TVN
  MTT = TVMTT
[structural_model]
  ode(states=[central, periph])
[odes]
  d/dt(central) = transit(n=N, mtt=MTT) - CL/V1*central - Q/V1*central + Q/V2*periph
  d/dt(periph) = Q/V1*central - Q/V2*periph
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    #[test]
    fn reduces_to_ode_2cpt_transit_in_domain() {
        // k10=0.1, k12=0.2, k21=0.1 → α≈0.37 < KTR=(3+1)/1=4: in domain.
        assert_reduces_to_ode(
            TWOCPT_TRANSIT,
            &[5.0, 50.0, 10.0, 100.0, 3.0, 1.0],
            &[0.0],
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        );
    }

    #[test]
    fn declines_2cpt_transit_when_macro_rate_out_of_domain() {
        let model = parse(TWOCPT_TRANSIT);
        // CL=5,V1=50 → k10=0.1; Q=100,V2=10 → k12=2, k21=10 → α≈12.
        // KTR=(1+1)/0.5=4. So ke=k10=0.1 < 4 (the OLD 1-cpt check WRONGLY passes),
        // but α≈12 ≥ 4 (the kernel returns 0) → the 2-cpt check must decline.
        let theta = [5.0, 50.0, 100.0, 10.0, 1.0, 0.5];
        let subject = mk_subject(
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            vec![0.5, 1.0, 2.0],
        );
        assert!(
            mr_predictions(&model, &subject, &theta, &[0.0]).is_none(),
            "a 2-cpt transit route past its macro-rate domain (α≥KTR while k10<KTR) must decline"
        );
    }

    // ---------------------------------------------------------------------
    // #1124: a model-time-reading `[odes]` RHS must never reach the closed form.
    //
    // Every fixture here is **multi-dose**. With a single dose `TAD == TAFD`, so
    // a one-dose subject cannot distinguish a `TAD` defect from a `TAFD` one and
    // would pass against the wrong anchor (`nonmem_anchor/tad_lag_A.ctl` is
    // deliberately single-dose for the opposite reason — it wants no `MTIME`).
    // ---------------------------------------------------------------------

    /// MR-shaped base model — mixed `first_order` + `zero_order` absorption into
    /// one central compartment, the shape `mr_scope` admits. `elim_factor` is
    /// spliced onto the elimination term and `pre` in front of the derivative
    /// statement, so a case differs from the control by exactly one edit.
    fn tad_model(pre: &str, elim_factor: &str) -> String {
        format!(
            r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVDUR(4.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  FZO = TVFZO
  FZO1 = 1 - TVFZO
  KA = TVKA
  DUR = TVDUR
[structural_model]
  ode(states=[central])
[odes]
{pre}  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central{elim_factor}
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#
        )
    }

    const TAD_THETA: [f64; 5] = [5.0, 50.0, 0.4, 1.0, 4.0];

    /// Two doses, so the `TAD` anchor genuinely switches mid-record-interval.
    fn tad_subject() -> crate::types::Subject {
        mk_subject(
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
            ],
            (1..=30).map(|i| i as f64 * 0.4).collect(),
        )
    }

    /// The event-driven ODE walk — the reference for every case here. It places
    /// each dose on its timeline at `d.time + lag`, so its `TAD` anchor is the
    /// arrival; it is the engine the NONMEM `TAD`-in-`$DES` anchors validate
    /// (`nonmem_anchor/tad_lag_{A,B}`), and the one production routes to.
    fn event_twin(
        model: &crate::types::CompiledModel,
        subject: &crate::types::Subject,
        theta: &[f64],
        eta: &[f64],
    ) -> Vec<f64> {
        let spec = model.ode_spec.as_ref().expect("ode model");
        let mut sc = crate::pk::EventPkParams::with_capacity_for(subject);
        crate::pk::compute_event_pk_params_into(model, subject, theta, eta, &mut sc);
        crate::ode::ode_predictions_event_driven(
            spec,
            subject,
            theta,
            eta,
            &sc.dose,
            &sc.obs,
            &sc.pk_only,
        )
    }

    /// The plain **dense** ODE driver: a second engine, not the one production
    /// takes. It walks one timeline from the first record and anchors `TAD` on
    /// doses that have already arrived, so for a model with **no lagtime** it is
    /// an independent reference for the same quantity. (Under a lagtime it is not
    /// — it returns `NaN` for every observation, which is #1110; use
    /// [`event_twin`] there and say so.)
    fn dense_twin(
        model: &crate::types::CompiledModel,
        subject: &crate::types::Subject,
        theta: &[f64],
        eta: &[f64],
    ) -> Vec<f64> {
        let spec = model.ode_spec.as_ref().expect("ode model");
        let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
        crate::ode::ode_predictions(spec, &pk.values, theta, eta, subject)
    }

    /// The whole point of #1124: production must integrate the `TAD` term, not
    /// drop it. Before the fix the closed form served this subject and returned
    /// numbers 8× the truth at 12 h.
    ///
    /// Checked against [`dense_twin`], **not** [`event_twin`]. Post-fix production
    /// *is* the event-driven walk for this model — same `EventPkParams`, same
    /// `ode_predictions_event_driven` call, and every post-walk step is a no-op
    /// here (Form-C ODE readout ⇒ `model.scaling == None`, no analytical init, no
    /// LTBS) — so an `event_twin` comparison is exactly `0.0` by construction and
    /// its tolerance is never exercised. That is the same by-construction
    /// agreement that hid #1124 inside the closed form (`mr_vs_prod` was likewise
    /// exactly `0.0`), one layer up. `tad_model` carries no lagtime, so the dense
    /// driver is a genuinely separate engine here.
    #[test]
    fn tad_reading_rhs_reduces_to_its_ode_twin() {
        let model =
            crate::parser::model_parser::parse_model_string(&tad_model("", "*(1.0 + 0.3*TAD)"))
                .unwrap();
        let subject = tad_subject();
        let eta = [0.0];
        let prod = crate::pk::compute_predictions_with_tv(&model, &subject, &TAD_THETA, &eta);
        let twin = dense_twin(&model, &subject, &TAD_THETA, &eta);
        // The two engines must differ *somewhere* in the last bits, else this has
        // silently become a self-comparison again (the failure mode above).
        let mut any_difference = false;
        for (i, (&a, &b)) in prod.iter().zip(&twin).enumerate() {
            assert!(
                a.is_finite() && b.is_finite(),
                "obs {i} (t={}): production {a}, dense twin {b} — both engines must \
                 serve a no-lagtime `TAD` model",
                subject.obs_times[i],
            );
            assert!(
                (a - b).abs() < 1e-6 * (1.0 + b.abs()),
                "obs {i} (t={}): production {a} vs dense twin {b}",
                subject.obs_times[i],
            );
            any_difference |= a.to_bits() != b.to_bits();
        }
        assert!(
            any_difference,
            "production and the dense twin agreed bit-for-bit at every observation — \
             the comparison has collapsed onto one engine and proves nothing"
        );
    }

    /// The sharp form of the same claim, and the one that actually localises the
    /// defect: the closed form returned values **bit-identical** to the model
    /// with the `TAD` factor deleted, which is what "the term is dropped, not
    /// approximated" means. A tolerance check against the twin can be satisfied
    /// by a merely-close wrong answer; this cannot.
    #[test]
    fn the_tad_term_is_live_in_the_predictions() {
        let with_tad =
            crate::parser::model_parser::parse_model_string(&tad_model("", "*(1.0 + 0.3*TAD)"))
                .unwrap();
        let without = crate::parser::model_parser::parse_model_string(&tad_model("", "")).unwrap();
        let subject = tad_subject();
        let eta = [0.0];
        let a = crate::pk::compute_predictions_with_tv(&with_tad, &subject, &TAD_THETA, &eta);
        let b = crate::pk::compute_predictions_with_tv(&without, &subject, &TAD_THETA, &eta);
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
            .fold(0.0f64, f64::max);
        assert!(
            worst > 1e-2,
            "predictions are within {worst:.3e} of the same model with the `TAD` \
             factor deleted — the term is being dropped, not integrated"
        );
    }

    /// `mr_scope` must decline model time in **every** spelling, including from
    /// inside an `if` condition. The behavioural probe in `identify_disposition`
    /// samples two times with `tad` pinned, so it sees none of these: bare `TAD`
    /// is constant across both probes, and a branch that neither probe time
    /// enters is unreachable regardless of which variable it tests.
    #[test]
    fn mr_scope_declines_every_model_time_spelling() {
        let subject = tad_subject();
        let eta = [0.0];

        // Control: no model time anywhere — must still be served by the fast path,
        // else this test would pass for the trivial reason that nothing is admitted.
        let control = crate::parser::model_parser::parse_model_string(&tad_model("", "")).unwrap();
        assert!(
            mr_scope(&control, &subject, &TAD_THETA, &eta).is_some(),
            "the control model must stay in scope — otherwise the declines below \
             prove nothing"
        );

        // Read directly in the derivative. Only `TAD` was wrongly admitted before
        // the fix — the other three already failed the two-time probe — but all
        // four are pinned so a future narrowing of this gate (to
        // `uses_dose_anchored_time_vars`, say) cannot quietly reopen them.
        // `TAFD - TAD` is here because it is what breaks if the probe is "fixed"
        // by passing a varying `tad`: the two anchors cancel to a constant.
        for factor in [
            "*(1.0 + 0.3*TAD)",
            "*(1.0 + 0.3*TAFD)",
            "*(1.0 + 0.3*T)",
            "*(1.0 + 0.3*TIME)",
            "*(1.0 + 0.3*(TAFD - TAD))",
        ] {
            let m =
                crate::parser::model_parser::parse_model_string(&tad_model("", factor)).unwrap();
            assert!(
                mr_scope(&m, &subject, &TAD_THETA, &eta).is_none(),
                "must decline a RHS reading model time: {factor}"
            );
        }

        // Threshold inside the observation window, so each branch really fires.
        for var in ["TAD", "TAFD", "T", "TIME"] {
            let pre =
                format!("  if ({var} > 6.0) {{\n    KE = 3.0\n  }} else {{\n    KE = 1.0\n  }}\n");
            let m =
                crate::parser::model_parser::parse_model_string(&tad_model(&pre, "*KE")).unwrap();
            assert!(
                mr_scope(&m, &subject, &TAD_THETA, &eta).is_none(),
                "must decline `{var}` read from an `if` condition"
            );
        }
    }

    /// All three closed-form entry points share `mr_scope`, so none of them may
    /// serve a model the ODE gate has already declined. The spelling matters: an
    /// *indexed* `ALAG1` populates `dose_attr_map`, which `mr_scope` rejects for
    /// an unrelated reason, so it cannot demonstrate this. A bare `ALAG` leaves
    /// that map empty and reaches the closed form — which is how the fast path
    /// bypassed #1070's FD gate and served a jet the gate exists to refuse.
    #[test]
    fn mr_declines_what_the_ode_analytic_gate_declines() {
        let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVDUR(4.0, 0.1, 24.0)
  theta TVLAG(0.5, 0.001, 12.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  FZO = TVFZO
  FZO1 = 1 - TVFZO
  KA = TVKA
  DUR = TVDUR
  ALAG = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central*(1.0 + 0.3*TAD)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        let spec = model.ode_spec.as_ref().unwrap();
        let subject = tad_subject();
        let theta = [5.0, 50.0, 0.4, 1.0, 4.0, 0.5];
        let eta = [0.0];

        // Pin the premise: this really is the bypass shape, not an unrelated decline.
        assert!(model.has_lagtime());
        assert!(
            spec.dose_attr_map.is_empty(),
            "a bare `ALAG` must leave `dose_attr_map` empty — otherwise `mr_scope` \
             declines for that reason and the gate bypass is not exercised"
        );
        assert!(
            !crate::sens::ode_provider::ode_analytical_supported(&model),
            "the #1070 gate must decline this model"
        );

        assert!(mr_scope(&model, &subject, &theta, &eta).is_none());
        assert!(mr_predictions(&model, &subject, &theta, &eta).is_none());
        assert!(mr_subject_sensitivities(&model, &subject, &theta, &eta).is_none());
        assert!(mr_subject_eta_grad(&model, &subject, &theta, &eta).is_none());
    }

    /// #1110: the dense predictor anchors `TAD` on doses that have already
    /// *arrived*, and with a lagtime none has at the first segment's start — so it
    /// folded to `NEG_INFINITY` and poisoned the trajectory with `NaN`. Routing a
    /// model-time-reading RHS to the event-driven walk, which starts at the
    /// arrival, makes that unreachable from production.
    #[test]
    fn a_tad_rhs_under_a_lagtime_predicts_finite_values() {
        let src = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(20.0, 0.001, 500.0)
  theta THETA_TAD(0.3, 0.0, 10.0)
  theta TVLAG(0.5, 0.001, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.02
  sigma PROP ~ 0.1 (sd)
[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  KTAD    = THETA_TAD
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central * (1.0 + KTAD * TAD)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        let subject = mk_subject(
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ],
            vec![2.0, 4.0, 8.0, 24.0],
        );
        let theta = [1.0, 20.0, 0.3, 0.5];
        let eta = [0.0, 0.0];
        let prod = crate::pk::compute_predictions_with_tv(&model, &subject, &theta, &eta);
        assert!(
            prod.iter().all(|p| p.is_finite() && *p > 0.0),
            "predictions must be finite and positive, got {prod:?}"
        );
        // The finite/positive assert above is the load-bearing half — that is the
        // #1110 symptom. The twin check below pins the *route* only: production is
        // the event-driven walk for this model, so this difference is exactly
        // `0.0` by construction and its tolerance is never exercised. The dense
        // engine cannot serve as the independent side here — under a lagtime it is
        // the very thing that returns `NaN` — so the independent references for
        // this cell are the NONMEM `tad_lag_{A,B}` anchors, not this assert.
        let twin = event_twin(&model, &subject, &theta, &eta);
        for (i, (&a, &b)) in prod.iter().zip(&twin).enumerate() {
            assert!(
                (a - b).abs() < 1e-6 * (1.0 + b.abs()),
                "obs {i}: {a} vs {b}"
            );
        }
    }

    /// The half of #1110 this fix does **not** reach, pinned so the CHANGELOG's
    /// scoping stays honest and a later change to `tad_anchor` is visible here.
    ///
    /// An observation recorded before the first dose *arrives* has no `TAD`:
    /// `tad_anchor` finds no qualifying dose and returns `NaN`, which the RHS
    /// multiplies into the state and which then poisons every later observation.
    /// Measured one variable at a time — the lagtime is **not** the trigger:
    ///
    /// | case                                    | result                  |
    /// |-----------------------------------------|-------------------------|
    /// | lag 0.5, obs 2/4/8/24 (all post-arrival)| finite                  |
    /// | lag 0.5, obs 0.4 added                  | `[0.0, NaN, NaN, …]`    |
    /// | lag 0.3, same obs list (0.4 now post)   | finite                  |
    /// | **no lagtime**, dose at 1.0, obs at 0.4 | `[0.0, NaN, NaN, …]`    |
    /// | autonomous RHS, lag 0.5, obs at 0.4     | finite                  |
    ///
    /// Pre-existing and unchanged by the routing fix: the dense predictor this
    /// model used to take produces the identical `[0.0, NaN, …]`. What `TAD`
    /// should mean before the first arrival is the open half of #1110; this test
    /// asserts today's behaviour, not that it is right.
    #[test]
    fn a_pre_arrival_observation_still_nans_a_tad_reading_rhs() {
        // No lagtime at all — the trigger is the pre-arrival window, not the lag.
        let src = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(20.0, 0.001, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central * (1.0 + 0.3*TAD)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        let theta = [1.0, 20.0];
        let eta = [0.0];
        let doses = || {
            vec![
                DoseEvent::new(1.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ]
        };
        // Control: every observation after the first arrival — finite, and the
        // routing fix is what makes it so.
        let after = mk_subject(doses(), vec![2.0, 4.0, 8.0, 24.0]);
        let ok = crate::pk::compute_predictions_with_tv(&model, &after, &theta, &eta);
        assert!(
            ok.iter().all(|p| p.is_finite() && *p > 0.0),
            "post-arrival observations must be finite, got {ok:?}"
        );
        // Add one observation before the dose at t = 1.0. Everything after it is
        // lost too — the `NaN` is carried in the state, not confined to that row.
        let before = mk_subject(doses(), vec![0.4, 2.0, 4.0, 8.0, 24.0]);
        let bad = crate::pk::compute_predictions_with_tv(&model, &before, &theta, &eta);
        assert!(
            bad[1..].iter().all(|p| p.is_nan()),
            "known gap (#1110): one pre-arrival observation should still poison every \
             later one — if this now passes, the pre-arrival `TAD` convention was \
             decided and the CHANGELOG / docs must say so. Got {bad:?}"
        );
        // …and the dense predictor agrees, which is what makes it pre-existing
        // rather than something the reroute introduced.
        let spec = model.ode_spec.as_ref().unwrap();
        let dense = crate::ode::ode_predictions(
            spec,
            &flat_params(&model, &theta, &eta),
            &theta,
            &eta,
            &before,
        );
        assert!(
            dense[1..].iter().all(|p| p.is_nan()),
            "the dense predictor this model used to take must show the same gap, \
             got {dense:?}"
        );
    }

    /// The routing widening also moves models that were **already correct** — a
    /// `TAFD`-reading RHS with no lagtime was served fine by the dense path. That
    /// must be a pure re-route, not a change in what ferx predicts.
    #[test]
    fn rerouting_a_bare_tafd_rhs_does_not_move_the_predictions() {
        let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  KA = TVKA
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central*(1.0 + 0.05*TAFD)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        let spec = model.ode_spec.as_ref().unwrap();
        let subject = tad_subject();
        let theta = [5.0, 50.0, 1.0];
        let eta = [0.0];
        let dense = crate::ode::ode_predictions(
            spec,
            &flat_params(&model, &theta, &eta),
            &theta,
            &eta,
            &subject,
        );
        let prod = crate::pk::compute_predictions_with_tv(&model, &subject, &theta, &eta);
        for (i, (&a, &b)) in prod.iter().zip(&dense).enumerate() {
            assert!(
                (a - b).abs() < 1e-9 * (1.0 + b.abs()),
                "obs {i}: rerouted {a} vs the dense path it used to take {b}"
            );
        }
    }

    /// The same claim for the models the closed form declined for reasons of its
    /// **own**, which the widened predicate still moves off the dense path.
    ///
    /// `mr_scope` rejects `init(...)`, a `dose_attr_map`, and `ss || is_infusion`
    /// before it ever looks at model time, so those subjects were already on the
    /// dense predictor; widening the routing guard moves them to the event-driven
    /// one. The two engines are known to seed `init` from different snapshots
    /// (`initial_state(pk_params_flat)` at `t = 0` versus
    /// `initial_state(&init_pk.values)` at the first record), so this is not
    /// covered by the plain-bolus case above.
    ///
    /// Steady state is deliberately **not** in this test: on a non-autonomous RHS
    /// both engines are wrong there — `NaN` for `TAD`/`TAFD` even at a zero
    /// coefficient, and 17% off an explicit 40-cycle pulse train for `T`/`TIME` —
    /// so an engine-vs-engine assert would pin agreement on a wrong answer. See
    /// `SS_NONAUTONOMOUS_ISSUE`; the routing neither causes nor cures it.
    #[test]
    fn rerouting_an_init_seeded_rhs_does_not_move_the_predictions() {
        let src = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(20.0, 0.001, 500.0)
  theta TVB(7.0, 0.001, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.1 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  BASE = TVB
[structural_model]
  ode(states=[central])
[odes]
  init(central) = BASE
  d/dt(central) = -(CL/V) * central * (1.0 + 0.3*TAFD)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        let spec = model.ode_spec.as_ref().unwrap();
        // Doses at 1 and 12 with the baseline live from the first record, so the
        // `init` seed is a real quantity on both sides rather than a zero state.
        let subject = mk_subject(
            vec![
                DoseEvent::new(1.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ],
            vec![2.0, 4.0, 8.0, 24.0],
        );
        let theta = [1.0, 20.0, 7.0];
        let eta = [0.0];
        assert!(
            mr_scope(&model, &subject, &theta, &eta).is_none(),
            "precondition: `init(...)` is out of closed-form scope, so this subject \
             was on the dense path before the routing widened"
        );
        // …and the reroute must actually be happening. Without this the test is
        // satisfied by a *narrowed* predicate too: production would then also take
        // the dense path and `prod == dense` would hold trivially, proving nothing
        // about the engine this model was moved to.
        assert!(
            crate::pk::model_uses_time_anywhere(&model),
            "precondition: the widened predicate must route this model off the dense \
             path — otherwise the comparison below is production against itself"
        );
        assert!(
            !crate::pk::model_uses_time_builtin(&model),
            "…and it must be the `[odes]` RHS doing it, not a `TIME`-reading \
             individual parameter, which the narrow predicate already caught"
        );
        let dense = crate::ode::ode_predictions(
            spec,
            &flat_params(&model, &theta, &eta),
            &theta,
            &eta,
            &subject,
        );
        let prod = crate::pk::compute_predictions_with_tv(&model, &subject, &theta, &eta);
        assert!(
            prod.iter().all(|p| p.is_finite()) && dense.iter().all(|p| p.is_finite()),
            "both engines must serve this model: rerouted {prod:?}, dense {dense:?}"
        );
        for (i, (&a, &b)) in prod.iter().zip(&dense).enumerate() {
            assert!(
                (a - b).abs() < 1e-8 * (1.0 + b.abs()),
                "obs {i}: rerouted {a} vs the dense path it used to take {b}"
            );
        }
    }

    /// The cross-entry-point invariant, and the oracle #1124 needed.
    ///
    /// `compute_predictions_with_tv` may take the closed form;
    /// `compute_predictions_with_states` has **no** modified-release branch and
    /// always integrates. That asymmetry is deliberate and load-bearing: it makes
    /// the two entry points genuinely independent, so comparing them tests the
    /// reduction rather than confirming it by construction (the routing site
    /// itself cannot — it *is* the closed form there). Do not "fix" the
    /// inconsistency by giving the states path an MR branch; that would make both
    /// sides wrong together, which is exactly how the 8× error stayed invisible.
    #[test]
    fn mr_matches_the_states_entry_point() {
        let zoo: &[(&str, &[f64], &[f64])] = &[
            (REDUCE_1CPT, &[5.0, 50.0, 0.6, 1.5, 0.3], &[0.0, 0.0]),
            (REDUCE_1CPT, &[5.0, 50.0, 0.6, 1.5, 0.3], &[0.3, -0.2]),
            (REDUCE_1CPT_LAG, &[5.0, 50.0, 0.5, 1.2, 0.8, 3.0], &[0.0]),
            (
                REDUCE_2CPT,
                &[5.0, 50.0, 10.0, 100.0, 0.6, 1.5, 0.3],
                &[0.0],
            ),
        ];
        for (i, (src, theta, eta)) in zoo.iter().enumerate() {
            let model = crate::parser::model_parser::parse_model_string(src).unwrap();
            let subject = tad_subject();
            // Non-degeneracy: this only tests anything while the fast path is live.
            assert!(
                mr_predictions(&model, &subject, theta, eta).is_some(),
                "zoo model {i} left `mr_scope` — the comparison below is vacuous"
            );
            let via_tv = crate::pk::compute_predictions_with_tv(&model, &subject, theta, eta);
            let (via_states, _) =
                crate::pk::compute_predictions_with_states(&model, &subject, theta, eta);
            for (j, (&a, &b)) in via_tv.iter().zip(&via_states).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5 * (1.0 + b.abs()),
                    "zoo model {i}, obs {j}: `with_tv` {a} vs `with_states` {b}"
                );
            }
        }
    }

    /// The verify-gate must actually reject a closed form that disagrees with its
    /// twin. Asserted by handing it a perturbed vector rather than a real defect,
    /// because a model that is admitted *and* wrong is precisely what no longer
    /// exists — this pins the mechanism (tolerance, message, which side is which)
    /// so the gate cannot rot into a no-op.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "disagrees with its ODE twin")]
    fn the_verify_gate_rejects_a_closed_form_that_misses_its_twin() {
        let model = crate::parser::model_parser::parse_model_string(REDUCE_1CPT).unwrap();
        let subject = tad_subject();
        let theta = [5.0, 50.0, 0.6, 1.5, 0.3];
        let eta = [0.0, 0.0];
        let s = mr_scope(&model, &subject, &theta, &eta).expect("in scope");
        // Built from the twin rather than from `mr_predictions`, which would run
        // the gate itself and consume this subject's memo entry — leaving the call
        // below a no-op and the test green for the wrong reason.
        let mut preds = crate::ode::ode_predictions(s.spec, &s.pk.values, &theta, &eta, &subject);
        // A 1 % error at one observation — far under the defect this guards
        // against (1e-1) and far over the tolerance (1e-4).
        preds[10] *= 1.01;
        verify_against_ode_twin(&s, &subject, &theta, &eta, &preds);
    }

    /// The memo must **not** retire a `(spec, subject, bucket)` on a call that
    /// compared nothing.
    ///
    /// Every non-finite twin entry is skipped, so a twin that fails to integrate
    /// can make a whole call vacuous. With the key recorded up front, that one
    /// call would silently retire the pair forever and every later parameter point
    /// in the same bucket would go unchecked — a gate that reports success without
    /// ever having run. Asserted by making the vacuous call first and then showing
    /// the gate is still live: the perturbed vector must still panic.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "disagrees with its ODE twin")]
    fn a_vacuous_verify_gate_call_does_not_retire_its_memo_key() {
        let model = crate::parser::model_parser::parse_model_string(REDUCE_1CPT).unwrap();
        let subject = tad_subject();
        let theta = [5.0, 50.0, 0.6, 1.5, 0.3];
        let eta = [0.0, 0.0];
        let s = mr_scope(&model, &subject, &theta, &eta).expect("in scope");
        // A length mismatch: nothing is compared, so nothing was verified. This is
        // the one vacuity path reachable from outside — the other (`compared == 0`,
        // every twin entry non-finite) needs the *twin* to be NaN, and the gate
        // integrates that itself from `s.spec`, so no caller can arrange it. Both
        // paths release the same key; only this one has a test, and the
        // `compared == 0` release is covered by mutation instead.
        verify_against_ode_twin(&s, &subject, &theta, &eta, &[]);
        // Same (spec, subject, bucket). If the vacuous call had kept the key, this
        // would return early and the test would fail by *not* panicking.
        let mut preds = crate::ode::ode_predictions(s.spec, &s.pk.values, &theta, &eta, &subject);
        preds[10] *= 1.01;
        verify_against_ode_twin(&s, &subject, &theta, &eta, &preds);
    }

    /// The memo must re-verify once the **parameter point** moves materially.
    ///
    /// Admission is not purely a property of the model and subject shape:
    /// `mr_scope` probes `identify_disposition` numerically at `(θ, η)` and runs
    /// the parameter-dependent `route_flips` domain check. Keyed on
    /// `(spec, subject)` alone the gate would validate only the first point a fit
    /// visits — the initial estimates — and never re-check a closed form degrading
    /// as the optimiser walks. The disposition bucket in the key is what prevents
    /// that; this pins it.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "disagrees with its ODE twin")]
    fn the_verify_gate_rechecks_after_the_disposition_moves() {
        let model = crate::parser::model_parser::parse_model_string(REDUCE_1CPT).unwrap();
        let subject = tad_subject();
        let eta = [0.0, 0.0];
        // Point 1: verified clean, which records the memo key for its bucket.
        let theta_a = [5.0, 50.0, 0.6, 1.5, 0.3];
        let s_a = mr_scope(&model, &subject, &theta_a, &eta).expect("in scope");
        let twin_a =
            crate::ode::ode_predictions(s_a.spec, &s_a.pk.values, &theta_a, &eta, &subject);
        verify_against_ode_twin(&s_a, &subject, &theta_a, &eta, &twin_a);

        // Point 2: CL moved 5 → 15, a different disposition bucket on the same
        // spec and subject. The gate must run again here.
        let theta_b = [15.0, 50.0, 0.6, 1.5, 0.3];
        let s_b = mr_scope(&model, &subject, &theta_b, &eta).expect("in scope");
        assert_ne!(
            disp_bucket(&s_a.dp),
            disp_bucket(&s_b.dp),
            "precondition: the two parameter points must land in different buckets, \
             else this test would pass for the trivial reason that nothing moved"
        );
        let mut preds =
            crate::ode::ode_predictions(s_b.spec, &s_b.pk.values, &theta_b, &eta, &subject);
        preds[10] *= 1.01;
        verify_against_ode_twin(&s_b, &subject, &theta_b, &eta, &preds);
    }

    /// …and the bucket must be **coarse**: an ordinary optimiser step that barely
    /// moves the disposition must still hit the memo, or the gate degrades to the
    /// unmemoized ~2× cost it was written to avoid.
    #[cfg(debug_assertions)]
    #[test]
    fn the_disposition_bucket_is_coarse_enough_to_memoize() {
        let model = crate::parser::model_parser::parse_model_string(REDUCE_1CPT).unwrap();
        let subject = tad_subject();
        let eta = [0.0, 0.0];
        let base = [5.0, 50.0, 0.6, 1.5, 0.3];
        let s_base = mr_scope(&model, &subject, &base, &eta).expect("in scope");
        // A 0.1 % nudge in CL — the scale of an FD probe or a late optimiser step.
        let nudged = [5.005, 50.0, 0.6, 1.5, 0.3];
        let s_nudged = mr_scope(&model, &subject, &nudged, &eta).expect("in scope");
        assert_eq!(
            disp_bucket(&s_base.dp),
            disp_bucket(&s_nudged.dp),
            "a 0.1% parameter nudge must stay in one bucket"
        );
        // A decade apart must not.
        let far = [50.0, 50.0, 0.6, 1.5, 0.3];
        let s_far = mr_scope(&model, &subject, &far, &eta).expect("in scope");
        assert_ne!(
            disp_bucket(&s_base.dp),
            disp_bucket(&s_far.dp),
            "a 10× parameter move must land in a different bucket"
        );
    }

    /// The other direction: a closed form that *does* agree must pass cleanly.
    /// Without this, a gate that panicked unconditionally would satisfy the
    /// rejection test above and quietly break every admitted model in debug.
    ///
    /// Note what this is and is not. It feeds the twin vector back in as the
    /// closed form, so the comparison is `twin == twin` and it holds for **any**
    /// tolerance down to `0.0` — including a `verify_tol = 0.0` that would panic
    /// on every real admitted model in a debug build. Catching an
    /// unconditionally-panicking gate is its whole job; the tolerance is pinned by
    /// `the_verify_gate_stays_quiet_at_default_solver_tolerances`, which goes
    /// through `mr_predictions` with a real closed form on the other side.
    #[cfg(debug_assertions)]
    #[test]
    fn the_verify_gate_accepts_an_agreeing_closed_form() {
        let model = crate::parser::model_parser::parse_model_string(REDUCE_1CPT).unwrap();
        let subject = tad_subject();
        let theta = [5.0, 50.0, 0.6, 1.5, 0.3];
        let eta = [0.0, 0.0];
        let s = mr_scope(&model, &subject, &theta, &eta).expect("in scope");
        let twin = crate::ode::ode_predictions(s.spec, &s.pk.values, &theta, &eta, &subject);
        verify_against_ode_twin(&s, &subject, &theta, &eta, &twin);
    }

    /// …and it must stay quiet at the **default** solver tolerances, over enough
    /// doses for the twin's own integration error to accumulate.
    ///
    /// That is the regime a fixed `1e-4` bound got wrong. The zoo fixtures pin
    /// `reltol = 1e-10`, so they never exercised it; a model that leaves the
    /// default `1e-4` produces a twin honestly differing from the exact closed
    /// form by up to 6.4e-3 — 64× `reltol`, and growing with dose count. The first
    /// version of this gate panicked on an ordinary multi-dose oral model for
    /// exactly that reason, and it was the NONMEM anchor that caught it, not any
    /// unit test. This is that missing unit test.
    #[cfg(debug_assertions)]
    #[test]
    fn the_verify_gate_stays_quiet_at_default_solver_tolerances() {
        let strip = |s: &str| -> String {
            s.replace("  ode_reltol = 1e-10\n", "")
                .replace("  ode_abstol = 1e-12\n", "")
        };
        let zoo: Vec<(String, Vec<f64>, Vec<f64>)> = vec![
            (
                strip(REDUCE_1CPT),
                vec![5.0, 50.0, 0.6, 1.5, 0.3],
                vec![0.3, -0.2],
            ),
            (
                strip(REDUCE_1CPT_LAG),
                vec![5.0, 50.0, 0.5, 1.2, 0.8, 3.0],
                vec![0.0],
            ),
            (
                strip(REDUCE_2CPT),
                vec![5.0, 50.0, 10.0, 100.0, 0.6, 1.5, 0.3],
                vec![0.0],
            ),
        ];
        for (src, theta, eta) in &zoo {
            let model = crate::parser::model_parser::parse_model_string(src).unwrap();
            let spec = model.ode_spec.as_ref().unwrap();
            assert!(
                (spec.solver_opts.reltol - 1e-4).abs() < 1e-12,
                "this test is only meaningful at the DEFAULT reltol; got {}",
                spec.solver_opts.reltol
            );
            let subject = mk_subject(
                vec![
                    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
                ],
                (1..=90).map(|i| i as f64 * 0.4).collect(),
            );
            // Through `mr_predictions`, so the gate runs exactly as production
            // reaches it rather than via a hand-assembled call.
            let preds =
                mr_predictions(&model, &subject, theta, eta).expect("zoo model must stay in scope");
            assert!(preds.iter().all(|p| p.is_finite()));
        }
    }
}
