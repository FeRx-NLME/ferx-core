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
/// (for two-compartment) peripheral. Built once per model by
/// [`identify_disposition`]; the per-subject rates are recovered separately over
/// `T` by [`recover_disp_params_g`] so `T = Dual2` carries their sensitivities.
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
/// nonlinear or non-canonical RHS fails. The runtime verify-gate is the real
/// safety net; these only pick *which* closed form to try.
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
    // `tafd`/`tad` are constants w.r.t. the parameters; a supported (static)
    // disposition is autonomous so their value is irrelevant to the Jacobian.
    prog.eval_rhs_g::<f64>(u, p, t, t, 0.0, &mut du, &mut vars, &mut stack);
    Some(du)
}

/// Identify the linear disposition of `spec` by numeric probing — its shape and
/// the central/peripheral state indices — or `None` if it is not a canonical
/// linear 1-/2-cpt system with all input-rate forcings into one central
/// compartment. Structure only; per-subject rates come from
/// [`recover_disp_params_g`]. `p` is a representative (positive) parameter point
/// and `t` a probe time; the checks are parameter-point-agnostic in structure,
/// and the runtime verify-gate re-validates numerically.
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
/// (degenerate) — the verify-gate/caller then declines to the ODE path.
pub fn recover_disp_params_g<T: PkNum>(
    spec: &OdeSpec,
    disp: &MrDisposition,
    p: &[T],
    cov: &std::collections::HashMap<String, f64>,
) -> Option<DispParams<T>> {
    let prog = spec.rhs_program.as_ref()?;
    let ro = spec.readout_program.as_ref()?;
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
    // A `TIME`-dependent model is time-varying: the single `t = 0` parameter
    // snapshot below cannot represent it, and a `TIME`-via-parameter dependence is
    // invisible to `identify_disposition`'s fixed-`p` Jacobian probe (only
    // `TIME` *in the RHS* fails the time-invariance check). Decline it here so a
    // direct caller is safe, independent of the routing site's own guard.
    if crate::pk::model_uses_time_builtin(model) {
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
    if subject
        .doses
        .iter()
        .any(|d| d.cmt.saturating_sub(1) != disp.central)
    {
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
    Some(
        subject
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
            .collect(),
    )
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
        out.push(crate::sens::provider::ObsSens {
            f: fd.value,
            df_deta,
            d2f_deta2,
            df_dtheta,
            d2f_deta_dtheta,
        });
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
}
