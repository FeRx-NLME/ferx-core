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
//! A model is only routed here when a **runtime verify-gate** (closed form vs a
//! short reference integration of its own `OdeSpec`, at several probe points)
//! agrees — so a mis-identification cannot mispredict, it merely declines back to
//! the ODE path (`crate::pk::mr_supported`). The failure surface is "missed
//! speed-up", never "wrong".

use crate::ode::predictions::OdeSpec;
use crate::pk::absorption::{InputRateForcing, InputRateKind};
use crate::sens::num::PkNum;
use crate::sens::one_cpt::{one_cpt_ig_g, one_cpt_oral_g, one_cpt_transit_g};
use crate::sens::two_cpt::{two_cpt_ig_g, two_cpt_oral_g, two_cpt_transit_g};
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

    // No autonomous term: f(0, p, t) ≈ 0.
    let du0 = probe_rhs_f64(spec, &vec![0.0; n], p, t)?;
    if du0.iter().any(|d| d.abs() > LINEARITY_TOL) {
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
    let y0 = ro.eval_output_g::<T>(&zero_state, p, cov, &mut vars, &mut stack);
    let y1 = ro.eval_output_g::<T>(&e_c, p, cov, &mut vars, &mut stack);
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
/// own `τ.val() < 0 → 0` guard supplies the pre-onset gate. Phase A covers the
/// smooth kernels (`first_order`/`transit`/`igd`); `zero_order` is Phase B (its
/// `∂/∂dur` carries a moving-boundary saltation) and `weibull` has no elementary
/// closed form — both are excluded by the support gate and return `0` here as a
/// defensive no-op.
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
            InputRateKind::ZeroOrder | InputRateKind::Weibull => T::from_f64(0.0),
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
            InputRateKind::ZeroOrder | InputRateKind::Weibull => T::from_f64(0.0),
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
}
