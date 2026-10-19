//! The phenomena the retired forward provider's tests guarded (#2093 review, finding 6), each
//! re-anchored on the adjoint: modeled `D{n}`/`R{n}`, steady state into a built-in absorption
//! kernel and on the non-linear walk, `F` on the ODE walk, the time-varying-covariate × lag
//! snapshot, `TIME` in `[individual_parameters]`, `CMT=0` doses, the non-finite exits, IOV
//! under lag / SS / reset, the Weibull and inverse-Gaussian input-rate forcings, per-route
//! lags, and the covariance R-matrix on a lagged, an M3 and an IOV × Laplace subject. Plus the
//! routing predicates that send a subject outside that scope to finite differences.
//!
//! Every fixture is live on both sides of the phenomenon (a later dose lands on residual
//! drug; every η column and the θ under test move the predictions, asserted), and every
//! check is against an oracle that shares no derivative code with the adjoint: the
//! reconverged-FD objective gradient ([`fd_oracle`]) and a Richardson difference of the
//! production predictor for `∂f/∂η`. Each bound is the measured worst with ~10× headroom.

use super::tests::{
    censor, ebe_for, fd_oracle, hessian_gaps, hessian_gaps_for, laplace_prod, moved_params,
    oracle_jacobian, parity, subject, subject_shell, tight, with_tol,
};
use super::*;
use crate::estimation::parameterization::pack_params;
use crate::parser::model_parser::parse_model_string;
use crate::types::{DoseEvent, RateMode, Subject};
use std::collections::HashMap;

// ---- shared measurement ---------------------------------------------------------------

/// Whether the adjoint serves `s` at the model defaults, at the EBE the fit would use (the
/// stacked `[η, κ]` under IOV, which `tests::in_scope`'s BSV-only `find_ebe` is not).
fn served(m: &CompiledModel, s: &Subject) -> bool {
    let p = m.default_params.clone();
    let x = pack_params(&p);
    let eta = ebe_for(m, s, &p, 1e-11, true);
    subject_adjoint_packed_gradient(m, s, &p, &x, eta.as_slice(), true).is_some()
}

/// A probe η off the mode, alternating in sign, so no column sits at a symmetric point.
fn probe_eta(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| if k % 2 == 0 { 0.06 } else { -0.04 } * (1.0 + 0.3 * k as f64))
        .collect()
}

/// `max_j |∂f_j/∂θ_k|` by central FD of the production predictor: the θ under test must
/// actually move a prediction, or its gradient row is vacuously right.
fn theta_effect(m: &CompiledModel, s: &Subject, eta: &[f64], k: usize) -> f64 {
    let theta = &m.default_params.theta;
    let h = 1e-4 * (1.0 + theta[k].abs());
    let at = |d: f64| {
        let mut t = theta.clone();
        t[k] += d;
        crate::pk::compute_predictions_with_tv(m, s, &t, eta)
    };
    let (p, q) = (at(h), at(-h));
    let mut worst = 0.0f64;
    for (a, b) in p.iter().zip(&q) {
        let d = (a - b) / (2.0 * h);
        assert!(d.is_finite(), "θ[{k}]: non-finite prediction difference");
        worst = worst.max(d.abs());
    }
    worst
}

/// Worst `|exact − FD| / (1 + |FD|)` between the recorded `∂f/∂η` and a Richardson
/// difference of the production predictor, asserting every η column is live.
fn jacobian_gap(m: &CompiledModel, s: &Subject, eta: &[f64], label: &str) -> f64 {
    let theta = &m.default_params.theta;
    let jac = exact_eta_jacobian(m, s, theta, eta)
        .unwrap_or_else(|| panic!("{label}: the recorder declined the η-Jacobian"));
    let fd = oracle_jacobian(m, s, theta, eta);
    assert_eq!(jac.shape(), fd.shape(), "{label}: Jacobian shape");
    for k in 0..m.n_eta {
        let col = fd.column(k).iter().fold(0.0f64, |a, v| a.max(v.abs()));
        assert!(
            col > 1e-6,
            "{label}: η[{k}] moves no prediction ({col:.1e})"
        );
    }
    let mut worst = 0.0f64;
    for (a, b) in jac.iter().zip(fd.iter()) {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label}: non-finite {a} vs {b}"
        );
        worst = worst.max((a - b).abs() / (1.0 + b.abs()));
    }
    worst
}

/// The phenomenon check on a non-IOV subject: in the adjoint's scope, every θ in
/// `live_theta` moves a prediction, and `(η-Jacobian gap, FOCEI gradient gap)`.
fn measure(m: &CompiledModel, s: &Subject, live_theta: &[usize], label: &str) -> (f64, f64) {
    assert!(served(m, s), "{label}: the adjoint must serve this subject");
    let eta = probe_eta(m.n_eta);
    for &k in live_theta {
        let e = theta_effect(m, s, &eta, k);
        assert!(e > 1e-6, "{label}: θ[{k}] moves no prediction ({e:.1e})");
    }
    let jac = jacobian_gap(m, s, &eta, label);
    let grad = parity(m, s);
    eprintln!("{label}: ∂f/∂η vs FD {jac:.2e}; FOCEI gradient vs reconverged FD {grad:.2e}");
    (jac, grad)
}

// ---- modeled D{n} / R{n} -------------------------------------------------------------

const MODELED_DUR_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP ~ 0.1 (sd)
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
"#;

const MODELED_RATE_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVR1(20.0, 0.1, 200.0)
  omega ETA_CL ~ 0.09
  omega ETA_R1 ~ 0.04
  sigma PROP ~ 0.1 (sd)
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
"#;

const MODELED_DUR_LAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  theta TVLAG(1.0, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_D1 ~ 0.04
  omega ETA_LAG ~ 0.02
  sigma PROP ~ 0.1 (sd)
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
"#;

/// Two modeled doses, the second landing on residual drug from the first, observations on
/// both sides of each modeled window end.
fn modeled_subject(m: &CompiledModel, mode: RateMode, times: &[f64]) -> Subject {
    let doses = vec![
        DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, mode),
        DoseEvent::modeled(8.0, 100.0, 1, false, 0.0, mode),
    ];
    subject(m, doses, times, &[0.1, -0.08, 0.05][..m.n_eta])
}

/// #530 / #369: a modeled duration `D1` (η on it) is a moving rate-off boundary on the
/// event walk; the second dose's window opens on residual drug.
#[test]
fn modeled_duration_matches_reconverged_fd() {
    let m = tight(parse_model_string(MODELED_DUR_ODE).unwrap());
    let s = modeled_subject(
        &m,
        RateMode::ModeledDuration,
        &[1.0, 3.0, 6.0, 7.5, 9.0, 12.0, 14.0, 20.0],
    );
    assert!(!s.all_doses_fixed(), "the doses must be modeled");
    let (jac, grad) = measure(&m, &s, &[0, 2], "modeled D1");
    // Measured jac 3.1e-12, grad 2.7e-8 on Linux; ~10× headroom.
    assert!(jac < 3e-11, "modeled D1 ∂f/∂η: {jac:.2e}");
    assert!(grad < 3e-7, "modeled D1 gradient: {grad:.2e}");
}

/// #530 mirror: a modeled rate `R1` sets the window `amt / R1`, again a moving boundary.
#[test]
fn modeled_rate_matches_reconverged_fd() {
    let m = tight(parse_model_string(MODELED_RATE_ODE).unwrap());
    let s = modeled_subject(
        &m,
        RateMode::ModeledRate,
        &[1.0, 3.0, 6.0, 7.5, 9.0, 12.0, 14.0, 20.0],
    );
    let (jac, grad) = measure(&m, &s, &[0, 2], "modeled R1");
    // Measured jac 3.0e-12, grad 1.2e-6 on Linux; ~10× headroom.
    assert!(jac < 3e-11, "modeled R1 ∂f/∂η: {jac:.2e}");
    assert!(grad < 1.2e-5, "modeled R1 gradient: {grad:.2e}");
}

/// #530 cross-term: a modeled duration under an estimated lag — the rate-on boundary moves
/// with the lag, the rate-off with lag + D1.
#[test]
fn modeled_duration_under_a_lag_matches_reconverged_fd() {
    let m = tight(parse_model_string(MODELED_DUR_LAG_ODE).unwrap());
    let s = modeled_subject(
        &m,
        RateMode::ModeledDuration,
        &[0.5, 2.0, 4.0, 7.0, 8.5, 10.0, 13.0, 15.0, 20.0],
    );
    let (jac, grad) = measure(&m, &s, &[0, 2, 3], "modeled D1 + lag");
    // Measured jac 2.5e-12, grad 8.8e-8 on Linux; ~10× headroom.
    assert!(jac < 3e-11, "modeled D1 + lag ∂f/∂η: {jac:.2e}");
    assert!(grad < 9e-7, "modeled D1 + lag gradient: {grad:.2e}");
}

// ---- steady state into a built-in absorption kernel; SS on the non-linear walk -------

const SS_FIRST_ORDER_F: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  theta THETA_F(0.7, 0.001, 0.999)
  omega ETA_CL ~ 0.09
  omega ETA_F ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = inv_logit(logit(THETA_F) + ETA_F)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const SS_TWOCPT_FIRST_ORDER: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV1(20.0, 0.5, 200.0)
  theta TVQ(2.0, 0.01, 100.0)
  theta TVV2(30.0, 1.0, 500.0)
  theta TVKA(0.2, 0.005, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA
[structural_model]
  ode(obs_cmt=central, states=[central, peripheral])
[odes]
  d/dt(central)    = first_order(ka=KA) - (CL/V1)*central - (Q/V1)*central + (Q/V2)*peripheral
  d/dt(peripheral) =  (Q/V1)*central - (Q/V2)*peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const SS_TRANSIT: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVN(5.0, 1.0, 30.0)
  theta TVMTT(2.0, 0.1, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_MTT ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  N   = TVN
  MTT = TVMTT * exp(ETA_MTT)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = transit(n=N, mtt=MTT) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const SS_IGD: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVMAT(2.0, 0.1, 20.0)
  theta TVCV2(0.5, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_MAT ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  MAT = TVMAT * exp(ETA_MAT)
  CV2 = TVCV2
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const SS_WEIBULL: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVTD(2.0, 0.05, 24.0)
  theta TVBETA(1.5, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_BETA ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  TD   = TVTD
  BETA = TVBETA * exp(ETA_BETA)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// The SS-absorption fixtures: a steady-state dose into each kernel the dual SS
/// equilibration admits (F estimated on the first), then an ordinary dose landing on that
/// trough, with `(model, θ that must move a prediction, label)`.
fn ss_absorption_cases() -> Vec<(CompiledModel, Subject, &'static [usize], &'static str)> {
    let times = [1.0, 3.0, 6.0, 9.0, 11.5, 13.0, 16.0, 20.0, 23.0];
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    [
        (SS_FIRST_ORDER_F, &[0usize, 2, 3][..], "SS first_order + F"),
        (
            SS_TWOCPT_FIRST_ORDER,
            &[0, 2, 4][..],
            "SS first_order 2-cpt",
        ),
        (SS_TRANSIT, &[0, 3][..], "SS transit"),
        (SS_IGD, &[0, 2, 3][..], "SS igd"),
        (SS_WEIBULL, &[0, 2, 3][..], "SS weibull"),
    ]
    .into_iter()
    .map(|(src, live, label)| {
        // At reltol 1e-10, not the 1e-12 the forcing tests below use: the SS equilibration
        // is the cost here. The ∂f/∂η gap is then the integrator's floor (it scales with the
        // tolerance: see `weibull_and_inverse_gaussian_forcing_match_reconverged_fd`).
        let m = tight(parse_model_string(src).unwrap());
        assert!(!m.ode_spec.as_ref().unwrap().input_rate.is_empty());
        let s = subject(&m, doses.clone(), &times, &[0.1, -0.05]);
        assert!(s.has_periodic_ss_dose() && !m.ss_absorption_out_of_scope(&s));
        (m, s, live, label)
    })
    .collect()
}

/// #835 / #867 / #913: steady state into a built-in absorption kernel (the SS trough
/// carries the kernel's carry-over), then an ordinary dose landing on that trough. Every
/// admitted kernel's `∂f/∂η` against FD of the predictor; the first-order kernels' gradient
/// against reconverged FD here, the transit / igd / Weibull gradients in the slow twin below
/// (≈ 40 s in debug together).
#[test]
fn steady_state_into_built_in_absorption_matches_reconverged_fd() {
    for (m, s, live, label) in ss_absorption_cases() {
        assert!(
            served(&m, &s),
            "{label}: the adjoint must serve this subject"
        );
        let eta = probe_eta(m.n_eta);
        for &k in live {
            let e = theta_effect(&m, &s, &eta, k);
            assert!(e > 1e-6, "{label}: θ[{k}] moves no prediction ({e:.1e})");
        }
        let jac = jacobian_gap(&m, &s, &eta, label);
        eprintln!("{label}: ∂f/∂η vs FD {jac:.2e}");
        // Measured worst 3.8e-7 (Weibull; 8.3e-9 / 7.9e-8 / 3.2e-9 / 9.3e-9 for first_order
        // + F / 2-cpt / transit / igd) on Linux; ~10× headroom.
        assert!(jac < 4e-6, "{label} ∂f/∂η: {jac:.2e}");
        if label.contains("first_order") {
            let grad = parity(&m, &s);
            eprintln!("{label}: FOCEI gradient vs reconverged FD {grad:.2e}");
            // Measured 3.3e-7 (+ F) / 1.3e-8 (2-cpt) on Linux; ~10× headroom.
            assert!(grad < 3.4e-6, "{label} gradient: {grad:.2e}");
        }
    }
}

/// The slow twin: the transit / igd / Weibull SS gradients against reconverged FD.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn steady_state_into_transit_igd_weibull_gradient_matches_reconverged_fd() {
    for (m, s, _, label) in ss_absorption_cases() {
        if label.contains("first_order") {
            continue;
        }
        let grad = parity(&m, &s);
        eprintln!("{label}: FOCEI gradient vs reconverged FD {grad:.2e}");
        // Measured 2.9e-7 / 1.3e-6 / 1.2e-6 (transit / igd / Weibull) on Linux; ~10×.
        assert!(grad < 1.3e-5, "{label} gradient: {grad:.2e}");
    }
}

const MM_DISPOSITION_ODE: &str = r#"
[parameters]
  theta TVVMAX(50.0, 1.0, 500.0)
  theta TVKM(30.0, 1.0, 500.0)
  omega ETA_VMAX ~ 0.09
  omega ETA_KM   ~ 0.04
  sigma PROP ~ 0.1 (sd)
[individual_parameters]
  VMAX = TVVMAX * exp(ETA_VMAX)
  KM   = TVKM   * exp(ETA_KM)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -VMAX * central / (KM + central)
[error_model]
  DV ~ proportional(PROP)
"#;

/// A steady-state **infusion** on a Michaelis–Menten disposition: the closed-form SS fixed
/// point self-declines, so the dual walk runs the pulse-train iteration with the infusion's
/// rate boundaries in every cycle; a later bolus lands on the SS trough.
#[test]
fn steady_state_infusion_on_the_nonlinear_walk_matches_reconverged_fd() {
    let m = tight(parse_model_string(MM_DISPOSITION_ODE).unwrap());
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 40.0, true, 8.0),
        DoseEvent::new(8.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let s = subject(
        &m,
        doses,
        &[1.0, 2.0, 3.0, 6.0, 7.9, 9.0, 12.0, 18.0],
        &[0.1, 0.05],
    );
    assert!(s.doses[0].ss && s.doses[0].is_infusion());
    let (jac, grad) = measure(&m, &s, &[0, 1], "SS infusion, MM");
    // Measured jac 6.9e-10, grad 1.4e-6 on Linux; ~10× headroom.
    assert!(jac < 7e-9, "SS infusion MM ∂f/∂η: {jac:.2e}");
    assert!(grad < 1.4e-5, "SS infusion MM gradient: {grad:.2e}");
}

// ---- F on the ODE walk ---------------------------------------------------------------

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
"#;

const IV_F_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVF(0.7, 0.05, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_F ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
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
"#;

/// #369 / #419: `F` (θ and η) on the ODE walks — scaling the depot bolus on the static walk,
/// and reshaping a rate-defined infusion's window (`F·amt/rate`, a moving boundary) on the
/// event walk. Two doses each, the second on residual drug.
#[test]
fn bioavailability_on_the_ode_walks_matches_reconverged_fd() {
    let m = tight(parse_model_string(BIOAV_ODE).unwrap());
    assert!(m.has_bioavailability());
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let s = subject(
        &m,
        doses,
        &[0.5, 1.0, 2.0, 4.0, 8.0, 12.5, 14.0, 18.0, 24.0],
        &[0.1, -0.05],
    );
    assert!(crate::sens::ode_provider::static_walk_subject_supported(
        &m, &s
    ));
    let (jac, grad) = measure(&m, &s, &[0, 3], "F, static walk");
    // Measured jac 2.8e-12, grad 4.2e-9 on Linux; ~10× headroom.
    assert!(jac < 3e-11, "F static ∂f/∂η: {jac:.2e}");
    assert!(grad < 4e-8, "F static gradient: {grad:.2e}");

    let m = tight(parse_model_string(IV_F_ODE).unwrap());
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 40.0, false, 0.0),
        DoseEvent::new(6.0, 100.0, 1, 40.0, false, 0.0),
    ];
    let s = subject(
        &m,
        doses,
        &[1.0, 2.0, 4.0, 6.5, 7.0, 8.0, 10.0, 14.0],
        &[0.1, 0.05],
    );
    assert!(s.has_rate_defined_infusion());
    assert!(!crate::sens::ode_provider::static_walk_subject_supported(
        &m, &s
    ));
    let (jac, grad) = measure(&m, &s, &[0, 2], "F, rate-defined infusion");
    // Measured jac 3.7e-12, grad 2.7e-8 on Linux; ~10× headroom.
    assert!(jac < 4e-11, "F infusion ∂f/∂η: {jac:.2e}");
    assert!(grad < 3e-7, "F infusion gradient: {grad:.2e}");
}

// ---- the time-varying-covariate × lag snapshot ---------------------------------------

const IV_LAG_CROSS_ODE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVP(0.7, 0.05, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_P  ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  LAGTIME = TVP * exp(ETA_P)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[covariates]
  WT continuous
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// #1060 / #1068–#1076: a lagged arrival that crosses a covariate change, so the
/// post-arrival field is the arrival segment's snapshot, not the dose row's — with a prior
/// dose, so the compartment is **not** empty at the arrival and a wrong snapshot is not
/// multiplied by zero (#1068).
#[test]
fn tvcov_lag_arrival_across_a_covariate_change_matches_reconverged_fd() {
    let m = tight(parse_model_string(IV_LAG_CROSS_ODE).unwrap());
    // Every observation after the first arrival (≈ 0.64): a pre-arrival row predicts 0, where
    // proportional error sits on its variance floor and the objective has a kink.
    let times = [0.8, 1.5, 2.0, 4.0, 8.0];
    let wts = [71.0, 74.0, 76.0, 80.0, 85.0];
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(1.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut s = subject_shell(doses, &times);
    s.covariates.insert("WT".into(), 70.0);
    s.dose_covariates = vec![wt(70.0), wt(70.0)];
    s.obs_covariates = wts.iter().map(|&w| wt(w)).collect();
    let preds = crate::pk::compute_predictions_with_tv(
        &m,
        &s,
        &m.default_params.theta,
        &[0.1, -0.05, 0.07],
    );
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
        .collect();
    assert!(s.has_tv_covariates() && m.has_lagtime());
    assert!(
        preds.iter().all(|&v| v > 0.0),
        "every row after the first arrival"
    );
    // The straddle: at the moved θ and its EBE, the second dose arrives strictly inside the
    // [1.5, 2.0] segment, whose snapshot (WT = 76) is not the dose row's (WT = 70).
    let p = moved_params(&m);
    let e = ebe_for(&m, &s, &p, 1e-12, true);
    let arrival = 1.0 + p.theta[3] * e[2].exp();
    assert!(
        arrival > 1.5 + 1e-2 && arrival < 2.0 - 1e-2,
        "arrival {arrival} must sit inside the [1.5, 2.0] segment"
    );
    let (jac, grad) = measure(&m, &s, &[0, 2, 3], "TV-cov × lag crossing");
    // Measured jac 2.3e-12, grad 4.1e-9 on Linux; ~10× headroom. (With the two pre-arrival
    // rows of the original #1060 subject, whose prediction is 0, the gradient measured
    // 2.4e-5, all of it on σ: the proportional variance floor, not the snapshot.)
    assert!(jac < 3e-11, "TV-cov × lag ∂f/∂η: {jac:.2e}");
    assert!(grad < 4e-8, "TV-cov × lag gradient: {grad:.2e}");
}

// ---- TIME in [individual_parameters] -------------------------------------------------

const ODE_TIME_PARAM: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 100.0)
  theta TVCL_LATE(0.5, 0.1, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.1 (sd)
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

/// #486 / #637: `TIME` read in `[individual_parameters]` — the event walk seeds each event's
/// parameters at that event's time, so `TVCL` acts only before t = 5 and `TVCL_LATE` only
/// after it (both θ rows live, asserted), with the second dose after the switch.
#[test]
fn time_in_individual_parameters_matches_reconverged_fd() {
    let m = tight(parse_model_string(ODE_TIME_PARAM).unwrap());
    assert!(crate::parser::model_parser::compiled_model_uses_time_builtin(&m));
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(8.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let s = subject(
        &m,
        doses,
        &[1.0, 3.0, 4.5, 6.0, 9.0, 12.0, 16.0],
        &[0.1, -0.05],
    );
    let (jac, grad) = measure(&m, &s, &[0, 1, 2], "TIME parameter");
    // Measured jac 1.8e-11, grad 2.7e-7 on Linux; ~10× headroom.
    assert!(jac < 2e-10, "TIME ∂f/∂η: {jac:.2e}");
    assert!(grad < 3e-6, "TIME gradient: {grad:.2e}");
}

// ---- CMT=0 doses ---------------------------------------------------------------------

const ONE_CPT_ORAL_ODE: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// #899: `CMT=0` is the default dose compartment (compartment 1) on every walk; the
/// adjoint gradient of a `CMT=0` subject must be **bit-identical** to its `CMT=1` twin and
/// match reconverged FD — on the static walk (bolus) and through the SS equilibration.
#[test]
fn cmt_zero_doses_are_the_default_compartment_and_match_reconverged_fd() {
    let m = tight(parse_model_string(ONE_CPT_ORAL_ODE).unwrap());
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.5, 14.0, 18.0, 23.0];
    for ss in [false, true] {
        let label = if ss { "CMT=0, SS" } else { "CMT=0, bolus" };
        let mk = |cmt: usize| {
            vec![
                DoseEvent::new(0.0, 100.0, cmt, 0.0, ss, if ss { 12.0 } else { 0.0 }),
                DoseEvent::new(12.0, 100.0, cmt, 0.0, false, 0.0),
            ]
        };
        let one = subject(&m, mk(1), &times, &[0.1, -0.05]);
        let mut zero = one.clone();
        zero.doses = mk(0);
        let (jac, grad) = measure(&m, &zero, &[0, 2], label);
        // Measured jac 7.0e-12 / 7.5e-12, grad 3.3e-7 / 2.5e-9 (bolus / SS) on Linux; ~10×.
        assert!(jac < 8e-11, "{label} ∂f/∂η: {jac:.2e}");
        assert!(grad < 3.3e-6, "{label} gradient: {grad:.2e}");
        let p = moved_params(&m);
        let x = pack_params(&p);
        let e = ebe_for(&m, &one, &p, 1e-12, true);
        let g1 = subject_adjoint_packed_gradient(&m, &one, &p, &x, e.as_slice(), true).unwrap();
        let g0 = subject_adjoint_packed_gradient(&m, &zero, &p, &x, e.as_slice(), true).unwrap();
        for (k, (a, b)) in g0.iter().zip(&g1).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{label}: x[{k}] {a} vs CMT=1 {b}");
        }
    }
}

const IV_LAG_INF_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
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
"#;

/// #1077 / #1196: a lagged `CMT=0` **infusion** delivers nothing on either engine (a
/// zero-order input has no default target), so it must inject no rate boundary into the
/// gradient either. Next to a live lagged bolus (so `η_LAG` is live), the subject's
/// recorded `∂f/∂η` must equal the FD of the predictor — which a spurious rate-off
/// saltation would move by O(1) — and its gradient must equal the gradient of the same
/// subject without the infusion.
#[test]
fn cmt_zero_lagged_infusion_injects_no_rate_boundary() {
    let m = tight(parse_model_string(IV_LAG_INF_ODE).unwrap());
    let times = [1.0, 2.0, 4.0, 6.5, 8.0, 10.0, 14.0];
    let bolus = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let without = subject(&m, vec![bolus.clone()], &times, &[0.1, 0.05]);
    let mut with = without.clone();
    // A 2 h infusion arriving at ≈ 6.5: observations on both sides of its window end.
    with.doses = vec![bolus, DoseEvent::new(6.0, 100.0, 0, 50.0, false, 0.0)];
    let th = &m.default_params.theta;
    let eta = probe_eta(m.n_eta);
    let (pa, pb) = (
        crate::pk::compute_predictions_with_tv(&m, &with, th, &eta),
        crate::pk::compute_predictions_with_tv(&m, &without, th, &eta),
    );
    for (a, b) in pa.iter().zip(&pb) {
        assert!(
            (a - b).abs() <= 1e-9 * b.abs(),
            "the CMT=0 infusion delivered: {a} vs {b}"
        );
    }
    let (jac, grad) = measure(&m, &with, &[0, 2], "CMT=0 lagged infusion");
    // Measured jac 5.8e-12, grad 2.5e-7 on Linux; ~10× headroom.
    assert!(jac < 6e-11, "CMT=0 infusion ∂f/∂η: {jac:.2e}");
    assert!(grad < 2.5e-6, "CMT=0 infusion gradient: {grad:.2e}");
    let p = moved_params(&m);
    let x = pack_params(&p);
    let e = ebe_for(&m, &without, &p, 1e-12, true);
    let ga = subject_adjoint_packed_gradient(&m, &with, &p, &x, e.as_slice(), true).unwrap();
    let gb = subject_adjoint_packed_gradient(&m, &without, &p, &x, e.as_slice(), true).unwrap();
    let mut worst = 0.0f64;
    for (a, b) in ga.iter().zip(&gb) {
        assert!(a.is_finite() && b.is_finite());
        worst = worst.max((a - b).abs() / (1.0 + b.abs()));
    }
    eprintln!("CMT=0 infusion vs the subject without it: {worst:.2e}");
    // Measured 5.3e-15 (round-off of the extra integration break) on Linux; ~10× headroom.
    assert!(
        worst < 5e-14,
        "the CMT=0 infusion moved the gradient: {worst:.2e}"
    );
}

// ---- the non-finite exits (#1234) ----------------------------------------------------

const BARE_STATE_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const BARE_STATE_LAG_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  theta TVLAG(0.3, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  LAGTIME = TVLAG * exp(WT)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// #1234: a subject whose timeline cannot be ordered — a `NaN` dose time on the static walk,
/// an `+inf` lag on the event walk — is diverged, and the walks return a `NaN` value with
/// finite-zero jets. The adjoint must not turn that into a finite gradient: it declines
/// (`None`, so the subject takes FD and is reported) or returns a non-finite one; never a
/// finite vector. Each walk is asserted to be the one the fixture reaches.
#[test]
fn a_diverged_timeline_never_yields_a_finite_adjoint_gradient() {
    let static_m = parse_model_string(BARE_STATE_ODE).unwrap();
    let mut s1 = subject_shell(
        vec![DoseEvent::new(f64::NAN, 100.0, 1, 0.0, false, 0.0)],
        &[1.0, 2.0, 4.0, 8.0],
    );
    s1.observations = vec![5.0, 4.0, 3.0, 2.0];
    assert!(crate::sens::ode_provider::static_walk_subject_supported(
        &static_m, &s1
    ));
    let lag_m = parse_model_string(BARE_STATE_LAG_ODE).unwrap();
    let times: Vec<f64> = (1..=8).map(f64::from).collect();
    let mut s2 = subject_shell(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(4.5, 100.0, 1, 0.0, false, 0.0),
        ],
        &times,
    );
    s2.covariates.insert("WT".into(), 1000.0);
    s2.observations = vec![5.0; 8];
    assert!(!crate::sens::ode_provider::static_walk_subject_supported(
        &lag_m, &s2
    ));
    assert!(crate::sens::ode_provider::event_walk_subject_supported(
        &lag_m, &s2
    ));
    for (m, s, label) in [(&static_m, &s1, "static walk"), (&lag_m, &s2, "event walk")] {
        let preds = crate::pk::compute_predictions_with_tv(m, s, &m.default_params.theta, &[0.0]);
        assert!(
            preds.iter().all(|v| !v.is_finite()),
            "{label}: the fixture must diverge"
        );
        let p = m.default_params.clone();
        let x = pack_params(&p);
        let g = subject_adjoint_packed_gradient(m, s, &p, &x, &[0.0], true);
        eprintln!("{label}: adjoint gradient {g:?}");
        if let Some(g) = &g {
            assert!(
                g.iter().any(|v| !v.is_finite()),
                "{label}: a diverged subject came back with a finite gradient {g:?}"
            );
        }
        if let Some(j) = exact_eta_jacobian(m, s, &p.theta, &[0.0]) {
            let vals: Vec<f64> = exact_obs_grads(m, s, &p.theta, &[0.0])
                .map(|r| r.iter().map(|o| o.f).collect())
                .unwrap_or_default();
            assert!(
                vals.iter().all(|v| !v.is_finite()),
                "{label}: the recorded value must stay non-finite, {vals:?} (J = {j})"
            );
        }
    }
}

// ---- IOV × lag / SS / reset ------------------------------------------------------------

const IOV_LAG_ODE: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(0.6, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.06
  kappa KAPPA_CL ~ 0.03
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V = TVV
  KA = TVKA
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
"#;

/// Three occasions, a dose opening each (the first `SS` when `ss`), an optional reset at
/// the start of the third, observations generated from `predict_iov` at known κ.
fn iov_lag_subject(m: &CompiledModel, ss: bool, reset: bool) -> Subject {
    let times = [1.0, 4.0, 10.0, 13.0, 16.0, 22.0, 25.0, 28.0, 34.0];
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, ss, if ss { 12.0 } else { 0.0 }),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let mut s = subject_shell(doses, &times);
    s.occasions = vec![1, 1, 1, 2, 2, 2, 3, 3, 3];
    s.dose_occasions = vec![1, 2, 3];
    if reset {
        s.reset_times = vec![23.5];
        s.reset_occasions = vec![3];
    }
    let kappas = vec![vec![0.1], vec![-0.08], vec![0.05]];
    let preds = crate::pk::predict_iov(m, &s, &m.default_params.theta, &[0.1, -0.05], &kappas);
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.12 * (1.7 * j as f64).sin()))
        .collect();
    s
}

/// Worst gap between the recorded stacked `∂f/∂[η, κ]` and a Richardson difference of
/// `predict_iov`, asserting every stacked column is live.
fn iov_jacobian_gap(m: &CompiledModel, s: &Subject, stacked: &[f64], label: &str) -> f64 {
    let theta = &m.default_params.theta;
    let jac = exact_eta_jacobian(m, s, theta, stacked).expect("stacked Jacobian");
    let n = m.n_eta;
    let pred = |st: &[f64]| {
        let kap: Vec<Vec<f64>> = st[n..].chunks(m.n_kappa).map(<[f64]>::to_vec).collect();
        crate::pk::predict_iov(m, s, theta, &st[..n], &kap)
    };
    let mut worst = 0.0f64;
    for k in 0..stacked.len() {
        let d = |h: f64| {
            let (mut p, mut q) = (stacked.to_vec(), stacked.to_vec());
            p[k] += h;
            q[k] -= h;
            let (a, b) = (pred(&p), pred(&q));
            a.iter()
                .zip(&b)
                .map(|(a, b)| (a - b) / (2.0 * h))
                .collect::<Vec<f64>>()
        };
        let (d1, d2) = (d(1e-4), d(2e-4));
        let mut col = 0.0f64;
        for j in 0..s.obs_times.len() {
            let fd = (4.0 * d1[j] - d2[j]) / 3.0;
            assert!(
                fd.is_finite() && jac[(j, k)].is_finite(),
                "{label}: non-finite"
            );
            col = col.max(fd.abs());
            worst = worst.max((jac[(j, k)] - fd).abs() / (1.0 + fd.abs()));
        }
        assert!(
            col > 1e-6,
            "{label}: stacked column {k} moves no prediction"
        );
    }
    worst
}

/// #598 / #931: IOV on the event walk under an estimated lag, with the first dose `SS`, and
/// with a reset opening the third occasion — κ_g of an earlier occasion reaching the later
/// ones through residual drug, except across the reset.
#[test]
fn iov_under_lag_ss_and_reset_matches_reconverged_fd() {
    let m = tight(parse_model_string(IOV_LAG_ODE).unwrap());
    for (ss, reset, label) in [
        (false, false, "IOV × lag"),
        (true, false, "IOV × lag × SS"),
        (false, true, "IOV × lag × reset"),
    ] {
        let s = iov_lag_subject(&m, ss, reset);
        assert!(
            served(&m, &s),
            "{label}: the adjoint must serve this subject"
        );
        let jac = iov_jacobian_gap(&m, &s, &[0.06, -0.04, 0.05, -0.03, 0.04], label);
        let grad = fd_oracle(&m, &s, true);
        eprintln!("{label}: ∂f/∂[η, κ] vs FD {jac:.2e}; FOCEI gradient vs FD {grad:.2e}");
        // Measured worst jac 7.2e-12, grad 4.5e-6 (the SS arm) over the three on Linux; ~10×.
        assert!(jac < 8e-11, "{label} ∂f/∂[η, κ]: {jac:.2e}");
        assert!(grad < 4.5e-5, "{label} gradient: {grad:.2e}");
    }
}

// ---- Weibull / inverse-Gaussian input-rate forcing --------------------------------------

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
"#;

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
"#;

/// The Weibull and inverse-Gaussian absorption kinds, which route through the input-rate
/// forcing rather than a dose bolus: two doses, the second's forcing overlapping the
/// first's tail.
#[test]
fn weibull_and_inverse_gaussian_forcing_match_reconverged_fd() {
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [0.5, 1.0, 2.0, 4.0, 6.5, 7.0, 9.0, 12.0, 18.0];
    for (src, live, label) in [
        (WEIBULL_ODE, &[0usize, 2, 3][..], "weibull forcing"),
        (IGD_ODE, &[0, 2, 3][..], "igd forcing"),
    ] {
        // At reltol 1e-12: the forcing's ∂f/∂η gap is the integrator's, not the recorder's
        // (measured on the Weibull fixture: 4.5e-6 / 5.9e-8 / 7.4e-10 at reltol 1e-8 / 1e-10 /
        // 1e-12), so the tighter solve is what lets the bound see a recorder defect.
        let m = with_tol(parse_model_string(src).unwrap(), 1e-12, 1e-14);
        let s = subject(&m, doses.clone(), &times, &[0.1, -0.05]);
        let (jac, grad) = measure(&m, &s, live, label);
        // Measured worst jac 7.4e-10 (Weibull), grad 9.5e-8 (igd) on Linux; ~10× headroom.
        assert!(jac < 8e-9, "{label} ∂f/∂η: {jac:.2e}");
        assert!(grad < 1e-6, "{label} gradient: {grad:.2e}");
    }
}

// ---- per-route lag ---------------------------------------------------------------------

const ROUTE_LAG_ODE: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  LAG = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA, lag=LAG) - CL/V*central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const TWO_ROUTE_LAG_ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVKA1(1.5, 0.05, 20.0)
  theta TVKA2(0.3, 0.05, 20.0)
  theta TVLAG2(2.0, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG2 ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  FR1  = TVFR1
  FR2  = 1 - TVFR1
  KA1  = TVKA1
  KA2  = TVKA2
  LAG2 = TVLAG2 * exp(ETA_LAG2)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2, lag=LAG2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// #859 / #880 / #883: a per-route lag (η on it) is a route-onset saltation on the event
/// walk — alone, and as the delayed route of a two-route model — with a second dose whose
/// onset lands on the first's residual drug. Observations bracket each onset.
#[test]
fn per_route_lag_matches_reconverged_fd() {
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [0.5, 1.0, 1.5, 2.5, 4.0, 6.5, 7.0, 8.5, 12.0, 18.0];
    for (src, live, label) in [
        (ROUTE_LAG_ODE, &[0usize, 2, 3][..], "route lag"),
        (
            TWO_ROUTE_LAG_ODE,
            &[0, 2, 3, 4, 5][..],
            "two routes, one lagged",
        ),
    ] {
        // At reltol 1e-12, for the reason given at the Weibull / igd forcing test.
        let m = with_tol(parse_model_string(src).unwrap(), 1e-12, 1e-14);
        assert!(m.has_route_absorption_lag());
        let s = subject(&m, doses.clone(), &times, &[0.1, 0.05]);
        let (jac, grad) = measure(&m, &s, live, label);
        // Measured worst jac 6.6e-10, grad 1.05e-6 (single route) on Linux; ~10× headroom.
        assert!(jac < 7e-9, "{label} ∂f/∂η: {jac:.2e}");
        assert!(grad < 1.1e-5, "{label} gradient: {grad:.2e}");
    }
}

// ---- the covariance R-matrix: lag, M3, IOV × Laplace ---------------------------------

const ODE_LAG: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.1
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  KA = TVKA
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const ONE_CPT_ORAL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
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

const CF_LAG: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVLAG(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.1
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  ALAG = TVLAG * exp(ETA_LAG)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=ALAG)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// Two doses, the second on residual drug, observations clear of both arrivals.
fn two_dose_subject(m: &CompiledModel) -> Subject {
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subject(
        m,
        doses,
        &[1.0, 2.0, 4.0, 8.0, 13.0, 14.0, 18.0, 24.0],
        &[0.1, -0.05],
    )
}

/// #520 / #1505: the exact R-matrix on a lagged subject, FOCEI and FOCE, against the
/// reconverged FD of the adjoint gradient — on the closed form (which evaluates the lag
/// exactly), and on the ODE twin with the lag removed: the control for the moving-boundary
/// defect pinned (ignored) by [`cov_hessian_on_an_ode_moving_boundary_matches_profile_fd`].
#[test]
fn cov_hessian_matches_profile_fd_on_a_lagged_closed_form_and_its_unlagged_ode_twin() {
    let unlagged = ODE_LAG
        .replace("TVLAG * exp(ETA_LAG)", "TVLAG")
        .replace("omega ETA_LAG ~ 0.1", "omega ETA_V ~ 0.1")
        .replace("V = TVV", "V = TVV * exp(ETA_V)")
        .replace("  LAGTIME = TVLAG\n", "");
    // Measured FOCEI / FOCE: closed form 2.5e-6 / 1.15e-5, unlagged ODE 5.1e-7 / 1.6e-8 on
    // Linux; ~10× headroom each.
    for (src, label, bounds) in [
        (CF_LAG, "closed-form lag", [2.5e-5, 1.2e-4]),
        (unlagged.as_str(), "ODE, no lag", [5e-6, 2e-7]),
    ] {
        let m = tight(parse_model_string(src).unwrap());
        assert_eq!(m.has_lagtime(), label == "closed-form lag");
        let s = two_dose_subject(&m);
        assert!(served(&m, &s));
        for (interaction, name, bound) in [(true, "FOCEI", bounds[0]), (false, "FOCE", bounds[1])] {
            let (fd, _) = hessian_gaps(&m, &s, interaction);
            eprintln!("{label} R-matrix {name}: {fd:.2e}");
            assert!(fd < bound, "{label} {name} vs FD {fd:.2e}");
        }
    }
}

/// Third-order derivatives across a *moving* `[odes]` event boundary (#2093 finding 6): an
/// estimated depot or central `LAGTIME`, a per-route `lag=`, a modeled `D1`.
///
/// The event walk used to cross such a boundary with a δ²-truncated saltation (an `f64` δ²
/// coefficient and no δ³ term), exact through second order only. Measured before the
/// moving-boundary flows (`cross_moving_boundary`): the FOCEI R-matrix (`∂²(∂f/∂η)/∂θ²`)
/// was off FD of the exact gradient by 3.2e-1 (depot `LAGTIME`), 1.7e-1 (route lag), 1.6e-3
/// (central `LAGTIME`) and 2.4e-3 (`D1`); Laplace's exact-anchor gradient off reconverged FD
/// by 6.6e-3 (`ODE_LAG`) and 1.5e-2 (route lag). The unlagged control measures 5.1e-7.
///
/// The route lag is checked against its **closed-form twin** instead (the route lag is the
/// depot dose's `lagtime`; the closed form evaluates the shifted time exactly): the FD of the
/// gradient is noise-limited on that fixture — 3.9e-5 at the default step, 3.3e-4 at a tenth
/// of it, 1.6e-5 at four times it, while the twin agrees to 9.5e-8.
#[test]
fn third_order_derivatives_across_an_ode_moving_boundary_are_exact() {
    let mut cases: Vec<(CompiledModel, Subject, &str)> = Vec::new();
    for (src, label) in [
        (ODE_LAG, "depot LAGTIME"),
        (IV_LAG_INF_ODE, "central LAGTIME"),
    ] {
        let m = tight(parse_model_string(src).unwrap());
        let s = two_dose_subject(&m);
        cases.push((m, s, label));
    }
    let m = tight(parse_model_string(MODELED_DUR_ODE).unwrap());
    let s = modeled_subject(
        &m,
        RateMode::ModeledDuration,
        &[1.0, 3.0, 6.0, 7.5, 9.0, 12.0, 14.0, 20.0],
    );
    cases.push((m, s, "modeled D1"));
    for (m, s, label) in &cases {
        let (fd, _) = hessian_gaps(m, s, true);
        eprintln!("{label}: R-matrix FOCEI vs FD {fd:.2e}");
        assert!(
            fd.is_finite() && fd < 1e-5,
            "{label}: R-matrix vs FD {fd:.2e}"
        );
    }
    let m = tight(parse_model_string(ODE_LAG).unwrap());
    let g = super::tests::fd_oracle_agq(&m, &two_dose_subject(&m), 1, true);
    eprintln!("depot LAGTIME: Laplace gradient vs FD {g:.2e}");
    assert!(g.is_finite() && g < 1e-5, "Laplace gradient vs FD {g:.2e}");

    // Route lag vs the closed-form twin, R-matrix and Laplace gradient.
    let scaled =
        ROUTE_LAG_ODE.replace("[error_model]", "[scaling]\n  obs_scale = V\n[error_model]");
    let ode = tight(parse_model_string(&scaled).unwrap());
    let cf = parse_model_string(ROUTE_LAG_CF).unwrap();
    let (s_ode, s_cf) = (two_dose_subject(&ode), two_dose_subject(&cf));
    let params = moved_params(&ode);
    let x = pack_params(&params);
    let e_ode = ebe_for(&ode, &s_ode, &params, 1e-12, true);
    let e_cf = ebe_for(&cf, &s_cf, &params, 1e-12, true);
    let rel = |a: &[f64], b: &[f64]| {
        let mut worst = 0.0f64;
        for (u, v) in a.iter().zip(b) {
            assert!(u.is_finite() && v.is_finite(), "{u} vs {v}");
            worst = worst.max((u - v).abs() / (1.0 + v.abs()));
        }
        worst
    };
    let hess = |m: &CompiledModel, s: &Subject, e: &[f64]| {
        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(crate::FIT_RAYON_STACK_SIZE)
                .spawn_scoped(scope, || {
                    super::subject_adjoint_cov_hessian(m, s, &params, &x, e, Objective::Focei)
                })
                .expect("spawn")
                .join()
                .expect("join")
        })
        .expect("served")
    };
    let r = rel(
        hess(&ode, &s_ode, e_ode.as_slice()).as_slice(),
        hess(&cf, &s_cf, e_cf.as_slice()).as_slice(),
    );
    eprintln!("route lag: R-matrix vs closed-form twin {r:.2e}");
    assert!(r < 1e-6, "route lag R-matrix vs closed-form twin {r:.2e}");
    let laplace = Objective::Agq {
        n_agq: 1,
        exact: true,
    };
    let lap = |m: &CompiledModel, s: &Subject, e: &[f64]| {
        subject_adjoint_gradient(m, s, &params, &x, e, laplace).expect("served")
    };
    let l = rel(
        &lap(&ode, &s_ode, e_ode.as_slice()),
        &lap(&cf, &s_cf, e_cf.as_slice()),
    );
    eprintln!("route lag: Laplace gradient vs closed-form twin {l:.2e}");
    assert!(
        l < 1e-6,
        "route lag Laplace gradient vs closed-form twin {l:.2e}"
    );
}

/// The FOCEI R-matrix on M3-censored rows (the inverse Mills ratio's second derivative),
/// one censored row per dose interval.
#[test]
fn cov_hessian_matches_profile_fd_on_m3_rows() {
    let mut m = parse_model_string(ONE_CPT_ORAL).unwrap();
    m.bloq_method = crate::types::BloqMethod::M3;
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 11.5, 14.0, 24.0];
    let s = censor(subject(&m, doses, &times, &[0.1, -0.05, 0.2]), &[0, 6, 8]);
    for (interaction, name) in [(true, "FOCEI"), (false, "FOCE")] {
        let (fd, _) = hessian_gaps(&m, &s, interaction);
        eprintln!("M3 R-matrix {name}: {fd:.2e}");
        // Measured 1.0e-6 (FOCEI) / 7.0e-8 (FOCE) on Linux; ~10× headroom on the worse.
        assert!(fd < 1e-5, "M3 {name} vs FD {fd:.2e}");
    }
}

const ODE_IOV_ONE_KAPPA: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  kappa KAPPA_CL ~ 0.03
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
"#;

/// Worst gap between the adjoint's IOV AGQ gradient and a Richardson difference of
/// production's population AGQ objective (`agq_population_nll`, which stacks `[η, κ]` per
/// occasion) with the stacked EBE re-solved at every point. `tests::fd_oracle_agq` cannot
/// serve IOV: its production objective is the non-IOV `agq_subject_objective_anchored`.
fn fd_oracle_agq_iov(m: &CompiledModel, s: &Subject, n_agq: usize, exact: bool) -> f64 {
    use crate::types::{HessianAnchor, Population};
    let pop = Population {
        subjects: vec![s.clone()],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let anchor = if exact {
        HessianAnchor::Exact
    } else {
        HessianAnchor::GaussNewton
    };
    let n = m.n_eta;
    let split = |e: &nalgebra::DVector<f64>| {
        let eta = nalgebra::DVector::from_column_slice(&e.as_slice()[..n]);
        let kap: Vec<nalgebra::DVector<f64>> = e.as_slice()[n..]
            .chunks(m.n_kappa)
            .map(nalgebra::DVector::from_column_slice)
            .collect();
        (eta, kap)
    };
    let params = moved_params(m);
    let x = pack_params(&params);
    let eta = ebe_for(m, s, &params, 1e-12, true);
    let adj = subject_adjoint_gradient(
        m,
        s,
        &params,
        &x,
        eta.as_slice(),
        Objective::Agq { n_agq, exact },
    )
    .expect("IOV AGQ subject must be in adjoint scope");
    let f_at = |xv: &[f64]| {
        let p = crate::estimation::parameterization::unpack_params(xv, &params);
        let (e, k) = split(&ebe_for(m, s, &p, 1e-12, true));
        crate::estimation::agq::agq_population_nll(m, &pop, &p, &[e], &[k], n_agq, anchor)
    };
    let mut worst = 0.0f64;
    for k in 0..x.len() {
        let fd = |h: f64| {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[k] += h;
            xm[k] -= h;
            (f_at(&xp) - f_at(&xm)) / (2.0 * h)
        };
        let rich = (4.0 * fd(1e-3) - fd(2e-3)) / 3.0;
        assert!(
            rich.is_finite() && adj[k].is_finite(),
            "x[{k}] adj={} fd={rich}",
            adj[k]
        );
        let rel = (adj[k] - rich).abs() / (1.0 + rich.abs());
        eprintln!(
            "x[{k}] adjoint={:.10e} reconverged-FD={rich:.10e} rel={rel:.2e}",
            adj[k]
        );
        worst = worst.max(rel);
    }
    assert!(
        adj.iter().fold(0.0f64, |a, v| a.max(v.abs())) > 1e-2,
        "gradient too small"
    );
    worst
}

/// The R-matrix of an IOV subject under Laplace's exact anchor and AGQ's Gauss–Newton one
/// (two occasions, stacked width 4: the exact anchor's first bucket), against the
/// reconverged FD of the adjoint gradient — and that gradient against reconverged FD of
/// production's AGQ objective, so the Hessian's reference is itself anchored.
#[test]
fn cov_hessian_matches_profile_fd_iov_under_laplace_and_agq() {
    let m = tight(parse_model_string(ODE_IOV_ONE_KAPPA).unwrap());
    let times = [1.0, 4.0, 10.0, 13.0, 16.0, 22.0];
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let mut s = subject_shell(doses, &times);
    s.occasions = vec![1, 1, 1, 2, 2, 2];
    s.dose_occasions = vec![1, 2];
    let preds = crate::pk::predict_iov(
        &m,
        &s,
        &m.default_params.theta,
        &[0.1, -0.05],
        &[vec![0.1], vec![-0.08]],
    );
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.12 * (1.7 * j as f64).sin()))
        .collect();
    assert_eq!(subject_width(&m, &s), Some(4));
    for (n_agq, exact, name) in [(1, true, "IOV Laplace"), (2, false, "IOV AGQ-2 H̃")] {
        let g = fd_oracle_agq_iov(&m, &s, n_agq, exact);
        let (fd, _) = hessian_gaps_for(&m, &s, Objective::Agq { n_agq, exact });
        eprintln!("{name}: gradient vs FD {g:.2e}; R-matrix vs FD {fd:.2e}");
        // Measured gradient 8.3e-7 / 3.7e-7, R-matrix 7.0e-7 / 3.6e-7 (Laplace / AGQ-2) on
        // Linux; ~10× headroom.
        assert!(g < 8e-6, "{name} gradient vs FD {g:.2e}");
        assert!(fd < 7e-6, "{name} R-matrix vs FD {fd:.2e}");
    }
}

// ---- routing of the kept scope predicates ---------------------------------------------

/// `has_infusion_into_input_rate` (#719 gap 2): an infusion into a built-in absorption
/// compartment double-counts on the dual walk, so the subject declines (FD), on both ODE
/// walks; the same subject dosed by bolus is served. Its predictions are live, so the
/// decline is not a vacuous one.
#[test]
fn an_infusion_into_an_absorption_kernel_declines_to_fd() {
    let m = parse_model_string(SS_FIRST_ORDER_F).unwrap();
    let times = [1.0, 3.0, 6.0, 9.0, 12.0];
    let bolus = subject(
        &m,
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &times,
        &[0.0, 0.0],
    );
    let mut inf = bolus.clone();
    inf.doses = vec![DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0)];
    assert!(!crate::sens::ode_provider::has_infusion_into_input_rate(
        &m, &bolus
    ));
    assert!(crate::sens::ode_provider::has_infusion_into_input_rate(
        &m, &inf
    ));
    assert!(
        served(&m, &bolus),
        "the bolus twin must be served (straddle)"
    );
    let preds =
        crate::pk::compute_predictions_with_tv(&m, &inf, &m.default_params.theta, &[0.0, 0.0]);
    assert!(preds.iter().all(|v| v.is_finite()) && preds.iter().any(|v| *v > 0.0));
    let p = m.default_params.clone();
    let x = pack_params(&p);
    assert!(subject_adjoint_packed_gradient(&m, &inf, &p, &x, &[0.0, 0.0], true).is_none());
    assert!(exact_eta_jacobian(&m, &inf, &p.theta, &[0.0, 0.0]).is_none());
}

const SS_ZERO_ORDER: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVDUR(2.0, 0.1, 20.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// `ss_absorption_out_of_scope` (#835 / #913): an SS dose into a zero-order or lagged
/// absorption compartment has no dual SS seed, so it declines — `CMT=0` scoped like
/// `CMT=1` — while the same dose without `SS` is served (straddle).
#[test]
fn steady_state_into_zero_order_or_route_lag_declines_to_fd() {
    let times = [1.0, 3.0, 6.0, 9.0, 11.0];
    for (src, label) in [(SS_ZERO_ORDER, "zero_order"), (ROUTE_LAG_ODE, "route lag")] {
        let m = parse_model_string(src).unwrap();
        let eta = vec![0.0; m.n_eta];
        let plain = subject(
            &m,
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            &times,
            &eta,
        );
        assert!(!m.ss_absorption_out_of_scope(&plain));
        assert!(
            served(&m, &plain),
            "{label}: the non-SS twin must be served"
        );
        for cmt in [1, 0] {
            let mut ss = plain.clone();
            ss.doses = vec![DoseEvent::new(0.0, 100.0, cmt, 0.0, true, 12.0)];
            assert!(m.ss_absorption_out_of_scope(&ss), "{label}, CMT={cmt}");
            let p = m.default_params.clone();
            let x = pack_params(&p);
            assert!(
                subject_adjoint_packed_gradient(&m, &ss, &p, &x, &eta, true).is_none(),
                "{label}, CMT={cmt}: an SS absorption dose without a dual seed must decline"
            );
        }
    }
}

#[cfg(feature = "survival")]
const JOINT_PKTTE_ODE: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.02, 1e-5, 10.0)
  theta TVBETA(0.30, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[event_model]
  cmt    = 3
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
";

/// `ss_dual_equilibration_out_of_scope` (#1210) and the non-Gaussian routing (#490, #905):
/// on a joint PK-TTE model the outer gradient is not the adjoint's (`model_in_scope` is
/// false) while the inner recorder still serves a Gaussian-only subject; an SS dose on that
/// model declines (the banked hazard run-in has no dual SS seed), and a subject carrying a
/// survival record declines at the likelihood scope. The same SS subject on the model
/// without the accumulator is served (straddle).
#[cfg(feature = "survival")]
#[test]
fn joint_pktte_routing_declines_ss_and_non_gaussian_subjects() {
    let m = parse_model_string(JOINT_PKTTE_ODE).unwrap();
    assert!(m.has_non_gaussian() && recorder_in_scope(&m) && !model_in_scope(&m));
    let times = [1.0, 4.0, 8.0, 12.0];
    let mk = |ss: bool| {
        let mut s = subject_shell(
            vec![DoseEvent::new(
                0.0,
                100.0,
                1,
                0.0,
                ss,
                if ss { 12.0 } else { 0.0 },
            )],
            &times,
        );
        s.obs_cmts = vec![2; times.len()];
        s.observations = vec![3.0, 2.5, 1.5, 1.0];
        s
    };
    let (plain, ss) = (mk(false), mk(true));
    let p = m.default_params.clone();
    let x = pack_params(&p);
    assert!(!crate::sens::ode_provider::ss_dual_equilibration_out_of_scope(&m, &plain));
    assert!(crate::sens::ode_provider::ss_dual_equilibration_out_of_scope(&m, &ss));
    assert!(
        exact_eta_jacobian(&m, &plain, &p.theta, &[0.0]).is_some(),
        "plain is served"
    );
    assert!(
        exact_eta_jacobian(&m, &ss, &p.theta, &[0.0]).is_none(),
        "SS must decline"
    );
    assert!(subject_adjoint_packed_gradient(&m, &ss, &p, &x, &[0.0], true).is_none());
    // A subject with a survival record is outside the Gaussian likelihood scope.
    let mut tte = plain.clone();
    tte.obs_records = vec![crate::types::ObsRecord::Event {
        time: 10.0,
        event_type: crate::types::EventType::Exact,
        entry_time: 0.0,
        cmt: 3,
    }];
    assert!(!likelihood_supported(&m, &tte, &p));
    assert!(subject_adjoint_packed_gradient(&m, &tte, &p, &x, &[0.0], true).is_none());
    // Straddle: the accumulator, not SS dosing alone, is what declines.
    let pk = parse_model_string(ONE_CPT_ORAL_ODE).unwrap();
    let mut ss_pk = ss.clone();
    ss_pk.obs_cmts = vec![1; times.len()];
    assert!(!crate::sens::ode_provider::ss_dual_equilibration_out_of_scope(&pk, &ss_pk));
    assert!(exact_eta_jacobian(&pk, &ss_pk, &pk.default_params.theta, &[0.0, 0.0]).is_some());
}

/// `analytic_terminal_work`: Laplace/AGQ reuse the terminal `(∂f/∂η, H)` only where the
/// analytic score applies and the subject has no occasions; where it does, the Jacobian is
/// the recorder's own, bit for bit (and that recorder is pinned against FD of the predictor
/// by the phenomena tests above). An IOV subject declines (`None`).
#[test]
fn analytic_terminal_work_routes_as_documented() {
    use crate::estimation::inner_optimizer::analytic_terminal_work;
    let m = tight(parse_model_string(ODE_LAG).unwrap());
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let s = subject(
        &m,
        doses,
        &[1.0, 2.0, 4.0, 8.0, 13.0, 14.0, 18.0, 24.0],
        &[0.1, -0.05],
    );
    let p = m.default_params.clone();
    let eta = probe_eta(m.n_eta);
    assert!(crate::estimation::agq::analytic_score_supported(&m));
    let (jac, h) = analytic_terminal_work(&m, &s, &p, &eta).expect("served");
    let direct = exact_eta_jacobian(&m, &s, &p.theta, &eta).unwrap();
    assert_eq!(jac.shape(), direct.shape());
    for (a, b) in jac.iter().zip(direct.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "terminal Jacobian is not the recorder's"
        );
    }
    let hd = exact_inner_hessian(&m, &s, &p, &eta).unwrap();
    for (a, b) in h.iter().zip(hd.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "terminal Hessian is not the recorder's"
        );
    }
    // An IOV subject with occasions declines.
    let iov = tight(parse_model_string(IOV_LAG_ODE).unwrap());
    let si = iov_lag_subject(&iov, false, false);
    let pi = iov.default_params.clone();
    let st = [0.06, -0.04, 0.05, -0.03, 0.04];
    assert!(
        exact_eta_jacobian(&iov, &si, &pi.theta, &st).is_some(),
        "recorder serves it"
    );
    assert!(
        analytic_terminal_work(&iov, &si, &pi, &st).is_none(),
        "IOV must decline"
    );
}

/// Worst gap between the exact inner η-Hessian and a Richardson difference of the
/// production individual objective.
fn inner_hessian_gap(m: &CompiledModel, s: &Subject, eta: &[f64]) -> f64 {
    let p = &m.default_params;
    let h = exact_inner_hessian(m, s, p, eta).expect("exact Hessian");
    let l = |e: &[f64]| {
        let mut scratch = crate::pk::EventPkParams::with_capacity_for(s);
        crate::stats::likelihood::individual_nll_into_with_schedule_cov(
            m,
            s,
            &p.theta,
            e,
            &p.omega,
            &p.sigma.values,
            m.residual_cov_with(&p.residual_correlations, &p.theta),
            &mut scratch,
            None,
        )
    };
    let n = eta.len();
    let mut worst = 0.0f64;
    for a in 0..n {
        for b in 0..n {
            let d = |st: f64| {
                let at = |sa: f64, sb: f64| {
                    let mut e = eta.to_vec();
                    e[a] += sa;
                    e[b] += sb;
                    l(&e)
                };
                (at(st, st) - at(st, -st) - at(-st, st) + at(-st, -st)) / (4.0 * st * st)
            };
            let rich = (4.0 * d(1e-3) - d(2e-3)) / 3.0;
            assert!(rich.is_finite() && h[(a, b)].is_finite());
            worst = worst.max((h[(a, b)] - rich).abs() / (1.0 + rich.abs()));
        }
    }
    worst
}

/// #880 / #883: the lag curvature — the exact inner η-Hessian (Laplace's anchor, the EBE
/// search's metric) on a depot `LAGTIME` and a per-route `lag=`, η on the lag, two doses,
/// against a Richardson difference of the production individual objective. This is the
/// second-order lag term the old δ² saltation coefficient existed for (#880's sign-mirrored
/// `∂²f/∂η²`), now one order of the `cross_moving_boundary` flow; it was exact even before
/// #2093, unlike the third-order quantities above.
#[test]
fn inner_eta_hessian_on_an_ode_lag_matches_fd_of_the_production_objective() {
    // Measured 2.8e-10 (depot) / 8.4e-8 (route lag) on Linux; ~10× headroom each.
    for (src, label, bound) in [
        (ODE_LAG, "depot LAGTIME", 3e-9),
        (ROUTE_LAG_ODE, "route lag", 8e-7),
    ] {
        let m = tight(parse_model_string(src).unwrap());
        let s = two_dose_subject(&m);
        let w = inner_hessian_gap(&m, &s, &probe_eta(m.n_eta));
        eprintln!("{label}: inner η-Hessian vs FD {w:.2e}");
        assert!(w < bound, "{label}: inner η-Hessian vs FD {w:.2e}");
    }
}

/// [`ROUTE_LAG_ODE`]'s closed-form twin: the route lag is the depot dose's `lagtime`.
const ROUTE_LAG_CF: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  LAG = TVLAG * exp(ETA_LAG)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAG)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// #1075: a `TAD`-reading RHS across an estimated lag. `TAD` is anchored at the previous
/// arrival before the moving boundary and at this one after it, so the field's explicit time
/// dependence differs between the two sides and does not cancel at second order — the term
/// the δ²-truncated saltation omitted (it held `J·g` only). The moving-boundary flows shift the
/// `TAD` anchor with the flow's offset.
///
/// Measured on Linux, FOCEI gradient vs reconverged FD: 1.33e-4 before the flows (#2093 commit
/// `50194698`), 8.8e-6 after — and the residual is the oracle's: the `Ω_LAG` component it sits
/// in moved by 1.4e-5 *in the reference* between those two builds while the new adjoint lands
/// on the old reference to 2e-8. The bound sits between, at 4e-5. The R-matrix vs FD of the
/// gradient measures 1.2e-5–3.0e-5 across FD steps of 2.5e-4–4e-3 (the same noise), bound 1e-4;
/// its moving-boundary exactness is pinned more tightly by
/// [`third_order_derivatives_across_an_ode_moving_boundary_are_exact`].
#[test]
fn tad_reading_rhs_across_an_estimated_lag_matches_fd() {
    let src = ODE_LAG.replace(
        "d/dt(central) = KA * depot - (CL/V) * central",
        "d/dt(central) = KA * depot - (CL/V) * central * (1 + 0.3 * exp(-0.5 * TAD))",
    );
    let m = tight(parse_model_string(&src).unwrap());
    let prog = m.ode_spec.as_ref().unwrap().rhs_program.as_ref().unwrap();
    assert!(prog.uses_time_vars() && !prog.reads_absolute_time());
    let s = two_dose_subject(&m);
    assert!(served(&m, &s));
    let g = parity(&m, &s);
    eprintln!("TAD RHS, lag: FOCEI gradient vs reconverged FD {g:.2e}");
    assert!(g.is_finite() && g < 4e-5, "gradient {g:.2e}");
    let (h, _) = hessian_gaps(&m, &s, true);
    eprintln!("TAD RHS, lag: R-matrix vs FD {h:.2e}");
    assert!(h.is_finite() && h < 1e-4, "R-matrix {h:.2e}");
}

/// #2121 (ported from `main`'s forward-provider test): the dual ODE walk seeds `init(...)` from
/// the subject's first record in file order — here an EVID=2 row (WT 70) tied at TIME 0 with
/// an observation (WT 140), where the row wins (`crate::dosing::first_record`). The seed is
/// pinned to its closed form `f(0) = TVB·WT·exp(η_B)/V` at the row's WT, and the recorder's
/// η-Jacobian to FD of production: a dual seed read off the observation would make
/// `∂f(0)/∂η_B = f(0)` twice the value path's.
#[test]
fn ode_init_seed_at_an_evid2_obs_tie_reads_the_first_record() {
    const SRC: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVB(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_B  ~ 0.09
  sigma PROP_ERR ~ 0.01
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  BASE = TVB * WT * exp(ETA_B)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE
  d/dt(central) = -(CL/V) * central
[covariates]
  WT continuous
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let m = parse_model_string(SRC).expect("parse");
    let wt = |v: f64| HashMap::from([("WT".to_string(), v)]);
    let mut s = subject(&m, Vec::new(), &[0.0, 2.0, 4.0, 8.0], &[0.15, -0.10]);
    s.covariates = wt(140.0);
    s.obs_covariates = vec![wt(140.0); 4];
    s.pk_only_times = vec![0.0];
    s.pk_only_raw_times = vec![0.0];
    s.pk_only_covariates = vec![wt(70.0)];
    s.pk_only_occasions = vec![1];
    assert!(s.has_tv_covariates(), "fixture must carry TV covariates");
    let eta = [0.15, -0.10];
    let f = crate::pk::compute_predictions_with_tv(&m, &s, &m.default_params.theta, &eta);
    let seed = |w: f64| 10.0 * w * (-0.10_f64).exp() / 50.0;
    assert!(
        (f[0] - seed(70.0)).abs() < 1e-9,
        "f(0) = {} must be the EVID=2 row's seed {} (the observation's: {})",
        f[0],
        seed(70.0),
        seed(140.0)
    );
    let gap = jacobian_gap(&m, &s, &eta, "init seed at an EVID=2 / obs tie");
    eprintln!("init seed tie: η-Jacobian vs FD {gap:.2e}");
    assert!(gap < 1e-7, "η-Jacobian vs FD {gap:.2e}");
}

/// Why an SPD `H` keeps the identity off the mode (#2093 review, Codex finding 1). At
/// `η̂ = η* + ε` with `∇_η l = Hε`, `∂Ψ/∂x` differs from the converged objective's total
/// derivative by `O(|∇_η l|)`: the identity's error is *first order in the stationarity gap*.
/// The salvage a decline would take — the held-EBE gradient `∂F/∂x` at fixed `η̂` — omits the
/// EBE response altogether, an `O(1)` error. So at a strict mode the solve is kept whatever
/// `∇_η l` is; the indefinite branch declines, because an indefinite `H` off the mode is not
/// near a minimum and `H⁻¹` there means nothing (`mode_solve_declines_an_indefinite_hessian_off_the_mode`).
#[test]
fn off_the_mode_the_identity_beats_the_held_ebe_gradient() {
    let m = parse_model_string(ONE_CPT_ORAL).unwrap();
    let s = two_dose_subject(&m);
    let params = moved_params(&m);
    let x = pack_params(&params);
    let laplace = Objective::Agq {
        n_agq: 1,
        exact: true,
    };
    let star = ebe_for(&m, &s, &params, 1e-12, true);
    let truth = subject_adjoint_gradient(&m, &s, &params, &x, star.as_slice(), laplace).unwrap();
    let rel = |g: &[f64]| {
        g.iter()
            .zip(&truth)
            .map(|(a, b)| {
                assert!(a.is_finite());
                (a - b).abs() / (1.0 + b.abs())
            })
            .fold(0.0f64, f64::max)
    };
    let dir = [1.0, -0.5, 0.7];
    let mut errs = Vec::new();
    for delta in [1e-3, 1e-2] {
        let eta: Vec<f64> = star.iter().zip(&dir).map(|(e, d)| e + delta * d).collect();
        let h = exact_inner_hessian(&m, &s, &params, &eta).expect("recorded");
        assert!(
            h.clone().cholesky().is_some(),
            "a strict mode's neighbourhood: SPD"
        );
        let psi = subject_adjoint_gradient(&m, &s, &params, &x, &eta, laplace).expect("served");
        let held: Vec<f64> = (0..x.len())
            .map(|k| {
                let hk = 1e-5 * (1.0 + x[k].abs());
                let at = |d: f64| {
                    let mut xv = x.clone();
                    xv[k] += d;
                    let p = crate::estimation::parameterization::unpack_params(&xv, &params);
                    laplace_prod(&m, &s, &p, &eta)
                };
                (at(hk) - at(-hk)) / (2.0 * hk)
            })
            .collect();
        let (ep, eh) = (rel(&psi), rel(&held));
        eprintln!("δ = {delta:.0e}: identity {ep:.2e}, held EBE {eh:.2e}");
        errs.push((ep, eh));
    }
    // First order: ten times the gap, about ten times the error.
    let ratio = errs[1].0 / errs[0].0;
    assert!(
        (5.0..20.0).contains(&ratio),
        "identity error ratio {ratio:.2}"
    );
    for (ep, eh) in errs {
        assert!(eh > 10.0 * ep, "held EBE {eh:.2e} vs identity {ep:.2e}");
    }
}

/// Two combinations `docs/estimation/foce.qmd` once described both as FD fallbacks and as
/// analytic (#2093 review, Codex finding 4): ODE `iiv_on_ruv` × M3 (an "inner-loop-only FD
/// fallback") and a closed-form steady-state dose under an estimated lag (a "deliberate FD
/// exception"). Both are served by the recorder on both loops and match reconverged FD.
#[test]
fn ode_iiv_on_ruv_with_m3_and_closed_form_ss_with_lag_are_analytic() {
    // (1) ODE iiv_on_ruv × M3, non-IOV: inner (η-gradient) and outer served, matching FD.
    let src = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_RUV ~ 0.05
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  KA = TVKA
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
"#;
    let mut m = tight(parse_model_string(src).unwrap());
    m.bloq_method = crate::types::BloqMethod::M3;
    let s = censor(two_dose_subject(&m), &[0, 7]);
    let inner =
        crate::sens::provider::subject_eta_grad(&m, &s, &m.default_params.theta, &[0.05, 0.02]);
    assert!(
        inner.is_some() && served(&m, &s),
        "ODE iiv_on_ruv × M3 must be served"
    );
    let g = parity(&m, &s);
    eprintln!("ODE iiv_on_ruv × M3: gradient vs reconverged FD {g:.2e}");
    // Measured 6.2e-7 on Linux; ~10× headroom.
    assert!(g < 6e-6, "{g:.2e}");
    // (2) closed-form SS bolus + estimated lag.
    let cf = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.1
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  KA = TVKA
  ALAG = TVLAG * exp(ETA_LAG)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=ALAG)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let m = parse_model_string(cf).unwrap();
    let s = subject(
        &m,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &[1.0, 2.0, 4.0, 8.0, 13.0, 14.0, 18.0, 24.0],
        &[0.1, -0.05],
    );
    let inner =
        crate::sens::provider::subject_eta_grad(&m, &s, &m.default_params.theta, &[0.05, 0.02]);
    assert!(
        inner.is_some() && served(&m, &s),
        "closed-form SS + lag must be served"
    );
    let g = parity(&m, &s);
    eprintln!("closed-form SS + lag: gradient vs reconverged FD {g:.2e}");
    // Measured 4.5e-6 on Linux; ~10× headroom.
    assert!(g < 5e-5, "{g:.2e}");
}

/// Modeled `D1`/`R1` on the **closed-form** event walk (#2093 review, Codex finding 2): the
/// same fixtures as the ODE tests above, with the structure swapped for `one_cpt_iv`. The
/// walk reads each dose's modeled slot as a dual (`slot_dual` in `record_analytic_events`);
/// read as a constant, the predictions stay right and only the moving infusion-end
/// derivative is lost, which these tests see.
#[test]
fn modeled_dose_on_the_closed_form_walk_matches_reconverged_fd() {
    let cf = |src: &str| {
        src.replace(
            "[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -CL/V * central\n[scaling]\n  y = central / V\n",
            "[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n",
        )
    };
    for (src, mode, label) in [
        (
            cf(MODELED_DUR_ODE),
            RateMode::ModeledDuration,
            "closed-form D1",
        ),
        (
            cf(MODELED_RATE_ODE),
            RateMode::ModeledRate,
            "closed-form R1",
        ),
    ] {
        let m = parse_model_string(&src).unwrap();
        assert!(m.ode_spec.is_none(), "{label}: must be closed form");
        let s = modeled_subject(&m, mode, &[1.0, 3.0, 6.0, 7.5, 9.0, 12.0, 14.0, 20.0]);
        assert!(!s.all_doses_fixed());
        let (jac, grad) = measure(&m, &s, &[0, 2], label);
        eprintln!("{label}: jac {jac:.2e} grad {grad:.2e}");
        // Measured jac 2.1e-12 / 1.8e-12, gradient 9.7e-9 / 7.0e-7 (D1 / R1) on Linux;
        // ~10× headroom on the worse.
        assert!(
            jac < 2e-11 && grad < 7e-6,
            "{label}: jac {jac:.2e} grad {grad:.2e}"
        );
    }
}
