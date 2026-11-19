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

// ---- a moving rate-off a few ULPs below a covariate record (#1160) --------------------

/// The infusion arm: a modeled duration `D1`, so the window end is a `K_INF_END`.
const MODELED_DUR_TVCOV_ODE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVD1(0.1, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV
  D1 = TVD1 * exp(ETA_D1)
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

/// The zero-order arm: the same geometry through `zero_order(dur)`, a `K_ZO_END`.
const ZERO_ORDER_TVCOV_ODE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVDUR(0.1, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - (CL/V) * central
[covariates]
  WT continuous
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// #1160: a moving rate-off a few ULPs **below** a covariate-changing record. A window
/// `0.7 + 0.1` ends at `0.7999999999999999`, so the timeline sorts its rate-off before the
/// observation at `0.8` (`WT = 70`), and the next record (`WT = 95`, at `2.0`) lies beyond
/// `INFUSION_EPS`. The flow on both sides of that window end is the one the `0.8` record
/// governs; and since moving a rate-off only shifts `rate·δ` of input across the stationary
/// `0.8` field jump, the first derivative is continuous there — the one-sided limits agree
/// (asserted), so either is the oracle.
///
/// The removed lookahead read the post side from the first record strictly later by more
/// than `INFUSION_EPS` — the `2.0` one — and so charged the *stationary* `0.8` record's field
/// jump `(f₇₀ − f₉₅)(x)·δ` to the moving boundary: a first-order term on neither branch (the
/// #1068 shape, on the rate-off side). Returns `(exact vs backward limit, |fwd − bwd|)` on the
/// window-length η's column, over the rows past the boundary.
fn ulp_below_rate_off_gaps(src: &str, mode: RateMode, label: &str) -> (f64, f64) {
    let m = tight(parse_model_string(src).unwrap());
    let (t_dose, t_rec) = (0.7, 0.8);
    let theta = m.default_params.theta.clone();
    let w_end = t_dose + theta[3];
    // The sliver itself, asserted so the fixture cannot silently stop being one.
    assert!(
        w_end < t_rec && t_rec - w_end <= crate::ode::predictions::INFUSION_EPS,
        "{label}: window end {w_end:e} must sit within INFUSION_EPS below {t_rec}"
    );
    let times = [t_rec, 2.0, 3.0, 5.0];
    let wts = [70.0, 95.0, 95.0, 95.0];
    // An earlier dose, so the window ends onto drug that is not just its own.
    let doses = vec![
        DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, mode),
        DoseEvent::modeled(t_dose, 100.0, 1, false, 0.0, mode),
    ];
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut s = subject_shell(doses, &times);
    s.covariates.insert("WT".into(), 70.0);
    s.dose_covariates = vec![wt(70.0), wt(70.0)];
    s.obs_covariates = wts.iter().map(|&w| wt(w)).collect();
    assert!(s.has_tv_covariates(), "{label}");

    // The window-length η at exactly 0, so the window is `TV = 0.1` and ends on the sliver.
    let eta = [0.05, 0.0];
    let jac = exact_eta_jacobian(&m, &s, &theta, &eta)
        .unwrap_or_else(|| panic!("{label}: the recorder must serve this subject"));
    let pred =
        |e: f64| crate::pk::compute_predictions_with_tv(&m, &s, &theta, &[eta[0], eta[1] + e]);
    // Live: the covariate jump moves the post-record rows, so a snapshot read from the wrong
    // side of it is visible (the `WT = 95` field is what the old post side read).
    let f0 = pred(0.0);
    let flat = {
        let mut t = s.clone();
        t.obs_covariates = times.iter().map(|_| wt(70.0)).collect();
        crate::pk::compute_predictions_with_tv(&m, &t, &theta, &eta)
    };
    let jump = (f0[1] - flat[1]).abs() / flat[1].abs();
    assert!(
        jump > 1e-2,
        "{label}: the WT change must move the t = 2 row ({jump:.2e})"
    );
    // One-sided Richardson limits, h ≫ the ULP gap: backward stays below the `0.8` record,
    // forward crosses it.
    let h = 1e-5;
    let (fm, fm2, fp, fp2) = (pred(-h), pred(-2.0 * h), pred(h), pred(2.0 * h));
    let (mut worst_bwd, mut fwd_gap, mut col) = (0.0f64, 0.0f64, 0.0f64);
    for j in 1..times.len() {
        let bwd = (3.0 * f0[j] - 4.0 * fm[j] + fm2[j]) / (2.0 * h);
        let fwd = (-3.0 * f0[j] + 4.0 * fp[j] - fp2[j]) / (2.0 * h);
        let got = jac[(j, 1)];
        assert!(
            got.is_finite() && bwd.is_finite() && fwd.is_finite(),
            "{label}, row {j}: non-finite {got} / {bwd} / {fwd}"
        );
        worst_bwd = worst_bwd.max((got - bwd).abs() / (1.0 + bwd.abs()));
        fwd_gap = fwd_gap.max((fwd - bwd).abs() / (1.0 + bwd.abs()));
        col = col.max(bwd.abs());
    }
    assert!(
        col > 1e-3,
        "{label}: the window length moves no row past it ({col:.1e})"
    );
    eprintln!("{label}: exact vs backward {worst_bwd:.2e}; |fwd − bwd| {fwd_gap:.2e}");
    (worst_bwd, fwd_gap)
}

#[test]
fn infusion_end_a_few_ulps_below_a_covariate_record_reads_one_snapshot() {
    let (gap, fwd_bwd) = ulp_below_rate_off_gaps(
        MODELED_DUR_TVCOV_ODE,
        RateMode::ModeledDuration,
        "K_INF_END",
    );
    // Measured 1.0e-10 and |fwd − bwd| 1.1e-10 on macOS, 1.41e-2 before #1160; the bounds
    // sit ~10× and ~30× above the first two and seven decades below the defect.
    assert!(
        fwd_bwd < 3e-9,
        "K_INF_END: the one-sided limits must agree ({fwd_bwd:.2e})"
    );
    assert!(
        gap < 1e-9,
        "K_INF_END: exact ∂f/∂η_D1 vs backward limit {gap:.2e}"
    );
}

#[test]
fn zero_order_end_a_few_ulps_below_a_covariate_record_reads_one_snapshot() {
    let (gap, fwd_bwd) = ulp_below_rate_off_gaps(ZERO_ORDER_TVCOV_ODE, RateMode::Fixed, "K_ZO_END");
    // Measured 1.3e-10 and |fwd − bwd| 2.6e-10 on macOS, 1.41e-2 before #1160.
    assert!(
        fwd_bwd < 3e-9,
        "K_ZO_END: the one-sided limits must agree ({fwd_bwd:.2e})"
    );
    assert!(
        gap < 1e-9,
        "K_ZO_END: exact ∂f/∂η_DUR vs backward limit {gap:.2e}"
    );
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
        assert!(s.has_periodic_ss_dose());
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

/// [`IV_LAG_CROSS_ODE`] with a second state fed by a built-in `first_order` forcing whose
/// `KA` reads the time-varying weight — the kernel `R_in(tad)` is an explicit function of
/// time *and* of the PK snapshot.
const LAG_STRADDLE_INPUT_RATE_ODE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVP(0.7, 0.05, 5.0)
  theta TVKA(0.8, 0.01, 10.0)
  theta TVK21(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_P  ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  KA = TVKA * (WT/70)^THETA_WT
  K21 = TVK21
  ALAG1 = TVP * exp(ETA_P)
[structural_model]
  ode(obs_cmt=central, states=[central, periph])
[odes]
  d/dt(central) = -(CL/V) * central + K21 * periph
  d/dt(periph)  = first_order(ka=KA) - K21 * periph
[covariates]
  WT continuous
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// #1075 (the geometry it was filed on, re-anchored from the retired forward provider's
/// `ode_provider_lagtime_tvcov_arrival_straddles_input_rate_matches_production`): a lagged
/// bolus into `central` arrives across a covariate change while a `first_order` forcing from
/// an earlier, unlagged dose into `periph` is active on both sides, its `KA` reading `WT`.
/// The forward provider's saltation carried only `J·f` of `ẍ = ∂f/∂t + J·f` and missed
/// `∂²f/∂η_LAG²` here by 6.4e-2, because each side evaluated the kernel at a different
/// snapshot. Two changes since make it exact: the kernel reads its own dose record's
/// snapshot (#1575), so `R_in(tad)` is one function on both sides of the arrival; and the
/// adjoint crosses every moving boundary by flowing over the shift (#2093), carrying `∂f/∂t`.
///
/// The oral dose row carries `WT = 90`, not the baseline 70, so `(WT/70)^THETA_WT ≠ 1` and
/// the covariate actually reaches the kernel (asserted against a `KA = TVKA` twin). Checked
/// at first order (`∂f/∂η`, FOCEI gradient) and at second order: the exact inner η-Hessian
/// (which carries `∂²f/∂η²`) against a Richardson difference of the production objective,
/// and the R-matrix against reconverged FD of the gradient.
#[test]
fn lagged_arrival_across_a_covariate_change_under_a_snapshot_reading_input_rate_is_exact() {
    let m = tight(parse_model_string(LAG_STRADDLE_INPUT_RATE_ODE).unwrap());
    assert!(
        !m.ode_spec.as_ref().unwrap().input_rate.is_empty(),
        "fixture must carry a built-in input-rate forcing"
    );
    // No t = 0 row: it predicts 0, where the proportional variance floor puts a kink in the
    // objective (see `tvcov_lag_arrival_across_a_covariate_change_matches_reconverged_fd`).
    let times = [0.5, 1.5, 2.0, 4.0, 8.0];
    let wts = [70.0, 70.0, 140.0, 150.0, 160.0];
    // The oral dose feeds `periph`'s `first_order` kernel unlagged (`ALAG1` lags compartment
    // 1 only); the lagged bolus lands in `central`.
    let doses = vec![
        DoseEvent::new(0.0, 400.0, 2, 0.0, false, 0.0),
        DoseEvent::new(1.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let mut s = subject_shell(doses, &times);
    s.covariates.insert("WT".into(), 70.0);
    s.dose_covariates = vec![wt(90.0), wt(70.0)];
    s.obs_covariates = wts.iter().map(|&w| wt(w)).collect();
    let eta0 = [0.1, -0.05, 0.07];
    let theta = &m.default_params.theta;
    let preds = crate::pk::compute_predictions_with_tv(&m, &s, theta, &eta0);
    assert!(preds.iter().all(|&v| v > 0.0), "every row positive");
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
        .collect();
    assert!(s.has_tv_covariates() && m.has_lagtime());
    let differs = |other: &[f64], what: &str| {
        assert!(
            preds
                .iter()
                .zip(other)
                .any(|(a, b)| (a - b).abs() > 1e-3 * b.abs().max(1e-6)),
            "{what} changes no prediction: the fixture is vacuous"
        );
    };
    // Non-vacuity: the straddling forcing carries mass, and `KA` reads the covariate.
    let mut bolus_only = s.clone();
    bolus_only.doses = vec![s.doses[1].clone()];
    bolus_only.dose_covariates = vec![wt(70.0)];
    differs(
        &crate::pk::compute_predictions_with_tv(&m, &bolus_only, theta, &eta0),
        "the `first_order` forcing",
    );
    let ka_free = parse_model_string(
        &LAG_STRADDLE_INPUT_RATE_ODE.replace("KA = TVKA * (WT/70)^THETA_WT", "KA = TVKA"),
    )
    .unwrap();
    differs(
        &crate::pk::compute_predictions_with_tv(&ka_free, &s, theta, &eta0),
        "`WT` on `KA`",
    );
    // The straddle: at the moved θ and its EBE the bolus arrives strictly inside the
    // [1.5, 2.0] segment, whose snapshot (WT = 140) is not the dose row's (WT = 70).
    let p = moved_params(&m);
    let e = ebe_for(&m, &s, &p, 1e-12, true);
    let arrival = 1.0 + p.theta[3] * e[2].exp();
    assert!(
        arrival > 1.5 + 1e-2 && arrival < 2.0 - 1e-2,
        "arrival {arrival} must sit inside the [1.5, 2.0] segment"
    );
    let label = "#1075 straddling input rate";
    let (jac, grad) = measure(&m, &s, &[0, 2, 3, 4, 5], label);
    let inner = inner_hessian_gap(&m, &s, &probe_eta(m.n_eta));
    let (r, _) = hessian_gaps(&m, &s, true);
    eprintln!("{label}: inner η-Hessian vs FD {inner:.2e}; R-matrix vs FD {r:.2e}");
    // Measured on macOS arm64: ∂f/∂η 4.9e-11, gradient 6.7e-9, inner η-Hessian 7.5e-10,
    // R-matrix 5.1e-10; ~10× headroom.
    assert!(jac < 6e-10, "{label}: ∂f/∂η {jac:.2e}");
    assert!(grad < 7e-8, "{label}: FOCEI gradient {grad:.2e}");
    assert!(inner < 8e-9, "{label}: inner η-Hessian {inner:.2e}");
    assert!(r.is_finite() && r < 6e-9, "{label}: R-matrix {r:.2e}");
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

/// κ on the `init(...)` baseline a reset re-seeds, and on `CL`.
const IOV_RESET_SEED_ODE: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVBASE(20.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA)
  V = TVV
  BASE = TVBASE * exp(KAPPA)
[structural_model]
  ode(states=[central])
[odes]
  init(central) = BASE
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// #1153: an occasion that ONLY a reset row carries is a κ group, on the ODE adjoint's
/// static-covariate branch. The reset at 18 is labelled 7, which no dose or observation
/// carries; the observations after it are occasion 2. Before #1153 occasion 7 had no
/// group and the static branch's `occ_to_k.get(&occ)?` declined the whole subject to FD
/// (`served` false), while `predict_iov` quietly seeded it under κ = 0. Now the recorded
/// `∂f/∂κ₇` must be live — `iov_jacobian_gap` asserts every stacked column moves a
/// prediction, and only the post-reset seed reads κ₇ — and match FD of `predict_iov`.
#[test]
fn iov_reset_only_occasion_matches_reconverged_fd() {
    let m = tight(parse_model_string(IOV_RESET_SEED_ODE).unwrap());
    let times = [1.0, 4.0, 8.0, 13.0, 16.0, 19.0, 21.0, 24.0, 30.0];
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let mut s = subject_shell(doses, &times);
    s.occasions = vec![1, 1, 1, 2, 2, 2, 2, 2, 2];
    s.dose_occasions = vec![1, 2];
    s.reset_times = vec![18.0];
    s.reset_occasions = vec![7];
    assert!(
        !s.has_tv_covariates() && s.reset_covariates.is_empty(),
        "the fixture must take the static-covariate seed branch"
    );
    let groups = crate::stats::likelihood::iov_occasion_groups(&s);
    assert_eq!(
        groups.iter().map(|g| g.0).collect::<Vec<_>>(),
        vec![1, 2, 7],
        "the reset-only occasion is the last group"
    );
    let kappas = vec![vec![0.1], vec![-0.08], vec![0.12]];
    let preds = crate::pk::predict_iov(&m, &s, &m.default_params.theta, &[0.1], &kappas);
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.12 * (1.7 * j as f64).sin()))
        .collect();
    // Second arm: the same reset with no label (a `Subject` built in memory), plus a second
    // unlabelled reset at 26 so the static branch reuses the κ = 0 seeding it built for the
    // first. Both seed under κ = 0 on every engine and must still be served, not declined
    // to FD.
    let mut unlabelled = s.clone();
    unlabelled.reset_occasions.clear();
    unlabelled.reset_times.push(26.0);
    // Third arm: the labelled reset with its own covariate snapshot, which sends the
    // subject down the per-event seed branch (`iov_walk_per_event`) instead of the static
    // one. The model reads no covariate, so only the branch changes, not the answer.
    let mut per_event = s.clone();
    per_event.reset_covariates = vec![per_event.covariates.clone()];
    assert!(
        crate::sens::ode_provider::iov_walk_per_event(&m, &per_event)
            && !crate::sens::ode_provider::iov_walk_per_event(&m, &s),
        "the third arm must take the per-event seed branch and the first the static one"
    );
    for (s, stacked, label) in [
        (
            &s,
            &[0.06, -0.04, 0.05, 0.08][..],
            "IOV × reset-only occasion",
        ),
        (
            &per_event,
            &[0.06, -0.04, 0.05, 0.08][..],
            "IOV × reset-only occasion, per-event seeding",
        ),
        (
            &unlabelled,
            &[0.06, -0.04, 0.05][..],
            "IOV × unlabelled reset",
        ),
    ] {
        assert!(
            served(&m, s),
            "{label}: the adjoint must serve this subject"
        );
        let jac = iov_jacobian_gap(&m, s, stacked, label);
        let grad = fd_oracle(&m, s, true);
        eprintln!("{label}: ∂f/∂[η, κ] vs FD {jac:.2e}; FOCEI gradient vs FD {grad:.2e}");
        // Measured worst jac 1.8e-12, grad 1.6e-10 on macOS arm64; ~10× (jac) and ~30×
        // (grad, whose siblings above run to 4e-7 on Linux) headroom.
        assert!(jac < 2e-11, "{label} ∂f/∂[η, κ]: {jac:.2e}");
        assert!(grad < 5e-9, "{label} gradient: {grad:.2e}");
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
        // Measured worst jac 6.6e-10 on Linux; ~10× headroom. The gradient bound is set by the
        // reconverged-FD reference's own accuracy, which is the stationarity of its EBE
        // re-solves: 1.05e-6 before #2093, when a tightly integrated ODE solve stopped on the
        // objective's resolution; 5.4e-8 (route lag) and 6.5e-9 (two routes) since the
        // noise-floor exit takes a step the exact gradient verifies (`gradient_verified_step`).
        // Without that verification (run): 4.2e-6 and 1.5e-6. The bound is ~9× over the fix
        // and ~8× under the mutant.
        assert!(jac < 7e-9, "{label} ∂f/∂η: {jac:.2e}");
        assert!(grad < 5e-7, "{label} gradient: {grad:.2e}");
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
pub(super) fn fd_oracle_agq_iov(m: &CompiledModel, s: &Subject, n_agq: usize, exact: bool) -> f64 {
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
/// (two occasions, stacked width 4), against the
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

/// An infusion into a built-in absorption kernel (#719 gap 2) with a fixed window is served on
/// both ODE walks: the generic walk delivers it through the convolved `R_in_inf` only (no plain
/// `+rate`, as the f64 predictor), the segment saying which side of the window end it lies on.
/// Multi-dose (a bolus lands on the infusion's residual drug). Two fixtures: a rate-defined
/// infusion on a model without `F` and a duration-defined one under an estimated `F` (the
/// window is fixed, the rate scales with `F`) are served and exact. The windows that move are
/// `moving_window_infusions_into_an_absorption_kernel_match_reconverged_fd`.
#[test]
fn an_infusion_into_an_absorption_kernel_matches_reconverged_fd() {
    let times = [1.0, 2.5, 4.0, 6.0, 8.0, 9.5, 11.0, 14.0, 18.0];
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 40.0, false, 0.0),
        DoseEvent::new(6.0, 80.0, 1, 0.0, false, 0.0),
    ];
    // No `F` at all: one η.
    let no_f = SS_FIRST_ORDER_F
        .replace(
            "  theta THETA_F(0.7, 0.001, 0.999)
",
            "",
        )
        .replace(
            "  omega ETA_F ~ 0.04
",
            "",
        )
        .replace(
            "  F  = inv_logit(logit(THETA_F) + ETA_F)
",
            "",
        );
    let m_no_f = tight(parse_model_string(&no_f).unwrap());
    let m_f = tight(parse_model_string(SS_FIRST_ORDER_F).unwrap());
    let rate_no_f = subject(&m_no_f, doses.clone(), &times, &[0.1]);
    let rate_f = subject(&m_f, doses.clone(), &times, &[0.1, -0.05]);
    let mut dur_f = rate_f.clone();
    dur_f.doses[0].infusion_def = crate::types::InfusionDef::DurationDefined;
    assert!(rate_f.doses[0].is_infusion() && !rate_f.doses[1].is_infusion());
    for (m, s, live, label) in [
        (&m_no_f, &rate_no_f, &[0usize, 2][..], "rate-defined, no F"),
        (
            &m_f,
            &dur_f,
            &[0, 2, 3][..],
            "duration-defined, F estimated",
        ),
    ] {
        assert!(served(m, s), "{label}: the infusion must be served");
        let eta = probe_eta(m.n_eta);
        for &k in live {
            let e = theta_effect(m, s, &eta, k);
            assert!(e > 1e-6, "{label}: θ[{k}] moves no prediction ({e:.1e})");
        }
        let jac = jacobian_gap(m, s, &eta, label);
        let grad = parity(m, s);
        eprintln!(
            "infusion into kernel, {label}: ∂f/∂η vs FD {jac:.2e}, gradient vs FD {grad:.2e}"
        );
        // Measured on Linux: ∂f/∂η 1.1e-9 / 8.8e-10 and gradient 2.0e-7 / 2.5e-9 (rate-defined
        // without F / duration-defined with F); ≥ 10× headroom.
        assert!(jac < 2e-8, "{label} ∂f/∂η: {jac:.2e}");
        assert!(grad < 2e-6, "{label} gradient: {grad:.2e}");
    }
}

/// An infusion (RATE > 0) into the compartment of an `igd` input kernel (#3004): the convolved
/// `R_in_inf` evaluates the kernel's CDF, `Phi` and `ln Phi`, whose branches (the `erfc` tail,
/// the Mills switch) used to be guards of the recorded ODE body, which fail at another stage of
/// the segment, so every subject with an infusion declined to FD. A rate-defined infusion first,
/// a bolus landing on its residual drug second, observations during the window and after it.
/// (The `transit` and `weibull` kernels were measured exact on the same fixture: their CDF is
/// one recorded primitive.)
#[test]
fn an_infusion_into_an_igd_kernel_matches_reconverged_fd() {
    let times = [0.5, 1.0, 1.8, 2.5, 4.0, 6.5, 7.0, 9.0, 12.0, 18.0];
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(6.0, 80.0, 1, 0.0, false, 0.0),
    ];
    // At reltol 1e-12, for the reason given at the Weibull / igd forcing test.
    let m = with_tol(parse_model_string(IGD_ODE).unwrap(), 1e-12, 1e-14);
    let s = subject(&m, doses, &times, &[0.1, -0.05]);
    assert!(s.doses[0].is_infusion() && !s.doses[1].is_infusion());
    let (jac, grad) = measure(&m, &s, &[0, 2, 3], "igd infusion");
    // Measured 1.1e-11 / 5.3e-9 on Linux; ~10x headroom on the gradient, the jacobian bound
    // sitting above the FD oracle's own noise.
    assert!(jac < 1e-9, "igd infusion df/deta: {jac:.2e}");
    assert!(grad < 5e-8, "igd infusion gradient: {grad:.2e}");
}

/// A first-order absorption model with the given extra `[parameters]` / individual
/// parameters and `first_order` arguments, for the moving-window infusions below.
fn kernel_infusion_model(theta: &str, omega: &str, ind: &str, args: &str) -> CompiledModel {
    tight(
        parse_model_string(&format!(
            "[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
{theta}
  omega ETA_CL ~ 0.09
{omega}
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
{ind}
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA{args}) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"
        ))
        .unwrap(),
    )
}

/// An infusion into an absorption kernel whose window END or START moves with the parameters
/// — a rate-defined one under an estimated `F` (the window is `F·amt/rate`), modeled
/// `RATE=-1` / `RATE=-2` doses, and any under a compartment or a per-route lag — is exact on
/// the event walk: the end is a break of the integration and a kink of the convolved
/// `R_in_inf` (`∂R/∂T` jumps by `(dose/T)·g(0)`), crossed as a moving boundary whose two
/// sides differ in which segment membership the forcing reads. Each fixture has two
/// infusions, the later one landing on the earlier one's residual drug, and observations on
/// both sides of every window start and end; each η that moves a window moves a prediction
/// (`measure`).
#[test]
fn moving_window_infusions_into_an_absorption_kernel_match_reconverged_fd() {
    let times = [0.5, 1.5, 2.8, 3.5, 5.0, 6.7, 8.5, 10.5, 12.0, 16.0, 22.0];
    let fixed = |t2: f64| {
        vec![
            DoseEvent::new(0.0, 100.0, 1, 40.0, false, 0.0),
            DoseEvent::new(t2, 80.0, 1, 20.0, false, 0.0),
        ]
    };
    let modeled = |mode: RateMode| {
        vec![
            DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, mode),
            DoseEvent::modeled(6.0, 80.0, 1, false, 0.0, mode),
        ]
    };
    let lag = ("  theta TVLAG(1.0, 0.01, 5.0)", "  omega ETA_LAG ~ 0.04");
    let lagged = "  LAGTIME = TVLAG * exp(ETA_LAG)";
    let route = "  RLAG = TVLAG * exp(ETA_LAG)";
    let f_par = (
        "  theta THETA_F(0.7, 0.001, 0.999)",
        "  omega ETA_F ~ 0.04",
        "  F  = inv_logit(logit(THETA_F) + ETA_F)",
    );
    let cases: Vec<(&str, CompiledModel, Vec<DoseEvent>, Vec<usize>)> = vec![
        (
            "rate-defined, F estimated",
            kernel_infusion_model(f_par.0, f_par.1, f_par.2, ""),
            fixed(6.0),
            vec![0, 2, 3],
        ),
        (
            "compartment lag",
            kernel_infusion_model(lag.0, lag.1, lagged, ""),
            fixed(6.0),
            vec![0, 2, 3],
        ),
        (
            "per-route lag",
            kernel_infusion_model(lag.0, lag.1, route, ", lag=RLAG"),
            fixed(6.0),
            vec![0, 2, 3],
        ),
        (
            "modeled RATE, R1",
            kernel_infusion_model(
                "  theta TVR1(20.0, 0.1, 200.0)",
                "  omega ETA_R1 ~ 0.04",
                "  R1 = TVR1 * exp(ETA_R1)",
                "",
            ),
            modeled(RateMode::ModeledRate),
            vec![0, 2, 3],
        ),
        (
            "modeled duration, D1",
            kernel_infusion_model(
                "  theta TVD1(4.0, 0.1, 24.0)",
                "  omega ETA_D1 ~ 0.04",
                "  D1 = TVD1 * exp(ETA_D1)",
                "",
            ),
            modeled(RateMode::ModeledDuration),
            vec![0, 2, 3],
        ),
    ];
    for (label, m, doses, live) in cases {
        let eta = [0.1, -0.05];
        let s = subject(&m, doses, &times, &eta[..m.n_eta.min(2)]);
        let (jac, grad) = measure(&m, &s, &live, label);
        eprintln!("moving-window infusion, {label}: ∂f/∂η {jac:.2e}, gradient {grad:.2e}");
        // Measured on Linux x86_64: ∂f/∂η ≤ 3.8e-11 and gradient ≤ 4.0e-9 over the five
        // fixtures; ≥ 12× headroom. Each saltation off (the window end, the compartment-lag
        // onset, the route's end) or the CDF's onset continuation off moves the gradient to
        // 1e-2 / 1e-2 / 2e-4 / 1.8e-2, so these bounds are decisive.
        assert!(jac < 5e-10, "{label} ∂f/∂η: {jac:.2e}");
        assert!(grad < 5e-8, "{label} gradient: {grad:.2e}");
    }
}

/// The covariance R-matrix (third and fourth order through the same boundaries) of a window
/// whose start and end both move: a compartment lag together with an estimated `F` on a
/// rate-defined infusion, so the end carries `lag + F·T` and the onset `lag`.
#[test]
fn a_moving_window_infusion_r_matrix_matches_fd_of_the_gradient() {
    let m = kernel_infusion_model(
        "  theta THETA_F(0.7, 0.001, 0.999)
  theta TVLAG(1.0, 0.01, 5.0)",
        "  omega ETA_F ~ 0.04
  omega ETA_LAG ~ 0.04",
        "  F  = inv_logit(logit(THETA_F) + ETA_F)
  LAGTIME = TVLAG * exp(ETA_LAG)",
        "",
    );
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 40.0, false, 0.0),
        DoseEvent::new(6.0, 80.0, 1, 20.0, false, 0.0),
    ];
    let times = [0.5, 1.5, 2.8, 3.5, 5.0, 6.7, 8.5, 10.5, 12.0, 16.0, 22.0];
    let s = subject(&m, doses, &times, &[0.1, -0.05, 0.08]);
    let (_, grad) = measure(&m, &s, &[0, 2, 3, 4], "F + lag");
    let (hess, _) = hessian_gaps(&m, &s, true);
    eprintln!("moving-window infusion, F + lag: gradient {grad:.2e}, R-matrix {hess:.2e}");
    assert!(hess < 1e-4, "R-matrix: {hess:.2e}");
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

/// A joint PK-TTE model (#1210, #490, #905) under a steady-state dose: its Gaussian rows are
/// served exactly — the event walk's dual SS run-in holds the injected accumulator still and
/// restores its record value, as the f64 run-in does — against the reconverged Richardson
/// oracle of production's objective. The same subject with an event record is served too: the
/// hazard's state solve takes the event-driven walk (`ode_states_at_g`), whose run-in holds the
/// accumulator, and its packed gradient matches the oracle as well.
#[cfg(feature = "survival")]
#[test]
fn joint_pktte_serves_ss_gaussian_rows_and_the_ss_hazard() {
    let m = tight(parse_model_string(JOINT_PKTTE_ODE).unwrap());
    assert!(m.has_non_gaussian() && recorder_in_scope(&m) && !model_in_scope(&m));
    assert!(outer_model_in_scope(&m));
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
    let ss = mk(true);
    let p = m.default_params.clone();
    assert!(
        exact_eta_jacobian(&m, &ss, &p.theta, &[0.0]).is_some(),
        "SS is served"
    );
    let mut ss_tte = ss.clone();
    ss_tte.obs_records = vec![crate::types::ObsRecord::Event {
        time: 10.0,
        event_type: crate::types::EventType::Exact,
        entry_time: 0.0,
        cmt: 3,
    }];
    assert!(
        !likelihood_supported(&m, &ss_tte, &p),
        "the kernel alone does not score it"
    );
    assert!(
        crate::survival::tte_record_served(&m, &ss_tte, 3),
        "the hazard under a steady-state dose is served"
    );
    let mut worst = 0.0f64;
    for (label, s) in [("Gaussian rows", &ss), ("with the hazard", &ss_tte)] {
        for interaction in [true, false] {
            let w = super::tests::fd_oracle(&m, s, interaction);
            eprintln!("joint PK-TTE SS {label} interaction={interaction}: worst {w:.2e}");
            assert!(w.is_finite(), "{label} interaction={interaction}: {w}");
            worst = worst.max(w);
        }
    }
    // Measured on Linux: Gaussian rows 5.3e-8 (FOCE), 5.1e-9 (FOCEI); with the hazard 8.9e-9
    // (FOCE), 8.3e-10 (FOCEI) — the oracle's floor; about 10× headroom.
    assert!(worst < 5e-7, "{worst:.2e}");
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
    assert!(prog.uses_time_vars() && !prog.pk_reads_absolute_time());
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
    // The SS dose as a bolus into the depot, and (#2786) as an infusion into the central
    // compartment, whose seed at the record carries the previous cycle's window: one that
    // closes before the record (2 h), and one still running across it (11.8 h against a phase
    // of `12 − ALAG ≈ 11.5 h`), whose residual window then ends at a lag-moved break. A second
    // dose lands on residual drug in each.
    for (label, cmt, rate) in [
        ("bolus", 1, 0.0),
        ("infusion closing before the record", 2, 50.0),
        ("infusion running across the record", 2, 100.0 / 11.8),
    ] {
        let s = subject(
            &m,
            vec![
                DoseEvent::new(0.0, 100.0, cmt, rate, true, 12.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ],
            &[0.1, 1.0, 2.0, 4.0, 8.0, 13.0, 14.0, 18.0, 24.0],
            &[0.1, -0.05],
        );
        let lag = m.default_params.theta[3];
        assert_eq!(
            crate::dosing::ss_residual_infusion_ends(
                &s.doses[0],
                lag,
                1.0,
                f64::NEG_INFINITY,
                f64::INFINITY,
            )
            .next()
            .is_some(),
            label.ends_with("across the record"),
            "{label}: the fixture's residual window"
        );
        let inner =
            crate::sens::provider::subject_eta_grad(&m, &s, &m.default_params.theta, &[0.05, 0.02]);
        assert!(
            inner.is_some() && served(&m, &s),
            "closed-form SS {label} + lag must be served"
        );
        let g = parity(&m, &s);
        eprintln!("closed-form SS {label} + lag: gradient vs reconverged FD {g:.2e}");
        // Measured 4.5e-6 (bolus) on Linux; ~10× headroom.
        assert!(g < 5e-5, "{label}: {g:.2e}");
    }
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

/// An RHS reading an **absolute** clock (`TIME`, `TAFD`) across an estimated lag, at third
/// order. The moving-boundary flows evaluate the field at `t + s` with `s` jet-only, and the
/// RHS binding carries that offset into every clock spelling (`OdeRhsProgram::eval_rhs_g`'s
/// `dt`; a bare `TIME` is compiled to read the time slot) — before which these recordings
/// declined: the absolute clock entered as an `f64` the flow could not shift.
#[test]
fn absolute_clock_rhs_across_an_estimated_lag_is_exact_at_third_order() {
    for (label, factor) in [("TIME", "(1 + 0.02 * TIME)"), ("TAFD", "(1 + 0.02 * TAFD)")] {
        let src = ODE_LAG.replace(
            "d/dt(central) = KA * depot - (CL/V) * central",
            &format!("d/dt(central) = KA * depot - (CL/V) * central * {factor}"),
        );
        let m = tight(parse_model_string(&src).unwrap());
        let s = two_dose_subject(&m);
        assert!(served(&m, &s), "{label}: served");
        let (h, _) = hessian_gaps(&m, &s, true);
        eprintln!("{label} RHS, lag: R-matrix vs FD {h:.2e}");
        // Measured 8.3e-6 (R-matrix) and 4.3e-6 (Laplace gradient) for both clocks on
        // Linux — the FD references' own noise on a time-dependent field (the TAD fixture
        // above moves 1.2e-5–3.0e-5 with the FD step) — so ~10× headroom over that.
        assert!(h.is_finite() && h < 8e-5, "{label}: R-matrix {h:.2e}");
        let g = super::tests::fd_oracle_agq(&m, &s, 1, true);
        eprintln!("{label} RHS, lag: Laplace gradient vs reconverged FD {g:.2e}");
        assert!(
            g.is_finite() && g < 4e-5,
            "{label}: Laplace gradient {g:.2e}"
        );
    }
}

/// An η-dependent `obs_scale` under LTBS on an ODE model, which took finite differences on
/// both loops: the walk applied the LTBS log inside its readout, before the expression
/// scale the adjoint applies after the walk, so it could not reproduce production's
/// scale-then-log `ln(f/s)`. The readout now defers the log under an expression scale and
/// `apply_expression_scale` takes it after the quotient. Pinned on the values (the recorder
/// against production's own log-scale predictions), FOCEI and FOCE gradients against
/// reconverged FD, and an IOV twin (κ on `V`, one scale per occasion group).
#[test]
fn eta_dependent_obs_scale_under_ltbs_on_an_ode_is_analytic() {
    let src = |iov: bool| {
        format!(
            "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
             theta TVKA(1.2, 0.05, 20.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.06\n{}  \
             sigma ADD_LOG ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  \
             V = TVV * exp(ETA_V{})\n  KA = TVKA\n[structural_model]\n  \
             ode(obs_cmt=central, states=[depot, central])\n[odes]\n  d/dt(depot) = -KA * depot\n  \
             d/dt(central) = KA * depot - (CL/V) * central\n[scaling]\n  obs_scale = V\n\
             [error_model]\n  log(DV) ~ additive(ADD_LOG)\n{}",
            if iov { "  kappa KAPPA_V ~ 0.03\n" } else { "" },
            if iov { " + KAPPA_V" } else { "" },
            if iov {
                "[fit_options]\n  iov_column = OCC\n"
            } else {
                ""
            },
        )
    };
    let m = tight(parse_model_string(&src(false)).unwrap());
    assert!(m.log_transform, "LTBS");
    assert!(matches!(
        m.scaling,
        crate::types::ScalingSpec::ExpressionScale { .. }
    ));
    let s = two_dose_subject(&m);
    let eta = [0.08, -0.05];
    let theta = &m.default_params.theta;
    let rec = crate::sens::provider::subject_eta_grad(&m, &s, theta, &eta).expect("served");
    let prod = crate::pk::compute_predictions_with_tv(&m, &s, theta, &eta);
    for (j, (r, p)) in rec.iter().zip(&prod).enumerate() {
        assert!(
            (r.f - p).abs() < 1e-9 * (1.0 + p.abs()),
            "obs {j}: {} vs {p}",
            r.f
        );
    }
    for (interaction, name) in [(true, "FOCEI"), (false, "FOCE")] {
        let g = fd_oracle(&m, &s, interaction);
        eprintln!("obs_scale × LTBS (ODE) {name}: gradient vs reconverged FD {g:.2e}");
        // Measured 7.9e-9 (FOCEI) / 9.7e-9 (FOCE) on Linux; ~10× headroom.
        assert!(g < 1e-7, "{name}: {g:.2e}");
    }
    let m = tight(parse_model_string(&src(true)).unwrap());
    let s = super::tests::iov_subject(&m);
    for (interaction, name) in [(true, "FOCEI"), (false, "FOCE")] {
        let g = super::tests::iov_parity(&m, &s, interaction);
        eprintln!("obs_scale × LTBS (ODE, IOV) {name}: gradient vs reconverged FD {g:.2e}");
        // Measured 1.9e-7 on Linux, both; ~10× headroom.
        assert!(g < 2e-6, "IOV {name}: {g:.2e}");
    }
}

/// A 9-occasion IOV subject: 3 η + 3 κ × 9 occasions = 30 stacked random effects, past the
/// 24-wide bucket that once capped the adjoint.
fn wide_iov_fixture() -> (CompiledModel, Subject) {
    let src = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  theta TVKA(1.2, 0.05, 20.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.06\n  omega ETA_KA ~ 0.1\n  kappa K_CL ~ 0.03\n  kappa K_V ~ 0.02\n  kappa K_KA ~ 0.04\n  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + K_CL)\n  V = TVV * exp(ETA_V + K_V)\n  KA = TVKA * exp(ETA_KA + K_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  iov_column = OCC\n";
    let m = parse_model_string(src).unwrap();
    let n_occ = 9u32;
    let mut doses = Vec::new();
    let mut times = Vec::new();
    let mut occ = Vec::new();
    let mut docc = Vec::new();
    for o in 0..n_occ {
        let t0 = 24.0 * o as f64;
        doses.push(DoseEvent::new(t0, 100.0, 1, 0.0, false, 0.0));
        docc.push(o + 1);
        for dt in [1.0, 4.0, 12.0] {
            times.push(t0 + dt);
            occ.push(o + 1);
        }
    }
    let mut s = subject_shell(doses, &times);
    s.occasions = occ;
    s.dose_occasions = docc;
    let kappas: Vec<Vec<f64>> = (0..n_occ)
        .map(|o| vec![0.05 * ((o as f64).sin()), -0.04, 0.03])
        .collect();
    let preds = crate::pk::predict_iov(
        &m,
        &s,
        &m.default_params.theta,
        &[0.1, -0.05, 0.08],
        &kappas,
    );
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.1 * (1.3 * j as f64).sin()))
        .collect();
    (m, s)
}

/// A subject wider than 24 stacked random effects is served and its stacked
/// η-Jacobian matches FD of production's IOV predictor. On a 20-subject population of these,
/// the FOCEI fit took 2739 s on the reconverged-FD route it used to fall to and 41.5 s on the
/// exact gradient, at the identical OFV (−1177.955802, Linux, 2 cores).
#[test]
fn a_subject_past_24_stacked_random_effects_is_served() {
    let (m, s) = wide_iov_fixture();
    let stacked: Vec<f64> = (0..30).map(|k| 0.02 * ((k as f64) * 0.7).sin()).collect();
    let gap = iov_jacobian_gap(&m, &s, &stacked, "30 stacked");
    eprintln!("30 stacked: η-Jacobian vs FD {gap:.2e}");
    // Measured 5.7e-12 on Linux; ~10× headroom.
    assert!(gap < 6e-11, "{gap:.2e}");
}

/// The same subject's FOCEI gradient against reconverged FD (77 s in a debug build).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_subject_past_24_stacked_random_effects_matches_reconverged_fd() {
    let (m, s) = wide_iov_fixture();
    let g = super::tests::iov_parity(&m, &s, true);
    eprintln!("30 stacked: FOCEI gradient vs reconverged FD {g:.2e}");
    // Measured 3.4e-7 on Linux; ~10× headroom.
    assert!(g < 3.4e-6, "{g:.2e}");
}

/// The same subject's analytic R-matrix against the reconverged-FD Hessian of its gradient.
/// Measured 3.04e-11 on Linux; ~10× headroom.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_subject_past_24_stacked_random_effects_has_an_exact_r_matrix() {
    let (m, s) = wide_iov_fixture();
    let (h, _) = hessian_gaps(&m, &s, true);
    eprintln!("30 stacked: FOCEI R-matrix vs reconverged FD {h:.2e}");
    assert!(h < 3e-10, "{h:.2e}");
}

/// A `TAD`-reading RHS under steady-state dosing (#2093): served now that the dual SS run-in
/// replays production's per-window clocks (`ss_run_in_params`: local time from 0, `TAD` from
/// the pulse, an infusion's quiet window at `TAD = τ + T_inf`, `TAFD` `NaN`). Pinned on the
/// values against production's predictor, the gradient against reconverged FD, for an SS bolus
/// and an SS infusion (whose quiet window is where the anchors differ); the same
/// RHS reading `TIME` is served too.
#[test]
fn tad_reading_rhs_under_steady_state_is_analytic() {
    let src = |clock: &str| {
        ODE_LAG
            .replace("  LAGTIME = TVLAG * exp(ETA_LAG)\n", "")
            .replace("theta TVLAG(0.4, 0.01, 5.0)", "theta TVK(0.5, 0.01, 5.0)")
            .replace("omega ETA_LAG ~ 0.1", "omega ETA_V ~ 0.1")
            .replace("V = TVV", "V = TVV * exp(ETA_V)\n  KT = TVK")
            .replace(
                "d/dt(central) = KA * depot - (CL/V) * central",
                &format!(
                    "d/dt(central) = KA * depot - (CL/V) * central * (1 + KT * exp(-0.3 * {clock}))"
                ),
            )
    };
    for (label, doses) in [
        (
            "SS bolus",
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ],
        ),
        (
            "SS infusion",
            vec![
                DoseEvent::new(0.0, 100.0, 2, 50.0, true, 12.0),
                DoseEvent::new(12.0, 100.0, 2, 50.0, false, 0.0),
            ],
        ),
    ] {
        let m = tight(parse_model_string(&src("TAD")).unwrap());
        let s = subject(
            &m,
            doses.clone(),
            &[1.0, 3.0, 6.0, 11.5, 13.0, 15.0, 20.0, 24.0],
            &[0.1, -0.05],
        );
        let eta = [0.08, -0.04];
        let theta = &m.default_params.theta;
        let rec = crate::sens::provider::subject_eta_grad(&m, &s, theta, &eta)
            .unwrap_or_else(|| panic!("{label}: TAD under SS must be served"));
        let prod = crate::pk::compute_predictions_with_tv(&m, &s, theta, &eta);
        let mut worst = 0.0f64;
        for (r, p) in rec.iter().zip(&prod) {
            assert!(r.f.is_finite() && p.is_finite());
            worst = worst.max((r.f - p).abs() / (1.0 + p.abs()));
        }
        eprintln!("{label}: recorder vs production values {worst:.2e}");
        // Measured 0 (bit-identical) on Linux.
        assert!(worst < 1e-12, "{label}: values {worst:.2e}");
        let g = fd_oracle(&m, &s, true);
        eprintln!("{label}: FOCEI gradient vs reconverged FD {g:.2e}");
        // Measured 1.2e-6 (bolus) / 1.6e-7 (infusion) on Linux; ~10× headroom on the worse.
        assert!(g < 1.2e-5, "{label}: gradient {g:.2e}");
        // `TIME` under SS is served too (`ss_into_a_clock_reading_rhs_is_served_and_matches_fd`).
        let mt = tight(parse_model_string(&src("TIME")).unwrap());
        let st = subject(
            &mt,
            doses,
            &[1.0, 3.0, 6.0, 11.5, 13.0, 15.0, 20.0, 24.0],
            &[0.1, -0.05],
        );
        assert!(
            crate::sens::provider::subject_eta_grad(&mt, &st, &mt.default_params.theta, &eta)
                .is_some(),
            "{label}: TIME under SS must be served"
        );
    }
}

/// [`super::tests::templated_twin`] on a joint PK-TTE subject: its hazard term's plan, too, is
/// served from another subject's template, bit for bit.
#[cfg(feature = "survival")]
#[test]
fn a_tte_subject_served_from_anothers_template_matches_its_fresh_plan_bitwise() {
    let m = tight(parse_model_string(JOINT_PKTTE_ODE).unwrap());
    let times = [1.1, 4.2, 8.3, 12.4];
    let mut a = subject_shell(vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)], &times);
    a.obs_cmts = vec![2; times.len()];
    a.observations = vec![3.1, 2.6, 1.6, 1.1];
    a.obs_records = vec![crate::types::ObsRecord::Event {
        time: 10.5,
        event_type: crate::types::EventType::Exact,
        entry_time: 0.0,
        cmt: 3,
    }];
    let mut b = super::tests::other_data(&a);
    b.obs_records = vec![crate::types::ObsRecord::Event {
        time: 11.5,
        event_type: crate::types::EventType::Exact,
        entry_time: 0.0,
        cmt: 3,
    }];
    let (n, k) = super::tests::templated_twin(&m, &a, &b);
    assert!(k > 0 && k == n, "PK-TTE: {k} of {n} plans from a template");
}

// ---- steady state over a moving infusion window --------------------------------------

/// [`MODELED_DUR_ODE`] / [`MODELED_RATE_ODE`], or a rate-defined infusion under an estimated
/// `F1`, varied one input at a time: `mm` a saturable disposition (the steady state's pulse
/// train instead of its affine fixed point), `tad` a `TAD`-reading right-hand side, `lag` an
/// estimated `ALAG1`, `cf` the closed form (`one_cpt_iv`) in place of the ODE.
fn ss_window_model(case: &str, mm: bool, tad: bool, lag: bool, cf: bool) -> CompiledModel {
    let mut src = match case {
        "D1" => MODELED_DUR_ODE.to_string(),
        "R1" => MODELED_RATE_ODE.to_string(),
        _ => MODELED_DUR_ODE
            .replace("theta TVD1(5.0, 0.1, 24.0)", "theta TVF1(0.6, 0.05, 1.0)")
            .replace("omega ETA_D1", "omega ETA_F1")
            .replace("D1 = TVD1 * exp(ETA_D1)", "F1 = TVF1 * exp(ETA_F1)"),
    };
    let mut rhs = if mm {
        "-CL * central / (V + 0.02 * central)"
    } else {
        "-CL/V * central"
    }
    .to_string();
    if tad {
        rhs = format!("({rhs}) * (1 + 0.05 * TAD)");
    }
    src = src.replace(
        "d/dt(central) = -CL/V * central",
        &format!("d/dt(central) = {rhs}"),
    );
    if lag {
        src = src
            .replace(
                "  sigma PROP",
                "  theta TVLAG(0.7, 0.01, 5.0)\n  omega ETA_LAG ~ 0.02\n  sigma PROP",
            )
            .replace(
                "  V  = TVV\n",
                "  V  = TVV\n  ALAG1 = TVLAG * exp(ETA_LAG)\n",
            );
    }
    if !cf {
        return tight(parse_model_string(&src).unwrap());
    }
    let args = match (lag, case) {
        (true, "F1") => ", lagtime=ALAG1, f=F1",
        (true, _) => ", lagtime=ALAG1",
        (false, "F1") => ", f=F1",
        _ => "",
    };
    let src = src.replace(
        "[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -CL/V * central\n[scaling]\n  y = central / V\n",
        &format!("[structural_model]\n  pk one_cpt_iv(cl=CL, v=V{args})\n"),
    );
    let m = parse_model_string(&src).unwrap();
    assert!(m.ode_spec.is_none(), "closed form");
    m
}

/// A dose of the case's kind at 0 (`SS=1` with interval `ii` when `ss`), a second at 12 on the
/// first's residual, observations on both sides of both window ends.
fn ss_window_subject(m: &CompiledModel, case: &str, ss: bool, ii: f64) -> Subject {
    let ii = if ss { ii } else { 0.0 };
    let dose = |t: f64, ss: bool| match case {
        "D1" => DoseEvent::modeled(t, 100.0, 1, ss, ii, RateMode::ModeledDuration),
        "R1" => DoseEvent::modeled(t, 100.0, 1, ss, ii, RateMode::ModeledRate),
        _ => DoseEvent::new(t, 100.0, 1, 20.0, ss, ii),
    };
    let eta = [0.1, -0.08, 0.05];
    subject(
        m,
        vec![dose(0.0, ss), dose(12.0, false)],
        &[1.0, 3.0, 4.6, 6.0, 9.0, 11.9, 13.0, 15.0, 17.5, 20.0, 24.0],
        &eta[..m.n_eta],
    )
}

/// `(∂f/∂η gap, FOCEI gradient gap)` of one cell, every η live ([`jacobian_gap`]).
fn ss_window_gaps(
    case: &str,
    mm: bool,
    tad: bool,
    lag: bool,
    cf: bool,
    ss: bool,
    ii: f64,
) -> (f64, f64) {
    let m = ss_window_model(case, mm, tad, lag, cf);
    // The overlapping saturable pulse train (`mm`, `II < T_inf`) is ~50 cycles of adaptive
    // solves, and the reconverged-FD oracle differences them at `h = 1e-3`: at the models'
    // `(1e-10, 1e-12)` the solver's step-selection noise is the floor — measured on `R1`
    // `II = 4`, FOCEI gradient gap 1.285e-6 (`D1`: 6.9e-8), every coordinate 1e-7–1e-6 off
    // while ∂f/∂η at fixed η is 4.5e-10 — and it falls with the solver tolerance, not the
    // gradient: 2.4e-9 (`R1`) and 4.2e-9 (`D1`) at `(1e-12, 1e-14)`, 4.1e-9 at `(1e-13, 1e-15)`.
    let m = if mm && ss && ii < 12.0 {
        with_tol(m, 1e-12, 1e-14)
    } else {
        m
    };
    let s = ss_window_subject(&m, case, ss, ii);
    let label = format!("{case} cf={cf} mm={mm} tad={tad} lag={lag} ss={ss} ii={ii}");
    let jac = jacobian_gap(&m, &s, &probe_eta(m.n_eta), &label);
    let grad = parity(&m, &s);
    eprintln!("{label}: ∂f/∂η vs FD {jac:.2e}; FOCEI gradient vs reconverged FD {grad:.2e}");
    (jac, grad)
}

/// An `SS=1` infusion whose window moves with the parameters — a modeled `D1` (`RATE=-2`), a
/// modeled `R1` (`RATE=-1`), a rate-defined infusion under an estimated `F1` — on the ODE event
/// walk, the second dose landing on the steady-state residual. One input varied at a time:
/// the non-SS control; SS on a linear disposition (the affine fixed point), under a
/// `TAD`-reading RHS, on a saturable one (the pulse train) and both; SS under an estimated lag
/// (the pre-arrival seed at phase `II − lag`, linear and saturable); and `II = 4` and `2`,
/// where the windows overlap the next pulses (`T_inf > II`: `D1`/`R1` at 4, all three at 2),
/// linear, saturable (the pulse train) and lagged. The overlap used to equilibrate to a zero
/// state, value and derivative alike — consistent, and not NONMEM's steady state.
///
/// Before the fix the run-in named the window's jet-carrying length as its segments' spans
/// and anchored the quiet window's `TAD` on it, on top of the crossing that carries the same
/// jet — the window's derivative twice on a linear disposition (whose exact segment takes a
/// span's jet), a wrong `TAD` jet on any: ∂f/∂η 1.0 / 1.1 / 0.35 off FD (D1 / R1 / F1, linear),
/// 0.10–0.13 under `TAD`. The lagged seed extended its quiet leg by the window's jet as well:
/// 0.42–0.77. Measured now on Linux: ∂f/∂η ≤ 1.5e-10, gradient ≤ 7.4e-9 over the cells.
/// (`R1`'s saturable lag cell is left out: its FOCEI EBE has two modes 0.07 apart in `η_R1`,
/// between which the reconverged oracle jumps under a 1e-3 step; its ∂f/∂η at the moved
/// parameters' EBE is 5.6e-11 and its values match production to 8e-13.)
#[test]
fn steady_state_over_a_moving_infusion_window_on_the_ode_walk_matches_reconverged_fd() {
    for case in ["D1", "R1", "F1"] {
        for (mm, tad, lag, ss, ii) in [
            (false, false, false, false, 12.0),
            (false, false, false, true, 12.0),
            (false, true, false, true, 12.0),
            (true, false, false, true, 12.0),
            (true, true, false, true, 12.0),
            (false, false, true, true, 12.0),
            (true, false, true, true, 12.0),
            (false, false, false, true, 4.0),
            (false, false, false, true, 2.0),
            (true, false, false, true, 4.0),
            (false, false, true, true, 4.0),
        ] {
            if case == "R1" && mm && lag {
                continue;
            }
            let (jac, grad) = ss_window_gaps(case, mm, tad, lag, false, ss, ii);
            assert!(
                jac < 2e-9,
                "{case} mm={mm} tad={tad} lag={lag} ss={ss} ii={ii}: ∂f/∂η {jac:.2e}"
            );
            assert!(
                grad < 1e-7,
                "{case} mm={mm} tad={tad} lag={lag} ss={ss} ii={ii}: gradient {grad:.2e}"
            );
        }
    }
}

/// The same windows on the closed-form event walk, with and without `SS=1`. A recording
/// keeps the walk's break order, which the moving window ends decide as much as a lagged
/// arrival does: before every adjacent pair of breaks was guarded, a plan recorded with the
/// window end on one side of an observation was replayed with it on the other — the
/// predictions of a different event order — so the FOCEI gradient was 0.29 off reconverged
/// FD (`D1`, no SS), while ∂f/∂η at a probe point recorded fresh was exact. And the steady
/// state's cycle dose under `F` dropped the `F`-scaled window the value walk keeps (#419):
/// ∂f/∂η 0.35 off (`F1`, SS).
///
/// The same walk under an estimated lag — the record-seeded state at phase `II − ALAG` and the
/// previous pulse's residual window after it, both moving with the lag — and with the pulses
/// overlapping (`II = 4` and `2`: `D1 = 5`, `R1`'s `100/R1 = 5`, `F1`'s 3 h window), whose
/// steady state replays the always-running pulses as one more window and carries them past the
/// record. Before: the lag declined to FD (`ss_lagtime_walk_unsupported`), and the overlap
/// equilibrated to a zero state, its gradient the derivative of a function the value path
/// (superposition) does not score. Measured now on Linux (`D1`, `R1`): ∂f/∂η ≤ 1.6e-11,
/// gradient ≤ 2.9e-9.
#[test]
fn steady_state_over_a_moving_infusion_window_on_the_closed_form_walk_matches_reconverged_fd() {
    for case in ["D1", "R1", "F1"] {
        for (lag, ss, ii) in [
            (false, false, 12.0),
            (false, true, 12.0),
            (false, true, 4.0),
            (false, true, 2.0),
            (true, true, 12.0),
            (true, true, 4.0),
        ] {
            // `R1` under a lag: its plan is re-recorded at every inner-search iterate (measured:
            // the reconverged-FD oracle spends minutes per coordinate in a debug build, its first
            // coordinate within 3.3e-9), which is too slow for Tier 1.
            if case == "R1" && lag {
                continue;
            }
            let (jac, grad) = ss_window_gaps(case, false, false, lag, true, ss, ii);
            assert!(
                jac < 5e-11,
                "{case} lag={lag} ss={ss} ii={ii}: ∂f/∂η {jac:.2e}"
            );
            assert!(
                grad < 3e-8,
                "{case} lag={lag} ss={ss} ii={ii}: gradient {grad:.2e}"
            );
        }
    }
}

/// The `nonmem_anchor` shapes whose steady state `periodic_ss_fixed_point_g` solves, anchored
/// there on values only, here on derivatives, on the closed form and its ODE twin:
/// `evid_reset_lag` ID 13 — an `SS=1` depot dose at 10 whose lag `≈ 11 < II` puts its pulse at
/// 21, after an `EVID=3` reset at 15 that cancels it — and the same subject without the reset,
/// so the steady state and the lagged arrival reach the observations, η on the lag; and
/// `ss_infusion_stop` ID 2, a depot infusion still running when an `SS=1` record lands at 10,
/// η on `CL` and `KA`. Observations off the model at a reference η.
///
/// The cancelled subject predicts exactly zero, so its exact Jacobian must be exactly zero (a
/// dual leaking the cancelled pulse would not be). Measured on Linux, the others: ∂f/∂η ≤
/// 3.5e-11, FOCEI gradient ≤ 2.5e-8 (ID 2 on the ODE twin; ≤ 6.9e-9 elsewhere).
#[test]
fn anchor_steady_state_shapes_match_reconverged_fd() {
    let load = |file: &str, id: &str| {
        crate::read_nonmem_csv(std::path::Path::new(file), None, None)
            .expect("anchor dataset")
            .subjects
            .into_iter()
            .find(|s| s.id == id)
            .expect("anchor subject")
    };
    let model = |ode: bool, lag: bool| {
        let (eta2, ka) = if lag {
            ("ETA_LAG ~ 0.02", "TVKA")
        } else {
            ("ETA_KA ~ 0.09", "TVKA * exp(ETA_KA)")
        };
        let (ind2, structure) = match (ode, lag) {
            (true, true) => ("ALAG1 = LAGC * exp(ETA_LAG)", ""),
            (false, true) => ("LAG = LAGC * exp(ETA_LAG)", ", lagtime=LAG"),
            _ => ("", ""),
        };
        let structure = if ode {
            "ode(obs_cmt=central, states=[depot, central])\n[odes]\n  d/dt(depot) = -KA*depot\n  \
             d/dt(central) = KA*depot - CL/V*central\n[scaling]\n  y = central / V"
                .to_string()
        } else {
            format!("pk one_cpt_oral(cl=CL, v=V, ka=KA{structure})")
        };
        let src = format!(
            "[parameters]\n  theta TVCL(2.0, 0.01, 50.0)\n  theta TVV(20.0, 0.5, 200.0)\n  \
             theta TVKA(0.15, 0.005, 20.0)\n  omega ETA_CL ~ 0.09\n  omega {eta2}\n  \
             sigma ADD_ERR ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  \
             V = TVV\n  KA = {ka}\n  {ind2}\n[structural_model]\n  {structure}\n\
             [error_model]\n  DV ~ additive(ADD_ERR)\n"
        );
        let m = parse_model_string(&src).unwrap();
        assert_eq!(m.ode_spec.is_some(), ode);
        if ode {
            tight(m)
        } else {
            m
        }
    };
    let observed = |m: &CompiledModel, s: &Subject| {
        let mut s = s.clone();
        let f =
            crate::pk::compute_predictions_with_tv(m, &s, &m.default_params.theta, &[0.1, -0.05]);
        s.observations = f
            .iter()
            .enumerate()
            .map(|(j, v)| v * (1.0 + 0.1 * (1.3 * j as f64).sin()))
            .collect();
        s
    };
    let reset = load("nonmem_anchor/evid_reset_lag.csv", "13");
    assert!(reset.has_resets() && reset.doses[0].ss);
    let mut no_reset = reset.clone();
    no_reset.reset_times.clear();
    no_reset.reset_covariates.clear();
    no_reset.reset_occasions.clear();
    no_reset.reset_period_times.clear();
    let stop = load("nonmem_anchor/ss_infusion_stop.csv", "2");
    assert!(stop.doses[0].is_infusion() && stop.doses[1].ss);
    for ode in [false, true] {
        let e = if ode { "ODE" } else { "closed form" };
        let (lagged, plain) = (model(ode, true), model(ode, false));
        let eta = probe_eta(2);
        let s = observed(&lagged, &reset);
        let f =
            crate::pk::compute_predictions_with_tv(&lagged, &s, &lagged.default_params.theta, &eta);
        let jac =
            exact_eta_jacobian(&lagged, &s, &lagged.default_params.theta, &eta).expect("served");
        assert!(
            f.iter().all(|v| *v == 0.0) && jac.iter().all(|v| *v == 0.0),
            "ID 13, {e}: the cancelled pulse leaks: f {f:?}, ∂f/∂η {jac}"
        );
        for (label, m, s) in [
            (
                format!("ID 13 without the reset, {e}"),
                &lagged,
                observed(&lagged, &no_reset),
            ),
            (format!("ID 2, {e}"), &plain, observed(&plain, &stop)),
        ] {
            let jac = jacobian_gap(m, &s, &eta, &label);
            let grad = parity(m, &s);
            eprintln!(
                "{label}: ∂f/∂η vs FD {jac:.2e}; FOCEI gradient vs reconverged FD {grad:.2e}"
            );
            assert!(jac < 4e-10, "{label}: ∂f/∂η {jac:.2e}");
            assert!(grad < 3e-7, "{label}: gradient {grad:.2e}");
        }
    }
}

// ---- SS into an absorption input with a lag, and into a zero-order window (#2786) --------
//
// A zero-order SS dose is walked as the duration-modeled SS infusion it is lowered to (the
// moving window end rides `dur`'s jet; the lag is `ALAG{c}`), a density SS dose through the
// periodic forcing whose record state is the fixed point of the cycle its own pulse arrives
// `lag` into. Every fixture dose is followed by an ordinary dose on residual drug; the lags
// sit below `II` and at / above it (the clamp), the windows below, past and above `II`. Each
// is checked against the reconverged-FD objective gradient and the Richardson `∂f/∂η`, served
// by the adjoint (no FD route) and with no plan failure.

/// A one-compartment model with `kernel` (`zero_order` | `first_order`) absorption and its lag:
/// `lag` is `"alag"` (compartment `ALAG1`), `"route"` (`lag=LAG` on the kernel) or `"none"`.
fn ss_lag_model(kernel: &str, lag: &str, lag_value: f64, dur_or_ka: f64) -> String {
    let (arg, extra) = if kernel == "zero_order" {
        (
            "dur=DUR",
            format!("  theta TVDUR({dur_or_ka}, 0.1, 40.0)\n"),
        )
    } else {
        ("ka=KA", format!("  theta TVKA({dur_or_ka}, 0.01, 20.0)\n"))
    };
    let (indiv, call) = match lag {
        "alag" => ("  ALAG1 = TVLAG\n", format!("{kernel}({arg})")),
        "route" => ("  LAG = TVLAG\n", format!("{kernel}({arg}, lag=LAG)")),
        _ => ("", format!("{kernel}({arg})")),
    };
    let var = if kernel == "zero_order" {
        "  DUR = TVDUR * exp(ETA_P)\n"
    } else {
        "  KA = TVKA * exp(ETA_P)\n"
    };
    format!(
        "[parameters]\n  theta TVCL(1.0, 0.01, 50.0)\n  theta TVV(20.0, 0.5, 200.0)\n{extra}  \
         theta TVLAG({lag_value}, 0.0, 30.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_P ~ 0.04\n  \
         sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  \
         V  = TVV\n{var}{indiv}[structural_model]\n  ode(obs_cmt=central, states=[central])\n\
         [odes]\n  d/dt(central) = {call} - CL/V*central\n[scaling]\n  y = central / V\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n"
    )
}

/// `(label, model source, live θ)`: θ order is CL, V, DUR|KA, LAG.
fn ss_lag_cases() -> Vec<(String, String, Vec<usize>)> {
    let mut v = Vec::new();
    for (label, kernel, lag, lag_value, p) in [
        ("ZO ALAG1 2 (dur 3)", "zero_order", "alag", 2.0, 3.0),
        ("ZO route lag 2 (dur 3)", "zero_order", "route", 2.0, 3.0),
        (
            "ZO ALAG1 2 + dur 11 past II",
            "zero_order",
            "alag",
            2.0,
            11.0,
        ),
        ("ZO ALAG1 14 >= II", "zero_order", "alag", 14.0, 3.0),
        ("ZO ALAG1 12 = II, dur 5", "zero_order", "alag", 12.0, 5.0),
        ("ZO no lag, dur 15 > II", "zero_order", "none", 0.0, 15.0),
        ("FO ALAG1 2", "first_order", "alag", 2.0, 0.4),
        ("FO route lag 2", "first_order", "route", 2.0, 0.4),
        ("FO ALAG1 14 >= II", "first_order", "alag", 14.0, 0.4),
        ("FO route lag 14 >= II", "first_order", "route", 14.0, 0.4),
    ] {
        let live = if lag == "none" {
            vec![0, 2]
        } else {
            vec![0, 2, 3]
        };
        v.push((
            label.to_string(),
            ss_lag_model(kernel, lag, lag_value, p),
            live,
        ));
    }
    v
}

#[test]
fn ss_into_absorption_with_a_lag_matches_reconverged_fd() {
    // >= 1 h from every window edge and arrival: an observation on an edge sits on a kink of
    // the prediction, where central FD and the one-sided analytic derivative differ.
    let times = [0.4, 4.0, 6.5, 7.5, 10.5, 16.0, 19.0, 22.5, 28.0, 31.0];
    // The `lag >= II` rows keep the observations out of the whole range the estimated window
    // end can sweep (`dur = 3 e^eta`, eta in +-0.5): an EBE pinned on a window-edge kink has no
    // gradient to match (`ss_modeled_duration_lag_ge_ii_ebe_on_window_edge_kink_is_not_differentiable`).
    let clamp_times = [
        0.4, 6.0, 7.5, 10.5, 13.0, 14.6, 21.0, 23.0, 25.0, 33.0, 36.0,
    ];
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let mut worst = (0.0f64, 0.0f64);
    let mut bad = Vec::new();
    for (label, src, live) in ss_lag_cases() {
        let times: &[f64] = if label.contains(">= II") {
            &clamp_times
        } else {
            &times
        };
        let m = tight(parse_model_string(&src).unwrap());
        let s = subject(&m, doses.clone(), times, &[0.1, -0.05]);
        assert!(s.has_periodic_ss_dose());
        let (jac, grad) = measure(&m, &s, &live, &label);
        assert!(
            crate::parser::model_parser::rhs_regcode::kernel_jit::plan_failures(
                &m,
                std::slice::from_ref(&s)
            )
            .is_none(),
            "{label}: a plan failed"
        );
        worst = (worst.0.max(jac), worst.1.max(grad));
        // Measured on Linux (reltol 1e-10): worst ∂f/∂η 8.6e-8 (first_order at lag >= II, whose
        // periodic sum the value path truncates at 1e-10 relative), worst gradient 2.8e-6 (a
        // 15 h zero-order window, no lag); the zero-order rows with a lag 1.4e-10 / 4.6e-7.
        // Bounds ~10x the worst.
        if jac >= 1e-6 || grad >= 3e-5 {
            bad.push(format!("{label}: ∂f/∂η {jac:.2e}, gradient {grad:.2e}"));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
    eprintln!(
        "SS absorption lag: worst ∂f/∂η {:.2e}, gradient {:.2e}",
        worst.0, worst.1
    );
}

/// Control for the zero-order SS case: the same configurations written as a user-coded
/// `RATE=-2` modeled-duration SS infusion (`D1`), the path the zero-order dose is lowered to.
/// The `lag >= II` rows with an estimated duration on a dense grid are in
/// `ss_modeled_duration_lag_ge_ii_dense_grid_matches_reconverged_fd`; the θ gradient that once
/// read as 0.3 off there is an EBE pinned on a window-edge kink, not a derivative defect (see
/// `..._ebe_on_window_edge_kink_is_not_differentiable`).
#[test]
fn ss_modeled_duration_infusion_with_a_lag_matches_reconverged_fd() {
    let times = [0.4, 4.0, 6.5, 7.5, 10.5, 16.0, 19.0, 22.5, 28.0, 31.0];
    let mut bad = Vec::new();
    for (dur, lag) in [
        (3.0, 2.0),
        (11.0, 0.0),
        (5.0, 9.0),
        (11.0, 2.0),
        (15.0, 0.0),
    ] {
        let src = format!(
            "[parameters]\n  theta TVCL(1.0, 0.01, 50.0)\n  theta TVV(20.0, 0.5, 200.0)\n  \
            theta TVDUR({dur}, 0.1, 40.0)\n  theta TVLAG({lag}, 0.0, 30.0)\n  omega ETA_CL ~ 0.09\n  \
            omega ETA_P ~ 0.04\n  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  \
            CL = TVCL * exp(ETA_CL)\n  V  = TVV\n  D1 = TVDUR * exp(ETA_P)\n  ALAG1 = TVLAG\n\
            [structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  \
            d/dt(central) = -CL/V*central\n[scaling]\n  y = central / V\n[error_model]\n  \
            DV ~ proportional(PROP_ERR)\n"
        );
        let m = tight(parse_model_string(&src).unwrap());
        let doses = vec![
            DoseEvent::modeled(0.0, 100.0, 1, true, 12.0, RateMode::ModeledDuration),
            DoseEvent::modeled(12.0, 100.0, 1, false, 0.0, RateMode::ModeledDuration),
        ];
        let s = subject(&m, doses, &times, &[0.1, -0.05]);
        let label = format!("modeled D1 SS dur {dur} lag {lag}");
        let (jac, grad) = measure(&m, &s, &[0, 2, 3], &label);
        // Measured on Linux: worst ∂f/∂η 1.6e-11, worst gradient 3.8e-9; ~10x.
        if jac > 2e-10 || grad > 4e-8 {
            bad.push(format!("{label}: {jac:.2e} {grad:.2e}"));
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}

/// A `lag >= II` modeled-duration (`D1`) SS infusion, estimated window end `dur e^eta`
/// (`TVDUR 3`, `TVLAG 14`, `II 12`).
fn ss_d1_lag_ge_ii_model(fixed_duration: bool) -> CompiledModel {
    let d1 = if fixed_duration {
        "TVDUR"
    } else {
        "TVDUR * exp(ETA_P)"
    };
    let src = format!(
        "[parameters]\n  theta TVCL(1.0, 0.01, 50.0)\n  theta TVV(20.0, 0.5, 200.0)\n  \
        theta TVDUR(3.0, 0.1, 40.0)\n  theta TVLAG(14.0, 0.0, 30.0)\n  omega ETA_CL ~ 0.09\n  \
        omega ETA_P ~ 0.04\n  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  \
        CL = TVCL * exp(ETA_CL)\n  V  = TVV\n  D1 = {d1}\n  ALAG1 = TVLAG\n\
        [structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  \
        d/dt(central) = -CL/V*central\n[scaling]\n  y = central / V\n[error_model]\n  \
        DV ~ proportional(PROP_ERR)\n"
    );
    tight(parse_model_string(&src).unwrap())
}

const SS_D1_DENSE_GRID: [f64; 10] = [0.4, 4.0, 6.5, 7.5, 10.5, 16.0, 19.0, 22.5, 28.0, 31.0];

/// The subject of [`ss_d1_lag_ge_ii_model`] on the ten-observation grid, its DV generated at
/// `eta_ref`, with the EBE the oracle will find and the largest jump of `∂f/∂η_P` across it
/// (`> 0` only where an observation sits on a window edge, a kink of the prediction).
fn ss_d1_dense_subject(m: &CompiledModel, eta_ref: [f64; 2]) -> (Subject, Vec<f64>, f64) {
    let doses = vec![
        DoseEvent::modeled(0.0, 100.0, 1, true, 12.0, RateMode::ModeledDuration),
        DoseEvent::modeled(12.0, 100.0, 1, false, 0.0, RateMode::ModeledDuration),
    ];
    let s = subject(m, doses, &SS_D1_DENSE_GRID, &eta_ref);
    assert!(
        crate::parser::model_parser::rhs_regcode::kernel_jit::plan_failures(
            m,
            std::slice::from_ref(&s)
        )
        .is_none(),
        "a plan failed"
    );
    let params = moved_params(m);
    let eta = ebe_for(m, &s, &params, 1e-12, true).as_slice().to_vec();
    let slope = |e: f64| -> Vec<f64> {
        let h = 1e-6;
        let at =
            |x: f64| crate::pk::compute_predictions_with_tv(m, &s, &params.theta, &[eta[0], x]);
        at(e - h)
            .iter()
            .zip(at(e).iter())
            .map(|(a, b)| (b - a) / h)
            .collect()
    };
    let hh = 2e-5;
    let (left, right) = (slope(eta[1] - hh), slope(eta[1] + hh));
    let jump = left
        .iter()
        .zip(&right)
        .map(|(l, r)| {
            assert!(l.is_finite() && r.is_finite());
            (l - r).abs()
        })
        .fold(0.0, f64::max); // nan-ok: both slopes asserted finite in the map above
    (s, eta, jump)
}

/// The ten-observation `lag >= II` modeled-duration grid of the control test above, at data
/// whose EBE is a smooth interior optimum: the θ gradient matches reconverged FD to 1e-8, the
/// plan serving every subject. (A dense grid does not break the derivative: it makes an EBE
/// pinned on a window-edge kink likely, which no gradient matches; see the next test.) Fixed
/// rate and `lag < II` rows are the controls; the realised worst gap is 2.4e-9 (Linux), the
/// bound ~10x.
#[test]
fn ss_modeled_duration_lag_ge_ii_dense_grid_matches_reconverged_fd() {
    let mut bad = Vec::new();
    let mut worst = 0.0f64;
    for (fixed, eta_ref) in [
        (false, [0.1, 0.1]),
        (false, [0.2, 0.2]),
        (false, [-0.1, 0.2]),
        (false, [0.0, 0.3]),
        (false, [-0.3, -0.3]),
        (true, [0.1, -0.1]),
    ] {
        let m = ss_d1_lag_ge_ii_model(fixed);
        let (s, eta, jump) = ss_d1_dense_subject(&m, eta_ref);
        assert!(served(&m, &s));
        // The EBE must be off every kink for the oracle to be a derivative.
        assert!(
            jump < 1e-3,
            "fixed={fixed} {eta_ref:?}: EBE {eta:?} on a kink ({jump:.2e})"
        );
        let g = parity(&m, &s);
        eprintln!("fixed={fixed} ref {eta_ref:?} EBE {eta:?}: gradient gap {g:.2e}");
        assert!(g.is_finite());
        worst = worst.max(g);
        if g > 3e-8 {
            bad.push(format!("fixed={fixed} {eta_ref:?}: {g:.2e}"));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
    eprintln!("SS D1 lag>=II dense grid: worst gradient gap {worst:.2e}");
}

/// The case that read as a gradient defect (the control test above: the θ gradient 0.1-0.9 off
/// reconverged FD on this grid): the DV's data pull the EBE onto the kink where the estimated
/// window end `lag - II + dur e^eta` crosses the observation at `t = 4` (`dur e^eta = 25/7`,
/// so `eta_P = ln(25/21) = 0.17435`). The minimiser is pinned there for every nearby θ, `∂f/∂η`
/// jumps across it (by `R/V · dD/dη`, ~5.5 here), and FOCEI's `H` and the EBE's stationarity
/// are not defined at a kink: reconverged FD differences an objective whose EBE flips sides
/// (slopes of ±100), and the adjoint's one-sided derivative cannot match it. Every other EBE on
/// the same grid matches to 1e-8 (the test above). Pins the diagnosis, both halves: the EBE sits
/// on the kink, and the oracle is not a derivative there.
#[test]
fn ss_modeled_duration_lag_ge_ii_ebe_on_window_edge_kink_is_not_differentiable() {
    let m = ss_d1_lag_ge_ii_model(false);
    for eta_ref in [[0.1, -0.1], [0.0, 0.0]] {
        let (s, eta, jump) = ss_d1_dense_subject(&m, eta_ref);
        let kink = (25.0f64 / 21.0).ln();
        assert!(
            (eta[1] - kink).abs() < 1e-6,
            "{eta_ref:?}: EBE η_P {} is not on the kink {kink}",
            eta[1]
        );
        assert!(
            jump > 1.0,
            "{eta_ref:?}: no slope jump at the EBE ({jump:.2e})"
        );
        let g = parity(&m, &s);
        assert!(
            g > 1e-2,
            "{eta_ref:?}: the oracle matched on a kink ({g:.2e})"
        );
    }
}

// ---- per-CMT `obs_scale` on the ODE walks (#2786, universal-engine item 5) ----------------

/// A two-observed-compartment ODE with a per-CMT `obs_scale` — a constant on CMT 1 (the depot)
/// and the η-dependent individual parameter `V` on CMT 2 — and, optionally, LTBS and κ.
fn per_cmt_scale_src(ltbs: bool, iov: bool) -> String {
    format!(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
         theta TVKA(1.2, 0.05, 20.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.06\n{}  \
         sigma S ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  \
         V = TVV * exp(ETA_V{})\n  KA = TVKA\n[structural_model]\n  \
         ode(obs_cmt=central, states=[depot, central])\n[odes]\n  d/dt(depot) = -KA * depot\n  \
         d/dt(central) = KA * depot - (CL/V) * central\n[scaling]\n  obs_scale[CMT=1] = 5\n  \
         obs_scale[CMT=2] = V\n[error_model]\n  {}\n{}",
        if iov { "  kappa KAPPA_V ~ 0.03\n" } else { "" },
        if iov { " + KAPPA_V" } else { "" },
        if ltbs {
            "log(DV) ~ additive(S)"
        } else {
            "DV ~ proportional(S)"
        },
        if iov {
            "[fit_options]\n  iov_column = OCC\n"
        } else {
            ""
        },
    )
}

/// A per-CMT `obs_scale` on an ODE model (static and event-driven walk) used to decline the
/// whole subject to finite differences (`apply_expression_scale`'s `PerCmt` arm was `None`).
/// Served now: each observation divides by its own CMT's scale — a constant on one, an
/// η-dependent expression on the other — then the LTBS log, against the reconverged-FD objective
/// gradient and the Richardson `∂f/∂η` of the production predictor. The observations alternate
/// CMTs and CMT 2's scale moves with `η_V` (`measure` asserts every η column live), so a scale
/// applied to the wrong CMT fails the Jacobian.
/// Mutations: return `None` for `PerCmt` -> `served` fails; look the scale up by observation
/// index instead of its CMT -> the Jacobian gap is O(1); drop `PerCmt` from
/// `apply_output_transform`'s deferral -> LTBS logs before the divisor and the gap is O(1).
#[test]
fn a_per_cmt_obs_scale_on_an_ode_is_analytic() {
    for ltbs in [false, true] {
        let m = tight(parse_model_string(&per_cmt_scale_src(ltbs, false)).unwrap());
        assert!(matches!(m.scaling, crate::types::ScalingSpec::PerCmt(_)));
        let mut s = two_dose_subject(&m);
        s.obs_cmts = (0..s.obs_times.len()).map(|j| 1 + j % 2).collect();
        // With a covariate that varies by row the same subject takes the event walk.
        for tv in [false, true] {
            let mut s = s.clone();
            if tv {
                s.covariates.insert("WT".into(), 70.0);
                s.obs_covariates = vec![s.covariates.clone(); s.obs_times.len()];
                s.dose_covariates = vec![s.covariates.clone(); s.doses.len()];
            }
            let label = format!("per-CMT obs_scale, ltbs {ltbs}, tv {tv}");
            let (jac, grad) = measure(&m, &s, &[0, 1, 2], &label);
            assert!(jac < 1e-6 && grad < 3e-5, "{label}: {jac:.2e} {grad:.2e}");
        }
    }
}

/// The IOV event walk's per-(occasion group, CMT) scale: κ on `V`, so CMT 2's scale differs per
/// occasion. Mutation: key the cache by CMT only (drop the group) -> the second occasion reads
/// the first's scale and the Jacobian gap is O(1).
#[test]
fn a_per_cmt_obs_scale_under_iov_is_analytic() {
    let m = tight(parse_model_string(&per_cmt_scale_src(false, true)).unwrap());
    let times = [1.0, 4.0, 10.0, 13.0, 16.0, 22.0];
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let mut s = subject_shell(doses, &times);
    s.occasions = vec![1, 1, 1, 2, 2, 2];
    s.dose_occasions = vec![1, 2];
    s.obs_cmts = vec![1, 2, 1, 2, 1, 2];
    let kappas = vec![vec![0.1], vec![-0.08]];
    let preds = crate::pk::predict_iov(&m, &s, &m.default_params.theta, &[0.1, -0.05], &kappas);
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.12 * (1.7 * j as f64).sin()))
        .collect();
    assert!(
        served(&m, &s),
        "the adjoint must serve a per-CMT scale under IOV"
    );
    let jac = iov_jacobian_gap(&m, &s, &[0.06, -0.04, 0.05, -0.03], "per-CMT IOV");
    let grad = fd_oracle(&m, &s, true);
    eprintln!("per-CMT obs_scale × IOV: ∂f/∂[η, κ] {jac:.2e}; gradient {grad:.2e}");
    assert!(jac < 8e-11, "∂f/∂[η, κ]: {jac:.2e}");
    assert!(grad < 4.5e-5, "gradient: {grad:.2e}");
}

// ---- variant churn of the exact-value guards (#2786, universal-engine item 6) -------------

/// How many variants a subject's plan holds after it is served at `points` `(θ scale, η)` —
/// 1 when one recording serves them all, one per point when a guard on a continuous value
/// fails at every other point and the subject is recorded again each time.
fn variants_over(m: &CompiledModel, s: &Subject, points: &[(f64, Vec<f64>)]) -> usize {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    let n = m.n_eta;
    for (scale, eta) in points {
        let theta: Vec<f64> = m.default_params.theta.iter().map(|t| t * scale).collect();
        let layout = terms::layout1(n);
        assert!(
            plan_forward(m, s, None, &theta, eta, layout).is_some(),
            "the subject records and serves at θ×{scale}, η {eta:?}"
        );
    }
    let key = kj::SubjectKey::new(m, s, n, m.default_params.theta.len()).with_layout(None);
    match kj::lookup(&key) {
        kj::Lookup::Found(ks) => ks.len(),
        _ => panic!("the subject has no plan"),
    }
}

fn churn_points() -> Vec<(f64, Vec<f64>)> {
    (0..6)
        .map(|k| {
            let x = k as f64;
            (1.0 + 0.04 * x, vec![0.05 * x - 0.1, 0.03 - 0.02 * x])
        })
        .collect()
}

const CF_SS_LAG: &str = r#"
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

/// A closed-form SS dose under an estimated lag, a later dose on residual drug: `cmt` 1 a bolus,
/// 2 an infusion at `rate`. The variants a plan holds after `churn_points`.
fn cf_ss_lag_variants(cmt: usize, rate: f64) -> usize {
    let m = parse_model_string(CF_SS_LAG).unwrap();
    let s = subject(
        &m,
        vec![
            DoseEvent::new(0.0, 100.0, cmt, rate, true, 12.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &[0.1, 1.0, 2.0, 4.0, 8.0, 13.0, 14.0, 18.0, 24.0],
        &[0.1, -0.05],
    );
    variants_over(&m, &s, &churn_points())
}

/// Over six distinct (theta, eta) points a subject keeps at most `MAX_CHURN` plans: a guard on
/// the exact value of a continuous quantity fails at every other point and records the subject
/// again on each evaluation (six variants here). A plan holds while its *decisions* do.
const MAX_CHURN: usize = 2;

/// A closed-form SS bolus under an estimated lag stays one plan across a fit. A control for the
/// bolus side of the seed (no exact-value guard on the lag or on `u_ss`); measured, it does NOT
/// die when `periodic_ss_fixed_point_g`'s self-check is mutated back to `br_value` reads (this
/// fixture's seed does not take that call) -- that site is pinned by
/// `the_ss_affinity_self_check_holds_at_another_point` (`kernel_jit_plan_tests.rs`).
#[test]
fn a_closed_form_ss_bolus_under_a_lag_is_one_plan_across_a_fit() {
    assert!(cf_ss_lag_variants(1, 0.0) <= MAX_CHURN);
}

/// `ss_state_at_phase_g` guarded `phase` and the lag by exact value: a lagged SS infusion seed
/// never replayed at another lag. Mutation: `phase.br_value()` / `lag.br_value()` back in the
/// non-overlapping arm -> 6 variants.
#[test]
fn a_lagged_ss_infusion_seed_is_one_plan_across_a_fit() {
    assert!(
        cf_ss_lag_variants(2, 50.0) <= MAX_CHURN,
        "closing before the record"
    );
}

/// An SS dose into `zero_order` (lowered to a modeled duration, an estimated window): the
/// window's count (`count_gt_multiples`, `residual_end_indices_g`) read the exact value of the
/// moving duration although recorded comparisons pin it. Mutation: `probe_value()` back to
/// `br_value()` in either -> one variant per point.
#[test]
fn an_ss_infusion_with_an_estimated_window_is_one_plan_across_a_fit() {
    for (kernel, v) in [("zero_order", 3.0), ("first_order", 0.4)] {
        let m = tight(parse_model_string(&ss_lag_model(kernel, "alag", 2.0, v)).unwrap());
        let s = subject(
            &m,
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ],
            &[0.4, 4.0, 6.5, 7.5, 10.5, 16.0, 19.0, 22.5, 28.0, 31.0],
            &[0.1, -0.05],
        );
        let n = variants_over(&m, &s, &churn_points());
        assert!(n <= MAX_CHURN, "{kernel}: {n} variants over six points");
    }
}

/// An overlapping SS infusion (`T_inf = 15 > II = 12`: one earlier pulse runs through the whole
/// cycle, the newest for `r = 3`) under an estimated lag is one plan across a fit, and its
/// outer gradient matches the reconverged central difference. Mutation: `phase.br_value()` /
/// `lag.br_value()` back in `ss_state_at_phase_g` for the overlapping case -> 6 variants.
#[test]
fn an_overlapping_ss_infusion_under_a_lag_is_one_plan_across_a_fit() {
    let n = cf_ss_lag_variants(2, 100.0 / 15.0);
    assert!(
        n <= MAX_CHURN,
        "overlapping SS infusion + lag: {n} variants over 6 points"
    );
    let m = parse_model_string(CF_SS_LAG).unwrap();
    let s = subject(
        &m,
        vec![
            DoseEvent::new(0.0, 100.0, 2, 100.0 / 15.0, true, 12.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &[0.1, 1.0, 2.0, 4.0, 8.0, 13.0, 14.0, 18.0, 24.0],
        &[0.1, -0.05],
    );
    assert!(
        served(&m, &s),
        "the adjoint serves the overlapping SS infusion + lag"
    );
    let gap = fd_oracle(&m, &s, true);
    eprintln!("overlapping SS infusion + lag: gradient gap {gap:.2e}");
    assert!(gap < 1e-4, "gradient vs reconverged FD: {gap:.2e}");
}

/// The infusion running at the record (closing at `11.8 < II`) under a lag: one plan.
#[test]
fn a_running_ss_infusion_under_a_lag_is_one_plan_across_a_fit() {
    let n = cf_ss_lag_variants(2, 100.0 / 11.8);
    assert!(
        n <= MAX_CHURN,
        "running-at-the-record SS infusion + lag: {n} variants"
    );
}

// ---- SDE (EKF) churn: an estimated F / modeled window on an infusion ----------------------

/// An SDE model on one amount, with a bioavailability `F1` (`"f"`) or a modeled duration `D1`
/// (`"d"`) the infusion's window carries: the window's end is a moving break of the filter.
fn sde_window_src(which: &str) -> String {
    let (theta, indiv, omega) = match which {
        "d" => (
            "  theta TVD1(0.3, 0.1, 24.0)\n",
            "  D1 = TVD1 * exp(ETA_D1)\n",
            "  omega ETA_D1 ~ 0.04\n",
        ),
        _ => (
            "  theta TVF1(0.8, 0.1, 5.0)\n",
            "  F1 = TVF1 * exp(ETA_F1)\n",
            "  omega ETA_F1 ~ 0.04\n",
        ),
    };
    format!(
        "[parameters]\n  theta TVCL(0.5, 0.01, 10.0)\n  theta TVV(8.0, 0.1, 500.0)\n{theta}  \
         omega ETA_CL ~ 0.07\n{omega}  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL)\n  V = TVV\n{indiv}[structural_model]\n  \
         ode(obs_cmt=central, states=[central])\n[odes]\n  d/dt(central) = -(CL / V) * central\n\
         [diffusion]\n  central ~ 0.02\n[error_model]\n  DV ~ proportional(PROP_ERR)\n"
    )
}

fn sde_window_fixture(which: &str) -> (CompiledModel, Subject) {
    let m = tight(parse_model_string(&sde_window_src(which)).unwrap());
    assert!(m.is_sde());
    // The windows end in `[0.28, 0.57]` after each dose and no record sits where the six churn
    // points could move one across it; every segment's Riccati sub-step count
    // (`ceil(dt / 0.5)`) is the same at all six, since a count that changes is a real change of
    // structure, a variant of its own, not churn.
    let times: &[f64] = &[1.0, 3.5, 9.2, 11.5, 14.0, 20.0];
    let doses = if which == "d" {
        vec![
            DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, RateMode::ModeledDuration),
            DoseEvent::modeled(8.0, 100.0, 1, false, 0.0, RateMode::ModeledDuration),
        ]
    } else {
        vec![
            DoseEvent::new(0.0, 80.0, 1, 200.0, false, 0.0),
            DoseEvent::new(8.0, 80.0, 1, 200.0, false, 0.0),
        ]
    };
    let eta = [0.1, -0.05];
    let mut s = subject(&m, doses, times, &eta[..m.n_eta]);
    let preds =
        crate::pk::compute_predictions_with_tv(&m, &s, &m.default_params.theta, &eta[..m.n_eta]);
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.1 * (1.3 * j as f64).sin()))
        .collect();
    (m, s)
}

/// The variants an SDE subject's variance plan holds after it is served at six points.
fn sde_variants_over(m: &CompiledModel, s: &Subject) -> usize {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    let n = m.n_eta;
    let sigma = m.default_params.sigma.values.clone();
    for k in 0..6 {
        let x = k as f64;
        let theta: Vec<f64> = m
            .default_params
            .theta
            .iter()
            .map(|t| t * (1.0 + 0.04 * x))
            .collect();
        let eta: Vec<f64> = (0..n).map(|j| 0.05 * x - 0.1 + 0.02 * j as f64).collect();
        assert!(
            plan_forward_in(m, s, None, &theta, Some(&sigma), &eta, terms::layout1(n)).is_some(),
            "the variance plan serves point {k}"
        );
        let key = kj::SubjectKey::new(m, s, n, m.default_params.theta.len() + sigma.len())
            .with_layout(None)
            .with_kind(SDE_P_KIND);
        if let kj::Lookup::Found(ks) = kj::lookup(&key) {
            eprintln!(
                "sde churn point {k}: {} variants, theta {theta:?} eta {eta:?}",
                ks.len()
            );
        }
    }
    let key = kj::SubjectKey::new(m, s, n, m.default_params.theta.len() + sigma.len())
        .with_layout(None)
        .with_kind(SDE_P_KIND);
    match kj::lookup(&key) {
        kj::Lookup::Found(ks) => ks.len(),
        _ => panic!("the subject has no variance plan"),
    }
}

/// An SDE subject whose infusion window moves with an estimated `F` (rate-defined) or a modeled
/// `D` is one variance plan across a fit, served exactly: the filter's break times, record
/// reads, dose matches and running infusions are verdicts on the moving break, not exact
/// values (`ekf.rs`: `point_time`, `advance`'s save point, `t_start`/`t_end`, the active set).
/// Mutation: `probe_value()` back to `br_value()` at any of those sites -> one variant per point.
#[test]
fn an_sde_infusion_with_an_estimated_window_is_one_plan_across_a_fit() {
    for which in ["f", "d"] {
        let (m, s) = sde_window_fixture(which);
        let n = sde_variants_over(&m, &s);
        assert!(n <= MAX_CHURN, "{which}: {n} variants over six points");
        assert!(served(&m, &s), "{which}: the adjoint serves the subject");
        let gap = fd_oracle(&m, &s, true);
        eprintln!("sde window {which}: gradient gap {gap:.2e}");
        assert!(gap < 1e-7, "{which}: gradient vs reconverged FD {gap:.2e}");
    }
}

// ---- the term evaluators' `None` declines, enumerated (#2786, AD everywhere G4) -----------

/// `(label, source, undefined identifier read)` of the non-Gaussian endpoint kinds whose term
/// evaluators return `None` without a compiled program: a binary linear predictor
/// (`DiscreteTerm::eval`, `lp_program?`), an analytic hazard (`TteTerm::eval_shared`,
/// `ep.program?`) and a CTMM intensity (`CtmmTerm`, `generator_program`). A parsed endpoint
/// always carries its program (the binder is total, #3004); a program is absent only for an
/// opaque Rust closure endpoint.
#[cfg(feature = "survival")]
fn unbound_identifier_sources() -> Vec<(&'static str, String)> {
    let head = "[parameters]\n  theta TVL(-0.3, -5.0, 5.0)\n  omega ETA_S ~ 0.25\n";
    let mut out = vec![
        (
            "binary predictor",
            format!("{head}[binary_model]\n  cmt = 1\n  logit = TVL + ETA_S + ZZ\n"),
        ),
        (
            "analytic hazard",
            format!(
                "{head}[event_model]\n  cmt = 3\n  family = exponential\n  \
                 scale = exp(TVL + ETA_S) * ZZ\n"
            ),
        ),
    ];
    #[cfg(feature = "markov")]
    out.push((
        "CTMM intensity",
        format!(
            "{head}[markov_model]\n  type = ctmm\n  cmt = 5\n  states = [off=0, on=1]\n  \
             transition off -> on = exp(TVL + ETA_S) * ZZ\n  transition on -> off = exp(TVL)\n"
        ),
    ));
    out
}

/// A binary / hazard / CTMM endpoint reading a name nothing defines is refused by the parser, so
/// the term evaluators' `lp_program?` / `ep.program?` / generator-program `None` is unreachable
/// for an accepted model; declaring the name as a covariate (the control) parses and carries the
/// program (`endpoint_programs_are_total` in `inner_optimizer_tests.rs` covers the one leaf the
/// parser accepts and the binder once could not place, `MACHEPS`).
#[cfg(feature = "survival")]
#[test]
fn an_endpoint_program_is_missing_only_for_a_model_the_parser_refuses() {
    use crate::types::EndpointLikelihood as E;
    let programs = |m: &CompiledModel| {
        m.endpoints.values().all(|ep| match ep {
            E::Binary { lp_program, .. } => lp_program.is_some(),
            E::Tte { hazard_program, .. } => hazard_program.is_some(),
            _ => true,
        })
    };
    for (label, src) in unbound_identifier_sources() {
        if let Ok(m) = parse_model_string(&src) {
            assert!(
                programs(&m),
                "{label}: accepted with an unbound identifier, no program"
            );
        }
        let bound = src.replace(
            "[parameters]",
            "[covariates]\n  ZZ continuous\n[parameters]",
        );
        let m = parse_model_string(&bound).unwrap_or_else(|e| panic!("{label}: control: {e:?}"));
        assert!(
            !m.endpoints.is_empty() && programs(&m),
            "{label}: the bound control has its program"
        );
    }
}

// ---- the inner-gradient FD route (`InnerFdReason::Shape`) of typical clinical ODE models ---

/// `(label, source, time-varying CRCL, doses)` of ODE shapes the benchmark's FD-inner subjects
/// had (vancomycin / cefepime: IV infusions on a 1-2 cpt model with a time-varying `CRCL`;
/// rifapentin: oral with a lag and transit compartments; thiotepa: 2 cpt with a metabolite
/// state). `subject_inner_fd_reason`'s `Shape` is exactly `exact_obs_grads(..).is_none()`
/// (`subject_inner_runs_fd`), so a subject the recorder serves at the prior mode is never counted.
fn clinical_inner_fixtures() -> Vec<(&'static str, String, bool, Vec<DoseEvent>)> {
    let head = |extra_theta: &str, extra_omega: &str| {
        format!(
            "[parameters]\n  theta TVCL(5.0, 0.1, 100.0)\n  theta TVV(50.0, 5.0, 500.0)\n{extra_theta}  \
             omega ETA_CL ~ 0.09\n{extra_omega}  sigma PROP_ERR ~ 0.1 (sd)\n"
        )
    };
    let tail = "[error_model]\n  DV ~ proportional(PROP_ERR)\n";
    let infusions = vec![
        DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0),
        DoseEvent::new(12.0, 1000.0, 1, 500.0, false, 0.0),
        DoseEvent::new(24.0, 1000.0, 1, 500.0, false, 0.0),
    ];
    let boluses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    vec![
        (
            "vancomycin-like: 2 cpt IV infusions, time-varying CRCL",
            format!(
                "[covariates]\n  CRCL continuous\n{}[individual_parameters]\n  \
                 CL = TVCL * exp(ETA_CL) * CRCL / 100\n  V = TVV\n  Q = TVQ\n  V2 = TVV2\n\
                 [structural_model]\n  ode(obs_cmt=central, states=[central, periph])\n[odes]\n  \
                 d/dt(central) = -(CL / V) * central - (Q / V) * central + (Q / V2) * periph\n  \
                 d/dt(periph) = (Q / V) * central - (Q / V2) * periph\n[scaling]\n  y = central / V\n{tail}",
                head("  theta TVQ(8.0, 0.1, 100.0)\n  theta TVV2(40.0, 1.0, 500.0)\n", "")
            ),
            true,
            infusions.clone(),
        ),
        (
            "cefepime-like: 1 cpt IV infusions, time-varying CRCL",
            format!(
                "[covariates]\n  CRCL continuous\n{}[individual_parameters]\n  \
                 CL = TVCL * exp(ETA_CL) * (CRCL / 100)^0.75\n  V = TVV\n[structural_model]\n  \
                 ode(obs_cmt=central, states=[central])\n[odes]\n  \
                 d/dt(central) = -(CL / V) * central\n[scaling]\n  y = central / V\n{tail}",
                head("", "")
            ),
            true,
            infusions.clone(),
        ),
        (
            "rifapentin-like: oral, lag, two transit compartments",
            format!(
                "{}[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  \
                 KTR = TVKTR\n  ALAG1 = TVLAG * exp(ETA_LAG)\n[structural_model]\n  \
                 ode(obs_cmt=central, states=[tr1, tr2, central])\n[odes]\n  \
                 d/dt(tr1) = -KTR * tr1\n  d/dt(tr2) = KTR * tr1 - KTR * tr2\n  \
                 d/dt(central) = KTR * tr2 - (CL / V) * central\n[scaling]\n  y = central / V\n{tail}",
                head(
                    "  theta TVKTR(1.5, 0.05, 20.0)\n  theta TVLAG(0.5, 0.01, 5.0)\n",
                    "  omega ETA_LAG ~ 0.1\n"
                )
            ),
            false,
            boluses.clone(),
        ),
        (
            "thiotepa-like: 2 cpt with a metabolite state",
            format!(
                "{}[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  Q = TVQ\n  \
                 V2 = TVV2\n  KM = TVKM\n[structural_model]\n  \
                 ode(obs_cmt=central, states=[central, periph, met])\n[odes]\n  \
                 d/dt(central) = -(CL / V) * central - (Q / V) * central + (Q / V2) * periph\n  \
                 d/dt(periph) = (Q / V) * central - (Q / V2) * periph\n  \
                 d/dt(met) = 0.6 * (CL / V) * central - KM * met\n[scaling]\n  y = central / V\n{tail}",
                head(
                    "  theta TVQ(8.0, 0.1, 100.0)\n  theta TVV2(40.0, 1.0, 500.0)\n  theta TVKM(0.3, 0.01, 10.0)\n",
                    ""
                )
            ),
            false,
            infusions,
        ),
    ]
}

/// Each clinical ODE fixture's subject is served by the exact inner gradient at the prior mode
/// (so `fd_fallback_warning` counts it as neither `Survival` nor `Shape`), with the shape the
/// benchmark's FD subjects had live (the time-varying covariate really varies). Mutation: route
/// any of these shapes to a scope decline (e.g. make `recorder_in_scope` false for a
/// time-varying covariate on an infusion) -> that fixture is `None`.
#[test]
fn clinical_ode_shapes_run_the_exact_inner_gradient() {
    for (label, src, tv, doses) in clinical_inner_fixtures() {
        let m = tight(parse_model_string(&src).unwrap_or_else(|e| panic!("{label}: {e:?}")));
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 11.5, 13.0, 16.0, 24.5, 30.0];
        let eta = vec![0.1, -0.05, 0.08][..m.n_eta].to_vec();
        let mut s = subject(&m, doses, &times, &eta);
        if tv {
            let w = |v: f64| HashMap::from([("CRCL".to_string(), v)]);
            s.covariates.insert("CRCL".into(), 90.0);
            s.dose_covariates = vec![w(90.0), w(70.0), w(50.0)];
            s.obs_covariates = times
                .iter()
                .map(|&t| {
                    w(if t < 12.0 {
                        90.0
                    } else if t < 24.0 {
                        70.0
                    } else {
                        50.0
                    })
                })
                .collect();
            assert!(s.has_tv_covariates(), "{label}: the covariate varies");
        }
        let zeros = vec![0.0; m.n_eta];
        let got = crate::sens::ode_provider::adjoint::exact_obs_grads(
            &m,
            &s,
            &m.default_params.theta,
            &zeros,
        );
        assert!(
            got.is_some(),
            "{label}: the inner gradient runs on FD (shape decline)"
        );
        assert!(served(&m, &s), "{label}: the outer gradient is served");
    }
}

// ---- which parser-accepted shapes the structural scope predicates still send to FD ---------

/// `(label, source)` of shapes the scope predicates were audited against (#2786, universal-engine
/// item 5), each parsed and asked of [`recorder_in_scope`] and, on a two-dose subject, of the
/// recorder itself. The table is printed (`--nocapture`) and the shapes that are served are
/// asserted; the shapes still declined are asserted too, so a change that serves one turns this
/// red and the report's "still open" row goes with it.
fn scope_probe_sources() -> Vec<(&'static str, String, Option<bool>)> {
    let ode = |odes: &str, ind: &str, cov: &str, scaling: &str, extra_theta: &str| {
        format!(
            "[parameters]\n  theta TVCL(5.0, 0.1, 100.0)\n  theta TVV(50.0, 5.0, 500.0)\n{extra_theta}  \
             omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  \
             CL = TVCL * exp(ETA_CL)\n  V = TVV\n{ind}[structural_model]\n  \
             ode(obs_cmt=central, states=[central])\n[odes]\n{odes}{cov}[scaling]\n{scaling}\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n"
        )
    };
    vec![
        (
            "init(central) = 2 * WT (a covariate in the init)",
            ode(
                "  init(central) = 2 * WT\n  d/dt(central) = -(CL/V) * central\n",
                "",
                "[covariates]\n  WT continuous\n",
                "  y = central / V\n",
                "",
            ),
            None,
        ),
        (
            "weibull + compartment lag",
            ode(
                "  d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central\n",
                "  TD = TVTD\n  BETA = TVBETA\n  ALAG1 = TVLAG\n",
                "",
                "  y = central / V\n",
                "  theta TVTD(2.0, 0.05, 24.0)\n  theta TVBETA(1.5, 0.1, 10.0)\n  theta TVLAG(0.5, 0.01, 5.0)\n",
            ),
            Some(true),
        ),
        (
            "weibull (beta < 1) + route lag",
            ode(
                "  d/dt(central) = weibull(td=TD, beta=BETA, lag=LAG) - CL/V*central\n",
                "  TD = TVTD\n  BETA = TVBETA\n  LAG = TVLAG\n",
                "",
                "  y = central / V\n",
                "  theta TVTD(2.0, 0.05, 24.0)\n  theta TVBETA(0.5, 0.1, 10.0)\n  theta TVLAG(0.5, 0.01, 5.0)\n",
            ),
            Some(true),
        ),
        (
            "transit (n < 1) + compartment lag",
            ode(
                "  d/dt(central) = transit(n=N, mtt=MTT) - CL/V*central\n",
                "  MTT = TVMTT\n  N = TVN\n  ALAG1 = TVLAG\n",
                "",
                "  y = central / V\n",
                "  theta TVMTT(2.0, 0.05, 24.0)\n  theta TVN(0.5, 0.1, 10.0)\n  theta TVLAG(0.5, 0.01, 5.0)\n",
            ),
            Some(true),
        ),
    ]
}

#[test]
fn scope_declines_are_enumerated() {
    for (label, src, expect_served) in scope_probe_sources() {
        let parsed = parse_model_string(&src);
        let Ok(m) = parsed else {
            eprintln!(
                "SCOPE-PROBE {label}: REJECTED BY THE PARSER ({:?})",
                parsed.err()
            );
            continue;
        };
        let m = tight(m);
        let s = two_dose_subject(&m);
        let in_scope = recorder_in_scope(&m);
        let ok = in_scope && served(&m, &s);
        eprintln!("SCOPE-PROBE {label}: recorder_in_scope {in_scope}, served {ok}");
        if let Some(want) = expect_served {
            assert_eq!(ok, want, "{label}");
        }
    }
}

/// A one-compartment oral model (closed form, whose infusions into the central compartment take
/// the event walk, or its one-state ODE twin) with an estimated lag (`ALAG`) and/or
/// a modeled duration `D1`, for the cells where two moving breaks land on one instant.
fn coincident_model(ode: bool, lag: bool, d1: bool, wt: bool) -> CompiledModel {
    let mut theta = String::from("  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n");
    let mut omega = String::from("  omega ETA_CL ~ 0.09\n");
    let mut indiv = String::from("  CL = TVCL * exp(ETA_CL)\n  V = TVV\n");
    let mut call = String::from("cl=CL, v=V, ka=KA");
    if !ode {
        theta += "  theta TVKA(1.2, 0.05, 20.0)\n";
        indiv += "  KA = TVKA\n";
    }
    if lag {
        theta += "  theta TVLAG(0.5, 0.01, 5.0)\n";
        omega += "  omega ETA_LAG ~ 0.1\n";
        indiv += "  ALAG = TVLAG * exp(ETA_LAG)\n";
        call += ", lagtime=ALAG";
    }
    if d1 {
        theta += "  theta TVD1(2.0, 0.1, 24.0)\n";
        omega += "  omega ETA_D1 ~ 0.04\n";
        // The duration of the central compartment's infusions: `D1` on the one-state ODE,
        // `D2` beside the closed form's depot.
        indiv += &format!("  D{} = TVD1 * exp(ETA_D1)\n", if ode { 1 } else { 2 });
    }
    let structure = if ode {
        "  ode(states=[central])\n[odes]\n  d/dt(central) = -CL/V * central\n[scaling]\n  y = central / V\n"
            .to_string()
    } else {
        format!("  pk one_cpt_oral({call})\n")
    };
    let mut src = format!(
        "[parameters]\n{theta}{omega}  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n{indiv}\
         [structural_model]\n{structure}[error_model]\n  DV ~ proportional(PROP_ERR)\n"
    );
    if ode && lag {
        src = src.replace("ALAG =", "LAGTIME =");
    }
    if wt {
        // `WT` scales the clearance: a covariate-change record moves it.
        src = src
            .replace(
                "CL = TVCL * exp(ETA_CL)",
                "CL = TVCL * (WT / 70) * exp(ETA_CL)",
            )
            .replace(
                "[error_model]",
                "[covariates]\n  WT continuous\n[error_model]",
            );
    }
    tight(parse_model_string(&src).unwrap())
}

/// The coincidence cells: `(label, lag, modeled D1, doses, observations)`. Where `D1` is
/// modeled the dose's window end carries the duration's jet next to the lag's, so the two
/// breaks of a group carry different jets; the groups sit at one instant when `ETA_D1 = 0`.
#[allow(clippy::type_complexity)]
fn coincident_cells(cmt: usize) -> Vec<(&'static str, bool, bool, Vec<DoseEvent>, Vec<f64>)> {
    let inf = |t: f64, amt: f64, rate: f64| DoseEvent::new(t, amt, cmt, rate, false, 0.0);
    let bolus = |t: f64, amt: f64| DoseEvent::new(t, amt, cmt, 0.0, false, 0.0);
    let modeled = |t: f64| DoseEvent::modeled(t, 100.0, cmt, false, 0.0, RateMode::ModeledDuration);
    let obs = vec![0.3, 1.0, 1.8, 2.2, 2.9, 3.6, 5.0, 8.0];
    vec![
        // `0 + 2` and `1 + 1`, both shifted by the lag: one jet.
        (
            "two window ends",
            true,
            false,
            vec![inf(0.0, 100.0, 50.0), inf(1.0, 50.0, 50.0)],
            obs.clone(),
        ),
        // The second window opens as the first closes.
        (
            "onset on a window end",
            true,
            false,
            vec![inf(0.0, 100.0, 50.0), inf(2.0, 50.0, 25.0)],
            obs.clone(),
        ),
        // A bolus lands as the first window closes, residual drug present.
        (
            "window end and a later dose's arrival",
            true,
            false,
            vec![inf(0.0, 100.0, 50.0), bolus(2.0, 50.0)],
            obs.clone(),
        ),
        // A modeled window end (the duration's jet and the lag's) on a fixed one (the lag's).
        (
            "two window ends, different jets",
            true,
            true,
            vec![modeled(0.0), inf(1.0, 50.0, 50.0)],
            obs.clone(),
        ),
        (
            "onset on a window end, different jets",
            true,
            true,
            vec![modeled(0.0), inf(2.0, 50.0, 25.0)],
            obs.clone(),
        ),
        (
            "window end and a later dose's arrival, different jets",
            true,
            true,
            vec![modeled(0.0), bolus(2.0, 50.0)],
            obs.clone(),
        ),
        // The bolus listed before the window it lands on: the walk stands at the window's end,
        // the bolus arrived at its own instant (the lag's jet, or none).
        (
            "a bolus listed before the modeled end it lands on",
            true,
            true,
            vec![bolus(2.0, 50.0), modeled(0.0)],
            obs.clone(),
        ),
        (
            "a bolus listed before the modeled end it lands on, no lag",
            false,
            true,
            vec![bolus(2.0, 50.0), modeled(0.0)],
            vec![0.5, 1.5, 2.5, 4.0, 8.0],
        ),
        // No lag: the modeled end meets a fixed record's arrival and a fixed window's end.
        (
            "modeled end on a fixed bolus",
            false,
            true,
            vec![modeled(0.0), bolus(2.0, 50.0)],
            vec![0.5, 1.5, 2.5, 4.0, 8.0],
        ),
        (
            "modeled end on a fixed window's end and onset",
            false,
            true,
            vec![modeled(0.0), inf(1.0, 50.0, 50.0), inf(2.0, 50.0, 25.0)],
            vec![0.5, 1.5, 2.5, 4.0, 8.0],
        ),
    ]
}

/// The `η` at which a cell's breaks coincide: any `ETA_LAG` keeps breaks sharing the lag's jet
/// together; a modeled `D1` meets its partner only at `ETA_D1 = 0`.
fn coincident_eta(lag: bool, d1: bool) -> Vec<f64> {
    let mut eta = vec![0.06];
    if lag {
        eta.push(-0.052);
    }
    if d1 {
        eta.push(0.0);
    }
    eta
}

/// Breaks that land on one instant are crossed together at their own jets, so a group is
/// served -- the closed-form walk and the ODE event walk alike -- with `∂f/∂η` matching the
/// reconverged central difference of the production predictor, which is smooth across the
/// coincidence (the flow is linear in its input: each break's effect is its own, whichever is
/// crossed first). It used to decline the point to FD (`T::fail_point`). Cells: two lagged window
/// ends (`0+2`, `1+1`), an onset on an end, an end on a later dose's arrival, each with one jet
/// and with two (a modeled duration beside the lag), and the no-lag modeled end on fixed breaks.
/// Mutations, each failing this test: restoring the decline of every coincidence in
/// `moving_bounds_separable`; dropping `break_rate_jump`'s rates between the crossings of a
/// group (the infusion cells); dropping the bolus's `EdgeGroup` gap (the bolus cells).
#[test]
fn coincident_moving_breaks_are_served_and_match_fd() {
    for ode in [false, true] {
        for (label, lag, d1, doses, obs) in coincident_cells(if ode { 1 } else { 2 }) {
            let label = format!("{} / {label}", if ode { "ODE" } else { "closed form" });
            let m = coincident_model(ode, lag, d1, false);
            let eta = coincident_eta(lag, d1);
            let s = subject(&m, doses, &obs, &eta);
            let jac = jacobian_gap(&m, &s, &eta, &label);
            let theta = m.default_params.theta.clone();
            let layout = terms::layout1(m.n_eta);
            assert!(
                plan_forward(&m, &s, None, &theta, &eta, layout).is_some(),
                "{label}: served"
            );
            eprintln!("{label}: dF/dEta vs FD {jac:.2e}");
            // Measured on Linux (dev profile): worst 9.6e-12 over the cells; ~10x headroom.
            assert!(jac < 1e-10, "{label}: {jac:.2e}");
            // One jet: the group persists at every η the fit visits, so the FOCEI gradient
            // (the EBE re-found at each point) is checked as well.
            if !d1 {
                let grad = parity(&m, &s);
                eprintln!("{label}: FOCEI gradient vs reconverged FD {grad:.2e}");
                // Measured: closed form <= 1.9e-9, ODE <= 4.4e-8 (the ODE oracle's own noise).
                let bound = if ode { 5e-7 } else { 2e-8 };
                assert!(grad < bound, "{label}: gradient {grad:.2e}");
            }
        }
    }
}

/// One cell of the tie tests: breaks that share their instant with a fixed record.
struct TieCell {
    label: &'static str,
    /// The model reads `WT` (a covariate change moves `CL`).
    wt: bool,
    lag: bool,
    d1: bool,
    /// The doses (the central compartment is `cmt`).
    doses: fn(usize) -> Vec<DoseEvent>,
    /// The sample times; `None` stands for the tie's instant.
    obs: &'static [Option<f64>],
    /// A covariate-change record at the tie (`WT` 70 -> 140), or a reset.
    record: Record,
    /// Which way `η_lag` (and `η_D1`) must move for the breaks to stay on the value path's
    /// side of the record: `+1` (later), `-1` (earlier), `0` (the function is smooth there).
    side: f64,
}

#[derive(Clone, Copy, PartialEq)]
enum Record {
    None,
    Change,
    Reset,
}

fn tie_cells() -> Vec<TieCell> {
    vec![
        // A window end on the record: the end is crossed after the new parameters hold.
        TieCell {
            label: "window end on a covariate change",
            wt: true,
            lag: true,
            d1: false,
            doses: |c| vec![DoseEvent::new(0.0, 100.0, c, 50.0, false, 0.0)],
            obs: &[
                Some(0.3),
                Some(1.0),
                Some(1.8),
                Some(2.9),
                Some(3.5),
                Some(5.0),
                Some(8.0),
            ],
            record: Record::Change,
            side: 1.0,
        },
        // A window opening on the record: it opens before the new parameters hold.
        TieCell {
            label: "window onset on a covariate change",
            wt: true,
            lag: true,
            d1: false,
            doses: |c| {
                vec![
                    DoseEvent::new(0.0, 25.0, c, 50.0, false, 0.0),
                    DoseEvent::new(2.0, 50.0, c, 25.0, false, 0.0),
                ]
            },
            obs: &[
                Some(0.3),
                Some(1.0),
                Some(1.8),
                Some(2.9),
                Some(3.5),
                Some(5.0),
                Some(8.0),
            ],
            record: Record::Change,
            side: -1.0,
        },
        // A bolus on the record, residual drug present.
        TieCell {
            label: "bolus on a covariate change",
            wt: true,
            lag: true,
            d1: false,
            doses: |c| {
                vec![
                    DoseEvent::new(0.0, 100.0, c, 100.0, false, 0.0),
                    DoseEvent::new(2.0, 50.0, c, 0.0, false, 0.0),
                ]
            },
            obs: &[
                Some(0.3),
                Some(1.0),
                Some(1.8),
                Some(2.9),
                Some(3.5),
                Some(5.0),
                Some(8.0),
            ],
            record: Record::Change,
            side: -1.0,
        },
        // The window end is cut short by a reset on it: nothing of it survives either way.
        TieCell {
            label: "window end on a reset",
            wt: false,
            lag: true,
            d1: false,
            doses: |c| {
                vec![
                    DoseEvent::new(0.0, 100.0, c, 50.0, false, 0.0),
                    DoseEvent::new(3.0, 50.0, c, 0.0, false, 0.0),
                ]
            },
            obs: &[
                Some(0.3),
                Some(1.0),
                Some(1.8),
                Some(2.9),
                Some(3.5),
                Some(5.0),
                Some(8.0),
            ],
            record: Record::Reset,
            side: 0.0,
        },
        // Samples on the group: the ends still contribute.
        TieCell {
            label: "sample on two window ends",
            wt: false,
            lag: true,
            d1: false,
            doses: |c| {
                vec![
                    DoseEvent::new(0.0, 100.0, c, 50.0, false, 0.0),
                    DoseEvent::new(1.0, 50.0, c, 50.0, false, 0.0),
                ]
            },
            obs: &[
                Some(0.3),
                Some(1.0),
                Some(1.8),
                None,
                Some(3.5),
                Some(5.0),
                Some(8.0),
            ],
            record: Record::None,
            side: 1.0,
        },
        TieCell {
            label: "sample on two window ends, different jets",
            wt: false,
            lag: true,
            d1: true,
            doses: |c| {
                vec![
                    DoseEvent::modeled(0.0, 100.0, c, false, 0.0, RateMode::ModeledDuration),
                    DoseEvent::new(1.0, 50.0, c, 50.0, false, 0.0),
                ]
            },
            obs: &[
                Some(0.3),
                Some(1.0),
                Some(1.8),
                None,
                Some(3.5),
                Some(5.0),
                Some(8.0),
            ],
            record: Record::None,
            side: 1.0,
        },
        // A modeled end and a fixed window's end on one sample, no lag.
        TieCell {
            label: "sample on a modeled end and a fixed window's end",
            wt: false,
            lag: false,
            d1: true,
            doses: |c| {
                vec![
                    DoseEvent::modeled(0.0, 100.0, c, false, 0.0, RateMode::ModeledDuration),
                    DoseEvent::new(1.0, 50.0, c, 50.0, false, 0.0),
                ]
            },
            obs: &[
                Some(0.3),
                Some(1.0),
                Some(1.8),
                None,
                Some(3.5),
                Some(5.0),
                Some(8.0),
            ],
            record: Record::None,
            side: 1.0,
        },
    ]
}

/// The subject of a [`TieCell`], its record and tie instant `t0` placed: `WT` is 70 on every
/// dose and sample and 140 on the covariate-change record.
fn tie_subject(m: &CompiledModel, c: &TieCell, cmt: usize, t0: f64) -> Subject {
    let wt = |v: f64| HashMap::from([("WT".to_string(), v)]);
    let times: Vec<f64> = c.obs.iter().map(|t| t.unwrap_or(t0)).collect();
    let mut s = subject_shell((c.doses)(cmt), &times);
    if c.wt {
        s.covariates = wt(70.0);
        s.dose_covariates = vec![wt(70.0); s.doses.len()];
        // The record's own `WT` governs the interval that ends on it; the sample after it is back
        // at 70, so the parameters differ across the instant the breaks land on.
        s.obs_covariates = vec![wt(70.0); times.len()];
    }
    match c.record {
        Record::Change => {
            s.pk_only_times = vec![t0];
            s.pk_only_raw_times = vec![t0];
            s.pk_only_covariates = vec![wt(140.0)];
            s.pk_only_occasions = vec![1];
        }
        Record::Reset => s.reset_times = vec![t0],
        Record::None => {}
    }
    let eta = coincident_eta(c.lag, c.d1);
    let preds = crate::pk::compute_predictions_with_tv(m, &s, &m.default_params.theta, &eta);
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
        .collect();
    s
}

/// `∂f/∂η` of the production predictor from one side: the second-order difference toward `side`
/// (`+1` later, `-1` earlier; a smooth cell takes `+1`).
fn one_sided_jacobian(
    m: &CompiledModel,
    s: &Subject,
    theta: &[f64],
    eta: &[f64],
    side: f64,
) -> nalgebra::DMatrix<f64> {
    let side = if side == 0.0 { 1.0 } else { side };
    let mut out = nalgebra::DMatrix::zeros(s.obs_times.len(), m.n_eta);
    let h = 1e-4 * side;
    for k in 0..m.n_eta {
        let at = |d: f64| {
            let mut e = eta.to_vec();
            e[k] += d;
            crate::pk::compute_predictions_with_tv(m, s, theta, &e)
        };
        let (f0, f1, f2) = (at(0.0), at(h), at(2.0 * h));
        for j in 0..f0.len() {
            out[(j, k)] = (-3.0 * f0[j] + 4.0 * f1[j] - f2[j]) / (2.0 * h);
        }
    }
    out
}

/// Breaks that share their instant with a fixed record are crossed on the value path's side of
/// it, in the closed-form walk and the ODE event walk alike, and never decline a point: a window
/// end after a covariate change (`WT` moves `CL`) crosses once the new parameters hold, a window
/// opening or a bolus before them, a reset discards the state whatever crossed, and a sample on
/// two coincident breaks reads them as a single break does (arrivals landed, ends still
/// contributing). `∂f/∂η` against the one-sided second-order difference of the production
/// predictor, which is how the kinked cells are read. Mutations, each failing this test:
/// dropping the `rec_right` / `rec_left` handling in `propagate_bounds_g` (the covariate-change
/// cells); `obs_group_reading` returning the state unflowed (the sample cells).
#[test]
fn breaks_on_a_fixed_record_are_served_and_match_one_sided_fd() {
    for ode in [false, true] {
        for c in tie_cells() {
            let label = format!("{} / {}", if ode { "ODE" } else { "closed form" }, c.label);
            let m = coincident_model(ode, c.lag, c.d1, c.wt);
            let eta = coincident_eta(c.lag, c.d1);
            let theta = m.default_params.theta.clone();
            let lag = if c.lag {
                theta[if ode { 2 } else { 3 }] * eta[1].exp()
            } else {
                0.0
            };
            let s = tie_subject(&m, &c, if ode { 1 } else { 2 }, 2.0 + lag);
            let jac = exact_eta_jacobian(&m, &s, &theta, &eta)
                .unwrap_or_else(|| panic!("{label}: the recorder declined the η-Jacobian"));
            let fd = one_sided_jacobian(&m, &s, &theta, &eta, c.side);
            let mut worst = 0.0f64;
            for (a, b) in jac.iter().zip(fd.iter()) {
                assert!(
                    a.is_finite() && b.is_finite(),
                    "{label}: non-finite {a} vs {b}"
                );
                worst = worst.max((a - b).abs() / (1.0 + b.abs()));
            }
            eprintln!("{label}: dF/dEta vs one-sided FD {worst:.2e}");
            assert!(worst < 1e-6, "{label}: {worst:.2e}");
        }
    }
}

/// The second-order half of the ties: the exact inner Hessian `∇²_η l` against a Richardson
/// difference of the exact inner gradient, at a tie of breaks sharing one jet (the two cells whose
/// function is smooth across the tie, so the central difference is the oracle), on the closed
/// form and on the ODE twin. Measured: the ODE walk's kind-ordered crossing of the tie leaves no
/// second-order gap on either cell, so its order stays as it is and this test pins that.
/// (A tie of one jet is an equality of duals, so crossing its breaks in any order, or with the
/// rates between them dropped, is the same function: the cells that see the rates are the
/// different-jet ones of `coincident_moving_breaks_are_served_and_match_fd`. Mutation here:
/// declining a coincident group, which leaves no exact Hessian to compare.)
#[test]
fn the_inner_hessian_at_a_tie_matches_the_gradient_difference() {
    for ode in [false, true] {
        for (label, lag, d1, doses, obs) in coincident_cells(if ode { 1 } else { 2 }) {
            if d1 || !matches!(label, "two window ends" | "onset on a window end") {
                continue;
            }
            let label = format!("{} / {label}", if ode { "ODE" } else { "closed form" });
            let m = coincident_model(ode, lag, d1, false);
            let eta = coincident_eta(lag, d1);
            let s = subject(&m, doses, &obs, &eta);
            let p = &m.default_params;
            let h = exact_inner_hessian(&m, &s, p, &eta)
                .unwrap_or_else(|| panic!("{label}: no exact Hessian"));
            let g = |e: &[f64]| {
                exact_inner_gradient(&m, &s, p, e, None)
                    .unwrap_or_else(|| panic!("{label}: no exact gradient"))
            };
            let mut worst = 0.0f64;
            for b in 0..m.n_eta {
                let d = |st: f64| {
                    let (mut ep, mut em) = (eta.clone(), eta.clone());
                    ep[b] += st;
                    em[b] -= st;
                    let (gp, gm) = (g(&ep), g(&em));
                    (0..m.n_eta)
                        .map(|a| (gp[a] - gm[a]) / (2.0 * st))
                        .collect::<Vec<f64>>()
                };
                let (d1v, d2v) = (d(1e-4), d(2e-4));
                for a in 0..m.n_eta {
                    let rich = (4.0 * d1v[a] - d2v[a]) / 3.0;
                    assert!(
                        h[(a, b)].is_finite() && rich.is_finite(),
                        "{label}: non-finite"
                    );
                    worst = worst.max((h[(a, b)] - rich).abs() / (1.0 + rich.abs()));
                }
            }
            eprintln!("{label}: exact Hessian vs Richardson of the exact gradient {worst:.2e}");
            assert!(worst < 1e-6, "{label}: {worst:.2e}");
        }
    }
}

// ---- steady state under a clock-reading right-hand side (AD everywhere, G1) ---------------

/// [`ODE_LAG`] (depot + central, an estimated `LAGTIME`) with the central's elimination scaled by
/// `factor` -- an expression in `TIME`/`T`/`TAD` -- and the lag kept or traded for an `eta_V`.
fn clock_model(factor: &str, lag: bool) -> CompiledModel {
    let mut src = ODE_LAG.replace(
        "d/dt(central) = KA * depot - (CL/V) * central",
        &format!("d/dt(central) = KA * depot - (CL/V) * central * ({factor})"),
    );
    if !lag {
        src = src
            .replace("omega ETA_LAG ~ 0.1", "omega ETA_V ~ 0.1")
            .replace("V = TVV", "V = TVV * exp(ETA_V)")
            .replace("  LAGTIME = TVLAG * exp(ETA_LAG)\n", "");
    }
    tight(parse_model_string(&src).unwrap())
}

/// An `SS=1` dose at 0 (`II = 12`, a bolus, or an infusion of 5 h when `inf`) into the depot, a
/// second on its residual at 12; observations before and after the lagged arrival and across
/// the infusion's end.
fn clock_subject(m: &CompiledModel, inf: bool) -> Subject {
    let first = if inf {
        DoseEvent::new(0.0, 100.0, 1, 20.0, true, 12.0)
    } else {
        DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)
    };
    subject(
        m,
        vec![first, DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0)],
        &[
            0.2, 0.3, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 11.0, 13.0, 14.0, 18.0, 24.0,
        ],
        &[0.1, -0.05],
    )
}

/// `SS=1` into a right-hand side that reads `TIME`/`T`, `TAD`, or both -- with and without an
/// estimated lag, bolus and infusion -- is served by the plan and matches reconverged FD of the
/// production objective. It used to route to FD (`ss_clock_out_of_scope`): the dual run-in did
/// not replay production's clocks (the quiet window's local `TIME`, the lagged seed's
/// pre-arrival `TAD` anchored at the previous cycle's pulse).
#[test]
fn ss_into_a_clock_reading_rhs_is_served_and_matches_fd() {
    for (factor, lag, inf) in [
        ("1 + 0.05 * TIME", false, false),
        ("1 + 0.05 * T", true, false),
        ("1 + 0.05 * TIME", true, false),
        ("1 + 0.3 * exp(-0.5 * TAD)", true, false),
        ("1 + 0.05 * TIME + 0.3 * exp(-0.5 * TAD)", true, false),
        ("1 + 0.05 * TIME", false, true),
        ("1 + 0.05 * TIME", true, true),
        ("1 + 0.3 * exp(-0.5 * TAD)", true, true),
        ("1 + 0.05 * TIME + 0.3 * exp(-0.5 * TAD)", true, true),
    ] {
        let label = format!("SS clock `{factor}` lag={lag} inf={inf}");
        let m = clock_model(factor, lag);
        let s = clock_subject(&m, inf);
        assert!(served(&m, &s), "{label}: served by the plan");
        let eta = probe_eta(m.n_eta);
        let jac = jacobian_gap(&m, &s, &eta, &label);
        let grad = parity(&m, &s);
        eprintln!("{label}: dF/dEta vs FD {jac:.2e}; FOCEI gradient vs reconverged FD {grad:.2e}");
        assert!(jac.is_finite() && grad.is_finite(), "{label}");
        // Measured on Linux (dev profile): dF/dEta <= 9.5e-11, FOCEI gradient <= 7.4e-7 (the
        // lagged bolus cells). The lagged infusion cells read up to 4.3e-5 before the flow
        // extension kept its explicit time derivative
        // (`a_lagged_ss_infusion_on_a_clock_reading_rhs_has_the_exact_inner_hessian`): that was a
        // real second-order gap, not the oracle's noise, so the bound is the measured one.
        assert!(jac < 1e-9 && grad < 1e-5, "{label}: {jac:.2e} / {grad:.2e}");
    }
}

// ---- readouts (#2786, AD-everywhere G2): none routes to finite differences by scope --------

/// `ONE_CPT_ORAL` (three η) with a `[scaling]` readout block.
fn cf_readout(lines: &str) -> CompiledModel {
    let src = ONE_CPT_ORAL.replace("[error_model]", &format!("[scaling]\n{lines}[error_model]"));
    tight(parse_model_string(&src).unwrap())
}

/// The model with a time-varying weight on `CL`, so a subject whose rows carry their own `WT`
/// takes the closed-form event walk.
fn tv_variant(src: &str) -> String {
    src.replace(
        "CL = TVCL * exp(ETA_CL)",
        "CL = TVCL * (WT / 70.0)^0.75 * exp(ETA_CL)",
    )
}

/// `s` with its observations redrawn from the production predictor at `eta`, off the
/// predictions by a deterministic multiplicative pattern of both signs.
fn redraw(m: &CompiledModel, s: &mut Subject, eta: &[f64]) {
    let preds = crate::pk::compute_predictions_with_tv(m, s, &m.default_params.theta, eta);
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
        .collect();
}

/// Two doses (the second on residual drug), eight observations; `tv` gives every record its
/// own `WT`, so the walk switches parameters with drug on board.
fn cf_readout_subject(m: &CompiledModel, tv: bool, cmts: Option<&[usize]>) -> Subject {
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [1.0, 2.0, 4.0, 8.0, 13.0, 14.0, 18.0, 24.0];
    let mut s = subject_shell(doses.clone(), &times);
    if tv {
        let wt_at = |t: f64| HashMap::from([("WT".to_string(), 60.0 + 1.5 * t)]);
        s.covariates.insert("WT".into(), 70.0);
        s.obs_covariates = times.iter().map(|&t| wt_at(t)).collect();
        s.dose_covariates = doses.iter().map(|d| wt_at(d.time)).collect();
        assert!(s.has_tv_covariates());
    }
    if let Some(c) = cmts {
        s.obs_cmts = c.to_vec();
    }
    redraw(m, &mut s, &probe_eta(m.n_eta));
    s
}

/// What a readout case asserts: the recorder is in scope, serves the subject, and the η-Jacobian
/// and FOCEI gradient agree with the production predictor's Richardson differences and the
/// reconverged-FD objective. Returns the two gaps.
fn readout_gaps(m: &CompiledModel, s: &Subject, label: &str) -> (f64, f64) {
    assert!(recorder_in_scope(m), "{label}: the recorder scope declines");
    measure(m, s, &[0, 1, 2], label)
}

/// Per-CMT `y[CMT=N]` readouts on a closed form used to decline every subject (the scope took
/// `Single` programs only). Each observation now evaluates its own CMT's program: two
/// different expressions, observations alternating between them, on the static superposition
/// and the event walk. Non-degenerate: the two programs give different predictions at one time
/// (asserted), so a program picked for the wrong CMT fails the Jacobian.
/// Mutations: take the `Single` arm for a `PerCmt` readout (`ar.program` is `None`) -> the
/// recorder declines and `measure` fails on `served`; index `readout_program_at` by the
/// observation index instead of its CMT -> the Jacobian gap is O(1).
#[test]
fn a_per_cmt_closed_form_readout_is_served_and_matches_reconverged_fd() {
    let ro = "  y[CMT=1] = central / V\n  y[CMT=2] = 2.0 * central / V + 0.1 * KA\n";
    let cmts: Vec<usize> = (0..8).map(|j| 1 + j % 2).collect();
    for tv in [false, true] {
        let m = if tv {
            tight(parse_model_string(&tv_variant(&cf_readout_src(ro))).unwrap())
        } else {
            cf_readout(ro)
        };
        assert!(matches!(
            m.analytic_readout.as_ref().unwrap().readout,
            crate::ode::OdeReadout::PerCmt(_)
        ));
        let s = cf_readout_subject(&m, tv, Some(&cmts));
        let mut s1 = s.clone();
        s1.obs_cmts = vec![1; 8];
        let (p, p1) = (
            crate::pk::compute_predictions_with_tv(&m, &s, &m.default_params.theta, &probe_eta(3)),
            crate::pk::compute_predictions_with_tv(&m, &s1, &m.default_params.theta, &probe_eta(3)),
        );
        assert!(
            (p[1] - p1[1]).abs() > 1e-3 * p1[1].abs(),
            "premise: the two programs differ at one time"
        );
        let label = format!("per-CMT closed-form readout, tv {tv}");
        let (jac, grad) = readout_gaps(&m, &s, &label);
        // Measured worst on Linux: Jacobian 4.9e-11, gradient 1.8e-9.
        assert!(jac < 5e-10 && grad < 3e-8, "{label}: {jac:.2e} {grad:.2e}");
    }
}

fn cf_readout_src(lines: &str) -> String {
    ONE_CPT_ORAL.replace("[error_model]", &format!("[scaling]\n{lines}[error_model]"))
}

/// A readout over the central amount of a closed form with `[initial_conditions]` used to
/// decline the model (`analytic_readout_dual_supported`: no readout beside an init), and the
/// event walk layered the init impulse AFTER the readout. Production adds it to the
/// concentration first, so a readout that is not the identity in the central amount (here
/// `central/V + 0.2·central/V²`, curved in V) reads `conc + init`. Static superposition and
/// the event walk (time-varying weight), both against the reconverged-FD gradient and the
/// production Jacobian; `TVC0` is live (asserted by `measure`).
/// Mutations: add the init impulse after the readout in `record_analytic_events` -> the event
/// walk's Jacobian gap is O(1); restore the `analytical_init` decline -> `recorder_in_scope`
/// fails.
#[test]
fn a_closed_form_readout_over_an_init_baseline_is_served_and_matches_reconverged_fd() {
    let src = super::tests::CF_INIT.replace(
        "[error_model]",
        "[scaling]\n  y = central / V + 0.2 * central / (V * V)\n[error_model]",
    );
    for tv in [false, true] {
        let src = if tv { tv_variant(&src) } else { src.clone() };
        let m = tight(parse_model_string(&src).unwrap());
        assert!(!m.analytical_init.is_empty() && m.analytic_readout.is_some());
        let s = cf_readout_subject(&m, tv, None);
        let label = format!("init baseline + readout, tv {tv}");
        assert!(recorder_in_scope(&m), "{label}: scope");
        let (jac, grad) = measure(&m, &s, &[0, 1, 2, 3], &label);
        // Measured worst on Linux: Jacobian 3.1e-12, gradient 1.0e-9.
        assert!(jac < 3e-11 && grad < 1e-8, "{label}: {jac:.2e} {grad:.2e}");
    }
}

/// One arm of the depot-readout check (see the tests below): the closed-form oral model with
/// `y = (central + 0.3·depot)/V` on the subject the flags describe, in scope, served, and
/// against the reconverged-FD gradient and the production Jacobian. The arm is live on the
/// depot: the prediction with `0.3·depot` differs from the one without it (asserted), and
/// `TVLAG` moves the lagged arms. Measured worst on Linux: Jacobian 1.1e-10, gradient 2.8e-9.
fn depot_readout_arm(name: &str, tv: bool, lag: bool, ss: bool, inf: bool) {
    let with_depot = "  y = (central + 0.3 * depot) / V\n";
    let without = "  y = central / V\n";
    let build = |ro: &str| {
        let mut src = cf_readout_src(ro);
        if tv {
            // The weight moves `KA` too: the depot reads it, so the snapshot it is built at
            // (the static one, not each observation's) decides the prediction.
            src = tv_variant(&src).replace(
                "KA = TVKA * exp(ETA_KA)",
                "KA = TVKA * (WT / 70.0)^0.5 * exp(ETA_KA)",
            );
        }
        if lag {
            src = src
                .replace(
                    "  theta TVKA(1.5, 0.01, 50.0)\n",
                    "  theta TVKA(1.5, 0.01, 50.0)\n  theta TVLAG(0.4, 0.01, 5.0)\n",
                )
                .replace(
                    "  KA = TVKA * exp(ETA_KA)\n",
                    "  KA = TVKA * exp(ETA_KA)\n  ALAG = TVLAG\n",
                )
                .replace(
                    "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
                    "pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=ALAG)",
                );
        }
        tight(parse_model_string(&src).unwrap())
    };
    let m = build(with_depot);
    assert!(m.analytic_readout.as_ref().unwrap().references_depot());
    let mut s = cf_readout_subject(&m, tv, None);
    if ss {
        s.doses[0] = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
    }
    if inf {
        // A zero-order input into central: it bypasses the depot, which holds only the first
        // (bolus) dose's amount.
        s.doses[1] = DoseEvent::new(12.0, 100.0, 2, 50.0, false, 0.0);
    }
    redraw(&m, &mut s, &probe_eta(3));
    let m0 = build(without);
    let theta = &m.default_params.theta;
    let eta = probe_eta(3);
    let (a, b) = (
        crate::pk::compute_predictions_with_tv(&m, &s, theta, &eta),
        crate::pk::compute_predictions_with_tv(&m0, &s, theta, &eta),
    );
    assert!(
        (a[0] - b[0]).abs() > 1e-3 * b[0].abs(),
        "{name}: premise: the depot term moves the first prediction"
    );
    let live: Vec<usize> = if lag { vec![0, 1, 2, 3] } else { vec![0, 1, 2] };
    assert!(recorder_in_scope(&m), "{name}: scope");
    let (jac, grad) = measure(&m, &s, &live, &format!("depot readout, {name}"));
    assert!(jac < 1e-9 && grad < 3e-8, "{name}: {jac:.2e} {grad:.2e}");
}

/// A readout reading the oral depot amount used to decline the model, whatever the subject.
/// Production reconstructs the depot by superposition at the subject's `t = 0` parameters
/// (`analytical_state_at_times`); the recorders now do the same on the tape. Static
/// superposition. Mutation: leave the depot at zero in the readout state -> the Jacobian gap is
/// O(1) (on every arm below too).
#[test]
fn a_depot_reading_closed_form_readout_matches_reconverged_fd() {
    depot_readout_arm("static", false, false, false, false);
}

/// The depot on a time-varying-covariate subject (the event walk) is the static `t = 0`
/// snapshot's, not the per-observation one the readout's own parameters read. Mutation: build
/// the depot from the observation's own snapshot -> this arm only.
#[test]
fn a_depot_reading_readout_on_a_time_varying_subject_uses_the_static_snapshot() {
    depot_readout_arm("tv (event walk)", true, false, false, false);
}

/// The depot of a lagged dose arrives at `t + ALAG`; `TVLAG` is live. Mutation: take the
/// unlagged elapsed time -> this arm and the SS one.
#[test]
fn a_depot_reading_readout_under_a_lag_matches_reconverged_fd() {
    depot_readout_arm("lagged", false, true, false, false);
}

/// An `SS=1` bolus carries the steady-state factor in the depot. Mutation: drop `ss_factor` in
/// `closed_form_depot_amounts_g` -> this arm only.
#[test]
fn a_depot_reading_readout_over_a_steady_state_dose_matches_reconverged_fd() {
    depot_readout_arm("steady state + lag", false, true, true, false);
}

/// An infusion into central bypasses the depot (production: `single_dose_states`), and takes
/// the event walk. Mutation: let an infusion contribute to the depot -> this arm only.
#[test]
fn a_depot_reading_readout_beside_an_infusion_into_central_matches_reconverged_fd() {
    depot_readout_arm(
        "infusion into central (event walk)",
        false,
        false,
        false,
        true,
    );
}

/// The depot readout under IOV: each occasion's depot is the superposition at that occasion's
/// own `[η, κ_g]` snapshot (`predict_iov` runs `apply_analytic_readout` once per occasion
/// group), so κ on `V`/`CL` reaches the depot term of its own occasion only.
/// Mutation: build one depot from the first occasion's snapshot -> the stacked Jacobian gap is
/// O(1) from the second occasion on.
#[test]
fn a_depot_reading_closed_form_readout_under_iov_matches_reconverged_fd() {
    // κ_V also moves `KA`, so each occasion's depot differs from the others'.
    let src = super::tests::CF_IOV
        .replace("  KA = TVKA\n", "  KA = TVKA * exp(KAPPA_V)\n")
        .replace(
            "[error_model]",
            "[scaling]\n  y = (central + 0.3 * depot) / V\n[error_model]",
        );
    let m = tight(parse_model_string(&src).unwrap());
    assert!(m.analytic_readout.as_ref().unwrap().references_depot());
    let s = super::tests::iov_subject(&m);
    assert!(recorder_in_scope(&m) && served(&m, &s), "scope");
    let jac = iov_jacobian_gap(
        &m,
        &s,
        &[0.06, -0.04, 0.05, -0.03, 0.04, 0.02, -0.05, 0.03],
        "iov depot",
    );
    let grad = fd_oracle(&m, &s, true);
    eprintln!("depot readout × IOV: ∂f/∂[η, κ] {jac:.2e}; gradient {grad:.2e}");
    // Measured worst on Linux: 4.2e-12 and 4.5e-12.
    assert!(jac < 5e-10 && grad < 5e-10, "{jac:.2e} {grad:.2e}");
}

/// More direct θ reads than the closed form has spare PK slots (twelve; the oral pool has
/// at most eleven) leave the θ reads in the readout, which declined the model
/// (`dual_evaluable` was false). They are read from the θ leaves now, so each `A_i` moves the
/// prediction and the gradient. Premise asserted: the readout does read leaves.
/// Both closed-form routes: static superposition and the event walk (time-varying weight).
/// Mutation: pass empty θ to the readout's `eval_output_g` (in `analytic_static_predictions`, or
/// in `readout_leaves` for the event walk) -> every `A_i` reads 0, the prediction is 0 and
/// `measure` fails on a dead θ, on that route only.
#[test]
fn a_readout_reading_more_thetas_than_the_slot_layout_holds_is_served() {
    let names: Vec<String> = (1..=12).map(|i| format!("A{i}")).collect();
    let thetas: String = names
        .iter()
        .map(|n| format!("  theta {n}(1.0, 0.1, 10.0)\n"))
        .collect();
    let expr: String = names
        .iter()
        .enumerate()
        .map(|(i, n)| format!("{}{n}", if i == 0 { "" } else { " + 0.05 * " }))
        .collect();
    let src = ONE_CPT_ORAL
        .replace(
            "  omega ETA_CL ~ 0.09\n",
            &format!("{thetas}  omega ETA_CL ~ 0.09\n"),
        )
        .replace(
            "[error_model]",
            &format!("[scaling]\n  y = central / V * ({expr})\n[error_model]"),
        );
    for tv in [false, true] {
        let src = if tv { tv_variant(&src) } else { src.clone() };
        let m = tight(parse_model_string(&src).unwrap());
        let prog = m
            .analytic_readout
            .as_ref()
            .unwrap()
            .program
            .as_ref()
            .unwrap();
        assert!(
            prog.reads_leaves(),
            "premise: the θ reads stayed in the readout"
        );
        let s = cf_readout_subject(&m, tv, None);
        assert!(recorder_in_scope(&m), "scope");
        let label = format!("twelve direct θ reads, tv {tv}");
        let (jac, grad) = measure(&m, &s, &[0, 1, 2, 3, 8, 14], &label);
        // Measured worst on Linux: Jacobian 4.6e-11, gradient 2.0e-9.
        assert!(jac < 5e-10 && grad < 3e-8, "{label}: {jac:.2e} {grad:.2e}");
    }
}

/// An ODE readout whose θ level block is indexed by a compartment amount (the index is a
/// dependent value in a recording): it stays in the readout, and used to decline the model.
/// The recorders read the block from the θ leaves and select the level on every evaluation
/// (`gather_select_g`), on the static walk, the event walk (time-varying weight) and under
/// IOV. Both levels are live (asserted by `measure`: the central amount crosses 40 within the
/// record), so a level fixed at the recording point fails the gradient.
/// Mutations: record the index as a constant (`value_read_refused` back in `PushThetaGather`)
/// -> the recording fails, `measure` fails on `served`; invert the window comparison in
/// `gather_select_g` -> the prediction is NaN.
#[test]
fn an_ode_readout_gather_indexed_by_a_state_is_served_on_every_walk() {
    let base = ODE_LAG
        .replace(
            "  theta TVLAG(0.4, 0.01, 5.0)\n",
            "  theta SCALE[2](1.0, 0.1, 10.0)\n",
        )
        .replace("  LAGTIME = TVLAG * exp(ETA_LAG)\n", "")
        .replace("  V = TVV\n", "  V = TVV * exp(ETA_LAG)\n")
        .replace(
            "  y = central / V\n",
            "  y = central / V * SCALE[if (central > 40.0) 2 else 1]\n",
        );
    for tv in [false, true] {
        let src = if tv { tv_variant(&base) } else { base.clone() };
        let m = tight(parse_model_string(&src).unwrap());
        let ro = m
            .ode_spec
            .as_ref()
            .unwrap()
            .readout_program
            .as_ref()
            .unwrap();
        assert!(
            ro.reads_leaves(),
            "premise: the gather stays in the readout"
        );
        let s = cf_readout_subject(&m, tv, None);
        let label = format!("state-indexed readout gather, tv {tv}");
        assert!(recorder_in_scope(&m), "{label}: scope");
        let (jac, grad) = measure(&m, &s, &[0, 1, 2, 3], &label);
        // Measured worst on Linux: Jacobian 5.8e-12, gradient 2.1e-9.
        assert!(jac < 5e-11 && grad < 3e-8, "{label}: {jac:.2e} {grad:.2e}");
    }
}

/// The init baseline under IOV: the impulse's amount is the BSV-only snapshot's (`κ = 0`), its
/// decay each observation's own occasion's, and the readout (curved in `V`, which carries κ)
/// reads the sum. Mutation: add the impulse after the readout in `record_analytic_events`
/// -> the stacked Jacobian gap is O(1).
#[test]
fn a_closed_form_readout_over_an_init_baseline_under_iov_matches_reconverged_fd() {
    let src = super::tests::CF_IOV
        .replace(
            "  omega ETA_CL ~ 0.09\n",
            "  theta TVC0(0.4, 0.01, 10.0)\n  omega ETA_CL ~ 0.09\n",
        )
        .replace(
            "[error_model]",
            "[initial_conditions]\n  init(central) = TVC0 * V\n\
             [scaling]\n  y = central / V + 0.2 * central / (V * V)\n[error_model]",
        );
    let m = tight(parse_model_string(&src).unwrap());
    assert!(!m.analytical_init.is_empty() && m.analytic_readout.is_some());
    let s = super::tests::iov_subject(&m);
    assert!(recorder_in_scope(&m) && served(&m, &s), "scope");
    let jac = iov_jacobian_gap(
        &m,
        &s,
        &[0.06, -0.04, 0.05, -0.03, 0.04, 0.02, -0.05, 0.03],
        "iov init + readout",
    );
    let grad = fd_oracle(&m, &s, true);
    eprintln!("init + readout × IOV: ∂f/∂[η, κ] {jac:.2e}; gradient {grad:.2e}");
    assert!(jac < 5e-10 && grad < 5e-8, "{jac:.2e} {grad:.2e}");
}

/// A hand-built spec whose readout carries no compiled program (the parser builds one for every
/// form) is outside the scope: the walks would read a zero for the observable. The guard is the
/// ODE scope's, and the closed-form readout declines the subject. Mutation: drop the
/// `programmed` clause in `ode_walk_model_scope` -> the ODE arm reads true.
#[test]
fn a_readout_without_a_compiled_program_is_declined_not_read_as_zero() {
    let mut m = tight(parse_model_string(ODE_LAG).unwrap());
    assert!(recorder_in_scope(&m), "premise");
    m.ode_spec.as_mut().unwrap().readout_program = None;
    assert!(
        matches!(
            m.ode_spec.as_ref().unwrap().readout,
            crate::ode::OdeReadout::Single(_)
        ),
        "premise: a Form C readout"
    );
    assert!(!recorder_in_scope(&m), "a Single readout without a program");
    // A readout of its own (no other test's plan is shared with it), declined before anything
    // is recorded for it; the same model with its program is served.
    let ro = "  y = 1.7 * central / V\n";
    let mut cf = cf_readout(ro);
    let s = cf_readout_subject(&cf, false, None);
    cf.analytic_readout.as_mut().unwrap().program = None;
    assert!(!served(&cf, &s), "a closed-form readout without a program");
    let cf = cf_readout(ro);
    assert!(served(&cf, &s), "premise: with its program it is served");
}

/// An ODE whose individual parameters take every usable PK slot: a θ read directly in the
/// readout then has no slot to be desugared into, stays in the readout, and used to decline the
/// model (the readout was not dual-evaluable). It is read from the θ leaves now, on the static
/// walk and the event walk (time-varying weight). Premise: the readout does read leaves, and no
/// synthetic parameter was injected. Mutation: pass empty θ to `resolve_obs_readout`'s program
/// (`readout_leaves` in `record_ode`) -> the readout reads 0 for `TVCL`, the prediction is 0 and
/// `measure` fails on a dead θ.
#[test]
fn an_ode_readout_reading_a_theta_with_every_pk_slot_taken_is_served() {
    let usable = crate::types::MAX_PK_PARAMS - crate::types::RESERVED_PK_SLOTS.len();
    let extras: String = (0..usable - 3)
        .map(|i| format!("  EXTRA{i} = TVV\n"))
        .collect();
    let src = format!(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
         theta TVKA(1.2, 0.05, 20.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.1 (sd)\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  KA = TVKA\n{extras}\
         [structural_model]\n  ode(states=[depot, central])\n[odes]\n  d/dt(depot) = -KA * depot\n  \
         d/dt(central) = KA * depot - (CL/V) * central\n[scaling]\n  y = central / V * TVCL\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n"
    );
    for tv in [false, true] {
        let src = if tv { tv_variant(&src) } else { src.clone() };
        let m = tight(parse_model_string(&src).unwrap());
        let ro = m
            .ode_spec
            .as_ref()
            .unwrap()
            .readout_program
            .as_ref()
            .unwrap();
        assert!(ro.reads_leaves(), "premise: the θ read stayed direct");
        assert!(
            !m.indiv_param_names
                .iter()
                .any(|n| n.starts_with("__ferx_ro_")),
            "premise: no synthetic parameter fit"
        );
        let s = cf_readout_subject(&m, tv, None);
        let label = format!("ODE direct θ read, every slot taken, tv {tv}");
        assert!(recorder_in_scope(&m), "{label}: scope");
        let (jac, grad) = measure(&m, &s, &[0, 1, 2], &label);
        assert!(jac < 5e-10 && grad < 3e-8, "{label}: {jac:.2e} {grad:.2e}");
    }
}

// ---- an SS dose into a route-lagged zero-order window (#2786) -------------------------

/// `zero_order(dur, lag=)` with an estimated window length and an estimated per-route lag, the
/// disposition reading `TAD` when `tad`.
fn route_lagged_zero_order_ss_model(tad: bool) -> CompiledModel {
    let clock = if tad { " * (1 + 0.05 * TAD)" } else { "" };
    let src = format!(
        r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(2.0, 0.1, 8.0)
  theta TVLAG(0.7, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_DUR ~ 0.04
  omega ETA_LAG ~ 0.02
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR * exp(ETA_DUR)
  LAG = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR, lag=LAG) - (CL/V) * central{clock}
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
"#
    );
    let m = tight(parse_model_string(&src).unwrap());
    // The route lag stays on the route: the model is not a compartment-lag model, and its
    // lowered SS dose is the one dose that carries the lag as its own.
    assert!(!m.has_lagtime() && m.has_route_absorption_lag() && m.has_dose_lag());
    m
}

/// An `SS=1` dose into a route-lagged zero-order window (lowered to an SS infusion that takes
/// the route's lag as its own), then a non-SS dose on its residual, whose window is the
/// forcing's: the plan serves the subject (`jacobian_gap` / `parity` expect it in scope, no
/// FD fallback) and its `∂f/∂η` and FOCEI gradient match reconverged FD, `η_LAG` live.
/// Mutation: the lowered dose reading the compartment lag (0) instead of the route's slot
/// (`DoseAttrMap::lagtime_for_dose` / `lag_slot_for_dose`) -> the lag η moves no prediction of
/// the SS train and the gap is O(1).
#[test]
fn ss_dose_into_a_route_lagged_zero_order_window_matches_reconverged_fd() {
    const TIMES_ALL: [f64; 12] = [
        0.4, 1.0, 3.0, 4.6, 6.0, 9.0, 11.9, 13.0, 15.0, 17.5, 20.0, 24.0,
    ];
    const TIMES_TAD: [f64; 9] = [0.4, 1.0, 3.0, 4.6, 6.0, 9.0, 11.9, 12.3, 12.6];
    for tad in [false, true] {
        let m = route_lagged_zero_order_ss_model(tad);
        let s = subject(
            &m,
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ],
            // The `TAD` cell stays inside the SS dose's first cycle (`< lag + II`): the walk's
            // `TAD` anchor does not follow an SS dose's later cycle starts (`tad_referent`'s
            // "arrival plus whole cycles"), so past the first wrap a dose of another lag
            // re-anchors differently from the value path -- the `TAD`-with-a-lagged-SS-dose gap,
            // independent of where the lag is spelled (measured: gap 2e-2 from t = 15).
            if tad { &TIMES_TAD[..] } else { &TIMES_ALL[..] },
            &[0.1, -0.08, 0.05],
        );
        let label = format!("route-lagged zero-order SS tad={tad}");
        let jac = jacobian_gap(&m, &s, &probe_eta(m.n_eta), &label);
        assert!(jac < 2e-9, "{label}: ∂f/∂η {jac:.2e}");
        // The FOCEI gradient is asserted without the `TAD` clock only: with it, the first-cycle
        // observations give ∂f/∂η 5.8e-11 but a gradient 2.4e-5 off reconverged FD (measured,
        // Linux) -- the `TAD`-with-a-lagged-SS-dose gap, open, not asserted away.
        if !tad {
            let grad = parity(&m, &s);
            eprintln!("{label}: ∂f/∂η {jac:.2e}; FOCEI gradient vs reconverged FD {grad:.2e}");
            // Measured 7.7e-8 on Linux.
            assert!(grad < 1e-6, "{label}: gradient {grad:.2e}");
        }
    }
}

// ---- Weibull / transit absorption under a compartment or route lag (AD everywhere, G5) -------

/// One-compartment ODE model absorbed through `weibull(td, beta)` or `transit(n, mtt)` with an
/// estimated `lag` of `"alag"` (compartment `ALAG1`) or `"route"` (`lag=LAG`); η on CL and on
/// the lag, so both the lag's θ and its η are live.
fn lagged_kernel_model(kernel: &str, shape: f64, lag: &str) -> String {
    let (call, extra, ind) = if kernel == "weibull" {
        (
            "weibull(td=TD, beta=BETA".to_string(),
            format!("  theta TVTD(2.0, 0.05, 24.0)\n  theta TVBETA({shape}, 0.1, 10.0)\n"),
            "  TD = TVTD\n  BETA = TVBETA\n",
        )
    } else {
        (
            "transit(n=N, mtt=MTT".to_string(),
            format!("  theta TVMTT(2.0, 0.05, 24.0)\n  theta TVN({shape}, 0.1, 10.0)\n"),
            "  MTT = TVMTT\n  N = TVN\n",
        )
    };
    let (lag_ind, call) = if lag == "alag" {
        ("  ALAG1 = TVLAG * exp(ETA_LAG)\n", format!("{call})"))
    } else {
        (
            "  LAG = TVLAG * exp(ETA_LAG)\n",
            format!("{call}, lag=LAG)"),
        )
    };
    format!(
        "[parameters]\n  theta TVCL(5.0, 0.1, 100.0)\n  theta TVV(50.0, 5.0, 500.0)\n{extra}  \
         theta TVLAG(1.0, 0.01, 5.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_LAG ~ 0.04\n  \
         sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  \
         V = TVV\n{ind}{lag_ind}[structural_model]\n  ode(obs_cmt=central, states=[central])\n\
         [odes]\n  d/dt(central) = {call} - CL/V*central\n[scaling]\n  y = central / V\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n"
    )
}

/// The subject every lagged-kernel fixture is measured on: two doses, the second landing on
/// residual drug (so the incoming side of its onset is live), observations before, across and
/// after both estimated onsets, none within 0.4 h of one.
fn lagged_kernel_subject(m: &CompiledModel) -> Subject {
    subject(
        m,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &[0.4, 1.3, 2.0, 3.0, 4.5, 6.5, 7.5, 9.0, 12.0, 18.0],
        &[0.1, -0.05],
    )
}

/// θ order of [`lagged_kernel_model`]: CL, V, TD|MTT, BETA|N, LAG.
const LAGGED_KERNEL_LIVE: &[usize] = &[0, 2, 3, 4];

/// `(kernel, shape, ∂f/∂η bound, FOCEI-gradient bound)`. Measured worst over the compartment and
/// route lag on Linux, at `tight` tolerances (jac / grad): Weibull β = 0.5 2.9e-9 / 6.6e-6 and
/// 0.9 2.1e-9 / 2.8e-8 (both in the cumulative frame), 1.5 4.4e-9 / 8.9e-8 and 3 6.0e-8 / 1.7e-7
/// (pointwise, on the anchored onset); transit n = 0.5 6.9e-10 / 9.6e-8 and 2.5 1.1e-10 / 2.1e-8
/// (pointwise, anchored). Bounds are ~10× the measured worst. Without the onset anchoring the
/// gaps are 4.9e-4 / 3.6 (β = 0.5), 7.1e-5 / 0.22 (β = 0.9), 1.7e-6 / 4.0e-4 (β = 1.5) and
/// 2.1e-6 / 2.1e-4 (n = 0.5) -- `an_unanchored_onset_loses_the_second_derivative` below.
const LAGGED_KERNEL_CASES: &[(&str, f64, f64, f64)] = &[
    ("weibull", 0.5, 3e-8, 7e-5),
    ("weibull", 0.9, 3e-8, 3e-7),
    ("weibull", 1.5, 5e-8, 1e-6),
    ("weibull", 3.0, 6e-7, 2e-6),
    ("transit", 0.5, 1e-8, 1e-6),
    ("transit", 2.5, 2e-9, 3e-7),
];

/// A Weibull (`β ∈ {0.5, 0.9, 1.5, 3}`) or transit (`n ∈ {0.5, 2.5}`) absorption under an
/// estimated compartment `ALAG1` or per-route `lag=`, each with an η on the lag, is served by
/// the plan (no fallback to finite differences) and matches the reconverged-FD objective
/// gradient and a Richardson `∂f/∂η`. The onset of these kernels is singular for `β < 2` /
/// `n < 1` (the density is infinite below 1, and its slope not integrable), so a pointwise
/// forcing's derivative with respect to the onset is the derivative of a quadrature of a singular
/// integrand. The segment after a moving onset is anchored at the true onset (`onset_shift`), so
/// the kernel is read at an age with no jet; a Weibull with `β < 1`, whose density is infinite
/// there, is delivered as absorbed mass besides (`CumulativeInputs`). The mutations that remove
/// either are the next two tests.
#[test]
fn lagged_weibull_and_transit_forcing_is_served_and_matches_reconverged_fd() {
    for &(kernel, shape, jac_bound, grad_bound) in LAGGED_KERNEL_CASES {
        for lag in ["alag", "route"] {
            let label = format!("{kernel} {shape} {lag}");
            let m = tight(parse_model_string(&lagged_kernel_model(kernel, shape, lag)).unwrap());
            let s = lagged_kernel_subject(&m);
            let theta = m.default_params.theta.clone();
            assert!(
                plan_forward(&m, &s, None, &theta, &[0.1, -0.05], terms::layout1(2)).is_some(),
                "{label}: the recorder must serve this subject (no finite-difference fallback)"
            );
            let (jac, grad) = measure(&m, &s, LAGGED_KERNEL_LIVE, &label);
            assert!(jac < jac_bound, "{label} ∂f/∂η: {jac:.2e}");
            assert!(grad < grad_bound, "{label} gradient: {grad:.2e}");
        }
    }
}

/// Runs `f` with the cumulative frame off (every dose on the pointwise forcing), restoring it.
fn with_cumulative_off<R>(f: impl FnOnce() -> R) -> R {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            crate::ode::predictions::CUMULATIVE_OFF.with(|c| c.set(false));
        }
    }
    crate::ode::predictions::CUMULATIVE_OFF.with(|c| c.set(true));
    let _restore = Restore;
    f()
}

/// Runs `f` with the onset anchoring off (the segment after a Weibull / transit onset on the
/// fixed grid), restoring it.
fn with_onset_shift_off<R>(f: impl FnOnce() -> R) -> R {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            crate::sens::ode_provider::ONSET_SHIFT_OFF.with(|c| c.set(false));
        }
    }
    crate::sens::ode_provider::ONSET_SHIFT_OFF.with(|c| c.set(true));
    let _restore = Restore;
    f()
}

/// `(∂f/∂η gap, gradient gap)` of the Weibull β = 0.5 route-lag fixture, on a fresh model.
fn singular_onset_gaps() -> (f64, f64) {
    let m = tight(parse_model_string(&lagged_kernel_model("weibull", 0.5, "route")).unwrap());
    let s = lagged_kernel_subject(&m);
    let eta = probe_eta(m.n_eta);
    (jacobian_gap(&m, &s, &eta, "mutated"), parity(&m, &s))
}

/// The mutation the parity above must see: with the cumulative frame off (the onset still
/// anchored), the stages near the onset of a `β = 0.5` Weibull read the infinite density again,
/// and the `∂f/∂η` gap is the measured 3.9e-7 against the frame's 2.9e-9 and the parity's bound of
/// 3e-8 (gradient 7.9e-4 against 6.6e-6 and 7e-5).
#[test]
fn a_cumulative_frame_off_loses_the_singular_onset() {
    let (jac, grad) = with_cumulative_off(singular_onset_gaps);
    eprintln!("G5 frame off: jac {jac:.2e} grad {grad:.2e}");
    assert!(jac > 1e-7, "∂f/∂η gap with the frame off: {jac:.2e}");
}

/// The second mutation: with the onset segment on the fixed grid (no `onset_shift`), the kernel's
/// age carries the lag's jet at stages a hair from the onset, where its slope is not integrable:
/// the second-order terms the FOCEI gradient needs are garbage -- for the Weibull `β = 0.5` (gap
/// 3.6 / 4.9e-4, measured); and the transit `n = 0.5`, which the frame does not serve, has no
/// Taylor series at its onset, so it leaves the adjoint altogether.
#[test]
fn an_unanchored_onset_loses_the_second_derivative() {
    let (jac, grad) = with_onset_shift_off(singular_onset_gaps);
    eprintln!("G5 unanchored weibull: jac {jac:.2e} grad {grad:.2e}");
    assert!(
        grad > 1e-3,
        "gradient gap with the onset unanchored: {grad:.2e}"
    );
    // The transit `n = 0.5` kernel has no Taylor series at its onset: unanchored, the recorder
    // meets the `NaN` of `rate_onset_continued` and the subject leaves the adjoint (the pre-fix
    // scope decline), instead of being served with the wrong second derivative.
    let m = tight(parse_model_string(&lagged_kernel_model("transit", 0.5, "route")).unwrap());
    let s = lagged_kernel_subject(&m);
    assert!(
        !with_onset_shift_off(|| served(&m, &s)),
        "transit n = 0.5, unanchored, must not be served"
    );
    // A fresh model: the unanchored run's failure marks `m`'s plan bad for good.
    let m = tight(parse_model_string(&lagged_kernel_model("transit", 0.5, "route")).unwrap());
    let s = lagged_kernel_subject(&m);
    assert!(served(&m, &s), "transit n = 0.5, anchored, is served");
}

/// Runs `f` with the refused-anchor clause (an event preceding the onset at its instant) on,
/// restoring it.
fn with_onset_alone_on<R>(f: impl FnOnce() -> R) -> R {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            crate::sens::ode_provider::ONSET_ALONE_ON.with(|c| c.set(false));
        }
    }
    crate::sens::ode_provider::ONSET_ALONE_ON.with(|c| c.set(true));
    let _restore = Restore;
    f()
}

/// A lagged-kernel subject whose first onset (`t = 0` plus the default lag 1.0, at `eta = 0`)
/// has a sibling event at its own instant, `t = 1`, of the kind `sibling` names. The kinds sort
/// before the onset in the walk's timeline: `"record"`, the record of a second dose at `t = 1`
/// (whose own onset is at 2); `"dose"`, a second dose at `t = 0` whose onset lands at the same
/// instant.
fn coincident_onset_subject(m: &CompiledModel, sibling: &str) -> Subject {
    let second = match sibling {
        "record" => 1.0,
        _ => 0.0,
    };
    let s = subject(
        m,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(second, 100.0, 1, 0.0, false, 0.0),
        ],
        &[0.4, 1.3, 2.4, 3.0, 4.5, 6.5, 9.0, 12.0],
        &[0.0, 0.0],
    );
    s
}

/// `(df/deta gap, inner-Hessian gap)` at `eta = 0` on [`coincident_onset_subject`]: the plan's
/// first derivative against a Richardson difference of the production predictor, and the plan's
/// exact inner Hessian against a central difference of the plan's own exact inner gradient
/// (which at `eta = +-h` has the onset off the coincidence). Both relative, `/(1+|fd|)`.
fn coincident_onset_gaps(kernel: &str, shape: f64, lag: &str, sibling: &str) -> (f64, f64) {
    let m = tight(parse_model_string(&lagged_kernel_model(kernel, shape, lag)).unwrap());
    let s = coincident_onset_subject(&m, sibling);
    let params = &m.default_params;
    let eta = [0.0, 0.0];
    assert_eq!(params.theta[4], 1.0, "fixture: the default lag is 1");
    let jac = jacobian_gap(&m, &s, &eta, "coincident onset");
    let h = crate::sens::ode_provider::adjoint::exact_inner_hessian(&m, &s, params, &eta)
        .expect("the plan serves the coincident onset");
    let g = |e: &[f64]| {
        crate::sens::ode_provider::adjoint::exact_inner_gradient(&m, &s, params, e, None)
            .expect("inner gradient")
    };
    let step = 1e-4;
    let mut worst = 0.0f64;
    for k in 0..2 {
        let (mut ep, mut em) = (eta, eta);
        ep[k] += step;
        em[k] -= step;
        let (gp, gm) = (g(&ep), g(&em));
        for i in 0..2 {
            let fd = (gp[i] - gm[i]) / (2.0 * step);
            assert!(fd.is_finite() && h[(i, k)].is_finite());
            worst = worst.max((h[(i, k)] - fd).abs() / (1.0 + fd.abs())); // nan-ok: finite asserted above
        }
    }
    (jac, worst)
}

/// G15 (3): an onset with a sibling event at its own instant is anchored like any other, and
/// exact. The fixture lands the first dose's onset (lag 1.0 at `eta = 0`) on the record of a
/// second dose at `t = 1` (`"record"`) or on the onset of a second dose given at `t = 0`
/// (`"dose"`). Measured, plan vs the reconverged production predictor (`df/deta` Richardson and
/// a central difference of the plan's own exact inner gradient for the inner Hessian), worst over
/// the siblings and the cases below: `df/deta` 3.9e-8, Hessian 2.2e-5. The clause G9 added
/// (refuse the anchor with a preceding event at the instant) sends the onset to the
/// moving-boundary crossing, whose rate-on is singular for `beta < 2`: Hessian gaps 1.6e1 /
/// 2.2e1 (`beta = 0.5`, record / dose), 1.3e2 / 1.7e2 (`1.5`), 3.9e-4 / 3.4e-4 (`3`, alag), and
/// no change for the transit `n = 2.5` (3.5e-6).
/// Mutation: the clause back on (`ONSET_ALONE_ON`) -> the Hessian bound fails for the Weibull rows.
#[test]
fn a_coincident_onset_is_anchored_with_a_sibling_at_its_instant() {
    // (kernel, shape, lag, whether the clause moves the answer past the bound)
    let cases = [
        ("weibull", 0.5, "route", true),
        ("weibull", 1.5, "route", true),
        ("weibull", 3.0, "alag", true),
        ("transit", 2.5, "alag", false),
    ];
    for sibling in ["record", "dose"] {
        for (kernel, shape, lag, moved) in cases {
            let label = format!("{sibling} {kernel} {shape} {lag}");
            let (jac, hess) = coincident_onset_gaps(kernel, shape, lag, sibling);
            assert!(jac < 5e-7, "{label}: df/deta gap {jac:.2e}");
            assert!(hess < 2e-4, "{label}: inner Hessian gap {hess:.2e}");
            let (_, mutant) =
                with_onset_alone_on(|| coincident_onset_gaps(kernel, shape, lag, sibling));
            eprintln!("G15 {label}: jac {jac:.2e} hess {hess:.2e}; clause on: hess {mutant:.2e}");
            if moved {
                assert!(
                    mutant > 2e-4,
                    "{label}: with the clause, Hessian gap {mutant:.2e}"
                );
            }
        }
    }
}

/// The cumulative frame is a reformulation of the same ODE, so the predictions move only at the
/// integrator's tolerance: the worst relative change over every fixture (only the Weibull
/// `β < 1` ones are framed, the rest are unchanged), at the `tight` tolerances (1e-10 / 1e-12)
/// the parity tests run at.
#[test]
fn the_cumulative_frame_moves_the_predictions_at_integrator_tolerance() {
    let mut worst = (0.0f64, String::new());
    for &(kernel, shape, _, _) in LAGGED_KERNEL_CASES {
        for lag in ["alag", "route"] {
            let m = tight(parse_model_string(&lagged_kernel_model(kernel, shape, lag)).unwrap());
            let s = lagged_kernel_subject(&m);
            let (theta, eta) = (m.default_params.theta.clone(), probe_eta(m.n_eta));
            let new = crate::pk::compute_predictions_with_tv(&m, &s, &theta, &eta);
            let old = with_cumulative_off(|| {
                crate::pk::compute_predictions_with_tv(&m, &s, &theta, &eta)
            });
            assert_eq!(new.len(), old.len());
            for (a, b) in new.iter().zip(&old) {
                assert!(a.is_finite() && b.is_finite(), "{kernel} {shape} {lag}");
                let rel = (a - b).abs() / b.abs().max(1e-12);
                if rel > worst.0 {
                    worst = (rel, format!("{kernel} {shape} {lag}"));
                }
            }
        }
    }
    eprintln!(
        "G5 value change: worst relative {:.2e} ({})",
        worst.0, worst.1
    );
    // Measured worst 6.8e-7 (Weibull β = 0.5, compartment lag); ~15x headroom.
    assert!(
        worst.0 < 1e-5,
        "worst relative change {:.2e} ({})",
        worst.0,
        worst.1
    );
}

/// The subject every lagged-infusion fixture is measured on: two infusions into the kernel's
/// compartment (2 h each), the second landing on the first's residual drug, an η on CL and on
/// the lag, and observations before, between and after both estimated onsets and both window
/// ends, none within 0.4 h of one.
fn lagged_infusion_subject(m: &CompiledModel) -> Subject {
    subject(
        m,
        vec![
            DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
            DoseEvent::new(6.0, 80.0, 1, 40.0, false, 0.0),
        ],
        &[
            0.4, 1.5, 2.5, 3.6, 4.5, 6.5, 7.6, 8.5, 9.6, 11.0, 14.0, 18.0,
        ],
        &[0.1, -0.05],
    )
}

/// `(kernel, shape, ∂f/∂η bound, FOCEI-gradient bound)` for a lagged infusion. Measured worst
/// over the compartment and route lag on Linux, at `tight` tolerances (jac / grad): Weibull
/// β = 0.5 9.7e-10 / 7.3e-7, 0.9 8.0e-10 / 2.3e-7, 1.5 2.6e-10 / 3.4e-8, 3 2.7e-9 / 4.2e-8;
/// transit n = 0.5 2.4e-10 / 2.4e-8, 2.5 5.7e-10 / 5.8e-8. Bounds are ~10x the measured worst.
/// Without the anchoring the gradient is off by 3.4e-3 (β = 1.5) and 1.5e-3 (n = 0.5), and a
/// Weibull with β < 1 has no continuation at its onset at all (it declines):
/// `an_unanchored_infusion_boundary_loses_the_second_derivative`.
const LAGGED_INFUSION_CASES: &[(&str, f64, f64, f64)] = &[
    ("weibull", 0.5, 1e-8, 7e-6),
    ("weibull", 0.9, 1e-8, 3e-6),
    ("weibull", 1.5, 3e-9, 4e-7),
    ("weibull", 3.0, 3e-8, 5e-7),
    ("transit", 0.5, 3e-9, 3e-7),
    ("transit", 2.5, 6e-9, 6e-7),
];

/// An infusion into a Weibull (`β ∈ {0.5, 0.9, 1.5, 3}`) or transit (`n ∈ {0.5, 2.5}`) kernel
/// under an estimated compartment `ALAG1` or per-route `lag=` is served by the plan and matches
/// the reconverged-FD objective gradient and a Richardson `∂f/∂η`. The convolved input
/// `R_in_inf = (D/T)·[G(τ) − G(τ − T)]` carries the kernel's CDF, singular where the infusion
/// starts (`G ~ τ^β` for a Weibull, `τ^(n+1)` for a transit) and again where its window ends
/// (`G(τ − T)` begins there): the segment after each is anchored at the true boundary
/// (`onset_shift`), so the kernel is read at an age with no jet. Both boundaries move with the
/// lag.
#[test]
fn a_lagged_infusion_into_a_weibull_or_transit_kernel_matches_reconverged_fd() {
    for &(kernel, shape, jac_bound, grad_bound) in LAGGED_INFUSION_CASES {
        for lag in ["alag", "route"] {
            let label = format!("infusion, {kernel} {shape} {lag}");
            let m = tight(parse_model_string(&lagged_kernel_model(kernel, shape, lag)).unwrap());
            let s = lagged_infusion_subject(&m);
            assert!(s.doses.iter().all(|d| d.is_infusion()), "{label}");
            assert!(served(&m, &s), "{label}: must be served");
            let (jac, grad) = measure(&m, &s, LAGGED_KERNEL_LIVE, &label);
            assert!(jac < jac_bound, "{label} ∂f/∂η: {jac:.2e}");
            assert!(grad < grad_bound, "{label} gradient: {grad:.2e}");
        }
    }
}

/// The mutation the parity above must see: with the boundaries unanchored (the segments after
/// the onset and the window end on the fixed grid), the kernel's age carries the lag's jet at
/// stages a hair from the singular CDF, and the gradient is off by 3.4e-3 (Weibull `β = 1.5`)
/// and 1.5e-3 (transit `n = 0.5`), measured.
#[test]
fn an_unanchored_infusion_boundary_loses_the_second_derivative() {
    for (kernel, shape) in [("weibull", 1.5), ("transit", 0.5)] {
        let m = tight(parse_model_string(&lagged_kernel_model(kernel, shape, "route")).unwrap());
        let s = lagged_infusion_subject(&m);
        let eta = probe_eta(m.n_eta);
        let (jac, grad) =
            with_onset_shift_off(|| (jacobian_gap(&m, &s, &eta, "mutated"), parity(&m, &s)));
        eprintln!("G9 unanchored {kernel} {shape}: jac {jac:.2e} grad {grad:.2e}");
        assert!(
            grad > 5e-4,
            "{kernel} {shape} unanchored: gradient gap {grad:.2e}"
        );
    }
}

/// The IOV model of [`iov_lag_subject`] with its depot replaced by an absorption `kernel`
/// (`weibull(td, beta)` or `transit(n, mtt)`) under the compartment lag `ALAG1`: a κ on CL, η on
/// CL and on the lag.
fn iov_kernel_lag_model(kernel: &str, shape: f64) -> String {
    let (call, extra, ind) = if kernel == "weibull" {
        (
            "weibull(td=TD, beta=BETA)",
            format!("  theta TVTD(2.0, 0.05, 24.0)\n  theta TVBETA({shape}, 0.1, 10.0)\n"),
            "  TD = TVTD\n  BETA = TVBETA\n",
        )
    } else {
        (
            "transit(n=N, mtt=MTT)",
            format!("  theta TVMTT(2.0, 0.05, 24.0)\n  theta TVN({shape}, 0.1, 10.0)\n"),
            "  MTT = TVMTT\n  N = TVN\n",
        )
    };
    format!(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n{extra}  \
         theta TVLAG(0.6, 0.01, 5.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_LAG ~ 0.06\n  \
         kappa KAPPA_CL ~ 0.03\n  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV\n{ind}  ALAG1 = TVLAG * exp(ETA_LAG)\n\
         [structural_model]\n  ode(obs_cmt=central, states=[central])\n\
         [odes]\n  d/dt(central) = {call} - (CL/V)*central\n[scaling]\n  y = central / V\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  iov_column = OCC\n"
    )
}

/// Three occasions, an infusion (2 h) opening each into the kernel, observations from
/// `predict_iov` at known κ, none within 0.4 h of an onset or a window end.
fn iov_infusion_subject(m: &CompiledModel) -> Subject {
    let times = [1.5, 4.0, 10.0, 13.5, 16.0, 22.0, 25.5, 28.0, 34.0];
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(24.0, 100.0, 1, 50.0, false, 0.0),
    ];
    let mut s = subject_shell(doses, &times);
    s.occasions = vec![1, 1, 1, 2, 2, 2, 3, 3, 3];
    s.dose_occasions = vec![1, 2, 3];
    let kappas = vec![vec![0.1], vec![-0.08], vec![0.05]];
    let preds = crate::pk::predict_iov(m, &s, &m.default_params.theta, &[0.1, -0.05], &kappas);
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.12 * (1.7 * j as f64).sin()))
        .collect();
    s
}

/// The IOV walk is the same event-driven walk as the non-IOV one, so a lagged infusion into a
/// Weibull / transit kernel is anchored at its onset and window end there too: served, with the
/// stacked `∂f/∂[η, κ]` and the FOCEI gradient matching reconverged FD (measured worst on
/// Linux: 1.6e-9 / 1.9e-7; the bounds are ~10x with a floor). The anchoring mutations of the
/// non-IOV test redden this one too.
#[test]
fn an_iov_lagged_infusion_into_a_weibull_or_transit_kernel_matches_reconverged_fd() {
    for (kernel, shape) in [("weibull", 0.5), ("weibull", 1.5), ("transit", 0.5)] {
        let label = format!("IOV infusion, {kernel} {shape}");
        let m = tight(parse_model_string(&iov_kernel_lag_model(kernel, shape)).unwrap());
        let s = iov_infusion_subject(&m);
        assert!(served(&m, &s), "{label}: must be served");
        let jac = iov_jacobian_gap(&m, &s, &[0.06, -0.04, 0.05, -0.03, 0.04], &label);
        let grad = fd_oracle(&m, &s, true);
        eprintln!("{label}: ∂f/∂[η, κ] vs FD {jac:.2e}; FOCEI gradient vs FD {grad:.2e}");
        assert!(jac < 2e-8, "{label} ∂f/∂[η, κ]: {jac:.2e}");
        assert!(grad < 2e-6, "{label} gradient: {grad:.2e}");
    }
}

/// One `SS=1` bolus at 0 (`II = 12`, an estimated lag of about 0.4 h) and nothing else, observed
/// across four cycles: after the lagged arrival every later cycle start is a virtual pulse that
/// `TAD` re-anchors to, with no real dose to take over the anchor.
fn ss_only_subject(m: &CompiledModel, inf: bool) -> Subject {
    let first = DoseEvent::new(0.0, 100.0, 1, if inf { 5.0 } else { 0.0 }, true, 12.0);
    subject(
        m,
        vec![first],
        &[
            0.5, 1.0, 2.0, 4.0, 8.0, 11.0, 12.2, 13.0, 15.0, 17.0, 20.0, 23.0, 25.0, 28.0, 31.0,
            36.0, 40.0,
        ],
        &[0.1, -0.05],
    )
}

/// A lagged `SS=1` dose into a `TAD`-reading right-hand side, observed past its first cycle: the
/// walk's `TAD` anchor follows every later cycle start (a virtual pulse before the record, and
/// each cycle after the lagged arrival), as `dosing::tad_referent` does for the value.
#[test]
fn a_lagged_ss_dose_re_anchors_tad_at_every_later_cycle() {
    for (factor, inf) in [
        ("1 + 0.3 * exp(-0.5 * TAD)", false),
        ("1 + 0.3 * exp(-0.5 * TAD)", true),
        ("1 + 0.01 * TIME + 0.3 * exp(-0.5 * TAD)", false),
    ] {
        let label = format!("SS-only TAD `{factor}` inf={inf}");
        let m = clock_model(factor, true);
        let s = ss_only_subject(&m, inf);
        let (jac, grad) = measure(&m, &s, &[0, 1, 2, 3], &label);
        // Measured on Linux (dev profile): dF/dEta <= 1.6e-11, FOCEI gradient <= 9.8e-8. Without
        // the per-cycle anchor the gradient is 0.1 .. 1.3 off from the first cycle on.
        assert!(
            jac < 2e-10 && grad < 1e-6,
            "{label}: {jac:.2e} / {grad:.2e}"
        );
    }
}

/// The scope clause "`indiv_param_program` absent, axis not `n_eta + n_kappa`, or no
/// `pk_indices`" (`ode_walk_model_scope`) is not reachable from a parsed model: the parser
/// rejects a model with no individual parameter (an absent, empty or comment-only
/// `[individual_parameters]` block is "Missing [individual_parameters] block"), so `pk_indices`
/// (one slot per declared individual parameter) is never empty, and it builds the ODE spec's
/// program in one place over the extended `[η, κ]` axis, for BSV and IOV models alike. Only a
/// hand-built spec can reach it, and there it must route to FD rather than record: each of the
/// three clauses declines on its own.
/// Mutations: drop the `!pk_indices.is_empty()` conjunct, or the axis comparison, or the
/// `let Some(p) = ..indiv_param_program else` -> the matching declined row turns green (and the
/// recorder would run over a spec with no parameter map).
#[test]
fn the_individual_parameter_scope_clause_is_only_reachable_by_a_hand_built_spec() {
    let base = "[parameters]\n  theta TVX(1.0, 0.1, 10.0)\n  sigma PROP_ERR ~ 0.1 (sd)\n";
    let rest = "[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  \
                d/dt(central) = -0.2 * central\n[scaling]\n  y = central / 10.0\n\
                [error_model]\n  DV ~ proportional(PROP_ERR)\n";
    for (label, ind) in [
        ("no block", ""),
        ("empty block", "[individual_parameters]\n"),
        (
            "comment-only block",
            "[individual_parameters]\n  # nothing\n",
        ),
    ] {
        let err = parse_model_string(&format!("{base}{ind}{rest}"))
            .err()
            .unwrap_or_else(|| panic!("{label}: an ODE model with no individual parameter parsed"));
        assert!(
            format!("{err:?}").contains("individual_parameters"),
            "{label}: {err:?}"
        );
    }
    // Every parsed ODE model has the program over the `[eta, kappa]` axis and a slot per parameter.
    for (label, src) in [
        ("BSV", ODE_LAG),
        ("IOV", IOV_LAG_ODE),
        ("IOV 1 kappa", ODE_IOV_ONE_KAPPA),
    ] {
        let m = tight(parse_model_string(src).unwrap());
        let p = m.ode_spec.as_ref().unwrap().indiv_param_program.as_ref();
        assert!(
            p.is_some_and(|p| p.n_eta_axis() == m.n_eta + m.n_kappa),
            "{label}: the program is built over the extended axis"
        );
        assert!(!m.pk_indices.is_empty(), "{label}: a slot per parameter");
        assert!(recorder_in_scope(&m), "{label}: control, in scope");
    }
    // The hand-built shapes decline, each on its own.
    let fresh = || tight(parse_model_string(ODE_LAG).unwrap());
    let mut no_slots = fresh();
    no_slots.pk_indices.clear();
    assert!(!recorder_in_scope(&no_slots), "no pk_indices");
    let mut no_program = fresh();
    no_program.ode_spec.as_mut().unwrap().indiv_param_program = None;
    assert!(!recorder_in_scope(&no_program), "no program");
    let mut wrong_axis = fresh();
    wrong_axis.n_eta += 1;
    assert!(!recorder_in_scope(&wrong_axis), "axis mismatch");
}

/// An SS dose into a built-in absorption kernel (`zero_order` / `first_order`, with a compartment
/// `ALAG1` or a per-route `lag=`) on a right-hand side reading `TAD`, observed across four
/// cycles: served by the plan, ∂f/∂η and the FOCEI gradient against reconverged FD. The route
/// lag of the zero-order kernel moves its window, not the SS dose's arrival, so only its
/// compartment-lag cells have the lagged-SS anchor; the route cells are the control that the
/// per-cycle anchor leaves an unlagged SS dose unchanged.
#[test]
fn ss_into_absorption_with_a_lag_on_a_tad_rhs_matches_reconverged_fd() {
    let times = [
        0.4, 4.0, 6.5, 10.5, 13.5, 16.0, 19.0, 22.5, 26.5, 28.0, 31.0, 36.5,
    ];
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    for (label, kernel, lag, lag_value, p) in [
        ("ZO ALAG1 2 (dur 3)", "zero_order", "alag", 2.0, 3.0),
        ("ZO route lag 2 (dur 3)", "zero_order", "route", 2.0, 3.0),
        ("FO ALAG1 2", "first_order", "alag", 2.0, 0.4),
        ("FO route lag 2", "first_order", "route", 2.0, 0.4),
    ] {
        let src = ss_lag_model(kernel, lag, lag_value, p).replace(
            "- CL/V*central\n",
            "- CL/V*central*(1 + 0.3*exp(-0.5*TAD))\n",
        );
        assert!(src.contains("TAD"), "{label}: premise");
        let m = tight(parse_model_string(&src).unwrap());
        let s = subject(&m, doses.clone(), &times, &[0.1, -0.05]);
        assert!(s.has_periodic_ss_dose());
        let (jac, grad) = measure(&m, &s, &[0, 2, 3], label);
        // Measured on Linux (dev profile): dF/dEta <= 4.8e-8, FOCEI gradient <= 4.2e-8 (the
        // first-order rows, whose periodic sum the value path truncates at 1e-10 relative);
        // 1.8e-2 / 0.46 off on the zero-order compartment-lag row without the per-cycle anchor.
        assert!(jac < 5e-7 && grad < 5e-7, "{label}: {jac:.2e} / {grad:.2e}");
    }
}

/// The second lag derivative of a clock-reading field through a lagged `SS=1` infusion's seed:
/// `ss_state_at_phase_g` advances the trough by `phase = II - lag` and extends the flow by the
/// phase's jet-only part, whose second-order term is `1/2 (J g + dg/dt) dt^2`. The extension ran
/// the field at a fixed clock, so for a right-hand side reading `TIME` or `TAD` (an explicit
/// `dg/dt`) it dropped `dg/dt`: every first derivative (`df/deta`, 5e-11) and the theta
/// dependence stayed exact while `d2l/deta_lag2` was off by 5e-4 (`TIME`) / 1.2e-5 (`TAD`), and
/// with it the FOCEI gradient by 4e-5 (theta and sigma coordinates alike, whatever the FD step,
/// the ODE tolerance or the EBE tolerance: a deterministic gap, not the oracle's conditioning).
/// The exact inner Hessian is checked against a Richardson difference of the exact inner
/// gradient, at an eta off the mode; the controls are the autonomous field (no `dg/dt`) and the
/// clock-reading field with no estimated lag (no phase jet), both exact before the fix.
/// Mutations: pass the step offset `s` of `extend_flow_by_dual_duration_g` through as `0` ->
/// the lagged clock rows go to 5e-4 / 1.2e-5 and the controls stay at 1e-11.
#[test]
fn a_lagged_ss_infusion_on_a_clock_reading_rhs_has_the_exact_inner_hessian() {
    let tt = "1 + 0.05 * TIME + 0.3 * exp(-0.5 * TAD)";
    for (name, factor, lag) in [
        ("autonomous", "1", true),
        ("TIME", "1 + 0.05 * TIME", true),
        ("TAD", "1 + 0.3 * exp(-0.5 * TAD)", true),
        ("TIME + TAD", tt, true),
        ("TIME + TAD, no lag", tt, false),
    ] {
        let m = with_tol(clock_model(factor, lag), 1e-12, 1e-14);
        let s = clock_subject(&m, true);
        let params = moved_params(&m);
        let mode = ebe_for(&m, &s, &params, 1e-12, true);
        let eta: Vec<f64> = mode.iter().map(|v| v + 0.03).collect();
        let hx = exact_inner_hessian(&m, &s, &params, &eta).expect("exact inner Hessian");
        let g = |e: &[f64]| exact_inner_gradient(&m, &s, &params, e, None).expect("inner gradient");
        let n = eta.len();
        let mut worst = 0.0f64;
        for k in 0..n {
            let col = |st: f64| {
                let (mut ep, mut em) = (eta.clone(), eta.clone());
                ep[k] += st;
                em[k] -= st;
                let (gp, gm) = (g(&ep), g(&em));
                (0..n)
                    .map(|j| (gp[j] - gm[j]) / (2.0 * st))
                    .collect::<Vec<f64>>()
            };
            let (c1, c2) = (col(1e-4), col(2e-4));
            for j in 0..n {
                let fd = (4.0 * c1[j] - c2[j]) / 3.0;
                assert!(
                    hx[(j, k)].is_finite() && fd.is_finite(),
                    "{name}: H[{j},{k}]"
                );
                worst = worst.max((hx[(j, k)] - fd).abs() / (1.0 + fd.abs()));
            }
        }
        // Measured on Linux (dev profile): 5.1e-12 .. 5.0e-10 on every row; the unfixed
        // extension reads 5.3e-4 (TIME) and 1.2e-5 (TAD) on the lagged clock rows.
        assert!(
            worst < 5e-9,
            "{name}: d2l/deta2 vs FD of the exact gradient {worst:.2e}"
        );
    }
}

/// An SS dose into a lagged built-in absorption kernel (η on the lag and on `DUR`/`KA`) on a
/// right-hand side reading `TIME`: the exact inner Hessian against a Richardson difference of
/// the exact inner gradient, the autonomous rows the control. The zero-order SS dose is lowered
/// to an infusion, and the run-in's quiet leg re-opens its `TIME` at the window's end, which
/// moves with `DUR`; the plan ran that leg's clock from the window's value, losing
/// `dTIME/dT_inf` -- `H[CL, DUR]` read -60.6 against FD's -167 (0.63 relative). Mutation: the
/// quiet legs' offset `-dtinf` / anchor `-inf_window` back to `0` / `-T_inf` -> the zero-order
/// `TIME` rows go to 0.63. The first-order rows (a lagged onset crossed inside the cycle by
/// `equilibrate_ss_input_rate_state_g`) are exact as they stand: both sides of that switch share
/// the base field, so its clock dependence cancels in the crossing.
/// Measured on Linux (dev profile): every row <= 1.4e-9.
#[test]
fn a_lagged_ss_absorption_kink_on_a_clock_reading_rhs_has_the_exact_inner_hessian() {
    let times = [0.4, 2.5, 4.0, 6.5, 10.5, 13.5, 16.0, 19.0, 22.5];
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    let mut bad = Vec::new();
    for (kernel, lag, p) in [
        ("zero_order", "alag", 3.0),
        ("zero_order", "route", 3.0),
        ("first_order", "alag", 0.4),
        ("first_order", "route", 0.4),
    ] {
        for (clock, factor) in [("autonomous", "1"), ("TIME", "(1 + 0.05*TIME)")] {
            let name = format!("{kernel} {lag} {clock}");
            let src = ss_lag_model(kernel, lag, 2.0, p)
                .replace("- CL/V*central\n", &format!("- CL/V*central*{factor}\n"))
                .replace("= TVLAG\n", "= TVLAG * exp(ETA_L)\n")
                .replace("  sigma PROP_ERR", "  omega ETA_L ~ 0.04\n  sigma PROP_ERR");
            assert!(src.contains("ETA_L)"), "{name}: premise, η on the lag");
            let m = with_tol(tight(parse_model_string(&src).unwrap()), 1e-12, 1e-14);
            let s = subject(&m, doses.clone(), &times, &[0.1, -0.05, 0.08]);
            let params = moved_params(&m);
            let mode = ebe_for(&m, &s, &params, 1e-12, true);
            let eta: Vec<f64> = mode.iter().map(|v| v + 0.03).collect();
            let hx = exact_inner_hessian(&m, &s, &params, &eta).expect("exact inner Hessian");
            let g =
                |e: &[f64]| exact_inner_gradient(&m, &s, &params, e, None).expect("inner gradient");
            let n = eta.len();
            let mut worst = 0.0f64;
            for k in 0..n {
                let col = |st: f64| {
                    let (mut ep, mut em) = (eta.clone(), eta.clone());
                    ep[k] += st;
                    em[k] -= st;
                    let (gp, gm) = (g(&ep), g(&em));
                    (0..n)
                        .map(|j| (gp[j] - gm[j]) / (2.0 * st))
                        .collect::<Vec<f64>>()
                };
                let (c1, c2) = (col(1e-4), col(2e-4));
                for j in 0..n {
                    let fd = (4.0 * c1[j] - c2[j]) / 3.0;
                    assert!(
                        hx[(j, k)].is_finite() && fd.is_finite(),
                        "{name}: H[{j},{k}]"
                    );
                    let e = (hx[(j, k)] - fd).abs() / (1.0 + fd.abs());
                    assert!(e.is_finite(), "{name}: H[{j},{k}] gap");
                    if e > 1e-6 {
                        eprintln!("{name}: H[{j},{k}] exact {:.6e} fd {fd:.6e}", hx[(j, k)]);
                    }
                    worst = worst.max(e);
                }
            }
            eprintln!("{name}: d2l/deta2 vs FD of the exact gradient {worst:.2e}");
            if worst >= 5e-9 {
                bad.push(format!("{name}: {worst:.2e}"));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "d2l/deta2 vs FD of the exact gradient: {bad:?}"
    );
}

/// #3004: `examples/per_route_lag_absorption.ferx` at its **start values** — the point the
/// outer optimizer's pre-flights evaluate the exact gradient, `R` diagonal and Hessian at —
/// with an observation exactly ON the delayed route's lagged arrival (`TVLAG2 = 3`, `t = 3`),
/// one either side of it, and a second dose on residual drug. The arrival is a kink in the
/// lag; the exact derivative follows the value path's convention, so the gradient, the full
/// Hessian and the diagonal along every packed axis are finite under both objectives (the
/// report of a non-finite start-value Hessian did not reproduce on the PR head), and the
/// Hessian matches the reconverged FD oracle at a point off the kink (`hessian_gaps_for`,
/// which moves the θ).
#[test]
fn per_route_lag_example_start_values_are_exact_and_finite() {
    let src = std::fs::read_to_string("examples/per_route_lag_absorption.ferx").unwrap();
    let m = parse_model_string(&src).unwrap();
    assert!(m.has_route_absorption_lag());
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [0.25, 1.0, 2.9, 3.0, 3.1, 4.0, 8.0, 13.0, 15.1, 15.5, 24.0];
    let s = subject(&m, doses.clone(), &times, &[0.1, -0.05]);
    let params = m.default_params.clone();
    let x = pack_params(&params);
    assert_eq!(
        params.theta[5], 3.0,
        "the start value sits on the observation"
    );
    let pt = PackedPoint::new(&params, &x).unwrap();
    let axes: Vec<usize> = (0..x.len()).collect();
    for obj in [Objective::Focei, Objective::Foce] {
        let eta = ebe_for(&m, &s, &params, 1e-10, obj != Objective::Foce);
        let eta = eta.as_slice();
        let g = subject_adjoint_gradient(&m, &s, &params, &x, eta, obj).expect("gradient served");
        let h = subject_adjoint_cov_hessian(&m, &s, &params, &x, eta, obj).expect("H served");
        let d = subject_adjoint_cov_diag_at(&m, &s, &pt, eta, obj, &axes).expect("diag served");
        assert!(g.iter().all(|v| v.is_finite()), "{obj:?} gradient {g:?}");
        assert!(h.iter().all(|v| v.is_finite()), "{obj:?} Hessian");
        assert!(d.iter().all(|v| v.is_finite()), "{obj:?} diagonal {d:?}");
        for (k, &a) in axes.iter().enumerate() {
            let gap = (d[k] - h[(a, a)]).abs() / (1.0 + h[(a, a)].abs());
            assert!(gap < 1e-9, "{obj:?} diag[{a}] {} vs H {}", d[k], h[(a, a)]);
        }
    }
    let m = with_tol(m, 1e-12, 1e-14);
    let off_kink = [0.25, 1.0, 2.5, 3.5, 4.0, 8.0, 13.0, 15.5, 24.0];
    let s = subject(&m, doses, &off_kink, &[0.1, -0.05]);
    let (gap, _) = hessian_gaps_for(&m, &s, Objective::Focei);
    // Measured 1.6e-9 on Linux; ~10x headroom.
    assert!(gap < 2e-8, "Hessian vs reconverged FD: {gap:.2e}");
}
