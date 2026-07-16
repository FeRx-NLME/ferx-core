//! Equivalence: the analytic `pk one_cpt_ig` / `pk two_cpt_ig` closed form (#790)
//! must match a numerical ODE with the same Freijer & Post `igd(mat, cv2)` forcing
//! into central, at eta = 0, across the dosing modes IG supports — single +
//! multiple bolus, with an optional absorption lag and bioavailability, for both
//! 1-cpt and 2-cpt disposition. Steady-state / IOV / time-varying-covariate /
//! infusion doses are rejected at validation, exercised by the `*_rejected` tests.
//!
//! The ODE twin's `igd()` forcing is itself NONMEM-anchored
//! (`tests/igd_nonmem_anchor.rs`), so matching it transitively anchors the analytic
//! exponential-tilting closed form to NONMEM — the same playbook as
//! `tests/transit_analytic_equivalence.rs`.
//!
//! `predict()` evaluates at eta = 0, so these check the structural closed form; a
//! `fit()`-level check would conflate it with the estimator.
//!
//! All parameter sets here are **in the tilting convergence domain**
//! (`ke = CL/V < 1/(2·MAT·CV²)`, and `α < 1/(2·MAT·CV²)` for 2-cpt), so the closed
//! form is actually exercised. The flip-flop (out-of-domain) reroute to the ODE twin
//! is checked separately in `ig_flip_flop_reroutes_to_ode_twin`.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{DoseEvent, EstimationMethod, FitOptions, Population};
use ferx_core::{fit, predict};

mod common;

/// The RK45 solver defaults to `abstol = 1e-6`, `reltol = 1e-4`. The IG density's
/// essential singularity at `tad → 0` makes the ODE forcing a touch harder to
/// integrate near onset than transit's smooth Gamma, so the equivalence bound is a
/// little looser than the transit suite's, with an absolute floor for the near-zero
/// early-absorption predictions.
const ATOL: f64 = 5e-5;
const RTOL: f64 = 5e-4;
/// Multi-dose trajectories accumulate per-step solver error across dose restarts.
const ACCUM_RTOL: f64 = 3e-3;

/// Build the (analytic, ODE) `.ferx` pair for a **1-cpt** IG model. In-domain
/// params: `ke = 5/50 = 0.1 < 1/(2·2·0.3) = 0.833`.
fn build_pair_1cpt(lag: bool, fbio: bool) -> (String, String) {
    let mut thetas = String::from(
        "  theta TVCL(5.0, 0.1, 100.0)\n  \
         theta TVV(50.0, 5.0, 500.0)\n  \
         theta TVMAT(2.0, 0.05, 24.0)\n  \
         theta TVCV2(0.3, 0.001, 10.0)\n",
    );
    let mut indiv =
        String::from("  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  MAT = TVMAT\n  CV2 = TVCV2\n");
    let mut pk_extra = String::new();
    if lag {
        thetas.push_str("  theta TVLAG(0.3, 0.0, 5.0)\n");
        indiv.push_str("  LAGTIME = TVLAG\n");
        pk_extra.push_str(", lagtime=LAGTIME");
    }
    if fbio {
        thetas.push_str("  theta TVF(0.7, 0.01, 1.0)\n");
        indiv.push_str("  F = TVF\n");
        pk_extra.push_str(", f=F");
    }
    let header = format!(
        "[parameters]\n{thetas}  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01 (sd)\n\n\
         [individual_parameters]\n{indiv}\n"
    );
    let analytical = format!(
        "{header}[structural_model]\n  \
         pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2{pk_extra})\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let ode = format!(
        "{header}[structural_model]\n  ode(obs_cmt=central, states=[central])\n\n\
         [odes]\n  d/dt(central) = igd(mat=MAT, cv2=CV2) - (CL/V) * central\n\n\
         [scaling]\n  obs_scale = V\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    (analytical, ode)
}

/// Build the (analytic, ODE) `.ferx` pair for a **2-cpt** IG model. In-domain:
/// `α ≈ 0.30 < 1/(2·1.5·0.3) = 1.111`.
fn build_pair_2cpt(lag: bool, fbio: bool) -> (String, String) {
    let mut thetas = String::from(
        "  theta TVCL(5.0, 0.1, 100.0)\n  \
         theta TVV1(50.0, 5.0, 500.0)\n  \
         theta TVQ(10.0, 0.1, 200.0)\n  \
         theta TVV2(100.0, 5.0, 1000.0)\n  \
         theta TVMAT(1.5, 0.05, 24.0)\n  \
         theta TVCV2(0.3, 0.001, 10.0)\n",
    );
    let mut indiv = String::from(
        "  CL = TVCL * exp(ETA_CL)\n  V1 = TVV1\n  Q = TVQ\n  V2 = TVV2\n  MAT = TVMAT\n  CV2 = TVCV2\n",
    );
    let mut pk_extra = String::new();
    if lag {
        thetas.push_str("  theta TVLAG(0.3, 0.0, 5.0)\n");
        indiv.push_str("  LAGTIME = TVLAG\n");
        pk_extra.push_str(", lagtime=LAGTIME");
    }
    if fbio {
        thetas.push_str("  theta TVF(0.7, 0.01, 1.0)\n");
        indiv.push_str("  F = TVF\n");
        pk_extra.push_str(", f=F");
    }
    let header = format!(
        "[parameters]\n{thetas}  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01 (sd)\n\n\
         [individual_parameters]\n{indiv}\n"
    );
    let analytical = format!(
        "{header}[structural_model]\n  \
         pk two_cpt_ig(cl=CL, v1=V1, q=Q, v2=V2, mat=MAT, cv2=CV2{pk_extra})\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let ode = format!(
        "{header}[structural_model]\n  ode(obs_cmt=central, states=[central, periph])\n\n\
         [odes]\n  d/dt(central) = igd(mat=MAT, cv2=CV2) - (CL/V1 + Q/V1) * central + (Q/V2) * periph\n  \
         d/dt(periph) = (Q/V1) * central - (Q/V2) * periph\n\n\
         [scaling]\n  obs_scale = V1\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    (analytical, ode)
}

/// One subject; the dose enters compartment 1 (the IG input for both forms).
fn population(doses: Vec<DoseEvent>, obs_times: Vec<f64>) -> Population {
    let n = obs_times.len();
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            doses,
            obs_times,
            vec![0.0; n],
            vec![2; n],
        )],
    }
}

fn bolus(time: f64, amt: f64) -> DoseEvent {
    DoseEvent::new(time, amt, 1, 0.0, false, 0.0)
}

fn assert_equiv(an_src: &str, ode_src: &str, label: &str, pop: &Population, rtol: f64) {
    let an = parse_full_model(an_src)
        .unwrap_or_else(|e| panic!("[{label}] analytic IG did not parse: {e}"))
        .model;
    let ode = parse_full_model(ode_src)
        .unwrap_or_else(|e| panic!("[{label}] ODE IG did not parse: {e}"))
        .model;
    // The analytic model must actually be the closed form (not silently an ODE).
    assert!(
        an.ode_spec.is_none(),
        "[{label}] analytic model must be a closed form, not an ODE"
    );

    let pa = predict(&an, pop, &an.default_params);
    let po = predict(&ode, pop, &ode.default_params);
    assert_eq!(pa.len(), po.len(), "[{label}] prediction count mismatch");
    assert!(!pa.is_empty(), "[{label}] produced no predictions");
    // At least one non-trivial prediction (guards against a degenerate all-zero match).
    assert!(
        pa.iter().any(|x| x.pred.abs() > 1e-6),
        "[{label}] all predictions ~0 — closed form may be clamping (flip-flop?)"
    );
    for (x, y) in pa.iter().zip(po.iter()) {
        let tol = ATOL + rtol * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "[{label}] t={:.3}: analytic PRED {:.6} vs ODE PRED {:.6} (|diff| {:.2e} > tol {:.2e})",
            x.time,
            x.pred,
            y.pred,
            (x.pred - y.pred).abs(),
            tol
        );
    }
}

// ── 1-cpt ────────────────────────────────────────────────────────────────────

#[test]
fn ig_1cpt_single_dose_matches_ode() {
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    );
    let (a, o) = build_pair_1cpt(false, false);
    assert_equiv(&a, &o, "1cpt-single", &pop, RTOL);
}

#[test]
fn ig_1cpt_multidose_matches_ode() {
    let doses = vec![bolus(0.0, 100.0), bolus(12.0, 100.0), bolus(24.0, 100.0)];
    let obs = vec![0.5, 2.0, 6.0, 11.5, 13.0, 18.0, 23.5, 26.0, 36.0, 48.0];
    let (a, o) = build_pair_1cpt(false, false);
    assert_equiv(
        &a,
        &o,
        "1cpt-multidose",
        &population(doses, obs),
        ACCUM_RTOL,
    );
}

#[test]
fn ig_1cpt_lag_and_fbio_match_ode() {
    let pop = population(
        vec![bolus(0.0, 100.0), bolus(12.0, 100.0)],
        vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.5, 14.0, 18.0, 24.0],
    );
    let (a, o) = build_pair_1cpt(true, true);
    assert_equiv(&a, &o, "1cpt-lag+f", &pop, ACCUM_RTOL);
}

/// #735: a FLIP-FLOP inverse-Gaussian model carrying `lagtime=` and `f=` now auto-routes to
/// its ODE `igd()` twin (instead of being rejected) and matches the hand-written ODE
/// reference with the same lag/F. Flip-flop: ke = CL/V = 5/4 = 1.25 ≥ 1/(2·MAT·CV²) =
/// 1/(2·2·0.3) = 0.833.
#[test]
fn ig_flipflop_lag_f_autoroutes_and_matches_twin() {
    let header = "\
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(4.0, 0.1, 500.0)
  theta TVMAT(2.0, 0.05, 24.0)
  theta TVCV2(0.3, 0.001, 10.0)
  theta TVLAG(0.4, 0.0, 5.0)
  theta TVF(0.7, 0.01, 1.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  MAT = TVMAT
  CV2 = TVCV2
  LAGTIME = TVLAG
  F = TVF
";
    let an_src = format!(
        "{header}\n[structural_model]\n  \
         pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2, lagtime=LAGTIME, f=F)\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let ode_src = format!(
        "{header}\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n\n\
         [odes]\n  d/dt(central) = igd(mat=MAT, cv2=CV2) - (CL/V) * central\n\n\
         [scaling]\n  obs_scale = V\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let an = parse_full_model(&an_src).expect("analytic IG parses").model;
    assert!(
        an.absorption_ode_equivalent.is_some(),
        "a flip-flop lag/f IG closed form must now auto-route to an ODE twin (#735)"
    );
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    );
    let pa = predict(&an, &pop, &an.default_params);
    assert!(
        pa.iter().map(|p| p.pred).fold(0.0_f64, f64::max) > 0.01,
        "the IG twin must produce a non-zero profile (not the closed form's flip-flop clamp)"
    );
    assert_equiv(&an_src, &ode_src, "ig-flipflop-lag+f", &pop, ACCUM_RTOL);
}

/// #735: the IG alias path — a non-canonically-named mapping (`lagtime=LAG`, `f=FBIO`) must route
/// into the twin's reserved slots via the emitted `lagtime = LAG` / `f = FBIO` alias lines, so it
/// predicts identically to the same flip-flop IG model whose params ARE named canonically
/// (`LAGTIME`/`F`, which self-route). If the alias were missing (or the `is_ig` branch diverged),
/// LAG/FBIO would not reach the F/lagtime slots and the profiles would diverge. Mirrors the
/// transit alias test so IG alias emission is not left unexercised.
#[test]
fn ig_flipflop_noncanonical_lag_f_alias_matches_canonical() {
    let mk = |lag_name: &str, f_name: &str| {
        format!(
            "[parameters]\n  \
             theta TVCL(5.0, 0.1, 100.0)\n  theta TVV(4.0, 0.1, 500.0)\n  \
             theta TVMAT(2.0, 0.05, 24.0)\n  theta TVCV2(0.3, 0.001, 10.0)\n  \
             theta TVLAG(0.4, 0.0, 5.0)\n  theta TVF(0.7, 0.01, 1.0)\n  \
             omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01 (sd)\n\n\
             [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  \
             MAT = TVMAT\n  CV2 = TVCV2\n  {lag_name} = TVLAG\n  {f_name} = TVF\n\n\
             [structural_model]\n  \
             pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2, lagtime={lag_name}, f={f_name})\n\n\
             [error_model]\n  DV ~ proportional(PROP)\n"
        )
    };
    let canonical = parse_full_model(&mk("LAGTIME", "F"))
        .expect("canonical parses")
        .model;
    let aliased = parse_full_model(&mk("LAG", "FBIO"))
        .expect("aliased parses")
        .model;
    assert!(aliased.absorption_ode_equivalent.is_some());
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    );
    let pc = predict(&canonical, &pop, &canonical.default_params);
    let pl = predict(&aliased, &pop, &aliased.default_params);
    assert_eq!(pc.len(), pl.len());
    assert!(pc.iter().map(|p| p.pred).fold(0.0_f64, f64::max) > 0.01);
    for (x, y) in pc.iter().zip(pl.iter()) {
        assert!(
            (x.pred - y.pred).abs() <= ATOL + ACCUM_RTOL * x.pred.abs(),
            "t={:.3}: canonical-name {:.6} vs aliased-name {:.6} — IG alias failed to route lag/f",
            x.time,
            x.pred,
            y.pred
        );
    }
}

// ── 2-cpt ────────────────────────────────────────────────────────────────────

#[test]
fn ig_2cpt_single_dose_matches_ode() {
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    );
    let (a, o) = build_pair_2cpt(false, false);
    assert_equiv(&a, &o, "2cpt-single", &pop, RTOL);
}

#[test]
fn ig_2cpt_multidose_lag_fbio_match_ode() {
    let doses = vec![bolus(0.0, 100.0), bolus(12.0, 100.0), bolus(24.0, 100.0)];
    let obs = vec![0.5, 2.0, 6.0, 11.5, 13.0, 18.0, 24.5, 30.0, 36.0, 48.0];
    let (a, o) = build_pair_2cpt(true, true);
    assert_equiv(
        &a,
        &o,
        "2cpt-multi-lag+f",
        &population(doses, obs),
        ACCUM_RTOL,
    );
}

// ── Flip-flop reroute + TIME desugar ──────────────────────────────────────────

/// A plain IG model whose typical parameters fall **outside** the tilting domain
/// (`ke ≥ 1/(2·MAT·CV²)`) is transparently rerouted to its ODE `igd()` twin, so the
/// prediction is still correct (not the closed form's spurious zero). Here
/// `ke = 5/10 = 0.5` and `1/(2·6·1.8) = 0.046`, deep in the flip-flop regime.
#[test]
fn ig_flip_flop_reroutes_to_ode_twin() {
    let header = "\
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(10.0, 1.0, 500.0)
  theta TVMAT(6.0, 0.05, 24.0)
  theta TVCV2(1.8, 0.001, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  MAT = TVMAT
  CV2 = TVCV2
";
    let analytic = format!(
        "{header}\n[structural_model]\n  pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2)\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let hand_ode = format!(
        "{header}\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n\n\
         [odes]\n  d/dt(central) = igd(mat=MAT, cv2=CV2) - (CL/V) * central\n\n\
         [scaling]\n  obs_scale = V\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let an = parse_full_model(&analytic)
        .expect("analytic IG parses")
        .model;
    let hd = parse_full_model(&hand_ode).expect("hand ODE parses").model;
    // The plain form stays closed-form but carries the ODE twin the flip-flop
    // reroute uses per subject.
    assert!(
        an.ode_spec.is_none() && an.absorption_ode_equivalent.is_some(),
        "plain IG carries an ODE twin (primary stays closed-form)"
    );
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    );
    let pa = predict(&an, &pop, &an.default_params);
    let ph = predict(&hd, &pop, &hd.default_params);
    assert_eq!(pa.len(), ph.len());
    assert!(!pa.is_empty());
    // Non-degenerate (the reroute avoids the closed form's spurious zero).
    assert!(
        pa.iter().any(|x| x.pred.abs() > 1e-6),
        "reroute produced all-zero"
    );
    for (x, y) in pa.iter().zip(ph.iter()) {
        let tol = ATOL + ACCUM_RTOL * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "flip-flop reroute t={:.3}: analytic {:.6} vs hand ODE {:.6}",
            x.time,
            x.pred,
            y.pred
        );
    }
}

/// An IG model with a mid-profile `TIME` switch on CL desugars to its ODE `igd()`
/// twin (the closed form can't hold a mid-window parameter switch). Written by hand
/// as that same ODE, the two must predict identically. Params in-domain.
#[test]
fn ig_time_desugar_matches_hand_written_ode() {
    let header = "\
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVCL_LATE(8.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMAT(2.0, 0.05, 24.0)
  theta TVCV2(0.3, 0.001, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  if (TIME > 6.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV
  MAT = TVMAT
  CV2 = TVCV2
";
    let shorthand = format!(
        "{header}\n[structural_model]\n  pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2)\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let hand_ode = format!(
        "{header}\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n\n\
         [odes]\n  d/dt(central) = igd(mat=MAT, cv2=CV2) - (CL/V) * central\n\n\
         [scaling]\n  obs_scale = V\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let sh = parse_full_model(&shorthand)
        .expect("shorthand IG+TIME parses")
        .model;
    let hd = parse_full_model(&hand_ode).expect("hand ODE parses").model;
    assert!(
        sh.ode_spec.is_none() && sh.absorption_ode_equivalent.is_some(),
        "IG + TIME shorthand carries an ODE equivalent (primary stays closed-form)"
    );
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![0.5, 2.0, 4.0, 5.9, 6.1, 8.0, 12.0, 24.0],
    );
    let ps = predict(&sh, &pop, &sh.default_params);
    let ph = predict(&hd, &pop, &hd.default_params);
    assert_eq!(ps.len(), ph.len());
    assert!(!ps.is_empty());
    for (x, y) in ps.iter().zip(ph.iter()) {
        assert!(
            (x.pred - y.pred).abs() <= 1e-9 + 1e-9 * x.pred.abs(),
            "t={:.3}: desugared {:.6} vs hand ODE {:.6}",
            x.time,
            x.pred,
            y.pred
        );
    }
}

// ── Sensitivity path + FOCEI likelihood (drives run_obs, not just predict) ─────

/// Fixed-parameter population OFV (FOCEI marginal NLL) at several eta values: the
/// analytic IG closed form and its ODE twin must agree, for both 1- and 2-cpt.
/// Unlike the eta=0 `predict()` sweeps above, this drives the analytic
/// **sensitivity** path (`run_obs` IG branch, the exact `∂f/∂{cl,v,mat,cv2,η}` jets
/// via `one_cpt_ig_conc_g` / `two_cpt_ig_conc_g`) and adds a likelihood-level anchor.
/// A fixed-parameter NLL (not a converged fit), so optimiser path / solver noise
/// can't confound it.
#[test]
fn ig_ofv_matches_ode() {
    use ferx_core::stats::likelihood::individual_nll;
    use ferx_core::CompiledModel;
    const OFV_RTOL: f64 = 3e-3;

    let obs_t = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    for (label, (an_src, ode_src)) in [
        ("1cpt", build_pair_1cpt(false, false)),
        ("2cpt", build_pair_2cpt(false, false)),
    ] {
        let an = parse_full_model(&an_src).unwrap().model;
        let ode = parse_full_model(&ode_src).unwrap().model;

        // DV from the analytic PRED at eta=0, scaled per subject so the NLL has real
        // (non-degenerate) residuals to weigh.
        let dose = || vec![bolus(0.0, 100.0)];
        let base = population(dose(), obs_t.clone());
        let preds = predict(&an, &base, &an.default_params);
        let n = obs_t.len();
        let mut subjects = Vec::new();
        for (i, fac) in [0.85_f64, 1.0, 1.15].into_iter().enumerate() {
            let mut s = common::subject(
                &format!("{}", i + 1),
                dose(),
                obs_t.clone(),
                vec![0.0; n],
                vec![2; n],
            );
            s.observations = preds.iter().map(|p| (p.pred * fac).max(1e-6)).collect();
            subjects.push(s);
        }
        let pop = Population { subjects, ..base };

        let pop_nll = |m: &CompiledModel, eta: f64| -> f64 {
            let p = &m.default_params;
            pop.subjects
                .iter()
                .map(|s| individual_nll(m, s, &p.theta, &[eta], &p.omega, &p.sigma.values))
                .sum::<f64>()
        };
        for eta in [0.0, 0.3, -0.3] {
            let a = pop_nll(&an, eta);
            let o = pop_nll(&ode, eta);
            let rel = (a - o).abs() / a.abs().max(1.0);
            assert!(
                rel <= OFV_RTOL,
                "[{label}] IG OFV mismatch at eta={eta}: analytic {a:.6} vs ODE {o:.6} (rel {rel:.2e})"
            );
        }
    }
}

/// A converged FOCEI fit of the analytic 1-cpt IG closed form must run and return a
/// finite OFV (exercises the full outer/inner optimiser through the closed-form
/// sensitivity path, in-domain). Data are the analytic PRED at eta=0, mildly
/// perturbed so the objective is non-degenerate.
#[test]
fn ig_fit_runs_and_converges() {
    let obs_t = vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let (an_src, _) = build_pair_1cpt(false, false);
    let model = parse_full_model(&an_src).unwrap().model;
    let base = population(vec![bolus(0.0, 100.0)], obs_t.clone());
    let sims = ferx_core::simulate_with_seed(&model, &base, &model.default_params, 1, 4321);
    let mut pop = base;
    for s in pop.subjects.iter_mut() {
        s.observations = sims
            .iter()
            .filter(|x| x.id == s.id)
            .map(|x| x.outcome.continuous_value().max(1e-6))
            .collect();
    }
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.run_covariance_step = false;
    let r = fit(&model, &pop, &model.default_params, &opts).expect("analytic IG fit ok");
    assert!(
        r.ofv.is_finite(),
        "IG fit OFV must be finite, got {}",
        r.ofv
    );
}

// ── Restrictions: features the closed form does not support are rejected up front ─

/// `fit()` the basic analytic 1-cpt IG model on `pop`, expecting an early `Err`.
fn ig_fit_err(pop: &Population) -> String {
    let (an_src, _) = build_pair_1cpt(false, false);
    let model = parse_full_model(&an_src).expect("IG model parses").model;
    fit(&model, pop, &model.default_params, &FitOptions::default())
        .expect_err("fit should reject the unsupported IG configuration")
}

/// Steady-state (`SS=1`) dosing on a closed-form inverse-Gaussian model reroutes to its
/// `igd()` ODE twin (#719) rather than being rejected: the twin equilibrates the dose
/// through the IG absorption kernel. The rerouted analytic path must match the hand-written
/// `igd()` ODE at steady state (eta = 0).
#[test]
fn ig_steady_state_matches_ode() {
    let (an_src, ode_src) = build_pair_1cpt(false, false);
    let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
    let pop = population(vec![ss], vec![1.0, 4.0, 8.0, 11.5]);
    assert_equiv(&an_src, &ode_src, "ig-ss", &pop, ACCUM_RTOL);
}

#[test]
fn ig_infusion_dose_rejected() {
    let inf = DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0); // rate > 0 → infusion
    let e = ig_fit_err(&population(vec![inf], vec![1.0, 4.0, 8.0]));
    assert!(
        e.contains("infusion"),
        "expected an infusion-rejection message, got: {e}"
    );
}

#[test]
fn ig_non_depot_dose_rejected() {
    let d = DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0); // dose into CMT 2 (non-depot)
    let e = ig_fit_err(&population(vec![d], vec![1.0, 4.0, 8.0]));
    assert!(
        e.contains("non-depot") || e.contains("CMT"),
        "expected a non-depot dose rejection message, got: {e}"
    );
}

// ── IOV (#719): the analytic IG closed form now serves inter-occasion variability by routing
//    IOV subjects to its `igd()` ODE twin, which integrates cross-occasion dose carryover (#104)
//    exactly and is the #486/#663-validated analytic IOV path. Mirrors the transit IOV tests. ──

/// Build the (analytic, ODE) `.ferx` pair for a 1-cpt IG model carrying IOV on CL (`KAPPA_CL`).
fn build_iov_pair_1cpt() -> (String, String) {
    let header = "[parameters]\n  \
        theta TVCL(5.0, 0.1, 100.0)\n  \
        theta TVV(50.0, 5.0, 500.0)\n  \
        theta TVMAT(2.0, 0.05, 24.0)\n  \
        theta TVCV2(0.3, 0.001, 10.0)\n  \
        omega ETA_CL ~ 0.09\n  \
        kappa KAPPA_CL ~ 0.04\n  \
        sigma PROP ~ 0.01 (sd)\n\n\
        [individual_parameters]\n  \
        CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV\n  MAT = TVMAT\n  CV2 = TVCV2\n\n";
    let analytical = format!(
        "{header}[structural_model]\n  \
         pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2)\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let ode = format!(
        "{header}[structural_model]\n  ode(obs_cmt=central, states=[central])\n\n\
         [odes]\n  d/dt(central) = igd(mat=MAT, cv2=CV2) - (CL/V) * central\n\n\
         [scaling]\n  obs_scale = V\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    (analytical, ode)
}

/// Build the (analytic, ODE) `.ferx` pair for a 2-cpt IG model carrying IOV on CL (`KAPPA_CL`).
fn build_iov_pair_2cpt() -> (String, String) {
    let header = "[parameters]\n  \
        theta TVCL(5.0, 0.1, 100.0)\n  \
        theta TVV1(50.0, 5.0, 500.0)\n  \
        theta TVQ(10.0, 0.1, 200.0)\n  \
        theta TVV2(100.0, 5.0, 1000.0)\n  \
        theta TVMAT(1.5, 0.05, 24.0)\n  \
        theta TVCV2(0.3, 0.001, 10.0)\n  \
        omega ETA_CL ~ 0.09\n  \
        kappa KAPPA_CL ~ 0.04\n  \
        sigma PROP ~ 0.01 (sd)\n\n\
        [individual_parameters]\n  \
        CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V1 = TVV1\n  Q = TVQ\n  V2 = TVV2\n  \
        MAT = TVMAT\n  CV2 = TVCV2\n\n";
    let analytical = format!(
        "{header}[structural_model]\n  \
         pk two_cpt_ig(cl=CL, v1=V1, q=Q, v2=V2, mat=MAT, cv2=CV2)\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let ode = format!(
        "{header}[structural_model]\n  ode(obs_cmt=central, states=[central, periph])\n\n\
         [odes]\n  \
         d/dt(central) = igd(mat=MAT, cv2=CV2) - (CL/V1 + Q/V1) * central + (Q/V2) * periph\n  \
         d/dt(periph) = (Q/V1) * central - (Q/V2) * periph\n\n\
         [scaling]\n  obs_scale = V1\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    (analytical, ode)
}

/// A two-occasion, multi-dose subject with cross-occasion carryover (dose 1 in occ 0 still has
/// drug on board at occ 1's observations, t=13/18) — the case a per-dose superposition gets
/// wrong (#104), so occ 1's `KAPPA_CL` must govern that tail via the twin's integration.
fn iov_subject() -> Population {
    let mut pop = population(
        vec![bolus(0.0, 100.0), bolus(12.0, 100.0)],
        vec![1.0, 6.0, 13.0, 18.0],
    );
    pop.subjects[0].occasions = vec![0, 0, 1, 1];
    pop.subjects[0].dose_occasions = vec![0, 1];
    pop
}

/// The analytic IG IOV model (auto-routed to its `igd()` ODE twin) must predict identically to
/// the hand-written `igd()` forcing IOV model at non-zero per-occasion kappas — the correctness
/// anchor for the reroute (cross-occasion carryover via the twin's exact integration, #104/#663).
fn assert_iov_matches_ode(an_src: &str, ode_src: &str, label: &str) {
    let an = parse_full_model(an_src)
        .unwrap_or_else(|e| panic!("[{label}] analytic IG IOV parses: {e}"))
        .model;
    let ode = parse_full_model(ode_src)
        .unwrap_or_else(|e| panic!("[{label}] ODE IG IOV parses: {e}"))
        .model;
    assert_eq!(an.n_kappa, 1, "[{label}] one IOV kappa");
    assert!(
        an.absorption_ode_equivalent.is_some(),
        "[{label}] IG IOV carries an ODE twin"
    );
    let pop = iov_subject();
    let subject = &pop.subjects[0];
    assert!(
        an.effective_for(subject).ode_spec.is_some(),
        "[{label}] IOV subject routes the closed form to its igd() ODE twin"
    );
    let theta = &an.default_params.theta;
    let eta_bsv = [0.0]; // one BSV eta (ETA_CL)
    let kappas = [vec![0.20], vec![-0.30]];
    let pa = ferx_core::pk::predict_iov(&an, subject, theta, &eta_bsv, &kappas);
    let po = ferx_core::pk::predict_iov(&ode, subject, theta, &eta_bsv, &kappas);
    assert_eq!(pa.len(), po.len(), "[{label}] prediction count mismatch");
    let pa0 = ferx_core::pk::predict_iov(&an, subject, theta, &eta_bsv, &[vec![0.0], vec![0.0]]);
    assert!(
        pa.iter().zip(pa0.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
        "[{label}] non-zero kappa should change the IG IOV predictions"
    );
    for (i, (x, y)) in pa.iter().zip(po.iter()).enumerate() {
        let tol = ATOL + ACCUM_RTOL * x.abs();
        assert!(
            (x - y).abs() <= tol,
            "[{label}] obs {i}: analytic-twin {x:.6} vs hand-written ODE {y:.6} (|diff| {:.2e} > tol {:.2e})",
            (x - y).abs(),
            tol
        );
    }
}

#[test]
fn ig_1cpt_iov_matches_hand_written_ode_forcing() {
    let (an, ode) = build_iov_pair_1cpt();
    assert_iov_matches_ode(&an, &ode, "ig-1cpt-iov");
}

#[test]
fn ig_2cpt_iov_matches_hand_written_ode_forcing() {
    let (an, ode) = build_iov_pair_2cpt();
    assert_iov_matches_ode(&an, &ode, "ig-2cpt-iov");
}

/// A bounded FOCEI fit of an IG IOV model — drives the analytic IOV **sensitivity** reroute
/// (`subject_sensitivities_iov` / `subject_eta_grad_iov` → the `igd()` twin's ODE-IOV gradients,
/// #663), the path `predict_iov` alone does not exercise. Tier-2: returns after two outer
/// iterations with a finite OFV, not convergence.
#[test]
fn ig_iov_now_served_and_drives_analytic_sens() {
    let (an_src, _) = build_iov_pair_1cpt();
    let model = parse_full_model(&an_src).expect("IG IOV parses").model;
    assert_eq!(model.n_kappa, 1);
    assert!(
        model.absorption_ode_equivalent.is_some(),
        "IG IOV carries an ODE twin"
    );
    let template = iov_subject();
    let base = predict(&model, &template, &model.default_params);
    // Two IOV subjects (each with two occasions) so the fit is identifiable; observations are
    // the eta=0 predictions, lightly perturbed per subject.
    let subjects: Vec<_> = [1.0_f64, 1.1]
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let mut s = template.subjects[0].clone();
            s.id = format!("{}", i + 1);
            s.observations = base.iter().map(|p| (p.pred * f).max(1e-6)).collect();
            s
        })
        .collect();
    let pop = Population {
        subjects,
        ..template
    };
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.outer_maxiter = 2;
    opts.run_covariance_step = false;
    let r = fit(&model, &pop, &model.default_params, &opts)
        .expect("IG IOV fits via the igd() ODE equivalent");
    assert!(r.ofv.is_finite(), "IG IOV fit OFV not finite: {}", r.ofv);
}
