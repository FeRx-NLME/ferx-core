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

#[test]
fn ig_steady_state_dose_rejected() {
    let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
    let e = ig_fit_err(&population(vec![ss], vec![1.0, 4.0, 8.0]));
    assert!(
        e.contains("steady-state") || e.contains("SS"),
        "expected an SS-rejection message, got: {e}"
    );
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
