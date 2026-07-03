//! Equivalence: the analytic `pk one_cpt_transit` closed form (#386) must match a
//! numerical ODE with the same Savic transit forcing into central, at eta = 0,
//! across the dosing modes transit supports — single + multiple bolus, with an
//! optional absorption lag and bioavailability. Steady-state / IOV / time-varying
//! covariate / infusion doses are rejected at validation, exercised by the
//! `*_rejected` tests below.
//!
//! The ODE twin's `transit()` forcing is itself NONMEM-anchored
//! (`tests/transit_nonmem_anchor.rs`), so matching it transitively anchors the
//! analytic exponential-tilting closed form to NONMEM.
//!
//! `predict()` evaluates at eta = 0, so the equivalence tests check the structural
//! closed form; a `fit()`-level check would conflate it with the estimator.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{DoseEvent, EstimationMethod, FitOptions, Population};
use ferx_core::{fit, predict};

mod common;

/// The RK45 solver defaults to `abstol = 1e-6`, `reltol = 1e-4`, so the ODE path
/// reproduces the closed form to about that level; combine an absolute floor so
/// near-zero early-absorption predictions don't blow up a pure relative check.
const ATOL: f64 = 1e-5;
const RTOL: f64 = 1e-4;
/// Multi-dose trajectories accumulate per-step solver error across dose restarts,
/// so they need the looser combined bound (as the sibling ODE-equivalence suite).
const ACCUM_RTOL: f64 = 1e-3;

/// Build the (analytical, ODE) `.ferx` pair for a transit model. A lagtime and/or
/// bioavailability mapping lives in the shared `[individual_parameters]`, so the
/// ODE twin applies them automatically (only the analytic `pk(...)` needs the
/// explicit `lagtime=`/`f=` mapping).
fn build_pair(lag: bool, fbio: bool) -> (String, String) {
    let mut thetas = String::from(
        "  theta TVCL(0.13, 0.001, 10.0)\n  \
         theta TVV(8.0, 0.1, 500.0)\n  \
         theta TVNTR(3.0, 0.0, 20.0)\n  \
         theta TVMTT(1.5, 0.05, 50.0)\n",
    );
    let mut indiv =
        String::from("  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  NTR = TVNTR\n  MTT = TVMTT\n");
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
         pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT{pk_extra})\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let ode = format!(
        "{header}[structural_model]\n  ode(obs_cmt=central, states=[central])\n\n\
         [odes]\n  d/dt(central) = transit(n=NTR, mtt=MTT) - (CL/V) * central\n\n\
         [scaling]\n  obs_scale = V\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    (analytical, ode)
}

/// One subject; the dose enters compartment 1 (the transit input for both forms).
/// `obs_cmts` is inert here — the analytic path always reads central and the ODE
/// twin reads its `obs_cmt=central` directive, so neither consults the column.
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

fn assert_equiv(label: &str, lag: bool, fbio: bool, pop: &Population, rtol: f64) {
    let (an_src, ode_src) = build_pair(lag, fbio);
    let an = parse_full_model(&an_src)
        .unwrap_or_else(|e| panic!("[{label}] analytic transit did not parse: {e}"))
        .model;
    let ode = parse_full_model(&ode_src)
        .unwrap_or_else(|e| panic!("[{label}] ODE transit did not parse: {e}"))
        .model;

    let pa = predict(&an, pop, &an.default_params);
    let po = predict(&ode, pop, &ode.default_params);
    assert_eq!(pa.len(), po.len(), "[{label}] prediction count mismatch");
    assert!(!pa.is_empty(), "[{label}] produced no predictions");
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

#[test]
fn transit_single_dose_matches_ode() {
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    );
    assert_equiv("single", false, false, &pop, RTOL);
}

#[test]
fn transit_multidose_matches_ode() {
    let doses = vec![bolus(0.0, 100.0), bolus(12.0, 100.0), bolus(24.0, 100.0)];
    let obs = vec![0.5, 2.0, 6.0, 11.5, 13.0, 18.0, 23.5, 26.0, 36.0, 48.0];
    assert_equiv(
        "multidose",
        false,
        false,
        &population(doses, obs),
        ACCUM_RTOL,
    );
}

/// A transit model with a mid-profile `TIME` switch on CL. The closed-form shorthand
/// `pk one_cpt_transit(...)` cannot honour it, so it is desugared to the ODE `transit()`
/// twin (#486). Written by hand as that same ODE, the two must predict identically — the
/// desugar produces exactly the twin, and the observation times straddle the switch (t=6)
/// so the `TIME` dependence is actually exercised.
#[test]
fn transit_time_desugar_matches_hand_written_ode() {
    let header = "\
[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVCL_LATE(0.30, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVNTR(3.0, 0.0, 20.0)
  theta TVMTT(1.5, 0.05, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  if (TIME > 6.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV
  NTR = TVNTR
  MTT = TVMTT
";
    let shorthand = format!(
        "{header}\n[structural_model]\n  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let hand_ode = format!(
        "{header}\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n\n\
         [odes]\n  d/dt(central) = transit(n=NTR, mtt=MTT) - (CL/V) * central\n\n\
         [scaling]\n  obs_scale = V\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let sh = parse_full_model(&shorthand)
        .expect("shorthand transit+TIME parses")
        .model;
    let hd = parse_full_model(&hand_ode).expect("hand ODE parses").model;
    // The shorthand stays a closed-form transit model but carries the ODE equivalent that
    // predict()/the gradient route TIME/TV-cov subjects to.
    assert!(
        sh.ode_spec.is_none() && sh.transit_ode_equivalent.is_some(),
        "transit + TIME shorthand carries an ODE equivalent (primary stays closed-form)"
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
    assert!(
        ps.iter().any(|p| p.time > 6.0) && ps.iter().any(|p| p.time < 6.0),
        "observations must straddle the TIME switch"
    );

    // Compartment/state columns (sdtab `[derived]`) must ALSO route to the equivalent — the
    // analytical states path has no valid states for a TIME/TV-cov transit subject and would
    // return NaN. They must be finite and match the hand-written ODE twin's states.
    let subj = &pop.subjects[0];
    let eta = vec![0.0; sh.n_eta];
    let (_, sh_states) =
        ferx_core::pk::compute_predictions_with_states(&sh, subj, &sh.default_params.theta, &eta);
    let (_, hd_states) =
        ferx_core::pk::compute_predictions_with_states(&hd, subj, &hd.default_params.theta, &eta);
    assert!(
        !sh_states.is_empty()
            && sh_states
                .iter()
                .all(|s| !s.is_empty() && s.iter().all(|x| x.is_finite())),
        "transit + TIME compartment states must be finite (not the NaN the analytical path returns)"
    );
    assert_eq!(sh_states.len(), hd_states.len());
    for (a, b) in sh_states.iter().zip(hd_states.iter()) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x - y).abs() <= ATOL + RTOL * x.abs(),
                "state mismatch: desugared {x:.6} vs hand ODE {y:.6}"
            );
        }
    }
}

#[test]
fn transit_with_lagtime_matches_ode() {
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    );
    assert_equiv("lagtime", true, false, &pop, RTOL);
}

#[test]
fn transit_with_bioavailability_matches_ode() {
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    );
    assert_equiv("fbio", false, true, &pop, RTOL);
}

/// Fixed-parameter population OFV (FOCEI marginal NLL) at several eta values: the
/// analytic transit and its ODE twin must agree. Unlike the eta=0 `predict()` sweep
/// above, this drives the analytic **sensitivity** path (`run_obs` transit branch,
/// the exact `∂f/∂{cl,v,n,mtt,η}` jets), and adds a likelihood-level anchor. (A
/// fixed-parameter NLL, not a converged fit, so optimizer path / solver noise can't
/// confound it — mirrors `analytical_ode_equivalence::assert_ofv_equiv`.)
#[test]
fn transit_ofv_matches_ode() {
    use ferx_core::stats::likelihood::individual_nll;
    use ferx_core::CompiledModel;
    const OFV_RTOL: f64 = 2e-3;

    let obs_t = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let (an_src, ode_src) = build_pair(false, false);
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
            "transit OFV mismatch at eta={eta}: analytic {a:.6} vs ODE {o:.6} (rel {rel:.2e})"
        );
    }
}

// ── Restrictions: features the v1 closed form does not support are rejected
//    up front by `fit()` (and would panic in `predict()`/`simulate()`), #386. ──

/// `fit()` the basic analytic transit model on `pop`, expecting an early `Err`.
fn transit_fit_err(pop: &Population) -> String {
    let (an_src, _) = build_pair(false, false);
    let model = parse_full_model(&an_src)
        .expect("transit model parses")
        .model;
    fit(&model, pop, &model.default_params, &FitOptions::default())
        .expect_err("fit should reject the unsupported transit configuration")
}

#[test]
fn transit_steady_state_dose_rejected() {
    let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
    let e = transit_fit_err(&population(vec![ss], vec![1.0, 4.0, 8.0]));
    assert!(
        e.contains("steady-state") || e.contains("SS"),
        "expected an SS-rejection message, got: {e}"
    );
}

#[test]
fn transit_infusion_dose_rejected() {
    // rate > 0 → infusion.
    let inf = DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0);
    let e = transit_fit_err(&population(vec![inf], vec![1.0, 4.0, 8.0]));
    assert!(
        e.contains("infusion"),
        "expected an infusion-rejection message, got: {e}"
    );
}

#[test]
fn transit_iov_rejected() {
    // Mark the (otherwise basic) transit model as carrying IOV; the up-front guard
    // must reject it before any n_kappa-dependent prediction runs.
    let (an_src, _) = build_pair(false, false);
    let mut model = parse_full_model(&an_src).expect("parses").model;
    model.n_kappa = 1;
    let pop = population(vec![bolus(0.0, 100.0)], vec![1.0, 4.0]);
    let e = fit(&model, &pop, &model.default_params, &FitOptions::default())
        .expect_err("transit + IOV should be rejected");
    assert!(
        e.contains("IOV"),
        "expected an IOV-rejection message, got: {e}"
    );
}

/// A plain `one_cpt_transit` model with time-varying covariates now **works**: the closed
/// form can't serve a subject whose parameters switch mid-absorption, so the runtime dispatch
/// routes it to the model's exact ODE `transit()` equivalent (built at parse time). It is no
/// longer rejected. (A transit form outside the equivalent's scope — a `lagtime=`/`f=`
/// mapping, custom scaling, or an init block — carries no equivalent and is still rejected.)
#[test]
fn transit_tv_covariate_now_served_by_ode_equivalent() {
    let (an_src, _) = build_pair(false, false);
    let model = parse_full_model(&an_src).expect("parses").model;
    assert!(
        model.transit_ode_equivalent.is_some(),
        "plain transit carries an ODE equivalent"
    );
    let mut pop = population(vec![bolus(0.0, 100.0)], vec![1.0, 4.0]);
    // Give the subject time-varying covariates (non-empty per-observation maps).
    pop.subjects[0].obs_covariates = vec![
        std::collections::HashMap::from([("WT".to_string(), 70.0)]),
        std::collections::HashMap::from([("WT".to_string(), 72.0)]),
    ];
    assert!(pop.subjects[0].has_tv_covariates());
    fit(&model, &pop, &model.default_params, &FitOptions::default())
        .expect("transit + time-varying covariates now fits via the ODE equivalent");
}

#[test]
fn transit_reset_rejected() {
    // A system reset (EVID=3/4) zeros compartment amounts mid-profile, which the
    // superposition closed form cannot express — `fit()` (superposition) would then
    // silently disagree with the sensitivity path (which washes out pre-reset doses),
    // so it must be rejected up front (#634 review finding 1).
    let (an_src, _) = build_pair(false, false);
    let model = parse_full_model(&an_src).expect("parses").model;
    let mut pop = population(vec![bolus(0.0, 100.0)], vec![1.0, 4.0]);
    pop.subjects[0].reset_times = vec![2.0];
    let e = fit(&model, &pop, &model.default_params, &FitOptions::default())
        .expect_err("transit + system reset should be rejected");
    assert!(
        e.contains("reset") && e.contains("EVID"),
        "expected a reset-rejection message, got: {e}"
    );
}

/// `ode_template one_cpt_transit(...)` desugars to the `transit()` forcing ODE
/// (#386), so it must `predict()` identically to the analytic `pk` form. Covers the
/// `ode_template` transit arm and pins the lowering to the closed form.
#[test]
fn transit_ode_template_matches_pk() {
    let header = "[parameters]\n  theta TVCL(0.13, 0.001, 10.0)\n  \
                  theta TVV(8.0, 0.1, 500.0)\n  theta TVNTR(3.0, 0.0, 20.0)\n  \
                  theta TVMTT(1.5, 0.05, 50.0)\n  omega ETA_CL ~ 0.09\n  \
                  sigma PROP ~ 0.01 (sd)\n\n[individual_parameters]\n  \
                  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  NTR = TVNTR\n  MTT = TVMTT\n\n";
    let pk_src = format!(
        "{header}[structural_model]\n  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let tmpl_src = format!(
        "{header}[structural_model]\n  \
         ode_template one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let pk = parse_full_model(&pk_src).expect("pk parses").model;
    let tmpl = parse_full_model(&tmpl_src)
        .expect("ode_template parses")
        .model;
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    );
    let pp = predict(&pk, &pop, &pk.default_params);
    let pt = predict(&tmpl, &pop, &tmpl.default_params);
    for (x, y) in pp.iter().zip(pt.iter()) {
        let tol = ATOL + RTOL * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "t={:.2}: pk PRED {:.6} vs ode_template PRED {:.6}",
            x.time,
            x.pred,
            y.pred
        );
    }
}

/// A `depot` (cmt 1) `[initial_conditions]` amount on the analytic transit model
/// must be **rejected at parse**, not silently dropped (#386). The transit model
/// is `is_oral()` (absorption layout), which the init-compartment resolver used to
/// read as "has a seedable depot" — but the runtime init dispatch has no transit
/// depot arm, so the amount would vanish with no error. `central` is still
/// accepted (an amount pre-loaded in central just decays as a 1-cpt IV bolus).
#[test]
fn transit_depot_init_rejected_central_ok() {
    let model_src = |ic: &str| {
        format!(
            "[parameters]\n  theta TVCL(0.13, 0.001, 10.0)\n  \
             theta TVV(8.0, 0.1, 500.0)\n  theta TVNTR(3.0, 0.0, 20.0)\n  \
             theta TVMTT(1.5, 0.05, 50.0)\n  omega ETA_CL ~ 0.09\n  \
             sigma PROP ~ 0.01 (sd)\n\n[individual_parameters]\n  \
             CL = TVCL * exp(ETA_CL)\n  V = TVV\n  NTR = TVNTR\n  MTT = TVMTT\n\n\
             [structural_model]\n  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)\n\n\
             [error_model]\n  DV ~ proportional(PROP)\n\n[initial_conditions]\n{ic}\n"
        )
    };
    // Named `depot` and numeric `init(1)` both name the lumped transit depot.
    for ic in ["  init(depot) = 50.0", "  init(1) = 50.0"] {
        let e = parse_full_model(&model_src(ic))
            .err()
            .unwrap_or_else(|| panic!("transit `{ic}` init should be rejected at parse"));
        assert!(
            e.contains("transit") && (e.contains("depot") || e.contains("cmt 1")),
            "expected a transit depot-init rejection, got: {e}"
        );
    }
    // Positive control: a central initial amount is supported and parses.
    parse_full_model(&model_src("  init(central) = 2.0"))
        .expect("transit central init should parse");
}

/// Short FOCEI fit (a couple of outer iterations) — this is the test that actually
/// drives the analytic **sensitivity** path for transit: `run_obs` / `run_obs_grad`'s
/// transit branch, `one_cpt_transit_conc_g`, `slot_to_dim` for `n`/`mtt`, and the
/// `ExKind` arm. (The fixed-eta OFV test above evaluates `individual_nll` and does *not*
/// go through the sens provider, so those exact `∂f/∂{cl,v,n,mtt}` jets the estimator
/// seeds were otherwise only exercised by an uncommitted benchmark.) Tier-2: returns
/// after a handful of iterations with a finite OFV, not convergence.
#[test]
fn transit_short_fit_drives_analytic_sens() {
    let (an_src, _) = build_pair(false, false);
    let model = parse_full_model(&an_src).unwrap().model;
    let obs_t = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let base = population(vec![bolus(0.0, 100.0)], obs_t.clone());
    let preds = predict(&model, &base, &model.default_params);
    let n = obs_t.len();
    let subjects: Vec<_> = [0.9_f64, 1.0, 1.1]
        .into_iter()
        .enumerate()
        .map(|(i, fac)| {
            let mut s = common::subject(
                &format!("{}", i + 1),
                vec![bolus(0.0, 100.0)],
                obs_t.clone(),
                vec![0.0; n],
                vec![2; n],
            );
            s.observations = preds.iter().map(|p| (p.pred * fac).max(1e-6)).collect();
            s
        })
        .collect();
    let pop = Population { subjects, ..base };
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.outer_maxiter = 2;
    opts.run_covariance_step = false;
    let r = fit(&model, &pop, &model.default_params, &opts)
        .expect("short transit FOCEI fit should return Ok");
    assert!(r.ofv.is_finite(), "transit fit OFV not finite: {}", r.ofv);
}

/// `predict()` / `simulate()` panic (rather than silently mis-predict) on an unsupported
/// transit configuration — the `Vec`-returning entry points use `assert_transit_support`.
#[test]
#[should_panic(expected = "one_cpt_transit")]
fn transit_ss_dose_panics_in_predict() {
    let (an_src, _) = build_pair(false, false);
    let model = parse_full_model(&an_src).unwrap().model;
    let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0); // SS dose
    let pop = population(vec![ss], vec![1.0, 4.0, 8.0]);
    let _ = predict(&model, &pop, &model.default_params);
}

/// Committed benchmark (slow): the analytic `pk one_cpt_transit` and the ODE `transit()`
/// forcing must converge to the **same** estimates on identical simulated data — a
/// fit-level equivalence stronger than the fixed-parameter OFV check, and the permanent
/// record of the speed-up (#386). Runtimes are logged for the nightly record; the analytic
/// path is the fast one (~28× locally at 50×12). Gated out of the per-PR job.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn transit_analytic_vs_ode_fit_benchmark() {
    use ferx_core::simulate_with_seed;
    use std::time::Instant;
    let (an_src, ode_src) = build_pair(false, false);
    let an = parse_full_model(&an_src).unwrap().model;
    let ode = parse_full_model(&ode_src).unwrap().model;
    let obs_t = vec![
        0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0,
    ];
    let n = obs_t.len();
    let subjects: Vec<_> = (1..=30)
        .map(|i| {
            common::subject(
                &format!("{i}"),
                vec![bolus(0.0, 100.0)],
                obs_t.clone(),
                vec![0.0; n],
                vec![2; n],
            )
        })
        .collect();
    let mut pop = Population {
        subjects,
        ..population(vec![bolus(0.0, 100.0)], obs_t.clone())
    };
    let sims = simulate_with_seed(&ode, &pop, &ode.default_params, 1, 12345);
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
    let t = Instant::now();
    let ra = fit(&an, &pop, &an.default_params, &opts).expect("analytic transit fit ok");
    let ta = t.elapsed();
    let t = Instant::now();
    let ro = fit(&ode, &pop, &ode.default_params, &opts).expect("ode transit fit ok");
    let to = t.elapsed();
    eprintln!(
        "[transit fit benchmark] analytic {:.2?} OFV={:.3} | ODE {:.2?} OFV={:.3} | speedup {:.1}x",
        ta,
        ra.ofv,
        to,
        ro.ofv,
        to.as_secs_f64() / ta.as_secs_f64().max(1e-9)
    );
    assert!(
        (ra.ofv - ro.ofv).abs() < 0.1,
        "fit OFV mismatch: analytic {} vs ODE {}",
        ra.ofv,
        ro.ofv
    );
    for (a, b) in ra.theta.iter().zip(&ro.theta) {
        assert!(
            (a - b).abs() / b.abs().max(1.0) < 5e-3,
            "fit theta mismatch: analytic {a} vs ODE {b}"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Two-compartment analytic transit — `pk two_cpt_transit` (#386 PR D). Same
// equivalence contract as the 1-cpt block above: the bi-exponential closed form
// (`convolve_2cpt`) must match a numerical ODE with the same Savic transit forcing
// into central on a 2-cpt disposition, transitively NONMEM-anchored via that ODE.
// ════════════════════════════════════════════════════════════════════════════

/// Build the (analytic, ODE) `.ferx` pair for a 2-cpt transit model. Params are
/// chosen absorption-rate-limited (`α < KTR`) so the tilting closed form converges.
fn build_pair_2cpt(lag: bool, fbio: bool) -> (String, String) {
    let mut thetas = String::from(
        "  theta TVCL(0.13, 0.001, 10.0)\n  \
         theta TVV1(8.0, 0.1, 500.0)\n  \
         theta TVQ(0.5, 0.001, 50.0)\n  \
         theta TVV2(4.0, 0.1, 500.0)\n  \
         theta TVNTR(3.0, 0.0, 20.0)\n  \
         theta TVMTT(1.5, 0.05, 50.0)\n",
    );
    let mut indiv = String::from(
        "  CL = TVCL * exp(ETA_CL)\n  V1 = TVV1\n  Q = TVQ\n  V2 = TVV2\n  \
         NTR = TVNTR\n  MTT = TVMTT\n",
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
         pk two_cpt_transit(cl=CL, v1=V1, q=Q, v2=V2, n=NTR, mtt=MTT{pk_extra})\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let ode = format!(
        "{header}[structural_model]\n  ode(obs_cmt=central, states=[central, periph])\n\n\
         [odes]\n  \
         d/dt(central) = transit(n=NTR, mtt=MTT) - (CL/V1 + Q/V1) * central + (Q/V2) * periph\n  \
         d/dt(periph)  = (Q/V1) * central - (Q/V2) * periph\n\n\
         [scaling]\n  obs_scale = V1\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    (analytical, ode)
}

fn assert_equiv_2cpt(label: &str, lag: bool, fbio: bool, pop: &Population, rtol: f64) {
    let (an_src, ode_src) = build_pair_2cpt(lag, fbio);
    let an = parse_full_model(&an_src)
        .unwrap_or_else(|e| panic!("[{label}] analytic 2cpt transit did not parse: {e}"))
        .model;
    let ode = parse_full_model(&ode_src)
        .unwrap_or_else(|e| panic!("[{label}] ODE 2cpt transit did not parse: {e}"))
        .model;
    let pa = predict(&an, pop, &an.default_params);
    let po = predict(&ode, pop, &ode.default_params);
    assert_eq!(pa.len(), po.len(), "[{label}] prediction count mismatch");
    assert!(!pa.is_empty(), "[{label}] produced no predictions");
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

#[test]
fn two_cpt_transit_single_dose_matches_ode() {
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    );
    assert_equiv_2cpt("2cpt single", false, false, &pop, RTOL);
}

#[test]
fn two_cpt_transit_multidose_matches_ode() {
    let pop = population(
        vec![bolus(0.0, 100.0), bolus(12.0, 100.0), bolus(24.0, 100.0)],
        vec![1.0, 4.0, 8.0, 13.0, 16.0, 25.0, 30.0, 36.0],
    );
    assert_equiv_2cpt("2cpt multidose", false, false, &pop, ACCUM_RTOL);
}

#[test]
fn two_cpt_transit_with_lagtime_matches_ode() {
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    );
    assert_equiv_2cpt("2cpt lagtime", true, false, &pop, RTOL);
}

#[test]
fn two_cpt_transit_with_bioavailability_matches_ode() {
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    );
    assert_equiv_2cpt("2cpt fbio", false, true, &pop, RTOL);
}

/// Fixed-parameter population OFV at several eta values — drives the analytic
/// **sensitivity** path (`run_obs` two_cpt_transit branch, the exact
/// `∂f/∂{cl,v1,q,v2,n,mtt,η}` jets), the 2-cpt counterpart of `transit_ofv_matches_ode`.
#[test]
fn two_cpt_transit_ofv_matches_ode() {
    use ferx_core::stats::likelihood::individual_nll;
    use ferx_core::CompiledModel;
    const OFV_RTOL: f64 = 2e-3;
    let obs_t = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let (an_src, ode_src) = build_pair_2cpt(false, false);
    let an = parse_full_model(&an_src).unwrap().model;
    let ode = parse_full_model(&ode_src).unwrap().model;
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
            "2cpt transit OFV mismatch at eta={eta}: analytic {a:.6} vs ODE {o:.6} (rel {rel:.2e})"
        );
    }
}

/// Short FOCEI fit (a couple of outer iterations) — drives the 2-cpt analytic
/// sensitivity provider end-to-end (`run_obs_grad` two_cpt_transit branch,
/// `two_cpt_transit_conc_g`). Tier-2: returns with a finite OFV, not convergence.
#[test]
fn two_cpt_transit_short_fit_drives_analytic_sens() {
    let (an_src, _) = build_pair_2cpt(false, false);
    let model = parse_full_model(&an_src).unwrap().model;
    let obs_t = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let base = population(vec![bolus(0.0, 100.0)], obs_t.clone());
    let preds = predict(&model, &base, &model.default_params);
    let n = obs_t.len();
    let subjects: Vec<_> = [0.9_f64, 1.0, 1.1]
        .into_iter()
        .enumerate()
        .map(|(i, fac)| {
            let mut s = common::subject(
                &format!("{}", i + 1),
                vec![bolus(0.0, 100.0)],
                obs_t.clone(),
                vec![0.0; n],
                vec![2; n],
            );
            s.observations = preds.iter().map(|p| (p.pred * fac).max(1e-6)).collect();
            s
        })
        .collect();
    let pop = Population { subjects, ..base };
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.outer_maxiter = 2;
    opts.run_covariance_step = false;
    let r = fit(&model, &pop, &model.default_params, &opts)
        .expect("short 2cpt transit FOCEI fit should return Ok");
    assert!(
        r.ofv.is_finite(),
        "2cpt transit fit OFV not finite: {}",
        r.ofv
    );
}

/// The shared `check_transit_support` guard names the offending model — confirm the
/// 2-cpt model surfaces its own canonical name in the SS-rejection message.
#[test]
fn two_cpt_transit_ss_dose_rejected() {
    let (an_src, _) = build_pair_2cpt(false, false);
    let model = parse_full_model(&an_src).expect("parses").model;
    let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
    let pop = population(vec![ss], vec![1.0, 4.0, 8.0]);
    let e = fit(&model, &pop, &model.default_params, &FitOptions::default())
        .expect_err("2cpt transit + SS dose should be rejected");
    assert!(
        e.contains("two_cpt_transit") && (e.contains("steady-state") || e.contains("SS")),
        "expected a two_cpt_transit SS-rejection message, got: {e}"
    );
}

/// Flip-flop regime (fast macro-rate `α ≥ KTR`): the analytic 2-cpt transit closed
/// form returns an identically-zero profile, which silently degenerates a
/// proportional-error objective. `check_model_data_warnings` must surface a
/// `W_TRANSIT_FLIP_FLOP` typical-value warning (#634 review finding 3). A huge MTT
/// (→ tiny `KTR = (n+1)/mtt`) pushes the typical disposition into that regime.
#[test]
fn two_cpt_transit_flip_flop_warns() {
    use ferx_core::api::check_model_data_warnings;
    let (an_src, _) = build_pair_2cpt(false, false);
    let mut model = parse_full_model(&an_src).expect("parses").model;
    let mtt_i = model
        .default_params
        .theta_names
        .iter()
        .position(|n| n == "TVMTT")
        .expect("has TVMTT");
    model.default_params.theta[mtt_i] = 50.0; // KTR = 4/50 = 0.08 < α ≈ 0.19
    let pop = population(vec![bolus(0.0, 100.0)], vec![1.0, 4.0, 8.0]);
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    assert!(
        diags.iter().any(|d| d.code == "W_TRANSIT_FLIP_FLOP"),
        "expected W_TRANSIT_FLIP_FLOP, got: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
    // And the valid default (small MTT) must *not* warn.
    let clean = parse_full_model(&an_src).expect("parses").model;
    let ok = check_model_data_warnings(&clean, &pop, &clean.default_params);
    assert!(
        !ok.iter().any(|d| d.code == "W_TRANSIT_FLIP_FLOP"),
        "in-domain params must not warn flip-flop"
    );
}

/// The reset guard also covers the 2-cpt model, naming it (#634 review finding 1).
#[test]
fn two_cpt_transit_reset_rejected() {
    let (an_src, _) = build_pair_2cpt(false, false);
    let model = parse_full_model(&an_src).expect("parses").model;
    let mut pop = population(vec![bolus(0.0, 100.0)], vec![1.0, 4.0]);
    pop.subjects[0].reset_times = vec![2.0];
    let e = fit(&model, &pop, &model.default_params, &FitOptions::default())
        .expect_err("2cpt transit + system reset should be rejected");
    assert!(
        e.contains("two_cpt_transit") && e.contains("reset"),
        "expected a two_cpt_transit reset-rejection message, got: {e}"
    );
}

/// Committed slow benchmark: analytic `pk two_cpt_transit` and the ODE `transit()`
/// forcing on a 2-cpt disposition must converge to the **same** estimates on identical
/// simulated data (fit-level equivalence + the speed record, #386 PR D). Gated.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn two_cpt_transit_analytic_vs_ode_fit_benchmark() {
    use ferx_core::simulate_with_seed;
    use std::time::Instant;
    let (an_src, ode_src) = build_pair_2cpt(false, false);
    let an = parse_full_model(&an_src).unwrap().model;
    let ode = parse_full_model(&ode_src).unwrap().model;
    let obs_t = vec![
        0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0,
    ];
    let n = obs_t.len();
    let subjects: Vec<_> = (1..=30)
        .map(|i| {
            common::subject(
                &format!("{i}"),
                vec![bolus(0.0, 100.0)],
                obs_t.clone(),
                vec![0.0; n],
                vec![2; n],
            )
        })
        .collect();
    let mut pop = Population {
        subjects,
        ..population(vec![bolus(0.0, 100.0)], obs_t.clone())
    };
    let sims = simulate_with_seed(&ode, &pop, &ode.default_params, 1, 12345);
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
    let t = Instant::now();
    let ra = fit(&an, &pop, &an.default_params, &opts).expect("analytic 2cpt transit fit ok");
    let ta = t.elapsed();
    let t = Instant::now();
    let ro = fit(&ode, &pop, &ode.default_params, &opts).expect("ode 2cpt transit fit ok");
    let to = t.elapsed();
    eprintln!(
        "[2cpt transit fit benchmark] analytic {:.2?} OFV={:.3} | ODE {:.2?} OFV={:.3} | speedup {:.1}x",
        ta, ra.ofv, to, ro.ofv, to.as_secs_f64() / ta.as_secs_f64().max(1e-9)
    );
    assert!(
        (ra.ofv - ro.ofv).abs() < 0.1,
        "2cpt fit OFV mismatch: analytic {} vs ODE {}",
        ra.ofv,
        ro.ofv
    );
    for (a, b) in ra.theta.iter().zip(&ro.theta) {
        assert!(
            (a - b).abs() / b.abs().max(1.0) < 5e-3,
            "2cpt fit theta mismatch: analytic {a} vs ODE {b}"
        );
    }
}

/// `ode_template two_cpt_transit(...)` desugars to the `transit()` forcing on a
/// 2-cpt disposition (#386 PR D), so it must `predict()` identically to the analytic
/// `pk two_cpt_transit` form — covers the `ode_template` 2-cpt transit arm.
#[test]
fn two_cpt_transit_ode_template_matches_pk() {
    let header = "[parameters]\n  theta TVCL(0.13, 0.001, 10.0)\n  \
                  theta TVV1(8.0, 0.1, 500.0)\n  theta TVQ(0.5, 0.001, 50.0)\n  \
                  theta TVV2(4.0, 0.1, 500.0)\n  theta TVNTR(3.0, 0.0, 20.0)\n  \
                  theta TVMTT(1.5, 0.05, 50.0)\n  omega ETA_CL ~ 0.09\n  \
                  sigma PROP ~ 0.01 (sd)\n\n[individual_parameters]\n  \
                  CL = TVCL * exp(ETA_CL)\n  V1 = TVV1\n  Q = TVQ\n  V2 = TVV2\n  \
                  NTR = TVNTR\n  MTT = TVMTT\n\n";
    let pk_src = format!(
        "{header}[structural_model]\n  \
         pk two_cpt_transit(cl=CL, v1=V1, q=Q, v2=V2, n=NTR, mtt=MTT)\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let tmpl_src = format!(
        "{header}[structural_model]\n  \
         ode_template two_cpt_transit(cl=CL, v1=V1, q=Q, v2=V2, n=NTR, mtt=MTT)\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );
    let pk = parse_full_model(&pk_src).expect("pk parses").model;
    let tmpl = parse_full_model(&tmpl_src)
        .expect("ode_template parses")
        .model;
    let pop = population(
        vec![bolus(0.0, 100.0)],
        vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    );
    let pp = predict(&pk, &pop, &pk.default_params);
    let pt = predict(&tmpl, &pop, &tmpl.default_params);
    for (x, y) in pp.iter().zip(pt.iter()) {
        let tol = ATOL + RTOL * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "t={:.2}: pk PRED {:.6} vs ode_template PRED {:.6}",
            x.time,
            x.pred,
            y.pred
        );
    }
}

/// `analytical_compartment_names` for the 2-cpt transit model — the `[derived]`
/// compartment labels (`[depot, central, peripheral]`); covers the `types.rs`
/// `TwoCptTransit` arm of that map.
#[test]
fn two_cpt_transit_compartment_names() {
    let (an_src, _) = build_pair_2cpt(false, false);
    let m = parse_full_model(&an_src).unwrap().model;
    assert_eq!(
        m.analytical_compartment_names(),
        &[
            "depot".to_string(),
            "central".to_string(),
            "peripheral".to_string()
        ]
    );
}
