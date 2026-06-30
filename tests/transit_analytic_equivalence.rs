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

#[test]
fn transit_tv_covariate_rejected() {
    let (an_src, _) = build_pair(false, false);
    let model = parse_full_model(&an_src).expect("parses").model;
    let mut pop = population(vec![bolus(0.0, 100.0)], vec![1.0, 4.0]);
    // Give the subject time-varying covariates (non-empty per-observation maps).
    pop.subjects[0].obs_covariates = vec![
        std::collections::HashMap::from([("WT".to_string(), 70.0)]),
        std::collections::HashMap::from([("WT".to_string(), 72.0)]),
    ];
    let e = fit(&model, &pop, &model.default_params, &FitOptions::default())
        .expect_err("transit + time-varying covariates should be rejected");
    assert!(
        e.contains("time-varying"),
        "expected a TV-covariate rejection message, got: {e}"
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
