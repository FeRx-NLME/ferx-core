//! End-to-end NONMEM cross-check for steady-state (`SS=1`) dosing into a **built-in
//! absorption** compartment — the acceptance anchor for issue #719 (gap 1).
//!
//! Exercises the public path parser → NONMEM CSV reader → `predict()` → the ODE
//! input-rate forcing walk with steady-state equilibration, and asserts the
//! population predictions (eta = 0) match NONMEM 7.6.0's exact analytic `ADVAN2`
//! steady-state solution to 1e-4 relative.
//!
//! `KA` is deliberately slow (`t½,abs ≈ 4.6 h`) relative to `II = 8 h`, so each
//! dose's first-order absorption tail spills into the following interval — exactly
//! the carryover that ferx's SS support must capture as (a) a pulse-train
//! equilibration trough and (b) a periodic-sum forward `R_in`
//! (`ode::predictions::equilibrate_ss_state` + `add_prepared_input_rate_forcing`).
//! A bolus-only SS treatment (the pre-#719 behaviour, had it not been rejected)
//! would land visibly off this reference. The unit test
//! `ode::predictions::tests::ss_into_first_order_absorption_matches_explicit_run_in`
//! pins the same physics against an explicit run-in; this is the independent NONMEM
//! cross-check required by CLAUDE.md.
//!
//! ## Reproducing the NONMEM reference
//!
//! The reference PRED values baked into `data/ss_first_order.csv` (the `DV` column)
//! were produced with NONMEM 7.6.0, `$ESTIMATION MAXEVAL=0` (pure evaluation at the
//! fixed thetas), from `nonmem_anchor/ss_first_order.ctl` (`ADVAN2 TRANS2`, `S2=V`,
//! a single subject dosed `SS=1, II=8, AMT=100` into the depot, observations at
//! TIME = 0.5, 1, 2, 4, 8-ε). See `nonmem_anchor/ss_first_order.{ctl,csv}` and
//! `nonmem_anchor/results/ss_first_order.*`.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, predict, read_nonmem_csv, simulate_with_options_diag, SimulateOptions};

// `first_order(ka)` appears in central directly (its R_in is the appearance rate of an
// ADVAN2 depot→central first-order absorption), and the readout `y = central/V` matches
// NONMEM's `S2 = V` concentration. Thetas are FIXed at the NONMEM values.
const SS_FIRST_ORDER_MODEL: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-10
"#;

#[test]
fn predict_matches_nonmem_ss_first_order_absorption() {
    let parsed = parse_full_model(SS_FIRST_ORDER_MODEL).expect("model parses");
    let model = parsed.model;

    let population = read_nonmem_csv(std::path::Path::new("data/ss_first_order.csv"), None, None)
        .expect("dataset loads");
    assert!(
        population.subjects.iter().any(|s| s.has_ss_doses()),
        "dataset should contain SS=1 doses"
    );

    let preds = predict(&model, &population, &model.default_params);

    // NONMEM 7.6.0 PRED (S2 = V), keyed by observation time.
    let nonmem: &[(f64, f64)] = &[
        (0.5, 12.231),
        (1.0, 12.402),
        (2.0, 12.634),
        (4.0, 12.735),
        (6.0, 12.490),
        (7.9, 12.044),
    ];
    assert_eq!(preds.len(), nonmem.len());

    for (p, &(t, expected)) in preds.iter().zip(nonmem) {
        assert!(
            (p.time - t).abs() < 1e-9,
            "prediction time {} != expected {}",
            p.time,
            t
        );
        let rel = (p.pred - expected).abs() / expected;
        assert!(
            rel < 1e-4,
            "t={t}: ferx PRED {:.5} vs NONMEM {expected:.5} (rel err {rel:.2e})",
            p.pred
        );
    }
}
/// End-to-end `fit()` on an SS-into-absorption model. Since #835 the sensitivity path is the
/// **analytic dual** SS-equilibration — the closed-form fixed point `u_ss = (I − M)⁻¹·b` carried
/// over `Dual2` (the parity follow-up this test's predecessor flagged), not finite differences.
/// The fit must converge to a sensible objective rather than stall or diverge. Slow-gated only
/// because it runs a full FOCEI fit to convergence (Tier 3).
///
/// The dataset's `DV` column is the deterministic NONMEM PRED, so the population minimum sits
/// at the fixed thetas; starting there, a bounded fit must keep the objective finite and near
/// that minimum.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: analytic FOCEI to convergence on an SS-absorption model; opt in with --features slow-tests"
)]
fn fit_on_ss_absorption_converges() {
    let src = SS_FIRST_ORDER_MODEL.replace("omega ETA_CL ~ 0.0", "omega ETA_CL ~ 0.05");
    let parsed = parse_full_model(&src).expect("model parses");
    let model = parsed.model;
    let population = read_nonmem_csv(std::path::Path::new("data/ss_first_order.csv"), None, None)
        .expect("dataset loads");

    let mut opts = parsed.fit_options;
    opts.ode_reltol = 1e-8;
    opts.ode_abstol = 1e-8;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("SS-absorption fit returns Ok");
    assert!(
        result.ofv.is_finite(),
        "SS-absorption fit objective must be finite, got {}",
        result.ofv
    );
    // CL/V/KA start FIXed-free at the truth; the fit must not drive them away (the residuals
    // are ~0 at the truth), so the recovered thetas stay close to their starting values.
    let truth = [1.0_f64, 20.0, 0.15];
    for (est, tv) in result.theta.iter().zip(truth) {
        let rel = (est - tv).abs() / tv;
        assert!(
            rel < 0.25,
            "SS-absorption fit drifted a theta: got {est:.4}, truth {tv} (rel {rel:.2})"
        );
    }
}

/// A **saturable (Michaelis–Menten) SS-absorption model dosed above its elimination capacity**: the
/// dataset's mean absorbed input (AMT/II = 100/8 = 12.5) *exceeds* `Vmax = 10`, so no periodic
/// steady state exists — the Anderson fixed-point solve (#867) correctly declines, equilibration
/// falls to the capped pulse train, and a non-convergence warning is raised. This drives the full
/// public `simulate` path and asserts the warning reaches `SimulationOutput.warnings` (the
/// api-boundary drain) rather than being swallowed silently. Fast (an evaluation-only simulate, no
/// convergence loop), so it is not slow-gated. The predictor-level detection and the converging
/// (Anderson-solved) case are unit-tested in `ode::predictions::tests` (`ss_input_rate_*`).
const SS_MM_NO_STEADY_STATE_MODEL: &str = r#"
[parameters]
  theta TVVMAX(10.0, 1.0, 100.0)
  theta TVKM(20.0, 1.0, 5000.0)
  theta TVKA(2.0, 0.05, 20.0)

  omega ETA_X ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  VMAX = TVVMAX
  KM   = TVKM
  KA   = TVKA

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = first_order(ka=KA) - VMAX*central/(KM+central)

[scaling]
  y = central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  ode_reltol = 1e-6
  ode_abstol = 1e-6
"#;

// ── #867: nonlinear (Michaelis–Menten) SS-absorption vs NONMEM ────────────────────────────────
//
// A saturable disposition declines the linear closed form, so ferx solves the periodic steady
// state by Anderson acceleration of the one-cycle fixed point (#867). This anchors that solve
// against NONMEM 7.6.0's own general-ODE (`ADVAN13`) steady-state solver.
//
// `first_order(ka)` appears directly in `central` (its `R_in` is the appearance rate of a
// depot→central first-order absorption); `y = central` matches NONMEM's `S2 = 1` amount. Mean
// absorbed input (AMT/II = 100/12 ≈ 8.33) sits below `VM = 15`, so a periodic SS exists and is
// saturable (`C_ss ≈ 375`, comparable to `KM = 300`). Thetas are FIXed at the NONMEM values.
//
// Reference `DV` in `data/ss_mm.csv` is the NONMEM PRED (`$ESTIMATION MAXEVAL=0`) from
// `nonmem_anchor/ss_mm.{ctl,csv}` — `ADVAN13 TOL=9`, `$DES` MM elimination, `SS=1, II=12, AMT=100`,
// observations at TIME = 1, 3, 6, 9, 11.9.
const SS_MM_MODEL: &str = r#"
[parameters]
  theta TVKA(1.0, 0.01, 20.0)
  theta TVVM(15.0, 0.1, 200.0)
  theta TVKM(300.0, 1.0, 5000.0)

  omega ETA_X ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  KA = TVKA
  VM = TVVM
  KM = TVKM

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = first_order(ka=KA) - VM*central/(KM+central)

[scaling]
  y = central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  ode_reltol = 1e-8
  ode_abstol = 1e-10
"#;

#[test]
fn predict_matches_nonmem_mm_ss_absorption() {
    let parsed = parse_full_model(SS_MM_MODEL).expect("model parses");
    let model = parsed.model;
    let population =
        read_nonmem_csv(std::path::Path::new("data/ss_mm.csv"), None, None).expect("dataset loads");
    assert!(
        population.subjects.iter().any(|s| s.has_ss_doses()),
        "dataset should contain SS=1 doses"
    );

    let preds = predict(&model, &population, &model.default_params);

    // NONMEM 7.6.0 PRED (ADVAN13 general-ODE steady state), keyed by observation time.
    let nonmem: &[(f64, f64)] = &[
        (1.0, 389.60439420601),
        (3.0, 404.23630702376),
        (6.0, 383.39866222919),
        (9.0, 358.75370734093),
        (11.9, 335.43641567078),
    ];
    assert_eq!(preds.len(), nonmem.len());

    for (p, &(t, expected)) in preds.iter().zip(nonmem) {
        assert!((p.time - t).abs() < 1e-9, "time {} != {t}", p.time);
        let rel = (p.pred - expected).abs() / expected;
        assert!(
            rel < 1e-3,
            "t={t}: ferx PRED {:.5} vs NONMEM {expected:.5} (rel err {rel:.2e})",
            p.pred
        );
    }
}

#[test]
fn simulate_surfaces_ss_nonconvergence_warning_when_no_steady_state() {
    let parsed = parse_full_model(SS_MM_NO_STEADY_STATE_MODEL).expect("model parses");
    let model = parsed.model;
    let population = read_nonmem_csv(std::path::Path::new("data/ss_first_order.csv"), None, None)
        .expect("dataset loads");

    let out = simulate_with_options_diag(
        &model,
        &population,
        &model.default_params,
        1,
        &SimulateOptions::default(),
    )
    .expect("simulate returns Ok");

    assert!(
        out.warnings
            .iter()
            .any(|w| w.contains("Steady-state (SS=1) equilibration")),
        "a saturable SS model dosed above capacity must surface a non-convergence warning through \
         SimulationOutput.warnings; got: {:?}",
        out.warnings
    );
    // Simulated rows are still produced (the returned trough is finite, just unreliable).
    assert!(
        !out.results.is_empty(),
        "simulate must still return rows alongside the warning"
    );
}

/// The blocking-bug regression (#874 review): `n_starts` defaults to `1`, so a plain `fit()` takes
/// the single-start fast path — which must drain the SS non-convergence sink into
/// `FitResult.warnings`, not just the multi-start arm. A couple of outer iterations run enough
/// objective prediction passes to populate the sink; no convergence loop is needed (Tier-2), and
/// covariance is off to keep it quick.
#[test]
fn fit_surfaces_ss_nonconvergence_warning_for_over_capacity_dosing() {
    let parsed = parse_full_model(SS_MM_NO_STEADY_STATE_MODEL).expect("model parses");
    let model = parsed.model;
    let population = read_nonmem_csv(std::path::Path::new("data/ss_first_order.csv"), None, None)
        .expect("dataset loads");

    let mut opts = parsed.fit_options;
    opts.outer_maxiter = 2;
    opts.run_covariance_step = false;

    let result = fit(&model, &population, &model.default_params, &opts).expect("fit returns Ok");

    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("Steady-state (SS=1) equilibration")),
        "default single-start fit() must surface the SS non-convergence warning to \
         FitResult.warnings; got: {:?}",
        result.warnings
    );
}

/// False-positive guard (#874 review): a *nonlinear* (Michaelis–Menten) SS-absorption model that
/// still declines the linear closed-form fast path, but eliminates fast (`Vmax` ≫ mean input, so
/// little saturation) — the pulse train contracts quickly and settles well inside the cycle
/// budget. The sink must stay **empty**: a converged nonlinear model must not trip the warning on
/// solver-noise jitter (the `rho ≥ 1` noise-floor short-circuit).
#[test]
fn simulate_stays_silent_for_benign_nonlinear_ss_absorption() {
    let src = SS_MM_NO_STEADY_STATE_MODEL.replace(
        "theta TVVMAX(10.0, 1.0, 100.0)",
        "theta TVVMAX(500.0, 1.0, 5000.0)",
    );
    let parsed = parse_full_model(&src).expect("model parses");
    let model = parsed.model;
    let population = read_nonmem_csv(std::path::Path::new("data/ss_first_order.csv"), None, None)
        .expect("dataset loads");

    let out = simulate_with_options_diag(
        &model,
        &population,
        &model.default_params,
        1,
        &SimulateOptions::default(),
    )
    .expect("simulate returns Ok");

    assert!(
        !out.warnings
            .iter()
            .any(|w| w.contains("Steady-state (SS=1) equilibration")),
        "benign fast-contracting nonlinear SS model must stay silent; got: {:?}",
        out.warnings
    );
}
