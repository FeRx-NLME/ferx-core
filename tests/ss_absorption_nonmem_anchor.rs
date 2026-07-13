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
use ferx_core::{fit, predict, read_nonmem_csv};

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
/// End-to-end `fit()` on an SS-into-absorption model: the finite-difference sensitivity path
/// (SS × input-rate is declined by the analytic dual gates, #719) must converge to a sensible
/// objective rather than stall or diverge. The linear fixed-point equilibration makes each
/// prediction cheap enough for this to complete, and smooth enough that the FD inner gradient
/// converges. Slow-gated (the FD FOCEI predict count is inherently large; a full-speed
/// analytic dual SS-equilibration is the parity follow-up).
///
/// The dataset's `DV` column is the deterministic NONMEM PRED, so the population minimum sits
/// at the fixed thetas; starting there, a bounded fit must keep the objective finite and near
/// that minimum (the FD gradient at the optimum is well-behaved).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: FD FOCEI on an SS-absorption model; opt in with --features slow-tests"
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
