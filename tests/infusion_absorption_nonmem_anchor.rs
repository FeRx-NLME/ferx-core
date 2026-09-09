//! End-to-end NONMEM cross-check for an **infusion (RATE>0) into a built-in absorption**
//! compartment — the acceptance anchor for issue #719 (gap 2).
//!
//! Exercises the public path parser → NONMEM CSV reader → `predict()` → the ODE input-rate
//! forcing walk with the infusion-into-kernel convolution, and asserts the population
//! predictions (eta = 0) match NONMEM 7.6.0's native `ADVAN2` zero-order-into-depot behaviour
//! to 1e-4 relative.
//!
//! ferx treats an infusion into a built-in absorption compartment as a **zero-order source
//! feeding the kernel**: the dose mass is delivered at a constant rate over the window `T`, so
//! `R_in` becomes the convolution `R_in_inf(tad) = (F·amt/T)·[G(tad) − G(tad − T)]`
//! (`PreparedInputRate::rate_infused`). NONMEM's `ADVAN2` does exactly this for a `RATE>0` dose
//! into the depot — the depot fills at a constant rate over `T = AMT/RATE = 4 h`, and first-
//! order `KA` carries it to central — so a native `ADVAN2` run (no `$DES`) is the reference.
//! The unit test `ode::predictions::tests::infusion_into_first_order_absorption_matches_subdose_train`
//! pins the same physics against an explicit sub-dose train; this is the independent NONMEM
//! cross-check required by CLAUDE.md.
//!
//! ## Reproducing the NONMEM reference
//!
//! The reference PRED values baked into `data/inf_first_order.csv` (the `DV` column) were
//! produced with NONMEM 7.6.0, `$ESTIMATION MAXEVAL=0` (pure evaluation at the fixed thetas),
//! from `nonmem_anchor/inf_first_order.ctl` (`ADVAN2 TRANS2`, `S2=V`, a single subject given a
//! `RATE=25, AMT=100` infusion into the depot, observations at TIME = 1, 2, 4, 6, 10). See
//! `nonmem_anchor/inf_first_order.{ctl,csv}` and `nonmem_anchor/results/inf_first_order.*`.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, predict, read_nonmem_csv};

// `first_order(ka)` appears in central directly (its R_in is the appearance rate of an ADVAN2
// depot→central first-order absorption), and the readout `y = central/V` matches NONMEM's
// `S2 = V` concentration. An infusion (RATE>0) into that compartment is served as the convolved
// `R_in_inf`. Thetas are FIXed at the NONMEM values.
const INF_FIRST_ORDER_MODEL: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.6, 0.005, 20.0)

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
fn predict_matches_nonmem_infusion_into_first_order_absorption() {
    let parsed = parse_full_model(INF_FIRST_ORDER_MODEL).expect("model parses");
    let model = parsed.model;

    let population = read_nonmem_csv(std::path::Path::new("data/inf_first_order.csv"), None, None)
        .expect("dataset loads");
    assert!(
        population
            .subjects
            .iter()
            .any(|s| s.doses.iter().any(|d| d.is_infusion())),
        "dataset should contain an infusion (RATE>0) dose"
    );

    let preds = predict(&model, &population, &model.default_params);

    // NONMEM 7.6.0 PRED (S2 = V), keyed by observation time.
    let nonmem: &[(f64, f64)] = &[
        (1.0, 0.30468),
        (2.0, 1.0071),
        (4.0, 2.8772),
        (6.0, 3.8508),
        (10.0, 3.6059),
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

/// The same NONMEM `PRED` column, driven through the **gated** engines instead of
/// `ode_predictions`.
///
/// `ode_predictions_with_states` (sdtab IPRED + compartment states) and
/// `ode_dense_solve_states` (the joint PK-TTE cumulative hazard, the CTMM NLL, `[derived]`
/// grids) resolve their infusions through `gated_infusions`, which — unlike its
/// `active_infusions` twin — carried no input-rate suppression (#1187). The dose was
/// therefore delivered twice: once by the convolved `R_in_inf`, and again as a plain
/// `+rate` injected straight into the compartment, bypassing the kernel. Measured against
/// this NONMEM column, deleting the suppression reads **1.52394 vs 0.30468 at t=1 — 5.00×**,
/// while `predict_matches_nonmem_infusion_into_first_order_absorption` and
/// `fit_on_infusion_absorption_runs` in this same file both stay green: they route through
/// `ode_predictions`, i.e. the `active_infusions` side that always had the guard. That is
/// exactly why the original anchor could not see this.
///
/// The two engines return different quantities and the anchor treats them as such:
/// `ode_predictions_with_states` returns the **readout**, so `[scaling] y = central / V`
/// is already applied and it compares to NONMEM (`S2 = V`) directly;
/// `ode_dense_solve_states` returns raw compartment **amounts**, so state 0 is divided by
/// `V` here. Measured, not assumed — comparing the readout as if it were an amount is
/// off by exactly `V`.
#[test]
fn gated_engines_match_nonmem_infusion_into_first_order_absorption() {
    let parsed = parse_full_model(INF_FIRST_ORDER_MODEL).expect("model parses");
    let model = parsed.model;

    let population = read_nonmem_csv(std::path::Path::new("data/inf_first_order.csv"), None, None)
        .expect("dataset loads");
    let subject = population.subjects.first().expect("one subject");
    assert!(
        subject.doses.iter().any(|d| d.is_infusion()),
        "dataset should contain an infusion (RATE>0) dose"
    );

    let ode = model
        .ode_spec
        .as_ref()
        .expect("model compiles to an ODE spec");
    let theta = &model.default_params.theta;
    let eta = [0.0_f64]; // population prediction, matching the `predict()` anchor above
    let pk = (model.pk_param_fn)(theta, &eta, &subject.covariates, 0.0);
    let v = pk.values[ferx_core::PK_IDX_V];
    assert!(
        v > 0.0,
        "V must be positive to form the concentration readout"
    );

    let (with_states_pred, _states) =
        ferx_core::ode::ode_predictions_with_states(ode, &pk.values, theta, &eta, subject);
    let dense = ferx_core::ode::ode_dense_solve_states(
        ode,
        &pk.values,
        theta,
        &eta,
        subject,
        &subject.obs_times,
    );

    // NONMEM 7.6.0 PRED (S2 = V), keyed by observation time — the same table as above.
    let nonmem: &[(f64, f64)] = &[
        (1.0, 0.30468),
        (2.0, 1.0071),
        (4.0, 2.8772),
        (6.0, 3.8508),
        (10.0, 3.6059),
    ];
    assert_eq!(subject.obs_times.len(), nonmem.len());
    assert_eq!(with_states_pred.len(), nonmem.len());
    assert_eq!(dense.len(), nonmem.len());

    for (i, &(t, expected)) in nonmem.iter().enumerate() {
        assert!(
            (subject.obs_times[i] - t).abs() < 1e-9,
            "observation time {} != expected {t}",
            subject.obs_times[i]
        );
        for (engine, got) in [
            ("ode_predictions_with_states", with_states_pred[i]),
            ("ode_dense_solve_states", dense[i][0] / v),
        ] {
            let rel = (got - expected).abs() / expected;
            assert!(
                rel < 1e-4,
                "t={t}: {engine} {got:.5} vs NONMEM {expected:.5} (rel err {rel:.2e})"
            );
        }
    }
}

/// End-to-end `fit()` on an infusion-into-absorption model: the finite-difference sensitivity
/// path (infusion × input-rate is declined at the analytic gates, #719 gap 2) must converge to a
/// sensible objective. Unlike the SS case, an infusion prediction is cheap (no equilibration), so
/// this runs at normal FD-FOCEI speed. The dataset's `DV` is the deterministic NONMEM PRED, so
/// the population minimum sits at the fixed thetas; a bounded fit from there keeps the objective
/// finite and the thetas close.
#[test]
fn fit_on_infusion_absorption_runs() {
    let src = INF_FIRST_ORDER_MODEL.replace("omega ETA_CL ~ 0.0", "omega ETA_CL ~ 0.05");
    let parsed = parse_full_model(&src).expect("model parses");
    let model = parsed.model;
    let population = read_nonmem_csv(std::path::Path::new("data/inf_first_order.csv"), None, None)
        .expect("dataset loads");

    let mut opts = parsed.fit_options;
    opts.outer_maxiter = 5;
    opts.ode_reltol = 1e-8;
    opts.ode_abstol = 1e-8;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("infusion-absorption fit returns Ok via the FD sensitivity fallback");
    assert!(
        result.ofv.is_finite(),
        "infusion-absorption fit objective must be finite, got {}",
        result.ofv
    );
    let truth = [1.0_f64, 20.0, 0.6];
    for (est, tv) in result.theta.iter().zip(truth) {
        let rel = (est - tv).abs() / tv;
        assert!(
            rel < 0.3,
            "infusion fit drifted a theta: got {est:.4}, truth {tv} (rel {rel:.2})"
        );
    }
}
