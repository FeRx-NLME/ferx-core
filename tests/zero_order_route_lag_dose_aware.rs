//! A zero-order route's per-route `lag=` stays on the route for every dose that is not an
//! `SS=1` dose lowered to an infusion (#2786). Non-SS doses behave exactly as with a lag-free
//! route: `TAD` is anchored at the dose record (the route's own delay is the forcing's, not
//! the compartment's), and the model is not a compartment-lag model. The lowered SS doses
//! take the route lag as their own through `DoseAttrMap::lagtime_for_dose`
//! (`tests/ss_absorption_lag_anchor.rs` pins their values against NONMEM).
//!
//! Regression caught: the parser moving the route lag onto the compartment's `ALAG{c}` for
//! ALL doses — `TAD` then reads `TIME - lag` after a dose, and `has_lagtime()` turns true.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, DoseEvent, Population};

mod common;

fn model(clock: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVDUR(2.0, 0.1, 12.0)
  theta TVLAG(1.5, 0.0, 12.0)
  omega ETA_CL ~ 0.0 FIX
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR
  LAG = TVLAG
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central*(1 + 0.3*{clock})
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#
    )
}

fn preds(src: &str) -> (bool, Vec<f64>) {
    let m = parse_full_model(src).expect("parses").model;
    let obs: Vec<f64> = (1..=24).map(|i| i as f64 * 0.5).collect();
    let n = obs.len();
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs,
            vec![0.0; n],
            vec![1; n],
        )],
    };
    let p = predict(&m, &pop, &m.default_params).unwrap();
    (m.has_lagtime(), p.iter().map(|p| p.pred).collect())
}

/// A non-SS dose into a route-lagged `zero_order` reads `TAD` from the dose record: with the
/// single dose at time 0, `TAD` and `TIME` are the same clock, so the two models predict the
/// same. Mutation (`lowered_dose_lag_slot` ignoring `d.ss`, i.e. the route lag moved onto
/// every dose) -> `TAD = TIME - 1.5` and the predictions differ far above the bound.
#[test]
fn non_ss_zero_order_route_lag_anchors_tad_at_the_dose_record() {
    let (lagged_tad, tad) = preds(&model("TAD"));
    let (lagged_time, time) = preds(&model("TIME"));
    assert!(
        !lagged_tad && !lagged_time,
        "a route lag is not a compartment lag"
    );
    // The clock is live: a model without the term predicts something else.
    let (_, none) = preds(&model("0.0"));
    assert!(
        tad.iter().chain(&none).all(|x| x.is_finite()),
        "non-finite prediction"
    );
    let live = tad
        .iter()
        .zip(&none)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max); // nan-ok: `tad` and `none` asserted finite above
    assert!(live > 1e-3, "the TAD term moves no prediction ({live:e})");
    for (j, (a, b)) in tad.iter().zip(&time).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "row {j}: {a} vs {b}");
        assert!(
            (a - b).abs() <= 1e-6 * (1.0 + b.abs()),
            "row {j}: TAD model {a} vs TIME model {b}"
        );
    }
}
