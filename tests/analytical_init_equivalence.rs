//! Equivalence between the analytical `[initial_conditions]` block (issue #521)
//! and its ODE transcription `init(state) = <expr>` in `[odes]`.
//!
//! A non-zero initial compartment amount on an analytical (closed-form) model
//! is layered on as the closed-form impulse of an F-bypassed bolus. This test
//! builds a 1-cpt oral model with a parameter-dependent central baseline
//! (`init(central) = 30 * V`) plus an oral dose, then `predict()`s both the
//! analytical form and a hand-written ODE twin and asserts the population PRED
//! agrees pointwise. This is the same structural-equivalence strategy as
//! `analytical_ode_equivalence.rs`, extended to the initial-condition path.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::predict;
use ferx_core::types::{DoseEvent, Population};

mod common;

// RK45 defaults (abstol 1e-6, reltol 1e-4) cap how exactly the ODE twin can
// reproduce the closed form; use a combined absolute+relative bound so the
// near-zero early-absorption points don't blow up a pure relative check.
const ATOL: f64 = 1e-5;
const RTOL: f64 = 1e-4;

const THETAS_AND_INDIV: &str = r"
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL ~ 0.09

  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
";

fn analytical_src() -> String {
    format!(
        "{THETAS_AND_INDIV}
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[initial_conditions]
  init(central) = 30 * V

[error_model]
  DV ~ proportional(PROP_ERR)
"
    )
}

fn ode_src() -> String {
    // ODE twin: central state holds an AMOUNT, seeded to 30*V; the observed
    // concentration is amount/V via `obs_scale = V` (NONMEM's S2). This mirrors
    // the analytical model, whose closed form reads concentration directly.
    format!(
        "{THETAS_AND_INDIV}
[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  init(central) = 30 * V
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - CL / V * central

[scaling]
  obs_scale = V

[error_model]
  DV ~ proportional(PROP_ERR)
"
    )
}

fn population() -> Population {
    // One oral dose (depot = cmt 1) plus a baseline already in central; observe
    // across the absorption rise and the baseline decay.
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let obs_times = vec![0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0];
    let n = obs_times.len();
    let subj = common::subject("1", doses, obs_times, vec![0.0; n], vec![2; n]);
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subj],
    }
}

#[test]
fn analytical_central_init_matches_ode_init() {
    let an = parse_full_model(&analytical_src())
        .expect("analytical model parses")
        .model;
    let ode = parse_full_model(&ode_src())
        .expect("ODE model parses")
        .model;

    // The analytical model must actually carry the parsed init.
    assert_eq!(
        an.analytical_init.len(),
        1,
        "[initial_conditions] should populate analytical_init"
    );
    assert!(
        ode.analytical_init.is_empty(),
        "ODE model seeds state via init_fn, not analytical_init"
    );

    let pop = population();
    let pa = predict(&an, &pop, &an.default_params);
    let po = predict(&ode, &pop, &ode.default_params);
    assert_eq!(pa.len(), po.len());
    assert!(!pa.is_empty());

    // At t=0 the baseline dominates: concentration = 30 (amount 30*V over V).
    assert!(
        (pa[0].pred - 30.0).abs() < 1e-6,
        "analytical baseline at t=0 should be 30, got {}",
        pa[0].pred
    );

    for (x, y) in pa.iter().zip(po.iter()) {
        let tol = ATOL + RTOL * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "t={:.3}: analytical PRED {:.6} vs ODE PRED {:.6} (|diff| {:.2e} > tol {:.2e})",
            x.time,
            x.pred,
            y.pred,
            (x.pred - y.pred).abs(),
            tol
        );
    }
}

/// Both engines' initial-condition blocks accept a read of a **dose attribute**, and
/// they must agree on what it means.
///
/// The two carve-outs shipped separately — analytical `[initial_conditions]` in
/// #1004/#1035, ODE `[odes] init(...)` in #1046 — each argued from its own engine's
/// source that an initial condition is not an absorbed dose, so `F` applies to it
/// exactly once and the seed is not a double use. Nothing tied the two arguments
/// together. This does: one model, `f=F` on the analytical side and the same `F`
/// routed to the engine's dose slot by name on the ODE side, seeding an identical
/// baseline while `F` also scales the oral dose.
///
/// If either engine ever started applying `F` to its seed, this diverges — the
/// analytical `add_analytical_init` impulse is F-bypassed (`analytical_init_concentration`
/// propagates with `F = 1`) and `OdeSpec::initial_state` writes the raw value, so
/// today they coincide. NONMEM agrees with both
/// (`nonmem_anchor/odes_init_dose_attr_f_{A,B}.ctl`, #1046).
#[test]
fn analytical_and_ode_init_agree_when_the_seed_reads_bioavailability() {
    const F_THETAS: &str = r"
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVF(0.5, 0.0, 1.0)

  omega ETA_CL ~ 0.09

  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = TVF
";
    let an = parse_full_model(&format!(
        "{F_THETAS}
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)

[initial_conditions]
  init(central) = F * 30 * V

[error_model]
  DV ~ proportional(PROP_ERR)
"
    ))
    .expect("analytical model with an F-reading init parses (#1004/#1035)")
    .model;

    let ode = parse_full_model(&format!(
        "{F_THETAS}
[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  init(central) = F * 30 * V
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - CL / V * central

[scaling]
  obs_scale = V

[error_model]
  DV ~ proportional(PROP_ERR)
"
    ))
    .expect("ODE model with an F-reading init parses (#1046)")
    .model;

    let pop = population();
    let pa = predict(&an, &pop, &an.default_params);
    let po = predict(&ode, &pop, &ode.default_params);
    assert_eq!(pa.len(), po.len());
    assert!(!pa.is_empty());

    // The seed carries `F` exactly once: baseline concentration at t=0 is
    // `F * 30 * V / V = 0.5 * 30 = 15`. A seed that also went through the dose's
    // bioavailability would read 7.5 here, and one that ignored `F` would read 30 —
    // so this single value separates all three readings.
    assert!(
        (pa[0].pred - 15.0).abs() < 1e-6,
        "analytical baseline at t=0 should be F*30 = 15, got {}",
        pa[0].pred
    );

    for (x, y) in pa.iter().zip(po.iter()) {
        let tol = ATOL + RTOL * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "t={:.3}: analytical PRED {:.6} vs ODE PRED {:.6} (|diff| {:.2e} > tol {:.2e})",
            x.time,
            x.pred,
            y.pred,
            (x.pred - y.pred).abs(),
            tol
        );
    }
}

/// `predict_iov` is the prediction path the importance-sampling (IMP) estimator
/// and the IOV likelihood use — even for non-IOV models. It must carry the
/// initial-compartment amount too, otherwise IMP mispredicts baseline subjects
/// and their importance weights collapse. Here we check that, for the non-IOV
/// init model, `predict_iov` (empty kappas) agrees with the public `predict()`.
#[test]
fn predict_iov_carries_analytical_init() {
    use ferx_core::pk::predict_iov;

    let an = parse_full_model(&analytical_src())
        .expect("analytical model parses")
        .model;
    let pop = population();
    let subject = &pop.subjects[0];

    let public = predict(&an, &pop, &an.default_params);
    // Non-IOV: no kappa groups. `predict_iov` takes the theta slice directly.
    let iov = predict_iov(&an, subject, &an.default_params.theta, &[0.0; 1], &[]);

    assert_eq!(iov.len(), public.len());
    // Baseline at t=0 must survive the predict_iov path (≈30), not be dropped.
    assert!(
        (iov[0] - 30.0).abs() < 1e-6,
        "predict_iov baseline at t=0 should be 30, got {}",
        iov[0]
    );
    for (i, p) in public.iter().enumerate() {
        let tol = ATOL + RTOL * p.pred.abs();
        assert!(
            (iov[i] - p.pred).abs() <= tol,
            "obs {i}: predict_iov {:.6} vs predict() {:.6}",
            iov[i],
            p.pred
        );
    }
}

// ── Multi-compartment central baselines (issue #521 review) ──────────────────
// The 2-/3-cpt branches of `analytical_init_concentration` had no coverage; the
// shipped tests above are all 1-cpt. These exercise the central IV-bolus impulse
// for 2-cpt and 3-cpt against hand-written ODE twins.

const MULTI_PARAMS: &str = r"
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVQ(2.0, 0.01, 100.0)
  theta TVV2(40.0, 1.0, 500.0)
  theta TVQ3(0.5, 0.01, 100.0)
  theta TVV3(80.0, 1.0, 500.0)

  omega ETA_CL ~ 0.09

  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  Q  = TVQ
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
";

fn iv_population() -> Population {
    // IV bolus into central (cmt 1) plus a baseline already in central; observe
    // across the distribution and elimination phases.
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let obs_times = vec![0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0];
    let n = obs_times.len();
    let subj = common::subject("1", doses, obs_times, vec![0.0; n], vec![1; n]);
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subj],
    }
}

fn assert_pred_agrees(an: &str, ode: &str, pop: &Population, t0_conc: f64) {
    let an = parse_full_model(an).expect("analytical model parses").model;
    let ode = parse_full_model(ode).expect("ODE model parses").model;
    assert_eq!(an.analytical_init.len(), 1);

    let pa = predict(&an, pop, &an.default_params);
    let po = predict(&ode, pop, &ode.default_params);
    assert_eq!(pa.len(), po.len());
    assert!(!pa.is_empty());
    assert!(
        (pa[0].pred - t0_conc).abs() < 1e-6,
        "analytical baseline at t=0 should be {t0_conc}, got {}",
        pa[0].pred
    );
    for (x, y) in pa.iter().zip(po.iter()) {
        let tol = ATOL + RTOL * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "t={:.3}: analytical PRED {:.6} vs ODE PRED {:.6} (|diff| {:.2e} > tol {:.2e})",
            x.time,
            x.pred,
            y.pred,
            (x.pred - y.pred).abs(),
            tol
        );
    }
}

#[test]
fn analytical_2cpt_central_init_matches_ode_init() {
    // baseline amount 30*V → concentration 30 at t=0.
    let an = format!(
        "{MULTI_PARAMS}
[structural_model]
  pk two_cpt_iv(cl=CL, v=V, q=Q, v2=V2)

[initial_conditions]
  init(central) = 30 * V

[error_model]
  DV ~ proportional(PROP_ERR)
"
    );
    let ode = format!(
        "{MULTI_PARAMS}
[structural_model]
  ode(obs_cmt=central, states=[central, periph])

[odes]
  init(central)  = 30 * V
  d/dt(central)  = -(CL / V) * central - (Q / V) * central + (Q / V2) * periph
  d/dt(periph)   =  (Q / V) * central - (Q / V2) * periph

[scaling]
  obs_scale = V

[error_model]
  DV ~ proportional(PROP_ERR)
"
    );
    // t=0: baseline 30 (=30*V/V) + IV bolus 100/V=5 → 35.
    assert_pred_agrees(&an, &ode, &iv_population(), 35.0);
}

#[test]
fn analytical_3cpt_central_init_matches_ode_init() {
    let an = format!(
        "{MULTI_PARAMS}
[structural_model]
  pk three_cpt_iv(cl=CL, v=V, q=Q, v2=V2, q3=Q3, v3=V3)

[initial_conditions]
  init(central) = 30 * V

[error_model]
  DV ~ proportional(PROP_ERR)
"
    );
    let ode = format!(
        "{MULTI_PARAMS}
[structural_model]
  ode(obs_cmt=central, states=[central, periph, periph2])

[odes]
  init(central)  = 30 * V
  d/dt(central)  = -(CL / V) * central - (Q / V) * central + (Q / V2) * periph - (Q3 / V) * central + (Q3 / V3) * periph2
  d/dt(periph)   =  (Q / V) * central - (Q / V2) * periph
  d/dt(periph2)  =  (Q3 / V) * central - (Q3 / V3) * periph2

[scaling]
  obs_scale = V

[error_model]
  DV ~ proportional(PROP_ERR)
"
    );
    // t=0: baseline 30 (=30*V/V) + IV bolus 100/V=5 → 35.
    assert_pred_agrees(&an, &ode, &iv_population(), 35.0);
}

/// A system reset (EVID=3/4) zeros every compartment, including the residual
/// init baseline; nothing re-deposits the t=0 amount. Observations at/after the
/// first reset must therefore carry NO baseline contribution (issue #521 review:
/// the old code decayed the baseline across the reset-shifted clock, double-
/// applying it to every post-reset occasion).
#[test]
fn analytical_init_zeroed_after_reset() {
    let an = parse_full_model(&analytical_src())
        .expect("analytical model parses")
        .model;

    // No doses: the only signal is the central baseline (30*V → conc 30 at t=0).
    // A reset at t=5 must wipe it, so obs at t>=5 see 0 baseline.
    let obs_times = vec![0.0, 1.0, 5.0, 8.0];
    let n = obs_times.len();
    let mut subj = common::subject("1", vec![], obs_times, vec![0.0; n], vec![2; n]);
    subj.reset_times = vec![5.0];
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subj],
    };

    let pa = predict(&an, &pop, &an.default_params);
    assert!(
        (pa[0].pred - 30.0).abs() < 1e-6,
        "pre-reset baseline at t=0 should be 30, got {}",
        pa[0].pred
    );
    assert!(
        pa[1].pred > 0.0 && pa[1].pred < 30.0,
        "pre-reset baseline at t=1 should decay below 30, got {}",
        pa[1].pred
    );
    assert!(
        pa[2].pred.abs() < 1e-12,
        "baseline at reset time t=5 should be wiped to 0, got {}",
        pa[2].pred
    );
    assert!(
        pa[3].pred.abs() < 1e-12,
        "baseline after reset (t=8) should stay 0, got {}",
        pa[3].pred
    );
}
