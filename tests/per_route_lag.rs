//! End-to-end tests for **per-route absorption lag** — the universal optional
//! `lag=` argument on an input-rate function (`first_order(ka=KA, lag=L)`,
//! `zero_order(dur=DUR, lag=L)`), which gives each parallel / mixed pathway its own
//! onset delay on top of any compartment lagtime. Exercises the public API
//! (parse → `predict()` → ODE integration → readout).
//!
//! The value-path anchor is a **reduction to an already-anchored term**: a single
//! route with `lag=L` is, by construction, the same time shift as the same route
//! carried on a compartment `lagtime`/`ALAG` — a lag form that is itself validated
//! against NONMEM (`tests/*` ALAG anchors, `docs/`). So a per-route-lagged route must
//! predict **identically** to its compartment-lagged twin. This has real teeth beyond
//! a self-check: the per-route-lag model (no compartment lag) rides the *static*
//! superposition f64 path, while its compartment-lag twin (`has_lagtime()`) rides the
//! *event-driven* walk — so the equality cross-validates two independent integration
//! paths applying the same physical delay. Composed with the linearity of the shared
//! disposition ODE, the parallel / mixed superposition oracles then pin distinct
//! per-route lags and the Channel-A (`first_order`) / Channel-B (`zero_order`) routing.

mod common;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{check_model_data, predict, DoseEvent, Population};

// ── Per-route-lag models under test ──────────────────────────────────────────

/// Single first-order route with its OWN lag (`lag=LAG`), no compartment lagtime.
/// Rides the static superposition path. ka @ theta[2], LAG @ theta[3].
const ROUTE_LAG_SINGLE: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.0, 0.01,  24.0)
  theta TVLAG(2.0, 0.0,  12.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  LAG = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
"#;

/// Compartment-lag twin of [`ROUTE_LAG_SINGLE`]: an unlagged `first_order(ka=KA)`
/// plus a bare `lagtime`. Rides the event-driven walk. Same theta layout, so the
/// two are compared at identical parameters.
const COMP_LAG_SINGLE: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.0, 0.01,  24.0)
  theta TVLAG(2.0, 0.0,  12.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  KA      = TVKA
  lagtime = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// Single zero-order route with its own lag (`zero_order(dur=DUR, lag=LAG)`).
/// dur @ theta[2], LAG @ theta[3].
const ROUTE_LAG_ZERO: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVDUR(3.0, 0.05, 24.0)
  theta TVLAG(2.0, 0.0,  12.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR
  LAG = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
"#;

/// Compartment-lag twin of [`ROUTE_LAG_ZERO`].
const COMP_LAG_ZERO: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVDUR(3.0, 0.05, 24.0)
  theta TVLAG(2.0, 0.0,  12.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  DUR     = TVDUR
  lagtime = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// Parallel with DISTINCT per-route lags: fast route (lag L1) + slow route (lag L2).
/// FR1 @ theta[2], KA1 @ 3, KA2 @ 4, L1 @ 5, L2 @ 6.
const ROUTE_LAG_PARALLEL: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVFR1(0.6, 0.05,  0.95)
  theta TVKA1(1.5, 0.05,  24.0)
  theta TVKA2(0.7, 0.05,  24.0)
  theta TVL1(0.5,  0.0,   12.0)
  theta TVL2(3.0,  0.0,   12.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  FR1 = TVFR1
  FR2 = 1 - TVFR1
  KA1 = TVKA1
  KA2 = TVKA2
  L1  = TVL1
  L2  = TVL2
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1, lag=L1) + FR2*first_order(ka=KA2, lag=L2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
"#;

/// Mixed: first-order route (no lag) + zero-order route WITH a lag. FZO @ theta[2],
/// KA @ 3, DUR @ 4, LAG @ 5.
const ROUTE_LAG_MIXED: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVFZO(0.4, 0.05,  0.95)
  theta TVKA(1.0,  0.05,  24.0)
  theta TVDUR(3.0, 0.05,  24.0)
  theta TVLAG(2.0, 0.0,   12.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  FZO  = TVFZO
  FZO1 = 1 - TVFZO
  KA   = TVKA
  DUR  = TVDUR
  LAG  = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
"#;

// ── Single-route compartment-lag reference curves (the anchored twins) ────────

/// Single-route reference at `ka` with a compartment lag `lag`.
fn comp_lag_first_order_curve(ka: f64, lag: f64, obs_times: &[f64]) -> Vec<f64> {
    let model = parse_full_model(COMP_LAG_SINGLE)
        .expect("comp-lag single first_order parses")
        .model;
    let mut params = model.default_params.clone();
    params.theta[2] = ka;
    params.theta[3] = lag;
    predict(&model, &pop_single(obs_times.to_vec()), &params)
        .iter()
        .map(|p| p.pred)
        .collect()
}

/// Single zero-order reference at `dur` with a compartment lag `lag`.
fn comp_lag_zero_order_curve(dur: f64, lag: f64, obs_times: &[f64]) -> Vec<f64> {
    let model = parse_full_model(COMP_LAG_ZERO)
        .expect("comp-lag single zero_order parses")
        .model;
    let mut params = model.default_params.clone();
    params.theta[2] = dur;
    params.theta[3] = lag;
    predict(&model, &pop_single(obs_times.to_vec()), &params)
        .iter()
        .map(|p| p.pred)
        .collect()
}

/// Doses into CMT 1 at each of `dose_times` (each 100 mg, fed to the input rate),
/// observed at `obs_times` on CMT 1.
fn pop_doses(dose_times: &[f64], obs_times: Vec<f64>) -> Population {
    let n = obs_times.len();
    let doses: Vec<DoseEvent> = dose_times
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
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
            vec![1; n],
        )],
    }
}

/// Single 100 mg bolus into CMT 1.
fn pop_single(obs_times: Vec<f64>) -> Population {
    pop_doses(&[0.0], obs_times)
}

fn predict_curve(src: &str, obs_times: &[f64]) -> Vec<f64> {
    let model = parse_full_model(src).expect("model parses").model;
    predict(
        &model,
        &pop_single(obs_times.to_vec()),
        &model.default_params,
    )
    .iter()
    .map(|p| p.pred)
    .collect()
}

fn assert_close(a: &[f64], b: &[f64], obs_times: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len());
    for (i, &t) in obs_times.iter().enumerate() {
        assert!(
            (a[i] - b[i]).abs() <= tol * (1.0 + b[i].abs()),
            "{what} diverge at t={t}: {} vs {}",
            a[i],
            b[i]
        );
    }
}

// ── Oracles ──────────────────────────────────────────────────────────────────

#[test]
fn route_lag_equals_compartment_lag_first_order() {
    // first_order(ka=KA, lag=L)  ≡  first_order(ka=KA) + compartment lagtime L.
    // Cross-path: route lag on the static walk vs compartment lag on the event-driven
    // walk. Both apply the same physical onset shift, so the curves must coincide.
    let obs: Vec<f64> = (0..=120).map(|i| i as f64 * 0.25).collect(); // 0..30 h
    let route = predict_curve(ROUTE_LAG_SINGLE, &obs);
    let comp = comp_lag_first_order_curve(1.0, 2.0, &obs);
    assert_close(
        &route,
        &comp,
        &obs,
        1e-5,
        "route-lag vs comp-lag (first-order)",
    );
    // A real delayed curve: ~0 before the lag, a real peak after.
    let before: f64 = obs
        .iter()
        .zip(&route)
        .filter(|(&t, _)| t < 2.0)
        .map(|(_, &c)| c.abs())
        .fold(0.0, f64::max);
    assert!(
        before < 1e-6,
        "concentration must be ~0 before the lag; got {before}"
    );
    assert!(
        route.iter().cloned().fold(0.0, f64::max) > 0.5,
        "expected a real peak"
    );
}

#[test]
fn route_lag_equals_compartment_lag_zero_order() {
    // zero_order(dur=DUR, lag=L) ≡ zero_order(dur=DUR) + compartment lagtime L. Pins
    // the window-edge shift (w_start / w_end / the cutoff break all move by L) — the
    // #504 mass-exactness invariant must hold under a route lag.
    let obs: Vec<f64> = (0..=120).map(|i| i as f64 * 0.25).collect();
    let route = predict_curve(ROUTE_LAG_ZERO, &obs);
    let comp = comp_lag_zero_order_curve(3.0, 2.0, &obs);
    assert_close(
        &route,
        &comp,
        &obs,
        1e-5,
        "route-lag vs comp-lag (zero-order)",
    );
    let before: f64 = obs
        .iter()
        .zip(&route)
        .filter(|(&t, _)| t < 2.0)
        .map(|(_, &c)| c.abs())
        .fold(0.0, f64::max);
    assert!(
        before < 1e-6,
        "zero-order onset must wait for the lag; got {before}"
    );
}

#[test]
fn parallel_distinct_route_lags_equal_superposition() {
    // FR1·first(ka1, lag=L1) + FR2·first(ka2, lag=L2)
    //   ≡ FR1·comp_lag_single(ka1, L1) + FR2·comp_lag_single(ka2, L2),
    // by ODE linearity. Pins DISTINCT per-route lags + the fraction split at once.
    //
    // Tolerance note (measured, not a fudge): the single-route oracles above show the
    // lag *logic* is exact to 1e-5 (they match the analytic Bateman-with-lag). Here the
    // reference sums two ISOLATED single-route integrations, each resolving its own
    // onset with a small central value; the `parallel` model integrates both routes at
    // once, so at the delayed onset (t=3) the fast route's large central value (~1.6)
    // loosens the RK45 relative error control (`scale_tol = abstol + reltol·|u|`) and
    // the sharp `R_in` input is slightly under-resolved — a fixed ~0.5% mass deficit
    // that then decays as e^(−ke·t). It is adaptive-quadrature at a mid-integration
    // onset, NOT a lag error, and it shrinks with denser sampling across the onset
    // (5e-3 → 5e-4 from 0.25 h → 0.01 h spacing). We therefore sample across both
    // onsets (as one would for a delayed-release drug) and hold 2e-3.
    let mut obs: Vec<f64> = (0..=160).map(|i| i as f64 * 0.25).collect(); // 0..40 h
    for j in 0..12 {
        obs.push(0.5 + j as f64 * 0.02); // across the fast onset
        obs.push(3.0 + j as f64 * 0.02); // across the slow onset
    }
    obs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    obs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    let parallel = predict_curve(ROUTE_LAG_PARALLEL, &obs);
    let (fr1, fr2) = (0.6, 0.4);
    let c1 = comp_lag_first_order_curve(1.5, 0.5, &obs);
    let c2 = comp_lag_first_order_curve(0.7, 3.0, &obs);
    let want: Vec<f64> = (0..obs.len()).map(|i| fr1 * c1[i] + fr2 * c2[i]).collect();
    assert_close(
        &parallel,
        &want,
        &obs,
        2e-3,
        "parallel per-route-lag superposition",
    );
    // Two distinct onsets: fast route from ~0.5 h, slow route only after 3 h — so the
    // slope must visibly increase across t=3 (a second phase), not a single curve.
    let idx = |tt: f64| obs.iter().position(|&t| (t - tt).abs() < 1e-9).unwrap();
    let slope = |t0: f64, t1: f64| (parallel[idx(t1)] - parallel[idx(t0)]) / (t1 - t0);
    assert!(
        slope(3.25, 4.0) > slope(2.0, 2.75),
        "the slow route's onset at t=3 must add a second absorption phase"
    );
}

#[test]
fn mixed_zero_order_route_lag_equals_superposition() {
    // FZO1·first(ka) + FZO·zero(dur, lag=L) ≡ FZO1·single_first(ka) +
    // FZO·comp_lag_zero(dur, L). Pins the lagged Channel-B window against the
    // unlagged Channel-A term.
    let obs: Vec<f64> = (0..=120).map(|i| i as f64 * 0.25).collect();
    let mixed = predict_curve(ROUTE_LAG_MIXED, &obs);
    let (fzo1, fzo) = (0.6, 0.4);
    // first-order (no lag) reference: comp-lag twin with lag 0.
    let c_fo = comp_lag_first_order_curve(1.0, 0.0, &obs);
    let c_zo = comp_lag_zero_order_curve(3.0, 2.0, &obs);
    let want: Vec<f64> = (0..obs.len())
        .map(|i| fzo1 * c_fo[i] + fzo * c_zo[i])
        .collect();
    assert_close(
        &mixed,
        &want,
        &obs,
        1e-5,
        "mixed zero-order-route-lag superposition",
    );
}

#[test]
fn zero_route_lag_is_bit_identical_to_no_lag() {
    // Degenerate oracle: lag=0 must reproduce the unlagged route bit-for-bit — the
    // feature is a strict superset, never perturbing an existing model.
    const NO_LAG: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.0, 0.01,  24.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    const LAG_ZERO: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.0, 0.01,  24.0)
  theta TVLAG(0.0, 0.0,  12.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  LAG = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let obs: Vec<f64> = (0..=80).map(|i| i as f64 * 0.25).collect();
    let no_lag = predict_curve(NO_LAG, &obs);
    let lag_zero = predict_curve(LAG_ZERO, &obs);
    for (i, &t) in obs.iter().enumerate() {
        assert_eq!(
            no_lag[i], lag_zero[i],
            "lag=0 must be bit-identical to no lag at t={t}"
        );
    }
}

#[test]
fn route_lag_multi_dose_equals_compartment_lag() {
    // Superposition across MULTIPLE doses: a per-route lag must shift EVERY dose's
    // onset, matching the compartment-lag twin dose-for-dose (not just the first).
    let obs: Vec<f64> = (0..=200).map(|i| i as f64 * 0.25).collect(); // 0..50 h
    let dose_times = [0.0, 12.0, 24.0];
    let route = {
        let m = parse_full_model(ROUTE_LAG_SINGLE).expect("parses").model;
        predict(&m, &pop_doses(&dose_times, obs.clone()), &m.default_params)
            .iter()
            .map(|p| p.pred)
            .collect::<Vec<_>>()
    };
    let comp = {
        let m = parse_full_model(COMP_LAG_SINGLE).expect("parses").model;
        predict(&m, &pop_doses(&dose_times, obs.clone()), &m.default_params)
            .iter()
            .map(|p| p.pred)
            .collect::<Vec<_>>()
    };
    assert_close(
        &route,
        &comp,
        &obs,
        1e-5,
        "multi-dose route-lag vs comp-lag",
    );
}

#[test]
fn route_lag_on_event_driven_path_equals_compartment_lag() {
    // A subject with an EVID=3 reset forces the EVENT-DRIVEN walk (not the static
    // superposition path the tests above exercise). With a dose after the reset, the
    // post-reset route onset (`dose + lag`) must be broken on the event-driven timeline
    // too — so route-lag ≡ compartment-lag here cross-validates the event-driven
    // route-onset break AND the reset × per-route-lag combination.
    let obs: Vec<f64> = (2..=96).map(|i| i as f64 * 0.25).collect(); // 0.5..24 h
    let reset_at = 12.0;
    let dose_times = [0.0, 13.0]; // one dose before, one after the reset
    let build = |src: &str| -> Vec<f64> {
        let m = parse_full_model(src).expect("parses").model;
        let n = obs.len();
        let doses: Vec<DoseEvent> = dose_times
            .iter()
            .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
            .collect();
        let mut subj = common::subject("1", doses, obs.clone(), vec![0.0; n], vec![1; n]);
        subj.reset_times = vec![reset_at];
        let pop = Population {
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
            subjects: vec![subj],
        };
        predict(&m, &pop, &m.default_params)
            .iter()
            .map(|p| p.pred)
            .collect()
    };
    let route = build(ROUTE_LAG_SINGLE);
    let comp = build(COMP_LAG_SINGLE);
    assert_close(
        &route,
        &comp,
        &obs,
        1e-5,
        "reset event-driven route-lag vs comp-lag",
    );
    // The reset zeros state at t=12; the post-reset dose at 13 (+lag 2) re-absorbs, so
    // there is real signal after the reset (not an all-zero degenerate match).
    let after: f64 = obs
        .iter()
        .zip(&route)
        .filter(|(&t, _)| t > 15.5)
        .map(|(_, &c)| c)
        .fold(0.0, f64::max);
    assert!(
        after > 0.3,
        "post-reset re-dose must produce real signal, got {after}"
    );
}

/// Tier-2 (fast): a per-route-lag model runs through the optimizer end-to-end. Since #859
/// a `first_order` per-route lag is served on the exact analytic path (the gate admits it),
/// so this now exercises the analytic gradient wiring. A couple of outer iterations, no
/// convergence loop — just proves the fit path is wired and returns a finite OFV.
#[test]
fn per_route_lag_fit_runs_end_to_end() {
    use ferx_core::{fit, FitOptions};
    let model = parse_full_model(ROUTE_LAG_SINGLE).expect("parses").model;
    let obs_times: Vec<f64> = (1..=20).map(|i| i as f64 * 0.5).collect();
    let truth = predict(
        &model,
        &pop_single(obs_times.clone()),
        &model.default_params,
    );
    let obs: Vec<f64> = truth.iter().map(|p| p.pred).collect();
    let n = obs_times.len();
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject("1", vec![dose], obs_times, obs, vec![1; n])],
    };
    let opts = FitOptions {
        outer_maxiter: 2,
        ..FitOptions::default()
    };
    let fitted = fit(&model, &pop, &model.default_params, &opts).expect("fit runs end-to-end");
    assert!(
        fitted.ofv.is_finite(),
        "OFV must be finite, got {}",
        fitted.ofv
    );
}

/// Tier-3 (slow): a full FOCEI fit (exact analytic gradients since #859) recovers the
/// per-route lag from a perturbed start on noise-free data — the optimum sits at the
/// data-generating lag.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn per_route_lag_fit_recovers_lag() {
    use ferx_core::{fit, FitOptions};
    let model = parse_full_model(ROUTE_LAG_SINGLE).expect("parses").model;
    // Noise-free data at truth (TVLAG = theta[3] = 2.0): the likelihood optimum is the
    // data-generating lag. Sample across the onset so it is well-identified.
    let mut obs_times: Vec<f64> = (1..=48).map(|i| i as f64 * 0.5).collect();
    for j in 0..12 {
        obs_times.push(2.0 + j as f64 * 0.05);
    }
    obs_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    obs_times.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    let truth = predict(
        &model,
        &pop_single(obs_times.clone()),
        &model.default_params,
    );
    let obs: Vec<f64> = truth.iter().map(|p| p.pred).collect();
    let n = obs_times.len();
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject("1", vec![dose], obs_times, obs, vec![1; n])],
    };
    let mut init = model.default_params.clone();
    init.theta[2] = 0.7; // KA off truth (1.0)
    init.theta[3] = 3.0; // LAG off truth (2.0)
    let fitted = fit(&model, &pop, &init, &FitOptions::default()).expect("fit converges");
    assert!(fitted.converged, "per-route-lag fit should converge");
    assert!(
        (fitted.theta[3] - 2.0).abs() < 0.1,
        "recovered per-route lag {} should be near 2.0",
        fitted.theta[3]
    );
}

#[test]
fn shipped_per_route_lag_example_parses() {
    // A typo in the shipped example is caught here.
    let src = std::fs::read_to_string("examples/per_route_lag_absorption.ferx")
        .expect("read per_route_lag_absorption.ferx");
    parse_full_model(&src).expect("per_route_lag_absorption.ferx parses");
}

// ── `lag=` is universal: transit() and igd() carry it too ─────────────────────
//
// The `lag=` argument is captured kind-agnostically by the parser and applied at a
// single, kind-agnostic `t_eff` site in `add_prepared_input_rate_forcing`, so the
// `first_order` / `zero_order` oracles above already exercise the shared code path.
// These two tests pin the *end-to-end numerical* behaviour for the remaining smooth
// kernels — `transit(n, mtt)` and `igd(mat, cv2)` — with the same reduction anchor:
// a route with `lag=L` must predict identically to that route on a compartment
// `lagtime`/`ALAG` (a NONMEM-anchored lag form), to 1e-5. Both are run in the
// non-flip-flop regime (KTR / IG tilting abscissa ≫ ke), i.e. exactly where the
// closed-form value path would be selected if it were eligible — so if the shared
// `t_eff` shift were ever bypassed for these kernels, the pre-lag "must be ~0"
// assertion would catch it.

const TRANSIT_ROUTE_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVN(3.0, 0.1, 20.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVLAG(2.0, 0.0, 12.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  N   = TVN
  MTT = TVMTT
  LAG = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = transit(n=N, mtt=MTT, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const TRANSIT_COMP_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVN(3.0, 0.1, 20.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVLAG(2.0, 0.0, 12.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  N       = TVN
  MTT     = TVMTT
  lagtime = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = transit(n=N, mtt=MTT) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const IGD_ROUTE_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMAT(2.0, 0.05, 24.0)
  theta TVCV2(0.5, 0.01, 10.0)
  theta TVLAG(2.0, 0.0, 12.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  MAT = TVMAT
  CV2 = TVCV2
  LAG = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const IGD_COMP_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMAT(2.0, 0.05, 24.0)
  theta TVCV2(0.5, 0.01, 10.0)
  theta TVLAG(2.0, 0.0, 12.0)
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  MAT     = TVMAT
  CV2     = TVCV2
  lagtime = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// `transit(n, mtt, lag=L)` ≡ `transit(n, mtt)` + compartment lagtime L (1e-5), and
/// the curve is genuinely delayed (~0 before the lag, a real rise after).
#[test]
fn route_lag_equals_compartment_lag_transit() {
    let obs: Vec<f64> = (0..=120).map(|i| i as f64 * 0.25).collect(); // 0..30 h
    let route = predict_curve(TRANSIT_ROUTE_LAG, &obs);
    let comp = predict_curve(TRANSIT_COMP_LAG, &obs);
    assert_close(&route, &comp, &obs, 1e-5, "route-lag vs comp-lag (transit)");
    let before = obs
        .iter()
        .zip(&route)
        .filter(|(&t, _)| t < 2.0)
        .map(|(_, &c)| c.abs())
        .fold(0.0, f64::max);
    let after = obs
        .iter()
        .zip(&route)
        .filter(|(&t, _)| t > 6.0)
        .map(|(_, &c)| c.abs())
        .fold(0.0, f64::max);
    assert!(
        before < 1e-6,
        "transit conc must be ~0 before the lag; got {before}"
    );
    assert!(
        after > 1e-3,
        "transit conc must rise after the lag; got {after}"
    );
}

/// `igd(mat, cv2, lag=L)` ≡ `igd(mat, cv2)` + compartment lagtime L (1e-5).
#[test]
fn route_lag_equals_compartment_lag_igd() {
    let obs: Vec<f64> = (0..=120).map(|i| i as f64 * 0.25).collect();
    let route = predict_curve(IGD_ROUTE_LAG, &obs);
    let comp = predict_curve(IGD_COMP_LAG, &obs);
    assert_close(&route, &comp, &obs, 1e-5, "route-lag vs comp-lag (igd)");
    let before = obs
        .iter()
        .zip(&route)
        .filter(|(&t, _)| t < 2.0)
        .map(|(_, &c)| c.abs())
        .fold(0.0, f64::max);
    let after = obs
        .iter()
        .zip(&route)
        .filter(|(&t, _)| t > 6.0)
        .map(|(_, &c)| c.abs())
        .fold(0.0, f64::max);
    assert!(
        before < 1e-6,
        "igd conc must be ~0 before the lag; got {before}"
    );
    assert!(
        after > 1e-3,
        "igd conc must rise after the lag; got {after}"
    );
}

/// Multi-dose transit + route lag ≡ multi-dose transit + compartment lag — the route
/// offset is re-applied per dose (mirrors `route_lag_multi_dose_equals_compartment_lag`
/// for the `first_order` kernel).
#[test]
fn route_lag_multi_dose_equals_compartment_lag_transit() {
    let obs: Vec<f64> = (0..=160).map(|i| i as f64 * 0.25).collect();
    let dose_times = [0.0_f64, 12.0, 24.0];
    let route = {
        let m = parse_full_model(TRANSIT_ROUTE_LAG).expect("parses").model;
        predict(&m, &pop_doses(&dose_times, obs.clone()), &m.default_params)
            .iter()
            .map(|p| p.pred)
            .collect::<Vec<_>>()
    };
    let comp = {
        let m = parse_full_model(TRANSIT_COMP_LAG).expect("parses").model;
        predict(&m, &pop_doses(&dose_times, obs.clone()), &m.default_params)
            .iter()
            .map(|p| p.pred)
            .collect::<Vec<_>>()
    };
    assert_close(
        &route,
        &comp,
        &obs,
        1e-5,
        "transit multi-dose route vs comp",
    );
}

// ── Composition with SS (#834) and infusion (#836) into an absorption compartment ──
//
// Both landed on `main` after this branch forked, editing the same forcing machinery.
// A per-route lag lives on the forcing (`lag_slot`), NOT `has_lagtime()`, so these guard
// the two combinations: SS must be *rejected* (the SS trough seed cannot route a lagged
// onset — the same limitation that rejects SS + compartment lag, `E_ABSORPTION_SS_LAG`),
// while a plain infusion must *compose* (a single infusion whose whole appearance is
// simply delayed by the route lag).

/// SS=1 into `first_order(ka, lag=L)` must raise `E_ABSORPTION_SS_LAG` — the per-route
/// lag is invisible to `has_lagtime_on_cmt`, so without the forcing-`lag_slot` arm of the
/// gate this SS dose would slip through and be silently mis-served by the (lag-unaware)
/// steady-state equilibration seed.
#[test]
fn ss_into_per_route_lag_is_rejected() {
    let model = parse_full_model(ROUTE_LAG_SINGLE).expect("parses").model;
    let obs: Vec<f64> = (0..=24).map(|i| i as f64 * 0.5).collect();
    let n = obs.len();
    let population = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)], // SS=1, II=12
            obs,
            vec![0.0; n],
            vec![1; n],
        )],
    };
    // `check_model_data` is the entry `fit()` runs (and aborts on the first error);
    // the SS/absorption gates live there, not in `check_model_data_warnings`.
    let diags = check_model_data(&model, &population);
    assert!(
        diags.iter().any(|d| d.code == "E_ABSORPTION_SS_LAG"),
        "SS into a per-route-lagged absorption compartment must be rejected with \
         E_ABSORPTION_SS_LAG (per-route lag is not seen by has_lagtime_on_cmt); got: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

/// A plain infusion (`RATE>0`) into `first_order(ka, lag=L)` composes correctly: it equals
/// the same infusion into `first_order(ka)` + compartment lagtime L (both delay the whole
/// infused-kernel appearance by L). No SS seed involved, so `tad` carries the route lag.
#[test]
fn infusion_into_per_route_lag_equals_compartment_lag() {
    let obs: Vec<f64> = (0..=60).map(|i| i as f64 * 0.5).collect();
    let inf = |src: &str| {
        let m = parse_full_model(src).expect("parses").model;
        let n = obs.len();
        let pop = Population {
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
            subjects: vec![common::subject(
                "1",
                vec![DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0)], // rate 50 → 2 h infusion
                obs.clone(),
                vec![0.0; n],
                vec![1; n],
            )],
        };
        predict(&m, &pop, &m.default_params)
            .iter()
            .map(|p| p.pred)
            .collect::<Vec<_>>()
    };
    let route = inf(ROUTE_LAG_SINGLE);
    let comp = inf(COMP_LAG_SINGLE);
    assert_close(&route, &comp, &obs, 1e-5, "infusion route-lag vs comp-lag");
    // Genuinely delayed: ~0 before the lag.
    let before = obs
        .iter()
        .zip(&route)
        .filter(|(&t, _)| t < 2.0)
        .map(|(_, &c)| c.abs())
        .fold(0.0, f64::max);
    assert!(
        before < 1e-6,
        "infused conc must be ~0 before the lag; got {before}"
    );
}
