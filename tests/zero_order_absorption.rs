//! End-to-end tests for built-in **zero-order absorption** (#322 Phase 2, #504).
//! Exercises the `zero_order(dur)` input-rate forcing through the *public* API —
//! parse → `predict()` → ODE integration → readout — the zero-order counterpart
//! of `tests/weibull_absorption.rs` / `tests/igd_absorption.rs`.
//!
//! Two anchors:
//!  - the model-independent **mass-balance invariant** `∫A dt = F·Dose/ke` (a
//!    missing forcing → AUC 0, a double-counted bolus → 2×, a mis-normalised rate
//!    → wrong total); and
//!  - an **explicit-infusion equivalence** anchor: `zero_order(dur)` is, by
//!    definition, a constant `Dose/dur` over `(0, dur]` — i.e. a zero-order
//!    infusion — so a `zero_order(dur=DUR)` model fed by a bolus must predict
//!    *identically* to the same disposition fed by an explicit infusion of
//!    duration `DUR`. This is the issue's "analytical infusion anchor" and needs
//!    no licensed NONMEM run (the value path reuses #324's modeled-duration `Dn`
//!    forcing).

mod common;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, DoseEvent, Population};

/// One-compartment model with built-in zero-order absorption straight into
/// central (no first-order `ka`). central (CMT 1) holds the drug AMOUNT (mg) and
/// receives `R_in(tad) = Dose/DUR` over `(0, DUR]`. η fixed at 0 so `predict()`
/// returns the typical-value curve. CL = 5, V = 50 ⇒ ke = 0.1 ⇒ amount
/// AUC∞ = Dose/ke = 100/0.1 = 1000 mg·h. F defaults to 1.0. DUR = 4 h.
const ZERO_ORDER_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVDUR(4.0, 0.05,  24.0)

  omega ETA_CL ~ 0.0 FIX

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#;

/// Same one-compartment disposition, but **no** absorption term — central is fed
/// by an explicit infusion supplied in the data (rate column). Used as the
/// equivalence reference for `ZERO_ORDER_MODEL`.
const PLAIN_1CPT_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V*central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#;

/// Single dose of 100 mg into CMT 1 over `obs_times`. `rate = 0` ⇒ bolus (fed to
/// `R_in` for the zero-order model); `rate > 0` ⇒ explicit infusion.
fn pop_single(obs_times: Vec<f64>, rate: f64) -> Population {
    let n = obs_times.len();
    let dose = DoseEvent::new(0.0, 100.0, 1, rate, false, 0.0);
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![dose],
            obs_times,
            vec![0.0; n],
            vec![1; n],
        )],
    }
}

#[test]
fn zero_order_curve_recovers_dose_auc_and_has_flat_input_onset() {
    let model = parse_full_model(ZERO_ORDER_MODEL)
        .expect("zero_order model parses")
        .model;

    // 0, 0.25, …, 72.0 — fine enough for trapezoidal AUC, long enough that the
    // truncated tail is negligible (ke = CL/V = 0.1 ⇒ t½ ≈ 6.9 h).
    let obs_times: Vec<f64> = (0..=288).map(|i| i as f64 * 0.25).collect();
    let pop = pop_single(obs_times, 0.0);
    let preds = predict(&model, &pop, &model.default_params);

    // (1) No instantaneous bolus jump: the dose enters as a constant rate over
    //     (0, DUR], so central starts at 0.
    assert_eq!(preds[0].time, 0.0);
    assert!(
        preds[0].pred.abs() < 1e-12,
        "zero_order dose leaked in as a bolus: central(0) = {}",
        preds[0].pred
    );

    // (2) The peak is at the end of the input window (DUR = 4 h): while R_in is on,
    //     input (25 mg/h) exceeds elimination, so the amount rises to a maximum at
    //     t = DUR, then falls. The max sample is the one nearest t = 4.
    let max_idx = (0..preds.len())
        .max_by(|&a, &b| preds[a].pred.partial_cmp(&preds[b].pred).unwrap())
        .unwrap();
    assert!(
        (preds[max_idx].time - 4.0).abs() <= 0.25,
        "expected Tmax at the window end (DUR = 4 h), got t = {}",
        preds[max_idx].time
    );

    // (3) Mass balance via the absorption-independent invariant ∫A dt = Dose/ke.
    let auc: f64 = preds
        .windows(2)
        .map(|w| 0.5 * (w[0].pred + w[1].pred) * (w[1].time - w[0].time))
        .sum();
    let auc_inf = 100.0 * 50.0 / 5.0; // F·Dose/ke = Dose·V/CL with F = 1
    let rel = (auc - auc_inf).abs() / auc_inf;
    assert!(
        rel < 0.02,
        "zero_order amount AUC {auc:.4} vs Dose·V/CL {auc_inf:.4} (rel err {rel:.2e})"
    );
}

#[test]
fn zero_order_equals_explicit_infusion_of_the_same_duration() {
    // The defining identity: `zero_order(dur=DUR)` fed by a 100 mg bolus is a
    // zero-order infusion of 100 mg over DUR = 4 h, i.e. rate = 25 mg/h. The same
    // disposition fed by that explicit infusion must predict identically — across
    // the whole curve, including the in-window ramp, the window-end kink, and the
    // post-window decay. This pins the value path (not just the total mass) and is
    // the in-engine analogue of a NONMEM zero-order-infusion anchor.
    let zo = parse_full_model(ZERO_ORDER_MODEL)
        .expect("zero_order model parses")
        .model;
    let plain = parse_full_model(PLAIN_1CPT_MODEL)
        .expect("plain 1-cpt model parses")
        .model;

    let obs_times: Vec<f64> = (0..=120).map(|i| i as f64 * 0.25).collect();
    let zo_preds = predict(&zo, &pop_single(obs_times.clone(), 0.0), &zo.default_params);
    // rate = Dose/DUR = 100/4 = 25 mg/h ⇒ infusion duration = amt/rate = 4 h.
    let inf_preds = predict(&plain, &pop_single(obs_times, 25.0), &plain.default_params);

    assert_eq!(zo_preds.len(), inf_preds.len());
    for (a, b) in zo_preds.iter().zip(inf_preds.iter()) {
        assert!(
            (a.pred - b.pred).abs() <= 1e-6 * (1.0 + b.pred.abs()),
            "zero_order vs explicit infusion diverge at t = {}: {} vs {}",
            a.time,
            a.pred,
            b.pred
        );
    }
}

/// Concentration readout (`S1 = V`, Form C `y = central/V`) over a zero-order
/// input of **duration 5 h** — the parameters of the NONMEM modeled-duration
/// anchor (CL = 5, V = 50, AMT = 100, D1 = 5 ⇒ rate 20 over 5 h).
const ZERO_ORDER_CONC_D5_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVDUR(5.0, 0.05,  24.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#;

#[test]
fn zero_order_matches_nonmem_modeled_duration_pred() {
    // DIRECT NONMEM anchor (not the in-engine infusion transitivity): a
    // `zero_order(dur)` input is, by definition, NONMEM's modeled-duration infusion
    // (`RATE=-2` → `D1`, rate `AMT/D1`). NONMEM's exact ADVAN1 PRED for the scenario
    // CL=5, V=50 (ke=0.1), AMT=100, D1=5 is published in
    // `tests/nonmem/modeled_duration.ctl` / `data/modeled_duration_ref.csv` (the
    // same anchor `tests/modeled_duration.rs` uses for the explicit-infusion path).
    // ferx's `zero_order(dur=5)` ODE forcing must reproduce those PRED values — so
    // the absorption term is tied to a real NONMEM run, not only to an in-engine
    // sibling. Tolerance 1e-3: NONMEM's PRED is printed to 3 decimals, so the
    // residual is dominated by that output rounding (observed max ≈ 4e-4), well
    // below any real model discrepancy (a wrong window/mass diverges by ≫1e-3).
    let nonmem_pred: &[(f64, f64)] = &[
        (0.5, 0.195),
        (1.0, 0.381),
        (2.0, 0.725),
        (5.0, 1.574), // end of the 5 h input window — the peak
        (8.0, 1.166),
        (12.0, 0.782),
        (18.0, 0.429),
        (24.0, 0.235),
    ];

    let model = parse_full_model(ZERO_ORDER_CONC_D5_MODEL)
        .expect("zero_order concentration model parses")
        .model;
    let times: Vec<f64> = nonmem_pred.iter().map(|&(t, _)| t).collect();
    let preds = predict(&model, &pop_single(times, 0.0), &model.default_params);

    assert_eq!(preds.len(), nonmem_pred.len());
    for (p, &(t, nm)) in preds.iter().zip(nonmem_pred.iter()) {
        assert!(
            (p.pred - nm).abs() < 1e-3,
            "zero_order(dur=5) vs NONMEM ADVAN1 PRED diverge at t = {t}: \
             ferx {} vs NONMEM {nm}",
            p.pred
        );
    }
}

#[test]
fn zero_order_restarts_after_reset_event_driven_path() {
    // A system reset (EVID=3/4) routes the subject through the event-driven walker
    // (not the dense `ode_predictions` loop), exercising the zero-order delivery on
    // that path — its per-dose-snapshot windows and reset-floor handling. Dose at
    // t=0 (window [0,4]); reset at t=12 zeros the state; re-dose at t=12 (window
    // [12,16]). From a freshly-zeroed system the second cycle reproduces the first,
    // so the concentration at matched offsets after each dose must coincide.
    let model = parse_full_model(ZERO_ORDER_MODEL)
        .expect("zero_order model parses")
        .model;

    let obs_times = vec![2.0, 4.0, 8.0, 14.0, 16.0, 20.0]; // offsets 2/4/8 in each cycle
    let n = obs_times.len();
    let mut subject = common::subject(
        "1",
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times,
        vec![0.0; n],
        vec![1; n],
    );
    subject.reset_times = vec![12.0]; // EVID=3 reset at t=12 → event-driven path
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subject],
    };
    let preds = predict(&model, &pop, &model.default_params);

    // pred(12 + x) == pred(x) for x ∈ {2, 4, 8}: the reset + re-dose reproduce the
    // first cycle's zero-order absorption exactly.
    for (first, second) in [(0, 3), (1, 4), (2, 5)] {
        assert!(
            (preds[first].pred - preds[second].pred).abs()
                <= 1e-6 * (1.0 + preds[first].pred.abs()),
            "post-reset cycle should mirror the first at offset {}: {} vs {}",
            preds[first].time,
            preds[first].pred,
            preds[second].pred
        );
    }
    // Sanity: the curve is non-trivial (not all zero).
    assert!(preds[1].pred > 1.0, "expected a real concentration at t=4");
}

#[test]
fn zero_order_window_open_at_reset_stops_like_a_cut_infusion() {
    // Adversarial reset case: the reset fires *while the window is still open*
    // (mid-input) — the branch the cycle-mirror test above does NOT reach (there
    // the window [0,4] has fully closed before the t=12 reset, so the `w_start >=
    // reset_floor` turn-off is never the deciding factor). Here a dose at t=0 opens
    // the window [0,4]; an EVID=3 reset at t=2 zeros the state and must also turn
    // the still-open window OFF (`w_start = 0 < reset_floor = 2`), so no further
    // mass is delivered after the reset.
    //
    // Oracle (not self-consistency): the same disposition fed by the *equivalent
    // explicit infusion* (25 mg/h over 4 h ≡ zero_order(dur=4) of a 100 mg dose),
    // cut by the identical reset. `active_infusions` and `active_zero_order_inputs`
    // share the byte-for-byte `w_start >= reset_floor` turn-off, so the two curves
    // must coincide across the whole timeline — the pre-reset ramp and the
    // post-reset decay from the zeroed state. A regression that kept the
    // zero-order window alive past the reset would inject extra mass and diverge
    // here, while the cycle-mirror test above would still pass.
    let zo = parse_full_model(ZERO_ORDER_MODEL)
        .expect("zero_order model parses")
        .model;
    let plain = parse_full_model(PLAIN_1CPT_MODEL)
        .expect("plain 1-cpt model parses")
        .model;

    let obs_times: Vec<f64> = (0..=80).map(|i| i as f64 * 0.25).collect(); // 0..20h
    let n = obs_times.len();
    let reset_at = 2.0; // strictly inside the window [0, 4]

    let mk = |rate: f64| {
        let mut s = common::subject(
            "1",
            vec![DoseEvent::new(0.0, 100.0, 1, rate, false, 0.0)],
            obs_times.clone(),
            vec![0.0; n],
            vec![1; n],
        );
        s.reset_times = vec![reset_at]; // EVID=3 mid-window → event-driven path
        Population {
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
            subjects: vec![s],
        }
    };

    let zo_preds = predict(&zo, &mk(0.0), &zo.default_params);
    let inf_preds = predict(&plain, &mk(25.0), &plain.default_params);

    assert_eq!(zo_preds.len(), inf_preds.len());
    for (a, b) in zo_preds.iter().zip(inf_preds.iter()) {
        assert!(
            (a.pred - b.pred).abs() <= 1e-6 * (1.0 + b.pred.abs()),
            "zero_order vs reset-cut infusion diverge at t = {} (reset at {}): {} vs {}",
            a.time,
            reset_at,
            a.pred,
            b.pred
        );
    }

    // Independent absolute check (in case *both* models shared a bug): the reset
    // must bite. A real pre-reset fill accrues (≈ 25·2 = 50 mg, minus elimination),
    // and after the reset the window is OFF, so the amount only decays from the
    // zeroed state — it never climbs back.
    let pre_reset = zo_preds
        .iter()
        .filter(|p| p.time < reset_at - 1e-9)
        .map(|p| p.pred)
        .fold(0.0_f64, f64::max);
    let post_reset = zo_preds
        .iter()
        .filter(|p| p.time > reset_at + 1e-9)
        .map(|p| p.pred)
        .fold(0.0_f64, f64::max);
    assert!(
        pre_reset > 10.0,
        "expected a real pre-reset partial fill, got {pre_reset}"
    );
    assert!(
        post_reset < 1e-6,
        "the window must be OFF after the reset (no further input); got post-reset max {post_reset}"
    );
}

/// `ZERO_ORDER_MODEL` with an absorption **lagtime** on the dosing compartment
/// (`LAGTIME1` ≡ NONMEM `ALAG1` for CMT 1): the input window shifts to
/// `(lag, lag+dur]`. Same CL/V/DUR as `ZERO_ORDER_MODEL`; lag = 2 h.
const ZERO_ORDER_LAGGED_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVDUR(4.0, 0.05,  24.0)
  theta TVLAG(2.0, 0.001, 12.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL       = TVCL * exp(ETA_CL)
  V        = TVV
  DUR      = TVDUR
  LAGTIME1 = TVLAG

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = zero_order(dur=DUR) - CL/V*central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#;

#[test]
fn zero_order_with_lagtime_is_the_unlagged_curve_shifted_by_the_lag() {
    // A dose lagtime shifts the input window to (lag, lag+dur], so the window-start
    // break moves to `dose.time + lag`. The full-containment filter only delivers
    // the correct mass if a segment boundary lands there — every prior zero-order
    // test runs lag = 0, where `w_start` coincides with the always-present dose-time
    // break and so never exercises the lagged-start boundary.
    //
    // Oracle (not self-consistency): a lagtime is a pure time-shift of the whole
    // input, so the lagged curve at `t` must equal the *unlagged* curve at `t - lag`
    // for t >= lag (before which the lagged system is empty). Compare the two
    // predict() curves on grids offset by exactly the lag.
    let lagged = parse_full_model(ZERO_ORDER_LAGGED_MODEL)
        .expect("lagged zero_order model parses")
        .model;
    let unlagged = parse_full_model(ZERO_ORDER_MODEL)
        .expect("zero_order model parses")
        .model;
    let lag = 2.0;

    let base: Vec<f64> = (0..=112).map(|i| i as f64 * 0.25).collect(); // 0..28h
    let lagged_times: Vec<f64> = base.iter().map(|&t| t + lag).collect();

    let unlagged_preds = predict(
        &unlagged,
        &pop_single(base.clone(), 0.0),
        &unlagged.default_params,
    );
    let lagged_preds = predict(
        &lagged,
        &pop_single(lagged_times, 0.0),
        &lagged.default_params,
    );

    // (1) Nothing absorbed before the lag: at the first lagged sample (t = lag) the
    //     amount is still 0 — the window has not opened yet.
    assert!(
        lagged_preds[0].pred.abs() < 1e-9,
        "lagged input must not start before t = lag; got {} at t = {}",
        lagged_preds[0].pred,
        lagged_preds[0].time
    );

    // (2) Time-shift identity: lagged(t + lag) == unlagged(t) across the whole
    //     curve, including the in-window ramp and the window-end kink at t = DUR
    //     (lagged: t = lag + DUR). This pins the lagged window-start break.
    assert_eq!(unlagged_preds.len(), lagged_preds.len());
    for (u, l) in unlagged_preds.iter().zip(lagged_preds.iter()) {
        assert!(
            (u.pred - l.pred).abs() <= 1e-6 * (1.0 + u.pred.abs()),
            "lagged curve should be the unlagged curve shifted by {lag}h: \
             unlagged(t={}) = {} vs lagged(t={}) = {}",
            u.time,
            u.pred,
            l.time,
            l.pred
        );
    }

    // (3) The shift is real, not a degenerate all-zero match: the peak (at the
    //     window end) is a genuine concentration.
    let peak = lagged_preds.iter().map(|p| p.pred).fold(0.0_f64, f64::max);
    assert!(peak > 1.0, "expected a real lagged peak, got {peak}");
}

/// `sequential` absorption: `zero_order(dur)` fills a depot, emptied to central by
/// a hand-written first-order `- KA*depot` (no new intrinsic). central (CMT 2)
/// holds the amount; the depot is CMT 1 and receives the zero-order input.
const SEQUENTIAL_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVKA(1.0,  0.05,  24.0)
  theta TVDUR(3.0, 0.05,  24.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  DUR = TVDUR

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = zero_order(dur=DUR) - KA*depot
  d/dt(central) = KA*depot - CL/V*central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#;

#[test]
fn sequential_model_parses_and_delivers_full_mass_to_central() {
    let model = parse_full_model(SEQUENTIAL_MODEL)
        .expect("sequential model parses")
        .model;

    // Dose into the depot (CMT 1); observe central (CMT 2).
    let obs_times: Vec<f64> = (0..=288).map(|i| i as f64 * 0.25).collect();
    let n = obs_times.len();
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![dose],
            obs_times,
            vec![0.0; n],
            vec![2; n], // observe central
        )],
    };
    let preds = predict(&model, &pop, &model.default_params);

    // central starts empty (the zero-order input fills the depot first, then ka
    // transfers) and the full dose still reaches central: ∫central dt = Dose/ke.
    assert!(preds[0].pred.abs() < 1e-12, "central(0) should be 0");
    let auc: f64 = preds
        .windows(2)
        .map(|w| 0.5 * (w[0].pred + w[1].pred) * (w[1].time - w[0].time))
        .sum();
    let auc_inf = 100.0 * 50.0 / 5.0;
    let rel = (auc - auc_inf).abs() / auc_inf;
    assert!(
        rel < 0.02,
        "sequential central AUC {auc:.4} vs Dose·V/CL {auc_inf:.4} (rel err {rel:.2e})"
    );
}

#[test]
fn zero_order_on_analytical_pk_is_rejected_pointing_at_ode_template() {
    // The error rule end-to-end: a `zero_order()` input rate on an analytical `pk`
    // disposition is a hard error pointing at `ode_template`, never a silent
    // analytical→ODE swap. (`zero_order` ships ODE-only in #504; it leaves the
    // ODE-only list when its closed-form acceleration lands.)
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVDUR(3.0, 0.05, 24.0)
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL
  V   = TVV
  KA  = TVKA
  DUR = TVDUR
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[odes]
  d/dt(depot) = zero_order(dur=DUR) - KA*depot
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let err = match parse_full_model(src) {
        Ok(_) => panic!("pk + zero_order() must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.contains("ode_template"),
        "error should point at ode_template, got: {err}"
    );
    assert!(
        err.contains("zero_order"),
        "error should name the zero_order function, got: {err}"
    );
}

#[test]
fn zero_order_normal_dosing_passes_data_checks() {
    // Positive control: ordinary (non-SS, bolus) dosing into the zero_order
    // compartment raises no absorption diagnostic.
    use ferx_core::check_model_data;
    let model = parse_full_model(ZERO_ORDER_MODEL)
        .expect("zero_order model parses")
        .model;
    let pop = pop_single(vec![0.5, 1.0, 2.0, 4.0, 8.0], 0.0);
    let diags = check_model_data(&model, &pop);
    assert!(
        !diags.iter().any(|d| d.code.starts_with("E_ABSORPTION")),
        "unexpected absorption diagnostic: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

#[test]
fn zero_order_steady_state_dosing_is_rejected() {
    // SS=1 into a zero_order compartment is rejected with E_ABSORPTION_SS_ZERO_ORDER (#719):
    // the density kernels (transit/igd/weibull/first_order) now support SS through the
    // periodic-sum R_in, but a zero_order input is a spanning window whose periodic steady
    // state needs a distinct window-based equilibration (a follow-up), so it stays rejected.
    use ferx_core::check_model_data;
    let model = parse_full_model(ZERO_ORDER_MODEL)
        .expect("zero_order model parses")
        .model;
    let mut ss_dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
    ss_dose.ii = 12.0;
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![ss_dose],
            vec![1.0, 4.0, 8.0],
            vec![0.0; 3],
            vec![1; 3],
        )],
    };
    let diags = check_model_data(&model, &pop);
    assert!(
        diags.iter().any(|d| d.code == "E_ABSORPTION_SS_ZERO_ORDER"),
        "SS into a zero_order compartment must raise E_ABSORPTION_SS_ZERO_ORDER, got: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

#[test]
fn infusion_into_zero_order_compartment_is_rejected() {
    // An infusion (RATE>0) into a zero_order compartment is rejected with
    // E_ABSORPTION_RATE_ZERO_ORDER (#719 gap 2): the density kernels now serve an infusion via
    // the convolution `R_in_inf`, but a zero_order input is itself a spanning window, so infusing
    // it is a window-with-window convolution (a follow-up).
    use ferx_core::check_model_data;
    let model = parse_full_model(ZERO_ORDER_MODEL)
        .expect("zero_order model parses")
        .model;
    // RATE=25 → an infusion window of AMT/RATE = 4 h into the zero_order compartment (CMT 1).
    let inf_dose = DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0);
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![inf_dose],
            vec![1.0, 4.0, 8.0],
            vec![0.0; 3],
            vec![1; 3],
        )],
    };
    let diags = check_model_data(&model, &pop);
    assert!(
        diags
            .iter()
            .any(|d| d.code == "E_ABSORPTION_RATE_ZERO_ORDER"),
        "infusion into a zero_order compartment must raise E_ABSORPTION_RATE_ZERO_ORDER, got: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

/// Tier-3 (slow): full FOCEI fit recovering the zero-order **duration**. Gated out
/// of the default PR job; runs nightly / on estimation-touching pushes.
///
/// Generates a noise-free single-dose curve at the truth (`DUR = 4 h`), then fits
/// it back from a **perturbed** start (`DUR = 2 h`) — so the optimiser must climb
/// the duration, not merely sit at the truth — and asserts it recovers `DUR ≈ 4`.
/// This exercises the FD sensitivity of the moving-boundary `dur` end-to-end (the
/// gradient path #504 ships, ahead of the analytic boundary impulse in #530).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn zero_order_fit_recovers_duration() {
    use ferx_core::{fit, FitOptions};

    let model = parse_full_model(ZERO_ORDER_MODEL)
        .expect("zero_order model parses")
        .model;

    // Noise-free data at the truth: predictions at the typical values become the
    // observations, so the likelihood optimum sits at the data-generating params.
    let obs_times: Vec<f64> = (1..=40).map(|i| i as f64 * 0.5).collect();
    let truth = predict(
        &model,
        &pop_single(obs_times.clone(), 0.0),
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

    // Perturb the initial duration (theta[2]) to 2 h so the fit has to move it.
    let mut init = model.default_params.clone();
    init.theta[2] = 2.0;
    let fitted = fit(&model, &pop, &init, &FitOptions::default()).expect("fit converges");

    assert!(fitted.converged, "zero_order fit should converge");
    let dur = fitted.theta[2];
    assert!(
        (dur - 4.0).abs() / 4.0 < 0.10,
        "recovered DUR {dur:.4} should be within 10% of the truth (4.0)"
    );
}
