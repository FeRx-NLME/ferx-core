use super::*;
use crate::parser::model_parser::{parse_full_model, parse_model_string};
use crate::sim::adaptive::{
    AdaptiveAction, AdaptiveDosingSpec, AdaptiveRoute, AdaptiveRule, Comparison, ControllerCtx,
    DoseAction, DoseStep, MonitorSpec, ObserveMode,
};
use std::collections::HashMap;

// 1-cpt IV ODE, state = central amount, readout y = central (amount units).
// CL=5, V=50 → k = 0.1/h. ETA_CL is declared (the parser requires an omega)
// but **not referenced** by CL, so predictions are η-invariant and the whole
// simulation is effectively deterministic — the bit-exact oracle below relies
// on this (a drawn-but-unused η leaves the trajectory at the η=0 value).
const ODE_NO_IIV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

// Same structural model, with between-subject variability on CL.
const ODE_IIV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

// IOV ODE model (#701): a per-occasion κ on CL over a near-degenerate BSV η
// (the parser requires an omega). A seeded run's η and per-occasion κ are both
// reconstructed exactly and checked against `predict_iov`.
const ODE_IOV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  kappa KAPPA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP)
"#;

// Analytical (non-ODE) twin — used to assert simulate_adaptive rejects it.
const ANALYTICAL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
"#;

// Time-varying-covariate ODE twin (model only, no [adaptive_dosing] block) — CL reads
// CRCL, so a subject with per-observation covariates is a TV-cov subject. Used to assert
// a base regimen × TV covariate is rejected (#702 scope).
const ODE_TV_COV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * CRCL / 100.0
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

// Time-varying covariate reaching the prediction ONLY through an `init(...)` seed
// (#1133). `CL`/`V` are plain thetas, so `CRCL` moves nothing but the baseline the
// system starts (and restarts) from — which makes the snapshot an EVID=3 reset re-seeds
// with directly observable in the trajectory instead of tangled with disposition.
const ODE_TV_INIT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL   = TVCL
  V    = TVV
  BASE = CRCL * 10.0
[structural_model]
  ode(states=[central])
[odes]
  init(central)  = BASE
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

// 1-cpt IV ODE whose RHS reads TAFD (time after first dose). The extra
// `-1e-3·TAFD·central` decay makes the trajectory depend on the TAFD anchor, so a stale
// anchor (e.g. earliest *base* dose instead of the true global earliest) integrates
// different forcing and the frozen-replay verifier catches it. Used to pin #934: a
// controller dose scheduled before the earliest base dose must lower the anchor.
const ODE_TAFD: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central - 1e-3 * TAFD * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

// Time-varying covariate on BIOAVAILABILITY (not CL): F reads CRCL, so a base dose given
// where CRCL != the t=0 baseline gets a different F — the case #930's per-dose F resolution
// must handle. CL/V are constant, so the *only* per-event effect is on each dose's F. The
// declared `omega` is unused (CL = TVCL), so a drawn η never perturbs the closed form.
const ODE_TV_F: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVF(0.8, 0.01, 1.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
  F  = TVF * CRCL / 100.0
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

// Inter-occasion variability on BIOAVAILABILITY (not CL): F carries a per-occasion κ, so a
// base dose given in a decision window enters with THAT window's F — the case #931's per-dose
// occasion-F resolution must handle. CL/V are effectively constant (`ETA_CL ~ N(0, 1e-10)`),
// so the only per-occasion effect is on each dose's F, giving a clean closed form. `TVF = 0.5`
// keeps F = 0.5·exp(κ_F) < 1 (physical) across the drawn κ.
const ODE_IOV_F: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVF(0.5, 0.01, 1.0)
  omega ETA_CL ~ 1e-10
  kappa KAPPA_F ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = TVF * exp(KAPPA_F)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

// Time-varying covariate on CL plus a fixed dose lag time (`LAGTIME`, the reserved ODE
// name → PK_IDX_LAGTIME). A *lagged* base dose under a time-varying covariate is a #930
// typed error (the hand-rolled TV frozen-replay engine carries no base lag yet).
const ODE_TV_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL      = TVCL * CRCL / 100.0
  V       = TVV
  LAGTIME = 2.0
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

// Time-varying covariate on CL plus a built-in first-order absorption input rate feeding
// `central` (the dosed compartment). A base dose into an input-rate-fed compartment under a
// time-varying covariate is a #930 typed error — the depot / absorption bookkeeping is not
// threaded through the hand-rolled TV frozen-replay engine yet. A single pathway (implicit
// fraction 1) clears the parallel-absorption fraction-sum check, so the ONLY rejection is the
// #930 input-rate arm of the base-dose guard.
const ODE_TV_ABSORB: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * CRCL / 100.0
  V  = TVV
  KA = TVKA
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - (CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

// Time-varying covariate on CL plus a declared modeled-rate parameter `R1`, so a base dose
// carrying a coded RATE=-1 (modeled infusion rate) is a VALID dose that actually reaches the
// #930 base-dose guard — rather than being turned away earlier for a missing `R1`. Such a base
// dose under a time-varying covariate is a #930 typed error: its rate-resolution bookkeeping
// (RATE=-1/-2 → R1/D1 from the PK snapshot) is not threaded through the TV frozen-replay engine
// yet. (This is also the one input that drives `resolve_subject_doses` down its owned/mutating
// branch; the guard reads `is_fixed()` on the pre-resolution dose, so it still fires.)
const ODE_TV_MRATE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * CRCL / 100.0
  V  = TVV
  R1 = 100.0
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

fn subj(id: &str, obs_times: Vec<f64>, doses: Vec<DoseEvent>) -> Subject {
    let n = obs_times.len();
    let n_dose = doses.len();
    Subject {
        id: id.to_string(),
        doses,
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1u32; n],
        obs_l2: Vec::new(),
        dose_occasions: vec![1u32; n_dose],
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

fn population(subjects: Vec<Subject>) -> Population {
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// A fixed 100 mg bolus into cmt 1 at every decision — the degenerate
/// (state-independent) controller.
fn fixed_bolus() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
}

/// A controller that never doses — `Hold` at every decision. Proves a pre-scheduled
/// base regimen (#702) is integrated on its own, with the controller adding nothing.
fn hold_all() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |_ctx: &ControllerCtx| vec![DoseAction::Hold]
}

#[test]
fn degenerate_oracle_matches_static_predict() {
    // A controller that doses at every decision must reproduce the static
    // engine on the same realized schedule. Last obs (54) is the global max
    // and a dose lands at every decision, so the segment structures align
    // (the verifier, on by default, also passes).
    //
    // The assertion below is a **1e-9 relative band, not bit equality** — the name said
    // `_bit_for_bit` until #1151 measured what it actually pins. On a model-time-reading RHS
    // the same comparison at default solver tolerances is 1.3e-6 apart from solver noise
    // alone, so the band is what this fixture can honestly claim. The bit-equality property
    // does hold for a read taken ON a boundary with no integration in between, and is
    // asserted there instead:
    // `degenerate_oracle_tad_rhs_observation_on_a_decision_boundary_is_bit_identical`.
    let model = parse_model_string(ODE_NO_IIV).expect("parse no-IIV ODE model");
    let decisions = vec![0.0, 24.0, 48.0];
    let obs = vec![6.0, 30.0, 54.0];

    let pop = population(vec![subj("1", obs.clone(), vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: decisions.clone(),
        ..Default::default()
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("adaptive sim runs");

    // Static reference: same doses pre-scheduled, run through predict() (η=0,
    // IPRED) — exactly what the realized ledger encodes.
    let static_doses: Vec<DoseEvent> = decisions
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let static_pop = population(vec![subj("1", obs.clone(), static_doses)]);
    let preds = predict(&model, &static_pop, &model.default_params).unwrap();

    assert_eq!(res.trajectories.len(), obs.len());
    assert_eq!(res.ledger.len(), 3, "a dose at every decision");
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-9 + 1e-9 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={}",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
}

/// A one-shot infusion at the second decision (t=24), holding otherwise. Its
/// window spans the mid-horizon reset so the reset-floor infusion turn-off is
/// exercised (#716).
fn infuse_at_second_decision() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |ctx: &ControllerCtx| {
        if ctx.decision_index == 1 {
            vec![DoseAction::Infuse {
                amt: 120.0,
                cmt: 1,
                rate: 5.0,
            }]
        } else {
            vec![DoseAction::Hold]
        }
    }
}

#[test]
fn adaptive_reset_matches_static_predict() {
    // Degenerate oracle for system resets (#716): a fixed-dose controller over a
    // dose-free base subject carrying a mid-horizon EVID=3 reset must reproduce the
    // trusted static engine — `predict()`, which routes reset subjects to the
    // reset-aware event-driven walker — on the same realized regimen. The model is
    // η-invariant, so the adaptive IPRED equals the η=0 static PRED.
    //
    // The default-on frozen-replay verifier (now reset-aware) also runs, and its Ok
    // is part of this assertion: a reset-blind verifier would compute the post-reset
    // observation at t=42 as ~0 in the driver but ~18 in the replay and error out, so
    // the `.expect` below fails unless BOTH the driver and the verifier honor the reset.
    //
    // Tolerance: `predict()` routes a reset subject to the event-driven engine
    // (`solve_ode`), while the reactive driver integrates with `integrate_segment`
    // (`solve_ode_dense`) — two independent integrators, so they agree to solver noise
    // (~1e-6 relative), not to the bit. That independence is the value here: a reset
    // dropped or mis-applied by the driver would move a prediction by O(dose) — tens of
    // percent — which this rel-1e-4 bound catches easily, while the tight same-engine
    // bookkeeping check is the auto-run frozen-replay verifier's job.
    let model = parse_model_string(ODE_NO_IIV).expect("parse no-IIV ODE model");
    let decisions = vec![0.0, 24.0, 48.0];
    let obs = vec![6.0, 30.0, 42.0, 54.0];
    let reset_at = 36.0;

    let mut base = subj("1", obs.clone(), vec![]);
    base.reset_times = vec![reset_at];
    let pop = population(vec![base]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: decisions.clone(),
        ..Default::default() // verify = true
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("adaptive reset sim runs and passes the reset-aware verifier");
    assert_eq!(res.ledger.len(), 3, "a bolus at every decision");

    // Static reference: the realized doses pre-scheduled on a subject carrying the
    // same reset, scored by predict() (η=0, event-driven, reset honored).
    let static_doses: Vec<DoseEvent> = decisions
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let mut static_subject = subj("1", obs.clone(), static_doses);
    static_subject.reset_times = vec![reset_at];
    let static_pop = population(vec![static_subject]);
    let preds = predict(&model, &static_pop, &model.default_params).unwrap();

    assert_eq!(res.trajectories.len(), obs.len());
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-6 + 1e-4 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={} (reset at {reset_at})",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
}

/// #1133 — the adaptive driver's reset re-seed must read the reset ROW's own covariate
/// snapshot, not the decision-time LOCF carry.
///
/// `adaptive_reset_matches_static_predict` above cannot see this: `ODE_NO_IIV` has no
/// `init(...)` and no covariate, so every candidate snapshot seeds the same zeros. Here
/// `init(central) = CRCL*10` and `CRCL` steps 100 → 40 exactly at the reset row, between
/// records carrying 100 and 25 — so the previous record, the reset row, and the next
/// record give three different baselines (1000 / 400 / 250).
///
/// Two oracles run at once, and they are independent:
///
///   * the **degenerate oracle** — the same realized regimen scored by `predict()`, which
///     routes a reset subject to `ode_predictions_event_driven` (the engine anchored
///     against NONMEM in `tests/reset_init_snapshot_nonmem_anchor.rs`);
///   * the **frozen-replay verifier**, on by default, which rebuilds the subject from the
///     realized dose ledger and walks it through `adaptive_frozen_replay_tv`. Its `Ok` is
///     part of this assertion, so the driver and the replay must agree with *each other*
///     as well as with the static engine — three engines, one convention.
#[test]
fn adaptive_reset_reseeds_init_from_the_reset_rows_covariates() {
    let model = parse_model_string(ODE_TV_INIT).expect("parse TV-init ODE model");
    let decisions = vec![0.0, 24.0, 48.0];
    let obs = vec![6.0, 30.0, 42.0, 54.0];
    let reset_at = 36.0;
    let crcl = |v: f64| HashMap::from([("CRCL".to_string(), v)]);

    // Records: obs at 6 and 30 carry CRCL=100; the reset row at 36 carries 40; the obs at
    // 42 and 54 carry 25. Only the middle one may reach the re-seed.
    let with_cov = |mut s: Subject| -> Subject {
        s.covariates = crcl(100.0);
        s.obs_covariates = vec![crcl(100.0), crcl(100.0), crcl(25.0), crcl(25.0)];
        s.dose_covariates = s.doses.iter().map(|_| crcl(100.0)).collect();
        s.reset_times = vec![reset_at];
        s.reset_covariates = vec![crcl(40.0)];
        s
    };

    let base = with_cov(subj("1", obs.clone(), vec![]));
    assert!(
        base.has_tv_covariates(),
        "the subject must take the TV path"
    );
    let pop = population(vec![base]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: decisions.clone(),
        ..Default::default() // verify = true
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("adaptive sim runs and passes the frozen-replay verifier");
    assert_eq!(res.ledger.len(), 3, "a bolus at every decision");

    let static_doses: Vec<DoseEvent> = decisions
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let static_pop = population(vec![with_cov(subj("1", obs.clone(), static_doses))]);
    let preds = predict(&model, &static_pop, &model.default_params).unwrap();

    assert_eq!(res.trajectories.len(), obs.len());
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-6 + 1e-4 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={}: the reset re-seed read a \
             different covariate snapshot than the static engine (#1133)",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }

    // Non-degeneracy: the post-reset observation must actually carry the reset row's
    // baseline. At t=42, six hours after the reset, the seed 400 has decayed by
    // exp(-0.1*6) = 0.5488 to ~219.5 — while the previous record's snapshot would give
    // ~548.8 and the next record's ~137.2. Assert the band that admits only 400.
    let t42 = res
        .trajectories
        .iter()
        .find(|t| (t.time - 42.0).abs() < 1e-9)
        .expect("an observation at t=42");
    let expected = 400.0 * (-(5.0 / 50.0) * 6.0f64).exp();
    assert!(
        (t42.ipred - expected).abs() < 1e-3,
        "post-reset IPRED {} is not the reset row's baseline decayed ({expected:.4}); \
         the previous record's snapshot would give {:.4} and the next record's {:.4}",
        t42.ipred,
        expected * 2.5,
        expected * 0.625
    );
}

#[test]
fn adaptive_reset_zeros_state_positive_control() {
    // Proof the reset actually zeros the compartments — so the degenerate oracle above
    // is not vacuous (both engines agreeing on an *un*-reset trajectory). The same
    // fixed-dose controller run with vs without a mid-horizon reset must diverge: with
    // the reset at 36 and no dose between 24 and 48, the post-reset observation at 42
    // reads exactly 0 (state zeroed, nothing re-entering); without it the 0 h and 24 h
    // boluses persist (~18 amount units). Removing the driver's reset-zeroing makes the
    // reset run read ~18 and trips the first assert.
    let model = parse_model_string(ODE_NO_IIV).expect("parse no-IIV ODE model");
    let decisions = vec![0.0, 24.0, 48.0];
    let obs = vec![42.0];
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: decisions.clone(),
        ..Default::default()
    };

    let mut with_reset = subj("1", obs.clone(), vec![]);
    with_reset.reset_times = vec![36.0];
    let res_reset = simulate_adaptive(
        &model,
        &population(vec![with_reset]),
        &model.default_params,
        1,
        fixed_bolus,
        &opts,
    )
    .expect("reset run");

    let no_reset = subj("1", obs.clone(), vec![]);
    let res_noreset = simulate_adaptive(
        &model,
        &population(vec![no_reset]),
        &model.default_params,
        1,
        fixed_bolus,
        &opts,
    )
    .expect("no-reset run");

    let y_reset = res_reset.trajectories[0].ipred;
    let y_noreset = res_noreset.trajectories[0].ipred;
    assert!(
        y_reset.abs() < 1e-9,
        "post-reset obs at t=42 must read ~0 (state zeroed at 36), got {y_reset}"
    );
    assert!(
        y_noreset > 10.0,
        "without the reset the 0 h + 24 h boluses persist at t=42 (~18), got {y_noreset}"
    );
}

#[test]
fn adaptive_reset_turns_off_spanning_infusion_matches_static_predict() {
    // Degenerate oracle exercising the reset FLOOR on a controller-issued infusion
    // (#716): an infusion issued at t=24 (window [24, 48]) spans the reset at t=36.
    // The reset must both zero the state AND turn the infusion off from 36 on, exactly
    // as the event-driven engine does (`active_infusions` honors `reset_floor`). So the
    // post-reset observation at t=42 reads 0, and the whole trajectory matches predict()
    // on the same pre-scheduled infusion + reset. If the reset floor were ignored, the
    // infusion would keep delivering past 36 and t=42 would be materially positive.
    let model = parse_model_string(ODE_NO_IIV).expect("parse no-IIV ODE model");
    let decisions = vec![0.0, 24.0, 48.0];
    let obs = vec![30.0, 42.0, 54.0];
    let reset_at = 36.0;

    let mut base = subj("1", obs.clone(), vec![]);
    base.reset_times = vec![reset_at];
    let pop = population(vec![base]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(11),
        decision_times: decisions.clone(),
        ..Default::default() // verify = true
    };
    let res = simulate_adaptive(
        &model,
        &pop,
        &model.default_params,
        1,
        infuse_at_second_decision,
        &opts,
    )
    .expect("adaptive reset+infusion sim runs and passes the reset-aware verifier");
    assert_eq!(res.ledger.len(), 1, "exactly one infusion, at t=24");
    assert!(res.ledger[0].rate > 0.0, "the realized dose is an infusion");

    // Static reference: the realized infusion pre-scheduled on a subject carrying the
    // same reset, scored by predict() (event-driven, reset floor turns the infusion off).
    let e = &res.ledger[0];
    let mut static_subject = subj(
        "1",
        obs.clone(),
        vec![DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0)],
    );
    static_subject.reset_times = vec![reset_at];
    let preds = predict(
        &model,
        &population(vec![static_subject]),
        &model.default_params,
    )
    .unwrap();

    assert_eq!(res.trajectories.len(), obs.len());
    // Locate the t=42 (post-reset) trajectory and assert it washed out.
    let y42 = res
        .trajectories
        .iter()
        .find(|t| (t.time - 42.0).abs() < 1e-12)
        .expect("t=42 trajectory row")
        .ipred;
    assert!(
        y42.abs() < 1e-9,
        "post-reset obs at t=42 must be ~0 (infusion turned off at 36), got {y42}"
    );
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        // Cross-engine tolerance (event-driven `predict()` vs dense reactive driver),
        // as in `adaptive_reset_matches_static_predict`.
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-6 + 1e-4 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={} (reset+infusion)",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
}

#[test]
fn adaptive_iov_matches_predict_iov_with_reconstructed_kappa() {
    // Full-stack IOV oracle (#701): run the reactive driver on a real IOV model
    // (κ drawn per decision window from the seeded substream), then reconstruct the
    // exact per-occasion κ and confirm the trajectory equals `predict_iov` on the
    // realized doses with occasions = decision windows. Obs coincide with decisions,
    // so every dosing occasion appears in predict_iov's obs-derived groups. The
    // default-on frozen-replay verifier also runs (bit-exact); its Ok is part of the
    // assertion. This is the api-level counterpart of the driver-level degenerate
    // oracle, exercising the real `pk_param_fn` → κ path end to end.
    let model = parse_model_string(ODE_IOV).expect("parse IOV ODE model");
    assert!(model.n_kappa == 1 && model.n_eta == 1);
    let decisions = vec![0.0, 24.0, 48.0, 72.0];
    let obs = decisions.clone();
    let seed = 20260708u64;
    let pop = population(vec![subj("1", obs.clone(), vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(seed),
        decision_times: decisions.clone(),
        ..Default::default() // verify = true
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("adaptive IOV sim runs and passes the default verifier");
    assert_eq!(res.ledger.len(), 4, "a dose at every decision");

    // Reconstruct the BSV η exactly as `run_adaptive_population` drew it: the same
    // seeded `StdRng`, one N(0,1) per η for this single (sim 1, subject "1"), then
    // Ω's Cholesky. (With `seed` set, the assay/κ streams don't touch this rng, so
    // the η draw is the only consumer — reproducible here.)
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let z_eta: Vec<f64> = (0..model.n_eta).map(|_| rng.sample(normal)).collect();
    let eta_bsv: Vec<f64> = (&model.default_params.omega.chol
        * nalgebra::DVector::from_column_slice(&z_eta))
    .iter()
    .copied()
    .collect();

    // Reconstruct the exact per-occasion κ from the seeded substream (the same
    // derivation `run_adaptive_population` uses: κ_g = chol(Ω_IOV) · z, z keyed by
    // (occasion, component); root = the run seed; replicate = 1).
    let omega_iov = model.default_params.omega_iov.as_ref().expect("omega_iov");
    let base = crate::sim::adaptive::subject_kappa_base_seed(seed, "1", 1);
    let kappas: Vec<Vec<f64>> = (0..decisions.len())
        .map(|g| {
            let z: Vec<f64> = (0..model.n_kappa)
                .map(|k| crate::sim::adaptive::kappa_standard_normal(base, g, k))
                .collect();
            (&omega_iov.chol * nalgebra::DVector::from_column_slice(&z))
                .iter()
                .copied()
                .collect()
        })
        .collect();
    assert!(
        kappas.iter().any(|k| k[0].abs() > 1e-6),
        "the reconstructed κ must be genuinely nonzero, else the oracle is vacuous"
    );

    // Static reference: predict_iov on the realized doses with occasion = window.
    let static_doses: Vec<DoseEvent> = res
        .ledger
        .iter()
        .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0))
        .collect();
    let mut static_subject = subj("1", obs.clone(), static_doses);
    static_subject.occasions = obs
        .iter()
        .map(|&t| crate::pk::occasion_of(&decisions, t).expect("obs in a window") as u32)
        .collect();
    static_subject.dose_occasions = res
        .ledger
        .iter()
        .map(|e| crate::pk::occasion_of(&decisions, e.time).expect("dose in a window") as u32)
        .collect();
    let preds = crate::pk::predict_iov(
        &model,
        &static_subject,
        &model.default_params.theta,
        &eta_bsv,
        &kappas,
    );

    assert_eq!(res.trajectories.len(), preds.len());
    for (traj, &pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred).abs() <= 1e-9 + 1e-9 * pred.abs(),
            "adaptive IOV IPRED {} != predict_iov {} at t={}",
            traj.ipred,
            pred,
            traj.time
        );
    }
}

#[test]
fn adaptive_iov_reset_matches_predict_iov_and_zeros_state() {
    // #716 × #701: the reset-aware path on the IOV frozen-replay engine
    // (`adaptive_frozen_replay_tv`) — the third of the three reset sites, and the one
    // no other test reaches. Every other reset test uses a constant model, and
    // `subject_needs_per_event_pk` does NOT flag resets, so only an IOV (or
    // TV-covariate) subject drives `event_pk = Some` and routes the verifier to this
    // engine rather than the constant replay.
    //
    // Oracle: run the reactive driver on a real IOV model carrying a mid-horizon EVID=3
    // reset, reconstruct the exact per-occasion κ, and confirm the trajectory equals
    // `predict_iov` — which routes an ODE subject to the reset-aware event-driven walker
    // (`ode_predictions_event_driven`) — on the realized doses PLUS the same reset. The
    // default-on frozen-replay verifier also runs (driver vs `adaptive_frozen_replay_tv`),
    // so its Ok is part of the assertion: a reset mis-applied on EITHER the driver or the
    // IOV replay surfaces here, and `predict_iov` is an INDEPENDENT third engine so a bug
    // shared by driver+replay cannot hide.
    let model = parse_model_string(ODE_IOV).expect("parse IOV ODE model");
    assert!(model.n_kappa == 1 && model.n_eta == 1);
    let decisions = vec![0.0, 24.0, 48.0, 72.0];
    let obs = decisions.clone();
    let reset_at = 36.0; // between decisions 24 and 48; no dose in (36, 48)
    let seed = 20260725u64;

    let mut base = subj("1", obs.clone(), vec![]);
    base.reset_times = vec![reset_at];
    let pop = population(vec![base]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(seed),
        decision_times: decisions.clone(),
        ..Default::default() // verify = true
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("adaptive IOV+reset sim runs and passes the reset-aware IOV verifier");
    assert_eq!(res.ledger.len(), 4, "a bolus at every decision");

    // Reconstruct BSV η and the per-occasion κ exactly as `run_adaptive_population` drew
    // them (identical derivation to `adaptive_iov_matches_predict_iov_with_reconstructed_kappa`).
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let z_eta: Vec<f64> = (0..model.n_eta).map(|_| rng.sample(normal)).collect();
    let eta_bsv: Vec<f64> = (&model.default_params.omega.chol
        * nalgebra::DVector::from_column_slice(&z_eta))
    .iter()
    .copied()
    .collect();
    let omega_iov = model.default_params.omega_iov.as_ref().expect("omega_iov");
    let base_seed = crate::sim::adaptive::subject_kappa_base_seed(seed, "1", 1);
    let kappas: Vec<Vec<f64>> = (0..decisions.len())
        .map(|g| {
            let z: Vec<f64> = (0..model.n_kappa)
                .map(|k| crate::sim::adaptive::kappa_standard_normal(base_seed, g, k))
                .collect();
            (&omega_iov.chol * nalgebra::DVector::from_column_slice(&z))
                .iter()
                .copied()
                .collect()
        })
        .collect();
    assert!(
        kappas.iter().any(|k| k[0].abs() > 1e-6),
        "the reconstructed κ must be genuinely nonzero, else the oracle is vacuous"
    );

    // Static reference: predict_iov on the realized doses + the SAME reset.
    let static_doses: Vec<DoseEvent> = res
        .ledger
        .iter()
        .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0))
        .collect();
    let mut static_subject = subj("1", obs.clone(), static_doses);
    static_subject.reset_times = vec![reset_at];
    static_subject.occasions = obs
        .iter()
        .map(|&t| crate::pk::occasion_of(&decisions, t).expect("obs in a window") as u32)
        .collect();
    static_subject.dose_occasions = res
        .ledger
        .iter()
        .map(|e| crate::pk::occasion_of(&decisions, e.time).expect("dose in a window") as u32)
        .collect();
    let preds = crate::pk::predict_iov(
        &model,
        &static_subject,
        &model.default_params.theta,
        &eta_bsv,
        &kappas,
    );

    assert_eq!(res.trajectories.len(), preds.len());
    for (traj, &pred) in res.trajectories.iter().zip(preds.iter()) {
        // Event-driven `predict_iov` (`solve_ode`) vs the dense reactive driver
        // (`solve_ode_dense`): agree to solver noise, not the bit. A dropped or
        // mis-applied reset would move a post-reset prediction by O(dose).
        assert!(
            (traj.ipred - pred).abs() <= 1e-6 + 1e-6 * pred.abs(),
            "adaptive IOV+reset IPRED {} != predict_iov {} at t={}",
            traj.ipred,
            pred,
            traj.time
        );
    }

    // Positive control (the reset is not vacuous): re-run without it, same seed, so the
    // κ draws are identical and only the reset differs. At t=48 (post-reset; the 48 h
    // bolus lands on both runs) the no-reset trajectory must read strictly higher, by the
    // retained 0 h + 24 h exposure the reset would have zeroed at t=36.
    let no_reset = subj("1", obs.clone(), vec![]);
    let res_nr = simulate_adaptive(
        &model,
        &population(vec![no_reset]),
        &model.default_params,
        1,
        fixed_bolus,
        &opts,
    )
    .expect("no-reset IOV run");
    let idx48 = obs.iter().position(|&t| t == 48.0).expect("t=48 obs");
    let y_reset = res.trajectories[idx48].ipred;
    let y_noreset = res_nr.trajectories[idx48].ipred;
    assert!(
        y_noreset - y_reset > 1.0,
        "reset must drop the retained 0h+24h exposure at t=48: with-reset {y_reset}, \
         no-reset {y_noreset}"
    );
}

#[test]
fn adaptive_reset_decision_collision_is_rejected() {
    // #716 guard: adding reset times to `break_times` introduces a new dedup collision
    // source — a decision within 1e-15 of a reset but NOT bit-identical would be merged
    // away by the break dedup, silently dropping that decision's exact-bit lookup. The
    // driver rejects it with a typed error instead. Build the collision with the two
    // adjacent doubles around t=1 (ULP ≈ 2.2e-16, below the 1e-15 dedup tolerance): the
    // reset at 1.0 sorts first, so the decision at `nextafter(1.0)` loses the dedup.
    let model = parse_model_string(ODE_NO_IIV).expect("parse no-IIV ODE model");
    let reset_at = 1.0_f64;
    let decision = f64::from_bits(reset_at.to_bits() + 1); // within 1e-15, not bit-equal
    assert!(decision != reset_at && (decision - reset_at).abs() < 1e-15);

    let mut base = subj("1", vec![2.0], vec![]);
    base.reset_times = vec![reset_at];
    let pop = population(vec![base]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: vec![decision],
        ..Default::default()
    };
    let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect_err("a decision within 1e-15 of a reset must be rejected, not silently dropped");
    assert!(
        err.contains("1e-15") && err.to_lowercase().contains("reset"),
        "error should cite the 1e-15 reset/break collision: {err}"
    );
}

#[test]
fn reactive_holds_run_and_pass_the_default_verifier() {
    // Genuinely reactive: dose 100 only when the monitored central amount is
    // below 50. k = 0.1/h, so after the t=0 dose the amount is 100·e^{-0.2} ≈
    // 81.9 at t=2 and ≈67.0 at t=4 — both > 50 → hold. Exactly one realized
    // dose, and the t=2 / t=4 holds make the driver break where the static
    // replay of the ledger (dose @0 only) does not — so this run passes
    // through the verifier's *tolerance* branch, not the bit-exact path.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let pop = population(vec![subj("1", vec![1.0, 3.0, 5.0], vec![])]);
    let monitors = vec![MonitorSpec::new("A", 1, ObserveMode::Ipred)];
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: vec![0.0, 2.0, 4.0],
        monitors,
        ..Default::default() // verify = true
    };
    let make = || {
        |ctx: &ControllerCtx| {
            if ctx.signal("A").expect("monitor A declared") < 50.0 {
                vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
            } else {
                vec![DoseAction::Hold]
            }
        }
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, make, &opts)
        .expect("verifier passes on the reactive run");
    // One decision per schedule point is logged, including the two holds.
    assert_eq!(res.decisions.len(), 3);
    // Only the t=0 decision (amount 0 < 50) dosed.
    assert_eq!(res.ledger.len(), 1);
    assert_eq!(res.ledger[0].time, 0.0);
}

#[test]
fn fresh_controller_per_subject_prevents_state_leak() {
    // The "no happy paths" guarantee of the factory signature: a *stateful*
    // controller that doses only on its first call must dose every subject —
    // proving each gets a fresh controller. A single shared FnMut would dose
    // only the first subject (its counter would already be spent), so this
    // asserts the leak is structurally impossible.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let pop = population(vec![
        subj("A", vec![1.0], vec![]),
        subj("B", vec![1.0], vec![]),
        subj("C", vec![1.0], vec![]),
    ]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: vec![0.0],
        ..Default::default()
    };
    let make = || {
        let mut calls = 0usize;
        move |_ctx: &ControllerCtx| {
            let actions = if calls == 0 {
                vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
            } else {
                vec![DoseAction::Hold]
            };
            calls += 1;
            actions
        }
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, make, &opts).expect("runs");
    assert_eq!(
        res.ledger.len(),
        3,
        "every subject's first decision should dose (fresh controller each)"
    );
    let mut ids: Vec<&str> = res.ledger.iter().map(|e| e.subject.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, ["A", "B", "C"]);
}

#[test]
fn replicate_tags_are_stamped_on_every_row() {
    // The single-subject driver emits draw/sim = 0; the orchestrator must
    // stamp the real replicate index onto trajectory, ledger, and decision
    // rows so the artifacts join.
    let model = parse_model_string(ODE_IIV).expect("parse");
    let pop = population(vec![subj("1", vec![6.0, 30.0], vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(42),
        decision_times: vec![0.0, 24.0],
        ..Default::default()
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 2, fixed_bolus, &opts)
        .expect("runs");

    let sims: std::collections::BTreeSet<usize> = res.ledger.iter().map(|e| e.sim).collect();
    assert_eq!(sims, [1, 2].into_iter().collect());
    assert!(res.ledger.iter().all(|e| e.draw == 1));
    assert!(res
        .trajectories
        .iter()
        .all(|t| t.draw == 1 && (t.sim == 1 || t.sim == 2)));
    assert!(res
        .decisions
        .iter()
        .all(|d| d.draw == 1 && (d.sim == 1 || d.sim == 2)));
}

#[test]
fn same_seed_is_deterministic() {
    let model = parse_model_string(ODE_IIV).expect("parse");
    let pop = population(vec![
        subj("1", vec![6.0, 30.0], vec![]),
        subj("2", vec![6.0, 30.0], vec![]),
    ]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: vec![0.0, 24.0],
        ..Default::default()
    };
    let a = simulate_adaptive(&model, &pop, &model.default_params, 2, fixed_bolus, &opts)
        .expect("run a");
    let b = simulate_adaptive(&model, &pop, &model.default_params, 2, fixed_bolus, &opts)
        .expect("run b");

    let ip =
        |r: &AdaptiveSimulationResult| r.trajectories.iter().map(|t| t.ipred).collect::<Vec<_>>();
    assert_eq!(
        ip(&a),
        ip(&b),
        "IPRED trajectories must be seed-reproducible"
    );
    assert_eq!(a.ledger, b.ledger);
    assert_eq!(a.decisions, b.decisions);
}

#[test]
fn rejects_analytical_model() {
    let model = parse_model_string(ANALYTICAL).expect("parse analytical");
    let pop = population(vec![subj("1", vec![6.0], vec![])]);
    let opts = AdaptiveSimulateOptions {
        decision_times: vec![0.0],
        ..Default::default()
    };
    let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect_err("analytical model must be rejected");
    assert!(err.contains("ODE"), "got: {err}");
}

#[test]
fn rejects_empty_decision_schedule() {
    // The default `decision_times` is empty; a caller who forgets to set it
    // would otherwise get a silent dose-free run that passes the verifier
    // trivially. That must be a typed error, not a happy-path no-op.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let pop = population(vec![subj("1", vec![6.0], vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default() // decision_times left empty
    };
    let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect_err("an empty decision schedule must be rejected");
    assert!(err.contains("decision schedule"), "got: {err}");
}

#[test]
fn adaptive_loading_dose_only_matches_static_predict() {
    // #702 degenerate oracle: a base loading regimen with a controller that never doses
    // (Hold at every decision) must reproduce predict() on the loading regimen alone —
    // the pre-scheduled doses are integrated, the controller adds nothing. The default-on
    // frozen-replay verifier (now base-aware) also runs, and its Ok is part of this
    // assertion: a verifier that dropped the base regimen would diverge and error.
    let model = parse_model_string(ODE_NO_IIV).expect("parse no-IIV ODE model");
    // Decisions coincide with the loading-dose times, so the reactive driver and predict()
    // build the identical segmentation and agree bit-for-bit (a decision landing off the
    // dose grid would add a break predict() lacks, diverging by RK45 step noise ~1e-5 —
    // still caught bit-exactly by the on-by-default frozen-replay verifier, which feeds the
    // decision times to both engines; the realistic off-grid maintenance schedule is what
    // the mrgsolve loading-dose anchor exercises).
    let decisions = vec![0.0, 24.0];
    let obs = vec![6.0, 30.0, 54.0];
    let loading = vec![
        DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0),
        DoseEvent::new(24.0, 250.0, 1, 0.0, false, 0.0),
    ];

    let pop = population(vec![subj("1", obs.clone(), loading.clone())]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(3),
        decision_times: decisions.clone(),
        ..Default::default() // verify = true
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, hold_all, &opts)
        .expect("base-regimen adaptive sim runs and passes the base-aware verifier");
    assert!(res.ledger.is_empty(), "controller issues no doses (Hold)");

    // Static reference: the loading regimen alone, scored by predict() (η=0, IPRED).
    let static_pop = population(vec![subj("1", obs.clone(), loading)]);
    let preds = predict(&model, &static_pop, &model.default_params).unwrap();

    assert_eq!(res.trajectories.len(), obs.len());
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-9 + 1e-9 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={}",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
}

#[test]
fn adaptive_loading_dose_changes_trajectory_positive_control() {
    // Proof the base regimen is actually integrated (the oracle above is not vacuously
    // matching two dose-free runs): the same Hold-all controller with vs without a loading
    // dose must diverge — with the 500 mg loading dose the first observation is hundreds of
    // units; without it (dose-free + Hold) every prediction is 0.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let decisions = vec![0.0, 24.0, 48.0];
    let obs = vec![6.0, 30.0, 54.0];
    let opts = AdaptiveSimulateOptions {
        seed: Some(3),
        decision_times: decisions.clone(),
        ..Default::default()
    };

    let with_load = population(vec![subj(
        "1",
        obs.clone(),
        vec![DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0)],
    )]);
    let r_load = simulate_adaptive(
        &model,
        &with_load,
        &model.default_params,
        1,
        hold_all,
        &opts,
    )
    .expect("loading-dose run");

    let no_dose = population(vec![subj("1", obs.clone(), vec![])]);
    let r_none = simulate_adaptive(&model, &no_dose, &model.default_params, 1, hold_all, &opts)
        .expect("dose-free run");

    assert!(
        r_load.trajectories[0].ipred > 100.0,
        "loading dose must be visible at t=6, got {}",
        r_load.trajectories[0].ipred
    );
    assert_eq!(
        r_none.trajectories[0].ipred, 0.0,
        "dose-free Hold: nothing in the system"
    );
}

#[test]
fn adaptive_loading_dose_plus_titration_matches_static_predict() {
    // #702: a base loading dose augmented by a fixed-dose controller must reproduce
    // predict() on (loading regimen ∪ realized ledger). State-independent dosing, so the
    // base-dose-then-decision ordering at t=0 (base 500 applied, then the decision's 100)
    // is unambiguous and the union is exactly what both engines integrate.
    let model = parse_model_string(ODE_NO_IIV).expect("parse no-IIV ODE model");
    let decisions = vec![0.0, 24.0, 48.0];
    let obs = vec![6.0, 30.0, 54.0];
    let loading = vec![DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0)];

    let pop = population(vec![subj("1", obs.clone(), loading.clone())]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(5),
        decision_times: decisions.clone(),
        ..Default::default()
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("base+titration adaptive sim runs and passes the verifier");
    assert_eq!(res.ledger.len(), 3, "a controller bolus at every decision");

    // Static reference: loading regimen + the realized controller doses.
    let mut static_doses = loading.clone();
    static_doses.extend(
        res.ledger
            .iter()
            .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0)),
    );
    let static_pop = population(vec![subj("1", obs.clone(), static_doses)]);
    let preds = predict(&model, &static_pop, &model.default_params).unwrap();

    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-9 + 1e-9 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={}",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
}

#[test]
fn adaptive_ss_base_dose_matches_static_predict() {
    // #702 steady-state: a base SS=1 (II>0) maintenance dose pre-equilibrates the
    // compartment, then a fixed controller augments it. Must reproduce predict() on the
    // same (SS base ∪ ledger) regimen — exercising `equilibrate_ss_state` through the
    // shared apply/verifier path. The base-aware frozen-replay verifier also runs.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let decisions = vec![0.0, 24.0];
    let obs = vec![1.0, 12.0, 30.0];
    // SS dose: 300 mg q24h at steady state, seeded at t=0.
    let ss = DoseEvent::new(0.0, 300.0, 1, 0.0, true, 24.0);
    let pop = population(vec![subj("1", obs.clone(), vec![ss.clone()])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(9),
        decision_times: decisions.clone(),
        ..Default::default()
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("SS base + titration runs and passes the verifier");
    assert_eq!(res.ledger.len(), 2);

    let mut static_doses = vec![ss];
    static_doses.extend(
        res.ledger
            .iter()
            .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0)),
    );
    let static_pop = population(vec![subj("1", obs.clone(), static_doses)]);
    let preds = predict(&model, &static_pop, &model.default_params).unwrap();

    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-9 + 1e-9 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={} (SS base dose)",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
}

#[test]
fn adaptive_base_infusion_matches_static_predict() {
    // #702: a pre-scheduled zero-order infusion (base regimen) delivered over its window,
    // augmented by a fixed bolus controller. Must reproduce predict() on (base ∪ ledger),
    // exercising base-infusion break placement + `active_infusions` on the reactive path.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let decisions = vec![0.0, 24.0];
    let obs = vec![1.0, 6.0, 30.0];
    // 200 mg infused over 4 h (rate 50) starting at t=0.
    let inf = DoseEvent::new(0.0, 200.0, 1, 50.0, false, 0.0);
    let pop = population(vec![subj("1", obs.clone(), vec![inf.clone()])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(11),
        decision_times: decisions.clone(),
        ..Default::default()
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("base infusion + titration runs and passes the verifier");

    let mut static_doses = vec![inf];
    static_doses.extend(
        res.ledger
            .iter()
            .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0)),
    );
    let static_pop = population(vec![subj("1", obs.clone(), static_doses)]);
    let preds = predict(&model, &static_pop, &model.default_params).unwrap();
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-9 + 1e-9 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={} (base infusion)",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
}

#[test]
fn adaptive_base_loading_under_tv_covariate_matches_closed_form() {
    // #930: a pre-scheduled loading dose integrated under a *time-varying* covariate. CL reads
    // a declining CRCL, so each segment's decay uses the covariate at the segment's END (NONMEM
    // end-of-interval, #700) — and the base loading dose rides that per-segment PK. Independent
    // hand-checked closed form (NOT the frozen-replay twin): a 1000-unit bolus at t=0 decays
    // over two 24 h segments with k = CL/V = (5·CRCL/100)/50. With CRCL = 50 on both
    // post-baseline records, k = 0.05/h in each segment, so the trajectory is a double
    // exponential. The default-on base-aware frozen-replay verifier also runs (its Ok is part
    // of the assertion). Mutation: freezing the segment PK at the t=0 CRCL=100 would give
    // k=0.1 and a far smaller trajectory (301 → 90 vs 91 → 8).
    let model = parse_model_string(ODE_TV_COV).expect("parse TV-cov ODE model");
    let mut s = subj(
        "1",
        vec![0.0, 24.0, 48.0],
        vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
    );
    s.covariates = HashMap::from([("CRCL".to_string(), 100.0)]);
    s.obs_covariates = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]),
        HashMap::from([("CRCL".to_string(), 50.0)]),
        HashMap::from([("CRCL".to_string(), 50.0)]),
    ];
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: vec![0.0, 24.0, 48.0],
        ..Default::default()
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, hold_all, &opts)
        .expect("base loading × TV covariate runs and passes the base-aware verifier");
    assert!(
        res.ledger.is_empty(),
        "controller holds — the loading dose is the only dose"
    );

    // k = CL/V = (5·50/100)/50 = 0.05 /h in each 24 h segment (end-of-interval CRCL = 50).
    let decay = (-0.05_f64 * 24.0).exp();
    let expect = [1000.0, 1000.0 * decay, 1000.0 * decay * decay];
    // RK45 vs the analytic closed form: bound by a small multiple of the solver's own error
    // control (default reltol 1e-4 / abstol 1e-6), the same 8× idiom the frozen-replay verifier
    // uses — ~1e4× tighter than the 70% trajectory shift a covariate-frozen base dose would give.
    for (traj, want) in res.trajectories.iter().zip(expect.iter()) {
        assert!(
            (traj.ipred - want).abs() <= 8.0 * (1e-6 + 1e-4 * want),
            "t={}: base loading IPRED {} != closed form {want} (per-segment CL under TV)",
            traj.time,
            traj.ipred
        );
    }
}

#[test]
fn adaptive_base_infusion_ending_between_records_matches_static_predict() {
    // Degenerate oracle for the #1073 record convention on the REACTIVE walk: a
    // holding controller over a pre-scheduled base infusion must reproduce the static
    // engine (`predict()`, which routes here to `ode_predictions_event_driven`) on the
    // same realized regimen.
    //
    // The geometry is the one #1073 moved. A 1000-unit infusion at rate 200 runs
    // `[0, 5]`, so its window **end** falls strictly between the `t = 0` record
    // (CRCL = 100) and the `t = 6` record (CRCL = 50). An infusion end is not a data
    // record: it subdivides the interval the `t = 6` record terminates, and both
    // pieces run on that record's snapshot (k = 0.05/h). Resolving the `(0, 5]` piece
    // to the *previous* record instead — the LOCF carry-forward — integrates it at
    // k = 0.1/h.
    //
    // The model is eta-invariant, so the adaptive IPRED equals the eta=0 static PRED.
    // The default-on frozen-replay verifier also runs, but it cannot see this: the
    // driver and `adaptive_frozen_replay_tv` share the same segment resolution, so they
    // agree with each other whichever record they pick. Only the static engine is an
    // independent reference.
    let model = parse_model_string(ODE_TV_COV).expect("parse TV-cov ODE model");
    let obs = vec![0.0, 6.0, 12.0];
    let cov = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]),
        HashMap::from([("CRCL".to_string(), 50.0)]),
        HashMap::from([("CRCL".to_string(), 50.0)]),
    ];
    // amt 1000 @ rate 200 -> a 5 h window, ending between the t=0 and t=6 records.
    let dose = DoseEvent::new(0.0, 1000.0, 1, 200.0, false, 0.0);

    let mut s = subj("1", obs.clone(), vec![dose.clone()]);
    s.covariates = HashMap::from([("CRCL".to_string(), 100.0)]);
    s.obs_covariates = cov.clone();
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];

    let opts = AdaptiveSimulateOptions {
        seed: Some(11),
        decision_times: vec![0.0, 6.0, 12.0],
        ..Default::default()
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, hold_all, &opts)
        .expect("base infusion x TV covariate runs");
    assert!(res.ledger.is_empty(), "the controller holds throughout");

    // Static reference: the identical subject through `predict()`.
    let mut s2 = subj("1", obs.clone(), vec![dose]);
    s2.covariates = HashMap::from([("CRCL".to_string(), 100.0)]);
    s2.obs_covariates = cov;
    let mut static_pop = population(vec![s2]);
    static_pop.covariate_names = vec!["CRCL".to_string()];
    let preds = predict(&model, &static_pop, &model.default_params).unwrap();

    assert_eq!(res.trajectories.len(), preds.len());
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-6 + 1e-6 * pred.pred.abs(),
            "t={}: adaptive IPRED {} != static predict {} - the reactive walk resolved \
             the infusion-end segment to a different record than production (#1073)",
            traj.time,
            traj.ipred,
            pred.pred
        );
    }
}

#[test]
fn adaptive_locf_carry_does_not_advance_past_a_non_record_break() {
    // Since #1073 the reactive walk resolves a segment FORWARD — a break that is not a
    // data record takes the next record ahead. The LOCF carry must NOT follow it there.
    //
    // `last_pk` is the covariate a controller sees: it feeds the decision-time readout
    // and fixes the bioavailability of any dose injected at that decision. Advancing it
    // from the segment's own (forward-looking) snapshot would let a decision at a
    // non-record instant read a covariate that has not been recorded yet — the walk
    // reaching into its own future. So the carry advances only at an actual record
    // (`records.at(t_end)`), which is also what keeps a dose arrival, an infusion end,
    // a zero-order cutoff and an EVID=3/4 reset from disturbing it at all.
    //
    // Fixture: `F` reads `CRCL`, which steps 100 -> 50 at the `t = 24` record. The
    // second decision is at `t = 12` — deliberately NOT a record — so its dose's `F`
    // comes through `last_pk`. LOCF gives `CRCL = 100` (the `t = 0` record) and
    // `F = 0.8`; leaking the forward resolution gives `CRCL = 50` and `F = 0.4`, a
    // factor of two in the delivered dose that shows up at every later observation.
    //
    // Non-IOV on purpose: under IOV `last_pk` is overwritten by `decision_pk[g]` at the
    // top of a decision break, which would mask the leak.
    let model = parse_model_string(ODE_TV_F).expect("parse TV-F ODE model");
    let obs = vec![0.0, 24.0, 48.0];
    let mut s = subj("1", obs.clone(), vec![]);
    s.covariates = HashMap::from([("CRCL".to_string(), 100.0)]);
    s.obs_covariates = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]),
        HashMap::from([("CRCL".to_string(), 50.0)]),
        HashMap::from([("CRCL".to_string(), 50.0)]),
    ];
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];

    let opts = AdaptiveSimulateOptions {
        seed: Some(13),
        // t=12 is not an observation, so the decision there is a non-record break.
        decision_times: vec![0.0, 12.0],
        ..Default::default()
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("TV-F subject with an off-record decision runs");
    assert_eq!(res.ledger.len(), 2, "a 100-unit bolus at each decision");

    // Both doses land under the LOCF covariate CRCL = 100, so both deliver
    // F * 100 = 0.8 * 100 = 80 units. The leak would make the second deliver 40.
    for (i, e) in res.ledger.iter().enumerate() {
        assert!(
            (e.f_applied - 0.8).abs() < 1e-12,
            "dose {i} at t={}: F = {} (LOCF CRCL = 100 gives 0.8; the forward leak \
             gives 0.4)",
            e.time,
            e.f_applied
        );
    }

    // And the trajectory, so the assertion is not only about bookkeeping. k = CL/V =
    // 0.1/h; 80 units at t=0 decay to t=12, 80 more land there, and the sum decays to
    // t=24 and t=48.
    let k: f64 = 5.0 / 50.0;
    let a12 = 80.0 * (-12.0 * k).exp() + 80.0;
    let expect = [80.0, a12 * (-12.0 * k).exp(), a12 * (-36.0 * k).exp()];
    for (traj, want) in res.trajectories.iter().zip(expect.iter()) {
        assert!(
            (traj.ipred - want).abs() <= 8.0 * (1e-6 + 1e-4 * want),
            "t={}: IPRED {} != closed form {want}",
            traj.time,
            traj.ipred
        );
    }
}

#[test]
fn adaptive_base_dose_f_under_tv_covariate_matches_closed_form() {
    // #930: the base dose's bioavailability F is resolved from ITS OWN covariate snapshot, not
    // the t=0 baseline — the crux of base × TV. F = TVF·CRCL/100 with CL/V constant. A 1000-unit
    // base bolus lands at t=24 where CRCL=60, so F = 0.8·60/100 = 0.48 and 480 units enter the
    // central compartment (a stale t=0 F=0.8 would inject 800 — the mutation this pins). It then
    // decays over 24 h at k = CL/V = 0.1/h. Independent hand-checked closed form.
    let model = parse_model_string(ODE_TV_F).expect("parse TV-F ODE model");
    let mut s = subj(
        "1",
        vec![0.0, 24.0, 48.0],
        vec![DoseEvent::new(24.0, 1000.0, 1, 0.0, false, 0.0)],
    );
    s.covariates = HashMap::from([("CRCL".to_string(), 100.0)]);
    s.obs_covariates = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]),
        HashMap::from([("CRCL".to_string(), 60.0)]),
        HashMap::from([("CRCL".to_string(), 60.0)]),
    ];
    // The dose row carries its own covariate (CRCL = 60 at t = 24) — the snapshot F resolves at.
    s.dose_covariates = vec![HashMap::from([("CRCL".to_string(), 60.0)])];
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];
    let opts = AdaptiveSimulateOptions {
        seed: Some(9),
        decision_times: vec![0.0, 24.0, 48.0],
        ..Default::default()
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, hold_all, &opts)
        .expect("base dose F × TV covariate runs and passes the verifier");
    assert!(res.ledger.is_empty(), "controller holds");

    let entered = (0.8 * 60.0 / 100.0) * 1000.0; // F(t=24)·amt = 0.48·1000 = 480
    let decay = (-0.1_f64 * 24.0).exp(); // k = CL/V = 5/50
    let expect = [0.0, entered, entered * decay];
    // t=24 (post-dose) is exact f64 (F·amt, no integration); t=48 carries one 24 h RK45 decay,
    // bounded by 8× the solver's error control (default reltol 1e-4). The stale-F mutation this
    // pins (F=0.8 → 800 not 480) is 67% off — vastly outside this band.
    for (traj, want) in res.trajectories.iter().zip(expect.iter()) {
        assert!(
            (traj.ipred - want).abs() <= 8.0 * (1e-6 + 1e-4 * want),
            "t={}: base-dose-F IPRED {} != closed form {want} (per-dose F under TV)",
            traj.time,
            traj.ipred
        );
    }
}

#[test]
fn adaptive_base_infusion_under_tv_covariate_matches_closed_form() {
    // #930: a base *infusion* (not just a bolus) integrated under a time-varying covariate.
    // A 1 h zero-order infusion (rate 500 → 500 mg) into CENT, then decay, with CL reading a
    // declining CRCL. The infusion window (0, 1] ends at the t=1 record (CRCL=100 → k=0.1/h),
    // so during infusion A(1) = (rate/k)(1−e^{−k}) = 5000·(1−e^{−0.1}) = 475.813. It then
    // decays under CRCL=60 (k=0.06/h): A(24) = A(1)·e^{−0.06·23}, A(48) = A(24)·e^{−0.06·24}.
    // Independent hand-checked closed form — the infusion analogue of the bolus oracles, so a
    // mis-carried base-infusion F / window under a covariate would miss it.
    let model = parse_model_string(ODE_TV_COV).expect("parse TV-cov ODE model");
    let mut s = subj(
        "1",
        vec![1.0, 24.0, 48.0],
        vec![DoseEvent::new(0.0, 500.0, 1, 500.0, false, 0.0)], // rate 500 → 1 h infusion
    );
    s.covariates = HashMap::from([("CRCL".to_string(), 100.0)]);
    s.obs_covariates = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]), // t=1: governs the infusion window
        HashMap::from([("CRCL".to_string(), 60.0)]),  // t=24
        HashMap::from([("CRCL".to_string(), 60.0)]),  // t=48
    ];
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];
    let opts = AdaptiveSimulateOptions {
        seed: Some(13),
        decision_times: vec![0.0, 24.0, 48.0],
        ..Default::default()
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, hold_all, &opts)
        .expect("base infusion × TV covariate runs and passes the base-aware verifier");
    assert!(
        res.ledger.is_empty(),
        "controller holds — the base infusion is the only dose"
    );

    let a1 = (500.0 / 0.1) * (1.0 - (-0.1_f64).exp()); // end of the 1 h infusion, k = 0.1
    let a24 = a1 * (-0.06_f64 * 23.0).exp(); // decay under CRCL=60 (k = 0.06)
    let a48 = a24 * (-0.06_f64 * 24.0).exp();
    let expect = [a1, a24, a48];
    for (traj, want) in res.trajectories.iter().zip(expect.iter()) {
        assert!(
            (traj.ipred - want).abs() <= 8.0 * (1e-6 + 1e-4 * want),
            "t={}: base-infusion IPRED {} != closed form {want} (infusion under per-segment TV CL)",
            traj.time,
            traj.ipred
        );
    }
}

#[test]
fn adaptive_base_loading_plus_titration_under_tv_covariate_passes_verifier() {
    // #930: a base loading dose augmented by a controller under a time-varying covariate. The
    // controller doses at every decision (state-independent), so base ∪ realized ledger
    // integrate under per-segment PK and the base-aware TV frozen-replay verifier (default-on,
    // now carrying the base dose's F in `dose_f`) must accept the run — exercising the
    // `verify_adaptive_frozen_replay` TV branch with a non-empty base slice.
    let model = parse_model_string(ODE_TV_COV).expect("parse TV-cov ODE model");
    let mut s = subj(
        "1",
        vec![0.0, 24.0, 48.0],
        vec![DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0)],
    );
    s.covariates = HashMap::from([("CRCL".to_string(), 100.0)]);
    s.obs_covariates = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]),
        HashMap::from([("CRCL".to_string(), 70.0)]),
        HashMap::from([("CRCL".to_string(), 50.0)]),
    ];
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];
    let opts = AdaptiveSimulateOptions {
        seed: Some(11),
        decision_times: vec![0.0, 24.0, 48.0],
        ..Default::default() // verify = true
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("base + titration × TV runs and passes the base-aware verifier");
    assert_eq!(res.ledger.len(), 3, "a controller bolus at every decision");
}

#[test]
fn adaptive_base_ss_dose_with_tv_covariate_is_rejected() {
    // #930 scope: a steady-state base dose under a time-varying covariate is a typed error (SS
    // equilibration needs per-dose PK threaded through the TV frozen-replay engine — a
    // follow-up), never a silent covariate-frozen SS seed.
    let model = parse_model_string(ODE_TV_COV).expect("parse TV-cov ODE model");
    let mut s = subj(
        "1",
        vec![6.0, 30.0, 54.0],
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)], // SS=1, II=24
    );
    s.covariates = HashMap::from([("CRCL".to_string(), 100.0)]);
    s.obs_covariates = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]),
        HashMap::from([("CRCL".to_string(), 90.0)]),
        HashMap::from([("CRCL".to_string(), 80.0)]),
    ];
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];
    let opts = AdaptiveSimulateOptions {
        decision_times: vec![0.0, 24.0, 48.0],
        ..Default::default()
    };
    let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect_err("SS base dose × TV covariate must be rejected");
    assert!(
        err.contains("plain fixed bolus or infusion") && err.contains("steady-state"),
        "got: {err}"
    );
}

#[test]
fn adaptive_base_lagged_dose_with_tv_covariate_is_rejected() {
    // #930 scope: a lagged base dose under a time-varying covariate is a typed error (the TV
    // frozen-replay engine carries no base lag yet), never a silently un-lagged integration.
    let model = parse_model_string(ODE_TV_LAG).expect("parse TV-cov + lag ODE model");
    let mut s = subj(
        "1",
        vec![6.0, 30.0, 54.0],
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
    );
    s.covariates = HashMap::from([("CRCL".to_string(), 100.0)]);
    s.obs_covariates = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]),
        HashMap::from([("CRCL".to_string(), 90.0)]),
        HashMap::from([("CRCL".to_string(), 80.0)]),
    ];
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];
    let opts = AdaptiveSimulateOptions {
        decision_times: vec![0.0, 24.0, 48.0],
        ..Default::default()
    };
    let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect_err("lagged base dose × TV covariate must be rejected");
    assert!(
        err.contains("plain fixed bolus or infusion") && err.contains("lagged"),
        "got: {err}"
    );
}

#[test]
fn adaptive_base_modeled_rate_dose_with_tv_covariate_is_rejected() {
    // #930 scope: a base dose carrying a MODELED (coded) RATE under a time-varying covariate is a
    // typed error — the rate-resolution bookkeeping (RATE=-1/-2 → R1/D1 from the PK snapshot) is
    // not threaded through the TV frozen-replay engine yet. The guard tests `is_fixed()` on the
    // *original* dose, before `resolve_subject_doses` collapses a coded RATE to `Fixed`, so this
    // is the arm only that pre-resolution check can trip. Plain central, no lag, not an
    // input-rate compartment, so the modeled-RATE arm is the sole reason this rejects (mutation:
    // drop the `!is_fixed()` arm and the dose is accepted → this `expect_err` fails).
    let model = parse_model_string(ODE_TV_MRATE).expect("parse TV-cov + modeled-rate ODE model");
    let mut s = subj(
        "1",
        vec![0.0, 24.0, 48.0],
        vec![DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledRate,
        )],
    );
    s.covariates = HashMap::from([("CRCL".to_string(), 100.0)]);
    s.obs_covariates = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]),
        HashMap::from([("CRCL".to_string(), 90.0)]),
        HashMap::from([("CRCL".to_string(), 80.0)]),
    ];
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];
    let opts = AdaptiveSimulateOptions {
        decision_times: vec![0.0, 24.0, 48.0],
        ..Default::default()
    };
    let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect_err("modeled-RATE base dose × TV covariate must be rejected");
    assert!(
        err.contains("plain fixed bolus or infusion") && err.contains("modeled-RATE"),
        "got: {err}"
    );
}

#[test]
fn adaptive_base_input_rate_dose_with_tv_covariate_is_rejected() {
    // #930 scope: a base dose into a compartment fed by a built-in input rate (here first-order
    // absorption into `central`) under a time-varying covariate is a typed error — the depot /
    // absorption bookkeeping is not threaded through the TV frozen-replay engine yet. The dose is
    // a plain fixed bolus (not SS, not lagged, not modeled-RATE), so `input_rate_consumes_cmt` is
    // the sole arm that rejects it (mutation: drop that arm and the dose is accepted).
    let model = parse_model_string(ODE_TV_ABSORB).expect("parse TV-cov + absorption ODE model");
    let mut s = subj(
        "1",
        vec![0.0, 24.0, 48.0],
        vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
    );
    s.covariates = HashMap::from([("CRCL".to_string(), 100.0)]);
    s.obs_covariates = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]),
        HashMap::from([("CRCL".to_string(), 90.0)]),
        HashMap::from([("CRCL".to_string(), 80.0)]),
    ];
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];
    let opts = AdaptiveSimulateOptions {
        decision_times: vec![0.0, 24.0, 48.0],
        ..Default::default()
    };
    let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect_err("input-rate base dose × TV covariate must be rejected");
    assert!(
        err.contains("plain fixed bolus or infusion") && err.contains("input-rate"),
        "got: {err}"
    );
}

/// Reconstruct the exact BSV η a seeded single-subject `simulate_adaptive` run drew for
/// (sim 1, subject `id`): the seeded `StdRng`, one N(0,1) per η, then Ω's Cholesky. With
/// `seed` set the assay/κ streams don't touch this rng, so the η draw is its only consumer —
/// reproducible here. Mirrors `run_adaptive_population`'s BSV draw.
fn reconstruct_eta_bsv(model: &CompiledModel, seed: u64) -> Vec<f64> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let z_eta: Vec<f64> = (0..model.n_eta).map(|_| rng.sample(normal)).collect();
    (&model.default_params.omega.chol * nalgebra::DVector::from_column_slice(&z_eta))
        .iter()
        .copied()
        .collect()
}

/// Reconstruct the exact per-occasion κ a seeded run drew for (subject `id`, sim 1):
/// `κ_g = chol(Ω_IOV)·z`, `z` keyed by (occasion, component) on the dedicated substream —
/// the same derivation `run_adaptive_population` and `verify_adaptive_snapshots` use. Returns
/// one κ vector per decision window (`n_occ` = `decision_times.len()`).
fn reconstruct_kappas(model: &CompiledModel, seed: u64, id: &str, n_occ: usize) -> Vec<Vec<f64>> {
    let omega_iov = model.default_params.omega_iov.as_ref().expect("omega_iov");
    let base = crate::sim::adaptive::subject_kappa_base_seed(seed, id, 1);
    (0..n_occ)
        .map(|g| {
            let z: Vec<f64> = (0..model.n_kappa)
                .map(|k| crate::sim::adaptive::kappa_standard_normal(base, g, k))
                .collect();
            (&omega_iov.chol * nalgebra::DVector::from_column_slice(&z))
                .iter()
                .copied()
                .collect()
        })
        .collect()
}

#[test]
fn adaptive_base_loading_under_iov_matches_predict_iov() {
    // #931: the full-stack reconstruction oracle the issue names. A pre-scheduled base loading
    // dose integrated under per-occasion κ must equal `predict_iov` — an INDEPENDENT engine — on
    // the same dose with occasion = decision window. κ is on CL here, so the base dose decays
    // under each window's own clearance (the base-dose analogue of the controller-dose oracle
    // `adaptive_iov_matches_predict_iov`). A build-loop error in the base dose's occasion PK
    // would be applied by both the driver and the frozen-replay verifier; `predict_iov` is a
    // third engine that shares neither, so it catches it. The default-on frozen-replay + #748
    // snapshot verifiers also run — their `Ok` is part of the assertion.
    let model = parse_model_string(ODE_IOV).expect("parse IOV ODE model");
    assert!(model.n_kappa == 1 && model.n_eta == 1);
    let decisions = vec![0.0, 24.0, 48.0, 72.0];
    // Base 1000-unit loading bolus at t=0 (occasion 0). Observe at t=12 (in occasion 0, so the
    // base dose's own-occasion decay is seen) and after each later decision, but NOT at t=0,
    // where a base dose coincident with an observation raises a pre/post-dose readout question
    // orthogonal to this oracle (the driver observes the trough, #933). The observation grid
    // covers occasions 0..=3 in order, so `predict_iov`'s occasion→group index (obs order, then
    // dose-only occasions; `iov_occasion_groups`) matches the reconstructed κ's occasion-id
    // order — the dose's occasion 0 is a group index 0, not appended last.
    let obs = vec![12.0, 24.0, 48.0, 72.0];
    let seed = 20260726u64;
    let base_dose = DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0);
    let pop = population(vec![subj("1", obs.clone(), vec![base_dose])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(seed),
        decision_times: decisions.clone(),
        ..Default::default() // verify = true → frozen-replay + #748 snapshot checks run
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, hold_all, &opts)
        .expect("adaptive IOV base-loading sim runs and passes the default verifiers");
    assert!(
        res.ledger.is_empty(),
        "controller holds — only the base loading dose"
    );

    let eta_bsv = reconstruct_eta_bsv(&model, seed);
    let kappas = reconstruct_kappas(&model, seed, "1", decisions.len());
    assert!(
        kappas.iter().any(|k| k[0].abs() > 1e-6),
        "the reconstructed κ must be genuinely nonzero, else the oracle is vacuous"
    );

    // Static reference: predict_iov on the base loading dose with occasion = decision window.
    let mut static_subject = subj(
        "1",
        obs.clone(),
        vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
    );
    static_subject.occasions = obs
        .iter()
        .map(|&t| crate::pk::occasion_of(&decisions, t).expect("obs in a window") as u32)
        .collect();
    static_subject.dose_occasions =
        vec![crate::pk::occasion_of(&decisions, 0.0).expect("dose in a window") as u32];
    let preds = crate::pk::predict_iov(
        &model,
        &static_subject,
        &model.default_params.theta,
        &eta_bsv,
        &kappas,
    );

    assert_eq!(res.trajectories.len(), preds.len());
    for (traj, &pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred).abs() <= 1e-9 + 1e-9 * pred.abs(),
            "adaptive IOV base-loading IPRED {} != predict_iov {} at t={}",
            traj.ipred,
            pred,
            traj.time
        );
    }
}

#[test]
fn adaptive_base_dose_f_under_iov_matches_closed_form() {
    // #931: the crux of base × IOV — a base dose's bioavailability F resolves under the κ of the
    // occasion (decision window) it lands in, NOT the frozen t=0 baseline (κ=0) and NOT another
    // occasion's κ. F = TVF·exp(κ_F) with CL/V constant, so the only per-occasion effect is on F.
    // A 1000-unit base bolus lands at t=24 which — with decisions at [0, 24] — is occasion 1, so
    // F = 0.5·exp(κ₁_F) and F·1000 units enter central; it then decays at k = CL/V = 0.1/h. Two
    // independent mutations are pinned: a frozen κ=0 gives F = 0.5 → 500 units, and the WRONG
    // occasion (κ₀) gives 0.5·exp(κ₀_F)·1000 — both distinct from the correct κ₁ value. The
    // default-on frozen-replay + #748 snapshot verifiers also run.
    let model = parse_model_string(ODE_IOV_F).expect("parse IOV-on-F ODE model");
    assert!(model.n_kappa == 1 && model.n_eta == 1);
    let decisions = vec![0.0, 24.0];
    let seed = 20260726u64;
    // Base 1000-unit bolus at t=24 (occasion 1); observe at 36 and 48 (both occasion 1, after the
    // dose — no observation coincides with the dose, sidestepping the pre/post-dose readout).
    let base_dose = DoseEvent::new(24.0, 1000.0, 1, 0.0, false, 0.0);
    let s = subj("1", vec![36.0, 48.0], vec![base_dose]);
    let pop = population(vec![s]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(seed),
        decision_times: decisions.clone(),
        ..Default::default() // verify = true
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, hold_all, &opts)
        .expect("base dose F × IOV runs and passes the verifiers");
    assert!(
        res.ledger.is_empty(),
        "controller holds — the base dose is the only dose"
    );

    // Reconstruct the exact per-occasion κ_F; occasion 1 governs the dose at t=24.
    let kappas = reconstruct_kappas(&model, seed, "1", decisions.len());
    let k0_f = kappas[0][0];
    let k1_f = kappas[1][0];
    let f = 0.5 * k1_f.exp(); // F = TVF·exp(κ₁_F), the occasion-1 bioavailability
                              // Non-vacuity guards tied to the closed-form band (rel tol ≈ 8e-4 below), NOT a bare 1e-6:
                              // each mutation must move F well outside the band or the oracle would pass vacuously. The
                              // frozen κ=0 mutation gives F=0.5; the wrong-occasion mutation gives 0.5·exp(κ₀_F). Require
                              // a 10× margin so the oracle's teeth rest on the band, not on the pinned seed alone.
    let rel_band = 8.0 * 1e-4;
    assert!(
        (f - 0.5).abs() / f > 10.0 * rel_band,
        "occasion-1 κ_F={k1_f} too small: a frozen κ=0 (F=0.5) would fall within the \
         closed-form band, making the oracle vacuous"
    );
    assert!(
        (f - 0.5 * k0_f.exp()).abs() / f > 10.0 * rel_band,
        "occasions 0/1 too close (κ₀_F={k0_f}, κ₁_F={k1_f}): the wrong-occasion \
         F=0.5·exp(κ₀_F) would fall within the band, making the oracle vacuous"
    );

    let entered = f * 1000.0;
    let k = 0.1_f64; // CL/V = 5/50; ETA_CL ~ N(0, 1e-10) ≈ 0
    let expect = [entered * (-k * 12.0).exp(), entered * (-k * 24.0).exp()];
    // t=36/48 each carry RK45 decay from the t=24 admin, bounded by 8× the solver's error
    // control. The κ=0 mutation (F=0.5 → 500 vs 0.5·exp(κ₁)) is far outside this band.
    for (traj, want) in res.trajectories.iter().zip(expect.iter()) {
        assert!(
            (traj.ipred - want).abs() <= 8.0 * (1e-6 + 1e-4 * want),
            "t={}: base-dose-F-under-IOV IPRED {} != closed form {want} (F = 0.5·exp(κ₁))",
            traj.time,
            traj.ipred
        );
    }
}

#[test]
fn adaptive_base_loading_plus_titration_under_iov_passes_verifier() {
    // #931: a base loading dose augmented by a controller under IOV. The controller doses at
    // every decision, so base ∪ realized ledger integrate under per-occasion κ, and the
    // base-aware IOV frozen-replay verifier (default-on, `adaptive_frozen_replay_tv` threading
    // `eta_occ`) plus the #748 snapshot check must accept the run — the bit-exact backstop for
    // base × IOV with reactive doses. Observations sit OFF the decision grid, so the driver and
    // the replay must agree segment-for-segment, not merely at the decisions.
    let model = parse_model_string(ODE_IOV).expect("parse IOV ODE model");
    let base_dose = DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0);
    let s = subj("1", vec![12.0, 36.0, 60.0], vec![base_dose]);
    let pop = population(vec![s]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: vec![0.0, 24.0, 48.0],
        ..Default::default() // verify = true
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("base + titration × IOV runs and passes the base-aware IOV verifiers");
    assert_eq!(res.ledger.len(), 3, "a controller bolus at every decision");
}

#[test]
fn adaptive_base_ss_dose_with_iov_is_rejected() {
    // #931 scope: a steady-state base dose under IOV is a typed error — SS equilibration needs
    // per-occasion κ threaded through the frozen-replay engine (a follow-up), never a silent
    // κ=0 SS seed. Proves the IOV path (event_pk = Some, so `tv` is true) reaches the SAME
    // base-dose scope guard the TV path does. The four rejection arms themselves (SS / lag /
    // input-rate / modeled-RATE) are pinned per-arm by the #930 TV tests; this confirms the IOV
    // routing into that shared guard, so an SS base dose under IOV can never slip to a silent
    // frozen-κ integration.
    let model = parse_model_string(ODE_IOV).expect("parse IOV ODE model");
    let s = subj(
        "1",
        vec![6.0, 30.0],
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)], // SS=1, II=24
    );
    let pop = population(vec![s]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: vec![0.0, 24.0],
        ..Default::default()
    };
    let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect_err("SS base dose × IOV must be rejected");
    assert!(
        err.contains("plain fixed bolus or infusion") && err.contains("steady-state"),
        "got: {err}"
    );
}

#[test]
fn adaptive_base_dose_f_before_first_decision_uses_baseline_kappa() {
    // #931 convention (pins the pre-first-decision boundary raised in review): a base dose
    // administered BEFORE the first decision has no open occasion window, so its bioavailability
    // F resolves at the baseline κ = 0 — exactly like an observation before the first decision —
    // NOT occasion 0's κ. F = TVF·exp(κ_F) with CL/V constant. A 1000-unit base bolus at t=0 with
    // the first decision at t=24 is in the baseline window, so F = TVF·exp(0) = 0.5 and 500 units
    // enter central; it decays at k = 0.1/h (CL constant, so the per-segment occasion never
    // touches the trajectory here). Mutation pinned: resolving the dose at occasion 0's κ would
    // give 0.5·exp(κ₀_F)·1000 ≠ 500.
    let model = parse_model_string(ODE_IOV_F).expect("parse IOV-on-F ODE model");
    let decisions = vec![24.0, 48.0]; // first decision at t=24; the dose at t=0 precedes it
    let seed = 20260726u64;
    let base_dose = DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0);
    let s = subj("1", vec![12.0, 36.0], vec![base_dose]);
    let pop = population(vec![s]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(seed),
        decision_times: decisions.clone(),
        ..Default::default() // verify = true
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, hold_all, &opts)
        .expect("pre-first-decision base dose F × IOV runs and passes the verifiers");
    assert!(res.ledger.is_empty(), "controller holds");

    // Occasion 0's κ must be far enough from baseline (0) that "baseline vs occasion 0" is
    // distinguishable beyond the closed-form band (rel tol ≈ 8e-4).
    let kappas = reconstruct_kappas(&model, seed, "1", decisions.len());
    let k0_f = kappas[0][0];
    assert!(
        (k0_f.exp() - 1.0).abs() > 10.0 * 8.0 * 1e-4,
        "occasion-0 κ_F={k0_f} too small to distinguish baseline from occasion 0"
    );

    let entered = 0.5 * 1000.0; // baseline κ = 0 → F = TVF·exp(0) = 0.5
    let k = 0.1_f64;
    // obs@12 is in the baseline window (12 < 24), obs@36 in occasion 0 — but CL is constant, so
    // decay is k=0.1 throughout; only F is occasion-sensitive, and here it is baseline.
    let expect = [entered * (-k * 12.0).exp(), entered * (-k * 36.0).exp()];
    for (traj, want) in res.trajectories.iter().zip(expect.iter()) {
        assert!(
            (traj.ipred - want).abs() <= 8.0 * (1e-6 + 1e-4 * want),
            "t={}: pre-first-decision base-dose F IPRED {} != closed form {want} (baseline F=0.5)",
            traj.time,
            traj.ipred
        );
    }
}

#[test]
fn adaptive_base_infusion_under_iov_matches_predict_iov() {
    // #931: a base INFUSION (not just a bolus) under IOV — closes the review's coverage gap (the
    // other IOV base tests all use boluses). κ on F reshapes the infusion's delivered amount per
    // occasion, and the infusion window feeds through `active_infusions` reading the
    // occasion-corrected F (a distinct delivery path from the bolus jump). Validated against the
    // independent `predict_iov` engine, which handles F-reshaped infusions with per-dose occasion
    // κ. Base infusion at t=0 (occasion 0), RATE=1000 over AMT=1000 (nominal 1 h, F-reshaped).
    let model = parse_model_string(ODE_IOV_F).expect("parse IOV-on-F ODE model");
    assert!(model.n_kappa == 1 && model.n_eta == 1);
    let decisions = vec![0.0, 24.0, 48.0, 72.0];
    // obs cover occasions 0..=3 in order so predict_iov's occasion→group index is identity (see
    // `adaptive_base_loading_under_iov_matches_predict_iov` for the group-index alignment note).
    let obs = vec![12.0, 24.0, 48.0, 72.0];
    let seed = 20260726u64;
    let base_inf = DoseEvent::new(0.0, 1000.0, 1, 1000.0, false, 0.0);
    let pop = population(vec![subj("1", obs.clone(), vec![base_inf])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(seed),
        decision_times: decisions.clone(),
        ..Default::default()
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, hold_all, &opts)
        .expect("base infusion × IOV runs and passes the default verifiers");
    assert!(
        res.ledger.is_empty(),
        "controller holds — only the base infusion"
    );

    let eta_bsv = reconstruct_eta_bsv(&model, seed);
    let kappas = reconstruct_kappas(&model, seed, "1", decisions.len());
    assert!(
        kappas.iter().any(|k| k[0].abs() > 1e-6),
        "reconstructed κ must be nonzero"
    );

    let mut static_subject = subj(
        "1",
        obs.clone(),
        vec![DoseEvent::new(0.0, 1000.0, 1, 1000.0, false, 0.0)],
    );
    static_subject.occasions = obs
        .iter()
        .map(|&t| crate::pk::occasion_of(&decisions, t).expect("obs in a window") as u32)
        .collect();
    static_subject.dose_occasions =
        vec![crate::pk::occasion_of(&decisions, 0.0).expect("dose in a window") as u32];
    let preds = crate::pk::predict_iov(
        &model,
        &static_subject,
        &model.default_params.theta,
        &eta_bsv,
        &kappas,
    );
    assert_eq!(res.trajectories.len(), preds.len());
    for (traj, &pred) in res.trajectories.iter().zip(preds.iter()) {
        // The infusion integrates over its window in both engines; a small cross-structure RK45
        // band (not 1e-9 as the bolus oracle uses) covers step-sequence differences, while a
        // wrong occasion F (~%-level) is still caught by ~100×.
        assert!(
            (traj.ipred - pred).abs() <= 8.0 * (1e-6 + 1e-4 * pred.abs()),
            "adaptive IOV base-infusion IPRED {} != predict_iov {} at t={}",
            traj.ipred,
            pred,
            traj.time
        );
    }
}

#[test]
fn adaptive_base_multiple_doses_across_occasions_pass_verifiers() {
    // #931: MULTIPLE base doses spanning different occasions — closes the review's coverage gap
    // (every other IOV base test has n_base = 1). Exercises the index-parallel
    // `event_pk.dose[k] ↔ resolved_base.doses[k]` alignment for k ≥ 2 across occasions: the
    // default-on #748 snapshot check independently re-derives EACH base dose's occasion snapshot
    // (k=0 in occasion 0, k=1 in occasion 1), and the default-on frozen-replay verifier
    // bit-exact-checks the driver's two-dose integration against the static engine. κ on CL, so
    // each base dose decays under its window's clearance. Base boluses at t=0 (occasion 0) and
    // t=24 (occasion 1); observations sit off the dose times.
    //
    // Oracle note — why NOT `predict_iov` here: the adaptive driver assigns a segment's occasion
    // from the last observation/EVID=2 record crossed (`segment_occ_at`, the #701 decision-window
    // / end-of-interval convention — the clearance in effect *during* the segment, the physically
    // faithful choice for real-time feedback). `predict_iov` follows the #104 OCC-column
    // convention (the segment ending at a *dose* record uses that dose's occasion). The two agree
    // when every segment ends at an observation (Test A, the mrgsolve anchor), but differ for a
    // base dose at an occasion boundary with no coincident observation — as here — so `predict_iov`
    // is not a valid oracle for this configuration. The driver's own verifiers are; the
    // independent-engine multi-occasion check is the mrgsolve anchor. See
    // `compute_event_pk_params_iov`'s docstring.
    let model = parse_model_string(ODE_IOV).expect("parse IOV ODE model");
    let doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), // occasion 0
        DoseEvent::new(24.0, 500.0, 1, 0.0, false, 0.0), // occasion 1 (at the decision boundary)
    ];
    let obs = vec![12.0, 36.0, 60.0]; // occasions 0, 1, 2 — none coincident with a dose
    let pop = population(vec![subj("1", obs, doses)]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(20260726),
        decision_times: vec![0.0, 24.0, 48.0, 72.0],
        ..Default::default() // verify = true → frozen-replay + #748 per-dose snapshot checks run
    };
    // The `expect` IS the assertion: the default-on #748 per-dose re-derivation (both k=0 and k=1)
    // and the bit-exact frozen replay of the two-dose trajectory must both pass.
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, hold_all, &opts)
        .expect("two base doses across occasions run and pass the frozen-replay + #748 verifiers");
    assert!(
        res.ledger.is_empty(),
        "controller holds — only the two base doses"
    );
    // Liveness: both base doses are integrated (every trough is a positive decayed amount).
    assert!(
        res.trajectories.iter().all(|t| t.ipred > 0.0),
        "both base doses contribute a positive decayed trajectory"
    );
}

#[test]
fn adaptive_base_regimen_with_reset_matches_static_predict() {
    // Degenerate oracle for #932: a pre-scheduled base regimen (a loading dose) combined
    // with a mid-horizon EVID=3 system reset and a fixed-dose controller must reproduce the
    // trusted static engine — `predict()`, which routes reset subjects to the reset-aware
    // event-driven walker — on the realized (base ∪ controller) regimen carrying the same
    // reset. The model is η-invariant, so the adaptive IPRED equals the η=0 static PRED.
    //
    // This composes #702's base-dose seeding with #716's reset machinery on the constant-
    // covariate path: the base 500 mg at t=0 drives the pre-reset observations (t=6, t=30),
    // the reset at t=36 zeros that mass (t=42 washes out), and the controller's 100 mg at
    // t=48 drives the post-reset tail (t=54). The default-on frozen-replay verifier (reset-
    // and base-aware) runs too, so its Ok is part of this assertion.
    let model = parse_model_string(ODE_NO_IIV).expect("parse no-IIV ODE model");
    let decisions = vec![24.0, 48.0];
    let obs = vec![6.0, 30.0, 42.0, 54.0];
    let reset_at = 36.0;
    let base_dose = DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0);

    let mut base = subj("1", obs.clone(), vec![base_dose.clone()]);
    base.reset_times = vec![reset_at];
    let pop = population(vec![base]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: decisions.clone(),
        ..Default::default() // verify = true
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("adaptive base+reset sim runs and passes the reset-aware verifier");
    assert_eq!(
        res.ledger.len(),
        2,
        "a controller bolus at each of the two decisions"
    );

    // Static reference: the base regimen + the realized controller doses pre-scheduled on a
    // subject carrying the same reset, scored by predict() (η=0, event-driven, reset honored).
    let mut static_doses = vec![base_dose];
    static_doses.extend(
        res.ledger
            .iter()
            .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0)),
    );
    let mut static_subject = subj("1", obs.clone(), static_doses);
    static_subject.reset_times = vec![reset_at];
    let preds = predict(
        &model,
        &population(vec![static_subject]),
        &model.default_params,
    )
    .unwrap();

    assert_eq!(res.trajectories.len(), obs.len());
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-6 + 1e-4 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={} (base 500@0, reset at {reset_at})",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
}

#[test]
fn adaptive_base_infusion_spanning_reset_is_turned_off() {
    // Positive control for #932 (the issue's non-vacuity proof): a PRE-SCHEDULED base infusion
    // spanning a reset must be turned OFF at the reset by the reset floor — exactly as it turns
    // off a controller-issued infusion (#716). The same base infusion run with vs without the
    // reset must diverge (so the degenerate oracle above is not two engines agreeing on an
    // un-reset trajectory), and the with-reset run must still reproduce predict() on that base
    // infusion carrying the reset.
    let model = parse_model_string(ODE_NO_IIV).expect("parse no-IIV ODE model");
    let decisions = vec![36.0]; // one late decision; the controller only holds
    let obs = vec![6.0, 18.0, 30.0];
    let reset_at = 12.0;
    // Base infusion: 2400 units at rate 100/h ⇒ a 24 h window [0, 24] spanning the reset at 12.
    let base_inf = DoseEvent::new(0.0, 2400.0, 1, 100.0, false, 0.0);
    let opts = AdaptiveSimulateOptions {
        seed: Some(5),
        decision_times: decisions.clone(),
        ..Default::default()
    };

    let mut with_reset = subj("1", obs.clone(), vec![base_inf.clone()]);
    with_reset.reset_times = vec![reset_at];
    let res = simulate_adaptive(
        &model,
        &population(vec![with_reset]),
        &model.default_params,
        1,
        hold_all,
        &opts,
    )
    .expect("base infusion + reset runs and passes the reset-aware verifier");
    assert!(res.ledger.is_empty(), "the hold controller issues no doses");

    // Non-vacuity: without the reset the infusion keeps delivering past 12, so t=18 is
    // materially positive; with the reset it is turned off at 12 and the zeroed state stays ~0.
    let no_reset = subj("1", obs.clone(), vec![base_inf.clone()]);
    let res_no = simulate_adaptive(
        &model,
        &population(vec![no_reset]),
        &model.default_params,
        1,
        hold_all,
        &opts,
    )
    .expect("no-reset run");
    let y18 = res
        .trajectories
        .iter()
        .find(|x| (x.time - 18.0).abs() < 1e-12)
        .expect("t=18 row (with reset)")
        .ipred;
    let y18_no = res_no
        .trajectories
        .iter()
        .find(|x| (x.time - 18.0).abs() < 1e-12)
        .expect("t=18 row (no reset)")
        .ipred;
    assert!(
        y18_no > 10.0 && y18 < 1e-6,
        "reset must turn the base infusion off: with-reset {y18} (want ~0) vs no-reset {y18_no} (want ≫0)"
    );

    // Oracle: the with-reset run reproduces predict() on the same base infusion + reset.
    let mut static_subject = subj("1", obs.clone(), vec![base_inf]);
    static_subject.reset_times = vec![reset_at];
    let preds = predict(
        &model,
        &population(vec![static_subject]),
        &model.default_params,
    )
    .unwrap();
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-6 + 1e-4 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={} (base infusion spanning reset)",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
}

#[test]
fn adaptive_evid4_reset_plus_dose_matches_static_predict() {
    // #932 EVID=4 (reset + dose): a data row that both zeros the state and administers a dose
    // records BOTH a `reset_times` entry and a `doses` entry at the same instant, so its dose
    // is a base dose landing exactly at the reset — zeroed first (Reset < Dose), then applied.
    // Today that dose tripped the base-regimen combo guard before the reset was considered; it
    // must now reach the adaptive path and reproduce predict() (which sorts Reset < Dose too).
    let model = parse_model_string(ODE_NO_IIV).expect("parse no-IIV ODE model");
    let decisions = vec![24.0];
    let obs = vec![6.0, 18.0, 30.0];
    let reset_at = 12.0;
    // A loading dose the reset wipes, plus the EVID=4 dose (300 mg at the reset instant).
    let loading = DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0);
    let evid4_dose = DoseEvent::new(reset_at, 300.0, 1, 0.0, false, 0.0);

    let mut base = subj("1", obs.clone(), vec![loading.clone(), evid4_dose.clone()]);
    base.reset_times = vec![reset_at];
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: decisions.clone(),
        ..Default::default()
    };
    let res = simulate_adaptive(
        &model,
        &population(vec![base]),
        &model.default_params,
        1,
        fixed_bolus,
        &opts,
    )
    .expect("EVID=4 (reset+dose) reaches the adaptive path and passes the verifier");
    assert_eq!(res.ledger.len(), 1, "one controller bolus at t=24");

    // The EVID=4 dose survives its own reset: t=18 reads 300·exp(-0.1·6) ≈ 165, not ~0 (which
    // a reset that also wiped its coincident dose would give).
    let y18 = res
        .trajectories
        .iter()
        .find(|x| (x.time - 18.0).abs() < 1e-12)
        .expect("t=18 row")
        .ipred;
    assert!(
        y18 > 100.0,
        "the EVID=4 dose must land AFTER its coincident reset (t=18 ≈ 165), got {y18}"
    );

    let mut static_doses = vec![loading, evid4_dose];
    static_doses.extend(
        res.ledger
            .iter()
            .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0)),
    );
    let mut static_subject = subj("1", obs.clone(), static_doses);
    static_subject.reset_times = vec![reset_at];
    let preds = predict(
        &model,
        &population(vec![static_subject]),
        &model.default_params,
    )
    .unwrap();
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-6 + 1e-4 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={} (EVID=4 reset+dose)",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
}

#[test]
fn adaptive_ss_base_dose_with_reset_matches_static_predict() {
    // #932 gate: a steady-state base dose (SS=1, II>0) pre-equilibrated at t=0, then wiped by a
    // mid-horizon reset, must still reproduce predict() on the same (SS base ∪ ledger) regimen
    // carrying the reset. Exercises `equilibrate_ss_state` seeding BEFORE the reset zeros it —
    // the reset floor then keeps the equilibrated tail from re-contributing. (If this ever
    // diverged, #932 would narrow to reject SS × reset; it holds, so SS base × reset is in.)
    let model = parse_model_string(ODE_NO_IIV).expect("parse no-IIV ODE model");
    let decisions = vec![0.0, 24.0];
    let obs = vec![1.0, 8.0, 20.0];
    let reset_at = 12.0;
    let ss = DoseEvent::new(0.0, 300.0, 1, 0.0, true, 24.0); // 300 mg q24h at steady state

    let mut base = subj("1", obs.clone(), vec![ss.clone()]);
    base.reset_times = vec![reset_at];
    let opts = AdaptiveSimulateOptions {
        seed: Some(9),
        decision_times: decisions.clone(),
        ..Default::default()
    };
    let res = simulate_adaptive(
        &model,
        &population(vec![base]),
        &model.default_params,
        1,
        fixed_bolus,
        &opts,
    )
    .expect("SS base dose + reset runs and passes the reset-aware verifier");
    assert_eq!(res.ledger.len(), 2, "a controller bolus at each decision");

    let mut static_doses = vec![ss];
    static_doses.extend(
        res.ledger
            .iter()
            .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0)),
    );
    let mut static_subject = subj("1", obs.clone(), static_doses);
    static_subject.reset_times = vec![reset_at];
    let preds = predict(
        &model,
        &population(vec![static_subject]),
        &model.default_params,
    )
    .unwrap();
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-6 + 1e-4 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={} (SS base dose × reset)",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
}

#[test]
fn adaptive_base_regimen_with_reset_under_iov_is_rejected() {
    // #932 scope boundary: base × reset is supported on the constant-covariate path, but under
    // inter-occasion variability (or a time-varying covariate) it stays a typed error — the
    // per-event-PK replay's reset+base composition is not yet oracle-verified, so it must
    // loud-fail rather than risk a silent mis-integration (a #932 follow-up). IOV sets
    // `event_pk`/`eta_occ` = Some, so the driver's `&& tv` guard fires.
    let model = parse_model_string(ODE_IOV).expect("parse IOV ODE model");
    let mut s = subj(
        "1",
        vec![6.0, 30.0],
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
    );
    s.reset_times = vec![12.0];
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: vec![0.0, 24.0],
        ..Default::default()
    };
    let err = simulate_adaptive(
        &model,
        &population(vec![s]),
        &model.default_params,
        1,
        fixed_bolus,
        &opts,
    )
    .expect_err("base × reset under IOV must be rejected");
    assert!(
        err.contains("system resets") && err.contains("constant-covariate path only"),
        "got: {err}"
    );
}

#[test]
fn adaptive_base_regimen_with_reset_under_tv_covariate_is_rejected() {
    // #932 scope boundary (the TV-covariate half, distinct from the IOV half above): base ×
    // reset is supported on the constant-covariate path, but under a TIME-VARYING covariate it
    // stays a typed error — the per-event-PK replay's reset+base composition is not yet oracle-
    // verified. A plain bolus base dose under a TV covariate is otherwise supported (#930), so
    // the reset is what trips the guard. This pins the `&& tv` boundary on the TV-cov path
    // specifically: a regression to `&& iov` would silently ACCEPT this (iov=false, tv=true),
    // routing it through the un-oracle-verified TV replay with every other test still green.
    let model = parse_model_string(ODE_TV_COV).expect("parse TV-cov ODE model");
    let mut s = subj(
        "1",
        vec![6.0, 30.0],
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)], // plain bolus (supported under TV alone)
    );
    s.reset_times = vec![12.0];
    s.covariates = HashMap::from([("CRCL".to_string(), 100.0)]);
    s.obs_covariates = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]),
        HashMap::from([("CRCL".to_string(), 80.0)]),
    ];
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: vec![0.0, 24.0],
        ..Default::default()
    };
    let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect_err("base × reset under a time-varying covariate must be rejected");
    assert!(
        err.contains("system resets") && err.contains("constant-covariate path only"),
        "got: {err}"
    );
}

#[test]
fn adaptive_base_dose_coincident_with_decision_is_observed_pre_dose() {
    // #933: a base dose sharing a time with a decision must be observed PRE-dose (the
    // trough), not post-dose (the peak). A rescue controller fires only when the monitored
    // signal is below a cut that the pre-dose trough clears but the post-dose peak does not,
    // so the realized ledger is the sole witness of which state the controller read. Base
    // 500 mg at t=0 and t=24; the only decision is at t=24. At t=24 the pre-dose trough is
    // 500·exp(-0.1·24) ≈ 45 (< 100 ⇒ rescue), the post-dose peak ≈ 545 (> 100 ⇒ hold).
    // Before the fix the base bolus landed before the hook and no rescue fired.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let decisions = vec![24.0];
    let obs = vec![24.0, 30.0, 54.0];
    let base = vec![
        DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0),
        DoseEvent::new(24.0, 500.0, 1, 0.0, false, 0.0),
    ];
    let pop = population(vec![subj("1", obs.clone(), base.clone())]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(3),
        decision_times: decisions.clone(),
        // Ipred monitor on the central compartment — the pre-dose readout the controller sees.
        monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Ipred)],
        ..Default::default()
    };
    // Rescue 999 mg if the observed central amount is below 100, else hold.
    let make = || {
        move |ctx: &ControllerCtx| {
            if ctx.signal("A").expect("monitor A declared") < 100.0 {
                vec![DoseAction::Bolus { amt: 999.0, cmt: 1 }]
            } else {
                vec![DoseAction::Hold]
            }
        }
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, make, &opts)
        .expect("run passes the base-aware verifier");

    // The witness: the rescue fired, so the controller read the pre-dose trough (≈45 < 100).
    // Post-dose observation (the pre-fix bug) would have read ≈545 > 100 and held (empty ledger).
    assert_eq!(
        res.ledger.len(),
        1,
        "rescue must fire off the pre-dose trough"
    );
    assert_eq!(res.ledger[0].amt, 999.0);
    assert_eq!(res.ledger[0].time, 24.0);

    // Integrity: the trajectory still equals predict() on (base ∪ realized ledger). Decisions
    // sit on the dose grid {0, 24}, so reactive and static segment identically (bit-exact).
    let mut static_doses = base;
    static_doses.extend(
        res.ledger
            .iter()
            .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0)),
    );
    let static_pop = population(vec![subj("1", obs.clone(), static_doses)]);
    let preds = predict(&model, &static_pop, &model.default_params).unwrap();
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-9 + 1e-9 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={}",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
}

#[test]
fn adaptive_base_regimen_controller_dose_before_base_anchors_tafd_globally() {
    // #934: a controller dose scheduled BEFORE the earliest base dose is the true first dose,
    // so TAFD must anchor at it — min(earliest base, first controller) — matching the static
    // frozen-replay verifier's global earliest. ODE_TAFD's RHS reads TAFD, so a stale anchor
    // (= earliest base, t=24) integrates different forcing than the verifier (t=0) and the
    // default-on verifier errors. The controller doses 300 mg at the first decision (t=0),
    // before the base 200 mg at t=24; decisions {0, 24} sit on the dose grid (bit-exact).
    let model = parse_model_string(ODE_TAFD).expect("parse TAFD-reading ODE model");
    let decisions = vec![0.0, 24.0];
    let obs = vec![6.0, 30.0, 48.0];
    let base = vec![DoseEvent::new(24.0, 200.0, 1, 0.0, false, 0.0)];
    let pop = population(vec![subj("1", obs.clone(), base.clone())]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(3),
        decision_times: decisions.clone(),
        ..Default::default()
    };
    // Dose once, at the first decision (t=0), then hold.
    let make = || {
        let mut fired = false;
        move |_ctx: &ControllerCtx| {
            if !fired {
                fired = true;
                vec![DoseAction::Bolus { amt: 300.0, cmt: 1 }]
            } else {
                vec![DoseAction::Hold]
            }
        }
    };
    // verify = true (default): Ok only if the reactive TAFD matches the verifier's global
    // earliest. Without the fix, TAFD stays at 24 (earliest base) and the TAFD-forced
    // trajectory diverges from the replay ⇒ Err.
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, make, &opts).expect(
        "controller-dose-before-base run passes the base-aware verifier (TAFD anchored globally)",
    );
    assert_eq!(res.ledger.len(), 1, "one controller dose at t=0");
    assert_eq!(res.ledger[0].time, 0.0);
    assert!(
        res.trajectories[0].ipred > 0.0,
        "TAFD-forced trajectory is actually integrated"
    );
}

#[test]
fn adaptive_base_dose_after_controller_stop_still_lands() {
    // #702 Finding 4: a pre-scheduled base dose scheduled PAST a controller `Stop` still
    // lands — the base regimen is the patient's standing prescription, independent of the
    // controller (and the frozen-replay verifier replays it too). The controller stops at the
    // first decision (t=0); base doses at t=24 and t=48 must still be integrated, so the run
    // equals predict() on the base regimen alone (the ledger is empty).
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let decisions = vec![0.0, 24.0];
    let obs = vec![12.0, 30.0, 54.0];
    let base = vec![
        DoseEvent::new(24.0, 400.0, 1, 0.0, false, 0.0),
        DoseEvent::new(48.0, 400.0, 1, 0.0, false, 0.0),
    ];
    let pop = population(vec![subj("1", obs.clone(), base.clone())]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: decisions.clone(),
        ..Default::default()
    };
    // Stop immediately at the first decision.
    let make = || move |_ctx: &ControllerCtx| vec![DoseAction::Stop];
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, make, &opts)
        .expect("post-Stop base doses run and pass the verifier");
    assert!(res.ledger.is_empty(), "controller Stopped, issued no doses");

    // The base doses (incl. the two past the Stop) are integrated: equals predict() on base.
    let static_pop = population(vec![subj("1", obs.clone(), base)]);
    let preds = predict(&model, &static_pop, &model.default_params).unwrap();
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-9 + 1e-9 * pred.pred.abs(),
            "adaptive IPRED {} != static predict {} at t={} (post-Stop base dose dropped?)",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
    // Positive control: the t=48 base dose is visible at the t=54 observation (non-vacuous).
    assert!(
        res.trajectories[2].ipred > 100.0,
        "the t=48 post-Stop base dose must be visible at t=54, got {}",
        res.trajectories[2].ipred
    );
}

#[test]
fn dv_monitor_without_error_model_is_rejected() {
    // Edge (a): a DV monitor on a model with no residual error (here sigma is
    // stripped) is a typed error, never a fabricated σ.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let mut params = model.default_params.clone();
    params.sigma.values = vec![]; // no [error_model] coverage for the monitor
    let pop = population(vec![subj("1", vec![6.0], vec![])]);
    let opts = AdaptiveSimulateOptions {
        decision_times: vec![0.0],
        monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Dv)],
        ..Default::default()
    };
    let err = simulate_adaptive(&model, &pop, &params, 1, fixed_bolus, &opts)
        .expect_err("DV monitor with no error model must be rejected");
    assert!(err.contains("error_model"), "got: {err}");
}

#[test]
fn verify_can_be_disabled() {
    // The verifier is the default-on safety net; it can be turned off for a
    // large run once the controller is trusted. Disabling it still produces
    // the same trajectories/ledger (verification observes, it does not alter).
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let pop = population(vec![subj("1", vec![6.0, 30.0], vec![])]);
    let mk = |verify: bool| AdaptiveSimulateOptions {
        seed: Some(3),
        decision_times: vec![0.0, 24.0],
        verify,
        ..Default::default()
    };
    let checked = simulate_adaptive(
        &model,
        &pop,
        &model.default_params,
        1,
        fixed_bolus,
        &mk(true),
    )
    .expect("verified run");
    let unchecked = simulate_adaptive(
        &model,
        &pop,
        &model.default_params,
        1,
        fixed_bolus,
        &mk(false),
    )
    .expect("unverified run");
    assert_eq!(checked.ledger, unchecked.ledger);
    let ip =
        |r: &AdaptiveSimulationResult| r.trajectories.iter().map(|t| t.ipred).collect::<Vec<_>>();
    assert_eq!(ip(&checked), ip(&unchecked));
}

// ----- S1.5: DV-mode (assay-noised) monitors --------------------------

/// Factory (mirrors `fixed_bolus`): titrate on the *measured* (DV) central
/// amount "A" — dose 100 mg when it falls below 50, else hold.
fn dv_threshold() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |ctx: &ControllerCtx| {
        if ctx.signal("A").expect("monitor A declared") < 50.0 {
            vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
        } else {
            vec![DoseAction::Hold]
        }
    }
}

#[test]
fn dv_monitor_run_passes_default_verifier() {
    // End-to-end: a controller titrating on the DV signal, verify = true. Covers
    // the assay-noise wiring (resid_var closure + per-subject base seed) and the
    // driver's DV branch, and confirms the frozen-replay verifier stays green on
    // a DV run (it replays realized doses, not decisions).
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let pop = population(vec![subj("1", vec![2.0, 6.0, 10.0], vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: vec![0.0, 4.0, 8.0],
        monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Dv)],
        ..Default::default()
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, dv_threshold, &opts)
        .expect("DV run passes the verifier");
    assert_eq!(res.decisions.len(), 3);
    // The DV value the controller saw is recorded, tagged as the Dv mode.
    assert!(res
        .decisions
        .iter()
        .all(|d| d.observed_signals[0].mode == ObserveMode::Dv));
}

#[test]
fn dv_monitor_collapses_to_ipred_as_sigma_to_zero() {
    // sigma -> 0: titrating on the DV reproduces titrating on the latent IPRED,
    // realized-ledger for realized-ledger.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let pop = population(vec![subj("1", vec![2.0, 6.0, 10.0], vec![])]);
    let mut params = model.default_params.clone();
    params.sigma.values = vec![1e-12];
    let decision_times = vec![0.0, 4.0, 8.0];

    let dv_opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: decision_times.clone(),
        monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Dv)],
        ..Default::default()
    };
    let ipred_opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times,
        monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Ipred)],
        ..Default::default()
    };
    let dv = simulate_adaptive(&model, &pop, &params, 1, dv_threshold, &dv_opts).expect("dv");
    let ip = simulate_adaptive(&model, &pop, &params, 1, dv_threshold, &ipred_opts).expect("ipred");
    // Compare the *dosing decisions* (time/amt/cmt/rate), not the recorded
    // observed signals: the residual variance floors at MIN_VARIANCE, so the DV
    // readout never collapses to IPRED bit-for-bit — but the decisions it drives
    // do. (The exact bit-for-bit collapse is pinned at the driver level with a
    // literal zero-variance resolver.)
    let schedule = |r: &AdaptiveSimulationResult| {
        r.ledger
            .iter()
            .map(|e| (e.time, e.amt, e.cmt, e.rate))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        schedule(&dv),
        schedule(&ip),
        "DV with sigma→0 must make the same dosing decisions as IPRED"
    );
}

#[test]
fn adding_dv_monitor_does_not_perturb_eta_trajectory() {
    // Isolation: the controller-assay substream is disjoint from the η stream,
    // so enabling a DV monitor leaves the η draws (hence the IPRED trajectory)
    // bit-identical. Uses ODE_IIV (η on CL) and a signal-independent controller,
    // so any trajectory difference could only come from a shifted η draw.
    let model = parse_model_string(ODE_IIV).expect("parse");
    let pop = population(vec![
        subj("1", vec![6.0, 30.0], vec![]),
        subj("2", vec![6.0, 30.0], vec![]),
    ]);
    let opts = |monitors: Vec<MonitorSpec>| AdaptiveSimulateOptions {
        seed: Some(3),
        decision_times: vec![0.0, 24.0],
        monitors,
        ..Default::default()
    };
    let no_mon = simulate_adaptive(
        &model,
        &pop,
        &model.default_params,
        3,
        fixed_bolus,
        &opts(vec![]),
    )
    .expect("no monitor");
    let with_dv = simulate_adaptive(
        &model,
        &pop,
        &model.default_params,
        3,
        fixed_bolus,
        &opts(vec![MonitorSpec::new("A", 1, ObserveMode::Dv)]),
    )
    .expect("dv monitor");
    let ip =
        |r: &AdaptiveSimulationResult| r.trajectories.iter().map(|t| t.ipred).collect::<Vec<_>>();
    assert_eq!(
        ip(&no_mon),
        ip(&with_dv),
        "a DV monitor must not shift the η draws / IPRED trajectory"
    );
}

#[test]
fn dv_assay_draws_are_permutation_invariant() {
    // The controller-assay substream is keyed by subject id, so a subject's DV
    // draws — and thus its decisions — do not depend on iteration order. A large
    // residual CV makes the noise flip decisions (so a position-keyed stream
    // *would* change the ledger), and ODE_NO_IIV keeps η out of the picture.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let mut params = model.default_params.clone();
    params.sigma.values = vec![0.3]; // 30% proportional CV: noise is consequential
    let opts = AdaptiveSimulateOptions {
        seed: Some(11),
        decision_times: vec![0.0, 4.0, 8.0],
        monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Dv)],
        ..Default::default()
    };
    let a = subj("A", vec![2.0, 6.0, 10.0], vec![]);
    let b = subj("B", vec![2.0, 6.0, 10.0], vec![]);
    let pop_ab = population(vec![a.clone(), b.clone()]);
    let pop_ba = population(vec![b, a]);
    let r_ab = simulate_adaptive(&model, &pop_ab, &params, 1, dv_threshold, &opts).expect("ab");
    let r_ba = simulate_adaptive(&model, &pop_ba, &params, 1, dv_threshold, &opts).expect("ba");

    let ledger_for = |r: &AdaptiveSimulationResult, id: &str| {
        r.ledger
            .iter()
            .filter(|e| e.subject == id)
            .map(|e| (e.time, e.amt))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        ledger_for(&r_ab, "A"),
        ledger_for(&r_ba, "A"),
        "A's DV-driven doses must be order-independent"
    );
    assert_eq!(
        ledger_for(&r_ab, "B"),
        ledger_for(&r_ba, "B"),
        "B's DV-driven doses must be order-independent"
    );

    // Non-vacuity: the assay noise must actually be id-keyed (not a no-op), or
    // the invariance above could hold even under a position-keyed stream. At
    // t=4 the trough is ~67 (non-zero, so proportional noise is live), and A and
    // B observe *different* measured values — confirming the noise is consequential.
    let dv_at = |r: &AdaptiveSimulationResult, id: &str, didx: usize| {
        r.decisions
            .iter()
            .find(|d| d.subject == id && d.decision_idx == didx)
            .map(|d| d.observed_signals[0].value)
            .expect("decision logged")
    };
    assert_ne!(
        dv_at(&r_ab, "A", 1),
        dv_at(&r_ab, "B", 1),
        "id-keyed assay noise must differ between subjects (test would be vacuous otherwise)"
    );
}

// ── S2.3: declarative [adaptive_dosing] entry — simulate_adaptive_from_spec ──

// The ODE_NO_IIV structural core plus a declarative block whose lone rule can
// never fire (`signal < 0` on a non-negative amount): the controller re-issues
// `start_dose` at every decision, i.e. a fixed 100 mg q24 regimen.
const SPEC_DEGENERATE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central
  at = [0, 24, 48]
  start_dose = 100
  route = bolus(cmt=1)
  dose_bounds = [0, 1000]
  when signal < 0 : increase 25%
"#;

// Same structural core; a two-sided band titration that walks the maintenance
// dose up until the pre-dose trough settles inside [8, 13].
const SPEC_TITRATE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central
  at = every 24 from 0 to 336
  start_dose = 20
  route = bolus(cmt=1)
  dose_bounds = [0, 1000]
  when signal < 8 : increase 25%
  when signal > 13 : decrease 25%
"#;

// Same structural core, but the controller's `observe` expression is the `TIME`
// built-in itself (#1028). `[adaptive_dosing] observe` compiles through the same
// `build_y_output_fn` as a `[scaling]` Form C readout, so it reads the model-time
// thread-local — and the reactive driver calls the compiled closure directly rather
// than through `OdeReadout::eval`, so it needs its own guard. Decisions at 0/24/48
// with a `signal < 30` rule therefore fire at t=0 and t=24 and *not* at t=48. Had
// the closure read a stale `TIME = 0` the rule would fire at all three, so the dose
// ladder below discriminates the two outright.
const SPEC_TIME_OBSERVE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = TIME
  at = [0, 24, 48]
  start_dose = 100
  route = bolus(cmt=1)
  dose_bounds = [0, 1000]
  when signal < 30 : increase 25%
"#;

/// A minimal valid titration spec (built directly, no file) for the rejection
/// tests, which never reach the run loop.
fn simple_titration_spec() -> AdaptiveDosingSpec {
    AdaptiveDosingSpec {
        observe: Some("central".to_string()),
        observe_declared_covariates: Vec::new(),
        with_assay_error: false,
        assay_cmt: None,
        at: vec![0.0, 24.0],
        start_dose: 100.0,
        route: AdaptiveRoute::Bolus { cmt: 1 },
        dose_bounds: (0.0, 400.0),
        confirm: 1,
        levels: None,
        target_window: None,
        auc_target: None,
        rules: vec![AdaptiveRule {
            op: Comparison::Lt,
            threshold: 50.0,
            action: AdaptiveAction::Increase(DoseStep::Percent(25.0)),
        }],
    }
}

#[test]
fn from_spec_degenerate_oracle_matches_static_predict() {
    // The declarative path's degenerate oracle: a block that never titrates
    // re-issues a fixed 100 mg bolus at every decision, so it must reproduce the
    // static engine on that same realized schedule. A dose lands at every
    // decision and the last obs (54) is the global max, and the frozen-replay
    // verifier (on by default) also passes. Like its programmatic twin above, the
    // assertion is a 1e-9 relative band, not bit equality (#1151).
    let parsed = parse_full_model(SPEC_DEGENERATE).expect("parse model + block");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("[adaptive_dosing] present");
    let obs = vec![6.0, 30.0, 54.0];
    let pop = population(vec![subj("1", obs.clone(), vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default()
    };
    let res = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect("declarative adaptive sim runs");

    assert_eq!(res.ledger.len(), 3, "100 mg re-issued at every decision");
    assert!(
        res.ledger.iter().all(|e| e.amt == 100.0),
        "fixed start_dose, never titrated"
    );
    // No `when` rule ever fires here, so every dose is a re-issue: `rule_fired`
    // records the route, never a fabricated rule (#391 S2 traceability).
    assert!(
        res.ledger.iter().all(|e| e.rule_fired == "bolus"),
        "a re-issue records the dose by its route, not a rule"
    );

    // Static reference: the same fixed schedule pre-scheduled through predict().
    let static_doses: Vec<DoseEvent> = [0.0, 24.0, 48.0]
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let static_pop = population(vec![subj("1", obs.clone(), static_doses)]);
    let preds = predict(&parsed.model, &static_pop, &parsed.model.default_params).unwrap();
    assert_eq!(res.trajectories.len(), preds.len());
    for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - pred.pred).abs() <= 1e-9 + 1e-9 * pred.pred.abs(),
            "declarative IPRED {} != static predict {} at t={}",
            traj.ipred,
            pred.pred,
            traj.time
        );
    }
}

// A covariate-dependent 1-cpt model (CL scales with CRCL) used by the #700
// time-varying-covariate end-to-end tests. No BSV in effect (ETA_CL declared
// but unreferenced), so the covariate is the only source of PK variation.
const SPEC_TV_COV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * CRCL / 100.0
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central
  at = every 24 from 0 to 96
  start_dose = 100
  route = bolus(cmt=1)
  dose_bounds = [0, 1000]
  when signal < 8 : increase 25%
"#;

fn tv_cov_pop(crcls: &[f64], obs: &[f64]) -> Population {
    let mut s = subj("1", obs.to_vec(), vec![]);
    s.covariates = HashMap::from([("CRCL".to_string(), crcls[0])]);
    s.obs_covariates = crcls
        .iter()
        .map(|&c| HashMap::from([("CRCL".to_string(), c)]))
        .collect();
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];
    pop
}

// IOV declarative spec (#701): a per-occasion κ on CL, no covariate (the parser
// requires an omega, so a near-degenerate BSV η rides along).
const SPEC_IOV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  kappa KAPPA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central
  at = every 24 from 0 to 96
  start_dose = 100
  route = bolus(cmt=1)
  dose_bounds = [0, 1000]
  when signal < 8 : increase 25%
"#;

// #700 × #701 co-occurrence: CL depends on BOTH a time-varying covariate (CRCL)
// and a per-occasion κ.
const SPEC_TV_IOV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  kappa KAPPA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * (CRCL / 100.0) * exp(KAPPA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central
  at = every 24 from 0 to 96
  start_dose = 100
  route = bolus(cmt=1)
  dose_bounds = [0, 1000]
  when signal < 8 : increase 25%
"#;

#[test]
fn from_spec_supports_time_varying_covariate() {
    // End-to-end #700: a covariate-dependent CL (CRCL declining — e.g. renal
    // decline) drives per-event PK through the declarative path, and the
    // default-on frozen-replay verifier validates the whole run. A constant-CRCL
    // subject yields a different trajectory, proving the covariate is consumed.
    let parsed = parse_full_model(SPEC_TV_COV).expect("model + block parse");
    let spec = parsed.adaptive_dosing.as_ref().expect("[adaptive_dosing]");
    let obs = vec![0.0, 24.0, 48.0, 72.0, 96.0];
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default()
    };

    // TV run: the verifier (on by default) passing is itself the bookkeeping
    // check for the per-event path.
    let tv_pop = tv_cov_pop(&[100.0, 80.0, 60.0, 45.0, 35.0], &obs);
    let tv = simulate_adaptive_from_spec(
        &parsed.model,
        &tv_pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect("TV adaptive sim runs + verifies");

    let const_pop = tv_cov_pop(&[100.0, 100.0, 100.0, 100.0, 100.0], &obs);
    let cst = simulate_adaptive_from_spec(
        &parsed.model,
        &const_pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect("constant adaptive sim runs");

    assert!(!tv.ledger.is_empty());
    // Declining CRCL lowers CL late ⇒ slower clearance ⇒ a materially different
    // late trajectory than the constant-CRCL subject.
    let tv_last = tv.trajectories.last().unwrap().ipred;
    let cst_last = cst.trajectories.last().unwrap().ipred;
    assert!(
        (tv_last - cst_last).abs() > 1e-6 * tv_last.abs().max(1.0),
        "time-varying covariate must change the trajectory: tv={tv_last}, const={cst_last}"
    );
}

#[test]
fn from_spec_supports_time_varying_covariate_with_infusion() {
    // #700 infusion coverage: an infusion route under a declining covariate
    // exercises the injected-infusion F (captured at injection) and the
    // infusion-end break in BOTH the reactive driver and the frozen-replay
    // engine (`adaptive_frozen_replay_tv`). The default-on verifier passing
    // confirms the per-event bookkeeping for infusions, not just boluses.
    let mut parsed = parse_full_model(SPEC_TV_COV).expect("model + block parse");
    parsed
        .adaptive_dosing
        .as_mut()
        .expect("[adaptive_dosing]")
        .route = AdaptiveRoute::Infuse { cmt: 1, over: 2.0 };
    let spec = parsed.adaptive_dosing.as_ref().unwrap();
    let obs = vec![0.0, 24.0, 48.0, 72.0, 96.0];
    let pop = tv_cov_pop(&[100.0, 80.0, 60.0, 45.0, 35.0], &obs);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default()
    };
    let res = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect("TV infusion adaptive sim runs + verifies");
    assert!(!res.ledger.is_empty());
    assert!(
        res.ledger.iter().all(|e| e.rate > 0.0),
        "infusion route must record a positive rate on every realized dose"
    );
}

#[test]
fn adaptive_auc_target_rejects_time_varying_covariate() {
    // #700: the exposure metric (`auc_target_attainment`) can't yet be computed
    // per-event, so declaring `auc_target` on a time-varying-covariate subject is
    // a typed error rather than a silently frozen-PK AUC.
    let mut parsed = parse_full_model(SPEC_TV_COV).expect("model + block parse");
    // Attach an `auc_target` to the parsed spec (the model string above omits it
    // so the support test above stays clean).
    parsed
        .adaptive_dosing
        .as_mut()
        .expect("[adaptive_dosing]")
        .auc_target = Some((400.0, 600.0));
    let spec = parsed.adaptive_dosing.as_ref().unwrap();
    let obs = vec![0.0, 24.0, 48.0, 72.0, 96.0];
    let pop = tv_cov_pop(&[100.0, 80.0, 60.0, 45.0, 35.0], &obs);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default()
    };

    let err = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect_err("auc_target on a TV subject must be rejected");
    assert!(
        err.to_lowercase().contains("auc_target") && err.to_lowercase().contains("time-varying"),
        "error should cite auc_target + time-varying: {err}"
    );
}

#[test]
fn adaptive_rejects_malformed_absorption_fractions() {
    // #721: `run_adaptive_population` now runs the same built-in-absorption
    // dose-precondition guard (#588) that `simulate()` / `predict()` / `fit()`
    // enforce. This parallel-first-order model's pathway fractions sum to 1.2
    // (FR1 = FR2 = 0.6), not 1 — the input rate would deliver 1.2× the dose. The
    // fraction check evaluates typical parameters (η = 0) at each dose record, where
    // the engine reads a pathway fraction (#1569), so it fires on the base regimen's
    // loading dose at the chokepoint, before any decision — the same typed error the
    // static paths raise, instead of a silent 1.2× run. (A dose-free base has no
    // record the fraction is read at; the controller cannot dose an input-rate
    // compartment either, so it is rejected by that guard instead.)
    const SPEC: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVKA1(1.5, 0.05, 24.0)
  theta TVKA2(0.3, 0.01, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  FR1 = TVFR1
  FR2 = TVFR1
  KA1 = TVKA1
  KA2 = TVKA2
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[adaptive_dosing]
  observe = central
  at = [0, 24, 48]
  start_dose = 100
  route = bolus(cmt=1)
  dose_bounds = [0, 1000]
  when signal < 5 : increase 25%
"#;
    let parsed = parse_full_model(SPEC).expect("malformed-fraction model still parses");
    let spec = parsed.adaptive_dosing.as_ref().expect("[adaptive_dosing]");
    let loading = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let pop = population(vec![subj("1", vec![0.0, 24.0, 48.0], vec![loading])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default()
    };
    let err = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect_err("malformed absorption fractions must be rejected on the adaptive path");
    assert!(
        err.to_lowercase().contains("fraction"),
        "expected an absorption-fraction error, got: {err}"
    );
}

#[test]
fn from_spec_supports_time_varying_covariate_with_iov() {
    // #700 × #701 co-occurrence: CL depends on BOTH a declining covariate (CRCL,
    // #700 LOCF) and a per-occasion κ (#701 decision window). Both fold into the
    // per-event PK in the reactive driver AND the frozen-replay engine, so the
    // default-on verifier passing is the bit-exact proof they compose. The run is
    // reproducible for a fixed seed and κ-sensitive across seeds.
    let parsed = parse_full_model(SPEC_TV_IOV).expect("model + block parse");
    assert!(parsed.model.n_kappa == 1);
    let spec = parsed.adaptive_dosing.as_ref().expect("[adaptive_dosing]");
    let obs = vec![0.0, 24.0, 48.0, 72.0, 96.0];
    let pop = tv_cov_pop(&[100.0, 80.0, 60.0, 45.0, 35.0], &obs);
    let run = |seed: u64| {
        let opts = AdaptiveSimulateOptions {
            seed: Some(seed),
            ..Default::default()
        };
        simulate_adaptive_from_spec(
            &parsed.model,
            &pop,
            &parsed.model.default_params,
            1,
            spec,
            &opts,
        )
        .expect("TV+IOV adaptive sim runs + verifies")
    };
    let a = run(1);
    assert!(!a.ledger.is_empty());
    let a_last = a.trajectories.last().unwrap().ipred;
    // Reproducible for a fixed seed (κ + η both seeded).
    assert_eq!(
        a_last,
        run(1).trajectories.last().unwrap().ipred,
        "same seed ⇒ identical trajectory"
    );
    // κ genuinely perturbs the trajectory across seeds.
    assert!(
        (a_last - run(999).trajectories.last().unwrap().ipred).abs() > 1e-9,
        "different seed ⇒ different κ ⇒ different trajectory"
    );
}

#[test]
fn adaptive_auc_target_rejects_iov() {
    // #701: like the TV-covariate case, the exposure metric can't yet be computed
    // per-occasion, so declaring `auc_target` on an IOV model is a typed error
    // rather than a silently κ-frozen AUC.
    let mut parsed = parse_full_model(SPEC_IOV).expect("model + block parse");
    parsed
        .adaptive_dosing
        .as_mut()
        .expect("[adaptive_dosing]")
        .auc_target = Some((400.0, 600.0));
    let spec = parsed.adaptive_dosing.as_ref().unwrap();
    let obs = vec![0.0, 24.0, 48.0, 72.0, 96.0];
    let pop = population(vec![subj("1", obs, vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default()
    };
    let err = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect_err("auc_target on an IOV model must be rejected");
    assert!(
        err.to_lowercase().contains("auc_target") && err.to_lowercase().contains("iov"),
        "error should cite auc_target + IOV: {err}"
    );
}

#[test]
fn adaptive_auc_target_rejects_reset() {
    // #716: system resets are now honored by the driver and the frozen-replay
    // verifier, but the exposure metric (`auc_target_attainment`) integrates a dense
    // grid that does NOT apply resets, so declaring `auc_target` on a reset subject is
    // a typed error rather than a silently un-reset AUC. The model is a plain ODE
    // (no TV covariate, no IOV), so only the reset can trip this guard — pinning the
    // reset branch specifically.
    let mut parsed = parse_full_model(SPEC_DEGENERATE).expect("model + block parse");
    parsed
        .adaptive_dosing
        .as_mut()
        .expect("[adaptive_dosing]")
        .auc_target = Some((400.0, 600.0));
    let spec = parsed.adaptive_dosing.as_ref().unwrap();
    let mut subject = subj("1", vec![24.0, 48.0], vec![]);
    subject.reset_times = vec![36.0];
    let pop = population(vec![subject]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default()
    };
    let err = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect_err("auc_target on a reset subject must be rejected");
    assert!(
        err.to_lowercase().contains("auc_target") && err.to_lowercase().contains("reset"),
        "error should cite auc_target + reset: {err}"
    );
}

#[test]
fn adaptive_iov_decision_pk_uses_locf_covariate_off_grid() {
    // #701 review regression (verifier-blind): on a TV-covariate + IOV model, an
    // off-record decision must resolve its PK snapshot (`decision_pk[g]`) at the
    // LOCF covariate — the most-recent record carried forward — exactly as the
    // driver's live `decision_cov` does, NOT the frozen t=0 baseline. Before the
    // fix, `run_adaptive_population` fell back to `subject.covariates` (t=0) for a
    // decision off the obs grid, re-introducing the #700 "decision covariate frozen
    // at t=0" defect on the IOV decision-PK path — and the frozen-replay verifier,
    // which reuses the same `decision_pk`, could not catch it.
    //
    // F (bioavailability) depends on CRCL and never enters the ODE, so each injected
    // dose's realized `f_applied` is an *unconfounded* readout of the covariate the
    // decision saw: F = CRCL/200 ⇒ 0.5 at CRCL=100, 1.0 at CRCL=200.
    let iov_tvf = r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  kappa KAPPA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
  f  = CRCL / 200.0
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;
    let model = parse_model_string(iov_tvf).expect("parse TV-F + IOV model");
    assert!(model.n_kappa == 1, "must be an IOV model");

    // obs at 0 (CRCL 100) and 24 (CRCL 200): the covariate jumps at t=24.
    let mut s = subj("1", vec![0.0, 24.0], vec![]);
    s.covariates = HashMap::from([("CRCL".to_string(), 100.0)]);
    s.obs_covariates = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]),
        HashMap::from([("CRCL".to_string(), 200.0)]),
    ];
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];

    // Decisions at 0, 12, 24, 36. t=12 is off-record but *before* the jump
    // (LOCF == baseline == 100, a control); t=36 is off-record and *after* the jump
    // (LOCF = 200, baseline = 100) — the discriminator. A fixed bolus doses at each.
    let decisions = vec![0.0, 12.0, 24.0, 36.0];
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: decisions.clone(),
        ..Default::default()
    };
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("adaptive TV-F + IOV sim runs and passes the default verifier");
    assert_eq!(res.ledger.len(), 4, "a dose at every decision");

    let f_at = |t: f64| {
        res.ledger
            .iter()
            .find(|e| e.time == t)
            .map(|e| e.f_applied)
            .unwrap_or_else(|| panic!("no dose at t={t}"))
    };
    // t=0 coincides with obs CRCL=100 → F=0.5; t=24 coincides with obs CRCL=200 → F=1.0.
    assert!((f_at(0.0) - 0.5).abs() < 1e-9, "t=0 F={}", f_at(0.0));
    assert!((f_at(24.0) - 1.0).abs() < 1e-9, "t=24 F={}", f_at(24.0));
    // t=12 off-record but before the jump → LOCF == baseline == 100 → F=0.5 (the
    // buggy and fixed code agree here; the control).
    assert!((f_at(12.0) - 0.5).abs() < 1e-9, "t=12 F={}", f_at(12.0));
    // t=36 off-record and AFTER the jump → LOCF CRCL=200 → F=1.0. The bug froze it
    // at the t=0 baseline CRCL=100 → F=0.5. This assertion fails without the fix.
    assert!(
        (f_at(36.0) - 1.0).abs() < 1e-9,
        "off-record decision at t=36 must use the LOCF covariate (CRCL=200 ⇒ F=1.0), \
             not the frozen t=0 baseline (CRCL=100 ⇒ F=0.5); got F={}",
        f_at(36.0)
    );
}

#[test]
fn adaptive_rejects_non_ascending_or_duplicate_decision_times() {
    // #701 review: the occasion bookkeeping (`occasion_of`, `decision_index_of`)
    // assumes a strictly-increasing decision schedule. An unsorted or duplicated
    // `decision_times` would silently mis-map a record's occasion κ — and the
    // frozen-replay verifier, reusing the same occasion arrays, could not catch it.
    // The programmatic `simulate_adaptive` path must reject it loudly at the shared
    // funnel, matching the declarative path's `validate_increasing_finite`.
    let model = parse_model_string(ODE_IOV).expect("parse IOV model");
    let pop = population(vec![subj("1", vec![0.0, 24.0, 48.0], vec![])]);
    let run = |dts: Vec<f64>| {
        let opts = AdaptiveSimulateOptions {
            seed: Some(1),
            decision_times: dts,
            ..Default::default()
        };
        simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
    };
    // Out of order.
    let err = run(vec![0.0, 48.0, 24.0]).expect_err("unsorted schedule must be rejected");
    assert!(
        err.to_lowercase().contains("increasing"),
        "error should cite the ordering: {err}"
    );
    // Duplicate time (strictly-increasing rejects equality).
    let err = run(vec![0.0, 24.0, 24.0, 48.0]).expect_err("duplicate time must be rejected");
    assert!(
        err.to_lowercase().contains("increasing"),
        "error should cite the ordering: {err}"
    );
    // The ascending control still runs.
    assert!(
        run(vec![0.0, 24.0, 48.0]).is_ok(),
        "ascending schedule runs"
    );
}

#[test]
fn from_spec_supports_time_dependent_pk() {
    // #700 also fixes a latent bug: a `TIME`-dependent PK parameter (no
    // covariate) was silently frozen at TIME=0 on the adaptive path. Now
    // `model_uses_time_builtin` routes it through per-event PK, so it verifies
    // and differs from a constant (TIME=0) baseline model. Same structural core;
    // only CL's TIME dependence differs.
    const TIME_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(0.01 * TIME)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central
  at = every 24 from 0 to 96
  start_dose = 100
  route = bolus(cmt=1)
  dose_bounds = [0, 1000]
  when signal < 8 : increase 25%
"#;
    const CONST_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central
  at = every 24 from 0 to 96
  start_dose = 100
  route = bolus(cmt=1)
  dose_bounds = [0, 1000]
  when signal < 8 : increase 25%
"#;
    // Observations offset from the decision grid (0,24,…,96): the constant
    // baseline below runs on the shipped single-snapshot verifier, which has a
    // pre-existing gap when a dose lands exactly at the last break *and* an obs
    // coincides with it (tracked separately). Offsetting keeps this test about
    // TIME, not that edge; the TV support test above pins the last-break edge on
    // the per-event path.
    let obs = vec![6.0, 30.0, 54.0, 78.0, 90.0];
    let pop = population(vec![subj("1", obs, vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default()
    };

    let tp = parse_full_model(TIME_MODEL).expect("TIME model parses");
    assert!(
        crate::pk::model_uses_time_builtin(&tp.model),
        "the TIME model must be detected as time-dependent"
    );
    let time_res = simulate_adaptive_from_spec(
        &tp.model,
        &pop,
        &tp.model.default_params,
        1,
        tp.adaptive_dosing.as_ref().unwrap(),
        &opts,
    )
    .expect("TIME-in-PK sim runs + verifies");

    let cp = parse_full_model(CONST_MODEL).expect("const model parses");
    let const_res = simulate_adaptive_from_spec(
        &cp.model,
        &pop,
        &cp.model.default_params,
        1,
        cp.adaptive_dosing.as_ref().unwrap(),
        &opts,
    )
    .expect("constant sim runs");

    let t_last = time_res.trajectories.last().unwrap().ipred;
    let c_last = const_res.trajectories.last().unwrap().ipred;
    assert!(
        (t_last - c_last).abs() > 1e-6 * t_last.abs().max(1.0),
        "TIME-dependent PK must differ from the TIME=0 baseline: time={t_last}, const={c_last}"
    );
}

#[test]
fn from_spec_three_witnesses_equals_handwritten_closure() {
    // Witness 1: the declarative block. Witness 2: a hand-written controller
    // closure with the identical ladder. Witness 3: the realized ledger they
    // must share, byte-for-byte. `observe = central` reads the same state the
    // programmatic monitor's cmt readout returns (model readout y = central), so
    // both controllers see the same signal and decide identically.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let at = vec![0.0, 24.0, 48.0, 72.0];
    let obs = vec![12.0, 36.0, 78.0];
    let pop = population(vec![subj("S", obs.clone(), vec![])]);

    let spec = AdaptiveDosingSpec {
        observe: Some("central".to_string()),
        observe_declared_covariates: Vec::new(),
        with_assay_error: false,
        assay_cmt: None,
        at: at.clone(),
        start_dose: 100.0,
        route: AdaptiveRoute::Bolus { cmt: 1 },
        dose_bounds: (0.0, 1000.0),
        confirm: 1,
        levels: None,
        target_window: None,
        auc_target: None,
        rules: vec![AdaptiveRule {
            op: Comparison::Lt,
            threshold: 50.0,
            action: AdaptiveAction::Increase(DoseStep::Percent(25.0)),
        }],
    };
    let spec_opts = AdaptiveSimulateOptions {
        seed: Some(7),
        ..Default::default()
    };
    let from_spec =
        simulate_adaptive_from_spec(&model, &pop, &model.default_params, 1, &spec, &spec_opts)
            .expect("declarative run");

    // The same ladder, hand-written: bump the running dose 25% (clamped) when
    // the signal is below 50, else re-issue it.
    let make = || {
        let mut dose = 100.0_f64;
        move |ctx: &ControllerCtx| {
            if ctx.signal("signal").expect("signal monitor") < 50.0 {
                dose = (dose * 1.25).clamp(0.0, 1000.0);
            }
            vec![DoseAction::Bolus { amt: dose, cmt: 1 }]
        }
    };
    let prog_opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: at.clone(),
        monitors: vec![MonitorSpec::new("signal", 1, ObserveMode::Ipred)],
        ..Default::default()
    };
    let programmatic = simulate_adaptive(&model, &pop, &model.default_params, 1, make, &prog_opts)
        .expect("programmatic run");

    assert!(
        !from_spec.ledger.is_empty(),
        "the ladder must fire and dose"
    );
    // Identical *dosing* — every realized-dose field matches bit-for-bit. The
    // one field that legitimately differs is the audit-only `rule_fired`: the
    // declarative path now names the rung that fired, while a hand-written
    // closure has no rule to name (asserted separately below).
    assert_eq!(from_spec.ledger.len(), programmatic.ledger.len());
    for (d, p) in from_spec.ledger.iter().zip(&programmatic.ledger) {
        assert_eq!(
            (
                &d.subject,
                d.draw,
                d.sim,
                d.dose_idx,
                d.time,
                d.amt,
                d.cmt,
                d.rate,
                d.decision_idx,
                &d.observed_signals,
                d.f_applied,
            ),
            (
                &p.subject,
                p.draw,
                p.sim,
                p.dose_idx,
                p.time,
                p.amt,
                p.cmt,
                p.rate,
                p.decision_idx,
                &p.observed_signals,
                p.f_applied,
            ),
            "declarative block and hand-written closure must realize the identical dosing"
        );
    }
    // Traceability (#391 S2): the declarative ledger names the rung that fired;
    // the programmatic ledger (arbitrary controller, no rules) records the
    // dose by its route.
    assert!(
        from_spec
            .ledger
            .iter()
            .all(|e| e.rule_fired == "signal < 50 : increase 25%"),
        "declarative rule_fired names the rung that fired: {:?}",
        from_spec
            .ledger
            .iter()
            .map(|e| e.rule_fired.as_str())
            .collect::<Vec<_>>()
    );
    assert!(
        programmatic.ledger.iter().all(|e| e.rule_fired == "bolus"),
        "programmatic rule_fired records the route, not a rule"
    );
    assert_eq!(
        from_spec.decisions, programmatic.decisions,
        "and the identical decision log"
    );
    let ip =
        |r: &AdaptiveSimulationResult| r.trajectories.iter().map(|t| t.ipred).collect::<Vec<_>>();
    assert_eq!(
        ip(&from_spec),
        ip(&programmatic),
        "and identical trajectories"
    );
}

#[test]
fn from_spec_closed_loop_converges_into_target_band() {
    // Closed-loop convergence: the trough (pre-dose central amount) starts far
    // below the [8, 13] band; the `increase 25%` rule walks the maintenance dose
    // up until the trough settles inside the band, after which no rule fires and
    // the dose is re-issued unchanged. k = 0.1/h with τ = 24 h washes out ~91%
    // each interval, so the trough tracks ~0.1·dose and the loop is stable.
    let parsed = parse_full_model(SPEC_TITRATE).expect("parse titrating block");
    let spec = parsed.adaptive_dosing.as_ref().expect("block present");
    // One observation after the last decision (336) so the schedule's last time
    // is the global max; the convergence assertions read the decision log.
    let pop = population(vec![subj("P", vec![342.0], vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(3),
        ..Default::default()
    };
    let res = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect("titration runs and the verifier passes");

    let signal_at = |d: usize| res.decisions[d].observed_signals[0].value;
    let n = res.decisions.len();
    assert_eq!(n, 15, "every-24h-from-0-to-336 ⇒ 15 decisions");

    // Started below the band ...
    assert!(
        signal_at(1) < 8.0,
        "decision 1 trough {} should start below the band",
        signal_at(1)
    );
    // ... and converged into it: the last three troughs sit inside [8, 13].
    for d in (n - 3)..n {
        let s = signal_at(d);
        assert!(
            (8.0..=13.0).contains(&s),
            "decision {d} trough {s} should be inside the target band [8, 13]"
        );
    }
    // Converged ⇒ the maintenance dose has settled (the loop holds it steady).
    let last = res.ledger.last().expect("doses issued");
    let prev = &res.ledger[res.ledger.len() - 2];
    assert_eq!(
        last.amt, prev.amt,
        "the maintenance dose should be steady once the trough is in band"
    );
}

#[test]
fn from_spec_observe_reads_the_time_builtin_at_the_decision() {
    // #1028: `observe = TIME` must see each decision's own time. The compiled
    // `observe` closure resolves the `TIME` built-in from the model-time
    // thread-local, and the reactive driver evaluates it directly (not through
    // `OdeReadout::eval`), so the driver enters the guard itself.
    let parsed = parse_full_model(SPEC_TIME_OBSERVE).expect("observe = TIME parses");
    let spec = parsed.adaptive_dosing.as_ref().expect("block present");
    let pop = population(vec![subj("P", vec![60.0], vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(3),
        ..Default::default()
    };
    let res = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect("the TIME-driven controller runs");

    // The observed signal IS the decision time.
    let times = [0.0, 24.0, 48.0];
    assert_eq!(res.decisions.len(), 3, "one decision per `at` entry");
    for (d, &t) in res.decisions.iter().zip(times.iter()) {
        approx::assert_relative_eq!(d.observed_signals[0].value, t, epsilon = 1e-9);
    }

    // …and the rule (`signal < 30 : increase 25%`) therefore fires at t=0 and t=24
    // but not at t=48: 100 → 125 → 156.25 → held. A stale `TIME = 0` would have
    // fired all three (100 → 125 → 156.25 → 195.3125).
    let amts: Vec<f64> = res.ledger.iter().map(|d| d.amt).collect();
    assert_eq!(amts.len(), 3, "one dose per decision");
    approx::assert_relative_eq!(amts[0], 125.0, max_relative = 1e-9);
    approx::assert_relative_eq!(amts[1], 156.25, max_relative = 1e-9);
    approx::assert_relative_eq!(amts[2], 156.25, max_relative = 1e-9);
}

#[test]
fn from_spec_observe_t_alias_loses_to_a_declared_covariate() {
    // `observe` compiles through the same `build_y_output_fn` as a `[scaling]` Form C
    // readout, so it must apply the same `T` / `t` name precedence — otherwise the
    // *same* model reads a declared `T` as the data column in `[scaling]` and as the
    // model-time built-in in `observe` (#1042 review of #1028). The declarations reach
    // the simulate-time compiler via `AdaptiveDosingSpec::observe_declared_covariates`,
    // captured at parse time.
    //
    // `T` is declared but referenced *only* by `observe`, which is exactly the case a
    // `referenced_covariates`-based precedence list would have missed.
    let src = SPEC_TIME_OBSERVE.replace("observe = TIME", "observe = T")
        + "[covariates]\n  T continuous\n";
    let parsed = parse_full_model(&src).expect("a declared T covariate parses");
    let spec = parsed.adaptive_dosing.as_ref().expect("block present");
    assert!(
        spec.observe_declared_covariates.iter().any(|c| c == "T"),
        "the spec must carry the [covariates] declarations, got {:?}",
        spec.observe_declared_covariates
    );

    // The controller reads the column (60.0 for this subject), not the decision time —
    // so `signal < 30` never fires and the dose is re-issued unchanged at all three
    // decisions. Under the model-time reading it would have fired at t=0 and t=24.
    let mut s = subj("P", vec![60.0], vec![]);
    s.covariates.insert("T".to_string(), 60.0);
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["T".to_string()];
    let opts = AdaptiveSimulateOptions {
        seed: Some(3),
        ..Default::default()
    };
    let res = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect("the covariate-driven controller runs");
    for d in &res.decisions {
        approx::assert_relative_eq!(d.observed_signals[0].value, 60.0, epsilon = 1e-9);
    }
    let amts: Vec<f64> = res.ledger.iter().map(|d| d.amt).collect();
    assert_eq!(amts, vec![100.0, 100.0, 100.0], "no rule may fire on 60.0");
}

#[test]
fn from_spec_observe_t_alias_fold_warns_at_parse_time() {
    // `observe` is compiled at simulate time, where `parse_warnings` is long sealed
    // and the adaptive result has no warnings channel — so the parser emits the
    // `T`-alias note itself, from the same helper the compiler runs. Without it the
    // fold would be silent on this one path (#1042 review of #1028).
    let src = SPEC_TIME_OBSERVE.replace("observe = TIME", "observe = T");
    let parsed = parse_full_model(&src).expect("the T alias parses");
    assert!(
        parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("[adaptive_dosing] observe") && w.contains("model-time built-in")),
        "expected a fold warning naming the block, got {:?}",
        parsed.model.parse_warnings
    );

    // The unambiguous `TIME` spelling stays quiet.
    let parsed_time = parse_full_model(SPEC_TIME_OBSERVE).expect("TIME parses");
    assert!(
        !parsed_time
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("model-time built-in")),
        "`TIME` must not warn, got {:?}",
        parsed_time.model.parse_warnings
    );
}

#[test]
fn from_spec_metrics_track_titration_and_window() {
    // End-to-end: the per-subject metrics row is populated from the same run's
    // ledger + decision log, keyed by (subject, draw, sim), and the
    // `pct_time_in_window` metric reads the spec's `target_window`. Uses the
    // converging titration with the band [8, 13] declared as the target.
    let parsed = parse_full_model(SPEC_TITRATE).expect("parse titrating block");
    let mut spec = parsed.adaptive_dosing.clone().expect("block present");
    spec.target_window = Some((8.0, 13.0));
    let pop = population(vec![subj("P", vec![342.0], vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(3),
        ..Default::default()
    };
    let res = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        &spec,
        &opts,
    )
    .expect("titration runs");

    assert_eq!(res.metrics.len(), 1, "one (subject, draw, sim) run");
    let m = &res.metrics[0];
    assert_eq!(m.subject, "P");
    assert_eq!((m.draw, m.sim), (1, 1));

    // The dose walks up from 20 and never holds/stops (a non-matching decision
    // re-issues the current dose), so every decision yields a ledger row.
    assert_eq!(m.n_doses, res.ledger.len());
    assert!(m.n_increases >= 1, "the dose climbs from start_dose = 20");
    assert_eq!(m.n_holds, 0);
    assert!(!m.discontinued);
    assert_eq!(m.time_to_discontinuation, None);

    // cumulative_dose mirrors the ledger sum, signal summary is populated.
    let ledger_sum: f64 = res.ledger.iter().map(|e| e.amt).sum();
    assert_eq!(m.cumulative_dose, ledger_sum);
    assert!(m.signal_min.is_some() && m.signal_max.is_some() && m.signal_mean.is_some());

    // pct_time_in_window re-derives from the decision log's observed troughs.
    let n = res.decisions.len();
    let in_band = res
        .decisions
        .iter()
        .filter(|d| (8.0..=13.0).contains(&d.observed_signals[0].value))
        .count();
    assert_eq!(m.pct_time_in_window, Some(in_band as f64 / n as f64));
    // No `auc_target` on this spec ⇒ the signal-AUC pass is skipped and the
    // exposure metric stays unreported.
    assert_eq!(m.auc_target_attainment, None);
}

#[test]
fn from_spec_reports_auc_target_attainment() {
    // End-to-end wiring of the exposure metric (#391 S2.5b): declaring
    // `auc_target` turns on the signal-AUC pass in `run_adaptive_population`,
    // which feeds `auc_target_attainment`. The exact AUCs are pinned by the
    // analytic unit test (`adaptive_window_signal_aucs_matches_closed_form`); here
    // two *extreme* bands make the attainment deterministic regardless of the
    // realized AUCs, so the run/pass/metric chain is exercised without re-deriving
    // the integral: an all-covering band attains 1, an unreachable band attains 0.
    let parsed = parse_full_model(SPEC_TITRATE).expect("parse titrating block");
    let mut spec = parsed.adaptive_dosing.clone().expect("block present");
    let pop = population(vec![subj("P", vec![342.0], vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(3),
        ..Default::default()
    };
    let run = |spec: &crate::sim::adaptive::AdaptiveDosingSpec| {
        simulate_adaptive_from_spec(
            &parsed.model,
            &pop,
            &parsed.model.default_params,
            1,
            spec,
            &opts,
        )
        .expect("titration runs")
    };

    // Every inter-decision exposure is non-negative, so `[0, +∞)` is always met.
    spec.auc_target = Some((0.0, f64::INFINITY));
    assert_eq!(run(&spec).metrics[0].auc_target_attainment, Some(1.0));

    // An exposure no finite window can reach ⇒ 0 attainment (not `None`: the
    // band *is* declared and there *are* windows).
    spec.auc_target = Some((1e18, f64::INFINITY));
    assert_eq!(run(&spec).metrics[0].auc_target_attainment, Some(0.0));
}

#[test]
fn auc_attainment_is_scored_over_realized_windows_after_discontinuation() {
    // Regression: under discontinuation, `auc_target_attainment` must score only
    // the windows between REALIZED decisions, not the full scheduled horizon.
    // After a `Stop` the later scheduled decisions never happen; counting their
    // dose-free, washed-out windows would fold discontinuation (already reported
    // by `discontinued` / `time_to_discontinuation`) into the exposure metric as
    // silent misses. Here the model doses once at decision 0; the pre-dose trough
    // then rises above the stop threshold at decision 1 → discontinue. Of the 15
    // SCHEDULED daily decisions only 2 are realized, so there is exactly ONE
    // realized window [0, 24] — well above the one-sided `[1, ∞)` floor ⇒
    // attainment 1.0. The old planned-horizon behavior scored all 14 scheduled
    // windows, the washout tail dragging attainment to ~3/14, so it fails here.
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central
  at = every 24 from 0 to 336
  start_dose = 100
  route = bolus(cmt=1)
  dose_bounds = [0, 1000]
  when signal < 5 : increase 25%
  when signal > 5 : stop
"#;
    let parsed = parse_full_model(src).expect("parse stop block");
    let mut spec = parsed.adaptive_dosing.clone().expect("block present");
    // One-sided floor: every dosed window clears it; the post-stop washout windows
    // (which the old behavior would have scored) fall below it.
    spec.auc_target = Some((1.0, f64::INFINITY));

    let pop = population(vec![subj("P", vec![342.0], vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default()
    };
    let res = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        &spec,
        &opts,
    )
    .expect("discontinuing run");

    // It actually discontinues, at the second decision (t = 24).
    let m = &res.metrics[0];
    assert!(m.discontinued, "run should hit the stop rule");
    assert_eq!(m.time_to_discontinuation, Some(24.0));
    // Only the realized decisions are logged (decision 0 dosed, decision 1
    // stopped); the 13 later scheduled decisions never occurred.
    assert_eq!(res.decisions.len(), 2, "only realized decisions logged");
    // One realized dose, at t=0: the `Stop` at decision 1 is DOSE-FREE. This is
    // the invariant that makes realized-window scoring lose no dosed window — a
    // declarative `Stop` never carries a final dose, so there is no dose issued
    // *at* the stop whose (dropped) post-stop window would have mattered.
    assert_eq!(
        res.ledger.len(),
        1,
        "one realized dose; the Stop dosed nothing"
    );

    // Scored over the ONE realized window [0, 24] ⇒ 1.0. (Old planned-horizon
    // behavior: ~3 of 14 scheduled windows in band ⇒ ~0.214, so this fails on it.)
    assert_eq!(m.auc_target_attainment, Some(1.0));
}

#[test]
fn from_spec_metrics_degenerate_run_has_no_changes() {
    // The metrics half of the degenerate oracle: a block that never titrates
    // re-issues a fixed 100 mg at each of the 3 decisions, so the metrics show no
    // dose changes, no holds, no discontinuation, and — with no `target_window`
    // declared — `pct_time_in_window` is unreported.
    let parsed = parse_full_model(SPEC_DEGENERATE).expect("parse model + block");
    let spec = parsed.adaptive_dosing.as_ref().expect("block present");
    let pop = population(vec![subj("1", vec![6.0, 30.0, 54.0], vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default()
    };
    let res = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect("degenerate run");

    assert_eq!(res.metrics.len(), 1);
    let m = &res.metrics[0];
    assert_eq!(m.n_doses, 3, "100 mg re-issued at every decision");
    assert_eq!(m.n_increases, 0);
    assert_eq!(m.n_decreases, 0);
    assert_eq!(m.n_holds, 0);
    assert!(!m.discontinued);
    assert_eq!(m.cumulative_dose, 300.0);
    assert_eq!(m.pct_time_in_window, None, "no target_window in the block");
}

#[test]
fn from_spec_rejects_decision_times_in_opts() {
    // The block's `at` is the schedule; an `opts.decision_times` is a second,
    // conflicting source of truth — rejected, not silently ignored.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let spec = simple_titration_spec();
    let pop = population(vec![subj("1", vec![6.0], vec![])]);
    let opts = AdaptiveSimulateOptions {
        decision_times: vec![0.0, 24.0],
        ..Default::default()
    };
    let err = simulate_adaptive_from_spec(&model, &pop, &model.default_params, 1, &spec, &opts)
        .unwrap_err();
    assert!(err.contains("opts.decision_times"), "got: {err}");
}

#[test]
fn from_spec_rejects_monitors_in_opts() {
    // The block's `observe` is the monitor; an `opts.monitors` is a second,
    // conflicting source of truth — rejected, not silently ignored.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let spec = simple_titration_spec();
    let pop = population(vec![subj("1", vec![6.0], vec![])]);
    let opts = AdaptiveSimulateOptions {
        monitors: vec![MonitorSpec::new("signal", 1, ObserveMode::Ipred)],
        ..Default::default()
    };
    let err = simulate_adaptive_from_spec(&model, &pop, &model.default_params, 1, &spec, &opts)
        .unwrap_err();
    assert!(err.contains("opts.monitors"), "got: {err}");
}

#[test]
fn from_spec_rejects_analytical_model() {
    // The reactive driver runs on the ODE engine; an analytical model has no ODE
    // spec and is rejected at compile (the ODE gate), never silently no-op.
    let model = parse_model_string(ANALYTICAL).expect("parse analytical");
    let spec = simple_titration_spec();
    let pop = population(vec![subj("1", vec![6.0], vec![])]);
    let opts = AdaptiveSimulateOptions::default();
    let err = simulate_adaptive_from_spec(&model, &pop, &model.default_params, 1, &spec, &opts)
        .unwrap_err();
    assert!(err.contains("ODE model"), "got: {err}");
}

#[test]
fn from_spec_rejects_observe_covariate_absent_from_data() {
    // An `observe` covariate not present in the data would silently read 0.0 and
    // drive the controller off a wrong signal (`central / BADCOV` → central/0).
    // Reject it loudly, exactly as a fit does for model covariates.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let spec = AdaptiveDosingSpec {
        observe: Some("central / BADCOV".to_string()),
        observe_declared_covariates: Vec::new(),
        ..simple_titration_spec()
    };
    let pop = population(vec![subj("1", vec![6.0], vec![])]); // no covariate columns
    let opts = AdaptiveSimulateOptions::default();
    let err = simulate_adaptive_from_spec(&model, &pop, &model.default_params, 1, &spec, &opts)
        .unwrap_err();
    assert!(
        err.contains("BADCOV") && err.contains("not found in data"),
        "got: {err}"
    );
}

#[test]
fn from_spec_dv_collapses_to_ipred_as_sigma_to_zero() {
    // The `with_assay_error` path adds the endpoint's residual draw to the signal
    // on the S1.5 controller-assay substream; as σ → 0 that draw vanishes, so the
    // assay-noised block must realize the same dose schedule as the latent (Ipred)
    // block. `observe = central` is a bare endpoint, so `assay_cmt` resolves to
    // compartment 1 with no ambiguity. (The full ledger does not bit-collapse —
    // residual variance floors at a minimum — so the *dose schedule* is the
    // collapse-invariant, exactly as for the programmatic Dv test.)
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let at = vec![0.0, 24.0, 48.0];
    let pop = population(vec![subj("1", vec![6.0, 30.0, 54.0], vec![])]);

    let ipred_spec = AdaptiveDosingSpec {
        observe: Some("central".to_string()),
        observe_declared_covariates: Vec::new(),
        with_assay_error: false,
        assay_cmt: None,
        at: at.clone(),
        start_dose: 100.0,
        route: AdaptiveRoute::Bolus { cmt: 1 },
        dose_bounds: (0.0, 1000.0),
        confirm: 1,
        levels: None,
        target_window: None,
        auc_target: None,
        rules: vec![AdaptiveRule {
            op: Comparison::Lt,
            threshold: 50.0,
            action: AdaptiveAction::Increase(DoseStep::Percent(25.0)),
        }],
    };
    // Choice-2 Dv: no observe expression — measure model output #1 (whose
    // readout `y = central` for ODE_NO_IIV is the same quantity the Ipred spec
    // observes) with assay noise.
    let dv_spec = AdaptiveDosingSpec {
        observe: None,
        observe_declared_covariates: Vec::new(),
        with_assay_error: true,
        assay_cmt: Some(1),
        ..ipred_spec.clone()
    };

    // Drive σ → 0 so the assay draw collapses onto the latent value.
    let mut params = model.default_params.clone();
    for s in params.sigma.values.iter_mut() {
        *s = 1e-12;
    }

    let opts = AdaptiveSimulateOptions {
        seed: Some(5),
        ..Default::default()
    };
    let ipred = simulate_adaptive_from_spec(&model, &pop, &params, 1, &ipred_spec, &opts)
        .expect("ipred run");
    let dv =
        simulate_adaptive_from_spec(&model, &pop, &params, 1, &dv_spec, &opts).expect("dv run");

    let schedule = |r: &AdaptiveSimulationResult| {
        r.ledger
            .iter()
            .map(|e| (e.time, e.amt, e.cmt))
            .collect::<Vec<_>>()
    };
    assert!(!dv.ledger.is_empty(), "the ladder fired");
    assert_eq!(
        schedule(&ipred),
        schedule(&dv),
        "with σ→0 the assay-noised block must realize the same doses as the latent block"
    );
}

#[test]
fn from_spec_is_deterministic_under_a_fixed_seed() {
    // Reproducibility through the declarative entry: a fixed seed gives identical
    // η draws (ODE_IIV puts variability on CL), so two runs agree exactly across
    // subjects and replicates.
    let model = parse_model_string(ODE_IIV).expect("parse");
    let spec = simple_titration_spec();
    let pop = population(vec![
        subj("A", vec![6.0, 30.0], vec![]),
        subj("B", vec![6.0, 30.0], vec![]),
    ]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(42),
        ..Default::default()
    };
    let r1 = simulate_adaptive_from_spec(&model, &pop, &model.default_params, 2, &spec, &opts)
        .expect("run 1");
    let r2 = simulate_adaptive_from_spec(&model, &pop, &model.default_params, 2, &spec, &opts)
        .expect("run 2");
    assert_eq!(r1.ledger, r2.ledger, "same seed ⇒ identical ledger");
    assert_eq!(
        r1.decisions, r2.decisions,
        "same seed ⇒ identical decisions"
    );
}

#[test]
fn from_spec_runs_the_shipped_tdm_example() {
    // The shipped example parses from disk, carries an [adaptive_dosing] block,
    // and runs end-to-end through the file-driven entry — exercising the
    // expression `observe`, the `with_assay_error` assay path, and the infusion
    // route together, with the frozen-replay verifier on by default.
    use crate::parser::model_parser::parse_full_model_file;
    use std::path::Path;
    let parsed = parse_full_model_file(Path::new("examples/adaptive_tdm_titration.ferx"))
        .expect("the shipped TDM example must parse");
    let spec = parsed
        .adaptive_dosing
        .as_ref()
        .expect("example declares an [adaptive_dosing] block");
    // Dose-free subjects; a trough is sampled just before each q12h decision,
    // with a final observation past the last infusion's end.
    let obs = vec![11.9, 23.9, 35.9, 47.9, 59.9, 71.9, 83.9, 95.9, 108.0];
    let pop = population(vec![
        subj("p1", obs.clone(), vec![]),
        subj("p2", obs.clone(), vec![]),
    ]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(2024),
        ..Default::default()
    };
    let res = simulate_adaptive_from_spec(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        spec,
        &opts,
    )
    .expect("the example titration runs and the verifier passes");
    assert!(!res.ledger.is_empty(), "the titration issues doses");
    assert!(
        res.ledger.iter().all(|e| e.amt >= 250.0 && e.amt <= 2000.0),
        "every realized dose must respect dose_bounds [250, 2000]"
    );
}

// ── diagnostics on the adaptive path (#1280 / #1304) ─────────────────────────

/// `[fit_options] ode_max_steps` starved to a handful of steps, so the driver's segments give
/// up and freeze-pad their tails. Same structural model as `ODE_NO_IIV`.
const ODE_STARVED_BUDGET: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_method = rk45
  ode_max_steps = 2
"#;

/// `simulate_adaptive` opens a `SolverStatsScope`, so its runs carry the ODE-solver
/// diagnostics (#1304 item 2) and the model/data bundle (#1280).
///
/// This is the entry point where a silent solver diagnostic matters most: the controller reads
/// the simulated state to pick the next dose, so a freeze-padded segment does not merely
/// mis-plot — it feeds the wrong signal into the next decision and the realized ledger
/// inherits it. Before this, `ode_predictions_adaptive_impl` and `adaptive_frozen_replay_tv`
/// recorded into an inactive sink on every path a user could take.
///
/// Regression this catches: no scope around the adaptive loop. Mutation — delete the
/// `solver_stats_scope` binding from `simulate_adaptive` (or make it `None`) and this test
/// fails while every other adaptive test stays green.
///
/// The straddle is the second arm: the identical call on the same model with an ordinary
/// budget must come back silent, so the assertion is about the starved solve and not about a
/// message the function always emits.
#[test]
fn simulate_adaptive_carries_the_solver_diagnostics_of_its_own_run() {
    let decisions = vec![0.0, 24.0];
    let obs = vec![6.0, 30.0, 54.0];
    let pop = population(vec![subj("1", obs.clone(), vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: decisions.clone(),
        // The frozen-replay verifier compares two freeze-padded trajectories on the starved
        // model; it is not what is under test here and its outcome is not the assertion.
        verify: false,
        ..Default::default()
    };

    let starved = parse_model_string(ODE_STARVED_BUDGET).expect("parse starved-budget model");
    let res = simulate_adaptive(
        &starved,
        &pop,
        &starved.default_params,
        1,
        fixed_bolus,
        &opts,
    )
    .expect("adaptive sim runs");
    assert!(
        res.warnings
            .iter()
            .any(|w| w.contains("W_ODE_SOLVER_DIAGNOSTICS")),
        "a starved adaptive run must name its solver trouble: {:?}",
        res.warnings
    );

    let ok = parse_model_string(ODE_NO_IIV).expect("parse no-IIV ODE model");
    let clean = simulate_adaptive(&ok, &pop, &ok.default_params, 1, fixed_bolus, &opts)
        .expect("adaptive sim runs");
    // The straddle: same model source, same population, same controller — only the step budget
    // differs, and a solve that finishes must say nothing *about the solver*.
    //
    // Not `is_empty()`. `ODE_NO_IIV` declares `ETA_CL` and never references it (deliberately —
    // the bit-exact oracles in this file rely on the trajectory being η-invariant), so the
    // model carries a parse warning, and since #1280 the bundle carries parse warnings. The
    // assertion below is scoped to the property under test and the one about *everything else*
    // is spelled out rather than dropped, so this stays a straddle instead of quietly widening
    // into "anything goes".
    assert!(
        !clean
            .warnings
            .iter()
            .any(|w| w.contains("W_ODE_SOLVER_DIAGNOSTICS")),
        "and a clean one must carry no solver diagnostic: {:?}",
        clean.warnings
    );
    assert_eq!(
        clean.warnings.len(),
        1,
        "the only thing a clean run of this fixture may carry is its unreferenced-omega parse \
         warning; a second entry means something else started firing: {:?}",
        clean.warnings
    );
    assert!(
        clean.warnings[0].contains("not referenced in any model expression"),
        "{:?}",
        clean.warnings
    );
}

// ===================== #1151: a model-time-reading [odes] RHS =====================
//
// The #391 degenerate oracle below (`degenerate_oracle_matches_static_predict`) runs on
// `ODE_NO_IIV`, whose RHS is **autonomous** — so it cannot see the one thing that split the
// reactive driver from the static engine in #1124: a RHS that reads model time. These are
// that missing half.
//
// The tolerances here are not decoration. Measured at `a6b67de5` on this model, the adaptive
// driver and `predict()` agree to **rel 8.6e-16** at `ode_reltol = 1e-12 / ode_abstol =
// 1e-14`, and both sit within 1.7e-13 of the closed form; at DEFAULT solver tolerances they
// are **1.3e-6** apart — solver noise that reads exactly like an anchoring defect. A fixture
// at default tolerances could therefore assert nothing sharper than ~1e-5, which a driver
// whose anchor was a whole segment stale would still pass.

/// 1-cpt IV ODE whose RHS reads `TAD`: `d/dt(central) = -(CL/V)·central·(1 + KT·TAD)`. The
/// `KT·TAD` factor makes the decay rate depend on time *since the last dose*, so an anchor
/// taken at the wrong instant (the segment start, the previous dose, a decision that did not
/// dose) integrates a different forcing and moves the trajectory.
///
/// `ETA_CL ~ 1e-10` is declared but unused by `CL`, so a drawn η leaves the trajectory at the
/// η=0 value and the adaptive IPRED is comparable to the static PRED. The tight solver
/// tolerances are what let the oracle assert at 1e-12 (see the module note above).
///
/// It has an exact closed form — on a segment after a dose at `t_d`, `TAD = t − t_d`, so
/// `C(t) = C(t_d⁺)·exp(−k[(t−t_d) + κ(t−t_d)²/2])` with `k = CL/V` and `κ = KT`. That is a
/// **third** reference, outside both engines: `tad_closed_form` below walks it.
const ODE_TAD_NO_IIV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVKT(0.01, 1e-4, 1.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
  KT = TVKT
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central * (1.0 + KT * TAD)
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

/// 1-cpt IV ODE whose RHS reads `TIME` — the **solver axis**, not a dose clock. It is the
/// control for #1151's refusal: `TIME` is anchored at the origin of the integration and is
/// finite everywhere, so a dose-free window carries no `NaN` and must NOT be refused. A guard
/// keyed on `pk_reads_model_time` (which unions `TAD`, `TAFD`, `T`/`TIME`) rejects this model.
const ODE_TIME_NO_IIV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central * (1.0 + 1e-3 * TIME)
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

/// A controller that doses 100 mg into cmt 1 at the listed decision **indices** and holds at
/// every other one. `fixed_bolus` (dose at every decision) is the `idx = all` instance; the
/// interesting cells are the ones where a decision passes without a dose.
fn dose_at_decisions(idx: Vec<usize>) -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    move |ctx: &ControllerCtx| {
        if idx.contains(&ctx.decision_index) {
            vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
        } else {
            vec![DoseAction::Hold]
        }
    }
}

/// The closed form of `ODE_TAD_NO_IIV`, walked outside both engines (#1151).
///
/// The equation is linear in `central` with a decay rate that depends only on `t`, so the
/// whole compartment shares one factor per dosing interval: on `(t_d, t']` with the anchor at
/// `t_d`, `C(t') = C(t_d⁺)·exp(−k[(t'−t_d) + κ(t'−t_d)²/2])`. This walks the realized bolus
/// schedule and reports the state at each requested time. `doses` and `obs` must be sorted
/// ascending, and every `obs` must be at or after the first dose (before it the closed form
/// has no anchor either — which is the whole of #1151's refusal). The walk starts at the
/// first dose or observation, the subject's origin (#936).
fn tad_closed_form(doses: &[f64], obs: &[f64]) -> Vec<f64> {
    let k = 5.0 / 50.0;
    let kappa = 0.01;
    let decay = |dt: f64| (-k * (dt + kappa * dt * dt / 2.0)).exp();

    let mut points: Vec<f64> = doses.iter().chain(obs.iter()).copied().collect();
    points.sort_by(f64::total_cmp);
    points.dedup();

    let mut c = 0.0f64;
    let mut anchor = f64::NAN;
    // The walk starts at the first event, not at t=0 (#936): nothing evolves before the
    // subject's origin, which is its first record or realized dose.
    let mut prev = points.first().copied().unwrap_or(0.0);
    let mut out: Vec<f64> = Vec::with_capacity(obs.len());
    for t in points {
        if t > prev {
            assert!(
                anchor.is_finite(),
                "the closed form has no anchor before the first dose either (t={t})"
            );
            // One factor for the whole compartment: advance from `prev` to `t` under the
            // anchor in force, as a ratio of the two elapsed-time factors.
            c = c * decay(t - anchor) / decay(prev - anchor);
            prev = t;
        }
        if doses.contains(&t) {
            c += 100.0;
            anchor = t;
        }
        if obs.contains(&t) {
            out.push(c);
        }
    }
    out
}

/// `(adaptive IPRED, static PRED)` for one degenerate-oracle cell: the controller doses at
/// `dose_idx`, and the same realized schedule (plus any base regimen) is pre-scheduled through
/// `predict()`. The frozen-replay verifier runs too (default `verify: true`), so its `Ok` is
/// part of every caller's assertion.
fn tad_oracle_cell(
    src: &str,
    decisions: &[f64],
    dose_idx: Vec<usize>,
    obs: &[f64],
    base: Vec<DoseEvent>,
) -> (Vec<f64>, Vec<f64>) {
    let model = parse_model_string(src).expect("parse model-time-reading ODE model");
    let pop = population(vec![subj("1", obs.to_vec(), base.clone())]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: decisions.to_vec(),
        ..Default::default()
    };
    let res = simulate_adaptive(
        &model,
        &pop,
        &model.default_params,
        1,
        || dose_at_decisions(dose_idx.clone()),
        &opts,
    )
    .expect("adaptive sim runs (the frozen-replay verifier passes too)");

    let mut static_doses = base;
    for i in &dose_idx {
        static_doses.push(DoseEvent::new(decisions[*i], 100.0, 1, 0.0, false, 0.0));
    }
    static_doses.sort_by(|x, y| x.time.total_cmp(&y.time));
    let static_pop = population(vec![subj("1", obs.to_vec(), static_doses)]);
    let preds = predict(&model, &static_pop, &model.default_params).unwrap();

    (
        res.trajectories.iter().map(|t| t.ipred).collect(),
        preds.iter().map(|p| p.pred).collect(),
    )
}

/// The `Err` from a `simulate_adaptive` run that must be refused (#1151). `verify: false`, so
/// the error can only come from the driver's own guard — with the verifier on, a `NaN`
/// trajectory would be caught by the replay instead and the message would be the verifier's.
fn tad_refusal_error(
    src: &str,
    decisions: &[f64],
    dose_idx: Vec<usize>,
    obs: &[f64],
    base: Vec<DoseEvent>,
) -> String {
    tad_refusal(src, decisions, dose_idx, obs, base)
        .expect("an unanchored dose clock must be refused, not returned as NaN")
}

/// [`tad_refusal_error`] without the expectation: `Some(message)` when the run is refused,
/// `None` when it returns — for a cell whose failure message must say which mechanism missed.
fn tad_refusal(
    src: &str,
    decisions: &[f64],
    dose_idx: Vec<usize>,
    obs: &[f64],
    base: Vec<DoseEvent>,
) -> Option<String> {
    let model = parse_model_string(src).expect("parse model-time-reading ODE model");
    let pop = population(vec![subj("1", obs.to_vec(), base)]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: decisions.to_vec(),
        verify: false,
        ..Default::default()
    };
    simulate_adaptive(
        &model,
        &pop,
        &model.default_params,
        1,
        || dose_at_decisions(dose_idx.clone()),
        &opts,
    )
    .err()
    .map(|err| format!("{err}"))
}

/// Every prediction is finite, with the worst relative gap against `want` reported.
///
/// `f64::max` returns the *other* operand on a `NaN`, so folding `worst` over a run that
/// produced `NaN` would report whatever the finite records produced and pass — the exact hole
/// AGENTS.md names. The finiteness of both sides is asserted per element, before the fold.
///
/// **A zero reference contributes an ABSOLUTE error** (`|got|`), since no relative one exists
/// there — so a caller's bound is read as absolute for those elements and relative for the
/// rest, two scales under one constant (#1534 review, nit 4). Every zero element in this file
/// is an empty compartment before the first dose, and the cells that have one
/// (`..._base_dose_anchors_the_window_before_the_first_dose`, the two finding-1 controls) pin
/// it exactly with `assert_eq!(adaptive[0], 0.0)` of their own; this fold is then only
/// guarding the non-zero elements, which is where the stated bound was measured.
fn worst_rel(got: &[f64], want: &[f64], what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(g.is_finite(), "{what}: got[{i}] = {g} is not finite");
        assert!(w.is_finite(), "{what}: want[{i}] = {w} is not finite");
        let rel = if w == 0.0 {
            g.abs()
        } else {
            (g - w).abs() / w.abs()
        };
        worst = worst.max(rel);
    }
    worst
}

#[test]
fn degenerate_oracle_tad_rhs_matches_static_predict_and_the_closed_form() {
    // T1 (#1151). The #391 degenerate oracle on a TAD-reading RHS: a controller re-emitting a
    // fixed regimen must equal `predict()` on that regimen — and both must equal the closed
    // form, which is computed outside either engine.
    //
    // Mutation that reddens it: anchor the driver's TAD at the FIRST dose
    // (`earliest_dose_time`, i.e. TAFD semantics) instead of the most recent one. Measured:
    // kills this test, T2, T3 and T4, plus seven pre-existing `TAD` tests.
    //
    // Note what does NOT redden it: anchoring at the segment start (`t_start`). A dose lands
    // at every decision here and the decisions are the breaks, so the segment start *is* the
    // last dose time and that mutation is invisible to this cell — measured, it kills only
    // the hold cell below. That asymmetry is why T2 exists.
    let decisions = [0.0, 24.0, 48.0];
    let obs = [6.0, 30.0, 54.0];
    let (adaptive, static_pred) =
        tad_oracle_cell(ODE_TAD_NO_IIV, &decisions, vec![0, 1, 2], &obs, vec![]);
    let closed = tad_closed_form(&decisions, &obs);

    // Measured at `a6b67de5`: 8.6e-16 against `predict()`, 1.4e-13 against the closed form.
    // The bounds carry three orders of headroom over each.
    let vs_static = worst_rel(&adaptive, &static_pred, "adaptive vs predict()");
    assert!(
        vs_static <= 1e-12,
        "adaptive vs predict(): worst rel {vs_static:e} > 1e-12 \
         (adaptive={adaptive:?}, static={static_pred:?})"
    );
    let vs_closed = worst_rel(&adaptive, &closed, "adaptive vs closed form");
    assert!(
        vs_closed <= 1e-10,
        "adaptive vs closed form: worst rel {vs_closed:e} > 1e-10 \
         (adaptive={adaptive:?}, closed={closed:?})"
    );
}

#[test]
fn degenerate_oracle_tad_rhs_holds_do_not_move_the_anchor() {
    // T2 (#1151). Decisions at 0/12/24/36, doses only at 0 and 24 — so two decisions pass
    // with no dose. `TAD` must stay anchored at the last DOSE across them.
    //
    // Mutation that reddens it: advance the anchor at every break — `tad_anchor(...)` replaced
    // by `t_start` — so a decision that did not dose moves it. The obs at 30 and 45 sit in the
    // window opened at 24; at 45 the mutated anchor would be 36, understating the elapsed time
    // by 12 h. Measured: this cell is the only one of the four oracle cells that mutation
    // kills, because it is the only one with a break that is not a dose.
    let decisions = [0.0, 12.0, 24.0, 36.0];
    let obs = [6.0, 30.0, 45.0];
    let (adaptive, static_pred) =
        tad_oracle_cell(ODE_TAD_NO_IIV, &decisions, vec![0, 2], &obs, vec![]);
    let closed = tad_closed_form(&[0.0, 24.0], &obs);

    // Measured: 4.2e-15 against `predict()`, 4.5e-14 against the closed form.
    let vs_static = worst_rel(&adaptive, &static_pred, "adaptive vs predict()");
    assert!(
        vs_static <= 1e-12,
        "adaptive vs predict() across two hold decisions: worst rel {vs_static:e} > 1e-12 \
         (adaptive={adaptive:?}, static={static_pred:?})"
    );
    let vs_closed = worst_rel(&adaptive, &closed, "adaptive vs closed form");
    assert!(
        vs_closed <= 1e-10,
        "adaptive vs closed form across two hold decisions: worst rel {vs_closed:e} > 1e-10 \
         (adaptive={adaptive:?}, closed={closed:?})"
    );
}

#[test]
fn degenerate_oracle_tad_rhs_observation_on_a_decision_boundary_is_bit_identical() {
    // T3 (#1151). An observation lands exactly ON a decision that doses: the read is taken
    // post-dose at the segment's left boundary, so it is a pure state readout with no
    // integration between the dose and the read — and the two engines must agree to the BIT,
    // not to a tolerance.
    //
    // Mutation that reddens it: take the boundary read from the pre-dose side, or off the
    // previous segment's integration. Either moves the value by the whole 100 mg bolus.
    let decisions = [0.0, 24.0, 48.0];
    let obs = [24.0, 48.0];
    let (adaptive, static_pred) =
        tad_oracle_cell(ODE_TAD_NO_IIV, &decisions, vec![0, 1, 2], &obs, vec![]);

    for (i, (&a, &s)) in adaptive.iter().zip(static_pred.iter()).enumerate() {
        assert!(a.is_finite(), "adaptive[{i}] = {a} is not finite");
        assert_eq!(
            a.to_bits(),
            s.to_bits(),
            "observation on a decision boundary must be bit-identical: \
             adaptive={a} static={s} at t={}",
            obs[i]
        );
    }
}

#[test]
fn adaptive_tad_rhs_base_dose_anchors_the_window_before_the_first_dose() {
    // T4 (#1151) — and message-table row 3's silence check. A pre-scheduled base dose at
    // t=10 is known to the driver up front, so `tad_anchor_for`'s first-arrival fallback gives
    // the pre-dose window a FINITE (negative) `TAD`, exactly as `predict()` does. The window
    // must not be refused, and the run must match the static engine.
    //
    // The read at t=6 is in that pre-dose window and is 0.0 — its job is to pin "not NaN",
    // since `0.0 * NaN` is what poisons the unanchored case.
    //
    // **This cell is degenerate for the anchor VALUE** (#1534 review round 3, finding F), and
    // the twin below is what is not. `central ≡ 0` on `(0, 10]`, so every anchor gives a zero
    // derivative there and the window's `TAD` cannot move anything. Measured with a
    // driver-only mis-anchor — the real `integrate_segment` call's `TAD` anchor forced to
    // `t_start` whenever the shadow has no dose at or before `t_start`, leaving `predict()`
    // untouched — this test stays GREEN at `[0.0, 79.07598442448013, 74.71656464518534]`. So
    // what it pins is the refusal's scope (a base dose silences it), not the anchoring itself.
    //
    // Mutation that reddens it: make the refusal fire whenever an observation precedes the
    // first REALIZED dose, ignoring base doses.
    let decisions = [12.0, 36.0];
    let obs = [6.0, 20.0, 40.0];
    let base = vec![DoseEvent::new(10.0, 100.0, 1, 0.0, false, 0.0)];
    let (adaptive, static_pred) =
        tad_oracle_cell(ODE_TAD_NO_IIV, &decisions, vec![0, 1], &obs, base);

    assert_eq!(
        adaptive[0], 0.0,
        "the pre-dose read is the empty compartment, not NaN (got {})",
        adaptive[0]
    );
    // Measured: 1.5e-15.
    let vs_static = worst_rel(&adaptive, &static_pred, "adaptive vs predict()");
    assert!(
        vs_static <= 1e-12,
        "a base-dose-anchored pre-dose window must match predict(): worst rel \
         {vs_static:e} > 1e-12 (adaptive={adaptive:?}, static={static_pred:?})"
    );
}

/// T4's twin with a **non-zero starting state**, so the value of the base-dose anchor is
/// observable (#1534 review round 3, finding F). `ODE_TAD_NO_IIV`'s compartment is empty until
/// the first dose, which makes every anchor give the same zero derivative in the pre-dose
/// window; here the window integrates a real decay under a real `TAD`, so a mis-anchored
/// window moves the trajectory.
const ODE_TAD_INIT50: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVKT(0.01, 1e-4, 1.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
  KT = TVKT
[structural_model]
  ode(states=[central])
[odes]
  init(central) = 50.0
  d/dt(central) = -(CL / V) * central * (1.0 + KT * TAD)
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

#[test]
fn adaptive_tad_rhs_base_dose_anchor_value_matches_the_static_engine() {
    // T4's non-degenerate twin (#1534 review round 3, finding F). Same shape as T4 — a
    // pre-scheduled base dose at t=10, controller doses at 12 and 36, a read at t=6 inside the
    // pre-dose window — but the compartment starts at 50, so the window integrates under a
    // real (negative) `TAD` and the anchor's VALUE reaches the trajectory.
    //
    // Mutation that reddens it: a driver-only mis-anchor — force the real `integrate_segment`
    // call's `TAD` anchor to `t_start` whenever the shadow has no dose at or before `t_start`,
    // which leaves `predict()` alone. Measured: 26.95107015382148 against 28.617631257835207 at t=6,
    // and the frozen-replay verifier catches it too. T4 itself stays green under that same
    // mutation, which is why this twin exists.
    let decisions = [12.0, 36.0];
    let obs = [0.0, 6.0, 20.0, 40.0];
    let base = vec![DoseEvent::new(10.0, 100.0, 1, 0.0, false, 0.0)];
    let (adaptive, static_pred) =
        tad_oracle_cell(ODE_TAD_INIT50, &decisions, vec![0, 1], &obs, base);

    // The read at t=0 is the `init` baseline, before anything has happened.
    assert_eq!(adaptive[0], 50.0, "init(central) = 50: {adaptive:?}");
    // The read at t=6 is inside the base-dose-anchored window and is NOT zero — that is the
    // whole point of the twin. Measured 28.617631257835207.
    assert!(
        adaptive[1] > 20.0 && adaptive[1] < 40.0,
        "the pre-dose read must carry the window's integration, not sit at zero: {adaptive:?}"
    );
    // Measured worst rel: 8.1e-16.
    let vs_static = worst_rel(&adaptive, &static_pred, "adaptive vs predict()");
    assert!(
        vs_static <= 1e-12,
        "a base-dose-anchored pre-dose window must match predict() in VALUE, not only in \
         finiteness: worst rel {vs_static:e} (adaptive={adaptive:?}, static={static_pred:?})"
    );
}

#[test]
fn adaptive_tad_rhs_refuses_the_window_before_the_first_dose() {
    // T5 (#1151). Dose-free base, first dose realized at the decision at t=12, an observation
    // at t=6 inside the unanchored window. `TAD` has no referent there in a causal run, and
    // the driver must say so rather than integrate `0.0 * NaN`.
    //
    // Mutation that reddens it: delete the guard — the call then returns `Ok` with
    // `[NaN, NaN, NaN]` (measured at `a6b67de5`), with no warning naming `TAD`.
    //
    // Every sentence of the message is asserted below, so deleting any one of them reddens
    // this test (AGENTS.md: a sentence no test can kill is a claim nobody has checked).
    //
    // #936: the obs at 6 is the subject's first record, so it is the origin and is read at
    // the refused segment's LEFT boundary (a finite `init` read); the obs at 9 is the one
    // read off the segment, which sentence 2 names.
    let err = tad_refusal_error(
        ODE_TAD_NO_IIV,
        &[12.0, 36.0],
        vec![0, 1],
        &[6.0, 9.0, 20.0, 40.0],
        vec![],
    );

    // Sentence 1 — the two observed facts, and the window they were observed on.
    assert!(err.contains("`TAD`"), "must name the slot: {err}");
    assert!(
        err.contains("integrated to a non-finite state"),
        "must report the outcome it actually observed: {err}"
    );
    assert!(
        err.contains("no dose has been given in this run"),
        "must say why the clock is unanchored: {err}"
    );
    // #936: the window opens at the subject's first record (t=6), not at t=0 — nothing is
    // integrated before the origin.
    assert!(
        err.contains("(6, 12]"),
        "must name the refused window: {err}"
    );
    // Sentence 2 — which read is affected.
    assert!(
        err.contains("The observation at t=9 is read off that segment"),
        "must name the observation in the window: {err}"
    );
    // Sentence 3 — why the static engines' answer is not copied. The leading condition is
    // load-bearing: on a record with no dose at all, `predict()` returns NaN here too, which
    // `adaptive_tad_rhs_refuses_when_the_controller_never_doses` pins.
    assert!(
        err.contains("Where the record contains a dose"),
        "the anchoring claim must be conditional on the record carrying a dose: {err}"
    );
    assert!(
        err.contains("not decided yet, and may never decide"),
        "must say the anchoring dose is undecided, and may never be decided: {err}"
    );
    // Sentence 4 — the fixes, and the case neither fix covers.
    assert!(
        err.contains("pre-scheduled base regimen"),
        "must offer the base-regimen fix: {err}"
    );
    assert!(
        err.contains("dose at a decision at or before the subject's first record"),
        "must offer the earlier-decision fix: {err}"
    );
    assert!(
        !err.contains("start of the horizon"),
        "#936: the origin is the first record, not t=0 — the old wording no longer holds: {err}"
    );
    assert!(
        err.contains("a controller that never doses leaves a model reading `TAD` unanchored"),
        "must name the never-dosing case, which neither fix addresses: {err}"
    );
    // The must-nots: the model is fine and the controller is not at fault.
    assert!(
        !err.contains("unsupported"),
        "the model is supported — only this window is refused: {err}"
    );

    // #936: the advice must be TRUE. Applied literally — a controller dose at a decision at
    // the subject's first record (t=6) — the window disappears and the run is valid, equal to
    // `predict()` (`tad_oracle_cell` expects `Ok`, with the verifier on).
    let (fixed, fixed_static) = tad_oracle_cell(
        ODE_TAD_NO_IIV,
        &[6.0, 12.0, 36.0],
        vec![0, 1, 2],
        &[6.0, 9.0, 20.0, 40.0],
        vec![],
    );
    let vs_static = worst_rel(
        &fixed,
        &fixed_static,
        "advice applied: adaptive vs predict()",
    );
    assert!(
        vs_static <= 1e-12,
        "with the advice applied the run must match predict(): worst rel {vs_static:e}"
    );
}

#[test]
fn adaptive_tad_rhs_refusal_tracks_the_first_dose_not_the_first_decision() {
    // T6 (#1151), cell 8. The first decision is at t=0 and HOLDS; the dose lands at the
    // second, t=24. The window `(0, 24]` is therefore after the run's start AND after the
    // first decision, yet still before any dose — so it must be refused, and the message must
    // name that window, not `(0, 12]`-style "before the first decision".
    //
    // Mutation that reddens it: key the guard on the decision schedule — run it only while
    // `t_end <= decision_times[0]`. That still refuses T5's window `(0, 12]` (whose `t_end` is
    // the first decision) but lets this one through, returning NaN. Measured: it kills this
    // test and nothing else.
    //
    // Since #936 the walk integrates nothing before the origin — here the first record, t=10,
    // since the hold at t=0 issues no dose — so the refused window is `(10, 24]`. Unanchored
    // segments are a prefix of the integrated walk (a dose is only ever appended), so the
    // first refused segment always starts at the origin; this test pins the `t_end` side,
    // which is the half that can differ.
    let err = tad_refusal_error(
        ODE_TAD_NO_IIV,
        &[0.0, 24.0],
        vec![1],
        &[10.0, 15.0, 30.0],
        vec![],
    );

    assert!(err.contains("`TAD`"), "must name the slot: {err}");
    assert!(
        err.contains("(10, 24]"),
        "the refused window runs from the origin (first record, 10) to the first DOSE (24), \
         not to the first decision (0): {err}"
    );
    assert!(
        err.contains("The observation at t=15 is read off that segment"),
        "must name the observation inside that window: {err}"
    );
}

#[test]
fn adaptive_tad_rhs_dose_free_base_with_every_record_after_the_first_dose_is_valid() {
    // #936 converts #1151's `..._refuses_a_dose_free_window_with_no_observation_in_it`. That
    // cell (dose-free base, obs 20/40, first dose at the decision at 12) was refused because
    // the driver integrated `(0, 12]` from a hard-coded t=0 with no dose clock. But the
    // subject's first event is the realized dose at 12 — the origin the static engine starts
    // at — so no window precedes it and the run is valid: `Ok`, equal to `predict()` on the
    // realized ledger and to the closed form. The #1151 refusal for a window that DOES precede
    // the first dose is still pinned by `adaptive_tad_rhs_refuses_the_window_before_the_first_dose`
    // (the same cell with an obs at 6).
    //
    // Mutation that reddens it: revert the driver's break seed to `0.0` (the walk then
    // integrates `(0, 12]` under a `NaN` clock and the run is refused again).
    let decisions = [12.0, 36.0];
    let obs = [20.0, 40.0];
    let (adaptive, static_pred) =
        tad_oracle_cell(ODE_TAD_NO_IIV, &decisions, vec![0, 1], &obs, vec![]);
    let closed = tad_closed_form(&decisions, &obs);
    let vs_static = worst_rel(&adaptive, &static_pred, "adaptive vs predict()");
    assert!(
        vs_static <= 1e-12,
        "adaptive vs predict(): worst rel {vs_static:e} (adaptive={adaptive:?}, \
         static={static_pred:?})"
    );
    let vs_closed = worst_rel(&adaptive, &closed, "adaptive vs closed form");
    assert!(
        vs_closed <= 1e-10,
        "adaptive vs closed form: worst rel {vs_closed:e} (adaptive={adaptive:?}, \
         closed={closed:?})"
    );
}

#[test]
fn adaptive_tafd_rhs_refuses_the_window_before_the_first_dose() {
    // T7 (#1151). The same defect on `TAFD`, on the file's existing TAFD fixture — whose
    // working case (`adaptive_base_regimen_controller_dose_before_base_anchors_tafd_globally`)
    // carries a base regimen and is T4's shape. The message must name `TAFD`, the spelling
    // that is actually unanchored.
    //
    // Mutation that reddens it: key the guard on `TAD` alone. `ODE_TAFD`'s RHS never reads
    // `TAD`, so the run returns `Ok` with `[NaN, NaN, NaN]` (measured).
    let err = tad_refusal_error(
        ODE_TAFD,
        &[12.0, 36.0],
        vec![0, 1],
        &[6.0, 20.0, 40.0],
        vec![],
    );

    assert!(err.contains("`TAFD`"), "must name TAFD: {err}");
    assert!(
        !err.contains("`TAD`"),
        "TAD is anchored here — it is not the unanchored slot: {err}"
    );
    assert!(
        err.contains("(6, 12]"),
        "must name the refused window: {err}"
    );
}

#[test]
fn adaptive_time_reading_rhs_is_not_refused_before_the_first_dose() {
    // #1151, message-table row 6 — the over-refusal control. `TIME` is the integration axis,
    // not a dose clock: it is finite in the pre-dose window, so a dose-free run on a
    // TIME-reading RHS integrates correctly and must be left alone.
    //
    // Mutation that reddens it: key the candidate slots on `pk_reads_model_time()` — which
    // unions `TAD`, `TAFD` and `T`/`TIME` — AND drop the dependence test. Measured on #1535,
    // neither half alone suffices: this RHS reads no clock slot, so anchoring one cannot move
    // its derivative, and without the model-time key it has no candidate slot at all. (Before
    // the #1534 review moved the guard after the solve, the `pk_reads_model_time()` mutation
    // alone killed this test, which is how the per-spelling split was pinned.)
    let (adaptive, static_pred) = tad_oracle_cell(
        ODE_TIME_NO_IIV,
        &[12.0, 36.0],
        vec![0, 1],
        &[6.0, 20.0, 40.0],
        vec![],
    );

    // Default solver tolerances on this fixture (the point is the absence of a refusal, not a
    // sharp oracle), so the bound is the two engines' measured noise floor: worst rel 5.2e-7.
    //
    // And note what it cannot see (#1534 review round 3, finding F): the first read is at t=6
    // and `central ≡ 0` before the first dose, so the agreement below confirms "not `NaN`",
    // never the value `TIME` took in the pre-dose window. Anything that changes only that
    // value — #936's integration origin, for one — is invisible here.
    let vs_static = worst_rel(&adaptive, &static_pred, "adaptive vs predict()");
    assert!(
        vs_static <= 1e-5,
        "a TIME-reading RHS must run, and agree with predict(): worst rel {vs_static:e} \
         (adaptive={adaptive:?}, static={static_pred:?})"
    );
}

#[test]
fn adaptive_autonomous_rhs_is_not_refused_before_the_first_dose() {
    // #1151, message-table row 5 — the other over-refusal control. `ODE_NO_IIV`'s RHS reads no
    // clock at all, so a dose-free window is ordinary integration of an empty compartment.
    //
    // Mutation that reddens it: drop the `rhs_program` spelling checks AND the dependence test
    // — i.e. refuse on the `NaN` anchor alone, which is `NaN` here too, simply never read.
    // Measured on #1535, neither half alone is enough: without the spelling checks the
    // dependence test still finds nothing, since this RHS reads no clock, and without the
    // dependence test the spelling checks still find no candidate slot. That redundancy is
    // deliberate — it is the "belt" a future narrowing of either half would land on — but it
    // means this test's guarantee is "a working run keeps working", not a single-mutation kill.
    let (adaptive, static_pred) = tad_oracle_cell(
        ODE_NO_IIV,
        &[12.0, 36.0],
        vec![0, 1],
        &[6.0, 20.0, 40.0],
        vec![],
    );

    // Measured worst rel: 0e0 — the two engines are BIT-identical here, so the bound is bit
    // equality rather than a band whose every order would be decoration (#1534 review,
    // finding 3). Structural, not luck: the autonomous RHS is the same function on both
    // sides, and the two walks share a break set (doses 12/36 are the driver's decisions, and
    // `t_last = 40` on both), so the step sequences coincide. The same argument as T3.
    for (i, (&a, &s)) in adaptive.iter().zip(static_pred.iter()).enumerate() {
        assert!(a.is_finite(), "adaptive[{i}] = {a} is not finite");
        assert_eq!(
            a.to_bits(),
            s.to_bits(),
            "an autonomous RHS must run unrefused, and match predict() exactly: \
             adaptive={adaptive:?}, static={static_pred:?}"
        );
    }
}

// ============ #1534 review, finding 1: a SYNTACTIC TAD read is not a TAD evaluation ============
//
// `OdeRhsProgram::pk_reads_tad()` is `stmts_read_slots`, which recurses into `if` arms and into
// conditions. It is therefore true whenever `TAD` appears anywhere in the `[odes]` body —
// including where the unanchored window never evaluates it. The two fixtures below are those
// shapes. Before the guard was moved after `integrate_segment`, both were refused although each
// returns a finite trajectory (and passes the frozen-replay verifier) with the guard removed.
// Since #1535 the guard asks whether the window's derivative depends on the clock, and in
// neither fixture as written does it.

/// `TAD` read only inside a branch the unanchored window does not take. Over `(0, 12]` the
/// `TIME > 20` test is false, so the else arm integrates and the `NaN` anchor never reaches the
/// state; the `TAD` term switches on only at t=20, by which point the dose at 12 has anchored it.
/// Tight solver tolerances, like the rest of the #1151 block, so the cell is an oracle and not
/// only a non-refusal smoke test (#1534 review round 2).
const ODE_TAD_IN_UNTAKEN_BRANCH: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  if (TIME > 20.0) {
    d/dt(central) = -(CL / V) * central * (1.0 + 0.01 * TAD)
  } else {
    d/dt(central) = -(CL / V) * central
  }
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

/// `TAD` read only in a *condition*. `NaN > 5.0` is `false` (IEEE: an ordering comparison
/// against NaN is false, verified by this test's green run, not recalled), so the unanchored
/// window takes the else arm and nothing non-finite reaches the state — while `pk_reads_tad()` is
/// still true.
///
/// **The `NaN` still chose the arm.** Here `central ≡ 0` on the window, where both arms give a
/// zero derivative, so which one ran cannot show — and no anchor can move a derivative that is
/// zero in both arms, which is why it is not refused (#1535). A non-zero start makes the arms
/// differ, yet `TAD > 5` still takes the else arm for every `TAD ≤ 0` a schedule can give the
/// pre-dose window, so even then the run equals `predict()`; it is the `TAD < 5` twin, whose arm
/// does differ there, that is refused (#1570 review, row 1). The test below runs all four cells.
/// So this cell pins "a condition read that cannot move the derivative is not refused", not "a
/// condition read is harmless" (#1534 review round 2, finding B).
const ODE_TAD_IN_CONDITION_ONLY: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  if (TAD > 5.0) { d/dt(central) = -(CL / V) * central * 2.0 }
  else           { d/dt(central) = -(CL / V) * central }
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

#[test]
fn adaptive_tad_in_an_untaken_branch_is_not_refused() {
    // #1534 review, finding 1. Dose-free base, first dose realized at t=12 — the same schedule
    // T5 refuses — but this RHS only evaluates `TAD` after t=20, when the clock is anchored.
    // The run must complete and match `predict()` on the realized schedule.
    //
    // Mutation that reddens it: drop the dependence test — refuse whenever `TAD` is read and
    // unanchored, which is also the pre-#1534 guard. The run is then refused, although no
    // anchor moves this window's derivative and the trajectory is finite and verifier-clean.
    let decisions = [12.0, 36.0];
    let obs = [6.0, 20.0, 40.0];
    let (adaptive, static_pred) = tad_oracle_cell(
        ODE_TAD_IN_UNTAKEN_BRANCH,
        &decisions,
        vec![0, 1],
        &obs,
        vec![],
    );

    assert_eq!(
        adaptive[0], 0.0,
        "the pre-dose read is the empty compartment, not NaN (got {})",
        adaptive[0]
    );
    // At the fixture's tight tolerances, measured worst rel 6.7e-15 — so this is an oracle
    // cell, not a non-refusal smoke test (#1534 review round 2). At DEFAULT tolerances the
    // same comparison reads 1.937e-4, because the RHS steps across a discontinuity at t=20
    // that neither engine breaks on; that number is the solver's, not the driver's.
    let vs_static = worst_rel(&adaptive, &static_pred, "adaptive vs predict()");
    assert!(
        vs_static <= 1e-12,
        "a TAD read in an untaken branch must run, and track predict(): worst rel \
         {vs_static:e} (adaptive={adaptive:?}, static={static_pred:?})"
    );
}

#[test]
fn adaptive_tad_read_only_in_a_condition_is_refused_only_when_the_arm_matters() {
    // #1534 review, finding 1, and #1535's straddle as settled on #1570's review (row 1). Here
    // `TAD` reaches nothing but a comparison. In the unanchored window the driver tests `NaN`
    // against 5, and the static engine tests the window's `TAD ≤ 0` (its first dose lands at 12
    // or later). Four cells, each one input away from a neighbour:
    //
    //   cell  condition   start       verdict   why
    //   A     TAD > 5     empty       runs      both arms give a zero derivative
    //   B     TAD > 5     init 50     runs      the unanchored arm (else) is the arm every
    //                                           TAD ≤ 0 takes, so the run equals predict()
    //   C     TAD < 5     empty       runs      both arms give a zero derivative
    //   D     TAD < 5     init 50     refused   every TAD ≤ 0 takes the ×2 arm, the unanchored
    //                                           clock the else arm
    //
    // A and B differ by the `init` line, C and D likewise, and A→C (so B→D) flips the one
    // comparison; each derivation is asserted, not assumed. The first cut of #1535 also refused
    // B, from an anchor at the window's start that put `TAD` in `[0, L]`, where no schedule can.
    //
    // Mutations that redden it: replace the dependence test with a load test — refuse whenever
    // the clock is read and unanchored (A, B and C are refused); drop the dependence test's
    // refusal (D runs); add a `t_start` anchor back (B is refused).
    let decisions = [12.0, 36.0];
    let obs = [6.0, 20.0, 40.0];
    let with_init = |src: &str| {
        let twin = src.replacen("[odes]\n", "[odes]\n  init(central) = 50.0\n", 1);
        assert_ne!(twin, src, "the twin must add the init line");
        assert_eq!(
            twin.replacen("  init(central) = 50.0\n", "", 1),
            src,
            "the twin must differ by the init line alone"
        );
        twin
    };
    let cell_a = ODE_TAD_IN_CONDITION_ONLY.to_string();
    let cell_b = with_init(&cell_a);
    let cell_c = cell_a.replacen("if (TAD > 5.0)", "if (TAD < 5.0)", 1);
    assert_ne!(cell_c, cell_a, "C must flip A's comparison");
    assert_eq!(
        cell_c.replacen("if (TAD < 5.0)", "if (TAD > 5.0)", 1),
        cell_a,
        "C must differ from A by the comparison alone"
    );
    let cell_d = with_init(&cell_c);

    // A. Tight tolerances; measured worst rel 4.183e-16. At DEFAULT tolerances the same
    // comparison reads 4.750e-8, the two engines' noise across the arm switch at TAD = 5.
    let (adaptive, static_pred) = tad_oracle_cell(&cell_a, &decisions, vec![0, 1], &obs, vec![]);
    assert_eq!(
        adaptive[0], 0.0,
        "A: the pre-dose read is the empty compartment, not NaN (got {})",
        adaptive[0]
    );
    let vs_static = worst_rel(&adaptive, &static_pred, "A: adaptive vs predict()");
    assert!(
        vs_static <= 1e-12,
        "A: a TAD read confined to a condition that cannot move the derivative must run, and \
         track predict(): worst rel {vs_static:e} (adaptive={adaptive:?}, \
         static={static_pred:?})"
    );

    // B. Measured bit-identical: the two engines take the same arm on the same segments.
    let (adaptive, static_pred) = tad_oracle_cell(&cell_b, &decisions, vec![0, 1], &obs, vec![]);
    assert!(
        adaptive[0] > 40.0,
        "B: the pre-dose read must carry the init baseline (got {})",
        adaptive[0]
    );
    for (i, (&a, &s)) in adaptive.iter().zip(static_pred.iter()).enumerate() {
        assert!(a.is_finite(), "B: adaptive[{i}] = {a} is not finite");
        assert_eq!(
            a.to_bits(),
            s.to_bits(),
            "B: `TAD > 5` takes its unanchored (else) arm on every TAD a schedule can give the \
             window, so the run must equal predict() exactly: adaptive={adaptive:?}, \
             static={static_pred:?}"
        );
    }

    // C. Measured bit-identical, as B.
    let (adaptive, static_pred) = tad_oracle_cell(&cell_c, &decisions, vec![0, 1], &obs, vec![]);
    for (i, (&a, &s)) in adaptive.iter().zip(static_pred.iter()).enumerate() {
        assert!(a.is_finite(), "C: adaptive[{i}] = {a} is not finite");
        assert_eq!(
            a.to_bits(),
            s.to_bits(),
            "C: on an empty compartment the arms cannot differ, so `TAD < 5` must run and equal \
             predict() exactly: adaptive={adaptive:?}, static={static_pred:?}"
        );
    }

    // D. With the check disabled the run is 0.108 off `predict()` at its worst read (measured).
    let err = tad_refusal(&cell_d, &decisions, vec![0, 1], &obs, vec![]).unwrap_or_else(|| {
        panic!(
            "D: with `central` at 50, every TAD ≤ 0 takes the ×2 arm and the unanchored clock the \
             else arm — the window must be refused"
        )
    });
    assert!(
        err.contains("`TAD`") && err.contains("(6, 12]"),
        "D: must name the slot and the window: {err}"
    );
    assert!(
        err.contains("The derivative it computes there changes when that clock is anchored"),
        "D: must say what was measured: {err}"
    );
}

#[test]
fn adaptive_tad_rhs_refuses_when_the_controller_never_doses() {
    // #1534 review, finding 2 — message-table row 7, the cell the first six rows all assumed
    // away. Every decision holds, so no dose is ever given and the clock is never anchored.
    //
    // The refusal is right, but two of the message's claims are only true BECAUSE they are
    // stated conditionally, and this test is what pins that:
    //
    //   * `predict()` on the very same record returns `[0.0, NaN, NaN]` — measured below rather
    //     than asserted from the doc comment. So "the static engines anchor such a window at
    //     the first dose in the whole record" is FALSE here; the message says "Where the record
    //     contains a dose, …", and that qualifier is load-bearing.
    //   * "has not been decided yet" would imply a dose is coming. None is. The message says
    //     "not decided yet, and may never decide".
    //
    // The driver cannot distinguish the two cases: the guard fires on the FIRST segment, before
    // any decision has been observed. So the repair is in the wording, and this test asserts the
    // wording holds on the cell that falsifies the unconditional version.
    //
    // Mutation that reddens it: restore either unconditional sentence.
    let err = tad_refusal_error(
        ODE_TAD_NO_IIV,
        &[12.0, 36.0],
        vec![],
        &[6.0, 20.0, 40.0],
        vec![],
    );

    assert!(err.contains("`TAD`"), "must name the slot: {err}");
    assert!(
        err.contains("Where the record contains a dose"),
        "the anchoring claim must be conditional — this record has no dose at all: {err}"
    );
    assert!(
        err.contains("not decided yet, and may never decide"),
        "must not imply a dose is still coming: {err}"
    );
    assert!(
        err.contains("a controller that never doses leaves a model reading `TAD` unanchored"),
        "must name this very case: {err}"
    );

    // The measurement the conditional rests on: the static engine has no anchor here either.
    let model = parse_model_string(ODE_TAD_NO_IIV).expect("parse TAD-reading ODE model");
    let static_pop = population(vec![subj("1", vec![6.0, 20.0, 40.0], vec![])]);
    let preds = predict(&model, &static_pop, &model.default_params).unwrap();
    let static_pred: Vec<f64> = preds.iter().map(|p| p.pred).collect();
    assert_eq!(
        static_pred[0], 0.0,
        "the t=0 read is the empty compartment: {static_pred:?}"
    );
    assert!(
        static_pred[1..].iter().all(|v| v.is_nan()),
        "predict() has no anchor on a dose-free record either, so the message must not claim \
         it does: {static_pred:?}"
    );
}

// ============ #1534 review round 2 / #1535: what the gate must and must not claim ============

/// `TAD` mentioned only in a branch the unanchored window does not take, PLUS a second state
/// that diverges for its own reasons (`X' = 0.5·X²` from `X(0) = 1` blows up at t = 2).
///
/// Every conjunct of the pre-causation guard held on this model — the segment's state came
/// back non-finite (in `X`), the `TAD` slot was `NaN`, and `pk_reads_tad()` was true — while
/// the `TAD` line never ran and `central` was finite throughout. It is the false positive the
/// counterfactual re-solve was added to remove, and the one the dependence test (#1535) must
/// not bring back: no anchor moves a derivative the window never evaluates.
const ODE_TAD_PLUS_DIVERGENT_STATE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central, X])
[odes]
  init(X) = 1.0
  if (TIME > 20.0) {
    d/dt(central) = -(CL / V) * central * (1.0 + 0.01 * TAD)
  } else {
    d/dt(central) = -(CL / V) * central
  }
  d/dt(X) = 0.5 * X * X
[error_model]
  DV ~ proportional(PROP)
"#;

/// `TAD` consumed by a **comparison**, with a non-zero starting state so the arm it chooses is
/// observable. `NaN < 5.0` is false, so the unanchored window silently takes the else arm and
/// the state stays finite — which, until #1535 gated the refusal on the derivative's dependence
/// on the clock rather than on a non-finite state, meant the guard never fired. `min`/`max`
/// desugar to the same shape (`if (a <= b) …`), so `min(TAD, 24)` yields 24 there by the same
/// route (`ODE_TAD_IN_A_MIN`).
///
/// `init(central) = 50` and an observation at t=0 are both load-bearing. Without the non-zero
/// start the pre-dose window is identically zero and both arms give a zero derivative, so the
/// choice cannot show — that degeneracy is why round 1 mis-read this class as harmless. Without
/// the t=0 observation the static engine starts integrating at the first scored record, which
/// is #936 and a different confound.
const ODE_TAD_IN_A_COMPARISON: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  init(central) = 50.0
  if (TAD < 5.0) { d/dt(central) = -(CL / V) * central * 3.0 }
  else           { d/dt(central) = -(CL / V) * central }
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

/// `ODE_TAD_IN_A_COMPARISON`'s other spelling, the #1535 issue's second regression cell: `min`
/// desugars to `if (a <= b) a else b`, so an unanchored `min(TAD, 24)` silently yields 24.
const ODE_TAD_IN_A_MIN: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  init(central) = 50.0
  d/dt(central) = -(CL / V) * central * (1.0 + 0.01 * min(TAD, 24.0))
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

/// The autonomous control for the two fixtures above: same `init`, same schedule, no clock
/// anywhere.
/// It agrees with `predict()` exactly, which is what makes the divergence below attributable to
/// `TAD` rather than to the non-zero start or to #936's integration origin.
const ODE_INIT50_AUTONOMOUS: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  init(central) = 50.0
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

#[test]
fn adaptive_unrelated_divergence_is_not_blamed_on_the_dose_clock() {
    // #1534 review round 2 (finding A), corrected in round 3 (finding C). A compartment that
    // blows up on its own, on a model that merely *mentions* `TAD` in a branch the unanchored
    // window does not take. At `68c9f5c0`, before the causation conjunct: `Err` naming `TAD`,
    // and the advice it gave was actively harmful — adding a base dose silences the message
    // while `X` stays just as divergent.
    //
    // **What this test can and cannot assert.** Round 2 called this "a run that works" and
    // pinned `worst_rel(adaptive, predict()) <= 1e-9`. That bound held for a reason neither
    // engine should be proud of: once any state goes non-finite the solve stops advancing
    // *every* state, so `central` freezes at its post-dose value and BOTH engines report the
    // frozen number. Measured with `predict()` alone on the autonomous twin of this model
    // (one dose of 100 at t=12, obs at 1/20/40): `[0.0, 99.99999999935972,
    // 99.99999999935972]`, against `[0.0, 44.93327596489944, 6.081716633303086]` for the
    // `X' = 0·X` control, which is what the closed form gives. That is an engine defect
    // (#1539), not something this guard fixes, and the frozen-replay verifier cannot see it
    // either, because the two engines share the solver.
    //
    // So this test asserts only what it can honestly see: **this guard does not refuse the
    // run, and no error names the dose clock**. It deliberately does NOT assert that the
    // numbers are right — they are not — and it does not pin agreement with `predict()`,
    // which would lock in the frozen output as if it were correct.
    //
    // #1570 review, row 2 — the second schedule holds at 12 and doses only at 36. `X` diverges
    // in `(0, 12]`, so the window `(12, 36]` starts from the solver's pad (`X` non-finite,
    // `central` frozen at 0) with the clock still unanchored. At `494a7af3` that window was
    // refused — "the segment (12, 36] integrated to a non-finite state, and the [odes] RHS reads
    // `TAD`" — pinning `X`'s earlier divergence on the clock; a state with a non-finite component
    // is no longer probed. With the check disabled that run equals `predict()` bit for bit:
    // frozen in both engines, as above.
    //
    // Mutations that redden it: drop the dependence test from `unanchored_dose_clock_error` —
    // refuse whenever the clock is read and unanchored (it was the `resolve_with_finite_clock`
    // conjunct before #1535) — for both schedules; probe `u_start` whatever its components,
    // for the hold schedule.
    let model =
        parse_model_string(ODE_TAD_PLUS_DIVERGENT_STATE).expect("parse divergent-state model");
    let obs = vec![0.0, 6.0, 20.0, 40.0];
    let decisions = vec![12.0, 36.0];
    let pop = population(vec![subj("1", obs, vec![])]);

    // Measured before #1539 (`verify: false` and `verify: true` alike), finite, frozen and
    // wrong: `[0.0, 0.0, 99.99999999935972, 199.99999999807915]` for doses at 12 and 36 (the
    // value at t=20 should be ~44.93, as the control above shows), and
    // `[0.0, 0.0, 0.0, 99.99999999935972]` for the hold schedule. Since #1539 the rows past
    // the divergence are `NaN` instead.
    for (schedule, dose_idx) in [
        ("doses at 12 and 36", vec![0, 1]),
        ("a hold at 12 and a dose at 36", vec![1]),
    ] {
        for verify in [false, true] {
            let opts = AdaptiveSimulateOptions {
                seed: Some(1),
                decision_times: decisions.clone(),
                verify,
                ..Default::default()
            };
            let res = simulate_adaptive(
                &model,
                &pop,
                &model.default_params,
                1,
                || dose_at_decisions(dose_idx.clone()),
                &opts,
            );
            // #1539: the rows past the divergence are now `NaN`, and the frozen-replay
            // verifier no longer counts `NaN == NaN` as agreement — so `verify: true` is
            // refused, by the *verifier* and never by this guard.
            match (verify, res) {
                (false, Err(e)) => panic!(
                    "a divergence in a compartment that never evaluates TAD must not be \
                     refused by this guard ({schedule}, verify=false): {e}"
                ),
                (false, Ok(_)) => {}
                (true, Ok(_)) => panic!(
                    "the frozen-replay verifier must not certify a run whose rows are NaN \
                     because a state diverged ({schedule}, #1539)"
                ),
                (true, Err(e)) => {
                    let e = format!("{e}");
                    assert!(
                        e.contains("frozen-schedule replay verification failed")
                            && e.contains("cannot confirm it")
                            && !e.contains("has no referent"),
                        "verify=true must be refused by the replay verifier, not by the dose \
                         clock guard ({schedule}): {e}"
                    );
                }
            }
        }

        // And the negative half, stated separately so it cannot pass by the run simply
        // erroring: whatever this model does, no message may name the dose clock for it.
        let opts = AdaptiveSimulateOptions {
            seed: Some(1),
            decision_times: decisions.clone(),
            verify: false,
            ..Default::default()
        };
        let res = simulate_adaptive(
            &model,
            &pop,
            &model.default_params,
            1,
            || dose_at_decisions(dose_idx.clone()),
            &opts,
        );
        if let Err(e) = res {
            let e = format!("{e}");
            assert!(
                !e.contains("has no referent"),
                "the dose clock is not the cause here and must not be named ({schedule}): {e}"
            );
        }
    }
}

/// #1539's model, run through the adaptive driver: `central` decays with `k = CL/V = 0.1` and
/// is observed; `X` is decoupled, unobserved, and runs to its pole at t = 2 when `x_rate` is
/// `0.5` (`X = 1/(1 − 0.5t)`). `x_rate = 0.0` is the control, which integrates both.
fn divergent_x_model(x_rate: &str) -> CompiledModel {
    parse_model_string(&format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central, X])
[odes]
  init(X) = 1.0
  d/dt(central) = -(CL / V) * central
  d/dt(X) = {x_rate} * X * X
[error_model]
  DV ~ proportional(PROP)
"#
    ))
    .expect("parse divergent-X model")
}

/// T6 (#1539), the headline: an adaptive run on a model whose unobserved state diverges.
///
/// Before #1539 the solver stopped every state at the pole and padded the rest of each segment
/// with the frozen last state, so `central` was served as a finite `99.999…` at t = 20 and 40
/// (closed form `100·e^{−0.1(t−12)}` = 44.93, 6.08), the warning blamed stiffness, and
/// `verify: true` returned `Ok` — the reactive driver and the frozen replay share the solver,
/// so they froze identically and "agreed".
///
/// Now: the rows past the pole are `NaN`, the warning says a state went non-finite, and the
/// verifier refuses to certify rows it cannot confirm. The control (`0·X`) agrees with the
/// closed form, carries no solver warning, and still verifies.
///
/// Engine: the reactive driver (`ode_predictions_adaptive_impl` → `integrate_segment` →
/// `integrate_dense_g`, `T = f64`) and, under `verify`, the frozen replay
/// (`ode_predictions_with_extra_breaks`) — callers of one solver, not two references. Mutations
/// that redden it: revert the tail pad to the last state (the `is_nan` and verify legs); count
/// `NaN == NaN` as agreement again in `verify_adaptive_frozen_replay` (the verify leg); drop
/// the diverged clause from the warning.
#[test]
fn adaptive_run_with_a_diverged_state_serves_nan_and_does_not_verify() {
    let obs = vec![1.0, 20.0, 40.0];
    let pop = population(vec![subj("1", obs, vec![])]);
    let run = |x_rate: &str, verify: bool| {
        let model = divergent_x_model(x_rate);
        let opts = AdaptiveSimulateOptions {
            seed: Some(1),
            decision_times: vec![12.0],
            verify,
            ..Default::default()
        };
        simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
    };

    let control = run("0.0", false).expect("the control runs");
    let diverged = run("0.5", false).expect("verify: false returns the rows");
    let rows =
        |r: &AdaptiveSimulationResult| r.trajectories.iter().map(|s| s.ipred).collect::<Vec<_>>();
    let (c, d) = (rows(&control), rows(&diverged));
    assert_eq!(c.len(), 3);
    assert_eq!(d.len(), 3);

    // Before the dose both are an empty compartment, integrated, and bit-identical.
    assert_eq!(
        d[0].to_bits(),
        c[0].to_bits(),
        "pre-dose row: {d:?} vs {c:?}"
    );
    assert!(d[0].is_finite());
    // The control is the closed form (η on CL has variance 1e-10, so ≤ ~3e-5 relative).
    for (i, t) in [(1, 20.0), (2, 40.0)] {
        let want = 100.0 * (-0.1f64 * (t - 12.0)).exp();
        assert!(
            c[i].is_finite() && (c[i] - want).abs() <= 1e-3 * want,
            "control {c:?}"
        );
        // The diverged run serves nothing it did not integrate.
        assert!(
            d[i].is_nan(),
            "t={t} must be NaN, not a frozen value: {d:?}"
        );
    }

    // The warning names the cause and gives no solver advice for it.
    let solver_msg = diverged
        .warnings
        .iter()
        .find(|w| w.contains("W_ODE_SOLVER_DIAGNOSTICS"))
        .unwrap_or_else(|| panic!("a solver warning: {:?}", diverged.warnings));
    assert!(
        solver_msg.contains("had a state become non-finite"),
        "{solver_msg}"
    );
    assert!(solver_msg.contains("simulate_adaptive"), "{solver_msg}");
    assert!(!solver_msg.contains("rodas5p"), "{solver_msg}");
    assert!(
        !control
            .warnings
            .iter()
            .any(|w| w.contains("W_ODE_SOLVER_DIAGNOSTICS")),
        "{:?}",
        control.warnings
    );

    // The verify straddle: the control verifies, the diverged run does not.
    run("0.0", true).expect("the control verifies");
    let e = match run("0.5", true) {
        Ok(_) => panic!("the verifier must not certify NaN rows as agreement (#1539)"),
        Err(e) => e,
    };
    assert!(
        e.contains("frozen-schedule replay verification failed") && e.contains("cannot confirm it"),
        "{e}"
    );
}

/// T6's second cell (#1539, inherited from #1570 review row 1): a pre-dose `[odes]` RHS that
/// reads `TAD^(-0.5)`, which no anchor a schedule can supply makes finite. #1570 lets it run
/// unrefused, and it returns the `NaN` rows `predict()` returns too — `[50, NaN, NaN, NaN]`,
/// measured. Until #1539 `verify: true` returned `Ok` on it, because `NaN == NaN` counted as
/// agreement. The rows are unchanged here; only the verdict is.
///
/// Engine: as above. Mutation that reddens it: count `NaN == NaN` as agreement again.
#[test]
fn adaptive_run_with_nan_rows_from_the_rhs_does_not_verify() {
    let model = parse_model_string(&ODE_INIT50_AUTONOMOUS.replace(
        "d/dt(central) = -(CL / V) * central",
        "d/dt(central) = -(CL / V) * central * (1.0 + 0.01 * TAD^(-0.5))",
    ))
    .expect("parse TAD^(-0.5) model");
    let pop = population(vec![subj("1", vec![0.0, 6.0, 20.0, 40.0], vec![])]);
    let opts = |verify| AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: vec![12.0, 36.0],
        verify,
        ..Default::default()
    };

    let res = simulate_adaptive(
        &model,
        &pop,
        &model.default_params,
        1,
        fixed_bolus,
        &opts(false),
    )
    .expect("verify: false runs unrefused");
    let rows: Vec<f64> = res.trajectories.iter().map(|s| s.ipred).collect();
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0], 50.0, "{rows:?}");
    assert!(rows[1..].iter().all(|v| v.is_nan()), "{rows:?}");

    match simulate_adaptive(
        &model,
        &pop,
        &model.default_params,
        1,
        fixed_bolus,
        &opts(true),
    ) {
        Ok(_) => panic!("NaN rows must not verify as agreement (#1539)"),
        Err(e) => assert!(e.contains("cannot confirm it"), "{e}"),
    }
}

#[test]
fn adaptive_tad_consumed_by_a_comparison_is_refused() {
    // #1535 — the issue's two regression cells, until then row 8b of the #1534 message table,
    // "caught only by the verifier". The unanchored `NaN` never reaches the state here: it is
    // consumed by `TAD < 5.0` (false, so the else arm runs) or by `min(TAD, 24)` (which yields
    // 24). The run used to return a number that depended on which way the comparison fell —
    // measured at `68c9f5c0` with `verify: false`: `[50, 27.44…, 19.02…, 31.28…]` and
    // `[50, 23.76…, 48.43…, 71.53…]`, against `predict()`'s `[50, 8.26…, 16.76…, 31.14…]` and
    // `[50, 28.96…, 50.56…, 71.75…]` on the realized schedule. With the default `verify: true`
    // only the frozen-replay verifier caught it, as a symptom that never named the clock.
    //
    // The dependence test sees it directly: on `(0, 12]` the derivative changes when `TAD` is
    // anchored. Both verifier settings must return the typed refusal — the driver refuses
    // before the verifier runs, so `verify: true` no longer reports the symptom.
    //
    // Mutation that reddens it: restore the finite-state early return
    // (`if u.iter().all(is_finite) { return None }`) — every arm then runs again.
    let obs = vec![0.0, 6.0, 20.0, 40.0];
    let decisions = vec![12.0, 36.0];

    // The control first: the same fixture with the clock removed agrees with `predict()`
    // exactly, so nothing below is an artifact of `init(central) = 50` or of where the static
    // engine starts integrating (#936).
    let (control, control_static) =
        tad_oracle_cell(ODE_INIT50_AUTONOMOUS, &decisions, vec![0, 1], &obs, vec![]);
    for (i, (&a, &s)) in control.iter().zip(control_static.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            s.to_bits(),
            "autonomous control must agree exactly at obs {i}: {control:?} vs {control_static:?}"
        );
    }

    let pop = population(vec![subj("1", obs.clone(), vec![])]);
    for (shape, src) in [
        ("if (TAD < 5)", ODE_TAD_IN_A_COMPARISON),
        ("min(TAD, 24)", ODE_TAD_IN_A_MIN),
    ] {
        let model = parse_model_string(src).expect("parse comparison-TAD model");
        for verify in [false, true] {
            let opts = AdaptiveSimulateOptions {
                seed: Some(1),
                decision_times: decisions.clone(),
                verify,
                ..Default::default()
            };
            let err = simulate_adaptive(
                &model,
                &pop,
                &model.default_params,
                1,
                || dose_at_decisions(vec![0, 1]),
                &opts,
            )
            .expect_err("a clock consumed by a comparison must be refused, not returned");
            let err = format!("{err}");
            let cell = format!("`{shape}`, verify: {verify}");
            assert!(
                err.contains("`TAD`") && err.contains("(0, 12]"),
                "{cell}: must name the slot and the window: {err}"
            );
            assert!(
                err.contains(
                    "The derivative it computes there changes when that clock is anchored"
                ),
                "{cell}: must say what was measured: {err}"
            );
            assert!(
                err.contains("an `if`, `min` or `max` on it picks a side"),
                "{cell}: must say how a finite run can depend on the clock: {err}"
            );
            assert!(
                err.contains("The observation at t=6 is read off that segment"),
                "{cell}: must name the read in the window: {err}"
            );
            assert!(
                !err.contains("non-finite state"),
                "{cell}: the state stayed finite — the non-finite opener would be false: {err}"
            );
            assert!(
                !err.contains("frozen-schedule replay verification failed"),
                "{cell}: the driver refuses before the verifier runs: {err}"
            );
        }
    }

    // The advice is true for this class too: a controller dose at the subject's first record
    // (t=0) leaves no unanchored window, and the run is valid and equal to `predict()`.
    let (fixed, fixed_static) = tad_oracle_cell(
        ODE_TAD_IN_A_COMPARISON,
        &[0.0, 12.0, 36.0],
        vec![0, 1, 2],
        &obs,
        vec![],
    );
    // Measured worst rel 5.1e-12, at t=20: the RHS switches arm at `TAD` = 5 (t = 5 and 17),
    // which neither engine breaks on, so the gap is their two step sequences' error across it
    // at `ode_reltol = 1e-12` — as in the untaken-branch cell at default tolerances. The bound
    // carries 20× headroom, against the factor-3.3 gap the unanchored run had at t=6.
    let vs_static = worst_rel(
        &fixed,
        &fixed_static,
        "advice applied: adaptive vs predict()",
    );
    assert!(
        vs_static <= 1e-10,
        "with the advice applied the run must match predict(): worst rel {vs_static:e} \
         (adaptive={fixed:?}, static={fixed_static:?})"
    );
}

/// Finite where `predict()` evaluates the clock (pre-dose `TAD` ≤ 0), overflowing where the
/// pre-#1535 counterfactual's `t_start` stand-in put it (`TAD ∈ [0, L]` for a window of length
/// `L`). With a long
/// pre-treatment baseline, `exp(TAD)` is the cheapest spelling of a class that is not
/// contrived: any RHS whose clock term is bounded going backwards and not forwards.
const ODE_EXP_TAD: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central * (1.0 + 1e-12 * exp(TAD))
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

/// Two causes in one segment: a genuine arithmetic `TAD` read in `central`, and a second state
/// that runs away on its own. A whole-state finiteness test on the pre-#1535 counterfactual
/// cleared the clock here — `X` is still `inf` under any anchor — although `TAD` is one of the
/// two causes. The dependence test compares derivatives component by component: `X`'s is the
/// same under every anchor, `central`'s is `NaN` unanchored and finite anchored.
const ODE_TAD_AND_DIVERGENT_STATE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central, X])
[odes]
  init(X) = 1.0
  d/dt(central) = -(CL / V) * central * (1.0 + 0.01 * TAD)
  d/dt(X) = 0.5 * X * X
[error_model]
  DV ~ proportional(PROP)
"#;

#[test]
fn adaptive_tad_rhs_refuses_over_a_pre_dose_window_longer_than_the_clock_survives() {
    // #1534 review round 3, finding D1 — why the pre-#1535 causation check needed TWO
    // stand-ins, kept as the long-window row of the verdict map.
    //
    // That check asked a counterfactual: would this segment be finite with the clock anchored?
    // The anchor it substitutes is not neutral. `t_start` puts `TAD` in `[0, L]`, and `L` here
    // is 800 h — longer than any `TAD` this model sees after a dose, and the opposite sign
    // from `predict()`, whose pre-dose `TAD` is ≤ 0. `exp(800)` overflows, so with only that
    // stand-in the counterfactual diverged too and the clock was cleared: measured at
    // `62b9a6ef`, `verify: false` returned `Ok([NaN, NaN])` and the default verifier gave its
    // symptom message (`reactive=NaN, static=54.881163607201266`).
    //
    // The dependence test anchors at the window's end only (#1570 review, row 1), which puts
    // `TAD` in `[-800, 0]` where `exp(TAD)` never overflows, and refuses this window on its
    // `NaN`-versus-finite derivative. The anchor's own pin is
    // `adaptive_tad_dependence_is_judged_at_the_window_end_anchor`.
    //
    // Mutation that reddens it: drop the refusal.
    //
    // #936: the record at t=0 is what makes `(0, L]` a real pre-dose window. Without it the
    // subject's first event is the dose at L itself, the run's origin, and nothing is
    // integrated before it — the run is then valid, not refused.
    let err = tad_refusal_error(
        ODE_EXP_TAD,
        &[800.0, 812.0],
        vec![0, 1],
        &[0.0, 806.0, 820.0],
        vec![],
    );
    assert!(err.contains("`TAD`"), "must name the slot: {err}");
    assert!(
        err.contains("(0, 800]"),
        "must name the refused window: {err}"
    );

    // The 600 h control. Under the counterfactual it was the other side of a straddle — the
    // `t_start` stand-in alone repaired it, since `exp(600)` is finite — so the window length
    // moved the cell across that check's boundary. The dependence test has no such boundary,
    // and refuses both.
    let control = tad_refusal_error(
        ODE_EXP_TAD,
        &[600.0, 612.0],
        vec![0, 1],
        &[0.0, 606.0, 620.0],
        vec![],
    );
    assert!(
        control.contains("(0, 600]"),
        "the shorter window must be refused too: {control}"
    );
}

#[test]
fn adaptive_tad_rhs_refuses_when_the_clock_is_one_of_two_causes() {
    // #1534 review round 3, finding D2 — the clock as one of two causes. `central` reads `TAD`
    // arithmetically and breaks because of it; `X` runs away on its own and is non-finite
    // under every anchor. A whole-state finiteness test on the pre-#1535 counterfactual said
    // "still broken, not the clock's fault" and went silent, although the clock IS one of the
    // two causes: measured at `62b9a6ef`, `verify: false` returned `Ok([0.0, NaN, NaN, NaN])`.
    //
    // The per-component counterfactual that fixed it took its "comes back finite" evidence
    // from the solver's frozen tail (#1539): with that issue's NaN tail pad applied, this test
    // went red (re-measured at `2a6076af`). The dependence test re-solves nothing — at
    // `u_start`, `central`'s derivative is `NaN` unanchored and finite anchored — and stays
    // refused under the same pad (measured on #1535).
    //
    // Mutation that reddens it: take the verdict from a re-solve of the segment again, under
    // #1539's NaN tail pad.
    let err = tad_refusal_error(
        ODE_TAD_AND_DIVERGENT_STATE,
        &[12.0, 36.0],
        vec![0, 1],
        &[0.0, 6.0, 20.0, 40.0],
        vec![],
    );
    assert!(err.contains("`TAD`"), "must name the slot: {err}");
    assert!(
        err.contains("(0, 12]"),
        "must name the refused window: {err}"
    );
    assert!(
        err.contains("The observation at t=6 is read off that segment"),
        "must name the read in the window: {err}"
    );
}

/// The mirror of `ODE_EXP_TAD`: `exp(-TAD)` is bounded where the `t_start` stand-in puts the
/// clock (`TAD ∈ [0, L]` ⇒ the term is ≤ 1) and overflows where `t_end` puts it
/// (`TAD ∈ [-L, 0]` ⇒ `exp(L)`). The two fixtures together were why the pre-#1535 causation
/// check ran BOTH stand-ins: each one alone was defeated by one of them. The dependence test,
/// anchored at the window's end only, skips the grid points where `exp(-TAD)` overflows and
/// compares the rest.
const ODE_EXP_NEG_TAD: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central * (1.0 + 1e-12 * exp(0.0 - TAD))
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

#[test]
fn adaptive_tad_rhs_refuses_a_long_window_whose_clock_term_survives_only_forwards() {
    // #1534 review round 3, finding D1 — the OTHER half of the pre-#1535 straddle.
    //
    // `..._longer_than_the_clock_survives` uses `exp(TAD)`, which the `t_start` stand-in
    // (`TAD ∈ [0, 800]`) overflows and `t_end` (`TAD ∈ [-800, 0]`) survives. This fixture is
    // its mirror, `exp(-TAD)`: `t_start` survives it and `t_end` overflows. Under the
    // counterfactual each needed the stand-in the other could do without. The dependence test
    // anchors at the window's end only (#1570 review, row 1): it skips the grid points where
    // `exp(-TAD)` overflows (`TAD < -709`) and refuses on the rest.
    //
    // Mutation that reddens it: drop the refusal.
    //
    // #936: the record at t=0 is what makes `(0, L]` a real pre-dose window. Without it the
    // subject's first event is the dose at L itself, the run's origin, and nothing is
    // integrated before it — the run is then valid, not refused.
    let err = tad_refusal_error(
        ODE_EXP_NEG_TAD,
        &[800.0, 812.0],
        vec![0, 1],
        &[0.0, 806.0, 820.0],
        vec![],
    );
    assert!(err.contains("`TAD`"), "must name the slot: {err}");
    assert!(
        err.contains("(0, 800]"),
        "must name the refused window: {err}"
    );
}

// ============ #1535: the dependence test's probe — the anchor and the grid ============
//
// The probe anchors the unanchored clock at the window's end — the earliest a first dose can
// land — so it judges a shape by the `TAD ≤ 0` a real schedule can give a pre-dose window, and
// never by the positive values no schedule can (#1570 review, row 1). Each cell below sits on
// one side of that line. All start from `init(central) = 50` (an empty compartment gives a zero
// derivative under every anchor) with a record at t=0, so a refused window is `(0, 12]`.

/// `TAD` clamped below at zero: `max(TAD, 0)` is `if (TAD >= 0) TAD else 0`. The unanchored
/// clock takes the else arm (0) — and so does every `TAD ≤ 0` a schedule can give the pre-dose
/// window, so the run already equals `predict()`, and is not refused.
const ODE_TAD_CLAMPED_BELOW: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  init(central) = 50.0
  d/dt(central) = -(CL / V) * central * (1.0 + 0.05 * max(TAD, 0.0))
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

/// The mirror: `min(TAD, 0)` equals its unanchored arm (0) only for `TAD ≥ 0`. On the `TAD ≤ 0`
/// a schedule gives the pre-dose window it is `TAD` itself, so the unanchored run is wrong, and
/// is refused.
const ODE_TAD_CLAMPED_ABOVE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  init(central) = 50.0
  d/dt(central) = -(CL / V) * central * (1.0 - 0.05 * min(TAD, 0.0))
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

/// A condition that holds only INSIDE the window: `TAD` between −8 and −6. Under the window-end
/// anchor the grid's interior points `TAD` = −7.5 and −6.75 fall in it; the endpoints (`TAD` =
/// −12 and 0) do not, and the unanchored clock takes the else arm everywhere. Once the dose lands
/// at 12, `predict()` takes the ×3 arm on `(4, 6)`, so the refusal is earned.
const ODE_TAD_IN_AN_INTERIOR_WINDOW: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  init(central) = 50.0
  if (TAD > -8.0 && TAD < -6.0) { d/dt(central) = -(CL / V) * central * 3.0 }
  else                           { d/dt(central) = -(CL / V) * central }
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

#[test]
fn adaptive_tad_dependence_is_judged_at_the_window_end_anchor() {
    // #1535 T6, as settled on #1570's review (row 1). The probe anchors the clock at the
    // window's end, the earliest a first dose can land, which gives the pre-dose window
    // `TAD ∈ [-L, 0]` — the only values any schedule can give it.
    //
    // - `min(TAD, 0)` is `TAD` there and 0 under the unanchored clock: the run is wrong, and is
    //   refused (measured with the check disabled: 0.31 off `predict()` at its worst read).
    // - `max(TAD, 0)` is 0 there, exactly as under the unanchored clock: the run already equals
    //   `predict()`, and must run. The first cut of #1535 also anchored at the window's start
    //   (`TAD ∈ [0, L]`, where no schedule can put it) and refused this cell.
    //
    // Mutations that redden it: anchor at `t_start` instead (both cells flip); add a `t_start`
    // anchor back (the `max` cell is refused).
    let decisions = [12.0, 36.0];
    let obs = [0.0, 6.0, 20.0, 40.0];
    let err = tad_refusal(ODE_TAD_CLAMPED_ABOVE, &decisions, vec![0, 1], &obs, vec![])
        .unwrap_or_else(|| {
            panic!(
                "`min(TAD, 0)` differs from its unanchored arm on every TAD < 0 a schedule can \
                 give the window: it must be refused"
            )
        });
    assert!(
        err.contains("`TAD`") && err.contains("(0, 12]"),
        "`min(TAD, 0)`: must name the slot and the window: {err}"
    );

    let (adaptive, static_pred) =
        tad_oracle_cell(ODE_TAD_CLAMPED_BELOW, &decisions, vec![0, 1], &obs, vec![]);
    // Measured worst rel 4.3e-16.
    let vs_static = worst_rel(
        &adaptive,
        &static_pred,
        "`max(TAD, 0)`: adaptive vs predict()",
    );
    assert!(
        vs_static <= 1e-12,
        "`max(TAD, 0)` takes its unanchored arm on every TAD a schedule can give the window, so \
         it must run and match predict(): worst rel {vs_static:e} (adaptive={adaptive:?}, \
         static={static_pred:?})"
    );
}

#[test]
fn adaptive_tad_dependence_is_probed_inside_the_window_not_only_at_its_ends() {
    // #1535 T7, re-pinned on #1570's review (row 1). `TAD` between −8 and −6 matters only
    // inside the window: at both of its ends the unanchored run and the anchored one take the
    // same arm, and only the grid's interior points −7.5 and −6.75 separate them. The refusal is
    // earned: with the check disabled the run is 0.49 off `predict()` at its worst read.
    //
    // The control is the same window moved to the positive side, `TAD` between 2 and 4. No
    // schedule gives the pre-dose window a positive clock, so that run already equals
    // `predict()` and must run (measured 1.4e-10: the engines' step error across the arm switch
    // at t = 14 and 16, which neither breaks on). The first cut of #1535 refused it.
    //
    // Mutations that redden it: probe the endpoints only (`k ∈ {0, 16}`: the first cell runs);
    // add a `t_start` anchor back (the control is refused).
    let decisions = [12.0, 36.0];
    let obs = [0.0, 6.0, 20.0, 40.0];
    let err = tad_refusal(
        ODE_TAD_IN_AN_INTERIOR_WINDOW,
        &decisions,
        vec![0, 1],
        &obs,
        vec![],
    )
    .unwrap_or_else(|| {
        panic!(
            "`TAD` in (-8, -6) is seen only at the grid's interior points: probing the window's \
             ends alone lets it run unrefused"
        )
    });
    assert!(
        err.contains("`TAD`") && err.contains("(0, 12]"),
        "must name the slot and the window: {err}"
    );

    let positive = ODE_TAD_IN_AN_INTERIOR_WINDOW.replacen(
        "TAD > -8.0 && TAD < -6.0",
        "TAD > 2.0 && TAD < 4.0",
        1,
    );
    assert_ne!(
        positive, ODE_TAD_IN_AN_INTERIOR_WINDOW,
        "the control moves the window to the positive side"
    );
    let (adaptive, static_pred) = tad_oracle_cell(&positive, &decisions, vec![0, 1], &obs, vec![]);
    let vs_static = worst_rel(
        &adaptive,
        &static_pred,
        "`TAD` in (2, 4): adaptive vs predict()",
    );
    assert!(
        vs_static <= 1e-8,
        "no schedule gives the pre-dose window a positive clock, so `TAD` in (2, 4) must run and \
         match predict(): worst rel {vs_static:e} (adaptive={adaptive:?}, static={static_pred:?})"
    );
}

// ---------------------------------------------------------------------------------------------
// #936 — the reactive run's integration origin.
//
// The state is `init(...)` and does not evolve until the origin
// `min(subject_integration_start(base record), first realized controller dose)`, which is the
// frozen-replay static subject's (base ∪ ledger) `subject_integration_start` — NONMEM's
// first-record convention (#573). Every cell below starts off-zero with a non-fixed-point
// `init`, so a phantom `[0, origin]` window is visible in the value. The closed form
// `50·e^{-k(t − t₀)} + Σ 100·e^{-k(t − tᵈ)}` (k = CL/V = 0.1) is computed outside every engine.
//
// Engines, per fixture: the reactive driver (`ode_predictions_adaptive_impl`); the constant-path
// verifier (`ode_predictions_with_extra_breaks`) and the TV/IOV verifier
// (`adaptive_frozen_replay_tv`), both run by `verify: true`; and `predict()` / `predict_iov` on
// the realized ledger as the static reference.
// ---------------------------------------------------------------------------------------------

const K_936: f64 = 5.0 / 50.0;

/// The closed form of `ODE_INIT50_AUTONOMOUS` (and of `ODE_TV_INIT` with `init = init0`): the
/// `init` baseline decays from the origin `t0`, and every 100-unit bolus at or before `t`
/// decays from its own time.
fn origin_closed_form(init0: f64, t0: f64, doses: &[f64], obs: &[f64]) -> Vec<f64> {
    obs.iter()
        .map(|&t| {
            let mut c = init0 * (-K_936 * (t - t0)).exp();
            for &d in doses.iter().filter(|&&d| d <= t) {
                c += 100.0 * (-K_936 * (t - d)).exp();
            }
            c
        })
        .collect()
}

/// One reactive run with a controller dosing 100 into cmt 1 at the decisions in `dose_idx`.
/// Returns the IPREDs and, per decision, the `(t, central)` the controller read.
fn origin_run(
    pop: &Population,
    model: &CompiledModel,
    decisions: &[f64],
    dose_idx: Vec<usize>,
    verify: bool,
) -> Result<(Vec<f64>, Vec<(f64, f64)>), String> {
    let reads = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(f64, f64)>::new()));
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: decisions.to_vec(),
        verify,
        ..Default::default()
    };
    let res = simulate_adaptive(
        model,
        pop,
        &model.default_params,
        1,
        || {
            let reads = reads.clone();
            let idx = dose_idx.clone();
            move |ctx: &ControllerCtx| {
                reads.lock().unwrap().push((ctx.t, ctx.state[0]));
                if idx.contains(&ctx.decision_index) {
                    vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
                } else {
                    vec![DoseAction::Hold]
                }
            }
        },
        &opts,
    )?;
    let reads = reads.lock().unwrap().clone();
    Ok((res.trajectories.iter().map(|t| t.ipred).collect(), reads))
}

/// `predict()` on the base regimen plus the realized controller doses.
fn origin_static(model: &CompiledModel, subject: Subject) -> Vec<f64> {
    predict(model, &population(vec![subject]), &model.default_params)
        .unwrap()
        .iter()
        .map(|p| p.pred)
        .collect()
}

/// Worst relative gap, finiteness asserted per element (see `worst_rel`); `what` names the
/// engine side so a failure says which leg drifted.
fn assert_rel(got: &[f64], want: &[f64], tol: f64, what: &str) {
    let worst = worst_rel(got, want, what);
    assert!(
        worst <= tol,
        "{what}: worst rel {worst:e} > {tol:e} (got={got:?}, want={want:?})"
    );
}

#[test]
fn adaptive_origin_at_first_record_matches_closed_form() {
    // T1 (#936), cell A. Dose-free base, first record the obs at 6, first dose at the
    // decision at 12. The origin is 6: nothing evolves on `[0, 6]`, so the read at 6 is the
    // `init` baseline 50. Before the fix the driver integrated from a hard-coded t=0 and read
    // 27.4406 = 50·e^{-0.6} there (measured at `4c09cef1`), and the verifier refused the run.
    //
    // Mutation that reddens it: revert the driver's origin — seed `break_times` with 0.0 AND
    // drop the `started` skip (the pre-#936 driver). Seeding with 0.0 alone is an equivalent
    // mutation: the un-started walk still inserts `t_base0` and integrates nothing before it.
    let model = parse_model_string(ODE_INIT50_AUTONOMOUS).unwrap();
    let decisions = [12.0, 36.0];
    let obs = [6.0, 20.0, 40.0];
    let pop = population(vec![subj("1", obs.to_vec(), vec![])]);

    let (driver, _) = origin_run(&pop, &model, &decisions, vec![0], false).expect("driver");
    let closed = origin_closed_form(50.0, 6.0, &[12.0], &obs);
    assert_rel(&driver, &closed, 1e-10, "driver vs closed form");
    assert_eq!(
        driver[0], 50.0,
        "the read at the origin is the init baseline"
    );

    let (checked, _) =
        origin_run(&pop, &model, &decisions, vec![0], true).expect("the verifier agrees");
    let static_pred = origin_static(
        &model,
        subj(
            "1",
            obs.to_vec(),
            vec![DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0)],
        ),
    );
    assert_rel(&checked, &static_pred, 1e-10, "driver vs predict()");
}

#[test]
fn adaptive_hold_before_first_record_does_not_evolve_state() {
    // T2 (#936), cell B. A hold decision at t=0, before the first record (obs at 6). The
    // controller must read the un-evolved `init` there, and the trajectory must equal cell A:
    // a decision is not a record and does not move the origin.
    //
    // Mutation that reddens it: drop the `started` skip, keeping the new seed. Decision 0 is
    // then the first break and `[0, 6]` integrates — the read at 6 falls to 27.44.
    let model = parse_model_string(ODE_INIT50_AUTONOMOUS).unwrap();
    let decisions = [0.0, 12.0];
    let obs = [6.0, 20.0, 40.0];
    let pop = population(vec![subj("1", obs.to_vec(), vec![])]);

    let (driver, reads) = origin_run(&pop, &model, &decisions, vec![1], true).expect("run");
    assert_eq!(
        reads[0],
        (0.0, 50.0),
        "the decision before the origin reads the init state"
    );
    // The reading at the decision at 12 is 50·e^{-0.6}, six hours after the origin — not
    // 50·e^{-1.2}, which a t=0 origin would give.
    let want12 = 50.0 * (-K_936 * 6.0).exp();
    assert!(
        ((reads[1].1 - want12) / want12).abs() <= 1e-10,
        "the controller's reading at t=12 must be decayed from the origin (6), got {:?}, want \
         {want12}",
        reads[1]
    );
    let closed = origin_closed_form(50.0, 6.0, &[12.0], &obs);
    assert_rel(&driver, &closed, 1e-10, "driver vs closed form");
}

#[test]
fn adaptive_controller_dose_before_first_record_is_the_origin() {
    // T3 (#936), cell C. The controller doses at 2, before the first record (obs at 6). That
    // dose is part of the static subject's record, so IT is the origin: the baseline decays
    // from 2, not from 6, and the read at 6 is 150·e^{-0.4} = 100.548.
    //
    // Mutation that reddens it: make `started` ignore controller doses (origin = `t_base0`
    // only — the issue body's suggested fix). The dose at 2 then lands on a state that does
    // not evolve until 6. It also reddens `degenerate_oracle_matches_static_predict` (dose at
    // 0, first obs at 6).
    let model = parse_model_string(ODE_INIT50_AUTONOMOUS).unwrap();
    let decisions = [2.0, 12.0];
    let obs = [6.0, 20.0, 40.0];
    let pop = population(vec![subj("1", obs.to_vec(), vec![])]);

    let (driver, _) = origin_run(&pop, &model, &decisions, vec![0], true).expect("run");
    let closed = origin_closed_form(50.0, 2.0, &[2.0], &obs);
    assert_rel(&driver, &closed, 1e-10, "driver vs closed form");
    let static_pred = origin_static(
        &model,
        subj(
            "1",
            obs.to_vec(),
            vec![DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0)],
        ),
    );
    assert_rel(&driver, &static_pred, 1e-10, "driver vs predict()");
}

#[test]
fn adaptive_base_dose_after_zero_is_the_origin() {
    // T4 (#936), cell E. A pre-scheduled base bolus at 4 precedes the first obs (6), so the
    // base record starts at 4 and the read at 6 is 150·e^{-0.2} = 122.8096.
    //
    // Mutation that reddens it: compute the origin from the observations only (ignore base
    // doses) — the baseline then holds until 6.
    let model = parse_model_string(ODE_INIT50_AUTONOMOUS).unwrap();
    let decisions = [12.0];
    let obs = [6.0, 20.0];
    let base = vec![DoseEvent::new(4.0, 100.0, 1, 0.0, false, 0.0)];
    let pop = population(vec![subj("1", obs.to_vec(), base.clone())]);

    let (driver, _) = origin_run(&pop, &model, &decisions, vec![0], true).expect("run");
    let closed = origin_closed_form(50.0, 4.0, &[4.0, 12.0], &obs);
    assert_rel(&driver, &closed, 1e-10, "driver vs closed form");
    let mut doses = base;
    doses.push(DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0));
    let static_pred = origin_static(&model, subj("1", obs.to_vec(), doses));
    assert_rel(&driver, &static_pred, 1e-10, "driver vs predict()");
}

#[test]
fn adaptive_tv_origin_matches_predict_and_the_verifier_sees_it() {
    // T5 (#936), cell TV-A. Cell A's shape on `ODE_TV_INIT`: `init = 10·CRCL`, the first
    // record carries CRCL 8 (init 80), later ones 5. This routes the verifier to
    // `adaptive_frozen_replay_tv`, which before the fix shared the driver's hard-coded t=0 —
    // so `verify: true` returned Ok on a wrong trajectory (43.9049 against 80, measured at
    // `4c09cef1`). Each engine side is asserted separately and named in its message:
    //
    // - DRIVER side (`verify: false` vs closed form and `predict()`): reddened by reverting the
    //   driver's origin (seed 0.0 + no `started` skip).
    // - VERIFIER side (`verify: true` must be Ok on the fixed driver): reddened by reverting
    //   `adaptive_frozen_replay_tv`'s seed to 0.0 alone — the replay then disagrees with the
    //   correct driver.
    let model = parse_model_string(ODE_TV_INIT).unwrap();
    let decisions = [12.0, 36.0];
    let obs = [6.0, 20.0, 40.0];
    let crcl = |v: f64| HashMap::from([("CRCL".to_string(), v)]);
    let with_cov = |mut s: Subject| -> Subject {
        s.covariates = crcl(8.0);
        s.obs_covariates = vec![crcl(8.0), crcl(5.0), crcl(5.0)];
        s.dose_covariates = s.doses.iter().map(|_| crcl(5.0)).collect();
        s
    };
    let base = with_cov(subj("1", obs.to_vec(), vec![]));
    assert!(
        base.has_tv_covariates(),
        "the subject must take the TV path"
    );
    let pop = population(vec![base]);

    let (driver, _) = origin_run(&pop, &model, &decisions, vec![0], false).expect("driver");
    let closed = origin_closed_form(80.0, 6.0, &[12.0], &obs);
    assert_rel(&driver, &closed, 1e-9, "DRIVER side: driver vs closed form");
    let static_pred = origin_static(
        &model,
        with_cov(subj(
            "1",
            obs.to_vec(),
            vec![DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0)],
        )),
    );
    assert_rel(
        &driver,
        &static_pred,
        1e-9,
        "DRIVER side: driver vs predict()",
    );

    let checked = origin_run(&pop, &model, &decisions, vec![0], true);
    assert!(
        checked.is_ok(),
        "VERIFIER side: adaptive_frozen_replay_tv must agree with the correct driver: {:?}",
        checked.err()
    );
}

// `ODE_IOV` with a non-fixed-point `init`, so the origin is visible under IOV too.
const ODE_IOV_INIT50: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  kappa KAPPA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = 50.0
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;

#[test]
fn adaptive_iov_origin_matches_predict_iov() {
    // T6 (#936). IOV routes the verifier to `adaptive_frozen_replay_tv` too. Decisions at
    // 6 / 12 / 24 (a hold at 6, doses at 12 and 24) and obs at 6 / 12 / 24, so every record
    // has an occasion and `predict_iov`'s obs-derived groups are the identity. The first
    // record is at 6: the old driver integrated `[0, 6]` and read ~27.4 there.
    //
    // DRIVER side: `verify: false` vs `predict_iov` (reddened by reverting the driver origin).
    // VERIFIER side: `verify: true` must be Ok (reddened by reverting
    // `adaptive_frozen_replay_tv`'s seed alone).
    let model = parse_model_string(ODE_IOV_INIT50).unwrap();
    let decisions = vec![6.0, 12.0, 24.0];
    let obs = decisions.clone();
    let seed = 1u64;
    let pop = population(vec![subj("1", obs.clone(), vec![])]);

    let (driver, _) = origin_run(&pop, &model, &decisions, vec![1, 2], false).expect("driver");

    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let z_eta: Vec<f64> = (0..model.n_eta).map(|_| rng.sample(normal)).collect();
    let eta_bsv: Vec<f64> = (&model.default_params.omega.chol
        * nalgebra::DVector::from_column_slice(&z_eta))
    .iter()
    .copied()
    .collect();
    let omega_iov = model.default_params.omega_iov.as_ref().expect("omega_iov");
    let kbase = crate::sim::adaptive::subject_kappa_base_seed(seed, "1", 1);
    let kappas: Vec<Vec<f64>> = (0..decisions.len())
        .map(|g| {
            let z: Vec<f64> = (0..model.n_kappa)
                .map(|k| crate::sim::adaptive::kappa_standard_normal(kbase, g, k))
                .collect();
            (&omega_iov.chol * nalgebra::DVector::from_column_slice(&z))
                .iter()
                .copied()
                .collect()
        })
        .collect();
    assert!(
        kappas.iter().any(|k| k[0].abs() > 1e-6),
        "κ must be non-zero, else the IOV leg is vacuous"
    );

    let static_doses = vec![
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let mut static_subject = subj("1", obs.clone(), static_doses);
    static_subject.occasions = vec![0, 1, 2];
    static_subject.dose_occasions = vec![1, 2];
    let preds = crate::pk::predict_iov(
        &model,
        &static_subject,
        &model.default_params.theta,
        &eta_bsv,
        &kappas,
    );
    assert_eq!(
        driver[0], 50.0,
        "DRIVER side: the read at the origin is init"
    );
    assert_rel(&driver, &preds, 1e-9, "DRIVER side: driver vs predict_iov");

    let checked = origin_run(&pop, &model, &decisions, vec![1, 2], true);
    assert!(
        checked.is_ok(),
        "VERIFIER side: adaptive_frozen_replay_tv must agree with the correct driver: {:?}",
        checked.err()
    );
}

#[test]
fn adaptive_origin_straddle_record_at_zero_vs_first_record_later() {
    // T7 (#936) — the straddle that keeps T1 from being a tautology. Cell D has a record at
    // t=0, so its origin is 0 before and after the fix and it must match `predict()` as
    // before (measured at `4c09cef1`: [50, 27.4406, 51.6997], identical to predict()). Cell A
    // is the same schedule without that record: its origin moves to 6, and the read at t=6
    // must differ between the two cells by more than 1 (27.44 vs 50). With the origin still
    // pinned to 0 the two cells would agree at t=6.
    let model = parse_model_string(ODE_INIT50_AUTONOMOUS).unwrap();
    let decisions = [12.0];
    let dose = vec![DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0)];

    let obs_d = [0.0, 6.0, 20.0];
    let pop_d = population(vec![subj("1", obs_d.to_vec(), vec![])]);
    let (d, _) = origin_run(&pop_d, &model, &decisions, vec![0], true).expect("cell D");
    let d_static = origin_static(&model, subj("1", obs_d.to_vec(), dose));
    assert_rel(&d, &d_static, 1e-12, "cell D (origin 0) vs predict()");
    assert_rel(
        &d,
        &origin_closed_form(50.0, 0.0, &[12.0], &obs_d),
        1e-10,
        "cell D (origin 0) vs closed form",
    );

    let obs_a = [6.0, 20.0];
    let pop_a = population(vec![subj("1", obs_a.to_vec(), vec![])]);
    let (a, _) = origin_run(&pop_a, &model, &decisions, vec![0], true).expect("cell A");
    assert!(
        (a[0] - d[1]).abs() > 1.0,
        "the straddle: the read at t=6 must depend on whether a record sits at t=0 \
         (A={}, D={})",
        a[0],
        d[1]
    );
}

#[test]
fn adaptive_tad_rhs_refuses_an_observation_free_window_after_a_reset_origin() {
    // #936 review R1. The refusal's state-carry sentence ("No observation is read off that
    // segment, but the state integrated there carries into every later read") lost its only
    // end-to-end test when this PR converted #1151's dose-free-window cell into a positive
    // (its origin moved to the dose). This cell restores a real observation-free window: an
    // EVID=3 reset at t=0 is the subject's first record, so the origin is 0; the obs are at
    // 20 and 40 and the first dose is at the decision at 12, so `(0, 12]` is integrated
    // under an unanchored `TAD` with no read inside it.
    //
    // Mutation that reddens it: delete the tail "…but the state integrated there carries
    // into every later read" from the message (review mutation S2).
    let model = parse_model_string(ODE_TAD_NO_IIV).unwrap();
    let mut s = subj("1", vec![20.0, 40.0], vec![]);
    s.reset_times = vec![0.0];
    let pop = population(vec![s]);
    let err = origin_run(&pop, &model, &[12.0, 36.0], vec![0, 1], false)
        .expect_err("a reset-anchored origin before an unanchored window must be refused");
    assert!(err.contains("`TAD`"), "must name the slot: {err}");
    assert!(
        err.contains("(0, 12]"),
        "must name the refused window: {err}"
    );
    assert!(
        err.contains(
            "No observation is read off that segment, but the state integrated there carries \
             into every later read."
        ),
        "must say the poisoned state — not a readout — is what is refused: {err}"
    );
}

#[test]
fn adaptive_reset_as_first_record_is_the_origin() {
    // #936 review R3. An EVID=3 reset at t=4 is the subject's first record, so it is the
    // origin: `init` decays from 4. Hold at 0, bolus at 12, obs 6/20/40. The reset is in
    // `subject_integration_start`, the same as the static engine.
    //
    // Mutation that reddens it: drop `reset_times` from `subject_integration_start`'s fold.
    // Measured (#1535): the driver's origin moves to the first observation, 6, while the
    // frozen-replay verifier's static replay still starts at the reset, so the run itself
    // errors (`reactive=50, static=40.93653765390065` for the read at t=6) and the test dies
    // at `.expect("run")` — before the closed-form comparison below, which would also fail.
    let model = parse_model_string(ODE_INIT50_AUTONOMOUS).unwrap();
    let obs = [6.0, 20.0, 40.0];
    let mut s = subj("1", obs.to_vec(), vec![]);
    s.reset_times = vec![4.0];
    let pop = population(vec![s]);

    let (driver, _) = origin_run(&pop, &model, &[0.0, 12.0], vec![1], true).expect("run");
    let closed = origin_closed_form(50.0, 4.0, &[12.0], &obs);
    // Measured: 6.0e-13 against the closed form.
    assert_rel(
        &driver,
        &closed,
        1e-10,
        "driver vs closed form (origin at the reset)",
    );
    let mut st = subj(
        "1",
        obs.to_vec(),
        vec![DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0)],
    );
    st.reset_times = vec![4.0];
    let static_pred = origin_static(&model, st);
    assert_rel(&driver, &static_pred, 1e-10, "driver vs predict()");
}

#[test]
fn adaptive_auc_pass_holds_init_before_the_origin() {
    // #936 review R3. The AUC-target pass (`adaptive_window_signal_aucs`, dense solve) was
    // already on the static origin; after #936 the driver shares it. Base record obs 6/20/40,
    // decisions 0/6/12/24, one ledger bolus at 12. Window [0, 6] lies before the origin (6),
    // so the state is held at `init` = 50 there and its AUC is exactly 300.
    //
    // Mutation that reddens it: seed the dense solve at 0.0 instead of the subject's start
    // (the [0, 6] window would integrate a decay: 500·(1 − e^{-0.6}) = 225.6, not 300).
    let model = parse_model_string(ODE_INIT50_AUTONOMOUS).unwrap();
    let ode = model.ode_spec.as_ref().unwrap();
    let mut pk = [0.0; crate::types::MAX_PK_PARAMS];
    pk[crate::types::PK_IDX_CL] = 5.0;
    pk[crate::types::PK_IDX_V] = 50.0;
    pk[crate::types::PK_IDX_F] = 1.0;
    let base = subj("1", vec![6.0, 20.0, 40.0], vec![]);
    let ledger = vec![crate::sim::adaptive::DoseLedgerEntry {
        subject: "1".into(),
        draw: 0,
        sim: 0,
        dose_idx: 0,
        time: 12.0,
        amt: 100.0,
        cmt: 1,
        rate: 0.0,
        decision_idx: 2,
        rule_fired: "bolus".into(),
        observed_signals: Vec::new(),
        pre_state: None,
        post_state: None,
        f_applied: 1.0,
    }];
    let aucs = crate::ode::predictions::adaptive_window_signal_aucs(
        ode,
        &pk,
        &model.default_params.theta,
        &[0.0],
        &base,
        &[0.0, 6.0, 12.0, 24.0],
        &ledger,
        None,
        1,
    );
    let k = K_936;
    let want = [
        300.0,
        (50.0 / k) * (1.0 - (-k * 6.0).exp()),
        (50.0 / k) * ((-k * 6.0).exp() - (-k * 18.0).exp())
            + (100.0 / k) * (1.0 - (-k * 12.0).exp()),
    ];
    for (i, &a) in aucs.iter().enumerate() {
        assert!(a.is_finite(), "window {i}: AUC {a} is not finite");
    }
    assert_eq!(aucs.len(), 3);
    // The pre-origin window holds `init` exactly — a constant integrand, no trapezoid error.
    assert!(
        (aucs[0] - 300.0).abs() <= 1e-9,
        "window [0, 6] precedes the origin, so init is held: {aucs:?}"
    );
    // Trapezoid error on the decaying windows, measured at 1.8e-6 and 7.3e-6 relative; the
    // bound carries ~7× headroom and is 2 000× below the pre-origin mutation's effect.
    for i in 1..3 {
        let rel = ((aucs[i] - want[i]) / want[i]).abs();
        assert!(
            rel <= 5e-5,
            "window {i}: rel {rel:e} (aucs={aucs:?}, want={want:?})"
        );
    }
}

// ---- #1571: both entry points run `check_simulation_data` ----------------------
//
// The programmatic `simulate_adaptive` once skipped the data checks every other
// simulate entry point runs, so a model covariate absent from the data read as 0.0
// and the run returned `Ok` — with the frozen-replay verifier passing too, since it
// replays the same snapshots. Each fixture below is the issue's measured one: a
// 1-cpt IV ODE seeded at `init(central) = 50`, dose-free base, 100 at each decision.

/// `WT` reaches the prediction only through the `[scaling]` readout. Without the
/// check the whole trajectory read `[0, 0, 0, 0]`.
const ODE_1571_WT_SCALING: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  init(central) = 50.0
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central * (WT / 70)
[error_model]
  DV ~ proportional(PROP)
"#;

/// `WT` reaches the prediction only through `CL`. Without the check `CL = 0`, so
/// there was no elimination at all: `[50, 50, 150, 250]`.
const ODE_1571_WT_CL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * (WT / 70)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  init(central) = 50.0
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

/// `TAD` outside `[odes]` is an ordinary covariate, not the solver's clock. Without
/// the check it read as 0, identical to the clock-free control.
const ODE_1571_TAD_SCALING: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  init(central) = 50.0
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central * (1 + 0.01 * TAD)
[error_model]
  DV ~ proportional(PROP)
"#;

/// The issue's population, with `cov` carried as a constant data column when given.
fn pop_1571(cov: Option<&str>) -> Population {
    let mut s = subj("1", vec![0.0, 6.0, 20.0, 40.0], vec![]);
    let mut pop_names = Vec::new();
    if let Some(name) = cov {
        s.covariates = HashMap::from([(name.to_string(), 70.0)]);
        s.obs_covariates = vec![HashMap::from([(name.to_string(), 70.0)]); 4];
        pop_names.push(name.to_string());
    }
    let mut pop = population(vec![s]);
    pop.covariate_names = pop_names;
    pop
}

fn run_1571(
    src: &str,
    cov: Option<&str>,
    verify: bool,
) -> Result<AdaptiveSimulationResult, String> {
    let model = parse_model_string(src).expect("parse");
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: vec![12.0, 36.0],
        verify,
        ..Default::default()
    };
    simulate_adaptive(
        &model,
        &pop_1571(cov),
        &model.default_params,
        1,
        fixed_bolus,
        &opts,
    )
}

/// Assert the straddle for one model on the programmatic entry point, under both
/// `verify` settings: with `cov` as a data column the run is `Ok`; without it the
/// run is the `Err` every other simulate entry point returns, naming `cov`.
fn assert_programmatic_straddle(src: &str, cov: &str, want: &str) {
    for verify in [false, true] {
        let ok = run_1571(src, Some(cov), verify);
        assert!(
            ok.is_ok(),
            "verify={verify}: with {cov} in the data the run must succeed, got {:?}",
            ok.err()
        );
        let err = run_1571(src, None, verify).expect_err(&format!(
            "verify={verify}: {cov} absent from the data must be an Err, not a silent 0.0"
        ));
        assert!(
            err.contains(cov) && err.contains(want),
            "verify={verify}: got: {err}"
        );
    }
}

#[test]
fn programmatic_rejects_scaling_covariate_absent_from_data() {
    assert_programmatic_straddle(ODE_1571_WT_SCALING, "WT", "not found in data");
}

#[test]
fn programmatic_rejects_individual_parameter_covariate_absent_from_data() {
    assert_programmatic_straddle(ODE_1571_WT_CL, "WT", "not found in data");
}

#[test]
fn programmatic_rejects_tad_outside_odes_absent_from_data() {
    assert_programmatic_straddle(ODE_1571_TAD_SCALING, "TAD", "solver-injected built-in");
}

#[test]
fn programmatic_and_spec_entry_points_return_the_same_missing_covariate_error() {
    // One helper behind both entry points (#1571): the same model and data must
    // fail with the same message on either path, so a future edit to one entry
    // point's checks cannot drift from the other's.
    let model = parse_model_string(ODE_1571_WT_SCALING).expect("parse");
    let pop = pop_1571(None);
    let programmatic = run_1571(ODE_1571_WT_SCALING, None, true).expect_err("programmatic");
    let spec = simulate_adaptive_from_spec(
        &model,
        &pop,
        &model.default_params,
        1,
        &simple_titration_spec(),
        &AdaptiveSimulateOptions::default(),
    )
    .expect_err("from_spec");
    assert!(
        spec.contains("WT") && spec.contains("not found in data"),
        "got: {spec}"
    );
    assert_eq!(programmatic, spec);
    // ...and the spec path's positive side: with WT present it runs.
    let with_wt = simulate_adaptive_from_spec(
        &model,
        &pop_1571(Some("WT")),
        &model.default_params,
        1,
        &simple_titration_spec(),
        &AdaptiveSimulateOptions::default(),
    );
    assert!(with_wt.is_ok(), "got {:?}", with_wt.err());
}

/// `ODE_1571_WT_SCALING`'s structure with a covariate-selected error model (#658):
/// `FREE` picks the endpoint. The selector column is supplied, so every data check
/// passes and the `Selected` reject is the one that must fire.
const ODE_1571_SELECTED: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_TOTAL   ~ 0.04
  sigma PROP_UNBOUND ~ 0.09
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  init(central) = 50.0
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  if (FREE == 0) {
    DV ~ proportional(PROP_TOTAL)
  } else {
    DV ~ proportional(PROP_UNBOUND)
  }
[covariates]
  FREE continuous
"#;

#[test]
fn both_entry_points_reject_a_selected_error_model_through_the_shared_helper() {
    // #658 was pinned only on `reject_selected_error_for_adaptive` directly, so deleting
    // it from `check_adaptive_model_data` (the helper both entry points now share, #1571)
    // passed every test (review of #1585, finding 2). Held here through each public entry
    // point, with the single-endpoint twin of the same model and data as the straddle:
    // it must run, so the `Err` is the `Selected` reject and nothing else.
    let want = "covariate-selected `[error_model]`";
    let selected = parse_model_string(ODE_1571_SELECTED).expect("parse selected");
    let single_src = ODE_1571_SELECTED.replace(
        "  if (FREE == 0) {\n    DV ~ proportional(PROP_TOTAL)\n  } else {\n    DV ~ proportional(PROP_UNBOUND)\n  }",
        "  DV ~ proportional(PROP_TOTAL)",
    );
    assert_ne!(single_src, ODE_1571_SELECTED, "the twin must differ");
    let single = parse_model_string(&single_src).expect("parse single");
    let pop = pop_1571(Some("FREE"));
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: vec![12.0, 36.0],
        ..Default::default()
    };
    let spec = simple_titration_spec();
    let spec_opts = AdaptiveSimulateOptions::default();

    let err = simulate_adaptive(
        &selected,
        &pop,
        &selected.default_params,
        1,
        fixed_bolus,
        &opts,
    )
    .expect_err("simulate_adaptive must reject a Selected error model");
    assert!(err.contains(want), "simulate_adaptive: got: {err}");
    let err = simulate_adaptive_from_spec(
        &selected,
        &pop,
        &selected.default_params,
        1,
        &spec,
        &spec_opts,
    )
    .expect_err("simulate_adaptive_from_spec must reject a Selected error model");
    assert!(
        err.contains(want),
        "simulate_adaptive_from_spec: got: {err}"
    );

    let ok = simulate_adaptive(&single, &pop, &single.default_params, 1, fixed_bolus, &opts);
    assert!(ok.is_ok(), "simulate_adaptive twin: {:?}", ok.err());
    let ok =
        simulate_adaptive_from_spec(&single, &pop, &single.default_params, 1, &spec, &spec_opts);
    assert!(
        ok.is_ok(),
        "simulate_adaptive_from_spec twin: {:?}",
        ok.err()
    );

    // The CHANGELOG's order claim: with the selector column itself missing, the data
    // checks run first, so the programmatic path names `FREE`, not the #658 restriction.
    let err = simulate_adaptive(
        &selected,
        &pop_1571(None),
        &selected.default_params,
        1,
        fixed_bolus,
        &opts,
    )
    .expect_err("a missing selector column must be rejected");
    assert!(
        err.contains("FREE") && err.contains("not found in data") && !err.contains(want),
        "missing selector: got: {err}"
    );
}

/// Depot → central with `ALAG1` on the depot only, for the #1588 tie fixtures.
fn ode_depot_alag1(lag: f64) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
  theta TVLAG({lag}, 0.0, 10.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  KA    = TVKA
  ALAG1 = TVLAG
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA*depot
  d/dt(central) = KA*depot - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-11
"#
    )
}

/// **The adaptive driver wipes a dose an `SS=1` record reset** (#1588). It shares the static
/// walker's two passes (`reseed_prescheduled_states_at`, then the bolus pass over the growing
/// `shadow` list), so it inherited the static walker's row-order defect: a bolus row before a
/// co-timed `SS=1` row applied after the reset (ID 1, lag 0: +127 %), and at lag 2 a first
/// `SS=1` dose's arrival re-loaded its own trough over a later record's seed (ID 9: −48 %).
/// Those figures are from #1588's tree. Since #1275 (`e00ba374`) only the bolus pass's gate
/// is load-bearing here; removing it measures a peak |rel| of 104 % / 26.1 % / 27.2 % on
/// IDs 1 / 9 / 11 (at `d36f944e`), and removing the reseed gate alone moves nothing.
/// A controller dose into central at 25 (after the tie) exercises the gate's index space over the shadow
/// list: base doses then the injected one, which is live. The frozen-replay verifier runs
/// (`verify = true`), and the oracle is the static `predict` on the live-only twin plus the
/// controller dose — the static walker is pinned against NONMEM separately
/// (`tests/ss_reset_tie_nonmem_anchor.rs`).
#[test]
fn adaptive_driver_wipes_a_dose_an_ss_record_reset() {
    let ss = |t: f64, amt: f64| DoseEvent::new(t, amt, 1, 0.0, true, 12.0);
    let central = |t: f64| DoseEvent::new(t, 100.0, 2, 0.0, false, 0.0);
    let obs = vec![11.5, 12.5, 15.0, 21.0, 26.0, 30.0];
    // (label, lag, base regimen, live-only twin)
    let fixtures = [
        (
            "ID 1: bolus central, then SS=1",
            0.0,
            vec![central(10.0), ss(10.0, 100.0)],
            vec![ss(10.0, 100.0)],
        ),
        (
            "ID 9: SS=1 100 at 10, SS=1 200 at 11",
            2.0,
            vec![ss(10.0, 100.0), ss(11.0, 200.0)],
            vec![ss(11.0, 200.0)],
        ),
        (
            "ID 11: SS=1 100, then SS=1 200 (co-timed)",
            2.0,
            vec![ss(10.0, 100.0), ss(10.0, 200.0)],
            vec![ss(10.0, 200.0)],
        ),
    ];
    let mut worst = 0.0_f64;
    for (label, lag, base, twin) in fixtures {
        let model = parse_full_model(&ode_depot_alag1(lag)).unwrap().model;
        let pop = population(vec![subj("1", obs.clone(), base)]);
        let mut opts = AdaptiveSimulateOptions::default();
        opts.seed = Some(1);
        opts.decision_times = vec![25.0];
        // Into central: lagged controller dosing (the depot) is rejected at injection.
        let central_bolus = || |_: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 2 }];
        // η = 0 exactly, as the static `pred` oracle: `omega ~ 0.0` is PD-regularised to
        // Ω = 1e-8, which would draw η with sd 1e-4 (#1603).
        let mut params = model.default_params.clone();
        params.omega.chol.fill(0.0);
        params.omega.matrix.fill(0.0);
        let res = simulate_adaptive(&model, &pop, &params, 1, central_bolus, &opts)
            .expect("adaptive sim runs and passes the frozen-replay verifier");
        assert_eq!(res.ledger.len(), 1, "{label}: one controller dose at 25");

        let mut want_doses = twin;
        want_doses.push(central(25.0));
        let want_pop = population(vec![subj("1", obs.clone(), want_doses)]);
        let want = predict(&model, &want_pop, &model.default_params).unwrap();
        assert_eq!(res.trajectories.len(), want.len());
        for (got, w) in res.trajectories.iter().zip(&want) {
            assert!(got.ipred.is_finite(), "{label} t={}: non-finite", got.time);
            let rel = (got.ipred - w.pred).abs() / w.pred.abs();
            // Measured 4.1e-14 at η = 0 (1.9e-5 while η was drawn from the regularised Ω,
            // #1603); the bound is the ODE engines' 1e-10, and the smallest defect it guards
            // peaks at 26.1 % (ID 9; bolus-gate mutation, measured at `d36f944e`).
            assert!(
                rel < 1e-10,
                "adaptive, {label}, lag {lag}, t = {}: {} vs static live-only twin {} \
                 (rel {rel:.3e}) — adaptive drifted from the static twin: a wiped dose still \
                 arriving peaks at ≥ 26 % per fixture; a smaller drift points at η or the solver",
                got.time,
                got.ipred,
                w.pred
            );
            worst = worst.max(rel);
        }
    }
    println!("#1588 adaptive vs static live-only twin: worst rel {worst:.3e}");
}

// ─── #1148: the record a decision reads ──────────────────────────────────────────────
//
// One record supplies everything a decision at `t` reads — the controller's covariates,
// the PK behind its readouts, and the PK behind an injected dose's `F`: the latest data
// record at or before `t` (dose / EVID=2 / obs / EVID=3-4 reset), co-timed records in the
// dense engine's order `Reset < Dose < EVID=2 < Obs` with the last one winning; with no
// such record, the baseline covariates and the t=0 snapshot.
//
// Every fixture runs `simulate_adaptive` (the reactive driver, `ode_predictions_adaptive_impl`)
// with the default-on frozen-replay and snapshot verifiers. There is no `Dual2` / FD path on
// the adaptive driver. The verifiers re-derive the decision covariate through the SAME
// resolver the driver uses, so they pin plumbing, not this rule: the oracles here are the
// captured `ctx` itself, `predict()` on the realized doses, and closed forms.
//
// `ODE_TV_F`: `F = 0.8·CRCL/100`, `CL`/`V` constant (k = 0.1/h), so the IPRED an injected
// 100-unit bolus leaves behind reads back the `F` — and therefore the record — the decision
// resolved: `100·F·e^(−0.1·Δt)`.

// `ODE_TV_F` with a (numerically negligible) κ on F, so the subject routes through the IOV
// `decision_pk` build (`run_adaptive_population`) instead of the non-IOV readout. κ ~ 1e-6
// moves F by ~1e-6 relative, far inside every band below.
const ODE_TV_F_IOV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVF(0.8, 0.01, 1.0)
  omega ETA_CL ~ 1e-10
  kappa KAPPA_F ~ 1e-12
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
  F  = TVF * CRCL / 100.0 * exp(KAPPA_F)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

fn crcl_1148(v: f64) -> HashMap<String, f64> {
    HashMap::from([("CRCL".to_string(), v)])
}

/// What a decision saw: `(t, ctx.covariates["CRCL"], ctx.state[0])`.
type Seen1148 = Vec<(f64, f64, f64)>;

/// Run `simulate_adaptive` with a fixed 100-unit bolus at every decision, capturing the
/// controller's context. Default-on verifiers.
fn run_capture_1148(
    model: &CompiledModel,
    s: Subject,
    decisions: &[f64],
) -> (AdaptiveSimulationResult, Seen1148) {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Seen1148::new()));
    let make = {
        let seen = seen.clone();
        move || {
            let seen = seen.clone();
            move |ctx: &ControllerCtx| {
                let crcl = ctx.covariates.get("CRCL").copied().unwrap_or(f64::NAN);
                seen.lock().unwrap().push((ctx.t, crcl, ctx.state[0]));
                vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
            }
        }
    };
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];
    let opts = AdaptiveSimulateOptions {
        seed: Some(1148),
        decision_times: decisions.to_vec(),
        ..Default::default() // verify = true
    };
    let res = simulate_adaptive(model, &pop, &model.default_params, 1, make, &opts)
        .expect("adaptive sim runs and passes the default-on verifiers");
    let seen = seen.lock().unwrap().clone();
    (res, seen)
}

fn ipred_at_1148(res: &AdaptiveSimulationResult, t: f64) -> f64 {
    res.trajectories
        .iter()
        .find(|r| (r.time - t).abs() < 1e-9)
        .unwrap_or_else(|| panic!("no trajectory row at t={t}"))
        .ipred
}

fn crcl_seen_at_1148(seen: &Seen1148, t: f64) -> f64 {
    seen.iter()
        .find(|s| (s.0 - t).abs() < 1e-9)
        .unwrap_or_else(|| panic!("no decision at t={t}"))
        .1
}

/// `predict()` on the realized doses — each dose row carrying `dose_cov` — for an
/// independent engine: the dense walk, whose reset-row rule is NONMEM-anchored by
/// `reset_init_snapshot_J`.
fn predict_realized_1148(
    model: &CompiledModel,
    mut s: Subject,
    res: &AdaptiveSimulationResult,
    dose_cov: Option<Vec<HashMap<String, f64>>>,
) -> Vec<(f64, f64)> {
    s.doses = res
        .ledger
        .iter()
        .map(|e| DoseEvent::new(e.time, e.amt, 1, 0.0, false, 0.0))
        .collect();
    s.dose_occasions = vec![1; s.doses.len()];
    if let Some(dc) = dose_cov {
        s.dose_covariates = dc;
    }
    let mut pop = population(vec![s]);
    pop.covariate_names = vec!["CRCL".to_string()];
    predict(model, &pop, &model.default_params)
        .unwrap()
        .iter()
        .map(|p| (p.time, p.pred))
        .collect()
}

/// P1 / P3 fixture: baseline CRCL 90, obs@12 (90), EVID=3 reset@24 (30), obs@42 (30);
/// decisions at 0 and 36. The reset wipes the t=0 dose, so IPRED@42 is the t=36 dose alone.
fn reset_subject_1148() -> Subject {
    let mut s = subj("1", vec![12.0, 42.0], vec![]);
    s.covariates = crcl_1148(90.0);
    s.obs_covariates = vec![crcl_1148(90.0), crcl_1148(30.0)];
    s.reset_times = vec![24.0];
    s.reset_covariates = vec![crcl_1148(30.0)];
    s.reset_occasions = vec![1];
    s
}

/// Closed-form band: 8× the RK45 error control (default reltol 1e-4), the convention the
/// rest of this file uses. Measured worst realized error across T1–T6: 7.64e-4 (T6's
/// `ctx.state@18`, 28.3662 vs 28.3654), 6.07e-4 (T5), the rest ≤ 3.3e-4 — against a band of
/// 1.7e-2..2.3e-2 at these magnitudes, so ~30× headroom. Every stale value the tests exist
/// to reject sits ≥ 2.6 away (T3b's reset-row value; ≥ 4.6 for the rest), > 100× the band.
///
/// The review-r1 fixtures: 4.49e-4 (the carry past the final instant; the first-record rule
/// is 10.1 away) and 1.97e-4 (the IOV decision-window κ; the record's-window F is 7.7 away).
///
/// The `predict()` comparisons use 1e-9 relative: measured **0** (bit-identical) on every row.
fn band_1148(want: f64) -> f64 {
    8.0 * (1e-6 + 1e-4 * want.abs())
}

/// `100·F·e^(−0.1·Δt)` with `F = 0.8·CRCL/100`.
fn bolus_closed_form_1148(crcl: f64, dt: f64) -> f64 {
    100.0 * (0.8 * crcl / 100.0) * (-0.1 * dt).exp()
}

#[test]
fn adaptive_decision_after_a_reset_reads_the_reset_row() {
    // T1 (#1148, P1). The decision at 36 has the reset@24 as its latest record. Before the
    // fix the covariate side scanned obs / EVID=2 only (ctx CRCL = 90) and the PK side carried
    // the obs@12 snapshot (F = 0.72): IPRED@42 = 39.514452, 3.0× the reset row's 13.171479.
    // Both halves are asserted, so fixing only one of them still reddens this test.
    let model = parse_model_string(ODE_TV_F).expect("parse TV-F ODE model");
    let (res, seen) = run_capture_1148(&model, reset_subject_1148(), &[0.0, 36.0]);

    let want = bolus_closed_form_1148(30.0, 6.0); // 13.171479
    let got = ipred_at_1148(&res, 42.0);
    assert!(
        (got - want).abs() <= band_1148(want),
        "IPRED@42 = {got}: the injected dose's F must come from the reset row \
         (F = 0.24 → {want:.6}); the stale obs@12 snapshot gives F = 0.72 → {:.6}",
        bolus_closed_form_1148(90.0, 6.0)
    );
    assert_eq!(
        crcl_seen_at_1148(&seen, 36.0),
        30.0,
        "ctx CRCL at the t=36 decision must be the reset row's 30 (stale: the obs@12's 90)"
    );

    // Not an oracle for the rule: the t=36 dose row below is handed the covariate the rule
    // should pick (CRCL 30), so agreement restates that choice. The closed form above judges
    // the rule; this checks that the reactive engine, given the rule, reproduces the dense one.
    let preds = predict_realized_1148(
        &model,
        reset_subject_1148(),
        &res,
        Some(vec![crcl_1148(90.0), crcl_1148(30.0)]),
    );
    let p42 = preds.iter().find(|p| (p.0 - 42.0).abs() < 1e-9).unwrap().1;
    assert!(
        (got - p42).abs() <= 1e-9 * p42.abs(),
        "IPRED@42 {got} != predict() {p42} on the realized doses"
    );
}

#[test]
fn adaptive_iov_decision_after_a_reset_reads_the_reset_row() {
    // T2 (#1148, P3): the same fixture through the IOV `decision_pk` build, which resolved
    // the covariate with its own obs / EVID=2 scan (39.514469 before the fix).
    let model = parse_model_string(ODE_TV_F_IOV).expect("parse TV-F IOV ODE model");
    assert_eq!(model.n_kappa, 1);
    let (res, seen) = run_capture_1148(&model, reset_subject_1148(), &[0.0, 36.0]);

    let want = bolus_closed_form_1148(30.0, 6.0);
    let got = ipred_at_1148(&res, 42.0);
    assert!(
        (got - want).abs() <= band_1148(want),
        "IOV IPRED@42 = {got}, want the reset row's {want:.6} (stale: {:.6})",
        bolus_closed_form_1148(90.0, 6.0)
    );
    assert_eq!(crcl_seen_at_1148(&seen, 36.0), 30.0, "IOV ctx CRCL at t=36");
}

#[test]
fn adaptive_decision_co_timed_with_a_reset_reads_the_reset_row() {
    // T3a (#1148, P7): the decision sits exactly on the reset@24 — no other record there. The
    // reset is at-or-before the decision (`<=`), so it is the in-force record. obs@30 carries 30.
    let model = parse_model_string(ODE_TV_F).expect("parse TV-F ODE model");
    let mut s = subj("1", vec![12.0, 30.0], vec![]);
    s.covariates = crcl_1148(90.0);
    s.obs_covariates = vec![crcl_1148(90.0), crcl_1148(30.0)];
    s.reset_times = vec![24.0];
    s.reset_covariates = vec![crcl_1148(30.0)];
    s.reset_occasions = vec![1];
    let (res, seen) = run_capture_1148(&model, s, &[0.0, 24.0]);

    let want = bolus_closed_form_1148(30.0, 6.0);
    let got = ipred_at_1148(&res, 30.0);
    assert!(
        (got - want).abs() <= band_1148(want),
        "IPRED@30 = {got}, want {want:.6} (stale: {:.6})",
        bolus_closed_form_1148(90.0, 6.0)
    );
    assert_eq!(crcl_seen_at_1148(&seen, 24.0), 30.0, "ctx CRCL at t=24");
}

#[test]
fn adaptive_decision_at_a_reset_and_obs_tie_reads_the_obs() {
    // T3b (#1148): a reset and an observation share t=24 with different covariates. The dense
    // engine processes `Reset < … < Obs`, so the observation is the LAST record at 24 and the
    // one in force for a decision there.
    let model = parse_model_string(ODE_TV_F).expect("parse TV-F ODE model");
    let mut s = subj("1", vec![12.0, 24.0, 42.0], vec![]);
    s.covariates = crcl_1148(90.0);
    s.obs_covariates = vec![crcl_1148(90.0), crcl_1148(50.0), crcl_1148(50.0)];
    s.reset_times = vec![24.0];
    s.reset_covariates = vec![crcl_1148(30.0)];
    s.reset_occasions = vec![1];
    let (res, seen) = run_capture_1148(&model, s, &[0.0, 24.0]);

    assert_eq!(
        crcl_seen_at_1148(&seen, 24.0),
        50.0,
        "the obs@24 (50) wins the tie with the reset@24 (30)"
    );
    let want = bolus_closed_form_1148(50.0, 18.0);
    let got = ipred_at_1148(&res, 42.0);
    assert!(
        (got - want).abs() <= band_1148(want),
        "IPRED@42 = {got}, want the obs row's {want:.6} (reset row: {:.6})",
        bolus_closed_form_1148(30.0, 18.0)
    );
}

/// P2 / P4 fixture: a 50-unit BASE dose row at 24 carrying CRCL 30, no reset; obs@12 (90),
/// obs@42 (30); decisions at 0 and 36.
fn base_dose_subject_1148() -> Subject {
    let mut s = subj(
        "1",
        vec![12.0, 42.0],
        vec![DoseEvent::new(24.0, 50.0, 1, 0.0, false, 0.0)],
    );
    s.covariates = crcl_1148(90.0);
    s.obs_covariates = vec![crcl_1148(90.0), crcl_1148(30.0)];
    s.dose_covariates = vec![crcl_1148(30.0)];
    s
}

#[test]
fn adaptive_decision_after_a_base_dose_row_reads_that_row() {
    // T4 (#1148, P2 / P4): the base dose row at 24 is the decision@36's latest record. Before
    // the fix the non-IOV PK side already read it (IPRED@42 16.234894) but the covariate side
    // did not (ctx CRCL 90), and the IOV `decision_pk` skipped it on both (IPRED@42 42.577880):
    // two paths, identical data, 2.6× apart.
    let want = 72.0 * (-4.2f64).exp() // decision@0: F 0.72, 42 h
        + 50.0 * 0.24 * (-1.8f64).exp() // base dose@24: F 0.24, 18 h
        + bolus_closed_form_1148(30.0, 6.0); // decision@36: F 0.24, 6 h → 16.234894
    for (label, src) in [("non-IOV", ODE_TV_F), ("IOV", ODE_TV_F_IOV)] {
        let model = parse_model_string(src).expect("parse");
        let (res, seen) = run_capture_1148(&model, base_dose_subject_1148(), &[0.0, 36.0]);
        let got = ipred_at_1148(&res, 42.0);
        assert!(
            (got - want).abs() <= band_1148(want),
            "{label}: IPRED@42 = {got}, want {want:.6} (the IOV path skipping the dose row \
             gave 42.577880)"
        );
        assert_eq!(
            crcl_seen_at_1148(&seen, 36.0),
            30.0,
            "{label}: ctx CRCL at t=36 must be the dose row's 30 (stale: 90)"
        );
    }
}

#[test]
fn adaptive_decision_before_any_record_reads_the_baseline() {
    // T5 (#1148, P6): a decision at 0 with no record at or before it reads the baseline
    // covariates (CRCL 90) and the t=0 snapshot (F 0.72). Before the fix the non-IOV readout
    // took the FIRST record ahead (obs@12, F 0.48 → IPRED@12 14.457712) — the state seed's
    // rule, not a decision rule — while the IOV path read the baseline: a 1.5× split.
    let want = bolus_closed_form_1148(90.0, 12.0); // 21.686568
    for (label, src) in [("non-IOV", ODE_TV_F), ("IOV", ODE_TV_F_IOV)] {
        let model = parse_model_string(src).expect("parse");
        let mut s = subj("1", vec![12.0], vec![]);
        s.covariates = crcl_1148(90.0);
        s.obs_covariates = vec![crcl_1148(60.0)];
        let (res, seen) = run_capture_1148(&model, s, &[0.0]);
        assert_eq!(
            crcl_seen_at_1148(&seen, 0.0),
            90.0,
            "{label}: ctx CRCL at t=0"
        );
        let got = ipred_at_1148(&res, 12.0);
        assert!(
            (got - want).abs() <= band_1148(want),
            "{label}: IPRED@12 = {got}, want the baseline's {want:.6} (first record ahead: \
             {:.6})",
            100.0 * 0.48 * (-1.2f64).exp()
        );
    }
}

#[test]
fn adaptive_state_before_a_reset_is_integrated_under_the_reset_row() {
    // T6 (#1148, P9). `ODE_TV_COV`: CL = 5·CRCL/100. A decision@18 sits between obs@12 (90)
    // and a reset@24 (30), with no record between it and the reset. The segment (12, 18] is
    // governed by the record that terminates it (#1073) — the reset row — so the state the
    // decision reads is 100·e^(−0.09·12)·e^(−0.03·6) = 28.365403. Skipping the reset to the
    // next record ahead (obs@42) gave 23.693414 in the arm where obs@42 carries 60.
    //
    // The twin arm (obs@42 = 30) agrees under both rules: it is the control that makes the
    // pair straddle the gate, and the straddle is asserted, so the pair cannot quietly turn
    // tautological.
    let model = parse_model_string(ODE_TV_COV).expect("parse TV-cov ODE model");
    let a12 = 100.0 * (-0.09f64 * 12.0).exp();
    let want = a12 * (-0.03f64 * 6.0).exp();
    let skip_reset = |crcl42: f64| a12 * (-(0.1 * crcl42 / 100.0) * 6.0f64).exp();
    assert!((skip_reset(60.0) - want).abs() > 1.0 && (skip_reset(30.0) - want).abs() < 1e-12);
    for crcl42 in [60.0, 30.0] {
        let mut s = subj("1", vec![12.0, 42.0], vec![]);
        s.covariates = crcl_1148(90.0);
        s.obs_covariates = vec![crcl_1148(90.0), crcl_1148(crcl42)];
        s.reset_times = vec![24.0];
        s.reset_covariates = vec![crcl_1148(30.0)];
        s.reset_occasions = vec![1];
        let (res, seen) = run_capture_1148(&model, s.clone(), &[0.0, 18.0]);
        let state18 = seen
            .iter()
            .find(|x| (x.0 - 18.0).abs() < 1e-9)
            .expect("a decision at 18")
            .2;
        assert!(
            (state18 - want).abs() <= band_1148(want),
            "arm obs@42 = {crcl42}: ctx.state@18 = {state18}, want the reset-governed \
             {want:.6} (skipping the reset to obs@42 gives {:.6})",
            skip_reset(crcl42)
        );

        // No prediction moves: every IPRED still equals `predict()` on the realized doses.
        let preds = predict_realized_1148(&model, s, &res, None);
        assert_eq!(preds.len(), res.trajectories.len());
        for (traj, p) in res.trajectories.iter().zip(preds.iter()) {
            assert!(
                (traj.ipred - p.1).abs() <= 1e-9 * p.1.abs(),
                "arm {crcl42}: IPRED {} != predict() {} at t={}",
                traj.ipred,
                p.1,
                traj.time
            );
        }
    }
}

// `ODE_TV_F_IOV` with a real κ on F (ω_IOV² = 0.09), so a decision's window and the window of
// the record it reads can carry distinguishably different F.
const ODE_TV_F_IOV_WIDE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVF(0.8, 0.01, 1.0)
  omega ETA_CL ~ 1e-10
  kappa KAPPA_F ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
  F  = TVF * CRCL / 100.0 * exp(KAPPA_F)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

#[test]
fn adaptive_iov_decision_reads_its_record_under_the_decisions_own_kappa() {
    // #1148 review r1 finding 1. Under IOV a decision reads the in-force record's
    // covariates, but under the κ of the DECISION's window — not the record's own window.
    // T2 / T4 / T5 run `kappa ~ 1e-12` and cannot tell the two apart; this fixture can.
    //
    // T4's geometry: the base dose row@24 (CRCL 30) is the in-force record for the decision
    // at 36. With decisions at [0, 36] the dose row sits in window 0 and the decision opens
    // window 1, so the injected dose's F is 0.24·e^κ₁ (the rule) versus 0.24·e^κ₀ (the
    // record's own snapshot, `event_pk.dose[0]`).
    let model = parse_model_string(ODE_TV_F_IOV_WIDE).expect("parse wide-κ TV-F IOV model");
    let decisions = [0.0, 36.0];
    let (res, seen) = run_capture_1148(&model, base_dose_subject_1148(), &decisions);
    assert_eq!(crcl_seen_at_1148(&seen, 36.0), 30.0, "ctx CRCL at t=36");

    // The κ `run_capture_1148` drew (seed 1148, subject "1", sim 1).
    let kappas = reconstruct_kappas(&model, 1148, "1", decisions.len());
    let (k0, k1) = (kappas[0][0], kappas[1][0]);
    let dose36 = |k: f64| 100.0 * 0.24 * k.exp() * (-0.6f64).exp();
    let rest = 72.0 * k0.exp() * (-4.2f64).exp() // decision@0: baseline CRCL 90, window 0
        + 50.0 * 0.24 * k0.exp() * (-1.8f64).exp(); // base dose@24: its own row, window 0
    let want = rest + dose36(k1);
    let wrong = rest + dose36(k0);
    // Non-vacuity: the record's-window answer must sit well outside the band, or this
    // fixture would pass either rule — the defect T2 / T4 / T5 had.
    assert!(
        (want - wrong).abs() > 10.0 * band_1148(want),
        "κ₀ = {k0}, κ₁ = {k1} too close: the record's-window F would pass the band"
    );

    let got = ipred_at_1148(&res, 42.0);
    assert!(
        (got - want).abs() <= band_1148(want),
        "IPRED@42 = {got}: the t=36 dose's F must use the decision's window (κ₁ = {k1:.4} → \
         {want:.6}); the dose row's own window (κ₀ = {k0:.4}) gives {wrong:.6}"
    );
}

#[test]
fn adaptive_carry_past_the_final_instant_takes_the_last_co_timed_record() {
    // #1148 review r1 finding 2. The LOCF carry (`last_pk`) advances through the LAST record
    // of a co-timed group (`last_at`), not the first (`at`). Past the subject's final record
    // nothing ahead governs a segment, so the carry is what the walk integrates under — and a
    // decision there reads that state.
    //
    // `ODE_TV_COV` (k = 0.1·CRCL/100). The final instant, t=24, holds a base dose row
    // (50 units, CRCL 30) and an observation (CRCL 60) at the same time. Processing order is
    // Dose then Obs, so the obs is left in force: (24, 30] runs at k = 0.06, as `predict()`'s
    // dense walk does. Taking the first record (the dose row) would run it at k = 0.03.
    let model = parse_model_string(ODE_TV_COV).expect("parse TV-cov ODE model");
    let mut s = subj(
        "1",
        vec![12.0, 24.0],
        vec![DoseEvent::new(24.0, 50.0, 1, 0.0, false, 0.0)],
    );
    s.covariates = crcl_1148(90.0);
    s.obs_covariates = vec![crcl_1148(90.0), crcl_1148(60.0)];
    s.dose_covariates = vec![crcl_1148(30.0)];
    let (res, seen) = run_capture_1148(&model, s.clone(), &[0.0, 30.0]);

    // (0, 12] under obs@12 (k 0.09); (12, 24] under the dose row that terminates it
    // (k 0.03); +50 at 24; (24, 30] under the carry.
    let a24 = 100.0 * (-0.09f64 * 12.0).exp() * (-0.03f64 * 12.0).exp() + 50.0;
    let want = a24 * (-0.06f64 * 6.0).exp();
    let first_record = a24 * (-0.03f64 * 6.0).exp();
    let state30 = seen
        .iter()
        .find(|x| (x.0 - 30.0).abs() < 1e-9)
        .expect("a decision at 30")
        .2;
    assert!(
        (state30 - want).abs() <= band_1148(want),
        "ctx.state@30 = {state30}: past the final instant the carry must be the obs@24 \
         (CRCL 60 → {want:.6}); the co-timed dose row (CRCL 30) gives {first_record:.6}"
    );
    assert_eq!(crcl_seen_at_1148(&seen, 30.0), 60.0, "ctx CRCL at t=30");

    // The trajectory still agrees with `predict()` on the realized doses (base dose kept).
    let mut st = s;
    st.doses.extend(
        res.ledger
            .iter()
            .map(|e| DoseEvent::new(e.time, e.amt, 1, 0.0, false, 0.0)),
    );
    st.doses.sort_by(|a, b| a.time.total_cmp(&b.time));
    st.dose_occasions = vec![1; st.doses.len()];
    st.dose_covariates = st
        .doses
        .iter()
        .map(|d| crcl_1148(if d.time == 24.0 { 30.0 } else { 90.0 }))
        .collect();
    let mut pop = population(vec![st]);
    pop.covariate_names = vec!["CRCL".to_string()];
    let preds = predict(&model, &pop, &model.default_params).unwrap();
    for (traj, p) in res.trajectories.iter().zip(preds.iter()) {
        assert!(
            (traj.ipred - p.pred).abs() <= 1e-9 * p.pred.abs(),
            "IPRED {} != predict() {} at t={}",
            traj.ipred,
            p.pred,
            traj.time
        );
    }
}
