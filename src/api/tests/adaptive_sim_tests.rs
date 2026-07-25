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
        cens: vec![0; n],
        occasions: vec![1u32; n],
        obs_l2: Vec::new(),
        dose_occasions: vec![1u32; n_dose],
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
fn degenerate_oracle_matches_static_predict_bit_for_bit() {
    // A controller that doses at every decision must reproduce the static
    // engine on the same realized schedule. Last obs (54) is the global max
    // and a dose lands at every decision, so the segment structures align and
    // agreement is exact (the verifier, on by default, also passes).
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
    let preds = predict(&model, &static_pop, &model.default_params);

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
    let preds = predict(&model, &static_pop, &model.default_params);

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
    );

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
    let preds = predict(&model, &static_pop, &model.default_params);

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
    let preds = predict(&model, &static_pop, &model.default_params);

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
    let preds = predict(&model, &static_pop, &model.default_params);

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
    let preds = predict(&model, &static_pop, &model.default_params);
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
fn adaptive_base_regimen_with_tv_covariate_is_rejected() {
    // #702 scope: a base regimen combined with time-varying covariates is a typed error
    // (the per-segment PK / verifier interplay is a follow-up), never a silent
    // covariate-frozen integration of the delivered dose.
    let model = parse_model_string(ODE_TV_COV).expect("parse TV-cov ODE model");
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
        .expect_err("base regimen × TV covariate must be rejected");
    assert!(
        err.contains("base regimen") && err.contains("time-varying covariates"),
        "got: {err}"
    );
}

#[test]
fn adaptive_base_regimen_with_iov_is_rejected() {
    // #702 scope: base regimen × IOV is a typed error — per-occasion κ bookkeeping for a
    // pre-scheduled dose is a follow-up, never a silent κ=0 integration.
    let model = parse_model_string(ODE_IOV).expect("parse IOV ODE model");
    let s = subj(
        "1",
        vec![6.0, 30.0],
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
    );
    let pop = population(vec![s]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: vec![0.0, 24.0],
        ..Default::default()
    };
    let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect_err("base regimen × IOV must be rejected");
    assert!(
        err.contains("base regimen") && err.contains("inter-occasion"),
        "got: {err}"
    );
}

#[test]
fn adaptive_base_regimen_with_reset_is_rejected() {
    // #702 scope: base regimen × system reset (EVID=3/4) is a typed error. An EVID=4
    // (reset+dose) subject carries doses, so it lands here rather than on the reset path.
    let model = parse_model_string(ODE_NO_IIV).expect("parse");
    let mut s = subj(
        "1",
        vec![6.0, 30.0],
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
    );
    s.reset_times = vec![12.0];
    let pop = population(vec![s]);
    let opts = AdaptiveSimulateOptions {
        decision_times: vec![0.0, 24.0],
        ..Default::default()
    };
    let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect_err("base regimen × reset must be rejected");
    assert!(
        err.contains("base regimen") && err.contains("system resets"),
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
    let preds = predict(&model, &static_pop, &model.default_params);
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
    let preds = predict(&model, &static_pop, &model.default_params);
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

/// A minimal valid titration spec (built directly, no file) for the rejection
/// tests, which never reach the run loop.
fn simple_titration_spec() -> AdaptiveDosingSpec {
    AdaptiveDosingSpec {
        observe: Some("central".to_string()),
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
fn from_spec_degenerate_oracle_matches_static_predict_bit_for_bit() {
    // The declarative path's degenerate oracle: a block that never titrates
    // re-issues a fixed 100 mg bolus at every decision, so it must reproduce the
    // static engine on that same realized schedule. A dose lands at every
    // decision and the last obs (54) is the global max ⇒ bit-exact agreement,
    // and the frozen-replay verifier (on by default) also passes.
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
    let preds = predict(&parsed.model, &static_pop, &parsed.model.default_params);
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
    // fraction check evaluates typical parameters (η = 0) with no dose, so it fires
    // on the dose-free adaptive base at the chokepoint, before any decision — the
    // same typed error the static paths raise, instead of a silent 1.2× run.
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
    let pop = population(vec![subj("1", vec![0.0, 24.0, 48.0], vec![])]);
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
