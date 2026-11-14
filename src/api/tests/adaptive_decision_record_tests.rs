//! Every decision of an adaptive run is a record (#2895).
//!
//! The external reference throughout is the static engine on the run's static equivalent:
//! the realized doses plus an `EVID=2` record at every decision, its covariate cells filled
//! by the run's `decision_covariates` rule (LOCF by default). That reference shares nothing
//! with the driver's governing rule: `predict()` / `predict_iov` resolve each interval by
//! the end-of-interval rule over the rows they are given, and the `EVID=2` row's own `OCC`
//! is anchored to NONMEM (`nonmem_anchor/iov_pk_only_*`, #1876). The frozen-replay verifier,
//! on by default in every run here, shares the driver's rule, so it cannot be the oracle.
use super::adaptive_sim_tests::{fixed_bolus, population, subj, ODE_IOV};
use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::sim::adaptive::{ControllerCtx, DecisionCovariates, DoseAction};
use std::collections::HashMap;

/// Decisions of every fixture here: four boluses (or three and a hold), so drug is present
/// at each later boundary and every occasion's κ acts on a non-zero state.
const DECISIONS: [f64; 4] = [0.0, 24.0, 48.0, 72.0];

/// The four record layouts of the #2895 table.
const LAYOUTS: [&[f64]; 4] = [
    &[0.0, 24.0, 48.0, 72.0],
    &[0.0, 23.0, 47.0, 71.0],
    &[0.0, 12.0, 48.0, 72.0],
    &[0.0, 72.0],
];

/// A 100-unit bolus at every decision except a hold at 48, so the decision at 48 leaves no
/// dose row: only the decision's own record can end the interval there.
fn hold_at_48() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |ctx: &ControllerCtx| {
        if ctx.t == 48.0 {
            vec![DoseAction::Hold]
        } else {
            vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
        }
    }
}

/// The BSV η and per-decision κ a seeded run draws for subject "1", replicate 1 (the
/// derivation of `adaptive_iov_matches_predict_iov_with_reconstructed_kappa`).
fn drawn_eta_and_kappas(model: &CompiledModel, seed: u64) -> (Vec<f64>, Vec<Vec<f64>>) {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let z: Vec<f64> = (0..model.n_eta).map(|_| rng.sample(normal)).collect();
    let eta: Vec<f64> = (&model.default_params.omega.chol
        * nalgebra::DVector::from_column_slice(&z))
    .iter()
    .copied()
    .collect();
    let omega_iov = model.default_params.omega_iov.as_ref().expect("omega_iov");
    let base = crate::sim::adaptive::subject_kappa_base_seed(seed, "1", 1);
    let kappas = (0..DECISIONS.len())
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
    (eta, kappas)
}

/// `predict_iov` on the static equivalent of an IOV run: the realized doses, `obs`, and (with
/// `records`) an `EVID=2` row at every decision, each row's occasion its decision window.
/// `predict_iov` indexes κ by `iov_occasion_groups` — observation occasions first — so the
/// drawn κ are handed over in that order, not in decision order.
fn predict_iov_static(
    model: &CompiledModel,
    res: &AdaptiveSimulationResult,
    obs: &[f64],
    eta: &[f64],
    kappas: &[Vec<f64>],
    records: bool,
) -> Vec<f64> {
    let doses: Vec<DoseEvent> = res
        .ledger
        .iter()
        .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0))
        .collect();
    let mut s = subj("1", obs.to_vec(), doses);
    let occ = |t: f64| crate::pk::occasion_of(&DECISIONS, t).expect("in a window") as u32;
    s.occasions = obs.iter().map(|&t| occ(t)).collect();
    s.dose_occasions = res.ledger.iter().map(|e| occ(e.time)).collect();
    if records {
        s.pk_only_times = DECISIONS.to_vec();
        s.pk_only_raw_times = DECISIONS.to_vec();
        s.pk_only_occasions = DECISIONS.iter().map(|&t| occ(t)).collect();
    }
    let ordered: Vec<Vec<f64>> = crate::stats::likelihood::iov_occasion_groups(&s)
        .iter()
        .map(|(o, _)| kappas[*o as usize].clone())
        .collect();
    crate::pk::predict_iov(model, &s, &model.default_params.theta, eta, &ordered)
}

fn run_iov(obs: &[f64], hold: bool, seed: u64) -> AdaptiveSimulationResult {
    let model = parse_model_string(ODE_IOV).expect("parse");
    let pop = population(vec![subj("1", obs.to_vec(), vec![])]);
    let opts = AdaptiveSimulateOptions {
        seed: Some(seed),
        decision_times: DECISIONS.to_vec(),
        ..Default::default() // verify = true: frozen replay + #748 snapshot checks
    };
    if hold {
        simulate_adaptive(&model, &pop, &model.default_params, 1, hold_at_48, &opts)
    } else {
        simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
    }
    .expect("runs and passes the default-on verifiers")
}

fn bits(v: impl IntoIterator<Item = f64>) -> Vec<u64> {
    v.into_iter().map(f64::to_bits).collect()
}

/// The external reference (#2895): on all four layouts of the table, with a bolus at every
/// decision and with a hold at 48, the adaptive trajectory equals `predict_iov` on the
/// realized doses plus an `EVID=2` record per decision — **to the bit**.
///
/// Before #2895 the sparse layouts did not: against this same reference, main at
/// `49f3857d8` (macOS arm64) was 1.71e-3 apart at t = 48 on records 0, 12, 48, 72 and
/// 1.88e-3 at t = 72 on records 0, 72; 2.12e-2 at t = 48 with the hold.
///
/// The record at 48 is what makes the hold case honest: without it, `predict_iov` on
/// records 0, 72 is 1.8e-3 from the run (asserted), because nothing in the static dataset
/// ends the interval at the held decision. So the reference needs a record at **every**
/// decision, not only at the dosing ones. Non-degenerate: four occasions whose CL differ
/// pairwise by more than 1e-2 (asserted), drug present at each later boundary.
#[test]
fn the_run_equals_predict_iov_on_the_realized_doses_plus_a_record_per_decision() {
    const SEED: u64 = 20260708;
    let model = parse_model_string(ODE_IOV).expect("parse");
    let (eta, kappas) = drawn_eta_and_kappas(&model, SEED);
    for (i, a) in kappas.iter().enumerate() {
        for b in &kappas[i + 1..] {
            assert!(
                (a[0].exp() - b[0].exp()).abs() > 1e-2,
                "the occasions' CL must differ: {kappas:?}"
            );
        }
    }
    for hold in [false, true] {
        for obs in LAYOUTS {
            let res = run_iov(obs, hold, SEED);
            assert_eq!(res.ledger.len(), if hold { 3 } else { 4 });
            let got: Vec<f64> = res.trajectories.iter().map(|r| r.ipred).collect();
            let want = predict_iov_static(&model, &res, obs, &eta, &kappas, true);
            assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
            assert_eq!(
                bits(got.iter().copied()),
                bits(want.iter().copied()),
                "hold={hold} records {obs:?}: adaptive {got:?} vs predict_iov {want:?}"
            );
        }
    }
    // The hold case needs the record at the held decision.
    let res = run_iov(&[0.0, 72.0], true, SEED);
    let without = predict_iov_static(&model, &res, &[0.0, 72.0], &eta, &kappas, false);
    let got = res.trajectories[1].ipred;
    assert!(
        ((got - without[1]) / without[1]).abs() > 1e-3,
        "without the record at 48 the reference must miss: {got} vs {}",
        without[1]
    );
}

/// The straddle pair (#2895): one subject with and one without a data record on decision 24,
/// on the sparse layout (records at 0 and 72). The decision is a record either way, so the two
/// trajectories agree at every shared time, to the bit.
///
/// Before #2895 the pair straddled the rule: without the data record, `(0, 24]` ran under the
/// record at 72 (κ_3); with it, under κ_1. That is asserted to be a real difference (the two
/// CL more than 1e-2 apart), so the agreement cannot be a tautology of indistinguishable κ.
#[test]
fn a_data_record_on_a_decision_changes_nothing() {
    const SEED: u64 = 4242;
    let model = parse_model_string(ODE_IOV).expect("parse");
    let (_, kappas) = drawn_eta_and_kappas(&model, SEED);
    assert!(
        (kappas[1][0].exp() - kappas[3][0].exp()).abs() > 1e-2,
        "the pre-#2895 straddle: κ_1 and κ_3 must differ, {kappas:?}"
    );
    let sparse = run_iov(&[0.0, 72.0], false, SEED);
    let with_record = run_iov(&[0.0, 24.0, 72.0], false, SEED);
    let at = |res: &AdaptiveSimulationResult, t: f64| {
        res.trajectories
            .iter()
            .find(|r| r.time == t)
            .expect("a row")
            .ipred
    };
    for t in [0.0, 72.0] {
        assert_eq!(
            at(&sparse, t).to_bits(),
            at(&with_record, t).to_bits(),
            "t={t}: {} vs {}",
            at(&sparse, t),
            at(&with_record, t)
        );
    }
    let trough = |res: &AdaptiveSimulationResult, t: f64| {
        res.decisions
            .iter()
            .find(|d| d.time == t)
            .expect("a decision")
            .observed_signals
            .first()
            .map(|s| s.value)
    };
    for t in [24.0, 48.0, 72.0] {
        assert_eq!(
            trough(&sparse, t).map(f64::to_bits),
            trough(&with_record, t).map(f64::to_bits),
            "the state decision {t} reads"
        );
    }
}

/// One compartment, CL on a time-varying `WT`; η declared but unread, so `predict()`'s
/// population prediction is the individual one the run integrates.
const ODE_TV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * WT / 70
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP)
"#;

/// The same with CL reading `TIME` and no covariate.
const ODE_TIME: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * (1 + 0.01 * TIME)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP)
"#;

fn wt(w: f64) -> HashMap<String, f64> {
    HashMap::from([("WT".to_string(), w)])
}

/// A non-IOV run under `run_rule` and `predict()` on a static equivalent filled by
/// `fill_rule`: the realized dose rows and an `EVID=2` row per decision, each carrying the
/// covariate row `fill_rule` gives it. `rows` are `(time, WT)` observation records (`WT`
/// unread by `ODE_TIME`). Returns the worst relative gap, `NaN`-safe.
fn tv_gap(
    model_src: &str,
    rows: &[(f64, f64)],
    run_rule: DecisionCovariates,
    fill_rule: DecisionCovariates,
) -> f64 {
    let model = parse_model_string(model_src).expect("parse");
    let obs: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let base = |doses: Vec<DoseEvent>| {
        let mut s = subj("1", obs.clone(), doses);
        s.covariates = wt(rows[0].1);
        s.obs_covariates = rows.iter().map(|r| wt(r.1)).collect();
        s
    };
    let mut pop = population(vec![base(vec![])]);
    pop.covariate_names = vec!["WT".to_string()];
    let mut opts = AdaptiveSimulateOptions {
        seed: Some(5),
        decision_times: DECISIONS.to_vec(),
        ..Default::default() // verify = true: frozen replay + #748 snapshot checks
    };
    opts.decision_covariates = Some(run_rule);
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
        .expect("runs and passes the default-on verifiers");
    let locf = |t: f64| rows.iter().rfind(|r| r.0 <= t).map_or(rows[0].1, |r| r.1);
    let nocb = |t: f64| rows.iter().find(|r| r.0 >= t).map_or(locf(t), |r| r.1);
    let fill = |t: f64| match fill_rule {
        DecisionCovariates::Nocb => nocb(t),
        _ => locf(t),
    };
    let doses: Vec<DoseEvent> = res
        .ledger
        .iter()
        .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0))
        .collect();
    let mut s = base(doses);
    s.dose_covariates = res.ledger.iter().map(|e| wt(fill(e.time))).collect();
    s.pk_only_times = DECISIONS.to_vec();
    s.pk_only_raw_times = DECISIONS.to_vec();
    s.pk_only_covariates = DECISIONS.iter().map(|&t| wt(fill(t))).collect();
    s.pk_only_occasions = vec![1; DECISIONS.len()];
    let mut sp = population(vec![s]);
    sp.covariate_names = vec!["WT".to_string()];
    let preds = predict(&model, &sp, &model.default_params).expect("predicts");
    assert_eq!(preds.len(), res.trajectories.len());
    let mut worst = 0.0_f64;
    for (r, p) in res.trajectories.iter().zip(&preds) {
        assert!(
            r.ipred.is_finite() && p.pred.is_finite(),
            "{} / {}",
            r.ipred,
            p.pred
        );
        let gap = ((r.ipred - p.pred) / p.pred).abs();
        assert!(gap.is_finite(), "gap {gap} at t={}", r.time);
        worst = worst.max(gap);
    }
    worst
}

/// The external reference off IOV (#2895). The rule is not IOV-only: a decision is the record
/// that ends the interval before it under a time-varying covariate and under a `TIME`-reading
/// PK too. Each setting of `decision_covariates` equals `predict()` on the dataset filled by
/// its own rule; the other fill is asserted far off, so the two settings are each observed.
///
/// Measured on macOS arm64: each run equals `predict()` on its own fill **to the bit** (worst
/// relative gap 0), and misses the other fill by more than 1e-2 (`WT` steps 70 → 50 at 72,
/// with nothing recorded between 12 and 72). Before #2895 the driver matched only a NOCB
/// fill and missed the LOCF one by 1.57e-2; with `TIME` in the PK it missed by 2.17e-4.
#[test]
fn each_decision_covariates_rule_equals_predict_on_its_own_fill() {
    let sparse = [(0.0, 70.0), (12.0, 70.0), (72.0, 50.0)];
    for (rule, other) in [
        (DecisionCovariates::Locf, DecisionCovariates::Nocb),
        (DecisionCovariates::Nocb, DecisionCovariates::Locf),
    ] {
        let own = tv_gap(ODE_TV, &sparse, rule, rule);
        assert_eq!(own, 0.0, "{rule:?} against its own fill: {own:e}");
        let crossed = tv_gap(ODE_TV, &sparse, rule, other);
        assert!(
            crossed > 1e-2,
            "{rule:?} against {other:?}'s fill: {crossed:e}"
        );
    }
    for rule in [DecisionCovariates::Locf, DecisionCovariates::Nocb] {
        let flat = [(0.0, 70.0), (12.0, 70.0), (72.0, 70.0)];
        let gap = tv_gap(ODE_TIME, &flat, rule, rule);
        assert_eq!(gap, 0.0, "TIME in the PK, {rule:?}: {gap:e}");
    }
}

fn read_csv(csv: &str, occ: Option<&str>) -> Population {
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    f.write_all(csv.as_bytes()).expect("write csv");
    crate::read_nonmem_csv(f.path(), None, occ).expect("data reads")
}

/// An `ADDL`-implied dose between a decision and the next data record (#2895): the decision
/// is a record, so it governs the dose — its occasion, `TIME` and covariate row — as an
/// `EVID=2` row there governs it in the static dataset (the reader resolves the implied
/// dose's governing record, #1909 / #2371). A dose at 10 implies doses at 34 and 58, the only
/// later data record is the observation at 96, decisions at 0, 24, 48, 72, 96 hold.
///
/// Bit-identical. The `EVID=2` rows go only where no data row sits (24, 48, 72): a co-timed
/// one would move `predict_iov` itself by a few ULP (1.9e-15 measured on the dense layout),
/// though it changes no governing record. Before #2895: 5.5e-3 (the implied doses governed by
/// the observation at 96, κ_4).
#[test]
fn an_implied_dose_before_the_next_data_record_is_governed_by_the_decision() {
    const SEED: u64 = 99;
    let model = parse_model_string(ODE_IOV).expect("parse");
    let decisions = [0.0, 24.0, 48.0, 72.0, 96.0];
    let data = "ID,TIME,DV,AMT,EVID,CMT,MDV,ADDL,II,OCC
1,0,0,0,0,1,0,0,0,0
1,10,0,100,1,1,1,2,24,0
1,96,0,0,0,1,0,0,0,4
";
    let pop = read_csv(data, None);
    let opts = AdaptiveSimulateOptions {
        seed: Some(SEED),
        decision_times: decisions.to_vec(),
        ..Default::default()
    };
    let hold = || |_ctx: &ControllerCtx| vec![DoseAction::Hold];
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, hold, &opts)
        .expect("runs and passes the default-on verifiers");
    let (eta, _) = drawn_eta_and_kappas(&model, SEED);
    let omega_iov = model.default_params.omega_iov.as_ref().expect("omega_iov");
    let base = crate::sim::adaptive::subject_kappa_base_seed(SEED, "1", 1);
    let kappas: Vec<f64> = (0..decisions.len())
        .map(|g| omega_iov.chol[(0, 0)] * crate::sim::adaptive::kappa_standard_normal(base, g, 0))
        .collect();
    assert!(
        (kappas[2].exp() - kappas[4].exp()).abs() > 1e-2,
        "the decision's and the observation's occasion must differ: {kappas:?}"
    );
    let mut csv = data.to_string();
    for (g, t) in decisions.iter().enumerate().skip(1).take(3) {
        csv.push_str(&format!("1,{t},0,0,2,1,1,0,0,{g}\n"));
    }
    let sp = read_csv(&csv, Some("OCC"));
    let s = &sp.subjects[0];
    let ordered: Vec<Vec<f64>> = crate::stats::likelihood::iov_occasion_groups(s)
        .iter()
        .map(|(o, _)| vec![kappas[*o as usize]])
        .collect();
    let want = crate::pk::predict_iov(&model, s, &model.default_params.theta, &eta, &ordered);
    let got = res.trajectories[1].ipred;
    assert!(got.is_finite() && want[1].is_finite() && got > 1.0);
    assert_eq!(
        got.to_bits(),
        want[1].to_bits(),
        "t=96: adaptive {got} vs predict_iov {}",
        want[1]
    );
}

/// A declarative block owns `decision_covariates`, so `simulate_adaptive_from_spec` refuses
/// the programmatic option rather than pick one of two sources (#2895), as it refuses
/// `decision_times`. The message names the option and the block's key.
#[test]
fn the_spec_path_refuses_the_decision_covariates_option() {
    let model = parse_model_string(ODE_IOV).expect("parse");
    let spec = super::adaptive_auc_tv_tests::fixed_dose_spec(Some("central"), &DECISIONS, 100.0);
    let pop = population(vec![subj("1", vec![0.0, 72.0], vec![])]);
    let mut opts = AdaptiveSimulateOptions::default();
    opts.decision_covariates = Some(DecisionCovariates::Nocb);
    let err = simulate_adaptive_from_spec(&model, &pop, &model.default_params, 1, &spec, &opts)
        .expect_err("refused");
    let msg = err.to_string();
    assert!(
        msg.contains("`opts.decision_covariates`") && msg.contains("`decision_covariates`"),
        "{msg}"
    );
    opts.decision_covariates = None;
    simulate_adaptive_from_spec(&model, &pop, &model.default_params, 1, &spec, &opts)
        .expect("runs without the option");
}
