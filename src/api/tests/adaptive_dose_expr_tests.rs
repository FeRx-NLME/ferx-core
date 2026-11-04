//! Per-subject dose expressions in `[adaptive_dosing]` (#944): `start_dose`,
//! `dose_bounds` and `levels` written as expressions over θ and data covariates,
//! evaluated once per subject at the first decision.
//!
//! Each test names the regression it exists to catch.

use super::*;
use crate::diagnostics::EngineError;
use crate::parser::model_parser::parse_full_model;
use crate::sim::adaptive::{AdaptiveDosingSpec, ControllerCtx, DoseAction};
use crate::types::ParsedModel;
use std::io::Write;

/// 1-cpt IV, CL and V scaled by weight: the mg/kg vancomycin example's model, without
/// its block. WT enters the PK as well as the dose, so a time-varying WT is live on
/// both sides.
const MODEL: &str = r#"
[parameters]
  theta TVCL(6.0,  0.1,  50.0)
  theta TVV(0.7,  0.05,   5.0)
  omega ETA_CL ~ 1e-10
  sigma ADD ~ 1.0
[individual_parameters]
  CL = TVCL * (WT / 70) * exp(ETA_CL)
  V  = TVV * WT
[structural_model]
  ode(states=[CENT])
[odes]
  d/dt(CENT) = -(CL / V) * CENT
[scaling]
  y = CENT / V
[error_model]
  DV ~ additive(ADD)
"#;

fn parsed(block: &str) -> ParsedModel {
    parse_full_model(&format!("{MODEL}\n[adaptive_dosing]\n{block}")).expect("model parses")
}

/// A population from CSV text, through the production reader. One temp file per call:
/// tests in one binary share a process.
fn pop(csv: &str) -> Population {
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    f.write_all(csv.as_bytes()).expect("write csv");
    crate::read_nonmem_csv(f.path(), None, None).expect("data reads")
}

/// One subject per `(id, WT)`, an observation row at each of `times`.
fn csv_const(subjects: &[(&str, f64)], times: &[f64]) -> String {
    let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,MDV,WT\n");
    for (id, wt) in subjects {
        for t in times {
            s.push_str(&format!("{id},{t},0,0,0,1,0,{wt}\n"));
        }
    }
    s
}

fn run(p: &ParsedModel, pop: &Population) -> Result<AdaptiveSimulationResult, EngineError> {
    let spec = p.adaptive_dosing.as_ref().expect("block present");
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        ..Default::default()
    };
    simulate_adaptive_from_spec(&p.model, pop, &p.model.default_params, 2, spec, &opts)
}

/// [`run`] at a supplied η = 0 for every subject: the typical-value patient the static
/// reference simulates. `omega ETA_CL ~ 1e-10` still draws an η of order 1e-5, which
/// moves the trough by ~1.5e-5 relative, far outside the 1e-9 band.
fn run_typical(p: &ParsedModel, pop: &Population) -> AdaptiveSimulationResult {
    let spec = p.adaptive_dosing.as_ref().expect("block present");
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        ..Default::default()
    };
    let ids: Vec<String> = pop.subjects.iter().map(|s| s.id.clone()).collect();
    let n = ids.len();
    let etas = crate::SuppliedEtas::from_table(
        ids,
        p.model.eta_names.clone(),
        vec![vec![0.0; p.model.n_eta]; n],
    )
    .expect("η = 0 table");
    let inputs = crate::AdaptiveInputs::new(&p.model.default_params).with_etas(&etas);
    simulate_adaptive_from_spec_with_inputs(&p.model, pop, &inputs, 1, spec, &opts).expect("runs")
}

fn first_dose(r: &AdaptiveSimulationResult, id: &str) -> f64 {
    r.ledger
        .iter()
        .find(|e| e.subject == id && e.sim == 1)
        .map(|e| e.amt)
        .expect("a dose for the subject")
}

const GRID: [f64; 5] = [0.0, 12.0, 24.0, 36.0, 48.0];

// ── D3: each subject's own plan, against the static engine ──────────────────────

/// Rules that never fire, so every decision re-issues `15 * WT`: each subject's ledger
/// holds its own weight's dose, and its trajectory equals `predict()` on that frozen
/// regimen. The realized gap is exactly 0 (macOS arm64, printed by the test); the bound
/// is the 1e-9 relative band every adaptive degenerate oracle uses (#1151). One subject
/// dosed at the other's weight moves its trajectory by tens of percent.
///
/// Regressions: the plan evaluated once per run (both subjects get one subject's
/// dose); the wrong subject's covariates read; a template plan shared across subjects.
#[test]
fn each_subject_is_dosed_from_its_own_weight_and_matches_the_static_engine() {
    let p = parsed(
        "  observe = CENT / V
  at = every 12 from 0 to 48
  start_dose = 15 * WT
  route = infuse(cmt=1, over=1)
  dose_bounds = [0, min(30 * WT, 4000)]
  when signal > 1000000 : decrease 25%",
    );
    let population = pop(&csv_const(&[("1", 50.0), ("2", 110.0)], &GRID));
    let res = run_typical(&p, &population);
    for (id, wt) in [("1", 50.0f64), ("2", 110.0)] {
        let want = 15.0 * wt;
        let amts: Vec<f64> = res
            .ledger
            .iter()
            .filter(|e| e.subject == id)
            .map(|e| e.amt)
            .collect();
        assert_eq!(amts.len(), GRID.len(), "subject {id}: a dose per decision");
        assert!(
            amts.iter().all(|a| a.to_bits() == want.to_bits()),
            "subject {id}: every dose is 15 * {wt} = {want}, got {amts:?}"
        );
    }
    // The static reference: each subject's realized regimen pre-scheduled.
    let mut static_pop = population.clone();
    for s in &mut static_pop.subjects {
        let wt = s.covariates["WT"];
        s.doses = GRID
            .iter()
            .map(|&t| crate::types::DoseEvent::new(t, 15.0 * wt, 1, 15.0 * wt, false, 0.0))
            .collect();
    }
    let preds = predict(&p.model, &static_pop, &p.model.default_params).expect("predict");
    let traj: Vec<_> = res.trajectories.iter().filter(|t| t.sim == 1).collect();
    assert_eq!(traj.len(), preds.len());
    let mut worst = 0.0f64;
    for (t, pr) in traj.iter().zip(&preds) {
        assert_eq!((t.id.as_str(), t.time), (pr.id.as_str(), pr.time));
        assert!(t.ipred.is_finite() && pr.pred.is_finite());
        worst = worst.max((t.ipred - pr.pred).abs() / pr.pred.abs().max(1e-300));
        assert!(
            (t.ipred - pr.pred).abs() <= 1e-9 + 1e-9 * pr.pred.abs(),
            "subject {} t={}: adaptive {} vs static {}",
            t.id,
            t.time,
            t.ipred,
            pr.pred
        );
    }
    eprintln!("D3 worst relative gap: {worst:.3e}");
}

// ── D1: a constant expression is the literal, bit for bit ───────────────────────

type Fp = (
    Vec<(String, u64, u64, u64, String)>,
    Vec<(String, u64, String, Vec<u64>)>,
    Vec<(String, u64, u64)>,
    Vec<String>,
);

/// Every `f64` of the result as bits: ledger, decisions, trajectories, metrics.
fn fingerprint(r: &AdaptiveSimulationResult) -> Fp {
    (
        r.ledger
            .iter()
            .map(|e| {
                (
                    e.subject.clone(),
                    e.time.to_bits(),
                    e.amt.to_bits(),
                    e.rate.to_bits(),
                    e.rule_fired.clone(),
                )
            })
            .collect(),
        r.decisions
            .iter()
            .map(|d| {
                (
                    d.subject.clone(),
                    d.time.to_bits(),
                    format!("{:?}", d.outcome),
                    d.observed_signals
                        .iter()
                        .map(|s| s.value.to_bits())
                        .collect(),
                )
            })
            .collect(),
        r.trajectories
            .iter()
            .map(|t| (t.id.clone(), t.time.to_bits(), t.ipred.to_bits()))
            .collect(),
        r.metrics.iter().map(|m| format!("{m:?}")).collect(),
    )
}

/// `25 * WT` and `[0, 100 * WT]` at WT = 40 against the literal `1000` and `[0, 4000]`:
/// identical bits everywhere, with a titration that climbs into the upper bound (so the
/// bound is read, not just carried).
///
/// Regressions: the expression path builds the controller differently from the literal
/// one — a bound not taken from the plan, the running dose not seeded from it.
#[test]
fn a_constant_expression_is_bit_identical_to_the_literal() {
    let block = |start: &str, hi: &str| {
        format!(
            "  observe = CENT / V
  at = every 12 from 0 to 96
  start_dose = {start}
  route = infuse(cmt=1, over=1)
  dose_bounds = [0, {hi}]
  confirm = 1
  when signal < 60 : increase 50%"
        )
    };
    let population = pop(&csv_const(&[("1", 40.0), ("2", 40.0)], &GRID));
    let lit = run(&parsed(&block("1000", "4000")), &population).expect("literal runs");
    let expr = run(&parsed(&block("25 * WT", "100 * WT")), &population).expect("expr runs");
    assert!(
        lit.ledger.iter().any(|e| e.amt == 4000.0),
        "the titration reaches the upper bound: {:?}",
        lit.ledger.iter().map(|e| e.amt).collect::<Vec<_>>()
    );
    assert_eq!(fingerprint(&lit), fingerprint(&expr));
}

// ── When: the covariates in force at the first decision ──────────────────────────

/// WT changes from 50 to 80 between t = 0 and the first decision (t = 12) for subject
/// 1, and after it (t = 30) for subject 2. Subject 1 starts at 15 * 80, subject 2 at
/// 15 * 50, and neither start dose follows a later change.
///
/// Regressions: evaluating at baseline (`subject.covariates`, t = 0: subject 1 gets
/// 750); evaluating at the last record (subject 2 gets 1200).
#[test]
fn a_time_varying_covariate_is_read_at_the_first_decision() {
    let p = parsed(
        "  observe = CENT / V
  at = every 12 from 12 to 48
  start_dose = 15 * WT
  route = infuse(cmt=1, over=1)
  dose_bounds = [0, 4000]
  when signal > 1000000 : decrease 25%",
    );
    let csv = "ID,TIME,DV,AMT,EVID,CMT,MDV,WT
1,0,0,0,0,1,0,50
1,6,0,0,0,1,0,80
1,24,0,0,0,1,0,80
1,48,0,0,0,1,0,80
2,0,0,0,0,1,0,50
2,24,0,0,0,1,0,50
2,30,0,0,0,1,0,80
2,48,0,0,0,1,0,80
";
    let res = run(&p, &pop(csv)).expect("runs");
    assert_eq!(
        first_dose(&res, "1"),
        15.0 * 80.0,
        "WT changed before at[0]"
    );
    assert_eq!(first_dose(&res, "2"), 15.0 * 50.0, "WT changed after at[0]");
    // Evaluated once: no later dose follows the t = 30 change.
    assert!(res
        .ledger
        .iter()
        .filter(|e| e.subject == "2")
        .all(|e| e.amt == 750.0));
}

/// A first decision before the subject's first record (#2125): the expression reads
/// exactly the `ctx.covariates` a programmatic controller is handed at decision 0. The
/// data are time-varying (60, then 90), so resolvers that differ can be told apart.
///
/// Regression: a second covariate resolver for dose expressions that drifts from the
/// driver's (e.g. the last record, or the record nearest `at[0]`).
#[test]
fn before_the_first_record_the_expression_reads_what_the_controller_reads() {
    let csv = "ID,TIME,DV,AMT,EVID,CMT,MDV,WT
1,24,0,0,0,1,0,60
1,48,0,0,0,1,0,90
1,72,0,0,0,1,0,90
";
    let population = pop(csv);
    let p = parsed(
        "  observe = CENT / V
  at = every 24 from 0 to 72
  start_dose = 15 * WT
  route = infuse(cmt=1, over=1)
  dose_bounds = [0, 4000]
  when signal > 1000000 : decrease 25%",
    );
    let res = run(&p, &population).expect("spec runs");

    let seen = std::sync::Mutex::new(Vec::new());
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        decision_times: vec![0.0, 24.0, 48.0, 72.0],
        monitors: vec![crate::sim::adaptive::MonitorSpec::new(
            "C",
            1,
            crate::sim::adaptive::ObserveMode::Ipred,
        )],
        ..Default::default()
    };
    simulate_adaptive(
        &p.model,
        &population,
        &p.model.default_params,
        1,
        || {
            |ctx: &ControllerCtx| {
                if ctx.decision_index == 0 {
                    seen.lock().unwrap().push(ctx.covariates["WT"]);
                }
                vec![DoseAction::Hold]
            }
        },
        &opts,
    )
    .expect("programmatic runs");
    let wt0: f64 = seen.lock().unwrap()[0];
    assert_eq!(first_dose(&res, "1").to_bits(), (15.0f64 * wt0).to_bits());
}

// ── Per-subject refusals (E_ADAPTIVE_DOSE_VALUE) ──────────────────────────────────

/// Run `block` on two subjects, `ok` passing and `bad` failing, and return the refusal.
fn refusal(block: &str, ok: (&str, f64), bad: (&str, f64)) -> EngineError {
    let population = pop(&csv_const(&[ok, bad], &GRID));
    let e = run(&parsed(block), &population).expect_err("refused");
    assert_eq!(e.code(), Some("E_ADAPTIVE_DOSE_VALUE"), "{e}");
    e
}

const RULE: &str = "  when signal > 1000000 : decrease 25%";

/// One block per invariant, each with a passing subject beside the failing one. The
/// whole message is asserted, so deleting any sentence of it, or naming the wrong
/// subject, fails here; the passing subject's ID must be absent.
///
/// Regressions: the check run on the first subject only (subject 2 is the bad one);
/// the passing subject named; a sentence of a message lost.
#[test]
fn each_dose_invariant_is_refused_per_subject_naming_that_subject() {
    let head =
        "  observe = CENT / V\n  at = every 12 from 0 to 48\n  route = infuse(cmt=1, over=1)\n";
    let cases: Vec<(String, (&str, f64), (&str, f64), String)> = vec![
        (
            format!("{head}  start_dose = 15 * WT - 1000\n  dose_bounds = [0, 4000]\n{RULE}"),
            ("1", 100.0),
            ("2", 50.0),
            "[adaptive_dosing] subject 2: start_dose = 15 * WT - 1000 evaluates to -250; a \
             dose must be finite and >= 0. The dose expressions read WT = 50, at the first \
             decision (t = 0)."
                .to_string(),
        ),
        (
            format!("{head}  start_dose = 20\n  dose_bounds = [WT - 60, 4000]\n{RULE}"),
            ("1", 70.0),
            ("2", 50.0),
            "[adaptive_dosing] subject 2: dose_bounds = [WT - 60, 4000] evaluate to [-10, \
             4000]; they must be finite with 0 <= low <= high. The dose expressions read WT \
             = 50, at the first decision (t = 0)."
                .to_string(),
        ),
        (
            format!("{head}  start_dose = 15 * WT\n  dose_bounds = [0, 1000]\n{RULE}"),
            ("1", 50.0),
            ("2", 100.0),
            "[adaptive_dosing] subject 2: start_dose 1500 is outside dose_bounds [0, 1000]. \
             The dose expressions read WT = 100, at the first decision (t = 0)."
                .to_string(),
        ),
        (
            format!(
                "{head}  start_dose = 10 * WT\n  dose_bounds = [0, 4000]\n  levels = [500, \
                 1000, 1500]\n  when signal > 1000000 : decrease"
            ),
            ("1", 50.0),
            ("2", 60.0),
            "[adaptive_dosing] subject 2: start_dose 600.0 must be one of the levels [500.0, \
             1000.0, 1500.0]. The dose expressions read WT = 60, at the first decision (t = 0)."
                .to_string(),
        ),
        (
            format!(
                "{head}  start_dose = 500\n  dose_bounds = [0, 1500]\n  levels = [500, 20 * \
                 WT]\n  when signal > 1000000 : decrease"
            ),
            ("1", 50.0),
            ("2", 100.0),
            "[adaptive_dosing] subject 2: level 2000 is outside dose_bounds [0, 1500]. The \
             dose expressions read WT = 100, at the first decision (t = 0)."
                .to_string(),
        ),
        (
            format!(
                "{head}  start_dose = 500\n  dose_bounds = [0, 4000]\n  levels = [500, 20 * \
                 WT]\n  when signal > 1000000 : decrease"
            ),
            ("1", 50.0),
            ("2", 20.0),
            "[adaptive_dosing] subject 2: levels evaluate to [500.0, 400.0]; they must be \
             finite and strictly increasing. The dose expressions read WT = 20, at the first \
             decision (t = 0)."
                .to_string(),
        ),
    ];
    for (block, ok, bad, want) in cases {
        let e = refusal(&block, ok, bad);
        assert_eq!(e.message(), want, "block:\n{block}");
        assert!(!e.message().contains("subject 1"), "{e}");
    }
}

/// The closing sentence names the covariates read; an expression that reads none (θ
/// only) has no such sentence.
///
/// Regression: the sentence emitted unconditionally ("read , at the first decision"),
/// or dropped.
#[test]
fn a_dose_expression_reading_no_covariate_has_no_reads_sentence() {
    let block = "  observe = CENT / V
  at = every 12 from 0 to 48
  route = infuse(cmt=1, over=1)
  start_dose = TVCL * 1000
  dose_bounds = [0, 4000]
  when signal > 1000000 : decrease 25%";
    let population = pop(&csv_const(&[("1", 50.0)], &GRID));
    let e = run(&parsed(block), &population).expect_err("refused");
    assert_eq!(
        e.message(),
        "[adaptive_dosing] subject 1: start_dose 6000 is outside dose_bounds [0, 4000]."
    );
}

// ── Run-start refusals ───────────────────────────────────────────────────────────

/// A dose expression's covariate absent from the data is `E_MISSING_COVARIATE`, naming
/// the key, the column, and the columns there are.
///
/// Regression: the dose covariates not checked, so `15 * WTX` doses 0 mg.
#[test]
fn a_dose_expression_covariate_missing_from_the_data_is_refused() {
    let block = "  observe = CENT / V
  at = every 12 from 0 to 48
  route = infuse(cmt=1, over=1)
  start_dose = 500
  dose_bounds = [0, 30 * WTX]
  when signal > 1000000 : decrease 25%";
    let population = pop(&csv_const(&[("1", 50.0)], &GRID));
    let e = run(&parsed(block), &population).expect_err("refused");
    assert_eq!(e.code(), Some("E_MISSING_COVARIATE"));
    assert_eq!(
        e.message(),
        "[adaptive_dosing] `dose_bounds` references covariate(s) not found in data \
         (case-sensitive): WTX. Available covariate columns: WT."
    );
}

/// A spec built in code bypasses the parser, so `compile_adaptive` re-runs the read
/// check: a state read is refused there with the parser's text.
///
/// Regression: `compile_dose_exprs` without the read check — `2 * CENT` compiles and
/// reads a zero state slice, so every subject starts at 0 mg.
#[test]
fn a_hand_built_spec_reading_a_state_is_refused_at_compile() {
    let p = parsed(
        "  observe = CENT / V
  at = every 12 from 0 to 48
  start_dose = 500
  route = infuse(cmt=1, over=1)
  dose_bounds = [0, 4000]
  when signal > 1000000 : decrease 25%",
    );
    let mut spec: AdaptiveDosingSpec = p.adaptive_dosing.clone().expect("block");
    spec.start_dose = crate::sim::adaptive::DoseValue::Expr("2 * CENT".to_string());
    let population = pop(&csv_const(&[("1", 50.0)], &GRID));
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        ..Default::default()
    };
    let e = simulate_adaptive_from_spec(
        &p.model,
        &population,
        &p.model.default_params,
        1,
        &spec,
        &opts,
    )
    .expect_err("refused");
    assert_eq!(e.code(), Some("E_ADAPTIVE_SPEC"));
    assert!(
        e.message()
            .starts_with("[adaptive_dosing] start_dose reads CENT, a state of the model."),
        "{e}"
    );
}
