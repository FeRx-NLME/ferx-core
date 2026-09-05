//! Tier-1 tests for the SCM search logic (#1180).
//!
//! Every test here drives [`search`] with a *scripted* fitter — a table of
//! OFVs keyed by the candidate's relation set — so what is under test is the
//! step logic: which candidate a step selects, when a phase stops, what the
//! backward phase removes, what the adaptive stash does, and what the report
//! says about a candidate that was excluded. None of that is observable
//! through real fits without putting every test in the slow tier. The path
//! from `ModelText` through the runner and `fit()` is covered by
//! `tests/covsearch_end_to_end.rs`.
//!
//! The fixture OFVs are chosen so the decisions are unambiguous at the levels
//! tested: PsN's tabulated cutoffs are 6.63 (p = 0.01, df = 1) and 10.83
//! (p = 0.001, df = 1).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use ferx_core::edit::ModelText;
use ferx_core::{CovariateForm, StrictnessVerdict};

use super::*;
use crate::search::test_support::converged_fit;
use crate::search::{CandidateError, CandidateResult, RunReport};

/// A base model with the three parameters a search can hang effects on.
const BASE: &str = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[covariates]
  WT continuous
  CRCL continuous
  SEX categorical(levels = [0, 1])

[error_model]
  DV ~ proportional(PROP_ERR)
";

const BASE_N: usize = 7;

fn effect(parameter: &str, covariate: &str, form: CovariateForm) -> Effect {
    Effect {
        parameter: parameter.into(),
        covariate: covariate.into(),
        form,
    }
}

fn pow(parameter: &str, covariate: &str) -> Effect {
    effect(parameter, covariate, CovariateForm::Power)
}

fn space(candidates: Vec<Effect>) -> Space {
    Space {
        base_model: ModelText::parse(BASE).unwrap(),
        candidates,
        included: Vec::new(),
        notes: Vec::new(),
    }
}

/// The scripted fitter. OFVs are keyed by the candidate's feature vector
/// (its relation set, `CL-WT=power;V-WT=power`); a candidate not in the
/// table gets `fallback`, which the tests set high enough never to matter.
/// Parameter counts are the base count plus one per relation, two for a
/// hockey stick — what the real fits would report.
struct Script {
    ofv: HashMap<String, f64>,
    fallback: f64,
    /// Feature keys — or candidate ids — whose candidate fails the
    /// strictness gate.
    failing: Vec<String>,
    /// Feature keys whose candidate produces no fit at all.
    erroring: Vec<String>,
    /// Feature keys whose candidate compiles to the parent's parameter count.
    no_extra_parameter: Vec<String>,
    /// Flip the report's `cancelled` after this many `fit_step` calls.
    cancel_after: Option<usize>,
    calls: Mutex<Vec<(String, Vec<String>)>>,
}

impl Script {
    fn new(table: &[(&str, f64)]) -> Script {
        Script {
            ofv: table.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            fallback: 1000.0,
            failing: Vec::new(),
            erroring: Vec::new(),
            no_extra_parameter: Vec::new(),
            cancel_after: None,
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Every step directory the search asked for, in order.
    fn dirs(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|(d, _)| d.clone())
            .collect()
    }

    /// The candidate ids fitted in one step directory.
    fn fitted_in(&self, dir: &str) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .find(|(d, _)| d == dir)
            .map(|(_, ids)| ids.clone())
            .unwrap_or_default()
    }

    fn n_fits(&self) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|(_, ids)| ids.len())
            .sum()
    }
}

fn n_parameters_of(features: &FeatureVector) -> usize {
    BASE_N
        + features
            .iter()
            .map(|(_, form)| if form == "hockey" { 2 } else { 1 })
            .sum::<usize>()
}

impl StepFitter for Script {
    fn fit_step(&self, phase_dir: &str, candidates: &[Candidate]) -> Result<RunReport, String> {
        let ids: Vec<String> = candidates.iter().map(|c| c.id.clone()).collect();
        let n_calls = {
            let mut calls = self.calls.lock().unwrap();
            calls.push((phase_dir.to_string(), ids));
            calls.len()
        };
        let mut results = Vec::new();
        for c in candidates {
            let key = c.features.render();
            let ofv = self.ofv.get(&key).copied().unwrap_or(self.fallback);
            if self.erroring.contains(&key) {
                results.push(CandidateResult {
                    id: c.id.clone(),
                    hash: c.hash(),
                    parent: c.parent.clone(),
                    features: c.features.clone(),
                    fit: None,
                    ofv: None,
                    converged: None,
                    verdict: StrictnessVerdict {
                        passed: false,
                        failures: vec!["no fit: does not compile".into()],
                        skipped: vec![],
                    },
                    criterion: f64::NAN,
                    seconds: 0.0,
                    error: Some(CandidateError::model("does not compile")),
                    duplicate_of: None,
                    reused: false,
                });
                continue;
            }
            let mut fit = converged_fit(ofv);
            fit.n_parameters = if self.no_extra_parameter.contains(&key) {
                BASE_N + c.features.len().saturating_sub(1)
            } else {
                n_parameters_of(&c.features)
            };
            let failing = self.failing.contains(&key) || self.failing.contains(&c.id);
            results.push(CandidateResult {
                id: c.id.clone(),
                hash: c.hash(),
                parent: c.parent.clone(),
                features: c.features.clone(),
                fit: Some(fit),
                ofv: Some(ofv),
                converged: Some(!failing),
                verdict: StrictnessVerdict {
                    passed: !failing,
                    failures: if failing {
                        vec!["stalled at the initial estimates (#751)".into()]
                    } else {
                        vec![]
                    },
                    skipped: vec![],
                },
                criterion: ofv,
                seconds: 0.0,
                error: None,
                duplicate_of: None,
                reused: false,
            });
        }
        Ok(RunReport {
            results,
            cancelled: self.cancel_after.is_some_and(|n| n_calls >= n),
            fitted: candidates.len(),
            reused: 0,
            deduped: 0,
            warnings: vec![],
        })
    }
}

fn run(script: &Script, space: Space, options: &CovsearchOptions) -> CovsearchResult {
    search(script, space, options, None).expect("search")
}

fn forward_only() -> CovsearchOptions {
    CovsearchOptions {
        algorithm: Algorithm::ScmForward,
        ..CovsearchOptions::default()
    }
}

fn labels(effects: &[Included]) -> Vec<String> {
    effects.iter().map(|i| i.effect.label()).collect()
}

fn selected(result: &CovsearchResult) -> Vec<(usize, &'static str, String)> {
    result
        .steps
        .iter()
        .filter(|r| r.selected)
        .map(|r| (r.step, r.phase.label(), r.effect.label()))
        .collect()
}

// ── forward ──────────────────────────────────────────────────────────────────

#[test]
fn forward_takes_the_largest_significant_drop_and_stops_when_none_is() {
    // Step 1: CL-WT drops 20, V-WT drops 8, CL-CRCL drops 3 (insignificant).
    // Step 2 from CL-WT: V-WT drops 7 more (significant), CL-CRCL 2.
    // Step 3 from CL-WT + V-WT: CL-CRCL drops 1 — stop.
    let script = Script::new(&[
        ("", 100.0),
        ("CL-WT=power", 80.0),
        ("V-WT=power", 92.0),
        ("CL-CRCL=power", 97.0),
        ("CL-WT=power;V-WT=power", 73.0),
        ("CL-CRCL=power;CL-WT=power", 78.0),
        ("CL-CRCL=power;CL-WT=power;V-WT=power", 72.0),
    ]);
    let result = run(
        &script,
        space(vec![pow("CL", "WT"), pow("V", "WT"), pow("CL", "CRCL")]),
        &forward_only(),
    );
    assert_eq!(result.base_ofv, 100.0);
    assert_eq!(
        selected(&result),
        vec![
            (1, "forward", "CL-WT-power".to_string()),
            (2, "forward", "V-WT-power".to_string())
        ]
    );
    assert_eq!(result.n_steps(), 3);
    assert_eq!(result.final_step, 2);
    assert_eq!(result.final_ofv, 73.0);
    assert_eq!(labels(&result.included), vec!["CL-WT-power", "V-WT-power"]);
    assert_eq!(
        result.final_model.block_lines("covariate_model"),
        vec![
            "CL ~ WT power(center = median)",
            "V ~ WT power(center = median)"
        ]
    );
    // Step 1 fitted three candidates, step 2 two, step 3 one.
    assert_eq!(
        script.dirs(),
        vec!["base", "forward-1", "forward-2", "forward-3"]
    );
    assert_eq!(script.fitted_in("forward-1").len(), 3);
    assert_eq!(script.fitted_in("forward-3"), vec!["f3-CL-CRCL-power"]);

    // The rows carry the numbers the decision was made on.
    let step1: Vec<&StepRow> = result.step_rows(1).collect();
    let cl_wt = step1
        .iter()
        .find(|r| r.effect.label() == "CL-WT-power")
        .unwrap();
    let t = cl_wt.lrt.unwrap();
    assert_eq!(t.dofv, 20.0);
    assert_eq!(t.df, 1);
    assert!(t.significant && t.p_value < 1e-4, "{t:?}");
    assert_eq!(t.alpha, 0.01);
    let crcl = step1
        .iter()
        .find(|r| r.effect.label() == "CL-CRCL-power")
        .unwrap();
    assert!(!crcl.lrt.unwrap().significant);
    assert!(!crcl.selected);
    assert_eq!(crcl.parent_ofv, 100.0);
    // Every row says how its fit ended.
    assert!(result
        .steps
        .iter()
        .all(|r| r.converged == Some(true) && r.passed));
}

#[test]
fn a_lower_ofv_that_fails_strictness_cannot_win_a_step() {
    // CL-WT has the best OFV but stalled at its initial estimates; V-WT is
    // the best *trustworthy* candidate and must be the one selected. This
    // is the issue's fourth validation bullet.
    let mut script = Script::new(&[
        ("", 100.0),
        ("CL-WT=power", 70.0),
        ("V-WT=power", 90.0),
        ("CL-WT=power;V-WT=power", 60.0),
    ]);
    script.failing.push("CL-WT=power".into());
    let result = run(
        &script,
        space(vec![pow("CL", "WT"), pow("V", "WT")]),
        &forward_only(),
    );
    assert_eq!(selected(&result)[0].2, "V-WT-power");
    let stalled = result
        .step_rows(1)
        .find(|r| r.effect.label() == "CL-WT-power")
        .unwrap();
    assert!(!stalled.passed);
    assert_eq!(stalled.converged, Some(false));
    assert!(stalled.failures[0].contains("stalled"));
    // Its ΔOFV is still reported — beside its status, never instead of it.
    assert_eq!(stalled.lrt.unwrap().dofv, 30.0);
    // …and the search does not stop there: CL-WT is tried again from V-WT
    // (where it may well converge) and is accepted.
    assert_eq!(selected(&result).len(), 2);
    assert_eq!(result.final_ofv, 60.0);
}

#[test]
fn a_candidate_that_adds_no_free_parameter_is_unjudgeable() {
    let mut script = Script::new(&[("", 100.0), ("CL-WT=power", 50.0), ("V-WT=power", 90.0)]);
    script.no_extra_parameter.push("CL-WT=power".into());
    let result = run(
        &script,
        space(vec![pow("CL", "WT"), pow("V", "WT")]),
        &forward_only(),
    );
    let row = result
        .step_rows(1)
        .find(|r| r.effect.label() == "CL-WT-power")
        .unwrap();
    assert!(row.lrt.is_none());
    assert!(
        row.note
            .as_ref()
            .unwrap()
            .contains("needs at least one more"),
        "{:?}",
        row.note
    );
    assert_eq!(selected(&result)[0].2, "V-WT-power");
}

#[test]
fn a_candidate_that_does_not_compile_is_a_row_not_a_crash() {
    let mut script = Script::new(&[("", 100.0), ("CL-WT=power", 50.0), ("V-WT=power", 90.0)]);
    script.erroring.push("CL-WT=power".into());
    let result = run(
        &script,
        space(vec![pow("CL", "WT"), pow("V", "WT")]),
        &forward_only(),
    );
    let row = result
        .step_rows(1)
        .find(|r| r.effect.label() == "CL-WT-power")
        .unwrap();
    assert!(row.ofv.is_none() && row.lrt.is_none() && !row.passed);
    assert!(row.failures[0].contains("does not compile"));
    assert_eq!(selected(&result)[0].2, "V-WT-power");
}

#[test]
fn selecting_a_pair_retires_its_other_forms() {
    // CL-WT is offered as power and linear; power wins step 1 and linear is
    // not fitted again — one `[covariate_model]` line per pair.
    let script = Script::new(&[
        ("", 100.0),
        ("CL-WT=power", 80.0),
        ("CL-WT=linear", 85.0),
        ("V-WT=power", 96.0),
        ("CL-WT=power;V-WT=power", 79.0),
    ]);
    let result = run(
        &script,
        space(vec![
            pow("CL", "WT"),
            effect("CL", "WT", CovariateForm::Linear),
            pow("V", "WT"),
        ]),
        &forward_only(),
    );
    assert_eq!(selected(&result).len(), 1);
    assert_eq!(script.fitted_in("forward-2"), vec!["f2-V-WT-power"]);
}

#[test]
fn degrees_of_freedom_come_off_the_fits() {
    // A hockey stick adds two θ, so a drop of 8 that is significant for one
    // parameter (cutoff 6.63) is not for two (cutoff 9.21).
    let script = Script::new(&[("", 100.0), ("CL-WT=hockey", 92.0), ("V-WT=power", 93.0)]);
    let result = run(
        &script,
        space(vec![
            effect("CL", "WT", CovariateForm::Hockey),
            pow("V", "WT"),
        ]),
        &forward_only(),
    );
    let hockey = result
        .step_rows(1)
        .find(|r| r.effect.label() == "CL-WT-hockey")
        .unwrap();
    assert_eq!(hockey.lrt.unwrap().df, 2);
    assert!(!hockey.lrt.unwrap().significant);
    assert_eq!(selected(&result)[0].2, "V-WT-power");
}

#[test]
fn max_steps_bounds_each_phase() {
    let script = Script::new(&[
        ("", 100.0),
        ("CL-WT=power", 80.0),
        ("V-WT=power", 90.0),
        ("CL-WT=power;V-WT=power", 60.0),
    ]);
    let options = CovsearchOptions {
        max_steps: Some(1),
        ..CovsearchOptions::default()
    };
    let result = run(
        &script,
        space(vec![pow("CL", "WT"), pow("V", "WT")]),
        &options,
    );
    // One forward step, then one backward step (which keeps CL-WT: removing
    // it costs 20 at p = 0.001, cutoff 10.83).
    assert_eq!(
        selected(&result),
        vec![(1, "forward", "CL-WT-power".to_string())]
    );
    assert_eq!(script.dirs(), vec!["base", "forward-1", "backward-2"]);
    assert_eq!(result.final_ofv, 80.0);
}

#[test]
fn no_candidates_returns_the_base_fit_untouched() {
    let script = Script::new(&[("", 100.0)]);
    let result = run(&script, space(vec![]), &CovsearchOptions::default());
    assert!(result.steps.is_empty());
    assert_eq!(result.final_step, 0);
    assert_eq!(result.final_ofv, 100.0);
    assert_eq!(result.final_model.render(), result.base_model.render());
    assert_eq!(script.dirs(), vec!["base"]);
    assert!(result.included.is_empty());
}

// ── backward ─────────────────────────────────────────────────────────────────

#[test]
fn backward_removes_the_cheapest_insignificant_effect_and_keeps_the_rest() {
    // Forward at p = 0.01 adds CL-WT (−20), then V-WT (−7), then CL-CRCL
    // (−6.7, just over 6.63). Backward at p = 0.001 (cutoff 10.83): removing
    // CL-CRCL costs 6.7 and V-WT 7.0 — both removable, CL-CRCL is cheaper and
    // goes first. Then removing V-WT from {CL-WT, V-WT} costs 7.0 — removable
    // too. Then removing CL-WT costs 20 — kept. Final: CL-WT only.
    let script = Script::new(&[
        ("", 100.0),
        ("CL-WT=power", 80.0),
        ("V-WT=power", 93.0),
        ("CL-CRCL=power", 95.0),
        ("CL-WT=power;V-WT=power", 73.0),
        ("CL-CRCL=power;CL-WT=power", 74.0),
        ("CL-CRCL=power;CL-WT=power;V-WT=power", 66.3),
    ]);
    let result = run(
        &script,
        space(vec![pow("CL", "WT"), pow("V", "WT"), pow("CL", "CRCL")]),
        &CovsearchOptions::default(),
    );
    assert_eq!(
        selected(&result),
        vec![
            (1, "forward", "CL-WT-power".to_string()),
            (2, "forward", "V-WT-power".to_string()),
            (3, "forward", "CL-CRCL-power".to_string()),
            (4, "backward", "CL-CRCL-power".to_string()),
            (5, "backward", "V-WT-power".to_string()),
        ]
    );
    assert_eq!(result.n_steps(), 6);
    assert_eq!(labels(&result.included), vec!["CL-WT-power"]);
    assert_eq!(result.final_ofv, 80.0);
    assert_eq!(result.final_step, 5);

    // The backward rows describe the *removal*: the increase and whether the
    // effect had to stay.
    let step4: Vec<&StepRow> = result.step_rows(4).collect();
    assert_eq!(step4.len(), 3);
    let crcl = step4
        .iter()
        .find(|r| r.effect.label() == "CL-CRCL-power")
        .unwrap();
    let t = crcl.lrt.unwrap();
    assert!((t.dofv - 6.7).abs() < 1e-9, "{t:?}");
    assert_eq!(t.alpha, 0.001);
    assert!(!t.significant);
    let wt = step4
        .iter()
        .find(|r| r.effect.label() == "CL-WT-power")
        .unwrap();
    assert!(wt.lrt.unwrap().significant);
    assert!(!wt.selected);
    let step6: Vec<&StepRow> = result.step_rows(6).collect();
    assert_eq!(step6.len(), 1);
    assert!(step6[0].lrt.unwrap().significant && !step6[0].selected);
    // The dropped relations are gone from the final text.
    assert_eq!(
        result.final_model.block_lines("covariate_model"),
        vec!["CL ~ WT power(center = median)"]
    );
}

#[test]
fn backward_never_touches_forced_or_base_relations() {
    let mut base = ModelText::parse(BASE).unwrap();
    base.apply(ModelEdit::AddCovariateRelation(pow("KA", "SEX").relation()))
        .unwrap();
    let space = Space {
        base_model: base,
        candidates: vec![pow("CL", "WT")],
        included: vec![Included {
            effect: pow("KA", "SEX"),
            origin: Origin::Forced,
        }],
        notes: vec![],
    };
    let script = Script::new(&[("KA-SEX=power", 100.0), ("CL-WT=power;KA-SEX=power", 80.0)]);
    let result = run(&script, space, &CovsearchOptions::default());
    assert_eq!(
        labels(&result.included),
        vec!["KA-SEX-power", "CL-WT-power"]
    );
    // The backward step offered exactly one removal.
    assert_eq!(script.fitted_in("backward-2"), vec!["b2-CL-WT-power"]);
    assert_eq!(result.included[0].origin, Origin::Forced);
    assert_eq!(result.included[1].origin, Origin::Forward(1));
}

#[test]
fn backward_stops_when_a_strictness_failure_is_the_only_removable_one() {
    let mut script = Script::new(&[("", 100.0), ("CL-WT=power", 80.0)]);
    // Removing CL-WT gives back the base model, whose refit now stalls.
    script.failing.push("b2-CL-WT-power".into());
    let result = run(
        &script,
        space(vec![pow("CL", "WT")]),
        &CovsearchOptions::default(),
    );
    assert_eq!(labels(&result.included), vec!["CL-WT-power"]);
    let row = result.step_rows(2).next().unwrap();
    assert!(!row.passed && !row.selected);
}

// ── adaptive scope reduction ─────────────────────────────────────────────────

#[test]
fn adaptive_scope_reduction_stashes_insignificant_effects_and_retests_them() {
    // Step 1: CL-WT wins; V-WT (−3) and CL-CRCL (−2) are insignificant and
    // stashed, so step 2 has nothing left and the ordinary forward phase
    // ends after one step. The adaptive pass then re-tests both from
    // CL-WT, where V-WT is now significant (−8) and CL-CRCL still not.
    let script = Script::new(&[
        ("", 100.0),
        ("CL-WT=power", 80.0),
        ("V-WT=power", 97.0),
        ("CL-CRCL=power", 98.0),
        ("CL-WT=power;V-WT=power", 72.0),
        ("CL-CRCL=power;CL-WT=power", 79.0),
        ("CL-CRCL=power;CL-WT=power;V-WT=power", 71.5),
    ]);
    let options = CovsearchOptions {
        algorithm: Algorithm::ScmForward,
        adaptive_scope_reduction: true,
        ..CovsearchOptions::default()
    };
    let result = run(
        &script,
        space(vec![pow("CL", "WT"), pow("V", "WT"), pow("CL", "CRCL")]),
        &options,
    );
    assert_eq!(
        script.dirs(),
        vec!["base", "forward-1", "adaptive-2", "adaptive-3"]
    );
    assert_eq!(script.fitted_in("adaptive-2").len(), 2);
    assert_eq!(
        selected(&result),
        vec![
            (1, "forward", "CL-WT-power".to_string()),
            (2, "adaptive", "V-WT-power".to_string()),
        ]
    );
    assert_eq!(result.final_ofv, 72.0);
    // Without the stash the same space costs one more fit.
    let plain = Script::new(&[
        ("", 100.0),
        ("CL-WT=power", 80.0),
        ("V-WT=power", 97.0),
        ("CL-CRCL=power", 98.0),
        ("CL-WT=power;V-WT=power", 72.0),
        ("CL-CRCL=power;CL-WT=power", 79.0),
        ("CL-CRCL=power;CL-WT=power;V-WT=power", 71.5),
    ]);
    let without = run(
        &plain,
        space(vec![pow("CL", "WT"), pow("V", "WT"), pow("CL", "CRCL")]),
        &forward_only(),
    );
    assert_eq!(without.final_ofv, 72.0);
    assert_eq!(plain.n_fits(), script.n_fits());
    assert_eq!(
        plain.dirs(),
        vec!["base", "forward-1", "forward-2", "forward-3"]
    );
}

#[test]
fn adaptive_stash_keeps_a_significant_runner_up_in_play() {
    // Step 1: CL-WT-power wins; CL-WT-linear is significant too but loses;
    // V-WT is insignificant and stashed. Step 2 must still offer nothing on
    // CL-WT (the pair is taken) — and the stash must not have thrown away
    // anything else that was significant.
    let script = Script::new(&[
        ("", 100.0),
        ("CL-WT=power", 80.0),
        ("CL-WT=linear", 82.0),
        ("V-WT=power", 98.0),
        ("CL-CRCL=power", 85.0),
        ("CL-CRCL=power;CL-WT=power", 70.0),
        ("CL-WT=power;V-WT=power", 79.0),
        ("CL-CRCL=power;CL-WT=power;V-WT=power", 69.0),
    ]);
    let options = CovsearchOptions {
        algorithm: Algorithm::ScmForward,
        adaptive_scope_reduction: true,
        ..CovsearchOptions::default()
    };
    let result = run(
        &script,
        space(vec![
            pow("CL", "WT"),
            effect("CL", "WT", CovariateForm::Linear),
            pow("V", "WT"),
            pow("CL", "CRCL"),
        ]),
        &options,
    );
    assert_eq!(script.fitted_in("forward-2"), vec!["f2-CL-CRCL-power"]);
    assert_eq!(
        selected(&result),
        vec![
            (1, "forward", "CL-WT-power".to_string()),
            (2, "forward", "CL-CRCL-power".to_string()),
        ]
    );
    // The adaptive pass re-tested V-WT alone.
    assert_eq!(script.fitted_in("adaptive-3"), vec!["a3-V-WT-power"]);
}

// ── cancellation ─────────────────────────────────────────────────────────────

#[test]
fn a_cancelled_step_ends_the_search_with_what_it_had() {
    let mut script = Script::new(&[("", 100.0), ("CL-WT=power", 80.0), ("V-WT=power", 90.0)]);
    script.cancel_after = Some(2);
    let result = run(
        &script,
        space(vec![pow("CL", "WT"), pow("V", "WT")]),
        &CovsearchOptions::default(),
    );
    assert!(result.cancelled);
    assert_eq!(script.dirs(), vec!["base", "forward-1"]);
    // The rows of the cancelled step are kept, but its winner is not taken:
    // the run may not have reached every candidate.
    assert_eq!(result.step_rows(1).count(), 2);
    assert_eq!(result.final_step, 0);
    assert_eq!(result.final_ofv, 100.0);
}

// ── options ──────────────────────────────────────────────────────────────────

const EXAMPLES: &str = "../../examples";

fn config(mfl: &str, extra: &str) -> Result<SearchConfig, String> {
    let text = format!(
        "base = \"two_cpt_oral_covmodel.ferx\"\ndata = \"../data/two_cpt_oral_cov.csv\"\n\
         [space]\nmfl = \"{mfl}\"\n{extra}"
    );
    SearchConfig::from_str(&text, Path::new(EXAMPLES))
}

#[test]
fn options_default_to_pharmpys_and_read_the_section() {
    let cfg = config("COVARIATE?(CL, WT, pow)", "").unwrap();
    let o = CovsearchOptions::from_config(&cfg).unwrap();
    assert_eq!(o, CovsearchOptions::default());
    assert_eq!(o.algorithm, Algorithm::ScmForwardThenBackward);
    assert_eq!((o.p_forward, o.p_backward), (0.01, 0.001));
    assert_eq!(o.max_steps, None);
    assert!(!o.adaptive_scope_reduction);

    let cfg = config(
        "COVARIATE?(CL, WT, pow)",
        "[covsearch]\nalgorithm = \"scm-forward\"\np_forward = 0.05\np_backward = 0.01\n\
         max_steps = 3\nadaptive_scope_reduction = true\n[rank]\ntype = \"ofv\"\n",
    )
    .unwrap();
    let o = CovsearchOptions::from_config(&cfg).unwrap();
    assert_eq!(o.algorithm, Algorithm::ScmForward);
    assert_eq!((o.p_forward, o.p_backward), (0.05, 0.01));
    assert_eq!(o.max_steps, Some(3));
    assert!(o.adaptive_scope_reduction);
}

#[test]
fn options_refuse_what_covsearch_cannot_honour() {
    let e = |extra: &str| {
        let cfg = config("COVARIATE?(CL, WT, pow)", extra).unwrap();
        CovsearchOptions::from_config(&cfg).unwrap_err()
    };
    assert!(e("[covsearch]\np_forward = 0.0\n").contains("p_forward = 0: must be a probability"));
    assert!(e("[covsearch]\np_backward = 1.5\n").contains("p_backward = 1.5"));
    assert!(e("[covsearch]\nmax_steps = 0\n").contains("max_steps = 0"));
    assert!(e("[covsearch]\nalgorithm = \"samba\"\n").contains("[covsearch]"));
    assert!(e("[covsearch]\np_fwd = 0.01\n").contains("p_fwd"));
    let bic = e("[rank]\ntype = \"bic\"\n");
    assert!(
        bic.contains("[rank] type = \"bic\"") && bic.contains("likelihood-ratio"),
        "{bic}"
    );
    let cutoff = e("[rank]\ncutoff = 3.84\n");
    assert!(cutoff.contains("[rank] cutoff = 3.84"), "{cutoff}");
}

// ── the space ────────────────────────────────────────────────────────────────

#[test]
fn the_space_keeps_base_relations_forces_structural_ones_and_drops_overlaps() {
    // The example base declares CL~WT, CL~CRCL, V1~WT. The space explores
    // WT and CRCL on every η-parameter (five of them) and forces KA~CRCL.
    let cfg = config(
        "COVARIATE?(@IIV, @CONTINUOUS, pow); COVARIATE(KA, CRCL, exp)",
        "",
    )
    .unwrap();
    let base = cfg.load_base().unwrap();
    let space = Space::from_config(&cfg, &base).unwrap();
    let candidates: Vec<String> = space.candidates.iter().map(Effect::label).collect();
    // 5 × 2 = 10 effects, minus the three the base has, minus KA-CRCL (forced).
    assert_eq!(
        candidates,
        vec![
            "V1-CRCL-power",
            "Q-WT-power",
            "Q-CRCL-power",
            "V2-WT-power",
            "V2-CRCL-power",
            "KA-WT-power",
        ]
    );
    assert_eq!(
        labels(&space.included),
        vec![
            "CL-WT-power",
            "CL-CRCL-power",
            "V1-WT-power",
            "KA-CRCL-exponential"
        ]
    );
    assert_eq!(space.included[3].origin, Origin::Forced);
    assert_eq!(
        space
            .base_model
            .block_lines("covariate_model")
            .last()
            .unwrap(),
        "KA ~ CRCL exponential(center = median)"
    );
    assert_eq!(space.notes.len(), 3, "{:?}", space.notes);
    assert!(space.notes[0].contains("not explored: CL-WT-power"));
}

#[test]
fn a_forced_effect_the_base_already_has_is_kept_not_duplicated() {
    let cfg = config("COVARIATE(CL, WT, pow); COVARIATE?(KA, WT, pow)", "").unwrap();
    let base = cfg.load_base().unwrap();
    let space = Space::from_config(&cfg, &base).unwrap();
    assert_eq!(space.base_model.render(), base.text.render());
    assert!(space.notes[0].contains("forced effect CL-WT-power is already in the base model"));
    assert_eq!(space.candidates.len(), 1);
}

#[test]
fn a_structural_statement_is_refused_by_name() {
    let cfg = config("PERIPHERALS(0..1); COVARIATE?(CL, WT, pow)", "").unwrap();
    let base = cfg.load_base().unwrap();
    let e = Space::from_config(&cfg, &base).unwrap_err();
    assert!(
        e.contains("`PERIPHERALS` is not a covariate statement"),
        "{e}"
    );
    assert!(e.contains("modelsearch"));
}

// ── the report ───────────────────────────────────────────────────────────────

#[test]
fn the_report_writes_the_step_table_and_the_seeded_final_model() {
    let script = Script::new(&[("", 100.0), ("CL-WT=power", 80.0), ("V-WT=power", 96.0)]);
    let result = run(
        &script,
        space(vec![pow("CL", "WT"), pow("V", "WT")]),
        &CovsearchOptions::default(),
    );
    let dir = tempfile::tempdir().unwrap();
    write_report(dir.path(), &result).unwrap();

    let table = std::fs::read_to_string(steps_path(dir.path())).unwrap();
    let mut lines = table.lines();
    assert_eq!(lines.next().unwrap(), STEP_COLUMNS.join(","));
    let rows: Vec<Vec<&str>> = lines.map(|l| l.split(',').collect()).collect();
    // Step 1: two forward rows; step 2: one forward (V-WT, insignificant);
    // step 3: one backward (CL-WT kept).
    assert_eq!(rows.len(), 4);
    assert_eq!(
        &rows[0][..6],
        &["1", "forward", "f1-CL-WT-power", "CL", "WT", "power"]
    );
    assert_eq!(rows[0][6], "100.000000");
    assert_eq!(rows[0][7], "80.000000");
    assert_eq!(rows[0][8], "20.000000");
    assert_eq!(rows[0][9], "1");
    assert_eq!(rows[0][11], "0.010000");
    assert_eq!(&rows[0][12..16], &["true", "true", "true", "true"]);
    assert_eq!(rows[3][1], "backward");
    assert_eq!(rows[3][11], "0.001000");
    assert_eq!(rows[3][13], "false");

    let final_model = std::fs::read_to_string(final_model_path(dir.path())).unwrap();
    assert!(final_model.contains("CL ~ WT power(center = median)"));
    // The fixture fit is the warfarin evaluation, whose θ names match this
    // base model, so its estimates were written into the initial values.
    let fit = converged_fit(80.0);
    let tvcl = fit.theta[fit.theta_names.iter().position(|n| n == "TVCL").unwrap()];
    assert!(
        final_model.contains(&format!("theta TVCL({tvcl}")),
        "{final_model}"
    );

    let summary = render_summary(&result);
    assert!(summary.contains("Base model: OFV 100.000"));
    assert!(summary.contains("SELECTED"));
    assert!(summary.contains("Final model: OFV 80.000 (step 1), 1 relation"));
    assert!(summary.contains("CL ~ WT power  [forward step 1]"));
}

#[test]
fn default_dir_sits_next_to_the_config() {
    assert_eq!(
        default_dir(Path::new("runs/warfarin.ferxsearch")),
        Path::new("runs/warfarin-covsearch")
    );
}
