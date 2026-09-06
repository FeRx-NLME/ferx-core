//! Tier-1 tests for the structural search logic (#1181).
//!
//! Every test drives [`search`] with a *scripted* fitter — OFVs keyed by the
//! candidate's structure — so what is under test is the enumeration and the
//! decisions: which candidates each algorithm generates from which parents,
//! what the reduced-stepwise collapse keeps, what wins, how a cutoff and a
//! failed gate change that, and what the report says. The path from
//! `ModelText` through the runner and `fit()` is
//! `tests/modelsearch_end_to_end.rs`; the enumeration against Pharmpy's own
//! is `pharmpy_anchor.rs`.

use std::collections::HashMap;
use std::sync::Mutex;

use ferx_core::edit::ModelText;
use ferx_core::StrictnessVerdict;

use super::structure::space_features;
use super::*;
use crate::search::mfl::Mfl;
use crate::search::test_support::converged_fit;
use crate::search::{CandidateError, CandidateResult, RunReport};

pub(crate) const BASE: &str = "\
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

[error_model]
  DV ~ proportional(PROP_ERR)
";

pub(crate) const BASE_IV: &str = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
";

fn fo(peripherals: u32) -> Structure {
    Structure {
        absorption: Absorption::Fo,
        peripherals,
        transits: None,
        lagtime: false,
    }
}

pub(crate) fn defaults(text: &ModelText) -> Defaults {
    let lines = text.block_lines("parameters");
    let mut names = Vec::new();
    let mut inits = Vec::new();
    for l in &lines {
        if let Some(rest) = l.strip_prefix("theta ") {
            let (name, args) = rest.split_once('(').unwrap();
            names.push(name.trim().to_string());
            inits.push(args.split(',').next().unwrap().trim().parse().unwrap());
        }
    }
    let etas = lines
        .iter()
        .filter_map(|l| l.strip_prefix("omega "))
        .map(|l| l.split('~').next().unwrap().trim().to_string())
        .collect();
    let params = text
        .block_lines("individual_parameters")
        .iter()
        .map(|l| l.split('=').next().unwrap().trim().to_string())
        .collect();
    Defaults::new(
        params,
        names,
        inits,
        etas,
        &crate::search::test_support::population(&["1"]),
    )
}

fn space_of(base: &str, structure: Structure, mfl: &str) -> Space {
    let text = ModelText::parse(base).unwrap();
    let features = space_features(&Mfl::parse(mfl).unwrap()).unwrap();
    let d = defaults(&text);
    Space::build(text, structure, &features, d, Vec::new()).expect("a buildable base")
}

fn space(mfl: &str) -> Space {
    space_of(BASE, fo(0), mfl)
}

/// The scripted fitter. OFVs are keyed by the candidate's structure (its
/// feature vector, `ABSORPTION=FO;LAGTIME=ON;PERIPHERALS=1;TRANSITS=0`) or by
/// id; a candidate in neither table gets `fallback`.
struct Script {
    ofv: HashMap<String, f64>,
    ofv_by_id: HashMap<String, f64>,
    fallback: f64,
    /// Ids or structure keys whose candidate fails the strictness gate.
    failing: Vec<String>,
    /// Ids whose candidate produces no fit at all.
    erroring: Vec<String>,
    /// Ids reported as canonical duplicates of another id: no fit of their
    /// own, `duplicate_of` set, the representative's score.
    duplicates: HashMap<String, String>,
    /// Ids reported as a resumed row whose cached fit is gone: an OFV and a
    /// verdict, no `FitResult`.
    fitless: Vec<String>,
    /// TVCL written into the fit of the named id, so a child seeded from it
    /// can be told apart.
    theta_override: HashMap<String, f64>,
    cancel_after: Option<usize>,
    /// `(step_dir, candidates)` per `fit_step` call, in order.
    calls: Mutex<Vec<(String, Vec<Candidate>)>>,
}

impl Script {
    fn new(table: &[(&str, f64)]) -> Script {
        Script {
            ofv: table.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            ofv_by_id: HashMap::new(),
            fallback: 1000.0,
            failing: Vec::new(),
            erroring: Vec::new(),
            duplicates: HashMap::new(),
            fitless: Vec::new(),
            theta_override: HashMap::new(),
            cancel_after: None,
            calls: Mutex::new(Vec::new()),
        }
    }

    fn by_id(mut self, table: &[(&str, f64)]) -> Script {
        self.ofv_by_id = table.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        self
    }

    fn dirs(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|(d, _)| d.clone())
            .collect()
    }

    fn candidates_in(&self, dir: &str) -> Vec<Candidate> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .find(|(d, _)| d == dir)
            .map(|(_, c)| c.clone())
            .unwrap_or_default()
    }

    fn n_fits(&self) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|(_, c)| c.len())
            .sum()
    }
}

impl StepFitter for Script {
    fn fit_step(&self, step_dir: &str, candidates: &[Candidate]) -> Result<RunReport, String> {
        let n_calls = {
            let mut calls = self.calls.lock().unwrap();
            calls.push((step_dir.to_string(), candidates.to_vec()));
            calls.len()
        };
        let mut results = Vec::new();
        for c in candidates {
            let key = c.features.render();
            let ofv = self
                .ofv_by_id
                .get(&c.id)
                .or_else(|| self.ofv.get(&key))
                .copied()
                .unwrap_or(self.fallback);
            if self.erroring.contains(&c.id) {
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
            let failing = self.failing.contains(&key) || self.failing.contains(&c.id);
            let mut fit = converged_fit(ofv);
            if let Some(v) = self.theta_override.get(&c.id) {
                fit.theta[0] = *v;
            }
            let duplicate_of = self.duplicates.get(&c.id).cloned();
            let has_fit = duplicate_of.is_none() && !self.fitless.contains(&c.id);
            results.push(CandidateResult {
                id: c.id.clone(),
                hash: c.hash(),
                parent: c.parent.clone(),
                features: c.features.clone(),
                fit: has_fit.then_some(fit),
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
                seconds: 1.5,
                error: None,
                reused: self.fitless.contains(&c.id),
                duplicate_of,
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

fn options(algorithm: Algorithm) -> ModelsearchOptions {
    ModelsearchOptions {
        algorithm,
        rank: RankType::Ofv,
        ..ModelsearchOptions::default()
    }
}

fn run(script: &Script, space: Space, options: &ModelsearchOptions) -> ModelsearchResult {
    search(script, space, options, None).expect("search")
}

fn paths(result: &ModelsearchResult, layer: usize) -> Vec<(String, String, String)> {
    result
        .layer_rows(layer)
        .map(|r| {
            (
                r.id.clone(),
                r.parent.clone().unwrap_or_default(),
                r.path
                    .iter()
                    .map(|k| k.to_string())
                    .collect::<Vec<_>>()
                    .join(";"),
            )
        })
        .collect()
}

fn triple(id: &str, parent: &str, path: &str) -> (String, String, String) {
    (id.into(), parent.into(), path.into())
}

const FO_LAG: &str = "ABSORPTION=FO;LAGTIME=ON;PERIPHERALS=0;TRANSITS=0";
const FO_P1: &str = "ABSORPTION=FO;LAGTIME=OFF;PERIPHERALS=1;TRANSITS=0";
const FO_P1_LAG: &str = "ABSORPTION=FO;LAGTIME=ON;PERIPHERALS=1;TRANSITS=0";
const FO_P2: &str = "ABSORPTION=FO;LAGTIME=OFF;PERIPHERALS=2;TRANSITS=0";
const FO_P2_LAG: &str = "ABSORPTION=FO;LAGTIME=ON;PERIPHERALS=2;TRANSITS=0";
const FO_BASE: &str = "ABSORPTION=FO;LAGTIME=OFF;PERIPHERALS=0;TRANSITS=0";

// ── enumeration ─────────────────────────────────────────────────────────────

#[test]
fn exhaustive_stepwise_applies_one_feature_at_a_time_in_every_order() {
    let script = Script::new(&[
        (FO_BASE, 100.0),
        (FO_LAG, 95.0),
        (FO_P1, 90.0),
        (FO_P1_LAG, 80.0),
    ]);
    let result = run(
        &script,
        space("PERIPHERALS(0..1); LAGTIME([OFF,ON])"),
        &options(Algorithm::ExhaustiveStepwise),
    );
    // Pharmpy's dictionary order: LAGTIME before PERIPHERALS.
    assert_eq!(
        paths(&result, 1),
        vec![
            triple("run1", "base", "LAGTIME(ON)"),
            triple("run2", "base", "PERIPHERALS(1)"),
        ]
    );
    assert_eq!(
        paths(&result, 2),
        vec![
            triple("run3", "run1", "LAGTIME(ON);PERIPHERALS(1)"),
            triple("run4", "run2", "PERIPHERALS(1);LAGTIME(ON)"),
        ]
    );
    assert_eq!(result.n_layers(), 2);
    assert_eq!(script.dirs(), vec!["base", "layer-1", "layer-2"]);
    assert_eq!(result.base_id, "base");
    assert_eq!(result.rows.len(), 5);
    // Both orders land on the same structure, and the earlier wins the tie.
    assert_eq!(result.final_id, "run3");
    assert_eq!(result.row("run3").unwrap().rank, Some(1));
    assert_eq!(result.row("run4").unwrap().rank, Some(2));
    assert_eq!(result.row("base").unwrap().rank, Some(5));
    assert!(result.rows.iter().all(|r| r.continued));
    assert_eq!(result.row("run3").unwrap().d_criterion, Some(-20.0));
    assert!(!result.cancelled);
}

#[test]
fn the_candidate_text_is_the_edited_model_seeded_from_its_parent() {
    let script = Script::new(&[(FO_BASE, 100.0), (FO_P1, 90.0)]);
    let _ = run(
        &script,
        space("PERIPHERALS(0..1)"),
        &options(Algorithm::ExhaustiveStepwise),
    );
    let layer1 = script.candidates_in("layer-1");
    assert_eq!(layer1.len(), 1);
    let c = &layer1[0];
    assert_eq!(c.id, "run1");
    assert_eq!(c.parent.as_deref(), Some("base"));
    let structural = c.model.block_lines("structural_model");
    assert_eq!(
        structural,
        vec!["pk two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, ka=KA)"]
    );
    // The new parameters scale from the parent's *seeded* estimates — the
    // fixture fit round-trips the file's inits through the packing, so the
    // seeded TVV is 10.000000000000002 and V2 follows it, not the file.
    let inits = super::structure::theta_inits_of(&c.model);
    assert_eq!(inits["TVQ"], inits["TVCL"], "{inits:?}");
    assert!(
        (inits["TVV2"] - 0.05 * inits["TVV"]).abs() < 1e-15,
        "{inits:?}"
    );
    let params = c.model.block_lines("parameters");
    assert!(
        params
            .iter()
            .any(|l| l.starts_with("theta TVQ(") && l.ends_with(", 0.0, 1000000.0)")),
        "{params:?}"
    );
    let indiv = c.model.block_lines("individual_parameters");
    assert!(indiv.contains(&"Q = TVQ".to_string()), "{indiv:?}");
    assert!(indiv.contains(&"V2 = TVV2".to_string()), "{indiv:?}");
}

#[test]
fn reduced_stepwise_extends_only_the_best_of_each_feature_set() {
    // funcs: LAGTIME(ON), PERIPHERALS(1), PERIPHERALS(2). Layer 1: L, P1
    // (P2 cannot come first). Layer 2: L→P1, P1→L, P1→P2. {L,P1} and
    // {P1,L} share a feature set and both still have a move (P2), so only
    // the lower OFV continues; layer 3 has one child from it plus P1→P2→L.
    let script = Script::new(&[
        (FO_BASE, 100.0),
        (FO_LAG, 95.0),
        (FO_P1, 90.0),
        (FO_P2, 92.0),
        (FO_P2_LAG, 70.0),
    ])
    .by_id(&[("run3", 85.0), ("run4", 80.0)]);
    let result = run(
        &script,
        space("PERIPHERALS(0..2); LAGTIME([OFF,ON])"),
        &options(Algorithm::ReducedStepwise),
    );
    assert_eq!(
        paths(&result, 2),
        vec![
            triple("run3", "run1", "LAGTIME(ON);PERIPHERALS(1)"),
            triple("run4", "run2", "PERIPHERALS(1);LAGTIME(ON)"),
            triple("run5", "run2", "PERIPHERALS(1);PERIPHERALS(2)"),
        ]
    );
    assert!(!result.row("run3").unwrap().continued);
    assert!(result.row("run4").unwrap().continued);
    assert!(
        result.row("run5").unwrap().continued,
        "a group of one is not collapsed"
    );
    assert_eq!(
        paths(&result, 3),
        vec![
            triple("run6", "run4", "PERIPHERALS(1);LAGTIME(ON);PERIPHERALS(2)"),
            triple("run7", "run5", "PERIPHERALS(1);PERIPHERALS(2);LAGTIME(ON)"),
        ]
    );
    // Layer 3's two models share a set but have no move left: no collapse.
    assert!(result.layer_rows(3).all(|r| r.continued));
    assert_eq!(script.n_fits(), 1 + 2 + 3 + 2);
    assert_eq!(result.final_id, "run6");

    // The same space under exhaustive_stepwise extends both orders.
    let script = Script::new(&[(FO_BASE, 100.0)]);
    let result = run(
        &script,
        space("PERIPHERALS(0..2); LAGTIME([OFF,ON])"),
        &options(Algorithm::ExhaustiveStepwise),
    );
    assert_eq!(result.layer_rows(3).count(), 3);
    assert_eq!(script.n_fits(), 1 + 2 + 3 + 3);
}

#[test]
fn reduced_stepwise_prefers_a_model_that_passed_the_gate() {
    let mut script = Script::new(&[(FO_BASE, 100.0)]).by_id(&[("run3", 50.0), ("run4", 80.0)]);
    script.failing.push("run3".into());
    let result = run(
        &script,
        space("PERIPHERALS(0..2); LAGTIME([OFF,ON])"),
        &options(Algorithm::ReducedStepwise),
    );
    // run3 has the lower OFV but failed the gate; run4 continues.
    assert!(!result.row("run3").unwrap().continued);
    assert!(result.row("run4").unwrap().continued);
    assert_eq!(
        result.layer_rows(3).next().unwrap().parent.as_deref(),
        Some("run4")
    );
    // When neither passed, the lower OFV continues, as in Pharmpy.
    let mut script = Script::new(&[(FO_BASE, 100.0)]).by_id(&[("run3", 50.0), ("run4", 80.0)]);
    script
        .failing
        .extend(["run3".to_string(), "run4".to_string()]);
    let result = run(
        &script,
        space("PERIPHERALS(0..2); LAGTIME([OFF,ON])"),
        &options(Algorithm::ReducedStepwise),
    );
    assert!(result.row("run3").unwrap().continued);
    assert!(!result.row("run4").unwrap().continued);
}

#[test]
fn exhaustive_generates_every_combination_from_the_base_in_one_layer() {
    let script = Script::new(&[(FO_BASE, 100.0), (FO_P2_LAG, 70.0)]);
    let result = run(
        &script,
        space("PERIPHERALS(0..2); LAGTIME([OFF,ON])"),
        &options(Algorithm::Exhaustive),
    );
    // Pharmpy's `product` over `(None, …)` per category, the first category
    // (LAGTIME) outermost: the combinations without it come first.
    assert_eq!(
        paths(&result, 1),
        vec![
            triple("run1", "base", "PERIPHERALS(1)"),
            triple("run2", "base", "PERIPHERALS(2)"),
            triple("run3", "base", "LAGTIME(ON)"),
            triple("run4", "base", "LAGTIME(ON);PERIPHERALS(1)"),
            triple("run5", "base", "LAGTIME(ON);PERIPHERALS(2)"),
        ]
    );
    assert_eq!(result.n_layers(), 1);
    assert_eq!(script.dirs(), vec!["base", "candidates"]);
    assert_eq!(result.final_id, "run5");
    // Every candidate derives from the base, whatever its structure.
    let candidates = script.candidates_in("candidates");
    assert_eq!(
        candidates[4].model.block_lines("structural_model"),
        vec!["pk three_cpt_oral(cl=CL, v1=V, q2=Q, v2=V2, q3=Q3, v3=V3, ka=KA, lagtime=ALAG)"]
    );
}

#[test]
fn exhaustive_skips_pharmpys_unsupported_pairs() {
    // INST with a lag time, and INST with transits, are never built; FO
    // with three transits is.
    let script = Script::new(&[(FO_BASE, 100.0)]);
    let result = run(
        &script,
        space_of(
            BASE_IV,
            Structure {
                absorption: Absorption::Inst,
                ..fo(0)
            },
            "ABSORPTION([INST,FO]); LAGTIME([OFF,ON]); TRANSITS([0,3], NODEPOT)",
        ),
        &options(Algorithm::Exhaustive),
    );
    let generated: Vec<String> = result
        .layer_rows(1)
        .map(|r| {
            r.path
                .iter()
                .map(|k| k.to_string())
                .collect::<Vec<_>>()
                .join(";")
        })
        .collect();
    assert_eq!(
        generated,
        vec![
            "ABSORPTION(FO)",
            "ABSORPTION(FO);TRANSITS(3, NODEPOT)",
            "ABSORPTION(FO);LAGTIME(ON)",
        ]
    );
    // What Pharmpy's pair table refuses (`LAGTIME(ON)` with transits) is not
    // generated and not noted — it is the rule, not a gap. A combination
    // Pharmpy would build and ferx cannot (a lag or a chain on the bolus
    // base) is noted by name.
    let noted: Vec<&String> = result
        .notes
        .iter()
        .filter(|n| n.starts_with("not generated"))
        .collect();
    assert_eq!(noted.len(), 2, "{:?}", result.notes);
    assert!(noted[0].contains("TRANSITS(3, NODEPOT)"), "{}", noted[0]);
    assert!(
        noted[1].contains("LAGTIME(ON)") && !noted[1].contains("TRANSITS"),
        "{}",
        noted[1]
    );
}

#[test]
fn a_feature_the_base_already_has_is_never_a_move() {
    // The base is FO with no lag: ABSORPTION(FO) and LAGTIME(OFF) are
    // filtered out, as Pharmpy's `filter_mfl_statements` does.
    let script = Script::new(&[(FO_BASE, 100.0)]);
    let result = run(
        &script,
        space("ABSORPTION(FO); LAGTIME([OFF,ON]); PERIPHERALS(0)"),
        &options(Algorithm::ExhaustiveStepwise),
    );
    assert_eq!(
        paths(&result, 1),
        vec![triple("run1", "base", "LAGTIME(ON)")]
    );
    assert_eq!(result.rows.len(), 2);
}

#[test]
fn a_single_point_space_fits_the_base_only() {
    let script = Script::new(&[(FO_BASE, 100.0)]);
    let result = run(
        &script,
        space("ABSORPTION(FO); PERIPHERALS(0); LAGTIME(OFF)"),
        &options(Algorithm::ReducedStepwise),
    );
    assert_eq!(script.n_fits(), 1);
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.final_id, "base");
    assert!(result.row("base").unwrap().selected);
    assert_eq!(result.final_model.render(), BASE);
}

#[test]
fn an_input_outside_the_space_is_fitted_then_moved_onto_it() {
    // An IV base in an FO-only space: the input is fitted, the base is the
    // input with ABSORPTION(FO), and the candidates move from the base.
    let script = Script::new(&[(FO_BASE, 100.0), (FO_LAG, 90.0)]).by_id(&[("input", 120.0)]);
    let result = run(
        &script,
        space_of(
            BASE_IV,
            Structure {
                absorption: Absorption::Inst,
                ..fo(0)
            },
            "ABSORPTION(FO); LAGTIME([OFF,ON])",
        ),
        &options(Algorithm::ReducedStepwise),
    );
    assert_eq!(script.dirs(), vec!["input", "base", "layer-1"]);
    assert_eq!(result.base_id, "base");
    assert_eq!(result.base_structure, fo(0));
    let input = result.row("input").unwrap();
    assert_eq!((input.layer, input.parent.as_deref()), (0, None));
    let base = result.row("base").unwrap();
    assert_eq!((base.layer, base.parent.as_deref()), (0, Some("input")));
    assert_eq!(base.path, vec![FeatureKey::Absorption(Absorption::Fo)]);
    assert_eq!(base.structure, fo(0));
    // The base's text carries the new absorption parameter.
    let base_text = &script.candidates_in("base")[0].model;
    assert_eq!(
        base_text.block_lines("structural_model"),
        vec!["pk one_cpt_oral(cl=CL, v=V, ka=KA)"]
    );
    assert!(base_text
        .block_lines("parameters")
        .iter()
        .any(|l| l.starts_with("theta TVKA(")));
    assert_eq!(
        paths(&result, 1),
        vec![triple("run1", "base", "LAGTIME(ON)")]
    );
    assert!(
        result.notes.iter().any(|n| n.contains("outside the space")),
        "{:?}",
        result.notes
    );
    assert_eq!(result.final_id, "run1");
    // Δ is against the base, not the input.
    assert_eq!(result.row("run1").unwrap().d_criterion, Some(-10.0));
    assert_eq!(result.row("input").unwrap().d_criterion, Some(20.0));
}

#[test]
fn a_derived_base_no_template_can_express_is_an_error_naming_the_fix() {
    // An IV input in a space listing only `LAGTIME(ON)` and `TRANSITS(3)`:
    // the least-transformation base would be a bolus with a lag and a
    // transit chain, which nothing can build.
    let text = ModelText::parse(BASE_IV).unwrap();
    let features =
        space_features(&Mfl::parse("LAGTIME(ON); TRANSITS(3, NODEPOT)").unwrap()).unwrap();
    let d = defaults(&text);
    let err = Space::build(
        text,
        Structure {
            absorption: Absorption::Inst,
            ..fo(0)
        },
        &features,
        d,
        Vec::new(),
    )
    .unwrap_err();
    assert!(err.contains("TRANSITS(3, NODEPOT), LAGTIME(ON)"), "{err}");
    assert!(err.contains("LAGTIME([OFF,ON])"), "{err}");
}

#[test]
fn a_move_no_template_can_express_is_noted_not_generated() {
    // Three compartments with transit absorption: Pharmpy would build it,
    // ferx has no `three_cpt_transit`.
    let three = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV1(10.0, 0.1, 500.0)
  theta TVQ2(1.0, 0.0, 100.0)
  theta TVV2(5.0, 0.0, 100.0)
  theta TVQ3(0.5, 0.0, 100.0)
  theta TVV3(20.0, 0.0, 100.0)
  theta TVKA(1.5, 0.01, 50.0)
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL
  V1 = TVV1
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA

[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";
    let script = Script::new(&[]);
    let result = run(
        &script,
        space_of(
            three,
            fo(2),
            "TRANSITS(0, NODEPOT); TRANSITS(N); LAGTIME([OFF,ON])",
        ),
        &options(Algorithm::ExhaustiveStepwise),
    );
    assert_eq!(
        paths(&result, 1),
        vec![triple("run1", "base", "LAGTIME(ON)")]
    );
    assert_eq!(result.n_layers(), 1);
    let note = result
        .notes
        .iter()
        .find(|n| n.contains("not generated: TRANSITS(N, NODEPOT) after the base"))
        .unwrap_or_else(|| panic!("{:?}", result.notes));
    assert!(note.contains("three_cpt_transit"), "{note}");
    // Exhaustive: the same gap, the same note.
    let script = Script::new(&[]);
    let result = run(
        &script,
        space_of(
            three,
            fo(2),
            "TRANSITS(0, NODEPOT); TRANSITS(N); LAGTIME([OFF,ON])",
        ),
        &options(Algorithm::Exhaustive),
    );
    assert_eq!(result.layer_rows(1).count(), 1);
    assert!(result.notes.iter().any(|n| n.contains("three_cpt_transit")));
}

// ── ranking ─────────────────────────────────────────────────────────────────

#[test]
fn the_cutoff_keeps_the_base_unless_a_candidate_beats_it_by_enough() {
    let script = Script::new(&[(FO_BASE, 100.0), (FO_LAG, 97.0), (FO_P1, 90.0)]);
    let mut opts = options(Algorithm::Exhaustive);
    opts.cutoff = Some(3.84);
    let result = run(
        &script,
        space("PERIPHERALS(0..1); LAGTIME([OFF,ON])"),
        &opts,
    );
    // run1 (PERIPHERALS(1)) drops 10 > 3.84; it wins.
    assert_eq!(result.final_id, "run1");
    let script = Script::new(&[(FO_BASE, 100.0), (FO_LAG, 97.0), (FO_P1, 97.5)]);
    let result = run(
        &script,
        space("PERIPHERALS(0..1); LAGTIME([OFF,ON])"),
        &opts,
    );
    assert_eq!(result.final_id, "base", "no candidate clears the cutoff");
    assert!(result.row("base").unwrap().selected);
    // Ranks are still assigned: the report shows what came closest.
    assert_eq!(result.row("run2").unwrap().rank, Some(1));
    assert_eq!(result.row("run1").unwrap().rank, Some(2));
    assert_eq!(result.row("base").unwrap().rank, Some(3));
    assert_eq!(result.final_criterion, 100.0);
}

#[test]
fn a_candidate_that_fails_the_gate_or_does_not_fit_is_reported_and_unranked() {
    let mut script = Script::new(&[(FO_BASE, 100.0), (FO_LAG, 50.0), (FO_P1, 90.0)]);
    script.failing.push(FO_LAG.into());
    script.erroring.push("run3".into());
    let result = run(
        &script,
        space("PERIPHERALS(0..1); LAGTIME([OFF,ON])"),
        &options(Algorithm::Exhaustive),
    );
    let lag = result.row("run2").unwrap();
    assert_eq!(lag.path, vec![FeatureKey::Lagtime(true)]);
    assert!(!lag.passed && lag.rank.is_none() && !lag.selected);
    assert_eq!(
        lag.failures,
        vec!["stalled at the initial estimates (#751)"]
    );
    assert_eq!(lag.converged, Some(false));
    assert_eq!(lag.ofv, Some(50.0), "the OFV is shown beside the reason");
    let broken = result.row("run3").unwrap();
    assert!(broken.error.is_some() && broken.ofv.is_none() && broken.rank.is_none());
    assert!(broken.criterion.is_nan());
    assert_eq!(result.final_id, "run1");
    assert_eq!(result.rows.len(), 4);
    assert_eq!(result.row("run1").unwrap().seconds, 1.5);
}

#[test]
fn a_base_that_fails_the_gate_leaves_the_candidates_ranked_among_themselves() {
    let mut script = Script::new(&[(FO_BASE, 100.0), (FO_LAG, 95.0)]);
    script.failing.push("base".into());
    let mut opts = options(Algorithm::Exhaustive);
    opts.cutoff = Some(3.84);
    let result = run(&script, space("LAGTIME([OFF,ON])"), &opts);
    assert!(result
        .notes
        .iter()
        .any(|n| n.contains("base model fails the strictness gate")));
    assert_eq!(result.final_id, "run1");
    assert!(result.row("run1").unwrap().d_criterion.is_none());
    // Nothing eligible at all: the base is returned, and the note says so.
    let mut script = Script::new(&[(FO_BASE, 100.0), (FO_LAG, 95.0)]);
    script
        .failing
        .extend(["base".to_string(), FO_LAG.to_string()]);
    let result = run(&script, space("LAGTIME([OFF,ON])"), &opts);
    assert_eq!(result.final_id, "base");
    assert!(result.notes.iter().any(|n| n.contains("no model passed")));
    assert!(result.ranked().is_empty());
}

#[test]
fn the_base_model_that_cannot_be_fitted_is_an_error() {
    let mut script = Script::new(&[]);
    script.erroring.push("base".into());
    let err = search(
        &script,
        space("LAGTIME([OFF,ON])"),
        &options(Algorithm::Exhaustive),
        None,
    )
    .unwrap_err();
    assert!(err.contains("base model could not be fitted"), "{err}");
}

#[test]
fn cancellation_stops_between_layers_and_returns_what_finished() {
    let mut script = Script::new(&[(FO_BASE, 100.0), (FO_LAG, 90.0)]);
    script.cancel_after = Some(2);
    let result = run(
        &script,
        space("PERIPHERALS(0..1); LAGTIME([OFF,ON])"),
        &options(Algorithm::ExhaustiveStepwise),
    );
    assert!(result.cancelled);
    assert_eq!(result.n_layers(), 1);
    assert_eq!(script.dirs(), vec!["base", "layer-1"]);
    // What finished is still ranked and a final model is named.
    assert_eq!(result.final_id, "run1");
    let mut script = Script::new(&[(FO_BASE, 100.0)]);
    script.cancel_after = Some(1);
    let result = run(
        &script,
        space("LAGTIME([OFF,ON])"),
        &options(Algorithm::ExhaustiveStepwise),
    );
    assert!(result.cancelled);
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.final_id, "base");
}

#[test]
fn progress_events_name_the_layers_and_their_best() {
    let script = Script::new(&[(FO_BASE, 100.0), (FO_LAG, 90.0), (FO_P1, 95.0)]);
    let events = Mutex::new(Vec::new());
    let progress = |e: ModelsearchEvent| events.lock().unwrap().push(format!("{e:?}"));
    let _ = search(
        &script,
        space("PERIPHERALS(0..1); LAGTIME([OFF,ON])"),
        &options(Algorithm::ReducedStepwise),
        Some(&progress),
    )
    .unwrap();
    let events = events.lock().unwrap();
    assert_eq!(events[0], "BaseStarted");
    assert!(events[1].starts_with("BaseFinished { ofv: 100.0"));
    assert_eq!(events[2], "LayerStarted { layer: 1, candidates: 2 }");
    assert_eq!(
        events[3],
        "LayerFinished { layer: 1, best: Some((\"run1\", 90.0)) }"
    );
}

// ── options ─────────────────────────────────────────────────────────────────

fn config(extra: &str) -> crate::search::SearchConfig {
    let text = format!("base = \"b.ferx\"\n[space]\nmfl = \"PERIPHERALS(0..1)\"\n{extra}");
    crate::search::SearchConfig::from_str(&text, std::path::Path::new(".")).unwrap()
}

#[test]
fn options_come_from_the_modelsearch_and_rank_sections() {
    let o = ModelsearchOptions::from_config(&config("")).unwrap();
    assert_eq!(o, ModelsearchOptions::default());
    assert_eq!(o.algorithm, Algorithm::ReducedStepwise);
    assert_eq!(o.iiv_strategy, IivStrategy::AbsorptionDelay);
    assert_eq!(o.rank, RankType::Bic);
    assert_eq!(o.criterion(), Criterion::Bic(ferx_core::BicType::Mixed));
    let o = ModelsearchOptions::from_config(&config(
        "[modelsearch]\nalgorithm = \"exhaustive\"\niiv_strategy = \"add_diagonal\"\n\
         [rank]\ntype = \"bic_iiv\"\ncutoff = 3.84\n",
    ))
    .unwrap();
    assert_eq!(o.algorithm, Algorithm::Exhaustive);
    assert_eq!(o.iiv_strategy, IivStrategy::AddDiagonal);
    assert_eq!(o.rank, RankType::BicIiv);
    assert_eq!(o.cutoff, Some(3.84));
    let err =
        ModelsearchOptions::from_config(&config("[modelsearch]\niiv_strategy = \"fullblock\"\n"))
            .unwrap_err();
    assert!(err.contains("fullblock") && err.contains("#1183"), "{err}");
    let err =
        ModelsearchOptions::from_config(&config("[modelsearch]\nalgoritm = \"exhaustive\"\n"))
            .unwrap_err();
    assert!(
        err.contains("[modelsearch]") && err.contains("algoritm"),
        "{err}"
    );
    let err = ModelsearchOptions::from_config(&config("[rank]\ncutoff = -1.0\n")).unwrap_err();
    assert!(err.contains("cutoff"), "{err}");
    let err = ModelsearchOptions::from_config(&config("[modelsearch]\nalgorithm = \"stepwise\"\n"))
        .unwrap_err();
    assert!(err.contains("stepwise"), "{err}");
}

// ── the report ──────────────────────────────────────────────────────────────

#[test]
fn the_report_writes_models_csv_and_the_final_model_with_its_estimates() {
    let mut script = Script::new(&[(FO_BASE, 100.0), (FO_LAG, 90.0), (FO_P1, 95.0)]);
    script.failing.push(FO_P1.into());
    let result = run(
        &script,
        space("PERIPHERALS(0..1); LAGTIME([OFF,ON])"),
        &options(Algorithm::Exhaustive),
    );
    let dir = tempfile::tempdir().unwrap();
    write_report(dir.path(), &result).unwrap();
    let csv = std::fs::read_to_string(models_path(dir.path())).unwrap();
    let mut lines = csv.lines();
    assert_eq!(lines.next().unwrap(), MODEL_COLUMNS.join(","));
    let rows: Vec<&str> = lines.collect();
    assert_eq!(rows.len(), 4);
    assert!(rows[0].starts_with("base,,0,,FO,0,0,OFF,"), "{}", rows[0]);
    assert!(
        rows[1].starts_with("run1,base,1,PERIPHERALS(1),FO,1,0,OFF,"),
        "{}",
        rows[1]
    );
    assert!(
        rows[1].contains("stalled at the initial estimates (#751)"),
        "{}",
        rows[1]
    );
    assert!(rows[1].ends_with(",false,true,false"), "{}", rows[1]);
    assert!(
        rows[2].starts_with("run2,base,1,LAGTIME(ON),FO,0,0,ON,"),
        "{}",
        rows[2]
    );
    assert!(
        rows[2].contains(",90.000000,90.000000,-10.000000,1,true,true,,,1.500000,true,true,false"),
        "{}",
        rows[2]
    );
    assert!(
        rows[3].starts_with("run3,base,1,LAGTIME(ON);PERIPHERALS(1),FO,1,0,ON,"),
        "{}",
        rows[3]
    );

    // Every fitted model is on disk as it was fitted.
    for id in ["base", "run1", "run2", "run3"] {
        let text = std::fs::read_to_string(models_dir(dir.path()).join(format!("{id}.ferx")))
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(text, result.models[id].render());
    }
    assert_eq!(result.models.len(), 4);
    let final_model = std::fs::read_to_string(final_model_path(dir.path())).unwrap();
    assert!(
        final_model.contains("pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=ALAG)"),
        "{final_model}"
    );
    assert!(
        final_model.contains("omega ETA_ALAG ~ 0.01"),
        "{final_model}"
    );

    let summary = render_summary(&result);
    assert!(
        summary.contains("Base model (base): FO, 0 peripherals"),
        "{summary}"
    );
    assert!(summary.contains("SELECTED"), "{summary}");
    assert!(
        summary.contains("stalled at the initial estimates"),
        "{summary}"
    );
    assert!(
        summary.contains("Final model: run2 — FO, 0 peripherals, lag (ofv 90.000)"),
        "{summary}"
    );
    // Best first: run2's line precedes the base's, and the unranked run1 is last.
    assert!(summary.find("  run2 ").unwrap() < summary.find("  base ").unwrap());
    assert!(summary.find("  base ").unwrap() < summary.find("  run1 ").unwrap());
}

#[test]
fn default_dir_sits_next_to_the_config() {
    assert_eq!(
        default_dir(std::path::Path::new("runs/warfarin.ferxsearch")),
        std::path::PathBuf::from("runs/warfarin-modelsearch")
    );
}

// ── the review on #1256 ─────────────────────────────────────────────────────

const BASE_THREE: &str = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV1(10.0, 0.1, 500.0)
  theta TVQ2(1.0, 0.0, 100.0)
  theta TVV2(5.0, 0.0, 100.0)
  theta TVQ3(0.5, 0.0, 100.0)
  theta TVV3(20.0, 0.0, 100.0)
  theta TVKA(1.5, 0.01, 50.0)
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL
  V1 = TVV1
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA

[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// A one-compartment transit base with a `FIX`ed count of 3 — what the
/// search itself writes for `TRANSITS(3)`.
const BASE_TRANSIT_FIXED: &str = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVNTR(3.0, 0.0, 64.0) FIX
  theta TVMTT(1.0, 0.0, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  NTR = TVNTR
  MTT = TVMTT

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// The space from a base whose structure is read off its own text.
fn space_from_text(base: &str, mfl: &str) -> Space {
    let text = ModelText::parse(base).unwrap();
    let template = text
        .block_lines("structural_model")
        .iter()
        .find_map(|l| crate::search::PkTemplate::parse_line(l))
        .unwrap()
        .unwrap();
    let structure = Structure::from_model(&template, Some(&text)).unwrap();
    let features = space_features(&Mfl::parse(mfl).unwrap()).unwrap();
    let d = defaults(&text);
    Space::build(text, structure, &features, d, Vec::new()).expect("a buildable base")
}

#[test]
fn a_narrowed_base_can_be_widened_again_because_declarations_come_from_the_parent() {
    // A three-compartment input in a `PERIPHERALS(0..1)` space: the base
    // is the input narrowed to one compartment, which prunes Q2/V2/Q3/V3.
    // The one-peripheral child must then re-declare Q and V2 — which it
    // cannot if "declared" is read off the input rather than its parent.
    let script = Script::new(&[]);
    let result = run(
        &script,
        space_from_text(BASE_THREE, "ABSORPTION(FO); PERIPHERALS(0..1)"),
        &options(Algorithm::ExhaustiveStepwise),
    );
    assert_eq!(result.base_id, "base");
    assert_eq!(result.base_structure, fo(0));
    let base_text = &result.models["base"];
    assert_eq!(
        base_text.block_lines("structural_model"),
        vec!["pk one_cpt_oral(cl=CL, v=V1, ka=KA)"]
    );
    assert!(!base_text
        .block_lines("parameters")
        .iter()
        .any(|l| l.contains("TVV2") || l.contains("TVQ2")));
    assert_eq!(
        paths(&result, 1),
        vec![triple("run1", "base", "PERIPHERALS(1)")]
    );
    let child = &result.models["run1"];
    assert_eq!(
        child.block_lines("structural_model"),
        vec!["pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)"]
    );
    let params = child.block_lines("parameters");
    assert!(
        params.iter().any(|l| l.starts_with("theta TVQ(")),
        "{params:?}"
    );
    assert!(
        params.iter().any(|l| l.starts_with("theta TVV2(")),
        "{params:?}"
    );
}

#[test]
fn a_fixed_transit_base_moves_to_zero_and_to_another_count_by_rebinding_n() {
    let script = Script::new(&[]);
    let result = run(
        &script,
        space_from_text(
            BASE_TRANSIT_FIXED,
            "ABSORPTION(FO); TRANSITS([0,1,3], NODEPOT)",
        ),
        &options(Algorithm::ExhaustiveStepwise),
    );
    // The base's own TRANSITS(3) is not a move; 0 and 1 are.
    assert_eq!(result.base_structure.transits, Some(TransitCount::Count(3)));
    assert_eq!(
        paths(&result, 1),
        vec![
            triple("run1", "base", "TRANSITS(0, NODEPOT)"),
            triple("run2", "base", "TRANSITS(1, NODEPOT)"),
        ]
    );
    // TRANSITS(0): first-order absorption, a new KA, the chain gone.
    let zero = &result.models["run1"];
    assert_eq!(
        zero.block_lines("structural_model"),
        vec!["pk one_cpt_oral(cl=CL, v=V, ka=KA)"]
    );
    let params = zero.block_lines("parameters");
    assert!(
        params.iter().any(|l| l.starts_with("theta TVKA(")),
        "{params:?}"
    );
    assert!(
        !params
            .iter()
            .any(|l| l.contains("TVNTR") || l.contains("TVMTT")),
        "{params:?}"
    );
    // TRANSITS(1): a fresh FIXed count; the old one pruned, MTT kept.
    let one = &result.models["run2"];
    assert_eq!(
        one.block_lines("structural_model"),
        vec!["pk one_cpt_transit(cl=CL, v=V, n=NTR2, mtt=MTT)"]
    );
    let params = one.block_lines("parameters");
    assert!(
        params.contains(&"theta TVNTR2(1.0, 0.0, 64.0) FIX".to_string()),
        "{params:?}"
    );
    assert!(
        !params.iter().any(|l| l.starts_with("theta TVNTR(")),
        "{params:?}"
    );
    assert!(
        params.iter().any(|l| l.starts_with("theta TVMTT(")),
        "{params:?}"
    );
    // Three different models, not one canonical text three times.
    let hashes: std::collections::HashSet<[u8; 32]> = ["base", "run1", "run2"]
        .iter()
        .map(|id| result.models[*id].canonical_hash())
        .collect();
    assert_eq!(hashes.len(), 3);
    assert_eq!(result.row("run1").unwrap().structure.transits, None);
    assert_eq!(
        result.row("run2").unwrap().structure.transits,
        Some(TransitCount::Count(1))
    );
}

#[test]
fn an_estimated_transit_base_moves_to_a_fixed_count_and_back() {
    let free = BASE_TRANSIT_FIXED.replace(
        "theta TVNTR(3.0, 0.0, 64.0) FIX",
        "theta TVNTR(3.0, 0.0, 64.0)",
    );
    let script = Script::new(&[]);
    let result = run(
        &script,
        space_from_text(
            &free,
            "ABSORPTION(FO); TRANSITS(N); TRANSITS([0,3], NODEPOT)",
        ),
        &options(Algorithm::ExhaustiveStepwise),
    );
    assert_eq!(result.base_structure.transits, Some(TransitCount::N));
    assert_eq!(
        paths(&result, 1),
        vec![
            triple("run1", "base", "TRANSITS(0, NODEPOT)"),
            triple("run2", "base", "TRANSITS(3, NODEPOT)"),
        ]
    );
    let three = &result.models["run2"];
    let params = three.block_lines("parameters");
    assert!(
        params.contains(&"theta TVNTR2(3.0, 0.0, 64.0) FIX".to_string()),
        "{params:?}"
    );
    assert!(
        !params.iter().any(|l| l.starts_with("theta TVNTR(")),
        "{params:?}"
    );
    // The reverse move, from the fixed base, gives a free count at Pharmpy's init.
    let script = Script::new(&[]);
    let result = run(
        &script,
        space_from_text(
            BASE_TRANSIT_FIXED,
            "ABSORPTION(FO); TRANSITS(N); TRANSITS(3, NODEPOT)",
        ),
        &options(Algorithm::ExhaustiveStepwise),
    );
    assert_eq!(
        paths(&result, 1),
        vec![triple("run1", "base", "TRANSITS(N, NODEPOT)")]
    );
    let params = result.models["run1"].block_lines("parameters");
    assert!(
        params.contains(&"theta TVNTR2(2.0, 0.0, 64.0)".to_string()),
        "{params:?}"
    );
}

#[test]
fn a_duplicates_children_are_seeded_from_its_representatives_fit() {
    // run2 is reported as a canonical duplicate of run1 (no fit of its
    // own); run1's fit carries a TVCL the file does not have, and run2's
    // child must start from it.
    let mut script = Script::new(&[(FO_BASE, 100.0)]);
    script.duplicates.insert("run2".into(), "run1".into());
    script.theta_override.insert("run1".into(), 0.777);
    let result = run(
        &script,
        space("PERIPHERALS(0..1); LAGTIME([OFF,ON])"),
        &options(Algorithm::ExhaustiveStepwise),
    );
    let child = result
        .rows
        .iter()
        .find(|r| r.parent.as_deref() == Some("run2"))
        .expect("run2 was extended");
    let params = result.models[&child.id].block_lines("parameters");
    assert!(
        params.iter().any(|l| l.starts_with("theta TVCL(0.777,")),
        "{params:?}"
    );
    // …and a duplicate that wins carries its representative's fit out.
    let mut script = Script::new(&[(FO_BASE, 100.0)]).by_id(&[("run2", 10.0)]);
    script.duplicates.insert("run2".into(), "run1".into());
    let result = run(
        &script,
        space("PERIPHERALS(0..1); LAGTIME([OFF,ON])"),
        &options(Algorithm::Exhaustive),
    );
    assert_eq!(result.final_id, "run2");
    assert!(result.final_fit.is_some());
    // A duplicate whose representative has no fit either is an error — the
    // representative's own, raised first, naming it and the fix.
    let mut script = Script::new(&[(FO_BASE, 100.0)]);
    script.duplicates.insert("run2".into(), "run1".into());
    script.fitless.push("run1".into());
    let err = search(
        &script,
        space("PERIPHERALS(0..1); LAGTIME([OFF,ON])"),
        &options(Algorithm::ExhaustiveStepwise),
        None,
    )
    .unwrap_err();
    assert!(err.starts_with("run1:") && err.contains("resume"), "{err}");
}

#[test]
fn a_resumed_row_without_its_cached_fit_is_an_error_naming_the_fix() {
    let mut script = Script::new(&[(FO_BASE, 100.0)]);
    script.fitless.push("run1".into());
    let err = search(
        &script,
        space("PERIPHERALS(0..1); LAGTIME([OFF,ON])"),
        &options(Algorithm::ExhaustiveStepwise),
        None,
    )
    .unwrap_err();
    assert!(err.contains("run1"), "{err}");
    assert!(
        err.contains("journal cache") && err.contains("resume"),
        "{err}"
    );
    // The base too.
    let mut script = Script::new(&[(FO_BASE, 100.0)]);
    script.fitless.push("base".into());
    let err = search(
        &script,
        space("LAGTIME([OFF,ON])"),
        &options(Algorithm::Exhaustive),
        None,
    )
    .unwrap_err();
    assert!(err.contains("base") && err.contains("resume"), "{err}");
}

#[test]
fn a_failed_candidates_children_start_unseeded_and_the_notes_say_so() {
    let mut script = Script::new(&[(FO_BASE, 100.0)]);
    script.erroring.push("run1".into());
    let result = run(
        &script,
        space("PERIPHERALS(0..1); LAGTIME([OFF,ON])"),
        &options(Algorithm::ExhaustiveStepwise),
    );
    assert!(result
        .rows
        .iter()
        .any(|r| r.parent.as_deref() == Some("run1")));
    assert!(
        result
            .notes
            .iter()
            .any(|n| n.starts_with("run1 produced no fit")),
        "{:?}",
        result.notes
    );
}
