//! The Pharmpy anchor for the candidate enumeration (#1181).
//!
//! `tests/data/modelsearch_pharmpy_anchor.json` is Pharmpy 2.0.0's own
//! `modelsearch` workflow — the base it derives, the candidate moves it
//! filters, and every `create_candidate` task with its feature set and
//! parents — for nine (base, space, algorithm) cases, built without fitting
//! anything. This module runs the same cases through [`search`] with a
//! fitter that returns one OFV for everything (so a reduced-stepwise
//! collapse keeps the first member, as Pharmpy's `min` does on a tie) and
//! compares the enumerations. Regenerate the JSON with
//! `tools/pharmpy-modelsearch-anchor/run.sh`.
//!
//! Where ferx deliberately differs, the case is named in [`DIVERGENCES`]
//! with the reason, and the test asserts the two sides really *do* differ
//! — so a divergence that quietly disappears (or a new one) is a red test
//! either way. Transit cases are absent on purpose; the README in the anchor
//! directory records the two Pharmpy behaviours that make them unusable as
//! an oracle.

use std::collections::BTreeMap;

use ferx_core::edit::ModelText;
use ferx_core::StrictnessVerdict;
use serde::Deserialize;

use super::structure::space_features;
use super::tests::{defaults, BASE, BASE_IV};
use super::*;
use crate::search::mfl::Mfl;
use crate::search::test_support::converged_fit;
use crate::search::{CandidateResult, RunReport};

const ANCHOR: &str = include_str!("../../tests/data/modelsearch_pharmpy_anchor.json");

#[derive(Deserialize)]
struct Anchor {
    pharmpy_version: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    model: String,
    space: String,
    algorithm: String,
    base_transformations: String,
    funcs: Vec<String>,
    candidates: Vec<PharmpyCandidate>,
}

#[derive(Deserialize, Clone)]
struct PharmpyCandidate {
    name: String,
    feature: Vec<String>,
    feature_set: Vec<String>,
    parents: Vec<String>,
    via_collector: bool,
}

/// How ferx differs from Pharmpy on a case, by (space, algorithm).
#[derive(Clone, Copy, PartialEq)]
enum Divergence {
    /// Pharmpy fills a category the space does not name with its default
    /// (`ABSORPTION(INST)`), and moves the base there; ferx leaves the base's
    /// own value alone. The enumeration is otherwise identical.
    DefaultFilledBase,
    /// Pharmpy's pair table is applied to the features a *path* applied, so
    /// a lag time on a bolus **base** is a candidate; ferx applies it to the
    /// whole structure and never builds a lagged bolus.
    LagOnBolusBase,
    /// Pharmpy's `reduced_stepwise` collapses only when a layer holds two or
    /// more duplicate groups (`len(groups) > 1`); ferx collapses every group.
    SingleGroupNotCollapsed,
    /// Pharmpy's `exhaustive` does not apply the pair table; ferx does.
    ExhaustivePairs,
}

const DIVERGENCES: &[(&str, &str, Divergence)] = &[
    (
        "PERIPHERALS(0..2); LAGTIME([OFF,ON])",
        "exhaustive_stepwise",
        Divergence::DefaultFilledBase,
    ),
    (
        "ABSORPTION([INST,FO]); LAGTIME([OFF,ON])",
        "exhaustive_stepwise",
        Divergence::LagOnBolusBase,
    ),
    (
        "ABSORPTION(FO); PERIPHERALS(0..2); LAGTIME([OFF,ON])",
        "reduced_stepwise",
        Divergence::SingleGroupNotCollapsed,
    ),
    (
        "ABSORPTION([INST,FO]); PERIPHERALS(0..1); LAGTIME([OFF,ON])",
        "exhaustive",
        Divergence::ExhaustivePairs,
    ),
];

/// One OFV for every candidate: the enumeration is what is under test.
struct Uniform;

impl StepFitter for Uniform {
    fn fit_step(&self, _dir: &str, candidates: &[Candidate]) -> Result<RunReport, String> {
        let results = candidates
            .iter()
            .map(|c| CandidateResult {
                id: c.id.clone(),
                hash: c.hash(),
                parent: c.parent.clone(),
                features: c.features.clone(),
                fit: Some(converged_fit(100.0)),
                ofv: Some(100.0),
                converged: Some(true),
                verdict: StrictnessVerdict {
                    passed: true,
                    failures: vec![],
                    skipped: vec![],
                },
                criterion: 100.0,
                seconds: 0.0,
                error: None,
                duplicate_of: None,
                reused: false,
            })
            .collect();
        Ok(RunReport {
            results,
            cancelled: false,
            fitted: candidates.len(),
            reused: 0,
            deduped: 0,
            warnings: vec![],
        })
    }
}

/// A candidate as both sides describe it: its feature set and its parent's.
type Edge = (Vec<String>, Vec<String>);

fn sorted(v: &[String]) -> Vec<String> {
    let mut v = v.to_vec();
    v.sort();
    v
}

/// Pharmpy's candidates as `(feature set, parent feature set)` edges. Every
/// parent of a collector-joined child carries the same set, so the first
/// one stands for the group.
fn pharmpy_edges(case: &Case) -> Vec<Edge> {
    let by_name: BTreeMap<&str, &PharmpyCandidate> = case
        .candidates
        .iter()
        .map(|c| (c.name.as_str(), c))
        .collect();
    let mut edges: Vec<Edge> = case
        .candidates
        .iter()
        .map(|c| {
            let parent = c
                .parents
                .first()
                .map(|p| sorted(&by_name[p.as_str()].feature_set))
                .unwrap_or_default();
            (sorted(&c.feature_set), parent)
        })
        .collect();
    edges.sort();
    edges
}

fn ferx_edges(result: &ModelsearchResult) -> Vec<Edge> {
    // The root's own moves (a derived base's `onto` path) are not part of
    // the enumeration Pharmpy records, so a layer-0 parent is the empty set.
    let path_of = |id: &str| -> Vec<String> {
        result
            .row(id)
            .filter(|r| r.layer >= 1)
            .map(|r| sorted(&r.path.iter().map(|k| k.to_string()).collect::<Vec<_>>()))
            .unwrap_or_default()
    };
    let mut edges: Vec<Edge> = result
        .rows
        .iter()
        .filter(|r| r.layer >= 1)
        .map(|r| {
            (
                path_of(&r.id),
                r.parent.as_deref().map(path_of).unwrap_or_default(),
            )
        })
        .collect();
    edges.sort();
    edges
}

fn run_case(case: &Case) -> (Vec<String>, String, ModelsearchResult) {
    let (text, structure) = match case.model.as_str() {
        "oral" => (
            BASE,
            Structure {
                absorption: Absorption::Fo,
                peripherals: 0,
                transits: None,
                lagtime: false,
            },
        ),
        "iv" => (
            BASE_IV,
            Structure {
                absorption: Absorption::Inst,
                peripherals: 0,
                transits: None,
                lagtime: false,
            },
        ),
        other => panic!("unknown base model kind {other}"),
    };
    let text = ModelText::parse(text).unwrap();
    let features = space_features(&Mfl::parse(&case.space).unwrap()).unwrap();
    let d = defaults(&text);
    let space = Space::build(text, structure, &features, d, Vec::new()).unwrap();
    let funcs: Vec<String> = space.funcs.iter().map(|k| k.to_string()).collect();
    let onto: Vec<String> = space.onto.iter().map(|k| k.to_string()).collect();
    let algorithm = match case.algorithm.as_str() {
        "exhaustive" => Algorithm::Exhaustive,
        "exhaustive_stepwise" => Algorithm::ExhaustiveStepwise,
        "reduced_stepwise" => Algorithm::ReducedStepwise,
        other => panic!("unknown algorithm {other}"),
    };
    let options = ModelsearchOptions {
        algorithm,
        rank: RankType::Ofv,
        ..ModelsearchOptions::default()
    };
    let result = search(&Uniform, space, &options, None).expect("search");
    (funcs, onto.join(";"), result)
}

fn divergence(case: &Case) -> Option<Divergence> {
    DIVERGENCES
        .iter()
        .find(|(space, alg, _)| *space == case.space && *alg == case.algorithm)
        .map(|(_, _, d)| *d)
}

#[test]
fn the_anchor_is_pharmpy_2() {
    let anchor: Anchor = serde_json::from_str(ANCHOR).unwrap();
    assert_eq!(anchor.pharmpy_version, "2.0.0");
    assert_eq!(anchor.cases.len(), 9);
    // Every divergence names a case that exists.
    for (space, alg, _) in DIVERGENCES {
        assert!(
            anchor
                .cases
                .iter()
                .any(|c| c.space == *space && c.algorithm == *alg),
            "divergence for a case not in the anchor: {space} / {alg}"
        );
    }
}

#[test]
fn the_candidate_moves_are_pharmpys_in_pharmpys_order() {
    let anchor: Anchor = serde_json::from_str(ANCHOR).unwrap();
    for case in &anchor.cases {
        let (funcs, onto, _) = run_case(case);
        let label = format!("{} / {}", case.space, case.algorithm);
        assert_eq!(funcs, case.funcs, "{label}");
        match divergence(case) {
            Some(Divergence::DefaultFilledBase) => {
                assert_eq!(case.base_transformations, "ABSORPTION(INST)", "{label}");
                assert_eq!(onto, "", "{label}: ferx keeps the base's absorption");
            }
            _ => assert_eq!(onto, case.base_transformations, "{label}"),
        }
    }
}

#[test]
fn the_enumeration_matches_pharmpy_or_differs_where_stated() {
    let anchor: Anchor = serde_json::from_str(ANCHOR).unwrap();
    for case in &anchor.cases {
        let label = format!("{} / {}", case.space, case.algorithm);
        let (_, _, result) = run_case(case);
        let pharmpy = pharmpy_edges(case);
        let ferx = ferx_edges(&result);
        match divergence(case) {
            None | Some(Divergence::DefaultFilledBase) => {
                assert_eq!(ferx, pharmpy, "{label}");
                // Collector-joined children exist on both sides in the same
                // places: a ferx row whose parent was not `continued`
                // cannot exist, and Pharmpy's `via_collector` children are
                // the ones whose group ferx collapsed.
                let collector_children = case.candidates.iter().filter(|c| c.via_collector).count();
                let not_continued = result.rows.iter().filter(|r| !r.continued).count();
                if case.algorithm == "reduced_stepwise" {
                    assert!(
                        collector_children > 0,
                        "{label}: no collector in the anchor"
                    );
                    assert_eq!(not_continued, collector_children, "{label}");
                } else {
                    assert_eq!(collector_children, 0, "{label}");
                    assert_eq!(not_continued, 0, "{label}");
                }
            }
            Some(Divergence::LagOnBolusBase) => {
                // The lagged bolus itself, and its child (a first-order
                // model reached *through* it), which ferx never builds
                // because the parent does not exist.
                let lagged_bolus = |set: &Vec<String>| {
                    set.contains(&"LAGTIME(ON)".to_string())
                        && !set.contains(&"ABSORPTION(FO)".to_string())
                };
                let expected: Vec<Edge> = pharmpy
                    .iter()
                    .filter(|(set, parent)| !lagged_bolus(set) && !lagged_bolus(parent))
                    .cloned()
                    .collect();
                assert_ne!(expected, pharmpy, "{label}: the divergence has disappeared");
                assert_eq!(ferx, expected, "{label}");
                assert!(result
                    .notes
                    .iter()
                    .any(|n| n.starts_with("not generated: LAGTIME(ON)")));
            }
            Some(Divergence::SingleGroupNotCollapsed) => {
                assert!(case.candidates.iter().all(|c| !c.via_collector), "{label}");
                assert_eq!(pharmpy.len(), 8, "{label}");
                assert_eq!(ferx.len(), 7, "{label}");
                // Layers 1 and 2 agree; ferx's layer 3 is a subset.
                for edge in &ferx {
                    assert!(pharmpy.contains(edge), "{label}: {edge:?}");
                }
                assert_eq!(result.rows.iter().filter(|r| !r.continued).count(), 1);
            }
            Some(Divergence::ExhaustivePairs) => {
                let expected: Vec<Edge> = pharmpy
                    .iter()
                    .filter(|(set, _)| {
                        !(set.contains(&"ABSORPTION(INST)".to_string())
                            && set.contains(&"LAGTIME(ON)".to_string()))
                    })
                    .cloned()
                    .collect();
                assert_ne!(expected, pharmpy, "{label}: the divergence has disappeared");
                assert_eq!(ferx, expected, "{label}");
            }
        }
    }
}

#[test]
fn exhaustive_numbers_its_candidates_as_pharmpy_does() {
    // The ordered combos, not just the set: `run{n}` follows Pharmpy's
    // `modelsearch_run{n}` where the enumerations agree.
    let anchor: Anchor = serde_json::from_str(ANCHOR).unwrap();
    let case = anchor
        .cases
        .iter()
        .find(|c| c.algorithm == "exhaustive" && divergence(c).is_none())
        .unwrap();
    let (_, _, result) = run_case(case);
    let ferx: Vec<Vec<String>> = result
        .layer_rows(1)
        .map(|r| r.path.iter().map(|k| k.to_string()).collect())
        .collect();
    let pharmpy: Vec<Vec<String>> = case.candidates.iter().map(|c| c.feature.clone()).collect();
    assert_eq!(ferx, pharmpy);
    for (row, c) in result.layer_rows(1).zip(&case.candidates) {
        assert_eq!(format!("modelsearch_{}", row.id), c.name);
    }
}
