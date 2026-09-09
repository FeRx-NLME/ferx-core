//! Tier-1 tests for the global search (#1185).
//!
//! Every test drives [`search`] with a *scripted* fitter — OFVs keyed by
//! the candidate's grid point — so what is under test is the grid, the
//! decoding of a genome into a model, the fitness the search charges on top
//! of the criterion, and what wins. The genetic algorithm's own behaviour
//! on a known landscape is `ga_tests.rs`; the path from `ModelText` through
//! the runner and `fit()` is `tests/globalsearch_end_to_end.rs`.

use ferx_core::edit::ModelText;

use super::*;
use crate::modelsearch::{Absorption, Elimination};
use crate::search::mfl::{CovariateEffect, CovariateOp, Mfl};
use crate::search::test_support::ScriptedFitter;
use crate::search::CovariateEffectSpec;

const BASE: &str = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)

[covariates]
  WT: continuous

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

fn fo() -> Structure {
    Structure {
        absorption: Absorption::Fo,
        elimination: Elimination::Fo,
        peripherals: 0,
        transits: None,
        lagtime: false,
    }
}

fn defaults(text: &ModelText) -> Defaults {
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
    let params = parameter_names_of(text);
    Defaults::new(
        params,
        names,
        inits,
        etas,
        &crate::search::test_support::population(&["1"]),
    )
}

fn effect(
    parameter: &str,
    covariate: &str,
    effect: CovariateEffect,
    optional: bool,
) -> CovariateEffectSpec {
    CovariateEffectSpec {
        parameter: parameter.into(),
        covariate: covariate.into(),
        effect,
        op: CovariateOp::Multiply,
        optional,
    }
}

/// A space over `BASE`: the structural MFL, plus covariate effects.
fn space_with(mfl: &str, effects: &[CovariateEffectSpec], existing: &[(&str, &str)]) -> Space {
    let text = ModelText::parse(BASE).unwrap();
    let structural = if mfl.is_empty() {
        None
    } else {
        let template = text
            .block_lines("structural_model")
            .iter()
            .find_map(|l| PkTemplate::parse_line(l))
            .unwrap()
            .unwrap();
        Some((fo(), defaults(&text), template))
    };
    let keys = if mfl.is_empty() {
        Vec::new()
    } else {
        structure::space_features(&Mfl::parse(mfl).unwrap()).unwrap()
    };
    let existing: Vec<(String, String)> = existing
        .iter()
        .map(|(p, c)| ((*p).to_string(), (*c).to_string()))
        .collect();
    Space::build(text, structural, &keys, effects, &existing, Vec::new()).expect("a space")
}

fn space(mfl: &str) -> Space {
    space_with(mfl, &[], &[])
}

/// The scripted fitter's key: the candidate's grid point, in the compact
/// spelling the tests script — `P1;LAG;CL-WT=power`, or `input`.
fn key(c: &Candidate) -> String {
    if c.id == "input" {
        return "input".into();
    }
    let f = &c.features;
    let mut parts = Vec::new();
    if let Some(a) = f.get("ABSORPTION") {
        if a != "FO" {
            parts.push(a.to_string());
        }
    }
    if let Some(p) = f.get("PERIPHERALS") {
        parts.push(format!("P{p}"));
    }
    if f.get("LAGTIME") == Some("ON") {
        parts.push("LAG".into());
    }
    for (k, v) in f.iter() {
        // A dead gene renders to the same model as `none`, and the runner
        // would fit it once; the script answers it as that model.
        if k.contains('-') && v != "none" && !v.ends_with("(non-influential)") {
            parts.push(format!("{k}={v}"));
        }
    }
    parts.join(";")
}

fn options(algorithm: Algorithm) -> GlobalsearchOptions {
    GlobalsearchOptions {
        algorithm,
        rank: RankType::Ofv,
        ..GlobalsearchOptions::default()
    }
}

fn run(script: &ScriptedFitter, space: Space, options: &GlobalsearchOptions) -> GlobalsearchResult {
    search(script, space, options, None).expect("search")
}

fn ids_by_description(result: &GlobalsearchResult) -> Vec<(String, String)> {
    result
        .rows
        .iter()
        .map(|r| (r.id.clone(), r.description.clone()))
        .collect()
}

// ── the grid ────────────────────────────────────────────────────────────────

#[test]
fn the_grid_is_one_axis_per_category_and_per_covariate_pair() {
    let space = space_with(
        "PERIPHERALS(0..2); LAGTIME([OFF,ON])",
        &[
            effect("CL", "WT", CovariateEffect::Pow, true),
            effect("CL", "WT", CovariateEffect::Exp, true),
            effect("V", "WT", CovariateEffect::Pow, true),
        ],
        &[],
    );
    let axes: Vec<(String, Vec<String>)> =
        space.axes.iter().map(|a| (a.name(), a.labels())).collect();
    assert_eq!(
        axes,
        vec![
            (
                "LAGTIME".to_string(),
                vec!["OFF".to_string(), "ON".to_string()]
            ),
            (
                "PERIPHERALS".to_string(),
                vec!["0".to_string(), "1".to_string(), "2".to_string()]
            ),
            (
                "CL-WT".to_string(),
                vec![
                    "none".to_string(),
                    "power".to_string(),
                    "exponential".to_string()
                ]
            ),
            (
                "V-WT".to_string(),
                vec!["none".to_string(), "power".to_string()]
            ),
        ]
    );
    assert_eq!(space.alleles(), vec![2, 3, 3, 2]);
    assert_eq!(space.size(), Some(36));
    assert_eq!(
        space.describe(&vec![1, 2, 1, 0]),
        "LAGTIME=ON;PERIPHERALS=2;CL-WT=power;V-WT=none"
    );
}

#[test]
fn a_pair_the_base_declares_is_not_an_axis_and_a_forced_effect_is_not_either() {
    let space = space_with(
        "LAGTIME([OFF,ON])",
        &[
            effect("CL", "WT", CovariateEffect::Pow, true),
            effect("V", "WT", CovariateEffect::Pow, false),
        ],
        &[("CL", "WT")],
    );
    assert_eq!(space.axes.len(), 1);
    assert_eq!(space.forced.len(), 1);
    assert!(
        space
            .notes
            .iter()
            .any(|n| n.contains("not explored: CL-WT-power")),
        "{:?}",
        space.notes
    );

    // Nothing left to search is an error, not an empty run.
    let text = ModelText::parse(BASE).unwrap();
    let e = Space::build(
        text,
        None,
        &[],
        &[effect("CL", "WT", CovariateEffect::Pow, true)],
        &[("CL".into(), "WT".into())],
        Vec::new(),
    )
    .unwrap_err();
    assert!(e.contains("no axis to search"), "{e}");
}

#[test]
fn the_partition_refuses_a_statement_the_grid_cannot_lay_out() {
    let e = partition(&Mfl::parse("PERIPHERALS(0..1); IIV(CL, exp)").unwrap()).unwrap_err();
    assert!(e.contains("`IIV`"), "{e}");
    let e = partition(&Mfl::parse("ALLOMETRY(WT, 70)").unwrap()).unwrap_err();
    assert!(e.contains("`ALLOMETRY`"), "{e}");
    let (structural, has_cov) =
        partition(&Mfl::parse("LET(X, [CL]); PERIPHERALS(1); COVARIATE?(@X, WT, pow)").unwrap())
            .unwrap();
    assert!(has_cov);
    // The LET travels with the structural half, the COVARIATE does not.
    assert_eq!(structural.statements.len(), 2);
}

// ── exhaustive ──────────────────────────────────────────────────────────────

#[test]
fn exhaustive_evaluates_every_grid_point_and_selects_the_best() {
    let space = space_with(
        "PERIPHERALS(0..1); LAGTIME([OFF,ON])",
        &[effect("CL", "WT", CovariateEffect::Pow, true)],
        &[],
    );
    let script = ScriptedFitter::new(
        key,
        &[
            ("input", 500.0),
            ("P0", 500.0),
            ("P1", 480.0),
            ("P0;LAG", 470.0),
            ("P1;LAG", 440.0),
            ("P0;CL-WT=power", 490.0),
            ("P1;CL-WT=power", 460.0),
            ("P0;LAG;CL-WT=power", 455.0),
            ("P1;LAG;CL-WT=power", 430.0),
        ],
    );
    let result = run(&script, space, &options(Algorithm::Exhaustive));
    assert_eq!(result.space_size, 8);
    assert_eq!(result.rows.len(), 9, "the input and eight grid points");
    assert_eq!(script.dirs(), vec!["input", "candidates"]);
    assert_eq!(script.candidates_in("candidates").len(), 8);
    // Mixed-radix order, last axis fastest: LAGTIME, PERIPHERALS, CL-WT.
    assert_eq!(
        ids_by_description(&result)[..3],
        [
            ("input".to_string(), "the input model".to_string()),
            (
                "run1".to_string(),
                "LAGTIME=OFF;PERIPHERALS=0;CL-WT=none".to_string()
            ),
            (
                "run2".to_string(),
                "LAGTIME=OFF;PERIPHERALS=0;CL-WT=power".to_string()
            ),
        ]
    );
    let winner = result.row(&result.final_id).unwrap();
    assert_eq!(winner.description, "LAGTIME=ON;PERIPHERALS=1;CL-WT=power");
    assert_eq!(winner.rank, Some(1));
    assert!(winner.selected);
    assert_eq!(winner.fitness, 430.0, "OFV criterion, no charges");
    assert_eq!(winner.criterion, 430.0);
    assert_eq!(result.final_fitness, 430.0);
    // The winner's model carries the relation and the second compartment.
    let text = result.final_model.render();
    assert!(text.contains("CL ~ WT power"), "{text}");
    assert!(text.contains("two_cpt_oral"), "{text}");
    assert!(text.contains("lagtime="), "{text}");
    // Every row is ranked (all eligible), best first, ties by creation order.
    let ranked: Vec<&str> = result.ranked().iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ranked.len(), 9);
    assert_eq!(ranked[0], winner.id);
    // The input is ranked but never selected.
    let input = result.row("input").unwrap();
    assert!(input.rank.is_some());
    assert!(!input.selected);
    assert!(result.generations.is_empty());
    assert!(!result.cancelled);
    // Every model's text is kept.
    assert_eq!(result.models.len(), 9);
}

#[test]
fn exhaustive_refuses_a_grid_above_max_models() {
    let space = space("PERIPHERALS(0..2); LAGTIME([OFF,ON])");
    let script = ScriptedFitter::new(key, &[]);
    let options = GlobalsearchOptions {
        max_models: 5,
        ..options(Algorithm::Exhaustive)
    };
    let e = search(&script, space, &options, None).unwrap_err();
    assert!(
        e.contains("6 points, above [globalsearch] max_models = 5"),
        "{e}"
    );
    assert!(
        script.dirs().is_empty(),
        "refused before the input was fitted"
    );
}

// ── the GA on the model layer ───────────────────────────────────────────────

/// The scripted OFV of a grid point, from its compact key: each feature
/// contributes independently, and one pair interacts — the model the GA has
/// to find is `P1;LAG;CL-WT=exponential;V-WT=power`.
fn landscape(k: &str) -> f64 {
    if k == "input" {
        return 500.0;
    }
    let mut ofv = 500.0;
    if k.contains("P1") {
        ofv -= 20.0;
    }
    if k.contains("P2") {
        ofv -= 12.0;
    }
    if k.contains("LAG") {
        ofv -= 15.0;
    }
    if k.contains("CL-WT=power") {
        ofv -= 8.0;
    }
    if k.contains("CL-WT=exponential") {
        ofv -= 6.0;
    }
    if k.contains("V-WT=power") {
        ofv -= 5.0;
    }
    // The interaction: exponential on CL pays off only with V-WT present.
    if k.contains("CL-WT=exponential") && k.contains("V-WT=power") {
        ofv -= 10.0;
    }
    ofv
}

fn landscape_fitter(space: &Space) -> ScriptedFitter {
    // Script every grid point by enumerating the space through the same
    // decoder the search uses — the key is what `key()` renders.
    let mut table: Vec<(String, f64)> = vec![("input".into(), 500.0)];
    for g in ga::enumerate(&space.alleles()) {
        let features: FeatureVector = space
            .describe(&g)
            .split(';')
            .map(|part| part.split_once('=').unwrap())
            .collect();
        let k = key(&Candidate::new("x", space.input_model.clone()).features(features));
        table.push((k.clone(), landscape(&k)));
    }
    let refs: Vec<(&str, f64)> = table.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    ScriptedFitter::new(key, &refs)
}

fn ga_space() -> Space {
    space_with(
        "PERIPHERALS(0..2); LAGTIME([OFF,ON])",
        &[
            effect("CL", "WT", CovariateEffect::Pow, true),
            effect("CL", "WT", CovariateEffect::Exp, true),
            effect("V", "WT", CovariateEffect::Pow, true),
        ],
        &[],
    )
}

#[test]
fn the_ga_finds_the_same_optimum_as_exhaustive_enumeration() {
    let exhaustive = {
        let space = ga_space();
        let script = landscape_fitter(&space);
        run(&script, space, &options(Algorithm::Exhaustive))
    };
    assert_eq!(exhaustive.space_size, 36);
    let best = exhaustive.row(&exhaustive.final_id).unwrap();
    assert_eq!(
        best.description,
        "LAGTIME=ON;PERIPHERALS=1;CL-WT=exponential;V-WT=power"
    );
    assert_eq!(best.fitness, 500.0 - 20.0 - 15.0 - 6.0 - 5.0 - 10.0);

    for seed in [1u64, 2, 3, 1185] {
        let space = ga_space();
        let script = landscape_fitter(&space);
        let options = GlobalsearchOptions {
            ga: GaOptions {
                population_size: 8,
                generations: 6,
                seed,
                ..GaOptions::default()
            },
            ..options(Algorithm::Ga)
        };
        let result = run(&script, space, &options);
        let winner = result.row(&result.final_id).unwrap();
        assert_eq!(winner.description, best.description, "seed {seed}");
        assert_eq!(winner.fitness, best.fitness, "seed {seed}");
        assert!(
            result.n_fitted() < 36,
            "seed {seed}: the GA fitted the whole grid"
        );
        assert_eq!(result.generations.len(), 7, "seed {seed}");
        assert_eq!(
            result.generations.last().unwrap().best_fitness,
            best.fitness
        );
        // The steps are journalled per generation and per downhill round.
        let dirs = script.dirs();
        assert_eq!(dirs[0], "input");
        assert_eq!(dirs[1], "generation-0");
        assert!(dirs.iter().any(|d| d.starts_with("downhill-")), "{dirs:?}");
        // A genome the GA proposed twice was fitted once.
        let mut described: Vec<&str> = result
            .rows
            .iter()
            .filter(|r| r.duplicate_of.is_none())
            .map(|r| r.description.as_str())
            .collect();
        described.sort();
        let n = described.len();
        described.dedup();
        assert_eq!(
            described.len(),
            n,
            "seed {seed}: a grid point was fitted twice"
        );
    }
}

// ── charges the criterion cannot see ────────────────────────────────────────

#[test]
fn a_gene_on_a_parameter_the_structure_removed_is_non_influential() {
    // `KA` exists on FO only: on INST the covariate gene changes nothing.
    let space = space_with(
        "ABSORPTION([INST,FO])",
        &[effect("KA", "WT", CovariateEffect::Pow, true)],
        &[],
    );
    let script = ScriptedFitter::new(
        key,
        &[
            ("input", 500.0),
            ("INST;P0", 490.0),
            ("P0", 495.0),
            ("P0;KA-WT=power", 485.0),
        ],
    );
    let result = run(&script, space, &options(Algorithm::Exhaustive));
    let p = Penalties::default();
    // Grid: ABSORPTION (FO, INST — alphabetical, as `space_features` sorts)
    // × KA-WT (none, power).
    let inst_none = &result.rows[3];
    let inst_power = &result.rows[4];
    assert_eq!(inst_none.description, "ABSORPTION=INST;KA-WT=none");
    assert_eq!(inst_power.description, "ABSORPTION=INST;KA-WT=power");
    assert_eq!(inst_none.non_influential, 0);
    assert_eq!(inst_power.non_influential, 1);
    assert_eq!(inst_none.fitness, 490.0);
    assert_eq!(inst_power.fitness, 490.0 + p.non_influential);
    assert!(
        inst_power.effects.is_empty(),
        "the relation was not written"
    );
    assert!(
        !result.models["run4"].render().contains("KA ~ WT"),
        "a relation on a parameter the model does not declare"
    );
    // The two INST genomes render to one model: the scripted fitter fits
    // both (dedup is the runner's), so the search sees two fits — but the
    // features say which gene was dead.
    let c = script.candidates_in("candidates");
    assert_eq!(c[3].features.get("KA-WT"), Some("power (non-influential)"));
    // The live gene is a real relation, and the best model.
    let fo_power = &result.rows[2];
    assert_eq!(fo_power.description, "ABSORPTION=FO;KA-WT=power");
    assert_eq!(fo_power.non_influential, 0);
    assert!(result.models["run2"].render().contains("KA ~ WT power"));
    assert_eq!(result.final_id, "run2");
}

#[test]
fn an_unbuildable_point_is_a_row_with_the_crash_fitness_and_is_never_fitted() {
    let space = space("ABSORPTION([INST,FO]); LAGTIME([OFF,ON])");
    let script = ScriptedFitter::new(key, &[("input", 500.0)]);
    let result = run(&script, space, &options(Algorithm::Exhaustive));
    assert_eq!(result.rows.len(), 5);
    let bolus_lag = &result.rows[4];
    assert_eq!(bolus_lag.description, "ABSORPTION=INST;LAGTIME=ON");
    assert_eq!(bolus_lag.fitness, Penalties::default().crash);
    assert!(bolus_lag
        .error
        .as_ref()
        .unwrap()
        .message
        .contains("not generated"));
    assert!(bolus_lag.rank.is_none());
    assert_eq!(
        script.candidates_in("candidates").len(),
        3,
        "the unbuildable point was not submitted"
    );
    // pyDarwin's pair table applies too: FO with a one-transit chain.
    let space = self::space("TRANSITS([0,1], NODEPOT)");
    let script = ScriptedFitter::new(key, &[("input", 500.0)]);
    let result = run(&script, space, &options(Algorithm::Exhaustive));
    let one = result
        .rows
        .iter()
        .find(|r| r.description == "TRANSITS=1")
        .unwrap();
    assert!(
        one.error
            .as_ref()
            .unwrap()
            .message
            .contains("does not combine"),
        "{:?}",
        one.error
    );
}

#[test]
fn a_failed_gate_is_charged_and_never_selected_and_a_crash_is_the_crash_value() {
    let space = space("PERIPHERALS(0..2)");
    let mut script = ScriptedFitter::new(
        key,
        &[
            ("input", 500.0),
            ("P0", 500.0),
            ("P1", 470.0),
            ("P2", 480.0),
        ],
    );
    script.failing = vec!["P1".into()];
    script.erroring = vec!["run3".into()];
    let result = run(&script, space, &options(Algorithm::Exhaustive));
    let p = Penalties::default();
    let p1 = result
        .rows
        .iter()
        .find(|r| r.description == "PERIPHERALS=1")
        .unwrap();
    assert!(!p1.passed);
    assert_eq!(p1.criterion, 470.0);
    assert_eq!(
        p1.fitness,
        470.0 + p.gate,
        "the gate charge, on top of the criterion"
    );
    assert!(p1.rank.is_none());
    let p2 = result
        .rows
        .iter()
        .find(|r| r.description == "PERIPHERALS=2")
        .unwrap();
    assert!(p2.error.is_some());
    assert_eq!(p2.fitness, p.crash);
    assert!(p2.criterion.is_nan());
    // The one clean candidate wins, at the input's OFV — and the input,
    // tied with it and created first, ranks above it without being
    // selectable.
    assert_eq!(result.final_id, "run1");
    assert_eq!(result.row("input").unwrap().rank, Some(1));
    assert!(!result.row("input").unwrap().selected);
    assert_eq!(result.row("run1").unwrap().rank, Some(2));

    // With every candidate refused, the input is the final model and the
    // notes say so.
    let space = self::space("PERIPHERALS(0..1)");
    let mut script = ScriptedFitter::new(key, &[("input", 500.0)]);
    script.failing = vec!["P0".into(), "P1".into()];
    let result = run(&script, space, &options(Algorithm::Exhaustive));
    assert_eq!(result.final_id, "input");
    assert!(result.row("input").unwrap().selected);
    assert!(
        result
            .notes
            .iter()
            .any(|n| n.contains("no candidate passed")),
        "{:?}",
        result.notes
    );
}

#[test]
fn the_penalized_criterion_ranks_and_the_gate_charge_stacks_on_it() {
    let space = space("PERIPHERALS(0..1)");
    let mut script = ScriptedFitter::new(key, &[("input", 500.0), ("P0", 500.0), ("P1", 470.0)]);
    script.failing = vec!["P1".into()];
    let options = GlobalsearchOptions {
        rank: RankType::Penalized,
        ..GlobalsearchOptions::default()
    };
    let result = run(&script, space, &options);
    let p = Penalties::default();
    let p1 = result
        .rows
        .iter()
        .find(|r| r.description == "PERIPHERALS=1")
        .unwrap();
    // The scripted fit carries the warfarin fixture's tally and no
    // covariance step, so the criterion is OFV + parameter charges +
    // covariance; the gate charge then comes on top. The fixture's own
    // terms are read back rather than hard-coded, since they are the
    // fixture's.
    let fit = crate::search::test_support::converged_fit(470.0);
    let expected_criterion = p.score(&fit);
    assert_eq!(p1.criterion, expected_criterion);
    assert_eq!(p1.fitness, expected_criterion + p.gate);
    assert_eq!(result.criterion.label(), "penalized");
}

#[test]
fn a_point_rendering_to_a_model_already_fitted_is_a_duplicate_across_batches() {
    // Drive the evaluator directly: INST genomes with and without the dead
    // KA gene, in two batches.
    let space = space_with(
        "ABSORPTION([INST,FO])",
        &[effect("KA", "WT", CovariateEffect::Pow, true)],
        &[],
    );
    let script = ScriptedFitter::new(key, &[("INST;P0", 490.0)]);
    let options = options(Algorithm::Ga);
    let mut evaluator = Evaluator {
        fitter: &script,
        space: &space,
        options: &options,
        criterion: options.criterion(),
        progress: None,
        root: space.input_model.clone(),
        by_genome: HashMap::new(),
        by_hash: HashMap::new(),
        rows: Vec::new(),
        models: HashMap::new(),
        fits: HashMap::new(),
        best_fitness: f64::INFINITY,
        notes: Vec::new(),
        next_id: 0,
        cancelled: false,
    };
    // ABSORPTION's alleles are FO (0) and INST (1).
    let first = ga::Oracle::evaluate(&mut evaluator, "generation-0", &[vec![1, 0]]).unwrap();
    let second =
        ga::Oracle::evaluate(&mut evaluator, "generation-1", &[vec![1, 1], vec![1, 0]]).unwrap();
    assert_eq!(first, vec![490.0]);
    assert_eq!(
        second,
        vec![490.0 + Penalties::default().non_influential, 490.0]
    );
    assert_eq!(evaluator.rows.len(), 2);
    assert_eq!(evaluator.rows[1].duplicate_of.as_deref(), Some("run1"));
    assert_eq!(evaluator.rows[1].non_influential, 1);
    assert_eq!(
        script.dirs(),
        vec!["generation-0"],
        "the second batch needed no fit at all"
    );
    // A batch repeating a genome asks once and answers twice.
    let third =
        ga::Oracle::evaluate(&mut evaluator, "generation-2", &[vec![0, 0], vec![0, 0]]).unwrap();
    assert_eq!(third[0], third[1]);
    assert_eq!(script.candidates_in("generation-2").len(), 1);
    // Fits are kept only while they can still win: the INST fit (criterion
    // 490, the best) stays; the FO point at the fallback 1000 was dropped
    // the moment its batch closed. Three models evaluated, one fit resident.
    assert_eq!(evaluator.best_fitness, 490.0);
    assert_eq!(evaluator.fits.len(), 1);
    assert_eq!(evaluator.models.len(), 3);
    let (criterion, fit) = evaluator.fits.values().next().unwrap();
    assert_eq!((*criterion, fit.ofv), (490.0, 490.0));
}

#[test]
fn a_cancellation_returns_the_rows_so_far() {
    let space = ga_space();
    let mut script = landscape_fitter(&space);
    script.cancel_after = Some(2);
    let options = GlobalsearchOptions {
        ga: GaOptions {
            population_size: 6,
            generations: 5,
            seed: 9,
            ..GaOptions::default()
        },
        ..options(Algorithm::Ga)
    };
    let result = run(&script, space, &options);
    assert!(result.cancelled);
    assert_eq!(script.dirs(), vec!["input", "generation-0"]);
    assert_eq!(result.rows.len(), 7, "the input and the initial population");
    assert!(result.row(&result.final_id).unwrap().selected);
}

#[test]
fn the_input_seeds_every_candidate_and_forced_effects_reach_every_one() {
    let space = space_with(
        "PERIPHERALS(0..1)",
        &[effect("V", "WT", CovariateEffect::Pow, false)],
        &[],
    );
    assert_eq!(space.forced.len(), 1);
    let script = ScriptedFitter::new(key, &[("input", 500.0)]);
    let result = run(&script, space, &options(Algorithm::Exhaustive));
    for c in script.candidates_in("candidates") {
        let text = c.model.render();
        assert!(text.contains("V ~ WT power"), "{text}");
    }
    for r in &result.rows[1..] {
        assert_eq!(r.effects.len(), 1);
    }
}

// ── the section ─────────────────────────────────────────────────────────────

fn load(text: &str) -> SearchConfig {
    SearchConfig::from_str(text, std::path::Path::new(".")).expect("config")
}

#[test]
fn the_section_is_read_with_its_defaults_and_refuses_what_it_cannot_honour() {
    let cfg = load("base = \"m.ferx\"\n[space]\nmfl = \"PERIPHERALS(0..1)\"\n");
    let o = GlobalsearchOptions::from_config(&cfg).unwrap();
    assert_eq!(o, GlobalsearchOptions::default());
    assert_eq!(
        o.rank,
        RankType::Penalized,
        "the file's silence is pyDarwin's ranking"
    );
    assert_eq!(o.criterion(), Criterion::Penalized(Penalties::default()));

    let cfg = load(
        "base = \"m.ferx\"\n[space]\nmfl = \"PERIPHERALS(0..1)\"\n[rank]\ntype = \"bic\"\n\
         [rank.penalties]\ngate = 50\n[globalsearch]\nalgorithm = \"exhaustive\"\n\
         max_models = 1000\niiv_strategy = \"add_diagonal\"\n[globalsearch.ga]\n\
         population_size = 30\ngenerations = 3\n",
    );
    let o = GlobalsearchOptions::from_config(&cfg).unwrap();
    assert_eq!(o.algorithm, Algorithm::Exhaustive);
    assert_eq!(o.max_models, 1000);
    assert_eq!(o.iiv_strategy, IivStrategy::AddDiagonal);
    assert_eq!(o.ga.population_size, 30);
    assert_eq!(o.ga.generations, 3);
    assert_eq!(o.rank, RankType::Bic);
    assert_eq!(o.penalties.gate, 50.0);
    assert_eq!(o.criterion().label(), "bic_mixed");

    for (text, msg) in [
        (
            "[globalsearch]\niiv_strategy = \"fullblock\"\n",
            "iiv_strategy = \"fullblock\"",
        ),
        (
            "[globalsearch]\nmax_models = 0\n",
            "max_models must be at least 1",
        ),
        (
            "[globalsearch]\npopulation = 3\n",
            "unknown field `population`",
        ),
        (
            "[globalsearch.ga]\npopulation_size = 1\n",
            "population_size must be at least 2",
        ),
        (
            "[rank.penalties]\ncrash = -1\n",
            "[rank.penalties] crash = -1",
        ),
    ] {
        let text = format!("base = \"m.ferx\"\n[space]\nmfl = \"PERIPHERALS(0..1)\"\n{text}");
        let e = SearchConfig::from_str(&text, std::path::Path::new("."))
            .and_then(|cfg| GlobalsearchOptions::from_config(&cfg))
            .unwrap_err();
        assert!(e.contains(msg), "{e}");
    }
    // A space the grid cannot lay out is refused before any data is read.
    let cfg = load("base = \"m.ferx\"\n[space]\nmfl = \"IIV(CL, exp)\"\n");
    let e = GlobalsearchOptions::check_space(&cfg).unwrap_err();
    assert!(e.contains("`IIV`"), "{e}");
    // …and a file with no space at all.
    let cfg = load("base = \"m.ferx\"\n");
    let e = GlobalsearchOptions::from_config(&cfg).unwrap_err();
    assert!(e.contains("globalsearch needs a [space]"), "{e}");
}

// ── the report ──────────────────────────────────────────────────────────────

#[test]
fn the_report_writes_the_table_the_trajectory_and_the_final_model() {
    let space = ga_space();
    let script = landscape_fitter(&space);
    let options = GlobalsearchOptions {
        ga: GaOptions {
            population_size: 8,
            generations: 3,
            seed: 1,
            ..GaOptions::default()
        },
        ..options(Algorithm::Ga)
    };
    let result = run(&script, space, &options);
    let dir = tempfile::tempdir().unwrap();
    write_report(dir.path(), &result).unwrap();

    let table = std::fs::read_to_string(models_path(dir.path())).unwrap();
    let header = table.lines().next().unwrap();
    assert_eq!(header, MODEL_COLUMNS.join(","));
    assert_eq!(table.lines().count(), result.rows.len() + 1);
    let selected: Vec<&str> = table
        .lines()
        .skip(1)
        .filter(|l| l.split(',').nth(20) == Some("true"))
        .collect();
    assert_eq!(selected.len(), 1);
    assert!(selected[0].starts_with(&format!("{},input,", result.final_id)));

    let generations = std::fs::read_to_string(report::generations_path(dir.path())).unwrap();
    assert_eq!(
        generations.lines().count(),
        5,
        "a header and four generations"
    );
    assert!(generations.lines().nth(1).unwrap().starts_with("0,"));

    let final_text = std::fs::read_to_string(final_model_path(dir.path())).unwrap();
    assert!(final_text.contains("[covariate_model]"), "{final_text}");
    assert!(models_dir(dir.path()).join("input.ferx").exists());
    assert!(models_dir(dir.path())
        .join(format!("{}.ferx", result.final_id))
        .exists());

    let summary = render_summary(&result);
    assert!(summary.contains("Grid: 36 points over 4 axes"), "{summary}");
    assert!(
        summary.contains("CL-WT: none | power | exponential"),
        "{summary}"
    );
    assert!(summary.contains("SELECTED"), "{summary}");
    assert!(summary.contains("generation"), "{summary}");
    assert!(
        summary.contains(&format!("Final model: {}", result.final_id)),
        "{summary}"
    );
}

#[test]
fn default_dir_sits_next_to_the_config() {
    assert_eq!(
        default_dir(std::path::Path::new("runs/warfarin.ferxsearch")),
        std::path::PathBuf::from("runs/warfarin-globalsearch")
    );
}

// ── review findings on PR #1299 ─────────────────────────────────────────────

#[test]
fn a_forced_effect_on_a_parameter_the_structure_removed_makes_the_point_unbuildable() {
    // `COVARIATE(KA, WT, pow)` without the `?` is part of every model in the
    // space; an INST candidate has no KA, so it is not a model of the space
    // — refused, never fitted without the relation, never selected.
    let space = space_with(
        "ABSORPTION([INST,FO])",
        &[effect("KA", "WT", CovariateEffect::Pow, false)],
        &[],
    );
    let script = ScriptedFitter::new(key, &[("input", 500.0), ("INST;P0", 400.0), ("P0", 490.0)]);
    let result = run(&script, space, &options(Algorithm::Exhaustive));
    let inst = result
        .rows
        .iter()
        .find(|r| r.description == "ABSORPTION=INST")
        .unwrap();
    assert!(inst.error.is_some(), "{inst:?}");
    assert!(
        inst.error
            .as_ref()
            .unwrap()
            .message
            .contains("forced effect KA-WT-power"),
        "{:?}",
        inst.error
    );
    assert_eq!(inst.fitness, Penalties::default().crash);
    assert!(inst.rank.is_none());
    assert!(!inst.selected);
    // Only the FO point was submitted, and it carries the relation.
    let submitted = script.candidates_in("candidates");
    assert_eq!(submitted.len(), 1);
    assert!(submitted[0].model.render().contains("KA ~ WT power"));
    assert_eq!(result.final_id, submitted[0].id);
    assert!(result.final_model.render().contains("KA ~ WT power"));
}

#[test]
fn a_run_whose_winner_is_a_cross_batch_duplicate_hands_back_the_fit() {
    // The review's case (PR #1299): on INST/FO × KA-WT the two INST genomes
    // render to one model; when the dead-gene one is fitted first and the
    // clean one is proposed in a later batch, the clean one is a cross-batch
    // duplicate stored without a fit — and, carrying no non-influential
    // charge, it outranks its representative and wins. The GA is seeded, so
    // the seeds are walked until a run has exactly that shape; the fixture
    // asserts the shape was reached, so it cannot go green by missing it.
    let mut reached = false;
    for seed in 1..=200u64 {
        let space = space_with(
            "ABSORPTION([INST,FO])",
            &[effect("KA", "WT", CovariateEffect::Pow, true)],
            &[],
        );
        let script =
            ScriptedFitter::new(key, &[("input", 500.0), ("INST;P0", 400.0), ("P0", 490.0)]);
        let options = GlobalsearchOptions {
            ga: GaOptions {
                population_size: 2,
                generations: 3,
                elites: 1,
                downhill_period: 0,
                final_downhill: false,
                seed,
                ..GaOptions::default()
            },
            ..options(Algorithm::Ga)
        };
        let result = run(&script, space, &options);
        let winner = result.row(&result.final_id).unwrap();
        let Some(rep) = winner.duplicate_of.clone() else {
            continue;
        };
        let rep_row = result.row(&rep).unwrap();
        if rep_row.step == winner.step {
            continue; // a within-batch duplicate: the runner's report has the fit
        }
        reached = true;
        assert_eq!(
            winner.description, "ABSORPTION=INST;KA-WT=none",
            "seed {seed}"
        );
        assert_eq!(rep_row.non_influential, 1, "seed {seed}");
        assert_eq!(
            result.final_fit.as_ref().map(|f| f.ofv),
            Some(400.0),
            "seed {seed}: {} (duplicate of {rep}) came back without its fit",
            winner.id
        );
        assert!(
            !result
                .notes
                .iter()
                .any(|n| n.contains("not in the journal cache")),
            "seed {seed}: {:?}",
            result.notes
        );
        break;
    }
    assert!(
        reached,
        "no seed produced a run whose winner is a cross-batch duplicate; the fixture cannot \
         see the defect it exists to catch"
    );
}

// ── second review pass on PR #1299 ──────────────────────────────────────────

#[test]
fn a_pair_that_is_both_forced_and_searched_is_refused_at_build() {
    let text = ModelText::parse(BASE).unwrap();
    let e = Space::build(
        text,
        None,
        &[],
        &[
            effect("CL", "WT", CovariateEffect::Pow, false),
            effect("CL", "WT", CovariateEffect::Exp, true),
            effect("V", "WT", CovariateEffect::Pow, true),
        ],
        &[],
        Vec::new(),
    )
    .unwrap_err();
    assert!(e.contains("`CL-WT` is both forced"), "{e}");
}

#[test]
fn a_covariate_edit_the_layer_refuses_costs_one_point_not_the_run() {
    // A base that already declares `CL ~ WT`, handed to `build` as if it
    // did not: every genome with a CL-WT allele asks the edit layer to add
    // a second line for the pair, which it refuses. That is a crash-valued
    // row for the point, and the search still returns with every other
    // fit and its files.
    let base = format!("{BASE}\n[covariate_model]\n  CL ~ WT power\n");
    let text = ModelText::parse(&base).unwrap();
    let space = Space::build(
        text,
        None,
        &[],
        &[
            effect("CL", "WT", CovariateEffect::Exp, true),
            effect("V", "WT", CovariateEffect::Pow, true),
        ],
        &[],
        Vec::new(),
    )
    .unwrap();
    let script = ScriptedFitter::new(key, &[("input", 500.0), ("V-WT=power", 480.0)]);
    let result = run(&script, space, &options(Algorithm::Exhaustive));
    assert_eq!(result.rows.len(), 5);
    let refused: Vec<&ModelRow> = result
        .rows
        .iter()
        .filter(|r| r.description.contains("CL-WT=exponential"))
        .collect();
    assert_eq!(refused.len(), 2);
    for r in refused {
        let e = r.error.as_ref().expect("a refused point");
        assert!(e.message.contains("already declares"), "{}", e.message);
        assert_eq!(r.fitness, Penalties::default().crash);
        assert!(r.rank.is_none());
    }
    assert_eq!(script.candidates_in("candidates").len(), 2);
    assert_eq!(
        result.row(&result.final_id).unwrap().description,
        "CL-WT=none;V-WT=power"
    );
}

#[test]
fn a_runner_warning_that_recurs_per_batch_is_one_note() {
    let space = space("PERIPHERALS(0..2)");
    let mut script = ScriptedFitter::new(key, &[("input", 500.0)]);
    script.warnings = vec!["no search directory found under `nope` to reuse fits from".into()];
    let options = GlobalsearchOptions {
        ga: GaOptions {
            population_size: 2,
            generations: 3,
            elites: 1,
            seed: 2,
            ..GaOptions::default()
        },
        ..options(Algorithm::Ga)
    };
    let result = run(&script, space, &options);
    assert!(script.dirs().len() >= 3, "{:?}", script.dirs());
    assert_eq!(
        result
            .notes
            .iter()
            .filter(|n| n.contains("no search directory"))
            .count(),
        1,
        "{:?}",
        result.notes
    );
}

#[test]
fn a_cutoff_keeps_the_input_unless_a_candidate_beats_it_by_that_much() {
    let table = [
        ("input", 500.0),
        ("P0", 500.0),
        ("P1", 497.0),
        ("P2", 496.0),
    ];
    let with = |cutoff: Option<f64>| GlobalsearchOptions {
        cutoff,
        ..options(Algorithm::Exhaustive)
    };
    // The best candidate is 4 better than the input: a cutoff of 5 keeps
    // the input, a cutoff of 4 selects it.
    let result = run(
        &ScriptedFitter::new(key, &table),
        space("PERIPHERALS(0..2)"),
        &with(Some(5.0)),
    );
    assert_eq!(result.final_id, "input");
    assert!(result.row("input").unwrap().selected);
    assert!(
        result
            .notes
            .iter()
            .any(|n| n.contains("beat the input by [rank] cutoff = 5")),
        "{:?}",
        result.notes
    );
    assert_eq!(result.final_fit.as_ref().map(|f| f.ofv), Some(500.0));
    let result = run(
        &ScriptedFitter::new(key, &table),
        space("PERIPHERALS(0..2)"),
        &with(Some(4.0)),
    );
    assert_eq!(
        result.row(&result.final_id).unwrap().description,
        "PERIPHERALS=2"
    );
    // The input failing the gate: the cutoff has no reference, the best
    // candidate is selected and the notes say why.
    let mut script = ScriptedFitter::new(key, &table);
    script.failing = vec!["input".into()];
    let result = run(&script, space("PERIPHERALS(0..2)"), &with(Some(50.0)));
    assert_eq!(
        result.row(&result.final_id).unwrap().description,
        "PERIPHERALS=2"
    );
    assert!(
        result
            .notes
            .iter()
            .any(|n| n.contains("cutoff cannot be applied")),
        "{:?}",
        result.notes
    );
    // And the option is validated.
    let e = with(Some(-1.0)).validate().unwrap_err();
    assert!(e.contains("[rank] cutoff = -1"), "{e}");
    let cfg =
        load("base = \"m.ferx\"\n[space]\nmfl = \"PERIPHERALS(0..1)\"\n[rank]\ncutoff = 3.84\n");
    assert_eq!(
        GlobalsearchOptions::from_config(&cfg).unwrap().cutoff,
        Some(3.84)
    );
}
