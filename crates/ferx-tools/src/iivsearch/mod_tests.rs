//! Tier-1 tests for the variability-structure search logic (#1183).
//!
//! Every test drives [`search`] with a *scripted* fitter — OFVs keyed by the
//! candidate's structure description (`[CL,V]+[KA]`) or by id — so what is
//! under test is the enumeration and the decisions: which candidates each
//! algorithm generates from which parent, in Pharmpy's order; what a step
//! ranks and picks; how a cutoff, a failed gate, a kept η and a fixed η
//! change that; what the edits write; and what the report says. The path
//! from `ModelText` through the runner and `fit()` is
//! `tests/iivsearch_end_to_end.rs`; the trajectory against Pharmpy's own
//! run is `tests/iivsearch_pharmpy_anchor.rs`.

use std::path::Path;

use ferx_core::edit::ModelText;

use super::*;
use crate::search::mfl::Mfl;
use crate::search::test_support::ScriptedFitter;
use crate::search::SearchConfig;

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

const ALL: &str = "IIV?([CL,V,KA],EXP);COVARIANCE?(IIV,[CL,V,KA])";
const KEEP_CL: &str = "IIV(CL,EXP);IIV?([V,KA],EXP);COVARIANCE?(IIV,[CL,V,KA])";

/// The candidate's structure in Pharmpy's spelling, read off its feature
/// vector — the key the scripted OFV table uses.
fn key(c: &Candidate) -> String {
    let etas: Vec<String> = c
        .features
        .get("iiv")
        .unwrap_or("")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let blocks: Vec<Vec<String>> = c
        .features
        .get("blocks")
        .unwrap_or("")
        .split('+')
        .filter(|s| !s.is_empty())
        .map(|b| {
            b.trim_matches(['[', ']'])
                .split(',')
                .map(str::to_string)
                .collect()
        })
        .collect();
    IivStructure::new(etas, blocks).description()
}

fn space_of(base: &str, mfl: &str) -> Space {
    let text = ModelText::parse(base).unwrap();
    let mfl = Mfl::parse(mfl).unwrap();
    Space::build(text, mfl.features(), Vec::new()).expect("a valid space")
}

fn space(mfl: &str) -> Space {
    space_of(BASE, mfl)
}

fn options(algorithm: Algorithm) -> IivsearchOptions {
    IivsearchOptions {
        algorithm,
        rank: RankType::Ofv,
        ..IivsearchOptions::default()
    }
}

fn run(script: &ScriptedFitter, space: Space, options: &IivsearchOptions) -> IivsearchResult {
    search(script, space, options, None).expect("search")
}

/// `(id, description)` of the rows fitted in `step`, in generation order.
fn fitted_in(result: &IivsearchResult, step: usize) -> Vec<(String, String)> {
    result
        .rows
        .iter()
        .filter(|r| r.step == step)
        .map(|r| (r.id.clone(), r.structure.description()))
        .collect()
}

fn pair(id: &str, desc: &str) -> (String, String) {
    (id.into(), desc.into())
}

fn config(text: &str) -> Result<SearchConfig, String> {
    SearchConfig::from_str(text, Path::new("."))
}

// ── options ─────────────────────────────────────────────────────────────────

#[test]
fn options_take_pharmpys_defaults_and_read_bic_as_the_bic_iiv() {
    let cfg = config("base = \"b.ferx\"\n[space]\nmfl = \"IIV?([CL,V],EXP)\"\n").unwrap();
    let o = IivsearchOptions::from_config(&cfg).unwrap();
    assert_eq!(o.algorithm, Algorithm::TopDownExhaustive);
    assert_eq!(o.correlation_algorithm, None);
    assert!(o.block_stage());
    assert_eq!(o.rank, RankType::BicIiv);
    assert_eq!(o.criterion(), Criterion::Bic(ferx_core::BicType::Iiv));
    assert_eq!(o.cutoff, None);
    assert_eq!(o.block_retries, 2);
    assert_eq!(
        o.starts, 3,
        "[run] retries defaults to 2, plus the exact start"
    );
    assert_eq!(o.starts_for(1), 3);
    assert_eq!(o.starts_for(2), 3);
    assert_eq!(o.starts_for(3), 5);
    assert_eq!(o.starts_for(4), 7);

    // `bic` is Pharmpy's `bic_iiv` here; the mixed BIC is asked for by name.
    let cfg =
        config("base = \"b.ferx\"\n[space]\nmfl = \"IIV?([CL,V],EXP)\"\n[rank]\ntype = \"bic\"\n")
            .unwrap();
    assert_eq!(
        IivsearchOptions::from_config(&cfg).unwrap().rank,
        RankType::BicIiv
    );
    let cfg = config(
        "base = \"b.ferx\"\n[space]\nmfl = \"IIV?([CL,V],EXP)\"\n[rank]\ntype = \"bic_mixed\"\n\
         cutoff = 3.84\n[run]\nretries = 0\n[iivsearch]\nalgorithm = \"bottom_up_stepwise\"\n\
         correlation_algorithm = \"skip\"\nblock_retries = 5\n",
    )
    .unwrap();
    let o = IivsearchOptions::from_config(&cfg).unwrap();
    assert_eq!(o.rank, RankType::BicMixed);
    assert_eq!(o.cutoff, Some(3.84));
    assert_eq!(o.algorithm, Algorithm::BottomUpStepwise);
    assert_eq!(o.correlation_algorithm, Some(CorrelationAlgorithm::Skip));
    assert!(!o.block_stage());
    assert_eq!(o.starts, 1);
    assert_eq!(o.starts_for(3), 6);

    // Pharmpy's `validate_input`.
    let err = config(
        "base = \"b.ferx\"\n[space]\nmfl = \"IIV?([CL,V],EXP)\"\n[iivsearch]\nalgorithm = \
         \"skip\"\n",
    )
    .and_then(|c| IivsearchOptions::from_config(&c))
    .unwrap_err();
    assert!(err.contains("nothing to search"), "{err}");
    let err = config(
        "base = \"b.ferx\"\n[space]\nmfl = \"IIV?([CL,V],EXP)\"\n[iivsearch]\nalgorithm = \
         \"simultaneous_stepwise\"\ncorrelation_algorithm = \"skip\"\n",
    )
    .and_then(|c| IivsearchOptions::from_config(&c))
    .unwrap_err();
    assert!(err.contains("no second stage"), "{err}");
    let err = config("base = \"b.ferx\"\n")
        .and_then(|c| IivsearchOptions::from_config(&c))
        .unwrap_err();
    assert!(err.contains("needs a [space] section"), "{err}");
    let err =
        config("base = \"b.ferx\"\n[space]\nmfl = \"IIV?([CL,V],EXP)\"\n[rank]\ncutoff = -1.0\n")
            .and_then(|c| IivsearchOptions::from_config(&c))
            .unwrap_err();
    assert!(err.contains("cutoff"), "{err}");
    // A skipped η stage with a block stage is fine.
    let cfg = config(
        "base = \"b.ferx\"\n[space]\nmfl = \"COVARIANCE?(IIV,[CL,V])\"\n[iivsearch]\n\
         algorithm = \"skip\"\ncorrelation_algorithm = \"top_down_exhaustive\"\n",
    )
    .unwrap();
    assert!(IivsearchOptions::from_config(&cfg).unwrap().block_stage());
}

// ── the space ───────────────────────────────────────────────────────────────

#[test]
fn the_space_reads_kept_and_searched_parameters_and_the_input_structure() {
    let s = space(KEEP_CL);
    assert_eq!(
        s.iiv,
        vec![
            ("CL".to_string(), true),
            ("KA".to_string(), false),
            ("V".to_string(), false)
        ],
        "alphabetical, Pharmpy's order"
    );
    assert_eq!(s.cov, vec!["CL", "KA", "V"]);
    assert!(s.forced_pairs.is_empty());
    assert_eq!(s.searched(), vec!["KA", "V"]);
    assert_eq!(s.kept(), vec!["CL"]);
    assert_eq!(s.input_structure.description(), "[CL]+[KA]+[V]");
    assert_eq!(s.eta_of["V"], ("ETA_V".to_string(), 0.04));

    // A forced covariance is a block every candidate carries; an existing
    // block is read as one.
    let blocked = BASE.replace("  omega ETA_CL ~ 0.09\n", "").replace(
        "  omega ETA_V ~ 0.04\n",
        "  block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04]\n",
    );
    let s = space_of(
        &blocked,
        "IIV?([CL,V,KA],EXP);COVARIANCE(IIV,[CL,V]);COVARIANCE?(IIV,[KA,V])",
    );
    assert_eq!(s.forced_pairs, vec![("CL".to_string(), "V".to_string())]);
    assert_eq!(s.input_structure.description(), "[CL,V]+[KA]");
    assert_eq!(
        s.forced_blocks(&["CL".into(), "KA".into(), "V".into()]),
        vec![vec!["CL".to_string(), "V".to_string()]]
    );

    // A parameter with no η yet gets a fresh name at Pharmpy's 0.09.
    let no_ka = BASE
        .replace("  KA = TVKA * exp(ETA_KA)", "  KA = TVKA")
        .replace("  omega ETA_KA ~ 0.30\n", "");
    let s = space_of(&no_ka, ALL);
    assert_eq!(s.eta_of["KA"], ("ETA_KA".to_string(), 0.09));
    assert_eq!(s.input_structure.description(), "[CL]+[V]");
}

#[test]
fn the_space_refuses_what_it_cannot_search_by_name() {
    let err = Space::build(
        ModelText::parse(&BASE.replace("  V = TVV * exp(ETA_V)", "  V = TVV * (1 + ETA_V)"))
            .unwrap(),
        Mfl::parse(ALL).unwrap().features(),
        Vec::new(),
    )
    .unwrap_err();
    assert!(err.contains("`V`"), "{err}");
    assert!(err.contains("canonical form"), "{err}");
    assert!(err.contains("`ETA_V`"), "{err}");

    let refuse = |mfl: &str| {
        Space::build(
            ModelText::parse(BASE).unwrap(),
            Mfl::parse(mfl).unwrap().features(),
            Vec::new(),
        )
        .unwrap_err()
    };
    let err = refuse("IIV?([CL,V],ADD)");
    assert!(err.contains("exponential form"), "{err}");
    let err = refuse("IIV?([CL,NOPE],EXP)");
    assert!(err.contains("`NOPE`"), "{err}");
    let err = refuse("COVARIATE?(CL,WT,pow)");
    assert!(err.contains("not a variability feature"), "{err}");
    let err = refuse("COVARIANCE?(IOV,[CL,V])");
    assert!(err.contains("IOV covariance"), "{err}");
    let err = refuse("COVARIANCE?(IIV,[CL])");
    assert!(err.contains("needs two"), "{err}");
    // A `FIX`ed η is kept and never blocked, with a note.
    let fixed = BASE.replace("omega ETA_KA ~ 0.30", "omega ETA_KA ~ 0.30 FIX");
    let s = space_of(&fixed, ALL);
    assert_eq!(s.kept(), vec!["KA"]);
    assert_eq!(s.cov, vec!["CL", "V"]);
    assert!(
        s.notes.iter().any(|n| n.contains("declared `FIX`")),
        "{:?}",
        s.notes
    );
    let err = Space::build(
        ModelText::parse(&fixed).unwrap(),
        Mfl::parse("IIV?([CL,V],EXP);COVARIANCE(IIV,[KA,V])")
            .unwrap()
            .features(),
        Vec::new(),
    )
    .unwrap_err();
    assert!(err.contains("`FIX`"), "{err}");
}

// ── top-down exhaustive ─────────────────────────────────────────────────────

#[test]
fn top_down_enumerates_every_subset_largest_first_then_the_blocks() {
    let script = ScriptedFitter::new(
        key,
        &[
            ("[CL]+[KA]+[V]", 100.0),
            ("[CL]+[V]", 99.0),
            ("[CL]+[KA]", 300.0),
            ("[CL,V]", 70.0),
        ],
    );
    let result = run(&script, space(ALL), &options(Algorithm::TopDownExhaustive));
    // The input lies on the space: it is the base, and no base row exists.
    assert_eq!(result.base_id, "input");
    assert!(result.row("base").is_none());
    // Pharmpy's order: subsets of {CL, KA, V} by size descending, lexicographic
    // within a size, the naive-pooled model last.
    assert_eq!(
        fitted_in(&result, 1),
        vec![
            pair("run1", "[CL]+[KA]"),
            pair("run2", "[CL]+[V]"),
            pair("run3", "[KA]+[V]"),
            pair("run4", "[CL]"),
            pair("run5", "[KA]"),
            pair("run6", "[V]"),
            pair("run7", ""),
        ]
    );
    assert_eq!(script.dirs(), vec!["input", "step-1", "step-2"]);
    // The η that went is gone from the text, its ω line with it.
    let run2 = script.model_of("step-1", "run2").render();
    assert!(!run2.contains("ETA_KA"), "{run2}");
    assert!(run2.contains("KA = TVKA\n"), "{run2}");
    // Step 2 blocks the retained η: with two of them, one candidate.
    assert_eq!(fitted_in(&result, 2), vec![pair("run8", "[CL,V]")]);
    let run8 = script.model_of("step-2", "run8");
    assert!(
        run8.block_lines("parameters")
            .iter()
            .any(|l| l.starts_with("block_omega (ETA_CL, ETA_V)")),
        "{}",
        run8.render()
    );
    // Rankings: step 1's parent is the input; step 2's the step-1 winner;
    // step 3 the comparison with the input.
    assert_eq!(result.steps.len(), 3);
    assert_eq!(result.steps[0].parent, "input");
    assert_eq!(result.steps[0].best, "run2");
    assert_eq!(result.steps[0].ranked[0].id, "run2");
    assert_eq!(result.steps[0].ranked[0].rank, Some(1));
    assert_eq!(result.steps[0].ranked[0].d_criterion, Some(1.0));
    assert_eq!(result.steps[1].parent, "run2");
    assert_eq!(result.steps[1].best, "run8");
    assert_eq!(result.steps[2].kind, StepKind::Input);
    assert_eq!(result.steps[2].best, "run8");
    assert_eq!(result.final_id, "run8");
    assert_eq!(result.final_structure.description(), "[CL,V]");
    assert!(result.row("run8").unwrap().selected);
    assert_eq!(result.row("run2").unwrap().rank, Some(1));
    assert_eq!(result.row("run2").unwrap().d_criterion, Some(-1.0));
    assert_eq!(result.row("run1").unwrap().rank, Some(3));
    assert!(!result.cancelled);
}

#[test]
fn top_down_blocks_enumerate_every_clique_plus_the_diagonal_from_a_blocked_base() {
    // A base that already carries a block: the all-diagonal model is a
    // candidate, so are the other cliques; the base's own structure is not.
    let blocked = BASE.replace("  omega ETA_CL ~ 0.09\n", "").replace(
        "  omega ETA_V ~ 0.04\n",
        "  block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04]\n",
    );
    let script = ScriptedFitter::new(key, &[("[CL,V]+[KA]", 100.0), ("[CL,KA,V]", 90.0)]);
    let mut o = options(Algorithm::Skip);
    o.correlation_algorithm = Some(CorrelationAlgorithm::TopDownExhaustive);
    let result = run(
        &script,
        space_of(&blocked, "COVARIANCE?(IIV,[CL,V,KA])"),
        &o,
    );
    assert_eq!(
        fitted_in(&result, 1),
        vec![
            pair("run1", "[CL]+[KA]+[V]"),
            pair("run2", "[CL,KA]+[V]"),
            pair("run3", "[KA,V]+[CL]"),
            pair("run4", "[CL,KA,V]"),
        ]
    );
    // The diagonal candidate split the block; the 3-block was made from the
    // split diagonals and gets the extra starts.
    let run1 = script.model_of("step-1", "run1");
    assert!(!run1.render().contains("block_omega"), "{}", run1.render());
    assert!(run1
        .block_lines("parameters")
        .contains(&"omega ETA_CL ~ 0.09".to_string()));
    let run4 = script
        .candidates_in("step-1")
        .into_iter()
        .find(|c| c.id == "run4")
        .unwrap();
    assert_eq!(run4.n_starts, Some(5));
    let block = run4
        .model
        .block_lines("parameters")
        .into_iter()
        .find(|l| l.starts_with("block_omega ("))
        .expect("the 3-block is written");
    for eta in ["ETA_CL", "ETA_V", "ETA_KA"] {
        assert!(block.contains(eta), "{block}");
    }
    assert_eq!(result.row("run4").unwrap().starts, 5);
    assert_eq!(result.row("run2").unwrap().starts, 3);
    assert_eq!(result.final_id, "run4");
}

// ── bottom-up and simultaneous ──────────────────────────────────────────────

#[test]
fn bottom_up_starts_from_the_kept_eta_and_adds_one_at_a_time() {
    let script = ScriptedFitter::new(
        key,
        &[
            ("[CL]+[KA]+[V]", 100.0),
            ("[CL]", 300.0),
            ("[CL]+[KA]", 250.0),
            ("[CL]+[V]", 101.0),
            ("[CL,V]", 80.0),
        ],
    );
    let script = script.by_id(&[("run3", 102.0)]);
    let result = run(
        &script,
        space(KEEP_CL),
        &options(Algorithm::BottomUpStepwise),
    );
    // The base is derived: the input minus the searched η.
    assert_eq!(result.base_id, "base");
    assert_eq!(result.row("base").unwrap().structure.description(), "[CL]");
    assert_eq!(result.row("base").unwrap().parent.as_deref(), Some("input"));
    assert_eq!(
        fitted_in(&result, 1),
        vec![pair("run1", "[CL]+[KA]"), pair("run2", "[CL]+[V]")]
    );
    // Step 2 adds the remaining η; it does not improve, so the stage ends.
    assert_eq!(fitted_in(&result, 2), vec![pair("run3", "[CL]+[KA]+[V]")]);
    assert_eq!(result.steps[1].best, "run2");
    // The block stage runs on the step winner.
    assert_eq!(fitted_in(&result, 3), vec![pair("run4", "[CL,V]")]);
    assert_eq!(result.steps[2].kind, StepKind::BlockStructure);
    // And the final comparison prefers the search's model.
    assert_eq!(result.steps[3].kind, StepKind::Input);
    assert_eq!(result.final_id, "run4");
    assert!(result
        .notes
        .iter()
        .any(|n| n.contains("not the search's base")));
    // The re-added η took the input's variance rather than Pharmpy's 0.09.
    let run2 = script.model_of("step-1", "run2");
    assert!(run2
        .block_lines("parameters")
        .contains(&"omega ETA_V ~ 0.04".to_string()));
}

#[test]
fn bottom_up_as_fullblock_blocks_every_eta_it_adds() {
    let script = ScriptedFitter::new(key, &[("[CL]", 300.0), ("[CL,V]", 80.0)]);
    let mut o = options(Algorithm::BottomUpStepwise);
    o.as_fullblock = true;
    o.correlation_algorithm = Some(CorrelationAlgorithm::Skip);
    let result = run(&script, space(KEEP_CL), &o);
    assert_eq!(
        fitted_in(&result, 1),
        vec![pair("run1", "[CL,KA]"), pair("run2", "[CL,V]")]
    );
    assert_eq!(fitted_in(&result, 2), vec![pair("run3", "[CL,KA,V]")]);
    assert_eq!(result.final_id, "run2");
}

#[test]
fn simultaneous_tries_each_new_eta_diagonal_in_a_block_and_paired() {
    let script = ScriptedFitter::new(
        key,
        &[
            ("[CL]", 300.0),
            ("[CL]+[V]", 101.0),
            ("[CL,V]", 80.0),
            ("[CL,V]+[KA]", 79.0),
            ("[CL,KA,V]", 85.0),
        ],
    );
    let result = run(
        &script,
        space(KEEP_CL),
        &options(Algorithm::SimultaneousStepwise),
    );
    // From [CL]: KA and V, each diagonal and each paired with CL.
    assert_eq!(
        fitted_in(&result, 1),
        vec![
            pair("run1", "[CL]+[KA]"),
            pair("run2", "[CL,KA]"),
            pair("run3", "[CL]+[V]"),
            pair("run4", "[CL,V]"),
        ]
    );
    // From [CL,V]: KA diagonal, and KA inside the block — no single η is
    // left to pair with.
    assert_eq!(
        fitted_in(&result, 2),
        vec![pair("run5", "[CL,V]+[KA]"), pair("run6", "[CL,KA,V]")]
    );
    assert_eq!(result.steps[1].kind, StepKind::Simultaneous);
    // No block stage after the simultaneous algorithm.
    assert_eq!(result.steps.len(), 3);
    assert_eq!(result.steps[2].kind, StepKind::Input);
    assert_eq!(result.final_id, "run5");
}

// ── ranking ─────────────────────────────────────────────────────────────────

#[test]
fn ranking_prefers_the_parent_on_a_tie_honours_the_cutoff_and_skips_the_gate() {
    let cands = vec![
        ("a".to_string(), 99.0, true),
        ("b".to_string(), 100.0, true),
        ("c".to_string(), 50.0, false),
        ("d".to_string(), f64::NAN, true),
    ];
    let (ranked, best) = rank_models(("p", 100.0, true), &cands, None);
    assert_eq!(best, "a");
    let ids: Vec<&str> = ranked.iter().map(|r| r.id.as_str()).collect();
    // Eligible first by criterion, the parent before its tie; the rest after.
    assert_eq!(ids, vec!["a", "p", "b", "c", "d"]);
    assert_eq!(ranked[0].rank, Some(1));
    assert_eq!(ranked[0].d_criterion, Some(1.0));
    assert_eq!(ranked[1].rank, Some(2));
    assert_eq!(ranked[2].rank, Some(3));
    assert_eq!(ranked[3].rank, None);
    assert_eq!(ranked[3].d_criterion, Some(50.0));
    assert_eq!(ranked[4].d_criterion, None);
    // A cutoff the improvement does not reach keeps the parent.
    let (_, best) = rank_models(("p", 100.0, true), &cands, Some(2.0));
    assert_eq!(best, "p");
    let (_, best) = rank_models(("p", 100.0, true), &cands, Some(1.0));
    assert_eq!(best, "a");
    // A parent that failed the gate is beaten by any eligible candidate, and
    // kept when none is.
    let (ranked, best) = rank_models(("p", 10.0, false), &cands, None);
    assert_eq!(best, "a");
    assert!(ranked.iter().all(|r| r.d_criterion.is_none()));
    let (_, best) = rank_models(("p", 10.0, false), &[("x".to_string(), 1.0, false)], None);
    assert_eq!(best, "p");
}

#[test]
fn a_gated_candidate_cannot_win_and_the_input_returns_when_it_ranks_better() {
    // Top-down: the best-looking candidate fails the gate, the rest are
    // worse than the input, so the input keeps every step and no final
    // comparison is needed.
    let mut script = ScriptedFitter::new(key, &[("[CL]+[KA]+[V]", 100.0), ("[CL]+[V]", 50.0)]);
    script.failing = vec!["[CL]+[V]".into()];
    let result = run(&script, space(ALL), &options(Algorithm::TopDownExhaustive));
    assert_eq!(result.steps[0].best, "input");
    assert_eq!(result.row("run2").unwrap().rank, None);
    assert!(!result.row("run2").unwrap().passed);
    assert_eq!(result.final_id, "input");
    assert!(result.steps.iter().all(|s| s.kind != StepKind::Input));
    assert!(result.row("input").unwrap().selected);

    // Bottom-up from a derived base: the search's best is worse than the
    // input, so the input is returned with a note.
    let script = ScriptedFitter::new(
        key,
        &[
            ("[CL]+[KA]+[V]", 10.0),
            ("[CL]", 300.0),
            ("[CL]+[V]", 101.0),
        ],
    );
    let mut o = options(Algorithm::BottomUpStepwise);
    o.correlation_algorithm = Some(CorrelationAlgorithm::Skip);
    let result = run(&script, space(KEEP_CL), &o);
    let last = result.steps.last().unwrap();
    assert_eq!(last.kind, StepKind::Input);
    assert_eq!(last.best, "input");
    assert_eq!(result.final_id, "input");
    assert!(result
        .notes
        .iter()
        .any(|n| n.contains("the input is returned")));

    // An errored candidate is a row with its reason and no rank.
    let mut script = ScriptedFitter::new(key, &[("[CL]+[KA]+[V]", 100.0)]);
    script.erroring = vec!["run2".into()];
    let result = run(&script, space(ALL), &options(Algorithm::TopDownExhaustive));
    let r = result.row("run2").unwrap();
    assert!(r.error.is_some());
    assert_eq!(r.rank, None);
    assert!(r.ofv.is_none());
}

#[test]
fn a_new_block_is_seeded_from_the_parents_ebe_correlation() {
    let script = ScriptedFitter::new(key, &[("[CL]+[KA]+[V]", 100.0), ("[CL,KA,V]", 90.0)]);
    let mut o = options(Algorithm::Skip);
    o.correlation_algorithm = Some(CorrelationAlgorithm::TopDownExhaustive);
    let result = run(&script, space(ALL), &o);
    let run = result
        .rows
        .iter()
        .find(|r| r.structure.description() == "[CL,V]+[KA]")
        .unwrap()
        .id
        .clone();
    let model = script.model_of("step-1", &run);
    let line = model
        .block_lines("parameters")
        .into_iter()
        .find(|l| l.starts_with("block_omega (ETA_CL, ETA_V)"))
        .unwrap();
    let tri: Vec<f64> = line
        .split_once('[')
        .unwrap()
        .1
        .trim_end_matches(']')
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    // The fixture fit carries 32 subjects' η; their correlation is neither the
    // edit's flat 0.1 nor zero, and the block factors.
    let flat = 0.1 * (tri[0] * tri[2]).sqrt();
    assert!((tri[1] - flat).abs() > 1e-9, "{line}");
    assert!(tri[1] != 0.0, "{line}");
    assert!(
        tri[1].abs() <= 0.95 * (tri[0] * tri[2]).sqrt() + 1e-12,
        "{line}"
    );
    let m = nalgebra::DMatrix::from_row_slice(2, 2, &[tri[0], tri[1], tri[1], tri[2]]);
    assert!(nalgebra::Cholesky::new(m).is_some(), "{line}");

    // And the mutation: without the seed the block is the edit's flat 0.1.
    let fit = crate::search::test_support::converged_fit(1.0);
    let etas = vec!["ETA_CL".to_string(), "ETA_V".to_string()];
    let seeded = block_seed(&fit, std::slice::from_ref(&etas)).expect("the fixture has subjects");
    let (i, j) = (
        fit.eta_names.iter().position(|n| n == "ETA_CL").unwrap(),
        fit.eta_names.iter().position(|n| n == "ETA_V").unwrap(),
    );
    assert!(seeded.omega[(i, j)] != 0.0);
    assert_eq!(seeded.omega[(i, j)], seeded.omega[(j, i)]);
    let mut no_subjects = fit.clone();
    no_subjects.subjects.clear();
    assert!(block_seed(&no_subjects, &[etas]).is_none());
}

#[test]
fn cancellation_returns_partial_rows_and_the_parent_so_far() {
    let mut script = ScriptedFitter::new(key, &[("[CL]+[KA]+[V]", 100.0), ("[CL]+[V]", 90.0)]);
    script.cancel_after = Some(2);
    let result = run(&script, space(ALL), &options(Algorithm::TopDownExhaustive));
    assert!(result.cancelled);
    assert_eq!(script.dirs(), vec!["input", "step-1"]);
    assert_eq!(result.final_id, "run2", "the step's winner stands");
    assert_eq!(result.steps.len(), 2, "the step, then the input comparison");
}

// ── structure and report ────────────────────────────────────────────────────

#[test]
fn structures_describe_themselves_as_pharmpy_does() {
    let s = IivStructure::new(
        vec!["V".into(), "CL".into(), "KA".into()],
        vec![vec!["V".into(), "CL".into()]],
    );
    assert_eq!(s.etas, vec!["CL", "KA", "V"]);
    assert_eq!(s.description(), "[CL,V]+[KA]");
    assert_eq!(describe(&s), "[CL,V]+[KA]");
    assert_eq!(s.largest_block(), 2);
    assert_eq!(s.block_of("V").map(|b| b.len()), Some(2));
    assert_eq!(s.block_of("KA"), None);
    assert_eq!(s.feature_vector().get("blocks"), Some("[CL,V]"));
    assert_eq!(s.feature_vector().get("iiv"), Some("CL,KA,V"));
    let none = IivStructure::new(vec![], vec![]);
    assert_eq!(none.description(), "");
    assert_eq!(describe(&none), "no η");
    assert_eq!(none.largest_block(), 1);
    // A block is pruned to the η that exist; a 1-member block is no block.
    let s = IivStructure::new(vec!["CL".into()], vec![vec!["CL".into(), "V".into()]]);
    assert!(s.blocks.is_empty());
    assert_eq!(
        components(
            vec![
                ("A".to_string(), "B".to_string()),
                ("C".to_string(), "B".to_string())
            ]
            .into_iter()
        ),
        vec![vec!["A".to_string(), "B".to_string(), "C".to_string()]]
    );
    assert_eq!(
        combinations(&[1, 2, 3], 2),
        vec![vec![1, 2], vec![1, 3], vec![2, 3]]
    );
    assert_eq!(combinations(&[1, 2], 3), Vec::<Vec<i32>>::new());
    assert_eq!(
        default_dir(Path::new("runs/warfarin.ferxsearch")),
        Path::new("runs/warfarin-iivsearch")
    );
}

#[test]
fn the_report_writes_the_table_the_models_and_the_final_model() {
    let script = ScriptedFitter::new(
        key,
        &[
            ("[CL]+[KA]+[V]", 100.0),
            ("[CL]+[V]", 99.0),
            ("[CL,V]", 70.0),
        ],
    );
    let result = run(&script, space(ALL), &options(Algorithm::TopDownExhaustive));
    let dir = tempfile::tempdir().unwrap();
    write_report(dir.path(), &result).unwrap();
    let table = std::fs::read_to_string(models_path(dir.path())).unwrap();
    let header = table.lines().next().unwrap();
    assert_eq!(header, MODEL_COLUMNS.join(","));
    assert_eq!(table.lines().count(), 1 + result.rows.len());
    assert!(
        table.contains("run8,run2,2,\"[CL,V]\",CL;V,\"CL,V\""),
        "{table}"
    );
    assert!(models_dir(dir.path()).join("run8.ferx").exists());
    assert!(models_dir(dir.path()).join("input.ferx").exists());
    let final_text = std::fs::read_to_string(final_model_path(dir.path())).unwrap();
    assert!(
        final_text.contains("block_omega (ETA_CL, ETA_V)"),
        "{final_text}"
    );
    ModelText::parse(&final_text).unwrap();
    let summary = render_summary(&result);
    assert!(
        summary.contains("Step 1 (no_of_etas), parent input"),
        "{summary}"
    );
    assert!(
        summary.contains("Step 2 (block_structure), parent run2"),
        "{summary}"
    );
    assert!(summary.contains("Final model: run8 — [CL,V]"), "{summary}");
    assert!(summary.contains("SELECTED"), "{summary}");
}

#[test]
fn a_partial_block_candidate_under_focei_carries_the_1018_note() {
    let s = IivStructure::new(
        vec!["CL".into(), "V".into(), "KA".into()],
        vec![vec!["CL".into(), "V".into()]],
    );
    assert!(s.is_partial_block());
    let full = IivStructure::new(
        vec!["CL".into(), "V".into(), "KA".into()],
        vec![vec!["CL".into(), "V".into(), "KA".into()]],
    );
    assert!(!full.is_partial_block());
    let diagonal = IivStructure::new(vec!["CL".into(), "V".into()], vec![]);
    assert!(!diagonal.is_partial_block());
    let two = IivStructure::new(
        vec!["CL".into(), "V".into()],
        vec![vec!["CL".into(), "V".into()]],
    );
    assert!(!two.is_partial_block());

    let script = ScriptedFitter::new(key, &[("[CL]+[KA]+[V]", 100.0), ("[CL,V]+[KA]", 90.0)]);
    let mut o = options(Algorithm::Skip);
    o.correlation_algorithm = Some(CorrelationAlgorithm::TopDownExhaustive);
    let mut space = space(ALL);
    space.outer_full_triangle = true;
    let result = run(&script, space, &o);
    assert!(
        result.notes.iter().any(|n| n.contains("#1018")),
        "{:?}",
        result.notes
    );
    // Under an estimator that honours the structure there is nothing to say.
    let script = ScriptedFitter::new(key, &[("[CL]+[KA]+[V]", 100.0), ("[CL,V]+[KA]", 90.0)]);
    let result = run(&script, self::space(ALL), &o);
    assert!(!result.notes.iter().any(|n| n.contains("#1018")));
}
