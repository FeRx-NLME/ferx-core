//! Tier-1 tests for the inter-occasion variability search logic (#1183).
//!
//! Every test drives [`search`] with a *scripted* fitter — OFVs keyed by the
//! candidate's structure description or by id — so what is under test is
//! the two-step brute force: which κ subsets and η subsets are tried, what
//! each step ranks and picks, what the κ distributions write, and what the
//! report says. The path through the runner and `fit()` is
//! `tests/iovsearch_end_to_end.rs`; the trajectory against Pharmpy's own run
//! is `tests/iovsearch_pharmpy_anchor.rs`.

use std::path::Path;

use ferx_core::edit::ModelText;
use ferx_core::{DoseEvent, Subject};

use super::*;
use crate::search::mfl::Mfl;
use crate::search::test_support::ScriptedFitter;
use crate::search::SearchConfig;

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

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  iov_column = OCC
";

/// Two subjects over two occasions.
fn population() -> Population {
    let subj = |id: &str| Subject {
        id: id.into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![1.0, 4.0, 25.0, 28.0],
        observations: vec![10.0, 8.0, 9.0, 6.0],
        obs_cmts: vec![1; 4],
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        ..Default::default()
    };
    Population {
        subjects: vec![subj("1"), subj("2")],
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// `IIV(...);IOV(...)` read off the candidate's feature vector — the key the
/// scripted OFV table uses. Blocks are not part of the key.
fn key(c: &Candidate) -> String {
    let list = |k: &str| -> String {
        c.features
            .get(k)
            .unwrap_or("")
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|p| format!("[{p}]"))
            .collect::<Vec<_>>()
            .join("+")
    };
    let wrap = |s: String| if s.is_empty() { "[]".to_string() } else { s };
    format!("IIV({});IOV({})", wrap(list("iiv")), wrap(list("iov")))
}

fn options() -> IovsearchOptions {
    IovsearchOptions {
        rank: RankType::Ofv,
        ..IovsearchOptions::default()
    }
}

fn space_of(base: &str, mfl: Option<&str>, options: &IovsearchOptions) -> Space {
    let text = ModelText::parse(base).unwrap();
    let features: Vec<Feature> = mfl
        .map(|m| Mfl::parse(m).unwrap().features().cloned().collect())
        .unwrap_or_default();
    Space::build(
        text,
        &features,
        Some("OCC"),
        &population(),
        options,
        Vec::new(),
    )
    .expect("a valid space")
}

fn run(script: &ScriptedFitter, space: Space, options: &IovsearchOptions) -> IovsearchResult {
    search(script, space, options, None).expect("search")
}

fn fitted_in(result: &IovsearchResult, step: usize) -> Vec<(String, String)> {
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
fn options_take_pharmpys_defaults_and_read_bic_as_the_bic_random() {
    let o = IovsearchOptions::from_config(&config("base = \"b.ferx\"\n").unwrap()).unwrap();
    assert_eq!(o.distribution, Distribution::SameAsIiv);
    assert_eq!(o.column, None);
    assert!(o.groups.is_empty());
    assert_eq!(o.rank, RankType::BicRandom);
    assert_eq!(o.criterion(), Criterion::Bic(ferx_core::BicType::Random));
    assert_eq!(o.block_retries, 2);
    assert_eq!(o.starts, 3);
    assert_eq!(o.starts_for(3), 5);
    let cfg = config("base = \"b.ferx\"\n[rank]\ntype = \"bic\"\n").unwrap();
    assert_eq!(
        IovsearchOptions::from_config(&cfg).unwrap().rank,
        RankType::BicRandom
    );
    let cfg = config(
        "base = \"b.ferx\"\n[rank]\ntype = \"bic_mixed\"\n[iovsearch]\ndistribution = \
         \"explicit\"\ngroups = [[\"CL\", \"V\"], [\"KA\"]]\ncolumn = \"OCC\"\n",
    )
    .unwrap();
    let o = IovsearchOptions::from_config(&cfg).unwrap();
    assert_eq!(o.rank, RankType::BicMixed);
    assert_eq!(o.distribution, Distribution::Explicit);
    assert_eq!(o.groups.len(), 2);
    assert_eq!(o.column.as_deref(), Some("OCC"));

    let refuse = |section: &str| {
        config(&format!("base = \"b.ferx\"\n[iovsearch]\n{section}\n"))
            .and_then(|c| IovsearchOptions::from_config(&c))
            .unwrap_err()
    };
    assert!(refuse("distribution = \"explicit\"").contains("needs `groups`"));
    assert!(refuse("groups = [[\"CL\"]]").contains("explicit"));
    assert!(
        refuse("distribution = \"explicit\"\ngroups = [[\"CL\"], [\"CL\"]]").contains("two groups")
    );
    assert!(refuse("distribution = \"explicit\"\ngroups = [[]]").contains("empty group"));
    assert!(refuse("distribution = \"nope\"").contains("distribution"));
}

// ── the space ───────────────────────────────────────────────────────────────

#[test]
fn the_space_defaults_to_every_free_eta_and_checks_the_occasions() {
    let s = space_of(BASE, None, &options());
    assert_eq!(
        s.params,
        vec![
            ("CL".to_string(), false),
            ("KA".to_string(), false),
            ("V".to_string(), false)
        ]
    );
    assert_eq!(
        s.input_structure.description(),
        "IIV([CL]+[KA]+[V]);IOV([])"
    );
    assert_eq!(
        s.names["CL"],
        (Some("ETA_CL".to_string()), "KAPPA_CL".to_string())
    );
    assert_eq!(s.free_eta, vec!["CL", "KA", "V"]);
    assert!(s.groups.is_empty(), "same-as-iiv on diagonal η: no κ block");

    // A fixed η is not a candidate; an explicit space names them.
    let fixed = BASE.replace("omega ETA_KA ~ 0.30", "omega ETA_KA ~ 0.30 FIX");
    let s = space_of(&fixed, None, &options());
    assert_eq!(s.searched(), vec!["CL", "V"]);
    let s = space_of(BASE, Some("IOV(CL,EXP);IOV?([V],EXP)"), &options());
    assert_eq!(s.kept(), vec!["CL"]);
    assert_eq!(s.searched(), vec!["V"]);

    // A parameter that already carries a κ is left alone, with a note.
    let with_kappa = BASE
        .replace(
            "  CL = TVCL * exp(ETA_CL)",
            "  CL = TVCL * exp(ETA_CL + KAPPA_CL)",
        )
        .replace(
            "  sigma PROP_ERR",
            "  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR",
        );
    let s = space_of(&with_kappa, None, &options());
    assert_eq!(s.searched(), vec!["KA", "V"]);
    assert!(s.notes.iter().any(|n| n.contains("already carries a κ")));
    assert_eq!(
        s.input_structure.description(),
        "IIV([CL]+[KA]+[V]);IOV([CL])"
    );

    // The base must read its occasions itself, and they must vary.
    let text = ModelText::parse(BASE).unwrap();
    let err = Space::build(text.clone(), &[], None, &population(), &options(), vec![]).unwrap_err();
    assert!(err.contains("iov_column"), "{err}");
    let mut wrong = options();
    wrong.column = Some("VISIT".into());
    let err = Space::build(
        text.clone(),
        &[],
        Some("OCC"),
        &population(),
        &wrong,
        vec![],
    )
    .unwrap_err();
    assert!(err.contains("must agree"), "{err}");
    let mut one = population();
    for s in &mut one.subjects {
        s.occasions = vec![1; 4];
    }
    let err = Space::build(text.clone(), &[], Some("OCC"), &one, &options(), vec![]).unwrap_err();
    assert!(err.contains("at least two"), "{err}");

    // What cannot be searched is refused by name.
    let refuse = |base: &str, mfl: &str| {
        let features: Vec<Feature> = Mfl::parse(mfl).unwrap().features().cloned().collect();
        Space::build(
            ModelText::parse(base).unwrap(),
            &features,
            Some("OCC"),
            &population(),
            &options(),
            vec![],
        )
        .unwrap_err()
    };
    let err = refuse(
        &BASE.replace("  V = TVV * exp(ETA_V)", "  V = TVV * (1 + ETA_V)"),
        "IOV?([CL,V],EXP)",
    );
    assert!(
        err.contains("`V`") && err.contains("canonical form"),
        "{err}"
    );
    assert!(refuse(BASE, "IOV?([CL],ADD)").contains("exponential form"));
    assert!(refuse(BASE, "IOV?([NOPE],EXP)").contains("`NOPE`"));
    assert!(refuse(BASE, "IIV?([CL],EXP)").contains("not an IOV feature"));
    assert!(refuse(BASE, "IOV?([CL,V],EXP);COVARIANCE?(IOV,[CL,V])").contains("not searched"));
    assert!(refuse(BASE, "IOV?([CL,V],EXP);COVARIANCE(IIV,[CL,V])").contains("η covariance"));
    // An explicit block via a plain COVARIANCE(IOV, …).
    let s = space_of(
        BASE,
        Some("IOV?([CL,V,KA],EXP);COVARIANCE(IOV,[CL,V])"),
        &options(),
    );
    assert_eq!(s.groups, vec![vec!["CL".to_string(), "V".to_string()]]);
}

// ── the search ──────────────────────────────────────────────────────────────

#[test]
fn step_one_tries_the_full_model_and_every_proper_subset_then_step_two_the_etas() {
    let script = ScriptedFitter::new(
        key,
        &[
            ("IIV([CL]+[KA]+[V]);IOV([])", 200.0),
            ("IIV([CL]+[KA]+[V]);IOV([CL]+[KA]+[V])", 110.0),
            ("IIV([CL]+[KA]+[V]);IOV([CL]+[V])", 105.0),
            ("IIV([CL]+[KA]+[V]);IOV([CL])", 100.0),
            ("IIV([KA]+[V]);IOV([CL])", 130.0),
        ],
    );
    let result = run(&script, space_of(BASE, None, &options()), &options());
    assert_eq!(script.dirs(), vec!["input", "iov-all", "step-1", "step-2"]);
    // run1 is the full-IOV model; then the removed subsets, smallest first,
    // lexicographic over the alphabetical κ — Pharmpy's numbering.
    assert_eq!(
        fitted_in(&result, 1),
        vec![
            pair("run1", "IIV([CL]+[KA]+[V]);IOV([CL]+[KA]+[V])"),
            pair("run2", "IIV([CL]+[KA]+[V]);IOV([KA]+[V])"),
            pair("run3", "IIV([CL]+[KA]+[V]);IOV([CL]+[V])"),
            pair("run4", "IIV([CL]+[KA]+[V]);IOV([CL]+[KA])"),
            pair("run5", "IIV([CL]+[KA]+[V]);IOV([V])"),
            pair("run6", "IIV([CL]+[KA]+[V]);IOV([KA])"),
            pair("run7", "IIV([CL]+[KA]+[V]);IOV([CL])"),
        ]
    );
    // The full model writes each κ beside its η at a tenth of the fitted
    // variance (the fixture's ω are the warfarin inits), diagonal.
    let run1 = script.model_of("iov-all", "run1");
    let lines = run1.block_lines("individual_parameters");
    assert!(
        lines.contains(&"CL = TVCL * exp(ETA_CL + KAPPA_CL)".to_string()),
        "{lines:?}"
    );
    assert!(
        lines.contains(&"KA = TVKA * exp(ETA_KA + KAPPA_KA)".to_string()),
        "{lines:?}"
    );
    let params = run1.block_lines("parameters");
    assert!(
        params.contains(&"kappa KAPPA_CL ~ 0.009".to_string()),
        "{params:?}"
    );
    assert!(
        params.contains(&"kappa KAPPA_V ~ 0.004".to_string()),
        "{params:?}"
    );
    assert!(
        params.contains(&"kappa KAPPA_KA ~ 0.03".to_string()),
        "{params:?}"
    );
    assert!(!run1.render().contains("block_kappa"));
    // A candidate derived from the full model has its parent's κ dropped.
    let run7 = script.model_of("step-1", "run7").render();
    assert!(
        !run7.contains("KAPPA_KA") && !run7.contains("KAPPA_V"),
        "{run7}"
    );
    assert!(run7.contains("kappa KAPPA_CL"), "{run7}");
    // Step 1 ranks the input, the full model and the candidates together.
    assert_eq!(result.steps[0].parent, "input");
    assert_eq!(result.steps[0].best, "run7");
    let ids: Vec<&str> = result.steps[0]
        .ranked
        .iter()
        .map(|r| r.id.as_str())
        .collect();
    assert_eq!(&ids[..4], &["run7", "run3", "run1", "input"]);
    assert_eq!(result.row("run7").unwrap().rank, Some(1));
    assert_eq!(result.row("run7").unwrap().d_criterion, Some(-100.0));
    // Step 2 from run7: the one η with a κ beside it, removed.
    assert_eq!(
        fitted_in(&result, 2),
        vec![pair("run8", "IIV([KA]+[V]);IOV([CL])")]
    );
    let run8 = script.model_of("step-2", "run8");
    assert!(run8
        .block_lines("individual_parameters")
        .contains(&"CL = TVCL * exp(KAPPA_CL)".to_string()));
    assert!(!run8.render().contains("ETA_CL"));
    assert_eq!(result.steps[1].parent, "run7");
    assert_eq!(result.steps[1].best, "run7");
    assert_eq!(result.final_id, "run7");
    assert_eq!(
        result.final_structure.description(),
        "IIV([CL]+[KA]+[V]);IOV([CL])"
    );
    assert!(result.row("run7").unwrap().selected);
    assert!(!result.cancelled);
}

#[test]
fn the_input_is_returned_when_no_iov_candidate_beats_it_and_a_kept_kappa_stays() {
    let script = ScriptedFitter::new(key, &[("IIV([CL]+[KA]+[V]);IOV([])", 10.0)]);
    let result = run(&script, space_of(BASE, None, &options()), &options());
    assert_eq!(result.final_id, "input");
    assert_eq!(result.steps.len(), 1);
    assert!(result
        .notes
        .iter()
        .any(|n| n.contains("the input is returned")));
    assert_eq!(script.dirs(), vec!["input", "iov-all", "step-1"]);

    // With one κ kept and one searched there is no proper subset to remove:
    // the full model is the only IOV candidate.
    let script = ScriptedFitter::new(
        key,
        &[
            ("IIV([CL]+[KA]+[V]);IOV([])", 200.0),
            ("IIV([CL]+[KA]+[V]);IOV([CL]+[V])", 100.0),
        ],
    );
    let result = run(
        &script,
        space_of(BASE, Some("IOV(CL,EXP);IOV?([V],EXP)"), &options()),
        &options(),
    );
    assert_eq!(
        fitted_in(&result, 1),
        vec![pair("run1", "IIV([CL]+[KA]+[V]);IOV([CL]+[V])")]
    );
    assert_eq!(result.steps[0].best, "run1");
    // Step 2: every subset of {CL, V}'s η.
    assert_eq!(
        fitted_in(&result, 2),
        vec![
            pair("run2", "IIV([KA]+[V]);IOV([CL]+[V])"),
            pair("run3", "IIV([CL]+[KA]);IOV([CL]+[V])"),
            pair("run4", "IIV([KA]);IOV([CL]+[V])"),
        ]
    );
    assert_eq!(result.final_id, "run1");
}

#[test]
fn distributions_write_the_kappa_blocks_and_scale_the_starts() {
    let joint = IovsearchOptions {
        distribution: Distribution::Joint,
        ..options()
    };
    let script = ScriptedFitter::new(key, &[]);
    run(&script, space_of(BASE, None, &joint), &joint);
    let run1 = script.model_of("iov-all", "run1");
    assert!(
        run1.block_lines("parameters")
            .iter()
            .any(|l| l.starts_with("block_kappa (KAPPA_CL, KAPPA_KA, KAPPA_V)")),
        "{}",
        run1.render()
    );
    let c = script.candidates_in("iov-all").into_iter().next().unwrap();
    assert_eq!(
        c.n_starts,
        Some(5),
        "a 3-block gets block_retries more starts"
    );
    // A candidate with one κ left has no block and the run's starts.
    let run7 = script
        .candidates_in("step-1")
        .into_iter()
        .find(|c| c.id == "run7")
        .unwrap();
    assert_eq!(run7.n_starts, Some(3));
    assert!(run7
        .model
        .block_lines("parameters")
        .contains(&"kappa KAPPA_CL ~ 0.009".to_string()));

    // same-as-iiv mirrors an η block.
    let blocked = BASE.replace("  omega ETA_CL ~ 0.09\n", "").replace(
        "  omega ETA_V ~ 0.04\n",
        "  block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04]\n",
    );
    let script = ScriptedFitter::new(key, &[]);
    run(&script, space_of(&blocked, None, &options()), &options());
    let run1 = script.model_of("iov-all", "run1");
    let params = run1.block_lines("parameters");
    assert!(
        params
            .iter()
            .any(|l| l.starts_with("block_kappa (KAPPA_CL, KAPPA_V)")),
        "{params:?}"
    );
    assert!(
        params.contains(&"kappa KAPPA_KA ~ 0.03".to_string()),
        "{params:?}"
    );
    assert_eq!(
        script.candidates_in("iov-all")[0]
            .features
            .get("iov_blocks"),
        Some("[CL,V]")
    );

    // explicit groups.
    let explicit = IovsearchOptions {
        distribution: Distribution::Explicit,
        groups: vec![vec!["KA".into(), "V".into()]],
        ..options()
    };
    let script = ScriptedFitter::new(key, &[]);
    run(&script, space_of(BASE, None, &explicit), &explicit);
    let run1 = script.model_of("iov-all", "run1");
    assert!(run1
        .block_lines("parameters")
        .iter()
        .any(|l| l.starts_with("block_kappa (KAPPA_KA, KAPPA_V)")));
    let err = Space::build(
        ModelText::parse(BASE).unwrap(),
        &[],
        Some("OCC"),
        &population(),
        &IovsearchOptions {
            distribution: Distribution::Explicit,
            groups: vec![vec!["NOPE".into(), "V".into()]],
            ..options()
        },
        vec![],
    )
    .unwrap_err();
    assert!(err.contains("`NOPE`"), "{err}");
}

#[test]
fn a_gated_candidate_cannot_win_and_cancellation_keeps_the_rows_so_far() {
    let mut script = ScriptedFitter::new(
        key,
        &[
            ("IIV([CL]+[KA]+[V]);IOV([])", 200.0),
            ("IIV([CL]+[KA]+[V]);IOV([CL])", 100.0),
            ("IIV([CL]+[KA]+[V]);IOV([CL]+[V])", 105.0),
        ],
    );
    script.failing = vec!["IIV([CL]+[KA]+[V]);IOV([CL])".into()];
    let result = run(&script, space_of(BASE, None, &options()), &options());
    assert_eq!(result.steps[0].best, "run3");
    assert_eq!(result.row("run7").unwrap().rank, None);
    assert_eq!(result.final_id, "run3");

    let mut script = ScriptedFitter::new(key, &[("IIV([CL]+[KA]+[V]);IOV([CL])", 100.0)]);
    script.cancel_after = Some(3);
    let result = run(&script, space_of(BASE, None, &options()), &options());
    assert!(result.cancelled);
    assert_eq!(script.dirs(), vec!["input", "iov-all", "step-1"]);
    assert_eq!(result.final_id, "run7");
}

#[test]
fn the_report_writes_the_table_the_models_and_the_final_model() {
    let script = ScriptedFitter::new(
        key,
        &[
            ("IIV([CL]+[KA]+[V]);IOV([])", 200.0),
            ("IIV([CL]+[KA]+[V]);IOV([CL])", 100.0),
        ],
    );
    let result = run(&script, space_of(BASE, None, &options()), &options());
    let dir = tempfile::tempdir().unwrap();
    write_report(dir.path(), &result).unwrap();
    let table = std::fs::read_to_string(models_path(dir.path())).unwrap();
    assert_eq!(table.lines().next().unwrap(), MODEL_COLUMNS.join(","));
    assert_eq!(table.lines().count(), 1 + result.rows.len());
    assert!(
        table.contains("run7,run1,1,IIV([CL]+[KA]+[V]);IOV([CL]),CL;KA;V,CL,"),
        "{table}"
    );
    assert!(models_dir(dir.path()).join("run7.ferx").exists());
    let final_text = std::fs::read_to_string(final_model_path(dir.path())).unwrap();
    assert!(final_text.contains("kappa KAPPA_CL"), "{final_text}");
    ModelText::parse(&final_text).unwrap();
    let summary = render_summary(&result);
    assert!(summary.contains("Step 1 (IOV), parent input"), "{summary}");
    assert!(summary.contains("Step 2 (IIV), parent run7"), "{summary}");
    assert!(
        summary.contains("Final model: run7 — IIV([CL]+[KA]+[V]);IOV([CL])"),
        "{summary}"
    );
    assert_eq!(
        default_dir(Path::new("runs/warfarin.ferxsearch")),
        Path::new("runs/warfarin-iovsearch")
    );
}

#[test]
fn a_kappa_the_input_already_carries_stays_in_every_candidate() {
    let with_kappa = BASE
        .replace(
            "  CL = TVCL * exp(ETA_CL)",
            "  CL = TVCL * exp(ETA_CL + KAPPA_CL)",
        )
        .replace(
            "  sigma PROP_ERR",
            "  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR",
        );
    let script = ScriptedFitter::new(
        key,
        &[
            ("IIV([CL]+[KA]+[V]);IOV([CL])", 200.0),
            ("IIV([CL]+[KA]+[V]);IOV([CL]+[V])", 100.0),
        ],
    );
    let result = run(&script, space_of(&with_kappa, None, &options()), &options());
    assert_eq!(
        fitted_in(&result, 1),
        vec![
            pair("run1", "IIV([CL]+[KA]+[V]);IOV([CL]+[KA]+[V])"),
            pair("run2", "IIV([CL]+[KA]+[V]);IOV([CL]+[V])"),
            pair("run3", "IIV([CL]+[KA]+[V]);IOV([CL]+[KA])"),
        ]
    );
    // The mutation this pins: dropping the existing κ along with the
    // searched ones, which the first version did.
    for id in ["run1", "run2", "run3"] {
        let text = result.models[id].render();
        assert!(text.contains("kappa KAPPA_CL ~ 0.01"), "{id}: {text}");
        assert!(
            text.contains("CL = TVCL * exp(ETA_CL + KAPPA_CL)"),
            "{id}: {text}"
        );
    }
    assert_eq!(result.final_id, "run2");
}
