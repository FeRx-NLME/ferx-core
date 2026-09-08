//! Tier-2 end-to-end checks for iivsearch (#1183), on a real model and
//! dataset but never to convergence: every fit is an evaluation
//! (`maxiter = 0`, ferx's `MAXEVAL=0`).
//!
//! The unit tests in `src/iivsearch/mod_tests.rs` script the fitter, so they
//! test the enumeration and never compile a candidate. This file is the
//! other half: from a `.ferxsearch` file through the η / block edits, the
//! runner, `fit()` and the files on disk. Three of the issue's validation
//! bullets live here:
//!
//! * the **degenerate oracle** — a single-point space returns the base fit
//!   *bit for bit*;
//! * **every generated candidate compiles and evaluates** — drops, adds,
//!   blocks and splits, on every algorithm;
//! * a generated candidate **is the hand-written model** — the
//!   `tests/covariate_model_equivalence.rs` pattern: a bit-identical
//!   evaluation against the model a pharmacometrician would have typed.
//!
//! And the fourth: a model outside the canonical form is refused **by
//! name** before anything is fitted.

use std::path::{Path, PathBuf};

use ferx_core::fit;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_tools::iivsearch::{run_iivsearch, IivsearchResult, IivsearchRun};
use ferx_tools::search::SearchConfig;

const DATA: &str = "../../data/warfarin.csv";

/// The warfarin one-compartment oral model, evaluating rather than fitting.
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
  method     = foce
  maxiter    = 0
  covariance = false
  checkpoint = false
";

fn write_config(dir: &Path, base: &str, mfl: &str, extra: &str) -> PathBuf {
    std::fs::write(dir.join("base.ferx"), base).unwrap();
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join(DATA);
    let config = format!(
        "base = \"base.ferx\"\ndata = \"{}\"\n\n[space]\nmfl = \"{mfl}\"\n\n\
         [strictness]\nrequire_converged = false\nreject_init_stall = false\n\
         reject_on_boundary = false\n\n[run]\nretries = 0\nthreads = 2\n{extra}",
        data.display()
    );
    let path = dir.join("search.ferxsearch");
    std::fs::write(&path, config).unwrap();
    path
}

fn run(dir: &Path, path: &Path) -> (SearchConfig, IivsearchResult) {
    let config = SearchConfig::load(path).unwrap();
    let base = config.load_base().unwrap();
    let result = run_iivsearch(
        &config,
        &base,
        IivsearchRun {
            dir: Some(dir.join("run")),
            ..IivsearchRun::default()
        },
    )
    .expect("search");
    (config, result)
}

fn evaluate(text: &str, base: &ferx_tools::search::BaseModel) -> ferx_core::FitResult {
    let parsed = parse_full_model(text).expect("the model parses");
    let mut o = parsed.fit_options.clone().quiet();
    o.threads = Some(2);
    fit(
        &parsed.model,
        &base.prepared.population,
        &parsed.model.default_params,
        &o,
    )
    .expect("evaluation")
}

#[test]
fn a_single_point_space_returns_the_input_fit_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();
    // Every η kept, no covariance to search: nothing can move.
    let path = write_config(
        dir.path(),
        BASE,
        "IIV([CL,V,KA],EXP)",
        "[iivsearch]\ncorrelation_algorithm = \"skip\"\n",
    );
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    let direct = fit(
        &base.prepared.parsed.model,
        &base.prepared.population,
        &base.prepared.init_params,
        &{
            let mut o = base.prepared.parsed.fit_options.clone().quiet();
            o.threads = Some(2);
            o
        },
    )
    .expect("direct evaluation");

    let result = run_iivsearch(
        &config,
        &base,
        IivsearchRun {
            dir: Some(dir.path().join("run")),
            ..IivsearchRun::default()
        },
    )
    .expect("search");
    assert_eq!(result.rows.len(), 1, "{:?}", result.rows);
    assert_eq!(result.final_id, "input");
    assert_eq!(result.base_id, "input");
    assert!(result.steps.is_empty(), "{:?}", result.steps);
    let final_fit = result.final_fit.as_ref().expect("the input fit");
    assert_eq!(
        final_fit.ofv.to_bits(),
        direct.ofv.to_bits(),
        "search OFV {} vs direct {}",
        final_fit.ofv,
        direct.ofv
    );
    assert_eq!(result.final_model.render(), base.text.render());
    assert_eq!(result.final_structure.description(), "[CL]+[KA]+[V]");
    let table = std::fs::read_to_string(dir.path().join("run/models.csv")).unwrap();
    assert_eq!(table.lines().count(), 2);
    assert!(dir.path().join("run/models/input.ferx").exists());
    // `final.ferx` re-evaluates to the same objective (15-digit inits, see
    // the modelsearch twin of this test for the bound's reasoning).
    let final_text = std::fs::read_to_string(dir.path().join("run/final.ferx")).unwrap();
    let again = evaluate(&final_text, &base);
    let gap = (again.ofv - direct.ofv).abs();
    eprintln!("final.ferx re-evaluation |ΔOFV| = {gap:.3e}");
    assert!(
        gap < 1e-9,
        "final.ferx OFV {} vs direct {}",
        again.ofv,
        direct.ofv
    );
}

#[test]
fn every_candidate_of_every_algorithm_compiles_and_evaluates() {
    for (algorithm, mfl, expect) in [
        (
            "top_down_exhaustive",
            "IIV?([CL,V,KA],EXP);COVARIANCE?(IIV,[CL,V,KA])",
            // 7 η subsets, then the blocks over whatever step 1 kept.
            8,
        ),
        (
            "bottom_up_stepwise",
            "IIV(CL,EXP);IIV?([V,KA],EXP);COVARIANCE?(IIV,[CL,V,KA])",
            4,
        ),
        (
            "simultaneous_stepwise",
            "IIV(CL,EXP);IIV?([V,KA],EXP);COVARIANCE?(IIV,[CL,V,KA])",
            6,
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = write_config(
            dir.path(),
            BASE,
            mfl,
            &format!("[iivsearch]\nalgorithm = \"{algorithm}\"\n"),
        );
        let (_, result) = run(dir.path(), &path);
        let failed: Vec<String> = result
            .rows
            .iter()
            .filter(|r| r.error.is_some() || r.ofv.is_none())
            .map(|r| format!("{} ({}): {:?}", r.id, r.structure.description(), r.error))
            .collect();
        assert!(failed.is_empty(), "{algorithm}: {failed:?}");
        assert!(
            result.rows.iter().filter(|r| r.step > 0).count() >= expect,
            "{algorithm}: {} candidates fitted: {:?}",
            result.rows.len(),
            result
                .rows
                .iter()
                .map(|r| r.structure.description())
                .collect::<Vec<_>>()
        );
        for r in &result.rows {
            assert!(r.ofv.unwrap().is_finite(), "{}: {:?}", algorithm, r);
            let text = &result.models[&r.id];
            parse_full_model(&text.render())
                .unwrap_or_else(|e| panic!("{algorithm} {}: {e}\n{}", r.id, text.render()));
        }
        // The number of free parameters follows the structure: one per η,
        // one per covariance.
        for r in &result.rows {
            let n_cov: usize = r
                .structure
                .blocks
                .iter()
                .map(|b| b.len() * (b.len() - 1) / 2)
                .sum();
            assert_eq!(
                r.n_parameters.unwrap(),
                4 + r.structure.etas.len() + n_cov,
                "{algorithm} {} ({})",
                r.id,
                r.structure.description()
            );
        }
        eprintln!(
            "{algorithm}: final {} ({})",
            result.final_id,
            result.final_structure.description()
        );
    }
}

#[test]
fn a_generated_candidate_is_the_hand_written_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        BASE,
        "IIV?([CL,V,KA],EXP)",
        "[iivsearch]\ncorrelation_algorithm = \"skip\"\n",
    );
    let (config, result) = run(dir.path(), &path);
    let base = config.load_base().unwrap();
    let dropped_ka = result
        .rows
        .iter()
        .find(|r| r.structure.description() == "[CL]+[V]")
        .expect("the candidate without KA's η");
    let generated = result.models[&dropped_ka.id].render();
    // What a pharmacometrician types for "no variability on KA": the line
    // without its factor, the ω line gone, everything else as it was. The
    // candidate was seeded from the input's *evaluation*, whose estimates
    // are the inits themselves, so the twin carries the file's numbers.
    let hand_written = BASE
        .replace("  KA = TVKA * exp(ETA_KA)\n", "  KA = TVKA\n")
        .replace("  omega ETA_KA ~ 0.30\n", "");
    let ours = evaluate(&generated, &base);
    let theirs = evaluate(&hand_written, &base);
    assert_eq!(
        ours.ofv.to_bits(),
        theirs.ofv.to_bits(),
        "generated {} vs hand-written {}\n{generated}",
        ours.ofv,
        theirs.ofv
    );
    assert_eq!(ours.n_parameters, theirs.n_parameters);
    assert_eq!(ours.eta_names, theirs.eta_names);
    assert_eq!(dropped_ka.ofv, Some(ours.ofv));
}

#[test]
fn a_block_candidate_is_the_hand_written_block_seeded_from_the_ebes() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        BASE,
        "COVARIANCE?(IIV,[CL,V])",
        "[iivsearch]\nalgorithm = \"skip\"\ncorrelation_algorithm = \"top_down_exhaustive\"\n",
    );
    let (config, result) = run(dir.path(), &path);
    let base = config.load_base().unwrap();
    let blocked = result
        .rows
        .iter()
        .find(|r| r.structure.description() == "[CL,V]+[KA]")
        .expect("the [CL,V] block candidate");
    let generated = result.models[&blocked.id].render();
    // The block's off-diagonal is the input evaluation's EBE correlation
    // times the geometric mean of the variances — read the same number off
    // the same fit, the way the search does.
    let input = evaluate(BASE, &base);
    let (i, j) = (
        input.eta_names.iter().position(|n| n == "ETA_CL").unwrap(),
        input.eta_names.iter().position(|n| n == "ETA_V").unwrap(),
    );
    let n = input.subjects.len() as f64;
    let mean = |k: usize| input.subjects.iter().map(|s| s.eta[k]).sum::<f64>() / n;
    let (mi, mj) = (mean(i), mean(j));
    let cov = |a: usize, b: usize, ma: f64, mb: f64| {
        input
            .subjects
            .iter()
            .map(|s| (s.eta[a] - ma) * (s.eta[b] - mb))
            .sum::<f64>()
    };
    let corr =
        (cov(i, j, mi, mj) / (cov(i, i, mi, mi) * cov(j, j, mj, mj)).sqrt()).clamp(-0.95, 0.95);
    let off = corr * (0.09f64 * 0.04).sqrt();
    let line = generated
        .lines()
        .find(|l| l.trim().starts_with("block_omega"))
        .expect("the block line");
    let written: f64 = line.split(',').nth(2).unwrap().trim().parse().unwrap();
    assert!(
        (written - off).abs() < 1e-12,
        "block off-diagonal {written} vs EBE-derived {off}: {line}"
    );
    let hand_written = BASE.replace("  omega ETA_CL ~ 0.09\n", "").replace(
        "  omega ETA_V ~ 0.04\n",
        &format!("  block_omega (ETA_CL, ETA_V) = [0.09, {off:.15e}, 0.04]\n"),
    );
    let ours = evaluate(&generated, &base);
    let theirs = evaluate(&hand_written, &base);
    // Two spellings of one number (`{:.15e}` versus the edit's 15
    // significant digits) can differ in the last ULP; the objectives agree
    // to well inside anything a search decides on.
    let gap = (ours.ofv - theirs.ofv).abs();
    eprintln!("block twin |ΔOFV| = {gap:.3e}");
    assert!(
        gap < 1e-9,
        "generated {} vs hand-written {}",
        ours.ofv,
        theirs.ofv
    );
    assert_eq!(ours.n_parameters, theirs.n_parameters);
    assert_eq!(blocked.starts, 1, "a 2-block gets the run's starts");
}

#[test]
fn a_model_outside_the_canonical_form_is_refused_by_name_before_any_fit() {
    let dir = tempfile::tempdir().unwrap();
    let proportional = BASE.replace("  V = TVV * exp(ETA_V)", "  V = TVV * (1 + ETA_V)");
    let path = write_config(dir.path(), &proportional, "IIV?([CL,V,KA],EXP)", "");
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    let err = run_iivsearch(
        &config,
        &base,
        IivsearchRun {
            dir: Some(dir.path().join("run")),
            ..IivsearchRun::default()
        },
    )
    .expect_err("a proportional η is not searchable");
    assert!(err.contains("`V`"), "{err}");
    assert!(err.contains("canonical form"), "{err}");
    assert!(err.contains("`ETA_V`"), "{err}");
    assert!(
        !dir.path().join("run/input").exists(),
        "refused before the input was fitted"
    );
    // The same model is fine when `V` is left out of the space.
    let path = write_config(
        dir.path(),
        &proportional,
        "IIV?([CL,KA],EXP)",
        "[iivsearch]\ncorrelation_algorithm = \"skip\"\n",
    );
    let (_, result) = run(dir.path(), &path);
    assert!(result.rows.len() > 1);
    assert!(result.rows.iter().all(|r| r.error.is_none()));
}
