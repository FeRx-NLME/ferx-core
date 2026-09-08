//! Tier-2 end-to-end checks for iovsearch (#1183), on the warfarin
//! two-occasion dataset but never to convergence: every fit is an
//! evaluation (`maxiter = 0`).
//!
//! The unit tests in `src/iovsearch/mod_tests.rs` script the fitter; this
//! file is the path from a `.ferxsearch` file through the κ / η edits, the
//! runner, `fit()` and the files on disk:
//!
//! * the **degenerate oracle** — the input row is the direct evaluation
//!   *bit for bit*, and with one candidate parameter the search fits
//!   exactly the input and the full-IOV model;
//! * **every candidate compiles and evaluates**, for each κ distribution;
//! * the full-IOV candidate **is the hand-written model** — the base with
//!   `kappa KAPPA_CL` declared and `exp(ETA_CL + KAPPA_CL)` on the line, a
//!   bit-identical evaluation;
//! * a base without `iov_column` is refused with the fix named.

use std::path::{Path, PathBuf};

use ferx_core::fit;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_tools::iovsearch::{run_iovsearch, IovsearchResult, IovsearchRun};
use ferx_tools::search::SearchConfig;

const DATA: &str = "../../data/warfarin_iov.csv";

/// The warfarin model over two occasions, IIV only, reading `OCC`.
const BASE: &str = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.2 (sd)

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
  iov_column = OCC
";

fn write_config(dir: &Path, base: &str, space: Option<&str>, extra: &str) -> PathBuf {
    std::fs::write(dir.join("base.ferx"), base).unwrap();
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join(DATA);
    let space = space
        .map(|m| format!("[space]\nmfl = \"{m}\"\n\n"))
        .unwrap_or_default();
    let config = format!(
        "base = \"base.ferx\"\ndata = \"{}\"\n\n{space}\
         [strictness]\nrequire_converged = false\nreject_init_stall = false\n\
         reject_on_boundary = false\n\n[run]\nretries = 0\nthreads = 2\n{extra}",
        data.display()
    );
    let path = dir.join("search.ferxsearch");
    std::fs::write(&path, config).unwrap();
    path
}

fn run(dir: &Path, path: &Path) -> (SearchConfig, IovsearchResult) {
    let config = SearchConfig::load(path).unwrap();
    let base = config.load_base().unwrap();
    let result = run_iovsearch(
        &config,
        &base,
        IovsearchRun {
            dir: Some(dir.join("run")),
            ..IovsearchRun::default()
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
fn the_input_row_is_the_direct_evaluation_and_one_parameter_means_two_fits() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), BASE, Some("IOV?([CL],EXP)"), "");
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    assert!(
        base.prepared.population.subjects[0].occasions.len() > 1,
        "the base's iov_column read the occasions"
    );
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
    let result = run_iovsearch(
        &config,
        &base,
        IovsearchRun {
            dir: Some(dir.path().join("run")),
            ..IovsearchRun::default()
        },
    )
    .expect("search");
    let input = result.row("input").unwrap();
    assert_eq!(
        input.ofv.unwrap().to_bits(),
        direct.ofv.to_bits(),
        "input OFV {} vs direct {}",
        input.ofv.unwrap(),
        direct.ofv
    );
    assert_eq!(input.n_parameters, Some(direct.n_parameters));
    // One candidate parameter: the full-IOV model and no proper subset, and
    // the IIV step from it (when it wins) tries the one η.
    let step1: Vec<&str> = result
        .rows
        .iter()
        .filter(|r| r.step == 1)
        .map(|r| r.id.as_str())
        .collect();
    assert_eq!(step1, vec!["run1"]);
    assert_eq!(
        result.row("run1").unwrap().structure.description(),
        "IIV([CL]+[KA]+[V]);IOV([CL])"
    );
    assert_eq!(
        result.row("run1").unwrap().n_parameters,
        Some(direct.n_parameters + 1)
    );
    assert!(
        result.rows.iter().all(|r| r.error.is_none()),
        "{:?}",
        result.rows
    );
    let table = std::fs::read_to_string(dir.path().join("run/models.csv")).unwrap();
    assert_eq!(table.lines().count(), 1 + result.rows.len());
    assert!(dir.path().join("run/models/run1.ferx").exists());
    let final_text = std::fs::read_to_string(dir.path().join("run/final.ferx")).unwrap();
    parse_full_model(&final_text).expect("final.ferx parses");
}

#[test]
fn every_candidate_compiles_and_evaluates_under_each_distribution() {
    for distribution in ["disjoint", "joint", "same-as-iiv"] {
        let dir = tempfile::tempdir().unwrap();
        let path = write_config(
            dir.path(),
            BASE,
            None,
            &format!("[iovsearch]\ndistribution = \"{distribution}\"\n"),
        );
        let (_, result) = run(dir.path(), &path);
        let failed: Vec<String> = result
            .rows
            .iter()
            .filter(|r| r.error.is_some() || r.ofv.is_none())
            .map(|r| format!("{} ({}): {:?}", r.id, r.structure.description(), r.error))
            .collect();
        assert!(failed.is_empty(), "{distribution}: {failed:?}");
        // The full model plus the six proper subsets of three κ.
        assert_eq!(
            result.rows.iter().filter(|r| r.step == 1).count(),
            7,
            "{distribution}"
        );
        for r in &result.rows {
            let text = result.models[&r.id].render();
            parse_full_model(&text).unwrap_or_else(|e| panic!("{distribution} {}: {e}", r.id));
            let n_kappa_cov: usize = r
                .structure
                .kappa_blocks
                .iter()
                .map(|b| b.len() * (b.len() - 1) / 2)
                .sum();
            assert_eq!(
                r.n_parameters.unwrap(),
                4 + r.structure.etas.len() + r.structure.kappas.len() + n_kappa_cov,
                "{distribution} {} ({})",
                r.id,
                r.structure.description()
            );
        }
        let full = result.row("run1").unwrap();
        match distribution {
            "joint" => assert_eq!(full.structure.kappa_blocks.len(), 1),
            _ => assert!(full.structure.kappa_blocks.is_empty()),
        }
        eprintln!(
            "{distribution}: final {} ({})",
            result.final_id,
            result.final_structure.description()
        );
    }
}

#[test]
fn the_full_iov_candidate_is_the_hand_written_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), BASE, Some("IOV?([CL],EXP)"), "");
    let (config, result) = run(dir.path(), &path);
    let base = config.load_base().unwrap();
    let generated = result.models["run1"].render();
    // A tenth of the η's variance as the input evaluation left it — the
    // init itself — declared as `kappa`, and the κ beside the η.
    let hand_written = BASE
        .replace(
            "  CL = TVCL * exp(ETA_CL)",
            "  CL = TVCL * exp(ETA_CL + KAPPA_CL)",
        )
        .replace(
            "  sigma PROP_ERR",
            "  kappa KAPPA_CL ~ 0.009\n  sigma PROP_ERR",
        );
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
    assert_eq!(ours.kappa_names, theirs.kappa_names);
    assert_eq!(result.row("run1").unwrap().ofv, Some(ours.ofv));
}

#[test]
fn a_base_without_iov_column_is_refused_with_the_fix_named() {
    let dir = tempfile::tempdir().unwrap();
    let no_column = BASE.replace("  iov_column = OCC\n", "");
    let path = write_config(dir.path(), &no_column, None, "");
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    let err =
        run_iovsearch(&config, &base, IovsearchRun::default()).expect_err("no occasions were read");
    assert!(err.contains("iov_column = OCC"), "{err}");
    // And a column the file names has to be the base's.
    let path = write_config(dir.path(), BASE, None, "[iovsearch]\ncolumn = \"VISIT\"\n");
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    let err = run_iovsearch(&config, &base, IovsearchRun::default()).expect_err("mismatch");
    assert!(err.contains("must agree"), "{err}");
}
