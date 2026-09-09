//! Tier-2 end-to-end checks for covsearch and allometry (#1180), on a real
//! model and dataset but never to convergence: every fit is an evaluation
//! (`maxiter = 0`, ferx's `MAXEVAL=0`).
//!
//! The unit tests in `src/covsearch/mod_tests.rs` script the fitter, so they
//! test the step logic and never compile a candidate. This file is the other
//! half: from a `.ferxsearch` file through symbol resolution, the edits, the
//! runner, `fit()` and the files on disk.
//!
//! Its oracle is **degenerate**, the issue's third validation bullet: a
//! search space that contains a single point — every effect already in the
//! base model — must return the base fit *bit for bit*, the OFV `fit()`
//! gives on the same model and population. Anything the search's own path
//! changes — an edit applied to the base, a fit option dropped, a seeded
//! initial estimate — moves that number.

use std::path::{Path, PathBuf};

use ferx_core::fit;
use ferx_tools::allometry::{run_allometry, AllometryOptions, AllometryRun};
use ferx_tools::covsearch::{run_covsearch, CovsearchRun, Phase};
use ferx_tools::search::{RunOptions, SearchConfig};

const DATA: &str = "../../data/two_cpt_oral_cov.csv";

/// A two-compartment oral base model that evaluates rather than fits.
/// `covariance = false` so no covariance step runs, and every gate that an
/// evaluation would trip is off in the config below.
fn base_model(relations: &str) -> String {
    format!(
        "[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(40.0, 1.0, 500.0)
  theta TVQ(8.0, 0.1, 100.0)
  theta TVV2(80.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[covariates]
  WT   continuous
  CRCL continuous
{relations}
[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 0
  covariance = false
  checkpoint = false
"
    )
}

/// Write a base model and a `.ferxsearch` beside it in a temp dir.
fn write_config(dir: &Path, relations: &str, mfl: &str, extra: &str) -> PathBuf {
    std::fs::write(dir.join("base.ferx"), base_model(relations)).unwrap();
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

const IN_BASE: &str = "
[covariate_model]
  CL ~ WT   power(center = median)
  CL ~ CRCL power(center = median)
";

#[test]
fn a_single_point_space_returns_the_base_fit_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();
    // Both exploratory effects are already in the base model, so the space
    // has exactly one point.
    let path = write_config(dir.path(), IN_BASE, "COVARIATE?(CL, [WT,CRCL], pow)", "");
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

    let result = run_covsearch(
        &config,
        &base,
        CovsearchRun {
            dir: Some(dir.path().join("run")),
            ..CovsearchRun::default()
        },
    )
    .expect("search");

    assert!(result.steps.is_empty(), "{:?}", result.steps);
    assert_eq!(result.final_step, 0);
    assert_eq!(
        result.final_ofv.to_bits(),
        direct.ofv.to_bits(),
        "search OFV {} vs direct {}",
        result.final_ofv,
        direct.ofv
    );
    assert_eq!(result.base_model.render(), base.text.render());
    assert_eq!(result.final_model.render(), base.text.render());
    let fit = result.final_fit.as_ref().expect("the base fit");
    assert_eq!(fit.n_parameters, direct.n_parameters);
    assert_eq!(result.included.len(), 2);
    assert_eq!(result.notes.len(), 2, "{:?}", result.notes);
    assert!(result.notes[0].contains("not explored: CL-WT-power"));

    // The files: an empty step table (header only) and the base model as
    // the final one, with its own estimates written in.
    let steps = std::fs::read_to_string(dir.path().join("run/steps.csv")).unwrap();
    assert_eq!(steps.lines().count(), 1);
    let final_model = std::fs::read_to_string(dir.path().join("run/final.ferx")).unwrap();
    assert!(final_model.contains("CL ~ WT   power(center = median)"));
    assert!(dir.path().join("run/base/candidates.csv").exists());
}

#[test]
fn a_forward_step_compiles_every_candidate_and_reports_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        "",
        "COVARIATE?(@IIV, @CONTINUOUS, [pow,lin])",
        "[covsearch]\nalgorithm = \"scm-forward\"\n",
    );
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    let result = run_covsearch(
        &config,
        &base,
        CovsearchRun {
            dir: Some(dir.path().join("run")),
            ..CovsearchRun::default()
        },
    )
    .expect("search");

    // Two η-parameters × two covariates × two forms = eight candidates, all
    // evaluated at their initial estimates: none is significant, so the
    // search stops after one step on the base model.
    let rows: Vec<_> = result.step_rows(1).collect();
    assert_eq!(rows.len(), 8);
    assert_eq!(result.n_steps(), 1);
    assert_eq!(result.final_step, 0);
    assert!(rows
        .iter()
        .all(|r| r.phase == Phase::Forward && !r.selected));
    for r in &rows {
        let ofv = r.ofv.expect("every candidate compiled and evaluated");
        assert!(ofv.is_finite());
        // The relation reached the engine: a θ of 0.001 on a centred factor
        // moves the objective, if only slightly.
        assert_ne!(
            ofv.to_bits(),
            result.base_ofv.to_bits(),
            "{}",
            r.effect.label()
        );
        let t = r.lrt.expect("a comparison was made");
        assert_eq!(t.df, 1, "{}", r.effect.label());
        assert!(!t.significant);
        assert!(t.p_value.is_finite());
        assert_eq!(r.converged, Some(false));
        assert!(r.passed, "{:?}", r.failures);
    }
    let forms: Vec<&str> = rows.iter().map(|r| r.effect.form_label()).collect();
    assert!(forms.contains(&"power") && forms.contains(&"linear"));

    let steps = std::fs::read_to_string(dir.path().join("run/steps.csv")).unwrap();
    assert_eq!(steps.lines().count(), 9);
    assert!(steps.contains("f1-CL-WT-power"));
    assert!(dir.path().join("run/forward-1/candidates.csv").exists());
    assert!(dir
        .path()
        .join("run/forward-1/search_journal.jsonl")
        .exists());
}

#[test]
fn allometry_scales_clearances_and_volumes_and_fits_both_models() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), "", "ALLOMETRY(WT, 70)", "");
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    let options = AllometryOptions::from_config(&config).unwrap();
    let result = run_allometry(
        &base,
        &options,
        AllometryRun {
            dir: Some(dir.path().join("allometry")),
            threads: Some(2),
            cancel: None,
            run_options: RunOptions {
                strictness: ferx_core::Strictness::none(),
                n_starts: 1,
                ..RunOptions::default()
            },
        },
    )
    .expect("allometry");

    let scaled: Vec<&str> = result
        .scalings
        .iter()
        .map(|s| s.parameter.as_str())
        .collect();
    assert_eq!(scaled, vec!["CL", "Q", "V1", "V2"]);
    let base_ofv = result.base.ofv.expect("base fit");
    let scaled_ofv = result.scaled.ofv.expect("scaled fit");
    assert!(base_ofv.is_finite() && scaled_ofv.is_finite());
    assert_ne!(base_ofv.to_bits(), scaled_ofv.to_bits());
    assert_eq!(result.dofv(), Some(base_ofv - scaled_ofv));
    // Fixed exponents add no free parameter.
    let fit = result.scaled.fit.as_ref().unwrap();
    assert_eq!(
        fit.n_parameters,
        result.base.fit.as_ref().unwrap().n_parameters
    );
    let relations: Vec<(String, f64)> = fit
        .covariate_relations
        .iter()
        .map(|r| (r.parameter.clone(), r.thetas[0].estimate))
        .collect();
    assert_eq!(
        relations,
        vec![
            ("CL".into(), 0.75),
            ("Q".into(), 0.75),
            ("V1".into(), 1.0),
            ("V2".into(), 1.0)
        ]
    );
    assert!(fit.covariate_relations.iter().all(|r| r.thetas[0].fixed));
    assert!(dir.path().join("allometry/candidates.csv").exists());
}
