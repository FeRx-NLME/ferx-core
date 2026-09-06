//! Tier-2 end-to-end checks for ruvsearch (#1182), on a real model and
//! dataset but never to convergence: every fit is an evaluation
//! (`maxiter = 0`, ferx's `MAXEVAL=0`).
//!
//! The unit tests in `src/ruvsearch/mod_tests.rs` script the fitter, so they
//! test the iteration logic and never compile a candidate. This file is the
//! other half: from a `.ferxsearch` file through the error-model edits, the
//! runner, `fit()` and the files on disk.
//!
//! * The **degenerate oracle**: a plain proportional input evaluated with
//!   `maxiter = 0` cannot improve, so the search must return the input fit
//!   *bit for bit* — the OFV `fit()` gives on the same model and population.
//! * **Every candidate compiles and evaluates**, on the real data and — with
//!   the pre-screen — on the CWRES data, including the compositions a second
//!   iteration would build.
//! * A candidate **is the hand-written model**: the search's `power`
//!   candidate and its hand-typed twin evaluate to the same OFV bit for bit.

use std::path::{Path, PathBuf};

use ferx_core::edit::ModelText;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, RuvMagnitude};
use ferx_tools::ruvsearch::{run_ruvsearch, RuvFeature, RuvsearchRun};
use ferx_tools::search::SearchConfig;

const DATA: &str = "../../data/warfarin.csv";

/// The warfarin one-compartment oral model, evaluating rather than fitting,
/// under FOCEI so `IIV_on_RUV` is a candidate.
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
  method     = focei
  maxiter    = 0
  covariance = false
  checkpoint = false
";

fn write_config(dir: &Path, base: &str, section: &str) -> PathBuf {
    std::fs::write(dir.join("base.ferx"), base).unwrap();
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join(DATA);
    let config = format!(
        "base = \"base.ferx\"\ndata = \"{}\"\n\n[ruvsearch]\n{section}\n\
         [strictness]\nrequire_converged = false\nreject_init_stall = false\n\
         reject_on_boundary = false\n\n[run]\nretries = 0\nthreads = 2\n",
        data.display()
    );
    let path = dir.join("search.ferxsearch");
    std::fs::write(&path, config).unwrap();
    path
}

fn direct_ofv(base: &ferx_tools::search::BaseModel) -> f64 {
    fit(
        &base.prepared.parsed.model,
        &base.prepared.population,
        &base.prepared.init_params,
        &{
            let mut o = base.prepared.parsed.fit_options.clone().quiet();
            o.threads = Some(2);
            o
        },
    )
    .expect("direct evaluation")
    .ofv
}

#[test]
fn an_evaluation_only_search_returns_the_input_bit_for_bit_and_fits_every_candidate() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), BASE, "max_iter = 1\n");
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    let direct = direct_ofv(&base);

    let result = run_ruvsearch(
        &config,
        &base,
        RuvsearchRun {
            dir: Some(dir.path().join("run")),
            ..RuvsearchRun::default()
        },
    )
    .expect("search");

    assert_eq!(
        result.base_id, "input",
        "a plain proportional input is its own base"
    );
    assert_eq!(
        result.input_ofv.to_bits(),
        direct.to_bits(),
        "search OFV {} vs direct {}",
        result.input_ofv,
        direct
    );
    // Nothing can be significant at its initial estimates, so the input is
    // the final model — the same text, the same fit.
    assert_eq!(result.final_id, "input");
    assert_eq!(result.final_ofv.to_bits(), direct.to_bits());
    assert_eq!(result.final_model.render(), base.text.render());
    assert!(result.features.is_empty());
    // Every candidate was built and evaluated: IIV_on_RUV, power, combined,
    // and the time-varying cuts (warfarin's TAD distribution has three
    // distinct quantiles).
    let it1: Vec<&str> = result
        .iteration_rows(1)
        .map(|r| r.candidate.as_str())
        .collect();
    assert_eq!(
        it1,
        vec![
            "IIV_on_RUV-1",
            "power-1",
            "combined-1",
            "time_varying1-1",
            "time_varying2-1",
            "time_varying3-1"
        ]
    );
    for r in result.iteration_rows(1) {
        assert!(r.ofv.is_some_and(|v| v.is_finite()), "{r:?}");
        let lrt = r
            .lrt
            .unwrap_or_else(|| panic!("{}: {:?}", r.candidate, r.note));
        assert_eq!(lrt.df, 1, "{}: one parameter added", r.candidate);
        assert!(!lrt.significant);
        assert!(!r.selected);
        // Each candidate's text is the parent plus one feature, and it
        // compiles — the compiled model carries what the edit wrote.
        let model = &result.models[&r.candidate];
        let compiled = parse_full_model(&model.render())
            .unwrap_or_else(|e| panic!("{}: {e}\n{}", r.candidate, model.render()));
        match r.feature.unwrap() {
            RuvFeature::IivOnRuv => assert!(compiled.model.residual_error_eta.is_some()),
            RuvFeature::Power => assert!(compiled.model.has_ruv_exponent()),
            RuvFeature::Combined => assert_eq!(compiled.model.default_params.sigma.values.len(), 2),
            RuvFeature::TimeVarying(_) => {
                assert!(compiled.model.has_custom_ruv_magnitude());
                assert!(!compiled.model.has_ruv_exponent());
            }
        }
    }
    // The neutral starts: a power candidate at `P = 1` and a time-varying
    // candidate at `θ = 1` are the parent's model, so their evaluated OFV is
    // the input's — to the ~1-ULP reassociation the magnitude-scaled variance
    // path is documented to carry against the bare one (`residual_error.rs`),
    // which is why this is a tolerance and not `to_bits`.
    for id in ["power-1", "time_varying1-1"] {
        let r = result.rows.iter().find(|r| r.candidate == id).unwrap();
        let ofv = r.ofv.unwrap();
        assert!(
            (ofv - direct).abs() <= 1e-9 * direct.abs(),
            "{id}: a neutral start is the parent's objective ({ofv} vs {direct})"
        );
    }
    // The files.
    let run = dir.path().join("run");
    assert!(ferx_tools::ruvsearch::steps_path(&run).exists());
    assert!(ferx_tools::ruvsearch::final_model_path(&run).exists());
    assert!(ferx_tools::ruvsearch::models_dir(&run)
        .join("power-1.ferx")
        .exists());
    let steps = std::fs::read_to_string(ferx_tools::ruvsearch::steps_path(&run)).unwrap();
    assert_eq!(steps.lines().count(), 1 + result.rows.len());
}

#[test]
fn a_generated_power_candidate_is_the_hand_written_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        BASE,
        "max_iter = 1\nskip = [\"IIV_on_RUV\", \"time_varying\"]\n",
    );
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    let result = run_ruvsearch(&config, &base, RuvsearchRun::default()).expect("search");
    let generated = result.models["power-1"].render();
    // The hand-written twin, at the same (seeded) estimates: the search seeds
    // the candidate from the input's evaluation, which is the file's values
    // rounded through the edit layer's 15 significant digits.
    let hand = BASE
        .replace(
            "  sigma PROP_ERR ~ 0.02 (sd)\n",
            "  sigma PROP_ERR ~ 0.02 (sd)\n  theta RUV_POW(1.0, 0.01, 10.0)\n",
        )
        .replace(
            "DV ~ proportional(PROP_ERR)",
            "DV ~ power(PROP_ERR, RUV_POW)",
        );
    let g = parse_full_model(&generated).unwrap_or_else(|e| panic!("{e}\n{generated}"));
    let h = parse_full_model(&hand).unwrap();
    assert_eq!(g.model.theta_names, h.model.theta_names);
    let opts = {
        let mut o = g.fit_options.clone().quiet();
        o.threads = Some(2);
        o
    };
    let fg = fit(
        &g.model,
        &base.prepared.population,
        &g.model.default_params,
        &opts,
    )
    .unwrap();
    let fh = fit(
        &h.model,
        &base.prepared.population,
        &h.model.default_params,
        &opts,
    )
    .unwrap();
    assert_eq!(
        fg.ofv.to_bits(),
        fh.ofv.to_bits(),
        "{} vs {}",
        fg.ofv,
        fh.ofv
    );
    assert_eq!(fg.n_parameters, fh.n_parameters);
    // And a power candidate at a non-neutral exponent is a different model
    // from the input: the straddle that makes the equality above meaningful.
    let other = hand.replace("RUV_POW(1.0,", "RUV_POW(1.3,");
    let o = parse_full_model(&other).unwrap();
    let fo = fit(
        &o.model,
        &base.prepared.population,
        &o.model.default_params,
        &opts,
    )
    .unwrap();
    assert_ne!(fo.ofv.to_bits(), fh.ofv.to_bits());
}

#[test]
fn an_additive_input_is_refitted_as_proportional_first() {
    let additive = BASE
        .replace("sigma PROP_ERR ~ 0.02 (sd)", "sigma ADD_ERR ~ 0.5 (sd)")
        .replace("DV ~ proportional(PROP_ERR)", "DV ~ additive(ADD_ERR)");
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        &additive,
        "max_iter = 1\nskip = [\"IIV_on_RUV\", \"time_varying\", \"power\"]\n",
    );
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    let result = run_ruvsearch(&config, &base, RuvsearchRun::default()).expect("search");
    assert_eq!(result.base_id, "base");
    let base_model = &result.models["base"];
    assert_eq!(
        base_model.block_lines("error_model"),
        vec!["DV ~ proportional(PROP_ERR)"]
    );
    assert!(!base_model.render().contains("ADD_ERR"));
    assert!(result.base_ofv.is_finite() && result.input_ofv.is_finite());
    // No candidate at all: `power` skipped retires `combined` too.
    assert_eq!(result.iteration_rows(1).count(), 0);
    assert!(result.notes.iter().any(|n| n.contains("no candidate left")));
    // The final comparison: the input is returned (the base cannot beat the
    // input by 10.83 in an evaluation), and the note says so.
    assert_eq!(result.final_id, "input");
    assert!(result
        .notes
        .iter()
        .any(|n| n.contains("did not beat the input model")));
}

#[test]
fn the_cwres_prescreen_evaluates_every_screening_model_on_the_residuals() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), BASE, "max_iter = 1\ncwres_prescreen = true\n");
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    let result = run_ruvsearch(
        &config,
        &base,
        RuvsearchRun {
            dir: Some(dir.path().join("run")),
            ..RuvsearchRun::default()
        },
    )
    .expect("search");
    let screened: Vec<_> = result.rows.iter().filter(|r| r.screened).collect();
    // The CWRES base and six candidates, every one with a finite OFV on the
    // residual dataset.
    assert_eq!(screened.len(), 7, "{screened:?}");
    for r in &screened {
        assert!(
            r.ofv.is_some_and(|v| v.is_finite()),
            "{}: {:?} {:?}",
            r.candidate,
            r.note,
            r.failures
        );
    }
    let base_row = screened
        .iter()
        .find(|r| r.candidate == "cwres-base-1")
        .unwrap();
    assert!(base_row.feature.is_none());
    // The screening models are compilable models on the result.
    for id in [
        "cwres-base-1",
        "cwres-IIV_on_RUV-1",
        "cwres-power-1",
        "cwres-combined-1",
        "cwres-time_varying1-1",
    ] {
        let m: &ModelText = &result.models[id];
        let compiled = parse_full_model(&m.render()).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert!(compiled.model.is_algebraic(), "{id}");
        let _: Option<&RuvMagnitude> = compiled.model.ruv_magnitude.as_ref();
    }
    // A screening *evaluation* compares the candidates' variance structures
    // at their initial values, so it may or may not clear the cutoff; what
    // is pinned is the shape — at most one refit, and it is the screened pick.
    let refits: Vec<_> = result
        .rows
        .iter()
        .filter(|r| !r.screened && r.iteration == 1)
        .collect();
    assert!(refits.len() <= 1, "{refits:?}");
    if let Some(refit) = refits.first() {
        let pick = screened.iter().find(|r| r.selected).expect("a pick");
        assert_eq!(refit.feature, pick.feature);
        assert!(refit.ofv.is_some_and(|v| v.is_finite()));
        assert!(refit.lrt.is_some(), "{:?}", refit.note);
    } else {
        assert!(screened.iter().all(|r| !r.selected));
        assert_eq!(result.final_id, "input");
    }
    // The screening step has its own journal directory.
    assert!(dir.path().join("run").join("screen-1").exists());
}
