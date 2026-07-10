//! Integration tests for checkpoint / restart of a stopped run (#755).
//!
//! These drive the public `fit()` with an absolute `checkpoint_path` (so nothing
//! is written into the working directory) and either a hand-built checkpoint
//! (to exercise the resume path deterministically) or a short real fit with a
//! zero-second interval (to exercise the write + delete-on-success wiring across
//! estimators). All fits are eval-only or a handful of iterations — no
//! convergence loops (Tier-2 rules).

use ferx_core::estimation::parameterization::{coordinate_names, pack_params};
use ferx_core::io::checkpoint::{self, Checkpoint, SCHEMA_VERSION};
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, CompiledModel, EstimationMethod, FitOptions, Population};
use std::path::Path;

const MODEL_SRC: &str = r"
[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_KA ~ 0.20
  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
";

fn load_model_and_pop() -> (CompiledModel, Population) {
    let model = parse_model_string(MODEL_SRC).expect("model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("data loads");
    (model, pop)
}

/// Absolute temp path unique to this process + tag (no working-dir pollution).
fn tmp_path(tag: &str) -> String {
    format!("/tmp/ferx_ckpt_it_{}_{}.tmp", tag, std::process::id())
}

fn base_opts(path: &str) -> FitOptions {
    let mut o = FitOptions::default();
    o.method = EstimationMethod::FoceI;
    o.run_covariance_step = false;
    o.verbose = false;
    o.checkpoint = true;
    o.checkpoint_path = Some(path.to_string());
    o.checkpoint_model_hash = Some("MODELHASH".to_string());
    o.checkpoint_data_hash = Some("DATAHASH".to_string());
    o
}

/// A checkpoint whose hashes/layout match this model but whose packed TVCL is
/// moved to `tvcl`, so a successful resume is observable as `theta[0] == tvcl`.
fn checkpoint_with_tvcl(model: &CompiledModel, tvcl: f64) -> Checkpoint {
    let mut packed = pack_params(&model.default_params);
    packed[0] = tvcl.ln(); // TVCL is packed as log (positive lower bound)
    Checkpoint {
        schema_version: SCHEMA_VERSION,
        ferx_version: "test".to_string(),
        model_hash: Some("MODELHASH".to_string()),
        data_hash: Some("DATAHASH".to_string()),
        method_chain: vec!["focei".to_string()],
        stage_idx: 0,
        packed,
        coord_names: coordinate_names(&model.default_params),
        iter: 11,
        ofv: 123.4,
        unix_time: 0,
    }
}

#[test]
fn resume_seeds_params_and_emits_banner() {
    let (model, pop) = load_model_and_pop();
    let path = tmp_path("resume");
    checkpoint::remove(&path);
    checkpoint::save_atomic(&path, &checkpoint_with_tvcl(&model, 0.25)).unwrap();

    let mut opts = base_opts(&path);
    opts.outer_maxiter = 0; // eval-only: resumed params are returned unchanged

    let result = fit(&model, &pop, &model.default_params, &opts).expect("resumed fit runs");

    // The resumed TVCL (0.25), not the model default (0.13), is what the fit used.
    assert!(
        (result.theta[0] - 0.25).abs() < 1e-6,
        "resume should seed TVCL from the checkpoint (got {})",
        result.theta[0]
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("Resuming from checkpoint")),
        "a resume banner should be recorded in warnings: {:?}",
        result.warnings
    );
    checkpoint::remove(&path);
}

#[test]
fn hash_mismatch_starts_fresh() {
    let (model, pop) = load_model_and_pop();
    let path = tmp_path("mismatch");
    checkpoint::remove(&path);
    checkpoint::save_atomic(&path, &checkpoint_with_tvcl(&model, 0.25)).unwrap();

    let mut opts = base_opts(&path);
    opts.outer_maxiter = 0;
    // Model hash differs from the checkpoint's → integrity check fails.
    opts.checkpoint_model_hash = Some("A_DIFFERENT_MODEL".to_string());

    let result = fit(&model, &pop, &model.default_params, &opts).expect("fresh fit runs");

    assert!(
        (result.theta[0] - 0.13).abs() < 1e-6,
        "hash mismatch must ignore the checkpoint and use the model default TVCL (got {})",
        result.theta[0]
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("Ignoring checkpoint")),
        "an 'ignoring checkpoint' warning should be recorded: {:?}",
        result.warnings
    );
    // The stale checkpoint was removed by the reject path.
    assert!(
        checkpoint::load(&path).is_none(),
        "stale checkpoint should be deleted"
    );
    checkpoint::remove(&path);
}

#[test]
fn disabled_ignores_existing_checkpoint() {
    let (model, pop) = load_model_and_pop();
    let path = tmp_path("disabled");
    checkpoint::remove(&path);
    checkpoint::save_atomic(&path, &checkpoint_with_tvcl(&model, 0.25)).unwrap();

    let mut opts = base_opts(&path);
    opts.outer_maxiter = 0;
    opts.checkpoint = false; // whole checkpoint subsystem off

    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit runs");

    assert!(
        (result.theta[0] - 0.13).abs() < 1e-6,
        "checkpoint = false must not resume (got TVCL {})",
        result.theta[0]
    );
    assert!(
        !result.warnings.iter().any(|w| w.contains("Resuming")),
        "no resume banner when checkpointing is disabled: {:?}",
        result.warnings
    );
    // The file is left untouched (neither read nor deleted).
    assert!(
        checkpoint::load(&path).is_some(),
        "disabled run must not touch the file"
    );
    checkpoint::remove(&path);
}

#[test]
fn hashless_options_do_not_resume() {
    // A checkpoint carrying hashes, but options with none: integrity can't be
    // verified, so the run must NOT resume (None == None must not count as a
    // match). It starts fresh from the model default.
    let (model, pop) = load_model_and_pop();
    let path = tmp_path("hashless");
    checkpoint::remove(&path);
    checkpoint::save_atomic(&path, &checkpoint_with_tvcl(&model, 0.25)).unwrap();

    let mut opts = base_opts(&path);
    opts.outer_maxiter = 0;
    opts.checkpoint_model_hash = None;
    opts.checkpoint_data_hash = None;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit runs");

    assert!(
        (result.theta[0] - 0.13).abs() < 1e-6,
        "must not resume without an integrity check (got TVCL {})",
        result.theta[0]
    );
    assert!(
        !result.warnings.iter().any(|w| w.contains("Resuming")),
        "no resume without verifiable hashes: {:?}",
        result.warnings
    );
    checkpoint::remove(&path);
}

/// A real FOCEI fit with a zero-second interval writes at least one checkpoint
/// during the run, then deletes it on success. Exercises the NLopt objective
/// write site and `finish_success` end to end.
#[test]
fn focei_fit_writes_then_removes_checkpoint() {
    let (model, pop) = load_model_and_pop();
    let path = tmp_path("focei_run");
    checkpoint::remove(&path);

    let mut opts = base_opts(&path);
    opts.outer_maxiter = 2; // a couple of iterations, not to convergence
    opts.checkpoint_interval_secs = 0; // every eval is "due"

    let result = fit(&model, &pop, &model.default_params, &opts).expect("short fit runs");
    assert!(result.theta[0].is_finite());
    assert!(
        checkpoint::load(&path).is_none(),
        "checkpoint must be removed after a successful fit"
    );
    checkpoint::remove(&path);
}

/// Same for the Gauss-Newton estimator, which owns its own loop and writes at a
/// different site.
#[test]
fn gauss_newton_fit_checkpoints_and_cleans_up() {
    let (model, pop) = load_model_and_pop();
    let path = tmp_path("gn_run");
    checkpoint::remove(&path);

    let mut opts = base_opts(&path);
    opts.method = EstimationMethod::FoceGn;
    opts.outer_maxiter = 2;
    opts.checkpoint_interval_secs = 0;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("short GN fit runs");
    assert!(result.theta[0].is_finite());
    assert!(
        checkpoint::load(&path).is_none(),
        "checkpoint must be removed after a successful GN fit"
    );
    checkpoint::remove(&path);
}

/// And for SAEM, whose write site snapshots the sampler state.
#[test]
fn saem_fit_checkpoints_and_cleans_up() {
    let (model, pop) = load_model_and_pop();
    let path = tmp_path("saem_run");
    checkpoint::remove(&path);

    let mut opts = base_opts(&path);
    opts.method = EstimationMethod::Saem;
    opts.saem_n_exploration = 2;
    opts.saem_n_convergence = 2;
    opts.checkpoint_interval_secs = 0;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("short SAEM fit runs");
    assert!(result.theta[0].is_finite());
    assert!(
        checkpoint::load(&path).is_none(),
        "checkpoint must be removed after a successful SAEM fit"
    );
    checkpoint::remove(&path);
}
