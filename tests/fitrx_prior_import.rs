//! `[priors] from_fit` naming a `.fitrx` bundle, parsed through ferx-core's public entry point
//! in a fresh process (#3041).
//!
//! The parser lives in `ferx-model`, below the `.fitrx` reader in `ferx_core::io::fitrx`, so
//! the reader is injected: each ferx-core parse entry point installs it before parsing. The
//! hook is a process-wide `OnceLock`, set by whichever install runs first, so a lib unit test
//! (where some earlier test may have installed it) cannot tell whether the public entry point
//! installs it. This binary can: it calls nothing but public API, never the install itself, so
//! if no public path installed the reader the `from_fit` parse fails with "no `.fitrx` reader is
//! installed". Tier 2: one two-iteration fit to produce the bundle, no convergence loop.

use ferx_core::io::fitrx::{save_fit, SaveFitOptions};
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const BODY: &str = "[individual_parameters]\n\
                    \x20 CL = TVCL * exp(ETA_CL)\n\
                    \x20 V  = TVV  * exp(ETA_V)\n\
                    \x20 KA = TVKA\n\n\
                    [structural_model]\n\
                    \x20 pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\n\
                    [error_model]\n\
                    \x20 DV ~ proportional(PROP_ERR)\n";

const PARAMS: &str = "[parameters]\n  \
                      theta TVCL(0.13, 0.001, 10.0)\n  \
                      theta TVV(8.0, 0.1, 500.0)\n  \
                      theta TVKA(1.0, 0.01, 50.0)\n  \
                      omega ETA_CL ~ 0.09\n  \
                      omega ETA_V  ~ 0.04\n  \
                      sigma PROP_ERR ~ 0.01\n\n";

#[test]
fn a_fitrx_named_in_from_fit_imports_on_the_first_parse_of_a_process() {
    // The bundle. A two-iteration fit has no covariance step, and an import refuses an
    // estimate without a standard error, so give every parameter a 1% one.
    let source = format!("{PARAMS}{BODY}");
    let parent = parse_model_string(&source).expect("parent model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin");
    let mut opts = FitOptions { method: EstimationMethod::FoceI, ..FitOptions::default() };
    opts.outer_maxiter = 2;
    opts.run_covariance_step = false;
    let mut parent_fit = fit(&parent, &pop, &parent.default_params, &opts).expect("short fit runs");
    parent_fit.se_theta = Some(parent_fit.theta.iter().map(|t| 0.01 * t).collect());
    parent_fit.se_omega = Some(vec![1e-3; parent.n_eta]);
    parent_fit.se_sigma = Some(vec![1e-3; parent_fit.sigma.len()]);
    let dir = tempfile::tempdir().unwrap();
    let bundle = dir.path().join("parent.fitrx");
    save_fit(&parent_fit, &pop, &source, &bundle, SaveFitOptions::default()).expect("save");

    let child = format!("{PARAMS}[priors]\n  from_fit = \"{}\"\n\n{BODY}", bundle.display());
    let model = parse_model_string(&child)
        .unwrap_or_else(|e| panic!("a `.fitrx` in `from_fit` must import: {e}"));

    // Every priorable parameter came from the bundle (3 θ, 2 Ω, 1 σ), centred on its estimate.
    assert_eq!(model.priors.len(), 6, "{:?}", model.priors);
    let i = parent_fit.theta_names.iter().position(|n| n == "TVCL").expect("TVCL");
    let cl = model.priors.iter().find(|p| p.name == "TVCL").expect("a TVCL prior");
    assert_eq!(cl.value, parent_fit.theta[i], "the prior is centred on the bundle's estimate");
}

/// The same import when the process's FIRST parse is a model FILE (#3041): `prepare_run`,
/// `fit_from_files` and the CLI all parse through `ModelSource::load`, a different entry point
/// from `parse_model_string`. The test above cannot see that path — its own parent parse goes
/// through `parse_model_string` and installs the reader for the whole process — so this one
/// builds the bundle, then re-runs this test binary as a child process, filtered to
/// [`child_prepare_run_imports_from_fit`], whose only engine call is `prepare_run`.
#[test]
fn a_fitrx_named_in_from_fit_imports_when_a_model_file_is_the_first_parse_of_a_process() {
    let source = format!("{PARAMS}{BODY}");
    let parent = parse_model_string(&source).expect("parent model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin");
    let mut opts = FitOptions { method: EstimationMethod::FoceI, ..FitOptions::default() };
    opts.outer_maxiter = 2;
    opts.run_covariance_step = false;
    let mut parent_fit = fit(&parent, &pop, &parent.default_params, &opts).expect("short fit runs");
    parent_fit.se_theta = Some(parent_fit.theta.iter().map(|t| 0.01 * t).collect());
    parent_fit.se_omega = Some(vec![1e-3; parent.n_eta]);
    parent_fit.se_sigma = Some(vec![1e-3; parent_fit.sigma.len()]);
    let dir = tempfile::tempdir().unwrap();
    save_fit(
        &parent_fit,
        &pop,
        &source,
        &dir.path().join("parent.fitrx"),
        SaveFitOptions::default(),
    )
    .expect("save");
    // Relative, so it resolves against the model file's directory, as a user's would.
    let child = format!("{PARAMS}[priors]\n  from_fit = \"parent.fitrx\"\n\n{BODY}");
    let model = dir.path().join("child.ferx");
    std::fs::write(&model, child).unwrap();

    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["child_prepare_run_imports_from_fit", "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_MODEL, &model)
        .output()
        .expect("re-run this test binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "prepare_run on a `from_fit` model as the first parse of a process failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

const CHILD_MODEL: &str = "FERX_FITRX_PRIOR_IMPORT_CHILD_MODEL";

/// The child half of the test above; a no-op unless that test spawned this process.
#[test]
fn child_prepare_run_imports_from_fit() {
    let Ok(model) = std::env::var(CHILD_MODEL) else { return };
    let run = ferx_core::prepare_run(&model, Some("data/warfarin.csv"))
        .unwrap_or_else(|e| panic!("a `.fitrx` in `from_fit` must import via prepare_run: {e}"));
    assert_eq!(run.parsed.model.priors.len(), 6, "{:?}", run.parsed.model.priors);
}
