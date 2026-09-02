//! Tier-1 test for quiet fits (#1115 G2).
//!
//! A tool running 200 fits cannot have each of them writing per-iteration
//! progress to the console, so `FitOptions::quiet()` (`verbose = false`) has to
//! mean *silent*, not *less chatty*. Every non-fatal message the engine produces
//! already goes into `FitResult.warnings` by convention; this test is the
//! backstop that keeps it true — a new estimator that reaches for `eprintln!`
//! outside a `verbose` gate turns it red.
//!
//! Capturing the real file descriptors is the only way to see the failure:
//! libtest's own capture intercepts `println!`/`eprintln!` from the test thread,
//! so an in-process check would pass no matter what the engine wrote. The test
//! therefore re-executes the test binary as a child process (guarded by
//! `FERX_QUIET_FIT_CHILD`) and inspects the child's real stdout/stderr.

use super::*;
use crate::parser::model_parser::parse_model_string;
use std::collections::HashMap;

/// Set in the child process, where this test runs the fit instead of spawning.
const CHILD_ENV: &str = "FERX_QUIET_FIT_CHILD";

/// libtest's `--exact` filter for this test — must track the module path.
const TEST_PATH: &str = "api::quiet_fit_tests::a_quiet_fit_writes_nothing_to_stdout_or_stderr";

fn model() -> CompiledModel {
    parse_model_string(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
"#,
    )
    .expect("parse")
}

fn subject(id: &str, scale: f64) -> Subject {
    let obs_times = vec![0.5, 2.0, 8.0, 24.0];
    let observations = obs_times.iter().map(|t| scale * 10.0 / (1.0 + t)).collect();
    Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![1, 1, 1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

fn population() -> Population {
    Population {
        subjects: vec![subject("1", 1.0), subject("2", 1.2), subject("3", 0.9)],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// The body that runs in the child: a real (if short) quiet fit, plus the other
/// half of the contract — the warnings are still reachable on the result.
fn run_quiet_fit() {
    let model = model();
    let pop = population();
    let opts = FitOptions {
        outer_maxiter: 2,
        run_covariance_step: false,
        // `optimizer_trace` used to print two ungated lines on this path; a
        // quiet fit must stay silent with it on, and surface a trace-file
        // failure as a warning instead (#1115).
        optimizer_trace: true,
        ..Default::default()
    }
    .quiet();
    assert!(!opts.verbose);

    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit");
    // `warnings` is the channel a quiet caller reads instead of the console; it
    // must remain populated (not merely silent) — the SS/boundary/shrinkage
    // machinery pushes into it regardless of `verbose`.
    let _: &Vec<String> = &result.warnings;
    assert!(result.ofv.is_finite());
}

/// libtest's own progress lines, which are not the engine talking.
fn is_harness_line(line: &str) -> bool {
    let l = line.trim();
    l.is_empty()
        || l.starts_with("running ")
        || l.starts_with("test ")
        || l.starts_with("Compiling")
        || l.starts_with("Finished")
}

#[test]
fn a_quiet_fit_writes_nothing_to_stdout_or_stderr() {
    if std::env::var(CHILD_ENV).is_ok() {
        run_quiet_fit();
        return;
    }

    let exe = std::env::current_exe().expect("path to this test binary");
    // `--nocapture` is required: without it libtest swallows the very output
    // this test exists to detect.
    let out = std::process::Command::new(&exe)
        .args([TEST_PATH, "--exact", "--nocapture", "--test-threads", "1"])
        .env(CHILD_ENV, "1")
        .output()
        .expect("re-run this test binary as a child process");

    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "the child run failed ({:?})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        out.status
    );
    assert!(
        stdout.contains(TEST_PATH),
        "the child did not actually run the fit — filter `{TEST_PATH}` matched nothing:\n{stdout}"
    );

    assert!(
        stderr.trim().is_empty(),
        "a quiet fit wrote to stderr:\n{stderr}"
    );
    let stray: Vec<&str> = stdout.lines().filter(|l| !is_harness_line(l)).collect();
    assert!(
        stray.is_empty(),
        "a quiet fit wrote to stdout:\n{}",
        stray.join("\n")
    );
}
