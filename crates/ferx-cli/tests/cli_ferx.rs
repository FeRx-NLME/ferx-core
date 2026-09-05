//! End-to-end coverage for the `ferx` CLI binary.
//!
//! These run the built binary as a subprocess. Cargo exposes its path via the
//! `CARGO_BIN_EXE_ferx` env var, which is only defined for test targets of the
//! package that declares the binary — which is why this file lives in
//! `ferx-cli` rather than alongside the `generate_data` tests in the
//! `ferx-core` package (`tests/cli_binaries.rs`, #1114).
//!
//! The `check` subcommand and the one real fit here are both fast, so these
//! stay in the default test job rather than the slow tier.

use std::path::PathBuf;
use std::process::Command;

/// The repo root — two levels up from this package, since `ferx-cli` sits at
/// `crates/ferx-cli/`. The model and data paths below are repo-relative.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root resolves")
}

// ── ferx check: validate a model with/without data, JSON and human output ────

fn ferx() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_ferx"));
    c.current_dir(repo_root());
    c
}

#[test]
fn check_json_reports_valid_model() {
    let out = ferx()
        .args(["check", "examples/one_cpt_iv.ferx", "--json"])
        .output()
        .expect("run ferx check --json");
    assert!(
        out.status.success(),
        "check should exit 0 for a valid model"
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    // Output is a JSON CheckReport.
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("stdout is JSON");
    assert_eq!(v["valid"], serde_json::Value::Bool(true));
}

#[test]
fn check_human_output_runs() {
    let out = ferx()
        .args(["check", "examples/one_cpt_iv.ferx"])
        .output()
        .expect("run ferx check");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(
        stdout.contains("ok:"),
        "human check summary missing: {stdout}"
    );
}

#[test]
fn check_with_data_runs_data_dependent_checks() {
    let out = ferx()
        .args([
            "check",
            "examples/one_cpt_iv.ferx",
            "--data",
            "data/one_cpt_iv.csv",
        ])
        .output()
        .expect("run ferx check --data");
    // Exit code is 0 (valid) or 1 (warnings/errors) — both exercise the data
    // path; only a usage/serialization failure (2) would be wrong here.
    assert_ne!(out.status.code(), Some(2), "should not be a usage error");
}

#[test]
fn check_missing_model_is_usage_error() {
    let out = ferx().arg("check").output().expect("run ferx check");
    assert_eq!(out.status.code(), Some(2), "missing model → usage exit 2");
}

#[test]
fn check_data_flag_without_value_is_usage_error() {
    let out = ferx()
        .args(["check", "examples/one_cpt_iv.ferx", "--data"])
        .output()
        .expect("run ferx check --data (no value)");
    assert_eq!(out.status.code(), Some(2), "--data without value → exit 2");
}

#[test]
fn no_arguments_prints_usage_and_exits_one() {
    let out = ferx().output().expect("run ferx with no args");
    assert_eq!(out.status.code(), Some(1), "no args → usage exit 1");
}

// ── ferx fit: full success path (writes sdtab/yaml + .fitrx bundle) ──────────

/// Drives the fit half of `main`: a small 10-subject 1-cpt IV FOCEI fit with
/// `--threads` and `--output` (so the thread-pool and .fitrx-bundle branches
/// are covered too). All outputs land in a tempdir, so the repo tree is left
/// untouched. Fast (analytical PK, ~1s) — stays in the default test tier.
#[test]
fn fit_with_data_writes_outputs_and_bundle() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model = repo_root().join("examples/one_cpt_iv.ferx");
    let data = repo_root().join("data/one_cpt_iv.csv");

    let out = Command::new(env!("CARGO_BIN_EXE_ferx"))
        .current_dir(tmp.path())
        .arg(&model)
        .arg("--data")
        .arg(&data)
        .args(["--threads", "2", "--output", "run.fitrx", "--include-data"])
        .output()
        .expect("run ferx fit");
    assert!(
        out.status.success(),
        "fit should succeed; stderr=\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(
        stdout.contains("Fit completed!"),
        "summary missing: {stdout}"
    );
    assert!(stdout.contains("OFV:"));

    // Output files are named from the model stem and written to cwd (the tmp).
    for name in ["one_cpt_iv-fit.yaml", "one_cpt_iv-sdtab.csv", "run.fitrx"] {
        let p = tmp.path().join(name);
        assert!(p.exists(), "expected output {} to be written", p.display());
    }
    // #704: the standalone timing file is gone — timing now lives in the yaml.
    assert!(!tmp.path().join("one_cpt_iv-timing.txt").exists());

    let yaml = std::fs::read_to_string(tmp.path().join("one_cpt_iv-fit.yaml")).unwrap();
    assert!(
        yaml.contains("\nestimation:"),
        "yaml missing estimation: {yaml}"
    );
    // #713: convergence time is broken out per method-chain stage (this fit
    // is a single-stage FOCEI run), plus the covariance step separately.
    assert!(yaml.contains("  focei_wall_time_secs:"));
    assert!(yaml.contains("  covariance_wall_time_secs:"));
    assert!(yaml.contains("  wall_time_secs:"));
    assert!(yaml.contains("  n_threads_used:"));
    assert!(
        yaml.contains("\nenvironment:"),
        "yaml missing environment: {yaml}"
    );
    assert!(yaml.contains("  os:"));
    assert!(yaml.contains("  arch:"));
    assert!(yaml.contains("  in_docker:"));
    assert!(yaml.contains("  username:"));
    assert!(yaml.contains("  ferx_version:"));
}

/// Drives the `--simulate` half of `main` (no data file). Uses a tiny inline
/// analytical 1-cpt model with a `[simulation]` block written to the tempdir,
/// so `run_model_simulate` generates synthetic data and the output-writing path
/// runs — fast (analytical PK, 5 subjects), unlike the ODE example models.
#[test]
fn simulate_writes_outputs() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_src = "\
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
[simulation]
  n_subjects = 5
  dose_amt   = 100.0
  dose_cmt   = 1
  times      = [0.5, 1.0, 2.0, 4.0, 8.0]
  seed       = 1
";
    let model = tmp.path().join("sim_model.ferx");
    std::fs::write(&model, model_src).expect("write model");

    let out = Command::new(env!("CARGO_BIN_EXE_ferx"))
        .current_dir(tmp.path())
        .arg(&model)
        .arg("--simulate")
        .output()
        .expect("run ferx --simulate");
    assert!(
        out.status.success(),
        "simulate should succeed; stderr=\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(tmp.path().join("sim_model-fit.yaml").exists());
}

/// A Gaussian `[simulation]` block given only `horizon` (no `times`) must fail
/// loudly. The relaxed `times`-or-`horizon` parser rule (added for TTE models)
/// accepts it, but a Gaussian model has nothing to observe at a TTE horizon, so
/// `--simulate` must error rather than silently build zero-observation subjects
/// and fit on empty data (#522 review). Fast — errors before any fitting.
#[test]
fn simulate_gaussian_horizon_only_errors() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_src = "\
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
[simulation]
  n_subjects = 5
  horizon    = 14
";
    let model = tmp.path().join("gauss_horizon_only.ferx");
    std::fs::write(&model, model_src).expect("write model");

    let out = Command::new(env!("CARGO_BIN_EXE_ferx"))
        .current_dir(tmp.path())
        .arg(&model)
        .arg("--simulate")
        .output()
        .expect("run ferx --simulate");
    assert!(
        !out.status.success(),
        "a Gaussian horizon-only [simulation] must fail, not fit on empty data"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("times"),
        "error must point at the missing `times`: {stderr}"
    );
}

/// Covers the arg-parsing error exits in `main` (each calls `process::exit(1)`
/// before any fitting, so these are fast). One subprocess per bad flag.
#[test]
fn bad_flags_exit_with_error() {
    // --threads with a non-numeric value.
    let out = ferx()
        .args(["examples/one_cpt_iv.ferx", "--threads", "notanumber"])
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(1), "bad --threads → exit 1");

    // --output present but missing its value.
    let out = ferx()
        .args([
            "examples/one_cpt_iv.ferx",
            "--data",
            "data/one_cpt_iv.csv",
            "--output",
        ])
        .output()
        .expect("run");
    assert_eq!(
        out.status.code(),
        Some(1),
        "--output without value → exit 1"
    );

    // --inits-from-nca with an unknown method.
    let out = ferx()
        .args([
            "examples/one_cpt_iv.ferx",
            "--data",
            "data/one_cpt_iv.csv",
            "--inits-from-nca=bogus",
        ])
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(1), "bad --inits-from-nca → exit 1");
}

/// A fit that fails (model file does not exist) takes the `Err` arm of `main`
/// and exits non-zero with an `Error:` message.
#[test]
fn fit_with_missing_files_errors() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = Command::new(env!("CARGO_BIN_EXE_ferx"))
        .current_dir(tmp.path())
        .args(["does_not_exist.ferx", "--data", "nope.csv"])
        .output()
        .expect("run ferx fit on missing files");
    assert!(!out.status.success(), "missing files should fail the fit");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("Error:"),
        "expected an Error: message on stderr"
    );
}

// ── ferx bootstrap (#1140) ───────────────────────────────────────────────────

#[test]
fn bootstrap_help_lists_the_psn_options() {
    let out = ferx()
        .args(["bootstrap", "--help"])
        .output()
        .expect("run ferx bootstrap --help");
    assert!(out.status.success(), "--help should exit 0");
    let stdout = String::from_utf8(out.stdout).unwrap();
    for flag in [
        "--samples",
        "--seed",
        "--stratify-on",
        "--sample-size",
        "--update-inits",
        "--keep-covariance",
        "--dofv",
        "--summarize",
    ] {
        assert!(stdout.contains(flag), "help does not mention {flag}");
    }
}

#[test]
fn bootstrap_without_a_model_is_a_usage_error() {
    let out = ferx()
        .args(["bootstrap"])
        .output()
        .expect("run ferx bootstrap");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("Usage: ferx bootstrap"));
}

#[test]
fn bootstrap_summarize_needs_a_directory() {
    let out = ferx()
        .args(["bootstrap", "--summarize"])
        .output()
        .expect("run ferx bootstrap --summarize");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--directory"));
}

#[test]
fn bootstrap_rejects_a_covstep_filter_without_the_covariance_step() {
    // The filter reads a diagnostic that only exists when the step ran, so
    // accepting it silently would drop a filter the user asked for.
    let out = ferx()
        .args([
            "bootstrap",
            "examples/one_cpt_iv.ferx",
            "--data",
            "data/one_cpt_iv.csv",
            "--samples",
            "1",
            "--skip-covariance-step-terminated",
        ])
        .output()
        .expect("run ferx bootstrap");
    assert_eq!(out.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("--keep-covariance"),
        "the error should name the way out"
    );
}

/// The top-level dispatcher must not have stolen the ordinary fit path: a model
/// file whose name happens to start with a subcommand-looking word is still a
/// model file, and `ferx <model>` is unchanged.
#[test]
fn the_bootstrap_subcommand_does_not_shadow_a_plain_fit() {
    let out = ferx()
        .args(["check", "examples/one_cpt_iv.ferx"])
        .output()
        .expect("run ferx check");
    assert!(out.status.success());
}

// ── ferx covsearch / ferx allometry (#1180) ─────────────────────────────────

#[test]
fn covsearch_help_lists_the_search_options() {
    let out = ferx()
        .args(["covsearch", "--help"])
        .output()
        .expect("run ferx covsearch --help");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    for word in [
        "p_forward",
        "p_backward",
        "scm-forward-then-backward",
        "adaptive_scope_reduction",
        "--directory",
        "--resume",
    ] {
        assert!(stdout.contains(word), "help does not mention {word}");
    }
}

#[test]
fn covsearch_without_a_file_is_a_usage_error_and_unknown_flags_are_refused() {
    let out = ferx().args(["covsearch"]).output().expect("run");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("Usage: ferx covsearch"));

    let out = ferx()
        .args([
            "covsearch",
            "examples/two_cpt_oral_cov.ferxsearch",
            "--samples",
            "3",
        ])
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown flag: --samples"));
}

#[test]
fn covsearch_refuses_a_file_meant_for_another_tool_before_reading_data() {
    // The shipped example ranks on BIC (a modelsearch file); covsearch
    // refuses it by name — and before loading the dataset, so the refusal
    // is immediate.
    let out = ferx()
        .args(["covsearch", "examples/two_cpt_oral_cov.ferxsearch"])
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("[rank] type = \"bic\": covsearch selects by the likelihood-ratio test"),
        "{stderr}"
    );
    assert!(!stderr.contains("Data:"), "{stderr}");
}

const ONE_CPT_WT: &str = "[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[covariates]
  WT continuous

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  maxiter    = 0
  covariance = false
  checkpoint = false
";

/// A whole search from the command line, on evaluations (`maxiter = 0`) so
/// it is seconds: the base model, one forward step over two candidates that
/// cannot be significant, and the files a user reads afterwards.
#[test]
fn covsearch_runs_a_search_and_writes_its_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("wt.ferx"), ONE_CPT_WT).unwrap();
    let data = repo_root().join("data/two_cpt_oral_cov.csv");
    let config = format!(
        "base = \"wt.ferx\"\ndata = \"{}\"\n[space]\nmfl = \"COVARIATE?([CL,V], WT, pow)\"\n\
         [covsearch]\nalgorithm = \"scm-forward\"\n\
         [strictness]\nrequire_converged = false\nreject_init_stall = false\n\
         reject_on_boundary = false\n[run]\nretries = 0\nthreads = 2\n",
        data.display()
    );
    std::fs::write(dir.path().join("wt.ferxsearch"), config).unwrap();

    let out = ferx()
        .args([
            "covsearch",
            &dir.path().join("wt.ferxsearch").to_string_lossy(),
        ])
        .output()
        .expect("run ferx covsearch");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "stderr: {stderr}\nstdout: {stdout}");
    assert!(
        stderr.contains("Step 1 (forward): fitting 2 candidates"),
        "{stderr}"
    );
    assert!(
        stderr.contains("Step 1 (forward): nothing accepted"),
        "{stderr}"
    );
    assert!(stdout.contains("CL-WT-power"), "{stdout}");
    assert!(stdout.contains("not significant"), "{stdout}");
    assert!(stdout.contains("Final model: OFV"), "{stdout}");

    let run = dir.path().join("wt-covsearch");
    assert!(run.join("steps.csv").exists(), "{stderr}");
    assert!(run.join("final.ferx").exists());
    assert!(run.join("final-fit.yaml").exists());
    assert!(run.join("base/candidates.csv").exists());
    assert!(run.join("forward-1/candidates.csv").exists());
    let steps = std::fs::read_to_string(run.join("steps.csv")).unwrap();
    assert_eq!(steps.lines().count(), 3, "{steps}");
}

#[test]
fn allometry_help_and_usage() {
    let out = ferx()
        .args(["allometry", "--help"])
        .output()
        .expect("run ferx allometry --help");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    for word in [
        "--covariate",
        "--reference",
        "--estimate",
        "--parameters",
        "ALLOMETRY(WT, 70)",
    ] {
        assert!(stdout.contains(word), "help does not mention {word}");
    }
    let out = ferx().args(["allometry"]).output().expect("run");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("Usage: ferx allometry"));
}

#[test]
fn allometry_scales_a_model_from_the_command_line() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model = "[parameters]
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
  WT continuous

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 0
  covariance = false
  checkpoint = false
";
    let path = dir.path().join("two_cpt.ferx");
    std::fs::write(&path, model).unwrap();
    let data = repo_root().join("data/two_cpt_oral_cov.csv");
    let out = ferx()
        .args([
            "allometry",
            &path.to_string_lossy(),
            "--data",
            &data.to_string_lossy(),
            "--retries",
            "0",
            "--threads",
            "2",
        ])
        .output()
        .expect("run ferx allometry");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "stderr: {stderr}\nstdout: {stdout}");
    for line in [
        "CL ~ WT power(center = 70, fix = 0.75)",
        "Q ~ WT power(center = 70, fix = 0.75)",
        "V1 ~ WT power(center = 70, fix = 1)",
        "V2 ~ WT power(center = 70, fix = 1)",
        "dOFV (base - allometric)",
    ] {
        assert!(stdout.contains(line), "missing `{line}` in:\n{stdout}");
    }
    let written = std::fs::read_to_string(dir.path().join("two_cpt-allometric.ferx")).unwrap();
    assert!(written.contains("[covariate_model]"));
    assert!(written.contains("V1 ~ WT power(center = 70, fix = 1.0)"));
}
