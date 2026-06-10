//! End-to-end coverage for the two CLI binaries (`ferx` and `generate_data`).
//!
//! These run the built binaries as subprocesses. Cargo exposes their paths via
//! the `CARGO_BIN_EXE_<name>` env vars. The `check` subcommand and the data
//! generator are both fast (no model fitting), so these stay in the default
//! test job rather than the slow tier.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// ── generate_data: writes the four example datasets into ./data ──────────────

#[test]
fn generate_data_writes_all_datasets() {
    let tmp = tempfile::tempdir().expect("tempdir");
    // The generator writes to relative `data/<name>.csv`, so it needs a `data`
    // subdirectory in its working directory.
    std::fs::create_dir(tmp.path().join("data")).expect("mkdir data");

    let status = Command::new(env!("CARGO_BIN_EXE_generate_data"))
        .current_dir(tmp.path())
        .status()
        .expect("run generate_data");
    assert!(status.success(), "generate_data exited with {status:?}");

    for name in [
        "warfarin.csv",
        "two_cpt_iv.csv",
        "two_cpt_oral_cov.csv",
        "mm_oral.csv",
    ] {
        let p = tmp.path().join("data").join(name);
        let meta = std::fs::metadata(&p)
            .unwrap_or_else(|e| panic!("expected {} to exist: {e}", p.display()));
        assert!(meta.len() > 0, "{} is empty", p.display());
        // First line should be the NONMEM-style header.
        let contents = std::fs::read_to_string(&p).unwrap();
        let header = contents.lines().next().unwrap_or("");
        assert!(
            header.starts_with("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV"),
            "unexpected header in {}: {header}",
            p.display()
        );
    }
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

// ── ferx fit: full success path (writes sdtab/yaml/timing + .fitrx bundle) ────

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
