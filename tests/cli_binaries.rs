//! End-to-end coverage for the `generate_data` maintainer binary.
//!
//! This runs the built binary as a subprocess; cargo exposes its path via the
//! `CARGO_BIN_EXE_<name>` env var. The generator is fast (no model fitting), so
//! it stays in the default test job rather than the slow tier.
//!
//! The `ferx` CLI half of this file moved to `crates/ferx-cli/tests/cli_ferx.rs`
//! when the binary moved to the `ferx-cli` package (#1114):
//! `CARGO_BIN_EXE_ferx` is only defined for test targets of the package that
//! declares the binary, so those tests cannot live here any more.

use std::process::Command;

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

    for name in ["warfarin.csv", "two_cpt_iv.csv", "two_cpt_oral_cov.csv", "mm_oral.csv"] {
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

/// A bare run is the four `data/` datasets only: the opt-in `peak_memory` generator
/// (#3081) reads repository paths and must not run from a temp directory. A name that
/// is neither set is refused with exit 2, listing both sets.
#[test]
fn generate_data_keeps_peak_memory_opt_in_and_refuses_an_unknown_name() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = Command::new(env!("CARGO_BIN_EXE_generate_data"))
        .arg("no_such_dataset")
        .current_dir(tmp.path())
        .output()
        .expect("run generate_data");
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("\"mm_oral\"") && err.contains("\"peak_memory\""), "{err}");
    assert!(!tmp.path().join("tools").exists(), "an unknown name wrote files");
}
