//! Guard: CI never reads `rust-toolchain.toml`'s `components` list (#2637).
//!
//! `rust-toolchain.toml` lists `rust-analyzer`, `clippy` and `rustfmt`, so a fresh
//! checkout gets find-references in agent sessions. Honoured, that list is a hazard for a
//! fresh-nightly CI install: when a component is missing from the day's nightly, rustup
//! either fails or walks back to an older nightly that has it. CI is immune only because
//! every job that touches the Rust toolchain sets `RUSTUP_TOOLCHAIN` (workflow- or
//! job-level), and an explicit `RUSTUP_TOOLCHAIN` makes rustup ignore the file entirely,
//! components included. Measured with rustup 1.29.1: a toolchain file naming a component
//! that does not exist runs `cargo --version` cleanly with `RUSTUP_TOOLCHAIN=nightly`
//! set, and triggers a download without it.
//!
//! So this test fails if a workflow job runs cargo / rustc / rustup, a script that does,
//! or an action that does (`rust-cache`, `audit-check`) without `RUSTUP_TOOLCHAIN` in
//! scope. A new action that shells out to cargo needs adding to `USES_MARKERS`.

use serde_json::Value;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Does this step reach the Rust toolchain (and so, without an override, the file)?
fn step_uses_toolchain(step: &Value) -> bool {
    const RUN_MARKERS: [&str; 6] = [
        "cargo ",
        "rustc ",
        "rustup ",
        "tools/preflight.sh",
        "tools/update-public-api.sh",
        "tools/nightly-mutants.sh",
    ];
    // Actions that call cargo / rustc themselves: `audit-check` runs `cargo install
    // cargo-audit` and `cargo audit`, `rust-cache` runs `rustc -vV` for its key.
    const USES_MARKERS: [&str; 2] = ["Swatinem/rust-cache", "rustsec/audit-check"];
    let run = step["run"].as_str().unwrap_or("");
    let uses = step["uses"].as_str().unwrap_or("");
    RUN_MARKERS.iter().any(|m| run.contains(m)) || USES_MARKERS.iter().any(|m| uses.contains(m))
}

fn sets_toolchain(env: &Value) -> bool {
    env.get("RUSTUP_TOOLCHAIN").is_some()
}

#[test]
fn every_toolchain_job_overrides_rust_toolchain_toml() {
    let dir = root().join(".github/workflows/");
    let mut checked = Vec::new();
    let mut missing = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read .github/workflows") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("yml") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(&path).expect("read workflow");
        let yml: Value = serde_saphyr::from_str(&text)
            .unwrap_or_else(|e| panic!("{name} does not parse as YAML: {e}"));
        let workflow_level = sets_toolchain(&yml["env"]);
        let jobs = yml["jobs"]
            .as_object()
            .unwrap_or_else(|| panic!("{name} has no `jobs:` mapping"));
        for (job, body) in jobs {
            let steps = body["steps"].as_array().map(Vec::as_slice).unwrap_or(&[]);
            if !steps.iter().any(step_uses_toolchain) {
                continue;
            }
            checked.push(format!("{name}:{job}"));
            if !workflow_level && !sets_toolchain(&body["env"]) {
                missing.push(format!("{name}:{job}"));
            }
        }
    }
    // Non-vacuous: the detector must actually see the cargo jobs (ci.yml alone has a
    // dozen). A marker list that matched nothing would pass the check below trivially.
    assert!(
        checked.iter().any(|j| j == "ci.yml:check") && checked.len() >= 10,
        "the toolchain-step detector found too few jobs to be trusted: {checked:?}"
    );
    assert!(
        missing.is_empty(),
        "these jobs reach the Rust toolchain without `RUSTUP_TOOLCHAIN`, so rustup would \
         honour rust-toolchain.toml and its `components` list (a missing nightly component \
         fails or silently rolls back the install): {missing:?}"
    );
}

#[test]
fn the_detector_sees_a_job_without_the_override() {
    // The negative side of the check above: a job that runs cargo and sets no
    // `RUSTUP_TOOLCHAIN` is detected, and the override is what clears it.
    let bare: Value = serde_saphyr::from_str("steps:\n  - run: cargo test\n").unwrap();
    let steps = bare["steps"].as_array().unwrap();
    assert!(steps.iter().any(step_uses_toolchain));
    assert!(!sets_toolchain(&bare["env"]));
    let pinned: Value = serde_saphyr::from_str("env:\n  RUSTUP_TOOLCHAIN: nightly\n").unwrap();
    assert!(sets_toolchain(&pinned["env"]));
    let audit: Value =
        serde_saphyr::from_str("steps:\n  - uses: rustsec/audit-check@v2.0.0\n").unwrap();
    assert!(audit["steps"]
        .as_array()
        .unwrap()
        .iter()
        .any(step_uses_toolchain));
    let quiet: Value = serde_saphyr::from_str("steps:\n  - run: quarto render docs\n").unwrap();
    assert!(!quiet["steps"]
        .as_array()
        .unwrap()
        .iter()
        .any(step_uses_toolchain));
}

#[test]
fn rust_toolchain_toml_installs_rust_analyzer() {
    let text = std::fs::read_to_string(root().join("rust-toolchain.toml"))
        .expect("read rust-toolchain.toml");
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("components"))
        .expect("rust-toolchain.toml has a `components` list");
    for c in ["rust-analyzer", "clippy", "rustfmt"] {
        assert!(
            line.contains(&format!("\"{c}\"")),
            "`components` lacks {c}: {line}"
        );
    }
}
