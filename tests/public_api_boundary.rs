//! Guards for the `ferx-core` / `ferx-tools` / `ferx-cli` split (#1114).
//!
//! The committed baseline at `api/ferx-core-public-api.txt` (A2) is the main
//! enforcement of the API boundary, and CI diffs it via
//! `tools/update-public-api.sh --check`. The invariants that baseline *cannot*
//! see are pinned here instead — plus one that has nothing to do with the API
//! surface but everything to do with the split: the checked-in commands that
//! invoke the `ferx` binary.
//!
//! Not feature-gated: these must run in the base `--features ci` job, the one
//! job guaranteed to build on every PR.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable directory") {
        let path = entry.expect("readable dir entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// **A3 — no `#[doc(hidden)] pub` escape hatch.**
///
/// This is not style policing: `cargo public-api` *omits* `#[doc(hidden)]`
/// items entirely, so marking a new `pub fn` hidden makes it invisible to the
/// A2 baseline diff. Verified directly — adding `#[doc(hidden)] pub fn probe()`
/// to `src/lib.rs` leaves `--check` green, while the same `pub fn` without the
/// attribute fails it with exactly one added line. So the attribute is a
/// working bypass of the boundary gate, and the only place to close it is here.
///
/// An item is either public API — documented, in the baseline, usable from
/// ferx-r — or it stays `pub(crate)` and the caller does without.
#[test]
fn no_doc_hidden_escape_hatch() {
    let mut sources = Vec::new();
    rust_sources(&repo_root().join("src"), &mut sources);
    assert!(!sources.is_empty(), "found no sources under src/");

    let mut offenders = Vec::new();
    for path in &sources {
        let src = std::fs::read_to_string(path).expect("source file is valid UTF-8");
        for (i, line) in src.lines().enumerate() {
            if line.contains("#[doc(hidden)]") {
                offenders.push(format!(
                    "{}:{}",
                    path.strip_prefix(repo_root()).unwrap_or(path).display(),
                    i + 1
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "`#[doc(hidden)]` hides an item from the `api/ferx-core-public-api.txt` \
         baseline, so it is a silent bypass of the A2 API gate (#1114 A3). \
         Make the item genuinely public (document it and regenerate the \
         baseline with `tools/update-public-api.sh`) or make it `pub(crate)`. \
         Found at: {offenders:?}"
    );
}

/// **A1 — one-way dependency.** `ferx-core` never depends on `ferx-tools` or
/// `ferx-cli`.
///
/// Cargo would reject an actual cycle, but not every violation is a cycle: a
/// `dev-dependency` on `ferx-cli` compiles fine and would quietly make the core
/// test suite depend on the layer above it, which is exactly the coupling the
/// split exists to prevent.
#[test]
fn ferx_core_does_not_depend_on_the_layers_above_it() {
    let manifest =
        std::fs::read_to_string(repo_root().join("Cargo.toml")).expect("root Cargo.toml is UTF-8");
    // Trim the `[workspace]` table — it names the members by path on purpose.
    let package_section = manifest
        .split_once("\n[package]")
        .map(|(_, rest)| rest)
        .expect("root manifest has a [package] section");

    for forbidden in ["ferx-tools", "ferx-cli"] {
        assert!(
            !package_section.contains(forbidden),
            "the `ferx-core` package must not depend on `{forbidden}` in any \
             dependency table (#1114 A1) — the dependency is strictly one-way"
        );
    }
}

// ── Checked-in `ferx` CLI invocations ───────────────────────────────────────

/// Directories that are build output, VCS metadata, or vendored — never sources
/// of instructions a reader would copy.
const SKIPPED_DIRS: [&str; 4] = ["target", ".git", "_site", "node_modules"];

/// Text files that can plausibly carry a shell command a reader would run.
const SCANNED_EXTENSIONS: [&str; 6] = ["md", "qmd", "sh", "ferx", "R", "rs"];

fn scannable_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable directory") {
        let path = entry.expect("readable dir entry").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            if !SKIPPED_DIRS.contains(&name) {
                scannable_files(&path, out);
            }
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| SCANNED_EXTENSIONS.contains(&e))
        {
            out.push(path);
        }
    }
}

/// **The `ferx` binary lives in `ferx-cli`, so every checked-in invocation of it
/// must say so.**
///
/// Moving the binary out of the root package silently invalidated ~53 commands
/// spread across `examples/*.ferx` headers, `nonmem_anchor/`, `plans/`,
/// `tests/reference/`, and the executable harnesses under `tools/`. They are
/// documentation and shell scripts, so nothing compiled them and nothing failed:
/// a build naming the root-package bin now dies with *"no bin target named
/// `ferx` in default-run packages"*, and a package-less `cargo run --release`
/// passing a model through `--` dies the same way. The first sweep of this PR
/// caught the `docs/` copies and missed every other tree, which is why the
/// invariant is pinned in a test rather than trusted to a grep done once.
///
/// Two failure shapes are checked:
///
/// * a run/build that reaches the CLI without selecting its package, and
/// * a `ferx-cli`-scoped command passing a bare `--features ci|survival|…`,
///   which fails with *"none of the selected packages contains this feature"*
///   because those features belong to `ferx-core` and need qualifying.
///
/// Invocations of the binaries that genuinely stayed in `ferx-core`
/// (`generate_data`, `pktte_sim_anchor`, `rtte_sim_anchor`) name their own
/// `--bin` and are left alone.
#[test]
fn checked_in_ferx_invocations_select_the_ferx_cli_package() {
    let mut files = Vec::new();
    scannable_files(&repo_root(), &mut files);
    assert!(!files.is_empty(), "found no scannable files");

    // Needles are built with `format!` rather than written literally so this
    // file does not match itself — the same trick
    // `ci_workflow_endpoint_coverage.rs` uses, and what lets the doc comment
    // above describe the broken commands without becoming an offender.
    let cli_pkg = format!("-p {}", "ferx-cli");
    let root_bin = format!("--bin {}", "ferx");
    let core_features = ["ci", "survival", "markov", "nn", "slow-tests"];

    let mut offenders = Vec::new();
    for path in &files {
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        for (i, line) in src.lines().enumerate() {
            if !(line.contains("cargo run") || line.contains("cargo build")) {
                continue;
            }
            // The root-package `ferx` bin, and not a longer name that merely
            // starts with it.
            let names_root_bin = line
                .split_once(root_bin.as_str())
                .is_some_and(|(_, rest)| !rest.starts_with('-') && !rest.starts_with('_'));
            // A run that passes arguments through `--` but never selects a
            // package. `--bin <other>` excuses it: that is one of the bins still
            // in core.
            let unscoped_run = line.contains(" -- ")
                && !line.contains(cli_pkg.as_str())
                && !line.contains("-p ferx-core")
                && !line.contains("--bin ");
            // Feature flags belong to `ferx-core`, so a member-scoped command
            // has to qualify them.
            let unqualified_feature = line.contains(cli_pkg.as_str())
                && core_features.iter().any(|f| {
                    line.contains(&format!("--features {f}"))
                        || line.contains(&format!("--features={f}"))
                });

            if names_root_bin || unscoped_run || unqualified_feature {
                offenders.push(format!(
                    "{}:{}: {}",
                    path.strip_prefix(repo_root()).unwrap_or(path).display(),
                    i + 1,
                    line.trim()
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "these checked-in commands no longer run: the `ferx` binary moved to the \
         `ferx-cli` package (#1114), so a run/build must select it with \
         `-p ferx-cli`, and any `ferx-core` feature must be written qualified \
         (`--features ferx-core/survival`). Offenders:\n  {}",
        offenders.join("\n  ")
    );
}
