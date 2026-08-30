//! Guards for the `ferx-core` / `ferx-tools` / `ferx-cli` API boundary (#1114).
//!
//! The committed baseline at `api/ferx-core-public-api.txt` (A2) is the main
//! enforcement, and CI diffs it via `tools/update-public-api.sh --check`. Two
//! invariants that baseline *cannot* see are pinned here instead.
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
