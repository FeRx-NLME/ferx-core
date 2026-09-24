//! `AGENTS.md` is the one agent-guidance file (#1528).
//!
//! Before #1528 the repo root carried both a Claude-specific guidance file and an `AGENTS.md` that
//! had been copied from it once (#1114) and never touched again: three weeks
//! later it still described `api.rs` as a single file and the pre-sibling test
//! layout, while every code comment pointed readers at the other copy. Two
//! copies of a rulebook drift exactly the way two copies of a formula do, so
//! these tests pin that there is one, and that nothing points at the old name.
//!
//! Not feature-gated: must run in the base `--features ci` job.

use std::path::PathBuf;
use std::process::Command;

/// Built at runtime so this file does not match its own scan.
fn legacy_name() -> String {
    ["CLAUDE", ".md"].concat()
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The changelog is history: an entry recording the rename names the old file
/// by necessity, and is not a pointer anyone follows.
const HISTORY_FILES: &[&str] = &["CHANGELOG.md"];

/// Every tracked path, as git sees it. Tracked, not walked: a walk also sees
/// untracked and gitignored files that exist only on one machine (local notes,
/// rendered `docs/_site/`, `.claude/`) and follows symlinks out of the checkout,
/// so it would redden locally on a clean tree.
fn tracked_files() -> Vec<String> {
    let out = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(repo_root())
        .output()
        .expect("git on PATH");
    assert!(out.status.success(), "git ls-files failed: {out:?}");
    String::from_utf8(out.stdout)
        .expect("utf-8 paths")
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(str::to_owned)
        .collect()
}

#[test]
fn agents_md_is_the_only_guidance_file() {
    let root = repo_root();
    let agents = std::fs::read_to_string(root.join("AGENTS.md")).expect("AGENTS.md at repo root");
    assert_eq!(
        agents.lines().next(),
        Some("# AGENTS.md"),
        "AGENTS.md must open with its own name as the title"
    );
    assert!(
        !root.join(legacy_name()).exists(),
        "{} is back at the repo root: keep one guidance file, AGENTS.md (#1528)",
        legacy_name()
    );
}

#[test]
fn nothing_points_at_the_legacy_guidance_file() {
    let root = repo_root();
    let needle = legacy_name();
    let files = tracked_files();
    // An empty listing would pass vacuously; the tree has thousands of files.
    assert!(
        files.len() > 100,
        "git ls-files listed only {} files — the listing is broken",
        files.len()
    );
    let mut hits = Vec::new();
    for rel in &files {
        if HISTORY_FILES.contains(&rel.as_str()) {
            continue;
        }
        let path = root.join(rel);
        // A tracked symlink (or a path deleted in the working tree) is not a
        // file to scan; everything else must read, so a read failure cannot
        // hide a stale reference.
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        // Bytes, not `read_to_string`: one invalid UTF-8 byte must not make the
        // rest of the file invisible, and extensionless files (hooks, dotfiles)
        // are scanned like any other.
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {rel}: {e}"));
        for (i, line) in bytes.split(|&b| b == b'\n').enumerate() {
            if line.windows(needle.len()).any(|w| w == needle.as_bytes()) {
                hits.push(format!("{rel}:{}", i + 1));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "{} references to {needle} (renamed to AGENTS.md in #1528):\n{}",
        hits.len(),
        hits.join("\n")
    );
}
