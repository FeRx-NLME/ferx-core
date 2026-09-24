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

use std::path::{Path, PathBuf};

/// Built at runtime so this file does not match its own scan.
fn legacy_name() -> String {
    ["CLAUDE", ".md"].concat()
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Directories that are generated, vendored or per-machine, never authored.
const SKIP_DIRS: &[&str] = &["target", ".git", ".claude", "_site", "node_modules"];

/// Extensions of the files that carry prose a reader follows.
const TEXT_EXTS: &[&str] = &[
    "rs", "md", "qmd", "toml", "yml", "yaml", "sh", "R", "ctl", "py", "txt",
];

fn text_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable directory") {
        let path = entry.expect("readable dir entry").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            if !SKIP_DIRS.contains(&name) {
                text_files(&path, out);
            }
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| TEXT_EXTS.contains(&e))
        {
            out.push(path);
        }
    }
}

#[test]
fn agents_md_is_the_only_guidance_file() {
    let root = repo_root();
    let agents = std::fs::read_to_string(root.join("AGENTS.md")).expect("AGENTS.md at repo root");
    assert!(
        agents.starts_with("# AGENTS.md\n"),
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
    let mut files = Vec::new();
    text_files(&root, &mut files);
    // A walk that found nothing would pass vacuously; the tree has hundreds.
    assert!(
        files.len() > 100,
        "scan found only {} files — the walk is broken",
        files.len()
    );
    let mut hits = Vec::new();
    for path in &files {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            if line.contains(&needle) {
                let rel = path.strip_prefix(&root).unwrap_or(path);
                hits.push(format!("{}:{}", rel.display(), i + 1));
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
