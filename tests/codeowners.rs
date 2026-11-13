//! Guard: `.github/CODEOWNERS` keeps covering the numerical core (#2636).
//!
//! The estimation, sensitivity, statistics, PK and ODE directories are where a defect is a
//! wrong-but-plausible number, so they are owned and review on them requires a code owner.
//! GitHub does not complain about a CODEOWNERS pattern that matches nothing: rename
//! `src/sens/` and its ownership quietly lapses. The tests below pin that
//!
//! - every directory in [`NUMERICAL_CORE`] has an ownership line with at least one owner;
//! - every path pattern in the file names a directory that exists, so a stale line fails.

use std::path::PathBuf;

const NUMERICAL_CORE: &[&str] = &[
    "/src/estimation/",
    "/src/sens/",
    "/src/stats/",
    "/src/pk/",
    "/src/ode/",
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `(pattern, owners)` for every non-comment, non-blank line.
fn rules() -> Vec<(String, Vec<String>)> {
    let text = std::fs::read_to_string(root().join(".github/CODEOWNERS"))
        .expect(".github/CODEOWNERS must exist");
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut fields = l.split_whitespace();
            let pattern = fields.next().unwrap().to_string();
            (pattern, fields.map(str::to_string).collect())
        })
        .collect()
}

#[test]
fn numerical_core_directories_have_owners() {
    let rules = rules();
    for dir in NUMERICAL_CORE {
        let owners = rules
            .iter()
            .find(|(p, _)| p == dir)
            .map(|(_, o)| o)
            .unwrap_or_else(|| panic!("{dir} has no CODEOWNERS line"));
        assert!(!owners.is_empty(), "{dir} is listed without an owner");
        assert!(
            owners.iter().all(|o| o.starts_with('@')),
            "{dir} owners must be @handles, got {owners:?}"
        );
    }
}

#[test]
fn every_codeowners_pattern_names_an_existing_directory() {
    for (pattern, _) in rules() {
        let rel = pattern.trim_start_matches('/');
        assert!(
            root().join(rel).is_dir(),
            "CODEOWNERS pattern {pattern} matches no directory — renamed or deleted?"
        );
    }
}
