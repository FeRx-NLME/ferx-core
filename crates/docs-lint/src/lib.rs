//! Structural linter for the Quarto docs under `docs/` (#1163).
//!
//! The docs are written to be read top-to-bottom, but they are addressed a
//! *section at a time* — by the site's own anchor links, and by anything that
//! retrieves one chunk rather than a whole page. Four rules keep that usable:
//!
//! | Rule | Check |
//! |------|-------|
//! | R1 | a section with no subsections stays under 5,000 characters |
//! | R2 | never skip a heading level |
//! | R3 | no two headings on a page may generate the same id |
//! | R4 | internal links must resolve, including the `#anchor` |
//!
//! Nothing here renders Quarto or shells out: the rules read `.qmd` source, and
//! the id algorithm in [`slug`] is pinned to pandoc's and unit-tested against
//! ids taken from a real render.

pub mod baseline;
pub mod parse;
pub mod rules;
pub mod slug;

use std::path::{Path, PathBuf};

pub use baseline::Baseline;
pub use rules::{Config, Rule, Violation, DEFAULT_MAX_CHARS};

/// The repo root, resolved from this crate's location so the linter works from
/// any working directory (`cargo run -p docs-lint` and `cargo test -p
/// docs-lint` both start somewhere different).
///
/// `DOCS_LINT_ROOT` overrides it, which is how the render cross-check can be
/// pointed at another checkout of the repo (a worktree has no `docs/_site`).
pub fn repo_root() -> PathBuf {
    if let Some(root) = std::env::var_os("DOCS_LINT_ROOT") {
        return PathBuf::from(root);
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .unwrap_or(manifest)
        .to_path_buf()
}

/// Default location of the R1 baseline.
pub fn default_baseline_path(root: &Path) -> PathBuf {
    root.join("docs/.lint-baseline")
}

/// Parse every `.qmd` under `docs_dir` and run all four rules.
pub fn lint_tree(docs_dir: &Path, root: &Path, cfg: &Config) -> std::io::Result<Vec<Violation>> {
    let mut docs = Vec::new();
    for path in parse::collect_qmd(docs_dir)? {
        docs.push(parse::parse_file(&path, root)?);
    }
    Ok(rules::lint(&docs, root, cfg))
}
