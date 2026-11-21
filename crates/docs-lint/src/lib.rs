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
//! R4 also covers `docs/llms.txt`, whose links are absolute site URLs (see
//! [`llms`]).
//!
//! Nothing here renders Quarto or shells out: the rules read `.qmd` source, and
//! the id algorithm in [`slug`] is pinned to pandoc's and unit-tested against
//! ids taken from a real render.

pub mod baseline;
pub mod llms;
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
    manifest.parent().and_then(Path::parent).unwrap_or(manifest).to_path_buf()
}

/// Default location of the R1 baseline.
pub fn default_baseline_path(root: &Path) -> PathBuf {
    root.join("docs/.lint-baseline")
}

/// Parse every `.qmd` under `docs_dir` and run all four rules, plus R4 over
/// `docs_dir/llms.txt` when there is one.
pub fn lint_tree(docs_dir: &Path, root: &Path, cfg: &Config) -> std::io::Result<Vec<Violation>> {
    let mut docs = Vec::new();
    for path in parse::collect_qmd(docs_dir)? {
        docs.push(parse::parse_file(&path, root)?);
    }
    let mut out = rules::lint(&docs, root, cfg);

    let llms_path = docs_dir.join("llms.txt");
    if llms_path.is_file() {
        let llms_doc = parse::parse_file(&llms_path, root)?;
        match llms::site_url(docs_dir) {
            Some(site) => {
                let ids_by_file = rules::ids_by_file(&docs);
                out.extend(llms::check_llms_links(&llms_doc, docs_dir, &site, &ids_by_file));
            }
            None => out.push(Violation {
                rule: Rule::R4,
                file: llms_doc.rel.clone(),
                line: 1,
                message:
                    "no `site-url` in _quarto.yml, so llms.txt's absolute links cannot be resolved"
                        .into(),
                chunk: None,
            }),
        }
        out.sort_by(|a, b| a.file.cmp(&b.file).then(a.line.cmp(&b.line)).then(a.rule.cmp(&b.rule)));
    }
    Ok(out)
}
