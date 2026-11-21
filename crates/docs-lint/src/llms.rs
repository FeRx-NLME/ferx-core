//! R4 for `docs/llms.txt` (#3073).
//!
//! `llms.txt` is the site's index for language models (<https://llmstxt.org>). It
//! is published verbatim at the site root by `docs/_llms.py`, so unlike a `.qmd`
//! page nothing rewrites its links: they have to be **absolute** URLs that a
//! model can fetch from the file alone. That puts every one of them outside
//! R4's reach as written (R4 skips `https://`), so this module maps each URL
//! under the site's `site-url` back to the `.qmd` source it is rendered from
//! and runs the same file-and-anchor check on it.
//!
//! - `<site-url>page.html` and its Markdown twin `<site-url>page.llms.md` both
//!   resolve to `docs/page.qmd`; a directory URL (`…/`) to its `index.qmd`.
//! - `#anchor` must be an id the target page defines, as for any R4 link.
//! - `llms.txt` and `llms-full.txt` are the two files the post-render step
//!   publishes; any other path must exist under `docs/`.
//! - A relative link is a violation: an agent that fetched `llms.txt` by URL
//!   may have nothing to resolve it against.
//! - A URL on any other site is left to `tools/check-external-links.py`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::parse::Doc;
use crate::rules::{normalize, Rule, Violation};

/// The files `docs/_llms.py` writes into the site root, which have no source
/// under `docs/` to resolve to (`llms.txt` does, but is listed for symmetry).
const GENERATED: &[&str] = &["llms.txt", "llms-full.txt"];

/// `website.site-url` from `<docs_dir>/_quarto.yml`, with a trailing slash.
///
/// A line-oriented read rather than a YAML parser, so the crate stays
/// dependency-free; `site-url` is a single scalar on one line.
pub fn site_url(docs_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(docs_dir.join("_quarto.yml")).ok()?;
    site_url_from_config(&text)
}

pub fn site_url_from_config(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let (key, value) = line.trim().split_once(':')?;
        if key.trim() != "site-url" {
            return None;
        }
        let url = value.trim().trim_matches(|c| c == '"' || c == '\'');
        if url.is_empty() {
            return None;
        }
        Some(if url.ends_with('/') { url.to_string() } else { format!("{url}/") })
    })
}

/// Check every link in a parsed `llms.txt`.
///
/// `site` is the site URL with its trailing slash; `ids_by_file` is keyed on
/// normalized `.qmd` paths, as in [`crate::rules::lint`].
pub fn check_llms_links(
    llms: &Doc, docs_dir: &Path, site: &str, ids_by_file: &HashMap<PathBuf, HashSet<String>>,
) -> Vec<Violation> {
    let mut out = Vec::new();
    let mut fail = |line: usize, message: String| {
        out.push(Violation { rule: Rule::R4, file: llms.rel.clone(), line, message, chunk: None });
    };

    for link in &llms.links {
        let target = &link.target;
        let Some(rest) = target.strip_prefix(site) else {
            if !target.contains("://") && !target.starts_with("mailto:") {
                fail(
                    link.line,
                    format!(
                        "link `{target}` is relative — llms.txt is read on its own, so write the absolute URL under `{site}`"
                    ),
                );
            }
            continue;
        };
        let (path_part, anchor) = match rest.split_once('#') {
            Some((p, a)) => (p, Some(a)),
            None => (rest, None),
        };
        if GENERATED.contains(&path_part) {
            continue;
        }

        let source = if path_part.is_empty() || path_part.ends_with('/') {
            format!("{path_part}index.qmd")
        } else if let Some(stem) = path_part.strip_suffix(".llms.md") {
            format!("{stem}.qmd")
        } else if let Some(stem) = path_part.strip_suffix(".html") {
            format!("{stem}.qmd")
        } else {
            path_part.to_string()
        };
        let key = normalize(&docs_dir.join(&source));
        let ids = ids_by_file.get(&key);
        if ids.is_none() && !key.exists() {
            fail(
                link.line,
                format!(
                    "link `{target}` does not resolve — no `{}` under docs/",
                    source.trim_start_matches('/')
                ),
            );
            continue;
        }
        let (Some(anchor), Some(ids)) = (anchor, ids) else { continue };
        if !anchor.is_empty() && !ids.contains(anchor) {
            fail(
                link.line,
                format!(
                    "link `{target}` points at an anchor that does not exist on the target page"
                ),
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_str;

    const SITE: &str = "https://example.org/docs/";

    fn corpus(pages: &[(&str, &str)]) -> HashMap<PathBuf, HashSet<String>> {
        pages
            .iter()
            .map(|(rel, text)| {
                let path = PathBuf::from("/root/docs").join(rel);
                let d = parse_str(text, &path, Path::new("/root"));
                (normalize(&d.path), d.ids)
            })
            .collect()
    }

    fn check(llms: &str, ids: &HashMap<PathBuf, HashSet<String>>) -> Vec<Violation> {
        let d = parse_str(llms, Path::new("/root/docs/llms.txt"), Path::new("/root"));
        check_llms_links(&d, Path::new("/root/docs"), SITE, ids)
    }

    #[test]
    fn site_url_is_read_and_slash_terminated() {
        let cfg = "website:\n  title: x\n  site-url: https://example.org/docs\n";
        assert_eq!(site_url_from_config(cfg).as_deref(), Some(SITE));
        let quoted = "  site-url: \"https://example.org/docs/\"\n";
        assert_eq!(site_url_from_config(quoted).as_deref(), Some(SITE));
        assert_eq!(site_url_from_config("title: x\n"), None);
    }

    #[test]
    fn html_and_markdown_twin_urls_resolve_to_the_qmd_source() {
        let ids = corpus(&[("agents.qmd", "# Agents\n"), ("model-file/index.qmd", "# M\n")]);
        let llms = format!(
            "# x\n\n- [a]({SITE}agents.html)\n- [b]({SITE}agents.llms.md)\n- [c]({SITE}model-file/)\n- [d]({SITE}model-file/index.html)\n"
        );
        assert!(check(&llms, &ids).is_empty(), "{:?}", check(&llms, &ids));
    }

    #[test]
    fn a_page_that_does_not_exist_is_reported_with_its_line() {
        let ids = corpus(&[("agents.qmd", "# Agents\n")]);
        let llms = format!("# x\n\n- [a]({SITE}agents.html)\n- [b]({SITE}nope.html)\n");
        let v = check(&llms, &ids);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].rule, Rule::R4);
        assert_eq!(v[0].line, 4);
        assert!(v[0].message.contains("nope.qmd"), "{}", v[0].message);
    }

    #[test]
    fn the_anchor_is_checked_against_the_target_page() {
        let ids = corpus(&[("output.qmd", "# Output\n\n## Fit status\n")]);
        let good = format!("- [a]({SITE}output.html#fit-status)\n");
        assert!(check(&good, &ids).is_empty());
        let bad = format!("- [a]({SITE}output.html#fit-state)\n");
        let v = check(&bad, &ids);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("#fit-state"));
    }

    #[test]
    fn relative_links_are_refused_but_other_sites_are_not() {
        let ids = corpus(&[("agents.qmd", "# Agents\n")]);
        let llms = "- [a](agents.html)\n- [b](https://elsewhere.org/x.html#y)\n";
        let v = check(llms, &ids);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("relative"));
        assert!(v[0].message.contains("`agents.html`"));
    }

    #[test]
    fn the_generated_root_files_need_no_source() {
        let ids = corpus(&[]);
        let llms = format!("- [a]({SITE}llms.txt)\n- [b]({SITE}llms-full.txt)\n");
        assert!(check(&llms, &ids).is_empty());
    }
}
