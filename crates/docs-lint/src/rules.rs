//! The four rules (#1163).
//!
//! R1 — a section with no subsections stays under [`DEFAULT_MAX_CHARS`].
//! R2 — never skip a heading level.
//! R3 — no two headings on a page may generate the same id.
//! R4 — internal links must resolve, including the `#anchor`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::parse::Doc;

/// ~2,000 tokens at this corpus's ~2.4 characters per token. Characters, not
/// tokens, so the linter needs no tokenizer, no network and no model.
pub const DEFAULT_MAX_CHARS: usize = 5_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rule {
    R1,
    R2,
    R3,
    R4,
}

impl Rule {
    pub fn as_str(self) -> &'static str {
        match self {
            Rule::R1 => "R1",
            Rule::R2 => "R2",
            Rule::R3 => "R3",
            Rule::R4 => "R4",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Rule::R1 => "a section with no subsections stays under the character limit",
            Rule::R2 => "never skip a heading level",
            Rule::R3 => "no two headings on a page may generate the same id",
            Rule::R4 => "internal links must resolve, including the anchor",
        }
    }
}

/// Which kind of independently-addressable chunk R1 measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKind {
    /// A section with no subsections: retrieved whole, or not at all.
    Terminal,
    /// The prose above a page's first heading.
    Preamble,
    /// The prose a parent section keeps above its own first subheading.
    ParentProse,
}

impl ChunkKind {
    fn label(self) -> &'static str {
        match self {
            ChunkKind::Terminal => "terminal section",
            ChunkKind::Preamble => "page preamble",
            ChunkKind::ParentProse => "prose above the first subsection of",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Violation {
    pub rule: Rule,
    pub file: String,
    pub line: usize,
    pub message: String,
    /// Set for R1 only: the stable key and size used by the baseline.
    pub chunk: Option<Chunk>,
}

/// An oversized R1 chunk, keyed by something that does not move when a line is
/// inserted above it.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub key: String,
    pub chars: usize,
    pub kind: ChunkKind,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub max_chars: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_MAX_CHARS,
        }
    }
}

fn chunk_chars(doc: &Doc, from_line_idx: usize, to_line_idx: usize) -> usize {
    if from_line_idx >= to_line_idx {
        return 0;
    }
    doc.lines[from_line_idx..to_line_idx]
        .join("\n")
        .trim()
        .chars()
        .count()
}

/// R1: every independently-addressable chunk stays under the limit.
pub fn check_sizes(doc: &Doc, cfg: &Config) -> Vec<Violation> {
    let mut out = Vec::new();
    let hs = &doc.headings;

    // The prose above the first heading.
    let first_heading_idx = hs.first().map(|h| h.line - 1).unwrap_or(doc.lines.len());
    let pre = chunk_chars(doc, doc.body_start, first_heading_idx);
    if pre > cfg.max_chars {
        out.push(Violation {
            rule: Rule::R1,
            file: doc.rel.clone(),
            line: doc.body_start + 1,
            message: format!(
                "{} is {pre} characters, over the {} limit — split it under subheadings",
                ChunkKind::Preamble.label(),
                cfg.max_chars
            ),
            chunk: Some(Chunk {
                key: format!("{}::<preamble>", doc.rel),
                chars: pre,
                kind: ChunkKind::Preamble,
            }),
        });
    }

    for (k, h) in hs.iter().enumerate() {
        let next = hs.get(k + 1);
        let end = next.map(|n| n.line - 1).unwrap_or(doc.lines.len());
        let is_terminal = next.is_none_or(|n| n.level <= h.level);
        let (kind, chars) = if is_terminal {
            (ChunkKind::Terminal, chunk_chars(doc, h.line, end))
        } else {
            (ChunkKind::ParentProse, chunk_chars(doc, h.line, end))
        };
        if chars <= cfg.max_chars || h.disabled.contains("R1") {
            continue;
        }
        out.push(Violation {
            rule: Rule::R1,
            file: doc.rel.clone(),
            line: h.line,
            message: format!(
                "{} \"{}\" is {chars} characters, over the {} limit — split it under subheadings",
                kind.label(),
                h.text,
                cfg.max_chars
            ),
            chunk: Some(Chunk {
                key: format!("{}::{}", doc.rel, h.id),
                chars,
                kind,
            }),
        });
    }
    out
}

/// R2: never skip a heading level.
pub fn check_levels(doc: &Doc) -> Vec<Violation> {
    let mut out = Vec::new();
    let mut prev: Option<usize> = None;
    for h in &doc.headings {
        if let Some(p) = prev {
            if h.level > p + 1 && !h.disabled.contains("R2") {
                out.push(Violation {
                    rule: Rule::R2,
                    file: doc.rel.clone(),
                    line: h.line,
                    message: format!(
                        "heading level jumps from h{p} to h{} at \"{}\" — the h{} attaches to the wrong parent",
                        h.level, h.text, h.level
                    ),
                    chunk: None,
                });
            }
        }
        prev = Some(h.level);
    }
    out
}

/// R3: no two headings on a page may generate the same id.
pub fn check_duplicate_ids(doc: &Doc) -> Vec<Violation> {
    let mut first_use: HashMap<&str, usize> = HashMap::new();
    let mut out = Vec::new();
    for h in &doc.headings {
        match first_use.get(h.base_id.as_str()) {
            None => {
                first_use.insert(&h.base_id, h.line);
            }
            Some(&first) => {
                if h.disabled.contains("R3") {
                    continue;
                }
                out.push(Violation {
                    rule: Rule::R3,
                    file: doc.rel.clone(),
                    line: h.line,
                    message: format!(
                        "\"{}\" repeats the id `#{}` first used on line {first}, so it is addressed as `#{}` — make the titles distinct or set an explicit {{#id}}",
                        h.text, h.base_id, h.id
                    ),
                    chunk: None,
                });
            }
        }
    }
    out
}

/// R4: internal links must resolve, including the anchor.
///
/// `ids_by_file` maps an absolute `.qmd` path to the ids that page defines.
pub fn check_links(
    doc: &Doc,
    root: &Path,
    ids_by_file: &HashMap<PathBuf, HashSet<String>>,
) -> Vec<Violation> {
    let mut out = Vec::new();
    let dir = doc.path.parent().unwrap_or(root);

    for link in &doc.links {
        let target = &link.target;
        if target.starts_with("http://")
            || target.starts_with("https://")
            || target.starts_with("mailto:")
            || target.starts_with("data:")
            || target.starts_with('{')
        {
            continue;
        }
        let (path_part, anchor) = match target.split_once('#') {
            Some((p, a)) => (p, Some(a)),
            None => (target.as_str(), None),
        };

        let key = if path_part.is_empty() {
            // `ids_by_file` is keyed on the NORMALIZED path, so this side has to
            // normalize too. Keying on the raw path made every same-page
            // `#anchor` check silently pass whenever the docs directory was
            // given with a `.` or `..` component (`docs-lint ./docs`).
            normalize(&doc.path)
        } else {
            let mut candidate = normalize(&dir.join(path_part));
            // Quarto links are written against the source tree, but a link may
            // point at the rendered `.html`; resolve that back to its source.
            if !candidate.exists() && candidate.extension().is_some_and(|e| e == "html") {
                candidate.set_extension("qmd");
            }
            if !candidate.exists() {
                out.push(Violation {
                    rule: Rule::R4,
                    file: doc.rel.clone(),
                    line: link.line,
                    message: format!("link target `{target}` does not exist"),
                    chunk: None,
                });
                continue;
            }
            candidate
        };

        let Some(anchor) = anchor else { continue };
        if anchor.is_empty() {
            continue;
        }
        // Only `.qmd` pages have ids we can check; a link into an asset or a
        // non-Quarto file is left alone.
        let Some(ids) = ids_by_file.get(&key) else {
            continue;
        };
        if !ids.contains(anchor) {
            out.push(Violation {
                rule: Rule::R4,
                file: doc.rel.clone(),
                line: link.line,
                message: format!(
                    "link `{target}` points at an anchor that does not exist on the target page"
                ),
                chunk: None,
            });
        }
    }
    out
}

/// Lexical path normalization (`..`/`.`), without touching the filesystem.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            c => out.push(c.as_os_str()),
        }
    }
    out
}

/// Run every rule over a parsed corpus, sorted by file then line.
pub fn lint(docs: &[Doc], root: &Path, cfg: &Config) -> Vec<Violation> {
    let ids_by_file: HashMap<PathBuf, HashSet<String>> = docs
        .iter()
        .map(|d| (normalize(&d.path), d.ids.clone()))
        .collect();

    let mut out = Vec::new();
    for doc in docs {
        out.extend(check_sizes(doc, cfg));
        out.extend(check_levels(doc));
        out.extend(check_duplicate_ids(doc));
        out.extend(check_links(doc, root, &ids_by_file));
    }
    out.sort_by(|a, b| {
        a.file
            .cmp(&b.file)
            .then(a.line.cmp(&b.line))
            .then(a.rule.cmp(&b.rule))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_str;

    fn doc_at(text: &str, rel: &str) -> Doc {
        let path = PathBuf::from("/root").join(rel);
        parse_str(text, &path, Path::new("/root"))
    }

    fn doc(text: &str) -> Doc {
        doc_at(text, "docs/x.qmd")
    }

    fn small_cfg() -> Config {
        Config { max_chars: 50 }
    }

    #[test]
    fn terminal_section_over_the_limit_is_reported() {
        let text = format!("# Top\n\n{}\n", "x".repeat(80));
        let v = check_sizes(&doc(&text), &small_cfg());
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].line, 1);
        assert_eq!(v[0].chunk.as_ref().unwrap().kind, ChunkKind::Terminal);
        assert_eq!(v[0].chunk.as_ref().unwrap().key, "docs/x.qmd::top");
    }

    #[test]
    fn subdividing_fixes_it_without_deleting_a_word() {
        let long = "x".repeat(40);
        let text = format!("# Top\n\n## A\n\n{long}\n\n## B\n\n{long}\n");
        assert!(check_sizes(&doc(&text), &small_cfg()).is_empty());
    }

    #[test]
    fn parent_prose_above_the_first_subsection_counts() {
        let text = format!("# Top\n\n{}\n\n## Child\n\nshort\n", "x".repeat(80));
        let v = check_sizes(&doc(&text), &small_cfg());
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].chunk.as_ref().unwrap().kind, ChunkKind::ParentProse);
    }

    #[test]
    fn page_preamble_counts() {
        let text = format!(
            "---\ntitle: x\n---\n\n{}\n\n# Top\n\nshort\n",
            "x".repeat(80)
        );
        let v = check_sizes(&doc(&text), &small_cfg());
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].chunk.as_ref().unwrap().kind, ChunkKind::Preamble);
        assert_eq!(v[0].chunk.as_ref().unwrap().key, "docs/x.qmd::<preamble>");
    }

    #[test]
    fn a_section_ends_at_the_next_heading_of_any_level() {
        // The h3 belongs to "B", not to "A" — "A" is terminal and short.
        let text = format!("## A\n\nshort\n\n## B\n\n### C\n\n{}\n", "x".repeat(80));
        let v = check_sizes(&doc(&text), &small_cfg());
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("\"C\""));
    }

    #[test]
    fn r1_disable_comment_is_honoured() {
        let text = format!("<!-- lint-disable R1 -->\n# Top\n\n{}\n", "x".repeat(80));
        assert!(check_sizes(&doc(&text), &small_cfg()).is_empty());
    }

    #[test]
    fn skipped_level_is_reported_but_a_climb_back_is_not() {
        let d = doc("# Top\n\n### Deep\n\n## Back\n\n### Fine\n");
        let v = check_levels(&d);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].line, 3);
        assert!(v[0].message.contains("h1 to h3"));
    }

    #[test]
    fn duplicate_ids_are_reported_from_the_second_occurrence() {
        let d = doc("## Syntax\n\n## Syntax\n\n## Syntax\n");
        let v = check_duplicate_ids(&d);
        assert_eq!(v.len(), 2);
        assert!(v[0].message.contains("`#syntax-1`"));
        assert!(v[1].message.contains("`#syntax-2`"));
    }

    #[test]
    fn distinct_titles_under_different_parents_are_fine() {
        let d = doc("## Weibull\n\n### Syntax — Weibull\n\n## Transit\n\n### Syntax — transit\n");
        assert!(check_duplicate_ids(&d).is_empty());
    }

    #[test]
    fn explicit_ids_resolve_a_collision() {
        let d = doc("## Syntax {#syntax-weibull}\n\n## Syntax\n");
        assert!(check_duplicate_ids(&d).is_empty());
    }

    fn link_corpus(pages: &[(&str, &str)]) -> (Vec<Doc>, PathBuf) {
        let root = PathBuf::from("/root");
        let docs = pages
            .iter()
            .map(|(rel, text)| doc_at(text, rel))
            .collect::<Vec<_>>();
        (docs, root)
    }

    fn check_anchors(docs: &[Doc], root: &Path) -> Vec<Violation> {
        // Keyed exactly as `lint()` keys it, so the helper cannot pass on a
        // corpus the real entry point would fail.
        let ids: HashMap<PathBuf, HashSet<String>> = docs
            .iter()
            .map(|d| (normalize(&d.path), d.ids.clone()))
            .collect();
        docs.iter()
            .flat_map(|d| check_links(d, root, &ids))
            .collect()
    }

    #[test]
    fn same_page_anchor_must_exist() {
        let (docs, root) = link_corpus(&[(
            "docs/x.qmd",
            "See [here](#no-such-thing) and [there](#real-one).\n\n## Real one\n",
        )]);
        let v = check_anchors(&docs, &root);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("#no-such-thing"));
    }

    #[test]
    fn numbered_heading_anchor_is_dead() {
        // `## 3. Communication` renders as `#communication`.
        let (docs, root) = link_corpus(&[(
            "docs/x.qmd",
            "[a](#3-communication) [b](#communication)\n\n## 3. Communication\n",
        )]);
        let v = check_anchors(&docs, &root);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("#3-communication"));
    }

    #[test]
    fn em_dash_anchor_takes_one_hyphen_not_two() {
        let (docs, root) = link_corpus(&[("docs/x.qmd", "[one](#a-b) [two](#a--b)\n\n## A — b\n")]);
        let v = check_anchors(&docs, &root);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("#a--b"));
    }

    #[test]
    fn explicit_div_anchor_resolves() {
        let (docs, root) = link_corpus(&[(
            "docs/x.qmd",
            "[a](#tip-x)\n\n::: {#tip-x .callout-note}\nhi\n:::\n",
        )]);
        assert!(check_anchors(&docs, &root).is_empty());
    }

    #[test]
    fn anchorless_and_external_links_are_left_alone() {
        let (docs, root) = link_corpus(&[(
            "docs/x.qmd",
            "[a](https://example.com/#x) [b](mailto:a@b.c) [c](#)\n",
        )]);
        assert!(check_anchors(&docs, &root).is_empty());
    }

    #[test]
    fn a_missing_file_target_is_reported_once() {
        // No filesystem entry for `nope.qmd`, so this is a dead file target and
        // the anchor is not also reported.
        let (docs, root) = link_corpus(&[("docs/x.qmd", "[a](nope.qmd#anchor)\n")]);
        let v = check_anchors(&docs, &root);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("does not exist"));
    }

    #[test]
    fn same_page_anchors_are_still_checked_when_the_path_is_unnormalized() {
        // `docs-lint ./docs` reaches every page as `/root/./docs/x.qmd`. The
        // lookup key has to be normalized on BOTH sides or the same-page branch
        // misses `ids_by_file` entirely and every `#anchor` silently passes.
        let doc = parse_str(
            "[a](#no-such-thing)\n\n## Real one\n",
            Path::new("/root/./docs/x.qmd"),
            Path::new("/root"),
        );
        let docs = [doc];
        let v = lint(&docs, Path::new("/root"), &Config::default());
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].rule, Rule::R4);
        assert!(v[0].message.contains("#no-such-thing"));
    }

    #[test]
    fn normalize_resolves_parent_components() {
        assert_eq!(
            normalize(Path::new("/root/docs/model-file/../faq.qmd")),
            PathBuf::from("/root/docs/faq.qmd")
        );
    }
}
