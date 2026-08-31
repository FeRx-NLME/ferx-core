//! Minimal `.qmd` reader: headings, internal links, and explicit `{#id}`
//! anchors, with YAML front matter and fenced code blocks excluded.
//!
//! This is deliberately not a Markdown parser. It only has to see the things
//! the four rules in [`crate::rules`] are about, and it has to see them the way
//! pandoc does — a `#` inside a code fence is not a heading, and a link inside
//! one is not a link.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::slug::IdAssigner;

#[derive(Debug, Clone)]
pub struct Heading {
    /// 1-based line number of the `#` line.
    pub line: usize,
    pub level: usize,
    pub text: String,
    /// Page-unique id, including any pandoc `-1` / `-2` disambiguation.
    pub id: String,
    /// Id before disambiguation. Differs from `id` only on a collision.
    pub base_id: String,
    /// Rules disabled by a `<!-- lint-disable RN -->` comment above the heading.
    pub disabled: HashSet<String>,
}

#[derive(Debug, Clone)]
pub struct Link {
    /// 1-based line number.
    pub line: usize,
    pub target: String,
}

#[derive(Debug)]
pub struct Doc {
    pub path: PathBuf,
    /// Path relative to the repo root, for reporting.
    pub rel: String,
    /// All lines of the file, 0-indexed.
    pub lines: Vec<String>,
    /// 0-based index of the first body line (after YAML front matter).
    pub body_start: usize,
    pub headings: Vec<Heading>,
    pub links: Vec<Link>,
    /// Every id the page defines — heading ids plus explicit `{#id}` anchors on
    /// divs, callouts and cross-referenced figures.
    pub ids: HashSet<String>,
}

/// True for a line that opens or closes a ``` / ~~~ code fence.
fn is_code_fence(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("```") || t.starts_with("~~~")
}

/// Strip inline code spans so a link inside backticks is not treated as a link.
fn strip_code_spans(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_code = false;
    for ch in line.chars() {
        if ch == '`' {
            in_code = !in_code;
            continue;
        }
        if !in_code {
            out.push(ch);
        }
    }
    out
}

/// Collect `](target)` inline-link targets from one line.
fn links_in_line(line: &str) -> Vec<String> {
    let mut found = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i + 1 < chars.len() {
        if chars[i] == ']' && chars[i + 1] == '(' {
            let mut j = i + 2;
            let mut target = String::new();
            while j < chars.len() && chars[j] != ')' {
                target.push(chars[j]);
                j += 1;
            }
            if j < chars.len() {
                // A title after the target (`](file.qmd "Title")`) is not part
                // of it.
                let target = target.split_whitespace().next().unwrap_or("").to_string();
                if !target.is_empty() {
                    found.push(target);
                }
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    found
}

/// Collect explicit `{#id}` anchors from one line (divs, callouts, figures).
fn explicit_ids_in_line(line: &str) -> Vec<String> {
    let mut found = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i + 1 < chars.len() {
        if chars[i] == '{' && chars[i + 1] == '#' {
            let mut j = i + 2;
            let mut id = String::new();
            while j < chars.len() && chars[j] != '}' && !chars[j].is_whitespace() {
                id.push(chars[j]);
                j += 1;
            }
            if !id.is_empty() {
                found.push(id);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    found
}

/// Rules named by a `<!-- lint-disable R1 R3 -->` comment, if the line is one.
fn disable_directive(line: &str) -> Option<HashSet<String>> {
    let t = line.trim();
    let inner = t.strip_prefix("<!--")?.strip_suffix("-->")?.trim();
    let rest = inner.strip_prefix("lint-disable")?;
    Some(
        rest.split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_ascii_uppercase())
            .collect(),
    )
}

pub fn parse_file(path: &Path, root: &Path) -> std::io::Result<Doc> {
    let text = std::fs::read_to_string(path)?;
    Ok(parse_str(&text, path, root))
}

pub fn parse_str(text: &str, path: &Path, root: &Path) -> Doc {
    let lines: Vec<String> = text.lines().map(str::to_string).collect();
    let rel = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");

    let mut body_start = 0;
    let mut i = 0;
    if lines.first().map(|l| l.trim()) == Some("---") {
        i = 1;
        while i < lines.len() && lines[i].trim() != "---" {
            i += 1;
        }
        i = (i + 1).min(lines.len());
        body_start = i;
    }

    let mut ids = IdAssigner::new();
    let mut headings = Vec::new();
    let mut links = Vec::new();
    let mut all_ids = HashSet::new();
    let mut in_fence = false;
    let mut pending_disable: HashSet<String> = HashSet::new();

    while i < lines.len() {
        let line = &lines[i];
        if is_code_fence(line) {
            in_fence = !in_fence;
            i += 1;
            continue;
        }
        if in_fence {
            i += 1;
            continue;
        }

        if let Some(rules) = disable_directive(line) {
            pending_disable.extend(rules);
            i += 1;
            continue;
        }

        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            let level = trimmed.chars().take_while(|c| *c == '#').count();
            let rest = &trimmed[level..];
            // `#hashtag` is not a heading; `#| label:` is a code-cell option.
            if level <= 6 && rest.starts_with(' ') {
                let raw = rest.trim();
                let (id, base_id) = ids.assign(raw);
                // Report the visible title, not the `{#id}` attribute block.
                let text = crate::slug::split_attributes(raw).0;
                all_ids.insert(id.clone());
                headings.push(Heading {
                    line: i + 1,
                    level,
                    text,
                    id,
                    base_id,
                    disabled: std::mem::take(&mut pending_disable),
                });
                i += 1;
                continue;
            }
        }

        if !trimmed.is_empty() {
            pending_disable.clear();
        }
        for id in explicit_ids_in_line(line) {
            all_ids.insert(id);
        }
        for target in links_in_line(&strip_code_spans(line)) {
            links.push(Link {
                line: i + 1,
                target,
            });
        }
        i += 1;
    }

    Doc {
        path: path.to_path_buf(),
        rel,
        lines,
        body_start,
        headings,
        links,
        ids: all_ids,
    }
}

/// Every `.qmd` under `dir`, excluding the generated `_site/` tree, sorted.
pub fn collect_qmd(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    walk(dir, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if entry.file_type()?.is_dir() {
            // `_site` is generated output; `.quarto` is Quarto's own cache.
            if name == "_site" || name.starts_with('.') {
                continue;
            }
            walk(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "qmd") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(text: &str) -> Doc {
        parse_str(text, Path::new("/root/docs/x.qmd"), Path::new("/root"))
    }

    #[test]
    fn front_matter_is_not_body() {
        let d = doc("---\ntitle: \"# not a heading\"\n---\n\n# Real\n");
        assert_eq!(d.body_start, 3);
        assert_eq!(d.headings.len(), 1);
        assert_eq!(d.headings[0].text, "Real");
        assert_eq!(d.rel, "docs/x.qmd");
    }

    #[test]
    fn headings_inside_fences_are_not_headings() {
        let d = doc("# Real\n\n```bash\n# a shell comment\n```\n\n## Also real\n");
        let texts: Vec<_> = d.headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, ["Real", "Also real"]);
    }

    #[test]
    fn hash_without_space_is_not_a_heading() {
        let d = doc("#| label: fig-x\n#hashtag\n# Real\n");
        assert_eq!(d.headings.len(), 1);
    }

    #[test]
    fn links_are_collected_but_not_from_code() {
        let d = doc("See [a](b.qmd#c) and `[x](y.qmd)`.\n\n```\n[z](w.qmd)\n```\n");
        let targets: Vec<_> = d.links.iter().map(|l| l.target.as_str()).collect();
        assert_eq!(targets, ["b.qmd#c"]);
        assert_eq!(d.links[0].line, 1);
    }

    #[test]
    fn link_title_is_not_part_of_the_target() {
        let d = doc("[a](b.qmd \"Title\")\n");
        assert_eq!(d.links[0].target, "b.qmd");
    }

    #[test]
    fn explicit_div_ids_count_as_page_ids() {
        let d = doc("::: {#tip-warm-start .callout-note}\nhi\n:::\n");
        assert!(d.ids.contains("tip-warm-start"));
    }

    #[test]
    fn disable_comment_attaches_to_the_next_heading() {
        let d = doc("<!-- lint-disable R1 -->\n## Big\n\ntext\n\n## Small\n");
        assert!(d.headings[0].disabled.contains("R1"));
        assert!(d.headings[1].disabled.is_empty());
    }

    #[test]
    fn disable_comment_does_not_leak_past_prose() {
        let d = doc("<!-- lint-disable R1 -->\n\nsome prose\n\n## Big\n");
        assert!(d.headings[0].disabled.is_empty());
    }
}
