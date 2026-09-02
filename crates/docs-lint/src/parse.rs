//! Minimal `.qmd` reader: headings, internal links, and explicit `{#id}`
//! anchors, with YAML front matter and fenced code blocks excluded.
//!
//! This is deliberately not a Markdown parser. It only has to see the things
//! the four rules in [`crate::rules`] are about, and it has to see them the way
//! pandoc does — a `#` inside a code fence is not a heading, a link inside one
//! is not a link, and the heading opening a callout is that callout's *title*,
//! not a section (#1190).

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

/// The marker character, run length, and "nothing follows the run" flag of a
/// code fence line (a run of `` ` `` or `~`), if the line is one.
///
/// The run length matters: a fence closes only on a run of the *same* marker
/// at least as long as the opener's. Toggling on any fence line makes a nested
/// fence — a four-backtick block quoting a three-backtick one, which is how
/// these docs show markdown — flip parity and expose its own contents to the
/// parser.
fn fence_marker(line: &str) -> Option<(char, usize, bool)> {
    let t = line.trim_start();
    let c = t.chars().next()?;
    if c != '`' && c != '~' {
        return None;
    }
    let n = t.chars().take_while(|&x| x == c).count();
    if n < 3 {
        return None;
    }
    Some((c, n, t.chars().skip(n).all(char::is_whitespace)))
}

/// Columns of leading whitespace. Four or more make the line an indented code
/// block, which is neither a heading nor a div marker.
fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// The colon run length and the attribute text of a fenced-div line (`:::`), if
/// the line is one. A run with nothing after it closes the innermost open div.
fn div_marker(line: &str) -> Option<(usize, String)> {
    let t = line.trim_start();
    if !t.starts_with(':') {
        return None;
    }
    let n = t.chars().take_while(|&c| c == ':').count();
    if n < 3 {
        return None;
    }
    Some((n, t[n..].trim().to_string()))
}

/// Whether a div's attribute text opens a Quarto callout. Both spellings occur
/// in these docs: `::: {.callout-note}` and the bare `::: callout-note`.
fn opens_callout(attrs: &str) -> bool {
    attrs.contains(".callout-")
        || attrs
            .split_whitespace()
            .next()
            .is_some_and(|w| w.starts_with("callout-"))
}

/// One open fenced div.
struct DivFrame {
    callout: bool,
    /// No block-level content has been seen inside this div yet, so a heading
    /// here is the callout title pandoc lifts out of the document.
    awaiting_first_block: bool,
}

/// Record that block-level content was seen inside the innermost open div, so a
/// heading after it is an ordinary heading rather than a callout title.
fn mark_content(divs: &mut [DivFrame]) {
    if let Some(f) = divs.last_mut() {
        f.awaiting_first_block = false;
    }
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
    // The open fence's marker and run length, while we are inside one.
    let mut fence: Option<(char, usize)> = None;
    // Open fenced divs, innermost last.
    let mut divs: Vec<DivFrame> = Vec::new();
    let mut pending_disable: HashSet<String> = HashSet::new();

    while i < lines.len() {
        let line = &lines[i];
        if let Some((c, n, bare)) = fence_marker(line) {
            match fence {
                // Only a bare run of the same marker, at least as long as the
                // opener, closes the block. Anything else is its content.
                Some((open_c, open_n)) => {
                    if c == open_c && n >= open_n && bare {
                        fence = None;
                    }
                }
                None => {
                    mark_content(&mut divs);
                    fence = Some((c, n));
                }
            }
            i += 1;
            continue;
        }
        if fence.is_some() {
            i += 1;
            continue;
        }

        if let Some(rules) = disable_directive(line) {
            pending_disable.extend(rules);
            i += 1;
            continue;
        }

        // A fenced div. Its own `{#id}` still counts as a page id, but the line
        // is not content — what matters is the frame it opens or closes. Four
        // columns of indent make it an indented code block instead, the same
        // guard the heading path below carries: without it a `:::` shown as
        // sample markup pushes a frame that never closes, and a `.callout-*`
        // one swallows the next real heading.
        if let Some((_, attrs)) = div_marker(line).filter(|_| indent_of(line) < 4) {
            if attrs.is_empty() {
                divs.pop();
            } else {
                mark_content(&mut divs);
                divs.push(DivFrame {
                    callout: opens_callout(&attrs),
                    awaiting_first_block: true,
                });
            }
            pending_disable.clear();
            for id in explicit_ids_in_line(line) {
                all_ids.insert(id);
            }
            i += 1;
            continue;
        }

        let trimmed = line.trim_start();
        // Four or more columns of indent is an indented code block, not a
        // heading — pandoc allows at most three. Without the guard a `#`
        // comment inside such a block reads as a heading of its own depth and
        // R2 reports a level jump that does not exist.
        if indent_of(line) < 4 && trimmed.starts_with('#') {
            let level = trimmed.chars().take_while(|c| *c == '#').count();
            let rest = &trimmed[level..];
            // `#hashtag` is not a heading; `#| label:` is a code-cell option.
            if level <= 6 && rest.starts_with(' ') {
                // The first block inside a callout is its title: pandoc lifts
                // it into the callout header, so the rendered page gets neither
                // a section nor an anchor from it. Counting it as a heading
                // chops the callout body into chunks R1 never measures whole,
                // and registers an id R3 and R4 then treat as real (#1190). A
                // *later* heading inside the same callout is a real section —
                // `docs/estimation/saem.qmd` has one — so this is the first
                // block, not any heading in a callout.
                let raw = rest.trim();
                if divs
                    .last()
                    .is_some_and(|f| f.callout && f.awaiting_first_block)
                {
                    // The title still *spends* its identifier. Pandoc's reader
                    // assigns auto-identifiers before Quarto's callout filter
                    // drops the header, so a later `## Dup` on a page whose
                    // callout title was also `Dup` renders as `#dup-1`. Claim
                    // the slot and throw the id away: it addresses nothing in
                    // the rendered page, so it belongs in neither `all_ids`
                    // nor `headings`.
                    let _ = ids.assign(raw);
                    mark_content(&mut divs);
                    pending_disable.clear();
                    i += 1;
                    continue;
                }
                mark_content(&mut divs);
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
            mark_content(&mut divs);
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
    fn a_nested_fence_does_not_flip_fence_parity() {
        // The standard way to show markdown in the docs: a four-backtick block
        // quoting a three-backtick one. Toggling on every fence line would end
        // the outer block early and parse its content as live markdown.
        let d = doc("## S\n\n````markdown\n## S\n\n```bash\necho hi\n```\n````\n\n## T\n");
        let texts: Vec<_> = d.headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, ["S", "T"]);
    }

    #[test]
    fn a_fence_closes_only_on_its_own_marker() {
        // `~~~` inside a ``` block is content, not the closing delimiter.
        let d = doc("```\n~~~\n# not a heading\n~~~\n```\n\n## Real\n");
        let texts: Vec<_> = d.headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, ["Real"]);
    }

    #[test]
    fn hashes_in_an_indented_code_block_are_not_headings() {
        let d = doc("# Top\n\n    ##### deep comment in code\n\n## Real\n");
        let levels: Vec<_> = d.headings.iter().map(|h| h.level).collect();
        assert_eq!(levels, [1, 2]);
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
    fn a_callouts_first_heading_is_its_title_not_a_section() {
        // Pandoc lifts it into the callout header: no section, no anchor. The
        // body it opens belongs to the section above (#1190).
        let d = doc("## Real\n\n::: {.callout-note}\n## Title\n\nbody\n:::\n\n## After\n");
        let texts: Vec<_> = d.headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, ["Real", "After"]);
        assert!(!d.ids.contains("title"));
    }

    #[test]
    fn the_bare_class_spelling_opens_a_callout_too() {
        let d = doc("::: callout-note\n## Title\n:::\n");
        assert!(d.headings.is_empty());
    }

    #[test]
    fn a_later_heading_inside_a_callout_is_a_real_section() {
        // `docs/estimation/saem.qmd` does exactly this: a long
        // `.callout-important` whose title is an `h2` and which then carries a
        // real, explicitly anchored `h3`. Dropping every heading in a callout
        // would delete a section the site really renders.
        let d =
            doc("::: {.callout-important}\n## Title\n\nbody\n\n### Deep {#kept}\n\nmore\n:::\n");
        let texts: Vec<_> = d.headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, ["Deep"]);
        assert_eq!(d.headings[0].id, "kept");
    }

    #[test]
    fn a_heading_after_prose_in_a_callout_is_not_the_title() {
        // The title is the callout's *first block*, which is what pandoc keys
        // on. Prose first means the callout gets its default title and the
        // heading stays a section.
        let d = doc("::: {.callout-note}\nprose first\n\n## Not a title\n:::\n");
        let texts: Vec<_> = d.headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, ["Not a title"]);
    }

    #[test]
    fn a_heading_after_a_code_block_in_a_callout_is_not_the_title() {
        let d = doc("::: {.callout-note}\n```bash\nls\n```\n\n## Not a title\n:::\n");
        let texts: Vec<_> = d.headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, ["Not a title"]);
    }

    #[test]
    fn a_heading_in_a_plain_div_is_a_real_section() {
        // Only callouts lift their first heading; an ordinary div does not.
        let d = doc("::: {.panel-tabset}\n## Tab\n:::\n");
        let texts: Vec<_> = d.headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, ["Tab"]);
    }

    #[test]
    fn a_callout_only_swallows_its_own_first_heading() {
        // After the div closes, the next heading is a section again.
        let d = doc("::: {.callout-note}\n## Title\n:::\n\n## After\n\n::: {.callout-tip}\n## Title2\n:::\n\n## Last\n");
        let texts: Vec<_> = d.headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, ["After", "Last"]);
    }

    #[test]
    fn a_colon_run_inside_a_code_fence_does_not_open_a_div() {
        let d = doc("````markdown\n::: {.callout-note}\n````\n\n## Real\n");
        let texts: Vec<_> = d.headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, ["Real"]);
    }

    #[test]
    fn a_nested_div_does_not_make_its_parents_heading_a_title() {
        let d = doc("::: {.callout-note}\n::: {.inner}\ncontent\n:::\n\n## Not a title\n:::\n");
        let texts: Vec<_> = d.headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, ["Not a title"]);
    }

    #[test]
    fn a_callout_title_spends_its_id_even_though_it_is_not_a_section() {
        // Pandoc's reader assigns auto-identifiers before Quarto's callout
        // filter drops the header, so the slot is gone by the time the real
        // section below asks for it: `quarto render` puts `id="dup-1"` on it.
        // Skipping the `assign` would make R4 bless a link to the dead `#dup`
        // and report the live `#dup-1` as broken.
        let d = doc("::: {.callout-note}\n## Dup\n:::\n\n## Dup\n");
        assert_eq!(d.headings.len(), 1);
        assert_eq!(d.headings[0].id, "dup-1");
        assert_eq!(d.headings[0].base_id, "dup");
        // The title's own id addresses nothing in the rendered page, so it is
        // not a link target …
        assert!(!d.ids.contains("dup"));
        assert!(d.ids.contains("dup-1"));
        // … and it is not an R3 collision either: only one anchor exists.
        assert!(crate::rules::check_duplicate_ids(&d).is_empty());
    }

    #[test]
    fn an_indented_div_marker_is_code_not_a_callout() {
        // Four columns of indent make it an indented code block — sample
        // markup, not a div. A phantom callout frame here would swallow the
        // heading that follows.
        let d = doc("    ::: {.callout-note}\n    Sample markup.\n    :::\n\n## Real\n");
        let texts: Vec<_> = d.headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, ["Real"]);
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
