//! Guard: every committed NONMEM control stream has a row in the anchor catalogue (#1024).
//!
//! `nonmem_anchor/` is committed so the next person can tell what already has an external
//! anchor without re-running NONMEM. That only works if the anchors can be found, and the
//! index is the table at the top of `nonmem_anchor/README.md`. When #1024 was filed, 11 of
//! 44 control streams were reachable from no index at all — discovering that #993 had a
//! committed NONMEM pair required already knowing its test file — and by the time it was
//! fixed, 100 of 246 were missing from the table, because nothing failed when a PR added
//! a `.ctl` without a row.
//!
//! What the tests below pin, all against the **control-stream column** of that first table
//! (a `.ctl` named in another column's prose is not an index entry — the catalogue had rows
//! quoting `tie.ctl` and `ss_inf.ctl`, names the `.lst` echoes, not files on disk):
//!
//! - every `*.ctl` under `nonmem_anchor/`, subdirectories included and `results/` (copies of
//!   what NONMEM echoed) excluded, is matched by some row's pattern;
//! - every pattern matches at least one file, so a renamed or deleted stream cannot leave
//!   a row pointing at nothing;
//! - every row has exactly four cells. GitHub splits a table cell on a bare `|` even inside
//!   a code span, so an unescaped `|Δ|` silently shifts the columns: five rows did, which
//!   both broke their rendering and moved their control streams out of the column read
//!   here.
//!
//! Patterns are the backticked spans ending in `.ctl` in that column, with `{a,b}`
//! alternation and a `*` that does not cross `/` — the spellings the table already used.

use std::path::{Path, PathBuf};

fn anchor_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("nonmem_anchor");
    p
}

/// The body rows of the first table in the README (the header and the `|---|` rule dropped).
fn catalogue_rows(readme: &str) -> Vec<&str> {
    let rows: Vec<&str> = readme
        .lines()
        .skip_while(|l| !l.starts_with('|'))
        .take_while(|l| l.starts_with('|'))
        .collect();
    assert!(
        rows.len() > 2,
        "nonmem_anchor/README.md has no catalogue table"
    );
    rows[2..].to_vec()
}

/// A row's cells, split on every `|` not escaped as `\|`.
fn cells(row: &str) -> Vec<&str> {
    let trimmed = row.trim();
    let inner = trimmed
        .strip_prefix('|')
        .and_then(|r| r.strip_suffix('|'))
        .unwrap_or_else(|| panic!("catalogue row does not start and end with `|`: {row}"));
    let mut out = Vec::new();
    let mut start = 0;
    let bytes = inner.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'|' && (i == 0 || bytes[i - 1] != b'\\') {
            out.push(&inner[start..i]);
            start = i + 1;
        }
    }
    out.push(&inner[start..]);
    out
}

/// `a_{x,y}.ctl` -> `a_x.ctl`, `a_y.ctl`; nested and repeated groups expand left to right.
fn expand_braces(pat: &str) -> Vec<String> {
    let Some(open) = pat.find('{') else {
        return vec![pat.to_string()];
    };
    let close = open
        + pat[open..]
            .find('}')
            .unwrap_or_else(|| panic!("unbalanced `{{` in catalogue pattern `{pat}`"));
    pat[open + 1..close]
        .split(',')
        .flat_map(|alt| expand_braces(&format!("{}{alt}{}", &pat[..open], &pat[close + 1..])))
        .collect()
}

/// Glob match where `*` is any run of characters other than `/`.
fn glob_match(pat: &str, path: &str) -> bool {
    match pat.split_once('*') {
        None => pat == path,
        Some((head, tail)) => {
            let Some(rest) = path.strip_prefix(head) else {
                return false;
            };
            (0..=rest.len())
                .take_while(|&i| !rest[..i].contains('/'))
                .filter(|&i| rest.is_char_boundary(i))
                .any(|i| glob_match(tail, &rest[i..]))
        }
    }
}

/// The patterns in the control-stream column (the third cell) of every row.
fn catalogue_patterns(readme: &str) -> Vec<String> {
    let mut pats = Vec::new();
    for row in catalogue_rows(readme) {
        let c = cells(row);
        // Spans alternate outside/inside backticks; the odd-indexed pieces are code.
        for span in c[2].split('`').skip(1).step_by(2) {
            if span.ends_with(".ctl") {
                pats.extend(expand_braces(span));
            }
        }
    }
    pats
}

/// Every `*.ctl` under `nonmem_anchor/`, relative to it with `/` separators, `results/` skipped.
fn control_streams() -> Vec<String> {
    fn walk(dir: &Path, rel: &str, out: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).expect("read nonmem_anchor dir") {
            let entry = entry.expect("dir entry");
            let name = entry.file_name().into_string().expect("utf-8 file name");
            let path = entry.path();
            let rel_name = if rel.is_empty() {
                name.clone()
            } else {
                format!("{rel}/{name}")
            };
            if path.is_dir() {
                if name != "results" {
                    walk(&path, &rel_name, out);
                }
            } else if name.ends_with(".ctl") {
                out.push(rel_name);
            }
        }
    }
    let mut out = Vec::new();
    walk(&anchor_dir(), "", &mut out);
    out.sort();
    out
}

fn readme() -> String {
    std::fs::read_to_string(anchor_dir().join("README.md")).expect("read nonmem_anchor/README.md")
}

#[test]
fn every_catalogue_row_has_four_cells() {
    let readme = readme();
    let bad: Vec<String> = catalogue_rows(&readme)
        .into_iter()
        .filter(|r| cells(r).len() != 4)
        .map(|r| format!("  {} cells: {}", cells(r).len(), &r[..r.len().min(100)]))
        .collect();
    assert!(
        bad.is_empty(),
        "nonmem_anchor/README.md catalogue rows without exactly 4 cells — escape a `|` inside \
         a cell as `\\|` (GitHub splits on it even inside a code span):\n{}",
        bad.join("\n")
    );
}

#[test]
fn every_control_stream_has_a_catalogue_row() {
    let pats = catalogue_patterns(&readme());
    let missing: Vec<String> = control_streams()
        .into_iter()
        .filter(|f| !pats.iter().any(|p| glob_match(p, f)))
        .collect();
    assert!(
        missing.is_empty(),
        "control streams under nonmem_anchor/ that no catalogue row names — add a row to the \
         table at the top of nonmem_anchor/README.md (anchor, ferx feature + issue, control \
         stream(s), ferx model / test file):\n  {}",
        missing.join("\n  ")
    );
}

#[test]
fn every_catalogue_pattern_names_a_control_stream() {
    let files = control_streams();
    let dead: Vec<String> = catalogue_patterns(&readme())
        .into_iter()
        .filter(|p| !files.iter().any(|f| glob_match(p, f)))
        .collect();
    assert!(
        dead.is_empty(),
        "nonmem_anchor/README.md catalogue names control streams that do not exist under \
         nonmem_anchor/ (renamed or deleted?):\n  {}",
        dead.join("\n  ")
    );
}

/// The matcher is what the two tests above stand on: one that matched everything would
/// pass the first on an empty table, so its negative cases are pinned here.
#[test]
fn pattern_helpers_match_only_what_they_spell() {
    assert_eq!(
        expand_braces("ss_lag_{ge_ii,infusion}{,_flatwt}.ctl"),
        [
            "ss_lag_ge_ii.ctl",
            "ss_lag_ge_ii_flatwt.ctl",
            "ss_lag_infusion.ctl",
            "ss_lag_infusion_flatwt.ctl"
        ]
    );
    assert!(glob_match(
        "dose_cmt_*.ctl",
        "dose_cmt_advan2_bolus_cmt2.ctl"
    ));
    assert!(glob_match(
        "residual_correlation/*.ctl",
        "residual_correlation/cs_r0.ctl"
    ));
    assert!(!glob_match("dose_cmt_*.ctl", "dose_attr_double_use_A.ctl"));
    assert!(!glob_match("*.ctl", "residual_correlation/cs_r0.ctl"));
    assert!(!glob_match("residual_correlation/*.ctl", "cs_r0.ctl"));
    assert!(!glob_match("ss_mm.ctl", "ss_mm.ctl.bak"));
    assert_eq!(
        cells(r"| a | `x \| y` | c.ctl | d |"),
        [" a ", r" `x \| y` ", " c.ctl ", " d "]
    );
}
