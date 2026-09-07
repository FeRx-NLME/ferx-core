//! The R1 baseline: a checked-in, **shrink-only** list of the oversized
//! sections that predate the linter.
//!
//! `main` had 14 of them when #1163 landed, some of them 20 kB+ tables that
//! cannot be split in the same PR as the linter. The baseline lets R1 be a hard
//! failure for everything new while those are burned down page by page.
//!
//! Shrink-only means three things, and the third is what keeps it honest:
//!
//! - a chunk that is not in the baseline and is over the limit fails;
//! - a baselined chunk that has *grown* fails;
//! - a baselined chunk that no longer violates fails too, as a stale entry —
//!   so fixing a section forces its line out of the file instead of leaving a
//!   permanent exemption behind.

use std::collections::BTreeMap;
use std::path::Path;

use crate::rules::Violation;

/// `chars` recorded per chunk key, keyed as `docs/page.qmd::section-id`.
#[derive(Debug, Default, Clone)]
pub struct Baseline {
    pub entries: BTreeMap<String, usize>,
}

pub const HEADER: &str = "\
# R1 baseline — oversized sections that predate the docs linter (#1163).
#
# Format: `R1 <characters> <docs/page.qmd::section-id>`
#
# SHRINK-ONLY. A new violation fails CI, a baselined section that grew fails CI,
# and a section that no longer violates fails CI as a stale entry — remove its
# line in the PR that splits it. Regenerate with:
#
#     cargo run -p docs-lint -- --update-baseline
";

impl Baseline {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut entries = BTreeMap::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let rule = parts.next().unwrap_or_default();
            let chars = parts.next().unwrap_or_default();
            let key = parts.next().unwrap_or_default();
            if rule != "R1" || key.is_empty() {
                return Err(format!(
                    "baseline line {}: expected `R1 <chars> <key>`, got `{line}`",
                    n + 1
                ));
            }
            let chars: usize = chars.parse().map_err(|_| {
                format!(
                    "baseline line {}: `{chars}` is not a character count",
                    n + 1
                )
            })?;
            entries.insert(key.to_string(), chars);
        }
        Ok(Self { entries })
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(&text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    pub fn render(&self) -> String {
        let mut out = String::from(HEADER);
        for (key, chars) in &self.entries {
            out.push_str(&format!("R1 {chars} {key}\n"));
        }
        out
    }

    /// Build a baseline from the R1 violations of a run.
    pub fn from_violations(violations: &[Violation]) -> Self {
        let mut entries = BTreeMap::new();
        for v in violations {
            if let Some(chunk) = &v.chunk {
                entries.insert(chunk.key.clone(), chunk.chars);
            }
        }
        Self { entries }
    }
}

/// The outcome of filtering a run's violations through the baseline.
pub struct Filtered {
    /// Violations that still fail: everything not R1, plus R1 chunks that are
    /// unbaselined or have grown.
    pub failing: Vec<Violation>,
    /// R1 violations excused by an entry, for reporting.
    pub baselined: usize,
    /// Baseline entries with nothing left to excuse.
    pub stale: Vec<String>,
}

pub fn apply(baseline: &Baseline, violations: Vec<Violation>) -> Filtered {
    let mut failing = Vec::new();
    let mut baselined = 0;
    let mut matched: Vec<&String> = Vec::new();

    for v in violations {
        let Some(chunk) = v.chunk.clone() else {
            failing.push(v);
            continue;
        };
        match baseline.entries.get_key_value(&chunk.key) {
            Some((key, &allowed)) => {
                matched.push(key);
                if chunk.chars > allowed {
                    let mut v = v;
                    v.message = format!(
                        "{} — baselined at {allowed} characters, now {}; it may only shrink",
                        v.message, chunk.chars
                    );
                    failing.push(v);
                } else {
                    baselined += 1;
                }
            }
            None => failing.push(v),
        }
    }

    let stale = baseline
        .entries
        .keys()
        .filter(|k| !matched.contains(k))
        .cloned()
        .collect();

    Filtered {
        failing,
        baselined,
        stale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{Chunk, ChunkKind, Rule};

    fn r1(key: &str, chars: usize) -> Violation {
        Violation {
            rule: Rule::R1,
            file: key.split("::").next().unwrap().to_string(),
            line: 1,
            message: "too long".into(),
            chunk: Some(Chunk {
                key: key.to_string(),
                chars,
                kind: ChunkKind::Terminal,
            }),
        }
    }

    fn other() -> Violation {
        Violation {
            rule: Rule::R4,
            file: "docs/x.qmd".into(),
            line: 9,
            message: "dead anchor".into(),
            chunk: None,
        }
    }

    fn baseline_of(pairs: &[(&str, usize)]) -> Baseline {
        Baseline {
            entries: pairs.iter().map(|(k, c)| (k.to_string(), *c)).collect(),
        }
    }

    #[test]
    fn a_baselined_chunk_at_the_same_size_is_excused() {
        let f = apply(
            &baseline_of(&[("docs/x.qmd::a", 9000)]),
            vec![r1("docs/x.qmd::a", 9000)],
        );
        assert!(f.failing.is_empty());
        assert_eq!(f.baselined, 1);
        assert!(f.stale.is_empty());
    }

    #[test]
    fn a_shrunk_but_still_oversized_chunk_is_excused() {
        let f = apply(
            &baseline_of(&[("docs/x.qmd::a", 9000)]),
            vec![r1("docs/x.qmd::a", 8000)],
        );
        assert!(f.failing.is_empty());
    }

    #[test]
    fn a_grown_chunk_fails_and_says_by_how_much() {
        let f = apply(
            &baseline_of(&[("docs/x.qmd::a", 9000)]),
            vec![r1("docs/x.qmd::a", 9001)],
        );
        assert_eq!(f.failing.len(), 1);
        assert!(f.failing[0].message.contains("baselined at 9000"));
        assert!(f.failing[0].message.contains("now 9001"));
    }

    #[test]
    fn an_unbaselined_chunk_fails() {
        let f = apply(&baseline_of(&[]), vec![r1("docs/x.qmd::a", 5001)]);
        assert_eq!(f.failing.len(), 1);
    }

    #[test]
    fn a_fixed_chunk_leaves_a_stale_entry() {
        let f = apply(&baseline_of(&[("docs/x.qmd::a", 9000)]), vec![]);
        assert_eq!(f.stale, vec!["docs/x.qmd::a".to_string()]);
    }

    #[test]
    fn the_baseline_never_excuses_another_rule() {
        let f = apply(&baseline_of(&[("docs/x.qmd::a", 9000)]), vec![other()]);
        assert_eq!(f.failing.len(), 1);
        assert_eq!(f.failing[0].rule, Rule::R4);
    }

    #[test]
    fn render_round_trips_through_parse() {
        let b = baseline_of(&[("docs/x.qmd::a", 9000), ("docs/y.qmd::<preamble>", 5100)]);
        let reparsed = Baseline::parse(&b.render()).unwrap();
        assert_eq!(reparsed.entries, b.entries);
    }

    #[test]
    fn a_malformed_line_is_an_error_not_a_silent_skip() {
        assert!(Baseline::parse("R2 10 docs/x.qmd::a").is_err());
        assert!(Baseline::parse("R1 lots docs/x.qmd::a").is_err());
        assert!(Baseline::parse("R1 10").is_err());
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let b = Baseline::parse("# hi\n\nR1 10 docs/x.qmd::a\n").unwrap();
        assert_eq!(b.entries.len(), 1);
    }
}
