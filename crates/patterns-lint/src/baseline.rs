//! The ratchet baseline: a checked-in, **shrink-only** count of the findings
//! that predate a rule — same contract as `docs/.lint-baseline` (#1163).
//!
//! Keyed per *function* (`tests/foo.rs::tests::my_test`), not per line, so an
//! unrelated edit that shifts line numbers does not churn it, and not per file,
//! so fixing one fold and adding another in the same file is still a failure.
//!
//! - a finding in a function with no entry fails;
//! - a function with more findings than its entry fails (every finding in it is
//!   printed, since which one is new cannot be told apart);
//! - an entry whose function now has *fewer* findings fails too, as stale — so
//!   paying a site down forces the count in this file down with it, instead of
//!   leaving headroom a later regression could spend.

use std::collections::BTreeMap;
use std::path::Path;

use crate::Finding;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Baseline {
    /// `(rule, key) -> count`.
    pub entries: BTreeMap<(String, String), usize>,
}

pub const HEADER: &str = "\
# patterns-lint ratchet baseline — test-code findings that predate a rule (#2634).
#
# Format: `<rule> <count> <file::fn::path>`
#
# SHRINK-ONLY. A new finding fails CI, a function whose count grew fails CI, and
# a function whose count fell fails CI as a stale entry, so paying a site down
# has to lower its line here. Pay it down rather than grow it: fix the fold, or
# use an escape hatch (see tools/lint-rules/nan-fold.md). Regenerate after a
# paydown with:
#
#     cargo run -p patterns-lint -- --update-baseline
#
# which only ever lowers or removes entries.
";

impl Baseline {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut entries = BTreeMap::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.splitn(3, ' ');
            let (Some(rule), Some(count), Some(key)) = (parts.next(), parts.next(), parts.next())
            else {
                return Err(format!(
                    "baseline line {}: expected `<rule> <count> <key>`, got `{line}`",
                    n + 1
                ));
            };
            let count: usize = count.parse().ok().filter(|&c| c > 0).ok_or_else(|| {
                format!("baseline line {}: `{count}` is not a positive count", n + 1)
            })?;
            if entries
                .insert((rule.to_string(), key.trim().to_string()), count)
                .is_some()
            {
                return Err(format!(
                    "baseline line {}: duplicate entry `{rule} {key}`",
                    n + 1
                ));
            }
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
        for ((rule, key), count) in &self.entries {
            out.push_str(&format!("{rule} {count} {key}\n"));
        }
        out
    }

    /// Exact current counts, as a baseline.
    pub fn from_findings(findings: &[Finding]) -> Self {
        let mut entries = BTreeMap::new();
        for f in findings {
            *entries
                .entry((f.rule.to_string(), f.key.clone()))
                .or_insert(0) += 1;
        }
        Self { entries }
    }

    /// The shrink-only update: every entry becomes `min(old, current)`, entries
    /// that reach zero are dropped, and nothing is ever added or raised.
    pub fn shrunk_to(&self, findings: &[Finding]) -> Self {
        let current = Self::from_findings(findings);
        let entries = self
            .entries
            .iter()
            .filter_map(|(k, &old)| {
                let now = current.entries.get(k).copied().unwrap_or(0);
                (now.min(old) > 0).then(|| (k.clone(), now.min(old)))
            })
            .collect();
        Self { entries }
    }
}

/// A stale entry: the function now has `now < recorded` findings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stale {
    pub rule: String,
    pub key: String,
    pub recorded: usize,
    pub now: usize,
}

#[derive(Debug, Default)]
pub struct Applied {
    pub failing: Vec<Finding>,
    pub stale: Vec<Stale>,
    pub baselined: usize,
}

pub fn apply(baseline: &Baseline, findings: Vec<Finding>) -> Applied {
    let current = Baseline::from_findings(&findings);
    let mut out = Applied::default();
    for f in findings {
        let k = (f.rule.to_string(), f.key.clone());
        let now = current.entries[&k];
        match baseline.entries.get(&k) {
            Some(&recorded) if now <= recorded => out.baselined += 1,
            _ => out.failing.push(f),
        }
    }
    for ((rule, key), &recorded) in &baseline.entries {
        let now = current
            .entries
            .get(&(rule.clone(), key.clone()))
            .copied()
            .unwrap_or(0);
        if now < recorded {
            out.stale.push(Stale {
                rule: rule.clone(),
                key: key.clone(),
                recorded,
                now,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(key: &str, line: usize) -> Finding {
        Finding {
            rule: "nan-fold",
            file: key.split("::").next().unwrap().to_string(),
            line,
            key: key.to_string(),
            message: String::new(),
            snippet: String::new(),
        }
    }

    #[test]
    fn round_trips_and_rejects_malformed_lines() {
        let b = Baseline::from_findings(&[f("a.rs::t", 1), f("a.rs::t", 2), f("b.rs::u", 3)]);
        assert_eq!(Baseline::parse(&b.render()).unwrap(), b);
        assert!(Baseline::parse("nan-fold 0 a.rs::t").is_err());
        assert!(Baseline::parse("nan-fold x a.rs::t").is_err());
        assert!(Baseline::parse("nan-fold 1").is_err());
        assert!(Baseline::parse("nan-fold 1 a.rs::t\nnan-fold 2 a.rs::t").is_err());
    }

    #[test]
    fn growth_fails_every_finding_in_the_function_and_new_functions_fail() {
        let b = Baseline::parse("nan-fold 1 a.rs::t").unwrap();
        let r = apply(&b, vec![f("a.rs::t", 1)]);
        assert!(r.failing.is_empty() && r.stale.is_empty());
        assert_eq!(r.baselined, 1);

        let r = apply(&b, vec![f("a.rs::t", 1), f("a.rs::t", 9)]);
        assert_eq!(r.failing.len(), 2, "which of the two is new cannot be told");

        let r = apply(&b, vec![f("a.rs::t", 1), f("a.rs::other", 4)]);
        assert_eq!(r.failing.len(), 1);
        assert_eq!(r.failing[0].key, "a.rs::other");
    }

    #[test]
    fn a_paid_down_entry_is_stale_until_the_baseline_shrinks() {
        let b = Baseline::parse("nan-fold 2 a.rs::t\nnan-fold 1 gone.rs::x").unwrap();
        let findings = vec![f("a.rs::t", 1)];
        let r = apply(&b, findings.clone());
        assert!(r.failing.is_empty());
        assert_eq!(r.stale.len(), 2);

        let shrunk = b.shrunk_to(&findings);
        assert_eq!(shrunk, Baseline::parse("nan-fold 1 a.rs::t").unwrap());
        let r = apply(&shrunk, findings);
        assert!(r.failing.is_empty() && r.stale.is_empty());
    }

    #[test]
    fn shrinking_never_grows_or_adds() {
        let b = Baseline::parse("nan-fold 1 a.rs::t").unwrap();
        let shrunk = b.shrunk_to(&[f("a.rs::t", 1), f("a.rs::t", 2), f("new.rs::n", 3)]);
        assert_eq!(shrunk, b);
    }
}
