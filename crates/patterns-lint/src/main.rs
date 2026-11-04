//! `patterns-lint` — source-pattern lints over the workspace's test code (#2634).
//!
//! ```text
//! cargo run -p patterns-lint                      # report, exit 1 on any failure
//! cargo run -p patterns-lint -- --no-baseline     # every finding, baselined or not
//! cargo run -p patterns-lint -- --update-baseline # shrink the baseline after a paydown
//! ```
//!
//! The CI gate is `tools/preflight.sh patterns`, which runs the same check as
//! `cargo test -p patterns-lint` (`tests/patterns_corpus.rs`).

use std::path::PathBuf;
use std::process::ExitCode;

use patterns_lint::baseline::{self, Baseline};
use patterns_lint::{default_baseline_path, lint_tree, render_report, repo_root};

const USAGE: &str = "\
patterns-lint — source-pattern lints over the workspace's test code (#2634)

USAGE:
    patterns-lint [OPTIONS]

OPTIONS:
    --root <DIR>         Repo root to scan (default: this checkout).
    --no-baseline        Ignore the baseline; report every finding.
    --update-baseline    Lower or remove baseline entries that were paid down.
                         Never adds or raises one.
    --record-baseline    Rewrite the baseline to exactly the current findings,
                         growth included. Only for the PR that introduces a
                         rule; anywhere else, fix the fold or use an escape
                         hatch — the growth is a reviewable diff either way.
    --explain <RULE>     Print a rule's full text.
    -h, --help           Show this help.

RULES:
    nan-fold   a max/min fold in test code, which discards NaN

Scope: tests/**, crates/*/tests/**, src/**/*_tests.rs, src/**/tests/**, files
reached through a #[cfg(test)] `mod`, and #[cfg(test)] / #[test] items anywhere.
Rule files and the ratchet baseline live in tools/lint-rules/.
";

const NAN_FOLD_DOC: &str = include_str!("../../../tools/lint-rules/nan-fold.md");

fn main() -> ExitCode {
    let mut root = repo_root();
    let mut use_baseline = true;
    let mut update = false;
    let mut record = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--no-baseline" => use_baseline = false,
            "--update-baseline" => update = true,
            "--record-baseline" => record = true,
            "--root" => match args.next() {
                Some(r) => root = PathBuf::from(r),
                None => return usage_error("--root needs a directory"),
            },
            "--explain" => {
                return match args.next().as_deref() {
                    Some("nan-fold") => {
                        print!("{NAN_FOLD_DOC}");
                        ExitCode::SUCCESS
                    }
                    other => usage_error(&format!("unknown rule {other:?} (known: nan-fold)")),
                }
            }
            other => return usage_error(&format!("unknown argument `{other}`")),
        }
    }

    let findings = match lint_tree(&root) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("patterns-lint: {e}");
            return ExitCode::from(2);
        }
    };
    let path = default_baseline_path(&root);
    let base = if use_baseline {
        match Baseline::load(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("patterns-lint: {e}");
                return ExitCode::from(2);
            }
        }
    } else {
        Baseline::default()
    };

    if update || record {
        let shrunk = if record {
            Baseline::from_findings(&findings)
        } else {
            base.shrunk_to(&findings)
        };
        if let Err(e) = std::fs::write(&path, shrunk.render()) {
            eprintln!("patterns-lint: {}: {e}", path.display());
            return ExitCode::from(2);
        }
        println!(
            "patterns-lint: baseline {} → {} entr(y/ies) in {}",
            base.entries.len(),
            shrunk.entries.len(),
            path.display()
        );
    }

    let applied = baseline::apply(
        &if update || record {
            Baseline::load(&path).unwrap_or_default()
        } else {
            base
        },
        findings,
    );
    let report = render_report(&applied);
    print!("{report}");
    if report.is_empty() {
        println!(
            "patterns-lint: clean ({} baselined finding(s) remain to pay down)",
            applied.baselined
        );
        ExitCode::SUCCESS
    } else {
        println!(
            "patterns-lint: {} failing finding(s), {} stale baseline entr(y/ies)",
            applied.failing.len(),
            applied.stale.len()
        );
        ExitCode::FAILURE
    }
}

fn usage_error(msg: &str) -> ExitCode {
    eprintln!("patterns-lint: {msg}\n\n{USAGE}");
    ExitCode::from(2)
}
