//! `docs-lint` — structural linter for `docs/**/*.qmd` (#1163).
//!
//! ```text
//! cargo run -p docs-lint                      # report violations, exit 1 if any
//! cargo run -p docs-lint -- --check           # what CI runs (identical, quieter)
//! cargo run -p docs-lint -- --json            # machine-readable
//! cargo run -p docs-lint -- --update-baseline # re-record the R1 baseline
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use docs_lint::baseline::{self, Baseline};
use docs_lint::rules::{Config, Violation};
use docs_lint::{default_baseline_path, lint_tree, repo_root, DEFAULT_MAX_CHARS};

const USAGE: &str = "\
docs-lint — structural linter for the Quarto docs (#1163)

USAGE:
    docs-lint [OPTIONS] [DOCS_DIR]

OPTIONS:
    --check              Same checks and exit code as the default run; prints
                         only the violations and a one-line summary.
    --json               Emit violations as JSON.
    --max-chars <N>      Override the R1 limit (default: 5000).
    --baseline <PATH>    R1 baseline file (default: docs/.lint-baseline).
    --no-baseline        Ignore the baseline; report every R1 violation.
    --update-baseline    Rewrite the baseline from this run and exit 0.
    -h, --help           Show this help.

RULES:
    R1  a section with no subsections stays under the character limit
    R2  never skip a heading level
    R3  no two headings on a page may generate the same id
    R4  internal links must resolve, including the anchor

Silence one rule for one section with `<!-- lint-disable R1 -->` on the line
above its heading, and say in the comment why it cannot be split.
";

struct Args {
    docs_dir: Option<PathBuf>,
    check: bool,
    json: bool,
    max_chars: usize,
    baseline_path: Option<PathBuf>,
    use_baseline: bool,
    update_baseline: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        docs_dir: None,
        check: false,
        json: false,
        max_chars: DEFAULT_MAX_CHARS,
        baseline_path: None,
        use_baseline: true,
        update_baseline: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--check" => args.check = true,
            "--json" => args.json = true,
            "--no-baseline" => args.use_baseline = false,
            "--update-baseline" => args.update_baseline = true,
            "--max-chars" => {
                let v = it.next().ok_or("--max-chars needs a value")?;
                args.max_chars = v.parse().map_err(|_| format!("bad --max-chars: {v}"))?;
            }
            "--baseline" => {
                args.baseline_path =
                    Some(PathBuf::from(it.next().ok_or("--baseline needs a path")?));
            }
            other if other.starts_with('-') => return Err(format!("unknown option: {other}")),
            other => args.docs_dir = Some(PathBuf::from(other)),
        }
    }
    Ok(args)
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn print_json(violations: &[Violation], stale: &[String]) {
    println!("{{");
    println!("  \"violations\": [");
    for (i, v) in violations.iter().enumerate() {
        let comma = if i + 1 == violations.len() { "" } else { "," };
        println!(
            "    {{\"rule\": \"{}\", \"file\": \"{}\", \"line\": {}, \"message\": \"{}\"}}{comma}",
            v.rule.as_str(),
            escape_json(&v.file),
            v.line,
            escape_json(&v.message)
        );
    }
    println!("  ],");
    println!("  \"stale_baseline_entries\": [");
    for (i, k) in stale.iter().enumerate() {
        let comma = if i + 1 == stale.len() { "" } else { "," };
        println!("    \"{}\"{comma}", escape_json(k));
    }
    println!("  ]");
    println!("}}");
}

fn run() -> Result<ExitCode, String> {
    let args = parse_args()?;
    let root = repo_root();
    let docs_dir = args.docs_dir.clone().unwrap_or_else(|| root.join("docs"));
    if !docs_dir.is_dir() {
        return Err(format!("no docs directory at {}", docs_dir.display()));
    }

    let cfg = Config {
        max_chars: args.max_chars,
    };
    let violations =
        lint_tree(&docs_dir, &root, &cfg).map_err(|e| format!("{}: {e}", docs_dir.display()))?;

    let baseline_path = args
        .baseline_path
        .clone()
        .unwrap_or_else(|| default_baseline_path(&root));

    if args.update_baseline {
        let new = Baseline::from_violations(&violations);
        let n = new.entries.len();
        std::fs::write(&baseline_path, new.render())
            .map_err(|e| format!("{}: {e}", baseline_path.display()))?;
        println!(
            "wrote {} R1 entr{} to {}",
            n,
            if n == 1 { "y" } else { "ies" },
            baseline_path.display()
        );
        return Ok(ExitCode::SUCCESS);
    }

    let filtered = if args.use_baseline {
        baseline::apply(&Baseline::load(&baseline_path)?, violations)
    } else {
        baseline::Filtered {
            failing: violations,
            baselined: 0,
            stale: Vec::new(),
        }
    };

    if args.json {
        print_json(&filtered.failing, &filtered.stale);
    } else {
        for v in &filtered.failing {
            println!("{}:{}: {}: {}", v.file, v.line, v.rule.as_str(), v.message);
        }
        for key in &filtered.stale {
            println!(
                "{}: R1: no longer over the limit — remove this line from {}",
                key,
                baseline_path
                    .strip_prefix(&root)
                    .unwrap_or(&baseline_path)
                    .display()
            );
        }
        let failures = filtered.failing.len() + filtered.stale.len();
        if failures == 0 {
            if !args.check {
                println!("docs-lint: clean");
            }
            if filtered.baselined > 0 {
                println!(
                    "docs-lint: {} R1 section(s) still excused by the baseline",
                    filtered.baselined
                );
            }
        } else {
            eprintln!("docs-lint: {failures} violation(s)");
        }
    }

    Ok(
        if filtered.failing.is_empty() && filtered.stale.is_empty() {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        },
    )
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("docs-lint: {e}");
            eprintln!();
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
