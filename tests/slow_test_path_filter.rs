//! Guard: the `push:` path filter in `.github/workflows/slow-tests.yml` must match every
//! input the heavy fit suite actually consumes.
//!
//! That filter decides whether a push to `main` re-runs the 5–60 minute fit suite. It has
//! two silent failure modes, and this repo has hit both:
//!
//! 1. **A dead glob.** A listed path is deleted, renamed, or split, leaving a pattern that
//!    matches nothing. `src/api.rs` was named in the filter, the file became `src/api/`,
//!    and the push leg went dark for months — because a stale glob and a live one are
//!    indistinguishable when you read the diff (#949).
//! 2. **An uncovered input.** A module, fixture directory, or data file the fits read is
//!    never added to the list. `src/pk`, `src/survival`, `examples/`, and `nonmem_anchor/`
//!    were all in this state.
//!
//! An "every listed glob matches something" check catches only (1). It passes happily on a
//! filter narrowed to a single directory, which is the general form of (2). So the
//! invariant pinned here is *coverage*, asserted in both directions:
//!
//! > every tracked file under an input root the suite reads is matched by the filter,
//! > and every pattern in the filter matches at least one tracked file.
//!
//! ## Why the input roots are derived, not listed
//!
//! Hard-coding the expected roots here would just mirror the workflow: change both, and the
//! test agrees with a filter that is wrong. Instead the roots come from the tree itself:
//!
//! - **Compile inputs** are mechanical — `cargo test` builds `src/` and `tests/` and is
//!   configured by `Cargo.toml` / `Cargo.lock`.
//! - **Runtime inputs** are recovered from the test sources: every repo-relative path
//!   literal in `tests/**/*.rs` (`"data/warfarin.csv"`, `"examples/adaptive_vanco_auc.ferx"`,
//!   `"nonmem_anchor/transit_iov.csv"`) contributes its top-level root.
//! - **The job's own definition**, since changing the feature set or cargo invocation
//!   changes what runs.
//!
//! That makes the guard self-maintaining in the direction that matters: add a test reading
//! `fixtures/foo.csv` and this fails until `fixtures/**` is in the filter. Nobody has to
//! remember.
//!
//! Not feature-gated and no `fit()` call — Tier 2, so it runs in the base `--features ci`
//! job that builds on every PR.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

const WORKFLOW: &str = ".github/workflows/slow-tests.yml";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// ---------------------------------------------------------------------------------------
// GitHub Actions filter-pattern matching
// ---------------------------------------------------------------------------------------

/// One entry of a `paths:` list: a glob, plus whether it was written as an exclusion.
///
/// GitHub supports `!`-prefixed exclusions, and later patterns override earlier ones. A
/// guard that ignored the `!` would reject a legitimate exclusion as a dead glob.
struct Pattern {
    glob: String,
    negated: bool,
}

impl Pattern {
    fn parse(raw: &str) -> Self {
        match raw.strip_prefix('!') {
            Some(rest) => Pattern {
                glob: rest.to_string(),
                negated: true,
            },
            None => Pattern {
                glob: raw.to_string(),
                negated: false,
            },
        }
    }
}

/// GitHub's filter-pattern semantics: `*` and `?` do not cross `/`, `**` does.
///
/// Written against bytes rather than chars because every construct here is ASCII, and a
/// multi-byte path is only ever matched literally.
fn glob_matches(pattern: &[u8], path: &[u8]) -> bool {
    match pattern.first() {
        None => path.is_empty(),
        Some(b'*') if pattern.get(1) == Some(&b'*') => {
            // `**` — consume any run of characters, slashes included.
            glob_matches(&pattern[2..], path)
                || (!path.is_empty() && glob_matches(pattern, &path[1..]))
        }
        Some(b'*') => {
            glob_matches(&pattern[1..], path)
                || (!path.is_empty() && path[0] != b'/' && glob_matches(pattern, &path[1..]))
        }
        Some(b'?') => {
            !path.is_empty() && path[0] != b'/' && glob_matches(&pattern[1..], &path[1..])
        }
        Some(&lit) => !path.is_empty() && path[0] == lit && glob_matches(&pattern[1..], &path[1..]),
    }
}

/// Whether the filter as a whole selects `path`. Last matching pattern wins, so an
/// exclusion placed after an inclusion removes the path, exactly as GitHub evaluates it.
fn filter_selects(patterns: &[Pattern], path: &str) -> bool {
    let mut selected = false;
    for pattern in patterns {
        if glob_matches(pattern.glob.as_bytes(), path.as_bytes()) {
            selected = !pattern.negated;
        }
    }
    selected
}

// ---------------------------------------------------------------------------------------
// Workflow parsing
// ---------------------------------------------------------------------------------------

/// The `paths:` entries of the `push:` trigger in `slow-tests.yml`.
///
/// Scoped to that one block rather than "any `paths:` key": the block is entered at the
/// `paths:` line nested under `push:`, and left as soon as indentation returns to that
/// key's level or shallower. A `branches:` list written after `paths:`, or a future
/// `pull_request:` trigger with its own `paths:`, would otherwise be absorbed and its
/// entries validated as if they were globs.
fn push_path_patterns(workflow: &str) -> Vec<Pattern> {
    let mut in_push = false;
    let mut push_indent = 0usize;
    let mut paths_indent: Option<usize> = None;
    let mut out = Vec::new();

    for line in workflow.lines() {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indent = line.len() - trimmed.len();

        if let Some(list_indent) = paths_indent {
            if indent <= list_indent && !trimmed.starts_with("- ") {
                // Back out to a sibling key of `paths:` — the list is over.
                paths_indent = None;
            } else if let Some(item) = trimmed.strip_prefix("- ") {
                // Strip the inline `# why this is here` comment, then the quotes.
                let value = strip_inline_comment(item).trim().trim_matches(['\'', '"']);
                if !value.is_empty() {
                    out.push(Pattern::parse(value));
                }
                continue;
            }
        }

        if in_push && indent <= push_indent {
            in_push = false;
        }
        if trimmed.starts_with("push:") {
            in_push = true;
            push_indent = indent;
            continue;
        }
        if in_push && trimmed == "paths:" {
            paths_indent = Some(indent);
        }
    }

    out
}

/// Drop a trailing YAML comment, honouring quotes so a `#` inside a glob survives.
fn strip_inline_comment(item: &str) -> &str {
    let bytes = item.as_bytes();
    let mut quote: Option<u8> = None;
    for (i, &b) in bytes.iter().enumerate() {
        match (quote, b) {
            (None, b'\'' | b'"') => quote = Some(b),
            (Some(q), c) if c == q => quote = None,
            (None, b'#') if i > 0 && bytes[i - 1] == b' ' => return &item[..i],
            _ => {}
        }
    }
    item
}

// ---------------------------------------------------------------------------------------
// Deriving the inputs the suite actually reads
// ---------------------------------------------------------------------------------------

/// Every file tracked by git, repo-relative and `/`-separated.
///
/// Git rather than a filesystem walk: the path filter is evaluated by GitHub against
/// committed paths, so untracked scratch files and `.gitignore`d NONMEM run output (there
/// is some under `nonmem_anchor/`) must not count as uncovered inputs.
fn tracked_files(root: &Path) -> Vec<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .output()
        .expect(
            "`git ls-files` runs (the guard needs the checkout git evaluates the filter against)",
        );
    assert!(
        out.status.success(),
        "`git ls-files` failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .expect("tracked paths are UTF-8")
        .split('\0')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// The top-level roots the suite reads at run time, recovered from the test sources.
///
/// A test that opens a fixture writes its path as a literal — `"data/warfarin.csv"`,
/// `"nonmem_anchor/transit_iov.csv"`. Harvesting those literals is what keeps this guard
/// self-maintaining: a new fixture directory announces itself the moment a test reads from
/// it, with nothing to remember and nothing to hand-edit here.
///
/// Only roots that exist as a tracked directory are returned, so a path built at run time
/// (a temp dir, an output file the test writes) cannot invent a requirement.
///
/// Dot-prefixed roots are skipped. They hold repo tooling configuration — `.github/`,
/// `.githooks/` — and the only tests that open them are CI guards asserting on that config,
/// this file included. Reading a workflow to check it is not the same as a fit consuming a
/// fixture, and pulling `.github/**` into the filter would fire the 5–60 minute suite on
/// every docs-workflow or Dependabot edit. Fixtures never live in a dot-directory, so this
/// is a rule about where data lives rather than an exemption for one path.
fn runtime_input_roots(root: &Path, tracked: &[String]) -> BTreeSet<String> {
    let tracked_roots: BTreeSet<&str> = tracked
        .iter()
        .filter_map(|p| p.split_once('/').map(|(head, _)| head))
        .collect();

    let mut roots = BTreeSet::new();
    for path in tracked
        .iter()
        .filter(|p| p.starts_with("tests/") && p.ends_with(".rs"))
    {
        let src = std::fs::read_to_string(root.join(path)).expect("test source is UTF-8");
        for literal in string_literals(&src) {
            let Some((head, rest)) = literal.split_once('/') else {
                continue;
            };
            if rest.is_empty() || head.starts_with('.') || !tracked_roots.contains(head) {
                continue;
            }
            roots.insert(head.to_string());
        }
    }
    roots
}

/// Double-quoted literals in Rust source, escapes skipped. Deliberately crude: a false
/// positive is filtered out by the "must be a tracked root" test above, and a false
/// negative only costs coverage this guard never claimed.
fn string_literals(src: &str) -> Vec<String> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut j = start;
        while j < bytes.len() && bytes[j] != b'"' {
            j += if bytes[j] == b'\\' { 2 } else { 1 };
        }
        if j <= bytes.len() {
            if let Ok(s) = std::str::from_utf8(&bytes[start..j.min(bytes.len())]) {
                out.push(s.to_string());
            }
        }
        i = j + 1;
    }
    out
}

/// Every tracked file the slow job compiles, reads, or is configured by.
///
/// Compile inputs are mechanical (`cargo test` builds `src/` and `tests/`, configured by
/// the manifest and the lock); runtime inputs are derived; the workflow is included because
/// changing its feature set or cargo invocation changes what runs.
fn required_inputs(root: &Path, tracked: &[String]) -> Vec<String> {
    let mut dir_roots: BTreeSet<String> = ["src".to_string(), "tests".to_string()]
        .into_iter()
        .collect();
    dir_roots.extend(runtime_input_roots(root, tracked));

    let files: BTreeSet<&str> = ["Cargo.toml", "Cargo.lock", WORKFLOW].into_iter().collect();

    tracked
        .iter()
        .filter(|p| {
            files.contains(p.as_str())
                || p.split_once('/')
                    .is_some_and(|(head, _)| dir_roots.contains(head))
        })
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------

#[test]
fn slow_test_filter_covers_every_input_the_heavy_fits_read() {
    let root = repo_root();
    let workflow =
        std::fs::read_to_string(root.join(WORKFLOW)).expect("slow-tests.yml is readable");
    let patterns = push_path_patterns(&workflow);
    let tracked = tracked_files(&root);
    let required = required_inputs(&root, &tracked);

    // Detector self-checks. Each of these silently passing would make the assertion below
    // vacuous, which is the failure mode the guard exists to prevent in the first place.
    assert!(!patterns.is_empty(), "parsed no `paths:` entries out of {WORKFLOW} — the `push:` block moved or changed shape. Fix this parser rather than deleting the guard: an empty parse would accept any filter at all.");
    assert!(
        required.len() > 100,
        "derived only {} required inputs — the detector is broken, since `src/` and `tests/` \
         alone are far larger than that",
        required.len()
    );
    for expected_root in ["src/", "tests/", "data/", "examples/", "nonmem_anchor/"] {
        assert!(
            required.iter().any(|p| p.starts_with(expected_root)),
            "no required input under `{expected_root}` — the derivation stopped seeing a root \
             the heavy fits demonstrably read"
        );
    }

    // Group the misses by top-level root: a whole uncovered directory should read as one
    // line, not as three thousand.
    let mut missed: BTreeMap<&str, (usize, &str)> = BTreeMap::new();
    for path in required.iter().filter(|p| !filter_selects(&patterns, p)) {
        let head = path.split_once('/').map_or(path.as_str(), |(h, _)| h);
        let entry = missed.entry(head).or_insert((0, path.as_str()));
        entry.0 += 1;
    }

    assert!(
        missed.is_empty(),
        "these inputs feed the heavy fit suite but no `paths:` entry in {WORKFLOW} matches \
         them, so a push to `main` that changes only these silently skips the suite (#949):\n{}\n\
         Add a matching entry to the workflow's `push: paths:` list.",
        missed
            .iter()
            .map(|(head, (n, example))| format!("  {head}/ — {n} file(s), e.g. {example}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn slow_test_filter_has_no_dead_globs() {
    let root = repo_root();
    let workflow =
        std::fs::read_to_string(root.join(WORKFLOW)).expect("slow-tests.yml is readable");
    let patterns = push_path_patterns(&workflow);
    let tracked = tracked_files(&root);

    assert!(
        !patterns.is_empty(),
        "parsed no `paths:` entries out of {WORKFLOW}"
    );

    let dead: Vec<&str> = patterns
        .iter()
        .filter(|p| {
            !tracked
                .iter()
                .any(|f| glob_matches(p.glob.as_bytes(), f.as_bytes()))
        })
        .map(|p| p.glob.as_str())
        .collect();

    assert!(
        dead.is_empty(),
        "these {WORKFLOW} `paths:` entries match no tracked file: {dead:?}. A pattern that \
         matches nothing looks identical to a live one in review and silently narrows the \
         filter — this is how `src/api.rs` survived the `src/api/` split (#949). Either the \
         path moved (update it) or it is gone (drop it)."
    );
}

#[test]
fn derived_runtime_roots_are_the_fixture_directories() {
    let root = repo_root();
    let tracked = tracked_files(&root);
    let roots = runtime_input_roots(&root, &tracked);

    // The fixture roots the fits demonstrably read. Named here (rather than only relied on
    // implicitly) so that a harvest which quietly stops seeing one fails loudly.
    for expected in ["data", "examples", "nonmem_anchor"] {
        assert!(
            roots.contains(expected),
            "harvest lost the `{expected}/` fixture root: {roots:?}"
        );
    }
    // Repo tooling config is read by CI-guard tests (this file opens `.github/workflows/`),
    // but it is not a fit input and must not drag `.github/**` into the filter.
    assert!(
        !roots.iter().any(|r| r.starts_with('.')),
        "a dot-prefixed tooling root leaked into the derived fit inputs: {roots:?}"
    );
}

#[test]
fn github_glob_semantics() {
    // `**` crosses `/`; `*` does not.
    assert!(glob_matches(b"src/**", b"src/lib.rs"));
    assert!(glob_matches(b"src/**", b"src/api/tests/fit.rs"));
    assert!(!glob_matches(b"src/*", b"src/api/fit.rs"));
    assert!(glob_matches(b"src/*", b"src/lib.rs"));
    assert!(glob_matches(b"**/*.rs", b"a/b/c.rs"));
    assert!(!glob_matches(b"src/**", b"tests/lib.rs"));

    // Exact files, and `?` as a single non-slash character.
    assert!(glob_matches(b"Cargo.toml", b"Cargo.toml"));
    assert!(!glob_matches(b"Cargo.toml", b"Cargo.lock"));
    assert!(glob_matches(b"a?c", b"abc"));
    assert!(!glob_matches(b"a?c", b"a/c"));

    // `**` matches the empty string, so `src/**` selects nothing above `src/`.
    assert!(!glob_matches(b"src/**", b"src"));

    // Exclusions: last matching pattern wins, in list order.
    let include_only = [Pattern::parse("src/**")];
    assert!(filter_selects(&include_only, "src/bin/run_model.rs"));

    let with_exclusion = [Pattern::parse("src/**"), Pattern::parse("!src/bin/**")];
    assert!(filter_selects(&with_exclusion, "src/lib.rs"));
    assert!(!filter_selects(&with_exclusion, "src/bin/run_model.rs"));

    // Order matters: re-including after an exclusion wins again.
    let re_included = [
        Pattern::parse("src/**"),
        Pattern::parse("!src/bin/**"),
        Pattern::parse("src/bin/keep.rs"),
    ];
    assert!(filter_selects(&re_included, "src/bin/keep.rs"));
}

#[test]
fn workflow_parser_is_scoped_to_the_push_trigger() {
    // A `branches:` block after `paths:`, and a second trigger with its own `paths:`, must
    // not leak into the parsed list — both were real false positives in the shell guard
    // this test replaces.
    let workflow = "\
on:
  push:
    paths:
      - 'src/**'   # inline comment
      - '!src/bin/**'
    branches:
      - 'main'
  pull_request:
    paths:
      - 'docs/**'

env:
  FOO: bar
";
    let patterns = push_path_patterns(workflow);
    let globs: Vec<&str> = patterns.iter().map(|p| p.glob.as_str()).collect();
    assert_eq!(globs, vec!["src/**", "src/bin/**"]);
    assert!(!patterns[0].negated);
    assert!(
        patterns[1].negated,
        "a `!`-prefixed entry must parse as an exclusion"
    );
}

#[test]
fn inline_comments_do_not_leak_into_globs() {
    assert_eq!(
        strip_inline_comment("'src/**'   # everything linked into the fits"),
        "'src/**'   "
    );
    // A `#` that is not comment-introducing (no preceding space, or inside quotes) stays.
    assert_eq!(strip_inline_comment("'src/a#b/**'"), "'src/a#b/**'");
    assert_eq!(strip_inline_comment("'src/**'"), "'src/**'");
}
