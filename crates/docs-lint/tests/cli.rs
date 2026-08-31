//! The CLI contract: what `docs-lint` prints and, above all, what it *exits*.
//!
//! The exit code is the whole gate — CI reads nothing else — so it is pinned per
//! shape here rather than left to the corpus test, which can only ever exercise
//! the one code path today's `docs/` happens to take.
//!
//! Each case builds a throwaway docs tree, so the assertions do not move when a
//! real page is edited.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A unique scratch directory, removed when the test ends.
struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let unique = format!(
            "docs-lint-{tag}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).expect("create fixture dir");
        Self { dir }
    }

    fn write(&self, rel: &str, body: &str) -> PathBuf {
        let path = self.dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create fixture subdir");
        }
        std::fs::write(&path, body).expect("write fixture file");
        path
    }

    fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_docs-lint"))
        .args(args)
        .output()
        .expect("run docs-lint")
}

fn code(out: &Output) -> i32 {
    out.status.code().expect("exit code")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// A page with one violation of each of R1, R2, R3 and R4.
const DIRTY: &str = "\
# Page

### Skipped a level

[dead](#nowhere)

## Syntax

## Syntax

## Long

PADDING
";

fn dirty_page() -> String {
    DIRTY.replace("PADDING", &"x".repeat(6000))
}

#[test]
fn help_exits_zero_and_names_the_rules() {
    let out = run(&["--help"]);
    assert_eq!(code(&out), 0);
    let text = stdout(&out);
    for rule in ["R1", "R2", "R3", "R4"] {
        assert!(
            text.contains(rule),
            "--help does not mention {rule}:\n{text}"
        );
    }
}

#[test]
fn a_clean_tree_exits_zero() {
    let fx = Fixture::new("clean");
    fx.write("page.qmd", "# Page\n\n## One\n\nshort\n\n## Two\n\nshort\n");
    let out = run(&[fx.path().to_str().unwrap(), "--no-baseline"]);
    assert_eq!(code(&out), 0, "stdout:\n{}", stdout(&out));
    assert!(stdout(&out).contains("clean"));
}

#[test]
fn every_rule_reports_and_the_run_exits_one() {
    let fx = Fixture::new("dirty");
    fx.write("page.qmd", &dirty_page());
    let out = run(&[fx.path().to_str().unwrap(), "--no-baseline"]);
    assert_eq!(code(&out), 1);
    let text = stdout(&out);
    for rule in ["R1:", "R2:", "R3:", "R4:"] {
        assert!(text.contains(rule), "no {rule} in:\n{text}");
    }
    assert!(stderr(&out).contains("violation"));
}

#[test]
fn check_is_the_same_gate_without_the_clean_banner() {
    let fx = Fixture::new("check");
    fx.write("page.qmd", "# Page\n\n## One\n\nshort\n");
    let out = run(&[fx.path().to_str().unwrap(), "--no-baseline", "--check"]);
    assert_eq!(code(&out), 0);
    assert!(
        !stdout(&out).contains("clean"),
        "--check should stay quiet when there is nothing to report"
    );
}

#[test]
fn json_output_is_parseable_and_escapes_quotes() {
    let fx = Fixture::new("json");
    fx.write("page.qmd", &dirty_page());
    let out = run(&[fx.path().to_str().unwrap(), "--no-baseline", "--json"]);
    assert_eq!(code(&out), 1);
    let text = stdout(&out);
    assert!(text.starts_with('{'));
    assert!(text.contains("\"violations\""));
    assert!(text.contains("\"rule\": \"R3\""), "no R3 entry:\n{text}");
    // Messages quote the heading text, so the escaping path is exercised.
    assert!(text.contains("\\\""), "quotes are not escaped:\n{text}");
    assert!(text.contains("\"stale_baseline_entries\""));
}

#[test]
fn max_chars_moves_the_r1_threshold() {
    let fx = Fixture::new("maxchars");
    fx.write("page.qmd", "# Page\n\n## One\n\n0123456789\n");
    let lenient = run(&[fx.path().to_str().unwrap(), "--no-baseline"]);
    assert_eq!(code(&lenient), 0);
    let strict = run(&[
        fx.path().to_str().unwrap(),
        "--no-baseline",
        "--max-chars",
        "5",
    ]);
    assert_eq!(code(&strict), 1);
    assert!(stdout(&strict).contains("R1:"));
}

#[test]
fn the_baseline_excuses_shrinks_and_fails_growth_and_staleness() {
    let fx = Fixture::new("baseline");
    fx.write("page.qmd", "# Page\n\n## One\n\n0123456789\n");
    let baseline = fx.dir.join("baseline.txt");
    let docs = fx.path().to_str().unwrap().to_string();
    let bl = baseline.to_str().unwrap().to_string();

    // Record the violation, then the same run passes.
    let update = run(&[
        &docs,
        "--max-chars",
        "5",
        "--baseline",
        &bl,
        "--update-baseline",
    ]);
    assert_eq!(code(&update), 0, "stderr:\n{}", stderr(&update));
    assert!(std::fs::read_to_string(&baseline).unwrap().contains("R1 "));
    let excused = run(&[&docs, "--max-chars", "5", "--baseline", &bl]);
    assert_eq!(code(&excused), 0, "stdout:\n{}", stdout(&excused));
    assert!(stdout(&excused).contains("excused by the baseline"));

    // Growth fails, and says by how much.
    fx.write("page.qmd", "# Page\n\n## One\n\n0123456789abcdef\n");
    let grown = run(&[&docs, "--max-chars", "5", "--baseline", &bl]);
    assert_eq!(code(&grown), 1);
    assert!(
        stdout(&grown).contains("may only shrink"),
        "{}",
        stdout(&grown)
    );

    // A section that no longer violates leaves a stale entry, which also fails.
    fx.write("page.qmd", "# Page\n\n## One\n\nx\n");
    let stale = run(&[&docs, "--max-chars", "5", "--baseline", &bl]);
    assert_eq!(code(&stale), 1);
    assert!(
        stdout(&stale).contains("remove this line"),
        "{}",
        stdout(&stale)
    );
}

#[test]
fn a_malformed_baseline_is_a_usage_error_not_a_pass() {
    let fx = Fixture::new("badbaseline");
    fx.write("page.qmd", "# Page\n\n## One\n\nshort\n");
    let baseline = fx.write("baseline.txt", "R1 not-a-number docs/x.qmd::a\n");
    let out = run(&[
        fx.path().to_str().unwrap(),
        "--baseline",
        baseline.to_str().unwrap(),
    ]);
    assert_eq!(code(&out), 2);
    assert!(stderr(&out).contains("character count"), "{}", stderr(&out));
}

#[test]
fn an_unknown_option_exits_two_with_the_usage() {
    let out = run(&["--wat"]);
    assert_eq!(code(&out), 2);
    assert!(stderr(&out).contains("unknown option"));
    assert!(stderr(&out).contains("USAGE"));
}

#[test]
fn a_missing_value_or_docs_directory_exits_two() {
    let missing_value = run(&["--max-chars"]);
    assert_eq!(code(&missing_value), 2);

    let bad_number = run(&["--max-chars", "lots"]);
    assert_eq!(code(&bad_number), 2);
    assert!(stderr(&bad_number).contains("bad --max-chars"));

    let no_baseline_path = run(&["--baseline"]);
    assert_eq!(code(&no_baseline_path), 2);

    let fx = Fixture::new("missingdir");
    let absent = fx.dir.join("nope");
    let out = run(&[absent.to_str().unwrap()]);
    assert_eq!(code(&out), 2);
    assert!(stderr(&out).contains("no docs directory"));
}
