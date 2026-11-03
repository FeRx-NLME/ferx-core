//! Guard: `tools/nightly-mutants.sh`, the planning and reporting half of the nightly
//! cargo-mutants workflow (`.github/workflows/mutants.yml`, #2633).
//!
//! The workflow itself only runs on a schedule, so a regression in this script would
//! surface as a quietly wrong job summary days later — or never, since the job is
//! report-only and cannot go red on a survivor. Each subcommand is therefore pinned
//! here on fixtures, without cargo-mutants installed:
//!
//! - `range` must bound the window by the **committer** date of **first-parent**
//!   commits — a merge's author date, or a merged branch's own commits, would shift it;
//! - `diff` must leave test files out (they are not production code to mutate);
//! - `select` must collapse a tuple return's combinatorial FnValue variants to one,
//!   and every `examine_re` it writes must match its own mutant name and no other,
//!   under the same `regex` crate cargo-mutants compiles it with;
//! - `summary` must list every survivor across shards as `file:line` + mutation, and
//!   say when the budget left mutants untested.
//!
//! Not feature-gated: it must run in the base `--features ci` job.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/nightly-mutants.sh")
}

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new("bash")
        .arg(script())
        .args(args)
        .current_dir(dir)
        .env("MUTANTS_MAIN_REF", "main")
        .output()
        .expect("bash runs")
}

fn ok_stdout(out: Output) -> String {
    assert!(
        out.status.success(),
        "script failed: {}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8 stdout")
}

fn git(dir: &Path, args: &[&str], date: Option<&str>) -> String {
    let mut c = Command::new("git");
    c.args(args)
        .current_dir(dir)
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@example.com")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@example.com")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_NOSYSTEM", "1");
    if let Some(d) = date {
        c.env("GIT_COMMITTER_DATE", d);
        // An author date far from the committer date: a window keyed on the author
        // date would pick different bounds, which is what the range test checks.
        c.env("GIT_AUTHOR_DATE", "2020-01-01T00:00:00Z");
    }
    let out = c.output().expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

fn write(dir: &Path, rel: &str, body: &str) {
    let p = dir.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, body).unwrap();
}

/// A `main` whose first-parent history is: root (day 0), merge A (day 1 23:00),
/// merge B (day 2 01:00), merge C (day 2 23:30), merge D (day 3 00:30). Each merge
/// brings in one side commit. B's is dated between merge A and midnight, and C's
/// between merge C and the next midnight, so a walk that is not first-parent can pick
/// a side commit as a bound of the day-2 window (dropping `--first-parent` from the
/// script moves the base bound to B's side commit; mutation-checked).
fn fixture_repo() -> (tempfile::TempDir, Vec<String>) {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path();
    git(d, &["init", "-q", "-b", "main"], None);
    write(d, "src/lib.rs", "pub fn a() {}\n");
    git(d, &["add", "."], None);
    git(
        d,
        &["commit", "-q", "-m", "root"],
        Some("2026-10-30T12:00:00Z"),
    );
    let mut merges = vec![git(d, &["rev-parse", "HEAD"], None)];
    let days = [
        ("a", "2026-10-31T23:00:00Z", "2026-11-01T12:00:00Z"),
        ("b", "2026-11-01T01:00:00Z", "2026-10-31T23:30:00Z"),
        ("c", "2026-11-01T23:30:00Z", "2026-11-01T23:45:00Z"),
        ("d", "2026-11-02T00:30:00Z", "2026-11-02T13:00:00Z"),
    ];
    for (name, merge_date, side_date) in days {
        git(d, &["checkout", "-q", "-b", name], None);
        write(d, &format!("src/{name}.rs"), "pub fn f() -> u32 { 1 }\n");
        write(d, &format!("src/{name}_tests.rs"), "#[test] fn t() {}\n");
        write(d, &format!("tests/{name}.rs"), "#[test] fn t() {}\n");
        git(d, &["add", "."], None);
        git(d, &["commit", "-q", "-m", name], Some(side_date));
        git(d, &["checkout", "-q", "main"], None);
        git(
            d,
            &[
                "merge",
                "-q",
                "--no-ff",
                "-m",
                &format!("merge {name}"),
                name,
            ],
            Some(merge_date),
        );
        merges.push(git(d, &["rev-parse", "HEAD"], None));
    }
    (tmp, merges)
}

#[test]
fn range_bounds_the_window_by_first_parent_committer_date() {
    let (tmp, m) = fixture_repo();
    // Window (2026-11-01T00:00, 2026-11-02T00:00]: merges B and C landed in it, so
    // the diff runs from A (the last merge before) to C (the last merge inside).
    let out = ok_stdout(run(
        tmp.path(),
        &["range", "--until", "2026-11-02T00:00:00Z"],
    ));
    assert_eq!(out.trim(), format!("{} {}", m[1], m[3]));

    // --hours widens the window back past A, to the root.
    let out = ok_stdout(run(
        tmp.path(),
        &["range", "--until", "2026-11-02T00:00:00Z", "--hours", "48"],
    ));
    assert_eq!(out.trim(), format!("{} {}", m[0], m[3]));

    // A window before the repo existed has no base commit: the script must say so
    // rather than diff from an arbitrary point.
    let out = run(tmp.path(), &["range", "--until", "2020-01-01T00:00:00Z"]);
    assert!(!out.status.success(), "a window before the root must fail");
}

#[test]
fn diff_leaves_test_files_out() {
    let (tmp, m) = fixture_repo();
    let out_path = tmp.path().join("day.diff");
    let n = ok_stdout(run(
        tmp.path(),
        &["diff", &m[1], &m[3], out_path.to_str().unwrap()],
    ));
    assert_eq!(n.trim(), "2", "src/b.rs and src/c.rs, nothing else");
    let diff = std::fs::read_to_string(&out_path).unwrap();
    assert!(diff.contains("+++ b/src/b.rs") && diff.contains("+++ b/src/c.rs"));
    assert!(
        !diff.contains("_tests.rs"),
        "a #[path] test sibling leaked:\n{diff}"
    );
    assert!(
        !diff.contains("tests/"),
        "an integration test leaked:\n{diff}"
    );
    assert!(!diff.contains("src/a.rs") && !diff.contains("src/d.rs"));

    // A window that touched only tests yields an empty diff and a zero count, which
    // is what lets the workflow skip the mutation jobs.
    let n = ok_stdout(run(
        tmp.path(),
        &["diff", &m[3], &m[3], out_path.to_str().unwrap()],
    ));
    assert_eq!(n.trim(), "0");
    assert!(std::fs::read_to_string(&out_path).unwrap().is_empty());
}

// One function per selection case. `pair` is a 2-slot tuple whose 2 × 3 product
// cargo-mutants lists in full; `wrapped` nests a tuple inside `Ok(..)`; `is_ok`,
// `go` and `scale` are scalar returns whose every variant is a distinct edit.
const LIST: &str = "\
src/a.rs:10:5: replace pair -> (u8, f64) with (0, 0.0)
src/a.rs:10:5: replace pair -> (u8, f64) with (0, 1.0)
src/a.rs:10:5: replace pair -> (u8, f64) with (0, -1.0)
src/a.rs:10:5: replace pair -> (u8, f64) with (1, 0.0)
src/a.rs:10:5: replace pair -> (u8, f64) with (1, 1.0)
src/a.rs:10:5: replace pair -> (u8, f64) with (1, -1.0)
src/a.rs:12:9: replace + with - in pair
src/a.rs:12:9: replace + with * in pair
src/a.rs:14:20: delete ! in pair
src/a.rs:20:5: replace fill with ()
src/a.rs:21:12: replace match guard x.is_empty() with true in fill
src/a.rs:25:5: replace is_ok -> bool with true
src/a.rs:25:5: replace is_ok -> bool with false
src/a.rs:30:5: replace <impl Foo for Bar<'_>>::go -> Result<Option<u32>> with Ok(None)
src/a.rs:30:5: replace <impl Foo for Bar<'_>>::go -> Result<Option<u32>> with Ok(Some(0))
src/a.rs:40:5: replace wrapped -> Result<(Vec<u8>, f64)> with Ok((vec![], 0.0))
src/a.rs:40:5: replace wrapped -> Result<(Vec<u8>, f64)> with Ok((vec![], 1.0))
src/a.rs:40:5: replace wrapped -> Result<(Vec<u8>, f64)> with Ok((vec![0], 0.0))
src/a.rs:40:5: replace wrapped -> Result<(Vec<u8>, f64)> with Ok((vec![0], 1.0))
src/b.rs:7:13: delete field want_dx from struct ModeOut expression in go
src/b.rs:9:5: replace scale -> f64 with 0.0
src/b.rs:9:5: replace scale -> f64 with 1.0
";

/// The `examine_re` array of the generated config, decoded as TOML basic strings.
/// The crate has no TOML parser, so this is a strict one for exactly the shape
/// `select` writes: one `"…",` per line, whose only escapes are `\\` and `\"` — any
/// other escape or an unterminated string fails, since cargo-mutants' TOML parser
/// would read it differently or not at all.
fn examine_re(text: &str) -> Vec<String> {
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    assert_eq!(lines.next(), Some("examine_re = ["));
    let mut out = Vec::new();
    for l in lines {
        if l == "]" {
            return out;
        }
        let body = l
            .strip_prefix("  \"")
            .and_then(|b| b.strip_suffix("\","))
            .unwrap_or_else(|| panic!("not a `  \"…\",` entry: {l}"));
        let mut s = String::new();
        let mut it = body.chars();
        while let Some(c) = it.next() {
            match c {
                '\\' => match it.next() {
                    Some(e @ ('\\' | '"')) => s.push(e),
                    other => panic!("unexpected TOML escape \\{other:?} in {l}"),
                },
                '"' => panic!("unescaped quote in {l}"),
                c => s.push(c),
            }
        }
        out.push(s);
    }
    panic!("examine_re array is not closed");
}

/// The selected names, read back out of the generated config with the same regexes
/// cargo-mutants will compile — so the assertion is about what it will actually run.
fn selected(tmp: &Path, extra: &[&str]) -> (usize, Vec<String>) {
    let list = tmp.join("list.txt");
    std::fs::write(&list, LIST).unwrap();
    let cfg = tmp.join("select.toml");
    let mut args = vec!["select", list.to_str().unwrap(), cfg.to_str().unwrap()];
    args.extend_from_slice(extra);
    let n: usize = ok_stdout(run(tmp, &args)).trim().parse().unwrap();

    let text = std::fs::read_to_string(&cfg).unwrap();
    let res: Vec<regex::Regex> = examine_re(&text)
        .iter()
        .map(|p| regex::Regex::new(p).expect("a valid Rust regex"))
        .collect();
    assert_eq!(res.len(), n, "the printed count is the number of patterns");

    let mut hit = Vec::new();
    for re in &res {
        let matches: Vec<&str> = LIST.lines().filter(|l| re.is_match(l)).collect();
        assert_eq!(
            matches.len(),
            1,
            "pattern {re} must name exactly one mutant, matched {matches:?}"
        );
        hit.push(matches[0].to_string());
    }
    (n, hit)
}

/// The function-body (FnValue) lines of [`LIST`] that `select` must keep by default.
const FN_BODIES: [&str; 11] = [
    // `pair`: the greedy cover of its 2 × 3 product. Every slot value — `0`, `1`
    // for the u8 and `0.0`, `1.0`, `-1.0` for the f64 — appears in some kept
    // variant, in 3 rebuilds instead of 6.
    "src/a.rs:10:5: replace pair -> (u8, f64) with (0, 0.0)",
    "src/a.rs:10:5: replace pair -> (u8, f64) with (0, -1.0)",
    "src/a.rs:10:5: replace pair -> (u8, f64) with (1, 1.0)",
    "src/a.rs:20:5: replace fill with ()",
    // Scalar returns: every variant. The review case for #2633: `true` caught and
    // `false` surviving must surface the `false` survivor.
    "src/a.rs:25:5: replace is_ok -> bool with true",
    "src/a.rs:25:5: replace is_ok -> bool with false",
    "src/a.rs:30:5: replace <impl Foo for Bar<'_>>::go -> Result<Option<u32>> with Ok(None)",
    "src/a.rs:30:5: replace <impl Foo for Bar<'_>>::go -> Result<Option<u32>> with Ok(Some(0))",
    // `wrapped`: the tuple inside `Ok(..)` is found and covered — 2 of 4.
    "src/a.rs:40:5: replace wrapped -> Result<(Vec<u8>, f64)> with Ok((vec![], 0.0))",
    "src/a.rs:40:5: replace wrapped -> Result<(Vec<u8>, f64)> with Ok((vec![0], 1.0))",
    "src/b.rs:9:5: replace scale -> f64 with 0.0",
];

#[test]
fn select_keeps_every_scalar_variant_and_covers_every_tuple_slot_value() {
    let tmp = tempfile::tempdir().unwrap();
    let (n, mut hit) = selected(tmp.path(), &[]);
    hit.sort();
    let mut want: Vec<String> = FN_BODIES.iter().map(|s| s.to_string()).collect();
    want.push("src/b.rs:9:5: replace scale -> f64 with 1.0".into());
    want.sort();
    assert_eq!(hit, want, "no operator, guard or field mutant by default");
    assert_eq!(n, 12);
}

#[test]
fn select_all_genres_adds_every_other_mutant_once() {
    let tmp = tempfile::tempdir().unwrap();
    let (n, hit) = selected(tmp.path(), &["--all-genres"]);
    // 12 function-body mutants + all 5 others; the two `+` replacements at one
    // position are distinct edits and both stay.
    assert_eq!(n, 17, "{hit:#?}");
    for must in [
        "src/a.rs:12:9: replace + with - in pair",
        "src/a.rs:12:9: replace + with * in pair",
        "src/a.rs:14:20: delete ! in pair",
        "src/a.rs:21:12: replace match guard x.is_empty() with true in fill",
        "src/b.rs:7:13: delete field want_dx from struct ModeOut expression in go",
    ] {
        assert!(hit.iter().any(|h| h == must), "missing {must}");
    }
    assert!(!hit.iter().any(|h| h.ends_with("with (1, -1.0)")));
}

#[test]
fn select_exact_names_every_listed_mutant() {
    let tmp = tempfile::tempdir().unwrap();
    let (n, hit) = selected(tmp.path(), &["--exact"]);
    assert_eq!(n, LIST.lines().count());
    assert_eq!(hit, LIST.lines().collect::<Vec<_>>());
}

#[test]
fn untested_is_the_selection_minus_every_recorded_outcome() {
    let tmp = tempfile::tempdir().unwrap();
    let sel = tmp.path().join("selected.txt");
    std::fs::write(&sel, "m:1:1: a\nm:2:1: b\nm:3:1: c\nm:4:1: d\nm:5:1: e\n").unwrap();
    let s0 = shard(
        tmp.path(),
        "s0",
        &[("caught.txt", "m:1:1: a\n"), ("missed.txt", "m:2:1: b\n")],
    );
    let s1 = shard(
        tmp.path(),
        "s1",
        &[
            ("timeout.txt", "m:3:1: c\n"),
            ("unviable.txt", "m:9:1: z\n"),
        ],
    );
    let out = tmp.path().join("untested.txt");
    let args = [
        "untested",
        sel.to_str().unwrap(),
        out.to_str().unwrap(),
        s0.to_str().unwrap(),
        s1.to_str().unwrap(),
    ];
    assert_eq!(ok_stdout(run(tmp.path(), &args)).trim(), "2");
    assert_eq!(
        std::fs::read_to_string(&out).unwrap(),
        "m:4:1: d\nm:5:1: e\n"
    );

    // No shard output at all (every shard died): the whole selection is untested.
    let args = ["untested", sel.to_str().unwrap(), out.to_str().unwrap()];
    assert_eq!(ok_stdout(run(tmp.path(), &args)).trim(), "5");
}

fn shard(root: &Path, name: &str, files: &[(&str, &str)]) -> PathBuf {
    let d = root.join(name);
    std::fs::create_dir_all(&d).unwrap();
    for (f, body) in files {
        std::fs::write(d.join(f), body).unwrap();
    }
    d
}

#[test]
fn summary_lists_every_survivor_across_shards_and_flags_a_budget_hit() {
    let tmp = tempfile::tempdir().unwrap();
    let s0 = shard(
        tmp.path(),
        "s0",
        &[
            (
                "caught.txt",
                "src/a.rs:1:1: replace x with ()\nsrc/a.rs:2:1: replace y with ()\n",
            ),
            ("missed.txt", "src/z.rs:40:5: replace fill with ()\n"),
            (
                "unviable.txt",
                "src/a.rs:3:1: replace w -> T with Default::default()\n",
            ),
        ],
    );
    let s1 = shard(
        tmp.path(),
        "s1",
        &[
            ("caught.txt", "src/b.rs:1:1: replace q with ()\n"),
            (
                "missed.txt",
                "src/b.rs:9:5: replace scale -> f64 with 0.0\n",
            ),
            ("timeout.txt", "src/c.rs:5:9: replace < with <= in spin\n"),
        ],
    );
    // A shard killed before its first outcome has no files at all.
    let s2 = shard(tmp.path(), "s2", &[]);
    let args = [
        "summary",
        "--selected",
        "9",
        "--window",
        "aaa..bbb",
        s0.to_str().unwrap(),
        s1.to_str().unwrap(),
        s2.to_str().unwrap(),
    ];
    let md = ok_stdout(run(tmp.path(), &args));

    assert!(md.contains("### Mutants: 2 survived of 7 examined"), "{md}");
    assert!(md.contains("| 3 | 2 | 1 | 1 | 7 |"), "{md}");
    assert!(md.contains("Window: `aaa..bbb`"), "{md}");
    assert!(
        md.contains("examined 7 of 9 selected") && md.contains("other 2 were not tested"),
        "a budget hit must be in the headline: {md}"
    );
    // Survivors as `file:line` + mutation, sorted by file then line.
    let b = md
        .find("src/b.rs:9:5: replace scale -> f64 with 0.0")
        .expect(&md);
    let z = md.find("src/z.rs:40:5: replace fill with ()").expect(&md);
    assert!(b < z, "survivors sorted by path: {md}");
    assert!(
        md.contains("src/c.rs:5:9: replace < with <= in spin"),
        "{md}"
    );
    assert!(
        md.contains("string literals"),
        "the not-covered caveat is part of the report"
    );

    // A clean, complete night: no survivor section, no budget line.
    let clean = shard(
        tmp.path(),
        "clean",
        &[("caught.txt", "src/a.rs:1:1: replace x with ()\n")],
    );
    let md = ok_stdout(run(
        tmp.path(),
        &["summary", "--selected", "1", clean.to_str().unwrap()],
    ));
    assert!(md.contains("### Mutants: 0 survived of 1 examined"), "{md}");
    assert!(md.contains("Examined all 1 selected"), "{md}");
    assert!(
        !md.contains("Surviving mutants") && !md.contains("Budget hit"),
        "{md}"
    );
    assert!(
        !md.contains("Measured on this run"),
        "no outcomes.json, no timing: {md}"
    );
}

/// The two endings of a budget-hit pass, asserted together so a gate stuck on
/// either branch fails: handed on (`--continued`) the tail is promised to the next
/// pass and not listed; on the last pass it is listed by name, never dropped.
#[test]
fn summary_lists_the_untested_tail_unless_a_follow_up_pass_takes_it() {
    let tmp = tempfile::tempdir().unwrap();
    // Two mutant outcomes and the baseline, in cargo-mutants 27's outcomes.json shape.
    let outcomes = r#"{"outcomes": [
        {"scenario": "Baseline", "summary": "Success",
         "phase_results": [{"phase": "Build", "duration": 900.0}, {"phase": "Test", "duration": 40.0}]},
        {"scenario": {"Mutant": {"name": "m:1:1: a"}}, "summary": "CaughtMutant",
         "phase_results": [{"phase": "Build", "duration": 200.0}, {"phase": "Test", "duration": 30.0}]},
        {"scenario": {"Mutant": {"name": "m:2:1: b"}}, "summary": "MissedMutant",
         "phase_results": [{"phase": "Build", "duration": 220.0}, {"phase": "Test", "duration": 50.0}]}
    ]}"#;
    let s0 = shard(
        tmp.path(),
        "s0",
        &[
            ("caught.txt", "m:1:1: a\n"),
            ("missed.txt", "m:2:1: b\n"),
            ("outcomes.json", outcomes),
        ],
    );
    let untested = tmp.path().join("untested.txt");
    std::fs::write(&untested, "src/q.rs:9:5: replace late -> u8 with 0\n").unwrap();
    let base = [
        "summary",
        "--selected",
        "3",
        "--untested",
        untested.to_str().unwrap(),
    ];

    let mut last = base.to_vec();
    last.push(s0.to_str().unwrap());
    let md = ok_stdout(run(tmp.path(), &last));
    assert!(
        md.contains("other 1 were not tested and are listed below"),
        "{md}"
    );
    assert!(md.contains("#### Not examined"), "{md}");
    assert!(
        md.contains("src/q.rs:9:5: replace late -> u8 with 0"),
        "{md}"
    );
    assert!(!md.contains("follow-up pass"), "{md}");
    // The baseline is not a mutant: medians over the two mutants only.
    assert!(
        md.contains("Measured on this run: 2 mutants, median 210 s build + 40 s test = 250 s per mutant (max 270 s)."),
        "{md}"
    );

    let mut handed_on = base.to_vec();
    handed_on.push("--continued");
    handed_on.push(s0.to_str().unwrap());
    let md = ok_stdout(run(tmp.path(), &handed_on));
    assert!(
        md.contains("other 1 carry over to a follow-up pass"),
        "{md}"
    );
    assert!(
        !md.contains("Not examined") && !md.contains("src/q.rs"),
        "{md}"
    );
}
