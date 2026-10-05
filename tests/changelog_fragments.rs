//! `tools/changelog.sh` — changelog fragments (#1545).
//!
//! Each user-facing PR adds `changelog.d/<N>.<category>.md` instead of editing
//! `CHANGELOG.md`, so concurrent PRs stop conflicting on one insertion point; the
//! release step assembles the fragments. These tests run the script against scratch
//! trees through `CHANGELOG_ROOT`, never the real `CHANGELOG.md`.
//!
//! Every rejection is paired with an accepted twin differing in the one property under
//! test, so a check that rejects everything (or nothing) cannot pass.
//!
//! Not feature-gated: must run in the base `--features ci` job.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn script() -> PathBuf {
    repo_root().join("tools").join("changelog.sh")
}

const HEAD: &str = "# Changelog\n\nIntro.\n\n## [Unreleased]\n\n";
const TAIL: &str = "## [0.1.0] - 2026-01-01\n\n### Fixed\n- Old fix (#1).\n\n\
[Unreleased]: https://example.org/r/compare/v0.1.0...HEAD\n\
[0.1.0]: https://example.org/r/releases/tag/v0.1.0\n";

/// A fresh scratch tree holding `CHANGELOG.md` (empty `[Unreleased]`) and the given
/// fragments.
fn tree(slot: &str, fragments: &[(&str, &str)]) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("changelog-{slot}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("changelog.d")).unwrap();
    std::fs::write(dir.join("CHANGELOG.md"), format!("{HEAD}{TAIL}")).unwrap();
    std::fs::write(
        dir.join("changelog.d").join("README.md"),
        "not a fragment\n",
    )
    .unwrap();
    for (name, body) in fragments {
        std::fs::write(dir.join("changelog.d").join(name), body).unwrap();
    }
    dir
}

fn run(root: &Path, args: &[&str]) -> Output {
    Command::new("bash")
        .arg(script())
        .args(args)
        .env("CHANGELOG_ROOT", root)
        .output()
        .expect("run tools/changelog.sh")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn the_script_is_executable() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(script()).unwrap().permissions().mode();
        assert!(mode & 0o111 != 0, "tools/changelog.sh is not executable");
    }
}

/// `(file name, body)`.
type Fragment<'a> = (&'a str, &'a str);

/// Each malformed fragment is rejected with a message naming it, and its corrected twin
/// is accepted.
#[test]
fn check_rejects_each_malformed_fragment_and_accepts_its_twin() {
    let good =
        "- **A fix** (#12). More\n  words on a continuation line.\n\n  | a | b |\n  |---|---|\n";
    // (slot, malformed, accepted twin, expected message)
    let cases: &[(&str, Fragment, Fragment, &str)] = &[
        (
            "bad-name",
            ("fix-12.md", good),
            ("12.fixed.md", good),
            "name must be <N>.<category>.md",
        ),
        (
            // A `..`-prefixed name escaped every glob spelling, so `check` never
            // saw it while `require` counted it as a fragment.
            "dot-dot-name",
            ("..12.fixed.md", good),
            ("12.fixed.md", good),
            "name must be <N>.<category>.md",
        ),
        (
            "bad-category",
            ("12.bugfix.md", good),
            ("12.fixed.md", good),
            "unknown category 'bugfix'",
        ),
        (
            "empty",
            ("12.fixed.md", "\n  \n"),
            ("12.fixed.md", good),
            "empty",
        ),
        (
            "not-a-bullet",
            ("12.fixed.md", "A fix (#12).\n"),
            ("12.fixed.md", "- A fix (#12).\n"),
            "line 1 must start a bullet",
        ),
        (
            "two-bullets",
            ("12.fixed.md", "- One (#12).\n- Two (#12).\n"),
            ("12.fixed.md", "- One (#12).\n  - Two, nested (#12).\n"),
            "must be a single bullet",
        ),
        (
            // One space does not reach the content column of `- `, so Markdown
            // reads ` - Two` as a SIBLING bullet, not a nested one.
            "one-space-sibling",
            ("12.fixed.md", "- One (#12).\n - Two (#12).\n"),
            ("12.fixed.md", "- One (#12).\n  - Two, nested (#12).\n"),
            "must be a single bullet",
        ),
        (
            "heading",
            ("12.fixed.md", "- One (#12).\n### Fixed\n"),
            ("12.fixed.md", "- One (#12).\n"),
            "must be a single bullet",
        ),
    ];
    for (slot, bad, ok, want) in cases {
        let t = tree(&format!("check-{slot}-ok"), &[*ok]);
        let o = run(&t, &["check"]);
        assert!(
            o.status.success(),
            "{slot}: twin {:?} rejected:\n{}",
            ok.0,
            stderr(&o)
        );

        let t = tree(&format!("check-{slot}-bad"), &[*bad]);
        let o = run(&t, &["check"]);
        let err = stderr(&o);
        assert!(!o.status.success(), "{slot}: {:?} accepted", bad.0);
        assert!(err.contains(want), "{slot}: expected {want:?} in:\n{err}");
        assert!(
            err.contains(bad.0),
            "{slot}: message does not name {:?}:\n{err}",
            bad.0
        );
    }
}

/// `[Unreleased]` must stay empty: an entry added there out of habit reintroduces the
/// conflict the fragments exist to remove.
#[test]
fn check_rejects_an_entry_written_under_unreleased() {
    let t = tree("unreleased-ok", &[]);
    assert!(run(&t, &["check"]).status.success());

    let t = tree("unreleased-bad", &[]);
    std::fs::write(
        t.join("CHANGELOG.md"),
        format!("{HEAD}### Fixed\n- Habit (#9).\n\n{TAIL}"),
    )
    .unwrap();
    let o = run(&t, &["check"]);
    let err = stderr(&o);
    assert!(!o.status.success(), "an [Unreleased] entry was accepted");
    assert!(err.contains("[Unreleased] must stay empty"), "{err}");
    assert!(
        err.contains("- Habit (#9)."),
        "does not quote the line:\n{err}"
    );
}

/// Without exactly one `## [Unreleased]` heading an empty section and a missing one look
/// alike, and `assemble` once rewrote the links and deleted every fragment having
/// written them nowhere.
#[test]
fn a_missing_or_duplicate_unreleased_heading_is_refused_before_anything_is_deleted() {
    for (slot, changelog) in [
        (
            "missing",
            format!("# Changelog\n\n## [Unreleased-typo]\n\n{TAIL}"),
        ),
        ("duplicate", format!("{HEAD}## [Unreleased]\n\n{TAIL}")),
    ] {
        let t = tree(&format!("heading-{slot}"), &[("1.fixed.md", "- x (#1).\n")]);
        std::fs::write(t.join("CHANGELOG.md"), &changelog).unwrap();

        let o = run(&t, &["check"]);
        assert!(!o.status.success(), "{slot}: check passed");
        assert!(
            stderr(&o).contains("exactly one '## [Unreleased]' heading"),
            "{slot}: {}",
            stderr(&o)
        );

        let o = run(&t, &["assemble", "0.2.0", "--date", "2026-10-05"]);
        assert!(!o.status.success(), "{slot}: assemble passed");
        assert!(
            t.join("changelog.d").join("1.fixed.md").exists(),
            "{slot}: a refused assemble deleted the fragment"
        );
        assert_eq!(
            std::fs::read_to_string(t.join("CHANGELOG.md")).unwrap(),
            changelog,
            "{slot}: a refused assemble modified CHANGELOG.md"
        );
    }
}

#[test]
fn assemble_writes_a_keep_a_changelog_section_and_removes_the_fragments() {
    let t = tree(
        "assemble",
        &[
            ("20.fixed.md", "- Second fix (#20).\n\n\n"),
            ("3.fixed.md", "- First fix (#3).\n  Continued.\n"),
            ("3-2.fixed.md", "- Another for #3.\n"),
            ("7.performance.md", "- Faster (#7).\n"),
            ("9.added.md", "- New thing (#9).\n"),
        ],
    );
    // A mode no default produces: the rewrite must keep it (a `mktemp` file is 0600).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            t.join("CHANGELOG.md"),
            std::fs::Permissions::from_mode(0o640),
        )
        .unwrap();
    }
    let o = run(&t, &["assemble", "v0.2.0", "--date", "2026-10-05"]);
    assert!(o.status.success(), "{}", stderr(&o));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(t.join("CHANGELOG.md"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o640, "assemble changed CHANGELOG.md's mode");
    }

    let got = std::fs::read_to_string(t.join("CHANGELOG.md")).unwrap();
    // Categories in Keep a Changelog order (Added before Fixed before Performance,
    // regardless of file order); within one, by number — 3 < 3-2 < 20, numerically.
    let want = "# Changelog\n\nIntro.\n\n## [Unreleased]\n\n\
## [0.2.0] - 2026-10-05\n\n\
### Added\n- New thing (#9).\n\n\
### Fixed\n- First fix (#3).\n  Continued.\n- Another for #3.\n- Second fix (#20).\n\n\
### Performance\n- Faster (#7).\n\n\
## [0.1.0] - 2026-01-01\n\n### Fixed\n- Old fix (#1).\n\n\
[Unreleased]: https://example.org/r/compare/v0.2.0...HEAD\n\
[0.2.0]: https://example.org/r/compare/v0.1.0...v0.2.0\n\
[0.1.0]: https://example.org/r/releases/tag/v0.1.0\n";
    assert_eq!(got, want);

    let left: Vec<_> = std::fs::read_dir(t.join("changelog.d"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(left, vec!["README.md".to_string()], "fragments not removed");

    // The result is itself a valid, empty-[Unreleased] tree.
    assert!(run(&t, &["check"]).status.success());

    // Twice is refused: there is nothing left, and 0.2.0 now exists.
    let o = run(&t, &["assemble", "0.2.0"]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("no fragments"), "{}", stderr(&o));
    std::fs::write(t.join("changelog.d").join("1.fixed.md"), "- x (#1).\n").unwrap();
    let o = run(&t, &["assemble", "0.2.0"]);
    assert!(!o.status.success());
    assert!(
        stderr(&o).contains("already has a ## [0.2.0]"),
        "{}",
        stderr(&o)
    );
}

#[test]
fn assemble_refuses_invalid_fragments_and_versions() {
    let t = tree("assemble-invalid", &[("1.bogus.md", "- x (#1).\n")]);
    let before = std::fs::read_to_string(t.join("CHANGELOG.md")).unwrap();
    let o = run(&t, &["assemble", "0.2.0"]);
    assert!(!o.status.success(), "assembled an invalid fragment");
    assert_eq!(
        std::fs::read_to_string(t.join("CHANGELOG.md")).unwrap(),
        before,
        "CHANGELOG.md was modified by a refused assemble"
    );
    assert!(t.join("changelog.d").join("1.bogus.md").exists());

    for bad in [
        "next",
        "01.2.3",
        "1.02.3",
        "1.2.3-alpha..1",
        "1.2.3-",
        "1.2.3+build.1",
    ] {
        let t = tree("assemble-version", &[("1.fixed.md", "- x (#1).\n")]);
        let o = run(&t, &["assemble", bad]);
        assert!(!o.status.success(), "{bad} accepted as a version");
        assert!(
            stderr(&o).contains("is not a release version"),
            "{bad}: {}",
            stderr(&o)
        );
        assert!(t.join("changelog.d").join("1.fixed.md").exists());
    }
    for good in ["0.10.0", "v1.2.3", "1.2.3-rc.1", "1.2.3-alpha-2"] {
        let t = tree("assemble-version-ok", &[("1.fixed.md", "- x (#1).\n")]);
        let o = run(&t, &["assemble", good, "--date", "2026-10-05"]);
        assert!(o.status.success(), "{good} refused: {}", stderr(&o));
    }
}

#[test]
fn preview_renders_the_pending_section() {
    let t = tree(
        "preview",
        &[
            ("5.changed.md", "- Changed (#5).\n"),
            ("4.added.md", "- Added (#4).\n"),
        ],
    );
    let o = run(&t, &["preview"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(
        String::from_utf8_lossy(&o.stdout),
        "## [Unreleased]\n\n### Added\n- Added (#4).\n\n### Changed\n- Changed (#5).\n"
    );
}

/// A scratch git repo with one base commit; returns (dir, base sha).
fn git_repo(slot: &str) -> (PathBuf, String) {
    let dir = tree(&format!("git-{slot}"), &[]);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("src").join("lib.rs"), "// v1\n").unwrap();
    git(&dir, &["init", "-q", "-b", "main"]);
    git(&dir, &["add", "-A"]);
    git(&dir, &["commit", "-q", "-m", "base"]);
    let base = git(&dir, &["rev-parse", "HEAD"]);
    (dir, base.trim().to_string())
}

fn git(dir: &Path, args: &[&str]) -> String {
    let o = Command::new("git")
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.org",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .current_dir(dir)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .output()
        .expect("git on PATH");
    assert!(o.status.success(), "git {args:?}: {}", stderr(&o));
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn commit_file(dir: &Path, rel: &str, body: &str) {
    let p = dir.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(&p, body).unwrap();
    git(dir, &["add", "-A"]);
    git(dir, &["commit", "-q", "-m", rel]);
}

/// The PR-level gate: a user-facing change without a fragment fails, naming the file it
/// expects; the fragment, or the opt-out label, makes it pass.
#[test]
fn require_fails_a_user_facing_change_without_a_fragment() {
    let (d, base) = git_repo("require");
    commit_file(&d, "src/lib.rs", "// v2\n");

    let o = run(&d, &["require", &base, "--pr", "77"]);
    let err = stderr(&o);
    assert!(!o.status.success(), "src/ change without a fragment passed");
    assert!(
        err.contains("Missing: changelog.d/77.<category>.md"),
        "{err}"
    );
    assert!(
        err.contains("src/lib.rs"),
        "does not list the changed path:\n{err}"
    );
    assert!(
        err.contains("no-changelog"),
        "does not name the opt-out label:\n{err}"
    );

    // The opt-out is the boolean the workflow computes from the label; anything but
    // `true`/`false` is a usage error, not a silent pass.
    let o = run(&d, &["require", &base, "--pr", "77", "--opt-out", "false"]);
    assert!(!o.status.success(), "--opt-out false opted out");
    let o = run(
        &d,
        &["require", &base, "--pr", "77", "--opt-out", "no-changelog"],
    );
    assert!(!o.status.success(), "a non-boolean --opt-out passed");
    assert!(
        stderr(&o).contains("must be true or false"),
        "{}",
        stderr(&o)
    );
    let o = run(&d, &["require", &base, "--pr", "77", "--opt-out", "true"]);
    assert!(o.status.success(), "{}", stderr(&o));

    // A README edit does not count as a fragment; a fragment does.
    commit_file(&d, "changelog.d/README.md", "edited\n");
    assert!(!run(&d, &["require", &base, "--pr", "77"]).status.success());
    commit_file(&d, "changelog.d/77.fixed.md", "- Fixed (#77).\n");
    let o = run(&d, &["require", &base, "--pr", "77"]);
    assert!(o.status.success(), "{}", stderr(&o));
}

/// Only a fragment this PR ADDS counts: rewording or renaming another PR's pending
/// fragment does not describe this change.
#[test]
fn require_counts_only_an_added_fragment() {
    for (slot, edit) in [("modify", false), ("rename", true)] {
        let (d, _) = git_repo(&format!("only-added-{slot}"));
        commit_file(
            &d,
            "changelog.d/10.fixed.md",
            "- Someone else's fix (#10).\n",
        );
        let base = git(&d, &["rev-parse", "HEAD"]).trim().to_string();
        commit_file(&d, "src/lib.rs", "// v2\n");
        if edit {
            git(
                &d,
                &["mv", "changelog.d/10.fixed.md", "changelog.d/11.fixed.md"],
            );
            git(&d, &["commit", "-q", "-m", "rename"]);
        } else {
            commit_file(&d, "changelog.d/10.fixed.md", "- Reworded fix (#10).\n");
        }
        let o = run(&d, &["require", &base, "--pr", "11"]);
        assert!(
            !o.status.success(),
            "{slot} of an existing fragment counted as this PR's own"
        );
    }
}

/// A rename reports only its destination, so moving production code into a test file
/// must still count as touching user-facing code.
#[test]
fn require_sees_the_source_side_of_a_rename() {
    let (d, base) = git_repo("rename-src");
    git(&d, &["mv", "src/lib.rs", "src/lib_tests.rs"]);
    git(&d, &["commit", "-q", "-m", "rename"]);
    let o = run(&d, &["require", &base, "--pr", "5"]);
    assert!(
        !o.status.success(),
        "src/lib.rs -> src/lib_tests.rs passed as test-only"
    );
    assert!(stderr(&o).contains("src/lib.rs"), "{}", stderr(&o));
}

#[test]
fn require_skips_changes_that_are_not_user_facing() {
    for (slot, rel) in [
        ("tests-sibling", "src/stats/npde_tests.rs"),
        ("tests-dir", "src/api/tests/fit_tests.rs"),
        ("root-tests", "tests/foo.rs"),
        ("docs", "docs/index.qmd"),
        ("member-tests", "crates/ferx-cli/tests/cli.rs"),
    ] {
        let (d, base) = git_repo(&format!("skip-{slot}"));
        commit_file(&d, rel, "x\n");
        let o = run(&d, &["require", &base, "--pr", "5"]);
        assert!(
            o.status.success(),
            "{rel} demanded a fragment:\n{}",
            stderr(&o)
        );
    }
    // The twin: a member's source IS user-facing.
    let (d, base) = git_repo("member-src");
    commit_file(&d, "crates/ferx-tools/src/lib.rs", "x\n");
    assert!(!run(&d, &["require", &base, "--pr", "5"]).status.success());
}

/// `preflight.sh changelog` must propagate the script's failure, not just print it.
#[test]
fn the_preflight_group_fails_when_the_fragments_do() {
    let preflight = repo_root().join("tools").join("preflight.sh");
    let ok = tree("preflight-ok", &[("1.fixed.md", "- x (#1).\n")]);
    let bad = tree("preflight-bad", &[("1.nope.md", "- x (#1).\n")]);

    let o = Command::new("bash")
        .arg(&preflight)
        .arg("changelog")
        .env("CHANGELOG_ROOT", &ok)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", stderr(&o));

    let o = Command::new("bash")
        .arg(&preflight)
        .arg("changelog")
        .env("CHANGELOG_ROOT", &bad)
        .output()
        .unwrap();
    let err = stderr(&o);
    assert!(
        !o.status.success(),
        "preflight passed over an invalid fragment"
    );
    assert!(
        err.contains("preflight FAILED in group 'changelog'"),
        "{err}"
    );
    assert!(err.contains("CI job:   Changelog"), "{err}");
}

/// The workflow's wiring is what makes `require` a gate: each of these lines has a
/// one-token edit (`base.sha` -> `github.sha`, dropping `unlabeled`, a shallow
/// checkout) that leaves every test above green and the gate inert or stale.
#[test]
fn the_changelog_workflow_is_wired_to_the_pr_base_and_its_labels() {
    let p = repo_root()
        .join(".github")
        .join("workflows")
        .join("changelog.yml");
    let yml = std::fs::read_to_string(&p).unwrap();
    let lines: Vec<&str> = yml
        .lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .collect();
    for want in [
        "types: [opened, synchronize, reopened, labeled, unlabeled]",
        "fetch-depth: 0",
        "BASE_SHA: ${{ github.event.pull_request.base.sha }}",
        "PR: ${{ github.event.pull_request.number }}",
        "NO_CHANGELOG: ${{ contains(github.event.pull_request.labels.*.name, 'no-changelog') }}",
        r#"run: tools/changelog.sh require "$BASE_SHA" --pr "$PR" --opt-out "$NO_CHANGELOG""#,
    ] {
        assert!(
            lines.contains(&want),
            "changelog.yml lost `{want}` — see #1545"
        );
    }
    // No step may mask the gate.
    assert!(
        !lines.iter().any(
            |l| l.trim_start_matches("- ").starts_with("continue-on-error:")
                || l.trim_start_matches("- ").starts_with("if:")
        ),
        "changelog.yml can skip itself or pass while red"
    );
}
