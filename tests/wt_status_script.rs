//! `tools/wt-status.sh` — one-call checkout status for worktree-isolated agent sessions.
//!
//! The script exists so an agent behind Claude Code's worktree fence asks "behind? conflicts?
//! dirty? pushed?" in one plain command instead of the `$(git …)` / `|` / `&&` compounds the
//! fence refuses (see the script header). Each test builds
//! a scratch `origin` + clone under `CARGO_TARGET_TMPDIR` and runs the script there, never on
//! this checkout.
//!
//! Every reported state is paired with a twin that differs in the one property under test, so
//! a line that is printed unconditionally (or never) cannot pass.
//!
//! Not feature-gated: must run in the base `--features ci` job.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tools")
        .join("wt-status.sh")
}

/// `git` isolated from the developer's and runner's configuration, so a global
/// `pull.rebase`, signing key, or hook path cannot change what the fixture builds.
fn git_cmd(dir: &Path) -> Command {
    let mut c = Command::new("git");
    c.current_dir(dir)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@example.org")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@example.org");
    c
}

fn git(dir: &Path, args: &[&str]) {
    let out = git_cmd(dir).args(args).output().expect("git on PATH");
    assert!(out.status.success(), "git {args:?} failed: {out:?}");
}

fn commit_file(dir: &Path, name: &str, body: &str, msg: &str) {
    std::fs::write(dir.join(name), body).unwrap();
    git(dir, &["add", name]);
    git(dir, &["commit", "-q", "-m", msg]);
}

/// A bare `origin` with one commit on `main` (`a.txt`, `b.txt`), a `seed` clone that
/// pushes to it, and a `work` clone on branch `topic`. Returns `(seed, work)`.
fn fixture(slot: &str) -> (PathBuf, PathBuf) {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("wt-status-{slot}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let origin = root.join("origin.git");
    git(
        &root,
        &[
            "init",
            "-q",
            "--bare",
            "-b",
            "main",
            origin.to_str().unwrap(),
        ],
    );
    let seed = root.join("seed");
    git(
        &root,
        &[
            "clone",
            "-q",
            origin.to_str().unwrap(),
            seed.to_str().unwrap(),
        ],
    );
    git(&seed, &["checkout", "-q", "-b", "main"]);
    commit_file(&seed, "a.txt", "one\n", "a");
    commit_file(&seed, "b.txt", "two\n", "b");
    git(&seed, &["push", "-q", "origin", "main"]);
    let work = root.join("work");
    git(
        &root,
        &[
            "clone",
            "-q",
            origin.to_str().unwrap(),
            work.to_str().unwrap(),
        ],
    );
    git(&work, &["checkout", "-q", "-b", "topic"]);
    (seed, work)
}

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new("bash")
        .arg(script())
        .args(args)
        .current_dir(dir)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .expect("run tools/wt-status.sh")
}

/// Stdout of a successful run.
fn report(dir: &Path, args: &[&str]) -> String {
    let out = run(dir, args);
    assert!(out.status.success(), "wt-status {args:?} failed: {out:?}");
    String::from_utf8(out.stdout).unwrap()
}

/// The value after `key: ` on the report's line for `key`.
fn field<'a>(rep: &'a str, key: &str) -> &'a str {
    let prefix = format!("{key}: ");
    rep.lines()
        .find_map(|l| l.strip_prefix(prefix.as_str()))
        .unwrap_or_else(|| panic!("no `{key}:` line in report:\n{rep}"))
}

#[test]
fn the_script_is_executable() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(script()).unwrap().permissions().mode();
        assert!(mode & 0o111 != 0, "tools/wt-status.sh is not executable");
    }
}

/// `main` and `topic` both edit `a.txt`'s one line → reported conflict naming the path.
/// The twin edits `b.txt` on `topic` instead → `none`. Both are 1 ahead / 1 behind, so the
/// conflict line, not the divergence, is what separates them.
#[test]
fn conflicts_name_the_path_and_a_disjoint_edit_reports_none() {
    for (slot, topic_file, want_conflict) in
        [("conflict", "a.txt", true), ("disjoint", "b.txt", false)]
    {
        let (seed, work) = fixture(slot);
        commit_file(&seed, "a.txt", "main's\n", "main edits a");
        git(&seed, &["push", "-q", "origin", "main"]);
        commit_file(&work, topic_file, "topic's\n", "topic edit");
        let rep = report(&work, &["--fetch"]);
        assert_eq!(field(&rep, "ahead"), "1", "{slot}:\n{rep}");
        assert_eq!(field(&rep, "behind"), "1", "{slot}:\n{rep}");
        assert_eq!(field(&rep, "changed"), "1", "{slot}:\n{rep}");
        if want_conflict {
            assert_eq!(field(&rep, "conflicts"), "yes", "{slot}:\n{rep}");
            assert!(
                rep.lines().any(|l| l == "  a.txt"),
                "{slot}: path not listed:\n{rep}"
            );
        } else {
            assert_eq!(field(&rep, "conflicts"), "none", "{slot}:\n{rep}");
            assert!(!rep.contains("  a.txt"), "{slot}:\n{rep}");
        }
    }
}

/// Without `--fetch` the remote-tracking ref is stale and the report says 0 behind; with it,
/// 1 behind. Pins that `--fetch` actually fetches rather than being accepted and ignored.
#[test]
fn fetch_flag_picks_up_a_new_base_commit() {
    let (seed, work) = fixture("fetch");
    commit_file(&seed, "c.txt", "three\n", "main moves");
    git(&seed, &["push", "-q", "origin", "main"]);
    let stale = report(&work, &[]);
    assert_eq!(field(&stale, "behind"), "0", "{stale}");
    let fresh = report(&work, &["--fetch"]);
    assert_eq!(field(&fresh, "behind"), "1", "{fresh}");
}

/// An untracked file is counted and listed in porcelain form; the clean twin says `clean`.
#[test]
fn dirty_lists_porcelain_and_clean_says_clean() {
    let (_seed, work) = fixture("dirty");
    let clean = report(&work, &[]);
    assert_eq!(field(&clean, "dirty"), "clean", "{clean}");
    std::fs::write(work.join("stray.txt"), "x\n").unwrap();
    let dirty = report(&work, &[]);
    assert_eq!(field(&dirty, "dirty"), "1", "{dirty}");
    assert!(dirty.lines().any(|l| l == "  ?? stray.txt"), "{dirty}");
}

/// No upstream → `none`; pushed → `in sync`; one more local commit → `ahead 1`.
#[test]
fn upstream_reports_none_in_sync_and_ahead() {
    let (_seed, work) = fixture("upstream");
    commit_file(&work, "d.txt", "four\n", "topic");
    assert_eq!(field(&report(&work, &[]), "upstream"), "none");
    git(&work, &["push", "-q", "-u", "origin", "topic"]);
    let synced = report(&work, &[]);
    let up = field(&synced, "upstream");
    assert!(
        up.starts_with("origin/topic ") && up.ends_with("(in sync)"),
        "{synced}"
    );
    commit_file(&work, "e.txt", "five\n", "topic 2");
    let ahead = report(&work, &[]);
    assert!(field(&ahead, "upstream").ends_with("(ahead 1)"), "{ahead}");
}

/// A base that does not exist is a usage error (exit 2) naming the ref; the same call with
/// a base that exists succeeds, so the failure is the missing ref and not the flag.
#[test]
fn missing_base_exits_2_and_an_existing_one_succeeds() {
    let (_seed, work) = fixture("base");
    let bad = run(&work, &["--base", "origin/nope"]);
    assert_eq!(bad.status.code(), Some(2), "{bad:?}");
    assert!(
        String::from_utf8_lossy(&bad.stderr).contains("origin/nope"),
        "{bad:?}"
    );
    let good = report(&work, &["--base", "origin/main"]);
    assert!(field(&good, "base").starts_with("origin/main "), "{good}");
}
