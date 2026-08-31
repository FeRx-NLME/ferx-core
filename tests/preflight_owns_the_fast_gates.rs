//! Guard: the `Check`, `Clippy` and `Format` jobs in `.github/workflows/ci.yml` must run
//! their cargo commands **through `tools/preflight.sh`**, and that script must actually
//! fail when a gate fails (#1157).
//!
//! The point of the script is not documentation — it is that CI executes the same list a
//! developer runs before pushing, so "green locally" and "green in CI" cannot mean
//! different things. A script that only *describes* the commands drifts the first time
//! somebody edits the workflow, and drifts silently: the local gate keeps passing while
//! CI checks something else. That is the failure this exists to prevent, and it has a
//! precedent — on #1133 a suite of 4634 passing tests was reported green while
//! `--features ci,nn,slow-tests` had been failing to compile for an entire commit,
//! because the local run and the CI run were not the same set.
//!
//! Three invariants:
//!
//!  1. Those three jobs contain **no inline `run: cargo …` step**, and neither skip
//!     themselves (`if:`) nor tolerate failure (`continue-on-error`). Each is a way for
//!     the two lists to diverge, or for a red gate to report green.
//!  2. Each of them **does** invoke `tools/preflight.sh <group>` with its own group.
//!  3. A failing command makes the script exit non-zero **from every position in the
//!     list**, not just the last one.
//!
//! (3) is not hypothetical. The first version of the script returned a status from `run`
//! and propagated it through a group function and a `case … esac || { …; exit 1; }` —
//! and bash suspends `errexit` for every command of an AND-OR list but the last, then
//! carries that suspension into the body of any function called there. So the `return 1`
//! stopped nothing, each group reported its *last* command's status, and four of the five
//! `cargo check` lines could fail with the script printing "preflight OK" and exiting 0.
//! The two tests that existed at the time both passed: they only ever ran `--list` and
//! `--help`, neither of which executes a gate. A gate nobody has watched fail is not a
//! gate.
//!
//! Deliberately NOT asserted: that the script's command list matches some second copy
//! here. There is no second copy — the script is the only list, which is the whole design.
//! What these tests pin is that the workflow keeps *delegating* to it and that the script
//! keeps *enforcing* it.
//!
//! The `public-api` job is out of scope for (1) and (2): it needs a pinned dated nightly
//! and a pinned `cargo-public-api` installed as separate workflow steps, so it calls
//! `tools/update-public-api.sh --check` directly. `preflight.sh public-api` runs that same
//! script, so a developer's full local run still covers it.
//!
//! Not feature-gated, so it runs in `Tests + coverage (core)` — the per-PR job that
//! executes the Tier-1/2 suite under `--features ferx-core/ci`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn preflight() -> PathBuf {
    repo_root().join("tools").join("preflight.sh")
}

fn ci_yml() -> String {
    let p = repo_root().join(".github").join("workflows").join("ci.yml");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// The body of one top-level job, from its `  <name>:` header to the next one.
///
/// Two-space indent is the job level in this file; `steps:` and everything under it is
/// deeper, so the next two-space `key:` is reliably the next job.
fn job_body(yml: &str, name: &str) -> String {
    let header = format!("\n  {name}:\n");
    let start = yml
        .find(&header)
        .unwrap_or_else(|| panic!("job `{name}` not found in ci.yml"))
        + header.len();
    let rest = &yml[start..];
    let end = rest
        .match_indices("\n  ")
        .find(|(i, _)| {
            let line = &rest[i + 3..];
            // A job header is `<ident>:` at exactly two spaces of indent — not a list
            // item (`- `), not a comment, and not a deeper-indented mapping key.
            let Some(colon) = line.find(':') else {
                return false;
            };
            !line.starts_with('-')
                && !line.starts_with('#')
                && !line.starts_with(' ')
                && line[..colon]
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
                && line[..colon].chars().next().is_some_and(|c| c != ' ')
        })
        .map(|(i, _)| i)
        .unwrap_or(rest.len());
    rest[..end].to_string()
}

/// Every shell command a job body runs.
///
/// Handles both `run: <cmd>` and the block form
///
/// ```yaml
/// run: |
///   cargo check --features ci
/// ```
///
/// The block form matters: it is the natural way to add a second command to a step, so a
/// scanner that only understood the inline form would wave through exactly the edit this
/// test exists to reject. Each line of a block counts as a command — a coarse split, but
/// it errs toward inspecting more text, which is the safe direction here.
fn run_commands(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut lines = body.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("run:") else {
            continue;
        };
        let indent = line.len() - trimmed.len();
        let rest = rest.trim();
        // `|`, `>`, and their chomping variants (`|-`, `>+`, …) all open a block scalar.
        let is_block = matches!(rest.chars().next(), Some('|' | '>'))
            && rest[1..]
                .chars()
                .all(|c| matches!(c, '-' | '+' | '0'..='9'));
        if is_block {
            while let Some(next) = lines.peek() {
                if next.trim().is_empty() {
                    lines.next();
                    continue;
                }
                let nt = next.trim_start();
                if next.len() - nt.len() <= indent {
                    break;
                }
                out.push(nt.to_string());
                lines.next();
            }
        } else if !rest.is_empty() {
            out.push(rest.to_string());
        }
    }
    out
}

/// The commands `preflight.sh <group> --list` says it would run, in order.
fn listed_commands(group: &str) -> Vec<String> {
    let out = Command::new("bash")
        .arg(preflight())
        .arg(group)
        .arg("--list")
        .current_dir(repo_root())
        .output()
        .expect("run tools/preflight.sh --list");
    assert!(
        out.status.success(),
        "`preflight.sh {group} --list` exited {:?}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix("$ "))
        .map(|c| c.trim().to_string())
        .collect()
}

#[test]
fn the_fast_gate_jobs_delegate_to_preflight_and_never_inline_cargo() {
    let yml = ci_yml();

    for (job, group) in [("check", "check"), ("clippy", "clippy"), ("fmt", "fmt")] {
        let body = job_body(&yml, job);
        let cmds = run_commands(&body);
        assert!(
            !cmds.is_empty(),
            "job `{job}` has no `run:` steps at all — did the job-splitting heuristic break?"
        );

        // (1a) no inline cargo — that is how the two lists diverge.
        let inline: Vec<&String> = cmds.iter().filter(|c| c.starts_with("cargo ")).collect();
        assert!(
            inline.is_empty(),
            "job `{job}` runs cargo inline instead of through tools/preflight.sh (#1157):\n  {}\n\
             Add the command to the `{group}` group in tools/preflight.sh instead, so the \
             local gate and the CI gate stay the same list.",
            inline
                .iter()
                .map(|c| c.as_str())
                .collect::<Vec<_>>()
                .join("\n  ")
        );

        // (1b) the job must not be able to skip itself or pass while red. Both turn the
        // gate into decoration without touching the command list, so neither is visible
        // to the assertions above. If a conditional is ever genuinely wanted here, that
        // is a deliberate edit to this test.
        //
        // Line-anchored rather than `body.contains("if:")`: a comment or a shell line
        // that merely mentions the string would false-positive, and a tripwire that
        // cries wolf gets deleted rather than fixed. A YAML key is the first token on
        // its line, optionally after a `- `.
        let masking: Vec<&str> = body
            .lines()
            .map(|l| l.trim_start().trim_start_matches("- ").trim_start())
            .filter(|l| l.starts_with("if:") || l.starts_with("continue-on-error:"))
            .collect();
        assert!(
            masking.is_empty(),
            "job `{job}` can skip itself or pass while red — a fast gate that does either \
             is not a gate (#1157):\n  {}",
            masking.join("\n  ")
        );

        // (2) it must actually call its own group.
        let want = format!("tools/preflight.sh {group}");
        assert!(
            cmds.iter().any(|c| c == &want),
            "job `{job}` never runs `{want}`; its steps are:\n  {}",
            cmds.join("\n  ")
        );
    }
}

/// The invariant that makes the script a gate rather than a log: **a failing command at
/// any position exits non-zero**.
///
/// Every `cargo`-driven position is failed in turn behind a fake `cargo` on `PATH`, so
/// this costs milliseconds and no compile. The positions come from `--list`, not from a
/// table here, so adding a sixth `cargo check` feature set is covered the moment it is
/// added — there is nothing to remember to update.
///
/// `public-api` is the one group not exercised: its command is `tools/update-public-api.sh
/// --check`, invoked by path rather than through `PATH`, and shimming it would mean
/// letting the real script hunt for a pinned toolchain. It runs through the same `run`
/// helper as all eight positions below, which is what is under test.
#[test]
fn a_failing_gate_fails_the_script_from_every_position() {
    let root = repo_root();
    let (counter, path) = fake_cargo_on_path("failure-path");

    for group in ["fmt", "check", "clippy"] {
        let listed = listed_commands(group);
        assert!(
            !listed.is_empty(),
            "group `{group}` lists no commands — nothing to fail"
        );

        for (idx, cmd) in listed.iter().enumerate() {
            let nth = idx + 1;
            let _ = std::fs::remove_file(&counter);
            let out = Command::new("bash")
                .arg(preflight())
                .arg(group)
                .current_dir(&root)
                .env("PATH", &path)
                .env("FAKE_CARGO_COUNT", &counter)
                .env("FAKE_CARGO_FAIL_AT", nth.to_string())
                .output()
                .expect("run tools/preflight.sh");

            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let ctx = format!("--- stdout ---\n{stdout}--- stderr ---\n{stderr}");

            assert!(
                !out.status.success(),
                "`preflight.sh {group}` exited 0 with command {nth} of {} FAILING \
                 (`{cmd}`).\nA gate that reports green while a command failed is worse than \
                 no gate: CI runs this same script, so the job goes green too.\n{ctx}",
                listed.len()
            );
            assert!(
                !stdout.contains("preflight OK"),
                "`preflight.sh {group}` printed its success banner although command {nth} \
                 (`{cmd}`) failed.\n{ctx}"
            );
            // It must stop AT the failing command and name it, not merely exit non-zero
            // for some later reason.
            assert!(
                stderr.contains("preflight FAILED in group"),
                "`preflight.sh {group}` exited non-zero for command {nth} (`{cmd}`) but \
                 printed no failure diagnostic — it may have died for an unrelated \
                 reason.\n{ctx}"
            );
            assert!(
                stderr.contains(cmd.as_str()),
                "`preflight.sh {group}` failed, but its diagnostic does not name the \
                 command that failed (`{cmd}`) — it stopped somewhere else.\n{ctx}"
            );
        }
    }
}

/// Install the fake `cargo` in its own directory and return `(counter file, PATH)`.
///
/// Each caller gets its own directory so the invocation counter cannot be shared —
/// integration tests in one binary run on parallel threads.
fn fake_cargo_on_path(slot: &str) -> (PathBuf, String) {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("preflight-{slot}"));
    std::fs::create_dir_all(&tmp).expect("create temp dir");
    write_fake_cargo(&tmp);
    let path = format!(
        "{}:{}",
        tmp.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    (tmp.join("count"), path)
}

/// The failure diagnostic must name the group that actually failed, and its CI job.
///
/// Only a MULTI-group run can see this. `CI_JOB` is a script-global assigned at the top
/// of each `group_*` function, so a group that forgets the assignment inherits the
/// previous group's value and the diagnostic then points confidently at the wrong
/// workflow job — worse than no diagnostic, because it sends you to a green log. The
/// per-position test above runs one group per process, where a stale value is
/// unreachable by construction.
#[test]
fn the_failure_diagnostic_names_the_group_that_actually_failed() {
    let (counter, path) = fake_cargo_on_path("group-attribution");
    let _ = std::fs::remove_file(&counter);

    // `fmt` is one command, so invocation 2 is the first command of `check`: fmt has
    // already passed, and the run is inside the second group when it dies.
    let out = Command::new("bash")
        .arg(preflight())
        .arg("fmt")
        .arg("check")
        .current_dir(repo_root())
        .env("PATH", &path)
        .env("FAKE_CARGO_COUNT", &counter)
        .env("FAKE_CARGO_FAIL_AT", "2")
        .output()
        .expect("run tools/preflight.sh fmt check");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "exited 0 with a failing command\n{stderr}"
    );
    assert!(
        stderr.contains("group 'check'"),
        "the failure happened in `check` but the diagnostic does not say so:\n{stderr}"
    );
    assert!(
        stderr.contains("Check"),
        "the diagnostic does not name the `Check` CI job:\n{stderr}"
    );
    assert!(
        !stderr.contains("Format"),
        "the diagnostic blames `Format`, the group that PASSED — a stale CI_JOB sends \
         the reader to a green log:\n{stderr}"
    );
    assert!(
        !stderr.contains("sets no CI_JOB"),
        "the `check` group set no CI_JOB at all:\n{stderr}"
    );
}

/// The gates must run on the channel CI runs, not on whatever rustup resolves.
///
/// `rust-toolchain.toml` says `nightly`, but it is not the last word: an inherited
/// `RUSTUP_TOOLCHAIN`, or a `rustup override` on this directory or any parent, beats it.
/// Measured in this repo, `RUSTUP_TOOLCHAIN=stable cargo fmt --version` reports
/// `rustfmt 1.9.0-stable` where a plain `cargo fmt --version` reports `1.10.0-nightly`.
/// A developer with a stable override would get a quietly laxer clippy — whole lint
/// groups are nightly-only — which is the local/CI split this whole script exists to
/// remove. `ci.yml` pins the channel at workflow level; the script has to as well.
#[test]
fn the_gates_run_on_ci_s_toolchain_unless_the_caller_says_otherwise() {
    let (counter, path) = fake_cargo_on_path("toolchain");

    // Nothing set, and a `rustup override` may well be in force: must still be nightly.
    let _ = std::fs::remove_file(&counter);
    let out = Command::new("bash")
        .arg(preflight())
        .arg("check")
        .current_dir(repo_root())
        .env("PATH", &path)
        .env("FAKE_CARGO_COUNT", &counter)
        .env_remove("RUSTUP_TOOLCHAIN")
        .output()
        .expect("run tools/preflight.sh check");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "all commands passed but the script failed:\n{stdout}"
    );
    assert!(
        stdout.contains("RUSTUP_TOOLCHAIN=nightly"),
        "the gates did not run on nightly, so a local `rustup override` would silently \
         check something CI never checks:\n{stdout}"
    );

    // A deliberate choice by the caller still wins — this is a default, not a clamp.
    let _ = std::fs::remove_file(&counter);
    let out = Command::new("bash")
        .arg(preflight())
        .arg("check")
        .current_dir(repo_root())
        .env("PATH", &path)
        .env("FAKE_CARGO_COUNT", &counter)
        .env("RUSTUP_TOOLCHAIN", "some-pinned-toolchain")
        .output()
        .expect("run tools/preflight.sh check");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("RUSTUP_TOOLCHAIN=some-pinned-toolchain"),
        "the script overrode a toolchain the caller set deliberately:\n{stdout}"
    );
}

/// A `cargo` that succeeds until its `FAKE_CARGO_FAIL_AT`-th invocation.
///
/// Counting invocations rather than matching arguments keeps the harness independent of
/// what the commands actually say, so it does not become a second copy of the list.
fn write_fake_cargo(dir: &Path) {
    let path = dir.join("cargo");
    std::fs::write(
        &path,
        r#"#!/bin/sh
# Test double for `cargo` (tests/preflight_owns_the_fast_gates.rs). Not used by any build.
n=$(cat "$FAKE_CARGO_COUNT" 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" > "$FAKE_CARGO_COUNT"
echo "fake cargo invocation #$n: $*"
echo "fake cargo saw RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN-<unset>}"
if [ "$n" = "$FAKE_CARGO_FAIL_AT" ]; then
  echo "error[E0063]: simulated failure on invocation $n" >&2
  exit 101
fi
exit 0
"#,
    )
    .expect("write fake cargo");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .expect("chmod fake cargo");
    }
}

/// The counts in `preflight_is_executable_and_lists_every_group` see a gate DELETED. They
/// do not see one NEUTERED — a command that stays in the list while quietly checking less.
/// Every mutation below is a one-token edit that leaves the count at its expected value:
///
///  * dropping `--all-targets` from clippy leaves lib and bins linted and roughly a third
///    of the repo — every `#[cfg(test)]` module, every sibling `*_tests.rs`, all of
///    `tests/` — unlinted. That gap hid 6 `approx_constant` ERRORS (#1023).
///  * dropping `--all` from `cargo fmt` formats only the root package, silently skipping
///    `crates/ferx-tools` and `crates/ferx-cli` (#1114).
///  * narrowing a `--features` set drops a whole cfg-gated surface out of the compile.
///    `ci,nn,slow-tests` becoming `ci` is exactly #1133 with the gate still present.
///
/// The feature check is deliberately a **union over the whole group**, not a per-line
/// assertion: it pins that the matrix still compiles each gated surface without caring how
/// the lines are arranged, so splitting or merging feature sets stays a free refactor.
#[test]
fn load_bearing_flags_and_feature_coverage_survive_in_the_command_list() {
    let clippy = listed_commands("clippy");
    assert!(
        clippy.iter().any(|c| c.contains("--all-targets")),
        "no clippy command passes `--all-targets`. Without it every `#[cfg(test)]` module, \
         every sibling `*_tests.rs` and all of `tests/` go unlinted — the gap that hid 6 \
         `approx_constant` errors in #1023.\n  {}",
        clippy.join("\n  ")
    );

    let fmt = listed_commands("fmt");
    assert!(
        fmt.iter().any(|c| c.contains(" --all")),
        "no fmt command passes `--all`; the workspace members would go unformatted \
         (#1114).\n  {}",
        fmt.join("\n  ")
    );

    // Union of every `--features` value in the check group.
    let check = listed_commands("check");
    let mut features: Vec<String> = Vec::new();
    for cmd in &check {
        let mut parts = cmd.split_whitespace();
        while let Some(p) = parts.next() {
            if p == "--features" {
                if let Some(list) = parts.next() {
                    for f in list.split(',') {
                        // Members take their features package-qualified (`ferx-core/ci`).
                        let f = f.rsplit('/').next().unwrap_or(f);
                        features.push(f.to_string());
                    }
                }
            }
        }
    }
    for want in ["ci", "survival", "slow-tests", "markov", "nn"] {
        assert!(
            features.iter().any(|f| f == want),
            "the `check` group compiles nothing under `{want}`, so every source and test \
             file behind that cfg is unchecked on this PR. #1133 is what that looks like: \
             one struct literal in an `nn`-gated file turned three CI jobs red after a \
             local run of 4634 tests reported clean.\nfeature sets seen: {features:?}\n  {}",
            check.join("\n  ")
        );
    }
}

#[test]
fn preflight_is_executable_and_lists_every_group() {
    let script = preflight();
    assert!(script.is_file(), "tools/preflight.sh is missing");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&script).unwrap().permissions().mode();
        assert!(
            mode & 0o111 != 0,
            "tools/preflight.sh is not executable (mode {mode:o}); CI runs it directly"
        );
    }

    let out = Command::new("bash")
        .arg(&script)
        .arg("--list")
        .current_dir(repo_root())
        .output()
        .expect("run tools/preflight.sh --list");
    assert!(
        out.status.success(),
        "tools/preflight.sh --list exited {:?}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    // Per group: the banner is NOT evidence the group ran. The driver loop prints it
    // before dispatching, so a group whose dispatch was gutted still shows a header and
    // exits 0 — CI would run `preflight.sh clippy`, lint nothing, and pass. That mutation
    // survived the first draft of this test. So count the commands each group emits.
    //
    // The counts are a tripwire against silent SHRINKAGE, not a second copy of the
    // command list — deleting a `cargo check` feature set has to be a deliberate edit
    // here too, and lowering a number is a reviewable line in the diff. Whether those
    // commands are actually *enforced* is
    // `a_failing_gate_fails_the_script_from_every_position`; `--list` executes nothing and
    // can never show it.
    let expected: [(&str, usize); 5] = [
        ("fmt", 1),        // cargo fmt --all -- --check
        ("check", 5),      // ci · ci,survival,slow-tests · ci,markov · ci,nn,slow-tests · members
        ("clippy", 2),     // ferx-core --all-targets · members
        ("docs", 2),       // cargo test -p docs-lint · cargo clippy -p docs-lint
        ("public-api", 1), // tools/update-public-api.sh --check
    ];

    // Every group `--list` actually walks must have an entry above. The loop below
    // iterates `expected`, not the groups, so without this a fifth group added to
    // `ALL_GROUPS` would get no count tripwire at all — it would simply never be
    // looked at, which is the silent-shrinkage failure one level up.
    let walked: Vec<&str> = stdout
        .lines()
        .filter_map(|l| l.trim().strip_prefix("══ "))
        .filter_map(|l| l.split_whitespace().next())
        .collect();
    for g in &walked {
        assert!(
            expected.iter().any(|(e, _)| e == g),
            "group `{g}` runs but has no expected command count in this test — add one, \
             or it is gated by nothing.\ngroups walked: {walked:?}"
        );
    }
    assert_eq!(
        walked.len(),
        expected.len(),
        "`--list` walked {walked:?} but this test expects {} group(s)",
        expected.len()
    );

    for (group, want) in expected {
        let banner = format!("══ {group} ");
        let start = stdout
            .find(&banner)
            .unwrap_or_else(|| panic!("`--list` never reached group `{group}`:\n{stdout}"));
        // Up to the next banner, or to the end.
        let rest = &stdout[start + banner.len()..];
        let end = rest.find("══ ").unwrap_or(rest.len());
        let got = rest[..end]
            .lines()
            .filter(|l| l.trim_start().starts_with("$ "))
            .count();
        assert_eq!(
            got, want,
            "group `{group}` lists {got} command(s), expected {want}. If you added or \
             removed a gate deliberately, update the count here; if you did not, the \
             group's dispatch or one of its `run` lines is silently doing nothing.\n{stdout}"
        );
    }
    // The default run is the full set; a subset would silently under-check a developer
    // who typed no arguments.
    assert!(
        stdout.contains("nothing was executed"),
        "`--list` should execute nothing:\n{stdout}"
    );
}

/// `--help` is the only place a developer learns which groups exist, so it must not be
/// able to fall behind the group list — and it must work from any directory, since the
/// natural way to reach it is an absolute path out of a stale shell.
#[test]
fn preflight_help_lists_every_group_and_works_from_any_cwd() {
    // Deliberately NOT `current_dir(repo_root())`: an earlier version read its usage text
    // out of its own source with `sed -n '3,8p' "${BASH_SOURCE[0]}"` *after* `cd`-ing to
    // the repo root, so a relative invocation from elsewhere resolved the path against
    // the wrong directory and printed nothing.
    let out = Command::new("bash")
        .arg(preflight())
        .arg("--help")
        .current_dir(std::env::temp_dir())
        .output()
        .expect("run tools/preflight.sh --help");
    assert!(
        out.status.success(),
        "`--help` exited {:?}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    for want in [
        "Run the fast CI gates locally",
        "tools/preflight.sh check",
        "tools/preflight.sh --list",
    ] {
        assert!(
            stdout.contains(want),
            "`--help` is missing {want:?}:\n{stdout}"
        );
    }

    // Every group the script will actually dispatch has to be documented. Naming them
    // from the `--list` banners rather than from a literal here means a new group cannot
    // be added without `--help` growing it too.
    let full = Command::new("bash")
        .arg(preflight())
        .arg("--list")
        .current_dir(repo_root())
        .output()
        .expect("run tools/preflight.sh --list");
    let groups: Vec<String> = String::from_utf8_lossy(&full.stdout)
        .lines()
        .filter_map(|l| l.trim().strip_prefix("══ "))
        .filter_map(|l| l.split_whitespace().next())
        .map(str::to_string)
        .collect();
    assert!(!groups.is_empty(), "no group banners in `--list` output");
    for g in &groups {
        assert!(
            stdout.contains(g.as_str()),
            "`--help` never mentions the `{g}` group, so a developer reading it would not \
             know to run it:\n{stdout}"
        );
    }
}
