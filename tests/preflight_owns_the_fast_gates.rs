//! Guard: the `Check`, `Clippy`, `Format`, `Rustdoc` and `Tests (debug-assertions)` jobs
//! in `.github/workflows/ci.yml` must run their cargo commands **through
//! `tools/preflight.sh`**, and that script must actually fail when a gate fails (#1157).
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
//! Four invariants:
//!
//!  1. Those jobs contain **no inline `run: cargo …` step**, and neither skip
//!     themselves (`if:`) nor tolerate failure (`continue-on-error`). Each is a way for
//!     the two lists to diverge, or for a red gate to report green.
//!  2. Each of them **does** invoke `tools/preflight.sh <group>` with its own group.
//!  3. A failing command makes the script exit non-zero **from every position in the
//!     list**, not just the last one.
//!  4. A no-argument run is `ALL_GROUPS` minus `OPT_IN_GROUPS`, and every opt-in group
//!     is named by a job in `ci.yml` (#344). An opt-in group exists because it runs a
//!     test suite rather than a compile — `debug-assertions` is the only one — and the
//!     failure it invites is a gate that runs in neither place.
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

/// One `NAME=(a b c)` array literal out of `tools/preflight.sh`.
///
/// The script is the single owner of the group list, so these tests read the list
/// FROM it rather than keeping a second copy. A group added to `ALL_GROUPS` is
/// therefore covered by every assertion below the moment it is added — including
/// the count tripwire, which is the one place a literal remains (deliberately: it
/// is a tripwire against silent shrinkage, not a copy of the commands).
fn script_array(name: &str) -> Vec<String> {
    let src = std::fs::read_to_string(preflight()).expect("read tools/preflight.sh");
    let needle = format!("\n{name}=(");
    let start = src
        .find(&needle)
        .unwrap_or_else(|| panic!("`{name}=(` not found in tools/preflight.sh"))
        + needle.len();
    let rest = &src[start..];
    let end = rest
        .find(')')
        .unwrap_or_else(|| panic!("unterminated `{name}=(` in tools/preflight.sh"));
    rest[..end].split_whitespace().map(str::to_string).collect()
}

/// Every group the script knows about, opt-in ones included.
fn all_groups() -> Vec<String> {
    let groups = script_array("ALL_GROUPS");
    assert!(!groups.is_empty(), "ALL_GROUPS parsed empty");
    groups
}

/// The groups a no-argument run deliberately SKIPS (#344): they run a test suite
/// rather than a compile, so they are named explicitly or run by CI.
fn opt_in_groups() -> Vec<String> {
    script_array("OPT_IN_GROUPS")
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

    for (job, group) in [
        ("check", "check"),
        ("clippy", "clippy"),
        ("fmt", "fmt"),
        ("rustdoc", "rustdoc"),
        // #344. This one is a TEST job, not a compile/lint gate, and it is the
        // only group a no-argument `tools/preflight.sh` skips — so the workflow
        // is the only place it is guaranteed to run. That makes delegation
        // matter more here, not less: an inline `cargo test` in the job would be
        // a command no developer can reproduce with a documented local call.
        ("debug-assertions", "debug-assertions"),
    ] {
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

    // Every group whose commands are `cargo` invocations, so the fake below intercepts
    // them. `public-api` shells out to `tools/update-public-api.sh` instead and is covered
    // by `the_failure_diagnostic_names_the_group_that_actually_failed`.
    for group in ["fmt", "check", "clippy", "rustdoc", "debug-assertions"] {
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

    // `cargo doc` exits 0 on link warnings. `-Dwarnings` is the ENTIRE gate: drop it and
    // both commands stay in the list, still build docs, still take ~8s — and pass over a
    // tree with 146 broken links. That is the neutered-gate shape this test exists for.
    let docs = listed_commands("rustdoc");
    for cmd in &docs {
        assert!(
            cmd.contains("RUSTDOCFLAGS=-Dwarnings"),
            "a docs command does not deny warnings, so rustdoc exits 0 and the gate \
             passes on any number of broken links:\n  {cmd}"
        );
        // Set via `env` in the argument vector rather than a `VAR=x run ...` prefix.
        // A prefix reaches rustdoc too, but `run` echoes `$*` and a prefix is not in it,
        // so `--list` would advertise a bare `cargo doc` that does NOT fail on warnings.
        assert!(
            cmd.starts_with("env "),
            "the docs command sets RUSTDOCFLAGS outside its argument vector, so \
             `--list` prints a command that behaves differently when pasted:\n  {cmd}"
        );
    }
    assert!(
        docs.iter().any(|c| c.contains("-p ferx-tools")),
        "no docs command covers the workspace members, so `ferx-tools` and `ferx-cli` \
         doc comments are gated by nothing — a root `cargo doc` documents only the root \
         package (#1114).\n  {}",
        docs.join("\n  ")
    );

    // The `debug-assertions` group is a gate ONLY because it builds on the default
    // dev profile (#344). Every named profile in this repo — `release`, `ci-test`,
    // `ci-fast` — has `debug-assertions` off, `ci-test`/`ci-fast` by inheriting
    // `release`, so a `--profile`/`--release` flag added to one of these commands
    // would leave two `cargo test` lines in the list, still run 4000-odd tests,
    // still take minutes, and compile every `debug_assert!` in the crate back down
    // to nothing. That is the exact shape of the neutered gate this test exists for:
    // the command list cannot show it, only its flags can.
    let dbg = listed_commands("debug-assertions");
    assert!(
        !dbg.is_empty(),
        "the `debug-assertions` group lists no commands"
    );
    for cmd in &dbg {
        assert!(
            !cmd.contains("--profile") && !cmd.contains("--release"),
            "`{cmd}` selects an explicit profile. Every named profile in this repo \
             inherits `release`, where `debug-assertions = false`, so the guards this \
             group exists to execute would compile to nothing while the job stayed \
             green (#344). Drop the flag; the dev default is the gate."
        );
        assert!(
            cmd.contains("cargo test "),
            "`{cmd}` is in the `debug-assertions` group but does not RUN anything. \
             A `cargo check`/`clippy` line compiles the assertions and never \
             evaluates them, which is the blind spot, not the fix (#344)."
        );

        // The POSITIVE half, and the one the flag check above cannot supply.
        // Rejecting `--profile` only rules out the ways of turning the guards off
        // that are VISIBLE in the command string. Two are not: `[profile.dev]
        // debug-assertions = false` in Cargo.toml, and a
        // `CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=false` in the workflow environment.
        // Either would leave
        // this group's two commands intact, run all 4366 tests, and keep every
        // assertion in this file green while the job verified nothing —
        // an invariant held by absence is exactly what #344 was filed about.
        // `debug_assertion_canary` in `src/lib.rs` closes that, but only when armed,
        // so the arming has to be pinned as tightly as the flags are.
        //
        // `starts_with`, not `contains`: it must be the argv-leading `env` form, so
        // `run` echoes it and `--list` tells the truth about what will run.
        assert!(
            cmd.starts_with("env FERX_REQUIRE_DEBUG_ASSERTIONS=1 cargo "),
            "`{cmd}` does not arm the debug-assertions canary. Prefix it with \
             `env FERX_REQUIRE_DEBUG_ASSERTIONS=1` (in the argument vector, so \
             `--list` shows it): without that, `debug_assertion_canary` in \
             src/lib.rs passes vacuously and nothing in this group ever verifies \
             that `debug_assert!` is actually live (#344)."
        );
    }

    // Same non-nesting trap as `check`: `--features ci` compiles neither
    // `src/survival`, `src/markov`, `src/categorical` nor `src/nn`, and all four
    // carry `debug_assert!`s.
    assert!(
        dbg.iter().any(|c| c.contains("ci,markov,nn")),
        "no `debug-assertions` command builds `ci,markov,nn`, so every guard in the \
         feature-gated endpoint and NN code is as dead here as in the release-profile \
         jobs:\n  {}",
        dbg.join("\n  ")
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

    // Every group by name, not a bare `--list`: since #344 a no-argument run is
    // `ALL_GROUPS` minus `OPT_IN_GROUPS`, so a bare `--list` would walk past the
    // opt-in group and it would get no count tripwire at all — exactly the
    // silent-shrinkage hole this test exists to close. The names come from the
    // script's own array, so there is still no second copy of the group list.
    let out = Command::new("bash")
        .arg(&script)
        .arg("--list")
        .args(all_groups())
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
    let expected: [(&str, usize); 7] = [
        ("fmt", 1),        // cargo fmt --all -- --check
        ("check", 5),      // ci · ci,survival,slow-tests · ci,markov · ci,nn,slow-tests · members
        ("clippy", 2),     // ferx-core --all-targets · members
        ("rustdoc", 2),    // ferx-core rustdoc · members
        ("docs", 2),       // cargo test -p docs-lint · cargo clippy -p docs-lint
        ("public-api", 1), // tools/update-public-api.sh --check
        // #344: `--features ci` · `--features ci,markov,nn`. The second line is
        // not padding — `ci` compiles neither `src/survival`, `src/markov`,
        // `src/categorical` nor `src/nn`, and all four carry `debug_assert!`s, so
        // dropping it would leave those guards exactly as dead as they are in
        // every release-profile job.
        ("debug-assertions", 2),
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
    assert!(
        stdout.contains("nothing was executed"),
        "`--list` should execute nothing:\n{stdout}"
    );
}

/// A no-argument run is **every group except the declared opt-in ones** — and an
/// opt-in group is only legitimate because CI runs it.
///
/// Both halves matter and they fail in opposite directions. Drop a group from the
/// default set without declaring it opt-in and a developer who types no arguments
/// silently under-checks; declare one opt-in without a CI job that names it and
/// the gate runs *nowhere*, which is strictly worse than the blind spot #344 was
/// filed about — there the guards were at least compiled out honestly.
#[test]
fn the_default_run_is_every_group_that_is_not_opt_in_and_ci_runs_the_rest() {
    let all = all_groups();
    let opt_in = opt_in_groups();

    for g in &opt_in {
        assert!(
            all.contains(g),
            "`{g}` is in OPT_IN_GROUPS but not in ALL_GROUPS, so naming it is an \
             argument error and nothing can ever run it"
        );
    }

    let out = Command::new("bash")
        .arg(preflight())
        .arg("--list")
        .current_dir(repo_root())
        .output()
        .expect("run tools/preflight.sh --list");
    assert!(
        out.status.success(),
        "`--list` exited {:?}",
        out.status.code()
    );
    let walked: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().strip_prefix("══ "))
        .filter_map(|l| l.split_whitespace().next())
        .map(str::to_string)
        .collect();

    let want: Vec<String> = all
        .iter()
        .filter(|g| !opt_in.contains(g))
        .cloned()
        .collect();
    assert_eq!(
        walked, want,
        "a no-argument run walks {walked:?} but ALL_GROUPS minus OPT_IN_GROUPS is \
         {want:?}. Either a gate silently left the default run, or a group was added \
         to ALL_GROUPS and to neither list's intent."
    );

    // The other half: each opt-in group is invoked by name in ci.yml — read from the
    // file's parsed `run:` steps, NOT from its raw text. A YAML *comment* that merely
    // mentions `tools/preflight.sh <group>` would satisfy a `contains` check while
    // nothing executed the group, and this workflow is full of comments that name
    // exactly these commands, so that check would have been self-satisfying. Applied
    // to the whole document rather than one job body on purpose: moving the group
    // between jobs is fine, running it nowhere is not.
    let yml = ci_yml();
    let executed = run_commands(&yml);
    for g in &opt_in {
        let want = format!("tools/preflight.sh {g}");
        assert!(
            executed.iter().any(|c| c == &want),
            "group `{g}` is opt-in locally and no job in ci.yml RUNS `{want}`, so it \
             is a gate that nothing executes. Naming it in a comment does not count — \
             this reads the `run:` steps.\nrun steps found:\n  {}",
            executed.join("\n  ")
        );
    }
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

    // Every group the script will actually dispatch has to be documented. Read from
    // the script's own `ALL_GROUPS` rather than from a literal here, so a new group
    // cannot be added without `--help` growing it too — and `ALL_GROUPS` rather than
    // the `--list` banners of a no-argument run, because since #344 that run walks
    // only the default set and an opt-in group is precisely the one a developer
    // will never discover any other way (#344).
    let groups = all_groups();
    for g in &groups {
        assert!(
            stdout.contains(g.as_str()),
            "`--help` never mentions the `{g}` group, so a developer reading it would not \
             know to run it:\n{stdout}"
        );
    }
}

/// A crate-level `#![allow(rustdoc::...)]` is a neutered gate the command list cannot see.
///
/// The `rustdoc` group can carry `-Dwarnings` on every line, run on every PR, and still pass
/// over a tree whose rendered HTML is full of `[<code>foo</code>]` — because the lint that
/// would have fired was switched off inside the crate. `private_intra_doc_links` is the
/// one that matters here: rustdoc drops a link to a `pub(crate)` item and leaves the
/// markdown brackets in the public page, exactly the defect the gate exists to catch. An
/// `allow` also silences every FUTURE such link, so the class leaves the gate permanently
/// rather than for one PR.
///
/// Rustdoc lints are therefore fixed at the link, not at the crate root: an internal target
/// becomes inline code, a public target gets a resolvable path. A per-item `#[allow]` is out
/// of scope for this test — it is visible in review at the place it applies; a crate-level
/// one is a single line thousands of lines from anything it affects.
#[test]
fn no_crate_level_allow_switches_a_rustdoc_lint_off() {
    let lib = std::fs::read_to_string(repo_root().join("src/lib.rs")).expect("read src/lib.rs");
    let offenders: Vec<&str> = lib
        .lines()
        .map(|l| l.trim())
        .filter(|l| l.starts_with("#!["))
        .filter(|l| l.contains("allow") && l.contains("rustdoc::"))
        .collect();
    assert!(
        offenders.is_empty(),
        "src/lib.rs switches a rustdoc lint off crate-wide. The `rustdoc` group then still \
         runs, still denies warnings, and still passes while the public HTML renders the \
         unresolved links with their brackets intact — and it does so for every link added \
         after this line, not just today's. Fix the links instead: inline code for an \
         internal target, a resolvable path for a public one.\n  {}",
        offenders.join("\n  ")
    );
}
