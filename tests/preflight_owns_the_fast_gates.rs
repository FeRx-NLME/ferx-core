//! Guard: the `Check`, `Clippy` and `Format` jobs in `.github/workflows/ci.yml` must run
//! their cargo commands **through `tools/preflight.sh`**, never inline (#1157).
//!
//! The point of that script is not documentation — it is that CI executes the same list a
//! developer runs before pushing, so "green locally" and "green in CI" cannot mean
//! different things. A script that only *describes* the commands drifts the first time
//! somebody edits the workflow, and drifts silently: the local gate keeps passing while
//! CI checks something else. That is the failure this exists to prevent, and it has a
//! precedent — on #1133 a suite of 4634 passing tests was reported green while
//! `--features ci,nn,slow-tests` had been failing to compile for an entire commit,
//! because the local run and the CI run were not the same set.
//!
//! Two invariants, both plain assertions on the workflow text:
//!
//!  1. Those three jobs contain **no inline `run: cargo …` step**. Adding one is exactly
//!     how the two lists diverge, and it is the easy thing to do when adding a gate.
//!  2. Each of them **does** invoke `tools/preflight.sh <group>` with its own group.
//!
//! Deliberately NOT asserted: that the script's command list matches some second copy
//! here. There is no second copy — the script is the only list, which is the whole design.
//! What this test pins is that the workflow keeps *delegating* to it.
//!
//! The `public-api` job is out of scope: it needs a pinned dated nightly and a pinned
//! `cargo-public-api` installed as separate workflow steps, so it calls
//! `tools/update-public-api.sh --check` directly. `preflight.sh public-api` runs that same
//! script, so a developer's full local run still covers it.
//!
//! Not feature-gated — it must run in the base `--features ci` job, the one job guaranteed
//! to build on every PR.

use std::path::PathBuf;

const SCRIPT_DIR: &str = "tools";
const SCRIPT_FILE: &str = "preflight.sh";

/// `tools/preflight.sh`, assembled from its two components.
///
/// Spelled this way on purpose. `tests/slow_test_path_filter.rs` (#949) byte-scans every
/// `tests/*.rs` for double-quoted literals and treats any `<tracked-root>/<rest>` it finds as
/// a **data input the heavy fit suite reads**, then requires `slow-tests.yml` to list that
/// root under `push: paths:`. Its own docs call the scan "deliberately crude" and rely on the
/// must-be-a-tracked-root filter to drop false positives — but `tools` *is* a tracked root, so
/// a bare path literal in an assertion message here registers this script as a fit input and
/// fails that guard.
///
/// It is not one: `slow-tests.yml` never invokes preflight, and nothing in this file can
/// change a fit result. The alternative fix — adding `tools/**` to that workflow's `paths:` —
/// would encode the opposite claim and trigger a multi-hour suite on unrelated tooling edits.
fn script_rel() -> String {
    format!("{SCRIPT_DIR}/{SCRIPT_FILE}")
}

fn ci_yml() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(".github")
        .join("workflows")
        .join("ci.yml");
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

/// Every `run:` command in a job body, one entry per step.
fn run_commands(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|l| l.trim_start().strip_prefix("run: "))
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

        // (1) no inline cargo — that is how the two lists diverge.
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

        // (2) it must actually call its own group.
        let want = format!("{} {group}", script_rel());
        assert!(
            cmds.iter().any(|c| c == &want),
            "job `{job}` never runs `{want}`; its steps are:\n  {}",
            cmds.join("\n  ")
        );
    }
}

#[test]
fn preflight_is_executable_and_lists_every_group() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = root.join(SCRIPT_DIR).join(SCRIPT_FILE);
    assert!(script.is_file(), "{} is missing", script_rel());

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&script).unwrap().permissions().mode();
        assert!(
            mode & 0o111 != 0,
            "{} is not executable (mode {mode:o}); CI runs it directly",
            script_rel()
        );
    }

    // `--list` prints the commands without executing them, so this stays a fast unit-tier
    // check rather than a compile.
    let out = std::process::Command::new("bash")
        .arg(&script)
        .arg("--list")
        .current_dir(&root)
        .output()
        .expect("run tools/preflight.sh --list");
    assert!(
        out.status.success(),
        "{} --list exited {:?}\nstderr:\n{}",
        script_rel(),
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    // Per group: the banner is NOT evidence the group ran. The driver loop prints it
    // before dispatching, so a group whose `case` arm was gutted to `:` still shows a
    // header and exits 0 — CI would run `preflight.sh clippy`, lint nothing, and pass.
    // That mutation survived the first draft of this test. So count the commands each
    // group actually emits, which is the only thing that distinguishes "ran" from
    // "silently did nothing".
    //
    // The counts are a tripwire against silent SHRINKAGE, not a second copy of the
    // command list — deleting a `cargo check` feature set has to be a deliberate edit
    // here too, and lowering a number is a reviewable line in the diff.
    let expected: [(&str, usize); 4] = [
        ("fmt", 1),        // cargo fmt --all -- --check
        ("check", 5),      // ci · ci,survival,slow-tests · ci,markov · ci,nn,slow-tests · members
        ("clippy", 2),     // ferx-core --all-targets · members
        ("public-api", 1), // tools/update-public-api.sh --check
    ];

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

/// `--help` extracts its usage block from the script's own header by LINE NUMBER, which
/// silently prints the wrong lines if that header ever grows or shrinks. Rather than make
/// the extraction clever, pin the output.
///
/// This also covers a portability trap the first version walked into: the strip was
/// `s/^#\s\?//`, and BSD sed (macOS, where most of us edit) does not understand `\s` — so
/// every `#` survived locally while the same line worked in CI. A local/CI split inside the
/// tool whose entire purpose is removing local/CI splits.
#[test]
fn preflight_help_renders_its_usage_block() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::process::Command::new("bash")
        .arg(root.join(SCRIPT_DIR).join(SCRIPT_FILE))
        .arg("--help")
        .current_dir(&root)
        .output()
        .expect("run tools/preflight.sh --help");
    assert!(
        out.status.success(),
        "`--help` exited {:?}",
        out.status.code()
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    for want in [
        "Run the fast CI gates locally",
        "preflight.sh check",
        "preflight.sh --list",
        "Groups: fmt, check, clippy, public-api",
    ] {
        assert!(
            stdout.contains(want),
            "`--help` is missing {want:?} — the header line range in the `-h|--help` arm \
             has probably drifted:\n{stdout}"
        );
    }
    // No line may still START with `#`. Checking for `#` anywhere would be wrong: the usage
    // lines carry legitimate inline comments (`tools/preflight.sh check  # just the …`).
    let leftover: Vec<&str> = stdout
        .lines()
        .filter(|l| l.trim_start().starts_with('#'))
        .collect();
    assert!(
        leftover.is_empty(),
        "`--help` left comment markers at line start — the sed strip is not working (BSD \
         sed does not support `\\s`):\n  {}",
        leftover.join("\n  ")
    );
}
