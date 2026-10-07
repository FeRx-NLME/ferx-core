//! Guard: `tools/linux-test.sh` builds the container layout #1688 measured, and its
//! failure messages say what actually failed.
//!
//! The script is how a contributor gets a Linux number for a fit (Linux is the reference
//! platform: macOS and Linux disagree on fits through the OS libm). Every piece of its
//! `docker run` is there because the plain version was measured broken:
//!
//! - the tree mounted read-write at `/src` lets the `gen_*_anchor` tests overwrite the
//!   committed `nonmem_anchor/*.csv` in the host tree; mounted `:ro` there, they fail;
//! - without the git common dir mounted at its own path, every `git ls-files` test fails;
//! - above one build job, the full sweep is OOM-killed on Docker Desktop's default VM;
//! - an arch-shared target volume fails with `exec format error`, not a clean rebuild.
//!
//! `--dry-run` prints the command without touching docker, so these run on every PR with
//! no daemon. Tier 2: they spawn `bash` and read no numerical path.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// A private directory for fake `docker` binaries. PID + nonce, so concurrent test
/// binaries never share a fixed path (the #1322 race).
fn scratch_dir() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("ferx-linux-test-{}-", std::process::id()))
        .tempdir()
        .expect("create a scratch dir")
}

/// An executable shell script at `dir/name`.
fn write_exe(dir: &Path, name: &str, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join(name);
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write fake docker");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    path
}

fn run_script(docker: &Path, args: &[&str]) -> Output {
    Command::new("bash")
        .arg(repo_root().join("tools/linux-test.sh"))
        .args(args)
        .env("FERX_DOCKER", docker)
        .current_dir(repo_root())
        .output()
        .expect("spawn bash tools/linux-test.sh")
}

/// The two halves of a `--dry-run`: the `docker run` argv (whitespace-split) and the
/// in-container script's lines.
struct DryRun {
    argv: Vec<String>,
    inner: Vec<String>,
}

fn dry_run(args: &[&str]) -> DryRun {
    let dir = scratch_dir();
    // A docker that leaves evidence if anything calls it. `/nonexistent` would also do for
    // "not called", but only by failing loudly; this one proves the negative even if a
    // regression tolerated the failure.
    let sentinel = dir.path().join("docker-was-called");
    let docker = write_exe(
        dir.path(),
        "docker",
        &format!("touch '{}'\nexit 0", sentinel.display()),
    );
    let out = run_script(&docker, args);
    assert!(
        out.status.success(),
        "--dry-run must exit 0 without docker; got {:?}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !sentinel.exists(),
        "--dry-run invoked the docker binary — it must print the command and touch nothing"
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    let mut lines = stdout.lines();
    assert_eq!(
        lines.next(),
        Some("# docker command:"),
        "dry-run layout:\n{stdout}"
    );
    let argv = lines
        .next()
        .expect("the docker argv line")
        .split_whitespace()
        .map(str::to_string)
        .collect();
    assert_eq!(
        lines.next(),
        Some("# inner script ($inner):"),
        "dry-run layout:\n{stdout}"
    );
    DryRun {
        argv,
        inner: lines.map(str::to_string).collect(),
    }
}

impl DryRun {
    /// Every value passed with `flag` (`-v`, `-e`, `--platform`).
    fn values(&self, flag: &str) -> Vec<&str> {
        self.argv
            .windows(2)
            .filter(|w| w[0] == flag)
            .map(|w| w[1].as_str())
            .collect()
    }

    fn inner_line(&self, pred: impl Fn(&str) -> bool) -> Option<usize> {
        self.inner.iter().position(|l| pred(l))
    }
}

#[test]
fn dry_run_prints_the_measured_container_layout() {
    let d = dry_run(&["--dry-run", "--", "test", "--test", "foo"]);
    let mounts = d.values("-v");
    let envs = d.values("-e");

    // The tree is read-only at /ro, and nothing of the host is writable at /src.
    assert!(
        mounts.iter().any(|m| m.ends_with(":/ro:ro")),
        "the tree must be mounted READ-ONLY at /ro (a writable mount lets gen_*_anchor \
         rewrite committed fixtures in the host tree): {mounts:?}"
    );
    assert!(
        !mounts.iter().any(|m| m.contains(":/src")),
        "nothing may be mounted at /src — it must be a copy, or the gen_*_anchor writes \
         reach the host tree: {mounts:?}"
    );

    // The copy-in excludes the build dir and the git dir.
    let tar = d
        .inner_line(|l| l.starts_with("tar -C /ro"))
        .map(|i| d.inner[i].clone())
        .expect("the inner script copies /ro with tar");
    assert!(
        tar.contains("--exclude=./target") && tar.contains("--exclude=./.git "),
        "the copy must exclude target/ and .git: {tar}"
    );

    // The git common dir, read-only, at its own absolute path.
    let git_mount = mounts.iter().find(|m| {
        let parts: Vec<&str> = m.split(':').collect();
        parts.len() == 3 && parts[0] == parts[1] && parts[0].ends_with(".git")
    });
    assert!(
        git_mount.is_some_and(|m| m.ends_with(":ro")),
        "the git common dir must be mounted read-only at its own absolute path, or every \
         `git ls-files` test fails in the container: {mounts:?}"
    );

    assert!(
        envs.contains(&"RAYON_NUM_THREADS=1"),
        "RAYON_NUM_THREADS=1 must be set, as ci.yml / slow-tests.yml set it: {envs:?}"
    );
    assert!(
        envs.contains(&"CARGO_BUILD_JOBS=1"),
        "CARGO_BUILD_JOBS must default to 1 (2 is OOM-killed on the default 8 GB VM): {envs:?}"
    );
    assert!(
        envs.contains(&"CARGO_TARGET_DIR=/target"),
        "CARGO_TARGET_DIR must point at the /target volume, or every run is a cold build in \
         the throwaway /src/target: {envs:?}"
    );

    // Marker first, then cargo with the passthrough args.
    let marker = d
        .inner_line(|l| l.starts_with("echo \"ferx-platform: Linux/"))
        .expect("the inner script must echo the `ferx-platform: Linux/` marker");
    let cargo = d
        .inner_line(|l| l.starts_with("cargo "))
        .expect("the inner script must run cargo");
    assert!(
        marker < cargo,
        "the platform marker must be printed before cargo runs:\n{}",
        d.inner.join("\n")
    );
    assert!(
        d.inner[cargo].starts_with("cargo test --test foo "),
        "the args after `--` must be passed to cargo, in order, right after `cargo`: {}",
        d.inner[cargo]
    );

    // The script's exit status is cargo's: a red Linux run must not report success.
    assert!(
        d.inner.get(cargo + 1).map(String::as_str) == Some("code=$?")
            && d.inner.last().map(String::as_str) == Some("exit \"$code\""),
        "the inner script must capture cargo's status on the next line and exit with it:\n{}",
        d.inner.join("\n")
    );

    // `set -e` is off around cargo, or a failing cargo exits the script before the OOM
    // hint below can ever print.
    assert!(
        d.inner[marker..cargo].iter().any(|l| l == "set +e"),
        "`set +e` must sit between the marker and cargo, or the OOM hint is unreachable:\n{}",
        d.inner.join("\n")
    );

    // An OOM-killed process (6.9 GB peak on an 8 GB VM at -j1) gets the hint that blames
    // the VM, not the test — and only when cargo failed with a SIGKILL in its stderr.
    let hint = d
        .inner_line(|l| {
            l.contains("SIGKILLed (likely out of memory) — lower --jobs or raise Docker")
        })
        .expect("the inner script must carry the OOM hint");
    assert!(
        hint > cargo
            && d.inner[hint - 1].contains("\"$code\" -ne 0")
            && d.inner[hint - 1].contains("grep -q 'SIGKILL'"),
        "the OOM hint must be gated on a failed cargo whose stderr shows SIGKILL:\n{}",
        d.inner.join("\n")
    );

    // --jobs N overrides the default, and only it.
    let d3 = dry_run(&["--dry-run", "--jobs", "3", "--", "test"]);
    let envs3 = d3.values("-e");
    assert!(
        envs3.contains(&"CARGO_BUILD_JOBS=3") && !envs3.contains(&"CARGO_BUILD_JOBS=1"),
        "--jobs 3 must set CARGO_BUILD_JOBS=3: {envs3:?}"
    );
}

#[test]
fn amd64_switches_platform_and_volumes_together() {
    // Both sides in one test: a flag stuck on either branch reddens it.
    let native = dry_run(&["--dry-run", "--", "test"]);
    let amd = dry_run(&["--amd64", "--dry-run", "--", "test"]);

    assert!(
        native.values("--platform").is_empty(),
        "without --amd64 no --platform may be passed (native arch): {:?}",
        native.argv
    );
    assert_eq!(
        amd.values("--platform"),
        vec!["linux/amd64"],
        "--amd64 must pass --platform linux/amd64"
    );

    let nv = native.values("-v");
    let av = amd.values("-v");
    assert!(
        nv.contains(&"ferx-linux-target:/target")
            && nv.contains(&"ferx-linux-registry:/usr/local/cargo/registry"),
        "native run uses the unsuffixed volumes: {nv:?}"
    );
    assert!(
        av.contains(&"ferx-linux-target-amd64:/target")
            && av.contains(&"ferx-linux-registry-amd64:/usr/local/cargo/registry"),
        "--amd64 must use its own -amd64 volumes (a cross-arch target dir fails with \
         `exec format error`, not a rebuild): {av:?}"
    );
}

#[test]
fn missing_docker_and_dead_daemon_are_told_apart() {
    let dir = scratch_dir();

    // No binary at all.
    let out = run_script(Path::new("/nonexistent/docker"), &["--", "test"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "no docker binary: exit 2; stderr: {err}"
    );
    assert!(
        err.contains("docker not found") && !err.contains("daemon"),
        "no docker binary must say `docker not found` and give no daemon advice: {err}"
    );

    // A binary whose `info` fails: the daemon is down.
    let docker = write_exe(dir.path(), "docker", "exit 1");
    let out = run_script(&docker, &["--", "test"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "daemon down: exit 2; stderr: {err}"
    );
    assert!(
        err.contains("docker daemon not reachable (docker info failed)")
            && err.contains("start the Docker daemon and retry")
            && !err.to_lowercase().contains("install"),
        "a present binary with a dead daemon must name the daemon, not suggest installing \
         docker: {err}"
    );
}

/// The free-disk guard, both sides of the gate in one test: a threshold above any disk
/// refuses before `docker run` with each sentence of its message, `0` skips the check and
/// reaches `docker run`, and a non-number is refused. On 2026-10-07 a slow sweep filled the
/// host disk and Docker restarted mid-build; the guard stops that before it starts.
#[test]
fn low_host_disk_is_refused_before_docker_run() {
    let dir = scratch_dir();
    let marker = dir.path().join("ran");
    // `info` succeeds; `run` leaves a marker, so the test sees whether the container started.
    let docker = write_exe(
        dir.path(),
        "docker",
        &format!(
            "if [ \"$1\" = run ]; then touch '{}'; fi\nexit 0",
            marker.display()
        ),
    );
    let run = |min_free: &str| {
        Command::new("bash")
            .arg(repo_root().join("tools/linux-test.sh"))
            .args(["--", "test"])
            .env("FERX_DOCKER", &docker)
            .env("FERX_LINUX_MIN_FREE_GB", min_free)
            .current_dir(repo_root())
            .output()
            .expect("spawn bash tools/linux-test.sh")
    };

    // No disk has a billion GB free: refused, and the container never starts.
    let out = run("1000000000");
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "low disk: exit 2; stderr: {err}"
    );
    for must in [
        "GB free on the host disk, below the 1000000000 GB a build may need.",
        "Free space first (docker system df; a merged branch's build folder: /ferx-clean)",
        "or set FERX_LINUX_MIN_FREE_GB to override.",
    ] {
        assert!(err.contains(must), "low-disk message lacks {must:?}: {err}");
    }
    assert!(
        !marker.exists(),
        "docker run started despite the low-disk refusal"
    );

    // `0` skips the check: the run goes ahead.
    let out = run("0");
    assert_eq!(
        out.status.code(),
        Some(0),
        "FERX_LINUX_MIN_FREE_GB=0 must skip the check; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        marker.exists(),
        "FERX_LINUX_MIN_FREE_GB=0 did not reach docker run"
    );

    // A non-number is refused rather than silently compared.
    let out = run("forty");
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "bad value: exit 2; stderr: {err}"
    );
    assert!(
        err.contains("FERX_LINUX_MIN_FREE_GB must be a whole number of GB, got 'forty'"),
        "{err}"
    );
}
