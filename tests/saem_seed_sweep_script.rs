//! `tools/saem-seed-sweep.sh` (#2602): the seed rewrite and the paired build comparison.
//!
//! Nothing here builds ferx. The script is copied into a throwaway git repo with two
//! commits, so it has two revisions to resolve, and each revision's `ferx` is a stub
//! the script finds already "built" in its workdir. The stub reads the SAEM seed the
//! way the parser does: `seed` and its alias `saem_seed` set the same option, the last
//! line in `[fit_options]` wins, and other blocks are not read. It prints an IS
//! objective that depends on that seed.
//!
//! Not feature-gated. Unix-only, since the script and the stubs are bash.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch(slot: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("seed-sweep-{slot}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn git(repo: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .output()
        .expect("git on PATH");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

/// A stub `ferx` for one revision. Objective = -(1000 * seed) - offset(seed), where
/// `offset` is the per-revision part the comparison has to recover. `nan_seed` makes
/// that seed print `NaN`, which must become NA and drop out of the pairs.
fn stub(workdir: &Path, sha: &str, offset_awk: &str, nan_seed: Option<u32>) {
    let dir = workdir.join(format!("target-{sha}")).join("ci-test");
    std::fs::create_dir_all(&dir).unwrap();
    let nan = nan_seed.map_or("-1".to_string(), |s| s.to_string());
    let body = format!(
        r#"#!/usr/bin/env bash
s=$(awk '/^[[:space:]]*\[/ {{ f = ($0 ~ /\[fit_options\]/) }}
         f && /^[[:space:]]*(seed|saem_seed)[[:space:]]*=/ {{ sub(/.*=[[:space:]]*/, ""); v = $0 }}
         END {{ print v }}' "$1")
if [ "$s" = "{nan}" ]; then v=NaN
else v=$(awk -v s="$s" 'BEGIN {{ printf "%.4f", -(1000 * s) - ({offset_awk}) }}'); fi
echo "  -2 log L (IS): $v  (MC SE = 0.5, K = 1000)"
"#
    );
    let path = dir.join("ferx");
    std::fs::write(&path, body).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn sweep_rewrites_saem_seed_alias_and_compares_builds_paired() {
    let root = scratch("paired");
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("tools")).unwrap();
    let script = repo.join("tools").join("saem-seed-sweep.sh");
    std::fs::copy(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/saem-seed-sweep.sh"),
        &script,
    )
    .unwrap();
    git(&repo, &["init", "-q"]);
    git(&repo, &["add", "-A"]);
    git(&repo, &["commit", "-q", "-m", "a"]);
    let sha_a = git(&repo, &["rev-parse", "HEAD"]);
    git(&repo, &["commit", "-q", "--allow-empty", "-m", "b"]);
    let sha_b = git(&repo, &["rev-parse", "HEAD"]);

    let workdir = root.join("work");
    // B is A shifted by 1.0 on even seeds and 1.1 on odd ones; its seed 5 is NaN.
    stub(&workdir, &sha_a, "0", None);
    stub(&workdir, &sha_b, "-(1 + 0.1 * (s % 2))", Some(5));

    // `saem_seed` sits BELOW where the script injects `seed`; under source-order
    // application it would pin every run to seed 7. `[simulation]`'s own `seed` must
    // survive the rewrite.
    let model = root.join("m.ferx");
    std::fs::write(
        &model,
        "[parameters]\n  theta T(1, 0, 10)\n\n[fit_options]\n  method = [saem, imp]\n  \
         imp_eval_only = true\n  saem_seed = 7\n  imp_seed = 9\n\n[simulation]\n  seed = 42\n",
    )
    .unwrap();
    let data = root.join("d.csv");
    std::fs::write(&data, "ID,TIME,DV\n").unwrap();

    let out = Command::new("bash")
        .arg(&script)
        .arg("--model")
        .arg(&model)
        .arg("--data")
        .arg(&data)
        .args([
            "--rev",
            &sha_a,
            "--rev",
            &sha_b,
            "--seeds",
            "1 2 3 4 5",
            "--jobs",
            "2",
        ])
        .arg("--workdir")
        .arg(&workdir)
        .output()
        .unwrap();
    let text =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{text}");

    // The swept seed reached the fit: with `saem_seed = 7` surviving, every row of
    // revision A would read -7000.
    for s in 1..=5 {
        let row = format!(
            "{}\t{s}\t{}\t{:.4}",
            &sha_a[..8],
            s + 1000,
            -1000.0 * s as f64
        );
        assert!(text.contains(&row), "missing `{row}` in\n{text}");
    }
    let run_model = std::fs::read_to_string(
        workdir
            .join("runs")
            .join(&sha_a)
            .join("s3-i1003")
            .join("model.ferx"),
    )
    .unwrap();
    assert!(!run_model.contains("saem_seed"), "{run_model}");
    assert!(!run_model.contains("imp_seed = 9"), "{run_model}");
    assert!(
        run_model.contains("  seed = 3\n  imp_seed = 1003\n"),
        "{run_model}"
    );
    assert!(
        run_model.contains("[simulation]\n  seed = 42"),
        "{run_model}"
    );

    // Paired: diffs 1.1, 1.0, 1.1, 1.0 (seed 5 is NA on B) -> mean 1.05,
    // sd 0.057735, SE 0.028868, t 36.37. The between-seed spread (1000 per seed) is
    // what an unpaired Welch SE would see: about 645 here, t about 0.002.
    assert!(
        text.contains(&format!(
            "{sha_b} - {sha_a} (paired, 4 pairs): +1.050 (SE 0.029, t = +36.37); \
             {sha_b} higher on 4 of 4"
        )),
        "{text}"
    );
    assert!(
        text.contains(&format!("{sha_b}    4    1")),
        "B must count its NaN as NA:\n{text}"
    );
}
