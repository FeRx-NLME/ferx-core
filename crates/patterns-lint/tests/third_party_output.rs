//! No verbatim third-party tool output in the tree (#3182): a NONMEM listing carries the
//! license holder's details and NONMEM's own text, a PsN log carries PsN's. Commit the
//! extracted numbers instead (`tools/nonmem-extract.py` for NONMEM).

use patterns_lint::repo_root;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Printed by NONMEM in every listing; spelled in two parts so this file does not match.
const LICENSE_LINE: &str = concat!("License ", "Registered to:");

/// Files PsN's `scm` writes into its run directory.
const PSN_LOGS: [&str; 3] =
    ["scmlog.txt", "covariate_statistics.txt", "psn_version_and_options.txt"];

/// Every tracked file still present in the working tree. Untracked and ignored files
/// (a listing just produced by the regeneration recipe) are not committed, so not checked.
fn tracked(root: &Path) -> Vec<PathBuf> {
    let out = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(root)
        .output()
        .expect("run git ls-files");
    assert!(out.status.success(), "git ls-files failed");
    out.stdout
        .split(|&b| b == 0)
        .filter(|p| !p.is_empty())
        .map(|p| root.join(String::from_utf8_lossy(p).as_ref()))
        .filter(|p| p.is_file())
        .collect()
}

fn is_listing(path: &Path, bytes: &[u8]) -> bool {
    let name = path.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase());
    let line = LICENSE_LINE.as_bytes();
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case("lst"))
        || name.is_some_and(|n| PSN_LOGS.contains(&n.as_str()))
        || bytes.windows(line.len()).any(|w| w.eq_ignore_ascii_case(line))
}

#[test]
fn no_verbatim_tool_output_is_committed() {
    let root = repo_root();
    let all = tracked(&root);
    let found: Vec<_> = all
        .iter()
        .filter(|p| is_listing(p, &std::fs::read(p).expect("read file")))
        .map(|p| p.strip_prefix(&root).unwrap_or(p))
        .collect();
    assert!(
        found.is_empty(),
        "verbatim tool output: {found:?}. Run tools/nonmem-extract.py on a NONMEM listing and \
         commit the .nm.json; for another tool, commit the numbers only."
    );
    assert!(all.iter().any(|p| p.ends_with("ss_chz_r1_drug.nm.json")), "the walk reaches results/");
}

#[test]
fn a_listing_is_recognised_by_name_and_by_content() {
    assert!(is_listing(Path::new("a/run.lst"), b""));
    assert!(is_listing(Path::new("a/RUN.LST"), b""));
    for log in PSN_LOGS {
        assert!(is_listing(&Path::new("a").join(log), b""));
    }
    assert!(is_listing(Path::new("a/run.out"), format!("x\n{LICENSE_LINE} Y").as_bytes()));
    let lower = concat!("x\n license ", "registered to: Y");
    assert!(is_listing(Path::new("a/run.out"), lower.as_bytes()));
    assert!(!is_listing(Path::new("a/run.nm.json"), b"{}"));
}
