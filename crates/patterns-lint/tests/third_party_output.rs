//! No verbatim third-party tool output in the tree (#3182): a NONMEM listing carries the
//! license holder's details and NONMEM's own text, a PsN log carries PsN's. Commit the
//! extracted numbers instead (`tools/nonmem-extract.py` for NONMEM).

use patterns_lint::repo_root;
use std::path::{Path, PathBuf};

/// Printed by NONMEM in every listing; spelled in two parts so this file does not match.
const LICENSE_LINE: &str = concat!("License ", "Registered to:");

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read dir") {
        let entry = entry.expect("dir entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = entry.path();
        if entry.file_type().expect("file type").is_dir() {
            if !name.starts_with('.') && name != "target" {
                files(&path, out);
            }
        } else {
            out.push(path);
        }
    }
}

fn is_listing(path: &Path, bytes: &[u8]) -> bool {
    path.extension().is_some_and(|e| e == "lst")
        || path.file_name().is_some_and(|n| n == "scmlog.txt")
        || bytes.windows(LICENSE_LINE.len()).any(|w| w == LICENSE_LINE.as_bytes())
}

#[test]
fn no_verbatim_tool_output_is_committed() {
    let root = repo_root();
    let mut all = Vec::new();
    for dir in ["crates", "data", "docs", "examples", "nonmem_anchor", "src", "tests", "tools"] {
        files(&root.join(dir), &mut all);
    }
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
    assert!(is_listing(Path::new("a/scmlog.txt"), b""));
    assert!(is_listing(Path::new("a/run.out"), format!("x\n{LICENSE_LINE} Y").as_bytes()));
    assert!(!is_listing(Path::new("a/run.nm.json"), b"{}"));
}
