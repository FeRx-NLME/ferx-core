//! `patterns-lint` — source-pattern lints over the **test code** of the
//! workspace (#2634).
//!
//! The first (and so far only) rule is [`nan_fold`]: a `max`/`min` fold in a
//! test, which discards `NaN` and lets an assertion pass on the finite rows. The
//! rule text lives in `tools/lint-rules/`, together with the shrink-only
//! [`baseline`] of the sites that predate it.
//!
//! The gate is `tools/preflight.sh patterns` (CI job `Patterns lint`), whose
//! `cargo test -p patterns-lint` runs the corpus check in `tests/patterns_corpus.rs`.

use std::path::{Path, PathBuf};

pub mod baseline;
pub mod nan_fold;
pub mod scope;

/// One finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub rule: &'static str,
    /// Repo-relative, `/`-separated.
    pub file: String,
    pub line: usize,
    /// `file::mod::fn` — the baseline's key.
    pub key: String,
    pub message: String,
    /// The source line, trimmed.
    pub snippet: String,
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}: {}: {}\n    {}",
            self.file, self.line, self.rule, self.message, self.snippet
        )
    }
}

/// The repo root: two levels above this crate's manifest.
pub fn repo_root() -> PathBuf {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .unwrap_or(here)
}

pub fn default_baseline_path(root: &Path) -> PathBuf {
    root.join("tools").join("lint-rules").join("baseline.txt")
}

/// Lint one source text as if it lived at `rel`.
pub fn lint_source(rel: &str, text: &str, whole_test: bool) -> Result<Vec<Finding>, String> {
    let ast = syn::parse_file(text).map_err(|e| format!("{rel}: cannot parse: {e}"))?;
    Ok(nan_fold::lint_file(rel, text, &ast, whole_test))
}

/// The result of scanning a tree.
pub struct Scan {
    /// Every scanned file, with its resolved whole-file test status.
    pub files: Vec<scope::SourceFile>,
    /// Every finding, sorted by file and line, before the baseline.
    pub findings: Vec<Finding>,
}

/// Scan the tree under `root`: discover, parse, resolve test scope, lint.
///
/// Parallel, because the tree is ~800k lines and a debug-built `syn` parses it
/// in ~15 s on one core — and `cargo test` is a debug build. The ASTs are not
/// `Send` (`proc_macro2` spans are thread-bound), so each worker keeps the files
/// it parsed, sends only their `mod` edges to this thread, waits for the
/// resolved test flags, and lints its own files. One parse per file.
///
/// A file `syn` cannot parse is an error, not a skip: a lint that silently drops
/// the file it cannot read is a gate with a hole exactly where nobody looks.
pub fn scan(root: &Path) -> Result<Scan, String> {
    use std::collections::BTreeMap;
    use std::sync::{mpsc, Arc};

    let paths = scope::source_paths(root)?;
    let index: BTreeMap<PathBuf, usize> = paths
        .iter()
        .enumerate()
        .map(|(i, (_, abs))| (abs.clone(), i))
        .collect();
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, paths.len().max(1));

    type Edges = Result<Vec<(PathBuf, bool)>, String>;
    let (edge_tx, edge_rx) = mpsc::channel::<(usize, Edges)>();
    let results = std::thread::scope(|s| {
        let mut flag_txs = Vec::new();
        let mut handles = Vec::new();
        for w in 0..workers {
            let (flag_tx, flag_rx) = mpsc::channel::<Option<Arc<Vec<bool>>>>();
            flag_txs.push(flag_tx);
            let edge_tx = edge_tx.clone();
            let paths = &paths;
            handles.push(s.spawn(move || -> Result<Vec<Finding>, String> {
                let mut mine = Vec::new();
                for (i, (rel, abs)) in paths.iter().enumerate().skip(w).step_by(workers) {
                    let parsed = std::fs::read_to_string(abs)
                        .map_err(|e| format!("{rel}: {e}"))
                        .and_then(|text| match syn::parse_file(&text) {
                            Ok(ast) => Ok((text, ast)),
                            Err(e) => {
                                let at = e.span().start();
                                Err(format!(
                                    "{rel}:{}:{}: cannot parse: {e}",
                                    at.line,
                                    at.column + 1
                                ))
                            }
                        });
                    match parsed {
                        Ok((text, ast)) => {
                            let _ = edge_tx.send((i, Ok(scope::child_edges(abs, rel, &ast))));
                            mine.push((i, text, ast));
                        }
                        Err(e) => {
                            let _ = edge_tx.send((i, Err(e)));
                        }
                    }
                }
                drop(edge_tx);
                let Ok(Some(flags)) = flag_rx.recv() else {
                    return Ok(Vec::new());
                };
                let mut out = Vec::new();
                for (i, text, ast) in &mine {
                    out.extend(nan_fold::lint_file(&paths[*i].0, text, ast, flags[*i]));
                }
                Ok(out)
            }));
        }
        drop(edge_tx);

        let mut edges: Vec<Vec<(usize, bool)>> = vec![Vec::new(); paths.len()];
        let mut errors = Vec::new();
        for (i, e) in edge_rx {
            match e {
                Ok(children) => {
                    edges[i] = children
                        .into_iter()
                        .filter_map(|(p, t)| index.get(&p).map(|&j| (j, t)))
                        .collect();
                }
                Err(msg) => errors.push(msg),
            }
        }
        let flags = (errors.is_empty()).then(|| {
            let initial = paths
                .iter()
                .map(|(rel, _)| scope::path_is_test_by_name(rel))
                .collect();
            Arc::new(scope::propagate(initial, &edges))
        });
        for tx in &flag_txs {
            let _ = tx.send(flags.clone());
        }
        let mut findings = Vec::new();
        for h in handles {
            match h.join() {
                Ok(Ok(f)) => findings.extend(f),
                Ok(Err(e)) => errors.push(e),
                Err(_) => errors.push("a lint worker panicked".to_string()),
            }
        }
        if errors.is_empty() {
            Ok((flags.unwrap_or_default(), findings))
        } else {
            errors.sort();
            Err(errors.join("\n"))
        }
    });
    let (flags, mut findings) = results?;
    findings.sort_by(|a, b| (&a.file, a.line).cmp(&(&b.file, b.line)));
    let files = paths
        .into_iter()
        .zip(flags.iter())
        .map(|((rel, abs), &whole_test)| scope::SourceFile {
            rel,
            abs,
            whole_test,
        })
        .collect();
    Ok(Scan { files, findings })
}

/// Every finding in the tree under `root`, sorted by file and line.
pub fn lint_tree(root: &Path) -> Result<Vec<Finding>, String> {
    scan(root).map(|s| s.findings)
}

/// The failure report for the corpus check and the CLI: every failing finding,
/// every stale entry, and what to do about each.
pub fn render_report(applied: &baseline::Applied) -> String {
    let mut report = String::new();
    for f in &applied.failing {
        report.push_str(&format!("{f}\n"));
    }
    for s in &applied.stale {
        report.push_str(&format!(
            "{}: {}: baseline records {} finding(s) but there are now {} — paid down, \
             so lower the entry in tools/lint-rules/baseline.txt \
             (`cargo run -p patterns-lint -- --update-baseline`)\n",
            s.key, s.rule, s.recorded, s.now
        ));
    }
    report
}

#[cfg(test)]
mod tests;
