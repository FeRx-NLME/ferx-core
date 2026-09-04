//! `candidates.csv` — the per-candidate table (#1178).
//!
//! One row per candidate the run was given, in the order it was given them,
//! including the ones that failed. A candidate excluded by the strictness gate
//! carries *why* in its `failures` cell rather than being absent: a candidate
//! missing from a search report cannot be told apart from one that was never
//! generated, which is exactly the question a user reading the report is asking.
//!
//! The table is rewritten from the finished results, so a completed run's file
//! is in candidate order however its fits were scheduled — the journal beside
//! it is in completion order and is the recovery log, not the report.

use std::path::{Path, PathBuf};

use super::CandidateResult;

pub fn table_path(dir: &Path) -> PathBuf {
    dir.join("candidates.csv")
}

/// Where a **cancelled** run's partial table goes.
///
/// A cancelled run has a report worth reading and a `candidates.csv` it must
/// not be the one to destroy: the run it resumed may have finished hundreds of
/// candidates, and `csv::Writer::from_path` truncates. So the partial table is
/// a file of its own, and a run that reaches the end removes it — a stale
/// partial next to a complete table would be the more confusing of the two
/// failure modes.
pub fn partial_table_path(dir: &Path) -> PathBuf {
    dir.join("candidates.partial.csv")
}

/// The columns of `candidates.csv`, in order.
pub const COLUMNS: [&str; 15] = [
    "id",
    "parent",
    "hash",
    "features",
    "criterion",
    "ofv",
    "converged",
    "passed",
    "failures",
    "skipped",
    "seconds",
    "error",
    "retryable",
    "duplicate_of",
    "reused",
];

/// A non-finite number has no cell — an empty string beats `NaN` in a file
/// meant to be opened in a spreadsheet, and it is what "there is no criterion"
/// actually means.
fn number(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.6}")
    } else {
        String::new()
    }
}

/// Multiple reasons in one cell, `;`-joined. The CSV writer quotes the cell, so
/// a reason containing a comma survives.
fn reasons(list: &[String]) -> String {
    list.join("; ")
}

/// Write `candidates.csv` into `dir`, and drop any partial table left by an
/// earlier cancelled run.
pub fn write_table(dir: &Path, results: &[CandidateResult]) -> Result<(), String> {
    write_table_to(&table_path(dir), dir, results)?;
    // Best-effort: a partial table that outlives its complete successor is
    // confusing, but failing the run over it would be worse.
    let _ = std::fs::remove_file(partial_table_path(dir));
    Ok(())
}

/// Write a cancelled run's rows to [`partial_table_path`], leaving any complete
/// `candidates.csv` beside it untouched.
pub fn write_partial_table(dir: &Path, results: &[CandidateResult]) -> Result<(), String> {
    write_table_to(&partial_table_path(dir), dir, results)
}

/// The table itself, written through a `.part` sibling renamed into place so a
/// reader never sees a half-written file and a failure mid-write does not
/// destroy the previous one.
fn write_table_to(path: &Path, dir: &Path, results: &[CandidateResult]) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("cannot create search directory `{}`: {e}", dir.display()))?;
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    let temp = path.with_file_name(name);
    {
        let mut writer = csv::Writer::from_path(&temp)
            .map_err(|e| format!("cannot write `{}`: {e}", temp.display()))?;
        writer
            .write_record(COLUMNS)
            .map_err(|e| format!("cannot write `{}`: {e}", temp.display()))?;
        for r in results {
            writer
                .write_record([
                    r.id.clone(),
                    r.parent.clone().unwrap_or_default(),
                    r.hash.clone(),
                    r.features.render(),
                    number(r.criterion),
                    number(r.ofv.unwrap_or(f64::NAN)),
                    r.converged.map(|c| c.to_string()).unwrap_or_default(),
                    r.verdict.passed.to_string(),
                    reasons(&r.verdict.failures),
                    reasons(&r.verdict.skipped),
                    number(r.seconds),
                    r.error
                        .as_ref()
                        .map(|e| e.message.clone())
                        .unwrap_or_default(),
                    // Empty rather than `false` when there is no failure to
                    // describe: the column answers "will a resume try this
                    // again?", and a row that succeeded is not asking.
                    r.error
                        .as_ref()
                        .map(|e| e.retryable.to_string())
                        .unwrap_or_default(),
                    r.duplicate_of.clone().unwrap_or_default(),
                    r.reused.to_string(),
                ])
                .map_err(|e| format!("cannot write `{}`: {e}", temp.display()))?;
        }
        writer
            .flush()
            .map_err(|e| format!("cannot flush `{}`: {e}", temp.display()))?;
    }
    std::fs::rename(&temp, path).map_err(|e| {
        let _ = std::fs::remove_file(&temp);
        format!("cannot rename `{}`: {e}", temp.display())
    })
}

#[cfg(test)]
#[path = "output_tests.rs"]
mod tests;
