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

/// The columns of `candidates.csv`, in order.
pub const COLUMNS: [&str; 14] = [
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

/// Write `candidates.csv` into `dir`.
pub fn write_table(dir: &Path, results: &[CandidateResult]) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("cannot create search directory `{}`: {e}", dir.display()))?;
    let path = table_path(dir);
    let mut writer = csv::Writer::from_path(&path)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    writer
        .write_record(COLUMNS)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    for r in results {
        let ofv = r.fit.as_ref().map(|f| f.ofv).unwrap_or(f64::NAN);
        let converged = r.fit.as_ref().map(|f| f.converged);
        writer
            .write_record([
                r.id.clone(),
                r.parent.clone().unwrap_or_default(),
                r.hash.clone(),
                r.features.render(),
                number(r.criterion),
                number(ofv),
                converged.map(|c| c.to_string()).unwrap_or_default(),
                r.verdict.passed.to_string(),
                reasons(&r.verdict.failures),
                reasons(&r.verdict.skipped),
                number(r.seconds),
                r.error.clone().unwrap_or_default(),
                r.duplicate_of.clone().unwrap_or_default(),
                r.reused.to_string(),
            ])
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("cannot flush `{}`: {e}", path.display()))
}

#[cfg(test)]
#[path = "output_tests.rs"]
mod tests;
