//! What an AMD run leaves behind: `steps.csv`, `candidates.csv`,
//! `final.ferx`, and the summary the CLI prints (#1184).
//!
//! The report *is* the product of this tool. A pipeline that prints only its
//! final model is unauditable — the reader cannot tell a step that found a
//! real improvement from one whose winner beat its siblings because they
//! stalled, and cannot tell a step that was skipped from one that ran and
//! found nothing. So two tables are written, at two altitudes:
//!
//! * **`steps.csv`** — one row per planned step, skipped ones included with
//!   their reason, carrying the criterion the step ranked on, the value before
//!   and after, the ΔOFV, what it selected, and its wall-clock cost.
//! * **`candidates.csv`** — one row per *candidate* of every step: what it
//!   was, what it scored, its Δ against its parent on both the criterion and
//!   the OFV, the strictness verdict **with its reasons**, whether the fit
//!   converged, and the seconds it took.
//!
//! Each tool's own directory keeps its own fuller record (`models.csv`,
//! `steps.csv`, the per-step `candidates.csv` the runner writes, and every
//! candidate's model text). Nothing here replaces those; this is the view
//! across them.

use std::path::{Path, PathBuf};

use super::{AmdResult, CandidateRow, StepOutcome};
use crate::search::{number, opt_number};

pub fn steps_path(dir: &Path) -> PathBuf {
    dir.join("steps.csv")
}

pub fn candidates_path(dir: &Path) -> PathBuf {
    dir.join("candidates.csv")
}

pub fn final_model_path(dir: &Path) -> PathBuf {
    dir.join("final.ferx")
}

/// The columns of `steps.csv`, in order.
pub const STEP_COLUMNS: [&str; 16] = [
    "step",
    "tool",
    "rerun",
    "directory",
    "status",
    "reason",
    "criterion",
    "value_before",
    "value_after",
    "d_value",
    "ofv_before",
    "ofv_after",
    "d_ofv",
    "candidates",
    "selected",
    "seconds",
];

/// The columns of `candidates.csv`, in order.
pub const CANDIDATE_COLUMNS: [&str; 18] = [
    "step",
    "tool",
    "id",
    "parent",
    "description",
    "criterion",
    "value",
    "d_value",
    "ofv",
    "d_ofv",
    "rank",
    "converged",
    "passed",
    "failures",
    "error",
    "note",
    "seconds",
    "selected",
];

/// Write `steps.csv`, `candidates.csv` and `final.ferx` into `dir`.
///
/// The final model is written as the pipeline left it — already seeded from
/// its own estimates, so it reads and refits at the fit beside it.
pub fn write_report(dir: &Path, result: &AmdResult) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("cannot create run directory `{}`: {e}", dir.display()))?;
    write_steps(&steps_path(dir), &result.steps)?;
    write_candidates(&candidates_path(dir), &result.rows)?;
    let path = final_model_path(dir);
    std::fs::write(&path, result.final_model.render())
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    Ok(())
}

fn d(after: Option<f64>, before: Option<f64>) -> Option<f64> {
    match (after, before) {
        (Some(a), Some(b)) => Some(a - b),
        _ => None,
    }
}

fn write_steps(path: &Path, steps: &[StepOutcome]) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    writer
        .write_record(STEP_COLUMNS)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    for s in steps {
        writer
            .write_record([
                s.step.label().to_string(),
                s.step.tool().to_string(),
                s.rerun.to_string(),
                s.dir.clone(),
                s.status().to_string(),
                s.reason().unwrap_or_default().to_string(),
                s.criterion.to_string(),
                opt_number(s.value_before),
                opt_number(s.value_after),
                opt_number(d(s.value_after, s.value_before)),
                opt_number(s.ofv_before),
                opt_number(s.ofv_after),
                opt_number(d(s.ofv_after, s.ofv_before)),
                s.candidates.to_string(),
                s.selected.join("; "),
                number(s.seconds),
            ])
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("cannot flush `{}`: {e}", path.display()))
}

fn write_candidates(path: &Path, rows: &[CandidateRow]) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    writer
        .write_record(CANDIDATE_COLUMNS)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    for r in rows {
        writer
            .write_record([
                r.step.to_string(),
                r.tool.clone(),
                r.id.clone(),
                r.parent.clone().unwrap_or_default(),
                r.description.clone(),
                r.criterion.to_string(),
                opt_number(r.value),
                opt_number(r.d_value),
                opt_number(r.ofv),
                opt_number(r.d_ofv),
                r.rank.map(|n| n.to_string()).unwrap_or_default(),
                r.converged.map(|c| c.to_string()).unwrap_or_default(),
                r.passed.to_string(),
                r.failures.join("; "),
                r.error.clone().unwrap_or_default(),
                r.note.clone().unwrap_or_default(),
                number(r.seconds),
                r.selected.to_string(),
            ])
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("cannot flush `{}`: {e}", path.display()))
}

/// The human-readable report: the pipeline, then every step's candidates, then
/// the final model's estimates with their standard errors.
///
/// What `ferx amd` prints, and the one place a reader can see the whole run
/// without opening six directories.
pub fn render_summary(result: &AmdResult) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "AMD pipeline: {} strategy, retries {}",
        result.options.strategy.label(),
        result.options.retries.label()
    );
    let _ = writeln!(
        out,
        "Start model: OFV {}",
        result
            .input_fit
            .as_ref()
            .map(|f| format!("{:.3}", f.ofv))
            .unwrap_or_else(|| "-".into())
    );

    let _ = writeln!(
        out,
        "\n  {:<3} {:<12} {:<12} {:>12} {:>12} {:>10} {:>5} {:>8}  selected",
        "#", "step", "criterion", "before", "after", "d", "cand", "seconds"
    );
    for s in &result.steps {
        if let Some(reason) = s.reason() {
            let _ = writeln!(
                out,
                "  {:<3} {:<12} {:<12} {:>12} {:>12} {:>10} {:>5} {:>8}  {}: {}",
                s.index,
                s.step.label(),
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                s.status(),
                reason
            );
            continue;
        }
        let _ = writeln!(
            out,
            "  {:<3} {:<12} {:<12} {:>12} {:>12} {:>10} {:>5} {:>8.1}  {}",
            s.index,
            format!("{}{}", s.step.label(), if s.rerun { "*" } else { "" }),
            s.criterion,
            dash(s.value_before),
            dash(s.value_after),
            signed(d(s.value_after, s.value_before)),
            s.candidates,
            s.seconds,
            s.selected.join("; ")
        );
    }

    for s in result.steps.iter().filter(|s| s.ok()) {
        let rows: Vec<&CandidateRow> = result
            .rows
            .iter()
            // The step's own candidates, and the retries pass on its winner:
            // the pass is part of what the step delivered, and reading it in
            // `candidates.csv` alone would hide the one row that says whether
            // the selected model's optimum was confirmed.
            .filter(|r| r.step == s.index && (r.tool == s.step.tool() || r.tool == "retries"))
            .collect();
        if rows.is_empty() {
            continue;
        }
        let _ = writeln!(
            out,
            "\nStep {} — {} ({}), ranked on {}",
            s.index,
            s.step.label(),
            s.dir,
            s.criterion
        );
        let _ = writeln!(
            out,
            "  {:<14} {:<34} {:>12} {:>10} {:>10} {:>4} {:>7}  status",
            "candidate", "model", "criterion", "d", "dOFV", "rank", "seconds"
        );
        for r in rows {
            let _ = writeln!(
                out,
                "  {:<14} {:<34} {:>12} {:>10} {:>10} {:>4} {:>7.1}  {}",
                truncate(&r.id, 14),
                truncate(&r.description, 34),
                dash(r.value),
                signed(r.d_value),
                signed(r.d_ofv),
                r.rank.map(|n| n.to_string()).unwrap_or_else(|| "-".into()),
                r.seconds,
                status(r)
            );
        }
    }

    let notes: Vec<&String> = result
        .notes
        .iter()
        .chain(result.steps.iter().flat_map(|s| s.notes.iter()))
        .collect();
    if !notes.is_empty() {
        let _ = writeln!(out, "\nNotes:");
        for n in notes {
            let _ = writeln!(out, "  - {n}");
        }
    }

    let _ = writeln!(
        out,
        "\nFinal model: OFV {}{}",
        result
            .final_fit
            .as_ref()
            .map(|f| format!("{:.3}", f.ofv))
            .unwrap_or_else(|| "-".into()),
        result
            .d_ofv()
            .map(|d| format!(" ({d:+.3} against the start model)"))
            .unwrap_or_default()
    );
    if result.cancelled {
        let _ = writeln!(out, "The run was cancelled; the steps after it never ran.");
    }
    if let Some(fit) = &result.final_fit {
        let _ = writeln!(out, "\n{}", ferx_core::io::output::parameter_table(fit));
    }
    out
}

/// The one-word decision on a candidate, and why when it is not `ok`.
fn status(r: &CandidateRow) -> String {
    if let Some(e) = &r.error {
        return format!("failed: {e}");
    }
    if r.selected {
        return match &r.note {
            Some(note) => format!("SELECTED ({note})"),
            None => "SELECTED".into(),
        };
    }
    if !r.passed {
        return format!(
            "excluded: {}",
            if r.failures.is_empty() {
                "did not pass the gate".to_string()
            } else {
                r.failures.join("; ")
            }
        );
    }
    r.note.clone().unwrap_or_default()
}

fn dash(v: Option<f64>) -> String {
    v.filter(|v| v.is_finite())
        .map(|v| format!("{v:.3}"))
        .unwrap_or_else(|| "-".into())
}

fn signed(v: Option<f64>) -> String {
    v.filter(|v| v.is_finite())
        .map(|v| format!("{v:+.3}"))
        .unwrap_or_else(|| "-".into())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    s.chars().take(n.saturating_sub(1)).collect::<String>() + "…"
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
