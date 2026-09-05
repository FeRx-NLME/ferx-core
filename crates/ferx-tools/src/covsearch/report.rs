//! `steps.csv` and `final.ferx` — what a search leaves behind (#1180).
//!
//! The step table has one row per candidate of every step, so a reader can
//! see not only which effect won each step but what every other effect did
//! at that step, and *why* a candidate that looks better on OFV alone was not
//! chosen: the `converged`, `passed` and `failures` columns sit beside
//! `dofv` and `p_value` on every row. The per-step `candidates.csv` the
//! runner writes under each step directory is the fuller record (hashes,
//! timings, dedup); this table is the one to read.

use std::path::{Path, PathBuf};

use ferx_core::edit::ModelEdit;

use super::CovsearchResult;

pub fn steps_path(dir: &Path) -> PathBuf {
    dir.join("steps.csv")
}

pub fn final_model_path(dir: &Path) -> PathBuf {
    dir.join("final.ferx")
}

/// The columns of `steps.csv`, in order.
pub const STEP_COLUMNS: [&str; 17] = [
    "step",
    "phase",
    "candidate",
    "parameter",
    "covariate",
    "form",
    "parent_ofv",
    "ofv",
    "dofv",
    "df",
    "p_value",
    "alpha",
    "significant",
    "selected",
    "converged",
    "passed",
    "failures",
];

fn number(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.6}")
    } else {
        String::new()
    }
}

fn opt_number(value: Option<f64>) -> String {
    number(value.unwrap_or(f64::NAN))
}

/// Write `steps.csv` and `final.ferx` into `dir`.
///
/// The final model is the winning candidate's text with its own final
/// estimates written into the initial values — Pharmpy's `update_inits` on
/// the final model — so it reads and refits as the search left it. A final
/// model whose fit is unavailable (a degraded resume) is written as it was
/// fitted, with a note in the result.
pub fn write_report(dir: &Path, result: &CovsearchResult) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("cannot create search directory `{}`: {e}", dir.display()))?;
    write_steps(&steps_path(dir), result)?;
    let mut model = result.final_model.clone();
    if let Some(fit) = &result.final_fit {
        model.apply(ModelEdit::SeedInits(fit))?;
    }
    let path = final_model_path(dir);
    std::fs::write(&path, model.render())
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    Ok(())
}

fn write_steps(path: &Path, result: &CovsearchResult) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    writer
        .write_record(STEP_COLUMNS)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    for r in &result.steps {
        let lrt = r.lrt;
        let failures = if r.failures.is_empty() {
            r.note.clone().unwrap_or_default()
        } else {
            r.failures.join("; ")
        };
        writer
            .write_record([
                r.step.to_string(),
                r.phase.label().to_string(),
                r.candidate.clone(),
                r.effect.parameter.clone(),
                r.effect.covariate.clone(),
                r.effect.form_label().to_string(),
                number(r.parent_ofv),
                opt_number(r.ofv),
                opt_number(lrt.map(|t| t.dofv)),
                lrt.map(|t| t.df.to_string()).unwrap_or_default(),
                opt_number(lrt.map(|t| t.p_value)),
                lrt.map(|t| number(t.alpha)).unwrap_or_default(),
                lrt.map(|t| t.significant.to_string()).unwrap_or_default(),
                r.selected.to_string(),
                r.converged.map(|c| c.to_string()).unwrap_or_default(),
                r.passed.to_string(),
                failures,
            ])
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("cannot flush `{}`: {e}", path.display()))
}

/// A human-readable rendering of the search: the step table and the final
/// relation set. What `ferx covsearch` prints.
pub fn render_summary(result: &CovsearchResult) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "Base model: OFV {:.3}", result.base_ofv);
    let n_steps = result.n_steps();
    for step in 1..=n_steps {
        let rows: Vec<_> = result.step_rows(step).collect();
        let Some(first) = rows.first() else { continue };
        let _ = writeln!(
            out,
            "\nStep {step} ({}), parent OFV {:.3}, p = {}",
            first.phase.label(),
            first.parent_ofv,
            first
                .lrt
                .map(|t| format!("{}", t.alpha))
                .unwrap_or_else(|| "-".into())
        );
        let _ = writeln!(
            out,
            "  {:<28} {:>12} {:>10} {:>3} {:>10}  {:<11} decision",
            "effect", "OFV", "dOFV", "df", "p", "status"
        );
        for r in rows {
            let status = match (r.converged, r.passed) {
                (_, true) => "ok".to_string(),
                (Some(false), false) => "not converged".to_string(),
                (_, false) => "excluded".to_string(),
            };
            let decision = if r.selected {
                "SELECTED".to_string()
            } else if !r.passed {
                r.failures
                    .first()
                    .cloned()
                    .or_else(|| r.note.clone())
                    .unwrap_or_default()
            } else if let Some(t) = r.lrt {
                match (r.phase.is_forward(), t.significant) {
                    (true, true) => "significant".into(),
                    (true, false) => "not significant".into(),
                    (false, true) => "kept (removal significant)".into(),
                    (false, false) => "removable".into(),
                }
            } else {
                r.note.clone().unwrap_or_default()
            };
            let _ = writeln!(
                out,
                "  {:<28} {:>12} {:>10} {:>3} {:>10}  {:<11} {}",
                r.effect.label(),
                r.ofv
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "-".into()),
                r.lrt
                    .map(|t| format!("{:.3}", t.dofv))
                    .unwrap_or_else(|| "-".into()),
                r.lrt
                    .map(|t| t.df.to_string())
                    .unwrap_or_else(|| "-".into()),
                r.lrt
                    .map(|t| format!("{:.4}", t.p_value))
                    .unwrap_or_else(|| "-".into()),
                status,
                decision
            );
        }
    }
    let _ = writeln!(
        out,
        "\nFinal model{}: OFV {:.3} (step {}), {} relation{}",
        if result.cancelled {
            " (search cancelled)"
        } else {
            ""
        },
        result.final_ofv,
        result.final_step,
        result.included.len(),
        if result.included.len() == 1 { "" } else { "s" }
    );
    for inc in &result.included {
        let _ = writeln!(
            out,
            "  {} ~ {} {}  [{}]",
            inc.effect.parameter,
            inc.effect.covariate,
            inc.effect.form_label(),
            inc.origin.label()
        );
    }
    if !result.notes.is_empty() {
        let _ = writeln!(out, "\nNotes:");
        for n in &result.notes {
            let _ = writeln!(out, "  - {n}");
        }
    }
    out
}
