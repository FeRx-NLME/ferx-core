//! `models.csv`, `models/<id>.ferx` and `final.ferx` — what an
//! inter-occasion variability search leaves behind (#1183).
//!
//! One row per fitted model — the input, the full-IOV model, every
//! candidate of both steps — with its structure in Pharmpy's spelling
//! (`IIV([CL]+[V]+[KA]);IOV([CL])`), its criterion and Δ against its step's
//! parent, its rank within the step, and beside them the convergence
//! status, the strictness verdict, the starts and the seconds.

use std::path::{Path, PathBuf};

use ferx_core::edit::ModelEdit;

use super::IovsearchResult;
use crate::search::{number, opt_number};

pub fn models_path(dir: &Path) -> PathBuf {
    dir.join("models.csv")
}

pub fn final_model_path(dir: &Path) -> PathBuf {
    dir.join("final.ferx")
}

/// Where every fitted model's text goes: `<dir>/models/<id>.ferx`.
pub fn models_dir(dir: &Path) -> PathBuf {
    dir.join("models")
}

/// The columns of `models.csv`, in order.
pub const MODEL_COLUMNS: [&str; 19] = [
    "id",
    "parent",
    "step",
    "description",
    "etas",
    "kappas",
    "kappa_blocks",
    "n_parameters",
    "ofv",
    "criterion",
    "d_criterion",
    "rank",
    "converged",
    "passed",
    "failures",
    "error",
    "starts",
    "seconds",
    "selected",
];

/// Write `models.csv`, `models/<id>.ferx` and `final.ferx` into `dir`.
///
/// The final model is the selected model's text with its own final
/// estimates written into the initial values — Pharmpy's `update_inits` on
/// the final model — so it reads and refits as the search left it.
pub fn write_report(dir: &Path, result: &IovsearchResult) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("cannot create search directory `{}`: {e}", dir.display()))?;
    write_models(&models_path(dir), result)?;
    let models = models_dir(dir);
    std::fs::create_dir_all(&models)
        .map_err(|e| format!("cannot create `{}`: {e}", models.display()))?;
    for (id, text) in &result.models {
        let path = models.join(format!("{id}.ferx"));
        std::fs::write(&path, text.render())
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    let mut model = result.final_model.clone();
    if let Some(fit) = &result.final_fit {
        model.apply(ModelEdit::SeedInits(fit))?;
    }
    let path = final_model_path(dir);
    std::fs::write(&path, model.render())
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    Ok(())
}

fn write_models(path: &Path, result: &IovsearchResult) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    writer
        .write_record(MODEL_COLUMNS)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    for r in &result.rows {
        writer
            .write_record([
                r.id.clone(),
                r.parent.clone().unwrap_or_default(),
                r.step.to_string(),
                r.structure.description(),
                r.structure.etas.join(";"),
                r.structure.kappas.join(";"),
                r.structure
                    .kappa_blocks
                    .iter()
                    .map(|b| b.join(","))
                    .collect::<Vec<_>>()
                    .join(";"),
                r.n_parameters.map(|n| n.to_string()).unwrap_or_default(),
                opt_number(r.ofv),
                number(r.criterion),
                opt_number(r.d_criterion),
                r.rank.map(|n| n.to_string()).unwrap_or_default(),
                r.converged.map(|c| c.to_string()).unwrap_or_default(),
                r.passed.to_string(),
                r.failures.join("; "),
                r.error
                    .as_ref()
                    .map(|e| e.message.clone())
                    .unwrap_or_default(),
                r.starts.to_string(),
                number(r.seconds),
                r.selected.to_string(),
            ])
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("cannot flush `{}`: {e}", path.display()))
}

/// A human-readable rendering of the search: the input, each step's
/// ranking, the final model. What `ferx iovsearch` prints.
pub fn render_summary(result: &IovsearchResult) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let label = result.criterion.label();
    let describe_id = |id: &str| -> String {
        result
            .row(id)
            .map(|r| r.structure.description())
            .unwrap_or_default()
    };
    if let Some(input) = result.row("input") {
        let _ = writeln!(
            out,
            "Input model: {} — OFV {}, {label} {}",
            input.structure.description(),
            input
                .ofv
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".into()),
            number_or_dash(input.criterion)
        );
    }
    let _ = writeln!(
        out,
        "Distribution: {}, {} model{} fitted{}",
        result.options.distribution.label(),
        result.rows.len(),
        if result.rows.len() == 1 { "" } else { "s" },
        if result.cancelled {
            " (search cancelled)"
        } else {
            ""
        }
    );
    for step in &result.steps {
        let _ = writeln!(
            out,
            "\nStep {} ({}), parent {} ({})",
            step.step,
            if step.step == 1 { "IOV" } else { "IIV" },
            step.parent,
            describe_id(&step.parent)
        );
        let _ = writeln!(
            out,
            "  {:<8} {:<36} {:>5} {:>12} {:>12} {:>9} {:>4}  {:<13} decision",
            "model", "structure", "npar", "OFV", label, "d", "rank", "status"
        );
        for r in &step.ranked {
            let row = result.row(&r.id);
            let status = match row {
                None => "-".to_string(),
                Some(row) => match (row.converged, row.passed, &row.error) {
                    (_, _, Some(_)) => "failed".to_string(),
                    (_, true, None) => "ok".to_string(),
                    (Some(false), false, None) => "not converged".to_string(),
                    (_, false, None) => "excluded".to_string(),
                },
            };
            let decision = if r.id == step.best {
                if r.id == step.parent {
                    "kept".to_string()
                } else {
                    "BEST".to_string()
                }
            } else if let Some(row) = row {
                if let Some(e) = &row.error {
                    e.message.clone()
                } else if !row.passed {
                    row.failures.first().cloned().unwrap_or_default()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            let _ = writeln!(
                out,
                "  {:<8} {:<36} {:>5} {:>12} {:>12} {:>9} {:>4}  {:<13} {}",
                r.id,
                describe_id(&r.id),
                row.and_then(|r| r.n_parameters)
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "-".into()),
                row.and_then(|r| r.ofv)
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "-".into()),
                number_or_dash(r.criterion),
                r.d_criterion
                    .map(|v| format!("{v:+.3}"))
                    .unwrap_or_else(|| "-".into()),
                r.rank.map(|n| n.to_string()).unwrap_or_else(|| "-".into()),
                status,
                decision
            );
        }
    }
    let _ = writeln!(
        out,
        "\nFinal model: {} — {} ({label} {})",
        result.final_id,
        result.final_structure.description(),
        number_or_dash(result.final_criterion)
    );
    if !result.notes.is_empty() {
        let _ = writeln!(out, "\nNotes:");
        for n in &result.notes {
            let _ = writeln!(out, "  - {n}");
        }
    }
    out
}

fn number_or_dash(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.3}")
    } else {
        "-".into()
    }
}
