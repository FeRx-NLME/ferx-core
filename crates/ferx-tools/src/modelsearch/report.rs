//! `models.csv` and `final.ferx` — what a structural search leaves behind
//! (#1181).
//!
//! One row per fitted model — input, base, every candidate — with its
//! structure, its criterion and Δ against the base, its rank among the
//! models that passed the gate, and beside them the convergence status, the
//! strictness verdict and the **wall-clock seconds** the fit took. The
//! per-layer `candidates.csv` the runner writes under each layer directory
//! is the fuller record (hashes, dedup, resume); this table is the one to
//! read.

use std::path::{Path, PathBuf};

use ferx_core::edit::ModelEdit;

use super::{ModelsearchResult, TransitCount};

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
pub const MODEL_COLUMNS: [&str; 21] = [
    "id",
    "parent",
    "layer",
    "path",
    "absorption",
    "peripherals",
    "transits",
    "lagtime",
    "n_parameters",
    "ofv",
    "criterion",
    "d_criterion",
    "rank",
    "converged",
    "passed",
    "failures",
    "error",
    "seconds",
    "selected",
    "continued",
    "reused",
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

/// `TRANSITS` as the table prints it: `0`, `3` or `N`.
fn transits(t: Option<TransitCount>) -> String {
    match t {
        None => "0".into(),
        Some(t) => t.to_string(),
    }
}

/// Write `models.csv`, `final.ferx` and `models/<id>.ferx` into `dir`.
///
/// The final model is the selected model's text with its own final
/// estimates written into the initial values — Pharmpy's `update_inits` on
/// the final model — so it reads and refits as the search left it. The
/// per-model files are the candidates as they were fitted, so a user can
/// read the one the table ranked second and refit it by hand.
pub fn write_report(dir: &Path, result: &ModelsearchResult) -> Result<(), String> {
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

fn write_models(path: &Path, result: &ModelsearchResult) -> Result<(), String> {
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
                r.layer.to_string(),
                r.path
                    .iter()
                    .map(|k| k.to_string())
                    .collect::<Vec<_>>()
                    .join(";"),
                r.structure.absorption.label().to_string(),
                r.structure.peripherals.to_string(),
                transits(r.structure.transits),
                if r.structure.lagtime { "ON" } else { "OFF" }.to_string(),
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
                number(r.seconds),
                r.selected.to_string(),
                r.continued.to_string(),
                r.reused.to_string(),
            ])
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("cannot flush `{}`: {e}", path.display()))
}

/// A human-readable rendering of the search: every model, best first, then
/// the notes. What `ferx modelsearch` prints.
pub fn render_summary(result: &ModelsearchResult) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let label = result.criterion.label();
    let base = result.row(&result.base_id);
    let _ = writeln!(
        out,
        "Base model ({}): {} — OFV {}, {label} {}",
        result.base_id,
        structure_label(&result.base_structure),
        base.and_then(|r| r.ofv)
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "-".into()),
        base.map(|r| number_or_dash(r.criterion))
            .unwrap_or_else(|| "-".into())
    );
    let _ = writeln!(
        out,
        "Algorithm: {}, {} model{} fitted{}",
        result.options.algorithm.label(),
        result.rows.len(),
        if result.rows.len() == 1 { "" } else { "s" },
        if result.cancelled {
            " (search cancelled)"
        } else {
            ""
        }
    );
    let _ = writeln!(
        out,
        "\n  {:<8} {:<40} {:>5} {:>12} {:>12} {:>9} {:>4} {:>7}  {:<13} decision",
        "model", "structure", "npar", "OFV", label, "d", "rank", "seconds", "status"
    );
    let mut rows: Vec<&super::ModelRow> = result.rows.iter().collect();
    // Ranked models first, best first; then the rest in generation order.
    rows.sort_by_key(|r| (r.rank.is_none(), r.rank.unwrap_or(0)));
    for r in rows {
        let status = match (r.converged, r.passed, &r.error) {
            (_, _, Some(_)) => "failed".to_string(),
            (_, true, None) => "ok".to_string(),
            (Some(false), false, None) => "not converged".to_string(),
            (_, false, None) => "excluded".to_string(),
        };
        let decision = if r.selected {
            "SELECTED".to_string()
        } else if let Some(e) = &r.error {
            e.message.clone()
        } else if !r.passed {
            r.failures.first().cloned().unwrap_or_default()
        } else if !r.continued {
            "not extended (reduced)".to_string()
        } else {
            String::new()
        };
        let _ = writeln!(
            out,
            "  {:<8} {:<40} {:>5} {:>12} {:>12} {:>9} {:>4} {:>7.1}  {:<13} {}",
            r.id,
            structure_label(&r.structure),
            r.n_parameters
                .map(|n| n.to_string())
                .unwrap_or_else(|| "-".into()),
            r.ofv
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".into()),
            number_or_dash(r.criterion),
            r.d_criterion
                .map(|v| format!("{v:+.3}"))
                .unwrap_or_else(|| "-".into()),
            r.rank.map(|n| n.to_string()).unwrap_or_else(|| "-".into()),
            r.seconds,
            status,
            decision
        );
    }
    let _ = writeln!(
        out,
        "\nFinal model: {} — {} ({label} {})",
        result.final_id,
        result
            .row(&result.final_id)
            .map(|r| structure_label(&r.structure))
            .unwrap_or_default(),
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

/// `FO, 1 peripheral, 3 transits, lag` — one line per structure.
pub fn structure_label(s: &super::Structure) -> String {
    let mut parts = vec![s.absorption.label().to_string()];
    parts.push(format!(
        "{} peripheral{}",
        s.peripherals,
        if s.peripherals == 1 { "" } else { "s" }
    ));
    if let Some(t) = s.transits {
        parts.push(match t {
            TransitCount::N => "N transits".into(),
            TransitCount::Count(n) => format!("{n} transit{}", if n == 1 { "" } else { "s" }),
        });
    }
    if s.lagtime {
        parts.push("lag".into());
    }
    parts.join(", ")
}
