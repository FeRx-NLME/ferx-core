//! `models.csv` and `final.ferx` — what a global search leaves behind
//! (#1185).
//!
//! One row per model — the input and every grid point evaluated — with its
//! genome as labels, its structure, the covariate relations it carries, its
//! criterion **and** its fitness (the criterion plus the search's own
//! charges), its rank among the models that passed the gate, and beside
//! them the convergence status, the strictness verdict and the wall-clock
//! seconds the fit took. The per-batch `candidates.csv` the runner writes
//! under each step directory is the fuller record (hashes, dedup, resume);
//! this table is the one to read. `generations.csv` is the GA's trajectory.

use std::path::{Path, PathBuf};

use ferx_core::edit::ModelEdit;

use super::{GlobalsearchResult, ModelRow};
use crate::modelsearch::{structure_label, TransitCount};
use crate::search::{number, opt_number};

pub fn models_path(dir: &Path) -> PathBuf {
    dir.join("models.csv")
}

pub fn generations_path(dir: &Path) -> PathBuf {
    dir.join("generations.csv")
}

pub fn final_model_path(dir: &Path) -> PathBuf {
    dir.join("final.ferx")
}

/// Where every fitted model's text goes: `<dir>/models/<id>.ferx`.
pub fn models_dir(dir: &Path) -> PathBuf {
    dir.join("models")
}

/// The columns of `models.csv`, in order.
pub const MODEL_COLUMNS: [&str; 24] = [
    "id",
    "parent",
    "step",
    "genome",
    "absorption",
    "elimination",
    "peripherals",
    "transits",
    "lagtime",
    "covariates",
    "n_parameters",
    "ofv",
    "criterion",
    "fitness",
    "rank",
    "converged",
    "passed",
    "failures",
    "error",
    "seconds",
    "selected",
    "non_influential",
    "duplicate_of",
    "reused",
];

/// The columns of `generations.csv`.
pub const GENERATION_COLUMNS: [&str; 5] = [
    "generation",
    "best_genome",
    "best_fitness",
    "mean_fitness",
    "polished",
];

fn transits(t: Option<TransitCount>) -> String {
    match t {
        None => "0".into(),
        Some(t) => t.to_string(),
    }
}

/// `CL-WT=power;V-WT=linear`.
pub fn covariates_label(row: &ModelRow) -> String {
    row.effects
        .iter()
        .map(|e| format!("{}={}", e.pair_key(), e.form_label()))
        .collect::<Vec<_>>()
        .join(";")
}

/// Write `models.csv`, `generations.csv`, `final.ferx` and
/// `models/<id>.ferx` into `dir`.
///
/// The final model is the selected model's text with its own final
/// estimates written into the initial values, so it reads and refits as
/// the search left it. The per-model files are the candidates as they were
/// fitted.
pub fn write_report(dir: &Path, result: &GlobalsearchResult) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("cannot create search directory `{}`: {e}", dir.display()))?;
    write_models(&models_path(dir), result)?;
    write_generations(&generations_path(dir), result)?;
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

fn write_models(path: &Path, result: &GlobalsearchResult) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    writer
        .write_record(MODEL_COLUMNS)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    for r in &result.rows {
        let s = r.structure;
        writer
            .write_record([
                r.id.clone(),
                r.parent.clone().unwrap_or_default(),
                r.step.clone(),
                r.genome
                    .as_ref()
                    .map(|_| r.description.clone())
                    .unwrap_or_default(),
                s.map(|s| s.absorption.label().to_string())
                    .unwrap_or_default(),
                s.map(|s| s.elimination.label().to_string())
                    .unwrap_or_default(),
                s.map(|s| s.peripherals.to_string()).unwrap_or_default(),
                s.map(|s| transits(s.transits)).unwrap_or_default(),
                s.map(|s| if s.lagtime { "ON" } else { "OFF" }.to_string())
                    .unwrap_or_default(),
                covariates_label(r),
                r.n_parameters.map(|n| n.to_string()).unwrap_or_default(),
                opt_number(r.ofv),
                number(r.criterion),
                number(r.fitness),
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
                r.non_influential.to_string(),
                r.duplicate_of.clone().unwrap_or_default(),
                r.reused.to_string(),
            ])
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("cannot flush `{}`: {e}", path.display()))
}

fn write_generations(path: &Path, result: &GlobalsearchResult) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    writer
        .write_record(GENERATION_COLUMNS)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    for g in &result.generations {
        let best = result
            .rows
            .iter()
            .find(|r| r.genome.as_ref() == Some(&g.best))
            .map(|r| r.description.clone())
            .unwrap_or_default();
        writer
            .write_record([
                g.index.to_string(),
                best,
                number(g.best_fitness),
                number(g.mean_fitness),
                g.polished.to_string(),
            ])
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("cannot flush `{}`: {e}", path.display()))
}

/// A human-readable rendering of the search: the grid, every model best
/// first, the GA's trajectory, then the notes. What `ferx globalsearch`
/// prints.
pub fn render_summary(result: &GlobalsearchResult) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let label = result.criterion.label();
    let _ = writeln!(
        out,
        "Grid: {} point{} over {} ax{}",
        result.space_size,
        if result.space_size == 1 { "" } else { "s" },
        result.axes.len(),
        if result.axes.len() == 1 { "is" } else { "es" }
    );
    for (name, alleles) in &result.axes {
        let _ = writeln!(out, "  {name}: {}", alleles.join(" | "));
    }
    let input = result.row("input");
    let _ = writeln!(
        out,
        "Input model: OFV {}, {label} {}",
        input
            .and_then(|r| r.ofv)
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "-".into()),
        input
            .map(|r| number_or_dash(r.criterion))
            .unwrap_or_else(|| "-".into())
    );
    let _ = writeln!(
        out,
        "Algorithm: {}, {} model{} fitted of {} evaluated{}",
        result.options.algorithm.label(),
        result.n_fitted(),
        if result.n_fitted() == 1 { "" } else { "s" },
        result.rows.len(),
        if result.cancelled {
            " (search cancelled)"
        } else {
            ""
        }
    );
    let _ = writeln!(
        out,
        "\n  {:<8} {:<44} {:>5} {:>12} {:>12} {:>12} {:>4} {:>7}  {:<13} decision",
        "model", "genome", "npar", "OFV", label, "fitness", "rank", "seconds", "status"
    );
    let mut rows: Vec<&ModelRow> = result.rows.iter().collect();
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
        } else if let Some(rep) = &r.duplicate_of {
            format!("same model as {rep}")
        } else {
            String::new()
        };
        let genome = if r.genome.is_some() {
            r.description.clone()
        } else {
            r.structure
                .map(|s| structure_label(&s))
                .unwrap_or_else(|| "input".into())
        };
        let _ = writeln!(
            out,
            "  {:<8} {:<44} {:>5} {:>12} {:>12} {:>12} {:>4} {:>7.1}  {:<13} {}",
            r.id,
            truncate(&genome, 44),
            r.n_parameters
                .map(|n| n.to_string())
                .unwrap_or_else(|| "-".into()),
            r.ofv
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".into()),
            number_or_dash(r.criterion),
            number_or_dash(r.fitness),
            r.rank.map(|n| n.to_string()).unwrap_or_else(|| "-".into()),
            r.seconds,
            status,
            decision
        );
    }
    if !result.generations.is_empty() {
        let _ = writeln!(
            out,
            "\n  {:>10} {:>12} {:>12} {:>8}  best",
            "generation", "best", "mean", "polished"
        );
        for g in &result.generations {
            let best = result
                .rows
                .iter()
                .find(|r| r.genome.as_ref() == Some(&g.best))
                .map(|r| r.id.clone())
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "  {:>10} {:>12} {:>12} {:>8}  {best}",
                g.index,
                number_or_dash(g.best_fitness),
                number_or_dash(g.mean_fitness),
                g.polished
            );
        }
    }
    let _ = writeln!(
        out,
        "\nFinal model: {} — {} (fitness {})",
        result.final_id,
        result
            .row(&result.final_id)
            .map(|r| r.description.clone())
            .unwrap_or_default(),
        number_or_dash(result.final_fitness)
    );
    if !result.notes.is_empty() {
        let _ = writeln!(out, "\nNotes:");
        for n in &result.notes {
            let _ = writeln!(out, "  - {n}");
        }
    }
    out
}

fn truncate(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        s.to_string()
    } else {
        let head: String = s.chars().take(width - 1).collect();
        format!("{head}…")
    }
}

fn number_or_dash(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.3}")
    } else {
        "-".into()
    }
}
