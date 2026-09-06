//! `steps.csv`, `models/<id>.ferx` and `final.ferx` — what a residual-error
//! search leaves behind (#1182).
//!
//! One row per fitted model — the input, the proportional base, every
//! candidate of every iteration, and (with the pre-screen) every CWRES
//! screening model — with the likelihood-ratio comparison beside the
//! convergence status and the strictness verdict, so a candidate that looks
//! better on OFV alone but was not chosen carries its reason on the row.

use std::path::{Path, PathBuf};

use ferx_core::edit::ModelEdit;

use super::RuvsearchResult;

pub fn steps_path(dir: &Path) -> PathBuf {
    dir.join("steps.csv")
}

pub fn final_model_path(dir: &Path) -> PathBuf {
    dir.join("final.ferx")
}

/// Where every fitted model's text goes: `<dir>/models/<id>.ferx`.
pub fn models_dir(dir: &Path) -> PathBuf {
    dir.join("models")
}

/// The columns of `steps.csv`, in order.
pub const STEP_COLUMNS: [&str; 17] = [
    "iteration",
    "candidate",
    "feature",
    "screened",
    "parent_ofv",
    "ofv",
    "dofv",
    "df",
    "p_value",
    "alpha",
    "significant",
    "cwres_dofv",
    "selected",
    "converged",
    "passed",
    "failures",
    "seconds",
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

/// Write `steps.csv`, `models/<id>.ferx` and `final.ferx` into `dir`.
///
/// The final model is the selected model's text with its own final
/// estimates written into the initial values — Pharmpy's `update_inits` on
/// the final model — so it reads and refits as the search left it.
pub fn write_report(dir: &Path, result: &RuvsearchResult) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("cannot create search directory `{}`: {e}", dir.display()))?;
    write_steps(&steps_path(dir), result)?;
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

fn write_steps(path: &Path, result: &RuvsearchResult) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    writer
        .write_record(STEP_COLUMNS)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    for r in &result.rows {
        let lrt = r.lrt;
        let failures = if r.failures.is_empty() {
            r.note.clone().unwrap_or_default()
        } else {
            r.failures.join("; ")
        };
        writer
            .write_record([
                r.iteration.to_string(),
                r.candidate.clone(),
                r.feature.map(|f| f.label()).unwrap_or_default(),
                r.screened.to_string(),
                number(r.parent_ofv),
                opt_number(r.ofv),
                opt_number(lrt.map(|t| t.dofv)),
                lrt.map(|t| t.df.to_string()).unwrap_or_default(),
                opt_number(lrt.map(|t| t.p_value)),
                lrt.map(|t| number(t.alpha)).unwrap_or_default(),
                lrt.map(|t| t.significant.to_string()).unwrap_or_default(),
                opt_number(r.cwres_dofv),
                r.selected.to_string(),
                r.converged.map(|c| c.to_string()).unwrap_or_default(),
                r.passed.to_string(),
                failures,
                number(r.seconds),
            ])
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("cannot flush `{}`: {e}", path.display()))
}

/// A human-readable rendering of the search: the input and base, each
/// iteration's table, the final model. What `ferx ruvsearch` prints.
pub fn render_summary(result: &RuvsearchResult) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "Input model: OFV {:.3}", result.input_ofv);
    if result.base_id != "input" {
        let _ = writeln!(
            out,
            "Proportional base: OFV {:.3} ({:+.3} vs input)",
            result.base_ofv,
            result.base_ofv - result.input_ofv
        );
    }
    let n_iter = result.n_iterations();
    for iteration in 1..=n_iter {
        let rows: Vec<_> = result.iteration_rows(iteration).collect();
        if rows.is_empty() {
            continue;
        }
        let screened: Vec<_> = rows.iter().filter(|r| r.screened).collect();
        let fitted: Vec<_> = rows.iter().filter(|r| !r.screened).collect();
        if !screened.is_empty() {
            let base_ofv = screened
                .iter()
                .find(|r| r.feature.is_none())
                .and_then(|r| r.ofv)
                .or_else(|| screened.iter().find_map(|r| Some(r.parent_ofv)))
                .unwrap_or(f64::NAN);
            let _ = writeln!(
                out,
                "\nIteration {iteration}, CWRES pre-screen (base OFV {:.3}, cutoff {:.3})",
                base_ofv,
                result.options.cutoff()
            );
            let _ = writeln!(
                out,
                "  {:<16} {:>12} {:>10}  {:<13} decision",
                "candidate", "OFV", "dOFV", "status"
            );
            for r in &screened {
                let _ = writeln!(
                    out,
                    "  {:<16} {:>12} {:>10}  {:<13} {}",
                    r.feature
                        .map(|f| f.label())
                        .unwrap_or_else(|| "cwres base".to_string()),
                    r.ofv
                        .map(|v| format!("{v:.3}"))
                        .unwrap_or_else(|| "-".into()),
                    r.cwres_dofv
                        .map(|v| format!("{v:.3}"))
                        .unwrap_or_else(|| "-".into()),
                    status(r),
                    if r.selected { "REFIT" } else { "" }
                );
            }
        }
        if let Some(first) = fitted.first() {
            let _ = writeln!(
                out,
                "\nIteration {iteration}, parent OFV {:.3}, p = {}",
                first.parent_ofv, result.options.p_value
            );
            let _ = writeln!(
                out,
                "  {:<16} {:>12} {:>10} {:>3} {:>10}  {:<13} decision",
                "candidate", "OFV", "dOFV", "df", "p", "status"
            );
            for r in &fitted {
                let decision = if r.selected {
                    "SELECTED".to_string()
                } else if let Some(n) = &r.note {
                    n.clone()
                } else if !r.passed {
                    r.failures.first().cloned().unwrap_or_default()
                } else if r.lrt.is_some_and(|t| !t.significant) {
                    "not significant".to_string()
                } else {
                    String::new()
                };
                let _ = writeln!(
                    out,
                    "  {:<16} {:>12} {:>10} {:>3} {:>10}  {:<13} {}",
                    r.feature.map(|f| f.label()).unwrap_or_default(),
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
                    status(r),
                    decision
                );
            }
        }
    }
    let features = if result.features.is_empty() {
        "none".to_string()
    } else {
        result
            .features
            .iter()
            .map(|f| f.label())
            .collect::<Vec<_>>()
            .join(" + ")
    };
    let _ = writeln!(
        out,
        "\nFinal model: {} — OFV {:.3} ({:+.3} vs input); features added: {features}{}",
        result.final_id,
        result.final_ofv,
        result.final_ofv - result.input_ofv,
        if result.cancelled {
            " (search cancelled)"
        } else {
            ""
        }
    );
    if !result.notes.is_empty() {
        let _ = writeln!(out, "\nNotes:");
        for n in &result.notes {
            let _ = writeln!(out, "  - {n}");
        }
    }
    out
}

fn status(r: &super::StepRow) -> String {
    match (r.converged, r.passed, r.ofv) {
        (_, _, None) => "failed".to_string(),
        (_, true, _) => "ok".to_string(),
        (Some(false), false, _) => "not converged".to_string(),
        (_, false, _) => "excluded".to_string(),
    }
}
