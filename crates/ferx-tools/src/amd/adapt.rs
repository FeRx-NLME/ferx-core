//! The production [`StepRunner`]: the real tools, and their rows adapted to
//! the pipeline's own table (#1184).
//!
//! Each tool reports its candidates in the shape of its own space — a
//! structure, an η set, an error form, a covariate effect — and AMD's report
//! has to hold all of them in one table. The adapters below are that
//! translation and nothing more: no decision is taken here that the tool did
//! not already take, and every row a tool produced yields exactly one row
//! here, failures and duplicates included.

use ferx_core::edit::ModelText;

use super::{CandidateRow, Step, StepOutput, StepRunner, ToolRunner};
use crate::allometry::{run_allometry, AllometryOptions, AllometryResult, AllometryRun};
use crate::covsearch::{run_covsearch, CovsearchResult, CovsearchRun};
use crate::iivsearch::{run_iivsearch, IivsearchResult, IivsearchRun};
use crate::iovsearch::{run_iovsearch, IovsearchResult, IovsearchRun};
use crate::modelsearch::{run_modelsearch, ModelsearchResult, ModelsearchRun};
use crate::ruvsearch::{run_ruvsearch, RuvsearchResult, RuvsearchRun};
use crate::search::{Candidate, CandidateResult, Criterion, Runner, SearchConfig};

impl StepRunner for ToolRunner {
    fn run_step(
        &self,
        index: usize,
        step: Step,
        dir_name: &str,
        config: &SearchConfig,
        model: &ModelText,
    ) -> Result<StepOutput, String> {
        let (dir, base) = self.base_for(dir_name, model)?;
        let threads = self.threads_for(config);
        let cancel = self.cancel.clone();
        let tool = step.tool();
        match step {
            Step::Structural => {
                let result = run_modelsearch(
                    config,
                    &base,
                    ModelsearchRun {
                        dir: Some(dir),
                        threads,
                        cancel,
                        progress: None,
                    },
                )?;
                Ok(from_modelsearch(index, tool, result))
            }
            Step::Iivsearch => {
                let result = run_iivsearch(
                    config,
                    &base,
                    IivsearchRun {
                        dir: Some(dir),
                        threads,
                        cancel,
                        progress: None,
                    },
                )?;
                Ok(from_iivsearch(index, tool, result))
            }
            Step::Residual => {
                let result = run_ruvsearch(
                    config,
                    &base,
                    RuvsearchRun {
                        dir: Some(dir),
                        threads,
                        cancel,
                        progress: None,
                    },
                )?;
                Ok(from_ruvsearch(index, tool, result))
            }
            Step::Iovsearch => {
                let result = run_iovsearch(
                    config,
                    &base,
                    IovsearchRun {
                        dir: Some(dir),
                        threads,
                        cancel,
                        progress: None,
                    },
                )?;
                Ok(from_iovsearch(index, tool, result))
            }
            Step::Allometry => {
                let options = AllometryOptions::from_config(config)?;
                let mut run_options = config.run_options();
                run_options.criterion = Criterion::Ofv;
                let result = run_allometry(
                    &base,
                    &options,
                    AllometryRun {
                        dir: Some(dir),
                        threads,
                        cancel,
                        run_options,
                    },
                )?;
                Ok(from_allometry(index, tool, &base.text, result))
            }
            Step::Covariates => {
                let result = run_covsearch(
                    config,
                    &base,
                    CovsearchRun {
                        dir: Some(dir),
                        threads,
                        cancel,
                        progress: None,
                    },
                )?;
                Ok(from_covsearch(index, tool, result))
            }
        }
    }

    fn fit_one(
        &self,
        index: usize,
        tool: &str,
        dir_name: &str,
        config: &SearchConfig,
        model: &ModelText,
        n_starts: usize,
    ) -> Result<StepOutput, String> {
        let (dir, base) = self.base_for(dir_name, model)?;
        let mut options = config.run_options();
        // One model is not a ranking: the OFV is the number to report, and the
        // BIC of a lone fit would say nothing the OFV does not.
        options.criterion = Criterion::Ofv;
        options.n_starts = n_starts.max(1);
        if tool == "retries" {
            // The retries pass starts *at* the model's own optimum, so start 0
            // not moving is the pass confirming it rather than the #751 init
            // stall the gate exists to catch — measured on
            // `examples/amd_start.ferxsearch`, where every pass came back
            // "stalled at the initial estimates" for exactly that reason. The
            // rest of the gate still applies, and a pass that fails it is not
            // adopted.
            options.strictness.reject_init_stall = false;
        }
        let mut runner = Runner::new().cache_dir(dir);
        if let Some(t) = self.threads_for(config) {
            runner = runner.threads(t);
        }
        if let Some(flag) = &self.cancel {
            runner = runner.cancel(flag.clone());
        }
        let candidates = [Candidate::new(tool, model.clone())];
        let report = runner.run(&candidates, &base.prepared.population, &options)?;
        let Some(result) = report.results.first() else {
            return Ok(StepOutput {
                model: model.clone(),
                fit: None,
                criterion: Criterion::Ofv,
                value: None,
                passed: false,
                rows: Vec::new(),
                selected: Vec::new(),
                notes: report.warnings,
                cancelled: report.cancelled,
            });
        };
        let mut row = candidate_row(index, tool, result, Criterion::Ofv);
        row.description = format!(
            "{} start{}",
            options.n_starts,
            if options.n_starts == 1 { "" } else { "s" }
        );
        row.selected = true;
        Ok(StepOutput {
            model: model.clone(),
            fit: result.fit.clone(),
            criterion: Criterion::Ofv,
            value: result.ofv,
            passed: result.eligible(),
            rows: vec![row],
            selected: Vec::new(),
            notes: report.warnings,
            cancelled: report.cancelled,
        })
    }
}

/// A [`CandidateResult`] as a pipeline row — the shape allometry and the
/// pipeline's own single-model fits report in.
fn candidate_row(
    index: usize,
    tool: &str,
    result: &CandidateResult,
    criterion: Criterion,
) -> CandidateRow {
    CandidateRow {
        step: index,
        tool: tool.to_string(),
        id: result.id.clone(),
        parent: result.parent.clone(),
        description: result.features.render(),
        criterion: criterion.label(),
        value: result.criterion.is_finite().then_some(result.criterion),
        d_value: None,
        ofv: result.ofv,
        d_ofv: None,
        rank: None,
        converged: result.converged,
        passed: result.verdict.passed,
        failures: result.verdict.failures.clone(),
        error: result.error.as_ref().map(|e| e.message.clone()),
        note: result
            .duplicate_of
            .as_ref()
            .map(|id| format!("same model as `{id}`")),
        seconds: result.seconds,
        selected: false,
    }
}

/// The fields every ranked-model row shares, filled from the parts each tool
/// spells differently.
#[allow(clippy::too_many_arguments)]
fn ranked_row(
    index: usize,
    tool: &str,
    id: &str,
    parent: Option<&String>,
    description: String,
    criterion: Criterion,
    value: f64,
    d_value: Option<f64>,
    ofv: Option<f64>,
    rank: Option<usize>,
    converged: Option<bool>,
    passed: bool,
    failures: &[String],
    error: Option<String>,
    seconds: f64,
    selected: bool,
) -> CandidateRow {
    CandidateRow {
        step: index,
        tool: tool.to_string(),
        id: id.to_string(),
        parent: parent.cloned(),
        description,
        criterion: criterion.label(),
        value: value.is_finite().then_some(value),
        d_value,
        ofv,
        d_ofv: None,
        rank,
        converged,
        passed,
        failures: failures.to_vec(),
        error,
        note: None,
        seconds,
        selected,
    }
}

fn from_modelsearch(index: usize, tool: &str, result: ModelsearchResult) -> StepOutput {
    let criterion = result.criterion;
    let rows = result
        .rows
        .iter()
        .map(|r| {
            ranked_row(
                index,
                tool,
                &r.id,
                r.parent.as_ref(),
                crate::modelsearch::structure_label(&r.structure),
                criterion,
                r.criterion,
                r.d_criterion,
                r.ofv,
                r.rank,
                r.converged,
                r.passed,
                &r.failures,
                r.error.as_ref().map(|e| e.message.clone()),
                r.seconds,
                r.selected,
            )
        })
        .collect();
    let selected = vec![crate::modelsearch::structure_label(
        &result
            .row(&result.final_id)
            .map(|r| r.structure)
            .unwrap_or(result.base_structure),
    )];
    StepOutput {
        model: result.final_model,
        fit: result.final_fit,
        criterion,
        passed: true,
        value: result
            .final_criterion
            .is_finite()
            .then_some(result.final_criterion),
        rows,
        selected,
        notes: result.notes,
        cancelled: result.cancelled,
    }
}

fn from_iivsearch(index: usize, tool: &str, result: IivsearchResult) -> StepOutput {
    let criterion = result.criterion;
    let rows = result
        .rows
        .iter()
        .map(|r| {
            let mut row = ranked_row(
                index,
                tool,
                &r.id,
                r.parent.as_ref(),
                r.structure.description(),
                criterion,
                r.criterion,
                r.d_criterion,
                r.ofv,
                r.rank,
                r.converged,
                r.passed,
                &r.failures,
                r.error.as_ref().map(|e| e.message.clone()),
                r.seconds,
                r.selected,
            );
            if r.starts > 1 {
                row.note = Some(format!("{} starts", r.starts));
            }
            row
        })
        .collect();
    StepOutput {
        model: result.final_model,
        fit: result.final_fit,
        criterion,
        passed: true,
        value: result
            .final_criterion
            .is_finite()
            .then_some(result.final_criterion),
        rows,
        selected: vec![result.final_structure.description()],
        notes: result.notes,
        cancelled: result.cancelled,
    }
}

fn from_iovsearch(index: usize, tool: &str, result: IovsearchResult) -> StepOutput {
    let criterion = result.criterion;
    let rows = result
        .rows
        .iter()
        .map(|r| {
            let mut row = ranked_row(
                index,
                tool,
                &r.id,
                r.parent.as_ref(),
                r.structure.description(),
                criterion,
                r.criterion,
                r.d_criterion,
                r.ofv,
                r.rank,
                r.converged,
                r.passed,
                &r.failures,
                r.error.as_ref().map(|e| e.message.clone()),
                r.seconds,
                r.selected,
            );
            if r.starts > 1 {
                row.note = Some(format!("{} starts", r.starts));
            }
            row
        })
        .collect();
    StepOutput {
        model: result.final_model,
        fit: result.final_fit,
        criterion,
        passed: true,
        value: result
            .final_criterion
            .is_finite()
            .then_some(result.final_criterion),
        rows,
        selected: vec![result.final_structure.description()],
        notes: result.notes,
        cancelled: result.cancelled,
    }
}

fn from_ruvsearch(index: usize, tool: &str, result: RuvsearchResult) -> StepOutput {
    let rows = result
        .rows
        .iter()
        .map(|r| {
            let mut row = ranked_row(
                index,
                tool,
                &r.candidate,
                None,
                r.feature
                    .map(|f| f.label())
                    .unwrap_or_else(|| "base".to_string()),
                Criterion::Ofv,
                r.ofv.unwrap_or(f64::NAN),
                r.ofv.map(|o| o - r.parent_ofv),
                r.ofv,
                None,
                r.converged,
                r.passed,
                &r.failures,
                None,
                r.seconds,
                r.selected,
            );
            row.d_ofv = r.ofv.map(|o| o - r.parent_ofv);
            row.note = match (&r.note, r.lrt.as_ref()) {
                (Some(note), _) => Some(note.clone()),
                (None, Some(lrt)) => Some(format!("p = {:.4} (df {})", lrt.p_value, lrt.df)),
                (None, None) if r.screened => Some("screened out on CWRES".to_string()),
                (None, None) => None,
            };
            row
        })
        .collect();
    StepOutput {
        model: result.final_model,
        fit: result.final_fit,
        criterion: Criterion::Ofv,
        passed: true,
        value: Some(result.final_ofv),
        rows,
        selected: if result.features.is_empty() {
            vec!["no residual-error feature added".to_string()]
        } else {
            result.features.iter().map(|f| f.label()).collect()
        },
        notes: result.notes,
        cancelled: result.cancelled,
    }
}

fn from_covsearch(index: usize, tool: &str, result: CovsearchResult) -> StepOutput {
    let rows = result
        .steps
        .iter()
        .map(|r| {
            let mut row = ranked_row(
                index,
                tool,
                &r.candidate,
                None,
                format!("{} {}", r.phase.label(), r.effect.label()),
                Criterion::Ofv,
                r.ofv.unwrap_or(f64::NAN),
                r.ofv.map(|o| o - r.parent_ofv),
                r.ofv,
                None,
                r.converged,
                r.passed,
                &r.failures,
                None,
                r.seconds,
                r.selected,
            );
            row.d_ofv = r.ofv.map(|o| o - r.parent_ofv);
            row.note = match (&r.note, r.lrt.as_ref()) {
                (Some(note), _) => Some(note.clone()),
                (None, Some(lrt)) => Some(format!("p = {:.4} (df {})", lrt.p_value, lrt.df)),
                (None, None) => None,
            };
            row
        })
        .collect();
    StepOutput {
        model: result.final_model,
        fit: result.final_fit,
        criterion: Criterion::Ofv,
        passed: true,
        value: Some(result.final_ofv),
        rows,
        selected: if result.included.is_empty() {
            vec!["no covariate effect included".to_string()]
        } else {
            result
                .included
                .iter()
                .map(|i| format!("{} ({})", i.effect.label(), i.origin.label()))
                .collect()
        },
        notes: result.notes,
        cancelled: result.cancelled,
    }
}

/// Allometry is not a ranked search: the scaling is a mechanistic statement,
/// and the tool fits it beside the unscaled model so a reader can see what it
/// cost. AMD adopts it when it passes the strictness gate — and keeps the
/// unscaled model, with a note, when it does not.
fn from_allometry(
    index: usize,
    tool: &str,
    input: &ModelText,
    result: AllometryResult,
) -> StepOutput {
    let accepted = result.scaled.eligible();
    let mut rows = vec![
        candidate_row(index, tool, &result.base, Criterion::Ofv),
        candidate_row(index, tool, &result.scaled, Criterion::Ofv),
    ];
    rows[1].selected = accepted;
    rows[0].selected = !accepted;
    rows[1].d_ofv = match (result.scaled.ofv, result.base.ofv) {
        (Some(a), Some(b)) => Some(a - b),
        _ => None,
    };
    let mut notes = result.notes;
    let selected = if accepted {
        result
            .scalings
            .iter()
            .map(|s| {
                format!(
                    "{} ~ size^{:.3}{}",
                    s.parameter,
                    s.exponent,
                    if s.fixed { " (fixed)" } else { "" }
                )
            })
            .collect()
    } else {
        notes.push(format!(
            "the allometric model did not pass the strictness gate ({}); the unscaled model was \
             kept",
            if result.scaled.verdict.failures.is_empty() {
                result
                    .scaled
                    .error
                    .as_ref()
                    .map(|e| e.message.clone())
                    .unwrap_or_else(|| "no fit".into())
            } else {
                result.scaled.verdict.failures.join("; ")
            }
        ));
        vec!["allometric scaling not applied".to_string()]
    };
    let (model, fit, ofv) = if accepted {
        (result.model, result.scaled.fit, result.scaled.ofv)
    } else {
        (input.clone(), result.base.fit, result.base.ofv)
    };
    StepOutput {
        model,
        fit,
        criterion: Criterion::Ofv,
        value: ofv,
        passed: accepted,
        rows,
        selected,
        notes,
        cancelled: result.cancelled,
    }
}
