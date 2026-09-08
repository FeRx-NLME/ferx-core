//! `amd` — the automatic model development pipeline (#1184).
//!
//! Pharmpy's `amd`: run the search tools of the epic one after another, each
//! starting from the model the last one selected, and report the whole thing
//! as one object. There is no new numerics here and no new search — every
//! decision is made by [`modelsearch`](crate::modelsearch),
//! [`iivsearch`](crate::iivsearch), [`ruvsearch`](crate::ruvsearch),
//! [`iovsearch`](crate::iovsearch), [`allometry`](crate::allometry) and
//! [`covsearch`](crate::covsearch). What this module owns is the sequencing,
//! the seeding between steps, and the report.
//!
//! # The pipeline
//!
//! The default order is Pharmpy's: **structural → IIV → residual → IOV →
//! allometry → covariates**. [`Strategy`] reorders the same components;
//! `reevaluation` appends a second IIV and residual pass after the covariate
//! step, because both were decided on a model that had not yet acquired its
//! covariates.
//!
//! Each step is handed a *subspace* of the one `[space] mfl` — see the
//! `space` module — and the model the previous step selected, seeded from that
//! step's estimates (`search::seed::seed_from`). A step
//! whose subspace is empty, or whose precondition the model does not meet (an
//! IOV search on a model with no `iov_column`), is **skipped with its reason
//! recorded**; it is never silently dropped and never run on a narrowed space.
//!
//! # The report is the product
//!
//! A search report that shows only the winner is unauditable, and the failure
//! modes this pipeline is most exposed to — init stalls (#751), inner-EBE
//! modes (#864, #891), boundary estimates — are exactly the ones that hide
//! behind a winner-only table. So [`AmdResult::rows`] carries **every
//! candidate of every step** with the criterion it was ranked on, its Δ
//! against its parent, the strictness verdict *with its reasons*, whether the
//! fit converged, and the wall-clock seconds it cost. `candidates.csv` is that
//! table; `steps.csv` is one row per step; each tool's own directory keeps its
//! own fuller record.
//!
//! # Retries
//!
//! Pharmpy's `retries` tool refits a selected model from perturbed initial
//! estimates, because a search ranks models on numbers a stalled or
//! locally-trapped fit invalidates. ferx has that as `n_starts` in core, and
//! the runner already applies `[run] retries` to **every candidate** before the
//! strictness gate — so the per-candidate half of Pharmpy's motivation is
//! already inside every step. What is left is the *selected* model, whose
//! initial estimates after seeding are its own optimum: refitting it with
//! `retries + 1` starts explores around that optimum rather than re-deriving
//! it. [`Retries`] says which selected models get that pass.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ferx_core::edit::ModelText;
use ferx_core::{prepare_run, CancelFlag, FitResult};
use serde::Deserialize;

use crate::search::{BaseModel, Criterion, Mfl, SearchConfig};

mod adapt;
#[cfg(test)]
mod pharmpy_anchor;
pub mod report;
mod space;

pub use report::{
    candidates_path, final_model_path, render_summary, steps_path, write_report, CANDIDATE_COLUMNS,
    STEP_COLUMNS,
};

/// One component of the pipeline, spelled as Pharmpy's `amd` spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Step {
    /// The structural PK model — [`modelsearch`](crate::modelsearch).
    Structural,
    /// The inter-individual variability structure — [`iivsearch`](crate::iivsearch).
    Iivsearch,
    /// The residual error model — [`ruvsearch`](crate::ruvsearch).
    Residual,
    /// Inter-occasion variability — [`iovsearch`](crate::iovsearch).
    Iovsearch,
    /// Allometric scaling on a size covariate — [`allometry`](crate::allometry).
    Allometry,
    /// The covariate model — [`covsearch`](crate::covsearch).
    Covariates,
}

impl Step {
    /// Every step, in the default order.
    pub const ALL: [Step; 6] = [
        Step::Structural,
        Step::Iivsearch,
        Step::Residual,
        Step::Iovsearch,
        Step::Allometry,
        Step::Covariates,
    ];

    /// The `[amd] skip` spelling, and Pharmpy's section name.
    pub fn label(&self) -> &'static str {
        match self {
            Step::Structural => "structural",
            Step::Iivsearch => "iivsearch",
            Step::Residual => "residual",
            Step::Iovsearch => "iovsearch",
            Step::Allometry => "allometry",
            Step::Covariates => "covariates",
        }
    }

    /// The tool that runs it — which is *not* the step's own name for
    /// `structural` (`modelsearch`), `residual` (`ruvsearch`) or `covariates`
    /// (`covsearch`).
    pub fn tool(&self) -> &'static str {
        match self {
            Step::Structural => "modelsearch",
            Step::Iivsearch => "iivsearch",
            Step::Residual => "ruvsearch",
            Step::Iovsearch => "iovsearch",
            Step::Allometry => "allometry",
            Step::Covariates => "covsearch",
        }
    }

    /// The MFL statements this step's subspace is built from, for the message
    /// that explains a skip.
    fn space_says(&self) -> &'static str {
        match self {
            Step::Structural => "ABSORPTION / ELIMINATION / PERIPHERALS / TRANSITS / LAGTIME",
            Step::Iivsearch => "IIV / COVARIANCE(IIV, ...)",
            Step::Residual => "",
            Step::Iovsearch => "IOV / COVARIANCE(IOV, ...)",
            Step::Allometry => "ALLOMETRY",
            Step::Covariates => "COVARIATE",
        }
    }
}

/// The step orderings Pharmpy's `amd` offers (`get_subtool_order`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Strategy {
    /// structural → IIV → residual → IOV → allometry → covariates.
    #[default]
    Default,
    /// [`Default`](Strategy::Default), then IIV and residual again — both were
    /// decided before the model had its covariates.
    Reevaluation,
    /// structural → IIV → residual.
    #[serde(alias = "SIR")]
    Sir,
    /// structural → residual → IIV.
    #[serde(alias = "SRI")]
    Sri,
    /// residual → structural → IIV.
    #[serde(alias = "RSI")]
    Rsi,
}

impl Strategy {
    /// The `[amd] strategy` spelling.
    pub fn label(&self) -> &'static str {
        match self {
            Strategy::Default => "default",
            Strategy::Reevaluation => "reevaluation",
            Strategy::Sir => "SIR",
            Strategy::Sri => "SRI",
            Strategy::Rsi => "RSI",
        }
    }

    /// The steps this strategy runs, in order. A step appearing twice is a
    /// rerun; [`plan`] marks the second occurrence.
    pub fn order(&self) -> Vec<Step> {
        use Step::*;
        match self {
            Strategy::Default => vec![
                Structural, Iivsearch, Residual, Iovsearch, Allometry, Covariates,
            ],
            Strategy::Reevaluation => vec![
                Structural, Iivsearch, Residual, Iovsearch, Allometry, Covariates, Iivsearch,
                Residual,
            ],
            Strategy::Sir => vec![Structural, Iivsearch, Residual],
            Strategy::Sri => vec![Structural, Residual, Iivsearch],
            Strategy::Rsi => vec![Residual, Structural, Iivsearch],
        }
    }
}

/// Which selected models get the perturbed-restart pass. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Retries {
    /// Every step's selected model, and the pipeline's final model — Pharmpy's
    /// default.
    #[default]
    AllFinal,
    /// The pipeline's final model only.
    Final,
    /// No extra pass. Every candidate is still fitted with `[run] retries`
    /// starts inside its step; this turns off the pass on the *winner*.
    Skip,
}

impl Retries {
    pub fn label(&self) -> &'static str {
        match self {
            Retries::AllFinal => "all_final",
            Retries::Final => "final",
            Retries::Skip => "skip",
        }
    }
}

/// `[amd]`.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct AmdOptions {
    /// The step ordering.
    #[serde(default)]
    pub strategy: Strategy,
    /// Which selected models get the perturbed-restart pass.
    #[serde(default)]
    pub retries: Retries,
    /// Steps to leave out, by [`Step::label`]. A step the space says nothing
    /// about is skipped anyway; this is for one the space *does* describe.
    #[serde(default)]
    pub skip: Vec<Step>,
}

impl AmdOptions {
    /// Read `[amd]` from a loaded `.ferxsearch` file, and check that the
    /// space's every statement belongs to a step this pipeline runs.
    pub fn from_config(config: &SearchConfig) -> Result<Self, String> {
        let options = match config.tools.get("amd") {
            Some(table) => table
                .clone()
                .try_into::<AmdOptions>()
                .map_err(|e| format!("[amd]: {e}"))?,
            None => AmdOptions::default(),
        };
        space::check(&config.mfl)?;
        options.validate()?;
        Ok(options)
    }

    pub fn validate(&self) -> Result<(), String> {
        let order = self.strategy.order();
        for step in &self.skip {
            if !order.contains(step) {
                return Err(format!(
                    "[amd] skip = \"{}\" is not a step of the {} strategy, which runs {}",
                    step.label(),
                    self.strategy.label(),
                    order
                        .iter()
                        .map(|s| s.label())
                        .collect::<Vec<_>>()
                        .join(" -> ")
                ));
            }
        }
        if order.iter().all(|s| self.skip.contains(s)) {
            return Err("[amd] skip leaves the pipeline with no steps to run".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

/// One entry of the pipeline before anything is fitted: which step, where its
/// files go, and — when it will not run — why.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedStep {
    /// 1-based position in the pipeline.
    pub index: usize,
    pub step: Step,
    /// `true` for the second occurrence of a step under `reevaluation`.
    pub rerun: bool,
    /// The directory name under the run directory: `02-iivsearch`,
    /// `07-rerun-iivsearch`.
    pub dir: String,
    /// Set when the step will be skipped, saying why.
    pub skipped: Option<String>,
}

/// What a step needs to know about the starting model before the pipeline
/// runs: the facts a skip decision is made on.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Context {
    /// `[fit_options] iov_column` of the starting model — `iovsearch` cannot
    /// run without one, since the dataset's occasions are not read.
    pub iov_column: Option<String>,
}

impl Context {
    /// Read from the model AMD starts on, which is where `iovsearch` reads it
    /// too (Pharmpy's `amd_start_model`).
    pub fn from_base(base: &BaseModel) -> Context {
        Context {
            iov_column: base.prepared.parsed.fit_options.iov_column.clone(),
        }
    }
}

/// The pipeline, decided up front: every step in order, each either runnable
/// or skipped with a reason.
///
/// Computed before the first fit so the decisions are inspectable — and
/// testable — without running a search.
pub fn plan(options: &AmdOptions, mfl: &Mfl, ctx: &Context) -> Result<Vec<PlannedStep>, String> {
    let order = options.strategy.order();
    let mut seen: Vec<Step> = Vec::new();
    let mut plan = Vec::new();
    for (i, step) in order.into_iter().enumerate() {
        let rerun = seen.contains(&step);
        seen.push(step);
        let dir = format!(
            "{:02}-{}{}",
            i + 1,
            if rerun { "rerun-" } else { "" },
            step.tool()
        );
        let skipped = skip_reason(step, options, mfl, ctx)?;
        plan.push(PlannedStep {
            index: i + 1,
            step,
            rerun,
            dir,
            skipped,
        });
    }
    Ok(plan)
}

/// Why `step` will not run, or `None` when it will.
fn skip_reason(
    step: Step,
    options: &AmdOptions,
    mfl: &Mfl,
    ctx: &Context,
) -> Result<Option<String>, String> {
    if options.skip.contains(&step) {
        return Ok(Some(format!(
            "left out by `[amd] skip = [\"{}\"]`",
            step.label()
        )));
    }
    // The residual step searches the error forms, not a space: it always runs.
    if step == Step::Residual {
        return Ok(None);
    }
    if step == Step::Iovsearch {
        // An IOV search reads the occasions from the model's `iov_column`; a
        // model without one has a population that never read them, so there is
        // nothing to search rather than an empty search.
        if ctx.iov_column.is_none() {
            return Ok(Some(
                "the starting model declares no `iov_column` in [fit_options], so the dataset's \
                 occasions were not read"
                    .into(),
            ));
        }
        // With an occasion column, iovsearch defaults to every parameter with
        // a free η, so an empty subspace is not a reason to skip it.
        return Ok(None);
    }
    if space::subspace(mfl, step)?.features().next().is_none() {
        return Ok(Some(format!(
            "the search space has no {} statement",
            step.space_says()
        )));
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

/// One candidate of one step, in the pipeline's own vocabulary.
///
/// Every search tool reports its candidates in a shape that suits its own
/// space — a structure, an effect, an error form — and AMD's table has to hold
/// all of them side by side. So each tool's rows are adapted to this, losing
/// nothing a reader needs: what the candidate was, what it scored, what the
/// gate said, and what it cost. The tool's own table (`models.csv`,
/// `steps.csv`) stays in its step directory with the fuller record.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateRow {
    /// The pipeline position of the step this candidate belongs to.
    pub step: usize,
    /// [`Step::tool`], or `start` / `retries` for the pipeline's own fits.
    pub tool: String,
    /// The id the step knew it by; unique only within the step.
    pub id: String,
    pub parent: Option<String>,
    /// What the candidate *is*: its structure, its η set, the effect added.
    pub description: String,
    /// [`Criterion::label`] of what this step ranked on.
    pub criterion: &'static str,
    /// The criterion's value; `None` when there is no fit to evaluate it on.
    pub value: Option<f64>,
    /// The criterion against the candidate's parent.
    pub d_value: Option<f64>,
    pub ofv: Option<f64>,
    /// `ofv − parent ofv`: negative when the candidate fits better.
    pub d_ofv: Option<f64>,
    /// Rank among the step's candidates that passed the gate.
    pub rank: Option<usize>,
    /// `None` when there is no fit, which is not `Some(false)`.
    pub converged: Option<bool>,
    /// The strictness gate's verdict.
    pub passed: bool,
    /// Every gate the fit failed, in the gate's own words.
    pub failures: Vec<String>,
    /// Why there is no fit, when that is the reason.
    pub error: Option<String>,
    /// Anything else the step said about this candidate — a likelihood-ratio
    /// p-value, a pre-screen decision.
    pub note: Option<String>,
    /// Wall-clock seconds; `0.0` for a reused or duplicated fit.
    pub seconds: f64,
    /// The candidate the step carried forward.
    pub selected: bool,
}

/// Fill `d_ofv` from each row's parent *within the same step*.
///
/// The tools that rank on a criterion report `d_criterion` and not `ΔOFV`;
/// #1184 asks for both, and the parent is in the table beside the child, so
/// the second is derived here rather than added to five row types.
fn fill_d_ofv(rows: &mut [CandidateRow]) {
    let parents: Vec<(String, Option<f64>)> = rows.iter().map(|r| (r.id.clone(), r.ofv)).collect();
    for row in rows.iter_mut() {
        if row.d_ofv.is_some() {
            continue;
        }
        let (Some(parent), Some(ofv)) = (&row.parent, row.ofv) else {
            continue;
        };
        if let Some((_, Some(parent_ofv))) = parents.iter().find(|(id, _)| id == parent) {
            row.d_ofv = Some(ofv - parent_ofv);
        }
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// What one step of the pipeline did.
#[derive(Debug, Clone, PartialEq)]
pub struct StepOutcome {
    pub index: usize,
    pub step: Step,
    pub rerun: bool,
    /// The step's directory name under the run directory.
    pub dir: String,
    /// Set when the step did not run.
    pub skipped: Option<String>,
    /// Set when the step ran and failed. The pipeline carries on from the
    /// model it was handed: a covariate step that cannot resolve its space is
    /// a reason to report that step as failed, not to throw away the four
    /// steps that already ran.
    pub failed: Option<String>,
    /// [`Criterion::label`] the step ranked on; empty for a skipped step.
    pub criterion: &'static str,
    /// The criterion of the model the step started from, on the step's own
    /// criterion.
    pub value_before: Option<f64>,
    /// The criterion of the model it selected.
    pub value_after: Option<f64>,
    pub ofv_before: Option<f64>,
    pub ofv_after: Option<f64>,
    /// What the step chose, in words: the structure, the η, the effects.
    pub selected: Vec<String>,
    /// How many candidates the step fitted.
    pub candidates: usize,
    /// Anything the step wanted said once.
    pub notes: Vec<String>,
    /// Wall-clock seconds for the whole step, the retries pass included.
    pub seconds: f64,
}

impl StepOutcome {
    /// Whether this step ran at all — a failed step did.
    pub fn ran(&self) -> bool {
        self.skipped.is_none()
    }

    /// Whether it ran and produced a selection.
    pub fn ok(&self) -> bool {
        self.skipped.is_none() && self.failed.is_none()
    }

    /// The one-word status the tables print.
    pub fn status(&self) -> &'static str {
        match (&self.skipped, &self.failed) {
            (Some(_), _) => "skipped",
            (None, Some(_)) => "failed",
            (None, None) => "ran",
        }
    }

    /// Why it was skipped, or why it failed.
    pub fn reason(&self) -> Option<&str> {
        self.skipped.as_deref().or(self.failed.as_deref())
    }
}

/// What the pipeline did.
#[derive(Debug, Clone)]
pub struct AmdResult {
    pub options: AmdOptions,
    /// The model AMD started from.
    pub input_model: ModelText,
    /// Its fit, when the start fit succeeded.
    pub input_fit: Option<FitResult>,
    /// One entry per planned step, skipped ones included.
    pub steps: Vec<StepOutcome>,
    /// Every candidate of every step, in pipeline order.
    pub rows: Vec<CandidateRow>,
    /// The model the pipeline ended on, seeded from its own estimates.
    pub final_model: ModelText,
    pub final_fit: Option<FitResult>,
    /// Notes from the pipeline itself, distinct from a step's own.
    pub notes: Vec<String>,
    /// The run stopped on a flipped [`CancelFlag`]; the steps after the last
    /// one in `steps` never ran.
    pub cancelled: bool,
}

impl AmdResult {
    /// The steps that actually ran, failures included.
    pub fn ran(&self) -> impl Iterator<Item = &StepOutcome> {
        self.steps.iter().filter(|s| s.ran())
    }

    /// The steps that ran and failed. A non-empty list is what makes a run
    /// incomplete without making it useless.
    pub fn failures(&self) -> impl Iterator<Item = &StepOutcome> {
        self.steps.iter().filter(|s| s.failed.is_some())
    }

    /// `final OFV − input OFV`: negative when the pipeline improved the model.
    pub fn d_ofv(&self) -> Option<f64> {
        let input = self.input_fit.as_ref()?.ofv;
        let last = self.final_fit.as_ref()?.ofv;
        Some(last - input)
    }
}

/// Progress, for a caller that wants to print it.
#[derive(Debug, Clone, PartialEq)]
pub enum AmdEvent {
    /// The plan, before the first fit.
    Planned {
        steps: Vec<PlannedStep>,
    },
    /// The starting model is being fitted.
    StartStarted,
    StartFinished {
        ofv: Option<f64>,
    },
    StepStarted {
        /// The step's position in the plan, skipped steps included — the
        /// number in its directory name.
        index: usize,
        /// Its position among the steps that will actually run, and how many
        /// there are: the `3/5` a progress line wants.
        position: usize,
        total: usize,
        step: Step,
        rerun: bool,
    },
    StepSkipped {
        index: usize,
        step: Step,
        reason: String,
    },
    /// The step ran and failed; the pipeline carries on from the model it was
    /// handed.
    StepFailed {
        index: usize,
        step: Step,
        error: String,
    },
    StepFinished {
        index: usize,
        step: Step,
        criterion: &'static str,
        before: Option<f64>,
        after: Option<f64>,
        selected: Vec<String>,
    },
    /// The perturbed-restart pass on a selected model.
    RetriesStarted {
        index: Option<usize>,
        starts: usize,
    },
    RetriesFinished {
        improved: bool,
        ofv: Option<f64>,
    },
}

/// The progress callback's type.
pub type ProgressFn<'a> = &'a (dyn Fn(AmdEvent) + Send + Sync);

/// Everything [`run_amd`] needs beyond the config and the base model.
#[derive(Default)]
pub struct AmdRun<'a> {
    /// Where every step's directory, `steps.csv`, `candidates.csv` and
    /// `final.ferx` go. Required: AMD materialises each step's input model as
    /// a file, both because that is what the next tool reads and because the
    /// report is the product.
    pub dir: PathBuf,
    /// Total worker threads; overrides `[run] threads`.
    pub threads: Option<usize>,
    pub cancel: Option<CancelFlag>,
    pub progress: Option<ProgressFn<'a>>,
}

/// Where a run's files go by default: `<config stem>-amd` next to the config.
pub fn default_dir(config_path: &Path) -> PathBuf {
    crate::search::default_dir(config_path, "amd")
}

// ---------------------------------------------------------------------------
// The seam
// ---------------------------------------------------------------------------

/// One step's outcome, as the runner behind the seam returns it.
pub(crate) struct StepOutput {
    /// The selected model, **not** yet seeded — AMD seeds it for the next step.
    pub model: ModelText,
    pub fit: Option<FitResult>,
    /// What the step ranked its candidates on.
    pub criterion: Criterion,
    /// The selected model's criterion value.
    pub value: Option<f64>,
    /// Whether [`fit`](Self::fit) passed the strictness gate. Read only for
    /// the retries pass, which must not adopt a fit the gate rejected.
    pub passed: bool,
    pub rows: Vec<CandidateRow>,
    pub selected: Vec<String>,
    pub notes: Vec<String>,
    pub cancelled: bool,
}

/// Runs one step.
///
/// The seam exists for the same reason [`StepFitter`](crate::search::fitter)
/// does one level down: every decision this module makes — the order, the
/// skips, the seeding between steps, the retries policy, the report — is
/// observable only through the models it hands the next step, and none of it
/// is reachable in a unit test if the orchestrator calls `run_modelsearch`
/// directly. Production is [`ToolRunner`]; the tests script it.
pub(crate) trait StepRunner {
    /// Run `step` on `model`, journalling under `<dir>/<dir_name>`.
    fn run_step(
        &self,
        index: usize,
        step: Step,
        dir_name: &str,
        config: &SearchConfig,
        model: &ModelText,
    ) -> Result<StepOutput, String>;

    /// Fit one model on its own, with `n_starts` starts — the pipeline's start
    /// fit and its perturbed-restart passes.
    fn fit_one(
        &self,
        index: usize,
        tool: &str,
        dir_name: &str,
        config: &SearchConfig,
        model: &ModelText,
        n_starts: usize,
    ) -> Result<StepOutput, String>;
}

/// The production step runner: the real tools, one directory each.
pub(crate) struct ToolRunner {
    pub dir: PathBuf,
    pub threads: Option<usize>,
    pub cancel: Option<CancelFlag>,
    /// The dataset every step is fitted against, resolved once from the
    /// starting model so a step's own `[data]` block — which travels with the
    /// edited model text into a *different* directory — cannot re-resolve to
    /// something else.
    pub data_path: String,
}

impl ToolRunner {
    /// Materialise `model` as `<dir>/<dir_name>/input.ferx` and read it back
    /// as a [`BaseModel`].
    ///
    /// Through a file rather than in memory, for three reasons: it is the
    /// artifact the step's report needs anyway, `prepare_run` re-validates the
    /// edited model against the data exactly as the user's own re-run would,
    /// and there is no second construction path for a `PreparedRun` to drift
    /// from.
    fn base_for(&self, dir_name: &str, model: &ModelText) -> Result<(PathBuf, BaseModel), String> {
        let dir = self.dir.join(dir_name);
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("cannot create `{}`: {e}", dir.display()))?;
        let path = dir.join("input.ferx");
        std::fs::write(&path, model.render())
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
        let prepared = prepare_run(&path.to_string_lossy(), Some(&self.data_path))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let text = ModelText::parse(&model.render())?;
        Ok((dir, BaseModel { prepared, text }))
    }

    fn threads_for(&self, config: &SearchConfig) -> Option<usize> {
        self.threads.or(config.run.threads)
    }
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

/// Run the pipeline.
///
/// `base` is the model AMD starts from; every step after the first starts from
/// the previous step's selection instead.
pub fn run_amd(
    config: &SearchConfig,
    base: &BaseModel,
    run: AmdRun<'_>,
) -> Result<AmdResult, String> {
    let options = AmdOptions::from_config(config)?;
    let ctx = Context::from_base(base);
    let plan = plan(&options, &config.mfl, &ctx)?;
    std::fs::create_dir_all(&run.dir)
        .map_err(|e| format!("cannot create run directory `{}`: {e}", run.dir.display()))?;
    let runner = ToolRunner {
        dir: run.dir.clone(),
        threads: run.threads,
        cancel: run.cancel.clone(),
        data_path: base.prepared.data_path.clone(),
    };
    let result = drive(&runner, config, &options, &plan, &base.text, run.progress)?;
    write_report(&run.dir, &result)?;
    Ok(result)
}

/// The pipeline itself, over the seam.
pub(crate) fn drive(
    runner: &dyn StepRunner,
    config: &SearchConfig,
    options: &AmdOptions,
    plan: &[PlannedStep],
    input: &ModelText,
    progress: Option<ProgressFn<'_>>,
) -> Result<AmdResult, String> {
    let emit = |event: AmdEvent| {
        if let Some(f) = progress {
            f(event);
        }
    };
    emit(AmdEvent::Planned {
        steps: plan.to_vec(),
    });

    let mut rows: Vec<CandidateRow> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let mut steps: Vec<StepOutcome> = Vec::new();
    let mut cancelled = false;

    // The starting model, fitted once so every later Δ has a referent and the
    // report can say what the pipeline was worth.
    emit(AmdEvent::StartStarted);
    let start = runner.fit_one(
        0,
        "start",
        "00-start",
        config,
        input,
        config.run.retries + 1,
    )?;
    let input_fit = start.fit.clone();
    let mut current = start.model.clone();
    let mut current_fit = start.fit.clone();
    rows.extend(start.rows);
    notes.extend(start.notes);
    notes.push(INIT_STALL_NOTE.to_string());
    if !start.passed {
        notes.push(format!(
            "the starting model does not pass the strictness gate ({}); every Δ in this report \
             is against a fit that has not been vouched for",
            rows.first()
                .map(|r| r.failures.join("; "))
                .filter(|f| !f.is_empty())
                .unwrap_or_else(|| "no reason given".into())
        ));
    }
    emit(AmdEvent::StartFinished {
        ofv: current_fit.as_ref().map(|f| f.ofv),
    });
    if start.fit.is_none() && !start.cancelled {
        return Err(format!(
            "the starting model could not be fitted, so the pipeline has no baseline to \
             compare against: {}",
            rows.first()
                .and_then(|r| r.error.clone())
                .or_else(|| rows.first().map(|r| r.failures.join("; ")))
                .unwrap_or_else(|| "no result".into())
        ));
    }
    if start.cancelled {
        return Ok(AmdResult {
            options: options.clone(),
            input_model: input.clone(),
            input_fit,
            steps,
            rows,
            final_model: current,
            final_fit: current_fit,
            notes,
            cancelled: true,
        });
    }
    seed(&mut current, current_fit.as_ref())?;

    let total = plan.iter().filter(|p| p.skipped.is_none()).count();
    let mut ran = 0usize;
    for planned in plan {
        if let Some(reason) = &planned.skipped {
            emit(AmdEvent::StepSkipped {
                index: planned.index,
                step: planned.step,
                reason: reason.clone(),
            });
            steps.push(skipped_outcome(planned, reason.clone()));
            continue;
        }
        ran += 1;
        emit(AmdEvent::StepStarted {
            index: planned.index,
            position: ran,
            total,
            step: planned.step,
            rerun: planned.rerun,
        });
        let started = Instant::now();
        let sub = step_config(config, planned.step)?;
        let fit_before = current_fit.clone();
        let mut output =
            match runner.run_step(planned.index, planned.step, &planned.dir, &sub, &current) {
                Ok(output) => output,
                // A step that fails is reported as a failed step, not as a
                // dead pipeline: the steps before it produced a model, and
                // throwing their work away to re-report one error is the
                // worst of both.
                Err(error) => {
                    notes.push(format!(
                        "step {} ({}) failed and was skipped over: {error}",
                        planned.index,
                        planned.step.label()
                    ));
                    emit(AmdEvent::StepFailed {
                        index: planned.index,
                        step: planned.step,
                        error: error.clone(),
                    });
                    let mut outcome = skipped_outcome(planned, String::new());
                    outcome.skipped = None;
                    outcome.failed = Some(error);
                    outcome.ofv_before = fit_before.as_ref().map(|f| f.ofv);
                    outcome.seconds = started.elapsed().as_secs_f64();
                    steps.push(outcome);
                    continue;
                }
            };
        let candidates = output.rows.len();
        fill_d_ofv(&mut output.rows);
        rows.append(&mut output.rows);
        // The step's own criterion, evaluated on the model it started from —
        // the tools report their candidates against their own base, and the
        // pipeline needs the comparison against what it handed them.
        let value_before = fit_before.as_ref().map(|f| output.criterion.of(f));
        let mut outcome = StepOutcome {
            index: planned.index,
            step: planned.step,
            rerun: planned.rerun,
            dir: planned.dir.clone(),
            skipped: None,
            failed: None,
            criterion: output.criterion.label(),
            value_before,
            value_after: output.value,
            ofv_before: fit_before.as_ref().map(|f| f.ofv),
            ofv_after: output.fit.as_ref().map(|f| f.ofv),
            selected: output.selected.clone(),
            candidates,
            notes: output.notes.clone(),
            seconds: 0.0,
        };
        emit(AmdEvent::StepFinished {
            index: planned.index,
            step: planned.step,
            criterion: output.criterion.label(),
            before: value_before,
            after: output.value,
            selected: output.selected.clone(),
        });
        current = output.model;
        if output.fit.is_some() {
            current_fit = output.fit;
        }
        seed(&mut current, current_fit.as_ref())?;

        if output.cancelled {
            outcome.seconds = started.elapsed().as_secs_f64();
            steps.push(outcome);
            cancelled = true;
            break;
        }

        if options.retries == Retries::AllFinal {
            let dir_name = format!("{}-retries", planned.dir);
            let outcome = retries_pass(
                runner,
                config,
                Some(planned.index),
                &dir_name,
                &mut current,
                &mut current_fit,
                &mut rows,
                &mut notes,
                progress,
            );
            note_retries_failure(&mut notes, &dir_name, outcome);
        }
        outcome.seconds = started.elapsed().as_secs_f64();
        steps.push(outcome);
    }

    if !cancelled && options.retries == Retries::Final {
        let outcome = retries_pass(
            runner,
            config,
            None,
            "retries",
            &mut current,
            &mut current_fit,
            &mut rows,
            &mut notes,
            progress,
        );
        note_retries_failure(&mut notes, "retries", outcome);
    }

    Ok(AmdResult {
        options: options.clone(),
        input_model: input.clone(),
        input_fit,
        steps,
        rows,
        final_model: current,
        final_fit: current_fit,
        notes,
        cancelled,
    })
}

/// A failed retries pass costs the pass, not the model it was refining — the
/// selected model and its fit are already in hand.
fn note_retries_failure(notes: &mut Vec<String>, dir_name: &str, outcome: Result<(), String>) {
    if let Err(error) = outcome {
        notes.push(format!(
            "the retries pass on `{dir_name}` could not be run: {error}; the selected model was \
             kept"
        ));
    }
}

/// Why the pipeline relaxes one gate, said once in every run's notes.
///
/// Every step refits the model it is handed as its own input, and that model
/// was seeded from the previous step's estimates — so it starts **at its own
/// optimum** and start 0 does not move. `reject_init_stall` (#751) reads that
/// as a stalled fit, and the tools refuse to search from an input that fails
/// the gate. Measured on `examples/amd_start.ferxsearch`: the residual and
/// covariate steps both came back `failed` with "no free parameter moved more
/// than 1% of its initial value", on a model that had just converged.
///
/// The gate's question — *did this fit ever leave its starting values?* — has
/// no meaning when the starting values are the answer, so it is switched off
/// for the steps. It still applies to the pipeline's own **start fit**, which
/// begins at the file's initial estimates and is exactly what #751 is about,
/// and every other gate (convergence, covariance, condition number, parameter
/// correlation, boundary estimates) applies throughout.
pub(crate) const INIT_STALL_NOTE: &str =
    "[strictness] reject_init_stall is off inside every step: the model a step is handed starts \
     at its own optimum, where \"did not leave the initial estimates\" (#751) is the expected \
     outcome rather than a failed fit. It still applies to the pipeline\'s own start fit, and \
     every other gate applies throughout";

fn skipped_outcome(planned: &PlannedStep, reason: String) -> StepOutcome {
    StepOutcome {
        index: planned.index,
        step: planned.step,
        rerun: planned.rerun,
        dir: planned.dir.clone(),
        skipped: Some(reason),
        failed: None,
        criterion: "",
        value_before: None,
        value_after: None,
        ofv_before: None,
        ofv_after: None,
        selected: Vec::new(),
        candidates: 0,
        notes: Vec::new(),
        seconds: 0.0,
    }
}

/// Carry `fit`'s estimates into `model` as the next step's starting values.
fn seed(model: &mut ModelText, fit: Option<&FitResult>) -> Result<(), String> {
    match fit {
        Some(fit) => crate::search::seed::seed_from(model, fit),
        None => Ok(()),
    }
}

/// The perturbed-restart pass on the model in hand: refit it with
/// `[run] retries + 1` starts and keep the better fit.
///
/// The comparison is on the OFV, and that is sound here where it would not be
/// between two candidates: this is the *same model text*, so the two fits have
/// the same parameter count and the lower OFV is the better optimum. A pass
/// that fails the gate, or that lands higher, leaves the model untouched and
/// says so in the notes.
#[allow(clippy::too_many_arguments)]
fn retries_pass(
    runner: &dyn StepRunner,
    config: &SearchConfig,
    index: Option<usize>,
    dir_name: &str,
    current: &mut ModelText,
    current_fit: &mut Option<FitResult>,
    rows: &mut Vec<CandidateRow>,
    notes: &mut Vec<String>,
    progress: Option<ProgressFn<'_>>,
) -> Result<(), String> {
    let starts = config.run.retries + 1;
    if starts < 2 {
        notes.push(
            "the retries pass was not run: `[run] retries = 0` asks for a single start, which \
             would refit the selected model at its own estimates"
                .into(),
        );
        return Ok(());
    }
    let emit = |event: AmdEvent| {
        if let Some(f) = progress {
            f(event);
        }
    };
    emit(AmdEvent::RetriesStarted { index, starts });
    let output = runner.fit_one(
        index.unwrap_or(0),
        "retries",
        dir_name,
        config,
        current,
        starts,
    )?;
    let before = current_fit.as_ref().map(|f| f.ofv);
    let after = output.fit.as_ref().map(|f| f.ofv);
    let improved = output.passed
        && (matches!((before, after), (Some(b), Some(a)) if a < b)
            || (before.is_none() && after.is_some()));
    if !output.passed {
        notes.push(format!(
            "the retries pass on `{dir_name}` did not pass the strictness gate; the selected \
             model was kept"
        ));
    }
    rows.extend(output.rows);
    notes.extend(output.notes);
    if improved {
        *current = output.model;
        *current_fit = output.fit;
        seed(current, current_fit.as_ref())?;
    } else if let (true, Some(b), Some(a)) = (output.passed, before, after) {
        notes.push(format!(
            "the retries pass on `{dir_name}` did not improve the fit (OFV {a:.3} against \
             {b:.3}); the selected model was kept"
        ));
    }
    emit(AmdEvent::RetriesFinished {
        improved,
        ofv: if improved { after } else { before },
    });
    Ok(())
}

/// The config one step is handed: this file, with the space narrowed to the
/// statements the step's tool accepts, and `[rank]` narrowed the same way.
///
/// The rank narrowing is not a convenience. `ruvsearch` and `covsearch` select
/// by a **likelihood-ratio test** at their own p-values, and both *refuse* a
/// `[rank] type` other than `ofv` and any `[rank] cutoff` rather than ignore
/// one — correctly, since a BIC ranking does not apply to what they do. But a
/// pipeline's one `[rank] type = "bic"` is exactly what the steps that *do*
/// rank need, and it is Pharmpy's default. So the criterion goes to the steps
/// that rank on it and the two LRT steps are left at their own defaults, which
/// is the same narrowing the space gets and for the same reason.
pub(crate) fn step_config(config: &SearchConfig, step: Step) -> Result<SearchConfig, String> {
    let mfl = if step == Step::Residual {
        // ruvsearch's candidates are the error forms, not MFL features.
        Mfl {
            statements: Vec::new(),
        }
    } else {
        space::subspace(&config.mfl, step)?
    };
    let mut sub = config.clone();
    sub.mfl_source = mfl.render();
    sub.mfl = mfl;
    if matches!(step, Step::Residual | Step::Covariates) {
        sub.rank.kind = None;
        sub.rank.cutoff = None;
    }
    // See `INIT_STALL_NOTE`: the model every step is handed starts *at* its own
    // optimum, so #751's gate cannot say anything about it.
    sub.strictness.reject_init_stall = Some(false);
    Ok(sub)
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
