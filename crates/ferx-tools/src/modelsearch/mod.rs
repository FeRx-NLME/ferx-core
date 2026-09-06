//! Structural PK model search — Pharmpy `modelsearch` (#1181).
//!
//! The second search tool of the #1175 epic. Where covsearch (#1180) adds
//! and removes `[covariate_model]` lines, this one swaps the `pk` template:
//! absorption route, peripheral compartments, transit compartments and lag
//! time, each a coordinate of a [`Structure`], each move a
//! [`ModelEdit::SetStructural`] that also declares the parameters the new
//! template needs and prunes the ones it no longer reads. Everything else is
//! the shared machinery: [`ModelText`] for the edit, the runner (through the
//! `StepFitter` seam) to fit a layer's candidates in parallel with dedup,
//! journal and strictness, and the `.ferxsearch` file ([`SearchConfig`]) for
//! the space, the criterion and the gate.
//!
//! # The algorithms
//!
//! All three are Pharmpy's (`tools/modelsearch/algorithms.py`), and the
//! candidate enumeration is checked against Pharmpy's own in
//! `pharmpy_anchor.rs`.
//!
//! * **`exhaustive`** — every combination of at most one feature per
//!   category, all derived from the base model in one layer.
//! * **`exhaustive_stepwise`** — one feature at a time, in every order: layer
//!   1 applies each allowed feature to the base, layer *k* applies each
//!   feature still allowed to every model of layer *k−1*, each child seeded
//!   from its parent's estimates. Two orders of the same features are two
//!   candidates, because they start from different parents.
//! * **`reduced_stepwise`** — the default, as in Pharmpy. As above, but
//!   before a layer is extended the models that carry the same feature set
//!   are collapsed to the one with the lowest OFV, and only it continues.
//!   Every model is still ranked at the end.
//!
//! Which feature may follow which is Pharmpy's `_is_allowed`
//! ([`structure::allowed`]): a category is moved along once per path, the
//! peripheral count starts at the smallest in the space, and the pairs
//! Pharmpy declares meaningless — a lag time or transit chain on a bolus, a
//! lag time with a transit chain, first-order absorption with a one-transit
//! chain — are never combined. ferx applies those pairs to the exhaustive
//! combinations too, which Pharmpy does not, and to the candidate's whole
//! structure rather than only to the features a path applied (see
//! [`Structure::unbuildable`]).
//!
//! # The base model
//!
//! The base is the input model when its structure lies in the space. When it
//! does not — a one-compartment base in a `PERIPHERALS(1..2)` space — the
//! input is fitted first and a **base** is derived from it with Pharmpy's
//! least number of transformations ([`structure::onto_space`]), fitted, and
//! used as the root; both appear in the table. The base's own features are
//! removed from the candidate moves, so a feature the base already carries
//! is never a step.
//!
//! # Ranking
//!
//! Every fitted model — input, base and candidates — is ranked on `[rank]
//! type` (the mixed BIC by default) among those that pass the strictness
//! gate. With a `cutoff`, a candidate is selectable only when it beats the
//! base by at least that much; the base wins otherwise. The **input is
//! ranked but never selected**: it has a row of its own only when a base had
//! to be derived, which is exactly when its structure lies outside the
//! declared space, so selecting it would write a final model the MFL
//! excluded. (Pharmpy ranks base and candidates only; the input row is
//! ferx's, and it is there to be read, not chosen.) A candidate that
//! fails the gate, does not compile or does not fit is in the table with its
//! reason and no rank. The per-candidate wall-clock time is in the table
//! too: the analytic templates cost about the same, but the column is what
//! shows when one does not.
//!
//! # Deviations, stated
//!
//! * The `ELIMINATION` gap. `ZO`, `MM` and `MIX-FO-MM` have no analytic
//!   template and are refused by the coverage check, so an elimination
//!   search is not offered rather than offered as `[odes]` candidates whose
//!   runtime and multistart needs differ from their siblings by an order of
//!   magnitude. `docs/tools/modelsearch.qmd` records the decision.
//! * The reduced-stepwise collapse prefers a model that passed the gate:
//!   Pharmpy takes the lowest OFV among the group's models that have any
//!   result. ferx takes the lowest OFV among those that passed, and only
//!   when none did the lowest OFV of the rest.
//! * `TRANSITS(n)` is a `FIX`ed θ, not a literal, so the count keeps a name
//!   the estimates file and the ODE twin can see.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use ferx_core::edit::{ModelEdit, ModelText};
use ferx_core::{CancelFlag, FitResult};
use serde::Deserialize;

use crate::search::fitter::{RunnerFitter, StepFitter};
use crate::search::seed::seed_from;
use crate::search::{
    BaseModel, Candidate, CandidateError, CandidateResult, Criterion, ModelContext, PkTemplate,
    RankType, RunReport, SearchConfig,
};

pub mod structure;

mod report;

pub use report::{
    final_model_path, models_dir, models_path, render_summary, write_report, MODEL_COLUMNS,
};
pub use structure::{
    Absorption, Defaults, FeatureKey, IivStrategy, Structure, Template, TransitCount,
};

/// `[modelsearch] algorithm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Algorithm {
    Exhaustive,
    ExhaustiveStepwise,
    #[default]
    ReducedStepwise,
}

impl Algorithm {
    pub fn label(&self) -> &'static str {
        match self {
            Algorithm::Exhaustive => "exhaustive",
            Algorithm::ExhaustiveStepwise => "exhaustive_stepwise",
            Algorithm::ReducedStepwise => "reduced_stepwise",
        }
    }

    fn is_stepwise(&self) -> bool {
        !matches!(self, Algorithm::Exhaustive)
    }
}

/// The `[modelsearch]` section of a `.ferxsearch` file, plus the `[rank]`
/// keys the tool reads. Every key has Pharmpy's default.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelsearchOptions {
    pub algorithm: Algorithm,
    /// How η is given to the parameters a candidate introduces.
    pub iiv_strategy: IivStrategy,
    /// `[rank] type`; the mixed BIC when the file does not say.
    pub rank: RankType,
    /// `[rank] cutoff`: the improvement over the base a candidate must show
    /// to be selected. `None` — Pharmpy's default — selects the best model
    /// on the criterion alone.
    pub cutoff: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Section {
    #[serde(default)]
    algorithm: Algorithm,
    #[serde(default)]
    iiv_strategy: IivStrategy,
}

impl ModelsearchOptions {
    /// Read `[modelsearch]` and `[rank]` off a loaded file.
    pub fn from_config(config: &SearchConfig) -> Result<Self, String> {
        config.require_space(
            "modelsearch",
            "ABSORPTION / PERIPHERALS / TRANSITS / LAGTIME statements",
        )?;
        let section = match config.tools.get("modelsearch") {
            Some(table) => table
                .clone()
                .try_into::<Section>()
                .map_err(|e| format!("[modelsearch]: {e}"))?,
            None => Section::default(),
        };
        let options = ModelsearchOptions {
            algorithm: section.algorithm,
            iiv_strategy: section.iiv_strategy,
            rank: config.rank.kind_or_default(),
            cutoff: config.rank.cutoff,
        };
        options.validate()?;
        Ok(options)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.iiv_strategy == IivStrategy::Fullblock {
            return Err(
                "[modelsearch] iiv_strategy = \"fullblock\": a block over the new and existing \
                 η is a variability-structure move, which is iivsearch's (#1183); use \
                 add_diagonal, absorption_delay or no_add"
                    .into(),
            );
        }
        if let Some(c) = self.cutoff {
            if !(c.is_finite() && c >= 0.0) {
                return Err(format!(
                    "[rank] cutoff = {c}: must be a finite, non-negative improvement on the \
                     criterion's own scale"
                ));
            }
        }
        // A file asking for an unimplemented criterion fails here, before
        // any data is read.
        self.rank.criterion()?;
        Ok(())
    }

    /// Refuse a file whose space is not a structural one — a covariate or
    /// variability file meant for another tool — before any data is read.
    /// The same check `Space::from_config` makes, on the unresolved space.
    pub fn check_space(config: &SearchConfig) -> Result<(), String> {
        structure::space_features(&config.mfl).map(|_| ())
    }

    /// The runner criterion this ranks on.
    pub fn criterion(&self) -> Criterion {
        self.rank
            .criterion()
            .expect("validated: the rank type has a criterion")
    }
}

/// One fitted model of the search, as the table reports it.
#[derive(Debug, Clone)]
pub struct ModelRow {
    /// `input`, `base`, or `run{n}` in generation order — Pharmpy's
    /// `modelsearch_run{n}`.
    pub id: String,
    pub parent: Option<String>,
    /// `0` for the input and the base; the stepwise layer otherwise
    /// (`1` for every exhaustive candidate).
    pub layer: usize,
    pub structure: Structure,
    /// The features applied on the path from the base, in order.
    pub path: Vec<FeatureKey>,
    pub ofv: Option<f64>,
    pub n_parameters: Option<usize>,
    /// The ranking criterion; `NaN` without a fit.
    pub criterion: f64,
    /// `criterion − base criterion`, when both exist. Negative is better.
    pub d_criterion: Option<f64>,
    /// 1 for the best eligible model; `None` for a model that is not
    /// eligible.
    pub rank: Option<usize>,
    pub converged: Option<bool>,
    /// Passed the strictness gate and has a fit.
    pub passed: bool,
    pub failures: Vec<String>,
    pub error: Option<CandidateError>,
    pub seconds: f64,
    /// The model the search selected.
    pub selected: bool,
    /// In `reduced_stepwise`, the model of its feature-set group that was
    /// extended in the next layer. `true` for every model otherwise.
    pub continued: bool,
    /// The outcome came from the journal of an earlier run, not a fit.
    pub reused: bool,
}

impl ModelRow {
    /// Eligible for ranking: fitted, passed the gate, finite criterion.
    pub fn eligible(&self) -> bool {
        self.error.is_none() && self.passed && self.criterion.is_finite()
    }
}

/// What a search reports as it runs, for a CLI progress line.
#[derive(Debug, Clone)]
pub enum ModelsearchEvent {
    /// The input model is being fitted (only when a base has to be derived
    /// from it).
    InputStarted,
    BaseStarted,
    BaseFinished {
        ofv: f64,
        criterion: f64,
    },
    LayerStarted {
        layer: usize,
        candidates: usize,
    },
    /// `best` is the layer's lowest-criterion eligible candidate.
    LayerFinished {
        layer: usize,
        best: Option<(String, f64)>,
    },
}

/// A progress callback.
pub type ProgressFn<'a> = &'a (dyn Fn(ModelsearchEvent) + Send + Sync);

/// The outcome of a search.
#[derive(Debug, Clone)]
pub struct ModelsearchResult {
    pub options: ModelsearchOptions,
    pub criterion: Criterion,
    /// The model the search started from — the file's, unedited.
    pub input_model: ModelText,
    /// The root of the search: the input, or the base derived from it.
    pub base_id: String,
    pub base_structure: Structure,
    /// Every fitted model, in generation order.
    pub rows: Vec<ModelRow>,
    pub final_id: String,
    pub final_model: ModelText,
    pub final_fit: Option<FitResult>,
    pub final_criterion: f64,
    /// Every fitted model's text, by id — the candidates a user may want to
    /// read or refit, which the table alone cannot give back.
    pub models: BTreeMap<String, ModelText>,
    /// Things a user should read once: combinations no template can
    /// express, empty symbols, journal warnings.
    pub notes: Vec<String>,
    /// The search stopped on a cancel flag; `rows` is partial.
    pub cancelled: bool,
}

impl ModelsearchResult {
    pub fn row(&self, id: &str) -> Option<&ModelRow> {
        self.rows.iter().find(|r| r.id == id)
    }

    /// The rows of one layer, in generation order.
    pub fn layer_rows(&self, layer: usize) -> impl Iterator<Item = &ModelRow> {
        self.rows.iter().filter(move |r| r.layer == layer)
    }

    pub fn n_layers(&self) -> usize {
        self.rows.iter().map(|r| r.layer).max().unwrap_or(0)
    }

    /// The eligible rows, best first.
    pub fn ranked(&self) -> Vec<&ModelRow> {
        let mut rows: Vec<&ModelRow> = self.rows.iter().filter(|r| r.rank.is_some()).collect();
        rows.sort_by_key(|r| r.rank);
        rows
    }
}

/// Everything a [`run_modelsearch`] call takes beyond the file.
#[derive(Default)]
pub struct ModelsearchRun<'a> {
    /// Where the per-layer journals, `models.csv` and `final.ferx` go.
    /// `None` keeps everything in memory: no resume, no files.
    pub dir: Option<PathBuf>,
    /// Overrides `[run] threads`.
    pub threads: Option<usize>,
    pub cancel: Option<CancelFlag>,
    pub progress: Option<ProgressFn<'a>>,
}

/// Run a structural search from a loaded `.ferxsearch` file and its base
/// model. Writes `models.csv` and `final.ferx` into `run.dir` when given.
pub fn run_modelsearch(
    config: &SearchConfig,
    base: &BaseModel,
    run: ModelsearchRun<'_>,
) -> Result<ModelsearchResult, String> {
    let options = ModelsearchOptions::from_config(config)?;
    let mut run_options = config.run_options();
    run_options.criterion = options.criterion();
    let fitter = RunnerFitter {
        threads: run.threads.or(config.run.threads).unwrap_or(0),
        dir: run.dir.clone(),
        cancel: run.cancel.clone(),
        data: &base.prepared.population,
        options: run_options,
    };
    let space = Space::from_config(config, base)?;
    let result = search(&fitter, space, &options, run.progress)?;
    if let Some(dir) = &run.dir {
        write_report(dir, &result)?;
    }
    Ok(result)
}

/// The search space after resolution against the base model.
#[derive(Debug)]
pub(crate) struct Space {
    /// The file's model.
    pub input_model: ModelText,
    pub input_structure: Structure,
    /// The moves that take the input onto the space; empty when it is on it.
    pub onto: Vec<FeatureKey>,
    /// The candidate moves: the space's features minus the base's own.
    pub funcs: Vec<FeatureKey>,
    pub defaults: Defaults,
    pub notes: Vec<String>,
}

impl Space {
    fn from_config(config: &SearchConfig, base: &BaseModel) -> Result<Space, String> {
        let resolved = config.resolve_space(base)?;
        let features = structure::space_features(&resolved.mfl)?;
        let ctx =
            ModelContext::from_model(&base.prepared.parsed, &base.text, &base.prepared.population)?;
        let template = ctx.template.as_ref().ok_or_else(|| {
            "modelsearch: the base model has no `pk NAME(...)` line; an `ode(...)` or \
             algebraic model has no template to swap, so there is nothing structural to search"
                .to_string()
        })?;
        let input_structure = Structure::from_model(template, Some(&base.text))?;
        let defaults = Defaults::new(
            ctx.parameters.clone(),
            base.prepared.init_params.theta_names.clone(),
            base.prepared.init_params.theta.clone(),
            base.prepared.parsed.model.eta_names.clone(),
            &base.prepared.population,
        );
        Self::build(
            base.text.clone(),
            input_structure,
            &features,
            defaults,
            resolved.notes.clone(),
        )
    }

    /// The space from its parts — the seam the unit tests use.
    pub(crate) fn build(
        input_model: ModelText,
        input_structure: Structure,
        features: &[FeatureKey],
        defaults: Defaults,
        mut notes: Vec<String>,
    ) -> Result<Space, String> {
        let onto = structure::onto_space(&input_structure, features);
        let mut base_structure = input_structure;
        for key in &onto {
            base_structure = base_structure.apply(key);
        }
        if let Some(why) = base_structure.unbuildable() {
            // Two different failures, and blaming the wrong one is unhelpful
            // advice: with `onto` empty no derivation happened at all — the
            // input's *own* structure has no template (a lag time on a bolus
            // dose, say), so listing its values in the space cannot fix it.
            return Err(if onto.is_empty() {
                format!(
                    "modelsearch: the input model lies on the space, but its own structure \
                     has no template to search from: {why}. Change the input model, or name \
                     a space whose base is buildable"
                )
            } else {
                format!(
                    "modelsearch: the input model lies outside the space, and the base \
                     Pharmpy's least number of transformations would derive from it ({}) has \
                     no template: {why}. List the input's own value in each category the \
                     space names (e.g. `LAGTIME([OFF,ON])`, `TRANSITS(0, NODEPOT)`) so the \
                     input can be the base",
                    onto.iter()
                        .map(|k| k.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            });
        }
        if !onto.is_empty() {
            notes.push(format!(
                "the input model lies outside the space; the base is the input with {} applied",
                onto.iter()
                    .map(|k| k.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        // Pharmpy's `filter_mfl_statements`: a feature the base has is no move.
        let has = base_structure.features();
        let funcs: Vec<FeatureKey> = features
            .iter()
            .filter(|k| !has.contains(k))
            .copied()
            .collect();
        Ok(Space {
            input_model,
            input_structure,
            onto,
            funcs,
            defaults,
            notes,
        })
    }
}

/// A fitted model the search may extend.
#[derive(Debug, Clone)]
struct Node {
    id: String,
    model: ModelText,
    fit: Option<FitResult>,
    ofv: Option<f64>,
    structure: Structure,
    path: Vec<FeatureKey>,
    /// The node's own `pk` line and `[individual_parameters]`, read off its
    /// text so a child's bindings follow what the parent actually declares.
    template: PkTemplate,
    lines: Vec<String>,
    /// Eligible for the reduced-stepwise collapse's first preference.
    passed: bool,
    /// Produced no fit at all (`CandidateResult::error`): its children
    /// start from its own initial estimates, and the notes say so.
    failed: bool,
}

impl Node {
    fn from_model(
        id: &str,
        model: ModelText,
        structure: Structure,
        path: Vec<FeatureKey>,
        result: Option<(&CandidateResult, &RunReport)>,
    ) -> Result<Node, String> {
        let template = model
            .block_lines("structural_model")
            .iter()
            .find_map(|l| PkTemplate::parse_line(l))
            .transpose()?
            .ok_or_else(|| format!("{id}: the model has no `pk NAME(...)` line"))?;
        let lines = model.block_lines("individual_parameters");
        let fit = match result {
            Some((r, report)) => resolve_fit(r, report)?,
            None => None,
        };
        Ok(Node {
            id: id.to_string(),
            fit,
            ofv: result.and_then(|(r, _)| r.ofv),
            passed: result.is_some_and(|(r, _)| r.eligible()),
            failed: result.is_some_and(|(r, _)| r.error.is_some()),
            model,
            structure,
            path,
            template,
            lines,
        })
    }
}

/// The fit a result stands for, so a child can be seeded from it.
///
/// A result legitimately carries no fit in three cases, and they are not
/// the same statement: a **failed** candidate has nothing to seed from and
/// says so through `error`; a **duplicate** (`duplicate_of`) was scored by
/// its representative, whose fit is the one to seed from; and a **resumed**
/// row whose `fits/<hash>.json` is missing or unreadable looks eligible
/// while having nothing to hand its children — silently starting them from
/// the file's initials would contradict the documented seeding, so that one
/// is an error naming the fix.
///
/// `error` is tested **before** `duplicate_of`, and the order is
/// load-bearing: `Runner` builds a duplicate by cloning the
/// representative's verdict, criterion *and error* (`search/runner.rs`), so
/// a duplicate of a candidate that failed to compile or fit carries both
/// fields. Taking the duplicate branch first turns that into an `Err` that
/// propagates out of `search()` and kills the whole run, where the right
/// outcome is the failed row the table already has.
fn resolve_fit(result: &CandidateResult, report: &RunReport) -> Result<Option<FitResult>, String> {
    if let Some(fit) = &result.fit {
        return Ok(Some(fit.clone()));
    }
    if result.error.is_some() {
        return Ok(None);
    }
    if let Some(rep) = &result.duplicate_of {
        return report
            .results
            .iter()
            .find(|r| r.id == *rep)
            .and_then(|r| r.fit.clone())
            .map(Some)
            .ok_or_else(|| {
                format!(
                    "{}: a duplicate of {rep}, whose fit is not available to seed from",
                    result.id
                )
            });
    }
    Err(format!(
        "{}: the fit is not in the journal cache (a resumed row whose `fits/<hash>.json` is \
         missing or unreadable), so nothing can be seeded from it; refit without `resume`",
        result.id
    ))
}

/// The candidate that moves `parent` by `keys`, seeded from the parent's
/// estimates.
fn derive(
    id: &str,
    parent: &Node,
    keys: &[FeatureKey],
    space: &Space,
    options: &ModelsearchOptions,
) -> Result<(Candidate, Structure, Vec<FeatureKey>), String> {
    let mut target = parent.structure;
    for k in keys {
        target = target.apply(k);
    }
    let mut model = parent.model.clone();
    if let Some(fit) = &parent.fit {
        seed_from(&mut model, fit)?;
    }
    // Declarations and inits come from the parent's own (seeded) text, not
    // the input's: a name an earlier step pruned is not present, a θ an
    // earlier step added is taken, and `Q = CL` is the parent's converged
    // clearance — Pharmpy updates the inits first and adds the compartment
    // second.
    let defaults = Defaults::of_text(&model, space.defaults.t_first);
    let spec = structure::structural_spec(
        &target,
        &parent.structure,
        &parent.template,
        &parent.lines,
        &defaults,
        options.iiv_strategy,
    )?;
    model
        .apply(ModelEdit::SetStructural(spec))
        .map_err(|e| format!("{id}: {e}"))?;
    let mut path = parent.path.clone();
    path.extend_from_slice(keys);
    let candidate = Candidate::new(id, model)
        .parent(parent.id.clone())
        .features(target.feature_vector());
    Ok((candidate, target, path))
}

/// The id of the input model's row, present only when a base had to be
/// derived from it.
const INPUT_ID: &str = "input";

/// The search proper, over an injected fitter.
pub(crate) fn search(
    fitter: &dyn StepFitter,
    space: Space,
    options: &ModelsearchOptions,
    progress: Option<ProgressFn<'_>>,
) -> Result<ModelsearchResult, String> {
    let emit = |event: ModelsearchEvent| {
        if let Some(p) = progress {
            p(event);
        }
    };
    let criterion = options.criterion();
    let mut notes = space.notes.clone();
    let mut rows: Vec<ModelRow> = Vec::new();
    // Every model's text and fit, by id, so the winner can be handed back
    // whatever layer it came from.
    let mut store: HashMap<String, (ModelText, Option<FitResult>)> = HashMap::new();
    let mut cancelled = false;

    // ── the input, and the base when it has to be derived ───────────────
    let derive_base = !space.onto.is_empty();
    let root_id = if derive_base { INPUT_ID } else { "base" };
    emit(if derive_base {
        ModelsearchEvent::InputStarted
    } else {
        ModelsearchEvent::BaseStarted
    });
    let candidate = Candidate::new(root_id, space.input_model.clone())
        .features(space.input_structure.feature_vector());
    let report = fitter.fit_step(root_id, std::slice::from_ref(&candidate))?;
    notes.extend(report.warnings.iter().cloned());
    let result = report
        .results
        .first()
        .ok_or("the base model was not fitted")?;
    if let Some(e) = &result.error {
        return Err(format!("the {root_id} model could not be fitted: {e}"));
    }
    rows.push(row_of(
        result,
        0,
        None,
        space.input_structure,
        vec![],
        criterion,
    ));
    let mut root = Node::from_model(
        root_id,
        space.input_model.clone(),
        space.input_structure,
        vec![],
        Some((result, &report)),
    )?;
    store.insert(root.id.clone(), (root.model.clone(), root.fit.clone()));
    cancelled |= report.cancelled;

    if !cancelled && derive_base {
        emit(ModelsearchEvent::BaseStarted);
        let (candidate, structure, path) = derive("base", &root, &space.onto, &space, options)?;
        let report = fitter.fit_step("base", std::slice::from_ref(&candidate))?;
        notes.extend(report.warnings.iter().cloned());
        let result = report
            .results
            .first()
            .ok_or("the base model was not fitted")?;
        if let Some(e) = &result.error {
            return Err(format!("the base model could not be fitted: {e}"));
        }
        rows.push(row_of(
            result,
            0,
            Some(INPUT_ID),
            structure,
            path,
            criterion,
        ));
        // The base is the root: its path is empty, since the candidate
        // moves are counted from it (Pharmpy filters the space by the base).
        root = Node::from_model(
            "base",
            candidate.model.clone(),
            structure,
            vec![],
            Some((result, &report)),
        )?;
        store.insert(root.id.clone(), (root.model.clone(), root.fit.clone()));
        cancelled |= report.cancelled;
    }
    let base_id = root.id.clone();
    let base_structure = root.structure;
    {
        let base_row = rows.last().expect("the root row");
        emit(ModelsearchEvent::BaseFinished {
            ofv: base_row.ofv.unwrap_or(f64::NAN),
            criterion: base_row.criterion,
        });
        if !base_row.passed {
            notes.push(format!(
                "the base model fails the strictness gate ({}); candidates are ranked among \
                 themselves and the cutoff cannot be applied",
                base_row.failures.join("; ")
            ));
        }
    }

    // ── the candidates ───────────────────────────────────────────────────
    let mut next_id = 0usize;
    let mut new_id = || {
        next_id += 1;
        format!("run{next_id}")
    };
    if !cancelled && options.algorithm.is_stepwise() {
        let mut leaves: Vec<Node> = vec![root.clone()];
        let mut layer = 0usize;
        loop {
            layer += 1;
            // Pharmpy's `_get_possible_actions`: leaves in creation order,
            // features in dictionary order.
            let mut planned: Vec<(usize, FeatureKey)> = Vec::new();
            for (i, leaf) in leaves.iter().enumerate() {
                if leaf.failed {
                    push_note(
                        &mut notes,
                        format!(
                            "{} produced no fit; its children start from its own initial \
                             estimates rather than a parent's",
                            leaf.id
                        ),
                    );
                }
                for key in &space.funcs {
                    if !structure::allowed(key, &leaf.path, &space.funcs, &leaf.structure) {
                        continue;
                    }
                    if let Some(why) = leaf.structure.apply(key).unbuildable() {
                        push_note(
                            &mut notes,
                            format!(
                                "not generated: {key} after {}: {why}",
                                path_label(&leaf.path)
                            ),
                        );
                        continue;
                    }
                    planned.push((i, *key));
                }
            }
            if planned.is_empty() {
                break;
            }
            emit(ModelsearchEvent::LayerStarted {
                layer,
                candidates: planned.len(),
            });
            let mut candidates = Vec::with_capacity(planned.len());
            let mut meta = Vec::with_capacity(planned.len());
            for (i, key) in &planned {
                let id = new_id();
                let (candidate, structure, path) =
                    derive(&id, &leaves[*i], std::slice::from_ref(key), &space, options)?;
                candidates.push(candidate);
                meta.push((leaves[*i].id.clone(), structure, path));
            }
            let report = fitter.fit_step(&format!("layer-{layer}"), &candidates)?;
            notes.extend(report.warnings.iter().cloned());
            let mut children: Vec<Node> = Vec::new();
            let mut layer_rows: Vec<ModelRow> = Vec::new();
            for (candidate, (parent, structure, path)) in candidates.iter().zip(meta) {
                let Some(result) = report.results.iter().find(|r| r.id == candidate.id) else {
                    continue; // cancelled before this candidate was reached
                };
                layer_rows.push(row_of(
                    result,
                    layer,
                    Some(&parent),
                    structure,
                    path.clone(),
                    criterion,
                ));
                let node = Node::from_model(
                    &candidate.id,
                    candidate.model.clone(),
                    structure,
                    path,
                    Some((result, &report)),
                )?;
                store.insert(node.id.clone(), (node.model.clone(), node.fit.clone()));
                children.push(node);
            }
            let best = layer_rows
                .iter()
                .filter(|r| r.eligible())
                .min_by(|a, b| a.criterion.total_cmp(&b.criterion))
                .map(|r| (r.id.clone(), r.criterion));
            emit(ModelsearchEvent::LayerFinished { layer, best });
            if report.cancelled {
                rows.extend(layer_rows);
                cancelled = true;
                break;
            }
            if options.algorithm == Algorithm::ReducedStepwise {
                let kept = collapse(&children, &space.funcs);
                for row in &mut layer_rows {
                    row.continued = kept.contains(&row.id);
                }
                children.retain(|n| kept.contains(&n.id));
            }
            rows.extend(layer_rows);
            leaves = children;
        }
    } else if !cancelled {
        let mut candidates = Vec::new();
        let mut meta = Vec::new();
        for combo in combinations(&space.funcs, &root.structure) {
            let mut target = root.structure;
            for k in &combo {
                target = target.apply(k);
            }
            if let Some(why) = target.unbuildable() {
                push_note(
                    &mut notes,
                    format!("not generated: {}: {why}", path_label(&combo)),
                );
                continue;
            }
            let id = new_id();
            let (candidate, structure, path) = derive(&id, &root, &combo, &space, options)?;
            candidates.push(candidate);
            meta.push((structure, path));
        }
        if !candidates.is_empty() {
            emit(ModelsearchEvent::LayerStarted {
                layer: 1,
                candidates: candidates.len(),
            });
            let report = fitter.fit_step("candidates", &candidates)?;
            notes.extend(report.warnings.iter().cloned());
            for (candidate, (structure, path)) in candidates.iter().zip(meta) {
                let Some(result) = report.results.iter().find(|r| r.id == candidate.id) else {
                    continue;
                };
                rows.push(row_of(
                    result,
                    1,
                    Some(&root.id),
                    structure,
                    path,
                    criterion,
                ));
                store.insert(
                    candidate.id.clone(),
                    (candidate.model.clone(), resolve_fit(result, &report)?),
                );
            }
            let best = rows
                .iter()
                .filter(|r| r.layer == 1 && r.eligible())
                .min_by(|a, b| a.criterion.total_cmp(&b.criterion))
                .map(|r| (r.id.clone(), r.criterion));
            emit(ModelsearchEvent::LayerFinished { layer: 1, best });
            cancelled |= report.cancelled;
        }
    }

    // ── ranking and selection ────────────────────────────────────────────
    let base_criterion = rows
        .iter()
        .find(|r| r.id == base_id)
        .filter(|r| r.eligible())
        .map(|r| r.criterion);
    for row in &mut rows {
        row.d_criterion = match (base_criterion, row.criterion.is_finite()) {
            (Some(b), true) => Some(row.criterion - b),
            _ => None,
        };
    }
    // Ties go to the earlier model, so a deterministic generation order
    // gives a deterministic winner (a stable sort keeps generation order).
    let mut order: Vec<usize> = (0..rows.len()).filter(|i| rows[*i].eligible()).collect();
    order.sort_by(|a, b| rows[*a].criterion.total_cmp(&rows[*b].criterion));
    for (rank, i) in order.iter().enumerate() {
        rows[*i].rank = Some(rank + 1);
    }
    let selectable = |r: &ModelRow| match (options.cutoff, base_criterion) {
        (Some(c), Some(b)) => r.id == base_id || b - r.criterion >= c,
        _ => true,
    };
    // The input is ranked but never selected. It only has a row of its own
    // when a base had to be derived, and that happens exactly when its
    // structure lies *outside* the declared space — so selecting it would
    // write a `final.ferx` the user's MFL excluded, and `[space]` would not
    // bind the result.
    let winner = order
        .iter()
        .copied()
        .find(|i| rows[*i].id != INPUT_ID && selectable(&rows[*i]))
        .or_else(|| rows.iter().position(|r| r.id == base_id))
        .expect("the base row exists");
    rows[winner].selected = true;
    let final_id = rows[winner].id.clone();
    let final_criterion = rows[winner].criterion;
    if !rows[winner].eligible() {
        notes.push("no model passed the strictness gate; the final model is the base".into());
    }
    let models: BTreeMap<String, ModelText> = store
        .iter()
        .map(|(id, (text, _))| (id.clone(), text.clone()))
        .collect();
    let (final_model, final_fit) = store
        .remove(&final_id)
        .expect("every fitted model is stored");
    Ok(ModelsearchResult {
        options: options.clone(),
        criterion,
        input_model: space.input_model,
        base_id,
        base_structure,
        rows,
        final_id,
        final_model,
        final_fit,
        final_criterion,
        models,
        notes,
        cancelled,
    })
}

fn push_note(notes: &mut Vec<String>, note: String) {
    if !notes.contains(&note) {
        notes.push(note);
    }
}

/// `PERIPHERALS(1) → LAGTIME(ON)`, or `the base` for an empty path.
fn path_label(path: &[FeatureKey]) -> String {
    if path.is_empty() {
        "the base".to_string()
    } else {
        path.iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
            .join(" → ")
    }
}

fn row_of(
    result: &CandidateResult,
    layer: usize,
    parent: Option<&str>,
    structure: Structure,
    path: Vec<FeatureKey>,
    criterion: Criterion,
) -> ModelRow {
    ModelRow {
        id: result.id.clone(),
        parent: parent.map(str::to_string),
        layer,
        structure,
        path,
        ofv: result.ofv,
        n_parameters: result.fit.as_ref().map(|f| f.n_parameters),
        // The runner scored the candidate on its own criterion; the search
        // reads it back through the same enum so the two cannot disagree.
        criterion: match &result.fit {
            Some(fit) => criterion.of(fit),
            None => result.criterion,
        },
        d_criterion: None,
        rank: None,
        converged: result.converged,
        passed: result.verdict.passed && result.error.is_none(),
        failures: result.verdict.failures.clone(),
        error: result.error.clone(),
        seconds: result.seconds,
        selected: false,
        continued: true,
        reused: result.reused,
    }
}

/// Pharmpy's `reduced_stepwise` collapse: among the nodes of one layer that
/// carry the same feature *set* (any order), keep the one to extend — but
/// only for groups every member of which still has a move, as Pharmpy adds
/// a collector node only there. Returns the ids that continue.
fn collapse(children: &[Node], funcs: &[FeatureKey]) -> Vec<String> {
    let has_action = |n: &Node| {
        funcs
            .iter()
            .any(|k| structure::allowed(k, &n.path, funcs, &n.structure))
    };
    let mut kept: Vec<String> = Vec::new();
    let mut grouped: Vec<bool> = vec![false; children.len()];
    for i in 0..children.len() {
        if grouped[i] {
            continue;
        }
        let set = feature_set(&children[i].path);
        let group: Vec<usize> = (i..children.len())
            .filter(|j| !grouped[*j] && feature_set(&children[*j].path) == set)
            .collect();
        for j in &group {
            grouped[*j] = true;
        }
        if group.len() == 1 || !group.iter().all(|j| has_action(&children[*j])) {
            kept.extend(group.iter().map(|j| children[*j].id.clone()));
            continue;
        }
        // Lowest OFV among the members that passed the gate, else lowest
        // OFV among those with any, else the first — see the module docs.
        let best = |eligible_only: bool| {
            group
                .iter()
                .copied()
                .filter(|j| !eligible_only || children[*j].passed)
                .filter(|j| children[*j].ofv.is_some_and(f64::is_finite))
                .min_by(|a, b| {
                    children[*a]
                        .ofv
                        .unwrap()
                        .total_cmp(&children[*b].ofv.unwrap())
                })
        };
        let chosen = best(true).or_else(|| best(false)).unwrap_or(group[0]);
        kept.push(children[chosen].id.clone());
    }
    kept
}

fn feature_set(path: &[FeatureKey]) -> Vec<(&'static str, String)> {
    let mut set: Vec<(&'static str, String)> = path.iter().map(|k| k.sort_key()).collect();
    set.sort();
    set
}

/// Pharmpy's `all_combinations`: the product over categories of "none or
/// one of the category's features", minus the empty choice and minus the
/// pairs Pharmpy's stepwise rules refuse — the first category outermost, in
/// dictionary order, so the candidate numbering matches.
fn combinations(funcs: &[FeatureKey], base: &Structure) -> Vec<Vec<FeatureKey>> {
    let mut groups: Vec<Vec<FeatureKey>> = Vec::new();
    for key in funcs {
        match groups
            .iter_mut()
            .find(|g| g[0].category() == key.category())
        {
            Some(g) => g.push(*key),
            None => groups.push(vec![*key]),
        }
    }
    let mut out: Vec<Vec<FeatureKey>> = vec![vec![]];
    for group in &groups {
        let mut next = Vec::new();
        for prefix in &out {
            next.push(prefix.clone());
            for key in group {
                let mut combo = prefix.clone();
                combo.push(*key);
                next.push(combo);
            }
        }
        out = next;
    }
    out.retain(|c| !c.is_empty() && structure::combination_allowed(c, base));
    out
}

/// Where a search run's files go by default: `<config stem>-modelsearch`
/// next to the config file.
pub fn default_dir(config_path: &Path) -> PathBuf {
    let stem = config_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("search");
    config_path.with_file_name(format!("{stem}-modelsearch"))
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "pharmpy_anchor.rs"]
mod pharmpy_anchor;
