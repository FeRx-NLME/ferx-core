//! Global model search — pyDarwin's genetic algorithm and exhaustive
//! enumeration over one candidate grid, with penalized fitness (#1185, P6 of
//! the #1175 epic).
//!
//! The stepwise tools of the epic each walk one axis of the model space:
//! modelsearch (#1181) the structure, covsearch (#1180) the covariate
//! relations. This tool lays the two out as one **grid** — every structural
//! category the `[space]` names is an axis with the category's values as
//! alleles, every optional `COVARIATE?` pair is an axis with `none` and each
//! of its forms — and searches the grid **globally**: either every point
//! (`exhaustive`), or a genetic algorithm ([`ga`]) that evaluates a fraction
//! of them. A point of the grid is a *genome*, one allele index per axis;
//! decoding it is the same edit the stepwise tools make (a `pk` template
//! swap through `ferx-core::edit`, a `[covariate_model]` line per relation),
//! applied to the input model seeded from its own fit.
//!
//! Every candidate goes through the shared runner (`StepFitter`) — the
//! same dedup by canonical hash, the same journal, the same strictness gate,
//! the same per-candidate cost tiers for `[odes]` structures — and is ranked
//! on `[rank] type`, which for this tool defaults to pyDarwin's
//! [`penalized`](crate::search::Penalties) fitness. Beyond the criterion the
//! search charges three things the criterion cannot see: a gene that changed
//! nothing in the rendered model (a covariate on a parameter the structural
//! choice removed), a candidate that produced no fit at all, and a fit the
//! gate refused — see [`Penalties`].
//!
//! # What pyDarwin has that this does not
//!
//! The template + tokens file pair (ferx has a parser and an edit layer, so
//! a candidate is a typed edit, not a string substitution), the GP / RF /
//! GBRT surrogate models (they pay off when a fit is a NONMEM subprocess
//! costing minutes; here it is in-process), and the grid run manager.
//! `docs/tools/global-search.qmd` states when a global search beats the
//! stepwise tools and when it does not.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use ferx_core::edit::{ModelEdit, ModelText};
use ferx_core::{CancelFlag, FitResult};
use serde::Deserialize;

use crate::covsearch::Effect;
use crate::modelsearch::structure::{
    self, parameter_names_of, Defaults, FeatureKey, IivStrategy, Structure, TransitCount,
};
use crate::search::fitter::{RunnerFitter, StepFitter};
use crate::search::seed::seed_from;
use crate::search::{
    BaseModel, Candidate, CandidateError, CandidateResult, Criterion, Feature, FeatureVector, Mfl,
    ModelContext, Penalties, PkTemplate, RankType, RunReport, SearchConfig, Statement,
};

pub mod ga;
mod report;

pub use ga::{GaOptions, Generation, Genome};
pub use report::{
    final_model_path, models_dir, models_path, render_summary, write_report, MODEL_COLUMNS,
};

/// `[globalsearch] algorithm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Algorithm {
    /// Every point of the grid, in one step.
    Exhaustive,
    /// The genetic algorithm — [`ga`].
    #[default]
    Ga,
}

impl Algorithm {
    pub fn label(&self) -> &'static str {
        match self {
            Algorithm::Exhaustive => "exhaustive",
            Algorithm::Ga => "ga",
        }
    }
}

/// The `[globalsearch]` section of a `.ferxsearch` file, plus the `[rank]`
/// keys the tool reads.
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalsearchOptions {
    pub algorithm: Algorithm,
    /// How η is given to the parameters a structural move introduces —
    /// modelsearch's `iiv_strategy`.
    pub iiv_strategy: IivStrategy,
    /// The largest grid `exhaustive` will enumerate. A bigger grid is an
    /// error naming the size and this key, not a silently truncated
    /// search; the GA has no such cap.
    pub max_models: usize,
    /// `[globalsearch.ga]`.
    pub ga: GaOptions,
    /// `[rank] type`; `penalized` when the file does not say.
    pub rank: RankType,
    /// `[rank.penalties]`: the schedule behind a `penalized` criterion, and
    /// the three search-level charges under any criterion.
    pub penalties: Penalties,
}

impl Default for GlobalsearchOptions {
    fn default() -> Self {
        GlobalsearchOptions {
            algorithm: Algorithm::default(),
            iiv_strategy: IivStrategy::default(),
            max_models: DEFAULT_MAX_MODELS,
            ga: GaOptions::default(),
            rank: RankType::Penalized,
            penalties: Penalties::default(),
        }
    }
}

/// The default cap on an exhaustive enumeration.
pub const DEFAULT_MAX_MODELS: usize = 500;

#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Section {
    #[serde(default)]
    algorithm: Algorithm,
    #[serde(default)]
    iiv_strategy: IivStrategy,
    #[serde(default)]
    max_models: Option<usize>,
    #[serde(default)]
    ga: GaOptions,
}

impl GlobalsearchOptions {
    /// Read `[globalsearch]` and `[rank]` off a loaded file.
    pub fn from_config(config: &SearchConfig) -> Result<Self, String> {
        config.require_space(
            "globalsearch",
            "ABSORPTION / ELIMINATION / PERIPHERALS / TRANSITS / LAGTIME and COVARIATE? \
             statements",
        )?;
        let section = match config.tools.get("globalsearch") {
            Some(table) => table
                .clone()
                .try_into::<Section>()
                .map_err(|e| format!("[globalsearch]: {e}"))?,
            None => Section::default(),
        };
        let options = GlobalsearchOptions {
            algorithm: section.algorithm,
            iiv_strategy: section.iiv_strategy,
            max_models: section.max_models.unwrap_or(DEFAULT_MAX_MODELS),
            ga: section.ga,
            // pyDarwin's ranking is the point of the tool, so the file's
            // silence means penalized here, where every other tool reads it
            // as a BIC.
            rank: config.rank.kind.unwrap_or(RankType::Penalized),
            penalties: config.rank.penalties(),
        };
        options.validate()?;
        Ok(options)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.iiv_strategy == IivStrategy::Fullblock {
            return Err(
                "[globalsearch] iiv_strategy = \"fullblock\": a block over the new and existing \
                 η is a variability-structure move, which is iivsearch's (#1183); use \
                 add_diagonal, absorption_delay or no_add"
                    .into(),
            );
        }
        if self.max_models == 0 {
            return Err("[globalsearch] max_models must be at least 1".into());
        }
        self.ga.validate()?;
        self.penalties.validate()?;
        Ok(())
    }

    /// Refuse a file whose space this tool cannot lay out as a grid —
    /// before any data is read. The same partition `Space::from_config`
    /// makes, on the unresolved space.
    pub fn check_space(config: &SearchConfig) -> Result<(), String> {
        partition(&config.mfl).map(|_| ())
    }

    /// The runner criterion this ranks on.
    pub fn criterion(&self) -> Criterion {
        self.rank.criterion_with(self.penalties)
    }
}

/// One axis of the grid.
#[derive(Debug, Clone, PartialEq)]
pub enum Axis {
    /// One structural category; the alleles are its values in the space.
    Structural {
        category: &'static str,
        keys: Vec<FeatureKey>,
    },
    /// One `COVARIATE?` pair; allele 0 is `none`, the rest its forms.
    Covariate { pair: String, forms: Vec<Effect> },
}

impl Axis {
    pub fn name(&self) -> String {
        match self {
            Axis::Structural { category, .. } => (*category).to_string(),
            Axis::Covariate { pair, .. } => pair.clone(),
        }
    }

    /// The number of alleles.
    pub fn len(&self) -> usize {
        match self {
            Axis::Structural { keys, .. } => keys.len(),
            Axis::Covariate { forms, .. } => forms.len() + 1,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The label of one allele: `FO`, `2`, `none`, `power`.
    pub fn label(&self, allele: usize) -> String {
        match self {
            Axis::Structural { keys, .. } => keys[allele].argument(),
            Axis::Covariate { forms, .. } => match allele {
                0 => "none".to_string(),
                k => forms[k - 1].form_label().to_string(),
            },
        }
    }

    /// Every allele's label.
    pub fn labels(&self) -> Vec<String> {
        (0..self.len()).map(|a| self.label(a)).collect()
    }
}

/// The grid, resolved against the base model.
#[derive(Debug)]
pub(crate) struct Space {
    pub input_model: ModelText,
    /// The input's structure, when the space has a structural axis.
    pub base_structure: Option<Structure>,
    pub defaults: Option<Defaults>,
    /// The input's `pk` line and `[individual_parameters]`, for the
    /// structural edit.
    pub template: Option<PkTemplate>,
    pub lines: Vec<String>,
    pub axes: Vec<Axis>,
    /// `COVARIATE(...)` effects every candidate carries.
    pub forced: Vec<Effect>,
    pub notes: Vec<String>,
}

/// Split a space into its structural statements and its covariate
/// effects, refusing anything else by name.
fn partition(mfl: &Mfl) -> Result<(Mfl, bool), String> {
    let mut structural = Mfl {
        statements: Vec::new(),
    };
    let mut has_covariate = false;
    for statement in &mfl.statements {
        match statement {
            Statement::Let { .. } => structural.statements.push(statement.clone()),
            Statement::Feature(f) => match f {
                Feature::Absorption(_)
                | Feature::Elimination(_)
                | Feature::Peripherals { .. }
                | Feature::Transits { .. }
                | Feature::Lagtime(_) => structural.statements.push(statement.clone()),
                Feature::Covariate { .. } => has_covariate = true,
                other => {
                    return Err(format!(
                        "[space] mfl: `{}` is not an axis globalsearch lays out; the grid \
                         takes the structural statements (ABSORPTION, ELIMINATION, \
                         PERIPHERALS, TRANSITS, LAGTIME) and COVARIATE?. Variability \
                         (iivsearch, iovsearch) and ALLOMETRY have their own tools (#1175)",
                        other.keyword()
                    ))
                }
            },
        }
    }
    Ok((structural, has_covariate))
}

impl Space {
    fn from_config(config: &SearchConfig, base: &BaseModel) -> Result<Space, String> {
        let resolved = config.resolve_space(base)?;
        let (structural, _) = partition(&resolved.mfl)?;
        let keys = if structural.features().next().is_some() {
            structure::space_features(&structural)?
        } else {
            Vec::new()
        };
        let ctx =
            ModelContext::from_model(&base.prepared.parsed, &base.text, &base.prepared.population)?;
        let existing: Vec<(String, String)> = base
            .prepared
            .parsed
            .model
            .covariate_model
            .as_ref()
            .map(|spec| {
                spec.relations
                    .iter()
                    .map(|r| (r.parameter.clone(), r.covariate.clone()))
                    .collect()
            })
            .unwrap_or_default();
        let structural_base = if keys.is_empty() {
            None
        } else {
            let template = ctx.template.as_ref().ok_or_else(|| {
                "globalsearch: the base model has no `pk NAME(...)` line; an `ode(...)` or \
                 algebraic model has no template to swap, so the space can carry no structural \
                 axis — search its covariates only (COVARIATE? statements) or use a `pk` base"
                    .to_string()
            })?;
            let structure = Structure::from_model(template, Some(&base.text))?;
            if let Some(why) = structure.unbuildable() {
                return Err(format!(
                    "globalsearch: the input model's own structure has no template to search \
                     from: {why}"
                ));
            }
            let defaults = Defaults::new(
                ctx.parameters.clone(),
                base.prepared.init_params.theta_names.clone(),
                base.prepared.init_params.theta.clone(),
                base.prepared.parsed.model.eta_names.clone(),
                &base.prepared.population,
            );
            Some((structure, defaults, template.clone()))
        };
        Self::build(
            base.text.clone(),
            structural_base,
            &keys,
            &resolved.covariate_effects,
            &existing,
            resolved.notes.clone(),
        )
    }

    /// The grid from its parts — the seam the unit tests use.
    pub(crate) fn build(
        input_model: ModelText,
        structural: Option<(Structure, Defaults, PkTemplate)>,
        keys: &[FeatureKey],
        covariate_effects: &[crate::search::CovariateEffectSpec],
        existing: &[(String, String)],
        mut notes: Vec<String>,
    ) -> Result<Space, String> {
        let mut axes: Vec<Axis> = Vec::new();
        // `space_features` sorts by category, so the axes come out in a
        // stable order whatever order the file wrote the statements.
        for key in keys {
            match axes.iter_mut().find(
                |a| matches!(a, Axis::Structural { category, .. } if *category == key.category()),
            ) {
                Some(Axis::Structural { keys, .. }) => keys.push(*key),
                _ => axes.push(Axis::Structural {
                    category: key.category(),
                    keys: vec![*key],
                }),
            }
        }
        let mut forced: Vec<Effect> = Vec::new();
        for spec in covariate_effects {
            let effect = Effect::from_spec(spec)?;
            let in_base = existing
                .iter()
                .any(|(p, c)| *p == effect.parameter && *c == effect.covariate);
            if in_base {
                notes.push(format!(
                    "not explored: {} — the base model already declares `{} ~ {}`, which the \
                     search keeps as written",
                    effect.label(),
                    effect.parameter,
                    effect.covariate
                ));
                continue;
            }
            if !spec.optional {
                forced.push(effect);
                continue;
            }
            let pair = effect.pair_key();
            match axes
                .iter_mut()
                .find(|a| matches!(a, Axis::Covariate { pair: p, .. } if *p == pair))
            {
                Some(Axis::Covariate { forms, .. }) => forms.push(effect),
                _ => axes.push(Axis::Covariate {
                    pair,
                    forms: vec![effect],
                }),
            }
        }
        if axes.is_empty() {
            return Err(
                "globalsearch: the space has no axis to search — every COVARIATE? pair is \
                 already in the base model and no structural statement is present"
                    .into(),
            );
        }
        let (base_structure, defaults, template) = match structural {
            Some((s, d, t)) => (Some(s), Some(d), Some(t)),
            None => (None, None, None),
        };
        let lines = input_model.block_lines("individual_parameters");
        Ok(Space {
            input_model,
            base_structure,
            defaults,
            template,
            lines,
            axes,
            forced,
            notes,
        })
    }

    /// The allele count per axis, for the GA and the enumeration.
    pub fn alleles(&self) -> Vec<usize> {
        self.axes.iter().map(Axis::len).collect()
    }

    /// The grid's size.
    pub fn size(&self) -> Option<usize> {
        ga::space_size(&self.alleles())
    }

    /// `ABSORPTION=FO;CL-WT=power;…` — the genome as its allele labels.
    pub fn describe(&self, genome: &Genome) -> String {
        self.axes
            .iter()
            .zip(genome)
            .map(|(axis, allele)| format!("{}={}", axis.name(), axis.label(*allele)))
            .collect::<Vec<_>>()
            .join(";")
    }
}

/// A decoded genome: the model it renders to, or why it cannot.
struct Decoded {
    model: ModelText,
    structure: Option<Structure>,
    effects: Vec<Effect>,
    features: FeatureVector,
    non_influential: usize,
    cost: f64,
    starts: Option<usize>,
    /// Why no model can express this point, when none can.
    problem: Option<String>,
}

/// One fitted model of the search, as the table reports it.
#[derive(Debug, Clone)]
pub struct ModelRow {
    /// `input`, or `run{n}` in creation order.
    pub id: String,
    pub parent: Option<String>,
    /// The batch that produced it: `input`, `candidates` (exhaustive),
    /// `generation-{g}`, `downhill-{g}-{k}-{round}`.
    pub step: String,
    /// The grid point; `None` for the input.
    pub genome: Option<Genome>,
    /// The genome as labels, `ABSORPTION=FO;CL-WT=power`.
    pub description: String,
    pub structure: Option<Structure>,
    /// The covariate effects the candidate carries (forced ones included).
    pub effects: Vec<Effect>,
    pub ofv: Option<f64>,
    pub n_parameters: Option<usize>,
    /// `[rank] type` on the fit; `NaN` without a fit.
    pub criterion: f64,
    /// What the search ranks on: the criterion plus the non-influential and
    /// gate charges, or the crash value without a fit. Always finite.
    pub fitness: f64,
    /// 1 for the best eligible model; `None` for a model that is not
    /// eligible.
    pub rank: Option<usize>,
    pub converged: Option<bool>,
    /// Passed the strictness gate and has a fit.
    pub passed: bool,
    pub failures: Vec<String>,
    pub error: Option<CandidateError>,
    pub seconds: f64,
    pub selected: bool,
    /// Genes that changed nothing in the rendered model.
    pub non_influential: usize,
    /// Rendered to the same model as the named row, which was the one
    /// fitted.
    pub duplicate_of: Option<String>,
    pub reused: bool,
}

impl ModelRow {
    /// Eligible for selection: fitted, passed the gate, finite criterion.
    pub fn eligible(&self) -> bool {
        self.error.is_none() && self.passed && self.criterion.is_finite()
    }
}

/// What a search reports as it runs, for a CLI progress line.
#[derive(Debug, Clone)]
pub enum GlobalsearchEvent {
    InputStarted,
    InputFinished {
        ofv: f64,
        criterion: f64,
    },
    /// The grid is laid out.
    Space {
        axes: usize,
        size: usize,
    },
    /// A batch is about to be fitted: `candidates` new models (cached and
    /// duplicate genomes cost nothing).
    BatchStarted {
        step: String,
        proposed: usize,
        candidates: usize,
    },
    /// `best` is the batch's lowest fitness among the eligible.
    BatchFinished {
        step: String,
        best: Option<(String, f64)>,
    },
}

pub type ProgressFn<'a> = &'a (dyn Fn(GlobalsearchEvent) + Send + Sync);

/// The outcome of a search.
#[derive(Debug, Clone)]
pub struct GlobalsearchResult {
    pub options: GlobalsearchOptions,
    pub criterion: Criterion,
    pub input_model: ModelText,
    /// The axes, each with its allele labels.
    pub axes: Vec<(String, Vec<String>)>,
    pub space_size: usize,
    /// Every fitted model, in creation order.
    pub rows: Vec<ModelRow>,
    /// The GA's per-generation summary; empty for `exhaustive`.
    pub generations: Vec<Generation>,
    pub final_id: String,
    pub final_model: ModelText,
    pub final_fit: Option<FitResult>,
    pub final_fitness: f64,
    /// Every fitted model's text, by id.
    pub models: BTreeMap<String, ModelText>,
    pub notes: Vec<String>,
    /// The search stopped on a cancel flag; `rows` is partial.
    pub cancelled: bool,
}

impl GlobalsearchResult {
    pub fn row(&self, id: &str) -> Option<&ModelRow> {
        self.rows.iter().find(|r| r.id == id)
    }

    /// The eligible rows, best first.
    pub fn ranked(&self) -> Vec<&ModelRow> {
        let mut rows: Vec<&ModelRow> = self.rows.iter().filter(|r| r.rank.is_some()).collect();
        rows.sort_by_key(|r| r.rank);
        rows
    }

    /// Models actually fitted (not duplicates, not unbuildable).
    pub fn n_fitted(&self) -> usize {
        self.rows
            .iter()
            .filter(|r| r.duplicate_of.is_none() && r.ofv.is_some())
            .count()
    }
}

/// Everything a [`run_globalsearch`] call takes beyond the file.
#[derive(Default)]
pub struct GlobalsearchRun<'a> {
    /// Where the per-batch journals, `models.csv` and `final.ferx` go.
    /// `None` keeps everything in memory: no resume, no files.
    pub dir: Option<PathBuf>,
    /// Overrides `[run] threads`.
    pub threads: Option<usize>,
    pub cancel: Option<CancelFlag>,
    pub progress: Option<ProgressFn<'a>>,
    /// Other searches' directories whose fits this run may reuse, on top of
    /// the file's `[run] reuse_from` (#1185).
    pub reuse_from: Vec<PathBuf>,
}

/// Run a global search from a loaded `.ferxsearch` file and its base model.
/// Writes `models.csv` and `final.ferx` into `run.dir` when given.
pub fn run_globalsearch(
    config: &SearchConfig,
    base: &BaseModel,
    run: GlobalsearchRun<'_>,
) -> Result<GlobalsearchResult, String> {
    let options = GlobalsearchOptions::from_config(config)?;
    let mut run_options = config.run_options();
    run_options.criterion = options.criterion();
    let fitter = RunnerFitter {
        threads: run.threads.or(config.run.threads).unwrap_or(0),
        dir: run.dir.clone(),
        cancel: run.cancel.clone(),
        data: &base.prepared.population,
        options: run_options,
        reuse_from: run
            .reuse_from
            .iter()
            .cloned()
            .chain(config.reuse_dirs())
            .collect(),
    };
    let space = Space::from_config(config, base)?;
    let result = search(&fitter, space, &options, run.progress)?;
    if let Some(dir) = &run.dir {
        write_report(dir, &result)?;
    }
    Ok(result)
}

const INPUT_ID: &str = "input";

/// The oracle the GA and the enumeration share: decodes genomes, fits the
/// new ones through the step fitter, caches by genome and by canonical
/// hash, and keeps every row.
struct Evaluator<'a> {
    fitter: &'a dyn StepFitter,
    space: &'a Space,
    options: &'a GlobalsearchOptions,
    criterion: Criterion,
    progress: Option<ProgressFn<'a>>,
    /// The input model, seeded from its fit, that every candidate derives
    /// from.
    root: ModelText,
    by_genome: HashMap<Genome, usize>,
    by_hash: HashMap<String, usize>,
    rows: Vec<ModelRow>,
    store: HashMap<String, (ModelText, Option<FitResult>)>,
    notes: Vec<String>,
    next_id: usize,
    cancelled: bool,
}

impl<'a> Evaluator<'a> {
    fn new_id(&mut self) -> String {
        self.next_id += 1;
        format!("run{}", self.next_id)
    }

    fn emit(&self, event: GlobalsearchEvent) {
        if let Some(p) = self.progress {
            p(event);
        }
    }

    fn note(&mut self, note: String) {
        if !self.notes.contains(&note) {
            self.notes.push(note);
        }
    }

    /// The model a genome renders to.
    fn decode(&mut self, genome: &Genome) -> Result<Decoded, String> {
        let space = self.space;
        let mut model = self.root.clone();
        let mut features = FeatureVector::new();
        let mut structure = None;
        let (mut cost, mut starts) = (1.0, None);
        if let (Some(base), Some(defaults), Some(template)) =
            (&space.base_structure, &space.defaults, &space.template)
        {
            let mut target = *base;
            for (axis, allele) in space.axes.iter().zip(genome) {
                if let Axis::Structural { keys, .. } = axis {
                    target = target.apply(&keys[*allele]);
                }
            }
            features = target.feature_vector();
            let mut problem = target.unbuildable();
            if problem.is_none() {
                // Pharmpy's pair table, over the *whole* structure — a
                // coordinate the base fixes counts as much as a chosen one
                // (`TRANSITS(1)` on a first-order base). An absent transit
                // chain is not a coordinate the table names.
                let live: Vec<FeatureKey> = target
                    .features()
                    .into_iter()
                    .filter(|k| !matches!(k, FeatureKey::Transits(TransitCount::Count(0))))
                    .collect();
                for (i, a) in live.iter().enumerate() {
                    for b in &live[i + 1..] {
                        if structure::pharmpy_incompatible(a, b) {
                            problem = Some(format!(
                                "Pharmpy does not combine {a} with {b}, and neither does ferx"
                            ));
                        }
                    }
                }
            }
            if let Some(why) = problem {
                return Ok(Decoded {
                    model,
                    structure: Some(target),
                    effects: Vec::new(),
                    features,
                    non_influential: 0,
                    cost,
                    starts,
                    problem: Some(why),
                });
            }
            if target != *base {
                let defaults = defaults.of_text(&model);
                let spec = structure::structural_spec(
                    &target,
                    base,
                    template,
                    &space.lines,
                    &defaults,
                    self.options.iiv_strategy,
                );
                let applied = spec.and_then(|spec| model.apply(ModelEdit::SetStructural(spec)));
                if let Err(e) = applied {
                    return Ok(Decoded {
                        model,
                        structure: Some(target),
                        effects: Vec::new(),
                        features,
                        non_influential: 0,
                        cost,
                        starts,
                        problem: Some(e),
                    });
                }
            }
            cost = target.cost();
            starts = target.starts();
            structure = Some(target);
        }
        let parameters = parameter_names_of(&model);
        let mut effects = Vec::new();
        let mut non_influential = 0usize;
        for effect in &space.forced {
            if parameters.contains(&effect.parameter) {
                model
                    .apply(ModelEdit::AddCovariateRelation(effect.relation()))
                    .map_err(|e| format!("forcing {}: {e}", effect.label()))?;
                effects.push(effect.clone());
            } else {
                // A `COVARIATE(...)` without the `?` is part of every model
                // the space describes. A structural allele that removes its
                // parameter leaves no model that carries it, so the point is
                // not in the space — refused, never fitted without the
                // relation, never selected.
                return Ok(Decoded {
                    model,
                    structure,
                    effects,
                    features,
                    non_influential,
                    cost,
                    starts,
                    problem: Some(format!(
                        "the forced effect {} needs a `{}` parameter, which this structure \
                         does not declare",
                        effect.label(),
                        effect.parameter
                    )),
                });
            }
        }
        for (axis, allele) in space.axes.iter().zip(genome) {
            let Axis::Covariate { pair, forms } = axis else {
                continue;
            };
            if *allele == 0 {
                features.set(pair.clone(), "none");
                continue;
            }
            let effect = &forms[*allele - 1];
            if parameters.contains(&effect.parameter) {
                model
                    .apply(ModelEdit::AddCovariateRelation(effect.relation()))
                    .map_err(|e| format!("adding {}: {e}", effect.label()))?;
                features.set(pair.clone(), effect.form_label());
                effects.push(effect.clone());
            } else {
                // The structural choice removed the parameter: the gene
                // changes nothing, and pyDarwin charges it for that.
                non_influential += 1;
                features.set(
                    pair.clone(),
                    format!("{} (non-influential)", effect.form_label()),
                );
            }
        }
        Ok(Decoded {
            model,
            structure,
            effects,
            features,
            non_influential,
            cost,
            starts,
            problem: None,
        })
    }

    /// The fitness of a fitted candidate: criterion, plus the search-level
    /// charges. Finite by construction.
    fn fitness_of(
        &self,
        result: &CandidateResult,
        fit: Option<&FitResult>,
        non_influential: usize,
    ) -> (f64, f64) {
        let p = self.options.penalties;
        if result.error.is_some() {
            return (f64::NAN, p.crash);
        }
        let criterion = match fit {
            Some(fit) => self.criterion.of(fit),
            None => result.criterion,
        };
        if !criterion.is_finite() {
            return (criterion, p.crash);
        }
        let mut fitness = criterion + p.non_influential_charge(non_influential);
        if !result.verdict.passed {
            fitness += p.gate;
        }
        (criterion, fitness)
    }

    fn row_of(
        &self,
        result: &CandidateResult,
        step: &str,
        genome: Option<Genome>,
        decoded: &Decoded,
    ) -> ModelRow {
        let fit = result.fit.as_ref();
        let (criterion, fitness) = self.fitness_of(result, fit, decoded.non_influential);
        ModelRow {
            id: result.id.clone(),
            parent: result.parent.clone(),
            step: step.to_string(),
            description: genome
                .as_ref()
                .map(|g| self.space.describe(g))
                .unwrap_or_else(|| "the input model".to_string()),
            genome,
            structure: decoded.structure,
            effects: decoded.effects.clone(),
            ofv: result.ofv,
            n_parameters: fit.map(|f| f.n_parameters),
            criterion,
            fitness,
            rank: None,
            converged: result.converged,
            passed: result.verdict.passed && result.error.is_none(),
            failures: result.verdict.failures.clone(),
            error: result.error.clone(),
            seconds: result.seconds,
            selected: false,
            non_influential: decoded.non_influential,
            duplicate_of: result.duplicate_of.clone(),
            reused: result.reused,
        }
    }

    /// The fit behind a result: its own, or its representative's.
    fn fit_behind(&self, result: &CandidateResult, report: &RunReport) -> Option<FitResult> {
        if let Some(fit) = &result.fit {
            return Some(fit.clone());
        }
        let rep = result.duplicate_of.as_ref()?;
        report
            .results
            .iter()
            .find(|r| r.id == *rep)
            .and_then(|r| r.fit.clone())
            .or_else(|| self.store.get(rep).and_then(|(_, f)| f.clone()))
    }
}

/// One genome of a batch, between decoding and the table.
enum Entry {
    /// Already answered: an unbuildable point, or a model an earlier batch
    /// fitted. `store` is the model text to keep, when there is one.
    Ready {
        row: ModelRow,
        store: Option<(String, ModelText)>,
    },
    /// To be fitted.
    Fit {
        genome: Genome,
        candidate: Candidate,
        decoded: Decoded,
    },
}

impl ga::Oracle for Evaluator<'_> {
    fn evaluate(&mut self, what: &str, genomes: &[Genome]) -> Result<Vec<f64>, String> {
        let crash = self.options.penalties.crash;
        // A genome proposed twice in one batch is one point: decode and fit
        // it once, answer both.
        let mut unique: Vec<Genome> = Vec::new();
        for g in genomes {
            if !unique.contains(g) {
                unique.push(g.clone());
            }
        }
        // Decoded in order; the rows are appended in that same order after
        // the fit, so the table reads as the batch was proposed whether a
        // point was fitted, reused or refused.
        let mut entries: Vec<Entry> = Vec::new();
        for genome in &unique {
            if self.by_genome.contains_key(genome) || self.cancelled {
                continue;
            }
            let decoded = self.decode(genome)?;
            let id = self.new_id();
            if let Some(why) = &decoded.problem {
                entries.push(Entry::Ready {
                    row: ModelRow {
                        id: id.clone(),
                        parent: Some(INPUT_ID.to_string()),
                        step: what.to_string(),
                        description: self.space.describe(genome),
                        genome: Some(genome.clone()),
                        structure: decoded.structure,
                        effects: Vec::new(),
                        ofv: None,
                        n_parameters: None,
                        criterion: f64::NAN,
                        fitness: crash,
                        rank: None,
                        converged: None,
                        passed: false,
                        failures: vec![format!("not generated: {why}")],
                        error: Some(CandidateError::model(format!("not generated: {why}"))),
                        seconds: 0.0,
                        selected: false,
                        non_influential: 0,
                        duplicate_of: None,
                        reused: false,
                    },
                    store: None,
                });
                continue;
            }
            let mut candidate = Candidate::new(&id, decoded.model.clone())
                .parent(INPUT_ID)
                .features(decoded.features.clone())
                .cost(decoded.cost);
            if let Some(n) = decoded.starts {
                candidate = candidate.starts(n);
            }
            // Rendered to a model an earlier batch already fitted: the same
            // score, no fit — the cross-batch half of the runner's dedup.
            if let Some(&rep) = self.by_hash.get(&candidate.hash()) {
                let source = &self.rows[rep];
                let p = self.options.penalties;
                let mut fitness = if source.error.is_some() || !source.criterion.is_finite() {
                    p.crash
                } else {
                    source.criterion
                        + p.non_influential_charge(decoded.non_influential)
                        + if source.passed { 0.0 } else { p.gate }
                };
                if !fitness.is_finite() {
                    fitness = p.crash;
                }
                entries.push(Entry::Ready {
                    row: ModelRow {
                        id: id.clone(),
                        parent: Some(INPUT_ID.to_string()),
                        step: what.to_string(),
                        description: self.space.describe(genome),
                        genome: Some(genome.clone()),
                        structure: decoded.structure,
                        effects: decoded.effects.clone(),
                        ofv: source.ofv,
                        n_parameters: source.n_parameters,
                        criterion: source.criterion,
                        fitness,
                        rank: None,
                        converged: source.converged,
                        passed: source.passed,
                        failures: source.failures.clone(),
                        error: source.error.clone(),
                        seconds: 0.0,
                        selected: false,
                        non_influential: decoded.non_influential,
                        duplicate_of: Some(source.id.clone()),
                        reused: source.reused,
                    },
                    store: Some((id, decoded.model.clone())),
                });
                continue;
            }
            entries.push(Entry::Fit {
                genome: genome.clone(),
                candidate,
                decoded,
            });
        }
        let submitted: Vec<Candidate> = entries
            .iter()
            .filter_map(|e| match e {
                Entry::Fit { candidate, .. } => Some(candidate.clone()),
                Entry::Ready { .. } => None,
            })
            .collect();
        let report = if submitted.is_empty() {
            None
        } else {
            self.emit(GlobalsearchEvent::BatchStarted {
                step: what.to_string(),
                proposed: unique.len(),
                candidates: submitted.len(),
            });
            let report = self.fitter.fit_step(what, &submitted)?;
            self.notes.extend(report.warnings.iter().cloned());
            Some(report)
        };
        let mut best: Option<(String, f64)> = None;
        for entry in entries {
            match entry {
                Entry::Ready { row, store } => {
                    if let Some((id, text)) = store {
                        self.store.insert(id, (text, None));
                    }
                    self.by_genome
                        .insert(row.genome.clone().expect("a grid row"), self.rows.len());
                    self.rows.push(row);
                }
                Entry::Fit {
                    genome,
                    candidate,
                    decoded,
                } => {
                    let report = report.as_ref().expect("a batch with candidates was fitted");
                    let Some(result) = report.results.iter().find(|r| r.id == candidate.id) else {
                        // Cancelled before this candidate was reached: no
                        // row, so a resumed run fits it.
                        self.cancelled = true;
                        continue;
                    };
                    let row = self.row_of(result, what, Some(genome.clone()), &decoded);
                    if row.eligible() && best.as_ref().is_none_or(|(_, f)| row.fitness < *f) {
                        best = Some((row.id.clone(), row.fitness));
                    }
                    let fit = self.fit_behind(result, report);
                    self.store
                        .insert(row.id.clone(), (candidate.model.clone(), fit));
                    if result.duplicate_of.is_none() {
                        self.by_hash.insert(candidate.hash(), self.rows.len());
                    }
                    self.by_genome.insert(genome, self.rows.len());
                    self.rows.push(row);
                }
            }
        }
        if let Some(report) = report {
            if report.cancelled {
                self.cancelled = true;
            }
            self.emit(GlobalsearchEvent::BatchFinished {
                step: what.to_string(),
                best,
            });
        }
        Ok(genomes
            .iter()
            .map(|g| {
                self.by_genome
                    .get(g)
                    .map(|i| self.rows[*i].fitness)
                    .filter(|f| f.is_finite())
                    .unwrap_or(crash)
            })
            .collect())
    }

    fn cancelled(&self) -> bool {
        self.cancelled
    }
}

/// The search proper, over an injected fitter.
pub(crate) fn search(
    fitter: &dyn StepFitter,
    space: Space,
    options: &GlobalsearchOptions,
    progress: Option<ProgressFn<'_>>,
) -> Result<GlobalsearchResult, String> {
    let emit = |event: GlobalsearchEvent| {
        if let Some(p) = progress {
            p(event);
        }
    };
    let criterion = options.criterion();
    let alleles = space.alleles();
    let size = space
        .size()
        .ok_or("globalsearch: the grid is too large to count")?;
    emit(GlobalsearchEvent::Space {
        axes: alleles.len(),
        size,
    });
    if options.algorithm == Algorithm::Exhaustive && size > options.max_models {
        return Err(format!(
            "globalsearch: the grid has {size} points, above [globalsearch] max_models = {}; \
             raise it or use algorithm = \"ga\"",
            options.max_models
        ));
    }

    // ── the input ────────────────────────────────────────────────────────
    emit(GlobalsearchEvent::InputStarted);
    let mut input_features = space
        .base_structure
        .map(|s| s.feature_vector())
        .unwrap_or_default();
    for axis in &space.axes {
        if let Axis::Covariate { pair, .. } = axis {
            input_features.set(pair.clone(), "none");
        }
    }
    let candidate = Candidate::new(INPUT_ID, space.input_model.clone()).features(input_features);
    let report = fitter.fit_step(INPUT_ID, std::slice::from_ref(&candidate))?;
    let mut notes = space.notes.clone();
    notes.extend(report.warnings.iter().cloned());
    let result = report
        .results
        .first()
        .ok_or("the input model was not fitted")?;
    if let Some(e) = &result.error {
        return Err(format!("the input model could not be fitted: {e}"));
    }
    let input_fit = result.fit.clone();
    let mut root = space.input_model.clone();
    if let Some(fit) = &input_fit {
        seed_from(&mut root, fit)?;
    } else {
        notes.push(
            "the input model's fit is not available (a resumed row whose cache is gone), so \
             candidates start from the file's initial estimates"
                .into(),
        );
    }
    let mut evaluator = Evaluator {
        fitter,
        space: &space,
        options,
        criterion,
        progress,
        root,
        by_genome: HashMap::new(),
        by_hash: HashMap::new(),
        rows: Vec::new(),
        store: HashMap::new(),
        notes,
        next_id: 0,
        cancelled: report.cancelled,
    };
    let input_decoded = Decoded {
        model: space.input_model.clone(),
        structure: space.base_structure,
        effects: Vec::new(),
        features: FeatureVector::new(),
        non_influential: 0,
        cost: 1.0,
        starts: None,
        problem: None,
    };
    let input_row = evaluator.row_of(result, INPUT_ID, None, &input_decoded);
    emit(GlobalsearchEvent::InputFinished {
        ofv: input_row.ofv.unwrap_or(f64::NAN),
        criterion: input_row.criterion,
    });
    if !input_row.passed {
        evaluator.note(format!(
            "the input model fails the strictness gate ({})",
            input_row.failures.join("; ")
        ));
    }
    evaluator
        .store
        .insert(INPUT_ID.to_string(), (space.input_model.clone(), input_fit));
    evaluator.rows.push(input_row);

    // ── the grid ─────────────────────────────────────────────────────────
    let mut generations = Vec::new();
    if !evaluator.cancelled {
        match options.algorithm {
            Algorithm::Exhaustive => {
                let genomes = ga::enumerate(&alleles);
                ga::Oracle::evaluate(&mut evaluator, "candidates", &genomes)?;
            }
            Algorithm::Ga => {
                if size < 2 {
                    return Err(
                        "globalsearch: the grid has a single point; there is nothing for the \
                         GA to search"
                            .into(),
                    );
                }
                let outcome = ga::run(&alleles, &options.ga, &mut evaluator)?;
                generations = outcome.generations;
            }
        }
    }

    // ── ranking and selection ────────────────────────────────────────────
    let Evaluator {
        mut rows,
        mut store,
        mut notes,
        cancelled,
        ..
    } = evaluator;
    let mut order: Vec<usize> = (0..rows.len()).filter(|i| rows[*i].eligible()).collect();
    order.sort_by(|a, b| rows[*a].fitness.total_cmp(&rows[*b].fitness));
    for (rank, i) in order.iter().enumerate() {
        rows[*i].rank = Some(rank + 1);
    }
    // The input is ranked for reference but never selected: it is not a
    // point of the grid the file describes (its own grid point, when it has
    // one, is a candidate of its own, seeded from it).
    let winner = order
        .iter()
        .copied()
        .find(|i| rows[*i].id != INPUT_ID)
        .unwrap_or(0);
    rows[winner].selected = true;
    let final_id = rows[winner].id.clone();
    let final_fitness = rows[winner].fitness;
    if rows[winner].id == INPUT_ID {
        notes.push("no candidate passed the strictness gate; the final model is the input".into());
    }
    let models: BTreeMap<String, ModelText> = store
        .iter()
        .map(|(id, (text, _))| (id.clone(), text.clone()))
        .collect();
    let (final_model, final_fit) =
        final_model_and_fit(&final_id, rows[winner].duplicate_of.as_deref(), &mut store)
            .unwrap_or_else(|| (space.input_model.clone(), None));
    if final_fit.is_none() {
        notes.push(format!(
            "{final_id}: the fit is not in the journal cache, so final-fit.yaml cannot be \
             written and final.ferx keeps the candidate's starting values"
        ));
    }
    Ok(GlobalsearchResult {
        options: options.clone(),
        criterion,
        axes: space.axes.iter().map(|a| (a.name(), a.labels())).collect(),
        space_size: size,
        input_model: space.input_model,
        rows,
        generations,
        final_id,
        final_model,
        final_fit,
        final_fitness,
        models,
        notes,
        cancelled,
    })
}

/// The selected model's text and fit out of the store.
///
/// A duplicate's text is its own (it was decoded), but its fit is its
/// representative's: a cross-batch duplicate is stored as `(text, None)`,
/// so when the store has no fit under the winner's own id the
/// representative's is taken. Without that fallback a winning duplicate
/// wrote a `final.ferx` at its starting values and no `final-fit.yaml`
/// while the fit sat one row over.
fn final_model_and_fit(
    final_id: &str,
    duplicate_of: Option<&str>,
    store: &mut HashMap<String, (ModelText, Option<FitResult>)>,
) -> Option<(ModelText, Option<FitResult>)> {
    let rep: Option<(ModelText, Option<FitResult>)> =
        duplicate_of.and_then(|rep| store.get(rep).cloned());
    match store.remove(final_id) {
        Some((text, Some(fit))) => Some((text, Some(fit))),
        Some((text, None)) => Some((text, rep.and_then(|(_, f)| f))),
        None => rep,
    }
}

/// Where a search run's files go by default: `<config stem>-globalsearch`
/// next to the config file.
pub fn default_dir(config_path: &Path) -> PathBuf {
    crate::search::default_dir(config_path, "globalsearch")
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
