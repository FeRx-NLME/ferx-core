//! Binding `theta NAME[COL, ...]` blocks to data (#1064).
//!
//! A θ level block declares *one θ per observed combination* of some data
//! columns — the unstructured-placebo model of an MBMA analysis, where every
//! (study × timepoint) cell gets its own fixed effect so no parametric placebo
//! time-course can bias the drug effect. The level count is therefore a
//! property of the dataset, which the model file cannot know.
//!
//! This module closes that gap in three steps:
//!
//! 1. **Discover** the observed combinations, in a deterministic order.
//! 2. **Synthesize** a per-record index column (`__level_NAME`) on every
//!    subject, so the block is read by the ordinary gather machinery and the
//!    existing time-varying-covariate plumbing carries it — no parallel path.
//! 3. **Re-parse** the model with the level count known, so the compiled
//!    closures are built once against the real θ vector.
//!
//! Re-parsing costs milliseconds and is what keeps the alternative — rebuilding
//! `pk_param_fn` and every sibling closure in place after the fact — off the
//! table.

use std::collections::HashMap;

use crate::api::apply_iov_occasion_rule;
use crate::diagnostics::{Diagnostic, EngineError};
use crate::parser::model_parser::{
    eval_gather, level_index_column, parse_full_model_with, DataBindings, EtaCoupling, EtaRoute,
    LevelBinding, LevelBindings, LevelBlockDecl, LevelContrast, LevelRule, ParseBindings,
    ScaleShare,
};
use crate::types::{CompiledModel, IovOccasionRule, ParsedModel, Population, Subject};

/// The record time, addressable as a level column even though it is not a
/// covariate.
const TIME_COLUMN: &str = "TIME";

/// Bind every level block in `parsed` against `population`, mutating
/// both: the population gains the synthesized index columns, and `parsed.model`
/// is replaced by a re-parse that knows the level counts.
///
/// A no-op (and no re-parse) for the overwhelming majority of models, which
/// declare no level block at all.
///
/// **Re-binding** (#1730). The blocks are read from what `model_text` declares, so
/// a `parsed` already bound to other data binds on `population` exactly as a fresh
/// parse would: the contrast an `auto` block resolved to there is not carried over.
/// Covariate statistics `parsed` was bound with are kept only when `population`
/// gives the same summary of every covariate they cover. Otherwise they are dropped,
/// and the relations are left for [`bind_covariate_stats`](crate::api::bind_covariate_stats)
/// to bind on `population`.
///
/// Refused, with nothing written, on a model laid out on a fit's bindings
/// ([`bind_from_fit`], [`layout_from_fit`]): binding it to this data's own levels
/// would read the fitted θ at other positions.
///
/// Every refusal carries the code `E_THETA_LEVEL_BINDING` on block `parameters`
/// (#1773), the code `ferx check --data` reports for it.
pub fn bind_theta_levels(
    parsed: &mut ParsedModel,
    model_text: &str,
    population: &mut Population,
) -> Result<(), EngineError> {
    bind_theta_levels_diag(parsed, model_text, population).map_err(EngineError::from_diagnostic)
}

/// [`bind_theta_levels`], refusing with the [`Diagnostic`] itself: what
/// `ferx check`'s binding step reads.
pub(crate) fn bind_theta_levels_diag(
    parsed: &mut ParsedModel,
    model_text: &str,
    population: &mut Population,
) -> Result<(), Diagnostic> {
    bind_levels_on_data(parsed, model_text, population).map_err(level_binding_error)
}

/// The diagnostic of a level block that cannot bind (#1773): to the data's own
/// levels, or to a fit's layout. The one place `E_THETA_LEVEL_BINDING` is assigned;
/// `ferx check`, the binders and the entry points calling them all read it here.
pub(crate) fn level_binding_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error("E_THETA_LEVEL_BINDING", message).with_block("parameters")
}

/// The body of [`bind_theta_levels`]; every refusal is the level half's.
fn bind_levels_on_data(
    parsed: &mut ParsedModel,
    model_text: &str,
    population: &mut Population,
) -> Result<(), String> {
    // Every level-block name and column is the declared one, whatever contrast a
    // binding stamped: the refusal reads them before a re-parse is spent.
    if parsed.model.bound_from_fit() && !parsed.model.theta_blocks().level_blocks().is_empty() {
        return Err(fit_bound_message(
            parsed.model.theta_blocks().level_blocks(),
        ));
    }
    let decls: Vec<LevelBlockDecl> = declared_model(parsed, model_text, Reads::Levels)?
        .theta_blocks()
        .level_blocks()
        .to_vec();
    if decls.is_empty() {
        return Ok(());
    }
    // The bindings this layout is re-parsed on: no level binding of `parsed`'s,
    // and its statistics only while they are still `population`'s.
    let mut base = unbound_bindings(parsed);
    if crate::api::covariate_stats::stats_hold_on(&parsed.bindings.covariate_stats, population) {
        base.covariate_stats = parsed.bindings.covariate_stats.clone();
    }

    // Pass 1: discover every block and write its index column. A level's index
    // is its position in the sorted level list whatever the contrast, so the
    // columns can be written before any contrast is resolved. They go onto a
    // scratch copy, so a refusal below leaves `population` untouched.
    let mut bound = population.clone();
    let mut discovered: Vec<Vec<Level>> = Vec::with_capacity(decls.len());
    let mut tables: Vec<Vec<(Level, usize)>> = Vec::with_capacity(decls.len());
    for decl in &decls {
        let levels = discover_levels(decl, &bound)?;
        let table: Vec<(Level, usize)> = levels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), i + 1))
            .collect();
        write_index_column(decl, &table, &mut bound.subjects)?;
        discovered.push(levels);
        tables.push(table);
    }

    // A kappa's unit is a subject-occasion, so the binder needs the occasions
    // `fit()` will use: a derived `iov_occasion` rule is applied on a copy, by
    // the same call `fit()` makes.
    let derived;
    let with_occasions: &Population =
        if parsed.model.n_kappa > 0 && parsed.fit_options.iov_occasion != IovOccasionRule::Column {
            let mut p = bound.clone();
            apply_iov_occasion_rule(
                &mut p,
                &parsed.fit_options.iov_occasion,
                parsed.fit_options.iov_column.is_some(),
                &mut Vec::new(),
            );
            derived = p;
            &derived
        } else {
            &bound
        };

    // Rule 1 (#1679): measure which levels the likelihood never reads. Whether
    // that costs the model anything depends on the contrast, so the verdict is
    // the contrast resolution's.
    let dead = measure_dead_levels(&base, model_text, &decls, &discovered, with_occasions)?;

    let mut bindings = LevelBindings::new();
    for ((decl, levels), dead) in decls.iter().zip(&discovered).zip(&dead) {
        let (contrast, groups) = resolve_contrast(decl, levels, dead, with_occasions)?;
        let binding = LevelBinding {
            labels: levels.iter().map(|l| l.label(decl.columns())).collect(),
            groups,
            contrast,
        };
        // #1672: the shapes the from-fit validation refuses are ones this binder
        // never writes. Asserted on its own output, so every level fixture in the
        // suite measures that claim instead of the refusal resting on it.
        debug_assert!(
            check_binding_shape(decl, &binding).is_ok(),
            "bind_theta_levels wrote a binding the from-fit validation refuses: {:?}",
            check_binding_shape(decl, &binding)
        );
        bindings.insert(decl.name().to_string(), binding);
    }

    let model_name = parsed.model.name.clone();
    // Re-parse with the statistics kept above, not just the level bindings: a
    // model that also declares `[covariate_model]` statistics (#1111) may have
    // had those bound on this data already, and re-parsing with the level
    // bindings alone would drop them.
    let mut all = base;
    all.levels = bindings;
    let rebound = parse_full_model_with(model_text, &all)?;
    // #1797: on the model as it will be fitted, so the measurement is the contrast the
    // block resolved to.
    refuse_read_unindexed(
        &rebound.model,
        &decls,
        &tables,
        population,
        LevelSource::Data,
    )?;
    parsed.bindings = all;
    parsed.model = rebound.model;
    parsed.model.name = model_name;
    *population = bound;
    Ok(())
}

/// Bind a simulation design against a **fit's** level bindings (#1614), so the
/// fit's θ vector can drive it.
///
/// `fitted` is the [`LevelBindings`] the fit was bound with — `parsed.bindings.levels`
/// right after [`bind_theta_levels`] ran on the fit data. Each design record gets the
/// index of its level *in the fit*, and the model is re-parsed with the fit's labels,
/// groups and resolved contrast, so the θ layout is the fit's by construction.
/// [`bind_theta_levels`] cannot be used for this: it re-discovers the levels from the
/// design, so a design whose combinations differ from the fit's — a subset, a
/// different time grid, a study the fit never saw, or only more subjects per study,
/// which can re-resolve the contrast — silently reads the fitted values at the wrong
/// positions.
///
/// A design level the fit never observed is refused, naming the block and every such
/// label: no θ was estimated for it, and any stand-in (zero, a group mean) would be a
/// modelling decision taken silently. On a block keyed on `TIME` that means the design
/// can only be simulated at the fit's observation times. Levels are matched by their
/// label (`STUDY=7,TIME=4`), so a design value differing from the fit's in its last
/// digit is a different level. The refusal names actions, not functions, since a
/// wrapper passes it through verbatim; from Rust, the second action it offers —
/// simulating the design on its own levels — is [`bind_theta_levels`] on the design,
/// with θ for the levels it discovers (the model's `default_params`, for example).
///
/// Also refused: `fitted` lacking a block the model declares, or carrying one it does
/// not, or listing a level of a block more than once. Nothing is written to `population`
/// unless every block binds.
///
/// The blocks are read from what `model_text` declares (#1730), so a `parsed` bound
/// to other data takes the fit's layout as a fresh parse would. Its covariate
/// statistics are kept as `parsed` holds them: this binder takes none from the fit,
/// so install the fit's first, or use [`bind_from_fit`], which binds both.
///
/// Every refusal carries `E_THETA_LEVEL_BINDING` on block `parameters` (#1773).
#[deprecated(
    since = "0.4.1",
    note = "use `bind_from_fit` with the fit's `data_bindings`, which also binds the \
            covariate statistics from the fit"
)]
pub fn bind_theta_levels_from_fit(
    parsed: &mut ParsedModel,
    model_text: &str,
    population: &mut Population,
    fitted: &LevelBindings,
) -> Result<(), EngineError> {
    bind_levels_from_fit_layout(parsed, model_text, population, fitted)
        .map_err(|e| EngineError::from_diagnostic(level_binding_error(e)))
}

/// The body of [`bind_theta_levels_from_fit`]; every refusal is the level half's.
fn bind_levels_from_fit_layout(
    parsed: &mut ParsedModel,
    model_text: &str,
    population: &mut Population,
    fitted: &LevelBindings,
) -> Result<(), String> {
    let decls: Vec<LevelBlockDecl> = declared_model(parsed, model_text, Reads::Levels)?
        .theta_blocks()
        .level_blocks()
        .to_vec();
    validate_fitted_levels(&decls, fitted)?;
    if decls.is_empty() {
        return Ok(());
    }
    let tables = fitted_level_tables(&decls, population, fitted)?;
    let model_name = parsed.model.name.clone();
    // The statistics are the caller's, as they always were for this binder: the
    // sequence it was documented with installs the fit's statistics by hand, and
    // nothing tells those apart from statistics bound to other data.
    let mut bindings = unbound_bindings(parsed);
    bindings.covariate_stats = parsed.bindings.covariate_stats.clone();
    bindings.levels = fitted.clone();
    let rebound = parse_full_model_with(model_text, &bindings)?;
    refuse_read_unindexed(
        &rebound.model,
        &decls,
        &tables,
        population,
        LevelSource::Fit,
    )?;
    for (decl, table) in decls.iter().zip(&tables) {
        write_index_column(decl, table, &mut population.subjects)?;
    }
    parsed.bindings = bindings;
    parsed.model = rebound.model;
    parsed.model.name = model_name;
    parsed.model.indiv_param_partials.bound_from_fit = true;
    Ok(())
}

/// Bind a model to a **fit's** data-derived bindings (#1619), so the fit's θ can
/// drive it on `population`: a simulation design, new data to predict, or the
/// fit's own data re-read.
///
/// `fitted` is the fit's [`DataBindings`] — `FitResult::data_bindings`, which a
/// `.fitrx` bundle carries too. Both halves are applied in one call:
///
/// - **level blocks**: each record of `population` gets the index of its level
///   *in the fit*, and the model is laid out with the fit's labels, groups and
///   resolved contrast, so the θ layout is the fit's by construction;
/// - **covariate statistics**: every symbolic centre (`center = median`, …)
///   resolves to the *fit's* value, never to `population`'s. A design of heavier
///   subjects than the fit's has a higher median weight, and centring on it would
///   move every covariate factor the fitted θ was estimated against.
///
/// [`layout_from_fit`] is the same call without the population: it lays the model
/// out and writes nothing.
///
/// After this call the model records that its bindings are the fit's (#1730), so
/// [`bind_covariate_stats`](crate::api::bind_covariate_stats) on it is a no-op — the
/// fitted θ was estimated against the fit's centres — and [`bind_theta_levels`]
/// refuses it, since the population's own levels would read the fitted θ at other
/// positions.
///
/// Refused, with nothing written to `population`:
///
/// - empty `fitted` on a model with a level block or a symbolic statistic: a fit
///   made before ferx recorded its bindings, or an older `.fitrx`. The bindings are
///   not re-discovered from the data, since a layout resolved today need not be the
///   one the fit was estimated with. What the model needs is read from `model_text`,
///   so this holds for a `parsed` already bound to other data too (#1686), whose
///   relations would otherwise keep that data's centres;
/// - level bindings that lack a block the model declares or carry one it does not,
///   list a level twice, split a contrast group, record `auto` as the contrast, or
///   whose labels and groups are not parallel;
/// - covariate statistics that lack a covariate a symbolic relation reads, or
///   carry one no relation reads;
/// - a `population` level the fit never observed, since no θ was estimated for it.
///
/// A refusal of the level half carries `E_THETA_LEVEL_BINDING` on block
/// `parameters`, one of the statistics half `E_COVARIATE_STATS_BINDING` on block
/// `covariate_model` (#1773): the codes `ferx check --data` reports for a binding
/// that fails on the data's own levels and statistics. An empty `fitted` on a model
/// needing both halves is the level code, the half checked first. A level the fit
/// never observed has no `ferx check` counterpart, since `check` binds data to its
/// own levels.
pub fn bind_from_fit(
    parsed: &mut ParsedModel,
    model_text: &str,
    population: &mut Population,
    fitted: &DataBindings,
) -> Result<(), EngineError> {
    bind_from_fit_on(parsed, model_text, Some(population), fitted)
        .map_err(EngineError::from_diagnostic)
}

/// [`bind_from_fit`], with the population optional: it is written to only when the
/// model declares a level block, so a caller holding a borrowed population of a
/// model without one (`run_sir` / `run_covariance`, #1622) need not copy it. `None`
/// on a model that declares a level block is refused.
///
/// Every step that can refuse — validation, the re-parse and the unseen-level
/// tables — runs before anything is written, so a refusal leaves both
/// `parsed` and the population as they were. The first two are
/// [`layout_from_fit`]'s, run by the same function.
pub(crate) fn bind_from_fit_on(
    parsed: &mut ParsedModel,
    model_text: &str,
    population: Option<&mut Population>,
    fitted: &DataBindings,
) -> Result<(), Diagnostic> {
    let Some(layout) = lay_out_on_fit(parsed, model_text, fitted)? else {
        return Ok(());
    };
    let tables = match (&population, layout.decls.is_empty()) {
        (_, true) => Vec::new(),
        (Some(p), false) => {
            fitted_level_tables(&layout.decls, p, &fitted.levels).map_err(level_binding_error)?
        }
        (None, false) => {
            return Err(level_binding_error(format!(
                "theta {}: a level block needs the population its index columns are written to",
                layout.decls[0].name()
            )))
        }
    };
    if let Some(population) = population {
        refuse_read_unindexed(
            &layout.model,
            &layout.decls,
            &tables,
            population,
            LevelSource::Fit,
        )
        .map_err(level_binding_error)?;
        for (decl, table) in layout.decls.iter().zip(&tables) {
            write_index_column(decl, table, &mut population.subjects)
                .map_err(level_binding_error)?;
        }
    }
    layout.apply(parsed);
    Ok(())
}

/// Lay a model out on a **fit's** data-derived bindings (#1703), with no population:
/// the θ vector, names and `FIX` flags the fit was estimated with, for a caller that
/// needs the model's shape but holds no data to bind — sizing a skeleton
/// `FitResult` for SIR or a standalone covariance step, for instance.
///
/// This is the model half of [`bind_from_fit`], and `bind_from_fit` runs it: the
/// same refusals and the same re-parse. What it leaves out is the population half, so
/// nothing is written to any population and a level absent from the fit is not
/// looked for. Before running the model on data, bind that data with
/// [`bind_from_fit`] instead: a level block reads its index columns from the
/// population, and only `bind_from_fit` writes them.
///
/// Whether the model needs the fit's bindings is decided from `model_text`, not from
/// `parsed`, so a model already bound to other data (a simulation design, by
/// [`prepare_run`](crate::api::prepare_run) or
/// [`bind_covariate_stats`](crate::api::bind_covariate_stats)) is laid out on the
/// fit's statistics, or refused when the fit carries none (#1686). Refused, with
/// `parsed` left as it was: everything [`bind_from_fit`] refuses except a level the
/// fit never observed, with the same codes (#1773).
pub fn layout_from_fit(
    parsed: &mut ParsedModel,
    model_text: &str,
    fitted: &DataBindings,
) -> Result<(), EngineError> {
    if let Some(layout) =
        lay_out_on_fit(parsed, model_text, fitted).map_err(EngineError::from_diagnostic)?
    {
        layout.apply(parsed);
    }
    Ok(())
}

/// The half of a model's data-derived bindings a binder reads the declaration of.
#[derive(Clone, Copy)]
pub(crate) enum Reads {
    /// The level blocks: their columns and declared contrast.
    Levels,
    /// The `[covariate_model]` relations: which are symbolic, and on what.
    Stats,
    /// Both halves.
    Both,
}

/// What a model declares, as one of [`declared_model`]'s two sources.
pub(crate) enum Declared<'a> {
    /// `parsed.model` itself: nothing was bound that changes what is read.
    Borrowed(&'a CompiledModel),
    /// An unbound re-parse of the model text.
    Reparsed(Box<CompiledModel>),
}

impl std::ops::Deref for Declared<'_> {
    type Target = CompiledModel;
    fn deref(&self) -> &CompiledModel {
        match self {
            Declared::Borrowed(m) => m,
            Declared::Reparsed(m) => m,
        }
    }
}

/// `parsed`'s bindings with every data-derived half removed: its `[priors]`
/// directory kept, its levels and covariate statistics empty.
pub(crate) fn unbound_bindings(parsed: &ParsedModel) -> ParseBindings {
    ParseBindings {
        levels: LevelBindings::new(),
        covariate_stats: Default::default(),
        ..parsed.bindings.clone()
    }
}

/// What `model_text` declares for the half `reads` (#1686, #1730), the one source
/// every binder reads its declaration from. A model bound to data already has its
/// relations resolved and its `auto` contrasts stamped, so read as it stands it
/// "declares" that data's resolution. Such a model is parsed again with no
/// data-derived binding.
///
/// A model with that half unbound *is* the unbound parse for it, and is borrowed:
/// no second parse, and no second read of a `[priors] from_fit` file. The halves
/// are independent — a level binding resolves no relation and a statistics binding
/// stamps no contrast — so the second binder of `prepare_run`'s pair (levels, then
/// statistics) borrows too.
pub(crate) fn declared_model<'a>(
    parsed: &'a ParsedModel,
    model_text: &str,
    reads: Reads,
) -> Result<Declared<'a>, String> {
    let bound = parsed.model.data_bindings();
    let unbound = match reads {
        Reads::Levels => bound.levels.is_empty(),
        Reads::Stats => bound.covariate_stats.is_empty(),
        Reads::Both => bound.is_empty(),
    };
    if unbound {
        return Ok(Declared::Borrowed(&parsed.model));
    }
    let model = parse_full_model_with(model_text, &unbound_bindings(parsed))?.model;
    Ok(Declared::Reparsed(Box::new(model)))
}

/// Whether level bindings `model` was bound with are still `population`'s
/// (#1730, #1735 review r1 finding 1): for every block, each subject carries the
/// index column, and the levels `population` shows are exactly the bound labels.
/// A level's index is its position in that sorted list, so equal labels mean the
/// column was written for this layout. A column present but written for other
/// levels, or a block `population` cannot show, counts as differing.
pub(crate) fn levels_hold_on(
    model: &CompiledModel,
    levels: &LevelBindings,
    population: &Population,
) -> bool {
    let decls = model.theta_blocks().level_blocks();
    levels.iter().all(|(name, binding)| {
        let Some(decl) = decls.iter().find(|d| d.name() == name) else {
            return false;
        };
        let column = level_index_column(name);
        population
            .subjects
            .iter()
            .all(|s| s.covariates.contains_key(&column))
            && discover_levels(decl, population).is_ok_and(|found| {
                found
                    .iter()
                    .map(|l| l.label(decl.columns()))
                    .eq(binding.labels.iter().cloned())
            })
    })
}

/// A model re-parsed on a fit's bindings, not yet installed in its `ParsedModel`.
struct FitLayout {
    /// The level blocks `model_text` declares.
    decls: Vec<LevelBlockDecl>,
    model: CompiledModel,
    bindings: ParseBindings,
}

impl FitLayout {
    fn apply(self, parsed: &mut ParsedModel) {
        let mut model = self.model;
        model.name = parsed.model.name.clone();
        model.indiv_param_partials.bound_from_fit = true;
        parsed.model = model;
        parsed.bindings = self.bindings;
    }
}

/// The one implementation of the model half of [`bind_from_fit`] and
/// [`layout_from_fit`]: validate `fitted` against the model and re-parse on it. `None` when the model declares nothing data-derived and
/// the fit carries nothing, which leaves `parsed` as it is. Reads `parsed` only for
/// its non-data bindings and name, and writes nothing.
fn lay_out_on_fit(
    parsed: &ParsedModel,
    model_text: &str,
    fitted: &DataBindings,
) -> Result<Option<FitLayout>, Diagnostic> {
    use crate::api::covariate_stats::stats_binding_error;
    // What the model needs comes from its text, parsed with no data-derived binding
    // (#1686): both halves are read here.
    // A refusal that is not one half's alone is the level half's when the model has
    // one: the half every binder sequence checks first (#1773). Before the
    // declaration is read, `parsed`'s own blocks say whether it has one.
    let either_on = |has_levels: bool, message: String| {
        if has_levels {
            level_binding_error(message)
        } else {
            stats_binding_error(message)
        }
    };
    let has_levels = !parsed.model.theta_blocks().level_blocks().is_empty();
    let declared = declared_model(parsed, model_text, Reads::Both)
        .map_err(|message| either_on(has_levels, message))?;
    let decls: Vec<LevelBlockDecl> = declared.theta_blocks().level_blocks().to_vec();
    let symbolic = crate::api::covariate_stats::symbolic_covariates(&declared);
    let either = |message: String| either_on(!decls.is_empty(), message);
    if fitted.is_empty() && (!decls.is_empty() || !symbolic.is_empty()) {
        return Err(either(no_fit_bindings_message(&decls, &symbolic)));
    }
    validate_fitted_levels(&decls, &fitted.levels).map_err(level_binding_error)?;
    crate::api::covariate_stats::validate_fitted_stats(&declared, &fitted.covariate_stats)
        .map_err(stats_binding_error)?;
    if decls.is_empty() && fitted.covariate_stats.is_empty() {
        return Ok(None);
    }
    let mut bindings = unbound_bindings(parsed);
    bindings.levels = fitted.levels.clone();
    bindings.covariate_stats = fitted.covariate_stats.clone();
    let model = parse_full_model_with(model_text, &bindings)
        .map_err(either)?
        .model;
    // Not a gate (#1728 review): `validate_fitted_stats` has already required a
    // statistic for every covariate a symbolic relation reads, and given one, the
    // parser resolves the relation or refuses it. Asserted on every from-fit fixture
    // so that claim is measured; it would fire if a relation came to need more
    // than a `CovariateSummary` to resolve.
    debug_assert!(
        crate::api::assert_covariate_model_bound(&model).is_ok(),
        "a from-fit layout left a relation unresolved: {:?}",
        crate::api::assert_covariate_model_bound(&model)
    );
    Ok(Some(FitLayout {
        decls,
        model,
        bindings,
    }))
}

/// Write the level index columns of a model **already** bound to a fit's level
/// bindings onto a population re-read for it: the `run_sir` / `run_covariance`
/// cell where the caller supplies the bound model but not the population (#1622).
/// The validation and unseen-level refusal of [`bind_from_fit`]; no re-parse, since
/// the model in hand already carries the layout.
pub(crate) fn write_fitted_level_columns(
    model: &crate::types::CompiledModel,
    population: &mut Population,
    fitted: &LevelBindings,
) -> Result<(), Diagnostic> {
    let decls: Vec<LevelBlockDecl> = model.theta_blocks().level_blocks().to_vec();
    (|| -> Result<(), String> {
        validate_fitted_levels(&decls, fitted)?;
        let tables = fitted_level_tables(&decls, population, fitted)?;
        refuse_read_unindexed(model, &decls, &tables, population, LevelSource::Fit)?;
        for (decl, table) in decls.iter().zip(&tables) {
            write_index_column(decl, table, &mut population.subjects)?;
        }
        Ok(())
    })()
    .map_err(level_binding_error)
}

/// The refusal for a fit that carries no data-derived bindings at all, on a model
/// that needs them (#1619). One clause per half the model actually has, so a
/// level-only model hears nothing about statistics and the reverse.
fn no_fit_bindings_message(decls: &[LevelBlockDecl], symbolic: &[String]) -> String {
    let mut needs: Vec<String> = Vec::new();
    if !decls.is_empty() {
        needs.push(format!(
            "its theta level block(s) {} take their level layout from the data it was \
             fitted on",
            decls
                .iter()
                .map(|d| format!("`{}[{}]`", d.name(), d.columns().join(", ")))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !symbolic.is_empty() {
        needs.push(format!(
            "its [covariate_model] relations state a statistic of {} symbolically, so their \
             centres come from the data it was fitted on",
            symbolic
                .iter()
                .map(|c| format!("`{c}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    format!(
        "this fit carries no data-derived bindings, so the model cannot be rebuilt the way \
         it was fitted: {}. The fit is an older `.fitrx` bundle, or was made before ferx \
         recorded these bindings with a fit. Refit the model to record them.",
        needs.join(", and ")
    )
}

/// The refusal of [`bind_theta_levels`] on a model laid out on a fit's bindings
/// (#1730). It names actions, not functions, since a wrapper passes it through
/// verbatim; it does not say the data's levels differ from the fit's, which is not
/// checked and is false on the fit's own data.
fn fit_bound_message(decls: &[LevelBlockDecl]) -> String {
    format!(
        "{}: this model is laid out on a fit's levels, so binding it to the levels of \
         this data would read the fitted theta at other positions. To run this data on \
         the fit's theta, bind it from the fit's bindings instead; to use this data's \
         own levels, parse the model again and bind the new parse.",
        decls
            .iter()
            .map(|d| format!("theta {}[{}]", d.name(), d.columns().join(", ")))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Check a fit's level bindings against the blocks the model declares, before
/// anything is written (#1621, #1672). A `.fitrx` bundle carries its bindings as
/// data a caller can edit, so every shape a fit could not have written (a θ the fit
/// never had, a contrast nobody resolved or the block does not declare, a split
/// group) is refused here, with the fault placed in the bindings. Some of these the
/// re-parse would accept silently; for the others this gives the message, and the
/// guarantee that nothing was written first.
fn validate_fitted_levels(decls: &[LevelBlockDecl], fitted: &LevelBindings) -> Result<(), String> {
    let mut extra: Vec<&str> = fitted
        .keys()
        .filter(|name| !decls.iter().any(|d| d.name() == name.as_str()))
        .map(String::as_str)
        .collect();
    if !extra.is_empty() {
        extra.sort_unstable();
        return Err(format!(
            "the fit's level bindings carry the block(s) {}, which this model does not \
             declare: the bindings belong to a different model",
            extra
                .iter()
                .map(|b| format!("`{b}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    for decl in decls {
        let binding = fitted.get(decl.name()).ok_or_else(|| {
            format!(
                "theta {}[{}]: the fit's level bindings carry no `{}`, so there is no fitted \
                 layout to bind the design against (was the model edited since the fit?)",
                decl.name(),
                decl.columns().join(", "),
                decl.name()
            )
        })?;
        if binding.groups.len() != binding.labels.len() {
            return Err(format!(
                "theta {}[{}]: the fit's level binding has {} labels but {} groups; \
                 they must be parallel",
                decl.name(),
                decl.columns().join(", "),
                binding.labels.len(),
                binding.groups.len()
            ));
        }
        let repeated = repeated_labels(&binding.labels);
        if !repeated.is_empty() {
            return Err(repeated_labels_message(decl, &repeated));
        }
        check_binding_shape(decl, binding)?;
    }
    Ok(())
}

/// The shapes a binding ferx wrote can never have (#1672): an unresolved `auto`
/// contrast, a contrast other than the one the block declares, and a contrast group
/// whose levels are not contiguous. Shared by the from-fit validation and by a
/// `debug_assert!` on [`bind_theta_levels`]'s own output, which is what measures the
/// claim that ferx never writes any of them.
///
/// The declared-contrast check is one the re-parse cannot make for itself: `ref` and
/// `sum_to_zero` free the same number of θ, so a swapped contrast lays out a θ vector
/// of the right length and reads the fitted values under the other convention (#1680
/// review). A split group the parser refuses too; the check here gives the message,
/// and the guarantee that nothing was written first.
fn check_binding_shape(decl: &LevelBlockDecl, binding: &LevelBinding) -> Result<(), String> {
    if binding.contrast == LevelContrast::Auto {
        return Err(format!(
            "theta {}[{}]: the fit's level bindings are malformed: they record the contrast \
             `auto`, and a fit never records `auto`, only the contrast it resolved to.",
            decl.name(),
            decl.columns().join(", ")
        ));
    }
    if decl.contrast() != LevelContrast::Auto && binding.contrast != decl.contrast() {
        return Err(format!(
            "theta {}[{}]: the fit's level bindings record the contrast `{}`, but this block \
             declares `contrast = {}`. A fit records the contrast its block declares, so the \
             bindings are malformed or belong to a different model.",
            decl.name(),
            decl.columns().join(", "),
            contrast_token(binding.contrast),
            contrast_token(decl.contrast())
        ));
    }
    let split =
        binding.groups.iter().enumerate().find(|&(i, g)| {
            i > 0 && binding.groups[i - 1] != *g && binding.groups[..i].contains(g)
        });
    if let Some((_, g)) = split {
        return Err(format!(
            "theta {}[{}]: the fit's level bindings are malformed: the levels of contrast \
             group {g} are split. A fit records each group's levels contiguously, since a \
             group's free theta occupy one contiguous range.",
            decl.name(),
            decl.columns().join(", ")
        ));
    }
    Ok(())
}

/// Each `population` level's index in the fit, per block, refusing a level the fit
/// never observed. Computed for every block before any column is written, so a
/// refusal leaves `population` untouched. `fitted` has passed
/// [`validate_fitted_levels`], so every block has an entry.
fn fitted_level_tables(
    decls: &[LevelBlockDecl],
    population: &Population,
    fitted: &LevelBindings,
) -> Result<Vec<Vec<(Level, usize)>>, String> {
    let mut tables: Vec<Vec<(Level, usize)>> = Vec::with_capacity(decls.len());
    for decl in decls {
        let (table, unseen) = match_level_table(decl, population, &fitted[decl.name()].labels)?;
        if !unseen.is_empty() {
            return Err(unseen_levels_message(decl, &unseen));
        }
        tables.push(table);
    }
    Ok(tables)
}

/// The one label matcher (#1762): each level `population` shows for `decl`, with its
/// 1-based position in `labels`, and the label of every level `labels` lacks, in level
/// order. [`bind_from_fit`] writes a population's index columns from the table, and
/// [`level_index_finding`] re-derives from it the index each record should carry, so
/// the binder and the check cannot disagree about what a label is.
fn match_level_table(
    decl: &LevelBlockDecl,
    population: &Population,
    labels: &[String],
) -> Result<(Vec<(Level, usize)>, Vec<String>), String> {
    // Keyed, not `position`: at MBMA scale (thousands of levels) a linear search per level
    // made the check on every run call quadratic (#1762, measured). A repeated label keeps
    // its first position, as `position` did.
    let mut at: HashMap<&str, usize> = HashMap::with_capacity(labels.len());
    for (i, l) in labels.iter().enumerate() {
        at.entry(l.as_str()).or_insert(i + 1);
    }
    let mut table = Vec::new();
    let mut unseen = Vec::new();
    for level in discover_levels(decl, population)? {
        let label = level.label(decl.columns());
        match at.get(label.as_str()) {
            Some(&i) => table.push((level, i)),
            None => unseen.push(label),
        }
    }
    Ok((table, unseen))
}

/// The hash key of a level's values: their bits, with `-0.0` folded onto `0.0` so two
/// keys are equal exactly when the values are `==` (every value is finite here).
fn level_key(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| (v + 0.0).to_bits()).collect()
}

/// Why a population's index column for one bound level block is not the index the
/// model's level table gives its records (#1762). The first that applies, in this
/// order; a population no subject of which carries the column is the caller's
/// (`check_level_index_columns`) to report, before this is asked.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LevelIndexFinding {
    /// `unbound` of the population's `of` subjects carry no index column; `first_id`
    /// is the first of them.
    SomeSubjectsUnbound {
        unbound: usize,
        of: usize,
        first_id: String,
    },
    /// Subject `id` carries the index but no finite value in the level column
    /// `column`, so the level its index names cannot be checked.
    ColumnMissing { id: String, column: String },
    /// Levels the population shows that the model's table does not hold, every one,
    /// in level order.
    Unseen { labels: Vec<String> },
    /// The first record whose index is not its label's position in the model's table:
    /// subject `id` at record time `time` is level `label`, position `want`, and
    /// carries `got` (`None`: no index on that record). `time` is `None` for a subject
    /// with no Gaussian observation, whose level is its baseline columns' (#1797).
    Misindexed {
        id: String,
        time: Option<f64>,
        label: String,
        want: usize,
        got: Option<f64>,
    },
}

/// Whether `population`'s index column for the bound block `decl` is the index the
/// model's own level table (`decl.labels()`, the binding the model was parsed with)
/// gives each record (#1762). `None` when it is.
///
/// The level columns are the stamp: [`write_index_column`] derives each record's index
/// from the record's own level values, so the index is re-derived here the same way,
/// through the binder's own label matcher ([`match_level_table`]), and compared with the
/// index the predictors read on that record (`Subject::obs_cov`: the record's snapshot,
/// else the subject's baseline). The Gaussian observation records are compared: they
/// define the block's levels. So is every subject with no Gaussian observation whose
/// baseline columns name a level of the table (#1797), against its baseline index, which
/// is what its likelihood reads; one whose columns name no level is skipped, since the
/// binder decided it (it binds at index 1 only when nothing it is scored on reads the
/// block). A dose, `EVID=2` or reset record carries an LOCF copy written by the same
/// binder and is not re-checked, so an edit to only such a snapshot is out of this
/// check's reach.
pub(crate) fn level_index_finding(
    decl: &LevelBlockDecl,
    population: &Population,
) -> Option<LevelIndexFinding> {
    let column = level_index_column(decl.name());
    let unbound: Vec<&Subject> = population
        .subjects
        .iter()
        .filter(|s| !s.covariates.contains_key(&column))
        .collect();
    if let Some(first) = unbound.first() {
        return Some(LevelIndexFinding::SomeSubjectsUnbound {
            unbound: unbound.len(),
            of: population.subjects.len(),
            first_id: first.id.clone(),
        });
    }
    for subject in &population.subjects {
        for j in 0..subject.obs_times.len() {
            for c in decl.columns() {
                if !column_value(subject, c, j).is_some_and(f64::is_finite) {
                    return Some(LevelIndexFinding::ColumnMissing {
                        id: subject.id.clone(),
                        column: c.clone(),
                    });
                }
            }
        }
    }
    if population.subjects.iter().all(|s| s.obs_times.is_empty()) {
        // No record defines a level, so no index can be wrong.
        return None;
    }
    if let Some(finding) = no_gaussian_misindexed(decl, population, &column) {
        return Some(finding);
    }
    let (table, unseen) = match_level_table(decl, population, decl.labels())
        .expect("every level column is present and finite on every record");
    if !unseen.is_empty() {
        return Some(LevelIndexFinding::Unseen { labels: unseen });
    }
    let index: HashMap<Vec<u64>, (usize, &Level)> = table
        .iter()
        .map(|(level, i)| (level_key(&level.values), (*i, level)))
        .collect();
    for subject in &population.subjects {
        for j in 0..subject.obs_times.len() {
            let values: Vec<f64> = decl
                .columns()
                .iter()
                .map(|c| column_value(subject, c, j).unwrap_or(f64::NAN))
                .collect();
            let &(want, level) = index
                .get(&level_key(&values))
                .expect("the table holds every level the population shows");
            let got = subject.obs_cov(j).get(&column).copied();
            if got != Some(want as f64) {
                return Some(LevelIndexFinding::Misindexed {
                    id: subject.id.clone(),
                    time: Some(
                        subject
                            .obs_raw_times
                            .get(j)
                            .copied()
                            .unwrap_or(subject.obs_times[j]),
                    ),
                    label: level.label(decl.columns()),
                    want,
                    got,
                });
            }
        }
    }
    None
}

/// [`level_index_finding`]'s arm for subjects with no Gaussian observation (#1797): the
/// first whose baseline columns name a level of the model's table but whose index is not
/// that level's position. The level is matched by label, as [`match_level_table`] does.
fn no_gaussian_misindexed(
    decl: &LevelBlockDecl,
    population: &Population,
    column: &str,
) -> Option<LevelIndexFinding> {
    for subject in population
        .subjects
        .iter()
        .filter(|s| s.obs_times.is_empty())
    {
        let Ok(values) = baseline_level(decl, subject) else {
            continue;
        };
        let label = Level { values }.label(decl.columns());
        let Some(want) = decl
            .labels()
            .iter()
            .position(|l| *l == label)
            .map(|i| i + 1)
        else {
            continue;
        };
        let got = subject.covariates.get(column).copied();
        if got != Some(want as f64) {
            return Some(LevelIndexFinding::Misindexed {
                id: subject.id.clone(),
                time: None,
                label,
                want,
                got,
            });
        }
    }
    None
}

/// Each label that occurs more than once in `labels`, once, in order of first
/// occurrence.
fn repeated_labels(labels: &[String]) -> Vec<&str> {
    let mut repeated: Vec<&str> = Vec::new();
    for (i, label) in labels.iter().enumerate() {
        if labels[..i].contains(label) && !repeated.contains(&label.as_str()) {
            repeated.push(label);
        }
    }
    repeated
}

/// The refusal for a fitted binding that lists a level more than once (#1621).
/// A `.fitrx` bundle carries its bindings as data a caller can edit, and a
/// repeated label otherwise binds: the re-parse lays out one θ per *label*, so
/// the model grows a θ the fit never had and the design reads the first copy.
/// The fault is in the bindings, not the design, so the message says nothing
/// about the design's levels.
fn repeated_labels_message(decl: &LevelBlockDecl, repeated: &[&str]) -> String {
    format!(
        "theta {}[{}]: the fit's level bindings list {} level(s) more than once: {}. \
         Each level has exactly one fitted theta, so the bindings are malformed — they \
         are not the ones the fit recorded.",
        decl.name(),
        decl.columns().join(", "),
        repeated.len(),
        repeated
            .iter()
            .map(|l| format!("`{l}`"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// The refusal for design levels the fit never observed. Every label is listed — a
/// caller (the R wrapper) passes the text through verbatim, and a list cut short would
/// leave the user guessing which records to drop.
fn unseen_levels_message(decl: &LevelBlockDecl, unseen: &[String]) -> String {
    let mut message = format!(
        "theta {}[{}]: the design has {} level(s) the fit estimated no theta for: {}. \
         A level's theta exists only for a combination the fit's data observed.",
        decl.name(),
        decl.columns().join(", "),
        unseen.len(),
        unseen
            .iter()
            .map(|l| format!("`{l}`"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if decl
        .columns()
        .iter()
        .any(|c| c.eq_ignore_ascii_case(TIME_COLUMN))
    {
        message.push_str(&format!(
            " `{TIME_COLUMN}` is a level column of this block, so the design can only be \
             simulated at the fit's observation times; a denser or different time grid has \
             no fitted theta."
        ));
    }
    message.push_str(
        " Either simulate only the fit's levels, or simulate the design without the fit's \
         theta, from a theta vector for the design's own levels (the model's initial \
         estimates, for example).",
    );
    message
}

/// A level: the tuple of column values that defines it.
#[derive(Debug, Clone, PartialEq)]
struct Level {
    values: Vec<f64>,
}

impl Level {
    /// `STUDY=7,TIME=4` — the label the θ is reported under.
    fn label(&self, columns: &[String]) -> String {
        columns
            .iter()
            .zip(&self.values)
            .map(|(c, v)| format!("{c}={}", format_level_value(*v)))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The leading columns' values — the sum-to-zero grouping key when the
    /// block is nested inside a random effect.
    fn leading(&self) -> &[f64] {
        &self.values[..self.values.len().saturating_sub(1)]
    }
}

/// Render a level value without a trailing `.0` on integers, which is what
/// study ids and visit numbers almost always are.
fn format_level_value(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// Compare two level tuples lexicographically. `total_cmp` rather than
/// `partial_cmp` so the order is total even if a column carries a NaN — the
/// binding must be reproducible run to run.
fn cmp_levels(a: &Level, b: &Level) -> std::cmp::Ordering {
    for (x, y) in a.values.iter().zip(&b.values) {
        let ord = x.total_cmp(y);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    std::cmp::Ordering::Equal
}

/// The value of one level column on observation row `j` of `subject`.
fn column_value(subject: &Subject, column: &str, j: usize) -> Option<f64> {
    if column.eq_ignore_ascii_case(TIME_COLUMN) {
        return subject.obs_times.get(j).copied();
    }
    subject
        .obs_covariates
        .get(j)
        .and_then(|m| m.get(column))
        .or_else(|| subject.covariates.get(column))
        .copied()
}

/// The observed level combinations, sorted so each group's levels are
/// contiguous and the binding is reproducible.
fn discover_levels(decl: &LevelBlockDecl, population: &Population) -> Result<Vec<Level>, String> {
    let mut levels: Vec<Level> = Vec::new();
    // Deduplicated by key rather than `Vec::contains`, which was O(records × levels) and
    // ran on every call through the level-table check (#1762). The first-seen values of a
    // level are kept, as before.
    let mut seen: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();
    for subject in &population.subjects {
        for j in 0..subject.obs_times.len() {
            let mut values = Vec::with_capacity(decl.columns().len());
            for column in decl.columns() {
                let v = column_value(subject, column, j).ok_or_else(|| {
                    format!(
                        "theta {}[...]: column `{column}` is not in the data \
                         (subject {})",
                        decl.name(),
                        subject.id
                    )
                })?;
                if !v.is_finite() {
                    return Err(format!(
                        "theta {}[...]: column `{column}` is non-finite on subject {}",
                        decl.name(),
                        subject.id
                    ));
                }
                values.push(v);
            }
            if seen.insert(level_key(&values)) {
                levels.push(Level { values });
            }
        }
    }
    if levels.is_empty() {
        return Err(format!(
            "theta {}[{}]: the data carries no observation rows, so the block \
             has no levels",
            decl.name(),
            decl.columns().join(", ")
        ));
    }
    levels.sort_by(cmp_levels);
    Ok(levels)
}

/// The columns that would key a block's levels to subjects: the leading ones,
/// or for a one-column block the column itself.
fn subject_key_columns(decl: &LevelBlockDecl) -> &[String] {
    match decl.columns().len() {
        1 => decl.columns(),
        n => &decl.columns()[..n - 1],
    }
}

/// The unit a random effect varies over: a subject for an η, an occasion of
/// one subject for a kappa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unit {
    Subject,
    Occasion,
}

/// The unit observation row `j` of `subject` lies in, within that subject:
/// the subject itself for [`Unit::Subject`], the row's occasion for
/// [`Unit::Occasion`] — `None` when the subject carries no occasions, so a
/// kappa counts nowhere (`fit()` reports the missing occasions later).
fn unit_of(subject: &Subject, unit: Unit, j: usize) -> Option<u32> {
    match unit {
        Unit::Subject => Some(0),
        Unit::Occasion => subject.occasions.get(j).copied(),
    }
}

/// Whether every level's observation records lie in a single unit (#1678,
/// #1696): one subject for an η, one occasion of one subject for a kappa.
///
/// This is the data-side half of "is there a random effect at a grouping
/// coarser than or equal to the block's": only then can a unit's random effect
/// carry its levels (#1064). For a one-column block it means each level is one
/// unit's (#1649). It does not ask the leading columns to identify subjects:
/// two subjects of one study on disjoint times nest a `[STUDY, TIME]` block in
/// subjects while sharing its `STUDY` key.
fn levels_nest_in_units(decl: &LevelBlockDecl, population: &Population, unit: Unit) -> bool {
    let mut owner: HashMap<Vec<u64>, (usize, u32)> = HashMap::new();
    for (s, subject) in population.subjects.iter().enumerate() {
        for j in 0..subject.obs_times.len() {
            let level: Option<Vec<u64>> = decl
                .columns()
                .iter()
                .map(|c| column_value(subject, c, j).map(f64::to_bits))
                .collect();
            let (Some(level), Some(u)) = (level, unit_of(subject, unit, j)) else {
                return false;
            };
            if *owner.entry(level).or_insert((s, u)) != (s, u) {
                return false;
            }
        }
    }
    true
}

/// Whether a block of two or more columns resolves every subject's
/// observations: no level holds two different observation times of one
/// subject (#1650). Replicates at one time still resolve, since a random
/// effect's effect is the same at both.
///
/// Such a block is a free curve per subject at the observed times, so it
/// reproduces any per-subject effect at all, whatever route a random effect
/// takes to `y`.
fn resolves_observations(decl: &LevelBlockDecl, population: &Population) -> bool {
    if decl.columns().len() < 2 {
        return false;
    }
    population.subjects.iter().all(|subject| {
        let mut seen: Vec<(Vec<f64>, f64)> = Vec::new();
        (0..subject.obs_times.len()).all(|j| {
            let level: Vec<f64> = decl
                .columns()
                .iter()
                .filter_map(|c| column_value(subject, c, j))
                .collect();
            let t = subject.obs_times[j];
            match seen.iter().find(|(l, _)| *l == level) {
                Some((_, t0)) => *t0 == t,
                None => {
                    seen.push((level, t));
                    true
                }
            }
        })
    })
}

/// Whether every covariate in `covariates` is constant within every unit's
/// observation records — the data-side condition of a funnel
/// (`EtaCoupling::funnels`): within each subject for an η, within each
/// occasion of a subject for a kappa.
fn constant_within_units(covariates: &[String], population: &Population, unit: Unit) -> bool {
    population.subjects.iter().all(|subject| {
        covariates.iter().all(|c| {
            // Missing compares like any other value: a column absent from every
            // record is constant, one present on only some records changes.
            let mut first: Vec<(u32, Option<f64>)> = Vec::new();
            (0..subject.obs_times.len()).all(|j| {
                let Some(u) = unit_of(subject, unit, j) else {
                    return false;
                };
                let v = column_value(subject, c, j);
                match first.iter().find(|(u0, _)| *u0 == u) {
                    Some((_, v0)) => *v0 == v,
                    None => {
                        first.push((u, v));
                        true
                    }
                }
            })
        })
    })
}

/// A random effect a block absorbs, and where the binder found it, for its
/// diagnostics.
struct Absorbed<'a> {
    coupling: &'a EtaCoupling,
    /// The parameters that read the random effect directly
    /// (`LevelBlockDecl::eta_readers`).
    readers: &'a [String],
    how: How<'a>,
}

enum How<'a> {
    /// An expression that reads the block and the random effect.
    Site(&'a ScaleShare),
    /// The random effect reaching `y` by a route that never meets the block, on
    /// a block that resolves the observations.
    Route(&'a EtaRoute),
}

impl Absorbed<'_> {
    fn name(&self) -> &str {
        &self.coupling.eta
    }

    fn kappa(&self) -> bool {
        self.coupling.kappa
    }

    fn clause(&self) -> String {
        match &self.how {
            How::Site(share) => share_site(share),
            How::Route(route) => {
                let how = match route {
                    EtaRoute::Direct => "directly".to_string(),
                    EtaRoute::Via(v) => format!("through `{v}`"),
                    EtaRoute::State => "through the model's states".to_string(),
                };
                format!("the random effect `{}` reaches `y` {how}", self.name())
            }
        }
    }
}

/// Every random effect the block absorbs (#1649, #1678, #1696): the block's
/// levels nest in the random effect's units ([`levels_nest_in_units`]), and
/// either a funnel holds on this data (its covariates constant within those
/// units) or the block resolves the observations and the random effect reaches
/// `y` at all. An η and a kappa follow the same rule, each on its own unit.
fn absorbed<'a>(decl: &'a LevelBlockDecl, population: &Population) -> Vec<Absorbed<'a>> {
    let has = |kappa: bool| decl.eta_couplings.iter().any(|c| c.kappa == kappa);
    let by_subject = has(false) && levels_nest_in_units(decl, population, Unit::Subject);
    let by_occasion = has(true) && levels_nest_in_units(decl, population, Unit::Occasion);
    let resolving = resolves_observations(decl, population);
    decl.eta_couplings
        .iter()
        .enumerate()
        .filter_map(|(k, c)| {
            let (unit, nested) = if c.kappa {
                (Unit::Occasion, by_occasion)
            } else {
                (Unit::Subject, by_subject)
            };
            if !nested {
                return None;
            }
            let funnel = c
                .funnels
                .iter()
                .find(|f| constant_within_units(&f.covariates, population, unit));
            let how = match funnel {
                Some(f) => How::Site(&f.site),
                None => {
                    let route = c.reach.as_ref().filter(|_| resolving)?;
                    match &c.share {
                        Some(share) => How::Site(share),
                        None => How::Route(route),
                    }
                }
            };
            let readers = decl.eta_readers.get(k).map_or(&[][..], |r| r.as_slice());
            Some(Absorbed {
                coupling: c,
                readers,
                how,
            })
        })
        .collect()
}

/// Whether the dead-level check reads every channel of the model's likelihood:
/// the per-record predictions and residual magnitudes. A hazard, a Markov or
/// binary endpoint, a mixture, or an EKF diffusion term reads the block through
/// something it does not compare, so a level it calls dead could be live there.
/// Such a model is not checked at all — a skipped check binds, it never refuses.
fn dead_check_reads_every_channel(model: &CompiledModel) -> bool {
    model.mixture.is_none() && !model.is_sde() && !model.has_non_gaussian()
}

/// `x` moved by `step` toward the inside of `[lo, hi]`: up when that stays
/// below `hi`, down when that stays above `lo`, else by half the larger gap to
/// a bound — never onto a bound, and never back to `x` unless `lo == x == hi`
/// leaves no room at all, which the dead-level check reads as "cannot step"
/// (#1702 review, F2).
fn toward_interior(x: f64, lo: f64, hi: f64, step: f64) -> f64 {
    if x + step < hi {
        x + step
    } else if x - step > lo {
        x - step
    } else if hi - x >= x - lo {
        x + 0.5 * (hi - x)
    } else {
        x - 0.5 * (x - lo)
    }
}

/// How far the dead-level check moves a random effect to see whether it reads
/// a record.
const RE_STEP: f64 = 0.1;

/// The values the likelihood reads from one subject: per record, its
/// prediction and its residual-magnitude multipliers. Every η and κ is at
/// `re`, and random effect `bump` (an η index, or `n_eta` + a κ index) at
/// `re + RE_STEP`.
fn record_values(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    re: f64,
    bump: Option<usize>,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut eta = vec![re; model.n_eta];
    let mut kappa = vec![re; model.n_kappa];
    match bump {
        Some(r) if r < model.n_eta => eta[r] += RE_STEP,
        Some(r) => kappa[r - model.n_eta] += RE_STEP,
        None => {}
    }
    let preds = if model.n_kappa > 0 {
        let groups = crate::stats::likelihood::iov_occasion_groups(subject).len();
        crate::pk::predict_iov(model, subject, theta, &eta, &vec![kappa; groups.max(1)])
    } else {
        crate::pk::compute_predictions_with_tv(model, subject, theta, &eta)
    };
    let mult = model.ruv_obs_mult(subject, theta).unwrap_or_default();
    (preds, mult)
}

/// Whether two [`record_values`] agree bitwise on the records `rows` (every
/// record when `None`), every value finite. A non-finite value is not an
/// agreement: it measures nothing.
fn same_records(
    a: &(Vec<f64>, Vec<Vec<f64>>),
    b: &(Vec<f64>, Vec<Vec<f64>>),
    rows: Option<&[usize]>,
) -> bool {
    let same = |x: &f64, y: &f64| x.is_finite() && y.is_finite() && x.to_bits() == y.to_bits();
    if a.0.len() != b.0.len() || a.1.len() != b.1.len() {
        return false;
    }
    let all: Vec<usize>;
    let rows = match rows {
        Some(r) => r,
        None => {
            all = (0..a.0.len().max(a.1.len())).collect();
            &all
        }
    };
    rows.iter().all(|&j| {
        let pred = match (a.0.get(j), b.0.get(j)) {
            (Some(x), Some(y)) => same(x, y),
            (None, None) => true,
            _ => false,
        };
        let mult = match (a.1.get(j), b.1.get(j)) {
            (Some(x), Some(y)) => x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same(x, y)),
            (None, None) => true,
            _ => false,
        };
        pred && mult
    })
}

/// The level indices (1-based) a subject's records carry for `column`, on
/// every kind of record: a level's θ is read by exactly these subjects.
fn indices_read(subject: &Subject, column: &str) -> Vec<f64> {
    let mut out: Vec<f64> = subject
        .covariates
        .get(column)
        .copied()
        .into_iter()
        .collect();
    for maps in [
        &subject.obs_covariates,
        &subject.dose_covariates,
        &subject.pk_only_covariates,
        &subject.reset_covariates,
    ] {
        out.extend(maps.iter().filter_map(|m| m.get(column).copied()));
    }
    out.sort_by(f64::total_cmp);
    out.dedup();
    out
}

/// The observation rows of `subject` whose `column` index is `index`.
fn rows_of(subject: &Subject, column: &str, index: f64) -> Vec<usize> {
    (0..subject.obs_times.len())
        .filter(|&j| {
            subject
                .obs_covariates
                .get(j)
                .and_then(|m| m.get(column))
                .or_else(|| subject.covariates.get(column))
                == Some(&index)
        })
        .collect()
}

/// What rule 1 measured on one block (#1679).
#[derive(Default)]
struct DeadLevels {
    /// The dead levels, as indices into the block's levels.
    dead: Vec<usize>,
    /// Parallel to `dead`: the random effects (by name) that move some record
    /// of that level, at either point. A level no random effect reads either
    /// can take a within-group constraint from one that is absorbed.
    read_by: Vec<Vec<String>>,
}

impl DeadLevels {
    fn reads(&self, dead_level: usize, random_effect: &str) -> bool {
        self.dead
            .iter()
            .position(|&i| i == dead_level)
            .is_some_and(|p| self.read_by[p].iter().any(|r| r == random_effect))
    }
}

/// The two points at which the level checks measure the likelihood: the initial θ
/// with the random effects at 0, and every non-`FIX` θ jittered inside its bounds
/// with the random effects at 0.05. One point is not enough — a θ initialised at 0,
/// or a random effect at 0, can switch a block off there alone. `dead_levels` and
/// `refuse_read_unindexed` share these θ points, so the two cannot drift apart
/// (#1822); the second element is the random-effect level, which `dead_levels`
/// applies to every η and κ alike and `refuse_read_unindexed` spreads per η
/// (`probe_etas`).
fn probe_points(model: &CompiledModel) -> [(Vec<f64>, f64); 2] {
    let p = &model.default_params;
    let theta0 = p.theta.clone();
    let mut jittered = theta0.clone();
    for (k, x) in jittered.iter_mut().enumerate() {
        if !p.theta_fixed.get(k).copied().unwrap_or(false) {
            *x = toward_interior(*x, p.theta_lower[k], p.theta_upper[k], jitter_step(k, *x));
        }
    }
    [(theta0, 0.0), (jittered, 0.05)]
}

/// How far the level checks move θ `k` from `x`: distinct per θ, so no two
/// jitters cancel and no two levels of a block moved together coincide.
fn jitter_step(k: usize, x: f64) -> f64 {
    (0.13 + 0.07 * (k % 5) as f64) * x.abs().max(1.0)
}

/// The `n_eta` random effects the readership measurement uses at a probe point whose
/// level is `re`: 0 at the initial point, and distinct per index at the moved one, so
/// a read gated by a difference of random effects (`ETA_CL − ETA_V`) does not cancel
/// there — the same reason `jitter_step` is distinct per θ (#1822 review r1).
fn probe_etas(re: f64, n_eta: usize) -> Vec<f64> {
    (0..n_eta).map(|j| re * (1.0 + 0.3 * j as f64)).collect()
}

/// Rule 1 (#1679): the non-`FIX` levels of `decl` whose θ the likelihood never
/// reads, and which random effects read their records.
///
/// `model` is the model bound with every block under `contrast = none`, so each
/// level has a θ of its own. A level is dead when a finite step in its θ leaves
/// every value [`record_values`] returns bitwise unchanged, on every subject
/// that carries it, at both of two points: the initial θ with η = κ = 0, and a
/// deterministic jitter of every non-`FIX` θ inside its bounds with η = κ =
/// 0.05. One point is not enough — a θ initialised at 0 can switch the block
/// off there alone. A point where anything is non-finite, or where the θ has
/// no room to step, is inconclusive, and the level counts as live: only a
/// measured "no change" counts. Whether a dead level costs the model anything
/// depends on the contrast; [`dead_failure`] decides that.
///
/// Values are compared per record under a step of `0.1·max(|θ|, 1)`, not
/// summed into an objective under a tiny step: a live level whose effect is
/// small against the rest of `y` must still move its own record.
fn dead_levels(
    model: &CompiledModel,
    decl: &LevelBlockDecl,
    levels: &[Level],
    population: &Population,
) -> DeadLevels {
    let p = &model.default_params;
    let points = probe_points(model);
    let column = level_index_column(decl.name());
    let readers: Vec<Vec<f64>> = population
        .subjects
        .iter()
        .map(|s| indices_read(s, &column))
        .collect();
    let base: Vec<Vec<_>> = points
        .iter()
        .map(|(th, re)| {
            population
                .subjects
                .iter()
                .map(|s| record_values(model, s, th, *re, None))
                .collect()
        })
        .collect();

    let mut out = DeadLevels::default();
    for (i, level) in levels.iter().enumerate() {
        let name = format!("{}[{}]", decl.name(), level.label(decl.columns()));
        let Some(k) = model.theta_names.iter().position(|n| *n == name) else {
            continue;
        };
        if p.theta_fixed.get(k).copied().unwrap_or(false) {
            continue;
        }
        let index = (i + 1) as f64;
        let unchanged_at = |(point, (th, re)): (usize, &(Vec<f64>, f64))| {
            let mut stepped = th.clone();
            let step = 0.1 * th[k].abs().max(1.0);
            stepped[k] = toward_interior(th[k], p.theta_lower[k], p.theta_upper[k], step);
            if stepped[k] == th[k] {
                return false;
            }
            population.subjects.iter().enumerate().all(|(s, subject)| {
                !readers[s].contains(&index)
                    || same_records(
                        &base[point][s],
                        &record_values(model, subject, &stepped, *re, None),
                        None,
                    )
            })
        };
        if !points.iter().enumerate().all(unchanged_at) {
            continue;
        }
        let names: Vec<&String> = model.eta_names.iter().chain(&model.kappa_names).collect();
        let read_by = names
            .iter()
            .enumerate()
            .filter(|&(r, _)| {
                points.iter().enumerate().any(|(point, (th, re))| {
                    population.subjects.iter().enumerate().any(|(s, subject)| {
                        let rows = rows_of(subject, &column, index);
                        !rows.is_empty()
                            && !same_records(
                                &base[point][s],
                                &record_values(model, subject, th, *re, Some(r)),
                                Some(&rows),
                            )
                    })
                })
            })
            .map(|(_, n)| (*n).clone())
            .collect();
        out.dead.push(i);
        out.read_by.push(read_by);
    }
    out
}

/// Run [`dead_levels`] on every block, on the model re-parsed with every block
/// under `contrast = none`, so the measurement is each level's own θ whatever
/// contrast the block will take. A model whose likelihood reads a channel the
/// check does not compare is not measured: every block comes back with no dead
/// level.
fn measure_dead_levels(
    base: &ParseBindings,
    model_text: &str,
    decls: &[LevelBlockDecl],
    discovered: &[Vec<Level>],
    population: &Population,
) -> Result<Vec<DeadLevels>, String> {
    let mut none = base.clone();
    none.levels = decls
        .iter()
        .zip(discovered)
        .map(|(decl, levels)| {
            let binding = LevelBinding {
                labels: levels.iter().map(|l| l.label(decl.columns())).collect(),
                groups: vec![0; levels.len()],
                contrast: LevelContrast::Unconstrained,
            };
            (decl.name().to_string(), binding)
        })
        .collect();
    let model = parse_full_model_with(model_text, &none)?.model;
    if !dead_check_reads_every_channel(&model) {
        return Ok(decls.iter().map(|_| DeadLevels::default()).collect());
    }
    Ok(decls
        .iter()
        .zip(discovered)
        .map(|(decl, levels)| dead_levels(&model, decl, levels, population))
        .collect())
}

/// Why a contrast cannot carry a block's dead levels (#1679, #1702 review F1).
enum DeadFailure {
    /// The coded block loses rank: a free θ moves only dead levels. Under
    /// `none` that is any dead level; under `ref` one that is not its group's
    /// reference; under sum-to-zero two in one group.
    Rank,
    /// Under the within-group contrast, these dead levels — which the absorbed
    /// random effect does not read either — take their groups' constraints, so
    /// the other levels are free to reproduce that random effect.
    Slack(Vec<usize>),
}

/// Whether `contrast`, grouping the levels as `groups`, can carry the block's
/// dead levels. A dead level costs a degree of freedom only where the coding
/// gives it a direction of its own: the reference level under `ref`, and one
/// level per group under sum-to-zero, are coded through the other levels, so a
/// dead one there loses nothing.
fn dead_failure(
    groups: &[usize],
    contrast: LevelContrast,
    dead: &DeadLevels,
    absorbed: Option<&Absorbed>,
) -> Option<DeadFailure> {
    if dead.dead.is_empty() {
        return None;
    }
    let count_in = |g: usize| dead.dead.iter().filter(|&&i| groups[i] == g).count();
    match contrast {
        LevelContrast::Unconstrained => return Some(DeadFailure::Rank),
        LevelContrast::Ref => {
            let is_reference = |i: usize| i == 0 || groups[i - 1] != groups[i];
            if dead.dead.iter().any(|&i| !is_reference(i)) {
                return Some(DeadFailure::Rank);
            }
        }
        _ => {
            if dead.dead.iter().any(|&i| count_in(groups[i]) >= 2) {
                return Some(DeadFailure::Rank);
            }
        }
    }
    if contrast == LevelContrast::SumToZeroWithin {
        if let Some(found) = absorbed {
            let slack: Vec<usize> = dead
                .dead
                .iter()
                .copied()
                .filter(|&i| !dead.reads(i, found.name()))
                .collect();
            if !slack.is_empty() {
                return Some(DeadFailure::Slack(slack));
            }
        }
    }
    None
}

/// How many dead labels a refusal lists before it counts the rest.
const DEAD_LABELS_SHOWN: usize = 5;

/// The refusal for a block whose dead levels `contrast` cannot carry: every
/// level dead (the block estimates nothing), or the levels at fault, listed,
/// with what they cost under `contrast` — and `alt`, a contrast that carries
/// them, when there is one. The `TIME = 0` advice appears only when every
/// listed level holds only records at `TIME = 0`.
#[allow(clippy::too_many_arguments)]
fn dead_levels_message(
    block: &str,
    decl: &LevelBlockDecl,
    levels: &[Level],
    dead: &DeadLevels,
    failure: &DeadFailure,
    contrast: LevelContrast,
    alt: Option<LevelContrast>,
    absorbed: Option<&Absorbed>,
    population: &Population,
) -> String {
    if dead.dead.len() == levels.len() {
        return format!(
            "{block}: no level of the block affects the likelihood at the initial estimates or \
             at a nearby point; the block estimates nothing. Check the expression that reads it."
        );
    }
    let listed: &[usize] = match failure {
        DeadFailure::Rank => &dead.dead,
        DeadFailure::Slack(levels) => levels,
    };
    let shown: Vec<String> = listed
        .iter()
        .take(DEAD_LABELS_SHOWN)
        .map(|&i| format!("`{}`", levels[i].label(decl.columns())))
        .collect();
    let more = match listed.len().saturating_sub(DEAD_LABELS_SHOWN) {
        0 => String::new(),
        n => format!(" and {n} more"),
    };
    let mut message = format!(
        "{block}: each of these levels has no effect on the predictions or the residual error \
         at any of its records, at the initial estimates or at a nearby point: {}{more}.",
        shown.join(", ")
    );
    match (failure, alt, absorbed) {
        (DeadFailure::Slack(_), _, Some(found)) => message.push_str(&format!(
            " Under `contrast = sum_to_zero_within` such a level takes its group's sum-to-zero \
             constraint, so the other levels reproduce {} — the two are the same quantity, so \
             the model is not identified.",
            found.clause()
        )),
        (_, Some(alt), _) => message.push_str(&format!(
            " Under `contrast = {}` that leaves θ the data cannot estimate, which \
             `contrast = {}` does not.",
            contrast_token(contrast),
            contrast_token(alt)
        )),
        _ => message.push_str(" That leaves θ the data cannot estimate under any contrast."),
    }
    let listed_values: Vec<&[f64]> = listed
        .iter()
        .map(|&i| levels[i].values.as_slice())
        .collect();
    let at_time_zero = population.subjects.iter().all(|subject| {
        (0..subject.obs_times.len()).all(|j| {
            let values: Option<Vec<f64>> = decl
                .columns()
                .iter()
                .map(|c| column_value(subject, c, j))
                .collect();
            let is_listed = values.is_some_and(|v| listed_values.contains(&v.as_slice()));
            !is_listed || subject.obs_times[j] == 0.0
        })
    });
    if at_time_zero {
        message.push_str(
            " Every one of them holds only records at `TIME = 0`. Key the block so those records \
             share a level with a later time, or read the block where it acts at `TIME = 0`.",
        );
    } else {
        message.push_str(" Key the block so those records share a level with records it acts on.");
    }
    message.push_str(
        " The check sees only the initial estimates and a point near them, so a θ that switches \
         the block off there (a lag or a threshold) makes a live level look dead: check that \
         θ's initial estimate.",
    );
    message
}

/// Resolve [`LevelContrast::Auto`], group the levels under it, and reject the
/// configurations that are still rank-deficient once resolved.
///
/// Three refusals, in this order: a block that absorbs two or more random
/// effects ([`absorbed`]), under every contrast (#1696); a block that absorbs
/// one under a contrast that leaves it free θ (#1064, #1642, #1649, #1650,
/// #1678); then a block left with no free θ at all (#1624) — whatever the
/// contrast, since a block that estimates nothing cannot be told apart from
/// not declaring it.
///
/// A nested block (two or more columns) that absorbs one random effect resolves
/// `auto` to the within-group contrast, which leaves each unit's mean to the
/// random effect; an explicit global contrast is refused. A one-column block
/// has no such rescue: each level *is* one unit's, so it is refused under every
/// contrast. Two absorbed random effects have no rescue either: within leaves
/// each unit's mean to one of them, and the levels reproduce the other.
fn resolve_contrast(
    decl: &LevelBlockDecl,
    levels: &[Level],
    dead: &DeadLevels,
    population: &Population,
) -> Result<(LevelContrast, Vec<usize>), String> {
    let nested = decl.columns().len() >= 2;
    let block = format!("theta {}[{}]", decl.name(), decl.columns().join(", "));
    let absorbed = absorbed(decl, population);
    if absorbed.len() >= 2 {
        return Err(jointly_absorbed_message(
            &block, decl, &absorbed, population,
        ));
    }
    let absorbed = absorbed.into_iter().next();
    // A contrast the block can take: one the absorbed random effect allows,
    // that leaves a free θ, and that carries the dead levels.
    let admissible = |c: LevelContrast| {
        let allowed = match &absorbed {
            None => true,
            Some(_) => nested && c == LevelContrast::SumToZeroWithin,
        };
        let groups = assign_groups(decl, levels, c);
        allowed
            && free_count(&groups, c) > 0
            && dead_failure(&groups, c, dead, absorbed.as_ref()).is_none()
    };
    let resolved = match decl.contrast() {
        LevelContrast::Auto if nested && absorbed.is_some() => LevelContrast::SumToZeroWithin,
        // Global sum-to-zero unless it cannot carry the dead levels and the
        // within-group contrast can (#1702 review, F1).
        LevelContrast::Auto
            if nested
                && !admissible(LevelContrast::SumToZero)
                && admissible(LevelContrast::SumToZeroWithin) =>
        {
            LevelContrast::SumToZeroWithin
        }
        LevelContrast::Auto => LevelContrast::SumToZero,
        other => other,
    };
    let groups = assign_groups(decl, levels, resolved);
    let free = free_count(&groups, resolved);

    // The configuration the feature exists to serve — an unstructured placebo
    // effect per study × timepoint under between-study variability — is
    // over-parameterised under any convention that leaves each group's mean
    // free: that study's η *is* the mean of its own levels. A check that only
    // looked for a fixed intercept would wave it through, which is precisely
    // the silent flat direction this codebase treats as a bug.
    if let Some(found) = absorbed.as_ref() {
        // A one-column block with no free θ is the plain single-level case
        // below: there is no level left for the random effect to be confused with.
        if !nested && free > 0 {
            let (unit, whose) = if found.kappa() {
                (
                    "lies within a single occasion of one subject",
                    "that occasion's",
                )
            } else {
                ("belongs to a single subject", "that subject's")
            };
            return Err(format!(
                "{block}: each `{}` level {unit}, and {}, so a level and {whose} random effect \
                 are the same quantity: the model is not identified under any contrast. Remove \
                 the block, or drop the random effect.",
                decl.columns()[0],
                found.clause(),
            ));
        }
        if nested
            && matches!(
                resolved,
                LevelContrast::SumToZero | LevelContrast::Ref | LevelContrast::Unconstrained
            )
        {
            let leading = subject_key_columns(decl).join(", ");
            // When every group is a single level the within-group contrast
            // would leave nothing to estimate, so it is not the advice (#1624).
            let within = assign_groups(decl, levels, LevelContrast::SumToZeroWithin);
            let fix = if free_count(&within, LevelContrast::SumToZeroWithin) == 0 {
                format!(
                    " Every {leading} group has a single level, so the random effect already \
                     carries each group's value: remove the block, or drop the random effect."
                )
            } else if !admissible(LevelContrast::SumToZeroWithin) {
                // The within-group rescue fails on the block's dead levels
                // (#1702 review, F1), so it is not the advice either.
                " `contrast = sum_to_zero_within` does not help: some levels have no effect at \
                 any of their records, which that contrast cannot carry either. Drop the random \
                 effect, or key the block so every level acts on its records."
                    .to_string()
            } else {
                " Use `contrast = sum_to_zero_within` (the default for this shape), or drop \
                 the random effect."
                    .to_string()
            };
            let why = match found.how {
                How::Site(_) => format!("{} at that grouping", found.clause()),
                How::Route(_) => format!(
                    "{}, and the block takes a level at every observation, so it can \
                     reproduce any effect that random effect has",
                    found.clause()
                ),
            };
            // A kappa is not a group's mean: the levels reproduce it occasion by
            // occasion, which the within-group contrast removes as it does an η.
            if found.kappa() {
                return Err(format!(
                    "{block}: `contrast = {}` lets the levels reproduce `{}` at every occasion: \
                     {why} — the two are the same quantity, so the model is not identified.{fix}",
                    contrast_token(resolved),
                    found.name(),
                ));
            }
            return Err(format!(
                "{block}: `contrast = {}` leaves each {leading} group's mean free, but {why} \
                 — the two are the same quantity, so the model is not identified.{fix}",
                contrast_token(resolved),
            ));
        }
    }

    // Rule 1 (#1679): a level the likelihood never reads, where the contrast
    // gives it a direction of its own.
    if free > 0 {
        if let Some(failure) = dead_failure(&groups, resolved, dead, absorbed.as_ref()) {
            let alt = [
                LevelContrast::SumToZero,
                LevelContrast::SumToZeroWithin,
                LevelContrast::Ref,
                LevelContrast::Unconstrained,
            ]
            .into_iter()
            .filter(|&c| c != resolved && (nested || c != LevelContrast::SumToZeroWithin))
            .find(|&c| admissible(c));
            return Err(dead_levels_message(
                &block,
                decl,
                levels,
                dead,
                &failure,
                resolved,
                alt,
                absorbed.as_ref(),
                population,
            ));
        }
    }

    if free == 0 {
        let single = "the data carries a single level";
        let none = "Use `contrast = none` if a single constant is what you meant.";
        // Only a nested block has groups to name; a one-column block's single
        // level reads as the plain single-level case.
        let found = absorbed.filter(|_| nested);
        let leading = subject_key_columns(decl).join(", ");
        return Err(match (resolved, found) {
            (_, Some(found)) => format!(
                "{block}: every {leading} group has a single level, and {} — the random \
                 effect already carries each group's value, so the block estimates nothing. \
                 Remove the block.",
                found.clause(),
            ),
            (LevelContrast::Ref, None) => {
                format!("{block}: {single}, which is the reference level, held at 0. {none}")
            }
            (LevelContrast::SumToZeroWithin, None) if levels.len() > 1 => format!(
                "{block}: every {leading} group has a single level, which the within-group \
                 sum-to-zero pins at 0, so the block estimates nothing. Use \
                 `contrast = sum_to_zero` to estimate the levels around their common mean, \
                 or `contrast = none`."
            ),
            (LevelContrast::SumToZeroWithin, None) => {
                format!("{block}: {single}, which the within-group sum-to-zero pins at 0. {none}")
            }
            _ => format!("{block}: {single}, which sum-to-zero pins at 0. {none}"),
        });
    }

    Ok((resolved, groups))
}

/// The refusal for a block that absorbs two or more random effects (#1696).
///
/// One case is not the block's doing: a kappa and an η absorbed at the same
/// site while every subject has a single occasion. The kappa is then the η by
/// another name, so the model is unidentified with or without the block, and
/// the refusal says so instead of blaming it.
fn jointly_absorbed_message(
    block: &str,
    decl: &LevelBlockDecl,
    absorbed: &[Absorbed],
    population: &Population,
) -> String {
    let one_occasion = population.subjects.iter().all(|s| {
        s.occasions
            .first()
            .is_some_and(|o| s.occasions.iter().all(|x| x == o))
    });
    if one_occasion {
        // The same site is the same readers: the individual parameters that
        // read each random effect directly (#1702 review, F4). Comparing the
        // diagnostic clauses would not do: a route clause names its random
        // effect, and two random effects on different parameters can share a
        // route.
        let twin = absorbed.iter().filter(|k| k.kappa()).find_map(|k| {
            absorbed
                .iter()
                .find(|e| !e.kappa() && e.readers == k.readers)
                .map(|e| (k, e))
        });
        if let Some((k, e)) = twin {
            // On a one-column block dropping either one still leaves a
            // random effect that each level is the same quantity as (F5).
            let one_column = if decl.columns().len() == 1 {
                format!(
                    " Each `{}` level also lies within a single subject, so remove the block as \
                     well.",
                    decl.columns()[0]
                )
            } else {
                String::new()
            };
            return format!(
                "{block}: every subject has a single occasion, so `{}` cannot be told apart from \
                 `{}` with or without this block: the model is not identified. Drop one of the \
                 two random effects.{one_column}",
                k.name(),
                e.name(),
            );
        }
    }
    let names: Vec<String> = absorbed.iter().map(|a| format!("`{}`", a.name())).collect();
    let (last, rest) = names.split_last().expect("two or more");
    let mut sites: Vec<String> = Vec::new();
    for a in absorbed {
        let clause = a.clause();
        if !sites.contains(&clause) {
            sites.push(clause);
        }
    }
    // Within leaves each unit's mean to one of them: a subject's when an η is
    // among them, an occasion's when they are all kappas (F6).
    let unit = if absorbed.iter().all(|a| a.kappa()) {
        "occasion"
    } else {
        "subject"
    };
    format!(
        "{block}: the levels absorb {} and {last} together ({}). `contrast = sum_to_zero_within` \
         leaves each {unit}'s mean to one of them, and the levels reproduce the other, so the \
         model is not identified under any contrast. Drop one of the random effects, or remove \
         the block.",
        rest.join(", "),
        sites.join("; "),
    )
}

/// The free θ a contrast leaves over `groups`: every level under `none`, one
/// fewer per group otherwise (the group's dependent or reference level).
fn free_count(groups: &[usize], contrast: LevelContrast) -> usize {
    if matches!(contrast, LevelContrast::Unconstrained) {
        return groups.len();
    }
    // Sort first: `dedup` only collapses adjacent ids, and the count must not
    // depend on `assign_groups` handing them out contiguously.
    let mut ids = groups.to_vec();
    ids.sort_unstable();
    ids.dedup();
    groups.len() - ids.len()
}

/// The expression where a block meets a random effect, as a diagnostic clause.
fn share_site(share: &ScaleShare) -> String {
    let via = share
        .eta_via
        .as_ref()
        .map(|v| format!(" (through `{v}`)"))
        .unwrap_or_default();
    match &share.param {
        Some(p) => {
            format!(
                "the individual parameter `{p}` reads this block and carries a random effect{via}"
            )
        }
        None => format!("the `y` readout reads this block and a random effect{via}"),
    }
}

/// The `contrast = ...` token for a resolved convention, for diagnostics.
fn contrast_token(c: LevelContrast) -> &'static str {
    match c {
        LevelContrast::Auto => "auto",
        LevelContrast::SumToZero => "sum_to_zero",
        LevelContrast::SumToZeroWithin => "sum_to_zero_within",
        LevelContrast::Ref => "ref",
        LevelContrast::Unconstrained => "none",
    }
}

/// Group id per level. Levels are sorted by their full tuple, so grouping by
/// the leading columns yields contiguous groups — which is what lets each
/// group's sum-to-zero contrast be a single `NegSum` range.
fn assign_groups(decl: &LevelBlockDecl, levels: &[Level], contrast: LevelContrast) -> Vec<usize> {
    let within = matches!(contrast, LevelContrast::SumToZeroWithin) && decl.columns().len() >= 2;
    if !within {
        return vec![0; levels.len()];
    }
    let mut groups = Vec::with_capacity(levels.len());
    let mut current: Option<&[f64]> = None;
    let mut id = 0usize;
    for level in levels {
        match current {
            Some(prev) if prev == level.leading() => {}
            None => current = Some(level.leading()),
            Some(_) => {
                id += 1;
                current = Some(level.leading());
            }
        }
        groups.push(id);
    }
    groups
}

/// Write the synthesized 1-based level index onto every subject.
///
/// `table` pairs each level with its index: the level's own position for
/// [`bind_theta_levels`], its position in the fit for [`bind_from_fit`] (and the
/// deprecated `bind_theta_levels_from_fit`). Every binder shares this one writer, so
/// the dose, EVID=2 and reset handling below cannot drift between them.
///
/// When the index is constant within a subject it goes into the subject-level
/// covariate map only — no time-varying machinery is engaged, so the model
/// keeps whatever fast path it had. When it varies (the unstructured-placebo
/// case, where the index moves with the timepoint) the per-event snapshots are
/// materialised, which is exactly what a genuinely per-record parameter needs.
///
/// A subject with no Gaussian observation (#1797) has no record that defines a level,
/// and its other records carry no covariate snapshot, so it is indexed at the level its
/// baseline `covariates` name — the values its likelihood reads. When `table` holds no
/// such level (or the block is keyed on `TIME`, which such a subject cannot name) it
/// gets index 1 and is returned, for [`refuse_read_unindexed`] to measure.
fn write_index_column(
    decl: &LevelBlockDecl,
    table: &[(Level, usize)],
    subjects: &mut [Subject],
) -> Result<Vec<Unindexed>, String> {
    let column = level_index_column(decl.name());
    let index_of = |values: &[f64]| -> Option<f64> {
        table
            .iter()
            .find(|(l, _)| l.values == values)
            .map(|&(_, i)| i as f64)
    };

    let mut unindexed = Vec::new();
    for subject in subjects.iter_mut() {
        let n_obs = subject.obs_times.len();
        let mut obs_index = Vec::with_capacity(n_obs);
        for j in 0..n_obs {
            let values: Vec<f64> = decl
                .columns()
                .iter()
                .map(|c| column_value(subject, c, j).unwrap_or(f64::NAN))
                .collect();
            let idx = index_of(&values).ok_or_else(|| {
                format!(
                    "theta {}[...]: subject {} row {j} has a level combination that \
                     was not discovered — the data changed between passes",
                    decl.name(),
                    subject.id
                )
            })?;
            obs_index.push(idx);
        }

        // A subject with no Gaussian observation (TTE, binary, categorical or Markov
        // records only, #1797) has no record defining a level, and its records carry no
        // covariate snapshot of their own: what its likelihood reads is its baseline
        // `covariates`, so that is the level it is indexed at. A level the table does
        // not hold gets index 1, and the subject is returned so the caller can refuse
        // the binding if anything that subject is scored on reads the block.
        let first = match obs_index.first() {
            Some(&i) => i,
            None => match baseline_level(decl, subject) {
                Ok(values) => index_of(&values).unwrap_or_else(|| {
                    unindexed.push(Unindexed {
                        id: subject.id.clone(),
                        why: Why::NoLevel(Level { values }.label(decl.columns())),
                    });
                    1.0
                }),
                Err(why) => {
                    unindexed.push(Unindexed {
                        id: subject.id.clone(),
                        why,
                    });
                    1.0
                }
            },
        };
        subject.covariates.insert(column.clone(), first);
        let varies = obs_index.iter().any(|&v| v != first);
        if !varies {
            // Constant within the subject: the baseline map is enough, but any
            // per-event snapshots that already exist must stay complete.
            for m in subject.obs_covariates.iter_mut() {
                m.insert(column.clone(), first);
            }
            for m in subject.dose_covariates.iter_mut() {
                m.insert(column.clone(), first);
            }
            for m in subject.pk_only_covariates.iter_mut() {
                m.insert(column.clone(), first);
            }
            // EVID=3/4 rows too (#1133): their snapshot feeds the `[odes] init(...)`
            // re-seed, so a missing column there reads as `0.0` at the reset while every
            // other record sees the real 1-based index.
            for m in subject.reset_covariates.iter_mut() {
                m.insert(column.clone(), first);
            }
            continue;
        }

        // Materialise per-event snapshots if this is the first time-varying
        // covariate the subject has. Seeding from `covariates` keeps every
        // other covariate at the value the LOCF snapshots would have carried.
        if subject.obs_covariates.is_empty() {
            subject.obs_covariates = vec![subject.covariates.clone(); n_obs];
        }
        if subject.dose_covariates.is_empty() {
            subject.dose_covariates = vec![subject.covariates.clone(); subject.doses.len()];
        }
        if subject.pk_only_covariates.is_empty() {
            subject.pk_only_covariates =
                vec![subject.covariates.clone(); subject.pk_only_times.len()];
        }
        if subject.reset_covariates.is_empty() {
            subject.reset_covariates = vec![subject.covariates.clone(); subject.reset_times.len()];
        }
        for (j, m) in subject.obs_covariates.iter_mut().enumerate() {
            m.insert(column.clone(), obs_index.get(j).copied().unwrap_or(first));
        }
        // Dose and EVID=2 rows carry the level of the most recent observation
        // at or before them (the first level before any observation). A level
        // is a property of an *observation*, so this only matters for a model
        // whose gathered parameter also drives the dosing dynamics — not the
        // unstructured-placebo case, which reads it in the prediction.
        let locf = |t: f64| -> f64 {
            let mut v = first;
            for (j, &ot) in subject.obs_times.iter().enumerate() {
                if ot <= t {
                    v = obs_index[j];
                } else {
                    break;
                }
            }
            v
        };
        let dose_times: Vec<f64> = subject.doses.iter().map(|d| d.time).collect();
        for (i, m) in subject.dose_covariates.iter_mut().enumerate() {
            let t = dose_times.get(i).copied().unwrap_or(0.0);
            m.insert(column.clone(), locf(t));
        }
        let pk_only_times = subject.pk_only_times.clone();
        for (i, m) in subject.pk_only_covariates.iter_mut().enumerate() {
            let t = pk_only_times.get(i).copied().unwrap_or(0.0);
            m.insert(column.clone(), locf(t));
        }
        // Reset rows take the same LOCF-of-observations rule as dose and EVID=2 rows
        // (#1133); the level is a property of an observation either way.
        let reset_times = subject.reset_times.clone();
        for (i, m) in subject.reset_covariates.iter_mut().enumerate() {
            let t = reset_times.get(i).copied().unwrap_or(0.0);
            m.insert(column.clone(), locf(t));
        }
    }

    // The column lives only in the subjects' covariate maps, where the
    // predictors (`Subject::obs_cov`) and `check_covariates`' carried-by-every-
    // subject test read it. It is deliberately *not* added to
    // `population.covariate_names`: that list is the data's columns as a user
    // and downstream tools read them (`FitResult::covariate_names`, "Available
    // covariate columns", GAM, the search resolver, the FREM CSV header), and a
    // synthesized column is engine plumbing (#1644). The binder rejects a
    // population with no observations, so every bound population has subjects
    // to carry it.
    Ok(unindexed)
}

/// A subject with no Gaussian observation whose own level the writer could not index
/// (#1797): [`write_index_column`] gave it index 1, a value only a likelihood that
/// does not read the block may carry.
#[derive(Debug, Clone, PartialEq)]
struct Unindexed {
    id: String,
    why: Why,
}

/// Why a subject with no Gaussian observation has no index of its own.
#[derive(Debug, Clone, PartialEq)]
enum Why {
    /// Its baseline columns name this level, which the table does not hold.
    NoLevel(String),
    /// The block is keyed on `TIME`, and the subject has no observation time.
    Time,
    /// Its baseline carries no finite value in this level column.
    Missing(String),
}

impl Why {
    fn item(&self, id: &str) -> String {
        match self {
            Why::NoLevel(label) => format!("subject {id} (`{label}`)"),
            Why::Time => format!("subject {id} (no observation time)"),
            Why::Missing(c) => format!("subject {id} (no value in level column `{c}`)"),
        }
    }
}

/// The level a subject with no Gaussian observation names: its baseline value in each
/// level column, the first non-missing one in its records (`Subject::covariates`).
fn baseline_level(decl: &LevelBlockDecl, subject: &Subject) -> Result<Vec<f64>, Why> {
    decl.columns()
        .iter()
        .map(|c| {
            if c.eq_ignore_ascii_case(TIME_COLUMN) {
                return Err(Why::Time);
            }
            subject
                .covariates
                .get(c)
                .copied()
                .filter(|v| v.is_finite())
                .ok_or_else(|| Why::Missing(c.clone()))
        })
        .collect()
}

/// Where a binding's level table came from, for [`refuse_read_unindexed`]'s text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LevelSource {
    /// Discovered from the population itself ([`bind_theta_levels`]).
    Data,
    /// A fit's bindings ([`bind_from_fit`] and the re-read of a fit's data).
    Fit,
}

/// Refuse a binding that would score a subject with no Gaussian observation at a
/// level other than its own (#1797).
///
/// Such a subject is indexed at the level its baseline columns name. When the table
/// (`tables`, parallel to `decls`) holds no such level — the data's Gaussian records
/// never showed it, the fit never estimated it, or the block is keyed on `TIME` — the
/// writer gives it index 1, which is harmless only if nothing it is scored on reads
/// the block. That is measured, not inferred from which endpoint reads which
/// parameter: the engine's own `individual_nll`, at the dead-level check's two θ
/// points (`probe_points`: the initial θ with η = 0, and every free θ jittered, here
/// with each η moved by a distinct amount from 0.05, `probe_etas`), on the subject
/// with its index set to each level of `model`'s block in turn, with every free θ of the block moved inside its bounds from that
/// point (so that no two levels share a value, a reference level fixed at 0
/// included), against index 1 at the point itself (so a one-level block is measured
/// too). The block is read when, at either point, any of those differ: a read that
/// another θ or a random effect switches off at the initial estimates is seen when
/// the moved point opens it (#1822). Only a measured, finite "no change" binds; a
/// non-finite value measures nothing and counts as a read. That branch is
/// defensive: the endpoint likelihoods map an ill-defined term to the finite `1e20`
/// sentinel (`crate::survival`) before it reaches here, so no fixture has reached
/// it (#1820 review r1). Like the dead-level check, the two points can miss a read
/// that is switched off at both — a threshold both values sit behind — and a read
/// through κ, which `individual_nll` does not take.
///
/// Reads `population` and writes nothing: only the subjects with no Gaussian
/// observation are copied, so a binder can call this before it writes any column.
fn refuse_read_unindexed(
    model: &CompiledModel,
    decls: &[LevelBlockDecl],
    tables: &[Vec<(Level, usize)>],
    population: &Population,
    source: LevelSource,
) -> Result<(), String> {
    let mut subjects: Vec<Subject> = population
        .subjects
        .iter()
        .filter(|s| s.obs_times.is_empty())
        .cloned()
        .collect();
    if subjects.is_empty() {
        return Ok(());
    }
    let mut unindexed: Vec<Vec<Unindexed>> = Vec::with_capacity(decls.len());
    for (decl, table) in decls.iter().zip(tables) {
        unindexed.push(write_index_column(decl, table, &mut subjects)?);
    }
    let p = &model.default_params;
    let points = probe_points(model);
    for (decl, unindexed) in decls.iter().zip(&unindexed) {
        if unindexed.is_empty() {
            continue;
        }
        let n_levels = model
            .theta_blocks()
            .level_blocks()
            .iter()
            .find(|d| d.name() == decl.name())
            .map_or(1, |d| d.labels().len().max(1));
        let prefix = format!("{}[", decl.name());
        // Each probe point with the block's free θ moved from it.
        let probes: Vec<(&Vec<f64>, Vec<f64>, Vec<f64>)> = points
            .iter()
            .map(|(base, re)| {
                let mut moved = base.clone();
                for (k, x) in moved.iter_mut().enumerate() {
                    if model.theta_names[k].starts_with(&prefix)
                        && !p.theta_fixed.get(k).copied().unwrap_or(false)
                    {
                        *x = toward_interior(
                            *x,
                            p.theta_lower[k],
                            p.theta_upper[k],
                            jitter_step(k, *x),
                        );
                    }
                }
                (base, moved, probe_etas(*re, model.n_eta))
            })
            .collect();
        let column = level_index_column(decl.name());
        let reads = |subject: &Subject| -> bool {
            let nll = |theta: &[f64], eta: &[f64], index: f64| {
                let mut s = subject.clone();
                set_index(&mut s, &column, index);
                crate::stats::likelihood::individual_nll(
                    model,
                    &s,
                    theta,
                    eta,
                    &p.omega,
                    &p.sigma.values,
                )
            };
            probes.iter().any(|(base, moved, eta)| {
                let at = nll(base, eta, 1.0);
                !at.is_finite()
                    || (1..=n_levels).any(|i| {
                        let v = nll(moved, eta, i as f64);
                        !v.is_finite() || v.to_bits() != at.to_bits()
                    })
            })
        };
        let read: Vec<&Unindexed> = unindexed
            .iter()
            .filter(|u| subjects.iter().find(|s| s.id == u.id).is_some_and(&reads))
            .collect();
        if !read.is_empty() {
            return Err(unindexed_read_message(decl, &read, source));
        }
    }
    Ok(())
}

/// `index` on every record of `subject`, for the readership measurement.
fn set_index(subject: &mut Subject, column: &str, index: f64) {
    subject.covariates.insert(column.to_string(), index);
    for maps in [
        &mut subject.obs_covariates,
        &mut subject.dose_covariates,
        &mut subject.pk_only_covariates,
        &mut subject.reset_covariates,
    ] {
        for m in maps.iter_mut() {
            m.insert(column.to_string(), index);
        }
    }
}

/// The refusal for subjects with no Gaussian observation that read a level block at a
/// level it has no θ for (#1797). Every subject is listed, as for unseen levels.
fn unindexed_read_message(
    decl: &LevelBlockDecl,
    read: &[&Unindexed],
    source: LevelSource,
) -> String {
    let items: Vec<String> = read.iter().map(|u| u.why.item(&u.id)).collect();
    let mut message = format!(
        "theta {}[{}]: {} subject(s) with no Gaussian observation are scored on something \
         that reads this block, at a level it has no theta for: {}.",
        decl.name(),
        decl.columns().join(", "),
        read.len(),
        items.join(", ")
    );
    message.push_str(
        " A subject with no Gaussian observation takes its level from its baseline columns.",
    );
    message.push_str(match source {
        LevelSource::Data => {
            " The block's levels are discovered from Gaussian observation records only, \
             so a level that only such subjects show is not one of them."
        }
        LevelSource::Fit => {
            " A level's theta exists only for a combination the fit's Gaussian \
             observations showed."
        }
    });
    if read.iter().any(|u| u.why == Why::Time) {
        message.push_str(&format!(
            " `{TIME_COLUMN}` is a level column of this block, and a subject with no \
             Gaussian observation has no observation time to name a level of it."
        ));
    }
    message.push_str(match source {
        LevelSource::Data if read.iter().any(|u| matches!(u.why, Why::NoLevel(_))) => {
            " Give each such level Gaussian observations, drop those subjects, or stop \
             reading the block in what they are scored on."
        }
        LevelSource::Data => {
            " Drop those subjects, or stop reading the block in what they are scored on."
        }
        LevelSource::Fit => " Drop those subjects: the fit has no theta for what they read.",
    });
    message
}

/// Complete level labels of every bound level block, keyed by block name.
/// Includes dependent contrast levels that have no independently estimated θ.
pub fn level_map(model: &crate::types::CompiledModel) -> HashMap<String, Vec<String>> {
    let mut out = HashMap::new();
    for decl in model.theta_blocks().level_blocks() {
        if !decl.labels().is_empty() {
            out.insert(decl.name().to_string(), decl.labels().to_vec());
        }
    }
    out
}

/// One level of a bound θ level block, as reported by [`theta_level_values`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ThetaLevelValue {
    /// The level's label (`STUDY=7,TIME=4`), as [`theta_level_map`](crate::theta_level_map)
    /// lists it.
    pub label: String,
    /// The value the model uses for this level at the supplied θ. A free level is its
    /// own θ; a dependent level is derived from the free ones — the negated sum of
    /// its group's free θ under `sum_to_zero` / `sum_to_zero_within`, and `0` for a
    /// `ref` reference level or a level whose group has no free θ.
    pub value: f64,
    /// `Some(k)` when the level is estimated directly, `k` being its position in the
    /// θ vector (and in `theta_names` and the θ standard errors); `None` for a
    /// dependent level, which has no θ of its own.
    pub theta_index: Option<usize>,
}

/// Every level's value, free and dependent, of each bound level block, keyed by
/// block name and in [`theta_level_map`](crate::theta_level_map) order (#1623).
///
/// The value is what the model's own evaluator reads for that level at `theta` — the
/// one every prediction and objective uses — so a dependent level is reported exactly
/// as the model applies it, never re-derived here. `theta_index` places each free
/// level in the θ vector, so standard errors can be joined to it.
///
/// `theta` must be laid out for this bound model: the θ of a fit of `model`, or of a
/// model rebound with [`bind_from_fit`]. A θ whose length is not the
/// model's is refused; one of the right length but from a different binding cannot be
/// detected, and is read at the wrong positions.
///
/// Unbound level blocks and counted `theta NAME[N]` blocks are omitted, as in
/// [`theta_level_map`](crate::theta_level_map); a model with neither gives an empty map.
pub fn theta_level_values(
    model: &crate::types::CompiledModel,
    theta: &[f64],
) -> Result<HashMap<String, Vec<ThetaLevelValue>>, String> {
    let expected = model.default_params.theta.len();
    if theta.len() != expected {
        return Err(format!(
            "the supplied theta has {} values but this model has {expected}; level values \
             are read from theta by position",
            theta.len()
        ));
    }
    let blocks = model.theta_blocks();
    let mut out = HashMap::new();
    for decl in blocks.level_blocks() {
        if decl.labels().is_empty() {
            continue;
        }
        // Unreachable by construction: the parser pushes a gather for every declared
        // level block, and its rules are built from the same binding as the labels, one
        // per label. Kept as an `Err` rather than a panic since this is a library entry.
        let gather = blocks
            .decls
            .iter()
            .find(|d| d.name == decl.name())
            .ok_or_else(|| {
                format!(
                    "theta {}: the level block is bound but has no level-to-theta map",
                    decl.name()
                )
            })?;
        debug_assert_eq!(decl.labels().len(), gather.spec.levels.len());
        let values = decl
            .labels()
            .iter()
            .zip(&gather.spec.levels)
            .enumerate()
            .map(|(i, (label, rule))| ThetaLevelValue {
                label: label.clone(),
                value: eval_gather(&gather.spec, theta, (i + 1) as f64),
                theta_index: match *rule {
                    LevelRule::Free(k) => Some(k as usize),
                    LevelRule::NegSum(..) => None,
                },
            })
            .collect();
        out.insert(decl.name().to_string(), values);
    }
    Ok(out)
}

#[cfg(test)]
#[path = "tests/theta_levels_tests.rs"]
mod tests;
