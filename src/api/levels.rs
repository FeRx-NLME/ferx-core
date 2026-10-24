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
use crate::api::postfit::theta_se_from_packed;
use crate::diagnostics::{Diagnostic, EngineError};
use crate::estimation::parameterization::{theta_from_packed, theta_packing_jacobian};
use crate::estimation::sir::percentile_ci;
use crate::parser::model_parser::{
    eval_gather, level_index_column, parse_full_model_with, DataBindings, EtaCoupling, EtaRoute,
    LevelBinding, LevelBindings, LevelBlockDecl, LevelContrast, LevelRule, ParseBindings,
    ScaleShare,
};
use crate::stats::nested_levels::{label_path, NestedLayout};
use crate::types::{
    CompiledModel, IovOccasionRule, NestedLevels, ParsedModel, Population, Subject,
};

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
    // the same call `fit()` makes. Under nested levels (#1082) those are the
    // level cells, written by the call `fit()` makes when it prepares the
    // population, which a derived rule never overwrites.
    let derived;
    let with_occasions: &Population = if parsed.model.n_kappa > 0
        && parsed.model.has_nested_levels()
    {
        let mut p = bound.clone();
        crate::stats::nested_levels::assign_level_cells(&parsed.model, &mut p);
        derived = p;
        &derived
    } else if parsed.model.n_kappa > 0 && parsed.fit_options.iov_occasion != IovOccasionRule::Column
    {
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

    // Rule 1 (#1679): measure which levels the likelihood never reads, and
    // (#1834) which random effects the block absorbs where the walk names no
    // site. Whether a dead level costs the model anything depends on the
    // contrast, so the verdict is the contrast resolution's.
    let measured = measure_levels(&base, model_text, &decls, &discovered, with_occasions)?;

    // Two passes (#1963). A block's certificate counts every other block as it
    // will be coded, so the blocks that absorb no random effect — whose
    // resolution never reads the certificate — resolve first, independently;
    // then each absorbing block, against those. Another absorbing block counts
    // raw, every level: two absorbing blocks that each depend on the other's
    // coding keep the conservative verdict. No second-pass block reads another
    // second-pass block's result, so the outcome does not depend on the order
    // the blocks are declared in — which is what lets the second pass run
    // lazily, in declaration order, stopping at the first refusal: a
    // certificate is never measured for a model already refused.
    let absorbing: Vec<bool> = decls
        .iter()
        .zip(&measured)
        .map(|(decl, m)| !absorbed(decl, with_occasions, m).is_empty())
        .collect();
    let mut resolved: Vec<Option<Result<(LevelContrast, Vec<usize>), String>>> =
        vec![None; decls.len()];
    for (i, ((decl, levels), m)) in decls.iter().zip(&discovered).zip(&measured).enumerate() {
        if !absorbing[i] {
            resolved[i] = Some(resolve_contrast(decl, levels, m, with_occasions, None));
        }
    }
    let first_pass: Vec<Coded> = (0..decls.len())
        .filter_map(|i| match &resolved[i] {
            Some(Ok((contrast, groups))) => Some(Coded {
                decl: &decls[i],
                levels: &discovered[i],
                contrast: *contrast,
                groups: groups.clone(),
            }),
            _ => None,
        })
        .collect();

    let mut bindings = LevelBindings::new();
    for (i, ((decl, levels), m)) in decls.iter().zip(&discovered).zip(&measured).enumerate() {
        let (contrast, groups) = match resolved[i].take() {
            Some(r) => r,
            None => resolve_contrast(decl, levels, m, with_occasions, Some(&first_pass)),
        }?;
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
/// keys are equal exactly when the values are `==`, for values holding no `NaN`. The
/// one spelling of level equality the binder ([`write_index_column`]) and the level-table
/// check ([`match_level_table`]) share (#1799).
fn level_key(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| (v + 0.0).to_bits()).collect()
}

/// Why a population's index column for one bound level block is not the index the
/// model's level table gives its records (#1762). The first that applies, in this
/// order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LevelIndexFinding {
    /// No subject carries the index column, so the population was never bound for the
    /// block (#1795). Decided from the subjects alone: a name listed in
    /// `covariate_names` is not an index, so a population listing it and carrying it on
    /// no subject is never bound too. An empty population is never bound.
    NeverBound,
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
    if unbound.len() == population.subjects.len() {
        return Some(LevelIndexFinding::NeverBound);
    }
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
                    // The clock the label's `TIME=` reads too ([`column_value`]), so
                    // the message names one time, not two (#1796).
                    time: Some(subject.readout_time(j)),
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
///
/// `TIME` is the time as written ([`Subject::readout_time`]), the TIME NONMEM keys
/// `$PK`/`$PRED` on (#1959). `obs_times` is the engines' ordering clock: the reader
/// moves a record one ULP below a tie to keep file order (#614, #1810), and a
/// stacked-reset subject's later periods sit on a shifted clock. Keyed on that, one
/// written time split into two cells, and a reset period's label named a time the
/// data never holds (#1796).
fn column_value(subject: &Subject, column: &str, j: usize) -> Option<f64> {
    if column.eq_ignore_ascii_case(TIME_COLUMN) {
        return (j < subject.obs_times.len()).then(|| subject.readout_time(j));
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
/// one subject for a kappa, or under nested levels (#1082) a group of the
/// kappa's own level within one subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unit {
    Subject,
    Occasion,
    /// A group of nested level `L` (outermost 0): the records whose label path
    /// starts with the group's, as `stats::nested_levels` keys a group, so `ARM 1`
    /// of two studies is two units.
    Level(usize),
}

/// The key of the unit observation row `j` of `subject` lies in, within that
/// subject: empty for [`Unit::Subject`], the row's occasion for
/// [`Unit::Occasion`], the row's label path down to the level for
/// [`Unit::Level`]. `None` when the row has none — no occasions, or a missing
/// label — so a kappa counts nowhere there (`fit()` reports the missing
/// occasions or labels later).
fn unit_of(
    subject: &Subject,
    unit: Unit,
    j: usize,
    nested: Option<&NestedLevels>,
) -> Option<Vec<u64>> {
    match unit {
        Unit::Subject => Some(Vec::new()),
        Unit::Occasion => subject.occasions.get(j).map(|&o| vec![u64::from(o)]),
        Unit::Level(l) => {
            let path = label_path(nested?, subject.obs_cov(j))?;
            path.get(..=l).map(<[u64]>::to_vec)
        }
    }
}

/// Whether every level's observation records lie in a single unit (#1678,
/// #1696): one subject for an η, one occasion of one subject for a kappa, one
/// group of the kappa's level of one subject under nested levels.
///
/// This is the data-side half of "is there a random effect at a grouping
/// coarser than or equal to the block's": only then can a unit's random effect
/// carry its levels (#1064). For a one-column block it means each level is one
/// unit's (#1649). It does not ask the leading columns to identify subjects:
/// two subjects of one study on disjoint times nest a `[STUDY, TIME]` block in
/// subjects while sharing its `STUDY` key.
fn levels_nest_in_units(
    decl: &LevelBlockDecl,
    population: &Population,
    unit: Unit,
    nested: Option<&NestedLevels>,
) -> bool {
    let mut owner: HashMap<Vec<u64>, (usize, Vec<u64>)> = HashMap::new();
    for (s, subject) in population.subjects.iter().enumerate() {
        for j in 0..subject.obs_times.len() {
            let level: Option<Vec<u64>> = decl
                .columns()
                .iter()
                .map(|c| column_value(subject, c, j).map(f64::to_bits))
                .collect();
            let (Some(level), Some(u)) = (level, unit_of(subject, unit, j, nested)) else {
                return false;
            };
            let u = (s, u);
            if *owner.entry(level).or_insert_with(|| u.clone()) != u {
                return false;
            }
        }
    }
    true
}

/// Whether a block of two or more columns resolves every subject's
/// observations: no level holds two different observation times of one
/// subject (#1650). Replicates at one time still resolve, since a random
/// effect's effect is the same at both — but only replicates ([`replicates`]):
/// two records at one instant that the data tells apart — one moved a ULP
/// below a co-timed dose, two arms of a study keyed as occasions, a `TRT`
/// that differs, or two compartments — can be moved differently by a random
/// effect that the one level they share moves alike, so the block does not
/// resolve them (#2131). The span certificate then
/// measures, row by row, whether the levels reproduce the random effect.
///
/// Such a block is a free curve per subject at the observed times, so it
/// reproduces any per-subject effect at all, whatever route a random effect
/// takes to `y`.
fn resolves_observations(decl: &LevelBlockDecl, population: &Population, lagged: bool) -> bool {
    if decl.columns().len() < 2 {
        return false;
    }
    population.subjects.iter().all(|subject| {
        let mut seen: Vec<(Vec<f64>, usize)> = Vec::new();
        (0..subject.obs_times.len()).all(|j| {
            let level: Vec<f64> = decl
                .columns()
                .iter()
                .filter_map(|c| column_value(subject, c, j))
                .collect();
            match seen.iter().find(|(l, _)| *l == level) {
                // Replicates, so a random effect moves both alike (#2131).
                Some(&(_, j0)) => replicates(subject, j0, j, lagged),
                None => {
                    seen.push((level, j));
                    true
                }
            }
        })
    })
}

/// Whether observation records `j` and `k` of `subject` are replicates: one
/// instant ([`Subject::same_instant`]) with no dose or reset between them, and
/// the same occasion, compartment and record covariates, so the data gives a
/// random effect no way to move one and not the other.
///
/// One instant is not enough on its own: the reader moves a record one ULP
/// below a co-timed row to keep file order (#614, #1810), and when that row is
/// a dose the two records straddle it, so a random effect on the dose or on
/// the state it changes moves the later one only. A move below a row that
/// changes no state — an `MDV = 1` row — leaves them replicates (#1959).
///
/// A dose can also change the state at its arrival, `dose.time + lag`, which
/// on a `lagged` model ([`lagged_arrivals`]) the θ and η place and the data
/// does not: a moved record can sit just below a lagged arrival with no dose
/// row between the two. So on such a model two records the engine clock
/// separates at all are not replicates, and the span certificate measures
/// them; two at one exact engine time still are, since every arrival at that
/// time acts on both or on neither.
fn replicates(subject: &Subject, j: usize, k: usize, lagged: bool) -> bool {
    fn same<T: PartialEq>(v: &[T], j: usize, k: usize) -> bool {
        v.get(j) == v.get(k)
    }
    let (Some(&a), Some(&b)) = (subject.obs_times.get(j), subject.obs_times.get(k)) else {
        return false;
    };
    let (lo, hi) = (a.min(b), a.max(b));
    let between = |t: f64| lo < t && t <= hi;
    subject.same_instant(j, k)
        && (a == b || !lagged)
        && !subject.doses.iter().any(|d| between(d.time))
        && !subject.reset_times.iter().copied().any(between)
        && same(&subject.occasions, j, k)
        && same(&subject.obs_cmts, j, k)
        && same(&subject.obs_covariates, j, k)
}

/// Whether every covariate in `covariates` is constant within every unit's
/// observation records — the data-side condition of a funnel
/// (`EtaCoupling::funnels`): within each subject for an η, within each
/// occasion of a subject for a kappa, and within each group of its level for a
/// kappa of a nested model.
fn constant_within_units(
    covariates: &[String],
    population: &Population,
    unit: Unit,
    nested: Option<&NestedLevels>,
) -> bool {
    population.subjects.iter().all(|subject| {
        covariates.iter().all(|c| {
            // Missing compares like any other value: a column absent from every
            // record is constant, one present on only some records changes.
            let mut first: Vec<(Vec<u64>, Option<f64>)> = Vec::new();
            (0..subject.obs_times.len()).all(|j| {
                let Some(u) = unit_of(subject, unit, j, nested) else {
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
    /// What one of the random effect's units is called ([`unit_noun`]).
    unit: String,
    /// The unit itself, for [`within_parts_them`].
    unit_kind: Unit,
}

enum How<'a> {
    /// An expression that reads the block and the random effect.
    Site(&'a ScaleShare),
    /// The random effect reaching `y` by a route that never meets the block, on
    /// a block that resolves the observations.
    Route(&'a EtaRoute),
    /// No site the walk can name, but the span certificate measured the random
    /// effect's effect on `y` inside the block's, unit by unit (#1834) — moved
    /// together with the random effects named, when it needs them (#1878).
    Measured(Span, Vec<String>),
}

/// What the span certificate measured for one absorbed random effect, by
/// [`spanned`]: where its combination lies, and `with`, the other random
/// effects (indices into `eta_couplings`) moved together with it — empty when
/// it lies in the levels' span alone (#1878).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Absorption {
    span: Span,
    with: Vec<usize>,
}

/// What the span certificate measured for one random effect (#1834): in every
/// unit, its effect on `y` is a combination of the block's level effects there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Span {
    /// Some combination: a contrast that leaves each group's mean free absorbs
    /// it, and `sum_to_zero_within` may not.
    Levels,
    /// A combination that sums to zero within each `sum_to_zero_within` group,
    /// on a nested block: the within-group contrast absorbs it too, so it is
    /// absorbed under every contrast (#1874 review, F1).
    Within,
}

impl Absorbed<'_> {
    fn name(&self) -> &str {
        &self.coupling.eta
    }

    fn kappa(&self) -> bool {
        self.coupling.kappa
    }

    /// Measured only in combination with other random effects (#1878).
    fn combined(&self) -> bool {
        matches!(&self.how, How::Measured(_, with) if !with.is_empty())
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
            How::Measured(span, with) => format!(
                "the random effect `{}`{} moves `y`, at every record of each {}, exactly as some \
                 combination of this block's levels does{} (measured at the initial estimates and \
                 at a point near them)",
                self.name(),
                together(with),
                self.unit,
                match span {
                    Span::Levels => "",
                    Span::Within => ", one that sums to zero within each group",
                },
            ),
        }
    }
}

/// `, moved together with `A` and `B`,` — the partners of a random effect the
/// span certificate measured jointly (#1878); empty when it has none.
fn together(with: &[String]) -> String {
    let names: Vec<String> = with.iter().map(|n| format!("`{n}`")).collect();
    match names.split_last() {
        None => String::new(),
        Some((last, [])) => format!(", moved together with {last},"),
        Some((last, rest)) => format!(", moved together with {} and {last},", rest.join(", ")),
    }
}

/// What one unit is called in a refusal: `subject`, `occasion`, or under
/// nested levels the level's group (`` `STUDY` group ``).
fn unit_noun(unit: Unit, nested: Option<&NestedLevels>) -> String {
    match unit {
        Unit::Subject => "subject".to_string(),
        Unit::Occasion => "occasion".to_string(),
        Unit::Level(l) => match nested.and_then(|n| n.levels.get(l)) {
            Some(level) => format!("`{}` group", level.name),
            None => "group".to_string(),
        },
    }
}

/// The unit of random effect `k` of `decl` (an index into `eta_couplings`, η
/// first): a subject for an η; for a kappa, an occasion, or under nested levels
/// (#1082) a group of the level the kappa is declared `at`
/// (`NestedLevels::kappa_level`).
fn unit_of_random_effect(decl: &LevelBlockDecl, k: usize, nested: Option<&NestedLevels>) -> Unit {
    if !decl.eta_couplings[k].kappa {
        return Unit::Subject;
    }
    let Some(nested) = nested else {
        return Unit::Occasion;
    };
    let n_eta = decl.eta_couplings.iter().filter(|c| !c.kappa).count();
    match nested.kappa_level.get(k - n_eta) {
        Some(&l) => Unit::Level(l),
        None => Unit::Occasion,
    }
}

/// Per random effect of `decl`, parallel to `eta_couplings`: the unit its
/// levels nest in ([`levels_nest_in_units`]) — a subject for an η, an occasion
/// for a kappa, a group of its level for a kappa of a nested model — or `None`
/// when they do not, and the block cannot absorb it.
fn nesting_units(
    decl: &LevelBlockDecl,
    population: &Population,
    nested: Option<&NestedLevels>,
) -> Vec<Option<Unit>> {
    let mut verdicts: Vec<(Unit, bool)> = Vec::new();
    (0..decl.eta_couplings.len())
        .map(|k| {
            let unit = unit_of_random_effect(decl, k, nested);
            let nests = match verdicts.iter().find(|(u, _)| *u == unit) {
                Some(&(_, v)) => v,
                None => {
                    let v = levels_nest_in_units(decl, population, unit, nested);
                    verdicts.push((unit, v));
                    v
                }
            };
            nests.then_some(unit)
        })
        .collect()
}

/// Whether each unit of `t` lies inside one unit of `p`: the same unit, a
/// subject around any other, or an outer nested level around an inner one.
/// A random effect at `t` is measured against the silent ones at every such
/// `p` ([`spanned`]), the joint oracle's coarser-or-equal random effects: an
/// occasion's kappa against every η, a nested kappa against the kappas of
/// its own and the outer levels (#1878 review).
fn coarser_or_equal(p: Unit, t: Unit) -> bool {
    match (p, t) {
        _ if p == t => true,
        (Unit::Subject, _) => true,
        (Unit::Level(a), Unit::Level(b)) => a <= b,
        _ => false,
    }
}

/// Where the structural walk places a nested random effect's absorption on
/// this data: a funnel whose covariates are constant within `unit`, or, on a
/// block that resolves the observations, any route the random effect takes to
/// `y`. `None` when the walk names no site, which is where the span certificate
/// is measured ([`spanned`]).
fn structural_site<'a>(
    c: &'a EtaCoupling,
    population: &Population,
    unit: Unit,
    nested: Option<&NestedLevels>,
    resolving: bool,
) -> Option<How<'a>> {
    let funnel = c
        .funnels
        .iter()
        .find(|f| constant_within_units(&f.covariates, population, unit, nested));
    if let Some(f) = funnel {
        return Some(How::Site(&f.site));
    }
    let route = c.reach.as_ref().filter(|_| resolving)?;
    Some(match &c.share {
        Some(share) => How::Site(share),
        None => How::Route(route),
    })
}

/// Where the structural walk leaves one random effect of a block on this data.
enum Placement<'a> {
    /// The block's levels do not nest in its units: the block cannot absorb it.
    Unnested,
    /// Nested, and the walk names the site ([`structural_site`]) on this unit.
    Site(How<'a>, Unit),
    /// Nested, and the walk names no site: the span certificate measures it
    /// ([`spanned`]) on this unit.
    Silent(Unit),
}

/// Per random effect of `decl`, parallel to `eta_couplings`: its
/// [`Placement`]. The one source of both gates on the walk's verdict — which
/// random effects [`measure_levels`] measures, and where [`absorbed`] takes a
/// site from the walk and where from the measurement — so the two cannot
/// disagree on which random effects are silent (#1874 review, F12).
fn placements<'a>(
    decl: &'a LevelBlockDecl,
    population: &Population,
    lagged: bool,
    nested: Option<&NestedLevels>,
) -> Vec<Placement<'a>> {
    let units = nesting_units(decl, population, nested);
    let resolving = resolves_observations(decl, population, lagged);
    decl.eta_couplings
        .iter()
        .zip(units)
        .map(|(c, unit)| match unit {
            None => Placement::Unnested,
            Some(unit) => match structural_site(c, population, unit, nested, resolving) {
                Some(how) => Placement::Site(how, unit),
                None => Placement::Silent(unit),
            },
        })
        .collect()
}

/// Every random effect the block absorbs (#1649, #1678, #1696, #1834): the
/// block's levels nest in the random effect's units ([`levels_nest_in_units`]),
/// and either the walk names a site ([`structural_site`]) or, where it names
/// none, `spanned` — parallel to `eta_couplings`, from [`spanned`] — measured
/// the absorption. A named site is preferred: it tells the user where to look.
/// An η and a kappa follow the same rule, each on its own unit.
fn absorbed<'a>(
    decl: &'a LevelBlockDecl,
    population: &Population,
    measured: &Measured,
) -> Vec<Absorbed<'a>> {
    let spanned = &measured.spanned;
    let nested = measured.nested.as_ref();
    placements(decl, population, measured.lagged, nested)
        .into_iter()
        .enumerate()
        .filter_map(|(k, placement)| {
            let (how, unit) = match placement {
                Placement::Unnested => return None,
                Placement::Site(how, unit) => (how, unit),
                Placement::Silent(unit) => {
                    let found = spanned.get(k).cloned().flatten()?;
                    let with = found
                        .with
                        .iter()
                        .map(|&j| decl.eta_couplings[j].eta.clone())
                        .collect();
                    (How::Measured(found.span, with), unit)
                }
            };
            let readers = decl.eta_readers.get(k).map_or(&[][..], |r| r.as_slice());
            Some(Absorbed {
                coupling: &decl.eta_couplings[k],
                readers,
                how,
                unit: unit_noun(unit, nested),
                unit_kind: unit,
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
/// prediction and its residual-magnitude multipliers. The η and κ are spread
/// around `re` by [`probe_res`], as the span certificate and the readership
/// check spread them, so a read gated by a difference of random effects does
/// not cancel at the moved point (#1852); random effect `bump` (an η index, or
/// `n_eta` + a κ index, on every occasion group) is moved by `RE_STEP` more.
fn record_values(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    re: f64,
    bump: Option<usize>,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let groups = match model.n_kappa {
        0 => 1,
        _ => crate::stats::stacked_layout::group_count(subject).max(1),
    };
    let (mut eta, mut kappa) = probe_res(re, model, groups);
    match bump {
        Some(r) if r < model.n_eta => eta[r] += RE_STEP,
        Some(r) => kappa.iter_mut().for_each(|k| k[r - model.n_eta] += RE_STEP),
        None => {}
    }
    values_at(model, subject, theta, &eta, &kappa)
}

/// [`record_values`] at explicit random effects: `eta`, and `kappa` per
/// occasion group in [`crate::stats::likelihood::iov_occasion_groups`] order.
fn values_at(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    kappa: &[Vec<f64>],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mult = model.ruv_obs_mult(subject, theta).unwrap_or_default();
    (predictions_at(model, subject, theta, eta, kappa), mult)
}

/// The predictions of [`values_at`] alone.
fn predictions_at(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    kappa: &[Vec<f64>],
) -> Vec<f64> {
    if model.n_kappa > 0 {
        crate::pk::predict_iov(model, subject, theta, eta, kappa)
    } else {
        crate::pk::compute_predictions_with_tv(model, subject, theta, eta)
    }
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
/// (#1822); the second element is the random-effect level, which every check
/// spreads per random effect the same way ([`probe_res`], [`probe_etas`]), so a
/// difference of random effects does not cancel at the moved point (#1852).
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

/// Per level of `decl` (in `levels` order): the index of its θ in `model` —
/// the model bound with every block under `contrast = none` — or `None` when
/// that θ is `FIX` or absent. A fixed level is never measured: it has no
/// direction for the data to estimate. Shared by [`dead_levels`] and
/// [`spanned`], so the two checks read the same levels (#1874 review, F12).
fn free_level_thetas(
    model: &CompiledModel,
    decl: &LevelBlockDecl,
    levels: &[Level],
) -> Vec<Option<usize>> {
    let p = &model.default_params;
    levels
        .iter()
        .map(|level| {
            let name = format!("{}[{}]", decl.name(), level.label(decl.columns()));
            let k = model.theta_names.iter().position(|n| *n == name)?;
            (!p.theta_fixed.get(k).copied().unwrap_or(false)).then_some(k)
        })
        .collect()
}

/// Rule 1 (#1679): the non-`FIX` levels of `decl` whose θ the likelihood never
/// reads, and which random effects read their records.
///
/// `model` is the model bound with every block under `contrast = none`, so each
/// level has a θ of its own. A level is dead when a finite step in its θ leaves
/// every value [`record_values`] returns bitwise unchanged, on every subject
/// that carries it, at both of two points: the initial θ with η = κ = 0, and a
/// deterministic jitter of every non-`FIX` θ inside its bounds with the random
/// effects spread around 0.05 by [`probe_res`]. One point is not enough — a θ initialised at 0 can switch the block
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
    for (i, theta_k) in free_level_thetas(model, decl, levels)
        .into_iter()
        .enumerate()
    {
        let Some(k) = theta_k else {
            continue;
        };
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

/// Under this relative residual a unit's random-effect column lies in its
/// levels' span (#1834). Measured over every unit of the `absorption` tests,
/// this module's own differences, on Linux aarch64 (`tools/linux-test.sh`,
/// `--profile ci-test`) and macOS arm64 debug alike to the digits printed:
/// absorbed units reach at most 2.9e-11 (#1874 review F1's `ETA_E0 * (OCC -
/// 2)`), identified ones at least 9.0e-3 (#1848's Q1, a parameter read before it
/// is overwritten); R11, the closest identified #1834 twin, 3.7e-2. The constant
/// sits 4.5 decades above the one and 3.9 below the other. The joint residuals
/// (#1878), over the same tests on macOS arm64 debug, fall on the same sides:
/// absorbed at most 3.0e-11 (`ETA_E0` and `ETA_X` under `(ETA_E0 - ETA_X)²`),
/// identified at least 9.3e-3.
const SPAN_TOL: f64 = 1e-6;

/// The central-difference step of the span certificate: on a random effect,
/// and, times `max(|θ|, 1)`, on a level's θ.
const SPAN_STEP: f64 = 1e-5;

/// Per record, the central difference of the predictions; `None` when the two
/// sides differ in length. The predictions only, as the structural walk judges
/// absorption on the mean: a random effect never enters the residual
/// multipliers, so stacking them would let a block that also scales the
/// residual error bind where its mean absorbs the random effect.
fn record_diffs(plus: &[f64], minus: &[f64], h: f64) -> Option<Vec<f64>> {
    (plus.len() == minus.len()).then(|| {
        plus.iter()
            .zip(minus)
            .map(|(a, b)| (a - b) / (2.0 * h))
            .collect()
    })
}

/// `‖g − P g‖ / ‖g‖`, `P` the orthogonal projection onto the span of `cols`;
/// `None` when `g` is zero, so the unit measures nothing. A non-finite entry
/// comes back as `NaN`, which no tolerance accepts.
fn span_residual(g: &[f64], cols: &[Vec<f64>]) -> Option<f64> {
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let norm = |a: &[f64]| dot(a, a).sqrt();
    // One modified Gram–Schmidt pass: a unit's level columns are nearly
    // orthogonal, since nested levels hold disjoint records.
    let project_out = |v: &mut Vec<f64>, basis: &[Vec<f64>]| {
        for b in basis {
            let d = dot(v, b);
            v.iter_mut().zip(b).for_each(|(x, y)| *x -= d * y);
        }
    };
    let g_norm = norm(g);
    if g_norm == 0.0 {
        return None;
    }
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for c in cols {
        let c_norm = norm(c);
        if c_norm == 0.0 || !c_norm.is_finite() {
            continue;
        }
        let mut v = c.clone();
        project_out(&mut v, &basis);
        let n = norm(&v);
        // A column already in the span adds no direction.
        if n > 1e-12 * c_norm {
            basis.push(v.into_iter().map(|x| x / n).collect());
        }
    }
    let mut r = g.to_vec();
    project_out(&mut r, &basis);
    Some(norm(&r) / g_norm)
}

/// The span certificate (#1834): for each random effect in `silent` (an index
/// into `eta_couplings` and the unit its levels nest in) — one the walk names
/// no site for — whether, in every unit, its effect on `y` lies in the span of
/// the block's effect on that unit's records, measured on `model` (every block
/// under `contrast = none`). Parallel to `eta_couplings`.
///
/// Proportionality is a property of the functions, not of how they are
/// written: `if (OCC > 1) S else 2 * (S)`, `S + S` against `2 * S`, or a factor
/// distributed over a sum all absorb the random effect with no expression
/// that reads both, and no syntactic rule is complete for that. So the binder
/// measures it, on the data, as it measures dead levels: per unit, the relative
/// residual of the random effect's column against the columns of the levels on
/// the unit's records, by central differences of the predictions
/// (`record_diffs` says why not the residual multipliers), at both
/// [`probe_points`], with the random effects spread by [`probe_res`]. One
/// point is not enough: at η = 0 an `ETA * ETA * OCC` term has derivative 0
/// and looks proportional. A unit whose column is zero measures nothing; a
/// random effect counts as absorbed only when some unit measured it at each
/// point and every unit's residual is under [`SPAN_TOL`]. Anything non-finite
/// fails the tolerance, so an inconclusive point binds, never refuses.
///
/// The span is joint over the silent random effects of one unit (#1878): a
/// block can absorb a combination of them that none reaches alone —
/// `S + (ETA_E0 - ETA_X)² · OCC` moves `y` with `OCC` under either, while the
/// two shifted together move it by a constant the level takes up. So each
/// random effect's column is measured, unit by unit, against the level
/// columns and the other silent random effects' columns of its unit there:
/// the rank of `[Z_silent | X_levels]` against `X_levels` one column at a
/// time. The others join only in a unit where they do not carry the column on
/// their own: a random effect inside their span is not identified against
/// them, which is no fault of the block (the joint oracle's baseline rank).
///
/// A coarser random effect — an η for an occasion's kappa, an outer level's
/// kappa for a nested one ([`coarser_or_equal`]) — couples the units: an η
/// and every occasion's kappa shifted against it is a combination too, but
/// the η takes one value across the occasions, so the units cannot be judged
/// one by one (#1878 review r1, r2). A random effect some unit keeps outside
/// is measured over the whole subject instead: whether the levels widen the
/// combinations of its unit columns the coarser columns match
/// ([`shared_dim`]). Its within verdict is then left to [`carried`], which
/// judges each contrast on the same joint rank.
///
/// One that needed partners comes back with them as its partners
/// ([`Absorption::with`]).
///
/// On a nested block a second residual decides whether `sum_to_zero_within`
/// rescues it (#1874 review, F1): against the differences of the unit's level
/// columns inside each within-group, the directions that contrast leaves free.
/// A random effect inside those is [`Span::Within`], absorbed under every
/// contrast; one only inside the full span is [`Span::Levels`].
fn spanned(
    model: &CompiledModel,
    decl: &LevelBlockDecl,
    levels: &[Level],
    population: &Population,
    silent: &[(usize, Unit)],
) -> Vec<Option<Absorption>> {
    let tree = model.nested_levels();
    let mut out = vec![None; decl.eta_couplings.len()];
    if silent.is_empty() {
        return out;
    }
    let column = level_index_column(decl.name());
    // Per level index (1-based, as written to the data): its free θ and its
    // within-group.
    let groups = assign_groups(decl, levels, LevelContrast::SumToZeroWithin);
    let level_theta: HashMap<u64, (usize, usize)> = free_level_thetas(model, decl, levels)
        .into_iter()
        .enumerate()
        .filter_map(|(i, k)| Some((((i + 1) as f64).to_bits(), (k?, groups[i]))))
        .collect();
    let nested = decl.columns().len() >= 2;
    // Per silent random effect: in the levels' span, and in the within span.
    let mut holds: Vec<bool> = vec![true; silent.len()];
    let mut within: Vec<bool> = vec![nested; silent.len()];
    // Per silent random effect: some unit needed the others to reach the span.
    let mut partnered: Vec<bool> = vec![false; silent.len()];
    for (theta, re) in probe_points(model) {
        if holds.iter().all(|h| !h) {
            break;
        }
        let mut measured = vec![false; silent.len()];
        for subject in &population.subjects {
            if holds.iter().all(|h| !h) {
                break;
            }
            let layout = crate::stats::stacked_layout::StackedLayout::of(model, subject);
            let cells = NestedLayout::of(model, subject);
            let (eta, kappa) = subject_probe_res(re, model, cells.as_ref(), layout.n_groups());
            let at = |th: &[f64], et: &[f64], ka: &[Vec<f64>]| {
                predictions_at(model, subject, th, et, ka)
            };
            let n_obs = subject.obs_times.len();
            let level_of = |j: usize| {
                subject
                    .obs_covariates
                    .get(j)
                    .and_then(|m| m.get(&column))
                    .or_else(|| subject.covariates.get(&column))
                    .map(|v| v.to_bits())
            };
            // The level columns this subject's records read, each over every record.
            let mut level_cols: HashMap<u64, Option<Vec<f64>>> = HashMap::new();
            for j in 0..n_obs {
                let Some(index) = level_of(j) else { continue };
                let Some(&(k, _)) = level_theta.get(&index) else {
                    continue;
                };
                level_cols.entry(index).or_insert_with(|| {
                    let h = SPAN_STEP * theta[k].abs().max(1.0);
                    let (mut up, mut down) = (theta.clone(), theta.clone());
                    up[k] += h;
                    down[k] -= h;
                    record_diffs(&at(&up, &eta, &kappa), &at(&down, &eta, &kappa), h)
                });
            }
            // The silent random effects of one unit are measured together
            // (#1878): a combination of them can lie in the levels' span where
            // none does alone, as `ETA_E0` and `ETA_X` shifted together do
            // under `(ETA_E0 - ETA_X)²`. Those at a coarser unit are measured
            // with them too, but over the whole subject: an η and every
            // occasion's kappa shifted against it is a combination as well,
            // and the η takes one value across the occasions.
            let mut kinds: Vec<Unit> = Vec::new();
            for &(_, unit) in silent {
                if !kinds.contains(&unit) {
                    kinds.push(unit);
                }
            }
            // The column of silent random effect `s` moved on its unit `key`,
            // over every record; `None` where it cannot be measured.
            let re_column = |s: usize, key: &[u64]| -> Option<Vec<f64>> {
                let (k, own) = silent[s];
                let (mut eta_up, mut eta_down) = (eta.clone(), eta.clone());
                let (mut ka_up, mut ka_down) = (kappa.clone(), kappa.clone());
                if k < model.n_eta {
                    eta_up[k] += SPAN_STEP;
                    eta_down[k] -= SPAN_STEP;
                } else {
                    // A kappa moves its own occasion only; under nested
                    // levels, every cell of its group (#1082).
                    let groups = unit_groups(&layout, cells.as_ref(), own, key);
                    if groups.is_empty() {
                        return None;
                    }
                    for g in groups {
                        ka_up[g][k - model.n_eta] += SPAN_STEP;
                        ka_down[g][k - model.n_eta] -= SPAN_STEP;
                    }
                }
                record_diffs(
                    &at(&theta, &eta_up, &ka_up),
                    &at(&theta, &eta_down, &ka_down),
                    SPAN_STEP,
                )
            };
            // The records of each unit of `unit` in this subject.
            let units_of = |unit: Unit| -> Vec<(Vec<u64>, Vec<usize>)> {
                let mut units: Vec<(Vec<u64>, Vec<usize>)> = Vec::new();
                for j in 0..n_obs {
                    let Some(u) = unit_of(subject, unit, j, tree) else {
                        continue;
                    };
                    match units.iter_mut().find(|(u0, _)| *u0 == u) {
                        Some((_, rows)) => rows.push(j),
                        None => units.push((u, vec![j])),
                    }
                }
                units
            };
            for unit in kinds {
                let members: Vec<usize> =
                    (0..silent.len()).filter(|&s| silent[s].1 == unit).collect();
                if members.iter().all(|&s| !holds[s]) {
                    continue;
                }
                let coarser: Vec<usize> = (0..silent.len())
                    .filter(|&s| silent[s].1 != unit && coarser_or_equal(silent[s].1, unit))
                    .collect();
                // Per member, its column on each unit, zero off the unit's
                // records; and whether some unit kept it outside the span of
                // the levels and the other members there.
                let mut unit_cols: Vec<Vec<Vec<f64>>> = vec![Vec::new(); members.len()];
                let mut outside: Vec<bool> = vec![false; members.len()];
                for (u, rows) in units_of(unit) {
                    let restrict = |col: &[f64]| -> Vec<f64> {
                        rows.iter().filter_map(|&j| col.get(j).copied()).collect()
                    };
                    let mut read: Vec<u64> = rows.iter().filter_map(|&j| level_of(j)).collect();
                    read.sort_unstable();
                    read.dedup();
                    // The unit's level columns, and per within-group the
                    // differences from its first level there.
                    let mut cols: Vec<Vec<f64>> = Vec::new();
                    let mut firsts: Vec<(usize, Vec<f64>)> = Vec::new();
                    let mut diffs: Vec<Vec<f64>> = Vec::new();
                    let mut failed = false;
                    for index in read {
                        let (Some(&(_, group)), Some(col)) =
                            (level_theta.get(&index), level_cols.get(&index))
                        else {
                            continue;
                        };
                        let Some(col) = col else {
                            failed = true;
                            continue;
                        };
                        let col = restrict(col);
                        match firsts.iter().find(|(g, _)| *g == group) {
                            Some((_, first)) => {
                                diffs.push(col.iter().zip(first).map(|(a, b)| a - b).collect())
                            }
                            None => firsts.push((group, col.clone())),
                        }
                        cols.push(col);
                    }
                    if failed {
                        members.iter().for_each(|&s| holds[s] = false);
                        break;
                    }
                    // Each member's column on the unit's records; `None` where
                    // it cannot be measured.
                    let full: Vec<Option<Vec<f64>>> =
                        members.iter().map(|&s| re_column(s, &u)).collect();
                    let re_cols: Vec<Option<Vec<f64>>> =
                        full.iter().map(|c| c.as_deref().map(restrict)).collect();
                    for (i, c) in full.iter().enumerate() {
                        if let Some(c) = c {
                            let mut masked = vec![0.0; c.len()];
                            rows.iter().for_each(|&j| masked[j] = c[j]);
                            unit_cols[i].push(masked);
                        }
                    }
                    for (i, &s) in members.iter().enumerate() {
                        if !holds[s] {
                            continue;
                        }
                        let Some(g) = &re_cols[i] else {
                            holds[s] = false;
                            continue;
                        };
                        let others: Vec<Vec<f64>> = re_cols
                            .iter()
                            .enumerate()
                            .filter(|&(j, _)| j != i)
                            .filter_map(|(_, c)| c.clone())
                            .collect();
                        // The others join the span only where they do not
                        // carry `g` on their own: a random effect inside their
                        // span is theirs to identify, not the block's to absorb
                        // — the joint oracle's baseline rank.
                        let joint = !others.is_empty()
                            && span_residual(g, &others).is_some_and(|r| r >= SPAN_TOL);
                        let with = |base: &[Vec<f64>]| -> Vec<Vec<f64>> {
                            let mut v = base.to_vec();
                            if joint {
                                v.extend(others.iter().cloned());
                            }
                            v
                        };
                        match span_residual(g, &with(&cols)) {
                            None => {}
                            Some(r) if r < SPAN_TOL => {
                                measured[s] = true;
                                if joint && !span_residual(g, &cols).is_some_and(|r| r < SPAN_TOL) {
                                    partnered[s] = true;
                                }
                                if within[s]
                                    && !span_residual(g, &with(&diffs))
                                        .is_some_and(|r| r < SPAN_TOL)
                                {
                                    within[s] = false;
                                }
                            }
                            // Outside on this unit: the coarser random effects
                            // may still reach it over the subject, below.
                            Some(_) if !coarser.is_empty() => outside[i] = true,
                            Some(_) => holds[s] = false,
                        }
                    }
                    if members.iter().all(|&s| !holds[s]) {
                        break;
                    }
                }
                // A member some unit kept outside, against the coarser random
                // effects over the subject (#1878 review r1, r2): the
                // dimension of `span K ∩ span C` outside the levels and the
                // other members — `K` its unit columns, `C` the coarser
                // columns, one per unit of theirs, so an η's coefficient is
                // one across the occasions — against the same dimension
                // outside the other members alone, the joint oracle's
                // baseline. Where it grows the block absorbs a combination of
                // them, which the per-unit within residual cannot judge (the
                // coarser random effect couples the units): it is left to
                // [`carried`], contrast by contrast.
                if !outside.iter().any(|o| *o) {
                    continue;
                }
                let mut coarse: Vec<Vec<f64>> = Vec::new();
                let mut levels: Vec<Vec<f64>> = Vec::new();
                let mut failed = false;
                for &p in &coarser {
                    for (key, _) in units_of(silent[p].1) {
                        match re_column(p, &key) {
                            Some(c) => coarse.push(c),
                            None => failed = true,
                        }
                    }
                }
                for j in 0..n_obs {
                    let Some(index) = level_of(j) else { continue };
                    if let Some(Some(c)) = level_cols.get(&index) {
                        if !levels.contains(c) {
                            levels.push(c.clone());
                        }
                    }
                }
                for (i, &s) in members.iter().enumerate() {
                    if !outside[i] || !holds[s] {
                        continue;
                    }
                    let same: Vec<Vec<f64>> = unit_cols
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .flat_map(|(_, c)| c.iter().cloned())
                        .collect();
                    let k = &unit_cols[i];
                    let grows = !failed && {
                        let base: Vec<Vec<f64>> = levels.iter().chain(&same).cloned().collect();
                        shared_dim(&base, k, &coarse) > shared_dim(&same, k, &coarse)
                    };
                    if grows {
                        measured[s] = true;
                        partnered[s] = true;
                        within[s] = false;
                    } else {
                        holds[s] = false;
                    }
                }
            }
        }
        for (h, m) in holds.iter_mut().zip(&measured) {
            *h &= *m;
        }
    }
    for (s, &(k, unit)) in silent.iter().enumerate() {
        if !holds[s] {
            continue;
        }
        let span = if within[s] {
            Span::Within
        } else {
            Span::Levels
        };
        // The partners: the others of its unit or coarser absorbed with it,
        // or failing that every other silent random effect there.
        let with = if partnered[s] {
            let others = |also: &dyn Fn(usize) -> bool| -> Vec<usize> {
                silent
                    .iter()
                    .enumerate()
                    .filter(|&(j, &(_, u))| j != s && coarser_or_equal(u, unit) && also(j))
                    .map(|(_, &(k, _))| k)
                    .collect()
            };
            let absorbed = others(&|j| holds[j] && partnered[j]);
            if absorbed.is_empty() {
                others(&|_| true)
            } else {
                absorbed
            }
        } else {
            Vec::new()
        };
        out[k] = Some(Absorption { span, with });
    }
    out
}

/// The random effects the level checks move at a probe point whose level is
/// `re`: the η by [`probe_etas`], and each κ distinct from them and per
/// occasion group, so a read gated by a difference of random effects does not
/// cancel at the moved point (#1874 review, F3; #1822 review r1).
fn probe_res(re: f64, model: &CompiledModel, groups: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let eta = probe_etas(re, model.n_eta);
    let kappa = (0..groups)
        .map(|g| {
            (0..model.n_kappa)
                .map(|q| re * (1.0 + 0.3 * (model.n_eta + q) as f64) * (1.0 + 0.11 * g as f64))
                .collect()
        })
        .collect();
    (eta, kappa)
}

/// [`probe_res`] for one subject with `n_groups` occasion groups. Under nested
/// levels (#1082) the κ are spread by the group of each κ's own level, not by
/// cell, so every cell of one group reads the same value: the probe is a point
/// the model can take, where a study's κ is one number shared by its arms.
fn subject_probe_res(
    re: f64,
    model: &CompiledModel,
    cells: Option<&NestedLayout>,
    n_groups: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let (eta, mut kappa) = probe_res(re, model, n_groups.max(1));
    let (Some(levels), Some(cells)) = (model.nested_levels(), cells) else {
        return (eta, kappa);
    };
    let widest = (0..levels.levels.len())
        .map(|l| cells.level_count(l))
        .max()
        .unwrap_or(1);
    let (_, by_group) = probe_res(re, model, widest.max(1));
    for (g, row) in kappa.iter_mut().enumerate().take(cells.n_cells()) {
        for (q, v) in row.iter_mut().enumerate() {
            if let Some(&l) = levels.kappa_level.get(q) {
                *v = by_group[cells.cell_group(g, l)][q];
            }
        }
    }
    (eta, kappa)
}

/// The κ groups — positions in the `kappas` [`predict_iov`](crate::pk::predict_iov)
/// reads — that a kappa of unit `key` (of `unit`, [`unit_of`]) moves: the
/// occasion's own group, or under nested levels every cell of the level's group
/// (#1082), a cell with no observation in it too. Empty when the key names no
/// group.
fn unit_groups(
    layout: &crate::stats::stacked_layout::StackedLayout,
    cells: Option<&NestedLayout>,
    unit: Unit,
    key: &[u64],
) -> Vec<usize> {
    match unit {
        Unit::Subject => Vec::new(),
        Unit::Occasion => key
            .first()
            .and_then(|&o| u32::try_from(o).ok())
            .and_then(|o| layout.group_of(o))
            .into_iter()
            .collect(),
        Unit::Level(l) => cells.map_or_else(Vec::new, |cells| {
            (0..cells.n_cells())
                .filter(|&g| {
                    let path: Vec<u64> = cells.labels(g, l).iter().map(|v| v.to_bits()).collect();
                    path == key
                })
                .collect()
        }),
    }
}

/// Per unit of kappa `q` in one subject with `n_groups` occasion groups, the κ
/// groups it moves ([`unit_groups`]): each occasion group on its own, or under
/// nested levels the cells of each group of the kappa's level. `None` for a
/// nested model whose subject has no cells, which measures nothing.
fn kappa_units(
    model: &CompiledModel,
    cells: Option<&NestedLayout>,
    n_groups: usize,
    q: usize,
) -> Option<Vec<Vec<usize>>> {
    let Some(levels) = model.nested_levels() else {
        return Some((0..n_groups.max(1)).map(|g| vec![g]).collect());
    };
    let (cells, &l) = (cells?, levels.kappa_level.get(q)?);
    Some(
        (0..cells.level_count(l))
            .map(|group| {
                (0..cells.n_cells())
                    .filter(|&g| cells.cell_group(g, l) == group)
                    .collect()
            })
            .collect(),
    )
}

/// An orthonormal basis of the span of `cols` by modified Gram–Schmidt with one
/// re-orthogonalisation pass: the population-wide columns [`carried`] stacks are
/// far from orthogonal (a θ read on every record against the levels'), where
/// one pass loses the orthogonality the residual relies on. A zero or
/// non-finite column, or one already in the span, adds no direction.
#[cfg(test)]
fn orthonormal_basis(cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for c in cols {
        let c_norm = dot(c, c).sqrt();
        if c_norm == 0.0 || !c_norm.is_finite() {
            continue;
        }
        let mut v = c.clone();
        project_out(&mut v, &basis);
        let n = dot(&v, &v).sqrt();
        if n > DROP_TOL * c_norm {
            basis.push(v.into_iter().map(|x| x / n).collect());
        }
    }
    basis
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The rank of `cols` at [`SPAN_TOL`]: how many keep at least that much of
/// their own norm against the ones before them, by modified Gram–Schmidt with
/// one re-orthogonalisation pass. Columns scaled to the norm the tolerance is
/// relative to — [`carried`]'s remnants, each divided by its column's norm — so
/// a column nearly inside the span already removed counts as none (#1878).
fn rank(cols: &[Vec<f64>]) -> usize {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for c in cols {
        let mut v = c.clone();
        project_out(&mut v, &basis);
        let n = dot(&v, &v).sqrt();
        if n.is_finite() && n >= SPAN_TOL {
            basis.push(v.into_iter().map(|x| x / n).collect());
        }
    }
    basis.len()
}

/// `dim(span K ∩ span C)` outside `span B`: the combinations of the columns
/// `k` that the columns `c` match, up to the columns `b` — counted as
/// `rank[B|K] + rank[B|C] − rank[B|K|C] − rank B`, each by [`rank`] on the
/// columns scaled to unit norm. A non-finite column makes the count 0: an
/// inconclusive measurement binds (#1878).
fn shared_dim(b: &[Vec<f64>], k: &[Vec<f64>], c: &[Vec<f64>]) -> usize {
    let unit = |v: &Vec<f64>| -> Option<Vec<f64>> {
        let n = dot(v, v).sqrt();
        (n > 0.0).then(|| v.iter().map(|x| x / n).collect())
    };
    let all: Vec<&Vec<f64>> = b.iter().chain(k).chain(c).collect();
    if all.iter().any(|v| v.iter().any(|x| !x.is_finite())) {
        return 0;
    }
    let scaled = |cols: &[&[Vec<f64>]]| -> Vec<Vec<f64>> {
        cols.iter()
            .flat_map(|c| c.iter())
            .filter_map(unit)
            .collect()
    };
    let rb = rank(&scaled(&[b]));
    let rbk = rank(&scaled(&[b, k]));
    let rbc = rank(&scaled(&[b, c]));
    let rbkc = rank(&scaled(&[b, k, c]));
    (rbk + rbc).saturating_sub(rbkc + rb)
}

/// `v` minus its projection on the orthonormal `basis`, projected twice (one
/// re-orthogonalisation pass, as [`orthonormal_basis`] builds it).
fn project_out(v: &mut [f64], basis: &[Vec<f64>]) {
    project_out_coefs(v, basis);
}

/// [`project_out`], returning the coefficient of `v` on each basis vector,
/// both passes summed.
fn project_out_coefs(v: &mut [f64], basis: &[Vec<f64>]) -> Vec<f64> {
    let mut coefs = vec![0.0; basis.len()];
    for _ in 0..2 {
        for (b, c) in basis.iter().zip(coefs.iter_mut()) {
            let d = dot(v, b);
            v.iter_mut().zip(b).for_each(|(x, y)| *x -= d * y);
            *c += d;
        }
    }
    coefs
}

/// A population-wide column of [`carried`], as its nonzero per-subject
/// pieces: `(subject, values on that subject's records)`. A level's, a
/// unit's or a θ read by one subject alone is one piece.
type Pieces = Vec<(usize, Vec<f64>)>;

/// `a − b`, piece by piece; a subject in both keeps one piece.
fn pieces_minus(a: &Pieces, b: &Pieces) -> Pieces {
    let mut out = a.clone();
    for (s, v) in b {
        match out.iter_mut().find(|(t, _)| t == s) {
            Some((_, w)) => w.iter_mut().zip(v).for_each(|(x, y)| *x -= y),
            None => out.push((*s, v.iter().map(|y| -y).collect())),
        }
    }
    out
}

/// Whether two columns are the same vector — `a − b` is zero. A raw column
/// holds only pieces with a nonzero entry, so that is the same subjects with
/// the same values.
fn pieces_equal(a: &Pieces, b: &Pieces) -> bool {
    fn sorted(p: &Pieces) -> Vec<&(usize, Vec<f64>)> {
        let mut v: Vec<&(usize, Vec<f64>)> = p.iter().collect();
        v.sort_by_key(|(s, _)| *s);
        v
    }
    a.len() == b.len() && sorted(a) == sorted(b)
}

/// Under this ratio a combination of sum-to-zero functionals is representable
/// on the columns of a [`BlockBasis`] (#1962): the misfit of the equations of
/// the columns Gram–Schmidt dropped as already in the span, each over its
/// column's norm, as a singular value of the misfit part of the orthonormalised
/// `(t, misfit)` pairs. A dropped column lies in the span to [`DROP_TOL`] of its
/// norm, so a representable combination leaves a misfit of that order; one that
/// is not leaves one of the order of the functional. Measured over the 2464
/// bases the `api::levels` tests build in [`carried`], macOS arm64 debug: a
/// representable combination reaches at most 4.2e-11, one that is not at least
/// 8.5e-2; the constant sits 4.4 decades above the one and 4.9 below the other.
/// In the same run the loose rank count's direction test ([`SPAN_TOL`] on a
/// group's last column against `A`) saw at most 5.3e-11 where the column is
/// held and at least 1.4e-2 where it adds a direction.
const CONSISTENT_TOL: f64 = 1e-6;

/// A column keeping at most this fraction of its norm against the columns
/// before it adds no direction to a [`BlockBasis`].
const DROP_TOL: f64 = 1e-12;

#[cfg(test)]
thread_local! {
    /// Per [`BlockBasis`] built on this thread, its population-wide vector
    /// count: the scale #1962 bounds, read by the tests without a clock.
    static DENSE_COUNTS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A column of a [`BlockBasis`] as Gram–Schmidt took it: its coefficients on
/// the basis vectors before it, as `(first coordinate, coefficients)`
/// segments, and the coordinate and norm of the basis vector it added, if it
/// added one.
struct Taken {
    coefs: Vec<(usize, Vec<f64>)>,
    added: Option<(usize, f64)>,
    norm: f64,
    group: Option<usize>,
}

/// An orthonormal basis of the span of [`Pieces`] columns that keeps their
/// subject structure (#1961 review, F1): columns on one subject's records are
/// orthonormalised per subject — they are orthogonal across subjects for free
/// — and only the columns spanning several subjects (a θ read on every
/// record, a level read by several subjects) become population-wide vectors,
/// orthogonalised against those per-subject bases first. The span is the one
/// a dense basis of every column would have; the cost is in the
/// population-wide columns alone.
///
/// A column may belong to a sum-to-zero group (#1962). The basis then spans
/// `A`: the ungrouped columns, and each group's columns under the constraint
/// that their coefficients sum to zero — the span of each column's difference
/// against its group's last, which is what the contrast codes. Those
/// differences cross subjects whenever the levels do, one population-wide
/// vector per level and a cost cubic in the subjects, so the basis is built on
/// the raw columns instead, spanning `Q ⊇ A`, and the directions the groups
/// remove are kept apart: an orthonormal `T` with `Q = A ⊕ T`, one direction
/// per group at most. `R_A z` then has `‖R_A z‖² = ‖R_Q z‖² + ‖Tᵀ z‖²`.
///
/// `t ∈ Q` is orthogonal to `A` exactly when it is orthogonal to every
/// ungrouped column and every column of group `g` has the same inner product
/// `λ_g` with it. In the coordinates of the orthonormal basis that is a
/// triangular system in the order Gram–Schmidt took the columns — per subject,
/// then population-wide — for the columns that added a basis vector, and an
/// equation to satisfy for each it dropped: solved per group for `λ = e_g`,
/// the `λ` whose dropped equations hold (to [`CONSISTENT_TOL`]) give `T`. A
/// group none of whose functionals is representable — the intercept that
/// carries the levels' mean — removes nothing, and `A = Q` there.
struct BlockBasis {
    /// Per subject, the orthonormal basis of its one-subject columns.
    local: Vec<Vec<Vec<f64>>>,
    /// The orthonormal population-wide part.
    dense: Vec<Vec<f64>>,
    offsets: Vec<usize>,
    n_rows: usize,
    /// Per nonzero finite column, `‖c − P c‖ / ‖c‖` against the columns
    /// orthonormalised before it.
    remnants: Vec<f64>,
    /// Per subject, the coordinate of its first local basis vector; the
    /// population-wide ones follow every subject's.
    local_at: Vec<usize>,
    /// `T`, orthonormal, in the coordinates of the basis.
    removed: Vec<Vec<f64>>,
    /// What the grouped columns change in [`BlockBasis::aliased`]'s count
    /// against the raw columns' remnants.
    coded_shift: isize,
    /// How many groups the joint solve for `T` took: the scale review r1 of
    /// #1962 bounds.
    #[cfg(test)]
    joint: usize,
}

impl BlockBasis {
    #[cfg(test)]
    fn new(cols: &[Pieces], lens: &[usize], offsets: &[usize]) -> Self {
        Self::grouped(cols, &vec![None; cols.len()], lens, offsets)
    }

    /// The basis of `cols`, column `j` in sum-to-zero group `groups[j]` (a
    /// group's last column is its last in `cols`).
    ///
    /// A group whose columns all sit on one subject — a `sum_to_zero_within`
    /// group of one subject's levels — enters as its difference columns, which
    /// sit on that subject too and stay per subject, as the coded columns did
    /// before #1962 (review r1): only a group crossing subjects goes through
    /// `T`, whose joint solve is dense in the groups. Under
    /// `sum_to_zero_within` with one group per subject that is none of them;
    /// a group that crosses subjects has at least one population-wide
    /// difference column, so `T` never costs more than those did.
    fn grouped(
        cols: &[Pieces],
        groups: &[Option<usize>],
        lens: &[usize],
        offsets: &[usize],
    ) -> Self {
        // Per group: its last column, and whether every column sits on one
        // subject (a zero column sits anywhere).
        let mut last: HashMap<usize, usize> = HashMap::new();
        let mut subject: HashMap<usize, Option<usize>> = HashMap::new();
        for (j, (c, g)) in cols.iter().zip(groups).enumerate() {
            let Some(g) = *g else { continue };
            last.insert(g, j);
            let on = subject.entry(g).or_insert(None);
            match c.as_slice() {
                [] => {}
                [(s, _)] if on.is_none_or(|t| t == *s) => *on = Some(*s),
                _ => *on = Some(usize::MAX),
            }
        }
        let local = |g: &usize| subject[g] != Some(usize::MAX);
        let mut kept_cols: Vec<Pieces> = Vec::with_capacity(cols.len());
        let mut kept_groups: Vec<Option<usize>> = Vec::with_capacity(cols.len());
        for (j, (c, g)) in cols.iter().zip(groups).enumerate() {
            match g {
                Some(g) if local(g) => {
                    if j != last[g] {
                        kept_cols.push(pieces_minus(c, &cols[last[g]]));
                        kept_groups.push(None);
                    }
                }
                _ => {
                    kept_cols.push(c.clone());
                    kept_groups.push(*g);
                }
            }
        }
        Self::grouped_raw(&kept_cols, &kept_groups, lens, offsets)
    }

    /// [`BlockBasis::grouped`] with every group taken through `T`.
    fn grouped_raw(
        cols: &[Pieces],
        groups: &[Option<usize>],
        lens: &[usize],
        offsets: &[usize],
    ) -> Self {
        let n_rows: usize = lens.iter().sum();
        let norm_of = |c: &Pieces| c.iter().map(|(_, v)| dot(v, v)).sum::<f64>().sqrt();
        let mut singles: Vec<Vec<usize>> = vec![Vec::new(); lens.len()];
        for (j, col) in cols.iter().enumerate().filter(|(_, c)| c.len() == 1) {
            singles[col[0].0].push(j);
        }
        let mut remnants: Vec<f64> = Vec::new();
        // Per column taken, in order: (column, subject or population-wide,
        // per-subject coefficients, population-wide coefficients, added norm).
        let mut taken_local: Vec<(usize, usize, Vec<f64>, Option<f64>)> = Vec::new();
        let mut local: Vec<Vec<Vec<f64>>> = Vec::with_capacity(lens.len());
        for (s, js) in singles.iter().enumerate() {
            let mut basis: Vec<Vec<f64>> = Vec::new();
            for &j in js {
                let c = &cols[j][0].1;
                let c_norm = dot(c, c).sqrt();
                if c_norm == 0.0 || !c_norm.is_finite() {
                    continue;
                }
                let mut v = c.clone();
                let coefs = project_out_coefs(&mut v, &basis);
                let n = dot(&v, &v).sqrt();
                remnants.push(n / c_norm);
                let added = (n > DROP_TOL * c_norm).then_some(n);
                if added.is_some() {
                    basis.push(v.into_iter().map(|x| x / n).collect());
                }
                taken_local.push((j, s, coefs, added));
            }
            local.push(basis);
        }
        let local_at: Vec<usize> = local
            .iter()
            .scan(0, |at, b| {
                let o = *at;
                *at += b.len();
                Some(o)
            })
            .collect();
        let dense_at: usize = local.iter().map(|b| b.len()).sum();
        let mut taken: Vec<Taken> = Vec::new();
        for (j, s, coefs, added) in taken_local {
            // A column's coefficients run over the basis before it, so the
            // vector it adds is the next one.
            let k = local_at[s] + coefs.len();
            taken.push(Taken {
                added: added.map(|n| (k, n)),
                coefs: vec![(local_at[s], coefs)],
                norm: norm_of(&cols[j]),
                group: groups[j],
            });
        }
        // Each multi-subject column, projected against the local bases and the
        // dense part so far, adds a direction only when what is left is not
        // rounding of the *original* column — the test `orthonormal_basis`
        // applies on a dense column. Judged against the remnant's own norm, a
        // column inside the local span would turn ~1e-17 of rounding into a
        // unit vector and shrink every residual (#1961 review r2, R2-1).
        let mut dense: Vec<Vec<f64>> = Vec::new();
        for (j, col) in cols.iter().enumerate().filter(|(_, c)| c.len() > 1) {
            let c_norm = norm_of(col);
            if c_norm == 0.0 || !c_norm.is_finite() {
                continue;
            }
            let mut v = vec![0.0; n_rows];
            let mut coefs: Vec<(usize, Vec<f64>)> = Vec::new();
            for (s, piece) in col {
                let mut piece = piece.clone();
                coefs.push((local_at[*s], project_out_coefs(&mut piece, &local[*s])));
                v[offsets[*s]..offsets[*s] + piece.len()].copy_from_slice(&piece);
            }
            coefs.push((dense_at, project_out_coefs(&mut v, &dense)));
            let n = dot(&v, &v).sqrt();
            remnants.push(n / c_norm);
            let added = (n > DROP_TOL * c_norm).then_some((dense_at + dense.len(), n));
            if added.is_some() {
                dense.push(v.into_iter().map(|x| x / n).collect());
            }
            taken.push(Taken {
                coefs,
                added,
                norm: c_norm,
                group: groups[j],
            });
        }
        let mut basis = BlockBasis {
            local,
            dense,
            offsets: offsets.to_vec(),
            n_rows,
            remnants,
            local_at,
            removed: Vec::new(),
            coded_shift: 0,
            #[cfg(test)]
            joint: 0,
        };
        basis.remove_groups(cols, groups, &taken, dense_at);
        #[cfg(test)]
        DENSE_COUNTS.with(|d| d.borrow_mut().push(basis.dense.len()));
        basis
    }

    /// `T` (see [`BlockBasis`]) and the grouped columns' shift of
    /// [`BlockBasis::aliased`], from the columns as Gram–Schmidt `taken` them.
    fn remove_groups(
        &mut self,
        cols: &[Pieces],
        groups: &[Option<usize>],
        taken: &[Taken],
        dense_at: usize,
    ) {
        let dim = dense_at + self.dense.len();
        // Per group, in order of first appearance: its columns.
        let mut members: Vec<(usize, Vec<usize>)> = Vec::new();
        for (j, g) in groups.iter().enumerate() {
            let Some(g) = *g else { continue };
            match members.iter_mut().find(|(h, _)| *h == g) {
                Some((_, js)) => js.push(j),
                None => members.push((g, vec![j])),
            }
        }
        if members.is_empty() {
            return;
        }
        let nonzero = |c: &Pieces| {
            let n = c.iter().map(|(_, v)| dot(v, v)).sum::<f64>();
            n != 0.0 && n.is_finite()
        };
        // The coded columns are each column's difference against its group's
        // last: count the nonzero ones against the nonzero raw ones.
        let mut shift: isize = 0;
        for (_, js) in &members {
            let last = &cols[*js.last().expect("a group has a column")];
            let coded = js[..js.len() - 1]
                .iter()
                .filter(|&&j| !pieces_equal(&cols[j], last))
                .count();
            let raw = js.iter().filter(|&&j| nonzero(&cols[j])).count();
            shift += coded as isize - raw as isize;
        }
        // A group with a zero column leaves its constraint nothing to act on:
        // that column's equation forces `λ_g = 0`, and the group removes no
        // direction.
        let live: Vec<&(usize, Vec<usize>)> = members
            .iter()
            .filter(|(_, js)| js.iter().all(|&j| nonzero(&cols[j])))
            .collect();
        #[cfg(test)]
        {
            self.joint = live.len();
        }
        // Per live group, `t` for `λ = e_g` in coordinates, and the misfit of
        // each dropped column's equation over its norm.
        let mut ts: Vec<Vec<f64>> = Vec::with_capacity(live.len());
        let mut misfits: Vec<Vec<f64>> = Vec::with_capacity(live.len());
        for (g, _) in &live {
            let mut t = vec![0.0; dim];
            let mut misfit = Vec::new();
            for col in taken {
                let rhs = if col.group == Some(*g) { 1.0 } else { 0.0 };
                let acc: f64 = col
                    .coefs
                    .iter()
                    .map(|(at, c)| dot(c, &t[*at..*at + c.len()]))
                    .sum();
                match col.added {
                    Some((k, n)) => t[k] = (rhs - acc) / n,
                    None => misfit.push((acc - rhs) / col.norm),
                }
            }
            ts.push(t);
            misfits.push(misfit);
        }
        // The `λ` whose misfit vanishes against its `t`. A group's `t` alone
        // can be zero — every column of it dropped — and still complete
        // another's: two levels of one block summing to a level of another
        // tie their `λ` (2 : 1), and neither unit `λ` is consistent. So the
        // stacked `(t, misfit)` are orthonormalised together, which no `λ`
        // annuls (both vanish only when every equation's right-hand side
        // does), and the misfit part's small singular directions are the `λ`
        // sought: a singular value there is `‖misfit‖ / ‖(t, misfit)‖`.
        let mut units: Vec<Vec<f64>> = Vec::new();
        for (t, m) in ts.into_iter().zip(misfits) {
            let mut w = t;
            w.extend(m);
            let w_norm = dot(&w, &w).sqrt();
            if w_norm == 0.0 || !w_norm.is_finite() {
                continue;
            }
            project_out(&mut w, &units);
            let n = dot(&w, &w).sqrt();
            if n > DROP_TOL * w_norm {
                units.push(w.into_iter().map(|x| x / n).collect());
            }
        }
        let n_drop = units.first().map_or(0, |u| u.len() - dim);
        if !units.is_empty() {
            let r = units.len();
            // A zero row keeps the matrix at least square, so the SVD returns
            // a right singular vector for every unit.
            let rows = n_drop.max(r);
            let misfit = nalgebra::DMatrix::from_fn(rows, r, |i, a| {
                units[a].get(dim + i).copied().unwrap_or(0.0)
            });
            let svd = misfit.svd(false, true);
            let v_t = svd.v_t.expect("requested");
            for (i, sv) in svd.singular_values.iter().enumerate() {
                if sv.is_nan() || *sv >= CONSISTENT_TOL {
                    continue;
                }
                let mut t = vec![0.0; dim];
                for (a, u) in units.iter().enumerate() {
                    let w = v_t[(i, a)];
                    t.iter_mut().zip(&u[..dim]).for_each(|(x, y)| *x += w * y);
                }
                let n = dot(&t, &t).sqrt();
                self.removed.push(t.into_iter().map(|x| x / n).collect());
            }
        }
        // Each live group's last column adds the direction its difference
        // columns lack, unless `A` and the groups before it already hold it:
        // that is the coded columns' rank against the raw ones.
        let mut held: Vec<Vec<f64>> = Vec::new();
        for (_, js) in &live {
            let d = &cols[*js.last().expect("a group has a column")];
            let d_norm = d.iter().map(|(_, v)| dot(v, v)).sum::<f64>().sqrt();
            let (coords, _) = self.project(d);
            let mut s: Vec<f64> = self
                .removed
                .iter()
                .map(|t| dot_coords(t, &coords))
                .collect();
            project_out(&mut s, &held);
            let n = dot(&s, &s).sqrt();
            if n >= SPAN_TOL * d_norm {
                shift += 1;
            }
            if n > DROP_TOL * d_norm {
                held.push(s.into_iter().map(|x| x / n).collect());
            }
        }
        self.coded_shift = shift;
    }

    /// `z`'s coordinates on the basis, as `(first coordinate, values)`
    /// segments, and its remnant outside the raw columns' span.
    fn project(&self, z: &Pieces) -> (Vec<(usize, Vec<f64>)>, Vec<f64>) {
        let mut coords = Vec::with_capacity(z.len() + 1);
        let mut r = vec![0.0; self.n_rows];
        for (s, piece) in z {
            let mut piece = piece.clone();
            coords.push((
                self.local_at[*s],
                project_out_coefs(&mut piece, &self.local[*s]),
            ));
            r[self.offsets[*s]..self.offsets[*s] + piece.len()].copy_from_slice(&piece);
        }
        let dense_at =
            self.local_at.last().copied().unwrap_or(0) + self.local.last().map_or(0, |b| b.len());
        coords.push((dense_at, project_out_coefs(&mut r, &self.dense)));
        (coords, r)
    }

    /// How many nonzero columns kept less than [`SPAN_TOL`] of their norm
    /// against the columns orthonormalised before them (#1977): the columns'
    /// rank deficiency at that tolerance, whatever order they were taken in.
    /// A sum-to-zero group counts as its coded columns, the differences
    /// against its last (#1962): those number one fewer than its raw columns,
    /// and its last column adds back what they lack unless the rest already
    /// holds it.
    fn aliased(&self) -> usize {
        let raw = self.remnants.iter().filter(|r| **r < SPAN_TOL).count() as isize;
        (raw + self.coded_shift).max(0) as usize
    }

    /// `z − P z`, `P` the projection on the coded span `A`, in coordinates
    /// where `A`'s complement is the raw columns' complement followed by `T`:
    /// the vector whose norm [`BlockBasis::residual`] takes. Linear in `z`, so
    /// a further column is projected out of it as out of `z` itself (#1878
    /// review).
    fn complement(&self, z: &Pieces) -> Vec<f64> {
        let (coords, mut r) = self.project(z);
        r.extend(self.removed.iter().map(|t| dot_coords(t, &coords)));
        r
    }

    /// `‖z − P z‖ / ‖z‖`, `P` the projection on the coded span `A`; `None`
    /// when `z` is zero. A non-finite value comes back as `NaN`, which no
    /// tolerance accepts.
    fn residual(&self, z: &Pieces) -> Option<f64> {
        let z_norm = z.iter().map(|(_, v)| dot(v, v)).sum::<f64>().sqrt();
        if z_norm == 0.0 {
            return None;
        }
        let (coords, r) = self.project(z);
        let removed: f64 = self
            .removed
            .iter()
            .map(|t| dot_coords(t, &coords).powi(2))
            .sum();
        Some((dot(&r, &r) + removed).sqrt() / z_norm)
    }
}

/// The coded columns of `cols` under `groups`, each grouped column but its
/// group's last taken against that last one: the columns
/// [`BlockBasis::grouped`] spans without building them.
#[cfg(test)]
fn coded_differences(cols: &[Pieces], groups: &[Option<usize>]) -> Vec<Pieces> {
    let mut out = Vec::new();
    for (j, (c, g)) in cols.iter().zip(groups).enumerate() {
        match g {
            None => out.push(c.clone()),
            Some(g) => {
                let last = (0..cols.len())
                    .rev()
                    .find(|&k| groups[k] == Some(*g))
                    .unwrap();
                if j != last {
                    out.push(pieces_minus(c, &cols[last]));
                }
            }
        }
    }
    out
}

/// `t · c`, `t` dense in the coordinates of a [`BlockBasis`] and `c` as
/// segments.
fn dot_coords(t: &[f64], c: &[(usize, Vec<f64>)]) -> f64 {
    c.iter().map(|(at, v)| dot(v, &t[*at..*at + v.len()])).sum()
}

/// Whether a contrast's coded block, with the rest of the model, reproduces an
/// absorbed random effect entirely (#1840).
///
/// The structural walk and the span certificate say that a random effect's
/// effect on `y` lies, unit by unit, in the span of the block's levels: under
/// `contrast = none` the levels are then that random effect. A contrast that
/// codes the levels with one direction fewer per group — the common mean under
/// sum-to-zero, the reference level under `ref` — leaves the random effect that
/// direction, unless something else in the model carries it: an intercept, or
/// any θ whose effect is some combination of the levels'. That is the joint
/// oracle's test: the random effect is absorbed under `c` when every one of its
/// unit columns lies in the span of the other free θ, the block coded by `c`,
/// and the random effects at a coarser-or-equal unit (the other η for an η;
/// every η and the other κ for a κ). Nothing in the model is constant across
/// all records when the only θ that would be is read through a factor varying
/// within each subject (`TVE0 * (1 + 0.1 * OCC) + PLACEBO + ETA_E0`): no θ
/// carries the levels' mean, and sum-to-zero identifies the random effect.
///
/// Measured on the model bound under `contrast = none`, by central differences
/// of the predictions over the whole population, at both [`probe_points`] with
/// the random effects spread by [`probe_res`]. The verdict is
/// [`Carried::Identified`] only when, at **both** points, every column is
/// finite and some unit's residual is at least [`SPAN_TOL`]; anything else —
/// a non-finite value, a point where no unit measures anything, a direction
/// carried at either point — is [`Carried::Held`], and the block stays refused.
/// That is the opposite of [`spanned`]'s convention, and on purpose: here the
/// measurement removes a refusal, so an inconclusive point must fall to it. A θ
/// initialised at 0 that switches the intercept off at the initial estimates
/// is seen at the moved point.
///
/// Measured over every call the `api::levels` tests make, macOS arm64 debug:
/// a unit set the verdict holds on reaches at most 2.1e-10 (its largest unit
/// residual; `a_common_factor_does_not_count_in_a_parameter_funnel` under
/// `ref`), an identified one at least 1.4e-2 (the kappa twin of
/// `a_coarser_random_effect_can_carry_the_mean` without `ETA_B`). [`SPAN_TOL`]
/// sits 3.7 decades above the one and 4.2 below the other.
///
/// The coded block must also keep its rank against the other θ (#1977): a
/// θ that is itself a combination of the levels carries exactly the direction
/// the contrast under which it lies in the coded span removes — `TVB * D2`,
/// with no weight on the reference level, under `ref` — so the certificate
/// would otherwise name as the rescue the one contrast that aliases the model.
/// Each coded column must keep at least [`SPAN_TOL`] of its norm against the
/// other θ and the coded columns before it, counted on the basis the residual
/// is measured in, beyond any deficiency the other θ have among themselves.
/// Measured over every guard the `api::levels` tests reach (2334 with the
/// two-block tests of #1963, macOS arm64 debug): an aliased column keeps at most 1.3e-9 of its norm, an independent
/// one at least 3.4e-4, so [`SPAN_TOL`] sits 2.9 decades above the one and 2.5
/// below the other. Under sum-to-zero the count is taken on the raw levels and
/// shifted to the coded ones ([`BlockBasis::aliased`], #1962); over the 2464
/// bases the tests build it gave the rank verdict the difference columns gave
/// in every one, and the count itself in all but one, where a difference of
/// two 2e-8-norm levels kept 1.3e-9 of its norm at its own rounding floor and
/// counted a third deficiency in a block whose exact deficiency is two.
///
/// Only branches that refuse without it ask for it, so a model that binds
/// without the certificate never pays for it. On those branches it costs the
/// population-wide columns of a [`BlockBasis`]: the other θ, and the levels
/// read by more than one subject. Under sum-to-zero the coded columns — each
/// level's difference against its group's last — cross subjects whenever the
/// levels do, one population-wide column per level and a cost cubic in the
/// subjects (#1961 review, F1: 19 s at 1000 subjects on a 5-η `two_cpt_oral`
/// intercept model, release, where the binder before #1840 refused in
/// ~20 ms). So the basis takes the raw levels, tagged with their group, and
/// keeps apart the at most one direction per group the contrast removes
/// (#1962, [`BlockBasis::grouped`]): on the same fixture 48 ms at 200
/// subjects, 56 ms at 400 and 0.29 s at 1000 (release, macOS arm64), the
/// refusal text unchanged. The coarser random effects'
/// columns, one per unit, are built only when the point is not already held
/// without them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Carried {
    /// Every unit column of the random effect lies in the span: the contrast
    /// absorbs it.
    Held,
    /// Some unit column keeps a direction outside the span at both points.
    Identified,
}

/// Another level block as the certificate counts it (#1963): resolved to
/// `contrast` over `groups`, so among the other θ it contributes the
/// directions that contrast leaves, not every level.
struct Coded<'a> {
    decl: &'a LevelBlockDecl,
    levels: &'a [Level],
    contrast: LevelContrast,
    groups: Vec<usize>,
}

/// The certificate behind [`Carried`], for random effect `name` (`kappa`
/// when it is one) of `decl` under `contrast`.
///
/// Another level block's θ are other θ too, and count as that block will be
/// coded (#1963): each block in `others` — the blocks that absorb nothing,
/// already resolved — by its contrast, from its raw level columns on the
/// `none` model through its coding, the chain rule `∂y/∂β = ∂y/∂θ_level · C`
/// this block's own coding applies. That is exact for a block that enters
/// linearly; for one read through a nonlinear funnel it is the Jacobian at
/// the probe point, the same approximation the own-block coding makes. A
/// block not in `others` — one that absorbs a random effect itself — counts
/// raw, every level, as under `none`: its own coding is not known yet, and
/// raw can only refuse more.
#[allow(clippy::too_many_arguments)]
fn carried(
    model: &CompiledModel,
    decl: &LevelBlockDecl,
    levels: &[Level],
    population: &Population,
    name: &str,
    kappa: bool,
    contrast: LevelContrast,
    others: &[Coded],
) -> Carried {
    let index = match kappa {
        false => model.eta_names.iter().position(|n| n == name),
        true => model.kappa_names.iter().position(|n| n == name),
    };
    let Some(index) = index else {
        return Carried::Held;
    };
    if contrast == LevelContrast::Unconstrained {
        return Carried::Held;
    }
    let level_thetas = free_level_thetas(model, decl, levels);
    // The θ that enter coded: this block's, and every block's in `others`.
    let other_levels: Vec<Vec<Option<usize>>> = others
        .iter()
        .map(|o| free_level_thetas(model, o.decl, o.levels))
        .collect();
    let coded: Vec<usize> = level_thetas
        .iter()
        .chain(other_levels.iter().flatten())
        .flatten()
        .copied()
        .collect();
    let p = &model.default_params;
    let other_thetas: Vec<usize> = (0..model.n_theta)
        .filter(|k| !p.theta_fixed.get(*k).copied().unwrap_or(false) && !coded.contains(k))
        .collect();
    let groups = assign_groups(decl, levels, contrast);
    let lens: Vec<usize> = population
        .subjects
        .iter()
        .map(|s| s.obs_times.len())
        .collect();
    let offsets: Vec<usize> = lens
        .iter()
        .scan(0, |at, n| {
            let o = *at;
            *at += n;
            Some(o)
        })
        .collect();

    for (theta, re) in probe_points(model) {
        let layouts: Vec<_> = population
            .subjects
            .iter()
            .map(|s| crate::stats::stacked_layout::StackedLayout::of(model, s))
            .collect();
        let cells: Vec<Option<NestedLayout>> = population
            .subjects
            .iter()
            .map(|s| NestedLayout::of(model, s))
            .collect();
        let res: Vec<(Vec<f64>, Vec<Vec<f64>>)> = layouts
            .iter()
            .zip(&cells)
            .map(|(l, c)| subject_probe_res(re, model, c.as_ref(), l.n_groups()))
            .collect();
        // One population-wide column: per subject, the central difference of
        // its predictions under `perturb`, which moves θ or that subject's
        // random effects by ±h, kept where it is not exactly zero; `None` when
        // anything is non-finite.
        let diff_col = |subjects: &[usize],
                        h: f64,
                        perturb: &dyn Fn(f64, &mut Vec<f64>, &mut Vec<f64>, &mut Vec<Vec<f64>>)|
         -> Option<Pieces> {
            let mut col = Pieces::new();
            for &s in subjects {
                let subject = &population.subjects[s];
                let (eta, kappa) = &res[s];
                let side = |sign: f64| {
                    let (mut th, mut et, mut ka) = (theta.clone(), eta.clone(), kappa.clone());
                    perturb(sign * h, &mut th, &mut et, &mut ka);
                    predictions_at(model, subject, &th, &et, &ka)
                };
                let d = record_diffs(&side(1.0), &side(-1.0), h)?;
                if d.len() != lens[s] || d.iter().any(|v| !v.is_finite()) {
                    return None;
                }
                if d.iter().any(|v| *v != 0.0) {
                    col.push((s, d));
                }
            }
            Some(col)
        };
        let all: Vec<usize> = (0..population.subjects.len()).collect();
        let theta_col = |k: usize| {
            let h = SPAN_STEP * theta[k].abs().max(1.0);
            diff_col(&all, h, &|d, th, _, _| th[k] += d)
        };

        // A block's level columns (a `FIX` level moves nothing), coded by its
        // contrast over its groups: under `none` every level, under `ref` the
        // first level of each group dropped. Under sum-to-zero every level,
        // tagged with its group: the coded span, each level but its group's
        // last taken against that last one, is what [`BlockBasis::grouped`]
        // spans on those tags (#1962). `None` when anything is non-finite.
        let coded_cols = |decl: &LevelBlockDecl,
                          thetas: &[Option<usize>],
                          groups: &[usize],
                          contrast: LevelContrast|
         -> Option<Vec<(Pieces, Option<usize>)>> {
            let column = level_index_column(decl.name());
            // Per level, the subjects that read it.
            let mut readers: Vec<Vec<usize>> = vec![Vec::new(); thetas.len()];
            for &s in &all {
                let mut read = indices_read(&population.subjects[s], &column);
                read.sort_by(f64::total_cmp);
                read.dedup();
                for index in read {
                    let i = index as usize;
                    if index == i as f64 && (1..=thetas.len()).contains(&i) {
                        readers[i - 1].push(s);
                    }
                }
            }
            let mut level_cols: Vec<Pieces> = Vec::with_capacity(thetas.len());
            for (i, k) in thetas.iter().enumerate() {
                let col = match *k {
                    Some(k) => {
                        let h = SPAN_STEP * theta[k].abs().max(1.0);
                        diff_col(&readers[i], h, &|d, th, _, _| th[k] += d)?
                    }
                    None => Pieces::new(),
                };
                level_cols.push(col);
            }
            let first = |i: usize| i == 0 || groups[i - 1] != groups[i];
            let mut out = Vec::new();
            for (i, col) in level_cols.into_iter().enumerate() {
                match contrast {
                    LevelContrast::Unconstrained => out.push((col, None)),
                    LevelContrast::Ref if !first(i) => out.push((col, None)),
                    LevelContrast::SumToZero | LevelContrast::SumToZeroWithin => {
                        out.push((col, Some(groups[i])))
                    }
                    _ => {}
                }
            }
            Some(out)
        };

        // The columns, and per column its sum-to-zero group, numbered across
        // the blocks.
        let mut cols: Vec<Pieces> = Vec::new();
        let mut tags: Vec<Option<usize>> = Vec::new();
        let push_coded = |coded: Vec<(Pieces, Option<usize>)>,
                          cols: &mut Vec<Pieces>,
                          tags: &mut Vec<Option<usize>>| {
            let base = tags.iter().flatten().map(|g| g + 1).max().unwrap_or(0);
            for (c, g) in coded {
                cols.push(c);
                tags.push(g.map(|g| base + g));
            }
        };
        for &k in &other_thetas {
            let Some(c) = theta_col(k) else {
                return Carried::Held;
            };
            cols.push(c);
            tags.push(None);
        }
        for (o, thetas) in others.iter().zip(&other_levels) {
            let Some(c) = coded_cols(o.decl, thetas, &o.groups, o.contrast) else {
                return Carried::Held;
            };
            push_coded(c, &mut cols, &mut tags);
        }
        let n_other = cols.len();
        let Some(own_coded) = coded_cols(decl, &level_thetas, &groups, contrast) else {
            return Carried::Held;
        };
        push_coded(own_coded, &mut cols, &mut tags);

        // Per unit of each random effect: its column. A unit is a subject for
        // an η, an occasion group of a subject for a κ, and under nested levels
        // every cell of one group of the κ's level (#1082).
        let unit_cols = |r: usize, is_kappa: bool| -> Option<Vec<(usize, Vec<usize>, Pieces)>> {
            let mut out = Vec::new();
            for s in 0..population.subjects.len() {
                let units = match is_kappa {
                    true => kappa_units(model, cells[s].as_ref(), layouts[s].n_groups(), r)?,
                    false => vec![Vec::new()],
                };
                for groups in units {
                    let col = diff_col(&[s], SPAN_STEP, &|d, _, et, ka| match is_kappa {
                        false => et[r] += d,
                        true => {
                            for &g in &groups {
                                if let Some(row) = ka.get_mut(g) {
                                    row[r] += d;
                                }
                            }
                        }
                    })?;
                    out.push((s, groups, col));
                }
            }
            Some(out)
        };
        let Some(own) = unit_cols(index, kappa) else {
            return Carried::Held;
        };
        // Whether some unit keeps a direction outside the span of `cols`. A
        // unit whose column is zero measures nothing; anything non-finite says
        // nothing either, and falls to the refusal.
        let outside = |basis: &BlockBasis| -> Option<bool> {
            let mut outside = false;
            for (.., z) in &own {
                match basis.residual(z) {
                    None => {}
                    Some(r) if !r.is_finite() => return None,
                    Some(r) => outside |= r >= SPAN_TOL,
                }
            }
            Some(outside)
        };
        // #1977: the coded block must add a direction of its own for every
        // column it has. A coded column inside the span of the other θ (and of
        // the coded columns before it) is a θ the data cannot tell from them:
        // `TVB * D2` is a combination of the levels with no weight on the
        // reference, so under `ref` it is one of the coded columns' sums, and
        // the model is aliased whatever the random effect keeps. The rank falls
        // out of the basis the residual needs anyway; the other θ's own
        // deficiency, if any, is not the block's and is subtracted.
        //
        // Without the coarser random effects first: adding columns only widens
        // the span, so a unit set held without them is held with them, and the
        // usual refusal — an intercept carrying the direction — never builds
        // their one column per unit (#1961 review, F1). It is held whatever the
        // rank says, so it never builds the other θ's basis either; nor does a
        // full basis with no aliased column, whose other θ are independent too.
        let basis = BlockBasis::grouped(&cols, &tags, &lens, &offsets);
        if outside(&basis) != Some(true) {
            return Carried::Held;
        }
        if basis.aliased() > 0
            && basis.aliased()
                > BlockBasis::grouped(&cols[..n_other], &tags[..n_other], &lens, &offsets).aliased()
        {
            return Carried::Held;
        }
        // The random effects at a coarser-or-equal unit: the other η for an η,
        // every η and the other κ for a κ — under nested levels the other κ at
        // its own level or an outer one (#1082).
        let level_of = |q: usize| model.nested_levels().and_then(|n| n.kappa_level.get(q));
        let coarser: Vec<(usize, bool)> = (0..model.n_eta)
            .filter(|&e| kappa || e != index)
            .map(|e| (e, false))
            .chain(
                (0..model.n_kappa)
                    .filter(|&q| kappa && q != index && level_of(q) <= level_of(index))
                    .map(|q| (q, true)),
            )
            .collect();
        if coarser.is_empty() {
            continue;
        }
        let mut per_subject: Vec<Vec<Pieces>> = vec![Vec::new(); population.subjects.len()];
        for (r, is_kappa) in coarser {
            let Some(c) = unit_cols(r, is_kappa) else {
                return Carried::Held;
            };
            tags.extend(std::iter::repeat_n(None, c.len()));
            for (s, _, col) in c {
                per_subject[s].push(col.clone());
                cols.push(col);
            }
        }
        let with_coarser = BlockBasis::grouped(&cols, &tags, &lens, &offsets);
        if outside(&with_coarser) != Some(true) {
            return Carried::Held;
        }
        // A kappa jointly with the coarser random effects (#1878 review r1,
        // r2): an η and every occasion's kappa shifted against it leave the
        // levels a shift, though no one occasion's column lies in the span of
        // the η's column. So per subject, the kappa's unit columns `K` and the
        // coarser columns `C` outside the coded span: the dimension of
        // `span K ∩ span C` there — a combination of the kappa's units the
        // coarser random effects match, with one coefficient each across the
        // subject — against the same dimension outside the other θ alone, the
        // joint oracle's baseline. The block holds the kappa under the
        // contrast when that dimension grows in every subject it moves. An
        // η's partners share its unit, which the full columns above already
        // judge.
        if !kappa {
            continue;
        }
        let without = BlockBasis::grouped(&cols[..n_other], &tags[..n_other], &lens, &offsets);
        let mut held = false;
        for (s, subject_cols) in per_subject.iter().enumerate() {
            let k: Vec<&Pieces> = own
                .iter()
                .filter(|(t, ..)| *t == s)
                .map(|(.., z)| z)
                .collect();
            if k.is_empty() {
                continue;
            }
            // `None` when anything is non-finite: inconclusive, which here
            // falls to the refusal.
            let shared = |b: &BlockBasis| -> Option<usize> {
                let scaled = |c: &&Pieces| -> Option<Option<Vec<f64>>> {
                    let n = c.iter().map(|(_, v)| dot(v, v)).sum::<f64>().sqrt();
                    if n == 0.0 {
                        return Some(None);
                    }
                    let r: Vec<f64> = b.complement(c).into_iter().map(|x| x / n).collect();
                    r.iter().all(|x| x.is_finite()).then_some(Some(r))
                };
                let kc: Vec<Vec<f64>> = k
                    .iter()
                    .map(scaled)
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                let cc: Vec<Vec<f64>> = subject_cols
                    .iter()
                    .map(|c| scaled(&c))
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                let both: Vec<Vec<f64>> = cc.iter().chain(&kc).cloned().collect();
                Some((rank(&kc) + rank(&cc)).saturating_sub(rank(&both)))
            };
            let (Some(with), Some(alone)) = (shared(&basis), shared(&without)) else {
                return Carried::Held;
            };
            if with <= alone {
                return Carried::Identified;
            }
            held = true;
        }
        if held {
            return Carried::Held;
        }
    }
    Carried::Identified
}

/// [`carried`] for one block's absorbed random effect, each contrast measured
/// at most once (#1840). `None` model: the likelihood reads a channel the
/// certificate does not compare, so nothing is measured and every contrast is
/// [`Carried::Held`] — the refusal stands.
struct Certificate<'a> {
    model: Option<&'a CompiledModel>,
    decl: &'a LevelBlockDecl,
    levels: &'a [Level],
    population: &'a Population,
    name: &'a str,
    kappa: bool,
    /// The blocks that absorb nothing, as they resolved (#1963).
    others: &'a [Coded<'a>],
    seen: std::cell::RefCell<Vec<(LevelContrast, Carried)>>,
}

impl Certificate<'_> {
    fn under(&self, contrast: LevelContrast) -> Carried {
        if let Some(&(_, c)) = self.seen.borrow().iter().find(|(k, _)| *k == contrast) {
            return c;
        }
        let verdict = match self.model {
            None => Carried::Held,
            Some(m) => carried(
                m,
                self.decl,
                self.levels,
                self.population,
                self.name,
                self.kappa,
                contrast,
                self.others,
            ),
        };
        self.seen.borrow_mut().push((contrast, verdict));
        verdict
    }

    fn identified(&self, contrast: LevelContrast) -> bool {
        self.under(contrast) == Carried::Identified
    }
}

/// What the binder measures on one block, on the model with every block under
/// `contrast = none`: its dead levels (rule 1, #1679), and per random effect,
/// parallel to `eta_couplings`, whether the span certificate measured it
/// absorbed (#1834). `model` is that model, kept for [`carried`], which only a
/// refusing branch asks for; `None` when the checks do not read every channel.
struct Measured {
    dead: DeadLevels,
    spanned: Vec<Option<Absorption>>,
    model: Option<std::rc::Rc<CompiledModel>>,
    /// Whether a dose of the model can arrive after its record
    /// ([`lagged_arrivals`]), measured or not: [`replicates`] reads it.
    lagged: bool,
    /// The model's nested levels (#1082), measured or not: a kappa's unit is a
    /// group of its own level ([`Unit::Level`]).
    nested: Option<NestedLevels>,
}

/// Whether a dose of `model` can change the state at a time the data does not
/// hold (#2131 review): a compartment lag — the bare lag slot or an indexed
/// `ALAGn` ([`CompiledModel::has_lagtime`]), or an `ALAGn` entry of the
/// `dose_attr_map` the engines resolve every dose's lag through — or a
/// per-route lag on an absorption forcing. A lag the model assigns but the
/// engines never apply moves no arrival, so it does not count;
/// `the_lag_predicate_sees_every_lag_the_engine_applies` pins the two
/// together over every spelling.
fn lagged_arrivals(model: &CompiledModel) -> bool {
    model.has_lagtime()
        || model
            .dose_attr_map
            .has_indexed_attr(crate::types::DoseAttr::Lag)
        || model.has_route_absorption_lag()
}

/// Run [`dead_levels`] and [`spanned`] on every block, on the model re-parsed
/// with every block under `contrast = none`, so the measurement is each
/// level's own θ whatever contrast the block will take. A model whose
/// likelihood reads a channel the check does not compare is not measured:
/// every block comes back with no dead level and no measured absorption.
fn measure_levels(
    base: &ParseBindings,
    model_text: &str,
    decls: &[LevelBlockDecl],
    discovered: &[Vec<Level>],
    population: &Population,
) -> Result<Vec<Measured>, String> {
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
    let lagged = lagged_arrivals(&model);
    let nested = model.nested_levels().cloned();
    if !dead_check_reads_every_channel(&model) {
        return Ok(decls
            .iter()
            .map(|decl| Measured {
                dead: DeadLevels::default(),
                spanned: vec![None; decl.eta_couplings.len()],
                model: None,
                lagged,
                nested: nested.clone(),
            })
            .collect());
    }
    let model = std::rc::Rc::new(model);
    Ok(decls
        .iter()
        .zip(discovered)
        .map(|(decl, levels)| {
            let silent: Vec<(usize, Unit)> = placements(decl, population, lagged, nested.as_ref())
                .into_iter()
                .enumerate()
                .filter_map(|(k, placement)| match placement {
                    Placement::Silent(unit) => Some((k, unit)),
                    _ => None,
                })
                .collect();
            Measured {
                dead: dead_levels(&model, decl, levels, population),
                spanned: spanned(&model, decl, levels, population, &silent),
                model: Some(model.clone()),
                lagged,
                nested: nested.clone(),
            }
        })
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
            // The message names `TIME = 0` as written (#1959).
            !is_listed || subject.readout_time(j) == 0.0
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
/// has no within-group rescue: each level *is* one unit's, so `none` is refused,
/// and so is every other contrast whose removed direction — the levels' mean,
/// the reference level — another θ of the model carries ([`carried`], #1840);
/// the nested block's explicit global contrasts follow the same certificate.
/// Two absorbed random effects have no rescue either: within leaves each unit's
/// mean to one of them, and the levels reproduce the other — unless the
/// within-group contrast parts them ([`within_parts_them`]): each within-group
/// lies inside one unit of every one of them, and no two group the records
/// alike. Within then leaves each its own groups' means, as a nested model
/// tells a study kappa from an arm kappa (#2034), and each is resolved as if it
/// were the only one, the first refusal winning.
///
/// `others` is `None` for a block that absorbs nothing, which
/// [`bind_levels_on_data`] resolves first and which never reads the
/// certificate; for an absorbing block, the blocks resolved that way (#1963).
fn resolve_contrast(
    decl: &LevelBlockDecl,
    levels: &[Level],
    measured: &Measured,
    population: &Population,
    others: Option<&[Coded]>,
) -> Result<(LevelContrast, Vec<usize>), String> {
    let block = format!("theta {}[{}]", decl.name(), decl.columns().join(", "));
    let absorbed = absorbed(decl, population, measured);
    if absorbed.len() < 2 {
        let one = absorbed.into_iter().next();
        return resolve_absorbing(decl, levels, measured, population, others, one);
    }
    // Absorbed only in combination, on a nested block (#1878 review r2): the
    // partition rule below is for random effects each absorbed alone. A
    // combination's within verdict is already joint — its `Span` measured
    // against its partners — and each contrast is judged on it by `carried`,
    // which takes the partners among the random effects at a coarser-or-equal
    // unit; so each is resolved on its own, as when within parts them.
    let combined = decl.columns().len() >= 2 && absorbed.iter().all(Absorbed::combined);
    if !combined
        && !within_parts_them(
            decl,
            levels,
            &absorbed,
            population,
            measured.nested.as_ref(),
        )
    {
        return Err(jointly_absorbed_message(
            &block, decl, &absorbed, population,
        ));
    }
    let mut resolved = None;
    for one in absorbed {
        let r = resolve_absorbing(decl, levels, measured, population, others, Some(one))?;
        resolved.get_or_insert(r);
    }
    Ok(resolved.expect("two or more"))
}

/// Whether the within-group contrast of a nested block parts the random effects
/// `absorbed` (#2034): every within-group's records lie in one unit of each of
/// them, and no two of them group the records into the same units. The first
/// keeps each random effect constant across a group, so within leaves it the
/// group means; the second keeps two of them from competing for the same means,
/// which a kappa does with an η on one occasion per subject (K3′), or with the
/// kappa of an outer level whose groups each hold one group of its level.
fn within_parts_them(
    decl: &LevelBlockDecl,
    levels: &[Level],
    absorbed: &[Absorbed],
    population: &Population,
    nested: Option<&NestedLevels>,
) -> bool {
    if decl.columns().len() < 2 {
        return false;
    }
    let groups = assign_groups(decl, levels, LevelContrast::SumToZeroWithin);
    let column = level_index_column(decl.name());
    // Per record of every subject: its within-group, and its unit under each
    // random effect.
    let mut records: Vec<(usize, Vec<(usize, Vec<u64>)>)> = Vec::new();
    for (s, subject) in population.subjects.iter().enumerate() {
        for j in 0..subject.obs_times.len() {
            let Some(index) = column_value(subject, &column, j) else {
                return false;
            };
            let Some(&group) = (index as usize).checked_sub(1).and_then(|i| groups.get(i)) else {
                return false;
            };
            let units: Option<Vec<(usize, Vec<u64>)>> = absorbed
                .iter()
                .map(|a| unit_of(subject, a.unit_kind, j, nested).map(|u| (s, u)))
                .collect();
            let Some(units) = units else {
                return false;
            };
            records.push((group, units));
        }
    }
    // `a` determines `b`: every key of `a` meets a single key of `b`.
    fn determines<A: Eq + std::hash::Hash, B: Eq + Clone>(pairs: &[(A, B)]) -> bool {
        let mut seen: HashMap<&A, &B> = HashMap::new();
        pairs.iter().all(|(a, b)| *seen.entry(a).or_insert(b) == b)
    }
    let nests = (0..absorbed.len()).all(|r| {
        let pairs: Vec<(usize, (usize, Vec<u64>))> =
            records.iter().map(|(g, u)| (*g, u[r].clone())).collect();
        determines(&pairs)
    });
    let apart = (0..absorbed.len()).all(|a| {
        (0..a).all(|b| {
            let ab: Vec<_> = records
                .iter()
                .map(|(_, u)| (u[a].clone(), u[b].clone()))
                .collect();
            let ba: Vec<_> = records
                .iter()
                .map(|(_, u)| (u[b].clone(), u[a].clone()))
                .collect();
            !(determines(&ab) && determines(&ba))
        })
    });
    nests && apart
}

/// [`resolve_contrast`] for a block that absorbs at most the one random effect
/// `absorbed`.
fn resolve_absorbing(
    decl: &LevelBlockDecl,
    levels: &[Level],
    measured: &Measured,
    population: &Population,
    others: Option<&[Coded]>,
    absorbed: Option<Absorbed>,
) -> Result<(LevelContrast, Vec<usize>), String> {
    let nested = decl.columns().len() >= 2;
    let block = format!("theta {}[{}]", decl.name(), decl.columns().join(", "));
    let dead = &measured.dead;
    // A random effect the levels reproduce through within-group differences
    // has no rescue: `sum_to_zero_within` leaves exactly those directions free
    // (#1874 review, F1).
    if let Some(found) = absorbed
        .as_ref()
        .filter(|a| matches!(a.how, How::Measured(Span::Within, _)))
    {
        return Err(format!(
            "{block}: {}, so the levels reproduce it under every contrast, \
             `sum_to_zero_within` included: the model is not identified. Remove the block, or \
             drop the random effect.",
            found.clause(),
        ));
    }
    // Whether a contrast leaves the absorbed random effect a direction no other
    // θ carries (#1840): measured only when a branch below would refuse.
    // A block resolved in the first pass absorbs nothing, so it never gets
    // here: were it to, its verdict would read the other blocks before they
    // resolve, and the passes would depend on declaration order.
    debug_assert!(
        absorbed.is_none() || others.is_some(),
        "{block}: a block resolved in the first pass measured a certificate"
    );
    let certificate = absorbed.as_ref().map(|found| Certificate {
        model: measured.model.as_deref(),
        decl,
        levels,
        population,
        name: found.name(),
        kappa: found.kappa(),
        others: others.unwrap_or(&[]),
        seen: Default::default(),
    });
    let identified = |c: LevelContrast| certificate.as_ref().is_some_and(|k| k.identified(c));
    // A contrast the block can take: one the absorbed random effect allows,
    // that leaves a free θ, and that carries the dead levels. The certificate
    // does not enter: a contrast carries a dead level only by coding it through
    // the others (the reference, or its group's last level), which hands that
    // level's slack — exactly the direction the contrast removes — to the live
    // levels, so the absorbed random effect is reproduced entirely there anyway.
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
    // A contrast that binds a one-column block next to its absorbed random
    // effect: the random effect keeps a direction there, the contrast leaves a
    // free θ, and it carries the dead levels (#1840).
    let rescues = |c: LevelContrast| {
        let groups = assign_groups(decl, levels, c);
        identified(c)
            && free_count(&groups, c) > 0
            && dead_failure(&groups, c, dead, absorbed.as_ref()).is_none()
    };
    let resolved = match decl.contrast() {
        LevelContrast::Auto if nested && absorbed.is_some() => LevelContrast::SumToZeroWithin,
        // A one-column block (the arm above took every nested one) whose
        // absorbed random effect keeps the levels' mean only under `ref`:
        // something else carries the mean (#1840).
        LevelContrast::Auto
            if absorbed.is_some()
                && !identified(LevelContrast::SumToZero)
                && rescues(LevelContrast::Ref) =>
        {
            LevelContrast::Ref
        }
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
        if !nested && free > 0 && !identified(resolved) {
            let alt = [LevelContrast::SumToZero, LevelContrast::Ref]
                .into_iter()
                .find(|&c| rescues(c));
            return Err(one_column_message(&block, decl, found, resolved, alt));
        }
        if nested
            && matches!(
                resolved,
                LevelContrast::SumToZero | LevelContrast::Ref | LevelContrast::Unconstrained
            )
            && !identified(resolved)
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
                // The measured clause already names its unit.
                How::Measured(..) => found.clause(),
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
                    "{block}: `contrast = {}` lets the levels reproduce `{}` at every {}: \
                     {why} — the two are the same quantity, so the model is not identified.{fix}",
                    contrast_token(resolved),
                    found.name(),
                    found.unit,
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

/// The refusal for a one-column block whose levels each lie within one unit of
/// the absorbed random effect `found`, under `resolved`. With `alt` `None` no
/// contrast leaves the random effect a direction of its own — something else
/// in the model carries the levels' mean and their reference level alike (an
/// intercept) — and the refusal says so for every contrast. With `alt` a
/// contrast that does (#1840), the refusal names what `resolved` loses and
/// that contrast.
fn one_column_message(
    block: &str,
    decl: &LevelBlockDecl,
    found: &Absorbed,
    resolved: LevelContrast,
    alt: Option<LevelContrast>,
) -> String {
    let (unit, whose) = if found.kappa() {
        (
            format!("lies within a single {} of one subject", found.unit),
            format!("that {}'s", found.unit),
        )
    } else {
        (
            "belongs to a single subject".to_string(),
            "that subject's".to_string(),
        )
    };
    let column = &decl.columns()[0];
    let Some(alt) = alt else {
        return format!(
            "{block}: each `{column}` level {unit}, and {}, so a level and {whose} random effect \
             are the same quantity: the model is not identified under any contrast. Remove \
             the block, or drop the random effect.",
            found.clause(),
        );
    };
    let lost = |c: LevelContrast| match c {
        LevelContrast::Ref => "the reference level",
        _ => "the levels' common mean",
    };
    let how = match resolved {
        LevelContrast::Unconstrained => format!(
            "Under `contrast = none` every level has a θ of its own, so the levels reproduce \
             {whose} random effect entirely."
        ),
        c => format!(
            "Under `contrast = {}` the levels reproduce {whose} random effect up to {}, and \
             another θ of the model carries that direction, so together they reproduce it \
             entirely.",
            contrast_token(c),
            lost(c),
        ),
    };
    format!(
        "{block}: each `{column}` level {unit}, and {}. {how} The two are the same quantity, so \
         the model is not identified under this contrast. Use `contrast = {}`: no other θ \
         carries {}, so the random effect keeps that direction.",
        found.clause(),
        contrast_token(alt),
        lost(alt),
    )
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
    // Kappas at two nested levels (#1082) share no one unit name: a group.
    let unit = if absorbed.iter().all(|a| a.kappa()) {
        match absorbed.split_first() {
            Some((a, rest)) if rest.iter().all(|b| b.unit == a.unit) => a.unit.clone(),
            _ => "group".to_string(),
        }
    } else {
        "subject".to_string()
    };
    // Measured only in combination, none of them alone (#1878): the levels
    // absorb the combination, not each one, so dropping one is the remedy the
    // one-column refusal below would deny. (On a nested block such random
    // effects never reach here: `resolve_contrast` resolves each.)
    if absorbed.iter().all(Absorbed::combined) {
        return format!(
            "{block}: the levels absorb {} and {last} in combination ({}): none of them alone, \
             but moved together they move `y` as the levels do, so the model is not identified \
             under any contrast. Drop one of the random effects, or remove the block.",
            rest.join(", "),
            sites.join("; "),
        );
    }
    // A one-column block has no within rescue and dropping one random effect
    // does not help: each level lies within one unit, so the block absorbs
    // every one of them on its own (#1880).
    if decl.columns().len() == 1 {
        let within = if unit == "subject" {
            "a single subject".to_string()
        } else {
            format!("a single {unit} of one subject")
        };
        return format!(
            "{block}: each `{}` level lies within {within}, and the levels absorb {} and {last} \
             ({}), so the model is not identified under any contrast. Dropping one random effect \
             leaves the block absorbing the other: remove the block, or drop all of them.",
            decl.columns()[0],
            rest.join(", "),
            sites.join("; "),
        );
    }
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
    // Keyed on `level_key`, the one spelling of level equality the level-table check
    // uses too (#1799), rather than a linear `==` scan per record. A level holding a
    // `NaN` is left out, since `==` matched it to nothing (`NaN != NaN`), and the
    // first entry of a repeated level wins, as `find` gave.
    let mut by_key: HashMap<Vec<u64>, usize> = HashMap::with_capacity(table.len());
    for (level, i) in table {
        if !level.values.iter().any(|v| v.is_nan()) {
            by_key.entry(level_key(&level.values)).or_insert(*i);
        }
    }
    let index_of =
        |values: &[f64]| -> Option<f64> { by_key.get(&level_key(values)).map(|&i| i as f64) };

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
    /// The level's standard error on θ's own scale (#1890), from the covariance
    /// handed to [`theta_level_values`]. A free level's is exactly its `se_theta`
    /// entry, conventions included: `0.0` for a `FIX` block, and also `0.0` for a
    /// non-positive packed variance, as `se_theta` reports it. A sum-to-zero
    /// dependent level's is the delta-method `sqrt(dᵀ C_g d)` over its group's free
    /// θ, `C_g` being their block of the packed covariance and `d` the packing
    /// Jacobian (`θ` for a log-packed θ, `1` otherwise); a dependent level of a `FIX`
    /// block is therefore `Some(0.0)`. `None` for a level with an empty range — a
    /// `ref` reference level, or a group with no θ at all — which is a constant, not
    /// an estimate; for every level when no covariance is given; and for a dependent
    /// level whose quadratic form is negative or not finite. That last rule is
    /// deliberately stricter than the free levels' `se_theta` convention.
    pub se: Option<f64>,
    /// The level's 95% SIR interval (2.5th and 97.5th percentiles) over the retained
    /// SIR draws handed to [`theta_level_values`] (#1890): the level's value
    /// evaluated on each draw. A free level's is exactly its `sir_ci_theta` entry.
    /// `None` for a reference level, and for every level when no draws (or an empty
    /// set) are given.
    pub sir_ci: Option<(f64, f64)>,
}

/// Every level's value, free and dependent, of each bound level block, keyed by
/// block name and in [`theta_level_map`](crate::theta_level_map) order (#1623), with
/// its standard error and SIR interval when the fit's uncertainty is supplied (#1890).
///
/// The value is what the model's own evaluator reads for that level at `theta` — the
/// one every prediction and objective uses — so a dependent level is reported exactly
/// as the model applies it, never re-derived here. `theta_index` places each free
/// level in the θ vector.
///
/// `covariance` is the fit's `covariance_matrix`, on the **packed** scale the
/// covariance step works in (log for a θ with a lower bound `>= 0`, identity
/// otherwise). Pass `None` when the fit has none (a failed or not-requested step,
/// or a SIR fallback), and every `se` is `None`. `sir_resamples_packed` is the fit's
/// `sir_resamples_packed`, kept only under `sir_keep_samples = true`; `None` leaves
/// every `sir_ci` `None`. See [`ThetaLevelValue`] for each level's rule.
///
/// `theta` must be laid out for this bound model: the θ of a fit of `model`, or of a
/// model rebound with [`bind_from_fit`]. A θ whose length is not the model's is
/// refused, as is a covariance that is not square or has fewer rows than θ, and a
/// SIR draw shorter than θ. Ones of the right size but from a different binding
/// cannot be detected, and are read at the wrong positions. The packing — which θ
/// are log-packed, and so the Jacobian and the draws' back-transform — is read from
/// `model`'s declared bounds (`default_params.theta_lower`), which are the fit's
/// unless the fit was given other bounds by hand.
///
/// Unbound level blocks and counted `theta NAME[N]` blocks are omitted, as in
/// [`theta_level_map`](crate::theta_level_map); a model with neither gives an empty map.
pub fn theta_level_values(
    model: &crate::types::CompiledModel,
    theta: &[f64],
    covariance: Option<&nalgebra::DMatrix<f64>>,
    sir_resamples_packed: Option<&[Vec<f64>]>,
) -> Result<HashMap<String, Vec<ThetaLevelValue>>, String> {
    let expected = model.default_params.theta.len();
    if theta.len() != expected {
        return Err(format!(
            "the supplied theta has {} values but this model has {expected}; level values \
             are read from theta by position",
            theta.len()
        ));
    }
    if let Some(c) = covariance {
        if c.nrows() != c.ncols() || c.nrows() < expected {
            return Err(format!(
                "the supplied covariance is {}x{} but must be square with at least the \
                 model's {expected} theta rows; level standard errors are read from it by \
                 position",
                c.nrows(),
                c.ncols()
            ));
        }
    }
    if let Some(short) = sir_resamples_packed
        .into_iter()
        .flatten()
        .find(|d| d.len() < expected)
    {
        return Err(format!(
            "a supplied SIR draw has {} values but must hold at least the model's \
             {expected} theta; level intervals are read from it by position",
            short.len()
        ));
    }
    let lower = &model.default_params.theta_lower;
    // Each draw's θ, through the one back-transform `unpack_params` uses, so a free
    // level's interval is the `sir_ci_theta` the SIR step reports. No draws is no
    // interval: `percentile_ci` of an empty set is `(NaN, NaN)`, not an answer.
    let draw_thetas: Option<Vec<Vec<f64>>> = sir_resamples_packed
        .filter(|draws| !draws.is_empty())
        .map(|draws| {
            draws
                .iter()
                .map(|d| {
                    (0..expected)
                        .map(|k| theta_from_packed(d[k], lower[k]))
                        .collect()
                })
                .collect()
        });
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
            .map(|(i, (label, &rule))| {
                let index = (i + 1) as f64;
                // A reference level, or a group with no free θ, is the empty range: a
                // constant with no uncertainty. Keyed on the range, never on the
                // level's group, whose free θ a reference level does not read.
                let constant = matches!(rule, LevelRule::NegSum(a, b) if a == b);
                ThetaLevelValue {
                    label: label.clone(),
                    value: eval_gather(&gather.spec, theta, index),
                    theta_index: match rule {
                        LevelRule::Free(k) => Some(k as usize),
                        LevelRule::NegSum(..) => None,
                    },
                    se: match covariance {
                        Some(c) if !constant => level_se(rule, c, theta, lower),
                        _ => None,
                    },
                    sir_ci: match &draw_thetas {
                        Some(draws) if !constant => {
                            let v: Vec<f64> = draws
                                .iter()
                                .map(|t| eval_gather(&gather.spec, t, index))
                                .collect();
                            Some(percentile_ci(&v))
                        }
                        _ => None,
                    },
                }
            })
            .collect();
        out.insert(decl.name().to_string(), values);
    }
    Ok(out)
}

/// One level's standard error from the packed covariance `c` (#1890). A free level
/// goes through `se_theta`'s own rule, so it keeps that rule's conventions — a
/// non-positive variance reports `0.0` — and agrees with `se_theta` to the bit. A
/// dependent level is the delta-method `sqrt(dᵀ C_g d)` over its range, `None` when
/// that form is negative or not finite: it has no `se_theta` entry to agree with, so
/// it is not clamped to `0`, which would report a certainty the matrix does not
/// carry. Whether `se_theta`'s own `0.0` convention should change is #1895.
fn level_se(
    rule: LevelRule,
    c: &nalgebra::DMatrix<f64>,
    theta: &[f64],
    lower: &[f64],
) -> Option<f64> {
    match rule {
        LevelRule::Free(k) => {
            let k = k as usize;
            Some(theta_se_from_packed(c[(k, k)], theta[k], lower[k]))
        }
        LevelRule::NegSum(a, b) => {
            let range = a as usize..b as usize;
            let d = |k: usize| theta_packing_jacobian(theta[k], lower[k]);
            let q: f64 = range
                .clone()
                .flat_map(|i| range.clone().map(move |j| (i, j)))
                .map(|(i, j)| d(i) * c[(i, j)] * d(j))
                .sum();
            (q.is_finite() && q >= 0.0).then(|| q.sqrt())
        }
    }
}

/// The levels of one `profile`d θ block (#1065) that each subject's scored rows read.
///
/// Each subject's objective depends on the structural θ and on these levels only, so
/// the profile solve differentiates a subject over its own set rather than over the
/// whole block. Built from the index column [`write_index_column`] wrote — per row
/// when the index varies within a subject, from the baseline when it does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveLevels {
    pub(crate) block: String,
    /// Per subject, in population order, the 0-based levels read, ascending.
    pub(crate) per_subject: Vec<Vec<usize>>,
}

/// [`ActiveLevels`] for every profiled block of `model`, in declaration order, on a
/// population whose level columns are written. A block some read of which is not
/// indexed by the one plain column (`PLACEBO[4 - STUDY]` beside `PLACEBO[STUDY]`,
/// `PLACEBO[K + 1]`) cannot be read off the data, so every subject is given every
/// level: a superset is safe, it only costs the solve some work. The parser decides
/// that over every read (`ThetaBlocks::profiled_index`), not over the plain-column
/// reads alone, which would name the wrong levels (#2103 review #2).
pub(crate) fn active_levels(model: &CompiledModel, population: &Population) -> Vec<ActiveLevels> {
    let blocks = &model.indiv_param_partials.theta_blocks;
    blocks
        .decls
        .iter()
        .filter(|d| d.profiled)
        .map(|d| {
            let n_levels = d.spec.levels.len();
            let column = blocks
                .profiled_index
                .iter()
                .find(|(b, _)| *b == d.name)
                .and_then(|(_, c)| c.clone());
            let per_subject = population
                .subjects
                .iter()
                .map(|s| match &column {
                    Some(c) => subject_levels(s, c, n_levels),
                    None => (0..n_levels).collect(),
                })
                .collect();
            ActiveLevels {
                block: d.name.clone(),
                per_subject,
            }
        })
        .collect()
}

/// The 0-based levels `subject`'s observation rows read from index column `column`.
/// A row's own snapshot wins over the baseline, as in the prediction.
fn subject_levels(subject: &Subject, column: &str, n_levels: usize) -> Vec<usize> {
    let mut out: Vec<usize> = (0..subject.obs_times.len())
        .filter_map(|j| {
            subject
                .obs_covariates
                .get(j)
                .and_then(|m| m.get(column))
                .or_else(|| subject.covariates.get(column))
                .copied()
        })
        .filter(|v| v.is_finite() && *v >= 1.0 && (v.round() as usize) <= n_levels)
        .map(|v| v.round() as usize - 1)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
#[path = "tests/theta_levels_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/profiled_levels_tests.rs"]
mod profiled_levels_tests;
