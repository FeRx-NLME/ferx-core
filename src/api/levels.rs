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

use crate::parser::model_parser::{
    eval_gather, level_index_column, parse_full_model_with, LevelBinding, LevelBindings,
    LevelBlockDecl, LevelContrast, LevelRule,
};
use crate::types::{ParsedModel, Population, Subject};

/// The record time, addressable as a level column even though it is not a
/// covariate.
const TIME_COLUMN: &str = "TIME";

/// Synthesized index columns are prefixed so they cannot collide with a real
/// data column — and so the file readers know not to look for them in the CSV.
pub(crate) const LEVEL_INDEX_PREFIX: &str = "__level_";

/// Whether `name` is a synthesized level index column rather than a real data
/// column. The CSV readers use this to skip it when collecting the columns they
/// must find in the file.
pub(crate) fn is_level_index_column(name: &str) -> bool {
    name.starts_with(LEVEL_INDEX_PREFIX)
}

/// Bind every level block in `parsed` against `population`, mutating
/// both: the population gains the synthesized index columns, and `parsed.model`
/// is replaced by a re-parse that knows the level counts.
///
/// A no-op (and no re-parse) for the overwhelming majority of models, which
/// declare no level block at all.
pub fn bind_theta_levels(
    parsed: &mut ParsedModel,
    model_text: &str,
    population: &mut Population,
) -> Result<(), String> {
    let decls: Vec<LevelBlockDecl> = parsed.model.theta_blocks().level_blocks().to_vec();
    if decls.is_empty() {
        return Ok(());
    }

    let mut bindings = LevelBindings::new();
    for decl in &decls {
        let levels = discover_levels(decl, population)?;
        let contrast = resolve_contrast(decl, &levels, population)?;
        let groups = assign_groups(decl, &levels, contrast);
        let table: Vec<(Level, usize)> = levels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), i + 1))
            .collect();
        write_index_column(decl, &table, population)?;
        bindings.insert(
            decl.name().to_string(),
            LevelBinding {
                labels: levels.iter().map(|l| l.label(decl.columns())).collect(),
                groups,
                contrast,
            },
        );
    }

    let model_name = parsed.model.name.clone();
    // Re-parse with *every* binding this model has been given, not just the
    // level ones: a model that also declares `[covariate_model]` statistics
    // (#1111) may have had those bound already, and re-parsing with the level
    // bindings alone would drop them.
    parsed.bindings.levels = bindings;
    let rebound = parse_full_model_with(model_text, &parsed.bindings)?;
    parsed.model = rebound.model;
    parsed.model.name = model_name;
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
/// not. Nothing is written to `population` unless every block binds.
pub fn bind_theta_levels_from_fit(
    parsed: &mut ParsedModel,
    model_text: &str,
    population: &mut Population,
    fitted: &LevelBindings,
) -> Result<(), String> {
    let decls: Vec<LevelBlockDecl> = parsed.model.theta_blocks().level_blocks().to_vec();
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
    if decls.is_empty() {
        return Ok(());
    }

    let mut tables: Vec<Vec<(Level, usize)>> = Vec::with_capacity(decls.len());
    for decl in &decls {
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
        let mut table = Vec::new();
        let mut unseen = Vec::new();
        for level in discover_levels(decl, population)? {
            let label = level.label(decl.columns());
            match binding.labels.iter().position(|l| *l == label) {
                Some(i) => table.push((level, i + 1)),
                None => unseen.push(label),
            }
        }
        if !unseen.is_empty() {
            return Err(unseen_levels_message(decl, &unseen));
        }
        tables.push(table);
    }

    for (decl, table) in decls.iter().zip(&tables) {
        write_index_column(decl, table, population)?;
    }
    let model_name = parsed.model.name.clone();
    parsed.bindings.levels = fitted.clone();
    let rebound = parse_full_model_with(model_text, &parsed.bindings)?;
    parsed.model = rebound.model;
    parsed.model.name = model_name;
    Ok(())
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
            let level = Level { values };
            if !levels.contains(&level) {
                levels.push(level);
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

/// Whether the block's leading columns identify subjects one-to-one: every
/// subject's records share one combination, and no two subjects share it.
///
/// This is the data-side half of "is there a random effect at a grouping
/// coarser than or equal to the block's". η in this engine is per subject, so
/// only when the leading tuple identifies one subject can that subject's η
/// carry the corresponding group mean.
fn leading_identifies_subjects(decl: &LevelBlockDecl, population: &Population) -> bool {
    if decl.columns().len() < 2 {
        return false;
    }
    let leading = &decl.columns()[..decl.columns().len() - 1];
    let mut subject_keys: Vec<Vec<f64>> = Vec::with_capacity(population.subjects.len());
    for subject in &population.subjects {
        let mut first: Option<Vec<f64>> = None;
        for j in 0..subject.obs_times.len() {
            let key: Option<Vec<f64>> = leading
                .iter()
                .map(|c| column_value(subject, c, j))
                .collect();
            let Some(key) = key else { return false };
            match &first {
                None => first = Some(key),
                Some(f) if *f == key => {}
                Some(_) => return false,
            }
        }
        let Some(key) = first else { return false };
        if subject_keys.contains(&key) {
            return false;
        }
        subject_keys.push(key);
    }
    true
}

/// Resolve [`LevelContrast::Auto`], and reject the configurations that are
/// still rank-deficient once resolved.
fn resolve_contrast(
    decl: &LevelBlockDecl,
    levels: &[Level],
    population: &Population,
) -> Result<LevelContrast, String> {
    let nested = leading_identifies_subjects(decl, population);
    let resolved = match decl.contrast() {
        LevelContrast::Auto => {
            if decl.shares_scale_with_eta() && nested {
                LevelContrast::SumToZeroWithin
            } else {
                LevelContrast::SumToZero
            }
        }
        other => other,
    };

    // The configuration the feature exists to serve — an unstructured placebo
    // effect per study × timepoint under between-study variability — is
    // over-parameterised under any convention that leaves each group's mean
    // free: that study's η *is* the mean of its own levels. A check that only
    // looked for a fixed intercept would wave it through, which is precisely
    // the silent flat direction this codebase treats as a bug.
    if decl.shares_scale_with_eta()
        && nested
        && matches!(
            resolved,
            LevelContrast::SumToZero | LevelContrast::Ref | LevelContrast::Unconstrained
        )
    {
        let leading = decl.columns()[..decl.columns().len() - 1].join(", ");
        return Err(format!(
            "theta {}[{}]: `contrast = {}` leaves each {leading} group's mean free, \
             but the individual parameter that reads this block also carries a random effect \
             at that grouping — the two are the same quantity, so the model is not identified. \
             Use `contrast = sum_to_zero_within` (the default for this shape), or drop the \
             random effect.",
            decl.name(),
            decl.columns().join(", "),
            contrast_token(resolved),
        ));
    }

    if levels.len() == 1 && matches!(resolved, LevelContrast::SumToZero) {
        // A one-level block under sum-to-zero has no free θ at all: the single
        // level is pinned at 0. That is a degenerate model, not a normalization.
        return Err(format!(
            "theta {}[{}]: the data carries a single level, which sum-to-zero pins \
             at 0. Use `contrast = none` if a single constant is what you meant.",
            decl.name(),
            decl.columns().join(", ")
        ));
    }

    Ok(resolved)
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
/// [`bind_theta_levels`], its position in the fit for [`bind_theta_levels_from_fit`].
/// Both binders share this one writer, so the dose, EVID=2 and reset handling below
/// cannot drift between them.
///
/// When the index is constant within a subject it goes into the subject-level
/// covariate map only — no time-varying machinery is engaged, so the model
/// keeps whatever fast path it had. When it varies (the unstructured-placebo
/// case, where the index moves with the timepoint) the per-event snapshots are
/// materialised, which is exactly what a genuinely per-record parameter needs.
fn write_index_column(
    decl: &LevelBlockDecl,
    table: &[(Level, usize)],
    population: &mut Population,
) -> Result<(), String> {
    let column = level_index_column(decl.name());
    let index_of = |values: &[f64]| -> Option<f64> {
        table
            .iter()
            .find(|(l, _)| l.values == values)
            .map(|&(_, i)| i as f64)
    };

    for subject in population.subjects.iter_mut() {
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

        let first = obs_index.first().copied().unwrap_or(1.0);
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

    if !population.covariate_names.contains(&column) {
        population.covariate_names.push(column);
    }
    Ok(())
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
/// model rebound with [`bind_theta_levels_from_fit`]. A θ whose length is not the
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
