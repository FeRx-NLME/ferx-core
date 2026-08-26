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
    level_index_column, parse_full_model_bound, LevelBinding, LevelBindings, LevelBlockDecl,
    LevelContrast,
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
        write_index_column(decl, &levels, population)?;
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
    let rebound = parse_full_model_bound(model_text, &bindings)?;
    parsed.model = rebound.model;
    parsed.model.name = model_name;
    Ok(())
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
/// When the index is constant within a subject it goes into the subject-level
/// covariate map only — no time-varying machinery is engaged, so the model
/// keeps whatever fast path it had. When it varies (the unstructured-placebo
/// case, where the index moves with the timepoint) the per-event snapshots are
/// materialised, which is exactly what a genuinely per-record parameter needs.
fn write_index_column(
    decl: &LevelBlockDecl,
    levels: &[Level],
    population: &mut Population,
) -> Result<(), String> {
    let column = level_index_column(decl.name());
    let index_of = |values: &[f64]| -> Option<f64> {
        levels
            .iter()
            .position(|l| l.values == values)
            .map(|i| (i + 1) as f64)
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

#[cfg(test)]
#[path = "tests/theta_levels_tests.rs"]
mod tests;
