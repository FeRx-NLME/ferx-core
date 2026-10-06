//! Binding `[covariate_model]` statistics to data (#1111).
//!
//! A relation may state its centring constant symbolically — `center = median`,
//! `ref = mode`, `levels = auto`. That is deliberate: requiring a literal would
//! make every generated covariate model dataset-specific, which defeats the
//! automation the block exists for. But `median` is a property of the dataset,
//! which the model file cannot know.
//!
//! This closes that gap the same way `theta NAME[COL, ...]` level blocks close
//! theirs (#1064, [`super::levels`]):
//!
//! 1. **Summarise** each covariate the relations need, one value per subject.
//! 2. **Re-parse** the model with the statistics known, so the desugared
//!    expression carries the resolved literal and the compiled closures are
//!    built once against the real θ vector.
//!
//! Re-parsing costs milliseconds. The alternative — a late-bound slot the
//! compiled closures read at evaluation time — would put a data-dependent value
//! behind every covariate factor for the life of the model, and a model reused
//! against a second dataset (a bootstrap resample, a VPC on new data) would
//! silently keep the first dataset's centres.
//!
//! An **unbound** model is never allowed to run: [`assert_covariate_model_bound`]
//! is called from every entry point, so a symbolic statistic that never reached
//! a population is a loud error rather than a fit that quietly drops the
//! covariate effect it declared.

use std::collections::HashSet;

use crate::api::levels::{declared_model, levels_hold_on, unbound_bindings, Reads};
use crate::parser::covariate_model::CovariateStatBindings;
use crate::parser::model_parser::parse_full_model_with;
use crate::types::{CompiledModel, CovariateSummary, ParsedModel, Population, Subject};

/// Resolve every symbolic statistic in `parsed`'s `[covariate_model]` block
/// against `population`, re-parsing the model so the desugared expressions
/// carry the resolved values.
///
/// A no-op (and no re-parse) for the overwhelming majority of models: those
/// with no `[covariate_model]` block, and those whose relations are all stated
/// with literal centres and explicit θ.
///
/// **Re-binding** (#1730). The relations are read from what `model_text`
/// declares, so a `parsed` already bound to other data is re-centred on
/// `population` exactly as a fresh parse would be. Level bindings `parsed` was
/// bound with are kept only when they are `population`'s: every subject carries
/// each block's index column, which [`bind_theta_levels`](crate::api::bind_theta_levels)
/// writes, and `population` shows exactly the bound levels, so the column was
/// written for this layout. Otherwise they are dropped, and the blocks are left for
/// `bind_theta_levels` to bind on `population`.
///
/// Also a no-op on a model laid out on a fit's bindings
/// ([`bind_from_fit`](crate::api::bind_from_fit),
/// [`layout_from_fit`](crate::api::layout_from_fit)): its centres are the fit's,
/// which the fitted θ was estimated against, and are never re-taken from the data
/// at hand. A model the deprecated `bind_theta_levels_from_fit` laid out without
/// the fit's statistics has no fit centre to keep, and is refused.
pub fn bind_covariate_stats(
    parsed: &mut ParsedModel,
    model_text: &str,
    population: &Population,
) -> Result<(), String> {
    if parsed.model.bound_from_fit() {
        // The fit's centres, when it gave any, are the ones to keep. A model laid out
        // by the deprecated binder without them has none, and the data's would be the
        // #1619 defect (#1735 review r1, finding 2).
        return match unresolved_lines(&parsed.model) {
            None => Ok(()),
            Some(lines) => Err(format!(
                "[covariate_model] relations still need data-derived statistics:\n\
                 {lines}\n\
                 This model is laid out on a fit's levels, so its centres must be the \
                 fit's too, not those of the data at hand. Bind it with `bind_from_fit` \
                 and the fit's `data_bindings`, which carry both, or install the fit's \
                 covariate statistics before binding its levels."
            )),
        };
    }
    let declared = declared_model(parsed, model_text, Reads::Stats)?;
    let Some(spec) = declared.covariate_model.as_ref() else {
        return Ok(());
    };
    if spec.unresolved().is_empty() {
        return Ok(());
    }
    // Summarise every covariate any relation reads, not only the unresolved
    // ones: a second relation on the same covariate costs one pass either way,
    // and the resulting table is what the fit YAML echoes.
    let wanted: HashSet<&str> = spec
        .relations
        .iter()
        .map(|r| r.covariate.as_str())
        .collect();
    let mut stats = CovariateStatBindings::new();
    for name in wanted {
        stats.insert(name.to_string(), summarize(name, population)?);
    }
    drop(declared);

    let mut bindings = unbound_bindings(parsed);
    if levels_hold_on(&parsed.model, &parsed.bindings.levels, population) {
        bindings.levels = parsed.bindings.levels.clone();
    }
    bindings.covariate_stats = stats;
    let model_name = parsed.model.name.clone();
    let rebound = parse_full_model_with(model_text, &bindings)?;
    parsed.bindings = bindings;
    parsed.model = rebound.model;
    parsed.model.name = model_name;
    assert_covariate_model_bound(&parsed.model)
}

/// Whether covariate statistics a model was bound with are still `population`'s
/// (#1730): every covariate they cover summarises to the same value on it. A
/// covariate `population` lacks counts as differing. Deterministic, since the
/// summary is the one the statistics were made by.
pub(crate) fn stats_hold_on(stats: &CovariateStatBindings, population: &Population) -> bool {
    stats
        .iter()
        .all(|(name, held)| summarize(name, population).is_ok_and(|s| &s == held))
}

/// Reject a model whose `[covariate_model]` still carries a relation waiting on
/// data-derived statistics.
///
/// Called from `fit` / `predict` / `simulate` / `check_model_data`, so the
/// failure mode a symbolic centre could otherwise have — running the fit with
/// the covariate effect silently absent — cannot happen. (A missing covariate
/// divides to `0.0` rather than `inf` in this engine, so nothing downstream
/// would have complained.)
pub fn assert_covariate_model_bound(model: &CompiledModel) -> Result<(), String> {
    let Some(lines) = unresolved_lines(model) else {
        return Ok(());
    };
    Err(format!(
        "[covariate_model] relations still need data-derived statistics:\n\
         {}\n\
         They are stated symbolically (`median` / `mean` / `min` / `max` / `mode` / \
         `levels = auto`), or use a form whose default bounds are functions of the covariate's \
         spread, so they can only be built once a dataset has been seen. Launch the fit from a \
         data file (`fit_from_files`, `ferx <model> --data ...`), which binds them; call \
         `ferx_core::api::bind_covariate_stats` before `fit` when driving the API directly; or \
         state the constants as literals (`center = 70`) and the θ explicitly \
         (`=> NAME(init, lower, upper)`), which needs no data at all. To run the model with \
         a fit's θ instead (a simulation, a prediction, SIR or a covariance step), bind it \
         with `ferx_core::api::bind_from_fit` and the fit's `data_bindings`: statistics \
         taken from the data at hand would centre the relations on that data, not on the \
         data the θ was estimated from.",
        lines
    ))
}

/// The source line of every relation of `model` still waiting on data-derived
/// statistics, one per line and indented; `None` when there is none.
fn unresolved_lines(model: &CompiledModel) -> Option<String> {
    let unresolved = model.covariate_model.as_ref()?.unresolved();
    if unresolved.is_empty() {
        return None;
    }
    let lines: Vec<String> = unresolved
        .iter()
        .map(|r| format!("  {}", r.source_line))
        .collect();
    Some(lines.join("\n"))
}

/// The covariates the model's still-unresolved relations read, sorted and
/// deduplicated: what a from-fit binding must supply statistics for (#1619).
pub(crate) fn symbolic_covariates(model: &CompiledModel) -> Vec<String> {
    let Some(spec) = model.covariate_model.as_ref() else {
        return Vec::new();
    };
    let mut names: Vec<String> = spec
        .unresolved()
        .iter()
        .map(|r| r.covariate.clone())
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// Check a fit's covariate statistics against the relations of `model`, before
/// anything is bound (#1619): every covariate an unresolved relation reads must
/// have an entry, and no entry may name a covariate no relation reads. Like the
/// level bindings, the statistics travel in a `.fitrx` as data a caller can edit,
/// and the binder must never fill a gap by summarising the population at hand.
pub(crate) fn validate_fitted_stats(
    model: &CompiledModel,
    fitted: &CovariateStatBindings,
) -> Result<(), String> {
    let missing: Vec<String> = symbolic_covariates(model)
        .into_iter()
        .filter(|c| !fitted.contains_key(c))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "[covariate_model] relations state a statistic of {} symbolically, but the fit's \
             covariate statistics carry no entry for it: they are not the statistics this \
             model was fitted with.",
            backticked(&missing)
        ));
    }
    let read: HashSet<&str> = model
        .covariate_model
        .as_ref()
        .map(|spec| {
            spec.relations
                .iter()
                .map(|r| r.covariate.as_str())
                .collect()
        })
        .unwrap_or_default();
    let mut extra: Vec<String> = fitted
        .keys()
        .filter(|c| !read.contains(c.as_str()))
        .cloned()
        .collect();
    if !extra.is_empty() {
        extra.sort_unstable();
        return Err(format!(
            "the fit's covariate statistics carry {}, which no [covariate_model] relation of \
             this model reads: the bindings belong to a different model.",
            backticked(&extra)
        ));
    }
    Ok(())
}

/// The first covariate statistic of `model` that `population` does not reproduce
/// (#1729): which covariate, which field, and both values.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StatMismatch {
    pub(crate) covariate: String,
    /// `median`, `mean`, `min`, `max`, `mode` or `levels`.
    pub(crate) field: &'static str,
    /// The model's value, formatted.
    pub(crate) model: String,
    /// The population's value, formatted.
    pub(crate) data: String,
}

/// Check that `model`'s bound covariate statistics are the ones `population`
/// summarises to (#1729): `Ok(None)` when every statistic matches, `Ok(Some(..))`
/// naming the first that does not, `Err` when a covariate cannot be summarised.
///
/// For a fit that recorded no bindings (an older `.fitrx`) this is the one way to
/// tell a lent model bound on the fit's data from one bound on a simulation
/// design. It re-summarises with `summarize` itself, so the fitted model passes
/// bit for bit (same function, same data, sorted-order summation), and the values
/// are compared exactly: a tolerance would accept a design whose median sits
/// within it. Covariates are visited in name order, so the one named is
/// reproducible.
pub(crate) fn check_stats_on(
    model: &CompiledModel,
    population: &Population,
) -> Result<Option<StatMismatch>, String> {
    let stats = &model.data_bindings().covariate_stats;
    let mut names: Vec<&String> = stats.keys().collect();
    names.sort_unstable();
    for name in names {
        let data = summarize(name, population)?;
        if let Some((field, model, data)) = first_difference(&stats[name], &data) {
            return Ok(Some(StatMismatch {
                covariate: name.clone(),
                field,
                model,
                data,
            }));
        }
    }
    Ok(None)
}

/// The first field, in the order median, mean, min, max, mode, levels, where two
/// summaries differ, with both values formatted.
fn first_difference(
    model: &CovariateSummary,
    data: &CovariateSummary,
) -> Option<(&'static str, String, String)> {
    let scalars = [
        ("median", model.median, data.median),
        ("mean", model.mean, data.mean),
        ("min", model.min, data.min),
        ("max", model.max, data.max),
        ("mode", model.mode, data.mode),
    ];
    if let Some((field, m, d)) = scalars.into_iter().find(|(_, m, d)| m != d) {
        return Some((field, m.to_string(), d.to_string()));
    }
    (model.levels != data.levels).then(|| {
        (
            "levels",
            format!("{:?}", model.levels),
            format!("{:?}", data.levels),
        )
    })
}

fn backticked(names: &[String]) -> String {
    names
        .iter()
        .map(|n| format!("`{n}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Summarise one covariate over the population.
///
/// Weighting is **per subject**, matching PsN: a subject with forty samples must
/// not drag the median toward their own weight. A time-varying covariate
/// contributes each distinct value it takes within the subject, so a weight that
/// changes across an admission is not collapsed to its first record.
///
/// **Every** event snapshot counts, not only the observation ones: a covariate
/// change carried on a dose (EVID=1), a covariate-change marker (EVID=2) or a
/// reset (EVID=3/4) is read by the event-driven evaluator, so a value the model
/// actually evaluates at must not be missing from `min`/`max` (which become
/// default θ bounds) or from `levels = auto` (where an omitted level silently
/// collapses to the reference factor).
fn summarize(name: &str, population: &Population) -> Result<CovariateSummary, String> {
    let mut values: Vec<f64> = Vec::with_capacity(population.subjects.len());
    for subject in &population.subjects {
        values.extend(subject_covariate_values(subject, name));
    }
    if values.is_empty() {
        return Err(format!(
            "[covariate_model] needs summary statistics for covariate \
             `{name}`, but the dataset carries no non-missing value for it"
        ));
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let median = if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    };
    Ok(CovariateSummary {
        median,
        mean: values.iter().sum::<f64>() / n as f64,
        min: values[0],
        max: values[n - 1],
        mode: mode_of(&values),
        levels: distinct(&values),
    })
}

/// Every distinct finite value `name` takes within one subject, in first-seen
/// order: the subject-static fallback, then each observation, dose, EVID=2
/// covariate-change marker and EVID=3/4 reset snapshot.
///
/// Shared with [`crate::api::validation`]'s categorical-level check, so the
/// values a summary is built from and the values a declared level set is
/// checked against are, by construction, the same set.
pub(crate) fn subject_covariate_values(subject: &Subject, name: &str) -> Vec<f64> {
    let mut seen: Vec<f64> = Vec::new();
    let mut push = |v: f64| {
        if v.is_finite() && !seen.contains(&v) {
            seen.push(v);
        }
    };
    if let Some(v) = subject.covariates.get(name) {
        push(*v);
    }
    for snapshot in subject
        .obs_covariates
        .iter()
        .chain(&subject.dose_covariates)
        .chain(&subject.pk_only_covariates)
        .chain(&subject.reset_covariates)
    {
        if let Some(v) = snapshot.get(name) {
            push(*v);
        }
    }
    seen
}

/// The most common value in a sorted slice. Ties break toward the smaller
/// value, so the reference level a `categorical(ref = mode)` picks is
/// reproducible run to run rather than a function of iteration order.
fn mode_of(sorted: &[f64]) -> f64 {
    let mut best = sorted[0];
    let mut best_count = 0usize;
    let mut current = sorted[0];
    let mut count = 0usize;
    for v in sorted {
        if *v == current {
            count += 1;
        } else {
            current = *v;
            count = 1;
        }
        if count > best_count {
            best_count = count;
            best = current;
        }
    }
    best
}

/// Distinct values of a sorted slice, ascending — what `levels = auto` binds to.
fn distinct(sorted: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::new();
    for v in sorted {
        if out.last() != Some(v) {
            out.push(*v);
        }
    }
    out
}

#[cfg(test)]
#[path = "tests/covariate_stats_tests.rs"]
mod tests;
