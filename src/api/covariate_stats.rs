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
pub fn bind_covariate_stats(
    parsed: &mut ParsedModel,
    model_text: &str,
    population: &Population,
) -> Result<(), String> {
    let Some(spec) = parsed.model.covariate_model.as_ref() else {
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

    let model_name = parsed.model.name.clone();
    parsed.bindings.covariate_stats = stats;
    let rebound = parse_full_model_with(model_text, &parsed.bindings)?;
    parsed.model = rebound.model;
    parsed.model.name = model_name;
    assert_covariate_model_bound(&parsed.model)
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
    let Some(spec) = model.covariate_model.as_ref() else {
        return Ok(());
    };
    let unresolved = spec.unresolved();
    if unresolved.is_empty() {
        return Ok(());
    }
    let lines: Vec<String> = unresolved
        .iter()
        .map(|r| format!("  {}", r.source_line))
        .collect();
    Err(format!(
        "[covariate_model] relations still need data-derived statistics:\n\
         {}\n\
         They are stated symbolically (`median` / `mean` / `min` / `max` / `mode` / \
         `levels = auto`), or use a form whose default bounds are functions of the covariate's \
         spread, so they can only be built once a dataset has been seen. Launch the fit from a \
         data file (`fit_from_files`, `ferx <model> --data ...`), which binds them; call \
         `ferx_core::api::bind_covariate_stats` before `fit` when driving the API directly; or \
         state the constants as literals (`center = 70`) and the θ explicitly \
         (`=> NAME(init, lower, upper)`), which needs no data at all.",
        lines.join("\n")
    ))
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
