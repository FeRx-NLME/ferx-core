//! Model/data validation gauntlet shared verbatim by `fit()`/`fit_inner()`
//! and the `ferx check` orchestrator (`validate_model_file`). Pure predicate
//! functions returning `Vec<Diagnostic>` / `Option<String>` (or panicking
//! asserts) — no numerical surface, no f64 arithmetic reordered by this move.
//!
//! Extracted verbatim from `api.rs`; every path stays resolvable through the
//! `pub` / `pub(crate)` re-exports in the parent module.
use super::*;

/// Does this identifier read as a random-effect name (`ETA_CL`, `eta_v`,
/// `KAPPA_CL`)?
///
/// A random effect exists only because an `omega` / `kappa` line declares it —
/// there is no reserved prefix — so an identifier that *looks* like one but binds
/// to nothing is far more likely a forgotten declaration than a covariate the data
/// was supposed to carry. Since #989 dropped the parse-time `No omega parameters
/// defined` rejection, this heuristic is what keeps a model whose whole `omega`
/// block was deleted from silently resolving its etas as covariate columns.
/// Prefix-only and case-insensitive; a genuine `ETA_CL` *column* in the data
/// resolves normally and never reaches this predicate.
pub(crate) fn is_random_effect_shaped(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    upper.starts_with("ETA") || upper.starts_with("KAPPA")
}

/// The model's unresolved identifiers that read as random-effect names.
///
/// Data-independent: reads `referenced_covariates`, which holds every identifier
/// the parser could not bind to a theta, a declared eta/kappa, or a built-in.
pub(crate) fn undeclared_random_effect_names(model: &CompiledModel) -> Vec<&str> {
    model
        .referenced_covariates
        .iter()
        .map(|s| s.as_str())
        .filter(|n| is_random_effect_shaped(n))
        .collect()
}

/// Message body shared by the data-present error and the data-absent warning, so
/// the two cannot drift apart.
pub(crate) fn undeclared_random_effect_message(names: &[&str]) -> String {
    format!(
        "Model references {} as {} but no `omega` / `kappa` declaration defines {}. \
         Declare the random effect in [parameters] (e.g. `omega {} ~ 0.09`), or — if this \
         is meant to be a data column — add that column to the data file. A model with no \
         random effects at all is valid (#989): to make it fixed-effects-only, drop the \
         `exp({})` term as well as the omega line.",
        names.join(", "),
        if names.len() == 1 {
            "a random effect"
        } else {
            "random effects"
        },
        if names.len() == 1 { "it" } else { "them" },
        names[0],
        names[0],
    )
}

/// Fail early if the model references covariates that the data doesn't carry.
/// Case-sensitive: `CRCL` and `crcl` are distinct names. Historically a missing
/// covariate silently evaluated to zero, which left fits stuck at the initial
/// estimates with no visible diagnostic (see commit introducing this check).
///
/// Unresolved names that read as random effects get their own `E_ETA_NOT_DECLARED`
/// diagnostic, pushed first so [`first_error`] — which is what `fit()` reports —
/// surfaces the actionable message instead of "covariate not found in data". The
/// `E_MISSING_COVARIATE` message text stays byte-for-byte identical to the
/// historical `Err(String)` for the names that remain.
///
/// "Carries" means the *predictor* can resolve the name, which is a slightly wider
/// test than membership in `Population::covariate_names`. A CSV-read population
/// always lists every covariate column there, but an in-memory `Population` built
/// programmatically may populate each `Subject::covariates` map and leave the name
/// list empty — a construction the predictors have always resolved fine, since they
/// read the per-subject map (`Subject::obs_cov`), not the list. So a name every
/// subject's map carries counts as present. Without this, `predict()` gaining the
/// check (#1028) would have started panicking on those callers even though nothing
/// about their predictions was undefined.
pub(crate) fn check_covariates(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    let carried = |name: &str| -> bool {
        population.covariate_names.iter().any(|n| n == name)
            || (!population.subjects.is_empty()
                && population
                    .subjects
                    .iter()
                    .all(|s| s.covariates.contains_key(name)))
    };
    let missing: Vec<&str> = model
        .referenced_covariates
        .iter()
        .filter(|name| !carried(name))
        .map(|s| s.as_str())
        .collect();

    if missing.is_empty() {
        return Vec::new();
    }

    let (eta_shaped, plain): (Vec<&str>, Vec<&str>) =
        missing.iter().partition(|n| is_random_effect_shaped(n));

    let mut diags = Vec::new();
    if !eta_shaped.is_empty() {
        diags.push(
            Diagnostic::error(
                "E_ETA_NOT_DECLARED",
                undeclared_random_effect_message(&eta_shaped),
            )
            .with_block("parameters")
            .with_suggestion(format!("declare `omega {} ~ <variance>`", eta_shaped[0])),
        );
    }

    if plain.is_empty() {
        return diags;
    }

    let available = if population.covariate_names.is_empty() {
        "(none)".to_string()
    } else {
        population.covariate_names.join(", ")
    };
    // `TAFD` / `TAD` / `MACHEPS` are solver-injected built-ins *inside `[odes]`* and
    // ordinary covariates everywhere else — including `[scaling]`, which is
    // deliberate (`docs/model-file/scaling.qmd`) so a dataset that really carries a
    // `TAD` column can use it. The failure mode is a user who expected the `[odes]`
    // built-in and gets the bare "covariate not found in data: TAD", which reads as
    // a typo report rather than a scope explanation. Name the scope when one of them
    // is among the missing (#1028). Appended only in that case, so the historical
    // message text is byte-for-byte unchanged for every ordinary covariate.
    const ODE_ONLY_BUILTINS: &[&str] = &["TAFD", "TAD", "MACHEPS"];
    let builtin_shaped: Vec<&str> = plain
        .iter()
        .copied()
        .filter(|n| ODE_ONLY_BUILTINS.contains(n))
        .collect();
    let builtin_hint = if builtin_shaped.is_empty() {
        String::new()
    } else {
        format!(
            " Note: {} {} a solver-injected built-in only inside `[odes]`; anywhere else \
             (including `[scaling]`) the name is an ordinary covariate and must be a data \
             column. Compute it as an `[odes]` intermediate and read that state instead.",
            builtin_shaped.join(", "),
            if builtin_shaped.len() == 1 {
                "is"
            } else {
                "are"
            },
        )
    };
    diags.push(
        Diagnostic::error(
            "E_MISSING_COVARIATE",
            format!(
                "Model references covariate(s) not found in data (case-sensitive): {}. \
                 Available covariate columns: {}.{}",
                plain.join(", "),
                available,
                builtin_hint,
            ),
        )
        .with_suggestion(format!("available covariate columns: {}", available)),
    );
    diags
}

/// Map an error string from [`read_population_for`] onto a `ferx check`
/// diagnostic, so the covariate-validation failures the strict reader raises at
/// fit time surface with the same code/block in `ferx check` (rather than as a
/// generic `E_DATA`). Classification keys off the reader's stable message
/// prefixes ([`ERR_COV_MISSING_COLUMNS`] / [`ERR_COV_NON_NUMERIC`]).
fn covariate_read_diagnostic(err: &str, path: &str) -> Diagnostic {
    if err.starts_with(ERR_COV_MISSING_COLUMNS) {
        Diagnostic::error("E_MISSING_COVARIATE", err.to_string()).with_block("covariates")
    } else if err.starts_with(ERR_COV_NON_NUMERIC) {
        Diagnostic::error("E_COVARIATE_NOT_NUMERIC", err.to_string()).with_block("covariates")
    } else {
        Diagnostic::error(
            "E_DATA",
            format!("Failed to read data file '{}': {}", path, err),
        )
    }
}

/// Per-CMT scaling needs every observed CMT to have an entry in the
/// `ScalingSpec::PerCmt` / `OdeReadout::PerCmt` map. Wraps the existing
/// `pk::validate_per_cmt_scaling` (which the parser can't run — it doesn't see
/// the data), preserving its message verbatim.
fn check_per_cmt_scaling(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    match pk::validate_per_cmt_scaling(model, &population.subjects) {
        Ok(()) => Vec::new(),
        Err(msg) => vec![Diagnostic::error("E_PER_CMT_SCALING", msg).with_block("scaling")],
    }
}

/// Per-CMT (multi-endpoint) error models: every observed CMT must have a
/// matching `CMT=N:` entry in `[error_model]`.
fn check_per_cmt_error_model(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    let crate::types::ErrorSpec::PerCmt(map) = &model.error_spec else {
        return Vec::new();
    };
    use std::collections::BTreeSet;
    let mut missing = BTreeSet::new();
    for subj in &population.subjects {
        for &cmt in &subj.obs_cmts {
            if !map.contains_key(&cmt) {
                missing.insert(cmt);
            }
        }
    }
    if missing.is_empty() {
        return Vec::new();
    }
    let list = missing
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    vec![Diagnostic::error(
        "E_PER_CMT_ERROR_MODEL",
        format!(
            "[error_model] has no entry for observed compartment(s) {}; \
             add a `CMT=N: DV ~ ...` line for each observed CMT.",
            list
        ),
    )
    .with_block("error_model")]
}

/// Every per-observation residual-error magnitude (#484) — including the one a
/// `weight = <expr>` modifier compiles to (#1029) — must be strictly positive
/// and finite at the initial estimates.
///
/// The magnitude multiplies a sigma loading, so the observation's variance is
/// proportional to its square: a zero multiplier collapses the variance onto the
/// `1e-12` floor and lets that single row dominate the entire objective, and a
/// non-finite one poisons the OFV. The concrete way to get there is a missing
/// cell in the weight column — an MBMA dataset with no reported standard error
/// for one arm — which evaluates the covariate to `0` and silently gives that
/// row infinite precision. A negative multiplier squares away to the same
/// variance as its absolute value, so it is never what was meant either.
pub(crate) fn check_residual_magnitude(
    model: &CompiledModel,
    population: &Population,
) -> Vec<Diagnostic> {
    if !model.has_custom_ruv_magnitude() {
        return Vec::new();
    }
    let theta = &model.default_params.theta;
    for subj in &population.subjects {
        let Some(mult) = model.ruv_obs_mult(subj, theta) else {
            continue;
        };
        for (j, row) in mult.iter().enumerate() {
            for (k, &m) in row.iter().enumerate() {
                if m.is_finite() && m > 0.0 {
                    continue;
                }
                let time = subj.obs_times.get(j).copied().unwrap_or(f64::NAN);
                return vec![Diagnostic::error(
                    "E_RUV_MAGNITUDE_NONPOSITIVE",
                    format!(
                        "[error_model] residual-error magnitude for sigma slot {k} evaluates to \
                         {m} at subject `{}`, TIME {time} — it must be strictly positive and \
                         finite. The variance is proportional to its square, so a zero magnitude \
                         collapses that observation's variance onto the floor and lets one row \
                         dominate the fit. The usual cause is a missing or non-numeric cell in a \
                         covariate the magnitude (or a `weight = ...` modifier) references.",
                        subj.id
                    ),
                )
                .with_block("error_model")];
            }
        }
    }
    Vec::new()
}

/// A kappa's sample-size weight (#1031) must be strictly positive and finite at
/// every record, and constant within an occasion.
///
/// `κ_ik ~ N(0, Ω_IOV / W_ik)` is applied as `κ / sqrt(W)`, so a zero or
/// missing weight divides an individual parameter by zero and a negative one
/// takes `sqrt` of a negative number — both land as a non-finite prediction
/// deep inside the inner loop, where the cause is invisible. The concrete way
/// to get there is a blank arm-size cell in an MBMA dataset.
///
/// This is the *error* half; the within-occasion-variation warning is
/// [`check_kappa_weight_variation`]. They are separate walks so neither caller
/// pays for the other's work — the fatal list and the warning list are built by
/// different entry points, and folding both into one function meant every fit
/// walked every observation *and* every dose twice.
/// A `theta NAME ~ factor(...)` block that was never bound to data has no
/// levels, so every gather out of it is `NaN` (#1064).
///
/// `fit()` refuses such a model up front with its own message; this is the same
/// guard for the simulate paths, which do not go through `fit()`. Binding
/// happens in `api::factor::bind_factor_thetas`, which every file entry point
/// calls — an in-memory caller that assembled the `CompiledModel` itself is the
/// case this catches.
pub(crate) fn check_unbound_factor_thetas(model: &CompiledModel) -> Vec<Diagnostic> {
    model
        .theta_blocks
        .unbound_factors()
        .iter()
        .map(|name| {
            Diagnostic::error(
                "E_FACTOR_THETA_UNBOUND",
                format!(
                    "`theta {name} ~ factor(...)` was never bound to data, so it has no \
                     levels and every value gathered from it is NaN."
                ),
            )
            .with_block("parameters")
            .with_suggestion(format!(
                "run through a file entry point, which binds factor blocks against the \
                 dataset or the [simulation] design — or declare the block explicitly as \
                 `theta {name}[N](...)` and index it with your own column"
            ))
        })
        .collect()
}

/// Every `NAME[COLUMN]` gather index the data carries must be an integer level
/// the block actually has (#1064).
///
/// This is the loud half of the gather's index policy. The evaluator's half is
/// `NaN` — deliberately not `0.0`, since `x/0` already underflows to `0.0` here
/// and a silent zero would be indistinguishable from a legitimately estimated
/// level — but a NaN prediction reports as a failed fit, not as "your `PLA_IDX`
/// column is 1-based and you wrote it 0-based". So the data is walked before the
/// fit starts and the offending value named.
///
/// Only gathers whose index is a bare data column are checked; a computed index
/// (`PLACEBO[2 * K]`) has no single column to walk and falls back to the NaN
/// guard. At most one diagnostic is raised per (block, subject) — a mis-coded
/// index column is wrong on every row, and 100k copies of the same finding help
/// nobody.
pub(crate) fn check_theta_gather_indices(
    model: &CompiledModel,
    population: &Population,
) -> Vec<Diagnostic> {
    if model.theta_blocks.is_empty() {
        return Vec::new();
    }
    let mut diags = Vec::new();
    for (block, column, n_levels) in model.theta_blocks.index_columns() {
        for subj in &population.subjects {
            // Observation rows first — a level is a property of an observation —
            // then dose rows, which read the individual parameters too.
            let rows = (0..subj.obs_times.len())
                .map(|j| (subj.obs_times[j], subj.obs_cov(j)))
                .chain((0..subj.doses.len()).map(|d| (subj.doses[d].time, subj.dose_cov(d))));
            for (time, cov) in rows {
                let Some(&raw) = cov.get(column) else {
                    diags.push(
                        Diagnostic::error(
                            "E_THETA_GATHER_INDEX_MISSING",
                            format!(
                                "theta `{block}` is indexed by `{column}`, but subject `{}` \
                                 carries no `{column}` value at TIME {time}. Every record that \
                                 reads the block needs one.",
                                subj.id
                            ),
                        )
                        .with_block("individual_parameters"),
                    );
                    break;
                };
                let level = raw.round();
                let ok = raw.is_finite()
                    && (level - raw).abs() <= 1e-6
                    && level >= 1.0
                    && level as usize <= n_levels;
                if !ok {
                    diags.push(
                        Diagnostic::error(
                            "E_THETA_GATHER_INDEX_RANGE",
                            format!(
                                "theta `{block}` has {n_levels} levels, but its index column \
                                 `{column}` is {raw} on subject `{}` at TIME {time}. The index \
                                 is 1-based and must be a whole number in 1..={n_levels}.",
                                subj.id
                            ),
                        )
                        .with_block("individual_parameters")
                        .with_suggestion(format!(
                            "check that `{column}` is 1-based (not 0-based) and that \
                             `theta {block}[N]` declares enough levels"
                        )),
                    );
                    break;
                }
            }
        }
    }
    diags
}

pub(crate) fn check_kappa_weights(
    model: &CompiledModel,
    population: &Population,
) -> Vec<Diagnostic> {
    if !model.has_weighted_kappa() {
        return Vec::new();
    }
    let theta = &model.default_params.theta;
    for (k, weight) in model.kappa_weights.iter().enumerate() {
        let Some(weight) = weight else { continue };
        let name = model
            .kappa_names
            .get(k)
            .map(String::as_str)
            .unwrap_or("<kappa>");
        for subj in &population.subjects {
            for j in 0..subj.obs_times.len() {
                let time = subj.obs_times[j];
                let w = (weight.eval)(theta, subj.obs_cov(j), time);
                if !(w.is_finite() && w > 0.0) {
                    return vec![Diagnostic::error(
                        "E_KAPPA_WEIGHT_NONPOSITIVE",
                        format!(
                            "[parameters] kappa `{name}` weight `{}` evaluates to {w} at \
                             subject `{}`, TIME {time} — it must be strictly positive and \
                             finite. The weight declares κ ~ N(0, Ω_IOV / W) and is applied \
                             as κ/√W, so a zero or missing one divides an individual \
                             parameter by zero. The usual cause is a missing or non-numeric \
                             cell in the covariate it references.",
                            weight.expr, subj.id
                        ),
                    )
                    .with_block("parameters")];
                }
            }
            // Dose records read the weight too (the individual parameters are
            // rebuilt at every dose event), and a dose-only occasion has no
            // observation to have caught it above.
            for d in 0..subj.doses.len() {
                let time = subj.doses[d].time;
                let w = (weight.eval)(theta, subj.dose_cov(d), time);
                if !(w.is_finite() && w > 0.0) {
                    return vec![Diagnostic::error(
                        "E_KAPPA_WEIGHT_NONPOSITIVE",
                        format!(
                            "[parameters] kappa `{name}` weight `{}` evaluates to {w} at the dose \
                             record of subject `{}`, TIME {time} — it must be strictly positive \
                             and finite. The weight declares κ ~ N(0, Ω_IOV / W) and is applied \
                             as κ/√W, so a zero or missing one divides an individual parameter by \
                             zero. The usual cause is a missing or non-numeric cell in the \
                             covariate it references.",
                            weight.expr, subj.id
                        ),
                    )
                    .with_block("parameters")];
                }
            }
        }
    }
    Vec::new()
}

/// Warning half of the kappa-weight check (#1031): a sample-size weight that is
/// not constant *within* an occasion.
///
/// A warning rather than an error: κ is drawn once per occasion, so a weight
/// that changes underneath it makes the arm's effective variance ambiguous —
/// but the model still evaluates, and a dataset whose arm size legitimately
/// changes mid-occasion (a dropout-adjusted N) may be exactly what the author
/// intends. Observations only; a dose-only occasion has no within-occasion
/// sequence to vary over.
fn check_kappa_weight_variation(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    if !model.has_weighted_kappa() {
        return Vec::new();
    }
    let theta = &model.default_params.theta;
    let mut diags = Vec::new();
    for (k, weight) in model.kappa_weights.iter().enumerate() {
        let Some(weight) = weight else { continue };
        let name = model
            .kappa_names
            .get(k)
            .map(String::as_str)
            .unwrap_or("<kappa>");
        // One variation warning per kappa — the condition is a property of the
        // dataset's shape, so repeating it per subject-occasion is noise.
        'kappa: for subj in &population.subjects {
            for (occ, obs_idx) in crate::stats::likelihood::iov_occasion_groups(subj) {
                let mut seen: Option<f64> = None;
                for &j in &obs_idx {
                    let time = subj.obs_times.get(j).copied().unwrap_or(0.0);
                    let w = (weight.eval)(theta, subj.obs_cov(j), time);
                    match seen {
                        None => seen = Some(w),
                        Some(prev) if (prev - w).abs() > 1e-9 * prev.abs().max(1.0) => {
                            diags.push(
                                Diagnostic::warning(
                                    "W_KAPPA_WEIGHT_VARIES_WITHIN_OCCASION",
                                    format!(
                                        "[parameters] kappa `{name}` weight `{}` is not constant \
                                         within occasion {occ} of subject `{}` ({prev} then {w}). \
                                         κ is drawn once per occasion, so the occasion's \
                                         effective variance Ω_IOV/W is ambiguous; the records are \
                                         each scaled by their own weight.",
                                        weight.expr, subj.id
                                    ),
                                )
                                .with_block("parameters"),
                            );
                            break 'kappa;
                        }
                        Some(_) => {}
                    }
                }
            }
        }
    }
    diags
}

/// Whether a data-reader warning is inapplicable to this model and should not be
/// surfaced.
///
/// The reader is deliberately model-blind, so a few of its findings are about a
/// shape the model cannot have. Shared by `fit()` and `ferx check` so the two
/// cannot disagree about which warnings a user sees.
///
/// Currently one case: `W_NO_DOSES` on a compartment-free model (#811). The reader
/// reads a dose-free dataset as a probable missing `AMT` column; for a model with
/// no compartments that is its normal shape — every model-based meta-analysis
/// dataset is dose-free — so the advice ("check that the dataset has an AMT
/// column") is wrong rather than merely noisy.
pub(crate) fn reader_warning_suppressed(model: &CompiledModel, warning: &str) -> bool {
    model.is_algebraic() && warning.starts_with("W_NO_DOSES")
}

/// The *fatal* model-vs-population checks every `simulate()` entry point owes its
/// caller (#1083).
///
/// `fit()` runs these through [`check_model_data`]; the simulation paths used to
/// each pick their own subset, and the subsets disagreed. That mattered because a
/// weight is the one covariate whose failure mode is silent in both directions:
/// `BinOp::Div` returns `0.0` rather than `inf` when its divisor underflows, so a
/// missing arm size makes `κ/√W` collapse to *zero* — the arm simply loses its
/// between-arm variability, with no `NaN` to trip over — and a missing standard
/// error makes the `weight = <expr>` residual magnitude collapse the additive
/// loading, so the simulated observation *is* its own IPRED. Both produce plots.
///
/// Ordered covariates-first so the first error reported by the existing callers is
/// unchanged.
pub(crate) fn check_simulation_data(
    model: &CompiledModel,
    population: &Population,
) -> Vec<Diagnostic> {
    let mut diags = check_unbound_factor_thetas(model);
    diags.extend(check_covariates(model, population));
    diags.extend(check_kappa_weights(model, population));
    diags.extend(check_theta_gather_indices(model, population));
    diags.extend(check_residual_magnitude(model, population));
    diags
}

/// All data-dependent *fatal* compatibility checks between a compiled model and
/// a dataset, collected into one diagnostic list. Shared by `fit()` (which
/// stops at the first error via [`first_error`]) and `ferx check` (which
/// reports every finding). Check order matches the historical inline order in
/// `fit()` so the first error is unchanged: covariates, scaling, error model,
/// iov occasions.
pub fn check_model_data(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    // Default (no model-side occasion rule): occasions, if any, come from the data.
    check_model_data_rule(model, population, &IovOccasionRule::Column)
}

/// [`check_model_data`] variant aware of the model-side IOV occasion rule, so the
/// `E_IOV_MISSING_OCC` check can be skipped when `iov_occasion` will derive the
/// labels at fit time. Used by `fit()` and the `ferx check` path.
pub fn check_model_data_rule(
    model: &CompiledModel,
    population: &Population,
    iov_rule: &IovOccasionRule,
) -> Vec<Diagnostic> {
    let mut diags = check_finite_observations(population);
    diags.extend(check_covariates(model, population));
    diags.extend(check_per_cmt_scaling(model, population));
    diags.extend(check_per_cmt_error_model(model, population));
    diags.extend(check_residual_magnitude(model, population));
    // Errors only; the within-occasion-variation *warning* is a separate walk,
    // surfaced by `check_model_data_warnings` so `fit()` reports it too (this
    // list is consumed error-first by `first_error`).
    diags.extend(check_kappa_weights(model, population));
    diags.extend(check_theta_gather_indices(model, population));
    diags.extend(check_iov_occasions(model, population, iov_rule));
    diags.extend(check_absorption_dosing(model, population));
    diags.extend(check_modeled_dose_rates(model, population));
    diags.extend(check_dose_compartments(model, population));
    diags.extend(validate_output_columns(model, population));
    diags
}

/// Every scored observation must be a finite number.
///
/// Nothing else on the fit path tests `observations` for finiteness — TIME is
/// checked, DV was not — so a `NaN` reached the likelihood as a silent
/// `NaN` objective, or worse got laundered into a finite value on the way
/// (LTBS case 2 floors `NaN.max(LTBS_FLOOR)` to `LTBS_FLOOR`, since `f64::max`
/// returns the non-NaN operand).
///
/// The concrete way to get here is misuse of
/// [`read_population_for_simulation`](crate::api::read_population_for_simulation)
/// (#957): it keeps a missing `DV` as a design point with a `NaN` placeholder,
/// and its "do not pass this to `fit()`" contract was documentation-only. This
/// makes it loud instead. A hand-built `Population` carrying a `NaN` DV is
/// caught by the same check.
fn check_finite_observations(population: &Population) -> Vec<Diagnostic> {
    let Some(subject) = population
        .subjects
        .iter()
        .find(|s| s.observations.iter().any(|v| !v.is_finite()))
    else {
        return Vec::new();
    };
    let n_bad = subject
        .observations
        .iter()
        .filter(|v| !v.is_finite())
        .count();
    vec![Diagnostic::error(
        "E_NONFINITE_DV",
        format!(
            "Subject '{}' has {} non-finite observation(s) (NaN/Inf). A DV must be a finite \
             number to be scored.",
            subject.id, n_bad
        ),
    )
    .with_suggestion(
        "If this population came from `read_population_for_simulation` (a `DV = .` design \
         template), it is a simulation input, not a fit input — its missing DVs are NaN \
         placeholders for values that have not been generated yet. Read the dataset with \
         `read_population_for` to fit it.",
    )]
}

/// IOV models require occasion labels in the dataset. When `n_kappa > 0` but
/// every subject has an empty `occasions` vector the kappa random effects are
/// silently ignored — catch this early instead.
fn check_iov_occasions(
    model: &CompiledModel,
    population: &Population,
    iov_rule: &IovOccasionRule,
) -> Vec<Diagnostic> {
    if model.n_kappa == 0 {
        return Vec::new();
    }
    // A model-side occasion rule (`iov_occasion = dose | time(...)`) populates the
    // occasion labels at fit time from each subject's timeline, so the dataset need
    // not carry them — the empty-occasions check does not apply.
    if *iov_rule != IovOccasionRule::Column {
        return Vec::new();
    }
    // `all()` on an empty iterator is vacuously true; an empty population is not
    // a missing-OCC problem so skip the check when there are no subjects.
    let all_empty = !population.subjects.is_empty()
        && population.subjects.iter().all(|s| s.occasions.is_empty());
    if !all_empty {
        return Vec::new();
    }
    vec![Diagnostic::error(
        "E_IOV_MISSING_OCC",
        "Model declares kappa (IOV) parameters but no occasion labels were found. Either set \
         `iov_column = \"OCC\"` (or the relevant column name) to read occasions from the dataset, \
         or set `iov_occasion = dose` / `iov_occasion = time(...)` to derive them in the model — \
         both in [fit_options] — so that per-occasion kappas can be estimated.",
    )
    .with_block("fit_options")]
}

/// Derive each subject's IOV occasion labels from a model-side rule
/// ([`IovOccasionRule::PerDose`] or [`IovOccasionRule::TimeWindows`]), writing
/// `Subject::occasions` (parallel to `obs_times`) and `Subject::dose_occasions`
/// (parallel to `doses`). This is the DSL alternative to a dataset occasion
/// column; downstream grouping (`split_obs_by_occasion`) is identical either way.
///
/// When `had_column` is true the model *also* set `iov_column`; the DSL rule wins
/// and a one-time override warning is pushed. A `Column` rule must never reach
/// here (the caller only clones the population when a derived rule is active).
pub(crate) fn apply_iov_occasion_rule(
    population: &mut Population,
    rule: &IovOccasionRule,
    had_column: bool,
    warnings: &mut Vec<String>,
) {
    if had_column {
        warnings.push(
            "iov_occasion is set in [fit_options], so the model-side occasion rule is used and \
             the iov_column dataset column is ignored."
                .to_string(),
        );
    }
    // `Column` supplies no derived labels — nothing to do (the caller only clones
    // the population for a derived rule, so this is a defensive no-op).
    if *rule == IovOccasionRule::Column {
        return;
    }
    // Occasion of a scalar time under a set of strictly-increasing breakpoints:
    // the number of edges at or before `t` (so `time(24,48)` → 0 | 1 | 2).
    let window_occ =
        |edges: &[f64], t: f64| -> u32 { edges.iter().filter(|&&e| e <= t).count() as u32 };
    for s in population.subjects.iter_mut() {
        match rule {
            IovOccasionRule::Column => unreachable!("handled by the early return above"),
            IovOccasionRule::TimeWindows(edges) => {
                // Bucket observations and doses on the *same* internal event clock.
                // For reset-stacked subjects (EVID 3/4) that clock is monotonic and
                // already shifted so each reset segment sits past the previous one,
                // so the breakpoints separate periods correctly. Using the raw user
                // TIME for observations (as an earlier version did) is wrong on two
                // counts: it restarts each period — folding period 2 back into
                // occasion 0 — and it puts observations on a different clock than the
                // doses (whose only stored time is the internal one), so a dose could
                // land in a different occasion than its co-timed samples (#757).
                s.occasions = s.obs_times.iter().map(|&t| window_occ(edges, t)).collect();
                s.dose_occasions = s.doses.iter().map(|d| window_occ(edges, d.time)).collect();
            }
            IovOccasionRule::PerDose => {
                // Each distinct administration *time* opens a new occasion. Doses
                // that share a time (e.g. a simultaneous depot + central bolus, or a
                // split dual-route dose) share one occasion rather than each spawning
                // its own — indexing by dose position would leave the earlier
                // co-timed dose stranded in an empty occasion, since observations can
                // only map to one of them (#757). Doses are stored time-sorted, so a
                // running counter that ticks on each time change is the label.
                let mut dose_occ = Vec::with_capacity(s.doses.len());
                let mut occ = 0u32;
                let mut prev: Option<f64> = None;
                for d in &s.doses {
                    if let Some(p) = prev {
                        if d.time != p {
                            occ += 1;
                        }
                    }
                    dose_occ.push(occ);
                    prev = Some(d.time);
                }
                s.dose_occasions = dose_occ;
                // Distinct dose times, ascending (one per occasion label above).
                let mut dose_times: Vec<f64> = Vec::new();
                for d in &s.doses {
                    if dose_times.last() != Some(&d.time) {
                        dose_times.push(d.time);
                    }
                }
                // Observation occasion = most recent dose at or before its time
                // (dose-before-obs tie-break at equal TIME); pre-first-dose rows fold
                // into occasion 0. obs_times are ascending, so a single forward walk
                // over the distinct dose times is linear rather than O(obs*doses)
                // (#757 cleanup).
                let mut j = 0usize;
                s.occasions = s
                    .obs_times
                    .iter()
                    .map(|&t| {
                        while j < dose_times.len() && dose_times[j] <= t {
                            j += 1;
                        }
                        j.saturating_sub(1) as u32
                    })
                    .collect();
            }
        }
    }
}

/// Built-in absorption input-rate models (e.g. `transit()`) are integrated
/// through the standard ODE prediction paths. Two combinations are not yet
/// supported and are rejected here — loudly — rather than silently mis-modeled:
///   - a steady-state dose (`SS=1`) into an input-rate compartment: periodic
///     steady state with an unfinished `R_in` tail needs dedicated treatment
///     (a later phase of `plans/absorption-models.md`);
///   - an input-rate model combined with a `[diffusion]` block (the SDE/EKF
///     path), whose Kalman propagation does not carry the `R_in` forcing.
pub(crate) fn check_absorption_dosing(
    model: &CompiledModel,
    population: &Population,
) -> Vec<Diagnostic> {
    let Some(ode) = &model.ode_spec else {
        return Vec::new();
    };
    if ode.input_rate.is_empty() {
        return Vec::new();
    }
    let mut diags = Vec::new();

    // input-rate + [diffusion]/EKF (model-level): the EKF propagator does not
    // apply the R_in forcing, so the absorption term would be silently dropped.
    if !ode.diffusion_var.is_empty() {
        diags.push(
            Diagnostic::error(
                "E_ABSORPTION_DIFFUSION",
                "A built-in absorption input-rate model (e.g. transit() or igd()) cannot yet \
                 be combined with a [diffusion] block (the SDE/EKF path): the EKF \
                 propagation does not carry the input-rate forcing. Remove the \
                 [diffusion] block or the absorption term.",
            )
            .with_block("odes"),
        );
    }

    // SS=1 dose into a built-in absorption input-rate compartment (#719, gap 1). Supported
    // for the smooth-density kernels (transit / igd / weibull / first_order): the dose is
    // equilibrated *through* the absorption kernel (a pulse-train trough) and its infinite
    // periodic pulse train is superposed as `R_in` on the forward walk, so predictions match
    // an explicit run-in exactly (`ode::predictions::equilibrate_ss_state` +
    // `add_prepared_input_rate_forcing`). Two combinations stay out of scope and are rejected
    // here rather than silently mis-served:
    //   * zero_order() absorption — its input is a spanning window, not a pointwise density,
    //     so SS needs a periodic-window equilibration (separate follow-up under #719).
    //   * a lagtime on the SS-dosed compartment — the SS+lagtime pre-arrival seed
    //     (`ss_state_at_phase`) is still bolus-only for input-rate compartments.
    use std::collections::BTreeSet;
    let cmts: BTreeSet<usize> = ode.input_rate.iter().map(|f| f.cmt + 1).collect();
    let zero_order_cmts: BTreeSet<usize> = ode
        .input_rate
        .iter()
        .filter(|f| f.kind == crate::pk::absorption::InputRateKind::ZeroOrder)
        .map(|f| f.cmt + 1)
        .collect();
    // `cmt_1based()` throughout so a `CMT=0` dose (the default dose compartment == compartment 1)
    // is scoped like `CMT=1` against these 1-based `f.cmt + 1` sets (#899, #913 review). The raw
    // `d.cmt` this replaced left `CMT=0` bypassing the SS-unsupported gates, so an SS `CMT=0` dose
    // into a zero-order / lagged absorption compartment skipped its clean rejection and fell
    // through to the silent zero-order drop the same review fixed in `zero_order_dur_and_frac`.
    let has_ss_zero_order = population.subjects.iter().any(|s| {
        s.doses
            .iter()
            .any(|d| d.ss && d.ii > 0.0 && zero_order_cmts.contains(&d.cmt_1based()))
    });
    // A per-route lag (`fn(..., lag=L)`, #856) lives on the forcing's `lag_slot`, NOT the
    // compartment-lag machinery, so `has_lagtime_on_cmt` does not see it — but the SS
    // equilibration seed is just as unable to route the dose through a per-route-lagged
    // onset over the previous interval as through a compartment `lagtime`/`ALAG`. Reject
    // the combination the same way, compartment-scoped so a per-route lag on a *different*
    // pathway does not block an SS dose into an un-lagged compartment.
    let route_lag_cmts: BTreeSet<usize> = ode
        .input_rate
        .iter()
        .filter(|f| f.lag_slot.is_some())
        .map(|f| f.cmt + 1)
        .collect();
    // Compartment-scoped, not model-wide (`model.has_lagtime()`): a lag declared on a
    // *different* absorption pathway (e.g. `ALAG2`) must not block an SS dose into an
    // un-lagged compartment (e.g. `CMT=1`). Covers both a compartment `lagtime`/`ALAG`
    // and a per-route `lag=` on a forcing feeding the SS-dosed compartment.
    let has_ss_lag = population.subjects.iter().any(|s| {
        s.doses.iter().any(|d| {
            d.ss && d.ii > 0.0
                && cmts.contains(&d.cmt_1based())
                && (model.has_lagtime_on_cmt(d.cmt_1based())
                    || route_lag_cmts.contains(&d.cmt_1based()))
        })
    });
    if has_ss_zero_order {
        diags.push(
            Diagnostic::error(
                "E_ABSORPTION_SS_ZERO_ORDER",
                "Steady-state dosing (SS=1) into a zero_order() absorption compartment is not \
                 yet supported: unlike the pointwise density kernels (transit/igd/weibull/\
                 first_order), the zero-order input is a spanning window and its periodic \
                 steady state needs a window-based equilibration (a follow-up under #719). \
                 Expand the run-in with explicit dosing records, or use a density absorption \
                 kernel.",
            )
            .with_block("odes"),
        );
    } else if has_ss_lag {
        diags.push(
            Diagnostic::error(
                "E_ABSORPTION_SS_LAG",
                "Steady-state dosing (SS=1) into a built-in absorption compartment combined \
                 with an absorption lagtime (a compartment `lagtime`/`ALAG` or a per-route \
                 `lag=`) is not yet supported: the SS+lagtime pre-arrival seed does not yet \
                 route the dose through the absorption kernel over the previous interval. \
                 Remove the lagtime, expand the run-in with explicit dosing records, or dose \
                 without SS.",
            )
            .with_block("odes"),
        );
    }

    // Infusion (RATE>0) into a built-in absorption input-rate compartment (#719 gap 2). Supported
    // for the smooth-density kernels: the dose is a *zero-order source feeding the kernel*, so
    // `R_in` becomes the convolution `(F·amt/T)·[G(tad) − G(tad − T)]` (`R_in_inf`,
    // `PreparedInputRate::rate_infused`), and the dose's plain `+rate` injection is suppressed
    // (`active_infusions` skips the compartment) so there is no double count. Two combinations
    // stay out of scope and are rejected here rather than silently mis-served:
    //   * an infusion into a zero_order() window — infusing a spanning window is a
    //     window-with-window convolution, a separate follow-up (like the SS case).
    //   * a steady-state infusion (SS=1 + RATE>0) into an absorption compartment — neither the SS
    //     equilibration (bolus-record only) nor the infusion convolution handles the periodic
    //     zero-order-into-kernel steady state yet.
    //
    // Unlike the SS-*bolus* gates above (`has_ss` / `has_ss_zero_order`), this reject is
    // **`II`-independent** — it fires on `d.ss && d.is_infusion()` with no `d.ii > 0.0` condition.
    // An SS bolus with `II ≤ 0` legitimately degrades to a single non-SS bolus (a valid prediction,
    // flagged by the `W_STEADY_STATE_II` warning), so those gates let it through. But an SS infusion
    // is an *unsupported combination*: were it allowed to skip this gate on `II = 0`, it would fall
    // to the plain-infusion branch of `add_prepared_input_rate_forcing` and be silently served as a
    // single non-SS infusion — a different model than the SS one asked for. Rejecting regardless of
    // `II` also keeps this ODE-native gate consistent with the closed-form absorption gate
    // (`check_absorption_closed_form_support`, which rejects `dose.ss && dose.is_infusion()` with no
    // `II` condition), so the same subject errors on both paths instead of one erroring and the
    // other silently mis-serving.
    // `cmt_raw()` here (not `cmt_1based()` like the SS-*bolus* gates above): a `CMT=0`
    // would miss these 1-based `f.cmt + 1` sets, so unlike most raw reads the accessor
    // choice is load-bearing (the only other place it bites is `has_lagtime_on_cmt` below,
    // where it is immaterial — that path is analytical-only and the indexed-lag branch is
    // dead behind `ode_spec.is_some()`). It is safe: both checks are `is_infusion()`-gated, and a
    // meaningful (`AMT>0`) `CMT=0` infusion is already a hard error from
    // `check_dose_compartments` (`E_DOSE_CMT_NOT_INFUSABLE` — `CMT=0` is a default *bolus*
    // compartment, undefined for a zero-order input), so it never reaches a fit regardless
    // of whether this gate also flags it; an `AMT=0` `CMT=0` infusion is inert. So
    // `contains(&d.cmt_raw())` cannot miss a reachable case. (Behaviour-preserving: this
    // was the pre-#912 raw `d.cmt`.)
    let has_ss_infusion = population.subjects.iter().any(|s| {
        s.doses
            .iter()
            .any(|d| d.ss && d.is_infusion() && cmts.contains(&d.cmt_raw()))
    });
    let has_infusion_zero_order = population.subjects.iter().any(|s| {
        s.doses
            .iter()
            .any(|d| d.is_infusion() && zero_order_cmts.contains(&d.cmt_raw()))
    });
    if has_ss_infusion {
        diags.push(
            Diagnostic::error(
                "E_ABSORPTION_SS_INFUSION",
                "A steady-state infusion (SS=1 with RATE>0) into a built-in absorption \
                 compartment is not yet supported: steady-state dosing and infusion-into-kernel \
                 are each supported alone, but their combination (a periodic zero-order source \
                 feeding the absorption kernel) is a follow-up. Use a non-SS infusion, or an SS \
                 bolus.",
            )
            .with_block("odes"),
        );
    } else if has_infusion_zero_order {
        diags.push(
            Diagnostic::error(
                "E_ABSORPTION_RATE_ZERO_ORDER",
                "An infusion (RATE>0) into a zero_order() absorption compartment is not yet \
                 supported: unlike the pointwise density kernels (transit/igd/weibull/\
                 first_order), a zero-order input is itself a spanning window, so infusing it is a \
                 window-with-window convolution (a follow-up). Use a plain bolus into the \
                 zero_order compartment, or a density absorption kernel.",
            )
            .with_block("odes"),
        );
    }

    // Parameter-domain validation (data-level): an out-of-domain or non-finite
    // input-rate parameter (e.g. transit `mtt ≤ 0` / `n < 0`, or igd `mat`/`cv2
    // ≤ 0`) would otherwise propagate as a NaN through the ODE RHS and surface
    // only as an opaque fit failure. Evaluated on typical values (η = 0) per
    // subject, so a covariate relationship that pushes a subject's typical
    // parameter out of range is caught too. Reported once — a single fatal
    // error already halts the fit.
    // Pathway-fraction **value** checks (below): each fraction in (0, 1], and the
    // fractions on a compartment sum to ≈ 1. The companion **structural** rules —
    // every term on a ≥2-pathway compartment must carry a fraction, and a *lone*
    // fractioned term is rejected — are enforced earlier, in the parser's
    // `build_ode_spec` (alongside the at-most-one-zero-order rule, #505), so they
    // also guard the `simulate()` / `predict()` paths that never reach this
    // data-level check (#388, #588). By the time a model reaches here every
    // *fractioned* compartment therefore has ≥2 partitioning terms, so the Σ ≈ 1
    // check below runs unconditionally on any compartment that carries a fraction.
    use std::collections::BTreeMap;

    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    'subjects: for subject in &population.subjects {
        // Typical-value snapshot for an ODE-forcing-rate diagnostic: TIME at the
        // start of the record (t=0).
        let pk = (model.pk_param_fn)(
            &model.default_params.theta,
            &zero_eta,
            &subject.covariates,
            0.0,
        );
        for forcing in &ode.input_rate {
            if let Err(msg) = forcing.validate(&pk.values) {
                diags.push(
                    Diagnostic::error(
                        "E_ABSORPTION_DOMAIN",
                        format!(
                            "Built-in absorption input-rate parameter out of domain at typical \
                             values (subject {}): {msg}. Constrain the parameter so it stays \
                             positive (e.g. a log-normal `P = TVP * exp(ETA_P)` \
                             parameterisation).",
                            subject.id
                        ),
                    )
                    .with_block("odes"),
                );
                break 'subjects;
            }
        }

        // Pathway-fraction value checks (#388, η = 0, per subject so a covariate on a
        // fraction is caught too): each fraction in (0, 1], and the fractions on a
        // compartment sum to ≈ 1. Reported once — a single fatal error halts the fit.
        let mut frac_sum: BTreeMap<usize, f64> = BTreeMap::new();
        for forcing in &ode.input_rate {
            let Some(slot) = forcing.frac_slot else {
                continue;
            };
            let fr = pk.values.get(slot).copied().unwrap_or(1.0);
            if !(fr > 0.0 && fr <= 1.0) {
                diags.push(
                    Diagnostic::error(
                        "E_ABSORPTION_FRACTION",
                        format!(
                            "Pathway fraction out of range at typical values (subject {}): \
                             {fr} is not in (0, 1]. A `FR*fn(...)` multiplier is a \
                             dose-splitting fraction — constrain it to (0, 1] (e.g. a logit \
                             `FR = 1/(1+exp(-X))` parameterisation).",
                            subject.id
                        ),
                    )
                    .with_block("odes"),
                );
                break 'subjects;
            }
            *frac_sum.entry(forcing.cmt).or_insert(0.0) += fr;
        }
        for (&cmt, &sum) in &frac_sum {
            // A lone fractioned term is rejected at parse (`build_ode_spec`), so every
            // compartment reaching `frac_sum` has ≥2 partitioning terms — the Σ ≈ 1
            // check needs no ≥2-pathway gate here (avoids the misleading "sum to <FR>,
            // not 1" a lone term would otherwise trigger; #388 review #1, #588).
            if (sum - 1.0).abs() > 1e-4 {
                diags.push(
                    Diagnostic::error(
                        "E_ABSORPTION_FRACTION",
                        format!(
                            "Pathway fractions on compartment {} sum to {sum} at typical values \
                             (subject {}), not 1. Declare fractions that partition the dose — \
                             e.g. a parameter `FR` and a complementary `FR2 = 1 - FR`.",
                            cmt + 1,
                            subject.id
                        ),
                    )
                    .with_block("odes"),
                );
                break 'subjects;
            }
        }
    }

    diags
}

/// NONMEM coded `RATE=-2` (modeled infusion duration → `D{cmt}`) needs a
/// model-aware check the datareader cannot make (it has no model). It is fatal
/// — never a silent fall-through to a bolus (the original #324 bug):
///   - **Matching `D{cmt}` parameter.** A `RATE=-2` dose into compartment `n`
///     requires a `D{n}` parameter (so `resolve_rate` has a slot to read);
///     otherwise it is rejected. Supported on both engines: ODE models record
///     the slot on `ode_spec.dose_attr_map`, analytical models (#394) on
///     `model.dose_attr_map`.
///
///   - **`D{cmt}`/`R{cmt}` not also used as an ordinary parameter.** Since #993 the
///     name collision *is* a runtime check rather than a docs note: if the model
///     reads the same parameter on its prediction path (`[odes]` RHS / `[scaling]`
///     readout), this dose is asking one value to be both the modeled infusion
///     {duration,rate} and a model term, so it is rejected
///     (`E_DOSE_ATTR_DOUBLE_USE`). It belongs here rather than in the parser
///     precisely because it is data-dependent: with no coded-`RATE` dose the
///     attribute is inert and an `R1` that is really a rate constant is a correct
///     model. The unconditional siblings `F{n}`/`ALAG{n}` (and bare `F`/`LAGTIME`)
///     apply to *every* dose, so the parser rejects those outright — see
///     `model_parser::check_dose_attr_double_use`.
///
/// Reported once per offending compartment (naming the first dose that hits it),
/// so a dataset with many `RATE=-2` rows yields one actionable error per cause.
/// The two checks are mutually exclusive per `(attr, cmt)`: a missing parameter
/// cannot also be a double use.
pub(crate) fn check_modeled_dose_rates(
    model: &CompiledModel,
    population: &Population,
) -> Vec<Diagnostic> {
    use crate::types::{DoseAttr, RateMode};
    let mut diags = Vec::new();
    // De-dup by (attribute, compartment) so N identical coded-RATE rows give one
    // error, not N — and a `D{cmt}` and an `R{cmt}` dose into the same compartment
    // are reported independently rather than masking each other.
    let mut reported: std::collections::HashSet<(DoseAttr, usize)> =
        std::collections::HashSet::new();
    for subject in &population.subjects {
        for dose in &subject.doses {
            // Map the modeled mode to its dose attribute + NONMEM-faithful naming
            // (the `RATE` code, the noun, the `$PK` parameter letter, and the
            // engine-agnostic error code). `Fixed` doses need no slot.
            let (attr, code, kind, err_code, param) = match dose.rate_mode {
                RateMode::Fixed => continue,
                RateMode::ModeledDuration => (
                    DoseAttr::Duration,
                    "-2",
                    "duration",
                    "E_MODELED_DURATION_NO_PARAM",
                    "D",
                ),
                RateMode::ModeledRate => {
                    (DoseAttr::Rate, "-1", "rate", "E_MODELED_RATE_NO_PARAM", "R")
                }
            };
            if !reported.insert((attr, dose.cmt_raw())) {
                continue;
            }
            let cmt = dose.cmt_raw();
            // A coded-`RATE` dose into compartment `cmt` requires a matching
            // `D{cmt}`/`R{cmt}` parameter so `resolve_rate` has a slot to read — for
            // BOTH engines. `active_dose_attr_map()` returns the engine-correct map
            // (the `OdeSpec`'s for ODE models, the analytical field otherwise,
            // #324), so an absent slot is the same actionable error on either engine.
            let has_slot = model
                .active_dose_attr_map()
                .indexed_slot(attr, cmt)
                .is_some();
            if !has_slot {
                diags.push(
                    Diagnostic::error(
                        err_code,
                        format!(
                            "subject {}, time {}: RATE={code} (modeled infusion {kind}) into \
                             compartment {cmt} requires a `{param}{cmt}` parameter in \
                             [individual_parameters], but none is declared. Add \
                             `{param}{cmt} = ...` (the modeled {kind}), or supply an explicit \
                             positive RATE.",
                            subject.id, dose.time
                        ),
                    )
                    .with_block("individual_parameters"),
                );
            } else if let Some(name) = model.active_dose_attr_map().prediction_path_read(attr, cmt)
            {
                // #993, the data-gated half. The parameter exists (so the check above
                // passed) but the model *also* reads it on the prediction path — the
                // `[odes]` RHS or the `[scaling]` readout. This dose codes `RATE`, so
                // the engine is using it as the infusion {kind} at the same time: one
                // value, two roles, and neither the model nor the data says which was
                // meant. Measured driving the infusion while simultaneously being read
                // in the RHS. Unlike `F{n}`/`ALAG{n}` — rejected by the parser, since
                // they apply to every dose — this can only be judged here: with no
                // coded-`RATE` dose in the dataset the same model is perfectly correct
                // and `{name}` is an ordinary rate constant.
                diags.push(
                    Diagnostic::error(
                        "E_DOSE_ATTR_DOUBLE_USE",
                        format!(
                            "subject {}, time {}: RATE={code} (modeled infusion {kind}) into \
                             compartment {cmt} makes `{name}` the {kind} of this dose, but \
                             `{name}` is also read on the model's prediction path ([odes] RHS \
                             or [scaling]), so its value is used twice. Rename the parameter if \
                             it is an ordinary rate constant — `{param}{cmt}` is a reserved \
                             dose-attribute name — or remove the read if it really is the \
                             modeled {kind}.",
                            subject.id, dose.time
                        ),
                    )
                    .with_block("individual_parameters"),
                );
            }
        }
    }
    diags
}

/// Whether **anything in this dataset** asks the analytical PK predictor for a
/// value — any Gaussian observation, or any `EVID=2` PK-only request.
///
/// This exists for one narrow case: a pure-TTE / pure-discrete model still needs
/// a `[structural_model]` line, and the idiom is a **placeholder** `one_cpt_iv`
/// with dummy `CL`/`V` (the parser supplies exactly that when the block is
/// absent, `model_parser.rs`). Validating dose compartments against that
/// placeholder's 1-compartment topology would reject a working model over dose
/// records no PK code ever reads.
///
/// Deliberately a **population**-level gate on *whether the dose-compartment check
/// runs at all*, not a per-subject one. If **any** subject asks the analytical
/// predictor for a value, the check validates **every** dose in the dataset — so a
/// TTE-only subject *of a joint PK-TTE model* (an early dropout, or a TTE-only arm,
/// that still has dose records) has its unroutable infusion caught up front, rather
/// than carried into the event-driven walk's `panic!`. Making *validation* per
/// subject would reopen exactly that.
///
/// This is distinct from — and complementary to — the per-subject predictor gate
/// (`subject_feeds_analytical_pk`, #905): validation is population-scoped so a live
/// population is fully checked, while the predictor declines the walk per subject for
/// any subject whose output is unused. In a live population the check already rejects
/// a bad dose, so the predictor gate never has to; in an exempt population the check
/// is skipped and the predictor gate is what keeps the walk unreached.
///
/// The per-subject truth (which observation CMTs are Gaussian, and the
/// no-`survival`-feature short circuit) lives in [`crate::pk::subject_feeds_analytical_pk`];
/// this is `any(...)` of it.
fn population_uses_analytical_pk(model: &CompiledModel, population: &Population) -> bool {
    // Single source of truth with the predictor gate: `subject_feeds_analytical_pk`
    // is the per-subject predicate, and this is `any(...)` of it. The two compose —
    // this population-level test decides whether the dose-compartment check runs at
    // all (a *live* population validates every dose, including a TTE-only subject's),
    // while the per-subject gate declines the walk for a subject the analytical
    // predictor would feed nothing (#905). See `subject_feeds_analytical_pk` for why
    // "no Gaussian observation" implies "predictor output unused" on an analytical
    // model.
    population
        .subjects
        .iter()
        .any(|s| crate::pk::subject_feeds_analytical_pk(model, s))
}

/// Every dose must name a compartment the **analytical** engine can actually
/// deliver it into (#375). The datareader cannot make this call — it has no
/// model — and the parse-time infusable check
/// (`model_parser`, [`PkModel::infusable_compartments`]) only fires for a
/// *declared* `D{cmt}`/`R{cmt}`, so an explicit positive `RATE` (or an
/// out-of-range bolus `CMT`) read straight from the dataset reached the
/// predictors unvalidated. The three analytical paths then disagreed on it:
/// the dose-superposition path never reads `dose.cmt` and silently routed the
/// infusion into central, the event-driven walk `panic!`ed, and the `sens`
/// gradient walk silently dropped the infusion. Rejecting up front removes that
/// disagreement **for the doses it rejects**, and keeps the prediction hot path
/// panic-free.
///
/// This check alone does **not** make the paths agree — it only removes the
/// doses that have no target at all. Agreement for the doses it *accepts* comes
/// from [`crate::pk::dose_needs_event_walk`], which reroutes every dose the
/// superposition dispatch cannot place (it never reads `dose.cmt`, so it would
/// compute the dose as going into the model's default route — the depot for an
/// oral model, central for an IV one) to the event-driven walk, which places it
/// correctly and is NONMEM-anchored. The two are complements and must be read
/// together: this one rejects the unroutable, that one routes the rest.
///
/// Two rules, both 1-based like NONMEM's `CMT`:
///   - **Range — every dose, whatever its amount.** `cmt` must not exceed the
///     model's state count. This is deliberately *not* skipped for `AMT=0`:
///     both walks bound-check the compartment before the amount is read, so an
///     inert row with an out-of-range `CMT` panics exactly like a real dose.
///     `CMT=0` is left alone — every path treats it as NONMEM's "default dose
///     compartment" (the walk via `saturating_sub(1)`, `dose_needs_event_walk`
///     by mapping it to 1, the superposition path by ignoring `cmt`), so it is
///     consistent, not a silent mis-route.
///   - **Routing — infusions that actually deliver.** The compartment must be
///     in [`PkModel::infusable_compartments`] — central for every model, the
///     oral depot since #400, the peripheral(s) of the 2-/3-cpt IV models, and
///     — since #375 — the oral models' peripheral(s) too (the oral propagators
///     reuse the IV forced response; nothing flows back into the depot). That
///     makes the set `1..=n_states` for all six walk-supported models, so with
///     the range rule claimed first the only value this rule can still reject
///     is `CMT=0`, which has no meaning for a zero-order input. A zero-amount
///     infusion is exempt (`duration = AMT/RATE = 0`, so no path opens a rate
///     window). Checked only for the six event-walk models: `dose_channel` *is*
///     the walk's routing table, and a transit/IG subject never takes that walk
///     — an infusion reroutes it to the model's ODE twin
///     (`CompiledModel::effective_for`, #719 gap 2), which addresses
///     compartments directly. Applying this table to them would reject a
///     supported model.
///
/// **ODE models take the range rule but not the routing rule** (#899).
/// `ode/predictions.rs` indexes the state vector directly, so every compartment
/// the `[odes]` block declares is addressable and the `infusable_compartments`
/// table has no ODE analogue — but "declares" is the operative word. A `CMT`
/// past the end of the declared state vector used to fall through that engine's
/// `if cmt_idx < n { … }` guards and be **silently dropped**: the fit converged,
/// reported a finite OFV, and quietly ignored the dose. So the range rule runs
/// for both engines, reading `OdeSpec::n_states` instead of a `PkModel`
/// topology, and names the declared `state_names` in the message. `CMT=0`
/// splits the same way on both engines: accepted for a **bolus** (it is the
/// default dose compartment, which every dose site resolves to compartment 1)
/// and rejected for an **infusion** (a zero-order input has no such default).
///
/// Reported once per `(kind, compartment)` so a dataset with many offending
/// rows yields one actionable error per cause, matching
/// [`check_modeled_dose_rates`].
pub(crate) fn check_dose_compartments(
    model: &CompiledModel,
    population: &Population,
) -> Vec<Diagnostic> {
    let pk_model = model.pk_model;
    let topology = pk_model.topology();
    let name = pk_model.canonical_name();
    // Range-rule input, per engine (#899). The ODE engine indexes its state
    // vector directly, so its bound is the `[odes]` block's declared state
    // count — a `PkModel` topology says nothing about it (`model.pk_model` is
    // still populated on an ODE model, and is *not* the model being solved).
    // The routing rule below stays analytical-only.
    let ode_spec = model.ode_spec.as_ref();
    let n_states = match ode_spec {
        Some(spec) => spec.n_states,
        None => topology.state_layout().0,
    };
    let mut diags = Vec::new();
    // De-dup by (is_infusion, compartment): an infusion and a bolus into the
    // same bad compartment are independent causes and are reported separately.
    let mut reported: std::collections::HashSet<(bool, usize)> = std::collections::HashSet::new();
    // Nothing in this dataset asks the analytical predictor for a value — the
    // `pk` line is a pure-TTE / pure-discrete model's placeholder. See
    // `population_uses_analytical_pk` for why this is a population-level test.
    //
    // **This exemption is safe because the predictor declines in lockstep.** Skipping
    // the dose-compartment check would be a landmine on its own: `dose_needs_event_walk`
    // still reroutes a `CMT ≥ 2` dose onto the event-driven walk, whose out-of-range
    // guard is a `panic!`. That was #905 — the check declined to validate a dose the
    // walk then aborted on. The fix is the shared predicate: the same
    // `subject_feeds_analytical_pk` that makes this population exempt also makes
    // `compute_predictions_with_tv_*` (value) and `subject_routes_to_event_walk`
    // (gradient) return a NaN/decline for every one of its subjects, so the walk is
    // never reached. The exemption's safety no longer rests on the far-away
    // `n_obs == 0` early return in a different module; it is the negation, per subject,
    // of this very predicate.
    //
    // **Analytical models only.** An `[odes]` block is never a placeholder: a
    // hazard that reads an ODE state has the system integrated for it even when
    // the dataset carries no PK observation at all, so the exemption that is
    // sound for a dummy `pk` line would re-open the silent drop for exactly the
    // TTE-only-plus-ODE-PK model where it is hardest to notice (#899).
    if ode_spec.is_none() && !population_uses_analytical_pk(model, population) {
        return diags;
    }
    for subject in &population.subjects {
        for dose in &subject.doses {
            let is_infusion = dose.is_infusion();
            let cmt = dose.cmt_raw();
            // **Range rule — every dose, whatever its amount.** Both walks
            // bound-check the compartment *before* the amount is read
            // (`event_driven::…impl`'s `if cmt_idx < n_states { … } else { panic! }`
            // and its `sens::propagate` twin), so an `AMT=0` row with an
            // out-of-range `CMT` reaches the panic exactly like a real dose —
            // and since #375 routes any `cmt != 1` bolus to the walk, it now
            // gets there. Exempting zero-amount doses here (as an earlier
            // revision did, on the incorrect premise that the bolus branch just
            // "adds `F·0`") re-opened that panic for a NONMEM-idiomatic
            // `EVID=4, AMT=0` reset row carrying a stale `CMT`.
            //
            // Transit/IG skip the routing rule below but are still 2-/3-state
            // models, so this also stops an out-of-range infusion on them from
            // passing validation and being silently dropped by the ODE twin's
            // bounds check.
            if cmt > n_states {
                if reported.insert((is_infusion, cmt)) {
                    // Same code and shape on both engines; only the source of
                    // the compartment list differs, and the block named in the
                    // message is the one the user would edit to add a state.
                    let diag = match ode_spec {
                        Some(spec) => Diagnostic::error(
                            "E_DOSE_CMT_OUT_OF_RANGE",
                            format!(
                                "subject {}, time {}: dose into compartment {cmt}, but the \
                                 `[odes]` block declares only {n_states} state(s) \
                                 (CMT is 1-based: {:?}).",
                                subject.id, dose.time, spec.state_names,
                            ),
                        )
                        .with_block("odes"),
                        None => Diagnostic::error(
                            "E_DOSE_CMT_OUT_OF_RANGE",
                            format!(
                                "subject {}, time {}: dose into compartment {cmt}, but the \
                                 analytical `{name}` model has only {n_states} compartment(s) \
                                 (CMT is 1-based: {:?}).",
                                subject.id, dose.time, topology.compartment_names,
                            ),
                        )
                        .with_block("structural_model"),
                    };
                    diags.push(diag);
                }
                continue;
            }
            // **Routing rule — infusions that actually deliver.** Unlike the
            // range rule, the zero-amount exemption is sound here: an `AMT=0`
            // infusion has `duration = AMT/RATE = 0`, and the walk's infusion
            // branch requires `duration > 0`, so no path ever opens a rate
            // window for it. Rejecting a whole fit over an inert row would be
            // stricter than the failure being fixed. (`is_infusion()` is
            // `rate > 0 || !is_fixed()`, deliberately broader than the walk's
            // `rate > 0 && duration > 0`; this is where the two reconcile.)
            if !is_infusion || dose.amt == 0.0 {
                continue;
            }
            // **ODE engine (#899): the `CMT=0` half of the routing rule carries
            // over, the infusable-subset half does not.** An in-range ODE dose is
            // delivered by indexing the state vector, or — for a compartment fed
            // by a built-in absorption function — as an `R_in` forcing over time
            // (`input_rate_consumes_cmt`, a legitimate skip of the bolus branch,
            // not a drop). Every declared state therefore accepts both a bolus
            // and a zero-order input, and there is no ODE analogue of
            // `PkModel::infusable_compartments` to consult. `CMT=0` is the one
            // value that still has no target: it is NONMEM's default dose
            // *compartment*, defined for a bolus (where both engines resolve it
            // to compartment 1) but not for a zero-order input. The analytical
            // engine rejects it, so rejecting it here keeps the two engines'
            // answer to the same dataset identical — the whole point of #899.
            if let Some(spec) = ode_spec {
                if cmt == 0 && reported.insert((is_infusion, cmt)) {
                    diags.push(
                        Diagnostic::error(
                            "E_DOSE_CMT_NOT_INFUSABLE",
                            format!(
                                "subject {}, time {}: infusion into compartment 0. `CMT=0` is \
                                 NONMEM's *default dose compartment*, which is defined for a \
                                 bolus but not for a zero-order input — name the target \
                                 compartment explicitly (1-based: {:?}).",
                                subject.id, dose.time, spec.state_names,
                            ),
                        )
                        .with_block("odes"),
                    );
                }
                continue;
            }
            // Transit/IG (`event_walk_supported == false`) never take the walk
            // this routing table belongs to — an infusion reroutes them to their
            // ODE twin (#719 gap 2), which addresses compartments directly.
            // Rejecting here would break a supported model.
            //
            // Caveat, tracked as #910: that twin does *not* preserve this model's
            // `CMT` numbering — it drops the depot (absorption becomes a forcing
            // term), so its state count is one lower and every compartment shifts
            // down by one. `CMT=2` on a `one_cpt_transit` therefore passes the
            // range rule above (2 <= n_states 2) and is out of range on the twin
            // that actually solves it. The range rule cannot catch it: the
            // primary carries `ode_spec: None`, and the twin's spec is built
            // lazily inside `effective_for`, after validation has run.
            if !topology.event_walk_supported {
                continue;
            }
            // For a walk-supported model `infusable` *equals* the walk's routing
            // table (pinned by `topology_fields_are_internally_consistent`), so
            // this is exactly the set `dose_channel` can route. Since #375 that
            // set is `1..=n_states` for all six of them, and `cmt > n_states` is
            // already claimed above — so the only value that reaches this arm is
            // `CMT=0`, which has no meaning for an infusion (see the message).
            if !pk_model.infusable_compartments().contains(&cmt)
                && reported.insert((is_infusion, cmt))
            {
                diags.push(
                    Diagnostic::error(
                        "E_DOSE_CMT_NOT_INFUSABLE",
                        format!(
                            "subject {}, time {}: infusion into compartment {cmt}, but the \
                             analytical `{name}` model can only infuse into compartment(s) \
                             {:?}. `CMT=0` is NONMEM's *default dose compartment*, which is \
                             defined for a bolus but not for a zero-order input — name the \
                             target compartment explicitly (1-based: {:?}).",
                            subject.id,
                            dose.time,
                            pk_model.infusable_compartments(),
                            topology.compartment_names,
                        ),
                    )
                    .with_block("structural_model"),
                );
            }
        }
    }
    diags
}

/// Precondition shared by [`predict`] and the `simulate*` family: every dose
/// must name a compartment the analytical engine can deliver it into (#375) —
/// the `predict()`/`simulate()` twin of [`check_dose_compartments`], which
/// `fit()` reaches through [`check_model_data`]. Without it these entry points
/// reach the event-driven walk's routing `match` with an unroutable
/// compartment, which used to `panic!` from deep inside the prediction loop
/// with no subject/time context.
pub(crate) fn assert_dose_compartments_supported(model: &CompiledModel, population: &Population) {
    panic_if_unsupported(
        first_error(&check_dose_compartments(model, population)).err(),
        "a dose into a compartment the model cannot deliver into",
    );
}

/// Precondition shared by [`predict`] and the `simulate*` family: every
/// modeled-`RATE` dose (#324: `RATE=-2` → `D{cmt}` duration, `RATE=-1` → `R{cmt}`
/// rate) must be supported by the model — the matching `D{cmt}`/`R{cmt}`
/// parameter exists and infuses an infusable compartment, on either engine.
///
/// `fit()` enforces this via [`first_error`] over the full [`check_model_data`],
/// but `predict()` / `simulate()` deliberately skip that data-check (they assume
/// a model the caller already validated, and run no other data validation). A
/// modeled dose slipping through would otherwise hit one of two failure modes
/// downstream that the per-path `debug_assert!` tripwires only catch in
/// debug/test builds — silently in release: a 0-rate "infusion" on the
/// analytical path, or [`DoseEvent::resolve_rate`]'s slot `.expect`. This gate
/// turns both into a loud, actionable panic carrying the same diagnostic message
/// `check_model_data` would have produced, reusing the single-source-of-truth
/// [`check_modeled_dose_rates`]. It is O(doses) and runs once per public call
/// (not in the inner loop), and is a no-op for the common all-`Fixed` dataset.
pub(crate) fn assert_modeled_doses_supported(model: &CompiledModel, population: &Population) {
    if let Err(msg) = first_error(&check_modeled_dose_rates(model, population)) {
        panic!(
            "predict()/simulate() received a dose the model cannot honour: {msg}\n\
             (fit() reports this as an error rather than panicking; validate with \
             `check_model_data` before predicting on untrusted input.)"
        );
    }
}

/// Panic when the data does not carry every covariate the model references, for
/// the `Vec`-returning [`predict()`](crate::predict) path (issue #1028).
///
/// `fit()` runs [`check_covariates`] through [`check_model_data`] and `simulate()`
/// calls it directly, so both already refuse a model whose identifiers don't bind
/// to a data column. `predict()` ran no data check at all, which is what let an
/// undefined name in `[scaling]` reach the predictor: the parser classifies any
/// identifier it cannot bind to a theta / eta / individual parameter / state as a
/// covariate, and a covariate missing from the data resolves to the map's `0.0`
/// default — so `y[CMT=1] = A * TOTALLY_UNDEFINED_NAME` returned an all-zero
/// structural prediction with no diagnostic. This closes that gap so `predict()`
/// fails as loudly as `fit()` does, on the same message.
pub(crate) fn assert_covariates_present(model: &CompiledModel, population: &Population) {
    panic_if_unsupported(
        first_error(&check_covariates(model, population)).err(),
        "a model referencing covariates the data does not carry",
    );
}

/// Shared check→`panic!` wrapper for the non-`fit` entry points (`predict()`/
/// `simulate()`): if the sibling `check_*` produced a message, fail loudly with
/// the common template naming `what` the engine received. `fit()` surfaces the
/// same conditions as an `Err` instead of panicking. Callers holding a
/// `Result`-returning check pass `first_error(&check).err()`.
fn panic_if_unsupported(result: Option<String>, what: &str) {
    if let Some(msg) = result {
        panic!(
            "predict()/simulate() received {what}: {msg}\n\
             (fit() reports this as an error rather than panicking.)"
        );
    }
}

/// The short parenthetical naming *why* a closed-form absorption model has no ODE twin, for the
/// twin-less rejection messages below.
///
/// Two shapes reach those messages. The desugar can decline by name before reconstructing a
/// source (a user `[odes]` / `[scaling]` / `[initial_conditions]` block, a reserved-slot shadow
/// or collision, a dose-attribute parameter in a disposition role) — "an unrecognised closed
/// form" describes that. Or the twin *was* reconstructed and its own parse rejected it (#1008),
/// which the model records as a `W_ABSORPTION_TWIN_DECLINED` parse warning. Both land on
/// `absorption_ode_equivalent.is_none()`, so without this the second case is told it is an
/// unrecognised form and pointed at a rewrite it does not need.
fn twin_less_cause(model: &CompiledModel) -> &'static str {
    if crate::types::absorption_twin_decline_reason(model).is_some() {
        "(its twin was built and declined, #1008)"
    } else {
        "(an unrecognised closed form)"
    }
}

/// The clause appended to a twin-less rejection when the twin was declined at parse time
/// (#1008), quoting the twin parser's own reason; empty otherwise. Pairs with
/// [`twin_less_cause`], which names the cause — this one carries only the reason, so the two
/// do not repeat each other.
///
/// This is the only place the reason reaches a user on the reject path. `fit()` copies
/// `model.parse_warnings` into the result at `api::fit`, but every call site of
/// [`check_absorption_closed_form_support`] / [`check_absorption_flip_flop_no_twin`] returns
/// `Err` (or panics, on `predict`/`simulate`) *before* that — so the warning that names the
/// colliding parameter is otherwise visible only through `ferx check`.
fn twin_decline_clause(model: &CompiledModel) -> String {
    match crate::types::absorption_twin_decline_reason(model) {
        Some(reason) => format!(
            " Note: this model is not outside the rewrite's scope — its twin was built and then \
             rejected by its own parse: {reason} Fixing that restores the twin and this feature, \
             and is usually easier than the rewrite suggested above."
        ),
        None => String::new(),
    }
}

/// Features the analytic closed-form absorption models — transit (`one_cpt_transit`,
/// `two_cpt_transit`, #386) and inverse-Gaussian (`one_cpt_ig`, `two_cpt_ig`, #790) —
/// do not support: the exponential-tilting closed form is a constant-parameter
/// depot-bolus superposition, so steady-state doses, infusion doses, system
/// resets, and non-depot (`CMT≠1`) doses are rejected up front — otherwise they would
/// silently mis-predict or hit an `unreachable!` in the superposition dispatch.
/// (Time-varying covariates, a `TIME`-dependent parameter, and IOV (`n_kappa > 0`, #719)
/// are instead transparently rerouted to the plain form's ODE twin — `transit()` / `igd()`
/// forcing — via [`CompiledModel::effective_for`]: the twin integrates the cross-occasion
/// dose carryover (#104/#663) the superposition cannot. Only a form outside that twin's
/// scope rejects them.) `fit()` surfaces this as an `Err`; `predict()`/`simulate()` panic via
/// [`assert_absorption_closed_form_support`], mirroring
/// [`assert_modeled_doses_supported`]. Returns the first offending feature's message,
/// or `None` when compatible.
pub(crate) fn check_absorption_closed_form_support(
    model: &CompiledModel,
    population: &Population,
) -> Option<String> {
    if !matches!(
        model.pk_model,
        PkModel::OneCptTransit | PkModel::TwoCptTransit | PkModel::OneCptIg | PkModel::TwoCptIg
    ) {
        return None;
    }
    let name = model.pk_model.canonical_name();
    // Why this model has no twin, and (for a #1008 decline) the twin parser's own reason. Every
    // message below is reached under `absorption_ode_equivalent.is_none()`, which covers both
    // "the desugar declined by name" and "the twin was built and rejected" — see
    // `twin_decline_clause`.
    let no_twin_cause = twin_less_cause(model);
    let decline = twin_decline_clause(model);
    // IOV (n_kappa > 0): the closed-form superposition cannot express cross-occasion dose
    // carryover (#104) — a dose whose drug persists into a later occasion must decay with that
    // occasion's disposition. The plain form carries an `absorption_ode_equivalent` (the
    // `transit()`/`igd()` forcing twin) that `effective_for` routes IOV subjects to (it
    // integrates the carryover exactly, #663), so it is NOT rejected. Only a twin-less form is
    // rejected here rather than mis-fit — either one the desugar declines by name (a user
    // `[odes]`/`[scaling]`/`[initial_conditions]` block, a role shadowing a reserved F/lagtime
    // slot) or one whose twin was built and rejected by its own parse (#1008); the cause set is
    // open-ended, which is why `no_twin_cause`/`decline` name it instead of the message text.
    if model.n_kappa > 0 && model.absorption_ode_equivalent.is_none() {
        return Some(format!(
            "{name} does not support IOV (n_kappa > 0) in this form: the analytic absorption \
             closed form assumes constant disposition over each absorption window, and this \
             model has no ODE twin {no_twin_cause} to reroute to. Write the model as an ODE \
             transit()/igd() forcing in [odes] directly.{decline}"
        ));
    }
    // A `TIME`-built-in structural parameter makes the disposition switch mid-profile — the
    // closed form assumes constant parameters over each absorption window, so it cannot serve
    // it. The plain form carries an `absorption_ode_equivalent` (built at parse time), which
    // the runtime dispatch routes such subjects to, so it is NOT rejected. Only a twin-less form
    // is rejected here rather than mis-predict — outside the desugar's scope (a custom
    // `[scaling]`, a reserved-slot shadow), or declined because its twin failed to parse (#1008).
    if crate::parser::model_parser::compiled_model_uses_time_builtin(model)
        && model.absorption_ode_equivalent.is_none()
    {
        return Some(format!(
            "{name} does not support a TIME-dependent structural parameter in this form: \
             the analytic absorption closed form assumes constant parameters over each \
             absorption window, and this model has no ODE twin {no_twin_cause} to reroute to. \
             Write the model as an ODE transit()/igd() forcing in [odes] directly.{decline}"
        ));
    }
    for subject in &population.subjects {
        // Time-varying covariates make the disposition switch mid-absorption, which the
        // closed form cannot serve. The plain form's `absorption_ode_equivalent` handles it
        // (the runtime dispatch routes TV-cov subjects there), so reject only the forms that
        // carry no equivalent — out of the desugar's scope, or declined at parse time (#1008).
        if subject.has_tv_covariates() && model.absorption_ode_equivalent.is_none() {
            return Some(format!(
                "{name} does not support within-subject time-varying covariates \
                 (subject {}): the analytic absorption closed form assumes constant parameters \
                 over each absorption window, and this model has no ODE twin {no_twin_cause} to \
                 reroute to. Use an ODE absorption model.{decline}",
                subject.id
            ));
        }
        if subject.has_resets() {
            return Some(format!(
                "{name} does not support system resets (EVID=3/4) (subject {}): resets zero \
                 compartment amounts mid-profile, which the absorption superposition closed form \
                 cannot express (it silently ignores the reset while the sensitivity path washes \
                 out pre-reset doses, so fit() and predict() would disagree). Use an ODE \
                 absorption model.",
                subject.id
            ));
        }
        for dose in &subject.doses {
            if dose.cmt_1based() != 1 {
                // The closed form folds *every* dose through the absorption chain into
                // central ignoring `dose.cmt` (the `sens/provider.rs` superposition loop),
                // while the ODE twin honours the compartment — a `CMT>1` dose there falls to
                // the direct-bolus branch and lands in a disposition compartment, bypassing the
                // transit/IG absorption the model asks for. Neither reading of a non-depot dose
                // on an absorption model is well-defined: a subject on the closed form and one
                // rerouted to the twin (TV-cov / `TIME` / IOV `n_kappa > 0`, #719) would
                // disagree, and the twin's direct-bolus reading is not the intended transit
                // input either. A closed-form absorption model only supports dosing the depot;
                // reject anything else up front. `CMT=0` is NONMEM's *default dose compartment*
                // — compartment 1, the depot — so it is accepted (both the closed form and its
                // twin resolve it exactly like `CMT=1` via `saturating_sub(1)`, #899); only
                // `CMT>=2` is a genuine non-depot target. `cmt_1based()` maps `CMT=0 -> 1`.
                return Some(format!(
                    "{name} does not support dosing into a non-depot compartment (subject \
                     {}, CMT={}): the analytic absorption closed form routes every dose through \
                     the absorption process into central regardless of compartment, so a CMT≠1 \
                     dose would be silently mistreated — and its ODE twin would instead bolus it \
                     into a disposition compartment, so the two paths would disagree. Dose the \
                     depot (CMT=1), or use an ODE absorption model for direct central/peripheral \
                     dosing.",
                    subject.id,
                    dose.cmt_raw()
                ));
            }
            if dose.ss && dose.is_infusion() {
                // A steady-state *infusion* into a closed-form absorption model has no route:
                // the twin serves SS (bolus) and infusion each alone, but not their combination
                // (a periodic zero-order source feeding the kernel), so reject rather than reroute
                // into a case the twin mis-handles (same scope as the ODE-path
                // `E_ABSORPTION_SS_INFUSION`).
                return Some(format!(
                    "{name} does not support a steady-state infusion (SS=1 with RATE>0) into an \
                     absorption compartment (subject {}): steady-state dosing and infusion are \
                     each supported alone (rerouted to the ODE twin), but their combination is a \
                     follow-up. Use a non-SS infusion, or an SS bolus.",
                    subject.id
                ));
            }
            if dose.ss {
                // Steady-state on a closed-form absorption model reroutes to its ODE twin
                // (#719, like TV-cov/TIME/IOV above): the twin serves SS through the absorption
                // kernel (a pulse-train equilibration trough + periodic forward `R_in`), so the
                // closed form need not carry a periodic-sum SS form itself. Reject only what the
                // twin can't yet serve: a twin-less form, or a model with an absorption lagtime
                // (the twin's SS+lagtime pre-arrival seed is still bolus-only — same scope as the
                // ODE-path `E_ABSORPTION_SS_LAG`). `effective_for` performs the reroute.
                if model.absorption_ode_equivalent.is_none()
                    || model.has_lagtime_on_cmt(dose.cmt_raw())
                {
                    return Some(format!(
                        "{name} does not support steady-state (SS) doses in this form (subject \
                         {}): SS reroutes to the ODE absorption twin, but this model has no twin \
                         {}. Use a non-SS multiple-dose schedule, or write the model as an ODE \
                         transit()/igd() forcing in [odes].{decline}",
                        subject.id,
                        if model.has_lagtime_on_cmt(dose.cmt_raw()) {
                            // The lagtime case is a twin-*carrying* scope limit, so it names the
                            // combination rather than the twin-less cause.
                            "for the SS + lagtime combination (a follow-up)"
                        } else {
                            no_twin_cause
                        }
                    ));
                }
            }
            if dose.is_infusion() {
                // Infusion on a closed-form absorption model reroutes to its ODE twin (#719 gap
                // 2): the twin delivers the dose as a zero-order source feeding the kernel (the
                // convolved `R_in_inf`), which the instantaneous-bolus closed form cannot express.
                // Reject only a twin-less form; `effective_for` performs the reroute.
                if model.absorption_ode_equivalent.is_none() {
                    return Some(format!(
                        "{name} does not support infusion doses in this form (subject {}): an \
                         infusion reroutes to the ODE absorption twin, but this model has no twin \
                         {no_twin_cause}. Write the model as an ODE transit()/igd() forcing in \
                         [odes] for a zero-order input into the kernel.{decline}",
                        subject.id
                    ));
                }
            }
        }
    }
    None
}

/// Fit-boundary validation for RTTE (repeated-event) endpoints, both `clock = forward`
/// (Andersen–Gill total time) and `clock = reset` (gap time). The parser enforces these
/// for `.ferx` models, but a Rust caller can hand-build a `CompiledModel` / `Population`
/// that bypasses it, so `fit()` re-checks up front — otherwise each of these would fold
/// into the RTTE NLL's `1e20` sentinel and produce a silent garbage fit instead of a
/// clear error. Rejects, per RTTE cmt:
///
///   * **interval-censored records** — unsupported for RTTE and folded into `1e20` by
///     both [`crate::survival::rtte_forward_nll_from_curves`] and
///     [`crate::survival::rtte_reset_nll_from_curves`]. Interval censoring is a *data*
///     property (a DV=0→DV=2 pair), so the parser cannot see it — it must be caught here.
///   * **non-finite `TIME`** — a `NaN`/`inf` time slips past the `time < prev` order
///     check (every NaN comparison is false) and poisons `prev`, so reject it explicitly.
///   * **out-of-order rows** — both RTTE clocks require time-sorted rows: clock-forward
///     telescopes the cumulative hazard across a subject's records (each record's time is
///     the lower limit for the next), and clock-reset measures each inter-event gap
///     `Δ = t − t_prev`; unsorted rows give a finite-but-wrong NLL (forward) or a negative
///     gap that folds to `1e20` (reset).
///   * **`entry_time > TIME`** — a left-truncation entry after the record's own time gives
///     a negative first gap (reset) or `H(entry) > H(TIME)` (forward), both folded to the
///     sentinel. The datareader skips such rows on load; a hand-built caller bypasses that.
///
/// And, for **`clock = reset`** endpoints only (clock-forward telescopes absolute `H` and
/// is unaffected):
///
///   * **mid-stream right-censoring** — a censoring does not reset the renewal clock, so a
///     `RightCensored` row before a later record would measure the next gap from the censor
///     rather than the last event (and diverge from the gap-time NONMEM anchor). Gap-time
///     RTTE supports a censor only as the subject's final record; interrupted-observation
///     renewal is a later slice.
///   * **tied event times** — two events at the same `TIME` give a zero gap `Δ = 0`, which
///     the gap-time hazard cannot represent (`h(0)` folds to the sentinel).
///
/// Left truncation (`entry_time > 0`) is **accepted** for clock-reset (#740): the first
/// sojourn is conditioned on survival to `entry` in absolute time (`H(t₁) − H(entry)`,
/// convention B), identical to the single-event and clock-forward delayed-entry handling —
/// see [`crate::survival::rtte_reset_nll_from_curves`].
///
/// Returns `None` when there is no RTTE endpoint or every subject's rows are valid.
#[cfg(feature = "survival")]
pub(crate) fn check_rtte_records(model: &CompiledModel, population: &Population) -> Option<String> {
    use crate::types::{EndpointLikelihood, EventType, ObsRecord, RtteClock, TteRecurrence};
    let mut rtte_cmts: Vec<usize> = Vec::new();
    // Clock-reset endpoints carry an extra data-shape contract (a censor may only be the
    // final record, gaps must be strictly positive); track them so those checks stay
    // scoped to reset and don't reject clock-forward data that telescopes fine.
    let mut reset_cmts: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (cmt, ep) in &model.endpoints {
        if let EndpointLikelihood::Tte {
            recurrence: TteRecurrence::Repeated { clock },
            ..
        } = ep
        {
            rtte_cmts.push(*cmt);
            if matches!(clock, RtteClock::Reset) {
                reset_cmts.insert(*cmt);
            }
        }
    }
    if rtte_cmts.is_empty() {
        return None;
    }
    for subject in &population.subjects {
        for &cmt in &rtte_cmts {
            let mut prev = f64::NEG_INFINITY;
            // Reset only: whether a right-censored row has already been seen for this
            // subject+cmt — any record after it is a mid-stream censor (rejected below).
            let mut seen_censor = false;
            for r in &subject.obs_records {
                let ObsRecord::Event {
                    time,
                    cmt: c,
                    event_type,
                    entry_time,
                    ..
                } = r
                else {
                    // Non-TTE records are handled by their own endpoint; skip them.
                    continue;
                };
                if *c != cmt {
                    continue;
                }
                if matches!(event_type, EventType::IntervalCensored { .. }) {
                    return Some(format!(
                        "Subject '{}': RTTE endpoint CMT={cmt} has an interval-censored record, \
                         which is not supported for repeated events. Use exact (DV=1) event rows \
                         and a right-censoring (DV=0) row.",
                        subject.id
                    ));
                }
                if !time.is_finite() {
                    return Some(format!(
                        "Subject '{}': RTTE endpoint CMT={cmt} has a non-finite TIME ({time}). \
                         Repeated-event rows require finite, time-sorted TIME values.",
                        subject.id
                    ));
                }
                // A left-truncation entry after the record's own event/censoring time gives a
                // negative first gap (Δ = time − entry < 0 under clock-reset) or a decreasing
                // cumulative-hazard increment (H(entry) > H(time) under clock-forward), both of
                // which fold to the silent 1e20 sentinel. The datareader skips such rows on
                // load; a hand-built `Population` bypasses that, so reject it here — the same
                // up-front guarantee this function gives for interval/non-finite/out-of-order.
                if *entry_time > *time {
                    return Some(format!(
                        "Subject '{}': RTTE endpoint CMT={cmt} has entry_time={entry_time} after \
                         TIME={time}. The left-truncation entry must not exceed the record's \
                         event/censoring time.",
                        subject.id
                    ));
                }
                if *time < prev {
                    return Some(format!(
                        "Subject '{}': RTTE endpoint CMT={cmt} has records out of time order \
                         (t={time} follows t={prev}). Repeated-event rows must be sorted by TIME \
                         per subject so the cumulative hazard accumulates correctly.",
                        subject.id
                    ));
                }
                // Clock-reset (gap-time) data-shape contract. The renewal clock resets only
                // at EVENTS and each gap is measured from the previous event, so two shapes
                // that pass every generic check above are still ill-posed under reset and
                // would silently fold to the 1e20 sentinel or diverge from the gap-time
                // definition. Reject them for reset endpoints only — clock-forward telescopes
                // absolute H across a subject's records and is unaffected by either.
                if reset_cmts.contains(&cmt) {
                    // A right-censored row before a later record is a mid-stream censor: a
                    // censoring does NOT reset the renewal clock, so the following gap would
                    // be measured from the censor instead of the last event (also diverging
                    // from the NONMEM gap-time anchor, which advances the clock only on
                    // events). Interrupted-observation renewal is a later slice; fail loud.
                    if seen_censor {
                        return Some(format!(
                            "Subject '{}': RTTE endpoint CMT={cmt} (clock=reset) has a \
                             right-censored (DV=0) row before a later record. Gap-time RTTE \
                             supports right-censoring only as the final record per subject; a \
                             mid-stream censor would restart the renewal clock. Use \
                             clock=forward for interrupted-observation data.",
                            subject.id
                        ));
                    }
                    // Two events at the same TIME give a zero inter-event gap (Δ = 0), which
                    // the gap-time hazard cannot represent — h(0) folds to the sentinel.
                    if matches!(event_type, EventType::Exact) && *time == prev {
                        return Some(format!(
                            "Subject '{}': RTTE endpoint CMT={cmt} (clock=reset) has two events \
                             at the same TIME={time} (zero inter-event gap). Repeated events \
                             must have strictly increasing times so each gap Δ > 0.",
                            subject.id
                        ));
                    }
                    // Left truncation (delayed entry, entry_time > 0) IS supported for
                    // clock-reset (#740): the first sojourn is conditioned on survival to
                    // entry in absolute time (H(t₁) − H(entry), convention B), matching the
                    // single-event and clock-forward delayed-entry handling. No guard here —
                    // `rtte_reset_nll_from_curves` applies the conditioning. (An `entry > TIME`
                    // is still rejected above, shared with clock-forward.)
                    if matches!(event_type, EventType::RightCensored) {
                        seen_censor = true;
                    }
                }
                prev = *time;
            }
        }
    }
    None
}

/// Reject a model / dataset where a **survival hazard references a time-varying
/// covariate** — currently silently frozen at its baseline value (#741).
///
/// The analytic hazard [`crate::types::HazardParamFn`] takes no time argument and is
/// evaluated with the subject's baseline covariate snapshot; the ODE-accumulated (joint
/// PK-TTE) hazard is integrated with the PK-parameter vector frozen at `t=0`
/// (`survival::ode_cumhaz_hazard`). In both cases a covariate that varies within a subject
/// is read at one fixed value, so the fit / simulation would use the wrong hazard with no
/// warning. Until time-varying covariates on hazards are supported, reject up front — as the
/// PK-path TV-covariate paths already do (`check_transit_support`) — rather than silently
/// mis-fit.
///
/// The scope is deliberately tight so a legitimate model is not rejected:
///   * an **analytic** hazard rejects only when a covariate its *own* closed-form parameters
///     reference (`hazard_covariates`) is time-varying — a covariate used solely by a shared
///     PK model (a frailty-only joint fit) is fine, since the analytic hazard never reads it;
///   * an **ODE-accumulated** hazard rejects on *any* time-varying covariate, because the
///     survival ODE solve freezes the whole PK-parameter vector at `t=0`, so a covariate
///     feeding the concentration (and hence the drug-driven hazard) is equally frozen.
///
/// Returns `None` when no TTE endpoint references a time-varying covariate. `fit()` surfaces
/// the message as an `Err`; `predict()`/`simulate()` panic via [`assert_survival_tv_covariates`].
#[cfg(feature = "survival")]
pub(crate) fn check_survival_tv_covariates(
    model: &CompiledModel,
    population: &Population,
) -> Option<String> {
    use crate::types::{EndpointLikelihood, HazardSpec};
    if !model.has_non_gaussian() {
        return None;
    }
    for subject in &population.subjects {
        let tv = subject.time_varying_covariate_names();
        if tv.is_empty() {
            continue;
        }
        let tv_set: std::collections::HashSet<&str> = tv.iter().map(String::as_str).collect();
        for (cmt, endpoint) in &model.endpoints {
            // Binary/categorical (#760): the logit predictor is evaluated at the baseline
            // covariate snapshot (its closure takes no time argument), so a time-varying
            // covariate on it would be silently frozen — the same #741 failure mode the
            // hazard guard below rejects. `lp_covariates` includes covariates reached
            // transitively through `[individual_parameters]`.
            if let EndpointLikelihood::Binary { lp_covariates, .. } = endpoint {
                if let Some(name) = lp_covariates.iter().find(|c| tv_set.contains(c.as_str())) {
                    return Some(format!(
                        "[binary_model] on CMT={cmt}: the logit predictor references covariate \
                         `{name}`, which is time-varying for subject `{}`. Time-varying covariates \
                         on the predictor are not yet supported — it is evaluated at the baseline \
                         covariate value, so the fit would silently use the wrong log-odds (#741). \
                         Use a time-constant (baseline) covariate for now.",
                        subject.id
                    ));
                }
                continue;
            }
            // CTMM (#759): the generator is built once per subject from the baseline
            // covariate snapshot (time-homogeneous — a drug-driven Q(t) is Phase 6), so a
            // time-varying covariate on a transition intensity would be silently frozen.
            #[cfg(feature = "markov")]
            if let EndpointLikelihood::Ctmm {
                generator_covariates,
                ..
            } = endpoint
            {
                if let Some(name) = generator_covariates
                    .iter()
                    .find(|c| tv_set.contains(c.as_str()))
                {
                    return Some(format!(
                        "[markov_model] on CMT={cmt}: a transition intensity references covariate \
                         `{name}`, which is time-varying for subject `{}`. Time-varying covariates \
                         on the generator are not yet supported — the generator is evaluated at the \
                         baseline covariate value (time-homogeneous CTMM), so the fit would \
                         silently use the wrong intensity (#741). A drug-driven Q(t) is Phase 6; \
                         use a time-constant (baseline) covariate for now.",
                        subject.id
                    ));
                }
                continue;
            }
            let EndpointLikelihood::Tte {
                hazard,
                hazard_covariates,
                ..
            } = endpoint
            else {
                continue;
            };
            match hazard {
                HazardSpec::Analytic { .. } => {
                    if let Some(name) = hazard_covariates
                        .iter()
                        .find(|c| tv_set.contains(c.as_str()))
                    {
                        return Some(format!(
                            "[event_model] on CMT={cmt}: the survival hazard references covariate \
                             `{name}`, which is time-varying for subject `{}`. Time-varying \
                             covariates on a hazard are not yet supported — the analytic hazard is \
                             evaluated at the baseline covariate value, so the fit would silently \
                             use the wrong hazard (#741). Use a time-constant (baseline) covariate \
                             on the hazard for now.",
                            subject.id
                        ));
                    }
                }
                HazardSpec::OdeAccumulated { .. } => {
                    // Any time-varying covariate is frozen by the survival ODE solve's t=0
                    // PK-parameter snapshot; name the first one deterministically (`tv` is sorted).
                    let name = &tv[0];
                    return Some(format!(
                        "[event_model] on CMT={cmt}: the ODE-accumulated (joint PK-TTE) hazard is \
                         integrated with covariates frozen at their t=0 baseline, but covariate \
                         `{name}` is time-varying for subject `{}`. Time-varying covariates on a \
                         drug-driven hazard are not yet supported (#741). Hold the covariate \
                         constant within each subject for now.",
                        subject.id
                    ));
                }
            }
        }
    }
    None
}

/// Panic wrapper over [`check_survival_tv_covariates`] for the non-`fit` entry points
/// (`predict()`/`simulate()`), mirroring [`assert_absorption_closed_form_support`]: the same
/// time-varying-covariate-on-hazard combination that `fit()` rejects as an `Err` cannot be
/// honoured by prediction / simulation either (the hazard would be silently frozen), so fail
/// loudly rather than return a subtly wrong result.
#[cfg(feature = "survival")]
pub(crate) fn assert_survival_tv_covariates(model: &CompiledModel, population: &Population) {
    panic_if_unsupported(
        check_survival_tv_covariates(model, population),
        "a survival model / data combination with a time-varying covariate on a hazard",
    );
}

/// Reject an analytic Form C readout (`[scaling] y = <expr>`, #650) that reads
/// the oral **depot** amount on a subject whose dose history dose superposition
/// cannot reconstruct — an EVID=3/4 reset, or a dose the superposition state
/// helper cannot place (#375).
///
/// `apply_analytic_readout` reconstructs the depot amount with
/// [`crate::pk::analytical_state_at_times`], and both failure modes make that
/// reconstruction wrong in a way that reaches `PRED` — and therefore the **OFV**,
/// not just a diagnostic column:
///
///  - a **reset** zeroes the compartments mid-record, which is not a sum of
///    independent dose responses, so the readout would see a zero depot after
///    the reset;
///  - a **non-default dose compartment** is invisible to `single_dose_states`,
///    which never reads `dose.cmt` and places every bolus in compartment 1 and
///    every infusion in central. A bolus written `CMT=2` on a `one_cpt_oral`
///    model is therefore reconstructed as if it had been absorbed through the
///    depot, adding a phantom depot amount. Measured on
///    `[scaling] y = (central + depot)/V`: OFV 188.37 against an ODE twin's
///    1761.47, where the same model with the dose at `CMT=1` agrees with the
///    twin exactly.
///
/// The `ipred` path itself is correct for these subjects (it reroutes to the
/// event-driven walk, `dose_needs_event_walk`), but that walk has no states
/// variant for the readout to read, so there is nothing to fall back to.
/// Rather than return a subtly wrong `PRED`/OFV, fail loudly and point at an ODE
/// model (which integrates the depot state directly). A `central`-only readout
/// is unaffected. Returns `None` for the common no-readout / IV / central-only case.
pub(crate) fn check_analytic_readout_support(
    model: &CompiledModel,
    population: &Population,
) -> Option<String> {
    let ar = model.analytic_readout.as_ref()?;
    if !ar.references_depot() {
        return None;
    }
    for subject in &population.subjects {
        if subject.has_resets() {
            return Some(format!(
                "[scaling] y: an analytic Form C readout that references the oral `depot` \
                 amount is not supported on a subject with an EVID=3/4 reset (subject {}): \
                 the depot amount is reconstructed by dose superposition, which is invalid \
                 across a reset. Reference only the `central` amount, or use an ODE model \
                 (`ode(states=[depot, central])`) which integrates the depot across resets. \
                 See issue #650.",
                subject.id
            ));
        }
        if crate::pk::dose_needs_event_walk(model.pk_model, subject) {
            return Some(format!(
                "[scaling] y: an analytic Form C readout that references the oral `depot` \
                 amount is not supported on a subject dosing a non-default compartment \
                 (subject {}): the depot amount is reconstructed by dose superposition, \
                 which places every bolus in compartment 1 and every infusion in central \
                 regardless of `CMT`, so a dose elsewhere would contribute a phantom depot \
                 amount to `PRED` (and to the objective). Reference only the `central` \
                 amount, dose the default compartment, or use an ODE model \
                 (`ode(states=[depot, central])`) which addresses compartments directly. \
                 See issue #375.",
                subject.id
            ));
        }
    }
    None
}

/// Panic on an unsupported transit / IG closed-form model/data combination, for the
/// `Vec`-returning `predict()`/`simulate()` paths (mirrors
/// [`assert_modeled_doses_supported`]). `fit()` returns these as an `Err`.
pub(crate) fn assert_absorption_closed_form_support(
    model: &CompiledModel,
    population: &Population,
) {
    panic_if_unsupported(
        check_absorption_closed_form_support(model, population),
        "a model/data combination the analytic absorption closed form cannot honour",
    );
}

/// Reject a transit closed form with **no ODE twin** whose η = 0 typical parameters
/// fall **outside the closed form's convergence domain** — the flip-flop regime
/// (`ke ≥ KTR` 1-cpt / `α ≥ KTR` 2-cpt), or the 2-cpt confluent-eigenvalue edge
/// (`|α − β| < 1e-12`) — where it returns an identically-zero profile that silently
/// degenerates the objective (a proportional error model collapses `(σ·pred)²` to 0).
/// Unlike a twin-carrying model — which [`crate::pk::effective_model_for_eval`]
/// auto-reroutes to its exact ODE `transit()` twin — a twin-less form (one carrying a
/// `lagtime`, bioavailability `f`, or a user `[odes]` / `[scaling]` /
/// `[initial_conditions]` block, which the desugar declines) has nothing to reroute
/// to. The boundary is parameter-dependent, so this can't be caught by
/// [`check_absorption_closed_form_support`]'s structural checks — it is evaluated at the typical
/// (η = 0) parameters via [`crate::pk::absorption_flip_flop_at`], matching the
/// informational `W_TRANSIT_FLIP_FLOP` warning that fires for the twin-carrying case.
/// Returns the first offending subject's message, or `None` when no reject applies
/// (non-transit, twin present, or in-domain). `fit()` / `ferx check` surface this as
/// an error; `predict()`/`simulate()` panic via [`assert_absorption_flip_flop_no_twin`],
/// mirroring [`check_absorption_closed_form_support`] / [`assert_absorption_closed_form_support`].
pub(crate) fn check_absorption_flip_flop_no_twin(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
) -> Option<String> {
    // A twin-carrying model is auto-rerouted (correct) — only warned, not rejected.
    if !matches!(
        model.pk_model,
        PkModel::OneCptTransit | PkModel::TwoCptTransit | PkModel::OneCptIg | PkModel::TwoCptIg
    ) || model.absorption_ode_equivalent.is_some()
    {
        return None;
    }
    // Model-family-specific domain / ODE-forcing wording (the tilting abscissa and the
    // `[odes]` forcing differ between transit and IG).
    let (domain, ode_fn, params) = match model.pk_model {
        PkModel::OneCptTransit | PkModel::TwoCptTransit => {
            ("transit rate KTR = (n+1)/mtt", "transit()", "MTT / CL")
        }
        _ => ("1/(2·MAT·CV²)", "igd()", "MAT / CV² / CL"),
    };
    // The twin-less cause, and (for a #1008 decline) the twin parser's own reason — the
    // desugar's by-name declines and a rejected twin build both land on `is_none()` here.
    let cause = if crate::types::absorption_twin_decline_reason(model).is_some() {
        "its twin was built and declined, #1008"
    } else {
        "a user `[odes]` / `[scaling]` / `[initial_conditions]` block, or a parameter whose name \
         collides with a reserved `f`/`lagtime` slot, declines the desugar — a `lagtime=`/`f=` \
         mapping alone now auto-routes, #735"
    };
    let decline = twin_decline_clause(model);
    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    for subject in &population.subjects {
        if crate::pk::absorption_flip_flop_at(model, subject, theta, &zero_eta) {
            return Some(format!(
                "{} starts outside the analytic absorption closed form's convergence domain at \
                 typical values (subject {}) — the flip-flop regime (disposition rate ≥ {domain}), \
                 or (2-cpt) coincident disposition eigenvalues — so it returns an \
                 identically-zero concentration profile, which silently degenerates the objective \
                 (a proportional error model collapses `(σ·pred)²` to 0). This model has no ODE \
                 twin to fall back on ({cause}) — rewrite it as an explicit ODE `{ode_fn}` model, \
                 or check the {params} starting estimates.{decline}",
                model.pk_model.canonical_name(),
                subject.id
            ));
        }
    }
    None
}

/// Panic on a twin-less flip-flop absorption model for the `Vec`-returning
/// `predict()`/`simulate()` paths (mirrors [`assert_absorption_closed_form_support`]). `fit()`
/// returns this as an `Err`.
pub(crate) fn assert_absorption_flip_flop_no_twin(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
) {
    panic_if_unsupported(
        check_absorption_flip_flop_no_twin(model, population, theta),
        "a model/data combination the absorption closed form cannot honour",
    );
}

/// Panic on a depot-referencing analytic Form C readout + reset subject, for the
/// `Vec`-returning `predict()`/`simulate()` paths (mirrors
/// [`assert_absorption_closed_form_support`]). `fit()` returns this as an `Err`.
pub(crate) fn assert_analytic_readout_support(model: &CompiledModel, population: &Population) {
    panic_if_unsupported(
        check_analytic_readout_support(model, population),
        "a model/data combination the analytic Form C readout cannot honour",
    );
}

/// Panic on a malformed built-in **absorption input-rate** model/data combination —
/// a pathway-fraction value out of `(0, 1]` or not summing to 1, an out-of-domain
/// forcing parameter, or an SS / infusion / `[diffusion]` dose into an input-rate
/// compartment — for the `Vec`-returning `predict()` / `simulate()` paths (mirrors
/// [`assert_absorption_closed_form_support`]). `fit()` (and `ferx check`) surface these as an `Err`
/// via [`check_model_data`], but the simulate/predict paths run no data-check, so
/// without this a malformed multi-pathway model would be simulated with silently
/// wrong dose delivery (#588). The data-independent *structural* fraction rules are
/// enforced even earlier, at parse time in `build_ode_spec`; this reuses
/// [`check_absorption_dosing`] for the *value* / domain checks that need data. A
/// no-op for any model with no built-in input-rate forcing (the common case).
pub(crate) fn assert_absorption_dosing_supported(model: &CompiledModel, population: &Population) {
    panic_if_unsupported(
        first_error(&check_absorption_dosing(model, population)).err(),
        "a model/data combination the built-in absorption input-rate machinery cannot honour",
    );
}

/// Model + estimation-option *compatibility* checks that don't depend on data:
/// estimation method vs an SDE (`[diffusion]`) model, IMP chain placement, and
/// optimizer vs IOV. These mirror the guards at the top of `fit_inner`, so a
/// clean `ferx check` and a `fit()` agree on which method/model combinations are
/// rejected (rather than reporting `valid: true` and then failing at fit time).
/// `fit_inner` consumes these via [`first_error`]; message text is identical to
/// the historical inline guards. Check order matches `fit_inner` so the first
/// error is unchanged.
pub fn check_model_options(model: &CompiledModel, options: &FitOptions) -> Vec<Diagnostic> {
    let chain = options.method_chain();
    let mut diags = Vec::new();

    // A fixed-effects-only model (`n_eta = 0`, #989 / #1007) has no latent
    // variable to integrate over, so every estimator whose objective *is* an
    // integral over η degenerates there. SAEM's E-step is empty — it runs,
    // reports `converged`, and lands slightly off the true optimum (−269.5986 vs
    // −269.6370 on the `one_cpt_iv` zero-Ω anchor) purely because the SA gain
    // sequence stops short; the other three collapse to the observation
    // likelihood FOCE/FOCEI already minimise exactly. One table, so adding a
    // method to the family cannot leave one of the two codes behind — `saem` keeps
    // `E_SAEM_NO_RANDOM_EFFECTS` (documented as stable since #1002, and tooling
    // may match on it) and the three #1007 additions share one new sibling code.
    // The per-method sentence is the only text that differs; the surrounding
    // message is byte-identical to the historical inline guards.
    //
    // Each of the three also refuses at run time — five sites, not the three the
    // issue lists: `importance_sampling.rs` (both the estimating and the marginal
    // entry points), `impmap.rs` (both the IMP and IMPMAP drivers) and
    // `bayes.rs`. Those stay as the backstop for callers that bypass this check;
    // rejecting here as well means `ferx check` catches it and a
    // `methods = [focei, imp]` chain fails up front rather than after the FOCEI
    // stage has run. Note this also fail-fasts an `imp_eval_only` chain whose IMP
    // failure was previously downgraded to a warning after the fit: requesting an
    // IMP objective at n_eta = 0 is a spec contradiction better surfaced before
    // the fit runs.
    if model.n_eta == 0 {
        for (method, name, code, why) in [
            (
                EstimationMethod::Saem,
                "saem",
                "E_SAEM_NO_RANDOM_EFFECTS",
                "SAEM is an EM over the random effects, so with none declared its \
                 E-step is empty and it only approaches the objective that FOCE/FOCEI \
                 minimise exactly.",
            ),
            (
                EstimationMethod::Imp,
                "imp",
                "E_METHOD_NO_RANDOM_EFFECTS",
                "With no random effects the marginal likelihood is just the \
                 observation likelihood, which FOCE/FOCEI minimise exactly.",
            ),
            (
                EstimationMethod::Impmap,
                "impmap",
                "E_METHOD_NO_RANDOM_EFFECTS",
                "With no random effects the marginal likelihood is just the \
                 observation likelihood, which FOCE/FOCEI minimise exactly.",
            ),
            (
                EstimationMethod::Bayes,
                "bayes",
                "E_METHOD_NO_RANDOM_EFFECTS",
                "With no random effects the marginal likelihood is just the \
                 observation likelihood, which FOCE/FOCEI minimise exactly.",
            ),
        ] {
            if chain.contains(&method) {
                diags.push(
                    Diagnostic::error(
                        code,
                        format!(
                            "method = {name} requires at least one random effect \
                             (n_eta = 0). {why} Use method = foce, focei, or laplace \
                             for a fixed-effects-only (naive-pooled) model."
                        ),
                    )
                    .with_block("fit_options"),
                );
            }
        }
    }

    // Pure Gauss-Newton at n_eta = 0 (#1006) is start-sensitive rather than
    // invalid: with no inner EBE loop to absorb a poor `sigma` start, the BHHH
    // outer-product Hessian over-estimates curvature far from the optimum and
    // the LM-damped step can collapse — on the `one_cpt_iv_pooled` anchor it
    // stops 8940 OFV units short, reporting nothing beyond `Converged: NO`.
    // A warning, not an error, because `gn` does reach the optimum from a good
    // start. Exempt whenever a later stage re-optimises the GN result: that is
    // `gn_hybrid` (whose FOCEI polish is internal to `run_foce_gn`) but equally
    // a hand-written `methods = [gn, focei]`, which is the same thing spelled
    // out — hence `is_last_estimating_stage` on the GN stage rather than a bare
    // `chain.contains`. An unconverged pure-GN run at n_eta = 0 additionally
    // pushes a post-fit warning (`gauss_newton.rs`), under the same rule, since
    // `Converged: NO` is the state that actually predicts the bad answer.
    if model.n_eta == 0
        && chain
            .iter()
            .rposition(|&m| m == EstimationMethod::FoceGn)
            .is_some_and(|idx| {
                crate::api::is_last_estimating_stage(&chain, idx, options.imp_eval_only)
            })
    {
        diags.push(
            Diagnostic::warning(
                "W_GN_NO_RANDOM_EFFECTS",
                "method = gn on a model with no random effects (n_eta = 0) is \
                 start-sensitive: without an inner EBE loop to absorb a poor start, \
                 the BHHH step can collapse far from the optimum and return a badly \
                 wrong result with no diagnostic beyond Converged: NO. Prefer \
                 method = gn_hybrid or focei for a fixed-effects-only model, or \
                 improve the sigma start.",
            )
            .with_block("fit_options"),
        );
    }

    // SDE ([diffusion]) is incompatible with SAEM, with the Gauss-Newton
    // methods, and with the analytic-sensitivity gradient path (EKF estimation
    // requires FD-FOCE/FOCEI).
    if model.is_sde() {
        if chain.iter().any(|&m| m == EstimationMethod::Saem) {
            diags.push(
                Diagnostic::error(
                    "E_SDE_INCOMPATIBLE",
                    "method = saem is not compatible with a [diffusion] block. \
                     SDE / EKF estimation requires FOCE or FOCEI. Use method = foce or method = focei.",
                )
                .with_block("fit_options"),
            );
        }
        if chain
            .iter()
            .any(|&m| matches!(m, EstimationMethod::FoceGn | EstimationMethod::FoceGnHybrid))
        {
            diags.push(
                Diagnostic::error(
                    "E_SDE_INCOMPATIBLE",
                    "SDE ([diffusion]) is not supported with method = gn or gn_hybrid. \
                     Use method = foce or method = focei.",
                )
                .with_block("fit_options"),
            );
        }
        if chain.iter().any(|&m| m == EstimationMethod::Impmap) {
            diags.push(
                Diagnostic::error(
                    "E_SDE_INCOMPATIBLE",
                    "method = impmap is not compatible with a [diffusion] block. \
                     The EKF process-noise variance is not threaded through the IMPMAP \
                     importance-sampling likelihood. Use method = foce or method = focei.",
                )
                .with_block("fit_options"),
            );
        }
        // `gradient_method = ad` is rejected unconditionally below (E_AD_RETIRED),
        // so it needs no SDE-specific case here.
    }

    // Binary / categorical (#760) endpoint compatibility. FOCEI (default), SAEM, and IMP
    // are supported; the Gauss-Newton gradient is Gaussian-specific (plan §13), and IOV is
    // not yet folded into the *outer* FOCE-IOV objective (the term enters only the inner
    // EBE objective). Reject both fail-loud rather than silently mis-fit.
    if model.has_binary() {
        if chain
            .iter()
            .any(|&m| matches!(m, EstimationMethod::FoceGn | EstimationMethod::FoceGnHybrid))
        {
            diags.push(
                Diagnostic::error(
                    "E_BINARY_GN_UNSUPPORTED",
                    "method = gn / gn_hybrid is not supported with a [binary_model] endpoint. \
                     The Gauss-Newton gradient is built from the Gaussian residual structure and \
                     ignores the logistic likelihood, so it would not move the parameters. Use \
                     method = focei (default), saem, or imp.",
                )
                .with_block("fit_options"),
            );
        }
        if model.n_kappa > 0 {
            diags.push(
                Diagnostic::error(
                    "E_BINARY_IOV_UNSUPPORTED",
                    "a [binary_model] endpoint is not yet supported together with inter-occasion \
                     variability ([iov] / κ): the binary term enters the inner EBE objective but \
                     not the outer IOV objective, so estimates would be biased. Fit the binary \
                     endpoint without [iov] for now.",
                )
                .with_block("fit_options"),
            );
        }
    }

    // CTMM (#759) endpoint compatibility — same shape as the binary case above. FOCEI
    // (default), SAEM, and IMP are supported; the Gauss-Newton gradient is Gaussian-
    // specific, and IOV is not folded into the outer FOCE-IOV objective (the generator
    // is built with BSV-only η). Reject both fail-loud rather than silently mis-fit.
    #[cfg(feature = "markov")]
    if model.has_ctmm() {
        if chain
            .iter()
            .any(|&m| matches!(m, EstimationMethod::FoceGn | EstimationMethod::FoceGnHybrid))
        {
            diags.push(
                Diagnostic::error(
                    "E_CTMM_GN_UNSUPPORTED",
                    "method = gn / gn_hybrid is not supported with a [markov_model] (CTMM) \
                     endpoint. The Gauss-Newton gradient is built from the Gaussian residual \
                     structure and ignores the Markov transition likelihood, so it would not move \
                     the parameters. Use method = focei (default), saem, or imp.",
                )
                .with_block("fit_options"),
            );
        }
        if model.n_kappa > 0 {
            diags.push(
                Diagnostic::error(
                    "E_CTMM_IOV_UNSUPPORTED",
                    "a [markov_model] (CTMM) endpoint is not yet supported together with \
                     inter-occasion variability ([iov] / κ): the generator is built with \
                     per-subject (BSV) η only, so κ would be ignored. Fit the CTMM endpoint \
                     without [iov] for now.",
                )
                .with_block("fit_options"),
            );
        }
    }

    // IMPMAP does not yet support inter-occasion variability (κ / [iov]); the κ
    // sufficient statistics and Ω_iov M-step are a planned follow-up. Surface it
    // at check time so `ferx check` rejects it rather than the fit failing at
    // runtime (possibly after a chained warm-up stage has already run).
    if model.n_kappa > 0 && chain.iter().any(|&m| m == EstimationMethod::Impmap) {
        diags.push(
            Diagnostic::error(
                "E_IMPMAP_IOV_UNSUPPORTED",
                "method = impmap does not yet support inter-occasion variability \
                 (κ / [iov]). Use method = saem or method = focei for IOV models.",
            )
            .with_block("fit_options"),
        );
    }

    // ── Quadrature: FOCEI + n_agq > 1 is the Gauss-Newton-anchored refinement (#251) ──────
    // The GN anchor's `H̃` comes from the sensitivity provider (`score_core`), so the
    // refinement is available only where that provider reaches — the same analytic scope
    // FOCEI's own outer gradient needs. Outside it (e.g. an ODE model without sensitivities,
    // or a pure-TTE endpoint), reject up front rather than fail deep in the objective.
    if chain.iter().any(|&m| matches!(m, EstimationMethod::FoceI))
        && options.n_agq > 1
        && !crate::estimation::agq::analytic_score_supported(model)
    {
        diags.push(
            Diagnostic::error(
                "E_FOCEI_NAGQ_UNSUPPORTED",
                "method = focei with n_agq > 1 (the Gauss-Newton-anchored quadrature) needs a \
                 model in the analytic sensitivity scope. Use method = laplace with n_agq > 1 \
                 for the exact-anchor quadrature, which has no such restriction.",
            )
            .with_block("fit_options"),
        );
    }

    // ── AGQ / Laplace (#251) ──────────────────────────────────────────────────────────────
    // Laplace is the exact-anchor quadrature. At `n_agq = 1` its grid is a single point no
    // matter how many random effects there are, so the caps below never fire; at `n_agq > 1`
    // they bound the tensor grid.
    // Both quadrature methods hit these caps: `laplace` at any `n_agq`, and `focei` at
    // `n_agq > 1` (the Gauss-Newton-anchored quadrature — same `n_agq^d` tensor grid). The
    // block must fire for either, or a non-IOV FOCEI quadrature would skip the check-time
    // caps entirely (the runtime cap in `fit_inner` only fires under IOV) and build a
    // 100k+-node grid unchecked (#251 review #1).
    let laplace_stage = chain
        .iter()
        .any(|&m| matches!(m, EstimationMethod::Laplace));
    let focei_quadrature =
        chain.iter().any(|&m| matches!(m, EstimationMethod::FoceI)) && options.n_agq > 1;
    if laplace_stage || focei_quadrature {
        // IOV is **supported**: the quadrature integrates over the stacked (η, κ₁..κ_K),
        // which is exactly what `individual_nll_iov` already scores. What IOV changes is the
        // *dimension* — and hence the tensor grid — not the method. The grid cap below is
        // what decides whether a given IOV model is tractable, and it is checked against the
        // stacked dimension in `fit_inner` (which, unlike this function, can see the
        // occasion count in the data).
        //
        // Name the quadrature method the user actually wrote.
        let label = if laplace_stage { "laplace" } else { "focei" };
        // The tensor rule costs `n_agq^d` full likelihood evaluations per subject per
        // outer iteration. Past the cap that is not "slow", it is a fit that will never
        // finish — so it is worth failing at check time, with the two levers spelled out.
        // Both quadrature methods use `n_agq` as the node count.
        let n_nodes = options.n_agq.max(1);
        // Per-dimension node cap. The parser (`apply_fit_option`) enforces this for
        // file-driven fits, but a `FitOptions` built directly against the Rust API
        // bypasses that path — so re-check here, alongside the grid cap, or an
        // out-of-range `n_agq` (e.g. 100 with a single η, which slips under the grid
        // cap) would reach `gauss_hermite` past the point Golub–Welsch stays accurate.
        if n_nodes > crate::estimation::agq::MAX_AGQ_NODES {
            diags.push(
                Diagnostic::error(
                    "E_AGQ_NODES_TOO_LARGE",
                    format!(
                        "method = {label} with n_agq = {n_nodes} exceeds the {}-node per-dimension \
                         limit: beyond ~20 nodes the Golub–Welsch rule loses its extreme nodes to \
                         round-off and the marginal gains nothing. Lower n_agq (n_agq = 1 is the \
                         Laplace approximation, i.e. FOCEI-equivalent cost).",
                        crate::estimation::agq::MAX_AGQ_NODES,
                    ),
                )
                .with_block("fit_options"),
            );
        }
        let grid = crate::estimation::agq::grid_size(n_nodes, model.n_eta);
        if grid > crate::estimation::agq::MAX_AGQ_GRID {
            diags.push(
                Diagnostic::error(
                    "E_AGQ_GRID_TOO_LARGE",
                    format!(
                        "method = {label} with n_agq = {n_nodes} and {} random effects needs a \
                         {n_nodes}^{} = {grid}-node tensor grid per subject per iteration, over \
                         the {} limit. Lower n_agq (n_agq = 1 is the Laplace approximation, i.e. \
                         FOCEI-equivalent cost), reduce the number of random effects, or use \
                         method = saem / imp, whose cost does not grow with the η dimension.",
                        model.n_eta,
                        model.n_eta,
                        crate::estimation::agq::MAX_AGQ_GRID,
                    ),
                )
                .with_block("fit_options"),
            );
        }
    }

    // ── Quadrature × trust_region: objective and gradient would disagree ──────────────────
    // A quadrature stage minimises the AGQ marginal, and every optimizer reaches it through
    // `pop_nll_opts`. Only the *gradient* differs: `optimize_nlopt` and `optimize_bfgs` both
    // go through `population_gradient`, which branches on `agq_nodes()` and hands the job to
    // `agq::agq_population_gradient`. The trust-region path has its own `Gradient` impl
    // (`trust_region::FoceiProblem`) with no such branch — it returns the FOCE/Laplace
    // closed form regardless.
    //
    // That mismatch does not fail loudly. `FitOptions::agq_nodes` and `population_gradient`
    // both spell out what it does instead: "an outer loop minimising the AGQ objective while
    // fed the analytic *FOCE* gradient would converge, silently, to the wrong parameters" —
    // reporting AGQ OFVs at the FOCE optimum. Reject up front until the trust region routes
    // the quadrature gradient (and gets a BHHH Hessian built from per-subject AGQ scores) — #1047.
    if (laplace_stage || focei_quadrature) && options.optimizer == Optimizer::TrustRegion {
        let label = if laplace_stage { "laplace" } else { "focei" };
        diags.push(
            Diagnostic::error(
                "E_OPTIMIZER_AGQ",
                format!(
                    "optimizer = trust_region does not support the quadrature objective \
                     (method = {label}{}). Its gradient is the FOCE/Laplace closed form, so it \
                     would descend on a different function than the one being scored and \
                     converge silently to the FOCE optimum while reporting quadrature OFVs. \
                     Use optimizer = bobyqa, slsqp, lbfgs, nlopt_lbfgs, mma, or bfgs, which \
                     route the quadrature gradient.",
                    if focei_quadrature && !laplace_stage {
                        format!(" with n_agq = {}", options.n_agq)
                    } else {
                        String::new()
                    }
                ),
            )
            .with_block("fit_options"),
        );
    }

    // Explicit `gradient_method = ad`: the Enzyme autodiff path was retired in
    // favour of the hand-rolled `Dual2` analytic sensitivities. Reject it rather
    // than silently running a different method. `auto` (analytic where in scope,
    // else FD) and `fd` are unaffected.
    if options.gradient_method == crate::types::GradientMethod::Ad {
        diags.push(
            Diagnostic::error(
                "E_AD_RETIRED",
                "gradient_method = ad is no longer supported: the Enzyme automatic-differentiation \
                 path was retired in favour of the analytic `Dual2` sensitivities. Set \
                 gradient_method = auto (uses the exact analytic gradient where it is in scope and \
                 falls back to finite differences otherwise) or fd.",
            )
            .with_block("fit_options"),
        );
    }

    if !model.residual_correlations.is_empty() {
        for &m in &chain {
            if !matches!(
                m,
                EstimationMethod::Foce
                    | EstimationMethod::FoceI
                    | EstimationMethod::Saem
                    | EstimationMethod::Imp
                    | EstimationMethod::Laplace
            ) {
                diags.push(
                    Diagnostic::error(
                        "E_BLOCK_SIGMA_METHOD_UNSUPPORTED",
                        "block_sigma correlated residual errors are currently supported for \
                         method = foce, focei, saem, imp, agq, and laplace only. The gn / \
                         gn_hybrid Gauss-Newton paths still use diagonal residual-error \
                         derivatives.",
                    )
                    .with_block("fit_options"),
                );
                break;
            }
        }
        if matches!(model.bloq_method, BloqMethod::M3) {
            diags.push(
                Diagnostic::error(
                    "E_BLOCK_SIGMA_M3_UNSUPPORTED",
                    "block_sigma correlated residual errors are not yet supported with \
                     M3 censored-observation likelihoods.",
                )
                .with_block("fit_options"),
            );
        }
        if model.frem_config.is_some() {
            diags.push(
                Diagnostic::error(
                    "E_BLOCK_SIGMA_FREM_UNSUPPORTED",
                    "block_sigma correlated residual errors are not yet supported with \
                     FREM covariate pseudo-observations.",
                )
                .with_block("fit_options"),
            );
        }
        if model.residual_error_eta.is_some() {
            diags.push(
                Diagnostic::error(
                    "E_BLOCK_SIGMA_IIV_ON_RUV_UNSUPPORTED",
                    "block_sigma correlated residual errors are not yet supported with \
                     iiv_on_ruv residual-error scaling.",
                )
                .with_block("fit_options"),
            );
        }
        if model.n_kappa > 0 {
            diags.push(
                Diagnostic::error(
                    "E_BLOCK_SIGMA_IOV_UNSUPPORTED",
                    "block_sigma correlated residual errors are not yet supported with \
                     inter-occasion variability (κ / [iov]) because the IOV inner \
                     objective does not yet use the full residual covariance matrix.",
                )
                .with_block("fit_options"),
            );
        }
        // SDE is unsupported for SAEM only: the FOCE non-interaction path adds the
        // EKF process-noise `p_obs` into the dense R (foce_subject_nll_standard),
        // so FOCE + block_sigma + SDE works, but SAEM's M-step data term
        // (`obs_nll_subject_into`) carries no `p_obs`, so its E-step sampler and
        // M-step objective would optimize different likelihoods. FOCE keeps the
        // combination it supported before this PR added SAEM support.
        let chain_has_saem = chain.iter().any(|&m| m == EstimationMethod::Saem);
        if chain_has_saem && model.is_sde() {
            diags.push(
                Diagnostic::error(
                    "E_BLOCK_SIGMA_SDE_UNSUPPORTED",
                    "block_sigma correlated residual errors are not yet supported with \
                     method = saem and SDE / EKF process-noise likelihoods.",
                )
                .with_block("fit_options"),
            );
        }
        // Every non-Gaussian endpoint is unsupported for every method: the Laplace path
        // (foce_subject_nll_interaction_with_tte) accumulates R diagonally and cannot
        // represent the cross-endpoint covariance, so it would silently drop the correlation
        // for FOCE as well as SAEM — for TTE and the #760 binary/categorical endpoint alike.
        if model.has_non_gaussian() {
            diags.push(
                Diagnostic::error(
                    "E_BLOCK_SIGMA_TTE_UNSUPPORTED",
                    "block_sigma correlated residual errors are not yet supported with \
                     time-to-event or binary/categorical endpoints.",
                )
                .with_block("fit_options"),
            );
        }
    }

    // `imp` may appear at most once in a chain. By default it is an MCEM
    // estimator (NONMEM `METHOD=IMP`) and may sit anywhere in the chain. With
    // `imp_eval_only = true` (NONMEM `IMP EONLY=1`) it instead evaluates the
    // marginal −2 log L at fixed parameters and must be the terminal stage —
    // placing an evaluator mid-chain would leave `FitResult.importance_sampling`
    // computed at parameters the following stage then overwrites.
    if chain.iter().any(|&m| m == EstimationMethod::Imp) {
        let n_imp = chain
            .iter()
            .filter(|&&m| m == EstimationMethod::Imp)
            .count();
        if n_imp > 1 {
            diags.push(
                Diagnostic::error(
                    "E_IMP_CHAIN",
                    "method `imp` may appear at most once in a chain.",
                )
                .with_block("fit_options"),
            );
        }
        if options.imp_eval_only && chain.last().copied() != Some(EstimationMethod::Imp) {
            diags.push(
                Diagnostic::error(
                    "E_IMP_CHAIN",
                    "method `imp` with `imp_eval_only = true` must be the final stage of the chain \
                     — placing the evaluator mid-chain would leave `FitResult.importance_sampling` \
                     populated with a log-likelihood computed at parameters that the following \
                     stage then overwrites. Move `imp` to the end, or drop `imp_eval_only` to run \
                     it as an estimator.",
                )
                .with_block("fit_options"),
            );
        }
    }

    // The trust-region outer optimizer does not thread kappas through its OFV.
    if model.n_kappa > 0 && options.optimizer == Optimizer::TrustRegion {
        diags.push(
            Diagnostic::error(
                "E_OPTIMIZER_IOV",
                "optimizer = trust_region does not support IOV (n_kappa > 0). \
                 Use optimizer = bobyqa, slsqp, lbfgs, nlopt_lbfgs, mma, or bfgs \
                 for models with kappa declarations.",
            )
            .with_block("fit_options"),
        );
    }

    diags
}

/// Data-dependent *warning*-level checks: malformed steady-state rows, EVID=3/4
/// resets under an SDE model, and a negative typical-value lag time. These are
/// non-fatal — `fit()` pushes their messages into `FitResult.warnings` and
/// proceeds; `ferx check` reports them as `Warning` diagnostics. Message text
/// is identical to the historical inline strings.
///
/// Feature-presence (data-independent) notices such as the experimental-feature
/// warnings live in [`check_experimental_features`] instead, so `ferx check`
/// surfaces them even without a `--data` file.
pub fn check_model_data_warnings(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
) -> Vec<Diagnostic> {
    let mut diags = Vec::new();

    // Warning half of the kappa-weight check (#1031) — a sample-size weight that
    // moves within an occasion. The error half rides the fatal list.
    diags.extend(check_kappa_weight_variation(model, population));

    // SS=1 with II ≤ 0 — the SS branch is gated on `dose.ii > 0`, so the dose
    // is silently treated as a single (non-SS) dose.
    let n_ss_bad_ii = population
        .subjects
        .iter()
        .filter(|s| s.doses.iter().any(|d| d.ss && d.ii <= 0.0))
        .count();
    if n_ss_bad_ii > 0 {
        diags.push(Diagnostic::warning(
            "W_STEADY_STATE_II",
            format!(
                "{} subject(s) have SS=1 doses with missing or non-positive II. \
                 SS predictions require II > 0 — these doses are treated as \
                 non-SS (no steady-state pre-equilibration). Set II in the \
                 dataset or remove the SS flag.",
                n_ss_bad_ii
            ),
        ));
    }

    // SS=1 infusion with T_inf > II (overlapping pulses). The analytical 1-/2-/3-
    // cpt closed forms now superpose the overlapping past pulse train (#379), so
    // these are handled exactly. Only ODE models — and subjects that route to the
    // event-driven walker (EVID 3/4 resets) — still skip SS pre-equilibration.
    let analytic_handles_overlap = model.ode_spec.is_none();
    // The effective infusion length is
    // `d.duration` for an ordinary infusion, but for a modeled dose (RATE=-2 →
    // `D{cmt}` duration, or RATE=-1 → `R{cmt}` rate; #324) it is unresolved here
    // (`rate`/`duration` are 0 until `resolve_rate`), so resolve it at the
    // typical-value point through the engine-correct `active_dose_attr_map()`.
    //
    // Resolution goes through the single-source-of-truth `DoseEvent::resolve_rate`
    // (the same rule + floor clamps the integrator applies at runtime), not a
    // hand-rolled slot read, so the warning's notion of "duration" can't drift
    // from the integrator's. It is a *typical-value* heuristic: it uses init
    // theta, eta = 0, and this subject's covariates, whereas runtime uses
    // per-occasion eta/IOV — a modeled SS infusion whose duration crosses `II`
    // only on some occasions may not be flagged here (and conversely a typical
    // overlap may not occur on every occasion). The runtime SS-skip is the
    // backstop; this catches the common typical-value / covariate-driven overlap.
    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    let effective_duration = |s: &Subject, d: &DoseEvent| -> f64 {
        use crate::types::{DoseAttr, RateMode};
        // The dose attribute whose slot holds this modeled dose's value.
        let attr = match d.rate_mode {
            RateMode::Fixed => return d.duration,
            RateMode::ModeledDuration => DoseAttr::Duration,
            RateMode::ModeledRate => DoseAttr::Rate,
        };
        // Guard the slot's existence: a modeled dose with no matching parameter is
        // an upstream *error* (`E_MODELED_DURATION_NO_PARAM` / `E_MODELED_RATE_NO_PARAM`),
        // but this warnings pass must stay panic-free if run on such a model rather
        // than hit `resolve_rate`'s slot `.expect`.
        let attr_map = model.active_dose_attr_map();
        if attr_map.indexed_slot(attr, d.cmt_raw()).is_some() {
            // Initial-estimate warning pass: typical values at t=0.
            let pk = (model.pk_param_fn)(&init_params.theta, &zero_eta, &s.covariates, 0.0);
            d.resolve_rate(attr_map, &pk.values).duration
        } else {
            0.0
        }
    };
    let n_ss_overlapping_inf = population
        .subjects
        .iter()
        .filter(|s| {
            let overlapping = s
                .doses
                .iter()
                .any(|d| d.ss && d.ii > 0.0 && d.is_infusion() && effective_duration(s, d) > d.ii);
            overlapping && (!analytic_handles_overlap || s.has_resets())
        })
        .count();
    if n_ss_overlapping_inf > 0 {
        diags.push(Diagnostic::warning(
            "W_STEADY_STATE_INFUSION",
            format!(
                "{} subject(s) have SS=1 infusions with T_inf > II (overlapping \
                 pulses) on a model path that does not pre-equilibrate steady \
                 state (ODE model, or EVID=3/4 resets routing to the event-driven \
                 walker) — the dose is applied as a single (non-SS) infusion, so \
                 the system is not at steady state at the dose time. The analytical \
                 1-/2-/3-cpt closed forms do handle this case; use a shorter \
                 infusion (T_inf ≤ II) or remove the SS flag otherwise.",
                n_ss_overlapping_inf
            ),
        ));
    }

    // SS=1 dose combined with an [initial_conditions] baseline (#521). The SS
    // closed form already assumes an infinite periodic dose history (the system
    // is pre-equilibrated at the dose time), while the analytical init lays down a
    // separate transient baseline that decays from t=0. Superposing the two is a
    // valid linear combination but rarely what a user intends — an SS dose and an
    // explicit starting amount usually express the same idea twice, double-counting
    // the baseline. Surface it rather than silently combine them.
    if !model.analytical_init.is_empty() {
        let n_ss_init = population
            .subjects
            .iter()
            .filter(|s| s.has_ss_doses())
            .count();
        if n_ss_init > 0 {
            diags.push(Diagnostic::warning(
                "W_STEADY_STATE_INIT",
                format!(
                    "{} subject(s) have SS=1 doses while the model declares an \
                     [initial_conditions] baseline. The steady-state closed form \
                     already pre-equilibrates an infinite dose history; the init \
                     baseline is superposed on top as a separate decaying transient, \
                     which double-counts the starting amount. Remove the SS flag or \
                     the [initial_conditions] block unless this superposition is \
                     intended.",
                    n_ss_init
                ),
            ));
        }
    }

    // Modeled coded-RATE parameters (`D{cmt}` for RATE=-2 duration, `R{cmt}` for
    // RATE=-1 rate; #324) that are non-positive at the initial typical-value point
    // (eta = 0). `resolve_rate` clamps a transient `D ≤ 0` / `R ≤ 0` to its floor
    // so `AMT/D` / `AMT/R` stays finite mid-search, but a non-positive value *at
    // the initial estimate* signals a misspecified parameterisation — a `D ≤ 0`
    // delivers the dose as a bolus-like spike, an `R ≤ 0` as a near-zero trickle
    // over a near-infinite duration — and the fit can converge wrong with no other
    // diagnostic. Flag it (analogous to W_NEGATIVE_LAGTIME) and point at a
    // positive-link parameterisation. Engine-agnostic via `active_dose_attr_map()`
    // (covers analytical models too, #394/#395). De-duped per compartment per kind.
    {
        use crate::types::{DoseAttr, DoseEvent, RateMode};
        let attr_map = model.active_dose_attr_map();
        let mut nonpos_dur: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        let mut nonpos_rate: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for s in &population.subjects {
            let mut pk_at_init: Option<crate::types::PkParams> = None;
            for d in &s.doses {
                let (attr, floor) = match d.rate_mode {
                    RateMode::Fixed => continue,
                    RateMode::ModeledDuration => (DoseAttr::Duration, DoseEvent::DURATION_FLOOR),
                    RateMode::ModeledRate => (DoseAttr::Rate, DoseEvent::RATE_FLOOR),
                };
                if let Some(slot) = attr_map.indexed_slot(attr, d.cmt_raw()) {
                    let pk = pk_at_init.get_or_insert_with(|| {
                        // Initial-estimate warning pass: typical values at t=0.
                        (model.pk_param_fn)(&init_params.theta, &zero_eta, &s.covariates, 0.0)
                    });
                    if pk.values[slot] <= floor {
                        match attr {
                            DoseAttr::Duration => nonpos_dur.insert(d.cmt_raw()),
                            DoseAttr::Rate => nonpos_rate.insert(d.cmt_raw()),
                            DoseAttr::F | DoseAttr::Lag => unreachable!("only D/R reach here"),
                        };
                    }
                }
            }
        }
        for cmt in nonpos_dur {
            diags.push(Diagnostic::warning(
                "W_MODELED_DURATION_NONPOSITIVE",
                format!(
                    "Modeled infusion duration D{cmt} (RATE=-2 into compartment \
                     {cmt}) evaluates to ≤ 0 at the initial typical-value point \
                     (eta = 0). A non-positive duration is clamped to {floor:e} to \
                     keep AMT/D finite, which delivers the dose as a bolus-like \
                     spike rather than an infusion — the fit may converge to a \
                     wrong optimum. Use a positive-link parameterisation \
                     (e.g. D{cmt} = exp(...)).",
                    cmt = cmt,
                    floor = DoseEvent::DURATION_FLOOR,
                ),
            ));
        }
        for cmt in nonpos_rate {
            diags.push(Diagnostic::warning(
                "W_MODELED_RATE_NONPOSITIVE",
                format!(
                    "Modeled infusion rate R{cmt} (RATE=-1 into compartment \
                     {cmt}) evaluates to ≤ 0 at the initial typical-value point \
                     (eta = 0). A non-positive rate is clamped to {floor:e} to keep \
                     the implied duration AMT/R finite, which delivers the dose as a \
                     near-zero trickle over a near-infinite duration — the fit may \
                     converge to a wrong optimum. Use a positive-link \
                     parameterisation (e.g. R{cmt} = exp(...)).",
                    cmt = cmt,
                    floor = DoseEvent::RATE_FLOOR,
                ),
            ));
        }
    }

    // EVID=3/4 resets are not honoured on the EKF/SDE path.
    if model.is_sde() {
        let n_reset_sde = population
            .subjects
            .iter()
            .filter(|s| s.has_resets())
            .count();
        if n_reset_sde > 0 {
            diags.push(Diagnostic::warning(
                "W_SDE_RESET",
                format!(
                    "{} subject(s) have EVID=3/4 reset rows with a [diffusion] (SDE) \
                     model. System resets are not yet honoured on the EKF/SDE path — \
                     the resets are ignored and compartment amounts carry through. \
                     Use an ODE or analytical model if resets are required.",
                    n_reset_sde
                ),
            ));
        }
    }

    // Negative typical-value lag time at the initial point (eta = 0).
    if model.has_lagtime() {
        if let Some(first_subj) = population.subjects.first() {
            let zero_eta = vec![0.0_f64; model.n_eta];
            // Negative-lag warning at the initial point: typical values at t=0.
            let pk =
                (model.pk_param_fn)(&init_params.theta, &zero_eta, &first_subj.covariates, 0.0);
            if pk.lagtime() < 0.0 {
                diags.push(Diagnostic::warning(
                    "W_NEGATIVE_LAGTIME",
                    format!(
                        "Lagtime evaluates to {:.4} (< 0) at the initial typical-value \
                         point (eta = 0). Negative lagtimes are physically nonsensical \
                         and are not clamped — consider an exp() or other positive-link \
                         parameterisation.",
                        pk.lagtime()
                    ),
                ));
            }
            // Compartment-indexed `ALAGn` lags (issue #369) live in their own
            // spare slots, so the bare check above never sees them — flag each one
            // that is negative at the typical-value point. Ordered by compartment
            // for deterministic diagnostics.
            if let Some(ode) = &model.ode_spec {
                for cmt in 1..=ode.n_states {
                    let Some(slot) = ode
                        .dose_attr_map
                        .indexed_slot(crate::types::DoseAttr::Lag, cmt)
                    else {
                        continue;
                    };
                    let lag = pk.values.get(slot).copied().unwrap_or(0.0);
                    if lag < 0.0 {
                        diags.push(Diagnostic::warning(
                            "W_NEGATIVE_LAGTIME",
                            format!(
                                "ALAG{cmt} (compartment-{cmt} lag) evaluates to {lag:.4} (< 0) \
                                 at the initial typical-value point (eta = 0). Negative lagtimes \
                                 are physically nonsensical and are not clamped — consider an \
                                 exp() or other positive-link parameterisation."
                            ),
                        ));
                    }
                }
            }
        }
    }

    // Negative per-route absorption lag (`fn(..., lag=L)`) at the initial typical-value
    // point (η = 0). A per-route lag lives on the forcing (its own `lag_slot`), NOT the
    // dose-attribute lag machinery, so `model.has_lagtime()` does not cover it and the
    // block above never sees it — this check stands on its own, gated on a forcing
    // actually carrying a `lag_slot`. Same convention as `W_NEGATIVE_LAGTIME`: warned,
    // not clamped (a negative lag shifts the onset earlier rather than crashing).
    if let Some(ode) = &model.ode_spec {
        if ode.has_route_lag() {
            if let Some(first_subj) = population.subjects.first() {
                let zero_eta = vec![0.0_f64; model.n_eta];
                let pk =
                    (model.pk_param_fn)(&init_params.theta, &zero_eta, &first_subj.covariates, 0.0);
                for f in ode.input_rate.iter().filter(|f| f.lag_slot.is_some()) {
                    let lag = f.route_lag(&pk.values);
                    if lag < 0.0 {
                        diags.push(Diagnostic::warning(
                            "W_NEGATIVE_LAGTIME",
                            format!(
                                "Per-route absorption lag (a `{:?}` route on compartment {}) \
                                 evaluates to {lag:.4} (< 0) at the initial typical-value point \
                                 (eta = 0). Negative lagtimes are physically nonsensical and are \
                                 not clamped — consider an exp() or other positive-link \
                                 parameterisation.",
                                f.kind,
                                f.cmt + 1
                            ),
                        ));
                    }
                }
            }
        }
    }

    // Analytic absorption flip-flop diagnostic (#634 review finding 3; IG #790). The
    // transit / IG closed forms converge only when the disposition rate is below the
    // tilting abscissa — `KTR = (n+1)/mtt` (transit) / `1/(2·MAT·CV²)` (IG), with the
    // *fast* macro-rate `α` for 2-cpt and `ke = CL/V` for 1-cpt. Outside that domain the
    // closed form returns an identically-zero profile — parameter-dependent, so
    // `check_absorption_closed_form_support` can't reject it up front, and with a
    // proportional error model an all-zero IPRED silently degenerates the likelihood with
    // no other diagnostic. Evaluated on typical values (η = 0) per subject so a covariate
    // pushing the typical profile into the flip-flop regime is caught. Reported once,
    // naming the first affected subject. Only a **twin-carrying** model is warned here — it
    // is auto-rerouted per evaluation (`pk::effective_model_for_eval`), so the closed form's
    // zero never reaches the objective and this is an informational heads-up. A **twin-less**
    // flip-flop model (lagtime / `f` / user `[odes]`) has nothing to reroute to and is a hard
    // error instead (`check_absorption_flip_flop_no_twin`), rejected at
    // fit / predict / simulate / `ferx check`.
    if matches!(
        model.pk_model,
        PkModel::OneCptTransit | PkModel::TwoCptTransit | PkModel::OneCptIg | PkModel::TwoCptIg
    ) && model.absorption_ode_equivalent.is_some()
    {
        let is_transit = matches!(
            model.pk_model,
            PkModel::OneCptTransit | PkModel::TwoCptTransit
        );
        for subject in &population.subjects {
            // Reuse the exact runtime reroute decision (validity + domain + confluent), so
            // this heads-up fires precisely when a flip-flop reroute happens.
            if !crate::pk::absorption_flip_flop_at(model, subject, &init_params.theta, &zero_eta) {
                continue;
            }
            let pk = (model.pk_param_fn)(&init_params.theta, &zero_eta, &subject.covariates, 0.0);
            let (cl, v1) = (pk.cl(), pk.v());
            // The fast disposition rate (2-cpt macro-rate `α`, else `ke`) and the tilting
            // abscissa, for the informational message.
            let disp_rate = match model.pk_model {
                PkModel::TwoCptTransit | PkModel::TwoCptIg => {
                    crate::sens::two_cpt::macro_rates_g::<f64>(cl, v1, pk.q(), pk.v2()).0
                }
                _ => cl / v1,
            };
            let (abscissa, abscissa_label, code, ode_fn, params) = if is_transit {
                (
                    (pk.n_transit() + 1.0) / pk.mtt(),
                    "transit rate KTR = (n+1)/mtt",
                    "W_TRANSIT_FLIP_FLOP",
                    "transit",
                    "MTT / CL",
                )
            } else {
                (
                    1.0 / (2.0 * pk.mat() * pk.cv2()),
                    "IG abscissa 1/(2·MAT·CV²)",
                    "W_IG_FLIP_FLOP",
                    "igd",
                    "MAT / CV² / CL",
                )
            };
            // `absorption_flip_flop_at` also returns true on the 2-cpt confluent-eigenvalue
            // corner (`|α−β|<1e-12`), where `disp_rate = α < abscissa`. That corner is
            // physically unreachable for positive params, but skip the flip-flop-worded
            // heads-up there anyway so its "disposition rate ≥ abscissa" claim can never
            // misreport (the reroute to the ODE twin is still correct either way).
            if disp_rate < abscissa {
                continue;
            }
            // Auto-handled: the flip-flop reroute (`pk::effective_model_for_eval`) evaluates
            // the ODE twin for these parameters, so the profile is correct — an informational
            // heads-up, not an error. (A twin-less flip-flop model is rejected up front by
            // `check_absorption_flip_flop_no_twin`.)
            let msg = format!(
                "{} disposition rate ({:.4}) ≥ {} ({:.4}) at typical values (subject {}): the \
                 flip-flop regime, outside the analytic absorption closed form's convergence \
                 domain. ferx automatically evaluates the equivalent ODE {} model for such \
                 parameters (correct, but slower than the closed form) — check the {} starting \
                 estimates if the flip-flop is unexpected.",
                model.pk_model.canonical_name(),
                disp_rate,
                abscissa_label,
                abscissa,
                subject.id,
                ode_fn,
                params
            );
            diags.push(Diagnostic::warning(code, msg));
            break;
        }
    }

    // An `L2` column that nothing consumes (#830). `L2` is a reserved level-2
    // grouping data item, so the reader always treats it as the block_sigma
    // pairing id and never as a covariate. If the model declares no block_sigma
    // correlated residual, an `L2` column is inert — and if the user actually
    // meant it as a covariate it has silently vanished. Surface that.
    if model.residual_correlations.is_empty()
        && population.subjects.iter().any(|s| !s.obs_l2.is_empty())
    {
        diags.push(Diagnostic::warning(
            "W_L2_UNUSED",
            "The dataset has an `L2` column but the model declares no block_sigma \
             correlated residual, so `L2` is read as the reserved level-2 grouping \
             id and is not available as a covariate. Rename the column if you \
             intended it as a covariate."
                .to_string(),
        ));
    }

    // block_sigma cross correlations with an ambiguous (time, occasion) fallback
    // pairing (#830). Without an `L2` grouping column the reader pairs co-temporal
    // correlated rows one-to-one in file row order. That is unambiguous for the
    // standard two-row layout (one total + one unbound per draw), but when a
    // (time, occasion) group holds a row that could correlate with two or more
    // others — e.g. replicate assays written as total1, unbound1, total2, unbound2
    // — the resulting cross covariances depend on the physical CSV row order.
    // Point the user at an explicit `L2` column, which makes the grouping
    // order-independent.
    if !model.residual_correlations.is_empty() {
        use crate::stats::residual_error::loadings_share_correlation;
        let n_sigma = init_params.sigma.values.len();
        let corrs = &model.residual_correlations;
        let n_ambiguous = population
            .subjects
            .iter()
            // A subject that already carries any L2 id is explicitly grouped —
            // its pairing does not depend on row order.
            .filter(|s| !s.obs_l2.iter().any(|&id| id != 0))
            .filter(|s| {
                let loads: Vec<Vec<(usize, f64)>> = s
                    .obs_cmts
                    .iter()
                    .map(|&cmt| model.error_spec.sigma_loadings(cmt, 1.0, n_sigma))
                    .collect();
                // Group observation indices by (time, occasion) and flag any group
                // where a row structurally shares a correlation with ≥2 others —
                // the case where greedy row-order pairing is ambiguous.
                let n = s.obs_times.len();
                let key = |j: usize| {
                    (
                        s.obs_times.get(j).copied().unwrap_or(0.0).to_bits(),
                        s.occasions.get(j).copied().unwrap_or(0),
                    )
                };
                (0..n).any(|j| {
                    if loads.get(j).is_none_or(|l| l.is_empty()) {
                        return false;
                    }
                    let degree = (0..n)
                        .filter(|&k| k != j && key(k) == key(j))
                        .filter(|&k| {
                            loads
                                .get(k)
                                .zip(loads.get(j))
                                .is_some_and(|(lk, lj)| loadings_share_correlation(lj, lk, corrs))
                        })
                        .count();
                    degree >= 2
                })
            })
            .count();
        if n_ambiguous > 0 {
            diags.push(Diagnostic::warning(
                "W_BLOCK_SIGMA_L2_ORDER",
                format!(
                    "{} subject(s) have a block_sigma correlated residual whose \
                     co-temporal rows can pair more than one way, and the dataset \
                     has no L2 grouping column — the cross covariances are paired \
                     one-to-one in CSV row order, so reordering rows would change \
                     the fit. Add an explicit `L2` column giving each correlated \
                     unit its own id to make the pairing order-independent.",
                    n_ambiguous
                ),
            ));
        }
    }

    diags.extend(additive_init_scale_warnings(model, population, init_params));

    diags
}

/// Warn when a `combined` error model's **additive** SD initial estimate is
/// negligible relative to the scale of the observations it covers.
///
/// With a combined variance `Var(f) = (f·σ_prop)² + σ_add²`, an additive init
/// orders of magnitude below the data starts the fit in a near-pure-proportional
/// regime where low-magnitude observations receive enormous `1/Var` weight. On
/// multimodal / over-parameterised problems the optimizer can then settle in a
/// local minimum where the additive term stays collapsed (σ_add ≈ 0) and the
/// proportional term inflates — the *worse* basin. The cyclophosphamide
/// parent→metabolite fit is the canonical case: additive variance seeded at 0.5
/// traps at ≈0.94, whereas the global optimum is ≈1878 (~50 OFV points better);
/// both ferx and NONMEM miss it from that start, Pumas found it.
///
/// This is a **pre-fit heuristic on the initial estimate only** — it never
/// changes the value (per the warn-only design), it advises a larger start.
/// The threshold is deliberately conservative (additive SD below 1% of the data
/// scale): a genuinely small-but-intended additive floor of a few % of the data
/// does not trip it; only an essentially-zero start does.
fn additive_init_scale_warnings(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
) -> Vec<Diagnostic> {
    use crate::types::{EndpointError, ErrorModel, ErrorSpec, SigmaType};
    use std::collections::HashMap;

    let mut diags = Vec::new();

    // (additive global sigma index, endpoint dispatch key or None for `Single`,
    // scope label). For `Combined` the additive sigma is the 2nd sigma
    // (residual_variance uses sigma[0]=prop, sigma[1]=add); for `Single` the
    // global sigma vector is in that order, for `PerCmt`/`Selected` the
    // endpoint's `sigma_idx[1]`. `PerCmt` keys are 1-based CMTs; `Selected` keys
    // are 0-based covariate-branch indices — labelled `endpoint=` not `CMT=` so
    // the message doesn't misname a branch as a compartment.
    let mut combined_endpoints: Vec<(usize, Option<usize>, String)> = Vec::new();
    let mut push_endpoints = |map: &HashMap<usize, EndpointError>, label: &str| {
        for (&key, ee) in map {
            if ee.error_model == ErrorModel::Combined {
                if let Some(pos) = ee
                    .error_model
                    .sigma_types()
                    .iter()
                    .position(|t| *t == SigmaType::Additive)
                {
                    if let Some(&idx) = ee.sigma_idx.get(pos) {
                        combined_endpoints.push((idx, Some(key), format!("{label}={key}")));
                    }
                }
            }
        }
    };
    match &model.error_spec {
        ErrorSpec::Single(ErrorModel::Combined) => {
            combined_endpoints.push((1, None, "all observations".to_string()))
        }
        ErrorSpec::Single(_) => {}
        ErrorSpec::PerCmt(map) => push_endpoints(map, "CMT"),
        ErrorSpec::Selected { endpoints, .. } => push_endpoints(endpoints, "endpoint"),
    }
    // Deterministic order: HashMap iteration (PerCmt/Selected) is unordered.
    combined_endpoints.sort_unstable();

    for (add_idx, key, scope) in combined_endpoints {
        let Some(&add_sd_init) = init_params.sigma.values.get(add_idx) else {
            continue;
        };
        // |DV| of the observations this endpoint covers.
        let mut mags: Vec<f64> = Vec::new();
        for subject in &population.subjects {
            for (j, &obs) in subject.observations.iter().enumerate() {
                let covered = match key {
                    None => true,
                    Some(k) => model.error_spec.obs_key(subject, j) == k,
                };
                if covered {
                    let v = obs.abs();
                    if v.is_finite() {
                        mags.push(v);
                    }
                }
            }
        }
        let sigma_name = init_params
            .sigma
            .names
            .get(add_idx)
            .cloned()
            .unwrap_or_else(|| format!("SIGMA({})", add_idx + 1));
        if let Some(d) = additive_init_scale_check(&scope, &sigma_name, add_sd_init, &mut mags) {
            diags.push(d);
        }
    }

    diags
}

/// Pure decision for [`additive_init_scale_warnings`]: given one combined
/// endpoint's additive SD initial estimate and the `|DV|` magnitudes of the
/// observations it covers, return a warning iff the additive start is negligible
/// (below [`NEGLIGIBLE_ADD_FRACTION`]) relative to the data scale (median `|DV|`).
///
/// `mags` is taken by `&mut` because it is sorted in place to find the median;
/// the caller does not reuse it afterwards. Returns `None` for an empty/all-zero
/// data scale (nothing to compare against) or a non-negligible additive start.
fn additive_init_scale_check(
    scope: &str,
    sigma_name: &str,
    add_sd_init: f64,
    mags: &mut [f64],
) -> Option<Diagnostic> {
    if mags.is_empty() {
        return None;
    }
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = mags.len() / 2;
    let data_scale = if mags.len() % 2 == 1 {
        mags[mid]
    } else {
        0.5 * (mags[mid - 1] + mags[mid])
    };
    if data_scale <= 0.0 || add_sd_init >= NEGLIGIBLE_ADD_FRACTION * data_scale {
        return None;
    }
    Some(Diagnostic::warning(
        "W_ADDITIVE_INIT_SCALE",
        format!(
            "Combined error model on {scope}: additive SD initial estimate \
             '{sigma_name}' = {add_sd_init:.4} is negligible (< {pct:.0}%) \
             relative to the data scale (median |DV| = {data_scale:.4}). A \
             near-zero additive start can trap the fit in a local minimum where \
             the additive term collapses and the proportional term inflates — \
             the worse basin on multimodal problems. If the fit converges with \
             this additive variance near zero, retry from a larger additive \
             initial estimate (e.g. SD ≈ {suggest:.4}, ~10% of the data scale \
             — median |DV| / 10).",
            pct = NEGLIGIBLE_ADD_FRACTION * 100.0,
            suggest = 0.1 * data_scale,
        ),
    ))
}

/// Additive SD init counts as "negligible" below this fraction of the data scale
/// (median `|DV|`) — deliberately conservative so a genuinely small-but-intended
/// additive floor of a few % of the data does not trip the warning.
const NEGLIGIBLE_ADD_FRACTION: f64 = 0.01;

#[cfg(test)]
#[path = "tests/additive_init_scale_tests.rs"]
mod additive_init_scale_tests;

#[cfg(test)]
#[path = "tests/residual_magnitude_tests.rs"]
mod residual_magnitude_tests;

#[cfg(test)]
#[path = "tests/kappa_weight_tests.rs"]
mod kappa_weight_tests;

/// Feature-presence (data-independent) *warning*-level checks for experimental
/// features (issue #175). Stochastic differential equations and neural-network
/// components are classified `experimental` in the Feature Maturity docs: tested
/// only on a handful of toy examples. We emit a warning whenever they are used
/// so results are applied with appropriate caution.
///
/// Kept separate from [`check_model_data_warnings`] because these depend only on
/// the compiled `model`, not on the dataset — so `ferx check model.ferx` (no
/// `--data`) and `fit()` both surface them. Non-fatal: `fit()` pushes the
/// messages into `FitResult.warnings`; `ferx check` reports them as warnings.
pub fn check_experimental_features(model: &CompiledModel) -> Vec<Diagnostic> {
    let mut diags = Vec::new();

    if model.is_sde() {
        diags.push(Diagnostic::warning(
            "W_EXPERIMENTAL_SDE",
            "Stochastic differential equations ([diffusion] / Extended Kalman \
             Filter) are an EXPERIMENTAL feature: validated only on a small set \
             of toy examples, with estimator support limited to FOCE/FOCEI. \
             Standard errors and convergence behaviour are not yet proven across \
             diverse datasets — validate results carefully before relying on \
             them. See the Feature Maturity page in the documentation.",
        ));
    }
    #[cfg(feature = "nn")]
    if !model.covariate_nns.is_empty() {
        diags.push(Diagnostic::warning(
            "W_EXPERIMENTAL_NN",
            "Neural-network model components ([covariate_nn] / deep compartment \
             models) are an EXPERIMENTAL feature: validated only on a small set \
             of toy examples. Standard errors for network weights are not \
             reliable and the syntax may still change — validate results \
             carefully before relying on them. See the Feature Maturity page in \
             the documentation.",
        ));
    }

    diags
}

/// Map a free-text parse **warning** to its check-report code.
///
/// Warning sibling of [`parse_error_to_diagnostic`], and split out of
/// `validate_model_file`'s loop for the same reason: the mapping is a pure
/// string→code function, so it is worth testing directly rather than only
/// through a model file on disk. Each arm matches a sentinel the emitting site
/// carries verbatim (the `W_…` token, or a pinned phrase), so a reworded warning
/// falls through to the generic `W_PARSE` rather than silently mislabelling —
/// which is why the codes are asserted at both ends.
fn parse_warning_to_code(w: &str) -> &'static str {
    if w.contains("declared in [parameters] but not referenced") {
        "W_UNUSED_PARAM"
    } else if w.contains("W_DERIVED_COVARIATE_SHADOW") {
        "W_DERIVED_COVARIATE_SHADOW"
    } else if w.contains("W_DERIVED_STEP_IGNORED") {
        "W_DERIVED_STEP_IGNORED"
    } else if w.contains("W_ABSORPTION_TWIN_DECLINED") {
        "W_ABSORPTION_TWIN_DECLINED"
    } else {
        "W_PARSE"
    }
}

/// Map a free-text parser error string to a single structured [`Diagnostic`].
/// Recognises the `"Missing [X] block"` shape (→ `E_MISSING_BLOCK`, with the block
/// name attached), the `--features nn` gate (→ `E_NN_FEATURE_DISABLED`), the
/// dose-attribute double use (→ `E_DOSE_ATTR_DOUBLE_USE`, #993), the
/// single-endpoint sigma order mismatch (→ `E_SIGMA_ORDER_MISMATCH`, #1001), and
/// the block-header shapes (→ `E_UNKNOWN_BLOCK` / `E_DEPRECATED_BLOCK` /
/// `E_BLOCK_INSTANCE_NAME` / `E_BLOCK_FEATURE_DISABLED`, #1040); everything else is
/// a generic `E_PARSE`. Each shape is matched on a sentinel the emitting site is
/// pinned to by a test, so a reworded message cannot silently fall through to the
/// catch-all.
fn parse_error_to_diagnostic(err: &str) -> Diagnostic {
    if let Some(rest) = err.strip_prefix("Missing [") {
        if let Some(end) = rest.find(']') {
            let block = &rest[..end];
            return Diagnostic::error("E_MISSING_BLOCK", err.to_string()).with_block(block);
        }
    }
    if err.contains("[covariate_nn]") && err.contains("--features nn") {
        return Diagnostic::error("E_NN_FEATURE_DISABLED", err.to_string())
            .with_block("covariate_nn");
    }
    // #993/#1004: a dose attribute applied by the engine *and* read on the
    // prediction path. Worth its own code rather than a generic `E_PARSE` — the
    // fix is mechanical (drop the read, rename, or drop the `pk(...)` mapping),
    // so a consumer (`ferxtranslate`, `ferx-r`) can act on it rather than
    // surfacing prose. Two sentinels, one per remediation clause
    // `check_dose_attr_double_use` emits: the ODE wording ends in the
    // reserved-name clause, the analytical wording (#1004) in the
    // drop-the-mapping clause — renaming is the wrong repair there, since the
    // `f=`/`lagtime=` mapping follows the parameter.
    //
    // No `.with_block()`: the message already opens with the block it came from
    // (`[odes]: ` / `[scaling]: ` / `[adaptive_dosing]: `) like every other
    // block-scoped parse error, and the renderer prefixes the block it is given —
    // setting both prints it twice.
    if err.contains("reserved dose-attribute name")
        || err.contains("mapping from the `pk(...)` call")
    {
        return Diagnostic::error("E_DOSE_ATTR_DOUBLE_USE", err.to_string());
    }
    // #1001: a single-endpoint `[error_model]` naming its sigmas in an order other
    // than the `[parameters]` declaration order — which used to fit silently against
    // the leading slots instead. Its own code for the same reason as #993: the remedy
    // is mechanical (reorder one list), so a consumer can offer it rather than
    // reprinting prose. The sentinel is the rule clause the emitting arm in
    // `build_error_spec` always carries.
    //
    // Matched on a substring rather than a prefix, so it must not overlap the
    // block-header shapes below: this message opens with `[error_model] argument
    // …`, which none of their `strip_prefix` sentinels match, and none of their
    // messages carries this clause. The two families are independent, so their
    // relative order here is not load-bearing.
    //
    // No `.with_block()`: the message already opens with `[error_model] `, and the
    // renderer prefixes whatever block it is given — setting both prints it twice.
    if err.contains("consumed positionally") {
        return Diagnostic::error("E_SIGMA_ORDER_MISMATCH", err.to_string());
    }
    // #1040: the block-header shapes the parser used to drop silently.
    // `check_block_names` writes each offending header as ``[name] (line N)``,
    // so the block / line the check report wants come straight out of the
    // message — this path has no `ParsedModel` to read `block_lines` from,
    // since a parse error is terminal.
    if let Some(rest) = err.strip_prefix("Unknown block `[") {
        let mut d = Diagnostic::error("E_UNKNOWN_BLOCK", String::new());
        if let Some(near) =
            block_type(rest).and_then(|b| crate::parser::model_parser::nearest_block_name(&b))
        {
            d = d.with_suggestion(format!("did you mean `[{near}]`?"));
        }
        return locate(d, err, rest);
    }
    if let Some(rest) = err.strip_prefix("Deprecated block `[") {
        return locate(
            Diagnostic::error("E_DEPRECATED_BLOCK", String::new()),
            err,
            rest,
        );
    }
    if let Some(rest) = err.strip_prefix("Block `[") {
        // Match each shape on its own sentinel rather than letting one be the
        // `else` of the other: a future ``Block `[…]`` message that is neither
        // must fall through to `E_PARSE`, not be mis-coded as the last branch.
        let code = if err.contains("instance name") {
            "E_BLOCK_INSTANCE_NAME"
        } else if err.contains("requires building ferx-core with `--features") {
            "E_BLOCK_FEATURE_DISABLED"
        } else {
            return Diagnostic::error("E_PARSE", err.to_string());
        };
        return locate(Diagnostic::error(code, String::new()), err, rest);
    }
    Diagnostic::error("E_PARSE", err.to_string())
}

#[cfg(test)]
#[path = "tests/block_name_diagnostic_tests.rs"]
mod block_name_diagnostic_tests;

/// Give a block-header diagnostic its message plus whatever location the
/// message text carries.
///
/// `err` is the whole parser error; `rest` is its tail, starting just after the
/// opening ``` `[ ```. The block type always transfers to `block`.
///
/// The line is only lifted into `line` when the message names exactly **one**
/// header — the structured field points at a single place, so promoting the
/// first of several would silently mislocate the rest. In that unambiguous case
/// the ` (line N)` is also *removed* from the message, since the renderer
/// already prints `block:line` ahead of it and repeating it reads as
/// `fit_option:16: … (line 16)`. With several headers the message keeps every
/// parenthetical (each finding needs its own) and no `line` is attached.
fn locate(mut d: Diagnostic, err: &str, rest: &str) -> Diagnostic {
    if let Some(b) = block_type(rest) {
        d = d.with_block(b);
    }
    let mut lines = line_spans(err);
    match (lines.next(), lines.next()) {
        (Some((span, n)), None) => {
            let mut message = String::with_capacity(err.len());
            message.push_str(&err[..span.0]);
            message.push_str(&err[span.1..]);
            d.message = message;
            d.with_line(n)
        }
        _ => {
            d.message = err.to_string();
            d
        }
    }
}

/// The block *type* out of the tail of a `check_block_names` message — `rest`
/// starts just after the opening ``` `[ ```, e.g. ``covariate_nn X]` (line 7)
/// requires …``. An instance name, when present, is dropped: `block` names the
/// type, matching every other block-scoped diagnostic.
fn block_type(rest: &str) -> Option<String> {
    rest.find(']')
        .and_then(|end| rest[..end].split_whitespace().next())
        .filter(|b| !b.is_empty())
        .map(str::to_string)
}

/// Every ` (line N)` in `msg`, as `((start, end), N)` byte spans over `msg`.
/// The leading space is part of the span so removing one leaves no double
/// space behind.
fn line_spans(msg: &str) -> impl Iterator<Item = ((usize, usize), usize)> + '_ {
    msg.match_indices(" (line ").filter_map(move |(at, pat)| {
        let digits_at = at + pat.len();
        let digits = &msg[digits_at..];
        let end = digits.find(')')?;
        let n = digits[..end].parse::<usize>().ok()?;
        Some(((at, digits_at + end + 1), n))
    })
}

/// Validate a model file (and optionally a dataset) **without fitting**.
///
/// Runs the parser plus every data-independent and data-dependent check,
/// collecting *all* findings into a [`CheckReport`] (rather than stopping at the
/// first, as `fit()` does). This is the engine behind the `ferx check` CLI
/// command and is the fast `author → diagnose → fix` loop for tools and agents.
///
/// When neither `data_path` nor the model's own `[data]` block (#690) resolve
/// to a path, only parse/structural and model/option compatibility validation
/// runs (no data is read). Otherwise the CSV is read and the covariate /
/// per-CMT / steady-state / lag-time checks run as well. An explicit
/// `data_path` overrides the model's `[data]` block, with a warning
/// diagnostic when the two differ (see [`resolve_data_path`]).
pub fn validate_model_file(model_path: &str, data_path: Option<&str>) -> CheckReport {
    use crate::parser::model_parser::parse_full_model_file;

    let model_name = Path::new(model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();

    // 1. Parse. A parse failure is terminal — without an AST there is nothing
    //    further to validate, so return a report carrying just that diagnostic.
    let parsed = match parse_full_model_file(Path::new(model_path)) {
        Ok(p) => p,
        Err(e) => {
            let data = data_path.map(|s| s.to_string());
            return CheckReport::new(model_name, data, vec![parse_error_to_diagnostic(&e)]);
        }
    };

    // Resolve which dataset (if any) to check against: an explicit
    // `data_path` wins over the model's `[data]` block; absence of both just
    // means the data-dependent checks below are skipped, so any `Err` here
    // (neither given) is not itself a diagnostic.
    let (data_path, data_path_warning) =
        match resolve_data_path(parsed.data_path.as_deref(), data_path) {
            Ok((p, w)) => (Some(p), w),
            Err(_) => (None, None),
        };
    let data_path = data_path.as_deref();
    let data = data_path.map(|s| s.to_string());

    let mut diags: Vec<Diagnostic> = Vec::new();
    if let Some(w) = data_path_warning {
        diags.push(Diagnostic::warning("W_DATA_PATH_OVERRIDE", w));
    }

    // 2a. Parse-time warnings collected during parsing (unused parameters,
    //     mu-referencing diagnostics, etc.). Each warning embeds its own block
    //     context in the message text; we use W_PARSE as the generic code here
    //     rather than a narrower code that would mislabel unrelated warnings.
    for w in &parsed.model.parse_warnings {
        diags.push(Diagnostic::warning(parse_warning_to_code(w), w.clone()));
    }

    // 2b. Model / estimation-option compatibility (data-independent): catches
    //    method/model combinations that `fit()` rejects before fitting, so a
    //    clean check and a fit agree. Uses the parsed `[fit_options]`, mirroring
    //    what the CLI fit path (`run_model_with_data`) passes to `fit()`.
    diags.extend(check_model_options(&parsed.model, &parsed.fit_options));

    // 2c. Experimental-feature notices (data-independent): these depend only on
    //    the model, so they surface from `ferx check model.ferx` even without a
    //    `--data` file.
    diags.extend(check_experimental_features(&parsed.model));

    // 2d. Undeclared random effects, *without* a dataset. With data present this
    //    is an `E_ETA_NOT_DECLARED` error from `check_covariates` below (we know
    //    the column is absent); without one we cannot know whether the data
    //    happens to carry an `ETA_CL` column, so it can only be a warning — but
    //    it must still be said. Before #989 a model that dropped its whole
    //    `omega` block was rejected at parse time; now it parses, and a bare
    //    `ferx check model.ferx` would otherwise report "no errors" for a model
    //    whose etas silently became covariate lookups.
    if data_path.is_none() {
        let undeclared = undeclared_random_effect_names(&parsed.model);
        if !undeclared.is_empty() {
            diags.push(
                Diagnostic::warning(
                    "W_ETA_NOT_DECLARED",
                    undeclared_random_effect_message(&undeclared),
                )
                .with_block("parameters")
                .with_suggestion(format!(
                    "re-run with --data to confirm whether the data carries the `{}` column",
                    undeclared[0]
                )),
            );
        }
    }

    // 3. Data-dependent checks (only when a dataset is supplied). Read through
    //    the same covariate-aware chokepoint the fit uses, so `ferx check` and
    //    `fit()` apply identical covariate validation (declared columns present
    //    + numeric). A covariate-validation failure surfaces as the matching
    //    diagnostic rather than a generic read error.
    if let Some(path) = data_path {
        let iov_col = parsed.fit_options.iov_column.as_deref();
        match read_population_for(
            &parsed.model,
            &parsed.covariate_decls,
            path,
            None,
            iov_col,
            None,
            &parsed.column_map,
        ) {
            Ok((population, _table)) => {
                // Surface datareader warnings (ADDL missing II, IOV OCC missing)
                // into the check report so `ferx check` sees the same findings as `fit()`.
                // Through the shared filter, or the two would disagree on exactly the
                // warnings `fit()` suppresses — which is the agreement this block exists
                // to keep.
                for w in population
                    .warnings
                    .iter()
                    .filter(|w| !reader_warning_suppressed(&parsed.model, w))
                {
                    let code = if w.starts_with("W_ADDL_MISSING_II") {
                        "W_ADDL_MISSING_II"
                    } else if w.starts_with("W_IOV_OCC_MISSING") {
                        "W_IOV_OCC_MISSING"
                    } else {
                        "W_DATA"
                    };
                    diags.push(Diagnostic::warning(code, w.clone()));
                }
                diags.extend(check_model_data_rule(
                    &parsed.model,
                    &population,
                    &parsed.fit_options.iov_occasion,
                ));
                let init_params = parsed.model.default_params.clone();
                diags.extend(check_model_data_warnings(
                    &parsed.model,
                    &population,
                    &init_params,
                ));
                // Surface the structural absorption-closed-form rejects (IOV / SS / CMT≠1 /
                // infusion / reset) so a clean `ferx check` and a `fit()` agree on transit /
                // IG models — previously only `fit()` reported these (#776 review, IG #790).
                // Model-family-specific code so an IG model reads `E_IG_*` not `E_TRANSIT_*`.
                let is_ig = matches!(parsed.model.pk_model, PkModel::OneCptIg | PkModel::TwoCptIg);
                if let Some(e) = check_absorption_closed_form_support(&parsed.model, &population) {
                    let code = if is_ig {
                        "E_IG_UNSUPPORTED"
                    } else {
                        "E_TRANSIT_UNSUPPORTED"
                    };
                    diags.push(Diagnostic::error(code, e));
                }
                // Twin-less flip-flop is a hard error, not a warning (#776): surface it the
                // same way `fit()` does, so `ferx check` catches it up front.
                if let Some(e) = check_absorption_flip_flop_no_twin(
                    &parsed.model,
                    &population,
                    &init_params.theta,
                ) {
                    let code = if is_ig {
                        "E_IG_FLIP_FLOP"
                    } else {
                        "E_TRANSIT_FLIP_FLOP"
                    };
                    diags.push(Diagnostic::error(code, e));
                }
            }
            Err(e) => {
                diags.push(covariate_read_diagnostic(&e, path));
            }
        }
    }

    // 4. Attach block-level line numbers to any diagnostic that named a block.
    for d in &mut diags {
        if d.line.is_none() {
            if let Some(block) = &d.block {
                if let Some(&ln) = parsed.block_lines.get(block) {
                    d.line = Some(ln);
                }
            }
        }
    }

    CheckReport::new(model_name, data, diags)
}

/// Validate `model.output_columns` against known quantities, emitting
/// `W_OUTPUT_DUPLICATE` and `E_OUTPUT_UNKNOWN_COLUMN` diagnostics.
pub fn validate_output_columns(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    let derived_names: Vec<&str> = model
        .derived_exprs
        .iter()
        .map(|s| s.name.as_str())
        .collect();
    let cov_names = &population.covariate_names;

    for col in &model.output_columns {
        // Already in mandatory minimum, or an ETA (reported via ebe_etas, not sdtab)?
        let is_eta = model.eta_names.iter().any(|e| e.eq_ignore_ascii_case(col));
        if OUTPUT_MANDATORY.iter().any(|m| m.eq_ignore_ascii_case(col)) || is_eta {
            let msg = if is_eta {
                // sdtab is per-observation only; per-subject EBEs live in
                // `ebe_etas` on the R side, so an ETA can't be an sdtab column.
                format!(
                    "[output] column `{col}` is an ETA estimate, reported via `ebe_etas` \
                     rather than as an sdtab column; the declaration is ignored"
                )
            } else {
                format!(
                    "[output] column `{col}` is already written to sdtab automatically; \
                     the declaration is ignored"
                )
            };
            diags.push(Diagnostic::warning("W_OUTPUT_DUPLICATE", msg));
            continue;
        }
        // Valid if it's a covariate, indiv param, or derived name. Synthetic
        // readout parameters (`__ferx_ro_*`, #486) are internal — not user-requestable.
        let known = cov_names.iter().any(|c| c.eq_ignore_ascii_case(col))
            || model.indiv_param_names.iter().any(|p| {
                !crate::parser::model_parser::is_synthetic_readout_param(p)
                    && p.eq_ignore_ascii_case(col)
            })
            || derived_names.iter().any(|d| d.eq_ignore_ascii_case(col));
        if !known {
            let mut candidates: Vec<&str> = cov_names.iter().map(|s| s.as_str()).collect();
            candidates.extend(
                model
                    .indiv_param_names
                    .iter()
                    .filter(|p| !crate::parser::model_parser::is_synthetic_readout_param(p))
                    .map(|s| s.as_str()),
            );
            candidates.extend(derived_names.iter().copied());
            candidates.extend(OUTPUT_MANDATORY.iter().copied());
            diags.push(Diagnostic::error(
                "E_OUTPUT_UNKNOWN_COLUMN",
                format!(
                    "[output] column `{col}` is not recognised as a covariate, individual \
                     parameter, or derived expression. Known: {}",
                    candidates.join(", ")
                ),
            ));
        }
    }
    diags
}
