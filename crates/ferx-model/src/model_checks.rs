//! Model and data checks the estimators run before a fit, a covariance step or SIR:
//! dosing compartments, endpoint routing, the ODE-method/CTMM/survival guards and the
//! nested-level index columns. Crate-level, not in `api`, so `estimation` uses them
//! without depending on it (#3041); `api::validation` re-exports them.
use crate::diagnostics::Diagnostic;
use crate::io::datareader::{CovariateReads, MissingDvPolicy, ObsRouting};
use crate::levels::{level_index_finding, LevelIndexFinding};
use crate::types::*;
use std::collections::HashMap;

/// Mandatory sdtab column names that are always written — declaring them in
/// `output` is allowed but produces a W_OUTPUT_DUPLICATE warning.
pub const OUTPUT_MANDATORY: &[&str] = &[
    "ID", "TIME", "DV", "CENS", "OCC", "CMT", "PRED", "IPRED", "CWRES", "IWRES", "NPDE", "NPD",
    "EPRED", "PWRES", "EWRES", "EBE_OFV", "N_OBS", "TAFD", "TAD",
];

/// `(block, synthesized index column)` of every **bound** level block whose column the
/// model reads. An unbound block's column is not here: an unbound block is
/// `check_unbound_theta_levels`'s to report, before any population is looked at.
pub fn bound_level_columns(model: &CompiledModel) -> Vec<(&str, String)> {
    let blocks = model.theta_blocks();
    blocks
        .level_blocks()
        .iter()
        .map(|d| d.name())
        .filter(|name| !blocks.unbound_level_blocks().iter().any(|u| u == name))
        .map(|name| (name, crate::parser::model_parser::level_index_column(name)))
        .filter(|(_, column)| model.referenced_covariates.iter().any(|c| c == column))
        .collect()
}

/// Which entry point met a population never bound for a bound level block — it
/// picks the binder [`check_level_index_columns`] names (#1647).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelDataEntry {
    /// `predict_diag`, `predict_survival`, `predict_categorical`, `compute_npde_npd` and
    /// the `simulate*` / adaptive entry points: they run a θ that is already laid out, so
    /// the population must be indexed with the levels that θ was laid out for.
    Run,
    /// `fit()` (and `ferx check`, which binds before it gets here): it estimates θ, so
    /// the model is laid out afresh on the population's own levels.
    Fit,
    /// `run_covariance` / `run_sir` on a fit with no current fingerprint to compare the
    /// population against (#1793): they run the fit's θ, so the binder is `Run`'s, but
    /// a population with levels the fit lacks is not the one it was given.
    PostHoc,
}

/// A **bound** `theta NAME[...]` block handed a population that was never bound for
/// it (#1647): no synthesized index column, so every record would gather `NaN`
/// (typically "fit, then predict or simulate new data"). The binder named depends
/// on `entry`, since only a fit lays θ out afresh:
///
/// | Entry | Binder named |
/// |---|---|
/// | `Run`, `PostHoc` | `bind_from_fit` with the bindings the θ was laid out on (a fit's `data_bindings`, or the model's own) |
/// | `Fit` | `bind_theta_levels` on a fresh parse |
///
/// A population that carries the column is checked record by record against the
/// model's own level table (#1762, [`level_index_finding`]). One diagnostic per
/// block, the first cell that applies:
///
/// | Population | Code | Remedy |
/// |---|---|---|
/// | no subject carries the index | `E_THETA_LEVELS_DATA_UNBOUND`, "never bound" | the binder above |
/// | some subjects carry none | `E_THETA_LEVELS_DATA_UNBOUND`, "`k` of `n` subjects" | the binder above |
/// | an index without its level column | `E_THETA_LEVELS_DATA_MISMATCH` | none: the level is unknown |
/// | levels the model's table lacks | `E_THETA_LEVELS_DATA_MISMATCH`, every one listed | `Run`: a θ from a fit of a model bound on this population; `Fit`: the `Fit` binder; `PostHoc`: the population the fit was given |
/// | indexed for another table | `E_THETA_LEVELS_DATA_MISMATCH`, the first record | the binder above |
///
/// A population is refused, never re-indexed (the check reads `&Population`).
pub fn check_level_index_columns(
    model: &CompiledModel, population: &Population, entry: LevelDataEntry,
) -> Vec<Diagnostic> {
    let binder = match entry {
        LevelDataEntry::Run | LevelDataEntry::PostHoc => {
            "Bind the population with `bind_from_fit(&mut parsed, &model_text, &mut \
             population, &fit.data_bindings)`, passing the bindings the θ you run was \
             laid out on: the fit's `data_bindings`, or, for the model's own θ, a clone \
             of `parsed.model.data_bindings()` taken before the call. Then run the model \
             it re-parses into `parsed`."
        }
        LevelDataEntry::Fit => {
            "To fit this population, parse the model text again and bind it with \
             `bind_theta_levels(&mut parsed, &model_text, &mut population)`, which lays \
             θ out for the levels it holds."
        }
    };
    let blocks = model.theta_blocks();
    bound_level_columns(model)
        .into_iter()
        .filter_map(|(name, _)| {
            let decl = blocks.level_blocks().iter().find(|d| d.name() == name)?;
            let (code, message) = match level_index_finding(decl, population)? {
                // "Never bound" only when no subject carries the index, whatever
                // `covariate_names` lists (#1795): a population some of whose subjects
                // do is partly bound, and is told so below (#1762).
                LevelIndexFinding::NeverBound => (
                    "E_THETA_LEVELS_DATA_UNBOUND",
                    format!(
                        "`theta {name}[...]` is bound, but this population was never bound for \
                         it, so its records carry no index into the block's levels. {binder}"
                    ),
                ),
                LevelIndexFinding::SomeSubjectsUnbound { unbound, of, first_id } => (
                    "E_THETA_LEVELS_DATA_UNBOUND",
                    format!(
                        "`theta {name}[...]` is bound, but {unbound} of {of} subjects in this \
                         population carry no index into the block's levels, the first being \
                         subject {first_id}: they were not bound with the rest. {binder}"
                    ),
                ),
                LevelIndexFinding::ColumnMissing { id, column } => (
                    "E_THETA_LEVELS_DATA_MISMATCH",
                    format!(
                        "`theta {name}[...]` is bound, and subject {id} carries an index into \
                         the block's levels but no value in its level column `{column}`, so \
                         which level the index names cannot be checked."
                    ),
                ),
                LevelIndexFinding::Unseen { labels } => {
                    let remedy = match entry {
                        LevelDataEntry::Run => {
                            "A θ for them comes from a fit of a model bound on this population."
                        }
                        LevelDataEntry::Fit => binder,
                        // #1793: the fit's own population has only the fit's levels.
                        LevelDataEntry::PostHoc => {
                            "So this is not the population the fit was given: pass that one."
                        }
                    };
                    (
                        "E_THETA_LEVELS_DATA_MISMATCH",
                        format!(
                            "`theta {name}[...]` is bound, but this population was bound for \
                             other levels: it has {} level(s) the model's block does not: {}. \
                             The θ being run has no value for them. {remedy}",
                            labels.len(),
                            labels.iter().map(|l| format!("`{l}`")).collect::<Vec<_>>().join(", ")
                        ),
                    )
                }
                LevelIndexFinding::Misindexed { id, time, label, want, got } => {
                    let got = match got {
                        Some(g) => format!("its index says {g}"),
                        None => "it carries no index".to_string(),
                    };
                    // #1797: a subject with no Gaussian observation has no record time
                    // its level comes from.
                    let at = match time {
                        Some(time) => format!(" at time {time}"),
                        None => ", which has no Gaussian observation (its level is read from \
                                 its baseline columns),"
                            .to_string(),
                    };
                    (
                        "E_THETA_LEVELS_DATA_MISMATCH",
                        format!(
                            "`theta {name}[...]` is bound, but this population's index into \
                             its levels was written for another level table: subject {id}{at} \
                             is level `{label}`, level {want} of the model's block, and \
                             {got}. {binder}"
                        ),
                    )
                }
            };
            Some(Diagnostic::error(code, message))
        })
        .map(|d| d.with_block("parameters"))
        .collect()
}

/// A declared non-Gaussian endpoint must have been *routed* by the reader (#1199).
///
/// `read_population_for` routes every row on a CMT the model declares as a TTE /
/// binary / CTMM endpoint into `Subject::obs_records`; the model-blind
/// `read_nonmem_csv` family cannot, so those rows land in the Gaussian grid
/// (`obs_cmts`) and the endpoint scores nothing: a finite, wrong objective (4606.6
/// against 1912.0 NONMEM on `pktte_tdep`) with no message. Two errors:
///
/// - `E_ENDPOINT_UNROUTED`: some subject has a Gaussian observation on an endpoint
///   CMT. The routed reader never does that (pinned by
///   `datareader_tests::routed_reader_never_places_an_endpoint_cmt_in_the_gaussian_grid`),
///   so there is no false positive. Checked at every entry point.
/// - `E_ENDPOINT_NO_RECORDS` (`require_records`): a declared endpoint has no routed
///   row anywhere (e.g. `CMT` column absent, so every row reads as CMT 1; same
///   4606.6). Fit only: a simulation design template carries no event rows and
///   `predict()` legitimately takes a PK-only population.
///
/// Not auto-repaired: re-classifying would lose `TENTRY` and the routed reader's
/// DV-code validation. A no-op without the `survival` feature.
#[cfg(feature = "survival")]
pub fn check_endpoint_routing(
    model: &CompiledModel, population: &Population, require_records: bool,
) -> Vec<Diagnostic> {
    use crate::types::{EndpointLikelihood, ObsRecord};
    use std::collections::BTreeMap;

    // (what to call the endpoint in the message, the block the diagnostic points at)
    let kind_of = |ep: &EndpointLikelihood| -> Option<(&'static str, &'static str)> {
        match ep {
            EndpointLikelihood::Tte { .. } => {
                Some(("time-to-event (`[event_model]`)", "event_model"))
            }
            EndpointLikelihood::Binary { .. } => {
                Some(("binary (`[binary_model]`)", "binary_model"))
            }
            EndpointLikelihood::Count { .. } => Some(("count (`[count_model]`)", "count_model")),
            #[cfg(feature = "markov")]
            EndpointLikelihood::Ctmm { .. } | EndpointLikelihood::Dtmm { .. } => {
                Some(("Markov (`[markov_model]`)", "markov_model"))
            }
            // Not an endpoint the parser ever registers here; skipped rather than
            // matched by a wildcard so a Gaussian CMT can never be reported as unrouted.
            EndpointLikelihood::Gaussian(_) => None,
        }
    };
    let endpoints: BTreeMap<usize, (&'static str, &'static str)> =
        model.endpoints.iter().filter_map(|(&cmt, ep)| kind_of(ep).map(|k| (cmt, k))).collect();
    if endpoints.is_empty() {
        return Vec::new();
    }
    // The record variant an endpoint's likelihood scores (#2877 review round 6): every
    // likelihood path matches on its own variant and skips the rest, so a record of another
    // kind on its CMT — possible only in a hand-built population, the routed reader picks
    // the variant from the endpoint — would be dropped from the objective without a word.
    let scores = |ep: &EndpointLikelihood, r: &ObsRecord| -> bool {
        match (ep, r) {
            (EndpointLikelihood::Tte { .. }, ObsRecord::Event { .. })
            | (EndpointLikelihood::Binary { .. }, ObsRecord::DiscreteState { .. })
            | (EndpointLikelihood::Count { .. }, ObsRecord::Count { .. }) => true,
            #[cfg(feature = "markov")]
            (
                EndpointLikelihood::Ctmm { .. } | EndpointLikelihood::Dtmm { .. },
                ObsRecord::DiscreteState { .. },
            ) => true,
            _ => false,
        }
    };
    let variant_of = |r: &ObsRecord| match r {
        ObsRecord::Event { .. } => "`ObsRecord::Event`",
        ObsRecord::DiscreteState { .. } => "`ObsRecord::DiscreteState`",
        ObsRecord::Count { .. } => "`ObsRecord::Count`",
    };
    let expected_variant = |ep: &EndpointLikelihood| match ep {
        EndpointLikelihood::Tte { .. } => "`ObsRecord::Event`",
        EndpointLikelihood::Count { .. } => "`ObsRecord::Count`",
        _ => "`ObsRecord::DiscreteState`",
    };

    let mut diags = Vec::new();
    // cmt -> (first offending subject, number of Gaussian rows on it)
    let mut unrouted: BTreeMap<usize, (String, usize)> = BTreeMap::new();
    for subj in &population.subjects {
        for &cmt in &subj.obs_cmts {
            if endpoints.contains_key(&cmt) {
                unrouted.entry(cmt).or_insert_with(|| (subj.id.clone(), 0)).1 += 1;
            }
        }
    }
    for (cmt, (id, n)) in &unrouted {
        let (kind, block) = endpoints[cmt];
        diags.push(
            Diagnostic::error(
                "E_ENDPOINT_UNROUTED",
                format!(
                    "E_ENDPOINT_UNROUTED: CMT {cmt} is declared as a {kind} endpoint, but the \
                     data carries {n} Gaussian observation(s) on it (first: subject {id}). This \
                     population was \
                     loaded without endpoint routing — `read_nonmem_csv` and its variants know \
                     nothing about the model — so the endpoint's rows would be scored as \
                     concentrations and its likelihood would contribute nothing. Load the data \
                     with `read_population_for(&model, ...)` so those rows become event records."
                ),
            )
            .with_block(block),
        );
    }
    // cmt -> (first offending subject, its record's variant, number of such records)
    let mut wrong_kind: BTreeMap<usize, (String, &'static str, usize)> = BTreeMap::new();
    for subj in &population.subjects {
        for r in &subj.obs_records {
            let cmt = r.cmt();
            if let Some(ep) = model.endpoints.get(&cmt) {
                if endpoints.contains_key(&cmt) && !scores(ep, r) {
                    wrong_kind
                        .entry(cmt)
                        .or_insert_with(|| (subj.id.clone(), variant_of(r), 0))
                        .2 += 1;
                }
            }
        }
    }
    for (cmt, (id, variant, n)) in &wrong_kind {
        let (kind, block) = endpoints[cmt];
        let want = expected_variant(&model.endpoints[cmt]);
        diags.push(
            Diagnostic::error(
                "E_ENDPOINT_RECORD_KIND",
                format!(
                    "E_ENDPOINT_RECORD_KIND: CMT {cmt} is declared as a {kind} endpoint, but \
                     {n} record(s) on it are {variant} (first: subject {id}), which its \
                     likelihood does not score, so they would contribute nothing. A {kind} \
                     record is {want}: build the population with \
                     `read_population_for(&model, ...)`, which picks the record from the \
                     endpoint, or construct those records as {want}."
                ),
            )
            .with_block(block),
        );
    }
    if require_records {
        // one pass over the records: cmt -> routed rows, of any kind — a CMT whose records
        // are all of the wrong kind has its own, more precise, error above
        let mut routed: BTreeMap<usize, usize> = BTreeMap::new();
        for r in population.subjects.iter().flat_map(|s| s.obs_records.iter()) {
            *routed.entry(r.cmt()).or_insert(0) += 1;
        }
        for (&cmt, &(kind, block)) in &endpoints {
            if unrouted.contains_key(&cmt) {
                continue; // already reported, with the more precise cause
            }
            if routed.get(&cmt).copied().unwrap_or(0) == 0 {
                diags.push(
                    Diagnostic::error(
                        "E_ENDPOINT_NO_RECORDS",
                        format!(
                            "E_ENDPOINT_NO_RECORDS: CMT {cmt} is declared as a {kind} endpoint, \
                             but no row in the data is routed to it, so its likelihood would \
                             contribute nothing. Check \
                             that the dataset has a `CMT` column (without one every row reads \
                             as CMT 1) and that the endpoint's rows carry CMT={cmt}."
                        ),
                    )
                    .with_block(block),
                );
            }
        }
    }
    diags
}

/// See the `survival` twin; no non-Gaussian endpoint exists on this build.
#[cfg(not(feature = "survival"))]
pub fn check_endpoint_routing(
    _model: &CompiledModel, _population: &Population, _require_records: bool,
) -> Vec<Diagnostic> {
    Vec::new()
}

/// Whether this model scores any **Gaussian** observation — i.e. whether its
/// `ErrorSpec` dispatches at least one error model.
///
/// An empty dispatch table is the parser's marker for a model with no
/// `[error_model]` block, which is an endpoint-only model (TTE, binary, categorical,
/// CTMM). Shared with `api::run::model_routes_rows_by_cmt`, which needs the same
/// question to tell "the only endpoint is at the default compartment, so the guess is
/// provably right" from "the guess picks between an endpoint and the Gaussian grid"
/// (#1409).
pub fn model_scores_gaussian_observations(model: &CompiledModel) -> bool {
    match &model.error_spec {
        ErrorSpec::Single(_) => true,
        ErrorSpec::PerCmt(m) => !m.is_empty(),
        ErrorSpec::Selected { endpoints, .. } => !endpoints.is_empty(),
    }
}

/// `E_LEVEL_MISSING_LABEL` (the first rule of `check_nested_levels`): the first record
/// of a nested model with no value in a level column. Shared with the simulate
/// chokepoint (#1082 PR4), which groups the κ draw by those labels and needs nothing
/// else `check_nested_levels` asks of a fit.
pub fn check_level_labels(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    use crate::stats::nested_levels::{record_paths, RecordKind};
    let Some(levels) = model.nested_levels() else {
        return Vec::new();
    };
    for subj in &population.subjects {
        for (kind, i, path) in record_paths(levels, subj) {
            if path.is_some() {
                continue;
            }
            let (what, cov, time) = match kind {
                RecordKind::Observation => ("observation", subj.obs_cov(i), subj.param_time_obs(i)),
                RecordKind::Dose => ("dose", subj.dose_cov(i), subj.dose_param_time(i)),
                RecordKind::PkOnly => (
                    "non-observation (EVID=2 or MDV=1)",
                    subj.pk_only_cov(i),
                    subj.param_time_pk_only(i),
                ),
                RecordKind::Reset => {
                    ("reset (EVID=3/4)", subj.reset_cov(i), subj.param_time_at(subj.reset_times[i]))
                }
            };
            let level = levels
                .levels
                .iter()
                .find(|l| cov.get(&l.column).is_none_or(|v| v.is_nan()))
                .expect("a missing path has a missing column");
            return vec![Diagnostic::error(
                "E_LEVEL_MISSING_LABEL",
                format!(
                    "level `{}` reads column `{}`, which has no numeric value on the {what} \
                     record of subject `{}` at TIME {time}. Every record needs a label at every \
                     level, since its group is identified by its labels from the outermost \
                     level down, and a label is a numeric code: write a text id such as `NCT01` \
                     as a number.",
                    level.name, level.column, subj.id
                ),
            )
            .with_block("parameters")];
        }
        // A blank cell the reader filled from another record (#2036): the model-blind
        // `read_nonmem_csv*`, which does not know the level columns, carries it
        // forward like a covariate's, so the record holds a label — the previous
        // record's — and only the reader's note says it was never written.
        for level in &levels.levels {
            let Some((_, time)) = subj.filled_cells.iter().find(|(c, _)| *c == level.column) else {
                continue;
            };
            return vec![Diagnostic::error(
                "E_LEVEL_MISSING_LABEL",
                format!(
                    "level `{}` reads column `{}`, which has no numeric value on a record of \
                     subject `{}` at TIME {time}. The dataset was read without the model, \
                     which fills a blank cell from the record before it (the first value, \
                     before any), so the record would silently join that record's group. \
                     Every record needs a label at every level, since its group is identified \
                     by its labels from the outermost level down, and a label is a numeric \
                     code: write a text id such as `NCT01` as a number.",
                    level.name, level.column, subj.id
                ),
            )
            .with_block("parameters")];
        }
    }
    Vec::new()
}

/// NONMEM coded `RATE=-2` (modeled infusion duration → `D{cmt}`) needs a model-aware
/// check the datareader cannot make. It is fatal, never a silent fall-through to a
/// bolus (the #324 bug):
///   - **Matching `D{cmt}` parameter.** A `RATE=-2` dose into compartment `n` requires a
///     `D{n}` parameter (so `resolve_rate` has a slot); supported on both engines
///     (`ode_spec.dose_attr_map` for ODE, `model.dose_attr_map` for analytical, #394).
///   - **`D{cmt}`/`R{cmt}` not also used as an ordinary parameter** (#993): if the model
///     reads the same parameter on its prediction path (`[odes]` RHS / `[scaling]`
///     readout) the dose asks one value to be both the modeled infusion {duration,rate}
///     and a model term (`E_DOSE_ATTR_DOUBLE_USE`). Data-dependent, so it lives here and
///     not in the parser: with no coded-`RATE` dose the attribute is inert and an `R1`
///     that is a rate constant is a correct model. The unconditional siblings
///     `F{n}`/`ALAG{n}` are rejected by `model_parser::check_dose_attr_double_use`.
///
/// Reported once per offending compartment (naming the first dose that hits it). The two
/// checks are mutually exclusive per `(attr, cmt)`.
pub fn check_modeled_dose_rates(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    use crate::types::{DoseAttr, RateMode};
    let mut diags = Vec::new();
    // A compartment-free model (#811) applies no dose at all, so a coded `RATE`
    // has no slot to resolve against on any engine and the "add a `D{cmt}` /
    // `R{cmt}` parameter" advice below would ask for a parameter the model cannot
    // consume (#1443). The dose records themselves are reported once, model-wide,
    // by `W_COMPARTMENT_FREE_DOSES` in `check_model_data_warnings`.
    if model.is_algebraic() {
        return diags;
    }
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
                RateMode::ModeledDuration => {
                    (DoseAttr::Duration, "-2", "duration", "E_MODELED_DURATION_NO_PARAM", "D")
                }
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
            let has_slot = model.active_dose_attr_map().indexed_slot(attr, cmt).is_some();
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

/// Whether **anything in this dataset** asks the analytical PK predictor for a value:
/// any Gaussian observation, or any `EVID=2` PK-only request.
///
/// For one narrow case: a pure-TTE / pure-discrete model still needs a
/// `[structural_model]` line, and the idiom is a **placeholder** `one_cpt_iv` with dummy
/// `CL`/`V` (the parser supplies it when the block is absent). Validating dose
/// compartments against that topology would reject a working model over doses no PK
/// code reads.
///
/// Deliberately a **population**-level gate on *whether the dose-compartment check runs
/// at all*: if **any** subject asks the predictor for a value, **every** dose is
/// validated, so a TTE-only subject of a joint PK-TTE model (an early dropout) with dose
/// records has its unroutable infusion caught up front rather than reaching the
/// event-driven walk's `panic!`.
///
/// Complementary to the per-subject predictor gate (`subject_feeds_analytical_pk`,
/// #905): in a live population the check rejects a bad dose so the predictor gate never
/// has to; in an exempt population the check is skipped and the predictor gate keeps the
/// walk unreached. The per-subject truth lives in
/// [`crate::pk::subject_feeds_analytical_pk`]; this is `any(...)` of it.
pub fn population_uses_analytical_pk(model: &CompiledModel, population: &Population) -> bool {
    // Single source of truth with the predictor gate: `subject_feeds_analytical_pk`
    // is the per-subject predicate, and this is `any(...)` of it. The two compose —
    // this population-level test decides whether the dose-compartment check runs at
    // all (a *live* population validates every dose, including a TTE-only subject's),
    // while the per-subject gate declines the walk for a subject the analytical
    // predictor would feed nothing (#905). See `subject_feeds_analytical_pk` for why
    // "no Gaussian observation" implies "predictor output unused" on an analytical
    // model.
    population.subjects.iter().any(|s| crate::pk::subject_feeds_analytical_pk(model, s))
}

/// Refuse what compartment on/off rows (`EVID=2`, `CMT=-N`, #2732) cannot do here.
///
/// Only the event-driven ODE walk has the record, so anything that integrates elsewhere
/// is refused rather than run with the row ignored: an analytical or compartment-free
/// model, an SDE, an endpoint model (its hazard or transition solve is the dense walk).
/// A compartment fed by a built-in absorption input has no amount of its own to switch.
///
/// The rest mirrors NONMEM 7.5.1, which stops on an observation of an off compartment.
/// The record replays status as the engines reach it: a reset turns every compartment
/// on, and a dose turns its own on at the record (NONMEM accepts an observation between
/// a lagged dose's record and its arrival, reading zero). An `SS=1` dose is refused
/// while the walk may still hold anything off, because its periodic solve would
/// integrate the compartment as if it were on.
pub fn check_cmt_toggles(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    let Some(subject) = population.subjects.iter().find(|s| s.has_cmt_toggles()) else {
        return Vec::new();
    };
    let refused = |why: &str, block: &str| {
        vec![Diagnostic::error(
            "E_CMT_TOGGLE_UNSUPPORTED",
            format!(
                "subject {} turns a compartment off (`EVID=2` with a negative `CMT`), which {why}.",
                subject.id
            ),
        )
        .with_block(block)]
    };
    let Some(spec) = model.ode_spec.as_ref() else {
        return refused(
            "needs an `[odes]` model: write the system as `ode(states=[...])`",
            "structural_model",
        );
    };
    if model.is_sde() {
        return refused("an SDE model cannot honour", "diffusion");
    }
    if model_routes_rows_by_cmt(model) {
        return refused("a model with a declared endpoint cannot honour", "structural_model");
    }

    #[derive(Clone, Copy)]
    enum Event {
        Reset,
        Dose(usize),
        Toggle(crate::types::CmtToggle),
        Obs(usize),
    }
    let rank = |e: &Event| match e {
        Event::Reset => 0,
        Event::Dose(_) => 1,
        Event::Toggle(_) => 2,
        Event::Obs(_) => 3,
    };
    // The `TIME` as written. A subject built in memory carries no written times.
    let written_at = |s: &crate::types::Subject, c: &crate::types::CmtToggle| {
        let engine = s.pk_only_times[c.pk_only];
        s.pk_only_raw_times.get(c.pk_only).copied().unwrap_or(engine)
    };
    let mut diags = Vec::new();
    for s in population.subjects.iter().filter(|s| s.has_cmt_toggles()) {
        // Reachable only from a subject built in memory: the reader's indices are its own.
        if let Some(c) = s.cmt_toggles.iter().find(|c| c.pk_only >= s.pk_only_times.len()) {
            return vec![Diagnostic::error(
                "E_CMT_TOGGLE_OUT_OF_RANGE",
                format!(
                    "subject {}: a compartment on/off record names pk-only row {}, but the \
                     subject has {}.",
                    s.id,
                    c.pk_only,
                    s.pk_only_times.len(),
                ),
            )
            .with_block("odes")];
        }
        for c in s.cmt_toggles.iter().filter(|c| !c.on) {
            let cmt = c.state + 1;
            if c.state >= spec.n_states {
                diags.push(
                    Diagnostic::error(
                        "E_CMT_TOGGLE_OUT_OF_RANGE",
                        format!(
                            "subject {}, time {}: `CMT=-{cmt}` turns off compartment {cmt}, but \
                             the `[odes]` block declares only {} state(s) (CMT is 1-based: {:?}).",
                            s.id,
                            written_at(s, c),
                            spec.n_states,
                            spec.state_names,
                        ),
                    )
                    .with_block("odes"),
                );
            } else if spec.input_rate.iter().any(|f| f.cmt == c.state) {
                diags.push(
                    Diagnostic::error(
                        "E_CMT_TOGGLE_UNSUPPORTED",
                        format!(
                            "subject {}, time {}: `CMT=-{cmt}` turns off `{}`, which a built-in \
                             absorption input feeds. Only a compartment dosed directly or fed by \
                             its `d/dt` terms can be switched.",
                            s.id,
                            written_at(s, c),
                            spec.state_names[c.state],
                        ),
                    )
                    .with_block("odes"),
                );
            }
        }
        if !diags.is_empty() {
            return diags;
        }

        let mut events: Vec<(f64, Event)> =
            s.reset_times.iter().map(|&t| (t, Event::Reset)).collect();
        events.extend(s.doses.iter().enumerate().map(|(k, d)| (d.time, Event::Dose(k))));
        events
            .extend(s.cmt_toggles.iter().map(|&c| (s.pk_only_times[c.pk_only], Event::Toggle(c))));
        events.extend(s.obs_times.iter().enumerate().map(|(j, &t)| (t, Event::Obs(j))));
        events.sort_by(|a, b| a.0.total_cmp(&b.0).then(rank(&a.1).cmp(&rank(&b.1))));
        // Two statuses, because a lagged dose separates them. `off` is what an observation is
        // checked against, which a dose clears at its record. `held` is what the walk holds at
        // zero, which it clears at the arrival. The arrival's time is a model parameter this
        // check cannot read, so a lagged dose leaves `held` set until an on row or a reset.
        let mut off: Vec<usize> = Vec::new();
        let mut held: Vec<usize> = Vec::new();
        let cutoffs = s.with_cmt_cutoffs();
        for (t, event) in events {
            match event {
                Event::Reset => {
                    off.clear();
                    held.clear();
                }
                Event::Toggle(c) => {
                    off.retain(|&x| x != c.state);
                    held.retain(|&x| x != c.state);
                    if !c.on {
                        off.push(c.state);
                        held.push(c.state);
                    }
                }
                Event::Dose(k) => {
                    let d = &s.doses[k];
                    // Before the dose turns its own compartment on: an `SS=1` dose into an
                    // off compartment is the same refusal.
                    if d.ss && !held.is_empty() {
                        diags.push(
                            Diagnostic::error(
                                "E_CMT_TOGGLE_SS",
                                format!(
                                    "subject {}, time {t}: an `SS=1` dose while `{}` is off, or \
                                     may be: a lagged dose only turns a compartment on when it \
                                     arrives. Turn the compartment on with an `EVID=2` row before \
                                     the steady-state record, or off after it.",
                                    s.id, spec.state_names[held[0]],
                                ),
                            )
                            .with_block("odes"),
                        );
                        break;
                    }
                    // A dose an off row cancelled never arrives, so it turns nothing on. The
                    // rest of an `ADDL` train stays in `doses` past that row.
                    if d.time >= cutoffs.doses[k].off_at {
                        continue;
                    }
                    off.retain(|&x| x != d.cmt_idx());
                    if !model.has_lagtime_on_cmt(d.cmt_1based()) {
                        held.retain(|&x| x != d.cmt_idx());
                    }
                }
                Event::Obs(j) => {
                    let cmt =
                        s.obs_cmts.get(j).copied().unwrap_or(crate::io::datareader::DEFAULT_CMT);
                    if cmt >= 1 && off.contains(&(cmt - 1)) {
                        diags.push(
                            Diagnostic::error(
                                "E_OBS_CMT_OFF",
                                format!(
                                    "subject {}, time {}: an observation of compartment {cmt} \
                                     (`{}`) while it is off. Turn it on before this record, or \
                                     observe it before the off row.",
                                    s.id,
                                    s.obs_raw_times.get(j).copied().unwrap_or(t),
                                    spec.state_names[cmt - 1],
                                ),
                            )
                            .with_block("odes"),
                        );
                        break;
                    }
                }
            }
        }
    }
    diags
}

/// Every dose must name a compartment the **analytical** engine can actually deliver it
/// into (#375). The datareader has no model, and the parse-time infusable check
/// (`model_parser`, `PkModel::infusable_compartments`) only fires for a *declared*
/// `D{cmt}`/`R{cmt}`, so an explicit positive `RATE` (or out-of-range bolus `CMT`) from
/// the dataset reached the predictors unvalidated and the paths disagreed: the
/// superposition path never reads `dose.cmt` and routed the infusion into central, the
/// event-driven walk `panic!`ed, the `sens` walk dropped it. Rejecting up front removes
/// that disagreement **for the doses it rejects** and keeps the hot path panic-free.
///
/// Agreement for the doses it *accepts* comes from [`crate::pk::dose_needs_event_walk`],
/// which reroutes every dose the superposition dispatch cannot place (it would send it to
/// the default route) to the event-driven walk. This one rejects the unroutable, that one
/// routes the rest.
///
/// Two rules, 1-based like NONMEM's `CMT`:
///   - **Range, every dose whatever its amount.** `cmt` must not exceed the state count.
///     Not skipped for `AMT=0`: both walks bound-check before the amount is read, so an
///     inert row with an out-of-range `CMT` panics like a real dose. `CMT=0` is NONMEM's
///     "default dose compartment" on every path (`saturating_sub(1)`,
///     `dose_needs_event_walk` maps it to 1, superposition ignores `cmt`), so it is left
///     alone.
///   - **Routing, infusions that actually deliver.** The compartment must be in
///     `PkModel::infusable_compartments` (central; oral depot since #400; the
///     peripheral(s) of the 2-/3-cpt IV and, since #375, the oral models). That is
///     `1..=n_states` for all six walk-supported models, so with the range rule claimed
///     first the only value still rejected is `CMT=0` (no meaning for a zero-order
///     input). A zero-amount infusion is exempt (`duration = AMT/RATE = 0`). Checked only
///     for the six event-walk models: a transit/IG subject never takes that walk (an
///     infusion reroutes it to its ODE twin, `CompiledModel::effective_for`, #719 gap 2).
///
/// **ODE models take the range rule but not the routing rule** (#899).
/// `ode/predictions.rs` indexes the state vector directly, so every declared compartment
/// is addressable. A `CMT` past the declared state vector used to be **silently dropped**
/// by the `if cmt_idx < n { … }` guards (finite OFV, dose ignored). The range rule reads
/// `OdeSpec::n_states` and names the declared `state_names`. `CMT=0` splits the same on
/// both engines: accepted for a **bolus** (default compartment 1), rejected for an
/// **infusion**.
///
/// Reported once per `(kind, compartment)`, matching [`check_modeled_dose_rates`].
/// Compartment on/off rows are checked first ([`check_cmt_toggles`]).
pub fn check_dose_compartments(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    // Ahead of everything else here, the compartment-free return below included: a model
    // that cannot honour an off row must not run with the row ignored (#2732).
    let toggle_diags = check_cmt_toggles(model, population);
    if !toggle_diags.is_empty() {
        return toggle_diags;
    }
    // A compartment-free model (#811) applies no dose on any engine, and its
    // `pk_model` is the `one_cpt_iv` **placeholder** the parser installs for the
    // class — so a `CMT=2` dose would be rejected as out of that placeholder's
    // range, with a message about an analytical model the user never wrote
    // (#1443 review). Nothing here can be routed or dropped; the dose records are
    // reported once, model-wide, by `W_COMPARTMENT_FREE_DOSES` in
    // `check_model_data_warnings`. Placed ahead of the analytical-only gate below
    // because `population_uses_analytical_pk` is true for this class (no endpoints).
    if model.is_algebraic() {
        return Vec::new();
    }
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
    // Nothing in this dataset asks the analytical predictor for a value: the `pk` line is a
    // pure-TTE / pure-discrete model's placeholder. See `population_uses_analytical_pk`.
    //
    // **The exemption is safe because the predictor declines in lockstep.** Skipping the
    // check alone would be a landmine: `dose_needs_event_walk` still reroutes a `CMT ≥ 2`
    // dose onto the walk, whose out-of-range guard is a `panic!` (#905). The shared
    // `subject_feeds_analytical_pk` that exempts this population also makes
    // `compute_predictions_with_tv_*` (value) and `subject_routes_to_event_walk` (gradient)
    // decline for every one of its subjects, so the walk is never reached.
    //
    // **Analytical models only.** An `[odes]` block is never a placeholder: a hazard that
    // reads an ODE state has the system integrated even with no PK observation, so the
    // exemption would re-open the silent drop for a TTE-plus-ODE-PK model (#899).
    if ode_spec.is_none() && !population_uses_analytical_pk(model, population) {
        return diags;
    }
    for subject in &population.subjects {
        for dose in &subject.doses {
            let is_infusion = dose.is_infusion();
            let cmt = dose.cmt_raw();
            // **Range rule, every dose whatever its amount.** Both walks bound-check the compartment
            // *before* the amount is read (`event_driven::…impl`'s `if cmt_idx < n_states { … } else
            // { panic! }` and its `sens::propagate` twin), so an `AMT=0` row with an out-of-range
            // `CMT` reaches the panic like a real dose, and since #375 routes any `cmt != 1` bolus to
            // the walk it now gets there. Exempting zero-amount doses re-opened that panic for an
            // `EVID=4, AMT=0` reset row carrying a stale `CMT`.
            //
            // Transit/IG skip the routing rule below but are still 2-/3-state models, so this also
            // stops an out-of-range infusion on them from being silently dropped by the ODE twin's
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
            // **ODE engine (#899): the `CMT=0` half of the routing rule carries over, the
            // infusable-subset half does not.** An in-range ODE dose is delivered by indexing the
            // state vector or, for a compartment fed by a built-in absorption function, as an `R_in`
            // forcing (`input_rate_consumes_cmt`), so every declared state accepts both a bolus and a
            // zero-order input. `CMT=0` is the one value with no target: NONMEM's default dose
            // *compartment* is defined for a bolus but not a zero-order input. The analytical engine
            // rejects it, so rejecting it here keeps both engines' answers identical.
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
            // Transit/IG (`event_walk_supported == false`) never take the walk this routing table
            // belongs to (an infusion reroutes them to their ODE twin, #719 gap 2), so rejecting here
            // would break a supported model.
            //
            // Caveat, tracked as #910: that twin does *not* preserve this model's `CMT` numbering (it
            // drops the depot, so every compartment shifts down by one). `CMT=2` on a
            // `one_cpt_transit` passes the range rule above (2 <= n_states 2) and is out of range on
            // the twin. The range rule cannot catch it: the primary carries `ode_spec: None` and the
            // twin's spec is built lazily inside `effective_for`, after validation.
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

/// A `[covariate_model]` relation stated with a symbolic statistic (#1111) is
/// only buildable once a dataset has been summarised. Reaching a fit with one
/// still unresolved means the covariate effect is simply *absent* from the
/// compiled expression — and a missing covariate divides to `0.0` rather than
/// `inf` in this engine, so nothing further down would have complained. Fail
/// here instead, exactly as `check_unbound_theta_levels` does for #1064.
pub fn check_covariate_model_bound(model: &CompiledModel) -> Vec<Diagnostic> {
    match crate::covariate_stats::assert_covariate_model_bound(model) {
        Ok(()) => Vec::new(),
        Err(msg) => {
            vec![Diagnostic::error("E_COVSTAT_UNRESOLVED", msg).with_block("covariate_model")]
        }
    }
}

/// Reject a categorical `[covariate_model]` relation whose data carries a value outside its
/// declared level set (#1111).
///
/// The generated factor is an `if (COV == l1) 1 + θ1 else if (COV == l2) … else 1` chain whose
/// trailing `1` is the *reference* level's factor, so an unlisted code (new site, typo, a level
/// only on dose records) is silently modelled as the reference.
///
/// Called by `fit()` and `ferx check` (through `check_model_data`), the `simulate*` and
/// adaptive entry points (through `check_simulation_data`), and directly by `predict_diag` /
/// `predict_survival` / `predict_categorical`, `compute_npde_npd` and `resolve_fit_inputs`
/// (`run_sir`, `run_covariance`) (#1740). The per-method estimators `fit()` dispatches to do not
/// call it. A missing value is not a level and passes (the documented neutral branch).
///
/// The advice depends on where the levels came from. A model laid out on a fit
/// (`bound_from_fit()`) has the fit's θ vector, so the repairs are to drop or recode the rows or
/// refit; `bound_from_fit()` is model-wide, so that advice also covers a relation whose levels
/// are written out, which needs the value added to `levels = [...]` before the refit (#1742).
pub fn check_covariate_levels(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    let Some(spec) = model.covariate_model.as_ref() else {
        return Vec::new();
    };
    let mut diags = Vec::new();
    for rel in &spec.relations {
        if !rel.form.is_categorical() {
            continue;
        }
        // Declared levels = the reference (`ref = …`) plus one θ per contrast.
        let mut declared: Vec<f64> = rel.thetas.iter().filter_map(|t| t.level).collect();
        let Some(reference) = rel.resolved_center else {
            // Still unbound — `check_covariate_model_bound` reports that.
            continue;
        };
        declared.push(reference);
        let mut unknown: Vec<f64> = Vec::new();
        for subject in &population.subjects {
            for v in crate::covariate_stats::subject_covariate_values(subject, &rel.covariate) {
                if !declared.contains(&v) && !unknown.contains(&v) {
                    unknown.push(v);
                }
            }
        }
        if unknown.is_empty() {
            continue;
        }
        unknown.sort_by(f64::total_cmp);
        declared.sort_by(f64::total_cmp);
        let (cov, from_fit) = (&rel.covariate, model.bound_from_fit());
        let whose = if from_fit { "the fit's levels" } else { "levels" };
        let facts = format!(
            "[covariate_model]: `{} ~ {cov} {}(...)` has {whose} {declared:?} (reference \
             {reference}), but `{cov}` takes {unknown:?} in this data.",
            rel.parameter,
            rel.form.label(),
        );
        let consequence = "A value outside the levels has no θ of its own and takes the \
                           reference level's factor, so it would be modelled as the reference \
                           with nothing to say so.";
        let advice = if from_fit {
            "The fit estimated no θ for these values: drop or recode those rows, or refit on \
             data that carries them, adding them to `levels = [...]` first if the levels are \
             written out."
                .to_string()
        } else {
            format!(
                "Add the value to `{cov} categorical(levels = [...])` (and refit), or use \
                 `levels = auto` to read the levels off the data; otherwise drop or recode those \
                 rows."
            )
        };
        diags.push(
            Diagnostic::error("E_COV_LEVEL_UNKNOWN", format!("{facts} {consequence} {advice}"))
                .with_block("covariate_model"),
        );
    }
    diags
}

/// Reject a model / dataset where a **survival hazard references a time-varying covariate**,
/// currently silently frozen at its baseline value (#741).
///
/// The analytic hazard [`crate::types::HazardParamFn`] takes no time argument and sees the
/// baseline covariate snapshot; the ODE-accumulated (joint PK-TTE) hazard is integrated with
/// the PK-parameter vector frozen at `t=0` (`survival::ode_cumhaz_hazard`). Either would use the
/// wrong hazard silently, so reject up front, as `check_transit_support` does on the PK path.
///
/// The scope is deliberately tight:
///   * an **analytic** hazard rejects only when a covariate its *own* closed-form parameters
///     reference (`hazard_covariates`) is time-varying (a covariate used solely by a shared PK
///     model, e.g. a frailty-only joint fit, is fine);
///   * an **ODE-accumulated** hazard rejects on *any* time-varying covariate, since the whole
///     PK-parameter vector is frozen at `t=0`.
///
/// Returns an empty list when none applies, else one `E_ENDPOINT_TV_COVARIATE` error whose block
/// is the endpoint's (`binary_model`, `markov_model` or `event_model`). `fit()` and every predict
/// / simulate entry point surface it as an `Err` (#898); `ferx check` through
/// `check_model_data_rule` (#1774).
///
/// How an `E_ENDPOINT_TV_COVARIATE` message names the time-varying input `name`: a data covariate
/// by its column, a θ level block's synthesized index column (#2562) by the block (`__level_*` is
/// engine plumbing).
#[cfg(feature = "survival")]
pub fn tv_reference(name: &str) -> String {
    match name.strip_prefix(crate::parser::model_parser::LEVEL_INDEX_PREFIX) {
        Some(block) => format!("θ level block `{block}`, whose level changes between records"),
        None => format!("covariate `{name}`"),
    }
}

/// The remedy sentence for [`tv_reference`]'s input, ending in `otherwise`.
#[cfg(feature = "survival")]
pub fn tv_remedy(name: &str, otherwise: &str) -> String {
    match name.strip_prefix(crate::parser::model_parser::LEVEL_INDEX_PREFIX) {
        Some(block) => format!(
            "Key `{block}` on columns that are constant within the subject, or {otherwise}."
        ),
        None => format!("Hold `{name}` constant within the subject, or {otherwise}."),
    }
}

#[cfg(feature = "survival")]
pub fn check_survival_tv_covariates(
    model: &CompiledModel, population: &Population,
) -> Vec<Diagnostic> {
    use crate::types::{EndpointLikelihood, HazardSpec};
    // One code for the four texts: the cause is the same (a time-varying covariate read
    // by a closure that takes no time), and only the endpoint's block differs.
    let refuse = |block: &str, message: String| {
        vec![Diagnostic::error("E_ENDPOINT_TV_COVARIATE", message).with_block(block)]
    };
    if !model.has_non_gaussian() {
        return Vec::new();
    }
    for subject in &population.subjects {
        let tv = subject.time_varying_covariate_names();
        if tv.is_empty() {
            continue;
        }
        let tv_set: std::collections::HashSet<&str> = tv.iter().map(String::as_str).collect();
        for (cmt, endpoint) in &model.endpoints {
            // Binary/categorical (#760, #2562): the logit predictor reads each record's own
            // covariate snapshot, as the DTMM below does, so a time-varying covariate — or a
            // θ level index that changes between visits — is honoured, provided the
            // subject's snapshot lines up with its records. One that does not (a subject
            // that also carries time-to-event rows) would silently fall back to the
            // baseline. `lp_covariates` includes covariates reached transitively through
            // `[individual_parameters]`.
            if let EndpointLikelihood::Binary { lp_covariates, .. } = endpoint {
                if let Some(name) = lp_covariates.iter().find(|c| tv_set.contains(c.as_str())) {
                    if subject.record_covariate_snapshots().is_none() {
                        return refuse(
                            "binary_model",
                            format!(
                                "[binary_model] on CMT={cmt}: the linear predictor references \
                                 {}, which is time-varying for subject `{}`, but that subject \
                                 carries no per-record covariate snapshot aligned with its \
                                 binary records (it also has time-to-event records, or was \
                                 built in memory without `endpoint_covariates`), so the \
                                 predictor could only read the baseline value (#741). {}",
                                tv_reference(name),
                                subject.id,
                                tv_remedy(
                                    name,
                                    "fit the binary endpoint without the \
                                                 time-to-event endpoint"
                                ),
                            ),
                        );
                    }
                }
                continue;
            }
            // Count (#2877): the mean and size read each record's own snapshot, as the binary
            // predictor does, with the same refusal when the snapshot cannot line up.
            if let EndpointLikelihood::Count { count_covariates, .. } = endpoint {
                if let Some(name) = count_covariates.iter().find(|c| tv_set.contains(c.as_str())) {
                    if subject.record_covariate_snapshots().is_none() {
                        return refuse(
                            "count_model",
                            format!(
                                "[count_model] on CMT={cmt}: the count mean or size references \
                                 {}, which is time-varying for subject `{}`, but that subject \
                                 carries no per-record covariate snapshot aligned with its \
                                 count records (it also has time-to-event records, or was built \
                                 in memory without `endpoint_covariates`), so the expression \
                                 could only read the baseline value (#741). {}",
                                tv_reference(name),
                                subject.id,
                                tv_remedy(
                                    name,
                                    "fit the count endpoint without the time-to-event endpoint"
                                ),
                            ),
                        );
                    }
                }
                continue;
            }
            // DTMM (#2473): the transition probabilities read each record's own covariate
            // snapshot, so a time-varying covariate is honoured — provided the subject's
            // snapshot lines up with its records (`record_covariate_snapshots`). One that
            // does not (a subject that also carries time-to-event rows, or an in-memory
            // subject without per-row snapshots) would silently fall back to the baseline.
            #[cfg(feature = "markov")]
            if let EndpointLikelihood::Dtmm { prob_covariates, .. } = endpoint {
                if let Some(name) = prob_covariates.iter().find(|c| tv_set.contains(c.as_str())) {
                    if subject.record_covariate_snapshots().is_none() {
                        return refuse(
                            "markov_model",
                            format!(
                                "[markov_model] (type = dtmm) on CMT={cmt}: a transition \
                                 probability references {}, which is time-varying for subject \
                                 `{}`, but that subject carries no per-record covariate \
                                 snapshot aligned with its state records (it also has \
                                 time-to-event records, or was built in memory without \
                                 `endpoint_covariates`), so the probability could only read the \
                                 baseline value (#741). {}",
                                tv_reference(name),
                                subject.id,
                                tv_remedy(name, "fit the DTMM without the time-to-event endpoint"),
                            ),
                        );
                    }
                }
                continue;
            }
            // CTMM (#759): the generator is built once per subject from the baseline
            // covariate snapshot (time-homogeneous — a drug-driven Q(t) is Phase 6), so a
            // time-varying covariate on a transition intensity would be silently frozen.
            #[cfg(feature = "markov")]
            if let EndpointLikelihood::Ctmm { generator_covariates, generator_states, .. } =
                endpoint
            {
                if let Some(name) =
                    generator_covariates.iter().find(|c| tv_set.contains(c.as_str()))
                {
                    return refuse(
                        "markov_model",
                        format!(
                        "[markov_model] on CMT={cmt}: a transition intensity references {}, \
                         which is time-varying for subject `{}`. Time-varying covariates \
                         on the generator are not yet supported — the generator is evaluated at the \
                         baseline covariate value (time-homogeneous CTMM), so the fit would \
                         silently use the wrong intensity (#741). A drug-driven Q(t) is Phase 6; \
                         use a time-constant (baseline) covariate for now.",
                        tv_reference(name),
                        subject.id
                    ),
                    );
                }
                // A drug-driven generator also reads the `[odes]` solve, which runs under
                // the same baseline snapshot: a time-varying covariate on the disposition
                // it integrates (CL, V, …) is frozen too, although no intensity names it
                // (#1854). Time-homogeneous generators never solve the ODE.
                if !generator_states.is_empty() {
                    match ctmm_tv_covariate_on_disposition(model, generator_states, &tv) {
                        Ok(None) => {}
                        Ok(Some((name, param, site))) => {
                            return refuse(
                                "markov_model",
                                format!(
                                    "[markov_model] on CMT={cmt}: the drug-driven \
                                 (time-inhomogeneous) CTMM generator reads model states solved \
                                 with the individual parameters at their baseline covariate \
                                 values, but `{param}` reads covariate `{name}`, which is \
                                 time-varying for subject `{}`, and {site} it. Time-varying \
                                 covariates on the disposition a state-driven transition \
                                 intensity reads are not yet supported (#1854): the Gaussian \
                                 predictions evaluate `[individual_parameters]` at every record \
                                 while the CTMM pass keeps the baseline value. Hold `{name}` \
                                 constant within each subject for now.",
                                    subject.id
                                ),
                            );
                        }
                        Err(()) => {
                            let name = &tv[0];
                            return refuse(
                                "markov_model",
                                format!(
                                "[markov_model] on CMT={cmt}: covariate `{name}` is time-varying \
                                 for subject `{}`, and this model's drug-driven \
                                 (time-inhomogeneous) CTMM generator reads an [odes] solve run \
                                 at the baseline covariate values, but it has no compiled \
                                 `[odes]` / `[individual_parameters]` program, so whether the \
                                 covariate reaches that solve cannot be determined (#1854). \
                                 Build the model through `parse_model_string` / \
                                 `parse_model_file`, or hold the covariate constant within each \
                                 subject.",
                                subject.id
                            ),
                            );
                        }
                    }
                }
                continue;
            }
            let EndpointLikelihood::Tte { hazard, hazard_covariates, .. } = endpoint else {
                continue;
            };
            match hazard {
                HazardSpec::Analytic { .. } => {
                    if let Some(name) =
                        hazard_covariates.iter().find(|c| tv_set.contains(c.as_str()))
                    {
                        return refuse(
                            "event_model",
                            format!(
                            "[event_model] on CMT={cmt}: the survival hazard references covariate \
                             `{name}`, which is time-varying for subject `{}`. Time-varying \
                             covariates on a hazard are not yet supported — the analytic hazard is \
                             evaluated at the baseline covariate value, so the fit would silently \
                             use the wrong hazard (#741). Use a time-constant (baseline) covariate \
                             on the hazard for now.",
                            subject.id
                        ),
                        );
                    }
                }
                HazardSpec::OdeAccumulated { .. } => {
                    // Any time-varying covariate is frozen by the survival ODE solve's t=0
                    // PK-parameter snapshot; name the first one deterministically (`tv` is sorted).
                    let name = &tv[0];
                    return refuse(
                        "event_model",
                        format!(
                        "[event_model] on CMT={cmt}: the ODE-accumulated (joint PK-TTE) hazard is \
                         integrated with covariates frozen at their t=0 baseline, but covariate \
                         `{name}` is time-varying for subject `{}`. Time-varying covariates on a \
                         drug-driven hazard are not yet supported (#741). Hold the covariate \
                         constant within each subject for now.",
                        subject.id
                    ),
                    );
                }
            }
        }
    }
    Vec::new()
}

/// Reject a model whose **time-inhomogeneous (drug-driven) CTMM generator** reads a
/// `TIME`-dependent disposition: the #1853 freeze on the `[markov_model]` endpoint (#1918).
///
/// `markov::endpoint`'s inhomogeneous scorer solves the `[odes]` system under one `$PK` snapshot,
/// `(model.pk_param_fn)(…, 0.0)`, and samples the states the intensities read. Dose-record
/// quantities are read per dose record (#1575), so what stays frozen is the integrated disposition
/// and `init()`. The intensity expressions themselves cannot read `TIME` (the parser refuses that).
///
/// Measured on a 1-cpt IV model, `CL = TVCL·(1 + KT·TIME)`, `V = 10`, a 100 mg bolus and both
/// intensities scaled by `central/V`: the CTMM `−log L` is **bit-identical** at `KT = 0` and
/// `KT = 0.05` (3.3747459385), matching the frozen-`CL` closed form (3.3747459877) to the scorer's
/// sub-grid error, against 3.3778309 for the per-record convention; `central(20)` is 13.53 frozen
/// against 3.68 per record.
///
/// The predicate is `check_survival_time_dependent_pk`'s, per parameter and per endpoint: the
/// dependency slice is the union of
/// `crate::parser::model_parser::OdeRhsProgram::chz_dependency_slice` over every state the
/// generator reads. A time-**homogeneous** CTMM reads no ODE state and returns `None`. Fails closed
/// like the hazard check.
#[cfg(feature = "markov")]
pub fn check_ctmm_time_dependent_pk(model: &CompiledModel) -> Option<String> {
    use crate::types::EndpointLikelihood;
    // Every inhomogeneous CTMM endpoint with the ODE states its generator reads, in CMT
    // order so the message is deterministic.
    let mut ctmms: Vec<(usize, Vec<usize>)> = model
        .endpoints
        .iter()
        .filter_map(|(cmt, e)| match e {
            EndpointLikelihood::Ctmm { generator_states, .. } if !generator_states.is_empty() => {
                Some((*cmt, generator_states.iter().map(|&(_, idx)| idx).collect()))
            }
            _ => None,
        })
        .collect();
    ctmms.sort_unstable();
    let first_cmt = ctmms.first().map(|(cmt, _)| *cmt)?;
    let unanalysable = |what: &str, how: &str| -> Option<String> {
        Some(format!(
            "[markov_model] on CMT={first_cmt}: this model carries a drug-driven \
             (time-inhomogeneous) CTMM generator, whose [odes] solve runs under the individual \
             parameters frozen at their TIME=0 values, but it has no {what}, so whether that \
             freeze applies cannot be determined (#1918). Every model parsed from a `.ferx` \
             file carries one; a `CompiledModel` assembled or edited by hand may not. Build it \
             through `parse_model_string` / `parse_model_file`{how}."
        ))
    };
    let Some(rhs) = model.ode_spec.as_ref().and_then(|o| o.rhs_program.as_ref()) else {
        return unanalysable(
            "compiled `[odes]` program (`OdeSpec::rhs_program`)",
            ", or make the transition intensities state-free",
        );
    };
    let Some(program) = model.indiv_param_partials.indiv_param_program.as_ref() else {
        return unanalysable(
            "compiled `[individual_parameters]` program (`CompiledModel::indiv_param_partials`)",
            " rather than replacing that field with `IndivParamPartials::empty()`",
        );
    };
    let reads_time = program.params_reading_time();
    if !reads_time.iter().any(|&t| t) {
        return None;
    }
    for (cmt, states) in ctmms {
        let slice = ctmm_generator_slice(rhs, &states);
        if let Some((name, site)) = first_param_reaching_slice(model, rhs, &reads_time, &slice) {
            return Some(format!(
                "[markov_model] on CMT={cmt}: the drug-driven (time-inhomogeneous) CTMM \
                 generator reads model states solved with the individual parameters frozen at \
                 their TIME=0 values, but `{name}` reads the `TIME` built-in and {site} it. A \
                 TIME-dependent disposition on a state-driven transition intensity is not yet \
                 supported (#1918): the Gaussian predictions evaluate `[individual_parameters]` \
                 at every record while the CTMM pass keeps the TIME=0 value. Hold `{name}` \
                 constant in TIME for now."
            ));
        }
    }
    None
}

/// What a drug-driven CTMM generator's ODE states depend on: the union of
/// `crate::parser::model_parser::OdeRhsProgram::chz_dependency_slice` over every state
/// it reads. Shared by the `TIME` (#1918) and covariate (#1854) freezes.
#[cfg(feature = "markov")]
pub fn ctmm_generator_slice(
    rhs: &crate::parser::model_parser::OdeRhsProgram, states: &[usize],
) -> Vec<bool> {
    let mut slice: Vec<bool> = Vec::new();
    for &s in states {
        let one = rhs.chz_dependency_slice(s);
        if slice.len() < one.len() {
            slice.resize(one.len(), false);
        }
        for (acc, x) in slice.iter_mut().zip(one) {
            *acc |= x;
        }
    }
    slice
}

/// The first individual parameter (declaration order) flagged in `mask` that the frozen
/// `[odes]` solve reads somewhere `slice` depends on, with where it reads it. A parameter
/// the compiled block does not carry, or one that reaches only a dose-record quantity
/// (#1575), is skipped — neither is frozen by the snapshot.
#[cfg(feature = "markov")]
pub fn first_param_reaching_slice<'m>(
    model: &'m CompiledModel, rhs: &crate::parser::model_parser::OdeRhsProgram, mask: &[bool],
    slice: &[bool],
) -> Option<(&'m str, &'static str)> {
    for (i, name) in model.indiv_param_names.iter().enumerate() {
        if !mask.get(i).copied().unwrap_or(false) || !rhs.has_indiv_param(i) {
            continue;
        }
        let in_rhs = rhs.indiv_param_reaches_chz(i, slice);
        let in_init = rhs.init_reads_indiv_param_reaching_chz(i, slice);
        let site = match (in_rhs, in_init) {
            (true, true) => "the [odes] RHS and an init(...) seed read",
            (true, false) => "the [odes] RHS reads",
            (false, true) => "an [odes] init(...) seed reads",
            (false, false) => continue,
        };
        return Some((name, site));
    }
    None
}

/// The covariate half of the CTMM arm of [`check_survival_tv_covariates`] (#1854): on a drug-driven
/// generator, the first time-varying covariate (sorted) that reaches an individual parameter the
/// frozen `[odes]` solve feeds into the generator's states, as `(covariate, parameter, site)`.
///
/// The inhomogeneous scorer integrates under one `$PK` snapshot at the **baseline** covariates, so a
/// time-varying covariate on `CL` or `V` (not named by any intensity, so invisible to
/// `generator_covariates`) was read at baseline while the Gaussian walk read it per record.
/// Dose-record quantities (`KA`, `F`, `ALAG`, …, #1575) and covariates on parameters the generator's
/// states never see are not refused. `Err(())` when the compiled programs are missing (fails closed).
#[cfg(feature = "markov")]
pub fn ctmm_tv_covariate_on_disposition<'m>(
    model: &'m CompiledModel, generator_states: &[(String, usize)], tv: &[String],
) -> Result<Option<(String, &'m str, &'static str)>, ()> {
    let rhs = model.ode_spec.as_ref().and_then(|o| o.rhs_program.as_ref()).ok_or(())?;
    let program = model.indiv_param_partials.indiv_param_program.as_ref().ok_or(())?;
    let states: Vec<usize> = generator_states.iter().map(|&(_, idx)| idx).collect();
    let slice = ctmm_generator_slice(rhs, &states);
    for cov in tv {
        let mask = program.params_reading_covariates(&[cov.as_str()]);
        if let Some((param, site)) = first_param_reaching_slice(model, rhs, &mask, &slice) {
            return Ok(Some((cov.clone(), param, site)));
        }
    }
    Ok(None)
}

/// [`check_ctmm_time_dependent_pk`] as a diagnostic: `E_ENDPOINT_TIME_DEPENDENT_PK` on block
/// `markov_model` (#1918). Kept apart from `check_survival_time_dependent_pk_diags` because
/// the two freezes are read by different entry points: the CTMM likelihood is scored by
/// `fit()` and re-scored by the post-fit `run_covariance()` / `run_sir()`, so those three and
/// `ferx check` call this. `predict()` drops CTMM rows when
/// continuous observations exist and its per-record walk never runs the frozen solve, and
/// `simulate()` refuses every CTMM with `E_CTMM_UNSUPPORTED` — a refusal this must not
/// pre-empt.
#[cfg(feature = "markov")]
pub fn check_ctmm_time_dependent_pk_diags(model: &CompiledModel) -> Vec<Diagnostic> {
    check_ctmm_time_dependent_pk(model)
        .map(|m| {
            vec![Diagnostic::error("E_ENDPOINT_TIME_DEPENDENT_PK", m).with_block("markov_model")]
        })
        .unwrap_or_default()
}

/// `E_ODE_METHOD_LINEAR`: a caller's `FitOptions { ode_method: Linear }` on a model that
/// cannot take the exact linear path (#852).
///
/// `ode_method = linear` is an assertion. From the model file the parser refuses it; a
/// caller's `FitOptions` never passes the parser, and without this the fit-scoped override
/// would quietly integrate an ineligible model (`effective_solver_opts` demotes it to `auto`)
/// or ignore the option on an analytical one. Every entry point that arms `FitOptions`' ODE
/// settings calls it: `fit()` (through `check_model_options`), `run_covariance` and
/// `run_sir`.
pub fn check_ode_method_linear(model: &CompiledModel, options: &FitOptions) -> Vec<Diagnostic> {
    if options.ode_method != crate::ode::OdeMethod::Linear {
        return Vec::new();
    }
    crate::ode::predictions::ode_method_linear_refusal(model.ode_spec.as_ref())
        .map(|refusal| Diagnostic::error("E_ODE_METHOD_LINEAR", refusal).with_block("fit_options"))
        .into_iter()
        .collect()
}

/// Every data column the model itself reads — the one owner of "what a
/// model-aware read must take from the CSV" (#1936). Passed to the reader as
/// `extra` on **both** the `[covariates]` path and the no-block path, so a model
/// reading a role column or its `iov_column` (NONMEM's `$PK` spelling of
/// `IF (OCC.EQ.2)`) reads the same columns whether or not it declares anything.
/// Before #1936 only the declared path passed them: an unrelated
/// `[covariates] WT` line turned `E_MISSING_COVARIATE` on `OCC` into a working fit.
pub fn referenced_columns(model: &CompiledModel) -> Vec<String> {
    model
        .referenced_covariates
        .iter()
        // A a level block index column (#1064) is synthesized by
        // `bind_theta_levels` after the read, never present in the CSV — asking
        // the reader for it would fail on a column the user never wrote.
        .filter(|c| !crate::parser::model_parser::is_level_index_column(c))
        .cloned()
        .collect()
}

/// The non-Gaussian row routing a dataset needs for `model`: every CMT the model
/// declares as a TTE / binary / CTMM endpoint, plus the design-row DV placeholder
/// per CTMM endpoint and the missing-DV policy. Empty routing sets for a
/// Gaussian-only model.
///
/// This is the **only** place the routing is derived from a model. Every reader
/// call that has a model in hand must go through it — the file-based
/// `read_population_for` family, and `read_population_routed_by` for the
/// `.fitrx` reload and the `run_covariance` / `run_sir` re-read — because a
/// population read *without* it carries the endpoint's rows as Gaussian
/// observations and no event records, which `fit()` then rejects with
/// `E_ENDPOINT_UNROUTED` (#1199).
pub fn obs_routing_for(model: &CompiledModel, missing_dv: MissingDvPolicy) -> ObsRouting {
    // Nothing to route without `survival`: no non-Gaussian endpoint can be parsed.
    #[cfg(not(feature = "survival"))]
    let _ = model;
    // Extract TTE CMTs from model endpoints so the reader can route TTE rows
    // to obs_records instead of the Gaussian parallel Vecs.
    #[cfg(feature = "survival")]
    let tte_cmts: std::collections::HashSet<usize> = model
        .endpoints
        .iter()
        .filter_map(
            |(&cmt, ep)| {
                if matches!(ep, EndpointLikelihood::Tte { .. }) {
                    Some(cmt)
                } else {
                    None
                }
            },
        )
        .collect();
    #[cfg(not(feature = "survival"))]
    let tte_cmts: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // Discrete-state CMTs route to `ObsRecord::DiscreteState` (integer DV), like TTE
    // rows route to `ObsRecord::Event` — both need the non-Gaussian reader path. This
    // set covers binary/categorical (#760) and CTMM (#759) endpoints, which share the
    // discrete-state plumbing.
    #[cfg(feature = "survival")]
    // Binary **∪ CTMM** — every CMT whose integer DV routes to `ObsRecord::DiscreteState`.
    // Deliberately not named `binary_cmts`: this function used to share that name with the
    // binary-only `Vec` above, and the two are equal today only because CTMM `--simulate`
    // is rejected upstream. When CTMM simulation lands (#820) they diverge.
    let discrete_cmts: std::collections::HashSet<usize> = {
        let discrete: std::collections::HashSet<usize> = model.binary_cmts().into_iter().collect();
        #[cfg(feature = "markov")]
        let discrete =
            discrete.into_iter().chain(model.ctmm_cmts()).chain(model.dtmm_cmts()).collect();
        discrete
    };
    #[cfg(not(feature = "survival"))]
    let discrete_cmts: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // One reader call covers every combination: the routing sets are empty for a
    // Gaussian-only model (so no row is routed to `obs_records`), and the reader
    // builds the covariate read set from `decls` when the model declares
    // `[covariates]`, from `fallback_columns` otherwise.
    // Placeholder DV code for an integer-coded design row: the endpoint's first
    // declared state code, so a `state_codes` table that is not 0-based (e.g.
    // `1`/`2`) never gets an undecodable `0` placeholder. Only CTMM declares
    // codes; Binary is `{0,1}` and keeps the `0` default.
    #[cfg(feature = "markov")]
    let design_states: HashMap<usize, usize> = model
        .endpoints
        .iter()
        .filter_map(|(&cmt, ep)| match ep {
            EndpointLikelihood::Ctmm { state_codes, .. }
            | EndpointLikelihood::Dtmm { state_codes, .. } => {
                state_codes.first().map(|&code| (cmt, code))
            }
            _ => None,
        })
        .collect();
    #[cfg(not(feature = "markov"))]
    let design_states: HashMap<usize, usize> = HashMap::new();

    // Nested levels (#1082): a blank level cell is missing, not the previous record's label.
    let no_carry_forward: std::collections::HashSet<String> = model
        .nested_levels()
        .map(|l| l.levels.iter().map(|l| l.column.clone()).collect())
        .unwrap_or_default();

    // Count CMTs (#2877) route to `ObsRecord::Count`.
    #[cfg(feature = "survival")]
    let count_cmts: std::collections::HashSet<usize> = model.count_cmts().into_iter().collect();
    #[cfg(not(feature = "survival"))]
    let count_cmts: std::collections::HashSet<usize> = std::collections::HashSet::new();

    ObsRouting::tte_and_discrete(&tte_cmts, &discrete_cmts)
        .with_count(count_cmts)
        .with_missing_dv(missing_dv)
        .with_design_states(design_states)
        .with_no_carry_forward(no_carry_forward)
        // `W_COVARIATE_MISSING` only for a column a relation reads (#2599).
        .with_warn_missing_covariates(columns_read_by_relations(model))
}

/// Every data column a relation of `model` reads, the set `W_COVARIATE_MISSING`
/// is raised for (#2599); the reader adds the columns its `[data_selection]`
/// filter reads. Matched exactly: the required [`referenced_columns`], which are
/// case-sensitive. Matched in any case: the `[output]` columns echoed from the
/// data, resolved case-insensitively. Matched as `[derived]` resolves them (the
/// verbatim, uppercase or lowercase spelling of the column): the names a
/// `[derived]` expression looks up at evaluation time (not required columns, so
/// not in `referenced_covariates`). A column only declared in `[covariates]` or
/// listed in `covariate_columns`, with nothing reading it, is in neither — a gap
/// in it reaches neither the objective nor an output.
pub fn columns_read_by_relations(model: &CompiledModel) -> CovariateReads {
    CovariateReads {
        exact: referenced_columns(model).into_iter().collect(),
        any_case: output_echo_columns(model).cloned().collect(),
        derived: model.derived_reads().iter().cloned().collect(),
    }
}

/// Whether this model routes any observation row to its endpoint **by CMT** (#1409).
///
/// Derived from [`obs_routing_for`] — the only place a routing set is built from a
/// model — rather than restated, so an endpoint family added there is in scope for
/// `W_CMT_DEFAULTED` without a second edit. The missing-DV policy does not enter the
/// question, so the fitting policy is passed and the answer is the same either way.
pub fn model_routes_rows_by_cmt(model: &CompiledModel) -> bool {
    let routing = obs_routing_for(model, MissingDvPolicy::Skip);
    if !routing.routes_by_cmt() {
        return false;
    }
    // The one endpoint shape where the default is provably the right answer, so
    // reporting it is a false positive rather than a caught ambiguity: an
    // **endpoint-only** model (no `[error_model]`, so no Gaussian grid to fall into)
    // whose routing names **only** the compartment the reader falls back to. Every
    // row keys to that endpoint whatever the cell said, and any other CMT is an
    // `E_PER_CMT_ERROR_MODEL` error rather than a silent re-route. Documented as a known
    // limit when the endpoint channel landed; closed here (#1409 review).
    //
    // Both halves are needed. Drop the Gaussian test and a model with a `cmt = 1`
    // endpoint *and* ordinary observations stops reporting, although a defaulted row
    // there really does pick between the endpoint and the Gaussian grid. Drop the
    // `routes_only` test and a competing-risks model at `cmt = 1`/`cmt = 2` stops
    // reporting, which is the #1404 measurement this whole channel exists for.
    if !model_scores_gaussian_observations(model)
        && routing.routes_only(crate::io::datareader::DEFAULT_CMT)
    {
        return false;
    }
    true
}

/// The `[output]` columns echoed from the subject's covariates (or, failing a
/// covariate of that name, its individual parameters): every declared column but
/// a `[derived]` one (filled by the derived pass), a mandatory sdtab column and an
/// ETA (both ignored with `W_OUTPUT_DUPLICATE`). The single rule both
/// `compute_extra_output_columns` and the reader's `W_COVARIATE_MISSING` gate
/// (#2599) read, so the warning covers exactly the columns the echo reads.
///
/// `MIXNUM` on a `[mixture]` model is not echoed: it is the class, as everywhere else
/// in the model file (#2784) — see [`is_mixture_class_column`].
pub fn output_echo_columns(model: &CompiledModel) -> impl Iterator<Item = &String> {
    model.output_columns.iter().filter(move |col| {
        let is = |n: &str| n.eq_ignore_ascii_case(col);
        !model.derived_exprs.iter().any(|d| is(&d.name))
            && !OUTPUT_MANDATORY.iter().any(|m| is(m))
            && !model.eta_names.iter().any(|e| is(e))
            && !is_mixture_class_column(model, col)
    })
}

/// The sdtab name of the mixture-class column, NONMEM's `$TABLE MIXNUM`.
pub const MIXNUM_COLUMN: &str = "MIXNUM";

/// Whether the `[output]` entry `col` names the mixture class rather than a data
/// column (#2784): `MIXNUM` (any case) on a `[mixture]` model. In an expression that
/// name is the class built-in and assigning to it is refused, so `[output] MIXNUM`
/// means the class too — the subject's simulated class where its data were drawn under
/// one (`Subject::simulated_mixnum`), its fitted `MIXEST` otherwise — and never a data
/// column that happens to share the name. On a model without `[mixture]` there is no
/// class and `MIXNUM` is an ordinary name.
pub fn is_mixture_class_column(model: &CompiledModel, col: &str) -> bool {
    model.mixture.is_some() && col.eq_ignore_ascii_case(MIXNUM_COLUMN)
}
