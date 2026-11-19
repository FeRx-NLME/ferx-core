//! Whether a population and solver options are the ones a fit ran on: the fingerprint check
//! behind `PopulationFingerprint::verify` and the ODE option domains. Below the estimators so
//! `types` can call it (#3041).
use crate::diagnostics::EngineError;
use crate::types::{CompiledModel, Population, PopulationDifference, PopulationFingerprint};

/// The note for a fit whose population fingerprint was made with another encoding
/// scheme (#1685): it cannot be compared, so the step runs as for a fit without one.
pub(crate) const STALE_FINGERPRINT_WARNING: &str = "this fit's population fingerprint is \
     from another ferx version's scheme, so the population was not verified against the one \
     the fit was given";

/// The checks after the subject list and the binding, in order: every categorical
/// covariate value against the model's levels (`E_COV_LEVEL_UNKNOWN`, #1740: the
/// subject IDs can match while a column was recoded), then, on a fit with a
/// current fingerprint, the population's content against it (#1685). On a fit
/// without one (absent or stale), the population's level index against the model's
/// level table instead (#1793): nothing else would see a population bound on its own
/// for other levels, or never or partly bound, and it would be scored silently.
pub(crate) fn check_levels_and_content(
    fingerprint: Option<&PopulationFingerprint>, model: &CompiledModel, population: &Population,
    source: PopulationSource,
) -> Result<(), EngineError> {
    crate::diagnostics::first_error(&crate::model_checks::check_covariate_levels(
        model, population,
    ))?;
    let Some(fingerprint) = fingerprint else {
        return crate::diagnostics::first_error(&crate::model_checks::check_level_index_columns(
            model,
            population,
            crate::model_checks::LevelDataEntry::PostHoc,
        ));
    };
    match fingerprint.first_content_difference(population) {
        Some(d) => Err(population_mismatch(&d, model, population, source)),
        None => Ok(()),
    }
}

/// [`PopulationFingerprint::verify`]: the resolver's checks on a population the
/// caller already bound, in the resolver's order and through its refusals.
pub(crate) fn verify_given_population(
    fingerprint: &PopulationFingerprint, model: &CompiledModel, population: &Population,
) -> Result<Vec<String>, EngineError> {
    if !fingerprint.is_current() {
        return Ok(vec![STALE_FINGERPRINT_WARNING.to_string()]);
    }
    let source = PopulationSource::Verify;
    if let Some(d) = fingerprint.first_subject_difference(population) {
        return Err(mismatch_refusal(&d, source));
    }
    check_levels_and_content(Some(fingerprint), model, population, source)?;
    Ok(Vec::new())
}

/// How the population a post-hoc step was about to run on was obtained, for
/// [`mismatch_refusal`]'s source sentence. A re-read also says whether the fit
/// recorded the data file's SHA-256: a recorded one that did not match was refused
/// before the read, so a recorded one here is one the file matches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PopulationSource {
    /// Passed as `population = Some(&pop)`: the caller knows where it came from.
    Supplied,
    /// Handed to [`PopulationFingerprint::verify`]: ferx does not know where it
    /// came from.
    Verify,
    /// Re-read from `fit.data_path` with the fit's recorded reader settings.
    RecordedSettings { data_hash: bool },
    /// Re-read with the model file's settings: the fit records none.
    ModelFileSettings { data_hash: bool },
    /// Re-read with the model's endpoint routing only: the fit records neither
    /// settings nor a `model_path` (review r1 #3).
    Routed { data_hash: bool },
}

/// The one remedy for `E_POPULATION_MISMATCH`, whatever the cell (#1813): true and
/// actionable from every front end, and in no front end's syntax.
pub(crate) const POPULATION_MISMATCH_SUGGESTION: &str =
    "use the population the fit was given, or refit";

/// The refusal for a population that is not the one the fit was given (#1685),
/// after the binding: [`mismatch_refusal`], unless an earlier cause pre-empts it.
///
/// Two causes keep their own text and code, since there the fix is a reader or a
/// binder, not another population: a population read without the model's endpoint
/// routing (`E_ENDPOINT_UNROUTED`), and one never bound — or only partly bound — for
/// the model's level block (#1647's and #1762's `E_THETA_LEVELS_DATA_UNBOUND`, from
/// `check_level_index_columns`). One bound for other levels gets the mismatch: for a
/// post-hoc step the fix is the fit's population, not a refit.
pub(crate) fn population_mismatch(
    d: &PopulationDifference, model: &CompiledModel, population: &Population,
    source: PopulationSource,
) -> EngineError {
    // A population read without the model's endpoint routing keeps `fit()`'s own
    // refusal and code (`E_ENDPOINT_UNROUTED`, #1199): it names the reader to use.
    let unrouted = crate::model_checks::check_endpoint_routing(model, population, true);
    if let Err(e) = crate::diagnostics::first_error(&unrouted) {
        return e;
    }
    // Only a supplied population can lack the columns: a re-read, and a supplied one
    // under `model = None`, are bound above. Only the unbound cells (#1647, #1762's
    // "`k` of `n`"): a population bound for other levels is not the fit's, and hears
    // so below — `E_THETA_LEVELS_DATA_MISMATCH`'s "a fit of a model bound on this
    // population" is advice for new data, not for a post-hoc step on this fit.
    let unbound: Vec<_> = crate::model_checks::check_level_index_columns(
        model,
        population,
        crate::model_checks::LevelDataEntry::Run,
    )
    .into_iter()
    .filter(|d| d.code == "E_THETA_LEVELS_DATA_UNBOUND")
    .collect();
    if let Err(e) = crate::diagnostics::first_error(&unbound) {
        return e;
    }
    mismatch_refusal(d, source)
}

/// `E_POPULATION_MISMATCH` (#1813): what differs, then how the population was
/// obtained — never a cause the comparison did not establish. The data may have
/// changed, the settings may have, or the reader may have; the fingerprint cannot
/// tell them apart, so the message names none. The remedy is
/// [`POPULATION_MISMATCH_SUGGESTION`], in `suggestion()` and folded into `Display`.
pub(crate) fn mismatch_refusal(d: &PopulationDifference, source: PopulationSource) -> EngineError {
    let mut msg = format!("this population is not the one the fit was given: {d}.");
    // A filter only removes rows, so it can explain another dose count but not an
    // edited dose under the same count (#1868 r1 #3).
    match d {
        PopulationDifference::Doses { population, fit, .. } if population != fit => msg.push_str(
            " A dose-row filter (`ignore = EVID == 1 && ...`) or an edited dose record \
             changes the doses without changing any observation.",
        ),
        PopulationDifference::Doses { .. } => msg
            .push_str(" An edited dose record changes the doses without changing any observation."),
        _ => {}
    }
    let data_hash = match source {
        PopulationSource::Supplied | PopulationSource::Verify => None,
        PopulationSource::RecordedSettings { data_hash } => {
            msg.push_str(
                " It was re-read from `fit.data_path` with the fit's recorded reader \
                 settings.",
            );
            Some(data_hash)
        }
        PopulationSource::ModelFileSettings { data_hash } => {
            msg.push_str(
                " The fit records no reader settings, so it was re-read from \
                 `fit.data_path` with the model file's `[data]` renames and \
                 `[data_selection]`.",
            );
            Some(data_hash)
        }
        PopulationSource::Routed { data_hash } => {
            msg.push_str(
                " The fit records neither reader settings nor a `model_path`, so it was \
                 re-read from `fit.data_path` with the model's endpoint routing only: no \
                 `[data]` renames and no `[data_selection]` were applied.",
            );
            Some(data_hash)
        }
    };
    match data_hash {
        Some(true) => msg.push_str(" The file matches the SHA-256 the fit recorded."),
        Some(false) => msg.push_str(
            " The fit records no SHA-256 of the file, so it may have changed since the fit.",
        ),
        None => {}
    }
    EngineError::with_suggestion_in_display(
        crate::diagnostics::Diagnostic::error("E_POPULATION_MISMATCH", msg)
            .with_suggestion(POPULATION_MISMATCH_SUGGESTION),
    )
}

/// A setting outside the domain `[fit_options]` accepts for its key (#1770): the one copy
/// of that domain for the keys a run records, shared by `fit()`'s option checks, `run_sir`,
/// the `.fitrx` writer and its reader. `field` is the key as the caller spells it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OutOfDomain {
    pub(crate) field: &'static str,
    pub(crate) why: String,
    pub(crate) got: String,
}

impl OutOfDomain {
    /// The refusal with `field` qualified by the record it was read from, as the `.fitrx`
    /// reader and writer report it: `sir.settings.n_agq must be …`.
    pub(crate) fn in_block(&self, block: &str) -> String {
        format!("{block}.{self}")
    }
}

/// `ode_reltol` / `ode_abstol`: positive and finite.
pub(crate) fn ode_tol_in_domain(field: &'static str, v: f64) -> Result<(), OutOfDomain> {
    if v > 0.0 && v.is_finite() {
        return Ok(());
    }
    Err(OutOfDomain { field, why: "a positive finite value".into(), got: v.to_string() })
}

/// `ode_max_steps`: positive.
pub(crate) fn ode_max_steps_in_domain(v: usize) -> Result<(), OutOfDomain> {
    if v > 0 {
        return Ok(());
    }
    Err(OutOfDomain {
        field: "ode_max_steps",
        why: "a positive integer".into(),
        got: v.to_string(),
    })
}

/// `ode_stiff_abort_after`: positive or absent. `[fit_options]` reads `0` as "off", i.e.
/// `None`, so `Some(0)` names no setting a model file can express.
pub(crate) fn ode_stiff_abort_after_in_domain(v: Option<u32>) -> Result<(), OutOfDomain> {
    if v != Some(0) {
        return Ok(());
    }
    Err(OutOfDomain {
        field: "ode_stiff_abort_after",
        why: "positive or absent".into(),
        got: "0".into(),
    })
}

/// `sir_df`: `>= 1`, or `∞` (the `normal` token). `NaN` is refused.
pub(crate) fn sir_df_in_domain(field: &'static str, v: f64) -> Result<(), OutOfDomain> {
    if v >= 1.0 {
        return Ok(());
    }
    Err(OutOfDomain { field, why: ">= 1 or \"normal\"".into(), got: v.to_string() })
}
