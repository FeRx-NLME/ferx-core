//! Whether a population and solver options are the ones a fit ran on: the fingerprint check
//! behind `PopulationFingerprint::verify` and the ODE option domains. Below the estimators so
//! `types` can call it (#3041).
use crate::diagnostics::EngineError;
use crate::types::{CompiledModel, Population, PopulationDifference, PopulationFingerprint};
use crate::types::{FitOptions, GradientMethod, InnerOptimizer};

/// The note for a fit whose population fingerprint was made with another encoding
/// scheme (#1685): it cannot be compared, so the step runs as for a fit without one.
pub const STALE_FINGERPRINT_WARNING: &str = "this fit's population fingerprint is \
     from another ferx version's scheme, so the population was not verified against the one \
     the fit was given";

/// The checks after the subject list and the binding, in order: every categorical
/// covariate value against the model's levels (`E_COV_LEVEL_UNKNOWN`, #1740: the
/// subject IDs can match while a column was recoded), then, on a fit with a
/// current fingerprint, the population's content against it (#1685). On a fit
/// without one (absent or stale), the population's level index against the model's
/// level table instead (#1793): nothing else would see a population bound on its own
/// for other levels, or never or partly bound, and it would be scored silently.
pub fn check_levels_and_content(
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
pub fn verify_given_population(
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
pub enum PopulationSource {
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
pub const POPULATION_MISMATCH_SUGGESTION: &str = "use the population the fit was given, or refit";

/// The refusal for a population that is not the one the fit was given (#1685),
/// after the binding: [`mismatch_refusal`], unless an earlier cause pre-empts it.
///
/// Two causes keep their own text and code, since there the fix is a reader or a
/// binder, not another population: a population read without the model's endpoint
/// routing (`E_ENDPOINT_UNROUTED`), and one never bound — or only partly bound — for
/// the model's level block (#1647's and #1762's `E_THETA_LEVELS_DATA_UNBOUND`, from
/// `check_level_index_columns`). One bound for other levels gets the mismatch: for a
/// post-hoc step the fix is the fit's population, not a refit.
pub fn population_mismatch(
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
pub fn mismatch_refusal(d: &PopulationDifference, source: PopulationSource) -> EngineError {
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
pub struct OutOfDomain {
    pub field: &'static str,
    pub why: String,
    pub got: String,
}

impl OutOfDomain {
    /// The refusal with `field` qualified by the record it was read from, as the `.fitrx`
    /// reader and writer report it: `sir.settings.n_agq must be …`.
    pub fn in_block(&self, block: &str) -> String {
        format!("{block}.{self}")
    }
}

/// `ode_reltol` / `ode_abstol`: positive and finite.
pub fn ode_tol_in_domain(field: &'static str, v: f64) -> Result<(), OutOfDomain> {
    if v > 0.0 && v.is_finite() {
        return Ok(());
    }
    Err(OutOfDomain { field, why: "a positive finite value".into(), got: v.to_string() })
}

/// `ode_max_steps`: positive.
pub fn ode_max_steps_in_domain(v: usize) -> Result<(), OutOfDomain> {
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
pub fn ode_stiff_abort_after_in_domain(v: Option<u32>) -> Result<(), OutOfDomain> {
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
pub fn sir_df_in_domain(field: &'static str, v: f64) -> Result<(), OutOfDomain> {
    if v >= 1.0 {
        return Ok(());
    }
    Err(OutOfDomain { field, why: ">= 1 or \"normal\"".into(), got: v.to_string() })
}

/// The [`FitOptions`] of the stage that produced a fit's estimates (and ran its inline
/// covariance step), as one record (#426): every
/// option a post-hoc step reads when it reconverges the EBEs and re-scores the objective
/// at the fit's estimates. Carried on [`FitResult::scoring_settings`] (the stage that
/// produced the estimates) and inside [`SirSettings`](crate::estimation::sir_settings::SirSettings)
/// (the options SIR's draws were scored under), so [`run_covariance`](crate::run_covariance)
/// and [`run_sir`](crate::run_sir) with default options score what the fit scored.
///
/// The settings that decide how a post-hoc step reconverges the EBEs and evaluates the
/// objective at the fit's estimates. `gradient_method` is here too (#1835): it is the inner
/// η-gradient route, and it also gates the covariance R-matrix, because `gradient = fd` is
/// one flag (#1613) — the covariance step must not build an R-matrix from the `Dual2` jets
/// the fit refused. `method` and `interaction` are not here: they ride on the fit
/// itself ([`FitResult::method`], [`FitResult::interaction`]). Nor are the settings that
/// define the covariance *step* rather than the objective it differentiates
/// (`covariance_method`, `fd_hessian_step`, `analytic_cov_hessian`, `cov_inner_tol`;
/// the Hessian anchor follows the fit's method): those say what the caller asks `run_covariance` to compute, and stay
/// the caller's. `min_obs_for_convergence_check` is not either: it only counts unconverged
/// EBEs, a tally both post-hoc steps discard.
///
/// `Default` is `ScoringSettings::from_options(&FitOptions::default())`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScoringSettings {
    /// `inner_maxiter`: the iteration cap of each EBE solve.
    pub inner_maxiter: usize,
    /// `inner_tol`: the EBE convergence tolerance, as the stage ran it (after the
    /// LTBS and quadrature tightenings `fit()` applies).
    pub inner_tol: f64,
    /// `inner_restarts`: the inner multi-start of a cold EBE solve.
    pub inner_restarts: usize,
    /// `inner_screen`: the Sobol screen of that multi-start (#2013).
    ///
    /// Absent from a record written before #2013, which reads `0`: such a fit ran no
    /// screen, so a post-hoc step re-scores it without one.
    #[serde(default)]
    pub inner_screen: usize,
    /// `inner_twin_seed`: the flip-flop twin seed of that multi-start (#2095).
    ///
    /// Absent from a record written before #2095, which reads `false`: such a fit ran no
    /// twin seed, so a post-hoc step re-scores it without one.
    #[serde(default)]
    pub inner_twin_seed: bool,
    /// `mu_referencing`: the EBE warm start from μ-referenced typical values.
    pub mu_referencing: bool,
    /// `n_agq`: quadrature nodes of the scored marginal.
    pub n_agq: usize,
    /// `inner_optimizer`: the per-subject EBE solver.
    pub inner_optimizer: InnerOptimizer,
    /// `ebe_warm_start`: warm-started Nelder–Mead fallback.
    pub ebe_warm_start: bool,
    /// `ode_reltol` of every ODE solve.
    pub ode_reltol: f64,
    /// `ode_abstol` of every ODE solve.
    pub ode_abstol: f64,
    /// `ode_max_steps` per ODE segment.
    pub ode_max_steps: usize,
    /// `ode_method`: the ODE stepper.
    pub ode_method: crate::ode::OdeMethod,
    /// `ode_stiff_abort_after`: the stiff-segment abort budget.
    pub ode_stiff_abort_after: Option<u32>,
    /// `ode_auto_switch`: in-segment stepper switching under `ode_method = auto`.
    pub ode_auto_switch: bool,
    /// `gradient_method`: `Fd` when the run took its gradients by finite differences —
    /// the model's flag, the options', or an SDE model ([`GradientMethod`]'s one rule) —
    /// else `Auto`. A post-hoc step resolves it as a union with the caller's: either `Fd`
    /// wins (#1835).
    ///
    /// Absent from a `FitResult` serialised before #1835 (the fit JSON, the ferx-tools search
    /// journal), which reads `Auto`: such a fit takes the caller's route, as it did then.
    #[serde(default)]
    pub gradient_method: GradientMethod,
    /// `nn_l2_lambda`: the covariate-NN L2 strength the stage's objective carried (#1352).
    /// Unlike the solver settings above it *defines* the objective — the covariance step adds
    /// its curvature to `R` — so a recorded value wins over the caller's, whatever the caller
    /// passes: re-running `run_covariance` with default options on a regularized fit must not
    /// drop the penalty, and a non-zero λ must not be added to a fit that never had one.
    ///
    /// `None` when unrecorded (a record written before #1352): the caller's value stands.
    #[serde(default)]
    pub nn_l2_lambda: Option<f64>,
    /// `nn_smooth_lambda`: the covariate-NN smoothness strength, recorded and restored as
    /// [`nn_l2_lambda`](Self::nn_l2_lambda) is (#1352).
    #[serde(default)]
    pub nn_smooth_lambda: Option<f64>,
}

impl ScoringSettings {
    /// The scoring settings of a stage running `options`: each field copied.
    pub fn from_options(options: &FitOptions) -> Self {
        Self {
            inner_maxiter: options.inner_maxiter,
            inner_tol: options.inner_tol,
            inner_restarts: options.inner_restarts,
            inner_screen: options.inner_screen,
            inner_twin_seed: options.inner_twin_seed,
            mu_referencing: options.mu_referencing,
            n_agq: options.n_agq,
            inner_optimizer: options.inner_optimizer,
            ebe_warm_start: options.ebe_warm_start,
            ode_reltol: options.ode_reltol,
            ode_abstol: options.ode_abstol,
            ode_max_steps: options.ode_max_steps,
            ode_method: options.ode_method,
            ode_stiff_abort_after: options.ode_stiff_abort_after,
            ode_auto_switch: options.ode_auto_switch,
            gradient_method: options.gradient_method,
            nn_l2_lambda: Some(options.nn_l2_lambda),
            nn_smooth_lambda: Some(options.nn_smooth_lambda),
        }
    }

    /// The record of a run of `model` under `options`: [`from_options`](Self::from_options)
    /// with `gradient_method` the route the run actually took — `Fd` when the model's own
    /// flag, the options' or an SDE model put it on finite differences (#1835).
    /// `from_options` stays a pure copy of the options; this is what a fit records.
    ///
    /// The retired `Ad` records as `Auto`: only `fit()` refuses it (`E_AD_RETIRED`), and a
    /// post-hoc `run_sir` handed it runs the analytic route, which is what `Auto` names. So a
    /// record holds `Auto` or `Fd`, the two values the `.fitrx` decode accepts (#1860 review).
    pub(crate) fn of_run(model: &CompiledModel, options: &FitOptions) -> Self {
        let gradient_method = match GradientMethod::effective(model, options) {
            GradientMethod::Ad => GradientMethod::Auto,
            g => g,
        };
        Self { gradient_method, ..Self::from_options(options) }
    }

    /// `options` with every scoring setting replaced by this record's, unconditionally:
    /// the inverse of [`from_options`](Self::from_options), so
    /// `from_options(&rec.overwrite(o)) == rec` for any `o`. `fit()` uses it to score
    /// its in-fit SIR under the producing stage's settings (#1806). Unlike
    /// [`with_scoring_record`], a caller's non-default value does not win: a Laplace fit
    /// with an explicit `inner_tol = 1e-6` ran its stage at 1e-8, and that is what the
    /// draws must be scored at.
    ///
    /// Destructured without `..`, like [`with_scoring_record`], so a field added to the
    /// record and left out here does not compile.
    pub(crate) fn overwrite(&self, options: &FitOptions) -> FitOptions {
        let ScoringSettings {
            inner_maxiter,
            inner_tol,
            inner_restarts,
            inner_screen,
            inner_twin_seed,
            mu_referencing,
            n_agq,
            inner_optimizer,
            ebe_warm_start,
            ode_reltol,
            ode_abstol,
            ode_max_steps,
            ode_method,
            ode_stiff_abort_after,
            ode_auto_switch,
            gradient_method,
            nn_l2_lambda,
            nn_smooth_lambda,
        } = self.clone();
        FitOptions {
            nn_l2_lambda: nn_l2_lambda.unwrap_or(options.nn_l2_lambda),
            nn_smooth_lambda: nn_smooth_lambda.unwrap_or(options.nn_smooth_lambda),
            inner_maxiter,
            inner_tol,
            inner_restarts,
            inner_screen,
            inner_twin_seed,
            mu_referencing,
            n_agq,
            inner_optimizer,
            ebe_warm_start,
            ode_reltol,
            ode_abstol,
            ode_max_steps,
            ode_method,
            ode_stiff_abort_after,
            ode_auto_switch,
            gradient_method,
            ..options.clone()
        }
    }
}

impl Default for ScoringSettings {
    fn default() -> Self {
        Self::from_options(&FitOptions::default())
    }
}

impl ScoringSettings {
    /// The first field outside the `[fit_options]` domain, if any (#1770). A record holds
    /// `gradient_method` `auto` or `fd`: `ad` is retired (`E_AD_RETIRED`), and
    /// [`of_run`](Self::of_run) records a run handed it as the `auto` it ran.
    pub(crate) fn check_domain(&self) -> Result<(), OutOfDomain> {
        ode_tol_in_domain("ode_reltol", self.ode_reltol)?;
        ode_tol_in_domain("ode_abstol", self.ode_abstol)?;
        ode_max_steps_in_domain(self.ode_max_steps)?;
        n_agq_in_domain(self.n_agq)?;
        ode_stiff_abort_after_in_domain(self.ode_stiff_abort_after)?;
        for (field, v) in
            [("nn_l2_lambda", self.nn_l2_lambda), ("nn_smooth_lambda", self.nn_smooth_lambda)]
        {
            if let Some(v) = v.filter(|v| !(*v >= 0.0 && v.is_finite())) {
                return Err(OutOfDomain {
                    field,
                    why: "a non-negative finite value".into(),
                    got: v.to_string(),
                });
            }
        }
        if self.gradient_method == GradientMethod::Ad {
            return Err(OutOfDomain {
                field: "gradient_method",
                why: "auto or fd".into(),
                got: "\"ad\"".into(),
            });
        }
        Ok(())
    }
}

impl std::fmt::Display for OutOfDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} must be {}, got {}", self.field, self.why, self.got)
    }
}

/// `n_agq`: in `1..=MAX_AGQ_NODES`.
pub fn n_agq_in_domain(v: usize) -> Result<(), OutOfDomain> {
    let max = crate::estimation::gauss_hermite::MAX_AGQ_NODES;
    if (1..=max).contains(&v) {
        return Ok(());
    }
    Err(OutOfDomain { field: "n_agq", why: format!("in 1..={max}"), got: v.to_string() })
}
