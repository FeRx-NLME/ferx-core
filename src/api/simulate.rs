#![allow(unused_imports)]
//! Extracted verbatim from `api/mod.rs` (production peel). See the module-
//! doc / Key Modules table for the split rationale.
use super::*;
use crate::diagnostics::{first_error, CheckReport, Diagnostic, EngineError};
use crate::estimation::outer_optimizer::optimize_population;
use crate::estimation::parameterization::{
    chol_lt_idx, lower_tri_iter, omega_packed_len, theta_packs_log,
};
use crate::estimation::saem;
use crate::io::datareader::{
    read_nonmem_csv_filtered_mapped, read_nonmem_csv_mapped,
    read_nonmem_csv_with_covariates_filtered_mapped, read_nonmem_csv_with_covariates_mapped,
    SelectionFilter, ERR_COV_MISSING_COLUMNS, ERR_COV_NON_NUMERIC,
};
use crate::pk;
use crate::propensity_match::MatchMethod;
use crate::sim::adaptive::{
    AdaptiveRun, AdaptiveSubjectMetrics, ControllerCtx, DecisionLogEntry, DoseAction,
    DoseLedgerEntry, MonitorSpec,
};
use crate::stats::likelihood::{build_frem_r_override, foce_subject_nll_iov};
use crate::stats::residual_error::{
    compute_iwres_with_correlations, compute_r_matrix_with_correlations_cov,
    compute_r_matrix_with_correlations_scaled_cov, iwres_autocorrelation,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Simulate observations from a model with given parameters (random seed).
///
/// Data-reader warnings (e.g. missing II for ADDL doses) are not echoed here;
/// callers that obtained `population` via [`crate::read_nonmem_csv`] should inspect
/// `population.warnings` before calling this function.
///
/// An IOV model's occasions are read from `population` as given: this entry point takes
/// no `iov_occasion` rule, and a subject with no occasion labels is simulated at κ = 0
/// without a warning. For a model fitted under `iov_occasion = dose | time(...)`, use
/// [`simulate_with_options_diag`] with [`SimulateOptions::iov_occasion`], which also
/// reports `W_IOV_NO_OCCASIONS` (#1955).
///
/// # Errors
///
/// On a model/data precondition failure — a dose the model cannot route or honour, an
/// unsupported absorption or survival combination, a covariate the data does not carry, an
/// IOV model without `omega_iov`, or (under `markov`) a CTMM endpoint, which has no
/// simulation path yet. The text is exactly the `Err` [`simulate_with_options`] returns for
/// the same input (#898).
pub fn simulate(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
) -> Result<Vec<SimulationResult>, EngineError> {
    let mut rng = rand::rng();
    simulate_inner(model, population, params, n_sim, run_root(None), &mut rng)
}

/// Simulate with a fixed seed for reproducibility.
///
/// Takes no `iov_occasion` rule and returns no warnings, as [`simulate`] (#1955).
///
/// # Errors
///
/// As [`simulate`]: a precondition failure is the `Err` [`simulate_with_options`] returns for
/// the same input (#898).
pub fn simulate_with_seed(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    seed: u64,
) -> Result<Vec<SimulationResult>, EngineError> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    simulate_inner(
        model,
        population,
        params,
        n_sim,
        run_root(Some(seed)),
        &mut rng,
    )
}

/// The root a run's keyed substreams derive from (the mixture class, #2163): its seed, or
/// a fresh entropy draw for an unseeded run, which the η stream never sees.
fn run_root(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(|| {
        let mut entropy: rand::rngs::StdRng = rand::make_rng();
        entropy.random::<u64>()
    })
}

/// Options controlling [`simulate_with_options`].
#[derive(Debug, Clone, Default)]
pub struct SimulateOptions {
    /// Seed for reproducibility. `None` draws from entropy.
    pub seed: Option<u64>,
    /// When `Some(method)`, reassign each replicate's drawn etas to subjects by
    /// **propensity-score matching** against the subjects' fitted (posthoc)
    /// etas — Mahalanobis matching under the model `Ω` via the chosen
    /// [`MatchMethod`]. This restores the design↔eta association present in
    /// adaptively-dosed real-world data and corrects the resulting VPC bias
    /// (see [`crate::propensity_match`]). `None` disables matching.
    ///
    /// Requires `population` to be observed data: every subject must carry
    /// observations so its posthoc eta can be computed. Has no effect for the
    /// synthetic `[simulation]` block (no observed designs to match against).
    pub match_method: Option<MatchMethod>,
    /// Administrative censoring horizon for TTE endpoints (#522). When `Some(t)`,
    /// `t` overrides every TTE record's per-record `observation_window` so a
    /// re-simulated event-bearing subject censors at the planned study end `t`
    /// rather than drawing unbounded — the decoupled horizon a competing-risks VPC
    /// needs. `None` keeps the per-record window. No effect on Gaussian endpoints.
    pub horizon: Option<f64>,
    /// The model-side IOV occasion rule (`iov_occasion = dose | time(...)` in
    /// `[fit_options]`, #1955). For a model that declares κ, a `PerDose` or `TimeWindows`
    /// rule re-derives every subject's occasion labels from its own timeline before
    /// anything else runs, the propensity branch's posthoc pass included, exactly as
    /// `fit()` derives them. One κ is then drawn per derived occasion.
    ///
    /// `None` and `Some(IovOccasionRule::Column)` keep the population's labels as given.
    /// A subject with none is simulated at κ = 0, which the `_diag` form reports as
    /// `W_IOV_NO_OCCASIONS`. Pass the rule the model was fitted under:
    /// [`FitResult::iov_occasion`], or the model file's `[fit_options] iov_occasion`.
    pub iov_occasion: Option<IovOccasionRule>,
}

/// What a `simulate*` entry point lets its caller supply as the TTE administrative
/// horizon, for [`validate_tte_simulatable`]'s refusal when one is required.
#[cfg(feature = "survival")]
#[derive(Clone, Copy, Debug)]
pub(crate) enum TteHorizon {
    /// The entry point takes a horizon; this is the one supplied (`None`: not set).
    Given(Option<f64>),
    /// The entry point takes none (`simulate_with_uncertainty`).
    NotAccepted,
}

/// Validate the preventable preconditions of TTE event-time simulation, returning a
/// clear `Err` for a caller to act on.
///
/// **Repeated events (RTTE, `type = rtte`, Slice 3.3)** are simulated as a recurrent
/// stream by `simulate_rtte_stream` (analytic hazards only). Several preconditions are
/// *user-fixable* and would otherwise corrupt the stream silently, so they are rejected
/// here (a clean `Err` from every `simulate*` entry point):
/// - a **finite, positive horizon** is required — the recurrent stream is generated up
///   to that administrative window; there is no implicit one;
/// - **left truncation** (`entry_time > 0`) *is* supported (#740): the stream is drawn on
///   the time origin conditioned on survival to entry, the simulate dual of the fit-side
///   conditioning (clock-forward seeds its clock at entry; clock-reset conditions its first
///   sojourn, convention B). `entry_time == 0` is byte-identical to the non-truncated draw;
/// - exactly **one recurrent stream per subject** — multiple RTTE CMTs, or an RTTE
///   cause mixed with a competing single-event TTE cause, are a later slice (they need
///   a shared-horizon multi-stream draw);
/// - **EVID-3/4 resets** are rejected — a reset would disturb the hazard clock
///   mid-stream (selective per-state reset is a later slice).
///
/// For **ODE-accumulated (joint PK-TTE)** endpoints, a model with no such endpoint is
/// otherwise unaffected (returns `Ok`).
///
/// Drug-driven event times are sampled by integrating the augmented ODE until the
/// cumulative hazard reaches `−log u` (Slice 2.2). Two conditions cannot be sampled
/// and are *user-fixable*, so they are reported here rather than panicking deep in
/// sampling:
/// - a **finite, positive horizon** is required — a drug-driven hazard can vanish
///   and never fire, so there is no implicit observation window to censor at;
/// - **EVID-3/4 resets** and **left truncation** (`entry_time > 0`) on an ODE-TTE
///   subject are unsupported (a full reset would zero the hazard accumulator
///   mid-flight; conditional sampling past entry is a deferred follow-up).
///
/// A genuinely divergent hazard (negative / non-finite) is *not* preventable here —
/// it surfaces loudly during integration (never as a silent censor).
///
/// `horizon` says what the caller could have supplied. Where a horizon is required and
/// missing, a caller that takes one is told to set it (`E_SIMULATE_HORIZON`); a caller
/// that takes none (`simulate_with_uncertainty`) cannot act on that, so it is told the
/// model cannot be simulated there and where it can (`E_TTE_NOT_SIMULATABLE`, #1774).
#[cfg(feature = "survival")]
pub(crate) fn validate_tte_simulatable(
    model: &CompiledModel,
    population: &Population,
    horizon: TteHorizon,
) -> Result<(), EngineError> {
    use super::entry_refusals::{simulate_horizon, tte_not_simulatable};
    // The refusal for a required horizon that is missing or not finite and positive.
    // `what` names the endpoint kind, `why` the reason it needs one.
    let missing_horizon = |what: &str, why: &str| match horizon {
        TteHorizon::Given(_) => simulate_horizon(format!(
            "{what} simulation requires a finite, positive administrative horizon: set \
             `[simulation] horizon` (or `SimulateOptions.horizon`). {why}"
        )),
        TteHorizon::NotAccepted => tte_not_simulatable(format!(
            "{what} simulation requires a finite, positive administrative horizon, and \
             `simulate_with_uncertainty` takes none (`SimulateUncertaintyOptions` has no \
             horizon). {why} Simulate this model with `simulate_with_options` and \
             `SimulateOptions.horizon` instead."
        )),
    };
    let horizon = match horizon {
        TteHorizon::Given(h) => h,
        TteHorizon::NotAccepted => None,
    };
    use crate::types::{EndpointLikelihood, HazardSpec, ObsRecord, TteRecurrence};

    // RTTE (repeated-event, Slice 3.3) preconditions. The stream generator
    // (`survival::simulate_rtte_stream`) handles a single analytic recurrent cause
    // per subject; the cases below are user-fixable and would otherwise corrupt the
    // draw silently, so reject them at every simulate entry point (`simulate*`,
    // uncertainty).
    let is_rtte = |cmt: &usize| {
        matches!(
            model.endpoints.get(cmt),
            Some(EndpointLikelihood::Tte {
                recurrence: TteRecurrence::Repeated { .. },
                ..
            })
        )
    };
    let is_single_tte = |cmt: &usize| {
        matches!(
            model.endpoints.get(cmt),
            Some(EndpointLikelihood::Tte {
                recurrence: TteRecurrence::Single,
                ..
            })
        )
    };
    let has_rtte = model.endpoints.values().any(|e| {
        matches!(
            e,
            EndpointLikelihood::Tte {
                recurrence: TteRecurrence::Repeated { .. },
                ..
            }
        )
    });
    if has_rtte {
        // A finite, positive administrative horizon bounds the recurrent stream.
        match horizon {
            Some(h) if h.is_finite() && h > 0.0 => {}
            _ => {
                return Err(missing_horizon(
                    "Repeated time-to-event (RTTE, `type = rtte`)",
                    "The recurrent event stream is generated up to that horizon.",
                ));
            }
        }
        for subject in &population.subjects {
            // Distinct RTTE CMTs on this subject, plus whether it also carries a
            // competing single-event TTE cause (a non-recurrent `Tte` sibling).
            let mut rtte_cmts = std::collections::BTreeSet::new();
            let mut has_single_tte_sibling = false;
            for r in &subject.obs_records {
                let ObsRecord::Event { cmt, .. } = r else {
                    continue;
                };
                // Left truncation (delayed entry, entry_time > 0) IS supported for RTTE
                // simulation (#740): the stream is drawn on the time origin conditioned on
                // survival to entry — clock-forward seeds its conditioning clock at entry,
                // clock-reset conditions its first sojourn (convention B). This is the
                // simulate dual of the fit-side conditioning; see
                // [`crate::survival::simulate_rtte_stream`].
                if is_rtte(cmt) {
                    rtte_cmts.insert(*cmt);
                } else if is_single_tte(cmt) {
                    has_single_tte_sibling = true;
                }
            }
            if rtte_cmts.is_empty() {
                continue;
            }
            if rtte_cmts.len() > 1 {
                return Err(tte_not_simulatable(format!(
                    "RTTE simulation supports one recurrent stream per subject, but subject '{}' \
                     has RTTE endpoints on multiple CMTs ({:?}); multi-cause RTTE simulation is a \
                     later slice",
                    subject.id, rtte_cmts
                )));
            }
            if has_single_tte_sibling {
                return Err(tte_not_simulatable(format!(
                    "RTTE simulation does not support an RTTE cause combined with a competing \
                     single-event TTE cause on subject '{}'; simulate them separately",
                    subject.id
                )));
            }
            if !subject.reset_times.is_empty() {
                return Err(tte_not_simulatable(format!(
                    "RTTE simulation does not support EVID=3/4 resets (a reset would disturb the \
                     hazard clock mid-stream; selective per-state reset is a later slice); subject \
                     '{}' has resets",
                    subject.id
                )));
            }
        }
    }
    let is_ode_tte = |cmt: &usize| {
        matches!(
            model.endpoints.get(cmt),
            Some(EndpointLikelihood::Tte {
                hazard: HazardSpec::OdeAccumulated { .. },
                ..
            })
        )
    };
    if !model.endpoints.values().any(|e| {
        matches!(
            e,
            EndpointLikelihood::Tte {
                hazard: HazardSpec::OdeAccumulated { .. },
                ..
            }
        )
    }) {
        return Ok(());
    }
    match horizon {
        Some(h) if h.is_finite() && h > 0.0 => {}
        _ => {
            return Err(missing_horizon(
                "ODE-accumulated TTE (joint PK-TTE)",
                "A drug-driven hazard can vanish, so there is no implicit observation window.",
            ));
        }
    }
    for subject in &population.subjects {
        if !subject
            .obs_records
            .iter()
            .any(|r| matches!(r, ObsRecord::Event { cmt, .. } if is_ode_tte(cmt)))
        {
            continue;
        }
        if !subject.reset_times.is_empty() {
            return Err(tte_not_simulatable(format!(
                "ODE-accumulated TTE simulation does not support EVID=3/4 resets (a full reset \
                 zeros the cumulative hazard mid-flight; selective per-state reset is a later \
                 slice); subject '{}' has resets",
                subject.id
            )));
        }
        if let Some(ObsRecord::Event { entry_time, .. }) = subject
            .obs_records
            .iter()
            .find(|r| matches!(r, ObsRecord::Event { cmt, entry_time, .. } if is_ode_tte(cmt) && *entry_time > 0.0))
        {
            return Err(tte_not_simulatable(format!(
                "ODE-accumulated TTE simulation does not support left truncation \
                 (entry_time={entry_time} for subject '{}'); conditional sampling past entry is \
                 a deferred follow-up",
                subject.id
            )));
        }
    }
    Ok(())
}

/// Reject a `params` value that cannot drive an IOV (`kappa`) simulation.
///
/// An IOV model draws one κ ~ N(0, Ω_IOV) per occasion, so `params.omega_iov` must
/// be present whenever `model.n_kappa > 0`. The parser guarantees that for
/// `CompiledModel::default_params`, but a caller that *rebuilds* `ModelParameters`
/// from a fit — e.g. the R `ferx_simulate(..., fit = f)` bridge (#1019) — can drop
/// the IOV block and reach the emitter with `None`. That is a caller bug rather than
/// a user-fixable model error, but it must be reported, not panicked across an FFI
/// boundary. Silently substituting the model's *initial* Ω_IOV is not an option: it
/// would under- or over-disperse every simulated occasion with no diagnostic.
pub(crate) fn validate_iov_simulatable(
    model: &CompiledModel,
    params: &ModelParameters,
) -> Result<(), EngineError> {
    if model.n_kappa > 0 && params.omega_iov.is_none() {
        // `E_PARAM_SHAPE`'s cause, "no Ω_IOV on a model with κ", so its code (#1774); the
        // message stays #1019's, which names the remedy a rebuilt parameter set needs.
        let message = format!(
            "model declares {} kappa (IOV) but the supplied parameters carry no omega_iov; \
             simulation draws one kappa vector per occasion from it. Rebuild the parameters \
             with the fitted IOV covariance (see `fitted_params_from_result`) before simulating",
            model.n_kappa
        );
        return Err(super::entry_refusals::param_shape(message));
    }
    if let Some(omega_iov) = &params.omega_iov {
        crate::stats::nested_levels::check_cross_level_omega_iov(model, &omega_iov.matrix)?;
    }
    Ok(())
}

/// Refuse a θ vector whose length is not the model's (#1614, #1615), as `E_THETA_LENGTH`.
///
/// Nothing downstream can: the compiled closures read θ by position, a θ past the end
/// reads `0.0` and a level θ past the end `NaN`, so a short vector simulates or predicts
/// finite, plausible-looking rows and a long one silently drops its tail. Every entry
/// point that takes a parameter vector runs this one function: the static `simulate*`
/// ones through `check_simulate_preconditions`, the two adaptive ones directly, since
/// they do not run that list, and `predict_diag` (so `predict`), `predict_survival`,
/// `predict_categorical` and `compute_npde_npd` directly (#1615).
///
/// The usual way to get here is a level-block model: its θ count is set by the data
/// it was bound against, so a fit's θ only fits a design bound with the *fit's*
/// level bindings. The message says so whenever the model declares a level block. It
/// names no function, since a wrapper reaches it too; from Rust, the binder is
/// [`bind_from_fit`](crate::bind_from_fit) with the fit's `data_bindings`.
pub(crate) fn check_theta_length(model: &CompiledModel, theta: &[f64]) -> Result<(), EngineError> {
    let expected = model.default_params.theta.len();
    if theta.len() == expected {
        return Ok(());
    }
    let mut message = format!(
        "the supplied theta has {} values but this model has {expected}; the model reads \
         theta by position, so these values would be read against the wrong parameters",
        theta.len()
    );
    let blocks: Vec<&str> = model
        .theta_blocks()
        .level_blocks()
        .iter()
        .map(|d| d.name())
        .collect();
    if !blocks.is_empty() {
        message.push_str(&format!(
            ". The model declares the theta level block(s) {}, whose theta count is set by \
             the data the model was bound against. A fit's theta fits only a design bound \
             against that fit's level bindings, which give the design the fit's theta layout",
            blocks
                .iter()
                .map(|b| format!("`{b}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Err(super::entry_refusals::theta_length(message))
}

/// The shape of one random-effect or residual block a caller supplied, for
/// [`check_param_shape`] (#1764). Each entry point passes the blocks it reads.
///
/// Ω and Ω_IOV carry `(rows, cols)`, not one dimension (#1833): a 1×2 block has the
/// model's row count, and a row-count check admitted it into a Cholesky that panicked.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ParamBlock {
    /// Ω's `(rows, cols)`.
    Omega((usize, usize)),
    /// σ's length.
    Sigma(usize),
    /// Ω_IOV's `(rows, cols)`, `None` when absent.
    OmegaIov(Option<(usize, usize)>),
}

impl ParamBlock {
    /// Ω, σ and Ω_IOV of a parameter set. An `OmegaMatrix`'s fields are `pub`, so a
    /// struct-literal one can be non-square too: read the matrix, not `dim()`.
    pub(crate) fn all_of(params: &ModelParameters) -> [ParamBlock; 3] {
        let shape = |m: &crate::types::OmegaMatrix| m.matrix.shape();
        [
            ParamBlock::Omega(shape(&params.omega)),
            ParamBlock::Sigma(params.sigma.values.len()),
            ParamBlock::OmegaIov(params.omega_iov.as_ref().map(shape)),
        ]
    }
}

/// Refuse an Ω, σ or Ω_IOV whose dimension is not the model's (#1764), or an Ω /
/// Ω_IOV that is not square (#1833), as `E_PARAM_SHAPE` — the sibling of
/// [`check_theta_length`].
///
/// One gate per boundary where a block enters, never one per entry point (#1833). A
/// caller handing over `ModelParameters` gates them here through [`ParamBlock::all_of`];
/// a `FitResult` is gated inside `fitted_params_from_result`, the one way a post-hoc step
/// gets parameters from a fit. Do not add a second gate on a fit: route through
/// `fitted_params_from_result`, which checks the fit's blocks before anything reads them.
/// A copy ahead of it rejects exactly the same inputs, so the tests cannot tell when
/// either copy is removed.
///
/// Measured before this gate: a mis-sized Ω panicked in the η draw or the inner
/// solve (`Gemv: dimensions mismatch`), a short σ panicked indexing the residual
/// variance, a long σ or an Ω_IOV on a model without κ ran to `Ok` with the block
/// silently ignored, and a fit with no Ω_IOV was run post hoc on the model's initial
/// one. `predict` and the survival predictors read none of the three (η = 0), so they
/// do not call this; `compute_npde_npd` leaves out an absent Ω_IOV, which it documents
/// as κ = 0.
pub(crate) fn check_param_shape(
    model: &CompiledModel,
    supplied: &[ParamBlock],
) -> Result<(), EngineError> {
    let by_position = |what: &str| {
        format!(
            "the model reads it by position, so its values would be read against the wrong {what}"
        )
    };
    // Squareness first, so a 2×1 is never reported as "2×2" (#1833).
    let not_square = |name: &str, (r, c): (usize, usize)| {
        format!(
            "the supplied {name} is {r}×{c}, which is not square; a variance-covariance \
             matrix has one row and one column per random effect"
        )
    };
    let want = &model.default_params;
    for block in supplied {
        let message = match *block {
            ParamBlock::Omega((r, c)) if r != c => not_square("omega", (r, c)),
            ParamBlock::OmegaIov(Some((r, c))) if r != c => not_square("omega_iov", (r, c)),
            ParamBlock::Omega((got, _)) if got != want.omega.dim() => format!(
                "the supplied omega is {got}×{got} but this model has {} eta; {}",
                want.omega.dim(),
                by_position("random effects")
            ),
            ParamBlock::Sigma(got) if got != want.sigma.values.len() => format!(
                "the supplied sigma has {got} values but this model has {}; {}",
                want.sigma.values.len(),
                by_position("residual errors")
            ),
            ParamBlock::OmegaIov(got) => match (
                got.map(|(r, _)| r),
                want.omega_iov.as_ref().map(|m| m.dim()),
            ) {
                (Some(g), Some(w)) if g != w => format!(
                    "the supplied omega_iov is {g}×{g} but this model has {w} kappa; {}",
                    by_position("random effects")
                ),
                (None, Some(w)) => format!(
                    "the supplied parameters carry no omega_iov but this model has {w} kappa; \
                     without it every occasion would get the same parameters"
                ),
                (Some(g), None) => format!(
                    "the supplied parameters carry a {g}×{g} omega_iov but this model has no \
                     kappa; the inter-occasion variability it describes would be dropped"
                ),
                _ => continue,
            },
            _ => continue,
        };
        return Err(super::entry_refusals::param_shape(message));
    }
    Ok(())
}

/// The model/data preconditions shared by `simulate`, `simulate_with_seed`,
/// `simulate_with_options{,_diag}` and `simulate_with_uncertainty`, as an `Err` carrying the
/// bare check message — the text `fit()` gives for that precondition (#898).
/// **`simulate_adaptive*` does not call this**: it runs its own, shorter list in
/// `api/adaptive.rs` (no flip-flop and no time-varying-covariate check), so a precondition
/// added here does not reach it.
///
/// One list, two callers: [`simulate_with_options_diag`] runs it up front (its propensity
/// branch integrates every subject *before* reaching the chokepoint), and
/// `simulate_inner_with_draw` runs it for the `Vec`-returning forms and the uncertainty
/// path. `simulate()` runs no `check_model_data`, so without these a modeled-`RATE` dose
/// (#324), an unroutable dose compartment (#375), an unsupported transit / IG closed form, a
/// twin-less flip-flop typical value, a time-varying covariate on a hazard (#741 — it would
/// be silently frozen), an individual parameter that reads `TIME` and reaches an
/// ODE-accumulated hazard (#1853 — the same freeze on the `TIME` axis, and
/// `draw_ode_tte_latent` root-finds that frozen snapshot for its event times), a
/// depot-reading analytic readout on a subject superposition cannot reconstruct (#650 / #375 /
/// #1488) or a malformed built-in absorption input rate (#588) reaches the emitter and
/// produces wrong rows or a panic deep in the prediction loop.
///
/// The list has two halves (#1288): the checks that read `params`, which run on every call,
/// and the ones that read only the model and the population ([`DrawIndependentChecked`]),
/// which `checked` lets a caller that already ran them on this same model and population
/// skip — `simulate_with_uncertainty` reaches here once per draw, and those recompute one
/// verdict each time. A `checked` for any other model or population is ignored, so the
/// argument can save work but never a check. The return is the proof for the next call.
fn check_simulate_preconditions<'a>(
    model: &'a CompiledModel,
    population: &'a Population,
    params: &ModelParameters,
    checked: Option<DrawIndependentChecked<'_>>,
) -> Result<DrawIndependentChecked<'a>, EngineError> {
    let theta = &params.theta;
    check_theta_length(model, theta)?;
    // A missing Ω_IOV keeps its own message (#1019), so it is checked before the shape.
    validate_iov_simulatable(model, params)?;
    check_param_shape(model, &ParamBlock::all_of(params))?;
    let checked = match checked {
        Some(c) if c.covers(model, population) => DrawIndependentChecked { model, population },
        _ => DrawIndependentChecked::run(model, population)?,
    };
    // ρ's range at the value drawn at (#1976).
    first_error(&super::validation::check_residual_correlation_value(
        model, population, params,
    ))?;
    // The one closed-form guard that reads θ: a draw can land in the flip-flop regime
    // its point estimate is clear of.
    first_error(&check_absorption_flip_flop_no_twin_diags(
        model, population, theta,
    ))?;
    Ok(checked)
}

/// Proof that the half of [`check_simulate_preconditions`] that reads no parameter value
/// passed on this model and population (#1288). Only [`DrawIndependentChecked::run`] makes
/// one, and it remembers what it checked, so it cannot vouch for another population.
#[derive(Clone, Copy)]
struct DrawIndependentChecked<'a> {
    model: &'a CompiledModel,
    population: &'a Population,
}

impl<'a> DrawIndependentChecked<'a> {
    /// Run the draw-independent checks. None reads `params`: `check_absorption_dosing`
    /// reads the model's own default θ, the same for every draw.
    fn run(model: &'a CompiledModel, population: &'a Population) -> Result<Self, EngineError> {
        super::note_draw_independent_run();
        // Nested levels (#1082 PR4): the κ draw groups records by their level labels, so a
        // record with none is refused, as `fit()` refuses it.
        first_error(&super::validation::check_level_labels(model, population))?;
        // The `correlation` line (#1976): the data checks `fit()` runs (RESET / TIES).
        first_error(&super::validation::check_residual_correlation_data(
            model, population,
        ))?;
        first_error(&check_modeled_dose_rates(model, population))?;
        first_error(&check_dose_compartments(model, population))?;
        first_error(&check_absorption_closed_form_support_diags(
            model, population,
        ))?;
        #[cfg(feature = "survival")]
        first_error(&check_survival_tv_covariates(model, population))?;
        // Same freeze on the `TIME` axis — `draw_ode_tte_latent` root-finds the same frozen
        // snapshot for its event times (#1853; as fit() does).
        #[cfg(feature = "survival")]
        first_error(&check_survival_time_dependent_pk_diags(model))?;
        // A Form C readout of the oral depot amount on a subject whose doses superposition
        // cannot reconstruct — an EVID=3/4 reset (#650) or a dose outside the default
        // compartment (#375). Without it a release build returns rows from a phantom depot
        // and a debug build panics in `pk::analytical_state_at_times` (#1488; as fit() and
        // predict_diag() do).
        first_error(&check_analytic_readout_support(model, population))?;
        first_error(&check_absorption_dosing(model, population))?;
        Ok(Self { model, population })
    }

    /// Whether this proof is for `model` and `population` — the same objects, not equal ones.
    fn covers(&self, model: &CompiledModel, population: &Population) -> bool {
        std::ptr::eq(self.model, model) && std::ptr::eq(self.population, population)
    }
}

/// Simulate observations under `opts`, returning only the observation rows.
///
/// Thin wrapper over [`simulate_with_options_diag`] that discards the per-subject
/// simulation diagnostics ([`SimulationOutput::warnings`]). Use the `_diag` form when a
/// degenerate-subject warning (#762 / #763) must be surfaced (e.g. a population VPC).
pub fn simulate_with_options(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    opts: &SimulateOptions,
) -> Result<Vec<SimulationResult>, EngineError> {
    simulate_with_options_impl(model, population, params, n_sim, opts, false).map(|o| o.results)
}

/// Simulate observations, optionally with propensity-score matching.
///
/// With `opts.match_method == None` this is identical to
/// [`simulate_with_seed`] (or [`simulate`] when `opts.seed` is `None`). With a
/// `Some(method)`, the freshly drawn etas of each replicate are reassigned to
/// subjects so each subject's observed design is paired with a drawn eta close
/// (under the model `Ω` Mahalanobis metric) to that subject's fitted eta. The
/// fitted (posthoc) etas are computed once from `params` + the observed
/// `population`.
///
/// Returns `Err` if matching is requested but the population is empty, or any
/// subject has no observations or carries a non-finite DV (a `DV = .` design
/// template read by [`crate::api::read_population_for_simulation`] is a
/// simulation input; there is nothing to compute a posthoc eta from).
///
/// This is the diagnostics-returning form: the [`SimulationOutput`] carries both the
/// rows and any non-fatal warnings. [`simulate_with_options`] is the thin wrapper that returns
/// only the rows.
///
/// Five sources feed `warnings`, in this order: per-subject simulation diagnostics (a
/// degenerate hazard draw, #763; a degenerate recurrent stream skipped, #762), one
/// `W_SIMULATED_FLIP_FLOP` count of the simulated individuals dosed before one of their
/// observations whose sampled η entered a twin-less transit / IG closed form's flip-flop
/// regime, so their entire dose-driven concentration was clamped to zero (#1823), capped
/// steady-state equilibrations (#867), `W_IOV_NO_OCCASIONS` for an IOV model handed
/// subjects with no occasion labels after `opts.iov_occasion` is applied (#1955), and —
/// new in #1280 / #1304 — the shared non-fit
/// diagnostics bundle (parse warnings, data-reader warnings, the model/data checks,
/// experimental-feature notices, and this pass's ODE-solver diagnostics). Before it, a model
/// `fit()` refuses to stay quiet about served the same rows here with nothing attached.
/// `postfit::non_fit_diagnostics` documents exactly what that bundle includes and what it
/// deliberately leaves out (the findings whose subject is a fit).
///
/// [`SimulationOutput::parameters`] carries the η, κ, individual parameters and
/// `[derived]` / `[output]` values each simulated subject was drawn with, one row per
/// `(draw, sim, id)` (#2505); [`SimulatedParameters`] documents the convention.
pub fn simulate_with_options_diag(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    opts: &SimulateOptions,
) -> Result<SimulationOutput, EngineError> {
    simulate_with_options_impl(model, population, params, n_sim, opts, true)
}

/// [`simulate_with_options_diag`], building the drawn-parameter table only when
/// `with_parameters` (#2505): the row-only wrapper discards it, so it does not pay for it.
fn simulate_with_options_impl(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    opts: &SimulateOptions,
    with_parameters: bool,
) -> Result<SimulationOutput, EngineError> {
    use rand::SeedableRng;

    // Start the SS-equilibration non-convergence sink clean, then drain it into this run's
    // `warnings` at each return (#867): a capped nonlinear pulse-train can silently under-report
    // the SS trough, biasing simulated concentrations low.
    crate::dosing::clear_ss_nonconvergence_warnings();

    // A derived rule on a nested model would regroup its level cells (#1082 PR4), the
    // refusal `fit()` gives the same rule.
    if let Some(levels) = model.nested_levels() {
        if opts
            .iov_occasion
            .as_ref()
            .is_some_and(|r| *r != IovOccasionRule::Column)
        {
            return Err(super::entry_refusals::level_mixed_with_iov_rule(
                &levels.levels[0].name,
            ));
        }
    }
    // A model-side occasion rule's labels come first (#1955), so every step below sees
    // them: the checks, the propensity branch's posthoc pass, and the κ draw. A nested
    // model's level cells are assigned here instead (#1082 PR4).
    let derived = with_derived_occasions(model, population, opts.iov_occasion.as_ref());
    let population: &Population = &derived;
    let iov_warning = no_occasion_warning(
        params.omega_iov.is_some(),
        population,
        "no kappa was drawn for them: they were simulated",
        "Label the occasions in the data (`iov_column` in [fit_options]), or derive them with \
         `iov_occasion = dose | time(...)`; `simulate_with_options` takes the rule as \
         `SimulateOptions::iov_occasion`.",
    );

    // ODE-accumulated (joint PK-TTE) TTE simulation samples drug-driven event times
    // via the augmented-ODE root-finder (Slice 2.2). Validate its preventable
    // preconditions up front so a caller gets a clean Err here rather than a panic
    // deep in sampling (a finite horizon is required; resets and left truncation are
    // not yet supported for an ODE hazard).
    #[cfg(feature = "survival")]
    validate_tte_simulatable(model, population, TteHorizon::Given(opts.horizon))?;

    // An IOV model needs the fitted Ω_IOV in `params` (#1019). Report a missing one
    // as a clean Err here; `simulate` / `simulate_with_seed` get the same `Err` from
    // `check_simulate_preconditions` at the chokepoint below.
    validate_iov_simulatable(model, params)?;

    // Parity with `fit()`: a referenced covariate absent from the data would
    // silently read 0.0 (e.g. a `Selected` error model's `if (FREE==0)` selector
    // would route every row to branch 0, applying the wrong residual variance
    // with no diagnostic), a blank arm-size cell would collapse a weighted κ
    // (#1031), and a blank standard-error cell would collapse a `weight = <expr>`
    // residual magnitude (#1029). One shared list so no simulate entry point can
    // carry a different subset than the others (#658 / #1083). Only the fatal
    // half; the within-occasion variation warning belongs to `fit()`'s warning
    // list.
    first_error(&check_simulation_data(model, population))?;

    // Validate the TTE horizon on the library path too — the `.ferx` parser
    // already rejects a non-finite / non-positive horizon, but a direct caller of
    // this API must get the same guard: a NaN window makes every `t_event < window`
    // test false (silent NaN event times), and a `<= 0` horizon censors every
    // subject at or before entry (#522 review).
    if let Some(h) = opts.horizon {
        if !h.is_finite() || h <= 0.0 {
            return Err(super::entry_refusals::simulate_horizon(format!(
                "SimulateOptions.horizon must be finite and > 0 (got {h})"
            )));
        }
        // A horizon below a subject's TTE entry_time would censor it before it
        // entered observation (a row with time = h < entry_time). The
        // `[simulation]`-block path always enters at 0, but a left-truncated
        // population passed to this API must be rejected (#522 review).
        #[cfg(feature = "survival")]
        for subject in &population.subjects {
            for record in &subject.obs_records {
                let crate::types::ObsRecord::Event { entry_time, .. } = record else {
                    continue;
                };
                if *entry_time > h {
                    return Err(super::entry_refusals::simulate_horizon(format!(
                        "SimulateOptions.horizon ({h}) is below subject '{}' entry_time \
                         ({entry_time}); the administrative horizon must be ≥ every \
                         subject's entry time",
                        subject.id
                    )));
                }
            }
        }
    }

    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::make_rng(),
    };
    let class_root = run_root(opts.seed);

    // Guard the modeled-`RATE` dose precondition up front (#324). The
    // non-propensity branch reaches it via the `simulate_inner_with_draw`
    // chokepoint, but the propensity branch first runs a full inner EBE pass
    // (`run_inner_loop_warm` below) that integrates every subject — on an
    // unsupported config that would hit the per-path tripwire (silently in
    // release) or `resolve_rate`'s opaque `.expect` *before* the chokepoint
    // guard. Checking here makes both branches return the same actionable
    // `Err` (#898: the text `fit()` returns for the same input, not a panic out
    // of a `Result`-returning function); it is a no-op O(doses) scan on the
    // common all-`Fixed` dataset.
    // The built-in absorption input-rate guard (#588) is hoisted for the same
    // reason: otherwise a malformed multi-pathway / SS / infusion absorption model
    // integrates the whole warm-EBE pass first and fails only at the chokepoint,
    // with a confusable "EBE did not converge" instead of the real cause.
    // The chokepoint then reruns only the half that reads `params` (#1288).
    let checked = check_simulate_preconditions(model, population, params, None)?;
    let mut table = SimulatedParameters::for_model(model);

    let method = match opts.match_method {
        Some(m) => m,
        None => {
            let mut warnings = Vec::new();
            let mut flip_flop = SimulatedFlipFlopTally::default();
            // One scope around the whole pass, not one per subject: this loop is serial (the
            // RNG draw order is part of `simulate`'s contract), so the thread-local scope sees
            // every subject's integrations. #1304.
            let (results, stats) = super::with_solver_stats(model, || {
                simulate_inner_with_draw(
                    model,
                    population,
                    params,
                    n_sim,
                    1,
                    None,
                    opts.horizon,
                    Some(checked),
                    class_root,
                    &mut rng,
                    &mut warnings,
                    &mut flip_flop,
                    with_parameters.then_some(&mut table),
                )
            });
            let results = results?;
            warnings.extend(flip_flop.warning(model));
            warnings.extend(crate::dosing::take_ss_nonconvergence_warnings());
            warnings.extend(iov_warning);
            // Only this branch: a propensity-matched run reassigns the draws by each
            // subject's posthoc η, covariate η included, which is the remedy named here.
            warnings.extend(frem_drawn_eta_warning(
                model,
                population.subjects.len(),
                &population.subjects,
                "To keep each subject's PK with its own covariates, set \
                 `SimulateOptions::match_method`, which reassigns the drawn etas by matching \
                 them to each subject's posthoc eta (approximately), or simulate the fitted \
                 subjects from the FREM fit's EBEs with `simulate_adaptive_with_inputs` and \
                 `SuppliedEtas`.",
            ));
            warnings.extend(super::non_fit_diagnostics(
                model,
                population,
                params,
                &stats,
                super::SolverStatsPhase::Simulate,
            ));
            return Ok(SimulationOutput {
                results,
                warnings,
                parameters: table,
            });
        }
    };

    // Matching pairs each subject with a drawn η by its posthoc η, and on a `[mixture]`
    // model both belong to a class: the posthoc pass below estimates every subject under
    // class 1, and the pool is drawn from one Ω (#2163).
    if model.mixture.is_some() {
        return Err(super::entry_refusals::propensity_input(
            "propensity-score matching is not supported on a [mixture] model: each \
             subject's posthoc eta would be estimated under class 1 rather than its own \
             class. Simulate without `match_method`; each subject then draws its class from \
             the mixing probabilities",
        ));
    }
    if population.subjects.is_empty() {
        return Err(super::entry_refusals::propensity_input(
            "propensity-score matching requires a non-empty observed population",
        ));
    }
    if let Some(s) = population
        .subjects
        .iter()
        .find(|s| s.observations.is_empty())
    {
        return Err(super::entry_refusals::propensity_input(format!(
            "propensity-score matching requires observations for every subject \
             (to compute posthoc etas); subject '{}' has none",
            s.id
        )));
    }
    // A `DV = .` design template read by `read_population_for_simulation` (#957)
    // has rows but NaN observations, so the emptiness check above no longer
    // catches it. Its posthoc EBE would be optimized against a NaN objective —
    // either tripping the non-finite-eta guard below with a misleading "did not
    // converge", or (if the inner optimizer hands back its finite starting eta)
    // matching every subject on all-zero etas and returning an arbitrary
    // assignment. Reject it here with the real cause.
    if let Some(s) = population
        .subjects
        .iter()
        .find(|s| s.observations.iter().any(|v| !v.is_finite()))
    {
        return Err(super::entry_refusals::propensity_input(format!(
            "propensity-score matching requires finite observations for every subject \
             (to compute posthoc etas); subject '{}' has non-finite DV values. A `DV = .` \
             design template carries NaN placeholders — match against the observed dataset \
             instead",
            s.id
        )));
    }

    // Fitted (posthoc) BSV etas depend only on the observed data + params, so
    // compute them once and reuse across replicates. The inner-loop budget here
    // is a self-contained MAP pass (this entry point takes no FitOptions); the
    // tolerances only need to localize each EBE well enough to match on, not to
    // reproduce a specific fit's inner settings.
    let (eta_hats, _h, _stats, _kappas) = crate::estimation::inner_optimizer::run_inner_loop_warm(
        model, population, params, 100, 1e-6, None, None, 1, 0,
    );

    // A divergent EBE can come back non-finite (`find_ebe` only gates its
    // `converged` flag on a finite nll, not the returned eta). A NaN/Inf eta
    // would poison the Mahalanobis cost matrix and make the optimal-assignment
    // solver spin forever (NaN compares false against every candidate), so fail
    // loudly here instead.
    if let Some((i, _)) = eta_hats
        .iter()
        .enumerate()
        .find(|(_, e)| e.iter().any(|x| !x.is_finite()))
    {
        return Err(EngineError::uncoded(format!(
            "propensity-score matching: the posthoc eta for subject '{}' is \
             non-finite (its EBE did not converge); cannot match",
            population.subjects[i].id
        )));
    }

    let omega_inv = &params.omega.inv;
    let mut warnings = Vec::new();
    let mut flip_flop = SimulatedFlipFlopTally::default();
    // Same scope the non-propensity branch opens. Both branches, not the shared chokepoint
    // below them: `simulate_inner_with_draw` is also what `simulate_with_uncertainty_diag`
    // calls once per draw, and that caller opens one scope around all of its draws (with its
    // own phase) — a scope in the chokepoint would nest inside it, once per draw.
    let (results, stats) = super::with_solver_stats(model, || {
        simulate_inner_with_draw(
            model,
            population,
            params,
            n_sim,
            1,
            Some((&eta_hats, omega_inv, method)),
            opts.horizon,
            Some(checked),
            class_root,
            &mut rng,
            &mut warnings,
            &mut flip_flop,
            with_parameters.then_some(&mut table),
        )
    });
    let results = results?;
    warnings.extend(flip_flop.warning(model));
    warnings.extend(crate::dosing::take_ss_nonconvergence_warnings());
    warnings.extend(iov_warning);
    warnings.extend(super::non_fit_diagnostics(
        model,
        population,
        params,
        &stats,
        super::SolverStatsPhase::Simulate,
    ));
    Ok(SimulationOutput {
        results,
        warnings,
        parameters: table,
    })
}

fn simulate_inner<R: rand::Rng>(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    class_root: u64,
    rng: &mut R,
) -> Result<Vec<SimulationResult>, EngineError> {
    // `simulate` / `simulate_with_seed` carry no horizon; the per-record window
    // applies. An explicit `[simulation] horizon` enters via `simulate_with_options`.
    // These entry points return only the rows; per-subject simulation diagnostics
    // (#762 degenerate RTTE, #763 degenerate hazard draw) are collected but surfaced
    // only on the `simulate_with_options` path (`SimulationOutput.warnings`). The
    // same holds for the #1823 flip-flop tally: it is counted (the chokepoint owns the
    // count, so every entry point counts identically) and then dropped here, because
    // these two entry points have no diagnostics channel at all —
    // `the_row_only_entry_points_stay_silent_by_contract` pins that. Use
    // `simulate_with_options_diag` when the count matters.
    let mut warnings = Vec::new();
    let mut flip_flop = SimulatedFlipFlopTally::default();
    simulate_inner_with_draw(
        model,
        population,
        params,
        n_sim,
        1,
        None,
        None,
        // No entry point ahead of this one ran the list.
        None,
        class_root,
        rng,
        &mut warnings,
        &mut flip_flop,
        None,
    )
}

/// Emit all observation rows for one subject given a fully-formed `eta_slice`
/// (length `n_eta + n_kappa`). Draws only residual epsilons from `rng`; the eta
/// is supplied by the caller (freshly sampled, or propensity-matched).
#[allow(clippy::too_many_arguments)]
/// Display time for observation row `j`: the raw data TIME when available
/// (matches sdtab / input), falling back to the internal `obs_times` clock,
/// which may be the shifted clock for stacked reset occasions. Shared by every
/// simulation row emitter so the static and reactive paths cannot drift.
pub(crate) fn obs_row_time(subject: &Subject, j: usize) -> f64 {
    subject.readout_time(j)
}

fn emit_subject_rows<R: rand::Rng>(
    model: &CompiledModel,
    subject: &Subject,
    kappa_draw: Option<&crate::stats::nested_levels::KappaDraw>,
    params: &ModelParameters,
    eta_slice: &[f64],
    draw: usize,
    sim: usize,
    normal: rand_distr::Normal<f64>,
    horizon: Option<f64>,
    rng: &mut R,
    results: &mut Vec<SimulationResult>,
    warnings: &mut Vec<String>,
) -> DrawnPrediction {
    // `horizon` and `warnings` are consumed only by the TTE path below; without the
    // `survival` feature there are no TTE endpoints, so discard them to avoid unused-arg warns.
    #[cfg(not(feature = "survival"))]
    let _ = (horizon, &mut *warnings);

    // Predict IPRED. For IOV models (`n_kappa > 0`) route through the
    // occasion-aware `predict_iov`, drawing one independent κ ~ N(0, Ω_IOV) per
    // occasion — otherwise every occasion would share the same (κ = 0) parameters
    // and the simulated data would carry NO inter-occasion variability, silently
    // under-dispersing a VPC relative to the fitted model / NONMEM `$SIM`. The
    // caller zeroes the κ tail of `eta_slice`; only the BSV head (`..n_eta`) is
    // used here, and the κ draws happen on this occasion-aware branch instead.
    // Non-IOV models keep the TV-covariate-aware fast-path dispatcher unchanged
    // and draw no extra randoms, so their output is byte-identical.
    let mut kappas: Vec<Vec<f64>> = Vec::new();
    let ipreds = if model.n_kappa > 0 {
        let omega_iov = params.omega_iov.as_ref().expect(
            "omega_iov is present whenever the model declares kappa (n_kappa > 0) — \
                 guaranteed by `validate_iov_simulatable` at every simulate entry point (#1019)",
        );
        // One κ vector per occasion group, in `iov_occasion_groups` order — the
        // exact order `predict_iov` indexes its `kappas` argument by. Empty when
        // the subject carries no occasion labels, in which case `predict_iov`
        // falls back to κ = 0 (matching the fit-time no-occasion diagnostic).
        //
        // Under nested levels (#1082 PR4) the groups are the level cells, and a κ is
        // drawn once per group of its own level: one study κ shared by every arm of
        // the study. The caller built `kappa_draw` per subject (`KappaDraw::of`, which
        // refuses a nested subject without cells), so it is present for every κ model.
        // The factor is the declared Ω_IOV's, a zero κ exactly 0 (#1605).
        let kappa_factor = crate::stats::nested_levels::KappaDraw::factor(model, omega_iov);
        kappas = kappa_draw
            .expect("simulate_inner_with_draw builds a KappaDraw per subject of a kappa model")
            .draw(&kappa_factor, normal, rng);
        pk::predict_iov(
            model,
            subject,
            &params.theta,
            &eta_slice[..model.n_eta],
            &kappas,
        )
    } else {
        pk::compute_predictions_with_tv(model, subject, &params.theta, eta_slice)
    };

    // Add residual error (Gaussian path). IIV on residual error (#409): the
    // drawn `eta_slice` includes η_ruv, so scale the residual variance by
    // exp(2·η_ruv) — i.e. simulate `Y = IPRED + EPS·EXP(η_ruv)`.
    let ruv_scale = model.residual_var_scale(eta_slice);
    // Per-observation custom residual magnitude (#484): η-independent, so build
    // the [obs][sigma-slot] matrix once per subject and index it per row.
    let ruv_mult = model.ruv_obs_mult(subject, &params.theta);
    // `block_sigma` cross-endpoint correlations (#672): draw the full
    // multivariate residual vector from the dense `R` that estimation already
    // builds (`compute_r_matrix_with_correlations`), so the simulated data
    // reproduces the fitted covariance instead of independent per-row draws.
    // FREM covariate pseudo-observations don't participate in the correlation
    // (mirrors the `has_frem_rows` gate in
    // `stats/likelihood.rs::individual_nll_into_with_schedule`). That gate also
    // carries a `!has_censored_m3` term we deliberately drop here: it exists so
    // the likelihood's M3 BLOQ integral falls back to the scalar path, whereas
    // simulate() draws the residual first and applies censoring afterwards, so a
    // to-be-censored row should still be drawn from the correlated R. (A
    // `block_sigma` + M3 model is rejected at fit by `check_model_options`
    // regardless, so the two paths can only differ on an unfitted fixed model.)
    let has_frem_rows = subject.fremtype.iter().any(|&ft| ft > 0);
    // The **live** correlations (#847), resolved once in `draw_correlations`.
    let draw_correlations = draw_correlations(model, params);
    if draw_correlations.is_dense() && !has_frem_rows && !ipreds.is_empty() {
        emit_correlated_residual_rows(
            model,
            subject,
            params,
            draw_correlations,
            &ipreds,
            ruv_scale,
            ruv_mult.as_deref(),
            draw,
            sim,
            normal,
            rng,
            results,
        );
    } else {
        for (j, &ipred) in ipreds.iter().enumerate() {
            // FREM covariate pseudo-observations (FREMTYPE>0) use the additive
            // covariate sigma, not the PK error model applied to the θ+η override
            // that `compute_predictions_with_tv` now writes into FREM rows.
            let var = model.sim_residual_variance(
                subject,
                j,
                ipred,
                &params.sigma.values,
                draw_correlations.pairs,
                ruv_scale,
                ruv_mult.as_ref().map(|m| m[j].as_slice()),
            );
            let eps: f64 = rng.sample(normal);
            let value = ipred + var.sqrt() * eps;

            results.push(SimulationResult {
                draw,
                sim,
                id: subject.id.clone(),
                // Raw data TIME (matches sdtab / input); `obs_times` may be
                // the internal shifted clock for stacked reset occasions.
                time: obs_row_time(subject, j),
                cmt: subject.obs_cmts[j],
                ipred,
                outcome: SimOutcome::Continuous { value },
                mixnum: None, // stamped by the caller that holds the class (#2372)
            });
        }
    }

    // TTE simulation path (requires survival feature)
    #[cfg(feature = "survival")]
    crate::survival::simulate_tte(
        model,
        subject,
        &params.theta,
        eta_slice,
        draw,
        sim,
        horizon,
        rng,
        results,
        warnings,
    );

    // Binary / discrete-endpoint simulation path (#760 Slice 1b). Like the TTE path
    // above, these rows live in `obs_records` and are invisible to the Gaussian
    // emitter, so they need their own producer — without it a binary endpoint
    // contributed zero rows to `simulate()` and raised no error.
    #[cfg(feature = "survival")]
    crate::categorical::simulate_binary(
        model,
        subject,
        &params.theta,
        eta_slice,
        draw,
        sim,
        rng,
        results,
        warnings,
    );

    DrawnPrediction { ipreds, kappas }
}

/// What [`emit_subject_rows`] predicted one subject with, for the drawn-parameter table
/// (#2505): the IPRED it added residual error to, and the κ it drew (one vector per
/// occasion group, in `predict_iov` order; empty without IOV).
struct DrawnPrediction {
    ipreds: Vec<f64>,
    kappas: Vec<Vec<f64>>,
}

/// The residual covariance structure — today the `block_sigma` correlations — a
/// residual draw under `params` must use (#847, #1733, #1081). `params` is the fitted (or drawn) vector, so reading
/// `model.residual_correlations` would draw residuals at the declared rho while
/// using the fitted sigmas — exactly the mismatch that would make a VPC or an
/// NPDE of an estimated `block_sigma` fail to reproduce it. Prefer the parameter
/// vector's correlations, falling back to the model's declaration when a caller
/// supplied a `ModelParameters` without them (a hand-built one, or an artifact
/// predating the field). Dropping a declared `block_sigma` silently —
/// independent draws for a model that asks for correlated ones — is the worse of
/// the two failures.
pub(crate) fn draw_correlations<'a>(
    model: &'a CompiledModel,
    params: &'a ModelParameters,
) -> crate::stats::residual_error::ResidualCov<'a> {
    if model.residual_cov_at(params).is_dense() {
        model.residual_cov_at(params)
    } else {
        model.residual_cov_theta(&params.theta)
    }
}

/// Draw the correlated residual vector for one subject's Gaussian observation
/// rows from the dense `R` built by [`compute_r_matrix_with_correlations`] (the
/// same matrix FOCE/FOCEI/SAEM/`imp` evaluate the likelihood against), instead
/// of the per-row independent draw. Callers must already have excluded FREM
/// rows and the empty-correlation case. Shared by `simulate()` and the NPDE
/// reference simulation (#1733), so both draw one distribution.
///
/// R is factored with a PSD-safe symmetric-eigen square root, so a singular
/// (e.g. `rho = ±1`) or mildly indefinite fixed `block_sigma` yields a valid
/// draw instead of a Cholesky panic. Subjects whose R is diagonal (no paired
/// rows) take a cheap per-row draw and skip the factorization entirely.
#[allow(clippy::too_many_arguments)]
pub(crate) fn correlated_residual_draw<R: rand::Rng>(
    model: &CompiledModel,
    subject: &Subject,
    sigma_values: &[f64],
    /* live correlations, resolved by the caller */
    residual_cov: crate::stats::residual_error::ResidualCov<'_>,
    ipreds: &[f64],
    ruv_scale: f64,
    ruv_mult: Option<&[Vec<f64>]>,
    normal: rand_distr::Normal<f64>,
    rng: &mut R,
) -> DVector<f64> {
    let err_keys = model.error_spec.obs_keys(subject);
    // A `correlation` line's blocks are the subject's `obs_rgroup`: assigned once per
    // call by `simulate_inner_with_draw`, and by `fitted_population` for NPDE (#1081).
    let obs_rgroup: &[u32] = &subject.obs_rgroup;
    let mut r = match ruv_mult {
        Some(mult) => compute_r_matrix_with_correlations_scaled_cov(
            &model.error_spec,
            ipreds,
            err_keys.as_ref(),
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            &subject.obs_l2,
            obs_rgroup,
            sigma_values,
            residual_cov,
            mult,
        ),
        None => compute_r_matrix_with_correlations_cov(
            &model.error_spec,
            ipreds,
            err_keys.as_ref(),
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            &subject.obs_l2,
            obs_rgroup,
            sigma_values,
            residual_cov,
        ),
    };
    if ruv_scale != 1.0 {
        r *= ruv_scale;
    }
    let n = ipreds.len();
    // z ~ N(0, Iₙ); the correlated residual is a matrix square-root factor of R
    // times z (`Cov(F·z) = F·Fᵀ = R`).
    let z = DVector::from_iterator(n, (0..n).map(|_| rng.sample(normal)));
    // Fast path: if R has no nonzero off-diagonal the subject has no actually
    // paired rows (every observation sits in its own residual block), so R is
    // diagonal and the draw is the same independent per-row draw the scalar
    // path uses. Skip the O(n³) factorization — this keeps a densely-sampled
    // endpoint of a `block_sigma` model cheap, and reproduces the scalar path's
    // RNG-for-RNG output for such subjects.
    let has_offdiag = (0..n).any(|j| ((j + 1)..n).any(|k| r[(j, k)] != 0.0));
    if !has_offdiag {
        DVector::from_iterator(n, (0..n).map(|j| r[(j, j)].max(0.0).sqrt() * z[j]))
    } else {
        // A fitted or fixed `block_sigma` can be positive-SEMIdefinite rather
        // than strictly positive-definite — a perfect cross-endpoint
        // correlation (`rho = ±1`, which the parser accepts on the inclusive
        // [-1, 1] range) makes R singular, so a Cholesky factor doesn't exist
        // and would panic the whole simulation. Use the symmetric-eigen square
        // root `V·diag(√max(λ,0))`, which is well defined for any PSD R and
        // clamps tiny negative eigenvalues (round-off, or a mildly indefinite
        // fixed R) to zero instead of aborting.
        let eig = r.symmetric_eigen();
        let mut factor = eig.eigenvectors;
        for (k, &lambda) in eig.eigenvalues.iter().enumerate() {
            let s = lambda.max(0.0).sqrt();
            factor.column_mut(k).scale_mut(s);
        }
        factor * z
    }
}

/// Emit one subject's Gaussian observation rows with the residual vector drawn
/// by [`correlated_residual_draw`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_correlated_residual_rows<R: rand::Rng>(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    /* live correlations, resolved by the caller */
    residual_cov: crate::stats::residual_error::ResidualCov<'_>,
    ipreds: &[f64],
    ruv_scale: f64,
    ruv_mult: Option<&[Vec<f64>]>,
    draw: usize,
    sim: usize,
    normal: rand_distr::Normal<f64>,
    rng: &mut R,
    results: &mut Vec<SimulationResult>,
) {
    let eps = correlated_residual_draw(
        model,
        subject,
        &params.sigma.values,
        residual_cov,
        ipreds,
        ruv_scale,
        ruv_mult,
        normal,
        rng,
    );
    for (j, &ipred) in ipreds.iter().enumerate() {
        let value = ipred + eps[j];
        results.push(SimulationResult {
            draw,
            sim,
            id: subject.id.clone(),
            time: obs_row_time(subject, j),
            cmt: subject.obs_cmts[j],
            ipred,
            outcome: SimOutcome::Continuous { value },
            mixnum: None, // stamped by the caller that holds the class (#2372)
        });
    }
}

/// Purpose tag folded into the mixture-class seed (#2163), so the class stream is disjoint
/// from the assay and κ substreams derived from the same run root
/// (`crate::sim::adaptive`), and from the η stream, which it never draws from.
const MIXTURE_CLASS_PURPOSE_SALT: u64 = 0x4D1C_C1A5_5E5E_0001;

/// The uniform a simulated subject's mixture class is drawn with, keyed by
/// `(root, subject id, draw, sim)` (#2163). Like the κ and assay substreams it is a pure
/// function of the key: drawing it consumes nothing from the η stream, so a `[mixture]`
/// model draws the same η a model without one draws from the same seed, and subject order
/// cannot change a subject's class.
pub(crate) fn mixture_class_uniform(root: u64, subject_id: &str, draw: usize, sim: usize) -> f64 {
    use crate::sim::adaptive::{combine_seed, stable_hash_str};
    let s = combine_seed(
        root ^ MIXTURE_CLASS_PURPOSE_SALT,
        stable_hash_str(subject_id),
    );
    let key = combine_seed(combine_seed(s, draw as u64), sim as u64);
    rand::rngs::StdRng::seed_from_u64(key).random::<f64>()
}

/// The class (0-based) the uniform `u ∈ [0, 1)` selects under the per-class
/// log-probabilities `logp`: the first class whose cumulative probability exceeds `u`. A
/// class of probability 0 is never selected. Should rounding leave the cumulative sum at or
/// below `u`, the last class of positive probability is.
pub(crate) fn pick_mixture_class(logp: &[f64], u: f64) -> usize {
    let mut cumulative = 0.0;
    let mut last_possible = 0;
    for (k, lp) in logp.iter().enumerate() {
        let p = lp.exp();
        if p > 0.0 {
            last_possible = k;
        }
        cumulative += p;
        if u < cumulative {
            return k;
        }
    }
    last_possible
}

/// Refuse, as `E_PARAM_SHAPE`, parameters for a `[mixture]` model that do not carry one Ω
/// and one Σ of the model's shape per class (#2163 review). Without `params.mixture` every
/// class would be simulated at the base Ω and Σ, silently dropping an `omega(k)` /
/// `sigma(k)` override; a per-class list shorter than the class count panicked indexing it
/// in `class_params`, and a per-class block of another shape panicked in the η draw or
/// the residual variance. `ModelParameters` is a public struct, so a hand-built one can
/// carry any of these.
pub(crate) fn check_mixture_param_shape(
    model: &CompiledModel,
    spec: &MixtureSpec,
    params: &ModelParameters,
) -> Result<(), EngineError> {
    let k = spec.n_classes;
    let Some(mp) = params.mixture.as_ref() else {
        return Err(super::entry_refusals::param_shape(format!(
            "this model declares a [mixture] of {k} classes, but the supplied parameters carry \
             no per-class omega/sigma (`mixture: None`); every class would be simulated at the \
             base omega and sigma, dropping any omega(k)/sigma(k) override. Pass the parsed \
             model's `default_params`, or the fit's parameters, which carry them"
        )));
    };
    let want = &model.default_params;
    let n_eta = want.omega.dim();
    let n_sigma = want.sigma.values.len();
    let message = if mp.omega.len() != k || mp.sigma.len() != k {
        Some(format!(
            "this model declares a [mixture] of {k} classes, but the supplied parameters carry \
             {} per-class omega and {} per-class sigma; each class needs one of each",
            mp.omega.len(),
            mp.sigma.len()
        ))
    } else {
        (0..k).find_map(|c| {
            let (r, cols) = mp.omega[c].matrix.shape();
            if r != n_eta || cols != n_eta {
                Some(format!(
                    "the supplied omega of mixture class {} is {r}×{cols} but this model has \
                     {n_eta} eta; the model reads it by position",
                    c + 1
                ))
            } else if mp.sigma[c].values.len() != n_sigma {
                Some(format!(
                    "the supplied sigma of mixture class {} has {} values but this model has \
                     {n_sigma}; the model reads it by position",
                    c + 1,
                    mp.sigma[c].values.len()
                ))
            } else {
                None
            }
        })
    };
    match message {
        Some(m) => Err(super::entry_refusals::param_shape(m)),
        None => Ok(()),
    }
}

/// The parameters a simulated subject runs under (#2163). On a `[mixture]` model each
/// subject draws a class from its mixing probabilities, as NONMEM's `$SIM` does under
/// `$MIX`, and runs under that class's `MIXNUM`, Ω and Σ; on any other model every subject
/// runs under `params`.
///
/// The class is drawn per `(subject, draw, sim)` on its own keyed substream
/// ([`mixture_class_uniform`]), from the mixing probabilities at `params.theta` and the
/// subject's baseline covariates, the ones the fit scores them at. Per-class Ω and Σ are
/// `params.mixture`'s; [`SimClasses::new`] refuses parameters without them, or with a
/// per-class block of another shape, as `E_PARAM_SHAPE` ([`check_mixture_param_shape`]).
pub(crate) enum SimClasses<'a> {
    /// No `[mixture]`: `params`, and the factor η is drawn with.
    Single {
        params: &'a ModelParameters,
        eta_factor: std::borrow::Cow<'a, DMatrix<f64>>,
    },
    /// A `[mixture]` model: each class's parameters and η factor, 0-based.
    Mixture {
        spec: &'a MixtureSpec,
        root: u64,
        classes: Vec<(ModelParameters, DMatrix<f64>)>,
    },
}

/// One simulated subject's class: its parameters, the factor its η is drawn with, and,
/// on a `[mixture]` model, the guard holding `MIXNUM` at its class until it drops.
pub(crate) struct SimClass<'a> {
    /// The class, 0-based; always 0 without a `[mixture]`. Indexes a caller's per-class
    /// cache of anything that reads `MIXNUM` (`SimClasses::n_classes` entries).
    pub(crate) index: usize,
    pub(crate) params: &'a ModelParameters,
    pub(crate) eta_factor: &'a DMatrix<f64>,
    _mixnum: Option<crate::parser::model_parser::MixtureClassGuard>,
}

impl SimClass<'_> {
    /// The class as [`SimulationResult::mixnum`] reports it (#2372): `Some(index + 1)` on a
    /// `[mixture]` model, `None` without one.
    pub(crate) fn mixnum(&self) -> Option<usize> {
        self._mixnum.as_ref().map(|_| self.index + 1)
    }
}

impl<'a> SimClasses<'a> {
    /// `root` keys the class substream: the run seed, or entropy for an unseeded run.
    ///
    /// On a `[mixture]` model, `params` must carry one Ω and one Σ of the model's shape per
    /// class ([`check_mixture_param_shape`]); this is the one gate every simulate path —
    /// `simulate*`, the adaptive driver and the NPDE reference — passes through.
    pub(crate) fn new(
        model: &'a CompiledModel,
        params: &'a ModelParameters,
        root: u64,
    ) -> Result<Self, EngineError> {
        let declared = &model.default_params;
        let Some(spec) = model.mixture.as_ref() else {
            return Ok(SimClasses::Single {
                params,
                eta_factor: params
                    .omega
                    .sim_draw_factor(Some(&declared.omega), &declared.omega_fixed),
            });
        };
        check_mixture_param_shape(model, spec, params)?;
        let classes = (0..spec.n_classes)
            .map(|k| {
                let class_params = crate::estimation::mixture::class_params(params, k);
                // Each class's η from its own declared Ω, a zero η exactly 0 (#1605). An
                // `omega(k)` override carries its own FIX flag, as in the class score view.
                let mut declared_omega = &declared.omega;
                let mut declared_fixed = declared.omega_fixed.clone();
                if let Some(m) = declared.mixture.as_ref() {
                    declared_omega = m.omega.get(k).unwrap_or(declared_omega);
                    for (&(c, e), &fixed) in
                        m.omega_override_addr.iter().zip(&m.omega_override_fixed)
                    {
                        if c == k {
                            if let Some(f) = declared_fixed.get_mut(e) {
                                *f = fixed;
                            }
                        }
                    }
                }
                let factor = class_params
                    .omega
                    .sim_draw_factor(Some(declared_omega), &declared_fixed)
                    .into_owned();
                (class_params, factor)
            })
            .collect();
        Ok(SimClasses::Mixture {
            spec,
            root,
            classes,
        })
    }

    /// How many classes [`SimClass::index`] ranges over: 1 without a `[mixture]`.
    pub(crate) fn n_classes(&self) -> usize {
        match self {
            SimClasses::Single { .. } => 1,
            SimClasses::Mixture { classes, .. } => classes.len(),
        }
    }

    /// The class `subject` runs under in replicate `(draw, sim)`. Hold the result for as
    /// long as the subject is being simulated: its guard is what `MIXNUM` reads.
    pub(crate) fn enter(&self, subject: &Subject, draw: usize, sim: usize) -> SimClass<'_> {
        match self {
            SimClasses::Single { params, eta_factor } => SimClass {
                index: 0,
                params,
                eta_factor,
                _mixnum: None,
            },
            SimClasses::Mixture {
                spec,
                root,
                classes,
            } => {
                let logp = crate::parser::model_parser::eval_mixing_log_probs(
                    spec,
                    &classes[0].0.theta,
                    &subject.covariates,
                );
                let u = mixture_class_uniform(*root, &subject.id, draw, sim);
                let k = pick_mixture_class(&logp, u);
                let (params, eta_factor) = &classes[k];
                SimClass {
                    index: k,
                    params,
                    eta_factor,
                    _mixnum: Some(crate::parser::model_parser::MixtureClassGuard::enter(k + 1)),
                }
            }
        }
    }
}

/// `matched`, when `Some((fitted_etas, omega_inv, method))`, reassigns each
/// replicate's drawn etas to subjects by propensity-score matching against
/// `fitted_etas` (Mahalanobis matching under `omega_inv` via `method`; see
/// `crate::propensity_match`). `None` is the standard per-subject independent
/// draw and reproduces the previous behaviour byte-for-byte (same RNG draw
/// order).
#[allow(clippy::too_many_arguments)]
fn simulate_inner_with_draw<R: rand::Rng>(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    draw: usize,
    matched: Option<(&[DVector<f64>], &nalgebra::DMatrix<f64>, MatchMethod)>,
    horizon: Option<f64>,
    checked: Option<DrawIndependentChecked<'_>>,
    class_root: u64,
    rng: &mut R,
    warnings: &mut Vec<String>,
    flip_flop: &mut SimulatedFlipFlopTally,
    mut table: Option<&mut SimulatedParameters>,
) -> Result<Vec<SimulationResult>, EngineError> {
    use rand_distr::Normal;

    // Single chokepoint for every `simulate*` variant (both `simulate_inner` and
    // the propensity path funnel through here). Guard the modeled-`RATE` dose
    // precondition once per call, as `predict()` does — `simulate()` runs no
    // data-check otherwise. #324. The dose-compartment routing guard (#375) rides
    // the same chokepoint.
    //
    // Every precondition below is an `Err`, never a panic (#898), and every
    // `simulate*` entry point returns it as is. `checked` skips only the half that reads
    // no parameter value, and only when it was run on this very model and population
    // (#1288); every caller states whether it has one.
    check_simulate_preconditions(model, population, params, checked)?;
    // Nested levels (#1082 PR4): each record's level labels name its cell, which the κ
    // draw groups by level. `simulate()` takes the caller's population as given, so the
    // cells are assigned here, as `fitted_population` does for `fit()`; a population an
    // entry point already prepared (`with_derived_occasions`) is borrowed, not copied.
    let with_cells = crate::stats::nested_levels::with_level_cells(model, population);
    let population: &Population = &with_cells;
    // `simulate()` takes the caller's population as given, so a `correlation` line's
    // residual blocks are read off its `by` columns here, once per call rather than
    // once per replicate (#1081). A fitted population carries them already.
    let with_blocks = crate::stats::residual_error::with_residual_blocks(model, population);
    let population: &Population = &with_blocks;
    // CTMM (#759) has no simulation path yet: the Gaussian/PK emitter below would write
    // meaningless all-zero DV rows for a discrete-state endpoint (the generator is never
    // sampled). Fail loud rather than return garbage — fit() supports CTMM, simulate()
    // does not. Mirrors the survival guards above; the CLI `--simulate` path returns this
    // as a clean Err in `run_model_simulate`. This is the single simulate chokepoint.
    #[cfg(feature = "markov")]
    if model.has_ctmm() {
        return Err(super::entry_refusals::ctmm_unsupported(
            "simulate()/predict() does not support a [markov_model] (CTMM) endpoint yet — its \
             discrete-state trajectory has no simulation path, so the Gaussian emitter would \
             produce meaningless all-zero observations. CTMM simulation is a later slice.",
        ));
    }

    // ODE-accumulated TTE simulation has preventable preconditions (finite horizon,
    // no resets / left truncation). `simulate_with_options` checks them first and
    // returns a clean Err; `simulate` / `simulate_with_seed` reach them only here,
    // and return the same `Err` rather than emitting wrong rows.
    #[cfg(feature = "survival")]
    validate_tte_simulatable(model, population, TteHorizon::Given(horizon))?;

    // Same split for the model-vs-population checks (#1083). The
    // `Result`-returning entry points above have already run this list and
    // returned a clean `Err`; `simulate` / `simulate_with_seed` reach it only
    // here, and the failures this catches are silent by
    // construction — a weight that underflows to zero produces finite, plottable
    // rows with the variability quietly removed.
    //
    // Redundant on those `Result` paths, and `simulate_with_uncertainty` reaches
    // here once per draw, so the walk repeats. It is affordable because both
    // expensive halves are gated on the model actually declaring a weight
    // (`has_weighted_kappa` / `has_custom_ruv_magnitude`), which is false for
    // every model that does not use one; and where it is true, the pass is linear
    // in the observations the draw is about to simulate anyway.
    first_error(&check_simulation_data(model, population))?;

    let normal = Normal::new(0.0, 1.0).unwrap();
    let n_eta = model.n_eta;
    // η is drawn from the declared Ω, a zero η exactly 0 (#1605), not from `chol`; on a
    // `[mixture]` model from the Ω of the class each subject draws (#2163).
    let classes = SimClasses::new(model, params, class_root)?;

    // How each subject's κ is drawn, built once per subject rather than once per
    // replicate: one κ per occasion group, or per group of each level (#1082 PR4).
    let kappa_draws: Vec<crate::stats::nested_levels::KappaDraw> = if model.n_kappa > 0 {
        population
            .subjects
            .iter()
            .map(|s| crate::stats::nested_levels::KappaDraw::of(model, s))
            .collect::<Result<_, _>>()?
    } else {
        Vec::new()
    };

    let mut results = Vec::new();

    for sim_idx in 0..n_sim {
        let sim = sim_idx + 1;
        // What each subject of this replicate was drawn with, for the parameter table
        // (#2505). Nothing is kept when the caller asked for no table.
        let mut drawn: Vec<DrawnSubject> = Vec::new();
        match matched {
            Some((fitted, omega_inv, method)) => {
                // Draw a pool of one eta per subject for this replicate, then
                // reassign the draws to subjects by matching them to the fitted
                // (posthoc) etas. Each subject keeps its own observed design. One Ω for
                // the whole pool: `simulate_with_options_diag` refuses matching on a
                // `[mixture]` model (#2163), whose subjects draw their Ω with their class.
                let SimClasses::Single { eta_factor, .. } = &classes else {
                    unreachable!("propensity matching is refused on a [mixture] model")
                };
                let n = population.subjects.len();
                let pool: Vec<DVector<f64>> = (0..n)
                    .map(|_| {
                        let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(normal)).collect();
                        &**eta_factor * DVector::from_column_slice(&z)
                    })
                    .collect();
                let assign = crate::propensity_match::match_draws_to_fitted(
                    &pool, fitted, omega_inv, method,
                );
                for (i, subject) in population.subjects.iter().enumerate() {
                    let mut eta_slice: Vec<f64> = pool[assign[i]].iter().copied().collect();
                    eta_slice.resize(n_eta + model.n_kappa, 0.0);
                    // #1823: on a twin-less transit / IG closed form this η can sit past
                    // the tilting abscissa, where the closed form clamps the whole profile
                    // to 0 and there is no twin to reroute to. Record it here, on the
                    // η the emitter is about to predict with — both branches, so a
                    // propensity-matched run is counted on the η it was assigned, not the
                    // one it drew.
                    flip_flop.observe(model, subject, &params.theta, &eta_slice);
                    let first_row = results.len();
                    let predicted = emit_subject_rows(
                        model,
                        subject,
                        kappa_draws.get(i),
                        params,
                        &eta_slice,
                        draw,
                        sim,
                        normal,
                        horizon,
                        rng,
                        &mut results,
                        &mut *warnings,
                    );
                    if table.is_some() {
                        drawn.push(DrawnSubject::new(
                            &eta_slice[..n_eta],
                            predicted,
                            None,
                            &results[first_row..],
                        ));
                    }
                }
            }
            None => {
                for (i, subject) in population.subjects.iter().enumerate() {
                    // The subject's mixture class (#2163), held until its rows are out.
                    let class = classes.enter(subject, draw, sim);
                    let first_row = results.len();
                    // Sample eta from N(0, Omega); append zero kappas for IOV models.
                    let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(normal)).collect();
                    let z_vec = DVector::from_column_slice(&z);
                    let eta = class.eta_factor * z_vec;
                    let mut eta_slice: Vec<f64> = eta.iter().copied().collect();
                    eta_slice.resize(n_eta + model.n_kappa, 0.0);
                    flip_flop.observe(model, subject, &params.theta, &eta_slice);
                    let predicted = emit_subject_rows(
                        model,
                        subject,
                        kappa_draws.get(i),
                        class.params,
                        &eta_slice,
                        draw,
                        sim,
                        normal,
                        horizon,
                        rng,
                        &mut results,
                        &mut *warnings,
                    );
                    // Every row the subject emitted — Gaussian, TTE and categorical alike —
                    // reports the class it ran under (#2372).
                    let mixnum = class.mixnum();
                    for row in &mut results[first_row..] {
                        row.mixnum = mixnum;
                    }
                    if table.is_some() {
                        drawn.push(DrawnSubject::new(
                            &eta_slice[..n_eta],
                            predicted,
                            mixnum,
                            &results[first_row..],
                        ));
                    }
                }
            }
        }
        if let Some(table) = table.as_deref_mut() {
            append_drawn_parameters(model, population, &params.theta, draw, sim, &drawn, table);
        }
    }

    Ok(results)
}

/// One subject's draw in one `(draw, sim)` replicate, as the chokepoint simulated it — the
/// input of [`append_drawn_parameters`] (#2505).
struct DrawnSubject {
    /// The BSV η (`n_eta`), without the κ tail.
    eta: Vec<f64>,
    /// One κ vector per occasion group, in `predict_iov` order; empty without IOV.
    kappas: Vec<Vec<f64>>,
    /// The class the subject ran under, 1-based; `None` without a `[mixture]`.
    mixnum: Option<usize>,
    ipreds: Vec<f64>,
    /// The simulated value of each Gaussian observation row, in observation order: the
    /// `DV` a `[derived]` expression reads for this replicate.
    values: Vec<f64>,
}

impl DrawnSubject {
    /// `rows` are the rows the subject emitted, Gaussian ones first, one per IPRED.
    fn new(
        eta: &[f64],
        predicted: DrawnPrediction,
        mixnum: Option<usize>,
        rows: &[SimulationResult],
    ) -> Self {
        let values = rows[..predicted.ipreds.len()]
            .iter()
            .map(|r| r.outcome.continuous_value())
            .collect();
        Self {
            eta: eta.to_vec(),
            kappas: predicted.kappas,
            mixnum,
            ipreds: predicted.ipreds,
            values,
        }
    }

    /// The `[η, κ]` vector `pk_param_fn` reads at a record of occasion `occ`: this
    /// occasion's κ, or zeros for one with no group — the selection
    /// `compute_extra_output_columns` makes for the sdtab.
    fn eta_at(
        &self,
        model: &CompiledModel,
        layout: &crate::stats::stacked_layout::StackedLayout,
        occ: u32,
    ) -> Vec<f64> {
        let mut full = Vec::with_capacity(self.eta.len() + model.n_kappa);
        full.extend_from_slice(&self.eta);
        if model.n_kappa > 0 {
            match layout.group_of(occ).and_then(|k| self.kappas.get(k)) {
                Some(kappa) => full.extend_from_slice(kappa),
                None => full.extend(std::iter::repeat_n(0.0, model.n_kappa)),
            }
        }
        full
    }
}

/// The indices into `indiv_param_names` the table reports: every declared individual
/// parameter, without the parser's synthetic readout slots.
fn reported_individual_indices(model: &CompiledModel) -> Vec<usize> {
    (0..model.indiv_param_names.len())
        .filter(|&k| {
            !crate::parser::model_parser::is_synthetic_readout_param(&model.indiv_param_names[k])
        })
        .collect()
}

/// Append one [`SimulatedSubjectParameters`] row per subject of one replicate (#2505).
///
/// Every value is the subject's value at its **first observation row** — its covariates
/// and `TIME` there, its η, and the κ of that row's occasion — NONMEM's `FIRSTONLY`. A
/// subject with no Gaussian observation (a TTE- or binary-only one) reports its
/// individual parameters at its baseline covariates and `TIME = 0`, and no `[derived]`
/// value. A column whose value differs across one subject's observation rows is named in
/// [`SimulatedParameters::varying`].
fn append_drawn_parameters(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    draw: usize,
    sim: usize,
    drawn: &[DrawnSubject],
    table: &mut SimulatedParameters,
) {
    let reported = reported_individual_indices(model);
    let derived = drawn_derived_columns(model, population, theta, drawn);
    if table.derived_names.is_empty() {
        if let Some(columns) = derived.iter().find(|c| !c.is_empty()) {
            table.derived_names = columns.iter().map(|(n, _)| n.clone()).collect();
        }
    }
    for (i, (subject, d)) in population.subjects.iter().zip(drawn).enumerate() {
        let layout = crate::stats::stacked_layout::StackedLayout::of(model, subject);
        let n_obs = subject.obs_times.len();
        let values_at = |j: usize| -> Vec<f64> {
            let occ = subject.occasions.get(j).copied().unwrap_or(0);
            model.indiv_param_values(
                theta,
                &d.eta_at(model, &layout, occ),
                subject.obs_cov(j),
                subject.param_time_obs(j),
                d.mixnum,
            )
        };
        let all = if n_obs == 0 {
            model.indiv_param_values(
                theta,
                &d.eta_at(model, &layout, 0),
                &subject.covariates,
                0.0,
                d.mixnum,
            )
        } else {
            let first = values_at(0);
            for j in 1..n_obs {
                let at_j = values_at(j);
                for &k in &reported {
                    if !same_value(first[k], at_j[k]) {
                        table.note_varying(&model.indiv_param_names[k]);
                    }
                }
            }
            first
        };
        let mut derived_values = vec![f64::NAN; table.derived_names.len()];
        if let Some(columns) = derived.get(i) {
            for (name, values) in columns {
                if let Some(col) = values.first() {
                    if let Some(slot) = table.derived_names.iter().position(|n| n == name) {
                        derived_values[slot] = *col;
                    }
                    if values.iter().any(|v| !same_value(*v, *col)) {
                        table.note_varying(name);
                    }
                }
            }
        }
        table.rows.push(SimulatedSubjectParameters {
            draw,
            sim,
            id: subject.id.clone(),
            mixnum: d.mixnum,
            eta: d.eta.clone(),
            kappas: d.kappas.clone(),
            individual: reported.iter().map(|&k| all[k]).collect(),
            derived: derived_values,
        });
    }
}

/// Equal, or both `NaN`: a column that is `NaN` on every row does not vary.
fn same_value(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// Each subject's `[derived]` / `[output]` columns for one replicate (#2505), from the
/// same `compute_extra_output_columns` the fit's sdtab is built by, with the replicate's
/// draws in place of the EBEs: its η and κ, the class it ran under, its IPRED, and its
/// simulated values as `DV`. Empty when the model declares neither block.
fn drawn_derived_columns(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    drawn: &[DrawnSubject],
) -> Vec<Vec<(String, Vec<f64>)>> {
    if model.derived_exprs.is_empty() && model.output_columns.is_empty() {
        return Vec::new();
    }
    let mut simulated = population.clone();
    for (subject, d) in simulated.subjects.iter_mut().zip(drawn) {
        for (dv, value) in subject.observations.iter_mut().zip(&d.values) {
            *dv = *value;
        }
    }
    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    let mut subjects: Vec<SubjectResult> = simulated
        .subjects
        .iter()
        .zip(drawn)
        .map(|(subject, d)| {
            let _mix_guard = d
                .mixnum
                .map(crate::parser::model_parser::MixtureClassGuard::enter);
            let n_obs = d.ipreds.len();
            // States are not reconstructed on the IOV path, as for a fitted IOV subject.
            let compartment_states = if model.n_kappa == 0 {
                pk::compute_predictions_with_states(model, subject, theta, &d.eta).1
            } else {
                Vec::new()
            };
            SubjectResult {
                id: subject.id.clone(),
                eta: DVector::from_column_slice(&d.eta),
                ipred: d.ipreds.clone(),
                pred: pk::compute_predictions_with_tv(model, subject, theta, &zero_eta),
                iwres: vec![f64::NAN; n_obs],
                cwres: vec![f64::NAN; n_obs],
                npde: Vec::new(),
                npd: Vec::new(),
                ofv_contribution: f64::NAN,
                cens: subject.cens.clone(),
                n_obs,
                pmix: None,
                mixest: None,
                extra_columns: Vec::new(),
                per_obs_tad: Vec::new(),
                compartment_states,
                #[cfg(feature = "survival")]
                discrete_rows: Vec::new(),
            }
        })
        .collect();
    let kappas: Vec<Vec<DVector<f64>>> = drawn
        .iter()
        .map(|d| {
            d.kappas
                .iter()
                .map(|k| DVector::from_column_slice(k))
                .collect()
        })
        .collect();
    let mixest: Option<Vec<usize>> = model.mixture.is_some().then(|| {
        drawn
            .iter()
            .map(|d| d.mixnum.map_or(0, |k| k - 1))
            .collect()
    });
    super::compute_extra_output_columns(
        model,
        &simulated,
        theta,
        &kappas,
        &mut subjects,
        mixest.as_deref(),
    );
    subjects.into_iter().map(|sr| sr.extra_columns).collect()
}

/// Options controlling `simulate_with_uncertainty()`.
///
/// Derives `Default` so an out-of-crate caller (the R wrapper) can build one
/// with `..Default::default()` and stay source-compatible when a field is
/// added here (#529). Note that the two counts default to `0`, which draws
/// nothing and returns an empty row set — a caller spreading the default is
/// expected to set both explicitly.
#[derive(Debug, Clone, Default)]
pub struct SimulateUncertaintyOptions {
    /// Number of parameter sets to draw from the uncertainty distribution.
    /// Defaults to `0`; set it explicitly when spreading `Default::default()`.
    pub n_uncertainty_draws: usize,
    /// Number of eta/eps replicates simulated *per* parameter draw.
    /// Defaults to `0`; set it explicitly when spreading `Default::default()`.
    pub n_sim_per_draw: usize,
    /// How to draw the parameter sets — asymptotic MVN or SIR resamples.
    pub method: crate::estimation::uncertainty_samples::UncertaintyMethod,
    /// Optional seed for reproducibility. `None` draws from entropy.
    pub seed: Option<u64>,
}

/// Simulate observations while propagating parameter uncertainty.
///
/// For each of `opts.n_uncertainty_draws` parameter sets drawn from the
/// uncertainty distribution (asymptotic MVN around the ML estimate or stored
/// SIR resamples), simulate `opts.n_sim_per_draw` replicates of every subject
/// — sampling etas from the drawn Omega and epsilons from the drawn Sigma.
///
/// Total rows returned: `n_uncertainty_draws * n_sim_per_draw * n_subjects *
/// n_obs`. Each `SimulationResult` carries the originating `draw` and `sim`
/// indices so downstream code can compute per-time uncertainty bands.
///
/// For an IOV model, a fit under a derived `iov_occasion = dose | time(...)` rule
/// ([`FitResult::iov_occasion`]) has the population's occasion labels re-derived with that
/// rule first, as `fit()` derived them, so κ is drawn per occasion (#1955).
///
/// Thin wrapper over [`simulate_with_uncertainty_diag`] that discards its
/// [`SimulationOutput::warnings`]; the rows are the same, bit for bit.
pub fn simulate_with_uncertainty(
    model: &CompiledModel,
    population: &Population,
    fit_result: &FitResult,
    opts: &SimulateUncertaintyOptions,
) -> Result<Vec<SimulationResult>, EngineError> {
    simulate_with_uncertainty_impl(model, population, fit_result, opts, false).map(|o| o.results)
}

/// [`simulate_with_uncertainty`] with the diagnostics attached (#1645) — the uncertainty twin of
/// [`simulate_with_options_diag`].
///
/// `results` is exactly what [`simulate_with_uncertainty`] returns. `warnings` carries, in this
/// order: the per-subject simulation diagnostics (a degenerate hazard draw, #763 / #762), one
/// `W_UNCERTAINTY_DRAWS_SKIPPED` warning naming every draw skipped because it landed in the
/// flip-flop regime of a twin-less transit / IG closed form (#786, #1485), one
/// `W_SIMULATED_FLIP_FLOP` count of the eligible individuals *inside* the draws that ran
/// whose own sampled η landed there (#1823 — the skip is decided at each draw's η = 0 θ, so
/// both can fire on one run), capped steady-state
/// equilibrations (#867), `W_IOV_NO_OCCASIONS` for an IOV model handed subjects with no
/// occasion labels after the fit's [`FitResult::iov_occasion`] is applied (#1955), and the
/// shared non-fit diagnostics bundle — parse warnings, the
/// data-reader warnings through the same suppression filter `fit()` uses, the model/data checks
/// at the fit's point estimate, experimental-feature notices, and the ODE-solver diagnostics
/// collected over every draw. `postfit::non_fit_diagnostics` documents what that bundle
/// includes and leaves out.
///
/// Refuses, with the `E_TRANSIT_FLIP_FLOP` / `E_IG_FLIP_FLOP` code `simulate()` uses, when the
/// fit's point estimate is itself in that flip-flop regime, or when every one of `n > 0` draws
/// is skipped.
///
/// A caller relaying data-reader findings should take them from here, never from
/// [`Population::warnings`] directly: the raw list still holds the findings the shared filter
/// withholds (`W_NO_DOSES` / `W_CMT_DEFAULTED` on a compartment-free model).
///
/// [`SimulationOutput::parameters`] carries one row per `(draw, sim, id)` of the η, κ,
/// individual parameters and `[derived]` / `[output]` values each subject was simulated
/// with, under its draw's θ (#2505). A skipped draw has no rows.
pub fn simulate_with_uncertainty_diag(
    model: &CompiledModel,
    population: &Population,
    fit_result: &FitResult,
    opts: &SimulateUncertaintyOptions,
) -> Result<SimulationOutput, EngineError> {
    simulate_with_uncertainty_impl(model, population, fit_result, opts, true)
}

/// [`simulate_with_uncertainty_diag`], building the drawn-parameter table only when
/// `with_parameters` (#2505).
fn simulate_with_uncertainty_impl(
    model: &CompiledModel,
    population: &Population,
    fit_result: &FitResult,
    opts: &SimulateUncertaintyOptions,
    with_parameters: bool,
) -> Result<SimulationOutput, EngineError> {
    use rand::SeedableRng;

    // Start the SS-equilibration non-convergence sink clean, as `simulate_with_options_diag`
    // does, so this run's `warnings` carry only its own capped equilibrations (#867).
    crate::dosing::clear_ss_nonconvergence_warnings();

    // The occasions the fit derived (#1955): `FitResult::iov_occasion` is the rule it ran
    // under (#1783), applied before anything reads the population.
    let derived = with_derived_occasions(model, population, fit_result.iov_occasion.as_ref());
    let population: &Population = &derived;
    let iov_warning = no_occasion_warning(
        fit_result.omega_iov.is_some(),
        population,
        "no kappa was drawn for them: they were simulated",
        "Label the occasions in the data (`iov_column` in [fit_options]), or derive them with \
         `iov_occasion = dose | time(...)`; `simulate_with_uncertainty` applies the rule the \
         fit ran under, `FitResult::iov_occasion`.",
    );

    // The fit's θ against this model's layout (#1614), before anything is drawn around
    // it: the per-draw chokepoint runs the same gate, but a mismatched point estimate
    // would first be handed to the draw machinery, and a zero-draw run never reaches it.
    check_theta_length(model, &fit_result.theta)?;
    // Ω / σ / Ω_IOV (#1764, #1833) are refused by `fitted_params_from_result` below,
    // before any draw: a second gate here rejected exactly the same inputs.

    // A repeated-event or ODE-accumulated TTE simulation needs a finite horizon, which this
    // uncertainty path does not expose — validate once here rather than per draw, and say
    // so rather than ask for an option the caller cannot set (#1774).
    #[cfg(feature = "survival")]
    validate_tte_simulatable(model, population, TteHorizon::NotAccepted)?;

    // Parity with `fit()`: reject a referenced covariate absent from the data
    // rather than silently reading it as 0.0 (a `Selected` error-model selector
    // would otherwise route every row to branch 0). See #658 — and, for a weighted
    // κ or a weighted residual, a non-positive weight before it collapses to zero
    // (#1031 / #1029 / #1083).
    first_error(&check_simulation_data(model, population))?;

    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        // Re-seed StdRng from entropy so simulate-without-seed is still
        // independent across calls but uses a uniform RNG type internally.
        None => rand::make_rng(),
    };

    let template =
        crate::estimation::uncertainty_samples::fitted_params_from_result(fit_result, model)?;
    // The point estimate itself must be simulatable (#1485): `simulate()` refuses a θ in the
    // flip-flop regime of a twin-less transit / IG closed form, and drawing around it would
    // otherwise skip most draws and return the rest as if the point estimate were sound.
    first_error(&check_absorption_flip_flop_no_twin_diags(
        model,
        population,
        &template.theta,
    ))?;
    // The sampler's input refusals carry their codes (#2044).
    let draws = crate::estimation::uncertainty_samples::draw_parameter_samples_coded(
        fit_result,
        &template,
        opts.n_uncertainty_draws,
        opts.method,
        &mut rng,
    )?;

    // Final size is deterministic, so we can size the buffer once and avoid
    // repeated reallocations for large simulations.
    let total_obs: usize = population.subjects.iter().map(|s| s.obs_times.len()).sum();
    let mut results =
        Vec::with_capacity(opts.n_uncertainty_draws * opts.n_sim_per_draw * total_obs);
    // Per-draw and per-subject simulation diagnostics (#786, #762/#763), returned in
    // `warnings`. The per-subject handling — no whole-run panic, degenerate subjects
    // censored — applies either way.
    let mut sim_warnings: Vec<String> = Vec::new();
    let mut table = SimulatedParameters::for_model(model);
    // One solver-stats scope around every draw: the loop is serial (the RNG draw order is
    // part of the contract), so the thread-local scope sees every integration. #1304.
    let (looped, stats) = super::with_solver_stats(model, || -> Result<(), EngineError> {
        simulate_uncertainty_draws(
            model,
            population,
            &draws,
            opts,
            run_root(opts.seed),
            &mut rng,
            &mut results,
            &mut sim_warnings,
            with_parameters.then_some(&mut table),
        )
    });
    looped?;
    sim_warnings.extend(crate::dosing::take_ss_nonconvergence_warnings());
    sim_warnings.extend(iov_warning);
    sim_warnings.extend(frem_drawn_eta_warning(
        model,
        population.subjects.len(),
        &population.subjects,
        "`simulate_with_uncertainty` has no matching option: to keep each subject's PK with \
         its own covariates, use `simulate_with_options` with `SimulateOptions::match_method`, \
         or `simulate_adaptive_with_inputs` with the FREM fit's EBEs as `SuppliedEtas`.",
    ));
    sim_warnings.extend(super::non_fit_diagnostics(
        model,
        population,
        &template,
        &stats,
        super::SolverStatsPhase::SimulateUncertainty,
    ));
    Ok(SimulationOutput {
        results,
        warnings: sim_warnings,
        parameters: table,
    })
}

/// The per-draw loop of [`simulate_with_uncertainty_diag`], appending each draw's rows to
/// `results`. Split out so the caller can run it inside one solver-stats scope.
fn simulate_uncertainty_draws(
    model: &CompiledModel,
    population: &Population,
    draws: &[ModelParameters],
    opts: &SimulateUncertaintyOptions,
    class_root: u64,
    rng: &mut rand::rngs::StdRng,
    results: &mut Vec<SimulationResult>,
    sim_warnings: &mut Vec<String>,
    mut table: Option<&mut SimulatedParameters>,
) -> Result<(), EngineError> {
    // 1-based indices of the skipped draws, and the first one's reason (#1485).
    let mut skipped: Vec<usize> = Vec::new();
    let mut first_reason: Option<String> = None;
    // One #1823 tally across every draw that *did* run: a skipped draw simulates no
    // individual, so it contributes to neither side of the ratio. The two counts answer
    // different questions — a skipped draw is a θ the closed form cannot serve at all, a
    // tallied individual is an η within an otherwise serviceable draw — so both warnings
    // can fire on one run.
    let mut flip_flop = SimulatedFlipFlopTally::default();
    // The draw-independent half of the chokepoint's list, run once at the first draw that is
    // simulated (#1288). Not "at draw 1": the skip below can pass over draw 1, and a check
    // keyed to it would then never run. Lazily, so a run whose every draw is skipped is
    // refused for that, as before, and a zero-draw run still returns no rows.
    let mut checked: Option<DrawIndependentChecked<'_>> = None;
    for (k, params) in draws.iter().enumerate() {
        // A parameter draw can land in the flip-flop regime even when the point
        // estimate is in-domain. For a twin-less transit/IG closed form,
        // `simulate_inner_with_draw`'s own flip-flop check would then return `Err` —
        // and the `?` below would fail the *entire* uncertainty run on one draw. Skip
        // such a draw instead, so the remaining draws still yield results (#786), and
        // report every skip in one warning after the loop (#1485). **This check must
        // stay ahead of that `?`** (#898). The single-shot `predict()`/`simulate()`
        // paths keep the failure (there is no other draw to fall back on).
        // Twin-carrying models return `None` here and proceed (they reroute per-eval),
        // so this only skips genuinely un-simulatable draws.
        if let Some(msg) = check_absorption_flip_flop_no_twin(model, population, &params.theta) {
            skipped.push(k + 1);
            first_reason.get_or_insert(msg);
            continue;
        }
        let proof = match checked {
            Some(c) => c,
            None => *checked.insert(DrawIndependentChecked::run(model, population)?),
        };
        let mut rows = simulate_inner_with_draw(
            model,
            population,
            params,
            opts.n_sim_per_draw,
            k + 1,
            None,
            None,
            Some(proof),
            class_root,
            rng,
            sim_warnings,
            &mut flip_flop,
            table.as_deref_mut(),
        )?;
        results.append(&mut rows);
    }
    if let Some(reason) = first_reason {
        if skipped.len() == draws.len() {
            first_error(&all_uncertainty_draws_skipped_diags(
                model,
                draws.len(),
                &reason,
            ))?;
        }
        sim_warnings.push(uncertainty_draws_skipped_warning(
            model,
            &skipped,
            draws.len(),
        ));
    }
    // After the skip warning, so the coarser finding (a θ the closed form cannot serve at
    // all) is read first. Not before the `?` above: a run whose every draw was skipped is
    // refused, and it simulated no individual to tally anyway.
    sim_warnings.extend(flip_flop.warning(model));
    Ok(())
}

/// A single simulated observation.
///
/// `draw` is the uncertainty draw index (1-based). For `simulate()` /
/// `simulate_with_seed()`, which use point-estimate parameters, `draw` is
/// always `1`. For `simulate_with_uncertainty()` it spans
/// `1..=n_uncertainty_draws`. `sim` is the replicate index *within* a draw.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub draw: usize,
    pub sim: usize,
    pub id: String,
    /// For Gaussian rows: the scheduled observation time from the subject's grid.
    /// For TTE rows: the sampled event time (equals `SimOutcome::Event::time`; the
    /// outer field exists for uniform iteration without matching on `outcome`).
    pub time: f64,
    /// CMT column value for this observation row. For Gaussian subjects this mirrors the data
    /// file's CMT (e.g. 1 for a central-compartment PK endpoint — not necessarily 0). For TTE
    /// rows (requires `survival` feature) it matches the `[event_model] cmt` declaration.
    pub cmt: usize,
    /// Individual prediction at η (Gaussian path only; NAN for non-Gaussian).
    pub ipred: f64,
    /// Simulated observation outcome.  For Gaussian: `SimOutcome::Continuous { value }`.
    /// For TTE (requires `survival` feature): `SimOutcome::Event { time, observed }`.
    pub outcome: SimOutcome,
    /// The mixture class (`MIXNUM`, 1-based) this row's subject was simulated under in its
    /// `(draw, sim)` replicate (#2372): `Some(k)` on a `[mixture]` model, `None` otherwise.
    /// NONMEM's `$TABLE MIXNUM` under `$SIMULATION`. Every row of one subject in one
    /// replicate carries the same class; the class is redrawn per replicate (#2163).
    pub mixnum: Option<usize>,
}

/// One simulated subject's drawn parameters in one `(draw, sim)` replicate (#2505) — a row
/// of [`SimulatedParameters`], keyed like the [`SimulationResult`] rows it produced.
///
/// Each value is the subject's value at its first observation row (NONMEM's `$TABLE …
/// FIRSTONLY`); [`SimulatedParameters::varying`] names any column that changes across a
/// subject's rows. The vectors are parallel to the table's name lists.
#[derive(Debug, Clone)]
pub struct SimulatedSubjectParameters {
    pub draw: usize,
    pub sim: usize,
    pub id: String,
    /// The mixture class (`MIXNUM`, 1-based) the subject ran under; `None` without a
    /// `[mixture]`, as [`SimulationResult::mixnum`].
    pub mixnum: Option<usize>,
    /// The drawn between-subject η, parallel to [`SimulatedParameters::eta_names`].
    pub eta: Vec<f64>,
    /// The drawn inter-occasion κ, one vector per occasion group of the subject (in the
    /// order `predict` indexes them); empty for a model without IOV.
    pub kappas: Vec<Vec<f64>>,
    /// The `[individual_parameters]` values, parallel to
    /// [`SimulatedParameters::individual_names`].
    pub individual: Vec<f64>,
    /// The `[derived]` / `[output]` values, parallel to
    /// [`SimulatedParameters::derived_names`]; `NaN` for a subject with no observation row.
    pub derived: Vec<f64>,
}

/// The parameters each simulated subject was drawn with (#2505): one
/// [`SimulatedSubjectParameters`] row per `(draw, sim, id)`, with the column names held
/// once. [`SimulationOutput::parameters`].
///
/// A row reports every value at the subject's **first observation row** — its covariates,
/// `TIME`, and the κ of that row's occasion — the `FIRSTONLY` convention of a NONMEM
/// simulation table. Under a time-varying covariate (or a `TIME`-dependent parameter,
/// or IOV) a value can differ across the subject's rows; every such column is named in
/// `varying`, so a reader knows which of the reported values are a first-row snapshot
/// rather than the subject's value at every row. A subject with no Gaussian observation
/// reports its individual parameters at its baseline covariates and `TIME = 0`.
///
/// `[derived]` / `[output]` values are the columns the fit's sdtab carries for the same
/// model, computed by the same code with the simulated η, κ, class, IPRED and `DV`.
#[derive(Debug, Clone, Default)]
pub struct SimulatedParameters {
    /// The model's η names.
    pub eta_names: Vec<String>,
    /// The model's `[individual_parameters]`, in declaration order.
    pub individual_names: Vec<String>,
    /// The `[derived]` / `[output]` columns; empty when the model declares neither.
    pub derived_names: Vec<String>,
    /// The columns (from `individual_names` / `derived_names`) whose value differed across
    /// the observation rows of at least one simulated subject.
    pub varying: Vec<String>,
    pub rows: Vec<SimulatedSubjectParameters>,
}

impl SimulatedParameters {
    /// An empty table with `model`'s η and individual-parameter names.
    fn for_model(model: &CompiledModel) -> Self {
        Self {
            eta_names: model.eta_names.clone(),
            individual_names: reported_individual_indices(model)
                .into_iter()
                .map(|k| model.indiv_param_names[k].clone())
                .collect(),
            ..Self::default()
        }
    }

    fn note_varying(&mut self, name: &str) {
        if !self.varying.iter().any(|n| n == name) {
            self.varying.push(name.to_string());
        }
    }
}

/// The result of [`simulate_with_options_diag`] and [`simulate_with_uncertainty_diag`]: the
/// simulated rows plus any non-fatal
/// per-subject diagnostics collected during the run.
///
/// `warnings` is the simulation analogue of [`FitResult::warnings`]: a subject whose
/// draw degenerates — a non-positive / non-finite analytic hazard rate that produces
/// no event (#763), or a hazard so extreme its recurrent stream is skipped rather than
/// materialised (#762) — is handled *per subject* (censored / skipped, the run
/// continues) and named here, instead of silently vanishing into the censored rows or
/// aborting the whole run. Since #1280 / #1304 it also carries the shared non-fit diagnostics
/// bundle: parse warnings, data-reader warnings, the model/data checks (`W_STEADY_STATE_*`,
/// `W_SDE_*`, `W_NEGATIVE_LAGTIME`, …), experimental-feature notices, and this pass's
/// ODE-solver diagnostics (`W_ODE_SOLVER_DIAGNOSTICS`). Empty for a clean simulation of a
/// well-formed model. The simpler `simulate()` /
/// `simulate_with_seed()` entry points apply the same per-subject handling but return
/// only the rows (no diagnostics channel) — use `simulate_with_options` when the
/// warnings matter (e.g. a population VPC).
///
/// `simulate_with_uncertainty()` returns a flat row vec and so has no channel either; since
/// #1645 its diagnostics — including a skipped flip-flop draw (#786) — come back on
/// [`simulate_with_uncertainty_diag`]'s `warnings`.
///
/// Since #1823 both `_diag` forms also carry one `W_SIMULATED_FLIP_FLOP` count of the
/// simulated individuals whose *sampled* η entered a twin-less transit / IG closed form's
/// flip-flop regime, where it clamps: none of that individual's rows carries any
/// contribution from the doses its subject was given. (*Undosed*, not drug-free — an
/// `[initial_conditions]` baseline is itself drug and survives the clamp.) Those rows are
/// returned either way — the count is the only thing that says they are degenerate, which
/// is why the row-only entry points cannot report it.
///
/// Its denominator is the individuals whose subject carries a dose of positive amount
/// **strictly before one of its observations**, not every dosed individual: without that,
/// no returned row could have carried the dose whether the closed form clamped or not, so
/// an ODE rewrite would produce the same rows. The message states that eligibility.
/// **Searching the rows for `ipred == 0.0` is not an alternative**: four things run on the
/// closed form's output after the clamp (an `[initial_conditions]` baseline, a Form-C
/// `[scaling] y = <expr>` readout, the LTBS log transform, the FREM row override), and
/// whether any of them leaves a clamped row at zero depends on the model's own
/// expressions — measured on #1823 review r2, `y = central/V` and `y = central/V + 2.5`
/// give 27 zero rows and 0 zero rows for the same 27 clamped individuals.
/// `every_diagnostic_carrying_entry_point_reports_the_same_bundle` /
/// `the_row_only_entry_points_stay_silent_by_contract` pin the two lists so a new entry point
/// cannot quietly join the silent one.
#[derive(Debug, Clone, Default)]
pub struct SimulationOutput {
    pub results: Vec<SimulationResult>,
    pub warnings: Vec<String>,
    /// The parameters each simulated subject was drawn with — η, κ, individual parameters
    /// and `[derived]` / `[output]` values — one row per `(draw, sim, id)` (#2505).
    pub parameters: SimulatedParameters,
}
