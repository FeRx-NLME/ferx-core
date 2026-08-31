#![allow(unused_imports)]
//! Extracted verbatim from `api/mod.rs` (production peel). See the module-
//! doc / Key Modules table for the split rationale.
use super::*;
use crate::diagnostics::{first_error, CheckReport, Diagnostic};
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
use crate::stats::likelihood::{
    build_frem_r_override, compute_cwres, foce_subject_nll, foce_subject_nll_iov,
};
use crate::stats::residual_error::{
    compute_iwres_with_correlations, compute_r_matrix_with_correlations,
    compute_r_matrix_with_correlations_scaled, iwres_autocorrelation,
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
/// callers that obtained `population` via `read_nonmem_csv` should inspect
/// `population.warnings` before calling this function.
pub fn simulate(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
) -> Vec<SimulationResult> {
    let mut rng = rand::rng();
    simulate_inner(model, population, params, n_sim, &mut rng)
}

/// Simulate with a fixed seed for reproducibility.
pub fn simulate_with_seed(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    seed: u64,
) -> Vec<SimulationResult> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    simulate_inner(model, population, params, n_sim, &mut rng)
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
}

/// Validate the preventable preconditions of TTE event-time simulation, returning a
/// clear `Err` for a caller to act on.
///
/// **Repeated events (RTTE, `type = rtte`, Slice 3.3)** are simulated as a recurrent
/// stream by `simulate_rtte_stream` (analytic hazards only). Several preconditions are
/// *user-fixable* and would otherwise corrupt the stream silently, so they are rejected
/// here (a clean `Err` from `simulate_with_options`; the `Vec`-returning entry points
/// re-raise the same message as a panic):
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
#[cfg(feature = "survival")]
pub(crate) fn validate_tte_simulatable(
    model: &CompiledModel,
    population: &Population,
    horizon: Option<f64>,
) -> Result<(), String> {
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
                return Err(
                    "Repeated time-to-event (RTTE, `type = rtte`) simulation requires a finite, \
                     positive administrative horizon: set `[simulation] horizon` (or \
                     `SimulateOptions.horizon`). The recurrent event stream is generated up to \
                     that horizon."
                        .to_string(),
                );
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
                return Err(format!(
                    "RTTE simulation supports one recurrent stream per subject, but subject '{}' \
                     has RTTE endpoints on multiple CMTs ({:?}); multi-cause RTTE simulation is a \
                     later slice",
                    subject.id, rtte_cmts
                ));
            }
            if has_single_tte_sibling {
                return Err(format!(
                    "RTTE simulation does not support an RTTE cause combined with a competing \
                     single-event TTE cause on subject '{}'; simulate them separately",
                    subject.id
                ));
            }
            if !subject.reset_times.is_empty() {
                return Err(format!(
                    "RTTE simulation does not support EVID=3/4 resets (a reset would disturb the \
                     hazard clock mid-stream; selective per-state reset is a later slice); subject \
                     '{}' has resets",
                    subject.id
                ));
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
            return Err(
                "ODE-accumulated TTE (joint PK-TTE) simulation requires a finite, positive \
                 administrative horizon: set `[simulation] horizon` (or `SimulateOptions.horizon`). \
                 A drug-driven hazard can vanish, so there is no implicit observation window."
                    .to_string(),
            );
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
            return Err(format!(
                "ODE-accumulated TTE simulation does not support EVID=3/4 resets (a full reset \
                 zeros the cumulative hazard mid-flight; selective per-state reset is a later \
                 slice); subject '{}' has resets",
                subject.id
            ));
        }
        if let Some(ObsRecord::Event { entry_time, .. }) = subject
            .obs_records
            .iter()
            .find(|r| matches!(r, ObsRecord::Event { cmt, entry_time, .. } if is_ode_tte(cmt) && *entry_time > 0.0))
        {
            return Err(format!(
                "ODE-accumulated TTE simulation does not support left truncation \
                 (entry_time={entry_time} for subject '{}'); conditional sampling past entry is \
                 a deferred follow-up",
                subject.id
            ));
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
) -> Result<(), String> {
    if model.n_kappa > 0 && params.omega_iov.is_none() {
        return Err(format!(
            "model declares {} kappa (IOV) but the supplied parameters carry no omega_iov; \
             simulation draws one kappa vector per occasion from it. Rebuild the parameters \
             with the fitted IOV covariance (see `fitted_params_from_result`) before simulating",
            model.n_kappa
        ));
    }
    Ok(())
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
) -> Result<Vec<SimulationResult>, String> {
    simulate_with_options_diag(model, population, params, n_sim, opts).map(|o| o.results)
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
/// rows and any non-fatal per-subject warnings (a degenerate hazard draw, #763; a
/// degenerate recurrent stream skipped, #762). [`simulate_with_options`] is the thin
/// wrapper that returns only the rows.
pub fn simulate_with_options_diag(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    opts: &SimulateOptions,
) -> Result<SimulationOutput, String> {
    use rand::SeedableRng;

    // Start the SS-equilibration non-convergence sink clean, then drain it into this run's
    // `warnings` at each return (#867): a capped nonlinear pulse-train can silently under-report
    // the SS trough, biasing simulated concentrations low.
    crate::dosing::clear_ss_nonconvergence_warnings();

    // ODE-accumulated (joint PK-TTE) TTE simulation samples drug-driven event times
    // via the augmented-ODE root-finder (Slice 2.2). Validate its preventable
    // preconditions up front so a caller gets a clean Err here rather than a panic
    // deep in sampling (a finite horizon is required; resets and left truncation are
    // not yet supported for an ODE hazard).
    #[cfg(feature = "survival")]
    validate_tte_simulatable(model, population, opts.horizon)?;

    // An IOV model needs the fitted Ω_IOV in `params` (#1019). Report a missing one
    // as a clean Err here; the Vec-returning entry points enforce the same contract
    // as a panic at the chokepoint below, since they cannot signal.
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
            return Err(format!(
                "SimulateOptions.horizon must be finite and > 0 (got {h})"
            ));
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
                    return Err(format!(
                        "SimulateOptions.horizon ({h}) is below subject '{}' entry_time \
                         ({entry_time}); the administrative horizon must be ≥ every \
                         subject's entry time",
                        subject.id
                    ));
                }
            }
        }
    }

    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::make_rng(),
    };

    // Guard the modeled-`RATE` dose precondition up front (#324). The
    // non-propensity branch reaches it via the `simulate_inner_with_draw`
    // chokepoint, but the propensity branch first runs a full inner EBE pass
    // (`run_inner_loop_warm` below) that integrates every subject — on an
    // unsupported config that would hit the per-path tripwire (silently in
    // release) or `resolve_rate`'s opaque `.expect` *before* the chokepoint
    // guard. Asserting here makes both branches fail with the same actionable
    // diagnostic; it is a no-op O(doses) scan on the common all-`Fixed` dataset.
    // The built-in absorption input-rate guard (#588) is hoisted for the same
    // reason: otherwise a malformed multi-pathway / SS / infusion absorption model
    // integrates the whole warm-EBE pass first and fails only at the chokepoint,
    // with a confusable "EBE did not converge" instead of the real cause.
    assert_modeled_doses_supported(model, population);
    assert_dose_compartments_supported(model, population);
    assert_absorption_closed_form_support(model, population);
    assert_absorption_flip_flop_no_twin(model, population, &params.theta);
    // A time-varying covariate on a survival hazard would be silently frozen — panic
    // rather than return a subtly wrong prediction / simulation (#741; fit() Err's).
    #[cfg(feature = "survival")]
    assert_survival_tv_covariates(model, population);
    assert_absorption_dosing_supported(model, population);

    let method = match opts.match_method {
        Some(m) => m,
        None => {
            let mut warnings = Vec::new();
            let results = simulate_inner_with_draw(
                model,
                population,
                params,
                n_sim,
                1,
                None,
                opts.horizon,
                &mut rng,
                &mut warnings,
            );
            warnings.extend(crate::dosing::take_ss_nonconvergence_warnings());
            return Ok(SimulationOutput { results, warnings });
        }
    };

    if population.subjects.is_empty() {
        return Err(
            "propensity-score matching requires a non-empty observed population".to_string(),
        );
    }
    if let Some(s) = population
        .subjects
        .iter()
        .find(|s| s.observations.is_empty())
    {
        return Err(format!(
            "propensity-score matching requires observations for every subject \
             (to compute posthoc etas); subject '{}' has none",
            s.id
        ));
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
        return Err(format!(
            "propensity-score matching requires finite observations for every subject \
             (to compute posthoc etas); subject '{}' has non-finite DV values. A `DV = .` \
             design template carries NaN placeholders — match against the observed dataset \
             instead",
            s.id
        ));
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
        return Err(format!(
            "propensity-score matching: the posthoc eta for subject '{}' is \
             non-finite (its EBE did not converge); cannot match",
            population.subjects[i].id
        ));
    }

    let omega_inv = &params.omega.inv;
    let mut warnings = Vec::new();
    let results = simulate_inner_with_draw(
        model,
        population,
        params,
        n_sim,
        1,
        Some((&eta_hats, omega_inv, method)),
        opts.horizon,
        &mut rng,
        &mut warnings,
    );
    warnings.extend(crate::dosing::take_ss_nonconvergence_warnings());
    Ok(SimulationOutput { results, warnings })
}

fn simulate_inner<R: rand::Rng>(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    rng: &mut R,
) -> Vec<SimulationResult> {
    // `simulate` / `simulate_with_seed` carry no horizon; the per-record window
    // applies. An explicit `[simulation] horizon` enters via `simulate_with_options`.
    // These entry points return only the rows; per-subject simulation diagnostics
    // (#762 degenerate RTTE, #763 degenerate hazard draw) are collected but surfaced
    // only on the `simulate_with_options` path (`SimulationOutput.warnings`).
    let mut warnings = Vec::new();
    simulate_inner_with_draw(
        model,
        population,
        params,
        n_sim,
        1,
        None,
        None,
        rng,
        &mut warnings,
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
    subject
        .obs_raw_times
        .get(j)
        .copied()
        .unwrap_or(subject.obs_times[j])
}

fn emit_subject_rows<R: rand::Rng>(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta_slice: &[f64],
    draw: usize,
    sim: usize,
    normal: rand_distr::Normal<f64>,
    horizon: Option<f64>,
    rng: &mut R,
    results: &mut Vec<SimulationResult>,
    warnings: &mut Vec<String>,
) {
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
    let ipreds = if model.n_kappa > 0 {
        let omega_iov = params.omega_iov.as_ref().expect(
            "omega_iov is present whenever the model declares kappa (n_kappa > 0) — \
                 guaranteed by `validate_iov_simulatable` at every simulate entry point (#1019)",
        );
        // One κ vector per occasion group, in `iov_occasion_groups` order — the
        // exact order `predict_iov` indexes its `kappas` argument by. Empty when
        // the subject carries no occasion labels, in which case `predict_iov`
        // falls back to κ = 0 (matching the fit-time no-occasion diagnostic).
        let occ_groups = crate::stats::likelihood::iov_occasion_groups(subject);
        let kappas: Vec<Vec<f64>> = (0..occ_groups.len())
            .map(|_| {
                let z: Vec<f64> = (0..model.n_kappa).map(|_| rng.sample(normal)).collect();
                (&omega_iov.chol * DVector::from_column_slice(&z))
                    .iter()
                    .copied()
                    .collect()
            })
            .collect();
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
    if !model.residual_correlations.is_empty() && !has_frem_rows && !ipreds.is_empty() {
        emit_correlated_residual_rows(
            model,
            subject,
            params,
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
}

/// Draw the correlated residual vector for one subject's Gaussian observation
/// rows from the dense `R` built by [`compute_r_matrix_with_correlations`] (the
/// same matrix FOCE/FOCEI/SAEM/`imp` evaluate the likelihood against), instead
/// of the per-row independent draw `emit_subject_rows` otherwise uses. Callers
/// must already have excluded FREM rows and the empty-correlation case.
///
/// R is factored with a PSD-safe symmetric-eigen square root, so a singular
/// (e.g. `rho = ±1`) or mildly indefinite fixed `block_sigma` yields a valid
/// draw instead of a Cholesky panic. Subjects whose R is diagonal (no paired
/// rows) take a cheap per-row draw and skip the factorization entirely.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_correlated_residual_rows<R: rand::Rng>(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    ipreds: &[f64],
    ruv_scale: f64,
    ruv_mult: Option<&[Vec<f64>]>,
    draw: usize,
    sim: usize,
    normal: rand_distr::Normal<f64>,
    rng: &mut R,
    results: &mut Vec<SimulationResult>,
) {
    let err_keys = model.error_spec.obs_keys(subject);
    let mut r = match ruv_mult {
        Some(mult) => compute_r_matrix_with_correlations_scaled(
            &model.error_spec,
            ipreds,
            err_keys.as_ref(),
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            &subject.obs_l2,
            &params.sigma.values,
            &model.residual_correlations,
            mult,
        ),
        None => compute_r_matrix_with_correlations(
            &model.error_spec,
            ipreds,
            err_keys.as_ref(),
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            &subject.obs_l2,
            &params.sigma.values,
            &model.residual_correlations,
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
    let eps = if !has_offdiag {
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
    };
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
        });
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
    rng: &mut R,
    warnings: &mut Vec<String>,
) -> Vec<SimulationResult> {
    use rand_distr::Normal;

    // Single chokepoint for every `simulate*` variant (both `simulate_inner` and
    // the propensity path funnel through here). Guard the modeled-`RATE` dose
    // precondition once per call, as `predict()` does — `simulate()` runs no
    // data-check otherwise. #324. The dose-compartment routing guard (#375) rides
    // the same chokepoint.
    assert_modeled_doses_supported(model, population);
    assert_dose_compartments_supported(model, population);
    assert_absorption_closed_form_support(model, population);
    assert_absorption_flip_flop_no_twin(model, population, &params.theta);
    // A time-varying covariate on a survival hazard would be silently frozen — panic
    // rather than return a subtly wrong prediction / simulation (#741; fit() Err's).
    #[cfg(feature = "survival")]
    assert_survival_tv_covariates(model, population);
    assert_absorption_dosing_supported(model, population);
    // CTMM (#759) has no simulation path yet: the Gaussian/PK emitter below would write
    // meaningless all-zero DV rows for a discrete-state endpoint (the generator is never
    // sampled). Fail loud rather than return garbage — fit() supports CTMM, simulate()
    // does not. Mirrors the survival guards above; the CLI `--simulate` path returns this
    // as a clean Err in `run_model_simulate`. This is the single simulate chokepoint.
    #[cfg(feature = "markov")]
    assert!(
        !model.has_ctmm(),
        "simulate()/predict() does not support a [markov_model] (CTMM) endpoint yet — its \
         discrete-state trajectory has no simulation path, so the Gaussian emitter would \
         produce meaningless all-zero observations. CTMM simulation is a later slice."
    );

    // ODE-accumulated TTE simulation has preventable preconditions (finite horizon,
    // no resets / left truncation). `simulate_with_options` checks them first and
    // returns a clean Err; the Vec-returning `simulate` / `simulate_with_seed` (and
    // the uncertainty path) funnel through here and cannot signal an error, so
    // enforce the identical contract as a panic rather than emitting wrong rows.
    #[cfg(feature = "survival")]
    if let Err(e) = validate_tte_simulatable(model, population, horizon) {
        panic!("{e}");
    }

    // Same split for the IOV precondition (#1019): `simulate_with_options*` already
    // returned a clean Err; the Vec-returning `simulate` / `simulate_with_seed` and the
    // uncertainty path funnel through here, where the only way to enforce the contract
    // is to fail loud rather than emit rows with no inter-occasion variability.
    if let Err(e) = validate_iov_simulatable(model, params) {
        panic!("{e}");
    }

    // Same split again for the model-vs-population checks (#1083). The
    // `Result`-returning entry points above have already run this list and
    // returned a clean `Err`; `simulate` / `simulate_with_seed` funnel through
    // here and cannot signal, and the failures this catches are silent by
    // construction — a weight that underflows to zero produces finite, plottable
    // rows with the variability quietly removed.
    //
    // Redundant on those `Result` paths, and `simulate_with_uncertainty` reaches
    // here once per draw, so the walk repeats. It is affordable because both
    // expensive halves are gated on the model actually declaring a weight
    // (`has_weighted_kappa` / `has_custom_ruv_magnitude`), which is false for
    // every model that does not use one; and where it is true, the pass is linear
    // in the observations the draw is about to simulate anyway.
    if let Err(e) = first_error(&check_simulation_data(model, population)) {
        panic!("{e}");
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    let n_eta = model.n_eta;

    let mut results = Vec::new();

    for sim_idx in 0..n_sim {
        let sim = sim_idx + 1;
        match matched {
            Some((fitted, omega_inv, method)) => {
                // Draw a pool of one eta per subject for this replicate, then
                // reassign the draws to subjects by matching them to the fitted
                // (posthoc) etas. Each subject keeps its own observed design.
                let n = population.subjects.len();
                let pool: Vec<DVector<f64>> = (0..n)
                    .map(|_| {
                        let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(normal)).collect();
                        &params.omega.chol * DVector::from_column_slice(&z)
                    })
                    .collect();
                let assign = crate::propensity_match::match_draws_to_fitted(
                    &pool, fitted, omega_inv, method,
                );
                for (i, subject) in population.subjects.iter().enumerate() {
                    let mut eta_slice: Vec<f64> = pool[assign[i]].iter().copied().collect();
                    eta_slice.resize(n_eta + model.n_kappa, 0.0);
                    emit_subject_rows(
                        model,
                        subject,
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
                }
            }
            None => {
                for subject in &population.subjects {
                    // Sample eta from N(0, Omega); append zero kappas for IOV models.
                    let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(normal)).collect();
                    let z_vec = DVector::from_column_slice(&z);
                    let eta = &params.omega.chol * z_vec;
                    let mut eta_slice: Vec<f64> = eta.iter().copied().collect();
                    eta_slice.resize(n_eta + model.n_kappa, 0.0);
                    emit_subject_rows(
                        model,
                        subject,
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
                }
            }
        }
    }

    results
}

/// Options controlling `simulate_with_uncertainty()`.
#[derive(Debug, Clone)]
pub struct SimulateUncertaintyOptions {
    /// Number of parameter sets to draw from the uncertainty distribution.
    pub n_uncertainty_draws: usize,
    /// Number of eta/eps replicates simulated *per* parameter draw.
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
pub fn simulate_with_uncertainty(
    model: &CompiledModel,
    population: &Population,
    fit_result: &FitResult,
    opts: &SimulateUncertaintyOptions,
) -> Result<Vec<SimulationResult>, String> {
    use rand::SeedableRng;

    // ODE-accumulated TTE event-time simulation needs a finite horizon, which this
    // uncertainty path does not yet expose — validate for a clean Err here (the
    // inner chokepoint would otherwise enforce the same contract as a panic).
    #[cfg(feature = "survival")]
    validate_tte_simulatable(model, population, None)?;

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
        crate::estimation::uncertainty_samples::fitted_params_from_result(fit_result, model);
    let draws = crate::estimation::uncertainty_samples::draw_parameter_samples(
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
    // Per-subject simulation diagnostics (#762/#763) are collected but not surfaced on
    // this uncertainty-aggregation path (its return is the flat row vec); the underlying
    // per-subject handling — no whole-run panic, degenerate subjects censored — still
    // applies. Use `simulate_with_options` when the warnings matter.
    let mut sim_warnings: Vec<String> = Vec::new();
    for (k, params) in draws.iter().enumerate() {
        // A parameter draw can land in the flip-flop regime even when the point
        // estimate is in-domain. For a twin-less transit/IG closed form,
        // `simulate_inner_with_draw`'s `assert_absorption_flip_flop_no_twin` would then
        // panic — aborting the *entire* uncertainty run. Skip such a draw with a
        // recorded warning instead, so the remaining draws still yield results (#786).
        // The single-shot `predict()`/`simulate()` paths keep the panic (their
        // Vec-returning contract). Twin-carrying models return `None` here and proceed
        // (they reroute per-eval), so this only skips genuinely un-simulatable draws.
        if let Some(msg) = check_absorption_flip_flop_no_twin(model, population, &params.theta) {
            sim_warnings.push(format!("uncertainty draw {} skipped — {}", k + 1, msg));
            continue;
        }
        let mut rows = simulate_inner_with_draw(
            model,
            population,
            params,
            opts.n_sim_per_draw,
            k + 1,
            None,
            None,
            &mut rng,
            &mut sim_warnings,
        );
        results.append(&mut rows);
    }
    Ok(results)
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
}

/// The result of [`simulate_with_options`]: the simulated rows plus any non-fatal
/// per-subject diagnostics collected during the run.
///
/// `warnings` is the simulation analogue of [`FitResult::warnings`]: a subject whose
/// draw degenerates — a non-positive / non-finite analytic hazard rate that produces
/// no event (#763), or a hazard so extreme its recurrent stream is skipped rather than
/// materialised (#762) — is handled *per subject* (censored / skipped, the run
/// continues) and named here, instead of silently vanishing into the censored rows or
/// aborting the whole run. Empty for a clean simulation. The simpler `simulate()` /
/// `simulate_with_seed()` entry points apply the same per-subject handling but return
/// only the rows (no diagnostics channel) — use `simulate_with_options` when the
/// warnings matter (e.g. a population VPC).
#[derive(Debug, Clone, Default)]
pub struct SimulationOutput {
    pub results: Vec<SimulationResult>,
    pub warnings: Vec<String>,
}
