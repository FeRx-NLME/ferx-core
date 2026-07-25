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
    read_nonmem_csv_filtered_mapped, read_nonmem_csv_filtered_tte, read_nonmem_csv_mapped,
    read_nonmem_csv_with_covariates_filtered_mapped, read_nonmem_csv_with_covariates_mapped,
    read_nonmem_csv_with_covariates_tte, SelectionFilter, ERR_COV_MISSING_COLUMNS,
    ERR_COV_NON_NUMERIC,
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

/// Reject an adaptive-dosing simulation on a covariate-selected error model (#658).
///
/// The adaptive assay resolves residual variance by the monitored **compartment**
/// number (`residual_variance_at(cmt, …)`), but `ErrorSpec::Selected`'s `endpoints`
/// map is keyed by the selector's **0-based branch index**, not CMT. A CMT-keyed
/// lookup misses and `variance_at` returns `NaN`, silently corrupting the assay
/// draw. The combination has no coherent meaning today, so reject it loudly rather
/// than emit NaN observations.
pub(crate) fn reject_selected_error_for_adaptive(model: &CompiledModel) -> Result<(), String> {
    if matches!(model.error_spec, ErrorSpec::Selected { .. }) {
        return Err(
            "adaptive-dosing simulation does not support a covariate-selected `[error_model]` \
             (`if (COV …) { … } else { … }`): the assay keys residual error by the monitored \
             compartment, but a selected error model keys endpoints by covariate branch. Use a \
             single-endpoint error model for the monitored signal."
                .to_string(),
        );
    }
    Ok(())
}

/// Reject the model / data combinations the reactive driver cannot yet simulate
/// *faithfully*. The adaptive path carries no process noise, so an SDE `[diffusion]`
/// model would be **silently** wrong — a violation of the "never a silent wrong
/// answer" contract this module promises. Until it is properly supported (#717),
/// reject it with a typed error. Both public entry points funnel through
/// `run_adaptive_population`, so guarding there covers `simulate_adaptive` and
/// `simulate_adaptive_from_spec`.
///
/// Time-varying covariates (and a `TIME`-in-PK model) are **no longer** rejected:
/// the driver recomputes PK per event/segment from the covariate active in that
/// segment (#700). Inter-occasion variability (IOV / `kappa`) is **no longer**
/// rejected either: a fresh κ is drawn per decision window and threaded through the
/// per-segment eta (#701), with occasion = decision index. System-reset events
/// (EVID=3) are **no longer** rejected: the reactive driver now zeros the
/// compartments at each reset and turns off infusions opened before it, and the
/// frozen-replay verifier is reset-aware, so the reset is honored and checked (#716).
pub(crate) fn reject_unsupported_adaptive(
    model: &CompiledModel,
    _population: &Population,
) -> Result<(), String> {
    if model.is_sde() {
        return Err(
            "adaptive-dosing simulation does not support stochastic (`[diffusion]` / SDE) \
             models: the reactive integrator is deterministic and would silently drop the \
             process-noise term. Use the deterministic ODE model for adaptive dosing."
                .to_string(),
        );
    }
    // Time-varying covariates (and `TIME`-in-PK, #700), IOV (#701), and system-reset
    // events (EVID=3, #716) are all now supported by the reactive driver and are no
    // longer rejected here.
    Ok(())
}

/// Options for [`simulate_adaptive`].
///
/// `#[non_exhaustive]`: later knobs (per-subject schedules, the `Dv` path) land
/// as added fields; construct via [`AdaptiveSimulateOptions::default`] and assign
/// the fields you need.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AdaptiveSimulateOptions {
    /// Seed for reproducibility. `None` draws η from entropy.
    pub seed: Option<u64>,
    /// Decision schedule on the subject clock — the times the controller is
    /// consulted. The same schedule is used for every subject in this slice.
    /// Must be non-empty: an empty schedule never consults the controller and is
    /// rejected (it would otherwise be a silent dose-free run).
    pub decision_times: Vec<f64>,
    /// Signals the controller may read at each decision. Each monitor resolves on
    /// its own [`crate::sim::adaptive::ObserveMode`]: `Ipred` reads the latent
    /// prediction; `Dv` adds the endpoint's assay-noise draw (`IPRED + ε·√σ²`) on
    /// the monitor's per-subject RNG substream (#391 S1.5).
    pub monitors: Vec<MonitorSpec>,
    /// Run the frozen-schedule replay verifier after each (subject, replicate)
    /// (default `true`). A divergence is a typed `Err` that taints the whole
    /// result — never a buried warning (#391 Part E).
    pub verify: bool,
    /// Per-run decision cap — the runaway / closed-loop guard. The driver errors
    /// if a schedule exceeds it.
    pub max_decisions: usize,
}

impl Default for AdaptiveSimulateOptions {
    fn default() -> Self {
        Self {
            seed: None,
            decision_times: Vec::new(),
            monitors: Vec::new(),
            verify: true,
            max_decisions: 10_000,
        }
    }
}

/// Result of [`simulate_adaptive`]: the three Part-D artifacts that must agree,
/// returned as one verified unit so a caller can never pair a trajectory with
/// the wrong ledger. All three are long-form rows tagged by `(draw, sim, id)`.
///
/// `#[non_exhaustive]`: the remaining Part-D artifacts (population summary, run
/// manifest) land as added fields without breaking callers who receive this struct.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AdaptiveSimulationResult {
    /// Per-observation predictions: one row per (replicate, subject, obs time).
    /// `ipred` and the `Continuous` value are the individual prediction. Assay-
    /// noised DV monitoring (used for the controller's decisions) is applied inside
    /// the run via each `Dv` monitor's RNG substream, not emitted as a separate row.
    pub trajectories: Vec<SimulationResult>,
    /// Every dose the controllers actually issued, tagged by `(draw, sim)`.
    pub ledger: Vec<DoseLedgerEntry>,
    /// One row per decision (including holds), in schedule order, up to and
    /// including any `Stop`.
    pub decisions: Vec<DecisionLogEntry>,
    /// Per-subject outcome metrics (#391 S2.4): one row per realized `(subject,
    /// draw, sim)` run — cumulative dose, dose-change / hold / discontinuation
    /// counts, the observed-signal summary, and `pct_time_in_window` when the block
    /// declares a `target_window`. Derived from `ledger` + `decisions` alone (no
    /// re-integration). The population summary *with bands* rides with the
    /// uncertainty slice (S5), where bands carry meaning.
    pub metrics: Vec<AdaptiveSubjectMetrics>,
}

/// Simulate state-reactive ("adaptive" / feedback) dosing over a population
/// (epic #391, **beta**).
///
/// For each subject and each of `n_sim` replicates: draw η ~ N(0, Ω), then run
/// one integration driven by a **fresh** controller minted from
/// `make_controller`. The controller is consulted at each `opts.decision_times`
/// point — it reads the declared `opts.monitors` (and the live state /
/// covariates / dose history via [`ControllerCtx`]) and returns the
/// [`DoseAction`]s to apply. The engine owns the timeline, applies
/// bioavailability / lag downstream exactly as for a static dose, and records
/// every realized dose ([`AdaptiveSimulationResult::ledger`]) and every decision
/// ([`AdaptiveSimulationResult::decisions`], including holds).
///
/// ## Controller factory — one per (subject, replicate)
///
/// `make_controller` is a **factory**, not a single shared closure: a fresh
/// controller is built for each run. Real controllers carry per-subject state
/// (debounce / `confirm` counters, windowed AUC, the current titration rung); a
/// single shared `FnMut` would leak that state across subjects and replicates —
/// a silent wrong answer that stateless test controllers never expose. A fresh
/// controller per run makes the isolation structural. A stateless rule is just a
/// factory whose closure ignores its environment, e.g.
/// `|| |ctx: &ControllerCtx| { … }`.
///
/// ## Requirements — typed errors, never a silent wrong answer
///
/// - **Non-empty decision schedule.** An empty `opts.decision_times` never
///   consults the controller (a silent dose-free run) and is rejected.
/// - **ODE model.** The reactive driver runs on the ODE engine; a model with no
///   `[odes]` block is rejected.
/// - **Dose-free subjects.** The regimen is controller-driven; a subject that
///   already carries `doses` is rejected (augmenting a pre-scheduled regimen is a
///   later step).
/// - **Ipred monitors only.** A `Dv` monitor is rejected (needs S1.5).
/// - **Verification (default on).** Each run's realized ledger is replayed
///   through the static engine and checked against the reactive trajectory; a
///   divergence beyond solver tolerance is an `Err`.
///
/// The `draw` tag on every row is `1`: this slice carries no parameter
/// uncertainty (that is Part C / a later slice), only between-subject η.
pub fn simulate_adaptive<F, C>(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    make_controller: F,
    opts: &AdaptiveSimulateOptions,
) -> Result<AdaptiveSimulationResult, String>
where
    F: Fn() -> C,
    C: FnMut(&ControllerCtx) -> Vec<DoseAction>,
{
    let ode = model.ode_spec.as_ref().ok_or_else(|| {
        "simulate_adaptive requires an ODE model (the reactive driver runs on the ODE \
         engine); this model has no [odes] block"
            .to_string()
    })?;

    // The adaptive assay keys residual variance by the monitored compartment
    // number (`residual_variance_at(cmt, …)`), but a `Selected` error model's
    // endpoints are keyed by the covariate selector's 0-based branch index, not
    // CMT — `map.get(&cmt)` would miss and `variance_at` returns NaN, corrupting
    // the assay draw. Reject the combination rather than emit NaN observations (#658).
    reject_selected_error_for_adaptive(model)?;

    // An empty schedule means the controller is never consulted: the result is a
    // dose-free simulation that the verifier (replaying an empty ledger) passes
    // trivially. That is almost always a forgotten `decision_times` (the field
    // defaults to empty), so reject it rather than return a silent no-op — the
    // same "never a silent wrong answer" contract as the other preconditions.
    if opts.decision_times.is_empty() {
        return Err("simulate_adaptive requires a non-empty decision schedule \
             (`AdaptiveSimulateOptions::decision_times`); with no decision times the controller \
             is never consulted and no dose is ever issued"
            .to_string());
    }

    // Programmatic path: cmt-based monitors (no compiled `observe` expression) and
    // a `Vec<DoseAction>` controller with no rule provenance. Adapt both into the
    // engine's paired-monitor / `ControllerDecision` contract; the all-`None`
    // `observe`/`rule` keeps the reactive driver byte-for-byte identical to the
    // pre-S2 behaviour.
    let monitors: Vec<crate::sim::adaptive::AdaptiveMonitor> = opts
        .monitors
        .iter()
        .map(|spec| crate::sim::adaptive::AdaptiveMonitor {
            spec,
            observe: None,
        })
        .collect();
    let make = move || {
        let mut c = make_controller();
        move |ctx: &ControllerCtx| crate::sim::adaptive::ControllerDecision {
            actions: c(ctx),
            rule: None,
        }
    };
    run_adaptive_population(
        model,
        ode,
        population,
        params,
        n_sim,
        &opts.decision_times,
        &monitors,
        make,
        // Programmatic path: no declarative block, so no target band — the
        // window-dependent metric (`pct_time_in_window`) is left unreported.
        None,
        // ...and no `auc_target`, so the signal-AUC pass is skipped and
        // `auc_target_attainment` is left unreported.
        None,
        opts,
    )
}

/// Shared per-(subject, replicate) orchestration behind [`simulate_adaptive`] and
/// [`simulate_adaptive_from_spec`]: draw η ~ N(0, Ω), mint a **fresh** controller,
/// run the reactive ODE driver, (optionally) verify the frozen-schedule replay,
/// and stamp the replicate tags onto the trajectory / ledger / decision rows.
///
/// The decision schedule and monitors are passed explicitly: the programmatic
/// entry takes them from `opts`, the declarative entry derives them from the
/// `[adaptive_dosing]` spec. Each [`AdaptiveMonitor`](crate::sim::adaptive) carries
/// its own optional compiled `observe` expression (the declarative signal) or
/// `None` (the programmatic cmt readout, byte-for-byte unchanged), and the
/// controller returns a `ControllerDecision` so the ledger can record which rule
/// fired.
///
/// Both entries resolve `ode` and validate the schedule before calling, so this
/// helper assumes them well-formed and focuses on the run loop.
#[allow(clippy::too_many_arguments)]
fn run_adaptive_population<F, C>(
    model: &CompiledModel,
    ode: &crate::ode::OdeSpec,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    decision_times: &[f64],
    monitors: &[crate::sim::adaptive::AdaptiveMonitor],
    make_controller: F,
    target_window: Option<(f64, f64)>,
    auc_target: Option<(f64, f64)>,
    opts: &AdaptiveSimulateOptions,
) -> Result<AdaptiveSimulationResult, String>
where
    F: Fn() -> C,
    C: FnMut(&ControllerCtx) -> crate::sim::adaptive::ControllerDecision,
{
    // Reject model/data the reactive driver cannot faithfully simulate (SDE) with a
    // typed error — never a silent wrong answer (#391). IOV, time-varying covariates,
    // and system resets are now supported (#700/#701/#716). Both public entry points
    // funnel through here.
    reject_unsupported_adaptive(model, population)?;

    // #721: the reactive path skipped the shared dose-precondition guards that
    // `simulate()` / `predict()` / `fit()` all run before integrating — modeled `RATE`
    // (#324), analytic-absorption closed-form model/data compatibility, and built-in
    // absorption input-rate pathway fractions / SS / infusion / domain (#588). Without
    // them a feedback-dosed model with, e.g., malformed absorption fractions, an
    // out-of-domain absorption parameter, or (once base regimens are supported, #702) a
    // modeled `RATE` would silently mis-deliver the dose — the exact class #588 closed
    // for the static paths. Surface them as a typed error (the `check_*` form `fit()`
    // uses, and the form `reject_unsupported_adaptive` just above already uses) rather
    // than the panic `simulate()` / `predict()` raise via `assert_*`: this
    // `Result`-returning chokepoint — and its ferx-r FFI surface — then fails with one
    // uniform, recoverable error contract instead of aborting the process.
    // (`check_absorption_closed_form_support` is a no-op here — it returns `None` unless
    // the model is an analytic absorption closed form, which this ODE-only path rejects
    // up front — but is wired for parity and #702 base-regimen support.)
    first_error(&check_modeled_dose_rates(model, population))?;
    // Live for this ODE-only path since #899: `check_dose_compartments` runs the range rule
    // (reject a dose `CMT` past the declared state count) and the `CMT=0`-infusion rule for ODE
    // models, so an adaptive-dosing dataset with an out-of-range or `CMT=0`-infusion seed dose now
    // returns a recoverable `Err` here instead of the engine silently dropping it. (It was a no-op
    // for ODE models before #899 — do not "simplify" this call away as dead code.)
    first_error(&check_dose_compartments(model, population))?;
    if let Some(msg) = check_absorption_closed_form_support(model, population) {
        return Err(msg);
    }
    first_error(&check_absorption_dosing(model, population))?;

    // The occasion bookkeeping assumes a strictly-increasing decision schedule:
    // `occasion_of` (per-decision-window κ selection) scans ascending and stops at the
    // first later time, and `decision_index_of` keys windows by exact time bits. An
    // unsorted schedule would mis-map a record to the wrong occasion's κ, and a
    // duplicated time would split the window-open index from the materialiser's
    // occasion — both silently, since the frozen-replay verifier reuses the same
    // occasion arrays and cannot catch a shared-wrong-input (#701 review). The
    // declarative `[adaptive_dosing]` path already enforces this on its `at` list via
    // `validate_increasing_finite`; guard the programmatic `simulate_adaptive` path
    // here too, at the shared funnel, with the *same* validator (finite + strictly
    // ascending ⇒ no duplicates) so the two paths cannot drift.
    crate::sim::adaptive::validate_increasing_finite(decision_times, "`decision_times`")?;

    use rand::SeedableRng;

    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::make_rng(),
    };
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n_eta = model.n_eta;

    // Root seed for the controller-assay substreams (DV-mode monitors, S1.5).
    // Resolved independently of the η `rng` above so that enabling a `Dv` monitor
    // never shifts the η draws (the all-`Ipred` path stays byte-identical). With no
    // run seed it is drawn from a fresh entropy source, matching the η stream's own
    // nondeterminism.
    let assay_root: u64 = match opts.seed {
        Some(s) => s,
        None => {
            let mut entropy: rand::rngs::StdRng = rand::make_rng();
            entropy.random::<u64>()
        }
    };

    let mut trajectories: Vec<SimulationResult> = Vec::new();
    let mut ledger: Vec<DoseLedgerEntry> = Vec::new();
    let mut decisions: Vec<DecisionLogEntry> = Vec::new();
    let mut metrics: Vec<AdaptiveSubjectMetrics> = Vec::new();

    // `auc_target_attainment` (#391 S2.5b) integrates a dense grid with a single PK
    // snapshot — exact only for constant-covariate subjects. A time-varying (or
    // TIME-in-PK) subject would silently get a frozen-PK AUC (there is no exact
    // per-event dense-grid solver — even the fit path approximates TV states with a
    // warning, and the adaptive result has no warnings channel). So reject the
    // combination loudly (#700) rather than report a wrong metric. Every other
    // adaptive output — predictions, decisions, the dose ledger, `target_window` — is
    // fully per-event covariate-aware; only this one exposure metric is deferred.
    //
    // System resets (EVID=3, #716) are deferred for a *different* reason — not a
    // frozen snapshot. The dense solver itself IS reset-aware: `adaptive_window_signal_aucs`
    // drives `ode_dense_solve_states` with the base subject's `reset_times`, and that
    // solver breaks at each reset and re-seeds the state (`build_segment_break_times` +
    // `apply_segment_boundary`), so a window entirely before or after a reset integrates
    // exactly. The gap is the trapezoid: each inter-decision window is integrated on a
    // *uniform* grid, and a reset is a state discontinuity at an arbitrary time, so the
    // one window straddling it would need pre- and post-reset nodes (values at rt⁻ and rt⁺)
    // to integrate the jump correctly — which the uniform grid does not carry. Rather than
    // report that one window's AUC biased low, defer `auc_target` on reset subjects until
    // the grid places nodes at reset instants. Every other adaptive output is reset-aware
    // (the driver and the frozen-replay verifier).
    if auc_target.is_some()
        && (model.n_kappa > 0
            || population
                .subjects
                .iter()
                .any(|s| crate::pk::subject_needs_per_event_pk(model, s) || s.has_resets()))
    {
        return Err(
            "adaptive-dosing `auc_target` is not yet supported for time-varying-covariate, \
             TIME-in-PK, IOV (`kappa`), or system-reset (EVID=3) subjects. For a drifting \
             covariate or a per-occasion κ the exposure metric integrates its dense grid from a \
             single frozen PK snapshot, which is silently wrong when the PK changes across the \
             horizon. For a reset the dense solver *does* zero the state, but the per-window \
             trapezoid grid carries no node at the reset instant, so the one window straddling \
             the reset would integrate that discontinuity inaccurately. Drop `auc_target` (all \
             other outputs remain per-event / per-occasion / reset aware), or track \
             #700/#701/#716 for a per-event, reset-node-aware AUC."
                .to_string(),
        );
    }

    // Reused per-event PK buffer (#700): filled in place per (sim, subject) on the
    // time-varying path via `compute_event_pk_params_into`, so the hot loop keeps
    // its backing `Vec`s instead of allocating three fresh ones per replicate. The
    // values are recomputed every iteration (η is redrawn); only the allocation is
    // reused. Unused on the constant path (`event_pk` stays `None`).
    let mut event_pk_buf = crate::pk::EventPkParams::default();

    for sim_idx in 0..n_sim {
        let sim = sim_idx + 1;
        for subject in &population.subjects {
            // Draw η ~ N(0, Ω). `eta_slice` (η with κ appended as zeros) is the
            // **baseline** window — the parameters before the first decision, and
            // the reactive driver's fixed eta on a non-IOV run.
            let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(normal)).collect();
            let eta = &params.omega.chol * DVector::from_column_slice(&z);
            let mut eta_slice: Vec<f64> = eta.iter().copied().collect();
            eta_slice.resize(n_eta + model.n_kappa, 0.0);

            // Inter-occasion variability (#701): draw a fresh κ per decision window
            // and build the per-window eta `[η_bsv | κ_g]` the driver threads through
            // each segment, plus `decision_pk[g]` = the PK at decision g under
            // occasion g's κ. Occasion = decision index (per-decision-window). κ is
            // drawn on a dedicated substream keyed by (subject, replicate), disjoint
            // from the η and assay streams, so a non-IOV model (`n_kappa == 0`) draws
            // nothing and this whole block is skipped — the path stays byte-identical.
            let (eta_occ, decision_pk): (
                Option<Vec<Vec<f64>>>,
                Option<Vec<crate::types::PkParams>>,
            ) = if model.n_kappa > 0 {
                let omega_iov = params
                    .omega_iov
                    .as_ref()
                    .expect("omega_iov present when n_kappa > 0");
                let kappa_base =
                    crate::sim::adaptive::subject_kappa_base_seed(assay_root, &subject.id, sim);
                let n_occ = decision_times.len();
                let mut eta_occ = Vec::with_capacity(n_occ);
                let mut decision_pk = Vec::with_capacity(n_occ);
                for g in 0..n_occ {
                    let zk: Vec<f64> = (0..model.n_kappa)
                        .map(|k| crate::sim::adaptive::kappa_standard_normal(kappa_base, g, k))
                        .collect();
                    let kappa_g = &omega_iov.chol * DVector::from_column_slice(&zk);
                    let mut e: Vec<f64> = eta_slice[..n_eta].to_vec();
                    e.extend(kappa_g.iter().copied());
                    // PK at decision g under occasion g's κ, at the covariate the
                    // driver actually sees at that decision. This MUST match the
                    // driver's live `decision_cov` (predictions.rs) exactly: the
                    // obs-coincident snapshot on a record, else the LOCF covariate of
                    // the most-recent record carried forward — NOT the frozen t=0
                    // baseline. On a time-varying-covariate + IOV model whose decision
                    // lands between records, a baseline fallback here would freeze the
                    // decision's covariate at t=0 (the exact #700 defect) while the
                    // driver's readout uses LOCF — and the frozen-replay verifier,
                    // which reuses this same `decision_pk`, could not catch it. Sharing
                    // the one `locf_decision_cov` helper the driver uses makes the two
                    // covariate resolutions a single source of truth (#701 review).
                    let dcov = crate::ode::predictions::locf_decision_cov(
                        decision_times[g],
                        subject,
                        &subject.covariates,
                    );
                    decision_pk.push((model.pk_param_fn)(
                        &params.theta,
                        &e,
                        dcov,
                        decision_times[g],
                    ));
                    eta_occ.push(e);
                }
                (Some(eta_occ), Some(decision_pk))
            } else {
                (None, None)
            };

            // Per-event PK when PK varies across the horizon: an IOV κ that switches
            // per occasion (#701), or a time-varying covariate / pk-only row / `TIME`
            // built-in (#700). The reactive driver then resolves PK per segment from
            // these snapshots. `None` ⇒ constant PK, driven from the frozen `pk`
            // snapshot below. The IOV snapshot is owned (recomputed per replicate as κ
            // is redrawn); the #700-only path reuses `event_pk_buf`. One shared
            // predicate (`subject_needs_per_event_pk`) gates the #700 branch, the
            // materialiser, and the `auc_target` guard so they cannot drift apart.
            let event_pk_iov: Option<crate::pk::EventPkParams> = eta_occ.as_ref().map(|eo| {
                // IOV: each obs / pk-only record carries its occasion's κ (and any
                // #700 covariate); records before the first decision use the baseline.
                crate::pk::compute_event_pk_params_iov(
                    model,
                    subject,
                    &params.theta,
                    &eta_slice,
                    eo,
                    decision_times,
                )
            });
            let event_pk: Option<&crate::pk::EventPkParams> =
                if let Some(ev) = event_pk_iov.as_ref() {
                    Some(ev)
                } else if crate::pk::subject_needs_per_event_pk(model, subject) {
                    crate::pk::compute_event_pk_params_into(
                        model,
                        subject,
                        &params.theta,
                        &eta_slice,
                        &mut event_pk_buf,
                    );
                    Some(&event_pk_buf)
                } else {
                    None
                };

            // Baseline PK snapshot at the start of the simulation horizon (t=0).
            let pk = (model.pk_param_fn)(&params.theta, &eta_slice, &subject.covariates, 0.0);

            // Controller-assay capability for any `Dv` monitor: resolve the
            // endpoint's residual variance by CMT (scaled by this subject's
            // `ruv_scale` for IIV-on-RUV), or `None` when no [error_model] covers
            // that compartment (S1.5 edge a). The base seed keys this
            // (subject, replicate)'s controller-assay substream.
            let ruv_scale = model.residual_var_scale(&eta_slice);
            let sigma = &params.sigma.values;
            let resid_var = |cmt: usize, ipred: f64| -> Option<f64> {
                if !model.has_residual_error_for_cmt(cmt, sigma) {
                    return None;
                }
                Some(model.residual_variance_at(cmt, ipred, sigma) * ruv_scale)
            };
            let assay = crate::sim::adaptive::AssayNoise {
                resid_var: &resid_var,
                base_seed: crate::sim::adaptive::subject_assay_base_seed(
                    assay_root,
                    &subject.id,
                    sim,
                ),
            };

            // A fresh controller per (subject, replicate) — see the factory note.
            let mut controller = make_controller();
            let run: AdaptiveRun = crate::ode::ode_predictions_adaptive_impl(
                ode,
                &pk.values,
                event_pk,
                decision_pk.as_deref(),
                eta_occ.as_deref(),
                &params.theta,
                &eta_slice,
                subject,
                decision_times,
                monitors,
                &mut controller,
                opts.max_decisions,
                Some(&assay),
            )
            .map_err(|e| format!("subject '{}' (sim {sim}): {e}", subject.id))?;

            if opts.verify {
                // #748: the frozen-replay verifier below reuses the *same* precomputed
                // snapshots (`event_pk`, `decision_pk`, `eta_occ`) the driver did, so it
                // validates that the two *consume* them identically — never that they are
                // *correct*. A build-loop that fed a leaf the wrong covariate / occasion /
                // κ (the #732 & #739 "decision covariate frozen at t=0" class) is applied
                // by both sides and passes the replay bit-exact. Close that blind spot by
                // independently re-deriving each snapshot from primitives and bit-asserting
                // it against what the driver was handed — default-on, before the replay.
                verify_adaptive_snapshots(
                    model,
                    params,
                    subject,
                    &eta_slice,
                    decision_times,
                    assay_root,
                    sim,
                    eta_occ.as_deref(),
                    decision_pk.as_deref(),
                    event_pk,
                )
                .map_err(|e| {
                    format!(
                        "adaptive snapshot verification failed for subject '{}' (sim {sim}): {e}",
                        subject.id
                    )
                })?;

                crate::ode::verify_adaptive_frozen_replay(
                    ode,
                    &pk.values,
                    event_pk,
                    decision_pk.as_deref(),
                    eta_occ.as_deref(),
                    &params.theta,
                    &eta_slice,
                    subject,
                    decision_times,
                    &run,
                )
                .map_err(|e| {
                    format!(
                        "frozen-schedule replay verification failed for subject '{}' (sim {sim}): {e}",
                        subject.id
                    )
                })?;
            }

            // Tagged trajectory rows (Ipred only). `run.predictions` is indexed
            // by the subject's observation grid, exactly like `emit_subject_rows`.
            for (j, &pred) in run.predictions.iter().enumerate() {
                trajectories.push(SimulationResult {
                    draw: 1,
                    sim,
                    id: subject.id.clone(),
                    time: obs_row_time(subject, j),
                    cmt: subject.obs_cmts[j],
                    ipred: pred,
                    outcome: SimOutcome::Continuous { value: pred },
                });
            }

            // Signal-AUC pass for the exposure metric (#391 S2.5b): run only when an
            // `auc_target` is declared (its sole consumer) and a monitor exists. It
            // re-integrates the realized ledger on its own dense grid — separate from,
            // and never perturbing, the reactive run + verifier above.
            //
            // Windowed over the **realized** decision times (`run.decisions`), NOT the
            // full scheduled `decision_times`: after a `Stop` the controller
            // discontinues and the later scheduled decisions never happen, so scoring
            // their dose-free, washed-out windows would fold discontinuation — already
            // a first-class outcome (`discontinued` / `time_to_discontinuation`) — into
            // the exposure metric as silent misses (double-counting one event into two
            // metrics). Confining to realized decisions keeps `auc_target_attainment`
            // a clean "of the windows we dosed, how many hit target", on the same
            // realized basis as `pct_time_in_window`. For a run that never
            // discontinues the two decision lists are identical.
            //
            // Dropping the post-`Stop` windows loses no *dosed* window: a declarative
            // `Stop` is dose-free (`adaptive_control::Controller::apply` maps it to
            // `[DoseAction::Stop]`, never `[dose, Stop]`), so the last realized window
            // already covers the last dose. The one controller that *can* dose-then-
            // stop is the programmatic `Vec<DoseAction>` API, and that path runs with
            // `auc_target = None` (the AUC pass is skipped) — so a dose issued *at* a
            // stop never coincides with this metric. If that ever changes (a
            // dose-on-stop reaching the AUC pass), the final dose's window would need
            // explicit handling — the `debug_assert!` in the match arm below is the
            // tripwire for exactly that, and `sim::adaptive::run_has_dose_on_stop` (with
            // its unit test plus the `..._after_discontinuation` test) pins the
            // dose-free-`Stop` invariant this relies on.
            let window_aucs: Vec<f64> = match (auc_target, monitors.first()) {
                (Some(_), Some(mon)) => {
                    // Tripwire (see the note above): realized-window scoring is exact only
                    // while no `Stop` carries a dose. Unreachable today; this fires in
                    // debug/test builds if a future change ever routes a dose-on-stop here,
                    // rather than silently under-reporting that final dose's exposure.
                    debug_assert!(
                        !crate::sim::adaptive::run_has_dose_on_stop(&run.decisions),
                        "auc_target_attainment: a Stop carried a final dose (`[dose, Stop]`); \
                         its post-stop exposure window is unscored under realized-window \
                         scoring — the dose-free-Stop invariant no longer holds, so that \
                         window now needs explicit handling"
                    );
                    let realized_decision_times: Vec<f64> =
                        run.decisions.iter().map(|d| d.time).collect();
                    crate::ode::adaptive_window_signal_aucs(
                        ode,
                        &pk.values,
                        &params.theta,
                        &eta_slice,
                        subject,
                        &realized_decision_times,
                        &run.ledger,
                        mon.observe,
                        mon.spec.cmt,
                    )
                }
                _ => Vec::new(),
            };

            // Per-subject outcome metrics for this run, computed from its realized
            // ledger + decision log (S2.4) and the window AUC series (S2.5b). Taken
            // before the rows are moved into the population vectors below; the
            // `(subject, draw, sim)` key is stamped here rather than read from the
            // rows (whose tags the single-subject driver still leaves at 0).
            metrics.push(crate::sim::adaptive::compute_subject_metrics(
                &subject.id,
                1,
                sim,
                &run.ledger,
                &run.decisions,
                target_window,
                auc_target,
                &window_aucs,
            ));

            // Stamp the replicate tags onto the ledger + decision rows — the
            // single-subject driver emits draw/sim = 0.
            for mut e in run.ledger {
                e.draw = 1;
                e.sim = sim;
                ledger.push(e);
            }
            for mut d in run.decisions {
                d.draw = 1;
                d.sim = sim;
                decisions.push(d);
            }
        }
    }

    Ok(AdaptiveSimulationResult {
        trajectories,
        ledger,
        decisions,
        metrics,
    })
}

/// Validate that the per-(subject, replicate) PK snapshots the reactive driver was
/// handed — and which the frozen-schedule replay verifier reuses **verbatim** — are
/// *correct*, not merely *consumed identically* by both engines (#748).
///
/// `run_adaptive_population` precomputes, once, the PK snapshots it feeds to both the
/// driver and `verify_adaptive_frozen_replay`: the baseline `pk` (t=0), `eta_occ[g]`
/// (the per-decision-window `[η_bsv | κ_g]`), `decision_pk[g]` (the PK at decision
/// `g`), and `event_pk` (the per-record obs / pk-only PK). Because the replay consumes
/// the identical arrays, a **build-loop** error — a wrong covariate, occasion, or κ
/// fed to a leaf — is applied by both sides and passes the replay bit-exact. That is
/// exactly how the "decision covariate frozen at t=0" defect reached production twice
/// (#732, #739): each time only a hand-written regression test using an off-record
/// `f_applied` readout could catch it, because the default-on verifier could not.
///
/// This check closes that blind spot for the three per-occasion / per-event snapshots
/// #748 names. It **independently re-derives** `eta_occ`, `decision_pk`, and `event_pk`
/// from the run's *primitive* inputs — θ, the drawn BSV η (`eta_slice`), the subject's
/// covariates, the decision schedule, and (re-drawn here) the per-occasion κ substream
/// — by calling the **leaf** rules directly (`pk_param_fn`, `locf_decision_cov`,
/// `occasion_of`, `subject_kappa_base_seed` / `kappa_standard_normal`), NOT by
/// re-invoking the build loop's composite assembler. In particular `event_pk` is
/// rebuilt record-by-record inline (occasion via `occasion_of`, PK via `pk_param_fn`
/// at the record's own covariate) rather than through `compute_event_pk_params_iov`,
/// so a wrong occasion or covariate the *composite* would introduce is caught too —
/// not only a wrong array argument. It then bit-asserts (`to_bits`) each re-derivation
/// against what the driver received; `pk_param_fn` and the seeded κ substream are pure,
/// so a correct build agrees to the bit and a divergent one fails loudly, default-on,
/// for **every** model.
///
/// The teeth come from this being a **second orchestration over the same leaves**: it
/// **must not** be refactored to call the build loop's composite assembler, or a defect
/// edited into that assembler would be mirrored on both sides and slip through again.
/// The leaf rules themselves (the covariate LOCF rule, the occasion map, the κ draw)
/// are the single source of truth and are unit-tested in isolation; θ and `eta_slice`
/// are trusted primitive draws (not "built" snapshots) so — like the seed and Ω — they
/// are consumed, not re-derived.
///
/// Scope: the constant-path baseline `pk` (a single t=0 evaluation at the subject-static
/// covariate) is a shared snapshot too, but a trivial one — it is left to the degenerate
/// oracle rather than re-derived per run here; extending the check to it is a possible
/// follow-up. On the constant-covariate / non-IOV path (no `eta_occ`, no `event_pk`)
/// this function is therefore a no-op.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_adaptive_snapshots(
    model: &CompiledModel,
    params: &ModelParameters,
    subject: &Subject,
    eta_slice: &[f64],
    decision_times: &[f64],
    assay_root: u64,
    sim: usize,
    eta_occ: Option<&[Vec<f64>]>,
    decision_pk: Option<&[PkParams]>,
    event_pk: Option<&crate::pk::EventPkParams>,
) -> Result<(), String> {
    let n_eta = model.n_eta;

    // ── IOV path: re-derive eta_occ + decision_pk, then event_pk inline ─────────
    if let (Some(eta_occ), Some(decision_pk)) = (eta_occ, decision_pk) {
        let omega_iov = params.omega_iov.as_ref().ok_or_else(|| {
            "an IOV run (eta_occ / decision_pk present) carries no omega_iov — a build-loop \
             invariant violation (#748)"
                .to_string()
        })?;
        let kappa_base =
            crate::sim::adaptive::subject_kappa_base_seed(assay_root, &subject.id, sim);
        let n_occ = decision_times.len();

        if eta_occ.len() != n_occ || decision_pk.len() != n_occ {
            return Err(format!(
                "occasion snapshot arrays are mis-sized: expected {n_occ} (one per decision), \
                 got eta_occ.len()={}, decision_pk.len()={} (#748)",
                eta_occ.len(),
                decision_pk.len()
            ));
        }

        let mut eta_occ_check: Vec<Vec<f64>> = Vec::with_capacity(n_occ);
        for g in 0..n_occ {
            // Re-draw occasion g's κ on the dedicated substream and assemble the
            // per-window eta [η_bsv | κ_g] — an independent copy of the build loop.
            let zk: Vec<f64> = (0..model.n_kappa)
                .map(|k| crate::sim::adaptive::kappa_standard_normal(kappa_base, g, k))
                .collect();
            let kappa_g = &omega_iov.chol * DVector::from_column_slice(&zk);
            let mut e: Vec<f64> = eta_slice[..n_eta].to_vec();
            e.extend(kappa_g.iter().copied());

            if !slice_bits_eq(&e, &eta_occ[g]) {
                return Err(format!(
                    "eta_occ[{g}] (occasion κ for the decision at t={}) diverges from an \
                     independent re-draw — a wrong occasion→κ keying or [η | κ] assembly in the \
                     build loop. The frozen-replay verifier reuses this array and cannot catch \
                     it (#748).",
                    decision_times[g]
                ));
            }

            // decision_pk[g] must equal pk_param_fn at the covariate the driver's live
            // `decision_cov` uses — LOCF of the most-recent record, NOT the frozen t=0
            // baseline (the twice-fixed #732 / #739 defect).
            let dcov = crate::ode::predictions::locf_decision_cov(
                decision_times[g],
                subject,
                &subject.covariates,
            );
            let dpk = (model.pk_param_fn)(&params.theta, &e, dcov, decision_times[g]);
            if !pk_bits_eq(&dpk, &decision_pk[g]) {
                return Err(format!(
                    "decision_pk[{g}] (decision at t={}) diverges from an independent \
                     re-derivation at the LOCF decision covariate — the 'decision covariate \
                     frozen at t=0' class fixed in #732 / #739. Both the driver and the \
                     frozen-replay verifier consume this snapshot, so the replay cannot catch \
                     it (#748).",
                    decision_times[g]
                ));
            }
            eta_occ_check.push(e);
        }

        // event_pk (per-record obs / pk-only PK) re-derived from the *check's* occasion
        // eta, so a corrupt build eta_occ cannot launder itself into event_pk.
        let ev = event_pk.ok_or_else(|| {
            "an IOV run carries no event_pk — κ makes PK per-occasion, so obs / pk-only records \
             must each carry a per-event snapshot (#748)"
                .to_string()
        })?;
        return check_event_pk_records(
            model,
            &params.theta,
            eta_slice,
            subject,
            decision_times,
            Some(&eta_occ_check),
            ev,
        );
    }

    // ── TV-covariate / TIME-in-PK path without IOV ──────────────────────────────
    // Every record uses the baseline η; re-derive per record and compare.
    if let Some(ev) = event_pk {
        return check_event_pk_records(
            model,
            &params.theta,
            eta_slice,
            subject,
            decision_times,
            None,
            ev,
        );
    }

    // ── Constant-covariate path: only the baseline `pk` (checked above) applies. ─
    Ok(())
}

/// Two `f64`s are the same to the bit (so two runs of the *same* deterministic
/// computation agree, and a NaN equals itself). Used by the #748 snapshot check,
/// where any difference is a real build divergence, not solver slack.
#[inline]
fn f64_bits_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

/// Bit-equality of two PK parameter snapshots (#748).
pub(crate) fn pk_bits_eq(a: &PkParams, b: &PkParams) -> bool {
    a.values
        .iter()
        .zip(b.values.iter())
        .all(|(x, y)| f64_bits_eq(*x, *y))
}

/// Bit-equality of two eta slices (#748).
fn slice_bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| f64_bits_eq(*x, *y))
}

/// Re-derive each per-record (obs / EVID=2 pk-only) PK snapshot **inline** — occasion
/// via [`occasion_of`](crate::pk::occasion_of), PK via `pk_param_fn` at the record's
/// own covariate — and bit-check it against the `event_pk` the driver used (#748).
///
/// This is an *independent second derivation*, NOT a re-call of
/// `compute_event_pk_params_iov`: a wrong occasion grouping or per-record covariate the
/// composite builder would introduce is caught here, not only a wrong array argument.
/// `eta_occ_check` is the independently re-derived per-window eta (IOV); `None` ⇒ the
/// non-IOV path, where every record uses the baseline `eta_slice`. The dose-free
/// adaptive base subject carries no dose events, so `event_pk.dose` must stay empty.
fn check_event_pk_records(
    model: &CompiledModel,
    theta: &[f64],
    eta_slice: &[f64],
    subject: &Subject,
    decision_times: &[f64],
    eta_occ_check: Option<&[Vec<f64>]>,
    got: &crate::pk::EventPkParams,
) -> Result<(), String> {
    // Eta in effect at a record time `t`: occasion `g`'s per-window eta when IOV is
    // active and a window is open, else the baseline (matches the driver's `eta_for`
    // over the same `occasion_of`).
    let eta_at = |t: f64| -> &[f64] {
        match eta_occ_check {
            Some(eo) => match crate::pk::occasion_of(decision_times, t) {
                Some(g) => eo[g].as_slice(),
                None => eta_slice,
            },
            None => eta_slice,
        }
    };

    if got.dose.len() != subject.doses.len() {
        return Err(format!(
            "event_pk.dose has {} entr(ies) but the subject has {} dose record(s) (#748)",
            got.dose.len(),
            subject.doses.len()
        ));
    }
    if got.obs.len() != subject.obs_times.len() {
        return Err(format!(
            "event_pk.obs has {} entr(ies) but the subject has {} observation record(s) (#748)",
            got.obs.len(),
            subject.obs_times.len()
        ));
    }
    for j in 0..subject.obs_times.len() {
        let t = subject.obs_times[j];
        let want = (model.pk_param_fn)(theta, eta_at(t), subject.obs_cov(j), t);
        if !pk_bits_eq(&want, &got.obs[j]) {
            return Err(format!(
                "event_pk.obs[{j}] (record at t={t}) diverges from an independent re-derivation \
                 — a wrong per-event covariate or occasion κ in the per-event PK builder, reused \
                 verbatim by the frozen-replay verifier, which cannot catch it (#748)"
            ));
        }
    }
    if got.pk_only.len() != subject.pk_only_times.len() {
        return Err(format!(
            "event_pk.pk_only has {} entr(ies) but the subject has {} EVID=2 record(s) (#748)",
            got.pk_only.len(),
            subject.pk_only_times.len()
        ));
    }
    for m in 0..subject.pk_only_times.len() {
        let t = subject.pk_only_times[m];
        let want = (model.pk_param_fn)(theta, eta_at(t), subject.pk_only_cov(m), t);
        if !pk_bits_eq(&want, &got.pk_only[m]) {
            return Err(format!(
                "event_pk.pk_only[{m}] (EVID=2 record at t={t}) diverges from an independent \
                 re-derivation — a wrong per-event covariate or occasion κ (#748)"
            ));
        }
    }
    Ok(())
}

/// Simulate a declarative `[adaptive_dosing]` block over a population — the
/// file-driven counterpart to [`simulate_adaptive`] (epic #391, **beta**).
///
/// Where [`simulate_adaptive`] takes a hand-written controller closure, this entry
/// point takes the parsed `[adaptive_dosing]` `spec` and compiles it into the
/// *same* reactive engine: the `observe` expression becomes the monitored signal,
/// the `when … : …` ladder becomes the controller, and the block's `at` becomes
/// the decision schedule. Everything downstream — the dose ledger, the decision
/// log (including holds), and the frozen-schedule replay verifier — is identical
/// to the programmatic path, so the declarative block inherits every S1 guarantee
/// (a re-emitted fixed regimen reproduces [`simulate`] bit-for-bit via the
/// verifier; a genuinely reactive schedule is replay-checked each run).
///
/// Obtain `spec` from [`crate::parse_full_model_file`]
/// (`ParsedModel::adaptive_dosing`, `None` when the model declares no block).
///
/// ## The spec owns the schedule and the monitor
///
/// The decision schedule (`spec.at`) and the monitored signal (`spec.observe`)
/// come from the block, so `opts.decision_times` and `opts.monitors` **must be
/// left empty**: setting either is a typed error rather than a silently-ignored
/// field (the spec would otherwise have two sources of truth). `opts.seed`,
/// `opts.verify`, and `opts.max_decisions` apply exactly as for
/// [`simulate_adaptive`].
///
/// The `observe` expression is compiled against the model, so it titrates on the
/// derived signal it names — `observe = central / V` drives on the concentration,
/// not the raw compartment amount. With `with_assay_error`, the designated
/// endpoint's residual error noises the reading on the S1.5 controller-assay
/// substream; an `observe` whose endpoint is ambiguous is rejected at parse time.
pub fn simulate_adaptive_from_spec(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    spec: &crate::sim::adaptive::AdaptiveDosingSpec,
    opts: &AdaptiveSimulateOptions,
) -> Result<AdaptiveSimulationResult, String> {
    // The spec is the single source of truth for the decision schedule and the
    // monitored signal; an `opts` that *also* sets them is ambiguous (two
    // schedules, two monitors) — reject it rather than silently pick one.
    if !opts.decision_times.is_empty() {
        return Err(
            "simulate_adaptive_from_spec takes its decision schedule from the \
             [adaptive_dosing] block's `at`; leave `opts.decision_times` empty"
                .to_string(),
        );
    }
    if !opts.monitors.is_empty() {
        return Err(
            "simulate_adaptive_from_spec takes its monitor from the [adaptive_dosing] \
             block's `observe`; leave `opts.monitors` empty"
                .to_string(),
        );
    }

    // Compile the block against the model: the `observe` expression, the single
    // `signal` monitor (carrying the assay endpoint under `with_assay_error`), and
    // the per-(subject, replicate) controller factory. This is also the ODE gate —
    // the reactive driver runs on the ODE engine, so an analytical model is rejected
    // here rather than allowed to silently produce nothing.
    let compiled = crate::sim::adaptive_control::compile_adaptive(model, spec)
        .map_err(|e| format!("simulate_adaptive_from_spec: {e}"))?;
    // Parity with `fit()`: model-referenced covariates (e.g. a `Selected` error
    // model's selector) must be present too, not just the `observe` signal (#658).
    first_error(&check_covariates(model, population))?;
    // A `Selected` error model keys endpoints by selector branch, not CMT, so the
    // compartment-keyed assay would draw NaN — reject it (see the helper's note, #658).
    reject_selected_error_for_adaptive(model)?;
    // An `observe` covariate absent from the data would silently read 0.0 and
    // drive the controller off a wrong signal (`central / WT` → central / 0 = inf).
    // Apply the same loud check fits use for model covariates (`check_covariates`).
    let missing: Vec<&str> = compiled
        .observe_covariates
        .iter()
        .filter(|name| !population.covariate_names.iter().any(|n| n == *name))
        .map(|s| s.as_str())
        .collect();
    if !missing.is_empty() {
        let available = if population.covariate_names.is_empty() {
            "(none)".to_string()
        } else {
            population.covariate_names.join(", ")
        };
        return Err(format!(
            "simulate_adaptive_from_spec: [adaptive_dosing] `observe` references covariate(s) \
             not found in data (case-sensitive): {}. Available covariate columns: {}.",
            missing.join(", "),
            available
        ));
    }
    let ode = model
        .ode_spec
        .as_ref()
        .expect("compile_adaptive accepted the model, so it carries an ODE spec");
    // Pair the single `signal` monitor with its compiled `observe` expression (the
    // latent/`Ipred` case); under `Dv` `compiled.observe` is `None`, so the driver
    // reads the model's own output for that cmt — value and σ from one source.
    let monitors = vec![crate::sim::adaptive::AdaptiveMonitor {
        spec: &compiled.monitors[0],
        observe: compiled.observe.as_ref(),
    }];

    run_adaptive_population(
        model,
        ode,
        population,
        params,
        n_sim,
        &spec.at,
        &monitors,
        compiled.make_controller.as_ref(),
        // The declarative block's optional therapeutic band feeds the
        // `pct_time_in_window` metric (it never influences dosing).
        spec.target_window,
        // ...and the optional exposure band feeds `auc_target_attainment` (likewise
        // metrics-only); `Some` here is what turns on the signal-AUC pass.
        spec.auc_target,
        opts,
    )
}
