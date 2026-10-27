//! NCA-based starting value estimation.
//!
//! The single entry point is [`inits_from_nca`], which selects one of three
//! NCA-based strategies via [`NcaInit`]:
//!
//! - [`NcaInit::Nca`] — non-compartmental analysis arithmetic only, no
//!   simulation or optimisation; fast enough to call before every fit.
//! - [`NcaInit::Sweep`] — runs the NCA step first, then an rRMSE grid sweep
//!   over parameters NCA cannot estimate (Q, V2, Q2, V3, or all thetas for ODE
//!   models) using [`crate::api`] predictions with etas = 0. ~100–500 ms for
//!   typical 2/3-cpt datasets. This is the default when enabled via a boolean.
//! - [`NcaInit::Ebe`] — like `Sweep` but evaluates the grid with empirical
//!   Bayes estimates (etas ≠ 0); more accurate under large IIV. Falls back to
//!   `Sweep` for ODE models.

mod nca;
mod pooling;
mod sweep;

use rayon::prelude::*;

use crate::diagnostics::first_error;

use crate::types::{
    CompiledModel, ModelParameters, OmegaMatrix, PkModel, Population, PK_IDX_CL, PK_IDX_CV2,
    PK_IDX_F, PK_IDX_KA, PK_IDX_LAGTIME, PK_IDX_MAT, PK_IDX_MTT, PK_IDX_N, PK_IDX_Q, PK_IDX_Q3,
    PK_IDX_V, PK_IDX_V2, PK_IDX_V3,
};
use nca::{
    biexponential_peel, first_dose_amt, first_dose_obs, nca_one_cpt_iv, nca_one_cpt_oral,
    SubjectNca,
};
use pooling::{pool_nca, PopNca};
use sweep::{sweep_slots, sweep_slots_ebe, sweep_unwritten_thetas, sweep_unwritten_thetas_ebe};

/// Output of [`inits_from_nca`].
#[derive(Debug, Clone)]
pub struct SuggestedStart {
    /// Clone of `model.default_params` with non-fixed thetas overwritten by NCA
    /// estimates and, for the CL eta, omega updated from inter-subject variability.
    pub params: ModelParameters,
    /// Per-theta (and per-omega) notes about what was and wasn't estimated.
    pub warnings: Vec<String>,
}

/// Which NCA-based starting-value strategy [`inits_from_nca`] should run.
///
/// User-facing names (fit option, CLI flag, R argument): `nca`, `nca_sweep`,
/// `nca_ebe`. A boolean `true` maps to [`NcaInit::Sweep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NcaInit {
    /// Non-compartmental analysis arithmetic only (fastest).
    Nca,
    /// NCA, then an etas = 0 rRMSE grid sweep for parameters NCA can't estimate.
    #[default]
    Sweep,
    /// NCA, then an EBE-based (etas ≠ 0) rRMSE grid sweep; falls back to
    /// `Sweep` for ODE models.
    Ebe,
}

/// Derive NCA-based starting values from the data, using the strategy selected
/// by `method`. See [`NcaInit`] for the trade-offs between the three variants.
///
/// Common to all strategies:
/// - Fixed thetas are never overwritten.
/// - Covariate-effect thetas (no mu-referencing link) keep the model default.
/// - All written values are clamped to `[theta_lower, theta_upper]`.
/// - Omega for the CL/CL_F eta is updated from inter-subject CV² when ≥ 3
///   subjects have a valid CL estimate; all other omegas keep their defaults.
///
/// # Errors
///
/// A modeled-`RATE` dose the model cannot honour, or a dose into a compartment it cannot
/// deliver into, is an `Err` carrying the text `fit()` gives for that precondition (#898).
/// These are the only two it checks.
pub fn inits_from_nca(
    model: &CompiledModel,
    population: &Population,
    method: NcaInit,
) -> Result<SuggestedStart, crate::diagnostics::EngineError> {
    // Reached from `fit()` (which validates first) but also a standalone public
    // entrypoint; the Sweep/Ebe strategies predict, so guard the modeled-`RATE`
    // dose precondition here too — same loud-not-silent contract as
    // `predict()`/`simulate()` (#324). No-op for the common all-`Fixed` dataset.
    first_error(&crate::api::check_modeled_dose_rates(model, population))?;
    first_error(&crate::api::check_dose_compartments(model, population))?;
    match method {
        NcaInit::Nca => Ok(nca_only(model, population)),
        NcaInit::Sweep => nca_with_sweep(model, population),
        NcaInit::Ebe => nca_with_ebe(model, population),
    }
}

/// Fast NCA-based starting value estimation (`NcaInit::Nca`).
///
/// Derives theta starting values from non-compartmental analysis — no inner loop,
/// no simulation.  Typical cost: < 5 ms for 100 subjects.
fn nca_only(model: &CompiledModel, population: &Population) -> SuggestedStart {
    let (nca_vals, mut warnings) = run_nca(model, population);
    let params = build_params(model, &nca_vals, &mut warnings);
    SuggestedStart { params, warnings }
}

/// NCA + rRMSE sweep (`NcaInit::Sweep`).
///
/// Runs the NCA step first, then sweeps every non-fixed theta it left at
/// the model default — via sequential 1D coordinate sweeps over a log-space grid
/// using population predictions (etas = 0) to minimise rRMSE.
///
/// This is model-agnostic: it works for analytical PK (sweeps Q/V2/etc. when
/// biexponential peeling failed), ODE PK, PD, and PKPD models without any
/// knowledge of parameter names or model structure.
///
/// Cost: 9 `predict()` calls per unwritten theta.  For a typical 2-cpt PK model
/// where peeling succeeded, 0–2 thetas remain; for a PD model with 5 free
/// parameters, ~45 calls (~50 ms on 100 subjects).
fn nca_with_sweep(
    model: &CompiledModel,
    population: &Population,
) -> Result<SuggestedStart, crate::diagnostics::EngineError> {
    let mut base = nca_only(model, population);

    // Collect non-fixed thetas that Option A left unchanged (still at model default),
    // less a θ two PK roles share: the collision guard promised it its default.
    let guarded = guard_slot_thetas(model);
    let mut remaining: Vec<usize> = (0..model.default_params.theta.len())
        .filter(|&i| {
            !base.params.theta_fixed[i]
                && !guarded.is_shared(i)
                && (base.params.theta[i] - model.default_params.theta[i]).abs() < 1e-12
        })
        .collect();

    if remaining.is_empty() {
        return Ok(base);
    }

    // Exclude covariate-effect thetas (e.g. THETA_WT, THETA_CRCL) from the sweep.
    // These have no mu-referencing link and no standard PK slot mapping, so rRMSE
    // sweeps on them produce meaningless estimates. A structural theta satisfies at
    // least one of: (a) it is a TV parameter referenced via mu_refs, or (b) it maps
    // to a known PK slot (CL, V, Q, V2, Ka, F, Q3, V3, LAGTIME).
    {
        let mu_ref_theta_names: std::collections::HashSet<&str> = model
            .mu_refs
            .values()
            .map(|mr| mr.theta_name.as_str())
            .collect();
        let pk_slot_indices: std::collections::HashSet<usize> = [
            PK_IDX_CL,
            PK_IDX_V,
            PK_IDX_Q,
            PK_IDX_V2,
            PK_IDX_KA,
            PK_IDX_F,
            PK_IDX_Q3,
            PK_IDX_V3,
            PK_IDX_LAGTIME,
        ]
        .iter()
        .filter_map(|&slot| find_theta_for_slot(model, slot))
        .collect();

        let n_before = remaining.len();
        remaining.retain(|&i| {
            let name = &model.default_params.theta_names[i];
            mu_ref_theta_names.contains(name.as_str()) || pk_slot_indices.contains(&i)
        });
        let n_excluded = n_before - remaining.len();
        if n_excluded > 0 {
            base.warnings.push(format!(
                "inits_from_nca (nca_sweep): excluded {n_excluded} covariate theta(s) from rRMSE sweep (no mu-ref link and no standard PK slot)"
            ));
        }
    }

    if remaining.is_empty() {
        return Ok(base);
    }

    // Joint 2D sweeps for highly correlated pairs before independent 1D sweeps.
    // CL and V (or CL/F, V/F) lie on a ridge in the rRMSE landscape — sweeping
    // them independently always moves along the ridge rather than across it.
    // Same applies to (Q, V2) and (F, CL) when both are unwritten.
    for (slot_a, slot_b, label) in &[
        (PK_IDX_CL, PK_IDX_V, "CL/V"),
        (PK_IDX_Q, PK_IDX_V2, "Q/V2"),
        (PK_IDX_F, PK_IDX_CL, "F/CL"),
    ] {
        let idx_a = find_theta_for_slot(model, *slot_a);
        let idx_b = find_theta_for_slot(model, *slot_b);
        if let (Some(ia), Some(ib)) = (idx_a, idx_b) {
            if ia != ib && remaining.contains(&ia) && remaining.contains(&ib) {
                let (swept, w) = sweep_slots(
                    model,
                    population,
                    &base.params,
                    *slot_a,
                    *slot_b,
                    9,
                    10.0,
                    label,
                )?;
                base.params = swept;
                base.warnings.extend(w);
                remaining.retain(|&i| i != ia && i != ib);
            }
        }
    }

    // Independent 1D sweeps for any remaining unwritten thetas.
    if !remaining.is_empty() {
        let (swept, w) =
            sweep_unwritten_thetas(model, population, &base.params, &remaining, 9, 10.0)?;
        base.params = swept;
        base.warnings.extend(w);
    }

    Ok(base)
}

/// EBE-based NCA + rRMSE sweep (`NcaInit::Ebe`).
///
/// Runs the NCA step first, then sweeps every non-fixed **lognormal** structural
/// theta it left at the model default, using individual empirical
/// Bayes estimates (etas ≠ 0) to evaluate rRMSE at each grid point.
///
/// Unlike Option B, this method correctly accounts for between-subject
/// variability: each grid point runs a warm-started inner optimisation
/// (`run_inner_loop_warm`, max 20 iterations, tolerance 1e-3) before
/// computing rRMSE against individual observations.  EBEs from one grid
/// point warm-start the next, so convergence is typically 3–5 iterations
/// and total cost is ~20–25× Option B for analytical PK models.
///
/// **ODE models**: EBE sweeps on ODE models are prohibitively slow (each
/// inner iteration requires numerical integration) and unreliable when
/// starting from uninformed defaults.  For ODE models this function falls
/// back to Option B (etas=0 sweep) automatically, with a warning.
///
/// **Logit-parameterised thetas** (e.g. bioavailability F via
/// `inv_logit(logit(THETA_F) + ETA_F)`): excluded from the EBE sweep
/// because the eta can partially compensate for a wrong TV on the logit
/// scale, making the rRMSE landscape unreliable.  They are swept
/// afterwards with etas=0 (Option B style).
///
/// Typical wall-clock cost on a 30-subject analytical 2-cpt model: 200–500 ms.
fn nca_with_ebe(
    model: &CompiledModel,
    population: &Population,
) -> Result<SuggestedStart, crate::diagnostics::EngineError> {
    // ODE fallback: EBE sweeps require per-subject numerical integration per
    // inner iteration — too slow (~minutes) and unreliable from uninformed
    // defaults.  Delegate to the etas=0 sweep directly (which runs NCA + sweep)
    // rather than calling nca_only() first and then nca_with_sweep()
    // (which would run NCA twice).
    if model.ode_spec.is_some() {
        let mut result = nca_with_sweep(model, population)?;
        result.warnings.insert(0,
            "inits_from_nca (nca_ebe): ODE model — EBE sweep skipped (too slow; ODE integration per inner iteration). Falling back to etas=0 sweep (nca_sweep).".into(),
        );
        return Ok(result);
    }

    let mut base = nca_only(model, population);

    // As in `nca_with_sweep`, a θ two PK roles share keeps its default.
    let guarded = guard_slot_thetas(model);
    let mut remaining: Vec<usize> = (0..model.default_params.theta.len())
        .filter(|&i| {
            !base.params.theta_fixed[i]
                && !guarded.is_shared(i)
                && (base.params.theta[i] - model.default_params.theta[i]).abs() < 1e-12
        })
        .collect();

    if remaining.is_empty() {
        return Ok(base);
    }

    // For EBE sweeps, only include lognormal-parameterised thetas
    // (mu_ref.log_transformed() = true, i.e. THETA * exp(ETA) form).
    // Logit thetas are excluded from the EBE sweep: the eta can compensate
    // for a wrong TV on the logit scale, creating spurious rRMSE minima.
    // Covariate thetas (no mu_ref at all) are also excluded.
    // Logit thetas will be swept afterwards with etas=0 (Option B style).
    // Partition remaining into three groups:
    //   lognormal  — THETA * exp(ETA): EBE sweep is reliable
    //   logit      — inv_logit(logit(THETA) + ETA): etas=0 sweep (eta compensates on logit scale)
    //   covariate  — no mu_ref at all: excluded entirely (same as Option B)
    let lognormal_theta_names: std::collections::HashSet<&str> = model
        .mu_refs
        .values()
        .filter(|mr| mr.log_transformed())
        .map(|mr| mr.theta_name.as_str())
        .collect();
    let logit_theta_names: std::collections::HashSet<&str> = model
        .mu_refs
        .values()
        .filter(|mr| !mr.log_transformed())
        .map(|mr| mr.theta_name.as_str())
        .collect();

    let mut logit_remaining: Vec<usize> = Vec::new();
    let mut n_covariate_excluded = 0usize;

    // Split remaining: lognormal → EBE sweep, logit → etas=0 sweep, else drop.
    remaining.retain(|&i| {
        let name = &model.default_params.theta_names[i];
        if lognormal_theta_names.contains(name.as_str()) {
            true // keep for EBE sweep
        } else if logit_theta_names.contains(name.as_str()) {
            logit_remaining.push(i);
            false
        } else {
            n_covariate_excluded += 1;
            false
        }
    });

    if n_covariate_excluded > 0 {
        base.warnings.push(format!(
            "inits_from_nca (nca_ebe): excluded {n_covariate_excluded} covariate theta(s) from rRMSE sweep"
        ));
    }

    if remaining.is_empty() && logit_remaining.is_empty() {
        return Ok(base);
    }

    // Joint 2D EBE sweeps for correlated pairs.
    for (slot_a, slot_b, label) in &[
        (PK_IDX_CL, PK_IDX_V, "CL/V"),
        (PK_IDX_Q, PK_IDX_V2, "Q/V2"),
        (PK_IDX_F, PK_IDX_CL, "F/CL"),
    ] {
        let idx_a = find_theta_for_slot(model, *slot_a);
        let idx_b = find_theta_for_slot(model, *slot_b);
        if let (Some(ia), Some(ib)) = (idx_a, idx_b) {
            if ia != ib && remaining.contains(&ia) && remaining.contains(&ib) {
                let (swept, w) = sweep_slots_ebe(
                    model,
                    population,
                    &base.params,
                    *slot_a,
                    *slot_b,
                    9,
                    10.0,
                    label,
                );
                base.params = swept;
                base.warnings.extend(w);
                remaining.retain(|&i| i != ia && i != ib);
            }
        }
    }

    // 1D EBE sweeps for remaining lognormal structural thetas.
    if !remaining.is_empty() {
        let (swept, w) =
            sweep_unwritten_thetas_ebe(model, population, &base.params, &remaining, 9, 10.0);
        base.params = swept;
        base.warnings.extend(w);
    }

    // Logit-parameterised thetas swept with etas=0 (same as Option B).
    // EBE sweeps are unreliable for logit params due to eta-compensation effects.
    if !logit_remaining.is_empty() {
        let (swept, w) =
            sweep_unwritten_thetas(model, population, &base.params, &logit_remaining, 9, 10.0)?;
        base.params = swept;
        base.warnings.extend(w);
    }

    Ok(base)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Run per-subject NCA in parallel and return pooled estimates + warnings.
fn run_nca(model: &CompiledModel, population: &Population) -> (PopNca, Vec<String>) {
    let mut warnings = Vec::new();

    if population.subjects.is_empty() {
        warnings.push("inits_from_nca: no subjects in population; using model defaults".into());
        return (empty_pop_nca(), warnings);
    }

    // ODE models: `pk_indices` does route a canonical name (CL, V, KA, …) to its PK
    // slot (`ode_param_slots`), but that is a naming convention, not a structural
    // guarantee — the `[odes]` RHS decides what a parameter means, so an ODE `CL`
    // need not be the clearance NCA estimates. Every other name (KE, EMAX, …) takes
    // the lowest free slot, often a canonical index the model left unused (declare
    // `V, KA, KE` and KE lands in slot 0 = `PK_IDX_CL`), so a slot lookup such as
    // `find_theta_for_slot` would take it for that PK parameter; and `pk_model` is
    // only a placeholder. NCA can't reliably map its estimates onto the user's
    // parameters, so fall back to model defaults and let the nca_sweep method
    // sweep them via rRMSE.
    if model.ode_spec.is_some() {
        warnings.push(
            "inits_from_nca: ODE model detected; NCA estimation skipped (parameter names are user-defined). Use the nca_sweep method for rRMSE-based sweep.".into(),
        );
        return (empty_pop_nca(), warnings);
    }
    // A compartment-free model (#811) has no doses, no concentrations and no
    // CL/V to estimate — non-compartmental analysis is undefined for it, and its
    // `pk_model` is a placeholder the dispatch below would otherwise read as a
    // real one-compartment IV model and "estimate" clearances from a
    // dose-response or time-course curve.
    if model.is_algebraic() {
        warnings.push(
            "inits_from_nca: compartment-free ($PRED-equivalent) model; NCA estimation skipped \
             (there are no doses or concentrations to analyse, and the parameters are not PK \
             parameters). Set starting estimates in [parameters], or use the nca_sweep method."
                .into(),
        );
        return (empty_pop_nca(), warnings);
    }

    // NCA runs from `prepare_run`, before any fit has installed a pool, so this
    // `par_iter` would otherwise land on Rayon's global pool at one worker per logical
    // CPU regardless of the requested thread count (#1460).
    let per_subject: Vec<SubjectNca> =
        crate::api::install_on_engine_pool(|| match model.pk_model {
            PkModel::OneCptOral
            | PkModel::OneCptTransit
            | PkModel::OneCptIg
            | PkModel::TwoCptOral
            | PkModel::TwoCptTransit
            | PkModel::TwoCptIg
            | PkModel::ThreeCptOral => population
                .subjects
                .par_iter()
                .map(nca_one_cpt_oral)
                .collect(),

            PkModel::OneCptIv | PkModel::TwoCptIv | PkModel::ThreeCptIv => {
                population.subjects.par_iter().map(nca_one_cpt_iv).collect()
            }
        });

    // Count how many subjects had valid CL estimates.
    let n_valid = per_subject.iter().filter(|s| s.cl_f.is_some()).count();
    let n_total = per_subject.len();
    if n_valid < n_total {
        warnings.push(format!(
            "inits_from_nca: {}/{} subjects had a valid AUC estimate; others excluded from NCA pooling",
            n_valid, n_total
        ));
    }

    // For 2-cpt/3-cpt models, attempt biexponential peeling on the pooled curve.
    let mut pop = pool_nca(&per_subject);
    match model.pk_model {
        PkModel::TwoCptIv | PkModel::TwoCptOral | PkModel::TwoCptTransit | PkModel::TwoCptIg => {
            try_biexp_peel(model, population, &mut pop, &mut warnings);
        }
        PkModel::ThreeCptIv => {
            try_biexp_peel(model, population, &mut pop, &mut warnings);
            warnings.push(
                "inits_from_nca: 3-cpt distribution parameters from biexponential peeling are unreliable; consider the nca_sweep method".into(),
            );
        }
        PkModel::ThreeCptOral => {
            warnings.push(
                "inits_from_nca: Q/V2/Q2/V3 not estimated for 3-cpt oral; using model defaults — consider the nca_sweep method".into(),
            );
        }
        _ => {}
    }

    (pop, warnings)
}

/// Attempt biexponential peeling on the pooled dose-normalised observations.
/// Updates `pop.vss` (Vss → split into V1+V2 downstream) with the peeled
/// estimates stored as separate fields via an extended PopNca.
/// We store A, α, B, β back through the existing vss field as a proxy and
/// handle conversion in `build_params`.
fn try_biexp_peel(
    _model: &CompiledModel,
    population: &Population,
    pop: &mut PopNca,
    warnings: &mut Vec<String>,
) {
    // Build pooled dose-normalised time-concentration from all subjects.
    let mut all_pairs: Vec<(f64, f64)> = Vec::new();
    for subj in &population.subjects {
        let dose = first_dose_amt(subj);
        if dose <= 0.0 {
            continue;
        }
        let (times, concs) = first_dose_obs(subj);
        for (&t, &c) in times.iter().zip(concs.iter()) {
            if c.is_finite() && c > 0.0 {
                all_pairs.push((t, c / dose));
            }
        }
    }

    if all_pairs.is_empty() {
        return;
    }

    all_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Average concentrations at the same time point.
    let mut pooled: Vec<(f64, f64)> = Vec::new();
    let mut i = 0;
    while i < all_pairs.len() {
        let t = all_pairs[i].0;
        let mut sum_c = 0.0;
        let mut cnt = 0;
        while i < all_pairs.len() && (all_pairs[i].0 - t).abs() < 1e-8 {
            sum_c += all_pairs[i].1;
            cnt += 1;
            i += 1;
        }
        pooled.push((t, sum_c / cnt as f64));
    }

    let times: Vec<f64> = pooled.iter().map(|p| p.0).collect();
    let concs: Vec<f64> = pooled.iter().map(|p| p.1).collect();

    match biexponential_peel(&times, &concs, 3, 3.0) {
        Some((a_cap, alpha, b_cap, beta)) => {
            // Convert macro constants to micro PK parameters.
            // C(0) = A + B, V1 = 1/(A+B) (dose-normalised).
            let c0 = a_cap + b_cap;
            if c0 <= 0.0 {
                return;
            }
            let v1 = 1.0 / c0;
            let k21 = (a_cap * beta + b_cap * alpha) / c0;
            if k21 <= 0.0 {
                return;
            }
            let k10 = alpha * beta / k21;
            let k12 = alpha + beta - k21 - k10;
            if k10 <= 0.0 || k12 <= 0.0 {
                return;
            }
            let cl = k10 * v1;
            let q = k12 * v1;
            let v2 = q / k21;

            // Override the NCA pooled estimates with peeled values.
            pop.cl_f = Some(cl);
            pop.v_f = Some(v1);
            // Store V2 and Q as vss (V1+V2) and lambda_z proxy (Q) temporarily;
            // `build_params` will interpret them via pk_model context.
            // We use two dedicated fields added to PopNca.
            pop.q_peel = Some(q);
            pop.v2_peel = Some(v2);
        }
        None => {
            warnings.push(
                "inits_from_nca: biexponential peeling failed (poor phase separation); Q/V2 will use model defaults — consider the nca_sweep method".into(),
            );
        }
    }
}

/// How [`resolve_theta_for_slot`] answered for one PK slot.
#[derive(Debug, Clone, PartialEq)]
enum SlotTheta {
    /// The θ index the slot's estimate is written to.
    Theta(usize),
    /// More than one η qualifies for the owning parameter and they carry
    /// different θ, `(η, θ)` pairs in η-name order. Nothing is written: picking
    /// one would be a guess (#1930).
    Ambiguous(Vec<(String, String)>),
    /// The owning parameter's mu-reference is anchored to something that is not
    /// a `[parameters]` θ — a `[covariate_nn]` output such as `TYPICAL_PK.CL` —
    /// named here. There is no θ to write, and the name fallbacks must not look
    /// for one: they would find an unrelated parameter's (#1930 review).
    NotATheta(String),
    /// No individual parameter owns the slot — the `pk(...)` role is bound to a
    /// literal or to `TIME`, or the model never binds it — so there is no
    /// estimate to route (#1930).
    Unowned,
    /// No η decided the slot and two θ names match it equally well as a whole
    /// word ([`best_theta_name_match`]), in declaration order. Nothing is
    /// written: falling through to a weaker layer would only guess (#1922).
    AmbiguousName(Vec<String>),
    /// The slot has an owner but no θ could be associated with it.
    Unmapped,
}

/// The η whose mu-reference drives the parameter `indiv_name`, as the sorted
/// list of `(η, θ index)` candidates — exactly one when the answer is
/// unambiguous. The θ index is `None` when the mu-reference's anchor is not a
/// declared θ (an NN output such as `TYPICAL_PK.CL`): such an η is still a
/// candidate, because it is still the answer — only not a writable one.
///
/// 1. **Association**: the η whose mu-reference the parser detected on
///    `indiv_name`'s own assignment (`IndivParamPartials::mu_ref_owners`). This
///    is exact and needs no naming convention: `CL = TVCL * exp(ETA_CL)` owns
///    `ETA_CL` whatever an unrelated `S = THS * exp(IIV_CL)` is called (#1930).
///    An owned η ends the search whatever its anchor is: dropping an NN-anchored
///    one here let the name rule below hand `CL` to `IIV_CL` and `THS`.
/// 2. **Name rule**, only when no mu-reference was detected on the parameter
///    itself (e.g. it reads an η-carrying intermediate): an η named exactly the
///    parameter or ending in `_<PARAM>` (case-insensitive), so `ETA_VMAX` does
///    not match `V`. Every qualifying η is returned, not the first one: `mu_refs`
///    is a `HashMap`, and "first" was its per-process hash order (#1930). An η
///    whose mu-reference another parameter owns qualifies only if `indiv_name`
///    actually depends on it ([`name_rule_may_use`]).
fn mu_ref_etas_for_param(model: &CompiledModel, indiv_name: &str) -> Vec<(String, Option<usize>)> {
    let theta_names = &model.default_params.theta_names;
    let candidate = |eta: &String| -> Option<(String, Option<usize>)> {
        let mu_ref = model.mu_refs.get(eta)?;
        let idx = theta_names.iter().position(|n| n == &mu_ref.theta_name);
        Some((eta.clone(), idx))
    };
    let owned: Vec<(String, Option<usize>)> = model
        .indiv_param_partials
        .mu_ref_owners
        .iter()
        .filter(|(_, owner)| owner.as_str() == indiv_name)
        .filter_map(|(eta, _)| candidate(eta))
        .collect();
    if !owned.is_empty() {
        return owned;
    }
    let indiv_upper = indiv_name.to_ascii_uppercase();
    let suffix = format!("_{indiv_upper}");
    let mut by_name: Vec<(String, Option<usize>)> = model
        .mu_refs
        .keys()
        .filter(|eta| {
            let u = eta.to_ascii_uppercase();
            u == indiv_upper || u.ends_with(&suffix)
        })
        .filter(|eta| name_rule_may_use(model, indiv_name, eta))
        .filter_map(candidate)
        .collect();
    by_name.sort();
    by_name
}

/// Whether layer 1's name rule may hand `eta` to parameter `param`. An η whose
/// mu-reference was detected on **another** parameter is that parameter's unless
/// `param` reads it: `CL = CLI` with `CLI = TVCL * exp(ETA_CL)` depends on
/// `ETA_CL`, but `CL = TVCL` (no IIV) next to `S = THS * exp(ETA_CL)` does not,
/// and the name rule used to send the clearance onto `THS` and key CL's omega to
/// `ETA_CL` (#1922 review r3). Dependence is read off the parse-time partial
/// `∂param/∂eta`, which folds to the literal `0` when `param` does not reach
/// `eta`. When that dependence cannot be read — a parameter with no partials
/// row, as an `if`-assigned one has none — the η is declined: it is known to be
/// another parameter's, and borrowing it on an unknown dependence sent a
/// conditional `CL`'s clearance onto `THS` the same way (#1922 review r5).
fn name_rule_may_use(model: &CompiledModel, param: &str, eta: &str) -> bool {
    let partials = &model.indiv_param_partials;
    match partials.mu_ref_owners.get(eta) {
        None => return true,
        Some(owner) if owner == param => return true,
        Some(_) => {}
    }
    let Some(row) = partials.names.iter().position(|n| n == param) else {
        return false;
    };
    let Some(k) = model.eta_names.iter().position(|n| n == eta) else {
        return false;
    };
    match partials.d_d_eta.get(row).and_then(|r| r.get(k)) {
        Some(crate::parser::model_parser::Expression::Literal(v)) => *v != 0.0,
        Some(_) => true,
        None => false,
    }
}

/// Find which theta index drives a given PK slot.
///
/// Three-layer lookup (model-agnostic, works with or without mu-referencing),
/// all of it keyed on the slot's **owning** individual parameter
/// ([`CompiledModel::indiv_param_for_pk_slot`]). Not a `pk_indices` scan: an
/// analytical model gives every unbound intermediate the placeholder slot `0`,
/// so `CLWT = exp(CLWT_EXP * log(WT/70))` declared above `CL` answered for
/// `PK_IDX_CL` and layer 2 then wrote NCA's clearance onto the allometric
/// exponent `CLWT_EXP` (#1899). A slot with no owner — a role bound to a literal
/// (`cl=1.0`) or to `TIME` — is [`SlotTheta::Unowned`]: there is nothing for NCA
/// to estimate, and searching every θ for the canonical name wrote the pooled
/// clearance onto a readout coefficient called `THCL_A` (#1930).
///
/// 1. Mu-reference: the θ of the owner's η ([`mu_ref_etas_for_param`]). Two
///    qualifying η carrying different anchors are [`SlotTheta::Ambiguous`], and
///    one anchored to a non-θ is [`SlotTheta::NotATheta`]; both stop the lookup
///    — the name fallbacks below would only guess again.
/// 2. Name-based fallback: the θ whose name matches the parameter's name **as a
///    whole word** ([`theta_name_match_rank`]) — `CLEARANCE → CLEARANCE`,
///    `TVCL → CL`, `THETA_CL → CL` — never a bare substring, so `TVCL` is not a
///    match for `V` (#1922). Two θ tied at the best rank are
///    [`SlotTheta::AmbiguousName`], which stops the lookup.
/// 3. Canonical-name fallback: the same whole-word search for the standard slot
///    name ("CL", "V", "Q", etc.).
fn resolve_theta_for_slot(model: &CompiledModel, pk_slot: usize) -> SlotTheta {
    // Canonical names for each slot (used only as last-resort fallback).
    let canonical = if pk_slot == PK_IDX_CL {
        "CL"
    } else if pk_slot == PK_IDX_V {
        "V"
    } else if pk_slot == PK_IDX_Q {
        "Q"
    } else if pk_slot == PK_IDX_V2 {
        "V2"
    } else if pk_slot == PK_IDX_KA {
        "KA"
    } else if pk_slot == PK_IDX_Q3 {
        "Q3"
    } else if pk_slot == PK_IDX_V3 {
        "V3"
    } else if pk_slot == PK_IDX_LAGTIME {
        "LAGTIME"
    } else if pk_slot == PK_IDX_F {
        "F"
    } else if pk_slot == PK_IDX_N {
        "N"
    } else if pk_slot == PK_IDX_MTT {
        "MTT"
    } else if pk_slot == PK_IDX_MAT {
        "MAT"
    } else if pk_slot == PK_IDX_CV2 {
        "CV2"
    } else {
        return SlotTheta::Unmapped;
    };

    let params = &model.default_params;
    let Some(indiv_pos) = model.indiv_param_for_pk_slot(pk_slot) else {
        return SlotTheta::Unowned;
    };
    let indiv_name = &model.indiv_param_names[indiv_pos];

    // Layer 1: owning individual parameter → its mu-referenced η → θ.
    // Candidates are compared by anchor *name*, so two η on one NN output agree
    // and an NN-anchored η disagrees with a θ-anchored one.
    let candidates: Vec<(String, String, Option<usize>)> = mu_ref_etas_for_param(model, indiv_name)
        .into_iter()
        .map(|(eta, idx)| {
            let anchor = model.mu_refs[&eta].theta_name.clone();
            (eta, anchor, idx)
        })
        .collect();
    let mut anchors: Vec<(&str, Option<usize>)> = candidates
        .iter()
        .map(|(_, a, idx)| (a.as_str(), *idx))
        .collect();
    anchors.sort_unstable();
    anchors.dedup();
    match anchors.as_slice() {
        [] => {}
        [(_, Some(idx))] => return SlotTheta::Theta(*idx),
        [(anchor, None)] => return SlotTheta::NotATheta(anchor.to_string()),
        _ => {
            return SlotTheta::Ambiguous(
                candidates
                    .into_iter()
                    .map(|(eta, anchor, _)| (eta, anchor))
                    .collect(),
            )
        }
    }

    // Layer 2: theta_names matching indiv_param_name as a whole word. A tie stops
    // the lookup rather than handing it to the weaker canonical search.
    let names = &params.theta_names;
    let tied = |idxs: Vec<usize>| {
        SlotTheta::AmbiguousName(idxs.into_iter().map(|i| names[i].clone()).collect())
    };
    match best_theta_name_match(names, indiv_name) {
        NameMatch::Found(idx) => return SlotTheta::Theta(idx),
        NameMatch::Tied(idxs) => return tied(idxs),
        NameMatch::Missing => {}
    }

    // Layer 3: theta_names matching the canonical slot name as a whole word.
    match best_theta_name_match(names, canonical) {
        NameMatch::Found(idx) => SlotTheta::Theta(idx),
        NameMatch::Tied(idxs) => tied(idxs),
        NameMatch::Missing => SlotTheta::Unmapped,
    }
}

/// How well θ name `theta` matches parameter (or canonical slot) name `param`,
/// both compared uppercased; `None` when it does not match at all.
///
/// * `0` — the whole name: `param` itself, or `param` behind one typical-value
///   prefix (`TV`, `TH`, `THETA`, each optionally followed by `_`): `TVCL`,
///   `THETA_CL`, `CLEARANCE` for `CLEARANCE`.
/// * `1` — one `_`-separated segment of the name, under the same rule:
///   `POP_CL`, `CL_POP`, `THCL_A`.
///
/// A bare substring is never a match: `TVCL` contains `V` but is not a word `V`
/// (#1922), and `CLWT_EXP` contains `CL` but no segment of it is `CL` (#1899).
fn theta_name_match_rank(theta: &str, param: &str) -> Option<u8> {
    let theta = theta.to_ascii_uppercase();
    let param = param.to_ascii_uppercase();
    if param.is_empty() {
        return None;
    }
    let is_word = |word: &str| {
        ["", "TV", "TV_", "TH", "TH_", "THETA", "THETA_"]
            .iter()
            .any(|prefix| word.strip_prefix(prefix) == Some(param.as_str()))
    };
    if is_word(&theta) {
        Some(0)
    } else if theta.split('_').any(is_word) {
        Some(1)
    } else {
        None
    }
}

/// The outcome of a whole-word θ-name search ([`best_theta_name_match`]).
#[derive(Debug, Clone, PartialEq, Eq)]
enum NameMatch {
    Found(usize),
    /// Two or more θ indices tie at the best rank, in declaration order.
    Tied(Vec<usize>),
    Missing,
}

/// The θ index whose name matches `param` best ([`theta_name_match_rank`]).
fn best_theta_name_match(theta_names: &[String], param: &str) -> NameMatch {
    let ranked: Vec<(u8, usize)> = theta_names
        .iter()
        .enumerate()
        .filter_map(|(i, n)| theta_name_match_rank(n, param).map(|r| (r, i)))
        .collect();
    let Some(best) = ranked.iter().map(|&(r, _)| r).min() else {
        return NameMatch::Missing;
    };
    let top: Vec<usize> = ranked
        .into_iter()
        .filter(|&(r, _)| r == best)
        .map(|(_, i)| i)
        .collect();
    match top.as_slice() {
        [idx] => NameMatch::Found(*idx),
        _ => NameMatch::Tied(top),
    }
}

/// The θ index a PK slot's NCA estimate is written to, or `None` when
/// [`resolve_theta_for_slot`] declines (ambiguous, not a θ, unowned or unmapped).
pub(crate) fn find_theta_for_slot(model: &CompiledModel, pk_slot: usize) -> Option<usize> {
    match resolve_theta_for_slot(model, pk_slot) {
        SlotTheta::Theta(idx) => Some(idx),
        SlotTheta::Ambiguous(_)
        | SlotTheta::AmbiguousName(_)
        | SlotTheta::NotATheta(_)
        | SlotTheta::Unowned
        | SlotTheta::Unmapped => None,
    }
}

/// Find the omega diagonal index for the eta driving a given PK slot.
///
/// Resolves the owning individual parameter through
/// [`CompiledModel::indiv_param_for_pk_slot`] for the same reason
/// [`find_theta_for_slot`] does: a `pk_indices` scan answers an unbound
/// intermediate for `PK_IDX_CL` (#1899), and its name then matches no eta, so the
/// slot's omega came back `None` and NCA left `ETA_CL`'s variance at the default.
/// The η is the one [`mu_ref_etas_for_param`] names, and only when it names
/// exactly one: two candidates return `None` rather than the first in hash order
/// (#1930).
fn find_omega_idx_for_slot(model: &CompiledModel, pk_slot: usize) -> Option<usize> {
    let indiv_pos = model.indiv_param_for_pk_slot(pk_slot)?;
    let candidates = mu_ref_etas_for_param(model, &model.indiv_param_names[indiv_pos]);
    let [(eta, _)] = candidates.as_slice() else {
        return None;
    };
    model
        .default_params
        .omega
        .eta_names
        .iter()
        .position(|n| n == eta)
}

/// The warning for a slot [`resolve_theta_for_slot`] declined, or `None` for a
/// slot it resolved. `fallback` is the message for [`SlotTheta::Unmapped`];
/// `None` there keeps a slot silent that the model need not have at all (Q, V2,
/// lag time).
fn slot_routing_warning(
    model: &CompiledModel,
    pk_slot: usize,
    label: &str,
    fallback: Option<&str>,
) -> Option<String> {
    match resolve_theta_for_slot(model, pk_slot) {
        SlotTheta::Theta(_) => None,
        SlotTheta::Ambiguous(pairs) => {
            let list = pairs
                .iter()
                .map(|(eta, theta)| format!("{eta} -> {theta}"))
                .collect::<Vec<_>>()
                .join(", ");
            Some(format!(
                "inits_from_nca: {label} not written — more than one eta qualifies for its \
                 parameter and they reference different thetas ({list}); using model \
                 default. Keep one eta named after the parameter."
            ))
        }
        SlotTheta::AmbiguousName(thetas) => Some(format!(
            "inits_from_nca: {label} not written — its parameter carries no \
             mu-referenced eta and the thetas {} match its name equally well; using \
             model default. Mu-reference the parameter or rename one of them.",
            thetas.join(", ")
        )),
        SlotTheta::NotATheta(anchor) => fallback.map(|_| {
            format!(
                "inits_from_nca: {label} not written — its typical value is `{anchor}`, \
                 not a [parameters] theta; nothing to route the estimate to"
            )
        }),
        SlotTheta::Unowned => fallback.map(|_| {
            format!(
                "inits_from_nca: {label} is not bound to an individual parameter on the \
                 [structural_model] line (a constant or TIME); nothing to estimate"
            )
        }),
        SlotTheta::Unmapped => fallback.map(str::to_string),
    }
}

/// The label a PK role carries in `inits_from_nca`'s warnings.
fn pk_slot_label(pk_slot: usize) -> &'static str {
    match pk_slot {
        PK_IDX_CL => "CL",
        PK_IDX_V => "V/V1",
        PK_IDX_Q => "Q",
        PK_IDX_V2 => "V2",
        PK_IDX_KA => "Ka",
        PK_IDX_F => "F",
        PK_IDX_Q3 => "Q3",
        PK_IDX_V3 => "V3",
        PK_IDX_LAGTIME => "LAGTIME",
        PK_IDX_N => "N",
        PK_IDX_MTT => "MTT",
        PK_IDX_MAT => "MAT",
        PK_IDX_CV2 => "CV2",
        _ => "?",
    }
}

/// Every PK role's θ after the collision guard ([`guard_slot_thetas`]).
struct GuardedSlots {
    /// `(slot, θ index)` per role the model reads; `None` when the role resolved
    /// to nothing or shares its θ with another role.
    thetas: Vec<(usize, Option<usize>)>,
    /// One entry per θ two or more roles resolved to: their labels and the θ.
    shared: Vec<(Vec<&'static str>, usize)>,
}

impl GuardedSlots {
    /// The θ a role's estimate may be written to.
    fn theta(&self, pk_slot: usize) -> Option<usize> {
        self.thetas
            .iter()
            .find(|&&(slot, _)| slot == pk_slot)
            .and_then(|&(_, idx)| idx)
    }

    /// Whether θ `idx` is shared by two roles, so keeps its model default — in
    /// the NCA stage and in every sweep after it.
    fn is_shared(&self, idx: usize) -> bool {
        self.shared.iter().any(|&(_, t)| t == idx)
    }

    fn warnings(&self, model: &CompiledModel) -> Vec<String> {
        self.shared
            .iter()
            .map(|(labels, idx)| {
                let theta = &model.default_params.theta_names[*idx];
                format!(
                    "inits_from_nca: {} all resolve to theta {theta}; none of them is \
                     written, so {theta} keeps its model default",
                    labels.join(", "),
                )
            })
            .collect()
    }
}

/// Resolve every PK role the model reads and clear each whose θ index another
/// role also resolved to.
///
/// `build_params` writes each pooled estimate in turn, so two roles sharing one
/// index would leave the later estimate on that θ and the earlier one nowhere —
/// a volume silently written onto `TVCL` (#1922). A mis-resolution must degrade
/// to "left at the model default", so neither role is written. Every role the
/// model reads takes part ([`PkModel::consumes_pk_slot`]), including those NCA
/// never writes (F, Q3, V3, and the transit / inverse-Gaussian absorption
/// parameters): a pooled clearance written onto the θ that F or `n` also
/// resolves to would move that role too. The shared θ also stays out of the
/// sweeps that follow ([`GuardedSlots::is_shared`]), or the default the warning
/// promises would last only until the sweep.
fn guard_slot_thetas(model: &CompiledModel) -> GuardedSlots {
    let resolved: Vec<(usize, Option<usize>)> = (0..=PK_IDX_CV2)
        .filter(|&slot| model.pk_model.consumes_pk_slot(slot))
        .map(|slot| (slot, find_theta_for_slot(model, slot)))
        .collect();
    let mut shared: Vec<(Vec<&'static str>, usize)> = Vec::new();
    for &(_, idx) in &resolved {
        let Some(idx) = idx else { continue };
        if shared.iter().any(|&(_, t)| t == idx) {
            continue;
        }
        let labels: Vec<&'static str> = resolved
            .iter()
            .filter(|&&(_, other)| other == Some(idx))
            .map(|&(slot, _)| pk_slot_label(slot))
            .collect();
        if labels.len() > 1 {
            shared.push((labels, idx));
        }
    }
    let thetas = resolved
        .into_iter()
        .map(|(slot, idx)| (slot, idx.filter(|i| !shared.iter().any(|&(_, t)| t == *i))))
        .collect();
    GuardedSlots { thetas, shared }
}

/// Route pooled NCA estimates to theta slots and return updated ModelParameters.
fn build_params(
    model: &CompiledModel,
    pop: &PopNca,
    warnings: &mut Vec<String>,
) -> ModelParameters {
    let mut params = model.default_params.clone();

    // Resolve all theta indices up-front (no closures capturing params).
    let guarded = guard_slot_thetas(model);
    warnings.extend(guarded.warnings(model));
    let cl_idx = guarded.theta(PK_IDX_CL);
    let v_idx = guarded.theta(PK_IDX_V);
    let ka_idx = guarded.theta(PK_IDX_KA);
    let q_idx = guarded.theta(PK_IDX_Q);
    let v2_idx = guarded.theta(PK_IDX_V2);
    let lagtime_idx = guarded.theta(PK_IDX_LAGTIME);
    // F is never written, only read for the bioavailability scale below, so it
    // is resolved outside the guard: a collision stops the writes to F's θ
    // without also dropping the F_default that rescales a separate V/F onto
    // true V.
    let f_idx = find_theta_for_slot(model, PK_IDX_F);
    let cl_eta_omega_idx = find_omega_idx_for_slot(model, PK_IDX_CL);

    // Bioavailability correction: NCA estimates are apparent (CL/F, V/F). When the
    // model carries an explicit free F theta, multiply back by F_default so that the
    // written TVCL/TVV are in the model's true-CL/true-V space.  F itself cannot be
    // estimated from oral-only NCA, so it stays at its model default; Option B sweeps
    // it via 1D rRMSE search.
    let f_scale = match f_idx {
        Some(fi) if !params.theta_fixed[fi] => {
            let fval = params.theta[fi]; // model default F
            if fval > 0.0 && fval < 1.0 {
                warnings.push(format!(
                    "inits_from_nca: bioavailability F theta found (default={fval:.3}); \
                     NCA CL/V scaled by F_default. F cannot be estimated from oral NCA alone \
                     — Option B will sweep it via rRMSE."
                ));
                fval
            } else {
                1.0
            }
        }
        _ => 1.0,
    };

    // Helper: write one theta with bounds-clamping.
    let write_theta = |params: &mut ModelParameters,
                       idx: usize,
                       value: f64,
                       param_name: &str,
                       warnings: &mut Vec<String>| {
        if params.theta_fixed[idx] {
            return;
        }
        let clamped = value.clamp(params.theta_lower[idx], params.theta_upper[idx]);
        if (clamped - value).abs() > 1e-10 {
            warnings.push(format!(
                "inits_from_nca: {param_name} estimate {value:.4} clamped to bounds [{:.4}, {:.4}]",
                params.theta_lower[idx], params.theta_upper[idx]
            ));
        }
        params.theta[idx] = clamped;
    };

    // Scaling correction for Option A: NCA uses raw observations directly, so if
    // model.scaling = ScalarScale(k) the predictions are divided by k before being
    // compared to observations. That means the model's CL is k× the NCA CL derived
    // from raw obs. Divide NCA CL and V by k to put them in the model's parameter space.
    // ExpressionScale / PerCmt are too complex to correct automatically — warn instead.
    let scale_factor = match &model.scaling {
        crate::types::ScalingSpec::ScalarScale(k) if *k > 0.0 => {
            warnings.push(format!(
                "inits_from_nca: obs_scale = {k} detected; NCA CL/V estimates divided by {k} to match model parameter space"
            ));
            *k
        }
        crate::types::ScalingSpec::ExpressionScale { .. }
        | crate::types::ScalingSpec::PerCmt(_) => {
            warnings.push(
                "inits_from_nca: expression/per-compartment obs_scale detected; NCA CL/V estimates may be in wrong units — recommend the nca_sweep method".into(),
            );
            1.0
        }
        _ => 1.0,
    };

    // Write CL / CL_F.  When an explicit F theta is present, NCA gives apparent
    // CL/F; multiply by f_scale (= F_default) to recover the true CL.
    if let Some(cl) = pop.cl_f {
        if let Some(idx) = cl_idx {
            write_theta(
                &mut params,
                idx,
                cl * f_scale / scale_factor,
                "CL",
                warnings,
            );
        } else {
            warnings.extend(slot_routing_warning(
                model,
                PK_IDX_CL,
                "CL",
                Some("inits_from_nca: could not map CL to a theta (no mu-referencing for CL eta); using model default"),
            ));
        }
    } else {
        warnings.push("inits_from_nca: CL not estimable from NCA; using model default".into());
    }

    // Write V / V1.  Same f_scale correction as CL.
    if let Some(v) = pop.v_f {
        if let Some(idx) = v_idx {
            write_theta(
                &mut params,
                idx,
                v * f_scale / scale_factor,
                "V/V1",
                warnings,
            );
        } else {
            warnings.extend(slot_routing_warning(
                model,
                PK_IDX_V,
                "V/V1",
                Some("inits_from_nca: could not map V/V1 to a theta; using model default"),
            ));
        }
    } else {
        warnings.push("inits_from_nca: V/V1 not estimable from NCA; using model default".into());
    }

    // Write Ka (oral models)
    match model.pk_model {
        PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral => {
            if let Some(ka) = pop.ka {
                if let Some(idx) = ka_idx {
                    write_theta(&mut params, idx, ka, "Ka", warnings);
                } else {
                    warnings.extend(slot_routing_warning(
                        model,
                        PK_IDX_KA,
                        "Ka",
                        Some("inits_from_nca: could not map Ka to a theta; using model default"),
                    ));
                }
            } else {
                warnings
                    .push("inits_from_nca: Ka not estimable from NCA; using model default".into());
            }
        }
        _ => {}
    }

    // Write lag time (oral models only, incl. transit / IG which are oral and accept lagtime=)
    match model.pk_model {
        PkModel::OneCptOral
        | PkModel::TwoCptOral
        | PkModel::ThreeCptOral
        | PkModel::OneCptTransit
        | PkModel::TwoCptTransit
        | PkModel::OneCptIg
        | PkModel::TwoCptIg => {
            if let Some(tlag) = pop.tlag {
                if let Some(idx) = lagtime_idx {
                    write_theta(&mut params, idx, tlag, "LAGTIME", warnings);
                } else {
                    warnings.extend(slot_routing_warning(model, PK_IDX_LAGTIME, "LAGTIME", None));
                }
                // No warning if idx is None: model may not have a lagtime theta at all.
            }
            // No warning if tlag is None: no detectable lag in the data is fine.
        }
        _ => {}
    }

    // Write Q and V2 from biexponential peeling
    if let Some(q) = pop.q_peel {
        if let Some(idx) = q_idx {
            write_theta(&mut params, idx, q, "Q", warnings);
        } else {
            warnings.extend(slot_routing_warning(model, PK_IDX_Q, "Q", None));
        }
    }
    if let Some(v2) = pop.v2_peel {
        if let Some(idx) = v2_idx {
            write_theta(&mut params, idx, v2, "V2", warnings);
        } else {
            warnings.extend(slot_routing_warning(model, PK_IDX_V2, "V2", None));
        }
    }

    // Update omega for CL eta if we have inter-subject CV².
    // Update the single diagonal element in-place so that block_omega off-diagonal
    // entries (correlated etas) are not discarded.
    if let Some(cv2) = pop.cl_cv2 {
        if let Some(cl_eta_idx) = cl_eta_omega_idx {
            let omega_var = cv2.clamp(0.01, 2.0);
            let mut m = params.omega.matrix.clone();
            m[(cl_eta_idx, cl_eta_idx)] = omega_var;
            let names = params.omega.eta_names.clone();
            let diagonal = params.omega.diagonal;
            let mask = params.omega.free_mask.clone();
            params.omega = OmegaMatrix::from_matrix_with_mask(m, names, diagonal, mask);
        }
    }

    params
}

fn empty_pop_nca() -> PopNca {
    PopNca {
        cl_f: None,
        v_f: None,
        vss: None,
        lambda_z: None,
        ka: None,
        c0: None,
        tmax_median: 0.0,
        cl_cv2: None,
        tlag: None,
        q_peel: None,
        v2_peel: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::datareader::read_nonmem_csv;
    use crate::parser::model_parser::parse_model_file;
    use std::path::Path;

    #[test]
    fn test_suggest_start_empty_population() {
        let model = parse_model_file(Path::new("examples/warfarin.ferx")).unwrap();
        let empty_pop = Population {
            subjects: vec![],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let result = inits_from_nca(&model, &empty_pop, NcaInit::Nca).unwrap();
        // Must not panic and should warn.
        assert!(!result.warnings.is_empty());
        // Params should equal model defaults.
        assert_eq!(result.params.theta, model.default_params.theta);
    }

    #[test]
    fn test_suggest_start_iv_model_uses_the_iv_nca_route() {
        // The oral and IV routes are two different `par_iter` arms of `run_nca`, and every
        // other test here is oral (warfarin), so without this one the IV arm is never
        // entered — a change to it, or to the pool the pass runs on (#1460), would be
        // invisible.
        let model = parse_model_file(Path::new("examples/one_cpt_iv.ferx")).unwrap();
        assert!(
            matches!(
                model.pk_model,
                PkModel::OneCptIv | PkModel::TwoCptIv | PkModel::ThreeCptIv
            ),
            "fixture must take the IV arm, got {:?}",
            model.pk_model
        );
        let population = read_nonmem_csv(Path::new("data/one_cpt_iv.csv"), None, None).unwrap();
        let result = inits_from_nca(&model, &population, NcaInit::Nca).unwrap();
        for (i, &theta) in result.params.theta.iter().enumerate() {
            let lo = result.params.theta_lower[i];
            let hi = result.params.theta_upper[i];
            assert!(
                theta.is_finite() && theta >= lo && theta <= hi,
                "theta[{i}] = {theta} outside bounds [{lo}, {hi}]"
            );
        }
        // An IV dataset has concentrations and doses, so NCA has something to work with:
        // the suggestion must actually move off the model's defaults.
        assert_ne!(
            result.params.theta, model.default_params.theta,
            "NCA returned the model defaults on a dataset it can analyse"
        );
    }

    #[test]
    fn test_suggest_start_respects_bounds() {
        let model = parse_model_file(Path::new("examples/warfarin.ferx")).unwrap();
        let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).unwrap();
        let result = inits_from_nca(&model, &population, NcaInit::Nca).unwrap();
        for (i, &theta) in result.params.theta.iter().enumerate() {
            let lo = result.params.theta_lower[i];
            let hi = result.params.theta_upper[i];
            assert!(
                theta >= lo && theta <= hi,
                "theta[{i}] = {theta} outside bounds [{lo}, {hi}]"
            );
        }
    }

    #[test]
    fn test_suggest_start_fixed_thetas_unchanged() {
        let mut model = parse_model_file(Path::new("examples/warfarin.ferx")).unwrap();
        // Fix the first theta.
        model.default_params.theta_fixed[0] = true;
        let original_val = model.default_params.theta[0];
        let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).unwrap();
        let result = inits_from_nca(&model, &population, NcaInit::Nca).unwrap();
        assert_eq!(
            result.params.theta[0], original_val,
            "fixed theta must not be overwritten"
        );
    }

    // The early returns of `nca_with_sweep` / `nca_with_ebe` below return `Ok` since
    // #898, and each is reached only by one fixture shape. The Tier-2 `tests/suggest_start.rs`
    // covers the ODE fallback and the full sweep, not these.

    /// Two-compartment covariate model with every PK theta fixed: the only unwritten
    /// thetas left are `THETA_WT` / `THETA_CRCL`, which both sweeps exclude as covariates.
    fn two_cpt_cov_only_covariates_free() -> (CompiledModel, Population) {
        let mut model = parse_model_file(Path::new("examples/two_cpt_oral_cov.ferx")).unwrap();
        for (i, name) in model.default_params.theta_names.iter().enumerate() {
            model.default_params.theta_fixed[i] = !name.starts_with("THETA_");
        }
        let population =
            read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None).unwrap();
        (model, population)
    }

    #[test]
    fn nca_sweep_returns_ok_when_only_covariate_thetas_are_unwritten() {
        let (model, population) = two_cpt_cov_only_covariates_free();
        let result = inits_from_nca(&model, &population, NcaInit::Sweep).unwrap();
        assert!(
            result.warnings.iter().any(|w| w.contains(
                "inits_from_nca (nca_sweep): excluded 2 covariate theta(s) from rRMSE sweep"
            )),
            "both covariate thetas must be excluded, got: {:?}",
            result.warnings
        );
        assert_eq!(
            result.params.theta, model.default_params.theta,
            "nothing is left to sweep, so every theta keeps its default"
        );
    }

    #[test]
    fn nca_ebe_returns_ok_when_only_covariate_thetas_are_unwritten() {
        let (model, population) = two_cpt_cov_only_covariates_free();
        let result = inits_from_nca(&model, &population, NcaInit::Ebe).unwrap();
        assert!(
            result.warnings.iter().any(|w| w.contains(
                "inits_from_nca (nca_ebe): excluded 2 covariate theta(s) from rRMSE sweep"
            )),
            "both covariate thetas must be excluded, got: {:?}",
            result.warnings
        );
        assert_eq!(result.params.theta, model.default_params.theta);
    }

    #[test]
    fn nca_ebe_returns_ok_when_every_theta_is_fixed() {
        let mut model = parse_model_file(Path::new("examples/warfarin.ferx")).unwrap();
        model
            .default_params
            .theta_fixed
            .iter_mut()
            .for_each(|f| *f = true);
        let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).unwrap();
        let result = inits_from_nca(&model, &population, NcaInit::Ebe).unwrap();
        assert_eq!(result.params.theta, model.default_params.theta);
        assert!(
            !result
                .warnings
                .iter()
                .any(|w| w.contains("covariate theta")),
            "no theta reaches the covariate filter, got: {:?}",
            result.warnings
        );
    }

    #[test]
    fn nca_ebe_sweeps_a_logit_theta_on_the_etas_zero_path() {
        // A logit-normal theta is routed to the etas=0 `sweep_unwritten_thetas`, not the
        // EBE sweep. Fixing the lognormal thetas leaves it the only one to sweep, so the
        // fit reaches `nca_with_ebe`'s final `Ok` through that arm alone.
        let mut model = parse_model_file(Path::new("examples/warfarin_logit_f.ferx")).unwrap();
        let f_idx = model
            .default_params
            .theta_names
            .iter()
            .position(|n| n == "THETA_F")
            .unwrap();
        for (i, fixed) in model.default_params.theta_fixed.iter_mut().enumerate() {
            *fixed = i != f_idx;
        }
        // Start F well below the data-generating 0.80 so the sweep has a direction to move.
        model.default_params.theta[f_idx] = 0.2;
        let population =
            read_nonmem_csv(Path::new("data/warfarin_logit_f.csv"), None, None).unwrap();
        let result = inits_from_nca(&model, &population, NcaInit::Ebe).unwrap();
        let (f, lo, hi) = (
            result.params.theta[f_idx],
            result.params.theta_lower[f_idx],
            result.params.theta_upper[f_idx],
        );
        assert!(
            f.is_finite() && f >= lo && f <= hi,
            "THETA_F = {f} outside [{lo}, {hi}]"
        );
        assert_ne!(
            f, model.default_params.theta[f_idx],
            "the logit sweep must move THETA_F off its default"
        );
    }

    /// The #1899 fixture: a one-cpt oral model whose `CL` reads an unbound
    /// allometric intermediate `CLWT`, which therefore has to be declared above
    /// it. `indiv` supplies the two `[individual_parameters]` lines so the
    /// declaration order can be flipped.
    ///
    /// Every parameter carries its own η, which matters: `find_theta_for_slot`'s
    /// layer 1 resolves through `mu_refs`, and only a parameter with no η falls
    /// through to the layer-2 substring search — where `TVCL` *contains* `V`, so a
    /// bare `V = TVV` resolves to `TVCL` and `build_params` then writes CL's
    /// estimate and V's to the same θ. That is a separate, pre-existing weakness
    /// of the substring fallback (reported separately), and a fixture that trips it
    /// cannot show the placeholder fix reaching the user-visible θ at all (review
    /// r1 #1).
    fn allometric_cl_model(indiv: &str) -> String {
        format!(
            "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  \
             theta CLWT_EXP(0.75, 0.0, 2.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
             theta TVKA(1.5, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  \
             omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.09\n  \
             sigma PROP_ERR ~ 0.02 (sd)\n\
             [individual_parameters]\n{indiv}  V = TVV * exp(ETA_V)\n  \
             KA = TVKA * exp(ETA_KA)\n\
             [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n"
        )
    }

    /// The issue's declaration order: `CL` reads `CLWT`, so the intermediate
    /// *must* come first — the broken order, and the one a covariate model is
    /// written in.
    fn allometric_cl_intermediate_first() -> String {
        allometric_cl_model(
            "  CLWT = exp(CLWT_EXP * log(WT/70))\n  CL = TVCL * CLWT * exp(ETA_CL)\n",
        )
    }

    /// #1899: an unbound intermediate declared **before** `CL` carries the
    /// placeholder PK slot `0` — `CL`'s slot — so the old
    /// `pk_indices.iter().position(|&s| s == PK_IDX_CL)` scan answered `CLWT` for
    /// the clearance slot. Layer 2 of `find_theta_for_slot` then matched the only
    /// θ whose name contains `CLWT`, and NCA wrote its pooled CL/F estimate onto
    /// the allometric exponent `CLWT_EXP`; `find_omega_idx_for_slot` matched no η
    /// against the name `CLWT` and left `ETA_CL`'s variance at the model default.
    ///
    /// Both declaration orders are asserted: the lookup must be
    /// order-independent, and the fix is a slot lookup that cannot return a
    /// placeholder, not a reordering. Every other PK slot is asserted in the same
    /// loop, so a fix that moved the collision somewhere else rather than removing
    /// it reddens here.
    #[test]
    fn nca_resolves_cl_past_an_unbound_intermediate_in_either_order() {
        // The mirror order, where `position` happened to find `CL` first and the
        // answer was right by accident. It has to be pinned too: it is the half
        // that a "reorder the block" fix would have claimed to solve, and a
        // regression here would be invisible on the fixture above.
        let after =
            allometric_cl_model("  CL = TVCL * exp(ETA_CL)\n  CLWT = exp(CLWT_EXP * log(WT/70))\n");
        for (label, src) in [
            ("CLWT first", allometric_cl_intermediate_first()),
            ("CLWT last", after),
        ] {
            let m = crate::parser::model_parser::parse_model_string(&src)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            // The precondition: `CLWT` really does carry CL's slot, so the old scan
            // could not tell the two apart.
            let clwt_pos = m
                .indiv_param_names
                .iter()
                .position(|n| n == "CLWT")
                .unwrap();
            assert_eq!(
                m.pk_indices[clwt_pos], PK_IDX_CL,
                "{label}: CLWT must still carry the placeholder slot"
            );
            let theta_idx = find_theta_for_slot(&m, PK_IDX_CL)
                .unwrap_or_else(|| panic!("{label}: no theta for PK_IDX_CL"));
            assert_eq!(
                m.default_params.theta_names[theta_idx], "TVCL",
                "{label}: NCA's clearance estimate must land on TVCL, not on the \
                 allometric exponent"
            );
            let omega_idx = find_omega_idx_for_slot(&m, PK_IDX_CL)
                .unwrap_or_else(|| panic!("{label}: no omega for PK_IDX_CL"));
            assert_eq!(
                m.default_params.omega.eta_names[omega_idx], "ETA_CL",
                "{label}: CL's omega must resolve to ETA_CL"
            );
            // Every slot resolves to its own θ, and each to a *distinct* one — two
            // slots sharing a θ index is how `build_params` silently overwrites one
            // estimate with another.
            let v_idx = find_theta_for_slot(&m, PK_IDX_V).unwrap();
            assert_eq!(m.default_params.theta_names[v_idx], "TVV", "{label}: V");
            let ka_idx = find_theta_for_slot(&m, PK_IDX_KA).unwrap();
            assert_eq!(m.default_params.theta_names[ka_idx], "TVKA", "{label}: KA");
            let mut idxs = [theta_idx, v_idx, ka_idx];
            idxs.sort_unstable();
            let n = idxs.len();
            let mut uniq = idxs.to_vec();
            uniq.dedup();
            assert_eq!(
                uniq.len(),
                n,
                "{label}: two PK slots share a theta: {idxs:?}"
            );
        }
    }

    /// #1922: a parameter with **no** mu-referenced η, `V = TVV`, resolves to
    /// `TVV` through layer 2's whole-word match — not to the earlier `TVCL`, which
    /// merely *contains* `V` and was what the substring search answered, sharing
    /// one θ index with `PK_IDX_CL`.
    #[test]
    fn nca_resolves_a_no_iiv_v_to_tvv_not_to_a_tvcl_containing_v() {
        let no_eta = allometric_cl_intermediate_first()
            .replace("  omega ETA_V ~ 0.04\n", "")
            .replace("  V = TVV * exp(ETA_V)\n", "  V = TVV\n");
        let m = crate::parser::model_parser::parse_model_string(&no_eta).expect("parses");
        assert!(
            !m.mu_refs.iter().any(|(e, _)| e.contains('V')),
            "no eta may reference V, or layer 1 resolves V and this tests nothing"
        );
        // The premise: the first theta containing `V` is `TVCL`.
        assert_eq!(
            m.default_params
                .theta_names
                .iter()
                .find(|n| n.contains('V'))
                .map(String::as_str),
            Some("TVCL")
        );
        let cl = find_theta_for_slot(&m, PK_IDX_CL).expect("CL theta");
        let v = find_theta_for_slot(&m, PK_IDX_V).expect("V theta");
        assert_eq!(m.default_params.theta_names[cl], "TVCL");
        assert_eq!(
            m.default_params.theta_names[v], "TVV",
            "#1922: V must resolve to TVV, not to TVCL"
        );
        assert_ne!(cl, v, "#1922: the two slots must not share a theta");
    }

    /// The owner route alone resolves a parameter whose θ no name search can match:
    /// `V = THETA_VOL * exp(ETA_VOL)`. With whole-word layers 2 and 3 able to answer
    /// most fixtures by name, this is the one where losing layer 1 loses the θ.
    #[test]
    fn nca_owner_route_resolves_a_theta_no_name_search_matches() {
        let m = crate::parser::model_parser::parse_model_string(
            &allometric_cl_intermediate_first()
                .replace("theta TVV(", "theta THETA_VOL(")
                .replace("  omega ETA_V ~ 0.04\n", "  omega ETA_VOL ~ 0.04\n")
                .replace(
                    "  V = TVV * exp(ETA_V)\n",
                    "  V = THETA_VOL * exp(ETA_VOL)\n",
                ),
        )
        .expect("parses");
        assert_eq!(
            best_theta_name_match(&m.default_params.theta_names, "V"),
            NameMatch::Missing,
            "premise: no theta name matches V"
        );
        assert_eq!(
            find_theta_for_slot(&m, PK_IDX_V).map(|i| m.default_params.theta_names[i].as_str()),
            Some("THETA_VOL")
        );
    }

    /// #1930: layer 1 keys on the parameter a mu-reference was detected on, so an
    /// η named against no convention at all resolves its parameter. Before the fix
    /// `ETA_VOL` missed the `_V` name rule, `V` fell through to layer 2 and NCA's
    /// volume landed on `TVCL` (the #1922 collision). Both lookups are asserted:
    /// the θ and the η's omega index ride on the same association.
    #[test]
    fn nca_resolves_an_eta_by_its_owner_not_its_name() {
        let src = allometric_cl_intermediate_first()
            .replace("  omega ETA_V ~ 0.04\n", "  omega ETA_VOL ~ 0.04\n")
            .replace("  V = TVV * exp(ETA_V)\n", "  V = TVV * exp(ETA_VOL)\n")
            .replace("  omega ETA_CL ~ 0.09\n", "  omega BSV1 ~ 0.09\n")
            .replace("exp(ETA_CL)", "exp(BSV1)");
        let m = crate::parser::model_parser::parse_model_string(&src).expect("parses");
        // The premise: no eta passes the old name rule for either parameter.
        assert!(
            !m.mu_refs.keys().any(|e| {
                let u = e.to_ascii_uppercase();
                u == "V" || u.ends_with("_V") || u == "CL" || u.ends_with("_CL")
            }),
            "no eta may satisfy the name rule, or this does not test the owner route: {:?}",
            m.mu_refs.keys().collect::<Vec<_>>()
        );
        let name = |i: Option<usize>| i.map(|i| m.default_params.theta_names[i].as_str());
        assert_eq!(name(find_theta_for_slot(&m, PK_IDX_V)), Some("TVV"));
        assert_eq!(name(find_theta_for_slot(&m, PK_IDX_CL)), Some("TVCL"));
        let omega = find_omega_idx_for_slot(&m, PK_IDX_CL)
            .map(|i| m.default_params.omega.eta_names[i].as_str());
        assert_eq!(omega, Some("BSV1"), "CL's omega must follow its owned eta");
    }

    /// Review r4 #3: the reverse lookup over **every** structural slot
    /// `find_theta_for_slot` supports, on a model whose θ names are deliberately
    /// non-canonical (`TH_A`, `TLAG`, `THBIO` — not `TVCL`, `LAGTIME`, `F`).
    ///
    /// The non-canonical names are the point. The earlier fixtures let layer 3's
    /// canonical-name search mask a loss in layer 1: a minimal edit
    /// `if pk_slot == PK_IDX_LAGTIME { return None; }` inside
    /// `indiv_param_for_pk_slot` was caught by no test, because a model whose lag θ
    /// is called `TVLAGTIME` resolves through layer 3 anyway. Here `TLAG` contains
    /// no canonical slot name, so skipping the owner route loses the θ outright and
    /// a detected NCA lag would be silently left at its default.
    ///
    /// Asserted per slot so a failure names which one drifted, and the owner side is
    /// asserted for all nine — that is the #1899 lookup itself, over its whole
    /// contract rather than the one slot the issue was reported on.
    #[test]
    fn reverse_lookup_covers_every_structural_slot_with_noncanonical_theta_names() {
        let src = "[parameters]\n  theta TH_A(5.0, 0.1, 100.0)\n  \
             theta TH_B(50.0, 1.0, 500.0)\n  theta TH_C(10.0, 0.1, 200.0)\n  \
             theta TH_D(100.0, 1.0, 1000.0)\n  theta TH_E(2.0, 0.1, 50.0)\n  \
             theta TH_F(200.0, 1.0, 2000.0)\n  theta TH_G(1.5, 0.01, 50.0)\n  \
             theta TLAG(0.5, 0.0, 5.0)\n  theta THBIO(0.8, 0.01, 1.0)\n  \
             omega ETA_CLR ~ 0.09\n  omega ETA_VC ~ 0.04\n  omega ETA_ALAG ~ 0.01\n  \
             omega ETA_FBIO ~ 0.02\n  sigma PROP_ERR ~ 0.02 (sd)\n\
             [individual_parameters]\n  CLR = TH_A * exp(ETA_CLR)\n  \
             VC = TH_B * exp(ETA_VC)\n  QQ = TH_C\n  VP = TH_D\n  QQ3 = TH_E\n  \
             VP3 = TH_F\n  KABS = TH_G\n  ALAG = TLAG * exp(ETA_ALAG)\n  \
             FBIO = THBIO * exp(ETA_FBIO)\n\
             [structural_model]\n  pk three_cpt_oral(cl=CLR, v1=VC, q=QQ, v2=VP, \
             q3=QQ3, v3=VP3, ka=KABS, lagtime=ALAG, f=FBIO)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n";
        let m = crate::parser::model_parser::parse_model_string(src).expect("parses");
        // No theta name may contain a canonical slot name, or layer 3 can answer for
        // layer 1 and the masking this test exists to remove is back.
        for name in &m.default_params.theta_names {
            let u = name.to_ascii_uppercase();
            for canon in ["CL", "V2", "V3", "Q3", "KA", "LAGTIME"] {
                assert!(
                    !u.contains(canon),
                    "theta `{name}` contains the canonical name `{canon}`, so layer 3 \
                     could mask a layer-1 loss"
                );
            }
        }
        // Every structural slot resolves to the parameter the `pk(...)` line bound to
        // it — the #1899 lookup over its whole contract.
        let owners: [(&str, usize, &str); 9] = [
            ("CL", PK_IDX_CL, "CLR"),
            ("V", PK_IDX_V, "VC"),
            ("Q", PK_IDX_Q, "QQ"),
            ("V2", PK_IDX_V2, "VP"),
            ("KA", PK_IDX_KA, "KABS"),
            ("Q3", PK_IDX_Q3, "QQ3"),
            ("V3", PK_IDX_V3, "VP3"),
            ("LAGTIME", PK_IDX_LAGTIME, "ALAG"),
            ("F", PK_IDX_F, "FBIO"),
        ];
        for (label, slot, name) in owners {
            assert_eq!(
                m.indiv_param_for_pk_slot(slot)
                    .map(|i| m.indiv_param_names[i].as_str()),
                Some(name),
                "{label}: slot {slot} must resolve to {name}"
            );
        }
        // The two slots whose parameter carries a matching eta reach their theta
        // through layer 1 only — nothing in `theta_names` contains `LAGTIME` or a
        // bare `F`-like canonical token, so losing the owner route loses the theta.
        for (label, slot, theta) in [
            ("LAGTIME", PK_IDX_LAGTIME, "TLAG"),
            ("F", PK_IDX_F, "THBIO"),
        ] {
            assert_eq!(
                find_theta_for_slot(&m, slot).map(|i| m.default_params.theta_names[i].as_str()),
                Some(theta),
                "{label}: must resolve to {theta} through the owning parameter"
            );
        }
        // CL and V likewise, via their own etas.
        assert_eq!(
            find_theta_for_slot(&m, PK_IDX_CL).map(|i| m.default_params.theta_names[i].as_str()),
            Some("TH_A")
        );
        assert_eq!(
            find_theta_for_slot(&m, PK_IDX_V).map(|i| m.default_params.theta_names[i].as_str()),
            Some("TH_B")
        );
    }

    /// #1930, first shape: a PK role bound to a **literal**. No individual
    /// parameter owns the slot, so there is nothing for NCA to estimate and the
    /// lookup must say so. Before the fix layer 3 searched every θ for the
    /// canonical slot name as a substring and found `THCL_A`, a readout
    /// coefficient, because it contains `CL` — and `build_params` then pinned it
    /// at its upper bound with the pooled clearance (measured 5.0 against the
    /// model default 0.5).
    #[test]
    fn nca_writes_nothing_for_a_literal_bound_role() {
        let src = "[parameters]\n  theta THCL_A(0.5, 0.0, 5.0)\n  \
             theta TVV(10.0, 0.1, 500.0)\n  theta TVKA(1.5, 0.01, 50.0)\n  \
             omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.02 (sd)\n\
             [individual_parameters]\n  V = TVV * exp(ETA_V)\n  KA = TVKA\n  \
             BASE = THCL_A\n\
             [structural_model]\n  pk one_cpt_oral(cl=1.0, v=V, ka=KA)\n\
             [scaling]\n  y = central / V + BASE\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n";
        let m = crate::parser::model_parser::parse_model_string(src).expect("parses");
        // The premise: no individual parameter owns CL's slot, because a literal
        // binds the role — and a θ whose name contains `CL` exists, so a canonical
        // name search *would* find one.
        assert_eq!(
            m.indiv_param_for_pk_slot(PK_IDX_CL),
            None,
            "a literal-bound role owns no individual parameter"
        );
        assert!(m
            .default_params
            .theta_names
            .iter()
            .any(|n| n.contains("CL")));
        assert_eq!(resolve_theta_for_slot(&m, PK_IDX_CL), SlotTheta::Unowned);
        assert_eq!(find_theta_for_slot(&m, PK_IDX_CL), None);
        assert_eq!(find_omega_idx_for_slot(&m, PK_IDX_CL), None);
        // The slots that do have owners are unaffected.
        let v = find_theta_for_slot(&m, PK_IDX_V).expect("V theta");
        assert_eq!(m.default_params.theta_names[v], "TVV");

        // End to end: the lookup's answer is not the user-visible claim —
        // `build_params` leaving the readout coefficient alone is. `NcaInit::Nca`,
        // so no sweep can move a theta instead.
        let pop = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
            .expect("data loads");
        let out = inits_from_nca(&m, &pop, NcaInit::Nca).expect("nca init runs");
        let i = out
            .params
            .theta_names
            .iter()
            .position(|n| n == "THCL_A")
            .expect("THCL_A");
        assert_eq!(
            out.params.theta[i], m.default_params.theta[i],
            "#1930: THCL_A must not receive the pooled clearance when `cl` is a literal"
        );
        // And the user is told why, rather than the "no mu-referencing" message
        // that describes a different situation.
        assert!(
            out.warnings.iter().any(|w| w.contains(
                "CL is not bound to an individual \
                 parameter on the [structural_model] line"
            )),
            "expected the unowned-role warning: {:?}",
            out.warnings
        );
        assert!(
            !out.warnings.iter().any(|w| w.contains("could not map CL")),
            "the unmapped-CL message describes a different situation: {:?}",
            out.warnings
        );
    }

    /// The #1930 two-qualifying-η model: `CL = TVCL * exp(ETA_CL)` alongside an
    /// unbound `S = THS * exp(IIV_CL)`, both η names ending in `_CL`. `cl_line`
    /// is the `CL` assignment, so the owner route can be taken away.
    fn two_qualifying_etas_model(cl_line: &str) -> String {
        format!(
            "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  \
             theta THS(3.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
             theta TVKA(1.5, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  \
             omega IIV_CL ~ 0.25\n  omega ETA_V ~ 0.04\n  \
             sigma PROP_ERR ~ 0.02 (sd)\n\
             [individual_parameters]\n{cl_line}  \
             V = TVV * exp(ETA_V)\n  KA = TVKA\n  S = THS * exp(IIV_CL)\n\
             [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n"
        )
    }

    /// The η names that pass layer 1's name rule for `CL`, sorted.
    fn etas_named_for_cl(m: &CompiledModel) -> Vec<String> {
        let mut v: Vec<String> = m
            .mu_refs
            .keys()
            .filter(|e| {
                let u = e.to_ascii_uppercase();
                u == "CL" || u.ends_with("_CL")
            })
            .cloned()
            .collect();
        v.sort();
        v
    }

    /// #1930, second shape, with the owner route available: two η names qualify
    /// for `CL` by name, but only `ETA_CL`'s mu-reference was detected on `CL`
    /// itself, so the answer is `TVCL` and `ETA_CL` — exactly, and independent of
    /// `mu_refs`'s hash order. Before the fix layer 1 took the first qualifying η
    /// in `HashMap` order: a 10/10 split between `TVCL` and `THS` over 20 process
    /// launches.
    ///
    /// The variability is reproduced inside one process: every `HashMap::new()`
    /// draws a fresh `RandomState`, so rebuilding `mu_refs` 64 times visits both
    /// iteration orders of the two qualifying η. The test asserts that it did —
    /// otherwise a hash-order-dependent lookup could pass by luck — and that the
    /// answer was the same every time.
    #[test]
    fn nca_resolves_cl_by_owner_when_two_eta_names_qualify() {
        let src = two_qualifying_etas_model("  CL = TVCL * exp(ETA_CL)\n");
        let mut m = crate::parser::model_parser::parse_model_string(&src).expect("parses");
        let cl_pos = m.indiv_param_names.iter().position(|n| n == "CL").unwrap();
        assert_eq!(m.indiv_param_for_pk_slot(PK_IDX_CL), Some(cl_pos));
        // The premise: both etas pass the name rule and point at different thetas.
        assert_eq!(etas_named_for_cl(&m), vec!["ETA_CL", "IIV_CL"]);
        assert_eq!(m.mu_refs["ETA_CL"].theta_name, "TVCL");
        assert_eq!(m.mu_refs["IIV_CL"].theta_name, "THS");
        // The association the fix reads.
        let owners = &m.indiv_param_partials.mu_ref_owners;
        assert_eq!(owners.get("ETA_CL").map(String::as_str), Some("CL"));
        assert_eq!(owners.get("IIV_CL").map(String::as_str), Some("S"));

        let mut orders_seen = std::collections::BTreeSet::new();
        for round in 0..64 {
            let refs: std::collections::HashMap<String, crate::types::MuRef> = m
                .mu_refs
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let first = refs
                .keys()
                .find(|e| e.ends_with("_CL"))
                .cloned()
                .expect("a qualifying eta");
            orders_seen.insert(first);
            m.mu_refs = refs;
            let theta = find_theta_for_slot(&m, PK_IDX_CL)
                .map(|i| m.default_params.theta_names[i].as_str());
            assert_eq!(
                theta,
                Some("TVCL"),
                "round {round}: CL must resolve to TVCL"
            );
            let omega = find_omega_idx_for_slot(&m, PK_IDX_CL)
                .map(|i| m.default_params.omega.eta_names[i].as_str());
            assert_eq!(omega, Some("ETA_CL"), "round {round}: CL's omega is ETA_CL");
        }
        assert_eq!(
            orders_seen.len(),
            2,
            "both iteration orders must have been visited, or this does not test \
             order-independence: {orders_seen:?}"
        );
    }

    /// #1930, second shape, without the owner route: `CL` reads two η-carrying
    /// intermediates (`CLI = TVCL * exp(ETA_CL)`, `CLS = THS * exp(IIV_CL)`,
    /// `CL = CLI * CLS`), so no mu-reference is detected on `CL` and only the name
    /// rule is left — under which `ETA_CL` and `IIV_CL` both qualify, carrying
    /// different θ, and `CL` depends on both. Picking one is a guess, so the lookup
    /// declines: no θ, no omega, and a warning that names both candidates. `CL`'s
    /// θ and omega stay at their defaults end to end.
    ///
    /// The straddle (#1922 review r3): with `CL = CLI`, `CL` does not read `CLS`,
    /// so `IIV_CL` is not CL's to take and the name rule is unique — `TVCL` and
    /// `ETA_CL`. The original #1930 fixture put `IIV_CL` on an unrelated `S` and
    /// asserted ambiguity; that was the name rule borrowing another parameter's η.
    #[test]
    fn nca_declines_cl_when_two_qualifying_etas_have_no_owner() {
        let src = two_qualifying_etas_model(
            "  CLS = THS * exp(IIV_CL)\n  CLI = TVCL * exp(ETA_CL)\n  CL = CLI * CLS\n",
        )
        .replace("  S = THS * exp(IIV_CL)\n", "");
        let m = crate::parser::model_parser::parse_model_string(&src).expect("parses");
        assert!(
            name_rule_may_use(&m, "CL", "IIV_CL"),
            "premise: CL reads CLS"
        );
        // The other side: CL reads only CLI, so IIV_CL is CLS's alone.
        let reads_one = src.replace("CL = CLI * CLS", "CL = CLI");
        let m1 = crate::parser::model_parser::parse_model_string(&reads_one).expect("parses");
        assert!(!name_rule_may_use(&m1, "CL", "IIV_CL"));
        assert_eq!(
            find_theta_for_slot(&m1, PK_IDX_CL).map(|i| m1.default_params.theta_names[i].as_str()),
            Some("TVCL")
        );
        assert_eq!(
            find_omega_idx_for_slot(&m1, PK_IDX_CL)
                .map(|i| m1.default_params.omega.eta_names[i].as_str()),
            Some("ETA_CL")
        );
        // The premise: neither eta's mu-reference is owned by CL, and both pass the
        // name rule with different thetas.
        assert!(
            !m.indiv_param_partials
                .mu_ref_owners
                .values()
                .any(|o| o == "CL"),
            "no mu-reference may be detected on CL itself: {:?}",
            m.indiv_param_partials.mu_ref_owners
        );
        assert_eq!(etas_named_for_cl(&m), vec!["ETA_CL", "IIV_CL"]);
        assert_eq!(
            resolve_theta_for_slot(&m, PK_IDX_CL),
            SlotTheta::Ambiguous(vec![
                ("ETA_CL".into(), "TVCL".into()),
                ("IIV_CL".into(), "THS".into()),
            ])
        );
        assert_eq!(find_theta_for_slot(&m, PK_IDX_CL), None);
        assert_eq!(find_omega_idx_for_slot(&m, PK_IDX_CL), None);
        // The control: drop `S`'s eta and the name rule is unique again, so the
        // refusal above is about the ambiguity and nothing else.
        let unique = src
            .replace("  omega IIV_CL ~ 0.25\n", "")
            .replace("CLS = THS * exp(IIV_CL)", "CLS = THS");
        let mu = crate::parser::model_parser::parse_model_string(&unique).expect("parses");
        assert_eq!(
            find_theta_for_slot(&mu, PK_IDX_CL).map(|i| mu.default_params.theta_names[i].as_str()),
            Some("TVCL")
        );
        assert_eq!(
            find_omega_idx_for_slot(&mu, PK_IDX_CL)
                .map(|i| mu.default_params.omega.eta_names[i].as_str()),
            Some("ETA_CL")
        );

        let pop = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
            .expect("data loads");
        let out = inits_from_nca(&m, &pop, NcaInit::Nca).expect("nca init runs");
        for name in ["TVCL", "THS"] {
            let i = out
                .params
                .theta_names
                .iter()
                .position(|n| n == name)
                .unwrap();
            assert_eq!(
                out.params.theta[i], m.default_params.theta[i],
                "{name} must keep its default when CL's theta is ambiguous"
            );
        }
        assert_eq!(
            out.params.omega.matrix, m.default_params.omega.matrix,
            "no omega may be updated from an ambiguous CL eta"
        );
        let w = out
            .warnings
            .iter()
            .find(|w| w.contains("CL not written"))
            .unwrap_or_else(|| panic!("expected an ambiguity warning: {:?}", out.warnings));
        assert!(
            w.contains("ETA_CL -> TVCL") && w.contains("IIV_CL -> THS"),
            "the warning must name both candidates: {w}"
        );
        assert!(
            !out.warnings.iter().any(|w| w.contains("could not map CL")),
            "the unmapped-CL message describes a different situation: {:?}",
            out.warnings
        );
    }

    /// #1930 review r1: an owned mu-reference whose anchor is **not a θ** — a
    /// `[covariate_nn]` output, `CL = TYPICAL_PK.CL * exp(ETA_CL)` — still ends
    /// the lookup. `TYPICAL_PK.CL` is deliberately absent from `theta_names`, and
    /// dropping that candidate let the name rule take over, where an unrelated
    /// `S = THS * exp(IIV_CL)` qualifies: θ routing chose `THS` and the omega
    /// lookup `IIV_CL`, the very unrelated-parameter write #1930 removes.
    ///
    /// The straddle is asserted: `IIV_CL` really does pass the name rule with a
    /// writable θ, so a fix that fell through to it would redden here.
    #[cfg(feature = "nn")]
    #[test]
    fn nca_nn_anchored_owner_blocks_the_name_fallback() {
        let src = "[parameters]\n  theta THS(3.0, 0.01, 100.0)\n  \
             theta TVV(10.0, 0.1, 500.0)\n  theta TVKA(1.5, 0.01, 50.0)\n  \
             omega ETA_CL ~ 0.09\n  omega IIV_CL ~ 0.25\n  omega ETA_V ~ 0.04\n  \
             sigma PROP_ERR ~ 0.02 (sd)\n\
             [covariate_nn TYPICAL_PK]\n  inputs = [WT]\n  outputs = [CL]\n  \
             layers = [4]\n  activation = tanh\n  output = softplus\n\
             [individual_parameters]\n  CL = TYPICAL_PK.CL * exp(ETA_CL)\n  \
             V = TVV * exp(ETA_V)\n  KA = TVKA\n  S = THS * exp(IIV_CL)\n\
             [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n";
        let m = crate::parser::model_parser::parse_model_string(src).expect("parses");
        // The premise: ETA_CL is owned by CL and anchored to the NN output, which
        // is not a theta; IIV_CL passes the name rule with a writable theta.
        let owners = &m.indiv_param_partials.mu_ref_owners;
        assert_eq!(owners.get("ETA_CL").map(String::as_str), Some("CL"));
        assert_eq!(m.mu_refs["ETA_CL"].theta_name, "TYPICAL_PK.CL");
        assert!(!m
            .default_params
            .theta_names
            .iter()
            .any(|n| n == "TYPICAL_PK.CL"));
        assert_eq!(etas_named_for_cl(&m), vec!["ETA_CL", "IIV_CL"]);
        assert_eq!(m.mu_refs["IIV_CL"].theta_name, "THS");

        assert_eq!(
            resolve_theta_for_slot(&m, PK_IDX_CL),
            SlotTheta::NotATheta("TYPICAL_PK.CL".into())
        );
        assert_eq!(
            find_theta_for_slot(&m, PK_IDX_CL),
            None,
            "THS must not be chosen"
        );
        assert_eq!(
            find_omega_idx_for_slot(&m, PK_IDX_CL)
                .map(|i| m.default_params.omega.eta_names[i].as_str()),
            Some("ETA_CL"),
            "CL's omega is its own eta, not IIV_CL"
        );
        // The θ-anchored slots are unaffected.
        assert_eq!(
            find_theta_for_slot(&m, PK_IDX_V).map(|i| m.default_params.theta_names[i].as_str()),
            Some("TVV")
        );
        // The user-visible message names why nothing was written.
        let mut w = Vec::new();
        w.extend(slot_routing_warning(&m, PK_IDX_CL, "CL", Some("unmapped")));
        assert_eq!(w.len(), 1);
        assert!(
            w[0].contains("CL not written") && w[0].contains("`TYPICAL_PK.CL`"),
            "{w:?}"
        );
    }

    /// Two spellings that resolved before #1922's fix and must keep resolving:
    ///
    /// * **`IIV_V` resolves `V`** — the `_<PARAM>` suffix form of the η name,
    ///   rather than the `ETA_` prefix the other fixtures use.
    /// * **a θ name that does not contain `V`** — with `TVCL` renamed `THCL`, a
    ///   `V = TVV` carrying no η reaches `TVV` through layer 2.
    #[test]
    fn nca_theta_lookup_honours_both_documented_workarounds() {
        // Workaround 1: name the eta after its parameter, using the `_<PARAM>`
        // suffix form rather than the `ETA_` prefix the other fixtures use.
        let iiv = allometric_cl_intermediate_first()
            .replace("  omega ETA_V ~ 0.04\n", "  omega IIV_V ~ 0.04\n")
            .replace("  V = TVV * exp(ETA_V)\n", "  V = TVV * exp(IIV_V)\n");
        let m = crate::parser::model_parser::parse_model_string(&iiv).expect("parses");
        assert!(
            m.mu_refs.iter().any(|(e, _)| e == "IIV_V"),
            "the eta must be named IIV_V, or this tests a different spelling"
        );
        let v = find_theta_for_slot(&m, PK_IDX_V).expect("V theta");
        assert_eq!(
            m.default_params.theta_names[v], "TVV",
            "an eta named IIV_V must resolve V through layer 1"
        );

        // Workaround 2: rename the colliding theta. `THCL` does not contain `V`, so
        // layer 2's search reaches `TVV` even with no eta on V at all.
        let renamed = allometric_cl_intermediate_first()
            .replace("  omega ETA_V ~ 0.04\n", "")
            .replace("  V = TVV * exp(ETA_V)\n", "  V = TVV\n")
            .replace("theta TVCL(", "theta THCL(")
            .replace("TVCL * CLWT", "THCL * CLWT");
        let m = crate::parser::model_parser::parse_model_string(&renamed).expect("parses");
        // The premise: no theta before `TVV` contains `V`. (`TVKA` does, but later.)
        assert_eq!(
            m.default_params
                .theta_names
                .iter()
                .find(|n| n.contains('V'))
                .map(String::as_str),
            Some("TVV"),
            "TVV must be the first theta whose name contains `V`, or the workaround \
             is not what is being tested: {:?}",
            m.default_params.theta_names
        );
        let cl = find_theta_for_slot(&m, PK_IDX_CL).expect("CL theta");
        let v = find_theta_for_slot(&m, PK_IDX_V).expect("V theta");
        assert_eq!(
            m.default_params.theta_names[cl], "THCL",
            "CL still resolves"
        );
        assert_eq!(
            m.default_params.theta_names[v], "TVV",
            "a non-colliding theta name must let layer 2 reach TVV"
        );
        assert_ne!(cl, v, "the two slots must no longer share a theta");
    }

    /// #1922 end to end: dropping V's η — the one difference between the two
    /// models — must not change what NCA writes, since the pooled estimates come
    /// from the data and the routing must find `TVV` either way.
    ///
    /// Before the fix the no-η model resolved `PK_IDX_V` to `TVCL`, so the pooled
    /// volume (98.8 on `data/two_cpt_oral_cov.csv`) overshot `TVCL`'s upper bound
    /// and landed as the clamp `TVCL = 10.0`, with `TVV` left at its default.
    #[test]
    fn nca_writes_the_same_thetas_whether_or_not_v_carries_an_eta() {
        let pop = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
            .expect("data loads");
        let run = |src: &str| {
            let m = crate::parser::model_parser::parse_model_string(src).expect("parses");
            let out = inits_from_nca(&m, &pop, NcaInit::Nca).expect("nca init runs");
            let at = |name: &str| {
                let i = out
                    .params
                    .theta_names
                    .iter()
                    .position(|n| n == name)
                    .unwrap_or_else(|| panic!("no theta {name}"));
                (
                    out.params.theta[i],
                    m.default_params.theta[i],
                    m.default_params.theta_upper[i],
                )
            };
            (at("TVCL"), at("TVV"))
        };
        let ((ok_cl, _, cl_upper), (ok_v, ok_v_default, _)) =
            run(&allometric_cl_intermediate_first());
        assert!(
            (ok_v - ok_v_default).abs() > 1e-6,
            "control: TVV must receive the volume estimate, got {ok_v}"
        );
        // The fixture is only sensitive if a misrouted volume would show on TVCL.
        assert!(
            ok_v > cl_upper && ok_cl < cl_upper,
            "the volume estimate ({ok_v}) must overshoot TVCL's upper bound \
             ({cl_upper}) and the clearance ({ok_cl}) must not"
        );
        let no_eta = allometric_cl_intermediate_first()
            .replace("  omega ETA_V ~ 0.04\n", "")
            .replace("  V = TVV * exp(ETA_V)\n", "  V = TVV\n");
        let ((cl, _, _), (v, _, _)) = run(&no_eta);
        assert_eq!(cl, ok_cl, "#1922: TVCL must hold the clearance");
        assert_eq!(v, ok_v, "#1922: TVV must hold the volume");
    }

    /// #1899, end to end (review r1 #1): the user-visible outcome of
    /// `inits_from_nca` on the same model — which θ values the returned
    /// `ModelParameters` actually carries. The lookup assertions above are
    /// necessary but not sufficient: they say which index NCA aims at, not that
    /// the estimate lands there and that the allometric exponent is left alone.
    ///
    /// `NcaInit::Nca` (no sweep) is deliberate: the sweep would move every
    /// remaining free θ, including `CLWT_EXP`, and so could not distinguish "NCA
    /// wrote the exponent" from "the sweep refined it".
    #[test]
    fn inits_from_nca_writes_clearance_to_tvcl_and_leaves_the_exponent_alone() {
        let m =
            crate::parser::model_parser::parse_model_string(&allometric_cl_intermediate_first())
                .expect("model parses");
        // An oral dataset carrying the `WT` the intermediate reads.
        let pop = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
            .expect("data loads");
        let out = inits_from_nca(&m, &pop, NcaInit::Nca).expect("nca init runs");
        let theta_of = |name: &str| {
            let i = out
                .params
                .theta_names
                .iter()
                .position(|n| n == name)
                .unwrap_or_else(|| panic!("no theta {name}"));
            (out.params.theta[i], m.default_params.theta[i])
        };
        let (tvcl, tvcl_default) = theta_of("TVCL");
        let (exp_got, exp_default) = theta_of("CLWT_EXP");
        assert!(
            tvcl.is_finite() && tvcl > 0.0,
            "TVCL must carry a usable NCA estimate, got {tvcl}"
        );
        assert!(
            (tvcl - tvcl_default).abs() > 1e-6,
            "TVCL must move off its default {tvcl_default}, got {tvcl}"
        );
        // The defect's user-visible signature, both halves: before the fix the
        // clearance estimate went to `CLWT_EXP` and `TVCL` never moved.
        assert_eq!(
            exp_got, exp_default,
            "the allometric exponent must be untouched by NCA"
        );
        // `find_omega_idx_for_slot` returned `None` before the fix, so ETA_CL's
        // variance stayed at the model default; it is updated now (>= 3 subjects
        // have a valid NCA estimate in this dataset).
        let om_i = out
            .params
            .omega
            .eta_names
            .iter()
            .position(|n| n == "ETA_CL")
            .expect("ETA_CL");
        let om_got = out.params.omega.matrix[(om_i, om_i)];
        let om_default = m.default_params.omega.matrix[(om_i, om_i)];
        assert!(
            om_got.is_finite() && om_got > 0.0,
            "ETA_CL variance must stay usable, got {om_got}"
        );
        assert!(
            (om_got - om_default).abs() > 1e-9,
            "ETA_CL variance must be updated from the inter-subject CV^2, \
             got {om_got} (default {om_default})"
        );
    }

    /// Layer 2's matching rule, one row per shape it has to accept or reject.
    #[test]
    fn theta_name_match_rank_is_whole_word_not_substring() {
        for (theta, param, want) in [
            ("CLEARANCE", "CLEARANCE", Some(0)),
            ("TVCL", "CL", Some(0)),
            ("tvcl", "CL", Some(0)),
            ("TV_CL", "CL", Some(0)),
            ("THCL", "CL", Some(0)),
            ("THETA_CL", "CL", Some(0)),
            ("CL", "CL", Some(0)),
            ("POP_CL", "CL", Some(1)),
            ("CL_POP", "CL", Some(1)),
            ("THCL_A", "CL", Some(1)),
            // #1922: substrings that are not the word.
            ("TVCL", "V", None),
            ("TVKA", "V", None),
            ("TVQ3", "Q", None),
            ("TVF1", "F", None),
            ("THETA_VOL", "V", None),
            // #1899: the allometric exponent is not a clearance.
            ("CLWT_EXP", "CL", None),
            ("TVCL", "", None),
        ] {
            assert_eq!(
                theta_name_match_rank(theta, param),
                want,
                "{theta} vs {param}"
            );
        }
        let names = |ns: &[&str]| ns.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        // The whole-name match outranks a segment match.
        assert_eq!(
            best_theta_name_match(&names(&["CL_WT", "TVCL"]), "CL"),
            NameMatch::Found(1)
        );
        // Two at the best rank: ambiguous.
        assert_eq!(
            best_theta_name_match(&names(&["TV_CL", "THCL"]), "CL"),
            NameMatch::Tied(vec![0, 1])
        );
        assert_eq!(
            best_theta_name_match(&names(&["TVCL", "TVKA"]), "V"),
            NameMatch::Missing
        );
    }

    /// An ambiguous layer **stops** the lookup rather than handing it to a weaker
    /// one (review round 1). `v=VOL` with `TVVOL` and `THVOL` ties layer 2, and
    /// layer 3's canonical search for `V` would answer the unrelated `TVV` — the
    /// premise asserts that weaker answer is really there to fall through to.
    #[test]
    fn an_ambiguous_name_match_declines_instead_of_falling_through() {
        let src = "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  \
             theta TVVOL(10.0, 0.1, 500.0)\n  theta THVOL(10.0, 0.1, 500.0)\n  \
             theta TVV(10.0, 0.1, 500.0)\n  theta TVKA(1.5, 0.01, 50.0)\n  \
             omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.02 (sd)\n\
             [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  \
             VOL = TVVOL * THVOL / TVV\n  KA = TVKA\n\
             [structural_model]\n  pk one_cpt_oral(cl=CL, v=VOL, ka=KA)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n";
        let m = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let names = &m.default_params.theta_names;
        assert!(matches!(
            best_theta_name_match(names, "VOL"),
            NameMatch::Tied(_)
        ));
        let tvv = names.iter().position(|n| n == "TVV").unwrap();
        assert_eq!(
            best_theta_name_match(names, "V"),
            NameMatch::Found(tvv),
            "premise: layer 3 has a weaker answer to fall through to"
        );
        assert_eq!(
            resolve_theta_for_slot(&m, PK_IDX_V),
            SlotTheta::AmbiguousName(vec!["TVVOL".into(), "THVOL".into()])
        );
        assert_eq!(find_theta_for_slot(&m, PK_IDX_V), None);
        let w = slot_routing_warning(&m, PK_IDX_V, "V/V1", None).expect("a warning");
        assert!(
            w.contains("V/V1 not written") && w.contains("TVVOL, THVOL"),
            "{w}"
        );
    }

    /// The η-name fallback must not borrow an η the parser attributed to another
    /// parameter (review round 3). `CL = TVCL` carries no IIV and `ETA_CL` lives
    /// in an unrelated `S = THS * exp(ETA_CL)`: the name `ETA_CL` qualifies for
    /// `CL`, and the fallback used to send the clearance to `THS` and key CL's
    /// omega to `ETA_CL`. `S` owns no PK role, so the collision guard cannot see
    /// it — the lookup itself has to decline the η.
    #[test]
    fn eta_name_fallback_skips_an_eta_owned_by_another_parameter() {
        let src = "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  \
             theta THS(3.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
             theta TVKA(1.5, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  \
             omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.02 (sd)\n\
             [individual_parameters]\n  CL = TVCL\n  V = TVV * exp(ETA_V)\n  \
             KA = TVKA\n  S = THS * exp(ETA_CL)\n\
             [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n";
        let m = crate::parser::model_parser::parse_model_string(src).expect("parses");
        // The premise: `ETA_CL` is mu-referenced to THS, owned by S, and passes the
        // name rule for CL — which does not depend on it.
        assert_eq!(m.mu_refs["ETA_CL"].theta_name, "THS");
        assert_eq!(
            m.indiv_param_partials
                .mu_ref_owners
                .get("ETA_CL")
                .map(String::as_str),
            Some("S")
        );
        assert_eq!(etas_named_for_cl(&m), vec!["ETA_CL"]);
        assert!(!name_rule_may_use(&m, "CL", "ETA_CL"));
        assert_eq!(
            find_theta_for_slot(&m, PK_IDX_CL).map(|i| m.default_params.theta_names[i].as_str()),
            Some("TVCL")
        );
        assert_eq!(find_omega_idx_for_slot(&m, PK_IDX_CL), None);

        // End to end: THS and ETA_CL's variance are untouched, TVCL moves.
        let pop = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
            .expect("data loads");
        let out = inits_from_nca(&m, &pop, NcaInit::Nca).expect("nca init runs");
        assert_eq!(out.params.theta[1], m.default_params.theta[1], "THS");
        assert_ne!(out.params.theta[0], m.default_params.theta[0], "TVCL");
        assert_eq!(
            out.params.omega.matrix[(0, 0)],
            m.default_params.omega.matrix[(0, 0)],
            "ETA_CL's variance"
        );
    }

    /// Layer 3 is whole-word too: a parameter owning the slot whose θ name matches
    /// neither its own name nor the canonical one resolves to nothing, where the
    /// substring search took the first θ containing `V` — `TVCL`.
    #[test]
    fn layer_three_declines_rather_than_take_a_substring() {
        let src = allometric_cl_intermediate_first()
            .replace("theta TVV(", "theta THETA_VOL(")
            .replace("  omega ETA_V ~ 0.04\n", "")
            .replace("  V = TVV * exp(ETA_V)\n", "  VC = THETA_VOL\n")
            .replace("v=V,", "v=VC,");
        let m = crate::parser::model_parser::parse_model_string(&src).expect("parses");
        assert!(m.indiv_param_for_pk_slot(PK_IDX_V).is_some());
        assert_eq!(find_theta_for_slot(&m, PK_IDX_V), None);
    }

    /// `build_params`' guard: two slots resolving to one θ (here `CL` and `V`
    /// genuinely mu-referenced to the same `TVX`) write neither and warn, rather
    /// than leave the later estimate on the θ and the earlier one nowhere.
    #[test]
    fn build_params_writes_neither_slot_when_two_share_a_theta() {
        let src = "[parameters]\n  theta TVX(5.0, 0.001, 1000.0)\n  \
             theta TVKA(1.5, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  \
             omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.02 (sd)\n\
             [individual_parameters]\n  CL = TVX * exp(ETA_CL)\n  \
             V = TVX * exp(ETA_V)\n  KA = TVKA\n\
             [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n";
        let m = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let cl = find_theta_for_slot(&m, PK_IDX_CL);
        assert_eq!(
            cl,
            find_theta_for_slot(&m, PK_IDX_V),
            "premise: shared theta"
        );
        assert!(cl.is_some());
        let pop = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
            .expect("data loads");
        let out = inits_from_nca(&m, &pop, NcaInit::Nca).expect("nca init runs");
        assert_eq!(
            out.params.theta[0], m.default_params.theta[0],
            "TVX must keep its default"
        );
        assert!(
            out.warnings
                .iter()
                .any(|w| w.contains("CL, V/V1 all resolve to theta TVX")
                    && w.contains("TVX keeps its model default")),
            "the collision must be reported: {:?}",
            out.warnings
        );
        // The slot that does not collide is still written.
        assert_ne!(out.params.theta[1], m.default_params.theta[1], "TVKA");
    }

    /// The guard covers roles NCA never writes (review round 1): `CL` and `F`
    /// mu-referenced to one `TVX` must leave `TVX` — and so `F` — at its default,
    /// not take the pooled clearance. Before, F was outside the guard, and the
    /// clearance write moved F with it.
    ///
    /// The guard blocks writes only: F's default still rescales the apparent
    /// V/F onto `TVV` (review round 2). The control differs in one thing — `CL`
    /// on its own θ — and `TVV` must come out identical, where clearing F's index
    /// had made it V/F, twice the right value at F = 0.5.
    #[test]
    fn build_params_guard_includes_bioavailability() {
        let src = "[parameters]\n  theta TVX(0.5, 0.001, 1000.0)\n  \
             theta TVV(10.0, 0.1, 500.0)\n  theta TVKA(1.5, 0.01, 50.0)\n  \
             omega ETA_CL ~ 0.09\n  omega ETA_F ~ 0.04\n  \
             sigma PROP_ERR ~ 0.02 (sd)\n\
             [individual_parameters]\n  CL = TVX * exp(ETA_CL)\n  V = TVV\n  \
             KA = TVKA\n  F = TVX * exp(ETA_F)\n\
             [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n";
        let m = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let cl = find_theta_for_slot(&m, PK_IDX_CL);
        assert!(cl.is_some());
        assert_eq!(
            cl,
            find_theta_for_slot(&m, PK_IDX_F),
            "premise: shared theta"
        );
        let pop = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
            .expect("data loads");
        let out = inits_from_nca(&m, &pop, NcaInit::Nca).expect("nca init runs");
        assert_eq!(
            out.params.theta[0], m.default_params.theta[0],
            "TVX (and so F) must keep its default"
        );
        assert!(
            out.warnings
                .iter()
                .any(|w| w.contains("CL, F all resolve to theta TVX")),
            "the collision must be reported: {:?}",
            out.warnings
        );
        assert_ne!(out.params.theta[1], m.default_params.theta[1], "TVV");

        let control = src
            .replace(
                "theta TVX(0.5, 0.001, 1000.0)\n",
                "theta TVX(0.5, 0.001, 1000.0)\n  theta TVCL(0.2, 0.001, 1000.0)\n",
            )
            .replace("CL = TVX * exp(ETA_CL)", "CL = TVCL * exp(ETA_CL)");
        let mc = crate::parser::model_parser::parse_model_string(&control).expect("parses");
        assert_ne!(
            find_theta_for_slot(&mc, PK_IDX_CL),
            find_theta_for_slot(&mc, PK_IDX_F),
            "control premise: no collision"
        );
        let oc = inits_from_nca(&mc, &pop, NcaInit::Nca).expect("nca init runs");
        let tvv_c = mc
            .default_params
            .theta_names
            .iter()
            .position(|n| n == "TVV")
            .unwrap();
        assert_eq!(
            out.params.theta[1], oc.params.theta[tvv_c],
            "TVV must be V/F rescaled by F_default whether or not CL collides with F"
        );
        assert!(
            out.warnings
                .iter()
                .any(|w| w.contains("bioavailability F theta found (default=0.500)")),
            "F_default must still be applied: {:?}",
            out.warnings
        );
    }

    /// A parameter with no partials row cannot show it does not read an η owned
    /// by another parameter, so the name rule declines that η (review round 5).
    /// An `if`-assigned `CL` has no row; next to `S = THS * exp(ETA_CL)` the name
    /// rule used to accept `ETA_CL` on that unknown, and NCA wrote the clearance
    /// onto `THS` — a θ no PK role owns, so the collision guard could not see it.
    #[test]
    fn a_conditional_parameter_does_not_borrow_another_parameters_eta() {
        let src = "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  \
             theta THS(3.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
             theta TVKA(1.5, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  \
             omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.02 (sd)\n\
             [individual_parameters]\n  if (WT > 70) {\n    CL = TVCL * (WT / 70)^0.75\n  \
             } else {\n    CL = TVCL\n  }\n  V = TVV * exp(ETA_V)\n  \
             KA = TVKA\n  S = THS * exp(ETA_CL)\n\
             [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n";
        let m = crate::parser::model_parser::parse_model_string(src).expect("parses");
        // The premise: CL has no partials row, and ETA_CL is S's and passes the
        // name rule for CL.
        assert!(!m.indiv_param_partials.names.iter().any(|n| n == "CL"));
        assert_eq!(
            m.indiv_param_partials
                .mu_ref_owners
                .get("ETA_CL")
                .map(String::as_str),
            Some("S")
        );
        assert!(!name_rule_may_use(&m, "CL", "ETA_CL"));
        assert_eq!(
            find_theta_for_slot(&m, PK_IDX_CL).map(|i| m.default_params.theta_names[i].as_str()),
            Some("TVCL")
        );
        assert_eq!(find_omega_idx_for_slot(&m, PK_IDX_CL), None);

        let pop = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
            .expect("data loads");
        let out = inits_from_nca(&m, &pop, NcaInit::Nca).expect("nca init runs");
        assert_eq!(out.params.theta[1], m.default_params.theta[1], "THS");
        assert_ne!(out.params.theta[0], m.default_params.theta[0], "TVCL");
        assert_eq!(
            out.params.omega.matrix[(0, 0)],
            m.default_params.omega.matrix[(0, 0)],
            "ETA_CL's variance"
        );
    }

    /// The guard takes its roles from the model, not a hard-coded subset (review
    /// round 5): a transit model whose `CL` and transit count `n` are both
    /// mu-referenced to `TVX` must keep `TVX` — and so `n` — at its default.
    /// The absorption roles also resolve like any other (`MTT → TVMTT`), and so
    /// do the inverse-Gaussian ones, which used to be `Unmapped` outright.
    #[test]
    fn build_params_guard_covers_the_transit_and_ig_absorption_roles() {
        let src = "[parameters]\n  theta TVX(3.0, 0.001, 1000.0)\n  \
             theta TVV(10.0, 0.1, 500.0)\n  theta TVMTT(1.0, 0.01, 50.0)\n  \
             omega ETA_CL ~ 0.09\n  omega ETA_N ~ 0.04\n  \
             sigma PROP_ERR ~ 0.02 (sd)\n\
             [individual_parameters]\n  CL = TVX * exp(ETA_CL)\n  V = TVV\n  \
             MTT = TVMTT\n  NTR = TVX * exp(ETA_N)\n\
             [structural_model]\n  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n";
        let m = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let name = |i: Option<usize>| i.map(|i| m.default_params.theta_names[i].as_str());
        assert_eq!(name(find_theta_for_slot(&m, PK_IDX_CL)), Some("TVX"));
        assert_eq!(name(find_theta_for_slot(&m, PK_IDX_N)), Some("TVX"));
        assert_eq!(name(find_theta_for_slot(&m, PK_IDX_MTT)), Some("TVMTT"));

        // NCA has to produce a clearance for the guard to matter, and on the sparse
        // `transit_oral.csv` it produces none.
        let pop = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
            .expect("data loads");
        assert!(run_nca(&m, &pop).0.cl_f.is_some(), "premise: a pooled CL");
        let out = inits_from_nca(&m, &pop, NcaInit::Nca).expect("nca init runs");
        assert_eq!(
            out.params.theta[0], m.default_params.theta[0],
            "TVX (and so n) must keep its default"
        );
        assert!(
            out.warnings
                .iter()
                .any(|w| w.contains("CL, N all resolve to theta TVX")),
            "the collision must be reported: {:?}",
            out.warnings
        );
        assert_ne!(out.params.theta[1], m.default_params.theta[1], "TVV");

        let ig = parse_model_file(Path::new("examples/one_cpt_ig.ferx")).unwrap();
        let ig_name = |i: Option<usize>| i.map(|i| ig.default_params.theta_names[i].as_str());
        assert_eq!(ig_name(find_theta_for_slot(&ig, PK_IDX_MAT)), Some("TVMAT"));
        assert_eq!(ig_name(find_theta_for_slot(&ig, PK_IDX_CV2)), Some("TVCV2"));
    }

    /// The shared θ keeps its default through the sweeps too (review round 5):
    /// `Sweep` and `Ebe` sweep every free θ the NCA stage left at its default,
    /// which a θ the guard declined is, so `TVX` moved despite the warning that
    /// it keeps its model default. `Ebe` runs here on an analytical model, so
    /// it does not delegate to the `Sweep` path.
    #[test]
    fn the_sweeps_leave_a_shared_theta_at_its_default() {
        let src = "[parameters]\n  theta TVX(5.0, 0.001, 1000.0)\n  \
             theta TVKA(1.5, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  \
             omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.02 (sd)\n\
             [individual_parameters]\n  CL = TVX * exp(ETA_CL)\n  \
             V = TVX * exp(ETA_V)\n  KA = TVKA\n\
             [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n";
        let m = crate::parser::model_parser::parse_model_string(src).expect("parses");
        assert!(m.ode_spec.is_none());
        let pop = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
            .expect("data loads");
        for method in [NcaInit::Sweep, NcaInit::Ebe] {
            let out = inits_from_nca(&m, &pop, method).expect("nca init runs");
            assert_eq!(
                out.params.theta[0], m.default_params.theta[0],
                "{method:?}: TVX must keep its default"
            );
            assert!(
                out.warnings
                    .iter()
                    .any(|w| w.contains("TVX keeps its model default")),
                "{method:?}: {:?}",
                out.warnings
            );
        }
    }
}
