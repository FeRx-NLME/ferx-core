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
    CompiledModel, ModelParameters, OmegaMatrix, PkModel, Population, PK_IDX_CL, PK_IDX_F,
    PK_IDX_KA, PK_IDX_LAGTIME, PK_IDX_Q, PK_IDX_Q3, PK_IDX_V, PK_IDX_V2, PK_IDX_V3,
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

    // Collect non-fixed thetas that Option A left unchanged (still at model default).
    let mut remaining: Vec<usize> = (0..model.default_params.theta.len())
        .filter(|&i| {
            !base.params.theta_fixed[i]
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

    let mut remaining: Vec<usize> = (0..model.default_params.theta.len())
        .filter(|&i| {
            !base.params.theta_fixed[i]
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

/// Find which theta index drives a given PK slot.
///
/// Three-layer lookup (model-agnostic, works with or without mu-referencing):
/// 1. [`CompiledModel::indiv_param_for_pk_slot`]: the `indiv_param_names` entry
///    whose **own** PK slot matches, then look for an eta in `mu_refs` whose theta
///    is in `theta_names`. Not a `pk_indices` scan: an analytical model gives every
///    unbound intermediate the placeholder slot `0`, so `CLWT = exp(CLWT_EXP *
///    log(WT/70))` declared above `CL` answered for `PK_IDX_CL` and layer 2 then
///    wrote NCA's clearance onto the allometric exponent `CLWT_EXP` (#1899).
/// 2. Name-based fallback: search `theta_names` for a name containing the user's
///    `indiv_param_names[pos]` string (handles CLEARANCE → CLEARANCE, TVCL → CL, etc.).
/// 3. Canonical-name fallback: search `theta_names` for the standard slot name
///    ("CL", "V", "Q", etc.).
pub(crate) fn find_theta_for_slot(model: &CompiledModel, pk_slot: usize) -> Option<usize> {
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
    } else {
        return None;
    };

    let params = &model.default_params;

    // Layer 1: owning individual parameter → mu_refs → theta index.
    if let Some(indiv_pos) = model.indiv_param_for_pk_slot(pk_slot) {
        let indiv_name = &model.indiv_param_names[indiv_pos];
        // Search mu_refs for an eta whose theta_name is in theta_names and whose
        // eta name matches the indiv_param_name as a whole word (case-insensitive).
        // We require the eta name to end with `_<INDIV>` or equal `<INDIV>` exactly
        // (after uppercasing both) to avoid "ETA_VMAX" matching indiv_name "V".
        let indiv_upper = indiv_name.to_ascii_uppercase();
        for (eta_name, mu_ref) in &model.mu_refs {
            let eta_upper = eta_name.to_ascii_uppercase();
            let matches =
                eta_upper == indiv_upper || eta_upper.ends_with(&format!("_{indiv_upper}"));
            if matches {
                if let Some(idx) = params
                    .theta_names
                    .iter()
                    .position(|n| n == &mu_ref.theta_name)
                {
                    return Some(idx);
                }
            }
        }
        // Layer 2: search theta_names for one containing indiv_param_name.
        let found = params
            .theta_names
            .iter()
            .position(|n| n.to_ascii_uppercase().contains(&indiv_upper));
        if found.is_some() {
            return found;
        }
    }

    // Layer 3: search theta_names for canonical slot name.
    params
        .theta_names
        .iter()
        .position(|n| n.to_ascii_uppercase().contains(canonical))
}

/// Find the omega diagonal index for the eta driving a given PK slot.
///
/// Resolves the owning individual parameter through
/// [`CompiledModel::indiv_param_for_pk_slot`] for the same reason
/// [`find_theta_for_slot`] does: a `pk_indices` scan answers an unbound
/// intermediate for `PK_IDX_CL` (#1899), and its name then matches no eta, so the
/// slot's omega came back `None` and NCA left `ETA_CL`'s variance at the default.
fn find_omega_idx_for_slot(model: &CompiledModel, pk_slot: usize) -> Option<usize> {
    let params = &model.default_params;
    if let Some(indiv_pos) = model.indiv_param_for_pk_slot(pk_slot) {
        let indiv_name = &model.indiv_param_names[indiv_pos];
        let indiv_upper = indiv_name.to_ascii_uppercase();
        for (eta_name, _mu_ref) in &model.mu_refs {
            if eta_name.to_ascii_uppercase().contains(&indiv_upper) {
                return params.omega.eta_names.iter().position(|n| n == eta_name);
            }
        }
    }
    None
}

/// Route pooled NCA estimates to theta slots and return updated ModelParameters.
fn build_params(
    model: &CompiledModel,
    pop: &PopNca,
    warnings: &mut Vec<String>,
) -> ModelParameters {
    let mut params = model.default_params.clone();

    // Resolve all theta indices up-front (no closures capturing params).
    let cl_idx = find_theta_for_slot(model, PK_IDX_CL);
    let v_idx = find_theta_for_slot(model, PK_IDX_V);
    let ka_idx = find_theta_for_slot(model, PK_IDX_KA);
    let q_idx = find_theta_for_slot(model, PK_IDX_Q);
    let v2_idx = find_theta_for_slot(model, PK_IDX_V2);
    let lagtime_idx = find_theta_for_slot(model, PK_IDX_LAGTIME);
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
            warnings.push("inits_from_nca: could not map CL to a theta (no mu-referencing for CL eta); using model default".into());
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
            warnings
                .push("inits_from_nca: could not map V/V1 to a theta; using model default".into());
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
                    warnings.push(
                        "inits_from_nca: could not map Ka to a theta; using model default".into(),
                    );
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
        }
    }
    if let Some(v2) = pop.v2_peel {
        if let Some(idx) = v2_idx {
            write_theta(&mut params, idx, v2, "V2", warnings);
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

    /// Characterization test for the **adjacent** defect #1922, which #1899's fix
    /// does not touch and which `docs/model-file/fit-options.qmd` now states as a
    /// caveat: layer 2 of `find_theta_for_slot` matches the first θ whose name
    /// merely *contains* the parameter's name, and `TVCL` contains `V`.
    ///
    /// Two ways in, both asserted, because the docs sentence claims both and review
    /// r2 #1 found the single-case version too narrow:
    ///
    /// * **no η at all** — `V = TVV`;
    /// * **an η whose name does not match the parameter** — `V = TVV * exp(ETA_VOL)`.
    ///   Layer 1 matches a mu-reference by *η name* (`eta == PARAM` or
    ///   `eta.ends_with("_PARAM")`), so `ETA_VOL` misses even though `mu_refs`
    ///   already records `ETA_VOL -> TVV`. Without this case the docs' advice
    ///   ("name the η after its parameter") is unsupported, and the earlier
    ///   wording — "give such a parameter an η" — was simply wrong.
    ///
    /// This pins today's wrong answer so the docs sentences describing it are backed
    /// by a test that can die — delete a caveat from the docs and this is the test
    /// that says the caveat was real. Invert it when #1922 is fixed.
    #[test]
    fn nca_layer_two_substring_match_can_still_mis_resolve_v() {
        let no_eta = allometric_cl_intermediate_first()
            .replace("  omega ETA_V ~ 0.04\n", "")
            .replace("  V = TVV * exp(ETA_V)\n", "  V = TVV\n");
        let wrong_eta_name = allometric_cl_intermediate_first()
            .replace("  omega ETA_V ~ 0.04\n", "  omega ETA_VOL ~ 0.04\n")
            .replace("  V = TVV * exp(ETA_V)\n", "  V = TVV * exp(ETA_VOL)\n");
        for (label, src) in [("no eta", no_eta), ("eta name misses", wrong_eta_name)] {
            let m = crate::parser::model_parser::parse_model_string(&src)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(
                !m.mu_refs.iter().any(|(e, _)| e == "ETA_V"),
                "{label}: no eta may be named ETA_V, or layer 1 resolves V and this \
                 tests nothing"
            );
            let cl = find_theta_for_slot(&m, PK_IDX_CL).expect("CL theta");
            let v = find_theta_for_slot(&m, PK_IDX_V).expect("V theta");
            assert_eq!(
                m.default_params.theta_names[cl], "TVCL",
                "{label}: CL still resolves correctly — #1899's half is unaffected"
            );
            assert_eq!(
                m.default_params.theta_names[v], "TVCL",
                "{label}: #1922: V resolves to TVCL because `TVCL` contains `V`. When \
                 this starts failing, #1922 has been fixed — invert the assertion and \
                 drop the caveat from docs/model-file/fit-options.qmd."
            );
            assert_eq!(cl, v, "{label}: #1922: the two slots share one theta index");
        }
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

    /// Characterization of #1930 (review r3 #2), the first of the two shapes
    /// `docs/model-file/fit-options.qmd` now names as *not* covered by step 1: a
    /// PK role bound to a **literal**. No individual parameter owns the slot, so
    /// layers 1 and 2 are skipped and layer 3 searches every θ for the canonical
    /// slot name as a substring — here `THCL_A`, a readout coefficient, because it
    /// contains `CL`. The clearance is a literal in this model, so the correct
    /// answer is `None`: there is nothing to estimate.
    ///
    /// Pins today's answer so the docs sentence describing it can die. Invert when
    /// #1930 is fixed.
    #[test]
    fn nca_resolves_a_literal_bound_role_by_canonical_name() {
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
        // binds the role. This is also #1899's own lookup answering correctly —
        // `None` rather than some unrelated name's index.
        assert_eq!(
            m.indiv_param_for_pk_slot(PK_IDX_CL),
            None,
            "a literal-bound role owns no individual parameter"
        );
        let cl = find_theta_for_slot(&m, PK_IDX_CL);
        assert_eq!(
            cl.map(|i| m.default_params.theta_names[i].as_str()),
            Some("THCL_A"),
            "#1930: layer 3 matches `THCL_A` because it contains `CL`. When this \
             starts failing (ideally returning None), #1930 has been fixed — update \
             docs/model-file/fit-options.qmd with it."
        );
        // The slots that do have owners are unaffected.
        let v = find_theta_for_slot(&m, PK_IDX_V).expect("V theta");
        assert_eq!(m.default_params.theta_names[v], "TVV");

        // The user-visible half (review r4 #1): the lookup's answer is not the
        // documented claim — `build_params` actually *writing* the pooled clearance
        // onto the readout coefficient is. An ownerless-write guard added for #1930
        // would leave the assertion above green while making the docs stale, so run
        // it end to end. `NcaInit::Nca`, so no sweep can move a theta instead.
        let pop = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
            .expect("data loads");
        let out = inits_from_nca(&m, &pop, NcaInit::Nca).expect("nca init runs");
        let i = out
            .params
            .theta_names
            .iter()
            .position(|n| n == "THCL_A")
            .expect("THCL_A");
        assert_ne!(
            out.params.theta[i], m.default_params.theta[i],
            "#1930: THCL_A receives the pooled clearance although `cl` is a literal. \
             When this starts failing, #1930 has been fixed — drop the paragraph from \
             docs/model-file/fit-options.qmd."
        );
        // And it is pinned at its upper bound, since a clearance is not a readout
        // coefficient: measured 5.0 against the model default 0.5.
        assert_eq!(out.params.theta[i], m.default_params.theta_upper[i]);
    }

    /// Characterization of #1930 (review r3 #2), second shape: the mu-reference
    /// match is **not checked for uniqueness**. Layer 1 iterates `model.mu_refs`,
    /// a `HashMap`, and takes the first η whose name qualifies, so when two η names
    /// both end in `_CL` the winner is whichever the iteration yields first — and
    /// with `HashMap`'s per-process seed that was measured as a 10/10 split over 20
    /// launches, alternating between `TVCL` (the real clearance θ) and `THS` (an
    /// unbound intermediate's).
    ///
    /// **What this asserts, precisely** (review r4 #1): the *structural* ambiguity —
    /// that two η names qualify, that they carry different θ, and that the lookup
    /// returns one of them rather than declining. It does **not** assert the
    /// run-to-run variability: that is a property of `mu_refs` being a `HashMap`,
    /// measured and recorded in #1930, and a fix that only made the iteration order
    /// deterministic would leave this test green while removing the variability. It
    /// would also still be returning an arbitrary one of two answers, which is why
    /// the assertion is on the ambiguity and not on the ordering.
    #[test]
    fn nca_mu_ref_match_is_ambiguous_when_two_eta_names_qualify() {
        let src = "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  \
             theta THS(3.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
             theta TVKA(1.5, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  \
             omega IIV_CL ~ 0.25\n  omega ETA_V ~ 0.04\n  \
             sigma PROP_ERR ~ 0.02 (sd)\n\
             [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  \
             V = TVV * exp(ETA_V)\n  KA = TVKA\n  S = THS * exp(IIV_CL)\n\
             [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n";
        let m = crate::parser::model_parser::parse_model_string(src).expect("parses");
        // #1899's lookup is exact and unaffected: CL's slot is owned by `CL`, not by
        // the unbound `S`, in either hash order.
        let cl_pos = m.indiv_param_names.iter().position(|n| n == "CL").unwrap();
        assert_eq!(m.indiv_param_for_pk_slot(PK_IDX_CL), Some(cl_pos));
        // The premise: two eta names satisfy layer 1's `_CL` rule.
        let qualifying: Vec<&String> = m
            .mu_refs
            .keys()
            .filter(|e| {
                let u = e.to_ascii_uppercase();
                u == "CL" || u.ends_with("_CL")
            })
            .collect();
        assert_eq!(
            qualifying.len(),
            2,
            "both ETA_CL and IIV_CL must qualify, or the ambiguity is not present: \
             {qualifying:?}"
        );
        // The two qualifying etas must carry DIFFERENT thetas, or there is no
        // ambiguity to characterize.
        let mut qualifying_thetas: Vec<&str> = qualifying
            .iter()
            .map(|e| m.mu_refs[*e].theta_name.as_str())
            .collect();
        qualifying_thetas.sort_unstable();
        assert_eq!(
            qualifying_thetas,
            vec!["THS", "TVCL"],
            "the two qualifying etas must point at different thetas"
        );
        // #1930: the lookup returns one of them rather than declining, and which one
        // is not determined by the model.
        let got = find_theta_for_slot(&m, PK_IDX_CL)
            .map(|i| m.default_params.theta_names[i].clone())
            .expect("a theta");
        assert!(
            got == "TVCL" || got == "THS",
            "#1930: layer 1 returns one of the two qualifying etas' thetas; got {got}"
        );
    }

    /// `docs/model-file/fit-options.qmd` offers two workarounds for the #1922
    /// collision and names a second η spelling that resolves. Each is a claim a
    /// reader will act on, so each gets an assertion:
    ///
    /// * **`IIV_V` resolves `V`** — the docs say the η must be the parameter's name
    ///   or end in `_<PARAM>`, and quote `IIV_V` as well as `ETA_V`. Only `ETA_V`
    ///   was covered, so half of that sentence was unchecked.
    /// * **a non-colliding θ name resolves** — "give the θ a name that does not
    ///   collide" was untested: with `TVCL` renamed `THCL`, which does not contain
    ///   `V`, a `V = TVV` carrying no η must reach `TVV` through layer 2.
    ///
    /// Delete either sentence from the docs and this is the test that says it was
    /// load-bearing.
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
        // Layer 2 takes the FIRST theta whose name contains `V`, so the premise is
        // that none before `TVV` does. (`TVKA` also contains `V`, but it is later.)
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

    /// The user-visible half of #1922's characterization (review r2 #1): asserting
    /// that two slots share a θ *index* does not assert what the docs claim, namely
    /// that the volume estimate lands on `TVCL` and `TVV` is never written. A later
    /// duplicate-write guard in `build_params` would keep the index assertion green
    /// while making that sentence false.
    ///
    /// Run against the η-named control, so the comparison is between two models that
    /// differ in **one** thing — the η's name.
    ///
    /// Measured on `data/two_cpt_oral_cov.csv`: the control writes `TVV = 98.835…`
    /// and `TVCL = 0.4387…`. The colliding model sends that volume to `TVCL`, whose
    /// upper bound is `10.0`, so what actually lands is the **clamp** — `TVCL =
    /// 10.0`, 23× the clearance estimate and pinned at its bound, with `TVV` left at
    /// its default. The clamp is why this asserts the bound rather than the raw
    /// volume: it is the same signature as #1899's `CLWT_EXP = 2.0`.
    #[test]
    fn nca_substring_collision_writes_the_volume_estimate_onto_tvcl() {
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
        // Control: `ETA_V` matches, so each slot reaches its own θ.
        let ((ok_cl, _, cl_upper), (ok_v, ok_v_default, _)) =
            run(&allometric_cl_intermediate_first());
        assert!(
            (ok_v - ok_v_default).abs() > 1e-6,
            "control: TVV must receive the volume estimate, got {ok_v}"
        );
        assert!(
            ok_v > cl_upper,
            "the fixture only shows the damage if the volume estimate ({ok_v})              overshoots TVCL's upper bound ({cl_upper})"
        );
        // #1922: rename the η and `PK_IDX_V` resolves to `TVCL`.
        let collide = allometric_cl_intermediate_first()
            .replace("  omega ETA_V ~ 0.04\n", "  omega ETA_VOL ~ 0.04\n")
            .replace("  V = TVV * exp(ETA_V)\n", "  V = TVV * exp(ETA_VOL)\n");
        let ((bad_cl, _, _), (bad_v, bad_v_default, _)) = run(&collide);
        assert_eq!(
            bad_v, bad_v_default,
            "#1922: TVV is never written when PK_IDX_V resolves elsewhere"
        );
        approx::assert_relative_eq!(bad_cl, cl_upper, max_relative = 1e-12);
        assert!(
            bad_cl > 1.5 * ok_cl,
            "#1922: TVCL must hold the clamped volume, not the clearance ({bad_cl} vs {ok_cl})"
        );
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
}
