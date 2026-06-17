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
pub fn inits_from_nca(
    model: &CompiledModel,
    population: &Population,
    method: NcaInit,
) -> SuggestedStart {
    // Reached from `fit()` (which validates first) but also a standalone public
    // entrypoint; the Sweep/Ebe strategies predict, so guard the modeled-`RATE`
    // dose precondition here too — same loud-not-silent contract as
    // `predict()`/`simulate()` (#324). No-op for the common all-`Fixed` dataset.
    crate::api::assert_modeled_doses_supported(model, population);
    match method {
        NcaInit::Nca => nca_only(model, population),
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
fn nca_with_sweep(model: &CompiledModel, population: &Population) -> SuggestedStart {
    let mut base = nca_only(model, population);

    // Collect non-fixed thetas that Option A left unchanged (still at model default).
    let mut remaining: Vec<usize> = (0..model.default_params.theta.len())
        .filter(|&i| {
            !base.params.theta_fixed[i]
                && (base.params.theta[i] - model.default_params.theta[i]).abs() < 1e-12
        })
        .collect();

    if remaining.is_empty() {
        return base;
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
        return base;
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
                );
                base.params = swept;
                base.warnings.extend(w);
                remaining.retain(|&i| i != ia && i != ib);
            }
        }
    }

    // Independent 1D sweeps for any remaining unwritten thetas.
    if !remaining.is_empty() {
        let (swept, w) =
            sweep_unwritten_thetas(model, population, &base.params, &remaining, 9, 10.0);
        base.params = swept;
        base.warnings.extend(w);
    }

    base
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
fn nca_with_ebe(model: &CompiledModel, population: &Population) -> SuggestedStart {
    // ODE fallback: EBE sweeps require per-subject numerical integration per
    // inner iteration — too slow (~minutes) and unreliable from uninformed
    // defaults.  Delegate to the etas=0 sweep directly (which runs NCA + sweep)
    // rather than calling nca_only() first and then nca_with_sweep()
    // (which would run NCA twice).
    if model.ode_spec.is_some() {
        let mut result = nca_with_sweep(model, population);
        result.warnings.insert(0,
            "inits_from_nca (nca_ebe): ODE model — EBE sweep skipped (too slow; ODE integration per inner iteration). Falling back to etas=0 sweep (nca_sweep).".into(),
        );
        return result;
    }

    let mut base = nca_only(model, population);

    let mut remaining: Vec<usize> = (0..model.default_params.theta.len())
        .filter(|&i| {
            !base.params.theta_fixed[i]
                && (base.params.theta[i] - model.default_params.theta[i]).abs() < 1e-12
        })
        .collect();

    if remaining.is_empty() {
        return base;
    }

    // For EBE sweeps, only include lognormal-parameterised thetas
    // (mu_ref.log_transformed = true, i.e. THETA * exp(ETA) form).
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
        .filter(|mr| mr.log_transformed)
        .map(|mr| mr.theta_name.as_str())
        .collect();
    let logit_theta_names: std::collections::HashSet<&str> = model
        .mu_refs
        .values()
        .filter(|mr| !mr.log_transformed)
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
        return base;
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
            sweep_unwritten_thetas(model, population, &base.params, &logit_remaining, 9, 10.0);
        base.params = swept;
        base.warnings.extend(w);
    }

    base
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

    // ODE models: pk_indices are sequential (slot i = position i), not semantic.
    // NCA can't reliably map estimates to the user's parameter names (which could
    // be KE, EMAX, or anything else — not necessarily CL/V).  Fall back to model
    // defaults and let the nca_sweep method sweep them via rRMSE.
    if model.ode_spec.is_some() {
        warnings.push(
            "inits_from_nca: ODE model detected; NCA estimation skipped (parameter names are user-defined). Use the nca_sweep method for rRMSE-based sweep.".into(),
        );
        return (empty_pop_nca(), warnings);
    }

    let per_subject: Vec<SubjectNca> = match model.pk_model {
        PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral => population
            .subjects
            .par_iter()
            .map(nca_one_cpt_oral)
            .collect(),

        PkModel::OneCptIv | PkModel::TwoCptIv | PkModel::ThreeCptIv => {
            population.subjects.par_iter().map(nca_one_cpt_iv).collect()
        }
    };

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
        PkModel::TwoCptIv | PkModel::TwoCptOral => {
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
/// 1. `pk_indices`: find the position in `indiv_param_names` whose PK slot matches,
///    then look for an eta in `mu_refs` whose theta is in `theta_names`.
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

    // Layer 1: pk_indices → indiv_param_names → mu_refs → theta index.
    if let Some(indiv_pos) = model.pk_indices.iter().position(|&s| s == pk_slot) {
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
fn find_omega_idx_for_slot(model: &CompiledModel, pk_slot: usize) -> Option<usize> {
    let params = &model.default_params;
    if let Some(indiv_pos) = model.pk_indices.iter().position(|&s| s == pk_slot) {
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

    // Write lag time (oral models only)
    match model.pk_model {
        PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral => {
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
        let result = inits_from_nca(&model, &empty_pop, NcaInit::Nca);
        // Must not panic and should warn.
        assert!(!result.warnings.is_empty());
        // Params should equal model defaults.
        assert_eq!(result.params.theta, model.default_params.theta);
    }

    #[test]
    fn test_suggest_start_respects_bounds() {
        let model = parse_model_file(Path::new("examples/warfarin.ferx")).unwrap();
        let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).unwrap();
        let result = inits_from_nca(&model, &population, NcaInit::Nca);
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
        let result = inits_from_nca(&model, &population, NcaInit::Nca);
        assert_eq!(
            result.params.theta[0], original_val,
            "fixed theta must not be overwritten"
        );
    }
}
