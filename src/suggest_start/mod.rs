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
///    whose **own** PK slot matches, then its mu-referenced θ. Not a
///    `pk_indices` scan: an analytical model gives every unbound intermediate
///    the placeholder slot `0`, so `CLWT = exp(CLWT_EXP * log(WT/70))` declared
///    above `CL` answered for `PK_IDX_CL` and layer 2 then wrote NCA's clearance
///    onto the allometric exponent `CLWT_EXP` (#1899). The η is found through
///    the parameter it appears in (`eta_param_info`), not through its name, so
///    `V = TVV * exp(ETA_VOL)` resolves to `TVV` (#1922); the η-name rule
///    (`ETA_V`, `IIV_V`) remains only as a fallback for an η the parser did not
///    attribute to the parameter. Either route declines when it reaches two
///    different θ.
/// 2. Name-based fallback: the θ whose name matches the parameter's name **as a
///    whole word** ([`theta_name_match_rank`]) — `CLEARANCE → CLEARANCE`,
///    `TVCL → CL`, `THETA_CL → CL` — never a bare substring, so `TVCL` is not a
///    match for `V` (#1922). Two θ tied at the best rank is ambiguous: `None`.
/// 3. Canonical-name fallback: the same whole-word search for the standard slot
///    name ("CL", "V", "Q", etc.).
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

    let theta_names = &model.default_params.theta_names;

    if let Some(indiv_pos) = model.indiv_param_for_pk_slot(pk_slot) {
        let indiv_name = &model.indiv_param_names[indiv_pos];
        // Layer 1: owning individual parameter → its etas → mu_refs → theta index.
        let theta_of_eta = |eta: &str| {
            let mu_ref = model.mu_refs.get(eta)?;
            theta_names.iter().position(|n| n == &mu_ref.theta_name)
        };
        let owned = unique(
            etas_of_param(model, indiv_name)
                .into_iter()
                .filter_map(theta_of_eta),
        );
        if let Some(answer) = owned.decided() {
            return answer;
        }
        let by_eta_name = unique(
            model
                .mu_refs
                .keys()
                .filter(|e| eta_name_fallback_applies(model, e, indiv_name))
                .filter_map(|e| theta_of_eta(e)),
        );
        if let Some(answer) = by_eta_name.decided() {
            return answer;
        }
        // Layer 2: theta_names matching indiv_param_name as a whole word.
        if let Some(answer) = best_theta_name_match(theta_names, indiv_name).decided() {
            return answer;
        }
    }

    // Layer 3: theta_names matching the canonical slot name as a whole word.
    best_theta_name_match(theta_names, canonical)
        .decided()
        .flatten()
}

/// The eta names the parser attributed to individual parameter `param`
/// (`eta_param_info`, case-insensitive on the parameter name).
fn etas_of_param<'a>(model: &'a CompiledModel, param: &str) -> Vec<&'a str> {
    model
        .eta_param_info
        .iter()
        .filter(|info| info.individual_param_name.eq_ignore_ascii_case(param))
        .map(|info| info.eta_name.as_str())
        .collect()
}

/// Whether the η-name fallback may use `eta` for parameter `param`: its name
/// qualifies ([`eta_name_matches`]) **and** the parser did not attribute it to a
/// different parameter. An η the parser placed in another parameter is that
/// parameter's, whatever it is called: with `CL = TVCL` (no IIV) next to
/// `S = THS * exp(ETA_CL)`, `ETA_CL` must not carry the clearance onto `THS`
/// (review round 3). The fallback is for η the parser left unattributed — a
/// hand-built model with no `eta_param_info`, say.
fn eta_name_fallback_applies(model: &CompiledModel, eta: &str, param: &str) -> bool {
    eta_name_matches(eta, param)
        && !model.eta_param_info.iter().any(|info| {
            info.eta_name == eta && !info.individual_param_name.eq_ignore_ascii_case(param)
        })
}

/// The η-name fallback: `eta` is exactly `param` or ends in `_<param>`
/// (case-insensitive), so `ETA_V` / `IIV_V` match `V` but `ETA_VMAX` does not.
fn eta_name_matches(eta: &str, param: &str) -> bool {
    let eta = eta.to_ascii_uppercase();
    let param = param.to_ascii_uppercase();
    eta == param || eta.ends_with(&format!("_{param}"))
}

/// The outcome of one lookup layer. `Ambiguous` is not `Missing`: a layer that
/// reaches two answers **stops** the lookup, because falling through would hand
/// the decision to a weaker layer — `v=VOL` with `TVVOL` and `THVOL` tied in
/// layer 2 would otherwise resolve to an unrelated `TVV` through layer 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lookup {
    Found(usize),
    Ambiguous,
    Missing,
}

impl Lookup {
    /// `Some(answer)` when this layer decides the lookup (`Some(Some(i))` found,
    /// `Some(None)` ambiguous), `None` when the next layer should be tried.
    fn decided(self) -> Option<Option<usize>> {
        match self {
            Lookup::Found(i) => Some(Some(i)),
            Lookup::Ambiguous => Some(None),
            Lookup::Missing => None,
        }
    }
}

/// The one distinct value `it` yields: `Missing` when it yields none,
/// `Ambiguous` when it yields two different ones — a lookup that reaches two
/// answers declines rather than picking whichever came first.
fn unique(it: impl IntoIterator<Item = usize>) -> Lookup {
    let mut found = Lookup::Missing;
    for x in it {
        match found {
            Lookup::Missing => found = Lookup::Found(x),
            Lookup::Found(y) if y != x => return Lookup::Ambiguous,
            _ => {}
        }
    }
    found
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

/// The θ index whose name matches `param` best ([`theta_name_match_rank`]);
/// `Missing` when nothing matches, `Ambiguous` when two θ tie at the best rank.
fn best_theta_name_match(theta_names: &[String], param: &str) -> Lookup {
    let ranked: Vec<(u8, usize)> = theta_names
        .iter()
        .enumerate()
        .filter_map(|(i, n)| theta_name_match_rank(n, param).map(|r| (r, i)))
        .collect();
    let Some(best) = ranked.iter().map(|&(r, _)| r).min() else {
        return Lookup::Missing;
    };
    unique(
        ranked
            .into_iter()
            .filter(|&(r, _)| r == best)
            .map(|(_, i)| i),
    )
}

/// Find the omega diagonal index for the eta driving a given PK slot.
///
/// Resolves the owning individual parameter through
/// [`CompiledModel::indiv_param_for_pk_slot`] for the same reason
/// [`find_theta_for_slot`] does: a `pk_indices` scan answers an unbound
/// intermediate for `PK_IDX_CL` (#1899), and its name then matches no eta, so the
/// slot's omega came back `None` and NCA left `ETA_CL`'s variance at the default.
/// The η is the mu-referenced one the parser attributed to that parameter, with
/// the same whole-word η-name fallback as [`find_theta_for_slot`] — never a bare
/// substring, which let `ETA_CLWT` key `CL`'s omega (#1922). Two candidates is
/// ambiguous: `None`.
fn find_omega_idx_for_slot(model: &CompiledModel, pk_slot: usize) -> Option<usize> {
    let indiv_pos = model.indiv_param_for_pk_slot(pk_slot)?;
    let indiv_name = &model.indiv_param_names[indiv_pos];
    let eta_names = &model.default_params.omega.eta_names;
    let omega_of = |eta: &str| {
        if !model.mu_refs.contains_key(eta) {
            return None;
        }
        eta_names.iter().position(|n| n == eta)
    };
    let owned = unique(
        etas_of_param(model, indiv_name)
            .into_iter()
            .filter_map(omega_of),
    );
    if let Some(answer) = owned.decided() {
        return answer;
    }
    unique(
        model
            .mu_refs
            .keys()
            .filter(|e| eta_name_fallback_applies(model, e, indiv_name))
            .filter_map(|e| omega_of(e)),
    )
    .decided()
    .flatten()
}

/// Clear every slot whose θ index another slot also resolved to, with a warning.
///
/// `build_params` writes each pooled estimate in turn, so two slots sharing one
/// index would leave the later estimate on that θ and the earlier one nowhere —
/// a volume silently written onto `TVCL` (#1922). A mis-resolution must degrade
/// to "left at the model default", so neither slot is written. Every PK role
/// takes part, including those NCA never writes (F, Q3, V3): a pooled clearance
/// written onto the θ that F also resolves to would move F too.
fn drop_shared_theta_indices(
    model: &CompiledModel,
    slots: &mut [(&str, &mut Option<usize>)],
    warnings: &mut Vec<String>,
) {
    let resolved: Vec<Option<usize>> = slots.iter().map(|(_, idx)| **idx).collect();
    let mut cleared = vec![false; slots.len()];
    for (i, idx) in resolved.iter().enumerate() {
        let Some(idx) = idx else { continue };
        if cleared[i] {
            continue;
        }
        let sharing: Vec<usize> = (0..slots.len())
            .filter(|&j| resolved[j] == Some(*idx))
            .collect();
        if sharing.len() < 2 {
            continue;
        }
        let names: Vec<&str> = sharing.iter().map(|&j| slots[j].0).collect();
        warnings.push(format!(
            "inits_from_nca: {} all resolve to theta {}; none of them is written, \
             so {} keeps its model default",
            names.join(", "),
            model.default_params.theta_names[*idx],
            model.default_params.theta_names[*idx],
        ));
        for j in sharing {
            cleared[j] = true;
        }
    }
    for (slot, clear) in slots.iter_mut().zip(cleared) {
        if clear {
            *slot.1 = None;
        }
    }
}

/// Route pooled NCA estimates to theta slots and return updated ModelParameters.
fn build_params(
    model: &CompiledModel,
    pop: &PopNca,
    warnings: &mut Vec<String>,
) -> ModelParameters {
    let mut params = model.default_params.clone();

    // Resolve all theta indices up-front (no closures capturing params).
    let mut cl_idx = find_theta_for_slot(model, PK_IDX_CL);
    let mut v_idx = find_theta_for_slot(model, PK_IDX_V);
    let mut ka_idx = find_theta_for_slot(model, PK_IDX_KA);
    let mut q_idx = find_theta_for_slot(model, PK_IDX_Q);
    let mut v2_idx = find_theta_for_slot(model, PK_IDX_V2);
    let mut lagtime_idx = find_theta_for_slot(model, PK_IDX_LAGTIME);
    // F is never written, only read for the bioavailability scale below, so it
    // takes part in the guard through a copy: a collision must stop the writes
    // to F's θ without also dropping the F_default that rescales a separate
    // V/F onto true V.
    let f_idx = find_theta_for_slot(model, PK_IDX_F);
    let mut f_guard_idx = f_idx;
    let mut q3_idx = find_theta_for_slot(model, PK_IDX_Q3);
    let mut v3_idx = find_theta_for_slot(model, PK_IDX_V3);
    drop_shared_theta_indices(
        model,
        &mut [
            ("CL", &mut cl_idx),
            ("V/V1", &mut v_idx),
            ("Ka", &mut ka_idx),
            ("Q", &mut q_idx),
            ("V2", &mut v2_idx),
            ("LAGTIME", &mut lagtime_idx),
            ("F", &mut f_guard_idx),
            ("Q3", &mut q3_idx),
            ("V3", &mut v3_idx),
        ],
        warnings,
    );
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
    /// Every parameter carries its own η, so each slot resolves through layer 1.
    /// The #1922 tests below take the η off `V` (or rename it) to reach the
    /// other routes on the same model: `TVCL` contains `V`, which the old
    /// substring fallback took for a match.
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

    /// #1922: `V` resolves to `TVV`, not to the earlier `TVCL` that merely
    /// *contains* `V`, along both routes that used to reach the substring search:
    ///
    /// * **no η at all** — `V = TVV`, layer 2's whole-word match;
    /// * **an η whose name does not match the parameter** —
    ///   `V = TVV * exp(ETA_VOL)`. Layer 1 now finds the η through the parameter
    ///   it appears in (`eta_param_info`), so the name `ETA_VOL` no longer has to
    ///   end in `_V`.
    ///
    /// Before the fix both returned `TVCL` for `PK_IDX_V` — the same index as
    /// `PK_IDX_CL`.
    #[test]
    fn nca_resolves_v_to_tvv_not_to_a_tvcl_containing_v() {
        let no_eta = allometric_cl_intermediate_first()
            .replace("  omega ETA_V ~ 0.04\n", "")
            .replace("  V = TVV * exp(ETA_V)\n", "  V = TVV\n");
        let other_eta_name = allometric_cl_intermediate_first()
            .replace("  omega ETA_V ~ 0.04\n", "  omega ETA_VOL ~ 0.04\n")
            .replace("  V = TVV * exp(ETA_V)\n", "  V = TVV * exp(ETA_VOL)\n");
        for (label, src) in [("no eta", no_eta), ("eta name misses", other_eta_name)] {
            let m = crate::parser::model_parser::parse_model_string(&src)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            // The premise: the η-name rule cannot resolve V, and the first θ whose
            // name contains `V` is `TVCL` — what the substring search answered.
            assert!(
                !m.mu_refs.keys().any(|e| eta_name_matches(e, "V")),
                "{label}: no eta may be named after V, or this tests the old route"
            );
            assert_eq!(
                m.default_params
                    .theta_names
                    .iter()
                    .find(|n| n.contains('V'))
                    .map(String::as_str),
                Some("TVCL"),
                "{label}: TVCL must be the first theta containing `V`"
            );
            let cl = find_theta_for_slot(&m, PK_IDX_CL).expect("CL theta");
            let v = find_theta_for_slot(&m, PK_IDX_V).expect("V theta");
            assert_eq!(m.default_params.theta_names[cl], "TVCL", "{label}: CL");
            assert_eq!(
                m.default_params.theta_names[v], "TVV",
                "{label}: #1922: V must resolve to TVV, not to TVCL"
            );
            assert_ne!(cl, v, "{label}: the two slots must not share a theta");
        }
        // The η route resolves V's omega too, which the η-name rule also missed.
        let m = crate::parser::model_parser::parse_model_string(
            &allometric_cl_intermediate_first()
                .replace("  omega ETA_V ~ 0.04\n", "  omega ETA_VOL ~ 0.04\n")
                .replace("  V = TVV * exp(ETA_V)\n", "  V = TVV * exp(ETA_VOL)\n"),
        )
        .expect("parses");
        let om = find_omega_idx_for_slot(&m, PK_IDX_V).expect("V omega");
        assert_eq!(m.default_params.omega.eta_names[om], "ETA_VOL");

        // The owning-parameter route alone: with the θ renamed `THETA_VOL`, which
        // no name search can match to `V`, only the η found through `V` itself
        // reaches it — the case above would also be answered by layer 2.
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
            Lookup::Missing,
            "premise: no theta name matches V"
        );
        assert_eq!(
            find_theta_for_slot(&m, PK_IDX_V).map(|i| m.default_params.theta_names[i].as_str()),
            Some("THETA_VOL"),
            "V must resolve through the eta it carries, whatever either is called"
        );
    }

    /// Layer 1's η-name fallback, alone: a model whose `eta_param_info` is empty
    /// (as a hand-built `CompiledModel` may be) still resolves `V` through an η
    /// named `ETA_V`, and declines rather than guesses when two η names qualify
    /// with different θ.
    #[test]
    fn layer_one_eta_name_fallback_resolves_and_declines_on_ambiguity() {
        let mut m =
            crate::parser::model_parser::parse_model_string(&allometric_cl_intermediate_first())
                .expect("parses");
        m.eta_param_info.clear();
        // Rename TVV so neither layer 2 nor layer 3 could answer for V: only the
        // η-name route reaches it.
        let tvv = m
            .default_params
            .theta_names
            .iter()
            .position(|n| n == "TVV")
            .unwrap();
        m.default_params.theta_names[tvv] = "THETA_VOL".into();
        m.mu_refs.get_mut("ETA_V").unwrap().theta_name = "THETA_VOL".into();
        assert_eq!(
            find_theta_for_slot(&m, PK_IDX_V).map(|i| m.default_params.theta_names[i].as_str()),
            Some("THETA_VOL"),
            "the eta-name rule must resolve V through ETA_V"
        );
        assert_eq!(
            find_omega_idx_for_slot(&m, PK_IDX_V)
                .map(|i| m.default_params.omega.eta_names[i].as_str()),
            Some("ETA_V")
        );
        // A second η qualifying for `V`, mu-referenced to another θ: ambiguous.
        let mut second = m.mu_refs["ETA_V"].clone();
        second.theta_name = "TVKA".into();
        m.mu_refs.insert("IIV_V".into(), second);
        assert_eq!(find_theta_for_slot(&m, PK_IDX_V), None);
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
            Lookup::Found(1)
        );
        // Two at the best rank: ambiguous.
        assert_eq!(
            best_theta_name_match(&names(&["TV_CL", "THCL"]), "CL"),
            Lookup::Ambiguous
        );
        assert_eq!(
            best_theta_name_match(&names(&["TVCL", "TVKA"]), "V"),
            Lookup::Missing
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
        assert_eq!(best_theta_name_match(names, "VOL"), Lookup::Ambiguous);
        let tvv = names.iter().position(|n| n == "TVV").unwrap();
        assert_eq!(
            best_theta_name_match(names, "V"),
            Lookup::Found(tvv),
            "premise: layer 3 has a weaker answer to fall through to"
        );
        assert_eq!(find_theta_for_slot(&m, PK_IDX_V), None);
    }

    /// The same for layer 1: a parameter whose own etas reach two different θ
    /// declines, although the η-name rule (`ETA_V`) and layer 2 (`TVV`) would
    /// each answer. The second attribution is injected by hand — it stands for
    /// any parameter carrying two mu-referenced etas.
    #[test]
    fn an_ambiguous_owned_eta_declines_instead_of_falling_through() {
        let mut m =
            crate::parser::model_parser::parse_model_string(&allometric_cl_intermediate_first())
                .expect("parses");
        let mut extra = m
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == "ETA_KA")
            .expect("ETA_KA info")
            .clone();
        extra.individual_param_name = "V".into();
        m.eta_param_info.push(extra);
        assert_eq!(
            m.mu_refs["ETA_KA"].theta_name, "TVKA",
            "premise: the injected eta reaches a different theta"
        );
        assert_eq!(find_theta_for_slot(&m, PK_IDX_V), None);
        assert_eq!(find_omega_idx_for_slot(&m, PK_IDX_V), None);
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
        // The premise: `ETA_CL` is mu-referenced to THS and attributed to S.
        assert_eq!(m.mu_refs["ETA_CL"].theta_name, "THS");
        assert!(m
            .eta_param_info
            .iter()
            .any(|i| i.eta_name == "ETA_CL" && i.individual_param_name == "S"));
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
    #[test]
    ///
    /// The guard blocks writes only: F's default still rescales the apparent
    /// V/F onto `TVV` (review round 2). The control differs in one thing — `CL`
    /// on its own θ — and `TVV` must come out identical, where clearing F's index
    /// had made it V/F, twice the right value at F = 0.5.
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

    /// #1930's second shape, two η names qualifying for one parameter by the
    /// η-name rule — `ETA_CL` on the clearance and `IIV_CL` on an unbound `S`.
    /// Layer 1 used to take whichever the `HashMap` yielded first (measured a
    /// 10/10 split over 20 launches between `TVCL` and `THS`). Since #1922 it
    /// finds the η through the parameter it appears in, so `CL` reaches `ETA_CL`
    /// and `TVCL` in every hash order; the η-name rule is now only a fallback,
    /// and it declines when ambiguous
    /// (`layer_one_eta_name_fallback_resolves_and_declines_on_ambiguity`).
    #[test]
    fn nca_mu_ref_match_follows_the_owning_parameter_when_two_eta_names_qualify() {
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
        // The owning parameter decides, not the hash order.
        let got = find_theta_for_slot(&m, PK_IDX_CL)
            .map(|i| m.default_params.theta_names[i].clone())
            .expect("a theta");
        assert_eq!(got, "TVCL");
        let om = find_omega_idx_for_slot(&m, PK_IDX_CL).expect("CL omega");
        assert_eq!(m.default_params.omega.eta_names[om], "ETA_CL");
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

    /// #1922 end to end: renaming V's η from `ETA_V` to `ETA_VOL` — the one
    /// difference between the two models — must not change what NCA writes.
    ///
    /// Before the fix the renamed model resolved `PK_IDX_V` to `TVCL`, so the
    /// pooled volume (98.8 on `data/two_cpt_oral_cov.csv`) overshot `TVCL`'s upper
    /// bound and landed as the clamp `TVCL = 10.0`, with `TVV` left at its default.
    #[test]
    fn nca_writes_the_same_thetas_whatever_v_eta_is_called() {
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
        let renamed = allometric_cl_intermediate_first()
            .replace("  omega ETA_V ~ 0.04\n", "  omega ETA_VOL ~ 0.04\n")
            .replace("  V = TVV * exp(ETA_V)\n", "  V = TVV * exp(ETA_VOL)\n");
        let ((cl, _, _), (v, _, _)) = run(&renamed);
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
}
