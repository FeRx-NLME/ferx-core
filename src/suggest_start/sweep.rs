/// `nca_sweep`: rRMSE grid sweep using population predictions (etas=0).
/// `nca_ebe`:   rRMSE grid sweep using individual EBEs (etas≠0, warm-started).
///
/// Theta indices are resolved via [`crate::suggest_start::find_theta_for_slot`]
/// so user-chosen parameter names and ODE models are handled uniformly.
use nalgebra::DVector;
use rayon::prelude::*;

use crate::api::{model_preds, predict};
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::suggest_start::find_theta_for_slot;
use crate::types::{CompiledModel, ModelParameters, Population};

/// Compute relative root mean squared error between population predictions and
/// observed concentrations, pooled across all subjects.
///
/// rRMSE = sqrt( mean( ((pred - obs) / obs)² ) ) for obs > 0.
fn rrmse(model: &CompiledModel, population: &Population, params: &ModelParameters) -> f64 {
    let preds = predict(model, population, params);

    // Build (pred, obs) pairs — preds are returned in the same subject/time order as
    // population.subjects[i].obs_times.
    let mut sum_sq = 0.0_f64;
    let mut n = 0usize;

    let mut pred_iter = preds.iter();
    for subj in &population.subjects {
        for (&obs, &cens) in subj.observations.iter().zip(subj.cens.iter()) {
            let pred_val = pred_iter.next().map(|p| p.pred).unwrap_or(f64::NAN);
            if cens == 0 && obs > 0.0 && obs.is_finite() && pred_val.is_finite() {
                let rel_err = (pred_val - obs) / obs;
                sum_sq += rel_err * rel_err;
                n += 1;
            }
        }
    }

    if n == 0 {
        f64::INFINITY
    } else {
        (sum_sq / n as f64).sqrt()
    }
}

/// Build a log-space grid of `n_pts` values centred on `centre`,
/// spanning `centre / factor` to `centre * factor`.
fn log_grid(centre: f64, factor: f64, n_pts: usize) -> Vec<f64> {
    if n_pts <= 1 {
        return vec![centre];
    }
    let log_lo = (centre / factor).ln();
    let log_hi = (centre * factor).ln();
    let step = (log_hi - log_lo) / (n_pts - 1) as f64;
    (0..n_pts)
        .map(|i| (log_lo + i as f64 * step).exp())
        .collect()
}

/// Sweep a pair of theta indices jointly over an `n_pts × n_pts` log-space grid.
///
/// For each (a, b) grid point, evaluates rRMSE via `predict()` (etas=0) and
/// returns the params with the best-found values.  Parallelised with rayon.
pub fn sweep_slots(
    model: &CompiledModel,
    population: &Population,
    base: &ModelParameters,
    slot_a: usize,
    slot_b: usize,
    n_pts: usize,
    factor: f64,
    label: &str,
) -> (ModelParameters, Vec<String>) {
    let mut warnings = Vec::new();

    let idx_a = find_theta_for_slot(model, slot_a);
    let idx_b = find_theta_for_slot(model, slot_b);

    let (Some(ia), Some(ib)) = (idx_a, idx_b) else {
        warnings.push(format!(
            "inits_from_nca (nca_sweep): could not locate theta indices for {label} sweep (PK slots {slot_a}/{slot_b}); keeping current estimates"
        ));
        return (base.clone(), warnings);
    };

    let grid_a = log_grid(base.theta[ia], factor, n_pts);
    let grid_b = log_grid(base.theta[ib], factor, n_pts);

    let pairs: Vec<(f64, f64)> = grid_a
        .iter()
        .flat_map(|&a| grid_b.iter().map(move |&b| (a, b)))
        .collect();

    let rrmses: Vec<f64> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let mut trial = base.clone();
            trial.theta[ia] = a.clamp(base.theta_lower[ia], base.theta_upper[ia]);
            trial.theta[ib] = b.clamp(base.theta_lower[ib], base.theta_upper[ib]);
            rrmse(model, population, &trial)
        })
        .collect();

    let best = rrmses
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let (best_a, best_b) = pairs[best];
    let mut result = base.clone();
    result.theta[ia] = best_a.clamp(base.theta_lower[ia], base.theta_upper[ia]);
    result.theta[ib] = best_b.clamp(base.theta_lower[ib], base.theta_upper[ib]);

    (result, warnings)
}

/// Sweep all thetas in `targets` (by theta index) sequentially via 1D coordinate
/// sweeps.  Each theta is swept independently over a log-space grid while the
/// others are held at their current best value; the winner becomes the new base
/// for the next theta.  This is model-agnostic: it works for analytical PK,
/// ODE PK, PD, and PKPD models without any knowledge of parameter semantics.
///
/// `targets` should be the indices of non-fixed thetas that Option A did not
/// write (still at model default) — see [`crate::suggest_start::inits_from_nca`].
pub fn sweep_unwritten_thetas(
    model: &CompiledModel,
    population: &Population,
    base: &ModelParameters,
    targets: &[usize],
    n_pts: usize,
    factor: f64,
) -> (ModelParameters, Vec<String>) {
    if targets.is_empty() {
        return (base.clone(), Vec::new());
    }

    let mut current = base.clone();
    let mut warnings = Vec::new();

    for &idx in targets {
        let centre = current.theta[idx];
        if centre <= 0.0 {
            warnings.push(format!(
                "inits_from_nca (nca_sweep): theta[{idx}] ({name}) has non-positive value {centre}; skipping sweep",
                name = current.theta_names[idx]
            ));
            continue;
        }

        let grid = log_grid(centre, factor, n_pts);
        let rrmses: Vec<f64> = grid
            .par_iter()
            .map(|&val| {
                let mut trial = current.clone();
                trial.theta[idx] = val.clamp(current.theta_lower[idx], current.theta_upper[idx]);
                rrmse(model, population, &trial)
            })
            .collect();

        let best = rrmses
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        current.theta[idx] = grid[best].clamp(current.theta_lower[idx], current.theta_upper[idx]);
    }

    (current, warnings)
}

// ---------------------------------------------------------------------------
// nca_ebe: EBE-based rRMSE (warm-started inner loop)
// ---------------------------------------------------------------------------

/// Compute rRMSE using per-subject empirical Bayes estimates.
///
/// `prev_etas` are warm-start EBEs from the previous grid point in eta_true
/// space (same ordering as `population.subjects`).  Returns (rRMSE, new_etas)
/// so the caller can thread warm-starts through sequential grid traversal.
fn rrmse_ebe(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    prev_etas: Option<&[DVector<f64>]>,
) -> (f64, Vec<DVector<f64>>) {
    // Run inner loop with warm-start. 20 iterations and 1e-3 tolerance are
    // enough for a grid sweep — full inner-loop precision is not needed here,
    // and warm-starting means convergence is typically 3–5 iterations.
    let (eta_hats, _h, _stats, _kappas) =
        run_inner_loop_warm(model, population, params, 20, 1e-3, prev_etas, None, 0);

    // Compute predictions using the per-subject EBEs.
    let mut sum_sq = 0.0_f64;
    let mut n = 0usize;

    for (subj, eta) in population.subjects.iter().zip(eta_hats.iter()) {
        let eta_slice = eta.as_slice();
        let pk_params = (model.pk_param_fn)(&params.theta, eta_slice, &subj.covariates);
        let preds = model_preds(model, subj, &pk_params, &params.theta, eta_slice);

        for (j, (&obs, &cens)) in subj.observations.iter().zip(subj.cens.iter()).enumerate() {
            if cens == 0 && obs > 0.0 && obs.is_finite() {
                let pred_val = preds.get(j).copied().unwrap_or(f64::NAN);
                if pred_val.is_finite() {
                    let rel_err = (pred_val - obs) / obs;
                    sum_sq += rel_err * rel_err;
                    n += 1;
                }
            }
        }
    }

    let rrmse_val = if n == 0 {
        f64::INFINITY
    } else {
        (sum_sq / n as f64).sqrt()
    };

    (rrmse_val, eta_hats)
}

/// EBE-based joint 2D sweep over two theta slots.
///
/// Grid traversal is sequential (row-major) so EBEs from each point warm-start
/// the next; the rayon parallelism is inside `run_inner_loop_warm` (subjects).
pub fn sweep_slots_ebe(
    model: &CompiledModel,
    population: &Population,
    base: &ModelParameters,
    slot_a: usize,
    slot_b: usize,
    n_pts: usize,
    factor: f64,
    label: &str,
) -> (ModelParameters, Vec<String>) {
    let mut warnings = Vec::new();

    let idx_a = find_theta_for_slot(model, slot_a);
    let idx_b = find_theta_for_slot(model, slot_b);

    let (Some(ia), Some(ib)) = (idx_a, idx_b) else {
        warnings.push(format!(
            "inits_from_nca (nca_ebe): could not locate theta indices for {label} sweep; keeping current estimates"
        ));
        return (base.clone(), warnings);
    };

    let grid_a = log_grid(base.theta[ia], factor, n_pts);
    let grid_b = log_grid(base.theta[ib], factor, n_pts);

    let mut best_rrmse = f64::INFINITY;
    let mut best_a = base.theta[ia];
    let mut best_b = base.theta[ib];
    let mut prev_etas: Option<Vec<DVector<f64>>> = None;

    for &a in &grid_a {
        for &b in &grid_b {
            let mut trial = base.clone();
            trial.theta[ia] = a.clamp(base.theta_lower[ia], base.theta_upper[ia]);
            trial.theta[ib] = b.clamp(base.theta_lower[ib], base.theta_upper[ib]);
            let (r, new_etas) = rrmse_ebe(model, population, &trial, prev_etas.as_deref());
            if r < best_rrmse {
                best_rrmse = r;
                best_a = a;
                best_b = b;
            }
            prev_etas = Some(new_etas);
        }
    }

    let mut result = base.clone();
    result.theta[ia] = best_a.clamp(base.theta_lower[ia], base.theta_upper[ia]);
    result.theta[ib] = best_b.clamp(base.theta_lower[ib], base.theta_upper[ib]);
    (result, warnings)
}

/// EBE-based sequential 1D sweeps for each theta in `targets`.
///
/// Warm-starts EBEs from the final grid point of each sweep as the initial
/// estimate for the next sweep.
pub fn sweep_unwritten_thetas_ebe(
    model: &CompiledModel,
    population: &Population,
    base: &ModelParameters,
    targets: &[usize],
    n_pts: usize,
    factor: f64,
) -> (ModelParameters, Vec<String>) {
    if targets.is_empty() {
        return (base.clone(), Vec::new());
    }

    let mut current = base.clone();
    let mut warnings = Vec::new();
    let mut prev_etas: Option<Vec<DVector<f64>>> = None;

    for &idx in targets {
        let centre = current.theta[idx];
        if centre <= 0.0 {
            warnings.push(format!(
                "inits_from_nca (nca_ebe): theta[{idx}] ({name}) has non-positive value {centre}; skipping",
                name = current.theta_names[idx]
            ));
            continue;
        }

        let grid = log_grid(centre, factor, n_pts);
        let mut best_rrmse = f64::INFINITY;
        let mut best_val = centre;

        for &val in &grid {
            let mut trial = current.clone();
            trial.theta[idx] = val.clamp(current.theta_lower[idx], current.theta_upper[idx]);
            let (r, new_etas) = rrmse_ebe(model, population, &trial, prev_etas.as_deref());
            if r < best_rrmse {
                best_rrmse = r;
                best_val = val;
            }
            prev_etas = Some(new_etas);
        }

        current.theta[idx] = best_val.clamp(current.theta_lower[idx], current.theta_upper[idx]);
    }

    (current, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::datareader::read_nonmem_csv;
    use crate::parser::model_parser::parse_model_file;
    use crate::types::{PK_IDX_CL, PK_IDX_Q, PK_IDX_V, PK_IDX_V2};
    use std::path::Path;

    // ── helpers ─────────────────────────────────────────────────────────────

    /// Parse the warfarin (1-cpt oral) model only — for tests that synthesise
    /// their own `Population` (e.g. the empty-observations rRMSE branch).
    fn warfarin_model() -> CompiledModel {
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse")
    }

    /// Load the warfarin (1-cpt oral) fixture used by the suite's other tests.
    /// Small (~30 subjects) and analytical so each sweep iteration is fast.
    fn warfarin() -> (CompiledModel, Population) {
        let model = warfarin_model();
        let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data must load");
        (model, population)
    }

    // ── log_grid ────────────────────────────────────────────────────────────

    #[test]
    fn test_log_grid_endpoints() {
        let grid = log_grid(1.0, 10.0, 9);
        assert_eq!(grid.len(), 9);
        assert!((grid[0] - 0.1).abs() < 1e-10, "first point should be 0.1");
        assert!((grid[8] - 10.0).abs() < 1e-10, "last point should be 10.0");
    }

    #[test]
    fn test_log_grid_single_point() {
        let grid = log_grid(5.0, 10.0, 1);
        assert_eq!(grid, vec![5.0]);
    }

    // ── rrmse ───────────────────────────────────────────────────────────────

    /// rRMSE on a population with no observations must return INFINITY rather
    /// than divide by zero — the sweep then prefers any grid point with valid
    /// observations over an empty one.
    #[test]
    fn test_rrmse_no_observations_returns_infinity() {
        let model = warfarin_model();
        let empty_pop = Population {
            subjects: vec![],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
        };
        let r = rrmse(&model, &empty_pop, &model.default_params);
        assert!(
            r.is_infinite() && r > 0.0,
            "empty population must give +∞ rRMSE, got {r}"
        );
    }

    // ── sweep_slots (etas=0) ────────────────────────────────────────────────

    /// `sweep_slots` with a PK slot the model doesn't have must skip cleanly
    /// (returns the base params unchanged) and emit one warning identifying
    /// the missing slot. Warfarin is 1-cpt so `PK_IDX_Q` is absent.
    #[test]
    fn test_sweep_slots_warns_when_slot_missing() {
        let (model, population) = warfarin();
        let base = model.default_params.clone();
        let (out, warnings) = sweep_slots(
            &model,
            &population,
            &base,
            PK_IDX_CL,
            PK_IDX_Q, // absent from warfarin (1-cpt model)
            9,
            10.0,
            "CL/Q (synthetic)",
        );
        assert_eq!(
            out.theta, base.theta,
            "missing slot must leave base unchanged"
        );
        assert_eq!(
            warnings.len(),
            1,
            "exactly one slot-missing warning expected"
        );
        assert!(
            warnings[0].contains("could not locate theta indices"),
            "warning text must mention missing-slot reason, got: {}",
            warnings[0]
        );
    }

    // ── sweep_unwritten_thetas (etas=0) ─────────────────────────────────────

    /// Empty `targets` must short-circuit to a clone of base with no warnings.
    #[test]
    fn test_sweep_unwritten_thetas_empty_targets_noop() {
        let (model, population) = warfarin();
        let base = model.default_params.clone();
        let (out, warnings) = sweep_unwritten_thetas(&model, &population, &base, &[], 9, 10.0);
        assert_eq!(out.theta, base.theta);
        assert!(warnings.is_empty());
    }

    /// A non-positive theta value (e.g. 0.0) can't be log-grid-swept; the
    /// function must skip it with a warning and continue with the rest.
    #[test]
    fn test_sweep_unwritten_thetas_skips_non_positive() {
        let (model, population) = warfarin();
        let mut base = model.default_params.clone();
        base.theta[0] = 0.0; // force a non-positive theta in slot 0
        let (out, warnings) = sweep_unwritten_thetas(&model, &population, &base, &[0], 9, 10.0);
        assert_eq!(out.theta[0], 0.0, "non-positive theta must not be touched");
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("non-positive"),
            "warning must mention non-positive cause, got: {}",
            warnings[0]
        );
    }

    // ── sweep_slots_ebe (etas≠0) ────────────────────────────────────────────

    /// EBE-variant equivalent of `test_sweep_slots_warns_when_slot_missing`.
    /// `PK_IDX_V2` is absent from the warfarin (1-cpt) model.
    #[test]
    fn test_sweep_slots_ebe_warns_when_slot_missing() {
        let (model, population) = warfarin();
        let base = model.default_params.clone();
        let (out, warnings) = sweep_slots_ebe(
            &model,
            &population,
            &base,
            PK_IDX_Q,
            PK_IDX_V2, // absent from warfarin
            3,
            5.0,
            "Q/V2 (synthetic)",
        );
        assert_eq!(out.theta, base.theta);
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("could not locate theta indices"),
            "warning text must identify the missing-slot cause, got: {}",
            warnings[0]
        );
    }

    /// `sweep_slots_ebe` happy path: 2D grid sweep on CL/V (both present in
    /// warfarin) with a small grid so the inner-loop warm-starts complete
    /// quickly. Asserts the function runs end-to-end and returns valid output;
    /// the no-call-coverage on `sweep_slots_ebe` is the gap this closes (no
    /// existing fixture exercises the 2D EBE pair sweep because peeling fills
    /// CL/V in `nca_with_ebe` upstream).
    #[test]
    fn test_sweep_slots_ebe_runs_2d_grid() {
        let (model, population) = warfarin();
        let base = model.default_params.clone();
        // 3×3 grid keeps the test fast (~1 s for warfarin's 30 subjects).
        let (out, warnings) = sweep_slots_ebe(
            &model,
            &population,
            &base,
            PK_IDX_CL,
            PK_IDX_V,
            3,
            3.0,
            "CL/V",
        );
        assert!(
            warnings.is_empty(),
            "happy path must not warn, got: {warnings:?}"
        );
        // Theta bounds must be respected.
        for (i, &t) in out.theta.iter().enumerate() {
            assert!(
                t >= out.theta_lower[i] && t <= out.theta_upper[i],
                "swept theta[{i}] = {t} outside bounds [{}, {}]",
                out.theta_lower[i],
                out.theta_upper[i]
            );
        }
    }

    // ── sweep_unwritten_thetas_ebe (etas≠0) ─────────────────────────────────

    /// EBE-variant equivalent of `test_sweep_unwritten_thetas_empty_targets_noop`.
    #[test]
    fn test_sweep_unwritten_thetas_ebe_empty_targets_noop() {
        let (model, population) = warfarin();
        let base = model.default_params.clone();
        let (out, warnings) = sweep_unwritten_thetas_ebe(&model, &population, &base, &[], 9, 10.0);
        assert_eq!(out.theta, base.theta);
        assert!(warnings.is_empty());
    }

    /// EBE-variant equivalent of `test_sweep_unwritten_thetas_skips_non_positive`.
    /// Uses a tiny 1-pt grid so even with the warm-started inner loop the test
    /// stays fast — the skip happens before any inner-loop work anyway.
    #[test]
    fn test_sweep_unwritten_thetas_ebe_skips_non_positive() {
        let (model, population) = warfarin();
        let mut base = model.default_params.clone();
        base.theta[0] = -1.0; // negative value triggers the centre <= 0 branch
        let (out, warnings) = sweep_unwritten_thetas_ebe(&model, &population, &base, &[0], 1, 5.0);
        assert_eq!(out.theta[0], -1.0, "non-positive theta must not be touched");
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("non-positive"),
            "warning must mention non-positive cause, got: {}",
            warnings[0]
        );
    }

    /// `sweep_unwritten_thetas` with all-FIX upstream (PK_IDX_Q3 absent from
    /// warfarin) — confirms the path that finds *no* slot among targets after
    /// the upstream filter still returns cleanly. Targets pre-filtered to be
    /// empty would short-circuit via `test_*_empty_targets_noop`; here the
    /// non-positive guard catches a single target that has been zeroed and
    /// also confirms `theta_names` lookup for the warning interpolation.
    #[test]
    fn test_sweep_unwritten_thetas_zero_theta_uses_theta_name() {
        let (model, population) = warfarin();
        let mut base = model.default_params.clone();
        let theta_idx = 0;
        base.theta[theta_idx] = 0.0;
        let theta_name = base.theta_names[theta_idx].clone();
        let (_, warnings) =
            sweep_unwritten_thetas(&model, &population, &base, &[theta_idx], 5, 5.0);
        assert!(
            warnings.iter().any(|w| w.contains(&theta_name)),
            "warning must interpolate the theta name '{theta_name}', got: {warnings:?}"
        );
    }
}
