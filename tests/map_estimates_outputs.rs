//! Output-shape regression tests for the per-subject EBE / individual-
//! parameter changes.
//!
//! Two contracts pinned here:
//! 1. `CompiledModel::indiv_param_names` mirrors the top-level
//!    `[individual_parameters]` declarations in source order. The downstream
//!    R FFI uses this list to label `fit$individual_estimates` columns.
//! 2. `output::sdtab()` no longer emits `ETA1..ETAn` columns. Per-subject
//!    EBEs live in `fit$ebe_etas` on the R side; sdtab is now strictly
//!    per-observation diagnostic data.

use ferx_core::io::output::sdtab;
use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, FitOptions, Optimizer};
use std::path::Path;

fn warfarin_setup() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
) {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin example must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    (model, population)
}

fn fast_options() -> FitOptions {
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.optimizer = Optimizer::Slsqp;
    opts.outer_maxiter = 30; // enough to populate diagnostics; not chasing convergence
    opts
}

#[test]
fn indiv_param_names_mirrors_warfarin_individual_parameters_block() {
    // Warfarin's [individual_parameters] block declares CL, V, KA in that
    // order. The compiled model must surface the same names so the R FFI
    // can label individual_estimates columns correctly.
    let (model, _) = warfarin_setup();
    assert_eq!(
        model.indiv_param_names,
        vec!["CL".to_string(), "V".to_string(), "KA".to_string()],
        "indiv_param_names must match the [individual_parameters] block in source order"
    );
    // Parallel to pk_indices (used by the FFI to read each value out of
    // the PkParams slot for analytical models).
    assert_eq!(
        model.indiv_param_names.len(),
        model.pk_indices.len(),
        "indiv_param_names and pk_indices must be aligned"
    );
}

#[test]
fn sdtab_omits_eta_columns_after_fit() {
    // Regression: sdtab used to carry ETA1..ETAn columns. They have moved
    // to `ebe_etas` on the R side, so sdtab must no longer surface any
    // column whose name starts with "ETA".
    let (model, population) = warfarin_setup();
    let opts = fast_options();
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("warfarin fit must succeed");

    let cols = sdtab(&result, &population);
    let names: Vec<&str> = cols.iter().map(|(n, _)| n.as_str()).collect();
    let eta_cols: Vec<&&str> = names.iter().filter(|n| n.starts_with("ETA")).collect();
    assert!(
        eta_cols.is_empty(),
        "sdtab must not contain ETA columns; found: {:?}",
        eta_cols
    );

    // Spot-check the per-observation columns we still expect, so a future
    // accidental column drop also fails this test.
    for required in [
        "ID", "TIME", "DV", "PRED", "IPRED", "CWRES", "IWRES", "EBE_OFV", "N_OBS",
    ] {
        assert!(
            names.contains(&required),
            "sdtab missing required column `{}`; have: {:?}",
            required,
            names
        );
    }
}

#[test]
fn sdtab_omits_npde_columns_when_disabled() {
    // Default npde_nsim = 0 → no simulation, no NPDE/NPD columns.
    let (model, population) = warfarin_setup();
    let opts = fast_options();
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("warfarin fit must succeed");

    let cols = sdtab(&result, &population);
    let names: Vec<&str> = cols.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        !names.contains(&"NPDE") && !names.contains(&"NPD"),
        "NPDE/NPD must be absent when npde_nsim = 0; have: {:?}",
        names
    );
}

#[test]
fn sdtab_emits_finite_npde_columns_when_enabled() {
    // With npde_nsim > 0, the post-fit simulation populates NPDE/NPD as
    // per-observation columns; every value must be finite for this well-posed
    // warfarin fit (no censoring, covariance non-degenerate).
    let (model, population) = warfarin_setup();
    let mut opts = fast_options();
    opts.npde_nsim = 500;
    opts.npde_seed = Some(12345);
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("warfarin fit must succeed");

    let cols = sdtab(&result, &population);
    let get = |name: &str| -> Vec<f64> {
        cols.iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| panic!("sdtab must contain `{name}` when npde_nsim > 0"))
    };
    let npde = get("NPDE");
    let npd = get("NPD");

    // Columns are per-observation, aligned with the other diagnostic columns.
    let n_obs_total: usize = result.subjects.iter().map(|s| s.npde.len()).sum();
    assert_eq!(npde.len(), n_obs_total);
    assert_eq!(npd.len(), n_obs_total);
    assert!(n_obs_total > 0);

    assert!(
        npde.iter().all(|v| v.is_finite()),
        "all NPDE values must be finite; got {:?}",
        npde
    );
    assert!(
        npd.iter().all(|v| v.is_finite()),
        "all NPD values must be finite; got {:?}",
        npd
    );

    // The whole-population NPD/NPDE should be roughly mean-zero, unit-variance
    // (standard-normal under a reasonable fit) — a loose sanity band, not a
    // convergence assertion (the fit is deliberately truncated).
    let mean = npde.iter().sum::<f64>() / npde.len() as f64;
    assert!(
        mean.abs() < 1.0,
        "NPDE population mean should be near 0, got {mean}"
    );
}

#[test]
fn npde_is_reproducible_across_runs_with_same_seed() {
    // A fixed npde_seed must give bit-identical NPDE/NPD across fits.
    let (model, population) = warfarin_setup();
    let mut opts = fast_options();
    opts.npde_nsim = 200;
    opts.npde_seed = Some(777);

    let r1 = fit(&model, &population, &model.default_params, &opts).expect("fit 1");
    let r2 = fit(&model, &population, &model.default_params, &opts).expect("fit 2");

    for (s1, s2) in r1.subjects.iter().zip(r2.subjects.iter()) {
        assert_eq!(
            s1.npde, s2.npde,
            "NPDE must be reproducible for a fixed seed"
        );
        assert_eq!(s1.npd, s2.npd, "NPD must be reproducible for a fixed seed");
    }
}
