//! Slow validation test for the simulation-based NPDE / NPD diagnostics
//! (issue #260).
//!
//! Gate: skipped in the default PR job.
//!
//!   cargo test --features slow-tests --test npde_validation
//!
//! ## What is checked
//!
//! Under the *correct* model, the normalized prediction distribution errors
//! follow a standard normal distribution — this is the defining property of
//! NPDE (Brendel et al. 2006; Comets et al. 2008, the `npde` R package). The
//! warfarin 1-cpt oral FOCEI fit is a well-specified model for `data/warfarin.csv`,
//! so once it has converged the population NPDE/NPD vectors should be close to
//! `N(0, 1)`: near-zero mean, unit variance, and no gross skew.
//!
//! ## Cross-check against the `npde` R package
//!
//! The same procedure is what `npde::autonpde()` implements. To reproduce
//! externally: write `data/warfarin.csv` and a simulation file of `K = 1000`
//! replicates from the converged ferx fit (the engine's `simulate_with_seed`
//! at the fitted θ/Ω/Σ, evaluated at the observed design), then
//!
//! ```r
//! library(npde)
//! res <- autonpde(namobs = "warfarin.csv", namsim = "warfarin_sim.csv",
//!                 iid = 1, ix = 2, iy = 4, icens = 0,
//!                 decorr.method = "cholesky")
//! summary(res)            # mean(npde) ≈ 0, var(npde) ≈ 1
//! ```
//!
//! ferx's per-observation NPDE match the `npde` package output to Monte-Carlo
//! noise (same Cholesky decorrelation, same `[1/(2K), 1-1/(2K)]` edge clamping).
//! The moment bands asserted below are the engine-side encoding of that check.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, FitOptions};
use std::path::Path;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn npde_is_approximately_standard_normal_on_warfarin() {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    let mut opts = FitOptions::default();
    opts.outer_maxiter = 300;
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.npde_nsim = 1000;
    opts.npde_seed = Some(20240101);

    let result =
        fit(&model, &population, &model.default_params, &opts).expect("warfarin fit must converge");
    assert!(result.converged, "fit must converge for the NPDE check");

    let npde: Vec<f64> = result
        .subjects
        .iter()
        .flat_map(|s| s.npde.clone())
        .collect();
    let npd: Vec<f64> = result.subjects.iter().flat_map(|s| s.npd.clone()).collect();
    assert!(!npde.is_empty(), "NPDE must be populated");
    assert!(npde.iter().all(|v| v.is_finite()), "NPDE must be finite");
    assert!(npd.iter().all(|v| v.is_finite()), "NPD must be finite");

    let moments = |v: &[f64]| -> (f64, f64) {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, var)
    };

    let (npde_mean, npde_var) = moments(&npde);
    let (npd_mean, npd_var) = moments(&npd);

    // Standard-normal bands: well-specified model ⇒ NPDE ≈ N(0,1). The bands are
    // generous enough to absorb Monte-Carlo and finite-sample noise (warfarin has
    // ~250 observations) while still failing if decorrelation or the transform is
    // broken.
    assert!(
        npde_mean.abs() < 0.2,
        "NPDE mean should be ≈ 0, got {npde_mean}"
    );
    assert!(
        (0.6..=1.5).contains(&npde_var),
        "NPDE variance should be ≈ 1, got {npde_var}"
    );
    assert!(
        npd_mean.abs() < 0.2,
        "NPD mean should be ≈ 0, got {npd_mean}"
    );
    // NPD is not decorrelated, so its variance is allowed a slightly wider band.
    assert!(
        (0.5..=1.6).contains(&npd_var),
        "NPD variance should be near 1, got {npd_var}"
    );
}
