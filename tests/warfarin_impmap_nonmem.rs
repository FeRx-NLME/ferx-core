//! NONMEM 7.5.1 `METHOD=IMPMAP` cross-check for ferx's IMPMAP estimator.
//!
//! Runs ferx IMPMAP standalone on warfarin and asserts the converged estimates
//! match NONMEM's `METHOD=IMPMAP` reference on the same model/data. This is the
//! cross-engine validation required for a new estimator (CLAUDE.md): IMPMAP is
//! NONMEM's method, so the anchor is NONMEM itself, not just ferx's FOCEI.
//!
//! Gated behind `slow-tests` (runs the full 150-iteration MCEM); skipped in the
//! default PR job, run nightly via `slow-tests.yml`.
//!
//! ## Reference values
//!
//! From `tests/nonmem/warfarin_impmap.ctl` run on `data/warfarin.csv` with
//! NONMEM 7.5.1 (`METHOD=IMPMAP INTERACTION NITER=200 ISAMPLE=300 SEED=12345`):
//!
//! | Parameter | NONMEM IMPMAP | ferx IMPMAP |
//! |-----------|--------------:|------------:|
//! | TVCL      | 0.135         | 0.1327      |
//! | TVV       | 7.89          | 7.737       |
//! | TVKA      | 0.730         | 0.811       |
//! | ω²(CL)    | 0.0291        | 0.0286      |
//! | ω²(V)     | 0.0101        | 0.0096      |
//! | ω²(KA)    | 0.340         | 0.336       |
//! | σ (SD)    | 0.01034       | 0.01056     |
//! | OFV       | −284.92       | −286.00     |
//!
//! TVKA is the least-identified parameter on this 10-subject extract (ETA_KA
//! variance ≈ 0.34, high shrinkage), so it carries the loosest band; the
//! well-determined CL/V structure and the variance components agree to a few
//! percent. The OFVs are on a comparable scale (NONMEM "without constant" and
//! ferx's Laplace OFV) and agree to ~1 unit, the usual cross-engine margin.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

// NONMEM 7.5.1 METHOD=IMPMAP reference (see module docstring / .ctl).
const NM_TVCL: f64 = 0.135;
const NM_TVV: f64 = 7.89;
const NM_TVKA: f64 = 0.730;
const NM_OMEGA_CL: f64 = 0.0291;
const NM_OMEGA_V: f64 = 0.0101;
const NM_OMEGA_KA: f64 = 0.340;
const NM_SIGMA_SD: f64 = 0.010344; // sqrt(1.07e-4)
const NM_OFV: f64 = -284.917;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-8)
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_impmap_matches_nonmem_impmap_on_warfarin() {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Impmap;
    opts.run_covariance_step = false;
    opts.impmap_iterations = 150;
    opts.impmap_samples = 500;
    opts.impmap_averaging = 50;
    opts.impmap_seed = Some(12345);
    let r =
        fit(&model, &population, &model.default_params, &opts).expect("IMPMAP fit must succeed");

    // Thetas: CL/V well-identified (5%), KA poorly-identified (15%).
    assert!(
        rel(r.theta[0], NM_TVCL) < 0.05,
        "TVCL {} vs NM {NM_TVCL}",
        r.theta[0]
    );
    assert!(
        rel(r.theta[1], NM_TVV) < 0.05,
        "TVV {} vs NM {NM_TVV}",
        r.theta[1]
    );
    assert!(
        rel(r.theta[2], NM_TVKA) < 0.15,
        "TVKA {} vs NM {NM_TVKA}",
        r.theta[2]
    );

    // Variance components (NONMEM prints 3 sig figs; allow 15%).
    assert!(
        rel(r.omega[(0, 0)], NM_OMEGA_CL) < 0.15,
        "omega_CL {}",
        r.omega[(0, 0)]
    );
    assert!(
        rel(r.omega[(1, 1)], NM_OMEGA_V) < 0.15,
        "omega_V {}",
        r.omega[(1, 1)]
    );
    assert!(
        rel(r.omega[(2, 2)], NM_OMEGA_KA) < 0.15,
        "omega_KA {}",
        r.omega[(2, 2)]
    );

    // Residual error (ferx reports the proportional SD).
    assert!(
        rel(r.sigma[0], NM_SIGMA_SD) < 0.10,
        "sigma_SD {} vs NM {NM_SIGMA_SD}",
        r.sigma[0]
    );

    // OFV within the cross-engine margin.
    assert!(
        (r.ofv - NM_OFV).abs() < 3.0,
        "OFV {} vs NONMEM {NM_OFV}",
        r.ofv
    );
}
