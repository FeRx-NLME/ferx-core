//! NONMEM 7.5.1 `METHOD=IMP` cross-check for ferx's **estimating** IMP
//! (`method = imp`, NONMEM `METHOD=IMP`).
//!
//! Runs ferx `[focei, imp]` on warfarin and asserts the converged estimates match
//! NONMEM's `METHOD=IMP` reference on the same model/data. This is the
//! cross-engine validation required for a new estimator (CLAUDE.md): IMP is
//! NONMEM's method, so the anchor is NONMEM itself.
//!
//! Gated behind `slow-tests`; skipped in the default PR job, run nightly.
//!
//! ## Why `[focei, imp]`, not standalone `imp`
//!
//! Plain `METHOD=IMP` re-centers its proposal from the *previous* iteration's
//! importance samples (the mode/variance are found only on the first iteration).
//! On rich data — warfarin has ~12 obs/subject, so each conditional posterior of
//! η is razor-sharp — a large early step moves the posterior past the lagged
//! proposal and the ESS collapses. This is the documented weakness of
//! `METHOD=IMP` (the reason NONMEM offers `METHOD=IMPMAP`). Both engines are run
//! warm-started from a FOCE/FOCEI pass, the robust configuration ferx recommends
//! on rich data; the NONMEM control stream (`tests/nonmem/warfarin_imp.ctl`)
//! chains `$EST METHOD=COND INTERACTION` → `$EST METHOD=IMP INTERACTION NITER=100
//! ISAMPLE=1000 SEED=12345`.
//!
//! ## Reference values
//!
//! NONMEM 7.5.1 `METHOD=IMP` (the `Importance Sampling` table in
//! `tests/nonmem/warfarin_imp.ext`/`.lst`), warm-started from `METHOD=COND`:
//!
//! | Parameter | NONMEM IMP | ferx `[focei, imp]` |
//! |-----------|-----------:|--------------------:|
//! | TVCL      | 0.1264     | 0.1327              |
//! | TVV       | 7.723      | 7.737               |
//! | TVKA      | 0.8857     | 0.811               |
//! | ω²(CL)    | 0.03044    | 0.0286              |
//! | ω²(V)     | 0.009586   | 0.0096              |
//! | ω²(KA)    | 0.3405     | 0.336               |
//! | σ (SD)    | 0.01047    | 0.01056             |
//! | OFV       | −285.69    | −286.00             |
//!
//! Both engines start from the same FOCEI basin (NONMEM `METHOD=COND` OFV
//! −286.00, identical to ferx's FOCEI). NONMEM's IMP MCEM then drifts a little
//! off that basin toward the importance-sampled marginal optimum (TVKA up to
//! 0.886, OFV −285.69 on the IMP objective), while ferx's warm-started IMP holds
//! near the FOCEI optimum — both are stable and agree within the cross-engine +
//! Monte-Carlo margin. TVKA is the least-identified parameter on this 10-subject
//! extract (ETA_KA variance ≈ 0.34, high shrinkage), so it carries the loosest
//! band; CL/V and the variance components agree to a few percent. The two OFVs
//! are different objectives (NONMEM's IMP MC objective vs ferx's final Laplace
//! pass) and agree to well within a cross-engine unit.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

// NONMEM 7.5.1 METHOD=IMP reference (see module docstring / .ext TABLE NO. 2).
const NM_TVCL: f64 = 0.12640;
const NM_TVV: f64 = 7.7234;
const NM_TVKA: f64 = 0.88565;
const NM_OMEGA_CL: f64 = 0.030443;
const NM_OMEGA_V: f64 = 0.0095858;
const NM_OMEGA_KA: f64 = 0.34052;
const NM_SIGMA_SD: f64 = 0.010468; // sqrt(1.09576e-4)
const NM_OFV: f64 = -285.685;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-8)
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_imp_matches_nonmem_on_warfarin() {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    let mut opts = FitOptions::default();
    opts.run_covariance_step = false;
    opts.outer_maxiter = 300;
    opts.methods = vec![EstimationMethod::FoceI, EstimationMethod::Imp];
    opts.is_iterations = 100;
    opts.is_samples = 1000;
    opts.is_averaging = 30;
    opts.is_seed = Some(12345);
    let r = fit(&model, &population, &model.default_params, &opts)
        .expect("FOCEI → estimating IMP fit must succeed");

    assert_eq!(
        r.method,
        EstimationMethod::Imp,
        "estimating IMP is the final stage"
    );

    // Thetas: CL/V well-identified (8%), KA poorly-identified (15%). The 8% CL/V
    // band absorbs the genuine FOCEI-vs-IMP-marginal drift between the engines.
    assert!(
        rel(r.theta[0], NM_TVCL) < 0.08,
        "TVCL {} vs NM {NM_TVCL}",
        r.theta[0]
    );
    assert!(
        rel(r.theta[1], NM_TVV) < 0.08,
        "TVV {} vs NM {NM_TVV}",
        r.theta[1]
    );
    assert!(
        rel(r.theta[2], NM_TVKA) < 0.15,
        "TVKA {} vs NM {NM_TVKA}",
        r.theta[2]
    );

    // Variance components (NONMEM prints ~3 sig figs; allow 15%).
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

    // OFV within the cross-engine margin (different objectives — see docstring).
    assert!(
        (r.ofv - NM_OFV).abs() < 3.0,
        "OFV {} vs NONMEM {NM_OFV}",
        r.ofv
    );
}
