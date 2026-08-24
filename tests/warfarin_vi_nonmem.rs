//! NONMEM 7.5.1 `METHOD=COND INTERACTION` (FOCEI) cross-check for `method = vi`
//! — Anchor D of `VI_VALIDATION.md`, and the CLAUDE.md "compare with NONMEM"
//! deliverable for the VI estimator.
//!
//! Runs ferx FOCEI, AGQ(9) and VI on warfarin and asserts all three land on
//! NONMEM's FOCEI reference for the same model and data. Gated behind
//! `slow-tests`: three convergence fits, skipped in the default PR job.
//!
//! ## Where the reference comes from
//!
//! `tests/nonmem/warfarin_imp.ctl` chains `$EST METHOD=COND INTERACTION` →
//! `$EST METHOD=IMP`. Its **first** step is FOCEI on exactly this model
//! (`ADVAN2 TRANS2`, lognormal `η` on CL/V/KA, proportional error, the same
//! initial estimates as `examples/warfarin.ferx`), so the licensed run already
//! committed for the IMP anchor carries the FOCEI column too — `TABLE NO. 1` of
//! `warfarin_imp.ext`, `#METH: First Order Conditional Estimation with
//! Interaction` in the `.lst`. `tests/warfarin_imp_nonmem.rs` asserts the
//! *second* step (`TABLE NO. 2`); this file asserts the first.
//!
//! | Parameter | NONMEM FOCEI | ferx FOCEI | ferx AGQ(9) | ferx VI |
//! |-----------|-------------:|-----------:|------------:|--------:|
//! | TVCL      | 0.132695     | 0.132695   | 0.132687    | 0.132693 |
//! | TVV       | 7.73771      | 7.73771    | 7.73746     | 7.73775  |
//! | TVKA      | 0.810796     | 0.81080    | 0.81090     | 0.81092  |
//! | ω²(CL)    | 0.0285884    | 0.028590   | 0.028592    | 0.028592 |
//! | ω²(V)     | 0.00959179   | 0.009592   | 0.009592    | 0.009587 |
//! | ω²(KA)    | 0.335880     | 0.335871   | 0.336036    | 0.335997 |
//! | σ (SD)    | 0.0105651    | 0.010565   | 0.010565    | 0.011175 |
//! | OFV       | −286.004219  | −286.004220 | −285.977   | −285.519 |
//!
//! ferx FOCEI reproduces NONMEM's FOCEI **to six decimal places on the OFV**
//! (−286.004220 against −286.004219) and to 5–6 significant figures on every
//! parameter, which is what makes this anchor worth having for VI: it pins the
//! shared plumbing — predictor, dose bookkeeping, residual model, objective
//! convention — so a VI disagreement cannot be blamed on any of them.
//!
//! ## What the VI arm does and does not claim
//!
//! VI is a different approximation, so this is placement, not identity: it
//! asserts VI lands in NONMEM's basin (θ to well under 1%, `Ω` to ~0.1%, the
//! Laplace objective within a few units). `σ` gets the loosest band — VI's is
//! ~6% high, the Monte-Carlo floor of the convergence rule documented in
//! `VI_VALIDATION.md` §4.11a — and `vi_final_ofv = laplace` is what makes the
//! objective comparable at all. The bound itself (`−2·ELBO`) is never compared
//! with an OFV across engines; that is the mistake the whole document is about.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{
    fit, read_nonmem_csv, CompiledModel, EstimationMethod, FitOptions, Population, ViFinalOfv,
};
use nalgebra::DMatrix;
use std::path::Path;

// NONMEM 7.5.1 FOCEI reference — `warfarin_imp.ext`, TABLE NO. 1, final iteration.
const NM_TVCL: f64 = 1.32695e-01;
const NM_TVV: f64 = 7.73771e+00;
const NM_TVKA: f64 = 8.10796e-01;
const NM_OMEGA_CL: f64 = 2.85884e-02;
const NM_OMEGA_V: f64 = 9.59179e-03;
const NM_OMEGA_KA: f64 = 3.35880e-01;
// $SIGMA is the proportional *variance*; ferx reports the SD. sqrt(1.11621e-4).
const NM_SIGMA_SD: f64 = 0.010565084;
const NM_OFV: f64 = -286.00421948870667;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-8)
}

/// `FitResult.omega` is a dense matrix; the anchor compares its diagonal.
fn omega_diag(omega: &DMatrix<f64>) -> [f64; 3] {
    [omega[(0, 0)], omega[(1, 1)], omega[(2, 2)]]
}

fn warfarin() -> (CompiledModel, Population) {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    (model, population)
}

/// Assert one fit against the NONMEM FOCEI column, with per-block tolerances.
fn assert_matches_nonmem(
    label: &str,
    theta: &[f64],
    omega_diag: &[f64],
    sigma_sd: f64,
    ofv: f64,
    tol_theta: f64,
    tol_omega: f64,
    tol_sigma: f64,
    tol_ofv: f64,
) {
    for (name, got, want) in [
        ("TVCL", theta[0], NM_TVCL),
        ("TVV", theta[1], NM_TVV),
        ("TVKA", theta[2], NM_TVKA),
    ] {
        let e = rel(got, want);
        assert!(
            e < tol_theta,
            "{label}: {name} = {got:.6} against NONMEM's {want:.6} ({:.3}% off, tol {:.1}%)",
            e * 100.0,
            tol_theta * 100.0
        );
    }
    for (name, got, want) in [
        ("omega2_CL", omega_diag[0], NM_OMEGA_CL),
        ("omega2_V", omega_diag[1], NM_OMEGA_V),
        ("omega2_KA", omega_diag[2], NM_OMEGA_KA),
    ] {
        let e = rel(got, want);
        assert!(
            e < tol_omega,
            "{label}: {name} = {got:.6} against NONMEM's {want:.6} ({:.3}% off, tol {:.1}%)",
            e * 100.0,
            tol_omega * 100.0
        );
    }
    let e = rel(sigma_sd, NM_SIGMA_SD);
    assert!(
        e < tol_sigma,
        "{label}: sigma (SD) = {sigma_sd:.6} against NONMEM's {NM_SIGMA_SD:.6} \
         ({:.2}% off, tol {:.1}%)",
        e * 100.0,
        tol_sigma * 100.0
    );
    let gap = (ofv - NM_OFV).abs();
    assert!(
        gap < tol_ofv,
        "{label}: OFV = {ofv:.6} against NONMEM's {NM_OFV:.6} (gap {gap:.6}, tol {tol_ofv})"
    );
}

/// ferx FOCEI against NONMEM `METHOD=COND INTERACTION` — the plumbing anchor.
///
/// Tolerances are tight on purpose: the measured agreement is 5–6 significant
/// figures, so 0.1% on every parameter and 0.01 on the OFV sit far above the
/// observed error while still failing on any real drift in the predictor, the
/// dose bookkeeping, the residual model, or the objective's constants.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: full warfarin FOCEI convergence fit; opt in with --features slow-tests"
)]
fn ferx_focei_matches_nonmem_focei_on_warfarin() {
    let (model, population) = warfarin();
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 300,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let f = fit(&model, &population, &model.default_params, &opts).expect("FOCEI fit converges");
    assert_matches_nonmem(
        "ferx FOCEI",
        &f.theta,
        &omega_diag(&f.omega),
        f.sigma[0],
        f.ofv,
        0.001,
        0.001,
        0.001,
        0.01,
    );
}

/// AGQ(9) against the same reference. AGQ integrates the marginal rather than
/// approximating it, so this is not expected to be identical to FOCEI — it is
/// the arbiter both FOCEI and VI are read against in `VI_VALIDATION.md` §4.14,
/// and it has to sit on NONMEM's basin for that role to mean anything.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: full warfarin AGQ convergence fit; opt in with --features slow-tests"
)]
fn ferx_agq_matches_nonmem_focei_on_warfarin() {
    let (model, population) = warfarin();
    let opts = FitOptions {
        method: EstimationMethod::Laplace,
        n_agq: 9,
        outer_maxiter: 300,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let f = fit(&model, &population, &model.default_params, &opts).expect("AGQ fit converges");
    assert_matches_nonmem(
        "ferx AGQ(9)",
        &f.theta,
        &omega_diag(&f.omega),
        f.sigma[0],
        f.ofv,
        0.005,
        0.005,
        0.005,
        0.5,
    );
}

/// VI against the same reference — Anchor D's actual deliverable.
///
/// Placement, not identity: VI is a different approximation, so the bands are
/// the measured ones plus headroom. `σ` carries 15% because VI's is ~6% high
/// (the Monte-Carlo floor of §4.11a, not a defect in the objective), and the
/// OFV band is 3 units on the same reasoning as
/// `tests/vi.rs::vi_recovers_the_agq_solution_on_warfarin`: observed ~0.6,
/// with the retracted regression at 11.3.
///
/// # Why `vi_iters` is raised
///
/// This model starts where NONMEM's control stream starts (`TVCL 0.2, TVV 10,
/// TVKA 1.5, σ 0.02`), which is deliberate — same model *and* same initial
/// estimates — but it is a long way from the optimum for a stochastic optimizer.
/// From there VI needs **~34 250 iterations** to settle, above the 25 000
/// default: at the default it stops on the ceiling with `−2·ELBO ≈ −240` and
/// `TVCL` 4.9% high, and no assertion here would be measuring the estimator.
/// Given the room it converges by its own criterion and lands on the anchor
/// (`TVCL 0.132693` against NONMEM's `0.132695`). The `tests/vi.rs` fixture
/// starts at `0.13 / 8 / 1.0` and needs ~17 500, which is why this is the first
/// place the ceiling shows up.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: full warfarin VI convergence fit; opt in with --features slow-tests"
)]
fn ferx_vi_matches_nonmem_focei_on_warfarin() {
    let (model, population) = warfarin();
    let opts = FitOptions {
        method: EstimationMethod::Vi,
        vi_mc_samples: 128,
        vi_iters: 60_000,
        vi_final_ofv: ViFinalOfv::Laplace,
        vi_seed: Some(7),
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let f = fit(&model, &population, &model.default_params, &opts).expect("VI fit converges");
    assert_matches_nonmem(
        "ferx VI",
        &f.theta,
        &omega_diag(&f.omega),
        f.sigma[0],
        f.ofv,
        0.01,
        0.01,
        0.15,
        3.0,
    );

    // The bound stays a bound against the external reference too. `−2·ELBO` is an
    // upper bound on `−2 log L`; NONMEM's FOCEI OFV is an approximation to
    // `−2 log L` at its own optimum, so the bound must sit above it.
    let v = f.vi.as_ref().expect("a VI fit reports its vi block");
    assert!(
        v.neg_two_elbo >= NM_OFV - 1e-6,
        "−2·ELBO = {:.3} is BELOW NONMEM's FOCEI OFV {NM_OFV:.3}, so it is not a bound",
        v.neg_two_elbo
    );
}
