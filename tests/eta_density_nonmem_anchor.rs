//! NONMEM reduction anchors for the omega-line random-effect densities (#1716, #1721
//! review finding 4).
//!
//! NONMEM has no non-Gaussian random-effect prior, but the same marginal model can be
//! written in it with a standard-normal η (`$OMEGA 1 FIX`) read through a map:
//!
//! | stream | ferx | NONMEM `$PK` | comparison |
//! |---|---|---|---|
//! | `eta_density_lognormal` | `omega ETA_CL ~ 0.2 density lognormal`, `CL = TVCL * ETA_CL` | `CL = TVCL*EXP(SQRT(S)*ETA(1) − S/2)`, `S = ln 1.2` | FOCEI, `MAXEVAL=0`: an **affine** map of η, so the Laplace / FOCEI objective is the same function — to arithmetic |
//! | `eta_density_logistic` | `omega ETA_CL ~ 0.2 density logistic`, `CL = TVCL * exp(ETA_CL)` | `X = B*LOG(P/(1−P))`, `P = PHI(ETA(1))` | IMP, `EONLY=1`: not affine, so only the exact marginal agrees — within Monte Carlo error |
//!
//! The first reduces to an object already anchored: its NONMEM side is an ordinary
//! Gaussian model, which ferx evaluates the same way it evaluates every other FOCEI
//! anchor in `nonmem_anchor/`. So `the_lognormal_density_is_its_affine_gaussian_twin`
//! runs that twin *inside ferx* on every PR: the density path against the Gaussian path
//! on one dataset, at one parameter vector. The NONMEM numbers themselves are pending a
//! run of the two streams in `nonmem:7.6.0-anchor` (not available where this was
//! written); `the_density_streams_match_nonmem` is ignored until their outputs are
//! committed to `nonmem_anchor/results/`.

use std::path::Path;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{FitOptions, FitResult};
use ferx_core::{fit, read_nonmem_csv};

const DATA: &str = "nonmem_anchor/eta_shape.csv";

/// The anchor model of `eta_shape_null.ctl` with `omega_cl` and `CL` replaced.
fn model(omega_cl: &str, cl: &str, extra_theta: &str) -> String {
    format!(
        "
[parameters]
  theta TVCL(2.0, 0.01, 100.0)
  theta TVV(20.0, 0.1, 1000.0)
{extra_theta}
  {omega_cl}
  omega ETA_V ~ 0.1
  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = {cl}
  V = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"
    )
}

/// The FOCEI evaluation at the declared parameters (`outer_maxiter = 0` is `MAXEVAL=0`).
fn evaluate(src: &str) -> FitResult {
    let parsed = parse_full_model(src).expect("the ferx model must parse");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("the dataset must load");
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let r = fit(&parsed.model, &pop, &parsed.model.default_params, &options)
        .expect("the evaluation must run");
    assert!(r.ofv.is_finite(), "OFV is not finite");
    r
}

/// `density lognormal` (mean 1, variance ω = 0.2) against the Gaussian stream NONMEM runs,
/// `TVCL·exp(√S·η − S/2)`, η ~ N(0, 1) FIX, `S = ln(1 + ω)`: ferx's latent is
/// `z = √S·η − S/2`, an affine map, under which the FOCEI objective is invariant (the
/// prior's `−2 ln |a|` and the curvature's `+2 ln |a|` cancel). So the OFVs agree to the
/// inner optimiser's tolerance and every subject's EBE maps onto the other's.
/// Dies under: a wrong latent mean (`−S/2`) or variance (`S = ln(1 + ω)`, not ω) in the
/// MvLogNormal / log-normal prior, or its curvature left out of `H̃`.
#[test]
fn the_lognormal_density_is_its_affine_gaussian_twin() {
    let s = 0.2f64.ln_1p();
    let density = evaluate(&model(
        "omega ETA_CL ~ 0.2 density lognormal",
        "TVCL * ETA_CL",
        "",
    ));
    let twin = evaluate(&model(
        "omega ETA_CL ~ 1.0 FIX",
        "TVCL * exp(sqrt(S) * ETA_CL - S / 2)",
        &format!("  theta S({s:?}, 0.0001, 10.0, FIX)"),
    ));
    let gap = (density.ofv - twin.ofv).abs();
    eprintln!(
        "lognormal density OFV {} vs affine Gaussian twin {}: |Δ| {gap:.2e}",
        density.ofv, twin.ofv
    );
    let mut worst_eta = 0.0f64;
    for (d, t) in density.subjects.iter().zip(&twin.subjects) {
        let (z, eta) = (d.eta[0], t.eta[0]);
        let mapped = (z + s / 2.0) / s.sqrt();
        assert!(
            mapped.is_finite() && eta.is_finite(),
            "subject {}: {mapped} vs {eta}",
            d.id
        );
        worst_eta = worst_eta.max((mapped - eta).abs());
    }
    eprintln!("worst |(z + S/2)/√S − η| {worst_eta:.2e}");
    assert!(gap < OFV_TOL, "|Δ OFV| {gap:e}");
    assert!(worst_eta < ETA_TOL, "worst EBE gap {worst_eta:e}");
}
// Measured: |Δ OFV| 5.08e-8 and worst EBE gap 3.94e-8 (the inner optimiser's
// tolerance, the two parameterisations converging separately); bounds 20× / 25×.
const OFV_TOL: f64 = 1e-6;
const ETA_TOL: f64 = 1e-6;

/// The two streams against NONMEM 7.6.0. Pending: their outputs are not yet in
/// `nonmem_anchor/results/` (no NONMEM where they were written). Once they are, this
/// compares the lognormal OFV to arithmetic and the logistic IMP −2LL within the
/// across-seed SD, as `warfarin_iov_imp_estimation_nonmem` does.
#[test]
#[ignore = "pending: run nonmem_anchor/eta_density_{lognormal,logistic}.ctl in nonmem:7.6.0-anchor and commit results/"]
fn the_density_streams_match_nonmem() {
    for file in ["eta_density_lognormal.ext", "eta_density_logistic.ext"] {
        let path = Path::new("nonmem_anchor/results").join(file);
        assert!(path.exists(), "{} must be committed", path.display());
    }
}
