//! NONMEM 7.5.1 `$COVARIANCE MATRIX=R` cross-check for the FOCEI covariance step
//! — issues #209 / #196 / #129 / #224.
//!
//! Guards the covariance-step rewrite that:
//!   1. reconverges the inner EBE loop at every FD perturbation point (NONMEM's
//!      behaviour) — holding the EBEs fixed gave an *indefinite* Hessian on this
//!      well-conditioned surface, which clipped eigenvalues (#129) and inflated
//!      the theta/sigma SEs 30–90×;
//!   2. builds the Hessian as a central finite difference of the analytical
//!      population gradient (#209), whose θ part reuses H-matrix columns for the
//!      mu-referenced parameters (#196);
//!   3. returns `cov = 2·H⁻¹` — the objective is −2·logL, so its Hessian is twice
//!      the observed information (without the factor of two every SE is 1/√2 too
//!      small).
//!
//! ## NONMEM reference
//!
//! NONMEM 7.5.1, FOCEI (`$ESTIMATION METHOD=1 INTER`), `$COVARIANCE MATRIX=R`,
//! on `data/warfarin.csv` (10 subjects, 1-cpt oral, proportional error). The
//! proportional error is coded as `W = THETA(4)*IPRED; Y = IPRED + W*EPS(1)` with
//! `$SIGMA 1 FIX`, so THETA(4) is the proportional SD and its SE is on the SD
//! scale — directly comparable to ferx's `sigma PROP_ERR ~ … (sd)`. SEs are the
//! `.ext` row at `ITERATION = -1000000001`.
//!
//! This test is `#[ignore]`d outside the `slow-tests` feature (it runs a fit to
//! convergence). The band is 20% relative: tight enough to catch the factor-of-2
//! (29%) and the indefinite-Hessian blow-up (orders of magnitude), loose enough
//! to tolerate the AD-vs-FD-Jacobian build difference and the FD-step truncation.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

// NONMEM 7.5.1 FOCEI $COVARIANCE MATRIX=R standard errors (.ext, ITER=-1000000001).
const NM_SE_TVCL: f64 = 7.09746e-3;
const NM_SE_TVV: f64 = 2.40053e-1;
const NM_SE_TVKA: f64 = 1.48649e-1;
const NM_SE_PROP_SD: f64 = 8.35411e-4; // THETA(4): proportional SD-scale SE
const NM_SE_OMEGA_CL: f64 = 1.27933e-2;
const NM_SE_OMEGA_V: f64 = 4.30540e-3;
const NM_SE_OMEGA_KA: f64 = 1.50376e-1;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored covariance SE cross-check (#209/#196/#129): opt in with --features slow-tests"
)]
fn covariance_se_matches_nonmem() {
    let model_src = r"
[parameters]
  theta TVCL(0.15, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.2, 0.01, 50.0)
  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  mu_referencing = true
";

    let model = parse_model_string(model_src).expect("warfarin model parses");
    let pop =
        read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data loads");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = true;
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("warfarin FOCEI fit runs");

    // The covariance step must succeed (the indefinite fixed-EBE Hessian used to
    // fail or clip here).
    assert!(
        result.covariance_matrix.is_some(),
        "covariance step must produce a matrix"
    );
    let se_theta = result.se_theta.as_ref().expect("theta SEs present");
    let se_omega = result.se_omega.as_ref().expect("omega SEs present");
    let se_sigma = result.se_sigma.as_ref().expect("sigma SEs present");

    // (name, ferx SE, NONMEM SE)
    let checks = [
        ("TVCL", se_theta[0], NM_SE_TVCL),
        ("TVV", se_theta[1], NM_SE_TVV),
        ("TVKA", se_theta[2], NM_SE_TVKA),
        ("PROP_ERR", se_sigma[0], NM_SE_PROP_SD),
        ("omega_CL", se_omega[0], NM_SE_OMEGA_CL),
        ("omega_V", se_omega[1], NM_SE_OMEGA_V),
        ("omega_KA", se_omega[2], NM_SE_OMEGA_KA),
    ];

    let tol = 0.20; // 20% relative band — see module docs
    for (name, ferx_se, nm_se) in checks {
        let rel = (ferx_se - nm_se).abs() / nm_se;
        assert!(
            ferx_se.is_finite() && rel < tol,
            "SE({name}) = {ferx_se:.6} vs NONMEM {nm_se:.6} — relative diff {:.1}% exceeds {:.0}% band",
            rel * 100.0,
            tol * 100.0
        );
    }
}
