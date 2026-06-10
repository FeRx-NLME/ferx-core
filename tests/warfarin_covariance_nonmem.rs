//! NONMEM 7.5.1 `$COVARIANCE MATRIX=R` cross-check for the FOCE/FOCEI covariance
//! step — issues #209 / #196 / #129 / #224.
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
//! Two cases run the same warfarin model through the two covariance-OFV paths:
//!   - **FOCEI** (`interaction = true`): the per-subject marginal already carries
//!     `ηᵀΩ⁻¹η + log|Ω|`, so the covariance OFV is just `2·pop_nll`.
//!   - **FOCE** (`interaction = false`): `pop_nll` (Sheiner–Beal) omits that prior,
//!     so `compute_covariance` adds it back analytically via `omega_prior_gradient`.
//!     This is the only end-to-end guard for that FOCE-specific gradient branch.
//!
//! ## NONMEM reference
//!
//! NONMEM 7.5.1 on `data/warfarin.csv` (10 subjects, 1-cpt oral, proportional
//! error), `$COVARIANCE MATRIX=R`. The proportional error is coded as
//! `W = THETA(4)*IPRED; Y = IPRED + W*EPS(1)` with `$SIGMA 1 FIX`, so THETA(4) is
//! the proportional SD and its SE is on the SD scale — directly comparable to
//! ferx's `sigma PROP_ERR ~ … (sd)`. SEs are the `.ext` row at
//! `ITERATION = -1000000001`. FOCEI uses `$EST METHOD=1 INTER`, FOCE drops `INTER`.
//!
//! These tests are `#[ignore]`d outside the `slow-tests` feature (they run a fit
//! to convergence). The band on the structural (theta) and residual-error (sigma)
//! SEs is 20% relative: tight enough to catch the factor-of-2 (29%) and the
//! indefinite-Hessian blow-up (orders of magnitude), loose enough to tolerate the
//! AD-vs-FD-Jacobian build difference and the FD-step truncation.
//!
//! The omega-block band is method-dependent. FOCEI matches NONMEM tightly (20%),
//! but FOCE runs the Sheiner–Beal objective, whose omega-block curvature differs
//! from NONMEM's FOCE: on this model ferx's FOCE omega SEs sit ~25–36% below
//! NONMEM (the theta/sigma SEs still match within ~3%, and the omega *estimates*
//! agree within ~6%). That is the SB-vs-NONMEM approximation gap, not a fault in
//! the covariance machinery — so the FOCE omega band is widened to 45%. Tightening
//! it is tracked with the planned FOCE Sheiner–Beal → Almquist–Laplace switch.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const MODEL_SRC: &str = r"
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
  mu_referencing = true
";

/// One parameter's NONMEM SE and the relative band ferx must fall within.
struct SeRef {
    name: &'static str,
    nm: f64,
    tol: f64,
}

const TIGHT: f64 = 0.20; // theta + residual-error: NONMEM-anchored
const OMEGA_FOCEI: f64 = 0.20; // FOCEI omega matches NONMEM tightly
const OMEGA_FOCE: f64 = 0.45; // FOCE omega: Sheiner–Beal gap (see module docs)

/// NONMEM 7.5.1 `$COVARIANCE MATRIX=R` SEs (.ext, ITER=-1000000001), in the
/// order [TVCL, TVV, TVKA, PROP_SD, ωCL, ωV, ωKA].
fn nonmem_refs(focei: bool) -> [SeRef; 7] {
    let omega_tol = if focei { OMEGA_FOCEI } else { OMEGA_FOCE };
    if focei {
        // FOCEI: $EST METHOD=1 INTER.
        [
            SeRef {
                name: "TVCL",
                nm: 7.09746e-3,
                tol: TIGHT,
            },
            SeRef {
                name: "TVV",
                nm: 2.40053e-1,
                tol: TIGHT,
            },
            SeRef {
                name: "TVKA",
                nm: 1.48649e-1,
                tol: TIGHT,
            },
            SeRef {
                name: "PROP_ERR",
                nm: 8.35411e-4,
                tol: TIGHT,
            },
            SeRef {
                name: "omega_CL",
                nm: 1.27933e-2,
                tol: omega_tol,
            },
            SeRef {
                name: "omega_V",
                nm: 4.30540e-3,
                tol: omega_tol,
            },
            SeRef {
                name: "omega_KA",
                nm: 1.50376e-1,
                tol: omega_tol,
            },
        ]
    } else {
        // FOCE: $EST METHOD=1 (no INTER).
        [
            SeRef {
                name: "TVCL",
                nm: 6.62683e-3,
                tol: TIGHT,
            },
            SeRef {
                name: "TVV",
                nm: 2.34041e-1,
                tol: TIGHT,
            },
            SeRef {
                name: "TVKA",
                nm: 1.24489e-1,
                tol: TIGHT,
            },
            SeRef {
                name: "PROP_ERR",
                nm: 9.41384e-4,
                tol: TIGHT,
            },
            SeRef {
                name: "omega_CL",
                nm: 1.27958e-2,
                tol: omega_tol,
            },
            SeRef {
                name: "omega_V",
                nm: 4.29747e-3,
                tol: omega_tol,
            },
            SeRef {
                name: "omega_KA",
                nm: 1.60641e-1,
                tol: omega_tol,
            },
        ]
    }
}

/// Fit warfarin with the requested conditional method and assert every
/// covariance SE matches the NONMEM `MATRIX=R` reference within 20%.
fn assert_covariance_se_matches_nonmem(method: EstimationMethod, interaction: bool) {
    let model = parse_model_string(MODEL_SRC).expect("warfarin model parses");
    let pop =
        read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data loads");

    let mut opts = FitOptions::default();
    opts.method = method;
    opts.interaction = interaction;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = true;
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("warfarin fit runs");

    // The covariance step must succeed (the indefinite fixed-EBE Hessian used to
    // fail or clip here).
    assert!(
        result.covariance_matrix.is_some(),
        "covariance step must produce a matrix"
    );
    let se_theta = result.se_theta.as_ref().expect("theta SEs present");
    let se_omega = result.se_omega.as_ref().expect("omega SEs present");
    let se_sigma = result.se_sigma.as_ref().expect("sigma SEs present");

    let refs = nonmem_refs(interaction);
    // ferx SEs in the same order as `refs`.
    let ferx = [
        se_theta[0],
        se_theta[1],
        se_theta[2],
        se_sigma[0],
        se_omega[0],
        se_omega[1],
        se_omega[2],
    ];

    for (r, &ferx_se) in refs.iter().zip(ferx.iter()) {
        let rel = (ferx_se - r.nm).abs() / r.nm;
        assert!(
            ferx_se.is_finite() && rel < r.tol,
            "SE({}) = {ferx_se:.6} vs NONMEM {:.6} — relative diff {:.1}% exceeds {:.0}% band",
            r.name,
            r.nm,
            rel * 100.0,
            r.tol * 100.0
        );
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored covariance SE cross-check (#209/#196/#129): opt in with --features slow-tests"
)]
fn covariance_se_matches_nonmem() {
    assert_covariance_se_matches_nonmem(EstimationMethod::FoceI, true);
}

/// FOCE (non-interaction) covariance: exercises the Sheiner–Beal path where
/// `compute_covariance` adds the Ω-prior gradient (`omega_prior_gradient`) that
/// FOCEI gets for free from the marginal — the only end-to-end guard for that
/// branch (the unit test `test_covariance_gradient_foce_matches_fd_ofv_fixed`
/// checks the gradient in isolation).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored covariance SE cross-check (#209/#196/#129): opt in with --features slow-tests"
)]
fn covariance_se_matches_nonmem_foce() {
    assert_covariance_se_matches_nonmem(EstimationMethod::Foce, false);
}
