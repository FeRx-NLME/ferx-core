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
//! Both methods take the covariance OFV as `2·pop_nll` (no separate omega-prior
//! add-back):
//!   - **FOCEI** (`interaction = true`): the Almquist–Laplace marginal carries
//!     `ηᵀΩ⁻¹η + log|Ω|` explicitly.
//!   - **FOCE** (`interaction = false`): the Sheiner–Beal marginal carries the Ω
//!     penalty via `R̃ = HΩHᵀ + R` (equivalent by Woodbury to the conditional form
//!     with `ηᵀΩ⁻¹η + log|Ω|`). An earlier covariance step added that prior again
//!     for FOCE, double-counting Ω and under-stating the FOCE omega SEs ~31%;
//!     removing the add-back is the fix in issue #243.
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
//! The omega-block band is 20% for both methods. ferx's FOCE estimates already
//! matched NONMEM FOCE (`METHOD=1`, no INTER): OFV −280.17 vs −280.36, θ within
//! ~1%. The only defect was the FOCE omega SEs sitting ~31% below NONMEM — a
//! covariance double-count of Ω (see above), not an objective gap. After the
//! issue #243 fix the FOCE omega SEs match NONMEM to ~3%, same as FOCEI.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::omega_se_at;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, Optimizer};
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
const OMEGA_FOCE: f64 = 0.20; // FOCE omega matches NONMEM after #243 cov fix

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
    assert!(
        model.residual_correlations.is_empty(),
        "Warfarin covariance NONMEM fixture must remain a diagonal-RUV anchor"
    );
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

/// FOCE (non-interaction) covariance: regression guard for issue #243 — the
/// FOCE omega SEs now match NONMEM to ~3% after removing the Ω double-count in
/// the covariance step (the unit test `test_covariance_gradient_foce_matches_fd_ofv_fixed`
/// checks the gradient in isolation).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored covariance SE cross-check (#209/#196/#129): opt in with --features slow-tests"
)]
fn covariance_se_matches_nonmem_foce() {
    assert_covariance_se_matches_nonmem(EstimationMethod::Foce, false);
}

// ── Block + diagonal omega covariance (structural-zero exclusion) ────────────

/// Warfarin with a 2×2 BLOCK omega on (CL,V) plus a separate diagonal omega on
/// KA. The cross-block elements ω(KA,CL) and ω(KA,V) are structural zeros
/// (`free_mask == false`): they are not estimated, so the covariance step must
/// exclude them from the free set. Before #243 the FOCE omega-prior add-back
/// iterated all lower-triangle entries and gave these a spurious curvature that
/// kept the Hessian non-singular; removing the add-back exposed that the
/// covariance step never excluded structural zeros, so the Hessian had flat
/// diagonals and the step failed. #243 excludes them explicitly (matching how
/// NONMEM omits non-estimated off-diagonals), fixing the step for **both** FOCE
/// and FOCEI (FOCEI had no add-back, so block+diagonal covariance failed on it
/// too).
const BLOCK_MODEL_SRC: &str = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method        = foce
  mu_referencing = true
";

/// FOCE covariance with a mixed block+diagonal omega — regression guard for the
/// #243 structural-zero exclusion. Asserts the step succeeds and the diagonal
/// SEs match NONMEM FOCE (`$EST METHOD=1`, no INTER, `$OMEGA BLOCK(2)` + diag),
/// `$COVARIANCE MATRIX=R`. For a block omega `se_omega` is the full
/// column-major lower triangle (#226), so the diagonal variance SEs are read
/// by `(i, i)` via `omega_se_at`; the CL–V covariance SE is exposed but not
/// asserted here.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored covariance SE cross-check (#209/#196/#129/#243): opt in with --features slow-tests"
)]
fn covariance_se_matches_nonmem_foce_block_omega() {
    let model = parse_model_string(BLOCK_MODEL_SRC).expect("block-omega model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin_block_omega.csv"), None, None)
        .expect("block-omega data loads");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Foce;
    opts.interaction = false;
    // Gradient outer optimizer (analytic Dual2 gradient): the derivative-free
    // BOBYQA default stalls on the weakly identified ω²(KA) direction, leaving
    // both ω²(KA) and its SE off NONMEM (#423); the gradient optimizer converges
    // it to the NONMEM-matching optimum where the SE cross-check holds. Do NOT
    // revert this to the BOBYQA default to "use the default" — BOBYQA is
    // known-divergent here (#423) and the SE cross-check below would no longer
    // reach the NONMEM optimum, silently losing the regression signal.
    opts.optimizer = Optimizer::Lbfgs;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = true;
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("block-omega fit runs");

    // The covariance step must SUCCEED — before #243 it failed here on the
    // structural-zero cross-block off-diagonals (flat Hessian diagonal).
    assert!(
        result.covariance_matrix.is_some(),
        "block+diagonal omega covariance step must produce a matrix"
    );
    let se_theta = result.se_theta.as_ref().expect("theta SEs present");
    let se_sigma = result.se_sigma.as_ref().expect("sigma SEs present");
    // Block omega ⇒ `se_omega` is the full column-major lower triangle
    // (len n·(n+1)/2, #226); read the diagonal variance SEs by `(i, i)`. Derive
    // `n_eta` from the fitted model (not a literal) so the column-major offset math
    // follows the model if its random-effect count ever changes — otherwise the
    // test would silently read the wrong offsets rather than failing loudly.
    let n_eta = result.eta_names.len();
    let omega_diag_se =
        |i: usize| omega_se_at(&result.se_omega, n_eta, i, i).expect("omega diagonal SE present");

    // NONMEM 7.5.1 FOCE (METHOD=1, no INTER), $OMEGA BLOCK(2) on (CL,V) + diag KA,
    // $COVARIANCE MATRIX=R; SEs from the .ext row at ITERATION = -1000000001.
    let refs = [
        SeRef {
            name: "TVCL",
            nm: 6.68028e-3,
            tol: TIGHT,
        },
        SeRef {
            name: "TVV",
            nm: 2.35936e-1,
            tol: TIGHT,
        },
        SeRef {
            name: "TVKA",
            nm: 1.24451e-1,
            tol: TIGHT,
        },
        SeRef {
            name: "PROP_ERR",
            nm: 9.46602e-4,
            tol: TIGHT,
        },
        SeRef {
            name: "omega_CL",
            nm: 1.27995e-2,
            tol: OMEGA_FOCE,
        },
        SeRef {
            name: "omega_V",
            nm: 4.29739e-3,
            tol: OMEGA_FOCE,
        },
        SeRef {
            name: "omega_KA",
            nm: 1.60542e-1,
            // The SE of the highest-shrinkage variance component is the single
            // hardest covariance quantity: it sits on the flattest curvature
            // direction, where ferx's FD R-matrix and NONMEM's MATRIX=R differ
            // ~25% (ferx 0.120 vs NONMEM 0.160 — within ferx's own R/S spread of
            // 0.112–0.185). Not a transform bug (audited); a flat-direction
            // FD-Hessian limitation tracked in #432. All other SEs match within
            // OMEGA_FOCE; this one component carries a wider band.
            tol: 0.30,
        },
    ];
    let ferx = [
        se_theta[0],
        se_theta[1],
        se_theta[2],
        se_sigma[0],
        omega_diag_se(0),
        omega_diag_se(1),
        omega_diag_se(2),
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

    // The omega_KA band above is one-sided in practice (NONMEM-relative, 30%): it
    // bites on a downward regression but would pass silently if ferx's SE drifted
    // *up* toward NONMEM's 0.160. Add a two-sided absolute window around ferx's
    // known-good 0.120 so the guard also catches ferx-internal regressions,
    // independent of the flat-direction gap to NONMEM tracked in #432.
    let se_omega_ka = omega_diag_se(2);
    assert!(
        (0.105..=0.135).contains(&se_omega_ka),
        "SE(ω²KA) = {se_omega_ka:.6} left ferx's known-good window [0.105, 0.135] \
         (regression independent of the #432 NONMEM gap)"
    );
}

// ── IOV covariance (the `is_iov` second-difference Hessian branch) ───────────

/// Warfarin with inter-occasion variability on CL (one κ per occasion sharing a
/// single variance). Mirrors `examples/warfarin_iov.ferx` and the NONMEM
/// reference `tests/nonmem/warfarin_iov.ctl`.
const IOV_MODEL_SRC: &str = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  iov_column = OCC
";

/// FOCEI covariance with IOV — the only end-to-end guard for the `is_iov` branch
/// of `compute_covariance`, which (lacking a fixed-EBE analytical κ gradient)
/// builds the Hessian from second differences of the reconverged objective
/// rather than the central-gradient stencil used for the non-IOV path.
///
/// NONMEM 7.5.1 FOCEI (`$EST METHOD=1 INTER`) `$COVARIANCE MATRIX=R` on
/// `data/warfarin_iov.csv` (10 subjects, 2 occasions), from `warfarin_iov.ctl`
/// with `MATRIX=R` added. SEs are the `.ext` row at `ITERATION = -1000000001`.
///
/// NONMEM reports OMEGA/SIGMA SEs on the variance scale; ferx's ω SEs are also
/// variance-scale (compared directly), but its PROP_ERR SE is SD-scale, so the
/// NONMEM `$SIGMA` variance SE is delta-method converted: with σ² = 0.0353876
/// and SE(σ²) = 0.00381432, SE(σ) = SE(σ²)/(2·σ) = 0.00381432/(2·0.188116) =
/// 1.0138e-2.
///
/// theta + residual-error + ω all match NONMEM within ~8% (20% band). The **κ**
/// (IOV) SE is held to a 40% band: with only two occasions the IOV variance is
/// weakly identified, so ferx (OFV 307.9) and NONMEM (308.8) settle at different
/// κ optima — the κ *estimate* differs ~27%, and its SE tracks that. This guards
/// that the IOV covariance step runs, succeeds, and lands in the NONMEM
/// neighbourhood; it is not a tight κ-SE anchor.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored covariance SE cross-check (#209/#196/#129): opt in with --features slow-tests"
)]
fn covariance_se_matches_nonmem_iov() {
    let model = parse_model_string(IOV_MODEL_SRC).expect("warfarin IOV model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data loads");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = true;
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("warfarin IOV fit runs");

    assert!(
        result.covariance_matrix.is_some(),
        "IOV covariance step must produce a matrix"
    );
    // This well-identified IOV fit has a PD Hessian, so the real covariance must
    // be returned — not the `covariance_fallback = sir` path (#223). Guards that
    // the SIR fallback stays inert when the covariance step succeeds.
    assert_eq!(
        result.covariance_status,
        ferx_core::CovarianceStatus::Computed,
        "well-identified IOV model must produce a real covariance, not a SIR fallback"
    );
    let se_theta = result.se_theta.as_ref().expect("theta SEs present");
    let se_omega = result.se_omega.as_ref().expect("omega SEs present");
    let se_sigma = result.se_sigma.as_ref().expect("sigma SEs present");
    let se_kappa = result.se_kappa.as_ref().expect("kappa SEs present");

    // NONMEM 7.5.1 FOCEI $COVARIANCE MATRIX=R SEs (.ext, ITER=-1000000001);
    // PROP_ERR converted from the variance-scale $SIGMA SE (see fn docs).
    let refs = [
        SeRef {
            name: "TVCL",
            nm: 1.33623e-2,
            tol: TIGHT,
        },
        SeRef {
            name: "TVV",
            nm: 3.41267e-1,
            tol: TIGHT,
        },
        SeRef {
            name: "TVKA",
            nm: 9.32221e-2,
            tol: TIGHT,
        },
        SeRef {
            name: "PROP_ERR",
            nm: 1.0138e-2,
            tol: TIGHT,
        },
        SeRef {
            name: "omega_CL",
            nm: 2.80628e-2,
            tol: TIGHT,
        },
        SeRef {
            name: "omega_V",
            nm: 6.44076e-3,
            tol: TIGHT,
        },
        SeRef {
            name: "omega_KA",
            nm: 2.79920e-2,
            tol: TIGHT,
        },
        SeRef {
            name: "kappa_CL",
            nm: 1.76108e-2,
            tol: 0.40,
        },
    ];
    let ferx = [
        se_theta[0],
        se_theta[1],
        se_theta[2],
        se_sigma[0],
        se_omega[0],
        se_omega[1],
        se_omega[2],
        se_kappa[0],
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
