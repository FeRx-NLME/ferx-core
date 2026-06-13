//! End-to-end checks for `covariance_method` (R / S / RSR) — issue #223.
//!
//! The estimator *math* is unit-tested in `outer_optimizer.rs`
//! (`test_combine_covariance_*`). These tests exercise the full `fit()` path:
//! that the per-subject score cross-product `S` assembles, that all three
//! estimators produce finite positive SEs, and that the FOCE guard fires.
//!
//! `covariance_se_matches_nonmem_s_rsr` anchors the `s` / `rsr` SEs to a
//! dedicated NONMEM 7.5.1 `$COV MATRIX=S` / `MATRIX=RSR` run (#266); the older
//! `covariance_methods_produce_consistent_ses_on_warfarin` keeps the
//! information-matrix sanity anchor (`R ≈ S` at the MLE) as a build-independent
//! cross-check that the estimators assemble at all.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{
    fit, read_nonmem_csv, CovarianceMethod, CovarianceStatus, EstimationMethod, FitOptions,
};
use std::path::Path;

const WARFARIN_FOCEI: &str = r"
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

fn warfarin_focei_opts(method: CovarianceMethod) -> FitOptions {
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = true;
    opts.covariance_method = method;
    opts.verbose = false;
    opts
}

/// All SEs (theta, omega, sigma) flattened, in a stable order.
fn all_ses(r: &ferx_core::FitResult) -> Vec<f64> {
    let mut v = Vec::new();
    v.extend(r.se_theta.as_ref().expect("theta SEs").iter().copied());
    v.extend(r.se_omega.as_ref().expect("omega SEs").iter().copied());
    v.extend(r.se_sigma.as_ref().expect("sigma SEs").iter().copied());
    v
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: full FOCEI fit ×3 for covariance_method R/S/RSR; opt in with --features slow-tests"
)]
fn covariance_methods_produce_consistent_ses_on_warfarin() {
    let model = parse_model_string(WARFARIN_FOCEI).expect("warfarin model parses");
    let pop =
        read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data loads");

    let run = |m: CovarianceMethod| {
        let r = fit(&model, &pop, &model.default_params, &warfarin_focei_opts(m))
            .unwrap_or_else(|e| panic!("{m:?} fit failed: {e}"));
        assert_eq!(
            r.covariance_status,
            CovarianceStatus::Computed,
            "{m:?}: covariance must be Computed"
        );
        let ses = all_ses(&r);
        assert!(
            ses.iter().all(|s| s.is_finite() && *s > 0.0),
            "{m:?}: all SEs must be finite and positive, got {ses:?}"
        );
        ses
    };

    let se_r = run(CovarianceMethod::Hessian);
    let se_s = run(CovarianceMethod::CrossProduct);
    let se_rsr = run(CovarianceMethod::Sandwich);

    // Information-matrix equality on a well-specified model: S ≈ R, so the S and
    // RSR SEs land within the same order of magnitude as the R SEs. A factor-of-3
    // band catches assembly bugs (transpose, mis-scaling, wrong free-block
    // embedding would be orders off) without being flaky on a 10-subject fit.
    for (name, se_alt) in [("s", &se_s), ("rsr", &se_rsr)] {
        for (k, (&r, &a)) in se_r.iter().zip(se_alt.iter()).enumerate() {
            let ratio = a / r;
            assert!(
                (0.33..=3.0).contains(&ratio),
                "covariance_method={name}: SE[{k}] = {a:.4e} differs from R-matrix {r:.4e} \
                 by ratio {ratio:.2} (outside [0.33, 3.0]) — likely an assembly bug"
            );
        }
    }
}

/// NONMEM-anchored `s` / `rsr` SE cross-check (#266).
///
/// Warfarin FOCEI (`$EST METHOD=1 INTER`) on `data/warfarin.csv`, two extra
/// covariance runs added to the existing `MATRIX=R` reference:
///   - `$COVARIANCE MATRIX=S`   → `S⁻¹` SEs (`covariance_method = s`)
///   - `$COVARIANCE MATRIX=RSR` → `R⁻¹SR⁻¹` SEs (`covariance_method = rsr`)
///
/// SEs are the `.ext` row at `ITERATION = -1000000001`, in the order
/// [TVCL, TVV, TVKA, PROP_SD, ωCL, ωV, ωKA]; ω are variance-scale, PROP_ERR is
/// the proportional-SD THETA(4) (SD-scale), matching ferx's `se_sigma[0]`.
///
/// Bands: the `s` cross-product is the noisiest estimator (a 10-subject
/// outer-product info matrix), so it gets the same 20% band as the R-matrix
/// reference (`warfarin_covariance_nonmem.rs`); on the `ci`/FD build ferx lands
/// within ~14%. The `rsr` sandwich tracks R closely (ferx within ~7%), so it is
/// held to 15%. A factor-of-2 error in the score scale would push the `s` SEs
/// ~29–41% off systematically — well outside these bands (issue #266 note).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored s/rsr covariance SE cross-check (#266): opt in with --features slow-tests"
)]
fn covariance_se_matches_nonmem_s_rsr() {
    let model = parse_model_string(WARFARIN_FOCEI).expect("warfarin model parses");
    let pop =
        read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data loads");

    // NONMEM 7.5.1 FOCEI SEs (.ext, ITER=-1000000001), order
    // [TVCL, TVV, TVKA, PROP_SD, ωCL, ωV, ωKA].
    let nm_s = [
        9.29659e-3, 4.60240e-1, 2.26835e-1, 1.54472e-3, 1.76426e-2, 8.28678e-3, 2.59564e-1,
    ];
    let nm_rsr = [
        7.09785e-3, 2.40313e-1, 1.48728e-1, 7.96181e-4, 1.09264e-2, 3.97787e-3, 1.39645e-1,
    ];
    let names = [
        "TVCL", "TVV", "TVKA", "PROP_ERR", "omega_CL", "omega_V", "omega_KA",
    ];

    let ferx_ses = |m: CovarianceMethod| {
        let r = fit(&model, &pop, &model.default_params, &warfarin_focei_opts(m))
            .unwrap_or_else(|e| panic!("{m:?} fit failed: {e}"));
        assert_eq!(
            r.covariance_status,
            CovarianceStatus::Computed,
            "{m:?}: covariance must be Computed"
        );
        let t = r.se_theta.expect("theta SEs");
        let om = r.se_omega.expect("omega SEs");
        let s = r.se_sigma.expect("sigma SEs");
        [t[0], t[1], t[2], s[0], om[0], om[1], om[2]]
    };

    for (method, nm, tol) in [
        (CovarianceMethod::CrossProduct, &nm_s, 0.20),
        (CovarianceMethod::Sandwich, &nm_rsr, 0.15),
    ] {
        let ferx = ferx_ses(method);
        for ((name, &f), &n) in names.iter().zip(ferx.iter()).zip(nm.iter()) {
            let rel = (f - n).abs() / n;
            assert!(
                f.is_finite() && rel < tol,
                "{method:?} SE({name}) = {f:.6e} vs NONMEM {n:.6e} — relative diff \
                 {:.1}% exceeds {:.0}% band",
                rel * 100.0,
                tol * 100.0
            );
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: runs a FOCE fit to exercise the covariance_method guard; opt in with --features slow-tests"
)]
fn covariance_method_rsr_rejects_foce_without_interaction() {
    // Sheiner–Beal FOCE (no interaction): the per-subject score omits the Ω prior,
    // so S would be inconsistent with R. The covariance step must refuse rather
    // than return a wrong matrix.
    let model = parse_model_string(WARFARIN_FOCEI).expect("warfarin model parses");
    let pop =
        read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data loads");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Foce;
    opts.interaction = false;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = true;
    opts.covariance_method = CovarianceMethod::Sandwich;
    opts.verbose = false;

    let r = fit(&model, &pop, &model.default_params, &opts).expect("FOCE fit runs");
    assert_eq!(
        r.covariance_status,
        CovarianceStatus::Failed,
        "rsr under non-interaction FOCE must fail the covariance step, not return a matrix"
    );
    assert!(
        r.warnings
            .iter()
            .any(|w| w.contains("not yet validated for FOCE")),
        "expected a `not yet validated for FOCE` covariance warning, got: {:?}",
        r.warnings
    );
}
