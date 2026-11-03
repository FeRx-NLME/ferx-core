//! #2145: the covariance step's eigenvalue floor was taken from the raw packed spectrum of `S`
//! or `R`, where an **identity-packed** θ (a negative lower bound) carries its own units. Writing
//! the warfarin volume in units `k` times smaller (`TVV_S = TVV·k`, `V = TVV_S / k`, bounds
//! `±100·TVV_S`) divides its row and column by `k`, and at `k ≥ 1e5` the floor clipped a
//! well-determined fit: `covariance_method = s` failed as "S singular or rank-deficient", and `r`
//! / `rsr` reported `SE(TVV_S)` up to 2000× too small under a "severe" regularization warning.
//! At small `k` it is the *other* parameters' eigenvalues that fall under a floor set by
//! `TVV_S`'s row, and TVKA's SE was understated 2.3× (`r`) and 5.4× (`rsr`) at `k = 3e-6`.
//!
//! Every SE must equal the `k = 1` fit's, `SE(TVV_S)` read as `SE(TVV_S) / k`, and no
//! regularization warning may be emitted. Evaluate-once fits (`outer_maxiter = 0`): Tier 2,
//! not gated.
use ferx_core::types::{CovarianceMethod, EstimationMethod, FitOptions, FitResult};
use ferx_core::{fit, read_nonmem_csv};
use std::path::Path;

const TVV: f64 = 7.737706;
const NAMES: [&str; 7] = [
    "TVCL", "TVV_S/k", "TVKA", "ETA_CL", "ETA_V", "ETA_KA", "PROP_ERR",
];

fn model_src(k: f64) -> String {
    let tvv_s = TVV * k;
    format!(
        "[parameters]
  theta TVCL(0.132695, 0.001, 10.0)
  theta TVV_S({tvv_s:e}, {lo:e}, {hi:e})
  theta TVKA(0.810796, 0.01, 50.0)
  omega ETA_CL ~ 0.028590
  omega ETA_V  ~ 0.009592
  omega ETA_KA ~ 0.335871
  sigma PROP_ERR ~ 0.010565 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV_S / {k:e} * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
",
        lo = -100.0 * tvv_s,
        hi = 100.0 * tvv_s,
    )
}

fn opts(cov: CovarianceMethod) -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        outer_maxiter: 0,
        run_covariance_step: true,
        covariance_method: cov,
        covariance_method_set: true,
        verbose: false,
        ..FitOptions::default()
    }
}

/// SEs in [`NAMES`] order, after asserting the step neither failed nor regularized.
fn ses(k: f64, cov: CovarianceMethod) -> Vec<f64> {
    let model = ferx_core::parser::model_parser::parse_model_string(&model_src(k)).unwrap();
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).unwrap();
    let r: FitResult = fit(&model, &pop, &model.default_params, &opts(cov)).unwrap();
    let th = r
        .se_theta
        .clone()
        .unwrap_or_else(|| panic!("{cov:?} k = {k}: no SEs; warnings: {:?}", r.warnings));
    assert!(
        !r.warnings.iter().any(|w| w.contains("regularized")),
        "{cov:?} k = {k}: the floor fired; warnings: {:?}",
        r.warnings
    );
    let mut v = vec![th[0], th[1] / k, th[2]];
    for i in 0..3 {
        v.push(ferx_core::types::omega_se_at(&r.se_omega, 3, i, i).unwrap());
    }
    v.extend(r.se_sigma.as_ref().unwrap());
    v
}

/// The worst relative gap of every SE at each `k` against `k = 1`, asserted under `tol`.
fn assert_unit_invariant(cov: CovarianceMethod, ks: &[f64], tol: f64) {
    let reference = ses(1.0, cov);
    assert!(
        reference.iter().all(|s| s.is_finite() && *s > 0.0),
        "{cov:?}: reference SEs {reference:?}"
    );
    for &k in ks {
        let got = ses(k, cov);
        // Every SE checked for finiteness before the fold, which `f64::max` would let a `NaN`
        // through.
        let mut worst = 0.0_f64;
        for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
            assert!(g.is_finite(), "{cov:?} k = {k}: SE({}) = {g}", NAMES[i]);
            let rel = (g - r).abs() / r;
            eprintln!(
                "{cov:?} k = {k:e}: SE({}) = {g:.6e} vs k = 1 {r:.6e}, rel {rel:.2e}",
                NAMES[i]
            );
            worst = worst.max(rel);
        }
        eprintln!("{cov:?} k = {k:e}: worst rel {worst:.3e}");
        assert!(worst < tol, "{cov:?} k = {k}: worst rel {worst}");
    }
}

#[test]
fn s_is_not_refused_for_a_large_unit_identity_theta() {
    // Measured (macOS arm64): worst 1.3e-10 (k = 1e5) and 2.3e-10 (k = 1e6), the score's own FD
    // noise; bound 1e-8, 40× headroom. Before the fix both refused S as singular.
    assert_unit_invariant(CovarianceMethod::CrossProduct, &[1e5, 1e6], 1e-8);
}

#[test]
fn r_is_not_regularized_for_an_identity_theta_in_large_or_small_units() {
    // Measured (macOS arm64): worst 5.2e-16 at every k — the analytic R is exact in the packed
    // coordinate, so only the inversion rounds; bound 1e-8 (shared with the FD-scored `s`).
    // Before the fix the floor fired at every k: SE(TVV_S)/k 0.0535 / 0.00535 against 0.240 at
    // k = 1e5 / 1e6 (#2145's table), SE(TVKA) 0.064 against 0.149 at k = 3e-6.
    assert_unit_invariant(CovarianceMethod::Hessian, &[1e5, 1e6, 3e-6], 1e-8);
}

#[test]
fn rsr_is_not_regularized_for_an_identity_theta_in_large_or_small_units() {
    // Measured (macOS arm64): worst 5.2e-12 (k = 1e5), 7.8e-11 (k = 1e6), 7.2e-11 (k = 3e-6),
    // from S's FD score; bound 1e-8, 100× headroom. Before the fix the floor fired at every k:
    // SE(TVV_S)/k 0.0118 / 1.2e-4 against 0.238 (#2145), SE(TVKA) 0.028 against 0.152 at k = 3e-6.
    assert_unit_invariant(CovarianceMethod::Sandwich, &[1e5, 1e6, 3e-6], 1e-8);
}

/// NONMEM, by reduction to the #432 anchor (`tests/covariance_se_at_nonmem_minimum.rs`): the
/// block-Ω warfarin model at NONMEM 7.5.1's own FOCEI minimum against its `$COVARIANCE
/// MATRIX=R` SEs, with `TVV` written as the identity-packed `TVV_S = 1e5·TVV` and the default
/// (analytic) R. Rescaling a θ multiplies its SE by the same factor and leaves the others alone,
/// so the anchor's numbers apply with `SE(TVV) = SE(TVV_S) / 1e5`. Before #2145 the floor
/// clipped `TVV_S`'s eigenvalue here and `SE(TVV)` came back 4.5× too small.
#[test]
fn r_on_a_large_unit_volume_matches_nonmem_matrix_r() {
    const K: f64 = 1e5;
    let tvv_s = 7.73752 * K;
    let src = format!(
        "[parameters]
  theta TVCL(0.132696, 0.001, 10.0)
  theta TVV_S({tvv_s:e}, {lo:e}, {hi:e})
  theta TVKA(0.810794, 0.01, 50.0)
  block_omega (ETA_CL, ETA_V) = [0.0285938, 0.00190087, 0.00959304]
  omega ETA_KA ~ 0.335846
  sigma PROP_ERR ~ 0.010565 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV_S / {K:e} * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
",
        lo = -100.0 * tvv_s,
        hi = 100.0 * tvv_s,
    );
    let model = ferx_core::parser::model_parser::parse_model_string(&src).unwrap();
    let pop = read_nonmem_csv(Path::new("data/warfarin_block_omega.csv"), None, None).unwrap();
    let opts = FitOptions {
        mu_referencing: true,
        inner_tol: 1e-8,
        inner_maxiter: 500,
        ..opts(CovarianceMethod::Hessian)
    };
    let r = fit(&model, &pop, &model.default_params, &opts).unwrap();
    let se_theta = r
        .se_theta
        .as_ref()
        .unwrap_or_else(|| panic!("no SEs; warnings: {:?}", r.warnings));
    assert!(
        !r.warnings.iter().any(|w| w.contains("regularized")),
        "the floor fired; warnings: {:?}",
        r.warnings
    );
    let se_sigma = r.se_sigma.as_ref().unwrap();
    let omega_se = |i: usize| ferx_core::types::omega_se_at(&r.se_omega, 3, i, i).unwrap();
    // NONMEM 7.5.1 FOCEI MATRIX=R SE row at this minimum, as in the #432 test.
    let checks: [(&str, f64, f64); 7] = [
        ("TVCL", se_theta[0], 7.09762e-3),
        ("TVV", se_theta[1] / K, 2.40059e-1),
        ("TVKA", se_theta[2], 1.48631e-1),
        ("PROP_ERR", se_sigma[0], 8.35394e-4),
        ("omega_CL", omega_se(0), 1.27953e-2),
        ("omega_V", omega_se(1), 4.30584e-3),
        ("omega_KA", omega_se(2), 1.50360e-1),
    ];
    let mut worst = 0.0_f64;
    for (name, se, nm) in checks {
        assert!(se.is_finite(), "SE({name}) = {se}");
        let rel = (se - nm).abs() / nm;
        eprintln!("NONMEM reduction: SE({name}) = {se:.6e} vs {nm:.6e}, rel {rel:.2e}");
        worst = worst.max(rel);
    }
    eprintln!("NONMEM reduction: worst rel {worst:.3e}");
    // Measured (macOS arm64): worst 1.8e-5 (TVKA), TVV 2.4e-6 — the #432 point on the analytic
    // R; bound 2e-4, 10× headroom. Before the fix the floor fired on TVV_S's eigenvalue.
    assert!(worst < 2e-4, "worst rel {worst}");
}
