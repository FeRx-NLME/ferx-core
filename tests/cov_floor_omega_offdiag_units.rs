//! #2274: an off-diagonal of a block Ω's Cholesky factor is identity-packed, and `L_ij` has the
//! units of `η_i`. When `η_i` enters additively it carries the units of the parameter it is added
//! to, so the covariance step's eigenvalue floor — which #2145 made blind to an identity-packed
//! θ's units only — decided what to clip by those units. Here the volume carries an **additive**
//! η in units `k` times smaller, `V = (TVV_S + ETA_VS) / k`, in one `block_omega` with the
//! exponential `ETA_CL`: `L₂₁ ∝ k`, so its diagonal of `R` scales by `1/k²`, while every other
//! packed coordinate (log-θ, log `L_ii`, log σ) only shifts.
//!
//! The fixture straddles the floor at `k = 3e-4` and only there. `TH_WEAK` is a dimensionless,
//! log-packed θ the data barely inform (a `1e-3`-scaled perturbation of the allometric exponent,
//! centred on its optimum so the `R` is positive definite): its eigenvalue sits ~2.6e-8 below
//! `λ_max = 2.5e3` at `k = 1`, inside the floor's `1e-10`. At `k = 3e-4` the `L₂₁` diagonal alone
//! is ~1.0e7, `λ_max` follows it and the floor rises past `TH_WEAK`'s eigenvalue: with the
//! off-diagonal unscaled, `r` and `rsr` report a "regularized" step and `s` refuses `S` as
//! rank-deficient (measured by mutation, all three tests red). The large-`k` side cannot be
//! pushed over the floor: the packing rails (`|L_ij| ≤ 10`, `ln L_ii ∈ [−6, 6]`) cap `k` at ~17
//! here, where the `L₂₁` eigenvalue is still ~1e-6 of `λ_max`, so `k = 15` is an invariance check
//! only. The same rails cap the small side at `L_ii ≥ e⁻⁶`, which `k = 3e-4` respects
//! (`L₂₂ = 3.2e-3`).
//!
//! Every SE must equal the `k = 1` fit's, `SE(TVV_S)`, `SE(ω_CL,VS)` and `SE(ω_VS)` read on the
//! `k = 1` scale, and no regularization warning may be emitted. Evaluate-once fits
//! (`outer_maxiter = 0`): Tier 2, not gated.
use ferx_core::types::{CovarianceMethod, EstimationMethod, FitOptions, FitResult};
use ferx_core::{fit, read_nonmem_csv};
use std::path::Path;

// The FOCEI optimum of the `k = 1` model (converged, OFV −1198.0291), so the evaluate-once `R`
// is taken where it is positive definite.
const TVV: f64 = 50.670449;
const W_CL: f64 = 0.068897;
const W_VS: f64 = 116.938296;
const C_CL_VS: f64 = -0.147595;
const NAMES: [&str; 7] = [
    "TVCL",
    "TH_WEAK",
    "TVV_S/k",
    "omega_CL",
    "omega_CL,VS/k",
    "omega_VS/k^2",
    "PROP_ERR",
];

fn model_src(k: f64) -> String {
    let tvv_s = TVV * k;
    let w_vs = W_VS * k * k;
    let c = C_CL_VS * k;
    format!(
        "[parameters]
  theta TVCL(4.820644, 0.01, 100.0)
  theta TH_WEAK(1.000317, 0.01, 100.0)
  theta TVV_S({tvv_s:e}, {lo:e}, {hi:e})
  block_omega (ETA_CL, ETA_VS) = [{W_CL:e}, {c:e}, {w_vs:e}]
  sigma PROP_ERR ~ 0.098500 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70.0) ^ (0.9124 + 1e-3 * (TH_WEAK - 1.0)) * exp(ETA_CL)
  V  = (TVV_S + ETA_VS) / {k:e}
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
",
        lo = 0.01 * tvv_s,
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

/// SEs in [`NAMES`] order on the `k = 1` scale, after asserting the step neither failed nor
/// regularized.
fn ses(k: f64, cov: CovarianceMethod) -> Vec<f64> {
    let model = ferx_core::parser::model_parser::parse_model_string(&model_src(k)).unwrap();
    let pop = read_nonmem_csv(Path::new("data/covmuref_power.csv"), None, None).unwrap();
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
    let om = |i: usize, j: usize| ferx_core::types::omega_se_at(&r.se_omega, 2, i, j).unwrap();
    let mut v = vec![th[0], th[1], th[2] / k];
    v.extend([om(0, 0), om(1, 0) / k, om(1, 1) / (k * k)]);
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
fn r_is_not_regularized_for_an_additive_eta_in_large_or_small_units() {
    // Measured (macOS arm64): worst 8.9e-10 (k = 15), 3.1e-7 (k = 3e-4, the weak direction's
    // own conditioning); bound 1e-5, 30× headroom. The mutation is caught by the
    // warning assertion, not by this bound.
    assert_unit_invariant(CovarianceMethod::Hessian, &[15.0, 3e-4], 1e-5);
}

#[test]
fn s_is_not_refused_for_an_additive_eta_in_large_or_small_units() {
    // Measured (macOS arm64): worst 8.2e-9 (k = 15), 8.4e-7 (k = 3e-4), the FD score's noise;
    // bound 1e-5, 12× headroom. Before the fix `S` was refused at k = 3e-4.
    assert_unit_invariant(CovarianceMethod::CrossProduct, &[15.0, 3e-4], 1e-5);
}

#[test]
fn rsr_is_not_regularized_for_an_additive_eta_in_large_or_small_units() {
    // Measured (macOS arm64): worst 1.1e-8 (k = 15), 5.7e-7 (k = 3e-4); bound 1e-5, 17× headroom.
    assert_unit_invariant(CovarianceMethod::Sandwich, &[15.0, 3e-4], 1e-5);
}
