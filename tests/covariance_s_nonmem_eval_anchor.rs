//! NONMEM `$COVARIANCE MATRIX=S` anchors for the covariance step's score cross-product
//! `S = Σᵢ gᵢgᵢᵀ`, each engine evaluated at **one fixed point** (#2974).
//!
//! Since #2974 each row `gᵢ` is the subject's total derivative `dnllᵢ/dx`, the mode
//! re-estimated at every `x` — the subject's term of the exact adjoint outer gradient. That
//! is what NONMEM's `MATRIX=S` differentiates, so the two `S` are the same quantity and
//! `S⁻¹` is compared directly. `S` contains no `R`, so this is the like-for-like check;
//! `R⁻¹SR⁻¹` is reported against NONMEM's default `$COVARIANCE` too, and its gap is the `R`
//! one (ferx's `R` is the exact analytic Hessian, NONMEM's a finite difference).
//!
//! The point is NONMEM 7.6's own MLE (`nonmem_anchor/wf_*_est.ctl`, `NSIG=4`), printed to six
//! significant figures: NONMEM evaluates it with `MAXEVAL=0` (`wf_*_cov_{s,r0,rsr}.ctl`, outputs
//! in `nonmem_anchor/results/`), ferx
//! with `outer_maxiter = 0`. Both re-estimate the modes there. The proportional error is a θ
//! (`Y = IPRED·(1 + θ₄·ε)`, `$SIGMA 1 FIX`), so its SE is on the SD scale, as ferx's
//! `se_sigma` is; Ω and the IOV variance are on the variance scale in both.
//!
//! Measured worst relative SE gaps (Linux x86_64, the reference platform; an earlier Windows run
//! printed the same figures to the digits shown), and the bound each gets:
//!
//! | model          | `s` vs `MATRIX=S` | before #2974 | `r` vs `MATRIX=R` | `rsr` vs default |
//! |----------------|-------------------|--------------|-------------------|------------------|
//! | warfarin FOCEI | 2.6e-6            | 5.9e-2       | 1.8e-4            | 3.1e-4           |
//! | warfarin FOCE  | 2.3e-6            | 8.8e-4       | 8.7e-5            | 1.5e-4           |
//! | warfarin IOV   | 8.5e-6            | 2.5e-1       | 3.1e-3            | 6.2e-3           |
//!
//! `s` is bounded at 3e-5 (3.5× the IOV row; NONMEM prints six figures, ≤ 5e-6 of rounding),
//! which every pre-#2974 row exceeds by ≥ 29×, so the anchor fails on the fixed-mode score.
//! The `r` / `rsr` gaps are `R`'s — ferx's exact analytic Hessian against NONMEM's finite
//! difference — and get 5× (FOCE/FOCEI: 1e-3, 2e-3) and 3× (IOV: 1e-2, 2e-2) headroom.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, CovarianceMethod, EstimationMethod, FitOptions};
use std::path::Path;

const NAMES: [&str; 7] = ["TVCL", "TVV", "TVKA", "PROP_ERR", "w2CL", "w2V", "w2KA"];
const NAMES_IOV: [&str; 8] = ["TVCL", "TVV", "TVKA", "PROP_ERR", "w2CL", "w2V", "w2KA", "k2CL"];

fn warfarin_src(theta: [f64; 4], omega: [f64; 3], kappa: Option<f64>) -> String {
    let [cl, v, ka, prop] = theta;
    let [ocl, ov, oka] = omega;
    let (kap_line, kap_term, iov_opt) = match kappa {
        Some(k) => (format!("  kappa KAPPA_CL ~ {k:?}\n"), " + KAPPA_CL", "  iov_column = OCC\n"),
        None => (String::new(), "", ""),
    };
    format!(
        r"
[parameters]
  theta TVCL({cl:?}, 0.001, 10.0)
  theta TVV({v:?}, 0.1, 500.0)
  theta TVKA({ka:?}, 0.01, 50.0)
  omega ETA_CL ~ {ocl:?}
  omega ETA_V  ~ {ov:?}
  omega ETA_KA ~ {oka:?}
{kap_line}  sigma PROP_ERR ~ {prop:?} (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL{kap_term})
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
{iov_opt}"
    )
}

/// ferx's SEs at the model's initial values (`outer_maxiter = 0`), in `NAMES[_IOV]` order.
fn ferx_ses(src: &str, data: &str, iov: bool, focei: bool, method: CovarianceMethod) -> Vec<f64> {
    let model = parse_model_string(src).expect("model parses");
    let pop = read_nonmem_csv(Path::new(data), None, iov.then_some("OCC")).expect("data loads");
    let opts = FitOptions {
        method: if focei { EstimationMethod::FoceI } else { EstimationMethod::Foce },
        interaction: focei,
        outer_maxiter: 0,
        run_covariance_step: true,
        covariance_method: method,
        covariance_method_set: true,
        ..Default::default()
    };
    let r = fit(&model, &pop, &model.default_params, &opts).expect("eval-only fit");
    let t = r.se_theta.clone().unwrap_or_else(|| panic!("{method:?}: no SEs: {:?}", r.warnings));
    let om = r.se_omega.clone().expect("omega SEs");
    let sg = r.se_sigma.clone().expect("sigma SEs");
    let mut v = vec![t[0], t[1], t[2], sg[0], om[0], om[1], om[2]];
    if iov {
        v.push(r.se_kappa.clone().expect("kappa SEs")[0]);
    }
    v
}

/// NONMEM's `.ext` SE row (`-1000000001`), in `NAMES[_IOV]` order: θ1..θ4, then the Ω
/// diagonal (and the first IOV block).
fn nonmem_ses(stem: &str, iov: bool) -> Vec<f64> {
    let path = format!("nonmem_anchor/results/{stem}.ext");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let header: Vec<&str> = text
        .lines()
        .find(|l| l.trim_start().starts_with("ITERATION"))
        .expect("header")
        .split_whitespace()
        .collect();
    let row: Vec<f64> = text
        .lines()
        .find(|l| l.split_whitespace().next() == Some("-1000000001"))
        .unwrap_or_else(|| panic!("{path}: no SE row"))
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let mut cols =
        vec!["THETA1", "THETA2", "THETA3", "THETA4", "OMEGA(1,1)", "OMEGA(2,2)", "OMEGA(3,3)"];
    if iov {
        cols.push("OMEGA(4,4)");
    }
    cols.iter()
        .map(|c| row[header.iter().position(|h| h == c).unwrap_or_else(|| panic!("{c}"))])
        .collect()
}

fn worst_rel(label: &str, names: &[&str], got: &[f64], want: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for ((n, g), w) in names.iter().zip(got).zip(want) {
        assert!(g.is_finite() && *g > 0.0, "{label} {n}: {g}");
        let rel = (g - w).abs() / w;
        assert!(rel.is_finite(), "{label} {n}: gap {rel} (NONMEM {w})");
        eprintln!("{label} {n}: ferx {g:.6e} NONMEM {w:.6e} rel {rel:.3e}");
        worst = worst.max(rel);
    }
    eprintln!("{label}: worst rel {worst:.3e}");
    worst
}

const FOCEI_THETA: [f64; 4] = [0.132695, 7.73775, 0.8108, 0.0105651];
const FOCEI_OMEGA: [f64; 3] = [0.0285883, 0.00959254, 0.335859];
const FOCE_THETA: [f64; 4] = [0.132974, 7.73057, 0.725261, 0.0107481];
const FOCE_OMEGA: [f64; 3] = [0.0285973, 0.00957817, 0.348907];
const IOV_THETA: [f64; 4] = [0.172777, 8.62822, 1.17856, 0.188115];
const IOV_OMEGA: [f64; 3] = [0.039936, 0.0107779, 0.0254341];
const IOV_KAPPA: f64 = 0.0357079;

fn check(
    label: &str, src: &str, data: &str, iov: bool, focei: bool, stem: &str, s_tol: f64, r_tol: f64,
    rsr_tol: f64,
) {
    let names: &[&str] = if iov { &NAMES_IOV } else { &NAMES };
    for (method, suf, tol) in [
        (CovarianceMethod::CrossProduct, "s", s_tol),
        (CovarianceMethod::Hessian, "r0", r_tol),
        (CovarianceMethod::Sandwich, "rsr", rsr_tol),
    ] {
        let got = ferx_ses(src, data, iov, focei, method);
        let want = nonmem_ses(&format!("{stem}_cov_{suf}"), iov);
        let w = worst_rel(&format!("{label} {suf}"), names, &got, &want);
        assert!(w < tol, "{label} {suf}: worst rel {w:.3e} ≥ {tol:.0e}");
    }
}

#[test]
fn warfarin_focei_s_matches_nonmem_matrix_s() {
    check(
        "warfarin FOCEI",
        &warfarin_src(FOCEI_THETA, FOCEI_OMEGA, None),
        "data/warfarin.csv",
        false,
        true,
        "wf_focei",
        3e-5,
        1e-3,
        2e-3,
    );
}

#[test]
fn warfarin_foce_s_matches_nonmem_matrix_s() {
    check(
        "warfarin FOCE",
        &warfarin_src(FOCE_THETA, FOCE_OMEGA, None),
        "data/warfarin.csv",
        false,
        false,
        "wf_foce",
        3e-5,
        1e-3,
        2e-3,
    );
}

#[test]
fn warfarin_iov_s_matches_nonmem_matrix_s() {
    check(
        "warfarin IOV",
        &warfarin_src(IOV_THETA, IOV_OMEGA, Some(IOV_KAPPA)),
        "data/warfarin_iov.csv",
        true,
        true,
        "wf_iov",
        3e-5,
        1e-2,
        2e-2,
    );
}
