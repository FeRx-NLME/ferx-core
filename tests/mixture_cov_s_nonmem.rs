//! NONMEM `$COVARIANCE MATRIX=S` / default (`R⁻¹SR⁻¹`) anchor for a **mixture**
//! covariance step (#1956).
//!
//! Before #1956 the score cross-product `S` scored a mixture subject through the
//! single-population objective, under the default `MIXNUM = 1` and the base Ω/Σ: the
//! class-2-only θ and the mixing θ had identically-zero score columns, so
//! `covariance_method = s` was refused as singular and `rsr` silently lost both
//! coordinates. On `mixture_iv` at the NONMEM finals the old S was
//! `diag ≈ [2454, 0, 5930, 0]` with eigenvalues `[7975, 409, ~0, 0]` (macOS arm64).
//!
//! The model and data are the existing FOCEI mixture anchor's
//! (`tests/nonmem/mixture_iv_cov.ctl`, `mixture_iv.csv`): two classes differing in CL,
//! constant mixing, Ω and Σ FIXed, so the four free coordinates are TVCL1, TVCL2, V
//! and the mixing fraction. ferx spells the mixing fraction as NONMEM does —
//! `p(1) = P1`, a θ on the probability scale — so the SEs compare without a delta map.
//!
//! Both engines evaluate one fixed point, the `mixture_iv_cov.ext` finals at their
//! six printed significant figures: NONMEM with `MAXEVAL=0`, ferx with
//! `outer_maxiter = 0`. Three NONMEM runs share that point (`tests/nonmem/`):
//!
//! | stem                 | `$COVARIANCE`       | ferx `covariance_method` |
//! |----------------------|---------------------|--------------------------|
//! | `mixture_iv_cov_s`   | `MATRIX=S`          | `s`                      |
//! | `mixture_iv_cov_rsr` | default (`R⁻¹SR⁻¹`) | `rsr`                    |
//! | `mixture_iv_cov_r0`  | `MATRIX=R`          | `r` (control)            |
//!
//! The `MATRIX=R` control at the same point separates an `S` disagreement from an
//! `R` one: `rsr` inherits both, `s` only `S`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, CovarianceMethod, FitOptions, FitResult};
use std::path::Path;

/// `mixture_iv_cov.ext` `-1000000000` row (NONMEM 7.5.1, FOCEI), as printed.
const NM_TVCL1: f64 = 0.980833;
const NM_TVCL2: f64 = 2.84198;
const NM_TVV: f64 = 9.97859;
const NM_P1: f64 = 0.471970;

fn model_src() -> String {
    format!(
        r"
[parameters]
  theta TVCL1({NM_TVCL1}, 0.01, 100.0)
  theta TVCL2({NM_TVCL2}, 0.01, 100.0)
  theta TVV({NM_TVV}, 0.1, 1000.0)
  theta P1({NM_P1}, 0.001, 0.999)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  p(1) = P1

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
    )
}

fn eval_only_fit(method: CovarianceMethod) -> FitResult {
    let pop = read_nonmem_csv(
        Path::new("tests/nonmem/mixture_iv.csv"),
        Some(&["WT"]),
        None,
    )
    .unwrap();
    let model = parse_model_string(&model_src()).unwrap();
    let mut opts = FitOptions::default();
    opts.interaction = true; // NONMEM METHOD=1 INTER
    opts.outer_maxiter = 0; // NONMEM MAXEVAL=0
    opts.covariance_method = method;
    opts.covariance_method_set = true;
    fit(&model, &pop, &model.default_params, &opts).expect("eval-only mixture fit Ok")
}

fn theta_ses(res: &FitResult, label: &str) -> Vec<f64> {
    let se = res
        .se_theta
        .as_ref()
        .unwrap_or_else(|| panic!("{label}: no θ SEs; warnings: {:?}", res.warnings))
        .clone();
    assert_eq!(se.len(), 4, "{label}: four θ");
    for (j, s) in se.iter().enumerate() {
        assert!(
            s.is_finite() && *s > 0.0,
            "{label}: SE θ{} = {s}; warnings: {:?}",
            j + 1,
            res.warnings
        );
    }
    se
}

/// Every free θ carries information under `s` and `rsr` — the property #1956 broke.
/// Runs on every PR: one objective evaluation plus the covariance step on 30 subjects.
#[test]
fn mixture_s_and_rsr_report_every_theta_se() {
    for (method, label) in [
        (CovarianceMethod::CrossProduct, "s"),
        (CovarianceMethod::Sandwich, "rsr"),
    ] {
        let res = eval_only_fit(method);
        let se = theta_ses(&res, label);
        assert!(
            res.warnings.iter().all(|w| !w.contains("singular")),
            "{label}: {:?}",
            res.warnings
        );
        eprintln!("ferx {label} SEs (TVCL1, TVCL2, V, P1): {se:?}");
    }
}

/// The `-1000000001` (standard error) row of a NONMEM `.ext` file: θ1..θ4.
fn nonmem_theta_ses(stem: &str) -> Vec<f64> {
    let path = format!("tests/nonmem/{stem}.ext");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let row = text
        .lines()
        .find(|l| l.split_whitespace().next() == Some("-1000000001"))
        .unwrap_or_else(|| panic!("{path}: no -1000000001 (SE) row"));
    row.split_whitespace()
        .skip(1)
        .take(4)
        .map(|v| v.parse::<f64>().unwrap())
        .collect()
}

#[test]
#[ignore = "awaiting NONMEM output: tests/nonmem/mixture_iv_cov_{s,rsr,r0}.ext (#1956)"]
fn mixture_s_rsr_ses_match_nonmem() {
    // Provisional bound: the existing R anchor's (`mixture_nonmem.rs`, < 3 %). Replace
    // with the measured worst gap plus stated headroom once the outputs are committed.
    const TOL: f64 = 0.03;
    for (method, stem) in [
        (CovarianceMethod::Hessian, "mixture_iv_cov_r0"),
        (CovarianceMethod::CrossProduct, "mixture_iv_cov_s"),
        (CovarianceMethod::Sandwich, "mixture_iv_cov_rsr"),
    ] {
        let nm = nonmem_theta_ses(stem);
        let se = theta_ses(&eval_only_fit(method), stem);
        let mut worst = 0.0f64;
        for (j, (f, n)) in se.iter().zip(&nm).enumerate() {
            assert!(
                n.is_finite() && *n > 0.0,
                "{stem}: NONMEM SE θ{} = {n}",
                j + 1
            );
            let rel = (f - n).abs() / n;
            eprintln!(
                "{stem} θ{}: ferx {f:.6e} NONMEM {n:.6e} rel {rel:.3e}",
                j + 1
            );
            assert!(rel < TOL, "{stem} θ{}: ferx {f} vs NONMEM {n}", j + 1);
            worst = worst.max(rel);
        }
        eprintln!("{stem}: worst rel SE gap {worst:.3e}");
    }
}
